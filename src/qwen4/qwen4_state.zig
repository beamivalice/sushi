//! What a Flash-Next model owns beyond the trunk: the n-gram hash and table, the final hyper-connection mixer, the MTP
//! head and the QSA scratch. Loaded after the trunk's own weights, freed before them.
const std = @import("std");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const model_mod = @import("../model.zig");
const layer_weights = @import("../layer_weights.zig");
const trunk = @import("../trunk.zig");
const qwen4_mod = @import("qwen4_exp.zig");
const qwen4_forward = @import("qwen4_forward.zig");
const qwen4_qsa_state = @import("qwen4_qsa_state.zig");
const qwen4_qsa_select = @import("qwen4_qsa_select.zig");
const qwen4_qsa_attn = @import("qwen4_qsa_attn.zig");
const qwen4_hc = @import("qwen4_hc.zig");
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;
const Trunk = trunk.Trunk;

/// The model-lifetime state. Until it replaces the trunk's `qwen4*`, `qsa_*`, `gdn_*` and `compiled_hc_*` fields, each
/// member is the type of the trunk field it is named for.
pub const State = struct {
    model: ?*qwen4_mod.Qwen4State = null,
    stream_f32: bool = false,
    mixer: ?layer_weights.HcWeights = null,
    qsa_consts: qwen4_qsa_select.QsaBlockConsts = .{},
    qsa_pooled_rope: qwen4_qsa_select.QsaPooledRope = .{},
    qsa_score_bank_builds: usize = 0,
    mtp: ?qwen4_forward.Qwen4Mtp = null,
    mtp_owner: ?*qwen4_forward.Qwen4MtpState = null,
    gdn_ones_w: ?mlx.mlx_array = null,
    gdn_q_scale: ?mlx.mlx_array = null,
    gdn_k_scale: ?mlx.mlx_array = null,
    gdn_eps: ?mlx.mlx_array = null,
    compiled_gdn_gate: ?mlx.mlx_closure = null,
    compiled_hc_silu: ?mlx.mlx_closure = null,
    compiled_hc_mix: ?mlx.mlx_closure = null,
    compiled_hc_inj: ?mlx.mlx_closure = null,
    compiled_hc_write: ?mlx.mlx_closure = null,

    /// Frees what `load` made and the QSA scratch; the trunk's teardown still frees the closures and GDN constants.
    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        release(allocator, &self.qsa_consts, &self.qsa_pooled_rope, &self.mtp, &self.model);
    }
};

/// What a forward context carries for Flash-Next: M-RoPE tables, the QSA selection of the layer in flight and the
/// deferred PLE rows.
pub const Ctx = struct {
    mrope_pos: ?[]const i32 = null,
    mrope_total: usize = 0,
    mrope_delta: i32 = 0,
    mrope_cos_cur: ?mlx.mlx_array = null,
    mrope_sin_cur: ?mlx.mlx_array = null,
    qsa_mask: mlx.mlx_array = .{ .ctx = null },
    qsa_blocks: mlx.mlx_array = .{ .ctx = null },
    qsa_arms: qwen4_qsa_attn.QsaArmCounts = .{},
    qsa_entry: ?*trunk.SSMCacheEntry = null,
    qsa_pos_base: c_int = 0,
    ple_defer: bool = false,
    ple_pending: ?qwen4_hc.PlePending = null,
    capture_stream_all: ?*mlx.mlx_array = null,
};

pub const Loaded = struct {
    state: *qwen4_mod.Qwen4State,
    mixer: layer_weights.HcWeights,
    mtp: ?qwen4_forward.Qwen4Mtp,
    stream: ?Trunk.ExpertStreamParts,
};

/// `owned_bf16` takes the mixer's pre-transposed arrays so the trunk frees them with its own.
pub fn load(
    allocator: std.mem.Allocator,
    config: ModelConfig,
    weights: *Weights,
    prefix: []const u8,
    name_buf: *[256]u8,
    s: mlx.mlx_stream,
    owned_bf16: *?[]mlx.mlx_array,
) !Loaded {
    var extra: std.ArrayList(mlx.mlx_array) = .empty;
    defer extra.deinit(allocator);
    var mb: [160]u8 = undefined;
    var mixer = try layer_weights.loadHcWeights(weights, std.fmt.bufPrint(&mb, "{s}.hyper_connection_mixer", .{prefix}) catch unreachable, false, config.hc_count, config.hidden_size, &extra, allocator, s);
    try layer_weights.bindLinears(&mixer, &config);
    if (extra.items.len > 0) {
        const old = owned_bf16.* orelse &[_]mlx.mlx_array{};
        const merged = try std.mem.concat(allocator, mlx.mlx_array, &.{ old, extra.items });
        if (owned_bf16.*) |o| allocator.free(o);
        owned_bf16.* = merged;
    }
    const eos: u32 = config.ngram_eos;
    // The `0` is the PLE's ordinal among the config's injection points, not
    // `config.ple_layer_idx` (pinned by the oracle fixture with `ple_layer_ids=[2]`).
    try qwen4_qsa_state.checkQsaCompressRatio(config.indexer_compress_ratio);
    const st = try allocator.create(qwen4_mod.Qwen4State);
    {
        errdefer allocator.destroy(st);
        st.* = .{
            .hash = try qwen4_mod.NgramHash.init(config.vocab_size, config.ngram_size, config.heads_per_ngram, config.ngram_vocab_base, config.ngram_vocab_divisor, config.ngram_seed, 0, eos),
            .table = switch (config.ngramTableSource()) {
                .bf16_override => blk: {
                    const dir = config.ngram_bf16_dir.?;
                    log.info("[qwen4] ngram table from the bf16 shards in {s} (SUSHI_NGRAM_BF16_DIR)\n", .{dir});
                    break :blk try qwen4_mod.NgramTable.openBf16(allocator, dir);
                },
                .bf16_streamed => try qwen4_mod.NgramTable.openBf16(allocator, config.expert_source_dir orelse return error.MissingExpertSourceDir),
                .quantized => try qwen4_mod.NgramTable.open(config.ngram_table_path orelse return error.MissingNgramTable),
            },
        };
    }
    // Once published the state may run its table's warm thread, which `deinit` joins.
    errdefer {
        st.deinit();
        allocator.destroy(st);
    }
    if (st.table.rows != st.hash.total_rows or st.table.dim * st.hash.n_heads != config.ple_embed_dim) {
        log.err("[qwen4] ngram_table.bin geometry {d}x{d} does not match the config ({d} rows, {d} heads x dim)\n", .{ st.table.rows, st.table.dim, st.hash.total_rows, st.hash.n_heads });
        return error.NgramTableMismatch;
    }
    var gpu_bytes: usize = 0;
    _ = mlx.mlx_get_active_memory(&gpu_bytes);
    st.table.calibrateArm();
    st.table.startWarm(); // the weights load just evicted the table from page cache
    var mtp = try qwen4_forward.loadQwen4Mtp(allocator, config, weights, name_buf, s);
    errdefer if (mtp) |*m| m.deinit(allocator);
    if (config.expert_streaming) {
        if (mtp) |*mtp_head| try qwen4_forward.evalQwen4MtpResident(mtp_head);
    }
    var stream: ?Trunk.ExpertStreamParts = null;
    errdefer if (stream) |parts| {
        if (parts.imatrix) |c| c.deinit();
        parts.engine.deinit();
        allocator.destroy(parts.engine);
    };
    if (config.expert_streaming) stream = try Trunk.initExpertStream(allocator, &config, s);
    log.info("[qwen4] n-gram table {d} rows x {d} ({d}-bit, {s}), PLE at layer {d}, QSA budget {d}/{d}\n", .{ st.table.rows, st.table.dim, st.table.bits, if (config.expert_streaming and st.table.bits == 16) "sharded pread" else "mmapped", config.ple_layer_idx, config.indexer_budget, config.indexer_compress_ratio });
    return .{ .state = st, .mixer = mixer, .mtp = mtp, .stream = stream };
}

fn release(
    allocator: std.mem.Allocator,
    consts: *qwen4_qsa_select.QsaBlockConsts,
    pooled_rope: *qwen4_qsa_select.QsaPooledRope,
    mtp: *?qwen4_forward.Qwen4Mtp,
    model: *?*qwen4_mod.Qwen4State,
) void {
    consts.deinit();
    pooled_rope.deinit();
    if (mtp.*) |*m| {
        m.deinit(allocator);
        mtp.* = null;
    }
    if (model.*) |st| {
        st.deinit();
        allocator.destroy(st);
        model.* = null;
    }
}

/// Frees what `load` made, and the QSA and persistent-group scratch the forward grew beside it.
pub fn deinit(self: *Trunk) void {
    release(self.allocator, &self.qsa_consts, &self.qsa_pooled_rope, &self.qwen4_mtp, &self.qwen4);
    self.ssm_group.deinit();
}

test "Flash-Next state: an empty state releases as a no-op, twice" {
    var state: State = .{};
    state.deinit(std.testing.allocator);
    state.deinit(std.testing.allocator);
    try std.testing.expect(state.model == null and state.mtp == null);
}

test "Flash-Next state: State holds the types the trunk's Flash-Next fields hold today" {
    const pairs = .{
        .{ "model", "qwen4" },
        .{ "stream_f32", "qwen4_stream_f32" },
        .{ "mixer", "qwen4_mixer" },
        .{ "qsa_consts", "qsa_consts" },
        .{ "qsa_pooled_rope", "qsa_pooled_rope" },
        .{ "qsa_score_bank_builds", "qsa_score_bank_builds" },
        .{ "mtp", "qwen4_mtp" },
        .{ "mtp_owner", "qwen4_mtp_owner" },
        .{ "gdn_ones_w", "gdn_ones_w" },
        .{ "gdn_q_scale", "gdn_q_scale" },
        .{ "gdn_k_scale", "gdn_k_scale" },
        .{ "gdn_eps", "gdn_eps" },
        .{ "compiled_gdn_gate", "compiled_gdn_gate" },
        .{ "compiled_hc_silu", "compiled_hc_silu" },
        .{ "compiled_hc_mix", "compiled_hc_mix" },
        .{ "compiled_hc_inj", "compiled_hc_inj" },
        .{ "compiled_hc_write", "compiled_hc_write" },
    };
    inline for (pairs) |pair| try std.testing.expect(@FieldType(State, pair[0]) == @FieldType(Trunk, pair[1]));
    try std.testing.expectEqual(pairs.len, std.meta.fieldNames(State).len);
}

test "Flash-Next state: Ctx holds the types the forward context's Flash-Next fields hold today" {
    const testing = std.testing;
    const names = comptime std.meta.fieldNames(Ctx);
    inline for (names) |name| try testing.expect(@FieldType(Ctx, name) == @FieldType(trunk.ForwardCtx, name));
    try testing.expectEqual(@as(usize, 13), names.len);
    const ctx: Ctx = .{};
    try testing.expect(ctx.mrope_pos == null and ctx.mrope_total == 0 and ctx.mrope_delta == 0);
    try testing.expect(ctx.qsa_mask.ctx == null and ctx.qsa_blocks.ctx == null and ctx.qsa_entry == null);
    try testing.expect(ctx.qsa_pos_base == 0 and !ctx.ple_defer and ctx.ple_pending == null);
}
