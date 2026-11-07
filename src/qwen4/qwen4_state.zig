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
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;
const Trunk = trunk.Trunk;

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
    const mixer = try layer_weights.loadHcWeights(weights, std.fmt.bufPrint(&mb, "{s}.hyper_connection_mixer", .{prefix}) catch unreachable, false, config.hc_count, config.hidden_size, &extra, allocator, s);
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

/// Frees what `load` made, and the QSA and persistent-group scratch the forward grew beside it.
pub fn deinit(self: *Trunk) void {
    self.qsa_consts.deinit();
    self.qsa_pooled_rope.deinit();
    self.ssm_group.deinit();
    if (self.qwen4_mtp) |*m| {
        m.deinit(self.allocator);
        self.qwen4_mtp = null;
    }
    if (self.qwen4) |st| {
        st.deinit();
        self.allocator.destroy(st);
        self.qwen4 = null;
    }
}
