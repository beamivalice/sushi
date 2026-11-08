//! What a Flash-Next model owns beyond the trunk: the n-gram hash and table, the final hyper-connection mixer, the MTP
//! head, the QSA scratch, the GDN constants and the persistent SSM group. It lives in `Trunk.arch_state.qwen4`
//! (`arch/arch_state.zig`), loaded after the trunk's own weights and freed before them.
const std = @import("std");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const mrope_mod = @import("../mrope.zig");
const layer_weights = @import("../layer_weights.zig");
const trunk = @import("../trunk.zig");
const arch_state = @import("../arch/arch_state.zig");
const qwen4_mod = @import("qwen4_exp.zig");
const qwen4_forward = @import("qwen4_forward.zig");
const qwen4_qsa_state = @import("qwen4_qsa_state.zig");
const qwen4_qsa_select = @import("qwen4_qsa_select.zig");
const qwen4_qsa_attn = @import("qwen4_qsa_attn.zig");
const qwen4_hc = @import("qwen4_hc.zig");
const qwen4_qsa = @import("qwen4_qsa.zig");
const qwen4_ssm_group = @import("qwen4_ssm_group.zig");
const Trunk = trunk.Trunk;

/// The model-lifetime state.
pub const State = struct {
    /// The n-gram hash and mmapped table; non-null exactly when a Flash-Next model is loaded.
    model: ?*qwen4_mod.Qwen4State = null,
    /// Run the residual streams in f32 (set by the oracle test): the checkpoint is bf16 end to end, the f32 fixture
    /// is the tighter bar for the math.
    stream_f32: bool = false,
    /// The final hyper-connection read that replaces `model.norm`.
    mixer: ?layer_weights.HcWeights = null,
    /// nb-keyed QSA block constants.
    qsa_consts: qwen4_qsa_select.QsaBlockConsts = .{},
    /// QSA pooled-block cos/sin, built once per forward.
    qsa_pooled_rope: qwen4_qsa_select.QsaPooledRope = .{},
    /// Test meter: how many times `qsaScoreBank` built or extended the transposed f32 operand.
    qsa_score_bank_builds: usize = 0,
    /// The MTP head, loaded when the pack carries `mtp.*`.
    mtp: ?qwen4_forward.Qwen4Mtp = null,
    /// Whose per-request state the head currently holds (`qwen4MtpActivate`).
    mtp_owner: ?*qwen4_forward.Qwen4MtpState = null,
    /// GatedDeltaNet decode constants, built on first use: ones([dk]) for the parameter-free rms_norm, the bf16
    /// scalars 1/dk and 1/sqrt(dk), and the f32 `rms_norm_eps` of the fused norm-gate kernel.
    gdn_ones_w: ?mlx.mlx_array = null,
    gdn_q_scale: ?mlx.mlx_array = null,
    gdn_k_scale: ?mlx.mlx_array = null,
    gdn_eps: ?mlx.mlx_array = null,
    /// exp(-exp(A_log)·softplus(a+dt_bias)) as one kernel.
    compiled_gdn_gate: ?mlx.mlx_closure = null,
    /// The hyper-connection elementwise tails, one kernel each: silu(down), mean(σ(up)·normed), 2σ(inject),
    /// stream + out·inj.
    compiled_hc_silu: ?mlx.mlx_closure = null,
    compiled_hc_mix: ?mlx.mlx_closure = null,
    compiled_hc_inj: ?mlx.mlx_closure = null,
    compiled_hc_write: ?mlx.mlx_closure = null,
    ssm_group: trunk.PersistentSsmGroup = .{},
    /// The batched decode's async-eval layer stride (0 = off); serial decode uses the trunk's `decode_async_ladder`.
    decode_async_ladder: u32 = 0,

    /// The n-gram hash and table are read-only. Every per-request thing lives on the slot's `SSMCacheEntry` (GDN pair,
    /// PLE window and tokens, QSA keys and pooled blocks), so slots interleave and batch; the MTP head's cache is the
    /// one model-owned piece, and the scheduler makes the slot that uses it exclusive (`slotExclusiveDecode`).
    pub fn sharesReadonly(self: *const State) bool {
        return self.model != null;
    }

    /// GDN per-position capture, the PLE window and token capture and the positional QSA key histories
    /// (`ssmRollbackFromCapture`) rewind a rejected tail, so the MTP head's verify can roll back.
    pub fn specRollsBack(self: *const State) bool {
        return self.mtp != null;
    }

    pub fn compile(t: *Trunk, set: arch_state.CompileSet) void {
        if (!isLoaded(t)) return;
        if (t.config.linear_num_key_heads > 0) qwen4_forward.compileGdnGate(t);
        if (set == .all) qwen4_forward.compileQwen4Hc(t);
    }

    pub const applyHeadKvQuant = qwen4_forward.qwen4MtpApplyKvQuant;
    pub const buildDraftRerank = qwen4_forward.qwen4BuildDraftRerank;
    pub const draftShortlists = qwen4_forward.qwen4DraftShortlistsBatched;
    pub const markHeadCheckpoint = qwen4_forward.qwen4MtpMarkQsaLeftover;
    pub const markTablesStale = qwen4_qsa.markQsaPooledRopeStale;
    pub const resetTables = qwen4_qsa.resetQsaPooledRope;
    pub const releaseGroup = qwen4_ssm_group.ssmGroupRelease;
    pub const dropGroup = qwen4_ssm_group.ssmGroupDrop;
    pub const flushDeferred = qwen4_hc.flushDeferredPle;
    pub const flushDeferredGroup = qwen4_hc.flushDeferredPleGroup;
    pub const discardDeferred = qwen4_hc.discardDeferredPle;

    pub fn init(self: *State, args: arch_state.InitArgs) !?Trunk.ExpertStreamParts {
        return load(self, args);
    }

    /// Frees the compiled closures; the forward falls back to the unfused ops until they are compiled again.
    pub fn clearCompiled(self: *State) void {
        const closures = .{ &self.compiled_gdn_gate, &self.compiled_hc_silu, &self.compiled_hc_mix, &self.compiled_hc_inj, &self.compiled_hc_write };
        inline for (closures) |c| {
            if (c.*) |closure| _ = mlx.mlx_closure_free(closure);
            c.* = null;
        }
    }

    /// Frees what `init` made and what the forward grew beside it; safe to call twice.
    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        self.qsa_consts.deinit();
        self.qsa_pooled_rope.deinit();
        if (self.mtp) |*m| m.deinit(allocator);
        self.mtp = null;
        if (self.model) |st| {
            st.deinit();
            allocator.destroy(st);
            self.model = null;
        }
        self.clearCompiled();
        inline for (.{ &self.gdn_ones_w, &self.gdn_q_scale, &self.gdn_k_scale, &self.gdn_eps }) |a| {
            if (a.*) |arr| _ = mlx.mlx_array_free(arr);
            a.* = null;
        }
        self.ssm_group.deinit();
    }
};

/// Whether this build serves Flash-Next.
pub const built = @hasField(arch_state.State, "qwen4");

/// Flash-Next's state on `t`; null in a build without qwen4.
pub fn stateOf(t: *Trunk) ?*State {
    return if (comptime built) &t.arch_state.qwen4 else null;
}

pub fn stateOfConst(t: *const Trunk) ?*const State {
    return if (comptime built) &t.arch_state.qwen4 else null;
}

/// Flash-Next's part of a forward context; null in a build without qwen4.
pub fn ctxOf(c: *trunk.ForwardCtx) ?*Ctx {
    return if (comptime built) &c.arch_ctx.qwen4 else null;
}

/// Whether `t` holds a loaded Flash-Next model.
pub fn isLoaded(t: *const Trunk) bool {
    const st = stateOfConst(t) orelse return false;
    return st.model != null;
}

/// The loaded MTP head; null without one, or in a build without qwen4.
pub fn mtpOf(t: *Trunk) ?*qwen4_forward.Qwen4Mtp {
    const st = stateOf(t) orelse return null;
    return if (st.mtp) |*m| m else null;
}

pub fn mtpOfConst(t: *const Trunk) ?*const qwen4_forward.Qwen4Mtp {
    const st = stateOfConst(t) orelse return null;
    return if (st.mtp) |*m| m else null;
}

/// What a forward context carries for Flash-Next (`ForwardCtx.arch_ctx.qwen4`).
pub const Ctx = struct {
    /// Interleaved M-RoPE: the server-computed flat `[3 × mrope_total]` position table (axis-major t,h,w rows),
    /// borrowed from the slot, and the delta that shifts the scalar decode RoPE. `mrope_cos/sin_cur` are the
    /// per-prefill-chunk tables `beginMropeChunk` builds for every full-attention layer of the chunk. All null/0 for a
    /// text-only request.
    mrope_pos: ?[]const i32 = null,
    mrope_total: usize = 0,
    mrope_delta: i32 = 0,
    mrope_cos_cur: ?mlx.mlx_array = null,
    mrope_sin_cur: ?mlx.mlx_array = null,
    /// The bool `[B, 1, S, kv]` QSA mask (indexer-selected blocks ∧ causal) this layer applies; null-ctx = dense.
    qsa_mask: mlx.mlx_array = .{ .ctx = null },
    /// The sorted `[B, S, kb]` int32 block selection (INT_MAX past each row's count) for the gather arms, set instead
    /// of `qsa_mask`; null-ctx = not selected.
    qsa_blocks: mlx.mlx_array = .{ .ctx = null },
    /// Which QSA arm served each full-attention call of this request: the `[qsa-arms]` line is a per-request meter
    /// (`logMeters`), and a batched forward notes each call on every slot's own ctx. The MTP head's forward builds its
    /// own ctx, so its QSA layer is not in the trunk's tally.
    qsa_arms: qwen4_qsa_attn.QsaArmCounts = .{},
    /// MTP head: its one QSA entry, not `ssm_entries[layer]`, and the absolute position of its key row 0. Null/0 on
    /// the trunk.
    qsa_entry: ?*trunk.SSMCacheEntry = null,
    qsa_pos_base: c_int = 0,
    /// The PLE rows a forward over lazy ids (`ForwardCtx.lazy_ids`) left unfilled: `flushDeferredPle` gathers them on
    /// the host right before the graph is evaluated; a forward evaluated with them pending reads zero rows.
    ple_pending: ?qwen4_hc.PlePending = null,
    /// Receives the pre-mixer hyper-connection stream `[B, L, hc*hidden]` (all positions), what the MTP head consumes.
    capture_stream_all: ?*mlx.mlx_array = null,
    /// A layer-at-a-time capture: the stream entering the layer, in place of the embedding, and whether the layer
    /// skips the PLE injection (it is not the PLE layer).
    stream_in: ?mlx.mlx_array = null,
    skip_ple: bool = false,

    pub fn mrope(self: *const Ctx) ?mrope_mod.PositionContext {
        const pos = self.mrope_pos orelse return null;
        return .{ .pos = pos, .total = self.mrope_total, .delta = self.mrope_delta };
    }

    pub fn setMrope(self: *Ctx, pos: ?[]const i32, total: usize, delta: i32) void {
        self.mrope_pos = pos;
        self.mrope_total = total;
        self.mrope_delta = delta;
    }

    /// The `[qsa-arms]` line: which QSA arm served this request's calls. Resets the tally.
    pub fn logMeters(self: *Ctx) void {
        const c = self.qsa_arms.take() orelse return;
        const arm = c.majority() orelse return;
        log.info(
            "  [qsa-arms] qsa={s} calls={d} mask={d} decode={d} verify={d} prefill={d}\n",
            .{ @tagName(arm), c.total(), c.mask, c.decode, c.verify, c.prefill },
        );
    }
};

/// `owned_bf16` takes the mixer's pre-transposed arrays so the trunk frees them with its own.
fn load(self: *State, args: arch_state.InitArgs) !?Trunk.ExpertStreamParts {
    const allocator = args.allocator;
    const config = args.config.*;
    const weights = args.weights;
    const prefix = args.prefix;
    const name_buf = args.name_buf;
    const s = args.s;
    const owned_bf16 = args.owned_bf16;
    var extra: std.ArrayList(mlx.mlx_array) = .empty;
    defer extra.deinit(allocator);
    var mb: [160]u8 = undefined;
    var mixer = try layer_weights.loadHcWeights(weights, std.fmt.bufPrint(&mb, "{s}.hyper_connection_mixer", .{prefix}) catch unreachable, false, config.hc_count, config.hidden_size, &extra, allocator, s);
    try layer_weights.bindHcLinears(&mixer, &config);
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
    self.* = .{
        .model = st,
        .mixer = mixer,
        .mtp = mtp,
        .decode_async_ladder = qwen4_forward.qwen4DecodeLadderStride(
            if (std.c.getenv("SUSHI_DECODE_ASYNC_LADDER")) |v| std.mem.span(v) else null,
        ),
    };
    return stream;
}

test "Flash-Next state: an empty state releases as a no-op, twice" {
    var state: State = .{};
    state.deinit(std.testing.allocator);
    state.deinit(std.testing.allocator);
    try std.testing.expect(state.model == null and state.mtp == null);
}
