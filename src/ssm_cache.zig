const trf = @import("transformer.zig");
const std = @import("std");
const qwen4_mod = @import("qwen4/qwen4_exp.zig");
const gdn_decode = @import("qwen4/gdn_decode.zig");
const mlx = @import("mlx.zig");
const testing = std.testing;
const kv_cache = @import("kv_cache.zig");
const KVCacheEntry = kv_cache.KVCacheEntry;
const qwen4_qsa_state = @import("qwen4/qwen4_qsa_state.zig");
const qsaHistoryRows = qwen4_qsa_state.qsaHistoryRows;
const qsaKeyAppendPlan = qwen4_qsa_state.qsaKeyAppendPlan;
const qsaLeftoverAt = qwen4_qsa_state.qsaLeftoverAt;
const qsaRawKeyBytes = qwen4_qsa_state.qsaRawKeyBytes;
const qsaResliceToKeep = qwen4_qsa_state.qsaResliceToKeep;
const ssmAuxIsQsaHistory = qwen4_qsa_state.ssmAuxIsQsaHistory;
const ssmFreeQsaState = qwen4_qsa_state.ssmFreeQsaState;
const test_support = @import("test_support.zig");
const qwen4ReadF32 = test_support.qwen4ReadF32;
const ForwardCtx = trf.ForwardCtx;

pub const SSMCacheEntry = struct {
    conv_state: mlx.mlx_array, // [B, kernel-1, conv_dim]
    ssm_state: mlx.mlx_array, // [B, Hv, Dv, Dk]
    initialized: bool,
    /// PLD spec-decode capture: per-position SSM states recorded by the GDN
    /// verify forward (`Transformer.spec_capture_ssm`). Shape [T, B, Hv, Dv, Dk]
    /// where T = verify length. Lets partial-accept rollback pick the
    /// accepted-position state WITHOUT a re-forward — the verify already ran the
    /// (sequential, expensive on a 48-layer GatedDeltaNet trunk) recurrence, so
    /// re-running it for the accepted prefix is pure waste. Null outside a
    /// capturing verify; freed at the end of every `nextPld` round.
    spec_state_seq: mlx.mlx_array = .{ .ctx = null },
    /// The verify that stored only its final state: what rollback replays the accepted rows from.
    /// Held instead of `spec_state_seq`, never beside it.
    spec_replay: gdn_decode.Replay = .{},
    /// PLD spec-decode capture: the conv1d input `[B, (kernel-1)+T, conv_dim]`
    /// of the verify forward, so the accepted-position `conv_state` is just a
    /// slice (no re-forward). Null outside capture.
    spec_conv_input: mlx.mlx_array = .{ .ctx = null },
    /// qwen4_exp per-request state beside the GDN pair: the PLE dilated-conv
    /// window `[B, (K-1)*dilation, hc*hidden]` on the PLE layer, the raw QSA
    /// indexer key history `[B, kv, idx_hd]` on full-attention layers.
    /// QSA history is a prefix of ONE buffer — stride checkpoints must NOT
    /// copy it (attachQsaHistoryToLatest puts one copy on the latest snap).
    aux_state: mlx.mlx_array = .{ .ctx = null },
    /// PLE: the (ngram_size-1) tokens preceding the next forward.
    /// The (ngram_size - 1) tokens preceding this entry.
    ple_prev: [qwen4_mod.MAX_NGRAM_SIZE]u32 = @splat(0),
    ple_prev_valid: bool = false,
    /// Spec-verify capture for the PLE layer: the dilated-conv input
    /// `[B, state_len + T, hc*hidden]` and the token history (prev ++ ids)
    /// of the verify forward, so a partial accept slices instead of
    /// re-forwarding. Null / 0 outside a capturing verify.
    spec_ple_input: mlx.mlx_array = .{ .ctx = null },
    spec_ple_tokens: [32]u32 = @splat(0),
    spec_ple_len: u32 = 0,
    /// QSA: pooled + normed + roped block keys `[B, nb, idx_hd]` for the
    /// complete blocks of `aux_state`, extended incrementally per forward.
    qsa_pooled: mlx.mlx_array = .{ .ctx = null },
    qsa_ratio: c_int = 4,
    /// QSA raw-key history accelerator: a pre-grown `[B, cap, hd]` capacity buffer whose first
    /// `qsa_key_rows` rows are the history; `aux_state` stays the authority as a tight view.
    /// `qsaAppendKeys` trusts it only when `qsa_key_rows` equals the position being appended
    /// at; live sites that replace `aux_state` go through `ssmFreeQsaState`.
    qsa_key_buf: mlx.mlx_array = .{ .ctx = null },
    qsa_key_rows: c_int = 0,
    /// Pooled-bank accelerators, same discipline: `qsa_pooled_buf` is the capacity buffer
    /// `qsa_pooled` views; `qsa_score_bank` is the f32 `[B, 1, hd, nb]` score operand,
    /// a view of `qsa_score_buf`, which grows by appended columns as blocks complete.
    qsa_pooled_buf: mlx.mlx_array = .{ .ctx = null },
    qsa_pooled_blocks: c_int = 0,
    qsa_score_bank: mlx.mlx_array = .{ .ctx = null },
    qsa_score_buf: mlx.mlx_array = .{ .ctx = null },
    qsa_score_blocks: c_int = 0,
    /// Rows this entry's QSA history is asked to hold up front (the counterpart of
    /// `KVCache.reserve_tokens`); the pooled bank derives its own by `ratio`. 0 = none.
    qsa_reserve_rows: usize = 0,
    /// Logical QSA history length in tokens (`pooled_blocks * ratio + ring_tail`).
    qsa_hist_rows: c_int = 0,
    group: ?*PersistentSsmGroup = null,
    group_row: u16 = 0,
    group_layer: u16 = 0,
    /// Bumped on every write to this entry outside its group; the group refuses a
    /// member whose gen moved. Read only from the entries, never from a ctx copy.
    member_gen: u64 = 0,
};

/// Raw indexer-key rows kept per layer (tokens): verify width plus one block.
pub const QSA_RING_ROWS: c_int = 32;

/// Widest `indexer_compress_ratio` the ring serves; past it a checkpoint is refused at load.
pub const QSA_MAX_COMPRESS_RATIO: c_int = 8;

pub fn arrayBytes(arr: mlx.mlx_array) u64 {
    if (arr.ctx == null) return 0;
    return @as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr));
}

pub fn kvEntriesBytes(entries: []const KVCacheEntry) u64 {
    var total: u64 = 0;
    for (entries) |e| {
        if (!e.initialized or e.shared_view) continue;
        inline for (.{ e.keys, e.values, e.keys_scales, e.keys_biases, e.values_scales, e.values_biases }) |arr| {
            total += arrayBytes(arr);
        }
    }
    return total;
}

/// A live request's recurrent state, the hybrid's counterpart of its KV. Each QSA bank is billed
/// by its capacity buffer when it has one, never again through the view that reads it.
pub fn ssmEntryBytes(e: *const SSMCacheEntry) u64 {
    return arrayBytes(e.conv_state) + arrayBytes(e.ssm_state) + qsaRawKeyBytes(e) +
        arrayBytes(if (e.qsa_pooled_buf.ctx != null) e.qsa_pooled_buf else e.qsa_pooled) +
        arrayBytes(if (e.qsa_score_buf.ctx != null) e.qsa_score_buf else e.qsa_score_bank);
}

/// Append `chunk` into a pre-grown capacity buffer along `axis` and republish `view` as the
/// tight `count`-long prefix. The published view is dropped first: while it lives mlx copies
/// the buffer instead of donating it.
/// Materialize `src` straight into a capacity buffer already `reserve` rows long: the seed
/// half of `capBufAppend`, done once. A tight seed made the next append grow (two full passes
/// over the prefix, ~81 ms at 393k). Byte-identical to the pair it replaces.
pub fn seedCapBuf(s: mlx.mlx_stream, src: mlx.mlx_array, axis: usize, reserve: usize) !mlx.mlx_array {
    const sh = mlx.getShape(src);
    const nd: usize = sh.len;
    std.debug.assert(nd <= 4 and axis < nd);
    const held: usize = @intCast(sh[axis]);
    // No reservation (or one the history exceeds): the tight copy is the right buffer.
    if (reserve <= held) return materializedOwnedCopy(s, src);
    var shape = [_]c_int{ 0, 0, 0, 0 };
    var stop = [_]c_int{ 0, 0, 0, 0 };
    for (sh, 0..) |d, i| {
        shape[i] = d;
        stop[i] = d;
    }
    shape[axis] = @intCast(reserve);
    const start = [_]c_int{ 0, 0, 0, 0 };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    var grown = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grown);
    try mlx.check(mlx.mlx_zeros(&grown, &shape, nd, mlx.mlx_array_dtype(src), s));
    var seeded = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(seeded);
    try mlx.check(mlx.mlx_slice_update(&seeded, grown, src, &start, nd, &stop, nd, &strides, nd, s));
    // Not counted in `qsa_cap_buf_allocs`: that counter means "capBufAppend had to grow".
    qwen4_qsa_state.qsa_history_seeds += 1;
    return seeded;
}

pub fn capBufAppend(s: mlx.mlx_stream, buf: *mlx.mlx_array, view: *mlx.mlx_array, count: *c_int, chunk: mlx.mlx_array, axis: usize, reserve: usize) !void {
    const csh = mlx.getShape(chunk);
    const nd: usize = csh.len;
    std.debug.assert(nd <= 4 and axis < nd);
    const add = csh[axis];
    if (view.ctx != null) _ = mlx.mlx_array_free(view.*);
    view.* = .{ .ctx = null };

    const cap: usize = if (buf.ctx != null) @intCast(mlx.getShape(buf.*)[axis]) else 0;
    const plan = qsaKeyAppendPlan(@intCast(count.*), cap, @intCast(add), reserve);
    var start = [_]c_int{ 0, 0, 0, 0 };
    var stop = [_]c_int{ 0, 0, 0, 0 };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    for (csh, 0..) |d, i| stop[i] = d;

    if (plan.new_cap != cap) {
        qwen4_qsa_state.qsa_cap_buf_allocs += 1;
        var shape = [_]c_int{ 0, 0, 0, 0 };
        for (csh, 0..) |d, i| shape[i] = d;
        shape[axis] = @intCast(plan.new_cap);
        var grown = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(grown);
        try mlx.check(mlx.mlx_zeros(&grown, &shape, nd, mlx.mlx_array_dtype(chunk), s));
        if (buf.ctx != null) {
            if (count.* > 0) {
                stop[axis] = count.*;
                var old = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(old);
                try mlx.check(mlx.mlx_slice(&old, buf.*, &start, nd, &stop, nd, &strides, nd, s));
                var seeded = mlx.mlx_array_new();
                // Safe unscoped: on failure `grown` is still the old handle, so the two errdefers free two different arrays.
                errdefer _ = mlx.mlx_array_free(seeded);
                try mlx.check(mlx.mlx_slice_update(&seeded, grown, old, &start, nd, &stop, nd, &strides, nd, s));
                _ = mlx.mlx_array_free(grown);
                grown = seeded;
            }
            _ = mlx.mlx_array_free(buf.*);
        }
        buf.* = grown;
    }
    start[axis] = count.*;
    stop[axis] = @intCast(plan.new_rows);
    // Block-scoped errdefer on purpose: `updated` becomes `buf.*` on the last line.
    {
        var updated = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(updated);
        try mlx.check(mlx.mlx_slice_update(&updated, buf.*, chunk, &start, nd, &stop, nd, &strides, nd, s));
        _ = mlx.mlx_array_free(buf.*);
        buf.* = updated;
    }
    count.* = @intCast(plan.new_rows);

    start[axis] = 0;
    view.* = mlx.mlx_array_new();
    errdefer {
        _ = mlx.mlx_array_free(view.*);
        view.* = .{ .ctx = null };
    }
    try mlx.check(mlx.mlx_slice(view, buf.*, &start, nd, &stop, nd, &strides, nd, s));
}

/// SSM snapshot value. Holds clones of conv_state and ssm_state via refcount —
/// the underlying buffer is shared with the source entry, but the snapshot
/// owns its own array handles and frees them on deinit. Used for
/// speculative-decoding rollback (PLD) where we must be able to revert one
/// decode step on a hybrid model.
pub const SSMCacheEntrySnapshot = struct {
    conv_state: mlx.mlx_array,
    ssm_state: mlx.mlx_array,
    initialized: bool,
    /// qwen4_exp per-request state beside the GDN pair (see SSMCacheEntry):
    /// the PLE conv window / QSA key history, the pooled block keys, and the
    /// PLE token history. Null / invalid on every other arch.
    aux_state: mlx.mlx_array = .{ .ctx = null },
    qsa_pooled: mlx.mlx_array = .{ .ctx = null },
    qsa_ratio: c_int = 4,
    qsa_rows: c_int = 0,
    /// The (ngram_size - 1) tokens preceding this entry.
    ple_prev: [qwen4_mod.MAX_NGRAM_SIZE]u32 = @splat(0),
    ple_prev_valid: bool = false,
};

pub fn ssmSnapshot(src: *const SSMCacheEntry) SSMCacheEntrySnapshot {
    var out: SSMCacheEntrySnapshot = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = src.initialized,
    };
    // mlx_array_set increments refcount on the underlying buffer; both handles
    // point at the same data. Subsequent writes to src.conv_state/ssm_state
    // create NEW handles so the snapshot's view is unaffected.
    //
    // We must guard each field independently — the two states are populated by
    // DIFFERENT code paths and either may legitimately be null even when
    // `initialized == true`:
    //   - An attention layer of a hybrid trunk holds neither.
    //   - GatedDeltaNet: `conv1dWithCache` sets `initialized=true` BEFORE the
    //     recurrence body initializes `ssm_state`, so a snapshot taken in the
    //     middle would see a null ssm_state. (Currently snapshots are only
    //     taken between full forward passes, but defensive null-handling
    //     prevents future regressions.)
    //
    // Calling `mlx_array_set` with a null source aborts the process via mlx-c's
    // default error handler ("expected a non-empty mlx_array"), so we cannot
    // rely on `try mlx.check(...)`.
    var grouped = false;
    if (src.group) |g| {
        if (g.live and src.group_layer < g.layers.len) {
            const stream = mlx.gpuStream();
            const m = g.layers[src.group_layer];
            const conv_ok = if (m.conv_state.ctx != null) blk: {
                const c = ssmMaterializeLive(stream, src, m.conv_state) catch break :blk false;
                _ = mlx.mlx_array_free(out.conv_state);
                out.conv_state = c;
                break :blk true;
            } else true;
            const ssm_ok = if (m.ssm_state.ctx != null) blk: {
                const c = ssmMaterializeLive(stream, src, m.ssm_state) catch break :blk false;
                _ = mlx.mlx_array_free(out.ssm_state);
                out.ssm_state = c;
                break :blk true;
            } else true;
            const aux_ok = if (m.aux_state.ctx != null) blk: {
                const c = ssmMaterializeLive(stream, src, m.aux_state) catch break :blk false;
                out.aux_state = c;
                break :blk true;
            } else blk: {
                if (src.aux_state.ctx != null) {
                    out.aux_state = mlx.mlx_array_new();
                    _ = mlx.mlx_array_set(&out.aux_state, src.aux_state);
                }
                break :blk true;
            };
            if (conv_ok and ssm_ok and aux_ok) {
                ssmEvalHandles(&.{ out.conv_state, out.ssm_state, out.aux_state });
                grouped = true;
            } else {
                ssmSnapshotDeinit(&out);
                out = .{
                    .conv_state = mlx.mlx_array_new(),
                    .ssm_state = mlx.mlx_array_new(),
                    .initialized = src.initialized,
                };
            }
        }
    }
    if (!grouped) {
        if (src.conv_state.ctx != null) {
            _ = mlx.mlx_array_set(&out.conv_state, src.conv_state);
        }
        if (src.ssm_state.ctx != null) {
            _ = mlx.mlx_array_set(&out.ssm_state, src.ssm_state);
        }
        if (src.aux_state.ctx != null) {
            out.aux_state = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&out.aux_state, src.aux_state);
        }
    }
    if (src.qsa_pooled.ctx != null) {
        out.qsa_pooled = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&out.qsa_pooled, src.qsa_pooled);
    }
    out.qsa_ratio = src.qsa_ratio;
    out.qsa_rows = qsaHistoryRows(src);
    out.ple_prev = src.ple_prev;
    out.ple_prev_valid = src.ple_prev_valid;
    return out;
}

pub fn ssmSnapshotDeinit(snap: *SSMCacheEntrySnapshot) void {
    _ = mlx.mlx_array_free(snap.conv_state);
    _ = mlx.mlx_array_free(snap.ssm_state);
    if (snap.aux_state.ctx != null) _ = mlx.mlx_array_free(snap.aux_state);
    if (snap.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(snap.qsa_pooled);
    snap.aux_state = .{ .ctx = null };
    snap.qsa_pooled = .{ .ctx = null };
}

pub fn ssmRestore(dst: *SSMCacheEntry, snap: *const SSMCacheEntrySnapshot) !void {
    _ = mlx.mlx_array_free(dst.conv_state);
    _ = mlx.mlx_array_free(dst.ssm_state);
    dst.conv_state = mlx.mlx_array_new();
    dst.ssm_state = mlx.mlx_array_new();
    dst.initialized = snap.initialized;
    // Mirror snapshot's per-field null guard — the snapshot may legitimately
    // have a null ssm_state or null conv_state.
    if (snap.conv_state.ctx != null) {
        try mlx.check(mlx.mlx_array_set(&dst.conv_state, snap.conv_state));
    }
    if (snap.ssm_state.ctx != null) {
        try mlx.check(mlx.mlx_array_set(&dst.ssm_state, snap.ssm_state));
    }
    // qwen4_exp aux state: REPLACED, never merged — the entry may hold the
    // previous request's key history.
    ssmFreeQsaState(dst);
    if (snap.aux_state.ctx != null) {
        dst.aux_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&dst.aux_state, snap.aux_state));
    }
    if (snap.qsa_pooled.ctx != null) {
        dst.qsa_pooled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&dst.qsa_pooled, snap.qsa_pooled));
    }
    dst.qsa_ratio = snap.qsa_ratio;
    dst.ple_prev = snap.ple_prev;
    dst.ple_prev_valid = snap.ple_prev_valid;
    dst.qsa_hist_rows = snap.qsa_rows;
    if (snap.aux_state.ctx != null) {
        const sh = mlx.getShape(dst.aux_state);
        dst.qsa_key_rows = if (sh.len >= 2) sh[1] else 0;
    }
    if (snap.qsa_pooled.ctx != null) {
        const ps = mlx.getShape(dst.qsa_pooled);
        dst.qsa_pooled_blocks = if (ps.len >= 2) ps[1] else 0;
    }
    if (dst.group) |g| {
        try ssmPatchGroupRow(
            mlx.gpuStream(),
            g,
            dst.group_row,
            dst.group_layer,
            dst.conv_state,
            dst.ssm_state,
            dst.aux_state,
            .{ .conv = dst.conv_state.ctx != null, .ssm = dst.ssm_state.ctx != null, .aux = dst.aux_state.ctx != null },
        );
    }
}

/// Free the transient PLD spec-decode capture buffers on an SSM entry
/// (`spec_state_seq` / `spec_conv_input`). Idempotent — safe to call on an
/// entry that never captured. Called at the end of every `nextPld` round and
/// in the cache teardown paths so the capture never outlives a round.
pub fn ssmFreeSpecCapture(entry: *SSMCacheEntry) void {
    entry.spec_replay.deinit();
    if (entry.spec_state_seq.ctx != null) {
        _ = mlx.mlx_array_free(entry.spec_state_seq);
        entry.spec_state_seq = .{ .ctx = null };
    }
    if (entry.spec_conv_input.ctx != null) {
        _ = mlx.mlx_array_free(entry.spec_conv_input);
        entry.spec_conv_input = .{ .ctx = null };
    }
    if (entry.spec_ple_input.ctx != null) {
        _ = mlx.mlx_array_free(entry.spec_ple_input);
        entry.spec_ple_input = .{ .ctx = null };
    }
    entry.spec_ple_len = 0;
}

fn ssmAxisView(s: mlx.mlx_stream, arr: mlx.mlx_array, axis: usize, i: usize) !mlx.mlx_array {
    const shape = mlx.getShape(arr);
    var start: [8]c_int = @splat(0);
    var stop: [8]c_int = @splat(0);
    const strides: [8]c_int = @splat(1);
    for (0..shape.len) |d| stop[d] = shape[d];
    start[axis] = @intCast(i);
    stop[axis] = @intCast(i + 1);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, arr, &start, @intCast(shape.len), &stop, @intCast(shape.len), &strides, @intCast(shape.len), s));
    return out;
}

pub fn splitSpecCaptures(s: mlx.mlx_stream, m: *const SSMCacheEntry, e: *SSMCacheEntry, i: usize) !void {
    if (m.spec_state_seq.ctx != null) {
        const own = try ssmAxisView(s, m.spec_state_seq, 1, i);
        if (e.spec_state_seq.ctx != null) _ = mlx.mlx_array_free(e.spec_state_seq);
        e.spec_state_seq = own;
    }
    if (m.spec_conv_input.ctx != null) {
        const own = try ssmAxisView(s, m.spec_conv_input, 0, i);
        if (e.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(e.spec_conv_input);
        e.spec_conv_input = own;
    }
    if (m.spec_ple_input.ctx != null) {
        const own = try ssmAxisView(s, m.spec_ple_input, 0, i);
        if (e.spec_ple_input.ctx != null) _ = mlx.mlx_array_free(e.spec_ple_input);
        e.spec_ple_input = own;
    }
}

/// PLD partial-accept rollback for a GatedDeltaNet layer, using the
/// per-position capture from the verify forward instead of a re-forward.
/// Sets `conv_state` + `ssm_state` to the state AFTER processing the first
/// `1 + accepted` verify tokens (t1 + accepted accepted drafts).
///
/// Numerically identical to a fresh forward over `[t1, draft[0..accepted-1]]`:
/// the GDN recurrence is sequential, so the state captured at position
/// `accepted` of a length-(1+m) verify run is the exact same float32→bf16
/// value a length-(1+accepted) run would store as its final state. Likewise
/// `conv_state` is the same windowed slice of the same conv input. So this
/// preserves PLD's byte-equivalence guarantee (pinned by
/// `tests/test_pld_equivalence.sh`).
///
/// No-op when the entry holds no capture (non-GDN hybrid layer); the caller
/// only takes the fast path when capture succeeded.
/// `verify_len` is the number of positions the verify forward ran (T). A
/// GatedDeltaNet layer records it implicitly in `spec_state_seq`, but a
/// conv-only capture (no recurrent ssm_state at all) has
/// only `spec_conv_input`, whose leading `kernel-1` rows cannot be told from
/// verify positions without it. Passing it explicitly is what lets one helper
/// roll back both shapes.
/// True when a verify left this entry's recurrent state recoverable at any accepted row.
pub fn ssmHasSpecCapture(entry: *const SSMCacheEntry) bool {
    return entry.spec_state_seq.ctx != null or entry.spec_replay.held();
}

pub fn ssmRollbackFromCapture(entry: *SSMCacheEntry, accepted: u32, verify_len: u32, s: mlx.mlx_stream) !void {
    // qwen4_exp per-request state beside the GDN pair. PLE layer: the conv
    // window + token history slice at 1+accepted (same shape as conv_state
    // below). Attention layer: the QSA raw-key history is positional — keep
    // the rows up to the accepted verify prefix.
    //
    // The two arms are exclusive: `aux_state` holds the PLE window on the PLE layer and the
    // QSA key history on a full-attention layer. `ssmFreeQsaState` frees AND nulls.
    var patch_conv = false;
    var patch_ssm = false;
    var patch_aux = false;
    if (entry.spec_ple_input.ctx != null) {
        const ci_shape = mlx.getShape(entry.spec_ple_input); // [B, state_len+T, hc*H]
        const state_len = ci_shape[1] - @as(c_int, @intCast(verify_len));
        const cstart: c_int = @intCast(1 + accepted);
        const start = [_]c_int{ 0, cstart, 0 };
        const stop = [_]c_int{ ci_shape[0], cstart + state_len, ci_shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, entry.spec_ple_input, &start, 3, &stop, 3, &strides, 3, s));
        const owned = try materializedOwnedCopy(s, view);
        ssmFreeQsaState(entry);
        entry.aux_state = owned;
        const ctx_len: usize = @as(usize, entry.spec_ple_len) - @as(usize, verify_len);
        for (0..ctx_len) |i| entry.ple_prev[i] = entry.spec_ple_tokens[1 + accepted + i];
        entry.ple_prev_valid = true;
        patch_aux = true;
    } else if ((entry.aux_state.ctx != null or entry.qsa_pooled.ctx != null) and entry.conv_state.ctx == null) {
        const hist = qsaHistoryRows(entry);
        const keep: c_int = hist - @as(c_int, @intCast(verify_len)) + @as(c_int, @intCast(1 + accepted));
        if (!try qsaResliceToKeep(entry, keep, s)) {
            if (entry.aux_state.ctx == null) {
                entry.qsa_hist_rows = keep;
                try truncatePooled(&entry.qsa_pooled, keep, entry.qsa_ratio, s, true);
            } else {
                const ks = mlx.getShape(entry.aux_state);
                const drop_tail = hist - keep;
                // A slice past the ring's own start clamps to empty instead of failing, so
                // the unservable rollback is named here.
                if (drop_tail > ks[1]) return error.QsaRollbackBeyondRing;
                const new_held = ks[1] - drop_tail;
                const start = [_]c_int{ 0, 0, 0 };
                const stop = [_]c_int{ ks[0], new_held, ks[2] };
                const strides = [_]c_int{ 1, 1, 1 };
                var view = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(view);
                try mlx.check(mlx.mlx_slice(&view, entry.aux_state, &start, 3, &stop, 3, &strides, 3, s));
                const owned = try materializedOwnedCopy(s, view);
                const pooled = entry.qsa_pooled;
                entry.qsa_pooled = .{ .ctx = null };
                ssmFreeQsaState(entry);
                entry.qsa_pooled = pooled;
                entry.aux_state = owned;
                entry.qsa_key_rows = new_held;
                entry.qsa_hist_rows = keep;
                try truncatePooled(&entry.qsa_pooled, keep, entry.qsa_ratio, s, true);
            }
        }
    }
    const acc: c_int = @intCast(accepted);
    if (entry.spec_state_seq.ctx == null and entry.spec_replay.held() and entry.spec_conv_input.ctx != null) {
        const replayed = try gdn_decode.replayState(entry.spec_replay, entry.spec_conv_input, acc + 1, s);
        _ = mlx.mlx_array_free(entry.ssm_state);
        entry.ssm_state = replayed;
        patch_ssm = true;
    } else if (entry.spec_state_seq.ctx != null) {
        const seq_shape = mlx.getShape(entry.spec_state_seq); // [T, B, Hv, Dv, Dk]

        // ssm_state = spec_state_seq[accepted]  →  [B, Hv, Dv, Dk]
        {
            const start = [_]c_int{ acc, 0, 0, 0, 0 };
            const stop = [_]c_int{ acc + 1, seq_shape[1], seq_shape[2], seq_shape[3], seq_shape[4] };
            const strides = [_]c_int{ 1, 1, 1, 1, 1 };
            var sliced = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&sliced, entry.spec_state_seq, &start, 5, &stop, 5, &strides, 5, s));
            defer _ = mlx.mlx_array_free(sliced);
            const new_shape = [_]c_int{ seq_shape[1], seq_shape[2], seq_shape[3], seq_shape[4] };
            var reshaped = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&reshaped, sliced, &new_shape, 4, s));
            _ = mlx.mlx_array_free(entry.ssm_state);
            entry.ssm_state = reshaped;
            patch_ssm = true;
        }
    }

    // conv_state = spec_conv_input[:, (1+accepted) : (1+accepted)+(kernel-1), :]
    if (entry.spec_conv_input.ctx != null) {
        const ci_shape = mlx.getShape(entry.spec_conv_input); // [B, (k-1)+T, conv_dim]
        const t_len: c_int = @intCast(verify_len);
        const km1 = ci_shape[1] - t_len; // kernel - 1
        const cstart: c_int = @intCast(1 + accepted);
        const start = [_]c_int{ 0, cstart, 0 };
        const stop = [_]c_int{ ci_shape[0], cstart + km1, ci_shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var new_conv = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&new_conv, entry.spec_conv_input, &start, 3, &stop, 3, &strides, 3, s));
        _ = mlx.mlx_array_free(entry.conv_state);
        entry.conv_state = new_conv;
        patch_conv = true;
    }
    if (entry.group) |g| {
        try ssmPatchGroupRow(s, g, entry.group_row, entry.group_layer, entry.conv_state, entry.ssm_state, entry.aux_state, .{
            .conv = patch_conv,
            .ssm = patch_ssm,
            .aux = patch_aux,
        });
    }
}

pub const SsmMergeKind = packed struct {
    linear: bool = false,
    ple: bool = false,
};

const SsmPatchFields = packed struct {
    conv: bool = false,
    ssm: bool = false,
    aux: bool = false,
};

/// The batched tick's one owner of the merged GDN/conv/PLE state across ticks.
/// A stable group reads the current generation and writes the next; slots hold
/// (group, row, member_gen), never a slice view. Rows leave through
/// `ssmExportRow` (materialized + evaluated) before any generation is freed, and
/// `entries[].member_gen` is the ONE guard: any write outside the group bumps it
/// and `matches()` then refuses the slot. Retirement is MLX refcount: every
/// consumer holds its own handle.
pub const PersistentSsmGroup = struct {
    allocator: std.mem.Allocator = undefined,
    n: usize = 0,
    keys: []usize = &.{},
    gens: []u64 = &.{},
    member_entries: [][]SSMCacheEntry = &.{},
    layers: []SSMCacheEntry = &.{},
    kinds: []SsmMergeKind = &.{},
    live: bool = false,

    pub fn matches(self: *const PersistentSsmGroup, ctxs: []const *ForwardCtx) bool {
        if (!self.live or self.n != ctxs.len) return false;
        for (ctxs, 0..) |c, i| {
            const entries = c.ssm_entries orelse return false;
            if (self.keys[i] != @intFromPtr(entries.ptr)) return false;
            if (entries.len == 0 or self.gens[i] != entries[0].member_gen) return false;
        }
        return true;
    }

    pub fn clear(self: *PersistentSsmGroup) void {
        if (!self.live) return;
        ssmUnbindAll(self);
        for (self.layers) |*m| ssmFreeMergedLayer(m);
        self.allocator.free(self.layers);
        self.allocator.free(self.keys);
        self.allocator.free(self.gens);
        self.allocator.free(self.member_entries);
        self.allocator.free(self.kinds);
        self.layers = &.{};
        self.keys = &.{};
        self.gens = &.{};
        self.member_entries = &.{};
        self.kinds = &.{};
        self.n = 0;
        self.live = false;
    }

    pub fn deinit(self: *PersistentSsmGroup) void {
        self.clear();
    }
};

pub fn ssmRowView(s: mlx.mlx_stream, arr: mlx.mlx_array, i: usize) !mlx.mlx_array {
    const shape = mlx.getShape(arr);
    if (shape.len > 8) return error.SsmRankTooHigh;
    var start: [8]c_int = @splat(0);
    var stop: [8]c_int = @splat(0);
    const strides: [8]c_int = @splat(1);
    start[0] = @intCast(i);
    stop[0] = @intCast(i + 1);
    for (1..shape.len) |d| stop[d] = shape[d];
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, arr, &start, @intCast(shape.len), &stop, @intCast(shape.len), &strides, @intCast(shape.len), s));
    return out;
}

pub fn ssmFreeMergedLayer(m: *SSMCacheEntry) void {
    if (m.conv_state.ctx != null) _ = mlx.mlx_array_free(m.conv_state);
    if (m.ssm_state.ctx != null) _ = mlx.mlx_array_free(m.ssm_state);
    if (m.aux_state.ctx != null) _ = mlx.mlx_array_free(m.aux_state);
    ssmFreeSpecCapture(m);
    m.conv_state = .{ .ctx = null };
    m.ssm_state = .{ .ctx = null };
    m.aux_state = .{ .ctx = null };
    m.initialized = false;
}

pub fn ssmUnbindEntries(group: *PersistentSsmGroup, entries: []SSMCacheEntry) void {
    for (entries) |*e| {
        if (e.group == group) e.group = null;
    }
}

pub fn ssmUnbindAll(group: *PersistentSsmGroup) void {
    if (!group.live) return;
    for (group.member_entries) |entries| ssmUnbindEntries(group, entries);
}

pub fn ssmEvalHandles(arrs: []const mlx.mlx_array) void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var count: usize = 0;
    for (arrs) |a| {
        if (a.ctx == null) continue;
        _ = mlx.mlx_vector_array_append_value(vec, a);
        count += 1;
    }
    if (count > 0) _ = mlx.mlx_eval(vec);
}

fn ssmSliceUpdateRow(s: mlx.mlx_stream, parent: *mlx.mlx_array, row: usize, update: mlx.mlx_array) !void {
    const sh = mlx.getShape(parent.*);
    if (sh.len > 8) return error.SsmRankTooHigh;
    var start: [8]c_int = @splat(0);
    var stop: [8]c_int = @splat(0);
    const strides: [8]c_int = @splat(1);
    start[0] = @intCast(row);
    stop[0] = @intCast(row + 1);
    for (1..sh.len) |d| stop[d] = sh[d];
    var updated = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(updated);
    try mlx.check(mlx.mlx_slice_update(&updated, parent.*, update, &start, sh.len, &stop, sh.len, &strides, sh.len, s));
    _ = mlx.mlx_array_free(parent.*);
    parent.* = updated;
}

fn ssmPatchGroupRow(
    s: mlx.mlx_stream,
    group: *PersistentSsmGroup,
    row: usize,
    layer: usize,
    conv: mlx.mlx_array,
    ssm: mlx.mlx_array,
    aux: mlx.mlx_array,
    fields: SsmPatchFields,
) !void {
    if (!group.live or layer >= group.layers.len) return;
    const m = &group.layers[layer];
    if (fields.conv and conv.ctx != null and m.conv_state.ctx != null) {
        try ssmSliceUpdateRow(s, &m.conv_state, row, conv);
    }
    if (fields.ssm and ssm.ctx != null and m.ssm_state.ctx != null) {
        try ssmSliceUpdateRow(s, &m.ssm_state, row, ssm);
    }
    if (fields.aux and aux.ctx != null and m.aux_state.ctx != null) {
        try ssmSliceUpdateRow(s, &m.aux_state, row, aux);
    }
}

pub fn ssmMaterializeLive(
    s: mlx.mlx_stream,
    src: *const SSMCacheEntry,
    arr: mlx.mlx_array,
) !mlx.mlx_array {
    if (arr.ctx == null) return mlx.mlx_array_new();
    if (src.group) |g| {
        if (g.live) {
            const view = try ssmRowView(s, arr, src.group_row);
            defer _ = mlx.mlx_array_free(view);
            return materializedOwnedCopy(s, view);
        }
    }
    return materializedOwnedCopy(s, arr);
}

pub fn ssmLiveAux(e: *const SSMCacheEntry) mlx.mlx_array {
    if (e.group) |g| {
        if (g.live and e.group_layer < g.layers.len) {
            const a = g.layers[e.group_layer].aux_state;
            if (a.ctx != null) return a;
        }
    }
    return e.aux_state;
}

/// Keep the pooled-key rows that are still complete blocks of a key history
/// truncated to `keep_rows` (the block size is implied by the two shapes).
/// Cut the pooled bank to the blocks `keep_rows` completes. `materialize` = a fresh buffer; false = a slice view.
pub fn truncatePooled(pooled: *mlx.mlx_array, keep_rows: c_int, ratio: c_int, s: mlx.mlx_stream, materialize: bool) !void {
    if (pooled.ctx == null) return;
    const ps = mlx.getShape(pooled.*);
    const keep_blocks: c_int = @divTrunc(keep_rows, ratio);
    if (keep_blocks >= ps[1]) return;
    if (keep_blocks <= 0) {
        _ = mlx.mlx_array_free(pooled.*);
        pooled.* = .{ .ctx = null };
        return;
    }
    const start = [_]c_int{ 0, 0, 0 };
    const stop = [_]c_int{ ps[0], keep_blocks, ps[2] };
    const strides = [_]c_int{ 1, 1, 1 };
    var view = mlx.mlx_array_new();
    mlx.check(mlx.mlx_slice(&view, pooled.*, &start, 3, &stop, 3, &strides, 3, s)) catch |err| {
        _ = mlx.mlx_array_free(view);
        return err;
    };
    const owned = if (materialize) blk: {
        defer _ = mlx.mlx_array_free(view);
        break :blk try materializedOwnedCopy(s, view);
    } else view;
    _ = mlx.mlx_array_free(pooled.*);
    pooled.* = owned;
}

/// Phase 1 (performance-plan): per-position SSM checkpoint covering ALL hybrid
/// layers in the model. Captures the SSM/conv state after the model has been
/// forwarded over the first `pos` tokens of some prompt. Used by the hot prefix
/// cache to restore mid-sequence SSM state on a multi-turn warm request, so we
/// don't have to cold-prefill the shared prefix every turn.
///
/// A single checkpoint = one `SSMCacheEntrySnapshot` per layer (matching the
/// shape of `ctx.ssm_entries`). Layers whose ssm/conv state is uninitialized
/// at snapshot time hold a null-handle snapshot — `ssmRestore` is null-safe.
///
/// Memory: BF16 GatedDeltaNet state is ~260 KB per layer at typical
/// configurations. A 48-layer 1k-token snapshot stride 128 stores 8
/// checkpoints × 48 layers × 260 KB ≈ 100 MB. Bounded via
/// `HotPrefixCache.max_kv_bytes` (counts toward the same budget as KV).
pub const SSMCheckpoint = struct {
    /// 1-based KV position immediately AFTER forwarding `pos` tokens. The
    /// caller restores the slot's KVCache to exactly this position alongside
    /// the SSM state, so the model behaves as if it had only seen the first
    /// `pos` tokens of the prompt.
    pos: usize,
    /// Per-layer SSM snapshots — same length as `ctx.ssm_entries`. Non-SSM
    /// layers (plain attention) get a null-handle snapshot which is a no-op
    /// in `ssmRestore`.
    layers: []SSMCacheEntrySnapshot,
    /// Nonzero once the state left RAM for SSD (`release`): the bytes it held, which a disk byte
    /// budget still prices. The position stays, so every restore-point list still names it.
    released_bytes: u64 = 0,

    /// Frees the recurrent state, keeping the position.
    pub fn release(self: *SSMCheckpoint) void {
        if (self.released_bytes != 0) return;
        self.released_bytes = ssmCheckpointBytes(self);
        for (self.layers) |*l| {
            ssmSnapshotDeinit(l);
            l.conv_state = mlx.mlx_array_new();
            l.ssm_state = mlx.mlx_array_new();
            l.initialized = false;
        }
    }

    pub fn isReleased(self: *const SSMCheckpoint) bool {
        return self.released_bytes != 0;
    }

    pub fn deinit(self: *SSMCheckpoint, allocator: std.mem.Allocator) void {
        for (self.layers) |*l| ssmSnapshotDeinit(l);
        allocator.free(self.layers);
        self.layers = &[_]SSMCacheEntrySnapshot{};
        self.pos = 0;
        self.released_bytes = 0;
    }
};

/// Checkpoint states one Qwen request holds in RAM when an SSD tier backs its cache: a write-through prefill
/// holds two (the one before the newest, until its file is written, and the one just captured); a request
/// it cannot serve is thinned to two and holds a third while one is captured. The admission bill and the
/// capture cap both read it.
pub const SSM_RESIDENT_MAX: u32 = 3;

/// Snapshot of every entry in `ssm_entries` at the current point. Caller owns
/// the resulting buffer (free via `SSMCheckpoint.deinit`).
///
/// Unlike the per-decode-step `ssmSnapshot` (PLD rollback — transient, so a
/// cheap refcount-share is right), checkpoints OUTLIVE the request inside the
/// hot prefix cache. The live conv/ssm states are routinely shared-buffer
/// SLICES of much larger parents (the prefill chunk's conv input
/// [B,(k-1)+T,C], the GDN capture sequence), so sharing the handle silently
/// retained the whole parent — the ~3.4x "[hot-cache] resident" under-count
/// on hybrid archs. `materializedOwnedCopy` forces a fresh buffer; the copies
/// are evaluated here, in one batch, so the parents' lifetimes end with the
/// chunk's activation graph instead of with the cache entry.
pub fn captureSsmCheckpoint(
    allocator: std.mem.Allocator,
    ssm_entries: []const SSMCacheEntry,
    pos: usize,
    s: mlx.mlx_stream,
) !SSMCheckpoint {
    const layers = try allocator.alloc(SSMCacheEntrySnapshot, ssm_entries.len);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*l| ssmSnapshotDeinit(l);
        allocator.free(layers);
    }
    for (ssm_entries, 0..) |*src, i| {
        var out: SSMCacheEntrySnapshot = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = src.initialized,
        };
        errdefer ssmSnapshotDeinit(&out);
        // Per-field null guards mirror `ssmSnapshot` — either state may
        // legitimately be null.
        const live_conv = if (src.group) |g| (if (g.live and src.group_layer < g.layers.len) g.layers[src.group_layer].conv_state else src.conv_state) else src.conv_state;
        const live_ssm = if (src.group) |g| (if (g.live and src.group_layer < g.layers.len) g.layers[src.group_layer].ssm_state else src.ssm_state) else src.ssm_state;
        const live_ple = ssmLiveAux(src);
        if (live_conv.ctx != null) {
            const c = try ssmMaterializeLive(s, src, live_conv);
            _ = mlx.mlx_array_free(out.conv_state);
            out.conv_state = c;
        }
        if (live_ssm.ctx != null) {
            const c = try ssmMaterializeLive(s, src, live_ssm);
            _ = mlx.mlx_array_free(out.ssm_state);
            out.ssm_state = c;
        }
        // PLE conv window (O(1), rides a GDN layer) is per-position state and
        // belongs in every checkpoint. QSA indexer history (aux && !conv) is
        // a prefix of one buffer — copying it at every stride is what OOM'd
        // a 400k prefill while auto-context advertised 700k–1M. One copy is
        // attached to the latest checkpoint after prefill.
        const grouped_ple = src.group != null and src.group.?.live and src.group_layer < src.group.?.layers.len and src.group.?.layers[src.group_layer].aux_state.ctx != null;
        if (grouped_ple or (src.aux_state.ctx != null and !ssmAuxIsQsaHistory(src))) {
            out.aux_state = try ssmMaterializeLive(s, src, live_ple);
        }
        out.qsa_ratio = src.qsa_ratio;
        if (!grouped_ple and ssmAuxIsQsaHistory(src)) {
            out.qsa_rows = @intCast(pos);
            if (src.aux_state.ctx != null) {
                out.aux_state = try qsaLeftoverAt(src.aux_state, qsaHistoryRows(src), @intCast(pos), src.qsa_ratio, s);
            }
        }
        out.ple_prev = src.ple_prev;
        out.ple_prev_valid = src.ple_prev_valid;
        layers[i] = out;
        built = i + 1;
    }
    // Materialize all copies in one batch so the parent buffers can be
    // released with the prefill chunk's graph. Without this eval the lazy
    // copy node itself keeps the parent alive.
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var count: usize = 0;
        for (layers) |*l| {
            inline for (.{ l.conv_state, l.ssm_state, l.aux_state, l.qsa_pooled }) |arr| {
                if (arr.ctx != null) {
                    _ = mlx.mlx_vector_array_append_value(vec, arr);
                    count += 1;
                }
            }
        }
        if (count > 0) _ = mlx.mlx_eval(vec);
    }
    return .{ .pos = pos, .layers = layers };
}

/// Force a REAL copy of `x` into a freshly allocated buffer. `mlx_copy` and
/// `mlx_contiguous` both take shared-buffer fast paths (Copy is a view op;
/// contiguous no-ops when the view's strides read as row-contiguous, which a
/// size-1-leading-dim slice does), so neither breaks the alias to a slice's
/// parent buffer. Adding a same-dtype scalar zero always runs a kernel with a
/// newly allocated output. Buffer donation can't alias either: the caller
/// still holds `x`, so its buffer is never donated.
pub fn materializedOwnedCopy(s: mlx.mlx_stream, x: mlx.mlx_array) !mlx.mlx_array {
    var zero = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zero);
    const scalar_shape = [_]c_int{1};
    try mlx.check(mlx.mlx_zeros(&zero, &scalar_shape, 0, mlx.mlx_array_dtype(x), s));
    var neg_zero = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_zero);
    try mlx.check(mlx.mlx_negative(&neg_zero, zero, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, x, neg_zero, s));
    return out;
}

/// `settle` evaluates the copy so it lets go of a prefill chunk; a verify's few rows stay
/// lazy and ride one dispatch with the lm_head and the accept read.
pub fn capturePrefillHidden(s: mlx.mlx_stream, dest: *mlx.mlx_array, src: mlx.mlx_array, settle: bool) !void {
    const owned = try materializedOwnedCopy(s, src);
    defer _ = mlx.mlx_array_free(owned);
    if (settle) try mlx.check(mlx.mlx_array_eval(owned));
    _ = mlx.mlx_array_set(dest, owned);
}

pub fn capturePrefillHiddenLast(s: mlx.mlx_stream, dest: *mlx.mlx_array, src: mlx.mlx_array, settle: bool) !void {
    const sh = mlx.getShape(src);
    const last = sh[1] - 1;
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    try mlx.check(mlx.mlx_slice(&sliced, src, &[_]c_int{ 0, last, 0 }, 3, &[_]c_int{ sh[0], sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    try capturePrefillHidden(s, dest, sliced, settle);
}

/// Test seams (inference thread only): the copy/eval counters let the retention and cadence tests see what the loop did.
pub var compact_conv_state_override: ?bool = null;
pub var compact_conv_state_copy_count: u64 = 0;
pub var cadence_conv_eval_count: u64 = 0;

pub fn compactConvStateEnabled() bool {
    return compact_conv_state_override orelse true;
}

/// A second owner for `cp`'s state: fresh handles and a fresh layer slice,
/// refcount-SHARING every underlying buffer (`mlx_array_set`, the same idiom
/// `ssmSnapshot` and `KVCache.snapshot` use). Both checkpoints then free
/// independently and the DATA outlives whichever dies first.
///
/// This is what makes checkpoint inheritance cheap enough to be unconditional:
/// a commit built on a restored prefix carries the donor entry's checkpoints
/// without re-copying multi-GB of GDN state, and evicting the donor later
/// frees nothing the inheritor still needs. `captureSsmCheckpoint` must
/// MATERIALIZE (its sources are slices of a live prefill graph); a checkpoint
/// is already materialized, so sharing it aliases nothing transient.
pub fn shareSsmCheckpoint(allocator: std.mem.Allocator, cp: *const SSMCheckpoint) !SSMCheckpoint {
    const layers = try allocator.alloc(SSMCacheEntrySnapshot, cp.layers.len);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*l| ssmSnapshotDeinit(l);
        allocator.free(layers);
    }
    for (cp.layers, 0..) |*src, i| {
        var out: SSMCacheEntrySnapshot = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = src.initialized,
        };
        // Per-field null guards: mlx_array_set on a null source aborts the
        // process via mlx-c's default error handler (see `ssmSnapshot`).
        if (src.conv_state.ctx != null) _ = mlx.mlx_array_set(&out.conv_state, src.conv_state);
        if (src.ssm_state.ctx != null) _ = mlx.mlx_array_set(&out.ssm_state, src.ssm_state);
        if (src.aux_state.ctx != null) {
            out.aux_state = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&out.aux_state, src.aux_state);
        }
        if (src.qsa_pooled.ctx != null) {
            out.qsa_pooled = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&out.qsa_pooled, src.qsa_pooled);
        }
        out.qsa_ratio = src.qsa_ratio;
        out.qsa_rows = src.qsa_rows;
        out.ple_prev = src.ple_prev;
        out.ple_prev_valid = src.ple_prev_valid;
        layers[i] = out;
        built = i + 1;
    }
    return .{ .pos = cp.pos, .layers = layers, .released_bytes = cp.released_bytes };
}

/// Restore every layer of `ssm_entries` from `cp`. Mirrors the per-layer
/// `ssmRestore` pattern — null-safe on either side.
pub fn restoreSsmCheckpoint(
    ssm_entries: []SSMCacheEntry,
    cp: *const SSMCheckpoint,
) !void {
    if (ssm_entries.len != cp.layers.len) return error.SsmCheckpointLayerMismatch;
    for (ssm_entries, cp.layers) |*dst, *src| {
        try ssmRestore(dst, src);
    }
}

/// Total bytes held by an SSM checkpoint (sum of conv_state + ssm_state across
/// all layers). Used for hot-cache memory budgeting alongside `KVCacheSnapshot`
/// bytes.
pub fn ssmCheckpointBytes(cp: *const SSMCheckpoint) u64 {
    var total: u64 = 0;
    for (cp.layers) |l| {
        inline for (.{ l.conv_state, l.ssm_state, l.aux_state, l.qsa_pooled }) |arr| {
            if (arr.ctx != null) total += @as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr));
        }
    }
    return total;
}

/// Which retention policy a checkpoint list is thinned with. The span-preserving thin moves
/// where a warm turn restores from and was measured on qwen4_exp only, so the policy is chosen
/// by `longCtxGated()` at each site and the ungated value names that site's previous
/// behaviour: `.oldest` (prefill capture, disk tier), `.min_span` (hot-cache merge/shed),
/// `.min_span_recency` (`.min_span` plus a dense newest quarter; qwen4_exp).
pub const ThinPolicy = enum { oldest, min_span, min_span_recency };

/// The checkpoint index to drop when a retention list is over its cap: the interior position
/// whose removal widens the coverage gap least; index 0 and the last are always kept. One
/// selection for every retention site (#330: end-anchored survivors left no affordable trim point).
pub fn spanPreservingDropIndex(
    comptime T: type,
    items: []const T,
    comptime posOf: fn (*const T) usize,
    policy: ThinPolicy,
    /// An index the caller knows a later turn must restore from (a media boundary): never
    /// selected. When it is the only candidate the oldest goes instead — that one sits below
    /// it, so the protected position survives either way.
    protect: ?usize,
) usize {
    std.debug.assert(items.len > 0);
    if (policy == .oldest) return 0;
    if (items.len < 3) return 0;
    // Recency bias: the newest quarter stays at capture density (a warm turn that edits near
    // the end restores from there); everything below it is thinned span-preservingly.
    const scan_end = if (policy == .min_span_recency and items.len >= RECENCY_DENSE_MIN)
        items.len - items.len / 4
    else
        items.len;
    var best_at: usize = 1;
    var best_span: usize = std.math.maxInt(usize);
    var k: usize = 1;
    while (k + 1 < items.len and k < scan_end) : (k += 1) {
        if (protect) |p| if (k == p) continue;
        const span = posOf(&items[k + 1]) -| posOf(&items[k - 1]);
        if (span < best_span) {
            best_span = span;
            best_at = k;
        }
    }
    if (best_span == std.math.maxInt(usize)) return 0;
    return best_at;
}

/// Shortest list that reserves a dense newest quarter.
const RECENCY_DENSE_MIN: usize = 8;

fn checkpointPosOf(cp: *const SSMCheckpoint) usize {
    return cp.pos;
}

fn u32PosOf(p: *const u32) usize {
    return p.*;
}

fn usizePosOf(p: *const usize) usize {
    return p.*;
}

/// `spanPreservingDropIndex` over a checkpoint list.
pub fn ssmCheckpointDropIndex(cps: []const SSMCheckpoint, policy: ThinPolicy, protect: ?usize) usize {
    return spanPreservingDropIndex(SSMCheckpoint, cps, checkpointPosOf, policy, protect);
}

/// `spanPreservingDropIndex` over a bare ascending position list.
pub fn positionDropIndex(positions: []const u32, policy: ThinPolicy) usize {
    return spanPreservingDropIndex(u32, positions, u32PosOf, policy, null);
}

pub fn positionDropIndexUsize(positions: []const usize, policy: ThinPolicy) usize {
    return spanPreservingDropIndex(usize, positions, usizePosOf, policy, null);
}

test "splitSpecCaptures: spec_ple_input row view is the rollback shape" {
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    const N: c_int = 2;
    const state_len: c_int = 3;
    const verify_len: c_int = 4;
    const width: c_int = 5;
    var merged: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeSpecCapture(&merged);
    const n: f64 = @floatFromInt(N * (state_len + verify_len) * width);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_arange(&flat, 0.0, n, 1.0, .float32, s));
    const shape = [_]c_int{ N, state_len + verify_len, width };
    merged.spec_ple_input = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&merged.spec_ple_input, flat, &shape, 3, s));

    var slots: [2]SSMCacheEntry = .{
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true },
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true },
    };
    defer for (&slots) |*e| {
        ssmFreeSpecCapture(e);
        if (e.aux_state.ctx != null) _ = mlx.mlx_array_free(e.aux_state);
    };
    try splitSpecCaptures(s, &merged, &slots[0], 0);
    try splitSpecCaptures(s, &merged, &slots[1], 1);
    for (&slots, 0..) |*e, i| {
        try t.expect(e.spec_ple_input.ctx != null);
        const sh = mlx.getShape(e.spec_ple_input);
        try t.expectEqual(@as(c_int, 1), sh[0]);
        try t.expectEqual(state_len + verify_len, sh[1]);
        try t.expectEqual(width, sh[2]);
        e.spec_ple_len = @intCast(state_len + verify_len);
        const accepted: u32 = if (i == 0) 1 else 0;
        try ssmRollbackFromCapture(e, accepted, @intCast(verify_len), s);
        try t.expect(e.aux_state.ctx != null);
        const ash = mlx.getShape(e.aux_state);
        try t.expectEqual(@as(c_int, 1), ash[0]);
        try t.expectEqual(state_len, ash[1]);
        try t.expectEqual(width, ash[2]);
    }
}

test "materializedOwnedCopy: -0.0 survives bit-for-bit" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const data = [_]f32{ -0.0, 0.0, 1.5, -2.5 };
    const x = mlx.mlx_array_new_data(@ptrCast(&data), &[_]c_int{4}, 1, .float32);
    defer _ = mlx.mlx_array_free(x);
    const out = try materializedOwnedCopy(s, x);
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    const got = mlx.mlx_array_data_float32(out) orelse return error.NoData;
    try testing.expect(std.math.signbit(got[0]));
    try testing.expect(!std.math.signbit(got[1]));
    try testing.expectEqual(@as(f32, 1.5), got[2]);
    try testing.expectEqual(@as(f32, -2.5), got[3]);
}

test "SSMCacheEntry snapshot/restore round-trip preserves arrays" {
    const s = mlx.gpuStream();
    var entry: SSMCacheEntry = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = false,
    };
    defer {
        _ = mlx.mlx_array_free(entry.conv_state);
        _ = mlx.mlx_array_free(entry.ssm_state);
    }

    // Populate with arbitrary state.
    const conv_shape = [_]c_int{ 1, 3, 4 };
    _ = mlx.mlx_array_free(entry.conv_state);
    entry.conv_state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&entry.conv_state, &conv_shape, 3, .float32, s));

    const ssm_shape = [_]c_int{ 1, 2, 8, 4 };
    _ = mlx.mlx_array_free(entry.ssm_state);
    entry.ssm_state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&entry.ssm_state, &ssm_shape, 4, .float32, s));
    entry.initialized = true;

    var snap = ssmSnapshot(&entry);
    defer ssmSnapshotDeinit(&snap);

    // Mutate: replace ssm_state with a different shape.
    _ = mlx.mlx_array_free(entry.ssm_state);
    entry.ssm_state = mlx.mlx_array_new();
    const new_shape = [_]c_int{ 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_zeros(&entry.ssm_state, &new_shape, 4, .float32, s));

    try ssmRestore(&entry, &snap);
    try testing.expect(entry.initialized);
    const restored_shape = mlx.getShape(entry.ssm_state);
    try testing.expectEqual(@as(c_int, 1), restored_shape[0]);
    try testing.expectEqual(@as(c_int, 2), restored_shape[1]);
    try testing.expectEqual(@as(c_int, 8), restored_shape[2]);
    try testing.expectEqual(@as(c_int, 4), restored_shape[3]);
}

test "captureSsmCheckpoint materializes state copies (parent-buffer retention class)" {
    // Hot-cache SSM checkpoints used to refcount-share the live conv/ssm
    // state handles. Those are routinely SLICES of much larger parent buffers
    // (e.g. the prefill chunk's conv input [B,(k-1)+T,C]), so a committed
    // entry silently retained the whole parent — the measured ~3.4x
    // "[hot-cache] resident" under-count on hybrid archs. Capture must
    // materialize an OWNED copy: the checkpoint's data pointer may not alias
    // the parent's buffer, and ssmCheckpointBytes must equal true retention.
    const s = mlx.gpuStream();

    // Parent buffer: [1, 1024, 8] f32, 32 KB.
    var parent = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(parent);
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, 8192.0, 1.0, .float32, s));
        const shape = [_]c_int{ 1, 1024, 8 };
        try mlx.check(mlx.mlx_reshape(&parent, flat, &shape, 3, s));
        _ = mlx.mlx_array_eval(parent);
    }

    // conv_state := parent[:, 1021:1024, :] — after eval this is a
    // shared-buffer view into `parent` (MLX slice keeps the parent alive).
    var entry: SSMCacheEntry = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = true,
    };
    defer {
        _ = mlx.mlx_array_free(entry.conv_state);
        _ = mlx.mlx_array_free(entry.ssm_state);
    }
    {
        const st = [_]c_int{ 0, 1021, 0 };
        const sp = [_]c_int{ 1, 1024, 8 };
        const sd = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&entry.conv_state, parent, &st, 3, &sp, 3, &sd, 3, s));
        _ = mlx.mlx_array_eval(entry.conv_state);
    }

    var entries = [_]SSMCacheEntry{entry};
    var cp = try captureSsmCheckpoint(testing.allocator, &entries, 3, s);
    defer cp.deinit(testing.allocator);

    // The checkpoint's conv_state must live in its own buffer, outside the
    // parent's data range.
    _ = mlx.mlx_array_eval(cp.layers[0].conv_state);
    const parent_ptr = mlx.mlx_array_data_float32(parent).?;
    const parent_bytes = @as(u64, mlx.mlx_array_size(parent)) * @as(u64, mlx.mlx_array_itemsize(parent));
    const cp_ptr = mlx.mlx_array_data_float32(cp.layers[0].conv_state).?;
    const p0 = @intFromPtr(parent_ptr);
    const p1 = p0 + parent_bytes;
    const c0 = @intFromPtr(cp_ptr);
    try testing.expect(c0 < p0 or c0 >= p1);

    // Values must survive the copy: parent row 1021 starts at 1021*8.
    try testing.expectEqual(@as(f32, 1021.0 * 8.0), cp_ptr[0]);
    try testing.expectEqual(@as(f32, 1021.0 * 8.0 + 23.0), cp_ptr[23]);

    // Accounting matches true retention: 3*8 f32 elements, nothing more.
    try testing.expectEqual(@as(u64, 3 * 8 * 4), ssmCheckpointBytes(&cp));
}

test "two-chunk prefill capture drops the chunk parent after the boundary eval" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const rows: c_int = 2048;
    const dim: c_int = 2048;
    const keep: c_int = 8;
    const parent_bytes: usize = @as(usize, @intCast(rows)) * @as(usize, @intCast(dim)) * 2;

    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    var baseline: usize = 0;
    _ = mlx.mlx_get_active_memory(&baseline);

    var kept0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kept0);
    var kept1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kept1);
    const kept = [_]*mlx.mlx_array{ &kept0, &kept1 };

    var chunk: usize = 0;
    while (chunk < 2) : (chunk += 1) {
        var parent = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&parent, &[_]c_int{ 1, rows, dim }, 3, .float16, s));
        try mlx.check(mlx.mlx_array_eval(parent));
        var view = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&view, parent, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, keep, dim }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try capturePrefillHidden(s, kept[chunk], view, true);
        var dummy = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dummy);
        try mlx.check(mlx.mlx_zeros(&dummy, &[_]c_int{1}, 1, .float16, s));
        try mlx.check(mlx.mlx_array_eval(dummy));
        _ = mlx.mlx_array_free(view);
        _ = mlx.mlx_array_free(parent);
        _ = mlx.mlx_synchronize(s);
        _ = mlx.mlx_clear_cache();
    }

    var after: usize = 0;
    _ = mlx.mlx_get_active_memory(&after);
    const delta = after -| baseline;
    try t.expect(delta < parent_bytes);
}

test "ssm retention: the span-preserving thin keeps both ends and spreads the survivors" {
    // Drop-oldest survivors covered only the last `max * stride` tokens (#330).
    const t = std.testing;
    var positions: [94]usize = undefined;
    for (&positions, 0..) |*p, i| p.* = (i + 1) * 4096;
    var n: usize = positions.len;
    while (n > 16) {
        const drop = positionDropIndexUsize(positions[0..n], .min_span_recency);
        try t.expect(drop != 0);
        try t.expect(drop + 1 != n);
        var k = drop;
        while (k + 1 < n) : (k += 1) positions[k] = positions[k + 1];
        n -= 1;
    }
    try t.expectEqual(@as(usize, 16), n);
    try t.expectEqual(@as(usize, 4096), positions[0]);
    try t.expectEqual(@as(usize, 94 * 4096), positions[n - 1]);
    // Spread, but not evenly: the newest quarter stays at capture density.
    const ideal = (positions[n - 1] - positions[0]) / (n - 1);
    var i: usize = 1;
    while (i < n) : (i += 1) try t.expect(positions[i] - positions[i - 1] <= 2 * ideal);
    try t.expectEqual(@as(usize, 4096), positions[n - 1] - positions[n - 2]);
    try t.expectEqual(@as(usize, 4096), positions[n - 2] - positions[n - 3]);
    try t.expect(positions[1] - positions[0] > 4 * 4096);
    try t.expectEqual(@as(usize, 0), positionDropIndexUsize(positions[0..2], .min_span_recency));
    try t.expectEqual(@as(usize, 0), positionDropIndexUsize(positions[0..1], .min_span_recency));
    const clustered = [_]u32{ 0, 1000, 1010, 2000 };
    try t.expectEqual(@as(usize, 2), positionDropIndex(&clustered, .min_span_recency));

    // Every other arch keeps the previous policy of its own site: `.oldest` (prefill capture,
    // disk tier) and `.min_span` (hot-cache sites, no recency quarter).
    try t.expectEqual(@as(usize, 0), positionDropIndex(&clustered, .oldest));
    try t.expectEqual(@as(usize, 0), positionDropIndexUsize(positions[0..n], .oldest));
    {
        // 12 entries; the tightest pair is inside the newest quarter (starts at 9).
        var pos12: [12]usize = .{ 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9010, 10_000 };
        try t.expectEqual(@as(usize, 10), positionDropIndexUsize(&pos12, .min_span));
        try t.expect(positionDropIndexUsize(&pos12, .min_span_recency) < 9);
        try t.expectEqual(
            positionDropIndexUsize(pos12[0..7], .min_span),
            positionDropIndexUsize(pos12[0..7], .min_span_recency),
        );
    }
}

test "hidden capture: a verify's stays lazy for its one dispatch, a prefill chunk's is settled" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 1, -2, 3, -4, 5, -6 };
    const shape = [_]c_int{ 1, 3, 2 };
    const base = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(base);
    for ([_]bool{ false, true }) |settle| for ([_]bool{ false, true }) |last| {
        var src = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(src);
        try mlx.check(mlx.mlx_add(&src, base, base, s));
        var dest = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dest);
        if (last) try capturePrefillHiddenLast(s, &dest, src, settle) else try capturePrefillHidden(s, &dest, src, settle);
        var available = false;
        try mlx.check(mlx._mlx_array_is_available(&available, dest));
        try testing.expectEqual(settle, available);
        const got = try qwen4ReadF32(testing.allocator, dest, s);
        defer testing.allocator.free(got);
        const want: []const f32 = if (last) &.{ 10, -12 } else &.{ 2, -4, 6, -8, 10, -12 };
        try testing.expectEqualSlices(f32, want, got);
    };
}
