const trf = @import("../transformer.zig");
const std = @import("std");
const mlx = @import("../mlx.zig");
const qwen4_forward = @import("qwen4_forward.zig");
const qwen4_qsa = @import("qwen4_qsa.zig");
const QsaHeadMarkSet = qwen4_qsa.QsaHeadMarkSet;
const Qwen4Mtp = qwen4_forward.Qwen4Mtp;
const round_cost_mod = @import("../round_cost.zig");
const testing = std.testing;
const qwen4_test_support = @import("qwen4_test_support.zig");
const Qwen4TestSlot = qwen4_test_support.Qwen4TestSlot;
const rollbackRowToAccepted = qwen4_test_support.rollbackRowToAccepted;
const test_support = @import("../test_support.zig");
const attn256MaxDiff = test_support.attn256MaxDiff;
const attn256RandBf16 = test_support.attn256RandBf16;
const KVCache = trf.KVCache;
const QSA_MAX_COMPRESS_RATIO = trf.QSA_MAX_COMPRESS_RATIO;
const QSA_RING_ROWS = trf.QSA_RING_ROWS;
const SSMCacheEntry = trf.SSMCacheEntry;
const SSMCacheEntrySnapshot = trf.SSMCacheEntrySnapshot;
const SSMCheckpoint = trf.SSMCheckpoint;
const Transformer = trf.Transformer;
const arrayBytes = trf.arrayBytes;
const captureSsmCheckpoint = trf.captureSsmCheckpoint;
const materializedOwnedCopy = trf.materializedOwnedCopy;
const restoreSsmCheckpoint = trf.restoreSsmCheckpoint;
const ssmCheckpointBytes = trf.ssmCheckpointBytes;
const ssmEntryBytes = trf.ssmEntryBytes;
const ssmFreeSpecCapture = trf.ssmFreeSpecCapture;
const ssmRestore = trf.ssmRestore;
const ssmRollbackFromCapture = trf.ssmRollbackFromCapture;
const ssmSnapshot = trf.ssmSnapshot;
const ssmSnapshotDeinit = trf.ssmSnapshotDeinit;
const truncatePooled = trf.truncatePooled;

comptime {
    // A partial-accept rollback re-slices the ring to `keep`, so the ring must still hold a
    // whole verify width plus the partial block that width reopened.
    std.debug.assert(QSA_RING_ROWS >= @as(c_int, @intCast(round_cost_mod.MAX_WIDTH)) + QSA_MAX_COMPRESS_RATIO + 1);
}

/// The ring is a fixed size, so a config asking for a wider block than it can hold is refused
/// by name instead of serving a leftover the rollback cannot re-pool.
pub fn checkQsaCompressRatio(ratio: u32) !void {
    if (ratio == 0 or ratio > QSA_MAX_COMPRESS_RATIO) return error.UnsupportedQsaCompressRatio;
}

pub fn qsaRawKeyBytes(e: *const SSMCacheEntry) u64 {
    return arrayBytes(if (e.qsa_key_buf.ctx != null) e.qsa_key_buf else e.aux_state);
}

pub fn qsaLeftoverAt(aux: mlx.mlx_array, hist: c_int, pos: c_int, ratio: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    return qsaRowsEndingAt(aux, hist, pos, @mod(pos, @max(ratio, 1)), s);
}

/// The `want` raw indexer rows ending at row `pos`, out of a buffer whose rows end at logical
/// row `hist`; null when the buffer no longer physically holds them.
fn qsaRowsEndingAt(aux: mlx.mlx_array, hist: c_int, pos: c_int, want: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    if (want <= 0 or aux.ctx == null) return .{ .ctx = null };
    const sh = mlx.getShape(aux);
    if (sh.len < 2 or sh[1] < want) return .{ .ctx = null };
    const held = sh[1];
    const ring_start = hist - held;
    const want_start = pos - want;
    if (want_start < ring_start or pos > hist) return .{ .ctx = null };
    const phys_from = want_start - ring_start;
    const phys_to = pos - ring_start;
    var view = mlx.mlx_array_new();
    mlx.check(mlx.mlx_slice(&view, aux, &[_]c_int{ 0, phys_from, 0 }, 3, &[_]c_int{ sh[0], phys_to, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s)) catch |err| {
        _ = mlx.mlx_array_free(view);
        return err;
    };
    defer _ = mlx.mlx_array_free(view);
    return materializedOwnedCopy(s, view);
}

fn qsaRingTail(src: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(src);
    const n: c_int = @min(sh[1], QSA_RING_ROWS);
    if (n == sh[1]) return materializedOwnedCopy(s, src);
    var view = mlx.mlx_array_new();
    mlx.check(mlx.mlx_slice(&view, src, &[_]c_int{ 0, sh[1] - n, 0 }, 3, &[_]c_int{ sh[0], sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s)) catch |err| {
        _ = mlx.mlx_array_free(view);
        return err;
    };
    defer _ = mlx.mlx_array_free(view);
    return materializedOwnedCopy(s, view);
}

/// Free and null a live entry's whole QSA state: the history, its capacity buffer and the pooled bank move together.
pub fn ssmFreeQsaState(e: *SSMCacheEntry) void {
    if (e.aux_state.ctx != null) _ = mlx.mlx_array_free(e.aux_state);
    if (e.qsa_key_buf.ctx != null) _ = mlx.mlx_array_free(e.qsa_key_buf);
    if (e.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(e.qsa_pooled);
    if (e.qsa_pooled_buf.ctx != null) _ = mlx.mlx_array_free(e.qsa_pooled_buf);
    if (e.qsa_score_bank.ctx != null) _ = mlx.mlx_array_free(e.qsa_score_bank);
    if (e.qsa_score_buf.ctx != null) _ = mlx.mlx_array_free(e.qsa_score_buf);
    e.aux_state = .{ .ctx = null };
    e.qsa_key_buf = .{ .ctx = null };
    e.qsa_key_rows = 0;
    e.qsa_hist_rows = 0;
    e.qsa_pooled = .{ .ctx = null };
    e.qsa_pooled_buf = .{ .ctx = null };
    e.qsa_pooled_blocks = 0;
    e.qsa_score_bank = .{ .ctx = null };
    e.qsa_score_buf = .{ .ctx = null };
    e.qsa_score_blocks = 0;
}

/// Where the next `add` rows of QSA raw-key history go, and how big the capacity buffer must
/// be. `cap == 0` = no buffer yet. Growth is the KV cache's proportional policy.
const QsaKeyAppendPlan = struct { new_cap: usize, write_at: usize, new_rows: usize };
pub fn qsaKeyAppendPlan(rows: usize, cap: usize, add: usize, reserve: usize) QsaKeyAppendPlan {
    const need = rows + add;
    // A reservation raises the first grow to the whole request; past it the proportional policy takes over.
    const grown = if (need <= cap) cap else @max(KVCache.nextCapacityPolicy(cap, need), reserve);
    return .{
        .new_cap = grown,
        .write_at = rows,
        .new_rows = need,
    };
}

/// How many times a capacity buffer actually allocated.
pub var qsa_cap_buf_allocs: usize = 0;

/// How many times a QSA history accelerator was re-seeded from its authority (`seedCapBuf`).
pub var qsa_history_seeds: usize = 0;


/// Ask every live entry's QSA history to hold `tokens` rows up front (same lever as `KVCache.reserve`).
pub fn reserveQsaHistory(entries: ?[]SSMCacheEntry, tokens: usize) void {
    const es = entries orelse return;
    for (es) |*e| {
        if (tokens > e.qsa_reserve_rows) e.qsa_reserve_rows = tokens;
    }
}

pub fn reserveQsaHistoryWithHead(entries: ?[]SSMCacheEntry, head: ?*SSMCacheEntry, tokens: usize) void {
    reserveQsaHistory(entries, tokens);
    if (head) |h| reserveQsaHistory(h[0..1], tokens);
}

/// Re-publish an entry's QSA views over the first `keep` rows of the capacity buffers it
/// already owns, instead of copying the history out and re-seeding it. False (the caller
/// falls back to the copy) when there is no capacity buffer: a restore, a cold entry, a
/// post-adopt head.
pub fn qsaResliceToKeep(entry: *SSMCacheEntry, keep: c_int, s: mlx.mlx_stream) !bool {
    const hist = qsaHistoryRows(entry);
    if (keep > hist) return false;
    const drop_tail = hist - keep;
    const held = entry.qsa_key_rows;
    if (drop_tail > held) return false;
    const new_held = held - drop_tail;
    // The retained ring must still hold `keep`'s own leftover for the next append to re-pool
    // it; short of that the clamp belongs to the caller's mark path, or to its named decline.
    if (new_held < @mod(keep, @max(entry.qsa_ratio, 1))) return false;
    if (new_held > 0) {
        const buf_ok = entry.qsa_key_buf.ctx != null and
            mlx.getShape(entry.qsa_key_buf)[1] >= new_held;
        if (!buf_ok) return false;
        try capBufReslice(s, entry.qsa_key_buf, &entry.aux_state, &entry.qsa_key_rows, 1, new_held);
    } else {
        if (entry.aux_state.ctx != null) _ = mlx.mlx_array_free(entry.aux_state);
        entry.aux_state = .{ .ctx = null };
        entry.qsa_key_rows = 0;
    }
    entry.qsa_hist_rows = keep;
    const keep_blocks: c_int = @divTrunc(keep, @max(entry.qsa_ratio, 1));
    if (entry.qsa_pooled_buf.ctx != null) {
        if (keep_blocks > 0 and mlx.getShape(entry.qsa_pooled_buf)[1] >= keep_blocks) {
            try capBufReslice(s, entry.qsa_pooled_buf, &entry.qsa_pooled, &entry.qsa_pooled_blocks, 1, keep_blocks);
        } else {
            if (entry.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(entry.qsa_pooled);
            entry.qsa_pooled = .{ .ctx = null };
            entry.qsa_pooled_blocks = 0;
        }
    } else {
        try truncatePooled(&entry.qsa_pooled, keep, entry.qsa_ratio, s, false);
    }
    if (entry.qsa_score_bank.ctx != null) _ = mlx.mlx_array_free(entry.qsa_score_bank);
    entry.qsa_score_bank = .{ .ctx = null };
    if (entry.qsa_score_buf.ctx != null) _ = mlx.mlx_array_free(entry.qsa_score_buf);
    entry.qsa_score_buf = .{ .ctx = null };
    entry.qsa_score_blocks = 0;
    return true;
}

fn capBufReslice(s: mlx.mlx_stream, buf: mlx.mlx_array, view: *mlx.mlx_array, count: *c_int, axis: usize, new_count: c_int) !void {
    const sh = mlx.getShape(buf);
    const nd: usize = sh.len;
    if (view.ctx != null) _ = mlx.mlx_array_free(view.*);
    view.* = .{ .ctx = null };
    count.* = 0;
    if (new_count <= 0) return;
    var start = [_]c_int{ 0, 0, 0, 0 };
    var stop = [_]c_int{ 0, 0, 0, 0 };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    for (sh, 0..) |d, i| stop[i] = d;
    stop[axis] = new_count;
    view.* = mlx.mlx_array_new();
    errdefer {
        _ = mlx.mlx_array_free(view.*);
        view.* = .{ .ctx = null };
    }
    try mlx.check(mlx.mlx_slice(view, buf, &start, nd, &stop, nd, &strides, nd, s));
    count.* = new_count;
}

fn qsaTestMeanBlocks(keys: mlx.mlx_array, row0: c_int, n_blocks: c_int, ratio: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(keys);
    const n_rows = n_blocks * ratio;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_slice(&flat, keys, &[_]c_int{ 0, row0, 0 }, 3, &[_]c_int{ sh[0], row0 + n_rows, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var kb4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kb4);
    try mlx.check(mlx.mlx_reshape(&kb4, flat, &[_]c_int{ sh[0], n_blocks, ratio, sh[2] }, 4, s));
    var f32b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32b);
    try mlx.check(mlx.mlx_astype(&f32b, kb4, .float32, s));
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, f32b, 2, false, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, mean, .bfloat16, s));
    return out;
}

pub fn qsaTestAppendPool(xfm: *Transformer, entry: *SSMCacheEntry, chunk: mlx.mlx_array, offset: c_int, s: mlx.mlx_stream) !void {
    const ratio = @max(entry.qsa_ratio, 1);
    const add = mlx.getShape(chunk)[1];
    const kv = offset + add;
    const nb = @divTrunc(kv, ratio);
    const nb_cached: c_int = if (entry.qsa_pooled.ctx != null) mlx.getShape(entry.qsa_pooled)[1] else 0;
    if (nb > nb_cached) {
        const leftover_n = offset - nb_cached * ratio;
        const n_new = nb - nb_cached;
        const need = n_new * ratio;
        const from_k = need - leftover_n;
        var kb_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kb_flat);
        if (leftover_n == 0) {
            try mlx.check(mlx.mlx_slice(&kb_flat, chunk, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, need, mlx.getShape(chunk)[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        } else {
            const ash = mlx.getShape(entry.aux_state);
            var leftover = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(leftover);
            try mlx.check(mlx.mlx_slice(&leftover, entry.aux_state, &[_]c_int{ 0, ash[1] - leftover_n, 0 }, 3, &[_]c_int{ ash[0], ash[1], ash[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            var k_part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_part);
            try mlx.check(mlx.mlx_slice(&k_part, chunk, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, from_k, mlx.getShape(chunk)[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            const parts = [_]mlx.mlx_array{ leftover, k_part };
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&kb_flat, vec, 1, s));
        }
        const hd = mlx.getShape(chunk)[2];
        var kb4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kb4);
        try mlx.check(mlx.mlx_reshape(&kb4, kb_flat, &[_]c_int{ 1, n_new, ratio, hd }, 4, s));
        var f32b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f32b);
        try mlx.check(mlx.mlx_astype(&f32b, kb4, .float32, s));
        var mean = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mean);
        try mlx.check(mlx.mlx_mean_axis(&mean, f32b, 2, false, s));
        var new3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(new3);
        try mlx.check(mlx.mlx_astype(&new3, mean, .bfloat16, s));
        try xfm.qsaAppendPooled(entry, new3, nb_cached);
    }
    try xfm.qsaAppendKeys(entry, chunk, offset);
}

pub fn qsaHistoryBytesHeld(cps: []const SSMCheckpoint) u64 {
    var total: u64 = 0;
    for (cps) |cp| {
        for (cp.layers) |l| {
            inline for (.{ l.aux_state, l.qsa_pooled }) |arr| {
                if (arr.ctx != null) total += @as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr));
            }
        }
    }
    return total;
}

/// QSA indexer key history lives on full-attention layers: `aux_state` is
/// `[B, kv, idx_hd]` and there is no conv recurrence (`conv_state` null or
/// an empty handle). PLE's dilated-conv window is `aux_state` on a GDN
/// layer that HAS a real `conv_state` — that window is O(1) and stays in
/// every checkpoint.
pub fn ssmAuxIsQsaHistory(e: *const SSMCacheEntry) bool {
    if (e.qsa_pooled.ctx != null) return true;
    if (e.aux_state.ctx == null) return false;
    if (e.conv_state.ctx == null) return true;
    return mlx.mlx_array_size(e.conv_state) == 0;
}

pub fn qsaHistoryRows(e: *const SSMCacheEntry) c_int {
    if (e.qsa_hist_rows > 0) return e.qsa_hist_rows;
    if (!ssmAuxIsQsaHistory(e)) return 0;
    if (e.aux_state.ctx == null) return 0;
    const sh = mlx.getShape(e.aux_state);
    return if (sh.len >= 2) sh[1] else 0;
}

/// `ssmAuxIsQsaHistory` over a snapshot layer.
pub fn snapshotHasQsaHistory(l: *const SSMCacheEntrySnapshot) bool {
    return ssmAuxIsQsaHistory(&.{
        .conv_state = l.conv_state,
        .ssm_state = l.ssm_state,
        .aux_state = l.aux_state,
        .qsa_pooled = l.qsa_pooled,
        .initialized = l.initialized,
    });
}

pub fn checkpointHasQsaHistory(cp: *const SSMCheckpoint) bool {
    for (cp.layers) |*l| {
        if (snapshotHasQsaHistory(l)) return true;
    }
    return false;
}

/// Whether the pooled indexer bank — the history proper — lives on THIS checkpoint. Every
/// checkpoint carries its own raw leftover, so `checkpointHasQsaHistory` cannot answer "is this
/// the bank's home": a mid-block snap says yes on the leftover alone.
pub fn checkpointHasQsaPooled(cp: *const SSMCheckpoint) bool {
    for (cp.layers) |*l| {
        if (l.qsa_pooled.ctx != null) return true;
    }
    return false;
}

pub fn checkpointQsaAuxRows(cp: *const SSMCheckpoint) c_int {
    var r: c_int = 0;
    for (cp.layers) |*l| {
        if (!snapshotHasQsaHistory(l)) continue;
        if (l.qsa_rows > r) r = l.qsa_rows;
    }
    return r;
}

/// Raw leftover rows a checkpoint carries (0 on a non-QSA layer, and on a QSA layer whose
/// position is block-aligned).
fn checkpointQsaLeftoverRows(cp: *const SSMCheckpoint) c_int {
    var rows: c_int = 0;
    for (cp.layers) |*l| {
        if (!snapshotHasQsaHistory(l)) continue;
        if (l.aux_state.ctx == null) continue;
        const sh = mlx.getShape(l.aux_state);
        if (sh.len >= 2 and sh[1] > rows) rows = sh[1];
    }
    return rows;
}

fn qsaCoverTestCp(ring: c_int, blocks: c_int, rows: c_int, ratio: c_int, hd: c_int, s: mlx.mlx_stream) !SSMCheckpoint {
    const layers = try std.testing.allocator.alloc(SSMCacheEntrySnapshot, 1);
    errdefer std.testing.allocator.free(layers);
    layers[0] = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio, .qsa_rows = rows };
    var cp: SSMCheckpoint = .{ .pos = @intCast(rows), .layers = layers };
    errdefer cp.deinit(std.testing.allocator);
    if (ring > 0) {
        layers[0].aux_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&layers[0].aux_state, &[_]c_int{ 1, ring, hd }, 3, .float32, s));
    }
    if (blocks > 0) {
        layers[0].qsa_pooled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&layers[0].qsa_pooled, &[_]c_int{ 1, blocks, hd }, 3, .float32, s));
    }
    return cp;
}

/// Whether a QSA restore at `pos` has its whole indexer history: `cp` — the checkpoint being
/// restored — carries the raw leftover for `pos` (the `pos % ratio` rows the next append
/// re-pools), and `src` a pooled bank reaching `pos`. A ring sliced at another position is a
/// miss, never a silently wrong leftover.
pub fn checkpointQsaCoversPos(cp: *const SSMCheckpoint, src: *const SSMCheckpoint, pos: usize) bool {
    const logical = checkpointQsaAuxRows(src);
    if (logical == 0) return true;
    const p: c_int = @intCast(pos);
    if (logical < p) return false;
    var has_pooled = false;
    var pblocks: c_int = 0;
    var ratio: c_int = 4;
    for (src.layers) |*l| {
        if (l.qsa_ratio > 0) ratio = l.qsa_ratio;
        if (l.qsa_pooled.ctx == null) continue;
        has_pooled = true;
        const ps = mlx.getShape(l.qsa_pooled);
        if (ps.len >= 2 and ps[1] > pblocks) pblocks = ps[1];
    }
    const r = @max(ratio, 1);
    if (has_pooled and pblocks < @divTrunc(p, r)) return false;
    if (!has_pooled) return qsaSourceHoldsRowsAt(src, p, p);
    const want_lv = @mod(p, r);
    const cp_rows = checkpointQsaLeftoverRows(cp);
    if (cp_rows == 0) return qsaSourceHoldsRowsAt(src, p, want_lv);
    if (cp_rows == want_lv) return true;
    return qsaSourceHoldsRowsAt(cp, p, want_lv) or qsaSourceHoldsRowsAt(src, p, want_lv);
}

pub fn qsaEntrySatisfiesForward(e: *const SSMCacheEntry, pos: c_int) bool {
    if (!ssmAuxIsQsaHistory(e)) return true;
    if (qsaHistoryRows(e) != pos) return false;
    const ratio = @max(e.qsa_ratio, 1);
    const blocks: c_int = if (e.qsa_pooled.ctx != null) mlx.getShape(e.qsa_pooled)[1] else 0;
    const held: c_int = if (e.aux_state.ctx != null) mlx.getShape(e.aux_state)[1] else 0;
    return blocks * ratio + held >= pos;
}

pub fn qsaRestoreSatisfiesForward(entries: []const SSMCacheEntry, pos: usize) bool {
    const p: c_int = @intCast(pos);
    for (entries) |*e| {
        if (!qsaEntrySatisfiesForward(e, p)) return false;
    }
    return true;
}

pub fn checkpointListHasQsaPooled(cps: []const SSMCheckpoint) bool {
    for (cps) |*cp| {
        if (checkpointHasQsaPooled(cp)) return true;
    }
    return false;
}

/// Whether every QSA layer of `src` physically holds the `want` raw rows ending at `pos`.
fn qsaSourceHoldsRowsAt(src: *const SSMCheckpoint, pos: c_int, want: c_int) bool {
    if (want <= 0) return true;
    var any = false;
    for (src.layers) |*l| {
        if (!snapshotHasQsaHistory(l)) continue;
        if (l.aux_state.ctx == null) return false;
        const sh = mlx.getShape(l.aux_state);
        if (sh.len < 2) return false;
        const hist: c_int = if (l.qsa_rows > 0) l.qsa_rows else sh[1];
        if (pos > hist or pos - want < hist - sh[1]) return false;
        any = true;
    }
    return any;
}

/// Copy QSA aux/pooled from the live entries onto the LATEST checkpoint —
/// ONE materialized copy, sliced to that checkpoint's position (a cancel
/// handoff attaches while the live history is ahead of the last stride
/// snap). Older snaps in `cps` drop any history they carried. No-op when
/// `cps` is empty or no layer holds QSA history.
pub fn attachQsaHistoryToLatest(cps: []SSMCheckpoint, live: []const SSMCacheEntry, s: mlx.mlx_stream) !void {
    return attachQsaHistoryToLatestMode(cps, live, s, .copy);
}

/// Rows a handed-off history may pin past the checkpoint's position before the handoff
/// materializes instead: an entry must never pin much more than `ssmCheckpointBytes` bills.
pub const QSA_HANDOFF_MAX_SLACK_ROWS: c_int = 16384;

/// The commit boundary: hand the slot's live QSA history to the latest checkpoint as a slice
/// view of the live capacity buffer, sliced to that checkpoint's position. The slot dies right
/// after, so the entry ends up the buffer's only owner. Too much slack materializes instead.
pub fn handoffQsaHistoryToLatest(cps: []SSMCheckpoint, live: []const SSMCacheEntry, s: mlx.mlx_stream) !void {
    return attachQsaHistoryToLatestMode(cps, live, s, .share);
}

pub const QsaAttachMode = enum { copy, share };

/// Does sharing `buf_rows` rows of capacity for a `keep`-row snap pin more slack than `QSA_HANDOFF_MAX_SLACK_ROWS`?
pub fn qsaHandoffExceedsSlack(buf_rows: c_int, keep: c_int) bool {
    return buf_rows - keep > QSA_HANDOFF_MAX_SLACK_ROWS;
}

/// The buffer a share of `src` would pin: the capacity buffer when the accelerator holds one, else the published history.
fn qsaHandoffMustMaterialize(src: *const SSMCacheEntry, keep: c_int) bool {
    const buf_rows: c_int = if (src.qsa_key_buf.ctx != null)
        mlx.getShape(src.qsa_key_buf)[1]
    else if (src.aux_state.ctx != null)
        mlx.getShape(src.aux_state)[1]
    else
        0;
    return qsaHandoffExceedsSlack(buf_rows, keep);
}

fn attachQsaHistoryToLatestMode(cps: []SSMCheckpoint, live: []const SSMCacheEntry, s: mlx.mlx_stream, mode: QsaAttachMode) !void {
    if (cps.len == 0) return;
    const cp = &cps[cps.len - 1];
    if (cp.layers.len != live.len) return error.SsmCheckpointLayerMismatch;
    const keep: c_int = @intCast(cp.pos);
    var copied: usize = 0;
    for (cp.layers, live) |*dst, src| {
        if (!ssmAuxIsQsaHistory(&src)) continue;
        const materialize = switch (mode) {
            .copy => true,
            .share => qsaHandoffMustMaterialize(&src, keep) or src.qsa_pooled_buf.ctx == null,
        };
        if (src.qsa_pooled.ctx != null) {
            if (dst.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(dst.qsa_pooled);
            if (materialize) {
                dst.qsa_pooled = try materializedOwnedCopy(s, src.qsa_pooled);
            } else {
                dst.qsa_pooled = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&dst.qsa_pooled, src.qsa_pooled));
            }
            try truncatePooled(&dst.qsa_pooled, keep, src.qsa_ratio, s, materialize);
        }
        dst.qsa_ratio = src.qsa_ratio;
        copied += 1;
    }
    if (copied == 0) return;
    recordQsaRowsOnCheckpoints(cps, live);
    keepOnlyLatestQsaHistory(cps);
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var n: usize = 0;
    for (cp.layers) |*l| {
        if (!snapshotHasQsaHistory(l)) continue;
        inline for (.{ l.aux_state, l.qsa_pooled }) |arr| {
            if (arr.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(vec, arr);
                n += 1;
            }
        }
    }
    if (n > 0) _ = mlx.mlx_eval(vec);
}

fn recordQsaRowsOnCheckpoints(cps: []SSMCheckpoint, live: []const SSMCacheEntry) void {
    for (cps) |*cp| {
        if (cp.layers.len != live.len) continue;
        const pos_rows: c_int = @intCast(cp.pos);
        for (cp.layers, live) |*dst, src| {
            if (!ssmAuxIsQsaHistory(&src)) continue;
            if (dst.qsa_rows == 0) dst.qsa_rows = pos_rows;
            dst.qsa_ratio = src.qsa_ratio;
        }
    }
}

/// Keep the QSA history on the NEWEST snap that carries it and drop it from
/// every older one. A prefix-extend merge inherits the previous turn's latest
/// snap, so without this an agent session holds one full copy per turn.
pub fn keepOnlyLatestQsaHistory(cps: []SSMCheckpoint) void {
    var i = cps.len;
    var kept = false;
    while (i > 0) {
        i -= 1;
        if (!checkpointHasQsaPooled(&cps[i])) continue;
        if (!kept) {
            kept = true;
            continue;
        }
        for (cps[i].layers) |*l| {
            if (l.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(l.qsa_pooled);
            l.qsa_pooled = .{ .ctx = null };
        }
    }
}

/// Any live QSA layer holds indexer history. On a QSA arch a restore that
/// leaves this false cannot prefill (QsaHistoryGap) — the cache treats it
/// as a miss.
pub fn entriesHaveQsaHistory(entries: []const SSMCacheEntry) bool {
    for (entries) |*e| {
        if (ssmAuxIsQsaHistory(e)) return true;
    }
    return false;
}

/// Overlay QSA history from `src_cp` (the entry's latest snap, full length)
/// onto `entries` already restored to `pos`. Slices aux to `pos` rows and
/// pooled to `pos/ratio` blocks. No-op on PLE/GDN layers (they already
/// hold the per-position window from `restoreSsmCheckpoint`).
/// `materialize` false = the destination holds a slice of `src_cp`'s bank (safe when the
/// checkpoint outlives the slot's first append); true = an owned copy, required when the
/// source is a transient such as the SSD tier's overlay.
pub fn applyQsaHistoryAt(entries: []SSMCacheEntry, src_cp: *const SSMCheckpoint, pos: usize, s: mlx.mlx_stream, materialize: bool) !void {
    if (entries.len != src_cp.layers.len) return error.SsmCheckpointLayerMismatch;
    const keep: c_int = @intCast(pos);
    for (entries, src_cp.layers) |*dst, src| {
        if (!snapshotHasQsaHistory(&src)) continue;
        // After restoreSsmCheckpoint a QSA layer has no conv window.
        if (dst.conv_state.ctx != null and mlx.mlx_array_size(dst.conv_state) > 0) continue;
        const src_hist: c_int = if (src.qsa_rows > 0) src.qsa_rows else blk: {
            if (src.qsa_pooled.ctx == null) break :blk 0;
            break :blk keep;
        };
        if (src_hist < keep) return error.QsaHistoryGap;
        const own = dst.aux_state;
        dst.aux_state = .{ .ctx = null };
        if (dst.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(dst.qsa_pooled);
        dst.qsa_pooled = .{ .ctx = null };
        const ratio = @max(src.qsa_ratio, 1);
        if (src.qsa_pooled.ctx != null) {
            const ps = mlx.getShape(src.qsa_pooled);
            const need_blocks = @divTrunc(keep, ratio);
            if (ps.len < 2 or ps[1] < need_blocks) return error.QsaHistoryGap;
            if (materialize) {
                dst.qsa_pooled = try materializedOwnedCopy(s, src.qsa_pooled);
            } else {
                dst.qsa_pooled = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&dst.qsa_pooled, src.qsa_pooled));
            }
            try truncatePooled(&dst.qsa_pooled, keep, ratio, s, materialize);
        }
        const from_src = try qsaLeftoverAt(src.aux_state, src_hist, keep, ratio, s);
        if (from_src.ctx != null) {
            if (own.ctx != null) _ = mlx.mlx_array_free(own);
            dst.aux_state = from_src;
        } else {
            dst.aux_state = if (own.ctx != null) own else .{ .ctx = null };
        }
        dst.qsa_ratio = src.qsa_ratio;
        dst.qsa_hist_rows = keep;
        dst.qsa_key_rows = if (dst.aux_state.ctx != null) mlx.getShape(dst.aux_state)[1] else 0;
        if (dst.qsa_pooled.ctx != null) {
            const ps = mlx.getShape(dst.qsa_pooled);
            dst.qsa_pooled_blocks = if (ps.len >= 2) ps[1] else 0;
        }
    }
}

/// Copy QSA aux/pooled from `src` onto `dst`, sliced to `pos` rows. Used when
/// a byte-budget trim would drop the latest snap — the only one that carries
/// the indexer history. A 122880-token trim that freed that snap left aux
/// empty; the next prefill then reshaped a 4096-row chunk into
/// `(1, nb, 4, 128)` and aborted (MLX reshape, 77% RAM — not an OOM).
/// `pos` is `dst.pos`: a checkpoint's aux is normalized to exactly `pos % ratio` rows at
/// capture, which is what the keep-own arm relies on when the source cannot supply them.
pub fn sliceQsaHistoryOntoCheckpoint(dst: *SSMCheckpoint, src: *const SSMCheckpoint, pos: usize, s: mlx.mlx_stream) !void {
    if (dst.layers.len != src.layers.len) return error.SsmCheckpointLayerMismatch;
    const keep: c_int = @intCast(pos);
    for (dst.layers, src.layers) |*d, s_l| {
        if (!snapshotHasQsaHistory(&s_l)) continue;
        if (d.conv_state.ctx != null and mlx.mlx_array_size(d.conv_state) > 0) continue;
        const ratio = @max(s_l.qsa_ratio, 1);
        if (s_l.qsa_pooled.ctx != null) {
            const pooled = try materializedOwnedCopy(s, s_l.qsa_pooled);
            if (d.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(d.qsa_pooled);
            d.qsa_pooled = pooled;
            try truncatePooled(&d.qsa_pooled, keep, ratio, s, true);
        }
        const src_hist: c_int = if (s_l.qsa_rows > 0) s_l.qsa_rows else keep;
        const leftover_n = @mod(keep, ratio);
        const leftover = try qsaLeftoverAt(s_l.aux_state, src_hist, keep, ratio, s);
        if (leftover.ctx != null) {
            if (d.aux_state.ctx != null) _ = mlx.mlx_array_free(d.aux_state);
            d.aux_state = leftover;
        } else if (leftover_n == 0) {
            if (d.aux_state.ctx != null) _ = mlx.mlx_array_free(d.aux_state);
            d.aux_state = .{ .ctx = null };
        }
        d.qsa_ratio = s_l.qsa_ratio;
        d.qsa_rows = keep;
    }
    // Materialize now: an unevaluated copy node keeps `src`'s buffer alive.
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var n: usize = 0;
    for (dst.layers) |*d| {
        if (!snapshotHasQsaHistory(d)) continue;
        inline for (.{ d.aux_state, d.qsa_pooled }) |arr| {
            if (arr.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(vec, arr);
                n += 1;
            }
        }
    }
    if (n > 0) _ = mlx.mlx_eval(vec);
}

test "checkQsaCompressRatio: a ratio the ring cannot hold is refused by name" {
    try std.testing.expectError(error.UnsupportedQsaCompressRatio, checkQsaCompressRatio(0));
    try std.testing.expectError(error.UnsupportedQsaCompressRatio, checkQsaCompressRatio(@intCast(QSA_MAX_COMPRESS_RATIO + 1)));
    try checkQsaCompressRatio(@intCast(QSA_MAX_COMPRESS_RATIO));
}

test "a PLE rollback leaves no freed-but-non-null QSA handle behind" {
    // The PLE arm never assigns `qsa_pooled` after `ssmFreeQsaState`; this pins that the helper nulls what it frees.
    const t = std.testing;
    const s = mlx.gpuStream();
    var e: SSMCacheEntry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    defer {
        ssmFreeSpecCapture(&e);
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
        ssmFreeQsaState(&e);
    }

    const state_len: c_int = 2;
    const verify_len: c_int = 3;
    const width: c_int = 4;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    const n: f64 = @floatFromInt((state_len + verify_len) * width);
    try mlx.check(mlx.mlx_arange(&flat, 0.0, n, 1.0, .float32, s));
    const shape = [_]c_int{ 1, state_len + verify_len, width };
    try mlx.check(mlx.mlx_reshape(&e.spec_ple_input, flat, &shape, 3, s));
    e.spec_ple_len = @intCast(state_len + verify_len);

    // The impossible-today entry: a PLE window AND a QSA pooled bank on the same layer.
    var pflat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pflat);
    try mlx.check(mlx.mlx_arange(&pflat, 0.0, 8.0, 1.0, .float32, s));
    const pshape = [_]c_int{ 1, 2, 4 };
    try mlx.check(mlx.mlx_reshape(&e.qsa_pooled, pflat, &pshape, 3, s));
    try mlx.check(mlx.mlx_reshape(&e.qsa_key_buf, pflat, &pshape, 3, s));
    e.qsa_key_rows = 2;
    e.qsa_pooled_blocks = 2;

    try ssmRollbackFromCapture(&e, 1, @intCast(verify_len), s);

    try t.expect(e.aux_state.ctx != null);
    const got = mlx.getShape(e.aux_state);
    try t.expectEqual(state_len, got[1]);
    try t.expectEqual(width, got[2]);
    // Every QSA handle the arm dropped is null, not dangling.
    try t.expect(e.qsa_pooled.ctx == null);
    try t.expect(e.qsa_key_buf.ctx == null);
    try t.expect(e.qsa_pooled_buf.ctx == null);
    try t.expect(e.qsa_score_bank.ctx == null);
    try t.expect(e.qsa_score_buf.ctx == null);
    try t.expectEqual(@as(c_int, 0), e.qsa_key_rows);
    try t.expectEqual(@as(c_int, 0), e.qsa_pooled_blocks);
}

test "ssmRollbackFromCapture: padded verify_len keeps PLE tokens inside the real prefix" {
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    var e: SSMCacheEntry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    defer {
        ssmFreeSpecCapture(&e);
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
        ssmFreeQsaState(&e);
    }
    const state_len: c_int = 2;
    const m: u32 = 2;
    const verify_len: c_int = 4;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    const n: f64 = @floatFromInt((state_len + verify_len) * 3);
    try mlx.check(mlx.mlx_arange(&flat, 0.0, n, 1.0, .float32, s));
    const shape = [_]c_int{ 1, state_len + verify_len, 3 };
    try mlx.check(mlx.mlx_reshape(&e.spec_ple_input, flat, &shape, 3, s));
    e.spec_ple_tokens = .{ 9, 8, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    e.spec_ple_len = @intCast(state_len + verify_len);
    try ssmRollbackFromCapture(&e, m, @intCast(verify_len), s);
    try t.expect(e.aux_state.ctx != null);
    try t.expectEqual(state_len, mlx.getShape(e.aux_state)[1]);
    try t.expectEqual(@as(u32, 2), e.ple_prev[0]);
    try t.expectEqual(@as(u32, 3), e.ple_prev[1]);
    try t.expect(e.ple_prev_valid);
}

test "QSA key history advances by S not 1" {
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.allocator = t.allocator;
    var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&entry);
    entry.qsa_ratio = 4;
    var prng = std.Random.DefaultPrng.init(0x51A0_0004);
    const chunk = try attn256RandBf16(prng.random(), &[_]c_int{ 1, 4, 16 }, s);
    defer _ = mlx.mlx_array_free(chunk);
    try xfm.qsaAppendKeys(&entry, chunk, 0);
    try t.expectEqual(@as(c_int, 4), qsaHistoryRows(&entry));
}

test "ssmEntryBytes bills each QSA capacity buffer once, never the view beside it" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const zeros = struct {
        fn make(shape: []const c_int, st: mlx.mlx_stream) mlx.mlx_array {
            var a = mlx.mlx_array_new();
            _ = mlx.mlx_zeros(&a, shape.ptr, shape.len, .bfloat16, st);
            return a;
        }
        fn rows(buf: mlx.mlx_array, n: c_int, st: mlx.mlx_stream) !mlx.mlx_array {
            const sh = mlx.getShape(buf);
            var v = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&v, buf, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ sh[0], n, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, st));
            return v;
        }
    };
    var e: SSMCacheEntry = .{ .conv_state = zeros.make(&.{ 1, 3, 8 }, s), .ssm_state = zeros.make(&.{ 1, 2, 4, 4 }, s), .initialized = true };
    defer _ = mlx.mlx_array_free(e.conv_state);
    defer _ = mlx.mlx_array_free(e.ssm_state);
    e.qsa_key_buf = zeros.make(&.{ 1, 64, 16 }, s);
    defer _ = mlx.mlx_array_free(e.qsa_key_buf);
    e.aux_state = try zeros.rows(e.qsa_key_buf, 10, s);
    defer _ = mlx.mlx_array_free(e.aux_state);
    e.qsa_pooled_buf = zeros.make(&.{ 1, 32, 16 }, s);
    defer _ = mlx.mlx_array_free(e.qsa_pooled_buf);
    e.qsa_pooled = try zeros.rows(e.qsa_pooled_buf, 2, s);
    defer _ = mlx.mlx_array_free(e.qsa_pooled);
    const two: u64 = 2; // bf16
    try testing.expectEqual(two * (3 * 8 + 2 * 4 * 4 + 64 * 16 + 32 * 16), ssmEntryBytes(&e));

    // A restored entry holds only the tight arrays until its first append grows the buffers.
    var restored: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    restored.aux_state = e.aux_state;
    restored.qsa_pooled = e.qsa_pooled;
    try testing.expectEqual(two * (10 * 16 + 2 * 16), ssmEntryBytes(&restored));
}

test "qsa raw-key ring is O(RING_ROWS) at N rows" {
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.allocator = t.allocator;

    var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&entry);
    entry.qsa_ratio = 4;

    var prng = std.Random.DefaultPrng.init(0x51A0_B17E);
    const rnd = prng.random();
    const n: c_int = 256;
    const hd: c_int = 16;
    try t.expect(n > QSA_RING_ROWS);

    const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, n, hd }, s);
    defer _ = mlx.mlx_array_free(chunk);
    try xfm.qsaAppendKeys(&entry, chunk, 0);
    try mlx.check(mlx.mlx_array_eval(entry.qsa_key_buf));

    const raw = qsaRawKeyBytes(&entry);
    const row: u64 = @as(u64, @intCast(hd)) * @as(u64, mlx.mlx_array_itemsize(entry.qsa_key_buf));
    const ring_cap = @as(u64, @intCast(QSA_RING_ROWS)) * row;
    try t.expectEqual(ring_cap, raw);
}

test "qsa leftover: restore at L-30 then append/rollback re-pools bit-identically to a full bank" {
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.allocator = t.allocator;

    var prng = std.Random.DefaultPrng.init(0x1E40_7E11);
    const rnd = prng.random();
    const hd: c_int = 16;
    const ratio: c_int = 4;
    const L: c_int = 37;
    const pos: c_int = L - 30;
    try t.expect(pos > 0 and @mod(pos, ratio) != 0);

    const keys = try attn256RandBf16(rnd, &[_]c_int{ 1, L, hd }, s);
    defer _ = mlx.mlx_array_free(keys);

    var live_arr = [_]SSMCacheEntry{.{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio }};
    defer ssmFreeQsaState(&live_arr[0]);
    {
        var first = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(first);
        try mlx.check(mlx.mlx_slice(&first, keys, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, pos, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try qsaTestAppendPool(&xfm, &live_arr[0], first, 0, s);
    }
    var cps: [2]SSMCheckpoint = undefined;
    cps[0] = try captureSsmCheckpoint(t.allocator, &live_arr, @intCast(pos), s);
    {
        var rest = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rest);
        try mlx.check(mlx.mlx_slice(&rest, keys, &[_]c_int{ 0, pos, 0 }, 3, &[_]c_int{ 1, L, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try qsaTestAppendPool(&xfm, &live_arr[0], rest, pos, s);
    }
    cps[1] = try captureSsmCheckpoint(t.allocator, &live_arr, @intCast(L), s);
    defer for (&cps) |*c| c.deinit(t.allocator);
    try attachQsaHistoryToLatest(&cps, &live_arr, s);

    var dest_arr = [_]SSMCacheEntry{.{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true }};
    defer ssmFreeQsaState(&dest_arr[0]);
    try restoreSsmCheckpoint(&dest_arr, &cps[0]);
    try applyQsaHistoryAt(&dest_arr, &cps[1], @intCast(pos), s, false);

    // The restore carries the raw rows of the block `pos` reopened, and nothing else.
    const tail: c_int = @mod(pos, ratio);
    try t.expectEqual(tail, mlx.getShape(dest_arr[0].aux_state)[1]);
    {
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_slice(&want, keys, &[_]c_int{ 0, pos - tail, 0 }, 3, &[_]c_int{ 1, pos, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dest_arr[0].aux_state, want, s));
    }

    const verify_len: c_int = 4;
    {
        var verify = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify);
        try mlx.check(mlx.mlx_slice(&verify, keys, &[_]c_int{ 0, pos, 0 }, 3, &[_]c_int{ 1, pos + verify_len, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try qsaTestAppendPool(&xfm, &dest_arr[0], verify, pos, s);
    }
    // The block the verify closed was pooled from leftover + chunk: it must already be the
    // full bank's block, before the rollback gets a chance to drop it.
    {
        const closed = @divTrunc(pos + verify_len, ratio);
        try t.expect(closed > @divTrunc(pos, ratio));
        const ref_closed = try qsaTestMeanBlocks(keys, 0, closed, ratio, s);
        defer _ = mlx.mlx_array_free(ref_closed);
        try t.expectEqual(closed, dest_arr[0].qsa_pooled_blocks);
        try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dest_arr[0].qsa_pooled, ref_closed, s));
    }
    try ssmRollbackFromCapture(&dest_arr[0], 1, @intCast(verify_len), s);
    const keep: c_int = pos + 1 + 1;
    try t.expectEqual(keep, dest_arr[0].qsa_hist_rows);
    {
        var again = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(again);
        try mlx.check(mlx.mlx_slice(&again, keys, &[_]c_int{ 0, keep, 0 }, 3, &[_]c_int{ 1, keep + 3, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try qsaTestAppendPool(&xfm, &dest_arr[0], again, keep, s);
    }
    const final_len = keep + 3;
    const n_blocks = @divTrunc(final_len, ratio);
    const ref = try qsaTestMeanBlocks(keys, 0, n_blocks, ratio, s);
    defer _ = mlx.mlx_array_free(ref);
    try t.expectEqual(n_blocks, dest_arr[0].qsa_pooled_blocks);
    try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dest_arr[0].qsa_pooled, ref, s));
}

test "qsa rollback: a partial accept keeps the raw-key buffer and re-slices to keep" {
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.allocator = t.allocator;
    xfm.qsa_score_bank_builds = 0;

    var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&entry);
    entry.qsa_ratio = 4;
    entry.qsa_reserve_rows = 64;

    var prng = std.Random.DefaultPrng.init(0x51A0_B0FF);
    const rnd = prng.random();
    const hd: c_int = 16;
    const prefix: c_int = 8;
    const verify_len: c_int = 4;
    const accepted: u32 = 1;
    const keep: c_int = prefix + @as(c_int, @intCast(1 + accepted));
    const keep_blocks: c_int = @divTrunc(keep, entry.qsa_ratio);

    {
        const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, prefix, hd }, s);
        defer _ = mlx.mlx_array_free(chunk);
        try xfm.qsaAppendKeys(&entry, chunk, 0);
        const blocks = try attn256RandBf16(rnd, &[_]c_int{ 1, @divTrunc(prefix, entry.qsa_ratio), hd }, s);
        defer _ = mlx.mlx_array_free(blocks);
        try xfm.qsaAppendPooled(&entry, blocks, 0);
    }
    {
        const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, verify_len, hd }, s);
        defer _ = mlx.mlx_array_free(chunk);
        try xfm.qsaAppendKeys(&entry, chunk, prefix);
        const blocks = try attn256RandBf16(rnd, &[_]c_int{ 1, @divTrunc(verify_len, entry.qsa_ratio), hd }, s);
        defer _ = mlx.mlx_array_free(blocks);
        try xfm.qsaAppendPooled(&entry, blocks, @divTrunc(prefix, entry.qsa_ratio));
    }
    try t.expectEqual(prefix + verify_len, entry.qsa_key_rows);
    try t.expect(entry.qsa_key_buf.ctx != null);
    _ = try xfm.qsaScoreBank(&entry, 1, @divTrunc(prefix + verify_len, entry.qsa_ratio), hd);

    const kept_ref = blk: {
        const ks = mlx.getShape(entry.aux_state);
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(
            &view,
            entry.aux_state,
            &[_]c_int{ 0, 0, 0 },
            3,
            &[_]c_int{ ks[0], keep, ks[2] },
            3,
            &[_]c_int{ 1, 1, 1 },
            3,
            s,
        ));
        break :blk try materializedOwnedCopy(s, view);
    };
    defer _ = mlx.mlx_array_free(kept_ref);

    const kept_pooled = blk: {
        const ps = mlx.getShape(entry.qsa_pooled);
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(
            &view,
            entry.qsa_pooled,
            &[_]c_int{ 0, 0, 0 },
            3,
            &[_]c_int{ ps[0], keep_blocks, ps[2] },
            3,
            &[_]c_int{ 1, 1, 1 },
            3,
            s,
        ));
        break :blk try materializedOwnedCopy(s, view);
    };
    defer _ = mlx.mlx_array_free(kept_pooled);

    const buf_before = entry.qsa_key_buf.ctx;
    const pooled_buf_before = entry.qsa_pooled_buf.ctx;
    try ssmRollbackFromCapture(&entry, accepted, @intCast(verify_len), s);

    try t.expect(entry.qsa_key_buf.ctx == buf_before);
    try t.expectEqual(keep, entry.qsa_key_rows);
    try t.expectEqual(keep, mlx.getShape(entry.aux_state)[1]);
    try t.expect(entry.qsa_pooled_buf.ctx == pooled_buf_before);
    try t.expectEqual(keep_blocks, entry.qsa_pooled_blocks);
    try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(entry.qsa_pooled, kept_pooled, s));
    // The score bank is invalidated rather than truncated: its columns are a function of
    // the pooled bank and the next completed block rebuilds them.
    try t.expect(entry.qsa_score_bank.ctx == null);
    try t.expectEqual(@as(c_int, 0), entry.qsa_score_blocks);

    const tail = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, hd }, s);
    defer _ = mlx.mlx_array_free(tail);
    var reference = mlx.mlx_array_new();
    {
        const parts = [_]mlx.mlx_array{ kept_ref, tail };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&reference, vec, 1, s));
    }
    defer _ = mlx.mlx_array_free(reference);

    try xfm.qsaAppendKeys(&entry, tail, keep);
    try t.expectEqual(keep + 2, mlx.getShape(entry.aux_state)[1]);
    try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(entry.aux_state, reference, s));
}

test "qsa rollback: a rollback that reaches past the ring is a named error, not a negative slice" {
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    const hd: c_int = 16;
    const ratio: c_int = 4;

    // A restored entry holds only the leftover at its position — no capacity buffer, so the
    // reslice declines and the copy path is what a too-wide verify reaches.
    var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio };
    defer ssmFreeQsaState(&entry);
    entry.aux_state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_ones(&entry.aux_state, &[_]c_int{ 1, 2, hd }, 3, .bfloat16, s));
    entry.qsa_pooled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_ones(&entry.qsa_pooled, &[_]c_int{ 1, 11, hd }, 3, .bfloat16, s));
    entry.qsa_key_rows = 2;
    entry.qsa_pooled_blocks = 11;
    entry.qsa_hist_rows = 46;

    try t.expectError(error.QsaRollbackBeyondRing, ssmRollbackFromCapture(&entry, 0, 16, s));
}

test "qwen4 MTP head: a clamp below the raw ring is named, not an emptied history" {
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.allocator = t.allocator;

    const hd: c_int = 16;
    const ratio: c_int = 4;
    // The live warm turn: the head commits at L, the trunk restores at its checkpoint
    // (SSM_SNAPSHOT_BACKOFF below the prompt end, plus the tokens this turn generated), and
    // `qwen4MtpAdopt` clamps the head to that. The gap is wider than the ring and mid-block.
    const L: c_int = 200;
    const want: c_int = 43;
    try t.expect(L - want > QSA_RING_ROWS and @mod(want, ratio) != 0);

    var prng = std.Random.DefaultPrng.init(0x4D7F_1105);
    const keys = try attn256RandBf16(prng.random(), &[_]c_int{ 1, L, hd }, s);
    defer _ = mlx.mlx_array_free(keys);

    var live: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio };
    defer ssmFreeQsaState(&live);
    try qsaTestAppendPool(&xfm, &live, keys, 0, s);

    // What `commitWithState` stores for the head, and what `qwen4MtpAdopt` seeds it back from.
    var head_aux = ssmSnapshot(&live);
    defer ssmSnapshotDeinit(&head_aux);

    const head_cache = try KVCache.init(t.allocator, 1);
    var head: Qwen4Mtp = undefined;
    head.cache = head_cache;
    head.entry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    head.qsa_marks = .{};
    head.seq_offset = 0;
    head.pos_base = 0;
    xfm.qwen4_mtp = head;
    const m = &xfm.qwen4_mtp.?;
    defer {
        ssmFreeQsaState(&m.entry);
        m.qsa_marks.deinit();
        m.cache.deinit();
    }

    try ssmRestore(&m.entry, &head_aux);
    m.seq_offset = @intCast(L);
    m.cache.step = @intCast(L);
    try t.expectError(error.MtpHeadQsaRingGap, xfm.qwen4MtpTruncate(@intCast(want)));

    // The guard is not the ring: a clamp the ring CAN serve still trims, and the append after
    // it re-pools from that leftover exactly as a full bank would.
    const near: c_int = L - 5;
    try t.expect(@mod(near, ratio) != 0);
    try ssmRestore(&m.entry, &head_aux);
    m.seq_offset = @intCast(L);
    m.cache.step = @intCast(L);
    try xfm.qwen4MtpTruncate(@intCast(near));
    try t.expectEqual(near, m.entry.qsa_hist_rows);
    {
        var again = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(again);
        try mlx.check(mlx.mlx_slice(&again, keys, &[_]c_int{ 0, near, 0 }, 3, &[_]c_int{ 1, L, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try qsaTestAppendPool(&xfm, &m.entry, again, near, s);
    }
    const n_blocks = @divTrunc(L, ratio);
    const ref = try qsaTestMeanBlocks(keys, 0, n_blocks, ratio, s);
    defer _ = mlx.mlx_array_free(ref);
    try t.expectEqual(n_blocks, m.entry.qsa_pooled_blocks);
    try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.entry.qsa_pooled, ref, s));
}

test "qwen4 MTP head: a warm clamp restores exactly from the leftover marked at the trunk's checkpoint" {
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.allocator = t.allocator;

    const hd: c_int = 16;
    const ratio: c_int = 4;
    // The live warm turn: the trunk checkpoints at a stride boundary and at its always-on
    // backoff position, and the head runs on to L (prompt end plus the generated tail). Both
    // clamps are mid-block and more than QSA_RING_ROWS below the head's committed length —
    // the second is the live gap (SSM_SNAPSHOT_BACKOFF 30 + 124 generated).
    const L: c_int = 400;
    const cp_lo: c_int = 46;
    const cp_hi: c_int = L - 154;
    inline for (.{ cp_lo, cp_hi }) |cp| {
        try t.expect(L - cp > QSA_RING_ROWS and @mod(cp, ratio) != 0);
    }

    var prng = std.Random.DefaultPrng.init(0x51A_C0DE);
    const keys = try attn256RandBf16(prng.random(), &[_]c_int{ 1, L, hd }, s);
    defer _ = mlx.mlx_array_free(keys);

    const head_cache = try KVCache.init(t.allocator, 1);
    var head: Qwen4Mtp = undefined;
    head.cache = head_cache;
    head.entry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio };
    head.qsa_marks = .{};
    head.seq_offset = 0;
    head.pos_base = 0;
    xfm.qwen4_mtp = head;
    const m = &xfm.qwen4_mtp.?;
    defer {
        ssmFreeQsaState(&m.entry);
        m.qsa_marks.deinit();
        m.cache.deinit();
    }

    // The head appends in chunks and the trunk checkpoints at the end of each, exactly as the
    // prefill loop does: the mark is taken while the ring still ends at that row.
    var from: c_int = 0;
    inline for (.{ cp_lo, cp_hi, L }) |to| {
        var chunk = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(chunk);
        try mlx.check(mlx.mlx_slice(&chunk, keys, &[_]c_int{ 0, from, 0 }, 3, &[_]c_int{ 1, to, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try qsaTestAppendPool(&xfm, &m.entry, chunk, from, s);
        m.seq_offset = @intCast(to);
        if (to != L) try xfm.qwen4MtpMarkQsaLeftover(@intCast(to));
        from = to;
    }

    // What `commitWithState` stores for the head, and what `qwen4MtpAdopt` seeds it back from.
    var head_aux = ssmSnapshot(&m.entry);
    defer ssmSnapshotDeinit(&head_aux);
    var committed_marks = QsaHeadMarkSet.share(xfm.qwen4MtpMarks());
    defer committed_marks.deinit();
    try t.expectEqual(@as(usize, 2), committed_marks.len);

    inline for (.{ cp_hi, cp_lo }) |want| {
        // A head that appended straight to `want` — the state the clamp must reproduce.
        var refe: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio };
        defer ssmFreeQsaState(&refe);
        {
            var upto = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(upto);
            try mlx.check(mlx.mlx_slice(&upto, keys, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, want, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            try qsaTestAppendPool(&xfm, &refe, upto, 0, s);
        }

        try ssmRestore(&m.entry, &head_aux);
        m.qsa_marks.deinit();
        m.qsa_marks = QsaHeadMarkSet.share(committed_marks.slice());
        m.seq_offset = @intCast(L);
        m.cache.step = @intCast(L);
        try xfm.qwen4MtpTruncate(@intCast(want));

        try t.expectEqual(@as(usize, @intCast(want)), m.seq_offset);
        try t.expectEqual(want, m.entry.qsa_hist_rows);
        try t.expectEqual(@mod(want, ratio), mlx.getShape(m.entry.aux_state)[1]);
        const ref_lv = try qsaLeftoverAt(refe.aux_state, qsaHistoryRows(&refe), want, ratio, s);
        defer _ = mlx.mlx_array_free(ref_lv);
        try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.entry.aux_state, ref_lv, s));
        try t.expectEqual(@divTrunc(want, ratio), mlx.getShape(m.entry.qsa_pooled)[1]);
        try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.entry.qsa_pooled, refe.qsa_pooled, s));

        // ...and the leftover is live: the next append re-pools through it to the full bank.
        {
            var rest = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(rest);
            try mlx.check(mlx.mlx_slice(&rest, keys, &[_]c_int{ 0, want, 0 }, 3, &[_]c_int{ 1, L, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            try qsaTestAppendPool(&xfm, &m.entry, rest, want, s);
        }
        const full = try qsaTestMeanBlocks(keys, 0, @divTrunc(L, ratio), ratio, s);
        defer _ = mlx.mlx_array_free(full);
        try t.expectEqual(@divTrunc(L, ratio), m.entry.qsa_pooled_blocks);
        try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.entry.qsa_pooled, full, s));
    }
}

test "qwen4 MTP head: a clamp whose retained ring is shorter than the kept leftover is not resliced" {
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.allocator = t.allocator;

    const hd: c_int = 16;
    const ratio: c_int = 4;
    // The band a live head's own ring can hold and still not serve: a clamp of nearly the
    // whole ring leaves fewer rows than a mid-block `keep` needs re-pooled. Its low end
    // (`QSA_RING_ROWS - ratio + 1`) the ring still serves — the same clamp must be exact
    // either way.
    const keep: c_int = 199;
    try t.expect(@mod(keep, ratio) == ratio - 1);

    var prng = std.Random.DefaultPrng.init(0xBA4D_0007);
    var drop_tail: c_int = QSA_RING_ROWS - ratio + 1;
    while (drop_tail <= QSA_RING_ROWS) : (drop_tail += 1) {
        const L: c_int = keep + drop_tail;
        const keys = try attn256RandBf16(prng.random(), &[_]c_int{ 1, L, hd }, s);
        defer _ = mlx.mlx_array_free(keys);
        const ref_pooled = try qsaTestMeanBlocks(keys, 0, @divTrunc(keep, ratio), ratio, s);
        defer _ = mlx.mlx_array_free(ref_pooled);
        const full = try qsaTestMeanBlocks(keys, 0, @divTrunc(L, ratio), ratio, s);
        defer _ = mlx.mlx_array_free(full);
        const ref_lv = try qsaLeftoverAt(keys, L, keep, ratio, s);
        defer _ = mlx.mlx_array_free(ref_lv);

        for ([_]bool{ true, false }) |with_mark| {
            const head_cache = try KVCache.init(t.allocator, 1);
            var head: Qwen4Mtp = undefined;
            head.cache = head_cache;
            head.entry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio };
            head.qsa_marks = .{};
            head.seq_offset = 0;
            head.pos_base = 0;
            xfm.qwen4_mtp = head;
            const m = &xfm.qwen4_mtp.?;
            defer {
                ssmFreeQsaState(&m.entry);
                m.qsa_marks.deinit();
                m.cache.deinit();
            }

            // The head runs live to L — its ring and capacity buffer are its own, the state a
            // commit-time trim clamps — and the trunk checkpoints at `keep` on the way.
            var from: c_int = 0;
            for ([_]c_int{ keep, L }) |to| {
                var chunk = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(chunk);
                try mlx.check(mlx.mlx_slice(&chunk, keys, &[_]c_int{ 0, from, 0 }, 3, &[_]c_int{ 1, to, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
                try qsaTestAppendPool(&xfm, &m.entry, chunk, from, s);
                m.seq_offset = @intCast(to);
                if (to == keep and with_mark) try xfm.qwen4MtpMarkQsaLeftover(@intCast(keep));
                from = to;
            }
            m.cache.step = @intCast(L);
            try t.expectEqual(QSA_RING_ROWS, mlx.getShape(m.entry.aux_state)[1]);

            const ring_serves = QSA_RING_ROWS - drop_tail >= @mod(keep, ratio);
            if (!with_mark and !ring_serves) {
                try t.expectError(error.MtpHeadQsaRingGap, xfm.qwen4MtpTruncate(@intCast(keep)));
                continue;
            }
            try xfm.qwen4MtpTruncate(@intCast(keep));
            try t.expectEqual(keep, m.entry.qsa_hist_rows);
            try t.expect(mlx.getShape(m.entry.aux_state)[1] >= @mod(keep, ratio));
            const got_lv = try qsaLeftoverAt(m.entry.aux_state, qsaHistoryRows(&m.entry), keep, ratio, s);
            defer _ = mlx.mlx_array_free(got_lv);
            try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got_lv, ref_lv, s));
            try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.entry.qsa_pooled, ref_pooled, s));

            // ...and the leftover is live: the append after the clamp re-pools through it.
            {
                var rest = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(rest);
                try mlx.check(mlx.mlx_slice(&rest, keys, &[_]c_int{ 0, keep, 0 }, 3, &[_]c_int{ 1, L, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
                try qsaTestAppendPool(&xfm, &m.entry, rest, keep, s);
            }
            try t.expectEqual(@divTrunc(L, ratio), m.entry.qsa_pooled_blocks);
            try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.entry.qsa_pooled, full, s));
        }
    }
}

test "checkpointQsaCoversPos: a bank-less leftover does not cover pos" {
    const t = std.testing;
    const s = mlx.gpuStream();
    var leftover = try qsaCoverTestCp(1, 0, 17, 4, 8, s);
    defer leftover.deinit(t.allocator);
    try t.expect(!checkpointQsaCoversPos(&leftover, &leftover, 17));
    try t.expect(!qsaRestoreSatisfiesForward(&.{
        .{
            .conv_state = .{ .ctx = null },
            .ssm_state = .{ .ctx = null },
            .aux_state = leftover.layers[0].aux_state,
            .qsa_pooled = .{ .ctx = null },
            .initialized = true,
            .qsa_ratio = 4,
            .qsa_hist_rows = 17,
        },
    }, 17));
}

test "checkpointHasQsaPooled: a mid-block leftover is not the pooled bank's home" {
    const t = std.testing;
    const s = mlx.gpuStream();
    // Every site that asks "does the indexer bank live on this checkpoint" must say no to a
    // leftover-only snap, or a warm turn at L-30 never gets the overlay and cold-prefills.
    var leftover_only = try qsaCoverTestCp(2, 0, 46, 4, 8, s);
    defer leftover_only.deinit(t.allocator);
    var with_bank = try qsaCoverTestCp(0, 16, 64, 4, 8, s);
    defer with_bank.deinit(t.allocator);
    try t.expect(checkpointHasQsaHistory(&leftover_only));
    try t.expect(!checkpointHasQsaPooled(&leftover_only));
    try t.expect(checkpointHasQsaPooled(&with_bank));
}

test "checkpointQsaCoversPos: a checkpoint without the leftover for pos is a miss" {
    const t = std.testing;
    const s = mlx.gpuStream();
    const hd: c_int = 8;
    const ratio: c_int = 4;
    // The history source: 64 rows as 16 pooled blocks beside a raw ring that is the newest
    // QSA_RING_ROWS rows — a v7 file's aux, the leftover at no position in particular.
    var src = try qsaCoverTestCp(QSA_RING_ROWS, 16, 64, ratio, hd, s);
    defer src.deinit(t.allocator);
    var at48 = try qsaCoverTestCp(0, 0, 48, ratio, hd, s);
    defer at48.deinit(t.allocator);
    var at46 = try qsaCoverTestCp(2, 0, 46, ratio, hd, s);
    defer at46.deinit(t.allocator);
    var at46_bare = try qsaCoverTestCp(0, 0, 46, ratio, hd, s);
    defer at46_bare.deinit(t.allocator);

    try t.expect(checkpointQsaCoversPos(&at48, &src, 48));
    try t.expect(checkpointQsaCoversPos(&at46, &src, 46));
    try t.expect(checkpointQsaCoversPos(&src, &src, 46));
    // A checkpoint with no leftover of its own (a v7 snap) covers pos when the source still
    // physically holds those rows — the whole raw history in a v7 entry file, and here the
    // rows 44..46 the newest ring happens to span.
    try t.expect(checkpointQsaCoversPos(&at46_bare, &src, 46));
    var full = try qsaCoverTestCp(64, 16, 64, ratio, hd, s);
    defer full.deinit(t.allocator);
    try t.expect(checkpointQsaCoversPos(&at46_bare, &full, 46));
    // Rows the source dropped are a miss, never a silently wrong leftover.
    var ring_at_256 = try qsaCoverTestCp(QSA_RING_ROWS, 64, 256, ratio, hd, s);
    defer ring_at_256.deinit(t.allocator);
    try t.expect(!checkpointQsaCoversPos(&at46_bare, &ring_at_256, 46));
    // A source whose pooled bank stops short of pos is still a miss.
    var short = try qsaCoverTestCp(QSA_RING_ROWS, 8, 64, ratio, hd, s);
    defer short.deinit(t.allocator);
    try t.expect(!checkpointQsaCoversPos(&at46, &short, 46));
}

test "checkpointQsaCoversPos: a 32-row ring on the bank covers every residue at or below its pos" {
    const t = std.testing;
    const s = mlx.gpuStream();
    const hd: c_int = 8;
    const ratio: c_int = 4;
    var src = try qsaCoverTestCp(QSA_RING_ROWS, 5, 20, ratio, hd, s);
    defer src.deinit(t.allocator);
    var r: usize = 0;
    while (r < 4) : (r += 1) {
        const pos: usize = 16 + r;
        var cp = try qsaCoverTestCp(0, 0, @intCast(pos), ratio, hd, s);
        defer cp.deinit(t.allocator);
        try t.expect(checkpointQsaCoversPos(&cp, &src, pos));
    }
}

test "QSA checkpoint aux+pooled bytes are O(rows + checkpoints)" {
    const s = mlx.gpuStream();
    const n: c_int = 64;
    const hd: c_int = 8;
    const ratio: c_int = 4;
    const aux_shape = [_]c_int{ 1, n, hd };
    const pooled_shape = [_]c_int{ 1, @divExact(n, ratio), hd };
    var aux = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aux);
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, @floatFromInt(n * hd), 1.0, .float32, s));
        try mlx.check(mlx.mlx_reshape(&aux, flat, &aux_shape, 3, s));
        try mlx.check(mlx.mlx_array_eval(aux));
    }
    var pooled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled);
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_arange(&flat, 1000.0, 1000.0 + @as(f64, @floatFromInt((n / ratio) * hd)), 1.0, .float32, s));
        try mlx.check(mlx.mlx_reshape(&pooled, flat, &pooled_shape, 3, s));
        try mlx.check(mlx.mlx_array_eval(pooled));
    }
    const live: SSMCacheEntry = .{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
        .aux_state = aux,
        .qsa_pooled = pooled,
        .qsa_ratio = ratio,
    };
    aux = mlx.mlx_array_new();
    pooled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&aux, live.aux_state));
    try mlx.check(mlx.mlx_array_set(&pooled, live.qsa_pooled));
    var live_arr = [_]SSMCacheEntry{live};
    defer {
        if (live_arr[0].aux_state.ctx != null) _ = mlx.mlx_array_free(live_arr[0].aux_state);
        if (live_arr[0].qsa_pooled.ctx != null) _ = mlx.mlx_array_free(live_arr[0].qsa_pooled);
    }

    // 46 is mid-block on purpose: a restore there needs the two raw rows of the block it
    // reopens, so the leftover assertions below are not vacuous.
    const positions = [_]usize{ 16, 32, 46, 64 };
    var cps: [positions.len]SSMCheckpoint = undefined;
    for (positions, 0..) |p, i| {
        cps[i] = try captureSsmCheckpoint(testing.allocator, &live_arr, p, s);
    }
    defer for (&cps) |*c| c.deinit(testing.allocator);
    try attachQsaHistoryToLatest(&cps, &live_arr, s);

    const held = qsaHistoryBytesHeld(&cps);
    const one: u64 = @as(u64, @intCast(n)) * @as(u64, @intCast(hd)) * 4 +
        @as(u64, @intCast(@divExact(n, ratio))) * @as(u64, @intCast(hd)) * 4;
    try testing.expect(held <= one + @as(u64, positions.len) * 64);
    for (positions, 0..) |p, i| {
        try testing.expectEqual(@as(c_int, @intCast(p)), cps[i].layers[0].qsa_rows);
        try testing.expectEqual(@as(c_int, @intCast(p % @as(usize, @intCast(ratio)))), checkpointQsaLeftoverRows(&cps[i]));
    }
    try testing.expect(checkpointHasQsaHistory(&cps[positions.len - 1]));

    var dest = [_]SSMCacheEntry{.{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
    }};
    defer ssmFreeQsaState(&dest[0]);
    try restoreSsmCheckpoint(&dest, &cps[2]);
    try applyQsaHistoryAt(&dest, &cps[positions.len - 1], 46, s, false);
    // The leftover is the checkpoint's own raw rows 44 and 45, not the newest ring's overlap.
    try testing.expectEqual(@as(c_int, 2), mlx.getShape(dest[0].aux_state)[1]);
    {
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_slice(&want, aux, &[_]c_int{ 0, 44, 0 }, 3, &[_]c_int{ 1, 46, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dest[0].aux_state, want, s));
    }
    try testing.expectEqual(@as(c_int, 11), dest[0].qsa_pooled_blocks);
    try testing.expectEqual(@as(c_int, 46), dest[0].qsa_hist_rows);

    var t: Transformer = undefined;
    t.s = s;
    t.allocator = testing.allocator;
    const extra_shape = [_]c_int{ 1, 1, hd };
    var extra = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(extra);
    try mlx.check(mlx.mlx_ones(&extra, &extra_shape, 3, .float32, s));
    try t.qsaAppendKeys(&dest[0], extra, 46);
    try mlx.check(mlx.mlx_array_eval(dest[0].aux_state));
    try testing.expectEqual(@as(c_int, 3), mlx.getShape(dest[0].aux_state)[1]);
    try testing.expect(cps[positions.len - 1].layers[0].aux_state.ctx == null);
}

test "qsa_rows is the checkpoint position, never current live length" {
    const s = mlx.gpuStream();
    const hd: c_int = 8;
    const ratio: c_int = 4;
    const aux_shape = [_]c_int{ 1, 16, hd };
    var aux = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aux);
    try mlx.check(mlx.mlx_ones(&aux, &aux_shape, 3, .float32, s));
    var pooled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled);
    const pooled_shape = [_]c_int{ 1, 4, hd };
    try mlx.check(mlx.mlx_ones(&pooled, &pooled_shape, 3, .float32, s));
    var live16 = [_]SSMCacheEntry{.{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
        .aux_state = aux,
        .qsa_pooled = pooled,
        .qsa_ratio = ratio,
    }};
    aux = mlx.mlx_array_new();
    pooled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&aux, live16[0].aux_state));
    try mlx.check(mlx.mlx_array_set(&pooled, live16[0].qsa_pooled));
    defer ssmFreeQsaState(&live16[0]);

    var cp32 = try captureSsmCheckpoint(testing.allocator, &live16, 32, s);
    defer cp32.deinit(testing.allocator);
    try testing.expectEqual(@as(c_int, 32), cp32.layers[0].qsa_rows);

    const aux64_shape = [_]c_int{ 1, 64, hd };
    var aux64 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aux64);
    try mlx.check(mlx.mlx_ones(&aux64, &aux64_shape, 3, .float32, s));
    var pooled64 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled64);
    const pooled64_shape = [_]c_int{ 1, 16, hd };
    try mlx.check(mlx.mlx_ones(&pooled64, &pooled64_shape, 3, .float32, s));
    var live64 = [_]SSMCacheEntry{.{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
        .aux_state = aux64,
        .qsa_pooled = pooled64,
        .qsa_ratio = ratio,
    }};
    aux64 = mlx.mlx_array_new();
    pooled64 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&aux64, live64[0].aux_state));
    try mlx.check(mlx.mlx_array_set(&pooled64, live64[0].qsa_pooled));
    defer ssmFreeQsaState(&live64[0]);
    var cps = [_]SSMCheckpoint{
        try captureSsmCheckpoint(testing.allocator, &live64, 16, s),
        try captureSsmCheckpoint(testing.allocator, &live64, 64, s),
    };
    defer for (&cps) |*c| c.deinit(testing.allocator);
    try testing.expectEqual(@as(c_int, 16), cps[0].layers[0].qsa_rows);
    try testing.expectEqual(@as(c_int, 64), cps[1].layers[0].qsa_rows);

    const aux8_shape = [_]c_int{ 1, 8, hd };
    var aux8 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aux8);
    try mlx.check(mlx.mlx_ones(&aux8, &aux8_shape, 3, .float32, s));
    var live8 = [_]SSMCacheEntry{.{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
        .aux_state = aux8,
        .qsa_ratio = ratio,
    }};
    aux8 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&aux8, live8[0].aux_state));
    defer ssmFreeQsaState(&live8[0]);
    try attachQsaHistoryToLatest(&cps, &live8, s);
    try testing.expectEqual(@as(c_int, 16), cps[0].layers[0].qsa_rows);
    try testing.expectEqual(@as(c_int, 64), cps[1].layers[0].qsa_rows);
}

test "captureSsmCheckpoint does not copy QSA aux history" {
    // QSA indexer keys grow with seq. Copying them into every stride
    // snapshot is what OOM'd a 400k prefill. Capture must leave them on
    // the live entry; attachQsaHistoryToLatest puts one copy on the latest.
    const s = mlx.gpuStream();
    const aux_shape = [_]c_int{ 1, 2048, 128 };
    var aux = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aux);
    try mlx.check(mlx.mlx_ones(&aux, &aux_shape, 3, .bfloat16, s));
    var pooled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled);
    const pooled_shape = [_]c_int{ 1, 512, 128 };
    try mlx.check(mlx.mlx_ones(&pooled, &pooled_shape, 3, .bfloat16, s));

    const entry: SSMCacheEntry = .{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
        .aux_state = aux,
        .qsa_pooled = pooled,
        .qsa_ratio = 4,
    };
    aux = mlx.mlx_array_new();
    pooled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&aux, entry.aux_state));
    try mlx.check(mlx.mlx_array_set(&pooled, entry.qsa_pooled));

    var entries = [_]SSMCacheEntry{entry};
    defer {
        if (entries[0].aux_state.ctx != null) _ = mlx.mlx_array_free(entries[0].aux_state);
        if (entries[0].qsa_pooled.ctx != null) _ = mlx.mlx_array_free(entries[0].qsa_pooled);
    }
    var cp = try captureSsmCheckpoint(testing.allocator, &entries, 2048, s);
    defer cp.deinit(testing.allocator);
    try testing.expect(!checkpointHasQsaHistory(&cp));
    try testing.expect(cp.layers[0].aux_state.ctx == null);
    try testing.expect(cp.layers[0].qsa_pooled.ctx == null);
    try testing.expect(entries[0].aux_state.ctx != null);
    try testing.expectEqual(@as(c_int, 2048), mlx.getShape(entries[0].aux_state)[1]);
}

test "attachQsaHistoryToLatest is one copy; applyQsaHistoryAt slices to pos" {
    const s = mlx.gpuStream();
    const n: c_int = 32;
    const aux_shape = [_]c_int{ 1, n, 8 };
    var aux = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aux);
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, @floatFromInt(n * 8), 1.0, .float32, s));
        try mlx.check(mlx.mlx_reshape(&aux, flat, &aux_shape, 3, s));
        try mlx.check(mlx.mlx_array_eval(aux));
    }
    var pooled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_ones(&pooled, &[_]c_int{ 1, n / 4, 8 }, 3, .float32, s));
    const live: SSMCacheEntry = .{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
        .aux_state = aux,
        .qsa_pooled = pooled,
        .qsa_ratio = 4,
    };
    pooled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&pooled, live.qsa_pooled));
    aux = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&aux, live.aux_state));

    var live_arr = [_]SSMCacheEntry{live};
    defer {
        if (live_arr[0].aux_state.ctx != null) _ = mlx.mlx_array_free(live_arr[0].aux_state);
        if (live_arr[0].qsa_pooled.ctx != null) _ = mlx.mlx_array_free(live_arr[0].qsa_pooled);
    }
    const cp_a = try captureSsmCheckpoint(testing.allocator, &live_arr, 16, s);
    const cp_b = try captureSsmCheckpoint(testing.allocator, &live_arr, 32, s);
    var cps = [_]SSMCheckpoint{ cp_a, cp_b };
    defer for (&cps) |*c| c.deinit(testing.allocator);

    try testing.expect(!checkpointHasQsaHistory(&cps[0]));
    try testing.expect(!checkpointHasQsaHistory(&cps[1]));
    const bytes_before = ssmCheckpointBytes(&cps[0]) + ssmCheckpointBytes(&cps[1]);

    try attachQsaHistoryToLatest(&cps, &live_arr, s);
    try testing.expect(!checkpointHasQsaHistory(&cps[0]));
    try testing.expect(checkpointHasQsaHistory(&cps[1]));
    const pooled_bytes: u64 = @as(u64, @intCast(n / 4)) * 8 * 4;
    try testing.expectEqual(bytes_before + pooled_bytes, ssmCheckpointBytes(&cps[0]) + ssmCheckpointBytes(&cps[1]));

    var dest_arr = [_]SSMCacheEntry{.{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
    }};
    defer ssmFreeQsaState(&dest_arr[0]);
    try restoreSsmCheckpoint(&dest_arr, &cps[0]);
    try applyQsaHistoryAt(&dest_arr, &cps[1], 16, s, false);
    try testing.expect(dest_arr[0].aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, 4), dest_arr[0].qsa_pooled_blocks);

    try sliceQsaHistoryOntoCheckpoint(&cps[0], &cps[1], 16, s);
    try testing.expect(checkpointHasQsaHistory(&cps[0]));
    try testing.expect(cps[0].layers[0].aux_state.ctx == null);

    keepOnlyLatestQsaHistory(&cps);
    try testing.expect(!checkpointHasQsaHistory(&cps[0]));
    try testing.expect(checkpointHasQsaHistory(&cps[1]));
    try testing.expect(cps[1].layers[0].aux_state.ctx == null);

    var short = [_]SSMCheckpoint{try captureSsmCheckpoint(testing.allocator, &live_arr, 16, s)};
    defer short[0].deinit(testing.allocator);
    try attachQsaHistoryToLatest(&short, &live_arr, s);
    try testing.expect(short[0].layers[0].aux_state.ctx == null);
    try testing.expect(short[0].layers[0].qsa_pooled.ctx != null);
}

test "handoffQsaHistoryToLatest: the newest snap takes a VIEW of the live buffer — zero allocation, sliced to pos, bytes equal to the copy arm" {
    // The handoff hands the dying slot's buffer to the entry: same rows, same bytes, no allocation.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var t: Transformer = undefined;
    t.s = s;
    t.allocator = testing.allocator;
    const hd: c_int = 16;
    var prng = std.Random.DefaultPrng.init(0x5A5A_1D0F);
    const rnd = prng.random();

    var live = [_]SSMCacheEntry{.{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true }};
    defer ssmFreeQsaState(&live[0]);
    // A reserved append: the buffer carries slack past the published rows.
    const keys = try attn256RandBf16(rnd, &[_]c_int{ 1, QSA_RING_ROWS, hd }, s);
    defer _ = mlx.mlx_array_free(keys);
    try t.qsaAppendKeys(&live[0], keys, 0);
    const blocks = try attn256RandBf16(rnd, &[_]c_int{ 1, QSA_RING_ROWS / 4, hd }, s);
    defer _ = mlx.mlx_array_free(blocks);
    try t.qsaAppendPooled(&live[0], blocks, 0);
    try mlx.check(mlx.mlx_array_eval(live[0].aux_state));
    try mlx.check(mlx.mlx_array_eval(live[0].qsa_pooled));
    try testing.expectEqual(QSA_RING_ROWS, mlx.getShape(live[0].qsa_key_buf)[1]);

    var ref = [_]SSMCheckpoint{try captureSsmCheckpoint(testing.allocator, &live, @intCast(QSA_RING_ROWS), s)};
    defer ref[0].deinit(testing.allocator);
    try attachQsaHistoryToLatest(&ref, &live, s);

    var cps = [_]SSMCheckpoint{
        try captureSsmCheckpoint(testing.allocator, &live, 16, s),
        try captureSsmCheckpoint(testing.allocator, &live, @intCast(QSA_RING_ROWS), s),
    };
    defer for (&cps) |*c| c.deinit(testing.allocator);
    var before: usize = 0;
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_get_active_memory(&before);
    try handoffQsaHistoryToLatest(&cps, &live, s);
    var after: usize = 0;
    _ = mlx.mlx_get_active_memory(&after);
    try testing.expect(after <= before);
    try testing.expect(!checkpointHasQsaHistory(&cps[0]));
    try testing.expect(checkpointHasQsaHistory(&cps[1]));
    const got = &cps[1].layers[0];
    try testing.expect(got.aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, QSA_RING_ROWS / 4), mlx.getShape(got.qsa_pooled)[1]);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got.qsa_pooled, ref[0].layers[0].qsa_pooled, s));
    try testing.expectEqual(ssmCheckpointBytes(&ref[0]), ssmCheckpointBytes(&cps[1]));
    try testing.expectEqual(@as(u64, 8 * @as(u64, @intCast(hd)) * 2), ssmCheckpointBytes(&cps[1]));
    try testing.expectEqual(QSA_RING_ROWS, mlx.getShape(live[0].aux_state)[1]);
    ssmFreeQsaState(&live[0]);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got.qsa_pooled, ref[0].layers[0].qsa_pooled, s));
}

test "handoffQsaHistoryToLatest materializes when the buffer's slack past the snap exceeds QSA_HANDOFF_MAX_SLACK_ROWS" {
    // Past the slack bar the handoff pays the copy.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var t: Transformer = undefined;
    t.s = s;
    t.allocator = testing.allocator;
    const hd: c_int = 16;
    var prng = std.Random.DefaultPrng.init(0x0DD5_1ACC);
    const rnd = prng.random();
    try testing.expect(!qsaHandoffExceedsSlack(512 + QSA_HANDOFF_MAX_SLACK_ROWS, 512));
    try testing.expect(qsaHandoffExceedsSlack(512 + QSA_HANDOFF_MAX_SLACK_ROWS + 1, 512));

    var live = [_]SSMCacheEntry{.{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true }};
    defer ssmFreeQsaState(&live[0]);
    const keys = try attn256RandBf16(rnd, &[_]c_int{ 1, QSA_RING_ROWS, hd }, s);
    defer _ = mlx.mlx_array_free(keys);
    try t.qsaAppendKeys(&live[0], keys, 0);
    try mlx.check(mlx.mlx_array_eval(live[0].aux_state));
    var ref = [_]SSMCheckpoint{try captureSsmCheckpoint(testing.allocator, &live, @intCast(QSA_RING_ROWS), s)};
    defer ref[0].deinit(testing.allocator);
    try attachQsaHistoryToLatest(&ref, &live, s);

    var cps = [_]SSMCheckpoint{try captureSsmCheckpoint(testing.allocator, &live, @intCast(QSA_RING_ROWS), s)};
    defer cps[0].deinit(testing.allocator);
    try handoffQsaHistoryToLatest(&cps, &live, s);
    try testing.expect(cps[0].layers[0].aux_state.ctx == null);
    ssmFreeQsaState(&live[0]);
}

test "applyQsaHistoryAt: a sliced restore is a VIEW of the entry's history; the trim's slice is a real copy" {
    // Restore: the slot appends into qsa_pooled_buf via seedCapBuf, so a view of the entry is not written through.
    // Trim: the inherit slice must own its rows; a view would pin the donor's full bank.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const n: c_int = 32;
    const aux_shape = [_]c_int{ 1, n, 8 };
    var aux = mlx.mlx_array_new();
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, @floatFromInt(n * 8), 1.0, .float32, s));
        try mlx.check(mlx.mlx_reshape(&aux, flat, &aux_shape, 3, s));
        try mlx.check(mlx.mlx_array_eval(aux));
    }
    var pooled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_ones(&pooled, &[_]c_int{ 1, n / 4, 8 }, 3, .float32, s));
    var live = [_]SSMCacheEntry{.{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .aux_state = aux, .qsa_pooled = pooled, .qsa_ratio = 4 }};
    defer ssmFreeQsaState(&live[0]);
    var cps = [_]SSMCheckpoint{try captureSsmCheckpoint(testing.allocator, &live, 32, s)};
    defer cps[0].deinit(testing.allocator);
    try attachQsaHistoryToLatest(&cps, &live, s);

    var dest = [_]SSMCacheEntry{.{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true }};
    defer ssmFreeQsaState(&dest[0]);
    try restoreSsmCheckpoint(&dest, &cps[0]);
    try applyQsaHistoryAt(&dest, &cps[0], 16, s, false);
    try testing.expectEqual(@as(c_int, 4), dest[0].qsa_pooled_blocks);
    try mlx.check(mlx.mlx_array_eval(dest[0].qsa_pooled));
    try mlx.check(mlx.mlx_array_eval(cps[0].layers[0].qsa_pooled));
    const dest_ptr = mlx.mlx_array_data_float32(dest[0].qsa_pooled) orelse return error.TestUnexpectedResult;
    const src_ptr = mlx.mlx_array_data_float32(cps[0].layers[0].qsa_pooled) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(src_ptr, dest_ptr);

    var t: Transformer = undefined;
    t.s = s;
    t.allocator = testing.allocator;
    var extra = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(extra);
    try mlx.check(mlx.mlx_ones(&extra, &[_]c_int{ 1, 1, 8 }, 3, .float32, s));
    try t.qsaAppendPooled(&dest[0], extra, dest[0].qsa_pooled_blocks);
    try mlx.check(mlx.mlx_array_eval(dest[0].qsa_pooled));
    try mlx.check(mlx.mlx_array_eval(cps[0].layers[0].qsa_pooled));
    const after_ptr = mlx.mlx_array_data_float32(dest[0].qsa_pooled) orelse return error.TestUnexpectedResult;
    const src_after = mlx.mlx_array_data_float32(cps[0].layers[0].qsa_pooled) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(src_ptr, src_after);
    try testing.expect(after_ptr != src_ptr);

    var kept = [_]SSMCheckpoint{try captureSsmCheckpoint(testing.allocator, &live, 16, s)};
    defer kept[0].deinit(testing.allocator);
    try sliceQsaHistoryOntoCheckpoint(&kept[0], &cps[0], 16, s);
    try testing.expect(kept[0].layers[0].aux_state.ctx == null);
    try testing.expect(kept[0].layers[0].qsa_pooled.ctx != null);
    try mlx.check(mlx.mlx_array_eval(kept[0].layers[0].qsa_pooled));
    const trim_ptr = mlx.mlx_array_data_float32(kept[0].layers[0].qsa_pooled) orelse return error.TestUnexpectedResult;
    try testing.expect(trim_ptr != src_ptr);
}

test "ssm checkpoint carries the qwen4_exp aux state (key history, pooled keys, PLE tokens) through capture and restore" {
    const s = mlx.gpuStream();
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA0C5);
    const rnd = prng.random();
    var src_e = [_]SSMCacheEntry{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = try attn256RandBf16(rnd, &[_]c_int{ 1, 3, 64 }, s), .ssm_state = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 8, 8 }, s), .initialized = true },
    };
    // Layer 0 = a full-attention layer holding the QSA key history + pooled keys
    // (no conv/ssm state); layer 1 = a GDN layer that also carries the PLE
    // window in aux_state and the token history.
    src_e[0].aux_state = try attn256RandBf16(rnd, &[_]c_int{ 1, 37, 16 }, s);
    src_e[0].qsa_pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, 9, 16 }, s);
    src_e[0].qsa_ratio = 4;
    src_e[1].aux_state = try attn256RandBf16(rnd, &[_]c_int{ 1, 6, 32 }, s);
    src_e[1].ple_prev = .{ 11, 22, 0, 0, 0, 0, 0, 0 };
    src_e[1].ple_prev_valid = true;
    defer for (&src_e) |*e| {
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
        ssmFreeQsaState(e);
    };

    var cps = [_]SSMCheckpoint{try captureSsmCheckpoint(a, &src_e, 37, s)};
    defer cps[0].deinit(a);
    try std.testing.expectEqual(@as(c_int, 1), mlx.getShape(cps[0].layers[0].aux_state)[1]);
    try std.testing.expect(cps[0].layers[0].qsa_pooled.ctx == null);
    try std.testing.expect(cps[0].layers[1].ple_prev_valid and cps[0].layers[1].ple_prev[1] == 22);
    try std.testing.expect(cps[0].layers[1].aux_state.ctx != null);
    try attachQsaHistoryToLatest(&cps, &src_e, s);
    const cp = &cps[0];
    try std.testing.expectEqual(@as(c_int, 1), mlx.getShape(cp.layers[0].aux_state)[1]);
    try std.testing.expect(cp.layers[0].qsa_pooled.ctx != null);
    const expect_bytes: u64 = (1 * 16 + 9 * 16 + 6 * 32 + 3 * 64 + 2 * 8 * 8) * 2;
    try std.testing.expectEqual(expect_bytes, ssmCheckpointBytes(cp));

    // A restore target that holds a LATER request's state gets it replaced.
    var dst = [_]SSMCacheEntry{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    dst[0].aux_state = try attn256RandBf16(rnd, &[_]c_int{ 1, 90, 16 }, s);
    dst[1].ple_prev = .{ 7, 7, 0, 0, 0, 0, 0, 0 };
    dst[1].ple_prev_valid = true;
    defer for (&dst) |*e| {
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
        ssmFreeQsaState(e);
    };
    try restoreSsmCheckpoint(&dst, cp);
    try applyQsaHistoryAt(&dst, cp, 37, s, false);
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dst[0].aux_state, cp.layers[0].aux_state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dst[0].qsa_pooled, src_e[0].qsa_pooled, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dst[1].aux_state, src_e[1].aux_state, s));
    try std.testing.expect(dst[1].qsa_pooled.ctx == null);
    try std.testing.expectEqual(@as(u32, 11), dst[1].ple_prev[0]);
    try std.testing.expect(dst[1].ple_prev_valid and !dst[0].ple_prev_valid);
    // The checkpoint owns its buffers: freeing the source leaves it intact.
    _ = mlx.mlx_array_free(src_e[0].aux_state);
    src_e[0].aux_state = .{ .ctx = null };
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(dst[0].aux_state, cp.layers[0].aux_state, s));
}

test "qsa key append plan: rows advance by S, capacity grows only when it must" {
    const p0 = qsaKeyAppendPlan(0, 0, 4, 0);
    try testing.expectEqual(@as(usize, 0), p0.write_at);
    try testing.expectEqual(@as(usize, 4), p0.new_rows);
    try testing.expect(p0.new_cap >= 4);

    const p1 = qsaKeyAppendPlan(100, 512, 1, 0);
    try testing.expectEqual(@as(usize, 512), p1.new_cap);
    try testing.expectEqual(@as(usize, 100), p1.write_at);
    try testing.expectEqual(@as(usize, 101), p1.new_rows);

    const p2 = qsaKeyAppendPlan(511, 512, 1, 0);
    try testing.expectEqual(@as(usize, 512), p2.new_cap);
    try testing.expectEqual(@as(usize, 512), p2.new_rows);

    const p3 = qsaKeyAppendPlan(512, 512, 1, 0);
    try testing.expect(p3.new_cap >= 513);
    try testing.expectEqual(@as(usize, 512), p3.write_at);
    try testing.expectEqual(@as(usize, 513), p3.new_rows);

    const p4 = qsaKeyAppendPlan(500, 512, 4096, 0);
    try testing.expect(p4.new_cap >= 4596);
    try testing.expectEqual(@as(usize, 4596), p4.new_rows);
}

test "reserveQsaHistory: the MTP head entry is reserved at the same length" {
    var slot = [_]SSMCacheEntry{.{ .ssm_state = .{ .ctx = null }, .conv_state = .{ .ctx = null }, .initialized = false }};
    var head: SSMCacheEntry = .{ .ssm_state = .{ .ctx = null }, .conv_state = .{ .ctx = null }, .initialized = false };
    const tokens: usize = 8192;
    reserveQsaHistoryWithHead(slot[0..], &head, tokens);
    try testing.expectEqual(tokens, slot[0].qsa_reserve_rows);
    try testing.expectEqual(tokens, head.qsa_reserve_rows);
}

test "qsa history reservation: a reserved prefill allocates its buffers ONCE (#353 follow-up)" {
    // These buffers walked the +25% ladder beside a KV cache that had been sized once; the loop
    // below allocated ~30 times before `qsa_reserve_rows`.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5E5E_4E5E);
    const rnd = prng.random();
    const hd: c_int = 128;
    const chunk_rows: c_int = 4096;
    const total: usize = 409_600;

    var t: Transformer = undefined;
    t.s = s;
    t.allocator = ta;
    var entries = [_]SSMCacheEntry{.{ .ssm_state = mlx.mlx_array_new(), .conv_state = mlx.mlx_array_new(), .initialized = false }};
    const entry = &entries[0];
    defer {
        ssmFreeQsaState(entry);
        _ = mlx.mlx_array_free(entry.ssm_state);
        _ = mlx.mlx_array_free(entry.conv_state);
    }

    reserveQsaHistory(entries[0..], total);
    try testing.expectEqual(total, entry.qsa_reserve_rows);

    var rows: c_int = 0;
    while (@as(usize, @intCast(rows)) < total) {
        const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, chunk_rows, hd }, s);
        defer _ = mlx.mlx_array_free(chunk);
        try t.qsaAppendKeys(entry, chunk, rows);
        rows += chunk_rows;
    }
    try testing.expectEqual(QSA_RING_ROWS, mlx.getShape(entry.aux_state)[1]);
    try testing.expect(mlx.getShape(entry.qsa_key_buf)[1] <= QSA_RING_ROWS);
    try testing.expectEqual(@as(c_int, @intCast(total)), entry.qsa_hist_rows);
}

test "qsa key append plan: a reservation raises the FIRST grow and nothing else" {
    try testing.expectEqual(@as(usize, 512), qsaKeyAppendPlan(100, 512, 1, 400_000).new_cap);
    try testing.expectEqual(@as(usize, 400_000), qsaKeyAppendPlan(0, 0, 4096, 400_000).new_cap);
    try testing.expect(qsaKeyAppendPlan(0, 0, 500_000, 400_000).new_cap >= 500_000);
    try testing.expect(qsaKeyAppendPlan(400_000, 400_000, 1, 400_000).new_cap > 400_000);
}

test "qsa key history: capacity-buffer append equals the concatenated reference, across a growth boundary" {
    // The published `aux_state` must stay byte-for-byte the old `concatenate`, including across
    // reallocations and after a restore re-seeds the accelerator.
    try mlx.requireGpu();
    const s = mlx.gpuStream();

    var t: Transformer = undefined;
    t.s = s;
    t.allocator = testing.allocator;

    const hd: c_int = 16;
    var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&entry);

    var prng = std.Random.DefaultPrng.init(0xB0FF_2C11);
    const rnd = prng.random();
    var reference = mlx.mlx_array{ .ctx = null };
    defer if (reference.ctx != null) {
        _ = mlx.mlx_array_free(reference);
    };
    var rows: c_int = 0;
    var step_i: usize = 0;
    while (step_i < 601) : (step_i += 1) {
        const add: c_int = if (step_i == 0) 7 else 1;
        const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, add, hd }, s);
        defer _ = mlx.mlx_array_free(chunk);
        var next_ref = mlx.mlx_array_new();
        if (reference.ctx != null) {
            const parts = [_]mlx.mlx_array{ reference, chunk };
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&next_ref, vec, 1, s));
            _ = mlx.mlx_array_free(reference);
        } else {
            _ = mlx.mlx_array_free(next_ref);
            next_ref = try materializedOwnedCopy(s, chunk);
        }
        reference = next_ref;

        try t.qsaAppendKeys(&entry, chunk, rows);
        rows += add;
        const tail = try qsaRingTail(reference, s);
        defer _ = mlx.mlx_array_free(tail);
        try testing.expectEqual(mlx.getShape(tail)[1], mlx.getShape(entry.aux_state)[1]);
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(entry.aux_state, tail, s));
        try testing.expect(mlx.getShape(entry.qsa_key_buf)[1] >= mlx.getShape(entry.aux_state)[1]);
    }
    try testing.expectEqual(QSA_RING_ROWS, mlx.getShape(entry.aux_state)[1]);

    // A restore replaces the history behind the accelerator's back; the append re-seeds from `aux_state`.
    const restored = try attn256RandBf16(rnd, &[_]c_int{ 1, rows, hd }, s);
    ssmFreeQsaState(&entry);
    entry.aux_state = restored;
    _ = mlx.mlx_array_free(reference);
    reference = try materializedOwnedCopy(s, restored);

    const tail = try attn256RandBf16(rnd, &[_]c_int{ 1, 3, hd }, s);
    defer _ = mlx.mlx_array_free(tail);
    {
        const parts = [_]mlx.mlx_array{ reference, tail };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var cat = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 1, s));
        _ = mlx.mlx_array_free(reference);
        reference = cat;
    }
    try t.qsaAppendKeys(&entry, tail, rows);
    {
        const ring = try qsaRingTail(reference, s);
        defer _ = mlx.mlx_array_free(ring);
        try testing.expectEqual(mlx.getShape(ring)[1], mlx.getShape(entry.aux_state)[1]);
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(entry.aux_state, ring, s));
    }
    rows += 3;

    // A site that replaced `aux_state` without dropping the buffer still cannot poison the history.
    const shorter = try attn256RandBf16(rnd, &[_]c_int{ 1, 12, hd }, s);
    const stale_view = entry.aux_state;
    entry.aux_state = shorter; // qsa_key_buf deliberately left in place
    _ = mlx.mlx_array_free(stale_view);
    _ = mlx.mlx_array_free(reference);
    reference = try materializedOwnedCopy(s, shorter);
    const tail2 = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, hd }, s);
    defer _ = mlx.mlx_array_free(tail2);
    {
        const parts = [_]mlx.mlx_array{ reference, tail2 };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var cat = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 1, s));
        _ = mlx.mlx_array_free(reference);
        reference = cat;
    }
    try t.qsaAppendKeys(&entry, tail2, 12);
    try testing.expectEqual(@as(c_int, 14), mlx.getShape(entry.aux_state)[1]);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(entry.aux_state, reference, s));
}

test "qsa pooled bank: capacity append equals the concat, and the f32 score operand is rebuilt only when a block completes" {
    // The bf16 bank must stay byte-for-byte the old `concatenate`; 12 layers of one forward reuse one f32 build.
    try mlx.requireGpu();
    const s = mlx.gpuStream();

    var t: Transformer = undefined;
    t.s = s;
    t.allocator = testing.allocator;
    t.qsa_score_bank_builds = 0;

    const hd: c_int = 16;
    var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&entry);

    var prng = std.Random.DefaultPrng.init(0x9A17_3D02);
    const rnd = prng.random();
    var reference = mlx.mlx_array{ .ctx = null };
    defer if (reference.ctx != null) {
        _ = mlx.mlx_array_free(reference);
    };
    var nb: c_int = 0;
    var step_i: usize = 0;
    while (step_i < 400) : (step_i += 1) {
        const add: c_int = if (step_i == 0) 5 else 1;
        const new3 = try attn256RandBf16(rnd, &[_]c_int{ 1, add, hd }, s);
        defer _ = mlx.mlx_array_free(new3);
        var next_ref = mlx.mlx_array_new();
        if (reference.ctx != null) {
            const parts = [_]mlx.mlx_array{ reference, new3 };
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&next_ref, vec, 1, s));
            _ = mlx.mlx_array_free(reference);
        } else {
            _ = mlx.mlx_array_free(next_ref);
            next_ref = try materializedOwnedCopy(s, new3);
        }
        reference = next_ref;

        try t.qsaAppendPooled(&entry, new3, nb);
        nb += add;
        try testing.expectEqual(nb, mlx.getShape(entry.qsa_pooled)[1]);
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(entry.qsa_pooled, reference, s));
    }
    try testing.expect(mlx.getShape(entry.qsa_pooled_buf)[1] > nb);

    const before = t.qsa_score_bank_builds;
    const bank = try t.qsaScoreBank(&entry, 1, nb, hd);
    try testing.expectEqual(before + 1, t.qsa_score_bank_builds);
    for (0..11) |_| {
        const again = try t.qsaScoreBank(&entry, 1, nb, hd);
        try testing.expect(again.ctx == bank.ctx);
    }
    try testing.expectEqual(before + 1, t.qsa_score_bank_builds);
    try testing.expectEqual(@as(c_int, hd), mlx.getShape(bank)[2]);
    try testing.expectEqual(nb, mlx.getShape(bank)[3]);

    {
        var k_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_rope);
        try mlx.check(mlx.mlx_reshape(&k_rope, entry.qsa_pooled, &[_]c_int{ 1, 1, nb, hd }, 4, s));
        var kt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kt);
        try mlx.check(mlx.mlx_transpose_axes(&kt, k_rope, &[_]c_int{ 0, 1, 3, 2 }, 4, s));
        var k32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k32);
        try mlx.check(mlx.mlx_astype(&k32, kt, .float32, s));
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(bank, k32, s));
    }

    const one = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, hd }, s);
    defer _ = mlx.mlx_array_free(one);
    try t.qsaAppendPooled(&entry, one, nb);
    _ = try t.qsaScoreBank(&entry, 1, nb + 1, hd);
    try testing.expectEqual(before + 2, t.qsa_score_bank_builds);
    try testing.expectEqual(nb + 1, mlx.getShape(entry.qsa_score_bank)[3]);
}

test "qsa score bank: incremental column append matches a from-scratch rebuild" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();

    var t: Transformer = undefined;
    t.s = s;
    t.allocator = testing.allocator;
    t.qsa_score_bank_builds = 0;

    const hd: c_int = 16;
    const batch: c_int = 1;
    var prng = std.Random.DefaultPrng.init(0x5C0E_BA11);
    const rnd = prng.random();

    var scratch: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&scratch);
    var incr: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&incr);

    const before = qwen4_qsa.qsa_score_incremental_appends;
    var nb: c_int = 0;
    var step_i: usize = 0;
    while (step_i < 24) : (step_i += 1) {
        const add: c_int = if (step_i == 0) 5 else 1;
        const new3 = try attn256RandBf16(rnd, &[_]c_int{ batch, add, hd }, s);
        defer _ = mlx.mlx_array_free(new3);
        try t.qsaAppendPooled(&scratch, new3, nb);
        try t.qsaAppendPooled(&incr, new3, nb);
        nb += add;
        _ = try t.qsaScoreBank(&incr, batch, nb, hd);
    }
    try testing.expectEqual(before + 23, qwen4_qsa.qsa_score_incremental_appends);
    try testing.expect(mlx.getShape(incr.qsa_score_buf)[3] > nb);
    const bank_inc = try t.qsaScoreBank(&incr, batch, nb, hd);
    const bank_scratch = try t.qsaScoreBank(&scratch, batch, nb, hd);
    try testing.expectEqual(@as(c_int, hd), mlx.getShape(bank_inc)[2]);
    try testing.expectEqual(nb, mlx.getShape(bank_inc)[3]);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(bank_inc, bank_scratch, s));
    {
        var k_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_rope);
        try mlx.check(mlx.mlx_reshape(&k_rope, scratch.qsa_pooled, &[_]c_int{ batch, 1, nb, hd }, 4, s));
        var kt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kt);
        try mlx.check(mlx.mlx_transpose_axes(&kt, k_rope, &[_]c_int{ 0, 1, 3, 2 }, 4, s));
        var k32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k32);
        try mlx.check(mlx.mlx_astype(&k32, kt, .float32, s));
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(bank_inc, k32, s));
    }
    try testing.expect(qwen4_qsa.qsa_score_incremental_appends > before);
}

test "qsa history: a restore-shaped re-seed lands AT the reservation, so the append never grows" {
    // A restore republishes `aux_state` but leaves the append accelerator empty, and the
    // re-seed used to produce a tight buffer that grew one line later (~81 ms at 384k).
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EED_5EED);
    const rnd = prng.random();
    const hd: c_int = 128;
    const held: c_int = 4096;
    const reserve: usize = 8192;

    var t: Transformer = undefined;
    t.s = s;
    t.allocator = ta;
    var entries = [_]SSMCacheEntry{.{ .ssm_state = mlx.mlx_array_new(), .conv_state = mlx.mlx_array_new(), .initialized = false }};
    const entry = &entries[0];
    defer {
        ssmFreeQsaState(entry);
        _ = mlx.mlx_array_free(entry.ssm_state);
        _ = mlx.mlx_array_free(entry.conv_state);
    }
    reserveQsaHistory(entries[0..], reserve);

    {
        const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, held, hd }, s);
        defer _ = mlx.mlx_array_free(chunk);
        try t.qsaAppendKeys(entry, chunk, 0);
    }
    // A restore: `aux_state` survives, the accelerator does not.
    var authority = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&authority, entry.aux_state));
    _ = mlx.mlx_array_free(entry.qsa_key_buf);
    entry.qsa_key_buf = .{ .ctx = null };
    entry.qsa_key_rows = 0;
    _ = mlx.mlx_array_free(entry.aux_state);
    entry.aux_state = authority;

    {
        const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, 30, hd }, s);
        defer _ = mlx.mlx_array_free(chunk);
        try t.qsaAppendKeys(entry, chunk, held);
    }
    try testing.expectEqual(QSA_RING_ROWS, mlx.getShape(entry.aux_state)[1]);
    try testing.expectEqual(@as(c_int, held + 30), entry.qsa_hist_rows);

    entry.qsa_reserve_rows = 0;
    _ = mlx.mlx_array_free(entry.qsa_key_buf);
    entry.qsa_key_buf = .{ .ctx = null };
    entry.qsa_key_rows = 0;
    {
        const chunk = try attn256RandBf16(rnd, &[_]c_int{ 1, 30, hd }, s);
        defer _ = mlx.mlx_array_free(chunk);
        try t.qsaAppendKeys(entry, chunk, held + 30);
    }
    try testing.expectEqual(QSA_RING_ROWS, mlx.getShape(entry.aux_state)[1]);
    try testing.expectEqual(@as(c_int, held + 60), entry.qsa_hist_rows);
}

test "rollback fixture preserves the separate final state at full acceptance" {
    const allocator = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    for ([_]u32{ 0, 1, 2 }) |accepted| {
        const slot = try Qwen4TestSlot.init(allocator, 1);
        defer slot.deinit(allocator);
        const entry = &slot.entries[0];
        _ = mlx.mlx_array_free(entry.ssm_state);
        entry.ssm_state = mlx.mlx_array_new_data(&[_]f32{33}, &.{ 1, 1, 1, 1 }, 4, .float32);
        entry.spec_state_seq = mlx.mlx_array_new_data(&[_]f32{ 11, 22, -999 }, &.{ 3, 1, 1, 1, 1 }, 5, .float32);
        try rollbackRowToAccepted(slot, accepted, 3, 0, 0, s);
        try mlx.check(mlx.mlx_array_eval(entry.ssm_state));
        const values = mlx.mlx_array_data_float32(entry.ssm_state) orelse return error.Unreadable;
        try testing.expectEqual(@as(f32, @floatFromInt((accepted + 1) * 11)), values[0]);
    }
}

test "QSA checkpoint copy failure preserves the destination pooled bank" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const data = [_]f32{ 1, 2, 3, 4 };
    const shape = [_]c_int{ 1, 4, 1 };
    const layers = try testing.allocator.alloc(SSMCacheEntrySnapshot, 1);
    layers[0] = .{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = true,
        .qsa_pooled = mlx.mlx_array_new_data(&data, &shape, 3, .float32),
        .qsa_ratio = 4,
        .qsa_rows = 16,
    };
    var dst: SSMCheckpoint = .{ .pos = 8, .layers = layers };
    defer dst.deinit(testing.allocator);
    var src_layers = [_]SSMCacheEntrySnapshot{layers[0]};
    src_layers[0].qsa_pooled = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(src_layers[0].qsa_pooled);
    const src: SSMCheckpoint = .{ .pos = 16, .layers = &src_layers };
    const original = dst.layers[0].qsa_pooled.ctx;
    mlx.fault.arm(1);
    defer mlx.fault.disarm();
    try testing.expectError(error.MlxError, sliceQsaHistoryOntoCheckpoint(&dst, &src, 8, s));
    try testing.expect(mlx.fault.didFire());
    try testing.expectEqual(original, dst.layers[0].qsa_pooled.ctx);
    try testing.expectEqualSlices(c_int, &shape, mlx.getShape(dst.layers[0].qsa_pooled));
}
