//! MiMo's argmax-only readout: the coarse head's top-32 per row, re-scored through the full head (a 32-row slice of the
//! same qmv, so each value is the full readout's), every other id -inf. Serial ticks and verify rows take it alike; the
//! trunk asks it through `arch_state.headShortlist`.
const std = @import("std");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const mtp_mod = @import("../mtp.zig");
const io_util = @import("../io_util.zig");
const mlx_util = @import("../mlx_util.zig");
const trunk = @import("../trunk.zig");
const Trunk = trunk.Trunk;

/// The greedy shortlist's one-time log and the coarse head it reads: the head's one owner, so a failed shortlist retires
/// it (`mtp_mod.retireCoarse`) for every reader. The MTP heads draft on the same copy, so a load with MTP and one
/// without read the same greedy argmax.
pub const Cell = struct { logged: bool = false, coarse: ?mtp_mod.RerankCoarse = null };

/// Rows an argmax-only shortlist readout serves: a decode tick and every verify width.
const MAX_ROWS: usize = 16;
var lmhead_full_env: ?bool = null;
var lmhead_shortlist_audit_env: ?bool = null;

fn shortlistOn() bool {
    return !io_util.diagEnvOnCached(&lmhead_full_env, "SUSHI_LMHEAD_FULL");
}

/// `x` through `t`'s head as the shortlist reads it; null keeps the full head.
pub fn readout(t: *const Trunk, cell: *Cell, x: mlx.mlx_array) !?mlx.mlx_array {
    const head = cell.coarse orelse return null;
    const bits = head.bits;
    if (!t.config.isMimo() or !shortlistOn()) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len != 3 or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const n: usize = @intCast(xsh[0] * xsh[1]);
    if (n < 1 or n > MAX_ROWS) return null;
    const vocab = head.rows;
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    try mlx.check(mlx.mlx_reshape(&x3, x, &[_]c_int{ @intCast(n), 1, xsh[2] }, 3, t.s));
    var logged = true;
    var lists: [MAX_ROWS]mtp_mod.Shortlist = undefined;
    if (!try mtp_mod.rerankShortlistsBatched(t.s, t, &cell.coarse, &logged, x3, t.suppress_mask, lists[0..n])) return null;
    defer for (lists[0..n]) |*sl| sl.deinit();
    const sparse = try sparseLogits(t.s, lists[0..n], vocab);
    defer _ = mlx.mlx_array_free(sparse.idx);
    defer _ = mlx.mlx_array_free(sparse.logits);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, sparse.logits, &[_]c_int{ xsh[0], xsh[1], vocab }, 3, t.s));
    if (io_util.diagEnvOnCached(&lmhead_shortlist_audit_env, "SUSHI_LMHEAD_SHORTLIST_AUDIT")) {
        const full = try t.qmatmul(x3, &t.lm_head);
        defer _ = mlx.mlx_array_free(full);
        try audit(t.s, full, sparse.logits, sparse.idx, n, vocab);
    }
    if (!cell.logged) {
        cell.logged = true;
        log.info("[lm_head] greedy shortlist engaged: {d}-bit coarse top-32 + full-head re-score (SUSHI_LMHEAD_FULL=1 reads the full head)\n", .{bits});
    }
    return out;
}

/// `[n, 1, vocab]` bf16 rows holding each shortlist's exact logits at its ids and -inf elsewhere, so
/// an argmax over a row picks the lowest id among the shortlist's maxima. That is the full row's
/// argmax whenever the full row's lowest-id maximum is in the shortlist; one outside is the miss the
/// audit counts. Also returns the `[n, 1, 32]` int32 ids.
fn sparseLogits(s: mlx.mlx_stream, lists: []const mtp_mod.Shortlist, vocab: c_int) !struct { logits: mlx.mlx_array, idx: mlx.mlx_array } {
    const n = lists.len;
    if (n == 0 or n > MAX_ROWS) return error.ShortlistRowsOutOfRange;
    var idx_parts: [MAX_ROWS]mlx.mlx_array = @splat(.{ .ctx = null });
    var val_parts: [MAX_ROWS]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (idx_parts[0..n], val_parts[0..n]) |a, b| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a);
        if (b.ctx != null) _ = mlx.mlx_array_free(b);
    };
    for (lists, 0..) |sl, i| {
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        try mlx.check(mlx.mlx_astype(&ids, sl.cands, .int32, s));
        idx_parts[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&idx_parts[i], ids, &[_]c_int{ 1, 1, 32 }, 3, s));
        var v = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v);
        try mlx.check(mlx.mlx_astype(&v, sl.exact, .bfloat16, s));
        val_parts[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&val_parts[i], v, &[_]c_int{ 1, 1, 32 }, 3, s));
    }
    const iv = mlx.mlx_vector_array_new_data(&idx_parts, n);
    defer _ = mlx.mlx_vector_array_free(iv);
    const vv = mlx.mlx_vector_array_new_data(&val_parts, n);
    defer _ = mlx.mlx_vector_array_free(vv);
    var idx = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_concatenate_axis(&idx, iv, 0, s));
    var vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals);
    try mlx.check(mlx.mlx_concatenate_axis(&vals, vv, 0, s));
    const neg_inf = try mlx_util.scalarOf(-std.math.inf(f32), .bfloat16, s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var base = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(base);
    try mlx.check(mlx.mlx_full(&base, &[_]c_int{ @intCast(n), 1, vocab }, 3, neg_inf, .bfloat16, s));
    var filled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(filled);
    try mlx.check(mlx.mlx_put_along_axis(&filled, base, idx, vals, -1, s));
    return .{ .logits = filled, .idx = idx };
}

/// DIAGNOSTIC (SUSHI_LMHEAD_SHORTLIST_AUDIT=1): every shortlist row also reads the full head and
/// counts rows whose full argmax is outside the shortlist or differs from the served one.
const Audit = struct {
    rows: u64 = 0,
    outside: u64 = 0,
    differ: u64 = 0,
    /// Rows where a shortlist logit is not bit-equal to the full head's at that id.
    inexact: u64 = 0,
};
var audit_counts: Audit = .{};

fn audit(s: mlx.mlx_stream, full: mlx.mlx_array, sparse: mlx.mlx_array, idx: mlx.mlx_array, n: usize, vocab: c_int) !void {
    var fa = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fa);
    try mlx.check(mlx.mlx_argmax_axis(&fa, full, -1, false, s));
    var sa = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sa);
    try mlx.check(mlx.mlx_argmax_axis(&sa, sparse, -1, false, s));
    var fa32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fa32);
    try mlx.check(mlx.mlx_astype(&fa32, fa, .int32, s));
    var sa32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sa32);
    try mlx.check(mlx.mlx_astype(&sa32, sa, .int32, s));
    var f32full = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32full);
    try mlx.check(mlx.mlx_astype(&f32full, full, .float32, s));
    var f32sparse = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32sparse);
    try mlx.check(mlx.mlx_astype(&f32sparse, sparse, .float32, s));
    try mlx.check(mlx.mlx_array_eval(fa32));
    try mlx.check(mlx.mlx_array_eval(sa32));
    try mlx.check(mlx.mlx_array_eval(idx));
    try mlx.check(mlx.mlx_array_eval(f32full));
    try mlx.check(mlx.mlx_array_eval(f32sparse));
    const fp = mlx.mlx_array_data_int32(fa32) orelse return;
    const sp = mlx.mlx_array_data_int32(sa32) orelse return;
    const ip = mlx.mlx_array_data_int32(idx) orelse return;
    const lp = mlx.mlx_array_data_float32(f32full) orelse return;
    const vp = mlx.mlx_array_data_float32(f32sparse) orelse return;
    const v: usize = @intCast(vocab);
    for (0..n) |r| {
        audit_counts.rows += 1;
        var inside = false;
        var exact = true;
        for (ip[r * 32 ..][0..32]) |c| {
            inside = inside or c == fp[r];
            const at = r * v + @as(usize, @intCast(c));
            exact = exact and vp[at] == lp[at];
        }
        if (!inside) audit_counts.outside += 1;
        if (!exact) audit_counts.inexact += 1;
        if (fp[r] != sp[r]) {
            audit_counts.differ += 1;
            const row = lp[r * v ..][0..v];
            var second: f32 = -std.math.inf(f32);
            for (row, 0..) |val, j| if (j != @as(usize, @intCast(fp[r])) and val > second) {
                second = val;
            };
            log.info("[lm_head-audit] argmax differs: full {d} ({d:.4}) served {d}, full top-2 gap {d:.5}, in shortlist {}\n", .{ fp[r], row[@intCast(fp[r])], sp[r], row[@intCast(fp[r])] - second, inside });
        }
    }
    if (audit_counts.rows % 256 < n) log.info("[lm_head-audit] rows {d}, full argmax outside the shortlist {d}, served argmax differs {d}, shortlist logits not bit-equal {d}\n", .{ audit_counts.rows, audit_counts.outside, audit_counts.differ, audit_counts.inexact });
}

test "argmax-only shortlist rows hold the exact logits at their ids and argmax to the lowest tied id" {
    const s = mlx.gpuStream();
    const vocab: c_int = 256;
    var cand_ids: [2][32]u32 = undefined;
    var vals: [2][32]u16 = undefined;
    for (0..32) |j| {
        cand_ids[0][j] = @intCast(200 - 5 * j);
        cand_ids[1][j] = @intCast(3 + 7 * j);
        // Integers and quarters are exact in bf16: the top half of the f32 bits.
        vals[0][j] = @truncate(@as(u32, @bitCast(-@as(f32, @floatFromInt(j)))) >> 16);
        vals[1][j] = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(j)) * 0.25)) >> 16);
    }
    // Row 0: ids 200 (slot 0) and 45 (slot 31) tie at the max; the lower id wins.
    vals[0][31] = vals[0][0];
    var lists: [2]mtp_mod.Shortlist = undefined;
    for (0..2) |r| {
        lists[r] = .{
            .cands = mlx.mlx_array_new_data(&cand_ids[r], &[_]c_int{32}, 1, .uint32),
            .exact = mlx.mlx_array_new_data(&vals[r], &[_]c_int{ 1, 1, 32 }, 3, .bfloat16),
            .rows = vocab,
        };
    }
    defer for (&lists) |*sl| sl.deinit();
    const sparse = try sparseLogits(s, &lists, vocab);
    defer _ = mlx.mlx_array_free(sparse.idx);
    defer _ = mlx.mlx_array_free(sparse.logits);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, sparse.logits, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const got = (mlx.mlx_array_data_float32(f) orelse return error.Unreadable)[0 .. 2 * 256];
    for (0..2) |r| {
        var expect: [256]f32 = @splat(-std.math.inf(f32));
        for (0..32) |j| expect[cand_ids[r][j]] = @bitCast(@as(u32, vals[r][j]) << 16);
        try std.testing.expectEqualSlices(f32, &expect, got[r * 256 ..][0..256]);
    }
    var am = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, sparse.logits, -1, false, s));
    var am32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am32);
    try mlx.check(mlx.mlx_astype(&am32, am, .int32, s));
    try mlx.check(mlx.mlx_array_eval(am32));
    const ids = mlx.mlx_array_data_int32(am32) orelse return error.Unreadable;
    try std.testing.expectEqual(@as(i32, 45), ids[0]);
    try std.testing.expectEqual(@as(i32, 3 + 7 * 31), ids[1]);
}
