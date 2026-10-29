const trf = @import("../transformer.zig");
const std = @import("std");
const gdn_decode = @import("gdn_decode.zig");
const mlx = @import("../mlx.zig");
const kv_quant = @import("../kv_quant.zig");
const KVQuantConfig = kv_quant.KVQuantConfig;
const testing = std.testing;
const test_support = @import("../test_support.zig");
const qwen4ReadF32 = test_support.qwen4ReadF32;
const ForwardCtx = trf.ForwardCtx;
const KVCache = trf.KVCache;
const SSMCacheEntry = trf.SSMCacheEntry;
const Transformer = trf.Transformer;
const nextSsmMemberGen = trf.nextSsmMemberGen;
const ssmFreeQsaState = trf.ssmFreeQsaState;
const ssmFreeSpecCapture = trf.ssmFreeSpecCapture;
const ssmRollbackFromCapture = trf.ssmRollbackFromCapture;

/// A QSA-shaped bool mask [1,1,qL,kL]: causal, bottom-right aligned, with a
/// random half of the complete 4-token blocks dropped per row and the row's
/// own tail always visible — the shape `qsaMask` emits.
pub fn attn256QsaMask(rnd: std.Random, qL: c_int, kL: c_int) !mlx.mlx_array {
    const n: usize = @intCast(qL * kL);
    const buf = try std.testing.allocator.alloc(bool, n);
    defer std.testing.allocator.free(buf);
    const q_off = kL - qL;
    for (0..@intCast(qL)) |r| {
        const p: c_int = q_off + @as(c_int, @intCast(r));
        const tail_start: c_int = 4 * @divTrunc(p + 1, 4);
        for (0..@intCast(kL)) |c| {
            const ci: c_int = @intCast(c);
            const blk_vis = (@rem(ci, 4) == 0 and rnd.boolean()) or (@rem(ci, 4) != 0 and buf[r * @as(usize, @intCast(kL)) + c - 1]);
            buf[r * @as(usize, @intCast(kL)) + c] = ci <= p and (ci >= tail_start or blk_vis);
        }
    }
    const shape = [_]c_int{ 1, 1, qL, kL };
    return mlx.mlx_array_new_data(buf.ptr, &shape, 4, .bool_);
}

pub fn qwen4ExpectMoeBitsEqual(expected: mlx.mlx_array, actual: mlx.mlx_array) !void {
    try mlx.check(mlx.mlx_array_eval(expected));
    try mlx.check(mlx.mlx_array_eval(actual));
    if (mlx.mlx_array_dtype(expected) != .bfloat16 or mlx.mlx_array_dtype(actual) != .bfloat16) return error.UnexpectedMoeDtype;
    const expected_len = mlx.mlx_array_size(expected);
    try testing.expectEqual(expected_len, mlx.mlx_array_size(actual));
    try testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(expected).?[0..expected_len], mlx.mlx_array_data_bfloat16(actual).?[0..expected_len]);
}

/// One per-request slot for the qwen4 batched-vs-serial test.
pub const Qwen4TestSlot = struct {
    cache: KVCache,
    off: usize = 0,
    entries: []SSMCacheEntry,
    ctx: ForwardCtx = undefined,
    /// The MTP head's per-request stash, so a committed-state bar can assert the trunk
    /// verify left it where the solo round did.
    head: Transformer.Qwen4MtpState,

    pub fn init(alloc: std.mem.Allocator, n_layers: u32) !*Qwen4TestSlot {
        return initWith(alloc, n_layers, KVQuantConfig.dense);
    }
    pub fn initWith(alloc: std.mem.Allocator, n_layers: u32, kv: KVQuantConfig) !*Qwen4TestSlot {
        const sl = try alloc.create(Qwen4TestSlot);
        const cache = try KVCache.initWithConfig(alloc, n_layers, kv);
        const head_cache = try KVCache.init(alloc, 1);
        sl.* = .{
            .cache = cache,
            .entries = try alloc.alloc(SSMCacheEntry, n_layers),
            .head = .{
                .cache = head_cache,
                .entry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            },
        };
        for (sl.entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        sl.ctx = .{ .cache = &sl.cache, .moe_seq_offset = &sl.off, .ssm_entries = sl.entries, .ssm_member_gen = nextSsmMemberGen(), .capture_hidden = null, .vision_embeddings = null };
        return sl;
    }
    pub fn deinit(sl: *Qwen4TestSlot, alloc: std.mem.Allocator) void {
        for (sl.entries) |*e| {
            ssmFreeSpecCapture(e);
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        }
        alloc.free(sl.entries);
        _ = mlx.mlx_array_free(sl.head.entry.conv_state);
        _ = mlx.mlx_array_free(sl.head.entry.ssm_state);
        sl.head.cache.deinit();
        sl.cache.deinit();
        alloc.destroy(sl);
    }
    pub fn forward(sl: *Qwen4TestSlot, xfm: *Transformer, ids: []const i32) !mlx.mlx_array {
        const shape = [_]c_int{ 1, @intCast(ids.len) };
        const arr = mlx.mlx_array_new_data(@ptrCast(ids.ptr), &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(arr);
        return xfm.forwardWith(&sl.ctx, arr);
    }
};

const Qwen4BatchedPleRun = struct {
    logits: mlx.mlx_array,
    pending: bool,
};

pub fn qwen4BatchedPleForward(
    xfm: *Transformer,
    slots: []const *Qwen4TestSlot,
    ids: []const i32,
    seq_len: c_int,
    capture: bool,
    defer_ple: bool,
    do_flush: bool,
) !Qwen4BatchedPleRun {
    const N = slots.len;
    const ctxs = try xfm.allocator.alloc(*ForwardCtx, N);
    defer xfm.allocator.free(ctxs);
    var off_buf = try xfm.allocator.alloc(i32, N);
    defer xfm.allocator.free(off_buf);
    for (slots, 0..) |sl, i| {
        ctxs[i] = &sl.ctx;
        off_buf[i] = @intCast(sl.off);
    }
    const ml = xfm.moe_layers.?;
    const merged = try xfm.allocator.alloc(SSMCacheEntry, ml.len);
    defer xfm.allocator.free(merged);
    for (merged) |*m| m.* = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false };
    defer for (merged) |*m| {
        if (m.conv_state.ctx != null) _ = mlx.mlx_array_free(m.conv_state);
        if (m.ssm_state.ctx != null) _ = mlx.mlx_array_free(m.ssm_state);
        if (m.aux_state.ctx != null) _ = mlx.mlx_array_free(m.aux_state);
        ssmFreeSpecCapture(m);
    };
    for (ml, 0..) |*lw, l| {
        if (!lw.is_linear and lw.ple == null) continue;
        merged[l] = try xfm.mergeSsmAcrossSlots(ctxs, l);
    }
    const off_shape = [_]c_int{@intCast(N)};
    const rope_offset_arr = mlx.mlx_array_new_data(off_buf.ptr, &off_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(rope_offset_arr);
    var scratch_offset: usize = @intCast(off_buf[0]);
    var bctx: ForwardCtx = .{
        .cache = ctxs[0].cache,
        .moe_seq_offset = &scratch_offset,
        .ssm_entries = merged,
        .capture_hidden = null,
        .capture_ssm_seq = capture,
        .vision_embeddings = null,
        .batch_slots = ctxs,
        .batch_rope_offsets = rope_offset_arr,
        .ple_defer = defer_ple,
    };
    const tshape = [_]c_int{ @intCast(N), seq_len };
    const tarr = mlx.mlx_array_new_data(@ptrCast(ids.ptr), &tshape, 2, .int32);
    defer _ = mlx.mlx_array_free(tarr);
    const logits = try xfm.forwardQwen4With(&bctx, tarr);
    errdefer _ = mlx.mlx_array_free(logits);
    const pending = bctx.ple_pending != null;
    if (do_flush) try xfm.flushDeferredPle(&bctx) else if (bctx.ple_pending != null) xfm.discardDeferredPle(&bctx);
    for (merged, 0..) |*m, li| {
        if (!ml[li].is_linear and ml[li].ple == null) continue;
        try xfm.splitSsmToSlots(m, ctxs, li);
    }
    for (slots) |sl| sl.off += @intCast(seq_len);
    return .{ .logits = logits, .pending = pending };
}

pub fn expectRowsByteEqual(alloc: std.mem.Allocator, s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array, tag: []const u8) !void {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var ok: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&ok, eq));
    if (ok) return;
    const ha = try qwen4ReadF32(alloc, a, s);
    defer alloc.free(ha);
    const hb = try qwen4ReadF32(alloc, b, s);
    defer alloc.free(hb);
    var max_diff: f64 = 0;
    var first: usize = std.math.maxInt(usize);
    var n_diff: usize = 0;
    for (ha, hb, 0..) |x, y, i| {
        if (x == y) continue;
        n_diff += 1;
        if (first == std.math.maxInt(usize)) first = i;
        const d = @abs(@as(f64, x) - @as(f64, y));
        if (d > max_diff) max_diff = d;
    }
    std.debug.print("[mtp verify] {s}: NOT byte-equal — {d}/{d} elements differ, first at {d} ({d} vs {d}), max |diff| {d}\n", .{ tag, n_diff, ha.len, first, ha[first], hb[first], max_diff });
    return error.RowNotByteEqual;
}

/// The solo MTP verify: capture armed, PLE deferred, one `forwardWithCaptureAll`, flush
/// after. `forwardRowAxisVerify` at one row must match it.
pub fn logicalSsmCapture(s: mlx.mlx_stream, entry: *const SSMCacheEntry) !mlx.mlx_array {
    if (entry.spec_state_seq.ctx == null and entry.spec_replay.held()) {
        // A replaying verify: the state after every accepted prefix, rebuilt, then the final state.
        const rows = mlx.getShape(entry.spec_conv_input)[1] - 3;
        var parts: [gdn_decode.MAX_SEQ]mlx.mlx_array = undefined;
        var built: usize = 0;
        defer for (parts[0..built]) |part| {
            _ = mlx.mlx_array_free(part);
        };
        var i: c_int = 1;
        while (i < rows) : (i += 1) {
            const state = try gdn_decode.replayState(entry.spec_replay, entry.spec_conv_input, i, s);
            defer _ = mlx.mlx_array_free(state);
            parts[built] = mlx.mlx_array_new();
            built += 1;
            try mlx.check(mlx.mlx_expand_dims(&parts[built - 1], state, 0, s));
        }
        parts[built] = mlx.mlx_array_new();
        built += 1;
        try mlx.check(mlx.mlx_expand_dims(&parts[built - 1], entry.ssm_state, 0, s));
        return Transformer.concatAxis0(s, parts[0..built]);
    }
    const shape = mlx.getShape(entry.spec_state_seq);
    if (shape.len != 5 or shape[0] < 1) return error.SpecCaptureShape;
    var final = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(final);
    try mlx.check(mlx.mlx_expand_dims(&final, entry.ssm_state, 0, s));
    if (shape[0] == 1) return final;
    defer _ = mlx.mlx_array_free(final);
    var prefix = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prefix);
    var stop: [5]c_int = undefined;
    @memcpy(&stop, shape);
    stop[0] -= 1;
    const start: [5]c_int = @splat(0);
    const stride: [5]c_int = @splat(1);
    try mlx.check(mlx.mlx_slice(&prefix, entry.spec_state_seq, &start, 5, &stop, 5, &stride, 5, s));
    return Transformer.concatAxis0(s, &.{ prefix, final });
}

pub fn rollbackRowToAccepted(sl: *Qwen4TestSlot, accepted: u32, verify_len: u32, off_snap: usize, step_snap: usize, s: mlx.mlx_stream) !void {
    const accepted_len: usize = 1 + @as(usize, accepted);
    try sl.cache.truncate(off_snap + accepted_len, s);
    sl.cache.step = step_snap;
    // Full acceptance keeps the dedicated final state; the capture tail is unwritten.
    if (accepted_len < verify_len) for (sl.entries) |*e| {
        try ssmRollbackFromCapture(e, accepted, verify_len, s);
    };
    sl.off = off_snap + accepted_len;
    for (sl.entries) |*e| ssmFreeSpecCapture(e);
}

pub fn expectSlotStateEqual(alloc: std.mem.Allocator, s: mlx.mlx_stream, a: *Qwen4TestSlot, b: *Qwen4TestSlot, row: usize) !void {
    try testing.expectEqual(a.off, b.off);
    try testing.expectEqual(a.cache.step, b.cache.step);
    try testing.expectEqual(a.entries.len, b.entries.len);
    var buf: [96]u8 = undefined;
    for (a.entries, b.entries, 0..) |*ea, *eb, l| {
        try testing.expectEqual(ea.initialized, eb.initialized);
        try testing.expectEqual(ea.ple_prev_valid, eb.ple_prev_valid);
        try testing.expectEqualSlices(u32, &ea.ple_prev, &eb.ple_prev);
        try testing.expectEqual(a.cache.entries[l].offset, b.cache.entries[l].offset);
        inline for (.{ "conv_state", "ssm_state", "qsa_pooled", "aux_state" }) |field| {
            const xa = @field(ea, field);
            const xb = @field(eb, field);
            try testing.expectEqual(xa.ctx == null, xb.ctx == null);
            if (xa.ctx != null) {
                const tag = try std.fmt.bufPrint(&buf, "row {d} layer {d} {s}", .{ row, l, field });
                try expectRowsByteEqual(alloc, s, xa, xb, tag);
            }
        }
    }
    const ma = a.head.meta();
    const mb = b.head.meta();
    try testing.expectEqual(ma.stash_len, mb.stash_len);
    try testing.expectEqual(ma.origin, mb.origin);
    try testing.expectEqual(ma.cache_step, mb.cache_step);
    try testing.expectEqual(a.head.cache.entries[0].offset, b.head.cache.entries[0].offset);
    try testing.expectEqual(a.head.entry.initialized, b.head.entry.initialized);
    try testing.expectEqualSlices(u32, &a.head.entry.ple_prev, &b.head.entry.ple_prev);
}
