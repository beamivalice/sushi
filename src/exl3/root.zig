//! EXL3 trellis experts: routed MoE banks in turboderp's EXL3 format, decoded
//! and multiplied by Metal kernels. A module, so another MLX host (mlx-serve)
//! can serve EXL3 packs through the same code: it needs an `mlx_host` import
//! whose root exposes `pub const mlx`, `log` and `io_util`.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
pub const format = @import("expert_exl3.zig");
pub const kernels = @import("expert_exl3_kernels.zig");
pub const glm_group2 = @import("glm_group2.zig");
pub const glm_prefill_grid = @import("glm_prefill_grid.zig");

/// One projection's bank, expert e on axis 0: `trellis` U16 [E, in/16, out/16, n],
/// `suh` F16 [E, in], `svh` F16 [E, out].
pub const Proj = struct { trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array };
pub const group_layout = @import("layout.zig");
pub const GroupLayout = group_layout.Layout;
pub const Bank = struct { gate: Proj, up: Proj, down: Proj };

/// Routed SwiGLU experts over `x` [B, S, D] for the top-k `inds`/`scores` [B, S, K],
/// returned in x's shape and dtype. Up to `kernels.DECODE_ROWS_MAX` rows (or any
/// verify rows) take the decode chain, wider the prefill GEMM.
pub fn moe(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool) !mlx.mlx_array {
    return moeOutput(s, x, bank, inds, scores, dec, verify_rows, mlx.mlx_array_dtype(x), null);
}

/// Add an already gated shared expert, preserving the routed output's rounding.
pub fn moeWithShared(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool, shared: mlx.mlx_array) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    if (xsh[0] * xsh[1] == 1 and mlx.mlx_array_dtype(shared) == mlx.mlx_array_dtype(x) and
        mlx.mlx_array_size(shared) == mlx.mlx_array_size(x) and mlx.getShape(bank.gate.trellis)[3] == mlx.getShape(bank.up.trellis)[3])
        return moeOutput(s, x, bank, inds, scores, dec, verify_rows, mlx.mlx_array_dtype(x), shared);
    const routed = try moe(s, x, bank, inds, scores, dec, verify_rows);
    defer _ = mlx.mlx_array_free(routed);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, routed, shared, s));
    return out;
}

fn moeOutput(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool, out_dtype: mlx.mlx_dtype, shared: ?mlx.mlx_array) !mlx.mlx_array {
    return moeOutputClamped(s, x, bank, inds, scores, dec, verify_rows, out_dtype, shared, 0);
}

/// GLM's gate is upper-clamped and its up branch is symmetrically clamped,
/// before SwiGLU and the down projection. Packed expert rates remain 2–4 bpw.
pub fn moeClamped(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, limit: c_int) !mlx.mlx_array {
    return moeClampedWithShared(s, x, bank, inds, scores, dec, limit, null);
}

/// A decode row may fold an already computed shared expert (same dtype and size as the output) into the finish reduce.
pub fn moeClampedWithShared(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, limit: c_int, shared: ?mlx.mlx_array) !mlx.mlx_array {
    if (limit <= 0 or limit > 128) return error.InvalidExl3Clamp;
    const sh = mlx.getShape(x);
    const ix = mlx.getShape(inds);
    if (sh.len != 3 or ix.len != 3 or sh[0] <= 0 or sh[1] <= 0 or sh[0] != ix[0] or sh[1] != ix[1] or ix[2] <= 0 or ix[2] > kernels.REDUCE_MAX_TOPK) return error.BadExl3Shape;
    if (sh[2] <= 0 or @mod(sh[2], 128) != 0 or !std.mem.eql(c_int, ix, mlx.getShape(scores))) return error.BadExl3Shape;
    const xd = mlx.mlx_array_dtype(x);
    const id = mlx.mlx_array_dtype(inds);
    const sd = mlx.mlx_array_dtype(scores);
    if ((xd != .bfloat16 and xd != .float32 and xd != .float16) or
        (id != .uint32 and id != .int32) or
        (sd != .float32 and sd != .bfloat16 and sd != .float16)) return error.BadExl3Dtype;
    const rows = std.math.mul(c_int, sh[0], sh[1]) catch return error.BadExl3Shape;
    _ = std.math.mul(c_int, rows, ix[2]) catch return error.BadExl3Shape;
    const gs = mlx.getShape(bank.gate.trellis);
    if (gs.len != 4 or gs[0] <= 0 or gs[2] <= 0 or gs[2] > std.math.maxInt(c_int) / 16) return error.BadExl3Shape;
    const width = gs[2] * 16;
    try validateClampedProjection(bank.gate, gs[0], sh[2], width);
    try validateClampedProjection(bank.up, gs[0], sh[2], width);
    try validateClampedProjection(bank.down, gs[0], width, sh[2]);
    return moeOutputClamped(s, x, bank, inds, scores, dec, false, mlx.mlx_array_dtype(x), shared, limit);
}

pub fn validateClampedProjection(p: Proj, experts: c_int, input: c_int, output: c_int) !void {
    const sh = mlx.getShape(p.trellis);
    if (input <= 0 or output <= 0 or @mod(input, 128) != 0 or @mod(output, 128) != 0 or
        sh.len != 4 or sh[0] != experts or sh[1] != @divExact(input, 16) or sh[2] != @divExact(output, 16) or sh[3] < 0 or
        format.kFromPackedDim(@intCast(sh[3])) == null or
        !std.mem.eql(c_int, &.{ experts, input }, mlx.getShape(p.suh)) or
        !std.mem.eql(c_int, &.{ experts, output }, mlx.getShape(p.svh))) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(p.trellis) != .uint16 or mlx.mlx_array_dtype(p.suh) != .float16 or mlx.mlx_array_dtype(p.svh) != .float16) return error.BadExl3Dtype;
}

fn moeOutputClamped(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool, out_dtype: mlx.mlx_dtype, shared: ?mlx.mlx_array, limit: c_int) !mlx.mlx_array {
    // Which kernel a dispatch picks is read off ONE process-global codebook,
    // and several EXL3 packs can be resident at once: set it per call.
    kernels.setDecodeParams(dec);
    const xsh = mlx.getShape(x);
    const xd = out_dtype;
    const B = xsh[0];
    const S = xsh[1];
    const D = xsh[xsh.len - 1];
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ B * S, D }, 2, s));
    const ish = mlx.getShape(inds);
    const K = ish[ish.len - 1];
    var slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slots);
    try mlx.check(mlx.mlx_reshape(&slots, inds, &[_]c_int{B * S * K}, 1, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_reshape(&sc, scores, &[_]c_int{B * S * K}, 1, s));
    var slots_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slots_u);
    try mlx.check(mlx.mlx_astype(&slots_u, slots, .uint32, s));
    const rows: usize = @intCast(B * S);
    if (rows >= 2) {
        kernels.dumpUnionHist(slots_u, rows, @intCast(K)) catch {};
    }
    const g = bank.gate;
    const u = bank.up;
    const d = bank.down;
    const y = if (limit > 0)
        try kernels.moeSwigluClampedWithShared(s, x2, g.trellis, g.suh, g.svh, u.trellis, u.suh, u.svh, d.trellis, d.suh, d.svh, slots_u, sc, K, limit, xd, shared)
    else if (rows <= kernels.DECODE_ROWS_MAX or verify_rows)
        if (mlx.getShape(g.trellis)[3] == mlx.getShape(u.trellis)[3])
            try kernels.moeSwigluFusedWithShared(s, x2, g.trellis, g.suh, g.svh, u.trellis, u.suh, u.svh, d.trellis, d.suh, d.svh, slots_u, sc, xd, shared)
        else
            try moeMixedDecode(s, x2, bank, slots_u, sc, K, xd)
    else
        try kernels.moePrefill(s, x2, g.trellis, g.suh, g.svh, u.trellis, u.suh, u.svh, d.trellis, d.suh, d.svh, slots_u, sc, K);
    defer _ = mlx.mlx_array_free(y);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, xsh.ptr, @intCast(xsh.len), s));
    if (mlx.mlx_array_dtype(out) != xd) {
        var cast = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&cast, out, xd, s));
        _ = mlx.mlx_array_free(out);
        return cast;
    }
    return out;
}

pub const Spec = struct {
    rate: format.Rate,
    codebook: format.Codebook,
    window: format.Window = .w16,
};

/// `k` is a rate, integer or fractional: admitted only when 16k is an even
/// whole number of halfwords per tile, which is what every reader indexes by.
fn rateFromConfigK(k_v: std.json.Value) ?format.Rate {
    const scaled: f64 = switch (k_v) {
        .integer => |i| @as(f64, @floatFromInt(i)) * 16.0,
        .float => |f| f * 16.0,
        else => return null,
    };
    const rounded = @round(scaled);
    if (@abs(scaled - rounded) > 1e-6) return null;
    if (rounded < 0 or rounded > 1024) return null;
    return format.kFromPackedDim(@intFromFloat(rounded));
}

/// `window` is the codeword width the pack's search hashed. Absent means 16,
/// the whole sliding window; a width this build cannot decode is refused, not
/// rounded — the same bitstream decodes to different weights at each width.
fn windowFromConfig(v: ?std.json.Value) ?format.Window {
    const raw = v orelse return format.Window.w16;
    if (raw != .integer) return null;
    return format.Window.fromBits(raw.integer);
}

pub fn parseExpertQuant(obj: std.json.ObjectMap) !Spec {
    const block = obj.get("expert_quant") orelse return error.ExpertLayoutUnsupported;
    if (block != .object) return error.ExpertLayoutUnsupported;
    const fmt_v = block.object.get("format") orelse return error.ExpertLayoutUnsupported;
    if (fmt_v != .string or !std.mem.eql(u8, fmt_v.string, "exl3")) return error.ExpertLayoutUnsupported;
    const k_v = block.object.get("k") orelse return error.ExpertLayoutUnsupported;
    const rate = rateFromConfigK(k_v) orelse return error.ExpertLayoutUnsupported;
    const cb_v = block.object.get("codebook") orelse return error.ExpertLayoutUnsupported;
    if (cb_v != .string) return error.ExpertLayoutUnsupported;
    const codebook = format.Codebook.fromName(cb_v.string) orelse return error.ExpertLayoutUnsupported;
    const window = windowFromConfig(block.object.get("window")) orelse return error.Exl3WindowUnsupported;
    return .{ .rate = rate, .codebook = codebook, .window = window };
}

fn specFromConfigJson(allocator: std.mem.Allocator, raw: []const u8) !Spec {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    defer parsed.deinit();
    return parseExpertQuant(parsed.value.object);
}

test "an EXL3 pack's codeword window is 16 unless its config names one this build decodes" {
    const t = std.testing;
    const base = "{\"expert_quant\":{\"format\":\"exl3\",\"k\":2.5,\"codebook\":\"mcg\"";
    try t.expectEqual(format.Window.w16, (try specFromConfigJson(t.allocator, base ++ "}}")).window);
    try t.expectEqual(format.Window.w16, (try specFromConfigJson(t.allocator, base ++ ",\"window\":16}}")).window);
    try t.expectEqual(format.Window.w12, (try specFromConfigJson(t.allocator, base ++ ",\"window\":12}}")).window);
    try t.expectEqual(format.Window.w11, (try specFromConfigJson(t.allocator, base ++ ",\"window\":11}}")).window);
    try t.expectEqual(format.Window.w10, (try specFromConfigJson(t.allocator, base ++ ",\"window\":10}}")).window);
    try t.expectEqual(format.Window.w8, (try specFromConfigJson(t.allocator, base ++ ",\"window\":8}}")).window);
    // The same bitstream decodes to different weights at each width, so an
    // unreadable one is a refusal, never a fallback to 16.
    try t.expectError(error.Exl3WindowUnsupported, specFromConfigJson(t.allocator, base ++ ",\"window\":17}}"));
    try t.expectError(error.Exl3WindowUnsupported, specFromConfigJson(t.allocator, base ++ ",\"window\":7}}"));
    try t.expectError(error.Exl3WindowUnsupported, specFromConfigJson(t.allocator, base ++ ",\"window\":\"12\"}}"));
}

test "an EXL3 pack naming a codebook this build does not decode is refused" {
    const t = std.testing;
    try t.expectError(error.ExpertLayoutUnsupported, specFromConfigJson(t.allocator,
        \\{"expert_quant":{"format":"exl3","k":2.5,"codebook":"mul2","window":12}}
    ));
}

pub fn admitTopK(topk: u32) !void {
    if (topk > kernels.REDUCE_MAX_TOPK) return error.Exl3TopKExceedsReduceBank;
}

pub fn trellisAdmitted(shape: []const c_int, experts: u32, in_dim: u32, out_dim: u32, rate: format.Rate) bool {
    if (shape.len != 4) return false;
    for (shape) |d| if (d <= 0) return false;
    if (in_dim % 16 != 0 or out_dim % 16 != 0) return false;
    const want = [3]u32{ experts, in_dim / 16, out_dim / 16 };
    for (want, 0..) |w, i| if (w != @as(u32, @intCast(shape[i]))) return false;
    const packed_rate = format.kFromPackedDim(@intCast(shape[3])) orelse return false;
    return packed_rate.n <= rate.n;
}

test "exl3 top-k above reduce-bank is a named refusal" {
    const t = std.testing;
    try admitTopK(10);
    try admitTopK(16);
    try admitTopK(32);
    try t.expectError(error.Exl3TopKExceedsReduceBank, admitTopK(33));
}

test "exl3 expert_quant admits integer and fractional K under every served codebook and refuses the rest" {
    const t = std.testing;
    const cases = [_]struct { text: []const u8, n: u32 }{
        .{ .text = "2", .n = 32 },
        .{ .text = "2.5", .n = 40 },
        .{ .text = "2.75", .n = 44 },
        .{ .text = "3", .n = 48 },
        .{ .text = "3.5", .n = 56 },
        .{ .text = "4", .n = 64 },
    };
    for (cases) |c| {
        for ([_]format.Codebook{ .mul1, .mcg }) |cb| {
            var buf: [96]u8 = undefined;
            const raw = try std.fmt.bufPrint(&buf, "{{\"expert_quant\":{{\"format\":\"exl3\",\"k\":{s},\"codebook\":\"{s}\"}}}}", .{ c.text, @tagName(cb) });
            const ok = try std.json.parseFromSlice(std.json.Value, t.allocator, raw, .{});
            defer ok.deinit();
            const spec = try parseExpertQuant(ok.value.object);
            try t.expectEqual(c.n, spec.rate.n);
            try t.expectEqual(cb, spec.codebook);
        }
    }
    // 2.3 is not a multiple of 1/16; 4.5 and 6 are off the served range.
    for ([_][]const u8{ "2.3", "8.125", "9", "0.875", "2.0625" }) |bad| {
        var buf: [96]u8 = undefined;
        const raw = try std.fmt.bufPrint(&buf, "{{\"expert_quant\":{{\"format\":\"exl3\",\"k\":{s},\"codebook\":\"mul1\"}}}}", .{bad});
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, raw, .{});
        defer parsed.deinit();
        try t.expectError(error.ExpertLayoutUnsupported, parseExpertQuant(parsed.value.object));
    }
    const bad_cb = try std.json.parseFromSlice(std.json.Value, t.allocator,
        \\{"expert_quant":{"format":"exl3","k":4,"codebook":"mul2"}}
    , .{});
    defer bad_cb.deinit();
    try t.expectError(error.ExpertLayoutUnsupported, parseExpertQuant(bad_cb.value.object));
    const missing = try std.json.parseFromSlice(std.json.Value, t.allocator, "{}", .{});
    defer missing.deinit();
    try t.expectError(error.ExpertLayoutUnsupported, parseExpertQuant(missing.value.object));
}

test "exl3 a layer may pack below the rate the config bills, never above it" {
    const t = std.testing;
    const E: u32 = 256;
    const h: u32 = 4096;
    const i: u32 = 2048;
    const k4: format.Rate = .{ .n = 64 };
    const shape = struct {
        fn at(n: c_int) [4]c_int {
            return .{ @intCast(E), @intCast(h / 16), @intCast(i / 16), n };
        }
    }.at;
    // A pack whose tail layers carry more bits than its body: the config bills
    // the widest, so a narrower layer admits and a wider one cannot.
    try t.expect(trellisAdmitted(&shape(64), E, h, i, k4));
    try t.expect(trellisAdmitted(&shape(40), E, h, i, k4));
    try t.expect(!trellisAdmitted(&shape(64), E, h, i, .{ .n = 40 }));
    try t.expect(!trellisAdmitted(&shape(41), E, h, i, k4));
    try t.expect(!trellisAdmitted(&shape(64), E, h, h, k4));
}

test {
    _ = format;
    _ = kernels;
}

test "exl3 Sushi CPU config and packed shapes admit K1 through K8" {
    const t = std.testing;
    for (0..145) |n| {
        const k = @as(f64, @floatFromInt(n)) / 16;
        const admitted = n >= 16 and n <= 128 and n % 2 == 0;
        try t.expectEqual(admitted, rateFromConfigK(.{ .float = k }) != null);
        try t.expectEqual(admitted, trellisAdmitted(&.{ 2, 8, 16, @intCast(n) }, 2, 128, 256, .{ .n = 128 }));
        if (!admitted) continue;
        for ([_][]const u8{ "mul1", "mcg" }) |cb| {
            const raw = try std.fmt.allocPrint(t.allocator, "{{\"expert_quant\":{{\"format\":\"exl3\",\"k\":{d},\"codebook\":\"{s}\"}}}}", .{ k, cb });
            defer t.allocator.free(raw);
            try t.expectEqual(@as(u32, @intCast(n)), (try specFromConfigJson(t.allocator, raw)).rate.n);
        }
        if (n > 16) try t.expect(!trellisAdmitted(&.{ 2, 8, 16, @intCast(n) }, 2, 128, 256, .{ .n = @intCast(n - 2) }));
    }
    for ([_]i64{ 1, 2, 3, 4, 5, 6, 7, 8 }) |k| try t.expectEqual(@as(u32, @intCast(k * 16)), rateFromConfigK(.{ .integer = k }).?.n);
}

test "sushi coder layouts validate groups route slots and bill stored bytes" {
    const t = std.testing;
    var plan = GroupLayout.init(128, 128, .{ .n = 64 });
    for ([_]u32{ 2, 1 }, 0..) |e, g| {
        for ([_][]const u8{ "gate", "up", "down" }, 0..) |proj, p| {
            var buf: [96]u8 = undefined;
            const n: u32 = @intCast(32 + 16 * ((g + p) % 3));
            _ = try plan.add(try std.fmt.bufPrint(&buf, "{s}_proj.g{d}.trellis", .{ proj, g }), &.{ e, 8, 8, n }, .u16);
            _ = try plan.add(try std.fmt.bufPrint(&buf, "{s}_proj.g{d}.suh", .{ proj, g }), &.{ e, 128 }, .f16);
            _ = try plan.add(try std.fmt.bufPrint(&buf, "{s}_proj.g{d}.svh", .{ proj, g }), &.{ e, 128 }, .f16);
        }
    }
    try plan.finish(3, 2, false, 512);
    try t.expectEqual(@as(usize, 2), plan.count);
    try t.expectEqual(@as(u64, 3 * (8 * 8 * (32 + 48 + 64) * 2 + 3 * 256 * 2)), plan.bytes);
    try t.expectEqual(@as(u32, 1), (try plan.locate(2)).group);
    try t.expectEqual(@as(u32, 0), (try plan.locate(2)).local);
    try t.expectEqual(@as(u32, 1), (try plan.locate(1)).local);
    try t.expectError(error.ExpertOutOfRange, plan.locate(3));
    try t.expectError(error.Exl3RouterWidthMismatch, plan.finish(4, 2, false, 512));
    try t.expectError(error.Exl3TopKExceedsExperts, plan.finish(3, 4, false, 512));
    try t.expectError(error.Exl3RaggedStreamingUnsupported, plan.finish(3, 2, true, 512));
}

test "sushi coder layouts refuse malformed missing mixed and unsupported groups" {
    const t = std.testing;
    var plan = GroupLayout.init(128, 128, .{ .n = 64 });
    try t.expectError(error.Exl3GroupNameInvalid, plan.add("gate_proj.g01.trellis", &.{ 1, 8, 8, 32 }, .u16));
    try t.expectError(error.Exl3TrellisGeometry, plan.add("gate_proj.g0.trellis", &.{ 1, 8, 8, 33 }, .u16));
    try t.expectError(error.Exl3GroupGeometry, plan.add("gate_proj.g0.trellis", &.{ 0, 8, 8, 32 }, .u16));
    try t.expectError(error.Exl3GroupGeometry, plan.add("gate_proj.g0.suh", &.{ 1, 64 }, .f16));
    _ = try plan.add("gate_proj.g1.trellis", &.{ 1, 8, 8, 32 }, .u16);
    try t.expectError(error.Exl3GroupMissing, plan.finish(1, 1, false, 512));
    try t.expectError(error.Exl3GroupGeometry, plan.add("up_proj.g1.trellis", &.{ 2, 8, 8, 32 }, .u16));
    try t.expectError(error.Exl3MixedGroupLayout, plan.add("gate_proj.trellis", &.{ 1, 8, 8, 32 }, .u16));
    try t.expectError(error.Exl3GroupDtype, plan.add("up_proj.g1.suh", &.{ 1, 128 }, .u16));
    try t.expectError(error.Exl3GroupNameInvalid, plan.add("gate_proj.g32.trellis", &.{ 1, 8, 8, 32 }, .u16));
}

test "sushi coder GPU two groups equal independently routed uniform banks" {
    const t = std.testing;
    const s = mlx.gpuStream();
    var banks: [2]Bank = undefined;
    var held: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (held.items) |a| _ = mlx.mlx_array_free(a);
        held.deinit(t.allocator);
    }
    for (&banks, 0..) |*bank, group| {
        for ([_]*Proj{ &bank.gate, &bank.up, &bank.down }, 0..) |p, projection| {
            const n: c_int = @intCast(32 + 16 * ((group + projection) % 3));
            const words = try t.allocator.alloc(u16, @intCast(8 * 8 * n));
            defer t.allocator.free(words);
            for (words, 0..) |*v, i| v.* = @truncate(i *% 7919 +% 23);
            const signs: [128]u16 = @splat(format.f32ToF16Bits(0.125));
            p.trellis = mlx.mlx_array_new_data(words.ptr, &[_]c_int{ 1, 8, 8, n }, 4, .uint16);
            try held.append(t.allocator, p.trellis);
            p.suh = mlx.mlx_array_new_data(&signs, &[_]c_int{ 1, 128 }, 2, .float16);
            try held.append(t.allocator, p.suh);
            p.svh = mlx.mlx_array_new_data(&signs, &[_]c_int{ 1, 128 }, 2, .float16);
            try held.append(t.allocator, p.svh);
        }
    }
    const xh: [128]u16 = @splat(format.f32ToF16Bits(0.25));
    const x = mlx.mlx_array_new_data(&xh, &[_]c_int{ 1, 1, 128 }, 3, .float16);
    defer _ = mlx.mlx_array_free(x);
    const ids = mlx.mlx_array_new_data(&[_]i32{ 1, 0 }, &[_]c_int{ 1, 1, 2 }, 3, .int32);
    defer _ = mlx.mlx_array_free(ids);
    const scores = mlx.mlx_array_new_data(&[_]f32{ 0.25, 0.75 }, &[_]c_int{ 1, 1, 2 }, 3, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const got = try moeGroups(s, x, &banks, ids, scores, .mcg, false);
    defer _ = mlx.mlx_array_free(got);
    const one_id = mlx.mlx_array_new_data(&[_]i32{0}, &[_]c_int{ 1, 1, 1 }, 3, .int32);
    defer _ = mlx.mlx_array_free(one_id);
    var outputs: [2]mlx.mlx_array = undefined;
    for (&outputs, 0..) |*out, i| {
        const score = mlx.mlx_array_new_data(&[_]f32{if (i == 0) 0.75 else 0.25}, &[_]c_int{ 1, 1, 1 }, 3, .float32);
        defer _ = mlx.mlx_array_free(score);
        out.* = try moe(s, x, banks[i], one_id, score, .mcg, false);
    }
    defer for (outputs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_add(&want, outputs[0], outputs[1], s));
    var actual = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(actual);
    var expected = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(expected);
    try mlx.check(mlx.mlx_astype(&actual, got, .float32, s));
    try mlx.check(mlx.mlx_astype(&expected, want, .float32, s));
    try mlx.check(mlx.mlx_array_eval(actual));
    try mlx.check(mlx.mlx_array_eval(expected));
    const a = mlx.mlx_array_data_float32(actual).?;
    const e = mlx.mlx_array_data_float32(expected).?;
    for (0..128) |i| try t.expectApproxEqAbs(e[i], a[i], 0.0001 + @abs(e[i]) * 0.01);
}

pub fn moeGroups(s: mlx.mlx_stream, x: mlx.mlx_array, banks: []const Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool) !mlx.mlx_array {
    if (banks.len == 0) return error.Exl3GroupMissing;
    if (banks.len == 1) return moe(s, x, banks[0], inds, scores, dec, verify_rows);
    var total = mlx.mlx_array{ .ctx = null };
    defer if (total.ctx != null) {
        _ = mlx.mlx_array_free(total);
    };
    var offset: c_int = 0;
    const zero = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero);
    for (banks) |bank| {
        const count = mlx.getShape(bank.gate.trellis)[0];
        const lo = mlx.mlx_array_new_int(offset);
        defer _ = mlx.mlx_array_free(lo);
        const hi = mlx.mlx_array_new_int(offset + count);
        defer _ = mlx.mlx_array_free(hi);
        var lower = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lower);
        try mlx.check(mlx.mlx_greater_equal(&lower, inds, lo, s));
        var upper = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(upper);
        try mlx.check(mlx.mlx_less(&upper, inds, hi, s));
        var mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask);
        try mlx.check(mlx.mlx_logical_and(&mask, lower, upper, s));
        var shifted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(shifted);
        try mlx.check(mlx.mlx_subtract(&shifted, inds, lo, s));
        var local = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(local);
        try mlx.check(mlx.mlx_where(&local, mask, shifted, zero, s));
        var weight = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(weight);
        try mlx.check(mlx.mlx_where(&weight, mask, scores, zero, s));
        const partial = try moeOutput(s, x, bank, local, weight, dec, verify_rows, .float32, null);
        defer _ = mlx.mlx_array_free(partial);
        var wide = mlx.mlx_array_new();
        defer if (wide.ctx != null) {
            _ = mlx.mlx_array_free(wide);
        };
        try mlx.check(mlx.mlx_astype(&wide, partial, .float32, s));
        if (total.ctx == null) {
            total = wide;
            wide = .{ .ctx = null };
        } else {
            var sum = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(sum);
            try mlx.check(mlx.mlx_add(&sum, total, wide, s));
            _ = mlx.mlx_array_free(total);
            total = sum;
        }
        offset += count;
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, total, mlx.mlx_array_dtype(x), s));
    return out;
}

fn moeMixedDecode(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, slots: mlx.mlx_array, scores: mlx.mlx_array, topk: c_int, dtype: mlx.mlx_dtype) !mlx.mlx_array {
    const hidden = mlx.getShape(x)[1];
    const rows = mlx.getShape(x)[0];
    const inter = mlx.getShape(bank.gate.trellis)[2] * 16;
    const nslots = rows * topk;
    const g = bank.gate;
    const u = bank.up;
    const d = bank.down;
    const gate = try kernels.pairGemv(s, x, g.suh, g.suh, g.trellis, g.trellis, slots, hidden, inter, nslots, topk, 0);
    defer _ = mlx.mlx_array_free(gate[0]);
    defer _ = mlx.mlx_array_free(gate[1]);
    const up = try kernels.pairGemv(s, x, u.suh, u.suh, u.trellis, u.trellis, slots, hidden, inter, nslots, topk, 0);
    defer _ = mlx.mlx_array_free(up[0]);
    defer _ = mlx.mlx_array_free(up[1]);
    const inner = try kernels.downGemvFusedMid(s, gate[0], up[0], d.trellis, g.svh, u.svh, d.suh, slots, inter, hidden, nslots);
    defer _ = mlx.mlx_array_free(inner);
    return kernels.downFinishReduce(s, inner, d.svh, slots, scores, hidden, rows, topk, dtype);
}

fn validationZeros(shape: []const c_int, dtype: mlx.mlx_dtype) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_zeros(&out, shape.ptr, shape.len, dtype, mlx.gpuStream()));
    return out;
}

test "GLM clamped EXL3 rejects malformed banks before kernel dispatch" {
    const s = mlx.gpuStream();
    const trellis = try validationZeros(&.{ 2, 8, 8, 36 }, .uint16);
    defer _ = mlx.mlx_array_free(trellis);
    const scales = try validationZeros(&.{ 2, 128 }, .float16);
    defer _ = mlx.mlx_array_free(scales);
    const x = try validationZeros(&.{ 1, 1, 128 }, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const ids = try validationZeros(&.{ 1, 1, 1 }, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const scores = try validationZeros(&.{ 1, 1, 1 }, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const projection = Proj{ .trellis = trellis, .suh = scales, .svh = scales };
    const valid = Bank{ .gate = projection, .up = projection, .down = projection };
    const Case = struct { target: enum { x, ids, scores, gate, up, down, suh, svh }, shape: []const c_int, dtype: mlx.mlx_dtype, dtype_error: bool = false };
    for ([_]Case{
        .{ .target = .x, .shape = &.{ 1, 1, 127 }, .dtype = .bfloat16 },
        .{ .target = .x, .shape = &.{ 1, 1, 128 }, .dtype = .int32, .dtype_error = true },
        .{ .target = .ids, .shape = &.{ 1, 1, 1 }, .dtype = .float32, .dtype_error = true },
        .{ .target = .scores, .shape = &.{ 1, 1, 2 }, .dtype = .float32 },
        .{ .target = .scores, .shape = &.{ 1, 1, 1 }, .dtype = .uint32, .dtype_error = true },
        .{ .target = .gate, .shape = &.{ 2, 16, 8, 36 }, .dtype = .uint16 },
        .{ .target = .up, .shape = &.{ 2, 8, 16, 36 }, .dtype = .uint16 },
        .{ .target = .down, .shape = &.{ 2, 8, 16, 36 }, .dtype = .uint16 },
        .{ .target = .up, .shape = &.{ 3, 8, 8, 36 }, .dtype = .uint16 },
        .{ .target = .gate, .shape = &.{ 2, 8, 8, 35 }, .dtype = .uint16 },
        .{ .target = .gate, .shape = &.{ 2, 8, 8, 36 }, .dtype = .int16, .dtype_error = true },
        .{ .target = .suh, .shape = &.{ 2, 129 }, .dtype = .float16 },
        .{ .target = .svh, .shape = &.{ 2, 129 }, .dtype = .float16 },
        .{ .target = .suh, .shape = &.{ 2, 128 }, .dtype = .bfloat16, .dtype_error = true },
    }) |c| {
        const bad = try validationZeros(c.shape, c.dtype);
        defer _ = mlx.mlx_array_free(bad);
        var bank = valid;
        var xv = x;
        var iv = ids;
        var sv = scores;
        switch (c.target) {
            .x => xv = bad,
            .ids => iv = bad,
            .scores => sv = bad,
            .gate => bank.gate.trellis = bad,
            .up => bank.up.trellis = bad,
            .down => bank.down.trellis = bad,
            .suh => bank.gate.suh = bad,
            .svh => bank.gate.svh = bad,
        }
        try std.testing.expectError(if (c.dtype_error) error.BadExl3Dtype else error.BadExl3Shape, moeClamped(s, xv, bank, iv, sv, .{ .codebook = .mcg, .window = .w12 }, 10));
    }
}

test { _ = glm_group2; }

test {
    _ = glm_prefill_grid;
}
