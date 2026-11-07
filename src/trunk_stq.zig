//! The weight-map binding of the STQ trunk banks: reads `<base>.{trellis,suh,svh}` from the loaded weights as the
//! `(w, s, b)` triple of an affine slot. The math is `exl3/trunk.zig`, re-exported here so callers keep one name.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const exl3 = @import("sushi_exl3");
const trunk = exl3.trunk;

pub const Triple = trunk.Triple;
pub const Shared = trunk.Shared;
pub const TestBank = trunk.TestBank;
pub const SHARED_ROWS_MAX = trunk.SHARED_ROWS_MAX;
pub const isStq = trunk.isStq;
pub const outDim = trunk.outDim;
pub const matmul = trunk.matmul;
pub const fuse = trunk.fuse;
pub const sharedProject = trunk.sharedProject;

/// Drops `<base>.{trellis,suh,svh}` from `weights`, for a bank stacked into another array.
pub fn drop(weights: *model_mod.Weights, base: []const u8) void {
    inline for (.{ "trellis", "suh", "svh" }) |suffix| {
        var buf: [160]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "{s}." ++ suffix, .{base}) catch return;
        if (weights.take(name)) |a| _ = mlx.mlx_array_free(a);
    }
}

/// `<base>.{trellis,suh,svh}` as a `Triple` whose arrays `owned` takes, or null when the pack stores `<base>` another way.
/// A bank is stored one-bank, `[in/16, out/16, n]` with `[in]` and `[out]` scales, both sides on the 128 grid.
pub fn load(
    weights: *const model_mod.Weights,
    base: []const u8,
    dec: ?exl3.format.Decode,
    owned: *std.ArrayList(mlx.mlx_array),
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
) !?Triple {
    var names: [3][160]u8 = undefined;
    const tname = std.fmt.bufPrint(&names[0], "{s}.trellis", .{base}) catch return error.NameTooLong;
    const uname = std.fmt.bufPrint(&names[1], "{s}.suh", .{base}) catch return error.NameTooLong;
    const vname = std.fmt.bufPrint(&names[2], "{s}.svh", .{base}) catch return error.NameTooLong;
    const tr = weights.get(tname) orelse return null;
    if (dec == null) {
        log.err("{s} is a trellis bank and config.json has no trunk_quant\n", .{base});
        return error.TrunkQuantMissing;
    }
    const su = weights.get(uname) orelse return missing(uname);
    const sv = weights.get(vname) orelse return missing(vname);
    const tsh = mlx.getShape(tr);
    const ok = mlx.mlx_array_dtype(tr) == .uint16 and tsh.len == 3 and mlx.mlx_array_dtype(su) == .float16 and
        mlx.mlx_array_dtype(sv) == .float16 and exl3.format.kFromPackedDim(@intCast(tsh[2])) != null;
    const in: c_int = if (tsh.len == 3) tsh[0] * 16 else 0;
    const out: c_int = if (tsh.len == 3) tsh[1] * 16 else 0;
    if (!ok or @rem(in, 128) != 0 or @rem(out, 128) != 0 or mlx.mlx_array_size(su) != @as(usize, @intCast(in)) or
        mlx.mlx_array_size(sv) != @as(usize, @intCast(out)))
    {
        log.err("{s} is not a one-bank trellis (U16 [in/16, out/16, n], F16 scales [in] and [out], both on the 128 grid)\n", .{base});
        return error.TrunkStqLayout;
    }
    const bank_dec = dec.?;
    if (bank_dec.codebook.tabledHashMask() and bank_dec.hashMask(.{ .n = @intCast(tsh[2]) }) == null) {
        log.err("{s} is a rate the trunk_quant codebook table has no row for\n", .{base});
        return error.Exl3CodebookTableRateMissing;
    }
    return .{
        .w = try view(owned, allocator, tr, &.{ 1, tsh[0], tsh[1], tsh[2] }, s),
        .s = try view(owned, allocator, su, &.{ 1, in }, s),
        .b = try view(owned, allocator, sv, &.{ 1, out }, s),
    };
}

fn missing(name: []const u8) error{TrunkStqLayout} {
    log.err("MISSING WEIGHT: {s}\n", .{name});
    return error.TrunkStqLayout;
}

fn view(owned: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, a: mlx.mlx_array, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, a, shape.ptr, shape.len, s));
    errdefer _ = mlx.mlx_array_free(o);
    try owned.append(allocator, o);
    return o;
}

// ---- Tests ----

const testing = std.testing;

fn putBank(weights: *model_mod.Weights, base: []const u8, bank: *const TestBank) !void {
    const t = bank.tensors();
    inline for (.{ "trellis", "suh", "svh" }, 0..) |suffix, i| {
        const key = try std.fmt.allocPrint(weights.allocator, "{s}." ++ suffix, .{base});
        try weights.map.put(key, t[i]);
    }
}

test "trunk_stq: load views a one-bank trellis as a triple and leaves every other tensor to the affine loader" {
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57a2);
    const bank = try TestBank.init(alloc, prng.random(), 256, 384, 48);
    var weights = model_mod.Weights.init(testing.allocator);
    defer weights.deinit();
    try putBank(&weights, "layers.0.self_attn.q_proj", &bank);
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (owned.items) |a| _ = mlx.mlx_array_free(a);
        owned.deinit(testing.allocator);
    }
    const dec = exl3.format.Decode.tabled(.spec1, .w15);
    try testing.expectEqual(@as(?Triple, null), try load(&weights, "layers.0.self_attn.k_proj", dec, &owned, testing.allocator, s));
    const t = (try load(&weights, "layers.0.self_attn.q_proj", dec, &owned, testing.allocator, s)).?;
    try testing.expectEqual(@as(usize, 3), owned.items.len);
    try testing.expect(isStq(t.w));
    try testing.expectEqualSlices(c_int, &.{ 1, 16, 24, 48 }, mlx.getShape(t.w));
    try testing.expectEqualSlices(c_int, &.{ 1, 256 }, mlx.getShape(t.s));
    try testing.expectEqualSlices(c_int, &.{ 1, 384 }, mlx.getShape(t.b));
    try testing.expectEqual(@as(usize, 384), outDim(t.w));
}

test "trunk_stq: load refuses a trellis it cannot serve by name" {
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57a3);
    const rnd = prng.random();
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer owned.deinit(testing.allocator);
    const dec = exl3.format.Decode.tabled(.spec1, .w15);

    var good = model_mod.Weights.init(testing.allocator);
    defer good.deinit();
    try putBank(&good, "w", &try TestBank.init(alloc, rnd, 256, 384, 48));
    try testing.expectError(error.TrunkQuantMissing, load(&good, "w", null, &owned, testing.allocator, s));

    // Off the 128 grid on the output side (a 48-row GDN gate).
    var off = model_mod.Weights.init(testing.allocator);
    defer off.deinit();
    try putBank(&off, "w", &try TestBank.init(alloc, rnd, 256, 48, 48));
    try testing.expectError(error.TrunkStqLayout, load(&off, "w", dec, &owned, testing.allocator, s));

    // A scale tensor missing.
    var torn = model_mod.Weights.init(testing.allocator);
    defer torn.deinit();
    try putBank(&torn, "w", &try TestBank.init(alloc, rnd, 256, 384, 48));
    const gone = torn.map.fetchRemove("w.svh").?;
    _ = mlx.mlx_array_free(gone.value);
    testing.allocator.free(gone.key);
    try testing.expectError(error.TrunkStqLayout, load(&torn, "w", dec, &owned, testing.allocator, s));

    // A rate no trellis tile carries.
    var bad_rate = model_mod.Weights.init(testing.allocator);
    defer bad_rate.deinit();
    try putBank(&bad_rate, "w", &try TestBank.init(alloc, rnd, 256, 384, 7));
    try testing.expectError(error.TrunkStqLayout, load(&bad_rate, "w", dec, &owned, testing.allocator, s));
    try testing.expectEqual(@as(usize, 0), owned.items.len);
}

test "trunk_stq: load refuses a bank at a rate the codebook table has no row for" {
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x57a7);
    var weights = model_mod.Weights.init(testing.allocator);
    defer weights.deinit();
    // STD1 stops at K4: a K6 bank under it has no pair to decode with.
    try putBank(&weights, "w", &try TestBank.init(arena.allocator(), prng.random(), 256, 384, 96));
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer owned.deinit(testing.allocator);
    try testing.expectError(error.Exl3CodebookTableRateMissing, load(&weights, "w", exl3.format.Decode.tabled(.std1, .w15), &owned, testing.allocator, s));
    try testing.expect((try load(&weights, "w", exl3.format.Decode.tabled(.spec1, .w15), &owned, testing.allocator, s)) != null);
    try testing.expectEqual(@as(usize, 3), owned.items.len);
    for (owned.items) |a| _ = mlx.mlx_array_free(a);
}
