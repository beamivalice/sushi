//! A trunk linear as the kind its weight is stored in, decided once when the weight is bound: dense, affine, FP8
//! block or trellis. A projection costs one tag switch, with no dtype query on the weight and no lookup of its
//! quantization parameters. A `Linear` borrows the arrays it holds: whoever loaded them frees them.
const std = @import("std");
const mlx = @import("mlx.zig");
const sushi_exl3 = @import("sushi_exl3");
const fp8_block = @import("fp8_block.zig");
const layer_weights = @import("layer_weights.zig");
const mlx_util = @import("mlx_util.zig");
const model_mod = @import("model.zig");
const verify_moe = @import("verify_moe.zig");

const exl3_trunk = sushi_exl3.trunk;
const ModelConfig = model_mod.ModelConfig;
const QuantMode = model_mod.QuantMode;

pub const Affine = struct {
    w: mlx.mlx_array,
    s: mlx.mlx_array,
    b: mlx.mlx_array,
    /// Solved by the first projection and shared by every copy of the `Linear`. Behind a pointer because a projection
    /// holds `*const Linear`, which the compiler may assume is never written. One cell per bound bank, kept for the
    /// process.
    solved: *Solved,

    /// `bits` is 0 until the first projection reads the activation's inner dimension, which a tensor whose group size
    /// differs from the model's needs to be told apart (`computeQuantParams`); the three are then fixed.
    pub const Solved = struct { bits: u32 = 0, group: u32 = 0, mode: QuantMode = .affine };

    /// The bank's (bits, group size, mode), solved on the first call from the activation `x` it is applied to.
    pub fn params(self: *const Affine, ctx: Ctx, x: mlx.mlx_array) layer_weights.QuantParams {
        const cell = self.solved;
        if (cell.bits == 0) {
            const qp = layer_weights.computeQuantParams(ctx.config, self.w, self.s, mlx_util.lastDim(x));
            cell.* = .{ .bits = qp.bits, .group = qp.group_size, .mode = qp.mode };
        }
        return .{ .bits = cell.bits, .group_size = cell.group, .mode = cell.mode };
    }
};

pub const Trellis = struct {
    w: mlx.mlx_array,
    suh: mlx.mlx_array,
    svh: mlx.mlx_array,
    dec: sushi_exl3.format.Decode,
    geo: *const sushi_exl3.linear.Geo,
};

/// What a projection needs of the engine, per call.
pub const Ctx = struct {
    s: mlx.mlx_stream,
    config: *const ModelConfig,
    /// A spec verify's launches (`trunk.matmul`): trellis rows stay on one arm up to 16.
    wide: bool = false,
};

pub const Linear = union(enum) {
    /// No weight bound: an absent projection, or a weight struct nobody bound.
    absent,
    /// `[in, out]`, transposed at load.
    dense: mlx.mlx_array,
    affine: Affine,
    /// u8 codes and f32 tile scales, the FP8 source linear.
    fp8_block: struct { w: mlx.mlx_array, s: mlx.mlx_array },
    trellis: Trellis,

    /// The kind `w`, `s`, `b` are stored as: u8 codes are FP8 block, u16 words are a trellis bank (`s` and `b` its
    /// input and output scales), a weight with no scales is dense, any other an affine bank.
    pub fn bind(config: *const ModelConfig, w: mlx.mlx_array, s: mlx.mlx_array, b: mlx.mlx_array) !Linear {
        if (w.ctx == null) return .absent;
        switch (mlx.mlx_array_dtype(w)) {
            .uint8 => return .{ .fp8_block = .{ .w = w, .s = s } },
            .uint16 => return .{ .trellis = .{
                .w = w,
                .suh = s,
                .svh = b,
                .dec = config.trunk_quant orelse return error.TrunkQuantMissing,
                .geo = try sushi_exl3.linear.Geo.get(w, null, .{ .exact = true }),
            } },
            else => {},
        }
        if (s.ctx == null) return .{ .dense = w };
        const cell = try std.heap.smp_allocator.create(Affine.Solved);
        cell.* = .{};
        return .{ .affine = .{ .w = w, .s = s, .b = b, .solved = cell } };
    }

    /// The weight handle, for the encoder's tap.
    pub fn weight(self: *const Linear) mlx.mlx_array {
        return switch (self.*) {
            .absent => .{ .ctx = null },
            .dense => |w| w,
            .fp8_block => |f| f.w,
            inline .affine, .trellis => |p| p.w,
        };
    }

    /// `x @ W`, in the numbers the kind's path has always given.
    pub fn apply(self: *const Linear, ctx: Ctx, x: mlx.mlx_array) !mlx.mlx_array {
        switch (self.*) {
            .absent => return error.LinearUnbound,
            .dense => |w| return verify_moe.qmatmulBits(x, w, .{}, .{}, 0, 64, .affine, ctx.s),
            .fp8_block => |f| return fp8_block.linearServing(ctx.s, x, f.w, f.s),
            .trellis => |*t| return exl3_trunk.matmulGeo(ctx.s, x, t.w, t.suh, t.svh, t.geo, t.dec, ctx.wide),
            .affine => |*a| {
                const qp = a.params(ctx, x);
                return verify_moe.qmatmulBits(x, a.w, a.s, a.b, qp.bits, qp.group_size, qp.mode, ctx.s);
            },
        }
    }
};

const testing = std.testing;
const expert_exl3 = sushi_exl3.format;

const test_rows = [_]c_int{ 1, 2, 5, 8 };

fn randBf16(rnd: std.Random, shape: []const c_int, scale: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    return @import("test_support.zig").attn256RandBf16Scaled(rnd, shape, scale, s);
}

fn bitsEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) !bool {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    return same;
}

/// The projection as `Trunk.qmatmul` ran it before kinds were bound: two dtype queries, then the quantization
/// parameters solved from the weight and the activation.
fn tripleOracle(s: mlx.mlx_stream, config: *const ModelConfig, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array) !mlx.mlx_array {
    if (mlx.mlx_array_dtype(w) == .uint8) return fp8_block.linearServing(s, x, w, sc);
    if (mlx.mlx_array_dtype(w) == .uint16) return exl3_trunk.matmul(s, x, w, sc, bi, config.trunk_quant.?, false);
    if (sc.ctx == null) return verify_moe.qmatmulBits(x, w, sc, bi, config.quant_bits, config.quant_group_size, config.quant_mode, s);
    const qp = layer_weights.computeQuantParams(config, w, sc, mlx_util.lastDim(x));
    return verify_moe.qmatmulBits(x, w, sc, bi, qp.bits, qp.group_size, qp.mode, s);
}

fn expectSameBytes(s: mlx.mlx_stream, config: *const ModelConfig, lin: *const Linear, rnd: std.Random, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, in: c_int) !void {
    for (test_rows) |rows| {
        const x = try randBf16(rnd, &.{ 1, rows, in }, 2.0, s);
        defer _ = mlx.mlx_array_free(x);
        const want = try tripleOracle(s, config, x, w, sc, bi);
        defer _ = mlx.mlx_array_free(want);
        // Twice: the first touch solves an affine bank's parameters, the second reads them.
        for (0..2) |_| {
            const got = try lin.apply(.{ .s = s, .config = config }, x);
            defer _ = mlx.mlx_array_free(got);
            try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
            try testing.expectEqual(mlx.mlx_array_dtype(want), mlx.mlx_array_dtype(got));
            try testing.expect(try bitsEqual(s, got, want));
        }
    }
}

test "linear: a bound dense weight gives the bytes of the triple path" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x11a1);
    const rnd = prng.random();
    const config = ModelConfig{ .quant_bits = 4, .quant_group_size = 64 };
    const w = try randBf16(rnd, &.{ 256, 384 }, 1.0, s);
    defer _ = mlx.mlx_array_free(w);
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    const lin = try Linear.bind(&config, w, none, none);
    try testing.expect(lin == .dense);
    try expectSameBytes(s, &config, &lin, rnd, w, none, none, 256);
}

fn affineCase(bits: u32, group: u32) !void {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x11a2 + bits);
    const rnd = prng.random();
    const config = ModelConfig{ .quant_bits = bits, .quant_group_size = group };
    const dense = try randBf16(rnd, &.{ 384, 256 }, 1.0, s);
    defer _ = mlx.mlx_array_free(dense);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_quantize(&parts, dense, .{ .value = @intCast(group), .has_value = true }, .{ .value = @intCast(bits), .has_value = true }, "affine", .{ .ctx = null }, s));
    var q: [3]mlx.mlx_array = undefined;
    for (&q, 0..) |*a, i| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(a, parts, i));
    }
    defer for (q) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const lin = try Linear.bind(&config, q[0], q[1], q[2]);
    try testing.expect(lin == .affine);
    try expectSameBytes(s, &config, &lin, rnd, q[0], q[1], q[2], 256);
    // Resolved by the first projection, and to what the triple path solves.
    const qp = layer_weights.computeQuantParams(&config, q[0], q[1], 256);
    try testing.expectEqual(qp.bits, lin.affine.solved.bits);
    try testing.expectEqual(qp.group_size, lin.affine.solved.group);
}

test "linear: a bound affine 4-bit weight gives the bytes of the triple path" {
    try affineCase(4, 64);
}

test "linear: a bound affine 6-bit weight gives the bytes of the triple path" {
    try affineCase(6, 64);
}

test "linear: a bound affine 8-bit weight gives the bytes of the triple path" {
    try affineCase(8, 64);
}

test "linear: a bound FP8 block weight gives the bytes of the triple path" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x11a6);
    const rnd = prng.random();
    const config = ModelConfig{};
    var codes: [256 * 128]u8 = undefined;
    for (&codes) |*c| c.* = @intCast(0x20 + rnd.uintLessThan(u8, 0x40));
    const w = mlx.mlx_array_new_data(&codes, &[_]c_int{ 256, 128 }, 2, .uint8);
    defer _ = mlx.mlx_array_free(w);
    var scales: [2]f32 = .{ 3e-4, 7e-4 };
    const sc = mlx.mlx_array_new_data(&scales, &[_]c_int{ 2, 1 }, 2, .float32);
    defer _ = mlx.mlx_array_free(sc);
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    const lin = try Linear.bind(&config, w, sc, none);
    try testing.expect(lin == .fp8_block);
    try expectSameBytes(s, &config, &lin, rnd, w, sc, none, 128);
}

fn trellisCase(n: u32) !void {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x11a7 + n);
    const rnd = prng.random();
    var config = ModelConfig{};
    config.trunk_quant = expert_exl3.Decode.tabled(.spec1, .w15);
    const bank = try exl3_trunk.TestBank.init(arena.allocator(), rnd, 256, 384, n);
    const t = try bank.triple(s);
    defer for ([_]mlx.mlx_array{ t.w, t.s, t.b }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const lin = try Linear.bind(&config, t.w, t.s, t.b);
    try testing.expect(lin == .trellis);
    try expectSameBytes(s, &config, &lin, rnd, t.w, t.s, t.b, 256);
    config.trunk_quant = null;
    try testing.expectError(error.TrunkQuantMissing, Linear.bind(&config, t.w, t.s, t.b));
}

test "linear: a bound trellis K4 weight gives the bytes of the triple path" {
    try trellisCase(64);
}

test "linear: a bound trellis K6 weight gives the bytes of the triple path" {
    try trellisCase(96);
}

test "linear: a Linear stays small enough to copy" {
    try testing.expect(@sizeOf(Linear) <= 128);
}

test "linear: an absent weight binds to nothing and refuses a projection" {
    try mlx.requireGpu();
    const config = ModelConfig{};
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    const lin = try Linear.bind(&config, none, none, none);
    try testing.expect(lin == .absent);
    try testing.expectError(error.LinearUnbound, lin.apply(.{ .s = mlx.gpuStream(), .config = &config }, none));
}

/// The shape, size and dtype queries `lin.apply` makes on `x`, run once to warm what a first call builds and once to count.
fn applyQueries(config: *const ModelConfig, lin: *const Linear, x: mlx.mlx_array, s: mlx.mlx_stream) !u64 {
    for (0..2) |i| {
        const before = mlx.query_count;
        const y = try lin.apply(.{ .s = s, .config = config }, x);
        _ = mlx.mlx_array_free(y);
        if (i == 1) return mlx.query_count - before;
    }
    unreachable;
}

test "linear: a decode projection of each kind costs its activation's queries and no more" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x11c1);
    const rnd = prng.random();
    var config = ModelConfig{ .quant_bits = 4, .quant_group_size = 64 };
    config.trunk_quant = expert_exl3.Decode.tabled(.spec1, .w15);
    const x = try randBf16(rnd, &.{ 1, 1, 256 }, 2.0, s);
    defer _ = mlx.mlx_array_free(x);
    var got: [5]u64 = undefined;

    const bank = try exl3_trunk.TestBank.init(arena.allocator(), rnd, 256, 384, 64);
    const t = try bank.triple(s);
    defer for ([_]mlx.mlx_array{ t.w, t.s, t.b }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const trellis = try Linear.bind(&config, t.w, t.s, t.b);
    got[0] = try applyQueries(&config, &trellis, x, s);

    var held: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (held.items) |a| _ = mlx.mlx_array_free(a);
        held.deinit(testing.allocator);
    }
    var parts: [2]exl3_trunk.Triple = undefined;
    for (&parts, [_]usize{ 256, 128 }) |*p, w| p.* = try (try exl3_trunk.TestBank.init(arena.allocator(), rnd, 256, w, 64)).triple(s);
    const orig = parts;
    defer for (orig) |o| for ([_]mlx.mlx_array{ o.w, o.s, o.b }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const stack = (try exl3_trunk.fuse(&.{ &parts[0], &parts[1] }, &held, testing.allocator, s)).?;
    {
        var outs: [2]mlx.mlx_array = undefined;
        for (0..2) |i| {
            const before = mlx.query_count;
            try testing.expect(try exl3_trunk.sharedProject(s, x, stack, config.trunk_quant.?, &outs, false));
            for (outs) |a| _ = mlx.mlx_array_free(a);
            if (i == 1) got[1] = mlx.query_count - before;
        }
    }

    const w = try randBf16(rnd, &.{ 256, 384 }, 1.0, s);
    defer _ = mlx.mlx_array_free(w);
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    got[2] = try applyQueries(&config, &(try Linear.bind(&config, w, none, none)), x, s);

    const dense = try randBf16(rnd, &.{ 384, 256 }, 1.0, s);
    defer _ = mlx.mlx_array_free(dense);
    var quantized = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(quantized);
    try mlx.check(mlx.mlx_quantize(&quantized, dense, .{ .value = 64, .has_value = true }, .{ .value = 4, .has_value = true }, "affine", .{ .ctx = null }, s));
    var q: [3]mlx.mlx_array = undefined;
    for (&q, 0..) |*a, i| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(a, quantized, i));
    }
    defer for (q) |a| {
        _ = mlx.mlx_array_free(a);
    };
    got[3] = try applyQueries(&config, &(try Linear.bind(&config, q[0], q[1], q[2])), x, s);

    var codes: [128 * 128]u8 = @splat(0x38);
    const f8 = mlx.mlx_array_new_data(&codes, &[_]c_int{ 128, 128 }, 2, .uint8);
    defer _ = mlx.mlx_array_free(f8);
    const f8_scale = [_]f32{1e-3};
    const f8_sc = mlx.mlx_array_new_data(&f8_scale, &[_]c_int{ 1, 1 }, 2, .float32);
    defer _ = mlx.mlx_array_free(f8_sc);
    const x128 = try randBf16(rnd, &.{ 1, 1, 128 }, 2.0, s);
    defer _ = mlx.mlx_array_free(x128);
    got[4] = try applyQueries(&config, &(try Linear.bind(&config, f8, f8_sc, none)), x128, s);
    // The entry points that read the operands' shapes, for the fall-back calls and as the comparison.
    var bare = stack;
    bare.geo = null;
    var old: [2]u64 = undefined;
    {
        var outs: [2]mlx.mlx_array = undefined;
        for (0..2) |i| {
            const before = mlx.query_count;
            try testing.expect(try exl3_trunk.sharedProject(s, x, bare, config.trunk_quant.?, &outs, false));
            for (outs) |a| _ = mlx.mlx_array_free(a);
            if (i == 1) old[0] = mlx.query_count - before;
        }
        for (0..2) |i| {
            const before = mlx.query_count;
            const y = try exl3_trunk.matmul(s, x, t.w, t.s, t.b, config.trunk_quant.?, false);
            _ = mlx.mlx_array_free(y);
            if (i == 1) old[1] = mlx.query_count - before;
        }
    }
    // A trellis bank costs its activation's shape (two calls) and dtype; a dense, affine or FP8 weight the queries of its
    // own matmul helpers. The shape-reading entry points cost 19 for one bank and 16 for a stacked pair; the old
    // `qmatmul` added two dtype queries to a bank's 19.
    try testing.expectEqual([2]u64{ 16, 19 }, old);
    try testing.expectEqual([5]u64{ 3, 3, 11, 8, 11 }, got);
    try testing.expect(old[0] > got[1] and old[1] > got[0]);
}
