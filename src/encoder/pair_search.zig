//! PAIR1's trellis search on the host: one step per weight pair, the state the 16-bit window of the pair's odd
//! weight. The reference the Metal pair search is held to, and the arithmetic of sashimi's pair search: half costs
//! carried between steps, `fma(d1, d1, fma(d0, d0, prev))`, the lowest candidate at a strict minimum, the lowest
//! state at pass 0's closing minimum, and the tail-biting pin of the scalar search.
const std = @import("std");
const format = @import("sushi_exl3").format;
const encode = @import("encode.zig");

pub const tile_len = encode.tile_len;
pub const pairs: usize = tile_len / 2;
/// Pass 0 over the whole tile: `pin_pass` pairs on each side of the wrap.
pub const full_pin_pass: u32 = pairs / 2;
const h_inf: f32 = 65504.0;

/// Fresh bits of pair step j at rate n, repeating every 8 pairs: floor((j+1)n/8) - floor(jn/8).
pub fn ksteps(rate: format.Rate) [8]u8 {
    var out: [8]u8 = undefined;
    for (&out, 0..) |*k, j| k.* = @intCast(((j + 1) * rate.n >> 3) - (j * rate.n >> 3));
    return out;
}

pub const Search = struct {
    w0: []f32,
    w1: []f32,
    ks: [8]u8,
    costs: [2][]f16,
    back: []u8,
    stride: usize,

    pub fn init(allocator: std.mem.Allocator, rate: format.Rate) !Search {
        const p = format.pair1Pair(rate) orelse return error.RateUnsupported;
        const ks = ksteps(rate);
        const kmin = std.mem.min(u8, &ks);
        const stride = @as(usize, 1) << @intCast(16 - kmin);
        const w0 = try allocator.alloc(f32, 1 << 16);
        errdefer allocator.free(w0);
        const w1 = try allocator.alloc(f32, 1 << 16);
        errdefer allocator.free(w1);
        for (0..1 << 16) |cw| {
            const v = format.decodePair(@intCast(cw), p);
            w0[cw] = format.f16BitsToF32(v[0]);
            w1[cw] = format.f16BitsToF32(v[1]);
        }
        const c0 = try allocator.alloc(f16, stride);
        errdefer allocator.free(c0);
        const c1 = try allocator.alloc(f16, stride);
        errdefer allocator.free(c1);
        const back = try allocator.alloc(u8, pairs * stride);
        return .{ .w0 = w0, .w1 = w1, .ks = ks, .costs = .{ c0, c1 }, .back = back, .stride = stride };
    }

    pub fn deinit(self: *Search, allocator: std.mem.Allocator) void {
        allocator.free(self.w0);
        allocator.free(self.w1);
        allocator.free(self.costs[0]);
        allocator.free(self.costs[1]);
        allocator.free(self.back);
    }

    /// The pair codewords of `x` (trellis order, weights 2j and 2j+1 form pair j).
    pub fn run(self: *Search, x: *const [tile_len]f32, pin_pass: u32, out: *[pairs]u16) void {
        var pin: usize = 0;
        var pinned = false;
        for (0..2) |pass| {
            const steps: usize = if (pass == 0) 2 * pin_pass else pairs;
            const roll: usize = if (pass == 0) pairs - pin_pass else 0;
            var prev = self.costs[0];
            var curr = self.costs[1];
            for (0..steps) |step| {
                const pi = (step + roll) & (pairs - 1);
                const kc: u5 = @intCast(self.ks[pi & 7]);
                const kn: u5 = @intCast(self.ks[(pi + 1) & 7]);
                const sb: u5 = 16 - kn;
                const t0 = x[2 * pi];
                const t1 = x[2 * pi + 1];
                const row = self.back[pi * self.stride ..];
                for (0..@as(usize, 1) << sb) |e| {
                    var best: f32 = h_inf;
                    var bf: u8 = 0;
                    for (0..@as(usize, 1) << kn) |a| {
                        const cw = (a << sb) | e;
                        const d0 = self.w0[cw] - t0;
                        const d1 = self.w1[cw] - t1;
                        const err: f32 = if (step == 0) blk: {
                            if (pinned and (cw >> kc) != pin) break :blk h_inf;
                            break :blk @min(@mulAdd(f32, d1, d1, d0 * d0), h_inf);
                        } else @mulAdd(f32, d1, d1, @mulAdd(f32, d0, d0, @as(f32, @floatCast(prev[cw >> kc]))));
                        if (err < best) {
                            best = err;
                            bf = @intCast(a);
                        }
                    }
                    curr[e] = @floatCast(best);
                    row[e] = bf;
                }
                std.mem.swap([]f16, &prev, &curr);
            }
            var edge: usize = pin;
            if (pass == 0) {
                var best: f32 = h_inf;
                edge = 0;
                const closing = @as(usize, 1) << @intCast(16 - self.ks[roll & 7]);
                for (prev[0..closing], 0..) |c, e| {
                    const v: f32 = @floatCast(c);
                    if (v < best) {
                        best = v;
                        edge = e;
                    }
                }
            }
            var st: usize = steps;
            while (st > 0) {
                st -= 1;
                const pi = (st + roll) & (pairs - 1);
                const sb: u5 = @intCast(16 - self.ks[(pi + 1) & 7]);
                const cw = (@as(usize, self.back[pi * self.stride + edge]) << sb) | edge;
                edge = cw >> @intCast(self.ks[pi & 7]);
                if (pass == 1) {
                    out[pi] = @intCast(cw);
                } else if (pi == 0) {
                    break;
                }
            }
            pin = edge;
            pinned = true;
        }
    }
};

/// Every weight's 16-bit window from the pair codewords, for the scalar packer: the odd window is the pair's
/// codeword, the even one the previous pair's shifted by weight 2j's fresh bits plus those bits (the high ones of
/// this pair's). Circular, as the tile is tail-biting.
pub fn expandScalar(rate: format.Rate, cws: *const [pairs]u16, out: *[tile_len]u16, ks: *[tile_len]u8) void {
    for (0..tile_len) |i| ks[i] = @intCast(((i + 1) * rate.n >> 4) - (i * rate.n >> 4));
    for (0..pairs) |j| {
        const ke: u5 = @intCast(ks[2 * j]);
        const ko: u5 = @intCast(ks[2 * j + 1]);
        const c: u32 = cws[j];
        const prev: u32 = cws[(j + pairs - 1) % pairs];
        out[2 * j + 1] = cws[j];
        out[2 * j] = @truncate(((prev << ke) | ((c >> ko) & ((@as(u32, 1) << ke) - 1))) & 0xFFFF);
    }
}

fn gaussTile(prng: *std.Random.DefaultPrng, tile: *[tile_len]f32) void {
    const r = prng.random();
    for (tile) |*v| v.* = r.floatNorm(f32);
}

test "pair_search: every pair the search writes decodes, through the packer and the engine's tile decode, to the values it scored" {
    const t = std.testing;
    var prng = std.Random.DefaultPrng.init(7);
    var perm: [tile_len]usize = undefined;
    format.tensorCorePerm(&perm);
    for ([_]u32{ 24, 26, 30, 32, 40, 46, 48, 56, 64 }) |n| {
        const rate = format.Rate{ .n = n };
        var s = try Search.init(t.allocator, rate);
        defer s.deinit(t.allocator);
        for (0..2) |_| {
            var x: [tile_len]f32 = undefined;
            gaussTile(&prng, &x);
            var cws: [pairs]u16 = undefined;
            s.run(&x, full_pin_pass, &cws);
            var states: [tile_len]u16 = undefined;
            var ks: [tile_len]u8 = undefined;
            expandScalar(rate, &cws, &states, &ks);
            const words = try t.allocator.alloc(u16, n);
            defer t.allocator.free(words);
            encode.Encoder.packStates(rate, &ks, &states, words);
            var got: [tile_len]u16 = undefined;
            format.decodeTile(words, rate, .{ .codebook = .pair1, .window = .w16 }, &got);
            for (0..pairs) |j| {
                try t.expectEqual(format.f32ToF16Bits(s.w0[cws[j]]), got[perm[2 * j]]);
                try t.expectEqual(format.f32ToF16Bits(s.w1[cws[j]]), got[perm[2 * j + 1]]);
            }
        }
    }
}

// tests/dump_pair1_search_fixtures.py: sashimi's production pair search at pin 64 and 32 pairs, for every even n from
// 24 to 64 in order, of 4 N(0,1) tiles (the first x8, past the half-cost clamp): tiles f32 [4,256], pair codewords
// u16 [4,128], decoded f32 [4,256].
const fixture_tiles = 4;
const fixtures = [_]struct { pin: u32, raw: []const u8 }{
    .{ .pin = 64, .raw = @embedFile("fixtures/pair1_search_pin64.bin") },
    .{ .pin = 32, .raw = @embedFile("fixtures/pair1_search_pin32.bin") },
};

fn readF32(b: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, b[i * 4 ..][0..4], .little));
}

test "pair_search: the host search writes sashimi's pair codewords and decoded tiles byte for byte" {
    const t = std.testing;
    const per_rate = fixture_tiles * (tile_len * 4 + pairs * 2 + tile_len * 4);
    for (fixtures) |fx| {
        try t.expectEqual(@as(usize, 21 * per_rate), fx.raw.len);
        var n: u32 = 24;
        while (n <= 64) : (n += 2) {
            const base = fx.raw[(n - 24) / 2 * per_rate ..][0..per_rate];
            const tiles_b = base[0 .. fixture_tiles * tile_len * 4];
            const idx_b = base[tiles_b.len..][0 .. fixture_tiles * pairs * 2];
            const q_b = base[tiles_b.len + idx_b.len ..];
            var s = try Search.init(t.allocator, .{ .n = n });
            defer s.deinit(t.allocator);
            for (0..fixture_tiles) |ti| {
                var x: [tile_len]f32 = undefined;
                for (&x, 0..) |*v, i| v.* = readF32(tiles_b, ti * tile_len + i);
                var cws: [pairs]u16 = undefined;
                s.run(&x, fx.pin, &cws);
                for (cws, 0..) |cw, j| {
                    try t.expectEqual(std.mem.readInt(u16, idx_b[(ti * pairs + j) * 2 ..][0..2], .little), cw);
                    try t.expectEqual(@as(u32, @bitCast(readF32(q_b, ti * tile_len + 2 * j))), @as(u32, @bitCast(s.w0[cw])));
                    try t.expectEqual(@as(u32, @bitCast(readF32(q_b, ti * tile_len + 2 * j + 1))), @as(u32, @bitCast(s.w1[cw])));
                }
            }
        }
    }
}

test "pair_search: squared error by pin pass on N(0,1) tiles (SUSHI_PAIR1_PIN_UBENCH)" {
    if (std.c.getenv("SUSHI_PAIR1_PIN_UBENCH") == null) return error.SkipZigTest;
    const t = std.testing;
    const arms = [_]u32{ 64, 32, 16, 8 };
    for ([_]u32{ 24, 32, 40, 48, 64 }) |n| {
        const rate = format.Rate{ .n = n };
        var s = try Search.init(t.allocator, rate);
        defer s.deinit(t.allocator);
        var prng = std.Random.DefaultPrng.init(31 + n);
        var sse: [arms.len]f64 = @splat(0);
        var energy: f64 = 0;
        for (0..256) |_| {
            var x: [tile_len]f32 = undefined;
            gaussTile(&prng, &x);
            for (x) |v| energy += @as(f64, v) * v;
            for (arms, 0..) |a, ai| {
                var cws: [pairs]u16 = undefined;
                s.run(&x, a, &cws);
                for (cws, 0..) |cw, j| {
                    const d0 = @as(f64, s.w0[cw]) - x[2 * j];
                    const d1 = @as(f64, s.w1[cw]) - x[2 * j + 1];
                    sse[ai] += d0 * d0 + d1 * d1;
                }
            }
        }
        for (arms, 0..) |a, ai| std.debug.print("pair1 n{d} pin {d}: relative MSE {d:.6} ({d:.3}% vs full)\n", .{ n, a, sse[ai] / energy, 100 * (sse[ai] / sse[0] - 1) });
    }
}
