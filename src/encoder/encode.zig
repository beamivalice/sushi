const std = @import("std");
const format = @import("sushi_exl3").format;
const st = @import("safetensors.zig");

pub const tile_len = format.TILE_VALUES;

pub const Error = error{ RateUnsupported, WindowTooNarrow, WindowUnsupported, ShapeMismatch, CodebookUnsupported } || std.mem.Allocator.Error;

/// The codebooks the encoder searches.
pub fn searchable(codebook: format.Codebook) bool {
    return codebook == .mcg or codebook == .mul1;
}

pub const Work = struct {
    cost: [2][]f64,
    best: []f64,
    back: []u8,
    stride: usize,

    pub fn init(allocator: std.mem.Allocator, w: u5, steps: usize, kmin: u5) !Work {
        const states = @as(usize, 1) << w;
        const stride = @as(usize, 1) << (w - kmin);
        const cost0 = try allocator.alloc(f64, states);
        errdefer allocator.free(cost0);
        const cost1 = try allocator.alloc(f64, states);
        errdefer allocator.free(cost1);
        const best = try allocator.alloc(f64, stride);
        errdefer allocator.free(best);
        const back = try allocator.alloc(u8, steps * stride);
        return .{ .cost = .{ cost0, cost1 }, .best = best, .back = back, .stride = stride };
    }

    pub fn deinit(self: *Work, allocator: std.mem.Allocator) void {
        allocator.free(self.cost[0]);
        allocator.free(self.cost[1]);
        allocator.free(self.best);
        allocator.free(self.back);
    }
};

/// Least-squares path of `x` through `table[0..2^w]`, one codeword per step: cw_t =
/// ((cw_{t-1} << k_t) | f_t) & (2^w - 1). `pin` fixes the register that enters step 0 (cw_0 >> k_0)
/// and requires the last codeword to carry it in its low w - k_0 bits. The first minimum wins every tie.
pub fn viterbiPass(work: *Work, x: []const f64, ks: []const u8, table: []const f64, w: u5, pin: ?u32, path: []u16) f64 {
    const states = @as(usize, 1) << w;
    const inf = std.math.inf(f64);
    const k0: u5 = @intCast(ks[0]);
    var cur: usize = 0;
    var prev = work.cost[0];
    var next = work.cost[1];
    for (prev[0..states], table[0..states], 0..) |*c, t, cw| {
        const d = x[0] - t;
        c.* = if (pin) |reg| (if (cw >> k0 == reg) d * d else inf) else d * d;
    }
    for (1..x.len) |i| {
        const k: u5 = @intCast(ks[i]);
        const s = @as(usize, 1) << (w - k);
        const row = work.back[i * work.stride ..][0..s];
        const best = work.best[0..s];
        @memcpy(best, prev[0..s]);
        @memset(row, 0);
        for (1..@as(usize, 1) << k) |j| {
            const p = prev[j * s ..][0..s];
            for (best, row, p) |*b, *r, v| {
                if (v < b.*) {
                    b.* = v;
                    r.* = @intCast(j);
                }
            }
        }
        const kk = @as(usize, 1) << k;
        for (best, 0..) |b, low| {
            for (next[low * kk ..][0..kk], table[low * kk ..][0..kk]) |*c, t| {
                const d = x[i] - t;
                c.* = b + d * d;
            }
        }
        std.mem.swap([]f64, &prev, &next);
    }
    var best_cost = inf;
    const reg_mask: usize = (@as(usize, 1) << (w - k0)) - 1;
    for (prev[0..states], 0..) |c, cw| {
        if (pin) |reg| if (cw & reg_mask != reg) continue;
        if (c < best_cost) {
            best_cost = c;
            cur = cw;
        }
    }
    path[x.len - 1] = @intCast(cur);
    var i = x.len - 1;
    while (i > 0) : (i -= 1) {
        const k: u5 = @intCast(ks[i]);
        const low = cur >> k;
        cur = (@as(usize, work.back[i * work.stride + low]) << (w - k)) | low;
        path[i - 1] = @intCast(cur);
    }
    return best_cost;
}

/// Every register in turn: 2^(w - k_0) passes, for narrow windows and tests.
pub fn exactTailBiting(allocator: std.mem.Allocator, work: *Work, x: []const f64, ks: []const u8, table: []const f64, w: u5, path: []u16) !f64 {
    const tmp = try allocator.alloc(u16, x.len);
    defer allocator.free(tmp);
    var best = std.math.inf(f64);
    const k0: u5 = @intCast(ks[0]);
    for (0..@as(usize, 1) << (w - k0)) |reg| {
        const c = viterbiPass(work, x, ks, table, w, @intCast(reg), tmp);
        if (c < best) {
            best = c;
            @memcpy(path, tmp);
        }
    }
    return best;
}

pub const Encoder = struct {
    allocator: std.mem.Allocator,
    rate: format.Rate,
    dec: format.Decode,
    w: u5,
    stride: usize,
    table: []f32,
    costs: [2][]f16,
    back: []u8,
    ks: [tile_len]u8,
    perm: [tile_len]usize,
    x: [tile_len]f32,
    path: [tile_len]u16,

    pub fn init(allocator: std.mem.Allocator, rate: format.Rate, dec: format.Decode) Error!Encoder {
        if (!searchable(dec.codebook)) return error.CodebookUnsupported;
        if (rate.n % 2 != 0 or rate.n < format.Rate.min_n or rate.n > format.Rate.max_n) return error.RateUnsupported;
        if (dec.window.bits() < 12 or dec.window.bits() > 16) return error.WindowUnsupported;
        const w: u5 = @intCast(dec.window.bits());
        var self: Encoder = undefined;
        self.allocator = allocator;
        self.rate = rate;
        self.dec = dec;
        self.w = w;
        var kmin: u5 = 31;
        var kmax: u5 = 0;
        for (&self.ks, 0..) |*k, t| {
            k.* = @intCast(rate.freshBits(t));
            kmin = @min(kmin, @as(u5, @intCast(k.*)));
            kmax = @max(kmax, @as(u5, @intCast(k.*)));
        }
        if (kmax >= w) return error.WindowTooNarrow;
        format.tensorCorePerm(&self.perm);
        self.stride = @as(usize, 1) << (w - kmin);
        self.table = try allocator.alloc(f32, @as(usize, 1) << w);
        errdefer allocator.free(self.table);
        for (self.table, 0..) |*v, cw| {
            v.* = format.f16BitsToF32(format.decodeCodeword(@intCast(cw), dec.codebook));
        }
        self.costs[0] = try allocator.alloc(f16, self.stride);
        errdefer allocator.free(self.costs[0]);
        self.costs[1] = try allocator.alloc(f16, self.stride);
        errdefer allocator.free(self.costs[1]);
        self.back = try allocator.alloc(u8, tile_len * self.stride);
        return self;
    }

    pub fn deinit(self: *Encoder) void {
        self.allocator.free(self.back);
        self.allocator.free(self.costs[1]);
        self.allocator.free(self.costs[0]);
        self.allocator.free(self.table);
    }

    /// Searches one row-major 16x16 tile, writes its `rate.n` packed halfwords exactly as the
    /// loader reads them, and returns the squared error of the path. `recon` receives the values
    /// the engine's decoder will produce, in the same row-major order.
    pub fn encodeTile(self: *Encoder, tile: *const [tile_len]f32, packed_out: []u16, recon: ?*[tile_len]f32) Error!f64 {
        if (packed_out.len != self.rate.n) return error.ShapeMismatch;
        for (&self.x, self.perm) |*v, p| v.* = tile[p];
        const h_inf: f32 = 65504.0;
        var pin: usize = 0;
        var pinned = false;
        for (0..2) |pass| {
            const roll: usize = if (pass == 0) tile_len / 2 else 0;
            var prev = self.costs[0];
            var curr = self.costs[1];
            for (0..tile_len) |step| {
                const ri = (step + roll) & (tile_len - 1);
                const k: u5 = @intCast(self.ks[ri]);
                const k_next: u5 = @intCast(self.ks[(ri + 1) & (tile_len - 1)]);
                const kr: u5 = self.w - k_next;
                const kk = @as(usize, 1) << k_next;
                const edges = @as(usize, 1) << kr;
                const target = self.x[ri];
                const back_row = self.back[ri * self.stride ..][0..edges];
                if (step == 0) {
                    for (0..edges) |e| {
                        var best: f32 = h_inf;
                        var best_fresh: u8 = 0;
                        for (0..kk) |fresh| {
                            const cw = (fresh << kr) | e;
                            const pred = cw >> k;
                            const err: f32 = if (pinned and pred != pin) blk: {
                                break :blk h_inf;
                            } else blk: {
                                const dh = self.table[cw] - target;
                                break :blk @min(dh * dh, h_inf);
                            };
                            if (err < best) {
                                best = err;
                                best_fresh = @intCast(fresh);
                            }
                        }
                        curr[e] = @floatCast(best);
                        back_row[e] = best_fresh;
                    }
                } else {
                    for (0..edges) |e| {
                        var best: f32 = h_inf;
                        var best_fresh: u8 = 0;
                        for (0..kk) |fresh| {
                            const cw = (fresh << kr) | e;
                            const pred = cw >> k;
                            const dh = self.table[cw] - target;
                            const err = @mulAdd(f32, dh, dh, @as(f32, @floatCast(prev[pred])));
                            if (err < best) {
                                best = err;
                                best_fresh = @intCast(fresh);
                            }
                        }
                        curr[e] = @floatCast(best);
                        back_row[e] = best_fresh;
                    }
                }
                std.mem.swap([]f16, &prev, &curr);
            }
            var edge: usize = pin;
            if (pass == 0) {
                const k0: u5 = @intCast(self.ks[0]);
                const final_edges = @as(usize, 1) << (self.w - k0);
                var best: f32 = h_inf;
                edge = 0;
                for (prev[0..final_edges], 0..) |c, e| {
                    const v: f32 = @floatCast(c);
                    if (v < best) {
                        best = v;
                        edge = e;
                    }
                }
            }
            var step_i: usize = tile_len;
            while (step_i > 0) {
                step_i -= 1;
                const ri = (step_i + roll) & (tile_len - 1);
                const k: u5 = @intCast(self.ks[ri]);
                const k_next: u5 = @intCast(self.ks[(ri + 1) & (tile_len - 1)]);
                const kr: u5 = self.w - k_next;
                const bfresh: usize = self.back[ri * self.stride + edge];
                const state = (bfresh << kr) | edge;
                edge = state >> k;
                if (pass == 1) {
                    self.path[ri] = @intCast(state);
                } else if (ri == 0) {
                    break;
                }
            }
            pin = edge;
            pinned = true;
        }
        self.pack(packed_out);
        var sse: f64 = 0;
        for (self.path, self.x) |cw, xv| {
            const d = @as(f64, xv) - @as(f64, self.table[cw]);
            sse += d * d;
        }
        if (recon) |out| {
            for (self.path, self.perm) |cw, p| out[p] = self.table[cw];
        }
        return sse;
    }

    pub fn packStates(rate: format.Rate, ks: *const [tile_len]u8, states: *const [tile_len]u16, out: []u16) void {
        @memset(out, 0);
        for (states, 0..) |cw, t| {
            const k = ks[t];
            const end = rate.bitEnd(t);
            for (0..k) |b| {
                const bit: u16 = (cw >> @intCast(k - 1 - b)) & 1;
                const p = end - k + b;
                const half_index = (p / 32) * 2 + @as(usize, if (p % 32 < 16) 1 else 0);
                out[half_index] |= bit << @intCast(15 - (p % 16));
            }
        }
    }

    fn pack(self: *const Encoder, out: []u16) void {
        packStates(self.rate, &self.ks, &self.path, out);
    }
};

fn testTile(prng: *std.Random.DefaultPrng, tile: *[tile_len]f32) void {
    const r = prng.random();
    for (tile) |*v| v.* = r.floatNorm(f32);
}

test "encode: every code the search writes decodes to the values it scored" {
    const t = std.testing;
    const Case = struct { dec: format.Decode, n: u32 };
    const cases = [_]Case{
        .{ .dec = .{ .codebook = .mcg, .window = .w12 }, .n = 24 },
        .{ .dec = .{ .codebook = .mcg, .window = .w12 }, .n = 36 },
        .{ .dec = .{ .codebook = .mcg, .window = .w15 }, .n = 40 },
        .{ .dec = .{ .codebook = .mcg, .window = .w16 }, .n = 52 },
        .{ .dec = .{ .codebook = .mul1, .window = .w12 }, .n = 44 },
        .{ .dec = .{ .codebook = .mul1, .window = .w15 }, .n = 48 },
        .{ .dec = .{ .codebook = .mul1, .window = .w16 }, .n = 64 },
    };
    var prng = std.Random.DefaultPrng.init(7);
    for (cases) |c| {
        const rate = format.Rate{ .n = c.n };
        var enc = try Encoder.init(t.allocator, rate, c.dec);
        defer enc.deinit();
        var tile: [tile_len]f32 = undefined;
        var packed_out: [64]u16 = undefined;
        for (0..3) |_| {
            testTile(&prng, &tile);
            var recon: [tile_len]f32 = undefined;
            const cost = try enc.encodeTile(&tile, packed_out[0..c.n], &recon);
            var decoded: [tile_len]u16 = undefined;
            format.decodeTile(packed_out[0..c.n], rate, c.dec, &decoded);
            var sse: f64 = 0;
            for (tile, recon, decoded) |x, r, d| {
                try t.expectEqual(format.f32ToF16Bits(r), d);
                const e = @as(f64, x) - @as(f64, format.f16BitsToF32(d));
                sse += e * e;
            }
            try t.expectApproxEqRel(sse, cost, 1e-9);
        }
    }
}

test "encode: the pinned pass equals brute force over every tail-biting path" {
    const t = std.testing;
    const patterns = [_][]const u8{ &.{ 2, 3, 2, 2, 3, 2 }, &.{ 1, 2, 1, 2, 1, 2 }, &.{ 3, 3, 3, 3, 3 } };
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    const w: u5 = 5;
    var table: [1 << w]f64 = undefined;
    for (patterns) |ks| {
        for (0..3) |_| {
            for (&table) |*v| v.* = r.float(f64) * 4 - 2;
            var x: [8]f64 = undefined;
            for (x[0..ks.len]) |*v| v.* = r.floatNorm(f64);
            const xs = x[0..ks.len];
            var kmin: u5 = 31;
            var total: u6 = 0;
            for (ks) |k| {
                kmin = @min(kmin, @as(u5, @intCast(k)));
                total += @intCast(k);
            }
            const k0: u5 = @intCast(ks[0]);
            const regs = @as(usize, 1) << (w - k0);
            var brute: [32]f64 = @splat(std.math.inf(f64));
            const mask: u32 = (1 << w) - 1;
            const reg_mask: u32 = (@as(u32, 1) << (w - k0)) - 1;
            for (0..(@as(usize, 1) << w)) |pre| {
                for (0..(@as(usize, 1) << total)) |fs| {
                    var cw: u32 = @intCast(pre);
                    var rest = fs;
                    var cost: f64 = 0;
                    for (ks, xs) |k, xv| {
                        const f: u32 = @intCast(rest & ((@as(usize, 1) << @intCast(k)) - 1));
                        rest >>= @intCast(k);
                        cw = ((cw << @intCast(k)) | f) & mask;
                        const d = xv - table[cw];
                        cost += d * d;
                    }
                    if ((cw & reg_mask) != (@as(u32, @intCast(pre)) & reg_mask)) continue;
                    const reg = pre & reg_mask;
                    brute[reg] = @min(brute[reg], cost);
                }
            }
            var work = try Work.init(t.allocator, w, ks.len, kmin);
            defer work.deinit(t.allocator);
            var path: [8]u16 = undefined;
            var best: f64 = std.math.inf(f64);
            for (0..regs) |reg| {
                const c = viterbiPass(&work, xs, ks, &table, w, @intCast(reg), path[0..ks.len]);
                try t.expectApproxEqAbs(brute[reg], c, 1e-12);
                var again: f64 = 0;
                for (xs, path[0..ks.len]) |xv, cw| {
                    const d = xv - table[cw];
                    again += d * d;
                }
                try t.expectApproxEqAbs(c, again, 1e-12);
                for (path[1..ks.len], ks[1..], 0..) |cur, k, i| {
                    try t.expectEqual(cur >> @intCast(k), path[i] & ((@as(u16, 1) << @intCast(w - k)) - 1));
                }
                best = @min(best, c);
            }
            const exact = try exactTailBiting(t.allocator, &work, xs, ks, &table, w, path[0..ks.len]);
            try t.expectApproxEqAbs(best, exact, 1e-12);
            const free = viterbiPass(&work, xs, ks, &table, w, null, path[0..ks.len]);
            try t.expect(free <= best + 1e-12);
        }
    }
}

test "encode: the half-tile warm-up pin reaches the exact tail-biting optimum on narrow windows" {
    const t = std.testing;
    var prng = std.Random.DefaultPrng.init(23);
    var table: [512]f64 = undefined;
    for (&table, 0..) |*v, cw| v.* = format.f16BitsToF32(format.decodeCodeword(@intCast(cw), .mcg));
    for ([_]u32{ 24, 40, 44, 56 }) |n| {
        const rate = format.Rate{ .n = n };
        var ks: [256]u8 = undefined;
        var kmin: u5 = 31;
        for (&ks, 0..) |*k, i| {
            k.* = @intCast(rate.freshBits(i));
            kmin = @min(kmin, @as(u5, @intCast(k.*)));
        }
        var work = try Work.init(t.allocator, 9, 256, kmin);
        defer work.deinit(t.allocator);
        var perm: [tile_len]usize = undefined;
        format.tensorCorePerm(&perm);
        const k0: u5 = @intCast(ks[0]);
        for (0..4) |_| {
            var tile: [tile_len]f32 = undefined;
            testTile(&prng, &tile);
            var x: [256]f64 = undefined;
            var rolled_x: [256]f64 = undefined;
            var rolled_k: [256]u8 = undefined;
            for (&x, 0..) |*v, i| {
                v.* = tile[perm[i]];
                rolled_x[i] = tile[perm[(i + 128) % 256]];
                rolled_k[i] = ks[(i + 128) % 256];
            }
            var path: [256]u16 = undefined;
            _ = viterbiPass(&work, &rolled_x, &rolled_k, &table, 9, null, &path);
            const reg: u32 = path[127] & ((@as(u32, 1) << (9 - k0)) - 1);
            const cost = viterbiPass(&work, &x, &ks, &table, 9, reg, &path);
            const exact = try exactTailBiting(t.allocator, &work, &x, &ks, &table, 9, &path);
            try t.expectEqual(exact, cost);
        }
    }
}

const tile_fixture_configs = [_]struct { name: []const u8, dec: format.Decode, n: u32 }{
    .{ .name = "mcg_n24_w12", .dec = .{ .codebook = .mcg, .window = .w12 }, .n = 24 },
    .{ .name = "mcg_n32_w12", .dec = .{ .codebook = .mcg, .window = .w12 }, .n = 32 },
    .{ .name = "mcg_n40_w15", .dec = .{ .codebook = .mcg, .window = .w15 }, .n = 40 },
    .{ .name = "mcg_n40_w16", .dec = .{ .codebook = .mcg, .window = .w16 }, .n = 40 },
    .{ .name = "mcg_n48_w12", .dec = .{ .codebook = .mcg, .window = .w12 }, .n = 48 },
    .{ .name = "mcg_n56_w15", .dec = .{ .codebook = .mcg, .window = .w15 }, .n = 56 },
    .{ .name = "mcg_n64_w16", .dec = .{ .codebook = .mcg, .window = .w16 }, .n = 64 },
    .{ .name = "mcg_n36_w12", .dec = .{ .codebook = .mcg, .window = .w12 }, .n = 36 },
    .{ .name = "mul1_n24_w15", .dec = .{ .codebook = .mul1, .window = .w15 }, .n = 24 },
    .{ .name = "mul1_n32_w12", .dec = .{ .codebook = .mul1, .window = .w12 }, .n = 32 },
    .{ .name = "mul1_n40_w16", .dec = .{ .codebook = .mul1, .window = .w16 }, .n = 40 },
    .{ .name = "mul1_n44_w12", .dec = .{ .codebook = .mul1, .window = .w12 }, .n = 44 },
    .{ .name = "mul1_n64_w15", .dec = .{ .codebook = .mul1, .window = .w15 }, .n = 64 },
};

fn fixtureSlice(comptime T: type, view: st.TensorView) []const T {
    return st.slice(T, view);
}

test "encode: packed codes match the reference search code for code" {
    const t = std.testing;
    var file = try st.parse(t.allocator, st.tiles);
    defer file.deinit(t.allocator);
    const map = file.tensors;
    for (tile_fixture_configs) |c| {
        var key: [64]u8 = undefined;
        const xs = fixtureSlice(f32, map.get(try std.fmt.bufPrint(&key, "{s}.x", .{c.name})).?);
        const want = fixtureSlice(u16, map.get(try std.fmt.bufPrint(&key, "{s}.packed", .{c.name})).?);
        const costs = fixtureSlice(f64, map.get(try std.fmt.bufPrint(&key, "{s}.cost", .{c.name})).?);
        const rate = format.Rate{ .n = c.n };
        var enc = try Encoder.init(t.allocator, rate, c.dec);
        defer enc.deinit();
        for (costs, 0..) |want_cost, i| {
            var packed_out: [64]u16 = undefined;
            const cost = try enc.encodeTile(xs[i * tile_len ..][0..tile_len], packed_out[0..c.n], null);
            try t.expectEqualSlices(u16, want[i * c.n ..][0..c.n], packed_out[0..c.n]);
            try t.expectApproxEqRel(want_cost, cost, 1e-12);
        }
    }
}

test "encode: a rate or window the trellis cannot carry is refused" {
    const t = std.testing;
    try t.expectError(error.RateUnsupported, Encoder.init(t.allocator, .{ .n = 41 }, .mcg));
    try t.expectError(error.RateUnsupported, Encoder.init(t.allocator, .{ .n = 8 }, .mcg));
    try t.expectError(error.WindowUnsupported, Encoder.init(t.allocator, .{ .n = 128 }, .{ .codebook = .mcg, .window = .w8 }));
}

test "encode: a short packed slice or a window outside 12..16 is refused" {
    const t = std.testing;
    try t.expectError(error.WindowUnsupported, Encoder.init(t.allocator, .{ .n = 32 }, .{ .codebook = .mcg, .window = .w11 }));
    try t.expectError(error.RateUnsupported, Encoder.init(t.allocator, .{ .n = 15 }, .mcg));
    var enc = try Encoder.init(t.allocator, .{ .n = 24 }, .mcg);
    defer enc.deinit();
    var tile: [tile_len]f32 = undefined;
    @memset(&tile, 0);
    var codes: [16]u16 = undefined;
    try t.expectError(error.ShapeMismatch, enc.encodeTile(&tile, &codes, null));
}
