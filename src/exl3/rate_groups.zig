//! Rate-grouped EXL3 layers (`.gN` groups, each with its own rates) served from one ragged bank per
//! projection. Decode rows run the served pair GEMV and fused-mid down in one dispatch each, the rate read
//! per threadgroup from a per-expert table; prefill runs each GEMM once per group. Every slot's planes are
//! the served kernels' bytes at its own rate.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const log = @import("mlx_host").log;
const io_util = @import("mlx_host").io_util;
const exl3 = @import("expert_exl3.zig");
const codebook_tables = @import("codebook_tables.zig");
const kernels = @import("expert_exl3_kernels.zig");
const group_layout = @import("layout.zig");
const root = @import("root.zig");
const fuse = @import("fuse.zig");
const riders = @import("qwen_riders.zig");

const max_groups = group_layout.max_groups;

/// u32 words per expert in `Ragged.meta`: the gate, up and down n, the group, the gate, up and down
/// offsets into the ragged trellis in u32 words, and the expert's index inside its group.
pub const META_WORDS: usize = 8;

/// One rate group's slots in a streamed layer: `slots` slots at the group's gate, up and down rates.
pub const Pool = struct { slots: u32, rates: [3]u32 };

/// A grouped layer's banks as one ragged trellis per projection (groups in order, each expert's
/// tiles contiguous) and scales concatenated over all experts, so every kernel indexes the router's
/// expert id. `views` are each group's banks over the same storage.
pub const Ragged = struct {
    trellis: [3]mlx.mlx_array = @splat(.{ .ctx = null }),
    suh: [3]mlx.mlx_array = @splat(.{ .ctx = null }),
    svh: [3]mlx.mlx_array = @splat(.{ .ctx = null }),
    meta: mlx.mlx_array = .{ .ctx = null },
    views: [max_groups]root.Bank = undefined,
    view_count: usize = 0,
    base: [max_groups]u32 = @splat(0),
    experts_of: [max_groups]u32 = @splat(0),
    /// n per projection per group.
    rates: [max_groups][3]u32 = @splat(@splat(0)),
    count: usize = 0,
    experts: u32 = 0,
    hidden: c_int = 0,
    inter: c_int = 0,
    /// Set by `initSplit`: its pools share no one trellis, which the decode kernels index.
    prefill_only: bool = false,
    /// Every projection's tiles are in `layout.tileMajor` order, which only the funnel decode kernels read.
    tile_major: bool = false,

    /// Builds the ragged storage from per-group banks (lazy or evaluated); the banks stay the caller's.
    pub fn init(s: mlx.mlx_stream, banks: []const root.Bank) !Ragged {
        if (banks.len == 0 or banks.len > max_groups) return error.Exl3GroupMissing;
        var self: Ragged = .{};
        errdefer self.deinit();
        const g0 = mlx.getShape(banks[0].gate.trellis);
        if (g0.len != 4) return error.BadExl3Shape;
        self.hidden = g0[1] * 16;
        self.inter = g0[2] * 16;
        self.count = banks.len;
        var meta: std.ArrayList(u32) = .empty;
        defer meta.deinit(std.heap.page_allocator);
        var words: [3]u64 = @splat(0);
        for (banks, 0..) |bank, g| {
            const projs = [3]root.Proj{ bank.gate, bank.up, bank.down };
            const e = mlx.getShape(bank.gate.trellis)[0];
            for (projs, 0..) |p, pi| {
                const sh = mlx.getShape(p.trellis);
                const in_t = if (pi == 2) self.inter else self.hidden;
                const out_t = if (pi == 2) self.hidden else self.inter;
                if (sh.len != 4 or sh[0] != e or sh[1] * 16 != in_t or sh[2] * 16 != out_t) return error.BadExl3Shape;
                const rate = exl3.kFromPackedDim(@intCast(sh[3])) orelse return error.BadExl3Shape;
                self.rates[g][pi] = rate.n;
            }
            self.base[g] = self.experts;
            self.experts_of[g] = @intCast(e);
            for (0..@intCast(e)) |local| {
                try meta.appendSlice(std.heap.page_allocator, &.{ self.rates[g][0], self.rates[g][1], self.rates[g][2], @intCast(g) });
                for (0..3) |pi| {
                    const tiles: u64 = @intCast(@divExact(self.hidden, 16) * @divExact(self.inter, 16));
                    const at = words[pi] + local * tiles * self.rates[g][pi] / 2;
                    if (at > std.math.maxInt(u32)) return error.ResidentBytesOverflow;
                    try meta.append(std.heap.page_allocator, @intCast(at));
                }
                try meta.append(std.heap.page_allocator, @intCast(local));
            }
            const tiles: u64 = @intCast(@divExact(self.hidden, 16) * @divExact(self.inter, 16));
            for (0..3) |pi| words[pi] += @as(u64, @intCast(e)) * tiles * self.rates[g][pi] / 2;
            self.experts += @intCast(e);
        }
        for (0..3) |pi| {
            var flat: [max_groups]mlx.mlx_array = undefined;
            var made: usize = 0;
            defer for (flat[0..made]) |a| {
                _ = mlx.mlx_array_free(a);
            };
            var sv: [max_groups]mlx.mlx_array = undefined;
            var dv: [max_groups]mlx.mlx_array = undefined;
            for (banks, 0..) |bank, g| {
                const p = projOf(bank, pi);
                flat[g] = mlx.mlx_array_new();
                made += 1;
                const len: c_int = @intCast(mlx.mlx_array_size(p.trellis));
                try mlx.check(mlx.mlx_reshape(&flat[g], p.trellis, &[_]c_int{len}, 1, s));
                sv[g] = p.suh;
                dv[g] = p.svh;
            }
            self.trellis[pi] = try concat(s, flat[0..made]);
            self.suh[pi] = try concat(s, sv[0..banks.len]);
            self.svh[pi] = try concat(s, dv[0..banks.len]);
        }
        self.meta = mlx.mlx_array_new_data(meta.items.ptr, &[_]c_int{@intCast(meta.items.len)}, 1, .uint32);
        var at: [3]c_int = @splat(0);
        for (banks, 0..) |bank, g| {
            var view: root.Bank = undefined;
            for (0..3) |pi| {
                const p = projOf(bank, pi);
                const len: c_int = @intCast(mlx.mlx_array_size(p.trellis));
                const dst = projPtr(&view, pi);
                dst.* = .{ .trellis = .{ .ctx = null }, .suh = .{ .ctx = null }, .svh = .{ .ctx = null } };
                dst.trellis = try rowsView(s, self.trellis[pi], at[pi], at[pi] + len, mlx.getShape(p.trellis));
                dst.suh = try rowsView(s, self.suh[pi], @intCast(self.base[g]), @intCast(self.base[g] + self.experts_of[g]), mlx.getShape(p.suh));
                dst.svh = try rowsView(s, self.svh[pi], @intCast(self.base[g]), @intCast(self.base[g] + self.experts_of[g]), mlx.getShape(p.svh));
                at[pi] += len;
            }
            self.views[g] = view;
            self.view_count += 1;
        }
        return self;
    }

    /// A ragged bank over storage the caller owns, for a streamed layer: each pool's slots are the experts of its
    /// group, so a router id remapped to a global slot id indexes the meta table the way an expert id does in
    /// `init`. `trellis[p]` is the flat U16 storage of projection p (pools back to back), `suh[p]` and `svh[p]` hold
    /// one row per slot over all pools. The handles are retained; the storage stays the caller's.
    pub fn initPools(s: mlx.mlx_stream, hidden: u32, inter: u32, pools: []const Pool, trellis: [3]mlx.mlx_array, suh: [3]mlx.mlx_array, svh: [3]mlx.mlx_array) !Ragged {
        if (pools.len == 0 or pools.len > max_groups) return error.Exl3GroupMissing;
        var self: Ragged = .{};
        errdefer self.deinit();
        self.hidden = @intCast(hidden);
        self.inter = @intCast(inter);
        self.count = pools.len;
        for (0..3) |pi| {
            self.trellis[pi] = mlx.mlx_array_new();
            self.suh[pi] = mlx.mlx_array_new();
            self.svh[pi] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&self.trellis[pi], trellis[pi]));
            try mlx.check(mlx.mlx_array_set(&self.suh[pi], suh[pi]));
            try mlx.check(mlx.mlx_array_set(&self.svh[pi], svh[pi]));
        }
        const tiles: u64 = @as(u64, hidden / 16) * (inter / 16);
        var meta: std.ArrayList(u32) = .empty;
        defer meta.deinit(std.heap.page_allocator);
        var words: [3]u64 = @splat(0);
        for (pools, 0..) |pool, g| {
            self.rates[g] = pool.rates;
            self.base[g] = self.experts;
            self.experts_of[g] = pool.slots;
            for (0..pool.slots) |local| {
                try meta.appendSlice(std.heap.page_allocator, &.{ pool.rates[0], pool.rates[1], pool.rates[2], @intCast(g) });
                for (0..3) |pi| {
                    const at = words[pi] + local * tiles * pool.rates[pi] / 2;
                    if (at > std.math.maxInt(u32)) return error.ResidentBytesOverflow;
                    try meta.append(std.heap.page_allocator, @intCast(at));
                }
                try meta.append(std.heap.page_allocator, @intCast(local));
            }
            for (0..3) |pi| words[pi] += @as(u64, pool.slots) * tiles * pool.rates[pi] / 2;
            self.experts += pool.slots;
        }
        self.meta = mlx.mlx_array_new_data(meta.items.ptr, &[_]c_int{@intCast(meta.items.len)}, 1, .uint32);
        try self.poolViews(s);
        return self;
    }

    /// Each pool's bank over the flat storage, as a view: the prefill GEMMs read these.
    fn poolViews(self: *Ragged, s: mlx.mlx_stream) !void {
        var at: [3]c_int = @splat(0);
        for (0..self.count) |g| {
            var view: root.Bank = undefined;
            const slots: c_int = @intCast(self.experts_of[g]);
            for (0..3) |pi| {
                const in_dim: c_int = if (pi == 2) self.inter else self.hidden;
                const out_dim: c_int = if (pi == 2) self.hidden else self.inter;
                const n: c_int = @intCast(self.rates[g][pi]);
                const len: c_int = slots * @divExact(in_dim, 16) * @divExact(out_dim, 16) * n;
                const dst = projPtr(&view, pi);
                dst.* = .{ .trellis = .{ .ctx = null }, .suh = .{ .ctx = null }, .svh = .{ .ctx = null } };
                dst.trellis = try rowsView(s, self.trellis[pi], at[pi], at[pi] + len, &[_]c_int{ slots, @divExact(in_dim, 16), @divExact(out_dim, 16), n });
                dst.suh = try rowsView(s, self.suh[pi], @intCast(self.base[g]), @intCast(self.base[g] + self.experts_of[g]), &[_]c_int{ slots, in_dim });
                dst.svh = try rowsView(s, self.svh[pi], @intCast(self.base[g]), @intCast(self.base[g] + self.experts_of[g]), &[_]c_int{ slots, out_dim });
                at[pi] += len;
            }
            self.views[g] = view;
            self.view_count += 1;
        }
    }

    /// A prefill bank over two streamed banks of one layer: each pool g of `cache` whole, then, where `overflow[g] > 0`,
    /// the first `overflow[g]` slots of `spill`'s pool g as a pool of its own. A route's slot ids follow the pools in
    /// that order. The scales are concatenated lazily, so they read the storage when the forward runs.
    pub fn initSplit(s: mlx.mlx_stream, cache: *const Ragged, spill: *const Ragged, overflow: []const u32) !Ragged {
        if (overflow.len != cache.count or spill.count != cache.count) return error.Exl3GroupMissing;
        if (cache.tile_major != spill.tile_major) return error.Exl3GroupMissing;
        var self: Ragged = .{ .hidden = cache.hidden, .inter = cache.inter, .prefill_only = true, .tile_major = cache.tile_major };
        errdefer self.deinit();
        var meta: std.ArrayList(u32) = .empty;
        defer meta.deinit(std.heap.page_allocator);
        var suh: [3][max_groups]mlx.mlx_array = undefined;
        var svh: [3][max_groups]mlx.mlx_array = undefined;
        var made: usize = 0;
        defer for (0..made) |p| for (0..3) |pi| {
            _ = mlx.mlx_array_free(suh[pi][p]);
            _ = mlx.mlx_array_free(svh[pi][p]);
        };
        for (0..cache.count) |g| {
            for ([_]struct { bank: *const Ragged, slots: u32 }{ .{ .bank = cache, .slots = cache.experts_of[g] }, .{ .bank = spill, .slots = overflow[g] } }) |part| {
                if (part.slots == 0) continue;
                if (self.count == max_groups) return error.Exl3GroupMissing;
                const p = self.count;
                var view: root.Bank = undefined;
                for (0..3) |pi| {
                    const src = projOf(part.bank.views[g], pi);
                    const sh = mlx.getShape(src.trellis);
                    const dst = projPtr(&view, pi);
                    dst.* = .{ .trellis = .{ .ctx = null }, .suh = .{ .ctx = null }, .svh = .{ .ctx = null } };
                    dst.trellis = try rowsView(s, src.trellis, 0, @intCast(part.slots), &[_]c_int{ @intCast(part.slots), sh[1], sh[2], sh[3] });
                    suh[pi][p] = try rowsView(s, src.suh, 0, @intCast(part.slots), &[_]c_int{ @intCast(part.slots), mlx.getShape(src.suh)[1] });
                    svh[pi][p] = try rowsView(s, src.svh, 0, @intCast(part.slots), &[_]c_int{ @intCast(part.slots), mlx.getShape(src.svh)[1] });
                }
                made += 1;
                self.views[p] = view;
                self.view_count += 1;
                self.rates[p] = cache.rates[g];
                self.base[p] = self.experts;
                self.experts_of[p] = part.slots;
                for (0..part.slots) |local| try meta.appendSlice(std.heap.page_allocator, &.{ self.rates[p][0], self.rates[p][1], self.rates[p][2], @intCast(p), 0, 0, 0, @intCast(local) });
                self.experts += part.slots;
                self.count += 1;
            }
        }
        for (0..3) |pi| {
            self.suh[pi] = try concat(s, suh[pi][0..made]);
            self.svh[pi] = try concat(s, svh[pi][0..made]);
        }
        self.meta = mlx.mlx_array_new_data(meta.items.ptr, &[_]c_int{@intCast(meta.items.len)}, 1, .uint32);
        for (self.views[0..self.view_count], 0..) |*view, p| {
            for (0..3) |pi| {
                const dst = projPtr(view, pi);
                const lo: c_int = @intCast(self.base[p]);
                const hi: c_int = @intCast(self.base[p] + self.experts_of[p]);
                dst.suh = try rowsView(s, self.suh[pi], lo, hi, mlx.getShape(suh[pi][p]));
                dst.svh = try rowsView(s, self.svh[pi], lo, hi, mlx.getShape(svh[pi][p]));
            }
        }
        return self;
    }

    pub fn deinit(self: *Ragged) void {
        for ([_][]mlx.mlx_array{ &self.trellis, &self.suh, &self.svh }) |set| {
            for (set) |*a| freeHandle(a);
        }
        freeHandle(&self.meta);
        for (self.views[0..self.view_count]) |*b| {
            for ([_]*root.Proj{ &b.gate, &b.up, &b.down }) |p| {
                freeHandle(&p.trellis);
                freeHandle(&p.suh);
                freeHandle(&p.svh);
            }
        }
        self.view_count = 0;
    }

    /// Moves every handle this bank owns onto `list`, whose owner frees them; the bank keeps borrowing.
    pub fn handOff(self: *const Ragged, alloc: std.mem.Allocator, list: *std.ArrayList(mlx.mlx_array)) !void {
        try list.ensureUnusedCapacity(alloc, 10 + 9 * self.view_count);
        for (self.storage()) |a| list.appendAssumeCapacity(a);
        for (self.views[0..self.view_count]) |b| {
            for ([_]root.Proj{ b.gate, b.up, b.down }) |p| list.appendSliceAssumeCapacity(&.{ p.trellis, p.suh, p.svh });
        }
    }

    /// The storage arrays an eval must materialize (the views share them).
    pub fn storage(self: *const Ragged) [10]mlx.mlx_array {
        return .{ self.trellis[0], self.trellis[1], self.trellis[2], self.suh[0], self.suh[1], self.suh[2], self.svh[0], self.svh[1], self.svh[2], self.meta };
    }

    /// Expert e's group (a host lookup: tests and the bench).
    pub fn groupOf(self: *const Ragged, e: u32) usize {
        var g: usize = 0;
        while (g + 1 < self.count and e >= self.base[g + 1]) g += 1;
        return g;
    }
};

/// `src` with every projection's trellis in tile-major order (host permute); the scales and meta are shared with `src`.
pub fn tileMajorCopy(s: mlx.mlx_stream, src: *const Ragged) !Ragged {
    if (src.tile_major or src.prefill_only) return error.Exl3GroupMissing;
    var out = src.*;
    out.view_count = 0;
    out.tile_major = true;
    out.trellis = @splat(.{ .ctx = null });
    out.suh = @splat(.{ .ctx = null });
    out.svh = @splat(.{ .ctx = null });
    out.meta = .{ .ctx = null };
    errdefer out.deinit();
    for (0..3) |pi| {
        out.suh[pi] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&out.suh[pi], src.suh[pi]));
        out.svh[pi] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&out.svh[pi], src.svh[pi]));
        const it: usize = @intCast(@divExact(if (pi == 2) src.inter else src.hidden, 16));
        const ot: usize = @intCast(@divExact(if (pi == 2) src.hidden else src.inter, 16));
        try mlx.check(mlx.mlx_array_eval(src.trellis[pi]));
        const from = (mlx.mlx_array_data_uint16(src.trellis[pi]) orelse return error.Unreadable)[0..mlx.mlx_array_size(src.trellis[pi])];
        const to = try std.heap.page_allocator.alloc(u16, from.len);
        defer std.heap.page_allocator.free(to);
        var at: usize = 0;
        for (0..src.count) |g| {
            const n: usize = src.rates[g][pi];
            for (0..src.experts_of[g]) |_| {
                const len = it * ot * n;
                group_layout.tileMajor(u16, to[at..][0..len], from[at..][0..len], it, ot, n);
                at += len;
            }
        }
        out.trellis[pi] = mlx.mlx_array_new_data(to.ptr, &[_]c_int{@intCast(to.len)}, 1, .uint16);
    }
    out.meta = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&out.meta, src.meta));
    try out.poolViews(s);
    return out;
}

fn bit(n: u32) u128 {
    return @as(u128, 1) << @intCast(n / 2);
}

fn freeHandle(a: *mlx.mlx_array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a.*);
    a.* = .{ .ctx = null };
}

fn projOf(bank: root.Bank, pi: usize) root.Proj {
    return switch (pi) {
        0 => bank.gate,
        1 => bank.up,
        else => bank.down,
    };
}

fn projPtr(bank: *root.Bank, pi: usize) *root.Proj {
    return switch (pi) {
        0 => &bank.gate,
        1 => &bank.up,
        else => &bank.down,
    };
}

fn concat(s: mlx.mlx_stream, parts: []const mlx.mlx_array) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, s));
    return out;
}

/// Rows lo..hi of `a` along axis 0, reshaped to `shape`: a view, never a copy.
fn rowsView(s: mlx.mlx_stream, a: mlx.mlx_array, lo: c_int, hi: c_int, shape: []const c_int) !mlx.mlx_array {
    const sh = mlx.getShape(a);
    if (sh.len > 4) return error.BadExl3Shape;
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = @splat(0);
    const strides: [4]c_int = @splat(1);
    @memcpy(stop[0..sh.len], sh);
    start[0] = lo;
    stop[0] = hi;
    var part = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part);
    try mlx.check(mlx.mlx_slice(&part, a, &start, sh.len, &stop, sh.len, &strides, sh.len, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, part, shape.ptr, shape.len, s));
    return out;
}

// ---- Mixed-rate decode kernels ----

const N_DECL = "constexpr uint N = uint(NHW);\nconstexpr uint PACKED_HW = N;\nconstexpr uint PACKED_W = N / 2u;\n";
const PAIR_LOOP = "for (uint proj = 0u; proj < 2u; proj++) {\n";

/// The four decode kernels: the pair and the fused-mid down that prepare their own inputs, and the pair and down that
/// read the inputs `pairPrepare` and `midMixed` made once per slot.
const Kind = enum {
    pair,
    down,
    pair_prepared,
    down_prepared,

    fn isDown(self: Kind) bool {
        return self == .down or self == .down_prepared;
    }
};

/// One rate class of a layer: gate and up n of the pair kernel, or down n (`up` 0).
pub const RateClass = struct { gate: u32, up: u32 = 0 };

/// The distinct gate/up n pairs, or (`down`) the distinct down n, in ascending order.
fn classesOf(rb: *const Ragged, down: bool, out: *[max_groups]RateClass) []RateClass {
    var k: usize = 0;
    for (rb.rates[0..rb.count]) |r| {
        const c: RateClass = if (down) .{ .gate = r[2] } else .{ .gate = r[0], .up = r[1] };
        for (out[0..k]) |seen| {
            if (std.meta.eql(seen, c)) break;
        } else {
            out[k] = c;
            k += 1;
        }
    }
    std.mem.sort(RateClass, out[0..k], {}, struct {
        fn lt(_: void, x: RateClass, y: RateClass) bool {
            return x.gate < y.gate or (x.gate == y.gate and x.up < y.up);
        }
    }.lt);
    return out[0..k];
}

fn caseHead(out: *std.ArrayList(u8), alloc: std.mem.Allocator, n: u32) !void {
    try out.print(alloc, "constexpr uint N = {d}u;\nconstexpr uint PACKED_HW = N;\nconstexpr uint PACKED_W = N / 2u;\n", .{n});
}

/// A funnel decode source reading `layout.tileMajor` tiles: the first tile of a threadgroup's K range and the strides of
/// an iteration's k-tiles, in place of the pack order's. The non-funnel arms keep the pack order and never run.
fn retile(alloc: std.mem.Allocator, src: []const u8) ![]u8 {
    const stride = try replaceEvery(alloc, src, "SGS * OT * PACKED_W", "SGS * 2u * PACKED_W");
    defer alloc.free(stride);
    const head = "((((size_t)(ot >> 1u) * (size_t)IT + ";
    const tail = ") << 1u) + (ot & 1u)) * PACKED_W";
    return replaceEvery(alloc, stride, "((size_t)(tk0 + sg) * (size_t)OT + ot) * PACKED_W", head ++ "(tk0 + sg)" ++ tail) catch
        try replaceEvery(alloc, stride, "((size_t)sg * (size_t)OT + ot) * PACKED_W", head ++ "sg" ++ tail);
}

/// The served pair GEMV and fused-mid down with N a constant per case of ONE switch over the layer's
/// rate classes, read per threadgroup from `meta` (a threadgroup is one slot, so the switch never
/// diverges), and the trellis at the expert's ragged offset. The pair kernel's case holds both
/// projections unrolled: a switch inside its projection loop kept the loop rolled, +40% at 21 cases.
fn mixedSource(alloc: std.mem.Allocator, kind: Kind, classes: []const RateClass, limit: c_int, tile_major: bool) ![:0]u8 {
    const down = kind.isDown();
    const clamped = if (kind == .down and limit > 0) try clampSwiglu(alloc, kernels.DOWN_FUSED_SOURCE, limit) else null;
    defer if (clamped) |c| alloc.free(c);
    const base: []const u8 = clamped orelse switch (kind) {
        .pair => kernels.PAIR_GEMV_SOURCE,
        .down => kernels.DOWN_FUSED_SOURCE,
        .pair_prepared => kernels.PAIR_GEMV_PREPARED_SOURCE,
        .down_prepared => kernels.DOWN_PREPARED_SOURCE,
    };
    const d = std.mem.indexOf(u8, base, N_DECL) orelse return error.RateGroupSourceAnchor;
    const a = if (down) std.mem.indexOf(u8, base, "float acc[OTPT][8] = {};") orelse return error.RateGroupSourceAnchor else std.mem.indexOf(u8, base, PAIR_LOOP) orelse return error.RateGroupSourceAnchor;
    // The pair body ends with its projection loop; the down body runs to the end.
    const z = if (down) base.len else std.mem.lastIndexOfScalar(u8, base, '}') orelse return error.RateGroupSourceAnchor;
    if (!(d < a and a < z)) return error.RateGroupSourceAnchor;
    const old_t = if (down)
        "(const device uint*)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW)"
    else
        "(const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW)";
    const new_t = if (down) "((const device uint *)trellis + (size_t)meta[eid * 8u + 6u])" else "((const device uint *)trellis + (size_t)meta[eid * 8u + 4u + proj])";
    const located = try replaceOnce(alloc, base[if (down) a else a + PAIR_LOOP.len..z], old_t, new_t);
    defer alloc.free(located);
    const body = if (tile_major) try retile(alloc, located) else located;
    defer if (tile_major) alloc.free(body);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    try out.appendSlice(alloc, base[0..d]);
    try out.appendSlice(alloc, base[d + N_DECL.len .. a]);
    try out.appendSlice(alloc, if (down) "switch (meta[eid * 8u + 2u]) {\n" else "switch (meta[eid * 8u] | (meta[eid * 8u + 1u] << 8u)) {\n");
    for (classes) |c| {
        try out.print(alloc, "case {d}u: {{\n", .{c.gate | c.up << 8});
        if (down) {
            try caseHead(&out, alloc, c.gate);
            try out.appendSlice(alloc, body);
        } else for ([_]u32{ c.gate, c.up }, 0..) |n, proj| {
            try out.print(alloc, "{{\nconstexpr uint proj = {d}u;\n", .{proj});
            try caseHead(&out, alloc, n);
            try out.appendSlice(alloc, body);
            try out.appendSlice(alloc, "\n}\n");
        }
        try out.appendSlice(alloc, "\n} break;\n");
    }
    try out.appendSlice(alloc, "default: break;\n}\n");
    if (!down) try out.appendSlice(alloc, base[z + 1 ..]);
    return out.toOwnedSliceSentinel(alloc, 0);
}

/// The fused mid with the gate capped at `limit` and the up branch clipped to +-limit, after their output scales
/// (DeepSeek's and GLM's clamped SwiGLU).
fn clampSwiglu(alloc: std.mem.Allocator, src: []const u8, limit: c_int) ![]u8 {
    var out = try alloc.dupe(u8, src);
    for ([_]u8{ 'g', 'u' }) |which| for (0..4) |k| {
        var head_buf: [32]u8 = undefined;
        const head = try std.fmt.bufPrint(&head_buf, "const float {c}{d} = ", .{ which, k });
        const at = std.mem.indexOf(u8, out, head) orelse return error.RateGroupSourceAnchor;
        const end = std.mem.indexOfScalarPos(u8, out, at, ';') orelse return error.RateGroupSourceAnchor;
        const rhs = out[at + head.len .. end];
        const wrapped = if (which == 'g')
            try std.fmt.allocPrint(alloc, "{s}min({s}, {d}.0f)", .{ head, rhs, limit })
        else
            try std.fmt.allocPrint(alloc, "{s}clamp({s}, -{d}.0f, {d}.0f)", .{ head, rhs, limit, limit });
        defer alloc.free(wrapped);
        const next = try replaceOnce(alloc, out, out[at..end], wrapped);
        alloc.free(out);
        out = next;
    };
    return out;
}

fn replaceOnce(alloc: std.mem.Allocator, src: []const u8, old: []const u8, new: []const u8) ![]u8 {
    const at = std.mem.indexOf(u8, src, old) orelse return error.RateGroupSourceAnchor;
    if (std.mem.indexOfPos(u8, src, at + old.len, old) != null) return error.RateGroupSourceAnchor;
    return std.mem.concat(alloc, u8, &.{ src[0..at], new, src[at + old.len ..] });
}

/// `base` with every `old` replaced by `new`.
fn replaceEvery(alloc: std.mem.Allocator, base: []const u8, old: []const u8, new: []const u8) ![]u8 {
    if (std.mem.indexOf(u8, base, old) == null) return error.RateGroupSourceAnchor;
    const out = try alloc.alloc(u8, std.mem.replacementSize(u8, base, old, new));
    _ = std.mem.replace(u8, base, old, new, out);
    return out;
}

/// The expert-grouped prepared pair or down kernel with N a constant per case, as `mixedSource` makes the one-slot
/// kernels'. An expert's members share its rate, so the switch sits after the member selection, which reads no N.
fn mixedGroupedSource(alloc: std.mem.Allocator, kind: Kind, classes: []const RateClass, group: c_int, tile_major: bool) ![:0]u8 {
    const down = kind == .down_prepared;
    const base = if (down) try kernels.downPreparedGroupedText(group) else try kernels.pairPreparedGroupedText(group);
    const n_decl = "constexpr uint N = uint(NHW);\nconstexpr uint PACKED_W = N / 2u;\n";
    const d = std.mem.indexOf(u8, base, n_decl) orelse return error.RateGroupSourceAnchor;
    const t = std.mem.indexOf(u8, base, "const uint prow = ") orelse return error.RateGroupSourceAnchor;
    if (d >= t) return error.RateGroupSourceAnchor;
    const old_t = "(const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * N)";
    const new_t = if (down) "((const device uint *)trellis + (size_t)meta[eid * 8u + 6u])" else "((const device uint *)trellis + (size_t)meta[eid * 8u + 4u + proj])";
    const retrellis = try replaceEvery(alloc, base[t..], old_t, new_t);
    defer alloc.free(retrellis);
    const looped = if (down) retrellis else try replaceEvery(alloc, retrellis, PAIR_LOOP, "{\n");
    defer if (!down) alloc.free(looped);
    const tail = if (tile_major) try retile(alloc, looped) else looped;
    defer if (tile_major) alloc.free(tail);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    try out.appendSlice(alloc, base[0..d]);
    try out.appendSlice(alloc, base[d + n_decl.len .. t]);
    try out.appendSlice(alloc, if (down) "switch (meta[eid * 8u + 2u]) {\n" else "switch (meta[eid * 8u] | (meta[eid * 8u + 1u] << 8u)) {\n");
    for (classes) |c| {
        try out.print(alloc, "case {d}u: {{\n", .{c.gate | c.up << 8});
        if (down) {
            try caseHead(&out, alloc, c.gate);
            try out.appendSlice(alloc, tail);
        } else for ([_]u32{ c.gate, c.up }, 0..) |n, proj| {
            try out.print(alloc, "{{\nconstexpr uint proj = {d}u;\n", .{proj});
            try caseHead(&out, alloc, n);
            try out.appendSlice(alloc, tail);
            try out.appendSlice(alloc, "\n}\n");
        }
        try out.appendSlice(alloc, "\n} break;\n");
    }
    try out.appendSlice(alloc, "default: break;\n}\n");
    return out.toOwnedSliceSentinel(alloc, 0);
}

fn mixedText(alloc: std.mem.Allocator, kind: Kind, classes: []const RateClass, limit: c_int, group: c_int, tile_major: bool) ![:0]u8 {
    return if (group > 0) mixedGroupedSource(alloc, kind, classes, group, tile_major) else mixedSource(alloc, kind, classes, limit, tile_major);
}

const KernelEntry = struct {
    kind: Kind,
    group: c_int,
    limit: c_int,
    tile_major: bool,
    classes: [max_groups]RateClass,
    count: usize,
    decode: exl3.Decode,
    kernel: ?mlx.mlx_fast_metal_kernel,
};

var kernel_cache: std.ArrayList(KernelEntry) = .empty;

fn header(dec: exl3.Decode) ![:0]const u8 {
    return switch (dec.codebook) {
        inline else => |cb| switch (dec.window) {
            inline else => |win| kernels.withTableRows(cb, comptime kernels.codebookHelpers(cb, win)),
        },
    };
}

/// A kernel per (codebook, window, table, rate classes): the name spells all four, since MLX keys its
/// libraries by name.
fn mixedKernel(kind: Kind, classes: []const RateClass, limit: c_int, group: c_int, tile_major: bool) !mlx.mlx_fast_metal_kernel {
    const dec = kernels.activeDecode();
    const down = kind.isDown();
    const slot = for (kernel_cache.items) |*k| {
        if (k.kind == kind and k.group == group and k.limit == limit and k.tile_major == tile_major and k.count == classes.len and std.mem.eql(u8, std.mem.sliceAsBytes(k.classes[0..k.count]), std.mem.sliceAsBytes(classes)) and
            k.decode.codebook == dec.codebook and k.decode.window == dec.window and k.decode.table == dec.table) break &k.kernel;
    } else blk: {
        var e: KernelEntry = .{ .kind = kind, .group = group, .limit = limit, .tile_major = tile_major, .classes = undefined, .count = classes.len, .decode = dec, .kernel = null };
        @memcpy(e.classes[0..classes.len], classes);
        try kernel_cache.append(std.heap.page_allocator, e);
        break :blk &kernel_cache.items[kernel_cache.items.len - 1].kernel;
    };
    if (slot.*) |k| return k;
    const alloc = std.heap.page_allocator;
    const src = try mixedText(alloc, kind, classes, limit, group, tile_major);
    defer alloc.free(src);
    var name: std.ArrayList(u8) = .empty;
    defer name.deinit(alloc);
    try name.print(alloc, "sushi_exl3_rg_{s}_{s}_w{d}", .{ @tagName(kind), @tagName(dec.codebook), dec.window.bits() });
    if (group > 0) try name.print(alloc, "_g{d}", .{group});
    if (dec.codebook.tabledHashMask()) try name.print(alloc, "_t{s}", .{codebook_tables.tables[dec.table.?].digest});
    for (classes) |c| try name.print(alloc, "_{d}x{d}", .{ c.gate, c.up });
    if (limit > 0) try name.print(alloc, "_clamp{d}", .{limit});
    if (tile_major) try name.appendSlice(alloc, "_tm");
    try name.append(alloc, 0);
    const ins: []const [*:0]const u8 = switch (kind) {
        .pair => &.{ "x", "suhg", "suhu", "tg", "tu", "slots", "meta" },
        .down => &.{ "ig", "iu", "trellis", "svhg", "svhu", "suhd", "slots", "meta" },
        .pair_prepared => &.{ "prep", "tg", "tu", "slots", "meta" },
        .down_prepared => &.{ "middle", "trellis", "slots", "meta" },
    };
    const outs: []const [*:0]const u8 = if (down) &.{"y"} else &.{ "yg", "yu" };
    return kernels.getNamedKernel(slot, @ptrCast(name.items.ptr), ins, outs, src, try header(dec));
}

const DecodeKey = struct { in_dim: c_int, out_dim: c_int, nslots: c_int, topk: c_int, down: bool, layout: kernels.GemvLayout, group: c_int = 0 };
var decode_cfgs: kernels.CfgCache(DecodeKey, 16) = .{};

/// The pair or down GEMV's launch: tiles of `layout.tiles` outputs across, a slot a row, a K split deep.
fn decodeLaunch(alloc: std.mem.Allocator, key: DecodeKey) !fuse.Launch {
    const tiles_x = @divExact(@divExact(key.out_dim, 16), key.layout.tiles);
    var ints: std.ArrayList(fuse.IntArg) = .empty;
    const outs = try alloc.alloc(fuse.Out, if (key.down) 1 else 2);
    var tgs: [3]c_int = undefined;
    if (key.down) {
        const nsplit: c_int = @intCast(kernels.pairSplitCountFor(key.out_dim));
        outs[0] = .{ .shape = try alloc.dupe(c_int, &.{ key.nslots, key.out_dim }), .dtype = .float16 };
        tgs = .{ tiles_x, key.nslots, 1 };
        try ints.append(alloc, .{ .name = "NSPLIT", .value = nsplit });
    } else {
        const nsplit: c_int = @intCast(kernels.pairSplitCountFor(key.in_dim));
        const sh = try alloc.dupe(c_int, &.{ key.nslots * nsplit, key.out_dim });
        outs[0] = .{ .shape = sh, .dtype = .float32 };
        outs[1] = .{ .shape = sh, .dtype = .float32 };
        tgs = .{ tiles_x, key.nslots, nsplit };
        try ints.append(alloc, .{ .name = "NSPLIT", .value = nsplit });
        try ints.append(alloc, .{ .name = "TOPK", .value = key.topk });
    }
    try ints.append(alloc, .{ .name = "IDIM", .value = key.in_dim });
    try ints.append(alloc, .{ .name = "ODIM", .value = key.out_dim });
    try ints.append(alloc, .{ .name = "FUNNEL", .value = @intFromBool(key.layout.funnel) });
    try ints.append(alloc, .{ .name = "OTPT", .value = key.layout.tiles });
    if (key.group > 0) {
        try ints.append(alloc, .{ .name = "GROUP", .value = key.group });
        try ints.append(alloc, .{ .name = "NSLOTS", .value = key.nslots });
    }
    return .{ .tgs = tgs, .threads = 128, .ints = ints.items, .outs = outs };
}

fn decodeConfig(key: DecodeKey) !mlx.mlx_fast_metal_kernel_config {
    if (decode_cfgs.get(key)) |c| return c;
    const c = try (try decodeLaunch(std.heap.page_allocator, key)).config();
    decode_cfgs.put(key, c);
    return c;
}

fn getOut(v: mlx.mlx_vector_array, i: usize) !mlx.mlx_array {
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, v, i));
    return a;
}

/// Gate and up inner planes (f32, one per K split) for every slot.
fn pairMixed(s: mlx.mlx_stream, x: mlx.mlx_array, rb: *const Ragged, slots: mlx.mlx_array, topk: c_int, classes: []const RateClass) ![2]mlx.mlx_array {
    const nslots: c_int = @intCast(mlx.mlx_array_size(slots));
    const cfg = try decodeConfig(.{ .in_dim = rb.hidden, .out_dim = rb.inter, .nslots = nslots, .topk = topk, .down = false, .layout = kernels.gemvLayout(@divExact(rb.inter, 16)) });
    const ov = try kernels.applyOuts(s, try mixedKernel(.pair, classes, 0, 0, rb.tile_major), &.{ x, rb.suh[0], rb.suh[1], rb.trellis[0], rb.trellis[1], slots, rb.meta }, cfg, 2);
    defer _ = mlx.mlx_vector_array_free(ov);
    const g = try getOut(ov, 0);
    errdefer _ = mlx.mlx_array_free(g);
    return .{ g, try getOut(ov, 1) };
}

/// The fused SwiGLU mid and down GEMV: f16 inner down planes [nslots, hidden].
fn downMixed(s: mlx.mlx_stream, inner: [2]mlx.mlx_array, rb: *const Ragged, slots: mlx.mlx_array, classes: []const RateClass, limit: c_int) !mlx.mlx_array {
    const nslots: c_int = @intCast(mlx.mlx_array_size(slots));
    const cfg = try decodeConfig(.{ .in_dim = rb.inter, .out_dim = rb.hidden, .nslots = nslots, .topk = 0, .down = true, .layout = kernels.gemvLayout(@divExact(rb.hidden, 16)) });
    const ov = try kernels.applyOuts(s, try mixedKernel(.down, classes, limit, 0, rb.tile_major), &.{ inner[0], inner[1], rb.trellis[2], rb.svh[0], rb.svh[1], rb.suh[2], slots, rb.meta }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    return getOut(ov, 0);
}

/// The pair GEMV on the inputs `pairPrepare` made once per slot (a pair threadgroup otherwise re-derives its K span of
/// x, 72 times a slot at DeepSeek's width): the planes `pairMixed` writes, byte for byte.
fn pairPreparedMixed(s: mlx.mlx_stream, prep: mlx.mlx_array, rb: *const Ragged, slots: mlx.mlx_array, topk: c_int, classes: []const RateClass, group: c_int) ![2]mlx.mlx_array {
    const nslots: c_int = @intCast(mlx.mlx_array_size(slots));
    const cfg = try decodeConfig(.{ .in_dim = rb.hidden, .out_dim = rb.inter, .nslots = nslots, .topk = topk, .down = false, .layout = kernels.groupedLayout(kernels.gemvLayout(@divExact(rb.inter, 16)), group), .group = group });
    const ov = try kernels.applyOuts(s, try mixedKernel(.pair_prepared, classes, 0, group, rb.tile_major), &.{ prep, rb.trellis[0], rb.trellis[1], slots, rb.meta }, cfg, 2);
    defer _ = mlx.mlx_vector_array_free(ov);
    const g = try getOut(ov, 0);
    errdefer _ = mlx.mlx_array_free(g);
    return .{ g, try getOut(ov, 1) };
}

const MidKey = struct { dim: c_int, nslots: c_int, nsplit: c_int };
var mid_cfgs: kernels.CfgCache(MidKey, 16) = .{};
var mid_kernels: [4]struct { limit: c_int = -1, kernel: ?mlx.mlx_fast_metal_kernel = null } = @splat(.{});

/// The SwiGLU mid and the down input's Hadamard once per slot, `[nslots, inter]` f16: what the fused-mid down
/// otherwise re-derives in every threadgroup of the slot.
fn midMixed(s: mlx.mlx_stream, inner: [2]mlx.mlx_array, rb: *const Ragged, slots: mlx.mlx_array, limit: c_int) !mlx.mlx_array {
    const nslots: c_int = @intCast(mlx.mlx_array_size(slots));
    const nsplit: c_int = @intCast(kernels.pairSplitCountFor(rb.hidden));
    const key: MidKey = .{ .dim = rb.inter, .nslots = nslots, .nsplit = nsplit };
    const cfg = mid_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ nslots, rb.inter }, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(rb.inter, 128) * 32, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", rb.inter));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NSPLIT", nsplit));
        mid_cfgs.put(key, c);
        break :blk c;
    };
    const slot = for (&mid_kernels) |*k| {
        if (k.limit == limit or k.limit == -1) break k;
    } else return error.Exl3GroupMissing;
    if (slot.kernel == null) {
        const alloc = std.heap.page_allocator;
        const clamped = if (limit > 0) try clampSwiglu(alloc, kernels.DECODE_MID_SOURCE, limit) else null;
        defer if (clamped) |c| alloc.free(c);
        const text: []const u8 = clamped orelse kernels.DECODE_MID_SOURCE;
        const src = try alloc.allocSentinel(u8, text.len, 0);
        defer alloc.free(src);
        @memcpy(src, text);
        var name_buf: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "sushi_exl3_rg_mid_clamp{d}\x00", .{limit});
        slot.limit = limit;
        slot.kernel = try kernels.getNamedKernel(&slot.kernel, @ptrCast(name.ptr), &.{ "ig", "iu", "svhg", "svhu", "suhd", "slots" }, &.{"prepared"}, src, "");
    }
    const ov = try kernels.applyOuts(s, slot.kernel.?, &.{ inner[0], inner[1], rb.svh[0], rb.svh[1], rb.suh[2], slots }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    return getOut(ov, 0);
}

/// The down GEMV on `midMixed`'s inputs: the plane `downMixed` writes, byte for byte.
fn downPreparedMixed(s: mlx.mlx_stream, middle: mlx.mlx_array, rb: *const Ragged, slots: mlx.mlx_array, classes: []const RateClass, group: c_int) !mlx.mlx_array {
    const nslots: c_int = @intCast(mlx.mlx_array_size(slots));
    const cfg = try decodeConfig(.{ .in_dim = rb.inter, .out_dim = rb.hidden, .nslots = nslots, .topk = 0, .down = true, .layout = kernels.groupedLayout(kernels.gemvLayout(@divExact(rb.hidden, 16)), group), .group = group });
    const ov = try kernels.applyOuts(s, try mixedKernel(.down_prepared, classes, 0, group, rb.tile_major), &.{ middle, rb.trellis[2], slots, rb.meta }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    return getOut(ov, 0);
}

// ---- The shared expert inside the routed decode launches ----

/// A one-expert trellis bank run beside the routed experts (DeepSeek's shared expert, SPEC1 at its own n).
pub const SharedExpert = struct { bank: root.Bank, dec: exl3.Decode };

const FuseStage = enum { prep, pair, mid, down };

/// What shapes a fused stage's kernel and launch.
const FuseKey = struct {
    stage: FuseStage,
    nslots: c_int,
    rows: c_int,
    topk: c_int,
    hidden: c_int,
    inter: c_int,
    group: c_int,
    tiles: c_int,
    funnel: bool,
    tile_major: bool,
    limit: c_int,
    classes: [max_groups]RateClass,
    class_count: usize,
    dec: exl3.Decode,
    sdec: exl3.Decode,
    sn: u32,
    one_row: bool,
    b_first: bool,
};

const FusedStage = struct { kernel: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config };
var fused_stages: std.AutoHashMapUnmanaged(FuseKey, FusedStage) = .empty;
var fused_reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;
const ReduceKey = struct { hidden: c_int, rows: c_int, topk: c_int };
var fused_reduce_cfgs: kernels.CfgCache(ReduceKey, 8) = .{};
/// Test and bench arm: which part's threadgroups the fused grid lists first.
pub var fused_shared_first: bool = false;

var zero_slot_rows: [kernels.DECODE_ROWS_MAX + 1]?mlx.mlx_array = @splat(null);

/// The shared expert's slot ids: expert 0, once per row.
fn zeroSlots(rows: usize) mlx.mlx_array {
    if (zero_slot_rows[rows]) |a| return a;
    const zeros: [kernels.DECODE_ROWS_MAX]u32 = @splat(0);
    const a = mlx.mlx_array_new_data(&zeros, &[_]c_int{@intCast(rows)}, 1, .uint32);
    zero_slot_rows[rows] = a;
    return a;
}

fn intArgs(alloc: std.mem.Allocator, pairs: anytype) ![]const fuse.IntArg {
    const out = try alloc.alloc(fuse.IntArg, pairs.len);
    inline for (pairs, 0..) |p, i| out[i] = .{ .name = p[0], .value = p[1] };
    return out;
}

fn halfOut(alloc: std.mem.Allocator, rows: c_int, dim: c_int) !fuse.Out {
    return .{ .shape = try alloc.dupe(c_int, &.{ rows, dim }), .dtype = .float16 };
}

fn outs1(alloc: std.mem.Allocator, a: fuse.Out) ![]const fuse.Out {
    return alloc.dupe(fuse.Out, &.{a});
}

fn outs2(alloc: std.mem.Allocator, a: fuse.Out, b: fuse.Out) ![]const fuse.Out {
    return alloc.dupe(fuse.Out, &.{ a, b });
}

fn routedPart(alloc: std.mem.Allocator, k: FuseKey, header_text: []const u8) !fuse.Part {
    const classes = k.classes[0..k.class_count];
    switch (k.stage) {
        .prep => return .{
            .launch = .{ .tgs = .{ @divExact(k.hidden, 128), k.nslots, 2 }, .threads = 32, .ints = try intArgs(alloc, .{ .{ "IDIM", k.hidden }, .{ "TOPK", k.topk } }), .outs = try outs1(alloc, try halfOut(alloc, k.nslots * 2, k.hidden)) },
            .body = kernels.PAIR_PREP_SOURCE,
            .ins = &.{ "x", "suhg", "suhu", "slots" },
            .outs = &.{"prep"},
        },
        .pair => return .{
            .launch = try decodeLaunch(alloc, .{ .in_dim = k.hidden, .out_dim = k.inter, .nslots = k.nslots, .topk = k.topk, .down = false, .layout = .{ .funnel = k.funnel, .tiles = k.tiles }, .group = k.group }),
            .body = try mixedText(alloc, .pair_prepared, classes, 0, k.group, k.tile_major),
            .ins = &.{ "prep", "tg", "tu", "slots", "meta" },
            .outs = &.{ "yg", "yu" },
            .header = header_text,
        },
        .mid => {
            const nsplit: c_int = @intCast(kernels.pairSplitCountFor(k.hidden));
            const body = if (k.limit > 0) try clampSwiglu(alloc, kernels.DECODE_MID_SOURCE, k.limit) else try alloc.dupe(u8, kernels.DECODE_MID_SOURCE);
            return .{
                .launch = .{ .tgs = .{ @divExact(k.inter, 128), k.nslots, 1 }, .threads = 32, .ints = try intArgs(alloc, .{ .{ "IDIM", k.inter }, .{ "NSPLIT", nsplit } }), .outs = try outs1(alloc, try halfOut(alloc, k.nslots, k.inter)) },
                .body = body,
                .ins = &.{ "ig", "iu", "svhg", "svhu", "suhd", "slots" },
                .outs = &.{"prepared"},
            };
        },
        .down => return .{
            .launch = try decodeLaunch(alloc, .{ .in_dim = k.inter, .out_dim = k.hidden, .nslots = k.nslots, .topk = 0, .down = true, .layout = .{ .funnel = k.funnel, .tiles = k.tiles }, .group = k.group }),
            .body = try mixedText(alloc, .down_prepared, classes, 0, k.group, k.tile_major),
            .ins = &.{ "middle", "trellis", "slots", "meta" },
            .outs = &.{"y"},
            .header = header_text,
        },
    }
}

/// The shared chain's kernel for the stage, as `moeSwigluClampedWithShared` runs it on the lane arm.
fn sharedPart(alloc: std.mem.Allocator, k: FuseKey, header_text: []const u8) !fuse.Part {
    const rows = k.rows;
    const n: c_int = @intCast(k.sn);
    const pair_tiles = @divExact(k.inter, 16);
    const down_tiles = @divExact(k.hidden, 16);
    switch (k.stage) {
        .prep => return .{
            .launch = .{ .tgs = .{ @divExact(k.hidden, 128), rows, 1 }, .threads = 32, .ints = try intArgs(alloc, .{ .{ "IDIM", k.hidden }, .{ "TOPK", 1 } }), .outs = try outs2(alloc, try halfOut(alloc, rows, k.hidden), try halfOut(alloc, rows, k.hidden)) },
            .body = kernels.LANE_PREP_UNSORTED_SOURCE,
            .ins = &.{ "x", "suhg", "suhu", "slots" },
            .outs = &.{ "yg", "yu" },
        },
        .pair => return .{
            .launch = .{ .tgs = .{ if (k.one_row) @divExact(pair_tiles, 2) else pair_tiles, rows, 2 }, .threads = 128, .ints = try intArgs(alloc, .{ .{ "IDIM", k.hidden }, .{ "ODIM", k.inter }, .{ "NHW", n } }), .outs = try outs2(alloc, try halfOut(alloc, rows, k.inter), try halfOut(alloc, rows, k.inter)) },
            .body = if (k.one_row) kernels.ONE_ROW_PAIR_SOURCE else kernels.LANE_PAIR_SOURCE,
            .ins = &.{ "xg", "xu", "tg", "tu", "slots" },
            .outs = &.{ "yg", "yu" },
            .header = header_text,
        },
        .mid => return .{
            .launch = .{ .tgs = .{ @divExact(k.inter, 128), rows, 1 }, .threads = 32, .ints = try intArgs(alloc, .{ .{ "ODIM", k.inter }, .{ "CLAMP_LIMIT", k.limit } }), .outs = try outs1(alloc, try halfOut(alloc, rows, k.inter)) },
            .body = kernels.DOWN_LANE_MID_SOURCE,
            .ins = &.{ "ig", "iu", "svhg", "svhu", "suhd", "slots" },
            .outs = &.{"yd"},
        },
        .down => return .{
            .launch = .{ .tgs = .{ if (k.one_row) @divExact(down_tiles, 2) else down_tiles, rows, 1 }, .threads = 128, .ints = try intArgs(alloc, .{ .{ "IDIM", k.inter }, .{ "ODIM", k.hidden }, .{ "NHW", n } }), .outs = try outs1(alloc, try halfOut(alloc, rows, k.hidden)) },
            .body = if (k.one_row) kernels.ONE_ROW_COOP_SOURCE else kernels.DOWN_LANE_COOP_SOURCE,
            .ins = &.{ "x", "trellis", "slots" },
            .outs = &.{"y"},
            .header = header_text,
        },
    }
}

pub fn headerOf(dec: exl3.Decode) ![:0]const u8 {
    const prev = kernels.activeDecode();
    kernels.setDecodeParams(dec);
    defer kernels.setDecodeParams(prev);
    return header(dec);
}

fn buildStage(k: FuseKey) !FusedStage {
    const alloc = std.heap.page_allocator;
    const coded = k.stage == .pair or k.stage == .down;
    const a = try routedPart(alloc, k, if (coded) try headerOf(k.dec) else "");
    const b = try sharedPart(alloc, k, if (coded) try headerOf(k.sdec) else "");
    const src = try fuse.source(alloc, a, b);
    const head = try fuse.header(alloc, a, b);
    var name: std.ArrayList(u8) = .empty;
    try name.print(alloc, "sushi_exl3_fz_{s}_{s}_w{d}", .{ @tagName(k.stage), @tagName(k.dec.codebook), k.dec.window.bits() });
    if (k.dec.codebook.tabledHashMask()) try name.print(alloc, "_t{s}", .{codebook_tables.tables[k.dec.table.?].digest});
    try name.print(alloc, "_s{s}_w{d}_n{d}", .{ @tagName(k.sdec.codebook), k.sdec.window.bits(), k.sn });
    if (k.sdec.codebook.tabledHashMask()) try name.print(alloc, "_t{s}", .{codebook_tables.tables[k.sdec.table.?].digest});
    if (coded) {
        for (k.classes[0..k.class_count]) |c| try name.print(alloc, "_{d}x{d}", .{ c.gate, c.up });
        try name.print(alloc, "_g{d}", .{k.group});
    }
    if (k.stage == .mid) try name.print(alloc, "_clamp{d}", .{k.limit});
    if (k.one_row) try name.appendSlice(alloc, "_one");
    if (k.tile_major) try name.appendSlice(alloc, "_tm");
    try name.append(alloc, 0);
    var slot: ?mlx.mlx_fast_metal_kernel = null;
    const kernel = try kernels.getNamedKernel(&slot, @ptrCast(name.items.ptr), try fuse.insOf(alloc, a, b), try fuse.outsOf(alloc, a, b), src, head);
    return .{ .kernel = kernel, .cfg = try fuse.config(alloc, a.launch, b.launch, k.b_first) };
}

/// The pointer stays valid until the next stage is built.
fn stageFor(k: FuseKey) !*const FusedStage {
    if (fused_stages.getPtr(k)) |f| return f;
    const built = try buildStage(k);
    try fused_stages.put(std.heap.page_allocator, k, built);
    return fused_stages.getPtr(k).?;
}

fn applyStage(s: mlx.mlx_stream, f: *const FusedStage, a: []const mlx.mlx_array, b: []const mlx.mlx_array, n_out: usize) !mlx.mlx_vector_array {
    var all: [16]mlx.mlx_array = undefined;
    @memcpy(all[0..a.len], a);
    @memcpy(all[a.len..][0..b.len], b);
    return kernels.applyOuts(s, f.kernel, all[0 .. a.len + b.len], f.cfg, n_out);
}

/// The finish reduce with the shared expert's down finish in one more simdgroup per threadgroup: its H128 and `svh`
/// scale, rounded to the shared chain's dtype `ST`, then the routed sum's own `T` add, in the statements of the two
/// reduces it replaces.
const REDUCE_FUSED_SOURCE: [:0]const u8 =
    \\threadgroup float vals[(uint(TOPK) + 1u) * 128u];
    \\const uint block = uint(threadgroup_position_in_grid.x);
    \\const uint row = uint(threadgroup_position_in_grid.y);
    \\const uint sg = uint(simdgroup_index_in_threadgroup);
    \\const ushort lane = thread_index_in_simdgroup;
    \\const uint base = block * 128u;
    \\const bool own = sg >= uint(TOPK);
    \\const uint slot = own ? row : row * uint(TOPK) + sg;
    \\const device half *rows_in = own ? s_inner : inner;
    \\const device half *scale = own ? s_svh : svh;
    \\const size_t xb = (size_t)slot * (size_t)(ODIM) + base;
    \\const size_t sb = own ? (size_t)base : (size_t)uint(slots[slot]) * (size_t)(ODIM) + base;
    \\const float scv = 0.08838834764831845f;
    \\float4 v = float4(float(rows_in[xb + lane]), float(rows_in[xb + lane + 32u]), float(rows_in[xb + lane + 64u]), float(rows_in[xb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\vals[sg * 128u + lane] = (s0 + s2) * scv * float(scale[sb + lane]);
    \\vals[sg * 128u + lane + 32u] = (s1 + s3) * scv * float(scale[sb + lane + 32u]);
    \\vals[sg * 128u + lane + 64u] = (s0 - s2) * scv * float(scale[sb + lane + 64u]);
    \\vals[sg * 128u + lane + 96u] = (s1 - s3) * scv * float(scale[sb + lane + 96u]);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (sg == 0u) {
    \\  float a0 = 0.0f;
    \\  float a1 = 0.0f;
    \\  float a2 = 0.0f;
    \\  float a3 = 0.0f;
    \\  for (uint k = 0u; k < uint(TOPK); k++) {
    \\    const float w = float(sc[row * uint(TOPK) + k]);
    \\    a0 += vals[k * 128u + lane] * w;
    \\    a1 += vals[k * 128u + lane + 32u] * w;
    \\    a2 += vals[k * 128u + lane + 64u] * w;
    \\    a3 += vals[k * 128u + lane + 96u] * w;
    \\  }
    \\  const float w1 = float(unit[0]);
    \\  float b0 = 0.0f;
    \\  float b1 = 0.0f;
    \\  float b2 = 0.0f;
    \\  float b3 = 0.0f;
    \\  b0 += vals[uint(TOPK) * 128u + lane] * w1;
    \\  b1 += vals[uint(TOPK) * 128u + lane + 32u] * w1;
    \\  b2 += vals[uint(TOPK) * 128u + lane + 64u] * w1;
    \\  b3 += vals[uint(TOPK) * 128u + lane + 96u] * w1;
    \\  const size_t yb = (size_t)row * (size_t)(ODIM) + base;
    \\  y[yb + lane] = T(float(T(a0)) + float(ST(b0)));
    \\  y[yb + lane + 32u] = T(float(T(a1)) + float(ST(b1)));
    \\  y[yb + lane + 64u] = T(float(T(a2)) + float(ST(b2)));
    \\  y[yb + lane + 96u] = T(float(T(a3)) + float(ST(b3)));
    \\}
;

var unit_score: ?mlx.mlx_array = null;

fn unitScore() mlx.mlx_array {
    if (unit_score) |a| return a;
    const one: f32 = 1;
    const a = mlx.mlx_array_new_data(&one, &[_]c_int{1}, 1, .float32);
    unit_score = a;
    return a;
}

fn reduceFused(s: mlx.mlx_stream, down: mlx.mlx_array, rb: *const Ragged, slots: mlx.mlx_array, scores: mlx.mlx_array, sdown: mlx.mlx_array, se: SharedExpert, rows: c_int, topk: c_int, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    const key: ReduceKey = .{ .hidden = rb.hidden, .rows = rows, .topk = topk };
    const cfg = fused_reduce_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = if (rows == 1) [_]c_int{ rb.hidden, 0 } else [_]c_int{ rows, rb.hidden };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, if (rows == 1) 1 else 2, out_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * (topk + 1) * @divExact(rb.hidden, 128), rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32 * (topk + 1), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", out_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "ST", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", rb.hidden));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        fused_reduce_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = try kernels.getNamedKernel(&fused_reduce_kernel, "sushi_exl3_down_reduce_fused_shared", &.{ "inner", "svh", "slots", "sc", "s_inner", "s_svh", "unit" }, &.{"y"}, REDUCE_FUSED_SOURCE, "");
    const ov = try kernels.applyOuts(s, kernel, &.{ down, rb.svh[2], slots, scores, sdown, se.bank.down.svh, unitScore() }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    return getOut(ov, 0);
}

/// The shared expert's lane-arm shapes at this bank, or null where its own chain would run another arm.
fn sharedRate(se: SharedExpert, rb: *const Ragged, rows: usize) ?struct { n: u32, one_row_pair: bool, one_row_down: bool } {
    if (!se.dec.codebook.servesGlmFastPaths() or rows < 1 or rows > kernels.DECODE_ROWS_MAX) return null;
    const g = mlx.getShape(se.bank.gate.trellis);
    const u = mlx.getShape(se.bank.up.trellis);
    const d = mlx.getShape(se.bank.down.trellis);
    if (g.len != 4 or u.len != 4 or d.len != 4 or g[0] != 1 or u[0] != 1 or d[0] != 1) return null;
    if (g[1] * 16 != rb.hidden or g[2] * 16 != rb.inter or d[1] * 16 != rb.inter or d[2] * 16 != rb.hidden) return null;
    if (g[3] != u[3] or g[3] != d[3] or !kernels.laneServesRate(g[3])) return null;
    const rate = exl3.kFromPackedDim(@intCast(g[3])) orelse return null;
    return .{ .n = rate.n, .one_row_pair = rows == 1 and kernels.oneRowCoopServes(g[1], g[2]), .one_row_down = rows == 1 and kernels.oneRowCoopServes(d[1], d[2]) };
}

/// `moeDecodeClamped` over the prepared chain with the shared expert `se` (bf16 `x`) riding its launches: the routed
/// sum plus the shared chain's bf16 output, in `out_dtype`, with the bytes `moeDecodeClamped` gives that output as its
/// `shared`. Null where the chain does not apply and the shared expert runs on its own.
pub fn moeDecodeClampedSharedFused(s: mlx.mlx_stream, x: mlx.mlx_array, rb: *const Ragged, slots: mlx.mlx_array, scores: mlx.mlx_array, topk: c_int, out_dtype: mlx.mlx_dtype, se: SharedExpert, limit: c_int) !?mlx.mlx_array {
    if (rb.prefill_only or !preparedOn(rb) or mlx.mlx_array_dtype(x) != .bfloat16 or topk > kernels.REDUCE_MAX_TOPK - 1) return null;
    if (rb.tile_major and !kernels.gemvLayout(1).funnel) return null;
    const rows = mlx.getShape(x)[0];
    const sr = sharedRate(se, rb, @intCast(rows)) orelse return null;
    const nslots: c_int = @intCast(mlx.mlx_array_size(slots));
    var pc: [max_groups]RateClass = undefined;
    var dc: [max_groups]RateClass = undefined;
    const pair = classesOf(rb, false, &pc);
    const downs = classesOf(rb, true, &dc);
    logEngaged(.decode, rb, pair.len, downs.len);
    const group = groupFor(@intCast(rows), @intCast(nslots));
    if (group > 0) kernels.logGroupedEngaged(group);
    const layout = kernels.gemvLayout(@divExact(rb.inter, 16));
    const down_layout = kernels.gemvLayout(@divExact(rb.hidden, 16));
    var key: FuseKey = .{
        .stage = .prep,
        .nslots = nslots,
        .rows = rows,
        .topk = topk,
        .hidden = rb.hidden,
        .inter = rb.inter,
        .group = 0,
        .tiles = 0,
        .funnel = layout.funnel,
        .tile_major = rb.tile_major,
        .limit = 0,
        .classes = std.mem.zeroes([max_groups]RateClass),
        .class_count = 0,
        .dec = kernels.activeDecode(),
        .sdec = se.dec,
        .sn = sr.n,
        .one_row = false,
        .b_first = fused_shared_first,
    };
    const zeros = zeroSlots(@intCast(rows));
    const prep_ov = try applyStage(s, try stageFor(key), &.{ x, rb.suh[0], rb.suh[1], slots }, &.{ x, se.bank.gate.suh, se.bank.up.suh, zeros }, 3);
    defer _ = mlx.mlx_vector_array_free(prep_ov);
    const prep = try getOut(prep_ov, 0);
    defer _ = mlx.mlx_array_free(prep);
    const sxg = try getOut(prep_ov, 1);
    defer _ = mlx.mlx_array_free(sxg);
    const sxu = try getOut(prep_ov, 2);
    defer _ = mlx.mlx_array_free(sxu);

    const pair_layout = kernels.groupedLayout(layout, group);
    key.stage = .pair;
    key.group = group;
    key.tiles = pair_layout.tiles;
    key.funnel = pair_layout.funnel;
    @memcpy(key.classes[0..pair.len], pair);
    key.class_count = pair.len;
    key.one_row = sr.one_row_pair;
    const pair_ov = try applyStage(s, try stageFor(key), &.{ prep, rb.trellis[0], rb.trellis[1], slots, rb.meta }, &.{ sxg, sxu, se.bank.gate.trellis, se.bank.up.trellis, zeros }, 4);
    defer _ = mlx.mlx_vector_array_free(pair_ov);
    const ig = try getOut(pair_ov, 0);
    defer _ = mlx.mlx_array_free(ig);
    const iu = try getOut(pair_ov, 1);
    defer _ = mlx.mlx_array_free(iu);
    const sig = try getOut(pair_ov, 2);
    defer _ = mlx.mlx_array_free(sig);
    const siu = try getOut(pair_ov, 3);
    defer _ = mlx.mlx_array_free(siu);

    key.stage = .mid;
    key.group = 0;
    key.tiles = 0;
    key.funnel = layout.funnel;
    key.classes = std.mem.zeroes([max_groups]RateClass);
    key.class_count = 0;
    key.limit = limit;
    key.one_row = false;
    const mid_ov = try applyStage(s, try stageFor(key), &.{ ig, iu, rb.svh[0], rb.svh[1], rb.suh[2], slots }, &.{ sig, siu, se.bank.gate.svh, se.bank.up.svh, se.bank.down.suh, zeros }, 2);
    defer _ = mlx.mlx_vector_array_free(mid_ov);
    const middle = try getOut(mid_ov, 0);
    defer _ = mlx.mlx_array_free(middle);
    const smid = try getOut(mid_ov, 1);
    defer _ = mlx.mlx_array_free(smid);

    const dn_layout = kernels.groupedLayout(down_layout, group);
    key.stage = .down;
    key.group = group;
    key.tiles = dn_layout.tiles;
    key.funnel = dn_layout.funnel;
    key.limit = 0;
    @memcpy(key.classes[0..downs.len], downs);
    key.class_count = downs.len;
    key.one_row = sr.one_row_down;
    const down_ov = try applyStage(s, try stageFor(key), &.{ middle, rb.trellis[2], slots, rb.meta }, &.{ smid, se.bank.down.trellis, zeros }, 2);
    defer _ = mlx.mlx_vector_array_free(down_ov);
    const down = try getOut(down_ov, 0);
    defer _ = mlx.mlx_array_free(down);
    const sdown = try getOut(down_ov, 1);
    defer _ = mlx.mlx_array_free(sdown);
    return try reduceFused(s, down, rb, slots, scores, sdown, se, rows, topk, out_dtype);
}

// ---- Qwen's shared expert inside the unprepared decode chain ----

const QwenKey = struct {
    down: bool,
    rows: c_int,
    topk: c_int,
    width: c_int,
    group: c_int,
    logit_group: c_int,
    logit: bool,
    hidden: c_int,
    inter: c_int,
    funnel: bool,
    tiles: c_int,
    tile_major: bool,
    classes: [max_groups]RateClass,
    class_count: usize,
    dec: exl3.Decode,
    b_first: bool,
};

var qwen_stages: std.AutoHashMapUnmanaged(QwenKey, FusedStage) = .empty;
var qwen_layers: usize = 0;
var qwen_launches: usize = 0;
/// Layers run folded, and the GPU launches they issued; test seams.
pub fn qwenFoldedCount() usize {
    return qwen_layers;
}
pub fn qwenLaunchCount() usize {
    return qwen_launches;
}

fn qwenStage(k: QwenKey) !*const FusedStage {
    if (qwen_stages.getPtr(k)) |f| return f;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const classes = k.classes[0..k.class_count];
    const nslots = k.rows * k.topk;
    const layout: kernels.GemvLayout = .{ .funnel = k.funnel, .tiles = k.tiles };
    const geo: riders.Geometry = .{ .hidden = k.hidden, .width = k.width, .group = k.group, .logit_group = k.logit_group };
    const head = try headerOf(k.dec);
    const a: fuse.Part = if (k.down) .{
        .launch = try decodeLaunch(alloc, .{ .in_dim = k.inter, .out_dim = k.hidden, .nslots = nslots, .topk = 0, .down = true, .layout = layout }),
        .body = try mixedText(alloc, .down, classes, 0, 0, k.tile_major),
        .ins = &.{ "ig", "iu", "trellis", "svhg", "svhu", "suhd", "slots", "meta" },
        .outs = &.{"y"},
        .header = head,
    } else .{
        .launch = try decodeLaunch(alloc, .{ .in_dim = k.hidden, .out_dim = k.inter, .nslots = nslots, .topk = k.topk, .down = false, .layout = layout }),
        .body = try mixedText(alloc, .pair, classes, 0, 0, k.tile_major),
        .ins = &.{ "x", "suhg", "suhu", "tg", "tu", "slots", "meta" },
        .outs = &.{ "yg", "yu" },
        .header = head,
    };
    const b = if (k.down) try riders.downPart(alloc, geo, k.rows) else try riders.gateUpPart(alloc, geo, k.rows, k.logit);
    const src = try fuse.source(alloc, a, b);
    const header_text = try fuse.header(alloc, a, b);
    var name: std.ArrayList(u8) = .empty;
    try name.print(alloc, "sushi_qwen_rg_fold_{s}_{s}_w{d}", .{ if (k.down) "down" else "pair", @tagName(k.dec.codebook), k.dec.window.bits() });
    if (k.dec.codebook.tabledHashMask()) try name.print(alloc, "_t{s}", .{codebook_tables.tables[k.dec.table.?].digest});
    for (classes) |c| try name.print(alloc, "_{d}x{d}", .{ c.gate, c.up });
    if (k.logit) try name.appendSlice(alloc, "_logit");
    if (k.tile_major) try name.appendSlice(alloc, "_tm");
    try name.append(alloc, 0);
    var slot: ?mlx.mlx_fast_metal_kernel = null;
    const kernel = try kernels.getNamedKernel(&slot, @ptrCast(name.items.ptr), try fuse.insOf(alloc, a, b), try fuse.outsOf(alloc, a, b), src, header_text);
    const built: FusedStage = .{ .kernel = kernel, .cfg = try fuse.config(alloc, a.launch, b.launch, k.b_first) };
    try qwen_stages.put(std.heap.page_allocator, k, built);
    return qwen_stages.getPtr(k).?;
}

/// `moeRagged`'s decode chain with the gated shared expert `sh` riding its pair and down launches, for one to eight rows
/// of `x` [1, rows, hidden]: the routed sum plus `sigmoid(gate) * shared`, bf16, in `x`'s shape. `gate` [rows] is the
/// shared-expert gate's bf16 logit unless `sh.logit` names its bank. Null where the chain or the riders do not serve the
/// inputs (the prepared chain, a prefill-only bank); nothing has run then.
pub fn moeGatedSharedFolded(s: mlx.mlx_stream, x: mlx.mlx_array, rb: *const Ragged, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: exl3.Decode, sh: riders.Shared, gate: ?mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or rb.prefill_only or preparedOn(rb)) return null;
    if (rb.tile_major and !kernels.gemvLayout(1).funnel) return null;
    const route = riders.Routing.of(x, inds, scores) orelse return null;
    const rows = route.rows;
    const topk = route.topk;
    const hidden = mlx.getShape(x)[2];
    if (hidden != rb.hidden) return null;
    const geo = riders.geometry(sh, hidden) orelse return null;
    const want_logit = sh.logit != null;
    if (!want_logit) {
        const g = gate orelse return null;
        if (mlx.mlx_array_dtype(g) != .bfloat16 or mlx.mlx_array_size(g) != @as(usize, @intCast(rows))) return null;
    }
    dec.requireTable() catch return null;

    kernels.setDecodeParams(dec);
    const nslots = rows * topk;
    var pc: [max_groups]RateClass = undefined;
    var dc: [max_groups]RateClass = undefined;
    const pair_classes = classesOf(rb, false, &pc);
    const down_classes = classesOf(rb, true, &dc);
    logEngaged(.decode, rb, pair_classes.len, down_classes.len);
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows, hidden }, 2, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, inds, &[_]c_int{nslots}, 1, s));
    var slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slots);
    try mlx.check(mlx.mlx_astype(&slots, flat, .uint32, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_reshape(&sc, scores, &[_]c_int{nslots}, 1, s));

    const pair_layout = kernels.gemvLayout(@divExact(rb.inter, 16));
    var key: QwenKey = .{
        .down = false,
        .rows = rows,
        .topk = topk,
        .width = geo.width,
        .group = geo.group,
        .logit_group = geo.logit_group,
        .logit = want_logit,
        .hidden = rb.hidden,
        .inter = rb.inter,
        .funnel = pair_layout.funnel,
        .tiles = pair_layout.tiles,
        .tile_major = rb.tile_major,
        .classes = std.mem.zeroes([max_groups]RateClass),
        .class_count = pair_classes.len,
        .dec = kernels.activeDecode(),
        .b_first = fused_shared_first,
    };
    @memcpy(key.classes[0..pair_classes.len], pair_classes);
    var buf: [11]mlx.mlx_array = undefined;
    const rider_in = riders.gateUpInputs(sh, x2, &buf);
    var pair_in: [18]mlx.mlx_array = undefined;
    pair_in[0..7].* = .{ x2, rb.suh[0], rb.suh[1], rb.trellis[0], rb.trellis[1], slots, rb.meta };
    @memcpy(pair_in[7..][0..rider_in.len], rider_in);
    const pair_ov = try kernels.applyOuts(s, (try qwenStage(key)).kernel, pair_in[0 .. 7 + rider_in.len], (try qwenStage(key)).cfg, if (want_logit) 4 else 3);
    defer _ = mlx.mlx_vector_array_free(pair_ov);
    const ig = try getOut(pair_ov, 0);
    defer _ = mlx.mlx_array_free(ig);
    const iu = try getOut(pair_ov, 1);
    defer _ = mlx.mlx_array_free(iu);
    const mid = try getOut(pair_ov, 2);
    defer _ = mlx.mlx_array_free(mid);
    const logit_out: ?mlx.mlx_array = if (want_logit) try getOut(pair_ov, 3) else null;
    defer if (logit_out) |l| {
        _ = mlx.mlx_array_free(l);
    };

    const down_layout = kernels.gemvLayout(@divExact(rb.hidden, 16));
    key.down = true;
    key.funnel = down_layout.funnel;
    key.tiles = down_layout.tiles;
    key.classes = std.mem.zeroes([max_groups]RateClass);
    key.class_count = down_classes.len;
    @memcpy(key.classes[0..down_classes.len], down_classes);
    const down_stage = try qwenStage(key);
    const down_ov = try kernels.applyOuts(s, down_stage.kernel, &.{ ig, iu, rb.trellis[2], rb.svh[0], rb.svh[1], rb.suh[2], slots, rb.meta, mid, sh.down.weight, sh.down.scales, sh.down.biases }, down_stage.cfg, 2);
    defer _ = mlx.mlx_vector_array_free(down_ov);
    const down = try getOut(down_ov, 0);
    defer _ = mlx.mlx_array_free(down);
    const shared_out = try getOut(down_ov, 1);
    defer _ = mlx.mlx_array_free(shared_out);

    qwen_layers += 1;
    // Pair, down, reduce.
    qwen_launches += 3;
    const y = try kernels.downFinishReduceWithShared(s, down, rb.svh[2], slots, sc, rb.hidden, rows, topk, .bfloat16, shared_out, logit_out orelse gate.?, false);
    defer _ = mlx.mlx_array_free(y);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const xs = mlx.getShape(x);
    try mlx.check(mlx.mlx_reshape(&out, y, xs.ptr, xs.len, s));
    return out;
}

/// Whether a decode chain prepares its inputs once per slot, at DeepSeek's geometry: a threadgroup of the fused chain
/// otherwise re-derives its slice of the input and of the SwiGLU mid, which the two extra dispatches cost less than.
fn preparedOn(rb: *const Ragged) bool {
    if (prepared_force) |v| return v;
    return rb.hidden == 5120 and rb.inter == 2304;
}

var engaged: [2]bool = @splat(false);

fn logEngaged(arm: enum { decode, prefill }, rb: *const Ragged, pair: usize, down: usize) void {
    if (engaged[@intFromEnum(arm)]) return;
    engaged[@intFromEnum(arm)] = true;
    if (@import("builtin").is_test) return;
    switch (arm) {
        .decode => log.info("[exl3-groups] decode engaged groups={d} pair classes={d} down classes={d}\n", .{ rb.count, pair, down }),
        .prefill => log.info("[exl3-groups] prefill engaged groups={d}, one GEMM per group\n", .{rb.count}),
    }
}

/// Test arms of the prepared verify chain: null follows the geometry, a value forces it.
pub var prepared_force: ?bool = null;
/// Test arm of the expert grouping: null follows the policy, 0 forces one threadgroup per slot, a bucket forces it.
pub var group_force: ?c_int = null;

/// Member bucket of a verify chain whose slots share experts: pairs, so one threadgroup decodes an expert once for two
/// rows and still carries two output tiles; wider buckets halve the tiles per threadgroup and lost at 5 rows and up.
/// 0 keeps one threadgroup per slot.
fn groupFor(rows: usize, nslots: usize) c_int {
    if (rows < 2 or nslots > @as(usize, @intCast(kernels.DECODE_GROUP_MAX_SLOTS)) or !kernels.gemvLayout(1).funnel) return 0;
    if (group_force) |g| return g;
    return kernels.GROUP_BUCKETS[0];
}

var prepared_engaged: bool = false;

/// Routed SwiGLU over decode or verify rows `x` [rows, hidden] for flat u32 `slots` and `scores`
/// [rows*topk]: ONE pair and ONE down dispatch whatever the group count (a pass per group lost at
/// every count: a dispatch costs a fixed ~10 us), then the served finish reduce, folding `shared`
/// (with `gate`: the bf16 `sigmoid(gate) * shared`) as the uniform chain does.
pub fn moeDecode(s: mlx.mlx_stream, x: mlx.mlx_array, rb: *const Ragged, slots: mlx.mlx_array, scores: mlx.mlx_array, topk: c_int, out_dtype: mlx.mlx_dtype, shared: ?mlx.mlx_array, gate: ?mlx.mlx_array) !mlx.mlx_array {
    return moeDecodeClamped(s, x, rb, slots, scores, topk, out_dtype, shared, gate, 0);
}

/// `moeDecode` with the SwiGLU clamped at `limit` (0: none).
pub fn moeDecodeClamped(s: mlx.mlx_stream, x: mlx.mlx_array, rb: *const Ragged, slots: mlx.mlx_array, scores: mlx.mlx_array, topk: c_int, out_dtype: mlx.mlx_dtype, shared: ?mlx.mlx_array, gate: ?mlx.mlx_array, limit: c_int) !mlx.mlx_array {
    if (rb.prefill_only) return error.Exl3SplitBankIsPrefillOnly;
    if (rb.tile_major and !kernels.gemvLayout(1).funnel) return error.Exl3TileMajorNeedsFunnel;
    const rows = mlx.getShape(x)[0];
    var pc: [max_groups]RateClass = undefined;
    var dc: [max_groups]RateClass = undefined;
    const pair = classesOf(rb, false, &pc);
    const downs = classesOf(rb, true, &dc);
    logEngaged(.decode, rb, pair.len, downs.len);
    const prepared = preparedOn(rb);
    const group = if (prepared) groupFor(@intCast(rows), @intCast(mlx.mlx_array_size(slots))) else 0;
    if (group > 0) kernels.logGroupedEngaged(group);
    const inner = if (prepared) blk: {
        const prep = try kernels.pairPrepare(s, x, rb.suh[0], rb.suh[1], slots, rb.hidden, @intCast(mlx.mlx_array_size(slots)), topk);
        defer _ = mlx.mlx_array_free(prep);
        break :blk try pairPreparedMixed(s, prep, rb, slots, topk, pair, group);
    } else try pairMixed(s, x, rb, slots, topk, pair);
    defer for (inner) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const down = if (prepared) blk: {
        const middle = try midMixed(s, inner, rb, slots, limit);
        defer _ = mlx.mlx_array_free(middle);
        prepared_engaged = true;
        break :blk try downPreparedMixed(s, middle, rb, slots, downs, group);
    } else try downMixed(s, inner, rb, slots, downs, limit);
    defer _ = mlx.mlx_array_free(down);
    return kernels.downFinishReduceWithShared(s, down, rb.svh[2], slots, scores, rb.hidden, rows, topk, out_dtype, shared, gate, false);
}

// ---- Prefill ----

/// One threadgroup (a thread per expert) builds the window table with one segment per group, group g
/// at `g * C + base_g` with room for `C + E_g` windows (C windows of all rows, plus one partial window
/// per expert), the inverse sort order, and each sorted row's expert index inside its group.
const GROUP_WINDOW_SOURCE: [:0]const u8 =
    \\const uint tid = uint(thread_index_in_threadgroup);
    \\const uint n = uint(count[0]), groups = uint(count[1]), experts = uint(threads_per_threadgroup.x);
    \\const uint C = (n + uint(WIN) - 1u) / uint(WIN);
    \\const uint capacity = groups * C + experts;
    \\threadgroup uint counts[MAX_EXPERTS];
    \\for (uint i = tid; i < capacity; i += experts) { starts[i] = 0u; nlives[i] = 0u; }
    \\for (uint i = tid; i < n; i += experts) {
    \\  inverse[uint(order[i])] = i;
    \\  local[i] = meta[uint(eids[i]) * 8u + 7u];
    \\}
    \\uint lo = 0u, hi = n;
    \\while (lo < hi) { uint m = (lo + hi) >> 1u; if (uint(eids[m]) < tid) lo = m + 1u; else hi = m; }
    \\const uint first = lo;
    \\hi = n;
    \\while (lo < hi) { uint m = (lo + hi) >> 1u; if (uint(eids[m]) <= tid) lo = m + 1u; else hi = m; }
    \\const uint length = lo - first;
    \\counts[tid] = (length + uint(WIN) - 1u) / uint(WIN);
    \\threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    \\const uint group_first = tid - meta[tid * 8u + 7u];
    \\uint offset = meta[tid * 8u + 3u] * C + group_first;
    \\for (uint e = group_first; e < tid; e++) offset += counts[e];
    \\for (uint i = 0u; i < counts[tid]; i++) {
    \\  starts[offset + i] = first + i * uint(WIN);
    \\  nlives[offset + i] = min(uint(WIN), length - i * uint(WIN));
    \\}
;

/// The most experts the window threadgroup holds, one thread each.
pub const PREFILL_MAX_EXPERTS: u32 = 512;

var group_window_kernel: ?mlx.mlx_fast_metal_kernel = null;

const GroupWindows = struct { starts: mlx.mlx_array, nlives: mlx.mlx_array, inverse: mlx.mlx_array, local: mlx.mlx_array, per_all: c_int };

fn groupWindows(s: mlx.mlx_stream, rb: *const Ragged, sorted: mlx.mlx_array, order: mlx.mlx_array, n: c_int, win: c_int) !GroupWindows {
    if (rb.experts > PREFILL_MAX_EXPERTS) return error.Exl3RaggedPrefillExperts;
    const per_all = @divTrunc(n + win - 1, win);
    const capacity = @as(c_int, @intCast(rb.count)) * per_all + @as(c_int, @intCast(rb.experts));
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (0..2) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{capacity}, 1, .uint32));
    for (0..2) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{n}, 1, .uint32));
    const e: c_int = @intCast(rb.experts);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, e, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, e, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WIN", win));
    // The expert and group counts of a split bank change per chunk, so they ride in `count`, not the kernel name.
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "MAX_EXPERTS", @intCast(PREFILL_MAX_EXPERTS)));
    const nn = [2]u32{ @intCast(n), @intCast(rb.count) };
    const count = mlx.mlx_array_new_data(&nn, &.{2}, 1, .uint32);
    defer _ = mlx.mlx_array_free(count);
    const kernel = try kernels.getNamedKernel(&group_window_kernel, "sushi_exl3_rg_windows", &.{ "eids", "order", "count", "meta" }, &.{ "starts", "nlives", "inverse", "local" }, GROUP_WINDOW_SOURCE, "");
    const ov = try kernels.applyOuts(s, kernel, &.{ sorted, order, count, rb.meta }, cfg, 4);
    defer _ = mlx.mlx_vector_array_free(ov);
    var out: GroupWindows = .{ .starts = .{ .ctx = null }, .nlives = .{ .ctx = null }, .inverse = .{ .ctx = null }, .local = .{ .ctx = null }, .per_all = per_all };
    errdefer for ([_]mlx.mlx_array{ out.starts, out.nlives, out.inverse, out.local }) |a| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a);
    };
    out.starts = try getOut(ov, 0);
    out.nlives = try getOut(ov, 1);
    out.inverse = try getOut(ov, 2);
    out.local = try getOut(ov, 3);
    return out;
}

/// Planes a combine dispatch reads at most: Metal binds 31 buffers.
const COMBINE_PLANES: usize = 24;
var combine_kernels: [2]?mlx.mlx_fast_metal_kernel = @splat(null);

/// Each sorted row of `planes` (group `first + i`'s GEMM output, valid on its own rows only), or of
/// `prev` for a row of a group before `first`.
fn combineSource(alloc: std.mem.Allocator, planes: usize, chained: bool) ![:0]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    try out.appendSlice(alloc,
        \\const uint i = thread_position_in_grid.x;
        \\const uint row = thread_position_in_grid.y;
        \\if (i >= uint(DIM) / 4u) return;
        \\const uint g = meta[uint(eids[row]) * 8u + 3u];
        \\const size_t at = (size_t)row * (uint(DIM) / 4u) + i;
        \\
    );
    if (chained) try out.appendSlice(alloc, "if (g < uint(FIRST)) { ((device half4 *)y)[at] = ((const device half4 *)prev)[at]; return; }\n");
    try out.appendSlice(alloc, "switch (g - uint(FIRST)) {\n");
    for (0..planes) |p| try out.print(alloc, "case {d}u: ((device half4 *)y)[at] = ((const device half4 *)p{d})[at]; break;\n", .{ p, p });
    try out.appendSlice(alloc, "default: break;\n}\n");
    return out.toOwnedSliceSentinel(alloc, 0);
}

const PLANE_NAMES = blk: {
    var names: [COMBINE_PLANES][*:0]const u8 = undefined;
    for (&names, 0..) |*n, i| n.* = std.fmt.comptimePrint("p{d}", .{i});
    break :blk names;
};

fn combine(s: mlx.mlx_stream, planes: []const mlx.mlx_array, sorted: mlx.mlx_array, meta: mlx.mlx_array) !mlx.mlx_array {
    const sh = mlx.getShape(planes[0]);
    var prev: ?mlx.mlx_array = null;
    defer if (prev) |p| {
        _ = mlx.mlx_array_free(p);
    };
    var first: usize = 0;
    while (true) {
        const m = @min(COMBINE_PLANES, planes.len - first);
        const chained = prev != null;
        const slot = &combine_kernels[@intFromBool(chained)];
        // Every dispatch binds all COMBINE_PLANES planes (the last real one repeated, its cases never reached), so the
        // source and the library are one per chain position whatever the group count.
        const kernel = slot.* orelse blk: {
            const alloc = std.heap.page_allocator;
            const src = try combineSource(alloc, COMBINE_PLANES, chained);
            defer alloc.free(src);
            const name = try std.fmt.allocPrintSentinel(alloc, "sushi_exl3_rg_combine{s}", .{if (chained) "_chained" else ""}, 0);
            defer alloc.free(name);
            var ins: [COMBINE_PLANES + 3][*:0]const u8 = undefined;
            ins[0] = "eids";
            ins[1] = "meta";
            var k: usize = 2;
            if (chained) {
                ins[k] = "prev";
                k += 1;
            }
            @memcpy(ins[k..][0..COMBINE_PLANES], PLANE_NAMES[0..COMBINE_PLANES]);
            break :blk try kernels.getNamedKernel(slot, name.ptr, ins[0 .. k + COMBINE_PLANES], &.{"y"}, src, "");
        };
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(sh[1], 4), sh[0], 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(256, @divExact(sh[1], 4)), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "DIM", sh[1]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "FIRST", @intCast(first)));
        var ins: [COMBINE_PLANES + 3]mlx.mlx_array = undefined;
        ins[0] = sorted;
        ins[1] = meta;
        var k: usize = 2;
        if (prev) |p| {
            ins[k] = p;
            k += 1;
        }
        @memcpy(ins[k..][0..m], planes[first..][0..m]);
        for (ins[k + m .. k + COMBINE_PLANES]) |*pad| pad.* = planes[first + m - 1];
        const ov = try kernels.applyOuts(s, kernel, ins[0 .. k + COMBINE_PLANES], cfg, 1);
        defer _ = mlx.mlx_vector_array_free(ov);
        const y = try getOut(ov, 0);
        first += m;
        if (first == planes.len) return y;
        if (prev) |p| _ = mlx.mlx_array_free(p);
        prev = y;
    }
}

/// One projection's GEMM, once per group over the group's own window segment and bank view, combined.
fn groupedGemm(s: mlx.mlx_stream, x: mlx.mlx_array, rb: *const Ragged, pi: usize, sorted: mlx.mlx_array, w: *const GroupWindows, win: c_int) !mlx.mlx_array {
    var planes: [max_groups]mlx.mlx_array = undefined;
    var made: usize = 0;
    defer for (planes[0..made]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (0..rb.count) |g| {
        const lo: c_int = @as(c_int, @intCast(g)) * w.per_all + @as(c_int, @intCast(rb.base[g]));
        const len = w.per_all + @as(c_int, @intCast(rb.experts_of[g]));
        var starts = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(starts);
        try mlx.check(mlx.mlx_slice(&starts, w.starts, &[_]c_int{lo}, 1, &[_]c_int{lo + len}, 1, &[_]c_int{1}, 1, s));
        var nlives = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(nlives);
        try mlx.check(mlx.mlx_slice(&nlives, w.nlives, &[_]c_int{lo}, 1, &[_]c_int{lo + len}, 1, &[_]c_int{1}, 1, s));
        const bank = rb.views[g];
        const trellis = projOf(bank, pi).trellis;
        planes[made] = try kernels.innerGemmSortedTableOrdered(s, x, trellis, w.local, win, true, .{ .starts = starts, .nlives = nlives, .nwin = len }, if (rb.tile_major) .tile_major else .pack);
        made += 1;
    }
    if (made == 1) {
        made = 0;
        return planes[0];
    }
    return combine(s, planes[0..made], sorted, rb.meta);
}

/// Routed SwiGLU over prefill rows `x` [rows, hidden]: the served prefill chain (token prepare, gate
/// and up GEMMs, SwiGLU mid, down GEMM, inverse-indexed finish) with each GEMM run once per group.
pub fn moePrefill(s: mlx.mlx_stream, x: mlx.mlx_array, rb: *const Ragged, slots: mlx.mlx_array, scores: mlx.mlx_array, topk: c_int) !mlx.mlx_array {
    return moePrefillClamped(s, x, rb, slots, scores, topk, 0);
}

/// `moePrefill` with the SwiGLU clamped at `limit` (0: none).
pub fn moePrefillClamped(s: mlx.mlx_stream, x: mlx.mlx_array, rb: *const Ragged, slots: mlx.mlx_array, scores: mlx.mlx_array, topk: c_int, limit: c_int) !mlx.mlx_array {
    const rows = mlx.getShape(x)[0];
    const n = rows * topk;
    logEngaged(.prefill, rb, 0, 0);
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots, 0, s));
    var order_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order_i);
    try mlx.check(mlx.mlx_astype(&order_i, order, .int32, s));
    var sorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted);
    try mlx.check(mlx.mlx_take_axis(&sorted, slots, order, 0, s));
    const prep = try kernels.pairPrepareFromTokens(s, x, rb.suh[0], rb.suh[1], sorted, order_i, rb.hidden, n, topk);
    defer _ = mlx.mlx_array_free(prep[0]);
    defer _ = mlx.mlx_array_free(prep[1]);
    const win = kernels.gemmWindowRows();
    const w = try groupWindows(s, rb, sorted, order_i, n, win);
    defer for ([_]mlx.mlx_array{ w.starts, w.nlives, w.inverse, w.local }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const g = try groupedGemm(s, prep[0], rb, 0, sorted, &w, win);
    defer _ = mlx.mlx_array_free(g);
    const u = try groupedGemm(s, prep[1], rb, 1, sorted, &w, win);
    defer _ = mlx.mlx_array_free(u);
    const mid = try kernels.midSwigluPrepWithLimit(s, g, u, rb.svh[0], rb.svh[1], rb.suh[2], sorted, rb.inter, n, limit);
    defer _ = mlx.mlx_array_free(mid);
    const d = try groupedGemm(s, mid, rb, 2, sorted, &w, win);
    defer _ = mlx.mlx_array_free(d);
    return kernels.finishMimoSorted(s, d, w.inverse, rb.svh[2], slots, scores, rb.hidden, rows, topk, mlx.mlx_array_dtype(x));
}

// ---- Tests ----

const Spec = struct {
    hidden: usize = 256,
    inter: usize = 256,
    /// n of gate, up and down per group.
    rates: []const [3]u32,
    /// Experts per group.
    experts: []const u32,
    rows: usize = 2,
    topk: usize = 4,
    seed: u64 = 0x5eed,
    /// suh(hidden), svh(inter), suh(inter), svh(hidden) magnitudes.
    banks: [4]f32 = .{ 0.9, 0.9, 0.9, 0.9 },
    x_scale: f32 = 0.05,
    /// Explicit slots [rows*topk]; random distinct experts per row otherwise.
    slots: ?[]const u32 = null,
};

/// A grouped layer with random trellises: the group banks as served today, the ragged bank over them,
/// and the host copies an oracle decodes.
const Fixture = struct {
    spec: Spec,
    banks: [max_groups]root.Bank = undefined,
    words: [max_groups][3][]u16 = undefined,
    suh: [max_groups][3][]u16 = undefined,
    svh: [max_groups][3][]u16 = undefined,
    rb: Ragged = .{},
    x: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
    xf: []f32,
    slot_ids: []u32,
    score_vals: []f32,
    experts: u32,

    fn init(alloc: std.mem.Allocator, s: mlx.mlx_stream, spec: Spec) !Fixture {
        var prng = std.Random.DefaultPrng.init(spec.seed);
        const rnd = prng.random();
        var f: Fixture = .{ .spec = spec, .x = undefined, .slots = undefined, .scores = undefined, .xf = undefined, .slot_ids = undefined, .score_vals = undefined, .experts = 0 };
        const tiles = (spec.hidden / 16) * (spec.inter / 16);
        for (spec.rates, spec.experts, 0..) |r, e, g| {
            for (0..3) |p| {
                const in_d = if (p == 2) spec.inter else spec.hidden;
                const out_d = if (p == 2) spec.hidden else spec.inter;
                const w = try alloc.alloc(u16, e * tiles * r[p]);
                for (w) |*v| v.* = @truncate(rnd.int(u32));
                const su = try alloc.alloc(u16, e * in_d);
                const sv = try alloc.alloc(u16, e * out_d);
                const mags = if (p == 2) [2]f32{ spec.banks[2], spec.banks[3] } else [2]f32{ spec.banks[0], spec.banks[1] };
                for (su) |*v| v.* = exl3.f32ToF16Bits(if (rnd.boolean()) mags[0] else -mags[0]);
                for (sv) |*v| v.* = exl3.f32ToF16Bits(if (rnd.boolean()) mags[1] else -mags[1]);
                f.words[g][p] = w;
                f.suh[g][p] = su;
                f.svh[g][p] = sv;
                const ci: c_int = @intCast(e);
                projPtr(&f.banks[g], p).* = .{
                    .trellis = mlx.mlx_array_new_data(w.ptr, &[_]c_int{ ci, @intCast(in_d / 16), @intCast(out_d / 16), @intCast(r[p]) }, 4, .uint16),
                    .suh = mlx.mlx_array_new_data(su.ptr, &[_]c_int{ ci, @intCast(in_d) }, 2, .float16),
                    .svh = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{ ci, @intCast(out_d) }, 2, .float16),
                };
            }
            f.experts += e;
        }
        f.rb = try Ragged.init(s, f.banks[0..spec.rates.len]);
        const xh = try alloc.alloc(u16, spec.rows * spec.hidden);
        f.xf = try alloc.alloc(f32, xh.len);
        for (xh, f.xf) |*b, *v| {
            b.* = exl3.f32ToF16Bits((rnd.float(f32) * 2 - 1) * spec.x_scale);
            v.* = exl3.f16BitsToF32(b.*);
        }
        f.slot_ids = try alloc.alloc(u32, spec.rows * spec.topk);
        if (spec.slots) |given| @memcpy(f.slot_ids, given) else for (0..spec.rows) |r| {
            const row = f.slot_ids[r * spec.topk ..][0..spec.topk];
            var k: usize = 0;
            while (k < spec.topk) {
                const pick = rnd.uintLessThan(u32, f.experts);
                if (std.mem.indexOfScalar(u32, row[0..k], pick) != null) continue;
                row[k] = pick;
                k += 1;
            }
        }
        f.score_vals = try alloc.alloc(f32, f.slot_ids.len);
        for (f.score_vals) |*v| v.* = 0.05 + rnd.float(f32) * 0.3;
        f.x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(spec.rows), @intCast(spec.hidden) }, 2, .float16);
        f.slots = mlx.mlx_array_new_data(f.slot_ids.ptr, &[_]c_int{@intCast(f.slot_ids.len)}, 1, .uint32);
        f.scores = mlx.mlx_array_new_data(f.score_vals.ptr, &[_]c_int{@intCast(f.score_vals.len)}, 1, .float32);
        return f;
    }

    fn deinit(self: *Fixture) void {
        self.rb.deinit();
        for (self.banks[0..self.spec.rates.len]) |*b| {
            for ([_]*root.Proj{ &b.gate, &b.up, &b.down }) |p| {
                _ = mlx.mlx_array_free(p.trellis);
                _ = mlx.mlx_array_free(p.suh);
                _ = mlx.mlx_array_free(p.svh);
            }
        }
        for ([_]mlx.mlx_array{ self.x, self.slots, self.scores }) |a| _ = mlx.mlx_array_free(a);
    }

    fn topk(self: *const Fixture) c_int {
        return @intCast(self.spec.topk);
    }
};

fn bytesOf(s: mlx.mlx_stream, a: mlx.mlx_array, out: *mlx.mlx_array) ![]const u8 {
    try mlx.check(mlx.mlx_contiguous(out, a, false, s));
    try mlx.check(mlx.mlx_array_eval(out.*));
    const p: [*]const u8 = @ptrCast(mlx.mlx_array_data_uint8(out.*) orelse return error.Unreadable);
    return p[0 .. mlx.mlx_array_size(out.*) * mlx.mlx_array_itemsize(out.*)];
}

/// Each slot's down inner plane from the served pair GEMV and fused down on its own group's bank, one
/// slot per call, assembled in slot order, then the served finish reduce over the concatenated svh.
fn servedComposition(s: mlx.mlx_stream, f: *const Fixture, out_dtype: mlx.mlx_dtype) !struct { inner: mlx.mlx_array, y: mlx.mlx_array } {
    const hidden: c_int = @intCast(f.spec.hidden);
    const inter: c_int = @intCast(f.spec.inter);
    var parts: [256]mlx.mlx_array = undefined;
    var made: usize = 0;
    defer for (parts[0..made]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (f.slot_ids, 0..) |e, i| {
        const g = f.rb.groupOf(e);
        const b = f.banks[g];
        const local: u32 = e - f.rb.base[g];
        const one = mlx.mlx_array_new_data(&local, &[_]c_int{1}, 1, .uint32);
        defer _ = mlx.mlx_array_free(one);
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        const r: c_int = @intCast(i / f.spec.topk);
        try mlx.check(mlx.mlx_slice(&row, f.x, &[_]c_int{ r, 0 }, 2, &[_]c_int{ r + 1, hidden }, 2, &[_]c_int{ 1, 1 }, 2, s));
        const pair = if (mlx.getShape(b.gate.trellis)[3] == mlx.getShape(b.up.trellis)[3])
            try kernels.pairGemv(s, row, b.gate.suh, b.up.suh, b.gate.trellis, b.up.trellis, one, hidden, inter, 1, 1, 0)
        else blk: {
            const gp = try kernels.pairGemv(s, row, b.gate.suh, b.gate.suh, b.gate.trellis, b.gate.trellis, one, hidden, inter, 1, 1, 0);
            defer _ = mlx.mlx_array_free(gp[1]);
            const up = try kernels.pairGemv(s, row, b.up.suh, b.up.suh, b.up.trellis, b.up.trellis, one, hidden, inter, 1, 1, 0);
            defer _ = mlx.mlx_array_free(up[1]);
            break :blk .{ gp[0], up[0] };
        };
        defer _ = mlx.mlx_array_free(pair[0]);
        defer _ = mlx.mlx_array_free(pair[1]);
        parts[made] = try kernels.downGemvFusedMid(s, pair[0], pair[1], b.down.trellis, b.gate.svh, b.up.svh, b.down.suh, one, inter, hidden, 1);
        made += 1;
    }
    const inner = try concat(s, parts[0..made]);
    errdefer _ = mlx.mlx_array_free(inner);
    const y = try kernels.downFinishReduce(s, inner, f.rb.svh[2], f.slots, f.scores, hidden, @intCast(f.spec.rows), f.topk(), out_dtype);
    return .{ .inner = inner, .y = y };
}

/// The routed SwiGLU in f32 from the host tile decode, each expert at its own rates.
fn hostTruth(alloc: std.mem.Allocator, f: *const Fixture, dec: exl3.Decode) ![]f32 {
    return hostTruthClamped(alloc, f, dec, 0);
}

fn hostTruthClamped(alloc: std.mem.Allocator, f: *const Fixture, dec: exl3.Decode, limit: f32) ![]f32 {
    const h = f.spec.hidden;
    const i_d = f.spec.inter;
    const out = try alloc.alloc(f32, f.spec.rows * h);
    @memset(out, 0);
    const scratch = try alloc.alloc(f32, @max(h, i_d));
    const gy = try alloc.alloc(f32, i_d);
    const uy = try alloc.alloc(f32, i_d);
    const dy = try alloc.alloc(f32, h);
    const proj = struct {
        fn run(src: []const f32, tr: []const u16, suh: []const u16, svh: []const u16, in_f: usize, out_f: usize, n: u32, d: exl3.Decode, sc: []f32, dst: []f32) void {
            for (src, suh, sc[0..in_f]) |v, sb, *o| o.* = v * exl3.f16BitsToF32(sb);
            var b: usize = 0;
            while (b < in_f) : (b += exl3.HAD_DIM) exl3.hadamard128(sc[b..][0..exl3.HAD_DIM]);
            @memset(dst, 0);
            var tile: [exl3.TILE_VALUES]u16 = undefined;
            const ot = out_f / 16;
            for (0..in_f / 16) |tk| for (0..ot) |tn| {
                exl3.decodeTile(tr[(tk * ot + tn) * n ..][0..n], .{ .n = n }, d, &tile);
                for (0..16) |r| for (0..16) |c| {
                    dst[tn * 16 + c] += sc[tk * 16 + r] * exl3.f16BitsToF32(tile[r * 16 + c]);
                };
            };
            var o: usize = 0;
            while (o < out_f) : (o += exl3.HAD_DIM) exl3.hadamard128(dst[o..][0..exl3.HAD_DIM]);
            for (dst, svh) |*v, sb| v.* *= exl3.f16BitsToF32(sb);
        }
    }.run;
    const tiles = (h / 16) * (i_d / 16);
    for (f.slot_ids, f.score_vals, 0..) |e, w, si| {
        const g = f.rb.groupOf(e);
        const l = e - f.rb.base[g];
        const r = f.spec.rates[g];
        const x = f.xf[(si / f.spec.topk) * h ..][0..h];
        proj(x, f.words[g][0][l * tiles * r[0] ..][0 .. tiles * r[0]], f.suh[g][0][l * h ..][0..h], f.svh[g][0][l * i_d ..][0..i_d], h, i_d, r[0], dec, scratch, gy);
        proj(x, f.words[g][1][l * tiles * r[1] ..][0 .. tiles * r[1]], f.suh[g][1][l * h ..][0..h], f.svh[g][1][l * i_d ..][0..i_d], h, i_d, r[1], dec, scratch, uy);
        for (gy, uy) |*gv, u_raw| {
            const gc = if (limit > 0) @min(gv.*, limit) else gv.*;
            const u = if (limit > 0) std.math.clamp(u_raw, -limit, limit) else u_raw;
            gv.* = (gc / (1.0 + @exp(-gc))) * u;
        }
        proj(gy, f.words[g][2][l * tiles * r[2] ..][0 .. tiles * r[2]], f.suh[g][2][l * i_d ..][0..i_d], f.svh[g][2][l * h ..][0..h], i_d, h, r[2], dec, scratch, dy);
        for (out[(si / f.spec.topk) * h ..][0..h], dy) |*o, v| o.* += w * v;
    }
    return out;
}

fn relRms(got: []const f32, want: []const f32) f64 {
    var ss: f64 = 0;
    var ref: f64 = 0;
    for (got, want) |a, b| {
        ss += (@as(f64, a) - b) * (@as(f64, a) - b);
        ref += @as(f64, b) * b;
    }
    return @sqrt(ss / @max(ref, 1e-30));
}

fn toF32(s: mlx.mlx_stream, a: mlx.mlx_array, out: *mlx.mlx_array) ![]const f32 {
    try mlx.check(mlx.mlx_astype(out, a, .float32, s));
    try mlx.check(mlx.mlx_array_eval(out.*));
    const p = mlx.mlx_array_data_float32(out.*) orelse return error.Unreadable;
    return p[0..mlx.mlx_array_size(out.*)];
}

/// The served codebooks at their pack windows; PAIR1's rates are its rows.
const CODEBOOKS = [_]exl3.Decode{
    .{ .codebook = .mcg, .window = .w16 },
    exl3.Decode.tabled(.spec1, .w15),
    exl3.Decode.tabled(.std1, .w15),
    .{ .codebook = .pair1, .window = .w16 },
    // A SPEC1 table with a different pair at every rate, at a window other than its own: each group decodes its row.
    .{ .codebook = .spec1, .window = .w9, .table = codebook_tables.indexOf("spec1-test-per-rate") },
};

/// One group per even n in lo..hi (two experts each), gate = up = down.
fn everyRate(buf: *[max_groups][3]u32, lo: u32, hi: u32) [][3]u32 {
    var k: usize = 0;
    var n = lo;
    while (n <= hi) : (n += 2) {
        buf[k] = .{ n, n, n };
        k += 1;
    }
    return buf[0..k];
}

test "exl3 rate groups: every slot writes the served kernels' bytes at its own rate" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    for (CODEBOOKS) |dec| {
        kernels.setDecodeParams(dec);
        defer kernels.setDecodeParams(.mul1);
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        var buf: [max_groups][3]u32 = undefined;
        const rates = everyRate(&buf, 24, 64);
        const experts: [max_groups]u32 = @splat(2);
        // Two rows, each taking one expert of every group: 42 slots, every rate on both rows.
        var slots: [2 * 21]u32 = undefined;
        for (0..21) |g| {
            slots[g] = @intCast(2 * g + g % 2);
            slots[21 + g] = @intCast(2 * (20 - g) + 1 - (20 - g) % 2);
        }
        var f = try Fixture.init(arena.allocator(), s, .{ .rates = rates, .experts = experts[0..rates.len], .rows = 2, .topk = 21, .slots = &slots });
        defer f.deinit();
        const want = try servedComposition(s, &f, .float16);
        defer _ = mlx.mlx_array_free(want.inner);
        defer _ = mlx.mlx_array_free(want.y);
        var cw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cw);
        const want_bytes = try bytesOf(s, want.y, &cw);
        const y = try moeDecode(s, f.x, &f.rb, f.slots, f.scores, f.topk(), .float16, null, null);
        defer _ = mlx.mlx_array_free(y);
        var cy = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cy);
        try t.expectEqualSlices(u8, want_bytes, try bytesOf(s, y, &cy));
    }
}

test "exl3 rate groups: a slot's planes keep their bytes at the rates outside K1.5 to K4 and with gate and up apart" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    kernels.setDecodeParams(exl3.Decode.tabled(.spec1, .w15));
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const rates = [_][3]u32{ .{ 16, 16, 128 }, .{ 128, 128, 16 }, .{ 24, 48, 36 }, .{ 64, 40, 64 }, .{ 18, 120, 70 } };
    const experts = [_]u32{ 2, 1, 3, 2, 1 };
    var f = try Fixture.init(arena.allocator(), s, .{ .rates = &rates, .experts = &experts, .rows = 3, .topk = 5 });
    defer f.deinit();
    const want = try servedComposition(s, &f, .float32);
    defer _ = mlx.mlx_array_free(want.inner);
    defer _ = mlx.mlx_array_free(want.y);
    var cw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cw);
    const want_bytes = try bytesOf(s, want.y, &cw);
    const y = try moeDecode(s, f.x, &f.rb, f.slots, f.scores, f.topk(), .float32, null, null);
    defer _ = mlx.mlx_array_free(y);
    var cy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cy);
    try t.expectEqualSlices(u8, want_bytes, try bytesOf(s, y, &cy));
}

test "exl3 rate groups: decode rows hold the f32-truth bar for every codebook" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    for (CODEBOOKS) |dec| {
        kernels.setDecodeParams(dec);
        defer kernels.setDecodeParams(.mul1);
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const alloc = arena.allocator();
        const rates = [_][3]u32{ .{ 24, 24, 24 }, .{ 24, 24, 48 }, .{ 48, 48, 48 } };
        const experts = [_]u32{ 6, 3, 7 };
        var f = try Fixture.init(alloc, s, .{ .hidden = 512, .inter = 256, .rates = &rates, .experts = &experts, .rows = 4, .topk = 8, .banks = .{ 0.0126, 1.03, 0.0083, 1.009 }, .x_scale = 40 });
        defer f.deinit();
        const want = try hostTruth(alloc, &f, dec);
        const y = try moeDecode(s, f.x, &f.rb, f.slots, f.scores, f.topk(), .float32, null, null);
        defer _ = mlx.mlx_array_free(y);
        var cy = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cy);
        const got = try toF32(s, y, &cy);
        for (got) |v| try t.expect(std.math.isFinite(v));
        try t.expect(relRms(got, want) < 0.01);
    }
}

test "exl3 rate groups: a clamped SwiGLU holds the f32-truth bar at decode and prefill where the clamp is live" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const limit: c_int = 10;
    for ([_]exl3.Decode{ .{ .codebook = .pair1, .window = .w16 }, exl3.Decode.tabled(.spec1, .w15) }) |dec| {
        kernels.setDecodeParams(dec);
        defer kernels.setDecodeParams(.mul1);
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const alloc = arena.allocator();
        const rates = [_][3]u32{ .{ 24, 24, 24 }, .{ 32, 32, 40 }, .{ 48, 48, 64 } };
        const experts = [_]u32{ 5, 3, 4 };
        for ([_]usize{ 4, 40 }) |rows| {
            var f = try Fixture.init(alloc, s, .{ .hidden = 512, .inter = 256, .rates = &rates, .experts = &experts, .rows = rows, .topk = 6, .banks = .{ 0.0126, 1.03, 0.0083, 1.009 }, .x_scale = 400 });
            defer f.deinit();
            const want = try hostTruthClamped(alloc, &f, dec, @floatFromInt(limit));
            // The clamp moves the answer at these magnitudes.
            try t.expect(relRms(try hostTruth(alloc, &f, dec), want) > 0.1);
            const y = if (rows <= kernels.DECODE_ROWS_MAX)
                try moeDecodeClamped(s, f.x, &f.rb, f.slots, f.scores, f.topk(), .float32, null, null, limit)
            else
                try moePrefillClamped(s, f.x, &f.rb, f.slots, f.scores, f.topk(), limit);
            defer _ = mlx.mlx_array_free(y);
            var cy = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cy);
            const got = try toF32(s, y, &cy);
            for (got) |v| try t.expect(std.math.isFinite(v));
            try t.expect(relRms(got, want) < 0.01);
        }
    }
}

test "exl3 rate groups: rows through the prepared pair input and middle write the self-preparing chain's bytes, 1 to 16 rows" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const limit: c_int = 10;
    for ([_]exl3.Decode{ .{ .codebook = .pair1, .window = .w16 }, exl3.Decode.tabled(.spec1, .w15) }) |dec| {
        kernels.setDecodeParams(dec);
        defer kernels.setDecodeParams(.mul1);
        defer prepared_force = null;
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const alloc = arena.allocator();
        const rates = [_][3]u32{ .{ 24, 24, 24 }, .{ 32, 32, 40 }, .{ 48, 48, 64 }, .{ 32, 26, 24 } };
        const experts = [_]u32{ 5, 3, 4, 2 };
        for ([_]usize{ 1, 2, 3, 4, 5, 6, 9, 16 }) |rows| {
            var f = try Fixture.init(alloc, s, .{ .hidden = 512, .inter = 256, .rates = &rates, .experts = &experts, .rows = rows, .topk = 6, .banks = .{ 0.0126, 1.03, 0.0083, 1.009 }, .x_scale = 400 });
            defer f.deinit();
            prepared_force = false;
            const self_prep = try moeDecodeClamped(s, f.x, &f.rb, f.slots, f.scores, f.topk(), .float32, null, null, limit);
            defer _ = mlx.mlx_array_free(self_prep);
            var c0 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(c0);
            const want = try bytesOf(s, self_prep, &c0);
            prepared_force = true;
            prepared_engaged = false;
            const prep = try moeDecodeClamped(s, f.x, &f.rb, f.slots, f.scores, f.topk(), .float32, null, null, limit);
            defer _ = mlx.mlx_array_free(prep);
            var c1 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(c1);
            try t.expect(prepared_engaged);
            try t.expectEqualSlices(u8, want, try bytesOf(s, prep, &c1));
        }
    }
}

/// Verify routes over 56 experts in four groups: random, every row on one set, no two rows sharing, and a common
/// core plus a private pick per row.
fn plantedPicks(alloc: std.mem.Allocator, kind: usize, rows: usize, topk: usize) !?[]u32 {
    if (kind == 0) return null;
    const core = [_]u32{ 1, 22, 35, 50, 19, 47 };
    const out = try alloc.alloc(u32, rows * topk);
    for (0..rows) |r| for (0..topk) |i| {
        out[r * topk + i] = switch (kind) {
            1 => core[i],
            2 => @intCast((r * topk + i) % 56),
            else => if (i < topk - 1) core[i] else @intCast(2 + (r * 3) % 17 + (r % 2) * 20),
        };
    };
    return out;
}

test "exl3 rate groups: verify rows grouped by expert write the per-slot chain's bytes, 2 to 16 rows, every bucket" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const limit: c_int = 10;
    for ([_]exl3.Decode{ .{ .codebook = .pair1, .window = .w16 }, exl3.Decode.tabled(.spec1, .w15) }) |dec| {
        kernels.setDecodeParams(dec);
        defer kernels.setDecodeParams(.mul1);
        defer prepared_force = null;
        defer group_force = null;
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const alloc = arena.allocator();
        const rates = [_][3]u32{ .{ 24, 24, 24 }, .{ 32, 32, 40 }, .{ 48, 48, 64 }, .{ 32, 26, 24 } };
        const experts = [_]u32{ 20, 12, 16, 8 };
        prepared_force = true;
        for ([_]usize{ 2, 3, 4, 5, 8, 9, 16 }) |rows| {
            for (0..4) |kind| {
                var f = try Fixture.init(alloc, s, .{ .hidden = 512, .inter = 256, .rates = &rates, .experts = &experts, .rows = rows, .topk = 6, .slots = try plantedPicks(alloc, kind, rows, 6), .banks = .{ 0.0126, 1.03, 0.0083, 1.009 }, .x_scale = 400 });
                defer f.deinit();
                group_force = 0;
                const one = try moeDecodeClamped(s, f.x, &f.rb, f.slots, f.scores, f.topk(), .float32, null, null, limit);
                defer _ = mlx.mlx_array_free(one);
                var c0 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(c0);
                const want = try bytesOf(s, one, &c0);
                for ([_]?c_int{ null, 2, 4, 8, 16 }) |arm| {
                    group_force = arm;
                    kernels.grouped_widest = 0;
                    const y = try moeDecodeClamped(s, f.x, &f.rb, f.slots, f.scores, f.topk(), .float32, null, null, limit);
                    defer _ = mlx.mlx_array_free(y);
                    var c1 = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(c1);
                    try t.expectEqual(arm orelse kernels.GROUP_BUCKETS[0], kernels.grouped_widest);
                    try t.expectEqualSlices(u8, want, try bytesOf(s, y, &c1));
                }
            }
        }
    }
}

test "exl3 rate groups: tile-major tiles write the pack order's chain bytes, 1 to 16 rows, self-preparing, prepared and grouped" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const limit: c_int = 10;
    for ([_]exl3.Decode{ .{ .codebook = .pair1, .window = .w16 }, exl3.Decode.tabled(.spec1, .w15) }) |dec| {
        kernels.setDecodeParams(dec);
        defer kernels.setDecodeParams(.mul1);
        defer prepared_force = null;
        defer group_force = null;
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const alloc = arena.allocator();
        const rates = [_][3]u32{ .{ 24, 24, 24 }, .{ 32, 32, 40 }, .{ 48, 48, 64 }, .{ 32, 26, 24 } };
        const experts = [_]u32{ 20, 12, 16, 8 };
        for ([_]usize{ 1, 2, 3, 4, 5, 6, 9, 16 }) |rows| {
            var f = try Fixture.init(alloc, s, .{ .hidden = 512, .inter = 256, .rates = &rates, .experts = &experts, .rows = rows, .topk = 6, .banks = .{ 0.0126, 1.03, 0.0083, 1.009 }, .x_scale = 400 });
            defer f.deinit();
            var tm = try tileMajorCopy(s, &f.rb);
            defer tm.deinit();
            for ([_]bool{ false, true }) |prepared| {
                prepared_force = prepared;
                for ([_]?c_int{ 0, null, 4, 16 }) |arm| {
                    group_force = arm;
                    const want = try moeDecodeClamped(s, f.x, &f.rb, f.slots, f.scores, f.topk(), .float32, null, null, limit);
                    defer _ = mlx.mlx_array_free(want);
                    var c0 = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(c0);
                    const wb = try bytesOf(s, want, &c0);
                    const got = try moeDecodeClamped(s, f.x, &tm, f.slots, f.scores, f.topk(), .float32, null, null, limit);
                    defer _ = mlx.mlx_array_free(got);
                    var c1 = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(c1);
                    try t.expectEqualSlices(u8, wb, try bytesOf(s, got, &c1));
                }
            }
        }
    }
}

test "exl3 rate groups: tile-major tiles write the pack order's prefill bytes, 17 to 256 rows" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const limit: c_int = 10;
    for ([_]exl3.Decode{ .{ .codebook = .pair1, .window = .w16 }, exl3.Decode.tabled(.spec1, .w15) }) |dec| {
        kernels.setDecodeParams(dec);
        defer kernels.setDecodeParams(.mul1);
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const alloc = arena.allocator();
        const rates = [_][3]u32{ .{ 24, 24, 24 }, .{ 32, 32, 40 }, .{ 48, 48, 64 }, .{ 32, 26, 24 } };
        const experts = [_]u32{ 20, 12, 16, 8 };
        for ([_]usize{ 17, 64, 256 }) |rows| {
            var f = try Fixture.init(alloc, s, .{ .hidden = 512, .inter = 256, .rates = &rates, .experts = &experts, .rows = rows, .topk = 6, .banks = .{ 0.0126, 1.03, 0.0083, 1.009 }, .x_scale = 400 });
            defer f.deinit();
            var tm = try tileMajorCopy(s, &f.rb);
            defer tm.deinit();
            const want = try moePrefillClamped(s, f.x, &f.rb, f.slots, f.scores, f.topk(), limit);
            defer _ = mlx.mlx_array_free(want);
            var c0 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(c0);
            const wb = try bytesOf(s, want, &c0);
            const got = try moePrefillClamped(s, f.x, &tm, f.slots, f.scores, f.topk(), limit);
            defer _ = mlx.mlx_array_free(got);
            var c1 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(c1);
            try t.expectEqualSlices(u8, wb, try bytesOf(s, got, &c1));
        }
    }
}

fn reshaped(s: mlx.mlx_stream, a: mlx.mlx_array, shape: []const c_int) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, a, shape.ptr, shape.len, s));
    return out;
}

fn cast(s: mlx.mlx_stream, a: mlx.mlx_array, dtype: mlx.mlx_dtype) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, a, dtype, s));
    return out;
}

/// One expert's bank at SPEC1's n64, scaled so a bf16 chain stays finite at the fixture's activations.
fn sharedBankAt(rnd: std.Random, hidden: usize, inter: usize, n: u32, mags: [4]f32) !root.Bank {
    var bank: root.Bank = undefined;
    const alloc = std.heap.page_allocator;
    for (0..3) |p| {
        const in_d: c_int = @intCast(if (p == 2) inter else hidden);
        const out_d: c_int = @intCast(if (p == 2) hidden else inter);
        const shape = [_]c_int{ 1, @divExact(in_d, 16), @divExact(out_d, 16), @intCast(n) };
        const words = try alloc.alloc(u16, @as(usize, @intCast(shape[1] * shape[2])) * n);
        defer alloc.free(words);
        randomWords(rnd, words);
        const su = try alloc.alloc(u16, @intCast(in_d));
        defer alloc.free(su);
        const sv = try alloc.alloc(u16, @intCast(out_d));
        defer alloc.free(sv);
        const m = if (p == 2) [2]f32{ mags[2], mags[3] } else [2]f32{ mags[0], mags[1] };
        for (su) |*v| v.* = exl3.f32ToF16Bits(if (rnd.boolean()) m[0] else -m[0]);
        for (sv) |*v| v.* = exl3.f32ToF16Bits(if (rnd.boolean()) m[1] else -m[1]);
        projPtr(&bank, p).* = .{
            .trellis = mlx.mlx_array_new_data(words.ptr, &shape, 4, .uint16),
            .suh = mlx.mlx_array_new_data(su.ptr, &[_]c_int{ 1, in_d }, 2, .float16),
            .svh = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{ 1, out_d }, 2, .float16),
        };
    }
    return bank;
}

/// The shared chain as `sharedFused` runs it (one expert of the lane chain, bf16), then the routed chain folding it
/// into its f32 reduce: the two-step path the fused launches replace.
fn separateShared(s: mlx.mlx_stream, xb: mlx.mlx_array, rb: *const Ragged, slots: mlx.mlx_array, scores: mlx.mlx_array, topk: c_int, se: SharedExpert, limit: c_int, dec: exl3.Decode) !mlx.mlx_array {
    const sh = mlx.getShape(xb);
    const rows = sh[0];
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    try mlx.check(mlx.mlx_reshape(&x3, xb, &[_]c_int{ 1, rows, sh[1] }, 3, s));
    var zi: [16]i32 = @splat(0);
    var one: [16]f32 = @splat(1);
    const zeros = mlx.mlx_array_new_data(&zi, &[_]c_int{ 1, rows, 1 }, 3, .int32);
    defer _ = mlx.mlx_array_free(zeros);
    const ones = mlx.mlx_array_new_data(&one, &[_]c_int{ 1, rows, 1 }, 3, .float32);
    defer _ = mlx.mlx_array_free(ones);
    const sy3 = try root.moeClamped(s, x3, se.bank, zeros, ones, se.dec, limit);
    defer _ = mlx.mlx_array_free(sy3);
    var sy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sy);
    try mlx.check(mlx.mlx_reshape(&sy, sy3, &[_]c_int{ rows, sh[1] }, 2, s));
    kernels.setDecodeParams(dec);
    return moeDecodeClamped(s, xb, rb, slots, scores, topk, .float32, sy, null, limit);
}

fn sharedFusedCase(t: std.mem.Allocator, s: mlx.mlx_stream, hidden: usize, inter: usize, rates: []const [3]u32, experts: []const u32, rows: usize, topk: usize, seed: u64) !void {
    const limit: c_int = 10;
    const dec: exl3.Decode = .{ .codebook = .pair1, .window = .w16 };
    const sdec = exl3.Decode.tabled(.spec1, .w15);
    var arena = std.heap.ArenaAllocator.init(t);
    defer arena.deinit();
    var f = try Fixture.init(arena.allocator(), s, .{ .hidden = hidden, .inter = inter, .rates = rates, .experts = experts, .rows = rows, .topk = topk, .seed = seed, .banks = .{ 0.0126, 1.03, 0.0083, 1.009 }, .x_scale = 400 });
    defer f.deinit();
    var prng = std.Random.DefaultPrng.init(seed ^ 0x5a5a);
    var sb = try sharedBankAt(prng.random(), hidden, inter, 64, .{ 0.0126, 1.03, 0.0083, 1.009 });
    defer freeBank(&sb);
    const se: SharedExpert = .{ .bank = sb, .dec = sdec };
    const xb = try cast(s, f.x, .bfloat16);
    defer _ = mlx.mlx_array_free(xb);
    kernels.setDecodeParams(sdec);
    const want = try separateShared(s, xb, &f.rb, f.slots, f.scores, f.topk(), se, limit, dec);
    defer _ = mlx.mlx_array_free(want);
    var cw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cw);
    const want_bytes = try bytesOf(s, want, &cw);
    // A chain of zeros or NaNs would equal itself.
    var live: usize = 0;
    for (std.mem.bytesAsSlice(f32, want_bytes)) |v| {
        try std.testing.expect(std.math.isFinite(v));
        if (v != 0) live += 1;
    }
    try std.testing.expect(live > want_bytes.len / 8);
    for ([_]bool{ false, true }) |first| {
        fused_shared_first = first;
        kernels.setDecodeParams(dec);
        const before = kernels.fusedDispatchCount();
        const got = (try moeDecodeClampedSharedFused(s, xb, &f.rb, f.slots, f.scores, f.topk(), .float32, se, limit)) orelse return error.TestUnexpectedResult;
        defer _ = mlx.mlx_array_free(got);
        var cg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cg);
        const got_bytes = try bytesOf(s, got, &cg);
        // The shared expert adds no dispatch: prepare, pair, mid, down and reduce are the routed chain's five.
        try std.testing.expectEqual(before + 5, kernels.fusedDispatchCount());
        try std.testing.expectEqualSlices(u8, want_bytes, got_bytes);
    }
}

test "exl3 rate groups: the shared expert riding the decode launches writes the shared chain folded into the reduce, 1 to 16 rows" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    defer {
        prepared_force = null;
        fused_shared_first = false;
    }
    prepared_force = true;
    const rates = [_][3]u32{ .{ 24, 24, 24 }, .{ 32, 32, 40 }, .{ 40, 40, 40 }, .{ 32, 26, 24 } };
    const experts = [_]u32{ 5, 3, 4, 2 };
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16 }) |rows| try sharedFusedCase(t.allocator, s, 512, 256, &rates, &experts, rows, 6, 0x51 + rows);
}

test "exl3 rate groups: the shared expert riding a DSpark stage's one-group bank, top-3, 1 to 6 rows" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    defer {
        prepared_force = null;
        fused_shared_first = false;
    }
    prepared_force = true;
    const rates = [_][3]u32{.{ 32, 32, 32 }};
    const experts = [_]u32{16};
    for ([_]usize{ 1, 2, 3, 5, 6 }) |rows| try sharedFusedCase(t.allocator, s, 512, 256, &rates, &experts, rows, 3, 0x77 + rows);
}

test "exl3 rate groups: the shared expert riding the decode launches at DeepSeek's geometry, 1 to 6 rows" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    defer {
        prepared_force = null;
        fused_shared_first = false;
    }
    const rates = [_][3]u32{ .{ 24, 24, 24 }, .{ 40, 40, 40 }, .{ 32, 32, 26 } };
    const experts = [_]u32{ 3, 2, 2 };
    for ([_]usize{ 1, 2, 4, 6 }) |rows| try sharedFusedCase(t.allocator, s, 5120, 2304, &rates, &experts, rows, 6, 0xd5 + rows);
}

test "exl3 rate groups: groups sharing one rate equal the uniform layer to the byte at n40 and K4, shared tail folded" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    for (CODEBOOKS) |dec| for ([_]u32{ 40, 64 }) |n| {
        kernels.setDecodeParams(dec);
        defer kernels.setDecodeParams(.mul1);
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const rates: [3][3]u32 = @splat(@splat(n));
        const experts = [_]u32{ 3, 5, 4 };
        var f = try Fixture.init(arena.allocator(), s, .{ .hidden = 512, .inter = 256, .rates = &rates, .experts = &experts, .rows = 3, .topk = 6 });
        defer f.deinit();
        var uniform: root.Bank = undefined;
        var held: [9]mlx.mlx_array = undefined;
        var n_held: usize = 0;
        defer for (held[0..n_held]) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for (0..3) |p| {
            var parts: [3][3]mlx.mlx_array = undefined;
            for (0..3) |g| {
                const q = projOf(f.banks[g], p);
                parts[0][g] = q.trellis;
                parts[1][g] = q.suh;
                parts[2][g] = q.svh;
            }
            for (0..3) |part| {
                held[n_held] = try concat(s, &parts[part]);
                n_held += 1;
            }
            projPtr(&uniform, p).* = .{ .trellis = held[n_held - 3], .suh = held[n_held - 2], .svh = held[n_held - 1] };
        }
        const xb = try cast(s, f.x, .bfloat16);
        defer _ = mlx.mlx_array_free(xb);
        const x3 = try reshaped(s, xb, &.{ 3, 1, 512 });
        defer _ = mlx.mlx_array_free(x3);
        const ids = try reshaped(s, f.slots, &.{ 3, 1, 6 });
        defer _ = mlx.mlx_array_free(ids);
        const sc = try reshaped(s, f.scores, &.{ 3, 1, 6 });
        defer _ = mlx.mlx_array_free(sc);
        const want = try root.moe(s, x3, uniform, ids, sc, dec, true);
        defer _ = mlx.mlx_array_free(want);
        var cw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cw);
        const want_bytes = try bytesOf(s, want, &cw);
        const y = try moeDecode(s, xb, &f.rb, f.slots, f.scores, 6, .bfloat16, null, null);
        defer _ = mlx.mlx_array_free(y);
        var cy = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cy);
        try t.expectEqualSlices(u8, want_bytes, try bytesOf(s, y, &cy));
        // One decode row with the gated bf16 shared expert in the reduce.
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_slice(&row, xb, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ 1, 512 }, 2, &[_]c_int{ 1, 1 }, 2, s));
        const row3 = try reshaped(s, row, &.{ 1, 1, 512 });
        defer _ = mlx.mlx_array_free(row3);
        var one_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(one_ids);
        try mlx.check(mlx.mlx_slice(&one_ids, f.slots, &[_]c_int{0}, 1, &[_]c_int{6}, 1, &[_]c_int{1}, 1, s));
        var one_sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(one_sc);
        try mlx.check(mlx.mlx_slice(&one_sc, f.scores, &[_]c_int{0}, 1, &[_]c_int{6}, 1, &[_]c_int{1}, 1, s));
        const ids1 = try reshaped(s, one_ids, &.{ 1, 1, 6 });
        defer _ = mlx.mlx_array_free(ids1);
        const sc1 = try reshaped(s, one_sc, &.{ 1, 1, 6 });
        defer _ = mlx.mlx_array_free(sc1);
        const down = try reshaped(s, row, &.{512});
        defer _ = mlx.mlx_array_free(down);
        const lg: u16 = 0xbf00;
        const logit = mlx.mlx_array_new_data(&lg, &[_]c_int{1}, 1, .bfloat16);
        defer _ = mlx.mlx_array_free(logit);
        const gw = try root.moeWithGatedShared(s, row3, uniform, ids1, sc1, dec, false, down, logit);
        defer _ = mlx.mlx_array_free(gw);
        var cg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cg);
        const gw_bytes = try bytesOf(s, gw, &cg);
        const y1 = try moeDecode(s, row, &f.rb, one_ids, one_sc, 6, .bfloat16, down, logit);
        defer _ = mlx.mlx_array_free(y1);
        var cy1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cy1);
        try t.expectEqualSlices(u8, gw_bytes, try bytesOf(s, y1, &cy1));
    };
}

test "exl3 rate groups: a shared expert folded into the clamped f32 reduce is the separate f32 add, to the byte" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const dec: exl3.Decode = .{ .codebook = .pair1, .window = .w16 };
    kernels.setDecodeParams(dec);
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const rates = [_][3]u32{ @splat(24), @splat(32), .{ 32, 32, 26 } };
    const experts = [_]u32{ 5, 3, 4 };
    for ([_]c_int{ 1, 5 }) |rows| {
        var f = try Fixture.init(arena.allocator(), s, .{ .hidden = 512, .inter = 256, .rates = &rates, .experts = &experts, .rows = @intCast(rows), .topk = 6 });
        defer f.deinit();
        const xb = try cast(s, f.x, .bfloat16);
        defer _ = mlx.mlx_array_free(xb);
        // A bf16 shared output at the magnitudes of the routed sum.
        const shared = try cast(s, f.x, .bfloat16);
        defer _ = mlx.mlx_array_free(shared);
        const routed = try moeDecodeClamped(s, xb, &f.rb, f.slots, f.scores, 6, .float32, null, null, 10);
        defer _ = mlx.mlx_array_free(routed);
        var sf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sf);
        try mlx.check(mlx.mlx_astype(&sf, shared, .float32, s));
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, routed, sf, s));
        var cw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cw);
        const want = try bytesOf(s, sum, &cw);
        const folded = try moeDecodeClamped(s, xb, &f.rb, f.slots, f.scores, 6, .float32, shared, null, 10);
        defer _ = mlx.mlx_array_free(folded);
        var cg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cg);
        try t.expectEqualSlices(u8, want, try bytesOf(s, folded, &cg));
    }
}

test "exl3 rate groups: decode rows agree with today's masked moeGroups to f32 summation order" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const dec = exl3.Decode.tabled(.spec1, .w15);
    kernels.setDecodeParams(dec);
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const rates = [_][3]u32{ .{ 24, 24, 24 }, .{ 24, 24, 48 }, .{ 48, 48, 48 } };
    const experts = [_]u32{ 6, 3, 7 };
    var f = try Fixture.init(arena.allocator(), s, .{ .hidden = 512, .inter = 256, .rates = &rates, .experts = &experts, .rows = 4, .topk = 8 });
    defer f.deinit();
    const x3 = try reshaped(s, f.x, &.{ 4, 1, 512 });
    defer _ = mlx.mlx_array_free(x3);
    const ids = try reshaped(s, f.slots, &.{ 4, 1, 8 });
    defer _ = mlx.mlx_array_free(ids);
    const sc = try reshaped(s, f.scores, &.{ 4, 1, 8 });
    defer _ = mlx.mlx_array_free(sc);
    const today = try root.moeGroups(s, x3, f.banks[0..3], ids, sc, dec, false);
    defer _ = mlx.mlx_array_free(today);
    var ct = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ct);
    const want = try toF32(s, today, &ct);
    const y = try moeDecode(s, f.x, &f.rb, f.slots, f.scores, 8, .float16, null, null);
    defer _ = mlx.mlx_array_free(y);
    var cy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cy);
    const got = try toF32(s, y, &cy);
    // moeGroups sums each group's slots in f32, adds the group totals in f32, then rounds once: the
    // same products in another order, so at most one f16 rounding step apart.
    for (got, want) |a, b| try t.expect(@abs(a - b) <= @abs(b) * (1.0 / 1024.0) + 1e-6);
}

// ---- Timing (a diagnostic, never a test: prints only under SUSHI_EXL3_RG_UBENCH) ----
// One Flash-Next MoE layer (2560 -> 640, E=512, top-10, bf16 rows, the gated shared expert) per step,
// 48 dependent steps per arm, every arm interleaved in one process: uniform n32 as served, today's
// masked `moeGroups` with its separate tail, and the one-dispatch path with the tail folded where the
// uniform chain folds it (one row).

const RG_HIDDEN: c_int = 2560;
const RG_INTER: c_int = 640;
const RG_EXPERTS: u32 = 512;
const RG_TOPK: usize = 10;
const RG_STEPS: usize = 48;
const RG_ROUTES: usize = 16;

var rg_link_kernel: ?mlx.mlx_fast_metal_kernel = null;
const RG_LINK_SOURCE: [:0]const u8 =
    \\const uint i = thread_position_in_grid.x;
    \\if (i >= uint(LEN)) return;
    \\out[i] = a[i];
    \\if (i == 0u && float(d[0]) == 1.5e30f) out[0] = a[1];
;

/// A copy of `x0` that waits on `y`, so the steps run one after another.
fn rgLink(s: mlx.mlx_stream, x0: mlx.mlx_array, y: mlx.mlx_array) !mlx.mlx_array {
    const k = try kernels.getNamedKernel(&rg_link_kernel, "sushi_exl3_rg_link", &.{ "a", "d" }, &.{"out"}, RG_LINK_SOURCE, "");
    const sh = mlx.getShape(x0);
    const len: c_int = @intCast(mlx.mlx_array_size(x0));
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, mlx.mlx_array_dtype(x0)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, len, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LEN", len));
    const ov = try kernels.applyOuts(s, k, &.{ x0, y }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    return getOut(ov, 0);
}

fn randomWords(rnd: std.Random, out: []u16) void {
    const w64: []align(1) u64 = std.mem.bytesAsSlice(u64, std.mem.sliceAsBytes(out[0 .. out.len / 4 * 4]));
    for (w64) |*v| v.* = rnd.int(u64);
    for (out[out.len / 4 * 4 ..]) |*v| v.* = @truncate(rnd.int(u32));
}

fn benchBank(rnd: std.Random, experts: u32, n: [3]u32) !root.Bank {
    return benchBankAt(rnd, experts, n, RG_HIDDEN, RG_INTER);
}

fn benchBankAt(rnd: std.Random, experts: u32, n: [3]u32, hidden: c_int, inter: c_int) !root.Bank {
    var bank: root.Bank = undefined;
    const alloc = std.heap.page_allocator;
    for (0..3) |p| {
        const in_d: c_int = if (p == 2) inter else hidden;
        const out_d: c_int = if (p == 2) hidden else inter;
        const e: c_int = @intCast(experts);
        const shape = [_]c_int{ e, @divExact(in_d, 16), @divExact(out_d, 16), @intCast(n[p]) };
        const words = try alloc.alloc(u16, @as(usize, @intCast(e * shape[1] * shape[2])) * n[p]);
        defer alloc.free(words);
        randomWords(rnd, words);
        const su = try alloc.alloc(u16, @intCast(e * in_d));
        defer alloc.free(su);
        const sv = try alloc.alloc(u16, @intCast(e * out_d));
        defer alloc.free(sv);
        for (su) |*v| v.* = exl3.f32ToF16Bits(if (rnd.boolean()) 0.9 else -0.9);
        for (sv) |*v| v.* = exl3.f32ToF16Bits(if (rnd.boolean()) 0.9 else -0.9);
        projPtr(&bank, p).* = .{
            .trellis = mlx.mlx_array_new_data(words.ptr, &shape, 4, .uint16),
            .suh = mlx.mlx_array_new_data(su.ptr, &[_]c_int{ e, in_d }, 2, .float16),
            .svh = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{ e, out_d }, 2, .float16),
        };
    }
    return bank;
}

fn freeBank(b: *root.Bank) void {
    for ([_]*root.Proj{ &b.gate, &b.up, &b.down }) |p| {
        _ = mlx.mlx_array_free(p.trellis);
        _ = mlx.mlx_array_free(p.suh);
        _ = mlx.mlx_array_free(p.svh);
    }
}

/// A layer layout: each group's (gate, up, down) n, and for routed slot j the groups it draws from.
const BenchLayout = struct {
    name: []const u8,
    rates: []const [3]u32,
    picks: []const []const u8,
};

fn rep(comptime n: usize, comptime v: []const u8) [n][]const u8 {
    return @splat(v);
}

const RG_LAYOUTS = [_]BenchLayout{
    // K1.5|K3, gate_up 5:5, down shared.
    .{ .name = "2 groups {n24,n48} 5:5", .rates = &.{ .{ 24, 24, 24 }, .{ 48, 48, 48 } }, .picks = &(rep(5, &.{0}) ++ rep(5, &.{1})) },
    // K1.5|K3 allocation: gate_up 5:5, down 3:7 as three pair classes.
    .{ .name = "3 pair classes (24,24)x3 (24,48)x2 (48,48)x5", .rates = &.{ .{ 24, 24, 24 }, .{ 24, 24, 48 }, .{ 48, 48, 48 } }, .picks = &(rep(3, &.{0}) ++ rep(2, &.{1}) ++ rep(5, &.{2})) },
    // Per expert: slot j takes group j or its near-rate twin 10+j (the per-expert allocation's mix).
    .{ .name = "20 groups per-expert mix", .rates = &PER_EXPERT, .picks = &.{ &.{ 0, 10 }, &.{ 1, 11 }, &.{ 2, 12 }, &.{ 3, 13 }, &.{ 4, 14 }, &.{ 5, 15 }, &.{ 6, 16 }, &.{ 7, 17 }, &.{ 8, 18 }, &.{ 9, 19 } } },
    // The per-expert mix with its n56-64 slot moved to the floor rates.
    .{ .name = "20 groups per-expert mix, n56-64 slot at n24-26", .rates = &PER_EXPERT, .picks = &.{ &.{ 0, 10 }, &.{ 1, 11 }, &.{ 2, 12 }, &.{ 3, 13 }, &.{ 4, 14 }, &.{ 5, 15 }, &.{ 6, 16 }, &.{ 7, 17 }, &.{ 8, 18 }, &.{ 0, 10 } } },
    // The per-expert kernel's switch alone: the same rate set, every slot in its one n32 group.
    .{ .name = "20 groups, every slot in the n32 group", .rates = &.{
        .{ 24, 24, 24 }, .{ 24, 24, 24 }, .{ 24, 24, 28 }, .{ 28, 28, 32 }, .{ 32, 32, 32 },
        .{ 36, 36, 40 }, .{ 40, 40, 40 }, .{ 48, 48, 48 }, .{ 48, 48, 48 }, .{ 64, 64, 64 },
        .{ 26, 26, 24 }, .{ 24, 24, 26 }, .{ 26, 26, 30 }, .{ 30, 30, 32 }, .{ 32, 32, 34 },
        .{ 36, 36, 44 }, .{ 44, 44, 40 }, .{ 48, 48, 44 }, .{ 44, 44, 56 }, .{ 56, 56, 64 },
    }, .picks = &rep(10, &.{4}) },
    // Twenty-one classes, every even n from 24 to 64 (an eighth-step menu), every slot in the n32 group.
    .{ .name = "21 groups n24..n64, every slot in the n32 group", .rates = &EVERY_N, .picks = &rep(10, &.{4}) },
    // The switch-free twin of both: twenty n32 groups, every slot in group 4.
    .{ .name = "20 groups all n32, every slot in group 4", .rates = &@as([20][3]u32, @splat(.{ 32, 32, 32 })), .picks = &rep(10, &.{4}) },
};

const EVERY_N = blk: {
    var r: [21][3]u32 = undefined;
    for (&r, 0..) |*v, i| v.* = @splat(24 + 2 * i);
    break :blk r;
};

const PER_EXPERT = [_][3]u32{
    .{ 24, 24, 24 }, .{ 24, 24, 24 }, .{ 24, 24, 28 }, .{ 28, 28, 32 }, .{ 32, 32, 36 },
    .{ 36, 36, 40 }, .{ 40, 40, 40 }, .{ 48, 48, 48 }, .{ 48, 48, 48 }, .{ 64, 64, 64 },
    .{ 26, 26, 24 }, .{ 24, 24, 26 }, .{ 26, 26, 30 }, .{ 30, 30, 32 }, .{ 32, 32, 34 },
    .{ 36, 36, 44 }, .{ 44, 44, 40 }, .{ 48, 48, 44 }, .{ 44, 44, 56 }, .{ 56, 56, 64 },
};

const BenchGroups = struct {
    rb: Ragged,
    routes: [RG_ROUTES]mlx.mlx_array,
};

fn benchRoutes(rnd: std.Random, rows: usize, rb: ?*const Ragged, picks: []const []const u8, out: *[RG_ROUTES]mlx.mlx_array) !void {
    for (out) |*r| {
        var ids: [16 * RG_TOPK]u32 = undefined;
        for (0..rows) |row| {
            const ri = ids[row * RG_TOPK ..][0..RG_TOPK];
            var k: usize = 0;
            while (k < RG_TOPK) {
                const pick: u32 = if (rb) |b| blk: {
                    const choices = picks[k];
                    const g = choices[rnd.uintLessThan(usize, choices.len)];
                    break :blk b.base[g] + rnd.uintLessThan(u32, b.experts_of[g]);
                } else rnd.uintLessThan(u32, RG_EXPERTS);
                if (std.mem.indexOfScalar(u32, ri[0..k], pick) != null) continue;
                ri[k] = pick;
                k += 1;
            }
        }
        r.* = mlx.mlx_array_new_data(&ids, &[_]c_int{@intCast(rows * RG_TOPK)}, 1, .uint32);
    }
}

const ArmKind = enum { uniform, today, one };
const BenchArm = struct {
    name: []const u8,
    kind: ArmKind,
    layout: usize = 0,
    times: [9]u64 = undefined,
    err: ?anyerror = null,
};

const BenchCtx = struct {
    s: mlx.mlx_stream,
    rows: usize,
    x0: mlx.mlx_array,
    x3: mlx.mlx_array,
    scores: mlx.mlx_array,
    shared: mlx.mlx_array,
    logit: mlx.mlx_array,
    uniform: root.Bank,
    uniform_routes: [RG_ROUTES]mlx.mlx_array,
    groups: []BenchGroups,
    dec: exl3.Decode,
};

/// sigmoid(logit) * shared + y, the separate tail every path not folding it runs.
fn addTail(s: mlx.mlx_stream, y: mlx.mlx_array, shared: mlx.mlx_array, logit: mlx.mlx_array) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, logit, s));
    var gated = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gated);
    try mlx.check(mlx.mlx_multiply(&gated, sig, shared, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, y, gated, s));
    return out;
}

fn benchStep(c: *const BenchCtx, arm: *const BenchArm, x: mlx.mlx_array, step: usize) !mlx.mlx_array {
    const s = c.s;
    const rows: c_int = @intCast(c.rows);
    const k: c_int = RG_TOPK;
    const one = c.rows == 1;
    switch (arm.kind) {
        .uniform => {
            const ids = try reshaped(s, c.uniform_routes[step % RG_ROUTES], &.{ rows, 1, k });
            defer _ = mlx.mlx_array_free(ids);
            const x3 = try reshaped(s, x, &.{ rows, 1, RG_HIDDEN });
            defer _ = mlx.mlx_array_free(x3);
            if (one) return root.moeWithGatedShared(s, x3, c.uniform, ids, c.scores, c.dec, false, c.shared, c.logit);
            const y = try root.moe(s, x3, c.uniform, ids, c.scores, c.dec, true);
            defer _ = mlx.mlx_array_free(y);
            return addTail(s, y, c.shared, c.logit);
        },
        .today => {
            const g = &c.groups[arm.layout];
            const ids = try reshaped(s, g.routes[step % RG_ROUTES], &.{ rows, 1, k });
            defer _ = mlx.mlx_array_free(ids);
            const x3 = try reshaped(s, x, &.{ rows, 1, RG_HIDDEN });
            defer _ = mlx.mlx_array_free(x3);
            const y = try root.moeGroups(s, x3, g.rb.views[0..g.rb.count], ids, c.scores, c.dec, !one);
            defer _ = mlx.mlx_array_free(y);
            return addTail(s, y, c.shared, c.logit);
        },
        .one => {
            kernels.setDecodeParams(c.dec);
            const g = &c.groups[arm.layout];
            const flat_sc = try reshaped(s, c.scores, &.{rows * k});
            defer _ = mlx.mlx_array_free(flat_sc);
            if (one) {
                const sh = try reshaped(s, c.shared, &.{RG_HIDDEN});
                defer _ = mlx.mlx_array_free(sh);
                const lg = try reshaped(s, c.logit, &.{1});
                defer _ = mlx.mlx_array_free(lg);
                const y = try moeDecode(s, x, &g.rb, g.routes[step % RG_ROUTES], flat_sc, k, .bfloat16, sh, lg);
                defer _ = mlx.mlx_array_free(y);
                return reshaped(s, y, &.{ 1, 1, RG_HIDDEN });
            }
            const y = try moeDecode(s, x, &g.rb, g.routes[step % RG_ROUTES], flat_sc, k, .bfloat16, null, null);
            defer _ = mlx.mlx_array_free(y);
            const y3 = try reshaped(s, y, &.{ rows, 1, RG_HIDDEN });
            defer _ = mlx.mlx_array_free(y3);
            return addTail(s, y3, c.shared, c.logit);
        },
    }
}

fn benchChainNs(c: *const BenchCtx, arm: *const BenchArm) !u64 {
    var x = try rgLink(c.s, c.x0, c.x0);
    defer _ = mlx.mlx_array_free(x);
    for (0..RG_STEPS) |i| {
        const y = try benchStep(c, arm, x, i);
        defer _ = mlx.mlx_array_free(y);
        const flat = try reshaped(c.s, y, &.{ @intCast(c.rows), RG_HIDDEN });
        defer _ = mlx.mlx_array_free(flat);
        const next = try rgLink(c.s, c.x0, flat);
        _ = mlx.mlx_array_free(x);
        x = next;
    }
    const io = std.Io.Threaded.global_single_threaded.io();
    var sw = io_util.Stopwatch.init(io);
    try mlx.check(mlx.mlx_array_eval(x));
    return sw.read();
}

fn diagOn(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return v[0] != 0 and v[0] != '0';
}

test "exl3 rate groups decode timing over uniform n32 and today's moeGroups (SUSHI_EXL3_RG_UBENCH=1)" {
    if (!diagOn("SUSHI_EXL3_RG_UBENCH")) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    var prng = std.Random.DefaultPrng.init(0x7a11);
    const rnd = prng.random();
    var uniform = try benchBank(rnd, RG_EXPERTS, .{ 32, 32, 32 });
    defer freeBank(&uniform);
    var groups: [RG_LAYOUTS.len]BenchGroups = undefined;
    var built: usize = 0;
    defer for (groups[0..built]) |*g| g.rb.deinit();
    for (RG_LAYOUTS, 0..) |lay, li| {
        var banks: [max_groups]root.Bank = undefined;
        const per: u32 = RG_EXPERTS / @as(u32, @intCast(lay.rates.len));
        for (lay.rates, 0..) |r, gi| {
            const e = if (gi + 1 == lay.rates.len) RG_EXPERTS - per * @as(u32, @intCast(gi)) else per;
            banks[gi] = try benchBank(rnd, e, r);
        }
        groups[li].rb = try Ragged.init(s, banks[0..lay.rates.len]);
        built += 1;
        const st = groups[li].rb.storage();
        const vec = mlx.mlx_vector_array_new_data(&st, st.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        for (banks[0..lay.rates.len]) |*b| freeBank(b);
    }
    for ([_]usize{ 1, 4 }) |rows| {
        var routes: [RG_LAYOUTS.len][RG_ROUTES]mlx.mlx_array = undefined;
        var uroutes: [RG_ROUTES]mlx.mlx_array = undefined;
        try benchRoutes(rnd, rows, null, &.{}, &uroutes);
        defer for (uroutes) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for (&groups, RG_LAYOUTS, 0..) |*g, lay, li| {
            try benchRoutes(rnd, rows, &g.rb, lay.picks, &routes[li]);
            g.routes = routes[li];
        }
        defer for (routes) |set| for (set) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const ci: c_int = @intCast(rows);
        const xh = try std.heap.page_allocator.alloc(u16, rows * @as(usize, @intCast(RG_HIDDEN)));
        defer std.heap.page_allocator.free(xh);
        for (xh) |*v| v.* = @truncate(@as(u32, @bitCast((rnd.float(f32) * 2 - 1) * 0.05)) >> 16);
        const x0 = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ ci, RG_HIDDEN }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(x0);
        const shared = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ ci, 1, RG_HIDDEN }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(shared);
        var lg: [16]u16 = @splat(0xbf00);
        const logit = mlx.mlx_array_new_data(&lg, &[_]c_int{ ci, 1, 1 }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(logit);
        var scv: [16 * RG_TOPK]u16 = undefined;
        for (&scv) |*v| v.* = @truncate(@as(u32, @bitCast(0.05 + rnd.float(f32) * 0.1)) >> 16);
        const scores = mlx.mlx_array_new_data(&scv, &[_]c_int{ ci, 1, RG_TOPK }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(scores);
        for ([_]exl3.Decode{ exl3.Decode.tabled(.spec1, .w15), .{ .codebook = .pair1, .window = .w16 } }) |dec| {
            const ctx = BenchCtx{ .s = s, .rows = rows, .x0 = x0, .x3 = x0, .scores = scores, .shared = shared, .logit = logit, .uniform = uniform, .uniform_routes = uroutes, .groups = &groups, .dec = dec };
            var arms: [1 + 2 * RG_LAYOUTS.len]BenchArm = undefined;
            arms[0] = .{ .name = "uniform n32 (served)", .kind = .uniform };
            for (0..RG_LAYOUTS.len) |li| {
                arms[1 + 2 * li] = .{ .name = "moeGroups today", .kind = .today, .layout = li };
                arms[2 + 2 * li] = .{ .name = "one dispatch", .kind = .one, .layout = li };
            }
            for (0..10) |round| {
                for (0..arms.len) |j| {
                    const a = &arms[(round + j) % arms.len];
                    if (a.err != null) continue;
                    const ns = benchChainNs(&ctx, a) catch |e| {
                        a.err = e;
                        continue;
                    };
                    if (round > 0) a.times[round - 1] = ns;
                }
            }
            var ref_med: f64 = 0;
            var ref_min: f64 = 0;
            for (&arms) |*a| {
                if (a.err != null) continue;
                std.mem.sort(u64, &a.times, {}, std.sort.asc(u64));
                if (a.kind == .uniform) {
                    ref_med = @floatFromInt(a.times[4]);
                    ref_min = @floatFromInt(a.times[0]);
                }
            }
            const per: f64 = 1e3 * @as(f64, RG_STEPS);
            for (arms) |a| {
                const lay = if (a.kind == .uniform) "" else RG_LAYOUTS[a.layout].name;
                if (a.err) |e| {
                    std.debug.print("[rg] {s} rows={d} {s} {s}: error {s}\n", .{ @tagName(dec.codebook), rows, lay, a.name, @errorName(e) });
                    continue;
                }
                const med: f64 = @floatFromInt(a.times[4]);
                const lo: f64 = @floatFromInt(a.times[0]);
                std.debug.print("[rg] {s} rows={d} {s} | {s}: {d:.2} us/step median ({d:.3}), {d:.2} min ({d:.3})\n", .{ @tagName(dec.codebook), rows, lay, a.name, med / per, med / ref_med, lo / per, lo / ref_min });
            }
        }
    }
}

/// The uniform bank of a fixture whose groups share one rate: each projection's group arrays stacked.
fn stackedBank(s: mlx.mlx_stream, f: *const Fixture, held: *[9]mlx.mlx_array) !root.Bank {
    var uniform: root.Bank = undefined;
    var k: usize = 0;
    for (0..3) |p| {
        var parts: [3][max_groups]mlx.mlx_array = undefined;
        const n = f.spec.rates.len;
        for (0..n) |g| {
            const q = projOf(f.banks[g], p);
            parts[0][g] = q.trellis;
            parts[1][g] = q.suh;
            parts[2][g] = q.svh;
        }
        for (0..3) |part| {
            held[k] = try concat(s, parts[part][0..n]);
            k += 1;
        }
        projPtr(&uniform, p).* = .{ .trellis = held[k - 3], .suh = held[k - 2], .svh = held[k - 1] };
    }
    return uniform;
}

test "exl3 rate groups: prefill rows of groups sharing one rate equal the uniform prefill to the byte" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    for (CODEBOOKS) |dec| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const rates = [_][3]u32{ .{ 36, 36, 36 }, .{ 36, 36, 36 }, .{ 36, 36, 36 }, .{ 36, 36, 36 } };
        const experts = [_]u32{ 3, 5, 1, 7 };
        var f = try Fixture.init(arena.allocator(), s, .{ .hidden = 512, .inter = 256, .rates = &rates, .experts = &experts, .rows = 40, .topk = 6 });
        defer f.deinit();
        var held: [9]mlx.mlx_array = undefined;
        const uniform = try stackedBank(s, &f, &held);
        defer for (held) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const x3 = try reshaped(s, f.x, &.{ 40, 1, 512 });
        defer _ = mlx.mlx_array_free(x3);
        const ids = try reshaped(s, f.slots, &.{ 40, 1, 6 });
        defer _ = mlx.mlx_array_free(ids);
        const sc = try reshaped(s, f.scores, &.{ 40, 1, 6 });
        defer _ = mlx.mlx_array_free(sc);
        const want = try root.moe(s, x3, uniform, ids, sc, dec, false);
        defer _ = mlx.mlx_array_free(want);
        const got = try root.moeRagged(s, x3, &f.rb, ids, sc, dec, false, null, null);
        defer _ = mlx.mlx_array_free(got);
        defer kernels.setDecodeParams(.mul1);
        var cw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cw);
        var cg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cg);
        try t.expectEqualSlices(u8, try bytesOf(s, want, &cw), try bytesOf(s, got, &cg));
    }
}

test "exl3 rate groups: prefill rows hold the f32-truth bar for every codebook, past 24 groups" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    for (CODEBOOKS) |dec| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const alloc = arena.allocator();
        var buf: [max_groups][3]u32 = undefined;
        var rates = everyRate(&buf, 24, 64);
        // Five more groups with gate, up and down apart: 26 in all, two combine dispatches.
        const extra = [_][3]u32{ .{ 24, 48, 36 }, .{ 64, 40, 64 }, .{ 32, 32, 56 }, .{ 48, 24, 24 }, .{ 40, 40, 26 } };
        @memcpy(buf[rates.len..][0..extra.len], &extra);
        rates = buf[0 .. rates.len + extra.len];
        var experts: [max_groups]u32 = undefined;
        for (experts[0..rates.len], 0..) |*e, i| e.* = 1 + @as(u32, @intCast(i % 3));
        var f = try Fixture.init(alloc, s, .{ .hidden = 512, .inter = 256, .rates = rates, .experts = experts[0..rates.len], .rows = 36, .topk = 8, .banks = .{ 0.0126, 1.03, 0.0083, 1.009 }, .x_scale = 40 });
        defer f.deinit();
        const want = try hostTruth(alloc, &f, dec);
        const x3 = try reshaped(s, f.x, &.{ 36, 1, 512 });
        defer _ = mlx.mlx_array_free(x3);
        const ids = try reshaped(s, f.slots, &.{ 36, 1, 8 });
        defer _ = mlx.mlx_array_free(ids);
        const sc = try reshaped(s, f.scores, &.{ 36, 1, 8 });
        defer _ = mlx.mlx_array_free(sc);
        const y = try root.moeRagged(s, x3, &f.rb, ids, sc, dec, false, null, null);
        defer _ = mlx.mlx_array_free(y);
        defer kernels.setDecodeParams(.mul1);
        var cy = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cy);
        const got = try toF32(s, y, &cy);
        for (got) |v| try t.expect(std.math.isFinite(v));
        try t.expect(relRms(got, want) < 0.01);
        // The decode chain on the same rows (verify rows) scores the same.
        const v = try root.moeRagged(s, x3, &f.rb, ids, sc, dec, true, null, null);
        defer _ = mlx.mlx_array_free(v);
        var cv = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cv);
        try t.expect(relRms(try toF32(s, v, &cv), want) < 0.01);
    }
}

fn prefillRoutes(alloc: std.mem.Allocator, rnd: std.Random, rows: usize, rb: ?*const Ragged, picks: []const []const u8) !mlx.mlx_array {
    const ids = try alloc.alloc(u32, rows * RG_TOPK);
    defer alloc.free(ids);
    for (0..rows) |row| {
        const ri = ids[row * RG_TOPK ..][0..RG_TOPK];
        var k: usize = 0;
        while (k < RG_TOPK) {
            const pick: u32 = if (rb) |b| blk: {
                const choices = picks[k];
                const g = choices[rnd.uintLessThan(usize, choices.len)];
                break :blk b.base[g] + rnd.uintLessThan(u32, b.experts_of[g]);
            } else rnd.uintLessThan(u32, RG_EXPERTS);
            if (std.mem.indexOfScalar(u32, ri[0..k], pick) != null) continue;
            ri[k] = pick;
            k += 1;
        }
    }
    return mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ @intCast(rows), 1, RG_TOPK }, 3, .uint32);
}

// One DeepSeek-V4.1 routed layer (5120 x 2304, 384 experts in four PAIR1 groups at n24/26/32/40, top-6) through the
// grouped prefill chain with every expert routed exactly r rows: the cost of a window against its live rows.
// Prints only under SUSHI_EXL3_RG_ROWS_UBENCH.
test "exl3 rate groups prefill cost by rows per expert at DeepSeek-V4.1 geometry (SUSHI_EXL3_RG_ROWS_UBENCH=1)" {
    if (!diagOn("SUSHI_EXL3_RG_ROWS_UBENCH")) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    const alloc = std.heap.page_allocator;
    var prng = std.Random.DefaultPrng.init(0xd541);
    const rnd = prng.random();
    const hidden: c_int = 5120;
    const inter: c_int = 2304;
    const experts: u32 = 384;
    const topk: usize = 6;
    var banks: [4]root.Bank = undefined;
    for ([_]u32{ 24, 26, 32, 40 }, 0..) |n, g| banks[g] = try benchBankAt(rnd, experts / 4, .{ n, n, n }, hidden, inter);
    var rb = try Ragged.init(s, &banks);
    defer rb.deinit();
    for (&banks) |*b| freeBank(b);
    const st = rb.storage();
    const vec = mlx.mlx_vector_array_new_data(&st, st.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
    const dec: exl3.Decode = .{ .codebook = .pair1, .window = .w16 };
    const per_expert = [_]usize{ 1, 4, 8, 12, 16, 17, 20, 24, 28, 32, 33, 40, 48 };
    const steps = 4;
    var times: [per_expert.len][5]u64 = undefined;
    var xs: [per_expert.len]mlx.mlx_array = undefined;
    var routes: [per_expert.len]mlx.mlx_array = undefined;
    var scores: [per_expert.len]mlx.mlx_array = undefined;
    for (per_expert, 0..) |r, a| {
        const rows = experts * r / topk;
        const xh = try alloc.alloc(u16, rows * @as(usize, @intCast(hidden)));
        defer alloc.free(xh);
        for (xh) |*v| v.* = @truncate(@as(u32, @bitCast((rnd.float(f32) * 2 - 1) * 0.05)) >> 16);
        xs[a] = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), 1, hidden }, 3, .bfloat16);
        const ids = try alloc.alloc(u32, rows * topk);
        defer alloc.free(ids);
        for (ids, 0..) |*id, j| id.* = @intCast(j % experts);
        routes[a] = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ @intCast(rows), 1, @intCast(topk) }, 3, .uint32);
        const scv = try alloc.alloc(u16, rows * topk);
        defer alloc.free(scv);
        for (scv) |*v| v.* = @truncate(@as(u32, @bitCast(0.05 + rnd.float(f32) * 0.1)) >> 16);
        scores[a] = mlx.mlx_array_new_data(scv.ptr, &[_]c_int{ @intCast(rows), 1, @intCast(topk) }, 3, .bfloat16);
    }
    defer for (0..per_expert.len) |a| {
        _ = mlx.mlx_array_free(xs[a]);
        _ = mlx.mlx_array_free(routes[a]);
        _ = mlx.mlx_array_free(scores[a]);
    };
    for (0..6) |round| {
        for (0..per_expert.len) |j| {
            const arm = (round + j) % per_expert.len;
            var x = try rgLink(s, xs[arm], xs[arm]);
            for (0..steps) |_| {
                const y = try root.moeRagged(s, x, &rb, routes[arm], scores[arm], dec, false, null, null);
                const next = try rgLink(s, xs[arm], y);
                _ = mlx.mlx_array_free(y);
                _ = mlx.mlx_array_free(x);
                x = next;
            }
            const io = std.Io.Threaded.global_single_threaded.io();
            var sw = io_util.Stopwatch.init(io);
            try mlx.check(mlx.mlx_array_eval(x));
            const ns = sw.read();
            _ = mlx.mlx_array_free(x);
            if (round > 0) times[arm][round - 1] = ns;
        }
    }
    for (&times, per_expert) |*t, r| {
        std.mem.sort(u64, t, {}, std.sort.asc(u64));
        const ms = @as(f64, @floatFromInt(t[2])) / steps / 1e6;
        const flop = @as(f64, @floatFromInt(experts * r)) * 3 * 2 * @as(f64, @floatFromInt(hidden * inter));
        std.debug.print("[rg-rows] rows/expert={d} rows={d}: {d:.3} ms/layer median, {d:.3} us/expert, {d:.1} TFLOP/s\n", .{ r, experts * r / topk, ms, ms * 1e3 / @as(f64, @floatFromInt(experts)), flop / (ms / 1e3) / 1e12 });
    }
}

// Prefill chunks of one Flash-Next MoE layer: uniform n32, today's moeGroups, and the grouped chain,
// interleaved, 4 dependent chunks per arm. Prints only under SUSHI_EXL3_RG_PREFILL_UBENCH.
test "exl3 rate groups prefill timing over uniform n32 and today's moeGroups (SUSHI_EXL3_RG_PREFILL_UBENCH=1)" {
    if (!diagOn("SUSHI_EXL3_RG_PREFILL_UBENCH")) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    const alloc = std.heap.page_allocator;
    var prng = std.Random.DefaultPrng.init(0x7a13);
    const rnd = prng.random();
    var uniform = try benchBank(rnd, RG_EXPERTS, .{ 32, 32, 32 });
    defer freeBank(&uniform);
    const lays = [_]usize{ 0, 1, 2 };
    var rbs: [lays.len]Ragged = undefined;
    var built: usize = 0;
    defer for (rbs[0..built]) |*rb| rb.deinit();
    for (lays, 0..) |li, k| {
        const lay = RG_LAYOUTS[li];
        var banks: [max_groups]root.Bank = undefined;
        const per: u32 = RG_EXPERTS / @as(u32, @intCast(lay.rates.len));
        for (lay.rates, 0..) |r, gi| banks[gi] = try benchBank(rnd, if (gi + 1 == lay.rates.len) RG_EXPERTS - per * @as(u32, @intCast(gi)) else per, r);
        rbs[k] = try Ragged.init(s, banks[0..lay.rates.len]);
        built += 1;
        const st = rbs[k].storage();
        const vec = mlx.mlx_vector_array_new_data(&st, st.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        for (banks[0..lay.rates.len]) |*b| freeBank(b);
    }
    const steps = 4;
    for ([_]usize{ 512, 2048 }) |rows| {
        const ci: c_int = @intCast(rows);
        const xh = try alloc.alloc(u16, rows * @as(usize, @intCast(RG_HIDDEN)));
        defer alloc.free(xh);
        for (xh) |*v| v.* = @truncate(@as(u32, @bitCast((rnd.float(f32) * 2 - 1) * 0.05)) >> 16);
        const x0 = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ ci, 1, RG_HIDDEN }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(x0);
        const scv = try alloc.alloc(u16, rows * RG_TOPK);
        defer alloc.free(scv);
        for (scv) |*v| v.* = @truncate(@as(u32, @bitCast(0.05 + rnd.float(f32) * 0.1)) >> 16);
        const scores = mlx.mlx_array_new_data(scv.ptr, &[_]c_int{ ci, 1, RG_TOPK }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(scores);
        const uroute = try prefillRoutes(alloc, rnd, rows, null, &.{});
        defer _ = mlx.mlx_array_free(uroute);
        var routes: [lays.len]mlx.mlx_array = undefined;
        for (lays, 0..) |li, k| routes[k] = try prefillRoutes(alloc, rnd, rows, &rbs[k], RG_LAYOUTS[li].picks);
        defer for (routes) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for ([_]exl3.Decode{ exl3.Decode.tabled(.spec1, .w15), .{ .codebook = .pair1, .window = .w16 } }) |dec| {
            const n_arms = 1 + 2 * lays.len;
            var times: [n_arms][5]u64 = undefined;
            for (0..6) |round| {
                for (0..n_arms) |j| {
                    const arm = (round + j) % n_arms;
                    var x = try rgLink(s, x0, x0);
                    for (0..steps) |_| {
                        const y = if (arm == 0)
                            try root.moe(s, x, uniform, uroute, scores, dec, false)
                        else blk: {
                            const k = (arm - 1) / 2;
                            break :blk if ((arm - 1) % 2 == 0)
                                try root.moeGroups(s, x, rbs[k].views[0..rbs[k].count], routes[k], scores, dec, false)
                            else
                                try root.moeRagged(s, x, &rbs[k], routes[k], scores, dec, false, null, null);
                        };
                        const next = try rgLink(s, x0, y);
                        _ = mlx.mlx_array_free(y);
                        _ = mlx.mlx_array_free(x);
                        x = next;
                    }
                    const io = std.Io.Threaded.global_single_threaded.io();
                    var sw = io_util.Stopwatch.init(io);
                    try mlx.check(mlx.mlx_array_eval(x));
                    const ns = sw.read();
                    _ = mlx.mlx_array_free(x);
                    if (round > 0) times[arm][round - 1] = ns;
                }
            }
            for (&times) |*t| std.mem.sort(u64, t, {}, std.sort.asc(u64));
            const ref: f64 = @floatFromInt(times[0][2]);
            for (times, 0..) |t, arm| {
                const name = if (arm == 0) "uniform n32 (served)" else if ((arm - 1) % 2 == 0) "moeGroups today" else "grouped prefill";
                const lay = if (arm == 0) "" else RG_LAYOUTS[lays[(arm - 1) / 2]].name;
                const med: f64 = @floatFromInt(t[2]);
                std.debug.print("[rg-prefill] {s} rows={d} {s} | {s}: {d:.3} ms/chunk median ({d:.3}), {d:.3} min\n", .{ @tagName(dec.codebook), rows, lay, name, med / 4e6, med / ref, @as(f64, @floatFromInt(t[0])) / 4e6 });
            }
        }
    }
}
