const std = @import("std");
const format = @import("expert_exl3.zig");

pub const max_groups = 32;
pub const Dtype = enum { u16, f16, other };
pub const Name = struct {
    group: u32,
    projection: u32,
    part: u32,
    grouped: bool,
    /// The trellis is stored `trellis_pm`, in `format.TileOrder.pair_major`.
    pair_major: bool = false,

    pub fn parse(name: []const u8) !Name {
        var it = std.mem.splitScalar(u8, name, '.');
        const proj = it.next() orelse return error.Exl3GroupNameInvalid;
        const p: u32 = if (std.mem.eql(u8, proj, "gate_proj")) 0 else if (std.mem.eql(u8, proj, "up_proj")) 1 else if (std.mem.eql(u8, proj, "down_proj")) 2 else return error.Exl3GroupNameInvalid;
        var leaf = it.next() orelse return error.Exl3GroupNameInvalid;
        var group: u32 = 0;
        const grouped = std.mem.startsWith(u8, leaf, "g");
        if (grouped) {
            if (leaf.len < 2 or (leaf.len > 2 and leaf[1] == '0')) return error.Exl3GroupNameInvalid;
            for (leaf[1..]) |c| if (c < '0' or c > '9') return error.Exl3GroupNameInvalid;
            group = std.fmt.parseInt(u32, leaf[1..], 10) catch return error.Exl3GroupNameInvalid;
            if (group >= max_groups) return error.Exl3GroupNameInvalid;
            leaf = it.next() orelse return error.Exl3GroupNameInvalid;
        }
        const pair_major = std.mem.eql(u8, leaf, "trellis_pm");
        const part: u32 = if (pair_major or std.mem.eql(u8, leaf, "trellis")) 0 else if (std.mem.eql(u8, leaf, "suh")) 1 else if (std.mem.eql(u8, leaf, "svh")) 2 else return error.Exl3GroupNameInvalid;
        if (it.next() != null) return error.Exl3GroupNameInvalid;
        return .{ .group = group, .projection = p, .part = part, .grouped = grouped, .pair_major = pair_major };
    }

    pub fn tileOrder(self: Name) format.TileOrder {
        return if (self.pair_major) .pair_major else .pack;
    }
};

pub const TileDims = struct { it: u64, ot: u64, n: u64 };

/// The tile grid of a trellis tensor: `[E, IT, OT, n]`, or `[E, OT/2, IT, 2, n]` for `pair_major`; null when the
/// shape is not that order's.
pub fn tileDims(order: format.TileOrder, shape: []const u64) ?TileDims {
    return switch (order) {
        .pack => if (shape.len == 4) .{ .it = shape[1], .ot = shape[2], .n = shape[3] } else null,
        .pair_major => if (shape.len == 5 and shape[3] == 2) .{ .it = shape[2], .ot = shape[1] * 2, .n = shape[4] } else null,
    };
}

pub const Group = struct {
    experts: u32 = 0,
    rates: [3]format.Rate = @splat(.{ .n = 0 }),
    mask: u16 = 0,
};

/// The rate of a trellis tensor of `order` over an `input` x `output` projection: its grid must be that projection's.
pub fn trellisRate(order: format.TileOrder, shape: []const u64, input: u32, output: u32) !format.Rate {
    const dims = tileDims(order, shape) orelse return error.Exl3GroupGeometry;
    if (dims.it != input / 16 or dims.ot != output / 16) return error.Exl3GroupGeometry;
    const halfwords = std.math.cast(usize, dims.n) orelse return error.Exl3TrellisGeometry;
    return format.kFromPackedDim(halfwords) orelse error.Exl3TrellisGeometry;
}

pub const Layout = struct {
    hidden: u32,
    intermediate: u32,
    ceiling: format.Rate,
    /// The bank rates the pack's codebook table has a row for (`Spec.rates`); null admits any up to `ceiling`.
    rates: ?u128 = null,
    groups: [max_groups]Group = @splat(.{}),
    count: usize = 0,
    grouped: ?bool = null,
    /// The order of the layer's trellis tensors; null before the first.
    tile_order: ?format.TileOrder = null,
    /// The streaming engine serves rate groups (pruned layers still refuse it).
    streams_groups: bool = false,
    bytes: u64 = 0,

    pub fn init(hidden: u32, intermediate: u32, ceiling: format.Rate) Layout {
        return .{ .hidden = hidden, .intermediate = intermediate, .ceiling = ceiling };
    }

    pub fn add(self: *Layout, suffix: []const u8, shape: []const u64, dtype: Dtype) !Name {
        const name = try Name.parse(suffix);
        if (self.grouped) |g| if (g != name.grouped) return error.Exl3MixedGroupLayout;
        if (dtype != (if (name.part == 0) Dtype.u16 else Dtype.f16)) return error.Exl3GroupDtype;
        const input = if (name.projection == 2) self.intermediate else self.hidden;
        const output = if (name.projection == 2) self.hidden else self.intermediate;
        if (input == 0 or output == 0 or input % 128 != 0 or output % 128 != 0) return error.Exl3GroupGeometry;
        if (name.part != 0 and shape.len != 2) return error.Exl3GroupGeometry;
        if (shape.len == 0 or shape[0] == 0 or shape[0] > std.math.maxInt(c_int)) return error.Exl3GroupGeometry;
        const group = &self.groups[name.group];
        if (group.experts != 0 and group.experts != shape[0]) return error.Exl3GroupGeometry;
        var rate: format.Rate = .{ .n = 0 };
        if (name.part == 0) {
            const order = name.tileOrder();
            if (self.tile_order) |o| if (o != order) return error.Exl3TileOrderMixed;
            rate = try trellisRate(order, shape, input, output);
            if (rate.n > self.ceiling.n) return error.Exl3TrellisGeometry;
            if (!rate.in(self.rates)) return error.Exl3CodebookTableRateMissing;
        } else if (shape[1] != (if (name.part == 1) input else output)) return error.Exl3GroupGeometry;
        const bit = @as(u16, 1) << @intCast(name.projection * 3 + name.part);
        if (group.mask & bit != 0) return error.Exl3GroupNameInvalid;
        var bytes: u64 = 2;
        for (shape) |d| bytes = std.math.mul(u64, bytes, d) catch return error.ResidentBytesOverflow;
        self.bytes = std.math.add(u64, self.bytes, bytes) catch return error.ResidentBytesOverflow;
        self.grouped = name.grouped;
        if (name.part == 0) self.tile_order = name.tileOrder();
        self.count = @max(self.count, name.group + 1);
        group.experts = @intCast(shape[0]);
        group.mask |= bit;
        if (name.part == 0) group.rates[name.projection] = rate;
        return name;
    }

    pub fn finish(self: *const Layout, router_experts: u32, topk: u32, streaming: bool, configured_experts: u32) !void {
        if (self.count == 0) return error.MissingWeight;
        var total: u64 = 0;
        for (self.groups[0..self.count]) |g| {
            if (g.mask != 511) return if (self.grouped.?) error.Exl3GroupMissing else error.MissingWeight;
            total += g.experts;
        }
        if (total != router_experts or total > std.math.maxInt(c_int)) return error.Exl3RouterWidthMismatch;
        if (topk > router_experts) return error.Exl3TopKExceedsExperts;
        if (topk > 32) return error.Exl3TopKExceedsReduceBank;
        if (streaming and ((self.grouped.? and !self.streams_groups) or router_experts != configured_experts)) return error.Exl3RaggedStreamingUnsupported;
    }

    pub fn locate(self: *const Layout, expert: u32) !struct { group: u32, local: u32 } {
        var offset: u32 = 0;
        for (self.groups[0..self.count], 0..) |g, i| {
            if (expert >= offset and expert - offset < g.experts) return .{ .group = @intCast(i), .local = expert - offset };
            offset += g.experts;
        }
        return error.ExpertOutOfRange;
    }
};

/// One expert's projection from the pack's `[IT][OT][n]` tile order into `[OT/2][IT][2][n]`: an output tile pair's whole
/// K range is one contiguous run, which the decode and prefill kernels stream. `n` is `T`s per tile.
pub fn tileMajor(comptime T: type, dst: []T, src: []const T, it: usize, ot: usize, n: usize) void {
    std.debug.assert(ot % 2 == 0 and dst.len == it * ot * n and src.len == dst.len);
    for (0..ot / 2) |pair| for (0..it) |tk| for (0..2) |o| {
        const from = (tk * ot + pair * 2 + o) * n;
        const to = ((pair * it + tk) * 2 + o) * n;
        @memcpy(dst[to..][0..n], src[from..][0..n]);
    };
}

/// `tileMajor` undone: an expert's `[OT/2][IT][2][n]` tiles back into the pack order `[IT][OT][n]`.
pub fn tileMajorInverse(comptime T: type, dst: []T, src: []const T, it: usize, ot: usize, n: usize) void {
    std.debug.assert(ot % 2 == 0 and dst.len == it * ot * n and src.len == dst.len);
    for (0..ot / 2) |pair| for (0..it) |tk| for (0..2) |o| {
        const to = (tk * ot + pair * 2 + o) * n;
        const from = ((pair * it + tk) * 2 + o) * n;
        @memcpy(dst[to..][0..n], src[from..][0..n]);
    };
}

test "exl3 layout: tile-major puts an output pair's k range in one run" {
    const it = 5;
    const ot = 4;
    const n = 3;
    var src: [it * ot * n]u16 = undefined;
    for (&src, 0..) |*v, i| v.* = @intCast(i);
    var dst: [src.len]u16 = undefined;
    tileMajor(u16, &dst, &src, it, ot, n);
    for (0..ot / 2) |pair| for (0..it) |tk| for (0..2) |o| {
        const want = src[(tk * ot + pair * 2 + o) * n ..][0..n];
        try std.testing.expectEqualSlices(u16, want, dst[((pair * it + tk) * 2 + o) * n ..][0..n]);
    };
}

test "exl3 layout: a pair-major trellis is named trellis_pm and its 5-D shape is read as the same tile grid" {
    const name = try Name.parse("gate_proj.g1.trellis_pm");
    try std.testing.expect(name.pair_major and name.part == 0 and name.group == 1);
    try std.testing.expectEqual(format.TileOrder.pair_major, name.tileOrder());
    try std.testing.expect(!(try Name.parse("gate_proj.g1.trellis")).pair_major);
    try std.testing.expectError(error.Exl3GroupNameInvalid, Name.parse("gate_proj.g1.suh_pm"));
    try std.testing.expectError(error.Exl3GroupNameInvalid, Name.parse("gate_proj.trellis_pm.x"));

    const d = tileDims(.pair_major, &.{ 3, 4, 16, 2, 24 }).?;
    try std.testing.expectEqual(TileDims{ .it = 16, .ot = 8, .n = 24 }, d);
    try std.testing.expectEqual(TileDims{ .it = 16, .ot = 8, .n = 24 }, tileDims(.pack, &.{ 3, 16, 8, 24 }).?);
    try std.testing.expectEqual(@as(?TileDims, null), tileDims(.pair_major, &.{ 3, 16, 8, 24 }));
    try std.testing.expectEqual(@as(?TileDims, null), tileDims(.pair_major, &.{ 3, 4, 16, 3, 24 }));
    try std.testing.expectEqual(@as(?TileDims, null), tileDims(.pack, &.{ 3, 4, 16, 2, 24 }));
}

test "exl3 layout: a layer admits a pair-major bank by its 5-D shape and refuses a bank of the wrong grid or a mixed one" {
    const t = std.testing;
    const rate: format.Rate = .{ .n = 32 };
    var ok = Layout.init(256, 128, .{ .n = 64 });
    _ = try ok.add("gate_proj.g0.trellis_pm", &.{ 3, 4, 16, 2, 32 }, .u16);
    _ = try ok.add("down_proj.g0.trellis_pm", &.{ 3, 8, 8, 2, 32 }, .u16);
    try t.expectEqual(format.TileOrder.pair_major, ok.tile_order.?);
    try t.expectEqual(rate.n, ok.groups[0].rates[0].n);
    try t.expectEqual(rate.n, ok.groups[0].rates[2].n);

    var wrong = Layout.init(256, 128, .{ .n = 64 });
    // The legacy grid of gate (16 x 8 tiles) under the pair-major rank, and a pair axis that is not 2.
    try t.expectError(error.Exl3GroupGeometry, wrong.add("gate_proj.g0.trellis_pm", &.{ 3, 16, 8, 32 }, .u16));
    try t.expectError(error.Exl3GroupGeometry, wrong.add("gate_proj.g0.trellis_pm", &.{ 3, 4, 16, 4, 32 }, .u16));
    // Out tiles halved wrongly (OT/2 = 8 would mean 16 output tiles) and an input grid that is the output's.
    try t.expectError(error.Exl3GroupGeometry, wrong.add("gate_proj.g0.trellis_pm", &.{ 3, 8, 16, 2, 32 }, .u16));
    try t.expectError(error.Exl3GroupGeometry, wrong.add("gate_proj.g0.trellis_pm", &.{ 3, 16, 4, 2, 32 }, .u16));
    try t.expectError(error.Exl3TrellisGeometry, wrong.add("gate_proj.g0.trellis_pm", &.{ 3, 4, 16, 2, 66 }, .u16));
    // The legacy 4-D shape under the legacy name does not read a pair-major tensor.
    try t.expectError(error.Exl3GroupGeometry, wrong.add("gate_proj.g0.trellis", &.{ 3, 4, 16, 2, 32 }, .u16));

    var mixed = Layout.init(256, 128, .{ .n = 64 });
    _ = try mixed.add("gate_proj.g0.trellis_pm", &.{ 3, 4, 16, 2, 32 }, .u16);
    try t.expectError(error.Exl3TileOrderMixed, mixed.add("up_proj.g0.trellis", &.{ 3, 16, 8, 32 }, .u16));
}

test "exl3 layout: tile-major inverse returns the pack order for odd and even grids" {
    for ([_][3]usize{ .{ 5, 4, 3 }, .{ 1, 2, 1 }, .{ 16, 8, 24 }, .{ 3, 10, 7 } }) |g| {
        const it, const ot, const n = g;
        const src = try std.testing.allocator.alloc(u16, it * ot * n);
        defer std.testing.allocator.free(src);
        for (src, 0..) |*v, i| v.* = @truncate(i *% 40503 +% 17);
        const major = try std.testing.allocator.alloc(u16, src.len);
        defer std.testing.allocator.free(major);
        const back = try std.testing.allocator.alloc(u16, src.len);
        defer std.testing.allocator.free(back);
        tileMajor(u16, major, src, it, ot, n);
        tileMajorInverse(u16, back, major, it, ot, n);
        try std.testing.expectEqualSlices(u16, src, back);
    }
}
