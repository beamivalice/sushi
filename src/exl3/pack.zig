//! Reading an EXL3/STQ pack's routed experts off its safetensors index: tensor keys, bank completeness, tile order and
//! the span plan of a rate-grouped layer. Std only, so a host that streams the pack from disk can share it.
const std = @import("std");
const format = @import("expert_exl3.zig");
const gl = @import("layout.zig");

pub const Component = enum(u4) {
    gate_w,
    gate_s,
    gate_b,
    up_w,
    up_s,
    up_b,
    down_w,
    down_s,
    down_b,
};

pub const component_count: usize = 9;

pub const Projection = enum { gate, up, down };
pub const Part = enum { weight, scales, biases };

pub fn projectionOf(c: Component) Projection {
    return switch (c) {
        .gate_w, .gate_s, .gate_b => .gate,
        .up_w, .up_s, .up_b => .up,
        .down_w, .down_s, .down_b => .down,
    };
}

pub fn partOf(c: Component) Part {
    return switch (c) {
        .gate_w, .up_w, .down_w => .weight,
        .gate_s, .up_s, .down_s => .scales,
        .gate_b, .up_b, .down_b => .biases,
    };
}

pub fn weightOf(p: Projection) Component {
    return switch (p) {
        .gate => .gate_w,
        .up => .up_w,
        .down => .down_w,
    };
}

pub fn scalesOf(p: Projection) Component {
    return switch (p) {
        .gate => .gate_s,
        .up => .up_s,
        .down => .down_s,
    };
}

pub fn biasesOf(p: Projection) Component {
    return switch (p) {
        .gate => .gate_b,
        .up => .up_b,
        .down => .down_b,
    };
}

pub const SourceSpan = struct {
    file: u16,
    offset: u64,
    len: u64,
};

/// `language_model.model.layers.` is the qwen4 pack's nesting, `model.layers.`
/// MiMo's; a pack uses one of them for every routed layer it owns.
pub const EXL3_PREFIXES = [_][]const u8{ "language_model.model.layers.", "model.layers.", "model.language_model.layers." };

pub fn exl3TensorKey(buf: []u8, layer: u16, c: Component) ![]const u8 {
    return exl3TensorKeyWithPrefix(buf, EXL3_PREFIXES[0], layer, c);
}

pub fn exl3TensorKeyWithPrefix(buf: []u8, prefix: []const u8, layer: u16, c: Component) ![]const u8 {
    const suffix: []const u8 = switch (partOf(c)) {
        .weight => "trellis",
        .scales => "suh",
        .biases => "svh",
    };
    return std.fmt.bufPrint(buf, "{s}{d}.mlp.switch_mlp.{s}_proj.{s}", .{ prefix, layer, @tagName(projectionOf(c)), suffix });
}

pub fn exl3BankComplete(map: std.json.ObjectMap, prefix: []const u8, first_moe_layer: u16, layers: u16) bool {
    if (first_moe_layer >= layers) return false;
    for (first_moe_layer..layers) |layer| {
        var buf: [192]u8 = undefined;
        const start = std.fmt.bufPrint(&buf, "{s}{d}.mlp.switch_mlp.", .{ prefix, layer }) catch return false;
        var masks: [gl.max_groups]u16 = @splat(0);
        var count: usize = 0;
        var grouped: ?bool = null;
        var it = map.iterator();
        while (it.next()) |entry| {
            const key = entry.key_ptr.*;
            if (!std.mem.startsWith(u8, key, start)) continue;
            if (entry.value_ptr.* != .string) return false;
            const name = gl.Name.parse(key[start.len..]) catch return false;
            if (grouped) |g| if (g != name.grouped) return false;
            grouped = name.grouped;
            masks[name.group] |= @as(u16, 1) << @intCast(name.projection * 3 + name.part);
            count = @max(count, name.group + 1);
        }
        if (count == 0) return false;
        for (masks[0..count]) |mask| if (mask != 511) return false;
    }
    return true;
}

pub fn hasAnyExl3Key(map: std.json.ObjectMap, layers: u16) bool {
    _ = layers;
    var it = map.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        for (EXL3_PREFIXES) |prefix| {
            if (!std.mem.startsWith(u8, key, prefix)) continue;
            const at = std.mem.indexOf(u8, key, ".mlp.switch_mlp.") orelse continue;
            _ = gl.Name.parse(key[at + ".mlp.switch_mlp.".len ..]) catch continue;
            return true;
        }
    }
    return false;
}

pub fn exl3LayerGrouped(map: std.json.ObjectMap, prefix: []const u8, layer: u16) bool {
    var buf: [192]u8 = undefined;
    for ([_][]const u8{ "trellis", "trellis_pm" }) |leaf| {
        const key = std.fmt.bufPrint(&buf, "{s}{d}.mlp.switch_mlp.gate_proj.g0.{s}", .{ prefix, layer, leaf }) catch return false;
        if (map.get(key) != null) return true;
    }
    return false;
}

pub fn hasGroupedExl3Index(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return false;
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(64 * 1024 * 1024)) catch return false;
    defer allocator.free(raw);
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    const map = parsed.value.object.get("weight_map") orelse return false;
    if (map != .object) return false;
    for (map.object.keys()) |key| {
        for (EXL3_PREFIXES) |prefix| {
            if (!std.mem.startsWith(u8, key, prefix)) continue;
            for ([_][]const u8{ ".mlp.switch_mlp.gate_proj.g", ".mlp.switch_mlp.up_proj.g", ".mlp.switch_mlp.down_proj.g" }) |marker| {
                if (std.mem.indexOf(u8, key, marker) != null) return true;
            }
        }
    }
    return false;
}

/// The tile order every routed trellis key of `map` (the layers' and a DSpark stage's) names; a map naming both is
/// mixed, and one naming neither is the pack order.
pub fn tileOrderOfMap(map: std.json.ObjectMap) !format.TileOrder {
    const marker = ".mlp.switch_mlp.";
    var seen_pack = false;
    var seen_major = false;
    for (map.keys()) |key| {
        const at = std.mem.indexOf(u8, key, marker) orelse continue;
        const name = gl.Name.parse(key[at + marker.len ..]) catch continue;
        if (name.part != 0) continue;
        if (name.pair_major) seen_major = true else seen_pack = true;
    }
    if (seen_pack and seen_major) return error.Exl3TileOrderMixed;
    return if (seen_major) .pair_major else .pack;
}

pub fn tileOrderOfDir(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !format.TileOrder {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
    defer dir.close(io);
    const raw = try dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(64 * 1024 * 1024));
    defer allocator.free(raw);
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{}) catch return error.InvalidSafetensorsIndex;
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidSafetensorsIndex;
    const map = parsed.value.object.get("weight_map") orelse return error.InvalidSafetensorsIndex;
    if (map != .object) return error.InvalidSafetensorsIndex;
    return tileOrderOfMap(map.object);
}

pub const rate_group_limit = gl.max_groups;

/// A rate-grouped layer's experts: contiguous runs, each at its own gate, up and down rate.
pub const LayerGroups = struct {
    count: u8 = 0,
    experts: [rate_group_limit]u16 = @splat(0),
    base: [rate_group_limit]u16 = @splat(0),
    n: [rate_group_limit][3]u32 = @splat(@splat(0)),
    /// Experts of each group `--drop-expert` leaves unloaded; they keep their ids and positions in the pack.
    dropped: [rate_group_limit]u16 = @splat(0),
    /// Each pool's trellis starts at a multiple of this many bytes (a pair-major pack's page, for direct reads); 0: packed.
    align_bytes: u32 = 0,

    /// What alignment can add to a layer's trellis storage: under one `align_bytes` before each pool but the first,
    /// for each projection.
    pub fn padBytes(self: *const LayerGroups) u64 {
        return if (self.align_bytes == 0 or self.count == 0) 0 else 3 * @as(u64, self.count - 1) * self.align_bytes;
    }

    pub fn keptOf(self: *const LayerGroups, g: usize) u16 {
        return self.experts[g] - self.dropped[g];
    }

    pub fn groupOf(self: *const LayerGroups, expert: u16) u8 {
        var g: u8 = 0;
        while (g + 1 < self.count and expert >= self.base[g + 1]) g += 1;
        return g;
    }

    /// Bytes of one expert of group g: its trellises and scale planes.
    pub fn expertBytes(self: *const LayerGroups, g: usize, hidden: u32, inter: u32) u64 {
        const tiles = @as(u64, hidden / 16) * (inter / 16);
        var total: u64 = 0;
        for (0..3) |pi| total += tiles * self.n[g][pi] * 2 + (@as(u64, hidden) + inter) * 2;
        return total;
    }

    /// Every kept expert of every group.
    pub fn allBytes(self: *const LayerGroups, hidden: u32, inter: u32) u64 {
        var total: u64 = 0;
        for (0..self.count) |g| total += @as(u64, self.keptOf(g)) * self.expertBytes(g, hidden, inter);
        return total + self.padBytes();
    }

    /// One expert of every group that still has one: the least a cache of this layer can hold.
    pub fn floorBytes(self: *const LayerGroups, hidden: u32, inter: u32) u64 {
        var total: u64 = 0;
        for (0..self.count) |g| total += self.expertBytes(g, hidden, inter);
        return total + self.padBytes();
    }
};

/// Where one tensor of a group sits: its shard, first byte, bytes per expert and expert count.
pub const GroupPiece = struct { file: u16 = 0, base: u64 = 0, per_expert: u64 = 0, experts: u32 = 0 };

/// A layer's pieces by group, then by component (weight, scales, biases per projection, as trellis, suh, svh).
pub const GroupPieces = [rate_group_limit][component_count]GroupPiece;

pub const LayerDims = struct { experts: u16, hidden: u32, intermediate: u32 };

pub const GroupedSpans = struct {
    groups: LayerGroups,
    /// Bytes of each projection's trellis over the kept experts of every group.
    weight_total: [3]u64,
};

/// Cuts a validated rate-group `plan` into the layer's `LayerGroups` and one span per expert and component, the
/// groups in order, into `spans` (`experts * component_count` long). `drop` is null or points at a set with
/// `droppedIn(layer, lo, hi)`; a group it leaves with no expert is refused.
pub fn groupedLayerSpans(
    plan: *const gl.Layout,
    pieces: *const GroupPieces,
    dims: LayerDims,
    drop: anytype,
    layer: u16,
    align_bytes: u32,
    spans: []SourceSpan,
) !GroupedSpans {
    var lg: LayerGroups = .{ .count = @intCast(plan.count), .align_bytes = align_bytes };
    var at: u32 = 0;
    var weight_total: [3]u64 = @splat(0);
    const tiles: u64 = (@as(u64, dims.hidden) / 16) * (dims.intermediate / 16);
    for (0..plan.count) |g| {
        lg.experts[g] = @intCast(plan.groups[g].experts);
        lg.base[g] = @intCast(at);
        if (drop) |d| {
            lg.dropped[g] = @intCast(d.droppedIn(layer, at, at + plan.groups[g].experts));
            if (lg.keptOf(g) == 0) return error.DropEmptiesRateGroup;
        }
        for (0..3) |pi| {
            lg.n[g][pi] = plan.groups[g].rates[pi].n;
            weight_total[pi] += @as(u64, lg.keptOf(g)) * tiles * lg.n[g][pi] * 2;
        }
        for (0..component_count) |ci| {
            const piece = pieces[g][ci];
            if (piece.experts != plan.groups[g].experts) return error.Exl3GroupGeometry;
            for (0..piece.experts) |local| {
                spans[(at + local) * component_count + ci] = .{
                    .file = piece.file,
                    .offset = std.math.add(u64, piece.base, @as(u64, @intCast(local)) * piece.per_expert) catch return error.InvalidExpertTensor,
                    .len = piece.per_expert,
                };
            }
        }
        at += plan.groups[g].experts;
    }
    return .{ .groups = lg, .weight_total = weight_total };
}

const TestDrop = struct {
    from: usize,
    to: usize,

    fn droppedIn(self: *const TestDrop, layer: usize, lo: usize, hi: usize) usize {
        _ = layer;
        return @min(hi, self.to) -| @max(lo, self.from);
    }
};

const projections = [_][]const u8{ "gate", "up", "down" };

test "a grouped layer is cut into one span per expert and component, groups in order" {
    const t = std.testing;
    var plan = gl.Layout.init(128, 128, .{ .n = 64 });
    var pieces: GroupPieces = @splat(@splat(.{}));
    for ([_]u32{ 2, 1 }, 0..) |e, g| {
        for (0..3) |p| {
            const n: u32 = @intCast(32 + 16 * ((g + p) % 3));
            const proj = projections[p];
            var buf: [96]u8 = undefined;
            _ = try plan.add(try std.fmt.bufPrint(&buf, "{s}_proj.g{d}.trellis", .{ proj, g }), &.{ e, 8, 8, n }, .u16);
            _ = try plan.add(try std.fmt.bufPrint(&buf, "{s}_proj.g{d}.suh", .{ proj, g }), &.{ e, 128 }, .f16);
            _ = try plan.add(try std.fmt.bufPrint(&buf, "{s}_proj.g{d}.svh", .{ proj, g }), &.{ e, 128 }, .f16);
            for (0..3) |part| pieces[g][p * 3 + part] = .{ .file = @intCast(g), .base = 1000 * (g + 1) + 100 * (p * 3 + part), .per_expert = 10, .experts = e };
        }
    }
    try plan.finish(3, 2, false, 3);
    var spans: [3 * component_count]SourceSpan = undefined;
    const dims: LayerDims = .{ .experts = 3, .hidden = 128, .intermediate = 128 };
    const none: ?*const TestDrop = null;
    const out = try groupedLayerSpans(&plan, &pieces, dims, none, 5, 0, &spans);
    try t.expectEqual(@as(u8, 2), out.groups.count);
    try t.expectEqual(@as(u16, 2), out.groups.base[1]);
    try t.expectEqual(@as(u32, 32), out.groups.n[0][0]);
    try t.expectEqual(@as(u32, 48), out.groups.n[1][0]);
    try t.expectEqual(@as(u64, (2 * 32 + 48) * 64 * 2), out.weight_total[0]);
    // Expert 1 is the second of group 0, expert 2 the first of group 1.
    try t.expectEqual(SourceSpan{ .file = 0, .offset = 1010, .len = 10 }, spans[1 * component_count + 0]);
    try t.expectEqual(SourceSpan{ .file = 1, .offset = 2100, .len = 10 }, spans[2 * component_count + 1]);

    const one_gone: TestDrop = .{ .from = 1, .to = 2 };
    const dropped = try groupedLayerSpans(&plan, &pieces, dims, @as(?*const TestDrop, &one_gone), 5, 0, &spans);
    try t.expectEqual(@as(u16, 1), dropped.groups.dropped[0]);
    try t.expectEqual(@as(u64, (1 * 32 + 48) * 64 * 2), dropped.weight_total[0]);
    const all_gone: TestDrop = .{ .from = 2, .to = 3 };
    try t.expectError(error.DropEmptiesRateGroup, groupedLayerSpans(&plan, &pieces, dims, @as(?*const TestDrop, &all_gone), 5, 0, &spans));

    pieces[1][0].experts = 2;
    try t.expectError(error.Exl3GroupGeometry, groupedLayerSpans(&plan, &pieces, dims, none, 5, 0, &spans));
}

test "sushi coder index recognizes grouped qwen4 and mimo layouts" {
    const t = std.testing;
    for (EXL3_PREFIXES) |prefix| {
        var map: std.json.ObjectMap = .empty;
        defer {
            for (map.keys()) |key| t.allocator.free(key);
            map.deinit(t.allocator);
        }
        for (0..2) |layer| {
            for (0..2) |group| {
                for ([_][]const u8{ "gate", "up", "down" }) |proj| {
                    for ([_][]const u8{ "trellis", "suh", "svh" }) |part| {
                        const key = try std.fmt.allocPrint(t.allocator, "{s}{d}.mlp.switch_mlp.{s}_proj.g{d}.{s}", .{ prefix, layer, proj, group, part });
                        try map.put(t.allocator, key, .{ .string = "experts.safetensors" });
                    }
                }
            }
        }
        try t.expect(exl3BankComplete(map, prefix, 0, 2));
        try t.expect(hasAnyExl3Key(map, 2));
    }
}
