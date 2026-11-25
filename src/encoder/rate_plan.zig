//! A rate plan: every routed expert of a layer gets its own gate/up rate and down rate. A planned layer is
//! written as rate groups (docs/pack-format.md): experts sharing a (gate/up, down) pair form one group, the
//! widest pair first (gate/up, then down), each group's experts in ascending id order, so the group order and
//! the router's row order are a function of the plan alone.
const std = @import("std");
const format = @import("sushi_exl3").format;
const max_groups = @import("sushi_exl3").group_layout.max_groups;
const convert = @import("convert.zig");
const encode = @import("encode.zig");

pub const plan_format = "exls-rate-plan-1";

pub const Error = error{ RatePlanInvalid, RatePlanRateUnsupported, RatePlanTooManyGroups } || std.mem.Allocator.Error;

pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    /// Indexed by layer; null where the plan names no rates.
    layers: []?[]const convert.Class,
    /// The widest rate any planned bank packs.
    widest: format.Rate,
    sha256: [64]u8,

    pub fn deinit(self: *Plan) void {
        self.arena.deinit();
    }

    pub fn classes(self: *const Plan, layer: u32) ?[]const convert.Class {
        return if (layer < self.layers.len) self.layers[layer] else null;
    }
};

fn rateOf(v: std.json.Value) ?format.Rate {
    const k: f64 = switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => return null,
    };
    if (!std.math.isFinite(k)) return null;
    const n = @round(k * 16.0);
    if (@abs(k * 16.0 - n) > 1e-9 or n < format.Rate.min_n or n > format.Rate.max_n) return null;
    return .{ .n = @intFromFloat(n) };
}

fn rates(a: std.mem.Allocator, v: ?std.json.Value, experts: usize) Error![]u32 {
    const arr = v orelse return error.RatePlanInvalid;
    if (arr != .array or arr.array.items.len != experts) return error.RatePlanInvalid;
    const out = try a.alloc(u32, experts);
    for (arr.array.items, out) |item, *o| {
        const rate = rateOf(item) orelse return error.RatePlanInvalid;
        if (!encode.runnableRate(rate)) return error.RatePlanRateUnsupported;
        o.* = rate.n;
    }
    return out;
}

/// The layer's groups in stored order.
pub fn classesOf(a: std.mem.Allocator, gate_up: []const u32, down: []const u32) Error![]convert.Class {
    var pairs = std.ArrayList([2]u32).empty;
    for (gate_up, down) |g, d| {
        const p = [2]u32{ g, d };
        for (pairs.items) |q| {
            if (q[0] == g and q[1] == d) break;
        } else try pairs.append(a, p);
    }
    std.mem.sortUnstable([2]u32, pairs.items, {}, struct {
        fn gt(_: void, x: [2]u32, y: [2]u32) bool {
            return x[0] > y[0] or (x[0] == y[0] and x[1] > y[1]);
        }
    }.gt);
    if (pairs.items.len > max_groups) return error.RatePlanTooManyGroups;
    const out = try a.alloc(convert.Class, pairs.items.len);
    for (pairs.items, out) |p, *c| {
        var ids = std.ArrayList(u32).empty;
        for (gate_up, down, 0..) |g, d, e| if (g == p[0] and d == p[1]) try ids.append(a, @intCast(e));
        c.* = .{ .gate_up = .{ .n = p[0] }, .down = .{ .n = p[1] }, .ids = try ids.toOwnedSlice(a) };
    }
    return out;
}

/// Original expert ids in stored order: the stored router's row i is the source router's row `perm[i]`.
pub fn permutation(allocator: std.mem.Allocator, classes: []const convert.Class) ![]u32 {
    var out = std.ArrayList(u32).empty;
    errdefer out.deinit(allocator);
    for (classes) |c| try out.appendSlice(allocator, c.ids);
    return out.toOwnedSlice(allocator);
}

pub fn parse(allocator: std.mem.Allocator, text: []const u8, experts: usize, layers: u32) Error!Plan {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    const root = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return error.RatePlanInvalid;
    if (root != .object) return error.RatePlanInvalid;
    const fmt = root.object.get("format") orelse return error.RatePlanInvalid;
    if (fmt != .string or !std.mem.eql(u8, fmt.string, plan_format)) return error.RatePlanInvalid;
    if (root.object.get("experts")) |e| if (e != .integer or e.integer != experts) return error.RatePlanInvalid;
    const by_layer = root.object.get("layers") orelse return error.RatePlanInvalid;
    if (by_layer != .object) return error.RatePlanInvalid;
    const out = try a.alloc(?[]const convert.Class, layers);
    @memset(out, null);
    var widest: format.Rate = .{ .n = 0 };
    var it = by_layer.object.iterator();
    while (it.next()) |entry| {
        const layer = std.fmt.parseInt(u32, entry.key_ptr.*, 10) catch return error.RatePlanInvalid;
        if (layer >= layers or out[layer] != null or entry.value_ptr.* != .object) return error.RatePlanInvalid;
        const row = entry.value_ptr.object;
        const gate_up = try rates(a, row.get("gate_up"), experts);
        const down = try rates(a, row.get("down"), experts);
        for ([2][]const u32{ gate_up, down }) |side| for (side) |n| {
            widest.n = @max(widest.n, n);
        };
        out[layer] = try classesOf(a, gate_up, down);
    }
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
    return .{ .arena = arena, .layers = out, .widest = widest, .sha256 = std.fmt.bytesToHex(digest, .lower) };
}

test "rate plan: groups widest pair first, experts ascending, and the router order follows them" {
    const t = std.testing;
    const text =
        \\{"format":"exls-rate-plan-1","experts":6,"rates":[1.5,2,3],
        \\ "layers":{"1":{"gate_up":[1.5,3,1.5,2,3,1.5],"down":[1.5,2,3,2,2,1.5]}}}
    ;
    var plan = try parse(t.allocator, text, 6, 3);
    defer plan.deinit();
    try t.expect(plan.classes(0) == null and plan.classes(2) == null and plan.classes(7) == null);
    const cls = plan.classes(1).?;
    try t.expectEqual(@as(usize, 4), cls.len);
    const want = [_]struct { g: u32, d: u32, ids: []const u32 }{
        .{ .g = 48, .d = 32, .ids = &.{ 1, 4 } },
        .{ .g = 32, .d = 32, .ids = &.{3} },
        .{ .g = 24, .d = 48, .ids = &.{2} },
        .{ .g = 24, .d = 24, .ids = &.{ 0, 5 } },
    };
    for (cls, want) |c, w| {
        try t.expectEqual(w.g, c.gate_up.n);
        try t.expectEqual(w.d, c.down.n);
        try t.expectEqualSlices(u32, w.ids, c.ids);
    }
    try t.expectEqual(@as(u32, 48), plan.widest.n);
    const perm = try permutation(t.allocator, cls);
    defer t.allocator.free(perm);
    try t.expectEqualSlices(u32, &.{ 1, 4, 3, 2, 0, 5 }, perm);
}

test "rate plan: a malformed plan is refused" {
    const t = std.testing;
    const bad = [_][]const u8{
        "{\"format\":\"other\",\"layers\":{}}",
        "{\"format\":\"exls-rate-plan-1\",\"experts\":3,\"layers\":{}}",
        "{\"format\":\"exls-rate-plan-1\",\"layers\":{\"0\":{\"gate_up\":[2,2,2],\"down\":[2,2,2]}}}",
        "{\"format\":\"exls-rate-plan-1\",\"layers\":{\"0\":{\"gate_up\":[2,2.01],\"down\":[2,2]}}}",
        "{\"format\":\"exls-rate-plan-1\",\"layers\":{\"5\":{\"gate_up\":[2,2],\"down\":[2,2]}}}",
        "{\"format\":\"exls-rate-plan-1\",\"layers\":{\"0\":{\"gate_up\":[2,2]}}}",
    };
    for (bad) |text| try t.expectError(error.RatePlanInvalid, parse(t.allocator, text, 2, 2));
}

test "rate plan: a rate the GPU search cannot run is refused by name, and the ends of its range are not" {
    const t = std.testing;
    const head = "{\"format\":\"exls-rate-plan-1\",\"layers\":{\"0\":{\"gate_up\":[";
    const tail = "],\"down\":[2,2]}}}";
    for ([_][]const u8{ "2.0625,2", "1,1", "1.375,2", "1.4375,2" }) |gate_up| {
        const text = try std.fmt.allocPrint(t.allocator, "{s}{s}{s}", .{ head, gate_up, tail });
        defer t.allocator.free(text);
        try t.expectError(error.RatePlanRateUnsupported, parse(t.allocator, text, 2, 2));
    }
    const text = try std.fmt.allocPrint(t.allocator, "{s}1.5,8{s}", .{ head, tail });
    defer t.allocator.free(text);
    var plan = try parse(t.allocator, text, 2, 2);
    plan.deinit();
}
