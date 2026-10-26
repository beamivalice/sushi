//! A pack's per-layer routed-expert usage (`expert_usage.json`) and the drop `--drop-expert n` ranks from it.
const std = @import("std");

pub const file_name = "expert_usage.json";

/// A drop is a multiple of this. The router kernel pads to 32 lanes with a bounds check and the ragged pools take any
/// slot count; the one expert-count tile left is `gateLogits`' 8 experts per threadgroup, so a multiple of 8 keeps
/// the kept width on whole threadgroups. 16 is a multiple of it.
pub const granularity: u16 = 8;

/// The widest router a drop covers (the window kernels hold one thread per expert).
pub const max_experts = 512;
const words = max_experts / 64;

pub const Error = error{
    ExpertUsageMissing,
    ExpertUsageMalformed,
    ExpertUsageVersion,
    ExpertUsageShape,
    DropExpertTooMany,
    DropRouterTooWide,
};

/// What a pack's router settles: more than `top_k` experts must stay in a layer.
pub fn checkRange(n: u32, experts: u32, top_k: u32) Error!void {
    if (n == 0) return;
    if (experts > max_experts) return error.DropRouterTooWide;
    if (n + top_k >= experts) return error.DropExpertTooMany;
}

/// The flag's own parse: a whole number of experts a layer that is a multiple of `granularity`, 0 = off.
pub fn parseCount(raw: []const u8) ?u16 {
    const n = std.fmt.parseInt(u16, raw, 10) catch return null;
    return if (n % granularity == 0) n else null;
}

pub const Usage = struct {
    allocator: std.mem.Allocator,
    source: []u8,
    /// `[routed layer][expert id]`.
    counts: [][]u64,

    pub fn deinit(self: *Usage) void {
        for (self.counts) |row| self.allocator.free(row);
        self.allocator.free(self.counts);
        self.allocator.free(self.source);
    }
};

/// A version-1 file with one row of `experts` counts for each of `layers` routed layers.
pub fn parse(allocator: std.mem.Allocator, raw: []const u8, layers: usize, experts: usize) !Usage {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{}) catch return error.ExpertUsageMalformed;
    defer parsed.deinit();
    if (parsed.value != .object) return error.ExpertUsageMalformed;
    const obj = parsed.value.object;
    const version = obj.get("version") orelse return error.ExpertUsageMalformed;
    if (version != .integer or version.integer != 1) return error.ExpertUsageVersion;
    const rows = obj.get("layers") orelse return error.ExpertUsageMalformed;
    if (rows != .array) return error.ExpertUsageMalformed;
    if (rows.array.items.len != layers) return error.ExpertUsageShape;
    const source = try allocator.dupe(u8, if (obj.get("source")) |s| (if (s == .string) s.string else "") else "");
    errdefer allocator.free(source);
    const counts = try allocator.alloc([]u64, layers);
    var made: usize = 0;
    errdefer {
        for (counts[0..made]) |row| allocator.free(row);
        allocator.free(counts);
    }
    for (rows.array.items) |row| {
        if (row != .array) return error.ExpertUsageMalformed;
        if (row.array.items.len != experts) return error.ExpertUsageShape;
        counts[made] = try allocator.alloc(u64, experts);
        made += 1;
        for (row.array.items, counts[made - 1]) |v, *c| {
            if (v != .integer or v.integer < 0) return error.ExpertUsageMalformed;
            c.* = @intCast(v.integer);
        }
    }
    return .{ .allocator = allocator, .source = source, .counts = counts };
}

/// The experts of `counts` from least to most used, ties by expert id.
pub fn rank(allocator: std.mem.Allocator, counts: []const u64) ![]u16 {
    const order = try allocator.alloc(u16, counts.len);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    const By = struct {
        counts: []const u64,
        fn less(ctx: @This(), a: u16, b: u16) bool {
            return if (ctx.counts[a] != ctx.counts[b]) ctx.counts[a] < ctx.counts[b] else a < b;
        }
    };
    std.mem.sort(u16, order, By{ .counts = counts }, By.less);
    return order;
}

/// The `n` least-used experts of every routed layer, as one bit row a layer.
pub const Drop = struct {
    allocator: std.mem.Allocator,
    n: u16,
    first_layer: u16,
    rows: [][words]u64,
    /// Where the ranking came from, as the file states it.
    source: []u8,

    pub fn build(allocator: std.mem.Allocator, usage: *const Usage, n: u16, first_layer: u16) !Drop {
        const rows = try allocator.alloc([words]u64, usage.counts.len);
        errdefer allocator.free(rows);
        const source = try allocator.dupe(u8, usage.source);
        errdefer allocator.free(source);
        for (rows, usage.counts) |*row, counts| {
            if (counts.len > max_experts) return error.DropRouterTooWide;
            if (n >= counts.len) return error.DropExpertTooMany;
            row.* = @splat(0);
            const order = try rank(allocator, counts);
            defer allocator.free(order);
            for (order[0..n]) |e| row[e / 64] |= @as(u64, 1) << @intCast(e % 64);
        }
        return .{ .allocator = allocator, .n = n, .first_layer = first_layer, .rows = rows, .source = source };
    }

    pub fn deinit(self: *Drop) void {
        self.allocator.free(self.rows);
        self.allocator.free(self.source);
    }

    /// Names the dropped set: two drops with the same digest leave out the same experts in every layer.
    pub fn digest(self: *const Drop) u64 {
        var h = std.hash.XxHash64.init(0x6472_6f70_6578_7031);
        h.update(std.mem.asBytes(&self.n));
        h.update(std.mem.asBytes(&self.first_layer));
        h.update(std.mem.sliceAsBytes(self.rows));
        return h.final();
    }

    /// How many of `lo..hi` this layer drops.
    pub fn droppedIn(self: *const Drop, layer: usize, lo: usize, hi: usize) usize {
        var count: usize = 0;
        for (lo..hi) |e| count += @intFromBool(self.isDropped(layer, e));
        return count;
    }

    pub fn isDropped(self: *const Drop, layer: usize, expert: usize) bool {
        if (layer < self.first_layer or layer - self.first_layer >= self.rows.len or expert >= max_experts) return false;
        return self.rows[layer - self.first_layer][expert / 64] >> @intCast(expert % 64) & 1 != 0;
    }
};

/// A drop that leaves out exactly `ids[row]` of each routed layer, for tests that need a known set.
pub fn fixedDrop(allocator: std.mem.Allocator, first_layer: u16, ids: []const []const u16) !Drop {
    const rows = try allocator.alloc([words]u64, ids.len);
    errdefer allocator.free(rows);
    for (rows, ids) |*row, row_ids| {
        row.* = @splat(0);
        for (row_ids) |e| row[e / 64] |= @as(u64, 1) << @intCast(e % 64);
    }
    const source = try allocator.dupe(u8, "fixed");
    return .{ .allocator = allocator, .n = @intCast(ids[0].len), .first_layer = first_layer, .rows = rows, .source = source };
}

/// The pack the flag is asked of: `layers` routed layers from `first_layer`, `experts` routed experts a layer.
pub const Request = struct { n: u16, layers: usize, first_layer: u16, experts: u32, top_k: u32 };

/// The drop of `req.n` experts a layer from the pack's `expert_usage.json`; every refusal is an error, before any
/// weight loads.
pub fn load(allocator: std.mem.Allocator, model_dir: []const u8, req: Request) !Drop {
    try checkRange(req.n, req.experts, req.top_k);
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return error.ExpertUsageMissing;
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, file_name, allocator, .limited(16 * 1024 * 1024)) catch return error.ExpertUsageMissing;
    defer allocator.free(raw);
    var usage = try parse(allocator, raw, req.layers, req.experts);
    defer usage.deinit();
    return Drop.build(allocator, &usage, req.n, req.first_layer);
}

/// The refusal of `load`, worded for the log.
pub fn describe(err: anyerror) []const u8 {
    return switch (err) {
        error.DropExpertTooMany => "--drop-expert would leave this model's router no more than its top-k experts a layer",
        error.DropRouterTooWide => "--drop-expert covers routers up to 512 experts",
        error.ExpertUsageMissing => "--drop-expert needs the pack's expert_usage.json, which this pack does not carry",
        error.ExpertUsageVersion => "expert_usage.json: only version 1 is read",
        error.ExpertUsageShape => "expert_usage.json does not hold one row of counts per routed layer and expert",
        error.ExpertUsageMalformed => "expert_usage.json is not a version-1 usage file",
        else => "--drop-expert could not read the pack's expert_usage.json",
    };
}

const testing = std.testing;

test "expert usage: a pack's router must keep more than top_k experts a layer and fit the window kernels" {
    try checkRange(0, 384, 6);
    try checkRange(8, 384, 6);
    try checkRange(16, 384, 6);
    try checkRange(376, 384, 6);
    try testing.expectError(error.DropExpertTooMany, checkRange(378, 384, 6));
    try testing.expectError(error.DropExpertTooMany, checkRange(384, 384, 6));
    try testing.expectError(error.DropRouterTooWide, checkRange(8, 1024, 6));
    try checkRange(0, 1024, 6);
}

test "expert usage: the flag value parses a multiple of the granularity and nothing else" {
    try testing.expectEqual(@as(?u16, 16), parseCount("16"));
    try testing.expectEqual(@as(?u16, 0), parseCount("0"));
    try testing.expectEqual(@as(?u16, null), parseCount("12"));
    try testing.expectEqual(@as(?u16, null), parseCount("-8"));
    try testing.expectEqual(@as(?u16, null), parseCount("eight"));
    try testing.expectEqual(@as(?u16, null), parseCount("70000"));
}

test "expert usage: ranking is count ascending with ties by expert id" {
    const order = try rank(testing.allocator, &.{ 5, 0, 3, 0, 3, 9 });
    defer testing.allocator.free(order);
    try testing.expectEqualSlices(u16, &.{ 1, 3, 2, 4, 0, 5 }, order);
}

const sample =
    \\{"version":1,"source":"tune coverage","tokens":100,"top_k":2,"layers":[[4,1,1,9],[0,7,7,0]]}
;

test "expert usage: the file parses to one count row per routed layer and refuses any other shape" {
    var u = try parse(testing.allocator, sample, 2, 4);
    defer u.deinit();
    try testing.expectEqualStrings("tune coverage", u.source);
    try testing.expectEqualSlices(u64, &.{ 0, 7, 7, 0 }, u.counts[1]);
    try testing.expectError(error.ExpertUsageShape, parse(testing.allocator, sample, 3, 4));
    try testing.expectError(error.ExpertUsageShape, parse(testing.allocator, sample, 2, 5));
    try testing.expectError(error.ExpertUsageVersion, parse(testing.allocator, "{\"version\":2,\"layers\":[]}", 0, 4));
    try testing.expectError(error.ExpertUsageMalformed, parse(testing.allocator, "{\"version\":1,\"layers\":[[1,-2,3,4]]}", 1, 4));
    try testing.expectError(error.ExpertUsageMalformed, parse(testing.allocator, "not json", 1, 4));
}

test "expert usage: the drop is the n least-used experts of each layer, offset by the first routed layer" {
    var u = try parse(testing.allocator, sample, 2, 4);
    defer u.deinit();
    var d = try Drop.build(testing.allocator, &u, 2, 3);
    defer d.deinit();
    // Layer 3 counts 4,1,1,9: ids 1 and 2 are least used. Layer 4 counts 0,7,7,0: ids 0 and 3, ties by id.
    for ([_]bool{ false, true, true, false }, 0..) |want, e| try testing.expectEqual(want, d.isDropped(3, e));
    for ([_]bool{ true, false, false, true }, 0..) |want, e| try testing.expectEqual(want, d.isDropped(4, e));
    try testing.expect(!d.isDropped(2, 1));
    try testing.expect(!d.isDropped(5, 0));
    try testing.expect(!d.isDropped(3, 4000));
}

test "expert usage: load reads the pack's file, and every refusal is an error before any weight loads" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    const req: Request = .{ .n = 8, .layers = 2, .first_layer = 1, .experts = 16, .top_k = 2 };
    try testing.expectError(error.ExpertUsageMissing, load(testing.allocator, root, req));
    // Layer row 0 counts expert e as e, row 1 as 15 - e: the eight least used are ids 0..7 and 8..15.
    const file = "{\"version\":1,\"source\":\"unit\",\"layers\":[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]]}";
    try tmp.dir.writeFile(io, .{ .sub_path = file_name, .data = file });
    var d = try load(testing.allocator, root, req);
    defer d.deinit();
    for (0..16) |e| {
        try testing.expectEqual(e < 8, d.isDropped(1, e));
        try testing.expectEqual(e >= 8, d.isDropped(2, e));
    }
    try testing.expectEqualStrings("unit", d.source);
    var bad = req;
    bad.n = 14;
    try testing.expectError(error.DropExpertTooMany, load(testing.allocator, root, bad));
    bad = req;
    bad.layers = 3;
    try testing.expectError(error.ExpertUsageShape, load(testing.allocator, root, bad));
    bad = req;
    bad.top_k = 8;
    try testing.expectError(error.DropExpertTooMany, load(testing.allocator, root, bad));
}
