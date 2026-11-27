//! The recipe of every pipeline the engine made or took from a manifest in this process, and the pipelines a manifest
//! made ahead of their first use. The inference thread only.

const std = @import("std");
const metal = @import("../metal.zig");
const manifest = @import("../pipeline_manifest.zig");

const Recipe = manifest.Recipe;
/// Every table here is the process's, so a test counts bytes through this one allocator, never a second heap.
var gpa: std.mem.Allocator = std.heap.c_allocator;

var recorded: std.ArrayList(Recipe) = .empty;

/// Recipes that could not be kept: a manifest written without them would be short, so none is written.
pub var lost: u32 = 0;

pub fn recipes() []const Recipe {
    return recorded.items;
}

fn dupeRecipe(r: Recipe) !Recipe {
    const function = try gpa.dupe(u8, r.function);
    errdefer gpa.free(function);
    const key = try gpa.dupe(u8, r.key);
    errdefer gpa.free(key);
    const text = try gpa.dupe(u8, r.text);
    errdefer gpa.free(text);
    const flags = try gpa.dupe(metal.Flag, r.flags);
    errdefer gpa.free(flags);
    const ints = try gpa.dupe(metal.IntConstant, r.ints);
    return .{ .kind = r.kind, .function = function, .key = key, .text = text, .flags = flags, .ints = ints };
}

fn freeRecipe(r: Recipe) void {
    gpa.free(r.function);
    gpa.free(r.key);
    gpa.free(r.text);
    gpa.free(r.flags);
    gpa.free(r.ints);
}

fn has(r: Recipe) bool {
    for (recorded.items) |x| if (x.eql(r)) return true;
    return false;
}

pub fn record(r: Recipe) void {
    if (has(r)) return;
    const owned = dupeRecipe(r) catch {
        lost += 1;
        return;
    };
    recorded.append(gpa, owned) catch {
        lost += 1;
    };
}

/// Made ahead from a manifest, by its table key, with the recipe to record when it is first used.
var stock: std.StringHashMapUnmanaged(struct { pipeline: metal.Pipeline, recipe: Recipe }) = .empty;

pub fn putStock(r: Recipe, pipeline: metal.Pipeline) !bool {
    if (stock.contains(r.key)) return false;
    const owned = try dupeRecipe(r);
    errdefer freeRecipe(owned);
    try stock.putNoClobber(gpa, owned.key, .{ .pipeline = pipeline, .recipe = owned });
    return true;
}

pub fn inStock(key: []const u8) bool {
    return stock.contains(key);
}

pub fn takeStock(key: []const u8) ?metal.Pipeline {
    const e = stock.get(key) orelse return null;
    record(e.recipe);
    return e.pipeline;
}

/// Markers a manifest held, not yet trusted: `Plan.settle` accepts them once every recipe of that manifest has made.
var pending_markers: std.ArrayList(Recipe) = .empty;
/// The markers of the manifest the load in progress settled: no other load's.
var accepted_markers: std.ArrayList(Recipe) = .empty;

pub fn putMarker(r: Recipe) !void {
    try pending_markers.ensureUnusedCapacity(gpa, 1);
    pending_markers.appendAssumeCapacity(try dupeRecipe(r));
}

/// Markers a warm recorded while no load's manifest stood settled (a vision tower warms as its weights load, before its
/// manifest opens): the next settle takes them as its own.
var early_markers: std.ArrayList(Recipe) = .empty;
/// Whether a load's manifest is settled and its model not yet unloaded.
var settled = false;

/// The model of the settled manifest unloaded: a marker recorded from now belongs to the next load's manifest.
pub fn manifestClosed() void {
    settled = false;
}

/// One load's settle: its manifest's markers replace the last load's, trusted (the manifest's recipes all made) or
/// dropped. A marker a warm recorded under the last load leaves `recorded` too, so the manifest written next holds
/// this load's markers only; the early markers are this load's.
pub fn settleMarkers(trusted: bool) void {
    var i: usize = 0;
    while (i < recorded.items.len) {
        if (recorded.items[i].kind == .marker) freeRecipe(recorded.orderedRemove(i)) else i += 1;
    }
    for (accepted_markers.items) |m| freeRecipe(m);
    accepted_markers.clearRetainingCapacity();
    if (trusted) {
        accepted_markers.appendSlice(gpa, pending_markers.items) catch {};
        for (pending_markers.items) |m| record(m);
    } else for (pending_markers.items) |m| freeRecipe(m);
    pending_markers.clearRetainingCapacity();
    settled = true;
    for (early_markers.items) |m| {
        record(m);
        freeRecipe(m);
    }
    early_markers.clearRetainingCapacity();
}

/// Whether a trusted manifest holds the marker `key`: the warm it names was made whole by recipes already made.
pub fn takeMarker(key: []const u8) bool {
    for (accepted_markers.items) |m| if (std.mem.eql(u8, m.key, key)) return true;
    return false;
}

/// The warm named `key` has run in this process: its recipes are recorded, so this marks them whole.
pub fn recordMarker(function: []const u8, key: []const u8) void {
    const r: Recipe = .{ .kind = .marker, .function = function, .key = key };
    if (settled) return record(r);
    const owned = dupeRecipe(r) catch {
        lost += 1;
        return;
    };
    early_markers.append(gpa, owned) catch {
        freeRecipe(owned);
        lost += 1;
    };
}

/// TEST-ONLY: the process's recipes forgotten, as a new process has none; pipelines already made stay made.
pub fn resetForTest() void {
    comptime std.debug.assert(@import("builtin").is_test);
    recorded.clearRetainingCapacity();
    lost = 0;
    pending_markers.clearRetainingCapacity();
    accepted_markers.clearRetainingCapacity();
    for (early_markers.items) |m| freeRecipe(m);
    early_markers.clearRetainingCapacity();
    settled = false;
}

// ── Tests ──

const testing = std.testing;

fn markerRecipe(key: []const u8) Recipe {
    return .{ .kind = .marker, .function = "served warm", .key = key };
}

fn markerRecorded(key: []const u8) bool {
    for (recipes()) |r| if (r.kind == .marker and std.mem.eql(u8, r.key, key)) return true;
    return false;
}

test "pipeline registry: a marker a trusted manifest held is not taken under a later load whose own manifest never held it" {
    resetForTest();
    defer resetForTest();
    try putMarker(markerRecipe("K"));
    settleMarkers(true);
    try testing.expect(takeMarker("K"));
    // The next load's manifest made every recipe and held no marker.
    settleMarkers(true);
    try testing.expect(!takeMarker("K"));
}

test "pipeline registry: a load whose manifest is not trusted, or whose workers did not start, takes no marker of the load before it" {
    resetForTest();
    defer resetForTest();
    try putMarker(markerRecipe("K"));
    settleMarkers(true);
    settleMarkers(false);
    try testing.expect(!takeMarker("K"));
    // Markers seen in a manifest that did not make whole are dropped, not kept for the next settle.
    try putMarker(markerRecipe("L"));
    settleMarkers(false);
    settleMarkers(true);
    try testing.expect(!takeMarker("L"));
}

test "pipeline registry: a marker one load's warm recorded is not among the recipes the next load's manifest is written with" {
    resetForTest();
    defer resetForTest();
    record(.{ .kind = .stock, .function = "kernel_a", .key = "kernel_a" });
    // Load 1 settled its manifest and its warm ran whole.
    settleMarkers(false);
    recordMarker("served warm", "K1");
    try testing.expect(markerRecorded("K1"));
    // Load 2, another scope: its manifest holds one marker of its own. Pipelines are the process's; markers are the manifest's.
    try putMarker(markerRecipe("K2"));
    settleMarkers(true);
    try testing.expect(markerRecorded("K2"));
    try testing.expect(!markerRecorded("K1"));
    try testing.expect(takeMarker("K2"));
    try testing.expect(!takeMarker("K1"));
    var has_stock = false;
    for (recipes()) |r| has_stock = has_stock or r.kind == .stock;
    try testing.expect(has_stock);
}

test "pipeline registry: a marker recorded while no manifest stood settled is the next settle's, and an unload closes the manifest" {
    resetForTest();
    defer resetForTest();
    // A vision tower warms as the weights load, before the load's manifest opens.
    recordMarker("vision tower", "T");
    settleMarkers(false);
    try testing.expect(markerRecorded("T"));
    // The model unloads; the tower of the next load records before its settle, which drops the last load's markers.
    recordMarker("served warm", "K");
    manifestClosed();
    recordMarker("vision tower", "T2");
    try testing.expect(markerRecorded("K"));
    settleMarkers(true);
    try testing.expect(!markerRecorded("K"));
    try testing.expect(markerRecorded("T2"));
}

/// Bytes the registry's allocator holds live, counted over the process's own heap.
const Live = struct {
    var bytes: usize = 0;
    const vt: std.mem.Allocator.VTable = .{ .alloc = alloc, .resize = resize, .remap = remap, .free = free };
    fn alloc(_: *anyopaque, len: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        const p = std.heap.c_allocator.rawAlloc(len, al, ra) orelse return null;
        bytes += len;
        return p;
    }
    fn resize(_: *anyopaque, mem: []u8, al: std.mem.Alignment, new_len: usize, ra: usize) bool {
        if (!std.heap.c_allocator.rawResize(mem, al, new_len, ra)) return false;
        bytes = bytes - mem.len + new_len;
        return true;
    }
    fn remap(_: *anyopaque, mem: []u8, al: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const p = std.heap.c_allocator.rawRemap(mem, al, new_len, ra) orelse return null;
        bytes = bytes - mem.len + new_len;
        return p;
    }
    fn free(_: *anyopaque, mem: []u8, al: std.mem.Alignment, ra: usize) void {
        std.heap.c_allocator.rawFree(mem, al, ra);
        bytes -= mem.len;
    }
};

test "pipeline registry: a pipeline put again, as a like-for-like reload's manifest does, holds no second copy of its recipe" {
    const was = gpa;
    defer gpa = was;
    gpa = .{ .ptr = undefined, .vtable = &Live.vt };
    const fake: metal.Pipeline = .{ .raw = undefined, .max_threads = 1 };
    const plain: Recipe = .{ .kind = .stock, .function = "reload_probe_plain", .key = "reload_probe_plain" };
    const stocked: Recipe = .{ .kind = .stock_flags, .function = "reload_probe_stock", .key = "reload_probe_stock#1", .flags = &.{.{ .index = 1, .value = true }}, .ints = &.{.{ .index = 2, .value = 7 }} };

    try testing.expect(try putStock(plain, fake));
    try testing.expect(try putStock(stocked, fake));
    const held = Live.bytes;
    var load: usize = 0;
    while (load < 3) : (load += 1) {
        try testing.expect(!try putStock(plain, fake));
        try testing.expect(!try putStock(stocked, fake));
    }
    try testing.expectEqual(held, Live.bytes);
}
