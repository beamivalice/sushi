//! The recipe of every pipeline the engine made or took from a manifest in this process, and the pipelines a manifest
//! made ahead of their first use. The inference thread only.

const std = @import("std");
const metal = @import("../metal.zig");
const manifest = @import("../pipeline_manifest.zig");

const Recipe = manifest.Recipe;
const gpa = std.heap.c_allocator;

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

const SourceKey = struct { function: []const u8, text: []const u8 };

const SourceContext = struct {
    pub fn hash(_: SourceContext, k: SourceKey) u64 {
        var h = std.hash.Wyhash.init(k.function.len);
        h.update(k.function);
        h.update(k.text);
        return h.final();
    }
    pub fn eql(_: SourceContext, a: SourceKey, b: SourceKey) bool {
        return std.mem.eql(u8, a.function, b.function) and std.mem.eql(u8, a.text, b.text);
    }
};

/// Made ahead from a manifest, by what a launch will ask for: the entry point and the exact source.
var sources: std.HashMapUnmanaged(SourceKey, metal.Pipeline, SourceContext, 80) = .empty;
/// The same for a stock kernel, by its table key, with the recipe to record when it is first used.
var stock: std.StringHashMapUnmanaged(struct { pipeline: metal.Pipeline, recipe: Recipe }) = .empty;

/// Keeps a source pipeline made ahead; false when the same one is already waiting.
pub fn putSource(r: Recipe, pipeline: metal.Pipeline) !bool {
    const owned = try dupeRecipe(r);
    const entry = try sources.getOrPut(gpa, .{ .function = owned.function, .text = owned.text });
    if (entry.found_existing) return false;
    entry.value_ptr.* = pipeline;
    return true;
}

/// The pipeline made ahead for this source, now recorded as used.
pub fn takeSource(function: []const u8, text: []const u8) ?metal.Pipeline {
    const p = sources.get(.{ .function = function, .text = text }) orelse return null;
    record(.{ .kind = .source, .function = function, .text = text });
    return p;
}

pub fn putStock(r: Recipe, pipeline: metal.Pipeline) !bool {
    const owned = try dupeRecipe(r);
    const entry = try stock.getOrPut(gpa, owned.key);
    if (entry.found_existing) return false;
    entry.value_ptr.* = .{ .pipeline = pipeline, .recipe = owned };
    return true;
}

pub fn takeStock(key: []const u8) ?metal.Pipeline {
    const e = stock.get(key) orelse return null;
    record(e.recipe);
    return e.pipeline;
}

/// Markers a manifest held, not yet trusted: `Plan.settle` accepts them once every recipe of that manifest has made.
var pending_markers: std.ArrayList(Recipe) = .empty;
var accepted_markers: std.ArrayList(Recipe) = .empty;

pub fn putMarker(r: Recipe) !void {
    try pending_markers.ensureUnusedCapacity(gpa, 1);
    pending_markers.appendAssumeCapacity(try dupeRecipe(r));
}

/// The markers seen are trusted (the manifest's recipes all made), or dropped.
pub fn settleMarkers(trusted: bool) void {
    if (trusted) accepted_markers.appendSlice(gpa, pending_markers.items) catch {};
    pending_markers.clearRetainingCapacity();
    if (!trusted) accepted_markers.clearRetainingCapacity();
}

/// Whether a trusted manifest holds the marker `key`: the warm it names was made whole by recipes already made.
/// The marker is then recorded, so a manifest written later keeps it.
pub fn takeMarker(key: []const u8) bool {
    for (accepted_markers.items) |m| if (std.mem.eql(u8, m.key, key)) {
        record(m);
        return true;
    };
    return false;
}

/// The warm named `key` has run in this process: its recipes are recorded, so this marks them whole.
pub fn recordMarker(function: []const u8, key: []const u8) void {
    record(.{ .kind = .marker, .function = function, .key = key });
}

/// TEST-ONLY: the process's recipes forgotten, as a new process has none; pipelines already made stay made.
pub fn resetForTest() void {
    comptime std.debug.assert(@import("builtin").is_test);
    recorded.clearRetainingCapacity();
    lost = 0;
    pending_markers.clearRetainingCapacity();
    accepted_markers.clearRetainingCapacity();
}
