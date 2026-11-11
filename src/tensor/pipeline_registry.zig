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
    return .{ .kind = r.kind, .function = function, .key = key, .text = text, .flags = flags };
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
