//! Makes the pipelines of a manifest's recipes on worker threads, then hands them to the engine's tables. The workers
//! call only our own Metal binding, never MLX; the tables are filled by the inference thread, in `finish`.

const std = @import("std");
const metal = @import("../metal.zig");
const manifest = @import("../pipeline_manifest.zig");
const registry = @import("pipeline_registry.zig");
const core = @import("core.zig");
const stock = @import("stock.zig");

const Recipe = manifest.Recipe;
const gpa = std.heap.c_allocator;

pub const Report = struct {
    /// Pipelines now waiting for their first launch.
    made: usize = 0,
    failed: usize = 0,
    first_error: [160]u8 = undefined,
    first_error_len: usize = 0,

    pub fn firstError(r: *const Report) []const u8 {
        return r.first_error[0..r.first_error_len];
    }
};

fn zdup(bytes: []const u8) ![:0]u8 {
    const out = try gpa.allocSentinel(u8, bytes.len, 0);
    @memcpy(out, bytes);
    return out;
}

pub const Job = struct {
    recipes: []const Recipe,
    pipelines: []?metal.Pipeline,
    lib: ?metal.Library,
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(usize) = .init(0),
    error_claimed: std.atomic.Value(bool) = .init(false),
    first_error: [160]u8 = undefined,
    first_error_len: usize = 0,
    handles: []std.Thread,
    started: usize = 0,

    fn fail(job: *Job, what: []const u8) void {
        _ = job.failed.fetchAdd(1, .monotonic);
        if (job.error_claimed.swap(true, .acq_rel)) return;
        const text = metal.lastError();
        const shown = if (text.len > 0) text else what;
        const n = @min(shown.len, job.first_error.len);
        @memcpy(job.first_error[0..n], shown[0..n]);
        job.first_error_len = n;
    }

    fn work(job: *Job) void {
        const dev = metal.device() catch {
            job.fail("no Metal device");
            return;
        };
        while (true) {
            const i = job.next.fetchAdd(1, .monotonic);
            if (i >= job.recipes.len) return;
            if (job.recipes[i].kind == .marker) continue;
            job.pipelines[i] = job.make(dev, job.recipes[i]) catch {
                job.fail(job.recipes[i].function);
                continue;
            };
        }
    }

    fn make(job: *Job, dev: *const metal.Device, r: Recipe) !metal.Pipeline {
        const function = try zdup(r.function);
        defer gpa.free(function);
        switch (r.kind) {
            .source => {
                const text = try zdup(r.text);
                defer gpa.free(text);
                return dev.compile(text, function);
            },
            .stock => return dev.pipelineOf(job.lib orelse return error.MetalUnavailable, function),
            .stock_flags => return dev.pipelineOfConstants(job.lib orelse return error.MetalUnavailable, function, r.flags, r.ints),
            .marker => unreachable,
        }
    }
};

/// Starts making the pipelines of `recipes` on up to `threads` threads. The recipes outlive `finish`.
pub fn start(recipes: []const Recipe, threads: usize) !*Job {
    const dev = try metal.device();
    var needs_lib = false;
    for (recipes) |r| needs_lib = needs_lib or r.kind != .source;
    const job = try gpa.create(Job);
    errdefer gpa.destroy(job);
    const results = try gpa.alloc(?metal.Pipeline, recipes.len);
    errdefer gpa.free(results);
    @memset(results, null);
    const handles = try gpa.alloc(std.Thread, @max(1, @min(threads, recipes.len)));
    errdefer gpa.free(handles);
    job.* = .{ .recipes = recipes, .pipelines = results, .lib = if (needs_lib) try stock.lib(dev) else null, .handles = handles };
    for (handles) |*h| {
        h.* = std.Thread.spawn(.{}, Job.work, .{job}) catch break;
        job.started += 1;
    }
    // Workers that could not start leave their share to the ones that did; with none, this thread does it all.
    if (job.started == 0) Job.work(job);
    return job;
}

/// Waits for the workers and installs what they made. A recipe that did not make is counted and named, not installed.
pub fn finish(job: *Job) Report {
    for (job.handles[0..job.started]) |t| t.join();
    defer {
        gpa.free(job.handles);
        gpa.free(job.pipelines);
        gpa.destroy(job);
    }
    var report: Report = .{ .failed = job.failed.load(.acquire) };
    @memcpy(report.first_error[0..job.first_error_len], job.first_error[0..job.first_error_len]);
    report.first_error_len = job.first_error_len;
    for (job.recipes, job.pipelines) |r, made| {
        const p = made orelse {
            if (r.kind == .marker) registry.putMarker(r) catch {
                report.failed += 1;
            };
            continue;
        };
        const kept = switch (r.kind) {
            .source => registry.putSource(r, p),
            .stock, .stock_flags => stock.preload(r, p),
            .marker => unreachable,
        } catch {
            report.failed += 1;
            continue;
        };
        // The same pipeline was already there: this copy is not needed.
        if (!kept) p.release();
        report.made += 1;
    }
    return report;
}

// ── Tests ──

const testing = std.testing;
const kernel = @import("kernel.zig");

fn launchOnce(name: []const u8, source: []const u8) !void {
    try @import("../mlx.zig").requireGpu();
    const def = try kernel.Def.create(name, &.{"x"}, &.{"out"}, source, "", true, false);
    const x = try core.fromHost(std.mem.sliceAsBytes(&[_]f32{ 1, 2, 3, 4 }), .float32, &.{4});
    defer core.release(x);
    var cfg: kernel.Config = .{};
    try cfg.addOutput(&.{4}, .float32);
    cfg.grid = .{ 4, 1, 1 };
    cfg.group = .{ 4, 1, 1 };
    var out: [1]*core.Tensor = undefined;
    try kernel.launch(def, &.{x}, &cfg, &out);
    defer core.release(out[0]);
    const bytes = try core.hostBytes(out[0]);
    const got = std.mem.bytesAsSlice(f32, bytes);
    try testing.expectEqual(@as(f32, 2), got[0]);
}

test "pipeline cache: a source kernel made from its recipe is what a launch finds, and nothing compiles" {
    const source = "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] * 2.0f; // pipeline cache probe";
    const before = core.pipelines;
    try launchOnce("pc_probe", source);
    try testing.expectEqual(before + 1, core.pipelines);

    var found: ?Recipe = null;
    for (registry.recipes()) |r| {
        if (r.kind == .source and std.mem.indexOf(u8, r.function, "pc_probe") != null) found = r;
    }
    const recipe = found orelse return error.RecipeNotRecorded;
    const bytes = try manifest.encode(testing.allocator, "k", &.{recipe});
    defer testing.allocator.free(bytes);
    var loaded = try manifest.decode(testing.allocator, bytes);
    defer loaded.deinit();

    const report = finish(try start(loaded.recipes, 2));
    try testing.expectEqual(@as(usize, 1), report.made);
    try testing.expectEqual(@as(usize, 0), report.failed);

    // A second maker of the same source, as a new process has: its first launch finds the pipeline waiting.
    const made = core.pipelines;
    const late = core.pipelines_late;
    const preloaded = core.pipelines_preloaded;
    const was_sealed = core.sealed;
    core.sealed = true;
    defer core.sealed = was_sealed;
    try launchOnce("pc_probe", source);
    try testing.expectEqual(made, core.pipelines);
    try testing.expectEqual(late, core.pipelines_late);
    try testing.expectEqual(preloaded + 1, core.pipelines_preloaded);
}

test "pipeline cache: a stock kernel made from its recipe is the pipeline kernel() answers" {
    try @import("../mlx.zig").requireGpu();
    const name = "rms_loopedfloat16";
    const recipe: Recipe = .{ .kind = .stock, .function = name, .key = name };
    const report = finish(try start(&.{recipe}, 1));
    try testing.expectEqual(@as(usize, 1), report.made);
    const made = core.pipelines;
    const preloaded = core.pipelines_preloaded;
    _ = try stock.kernel(name);
    try testing.expectEqual(made, core.pipelines);
    try testing.expectEqual(preloaded + 1, core.pipelines_preloaded);
    var recorded = false;
    for (registry.recipes()) |r| recorded = recorded or (r.kind == .stock and std.mem.eql(u8, r.key, name));
    try testing.expect(recorded);
}

test "pipeline cache: a recipe that does not compile is counted and named, and the others are still made" {
    try @import("../mlx.zig").requireGpu();
    const good: Recipe = .{ .kind = .stock, .function = "rms_loopedbfloat16", .key = "rms_loopedbfloat16" };
    const bad: Recipe = .{ .kind = .source, .function = "broken", .text = "[[kernel]] void broken(uint3 at [[thread_position_in_grid]]) { not_a_thing(at); }" };
    const missing: Recipe = .{ .kind = .stock, .function = "no_such_stock_function", .key = "no_such_stock_function" };
    const report = finish(try start(&.{ good, bad, missing }, 3));
    try testing.expectEqual(@as(usize, 1), report.made);
    try testing.expectEqual(@as(usize, 2), report.failed);
    try testing.expect(report.firstError().len > 0);
}
