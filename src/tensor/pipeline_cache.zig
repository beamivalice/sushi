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

pub const Archive = metal.BinaryArchive;

var archive_temp_seq = std.atomic.Value(u32).init(0);

pub fn openArchive(path: [:0]const u8) !Archive {
    const dev = try metal.device();
    return dev.newBinaryArchive(path);
}

pub fn lastMetalError() []const u8 {
    return metal.lastError();
}

/// Builds and atomically writes an `MTLBinaryArchive` at `path` for every stock/specialized recipe in `recipes`.
pub fn writeArchive(io: std.Io, recipes: []const Recipe, path: [:0]const u8) !usize {
    const dev = try metal.device();
    const lib = try stock.lib(dev);
    const archive = try dev.newBinaryArchive(null);
    defer archive.release();
    var added: usize = 0;
    for (recipes) |r| {
        if (r.kind != .stock and r.kind != .stock_flags) continue;
        const function = try zdup(r.function);
        defer gpa.free(function);
        dev.addComputeToArchive(archive, lib, function, r.flags, r.ints) catch continue;
        added += 1;
    }
    if (added == 0) return 0;
    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try std.fmt.bufPrintSentinel(&tmp_buf, "{s}.{d}-{d}.tmp", .{ path, std.c.getpid(), archive_temp_seq.fetchAdd(1, .monotonic) }, 0);
    try archive.serialize(tmp_path);
    errdefer std.Io.Dir.cwd().deleteFile(io, tmp_path) catch {};
    try std.Io.Dir.cwd().rename(tmp_path, std.Io.Dir.cwd(), path, io);
    return added;
}

pub const Job = struct {
    recipes: []const Recipe,
    pipelines: []?metal.Pipeline,
    skip: []bool,
    lib: ?metal.Library,
    archive: ?Archive = null,
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
            if (job.recipes[i].kind == .marker or job.skip[i]) continue;
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
            .source => return error.MetalCompile,
            .stock => {
                const lib = job.lib orelse return error.MetalUnavailable;
                if (job.archive) |arch| return dev.pipelineFromArchive(arch, lib, function, &.{}, &.{}, false);
                return dev.pipelineOf(lib, function);
            },
            .stock_flags => {
                const lib = job.lib orelse return error.MetalUnavailable;
                if (job.archive) |arch| return dev.pipelineFromArchive(arch, lib, function, r.flags, r.ints, false);
                return dev.pipelineOfConstants(lib, function, r.flags, r.ints);
            },
            .marker => unreachable,
        }
    }
};

/// Starts making the pipelines of `recipes` on up to `threads` threads. The recipes outlive `finish`.
pub fn start(recipes: []const Recipe, threads: usize) !*Job {
    return startWithArchive(recipes, threads, null);
}

/// `start` backed by an optional `MTLBinaryArchive` (`archive` must outlive `finish`).
/// Recipes already loaded in `stock` are skipped before spawning workers so in-process reloads avoid duplicate Metal work.
pub fn startWithArchive(recipes: []const Recipe, threads: usize, archive: ?Archive) !*Job {
    const dev = try metal.device();
    const skip = try gpa.alloc(bool, recipes.len);
    errdefer gpa.free(skip);
    var needs_lib = false;
    var to_make: usize = 0;
    for (recipes, 0..) |r, i| {
        const is_stock = r.kind == .stock or r.kind == .stock_flags;
        const already = is_stock and stock.has(r.key);
        skip[i] = already;
        if (!already and r.kind != .marker) {
            to_make += 1;
            if (is_stock) needs_lib = true;
        }
    }
    const job = try gpa.create(Job);
    errdefer gpa.destroy(job);
    const results = try gpa.alloc(?metal.Pipeline, recipes.len);
    errdefer gpa.free(results);
    @memset(results, null);
    const worker_n = if (to_make == 0) 0 else @max(1, @min(threads, to_make));
    const handles = try gpa.alloc(std.Thread, worker_n);
    errdefer gpa.free(handles);
    job.* = .{ .recipes = recipes, .pipelines = results, .skip = skip, .lib = if (needs_lib) try stock.lib(dev) else null, .archive = archive, .handles = handles };
    for (handles) |*h| {
        h.* = std.Thread.spawn(.{}, Job.work, .{job}) catch break;
        job.started += 1;
    }
    // Workers that could not start leave their share to the ones that did; with none, this thread does it all.
    if (to_make > 0 and job.started == 0) Job.work(job);
    return job;
}

/// Waits for the workers and installs what they made. A recipe that did not make is counted and named, not installed.
pub fn finish(job: *Job) Report {
    for (job.handles[0..job.started]) |t| t.join();
    defer {
        gpa.free(job.handles);
        gpa.free(job.pipelines);
        gpa.free(job.skip);
        gpa.destroy(job);
    }
    var report: Report = .{ .failed = job.failed.load(.acquire) };
    @memcpy(report.first_error[0..job.first_error_len], job.first_error[0..job.first_error_len]);
    report.first_error_len = job.first_error_len;
    for (job.recipes, job.pipelines, job.skip) |r, made, skipped| {
        if (skipped) {
            report.made += 1;
            continue;
        }
        const p = made orelse {
            if (r.kind == .marker) registry.putMarker(r) catch {
                report.failed += 1;
            };
            continue;
        };
        const kept = switch (r.kind) {
            .stock, .stock_flags => stock.preload(r, p),
            .source, .marker => unreachable,
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

/// Cancels/joins a running job without installing its pipelines (for error unwind paths).
pub fn abort(job: *Job) void {
    for (job.handles[0..job.started]) |t| t.join();
    for (job.pipelines) |made| if (made) |p| p.release();
    gpa.free(job.handles);
    gpa.free(job.pipelines);
    gpa.free(job.skip);
    gpa.destroy(job);
}

// ── Tests ──

const testing = std.testing;
const kernel = @import("kernel.zig");

fn launchScaleOnce(scale: c_int) !void {
    try @import("../mlx.zig").requireGpu();
    const def = try kernel.Def.create("sushi_test_scale", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = float(x[thread_position_in_grid.x]) * float(SCALE);", "", true, false);
    const x = try core.fromHost(std.mem.sliceAsBytes(&[_]f32{ 1, 2, 3, 4 }), .float32, &.{4});
    defer core.release(x);
    var cfg: kernel.Config = .{};
    try cfg.addOutput(&.{4}, .float32);
    try cfg.addTemplate("SCALE", .{ .int = scale });
    cfg.grid = .{ 4, 1, 1 };
    cfg.group = .{ 4, 1, 1 };
    var out: [1]*core.Tensor = undefined;
    try kernel.launch(def, &.{x}, &cfg, &out);
    defer core.release(out[0]);
    const bytes = try core.hostBytes(out[0]);
    const got = std.mem.bytesAsSlice(f32, bytes);
    try testing.expectEqual(@as(f32, @floatFromInt(scale)), got[0]);
}

test "pipeline cache: a precompiled custom kernel made from its recipe is what a launch finds, and nothing compiles" {
    const before = core.pipelines;
    try launchScaleOnce(77);
    try testing.expectEqual(before + 1, core.pipelines);

    var found: ?Recipe = null;
    for (registry.recipes()) |r| {
        if (r.kind == .stock_flags and std.mem.indexOf(u8, r.key, "sushi_test_scale") != null and std.mem.indexOf(u8, r.key, "=77") != null) found = r;
    }
    const recipe = found orelse return error.RecipeNotRecorded;
    const bytes = try manifest.encode(testing.allocator, "k", &.{recipe});
    defer testing.allocator.free(bytes);
    var loaded = try manifest.decode(testing.allocator, bytes);
    defer loaded.deinit();

    // Forget the in-process cache entry so the next launch exercises manifest preload.
    stock.forget(recipe.key);
    const report = finish(try start(loaded.recipes, 2));
    try testing.expectEqual(@as(usize, 1), report.made);
    try testing.expectEqual(@as(usize, 0), report.failed);

    // A second maker of the same kernel, as a new process has: its first launch finds the pipeline waiting.
    const made = core.pipelines;
    const late = core.pipelines_late;
    const preloaded = core.pipelines_preloaded;
    const was_sealed = core.sealed;
    core.sealed = true;
    defer core.sealed = was_sealed;
    try launchScaleOnce(77);
    try testing.expectEqual(made, core.pipelines);
    try testing.expectEqual(late, core.pipelines_late);
    try testing.expectEqual(preloaded + 1, core.pipelines_preloaded);
}

test "pipeline cache: a stock kernel made from its recipe is the pipeline kernel() answers" {
    try @import("../mlx.zig").requireGpu();
    const name = "rms_loopedfloat16";
    stock.forget(name);
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

test "pipeline cache: a source recipe and a missing stock recipe are refused, and the valid ones are still made" {
    try @import("../mlx.zig").requireGpu();
    const good: Recipe = .{ .kind = .stock, .function = "rms_loopedbfloat16", .key = "rms_loopedbfloat16" };
    const source: Recipe = .{ .kind = .source, .function = "source_refused", .text = "[[kernel]] void source_refused() {}" };
    const missing: Recipe = .{ .kind = .stock, .function = "no_such_stock_function", .key = "no_such_stock_function" };
    const report = finish(try start(&.{ good, source, missing }, 3));
    try testing.expectEqual(@as(usize, 1), report.made);
    try testing.expectEqual(@as(usize, 2), report.failed);
    try testing.expect(report.firstError().len > 0);
}

test "pipeline cache: writeArchive persists stock and specialized pipelines and startWithArchive loads them" {
    try @import("../mlx.zig").requireGpu();
    try launchScaleOnce(88);
    var custom_recipe: ?Recipe = null;
    for (registry.recipes()) |r| {
        if (r.kind == .stock_flags and std.mem.indexOf(u8, r.key, "sushi_test_scale") != null and std.mem.indexOf(u8, r.key, "=88") != null) custom_recipe = r;
    }
    const r_custom = custom_recipe orelse return error.RecipeNotRecorded;
    const r_stock: Recipe = .{ .kind = .stock, .function = "rms_loopedbfloat16", .key = "rms_loopedbfloat16" };
    const both = [_]Recipe{ r_stock, r_custom };

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_len = try tmp.dir.realPath(testing.io, &dir_buf);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const archive_path = try std.fmt.bufPrintSentinel(&path_buf, "{s}/archive.bin", .{dir_buf[0..dir_len]}, 0);

    const added = try writeArchive(testing.io, &both, archive_path);
    try testing.expectEqual(@as(usize, 2), added);

    stock.forget(r_stock.key);
    stock.forget(r_custom.key);
    const arch = try openArchive(archive_path);
    defer arch.release();
    const report = finish(try startWithArchive(&both, 2, arch));
    try testing.expectEqual(@as(usize, 2), report.made);
    try testing.expectEqual(@as(usize, 0), report.failed);
}
