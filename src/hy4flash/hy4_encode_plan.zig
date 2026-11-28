//! Hy4-Flash's image encode on the static bill (docs/bill_redesign.md, D97): an image's tower and merger run as one
//! wide execution in the load's region, which the load measures by recording the encode dry at the largest grid of each
//! class it accepts (`probeGrids`, to `maxImagePatches`), and write their rows into the request's `Media`, a buffer of
//! its own sized from its grids.
const std = @import("std");
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const plan = @import("../tensor/plan/plan_api.zig");
const plan_trace = @import("../tensor/plan/plan_trace.zig");
const mlx_pages = @import("../mlx_pages.zig");
const log = @import("../log.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const hy4_config = @import("hy4_config.zig");
const hy4_vision = @import("hy4_vision.zig");
const Tower = hy4_vision.Tower;
const Plans = @import("hy4_plan.zig").Plans;
const Arr = mlx.mlx_array;

/// What decides an encode's structure, so its probes take each one's largest: whether its attention takes one chunk of
/// query rows or several, which joins them; its patches and merge cells up to `vector_rows`, each count its own class, where a product of that many rows
/// takes the vector kernels in place of the tiled ones (`matmul`); and whether its merged grid is one row or one column
/// across, where the merger's reshapes are views.
pub const Class = struct {
    several: bool,
    patches: u64,
    cells: u64,
    one_row: bool,
    one_col: bool,

    pub fn of(t: *const Tower, grid_h: u32, grid_w: u32) Class {
        const n = @as(u64, grid_h) * grid_w;
        const m = t.v.merge;
        return .{
            .several = n > t.attn_rows,
            .patches = @min(n, vector_rows + 1),
            .cells = @min(n / (@as(u64, m) * m), vector_rows + 1),
            .one_row = grid_h / m == 1,
            .one_col = grid_w / m == 1,
        };
    }
};

/// The most rows a product takes the vector kernels at (`matmul.gemvWide`: five a pass, at most three passes).
const vector_rows: u64 = 15;

pub const Grid = struct { h: u32, w: u32 };

/// The grids an encode is recorded at, for a load that accepts images of at most `max_patches` patches. Every ask of an
/// encode grows with its merge cells or its merged rows (each cell's and each merged row's newline), so for each count
/// of cells `c` a probe takes, the grids of its classes that hold the most of either: one merge cell wide (the most
/// rows), one cell tall, and of those at least two each way, the most cells and the most rows (two cells wide). The
/// counts: each to `vector_rows`, each class's most, and each at which a tiled product's path may turn
/// (`matmul.steelRowEdges`), so each path's variant holds its largest product. Sorted by cells, each once; the caller
/// frees them.
pub fn probeGrids(gpa: std.mem.Allocator, t: *const Tower, max_patches: u64) ![]Grid {
    const v = t.v;
    const m: u64 = v.merge;
    const cell = m * m;
    const most = max_patches / cell;
    var cells: std.ArrayList(u64) = .empty;
    defer cells.deinit(gpa);
    const smallest: u64 = @min(vector_rows + 1, most);
    for (1..smallest + 1) |c| try cells.append(gpa, c);
    try cells.append(gpa, @min(t.attn_rows, max_patches) / cell);
    try cells.append(gpa, most);
    // Each tiled product of the tower: its inner and outer widths, and how many patches make one of its rows.
    const d: u64 = v.hidden;
    const products = [_][3]u64{
        .{ 3 * @as(u64, v.patch) * v.patch, d, 1 }, .{ d, d, 1 },                 .{ d, v.inter, 1 },
        .{ v.inter, d, 1 },                         .{ d * cell, d * cell, cell }, .{ d * cell, v.out_hidden, cell },
    };
    for (products) |p| for (@import("../tensor/matmul.zig").steelRowEdges(p[0], p[1])) |rows| {
        const c = rows * p[2] / cell;
        if (c > 0 and c <= most) try cells.append(gpa, c);
    };
    std.mem.sort(u64, cells.items, {}, std.sort.asc(u64));
    var grids: std.ArrayList(Grid) = .empty;
    errdefer grids.deinit(gpa);
    for (cells.items, 0..) |c, i| {
        if (c == 0 or (i > 0 and cells.items[i - 1] == c)) continue;
        // In merge cells: rows by columns.
        var shapes: [4][2]u64 = undefined;
        var n: usize = 0;
        shapes[n] = .{ c, 1 };
        n += 1;
        if (c >= 2) {
            shapes[n] = .{ 1, c };
            n += 1;
        }
        if (c >= 4) {
            var best: [2]u64 = .{ c / 2, 2 };
            var w: u64 = 3;
            while (w * w <= c) : (w += 1) if ((c / w) * w > best[0] * best[1]) {
                best = .{ c / w, w };
            };
            shapes[n] = best;
            n += 1;
            if (best[1] != 2) {
                shapes[n] = .{ c / 2, 2 };
                n += 1;
            }
        }
        for (shapes[0..n]) |sh| try grids.append(gpa, .{ .h = @intCast(sh[0] * m), .w = @intCast(sh[1] * m) });
    }
    return grids.toOwnedSlice(gpa);
}

/// The rows a request's images give the prompt, as `Tower.rowsOf` counts them, held in f32.
pub fn mediaBytes(v: hy4_config.Vision, rows: u64) u64 {
    return std.mem.alignForward(u64, @max(rows * v.out_hidden * @sizeOf(f32), 1), std.heap.page_size_min);
}

/// A request's image rows after the encode: [1, rows, out_hidden] f32 over pages mapped for it alone, never the pool's.
/// The encode of each image writes its rows into it (`run`); the prefill's splice reads it; its last handle's release
/// unmaps it once the last pass that binds it has run.
pub fn makeMedia(v: hy4_config.Vision, rows: usize) !Arr {
    const len: usize = @intCast(mediaBytes(v, rows));
    const mapped = std.c.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    if (mapped == std.c.MAP_FAILED) return error.OutOfMemory;
    const region = std.heap.c_allocator.create(Region) catch {
        _ = std.c.munmap(@alignCast(mapped), len);
        return error.OutOfMemory;
    };
    region.* = .{ .ptr = @ptrCast(@alignCast(mapped)), .len = len };
    const a = mlx.engine.wrapOwned(region.ptr, &[_]c_int{ 1, @intCast(rows), @intCast(v.out_hidden) }, .float32, region, Region.release);
    if (a.ctx == null) {
        Region.release(region);
        return error.Hy4MediaWrap;
    }
    return a;
}

const Region = struct {
    ptr: [*]align(std.heap.page_size_min) u8,
    len: usize,

    fn release(payload: ?*anyopaque) callconv(.c) void {
        const r: *Region = @ptrCast(@alignCast(payload.?));
        _ = std.c.munmap(r.ptr, r.len);
        std.heap.c_allocator.destroy(r);
    }
};

/// The image's rows from its patches (host f32 [N, 3·P·P]) written into `media` from row `row0`: a wide execution in the
/// region the load measured where it plans (`plans`), else on the pool. An ask past the region refuses the encode by
/// name (`WideFull`).
pub fn run(plans: ?*const Plans, t: *const Tower, media: *Arr, row0: usize, pixels: []const u8, grid_h: u32, grid_w: u32) !void {
    if (plans == null) return encodeInto(t, media, row0, pixels, grid_h, grid_w);
    try core.beginWide();
    encodeInto(t, media, row0, pixels, grid_h, grid_w) catch |e| {
        core.abortWide();
        if (core.refused_wide) |r| {
            log.err("[plan] an encode of {d} x {d} patches asked {d} KB of the region's {s}, past the {d} KB the load holds\n", .{ grid_h, grid_w, r.asked >> 10, if (r.host) "host ring" else "scratch", r.room >> 10 });
            return error.WideFull;
        }
        return e;
    };
    try core.endWide();
}

fn encodeInto(t: *const Tower, media: *Arr, row0: usize, pixels: []const u8, grid_h: u32, grid_w: u32) !void {
    const s = mlx.engine.stream();
    var o: Ops = .{ .s = s };
    defer o.deinit();
    const rows = try t.encodeHost(&o, pixels, grid_h, grid_w);
    const r: c_int = mlx.getShape(rows)[0];
    const out: c_int = @intCast(t.v.out_hidden);
    const at: c_int = @intCast(row0);
    if (mlx.getShape(media.*)[1] < at + r) return error.Hy4MediaShort;
    const update = try o.reshape(rows, &.{ 1, r, out });
    const next = blk: {
        var up = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(up);
        break :blk try mlx_pages.InPlace.adopt(media.*, .ring).write(&up, update, &.{ 0, at, 0 }, &.{ 1, at + r, out }, s);
    };
    _ = mlx.mlx_array_free(media.*);
    media.* = next.array;
    try next.landed(s);
}

/// `grid`'s encode recorded dry under `fill` over zero pixels, or (`spread`) pixels that vary, into a media of its own
/// rows: one trace a pass of the driver.
fn record(t: *const Tower, gpa: std.mem.Allocator, grid: Grid, fill: core.Fill, spread: bool) ![]plan_trace.Trace {
    const n: usize = @as(usize, grid.h) * grid.w;
    const values = try gpa.alloc(f32, n * 3 * @as(usize, t.v.patch) * t.v.patch);
    defer gpa.free(values);
    for (values, 0..) |*x, i| x.* = if (spread) @as(f32, @floatFromInt(@as(i32, @intCast(i % 255)) - 127)) / 64 else 0;
    const pixels = std.mem.sliceAsBytes(values);
    var passes: std.ArrayList(plan_trace.Trace) = .empty;
    errdefer {
        for (passes.items) |*tr| tr.deinit();
        passes.deinit(gpa);
    }
    var driver: plan.Driver = .{ .gpa = gpa };
    defer driver.deinit();
    while (true) {
        var media = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(media);
        {
            // The media is the request's, made beside the encode: a recording of its own, dropped.
            var rec = plan.Recorder.begin(gpa);
            defer rec.deinit();
            var state_driver: plan.Driver = .{ .gpa = gpa };
            defer state_driver.deinit();
            try core.beginRecording(.{ .dry = .{ .driver = &state_driver } }, &rec);
            defer core.endRecording() catch {};
            try mlx.check(mlx.engine.freshArray(&media, &[_]c_int{ 1, @intCast(t.rowsOf(grid.h, grid.w)), @intCast(t.v.out_hidden) }, .float32));
        }
        var rec = plan.Recorder.begin(gpa);
        defer rec.deinit();
        {
            try core.beginRecording(.{ .dry = .{ .driver = &driver, .fill = fill } }, &rec);
            defer core.endRecording() catch {};
            try encodeInto(t, &media, 0, pixels, grid.h, grid.w);
        }
        var tr = rec.end() catch |e| {
            if (e == error.PlanEscape) log.err("[plan] an ask outlives an encode of {d} x {d} patches\n", .{ grid.h, grid.w });
            return e;
        };
        passes.append(gpa, tr) catch |e| {
            tr.deinit();
            return e;
        };
        if (!try driver.next()) break;
    }
    return passes.toOwnedSlice(gpa);
}

/// `grid`'s encode recorded twice (D99), over zero pixels under the zeros fill and over pixels that vary under the
/// ramp: the first's passes, once the second's are the same execution, sizes and frames (its dispatch levels aside,
/// which the region does not read). A size its pixels or a readback steer with no `plan.bound` parts them, refused by
/// name.
pub fn recordChecked(t: *const Tower, gpa: std.mem.Allocator, grid: Grid) ![]plan_trace.Trace {
    const zeros = try record(t, gpa, grid, .zeros, false);
    errdefer freeTraces(gpa, zeros);
    const other = try record(t, gpa, grid, .ramp, true);
    defer freeTraces(gpa, other);
    if (zeros.len != other.len) return error.Hy4PlanDataSteers;
    for (zeros, other) |z, r| if (plan_trace.firstDifference(z.view(), r.view())) |d| {
        log.err("[plan] an encode of {d} x {d} patches records two shapes over two inputs ({t} at {d}): its data steers a size no `plan.bound` declares\n", .{ grid.h, grid.w, d.what, d.at });
        return error.Hy4PlanDataSteers;
    };
    return zeros;
}

fn freeTraces(gpa: std.mem.Allocator, traces: []plan_trace.Trace) void {
    for (traces) |*tr| tr.deinit();
    gpa.free(traces);
}

/// The region an encode of an image the load accepts asks at most (D97), each place rounded to `alignment`: the most of
/// each probe grid's checked recording (`recordChecked`).
pub fn need(t: *const Tower, gpa: std.mem.Allocator, max_patches: u64, alignment: u64, guard: u64) !plan.region.Need {
    const grids = try probeGrids(gpa, t, max_patches);
    defer gpa.free(grids);
    var most: plan.region.Need = .{};
    for (grids) |g| {
        const traces = try recordChecked(t, gpa, g);
        defer freeTraces(gpa, traces);
        for (traces) |*tr| most = most.max(try plan.region.needWithGuard(gpa, tr, alignment, guard));
    }
    return most;
}

test "hy4 plan: every grid the load accepts has, among the encode's probe grids of its class at most the next count of cells probed, one with as many cells and one with as many merged rows: every count to the vector kernels' rows, each class's most and each tiled product's turn are probed" {
    const testing = std.testing;
    var c = try hy4_config.parseText(hy4_config.release_file);
    _ = &c;
    const v = c.vision.?;
    var t: Tower = undefined;
    t.v = v;
    t.attn_rows = hy4_vision.attn_query_rows;
    const m = v.merge;
    const most: u64 = 9216;
    const grids = try probeGrids(testing.allocator, &t, most);
    defer testing.allocator.free(grids);
    // The counts probed: each has its grid one merge cell wide.
    var counts: std.ArrayList(u64) = .empty;
    defer counts.deinit(testing.allocator);
    for (grids) |g| {
        try testing.expect(@as(u64, g.h) * g.w <= most);
        if (g.w == m) try counts.append(testing.allocator, g.h / m);
    }
    for (1..vector_rows + 2) |n| try testing.expect(std.mem.indexOfScalar(u64, counts.items, n) != null);
    try testing.expect(std.mem.indexOfScalar(u64, counts.items, most / (m * m)) != null);
    // The patch embedding's split-K edge of K / 3 rows: 768 / 3 = 256 patches, 64 cells.
    try testing.expect(std.mem.indexOfScalar(u64, counts.items, 64) != null);
    for (1..most / (m * m) + 1) |hb| for (1..most / (m * m) / hb + 1) |wb| {
        const cells = hb * wb;
        const next = for (counts.items) |n| {
            if (n >= cells) break n;
        } else unreachable;
        const gh: u32 = @intCast(hb * m);
        const gw: u32 = @intCast(wb * m);
        const class = Class.of(&t, gh, gw);
        var most_cells = false;
        var most_rows = false;
        for (grids) |g| if (std.meta.eql(Class.of(&t, g.h, g.w), class)) {
            const gc = @as(u64, g.h) * g.w / (m * m);
            if (gc > next) continue;
            most_cells = most_cells or gc >= cells;
            most_rows = most_rows or t.rowsOf(g.h, g.w) >= t.rowsOf(gh, gw);
        };
        errdefer std.debug.print("probe grids: a grid of {d} x {d} cells has no probe of its class to {d} cells with as many cells ({}) or rows ({})\n", .{ hb, wb, next, most_cells, most_rows });
        try testing.expect(most_cells and most_rows);
    };
}
