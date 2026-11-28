//! DeepSeek-V4.1's image encode on the static bill (`docs/bill_redesign.md`, D97): each Cooler tower encode runs as one
//! wide execution inside the load's measured region and writes its `.bfloat16` span rows directly into the request's
//! mmapped `Media` buffer, asking nothing of the pool.
const std = @import("std");
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const plan = @import("../tensor/plan/plan_api.zig");
const plan_trace = @import("../tensor/plan/plan_trace.zig");
const mlx_pages = @import("../mlx_pages.zig");
const log = @import("../log.zig");
const matmul = @import("../tensor/matmul.zig");
const cfg_mod = @import("dsv41_config.zig");
const dsv41_vision = @import("dsv41_vision.zig");
const Dsv41Vision = dsv41_vision.Dsv41Vision;
const Plans = @import("dsv41_plan.zig").Plans;
const Arr = mlx.mlx_array;

pub const Grid = struct { h: u32, w: u32 };

fn validGrid(v: cfg_mod.Vision, max_patches: u64, gh: u32, gw: u32) bool {
    if (gh == 0 or gw == 0) return false;
    const n = @as(u64, gh) * gw;
    if (n < 8 or n > max_patches) return false;
    if (v.max_tokens >= 4 and dsv41_vision.spanRows(gh, gw, v.downsample) > v.max_tokens) return false;
    return true;
}

fn addGrid(
    gpa: std.mem.Allocator,
    seen: *std.AutoHashMapUnmanaged(u64, void),
    out: *std.ArrayList(Grid),
    v: cfg_mod.Vision,
    max_patches: u64,
    gh: u32,
    gw: u32,
) !void {
    if (!validGrid(v, max_patches, gh, gw)) return;
    const k = (@as(u64, gh) << 32) | gw;
    if ((try seen.getOrPut(gpa, k)).found_existing) return;
    try out.append(gpa, .{ .h = gh, .w = gw });
}

fn addTargetCountGrids(
    gpa: std.mem.Allocator,
    seen: *std.AutoHashMapUnmanaged(u64, void),
    out: *std.ArrayList(Grid),
    v: cfg_mod.Vision,
    max_patches: u64,
    n: u64,
) !void {
    if (n < 8 or n > max_patches) return;
    const r: u64 = @max(v.downsample, 1);
    const max_tokens_cap: u64 = if (v.max_tokens >= 4) v.max_tokens else std.math.maxInt(u64);
    _ = max_tokens_cap;

    // 1. Wide strip 1 x n and factor pairs around small widths and sqrt(n).
    try addGrid(gpa, seen, out, v, max_patches, 1, @intCast(n));
    try addGrid(gpa, seen, out, v, max_patches, @intCast(n), 1);
    if (n % 2 == 0) {
        try addGrid(gpa, seen, out, v, max_patches, 2, @intCast(n / 2));
        try addGrid(gpa, seen, out, v, max_patches, @intCast(n / 2), 2);
    }
    if (r > 1) {
        if (n / r > 0) {
            try addGrid(gpa, seen, out, v, max_patches, @intCast(n / r), @intCast(r));
            try addGrid(gpa, seen, out, v, max_patches, @intCast(r), @intCast(n / r));
        }
        const cell = r * r;
        const cells = n / cell;
        if (cells > 0) {
            const side = @max(@as(u64, 1), std.math.sqrt(cells));
            const h_cells = @max(@as(u64, 1), cells / side);
            try addGrid(gpa, seen, out, v, max_patches, @intCast(h_cells * r), @intCast(side * r));
            if (h_cells * r > 1) try addGrid(gpa, seen, out, v, max_patches, @intCast(h_cells * r - 1), @intCast(side * r));
            if (side * r > 1) try addGrid(gpa, seen, out, v, max_patches, @intCast(h_cells * r), @intCast(side * r - 1));
        }
    } else {
        const side = @max(@as(u64, 1), std.math.sqrt(n));
        try addGrid(gpa, seen, out, v, max_patches, @intCast(n / side), @intCast(side));
    }
}

/// Probe grids covering vector-row thresholds, attention query chunks, tiled GEMM row edges,
/// padded/unpadded aligner paths, and max-token / max-patch boundaries.
pub fn probeGrids(gpa: std.mem.Allocator, v: cfg_mod.Vision, out_dim: u32, max_patches: u64) ![]Grid {
    if (max_patches < 8) return gpa.alloc(Grid, 0);
    var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer seen.deinit(gpa);
    var out: std.ArrayList(Grid) = .empty;
    errdefer out.deinit(gpa);

    var counts: std.ArrayList(u64) = .empty;
    defer counts.deinit(gpa);

    for (8..18) |n| try counts.append(gpa, n);
    const qc: u64 = dsv41_vision.QUERY_CHUNK;
    for ([_]u64{ qc, qc + 1, qc + 4, qc + 8, 2 * qc, 2 * qc + 1, 2 * qc + 8 }) |n| {
        if (n <= max_patches) try counts.append(gpa, n);
    }

    const r: u64 = @max(v.downsample, 1);
    const cell = r * r;
    const d: u64 = v.hidden;
    const feat: u64 = 3 * @as(u64, v.patch) * v.patch;
    const patch_products = [_][2]u64{
        .{ feat, d },
        .{ d, 3 * d },
        .{ d, d },
        .{ d, 2 * v.inter },
        .{ v.inter, d },
    };
    for (patch_products) |pr| for (matmul.steelRowEdges(pr[0], pr[1])) |rows| {
        if (rows >= 8 and rows <= max_patches) try counts.append(gpa, rows);
    };
    const cell_products = [_][2]u64{
        .{ d * cell, out_dim },
        .{ out_dim, out_dim },
    };
    for (cell_products) |pr| for (matmul.steelRowEdges(pr[0], pr[1])) |rows| {
        const patches = rows * cell;
        if (patches >= 8 and patches <= max_patches) try counts.append(gpa, patches);
    };

    try counts.append(gpa, max_patches);
    try counts.append(gpa, max_patches -| 1);

    std.mem.sort(u64, counts.items, {}, std.sort.asc(u64));
    for (counts.items, 0..) |n, i| {
        if (i > 0 and counts.items[i - 1] == n) continue;
        try addTargetCountGrids(gpa, &seen, &out, v, max_patches, n);
    }

    // Explicit max-token & max-patch corner grids (near-square, tall, wide, and padded aligner variants).
    if (r > 1) {
        const most_cells = max_patches / cell;
        if (most_cells > 0) {
            const tok_budget: u64 = if (v.max_tokens >= 4) v.max_tokens - 2 else std.math.maxInt(u64);
            var best_cells: u64 = 0;
            var best_lh: u64 = 1;
            var best_lw: u64 = 1;
            var lw_try: u64 = 2;
            while (lw_try * lw_try <= most_cells + 16 and lw_try <= tok_budget) : (lw_try += 1) {
                const lh_cap = tok_budget / (lw_try + 1);
                const lh_try = @min(lh_cap, most_cells / lw_try);
                if (lh_try > 0 and lh_try * lw_try > best_cells) {
                    best_cells = lh_try * lw_try;
                    best_lh = lh_try;
                    best_lw = lw_try;
                }
            }
            if (best_cells > 0) {
                try addGrid(gpa, &seen, &out, v, max_patches, @intCast(best_lh * r), @intCast(best_lw * r));
                if (best_lh * r > 1 and best_lw * r > 1) {
                    try addGrid(gpa, &seen, &out, v, max_patches, @intCast(best_lh * r - 1), @intCast(best_lw * r - 1));
                }
            }
            const tall_cells = @min(most_cells, tok_budget / 2);
            if (tall_cells >= 1) {
                try addGrid(gpa, &seen, &out, v, max_patches, @intCast(tall_cells * r), @intCast(r));
                if (tall_cells * r > 1) try addGrid(gpa, &seen, &out, v, max_patches, @intCast(tall_cells * r - 1), @intCast(r - 1));
            }
            const wide_cells = @min(most_cells, tok_budget -| 1);
            if (wide_cells >= 1) {
                try addGrid(gpa, &seen, &out, v, max_patches, @intCast(r), @intCast(wide_cells * r));
                if (wide_cells * r > 1) try addGrid(gpa, &seen, &out, v, max_patches, @intCast(r - 1), @intCast(wide_cells * r - 1));
            }
        }
    }

    std.mem.sort(Grid, out.items, {}, struct {
        fn less(_: void, a: Grid, b: Grid) bool {
            const pa = @as(u64, a.h) * a.w;
            const pb = @as(u64, b.h) * b.w;
            if (pa != pb) return pa < pb;
            if (a.h != b.h) return a.h < b.h;
            return a.w < b.w;
        }
    }.less);
    return out.toOwnedSlice(gpa);
}

/// Page-aligned bytes of one request's `[1, rows, out_dim]` bfloat16 Media buffer.
pub fn mediaBytes(out_dim: u32, rows: usize) u64 {
    return std.mem.alignForward(u64, @max(@as(u64, rows) * out_dim * 2, 1), std.heap.page_size_min);
}

/// Request's own mmapped bfloat16 `[1, rows, out_dim]` Media buffer, unmapped cleanly on release.
pub fn makeMedia(out_dim: u32, rows: usize) !Arr {
    const len: usize = @intCast(mediaBytes(out_dim, rows));
    const mapped = std.c.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    if (mapped == std.c.MAP_FAILED) return error.OutOfMemory;
    const region = std.heap.c_allocator.create(Region) catch {
        _ = std.c.munmap(@alignCast(mapped), len);
        return error.OutOfMemory;
    };
    region.* = .{ .ptr = @ptrCast(@alignCast(mapped)), .len = len };
    const a = mlx.engine.wrapOwned(region.ptr, &[_]c_int{ 1, @intCast(rows), @intCast(out_dim) }, .bfloat16, region, Region.release);
    if (a.ctx == null) {
        Region.release(region);
        return error.Dsv41MediaWrap;
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

/// Encodes one image into `media` starting at row `row0`, running in the measured wide region when `plans != null`.
pub fn run(plans: ?*const Plans, t: *Dsv41Vision, media: *Arr, row0: usize, pixels: []const u8, grid_h: u32, grid_w: u32) !void {
    const p = plans orelse return encodeInto(t, media, row0, pixels, grid_h, grid_w);
    try p.acquireAliasLease();
    try core.beginWide();
    encodeInto(t, media, row0, pixels, grid_h, grid_w) catch |e| {
        core.abortWide();
        if (core.refused_wide) |r| {
            log.err("[dsv41-plan] an encode of {d} x {d} patches asked {d} KB of the region's {s}, past the {d} KB the load holds\n", .{ grid_h, grid_w, r.asked >> 10, if (r.host) "host ring" else "scratch", r.room >> 10 });
            return error.WideFull;
        }
        return e;
    };
    try core.endWide();
}

fn encodeInto(t: *Dsv41Vision, media: *Arr, row0: usize, pixels: []const u8, grid_h: u32, grid_w: u32) !void {
    const s = mlx.engine.stream();
    const scope = mlx.engine.enter(s);
    defer scope.end();
    const n: usize = @as(usize, grid_h) * grid_w;
    const feat: usize = 3 * @as(usize, t.v.patch) * t.v.patch;
    if (pixels.len != n * feat * @sizeOf(f32)) return error.InvalidPatchGrid;
    const patches = try mlx.newData(pixels.ptr, &[_]c_int{ @intCast(n), @intCast(feat) }, .float32);
    defer _ = mlx.mlx_array_free(patches);
    const sp = try t.encode(patches, grid_h, grid_w);
    defer _ = mlx.mlx_array_free(sp);

    const r: c_int = mlx.getShape(sp)[1];
    const out: c_int = @intCast(t.out_dim);
    const at: c_int = @intCast(row0);
    if (mlx.getShape(media.*)[1] < at + r) return error.Dsv41MediaShort;

    const media_dt = mlx.mlx_array_dtype(media.*);
    var cast = mlx.mlx_array_new();
    defer if (cast.ctx != null) {
        _ = mlx.mlx_array_free(cast);
    };
    const update = if (mlx.mlx_array_dtype(sp) == media_dt) sp else blk: {
        try mlx.check(mlx.mlx_astype(&cast, sp, media_dt, s));
        break :blk cast;
    };
    const next = blk: {
        var up = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(up);
        break :blk try mlx_pages.InPlace.adopt(media.*, .ring).write(&up, update, &.{ 0, at, 0 }, &.{ 1, at + r, out }, s);
    };
    _ = mlx.mlx_array_free(media.*);
    media.* = next.array;
    try next.landed(s);
}

fn record(t: *Dsv41Vision, gpa: std.mem.Allocator, grid: Grid, fill: core.Fill, spread: bool) ![]plan_trace.Trace {
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
            var rec = plan.Recorder.begin(gpa);
            defer rec.deinit();
            var state_driver: plan.Driver = .{ .gpa = gpa };
            defer state_driver.deinit();
            try core.beginRecording(.{ .dry = .{ .driver = &state_driver } }, &rec);
            defer core.endRecording() catch {};
            const rows = dsv41_vision.spanRows(grid.h, grid.w, t.v.downsample);
            try mlx.check(mlx.engine.freshArray(&media, &[_]c_int{ 1, @intCast(rows), @intCast(t.out_dim) }, .bfloat16));
        }
        var rec = plan.Recorder.begin(gpa);
        defer rec.deinit();
        {
            try core.beginRecording(.{ .dry = .{ .driver = &driver, .fill = fill } }, &rec);
            defer core.endRecording() catch {};
            try encodeInto(t, &media, 0, pixels, grid.h, grid.w);
        }
        var tr = rec.end() catch |e| {
            if (e == error.PlanEscape) log.err("[dsv41-plan] an ask outlives an encode of {d} x {d} patches\n", .{ grid.h, grid.w });
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

pub fn recordChecked(t: *Dsv41Vision, gpa: std.mem.Allocator, grid: Grid) ![]plan_trace.Trace {
    const zeros = try record(t, gpa, grid, .zeros, false);
    errdefer freeTraces(gpa, zeros);
    const other = try record(t, gpa, grid, .ramp, true);
    defer freeTraces(gpa, other);
    if (zeros.len != other.len) return error.Dsv41PlanDataSteers;
    for (zeros, other) |z, r| if (plan_trace.firstDifference(z.view(), r.view())) |d| {
        log.err("[dsv41-plan] an encode of {d} x {d} patches records two shapes over two inputs ({t} at {d})\n", .{ grid.h, grid.w, d.what, d.at });
        return error.Dsv41PlanDataSteers;
    };
    return zeros;
}

fn freeTraces(gpa: std.mem.Allocator, traces: []plan_trace.Trace) void {
    for (traces) |*tr| tr.deinit();
    gpa.free(traces);
}

/// Measures the shared wide region (`scratch`, `host`) across all probe grids up to `max_patches`.
pub fn need(t: *Dsv41Vision, gpa: std.mem.Allocator, max_patches: u64, alignment: u64, guard: u64) !plan.region.Need {
    if (!t.own) return error.EngineNotEnabled;
    const grids = try probeGrids(gpa, t.v, t.out_dim, max_patches);
    defer gpa.free(grids);
    var most: plan.region.Need = .{};
    for (grids) |g| {
        const traces = try recordChecked(t, gpa, g);
        defer freeTraces(gpa, traces);
        for (traces) |*tr| most = most.max(try plan.region.needWithGuard(gpa, tr, alignment, guard));
    }
    return most;
}

test "dsv41 encode plan: release config probeGrids satisfy patch/span caps and cover chunk, padded, unpadded, and corner regimes" {
    const testing = std.testing;
    const cfg = try cfg_mod.parseText(cfg_mod.release_config);
    const v = cfg.vision.?;
    const max_patches = v.maxPatches();
    const grids = try probeGrids(testing.allocator, v, cfg.dim, max_patches);
    defer testing.allocator.free(grids);
    try testing.expect(grids.len > 10);

    var saw_qc = false;
    var saw_past_qc = false;
    var saw_padded = false;
    var saw_unpadded = false;
    var saw_tall = false;
    var saw_large_square = false;

    for (grids) |g| {
        const n = @as(u64, g.h) * g.w;
        try testing.expect(n >= 8);
        try testing.expect(n <= max_patches);
        try testing.expect(dsv41_vision.spanRows(g.h, g.w, v.downsample) <= v.max_tokens);
        if (n == dsv41_vision.QUERY_CHUNK) saw_qc = true;
        if (n > dsv41_vision.QUERY_CHUNK) saw_past_qc = true;
        if (g.h % v.downsample != 0 or g.w % v.downsample != 0) saw_padded = true else saw_unpadded = true;
        if (g.w == v.downsample and g.h >= v.downsample * 100) saw_tall = true;
        if (n >= (max_patches * 9) / 10 and g.h > v.downsample * 4 and g.w > v.downsample * 4) saw_large_square = true;
    }
    try testing.expect(saw_qc);
    try testing.expect(saw_past_qc);
    try testing.expect(saw_padded);
    try testing.expect(saw_unpadded);
    try testing.expect(saw_tall);
    try testing.expect(saw_large_square);
}
