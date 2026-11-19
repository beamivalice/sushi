//! MLX's page grid for a buffer written in place. MLX rounds an allocation past a page up to whole pages, its pool
//! hands an allocation a free buffer up to two pages larger (`reuse_from_cache`), and it donates a buffer to a write
//! only when the buffer is at most one page past the array's bytes (`is_donatable`).

pub const page: u64 = 16384;

/// PURE: the rows a buffer appended in place holds for `rows`, where `part_bytes` is a row's bytes in each buffer that
/// grows with it: 256-row steps on which every part past a page is a whole number of pages, so whatever buffer MLX's
/// pool hands a part is one MLX donates, and the append writes in place.
pub fn pagedRows(rows: u64, part_bytes: []const u64) u64 {
    var capacity = (rows +| 255) / 256 * 256;
    while (!onPages(capacity, part_bytes)) capacity += 256;
    return capacity;
}

fn onPages(rows: u64, part_bytes: []const u64) bool {
    for (part_bytes) |b| if (!onGrid(rows *| b)) return false;
    return true;
}

/// PURE: whether a buffer of `bytes` is one every append writes in place: within a page, or whole pages.
pub fn onGrid(bytes: u64) bool {
    return bytes <= page or bytes % page == 0;
}

/// PURE: the fewest rows, at least `rows`, of a buffer of `row_bytes` a row that are within a page or whole pages: the
/// finest grid a buffer appended in place can take (`pagedRows` keeps the KV cache's 256-row steps).
pub fn pagedCount(rows: u64, row_bytes: u64) u64 {
    if (row_bytes == 0 or rows *| row_bytes <= page) return rows;
    const step = page / std.math.gcd(page, row_bytes);
    return (rows +| step - 1) / step * step;
}

/// PURE: `pagedCount` for buffers that grow together, a row of each `part_bytes`: the fewest rows on every part's grid.
pub fn pagedCountAll(rows: u64, part_bytes: []const u64) u64 {
    var count = rows;
    while (true) {
        var next = count;
        for (part_bytes) |b| next = @max(next, pagedCount(next, b));
        if (next == count) return count;
        count = next;
    }
}

/// PURE: what an append into a buffer of `bytes` copies when it sits off the page grid: the most MLX's pool hands it
/// (its pages and one more), which MLX never donates. Zero on the grid or within a page, where the append is in place.
pub fn offGridCopyBytes(bytes: u64) u64 {
    if (onGrid(bytes)) return 0;
    return (bytes / page + 2) * page;
}

/// The one constructor of a buffer later rows are written into in place (lint `in-place-buffer`): zeros of `shape`,
/// which the caller sizes on the page grid (`pagedRows`, `pagedCount`). The caller owns the result. Each constructor
/// here takes the engine's room for the buffer once (lint `cache-room`), as admission bills it.
pub fn inPlaceBuffer(shape: []const c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    try requireGrid(shape, dtype, s);
    return zeros(shape, dtype, s);
}

/// `inPlaceBuffer` for a ring's fixed rows, which keep their size off the grid (MiMo's sliding window).
pub fn ringBuffer(shape: []const c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    return zeros(shape, dtype, s);
}

/// A buffer of `shape` grown from `kept`, the rows it keeps at the front of `axis`, with a zero tail: one join, so no
/// second whole-size buffer is held while the rows copy. On the grid unless `ring`.
pub fn grownBuffer(kept: mlx.mlx_array, shape: []const c_int, axis: usize, dtype: mlx.mlx_dtype, ring: bool, s: mlx.mlx_stream) !mlx.mlx_array {
    if (!ring) try requireGrid(shape, dtype, s);
    mlx.engine.roomFor(shape, dtype);
    var tail_shape: [4]c_int = undefined;
    @memcpy(tail_shape[0..shape.len], shape);
    tail_shape[axis] -= mlx.getShape(kept)[axis];
    var tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail);
    try mlx.check(mlx.mlx_zeros(&tail, &tail_shape, shape.len, dtype, s));
    const parts = [_]mlx.mlx_array{ kept, tail };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var joined = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(joined);
    try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, @intCast(axis), s));
    return joined;
}

fn zeros(shape: []const c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    mlx.engine.roomFor(shape, dtype);
    var made = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(made);
    try mlx.check(mlx.mlx_zeros(&made, shape.ptr, shape.len, dtype, s));
    return made;
}

/// Refuses a buffer of `shape` written in place on MLX off its page grid, where every write into it could copy all of
/// it. The engine of our own writes in place at any size.
pub fn requireGrid(shape: []const c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !void {
    if (mlx.engine.isStream(s)) return;
    var bytes: u64 = core.itemSize(dtype);
    for (shape) |d| bytes *|= @intCast(d);
    if (!onGrid(bytes)) return error.InPlaceBufferOffPageGrid;
}

test "a buffer made on MLX's page grid takes every append in place while MLX's pool holds buffers a page past it" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    // 1,000 rows of 1 KiB are 61.04 pages: the grid gives 1,008 rows, 63 pages, and a free 64-page buffer is donatable.
    const rows: c_int = @intCast(pagedCount(1000, 1024));
    try std.testing.expectError(error.InPlaceBufferOffPageGrid, inPlaceBuffer(&.{ 1000, 512 }, .bfloat16, s));
    try @import("test_support.zig").seedMlxPool(s, 64 * page, 16);
    var buf = try inPlaceBuffer(&.{ rows, 512 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(buf);
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    try mlx.check(mlx.mlx_ones(&row, &[_]c_int{ 1, 512 }, 2, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(row));
    var first: ?[*]const u8 = null;
    for (0..8) |i| {
        var up = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(up);
        const at: c_int = @intCast(i);
        try mlx.check(mlx.mlx_slice_update(&up, buf, row, &[_]c_int{ at, 0 }, 2, &[_]c_int{ at + 1, 512 }, 2, &[_]c_int{ 1, 1 }, 2, s));
        _ = mlx.mlx_array_free(buf);
        buf = up;
        try mlx.check(mlx.mlx_array_eval(buf));
        try mlx.check(mlx.mlx_synchronize(s));
        const p = mlx.mlx_array_data_uint8(buf);
        if (first) |f| try std.testing.expectEqual(f, p) else first = p;
    }
}

const std = @import("std");
const mlx = @import("mlx.zig");
const core = @import("tensor/core.zig");
