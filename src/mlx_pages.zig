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

/// PURE: the bytes one grid step of `pagedCountAll` spans across buffers of `part_bytes` a row: the most the grid adds.
pub fn gridStepBytes(part_bytes: []const u64) u64 {
    var step: u64 = 1;
    for (part_bytes) |b| if (b > 0) {
        const s = page / std.math.gcd(page, b);
        step = step / std.math.gcd(step, s) * s;
    };
    var bytes: u64 = 0;
    for (part_bytes) |b| bytes +|= step *| b;
    return bytes;
}

/// PURE: what an append into a buffer of `bytes` copies when it sits off the page grid: the most MLX's pool hands it
/// (its pages and one more), which MLX never donates. Zero on the grid or within a page, where the append is in place.
pub fn offGridCopyBytes(bytes: u64) u64 {
    if (onGrid(bytes)) return 0;
    return (bytes / page + 2) * page;
}

/// What grid a buffer written in place keeps: `paged` for one that grows with its rows (written only while it sits on
/// the grid), `ring` for a ring's fixed rows, which keep their size off it.
pub const Grid = enum { paged, ring };

/// A buffer rows are written into in place. Only this file makes one (lint `in-place-write` refuses the write
/// anywhere else, and the field below anywhere else): `capacity_bytes` is the buffer's bytes, recorded when it was made
/// or taken, so `write` checks the page grid against it and asks MLX nothing. A write keeps the buffer's shape.
pub const InPlace = struct {
    array: mlx.mlx_array = .{ .ctx = null },
    capacity_bytes: u64 = 0,
    grid: Grid = .paged,

    pub const none: InPlace = .{};

    /// An array a holder was handed already made (a restore's exact copy, a disk chunk, a seed): its bytes are read here,
    /// once. One off the grid is held as it is and refuses every write (`write`), so the holder grows it first.
    pub fn adopt(array: mlx.mlx_array, grid: Grid) InPlace {
        if (array.ctx == null) return .{ .grid = grid };
        return .{ .array = array, .capacity_bytes = @as(u64, mlx.mlx_array_size(array)) * mlx.mlx_array_itemsize(array), .grid = grid };
    }

    /// Whether a write into the buffer could copy all of it on MLX: a `paged` buffer off the page grid that is not the
    /// engine of our own's.
    pub fn offGrid(self: InPlace) bool {
        return self.grid == .paged and !onGrid(self.capacity_bytes) and !mlx.engine.isOurs(self.array);
    }

    /// Whether `write` takes this buffer on `s`; the engine of our own writes in place at any size.
    pub fn writable(self: InPlace, s: mlx.mlx_stream) bool {
        return !self.offGrid() or mlx.engine.isStream(s);
    }

    /// THE write of `rows` into the buffer over `[start, stop)`: the result lands in `out`, and the buffer that holds it
    /// comes back. The caller frees the old array. A buffer off the grid is refused, whoever made it.
    pub fn write(self: InPlace, out: *mlx.mlx_array, rows: mlx.mlx_array, start: []const c_int, stop: []const c_int, s: mlx.mlx_stream) !InPlace {
        std.debug.assert(start.len == stop.len and start.len <= 8);
        if (!self.writable(s)) return error.InPlaceBufferOffPageGrid;
        const strides: [8]c_int = @splat(1);
        try mlx.check(mlx.mlx_slice_update(out, self.array, rows, start.ptr, start.len, stop.ptr, stop.len, &strides, start.len, s));
        return .{ .array = out.*, .capacity_bytes = self.capacity_bytes, .grid = self.grid };
    }

    /// The buffer as another launch leaves it: `array` is this buffer's own shape (a ring a kernel wrote), so its record
    /// stands. The caller frees the old array.
    pub fn replaced(self: InPlace, array: mlx.mlx_array) InPlace {
        return .{ .array = array, .capacity_bytes = self.capacity_bytes, .grid = self.grid };
    }

    /// Another handle to the same buffer (a rollback's saved copy): the record goes with it.
    pub fn share(self: InPlace) InPlace {
        if (self.array.ctx == null) return self;
        var copy = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&copy, self.array);
        return .{ .array = copy, .capacity_bytes = self.capacity_bytes, .grid = self.grid };
    }

    pub fn deinit(self: *InPlace) void {
        if (self.array.ctx != null) _ = mlx.mlx_array_free(self.array);
        self.* = .none;
    }
};

/// The constructor of a buffer later rows are written into in place: zeros of `shape`, which the caller sizes on the
/// page grid (`pagedRows`, `pagedCount`). The caller owns the result. Each constructor here takes the engine's room for
/// the buffer once (lint `cache-room`), as admission bills it.
pub fn inPlaceBuffer(shape: []const c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !InPlace {
    try requireGrid(shape, dtype, s);
    return .{ .array = try zeros(shape, dtype, s), .capacity_bytes = shapeBytes(shape, dtype), .grid = .paged };
}

/// `inPlaceBuffer` for a ring's fixed rows, which keep their size off the grid (MiMo's sliding window).
pub fn ringBuffer(shape: []const c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !InPlace {
    return .{ .array = try zeros(shape, dtype, s), .capacity_bytes = shapeBytes(shape, dtype), .grid = .ring };
}

/// A buffer of `shape` grown from `kept`, the rows it keeps at the front of `axis`, with a zero tail: one join, so no
/// second whole-size buffer is held while the rows copy. On the grid unless `ring`.
pub fn grownBuffer(kept: mlx.mlx_array, shape: []const c_int, axis: usize, dtype: mlx.mlx_dtype, ring: bool, s: mlx.mlx_stream) !InPlace {
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
    return .{ .array = joined, .capacity_bytes = shapeBytes(shape, dtype), .grid = if (ring) .ring else .paged };
}

/// After a grown buffer was evaluated and its old handle freed, before the next is rebuilt. MLX releases an evaluated
/// graph's inputs when the GPU's command buffer completes, a moment after the eval returns, so the stream is
/// synchronized and its pool emptied, or the next rebuild starts beside this one's old copy and tail; the engine's pool
/// keeps a freed buffer as a free one, which `trim` returns.
pub fn settleGrown(s: mlx.mlx_stream) !void {
    if (mlx.engine.isStream(s)) {
        mlx.engine.trim();
    } else {
        try mlx.check(mlx.mlx_synchronize(s));
        _ = mlx.mlx_clear_cache();
    }
}

fn zeros(shape: []const c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    mlx.engine.roomFor(shape, dtype);
    var made = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(made);
    try mlx.check(mlx.mlx_zeros(&made, shape.ptr, shape.len, dtype, s));
    return made;
}

fn shapeBytes(shape: []const c_int, dtype: mlx.mlx_dtype) u64 {
    var bytes: u64 = core.itemSize(dtype);
    for (shape) |d| bytes *|= @intCast(d);
    return bytes;
}

/// Refuses a buffer of `shape` made on MLX off its page grid, where every write into it could copy all of it. The
/// engine of our own writes in place at any size.
fn requireGrid(shape: []const c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !void {
    if (mlx.engine.isStream(s)) return;
    if (!onGrid(shapeBytes(shape, dtype))) return error.InPlaceBufferOffPageGrid;
}

test "a buffer made on MLX's page grid takes every append in place while MLX's pool holds buffers a page past it" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    // 1,000 rows of 1 KiB are 61.04 pages: the grid gives 1,008 rows, 63 pages, and a free 64-page buffer is donatable.
    const rows: c_int = @intCast(pagedCount(1000, 1024));
    try std.testing.expectError(error.InPlaceBufferOffPageGrid, inPlaceBuffer(&.{ 1000, 512 }, .bfloat16, s));
    try @import("test_support.zig").seedMlxPool(s, 64 * page, 16);
    var buf = try inPlaceBuffer(&.{ rows, 512 }, .bfloat16, s);
    defer buf.deinit();
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    try mlx.check(mlx.mlx_ones(&row, &[_]c_int{ 1, 512 }, 2, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(row));
    var first: ?[*]const u8 = null;
    for (0..8) |i| {
        var up = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(up);
        const at: c_int = @intCast(i);
        const next = try buf.write(&up, row, &.{ at, 0 }, &.{ at + 1, 512 }, s);
        _ = mlx.mlx_array_free(buf.array);
        buf = next;
        try mlx.check(mlx.mlx_array_eval(buf.array));
        try mlx.check(mlx.mlx_synchronize(s));
        const p = mlx.mlx_array_data_uint8(buf.array);
        if (first) |f| try std.testing.expectEqual(f, p) else first = p;
    }
}

const std = @import("std");
const mlx = @import("mlx.zig");
const core = @import("tensor/core.zig");
