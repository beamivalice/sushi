//! Rows appended a chunk at a time into buffers that grow in place: a request's history (latent rows, index keys,
//! pooled keys) as one or more parts, a format's planes (codes, scales, biases) that hold the same rows. An append
//! writes only its new rows (`slice_update`, in place while the buffer has one holder); a grow copies the rows held once
//! into a buffer a `div`th larger. A pending read of the rows holds the buffer, so the write past it goes to a copy and
//! never reaches what the read sees (docs/engine-own-forward.md#crossing-between-the-two).
const std = @import("std");
const mlx = @import("mlx.zig");
const mlx_pages = @import("mlx_pages.zig");

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const none: A = .{ .ctx = null };

fn ci(x: usize) c_int {
    return @intCast(x);
}

/// A new buffer of `cap` rows of `cols` holding `buf`'s first `used` rows (one join, `mlx_pages.grownBuffer`); `buf` is
/// left as it was, so a caller can build every part's buffer before it frees any old one.
fn grown(buf: mlx_pages.InPlace, used: usize, cap: usize, cols: usize, dt: mlx.mlx_dtype, s: S) !mlx_pages.InPlace {
    if (used == 0) return mlx_pages.inPlaceBuffer(&.{ ci(cap), ci(cols) }, dt, s);
    var kept = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kept);
    try mlx.check(mlx.mlx_slice(&kept, buf.array, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(used), ci(cols) }, 2, &[_]c_int{ 1, 1 }, 2, s));
    return mlx_pages.grownBuffer(kept, &.{ ci(cap), ci(cols) }, 0, dt, false, s);
}

/// The rows `x` [n, cols] written into `buf` from row `at` (`InPlace.write`).
fn put(buf: *mlx_pages.InPlace, x: A, at: usize, s: S) !void {
    const sh = mlx.getShape(x);
    const next = blk: {
        var up = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(up);
        break :blk try buf.write(&up, x, &.{ ci(at), 0 }, &.{ ci(at) + sh[0], sh[1] }, s);
    };
    _ = mlx.mlx_array_free(buf.array);
    buf.* = next;
    // The store holds the write from here: one that fails to land is still the store's to release.
    try next.landed(s);
}

/// `count` rows on MLX's page grid for every buffer that grows with these rows, `part_bytes` a row of each
/// (`mlx_pages`), so each append writes in place; the engine of our own appends in place at any size.
pub fn pagedCap(count: usize, part_bytes: []const u64, s: S) usize {
    if (mlx.engine.isStream(s)) return count;
    return @intCast(mlx_pages.pagedCountAll(count, part_bytes));
}

/// The rows an append of `n` past `used` grows a store of `cap` to (before the page grid): a `div`th more, 64 at least.
pub fn appendCap(used: usize, n: usize, cap: usize, div: usize) usize {
    return @max(used + n, @max(64, cap + cap / div));
}

/// The rows a store of `used` keeps with the next append's slack (before the page grid).
pub fn reserveCap(used: usize, div: usize) usize {
    return @max(64, used + @max(1, used / div));
}

/// The most rows a store that has held at most `rows` can have (before the page grid): each grow is to the rows it holds
/// or a `div`th past the cap it outgrew, and a reserve a `div`th past the rows.
pub fn capBound(rows: usize, div: usize) usize {
    return @max(64, rows + @max(1, rows / div));
}

pub const max_parts = 4;

/// One plane of a store: its columns and dtype.
pub const Part = struct { cols: usize, dtype: mlx.mlx_dtype };

/// Rows held as `parts.len` planes that grow together: the first `used` rows of each buffer are the store's.
pub const Store = struct {
    bufs: [max_parts]mlx_pages.InPlace = @splat(.none),
    parts: [max_parts]Part = undefined,
    n: usize = 0,
    used: usize = 0,
    cap: usize = 0,
    div: usize,

    pub fn init(parts: []const Part, div: usize) Store {
        var st: Store = .{ .div = div, .n = parts.len };
        @memcpy(st.parts[0..parts.len], parts);
        return st;
    }

    pub fn deinit(self: *Store) void {
        for (self.bufs[0..self.n]) |*b| b.deinit();
        self.used = 0;
        self.cap = 0;
    }

    /// Part `i`'s buffer: its first `used` rows are the store's, the rest its slack.
    pub fn buf(self: *const Store, i: usize) A {
        return self.bufs[i].array;
    }

    /// A row's bytes in each part.
    pub fn partBytes(self: *const Store, out: *[max_parts]u64) []const u64 {
        for (self.parts[0..self.n], out[0..self.n]) |p, *b| b.* = p.cols * @import("tensor/core.zig").itemSize(p.dtype);
        return out[0..self.n];
    }

    pub fn rowBytes(self: *const Store) u64 {
        var pb: [max_parts]u64 = undefined;
        var sum: u64 = 0;
        for (self.partBytes(&pb)) |b| sum += b;
        return sum;
    }

    fn pagedTo(self: *const Store, count: usize, s: S) usize {
        var pb: [max_parts]u64 = undefined;
        return pagedCap(count, self.partBytes(&pb), s);
    }

    fn growTo(self: *Store, cap: usize, s: S) !void {
        var next: [max_parts]mlx_pages.InPlace = @splat(.none);
        var built: usize = 0;
        errdefer for (next[0..built]) |*b| b.deinit();
        for (self.bufs[0..self.n], self.parts[0..self.n], next[0..self.n]) |b, p, *dst| {
            dst.* = try grown(b, self.used, cap, p.cols, p.dtype, s);
            built += 1;
        }
        for (self.bufs[0..self.n], next[0..self.n]) |*b, z| {
            b.deinit();
            b.* = z;
        }
        self.cap = cap;
    }

    /// Appends `rows`, one array a part ([n, cols] in its dtype): only the new rows are written.
    pub fn append(self: *Store, rows: []const A, s: S) !void {
        return self.appendN(rows, @intCast(mlx.getShape(rows[0])[0]), s);
    }

    /// Room for `n` more rows, grown as `append` would grow it, for a caller that writes the rows itself.
    pub fn reserveAppend(self: *Store, n: usize, s: S) !void {
        if (n == 0) return;
        if (self.used + n > self.cap) try self.growTo(self.pagedTo(appendCap(self.used, n, self.cap, self.div), s), s);
    }

    /// The store sized for `rows` rows in all, made now, between forwards: what a bill of them holds (`capBound`), grown
    /// or cut back to it, so no append within them grows the store and it holds no more than that bill.
    pub fn reserveRows(self: *Store, rows: usize, s: S) !void {
        if (rows < self.used) return error.RowStoreReserveBelowRows;
        const cap = self.pagedTo(capBound(rows, self.div), s);
        if (cap != self.cap) try self.growTo(cap, s);
    }

    /// `append` of `n` rows a caller already knows: each shape read is an FFI call on the decode path.
    pub fn appendN(self: *Store, rows: []const A, n: usize, s: S) !void {
        if (rows.len != self.n) return error.RowStorePartsMismatch;
        if (n == 0) return;
        try self.reserveAppend(n, s);
        for (self.bufs[0..self.n], rows) |*b, x| try put(b, x, self.used, s);
        self.used += n;
    }

    /// The store cut to its first `n` rows: the rows past them are the slack the next append writes over.
    pub fn truncate(self: *Store, n: usize) !void {
        if (n > self.used) return error.RowStoreTruncatePastRows;
        self.used = n;
    }

    /// Whether the store is no larger than `billed` rows on the page grid: what a bill of those rows pays for.
    pub fn fitsBill(self: *const Store, billed: usize, s: S) bool {
        return self.cap <= self.pagedTo(billed, s);
    }

    /// The grow the next append would make, made now and written: buffers of the store's own with that slack.
    pub fn reserve(self: *Store, s: S) !void {
        if (self.used == 0) return;
        try self.growTo(self.pagedTo(reserveCap(self.used, self.div), s), s);
        var i = self.n;
        while (i > 0) {
            i -= 1;
            try mlx.check(mlx.mlx_array_eval(self.bufs[i].array));
        }
    }

    /// The buffers cut back to the rows they hold with `reserve`'s slack, or gone when they hold none.
    pub fn trim(self: *Store, s: S) !void {
        if (self.used > 0) return self.reserve(s);
        self.deinit();
    }

    /// Makes the store `rows` (one array a part, the same rows each): shared handles, the next append copies them into
    /// a buffer with slack. Empty arrays leave the store empty.
    pub fn adopt(self: *Store, rows: []const A) !void {
        if (rows.len != self.n) return error.RowStorePartsMismatch;
        self.deinit();
        const n: usize = @intCast(mlx.getShape(rows[0])[0]);
        if (n == 0) return;
        for (self.bufs[0..self.n], rows) |*b, x| {
            var h = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&h, x));
            b.* = mlx_pages.InPlace.adopt(h, .paged);
        }
        self.used = n;
        self.cap = n;
    }

    /// The bytes the buffers hold: every part at its cap.
    pub fn bytes(self: *const Store) u64 {
        return @as(u64, self.cap) * self.rowBytes();
    }
};

test "row store: rows appended in chunks read back as their concatenation, a grow keeps them, truncate drops the tail the next append writes over" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var vals: [200 * 3]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @floatFromInt(i);
    const all = try mlx.newData(&vals, &[_]c_int{ 200, 3 }, .float32);
    defer _ = mlx.mlx_array_free(all);
    var st = Store.init(&.{ .{ .cols = 3, .dtype = .float32 }, .{ .cols = 3, .dtype = .float32 } }, 16);
    defer st.deinit();
    var at: usize = 0;
    for ([_]usize{ 1, 7, 60, 2, 100, 30 }) |n| {
        var part = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(part);
        try mlx.check(mlx.mlx_slice(&part, all, &[_]c_int{ ci(at), 0 }, 2, &[_]c_int{ ci(at + n), 3 }, 2, &[_]c_int{ 1, 1 }, 2, s));
        try st.append(&.{ part, part }, s);
        at += n;
        try std.testing.expect(st.cap >= st.used and st.cap <= capBound(at, 16));
    }
    try std.testing.expectEqual(@as(usize, 200), st.used);
    try st.truncate(150);
    var tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail);
    try mlx.check(mlx.mlx_zeros(&tail, &[_]c_int{ 50, 3 }, 2, .float32, s));
    try st.append(&.{ tail, tail }, s);
    for (0..2) |i| {
        var head = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(head);
        try mlx.check(mlx.mlx_slice(&head, st.buf(i), &[_]c_int{ 0, 0 }, 2, &[_]c_int{ 200, 3 }, 2, &[_]c_int{ 1, 1 }, 2, s));
        var c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_contiguous(&c, head, false, s));
        try mlx.check(mlx.mlx_array_eval(c));
        const got = (mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..600];
        try std.testing.expectEqualSlices(f32, vals[0..450], got[0..450]);
        for (got[450..]) |v| try std.testing.expectEqual(@as(f32, 0), v);
    }
    try std.testing.expect(st.fitsBill(capBound(200, 16), s));
    try std.testing.expect(!st.fitsBill(50, s));
}

test "row store: a grow that fails at any part's MLX call leaves every part and the cap as they were" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const parts = [_]Part{
        .{ .cols = 3, .dtype = .float32 },
        .{ .cols = 2, .dtype = .float32 },
        .{ .cols = 4, .dtype = .uint8 },
    };
    var f0: [128 * 3]f32 = undefined;
    for (&f0, 0..) |*v, i| v.* = @floatFromInt(i + 1);
    var f1: [128 * 2]f32 = undefined;
    for (&f1, 0..) |*v, i| v.* = @floatFromInt(i + 1001);
    var b2: [128 * 4]u8 = undefined;
    for (&b2, 0..) |*v, i| v.* = @intCast((i % 251) + 1);

    const a0 = try mlx.newData(&f0, &[_]c_int{ 128, 3 }, .float32);
    defer _ = mlx.mlx_array_free(a0);
    const a1 = try mlx.newData(&f1, &[_]c_int{ 128, 2 }, .float32);
    defer _ = mlx.mlx_array_free(a1);
    const a2 = try mlx.newData(&b2, &[_]c_int{ 128, 4 }, .uint8);
    defer _ = mlx.mlx_array_free(a2);

    var init0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(init0);
    var init1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(init1);
    var init2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(init2);
    try mlx.check(mlx.mlx_slice(&init0, a0, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ 64, 3 }, 2, &[_]c_int{ 1, 1 }, 2, s));
    try mlx.check(mlx.mlx_slice(&init1, a1, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ 64, 2 }, 2, &[_]c_int{ 1, 1 }, 2, s));
    try mlx.check(mlx.mlx_slice(&init2, a2, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ 64, 4 }, 2, &[_]c_int{ 1, 1 }, 2, s));

    var next0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(next0);
    var next1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(next1);
    var next2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(next2);
    try mlx.check(mlx.mlx_slice(&next0, a0, &[_]c_int{ 64, 0 }, 2, &[_]c_int{ 124, 3 }, 2, &[_]c_int{ 1, 1 }, 2, s));
    try mlx.check(mlx.mlx_slice(&next1, a1, &[_]c_int{ 64, 0 }, 2, &[_]c_int{ 124, 2 }, 2, &[_]c_int{ 1, 1 }, 2, s));
    try mlx.check(mlx.mlx_slice(&next2, a2, &[_]c_int{ 64, 0 }, 2, &[_]c_int{ 124, 4 }, 2, &[_]c_int{ 1, 1 }, 2, s));

    const checkRows = struct {
        fn run(st: *const Store, n_rows: usize, exp0: []const f32, exp1: []const f32, exp2: []const u8, stream: S) !void {
            var h0 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(h0);
            try mlx.check(mlx.mlx_slice(&h0, st.buf(0), &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(n_rows), 3 }, 2, &[_]c_int{ 1, 1 }, 2, stream));
            var c0 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(c0);
            try mlx.check(mlx.mlx_contiguous(&c0, h0, false, stream));
            try mlx.check(mlx.mlx_array_eval(c0));
            try std.testing.expectEqualSlices(f32, exp0, (mlx.mlx_array_data_float32(c0) orelse return error.HostRead)[0 .. n_rows * 3]);

            var h1 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(h1);
            try mlx.check(mlx.mlx_slice(&h1, st.buf(1), &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(n_rows), 2 }, 2, &[_]c_int{ 1, 1 }, 2, stream));
            var c1 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(c1);
            try mlx.check(mlx.mlx_contiguous(&c1, h1, false, stream));
            try mlx.check(mlx.mlx_array_eval(c1));
            try std.testing.expectEqualSlices(f32, exp1, (mlx.mlx_array_data_float32(c1) orelse return error.HostRead)[0 .. n_rows * 2]);

            var h2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(h2);
            try mlx.check(mlx.mlx_slice(&h2, st.buf(2), &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(n_rows), 4 }, 2, &[_]c_int{ 1, 1 }, 2, stream));
            var c2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(c2);
            try mlx.check(mlx.mlx_contiguous(&c2, h2, false, stream));
            try mlx.check(mlx.mlx_array_eval(c2));
            try std.testing.expectEqualSlices(u8, exp2, (mlx.mlx_array_data_uint8(c2) orelse return error.HostRead)[0 .. n_rows * 4]);
        }
    }.run;

    for (1..10) |k| {
        var st = Store.init(&parts, 16);
        defer st.deinit();
        try st.appendN(&.{ init0, init1, init2 }, 64, s);
        try std.testing.expectEqual(@as(usize, 64), st.used);
        try std.testing.expectEqual(@as(usize, 64), st.cap);

        mlx.fault.arm(k);
        const res = st.appendN(&.{ next0, next1, next2 }, 60, s);
        const fired = mlx.fault.didFire();
        mlx.fault.disarm();
        try std.testing.expect(fired);
        try std.testing.expectError(error.MlxError, res);

        try std.testing.expectEqual(@as(usize, 64), st.cap);
        try std.testing.expectEqual(@as(usize, 64), st.used);
        for (0..parts.len) |i| try std.testing.expectEqual(st.cap, @as(usize, @intCast(mlx.getShape(st.buf(i))[0])));
        try checkRows(&st, 64, f0[0 .. 64 * 3], f1[0 .. 64 * 2], b2[0 .. 64 * 4], s);

        try st.appendN(&.{ next0, next1, next2 }, 60, s);
        try std.testing.expectEqual(@as(usize, 124), st.used);
        for (0..parts.len) |i| try std.testing.expectEqual(st.cap, @as(usize, @intCast(mlx.getShape(st.buf(i))[0])));
        try checkRows(&st, 124, f0[0 .. 124 * 3], f1[0 .. 124 * 2], b2[0 .. 124 * 4], s);
    }

    var exp_after0: [70 * 3]f32 = undefined;
    @memcpy(exp_after0[0 .. 10 * 3], f0[0 .. 10 * 3]);
    @memcpy(exp_after0[10 * 3 .. 70 * 3], f0[64 * 3 .. 124 * 3]);
    var exp_after1: [70 * 2]f32 = undefined;
    @memcpy(exp_after1[0 .. 10 * 2], f1[0 .. 10 * 2]);
    @memcpy(exp_after1[10 * 2 .. 70 * 2], f1[64 * 2 .. 124 * 2]);
    var exp_after2: [70 * 4]u8 = undefined;
    @memcpy(exp_after2[0 .. 10 * 4], b2[0 .. 10 * 4]);
    @memcpy(exp_after2[10 * 4 .. 70 * 4], b2[64 * 4 .. 124 * 4]);

    for (1..10) |k| {
        var st = Store.init(&parts, 16);
        defer st.deinit();
        try st.appendN(&.{ a0, a1, a2 }, 128, s);
        try std.testing.expectEqual(@as(usize, 128), st.used);
        try std.testing.expectEqual(@as(usize, 128), st.cap);
        try st.truncate(10);

        mlx.fault.arm(k);
        const res = st.trim(s);
        const fired = mlx.fault.didFire();
        mlx.fault.disarm();
        try std.testing.expect(fired);
        try std.testing.expectError(error.MlxError, res);

        try std.testing.expectEqual(@as(usize, 128), st.cap);
        try std.testing.expectEqual(@as(usize, 10), st.used);
        for (0..parts.len) |i| try std.testing.expectEqual(st.cap, @as(usize, @intCast(mlx.getShape(st.buf(i))[0])));
        try checkRows(&st, 10, f0[0 .. 10 * 3], f1[0 .. 10 * 2], b2[0 .. 10 * 4], s);

        try st.appendN(&.{ next0, next1, next2 }, 60, s);
        try std.testing.expectEqual(@as(usize, 70), st.used);
        try std.testing.expectEqual(@as(usize, 128), st.cap);
        for (0..parts.len) |i| try std.testing.expectEqual(st.cap, @as(usize, @intCast(mlx.getShape(st.buf(i))[0])));
        try checkRows(&st, 70, &exp_after0, &exp_after1, &exp_after2, s);
    }
}

test "row store: an append on the engine lands at once: the rows it was handed die inside the execution" {
    try mlx.requireGpu();
    const core = @import("tensor/core.zig");
    const plan = @import("tensor/plan/plan_api.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("test_support.zig").restoreEngine(was);
    const s = mlx.engine.stream();
    const scope = mlx.engine.enter(s);
    defer scope.end();
    var st = Store.init(&.{.{ .cols = 4, .dtype = .float32 }}, 16);
    defer st.deinit();
    // Room first: a grow inside the recording would be an ask of its own.
    try st.reserveAppend(8, s);
    var rec = plan.Recorder.begin(std.testing.allocator);
    defer rec.deinit();
    {
        try core.beginRecording(.real, &rec);
        defer core.endRecording() catch {};
        const vals = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        const rows = try mlx.newData(&vals, &[_]c_int{ 2, 4 }, .float32);
        try st.append(&.{rows}, s);
        _ = mlx.mlx_array_free(rows);
    }
    const asked = rec.asks.items;
    try std.testing.expect(asked.len >= 1);
    try std.testing.expect(asked[0].death != @import("tensor/plan/plan_trace.zig").none);
}

test "row store: an append whose write fails to land leaves the store its buffer, released once by the store" {
    try mlx.requireGpu();
    const core = @import("tensor/core.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("test_support.zig").restoreEngine(was);
    const s = mlx.engine.stream();
    const scope = mlx.engine.enter(s);
    defer scope.end();
    const released_twice = core.double_releases;
    {
        var st = Store.init(&.{.{ .cols = 4, .dtype = .float32 }}, 16);
        defer st.deinit();
        try st.reserveAppend(8, s);
        const vals = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        const rows = try mlx.newData(&vals, &[_]c_int{ 2, 4 }, .float32);
        defer _ = mlx.mlx_array_free(rows);
        // A second holder of the buffer: the write cannot go in place, so landing it copies, and the copy is refused.
        var other = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(other);
        try mlx.check(mlx.mlx_array_set(&other, st.buf(0)));
        core.fault.arm(1);
        defer core.fault.disarm();
        try std.testing.expectError(error.PoolOverBudget, st.append(&.{rows}, s));
        core.fault.disarm();
        var buf: [256]u8 = undefined;
        _ = mlx.takeError(&buf);
        try std.testing.expectEqual(@as(usize, 0), st.used);
    }
    try std.testing.expectEqual(released_twice, core.double_releases);
    var buf: [256]u8 = undefined;
    _ = mlx.takeError(&buf);
}
