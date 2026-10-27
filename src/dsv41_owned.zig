//! DeepSeek-V4.1 weights that sushi builds in its own mappings instead of MLX's allocator: a build step writes
//! the bytes on the host into a `model.ownedRegion` and wraps it as an array (`model.wrapOwned`, no copy), so
//! the weight memory is the engine's and a refused import is an error.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const sc_mod = @import("dsv41_scope.zig");
const exl3 = @import("sushi_exl3");

const A = sc_mod.A;

fn rowContiguous(x: A) bool {
    const sh = mlx.getShape(x);
    const st = mlx.mlx_array_strides(x);
    var expect: usize = 1;
    var i = sh.len;
    while (i > 0) {
        i -= 1;
        if (sh[i] != 1 and st[i] != expect) return false;
        expect *= @intCast(sh[i]);
    }
    return true;
}

/// `x`'s bytes on the host: evaluated, row-contiguous, `want` bytes.
fn hostBytes(x: A, want: usize) ![]const u8 {
    try mlx.check(mlx.mlx_array_eval(x));
    if (!rowContiguous(x) or mlx.mlx_array_size(x) * mlx.mlx_array_itemsize(x) != want) return error.OwnedSourceLayout;
    const p = mlx.mlx_array_data_uint8(x) orelse return error.OwnedSourceLayout;
    return p[0..want];
}

/// Bf16 `x` as f32 in an owned region, bit-equal to MLX's `astype`: the f32 bits are the bf16 bits shifted up.
/// Null when `x` is not bf16 or the f32 copy is under `import_min_bytes`; the caller then casts in MLX.
pub fn widenF32(x: A) !?A {
    if (mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const n = mlx.mlx_array_size(x);
    if (n * 4 < model.import_min_bytes) return null;
    const src = std.mem.bytesAsSlice(u16, try hostBytes(x, n * 2));
    const region = try model.ownedRegion(n * 4);
    const dst = std.mem.bytesAsSlice(u32, region.ptr[0 .. n * 4]);
    for (dst, src) |*d, b| d.* = @as(u32, b) << 16;
    return try model.wrapOwned(region, mlx.getShape(x), .float32);
}

/// How `stack` joins its parts.
pub const StackKind = enum {
    /// `[1, in/16, out_i/16, n]` joined on axis 2.
    trellis,
    /// `[1, w]` rows joined on axis 0 (`suh`).
    rows,
    /// `[1, w_i]` joined on axis 1 (`svh`).
    cols,
};

/// `parts` stacked as `mlx_concatenate_axis` would on the axis of `kind`, in an owned region; null when the
/// result is under `import_min_bytes`.
pub fn stack(parts: []const A, kind: StackKind) !?A {
    const first = parts[0];
    const item = mlx.mlx_array_itemsize(first);
    const dtype = mlx.mlx_array_dtype(first);
    var total: usize = 0;
    for (parts) |p| total += mlx.mlx_array_size(p) * item;
    if (total < model.import_min_bytes) return null;
    const sh0 = mlx.getShape(first);
    var shape: [4]c_int = @splat(0);
    @memcpy(shape[0..sh0.len], sh0);
    switch (kind) {
        .trellis => shape[2] = 0,
        .rows => shape[0] = 0,
        .cols => shape[1] = 0,
    }
    for (parts) |p| {
        const sh = mlx.getShape(p);
        if (sh.len != sh0.len or mlx.mlx_array_dtype(p) != dtype) return error.OwnedSourceLayout;
        switch (kind) {
            .trellis => shape[2] += sh[2],
            .rows => shape[0] += sh[0],
            .cols => shape[1] += sh[1],
        }
    }
    const region = try model.ownedRegion(total);
    errdefer model.destroyRegion(region);
    const out = region.ptr[0..total];
    var at: usize = 0;
    switch (kind) {
        .trellis => {
            // Each tile row of the stack holds every part's tiles of that row, parts in order.
            const row_elems: usize = @intCast(shape[2] * shape[3]);
            for (0..@intCast(shape[1])) |r| {
                var col: usize = 0;
                for (parts) |p| {
                    const w: usize = @intCast(mlx.getShape(p)[2] * mlx.getShape(p)[3]);
                    const src = try hostBytes(p, mlx.mlx_array_size(p) * item);
                    @memcpy(out[(r * row_elems + col) * item ..][0 .. w * item], src[r * w * item ..][0 .. w * item]);
                    col += w;
                }
            }
        },
        .rows, .cols => for (parts) |p| {
            const bytes = mlx.mlx_array_size(p) * item;
            @memcpy(out[at..][0..bytes], try hostBytes(p, bytes));
            at += bytes;
        },
    }
    return try model.wrapOwned(region, shape[0..sh0.len], dtype);
}

/// Called after group `g`'s source arrays are copied, so the caller can free them before the next group.
pub const Release = struct {
    ctx: ?*anyopaque,
    run: *const fn (?*anyopaque, usize) void,
};

/// The ragged bank `exl3.Ragged.init` builds from `banks`, with its nine storage arrays in owned regions: each
/// projection's trellis and scales are the groups' bytes back to back.
pub fn raggedOwned(s: mlx.mlx_stream, banks: []const exl3.Bank, release: ?Release) !exl3.Ragged {
    const max_groups = exl3.group_layout.max_groups;
    if (banks.len == 0 or banks.len > max_groups) return error.Exl3GroupMissing;
    const g0 = mlx.getShape(banks[0].gate.trellis);
    if (g0.len != 4) return error.BadExl3Shape;
    const hidden: usize = @intCast(g0[1] * 16);
    const inter: usize = @intCast(g0[2] * 16);
    var pools: [max_groups]exl3.rate_groups.Pool = undefined;
    var trellis_bytes: [3]usize = @splat(0);
    var rows: usize = 0;
    for (banks, 0..) |bank, g| {
        const e = mlx.getShape(bank.gate.trellis)[0];
        pools[g] = .{ .slots = @intCast(e), .rates = undefined };
        for ([3]exl3.Proj{ bank.gate, bank.up, bank.down }, 0..) |p, pi| {
            const in: usize = if (pi == 2) inter else hidden;
            const out: usize = if (pi == 2) hidden else inter;
            const sh = mlx.getShape(p.trellis);
            const ss = mlx.getShape(p.suh);
            const sv = mlx.getShape(p.svh);
            if (sh.len != 4 or sh[0] != e or sh[1] * 16 != in or sh[2] * 16 != out or mlx.mlx_array_dtype(p.trellis) != .uint16 or
                ss.len != 2 or ss[0] != e or ss[1] != in or sv.len != 2 or sv[0] != e or sv[1] != out or
                mlx.mlx_array_dtype(p.suh) != .float16 or mlx.mlx_array_dtype(p.svh) != .float16) return error.BadExl3Shape;
            pools[g].rates[pi] = (exl3.format.kFromPackedDim(@intCast(sh[3])) orelse return error.BadExl3Shape).n;
            trellis_bytes[pi] += mlx.mlx_array_size(p.trellis) * 2;
        }
        rows += @intCast(e);
    }

    var regions: [9]?*model.MappedRegion = @splat(null);
    errdefer for (regions) |r| if (r) |p| model.destroyRegion(p);
    for (0..3) |pi| {
        const in: usize = if (pi == 2) inter else hidden;
        const out: usize = if (pi == 2) hidden else inter;
        regions[pi] = try model.ownedRegion(trellis_bytes[pi]);
        regions[3 + pi] = try model.ownedRegion(rows * in * 2);
        regions[6 + pi] = try model.ownedRegion(rows * out * 2);
    }
    var at: [9]usize = @splat(0);
    for (banks, 0..) |bank, g| {
        for ([3]exl3.Proj{ bank.gate, bank.up, bank.down }, 0..) |p, pi| {
            for ([3]A{ p.trellis, p.suh, p.svh }, 0..) |a, part| {
                const slot = part * 3 + pi;
                const bytes = mlx.mlx_array_size(a) * mlx.mlx_array_itemsize(a);
                @memcpy(regions[slot].?.ptr[at[slot]..][0..bytes], try hostBytes(a, bytes));
                at[slot] += bytes;
            }
        }
        if (release) |r| r.run(r.ctx, g);
    }

    var arrays: [9]A = @splat(.{ .ctx = null });
    defer for (arrays) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    // `wrapOwned` takes the region whether it succeeds or not.
    const taken = struct {
        fn of(r: *?*model.MappedRegion) *model.MappedRegion {
            defer r.* = null;
            return r.*.?;
        }
    }.of;
    for (0..3) |pi| {
        const in: c_int = @intCast(if (pi == 2) inter else hidden);
        const out: c_int = @intCast(if (pi == 2) hidden else inter);
        const rc: c_int = @intCast(rows);
        arrays[pi] = try model.wrapOwned(taken(&regions[pi]), &.{@intCast(trellis_bytes[pi] / 2)}, .uint16);
        arrays[3 + pi] = try model.wrapOwned(taken(&regions[3 + pi]), &.{ rc, in }, .float16);
        arrays[6 + pi] = try model.wrapOwned(taken(&regions[6 + pi]), &.{ rc, out }, .float16);
    }
    return exl3.Ragged.initPools(s, @intCast(hidden), @intCast(inter), pools[0..banks.len], arrays[0..3].*, arrays[3..6].*, arrays[6..9].*);
}

/// What an owned set holds, by where its buffers live.
pub const Audit = struct {
    /// Arrays whose data lies in an owned region, and their bytes.
    owned_arrays: usize = 0,
    owned_bytes: u64 = 0,
    /// The others, which MLX's allocator holds, and their bytes (one buffer once).
    mlx_arrays: usize = 0,
    mlx_bytes: u64 = 0,
    /// Of those, the arrays of at least `import_min_bytes`.
    mlx_large: usize = 0,
};

/// Classifies `arrays` by whether their data pointer lies in an owned region. Evaluates them.
pub fn audit(gpa: std.mem.Allocator, arrays: []const A) !Audit {
    var out: Audit = .{};
    var seen: std.AutoHashMapUnmanaged(usize, void) = .empty;
    defer seen.deinit(gpa);
    for (arrays) |a| {
        if (a.ctx == null) continue;
        try mlx.check(mlx.mlx_array_eval(a));
        const bytes: u64 = mlx.mlx_array_size(a) * mlx.mlx_array_itemsize(a);
        const ptr = mlx.mlx_array_data_uint8(a);
        if (ptr != null and model.ownsPointer(ptr)) {
            out.owned_arrays += 1;
            out.owned_bytes += bytes;
            continue;
        }
        if ((try seen.getOrPut(gpa, @intFromPtr(ptr))).found_existing) continue;
        out.mlx_arrays += 1;
        out.mlx_bytes += bytes;
        if (bytes >= model.import_min_bytes) out.mlx_large += 1;
    }
    return out;
}

const testing = std.testing;

/// A fresh MLX array of `shape`/`dtype` filled with random bytes.
fn randomArray(rng: *std.Random.DefaultPrng, shape: []const c_int, dtype: mlx.mlx_dtype) !A {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const bytes = try testing.allocator.alloc(u8, n * 2);
    defer testing.allocator.free(bytes);
    rng.random().bytes(bytes);
    const a = mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), dtype);
    try mlx.check(mlx.mlx_array_eval(a));
    return a;
}

fn expectSameBytes(want: A, got: A) !void {
    try mlx.check(mlx.mlx_array_eval(want));
    try mlx.check(mlx.mlx_array_eval(got));
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    try testing.expectEqual(mlx.mlx_array_dtype(want), mlx.mlx_array_dtype(got));
    const n = mlx.mlx_array_size(want) * mlx.mlx_array_itemsize(want);
    try testing.expectEqualSlices(u8, mlx.mlx_array_data_uint8(want).?[0..n], mlx.mlx_array_data_uint8(got).?[0..n]);
}

test "dsv41 owned: the owned ragged bank is exl3.Ragged.init's storage, meta, views and offsets" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var rng = std.Random.DefaultPrng.init(7);
    // Three groups of different sizes and rates over hidden 128, inter 64.
    const experts = [_]c_int{ 2, 3, 1 };
    const rates = [_][3]c_int{ .{ 32, 48, 32 }, .{ 64, 32, 48 }, .{ 48, 48, 64 } };
    var banks: [3]exl3.Bank = undefined;
    defer for (banks) |b| for ([_]A{ b.gate.trellis, b.gate.suh, b.gate.svh, b.up.trellis, b.up.suh, b.up.svh, b.down.trellis, b.down.suh, b.down.svh }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&banks, experts, rates) |*b, e, r| {
        b.gate = .{ .trellis = try randomArray(&rng, &.{ e, 8, 4, r[0] }, .uint16), .suh = try randomArray(&rng, &.{ e, 128 }, .float16), .svh = try randomArray(&rng, &.{ e, 64 }, .float16) };
        b.up = .{ .trellis = try randomArray(&rng, &.{ e, 8, 4, r[1] }, .uint16), .suh = try randomArray(&rng, &.{ e, 128 }, .float16), .svh = try randomArray(&rng, &.{ e, 64 }, .float16) };
        b.down = .{ .trellis = try randomArray(&rng, &.{ e, 4, 8, r[2] }, .uint16), .suh = try randomArray(&rng, &.{ e, 64 }, .float16), .svh = try randomArray(&rng, &.{ e, 128 }, .float16) };
    }
    var want = try exl3.Ragged.init(cpu, &banks);
    defer want.deinit();
    const Released = struct {
        var seen: usize = 0;
        fn run(_: ?*anyopaque, g: usize) void {
            seen += 1;
            std.debug.assert(g + 1 == seen);
        }
    };
    var got = try raggedOwned(cpu, &banks, .{ .ctx = null, .run = Released.run });
    defer got.deinit();
    try testing.expectEqual(banks.len, Released.seen);

    try testing.expectEqual(want.count, got.count);
    try testing.expectEqual(want.experts, got.experts);
    try testing.expectEqual(want.hidden, got.hidden);
    try testing.expectEqual(want.inter, got.inter);
    try testing.expectEqual(want.view_count, got.view_count);
    try testing.expectEqual(want.tile_major, got.tile_major);
    try testing.expectEqual(want.prefill_only, got.prefill_only);
    for (0..want.count) |g| {
        try testing.expectEqual(want.base[g], got.base[g]);
        try testing.expectEqual(want.experts_of[g], got.experts_of[g]);
        try testing.expectEqual(want.rates[g], got.rates[g]);
    }
    for (want.storage(), got.storage()) |w, a| try expectSameBytes(w, a);
    for (0..3) |pi| try testing.expect(model.ownsPointer(mlx.mlx_array_data_uint8(got.trellis[pi])) and model.ownsPointer(mlx.mlx_array_data_uint8(got.suh[pi])) and model.ownsPointer(mlx.mlx_array_data_uint8(got.svh[pi])));
    // Each view reads the same shape at the same offset into its storage.
    for (want.views[0..want.view_count], got.views[0..got.view_count]) |wv, gv| {
        for ([3]exl3.Proj{ wv.gate, wv.up, wv.down }, [3]exl3.Proj{ gv.gate, gv.up, gv.down }, 0..) |wp, gp, pi| {
            const w_storage = [3]A{ want.trellis[pi], want.suh[pi], want.svh[pi] };
            const g_storage = [3]A{ got.trellis[pi], got.suh[pi], got.svh[pi] };
            for ([3]A{ wp.trellis, wp.suh, wp.svh }, [3]A{ gp.trellis, gp.suh, gp.svh }, w_storage, g_storage) |w, a, ws, gs| {
                try expectSameBytes(w, a);
                const w_at = @intFromPtr(mlx.mlx_array_data_uint8(w).?) - @intFromPtr(mlx.mlx_array_data_uint8(ws).?);
                const g_at = @intFromPtr(mlx.mlx_array_data_uint8(a).?) - @intFromPtr(mlx.mlx_array_data_uint8(gs).?);
                try testing.expectEqual(w_at, g_at);
            }
        }
    }
}

test "dsv41 owned: the stacks are mlx_concatenate_axis's bytes on each axis" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const min = model.import_min_bytes;
    model.import_min_bytes = 1;
    defer model.import_min_bytes = min;
    var rng = std.Random.DefaultPrng.init(11);
    const cases = [_]struct { kind: StackKind, axis: c_int, shapes: [3][4]c_int, dtype: mlx.mlx_dtype }{
        .{ .kind = .trellis, .axis = 2, .shapes = .{ .{ 1, 8, 6, 32 }, .{ 1, 8, 4, 32 }, .{ 1, 8, 2, 32 } }, .dtype = .uint16 },
        .{ .kind = .rows, .axis = 0, .shapes = .{ .{ 1, 128, 0, 0 }, .{ 1, 128, 0, 0 }, .{ 1, 128, 0, 0 } }, .dtype = .float16 },
        .{ .kind = .cols, .axis = 1, .shapes = .{ .{ 1, 96, 0, 0 }, .{ 1, 64, 0, 0 }, .{ 1, 32, 0, 0 } }, .dtype = .float16 },
    };
    for (cases) |c| {
        var parts: [3]A = undefined;
        const nd: usize = if (c.kind == .trellis) 4 else 2;
        for (&parts, c.shapes) |*p, sh| p.* = try randomArray(&rng, sh[0..nd], c.dtype);
        defer for (parts) |p| {
            _ = mlx.mlx_array_free(p);
        };
        const vec = mlx.mlx_vector_array_new_data(&parts, parts.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_concatenate_axis(&want, vec, c.axis, cpu));
        const got = (try stack(&parts, c.kind)).?;
        defer _ = mlx.mlx_array_free(got);
        try expectSameBytes(want, got);
        try testing.expect(model.ownsPointer(mlx.mlx_array_data_uint8(got)));
    }
    model.import_min_bytes = 1 << 20;
    const tiny = try randomArray(&rng, &.{ 1, 8 }, .float16);
    defer _ = mlx.mlx_array_free(tiny);
    try testing.expect((try stack(&.{ tiny, tiny }, .rows)) == null);
}

test "dsv41 owned: a bf16 tensor widens to MLX's astype f32 bits, NaN, Inf and denormals included" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const min = model.import_min_bytes;
    model.import_min_bytes = 1;
    defer model.import_min_bytes = min;
    var bits: [4096]u16 = undefined;
    var rng = std.Random.DefaultPrng.init(3);
    rng.random().bytes(std.mem.sliceAsBytes(&bits));
    // 0x7fc0 quiet NaN, 0xffc1 negative NaN with a payload, 0x7f80/0xff80 +-Inf, 0x8000 -0, 0x0001 denormal.
    const special = [_]u16{ 0x7fc0, 0xffc1, 0x7f80, 0xff80, 0x8000, 0x0001 };
    @memcpy(bits[0..special.len], &special);
    const x = mlx.mlx_array_new_data(&bits, &[_]c_int{ 64, 64 }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_astype(&want, x, .float32, cpu));
    const got = (try widenF32(x)).?;
    defer _ = mlx.mlx_array_free(got);
    try expectSameBytes(want, got);
    try testing.expect(model.ownsPointer(mlx.mlx_array_data_uint8(got)));
    const f32_bits = std.mem.bytesAsSlice(u32, mlx.mlx_array_data_uint8(got).?[0 .. special.len * 4]);
    try testing.expectEqual(@as(u32, 0x7fc00000), f32_bits[0]);
    try testing.expectEqual(@as(u32, 0xff800000), f32_bits[3]);

    const wide = mlx.mlx_array_new_data(&bits, &[_]c_int{4096}, 1, .float16);
    defer _ = mlx.mlx_array_free(wide);
    try testing.expect((try widenF32(wide)) == null);
    model.import_min_bytes = 1 << 20;
    try testing.expect((try widenF32(x)) == null);
}
