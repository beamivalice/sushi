//! MLX's matrix product on the engine: `mlx::core::matmul` and `Matmul::eval_gpu` (mlx/ops.cpp,
//! mlx/backend/metal/matmul.cpp), choice for choice, over the compiled kernels of `sushi.metallib`, so a product here
//! writes the bytes it writes under MLX. Ported from MLX's Metal backend (MIT, see NOTICE). An empty operand is not
//! ported.

const std = @import("std");
const builtin = @import("builtin");
const metal = @import("../metal.zig");
const mlx = @import("../mlx.zig");
const core = @import("core.zig");
const plan_api = @import("plan/plan_api.zig");
const device = @import("sushi_exl3").device;
const ops = @import("ops.zig");
const stock = @import("stock.zig");
const kernel = @import("kernel.zig");
const log = @import("../log.zig");

const Tensor = core.Tensor;
const Error = core.Error;
const max_batch = core.max_dims;

/// The GPU as MLX's dispatch reads it: its class letter, its generation, and whether the NAX kernels run (macOS 26.2
/// is this engine's floor) and take f32.
pub const Gpu = struct { class: device.Class, gen: u32, nax: bool, tf32: bool };

/// Whether MLX takes f32 products on TF32 tiles, read once: MLX keeps its first reading of the variable for the process,
/// and its tiles follow the chip, so the chip's units decide how it is read, not what the profile withholds.
pub var tf32_known: ?bool = null;

fn atoi(text: []const u8) i64 {
    var end: usize = 0;
    while (end < text.len and (std.ascii.isDigit(text[end]) or (end == 0 and (text[end] == '-' or text[end] == '+')))) end += 1;
    return std.fmt.parseInt(i64, text[0..end], 10) catch 0;
}

/// Inside `exactF32`, an f32 product takes MLX's f32 tiles whatever its TF32 reading (what `MLX_ENABLE_TF32=0` gives).
var exact_f32: bool = false;

/// For a forward whose f32 products are held to f32: every product, gathered product and attention of f32 operands
/// the engine runs until `ExactF32.end` keeps f32 tiles.
pub fn exactF32() ExactF32 {
    const was = exact_f32;
    exact_f32 = true;
    return .{ .was = was };
}

pub const ExactF32 = struct {
    was: bool,

    pub fn end(self: ExactF32) void {
        exact_f32 = self.was;
    }
};

/// The GPU as the profile gives it, so a withheld unit or a test's profile is the engine's too.
pub fn gpu() Error!Gpu {
    const p = device.current();
    if (tf32_known == null) {
        // MLX's own override, read as MLX reads it.
        var tf32 = if (std.c.getenv("MLX_ENABLE_TF32")) |v| atoi(std.mem.span(v)) != 0 else true;
        if (comptime builtin.is_test) {
            if (p.chip.contains(.matrix_g17)) tf32 = mlxTf32() catch tf32;
        }
        tf32_known = tf32;
    }
    return .{ .class = p.class, .gen = p.generation, .nax = p.has(.matrix_g17), .tf32 = tf32_known.? and !exact_f32 };
}

/// In a test: what MLX's own reading answers (`mlxTf32`), and the engine's cached reading.
pub var mlx_tf32_for_test: ?bool = null;

pub fn cacheTf32ForTest(v: ?bool) void {
    tf32_known = v;
}

/// Reads TF32 again, from MLX itself on a chip with the units: a process that changed the variable before MLX's first
/// product gets the new value, one whose MLX already multiplied keeps MLX's first reading. True when f32 products run
/// on TF32 tiles (only a chip with the units has them).
pub fn rereadTf32() Error!bool {
    const p = device.current();
    tf32_known = if (p.chip.contains(.matrix_g17)) try mlxTf32() else if (std.c.getenv("MLX_ENABLE_TF32")) |v| atoi(std.mem.span(v)) != 0 else true;
    const g = try gpu();
    return g.nax and g.tf32;
}

/// Whether the GPU has the matrix units MLX's NAX kernels run on.
pub fn matrixUnits() bool {
    return (gpu() catch return false).nax;
}

/// Whether MLX's f32 products run on the matrix units' TF32 tiles: the arm a byte pin or a twin bound made there holds on.
pub fn tf32Units() bool {
    const g = gpu() catch return false;
    return g.nax and g.tf32;
}

/// Whether MLX takes an f32 product on the NAX tiles, by what a product of its own gives: those keep 10 mantissa bits
/// of an operand, so 1 + 2^-12 multiplies as 1. Tests only: MLX keeps its first reading of `MLX_ENABLE_TF32` for the
/// process, so a test that changes the variable around an MLX product leaves MLX on a value the environment no
/// longer shows, and every test after it that holds the engine to MLX's bytes would fail by the order they ran in.
fn mlxTf32() !bool {
    if (builtin.is_test) if (mlx_tf32_for_test) |v| return v;
    const n = 64;
    const values: [n * n]f32 = @splat(1.0 + 1.0 / 4096.0);
    const x = mlx.raw.mlx_array_new_data(&values, &[_]c_int{ n, n }, 2, .float32);
    defer _ = mlx.raw.mlx_array_free(x);
    const gpu_stream = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu_stream);
    var squared = mlx.raw.mlx_array_new();
    defer _ = mlx.raw.mlx_array_free(squared);
    // The call under test may be the first to read this: the fault it is armed for is its own.
    try mlx.check(mlx.past_fault.mlx_matmul(&squared, x, x, gpu_stream));
    try mlx.check(mlx.raw.mlx_array_eval(squared));
    const got = mlx.mlx_array_data_float32(squared) orelse return error.NotEvaluated;
    return got[0] == n;
}

pub const Operand = struct { t: *Tensor, transposed: bool, ld: i64 };

/// MLX's `check_transpose`: the operand as it is when its last two axes are a row-major or a column-major matrix,
/// else a row-contiguous copy.
pub fn matrixOperand(t: *Tensor, is_vector: bool) Error!Operand {
    const stx: i64 = @intCast(t.strides[t.ndim - 2]);
    const sty: i64 = @intCast(t.strides[t.ndim - 1]);
    if (sty == 1 and (!is_vector or stx == t.shape[t.ndim - 1])) return .{ .t = try core.share(t), .transposed = false, .ld = stx };
    if (stx == 1 and (!is_vector or sty == t.shape[t.ndim - 2])) return .{ .t = try core.share(t), .transposed = true, .ld = sty };
    return .{ .t = try ops.contiguousCopy(t), .transposed = false, .ld = t.shape[t.ndim - 1] };
}

/// The batch axes of two operands as the kernels walk them: runs of axes that are one run of memory in both become
/// one axis.
const Batches = struct {
    n: usize = 0,
    shape: [max_batch]c_int = undefined,
    a: [max_batch]i64 = undefined,
    b: [max_batch]i64 = undefined,

    const one: Batches = blk: {
        var x: Batches = .{};
        x.push(1, 0, 0);
        break :blk x;
    };

    fn push(self: *Batches, len: c_int, sa: i64, sb: i64) void {
        self.shape[self.n] = len;
        self.a[self.n] = sa;
        self.b[self.n] = sb;
        self.n += 1;
    }
};

/// MLX's `collapse_batches` over `collapse_contiguous_dims`.
fn collapseBatches(a: *const Tensor, b: *const Tensor) Batches {
    if (a.ndim == 2) return .one;
    const nb = a.ndim - 2;
    const shape = a.shape[0..nb];
    const sa = a.strides[0..nb];
    const sb = b.strides[0..nb];
    const cut: i32 = -1;
    var plan: [2 * max_batch + 2]i32 = undefined;
    var len: usize = 0;
    if (shape[0] != 1) {
        plan[len] = 0;
        len += 1;
    }
    var size: i64 = shape[0];
    for (1..nb) |i| {
        size *= shape[i];
        const run = for ([_][]const usize{ sa, sb }) |st| {
            if (st[i] * @as(usize, @intCast(shape[i])) != st[i - 1] or size > std.math.maxInt(i32)) break false;
        } else true;
        if (!run) {
            size = shape[i];
            plan[len] = cut;
            len += 1;
        }
        if (shape[i] != 1) {
            plan[len] = @intCast(i);
            len += 1;
        }
    }
    plan[len] = cut;
    len += 1;

    var out: Batches = .{};
    var i: usize = 0;
    while (true) {
        while (i < len and plan[i] == cut) i += 1;
        if (i == len) break;
        var extent: c_int = shape[@intCast(plan[i])];
        var k = i + 1;
        while (plan[k] != cut) : (k += 1) extent *= shape[@intCast(plan[k])];
        const last: usize = @intCast(plan[k - 1]);
        out.push(extent, @intCast(sa[last]), @intCast(sb[last]));
        i = k + 1;
    }
    return if (out.n == 0) .one else out;
}

fn letter(yes: bool) u8 {
    return if (yes) 't' else 'n';
}

/// `a @ b` as MLX's `matmul` gives it: a 1-D operand is a row or a column for the product, batch axes broadcast,
/// and a batched `a` against one matrix is flattened into rows.
pub fn matmul(a_in: *Tensor, b_in: *Tensor) Error!*Tensor {
    return matmulImpl(a_in, b_in, null);
}

/// `matmul` written into `dst`, a contiguous region of a larger buffer (a verify row's slice of its block's result):
/// the same kernel, tile and bytes as `matmul` at that shape, with no copy of the result.
pub fn matmulInto(a_in: *Tensor, b_in: *Tensor, dst: *Tensor) Error!void {
    const out = try matmulImpl(a_in, b_in, dst);
    core.release(out);
}

fn matmulImpl(a_in: *Tensor, b_in: *Tensor, dst_in: ?*Tensor) Error!*Tensor {
    if (a_in.ndim == 0 or b_in.ndim == 0) return error.TensorShape;
    if (a_in.dtype != b_in.dtype) return core.refusing("a matmul of operands of two dtypes", error.TensorDtype);
    switch (a_in.dtype) {
        .float16, .bfloat16, .float32 => {},
        else => return core.refusing("a matmul in a dtype other than f32, f16 and bf16", error.TensorDtype),
    }
    if (a_in.size() == 0 or b_in.size() == 0) return core.refusing("a matmul of an empty operand", error.TensorUnsupported);
    var a = if (a_in.ndim == 1) try ops.expandDims(a_in, 0) else try core.share(a_in);
    defer core.release(a);
    var b = if (b_in.ndim == 1) try ops.expandDims(b_in, 1) else try core.share(b_in);
    defer core.release(b);
    const k = a.shape[a.ndim - 1];
    if (k != b.shape[b.ndim - 2]) return error.TensorShape;
    const n = b.shape[b.ndim - 1];

    const flattened = a_in.ndim > 2 and b_in.ndim <= 2;
    if (flattened) {
        const flat = try ops.reshape(a, &.{ -1, k });
        core.release(a);
        a = flat;
    } else if (a_in.ndim > 2 or b_in.ndim > 2) {
        const an = a.ndim - 2;
        const bn = b.ndim - 2;
        const batch = @max(an, bn);
        var sa: core.Dims = .{};
        var sb: core.Dims = .{};
        for (0..batch) |i| {
            const da: c_int = if (i + an >= batch) a.shape[i + an - batch] else 1;
            const db: c_int = if (i + bn >= batch) b.shape[i + bn - batch] else 1;
            const d = if (da == db or db == 1) da else if (da == 1) db else return error.TensorShape;
            try sa.append(d);
            try sb.append(d);
        }
        try sa.appendSlice(a.shape[an..][0..2]);
        try sb.appendSlice(b.shape[bn..][0..2]);
        const wide_a = try ops.broadcastTo(a, sa.slice());
        core.release(a);
        a = wide_a;
        const wide_b = try ops.broadcastTo(b, sb.slice());
        core.release(b);
        b = wide_b;
    }

    if (dst_in) |d| {
        if (d.store == a_in.store or d.store == b_in.store) return core.refusing("matmulInto with dst aliasing an operand", error.TensorShape);
        var want: core.Dims = .{};
        if (flattened) {
            try want.appendSlice(a_in.shape[0 .. a_in.ndim - 1]);
            if (b_in.ndim != 1) try want.append(n);
        } else {
            try want.appendSlice(a.shape[0 .. a.ndim - 2]);
            if (a_in.ndim != 1) try want.append(a.shape[a.ndim - 2]);
            if (b_in.ndim != 1) try want.append(n);
        }
        if (d.dtype != a_in.dtype or !d.rowContiguous() or d.ndim != want.len) return error.TensorShape;
        for (d.shape[0..d.ndim], want.slice()) |x, y| if (x != y) return error.TensorShape;
    }

    var out = try product(a, b, null, dst_in);
    errdefer core.release(out);
    if (dst_in != null) return out;
    if (flattened) {
        var shape: [core.max_dims]c_int = undefined;
        const lead = a_in.ndim - 1;
        @memcpy(shape[0..lead], a_in.shape[0..lead]);
        shape[lead] = n;
        const shaped = try ops.reshape(out, shape[0 .. lead + 1]);
        core.release(out);
        out = shaped;
    }
    var axes: [2]c_int = undefined;
    var dropped: usize = 0;
    if (a_in.ndim == 1) {
        axes[dropped] = @as(c_int, out.ndim) - 2;
        dropped += 1;
    }
    if (b_in.ndim == 1) {
        axes[dropped] = @as(c_int, out.ndim) - 1;
        dropped += 1;
    }
    if (dropped == 0) return out;
    const squeezed = try ops.squeeze(out, axes[0..dropped]);
    core.release(out);
    return squeezed;
}

/// Two matrices as `matmul` multiplies them, but with the kernel, its tile and its k split chosen as for `rows_as` rows
/// of `a`. MLX picks those by the row count and its arms sum in different orders, so a row's result changes with the rows
/// beside it; a caller whose rows must not depend on their company fixes the count. The rows are computed as they are:
/// nothing is padded.
pub fn matmulRowsAs(a: *Tensor, b: *Tensor, rows_as: usize) Error!*Tensor {
    if (a.ndim < 2 or a.ndim != b.ndim or a.dtype != b.dtype) return error.TensorShape;
    if (a.dtype != .float32 and a.dtype != .float16 and a.dtype != .bfloat16) return core.refusing("a matmul in a dtype other than f32, f16 and bf16", error.TensorDtype);
    if (!std.mem.eql(c_int, a.shape[0 .. a.ndim - 2], b.shape[0 .. b.ndim - 2])) return error.TensorShape;
    if (a.shape[a.ndim - 1] != b.shape[b.ndim - 2]) return error.TensorShape;
    if (a.size() == 0 or b.size() == 0 or rows_as < 2) return core.refusing("a rows-as matmul of an empty operand or under two rows", error.TensorUnsupported);
    return product(a, b, rows_as, null);
}

pub fn matmulGrouped(a: *Tensor, b: *Tensor, transposed_b: bool) Error!*Tensor {
    if (a.ndim != 3 or b.ndim != 3) return core.refusing("a grouped product whose operands are not [groups, rows, inner]", error.TensorShape);
    switch (a.dtype) {
        .float16, .bfloat16, .float32 => {},
        else => return core.refusing("a grouped product in a dtype other than f32, f16 and bf16", error.TensorDtype),
    }
    if (a.dtype != b.dtype) return core.refusing("a grouped product of operands of two dtypes", error.TensorDtype);
    if (a.shape[0] != b.shape[0]) return error.TensorShape;
    const groups: usize = @intCast(a.shape[0]);
    const m: usize = @intCast(a.shape[1]);
    const k: usize = @intCast(a.shape[2]);
    const n: usize = @intCast(b.shape[if (transposed_b) 1 else 2]);
    if ((if (transposed_b) b.shape[2] else b.shape[1]) != a.shape[2]) return error.TensorShape;
    if (m == 0 or n == 0 or k == 0) return core.refusing("a grouped product of an empty operand", error.TensorUnsupported);
    if (groups < 2) return core.refusing("a grouped product of fewer than two groups", error.TensorUnsupported);
    if (!a.rowContiguous() or !b.rowContiguous()) return core.refusing("a grouped product of an operand not row-contiguous", error.TensorUnsupported);
    const out = try core.fresh(a.dtype, &.{ a.shape[0], @intCast(m), @intCast(n) });
    errdefer core.release(out);
    const one: Call = .{
        .g = try gpu(),
        .a = .{ .t = a, .transposed = false, .ld = @intCast(k) },
        .b = if (transposed_b) .{ .t = b, .transposed = true, .ld = @intCast(k) } else .{ .t = b, .transposed = false, .ld = @intCast(n) },
        .out = out,
        .m = m,
        .n = n,
        .k = k,
        .batch_out = 1,
        .batches = .one,
        .arm_m = m,
    };
    if (@min(m, n) == 1) {
        try groupsSolo(a, b, transposed_b, out, m, n, k);
        return out;
    }
    switch (steelPath(one)) {
        .fused => try groupedFused(a, b, transposed_b, out, one, "steel_gemm_fused", fusedTile(one), 0),
        .fused_nax => {
            const ts = fusedNaxTile(one);
            try groupedFused(a, b, transposed_b, out, one, "steel_gemm_fused_nax", ts.t, ts.swizzle);
        },
        .split_k => try groupedSplitK(a, b, transposed_b, out, one),
        .split_k_nax => try groupsSolo(a, b, transposed_b, out, m, n, k),
    }
    return out;
}

fn groupsSolo(a: *Tensor, b: *Tensor, transposed_b: bool, out: *Tensor, m: usize, n: usize, k: usize) Error!void {
    const groups: c_int = a.shape[0];
    var i: c_int = 0;
    while (i < groups) : (i += 1) {
        const as = try ops.slice(a, &.{ i, 0, 0 }, &.{ i + 1, @intCast(m), @intCast(k) }, &.{ 1, 1, 1 });
        defer core.release(as);
        const ar = try ops.reshape(as, &.{ @intCast(m), @intCast(k) });
        defer core.release(ar);
        const br0 = try ops.slice(b, &.{ i, 0, 0 }, &.{ i + 1, if (transposed_b) @intCast(n) else @intCast(k), if (transposed_b) @intCast(k) else @intCast(n) }, &.{ 1, 1, 1 });
        defer core.release(br0);
        const brw = try ops.reshape(br0, if (transposed_b) &.{ @intCast(n), @intCast(k) } else &.{ @intCast(k), @intCast(n) });
        defer core.release(brw);
        const bm = if (transposed_b) try ops.transpose(brw, &.{ 1, 0 }) else try core.share(brw);
        defer core.release(bm);
        const ds = try ops.slice(out, &.{ i, 0, 0 }, &.{ i + 1, @intCast(m), @intCast(n) }, &.{ 1, 1, 1 });
        defer core.release(ds);
        const dr = try ops.reshape(ds, &.{ @intCast(m), @intCast(n) });
        defer core.release(dr);
        try matmulInto(ar, bm, dr);
    }
}

fn groupedFused(a: *Tensor, b: *Tensor, transposed_b: bool, out: *Tensor, one: Call, comptime family: []const u8, t: Tile, swizzle: u5) Error!void {
    const view = if (transposed_b) try ops.transpose(b, &.{ 0, 2, 1 }) else try core.share(b);
    defer core.release(view);
    var c = one;
    c.out = out;
    c.batches = collapseBatches(a, view);
    c.batch_out = @intCast(a.shape[0]);
    c.a = .{ .t = a, .transposed = false, .ld = @intCast(one.k) };
    c.b = if (transposed_b) .{ .t = view, .transposed = true, .ld = @intCast(one.k) } else .{ .t = view, .transposed = false, .ld = @intCast(one.n) };
    return fusedLaunch(c, family, t, swizzle);
}

pub const group_accum_source =
    \\const int mn = MN;
    \\const int at = int(thread_position_in_grid.x);
    \\const int g = at / mn;
    \\const int e = at - g * mn;
    \\const device float* part = parts + (size_t)(g * PARTS) * (size_t)mn + (size_t)e;
    \\float v = 0.0f;
    \\for (int i = 0; i < PARTS; ++i) v += part[(size_t)i * (size_t)mn];
    \\out[at] = v;
;

var group_accum: ?*kernel.Def = null;
const GroupAccumKey = struct { parts: c_int, mn: c_int, total: c_int };
var group_accum_cfgs: std.AutoHashMapUnmanaged(GroupAccumKey, *kernel.Config) = .empty;

fn groupAccumDef() Error!*kernel.Def {
    if (group_accum) |d| return d;
    const made = try kernel.Def.create("sushi_group_splitk_accum", &.{"parts"}, &.{"out"}, group_accum_source, "", false, false);
    group_accum = made;
    return made;
}

fn groupAccum(parts: *Tensor, out: *Tensor, parts_n: usize, m: usize, n: usize) Error!void {
    const key: GroupAccumKey = .{ .parts = @intCast(parts_n), .mn = @intCast(m * n), .total = @intCast(out.size()) };
    const cfg = group_accum_cfgs.get(key) orelse blk: {
        const made = try std.heap.c_allocator.create(kernel.Config);
        made.* = .{};
        try made.addOutput(out.dims(), .float32);
        try made.addTemplate("PARTS", .{ .int = key.parts });
        try made.addTemplate("MN", .{ .int = key.mn });
        made.grid = .{ key.total, 1, 1 };
        made.group = .{ 256, 1, 1 };
        group_accum_cfgs.put(std.heap.c_allocator, key, made) catch return error.OutOfMemory;
        break :blk made;
    };
    var dsts: [1]*Tensor = .{out};
    kernel.launchInPlace(try groupAccumDef(), &.{parts}, cfg, &dsts) catch |e| switch (e) {
        error.ThreadGroupTooLarge => return core.refusing("a grouped product's accumulation over the GPU's threadgroup limit", error.TensorUnsupported),
        else => |o| return o,
    };
}

fn groupedSplitK(a: *Tensor, b: *Tensor, transposed_b: bool, out: *Tensor, one: Call) Error!void {
    const t = splitKTile(one);
    const m = one.m;
    const n = one.n;
    const k = one.k;
    const groups: usize = @intCast(a.shape[0]);
    const tm = (m + 31) / 32;
    const tn = (n + 31) / 32;
    const tk = k / 16;
    const parts_n = @min(@max(2, nextPowerOfTwo(tk / (tm * tn))), split_k_parts_max);
    const iterations = (k / t.bk) / parts_n;
    if (iterations == 0 or iterations * parts_n * t.bk != k) return groupsSolo(a, b, transposed_b, out, m, n, k);
    const a_g = try ops.transpose(a, &.{ 1, 0, 2 });
    defer core.release(a_g);
    const a_i = try ops.contiguousCopy(a_g);
    defer core.release(a_i);
    const b_i = if (transposed_b) blk: {
        const b_g = try ops.transpose(b, &.{ 1, 0, 2 });
        defer core.release(b_g);
        break :blk try ops.contiguousCopy(b_g);
    } else try core.share(b);
    defer core.release(b_i);
    const ld: i64 = @intCast(groups * k);
    var c = one;
    c.out = out;
    c.a = .{ .t = a_i, .transposed = false, .ld = ld };
    c.b = if (transposed_b) .{ .t = b_i, .transposed = true, .ld = ld } else .{ .t = b_i, .transposed = false, .ld = @intCast(n) };
    const total_parts = parts_n * groups;
    const held = if (plan_api.planned()) @max(total_parts, split_k_parts_max) else total_parts;
    const parts = try core.fresh(.float32, &.{ @intCast(held), @intCast(m), @intCast(n) });
    defer core.release(parts);
    const kern = try splitKKernel(c, t, m % t.bm == 0 and n % t.bn == 0, k % t.bk == 0);
    const tiles_n = (n + t.bn - 1) / t.bn;
    const tiles_m = (m + t.bm - 1) / t.bm;
    const params: SplitKParams = .{
        .m = Call.int(m),
        .n = Call.int(n),
        .k = Call.int(k),
        .lda = @intCast(c.a.ld),
        .ldb = @intCast(c.b.ld),
        .ldc = Call.int(n),
        .tiles_n = Call.int(tiles_n),
        .tiles_m = Call.int(tiles_m),
        .split_k_partitions = Call.int(total_parts),
        .split_k_partition_stride = Call.int(m * n),
        .split_k_partition_size = Call.int(iterations * t.bk),
        .swizzle_log = 0,
        .gemm_k_iterations_aligned = Call.int(iterations),
    };
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, c.a.t, 0);
    core.bind(p, c.b.t, 1);
    core.bindOut(p, parts, 2);
    p.constant(params, 3);
    core.dispatchGroups(p, .{ .x = tiles_n, .y = tiles_m, .z = total_parts }, .{ .x = 32, .y = t.wn, .z = t.wm });
    try groupAccum(parts, out, parts_n, m, n);
}

/// `Matmul::eval_gpu`: operands of one rank whose batch axes agree. `rows_as` fixes the row count the arm is chosen for.
fn product(a_pre: *Tensor, b_pre: *Tensor, rows_as: ?usize, dst_in: ?*Tensor) Error!*Tensor {
    var m: usize = @intCast(a_pre.shape[a_pre.ndim - 2]);
    const n: usize = @intCast(b_pre.shape[b_pre.ndim - 1]);
    const k: usize = @intCast(a_pre.shape[a_pre.ndim - 1]);
    var shape: [core.max_dims]c_int = undefined;
    @memcpy(shape[0..a_pre.ndim], a_pre.dims());
    shape[a_pre.ndim - 1] = @intCast(n);

    const a = try matrixOperand(a_pre, m == 1);
    defer core.release(a.t);
    const b = try matrixOperand(b_pre, n == 1);
    defer core.release(b.t);
    var batches = collapseBatches(a.t, b.t);
    var batch_out: usize = 1;
    for (shape[0 .. a_pre.ndim - 2]) |d| batch_out *= @intCast(d);

    // A batch of `a` that is one run of rows against one `b` is one taller product.
    if (batch_out > 1 and !a.transposed and batches.n == 1 and a.t.strides[a.t.ndim - 2] == k and
        batches.a[0] == @as(i64, @intCast(m * k)) and batches.b[0] == 0)
    {
        m *= @intCast(batches.shape[0]);
        batch_out = 1;
        batches = .one;
    }
    if (rows_as == null and m == 1 and n == 1 and batch_out == 1 and a.t.rowContiguous() and b.t.rowContiguous()) return dot(a.t, b.t, k, shape[0..a_pre.ndim], dst_in);

    const out = if (dst_in) |d| blk: {
        if (d.dtype != a_pre.dtype or !d.rowContiguous() or d.size() != batch_out * m * n) return error.TensorShape;
        break :blk try core.share(d);
    } else try core.fresh(a_pre.dtype, shape[0..a_pre.ndim]);
    errdefer core.release(out);
    const g = try gpu();
    const call: Call = .{ .g = g, .a = a, .b = b, .out = out, .m = m, .n = n, .k = k, .batch_out = batch_out, .batches = batches, .arm_m = rows_as orelse m, .disjoint = dst_in != null };
    // A fixed row count asks for the tiled kernels' sum: the vector kernels are other arithmetic.
    if (rows_as == null) {
        if (!a.transposed and b.transposed and try gemvWide(call)) return out;
        if (@min(m, n) == 1) {
            try gemv(call);
            return out;
        }
    }
    try steel(call);
    return out;
}

/// MLX's `dot_product`: partial sums in f32 per block of 512 threads, summed by MLX's all-reduce, then cast.
fn dot(a: *Tensor, b: *Tensor, k: usize, shape: []const c_int, dst_in: ?*Tensor) Error!*Tensor {
    const group = 512;
    const blocks = ((k + 31) / 32 + group - 1) / group;
    const partials = try core.fresh(.float32, &.{@intCast(blocks)});
    defer core.release(partials);
    var buf: [64]u8 = undefined;
    const p = try core.pass();
    p.pipeline(try stock.kernel(try stock.named(&buf, "dot_product_{s}_it32_tg512_sg16", .{try stock.typeName(a.dtype)})));
    core.bind(p, a, 0);
    core.bind(p, b, 1);
    core.bindOut(p, partials, 2);
    p.constant(@as(c_int, @intCast(k)), 3);
    core.dispatch(p, .{ .x = blocks * group }, .{ .x = group });

    if (dst_in) |d| {
        if (a.dtype != .float32) return core.refusing("matmulInto of a 1x1 non-f32 dot product", error.TensorUnsupported);
        try @import("reduce.zig").all(.sum, partials, d);
        return core.share(d);
    }
    const sum = try core.fresh(.float32, shape);
    errdefer core.release(sum);
    try @import("reduce.zig").all(.sum, partials, sum);
    if (a.dtype == .float32) return sum;
    const cast = try ops.astype(sum, a.dtype);
    core.release(sum);
    return cast;
}

const Call = struct {
    g: Gpu,
    a: Operand,
    b: Operand,
    out: *Tensor,
    m: usize,
    n: usize,
    k: usize,
    batch_out: usize,
    batches: Batches,
    /// The row count the kernel, its tile and its k split are chosen for: `m` unless the caller fixed it.
    arm_m: usize,
    disjoint: bool = false,

    fn int(v: usize) c_int {
        return @intCast(v);
    }
};

fn byteOffset(t: *const Tensor) usize {
    return t.store.at + t.offset;
}

/// MLX's `gemv_wide`: up to fifteen rows against a transposed half-precision weight, the weight streamed once per
/// five rows. False when the plan does not apply.
fn gemvWide(c: Call) Error!bool {
    const mat = c.b;
    const vec = c.a;
    // Generations before M3 keep the other kernels.
    if (c.g.gen < 15) return false;
    var aligned = byteOffset(mat.t) % 8 == 0 and byteOffset(vec.t) % 8 == 0 and @rem(mat.ld, 4) == 0 and @rem(vec.ld, 4) == 0;
    for (c.batches.b[0..c.batches.n], c.batches.a[0..c.batches.n]) |sm, sv| aligned = aligned and @rem(sm, 4) == 0 and @rem(sv, 4) == 0;
    if (c.m <= 1 or c.n <= 1 or c.k % 4 != 0 or (c.out.dtype != .float16 and c.out.dtype != .bfloat16) or !aligned) return false;
    const passes = (c.m + 4) / 5;
    if (passes > 3) return false;
    const full_simd = passes == 1 or c.n <= 64;
    const grid_x: usize = if (c.n >= 65536) 1 else passes;
    const vecs = (c.m + passes - 1) / passes;
    const lanes: usize = if (full_simd) 32 else 16;
    const has_batch = c.batches.n != 1;

    var name_buf: [96]u8 = undefined;
    const name = try stock.named(&name_buf, "gemv_wide_{s}_nv{d}_kl{d}", .{ try stock.typeName(c.out.dtype), vecs, lanes });
    var hash_buf: [128]u8 = undefined;
    const hash = try stock.named(&hash_buf, "{s}_nc{d}_axpby0", .{ name, @intFromBool(has_batch) });
    const kern = try stock.kernelFlags(name, hash, &.{ .{ .index = 0, .value = has_batch }, .{ .index = 1, .value = false } });
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, mat.t, 0);
    core.bind(p, vec.t, 1);
    if (c.disjoint) core.bindOutDisjoint(p, c.out, 3) else core.bindOut(p, c.out, 3);
    p.constant(Call.int(c.k), 4);
    p.constant(Call.int(c.n), 5);
    p.constant(Call.int(c.m), 6);
    p.constant(@as(c_int, @intCast(mat.ld)), 7);
    p.constant(@as(c_int, @intCast(vec.ld)), 8);
    p.constant(Call.int(c.batches.n), 11);
    p.bytes(std.mem.sliceAsBytes(c.batches.shape[0..c.batches.n]), 12);
    p.bytes(std.mem.sliceAsBytes(c.batches.a[0..c.batches.n]), 13);
    p.bytes(std.mem.sliceAsBytes(c.batches.b[0..c.batches.n]), 14);
    core.dispatchGroups(p, .{ .x = grid_x, .y = (c.n + 3) / 4, .z = c.batch_out }, .{ .x = 32, .y = lanes / 8, .z = 1 });
    return true;
}

/// MLX's `gemv`: one side is a single row or column per batch.
const GemvTile = struct { bm: usize, bn: usize, sm: usize, sn: usize, tm: usize, tn: usize, per_group: usize };

fn gemvTile(c: Call) GemvTile {
    const b_is_matrix = c.n != 1;
    const transpose_mat = if (b_is_matrix) !c.b.transposed else c.a.transposed;
    const out_len = if (b_is_matrix) c.n else c.m;
    var tm: usize = 4;
    var tn: usize = 4;
    var sm: usize = 1;
    var sn: usize = 32;
    var bm: usize = 1;
    var bn: usize = 1;
    var per_group: usize = undefined;
    if (transpose_mat) {
        if (c.k >= 8192 and out_len >= 2048) {
            sm = 4;
            sn = 8;
        } else {
            sm = 8;
            sn = 4;
        }
        bn = if (out_len >= 2048) 16 else if (out_len >= 512) 4 else 2;
        if (out_len < tn) tn = 1;
        per_group = bn * sn * tn;
    } else {
        bm = if (out_len >= 4096) 8 else 4;
        if (c.k <= 64) {
            bm = 1;
            sm = 8;
            sn = 4;
        } else if (c.k >= 16 * out_len) {
            bm = 1;
            bn = 8;
        }
        if (out_len < tm) tm = 1;
        per_group = bm * sm * tm;
    }
    return .{ .bm = bm, .bn = bn, .sm = sm, .sn = sn, .tm = tm, .tn = tn, .per_group = per_group };
}

fn gemvKernel(c: Call, t: GemvTile) Error!metal.Pipeline {
    const b_is_matrix = c.n != 1;
    const transpose_mat = if (b_is_matrix) !c.b.transposed else c.a.transposed;
    var buf: [96]u8 = undefined;
    return stock.kernel(try stock.named(&buf, "gemv_{s}{s}_bm{d}_bn{d}_sm{d}_sn{d}_tm{d}_tn{d}_nc{d}_axpby0", .{ if (transpose_mat) "t_" else "", try stock.typeName(c.out.dtype), t.bm, t.bn, t.sm, t.sn, t.tm, t.tn, @intFromBool(c.batches.n != 1) }));
}

fn gemv(c: Call) Error!void {
    const b_is_matrix = c.n != 1;
    const mat = if (b_is_matrix) c.b else c.a;
    const vec = if (b_is_matrix) c.a else c.b;
    const out_len = if (b_is_matrix) c.n else c.m;
    const strides_mat = if (b_is_matrix) &c.batches.b else &c.batches.a;
    const strides_vec = if (b_is_matrix) &c.batches.a else &c.batches.b;

    const t = gemvTile(c);
    const per_group = t.per_group;
    // Until the load seals the pipelines, a product whose output runs along `a`'s rows makes the kernels every row
    // count to the widest chunk picks: the length of the output picks the tile.
    if (!core.sealed and !b_is_matrix and c.arm_m == c.m) {
        var m: usize = 2;
        while (m <= rows_ceiling) : (m = @max(m + 1, m + m / 8)) {
            var at = c;
            at.m = m;
            at.arm_m = m;
            _ = gemvKernel(at, gemvTile(at)) catch {};
        }
    }
    const kern = try gemvKernel(c, t);
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, mat.t, 0);
    core.bind(p, vec.t, 1);
    if (c.disjoint) core.bindOutDisjoint(p, c.out, 3) else core.bindOut(p, c.out, 3);
    p.constant(Call.int(c.k), 4);
    p.constant(Call.int(out_len), 5);
    p.constant(@as(c_int, @intCast(mat.ld)), 6);
    p.constant(Call.int(c.batches.n), 9);
    p.bytes(std.mem.sliceAsBytes(c.batches.shape[0..c.batches.n]), 10);
    p.bytes(std.mem.sliceAsBytes(strides_vec[0..c.batches.n]), 11);
    p.bytes(std.mem.sliceAsBytes(strides_mat[0..c.batches.n]), 12);
    core.dispatchGroups(p, .{ .x = (out_len + per_group - 1) / per_group, .y = 1, .z = c.batch_out }, .{ .x = 32, .y = t.bn, .z = t.bm });
}

const SteelPath = enum { split_k, split_k_nax, fused, fused_nax };

/// PURE: which of MLX's tiled kernels `steel` takes for `c`.
fn steelPath(c: Call) SteelPath {
    const use_nax = c.g.nax and (c.g.tf32 or c.out.dtype != .float32);
    const threshold: usize = if (c.g.class == .s or c.g.class == .d) 2048 else 1024;
    const big = @max(c.arm_m, c.n);
    if (!use_nax and c.batch_out == 1 and ((c.arm_m + 15) / 16) * ((c.n + 15) / 16) <= threshold and c.k / 16 >= 8 and c.k >= big) return .split_k;
    if (use_nax and c.batch_out == 1 and (c.k >= 3 * big or (big <= 1024 and c.k > 2 * big))) return .split_k_nax;
    return if (use_nax) .fused_nax else .fused;
}

/// PURE: for a product of `k` x `n`, the row counts at which `steelPath` may turn, on any GPU class: a path holds every
/// row count up to one of them. A plan probes each, so each path's variant holds its largest instance.
pub fn steelRowEdges(k: usize, n: usize) [6]usize {
    const tiles_n = (n + 15) / 16;
    return .{ 16 * (1024 / tiles_n), 16 * (2048 / tiles_n), k, k / 3, (k -| 1) / 2, 1024 };
}

/// MLX's `steel_matmul`: two matrices, by the tiled kernels.
fn steel(c: Call) Error!void {
    // A dry recording makes no pipeline: the dtype stays open for the first real product (the warm's).
    if (!core.sealed and !core.dryOn()) completeFamily(c);
    const path = steelPath(c);
    // The path follows the product's sizes: under a plan each is a variant of its own.
    const f = try plan_api.frame(@src(), .{});
    defer f.end();
    const v = if (plan_api.planned()) try plan_api.each(@src(), path) else plan_api.Frame.none;
    defer v.end();
    return switch (path) {
        .split_k => splitK(c),
        .split_k_nax => splitKNax(c),
        .fused => fused(c),
        .fused_nax => fusedNax(c),
    };
}

/// The widest product a request's rows make: a prefill chunk at its widest.
const rows_ceiling: usize = 16384;

/// The (dtype, NAX-eligibility) pairs whose every tiled kernel is made (`completeFamily`).
/// `float32` takes NAX tiles when `tf32` is on and SIMD tiles when `exact_f32` turns `tf32` off;
/// keying by `use_nax` ensures a cold load that hits both saves both families into the manifest/archive.
const FamilyKey = struct { dtype: core.Dtype, use_nax: bool };
var families_done: std.AutoHashMapUnmanaged(FamilyKey, void) = .empty;

/// A test that asserts a family is made from its first product starts from none made.
pub fn forgetFamiliesForTest() void {
    families_done.clearRetainingCapacity();
}

/// Sizes on both sides of every turn the path and tile choosers take, on any GPU class: products at every triple of
/// them take every tile a product of one layout and dtype can.
const family_sizes = [_]usize{ 16, 39, 40, 512, 1024, 2048, 2049, 4096, 4097, 8192, 8193, 16384, 1 << 20 };

/// Until the load seals the pipelines, the first tiled product of a dtype makes every kernel any layout, batching, M, N
/// and K of it take: the sizes pick the path, its tile and its K step, a request's sizes (a context's keys, a narrower
/// chunk) are not the warm's, and a request may take a layout no warm forward does (a restored prefix's keys).
fn completeFamily(c: Call) void {
    const use_nax = c.g.nax and (c.g.tf32 or c.out.dtype != .float32);
    const seen = families_done.getOrPut(std.heap.c_allocator, .{ .dtype = c.out.dtype, .use_nax = use_nax }) catch return;
    if (seen.found_existing) return;
    // The kernels not made yet are noted, made on worker threads as a manifest's are, then taken from the stock.
    var notes: stock.Notes = .{};
    defer notes.deinit();
    stock.noting = &notes;
    prepareFamily(c);
    stock.noting = null;
    if (notes.recipes.items.len > 0) {
        const cache = @import("pipeline_cache.zig");
        const threads = @min(std.Thread.getCpuCount() catch 1, 8);
        if (cache.start(notes.recipes.items, threads)) |job| {
            const report = cache.finish(job);
            if (report.failed > 0) log.warn("[engine] {d} of {d} matrix kernels did not make on the workers ({s}); the first product makes them\n", .{ report.failed, notes.recipes.items.len, report.firstError() });
        } else |e| log.warn("[engine] the matrix kernels' workers did not start ({s}); the first product makes them\n", .{@errorName(e)});
    }
    prepareFamily(c);
}

/// `prepare` of every layout, batching and size triple of `c`'s dtype.
fn prepareFamily(c: Call) void {
    var batched = Batches.one;
    batched.push(2, 0, 0);
    for ([_]bool{ false, true }) |a_t| for ([_]bool{ false, true }) |b_t| for ([_]bool{ false, true }) |many| {
        var layout = c;
        layout.a.transposed = a_t;
        layout.b.transposed = b_t;
        layout.batches = if (many) batched else Batches.one;
        layout.batch_out = if (many) 2 else 1;
        for (family_sizes) |m| for (family_sizes) |n| for (family_sizes) |k| {
            var at = layout;
            at.m = m;
            at.arm_m = m;
            at.n = n;
            at.k = k;
            prepare(at);
        };
    };
}

/// Makes, without launching, the kernels `steel` launches for `c` in every alignment a request's lengths pick.
fn prepare(c: Call) void {
    switch (steelPath(c)) {
        .split_k => {
            _ = accumulateKernel(c) catch {};
            const t = splitKTile(c);
            for (0..16) |bits| {
                const other: Tile = .{ .bm = if (bits & 4 != 0) 32 else 16, .bn = if (bits & 8 != 0) 32 else 16, .bk = t.bk, .wm = t.wm, .wn = t.wn };
                _ = splitKKernel(c, other, bits & 1 != 0, bits & 2 != 0) catch continue;
            }
        },
        .split_k_nax => {
            _ = accumulateKernel(c) catch {};
            const t = splitKNaxTile(c).t;
            for (0..8) |bits| _ = splitKNaxKernel(c, t, bits & 1 != 0, bits & 2 != 0, bits & 4 != 0) catch continue;
        },
        .fused => {
            const t = fusedTile(c);
            for (0..8) |bits| _ = fusedKernel("steel_gemm_fused", c, t, bits & 1 != 0, bits & 2 != 0, bits & 4 != 0) catch continue;
        },
        .fused_nax => {
            const t = fusedNaxTile(c).t;
            for (0..8) |bits| _ = fusedKernel("steel_gemm_fused_nax", c, t, bits & 1 != 0, bits & 2 != 0, bits & 4 != 0) catch continue;
        },
    }
}

pub const Tile = struct { bm: usize, bn: usize, bk: usize, wm: usize, wn: usize };

/// `GEMMParams` of mlx/backend/metal/kernels/steel/gemm/params.h.
pub const GemmParams = extern struct {
    m: c_int,
    n: c_int,
    k: c_int,
    lda: c_int,
    ldb: c_int,
    ldd: c_int,
    tiles_n: c_int,
    tiles_m: c_int,
    batch_stride_a: i64,
    batch_stride_b: i64,
    batch_stride_d: i64,
    swizzle_log: c_int,
    gemm_k_iterations_aligned: c_int,
    batch_ndim: c_int,
};

/// `GEMMSpiltKParams` of the same header.
const SplitKParams = extern struct {
    m: c_int,
    n: c_int,
    k: c_int,
    lda: c_int,
    ldb: c_int,
    ldc: c_int,
    tiles_n: c_int,
    tiles_m: c_int,
    split_k_partitions: c_int,
    split_k_partition_stride: c_int,
    split_k_partition_size: c_int,
    swizzle_log: c_int,
    gemm_k_iterations_aligned: c_int,
};

/// `steel_matmul_regular_axpby`'s tile, MLX's `GEMM_TPARAM_MACRO`.
fn fused(c: Call) Error!void {
    return fusedLaunch(c, "steel_gemm_fused", fusedTile(c), 0);
}

fn fusedTile(c: Call) Tile {
    return gemmTile(c.g, c.out.dtype, c.a.transposed, c.b.transposed, c.batch_out, c.arm_m, c.n, c.k);
}

/// PURE: MLX's `GEMM_TPARAM_MACRO`, for every steel product that takes it.
pub fn gemmTile(g: Gpu, dtype: core.Dtype, transpose_a: bool, transpose_b: bool, batch_out: usize, m: usize, n: usize, k: usize) Tile {
    var t: Tile = .{ .bm = 64, .bn = 64, .bk = 16, .wm = 2, .wn = 2 };
    const nt = !transpose_a and transpose_b;
    const half = dtype != .float32;
    const nt_tile: Tile = .{ .bm = 64, .bn = 32, .bk = 32, .wm = 2, .wn = 2 };
    const half_tile: Tile = .{ .bm = 64, .bn = 64, .bk = 16, .wm = 1, .wn = 2 };
    if (g.class == .g or g.class == .p) {
        if (nt) t = nt_tile else if (half) t = half_tile;
    } else if (g.class == .d) {
        if (batch_out * m * n >= 1 << 20) {
            if (half) {
                t = if (2 * @max(m, n) > k) half_tile else if (nt) nt_tile else .{ .bm = 32, .bn = 64, .bk = 16, .wm = 1, .wn = 2 };
            }
        } else if (half) {
            t = if (nt) nt_tile else half_tile;
        } else {
            t = if (nt) .{ .bm = 32, .bn = 64, .bk = 16, .wm = 1, .wn = 2 } else nt_tile;
        }
    }
    return t;
}

/// `steel_matmul_regular_axpby_nax`'s tile and swizzle.
fn fusedNax(c: Call) Error!void {
    const ts = fusedNaxTile(c);
    return fusedLaunch(c, "steel_gemm_fused_nax", ts.t, ts.swizzle);
}

/// A fused NAX launch's tile and swizzle.
pub const FusedNaxTile = struct { t: Tile, swizzle: u5 };

fn fusedNaxTile(c: Call) FusedNaxTile {
    return fusedNaxTileFor(c.g.class, c.m, c.arm_m, c.n, c.k);
}

/// The fused NAX tile of an m x n x k product on GPU class `class`. On class d a wide K (12,288 or more, both other
/// sides past 1,024) takes the 128 x 64 tile, swizzled, which reuses a large weight from cache; a tile never moves a
/// sum's K order, so the bytes are the other tile's.
pub fn fusedNaxTileFor(class: device.Class, m: usize, arm_m: usize, n: usize, k: usize) FusedNaxTile {
    var t: Tile = .{ .bm = 128, .bn = 128, .bk = 512, .wm = 4, .wn = 4 };
    const wide_k_d = class == .d and k >= 12288 and m >= 1024 and n >= 1024;
    if (class == .s or class == .c) {
        t.bk = if (k >= 8192 and k > arm_m + n) 64 else 256;
        t.bm = 64;
        t.wm = 2;
    } else if (class == .d) {
        t = if (n > arm_m or wide_k_d) .{ .bm = 128, .bn = 64, .bk = 512, .wm = 4, .wn = 2 } else .{ .bm = 64, .bn = 128, .bk = 512, .wm = 2, .wn = 4 };
    }
    const tiles_m = (m + t.bm - 1) / t.bm;
    var swizzle: u5 = if (tiles_m <= 3) 0 else 1;
    if (class == .s or class == .c) swizzle = 2 else if (class == .d) swizzle = if (m > n and !wide_k_d) 0 else 2;
    return .{ .t = t, .swizzle = swizzle };
}

/// The fused kernel of `family` for a product whose M, N and K are whole tiles or not: MLX specializes it by those three
/// facts, so a prompt's length alone picks which of the eight a request launches.
fn fusedKernel(comptime family: []const u8, c: Call, t: Tile, align_m: bool, align_n: bool, align_k: bool) Error!metal.Pipeline {
    const type_name = try stock.typeName(c.out.dtype);
    var name_buf: [128]u8 = undefined;
    const name = try stock.named(&name_buf, family ++ "_{c}{c}_{s}_{s}_bm{d}_bn{d}_bk{d}_wm{d}_wn{d}", .{ letter(c.a.transposed), letter(c.b.transposed), type_name, type_name, t.bm, t.bn, t.bk, t.wm, t.wn });
    const has_batch = c.batches.n > 1;
    var hash_buf: [256]u8 = undefined;
    const hash = try stock.named(&hash_buf, "{s}_has_batch_{c}_use_out_source_n_do_axpby_n_align_M_{c}_align_N_{c}_align_K_{c}", .{ name, letter(has_batch), letter(align_m), letter(align_n), letter(align_k) });
    return stock.kernelFlags(name, hash, &.{
        .{ .index = 10, .value = has_batch },
        .{ .index = 100, .value = false },
        .{ .index = 110, .value = false },
        .{ .index = 200, .value = align_m },
        .{ .index = 201, .value = align_n },
        .{ .index = 202, .value = align_k },
    });
}

fn fusedLaunch(c: Call, comptime family: []const u8, t: Tile, swizzle: u5) Error!void {
    const has_batch = c.batches.n > 1;
    const kern = try fusedKernel(family, c, t, c.m % t.bm == 0, c.n % t.bn == 0, c.k % t.bk == 0);
    var tiles_n = (c.n + t.bn - 1) / t.bn;
    var tiles_m = (c.m + t.bm - 1) / t.bm;
    const params: GemmParams = .{
        .m = Call.int(c.m),
        .n = Call.int(c.n),
        .k = Call.int(c.k),
        .lda = @intCast(c.a.ld),
        .ldb = @intCast(c.b.ld),
        .ldd = Call.int(c.n),
        .tiles_n = Call.int(tiles_n),
        .tiles_m = Call.int(tiles_m),
        .batch_stride_a = c.batches.a[c.batches.n - 1],
        .batch_stride_b = c.batches.b[c.batches.n - 1],
        .batch_stride_d = @intCast(c.m * c.n),
        .swizzle_log = swizzle,
        .gemm_k_iterations_aligned = Call.int(c.k / t.bk),
        .batch_ndim = Call.int(c.batches.n),
    };
    const tile = @as(usize, 1) << swizzle;
    tiles_m = (tiles_m + tile - 1) / tile;
    tiles_n *= tile;

    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, c.a.t, 0);
    core.bind(p, c.b.t, 1);
    if (c.disjoint) core.bindOutDisjoint(p, c.out, 3) else core.bindOut(p, c.out, 3);
    p.constant(params, 4);
    if (has_batch) {
        var strides: [2 * max_batch]i64 = undefined;
        @memcpy(strides[0..c.batches.n], c.batches.a[0..c.batches.n]);
        @memcpy(strides[c.batches.n..][0..c.batches.n], c.batches.b[0..c.batches.n]);
        p.bytes(std.mem.sliceAsBytes(c.batches.shape[0..c.batches.n]), 6);
        p.bytes(std.mem.sliceAsBytes(strides[0 .. 2 * c.batches.n]), 7);
    }
    core.dispatchGroups(p, .{ .x = tiles_n, .y = tiles_m, .z = c.batch_out }, .{ .x = 32, .y = t.wn, .z = t.wm });
}

/// MLX's `get_block_dims`: a threadgroup of up to 1024 threads, grown one axis at a time.
fn blockDims(dims: [3]usize) metal.Size {
    var pows: [3]u6 = .{ 0, 0, 0 };
    var sum: usize = 0;
    outer: while (true) {
        const before = sum;
        for (dims, &pows, 0..) |d, *p, axis| {
            if (d >= @as(usize, 1) << (p.* + 1)) {
                p.* += 1;
                sum += 1;
            }
            if (sum == 10 and axis < 2) break :outer;
        }
        if (sum == before or sum == 10) break;
    }
    return .{ .x = @as(usize, 1) << pows[0], .y = @as(usize, 1) << pows[1], .z = @as(usize, 1) << pows[2] };
}

/// The f32 parts of a split product summed into `out`.
fn accumulate(c: Call, parts: *Tensor, count: usize) Error!void {
    const kern = try accumulateKernel(c);
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, parts, 0);
    if (c.disjoint) core.bindOutDisjoint(p, c.out, 1) else core.bindOut(p, c.out, 1);
    p.constant(Call.int(count), 2);
    p.constant(Call.int(c.m * c.n), 3);
    p.constant(Call.int(c.n), 4);
    core.dispatch(p, .{ .x = c.n, .y = c.m, .z = 1 }, blockDims(.{ c.n, c.m, 1 }));
}

fn accumulateKernel(c: Call) Error!metal.Pipeline {
    var buf: [96]u8 = undefined;
    return stock.kernel(try stock.named(&buf, "steel_gemm_splitk_accum_{s}_float32", .{try stock.typeName(c.out.dtype)}));
}

fn nextPowerOfTwo(v: usize) usize {
    return if (v <= 1) v else std.math.ceilPowerOfTwo(usize, v) catch v;
}

fn splitKKernel(c: Call, t: Tile, aligned_mn: bool, aligned_k: bool) Error!metal.Pipeline {
    var buf: [160]u8 = undefined;
    return stock.kernel(try stock.named(&buf, "steel_gemm_splitk_{c}{c}_{s}_float32_bm{d}_bn{d}_bk{d}_wm{d}_wn{d}_MN_{c}aligned_K_{c}aligned", .{ letter(c.a.transposed), letter(c.b.transposed), try stock.typeName(c.out.dtype), t.bm, t.bn, t.bk, t.wm, t.wn, letter(aligned_mn), letter(aligned_k) }));
}

fn splitKTile(c: Call) Tile {
    return .{ .bm = if (c.arm_m < 40) 16 else 32, .bn = if (c.n < 40) 16 else 32, .bk = 16, .wm = 2, .wn = 2 };
}

const split_k_parts_max = 32;

/// `steel_gemm_splitk_axpby`: a long inner axis cut into parts, each a product into f32, then summed.
fn splitK(c: Call) Error!void {
    const tm = (c.arm_m + 31) / 32;
    const tn = (c.n + 31) / 32;
    const tk = c.k / 16;
    const t = splitKTile(c);
    const parts_n = @min(@max(2, nextPowerOfTwo(tk / (tm * tn))), split_k_parts_max);
    const iterations = (c.k / t.bk) / parts_n;
    // The count steps with the tiles, so its bytes do not grow with the product's sizes: under a plan the partials take
    // the most there can be, and the largest product a probe records holds every smaller one.
    const held = if (plan_api.planned()) split_k_parts_max else parts_n;
    const parts = try core.fresh(.float32, &.{ Call.int(held), Call.int(c.m), Call.int(c.n) });
    defer core.release(parts);
    const kern = try splitKKernel(c, t, c.m % t.bm == 0 and c.n % t.bn == 0, c.k % t.bk == 0);
    const tiles_n = (c.n + t.bn - 1) / t.bn;
    const tiles_m = (c.m + t.bm - 1) / t.bm;
    const params: SplitKParams = .{
        .m = Call.int(c.m),
        .n = Call.int(c.n),
        .k = Call.int(c.k),
        .lda = @intCast(c.a.ld),
        .ldb = @intCast(c.b.ld),
        .ldc = Call.int(c.n),
        .tiles_n = Call.int(tiles_n),
        .tiles_m = Call.int(tiles_m),
        .split_k_partitions = Call.int(parts_n),
        .split_k_partition_stride = Call.int(c.m * c.n),
        .split_k_partition_size = Call.int(iterations * t.bk),
        .swizzle_log = 0,
        .gemm_k_iterations_aligned = Call.int(iterations),
    };
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, c.a.t, 0);
    core.bind(p, c.b.t, 1);
    core.bindOut(p, parts, 2);
    p.constant(params, 3);
    core.dispatchGroups(p, .{ .x = tiles_n, .y = tiles_m, .z = parts_n }, .{ .x = 32, .y = t.wn, .z = t.wm });
    try accumulate(c, parts, parts_n);
}

const SplitKNaxTile = struct { t: Tile, part_size: usize };

fn splitKNaxTile(c: Call) SplitKNaxTile {
    return splitKNaxTileFor(c.g.class, c.arm_m, c.n, c.k);
}

/// PURE: the NAX split-K tile and inner-axis part a product takes on a GPU class (the Ultra cuts smaller parts).
pub fn splitKNaxTileFor(class: device.Class, arm_m: usize, n: usize, k: usize) SplitKNaxTile {
    var t: Tile = .{ .bm = 128, .bn = 128, .bk = 512, .wm = 4, .wn = 4 };
    var part_size: usize = 4096;
    if (class == .d) {
        if (arm_m <= 2048 and n <= 2048) {
            t = .{ .bm = 64, .bn = 64, .bk = 256, .wm = 2, .wn = 2 };
            part_size = 2048;
        } else if (n > arm_m) {
            t = .{ .bm = 128, .bn = 64, .bk = 512, .wm = 4, .wn = 2 };
        } else {
            t = .{ .bm = 64, .bn = 128, .bk = 512, .wm = 2, .wn = 4 };
        }
    }
    if ((arm_m + n) / 2 < 512 or k <= 4096) t = .{ .bm = 64, .bn = 64, .bk = 256, .wm = 2, .wn = 2 };
    if (k <= 1024) part_size = k / 2 else if (k <= 2048) part_size = 1024 else if (k <= 4096) part_size = 2048;
    return .{ .t = t, .part_size = part_size };
}

fn splitKNaxKernel(c: Call, t: Tile, align_m: bool, align_n: bool, align_k: bool) Error!metal.Pipeline {
    var name_buf: [128]u8 = undefined;
    const name = try stock.named(&name_buf, "steel_gemm_splitk_nax_{c}{c}_{s}_float32_bm{d}_bn{d}_bk{d}_wm{d}_wn{d}", .{ letter(c.a.transposed), letter(c.b.transposed), try stock.typeName(c.out.dtype), t.bm, t.bn, t.bk, t.wm, t.wn });
    var hash_buf: [192]u8 = undefined;
    const hash = try stock.named(&hash_buf, "{s}_align_M_{c}_align_N_{c}_align_K_{c}", .{ name, letter(align_m), letter(align_n), letter(align_k) });
    return stock.kernelFlags(name, hash, &.{ .{ .index = 200, .value = align_m }, .{ .index = 201, .value = align_n } });
}

pub const split_k_nax_parts_least = 3;

/// `steel_gemm_splitk_axpby_nax`.
fn splitKNax(c: Call) Error!void {
    const tp = splitKNaxTile(c);
    const t = tp.t;
    const part_size = tp.part_size;
    const ultra = c.g.class == .d;
    const parts_n = (c.k + part_size - 1) / part_size;
    // An odd inner axis to 1,024 halves into 3 parts, a longer one to 8,192 into 2: under a plan the partials take at
    // least 3, so their bytes never fall as the inner axis grows.
    const held = if (plan_api.planned()) @max(parts_n, split_k_nax_parts_least) else parts_n;
    const parts = try core.fresh(.float32, &.{ Call.int(held), Call.int(c.m), Call.int(c.n) });
    defer core.release(parts);
    const kern = try splitKNaxKernel(c, t, c.m % t.bm == 0, c.n % t.bn == 0, c.k % t.bk == 0);
    const tiles_n = (c.n + t.bn - 1) / t.bn;
    const tiles_m = (c.m + t.bm - 1) / t.bm;
    var swizzle: u5 = if (tiles_m <= 3) 0 else 1;
    if (ultra) swizzle = if (c.m > c.n) 0 else 1;
    const tile = @as(usize, 1) << swizzle;
    const params: SplitKParams = .{
        .m = Call.int(c.m),
        .n = Call.int(c.n),
        .k = Call.int(c.k),
        .lda = @intCast(c.a.ld),
        .ldb = @intCast(c.b.ld),
        .ldc = Call.int(c.n),
        .tiles_n = Call.int(tiles_n),
        .tiles_m = Call.int(tiles_m),
        .split_k_partitions = Call.int(parts_n),
        .split_k_partition_stride = Call.int(c.m * c.n),
        .split_k_partition_size = Call.int(part_size),
        .swizzle_log = swizzle,
        .gemm_k_iterations_aligned = Call.int(part_size / t.bk),
    };
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, c.a.t, 0);
    core.bind(p, c.b.t, 1);
    core.bindOut(p, parts, 2);
    p.constant(params, 3);
    core.dispatchGroups(p, .{ .x = (tiles_n * tile) * ((tiles_m + tile - 1) / tile) * parts_n, .y = 1, .z = 1 }, .{ .x = 32, .y = t.wn, .z = t.wm });
    try accumulate(c, parts, parts_n);
}
