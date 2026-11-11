//! MLX's stock kernels launched by the engine: the compiled functions of `mlx.metallib`, fetched by the names MLX
//! gives them and dispatched with the buffers, constants and grids MLX's `eval_gpu` uses, so an op here writes the
//! bytes it writes under MLX. Ported from MLX's Metal backend (MIT, see NOTICE).

const std = @import("std");
const mlx = @import("../mlx.zig");
const metal = @import("../metal.zig");
const core = @import("core.zig");
const ops = @import("ops.zig");
const registry = @import("pipeline_registry.zig");
const manifest = @import("../pipeline_manifest.zig");

const Tensor = core.Tensor;
const Dtype = core.Dtype;
const Error = core.Error;
const gpa = std.heap.c_allocator;

const DlInfo = extern struct { fname: ?[*:0]const u8, fbase: ?*anyopaque, sname: ?[*:0]const u8, saddr: ?*anyopaque };
extern "c" fn dladdr(addr: ?*const anyopaque, info: *DlInfo) c_int;

var library: ?metal.Library = null;
var pipelines: std.StringHashMapUnmanaged(metal.Pipeline) = .empty;

/// `mlx.metallib` beside the MLX libraries this process loaded, as MLX finds it.
pub fn metallibPath(buf: []u8) Error![:0]const u8 {
    var info: DlInfo = undefined;
    if (dladdr(mlx.libraryAnchor(), &info) == 0) return error.MetalUnavailable;
    const binary = std.mem.span(info.fname orelse return error.MetalUnavailable);
    const dir = std.fs.path.dirname(binary) orelse return error.MetalUnavailable;
    return std.fmt.bufPrintSentinel(buf, "{s}/mlx.metallib", .{dir}, 0) catch return error.MetalUnavailable;
}

pub fn lib(dev: *const metal.Device) Error!metal.Library {
    if (library) |l| return l;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    library = try dev.load(try metallibPath(&buf));
    return library.?;
}

/// Keeps `pipeline`, made ahead from `recipe`, for the first `kernel` or `kernelFlags` that asks for its key.
pub fn preload(recipe: manifest.Recipe, pipeline: metal.Pipeline) Error!bool {
    if (pipelines.contains(recipe.key)) return false;
    return registry.putStock(recipe, pipeline) catch error.OutOfMemory;
}

/// The pipeline of the stock kernel `name`, built once.
pub fn kernel(name: []const u8) Error!metal.Pipeline {
    if (pipelines.get(name)) |p| return p;
    const dev = try metal.device();
    const owned = gpa.allocSentinel(u8, name.len, 0) catch return error.OutOfMemory;
    errdefer gpa.free(owned);
    @memcpy(owned, name);
    if (registry.takeStock(name)) |p| {
        core.preloadedPipeline(name, p);
        pipelines.put(gpa, owned, p) catch return error.OutOfMemory;
        return p;
    }
    const t0 = core.nowNs();
    const p = try dev.pipelineOf(try lib(dev), owned);
    core.madePipeline(name, t0, p);
    registry.record(.{ .kind = .stock, .function = name, .key = name });
    pipelines.put(gpa, owned, p) catch return error.OutOfMemory;
    return p;
}

/// The pipeline of the stock kernel `name` specialized by `flags`, built once per `hash`, MLX's name for the pair.
pub fn kernelFlags(name: []const u8, hash: []const u8, flags: []const metal.Flag) Error!metal.Pipeline {
    if (pipelines.get(hash)) |p| return p;
    const dev = try metal.device();
    var buf: [192]u8 = undefined;
    const function = std.fmt.bufPrintSentinel(&buf, "{s}", .{name}, 0) catch return error.TensorShape;
    if (registry.takeStock(hash)) |p| {
        core.preloadedPipeline(hash, p);
        const owned = gpa.dupe(u8, hash) catch return error.OutOfMemory;
        pipelines.put(gpa, owned, p) catch return error.OutOfMemory;
        return p;
    }
    const t0 = core.nowNs();
    const p = try dev.pipelineOfFlags(try lib(dev), function, flags);
    core.madePipeline(hash, t0, p);
    registry.record(.{ .kind = .stock_flags, .function = name, .key = hash, .flags = flags });
    const owned = gpa.dupe(u8, hash) catch return error.OutOfMemory;
    pipelines.put(gpa, owned, p) catch return error.OutOfMemory;
    return p;
}

/// The pipeline of the stock kernel `name` specialized by boolean and int function constants, built once per `hash`.
pub fn kernelConstants(name: []const u8, hash: []const u8, flags: []const metal.Flag, ints: []const metal.IntConstant) Error!metal.Pipeline {
    if (pipelines.get(hash)) |p| return p;
    const dev = try metal.device();
    var buf: [192]u8 = undefined;
    const function = std.fmt.bufPrintSentinel(&buf, "{s}", .{name}, 0) catch return error.TensorShape;
    if (registry.takeStock(hash)) |p| {
        core.preloadedPipeline(hash, p);
        const owned = gpa.dupe(u8, hash) catch return error.OutOfMemory;
        pipelines.put(gpa, owned, p) catch return error.OutOfMemory;
        return p;
    }
    const t0 = core.nowNs();
    const p = try dev.pipelineOfConstants(try lib(dev), function, flags, ints);
    core.madePipeline(hash, t0, p);
    registry.record(.{ .kind = .stock_flags, .function = name, .key = hash, .flags = flags, .ints = ints });
    const owned = gpa.dupe(u8, hash) catch return error.OutOfMemory;
    pipelines.put(gpa, owned, p) catch return error.OutOfMemory;
    return p;
}

/// MLX's name of a dtype in a kernel name.
pub fn typeName(d: Dtype) Error![]const u8 {
    return switch (d) {
        .bool_ => "bool_",
        .uint8 => "uint8",
        .uint16 => "uint16",
        .uint32 => "uint32",
        .uint64 => "uint64",
        .int8 => "int8",
        .int16 => "int16",
        .int32 => "int32",
        .int64 => "int64",
        .float16 => "float16",
        .float32 => "float32",
        .bfloat16 => "bfloat16",
        else => error.TensorDtype,
    };
}

pub fn named(buf: []u8, comptime fmt: []const u8, args: anytype) Error![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch error.TensorShape;
}

const rms_reads = 4;
const rms_looped_limit = 4096;

/// `fast::RMSNorm`: `x` scaled by `w / rms(x)` over the last axis.
pub fn rmsNorm(x_in: *Tensor, w: *Tensor, eps: f32) Error!*Tensor {
    if (x_in.ndim == 0 or x_in.dtype != w.dtype) return error.TensorShape;
    const x = try ops.contiguous(x_in);
    defer core.release(x);
    const out = try core.fresh(x.dtype, x.dims());
    errdefer core.release(out);
    const axis: u32 = @intCast(x.shape[x.ndim - 1]);
    if (axis == 0) return out;
    const rows = x.size() / axis;
    var buf: [64]u8 = undefined;
    const looped = axis > rms_looped_limit;
    const k = try kernel(try named(&buf, "rms{s}{s}", .{ if (looped) "_looped" else "", try typeName(x.dtype) }));
    const group = if (looped) k.max_threads else 32 * ((((axis + rms_reads - 1) / rms_reads) + 31) / 32);
    if (group > k.max_threads) return error.TensorShape;
    const p = try core.pass();
    p.pipeline(k);
    core.bind(p, x, 0);
    core.bind(p, w, 1);
    core.bindOut(p, out, 2);
    p.constant(eps, 3);
    p.constant(axis, 4);
    p.constant(@as(u32, if (w.ndim == 1) @intCast(w.strides[0]) else 0), 5);
    core.dispatch(p, .{ .x = rows * group }, .{ .x = group });
    return out;
}

const softmax_reads = 4;
const softmax_looped_limit = 4096;

/// The `Softmax` primitive: over the last axis, in a float dtype. `precise` accumulates a 16-bit input in float.
pub fn softmaxLast(x_in: *Tensor, precise: bool) Error!*Tensor {
    if (x_in.ndim == 0) return error.TensorShape;
    const x = try ops.contiguous(x_in);
    defer core.release(x);
    const out = try core.fresh(x.dtype, x.dims());
    errdefer core.release(out);
    const axis: usize = @intCast(x.shape[x.ndim - 1]);
    if (x.size() == 0) return out;
    const rows = x.size() / axis;
    var buf: [64]u8 = undefined;
    const looped = axis > softmax_looped_limit;
    const k = try kernel(try named(&buf, "{s}softmax_{s}{s}", .{ if (looped) "looped_" else "block_", if (x.dtype != .float32 and precise) "precise_" else "", try typeName(x.dtype) }));
    const group = if (looped) k.max_threads else 32 * ((((axis + softmax_reads - 1) / softmax_reads) + 31) / 32);
    if (group > k.max_threads) return error.TensorShape;
    const p = try core.pass();
    p.pipeline(k);
    core.bind(p, x, 0);
    core.bindOut(p, out, 1);
    p.constant(@as(c_int, @intCast(axis)), 2);
    core.dispatch(p, .{ .x = rows * group }, .{ .x = group });
    return out;
}

pub const QuantMode = enum { affine, mxfp4, mxfp8, nvfp4 };

/// The C type MLX's quantized kernels are instantiated on.
pub fn cType(d: Dtype) Error![]const u8 {
    return switch (d) {
        .float32 => "float",
        .float16 => "float16_t",
        .bfloat16 => "bfloat16_t",
        else => error.TensorDtype,
    };
}

/// `dequantize`: packed codes `w` (uint32) with a scale, and for `affine` a bias, per group of `group` outputs.
/// Affine computes in the scales' dtype and casts to `out_type`; the float modes write `out_type`.
pub fn dequantize(w_in: *Tensor, scales_in: *Tensor, biases_in: ?*Tensor, group: usize, bits: usize, mode: QuantMode, global: ?*Tensor, out_type: Dtype) Error!*Tensor {
    if (w_in.dtype != .uint32 or w_in.ndim < 2 or scales_in.ndim != w_in.ndim or bits == 0 or group == 0) return error.TensorShape;
    var shape: [core.max_dims]c_int = undefined;
    @memcpy(shape[0..w_in.ndim], w_in.dims());
    const last = w_in.ndim - 1;
    for (0..last) |i| if (scales_in.shape[i] != shape[i]) return error.TensorShape;
    shape[last] = @intCast(@as(usize, @intCast(shape[last])) * 32 / bits);
    if (@as(usize, @intCast(shape[last])) != @as(usize, @intCast(scales_in.shape[last])) * group) return error.TensorShape;
    const w = try ops.contiguous(w_in);
    defer core.release(w);
    const scales = try ops.contiguous(scales_in);
    defer core.release(scales);
    const kernel_type = if (mode == .affine) scales.dtype else out_type;
    var buf: [96]u8 = undefined;
    const k = if (mode == .affine)
        try kernel(try named(&buf, "affine_dequantize_{s}_gs_{d}_b_{d}", .{ try cType(kernel_type), group, bits }))
    else
        try kernel(try named(&buf, "{s}_dequantize_{s}_gs_{d}_b_{d}_hgs_{s}", .{ @tagName(mode), try cType(kernel_type), group, bits, if (global != null) "true" else "false" }));
    const out = try core.fresh(kernel_type, shape[0..w_in.ndim]);
    errdefer core.release(out);
    const packs: usize = if (bits == 3 or bits == 5) 8 else if (bits == 6) 4 else 8 / bits;
    const threads = out.size() / packs;
    if (threads == 0) return out;
    if (threads > std.math.maxInt(u32)) return error.TensorShape;
    // A copy is a dispatch of its own: it is made before this one's binds.
    var biases: ?*Tensor = null;
    defer if (biases) |b| core.release(b);
    if (mode == .affine) {
        const given = biases_in orelse return error.TensorShape;
        if (given.dtype != scales.dtype) return error.TensorDtype;
        biases = try ops.contiguous(given);
    }
    const p = try core.pass();
    p.pipeline(k);
    core.bind(p, w, 0);
    core.bind(p, scales, 1);
    if (biases) |b| core.bind(p, b, 2) else if (global) |g| core.bind(p, g, 2);
    core.bindOut(p, out, 3);
    core.dispatch(p, .{ .x = threads }, .{ .x = @min(k.max_threads, threads) });
    if (kernel_type == out_type) return out;
    defer core.release(out);
    return ops.astype(out, out_type);
}

/// `quantize` in affine mode (`quantize_impl`): the packed codes of `w` (uint32) and, per group of `group` values
/// along the last axis, a scale and a bias in `w`'s dtype. The caller releases the three.
pub fn quantize(w_in: *Tensor, group: usize, bits: usize) Error![3]*Tensor {
    if (w_in.ndim == 0) return error.TensorShape;
    if (group != 32 and group != 64 and group != 128) return error.TensorUnsupported;
    if (bits < 2 or bits > 8 or bits == 7) return error.TensorUnsupported;
    const last = w_in.ndim - 1;
    const width: usize = @intCast(w_in.shape[last]);
    if (width == 0 or width % group != 0 or (width * bits) % 32 != 0) return error.TensorShape;
    const w = try ops.contiguous(w_in);
    defer core.release(w);
    var buf: [96]u8 = undefined;
    const k = try kernel(try named(&buf, "affine_quantize_{s}_gs_{d}_b_{d}", .{ try cType(w.dtype), group, bits }));
    var shape: [core.max_dims]c_int = undefined;
    @memcpy(shape[0..w.ndim], w.dims());
    shape[last] = @intCast(width * bits / 32);
    const codes = try core.fresh(.uint32, shape[0..w.ndim]);
    errdefer core.release(codes);
    shape[last] = @intCast(width / group);
    const scales = try core.fresh(w.dtype, shape[0..w.ndim]);
    errdefer core.release(scales);
    const biases = try core.fresh(w.dtype, shape[0..w.ndim]);
    errdefer core.release(biases);
    // A thread reduces `group / 32` values; the 32 threads of a SIMD group share a group's extremes.
    const threads = w.size() / (group / 32);
    if (threads == 0) return .{ codes, scales, biases };
    if (threads > std.math.maxInt(u32)) return error.TensorShape;
    const p = try core.pass();
    p.pipeline(k);
    core.bind(p, w, 0);
    core.bindOut(p, codes, 1);
    core.bindOut(p, scales, 2);
    core.bindOut(p, biases, 3);
    core.dispatch(p, .{ .x = threads }, .{ .x = @min(k.max_threads, threads) });
    return .{ codes, scales, biases };
}

/// `Erf::eval_gpu` (`unary_op_gpu_inplace`) over a row-contiguous float tensor: MLX's `erf`, not `metal::erf`.
pub fn erf(x_in: *Tensor) Error!*Tensor {
    switch (x_in.dtype) {
        .float32, .float16, .bfloat16 => {},
        else => return error.TensorDtype,
    }
    // MLX gives a contiguous input that is not row-major an output of its strides; this port makes row-major ones only.
    if (!x_in.rowContiguous()) return error.TensorUnsupported;
    const x = try core.share(x_in);
    defer core.release(x);
    const out = try core.fresh(x.dtype, x.dims());
    errdefer core.release(out);
    const size = x.size();
    if (size == 0) return out;
    if (size > std.math.maxInt(u32)) return error.TensorUnsupported;
    const type_name = try typeName(x.dtype);
    // `get_work_per_thread`: 8 bytes a thread from 65536 elements.
    const per_thread: usize = if (size < 1 << 16) 1 else @max(1, 8 / core.itemSize(x.dtype));
    var buf: [64]u8 = undefined;
    const k = try kernel(try named(&buf, "{s}_Erf{s}{s}", .{ if (per_thread > 1) "vn" else "v", type_name, type_name }));
    const threads = (size + per_thread - 1) / per_thread;
    const p = try core.pass();
    p.pipeline(k);
    core.bind(p, x, 0);
    core.bindOut(p, out, 1);
    p.constant(@as(c_int, @intCast(size)), 2);
    core.dispatch(p, .{ .x = threads }, .{ .x = @min(k.max_threads, threads) });
    return out;
}

/// A binary op of `mlx.metallib` on floats, under `binary_op_gpu_inplace`: the scalar and vector kernels of one flat
/// run. An operand pair MLX would run through its strided kernels is refused.
pub fn binaryOf(comptime op: []const u8, a: *Tensor, b: *Tensor) Error!*Tensor {
    switch (a.dtype) {
        .float16, .bfloat16, .float32 => {},
        else => return error.TensorDtype,
    }
    const lead = @max(a.ndim, b.ndim);
    var shape: [core.max_dims]c_int = undefined;
    for (0..lead) |i| {
        const da: c_int = if (i + a.ndim >= lead) a.shape[i + a.ndim - lead] else 1;
        const db: c_int = if (i + b.ndim >= lead) b.shape[i + b.ndim - lead] else 1;
        if (da != db and da != 1 and db != 1) return error.TensorShape;
        shape[i] = if (da == 1) db else da;
    }
    const out = try core.fresh(a.dtype, shape[0..lead]);
    errdefer core.release(out);
    const size = out.size();
    if (size == 0) return out;
    if (size > std.math.maxInt(u32)) return error.TensorUnsupported;
    const kind: []const u8 = if (a.size() == 1 and b.size() == 1)
        "ss"
    else if (a.size() == 1 and b.rowContiguous() and b.size() == size)
        "sv"
    else if (b.size() == 1 and a.rowContiguous() and a.size() == size)
        "vs"
    else if (a.rowContiguous() and b.rowContiguous() and a.size() == size and b.size() == size)
        "vv"
    else
        return error.TensorUnsupported;
    const flat = !std.mem.eql(u8, kind, "ss");
    // Eight bytes of a row per thread once it is wide, as `get_work_per_thread`.
    const per_thread: usize = if (flat and size >= 1 << 16) @max(1, 8 / core.itemSize(a.dtype)) else 1;
    var buf: [96]u8 = undefined;
    const k = try kernel(try named(&buf, "{s}{s}_" ++ op ++ "{s}", .{ kind, if (flat and per_thread > 1) "n" else "", try typeName(a.dtype) }));
    const threads = (size + per_thread - 1) / per_thread;
    const p = try core.pass();
    p.pipeline(k);
    core.bind(p, a, 0);
    core.bind(p, b, 1);
    core.bindOut(p, out, 2);
    if (!std.mem.eql(u8, kind, "ss")) p.constant(@as(u32, @intCast(size)), 3);
    core.dispatch(p, .{ .x = threads }, .{ .x = @min(k.max_threads, threads) });
    return out;
}
