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

/// Tests only: `kernel` meets `name` next as a new process does, whatever an earlier test built. A holder of the
/// pipeline keeps it.
pub fn forget(name: []const u8) void {
    _ = pipelines.remove(name);
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

const ln_reads = 8;
const ln_looped_limit = 6656;

/// `LayerNorm::eval_gpu`: x, w and b in one dtype (w and b `[axis]` or 0-d, as `fast::layer_norm` passes them).
pub fn layerNorm(x_in: *Tensor, w: *Tensor, b: *Tensor, eps: f32) Error!*Tensor {
    if (x_in.ndim == 0 or x_in.dtype != w.dtype or x_in.dtype != b.dtype) return error.TensorShape;
    const x = try ops.contiguous(x_in);
    defer core.release(x);
    const out = try core.fresh(x.dtype, x.dims());
    errdefer core.release(out);
    const axis: u32 = @intCast(x.shape[x.ndim - 1]);
    if (axis == 0) return out;
    const rows = x.size() / axis;
    var buf: [64]u8 = undefined;
    const looped = axis > ln_looped_limit;
    const k = try kernel(try named(&buf, "layer_norm{s}{s}", .{ if (looped) "_looped" else "", try typeName(x.dtype) }));
    const group = if (looped) k.max_threads else 32 * ((((axis + ln_reads - 1) / ln_reads) + 31) / 32);
    if (group > k.max_threads) return core.refusing("a layer norm wider than the pipeline's threadgroup", error.TensorUnsupported);
    const p = try core.pass();
    p.pipeline(k);
    core.bind(p, x, 0);
    core.bind(p, w, 1);
    core.bind(p, b, 2);
    core.bindOut(p, out, 3);
    p.constant(eps, 4);
    p.constant(axis, 5);
    p.constant(@as(u32, if (w.ndim == 1) @intCast(w.strides[0]) else 0), 6);
    p.constant(@as(u32, if (b.ndim == 1) @intCast(b.strides[0]) else 0), 7);
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

const logsumexp_looped_limit = 4096;

/// `LogSumExp` over the last axis (`logsumexp.cpp`): the block kernel to 4096 elements a row, the looped one past.
pub fn logsumexpLast(x_in: *Tensor, keepdims: bool) Error!*Tensor {
    switch (x_in.dtype) {
        .float32, .float16, .bfloat16 => {},
        else => return core.refusing("logsumexp of an array that is not float", error.TensorDtype),
    }
    if (x_in.ndim == 0 or x_in.size() == 0) return error.TensorShape;
    const x = try ops.contiguous(x_in);
    defer core.release(x);
    const axis: usize = @intCast(x.shape[x.ndim - 1]);
    var shape: [core.max_dims]c_int = undefined;
    @memcpy(shape[0..x.ndim], x.dims());
    shape[x.ndim - 1] = 1;
    const out = try core.fresh(x.dtype, shape[0 .. if (keepdims) x.ndim else x.ndim - 1]);
    errdefer core.release(out);
    const rows = x.size() / axis;
    var buf: [64]u8 = undefined;
    const looped = axis > logsumexp_looped_limit;
    const k = try kernel(try named(&buf, "{s}logsumexp_{s}", .{ if (looped) "looped_" else "block_", try typeName(x.dtype) }));
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

/// `get_block_dims_common` (backend/common/utils.cpp): powers of two over three axes, 1024 threads at most.
fn blockDims(d0: usize, d1: usize, d2: usize) metal.Size {
    var pows = [3]u6{ 0, 0, 0 };
    const dims = [3]usize{ d0, d1, d2 };
    var sum: usize = 0;
    outer: while (true) {
        const presum = sum;
        for (0..3) |i| {
            if (dims[i] >= (@as(usize, 1) << (pows[i] + 1))) {
                pows[i] += 1;
                sum += 1;
            }
            if (sum == 10) break :outer;
        }
        if (sum == presum) break;
    }
    return .{ .x = @as(usize, 1) << pows[0], .y = @as(usize, 1) << pows[1], .z = @as(usize, 1) << pows[2] };
}

/// `depthwise_conv_1D_gpu`, the fast path of `conv_1D_gpu` for groups = C = O at stride 1, no padding and no dilation:
/// `in [B, T, C]`, `w [C, K, 1]`, both made row-contiguous, into `[B, T - K + 1, C]`.
pub fn depthwiseConv1d(in_pre: *Tensor, w_pre: *Tensor) Error!*Tensor {
    switch (in_pre.dtype) {
        .float32, .float16, .bfloat16 => {},
        else => return core.refusing("a depthwise conv1d over a dtype other than f32, f16 and bf16", error.TensorDtype),
    }
    if (w_pre.dtype != in_pre.dtype) return core.refusing("a conv1d of two dtypes", error.TensorDtype);
    if (in_pre.ndim != 3 or w_pre.ndim != 3) return error.TensorShape;
    const in = try ops.contiguous(in_pre);
    defer core.release(in);
    const w = try ops.contiguous(w_pre);
    defer core.release(w);
    const B: usize = @intCast(in.shape[0]);
    const T: usize = @intCast(in.shape[1]);
    const C: usize = @intCast(in.shape[2]);
    const K: usize = @intCast(w.shape[1]);
    if (K == 0 or K > T) return error.TensorShape;
    const tout = T - K + 1;
    const out = try core.fresh(in.dtype, &.{ @intCast(B), @intCast(tout), @intCast(C) });
    errdefer core.release(out);
    if (out.size() == 0) return out;
    if (in.size() > std.math.maxInt(i32)) return core.refusing("a conv1d input past int32 elements", error.TensorUnsupported);
    var buf: [64]u8 = undefined;
    const k = try kernel(try named(&buf, "depthwise_conv_1d_{s}", .{try typeName(in.dtype)}));
    const strides = [3]i32{ @intCast(in.strides[0]), @intCast(in.strides[1]), @intCast(in.strides[2]) };
    const p = try core.pass();
    p.pipeline(k);
    core.bind(p, in, 0);
    core.bind(p, w, 1);
    core.bindOut(p, out, 2);
    p.bytes(std.mem.sliceAsBytes(&strides), 3);
    p.constant(@as(c_int, @intCast(K)), 4);
    core.dispatch(p, .{ .x = C, .y = tout, .z = B }, blockDims(C, tout, B));
    return out;
}

const rope_per_thread = 4;

/// `fast::RoPE` forward (rope.cpp): `dims` of the last axis rotated, at `offset` (a scalar or one a row of axis 0), by
/// `base` or by `freqs`. A row-contiguous input is read as is; any other layout, and a partial rotation's untouched
/// dims, are a copy into the output that the kernel rotates in place, as MLX's donated cases.
pub fn rope(x: *Tensor, dims: usize, traditional: bool, base: f32, scale: f32, offset_in: *Tensor, freqs_in: ?*Tensor) Error!*Tensor {
    switch (x.dtype) {
        .float32, .float16, .bfloat16 => {},
        else => return core.refusing("rope over a dtype other than f32, f16 and bf16", error.TensorDtype),
    }
    const nd: usize = x.ndim;
    if (nd < 3) return error.TensorShape;
    const lead: usize = @intCast(x.shape[0]);
    const seq: usize = @intCast(x.shape[nd - 2]);
    const width: usize = @intCast(x.shape[nd - 1]);
    var heads: usize = 1;
    for (1..nd - 2) |i| heads *= @intCast(x.shape[i]);
    if (dims == 0 or dims % 2 != 0 or dims > width) return error.TensorShape;
    if (offset_in.ndim > 1 or (offset_in.size() != 1 and offset_in.size() != lead)) return error.TensorShape;
    const offset = switch (offset_in.dtype) {
        .int32, .uint32 => try core.share(offset_in),
        .int8, .int16, .int64, .uint8, .uint16, .uint64 => try ops.astype(offset_in, .int32),
        else => return core.refusing("rope with an offset that is not an integer", error.TensorDtype),
    };
    defer core.release(offset);
    if (x.size() > std.math.maxInt(i32)) return core.refusing("a rope input past int32 elements", error.TensorUnsupported);
    const single = x.rowContiguous() and seq == 1 and offset.size() == 1;
    const donated = dims < width or !x.rowContiguous();
    const out = if (donated) try ops.contiguousCopy(x) else try core.fresh(x.dtype, x.dims());
    errdefer core.release(out);
    if (out.size() == 0) return out;
    const src = if (donated) out else x;
    const mat: i64 = @intCast(seq * width);
    const strides = [3]i64{ mat, @intCast(src.strides[nd - 2]), @intCast(src.strides[nd - 1]) };
    const out_strides = [3]i64{ mat, @intCast(out.strides[nd - 2]), @intCast(out.strides[nd - 1]) };
    var buf: [64]u8 = undefined;
    const name = try named(&buf, "rope_{s}{s}{s}", .{ if (single) "single_" else "", if (freqs_in != null) "freqs_" else "", try typeName(x.dtype) });
    var hbuf: [96]u8 = undefined;
    const hash = try named(&hbuf, "{s}_{s}", .{ name, if (traditional) "traditional_" else "" });
    const k = try kernelFlags(name, hash, &.{ .{ .index = 1, .value = true }, .{ .index = 2, .value = traditional }, .{ .index = 3, .value = false } });
    const freqs = if (freqs_in) |f| try ops.astype(f, .float32) else null;
    defer if (freqs) |f| core.release(f);
    const p = try core.pass();
    p.pipeline(k);
    core.bind(p, src, 0);
    core.bindOut(p, out, 1);
    core.bind(p, offset, 2);
    p.constant(scale, 3);
    var grid: metal.Size = undefined;
    if (single) {
        p.constant(out_strides[0], 4);
        grid = .{ .x = dims / 2, .y = lead * heads, .z = 1 };
    } else {
        p.bytes(std.mem.sliceAsBytes(&strides), 4);
        p.bytes(std.mem.sliceAsBytes(&out_strides), 5);
        p.constant(@as(i64, if (offset.ndim > 0) @intCast(offset.strides[0]) else 0), 6);
        p.constant(@as(c_int, @intCast(heads)), 7);
        grid = .{ .x = dims / 2, .y = seq, .z = lead * ((heads + rope_per_thread - 1) / rope_per_thread) };
    }
    if (freqs) |f| {
        core.bind(p, f, 10);
        p.constant(@as(i64, @intCast(f.strides[0])), 11);
    } else p.constant(std.math.log2(base), 10);
    core.dispatch(p, grid, blockDims(grid.x, grid.y, grid.z));
    return out;
}

/// `MLXConvParams<2>` (kernels/steel/conv/params.h), field for field.
const ConvParams2 = extern struct {
    N: i32,
    C: i32,
    O: i32,
    iS: [2]i32,
    wS: [2]i32,
    oS: [2]i32,
    str: [2]i32,
    pad: [2]i32,
    kdil: [2]i32,
    idil: [2]i32,
    in_strides: [4]i64,
    wt_strides: [4]i64,
    out_strides: [4]i64,
    groups: i32,
    flip: bool,
};

/// `ImplicitGemmConv2DParams`.
const ImplicitGemmParams = extern struct {
    M: i32,
    N: i32,
    K: i32,
    gemm_k_iterations: i32,
    inp_jump_w: i32,
    inp_jump_h: i32,
    inp_jump_c: i32,
    tiles_n: i32,
    tiles_m: i32,
    swizzle_log: i32,
};

/// A depthwise conv1d with a dilated kernel, as `conv_1D_gpu` sends it on: as a 2-D conv with a unit second axis to
/// `dispatch_conv_2D_gpu`, which, for one channel a group under a dilated kernel, runs `implicit_gemm_conv_2D_gpu`
/// (the steel implicit GEMM). Qwen's PLE conv (dilation the n-gram size) only ever takes this path.
pub fn dilatedDepthwiseConv1d(in_pre: *Tensor, w_pre: *Tensor, dilation: usize) Error!*Tensor {
    switch (in_pre.dtype) {
        .float32, .float16, .bfloat16 => {},
        else => return core.refusing("a dilated depthwise conv1d over a dtype other than f32, f16 and bf16", error.TensorDtype),
    }
    if (w_pre.dtype != in_pre.dtype) return core.refusing("a conv1d of two dtypes", error.TensorDtype);
    if (in_pre.ndim != 3 or w_pre.ndim != 3 or dilation == 0) return error.TensorShape;
    const in = try ops.contiguous(in_pre);
    defer core.release(in);
    const w = try ops.contiguous(w_pre);
    defer core.release(w);
    const n: usize = @intCast(in.shape[0]);
    const t: usize = @intCast(in.shape[1]);
    const c: usize = @intCast(in.shape[2]);
    const k: usize = @intCast(w.shape[1]);
    const span = dilation * (k - 1) + 1;
    if (k == 0 or span > t) return error.TensorShape;
    const tout = t - span + 1;
    const out = try core.fresh(in.dtype, &.{ @intCast(n), @intCast(tout), @intCast(c) });
    errdefer core.release(out);
    if (out.size() == 0) return out;
    if (in.size() > std.math.maxInt(i32) or n * tout > std.math.maxInt(i32)) return core.refusing("a conv1d past int32 elements", error.TensorUnsupported);
    const params = ConvParams2{
        .N = @intCast(n),
        .C = @intCast(c),
        .O = @intCast(c),
        .iS = .{ @intCast(t), 1 },
        .wS = .{ @intCast(k), 1 },
        .oS = .{ @intCast(tout), 1 },
        .str = .{ 1, 1 },
        .pad = .{ 0, 0 },
        .kdil = .{ @intCast(dilation), 1 },
        .idil = .{ 1, 1 },
        .in_strides = .{ @intCast(in.strides[0]), @intCast(in.strides[1]), 0, @intCast(in.strides[2]) },
        .wt_strides = .{ @intCast(w.strides[0]), @intCast(w.strides[1]), 0, @intCast(w.strides[2]) },
        .out_strides = .{ @intCast(out.strides[0]), @intCast(out.strides[1]), 0, @intCast(out.strides[2]) },
        .groups = @intCast(c),
        .flip = false,
    };
    // `implicit_gemm_conv_2D_gpu` at one channel a group: N = 1 forces bn 8, wm 4, wn 1; C_per_group 1 is the
    // one-channel specialization, so the filter is the large one.
    const m: i32 = @intCast(n * tout);
    const bm: i32 = 32;
    const bn: i32 = 8;
    const bk: i32 = 16;
    const ijh: i32 = @intCast(in.strides[1] * dilation);
    const gemm = ImplicitGemmParams{
        .M = m,
        .N = 1,
        .K = @intCast(k),
        .gemm_k_iterations = @divTrunc(@as(i32, @intCast(k)) + bk - 1, bk),
        .inp_jump_w = 0,
        .inp_jump_h = ijh,
        .inp_jump_c = bk - (@as(i32, @intCast(k)) - 1) * ijh,
        .tiles_n = @divTrunc(1 + bn - 1, bn),
        .tiles_m = @divTrunc(m + bm - 1, bm),
        .swizzle_log = 0,
    };
    var buf: [96]u8 = undefined;
    const kern = try kernel(try named(&buf, "implicit_gemm_conv_2d_{s}_bm32_bn8_bk16_wm4_wn1_channel_1_filter_l", .{try typeName(in.dtype)}));
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, in, 0);
    core.bind(p, w, 1);
    core.bindOut(p, out, 2);
    p.bytes(std.mem.asBytes(&params), 3);
    p.bytes(std.mem.asBytes(&gemm), 4);
    core.dispatchGroups(p, .{ .x = @intCast(gemm.tiles_n), .y = @intCast(gemm.tiles_m), .z = c }, .{ .x = 32, .y = 1, .z = 4 });
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
        if (given.dtype != scales.dtype) return core.refusing("dequantize with biases whose dtype is not the scales'", error.TensorDtype);
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
    const cast = try ops.astype(out, out_type);
    core.release(out);
    return cast;
}

/// `quantize` in affine mode (`quantize_impl`): the packed codes of `w` (uint32) and, per group of `group` values
/// along the last axis, a scale and a bias in `w`'s dtype. The caller releases the three.
pub fn quantize(w_in: *Tensor, group: usize, bits: usize) Error![3]*Tensor {
    if (w_in.ndim == 0) return error.TensorShape;
    if (group != 32 and group != 64 and group != 128) return core.refusing("an affine group size other than 32, 64 and 128", error.TensorUnsupported);
    if (bits < 2 or bits > 8 or bits == 7) return core.refusing("affine bits other than 2 to 6 and 8", error.TensorUnsupported);
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
    return unaryOf("Erf", .float, x_in);
}

pub fn tanh(x: *Tensor) Error!*Tensor {
    return unaryOf("Tanh", .float, x);
}
pub fn cos(x: *Tensor) Error!*Tensor {
    return unaryOf("Cos", .float, x);
}
pub fn sin(x: *Tensor) Error!*Tensor {
    return unaryOf("Sin", .float, x);
}
pub fn log1p(x: *Tensor) Error!*Tensor {
    return unaryOf("Log1p", .float, x);
}
pub fn sign(x: *Tensor) Error!*Tensor {
    return unaryOf("Sign", .all, x);
}

/// The dtypes `kernels/unary.metal` instantiates an op for: `instantiate_unary_float` or `instantiate_unary_types`.
pub const UnaryTypes = enum { float, all };

/// A unary op of `mlx.metallib` over a row-major input, as `unary_op_gpu` launches its contiguous kernels: `v` under
/// 65536 elements, `vn` (eight bytes a thread) from there. MLX's own kernel, so MLX's bytes.
pub fn unaryOf(comptime op: []const u8, comptime types: UnaryTypes, x_in: *Tensor) Error!*Tensor {
    switch (x_in.dtype) {
        .float32, .float16, .bfloat16 => {},
        .bool_, .uint8, .uint16, .uint32, .int8, .int16, .int32 => if (types == .float) return core.refusing("a float-only unary op over an integer or bool: the seam casts it first", error.TensorDtype),
        else => return core.refusing("a unary op over a dtype MLX's kernels do not take", error.TensorDtype),
    }
    // MLX gives a contiguous input that is not row-major an output of its strides; this port makes row-major ones only.
    if (!x_in.rowContiguous()) return core.refusing("a unary input that is not row-major", error.TensorUnsupported);
    const x = try core.share(x_in);
    defer core.release(x);
    const out = try core.fresh(x.dtype, x.dims());
    errdefer core.release(out);
    const size = x.size();
    if (size == 0) return out;
    if (size > std.math.maxInt(u32)) return core.refusing("a unary op over more than 2^32 elements", error.TensorUnsupported);
    const type_name = try typeName(x.dtype);
    // `get_work_per_thread`: 8 bytes a thread from 65536 elements.
    const per_thread: usize = if (size < 1 << 16) 1 else @max(1, 8 / core.itemSize(x.dtype));
    var buf: [64]u8 = undefined;
    const k = try kernel(try named(&buf, "{s}_" ++ op ++ "{s}{s}", .{ if (per_thread > 1) "vn" else "v", type_name, type_name }));
    // Until the load seals the pipelines, the other size's kernel is made with it: a request's length picks it.
    if (!core.sealed) _ = kernel(try named(&buf, "{s}_" ++ op ++ "{s}{s}", .{ if (per_thread > 1) "v" else "vn", type_name, type_name })) catch {};
    const threads = (size + per_thread - 1) / per_thread;
    const p = try core.pass();
    p.pipeline(k);
    core.bind(p, x, 0);
    core.bindOut(p, out, 1);
    p.constant(@as(c_int, @intCast(size)), 2);
    core.dispatch(p, .{ .x = threads }, .{ .x = @min(k.max_threads, threads) });
    return out;
}

/// A binary op of `mlx.metallib` over two operands of one dtype (`types` the ones MLX instantiates it for), under
/// `binary_op_gpu_inplace`: the scalar and vector kernels of one flat run. An operand pair MLX would run through its
/// strided kernels is refused.
pub fn binaryOf(comptime op: []const u8, comptime types: UnaryTypes, a: *Tensor, b: *Tensor) Error!*Tensor {
    if (a.dtype != b.dtype) return core.refusing("a binary op of the metallib over two dtypes: the seam promotes them first", error.TensorDtype);
    switch (a.dtype) {
        .float16, .bfloat16, .float32 => {},
        .bool_, .uint8, .uint16, .uint32, .int8, .int16, .int32 => if (types == .float) return core.refusing("a float-only binary op of the metallib over an integer or bool", error.TensorDtype),
        else => return core.refusing("a binary op of the metallib over a dtype its kernels do not take", error.TensorDtype),
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
    if (size > std.math.maxInt(u32)) return core.refusing("a binary op over more than 2^32 elements", error.TensorUnsupported);
    const kind: []const u8 = if (a.size() == 1 and b.size() == 1)
        "ss"
    else if (a.size() == 1 and b.rowContiguous() and b.size() == size)
        "sv"
    else if (b.size() == 1 and a.rowContiguous() and a.size() == size)
        "vs"
    else if (a.rowContiguous() and b.rowContiguous() and a.size() == size and b.size() == size)
        "vv"
    else
        return core.refusing("a binary op on broadcast or strided inputs (only the scalar and row-major arms are ported)", error.TensorUnsupported);
    const flat = !std.mem.eql(u8, kind, "ss");
    // Eight bytes of a row per thread once it is wide, as `get_work_per_thread`.
    const per_thread: usize = if (flat and size >= 1 << 16) @max(1, 8 / core.itemSize(a.dtype)) else 1;
    var buf: [96]u8 = undefined;
    const k = try kernel(try named(&buf, "{s}{s}_" ++ op ++ "{s}", .{ kind, if (flat and per_thread > 1) "n" else "", try typeName(a.dtype) }));
    if (!core.sealed and flat) _ = kernel(try named(&buf, "{s}{s}_" ++ op ++ "{s}", .{ kind, if (per_thread > 1) "" else "n", try typeName(a.dtype) })) catch {};
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
