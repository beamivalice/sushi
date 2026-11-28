//! Sushi Cooler's stock kernels: the compiled functions of `sushi.metallib` (built from MLX's pinned MIT Metal kernel
//! headers, see NOTICE), fetched by the names MLX gives them and dispatched with the buffers, constants and grids MLX's
//! `eval_gpu` uses, so an op here writes the bytes it writes under MLX. Ported from MLX's Metal backend (MIT, see NOTICE).

const std = @import("std");
const metal = @import("../metal.zig");
const metallib_path = @import("../metallib_path.zig");
const log = @import("../log.zig");
const core = @import("core.zig");
const ops = @import("ops.zig");
const registry = @import("pipeline_registry.zig");
const manifest = @import("../pipeline_manifest.zig");

const Tensor = core.Tensor;
const Dtype = core.Dtype;
const Error = core.Error;
const gpa = std.heap.c_allocator;

var library: ?metal.Library = null;
var pipelines: std.StringHashMapUnmanaged(metal.Pipeline) = .empty;

/// This process's `sushi.metallib` (`metallib_path.forThisProcess`). A binary without one beside it says where it looked.
pub fn metallibPath(buf: []u8) Error![:0]const u8 {
    return metallib_path.forThisProcess(buf) catch {
        log.err("[engine] sushi.metallib is not beside this binary (looked for {s}): reinstall sushi\n", .{metallib_path.searched});
        return error.MetalUnavailable;
    };
}

/// A binary without its library stays without it: the first miss is logged and every later ask fails at once.
var library_missing = false;

pub fn lib(dev: *const metal.Device) Error!metal.Library {
    if (library) |l| return l;
    if (library_missing) return error.MetalUnavailable;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = metallibPath(&buf) catch |e| {
        library_missing = true;
        return e;
    };
    library = try dev.load(path);
    return library.?;
}

/// While set, a stock pipeline neither made nor kept is noted as its recipe and not made: the caller makes the notes on
/// worker threads (`pipeline_cache`) and then asks again, taking each from the stock.
pub var noting: ?*Notes = null;

pub const Notes = struct {
    arena: std.heap.ArenaAllocator = .init(std.heap.c_allocator),
    recipes: std.ArrayList(manifest.Recipe) = .empty,
    keys: std.StringHashMapUnmanaged(void) = .empty,

    pub fn deinit(n: *Notes) void {
        n.arena.deinit();
    }

    fn note(n: *Notes, r: manifest.Recipe) Error!void {
        if (pipelines.contains(r.key) or registry.inStock(r.key) or n.keys.contains(r.key)) return;
        const a = n.arena.allocator();
        const key = a.dupe(u8, r.key) catch return error.OutOfMemory;
        n.keys.put(a, key, {}) catch return error.OutOfMemory;
        n.recipes.append(a, .{
            .kind = r.kind,
            .function = a.dupe(u8, r.function) catch return error.OutOfMemory,
            .key = key,
            .flags = a.dupe(metal.Flag, r.flags) catch return error.OutOfMemory,
            .ints = a.dupe(metal.IntConstant, r.ints) catch return error.OutOfMemory,
        }) catch return error.OutOfMemory;
    }
};

/// Keeps `pipeline`, made ahead from `recipe`, for the first `kernel` or `kernelFlags` that asks for its key.
pub fn preload(recipe: manifest.Recipe, pipeline: metal.Pipeline) Error!bool {
    if (pipelines.contains(recipe.key)) return false;
    return registry.putStock(recipe, pipeline) catch error.OutOfMemory;
}

/// True when `name` is already loaded in `pipelines` or waiting in preloaded stock.
pub fn has(name: []const u8) bool {
    return pipelines.contains(name) or registry.inStock(name);
}

/// Tests only: `kernel` meets `name` next as a new process does, whatever an earlier test built. A holder of the
/// pipeline keeps it.
pub fn forget(name: []const u8) void {
    _ = pipelines.remove(name);
    registry.forgetStock(name);
}

/// The pipeline of the stock kernel `name`, built once.
pub fn kernel(name: []const u8) Error!metal.Pipeline {
    if (core.dryOn()) return core.dry_pipeline;
    if (pipelines.get(name)) |p| return p;
    if (noting) |n| {
        try n.note(.{ .kind = .stock, .function = name, .key = name });
        return core.dry_pipeline;
    }
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
    if (core.dryOn()) return core.dry_pipeline;
    if (pipelines.get(hash)) |p| return p;
    if (noting) |n| {
        try n.note(.{ .kind = .stock_flags, .function = name, .key = hash, .flags = flags });
        return core.dry_pipeline;
    }
    const dev = try metal.device();
    var buf: [512]u8 = undefined;
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
    if (core.dryOn()) return core.dry_pipeline;
    if (pipelines.get(hash)) |p| return p;
    if (noting) |n| {
        try n.note(.{ .kind = .stock_flags, .function = name, .key = hash, .flags = flags, .ints = ints });
        return core.dry_pipeline;
    }
    const dev = try metal.device();
    var buf: [512]u8 = undefined;
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

/// A unary op of `sushi.metallib` over a row-major input, as `unary_op_gpu` launches its contiguous kernels: `v` under
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

/// A binary op of `sushi.metallib` over two operands of one dtype (`types` the ones MLX instantiates it for), under
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

// ── Tests ──

const testing = std.testing;
const matmul = @import("matmul.zig");
const device = @import("sushi_exl3").device;

fn specializeMaxThreads(d: *const metal.Device, lib_h: metal.Library, r: *const @import("../metallib_catalog.zig").Resolved) !usize {
    const catalog = @import("../metallib_catalog.zig");
    var fbuf: [512]u8 = undefined;
    const fname = try std.fmt.bufPrintSentinel(&fbuf, "{s}", .{r.fnName()}, 0);
    var flags: [catalog.max_fc_bools]metal.Flag = undefined;
    for (r.flagSlice(), 0..) |f, i| flags[i] = .{ .index = f.index, .value = f.value };
    var ints_fc: [catalog.max_fc_ints]metal.IntConstant = undefined;
    for (r.intSlice(), 0..) |c, i| ints_fc[i] = .{ .index = c.index, .value = c.value };
    const p = try d.pipelineOfConstants(lib_h, fname, flags[0..r.n_flags], ints_fc[0..r.n_ints]);
    return p.max_threads;
}

fn readU16(t: *const Tensor) ![]const u16 {
    const b = try core.hostRead(t);
    return std.mem.bytesAsSlice(u16, @as([]align(2) const u8, @alignCast(b)));
}

/// An identity matmul accumulates into `+0.0f`, so IEEE-754 `+0.0f + 1.0 * (-0.0f)` yields `+0.0h` (`0x0000`)
/// even when a codebook codeword decodes `-0.0h` (`0x8000`). Every nonzero half matches bit-for-bit.
fn canonF16Zero(bits: u16) u16 {
    return if (bits == 0x8000) 0 else bits;
}

fn expectMatmulU16(want: []const u16, got: []const u16) !void {
    try testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try testing.expectEqual(canonF16Zero(w), g);
}

test "stock metallib: metallibPath resolves sushi.metallib and never mlx.metallib" {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try metallibPath(&buf);
    try testing.expect(std.mem.endsWith(u8, path, "/sushi.metallib"));
    try testing.expect(!std.mem.endsWith(u8, path, "/mlx.metallib"));
    try testing.expect(std.c.access(path.ptr, std.c.F_OK) == 0);

}

test "stock metallib: every kernel src/tensor can construct across all GPU classes exists in sushi.metallib" {
    try @import("../mlx.zig").requireGpu();
    const dev = try metal.device();
    const l = try lib(dev);
    const names = try l.functionNames(testing.allocator);
    defer {
        for (names) |s| testing.allocator.free(s);
        testing.allocator.free(names);
    }
    var have: std.StringHashMapUnmanaged(void) = .empty;
    defer have.deinit(testing.allocator);
    for (names) |s| try have.put(testing.allocator, s, {});

    var missing: usize = 0;
    var first_buf: [192]u8 = undefined;
    var first_len: usize = 0;
    const Check = struct {
        map: *const std.StringHashMapUnmanaged(void),
        missing: *usize,
        first_buf: *[192]u8,
        first_len: *usize,
        fn has(self: @This(), name: []const u8) void {
            if (!self.map.contains(name)) {
                if (self.first_len.* == 0) {
                    const n = @min(name.len, self.first_buf.len);
                    @memcpy(self.first_buf[0..n], name[0..n]);
                    self.first_len.* = n;
                }
                self.missing.* += 1;
            }
        }
        fn fmt(self: @This(), comptime pattern: []const u8, args: anytype) void {
            var buf: [192]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, pattern, args) catch unreachable;
            self.has(s);
        }
    };
    const check: Check = .{ .map = &have, .missing = &missing, .first_buf = &first_buf, .first_len = &first_len };
    const floats = [_][]const u8{ "float16", "bfloat16", "float32" };
    const c_floats = [_][]const u8{ "float", "float16_t", "bfloat16_t" };
    const all_elem = [_][]const u8{ "bool_", "uint8", "uint16", "uint32", "int8", "int16", "int32", "float16", "bfloat16", "float32" };
    const sortable = [_][]const u8{ "uint8", "uint16", "uint32", "int8", "int16", "int32", "float16", "float32", "bfloat16" };
    const argmax_types = [_][]const u8{ "bool_", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float16", "float32", "bfloat16" };

    // stock.zig: norms, softmax, logsumexp, conv1d, rope
    for (floats) |t| {
        check.fmt("rms{s}", .{t});
        check.fmt("rms_looped{s}", .{t});
        check.fmt("layer_norm{s}", .{t});
        check.fmt("layer_norm_looped{s}", .{t});
        check.fmt("block_softmax_{s}", .{t});
        check.fmt("looped_softmax_{s}", .{t});
        check.fmt("block_logsumexp_{s}", .{t});
        check.fmt("looped_logsumexp_{s}", .{t});
        check.fmt("depthwise_conv_1d_{s}", .{t});
        check.fmt("implicit_gemm_conv_2d_{s}_bm32_bn8_bk16_wm4_wn1_channel_1_filter_l", .{t});
        for ([_][]const u8{ "", "single_" }) |s| for ([_][]const u8{ "", "freqs_" }) |f| {
            check.fmt("rope_{s}{s}{s}", .{ s, f, t });
        };
    }
    for ([_][]const u8{ "float16", "bfloat16" }) |t| {
        check.fmt("block_softmax_precise_{s}", .{t});
        check.fmt("looped_softmax_precise_{s}", .{t});
    }

    // stock.zig / ops.zig / route.zig: unary & binary
    for ([_][]const u8{ "Erf", "Tanh", "Cos", "Sin", "Log1p" }) |op| for (floats) |t| {
        check.fmt("v_{s}{s}{s}", .{ op, t, t });
        check.fmt("vn_{s}{s}{s}", .{ op, t, t });
    };
    for ([_][]const u8{ "Sign", "Abs", "Floor" }) |op| for (all_elem) |t| {
        check.fmt("v_{s}{s}{s}", .{ op, t, t });
        check.fmt("vn_{s}{s}{s}", .{ op, t, t });
    };
    for (floats) |t| for ([_][]const u8{ "ss", "sv", "vs", "vv", "svn", "vsn", "vvn" }) |k| {
        check.fmt("{s}_LogAddExp{s}", .{ k, t });
    };
    for ([_][]const u8{ "Power", "Remainder" }) |op| for (all_elem) |t| for ([_][]const u8{ "ss", "sv", "vs", "vv", "svn", "vsn", "vvn" }) |k| {
        check.fmt("{s}_{s}{s}", .{ k, op, t });
    };

    // matmul.zig: dot, gemv, gemv_wide, steel_gemm_fused{,_nax}, steel_gemm_splitk{,_nax}
    for (floats) |t| {
        check.fmt("dot_product_{s}_it32_tg512_sg16", .{t});
        check.fmt("steel_gemm_splitk_accum_{s}_float32", .{t});
    }
    for ([_][]const u8{ "float16", "bfloat16" }) |t| for ([_]usize{ 2, 3, 4, 5 }) |nv| for ([_]usize{ 16, 32 }) |kl| {
        check.fmt("gemv_wide_{s}_nv{d}_kl{d}", .{ t, nv, kl });
        check.fmt("gemv_wide_gather_{s}_nv{d}_kl{d}", .{ t, nv, kl });
    };
    const gemv_blocks = [_][6]usize{
        .{ 1, 8, 1, 32, 4, 4 }, .{ 1, 8, 1, 32, 1, 4 }, .{ 1, 1, 8, 4, 4, 4 }, .{ 1, 1, 8, 4, 1, 4 },
        .{ 4, 1, 1, 32, 1, 4 }, .{ 4, 1, 1, 32, 4, 4 }, .{ 8, 1, 1, 32, 4, 4 },
    };
    const gemv_gather_blocks = [_][6]usize{
        .{ 4, 1, 1, 32, 1, 4 }, .{ 4, 1, 1, 32, 4, 4 }, .{ 8, 1, 1, 32, 4, 4 },
    };
    const gemv_t_blocks = [_][6]usize{
        .{ 1, 2, 8, 4, 4, 1 }, .{ 1, 2, 8, 4, 4, 4 }, .{ 1, 4, 8, 4, 4, 4 }, .{ 1, 16, 8, 4, 4, 4 }, .{ 1, 16, 4, 8, 4, 4 },
    };
    for (floats) |t| {
        for (gemv_blocks) |b| for ([_]usize{ 0, 1 }) |nc| {
            check.fmt("gemv_{s}_bm{d}_bn{d}_sm{d}_sn{d}_tm{d}_tn{d}_nc{d}_axpby0", .{ t, b[0], b[1], b[2], b[3], b[4], b[5], nc });
        };
        for (gemv_gather_blocks) |b| {
            check.fmt("gemv_gather_{s}_bm{d}_bn{d}_sm{d}_sn{d}_tm{d}_tn{d}", .{ t, b[0], b[1], b[2], b[3], b[4], b[5] });
        }
        for (gemv_t_blocks) |b| {
            for ([_]usize{ 0, 1 }) |nc| {
                check.fmt("gemv_t_{s}_bm{d}_bn{d}_sm{d}_sn{d}_tm{d}_tn{d}_nc{d}_axpby0", .{ t, b[0], b[1], b[2], b[3], b[4], b[5], nc });
            }
            check.fmt("gemv_t_gather_{s}_bm{d}_bn{d}_sm{d}_sn{d}_tm{d}_tn{d}", .{ t, b[0], b[1], b[2], b[3], b[4], b[5] });
        }
    }
    const classes = [_]device.Class{ .g, .p, .s, .c, .d };
    const edge_sizes = [_]usize{ 16, 512, 1024, 2048, 2049, 4096, 8192, 12288, 1 << 20 };
    const dtypes = [_]Dtype{ .float16, .bfloat16, .float32 };
    for (classes) |class| {
        const g: matmul.Gpu = .{ .class = class, .gen = 17, .nax = true, .tf32 = true };
        for (dtypes) |dt| {
            const tn = try typeName(dt);
            for ([_]bool{ false, true }) |ta| for ([_]bool{ false, true }) |tb| {
                const ca: u8 = if (ta) 't' else 'n';
                const cb: u8 = if (tb) 't' else 'n';
                for (edge_sizes) |m| for (edge_sizes) |n| for (edge_sizes) |k| {
                    const ft = matmul.gemmTile(g, dt, ta, tb, 1, m, n, k);
                    check.fmt("steel_gemm_fused_{c}{c}_{s}_{s}_bm{d}_bn{d}_bk{d}_wm{d}_wn{d}", .{ ca, cb, tn, tn, ft.bm, ft.bn, ft.bk, ft.wm, ft.wn });
                    check.fmt("steel_gather_mm_{c}{c}_{s}_{s}_bm{d}_bn{d}_bk{d}_wm{d}_wn{d}", .{ ca, cb, tn, tn, ft.bm, ft.bn, ft.bk, ft.wm, ft.wn });
                    const nt = matmul.fusedNaxTileFor(class, m, m, n, k).t;
                    check.fmt("steel_gemm_fused_nax_{c}{c}_{s}_{s}_bm{d}_bn{d}_bk{d}_wm{d}_wn{d}", .{ ca, cb, tn, tn, nt.bm, nt.bn, nt.bk, nt.wm, nt.wn });
                    const sk = matmul.splitKNaxTileFor(class, m, n, k).t;
                    check.fmt("steel_gemm_splitk_nax_{c}{c}_{s}_float32_bm{d}_bn{d}_bk{d}_wm{d}_wn{d}", .{ ca, cb, tn, sk.bm, sk.bn, sk.bk, sk.wm, sk.wn });
                };
            };
        }
    }
    for (floats) |t| for ([_][2]u8{ .{ 'n', 'n' }, .{ 'n', 't' }, .{ 't', 'n' }, .{ 't', 't' } }) |tr| {
        for ([_]usize{ 16, 32 }) |bm| for ([_]usize{ 16, 32 }) |bn| for ([_]u8{ 't', 'n' }) |amn| for ([_]u8{ 't', 'n' }) |ak| {
            check.fmt("steel_gemm_splitk_{c}{c}_{s}_float32_bm{d}_bn{d}_bk16_wm2_wn2_MN_{c}aligned_K_{c}aligned", .{ tr[0], tr[1], t, bm, bn, amn, ak });
        };
    };

    // quantized.zig + stock.zig quantize/dequantize
    check.has("gather_mm_offsets");
    for (c_floats) |ct| {
        for ([_]usize{ 32, 64, 128 }) |gs| for ([_]usize{ 2, 3, 4, 5, 6, 8 }) |bits| {
            check.fmt("affine_quantize_{s}_gs_{d}_b_{d}", .{ ct, gs, bits });
            check.fmt("affine_dequantize_{s}_gs_{d}_b_{d}", .{ ct, gs, bits });
            check.fmt("affine_gather_qmv_{s}_gs_{d}_b_{d}", .{ ct, gs, bits });
            check.fmt("affine_gather_qmv_fast_{s}_gs_{d}_b_{d}", .{ ct, gs, bits });
            check.fmt("affine_gather_qmm_rhs_nt_{s}_gs_{d}_b_{d}_bm_16_bn_32_bk_32_wm_1_wn_2", .{ ct, gs, bits });
            for ([_]usize{ 32, 64 }) |bm| {
                check.fmt("affine_gather_qmm_rhs_nax_nt_{s}_gs_{d}_b_{d}_bm_{d}_bn_64_bk_64_wm_2_wn_2", .{ ct, gs, bits, bm });
            }
            for ([_]usize{ 0, 1 }) |b| {
                check.fmt("affine_qmv_{s}_gs_{d}_b_{d}_batch_{d}", .{ ct, gs, bits, b });
                check.fmt("affine_qmv_fast_{s}_gs_{d}_b_{d}_batch_{d}", .{ ct, gs, bits, b });
                if (std.math.isPowerOfTwo(bits)) for ([_]usize{ 64, 128 }) |d| {
                    check.fmt("affine_qmv_quad_{s}_gs_{d}_b_{d}_d_{d}_batch_{d}", .{ ct, gs, bits, d, b });
                };
                for ([_]usize{ 2, 3, 4, 5 }) |nv| {
                    check.fmt("affine_qmv_wide_{s}_gs_{d}_b_{d}_nv_{d}_kl_8_batch_{d}", .{ ct, gs, bits, nv, b });
                }
            }
            for ([_][]const u8{ "true", "false" }) |al| {
                check.fmt("affine_qmm_t_splitk_{s}_gs_{d}_b_{d}_alN_{s}", .{ ct, gs, bits, al });
                check.fmt("affine_gather_qmm_t_{s}_gs_{d}_b_{d}_alN_{s}", .{ ct, gs, bits, al });
                check.fmt("affine_gather_qmm_t_nax_{s}_gs_{d}_b_{d}_bm64_bn64_bk64_wm2_wn2_alN_{s}", .{ ct, gs, bits, al });
                for ([_]usize{ 0, 1 }) |b| {
                    check.fmt("affine_qmm_t_{s}_gs_{d}_b_{d}_alN_{s}_batch_{d}", .{ ct, gs, bits, al, b });
                    for ([_]usize{ 32, 64 }) |bm| {
                        check.fmt("affine_qmm_t_nax_{s}_gs_{d}_b_{d}_bm{d}_bn64_bk64_wm2_wn2_alN_{s}_batch_{d}", .{ ct, gs, bits, bm, al, b });
                    }
                }
            }
        };
        const mx_modes = [_]struct { name: []const u8, bits: usize }{ .{ .name = "mxfp4", .bits = 4 }, .{ .name = "mxfp8", .bits = 8 } };
        for (mx_modes) |mx| {
            check.fmt("{s}_dequantize_{s}_gs_32_b_{d}_hgs_false", .{ mx.name, ct, mx.bits });
            check.fmt("{s}_gather_qmv_{s}_gs_32_b_{d}", .{ mx.name, ct, mx.bits });
            check.fmt("{s}_gather_qmv_fast_{s}_gs_32_b_{d}", .{ mx.name, ct, mx.bits });
            check.fmt("{s}_gather_qmm_rhs_nt_{s}_gs_32_b_{d}_bm_16_bn_32_bk_32_wm_1_wn_2", .{ mx.name, ct, mx.bits });
            for ([_]usize{ 32, 64 }) |bm| {
                check.fmt("{s}_gather_qmm_rhs_nax_nt_{s}_gs_32_b_{d}_bm_{d}_bn_64_bk_64_wm_2_wn_2", .{ mx.name, ct, mx.bits, bm });
            }
            for ([_]usize{ 0, 1 }) |b| {
                check.fmt("{s}_qmv_{s}_gs_32_b_{d}_batch_{d}", .{ mx.name, ct, mx.bits, b });
                check.fmt("{s}_qmv_fast_{s}_gs_32_b_{d}_batch_{d}", .{ mx.name, ct, mx.bits, b });
                for ([_]usize{ 64, 128 }) |d| {
                    check.fmt("{s}_qmv_quad_{s}_gs_32_b_{d}_d_{d}_batch_{d}", .{ mx.name, ct, mx.bits, d, b });
                }
                for ([_]usize{ 2, 3, 4, 5 }) |nv| {
                    check.fmt("{s}_qmv_wide_{s}_gs_32_b_{d}_nv_{d}_kl_16_batch_{d}", .{ mx.name, ct, mx.bits, nv, b });
                }
            }
            for ([_][]const u8{ "true", "false" }) |al| {
                check.fmt("{s}_qmm_t_splitk_{s}_gs_32_b_{d}_alN_{s}", .{ mx.name, ct, mx.bits, al });
                check.fmt("{s}_gather_qmm_t_{s}_gs_32_b_{d}_alN_{s}", .{ mx.name, ct, mx.bits, al });
                check.fmt("{s}_gather_qmm_t_nax_{s}_gs_32_b_{d}_bm64_bn64_bk64_wm2_wn2_alN_{s}", .{ mx.name, ct, mx.bits, al });
                for ([_]usize{ 0, 1 }) |b| {
                    check.fmt("{s}_qmm_t_{s}_gs_32_b_{d}_alN_{s}_batch_{d}", .{ mx.name, ct, mx.bits, al, b });
                    for ([_]usize{ 32, 64 }) |bm| {
                        check.fmt("{s}_qmm_t_nax_{s}_gs_32_b_{d}_bm{d}_bn64_bk64_wm2_wn2_alN_{s}_batch_{d}", .{ mx.name, ct, mx.bits, bm, al, b });
                    }
                }
            }
        }
        check.fmt("nvfp4_dequantize_{s}_gs_16_b_4_hgs_false", .{ct});
        check.fmt("nvfp4_dequantize_{s}_gs_16_b_4_hgs_true", .{ct});
    }

    // attention.zig: SIMD & NAX full attention + vector & 2-pass attention
    for (floats) |t| {
        check.fmt("steel_attention_{s}_bq32_bk32_bd64_bv64_wm4_wn1_mask{s}", .{ t, t });
        check.fmt("steel_attention_{s}_bq64_bk32_bd64_bv64_wm4_wn1_mask{s}", .{ t, t });
        check.fmt("steel_attention_dsplit_{s}_bq64_bk32_bd256_bv256_wm4_wn2_mask{s}", .{ t, t });
    }
    for (c_floats) |ct| {
        for ([_]usize{ 64, 96, 128, 256 }) |d| {
            check.fmt("sdpa_vector_{s}_{d}_{d}", .{ ct, d, d });
            check.fmt("sdpa_vector_2pass_1_{s}_{d}_{d}", .{ ct, d, d });
            check.fmt("sdpa_vector_2pass_2_{s}_{d}", .{ ct, d });
        }
        for ([_][2]usize{ .{ 8, 64 }, .{ 8, 128 }, .{ 12, 128 }, .{ 16, 128 } }) |gd| {
            check.fmt("sdpa_vector_2pass_1_gqa_{d}_{s}_{d}_{d}", .{ gd[0], ct, gd[1], gd[1] });
        }
    }

    // reduce.zig: sum, max, argmax
    for ([_][]const u8{ "sum", "max" }) |op| for (floats) |t| {
        check.fmt("all_reduce_{s}{s}", .{ op, t });
        check.fmt("row_reduce_simple_{s}{s}", .{ op, t });
        check.fmt("row_reduce_small_1_reduce_{s}{s}", .{ op, t });
        check.fmt("row_reduce_looped_1_reduce_{s}{s}", .{ op, t });
        check.fmt("col_reduce_small_1_reduce_{s}{s}", .{ op, t });
        check.fmt("col_reduce_longcolumn_1_reduce_{s}{s}", .{ op, t });
        check.fmt("col_reduce_2pass_1_32_32_reduce_{s}{s}", .{ op, t });
        check.fmt("col_reduce_looped_1_32_32_reduce_{s}{s}", .{ op, t });
    };
    for (argmax_types) |t| check.fmt("argmax_{s}", .{t});

    // scan.zig
    for (sortable) |t| for ([_][]const u8{ "contig", "strided" }) |k| for ([_][]const u8{ "", "reverse_" }) |r| for ([_][]const u8{ "inclusive", "exclusive" }) |inc| {
        check.fmt("{s}_scan_{s}{s}_sum_{s}_{s}", .{ k, r, inc, t, t });
    };

    // sort.zig
    for (sortable) |t| {
        for ([_][]const u8{ "c", "nc" }) |c| for ([_]usize{ 32, 64, 128, 256, 512 }) |bn| {
            check.fmt("{s}_block_sort_{s}_{s}_bn{d}_tn4", .{ c, t, t, bn });
            check.fmt("{s}arg_block_sort_{s}_uint32_bn{d}_tn4", .{ c, t, bn });
        };
        check.fmt("sort_mbsort_{s}_uint32_bn512_tn4", .{t});
        check.fmt("partition_mbsort_{s}_uint32_bn512_tn4", .{t});
        check.fmt("merge_mbsort_{s}_uint32_bn512_tn4", .{t});
        check.fmt("searchsorted_{s}_left", .{t});
        check.fmt("searchsorted_{s}_right", .{t});
    }

    // random.zig
    check.has("rbitsc");

    // ops.zig + chains.zig (derived by construction from metallib_spec.visitOps — never a hand list)
    const spec = @import("metallib_spec.zig");
    try spec.visitOps(&check, struct {
        fn onElem(c: *const Check, _: spec.OpsUnit, name: []const u8, _: spec.ElemOp, _: spec.ElemVariant, _: []const spec.Dtype, _: spec.Dtype) !void {
            c.has(name);
        }
    }.onElem, struct {
        fn onSpecial(c: *const Check, _: spec.OpsUnit, name: []const u8, _: spec.SpecialOp, _: []const spec.Dtype, _: spec.Dtype) !void {
            c.has(name);
        }
    }.onSpecial);

    // Every precompiled custom kernel (shared, EXL3, and all built archs) — by construction from metallib_catalog.
    const catalog = @import("../metallib_catalog.zig");
    try catalog.visitAll(&check, struct {
        fn onName(c: *const Check, name: []const u8) !void {
            c.has(name);
        }
    }.onName);

    try testing.expectEqualStrings("", first_buf[0..first_len]);
    try testing.expectEqual(@as(usize, 0), missing);
    try testing.expect(!have.contains("no_such_sushi_kernel"));
    try testing.expectError(error.MetalCompile, dev.pipelineOf(l, "no_such_sushi_kernel"));

    // An unregistered custom kernel launched on the engine is refused by name, never source-compiled.
    const kernel_mod = @import("kernel.zig");
    const bogus = try kernel_mod.Def.create("no_such_sushi_custom_kernel", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = x[thread_position_in_grid.x];", "", true, false);
    const x = try core.fromHost(std.mem.sliceAsBytes(&[_]f32{ 1, 2, 3, 4 }), .float32, &.{4});
    defer core.release(x);
    var cfg: kernel_mod.Config = .{};
    try cfg.addOutput(&.{4}, .float32);
    cfg.grid = .{ 4, 1, 1 };
    cfg.group = .{ 4, 1, 1 };
    var out: [1]*Tensor = undefined;
    try testing.expectError(error.MetalCompile, kernel_mod.launch(bogus, &.{x}, &cfg, &out));
    try testing.expect(std.mem.indexOf(u8, metal.lastError(), "no_such_sushi_custom_kernel") != null);
}

test "stock metallib: all 57 admitted scalar rates and 21 PAIR1 rates resolve to precompiled FC_NHW functions" {
    try @import("../mlx.zig").requireGpu();
    const dev = try metal.device();
    const l = try lib(dev);
    const names = try l.functionNames(testing.allocator);
    defer {
        for (names) |s| testing.allocator.free(s);
        testing.allocator.free(names);
    }
    var have: std.StringHashMapUnmanaged(void) = .empty;
    defer have.deinit(testing.allocator);
    for (names) |s| try have.put(testing.allocator, s, {});

    const catalog = @import("../metallib_catalog.zig");
    const gemm_ins = [_]catalog.RuntimeInput{
        .{ .name = "x", .dtype = .float16, .scalar = false },
        .{ .name = "trellis", .dtype = .uint16, .scalar = false },
        .{ .name = "eids", .dtype = .uint32, .scalar = false },
        .{ .name = "wstarts", .dtype = .uint32, .scalar = false },
        .{ .name = "wnlive", .dtype = .uint32, .scalar = false },
    };
    const rows_ins = [_]catalog.RuntimeInput{
        .{ .name = "x", .dtype = .bfloat16, .scalar = false },
        .{ .name = "trellis", .dtype = .uint16, .scalar = false },
        .{ .name = "suh", .dtype = .float16, .scalar = false },
        .{ .name = "svh", .dtype = .float16, .scalar = false },
    };
    const pair_ins = [_]catalog.RuntimeInput{
        .{ .name = "x", .dtype = .bfloat16, .scalar = false },
        .{ .name = "suhg", .dtype = .float16, .scalar = false },
        .{ .name = "suhu", .dtype = .float16, .scalar = false },
        .{ .name = "tg", .dtype = .uint16, .scalar = false },
        .{ .name = "tu", .dtype = .uint16, .scalar = false },
        .{ .name = "slots", .dtype = .uint32, .scalar = false },
    };

    const scalar_cases = [_]struct { suf: []const u8, min_n: i32, max_n: i32 }{
        .{ .suf = "", .min_n = 16, .max_n = 128 },
        .{ .suf = "_mcg", .min_n = 16, .max_n = 128 },
        .{ .suf = "_alpha", .min_n = 16, .max_n = 128 },
        .{ .suf = "_spec1_w15_t37f72431", .min_n = 16, .max_n = 128 },
        .{ .suf = "_spec1_t753ffa03", .min_n = 24, .max_n = 64 },
        .{ .suf = "_std1_tc5dba171", .min_n = 24, .max_n = 64 },
    };

    var res: catalog.Resolved = undefined;
    var n: i32 = 16;
    while (n <= 128) : (n += 2) {
        const gemm_targs = [_]catalog.RuntimeTarg{
            .{ .name = "NHW", .value = .{ .int = n } },
            .{ .name = "IDIM", .value = .{ .int = 512 } },
            .{ .name = "ODIM", .value = .{ .int = 512 } },
            .{ .name = "WIN", .value = .{ .int = 32 } },
        };
        const rows_nax_targs = [_]catalog.RuntimeTarg{
            .{ .name = "OutT", .value = .{ .dtype = .bfloat16 } },
            .{ .name = "KPH", .value = .{ .int = 1 } },
            .{ .name = "UNR", .value = .{ .int = 2 } },
            .{ .name = "NHW", .value = .{ .int = n } },
            .{ .name = "IDIM", .value = .{ .int = 512 } },
            .{ .name = "ODIM", .value = .{ .int = 512 } },
            .{ .name = "ROWS", .value = .{ .int = 8 } },
            .{ .name = "GROUPS", .value = .{ .int = 1 } },
            .{ .name = "TGRP", .value = .{ .int = 4 } },
            .{ .name = "NSPLIT", .value = .{ .int = 1 } },
            .{ .name = "CHUNK", .value = .{ .int = 128 } },
            .{ .name = "FOLD", .value = .{ .int = 1 } },
            .{ .name = "R32", .value = .{ .int = 0 } },
            .{ .name = "SW", .value = .{ .int = 0 } },
            .{ .name = "SB1", .value = .{ .int = 512 } },
            .{ .name = "SB2", .value = .{ .int = 512 } },
            .{ .name = "SB3", .value = .{ .int = 512 } },
        };
        const rows_simd_targs = [_]catalog.RuntimeTarg{
            .{ .name = "OutT", .value = .{ .dtype = .bfloat16 } },
            .{ .name = "TGRP", .value = .{ .int = 4 } },
            .{ .name = "KPH", .value = .{ .int = 2 } },
            .{ .name = "NHW", .value = .{ .int = n } },
            .{ .name = "IDIM", .value = .{ .int = 512 } },
            .{ .name = "ODIM", .value = .{ .int = 512 } },
            .{ .name = "ROWS", .value = .{ .int = 4 } },
            .{ .name = "GROUPS", .value = .{ .int = 1 } },
            .{ .name = "NSPLIT", .value = .{ .int = 1 } },
            .{ .name = "CHUNK", .value = .{ .int = 128 } },
            .{ .name = "FOLD", .value = .{ .int = 1 } },
            .{ .name = "SB1", .value = .{ .int = 512 } },
            .{ .name = "SB2", .value = .{ .int = 512 } },
            .{ .name = "SB3", .value = .{ .int = 512 } },
        };
        const pair_targs = [_]catalog.RuntimeTarg{
            .{ .name = "NHW", .value = .{ .int = n } },
            .{ .name = "OTPT", .value = .{ .int = 2 } },
            .{ .name = "IDIM", .value = .{ .int = 512 } },
            .{ .name = "ODIM", .value = .{ .int = 512 } },
            .{ .name = "NSPLIT", .value = .{ .int = 1 } },
            .{ .name = "TOPK", .value = .{ .int = 8 } },
            .{ .name = "FUNNEL", .value = .{ .int = 1 } },
        };

        var nbuf: [128]u8 = undefined;
        for (scalar_cases) |sc| {
            const want = n >= sc.min_n and n <= sc.max_n;

            const g_nax = try std.fmt.bufPrint(&nbuf, "sushi_exl3_k4_gemm_nax{s}", .{sc.suf});
            try testing.expectEqual(want, catalog.resolve(g_nax, &gemm_ins, &.{.float16}, &gemm_targs, &res));
            if (want) try testing.expect(have.contains(res.fnName()));

            const g_smat = try std.fmt.bufPrint(&nbuf, "sushi_exl3_k4_gemm_simdmat{s}", .{sc.suf});
            try testing.expectEqual(want, catalog.resolve(g_smat, &gemm_ins, &.{.float16}, &gemm_targs, &res));
            if (want) try testing.expect(have.contains(res.fnName()));

            const r_nax = try std.fmt.bufPrint(&nbuf, "sushi_exl3_rows_nax{s}", .{sc.suf});
            try testing.expectEqual(want, catalog.resolve(r_nax, &rows_ins, &.{.bfloat16}, &rows_nax_targs, &res));
            if (want) try testing.expect(have.contains(res.fnName()));

            const r_simd = try std.fmt.bufPrint(&nbuf, "sushi_exl3_rows{s}", .{sc.suf});
            try testing.expectEqual(want, catalog.resolve(r_simd, &rows_ins, &.{.bfloat16}, &rows_simd_targs, &res));
            if (want) try testing.expect(have.contains(res.fnName()));

            const p_gemv = try std.fmt.bufPrint(&nbuf, "sushi_exl3_pair_gemv{s}", .{sc.suf});
            try testing.expectEqual(want, catalog.resolve(p_gemv, &pair_ins, &.{ .float32, .float32 }, &pair_targs, &res));
            if (want) try testing.expect(have.contains(res.fnName()));
        }

        // PAIR1 admits 24..64 even only.
        const p1_ok = n >= 24 and n <= 64;
        try testing.expectEqual(p1_ok, catalog.resolve("sushi_exl3_k4_gemm_nax_pair1_w15", &gemm_ins, &.{.float16}, &gemm_targs, &res));
        if (p1_ok) try testing.expect(have.contains(res.fnName()));
        try testing.expectEqual(p1_ok, catalog.resolve("sushi_exl3_pair_gemv_pair1_w15", &pair_ins, &.{ .float32, .float32 }, &pair_targs, &res));
        if (p1_ok) try testing.expect(have.contains(res.fnName()));
    }

    // Out-of-range and odd rates are refused.
    for ([_]i32{ 14, 17, 65, 130 }) |bad_n| {
        const bad_targs = [_]catalog.RuntimeTarg{
            .{ .name = "NHW", .value = .{ .int = bad_n } },
            .{ .name = "IDIM", .value = .{ .int = 512 } },
            .{ .name = "ODIM", .value = .{ .int = 512 } },
            .{ .name = "WIN", .value = .{ .int = 32 } },
        };
        try testing.expect(!catalog.resolve("sushi_exl3_k4_gemm_nax_spec1_w15_t37f72431", &gemm_ins, &.{.float16}, &bad_targs, &res));
    }

    // `_unfolded` test-seam names resolve to the same precompiled functions with FC_B12 = false.
    const unfold_targs = [_]catalog.RuntimeTarg{
        .{ .name = "NHW", .value = .{ .int = 64 } },
        .{ .name = "IDIM", .value = .{ .int = 512 } },
        .{ .name = "ODIM", .value = .{ .int = 512 } },
        .{ .name = "WIN", .value = .{ .int = 32 } },
    };
    for ([_][]const u8{
        "sushi_exl3_k4_gemm_nax_spec1_unfolded_w15_t37f72431",
        "sushi_exl3_k4_gemm_nax_tm_spec1_unfolded_w15_t37f72431",
        "sushi_exl3_k4_gemm_nax_ref_spec1_unfolded_w15_t37f72431",
        "sushi_exl3_k4_gemm_simdmat_spec1_unfolded_w15_t37f72431",
        "sushi_exl3_k4_gemm_nax_pair1_unfolded",
        "sushi_exl3_k4_gemm_nax_tm_pair1_unfolded",
    }) |uname| {
        try testing.expect(catalog.resolve(uname, &gemm_ins, &.{.float16}, &unfold_targs, &res));
        try testing.expect(have.contains(res.fnName()));
        var saw_b12 = false;
        for (res.flagSlice()) |fl| {
            if (fl.index == 60) {
                try testing.expect(!fl.value);
                saw_b12 = true;
            }
        }
        try testing.expect(saw_b12);
    }

    // Specialize live PSOs across one rate per structural leaf arm (scalar & PAIR1) and verify threadgroup headroom.
    const down_ins = [_]catalog.RuntimeInput{
        .{ .name = "ig", .dtype = .float32, .scalar = false },
        .{ .name = "iu", .dtype = .float32, .scalar = false },
        .{ .name = "trellis", .dtype = .uint16, .scalar = false },
        .{ .name = "svhg", .dtype = .float16, .scalar = false },
        .{ .name = "svhu", .dtype = .float16, .scalar = false },
        .{ .name = "suhd", .dtype = .float16, .scalar = false },
        .{ .name = "slots", .dtype = .uint32, .scalar = false },
    };
    const has_nax = @import("sushi_exl3").device.current().has(.matrix_g17);
    const coop_ins = [_]catalog.RuntimeInput{
        .{ .name = "x", .dtype = .float16, .scalar = false },
        .{ .name = "trellis", .dtype = .uint16, .scalar = false },
        .{ .name = "slots", .dtype = .uint32, .scalar = false },
    };
    // One rate per distinct smat_group_fc leaf arm (7 scalar + 3 PAIR1).
    for ([_]i32{ 16, 38, 58, 64, 86, 110, 126 }) |n_smat| {
        const smat_targs = [_]catalog.RuntimeTarg{
            .{ .name = "NHW", .value = .{ .int = n_smat } },
            .{ .name = "IDIM", .value = .{ .int = 512 } },
            .{ .name = "ODIM", .value = .{ .int = 512 } },
            .{ .name = "WIN", .value = .{ .int = 32 } },
        };
        try testing.expect(catalog.resolve("sushi_exl3_k4_gemm_simdmat_spec1_w15_t37f72431", &gemm_ins, &.{.float16}, &smat_targs, &res));
        try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 256);
    }
    // One rate per distinct exl3_lane_fc arm + exl3_lane_word_fc shift bucket (6 scalar).
    for ([_]i32{ 16, 38, 42, 80, 110, 126 }) |n_lane| {
        const rsimd_targs = [_]catalog.RuntimeTarg{
            .{ .name = "OutT", .value = .{ .dtype = .bfloat16 } },
            .{ .name = "TGRP", .value = .{ .int = 4 } },
            .{ .name = "KPH", .value = .{ .int = 2 } },
            .{ .name = "NHW", .value = .{ .int = n_lane } },
            .{ .name = "IDIM", .value = .{ .int = 512 } },
            .{ .name = "ODIM", .value = .{ .int = 512 } },
            .{ .name = "ROWS", .value = .{ .int = 4 } },
            .{ .name = "GROUPS", .value = .{ .int = 1 } },
            .{ .name = "NSPLIT", .value = .{ .int = 1 } },
            .{ .name = "CHUNK", .value = .{ .int = 128 } },
            .{ .name = "FOLD", .value = .{ .int = 1 } },
            .{ .name = "SB1", .value = .{ .int = 512 } },
            .{ .name = "SB2", .value = .{ .int = 512 } },
            .{ .name = "SB3", .value = .{ .int = 512 } },
        };
        try testing.expect(catalog.resolve("sushi_exl3_rows_spec1_w15_t37f72431", &rows_ins, &.{.bfloat16}, &rsimd_targs, &res));
        try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 256);

        const p_targs = [_]catalog.RuntimeTarg{
            .{ .name = "NHW", .value = .{ .int = n_lane } },
            .{ .name = "OTPT", .value = .{ .int = 2 } },
            .{ .name = "IDIM", .value = .{ .int = 512 } },
            .{ .name = "ODIM", .value = .{ .int = 512 } },
            .{ .name = "NSPLIT", .value = .{ .int = 1 } },
            .{ .name = "TOPK", .value = .{ .int = 8 } },
            .{ .name = "FUNNEL", .value = .{ .int = 1 } },
        };
        try testing.expect(catalog.resolve("sushi_exl3_pair_gemv_spec1_w15_t37f72431", &pair_ins, &.{ .float32, .float32 }, &p_targs, &res));
        try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 256);

        const d_targs = [_]catalog.RuntimeTarg{
            .{ .name = "NHW", .value = .{ .int = n_lane } },
            .{ .name = "OTPT", .value = .{ .int = 2 } },
            .{ .name = "IDIM", .value = .{ .int = 512 } },
            .{ .name = "ODIM", .value = .{ .int = 512 } },
            .{ .name = "NSPLIT", .value = .{ .int = 1 } },
            .{ .name = "FUNNEL", .value = .{ .int = 1 } },
        };
        try testing.expect(catalog.resolve("sushi_exl3_k4_down_fused_spec1_w15_t37f72431", &down_ins, &.{.float16}, &d_targs, &res));
        try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 256);

        const coop_targs = [_]catalog.RuntimeTarg{
            .{ .name = "NHW", .value = .{ .int = n_lane } },
            .{ .name = "TILES", .value = .{ .int = 1 } },
            .{ .name = "IDIM", .value = .{ .int = 512 } },
            .{ .name = "ODIM", .value = .{ .int = 512 } },
            .{ .name = "NSLOTS", .value = .{ .int = 16 } },
            .{ .name = "SERIAL_REDUCTION", .value = .{ .int = 1 } },
        };
        try testing.expect(catalog.resolve("sushi_exl3_group2_spec1_w15_t37f72431", &coop_ins, &.{.float16}, &coop_targs, &res));
        try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 256);
    }
    // One rate per distinct nax_wfrag_fc leaf arm (9 scalar + 5 PAIR1).
    if (has_nax) {
        for ([_]i32{ 16, 32, 60, 64, 66, 80, 96, 110, 128 }) |n_nax| {
            const nax_targs = [_]catalog.RuntimeTarg{
                .{ .name = "NHW", .value = .{ .int = n_nax } },
                .{ .name = "IDIM", .value = .{ .int = 512 } },
                .{ .name = "ODIM", .value = .{ .int = 512 } },
                .{ .name = "WIN", .value = .{ .int = 32 } },
            };
            try testing.expect(catalog.resolve("sushi_exl3_k4_gemm_nax_spec1_w15_t37f72431", &gemm_ins, &.{.float16}, &nax_targs, &res));
            try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 256);
        }
    }
    for ([_]i32{ 24, 56, 64 }) |n_p1| {
        const smat_targs = [_]catalog.RuntimeTarg{
            .{ .name = "NHW", .value = .{ .int = n_p1 } },
            .{ .name = "IDIM", .value = .{ .int = 512 } },
            .{ .name = "ODIM", .value = .{ .int = 512 } },
            .{ .name = "WIN", .value = .{ .int = 32 } },
        };
        try testing.expect(catalog.resolve("sushi_exl3_k4_gemm_simdmat_pair1", &gemm_ins, &.{.float16}, &smat_targs, &res));
        try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 640);
    }
    if (has_nax) {
        for ([_]i32{ 24, 32, 36, 56, 64 }) |n_p1| {
            const nax_targs = [_]catalog.RuntimeTarg{
                .{ .name = "NHW", .value = .{ .int = n_p1 } },
                .{ .name = "IDIM", .value = .{ .int = 512 } },
                .{ .name = "ODIM", .value = .{ .int = 512 } },
                .{ .name = "WIN", .value = .{ .int = 32 } },
            };
            try testing.expect(catalog.resolve("sushi_exl3_k4_gemm_nax_pair1", &gemm_ins, &.{.float16}, &nax_targs, &res));
            try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 640);
        }
    }
}

test "stock metallib: precompiled FC_NHW readers match host bit-for-bit across scalar and PAIR1 leaf arms" {
    try @import("../mlx.zig").requireGpu();
    const kernel_mod = @import("kernel.zig");
    const exl3_mod = @import("sushi_exl3");
    const format = exl3_mod.format;
    const has_nax = @import("sushi_exl3").device.current().has(.matrix_g17);

    // 16x16 identity in f16 for GEMM (y = I * W = W bit-for-bit) and lane-ordered [16, 128] identity prefix for down_prepared.
    var eye16: [16 * 16]u16 = @splat(0);
    var eye128: [16 * 128]u16 = @splat(0);
    for (0..16) |r| {
        eye16[r * 16 + r] = 0x3C00; // 1.0h
        const half_i = r >> 3;
        const q = (r & 7) >> 1;
        const p = r & 1;
        eye128[r * 128 + q * 4 + half_i * 2 + p] = 0x3C00; // 1.0h in LANE_ORDER_SLOT
    }
    const x_eye16 = try core.fromHost(std.mem.sliceAsBytes(&eye16), .float16, &.{ 16, 16 });
    defer core.release(x_eye16);
    const x_eye128 = try core.fromHost(std.mem.sliceAsBytes(&eye128), .float16, &.{ 16, 128 });
    defer core.release(x_eye128);
    const zeros16: [16]u32 = @splat(0);
    const eids_t = try core.fromHost(std.mem.sliceAsBytes(&zeros16), .uint32, &.{16});
    defer core.release(eids_t);
    const wstarts_t = try core.fromHost(std.mem.sliceAsBytes(&[_]u32{0}), .uint32, &.{1});
    defer core.release(wstarts_t);
    const wnlive_t = try core.fromHost(std.mem.sliceAsBytes(&[_]u32{16}), .uint32, &.{1});
    defer core.release(wnlive_t);

    const gemm_in_names = &[_][]const u8{ "x", "trellis", "eids", "wstarts", "wnlive" };
    const dp_in_names = &[_][]const u8{ "middle", "trellis", "slots" };
    const dense_in_names = &[_][]const u8{"trellis"};
    const rows_in_names = &[_][]const u8{ "x", "trellis", "suh", "svh" };

    const Case = struct {
        suf: []const u8,
        unfolded_suf: ?[]const u8,
        dec: format.Decode,
        rates: []const u32,
        min_n: u32,
        max_n: u32,
    };
    const cases = [_]Case{
        // Every structural leaf arm of exl3_lane_fc, exl3_lane_word_fc, smat_group_fc, and nax_wfrag_fc.
        .{ .suf = "_spec1_w15_t37f72431", .unfolded_suf = "_spec1_unfolded_w15_t37f72431", .dec = format.Decode.tabled(.spec1, .w15), .rates = &.{ 16, 32, 38, 42, 58, 60, 64, 80, 86, 96, 110, 126, 128 }, .min_n = 16, .max_n = 128 },
        // Representative w16 (unmasked 16-bit) rates across narrow, funnel, K4, and wide tiers.
        .{ .suf = "_mcg", .unfolded_suf = null, .dec = .{ .codebook = .mcg, .window = .w16 }, .rates = &.{ 16, 42, 64, 128 }, .min_n = 16, .max_n = 128 },
        // Every PAIR1 leaf-arm boundary across exl3_lane_fc (incl. simd_shuffle at n=56), smat_group_fc, and nax_wfrag_fc.
        .{ .suf = "_pair1", .unfolded_suf = "_pair1_unfolded", .dec = .{ .codebook = .pair1, .window = .w16 }, .rates = &.{ 24, 32, 36, 38, 42, 44, 56, 64 }, .min_n = 24, .max_n = 64 },
    };

    var prng = std.Random.DefaultPrng.init(0x5EC1_FC01);
    const rnd = prng.random();
    var trellis_buf: [8 * 128]u16 = undefined;
    var want_tiles: [8][format.TILE_VALUES]u16 = undefined;
    var name_buf: [128]u8 = undefined;

    for (cases) |cs| {
        const smat_def = try kernel_mod.Def.create(try std.fmt.bufPrint(&name_buf, "sushi_exl3_k4_gemm_simdmat{s}", .{cs.suf}), gemm_in_names, &.{"y"}, exl3_mod.kernels.GEMM_SIMDMAT_SOURCE, "", true, false);
        const dp_def = try kernel_mod.Def.create(try std.fmt.bufPrint(&name_buf, "sushi_exl3_down_prepared{s}", .{cs.suf}), dp_in_names, &.{"y"}, exl3_mod.kernels.DOWN_PREPARED_SOURCE, "", true, false);
        const nax_def = if (has_nax) try kernel_mod.Def.create(try std.fmt.bufPrint(&name_buf, "sushi_exl3_k4_gemm_nax{s}", .{cs.suf}), gemm_in_names, &.{"y"}, exl3_mod.kernels.GEMM_NAX_SOURCE, "", true, false) else null;
        const dense_def = if (has_nax) try kernel_mod.Def.create(try std.fmt.bufPrint(&name_buf, "sushi_exl3_trunk_dense_nax{s}", .{cs.suf}), dense_in_names, &.{"y"}, exl3_mod.trunk_gemm.DENSE_SOURCE, "", true, false) else null;
        const nax_u_def = if (has_nax and cs.unfolded_suf != null) try kernel_mod.Def.create(try std.fmt.bufPrint(&name_buf, "sushi_exl3_k4_gemm_nax{s}", .{cs.unfolded_suf.?}), gemm_in_names, &.{"y"}, exl3_mod.kernels.GEMM_NAX_SOURCE, "", true, false) else null;

        for (cs.rates) |n| {
            const rate: format.Rate = .{ .n = n };
            const n_c: c_int = @intCast(n);
            const slice = trellis_buf[0 .. 8 * n];
            for (slice) |*w| w.* = rnd.int(u16);
            for (0..8) |t_idx| format.decodeTile(slice[t_idx * n .. (t_idx + 1) * n], rate, cs.dec, &want_tiles[t_idx]);

            const tr_gemm = try core.fromHost(std.mem.sliceAsBytes(slice), .uint16, &.{ 1, 1, 8, n_c });
            defer core.release(tr_gemm);
            const tr_dp = try core.fromHost(std.mem.sliceAsBytes(slice), .uint16, &.{ 1, 8, 1, n_c });
            defer core.release(tr_dp);

            // 1. smat_group_fc (sushi_exl3_k4_gemm_simdmat): folded MMA + exl3_out matches decodeTile bit-for-bit.
            {
                var cfg: kernel_mod.Config = .{ .grid = .{ 128, 1, 1 }, .group = .{ 128, 1, 1 } };
                try cfg.addOutput(&.{ 16, 128 }, .float16);
                try cfg.addTemplate("NHW", .{ .int = n_c });
                try cfg.addTemplate("IDIM", .{ .int = 16 });
                try cfg.addTemplate("ODIM", .{ .int = 128 });
                try cfg.addTemplate("WIN", .{ .int = 16 });
                var out: [1]*Tensor = undefined;
                try kernel_mod.launch(smat_def, &.{ x_eye16, tr_gemm, eids_t, wstarts_t, wnlive_t }, &cfg, &out);
                defer core.release(out[0]);
                const got = try readU16(out[0]);
                for (0..16) |r| {
                    for (0..8) |t_idx| {
                        for (0..16) |c| {
                            try testing.expectEqual(canonF16Zero(want_tiles[t_idx][r * 16 + c]), got[r * 128 + t_idx * 16 + c]);
                        }
                    }
                }
            }

            // 2. exl3_lane_fc + exl3_lane_word_fc (sushi_exl3_down_prepared): matches decodeTile bit-for-bit
            //    at every rate including n=64 and PAIR1 n=56 (simd_shuffle).
            {
                var cfg: kernel_mod.Config = .{ .grid = .{ 128, 16, 1 }, .group = .{ 128, 1, 1 } };
                try cfg.addOutput(&.{ 16, 16 }, .float16);
                try cfg.addTemplate("NHW", .{ .int = n_c });
                try cfg.addTemplate("OTPT", .{ .int = 1 });
                try cfg.addTemplate("IDIM", .{ .int = 128 });
                try cfg.addTemplate("ODIM", .{ .int = 16 });
                try cfg.addTemplate("FUNNEL", .{ .int = 1 });
                var out: [1]*Tensor = undefined;
                try kernel_mod.launch(dp_def, &.{ x_eye128, tr_dp, eids_t }, &cfg, &out);
                defer core.release(out[0]);
                const got = try readU16(out[0]);
                try expectMatmulU16(&want_tiles[0], got);
            }

            if (has_nax) {
                // 3. nax_wfrag_fc<true> (sushi_exl3_trunk_dense_nax): scaled dense decode matches decodeTile bit-for-bit.
                {
                    const tr_dense = try core.fromHost(std.mem.sliceAsBytes(slice[0 .. 2 * n]), .uint16, &.{ 1, 1, 2, n_c });
                    defer core.release(tr_dense);
                    var cfg: kernel_mod.Config = .{ .grid = .{ 128, 1, 1 }, .group = .{ 128, 1, 1 } };
                    try cfg.addOutput(&.{ 32, 16 }, .float16);
                    try cfg.addTemplate("NHW", .{ .int = n_c });
                    try cfg.addTemplate("IDIM", .{ .int = 16 });
                    try cfg.addTemplate("ODIM", .{ .int = 32 });
                    var out: [1]*Tensor = undefined;
                    try kernel_mod.launch(dense_def.?, &.{tr_dense}, &cfg, &out);
                    defer core.release(out[0]);
                    const got = try readU16(out[0]);
                    for (0..2) |t_idx| {
                        for (0..16) |r| {
                            for (0..16) |c| {
                                try testing.expectEqual(want_tiles[t_idx][r * 16 + c], got[(t_idx * 16 + c) * 16 + r]);
                            }
                        }
                    }
                }

                // 4. nax_wfrag_fc<false> (sushi_exl3_k4_gemm_nax): folded MMA + exl3_out matches decodeTile bit-for-bit.
                {
                    var cfg: kernel_mod.Config = .{ .grid = .{ 128, 1, 1 }, .group = .{ 128, 1, 1 } };
                    try cfg.addOutput(&.{ 16, 128 }, .float16);
                    try cfg.addTemplate("NHW", .{ .int = n_c });
                    try cfg.addTemplate("IDIM", .{ .int = 16 });
                    try cfg.addTemplate("ODIM", .{ .int = 128 });
                    try cfg.addTemplate("WIN", .{ .int = 16 });
                    var out: [1]*Tensor = undefined;
                    try kernel_mod.launch(nax_def.?, &.{ x_eye16, tr_gemm, eids_t, wstarts_t, wnlive_t }, &cfg, &out);
                    defer core.release(out[0]);
                    const got = try readU16(out[0]);
                    for (0..16) |r| {
                        for (0..8) |t_idx| {
                            for (0..16) |c| {
                                try testing.expectEqual(canonF16Zero(want_tiles[t_idx][r * 16 + c]), got[r * 128 + t_idx * 16 + c]);
                            }
                        }
                    }
                    // Also verify _unfolded (FC_B12 = false) on boundary rates matches bit-for-bit.
                    if (nax_u_def != null and (n == cs.min_n or n == 32 or n == 56 or n == 64 or n == 80 or n == 96 or n == cs.max_n)) {
                        var out_u: [1]*Tensor = undefined;
                        try kernel_mod.launch(nax_u_def.?, &.{ x_eye16, tr_gemm, eids_t, wstarts_t, wnlive_t }, &cfg, &out_u);
                        defer core.release(out_u[0]);
                        const got_u = try readU16(out_u[0]);
                        try testing.expectEqualSlices(u16, got, got_u);
                    }
                }
            }
        }

        // 5. rows_wfrag_fc (sushi_exl3_rows_nax, FOLD=1): folded vs unfolded bit-exact on boundary rates.
        if (has_nax and cs.unfolded_suf != null) {
            const rn_f = try kernel_mod.Def.create(try std.fmt.bufPrint(&name_buf, "sushi_exl3_rows_nax{s}", .{cs.suf}), rows_in_names, &.{"y"}, exl3_mod.project_rows.ROWS_NAX_SOURCE, "", true, false);
            const rn_u = try kernel_mod.Def.create(try std.fmt.bufPrint(&name_buf, "sushi_exl3_rows_nax{s}", .{cs.unfolded_suf.?}), rows_in_names, &.{"y"}, exl3_mod.project_rows.ROWS_NAX_SOURCE, "", true, false);
            var rx_data: [128]u16 = undefined;
            const ones128: [128]u16 = @splat(0x3C00);
            for (&rx_data, 0..) |*v, i| v.* = if ((i & 1) == 0) 0x3F80 else 0xBF00; // bf16 +1.0 / -0.5
            const rx = try core.fromHost(std.mem.sliceAsBytes(&rx_data), .bfloat16, &.{ 1, 128 });
            defer core.release(rx);
            const suh_t = try core.fromHost(std.mem.sliceAsBytes(&ones128), .float16, &.{128});
            defer core.release(suh_t);
            const svh_t = try core.fromHost(std.mem.sliceAsBytes(&ones128), .float16, &.{128});
            defer core.release(svh_t);
            var r_tr: [64 * 128]u16 = undefined;
            for ([_]u32{ 32, 56, 80, 128 }) |rn| {
                if (rn < cs.min_n or rn > cs.max_n) continue;
                const rslice = r_tr[0 .. 64 * rn];
                for (rslice) |*w| w.* = rnd.int(u16);
                const rtr_t = try core.fromHost(std.mem.sliceAsBytes(rslice), .uint16, &.{ 8, 8, @intCast(rn) });
                defer core.release(rtr_t);
                var cfg: kernel_mod.Config = .{ .grid = .{ 128, 1, 1 }, .group = .{ 128, 1, 1 } };
                try cfg.addOutput(&.{ 1, 128 }, .bfloat16);
                try cfg.addTemplate("OutT", .{ .dtype = .bfloat16 });
                try cfg.addTemplate("KPH", .{ .int = 1 });
                try cfg.addTemplate("UNR", .{ .int = 2 });
                try cfg.addTemplate("NHW", .{ .int = @intCast(rn) });
                try cfg.addTemplate("IDIM", .{ .int = 128 });
                try cfg.addTemplate("ODIM", .{ .int = 128 });
                try cfg.addTemplate("ROWS", .{ .int = 1 });
                try cfg.addTemplate("GROUPS", .{ .int = 1 });
                try cfg.addTemplate("TGRP", .{ .int = 4 });
                try cfg.addTemplate("NSPLIT", .{ .int = 1 });
                try cfg.addTemplate("CHUNK", .{ .int = 128 });
                try cfg.addTemplate("FOLD", .{ .int = 1 });
                try cfg.addTemplate("R32", .{ .int = 0 });
                try cfg.addTemplate("SW", .{ .int = 0 });
                try cfg.addTemplate("SB1", .{ .int = 128 });
                try cfg.addTemplate("SB2", .{ .int = 128 });
                try cfg.addTemplate("SB3", .{ .int = 128 });
                var out_f: [1]*Tensor = undefined;
                var out_u: [1]*Tensor = undefined;
                try kernel_mod.launch(rn_f, &.{ rx, rtr_t, suh_t, svh_t }, &cfg, &out_f);
                defer core.release(out_f[0]);
                try kernel_mod.launch(rn_u, &.{ rx, rtr_t, suh_t, svh_t }, &cfg, &out_u);
                defer core.release(out_u[0]);
                const gf = try readU16(out_f[0]);
                const gu = try readU16(out_u[0]);
                try testing.expectEqualSlices(u16, gu, gf);
            }
        }
    }
}

test "stock metallib: shared SIMD FC specs resolve valid parameters, reject OOB, and hold PSO occupancy" {
    try @import("../mlx.zig").requireGpu();
    const dev = try metal.device();
    const l = try lib(dev);
    const catalog = @import("../metallib_catalog.zig");
    const has_nax = @import("sushi_exl3").device.current().has(.matrix_g17);
    const min_threads: usize = if (has_nax) 1024 else 640;

    const checkPso = struct {
        fn run(d: *const metal.Device, lib_h: metal.Library, r: *const catalog.Resolved, floor: usize) !void {
            try testing.expect((try specializeMaxThreads(d, lib_h, r)) >= floor);
        }
    }.run;

    var res: catalog.Resolved = undefined;

    // 1. dsv41_hc_mix & hc_ident_mix_expand: valid R/M/HC/TG + OOB rejection.
    const hc_mix_ins = [_]catalog.RuntimeInput{
        .{ .name = "x", .dtype = .bfloat16 },
        .{ .name = "fn", .dtype = .float32 },
        .{ .name = "scale", .dtype = .float32 },
        .{ .name = "base", .dtype = .float32 },
        .{ .name = "eps", .dtype = .float32 },
    };
    const hc_mix_ok = [_]catalog.RuntimeTarg{
        .{ .name = "TG", .value = .{ .int = 1024 } },
        .{ .name = "R", .value = .{ .int = 2 } },
        .{ .name = "M", .value = .{ .int = 12 } },
        .{ .name = "HC", .value = .{ .int = 4 } },
    };
    try testing.expect(catalog.resolve("dsv41_hc_mix", &hc_mix_ins, &.{.float32}, &hc_mix_ok, &res));
    try checkPso(dev, l, &res, min_threads);
    const hc_ident_ins = [_]catalog.RuntimeInput{
        .{ .name = "h", .dtype = .float32 },
        .{ .name = "y", .dtype = .bfloat16 },
        .{ .name = "post", .dtype = .float32 },
        .{ .name = "fn", .dtype = .float32 },
        .{ .name = "scale", .dtype = .float32 },
        .{ .name = "base", .dtype = .float32 },
        .{ .name = "eps", .dtype = .float32 },
    };
    try testing.expect(catalog.resolve("hc_ident_mix_expand", &hc_ident_ins, &.{ .float32, .float32 }, &hc_mix_ok, &res));
    try checkPso(dev, l, &res, min_threads);
    for ([_][4]i32{ .{ 1024, 5, 12, 4 }, .{ 1024, 2, 16, 4 }, .{ 1024, 2, 12, 9 }, .{ 2048, 2, 12, 4 }, .{ 32, 4, 12, 4 } }) |bad| {
        const targs = [_]catalog.RuntimeTarg{
            .{ .name = "TG", .value = .{ .int = bad[0] } },
            .{ .name = "R", .value = .{ .int = bad[1] } },
            .{ .name = "M", .value = .{ .int = bad[2] } },
            .{ .name = "HC", .value = .{ .int = bad[3] } },
        };
        try testing.expect(!catalog.resolve("dsv41_hc_mix", &hc_mix_ins, &.{.float32}, &targs, &res));
        try testing.expect(!catalog.resolve("hc_ident_mix_expand", &hc_ident_ins, &.{ .float32, .float32 }, &targs, &res));
    }

    // 2. dsv41_hc_stage{,_z}: valid HC/ND/TG/ITERS + OOB rejection.
    const hc_st_ins = [_]catalog.RuntimeInput{
        .{ .name = "x", .dtype = .bfloat16 },
        .{ .name = "cpre", .dtype = .float32 },
        .{ .name = "w", .dtype = .bfloat16 },
        .{ .name = "fn", .dtype = .float32 },
        .{ .name = "scale", .dtype = .float32 },
        .{ .name = "base", .dtype = .float32 },
        .{ .name = "eps", .dtype = .float32 },
    };
    const hc_st_ok = [_]catalog.RuntimeTarg{
        .{ .name = "TG", .value = .{ .int = 1024 } },
        .{ .name = "HC", .value = .{ .int = 4 } },
        .{ .name = "ND", .value = .{ .int = 5 } },
        .{ .name = "T", .value = .{ .dtype = .bfloat16 } },
    };
    try testing.expect(catalog.resolve("dsv41_hc_stage", &hc_st_ins, &.{ .bfloat16, .float32 }, &hc_st_ok, &res));
    try checkPso(dev, l, &res, min_threads);
    const hc_stz_ins = [_]catalog.RuntimeInput{
        .{ .name = "x", .dtype = .bfloat16 },
        .{ .name = "y", .dtype = .bfloat16 },
        .{ .name = "za", .dtype = .float32 },
        .{ .name = "w", .dtype = .bfloat16 },
        .{ .name = "fn", .dtype = .float32 },
        .{ .name = "scale", .dtype = .float32 },
        .{ .name = "base", .dtype = .float32 },
        .{ .name = "eps", .dtype = .float32 },
    };
    const hc_stz_ok = hc_st_ok ++ [_]catalog.RuntimeTarg{.{ .name = "ITERS", .value = .{ .int = 20 } }};
    try testing.expect(catalog.resolve("dsv41_hc_stage_z", &hc_stz_ins, &.{ .bfloat16, .float32, .bfloat16 }, &hc_stz_ok, &res));
    try checkPso(dev, l, &res, min_threads);
    var hc_stz_bad = hc_stz_ok;
    hc_stz_bad[4] = .{ .name = "ITERS", .value = .{ .int = 0 } };
    try testing.expect(!catalog.resolve("dsv41_hc_stage_z", &hc_stz_ins, &.{ .bfloat16, .float32, .bfloat16 }, &hc_stz_bad, &res));
    const hc_st_narrow = [_]catalog.RuntimeTarg{
        .{ .name = "TG", .value = .{ .int = 256 } },
        .{ .name = "HC", .value = .{ .int = 4 } },
        .{ .name = "ND", .value = .{ .int = 20 } },
        .{ .name = "T", .value = .{ .dtype = .bfloat16 } },
    };
    try testing.expect(catalog.resolve("dsv41_hc_stage", &hc_st_ins, &.{ .bfloat16, .float32 }, &hc_st_narrow, &res));
    try checkPso(dev, l, &res, 256);
    for ([_][3]i32{ .{ 1024, 8, 5 }, .{ 1024, 4, 33 }, .{ 2048, 4, 5 }, .{ 1024, 4, 0 } }) |bad| {
        const targs = [_]catalog.RuntimeTarg{
            .{ .name = "TG", .value = .{ .int = bad[0] } },
            .{ .name = "HC", .value = .{ .int = bad[1] } },
            .{ .name = "ND", .value = .{ .int = bad[2] } },
            .{ .name = "T", .value = .{ .dtype = .bfloat16 } },
        };
        try testing.expect(!catalog.resolve("dsv41_hc_stage", &hc_st_ins, &.{ .bfloat16, .float32 }, &targs, &res));
    }

    // 3. dsv41_index_scores & dsv41_index_tile: valid QS_CAP/IH/IHD + OOB rejection.
    const idx_ins = [_]catalog.RuntimeInput{
        .{ .name = "q", .dtype = .float32 },
        .{ .name = "keys", .dtype = .bfloat16 },
        .{ .name = "w", .dtype = .float32 },
        .{ .name = "meta", .dtype = .int32 },
        .{ .name = "wsc", .dtype = .float32 },
    };
    const idx_ok = [_]catalog.RuntimeTarg{
        .{ .name = "QS_CAP", .value = .{ .int = 4096 } },
        .{ .name = "IH", .value = .{ .int = 32 } },
        .{ .name = "IHD", .value = .{ .int = 128 } },
        .{ .name = "KT", .value = .{ .dtype = .bfloat16 } },
    };
    try testing.expect(catalog.resolve("dsv41_index_scores", &idx_ins, &.{.float32}, &idx_ok, &res));
    try checkPso(dev, l, &res, min_threads);
    for ([_][3]i32{ .{ 2048, 32, 128 }, .{ 8192, 65, 128 }, .{ 8192, 32, 136 } }) |bad| {
        const targs = [_]catalog.RuntimeTarg{
            .{ .name = "QS_CAP", .value = .{ .int = bad[0] } },
            .{ .name = "IH", .value = .{ .int = bad[1] } },
            .{ .name = "IHD", .value = .{ .int = bad[2] } },
            .{ .name = "KT", .value = .{ .dtype = .bfloat16 } },
        };
        try testing.expect(!catalog.resolve("dsv41_index_scores", &idx_ins, &.{.float32}, &targs, &res));
    }
    const tile_ins = [_]catalog.RuntimeInput{
        .{ .name = "q", .dtype = .bfloat16 },
        .{ .name = "keys", .dtype = .bfloat16 },
        .{ .name = "w", .dtype = .float32 },
        .{ .name = "meta", .dtype = .int32 },
        .{ .name = "wsc", .dtype = .float32 },
    };
    const tile_ok = [_]catalog.RuntimeTarg{
        .{ .name = "IH", .value = .{ .int = 32 } },
        .{ .name = "IHD", .value = .{ .int = 128 } },
        .{ .name = "KT", .value = .{ .int = 64 } },
        .{ .name = "T", .value = .{ .dtype = .bfloat16 } },
    };
    try testing.expect(catalog.resolve("dsv41_index_tile", &tile_ins, &.{.float32}, &tile_ok, &res));
    try checkPso(dev, l, &res, min_threads);
    for ([_][2]i32{ .{ 72, 128 }, .{ 12, 128 }, .{ 32, 136 } }) |bad| {
        const targs = [_]catalog.RuntimeTarg{
            .{ .name = "IH", .value = .{ .int = bad[0] } },
            .{ .name = "IHD", .value = .{ .int = bad[1] } },
            .{ .name = "KT", .value = .{ .int = 64 } },
            .{ .name = "T", .value = .{ .dtype = .bfloat16 } },
        };
        try testing.expect(!catalog.resolve("dsv41_index_tile", &tile_ins, &.{.float32}, &targs, &res));
    }

    // 4. gated_delta_step{,_vec}: valid Dk + KDA (InT=f32, BT=bf16) + OOB rejection.
    const gdn_ins = [_]catalog.RuntimeInput{
        .{ .name = "q", .dtype = .float32 },
        .{ .name = "k", .dtype = .float32 },
        .{ .name = "v", .dtype = .float32 },
        .{ .name = "g", .dtype = .float32 },
        .{ .name = "beta", .dtype = .bfloat16 },
        .{ .name = "state_in", .dtype = .float32 },
        .{ .name = "T", .dtype = .int32, .scalar = true },
    };
    const gdn_ok = [_]catalog.RuntimeTarg{
        .{ .name = "InT", .value = .{ .dtype = .float32 } },
        .{ .name = "StT", .value = .{ .dtype = .float32 } },
        .{ .name = "OutT", .value = .{ .dtype = .float32 } },
        .{ .name = "Dk", .value = .{ .int = 128 } },
        .{ .name = "Dv", .value = .{ .int = 128 } },
        .{ .name = "Hk", .value = .{ .int = 16 } },
        .{ .name = "Hv", .value = .{ .int = 48 } },
    };
    try testing.expect(catalog.resolve("gated_delta_step_vec", &gdn_ins, &.{ .float32, .float32 }, &gdn_ok, &res));
    try checkPso(dev, l, &res, min_threads);
    const gdn_seq_ins = gdn_ins ++ [_]catalog.RuntimeInput{
        .{ .name = "seq_stride", .dtype = .int32, .scalar = true },
    };
    var gdn_seq_ok = gdn_ok;
    gdn_seq_ok[2] = .{ .name = "OutT", .value = .{ .dtype = .bfloat16 } };
    try testing.expect(catalog.resolve("gated_delta_step_seq_vec", &gdn_seq_ins, &.{ .bfloat16, .float32, .float32 }, &gdn_seq_ok, &res));
    try checkPso(dev, l, &res, min_threads);
    for ([_][3]i32{ .{ 288, 16, 48 }, .{ 48, 16, 48 }, .{ 128, 16, 50 } }) |bad| {
        const targs = [_]catalog.RuntimeTarg{
            .{ .name = "InT", .value = .{ .dtype = .float32 } },
            .{ .name = "StT", .value = .{ .dtype = .float32 } },
            .{ .name = "OutT", .value = .{ .dtype = .float32 } },
            .{ .name = "Dk", .value = .{ .int = bad[0] } },
            .{ .name = "Dv", .value = .{ .int = 128 } },
            .{ .name = "Hk", .value = .{ .int = bad[1] } },
            .{ .name = "Hv", .value = .{ .int = bad[2] } },
        };
        try testing.expect(!catalog.resolve("gated_delta_step_vec", &gdn_ins, &.{ .float32, .float32 }, &targs, &res));
    }

    // 5. sushi_attn_qsa256_qsplit: valid BITS ∈ {2,3,4,6,8}, BK ∈ {16,32}, DK/DV + OOB rejection.
    const qsplit_ins = [_]catalog.RuntimeInput{
        .{ .name = "q", .dtype = .bfloat16 },
        .{ .name = "scl", .dtype = .float32 },
        .{ .name = "blocks", .dtype = .int32 },
        .{ .name = "kq", .dtype = .uint32 },
        .{ .name = "ksc", .dtype = .bfloat16 },
        .{ .name = "kbi", .dtype = .bfloat16 },
        .{ .name = "vq", .dtype = .uint32 },
        .{ .name = "vsc", .dtype = .bfloat16 },
        .{ .name = "vbi", .dtype = .bfloat16 },
    };
    const qsplit_ok = [_]catalog.RuntimeTarg{
        .{ .name = "T", .value = .{ .dtype = .bfloat16 } },
        .{ .name = "NSG", .value = .{ .int = 2 } },
        .{ .name = "NSGT", .value = .{ .int = 2 } },
        .{ .name = "BK", .value = .{ .int = 16 } },
        .{ .name = "RATIO", .value = .{ .int = 4 } },
        .{ .name = "BITS", .value = .{ .int = 8 } },
        .{ .name = "GS", .value = .{ .int = 64 } },
        .{ .name = "QUANT", .value = .{ .int = 1 } },
        .{ .name = "DK", .value = .{ .int = 256 } },
        .{ .name = "DV", .value = .{ .int = 256 } },
        .{ .name = "SPARSE", .value = .{ .int = 1 } },
    };
    for ([_]i32{ 2, 3, 4, 6, 8 }) |bits| {
        var t = qsplit_ok;
        t[5] = .{ .name = "BITS", .value = .{ .int = bits } };
        try testing.expect(catalog.resolve("sushi_attn_qsa256_qsplit", &qsplit_ins, &.{ .float32, .float32 }, &t, &res));
        try checkPso(dev, l, &res, min_threads);
    }
    var qsplit_bk32 = qsplit_ok;
    qsplit_bk32[3] = .{ .name = "BK", .value = .{ .int = 32 } };
    qsplit_bk32[8] = .{ .name = "DK", .value = .{ .int = 64 } };
    qsplit_bk32[9] = .{ .name = "DV", .value = .{ .int = 64 } };
    try testing.expect(catalog.resolve("sushi_attn_qsa256_qsplit", &qsplit_ins, &.{ .float32, .float32 }, &qsplit_bk32, &res));
    try checkPso(dev, l, &res, min_threads);
    for ([_][4]i32{ .{ 5, 64, 256, 256 }, .{ 8, 64, 320, 256 }, .{ 8, 64, 256, 264 }, .{ 8, 12, 256, 256 }, .{ 8, 64, 252, 256 } }) |bad| {
        const targs = [_]catalog.RuntimeTarg{
            .{ .name = "T", .value = .{ .dtype = .bfloat16 } },
            .{ .name = "NSG", .value = .{ .int = 2 } },
            .{ .name = "NSGT", .value = .{ .int = 2 } },
            .{ .name = "BK", .value = .{ .int = 16 } },
            .{ .name = "RATIO", .value = .{ .int = 4 } },
            .{ .name = "BITS", .value = .{ .int = bad[0] } },
            .{ .name = "GS", .value = .{ .int = bad[1] } },
            .{ .name = "QUANT", .value = .{ .int = 1 } },
            .{ .name = "DK", .value = .{ .int = bad[2] } },
            .{ .name = "DV", .value = .{ .int = bad[3] } },
            .{ .name = "SPARSE", .value = .{ .int = 1 } },
        };
        try testing.expect(!catalog.resolve("sushi_attn_qsa256_qsplit", &qsplit_ins, &.{ .float32, .float32 }, &targs, &res));
    }

    // 6. sushi_fp8_block_gemv{,_staged}: valid M/NR + OOB rejection.
    const fp8_ins = [_]catalog.RuntimeInput{
        .{ .name = "x", .dtype = .bfloat16 },
        .{ .name = "w", .dtype = .uint8 },
        .{ .name = "scales", .dtype = .float32 },
    };
    const fp8_ok = [_]catalog.RuntimeTarg{
        .{ .name = "M", .value = .{ .int = 4 } },
        .{ .name = "NR", .value = .{ .int = 2 } },
        .{ .name = "SGS", .value = .{ .int = 8 } },
        .{ .name = "S", .value = .{ .int = 0 } },
        .{ .name = "T", .value = .{ .dtype = .bfloat16 } },
        .{ .name = "K", .value = .{ .int = 4096 } },
        .{ .name = "TP", .value = .{ .int = 1 } },
        .{ .name = "P0", .value = .{ .int = 4096 } },
        .{ .name = "P1", .value = .{ .int = 0 } },
        .{ .name = "P2", .value = .{ .int = 0 } },
        .{ .name = "BPR", .value = .{ .int = 32 } },
        .{ .name = "VSB", .value = .{ .int = 0 } },
    };
    try testing.expect(catalog.resolve("sushi_fp8_block_gemv", &fp8_ins, &.{.bfloat16}, &fp8_ok, &res));
    try checkPso(dev, l, &res, min_threads);
    const fp8_bad_m = [_]catalog.RuntimeTarg{
        .{ .name = "M", .value = .{ .int = 9 } },
        .{ .name = "NR", .value = .{ .int = 2 } },
        .{ .name = "SGS", .value = .{ .int = 8 } },
        .{ .name = "S", .value = .{ .int = 0 } },
        .{ .name = "T", .value = .{ .dtype = .bfloat16 } },
        .{ .name = "K", .value = .{ .int = 4096 } },
        .{ .name = "TP", .value = .{ .int = 1 } },
        .{ .name = "P0", .value = .{ .int = 4096 } },
        .{ .name = "P1", .value = .{ .int = 0 } },
        .{ .name = "P2", .value = .{ .int = 0 } },
        .{ .name = "BPR", .value = .{ .int = 32 } },
        .{ .name = "VSB", .value = .{ .int = 0 } },
    };
    try testing.expect(!catalog.resolve("sushi_fp8_block_gemv", &fp8_ins, &.{.bfloat16}, &fp8_bad_m, &res));
    var fp8_bad_p = fp8_ok;
    fp8_bad_p[7] = .{ .name = "P0", .value = .{ .int = 0 } };
    try testing.expect(!catalog.resolve("sushi_fp8_block_gemv", &fp8_ins, &.{.bfloat16}, &fp8_bad_p, &res));
    const fp8_st_ok = [_]catalog.RuntimeTarg{
        .{ .name = "M", .value = .{ .int = 16 } },
        .{ .name = "NR", .value = .{ .int = 4 } },
        .{ .name = "SGS", .value = .{ .int = 8 } },
        .{ .name = "S", .value = .{ .int = 1 } },
        .{ .name = "T", .value = .{ .dtype = .bfloat16 } },
        .{ .name = "K", .value = .{ .int = 4096 } },
        .{ .name = "TP", .value = .{ .int = 1 } },
        .{ .name = "P0", .value = .{ .int = 4096 } },
        .{ .name = "P1", .value = .{ .int = 0 } },
        .{ .name = "P2", .value = .{ .int = 0 } },
        .{ .name = "BPR", .value = .{ .int = 32 } },
        .{ .name = "VSB", .value = .{ .int = 0 } },
    };
    try testing.expect(catalog.resolve("sushi_fp8_block_gemv_staged", &fp8_ins, &.{.bfloat16}, &fp8_st_ok, &res));
    try checkPso(dev, l, &res, min_threads);
    var fp8_st_bad = fp8_st_ok;
    fp8_st_bad[0] = .{ .name = "M", .value = .{ .int = 17 } };
    try testing.expect(!catalog.resolve("sushi_fp8_block_gemv_staged", &fp8_ins, &.{.bfloat16}, &fp8_st_bad, &res));

    // 7. sushi_moe_gather_qmv_downred, sushi_expert_bf16_down_reduce, mtp_serial_qmv8_rows{,_stream}: valid + OOB.
    const dr_ins = [_]catalog.RuntimeInput{
        .{ .name = "x", .dtype = .bfloat16 },
        .{ .name = "w_q", .dtype = .uint32 },
        .{ .name = "scales", .dtype = .bfloat16 },
        .{ .name = "biases", .dtype = .bfloat16 },
        .{ .name = "inds", .dtype = .uint32 },
        .{ .name = "scores", .dtype = .bfloat16 },
        .{ .name = "K_size", .dtype = .int32, .scalar = true },
        .{ .name = "N_size", .dtype = .int32, .scalar = true },
    };
    const dr_ok = [_]catalog.RuntimeTarg{
        .{ .name = "T", .value = .{ .dtype = .bfloat16 } },
        .{ .name = "GS", .value = .{ .int = 64 } },
        .{ .name = "BITS", .value = .{ .int = 4 } },
        .{ .name = "TOPK", .value = .{ .int = 8 } },
        .{ .name = "ROWS", .value = .{ .int = 4 } },
        .{ .name = "LPR", .value = .{ .int = 8 } },
        .{ .name = "KP", .value = .{ .int = 256 } },
    };
    try testing.expect(catalog.resolve("sushi_moe_gather_qmv_downred", &dr_ins, &.{.bfloat16}, &dr_ok, &res));
    try checkPso(dev, l, &res, min_threads);
    var dr_bad = dr_ok;
    dr_bad[2] = .{ .name = "BITS", .value = .{ .int = 5 } };
    try testing.expect(!catalog.resolve("sushi_moe_gather_qmv_downred", &dr_ins, &.{.bfloat16}, &dr_bad, &res));
    dr_bad = dr_ok;
    dr_bad[3] = .{ .name = "TOPK", .value = .{ .int = 33 } };
    try testing.expect(!catalog.resolve("sushi_moe_gather_qmv_downred", &dr_ins, &.{.bfloat16}, &dr_bad, &res));
    dr_bad = dr_ok;
    dr_bad[6] = .{ .name = "KP", .value = .{ .int = 513 } }; // ceil(513/8)=65 > pw[64]
    try testing.expect(!catalog.resolve("sushi_moe_gather_qmv_downred", &dr_ins, &.{.bfloat16}, &dr_bad, &res));
    dr_bad = dr_ok;
    dr_bad[6] = .{ .name = "KP", .value = .{ .int = std.math.maxInt(i32) } };
    try testing.expect(!catalog.resolve("sushi_moe_gather_qmv_downred", &dr_ins, &.{.bfloat16}, &dr_bad, &res));

    const bf16_dr_ins = [_]catalog.RuntimeInput{
        .{ .name = "h", .dtype = .bfloat16 },
        .{ .name = "down", .dtype = .bfloat16 },
        .{ .name = "slots", .dtype = .int32 },
        .{ .name = "weights", .dtype = .float32 },
    };
    const bf16_dr_ok = [_]catalog.RuntimeTarg{
        .{ .name = "T", .value = .{ .dtype = .bfloat16 } },
        .{ .name = "TOPK", .value = .{ .int = 8 } },
        .{ .name = "NDIM", .value = .{ .int = 2560 } },
        .{ .name = "KDIM", .value = .{ .int = 640 } },
        .{ .name = "LPR", .value = .{ .int = 8 } },
        .{ .name = "NPT", .value = .{ .int = 4 } },
        .{ .name = "SSG", .value = .{ .int = 4 } },
        .{ .name = "STAGE", .value = .{ .int = 1 } },
    };
    try testing.expect(catalog.resolve("sushi_expert_bf16_down_reduce", &bf16_dr_ins, &.{.bfloat16}, &bf16_dr_ok, &res));
    try checkPso(dev, l, &res, min_threads);
    var bf16_dr_npt3 = bf16_dr_ok;
    bf16_dr_npt3[5] = .{ .name = "NPT", .value = .{ .int = 3 } };
    try testing.expect(catalog.resolve("sushi_expert_bf16_down_reduce", &bf16_dr_ins, &.{.bfloat16}, &bf16_dr_npt3, &res));
    try checkPso(dev, l, &res, min_threads);
    var bf16_dr_bad = bf16_dr_ok;
    bf16_dr_bad[5] = .{ .name = "NPT", .value = .{ .int = 16 } };
    try testing.expect(!catalog.resolve("sushi_expert_bf16_down_reduce", &bf16_dr_ins, &.{.bfloat16}, &bf16_dr_bad, &res));
    bf16_dr_bad = bf16_dr_ok;
    bf16_dr_bad[3] = .{ .name = "KDIM", .value = .{ .int = 4104 } }; // STAGE=1 exceeds hstage[512]
    try testing.expect(!catalog.resolve("sushi_expert_bf16_down_reduce", &bf16_dr_ins, &.{.bfloat16}, &bf16_dr_bad, &res));

    const mtp_tile_ins = [_]catalog.RuntimeInput{
        .{ .name = "x", .dtype = .bfloat16 },
        .{ .name = "w", .dtype = .uint32 },
        .{ .name = "sc", .dtype = .bfloat16 },
        .{ .name = "bi", .dtype = .bfloat16 },
        .{ .name = "K_size", .dtype = .int32, .scalar = true },
        .{ .name = "N_size", .dtype = .int32, .scalar = true },
    };
    const mtp_tile_bad = [_]catalog.RuntimeTarg{
        .{ .name = "T", .value = .{ .dtype = .bfloat16 } },
        .{ .name = "M", .value = .{ .int = 4 } },
        .{ .name = "GS", .value = .{ .int = 12 } }, // FAST=1 requires GS % 8 == 0
        .{ .name = "NV", .value = .{ .int = 4 } },
        .{ .name = "FAST", .value = .{ .int = 1 } },
        .{ .name = "RPS", .value = .{ .int = 2 } },
    };
    try testing.expect(!catalog.resolve("mtp_serial_qmv8_rows", &mtp_tile_ins, &.{.bfloat16}, &mtp_tile_bad, &res));

    const mtp_st_ins = [_]catalog.RuntimeInput{
        .{ .name = "x", .dtype = .bfloat16 },
        .{ .name = "w", .dtype = .uint32 },
        .{ .name = "scales", .dtype = .bfloat16 },
        .{ .name = "biases", .dtype = .bfloat16 },
    };
    const mtp_st_ok = [_]catalog.RuntimeTarg{
        .{ .name = "M", .value = .{ .int = 7 } },
        .{ .name = "N", .value = .{ .int = 2560 } },
        .{ .name = "K", .value = .{ .int = 2560 } },
        .{ .name = "GS", .value = .{ .int = 64 } },
        .{ .name = "O", .value = .{ .int = 2 } },
        .{ .name = "SG", .value = .{ .int = 8 } },
        .{ .name = "MT", .value = .{ .int = 4 } },
        .{ .name = "SPLIT", .value = .{ .int = 1 } },
    };
    try testing.expect(catalog.resolve("mtp_serial_qmv8_rows_stream", &mtp_st_ins, &.{.bfloat16}, &mtp_st_ok, &res));
    try checkPso(dev, l, &res, min_threads);
    var mtp_st_bad = mtp_st_ok;
    mtp_st_bad[6] = .{ .name = "MT", .value = .{ .int = 8 } };
    try testing.expect(!catalog.resolve("mtp_serial_qmv8_rows_stream", &mtp_st_ins, &.{.bfloat16}, &mtp_st_bad, &res));
    mtp_st_bad = mtp_st_ok;
    mtp_st_bad[5] = .{ .name = "SG", .value = .{ .int = 1 } }; // SPLIT=1 requires even SG >= 2
    try testing.expect(!catalog.resolve("mtp_serial_qmv8_rows_stream", &mtp_st_ins, &.{.bfloat16}, &mtp_st_bad, &res));
}

test "stock metallib: fused EXL3 FC resolution, slot disjointness, PSO headroom, and GPU bit-exact decode" {
    try @import("../mlx.zig").requireGpu();
    const dev = try metal.device();
    const l = try lib(dev);
    const names = try l.functionNames(testing.allocator);
    defer {
        for (names) |s| testing.allocator.free(s);
        testing.allocator.free(names);
    }
    var have: std.StringHashMapUnmanaged(void) = .empty;
    defer have.deinit(testing.allocator);
    for (names) |s| try have.put(testing.allocator, s, {});
    const catalog = @import("../metallib_catalog.zig");

    const assertDisjointFcs = struct {
        fn check(r: *const catalog.Resolved) !void {
            if (r.n_ints > 1) for (1..r.n_ints) |i| try testing.expect(r.ints[i - 1].index < r.ints[i].index);
            if (r.n_flags > 1) for (1..r.n_flags) |i| try testing.expect(r.flags[i - 1].index < r.flags[i].index);
        }
    }.check;

    const has_nax = @import("sushi_exl3").device.current().has(.matrix_g17);
    var res: catalog.Resolved = undefined;
    var nbuf: [192]u8 = undefined;

    // 1. Resolution + disjoint FC slots + live PSO headroom across all 4 (Part A, Part B) codebook pairings,
    //    all SIMD modes (lane/one/shared_rows 2 & 3), snax (UNR 1 & 2), and rg_fold/gated_fold (affine & trellis).
    const fz_down_ins = [_]catalog.RuntimeInput{
        .{ .name = "middle", .dtype = .float16, .scalar = false },
        .{ .name = "trellis", .dtype = .uint16, .scalar = false },
        .{ .name = "slots", .dtype = .uint32, .scalar = false },
        .{ .name = "meta", .dtype = .uint32, .scalar = false },
        .{ .name = "s_x", .dtype = .float16, .scalar = false },
        .{ .name = "s_trellis", .dtype = .uint16, .scalar = false },
        .{ .name = "s_slots", .dtype = .uint32, .scalar = false },
    };
    const fz_down_snax_ins = [_]catalog.RuntimeInput{
        .{ .name = "middle", .dtype = .float16, .scalar = false },
        .{ .name = "trellis", .dtype = .uint16, .scalar = false },
        .{ .name = "slots", .dtype = .uint32, .scalar = false },
        .{ .name = "meta", .dtype = .uint32, .scalar = false },
        .{ .name = "s_x", .dtype = .float16, .scalar = false },
        .{ .name = "s_trellis", .dtype = .uint16, .scalar = false },
        .{ .name = "s_svh", .dtype = .float16, .scalar = false },
    };
    const gated_down_ins = [_]catalog.RuntimeInput{
        .{ .name = "ig", .dtype = .float32, .scalar = false },
        .{ .name = "iu", .dtype = .float32, .scalar = false },
        .{ .name = "trellis", .dtype = .uint16, .scalar = false },
        .{ .name = "svhg", .dtype = .float16, .scalar = false },
        .{ .name = "svhu", .dtype = .float16, .scalar = false },
        .{ .name = "suhd", .dtype = .float16, .scalar = false },
        .{ .name = "slots", .dtype = .uint32, .scalar = false },
        .{ .name = "s_x", .dtype = .bfloat16, .scalar = false },
        .{ .name = "s_w", .dtype = .uint32, .scalar = false },
        .{ .name = "s_sc", .dtype = .bfloat16, .scalar = false },
        .{ .name = "s_bi", .dtype = .bfloat16, .scalar = false },
    };
    const gated_down_tr_ins = [_]catalog.RuntimeInput{
        .{ .name = "ig", .dtype = .float32, .scalar = false },
        .{ .name = "iu", .dtype = .float32, .scalar = false },
        .{ .name = "trellis", .dtype = .uint16, .scalar = false },
        .{ .name = "svhg", .dtype = .float16, .scalar = false },
        .{ .name = "svhu", .dtype = .float16, .scalar = false },
        .{ .name = "suhd", .dtype = .float16, .scalar = false },
        .{ .name = "slots", .dtype = .uint32, .scalar = false },
        .{ .name = "s_x", .dtype = .bfloat16, .scalar = false },
        .{ .name = "s_trellis", .dtype = .uint16, .scalar = false },
        .{ .name = "s_suh", .dtype = .float16, .scalar = false },
        .{ .name = "s_svh", .dtype = .float16, .scalar = false },
        .{ .name = "s_g1", .dtype = .float32, .scalar = false },
        .{ .name = "s_g1svh", .dtype = .float16, .scalar = false },
        .{ .name = "s_sigmoid", .dtype = .bfloat16, .scalar = false },
    };
    const rg_down_tr_ins = [_]catalog.RuntimeInput{
        .{ .name = "ig", .dtype = .float32, .scalar = false },
        .{ .name = "iu", .dtype = .float32, .scalar = false },
        .{ .name = "trellis", .dtype = .uint16, .scalar = false },
        .{ .name = "svhg", .dtype = .float16, .scalar = false },
        .{ .name = "svhu", .dtype = .float16, .scalar = false },
        .{ .name = "suhd", .dtype = .float16, .scalar = false },
        .{ .name = "slots", .dtype = .uint32, .scalar = false },
        .{ .name = "meta", .dtype = .uint32, .scalar = false },
        .{ .name = "s_x", .dtype = .bfloat16, .scalar = false },
        .{ .name = "s_trellis", .dtype = .uint16, .scalar = false },
        .{ .name = "s_suh", .dtype = .float16, .scalar = false },
        .{ .name = "s_svh", .dtype = .float16, .scalar = false },
        .{ .name = "s_g1", .dtype = .float32, .scalar = false },
        .{ .name = "s_g1svh", .dtype = .float16, .scalar = false },
        .{ .name = "s_sigmoid", .dtype = .bfloat16, .scalar = false },
    };

    const AbPair = struct { a_tag: []const u8, a_cls: []const u8, b_tag: []const u8, b_is_p1: bool, b_rates: []const i32 };
    const ab_pairs = [_]AbPair{
        // All 4 (Part A, Part B) pairings across boundary scalar (incl. previously-unserved 34) & PAIR1 rates.
        .{ .a_tag = "spec1_w15_t37f72431", .a_cls = "32x0_64x0", .b_tag = "spec1_w15", .b_is_p1 = false, .b_rates = &.{ 16, 34, 64, 80, 128 } },
        .{ .a_tag = "spec1_w15_t37f72431", .a_cls = "32x0_64x0", .b_tag = "pair1_w16", .b_is_p1 = true, .b_rates = &.{ 24, 42, 56, 64 } },
        .{ .a_tag = "pair1_w16", .a_cls = "24x0_32x0_40x0", .b_tag = "spec1_w15", .b_is_p1 = false, .b_rates = &.{ 16, 34, 64, 128 } },
        .{ .a_tag = "pair1_w16", .a_cls = "24x0_32x0_40x0", .b_tag = "pair1_w16", .b_is_p1 = true, .b_rates = &.{ 24, 42, 56, 64 } },
    };

    for (ab_pairs) |ab| {
        const b_dig = if (ab.b_is_p1) "" else "_t37f72431";
        for (ab.b_rates) |bn| {
            // Check fz_down SIMD modes: lane, one, shared_rows(2), shared_rows(3)
            for ([_]struct { suf: []const u8, sr: i32 }{
                .{ .suf = "", .sr = 2 },
                .{ .suf = "_one", .sr = 2 },
                .{ .suf = "_shared_rows", .sr = 2 },
                .{ .suf = "_shared_rows", .sr = 3 },
            }) |m| {
                const nm = try std.fmt.bufPrint(&nbuf, "sushi_exl3_fz_down_{s}_s{s}_n{d}{s}_{s}_g0{s}_tm", .{ ab.a_tag, ab.b_tag, bn, b_dig, ab.a_cls, m.suf });
                const targs = [_]catalog.RuntimeTarg{
                    .{ .name = "OTPT", .value = .{ .int = 2 } },
                    .{ .name = "NSPLIT", .value = .{ .int = 1 } },
                    .{ .name = "IDIM", .value = .{ .int = 2304 } },
                    .{ .name = "ODIM", .value = .{ .int = 5120 } },
                    .{ .name = "FUNNEL", .value = .{ .int = 1 } },
                    .{ .name = "S_IDIM", .value = .{ .int = 2304 } },
                    .{ .name = "S_ODIM", .value = .{ .int = 5120 } },
                    .{ .name = "S_NHW", .value = .{ .int = bn } },
                    .{ .name = "S_ROWS", .value = .{ .int = 6 } },
                    .{ .name = "S_SHARED_ROWS", .value = .{ .int = m.sr } },
                    .{ .name = "FZ_AX", .value = .{ .int = 160 } },
                    .{ .name = "FZ_AY", .value = .{ .int = 6 } },
                    .{ .name = "FZ_BX", .value = .{ .int = 160 } },
                    .{ .name = "FZ_BY", .value = .{ .int = 2 } },
                    .{ .name = "FZ_NA", .value = .{ .int = 960 } },
                    .{ .name = "FZ_NB", .value = .{ .int = 320 } },
                    .{ .name = "FZ_B_FIRST", .value = .{ .int = 0 } },
                };
                try testing.expect(catalog.resolve(nm, &fz_down_ins, &.{ .float16, .float16 }, &targs, &res));
                try testing.expect(have.contains(res.fnName()));
                try assertDisjointFcs(&res);
                if (bn == ab.b_rates[0] or (m.suf.len == 0 and bn == ab.b_rates[ab.b_rates.len - 1])) {
                    try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 640);
                }
            }

            // Check fz_down_snax (UNR=1 & UNR=2) and rg_fold_down_tr + gated_fold_down_tr
            if (has_nax) {
                for ([_]i32{ 1, 2 }) |unr| {
                    const nm = try std.fmt.bufPrint(&nbuf, "sushi_exl3_fz_down_{s}_s{s}_n{d}{s}_{s}_g2_snax_tm", .{ ab.a_tag, ab.b_tag, bn, b_dig, ab.a_cls });
                    const targs = [_]catalog.RuntimeTarg{
                        .{ .name = "GROUP", .value = .{ .int = 2 } },
                        .{ .name = "OTPT", .value = .{ .int = 2 } },
                        .{ .name = "NSPLIT", .value = .{ .int = 1 } },
                        .{ .name = "IDIM", .value = .{ .int = 2304 } },
                        .{ .name = "ODIM", .value = .{ .int = 5120 } },
                        .{ .name = "FUNNEL", .value = .{ .int = 1 } },
                        .{ .name = "NSLOTS", .value = .{ .int = 36 } },
                        .{ .name = "S_IDIM", .value = .{ .int = 2304 } },
                        .{ .name = "S_ODIM", .value = .{ .int = 5120 } },
                        .{ .name = "S_ROWS", .value = .{ .int = 6 } },
                        .{ .name = "S_GROUPS", .value = .{ .int = 1 } },
                        .{ .name = "S_TGRP", .value = .{ .int = 4 } },
                        .{ .name = "S_KPH", .value = .{ .int = 1 } },
                        .{ .name = "S_NSPLIT", .value = .{ .int = 2 } },
                        .{ .name = "S_CHUNK", .value = .{ .int = 512 } },
                        .{ .name = "S_FOLD", .value = .{ .int = 0 } },
                        .{ .name = "S_NHW", .value = .{ .int = bn } },
                        .{ .name = "S_SB1", .value = .{ .int = 5120 } },
                        .{ .name = "S_SB2", .value = .{ .int = 5120 } },
                        .{ .name = "S_SB3", .value = .{ .int = 5120 } },
                        .{ .name = "S_UNR", .value = .{ .int = unr } },
                        .{ .name = "S_R32", .value = .{ .int = 1 } },
                        .{ .name = "S_SW", .value = .{ .int = 1 } },
                        .{ .name = "FZ_AX", .value = .{ .int = 160 } },
                        .{ .name = "FZ_AY", .value = .{ .int = 18 } },
                        .{ .name = "FZ_BX", .value = .{ .int = 40 } },
                        .{ .name = "FZ_BY", .value = .{ .int = 2 } },
                        .{ .name = "FZ_NA", .value = .{ .int = 2880 } },
                        .{ .name = "FZ_NB", .value = .{ .int = 80 } },
                        .{ .name = "FZ_B_FIRST", .value = .{ .int = 0 } },
                    };
                    try testing.expect(catalog.resolve(nm, &fz_down_snax_ins, &.{ .float16, .float32 }, &targs, &res));
                    try testing.expect(have.contains(res.fnName()));
                    try assertDisjointFcs(&res);
                    if (bn == ab.b_rates[0] or (unr == 2 and bn == ab.b_rates[ab.b_rates.len - 1])) {
                        try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 640);
                    }
                }

                // rg_fold_down_tr (mixed A + single-rate B)
                {
                    const nm = try std.fmt.bufPrint(&nbuf, "sushi_rg_fold_down_{s}_{s}_tm_tr_{s}_n{d}_{d}{s}", .{ ab.a_tag, ab.a_cls, ab.b_tag, bn, bn, b_dig });
                    const targs = [_]catalog.RuntimeTarg{
                        .{ .name = "OTPT", .value = .{ .int = 2 } },
                        .{ .name = "NSPLIT", .value = .{ .int = 2 } },
                        .{ .name = "IDIM", .value = .{ .int = 640 } },
                        .{ .name = "ODIM", .value = .{ .int = 2560 } },
                        .{ .name = "FUNNEL", .value = .{ .int = 1 } },
                        .{ .name = "S_IDIM", .value = .{ .int = 640 } },
                        .{ .name = "S_ODIM", .value = .{ .int = 2560 } },
                        .{ .name = "S_ROWS", .value = .{ .int = 4 } },
                        .{ .name = "S_GROUPS", .value = .{ .int = 1 } },
                        .{ .name = "S_TGRP", .value = .{ .int = 4 } },
                        .{ .name = "S_KPH", .value = .{ .int = 1 } },
                        .{ .name = "S_NSPLIT", .value = .{ .int = 2 } },
                        .{ .name = "S_CHUNK", .value = .{ .int = 256 } },
                        .{ .name = "S_FOLD", .value = .{ .int = 0 } },
                        .{ .name = "S_NHW", .value = .{ .int = bn } },
                        .{ .name = "S_SB1", .value = .{ .int = 2560 } },
                        .{ .name = "S_SB2", .value = .{ .int = 2560 } },
                        .{ .name = "S_SB3", .value = .{ .int = 2560 } },
                        .{ .name = "S_UNR", .value = .{ .int = 2 } },
                        .{ .name = "S_R32", .value = .{ .int = 1 } },
                        .{ .name = "S_SW", .value = .{ .int = 1 } },
                        .{ .name = "S_GNSPLIT", .value = .{ .int = 2 } },
                        .{ .name = "S_GODIM", .value = .{ .int = 1280 } },
                        .{ .name = "S_GWIDTH", .value = .{ .int = 640 } },
                        .{ .name = "FZ_AX", .value = .{ .int = 80 } },
                        .{ .name = "FZ_AY", .value = .{ .int = 10 } },
                        .{ .name = "FZ_BX", .value = .{ .int = 20 } },
                        .{ .name = "FZ_BY", .value = .{ .int = 2 } },
                        .{ .name = "FZ_NA", .value = .{ .int = 800 } },
                        .{ .name = "FZ_NB", .value = .{ .int = 40 } },
                        .{ .name = "FZ_B_FIRST", .value = .{ .int = 0 } },
                    };
                    try testing.expect(catalog.resolve(nm, &rg_down_tr_ins, &.{ .float16, .float32 }, &targs, &res));
                    try testing.expect(have.contains(res.fnName()));
                    try assertDisjointFcs(&res);
                    if (bn == ab.b_rates[0]) {
                        try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 640);
                    }
                }

                // gated_fold_down_tr (single-rate A + single-rate B)
                {
                    const an: i32 = if (std.mem.startsWith(u8, ab.a_tag, "pair1")) 40 else 34;
                    const nm = try std.fmt.bufPrint(&nbuf, "sushi_gated_fold_down_{s}_tr_{s}_n{d}_{d}{s}", .{ ab.a_tag, ab.b_tag, bn, bn, b_dig });
                    const targs = [_]catalog.RuntimeTarg{
                        .{ .name = "OTPT", .value = .{ .int = 2 } },
                        .{ .name = "IDIM", .value = .{ .int = 640 } },
                        .{ .name = "ODIM", .value = .{ .int = 2560 } },
                        .{ .name = "NSPLIT", .value = .{ .int = 2 } },
                        .{ .name = "NHW", .value = .{ .int = an } },
                        .{ .name = "FUNNEL", .value = .{ .int = 1 } },
                        .{ .name = "S_IDIM", .value = .{ .int = 640 } },
                        .{ .name = "S_ODIM", .value = .{ .int = 2560 } },
                        .{ .name = "S_ROWS", .value = .{ .int = 4 } },
                        .{ .name = "S_GROUPS", .value = .{ .int = 1 } },
                        .{ .name = "S_TGRP", .value = .{ .int = 4 } },
                        .{ .name = "S_KPH", .value = .{ .int = 1 } },
                        .{ .name = "S_NSPLIT", .value = .{ .int = 2 } },
                        .{ .name = "S_CHUNK", .value = .{ .int = 256 } },
                        .{ .name = "S_FOLD", .value = .{ .int = 0 } },
                        .{ .name = "S_NHW", .value = .{ .int = bn } },
                        .{ .name = "S_SB1", .value = .{ .int = 2560 } },
                        .{ .name = "S_SB2", .value = .{ .int = 2560 } },
                        .{ .name = "S_SB3", .value = .{ .int = 2560 } },
                        .{ .name = "S_UNR", .value = .{ .int = 2 } },
                        .{ .name = "S_R32", .value = .{ .int = 1 } },
                        .{ .name = "S_SW", .value = .{ .int = 1 } },
                        .{ .name = "S_GNSPLIT", .value = .{ .int = 2 } },
                        .{ .name = "S_GODIM", .value = .{ .int = 1280 } },
                        .{ .name = "S_GWIDTH", .value = .{ .int = 640 } },
                        .{ .name = "FZ_AX", .value = .{ .int = 80 } },
                        .{ .name = "FZ_AY", .value = .{ .int = 10 } },
                        .{ .name = "FZ_BX", .value = .{ .int = 20 } },
                        .{ .name = "FZ_BY", .value = .{ .int = 2 } },
                        .{ .name = "FZ_NA", .value = .{ .int = 800 } },
                        .{ .name = "FZ_NB", .value = .{ .int = 40 } },
                        .{ .name = "FZ_B_FIRST", .value = .{ .int = 0 } },
                    };
                    try testing.expect(catalog.resolve(nm, &gated_down_tr_ins, &.{ .float16, .float32 }, &targs, &res));
                    try testing.expect(have.contains(res.fnName()));
                    try assertDisjointFcs(&res);
                    if (bn == ab.b_rates[0]) {
                        try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 640);
                    }
                }
            }
        }
    }

    // Check gated_fold affine across previously-unserved scalar rate (n=34) and all 21 PAIR1 rates (24..64)
    {
        var p1_n: i32 = 24;
        while (p1_n <= 64) : (p1_n += 2) {
            const targs = [_]catalog.RuntimeTarg{
                .{ .name = "OTPT", .value = .{ .int = 2 } },
                .{ .name = "IDIM", .value = .{ .int = 640 } },
                .{ .name = "ODIM", .value = .{ .int = 2560 } },
                .{ .name = "NSPLIT", .value = .{ .int = 2 } },
                .{ .name = "NHW", .value = .{ .int = p1_n } },
                .{ .name = "FUNNEL", .value = .{ .int = 1 } },
                .{ .name = "S_K", .value = .{ .int = 640 } },
                .{ .name = "S_N", .value = .{ .int = 2560 } },
                .{ .name = "S_GS", .value = .{ .int = 64 } },
                .{ .name = "S_FAST", .value = .{ .int = 1 } },
                .{ .name = "FZ_AX", .value = .{ .int = 80 } },
                .{ .name = "FZ_AY", .value = .{ .int = 10 } },
                .{ .name = "FZ_BX", .value = .{ .int = 160 } },
                .{ .name = "FZ_BY", .value = .{ .int = 1 } },
                .{ .name = "FZ_NA", .value = .{ .int = 800 } },
                .{ .name = "FZ_NB", .value = .{ .int = 160 } },
                .{ .name = "FZ_B_FIRST", .value = .{ .int = 0 } },
            };
            try testing.expect(catalog.resolve("sushi_gated_fold_down_pair1_w16", &gated_down_ins, &.{ .float16, .bfloat16 }, &targs, &res));
            try testing.expect(have.contains(res.fnName()));
            try assertDisjointFcs(&res);
            if (p1_n == 24 or p1_n == 42 or p1_n == 56 or p1_n == 64) {
                try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 640);
            }
        }
    }

    // Negative guards: mismatched S_NHW, bad S_SHARED_ROWS, odd/OOB rates, out-of-table std1/spec1-w16 rates,
    // codebook/digest mismatches, and SNAX geometry / TG SRAM overflow caps.
    {
        var bad_targs = [_]catalog.RuntimeTarg{
            .{ .name = "OTPT", .value = .{ .int = 1 } },
            .{ .name = "NSPLIT", .value = .{ .int = 1 } },
            .{ .name = "IDIM", .value = .{ .int = 128 } },
            .{ .name = "ODIM", .value = .{ .int = 128 } },
            .{ .name = "FUNNEL", .value = .{ .int = 1 } },
            .{ .name = "S_IDIM", .value = .{ .int = 128 } },
            .{ .name = "S_ODIM", .value = .{ .int = 128 } },
            .{ .name = "S_NHW", .value = .{ .int = 32 } },
            .{ .name = "S_ROWS", .value = .{ .int = 6 } },
            .{ .name = "S_SHARED_ROWS", .value = .{ .int = 2 } },
            .{ .name = "FZ_AX", .value = .{ .int = 1 } },
            .{ .name = "FZ_AY", .value = .{ .int = 1 } },
            .{ .name = "FZ_BX", .value = .{ .int = 1 } },
            .{ .name = "FZ_BY", .value = .{ .int = 1 } },
            .{ .name = "FZ_NA", .value = .{ .int = 1 } },
            .{ .name = "FZ_NB", .value = .{ .int = 1 } },
            .{ .name = "FZ_B_FIRST", .value = .{ .int = 0 } },
        };
        // S_NHW=32 mismatches name _n64
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_pair1_w16_spair1_w16_n64_32x0_g0_shared_rows", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        // Bad S_SHARED_ROWS=4
        bad_targs[7].value = .{ .int = 64 };
        bad_targs[9].value = .{ .int = 4 };
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_pair1_w16_spair1_w16_n64_32x0_g0_shared_rows", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        // Out-of-range PAIR1 rate n=66
        bad_targs[7].value = .{ .int = 66 };
        bad_targs[9].value = .{ .int = 2 };
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_pair1_w16_spair1_w16_n66_32x0_g0_shared_rows", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        // Odd scalar rate n=33 and out-of-range scalar rates n=14, n=130
        bad_targs[7].value = .{ .int = 33 };
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_pair1_w16_smcg_w16_n33_32x0_g0", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        bad_targs[7].value = .{ .int = 14 };
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_pair1_w16_smcg_w16_n14_32x0_g0", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        bad_targs[7].value = .{ .int = 130 };
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_pair1_w16_smcg_w16_n130_32x0_g0", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        // std1 (`c5dba171`) and spec1-w16 (`753ffa03`) only cover 24..64: reject out-of-table rates on Part B and Part A
        bad_targs[7].value = .{ .int = 16 };
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_pair1_w16_sstd1_w15_n16_tc5dba171_32x0_g0", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_pair1_w16_sspec1_w16_n16_t753ffa03_32x0_g0", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        bad_targs[7].value = .{ .int = 66 };
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_pair1_w16_sstd1_w15_n66_tc5dba171_32x0_g0", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_pair1_w16_sspec1_w16_n66_t753ffa03_32x0_g0", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        bad_targs[7].value = .{ .int = 32 };
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_std1_w15_tc5dba171_smcg_w16_n32_16x0_g0", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_std1_w15_tc5dba171_smcg_w16_n32_80x0_g0", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_spec1_w16_t753ffa03_smcg_w16_n32_16x0_g0", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_spec1_w16_t753ffa03_smcg_w16_n32_80x0_g0", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        // Codebook/digest mismatches (spec1 + std1 digest, std1 + spec1 digest, spec1 + pair1 digest)
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_spec1_w15_tc5dba171_smcg_w16_n32_32x0_g0", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_std1_w15_t37f72431_smcg_w16_n32_32x0_g0", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_pair1_w16_sspec1_w16_n32_tc089b4eb_32x0_g0", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_pair1_w16_sspec1_w15_n32_tc5dba171_32x0_g0", &fz_down_ins, &.{ .float16, .float16 }, &bad_targs, &res));

        // SNAX geometry guards: S_ROWS=17, S_UNR=3, S_GROUPS=2 on fz_down_snax, and PREP_F TG overflow on fz_pair_snax / rg_fold_down_tr
        var bad_snax = [_]catalog.RuntimeTarg{
            .{ .name = "OTPT", .value = .{ .int = 1 } },
            .{ .name = "NSPLIT", .value = .{ .int = 1 } },
            .{ .name = "IDIM", .value = .{ .int = 128 } },
            .{ .name = "ODIM", .value = .{ .int = 128 } },
            .{ .name = "FUNNEL", .value = .{ .int = 1 } },
            .{ .name = "S_IDIM", .value = .{ .int = 128 } },
            .{ .name = "S_ODIM", .value = .{ .int = 128 } },
            .{ .name = "S_ROWS", .value = .{ .int = 17 } },
            .{ .name = "S_GROUPS", .value = .{ .int = 1 } },
            .{ .name = "S_TGRP", .value = .{ .int = 4 } },
            .{ .name = "S_KPH", .value = .{ .int = 1 } },
            .{ .name = "S_NSPLIT", .value = .{ .int = 1 } },
            .{ .name = "S_CHUNK", .value = .{ .int = 128 } },
            .{ .name = "S_FOLD", .value = .{ .int = 0 } },
            .{ .name = "S_NHW", .value = .{ .int = 32 } },
            .{ .name = "S_SB1", .value = .{ .int = 128 } },
            .{ .name = "S_SB2", .value = .{ .int = 128 } },
            .{ .name = "S_SB3", .value = .{ .int = 128 } },
            .{ .name = "S_UNR", .value = .{ .int = 2 } },
            .{ .name = "S_R32", .value = .{ .int = 1 } },
            .{ .name = "S_SW", .value = .{ .int = 1 } },
            .{ .name = "FZ_AX", .value = .{ .int = 1 } },
            .{ .name = "FZ_AY", .value = .{ .int = 1 } },
            .{ .name = "FZ_BX", .value = .{ .int = 1 } },
            .{ .name = "FZ_BY", .value = .{ .int = 1 } },
            .{ .name = "FZ_NA", .value = .{ .int = 1 } },
            .{ .name = "FZ_NB", .value = .{ .int = 1 } },
            .{ .name = "FZ_B_FIRST", .value = .{ .int = 0 } },
        };
        // S_ROWS=17 > ROW_CAP(16)
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_pair1_w16_smcg_w16_n32_32x0_g0_snax", &fz_down_snax_ins, &.{ .float16, .float32 }, &bad_snax, &res));
        // S_UNR=3 > 2
        bad_snax[7].value = .{ .int = 16 };
        bad_snax[18].value = .{ .int = 3 };
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_pair1_w16_smcg_w16_n32_32x0_g0_snax", &fz_down_snax_ins, &.{ .float16, .float32 }, &bad_snax, &res));
        // S_GROUPS=2 on fz_down_snax (ignored in ROWS_NAX_PREPPED_SOURCE)
        bad_snax[18].value = .{ .int = 2 };
        bad_snax[8].value = .{ .int = 2 };
        try testing.expect(!catalog.resolve("sushi_exl3_fz_down_pair1_w16_smcg_w16_n32_32x0_g0_snax", &fz_down_snax_ins, &.{ .float16, .float32 }, &bad_snax, &res));
        // S_ROWS=17 > ROW_CAP(16) on fz_pair_snax (now using ROWS_NAX_PREPPED_SOURCE inputs `s_xg, s_xu, s_tg, s_tu`)
        const fz_pair_snax_ins = [_]catalog.RuntimeInput{
            .{ .name = "prep", .dtype = .float16, .scalar = false },
            .{ .name = "tg", .dtype = .uint16, .scalar = false },
            .{ .name = "tu", .dtype = .uint16, .scalar = false },
            .{ .name = "slots", .dtype = .uint32, .scalar = false },
            .{ .name = "meta", .dtype = .uint32, .scalar = false },
            .{ .name = "s_xg", .dtype = .float16, .scalar = false },
            .{ .name = "s_xu", .dtype = .float16, .scalar = false },
            .{ .name = "s_tg", .dtype = .uint16, .scalar = false },
            .{ .name = "s_tu", .dtype = .uint16, .scalar = false },
        };
        var bad_pair_snax = [_]catalog.RuntimeTarg{
            .{ .name = "OTPT", .value = .{ .int = 1 } },
            .{ .name = "NSPLIT", .value = .{ .int = 1 } },
            .{ .name = "TOPK", .value = .{ .int = 1 } },
            .{ .name = "IDIM", .value = .{ .int = 128 } },
            .{ .name = "ODIM", .value = .{ .int = 128 } },
            .{ .name = "FUNNEL", .value = .{ .int = 1 } },
            .{ .name = "S_IDIM", .value = .{ .int = 128 } },
            .{ .name = "S_ODIM", .value = .{ .int = 128 } },
            .{ .name = "S_ROWS", .value = .{ .int = 17 } },
            .{ .name = "S_GROUPS", .value = .{ .int = 1 } },
            .{ .name = "S_TGRP", .value = .{ .int = 4 } },
            .{ .name = "S_KPH", .value = .{ .int = 1 } },
            .{ .name = "S_NSPLIT", .value = .{ .int = 1 } },
            .{ .name = "S_CHUNK", .value = .{ .int = 128 } },
            .{ .name = "S_FOLD", .value = .{ .int = 0 } },
            .{ .name = "S_NHW", .value = .{ .int = 32 } },
            .{ .name = "S_SB1", .value = .{ .int = 128 } },
            .{ .name = "S_SB2", .value = .{ .int = 128 } },
            .{ .name = "S_SB3", .value = .{ .int = 128 } },
            .{ .name = "S_UNR", .value = .{ .int = 2 } },
            .{ .name = "S_R32", .value = .{ .int = 1 } },
            .{ .name = "S_SW", .value = .{ .int = 1 } },
            .{ .name = "FZ_AX", .value = .{ .int = 1 } },
            .{ .name = "FZ_AY", .value = .{ .int = 1 } },
            .{ .name = "FZ_BX", .value = .{ .int = 1 } },
            .{ .name = "FZ_BY", .value = .{ .int = 1 } },
            .{ .name = "FZ_NA", .value = .{ .int = 1 } },
            .{ .name = "FZ_NB", .value = .{ .int = 1 } },
            .{ .name = "FZ_B_FIRST", .value = .{ .int = 0 } },
        };
        try testing.expect(!catalog.resolve("sushi_exl3_fz_pair_pair1_w16_smcg_w16_n32_32x32_g0_snax", &fz_pair_snax_ins, &.{ .float32, .float32, .float32, .float32 }, &bad_pair_snax, &res));
        // Odd KT (`S_IDIM / S_NSPLIT / 16 = 9`) with `S_UNR = 2` on fz_pair_snax must be rejected.
        bad_pair_snax[6].value = .{ .int = 144 };
        bad_pair_snax[8].value = .{ .int = 6 };
        try testing.expect(!catalog.resolve("sushi_exl3_fz_pair_pair1_w16_smcg_w16_n32_32x32_g0_snax", &fz_pair_snax_ins, &.{ .float32, .float32, .float32, .float32 }, &bad_pair_snax, &res));
        // Valid fz_pair_snax with asymmetric Part A `gate != up` rates (`24x32_40x56`) resolves and meets >= 640 headroom.
        bad_pair_snax[6].value = .{ .int = 5120 };
        bad_pair_snax[7].value = .{ .int = 2304 };
        bad_pair_snax[12].value = .{ .int = 4 };
        bad_pair_snax[15].value = .{ .int = 80 };
        try testing.expect(catalog.resolve("sushi_exl3_fz_pair_pair1_w16_sspec1_w15_n80_t37f72431_24x32_40x56_g0_snax_tm", &fz_pair_snax_ins, &.{ .float32, .float32, .float32, .float32 }, &bad_pair_snax, &res));
        try testing.expect(have.contains(res.fnName()));
        try assertDisjointFcs(&res);
        if (has_nax) try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 640);
        // PREP_F TG overflow on rg_fold_down_tr (`S_ROWS * S_CHUNK = 16 * 512 > 4096`)
        const bad_rg_tr = [_]catalog.RuntimeTarg{
            .{ .name = "OTPT", .value = .{ .int = 2 } },
            .{ .name = "NSPLIT", .value = .{ .int = 2 } },
            .{ .name = "IDIM", .value = .{ .int = 640 } },
            .{ .name = "ODIM", .value = .{ .int = 2560 } },
            .{ .name = "FUNNEL", .value = .{ .int = 1 } },
            .{ .name = "S_IDIM", .value = .{ .int = 640 } },
            .{ .name = "S_ODIM", .value = .{ .int = 2560 } },
            .{ .name = "S_ROWS", .value = .{ .int = 16 } },
            .{ .name = "S_GROUPS", .value = .{ .int = 1 } },
            .{ .name = "S_TGRP", .value = .{ .int = 4 } },
            .{ .name = "S_KPH", .value = .{ .int = 1 } },
            .{ .name = "S_NSPLIT", .value = .{ .int = 2 } },
            .{ .name = "S_CHUNK", .value = .{ .int = 512 } },
            .{ .name = "S_FOLD", .value = .{ .int = 0 } },
            .{ .name = "S_NHW", .value = .{ .int = 32 } },
            .{ .name = "S_SB1", .value = .{ .int = 2560 } },
            .{ .name = "S_SB2", .value = .{ .int = 2560 } },
            .{ .name = "S_SB3", .value = .{ .int = 2560 } },
            .{ .name = "S_UNR", .value = .{ .int = 2 } },
            .{ .name = "S_R32", .value = .{ .int = 1 } },
            .{ .name = "S_SW", .value = .{ .int = 1 } },
            .{ .name = "S_GNSPLIT", .value = .{ .int = 2 } },
            .{ .name = "S_GODIM", .value = .{ .int = 1280 } },
            .{ .name = "S_GWIDTH", .value = .{ .int = 640 } },
            .{ .name = "FZ_AX", .value = .{ .int = 80 } },
            .{ .name = "FZ_AY", .value = .{ .int = 10 } },
            .{ .name = "FZ_BX", .value = .{ .int = 20 } },
            .{ .name = "FZ_BY", .value = .{ .int = 2 } },
            .{ .name = "FZ_NA", .value = .{ .int = 800 } },
            .{ .name = "FZ_NB", .value = .{ .int = 40 } },
            .{ .name = "FZ_B_FIRST", .value = .{ .int = 0 } },
        };
        try testing.expect(!catalog.resolve("sushi_rg_fold_down_pair1_w16_32x0_tr_mcg_w16_n32_32", &rg_down_tr_ins, &.{ .float16, .float32 }, &bad_rg_tr, &res));

        // Zero-flag `sortAndFinalize` regression: `fz_mid`, `fz_mid_snax`, and `rg_mid_clamp*` emit `n_flags == 0`.
        const fz_mid_ins = [_]catalog.RuntimeInput{
            .{ .name = "ig", .dtype = .float32, .scalar = false },
            .{ .name = "iu", .dtype = .float32, .scalar = false },
            .{ .name = "svhg", .dtype = .float16, .scalar = false },
            .{ .name = "svhu", .dtype = .float16, .scalar = false },
            .{ .name = "suhd", .dtype = .float16, .scalar = false },
            .{ .name = "slots", .dtype = .uint32, .scalar = false },
            .{ .name = "s_ig", .dtype = .float16, .scalar = false },
            .{ .name = "s_iu", .dtype = .float16, .scalar = false },
            .{ .name = "s_svhg", .dtype = .float16, .scalar = false },
            .{ .name = "s_svhu", .dtype = .float16, .scalar = false },
            .{ .name = "s_suhd", .dtype = .float16, .scalar = false },
            .{ .name = "s_slots", .dtype = .uint32, .scalar = false },
        };
        const fz_mid_targs = [_]catalog.RuntimeTarg{
            .{ .name = "IDIM", .value = .{ .int = 1536 } },
            .{ .name = "NSPLIT", .value = .{ .int = 2 } },
            .{ .name = "S_ODIM", .value = .{ .int = 1536 } },
            .{ .name = "S_CLAMP_LIMIT", .value = .{ .int = 10 } },
            .{ .name = "FZ_AX", .value = .{ .int = 12 } },
            .{ .name = "FZ_AY", .value = .{ .int = 8 } },
            .{ .name = "FZ_BX", .value = .{ .int = 12 } },
            .{ .name = "FZ_BY", .value = .{ .int = 1 } },
            .{ .name = "FZ_NA", .value = .{ .int = 96 } },
            .{ .name = "FZ_NB", .value = .{ .int = 12 } },
            .{ .name = "FZ_B_FIRST", .value = .{ .int = 1 } },
        };
        try testing.expect(catalog.resolve("sushi_exl3_fz_mid_pair1_w16_sspec1_w15_n80_t37f72431_clamp10_tm", &fz_mid_ins, &.{ .float16, .float16 }, &fz_mid_targs, &res));
        try testing.expectEqual(@as(usize, 0), res.n_flags);
        try assertDisjointFcs(&res);
        var fz_mid_snax_ins = fz_mid_ins;
        fz_mid_snax_ins[6].dtype = .float32;
        fz_mid_snax_ins[7].dtype = .float32;
        const fz_mid_snax_targs = [_]catalog.RuntimeTarg{
            .{ .name = "IDIM", .value = .{ .int = 1536 } },
            .{ .name = "NSPLIT", .value = .{ .int = 2 } },
            .{ .name = "S_IDIM", .value = .{ .int = 1536 } },
            .{ .name = "S_NSPLIT", .value = .{ .int = 1 } },
            .{ .name = "FZ_AX", .value = .{ .int = 12 } },
            .{ .name = "FZ_AY", .value = .{ .int = 24 } },
            .{ .name = "FZ_BX", .value = .{ .int = 12 } },
            .{ .name = "FZ_BY", .value = .{ .int = 3 } },
            .{ .name = "FZ_NA", .value = .{ .int = 288 } },
            .{ .name = "FZ_NB", .value = .{ .int = 36 } },
            .{ .name = "FZ_B_FIRST", .value = .{ .int = 1 } },
        };
        try testing.expect(catalog.resolve("sushi_exl3_fz_mid_pair1_w16_sspec1_w15_n80_t37f72431_clamp10_snax_tm", &fz_mid_snax_ins, &.{ .float16, .float16 }, &fz_mid_snax_targs, &res));
        try testing.expectEqual(@as(usize, 0), res.n_flags);
        try assertDisjointFcs(&res);
        const rg_mid_targs = [_]catalog.RuntimeTarg{
            .{ .name = "IDIM", .value = .{ .int = 1536 } },
            .{ .name = "NSPLIT", .value = .{ .int = 2 } },
        };
        try testing.expect(catalog.resolve("sushi_exl3_rg_mid_clamp10", fz_mid_ins[0..6], &.{.float16}, &rg_mid_targs, &res));
        try testing.expectEqual(@as(usize, 0), res.n_flags);
        try assertDisjointFcs(&res);
        try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 640);

        // Wave 1 multi-row H128 prep/finish, fz_prep_snax, and split-proj rg_pair_prepared_g2 resolution + headroom.
        const prep_h128_ins = [_]catalog.RuntimeInput{
            .{ .name = "x", .dtype = .bfloat16 },
            .{ .name = "suh", .dtype = .float16 },
            .{ .name = "slots", .dtype = .uint32 },
        };
        try testing.expect(catalog.resolve("sushi_exl3_prepare_h128", &prep_h128_ins, &.{.float16}, &.{.{ .name = "IDIM", .value = .{ .int = 5120 } }}, &res));
        try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 640);
        const fin_h128_ins = [_]catalog.RuntimeInput{
            .{ .name = "inner", .dtype = .float16 },
            .{ .name = "svh", .dtype = .float16 },
            .{ .name = "slots", .dtype = .uint32 },
        };
        try testing.expect(catalog.resolve("sushi_exl3_finish_h128", &fin_h128_ins, &.{.bfloat16}, &.{ .{ .name = "OutT", .value = .{ .dtype = .bfloat16 } }, .{ .name = "ODIM", .value = .{ .int = 5120 } } }, &res));
        try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 640);
        const fz_prep_ins = [_]catalog.RuntimeInput{
            .{ .name = "x", .dtype = .bfloat16 },
            .{ .name = "suhg", .dtype = .float16 },
            .{ .name = "suhu", .dtype = .float16 },
            .{ .name = "slots", .dtype = .uint32 },
            .{ .name = "s_x", .dtype = .bfloat16 },
            .{ .name = "s_suhg", .dtype = .float16 },
            .{ .name = "s_suhu", .dtype = .float16 },
            .{ .name = "s_slots", .dtype = .uint32 },
        };
        const fz_prep_targs = [_]catalog.RuntimeTarg{
            .{ .name = "IDIM", .value = .{ .int = 5120 } },
            .{ .name = "TOPK", .value = .{ .int = 6 } },
            .{ .name = "S_IDIM", .value = .{ .int = 5120 } },
            .{ .name = "S_TOPK", .value = .{ .int = 1 } },
            .{ .name = "FZ_AX", .value = .{ .int = 40 } },
            .{ .name = "FZ_AY", .value = .{ .int = 18 } },
            .{ .name = "FZ_BX", .value = .{ .int = 40 } },
            .{ .name = "FZ_BY", .value = .{ .int = 3 } },
            .{ .name = "FZ_NA", .value = .{ .int = 1440 } },
            .{ .name = "FZ_NB", .value = .{ .int = 120 } },
            .{ .name = "FZ_B_FIRST", .value = .{ .int = 1 } },
        };
        try testing.expect(catalog.resolve("sushi_exl3_fz_prep_snax", &fz_prep_ins, &.{ .float16, .float16, .float16 }, &fz_prep_targs, &res));
        try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 640);
        const rg_pp_g2_targs = [_]catalog.RuntimeTarg{
            .{ .name = "GROUP", .value = .{ .int = 2 } },
            .{ .name = "OTPT", .value = .{ .int = 2 } },
            .{ .name = "NSPLIT", .value = .{ .int = 2 } },
            .{ .name = "TOPK", .value = .{ .int = 6 } },
            .{ .name = "IDIM", .value = .{ .int = 5120 } },
            .{ .name = "ODIM", .value = .{ .int = 2304 } },
            .{ .name = "FUNNEL", .value = .{ .int = 1 } },
            .{ .name = "NSLOTS", .value = .{ .int = 36 } },
        };
        try testing.expect(catalog.resolve("sushi_exl3_rg_pair_prepared_pair1_w16_g2_24x32_40x56_tm", fz_pair_snax_ins[0..5], &.{ .float32, .float32 }, &rg_pp_g2_targs, &res));
        try assertDisjointFcs(&res);
        try testing.expect((try specializeMaxThreads(dev, l, &res)) >= 640);
    }

    // 2. Live GPU bit-for-bit decode verification on converted fused Part B (SIMD lane/one/shared_rows + NAX snax)
    //    and single-rate Part A (`gated_fold_down`).
    const kernel_mod = @import("kernel.zig");
    const exl3_mod = @import("sushi_exl3");
    const format = exl3_mod.format;

    // Lane-ordered [16, 128] identity prefix: row r places 1.0h at `lane_slot(r)` in k-tile 0.
    var lane_eye128: [16 * 128]u16 = @splat(0);
    var plain_eye128: [16 * 128]u16 = @splat(0);
    for (0..16) |r| {
        const half_i = r >> 3;
        const q = (r & 7) >> 1;
        const p = r & 1;
        lane_eye128[r * 128 + q * 4 + half_i * 2 + p] = 0x3C00; // 1.0h
        plain_eye128[r * 128 + r] = 0x3C00; // 1.0h
    }
    const x_lane16 = try core.fromHost(std.mem.sliceAsBytes(&lane_eye128), .float16, &.{ 16, 128 });
    defer core.release(x_lane16);
    const x_lane6 = try core.fromHost(std.mem.sliceAsBytes(lane_eye128[0 .. 6 * 128]), .float16, &.{ 6, 128 });
    defer core.release(x_lane6);
    const x_lane1 = try core.fromHost(std.mem.sliceAsBytes(lane_eye128[0..128]), .float16, &.{ 1, 128 });
    defer core.release(x_lane1);
    const x_plain16 = try core.fromHost(std.mem.sliceAsBytes(&plain_eye128), .float16, &.{ 16, 128 });
    defer core.release(x_plain16);

    const zeros16_u32: [16]u32 = @splat(0);
    const slots16_t = try core.fromHost(std.mem.sliceAsBytes(&zeros16_u32), .uint32, &.{16});
    defer core.release(slots16_t);
    const meta_dummy_t = try core.fromHost(std.mem.sliceAsBytes(&zeros16_u32), .uint32, &.{16});
    defer core.release(meta_dummy_t);
    const ones128_f16: [128]u16 = @splat(0x3C00);
    const svh128_t = try core.fromHost(std.mem.sliceAsBytes(&ones128_f16), .float16, &.{128});
    defer core.release(svh128_t);

    const readF32 = struct {
        fn f(t: *const Tensor) ![]const f32 {
            const b = try core.hostRead(t);
            return std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(b)));
        }
    }.f;

    var prng = std.Random.DefaultPrng.init(0xF2_E413_01);
    const rnd = prng.random();
    var tr_buf: [64 * 128]u16 = undefined;
    var want_tiles: [8][format.TILE_VALUES]u16 = undefined;
    const fz_in_names = &[_][]const u8{ "middle", "trellis", "slots", "meta", "s_x", "s_trellis", "s_slots" };
    const fz_snax_in_names = &[_][]const u8{ "middle", "trellis", "slots", "meta", "s_x", "s_trellis", "s_svh" };

    const FzCase = struct { a_tag: []const u8, a_cls: []const u8, b_tag: []const u8, b_dig: []const u8, dec: format.Decode, rates: []const u32, full_modes: bool = false };
    const fz_cases = [_]FzCase{
        // All 4 (Part A, Part B) pairings (`ap1_bsc`, `asc_bp1`, `asc_bsc`, `ap1_bp1`) and all 6 Part B codebooks.
        // One representative rate per pairing exercises all 4 FC_I34 SIMD modes + S_UNR={1,2}; remaining rates/codebooks verify _lane + _snax(UNR=2).
        .{ .a_tag = "pair1_w16", .a_cls = "32x0", .b_tag = "spec1_w15", .b_dig = "_t37f72431", .dec = format.Decode.tabled(.spec1, .w15), .rates = &.{ 16, 64, 128 }, .full_modes = true },
        .{ .a_tag = "spec1_w15_t37f72431", .a_cls = "32x0", .b_tag = "pair1_w16", .b_dig = "", .dec = .{ .codebook = .pair1, .window = .w16 }, .rates = &.{ 56, 24, 42, 64 }, .full_modes = true },
        .{ .a_tag = "pair1_w16", .a_cls = "32x0", .b_tag = "mcg_w16", .b_dig = "", .dec = .{ .codebook = .mcg, .window = .w16 }, .rates = &.{64} },
        .{ .a_tag = "spec1_w15_t37f72431", .a_cls = "32x0", .b_tag = "mul1_w16", .b_dig = "", .dec = .{ .codebook = .mul1, .window = .w16 }, .rates = &.{64}, .full_modes = true },
        .{ .a_tag = "spec1_w15_t37f72431", .a_cls = "32x0", .b_tag = "alpha_w16", .b_dig = "", .dec = .{ .codebook = .alpha, .window = .w16 }, .rates = &.{32} },
        .{ .a_tag = "spec1_w15_t37f72431", .a_cls = "32x0", .b_tag = "std1_w15", .b_dig = "_tc5dba171", .dec = format.Decode.tabled(.std1, .w15), .rates = &.{24} },
        .{ .a_tag = "pair1_w16", .a_cls = "32x0", .b_tag = "pair1_w16", .b_dig = "", .dec = .{ .codebook = .pair1, .window = .w16 }, .rates = &.{56}, .full_modes = true },
    };

    for (fz_cases) |fc| {
        for (fc.rates, 0..) |n, ri| {
            const run_all_modes = fc.full_modes and ri == 0;
            const rate: format.Rate = .{ .n = n };
            const n_c: c_int = @intCast(n);
            const slice_8 = tr_buf[0 .. 8 * n];
            for (slice_8) |*w| w.* = rnd.int(u16);
            for (0..8) |ti| format.decodeTile(slice_8[ti * n .. (ti + 1) * n], rate, fc.dec, &want_tiles[ti]);

            // Layout [1, IT=8, OT=1, n] for lane / shared_rows (k=0..7, tile 0 = want_tiles[0])
            const tr_1 = try core.fromHost(std.mem.sliceAsBytes(slice_8), .uint16, &.{ 1, 8, 1, n_c });
            defer core.release(tr_1);
            // Layout [1, IT=8, OT=2, n] for _one (k=0 has tile 0 = want_tiles[0] and tile 1 = want_tiles[1])
            const slice_16 = tr_buf[0 .. 16 * n];
            for (slice_16[8 * n ..]) |*w| w.* = 0;
            const tr_2 = try core.fromHost(std.mem.sliceAsBytes(slice_16), .uint16, &.{ 1, 8, 2, n_c });
            defer core.release(tr_2);

            // (a) Part B `_lane` (FC_I34 = 0)
            {
                const kn = try std.fmt.bufPrint(&nbuf, "sushi_exl3_fz_down_{s}_s{s}_n{d}{s}_{s}_g0", .{ fc.a_tag, fc.b_tag, n, fc.b_dig, fc.a_cls });
                const def = try kernel_mod.Def.create(kn, fz_in_names, &.{ "y", "s_y" }, "", "", true, false);
                var cfg: kernel_mod.Config = .{ .grid = .{ 16 * 128, 1, 1 }, .group = .{ 128, 1, 1 } };
                try cfg.addOutput(&.{ 1, 16 }, .float16);
                try cfg.addOutput(&.{ 16, 16 }, .float16);
                try cfg.addTemplate("OTPT", .{ .int = 1 });
                try cfg.addTemplate("NSPLIT", .{ .int = 1 });
                try cfg.addTemplate("IDIM", .{ .int = 128 });
                try cfg.addTemplate("ODIM", .{ .int = 16 });
                try cfg.addTemplate("FUNNEL", .{ .int = 1 });
                try cfg.addTemplate("S_IDIM", .{ .int = 128 });
                try cfg.addTemplate("S_ODIM", .{ .int = 16 });
                try cfg.addTemplate("S_NHW", .{ .int = n_c });
                try cfg.addTemplate("S_ROWS", .{ .int = 16 });
                try cfg.addTemplate("S_SHARED_ROWS", .{ .int = 2 });
                try cfg.addTemplate("FZ_AX", .{ .int = 1 });
                try cfg.addTemplate("FZ_AY", .{ .int = 1 });
                try cfg.addTemplate("FZ_BX", .{ .int = 1 });
                try cfg.addTemplate("FZ_BY", .{ .int = 16 });
                try cfg.addTemplate("FZ_NA", .{ .int = 0 });
                try cfg.addTemplate("FZ_NB", .{ .int = 16 });
                try cfg.addTemplate("FZ_B_FIRST", .{ .int = 1 });
                var outs: [2]*Tensor = undefined;
                try kernel_mod.launch(def, &.{ x_lane16, tr_1, slots16_t, meta_dummy_t, x_lane16, tr_1, slots16_t }, &cfg, &outs);
                defer core.release(outs[0]);
                defer core.release(outs[1]);
                const got = try readU16(outs[1]);
                try expectMatmulU16(&want_tiles[0], got);
            }

            // (b) Part B `_one` (FC_I34 = 1): 1 row × 2 output tiles (32 cols)
            if (run_all_modes) {
                const kn = try std.fmt.bufPrint(&nbuf, "sushi_exl3_fz_down_{s}_s{s}_n{d}{s}_{s}_g0_one", .{ fc.a_tag, fc.b_tag, n, fc.b_dig, fc.a_cls });
                const def = try kernel_mod.Def.create(kn, fz_in_names, &.{ "y", "s_y" }, "", "", true, false);
                var cfg: kernel_mod.Config = .{ .grid = .{ 128, 1, 1 }, .group = .{ 128, 1, 1 } };
                try cfg.addOutput(&.{ 1, 16 }, .float16);
                try cfg.addOutput(&.{ 1, 32 }, .float16);
                try cfg.addTemplate("OTPT", .{ .int = 1 });
                try cfg.addTemplate("NSPLIT", .{ .int = 1 });
                try cfg.addTemplate("IDIM", .{ .int = 128 });
                try cfg.addTemplate("ODIM", .{ .int = 16 });
                try cfg.addTemplate("FUNNEL", .{ .int = 1 });
                try cfg.addTemplate("S_IDIM", .{ .int = 128 });
                try cfg.addTemplate("S_ODIM", .{ .int = 32 });
                try cfg.addTemplate("S_NHW", .{ .int = n_c });
                try cfg.addTemplate("S_ROWS", .{ .int = 1 });
                try cfg.addTemplate("S_SHARED_ROWS", .{ .int = 2 });
                try cfg.addTemplate("FZ_AX", .{ .int = 1 });
                try cfg.addTemplate("FZ_AY", .{ .int = 1 });
                try cfg.addTemplate("FZ_BX", .{ .int = 1 });
                try cfg.addTemplate("FZ_BY", .{ .int = 1 });
                try cfg.addTemplate("FZ_NA", .{ .int = 0 });
                try cfg.addTemplate("FZ_NB", .{ .int = 1 });
                try cfg.addTemplate("FZ_B_FIRST", .{ .int = 1 });
                var outs: [2]*Tensor = undefined;
                try kernel_mod.launch(def, &.{ x_lane1, tr_2, slots16_t, meta_dummy_t, x_lane1, tr_2, slots16_t }, &cfg, &outs);
                defer core.release(outs[0]);
                defer core.release(outs[1]);
                const got = try readU16(outs[1]);
                try expectMatmulU16(want_tiles[0][0..16], got[0..16]);
                try expectMatmulU16(want_tiles[1][0..16], got[16..32]);
            }

            // (c) Part B `_shared_rows` for both S_SHARED_ROWS=2 (FC_I34=2) and S_SHARED_ROWS=3 (FC_I34=3)
            if (run_all_modes) for ([_]struct { sr: i32, by: i32 }{ .{ .sr = 2, .by = 3 }, .{ .sr = 3, .by = 2 } }) |sr| {
                const kn = try std.fmt.bufPrint(&nbuf, "sushi_exl3_fz_down_{s}_s{s}_n{d}{s}_{s}_g0_shared_rows", .{ fc.a_tag, fc.b_tag, n, fc.b_dig, fc.a_cls });
                const def = try kernel_mod.Def.create(kn, fz_in_names, &.{ "y", "s_y" }, "", "", true, false);
                var cfg: kernel_mod.Config = .{ .grid = .{ sr.by * 128, 1, 1 }, .group = .{ 128, 1, 1 } };
                try cfg.addOutput(&.{ 1, 16 }, .float16);
                try cfg.addOutput(&.{ 6, 16 }, .float16);
                try cfg.addTemplate("OTPT", .{ .int = 1 });
                try cfg.addTemplate("NSPLIT", .{ .int = 1 });
                try cfg.addTemplate("IDIM", .{ .int = 128 });
                try cfg.addTemplate("ODIM", .{ .int = 16 });
                try cfg.addTemplate("FUNNEL", .{ .int = 1 });
                try cfg.addTemplate("S_IDIM", .{ .int = 128 });
                try cfg.addTemplate("S_ODIM", .{ .int = 16 });
                try cfg.addTemplate("S_NHW", .{ .int = n_c });
                try cfg.addTemplate("S_ROWS", .{ .int = 6 });
                try cfg.addTemplate("S_SHARED_ROWS", .{ .int = sr.sr });
                try cfg.addTemplate("FZ_AX", .{ .int = 1 });
                try cfg.addTemplate("FZ_AY", .{ .int = 1 });
                try cfg.addTemplate("FZ_BX", .{ .int = 1 });
                try cfg.addTemplate("FZ_BY", .{ .int = sr.by });
                try cfg.addTemplate("FZ_NA", .{ .int = 0 });
                try cfg.addTemplate("FZ_NB", .{ .int = sr.by });
                try cfg.addTemplate("FZ_B_FIRST", .{ .int = 1 });
                var outs: [2]*Tensor = undefined;
                try kernel_mod.launch(def, &.{ x_lane6, tr_1, slots16_t, meta_dummy_t, x_lane6, tr_1, slots16_t }, &cfg, &outs);
                defer core.release(outs[0]);
                defer core.release(outs[1]);
                const got = try readU16(outs[1]);
                try expectMatmulU16(want_tiles[0][0 .. 6 * 16], got);
            };

            // (d) Part B `_snax` (`exl3_fz_nax`, both S_UNR=1 and S_UNR=2): 16×128 NAX decode matches host bit-for-bit
            if (has_nax) {
                const slice_64 = tr_buf[0 .. 64 * n];
                for (slice_64[8 * n ..]) |*w| w.* = 0;
                const tr_nax = try core.fromHost(std.mem.sliceAsBytes(slice_64), .uint16, &.{ 8, 8, n_c });
                defer core.release(tr_nax);
                const unrs: []const i32 = if (run_all_modes) &.{ 1, 2 } else &.{2};
                for (unrs) |unr| {
                    const kn = try std.fmt.bufPrint(&nbuf, "sushi_exl3_fz_down_{s}_s{s}_n{d}{s}_{s}_g0_snax", .{ fc.a_tag, fc.b_tag, n, fc.b_dig, fc.a_cls });
                    const def = try kernel_mod.Def.create(kn, fz_snax_in_names, &.{ "y", "s_y" }, "", "", true, false);
                    var cfg: kernel_mod.Config = .{ .grid = .{ 128, 1, 1 }, .group = .{ 128, 1, 1 } };
                    try cfg.addOutput(&.{ 1, 16 }, .float16);
                    try cfg.addOutput(&.{ 16, 128 }, .float32);
                    try cfg.addTemplate("OTPT", .{ .int = 1 });
                    try cfg.addTemplate("NSPLIT", .{ .int = 1 });
                    try cfg.addTemplate("IDIM", .{ .int = 128 });
                    try cfg.addTemplate("ODIM", .{ .int = 16 });
                    try cfg.addTemplate("FUNNEL", .{ .int = 1 });
                    try cfg.addTemplate("S_IDIM", .{ .int = 128 });
                    try cfg.addTemplate("S_ODIM", .{ .int = 128 });
                    try cfg.addTemplate("S_ROWS", .{ .int = 16 });
                    try cfg.addTemplate("S_GROUPS", .{ .int = 1 });
                    try cfg.addTemplate("S_TGRP", .{ .int = 4 });
                    try cfg.addTemplate("S_KPH", .{ .int = 1 });
                    try cfg.addTemplate("S_NSPLIT", .{ .int = 1 });
                    try cfg.addTemplate("S_CHUNK", .{ .int = 128 });
                    try cfg.addTemplate("S_FOLD", .{ .int = 0 });
                    try cfg.addTemplate("S_NHW", .{ .int = n_c });
                    try cfg.addTemplate("S_SB1", .{ .int = 128 });
                    try cfg.addTemplate("S_SB2", .{ .int = 128 });
                    try cfg.addTemplate("S_SB3", .{ .int = 128 });
                    try cfg.addTemplate("S_UNR", .{ .int = unr });
                    try cfg.addTemplate("S_R32", .{ .int = 1 });
                    try cfg.addTemplate("S_SW", .{ .int = 1 });
                    try cfg.addTemplate("FZ_AX", .{ .int = 1 });
                    try cfg.addTemplate("FZ_AY", .{ .int = 1 });
                    try cfg.addTemplate("FZ_BX", .{ .int = 1 });
                    try cfg.addTemplate("FZ_BY", .{ .int = 1 });
                    try cfg.addTemplate("FZ_NA", .{ .int = 0 });
                    try cfg.addTemplate("FZ_NB", .{ .int = 1 });
                    try cfg.addTemplate("FZ_B_FIRST", .{ .int = 1 });
                    var outs: [2]*Tensor = undefined;
                    try kernel_mod.launch(def, &.{ x_plain16, tr_1, slots16_t, meta_dummy_t, x_plain16, tr_nax, svh128_t }, &cfg, &outs);
                    defer core.release(outs[0]);
                    defer core.release(outs[1]);
                    const got = try readF32(outs[1]);
                    for (0..16) |r| {
                        for (0..8) |ti| {
                            for (0..16) |c| {
                                const want_f: f32 = @floatCast(@as(f16, @bitCast(want_tiles[ti][r * 16 + c])));
                                try testing.expectEqual(want_f, got[r * 128 + ti * 16 + c]);
                            }
                        }
                    }
                }
            }
        }
    }

    // (e) Part A single-rate `FC_NHW` (`sushi_gated_fold_down`) matches standalone `sushi_exl3_k4_down_fused` bit-for-bit
    //     across both scalar (`spec1`) and `pair1` (direct `fc_pair1_*` constants).
    {
        var ig_buf: [128]f32 = undefined;
        var iu_buf: [128]f32 = undefined;
        for (&ig_buf, &iu_buf) |*g, *u| {
            g.* = (rnd.float(f32) - 0.5) * 2.0;
            u.* = (rnd.float(f32) - 0.5) * 2.0;
        }
        const ig_t = try core.fromHost(std.mem.sliceAsBytes(&ig_buf), .float32, &.{ 1, 128 });
        defer core.release(ig_t);
        const iu_t = try core.fromHost(std.mem.sliceAsBytes(&iu_buf), .float32, &.{ 1, 128 });
        defer core.release(iu_t);
        const slot1_t = try core.fromHost(std.mem.sliceAsBytes(&[_]u32{0}), .uint32, &.{1});
        defer core.release(slot1_t);
        const bf_dummy: [16]u16 = @splat(0);
        const bf_t = try core.fromHost(std.mem.sliceAsBytes(&bf_dummy), .bfloat16, &.{16});
        defer core.release(bf_t);
        const gf_in_names = &[_][]const u8{ "ig", "iu", "trellis", "svhg", "svhu", "suhd", "slots", "s_x", "s_w", "s_sc", "s_bi" };
        const df_in_names = &[_][]const u8{ "ig", "iu", "trellis", "svhg", "svhu", "suhd", "slots" };

        const GfCase = struct { df_suf: []const u8, gf_tag: []const u8, rates: []const u32 };
        for ([_]GfCase{
            .{ .df_suf = "_spec1_w15_t37f72431", .gf_tag = "spec1_w15_t37f72431", .rates = &.{ 16, 64, 128 } },
            .{ .df_suf = "_pair1", .gf_tag = "pair1_w16", .rates = &.{ 24, 56, 64 } },
        }) |gc| {
            const df_def = try kernel_mod.Def.create(try std.fmt.bufPrint(&nbuf, "sushi_exl3_k4_down_fused{s}", .{gc.df_suf}), df_in_names, &.{"y"}, "", "", true, false);
            var nbuf2: [128]u8 = undefined;
            const gf_def = try kernel_mod.Def.create(try std.fmt.bufPrint(&nbuf2, "sushi_gated_fold_down_{s}", .{gc.gf_tag}), gf_in_names, &.{ "y", "s_y" }, "", "", true, false);
            for (gc.rates) |n| {
                const n_c: c_int = @intCast(n);
                const slice = tr_buf[0 .. 8 * n];
                for (slice) |*w| w.* = rnd.int(u16);
                const tr_t = try core.fromHost(std.mem.sliceAsBytes(slice), .uint16, &.{ 1, 8, 1, n_c });
                defer core.release(tr_t);

                var df_cfg: kernel_mod.Config = .{ .grid = .{ 128, 1, 1 }, .group = .{ 128, 1, 1 } };
                try df_cfg.addOutput(&.{ 1, 16 }, .float16);
                try df_cfg.addTemplate("OTPT", .{ .int = 1 });
                try df_cfg.addTemplate("IDIM", .{ .int = 128 });
                try df_cfg.addTemplate("ODIM", .{ .int = 16 });
                try df_cfg.addTemplate("NSPLIT", .{ .int = 1 });
                try df_cfg.addTemplate("NHW", .{ .int = n_c });
                try df_cfg.addTemplate("FUNNEL", .{ .int = 1 });
                var df_out: [1]*Tensor = undefined;
                try kernel_mod.launch(df_def, &.{ ig_t, iu_t, tr_t, svh128_t, svh128_t, svh128_t, slot1_t }, &df_cfg, &df_out);
                defer core.release(df_out[0]);

                var gf_cfg: kernel_mod.Config = .{ .grid = .{ 128, 1, 1 }, .group = .{ 128, 1, 1 } };
                try gf_cfg.addOutput(&.{ 1, 16 }, .float16);
                try gf_cfg.addOutput(&.{ 1, 16 }, .bfloat16);
                try gf_cfg.addTemplate("OTPT", .{ .int = 1 });
                try gf_cfg.addTemplate("IDIM", .{ .int = 128 });
                try gf_cfg.addTemplate("ODIM", .{ .int = 16 });
                try gf_cfg.addTemplate("NSPLIT", .{ .int = 1 });
                try gf_cfg.addTemplate("NHW", .{ .int = n_c });
                try gf_cfg.addTemplate("FUNNEL", .{ .int = 1 });
                try gf_cfg.addTemplate("S_K", .{ .int = 128 });
                try gf_cfg.addTemplate("S_N", .{ .int = 16 });
                try gf_cfg.addTemplate("S_GS", .{ .int = 64 });
                try gf_cfg.addTemplate("S_FAST", .{ .int = 0 });
                try gf_cfg.addTemplate("FZ_AX", .{ .int = 1 });
                try gf_cfg.addTemplate("FZ_AY", .{ .int = 1 });
                try gf_cfg.addTemplate("FZ_BX", .{ .int = 1 });
                try gf_cfg.addTemplate("FZ_BY", .{ .int = 1 });
                try gf_cfg.addTemplate("FZ_NA", .{ .int = 1 });
                try gf_cfg.addTemplate("FZ_NB", .{ .int = 0 });
                try gf_cfg.addTemplate("FZ_B_FIRST", .{ .int = 0 });
                var gf_out: [2]*Tensor = undefined;
                try kernel_mod.launch(gf_def, &.{ ig_t, iu_t, tr_t, svh128_t, svh128_t, svh128_t, slot1_t, bf_t, slots16_t, bf_t, bf_t }, &gf_cfg, &gf_out);
                defer core.release(gf_out[0]);
                defer core.release(gf_out[1]);

                const want_u16 = try readU16(df_out[0]);
                const got_u16 = try readU16(gf_out[0]);
                try testing.expectEqualSlices(u16, want_u16, got_u16);
            }
        }
    }

    // (f) Simultaneous Part A + Part B (`FZ_NA > 0 && FZ_NB > 0`) live GPU bit-exact decode on BOTH
    //     cross-pairings (`ap1_bsc` and `asc_bp1`) for `fz_down` (SIMD), `fz_down_snax` (NAX), and `fz_pair` (SIMD),
    //     proving zero Part A / Part B FC or `EXL3_LANE_FIRST` / `FC_B15` cross-contamination in the same dispatch.
    {
        var prep_lane32: [32 * 128]u16 = @splat(0);
        for (0..16) |r| {
            @memcpy(prep_lane32[(2 * r) * 128 .. (2 * r + 1) * 128], lane_eye128[r * 128 .. (r + 1) * 128]);
            @memcpy(prep_lane32[(2 * r + 1) * 128 .. (2 * r + 2) * 128], lane_eye128[r * 128 .. (r + 1) * 128]);
        }
        const prep_lane32_t = try core.fromHost(std.mem.sliceAsBytes(&prep_lane32), .float16, &.{ 32, 128 });
        defer core.release(prep_lane32_t);

        const CrossCase = struct {
            a_tag: []const u8,
            a_dec: format.Decode,
            a_n_g: u32,
            a_n_u: u32,
            b_tag: []const u8,
            b_dig: []const u8,
            b_dec: format.Decode,
            b_n: u32,
        };
        const cross_cases = [_]CrossCase{
            // Part A = PAIR1 (`EXL3_LANE_FIRST=1`), Part B = scalar (`FC_B15=false`)
            .{ .a_tag = "pair1_w16", .a_dec = .{ .codebook = .pair1, .window = .w16 }, .a_n_g = 40, .a_n_u = 56, .b_tag = "spec1_w15", .b_dig = "_t37f72431", .b_dec = format.Decode.tabled(.spec1, .w15), .b_n = 34 },
            // Part A = scalar (`EXL3_LANE_FIRST=0`), Part B = PAIR1 (`FC_B15=true`)
            .{ .a_tag = "spec1_w15_t37f72431", .a_dec = format.Decode.tabled(.spec1, .w15), .a_n_g = 34, .a_n_u = 80, .b_tag = "pair1_w16", .b_dig = "", .b_dec = .{ .codebook = .pair1, .window = .w16 }, .b_n = 42 },
        };
        const fz_pair_in_names = &[_][]const u8{ "prep", "tg", "tu", "slots", "meta", "s_xg", "s_xu", "s_tg", "s_tu", "s_slots" };

        var tr_ag_buf: [8 * 128]u16 = undefined;
        var tr_au_buf: [8 * 128]u16 = undefined;
        var tr_bg_buf: [64 * 128]u16 = undefined;
        var tr_bu_buf: [8 * 128]u16 = undefined;
        var want_ag: [format.TILE_VALUES]u16 = undefined;
        var want_au: [format.TILE_VALUES]u16 = undefined;
        var want_bg: [8][format.TILE_VALUES]u16 = undefined;
        var want_bu: [format.TILE_VALUES]u16 = undefined;

        for (cross_cases) |cc| {
            const ag_slice = tr_ag_buf[0 .. 8 * cc.a_n_g];
            const au_slice = tr_au_buf[0 .. 8 * cc.a_n_u];
            const bg_slice = tr_bg_buf[0 .. 8 * cc.b_n];
            const bu_slice = tr_bu_buf[0 .. 8 * cc.b_n];
            for (ag_slice) |*w| w.* = rnd.int(u16);
            for (au_slice) |*w| w.* = rnd.int(u16);
            for (bg_slice) |*w| w.* = rnd.int(u16);
            for (bu_slice) |*w| w.* = rnd.int(u16);

            format.decodeTile(ag_slice[0..cc.a_n_g], .{ .n = cc.a_n_g }, cc.a_dec, &want_ag);
            format.decodeTile(au_slice[0..cc.a_n_u], .{ .n = cc.a_n_u }, cc.a_dec, &want_au);
            for (0..8) |ti| format.decodeTile(bg_slice[ti * cc.b_n .. (ti + 1) * cc.b_n], .{ .n = cc.b_n }, cc.b_dec, &want_bg[ti]);
            format.decodeTile(bu_slice[0..cc.b_n], .{ .n = cc.b_n }, cc.b_dec, &want_bu);

            const tr_ag = try core.fromHost(std.mem.sliceAsBytes(ag_slice), .uint16, &.{ 1, 8, 1, @as(c_int, @intCast(cc.a_n_g)) });
            defer core.release(tr_ag);
            const tr_au = try core.fromHost(std.mem.sliceAsBytes(au_slice), .uint16, &.{ 1, 8, 1, @as(c_int, @intCast(cc.a_n_u)) });
            defer core.release(tr_au);
            const tr_bg = try core.fromHost(std.mem.sliceAsBytes(bg_slice), .uint16, &.{ 1, 8, 1, @as(c_int, @intCast(cc.b_n)) });
            defer core.release(tr_bg);
            const tr_bu = try core.fromHost(std.mem.sliceAsBytes(bu_slice), .uint16, &.{ 1, 8, 1, @as(c_int, @intCast(cc.b_n)) });
            defer core.release(tr_bu);

            const meta_words = [_]u32{ cc.a_n_g, cc.a_n_u, cc.a_n_g, 0, 0, 0, 0, 0 };
            const meta_t = try core.fromHost(std.mem.sliceAsBytes(&meta_words), .uint32, &.{8});
            defer core.release(meta_t);

            for ([_]i32{ 0, 1 }) |b_first| {
                // (f.1) Simultaneous `fz_down` (SIMD): Part A [16,16] f16 + Part B [16,16] f16
                {
                    const kn = try std.fmt.bufPrint(&nbuf, "sushi_exl3_fz_down_{s}_s{s}_n{d}{s}_{d}x0_g0", .{ cc.a_tag, cc.b_tag, cc.b_n, cc.b_dig, cc.a_n_g });
                    const def = try kernel_mod.Def.create(kn, fz_in_names, &.{ "y", "s_y" }, "", "", true, false);
                    var cfg: kernel_mod.Config = .{ .grid = .{ (16 + 16) * 128, 1, 1 }, .group = .{ 128, 1, 1 } };
                    try cfg.addOutput(&.{ 16, 16 }, .float16);
                    try cfg.addOutput(&.{ 16, 16 }, .float16);
                    try cfg.addTemplate("OTPT", .{ .int = 1 });
                    try cfg.addTemplate("NSPLIT", .{ .int = 1 });
                    try cfg.addTemplate("IDIM", .{ .int = 128 });
                    try cfg.addTemplate("ODIM", .{ .int = 16 });
                    try cfg.addTemplate("FUNNEL", .{ .int = 1 });
                    try cfg.addTemplate("S_IDIM", .{ .int = 128 });
                    try cfg.addTemplate("S_ODIM", .{ .int = 16 });
                    try cfg.addTemplate("S_NHW", .{ .int = @as(c_int, @intCast(cc.b_n)) });
                    try cfg.addTemplate("S_ROWS", .{ .int = 16 });
                    try cfg.addTemplate("S_SHARED_ROWS", .{ .int = 2 });
                    try cfg.addTemplate("FZ_AX", .{ .int = 1 });
                    try cfg.addTemplate("FZ_AY", .{ .int = 16 });
                    try cfg.addTemplate("FZ_BX", .{ .int = 1 });
                    try cfg.addTemplate("FZ_BY", .{ .int = 16 });
                    try cfg.addTemplate("FZ_NA", .{ .int = 16 });
                    try cfg.addTemplate("FZ_NB", .{ .int = 16 });
                    try cfg.addTemplate("FZ_B_FIRST", .{ .int = b_first });
                    var outs: [2]*Tensor = undefined;
                    try kernel_mod.launch(def, &.{ x_lane16, tr_ag, slots16_t, meta_t, x_lane16, tr_bg, slots16_t }, &cfg, &outs);
                    defer core.release(outs[0]);
                    defer core.release(outs[1]);
                    try expectMatmulU16(&want_ag, try readU16(outs[0]));
                    try expectMatmulU16(&want_bg[0], try readU16(outs[1]));
                }

                // (f.2) Simultaneous `fz_pair` (SIMD): Part A gate+up (split along z, different rates via FC_I46/FC_I47) + Part B gate+up
                {
                    const kn = try std.fmt.bufPrint(&nbuf, "sushi_exl3_fz_pair_{s}_s{s}_n{d}{s}_{d}x{d}_g0", .{ cc.a_tag, cc.b_tag, cc.b_n, cc.b_dig, cc.a_n_g, cc.a_n_u });
                    const def = try kernel_mod.Def.create(kn, fz_pair_in_names, &.{ "yg", "yu", "s_yg", "s_yu" }, "", "", true, false);
                    var cfg: kernel_mod.Config = .{ .grid = .{ (32 + 32) * 128, 1, 1 }, .group = .{ 128, 1, 1 } };
                    try cfg.addOutput(&.{ 16, 16 }, .float32);
                    try cfg.addOutput(&.{ 16, 16 }, .float32);
                    try cfg.addOutput(&.{ 16, 16 }, .float16);
                    try cfg.addOutput(&.{ 16, 16 }, .float16);
                    try cfg.addTemplate("OTPT", .{ .int = 1 });
                    try cfg.addTemplate("NSPLIT", .{ .int = 1 });
                    try cfg.addTemplate("TOPK", .{ .int = 1 });
                    try cfg.addTemplate("IDIM", .{ .int = 128 });
                    try cfg.addTemplate("ODIM", .{ .int = 16 });
                    try cfg.addTemplate("FUNNEL", .{ .int = 1 });
                    try cfg.addTemplate("S_IDIM", .{ .int = 128 });
                    try cfg.addTemplate("S_ODIM", .{ .int = 16 });
                    try cfg.addTemplate("S_NHW", .{ .int = @as(c_int, @intCast(cc.b_n)) });
                    try cfg.addTemplate("S_ROWS", .{ .int = 16 });
                    try cfg.addTemplate("S_SHARED_ROWS", .{ .int = 2 });
                    try cfg.addTemplate("FZ_AX", .{ .int = 1 });
                    try cfg.addTemplate("FZ_AY", .{ .int = 16 });
                    try cfg.addTemplate("FZ_BX", .{ .int = 1 });
                    try cfg.addTemplate("FZ_BY", .{ .int = 16 });
                    try cfg.addTemplate("FZ_NA", .{ .int = 32 });
                    try cfg.addTemplate("FZ_NB", .{ .int = 32 });
                    try cfg.addTemplate("FZ_B_FIRST", .{ .int = b_first });
                    var outs: [4]*Tensor = undefined;
                    try kernel_mod.launch(def, &.{ prep_lane32_t, tr_ag, tr_au, slots16_t, meta_t, x_lane16, x_lane16, tr_bg, tr_bu, slots16_t }, &cfg, &outs);
                    defer core.release(outs[0]);
                    defer core.release(outs[1]);
                    defer core.release(outs[2]);
                    defer core.release(outs[3]);
                    const got_ag = try readF32(outs[0]);
                    const got_au = try readF32(outs[1]);
                    for (0..format.TILE_VALUES) |idx| {
                        try testing.expectEqual(@as(f32, @floatCast(@as(f16, @bitCast(want_ag[idx])))), got_ag[idx]);
                        try testing.expectEqual(@as(f32, @floatCast(@as(f16, @bitCast(want_au[idx])))), got_au[idx]);
                    }
                    try expectMatmulU16(&want_bg[0], try readU16(outs[2]));
                    try expectMatmulU16(&want_bu, try readU16(outs[3]));
                }

                // (f.3) Simultaneous `fz_down_snax` (NAX): Part A [16,16] f16 + Part B [16,128] f32
                if (has_nax) {
                    const bg_nax_slice = tr_bg_buf[0 .. 64 * cc.b_n];
                    for (bg_nax_slice[8 * cc.b_n ..]) |*w| w.* = 0;
                    const tr_bg_nax = try core.fromHost(std.mem.sliceAsBytes(bg_nax_slice), .uint16, &.{ 8, 8, @as(c_int, @intCast(cc.b_n)) });
                    defer core.release(tr_bg_nax);
                    const kn = try std.fmt.bufPrint(&nbuf, "sushi_exl3_fz_down_{s}_s{s}_n{d}{s}_{d}x0_g0_snax", .{ cc.a_tag, cc.b_tag, cc.b_n, cc.b_dig, cc.a_n_g });
                    const def = try kernel_mod.Def.create(kn, fz_snax_in_names, &.{ "y", "s_y" }, "", "", true, false);
                    var cfg: kernel_mod.Config = .{ .grid = .{ (16 + 1) * 128, 1, 1 }, .group = .{ 128, 1, 1 } };
                    try cfg.addOutput(&.{ 16, 16 }, .float16);
                    try cfg.addOutput(&.{ 16, 128 }, .float32);
                    try cfg.addTemplate("OTPT", .{ .int = 1 });
                    try cfg.addTemplate("NSPLIT", .{ .int = 1 });
                    try cfg.addTemplate("IDIM", .{ .int = 128 });
                    try cfg.addTemplate("ODIM", .{ .int = 16 });
                    try cfg.addTemplate("FUNNEL", .{ .int = 1 });
                    try cfg.addTemplate("S_IDIM", .{ .int = 128 });
                    try cfg.addTemplate("S_ODIM", .{ .int = 128 });
                    try cfg.addTemplate("S_ROWS", .{ .int = 16 });
                    try cfg.addTemplate("S_GROUPS", .{ .int = 1 });
                    try cfg.addTemplate("S_TGRP", .{ .int = 4 });
                    try cfg.addTemplate("S_KPH", .{ .int = 1 });
                    try cfg.addTemplate("S_NSPLIT", .{ .int = 1 });
                    try cfg.addTemplate("S_CHUNK", .{ .int = 128 });
                    try cfg.addTemplate("S_FOLD", .{ .int = 0 });
                    try cfg.addTemplate("S_NHW", .{ .int = @as(c_int, @intCast(cc.b_n)) });
                    try cfg.addTemplate("S_SB1", .{ .int = 128 });
                    try cfg.addTemplate("S_SB2", .{ .int = 128 });
                    try cfg.addTemplate("S_SB3", .{ .int = 128 });
                    try cfg.addTemplate("S_UNR", .{ .int = 2 });
                    try cfg.addTemplate("S_R32", .{ .int = 1 });
                    try cfg.addTemplate("S_SW", .{ .int = 1 });
                    try cfg.addTemplate("FZ_AX", .{ .int = 1 });
                    try cfg.addTemplate("FZ_AY", .{ .int = 16 });
                    try cfg.addTemplate("FZ_BX", .{ .int = 1 });
                    try cfg.addTemplate("FZ_BY", .{ .int = 1 });
                    try cfg.addTemplate("FZ_NA", .{ .int = 16 });
                    try cfg.addTemplate("FZ_NB", .{ .int = 1 });
                    try cfg.addTemplate("FZ_B_FIRST", .{ .int = b_first });
                    var outs: [2]*Tensor = undefined;
                    try kernel_mod.launch(def, &.{ x_lane16, tr_ag, slots16_t, meta_t, x_plain16, tr_bg_nax, svh128_t }, &cfg, &outs);
                    defer core.release(outs[0]);
                    defer core.release(outs[1]);
                    try expectMatmulU16(&want_ag, try readU16(outs[0]));
                    const got_b = try readF32(outs[1]);
                    for (0..16) |r| {
                        for (0..8) |ti| {
                            for (0..16) |c| {
                                const want_f: f32 = @floatCast(@as(f16, @bitCast(want_bg[ti][r * 16 + c])));
                                try testing.expectEqual(want_f, got_b[r * 128 + ti * 16 + c]);
                            }
                        }
                    }
                }

                // (f.4) Simultaneous `fz_pair_snax` (NAX): Part A split-proj asymmetric gate!=up + Part B ROWS_NAX_PREPPED_SOURCE
                if (has_nax) {
                    const bg_nax_slice = tr_bg_buf[0 .. 64 * cc.b_n];
                    for (bg_nax_slice[8 * cc.b_n ..]) |*w| w.* = 0;
                    const tr_bg_nax = try core.fromHost(std.mem.sliceAsBytes(bg_nax_slice), .uint16, &.{ 8, 8, @as(c_int, @intCast(cc.b_n)) });
                    defer core.release(tr_bg_nax);
                    var bu_nax_buf: [64 * 128]u16 = @splat(0);
                    const bu_nax_slice = bu_nax_buf[0 .. 64 * cc.b_n];
                    @memcpy(bu_nax_slice[0 .. cc.b_n], bu_slice[0 .. cc.b_n]);
                    const tr_bu_nax = try core.fromHost(std.mem.sliceAsBytes(bu_nax_slice), .uint16, &.{ 8, 8, @as(c_int, @intCast(cc.b_n)) });
                    defer core.release(tr_bu_nax);
                    const fz_pair_snax_names = &[_][]const u8{ "prep", "tg", "tu", "slots", "meta", "s_xg", "s_xu", "s_tg", "s_tu" };
                    const kn = try std.fmt.bufPrint(&nbuf, "sushi_exl3_fz_pair_{s}_s{s}_n{d}{s}_{d}x{d}_g0_snax", .{ cc.a_tag, cc.b_tag, cc.b_n, cc.b_dig, cc.a_n_g, cc.a_n_u });
                    const def = try kernel_mod.Def.create(kn, fz_pair_snax_names, &.{ "yg", "yu", "s_yg", "s_yu" }, "", "", true, false);
                    var cfg: kernel_mod.Config = .{ .grid = .{ (32 + 2) * 128, 1, 1 }, .group = .{ 128, 1, 1 } };
                    try cfg.addOutput(&.{ 16, 16 }, .float32);
                    try cfg.addOutput(&.{ 16, 16 }, .float32);
                    try cfg.addOutput(&.{ 16, 128 }, .float32);
                    try cfg.addOutput(&.{ 16, 128 }, .float32);
                    try cfg.addTemplate("OTPT", .{ .int = 1 });
                    try cfg.addTemplate("NSPLIT", .{ .int = 1 });
                    try cfg.addTemplate("TOPK", .{ .int = 1 });
                    try cfg.addTemplate("IDIM", .{ .int = 128 });
                    try cfg.addTemplate("ODIM", .{ .int = 16 });
                    try cfg.addTemplate("FUNNEL", .{ .int = 1 });
                    try cfg.addTemplate("S_IDIM", .{ .int = 128 });
                    try cfg.addTemplate("S_ODIM", .{ .int = 128 });
                    try cfg.addTemplate("S_ROWS", .{ .int = 16 });
                    try cfg.addTemplate("S_GROUPS", .{ .int = 1 });
                    try cfg.addTemplate("S_TGRP", .{ .int = 4 });
                    try cfg.addTemplate("S_KPH", .{ .int = 1 });
                    try cfg.addTemplate("S_NSPLIT", .{ .int = 1 });
                    try cfg.addTemplate("S_CHUNK", .{ .int = 128 });
                    try cfg.addTemplate("S_FOLD", .{ .int = 0 });
                    try cfg.addTemplate("S_NHW", .{ .int = @as(c_int, @intCast(cc.b_n)) });
                    try cfg.addTemplate("S_SB1", .{ .int = 128 });
                    try cfg.addTemplate("S_SB2", .{ .int = 128 });
                    try cfg.addTemplate("S_SB3", .{ .int = 128 });
                    try cfg.addTemplate("S_UNR", .{ .int = 2 });
                    try cfg.addTemplate("S_R32", .{ .int = 1 });
                    try cfg.addTemplate("S_SW", .{ .int = 1 });
                    try cfg.addTemplate("FZ_AX", .{ .int = 1 });
                    try cfg.addTemplate("FZ_AY", .{ .int = 16 });
                    try cfg.addTemplate("FZ_BX", .{ .int = 1 });
                    try cfg.addTemplate("FZ_BY", .{ .int = 1 });
                    try cfg.addTemplate("FZ_NA", .{ .int = 32 });
                    try cfg.addTemplate("FZ_NB", .{ .int = 2 });
                    try cfg.addTemplate("FZ_B_FIRST", .{ .int = b_first });
                    var outs: [4]*Tensor = undefined;
                    try kernel_mod.launch(def, &.{ prep_lane32_t, tr_ag, tr_au, slots16_t, meta_t, x_plain16, x_plain16, tr_bg_nax, tr_bu_nax }, &cfg, &outs);
                    defer for (outs) |o| core.release(o);
                    const got_ag = try readF32(outs[0]);
                    const got_au = try readF32(outs[1]);
                    for (0..format.TILE_VALUES) |idx| {
                        try testing.expectEqual(@as(f32, @floatCast(@as(f16, @bitCast(want_ag[idx])))), got_ag[idx]);
                        try testing.expectEqual(@as(f32, @floatCast(@as(f16, @bitCast(want_au[idx])))), got_au[idx]);
                    }
                    const got_bg = try readF32(outs[2]);
                    const got_bu = try readF32(outs[3]);
                    for (0..16) |r| {
                        for (0..16) |c| {
                            try testing.expectEqual(@as(f32, @floatCast(@as(f16, @bitCast(want_bg[0][r * 16 + c])))), got_bg[r * 128 + c]);
                            try testing.expectEqual(@as(f32, @floatCast(@as(f16, @bitCast(want_bu[r * 16 + c])))), got_bu[r * 128 + c]);
                        }
                    }
                }
            }
        }
    }

    // (g) Wave 1 boundary GPU bit-exact checks:
    //   (g.1) 128-thread `prepare_h128` / `finish_h128` at unaligned rows (1, 31, 33) with mid-SG & tail `eid` switches;
    //   (g.2) `rg_pair_prepared_g2` with `NSPLIT=2` (`tgs.z=4`), asymmetric `gate!=up` (`40x56`), and `tile_major=true`;
    //   (g.3) `fz_mid_snax` Part A lane-ordered (`DECODE_MID_SOURCE`) vs Part B linear (`DECODE_MID_LINEAR_SOURCE`);
    //   (g.4) `fz_prep_snax` Part A lane-ordered (`PAIR_PREP_SOURCE`) vs Part B linear (`SHARED_PREPARE_SOURCE`).
    {
        const laneSlot = struct {
            fn f(i: usize) usize {
                return (i & ~@as(usize, 15)) + ((i & 7) >> 1) * 4 + ((i & 15) >> 3) * 2 + (i & 1);
            }
        }.f;

        // (g.1) 128-thread prepare_h128 & finish_h128 across rows = 1, 31, 33 and 3 experts with eid transitions.
        const prep_def = try kernel_mod.Def.create("sushi_exl3_prepare_h128", &.{ "x", "suh", "slots" }, &.{"y"}, exl3_mod.kernels.PREPARE_SOURCE, "", true, false);
        const fin_def = try kernel_mod.Def.create("sushi_exl3_finish_h128", &.{ "inner", "svh", "slots" }, &.{"y"}, exl3_mod.kernels.FINISH_SOURCE, "", true, false);
        var suh3: [3 * 128]u16 = undefined;
        var svh3: [3 * 128]u16 = undefined;
        for (&suh3, &svh3) |*a, *b| {
            a.* = format.f32ToF16Bits((rnd.float(f32) - 0.5) * 2.0);
            b.* = format.f32ToF16Bits((rnd.float(f32) - 0.5) * 2.0);
        }
        const suh3_t = try core.fromHost(std.mem.sliceAsBytes(&suh3), .float16, &.{ 3, 128 });
        defer core.release(suh3_t);
        const svh3_t = try core.fromHost(std.mem.sliceAsBytes(&svh3), .float16, &.{ 3, 128 });
        defer core.release(svh3_t);
        for ([_]usize{ 1, 31, 33 }) |rows| {
            var xh: [33 * 128]u16 = undefined;
            var slh: [33]u32 = undefined;
            for (xh[0 .. rows * 128]) |*v| v.* = format.f32ToF16Bits((rnd.float(f32) - 0.5) * 4.0);
            // Force repeated and changing eids across SGs (stride 4) and tail row 32.
            for (slh[0..rows], 0..) |*e, r| e.* = @intCast(if (r < 8) 0 else if (r == 32) 2 else ((r / 3) % 3));
            const x_t = try core.fromHost(std.mem.sliceAsBytes(xh[0 .. rows * 128]), .float16, &.{ @intCast(rows), 128 });
            defer core.release(x_t);
            const sl_t = try core.fromHost(std.mem.sliceAsBytes(slh[0..rows]), .uint32, &.{@intCast(rows)});
            defer core.release(sl_t);
            const tgy: c_int = @intCast((rows + 31) / 32);
            var pcfg: kernel_mod.Config = .{ .grid = .{ 128, tgy, 1 }, .group = .{ 128, 1, 1 } };
            try pcfg.addOutput(&.{ @intCast(rows), 128 }, .float16);
            try pcfg.addTemplate("IDIM", .{ .int = 128 });
            var pout: [1]*Tensor = undefined;
            try kernel_mod.launch(prep_def, &.{ x_t, suh3_t, sl_t }, &pcfg, &pout);
            defer core.release(pout[0]);
            const got_p = try readU16(pout[0]);

            var fcfg: kernel_mod.Config = .{ .grid = .{ 128, tgy, 1 }, .group = .{ 128, 1, 1 } };
            try fcfg.addOutput(&.{ @intCast(rows), 128 }, .float32);
            try fcfg.addTemplate("OutT", .{ .dtype = .float32 });
            try fcfg.addTemplate("ODIM", .{ .int = 128 });
            var fout: [1]*Tensor = undefined;
            try kernel_mod.launch(fin_def, &.{ pout[0], svh3_t, sl_t }, &fcfg, &fout);
            defer core.release(fout[0]);
            const got_f = try readF32(fout[0]);

            for (0..rows) |r| {
                const eid = slh[r];
                var hv: [128]f32 = undefined;
                for (0..128) |j| hv[j] = format.f16BitsToF32(xh[r * 128 + j]) * format.f16BitsToF32(suh3[eid * 128 + j]);
                format.hadamard128(&hv);
                var hf: [128]f32 = undefined;
                for (0..128) |j| {
                    try testing.expect(@abs(hv[j] - format.f16BitsToF32(got_p[r * 128 + j])) <= 2e-2);
                    hf[j] = format.f16BitsToF32(got_p[r * 128 + j]);
                }
                format.hadamard128(&hf);
                for (0..128) |j| {
                    const want_f = hf[j] * format.f16BitsToF32(svh3[eid * 128 + j]);
                    try testing.expect(@abs(want_f - got_f[r * 128 + j]) <= 1e-4);
                }
            }
        }

        // (g.2) `rg_pair_prepared_g2` with `NSPLIT=2` (`tgs.z=4`), asymmetric `40x56`, and `tile_major=true`.
        {
            const n_g: u32 = 40;
            const n_u: u32 = 56;
            const p1_dec: format.Decode = .{ .codebook = .pair1, .window = .w16 };
            // IT=16 (IDIM=256, NSPLIT=2 -> 8 k-tiles/split), OT=2 (ODIM=32, OTPT=2 -> 1 tile_x).
            var tg_pack: [16 * 2 * n_g]u16 = undefined;
            var tu_pack: [16 * 2 * n_u]u16 = undefined;
            for (&tg_pack) |*w| w.* = rnd.int(u16);
            for (&tu_pack) |*w| w.* = rnd.int(u16);
            var tg_tm: [16 * 2 * n_g]u16 = undefined;
            var tu_tm: [16 * 2 * n_u]u16 = undefined;
            exl3_mod.group_layout.tileMajor(u16, &tg_tm, &tg_pack, 16, 2, n_g);
            exl3_mod.group_layout.tileMajor(u16, &tu_tm, &tu_pack, 16, 2, n_u);
            var want_g0: [2][format.TILE_VALUES]u16 = undefined;
            var want_g8: [2][format.TILE_VALUES]u16 = undefined;
            var want_u0: [2][format.TILE_VALUES]u16 = undefined;
            var want_u8: [2][format.TILE_VALUES]u16 = undefined;
            for (0..2) |o| {
                format.decodeTile(tg_pack[(0 * 2 + o) * n_g ..][0..n_g], .{ .n = n_g }, p1_dec, &want_g0[o]);
                format.decodeTile(tg_pack[(8 * 2 + o) * n_g ..][0..n_g], .{ .n = n_g }, p1_dec, &want_g8[o]);
                format.decodeTile(tu_pack[(0 * 2 + o) * n_u ..][0..n_u], .{ .n = n_u }, p1_dec, &want_u0[o]);
                format.decodeTile(tu_pack[(8 * 2 + o) * n_u ..][0..n_u], .{ .n = n_u }, p1_dec, &want_u8[o]);
            }
            // 2 slots routed to expert 0 (exercises `g2` member-pair rung): slot 0 picks row 3, slot 1 picks row 11 in k-tile 0 and k-tile 8.
            var prep_buf: [4 * 256]u16 = @splat(0);
            const rows_pick = [2]usize{ 3, 11 };
            for (rows_pick, 0..) |r, slot| {
                for (0..2) |pj| {
                    prep_buf[(slot * 2 + pj) * 256 + laneSlot(r)] = 0x3C00; // split 0, k-tile 0, row r
                    prep_buf[(slot * 2 + pj) * 256 + 8 * 16 + laneSlot(r)] = 0x3C00; // split 1, k-tile 8, row r
                }
            }
            const prep_t = try core.fromHost(std.mem.sliceAsBytes(&prep_buf), .float16, &.{ 4, 256 });
            defer core.release(prep_t);
            const tg_t = try core.fromHost(std.mem.sliceAsBytes(&tg_tm), .uint16, &.{ 1, 16, 2, n_g });
            defer core.release(tg_t);
            const tu_t = try core.fromHost(std.mem.sliceAsBytes(&tu_tm), .uint16, &.{ 1, 16, 2, n_u });
            defer core.release(tu_t);
            const sl2_t = try core.fromHost(std.mem.sliceAsBytes(&[_]u32{ 0, 0 }), .uint32, &.{2});
            defer core.release(sl2_t);
            const m_t = try core.fromHost(std.mem.sliceAsBytes(&[_]u32{ n_g, n_u, n_g, 0, 0, 0, 0, 0 }), .uint32, &.{8});
            defer core.release(m_t);

            const rg_def = try kernel_mod.Def.create("sushi_exl3_rg_pair_prepared_pair1_w16_g2_40x56_tm", &.{ "prep", "tg", "tu", "slots", "meta" }, &.{ "yg", "yu" }, "", "", true, false);
            var rg_cfg: kernel_mod.Config = .{ .grid = .{ 128, 2, 4 }, .group = .{ 128, 1, 1 } };
            try rg_cfg.addOutput(&.{ 4, 32 }, .float32);
            try rg_cfg.addOutput(&.{ 4, 32 }, .float32);
            try rg_cfg.addTemplate("GROUP", .{ .int = 2 });
            try rg_cfg.addTemplate("OTPT", .{ .int = 2 });
            try rg_cfg.addTemplate("NSPLIT", .{ .int = 2 });
            try rg_cfg.addTemplate("TOPK", .{ .int = 1 });
            try rg_cfg.addTemplate("IDIM", .{ .int = 256 });
            try rg_cfg.addTemplate("ODIM", .{ .int = 32 });
            try rg_cfg.addTemplate("FUNNEL", .{ .int = 1 });
            try rg_cfg.addTemplate("NSLOTS", .{ .int = 2 });
            var rg_out: [2]*Tensor = undefined;
            try kernel_mod.launch(rg_def, &.{ prep_t, tg_t, tu_t, sl2_t, m_t }, &rg_cfg, &rg_out);
            defer core.release(rg_out[0]);
            defer core.release(rg_out[1]);
            const got_g = try readF32(rg_out[0]);
            const got_u = try readF32(rg_out[1]);
            for (rows_pick, 0..) |r, slot| {
                for (0..2) |o| {
                    for (0..16) |c| {
                        const col = o * 16 + c;
                        try testing.expectEqual(@as(f32, @floatCast(@as(f16, @bitCast(want_g0[o][r * 16 + c])))), got_g[(slot * 2 + 0) * 32 + col]);
                        try testing.expectEqual(@as(f32, @floatCast(@as(f16, @bitCast(want_g8[o][r * 16 + c])))), got_g[(slot * 2 + 1) * 32 + col]);
                        try testing.expectEqual(@as(f32, @floatCast(@as(f16, @bitCast(want_u0[o][r * 16 + c])))), got_u[(slot * 2 + 0) * 32 + col]);
                        try testing.expectEqual(@as(f32, @floatCast(@as(f16, @bitCast(want_u8[o][r * 16 + c])))), got_u[(slot * 2 + 1) * 32 + col]);
                    }
                }
            }
        }

        // (g.3) `fz_mid_snax` Part A (`DECODE_MID_SOURCE`, lane-ordered) vs Part B (`DECODE_MID_LINEAR_SOURCE`, linear).
        {
            var ig_v: [128]f32 = undefined;
            var iu_v: [128]f32 = undefined;
            for (&ig_v, &iu_v) |*g, *u| {
                g.* = (rnd.float(f32) - 0.5) * 4.0;
                u.* = (rnd.float(f32) - 0.5) * 4.0;
            }
            const ig_t = try core.fromHost(std.mem.sliceAsBytes(&ig_v), .float32, &.{ 1, 128 });
            defer core.release(ig_t);
            const iu_t = try core.fromHost(std.mem.sliceAsBytes(&iu_v), .float32, &.{ 1, 128 });
            defer core.release(iu_t);
            const sl1_t = try core.fromHost(std.mem.sliceAsBytes(&[_]u32{0}), .uint32, &.{1});
            defer core.release(sl1_t);
            const mid_def = try kernel_mod.Def.create(
                "sushi_exl3_fz_mid_pair1_w16_sspec1_w15_n80_t37f72431_clamp10_snax_tm",
                &.{ "ig", "iu", "svhg", "svhu", "suhd", "slots", "s_ig", "s_iu", "s_svhg", "s_svhu", "s_suhd", "s_slots" },
                &.{ "prepared", "s_prepared" },
                "",
                "",
                true,
                false,
            );
            var mcfg: kernel_mod.Config = .{ .grid = .{ 2 * 32, 1, 1 }, .group = .{ 32, 1, 1 } };
            try mcfg.addOutput(&.{ 1, 128 }, .float16);
            try mcfg.addOutput(&.{ 1, 128 }, .float16);
            try mcfg.addTemplate("IDIM", .{ .int = 128 });
            try mcfg.addTemplate("NSPLIT", .{ .int = 1 });
            try mcfg.addTemplate("S_IDIM", .{ .int = 128 });
            try mcfg.addTemplate("S_NSPLIT", .{ .int = 1 });
            try mcfg.addTemplate("FZ_AX", .{ .int = 1 });
            try mcfg.addTemplate("FZ_AY", .{ .int = 1 });
            try mcfg.addTemplate("FZ_BX", .{ .int = 1 });
            try mcfg.addTemplate("FZ_BY", .{ .int = 1 });
            try mcfg.addTemplate("FZ_NA", .{ .int = 1 });
            try mcfg.addTemplate("FZ_NB", .{ .int = 1 });
            try mcfg.addTemplate("FZ_B_FIRST", .{ .int = 0 });
            var mout: [2]*Tensor = undefined;
            try kernel_mod.launch(mid_def, &.{ ig_t, iu_t, svh128_t, svh128_t, svh128_t, sl1_t, ig_t, iu_t, svh128_t, svh128_t, svh128_t, sl1_t }, &mcfg, &mout);
            defer core.release(mout[0]);
            defer core.release(mout[1]);
            const got_lane = try readU16(mout[0]);
            const got_lin = try readU16(mout[1]);
            var diff_pos: usize = 0;
            for (0..128) |i| {
                try testing.expectEqual(got_lin[i], got_lane[laneSlot(i)]);
                if (got_lin[i] != got_lane[i]) diff_pos += 1;
            }
            try testing.expect(diff_pos > 32);
        }

        // (g.4) `fz_prep_snax` Part A (`PAIR_PREP_SOURCE`, lane-ordered) vs Part B (`SHARED_PREPARE_SOURCE`, linear).
        {
            var xb: [3 * 128]u16 = undefined;
            for (&xb) |*v| v.* = @truncate(@as(u32, @bitCast((rnd.float(f32) - 0.5) * 4.0)) >> 16);
            const xb_t = try core.fromHost(std.mem.sliceAsBytes(&xb), .bfloat16, &.{ 3, 128 });
            defer core.release(xb_t);
            const sl3_t = try core.fromHost(std.mem.sliceAsBytes(&[_]u32{ 0, 0, 0 }), .uint32, &.{3});
            defer core.release(sl3_t);
            const suhg1_t = try core.fromHost(std.mem.sliceAsBytes(suh3[0..128]), .float16, &.{ 1, 128 });
            defer core.release(suhg1_t);
            const suhu1_t = try core.fromHost(std.mem.sliceAsBytes(suh3[128..256]), .float16, &.{ 1, 128 });
            defer core.release(suhu1_t);
            const pdef = try kernel_mod.Def.create(
                "sushi_exl3_fz_prep_snax",
                &.{ "x", "suhg", "suhu", "slots", "s_x", "s_suhg", "s_suhu", "s_slots" },
                &.{ "prep", "s_yg", "s_yu" },
                "",
                "",
                true,
                false,
            );
            var pcfg: kernel_mod.Config = .{ .grid = .{ (6 + 3) * 32, 1, 1 }, .group = .{ 32, 1, 1 } };
            try pcfg.addOutput(&.{ 6, 128 }, .float16);
            try pcfg.addOutput(&.{ 3, 128 }, .float16);
            try pcfg.addOutput(&.{ 3, 128 }, .float16);
            try pcfg.addTemplate("IDIM", .{ .int = 128 });
            try pcfg.addTemplate("TOPK", .{ .int = 1 });
            try pcfg.addTemplate("S_IDIM", .{ .int = 128 });
            try pcfg.addTemplate("S_TOPK", .{ .int = 1 });
            try pcfg.addTemplate("FZ_AX", .{ .int = 1 });
            try pcfg.addTemplate("FZ_AY", .{ .int = 3 });
            try pcfg.addTemplate("FZ_BX", .{ .int = 1 });
            try pcfg.addTemplate("FZ_BY", .{ .int = 3 });
            try pcfg.addTemplate("FZ_NA", .{ .int = 6 });
            try pcfg.addTemplate("FZ_NB", .{ .int = 3 });
            try pcfg.addTemplate("FZ_B_FIRST", .{ .int = 1 });
            var pout: [3]*Tensor = undefined;
            try kernel_mod.launch(pdef, &.{ xb_t, suhg1_t, suhu1_t, sl3_t, xb_t, suhg1_t, suhu1_t, sl3_t }, &pcfg, &pout);
            defer for (pout) |o| core.release(o);
            const got_a = try readU16(pout[0]);
            const got_bg = try readU16(pout[1]);
            const got_bu = try readU16(pout[2]);
            for (0..3) |r| {
                for (0..128) |i| {
                    try testing.expectEqual(got_bg[r * 128 + i], got_a[(r * 2 + 0) * 128 + laneSlot(i)]);
                    try testing.expectEqual(got_bu[r * 128 + i], got_a[(r * 2 + 1) * 128 + laneSlot(i)]);
                }
            }
        }
    }
}
