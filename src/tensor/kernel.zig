//! Custom kernels on the engine: the kernel a caller made with `mlx_fast_metal_kernel_new`, launched on tensors of
//! ours. One instantiation (template values, each input's dtype and how it is passed, output dtypes) is written and
//! compiled once and its pipeline kept.
//!
//! The signature is written as MLX's `metal_kernel` writes it (mlx/backend/common/metal_kernel.cpp, MIT), so a
//! source compiles here to the code it compiles to under MLX.

const std = @import("std");
const metal = @import("../metal.zig");
const core = @import("core.zig");
const ops = @import("ops.zig");

const Tensor = core.Tensor;
const Dtype = core.Dtype;
const gpa = std.heap.c_allocator;


/// What a kernel source may name that MLX's preamble declared for it.
const preamble =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\typedef bfloat bfloat16_t;
    \\typedef half float16_t;
    \\
;

pub fn typeName(d: Dtype) core.Error![]const u8 {
    return switch (d) {
        .bool_ => "bool",
        .uint8 => "uint8_t",
        .uint16 => "uint16_t",
        .uint32 => "uint32_t",
        .uint64 => "uint64_t",
        .int8 => "int8_t",
        .int16 => "int16_t",
        .int32 => "int32_t",
        .int64 => "int64_t",
        .float16 => "float16_t",
        .float32 => "float",
        .bfloat16 => "bfloat16_t",
        .float64, .complex64 => error.TensorDtype,
    };
}

pub const max_outputs = 8;
pub const max_template_args = 32;

pub const TemplateValue = union(enum) { int: c_int, boolean: bool, dtype: Dtype };

pub const TemplateArg = struct {
    name: [48]u8 = undefined,
    name_len: u8 = 0,
    value: TemplateValue,
};

/// One launch's outputs, grid, threadgroup and template values.
pub const Config = struct {
    out_shapes: [max_outputs][core.max_dims]c_int = undefined,
    out_ndims: [max_outputs]u8 = undefined,
    out_dtypes: [max_outputs]Dtype = undefined,
    n_out: usize = 0,
    grid: [3]c_int = .{ 1, 1, 1 },
    group: [3]c_int = .{ 1, 1, 1 },
    targs: [max_template_args]TemplateArg = undefined,
    n_targs: usize = 0,
    /// A mirror that could not hold what its MLX config was given: a launch through it is refused.
    partial: bool = false,

    pub fn addOutput(c: *Config, shape: []const c_int, dtype: Dtype) core.Error!void {
        if (c.n_out == max_outputs or shape.len > core.max_dims) return error.TensorShape;
        @memcpy(c.out_shapes[c.n_out][0..shape.len], shape);
        c.out_ndims[c.n_out] = @intCast(shape.len);
        c.out_dtypes[c.n_out] = dtype;
        c.n_out += 1;
    }

    pub fn addTemplate(c: *Config, name: []const u8, value: TemplateValue) core.Error!void {
        if (c.n_targs == max_template_args or name.len > 48) return error.TensorShape;
        var arg: TemplateArg = .{ .value = value, .name_len = @intCast(name.len) };
        @memcpy(arg.name[0..name.len], name);
        c.targs[c.n_targs] = arg;
        c.n_targs += 1;
    }
};

/// A kernel as its maker gave it.
pub const Def = struct {
    name: []u8,
    inputs: [][]u8,
    outputs: [][]u8,
    source: []u8,
    header: []u8,
    ensure_row_contiguous: bool,
    atomic_outputs: bool,
    /// Per input: whether the source reads its `_shape`, `_strides`, `_ndim`, each one more buffer.
    shape_infos: [][3]bool,
    /// The `[[attribute]]` parameters the source names, as signature lines.
    attributes: [][]const u8,
    instances: std.StringHashMapUnmanaged(metal.Pipeline) = .empty,

    pub fn create(name: []const u8, inputs: []const []const u8, outputs: []const []const u8, source: []const u8, header: []const u8, ensure_row_contiguous: bool, atomic_outputs: bool) !*Def {
        const d = try gpa.create(Def);
        d.* = .{
            .name = try gpa.dupe(u8, name),
            .inputs = try gpa.alloc([]u8, inputs.len),
            .outputs = try gpa.alloc([]u8, outputs.len),
            .source = try gpa.dupe(u8, source),
            .header = try gpa.dupe(u8, header),
            .ensure_row_contiguous = ensure_row_contiguous,
            .atomic_outputs = atomic_outputs,
            .shape_infos = try gpa.alloc([3]bool, inputs.len),
            .attributes = &.{},
        };
        for (inputs, 0..) |n, i| {
            d.inputs[i] = try gpa.dupe(u8, n);
            inline for (.{ "_shape", "_strides", "_ndim" }, 0..) |suffix, k| {
                const needle = try std.mem.concat(gpa, u8, &.{ n, suffix });
                defer gpa.free(needle);
                d.shape_infos[i][k] = std.mem.indexOf(u8, source, needle) != null;
            }
        }
        for (outputs, 0..) |n, i| d.outputs[i] = try gpa.dupe(u8, n);
        var attrs: std.ArrayList([]const u8) = .empty;
        for (metal_attributes) |a| {
            if (std.mem.indexOf(u8, source, a[0]) == null) continue;
            try attrs.append(gpa, try std.fmt.allocPrint(gpa, "  {s} {s} [[{s}]]", .{ a[1], a[0], a[0] }));
        }
        d.attributes = try attrs.toOwnedSlice(gpa);
        return d;
    }
};

const metal_attributes = [_][2][]const u8{
    .{ "dispatch_quadgroups_per_threadgroup", "uint" },
    .{ "dispatch_simdgroups_per_threadgroup", "uint" },
    .{ "dispatch_threads_per_threadgroup", "uint3" },
    .{ "grid_origin", "uint3" },
    .{ "grid_size", "uint3" },
    .{ "quadgroup_index_in_threadgroup", "uint" },
    .{ "quadgroups_per_threadgroup", "uint" },
    .{ "simdgroup_index_in_threadgroup", "uint" },
    .{ "simdgroups_per_threadgroup", "uint" },
    .{ "thread_execution_width", "uint" },
    .{ "thread_index_in_quadgroup", "uint" },
    .{ "thread_index_in_simdgroup", "uint" },
    .{ "thread_index_in_threadgroup", "uint" },
    .{ "thread_position_in_grid", "uint3" },
    .{ "thread_position_in_threadgroup", "uint3" },
    .{ "threadgroup_position_in_grid", "uint3" },
    .{ "threadgroups_per_grid", "uint3" },
    .{ "threads_per_grid", "uint3" },
    .{ "threads_per_simdgroup", "uint" },
    .{ "threads_per_threadgroup", "uint3" },
};

/// How an input reaches the kernel: a 0-d one by reference, any other as a device pointer. MLX also puts an array
/// under 8 elements in the constant address space, a second pipeline for the same source that a request would
/// compile the first time an input came short; the engine has one.
fn passing(t: *const Tensor) u8 {
    return if (t.ndim == 0) 's' else 'd';
}

/// The kernel function's name for one instantiation: MLX's for the same template values and dtypes.
fn kernelName(out: *std.ArrayList(u8), def: *const Def, inputs: []const *Tensor, cfg: *const Config) !void {
    try out.appendSlice(gpa, "custom_kernel_");
    try out.appendSlice(gpa, def.name);
    if (cfg.n_targs > 0) {
        // MLX joins the name and the values with two underscores.
        try out.appendSlice(gpa, "__");
        for (cfg.targs[0..cfg.n_targs], 0..) |arg, i| {
            if (i > 0) try out.append(gpa, '_');
            try templateValue(out, arg);
        }
    }
    for (inputs) |in| {
        try out.append(gpa, '_');
        try out.appendSlice(gpa, try typeName(in.dtype));
        if (passing(in) == 's') try out.append(gpa, 's');
    }
    for (cfg.out_dtypes[0..cfg.n_out]) |d| {
        try out.append(gpa, '_');
        try out.appendSlice(gpa, try typeName(d));
    }
}

fn templateValue(out: *std.ArrayList(u8), arg: TemplateArg) !void {
    switch (arg.value) {
        .int => |v| try out.print(gpa, "{d}", .{v}),
        .boolean => |v| try out.append(gpa, if (v) '1' else '0'),
        .dtype => |d| try out.appendSlice(gpa, try typeName(d)),
    }
}

/// The whole Metal source of one instantiation.
fn render(def: *const Def, name: []const u8, inputs: []const *Tensor, cfg: *const Config) ![:0]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, preamble);
    try out.appendSlice(gpa, def.header);
    const targs = cfg.targs[0..cfg.n_targs];
    if (targs.len > 0) {
        try out.appendSlice(gpa, "template <");
        for (targs, 0..) |arg, i| {
            if (i > 0) try out.appendSlice(gpa, ", ");
            try out.appendSlice(gpa, switch (arg.value) {
                .int => "int",
                .boolean => "bool",
                .dtype => "typename",
            });
            try out.append(gpa, ' ');
            try out.appendSlice(gpa, arg.name[0..arg.name_len]);
        }
        try out.appendSlice(gpa, ">\n");
    }
    try out.print(gpa, "[[kernel]] void {s}(\n", .{name});

    var index: usize = 0;
    for (inputs, 0..) |in, i| {
        try out.print(gpa, "  const {s} {s}{s} {s} [[buffer({d})]],\n", .{
            if (in.ndim == 0) "constant" else "device",
            try typeName(in.dtype),
            if (in.ndim == 0) "&" else "*",
            def.inputs[i],
            index,
        });
        index += 1;
        if (in.ndim == 0) continue;
        if (def.shape_infos[i][0]) {
            try out.print(gpa, "  const constant int* {s}_shape [[buffer({d})]],\n", .{ def.inputs[i], index });
            index += 1;
        }
        if (def.shape_infos[i][1]) {
            try out.print(gpa, "  const constant int64_t* {s}_strides [[buffer({d})]],\n", .{ def.inputs[i], index });
            index += 1;
        }
        if (def.shape_infos[i][2]) {
            try out.print(gpa, "  const constant int& {s}_ndim [[buffer({d})]],\n", .{ def.inputs[i], index });
            index += 1;
        }
    }
    for (def.outputs, 0..) |out_name, i| {
        const t = try typeName(cfg.out_dtypes[i]);
        if (def.atomic_outputs) {
            try out.print(gpa, "  device atomic<{s}>* {s} [[buffer({d})]]", .{ t, out_name, index });
        } else {
            try out.print(gpa, "  device {s}* {s} [[buffer({d})]]", .{ t, out_name, index });
        }
        const last = i + 1 == def.outputs.len and def.attributes.len == 0;
        try out.appendSlice(gpa, if (last) ") {\n" else ",\n");
        index += 1;
    }
    for (def.attributes, 0..) |attr, i| {
        try out.appendSlice(gpa, attr);
        try out.appendSlice(gpa, if (i + 1 == def.attributes.len) ") {\n" else ",\n");
    }
    try out.appendSlice(gpa, def.source);
    try out.appendSlice(gpa, "\n}\n");
    if (targs.len > 0) {
        try out.print(gpa, "\ntemplate [[host_name(\"{s}\")]] [[kernel]] decltype({s}<", .{ name, name });
        for (targs, 0..) |arg, i| {
            if (i > 0) try out.appendSlice(gpa, ", ");
            try templateValue(&out, arg);
        }
        try out.print(gpa, ">) {s}<", .{name});
        for (targs, 0..) |arg, i| {
            if (i > 0) try out.appendSlice(gpa, ", ");
            try templateValue(&out, arg);
        }
        try out.appendSlice(gpa, ">;\n");
    }
    return out.toOwnedSliceSentinel(gpa, 0);
}

threadlocal var key_buf: std.ArrayList(u8) = .empty;

/// The pipeline of `def` at these inputs and this config, compiled at its first launch.
fn pipelineOf(def: *Def, inputs: []const *Tensor, cfg: *const Config) !metal.Pipeline {
    key_buf.clearRetainingCapacity();
    for (cfg.targs[0..cfg.n_targs]) |arg| {
        try key_buf.appendSlice(gpa, arg.name[0..arg.name_len]);
        try key_buf.append(gpa, 0);
        switch (arg.value) {
            .int => |v| try key_buf.print(gpa, "i{d}", .{v}),
            .boolean => |v| try key_buf.print(gpa, "b{d}", .{@intFromBool(v)}),
            .dtype => |d| try key_buf.print(gpa, "t{d}", .{@intFromEnum(d)}),
        }
        try key_buf.append(gpa, 0);
    }
    try key_buf.append(gpa, 1);
    for (inputs) |in| {
        try key_buf.append(gpa, @intCast(@intFromEnum(in.dtype)));
        try key_buf.append(gpa, passing(in));
    }
    for (cfg.out_dtypes[0..cfg.n_out]) |d| try key_buf.append(gpa, @intCast(@intFromEnum(d)));
    if (def.instances.get(key_buf.items)) |p| return p;

    var name: std.ArrayList(u8) = .empty;
    defer name.deinit(gpa);
    try kernelName(&name, def, inputs, cfg);
    try name.append(gpa, 0);
    const name_z: [:0]const u8 = name.items[0 .. name.items.len - 1 :0];
    const text = try render(def, name_z, inputs, cfg);
    defer gpa.free(text);
    const pipeline = try core.pipelineFromSource(name_z, text);
    try def.instances.put(gpa, try gpa.dupe(u8, key_buf.items), pipeline);
    return pipeline;
}

pub const LaunchError = core.Error || error{ThreadGroupTooLarge};

/// The threadgroup limit of the pipeline a launch refused for.
pub var refused_limit: usize = 0;
pub var refused_threads: usize = 0;

/// Runs `def` on `inputs` into fresh tensors written to `outputs` (one per output of `cfg`), which the caller owns.
pub fn launch(def: *Def, inputs: []const *Tensor, cfg: *const Config, outputs: []*Tensor) LaunchError!void {
    if (inputs.len != def.inputs.len or cfg.n_out != def.outputs.len or outputs.len != cfg.n_out) return error.TensorShape;

    // A kernel that asks for row-contiguous inputs gets a copy of each input that is not.
    var bound: [32]*Tensor = undefined;
    var copied: [32]bool = @splat(false);
    if (inputs.len > bound.len) return error.TensorShape;
    var n_bound: usize = 0;
    defer for (bound[0..n_bound], copied[0..n_bound]) |t, c| {
        if (c) core.release(t);
    };
    for (inputs) |in| {
        if (def.ensure_row_contiguous and !in.rowContiguous()) {
            bound[n_bound] = try ops.contiguousCopy(in);
            copied[n_bound] = true;
        } else bound[n_bound] = in;
        n_bound += 1;
    }

    const pipeline = pipelineOf(def, bound[0..n_bound], cfg) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => |other| @as(LaunchError, @errorCast(other)),
    };
    const threads = @as(usize, @intCast(cfg.group[0])) * @as(usize, @intCast(cfg.group[1])) * @as(usize, @intCast(cfg.group[2]));
    if (threads > pipeline.max_threads) {
        refused_limit = pipeline.max_threads;
        refused_threads = threads;
        return error.ThreadGroupTooLarge;
    }

    var made: usize = 0;
    errdefer for (outputs[0..made]) |t| core.release(t);
    for (0..cfg.n_out) |i| {
        outputs[i] = try core.fresh(cfg.out_dtypes[i], cfg.out_shapes[i][0..cfg.out_ndims[i]]);
        made += 1;
    }

    const p = try core.pass();
    p.pipeline(pipeline);
    var index: usize = 0;
    for (bound[0..n_bound], 0..) |in, i| {
        core.bind(p, in, index);
        index += 1;
        if (in.ndim == 0) continue;
        if (def.shape_infos[i][0]) {
            p.bytes(std.mem.sliceAsBytes(in.shape[0..in.ndim]), index);
            index += 1;
        }
        if (def.shape_infos[i][1]) {
            var strides: [core.max_dims]i64 = undefined;
            for (in.strides[0..in.ndim], 0..) |s, k| strides[k] = @intCast(s);
            p.bytes(std.mem.sliceAsBytes(strides[0..in.ndim]), index);
            index += 1;
        }
        if (def.shape_infos[i][2]) {
            p.constant(@as(c_int, in.ndim), index);
            index += 1;
        }
    }
    for (outputs) |out| {
        core.bindOut(p, out, index);
        index += 1;
    }
    const grid: metal.Size = .{ .x = @intCast(cfg.grid[0]), .y = @intCast(cfg.grid[1]), .z = @intCast(cfg.grid[2]) };
    core.dispatch(p, grid, .{
        .x = @min(@as(usize, @intCast(cfg.group[0])), grid.x),
        .y = @min(@as(usize, @intCast(cfg.group[1])), grid.y),
        .z = @min(@as(usize, @intCast(cfg.group[2])), grid.z),
    });
}
