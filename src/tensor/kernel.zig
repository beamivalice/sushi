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
pub const max_inputs = 32;
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
    /// The pipeline of the last launch, for a config that outlives its launch. The outputs and template values only
    /// change through `addOutput` and `addTemplate`, which clear it; a launch checks the kernel and the inputs.
    last: ?Last = null,
    /// Per `addOutput` and `addTemplate` call, in the order made: whether it was an output. MLX's config is filled
    /// in this order.
    order: [max_outputs + max_template_args]bool = undefined,
    /// MLX's config of the same contents, made when a config of the engine's own first launches on MLX's side.
    twin: ?*anyopaque = null,

    const Last = struct {
        def: u64,
        n_in: u8,
        sig: [2 * max_inputs]u8,
        pipeline: metal.Pipeline,
    };

    pub fn addOutput(c: *Config, shape: []const c_int, dtype: Dtype) core.Error!void {
        c.last = null;
        if (c.n_out == max_outputs or shape.len > core.max_dims) return error.TensorShape;
        c.order[c.n_out + c.n_targs] = true;
        @memcpy(c.out_shapes[c.n_out][0..shape.len], shape);
        c.out_ndims[c.n_out] = @intCast(shape.len);
        c.out_dtypes[c.n_out] = dtype;
        c.n_out += 1;
    }

    pub fn addTemplate(c: *Config, name: []const u8, value: TemplateValue) core.Error!void {
        c.last = null;
        if (c.n_targs == max_template_args or name.len > 48) return error.TensorShape;
        c.order[c.n_out + c.n_targs] = false;
        var arg: TemplateArg = .{ .value = value, .name_len = @intCast(name.len) };
        @memcpy(arg.name[0..name.len], name);
        c.targs[c.n_targs] = arg;
        c.n_targs += 1;
    }
};

const recent_slots = 16;
/// A launch with more template arguments than a `Recent` holds takes the key every time.
const recent_targs = 16;

/// One instantiation as a launch states it, and its pipeline.
const Recent = struct {
    n_targs: u8,
    targs: [recent_targs]TemplateArg,
    n_out: u8,
    out_dtypes: [max_outputs]Dtype,
    n_in: u8,
    /// Per input: its dtype and how it is passed.
    sig: [2 * max_inputs]u8,
    pipeline: metal.Pipeline,

    fn answers(r: *const Recent, inputs: []const *Tensor, cfg: *const Config) bool {
        if (r.n_targs != cfg.n_targs or r.n_out != cfg.n_out or r.n_in != inputs.len) return false;
        for (r.targs[0..r.n_targs], cfg.targs[0..cfg.n_targs]) |a, b| {
            if (a.name_len != b.name_len or !std.mem.eql(u8, a.name[0..a.name_len], b.name[0..b.name_len]) or !std.meta.eql(a.value, b.value)) return false;
        }
        if (!std.mem.eql(Dtype, r.out_dtypes[0..r.n_out], cfg.out_dtypes[0..cfg.n_out])) return false;
        return sigMatches(&r.sig, inputs);
    }

    fn of(inputs: []const *Tensor, cfg: *const Config, pipeline: metal.Pipeline) Recent {
        var r: Recent = .{ .n_targs = @intCast(cfg.n_targs), .targs = undefined, .n_out = @intCast(cfg.n_out), .out_dtypes = undefined, .n_in = @intCast(inputs.len), .sig = undefined, .pipeline = pipeline };
        @memcpy(r.targs[0..cfg.n_targs], cfg.targs[0..cfg.n_targs]);
        @memcpy(r.out_dtypes[0..cfg.n_out], cfg.out_dtypes[0..cfg.n_out]);
        fillSig(&r.sig, inputs);
        return r;
    }
};


var next_def: u64 = 1;

/// A kernel as its maker gave it.
pub const Def = struct {
    /// Never reused, so a config's `last` cannot answer for a kernel made at a freed one's address.
    id: u64,
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
    /// The instantiations launched last, compared field by field: a launch that finds its own skips the key.
    recent: [recent_slots]Recent = undefined,
    recent_count: u8 = 0,
    recent_next: u8 = 0,
    /// MLX's kernel of the same source, made when a kernel of the engine's own first launches on MLX's side.
    twin: ?*anyopaque = null,

    pub fn create(name: []const u8, inputs: []const []const u8, outputs: []const []const u8, source: []const u8, header: []const u8, ensure_row_contiguous: bool, atomic_outputs: bool) !*Def {
        const d = try gpa.create(Def);
        d.* = .{
            .id = next_def,
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
        next_def += 1;
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

/// Instance keys built since the process began.
pub var key_builds: u64 = 0;

/// The pipeline of `def` at these inputs and this config, compiled at its first launch.
fn pipelineOf(def: *Def, inputs: []const *Tensor, cfg: *Config) !metal.Pipeline {
    if (cfg.last) |*l| if (l.def == def.id and l.n_in == inputs.len and sigMatches(&l.sig, inputs)) return l.pipeline;
    const pipeline = try pipelineByValue(def, inputs, cfg);
    var last: Config.Last = .{ .def = def.id, .n_in = @intCast(inputs.len), .sig = undefined, .pipeline = pipeline };
    fillSig(&last.sig, inputs);
    cfg.last = last;
    return pipeline;
}

fn sigMatches(sig: *const [2 * max_inputs]u8, inputs: []const *Tensor) bool {
    for (inputs, 0..) |in, i| {
        if (sig[2 * i] != @as(u8, @intCast(@intFromEnum(in.dtype))) or sig[2 * i + 1] != passing(in)) return false;
    }
    return true;
}

fn fillSig(sig: *[2 * max_inputs]u8, inputs: []const *Tensor) void {
    for (inputs, 0..) |in, i| {
        sig[2 * i] = @intCast(@intFromEnum(in.dtype));
        sig[2 * i + 1] = passing(in);
    }
}

/// `def`'s pipeline for a launch that no config remembers: one of the instantiations it launched last, else by key.
fn pipelineByValue(def: *Def, inputs: []const *Tensor, cfg: *const Config) !metal.Pipeline {
    for (def.recent[0..def.recent_count]) |*r| if (r.answers(inputs, cfg)) return r.pipeline;
    const pipeline = try instanceOf(def, inputs, cfg);
    if (cfg.n_targs <= recent_targs) {
        const slot = def.recent_next;
        def.recent[slot] = Recent.of(inputs, cfg, pipeline);
        def.recent_count = @max(def.recent_count, slot + 1);
        def.recent_next = (slot + 1) % recent_slots;
    }
    return pipeline;
}

fn instanceOf(def: *Def, inputs: []const *Tensor, cfg: *const Config) !metal.Pipeline {
    key_builds += 1;
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
pub fn launch(def: *Def, inputs: []const *Tensor, cfg: *Config, outputs: []*Tensor) LaunchError!void {
    if (inputs.len != def.inputs.len or cfg.n_out != def.outputs.len or outputs.len != cfg.n_out) return error.TensorShape;

    // A kernel that asks for row-contiguous inputs gets a copy of each input that is not.
    var bound: [max_inputs]*Tensor = undefined;
    var copied: [max_inputs]bool = @splat(false);
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

// ── Tests ──

const testing = std.testing;
const mlx = @import("../mlx.zig");

fn floats(vals: []const f32) !*Tensor {
    return core.fromHost(std.mem.sliceAsBytes(vals), .float32, &.{@intCast(vals.len)});
}

fn ints(vals: []const i32) !*Tensor {
    return core.fromHost(std.mem.sliceAsBytes(vals), .int32, &.{@intCast(vals.len)});
}

fn oneOutput(cfg: *Config, n: c_int) !void {
    try cfg.addOutput(&.{n}, .float32);
    cfg.grid = .{ n, 1, 1 };
    cfg.group = .{ n, 1, 1 };
}

/// Launches `def` on `x` and returns the output's four floats.
fn run(def: *Def, x: *Tensor, cfg: *Config) ![4]f32 {
    var out: [1]*Tensor = undefined;
    try launch(def, &.{x}, cfg, &out);
    defer core.release(out[0]);
    return std.mem.bytesAsSlice(f32, try core.hostBytes(out[0]))[0..4].*;
}

test "kernel: a repeated launch builds no instance key, whether its config is kept or made anew" {
    try mlx.requireGpu();
    const def = try Def.create("kn_memo_same", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] * 2.0f;", "", true, false);
    const x = try floats(&.{ 1, 2, 3, 4 });
    defer core.release(x);
    var cfg: Config = .{};
    try oneOutput(&cfg, 4);
    const before = key_builds;
    const made = core.pipelines;
    try testing.expectEqual([4]f32{ 2, 4, 6, 8 }, try run(def, x, &cfg));
    try testing.expectEqual([4]f32{ 2, 4, 6, 8 }, try run(def, x, &cfg));
    try testing.expectEqual([4]f32{ 2, 4, 6, 8 }, try run(def, x, &cfg));
    try testing.expectEqual(before + 1, key_builds);
    try testing.expectEqual(made + 1, core.pipelines);

    // A config made anew for each launch, as the DeepSeek kernels do, answers the same way.
    const fresh_before = key_builds;
    for (0..3) |_| {
        var again: Config = .{};
        try oneOutput(&again, 4);
        try testing.expectEqual([4]f32{ 2, 4, 6, 8 }, try run(def, x, &again));
    }
    try testing.expectEqual(fresh_before, key_builds);
    try testing.expectEqual(made + 1, core.pipelines);
}

test "kernel: a changed input dtype launches the pipeline of that dtype" {
    try mlx.requireGpu();
    const def = try Def.create("kn_memo_dtype", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] * 3;", "", true, false);
    const xf = try floats(&.{ 1, 2, 3, 4 });
    defer core.release(xf);
    const xi = try ints(&.{ 10, 20, 30, 40 });
    defer core.release(xi);
    var cfg: Config = .{};
    try oneOutput(&cfg, 4);
    try testing.expectEqual([4]f32{ 3, 6, 9, 12 }, try run(def, xf, &cfg));
    const made = core.pipelines;
    const keys = key_builds;
    try testing.expectEqual([4]f32{ 30, 60, 90, 120 }, try run(def, xi, &cfg));
    try testing.expectEqual(made + 1, core.pipelines);
    try testing.expectEqual(keys + 1, key_builds);
    try testing.expectEqual([4]f32{ 3, 6, 9, 12 }, try run(def, xf, &cfg));
    try testing.expectEqual(made + 1, core.pipelines);
    try testing.expectEqual(keys + 1, key_builds);
}

test "kernel: two kernels launched in turn through one config each take their own pipeline" {
    try mlx.requireGpu();
    const twice = try Def.create("kn_memo_twice", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] * 2.0f;", "", true, false);
    const thrice = try Def.create("kn_memo_thrice", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] * 3.0f;", "", true, false);
    const x = try floats(&.{ 1, 2, 3, 4 });
    defer core.release(x);
    var cfg: Config = .{};
    try oneOutput(&cfg, 4);
    try testing.expectEqual([4]f32{ 2, 4, 6, 8 }, try run(twice, x, &cfg));
    try testing.expectEqual([4]f32{ 3, 6, 9, 12 }, try run(thrice, x, &cfg));
    try testing.expectEqual([4]f32{ 2, 4, 6, 8 }, try run(twice, x, &cfg));
}

test "kernel: a template value added to a launched config changes the pipeline the next launch takes" {
    try mlx.requireGpu();
    const source = "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] + float(K);";
    const def = try Def.create("kn_memo_template", &.{"x"}, &.{"out"}, source, "", true, false);
    const x = try floats(&.{ 1, 2, 3, 4 });
    defer core.release(x);
    var cfg: Config = .{};
    try oneOutput(&cfg, 4);
    try cfg.addTemplate("K", .{ .int = 5 });
    try testing.expectEqual([4]f32{ 6, 7, 8, 9 }, try run(def, x, &cfg));
    const made = core.pipelines;
    try cfg.addTemplate("L", .{ .int = 1 });
    try testing.expectEqual([4]f32{ 6, 7, 8, 9 }, try run(def, x, &cfg));
    try testing.expectEqual(made + 1, core.pipelines);

    // Two configs of one kernel at different values keep their own pipelines.
    var other: Config = .{};
    try oneOutput(&other, 4);
    try other.addTemplate("K", .{ .int = 9 });
    try testing.expectEqual([4]f32{ 10, 11, 12, 13 }, try run(def, x, &other));
    try testing.expectEqual([4]f32{ 6, 7, 8, 9 }, try run(def, x, &cfg));
    try testing.expectEqual([4]f32{ 10, 11, 12, 13 }, try run(def, x, &other));
}

test "kernel: more instantiations than a kernel remembers each launch their own pipeline" {
    try mlx.requireGpu();
    const def = try Def.create("kn_recent_many", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] + float(K);", "", true, false);
    const x = try floats(&.{ 1, 2, 3, 4 });
    defer core.release(x);
    for (0..2) |_| for (0..recent_slots + 4) |k| {
        var cfg: Config = .{};
        try oneOutput(&cfg, 4);
        try cfg.addTemplate("K", .{ .int = @intCast(k) });
        const add: f32 = @floatFromInt(k);
        try testing.expectEqual([4]f32{ 1 + add, 2 + add, 3 + add, 4 + add }, try run(def, x, &cfg));
    };
}

test "kernel: a launch with more template arguments than a kernel remembers builds its key each time and launches right" {
    try mlx.requireGpu();
    const wide = recent_targs + 1;
    const sum = comptime blk: {
        var text: []const u8 = "float(0";
        for (0..wide) |i| text = text ++ std.fmt.comptimePrint(" + A{d}", .{i});
        break :blk text ++ ")";
    };
    const def = try Def.create("kn_recent_wide", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] + " ++ sum ++ ";", "", true, false);
    const x = try floats(&.{ 1, 2, 3, 4 });
    defer core.release(x);
    const want: f32 = wide;
    const before = key_builds;
    for (0..2) |_| {
        var cfg: Config = .{};
        try oneOutput(&cfg, 4);
        inline for (0..wide) |i| try cfg.addTemplate(std.fmt.comptimePrint("A{d}", .{i}), .{ .int = 1 });
        try testing.expectEqual([4]f32{ 1 + want, 2 + want, 3 + want, 4 + want }, try run(def, x, &cfg));
    }
    try testing.expectEqual(before + 2, key_builds);
}
