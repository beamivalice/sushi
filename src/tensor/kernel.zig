//! Custom kernels on the engine: the kernel a caller made with `mlx_fast_metal_kernel_new`, launched on tensors of
//! ours. Every instantiation resolves exclusively to a precompiled function in `sushi.metallib` via `metallib_catalog`,
//! with zero runtime Metal source compilation; an unregistered kernel is refused by name.

const std = @import("std");
const metal = @import("../metal.zig");
const log = @import("../log.zig");
const catalog = @import("../metallib_catalog.zig");
const core = @import("core.zig");
const ops = @import("ops.zig");
const stock = @import("stock.zig");

const Tensor = core.Tensor;
const Dtype = core.Dtype;
const gpa = std.heap.c_allocator;

pub const max_outputs = 8;
pub const max_inputs = 32;
pub const max_template_args = 48;

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
    /// A mirror of MLX's config whose grid or thread group was set inside the engine's scope: MLX's own config is
    /// given them when it next launches on MLX's side, not before.
    behind: bool = false,
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
        next_def += 1;
        return d;
    }
};

/// How an input reaches the kernel: a 0-d one by reference, any other as a device pointer. MLX also puts an array
/// under 8 elements in the constant address space, a second pipeline for the same source that a request would
/// compile the first time an input came short; the engine has one.
fn passing(t: *const Tensor) u8 {
    return if (t.ndim == 0) 's' else 'd';
}

threadlocal var key_buf: std.ArrayList(u8) = .empty;

/// Instance keys built since the process began.
pub var key_builds: u64 = 0;

/// The pipeline of `def` at these inputs and this config, fetched once from `sushi.metallib`.
fn pipelineOf(def: *Def, inputs: []const *Tensor, cfg: *Config) !metal.Pipeline {
    if (stock.noting != null) return instanceOf(def, inputs, cfg);
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
    if (stock.noting == null) {
        if (def.instances.get(key_buf.items)) |p| return p;
    }

    if (def.atomic_outputs) {
        var msg: [256]u8 = undefined;
        metal.setError(std.fmt.bufPrint(&msg, "atomic custom kernel not in sushi.metallib: {s}", .{def.name}) catch def.name);
        log.err("[engine] atomic custom kernel not supported in sushi.metallib: {s}\n", .{def.name});
        return error.MetalCompile;
    }

    var rt_in: [max_inputs]catalog.RuntimeInput = undefined;
    for (inputs, 0..) |in, i| {
        rt_in[i] = .{
            .name = def.inputs[i],
            .dtype = catalog.Dtype.fromInt(@intFromEnum(in.dtype)) orelse return core.refusing("a custom kernel input dtype not supported on Cooler", error.TensorDtype),
            .scalar = in.ndim == 0,
        };
    }
    var rt_out: [max_outputs]catalog.Dtype = undefined;
    for (cfg.out_dtypes[0..cfg.n_out], 0..) |d, i| {
        rt_out[i] = catalog.Dtype.fromInt(@intFromEnum(d)) orelse return core.refusing("a custom kernel output dtype not supported on Cooler", error.TensorDtype);
    }
    var rt_targs: [max_template_args]catalog.RuntimeTarg = undefined;
    for (cfg.targs[0..cfg.n_targs], 0..) |*arg, i| {
        rt_targs[i] = .{
            .name = arg.name[0..arg.name_len],
            .value = switch (arg.value) {
                .int => |v| .{ .int = v },
                .boolean => |v| .{ .boolean = v },
                .dtype => |d| .{ .dtype = catalog.Dtype.fromInt(@intFromEnum(d)) orelse return core.refusing("a custom kernel template dtype not supported on Cooler", error.TensorDtype) },
            },
        };
    }

    var resolved: catalog.Resolved = .{};
    if (!catalog.resolve(def.name, rt_in[0..inputs.len], rt_out[0..cfg.n_out], rt_targs[0..cfg.n_targs], &resolved)) {
        var msg: [512]u8 = undefined;
        var w: std.ArrayList(u8) = .initBuffer(&msg);
        w.appendSliceBounded(def.name) catch {};
        w.appendSliceBounded(" not in sushi.metallib (in:") catch {};
        for (rt_in[0..inputs.len], 0..) |ri, idx| {
            if (idx > 0) w.appendBounded(',') catch {};
            w.appendSliceBounded(ri.dtype.shortName()) catch {};
        }
        w.appendSliceBounded(" out:") catch {};
        for (rt_out[0..cfg.n_out], 0..) |od, idx| {
            if (idx > 0) w.appendBounded(',') catch {};
            w.appendSliceBounded(od.shortName()) catch {};
        }
        w.appendBounded(')') catch {};
        metal.setError(w.items);
        log.err("[engine] kernel not in sushi.metallib: {s}\n", .{w.items});
        return error.MetalCompile;
    }

    const pipeline = if (resolved.n_flags == 0 and resolved.n_ints == 0)
        try stock.kernel(resolved.fnName())
    else blk: {
        var flags: [catalog.max_fc_bools]metal.Flag = undefined;
        for (resolved.flagSlice(), 0..) |f, i| flags[i] = .{ .index = f.index, .value = f.value };
        var ints_fc: [catalog.max_fc_ints]metal.IntConstant = undefined;
        for (resolved.intSlice(), 0..) |c, i| ints_fc[i] = .{ .index = c.index, .value = c.value };
        break :blk try stock.kernelConstants(resolved.fnName(), resolved.hashKey(), flags[0..resolved.n_flags], ints_fc[0..resolved.n_ints]);
    };

    if (stock.noting == null) {
        try def.instances.ensureUnusedCapacity(gpa, 1);
        def.instances.putAssumeCapacity(try gpa.dupe(u8, key_buf.items), pipeline);
    }
    return pipeline;
}

pub const LaunchError = core.Error || error{ThreadGroupTooLarge};

/// A test's grant: a kernel whose name starts with `grant_cap_name` (any, when empty) is given at most `grant_cap`
/// threads a group, as a GPU that grants it less than it asks would.
pub var grant_cap: usize = std.math.maxInt(usize);
pub var grant_cap_name: []const u8 = "";

/// The threadgroup limit of the pipeline a launch refused for.
pub var refused_limit: usize = 0;
pub var refused_threads: usize = 0;

/// Runs `def` on `inputs` into fresh tensors written to `outputs` (one per output of `cfg`), which the caller owns.
pub fn launch(def: *Def, inputs: []const *Tensor, cfg: *Config, outputs: []*Tensor) LaunchError!void {
    return launchWith(def, inputs, cfg, outputs, .fresh);
}

/// `launch`, its one output written over `into` (row-contiguous, of the output's shape and dtype): the elements the
/// kernel does not write keep what `into` holds. `outputs[0]` shares its bytes. The caller promises the kernel writes
/// none of the elements a pending write to `into` does, so it runs beside them (`core.bindOutDisjoint`).
pub fn launchInto(def: *Def, inputs: []const *Tensor, cfg: *Config, outputs: []*Tensor, into: ?*Tensor) LaunchError!void {
    return launchWith(def, inputs, cfg, outputs, if (into) |t| .{ .into = t } else .fresh);
}

/// `launchInto` of a fresh output the later launches write into: bound as theirs are, it runs beside them.
pub fn launchDisjoint(def: *Def, inputs: []const *Tensor, cfg: *Config, outputs: []*Tensor) LaunchError!void {
    return launchWith(def, inputs, cfg, outputs, .disjoint);
}

/// `launch` writing its outputs into `dsts`, the caller's tensors, instead of new ones: a pending update is settled
/// first, and a shared or strided destination is copied first so the write never reaches another holder.
pub fn launchInPlace(def: *Def, inputs: []const *Tensor, cfg: *Config, dsts: []*Tensor) LaunchError!void {
    if (inputs.len != def.inputs.len or cfg.n_out != def.outputs.len or dsts.len != cfg.n_out) return error.TensorShape;
    for (dsts, 0..) |dst, i| {
        if (dst.dtype != cfg.out_dtypes[i]) return error.TensorShape;
    }

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

    // A dry recording looks no pipeline up (`core.dry_pipeline`), and caches none.
    const pipeline = if (core.dryOn()) core.dry_pipeline else pipelineOf(def, bound[0..n_bound], cfg) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => |other| @as(LaunchError, @errorCast(other)),
    };
    const threads = @as(usize, @intCast(cfg.group[0])) * @as(usize, @intCast(cfg.group[1])) * @as(usize, @intCast(cfg.group[2]));
    const granted = if (std.mem.startsWith(u8, def.name, grant_cap_name)) @min(pipeline.max_threads, grant_cap) else pipeline.max_threads;
    if (threads > granted) {
        refused_limit = granted;
        refused_threads = threads;
        return error.ThreadGroupTooLarge;
    }

    for (dsts) |dst| {
        if (dst.pending != null) try ops.settle(dst);
        const in_place = !dst.repeats() and dst.rowContiguous() and try core.sole(dst.store);
        if (!in_place) core.adopt(dst, try ops.contiguousCopy(dst));
        core.countUpdate(in_place);
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
    for (dsts) |dst| {
        core.bindOut(p, dst, index);
        index += 1;
    }
    const grid: metal.Size = .{ .x = @intCast(cfg.grid[0]), .y = @intCast(cfg.grid[1]), .z = @intCast(cfg.grid[2]) };
    core.dispatch(p, grid, .{
        .x = @min(@as(usize, @intCast(cfg.group[0])), grid.x),
        .y = @min(@as(usize, @intCast(cfg.group[1])), grid.y),
        .z = @min(@as(usize, @intCast(cfg.group[2])), grid.z),
    });
}

/// `launch` whose last output (u32 [core.counter_page_words]) is the engine's counter page, read and written: counted
/// launches run in the order they were made, and each must leave the page's counters zero. `outputs`' last is a handle
/// on the page.
pub fn launchCounted(def: *Def, inputs: []const *Tensor, cfg: *Config, outputs: []*Tensor) LaunchError!void {
    return launchWith(def, inputs, cfg, outputs, .counted);
}

/// Where a launch's outputs go: fresh; its one output over a tensor (bound disjoint); fresh and bound as the writes
/// into it are; or its last output the counter page.
const Outputs = union(enum) { fresh, into: *Tensor, disjoint, counted };

fn launchWith(def: *Def, inputs: []const *Tensor, cfg: *Config, outputs: []*Tensor, how: Outputs) LaunchError!void {
    if (core.intoArmed()) |dst| for (inputs) |in| if (in.store == dst) return core.refusing("a launch whose output `into` names reads it too", error.TensorUnsupported);
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

    // A dry recording looks no pipeline up (`core.dry_pipeline`), and caches none.
    const pipeline = if (core.dryOn()) core.dry_pipeline else pipelineOf(def, bound[0..n_bound], cfg) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => |other| @as(LaunchError, @errorCast(other)),
    };
    const threads = @as(usize, @intCast(cfg.group[0])) * @as(usize, @intCast(cfg.group[1])) * @as(usize, @intCast(cfg.group[2]));
    const granted = if (std.mem.startsWith(u8, def.name, grant_cap_name)) @min(pipeline.max_threads, grant_cap) else pipeline.max_threads;
    if (threads > granted) {
        refused_limit = granted;
        refused_threads = threads;
        return error.ThreadGroupTooLarge;
    }

    var made: usize = 0;
    errdefer for (outputs[0..made]) |t| core.release(t);
    for (0..cfg.n_out) |i| {
        outputs[i] = if (how == .into and i == 0) blk: {
            const t = how.into;
            const shape = cfg.out_shapes[i][0..cfg.out_ndims[i]];
            if (t.dtype != cfg.out_dtypes[i] or t.ndim != shape.len or !t.rowContiguous()) return error.TensorShape;
            for (t.shape[0..t.ndim], shape) |a, b| if (a != b) return error.TensorShape;
            break :blk try core.share(t);
        } else if (how == .counted and i + 1 == cfg.n_out) blk: {
            const shape = cfg.out_shapes[i][0..cfg.out_ndims[i]];
            if (cfg.out_dtypes[i] != .uint32 or shape.len != 1 or shape[0] != core.counter_page_words) return error.TensorShape;
            break :blk try core.counterPage();
        } else try core.freshOut(cfg.out_dtypes[i], cfg.out_shapes[i][0..cfg.out_ndims[i]]);
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
    for (outputs, 0..) |out, i| {
        if ((how == .into and i == 0) or how == .disjoint) core.bindOutDisjoint(p, out, index) else core.bindOut(p, out, index);
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
    const def = try Def.create("sushi_test_scale", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = float(x[thread_position_in_grid.x]) * float(SCALE);", "", true, false);
    const x = try floats(&.{ 1, 2, 3, 4 });
    defer core.release(x);
    var cfg: Config = .{};
    try oneOutput(&cfg, 4);
    try cfg.addTemplate("SCALE", .{ .int = 101 });
    const before = key_builds;
    const made = core.pipelines;
    try testing.expectEqual([4]f32{ 101, 202, 303, 404 }, try run(def, x, &cfg));
    try testing.expectEqual([4]f32{ 101, 202, 303, 404 }, try run(def, x, &cfg));
    try testing.expectEqual([4]f32{ 101, 202, 303, 404 }, try run(def, x, &cfg));
    try testing.expectEqual(before + 1, key_builds);
    try testing.expectEqual(made + 1, core.pipelines);

    // A config made anew for each launch, as the DeepSeek kernels do, answers the same way.
    const fresh_before = key_builds;
    for (0..3) |_| {
        var again: Config = .{};
        try oneOutput(&again, 4);
        try again.addTemplate("SCALE", .{ .int = 101 });
        try testing.expectEqual([4]f32{ 101, 202, 303, 404 }, try run(def, x, &again));
    }
    try testing.expectEqual(fresh_before, key_builds);
    try testing.expectEqual(made + 1, core.pipelines);
}

test "kernel: a changed input dtype launches the pipeline of that dtype" {
    try mlx.requireGpu();
    const def = try Def.create("sushi_test_scale", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = float(x[thread_position_in_grid.x]) * float(SCALE);", "", true, false);
    const xf = try floats(&.{ 1, 2, 3, 4 });
    defer core.release(xf);
    const xi = try ints(&.{ 10, 20, 30, 40 });
    defer core.release(xi);
    var cfg: Config = .{};
    try oneOutput(&cfg, 4);
    try cfg.addTemplate("SCALE", .{ .int = 103 });
    try testing.expectEqual([4]f32{ 103, 206, 309, 412 }, try run(def, xf, &cfg));
    const made = core.pipelines;
    const keys = key_builds;
    try testing.expectEqual([4]f32{ 1030, 2060, 3090, 4120 }, try run(def, xi, &cfg));
    try testing.expectEqual(made + 1, core.pipelines);
    try testing.expectEqual(keys + 1, key_builds);
    try testing.expectEqual([4]f32{ 103, 206, 309, 412 }, try run(def, xf, &cfg));
    try testing.expectEqual(made + 1, core.pipelines);
    try testing.expectEqual(keys + 1, key_builds);
}

test "kernel: two kernels launched in turn through one config each take their own pipeline" {
    try mlx.requireGpu();
    const scaled = try Def.create("sushi_test_scale", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = float(x[thread_position_in_grid.x]) * float(SCALE);", "", true, false);
    const native = try Def.create("sushi_test_native_probe", &.{"x"}, &.{"out"}, "uint i = thread_position_in_grid.x;\nout[i] = x[i] * float(SCALE) + float(threads_per_threadgroup.x > 0);", "", true, false);
    const x = try floats(&.{ 1, 2, 3, 4 });
    defer core.release(x);
    var cfg: Config = .{};
    try oneOutput(&cfg, 4);
    try cfg.addTemplate("SCALE", .{ .int = 2 });
    try testing.expectEqual([4]f32{ 2, 4, 6, 8 }, try run(scaled, x, &cfg));
    try testing.expectEqual([4]f32{ 3, 5, 7, 9 }, try run(native, x, &cfg));
    try testing.expectEqual([4]f32{ 2, 4, 6, 8 }, try run(scaled, x, &cfg));
}

test "kernel: a template value added to a launched config changes the pipeline the next launch takes" {
    try mlx.requireGpu();
    const source = "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] + float(K);";
    const def = try Def.create("sushi_test_add", &.{"x"}, &.{"out"}, source, "", true, false);
    const x = try floats(&.{ 1, 2, 3, 4 });
    defer core.release(x);
    var cfg: Config = .{};
    try oneOutput(&cfg, 4);
    try cfg.addTemplate("K", .{ .int = 105 });
    try testing.expectEqual([4]f32{ 106, 107, 108, 109 }, try run(def, x, &cfg));
    const made = core.pipelines;
    try cfg.addTemplate("L", .{ .int = 1 });
    try testing.expectEqual([4]f32{ 106, 107, 108, 109 }, try run(def, x, &cfg));
    try testing.expectEqual(made + 1, core.pipelines);

    // Two configs of one kernel at different values keep their own pipelines.
    var other: Config = .{};
    try oneOutput(&other, 4);
    try other.addTemplate("K", .{ .int = 109 });
    try testing.expectEqual([4]f32{ 110, 111, 112, 113 }, try run(def, x, &other));
    try testing.expectEqual([4]f32{ 106, 107, 108, 109 }, try run(def, x, &cfg));
    try testing.expectEqual([4]f32{ 110, 111, 112, 113 }, try run(def, x, &other));
}

test "kernel: more instantiations than a kernel remembers each launch their own pipeline" {
    try mlx.requireGpu();
    const def = try Def.create("sushi_test_add", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] + float(K);", "", true, false);
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
    const def = try Def.create("sushi_test_wide_targs", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] + " ++ sum ++ ";", "", true, false);
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

test "kernel: launches into one tensor over disjoint ranges run with no barrier between, and a reader after sees them all" {
    try mlx.requireGpu();
    const n: c_int = 1024;
    const part = try Def.create("sushi_test_range_copy", &.{ "x", "r" }, &.{"out"}, "uint i = thread_position_in_grid.x; if (i >= uint(r[0]) && i < uint(r[1])) out[i] = x[i];", "", true, false);
    const twice = try Def.create("sushi_test_scale", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = float(x[thread_position_in_grid.x]) * float(SCALE);", "", true, false);
    var vals: [1024]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @floatFromInt(i);
    const x = try floats(&vals);
    defer core.release(x);
    var cfg: Config = .{};
    try cfg.addOutput(&.{n}, .float32);
    cfg.grid = .{ n, 1, 1 };
    cfg.group = .{ 256, 1, 1 };
    var scale_cfg = cfg;
    try scale_cfg.addTemplate("SCALE", .{ .int = 2 });
    var ranges: [4]*Tensor = undefined;
    for (&ranges, 0..) |*r, k| r.* = try ints(&.{ @intCast(k * 256), @intCast(k * 256 + 256) });
    defer for (ranges) |r| core.release(r);
    var plane: [1]*Tensor = undefined;
    const barriers = core.stats().barriers;
    try launchDisjoint(part, &.{ x, ranges[0] }, &cfg, &plane);
    for (ranges[1..]) |r| {
        var next: [1]*Tensor = undefined;
        try launchInto(part, &.{ x, r }, &cfg, &next, plane[0]);
        core.release(plane[0]);
        plane[0] = next[0];
    }
    defer core.release(plane[0]);
    try testing.expectEqual(barriers, core.stats().barriers);
    var out: [1]*Tensor = undefined;
    try launch(twice, &.{plane[0]}, &scale_cfg, &out);
    defer core.release(out[0]);
    const got = std.mem.bytesAsSlice(f32, try core.hostBytes(out[0]));
    for (got[0..1024], 0..) |v, i| try testing.expectEqual(@as(f32, @floatFromInt(2 * i)), v);
}

test "kernel: a launch's output is the tensor `core.into` names, recorded state-owned; a launch that reads it, an elementwise op and a recording begun with one armed are refused by name" {
    try mlx.requireGpu();
    const plan = @import("plan/plan_api.zig");
    const def = try Def.create("sushi_test_scale", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = float(x[thread_position_in_grid.x]) * float(SCALE);", "", true, false);
    const x = try floats(&.{ 1, 2, 3, 4 });
    defer core.release(x);
    const dst = try core.fresh(.float32, &.{4});
    defer core.release(dst);
    var cfg: Config = .{};
    try oneOutput(&cfg, 4);
    try cfg.addTemplate("SCALE", .{ .int = 2 });
    var rec = plan.Recorder.begin(testing.allocator);
    defer rec.deinit();
    {
        try core.beginRecording(.real, &rec);
        defer core.endRecording() catch {};
        const g = try core.into(dst);
        defer g.end();
        var out: [1]*Tensor = undefined;
        try launch(def, &.{x}, &cfg, &out);
        defer core.release(out[0]);
        try g.taken(out[0]);
        try testing.expect(out[0].store == dst.store);
    }
    try testing.expect(rec.asks.items[0].external);
    try testing.expectEqualSlices(f32, &.{ 2, 4, 6, 8 }, @alignCast(std.mem.bytesAsSlice(f32, try core.hostBytes(dst))));
    {
        const g = try core.into(dst);
        defer g.end();
        var out: [1]*Tensor = undefined;
        try testing.expectError(error.TensorUnsupported, launch(def, &.{dst}, &cfg, &out));
    }
    {
        // An elementwise op's output, an input's copy or a cast as often as a result, takes no `into`.
        const g = try core.into(dst);
        defer g.end();
        const sum = try ops.binary(.add, x, x);
        defer core.release(sum);
        try testing.expect(sum.store != dst.store);
        try testing.expectError(error.TensorUnsupported, g.taken(sum));
        try testing.expectError(error.TensorUnsupported, core.beginRecording(.real, &rec));
    }
}

test "kernel: a counted launch's last threadgroup reads what every threadgroup wrote and leaves the counters zero, and counted launches on the same inputs run in order" {
    try mlx.requireGpu();
    const groups = 256;
    const src =
        \\const uint g = threadgroup_position_in_grid.x;
        \\device atomic_uint *c = (device atomic_uint *)ctr;
        \\threadgroup bool last[1];
        \\if (thread_index_in_threadgroup == 0) {
        \\  atomic_store_explicit(c + CTR + g, x[g], memory_order_relaxed);
        \\  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        \\  last[0] = atomic_fetch_add_explicit(c, 1u, memory_order_relaxed) == uint(G - 1);
        \\  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        \\}
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\if (!last[0] || thread_index_in_threadgroup != 0) return;
        \\uint sum = 0;
        \\for (uint i = 0; i < uint(G); i++) sum += atomic_load_explicit(c + CTR + i, memory_order_relaxed);
        \\out[0] = sum;
        \\atomic_store_explicit(c, 0u, memory_order_relaxed);
    ;
    const def = try Def.create("sushi_test_counted_sum", &.{"x"}, &.{ "out", "ctr" }, src, "", true, false);
    var vals: [groups]u32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @intCast(3 * i + 1);
    const x = try core.fromHost(std.mem.sliceAsBytes(&vals), .uint32, &.{groups});
    defer core.release(x);
    var cfg: Config = .{};
    try cfg.addOutput(&.{1}, .uint32);
    try cfg.addOutput(&.{core.counter_page_words}, .uint32);
    try cfg.addTemplate("G", .{ .int = groups });
    try cfg.addTemplate("CTR", .{ .int = core.counter_words });
    cfg.grid = .{ groups * 32, 1, 1 };
    cfg.group = .{ 32, 1, 1 };
    var outs: [2][2]*Tensor = undefined;
    const barriers = core.stats().barriers;
    for (&outs) |*o| try launchCounted(def, &.{x}, &cfg, o);
    defer for (outs) |o| for (o) |t| core.release(t);
    var want: u32 = 0;
    for (vals) |v| want += v;
    for (outs) |o| try testing.expectEqual(want, std.mem.bytesAsSlice(u32, try core.hostBytes(o[0]))[0]);
    // The second launch read nothing the first wrote but the page, and still waited for it.
    try testing.expect(core.stats().barriers > barriers);
    for (std.mem.bytesAsSlice(u32, try core.hostBytes(outs[1][1]))[0..core.counter_words]) |w| try testing.expectEqual(@as(u32, 0), w);
}

test "kernel: an unregistered custom kernel is refused with MetalCompile and compiles no source" {
    try mlx.requireGpu();
    const def = try Def.create("sushi_unregistered_kernel_probe", &.{"x"}, &.{"out"}, "out[thread_position_in_grid.x] = x[thread_position_in_grid.x];", "", true, false);
    const x = try floats(&.{ 1, 2, 3, 4 });
    defer core.release(x);
    var cfg: Config = .{};
    try oneOutput(&cfg, 4);
    var out: [1]*Tensor = undefined;
    const before = core.pipelines;
    try testing.expectError(error.MetalCompile, launch(def, &.{x}, &cfg, &out));
    try testing.expectEqual(before, core.pipelines);
}
