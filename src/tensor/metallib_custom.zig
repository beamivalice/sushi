//! Pure-Zig specification, Metal unit emitter, and runtime resolver for Sushi Cooler's
//! precompiled custom kernels (`sushi.metallib`).
//!
//! Single source of truth imported by:
//! - `src/metallib_gen_main.zig` (build-time `.metal` codegen),
//! - `src/tensor/kernel.zig` (runtime lookup -> `stock.kernel` / `stock.kernelConstants`),
//! - `src/tensor/stock.zig` (completeness test verifying every catalogued function by construction).
//!
//! Imports no MLX, Metal, or Apple framework symbols so host-only build tools compile it directly.

const std = @import("std");
const spec = @import("metallib_spec.zig");

pub const Dtype = spec.Dtype;

pub const max_inputs = 32;
pub const max_outputs = 8;
pub const max_ct = 16;
pub const max_fc_ints = spec.fc_int_count;
pub const max_fc_bools = spec.fc_bool_count;

pub const Flag = struct { index: usize, value: bool };
pub const IntConstant = struct { index: usize, value: i32 };

pub const CtVal = union(enum) {
    int: i32,
    boolean: bool,
    dtype: Dtype,

    pub fn eql(a: CtVal, b: CtVal) bool {
        return switch (a) {
            .int => |x| b == .int and x == b.int,
            .boolean => |x| b == .boolean and x == b.boolean,
            .dtype => |x| b == .dtype and x == b.dtype,
        };
    }
};

pub const RuntimeTarg = struct {
    name: []const u8,
    value: CtVal,
};

pub const RuntimeInput = struct {
    name: []const u8,
    dtype: Dtype,
    scalar: bool,
};

pub const ParamKind = enum {
    /// Compile-time template int parameter.
    ct_int,
    /// Compile-time template bool parameter.
    ct_bool,
    /// Compile-time template typename parameter.
    ct_dtype,
    /// Implicit dtype variable (not in `cfg.targs`), varying input/output buffer dtypes.
    io_dtype,
    /// Runtime `[[function_constant(slot)]]` int parameter (`const int NAME = FC_I<slot>;`).
    fc_int,
    /// Runtime `[[function_constant(slot)]]` bool parameter (`const bool NAME = FC_B<slot>;`).
    fc_bool,
    /// Unused template parameter accepted at runtime and omitted from Metal.
    ignored,

    pub fn isVariant(k: ParamKind) bool {
        return k == .ct_int or k == .ct_bool or k == .ct_dtype or k == .io_dtype;
    }

    pub fn isTemplate(k: ParamKind) bool {
        return k == .ct_int or k == .ct_bool or k == .ct_dtype;
    }
};

pub const ParamSpec = struct {
    name: []const u8,
    kind: ParamKind,
    /// Explicit function-constant index override (null = auto-assigned sequentially).
    slot: ?u32 = null,
    /// Allowed values for `ct_int` when `ct_tuples` is null.
    ints: []const i32 = &.{},
    /// Allowed values for `ct_bool` when `ct_tuples` is null.
    bools: []const bool = &.{ false, true },
    /// Allowed values for `ct_dtype` / `io_dtype` when `ct_tuples` is null.
    dtypes: []const Dtype = &.{},
};

pub const TypeRef = union(enum) {
    fixed: Dtype,
    /// Binds to a `ct_dtype` or `io_dtype` param by name.
    param: []const u8,
};

pub const InputSlot = struct {
    name: []const u8,
    ty: TypeRef,
    /// True iff the input is passed as a 0-d scalar (`const constant T&`), false for `const device T*`.
    scalar: bool = false,
};

pub const OutputSlot = struct {
    name: []const u8,
    ty: TypeRef,
};

pub const SourceFn = *const fn (std.mem.Allocator, usize) anyerror![]const u8;
pub const FilterFn = *const fn ([]const CtVal) bool;

pub const KernelSpec = struct {
    /// Runtime kernel name passed to `mlx_fast_metal_kernel_new` / `kernel.Def.create`.
    name: []const u8,
    /// Optional disambiguator appended to the emitted metallib function name (for colliding names).
    disambig: []const u8 = "",
    /// True for NAX (`MetalPerformancePrimitives`) kernels — strictly isolated into `*_nax.metal` units.
    nax: bool = false,
    /// Metal header text (optional). `#include <...>` lines are stripped at emit time.
    header: []const u8 = "",
    /// Metal kernel body source (used when `source_fn` is null).
    source: []const u8 = "",
    /// Optional dynamic header/source builders called only at build time (`gen-metallib`).
    header_fn: ?SourceFn = null,
    source_fn: ?SourceFn = null,
    source_arg: usize = 0,
    inputs: []const InputSlot,
    outputs: []const OutputSlot,
    params: []const ParamSpec = &.{},
    /// Starting FC slots when auto-assigning `fc_int` / `fc_bool` params.
    fc_int_base: u32 = spec.fc_int_base,
    fc_bool_base: u32 = spec.fc_bool_base,
    /// Optional explicit list of valid tuples for variant params (overrides Cartesian product).
    ct_tuples: ?[]const []const CtVal = null,
    /// Optional filter predicate over variant param tuples.
    filter: ?FilterFn = null,
};

/// Result of resolving a runtime custom kernel launch to a precompiled `sushi.metallib` function.
pub const Resolved = struct {
    fn_buf: [512]u8 = undefined,
    fn_len: usize = 0,
    hash_buf: [1024]u8 = undefined,
    hash_len: usize = 0,
    flags: [max_fc_bools]Flag = undefined,
    n_flags: usize = 0,
    ints: [max_fc_ints]IntConstant = undefined,
    n_ints: usize = 0,

    pub fn fnName(self: *const Resolved) []const u8 {
        return self.fn_buf[0..self.fn_len];
    }

    pub fn hashKey(self: *const Resolved) []const u8 {
        return self.hash_buf[0..self.hash_len];
    }

    pub fn flagSlice(self: *const Resolved) []const Flag {
        return self.flags[0..self.n_flags];
    }

    pub fn intSlice(self: *const Resolved) []const IntConstant {
        return self.ints[0..self.n_ints];
    }

    pub fn finalizeHash(self: *Resolved) error{OutOfMemory}!void {
        var w: std.ArrayList(u8) = .initBuffer(&self.hash_buf);
        try w.appendSliceBounded(self.fnName());
        if (self.n_flags > 0 or self.n_ints > 0) {
            try w.appendBounded('|');
            var first = true;
            for (self.flagSlice()) |f| {
                if (!first) try w.appendBounded(',');
                first = false;
                var tmp: [32]u8 = undefined;
                const s = std.fmt.bufPrint(&tmp, "b{d}={d}", .{ f.index, @intFromBool(f.value) }) catch return error.OutOfMemory;
                try w.appendSliceBounded(s);
            }
            for (self.intSlice()) |c| {
                if (!first) try w.appendBounded(',');
                first = false;
                var tmp: [32]u8 = undefined;
                const s = std.fmt.bufPrint(&tmp, "i{d}={d}", .{ c.index, c.value }) catch return error.OutOfMemory;
                try w.appendSliceBounded(s);
            }
        }
        self.hash_len = w.items.len;
    }
};

/// Same 20 Metal attributes in the exact order `src/tensor/kernel.zig` scans them.
pub const metal_attributes = [_][2][]const u8{
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

pub fn variantParamCount(ks: *const KernelSpec) usize {
    var n: usize = 0;
    for (ks.params) |p| if (p.kind.isVariant()) {
        n += 1;
    };
    return n;
}

pub fn variantParamIndex(ks: *const KernelSpec, param_name: []const u8) ?usize {
    var idx: usize = 0;
    for (ks.params) |p| {
        if (!p.kind.isVariant()) continue;
        if (std.mem.eql(u8, p.name, param_name)) return idx;
        idx += 1;
    }
    return null;
}

fn findParam(ks: *const KernelSpec, param_name: []const u8) ?ParamSpec {
    for (ks.params) |p| if (std.mem.eql(u8, p.name, param_name)) return p;
    return null;
}

pub fn resolveTypeRef(ks: *const KernelSpec, tr: TypeRef, vals: []const CtVal) Dtype {
    return switch (tr) {
        .fixed => |d| d,
        .param => |pname| blk: {
            const idx = variantParamIndex(ks, pname) orelse @panic("metallib_custom: TypeRef names unknown variant param");
            break :blk vals[idx].dtype;
        },
    };
}

/// Canonical metallib function name for one instantiation of `ks`.
/// Shared identically by `gen-metallib`, runtime resolution, and `stock.zig`'s completeness test.
pub fn customFnName(
    buf: []u8,
    ks: *const KernelSpec,
    vals: []const CtVal,
    in_dtypes: []const Dtype,
    out_dtypes: []const Dtype,
) error{OutOfMemory}![]const u8 {
    var w: std.ArrayList(u8) = .initBuffer(buf);
    try w.appendSliceBounded("sushi_k_");
    try w.appendSliceBounded(ks.name);
    if (ks.disambig.len > 0) {
        try w.appendBounded('_');
        try w.appendSliceBounded(ks.disambig);
    }
    var vi: usize = 0;
    for (ks.params) |p| {
        if (!p.kind.isVariant()) continue;
        defer vi += 1;
        if (!p.kind.isTemplate()) continue;
        try w.appendBounded('_');
        switch (vals[vi]) {
            .int => |v| {
                var tmp: [24]u8 = undefined;
                const s = if (v < 0)
                    std.fmt.bufPrint(&tmp, "m{d}", .{@as(i64, -@as(i64, v))}) catch return error.OutOfMemory
                else
                    std.fmt.bufPrint(&tmp, "{d}", .{v}) catch return error.OutOfMemory;
                try w.appendSliceBounded(s);
            },
            .boolean => |v| try w.appendBounded(if (v) '1' else '0'),
            .dtype => |d| try w.appendSliceBounded(d.shortName()),
        }
    }
    try w.appendSliceBounded("__");
    for (ks.inputs, 0..) |inp, i| {
        if (i > 0) try w.appendBounded('_');
        try w.appendSliceBounded(in_dtypes[i].shortName());
        if (inp.scalar) try w.appendBounded('s');
    }
    try w.appendSliceBounded("__");
    for (out_dtypes, 0..) |d, i| {
        if (i > 0) try w.appendBounded('_');
        try w.appendSliceBounded(d.shortName());
    }
    return w.items;
}

/// Visits every valid instantiation of `ks`, calling `onFn(ctx, ks, vals, in_dtypes, out_dtypes, fn_name)`.
pub fn visitSpec(
    ks: *const KernelSpec,
    ctx: anytype,
    comptime onFn: fn (@TypeOf(ctx), *const KernelSpec, []const CtVal, []const Dtype, []const Dtype, []const u8) anyerror!void,
) !void {
    const n_var = variantParamCount(ks);
    var vals: [max_ct]CtVal = undefined;
    var in_dt: [max_inputs]Dtype = undefined;
    var out_dt: [max_outputs]Dtype = undefined;
    var name_buf: [512]u8 = undefined;

    if (ks.ct_tuples) |tuples| {
        for (tuples) |tup| {
            std.debug.assert(tup.len == n_var);
            if (ks.filter) |flt| if (!flt(tup)) continue;
            for (ks.inputs, 0..) |inp, i| in_dt[i] = resolveTypeRef(ks, inp.ty, tup);
            for (ks.outputs, 0..) |out, i| out_dt[i] = resolveTypeRef(ks, out.ty, tup);
            const fname = customFnName(&name_buf, ks, tup, in_dt[0..ks.inputs.len], out_dt[0..ks.outputs.len]) catch unreachable;
            try onFn(ctx, ks, tup, in_dt[0..ks.inputs.len], out_dt[0..ks.outputs.len], fname);
        }
        return;
    }

    var var_params: [max_ct]ParamSpec = undefined;
    var vi: usize = 0;
    for (ks.params) |p| if (p.kind.isVariant()) {
        var_params[vi] = p;
        vi += 1;
    };
    const Rec = struct {
        fn go(
            k: *const KernelSpec,
            vps: []const ParamSpec,
            depth: usize,
            vbuf: *[max_ct]CtVal,
            idt: *[max_inputs]Dtype,
            odt: *[max_outputs]Dtype,
            nbuf: *[512]u8,
            c: @TypeOf(ctx),
        ) !void {
            if (depth == vps.len) {
                const tup = vbuf[0..vps.len];
                if (k.filter) |flt| if (!flt(tup)) return;
                for (k.inputs, 0..) |inp, i| idt[i] = resolveTypeRef(k, inp.ty, tup);
                for (k.outputs, 0..) |out, i| odt[i] = resolveTypeRef(k, out.ty, tup);
                const fname = customFnName(nbuf, k, tup, idt[0..k.inputs.len], odt[0..k.outputs.len]) catch unreachable;
                try onFn(c, k, tup, idt[0..k.inputs.len], odt[0..k.outputs.len], fname);
                return;
            }
            const p = vps[depth];
            switch (p.kind) {
                .ct_int => for (p.ints) |v| {
                    vbuf[depth] = .{ .int = v };
                    try go(k, vps, depth + 1, vbuf, idt, odt, nbuf, c);
                },
                .ct_bool => for (p.bools) |v| {
                    vbuf[depth] = .{ .boolean = v };
                    try go(k, vps, depth + 1, vbuf, idt, odt, nbuf, c);
                },
                .ct_dtype, .io_dtype => for (p.dtypes) |v| {
                    vbuf[depth] = .{ .dtype = v };
                    try go(k, vps, depth + 1, vbuf, idt, odt, nbuf, c);
                },
                else => unreachable,
            }
        }
    };
    try Rec.go(ks, var_params[0..n_var], 0, &vals, &in_dt, &out_dt, &name_buf, ctx);
}

/// Visits every function name emitted across a slice of `KernelSpec`s.
pub fn visitSpecs(
    specs: []const KernelSpec,
    ctx: anytype,
    comptime onName: fn (@TypeOf(ctx), []const u8) anyerror!void,
) !void {
    const Wrap = struct {
        inner: @TypeOf(ctx),
        fn cb(self: @This(), _: *const KernelSpec, _: []const CtVal, _: []const Dtype, _: []const Dtype, fname: []const u8) !void {
            try onName(self.inner, fname);
        }
    };
    for (specs) |*ks| {
        try visitSpec(ks, Wrap{ .inner = ctx }, Wrap.cb);
    }
}

/// Checks whether a concrete variant tuple `vals` is actually emitted by `ks`.
fn tupleEmitted(ks: *const KernelSpec, vals: []const CtVal) bool {
    if (ks.ct_tuples) |tuples| {
        for (tuples) |tup| {
            if (tup.len != vals.len) continue;
            var ok = true;
            for (tup, vals) |a, b| if (!a.eql(b)) {
                ok = false;
                break;
            };
            if (ok) {
                if (ks.filter) |flt| return flt(vals);
                return true;
            }
        }
        return false;
    }
    var vi: usize = 0;
    for (ks.params) |p| {
        if (!p.kind.isVariant()) continue;
        defer vi += 1;
        const v = vals[vi];
        const found = switch (p.kind) {
            .ct_int => blk: {
                for (p.ints) |x| if (x == v.int) break :blk true;
                break :blk false;
            },
            .ct_bool => blk: {
                for (p.bools) |x| if (x == v.boolean) break :blk true;
                break :blk false;
            },
            .ct_dtype, .io_dtype => blk: {
                for (p.dtypes) |x| if (x == v.dtype) break :blk true;
                break :blk false;
            },
            else => unreachable,
        };
        if (!found) return false;
    }
    if (ks.filter) |flt| return flt(vals);
    return true;
}

/// Attempts to match and resolve a runtime kernel launch against `ks`.
/// Returns `true` and fills `out` if matched and valid, `false` otherwise.
pub fn resolveSpec(
    ks: *const KernelSpec,
    name: []const u8,
    inputs: []const RuntimeInput,
    out_dtypes: []const Dtype,
    targs: []const RuntimeTarg,
    out: *Resolved,
) bool {
    if (!std.mem.eql(u8, ks.name, name)) return false;
    if (inputs.len != ks.inputs.len or out_dtypes.len != ks.outputs.len) return false;
    for (inputs, ks.inputs) |ri, si| {
        if (ri.scalar != si.scalar) return false;
        if (!std.mem.eql(u8, ri.name, si.name)) return false;
    }

    const n_var = variantParamCount(ks);
    var vals: [max_ct]CtVal = undefined;
    var val_set: [max_ct]bool = @splat(false);

    // Precompute FC slot indices for each param.
    var fc_slots: [64]u32 = undefined;
    var next_int: u32 = ks.fc_int_base;
    var next_bool: u32 = ks.fc_bool_base;
    for (ks.params, 0..) |p, pi| {
        switch (p.kind) {
            .fc_int => {
                fc_slots[pi] = p.slot orelse next_int;
                next_int = fc_slots[pi] + 1;
            },
            .fc_bool => {
                fc_slots[pi] = p.slot orelse next_bool;
                next_bool = fc_slots[pi] + 1;
            },
            else => fc_slots[pi] = 0,
        }
    }

    var param_seen: [64]bool = @splat(false);
    out.n_flags = 0;
    out.n_ints = 0;

    for (targs) |ta| {
        const pi = for (ks.params, 0..) |p, idx| {
            if (p.kind != .io_dtype and std.mem.eql(u8, p.name, ta.name)) break idx;
        } else return false;
        if (param_seen[pi]) return false;
        param_seen[pi] = true;
        const p = ks.params[pi];
        switch (p.kind) {
            .ct_int => {
                if (ta.value != .int) return false;
                const vi = variantParamIndex(ks, p.name).?;
                vals[vi] = ta.value;
                val_set[vi] = true;
            },
            .ct_bool => {
                const bv: bool = switch (ta.value) {
                    .boolean => |b| b,
                    .int => |iv| iv != 0,
                    .dtype => return false,
                };
                const vi = variantParamIndex(ks, p.name).?;
                vals[vi] = .{ .boolean = bv };
                val_set[vi] = true;
            },
            .ct_dtype => {
                if (ta.value != .dtype) return false;
                const vi = variantParamIndex(ks, p.name).?;
                vals[vi] = ta.value;
                val_set[vi] = true;
            },
            .fc_int => {
                const iv: i32 = switch (ta.value) {
                    .int => |v| v,
                    .boolean => |b| @intFromBool(b),
                    .dtype => return false,
                };
                if (out.n_ints >= max_fc_ints) return false;
                out.ints[out.n_ints] = .{ .index = fc_slots[pi], .value = iv };
                out.n_ints += 1;
            },
            .fc_bool => {
                const bv: bool = switch (ta.value) {
                    .boolean => |b| b,
                    .int => |iv| iv != 0,
                    .dtype => return false,
                };
                if (out.n_flags >= max_fc_bools) return false;
                out.flags[out.n_flags] = .{ .index = fc_slots[pi], .value = bv };
                out.n_flags += 1;
            },
            .ignored => {},
            .io_dtype => unreachable,
        }
    }

    // Verify every non-io_dtype non-ignored param was provided.
    for (ks.params, 0..) |p, pi| {
        if (p.kind == .io_dtype or p.kind == .ignored) continue;
        if (!param_seen[pi]) return false;
    }

    // Bind `.io_dtype` params and check all input/output dtypes.
    var in_dt: [max_inputs]Dtype = undefined;
    for (inputs, ks.inputs, 0..) |ri, si, i| {
        in_dt[i] = ri.dtype;
        switch (si.ty) {
            .fixed => |d| if (ri.dtype != d) return false,
            .param => |pname| {
                const vi = variantParamIndex(ks, pname) orelse return false;
                if (val_set[vi]) {
                    if (vals[vi].dtype != ri.dtype) return false;
                } else {
                    vals[vi] = .{ .dtype = ri.dtype };
                    val_set[vi] = true;
                }
            },
        }
    }
    for (out_dtypes, ks.outputs) |od, so| {
        switch (so.ty) {
            .fixed => |d| if (od != d) return false,
            .param => |pname| {
                const vi = variantParamIndex(ks, pname) orelse return false;
                if (val_set[vi]) {
                    if (vals[vi].dtype != od) return false;
                } else {
                    vals[vi] = .{ .dtype = od };
                    val_set[vi] = true;
                }
            },
        }
    }

    for (val_set[0..n_var]) |s| if (!s) return false;
    if (!tupleEmitted(ks, vals[0..n_var])) return false;

    const fname = customFnName(&out.fn_buf, ks, vals[0..n_var], in_dt[0..inputs.len], out_dtypes) catch return false;
    out.fn_len = fname.len;
    // Sort FC flags/ints by index so hash keys are canonical regardless of caller targ order.
    std.mem.sortUnstable(Flag, out.flags[0..out.n_flags], {}, struct {
        fn lt(_: void, a: Flag, b: Flag) bool {
            return a.index < b.index;
        }
    }.lt);
    std.mem.sortUnstable(IntConstant, out.ints[0..out.n_ints], {}, struct {
        fn lt(_: void, a: IntConstant, b: IntConstant) bool {
            return a.index < b.index;
        }
    }.lt);
    out.finalizeHash() catch return false;
    return true;
}

pub fn resolveSpecs(
    specs: []const KernelSpec,
    name: []const u8,
    inputs: []const RuntimeInput,
    out_dtypes: []const Dtype,
    targs: []const RuntimeTarg,
    out: *Resolved,
) bool {
    for (specs) |*ks| {
        if (resolveSpec(ks, name, inputs, out_dtypes, targs, out)) return true;
    }
    return false;
}

// ── Build-Time Metal Unit Emitter ──

const bf16_math_metal = @embedFile("mlx_bf16_math.metal");

/// Writes the shared file-scope preamble for a generated custom `.metal` unit.
pub fn writeUnitPreamble(out: *std.ArrayList(u8), a: std.mem.Allocator, is_nax: bool) !void {
    try out.appendSlice(a,
        \\#define SUSHI_METALLIB 1
        \\#include <metal_stdlib>
        \\#include <metal_simdgroup>
        \\#include <metal_simdgroup_matrix>
        \\
    );
    if (is_nax) {
        try out.appendSlice(a, "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
    }
    try out.appendSlice(a,
        \\using namespace metal;
        \\typedef bfloat bfloat16_t;
        \\typedef half float16_t;
        \\
    );
    try out.appendSlice(a, bf16_math_metal);
    try out.appendSlice(a, "\n");
    for (0..spec.fc_int_count) |i| {
        try out.print(a, "constant int FC_I{d} [[function_constant({d})]];\n", .{ i, spec.fc_int_base + i });
    }
    for (0..spec.fc_bool_count) |i| {
        try out.print(a, "constant bool FC_B{d} [[function_constant({d})]];\n", .{ i, spec.fc_bool_base + i });
    }
    try out.appendSlice(a, "\n");
}

/// Strips `//` and `/* ... */` comments so `_shape`/`_strides`/`_ndim` and attribute scanning
/// matches `metal_minify.minify` identically.
pub fn stripComments(a: std.mem.Allocator, src: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.ensureTotalCapacity(a, src.len);
    var i: usize = 0;
    while (i < src.len) {
        if (src[i] == '"' or src[i] == '\'') {
            const q = src[i];
            out.appendAssumeCapacity(q);
            i += 1;
            while (i < src.len) {
                if (src[i] == '\\' and i + 1 < src.len) {
                    out.appendAssumeCapacity(src[i]);
                    out.appendAssumeCapacity(src[i + 1]);
                    i += 2;
                    continue;
                }
                out.appendAssumeCapacity(src[i]);
                if (src[i] == q) {
                    i += 1;
                    break;
                }
                i += 1;
            }
        } else if (src[i] == '/' and i + 1 < src.len and src[i + 1] == '/') {
            while (i < src.len and src[i] != '\n') i += 1;
        } else if (src[i] == '/' and i + 1 < src.len and src[i + 1] == '*') {
            i = if (std.mem.indexOfPos(u8, src, i + 2, "*/")) |e| e + 2 else src.len;
            out.appendAssumeCapacity(' ');
        } else {
            out.appendAssumeCapacity(src[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(a);
}

/// Appends `text` with `#include <...>` lines stripped (already in unit preamble).
pub fn appendHeaderFiltered(out: *std.ArrayList(u8), a: std.mem.Allocator, text: []const u8) !void {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trimStart(u8, line, " \t");
        if (std.mem.startsWith(u8, trimmed, "#include <") or std.mem.startsWith(u8, trimmed, "#include<")) {
            continue;
        }
        try out.appendSlice(a, line);
        try out.append(a, '\n');
    }
}

/// Emits `#undef <IDENT>` for every `#define <IDENT>` appearing in `texts`, deduplicated.
pub fn appendUndefs(out: *std.ArrayList(u8), a: std.mem.Allocator, texts: []const []const u8) !void {
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(a);
    for (texts) |text| {
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |line| {
            const trimmed = std.mem.trimStart(u8, line, " \t");
            if (!std.mem.startsWith(u8, trimmed, "#define")) continue;
            if (trimmed.len <= 7 or (trimmed[7] != ' ' and trimmed[7] != '\t')) continue;
            const rest = std.mem.trimStart(u8, trimmed[7..], " \t");
            var end: usize = 0;
            while (end < rest.len and (std.ascii.isAlphanumeric(rest[end]) or rest[end] == '_')) end += 1;
            if (end == 0) continue;
            const macro = rest[0..end];
            if (std.mem.eql(u8, macro, "SUSHI_METALLIB")) continue;
            if (seen.contains(macro)) continue;
            try seen.put(a, macro, {});
            try out.print(a, "#undef {s}\n", .{macro});
        }
    }
}

const IoSigKey = struct {
    n_in: usize,
    n_out: usize,
    /// Only concrete dtypes for non-ct_dtype slots; ct_dtype slots share the same key (0xFF).
    in_sig: [max_inputs]u8,
    out_sig: [max_outputs]u8,
    in_dt: [max_inputs]Dtype,
    out_dt: [max_outputs]Dtype,
};

fn ioSigKeyOf(ks: *const KernelSpec, in_dt: []const Dtype, out_dt: []const Dtype) IoSigKey {
    var k: IoSigKey = .{
        .n_in = ks.inputs.len,
        .n_out = ks.outputs.len,
        .in_sig = @splat(0),
        .out_sig = @splat(0),
        .in_dt = @splat(.bool_),
        .out_dt = @splat(.bool_),
    };
    for (ks.inputs, 0..) |inp, i| {
        k.in_dt[i] = in_dt[i];
        const is_ct = switch (inp.ty) {
            .fixed => false,
            .param => |pn| findParam(ks, pn).?.kind == .ct_dtype,
        };
        k.in_sig[i] = if (is_ct) 0xFF else @intFromEnum(in_dt[i]);
    }
    for (ks.outputs, 0..) |o, i| {
        k.out_dt[i] = out_dt[i];
        const is_ct = switch (o.ty) {
            .fixed => false,
            .param => |pn| findParam(ks, pn).?.kind == .ct_dtype,
        };
        k.out_sig[i] = if (is_ct) 0xFF else @intFromEnum(out_dt[i]);
    }
    return k;
}

fn ioSigEql(a: IoSigKey, b: IoSigKey) bool {
    if (a.n_in != b.n_in or a.n_out != b.n_out) return false;
    return std.mem.eql(u8, a.in_sig[0..a.n_in], b.in_sig[0..b.n_in]) and
        std.mem.eql(u8, a.out_sig[0..a.n_out], b.out_sig[0..b.n_out]);
}

/// Emits one `KernelSpec` inside `namespace k_<ns_idx> { ... }` with exact buffer layout,
/// FC bindings, template instantiations, and macro cleanup.
pub fn emitSpec(out: *std.ArrayList(u8), a: std.mem.Allocator, ks: *const KernelSpec, ns_idx: usize) !void {
    const header = if (ks.header_fn) |hf| try hf(a, ks.source_arg) else ks.header;
    const source = if (ks.source_fn) |sf| try sf(a, ks.source_arg) else ks.source;
    const stripped = try stripComments(a, source);
    defer a.free(stripped);

    // Scan shape_infos and attributes identically to `src/tensor/kernel.zig:182-197`.
    var shape_infos: [max_inputs][3]bool = @splat(.{ false, false, false });
    for (ks.inputs, 0..) |inp, i| {
        inline for (.{ "_shape", "_strides", "_ndim" }, 0..) |suffix, k| {
            const needle = try std.mem.concat(a, u8, &.{ inp.name, suffix });
            defer a.free(needle);
            shape_infos[i][k] = std.mem.indexOf(u8, stripped, needle) != null;
        }
    }
    var used_attrs: [metal_attributes.len]bool = @splat(false);
    var n_attrs: usize = 0;
    for (metal_attributes, 0..) |attr, i| {
        if (std.mem.indexOf(u8, stripped, attr[0]) != null) {
            used_attrs[i] = true;
            n_attrs += 1;
        }
    }

    // Collect all instantiations and group by IO signature.
    const Inst = struct {
        sig_idx: usize,
        vals: [max_ct]CtVal,
        n_vals: usize,
        fname: []u8,
    };
    const Collect = struct {
        a: std.mem.Allocator,
        sigs: std.ArrayList(IoSigKey) = .empty,
        insts: std.ArrayList(Inst) = .empty,

        fn cb(self: *@This(), k: *const KernelSpec, vals: []const CtVal, in_dt: []const Dtype, out_dt: []const Dtype, fname: []const u8) !void {
            const sk = ioSigKeyOf(k, in_dt, out_dt);
            const sig_idx = for (self.sigs.items, 0..) |existing, i| {
                if (ioSigEql(existing, sk)) break i;
            } else blk: {
                const idx = self.sigs.items.len;
                try self.sigs.append(self.a, sk);
                break :blk idx;
            };
            var inst: Inst = .{
                .sig_idx = sig_idx,
                .vals = undefined,
                .n_vals = vals.len,
                .fname = try self.a.dupe(u8, fname),
            };
            @memcpy(inst.vals[0..vals.len], vals);
            try self.insts.append(self.a, inst);
        }
    };
    var col: Collect = .{ .a = a };
    try visitSpec(ks, &col, Collect.cb);
    if (col.insts.items.len == 0) return;

    var has_tparams = false;
    for (ks.params) |p| if (p.kind.isTemplate()) {
        has_tparams = true;
        break;
    };

    try out.print(a, "// Kernel: {s}\nnamespace k_{d} {{\n", .{ ks.name, ns_idx });
    if (header.len > 0) {
        try appendHeaderFiltered(out, a, header);
    }

    // Emit one template kernel per unique IO signature.
    for (col.sigs.items, 0..) |sk, sig_idx| {
        if (has_tparams) {
            try out.appendSlice(a, "template <");
            var first = true;
            for (ks.params) |p| {
                if (!p.kind.isTemplate()) continue;
                if (!first) try out.appendSlice(a, ", ");
                first = false;
                try out.appendSlice(a, switch (p.kind) {
                    .ct_int => "int ",
                    .ct_bool => "bool ",
                    .ct_dtype => "typename ",
                    else => unreachable,
                });
                try out.appendSlice(a, p.name);
            }
            try out.appendSlice(a, ">\n");
        } else {
            try out.appendSlice(a, "template <int _SUSHI_D>\n");
        }
        try out.print(a, "[[kernel]] void entry_s{d}(\n", .{sig_idx});

        var buf_idx: usize = 0;
        for (ks.inputs, 0..) |inp, i| {
            const ty_str = switch (inp.ty) {
                .fixed => |d| d.metalType(),
                .param => |pn| if (findParam(ks, pn).?.kind == .ct_dtype) pn else sk.in_dt[i].metalType(),
            };
            try out.print(a, "  const {s} {s}{s} {s} [[buffer({d})]],\n", .{
                if (inp.scalar) "constant" else "device",
                ty_str,
                if (inp.scalar) "&" else "*",
                inp.name,
                buf_idx,
            });
            buf_idx += 1;
            if (inp.scalar) continue;
            if (shape_infos[i][0]) {
                try out.print(a, "  const constant int* {s}_shape [[buffer({d})]],\n", .{ inp.name, buf_idx });
                buf_idx += 1;
            }
            if (shape_infos[i][1]) {
                try out.print(a, "  const constant int64_t* {s}_strides [[buffer({d})]],\n", .{ inp.name, buf_idx });
                buf_idx += 1;
            }
            if (shape_infos[i][2]) {
                try out.print(a, "  const constant int& {s}_ndim [[buffer({d})]],\n", .{ inp.name, buf_idx });
                buf_idx += 1;
            }
        }
        for (ks.outputs, 0..) |o, i| {
            const ty_str = switch (o.ty) {
                .fixed => |d| d.metalType(),
                .param => |pn| if (findParam(ks, pn).?.kind == .ct_dtype) pn else sk.out_dt[i].metalType(),
            };
            const last = i + 1 == ks.outputs.len and n_attrs == 0;
            try out.print(a, "  device {s}* {s} [[buffer({d})]]{s}", .{
                ty_str,
                o.name,
                buf_idx,
                if (last) ") {\n" else ",\n",
            });
            buf_idx += 1;
        }
        var emitted_attr: usize = 0;
        for (metal_attributes, 0..) |attr, i| {
            if (!used_attrs[i]) continue;
            emitted_attr += 1;
            const last = emitted_attr == n_attrs;
            try out.print(a, "  {s} {s} [[{s}]]{s}", .{ attr[1], attr[0], attr[0], if (last) ") {\n" else ",\n" });
        }

        // Bind function constants inside kernel body.
        var next_int: u32 = ks.fc_int_base;
        var next_bool: u32 = ks.fc_bool_base;
        for (ks.params) |p| {
            switch (p.kind) {
                .fc_int => {
                    const slot = p.slot orelse next_int;
                    next_int = slot + 1;
                    try out.print(a, "  const int {s} = FC_I{d}; (void){s};\n", .{ p.name, slot, p.name });
                },
                .fc_bool => {
                    const slot = p.slot orelse next_bool;
                    next_bool = slot + 1;
                    try out.print(a, "  const bool {s} = FC_B{d}; (void){s};\n", .{ p.name, slot - spec.fc_bool_base, p.name });
                },
                else => {},
            }
        }

        try out.appendSlice(a, "{\n");
        try out.appendSlice(a, source);
        try out.appendSlice(a, "\n}\n}\n");
    }

    // Emit explicit template instantiations with `[[host_name(...)]]`.
    for (col.insts.items) |inst| {
        try out.print(a, "template [[host_name(\"{s}\")]] [[kernel]] decltype(entry_s{d}<", .{ inst.fname, inst.sig_idx });
        try appendTemplateArgs(out, a, ks, inst.vals[0..inst.n_vals], has_tparams);
        try out.print(a, ">) entry_s{d}<", .{inst.sig_idx});
        try appendTemplateArgs(out, a, ks, inst.vals[0..inst.n_vals], has_tparams);
        try out.appendSlice(a, ">;\n");
    }

    try out.print(a, "}} // namespace k_{d}\n", .{ns_idx});
    try appendUndefs(out, a, &.{ header, source });
    try out.appendSlice(a, "\n");
}

fn appendTemplateArgs(out: *std.ArrayList(u8), a: std.mem.Allocator, ks: *const KernelSpec, vals: []const CtVal, has_tparams: bool) !void {
    if (!has_tparams) {
        try out.append(a, '0');
        return;
    }
    var vi: usize = 0;
    var first = true;
    for (ks.params) |p| {
        if (!p.kind.isVariant()) continue;
        defer vi += 1;
        if (!p.kind.isTemplate()) continue;
        if (!first) try out.appendSlice(a, ", ");
        first = false;
        switch (vals[vi]) {
            .int => |v| try out.print(a, "{d}", .{v}),
            .boolean => |v| try out.appendSlice(a, if (v) "true" else "false"),
            .dtype => |d| try out.appendSlice(a, d.metalType()),
        }
    }
}

/// Emits all specs in `specs` matching `is_nax` into a complete `.metal` unit.
pub fn emitSpecsUnit(out: *std.ArrayList(u8), a: std.mem.Allocator, specs: []const KernelSpec, is_nax: bool) !void {
    try writeUnitPreamble(out, a, is_nax);
    var ns_idx: usize = 0;
    for (specs) |*ks| {
        if (ks.nax != is_nax) continue;
        try emitSpec(out, a, ks, ns_idx);
        ns_idx += 1;
    }
}
