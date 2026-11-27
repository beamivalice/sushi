//! Build-time Metal source generator for Sushi Cooler's precompiled kernels (`sushi.metallib`).
//! Invoked by `build.zig:addSushiMetallib`: writes one `.metal` file per requested `(unit_name, out_path)`
//! pair, derived directly from `src/tensor/metallib_spec.zig` (ops/chains) and `src/metallib_catalog.zig`
//! (shared, EXL3, and enabled arch custom kernels).
//!
//! Usage: gen-metallib <unit_name> <out_path> [<unit_name> <out_path> ...]

const std = @import("std");
const spec = @import("tensor/metallib_spec.zig");
const catalog = @import("metallib_catalog.zig");

const Dtype = spec.Dtype;
const OpsUnit = spec.OpsUnit;
const ElemOp = spec.ElemOp;
const ElemVariant = spec.ElemVariant;
const SpecialOp = spec.SpecialOp;

const OpsBuffers = struct {
    a: std.mem.Allocator,
    ops: [std.enums.values(OpsUnit).len]std.ArrayList(u8),
    built: bool = false,

    fn init(a: std.mem.Allocator) !OpsBuffers {
        var self: OpsBuffers = .{ .a = a, .ops = undefined };
        for (&self.ops) |*buf| {
            buf.* = .empty;
            try buf.appendSlice(a, spec.ops_metal_helpers);
        }
        return self;
    }

    fn deinit(self: *OpsBuffers) void {
        for (&self.ops) |*buf| buf.deinit(self.a);
    }

    fn opsBuf(self: *OpsBuffers, unit: OpsUnit) *std.ArrayList(u8) {
        return &self.ops[@intFromEnum(unit)];
    }

    fn ensureBuilt(self: *OpsBuffers) !void {
        if (self.built) return;
        try spec.visitOps(self, emitElem, emitSpecial);
        self.built = true;
    }
};

fn appendSubst(out: *std.ArrayList(u8), a: std.mem.Allocator, text: []const u8, in_types: []const Dtype, out_type: Dtype) !void {
    var at: usize = 0;
    while (at < text.len) {
        if (text[at] == '$' and at + 1 < text.len) {
            const c = text[at + 1];
            const name: ?[]const u8 = if (c == 'O')
                out_type.metalType()
            else if (c >= '0' and c <= '9' and c - '0' < in_types.len)
                in_types[c - '0'].metalType()
            else
                null;
            if (name) |n| {
                try out.appendSlice(a, n);
                at += 2;
                continue;
            }
        }
        try out.append(a, text[at]);
        at += 1;
    }
}

fn emitElem(ub: *OpsBuffers, unit: OpsUnit, fn_name: []const u8, op: ElemOp, variant: ElemVariant, in_types: []const Dtype, out_type: Dtype) !void {
    const a = ub.a;
    const out = ub.opsBuf(unit);
    const names = [_][]const u8{ "x", "y", "z" };

    const expr = if (std.mem.eql(u8, op.tag, "bin_floor_divide") and out_type.isUnsignedInt())
        "$O(x / y)"
    else
        op.expr;

    switch (variant) {
        .flat => {
            try out.print(a, "[[kernel]] void {s}(\n", .{fn_name});
            for (0..op.n_in) |i| {
                try out.print(a, "  const device {s}* in{d} [[buffer({d})]],\n", .{ in_types[i].metalType(), i, i });
            }
            try out.print(a,
                \\  device {s}* out [[buffer({d})]],
                \\  const constant uint& n [[buffer({d})]],
                \\  uint at [[thread_position_in_grid]]) {{
                \\  for (uint e = at * {d}u; e < min(n, at * {d}u + {d}u); ++e) {{
                \\
            , .{ out_type.metalType(), op.n_in, op.n_in + 1, spec.flat_per_thread, spec.flat_per_thread, spec.flat_per_thread });
            for (0..op.n_in) |i| {
                try out.print(a, "    const {s} {s} = in{d}[FC_B{d} ? 0u : e];\n", .{ in_types[i].metalType(), names[i], i, i });
            }
            try out.appendSlice(a, "    out[e] = ");
            try appendSubst(out, a, expr, in_types, out_type);
            try out.appendSlice(a, ";\n  }\n}\n");
        },
        .g3, .i32, .wide => {
            const wide = variant == .wide;
            const three = variant == .g3;
            const idx = if (wide) "long" else "int";
            const loc = if (wide) "sushi_loc" else "sushi_loc32";
            const base = 2 * @as(usize, op.n_in);
            try out.print(a, "[[kernel]] void {s}(\n", .{fn_name});
            for (0..op.n_in) |i| {
                try out.print(a,
                    \\  const device {s}* in{d} [[buffer({d})]],
                    \\  const constant {s}* s{d} [[buffer({d})]],
                    \\
                , .{ in_types[i].metalType(), i, 2 * i, idx, i, 2 * i + 1 });
            }
            try out.print(a,
                \\  device {s}* out [[buffer({d})]],
                \\  const constant int* shape [[buffer({d})]],
                \\  const constant int& ndim [[buffer({d})]],
                \\  uint3 at [[thread_position_in_grid]],
                \\  uint3 grid [[threads_per_grid]]) {{
                \\
            , .{ out_type.metalType(), base, base + 1, base + 2 });
            if (three) {
                try out.appendSlice(a, "  (void)shape; (void)ndim;\n");
            }
            for (0..op.n_in) |i| {
                if (three) {
                    try out.print(a, "  const {s} {s} = in{d}[int(at.x) * s{d}[2] + int(at.y) * s{d}[1] + int(at.z) * s{d}[0]];\n", .{ in_types[i].metalType(), names[i], i, i, i, i });
                } else {
                    try out.print(a, "  const {s} {s} = in{d}[{s}(at, shape, s{d}, ndim)];\n", .{ in_types[i].metalType(), names[i], i, loc, i });
                }
            }
            const flat = if (wide) "sushi_flat(at, grid)" else "at.x + grid.x * (at.y + grid.y * at.z)";
            try out.print(a, "  out[{s}] = ", .{flat});
            try appendSubst(out, a, expr, in_types, out_type);
            try out.appendSlice(a, ";\n}\n");
        },
    }
}

fn emitSpecial(ub: *OpsBuffers, unit: OpsUnit, fn_name: []const u8, op: SpecialOp, in_types: []const Dtype, out_type: Dtype) !void {
    const a = ub.a;
    const out = ub.opsBuf(unit);
    switch (op) {
        .zeros => {
            try out.print(a,
                \\[[kernel]] void {s}(
                \\  device {s}* out [[buffer(0)]],
                \\  const constant uint& n [[buffer(1)]],
                \\  uint at [[thread_position_in_grid]]) {{
                \\  for (uint e = at * {d}u; e < min(n, at * {d}u + {d}u); ++e) {{
                \\    out[e] = 0;
                \\  }}
                \\}}
                \\
            , .{ fn_name, out_type.metalType(), spec.flat_per_thread, spec.flat_per_thread, spec.flat_per_thread });
        },
        .copy_into => {
            try out.print(a,
                \\[[kernel]] void {s}(
                \\  const device {s}* in [[buffer(0)]],
                \\  const constant long* s0 [[buffer(1)]],
                \\  device {s}* out [[buffer(2)]],
                \\  const constant long* so [[buffer(3)]],
                \\  const constant int* shape [[buffer(4)]],
                \\  const constant int& ndim [[buffer(5)]],
                \\  uint3 at [[thread_position_in_grid]]) {{
                \\  out[sushi_loc(at, shape, so, ndim)] = {s}(in[sushi_loc(at, shape, s0, ndim)]);
                \\}}
                \\
            , .{ fn_name, in_types[0].metalType(), out_type.metalType(), out_type.metalType() });
        },
        .red_sum, .red_max => {
            const step_line = if (op == .red_sum)
                "    acc += in[loc];\n"
            else
                "    acc = max(acc, in[loc]);\n";
            try out.print(a,
                \\[[kernel]] void {s}(
                \\  const device {s}* in [[buffer(0)]],
                \\  const constant long* s0 [[buffer(1)]],
                \\  device {s}* out [[buffer(2)]],
                \\  const constant int* shape [[buffer(3)]],
                \\  const constant int& ndim [[buffer(4)]],
                \\  const constant long& step [[buffer(5)]],
                \\  const constant int& count [[buffer(6)]],
                \\  uint3 at [[thread_position_in_grid]],
                \\  uint3 grid [[threads_per_grid]]) {{
                \\  long loc = sushi_loc(at, shape, s0, ndim);
                \\  {s} acc = in[loc];
                \\  for (int k = 1; k < count; ++k) {{
                \\    loc += step;
                \\{s}  }}
                \\  out[sushi_flat(at, grid)] = acc;
                \\}}
                \\
            , .{ fn_name, in_types[0].metalType(), out_type.metalType(), out_type.metalType(), step_line });
        },
        .concat => |k| {
            const names = [_][]const u8{ "a", "b", "c", "d", "e", "f", "g", "h", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15" };
            try out.print(a, "[[kernel]] void {s}(\n", .{fn_name});
            for (0..k) |i| {
                try out.print(a, "  const device {s}* {s} [[buffer({d})]],\n", .{ in_types[0].metalType(), names[i], i });
            }
            try out.print(a,
                \\  device {s}* out [[buffer({d})]],
                \\  const constant uint& inner [[buffer({d})]],
                \\  const constant uint* ext [[buffer({d})]],
                \\  const constant uint& n [[buffer({d})]],
                \\  uint i [[thread_position_in_grid]]) {{
                \\  if (i >= n) return;
                \\  uint total = 0;
                \\  for (uint j = 0; j < {d}u; ++j) total += ext[j];
                \\  uint within = i % inner;
                \\  uint row = i / inner;
                \\  uint pos = row % total;
                \\  uint outer = row / total;
                \\  uint start = 0u;
                \\
            , .{ out_type.metalType(), k, k + 1, k + 2, k + 3, k });
            for (0..k) |i| {
                if (i + 1 < k) {
                    try out.print(a, "  if (pos < start + ext[{d}]) {{ out[i] = {s}[(outer * ext[{d}] + pos - start) * inner + within]; return; }}\n  start += ext[{d}];\n", .{ i, names[i], i, i });
                } else {
                    try out.print(a, "  out[i] = {s}[(outer * ext[{d}] + pos - start) * inner + within];\n", .{ names[i], i });
                }
            }
            try out.appendSlice(a, "}\n");
        },
        .take, .take_along => {
            try out.print(a,
                \\[[kernel]] void {s}(
                \\  const device {s}* table [[buffer(0)]],
                \\  const constant long* st [[buffer(1)]],
                \\  const device {s}* index [[buffer(2)]],
                \\  const constant long* si [[buffer(3)]],
                \\  device {s}* out [[buffer(4)]],
                \\  const constant int* shape [[buffer(5)]],
                \\  const constant int& ndim [[buffer(6)]],
                \\  const constant long& axis_stride [[buffer(7)]],
                \\  const constant int& axis_len [[buffer(8)]],
                \\  uint3 at [[thread_position_in_grid]],
                \\  uint3 grid [[threads_per_grid]]) {{
                \\  long pick = long(index[sushi_loc(at, shape, si, ndim)]);
                \\  if (pick < 0) pick += axis_len;
                \\  out[sushi_flat(at, grid)] = table[sushi_loc(at, shape, st, ndim) + pick * axis_stride];
                \\}}
                \\
            , .{ fn_name, in_types[0].metalType(), in_types[1].metalType(), out_type.metalType() });
        },
        .put_along => {
            try out.print(a,
                \\[[kernel]] void {s}(
                \\  const device {s}* upd [[buffer(0)]],
                \\  const constant long* su [[buffer(1)]],
                \\  const device {s}* index [[buffer(2)]],
                \\  const constant long* si [[buffer(3)]],
                \\  device {s}* out [[buffer(4)]],
                \\  const constant long* so [[buffer(5)]],
                \\  const constant int* shape [[buffer(6)]],
                \\  const constant int& ndim [[buffer(7)]],
                \\  const constant long& axis_stride [[buffer(8)]],
                \\  const constant int& axis_len [[buffer(9)]],
                \\  uint3 at [[thread_position_in_grid]]) {{
                \\  long pick = long(index[sushi_loc(at, shape, si, ndim)]);
                \\  if (pick < 0) pick += axis_len;
                \\  out[sushi_loc(at, shape, so, ndim) + pick * axis_stride] = upd[sushi_loc(at, shape, su, ndim)];
                \\}}
                \\
            , .{ fn_name, in_types[0].metalType(), in_types[1].metalType(), out_type.metalType() });
        },
        .pad => {
            try out.print(a,
                \\[[kernel]] void {s}(
                \\  const device {s}* in [[buffer(0)]],
                \\  const constant long* ist [[buffer(1)]],
                \\  const constant int* lo [[buffer(2)]],
                \\  const constant int* extent [[buffer(3)]],
                \\  const device {s}* value [[buffer(4)]],
                \\  device {s}* out [[buffer(5)]],
                \\  const constant int* shape [[buffer(6)]],
                \\  const constant int& ndim [[buffer(7)]],
                \\  uint3 at [[thread_position_in_grid]],
                \\  uint3 grid [[threads_per_grid]]) {{
                \\  long loc = 0;
                \\  bool inside = true;
                \\  int c = int(at.x) - lo[ndim - 1];
                \\  inside = inside && c >= 0 && c < extent[ndim - 1];
                \\  loc += long(c) * ist[ndim - 1];
                \\  c = int(at.y) - lo[ndim - 2];
                \\  inside = inside && c >= 0 && c < extent[ndim - 2];
                \\  loc += long(c) * ist[ndim - 2];
                \\  uint z = at.z;
                \\  for (int d = ndim - 3; d >= 0; --d) {{
                \\    c = int(z % uint(shape[d])) - lo[d];
                \\    z /= uint(shape[d]);
                \\    inside = inside && c >= 0 && c < extent[d];
                \\    loc += long(c) * ist[d];
                \\  }}
                \\  out[sushi_flat(at, grid)] = inside ? in[loc] : value[0];
                \\}}
                \\
            , .{ fn_name, in_types[0].metalType(), in_types[0].metalType(), out_type.metalType() });
        },
    }
}

const WorkItem = struct {
    unit_name: []const u8,
    out_path: []const u8,
};

fn emitOneUnit(io: std.Io, gpa: std.mem.Allocator, item: WorkItem) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();

    if (std.mem.eql(u8, item.unit_name, "ops_all")) {
        var ops_bufs = try OpsBuffers.init(a);
        defer ops_bufs.deinit();
        try ops_bufs.ensureBuilt();
        var combined: std.ArrayList(u8) = .empty;
        defer combined.deinit(a);
        try combined.appendSlice(a, spec.ops_metal_helpers);
        for (ops_bufs.ops) |buf| {
            try combined.appendSlice(a, buf.items[spec.ops_metal_helpers.len..]);
        }
        try cwd.writeFile(io, .{ .sub_path = item.out_path, .data = combined.items });
        return;
    }
    if (std.meta.stringToEnum(OpsUnit, item.unit_name)) |ops_unit| {
        var ops_bufs = try OpsBuffers.init(a);
        defer ops_bufs.deinit();
        try ops_bufs.ensureBuilt();
        try cwd.writeFile(io, .{ .sub_path = item.out_path, .data = ops_bufs.opsBuf(ops_unit).items });
        return;
    }
    var custom_buf: std.ArrayList(u8) = .empty;
    defer custom_buf.deinit(a);
    if (try catalog.emitCustomUnit(&custom_buf, a, item.unit_name)) {
        try cwd.writeFile(io, .{ .sub_path = item.out_path, .data = custom_buf.items });
        return;
    }
    return error.UnknownUnit;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, a);
    _ = args.next();

    var items: std.ArrayList(WorkItem) = .empty;
    while (args.next()) |unit_name| {
        const out_path = args.next() orelse return error.MissingOutPath;
        try items.append(a, .{ .unit_name = unit_name, .out_path = out_path });
    }
    if (items.items.len <= 1) {
        for (items.items) |it| try emitOneUnit(io, init.gpa, it);
        return;
    }

    const WorkerCtx = struct {
        io: std.Io,
        gpa: std.mem.Allocator,
        items: []const WorkItem,
        next: std.atomic.Value(usize) = .init(0),
        failed: std.atomic.Value(bool) = .init(false),

        fn run(ctx: *@This()) void {
            while (true) {
                const idx = ctx.next.fetchAdd(1, .monotonic);
                if (idx >= ctx.items.len) return;
                emitOneUnit(ctx.io, ctx.gpa, ctx.items[idx]) catch |e| {
                    std.debug.print("gen-metallib: unit {s}: {s}\n", .{ ctx.items[idx].unit_name, @errorName(e) });
                    ctx.failed.store(true, .release);
                    return;
                };
            }
        }
    };
    var ctx: WorkerCtx = .{ .io = io, .gpa = init.gpa, .items = items.items };
    const n_threads = @min(items.items.len, @min(std.Thread.getCpuCount() catch 4, 16));
    var threads: [16]std.Thread = undefined;
    var spawned: usize = 0;
    for (0..n_threads - 1) |i| {
        threads[i] = try std.Thread.spawn(.{}, WorkerCtx.run, .{&ctx});
        spawned += 1;
    }
    ctx.run();
    for (threads[0..spawned]) |t| t.join();
    if (ctx.failed.load(.acquire)) return error.EmitUnitFailed;
}
