//! Two independent decode kernels in one launch: the threadgroups of part A run A's statements, the rest run B's, each
//! on its own inputs, outputs and codec helpers, so every output element has the bytes of its kernel run alone. Only
//! the grid changes (A's and B's threadgroups laid end to end), which saves a dispatch and lets B's threadgroups fill
//! the tail of A's.
const std = @import("std");
const mlx = @import("mlx_host").mlx;

pub const IntArg = struct { name: [:0]const u8, value: c_int };
pub const Out = struct { shape: []const c_int, dtype: mlx.mlx_dtype };

/// What a launch needs from its config: threadgroups per grid axis, threads per group, template ints, outputs.
pub const Launch = struct {
    tgs: [3]c_int,
    threads: c_int,
    ints: []const IntArg,
    outs: []const Out,

    fn count(self: Launch) c_int {
        return self.tgs[0] * self.tgs[1] * self.tgs[2];
    }

    pub fn config(self: Launch) !mlx.mlx_fast_metal_kernel_config {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        for (self.outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, o.shape.ptr, o.shape.len, o.dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, self.tgs[0] * self.threads, self.tgs[1], self.tgs[2]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, self.threads, 1, 1));
        for (self.ints) |i| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, i.name.ptr, i.value));
        return c;
    }
};

pub const Part = struct {
    launch: Launch,
    body: []const u8,
    ins: []const [*:0]const u8,
    outs: []const [*:0]const u8,
    header: []const u8 = "",
};

const shared_ins_prefix = "s_";
const shared_int_prefix = "S_";
/// A fused kernel's own template ints: A's grid in threadgroups, B's, and whether B's threadgroups come first.
const grid_ints = [_][:0]const u8{ "FZ_AX", "FZ_AY", "FZ_BX", "FZ_BY", "FZ_NA", "FZ_NB", "FZ_B_FIRST" };

fn isIdentStart(c: u8) bool {
    return std.ascii.isAlphabetic(c) or c == '_';
}

fn isIdent(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

/// `src` with each identifier of `names` prefixed by `prefix`, `grid` swapped for `fz_tgp`, and every `exl3_` /
/// `EXL3_` identifier prefixed by `sx_`: B's codec helpers, which would otherwise redefine A's. A member (`v.x`) is
/// never renamed.
fn renameB(alloc: std.mem.Allocator, src: []const u8, names: []const []const u8, ints: []const IntArg) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var i: usize = 0;
    while (i < src.len) {
        const c = src[i];
        if (std.ascii.isDigit(c)) {
            const start = i;
            while (i < src.len and isIdent(src[i])) i += 1;
            try out.appendSlice(alloc, src[start..i]);
            continue;
        }
        if (!isIdentStart(c)) {
            try out.append(alloc, c);
            i += 1;
            continue;
        }
        const start = i;
        while (i < src.len and isIdent(src[i])) i += 1;
        const word = src[start..i];
        const member = start > 0 and src[start - 1] == '.';
        if (member) {
            try out.appendSlice(alloc, word);
        } else if (std.mem.eql(u8, word, "threadgroup_position_in_grid")) {
            try out.appendSlice(alloc, "fz_tgp");
        } else if (std.mem.startsWith(u8, word, "exl3_") or std.mem.startsWith(u8, word, "EXL3_")) {
            try out.appendSlice(alloc, "sx_");
            try out.appendSlice(alloc, word);
        } else if (for (names) |n| {
            if (std.mem.eql(u8, n, word)) break true;
        } else false) {
            try out.appendSlice(alloc, shared_ins_prefix);
            try out.appendSlice(alloc, word);
        } else if (for (ints) |n| {
            if (std.mem.eql(u8, n.name, word)) break true;
        } else false) {
            try out.appendSlice(alloc, shared_int_prefix);
            try out.appendSlice(alloc, word);
        } else try out.appendSlice(alloc, word);
    }
    return out.toOwnedSlice(alloc);
}

fn renameA(alloc: std.mem.Allocator, src: []const u8) ![]u8 {
    return std.mem.replaceOwned(u8, alloc, src, "threadgroup_position_in_grid", "fz_tgp");
}

fn branch(out: *std.ArrayList(u8), alloc: std.mem.Allocator, ax: []const u8, ay: []const u8, body: []const u8, is_a: bool) !void {
    try out.print(alloc, "{{\nconst uint fz_l = {s};\nconst uint3 fz_tgp = uint3(fz_l % uint({s}), (fz_l / uint({s})) % uint({s}), fz_l / uint({s} * {s}));\n", .{
        if (is_a) "(FZ_B_FIRST != 0 ? fz_lin - uint(FZ_NB) : fz_lin)" else "(FZ_B_FIRST != 0 ? fz_lin : fz_lin - uint(FZ_NA))",
        ax, ax, ay, ax, ay,
    });
    try out.appendSlice(alloc, body);
    try out.appendSlice(alloc, "\n}\n");
}

/// The fused kernel's body: `a` where the threadgroup is A's, `b` (renamed) elsewhere.
pub fn source(alloc: std.mem.Allocator, a: Part, b: Part) ![:0]u8 {
    const a_body = try renameA(alloc, a.body);
    defer alloc.free(a_body);
    const names = try alloc.alloc([]const u8, b.ins.len + b.outs.len);
    defer alloc.free(names);
    for (b.ins, names[0..b.ins.len]) |n, *o| o.* = std.mem.span(n);
    for (b.outs, names[b.ins.len..]) |n, *o| o.* = std.mem.span(n);
    const b_body = try renameB(alloc, b.body, names, b.launch.ints);
    defer alloc.free(b_body);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    try out.appendSlice(alloc, "const uint fz_lin = uint(threadgroup_position_in_grid.x);\n");
    try out.appendSlice(alloc, "if (FZ_B_FIRST != 0 ? fz_lin >= uint(FZ_NB) : fz_lin < uint(FZ_NA)) ");
    try branch(&out, alloc, "FZ_AX", "FZ_AY", a_body, true);
    try out.appendSlice(alloc, "else ");
    try branch(&out, alloc, "FZ_BX", "FZ_BY", b_body, false);
    return out.toOwnedSliceSentinel(alloc, 0);
}

/// A's header, then B's with its helpers renamed.
pub fn header(alloc: std.mem.Allocator, a: Part, b: Part) ![:0]u8 {
    const b_head = try renameB(alloc, b.header, &.{}, &.{});
    defer alloc.free(b_head);
    return std.mem.concatWithSentinel(alloc, u8, &.{ a.header, "\n", b_head }, 0);
}

pub fn insOf(alloc: std.mem.Allocator, a: Part, b: Part) ![]const [*:0]const u8 {
    const out = try alloc.alloc([*:0]const u8, a.ins.len + b.ins.len);
    @memcpy(out[0..a.ins.len], a.ins);
    for (b.ins, out[a.ins.len..]) |n, *o| o.* = (try std.fmt.allocPrintSentinel(alloc, "{s}{s}", .{ shared_ins_prefix, std.mem.span(n) }, 0)).ptr;
    return out;
}

pub fn outsOf(alloc: std.mem.Allocator, a: Part, b: Part) ![]const [*:0]const u8 {
    const out = try alloc.alloc([*:0]const u8, a.outs.len + b.outs.len);
    @memcpy(out[0..a.outs.len], a.outs);
    for (b.outs, out[a.outs.len..]) |n, *o| o.* = (try std.fmt.allocPrintSentinel(alloc, "{s}{s}", .{ shared_ins_prefix, std.mem.span(n) }, 0)).ptr;
    return out;
}

/// The fused launch: A's outputs then B's, one grid of both parts' threadgroups, every template int of both.
pub fn config(alloc: std.mem.Allocator, a: Launch, b: Launch, b_first: bool) !mlx.mlx_fast_metal_kernel_config {
    if (a.threads != b.threads) return error.FuseThreadsDiffer;
    const na = a.count();
    const nb = b.count();
    var ints: std.ArrayList(IntArg) = .empty;
    defer ints.deinit(alloc);
    try ints.appendSlice(alloc, a.ints);
    for (b.ints) |i| try ints.append(alloc, .{ .name = try std.fmt.allocPrintSentinel(alloc, "{s}{s}", .{ shared_int_prefix, i.name }, 0), .value = i.value });
    const values = [_]c_int{ a.tgs[0], a.tgs[1], b.tgs[0], b.tgs[1], na, nb, @intFromBool(b_first) };
    for (grid_ints, values) |name, v| try ints.append(alloc, .{ .name = name, .value = v });
    const outs = try std.mem.concat(alloc, Out, &.{ a.outs, b.outs });
    defer alloc.free(outs);
    return (Launch{ .tgs = .{ na + nb, 1, 1 }, .threads = a.threads, .ints = ints.items, .outs = outs }).config();
}
