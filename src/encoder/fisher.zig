//! Fisher-weighted output scales: each routed expert's output channels weighted by the model's sensitivity to them
//! (a diagonal output Fisher, the mean squared gradient of the sampled-token log-likelihood at that channel). With a
//! white inner quantizer channel j's error goes as its scale t_j and the inner RMS as mean(c / t^2), c_j the
//! channel's mean square weight, so the Fisher-weighted error is least at t_j ~ (c_j / F_j)^(1/4).
//!
//! The Fisher comes from a work directory another tool computed: `fisher-LL.npz` per layer with float32
//! `fisher_gate`, `fisher_up` [experts, intermediate] and `fisher_down` [experts, hidden], indexed by source expert.
const std = @import("std");
const calib = @import("calib.zig");
const source = @import("source.zig");

/// The share of the mean Fisher added to every channel, so a channel the calibration never moved keeps a scale.
pub const damp: f64 = 0.01;

pub const Error = error{ FisherFileMissing, FisherFileInvalid, FisherShapeMismatch } || std.mem.Allocator.Error;

pub const LayerFisher = struct {
    allocator: std.mem.Allocator,
    experts: usize,
    hidden: usize,
    inter: usize,
    gate: []f32,
    up: []f32,
    down: []f32,

    pub fn deinit(self: *LayerFisher) void {
        self.allocator.free(self.gate);
        self.allocator.free(self.up);
        self.allocator.free(self.down);
    }

    /// Expert `e`'s output-channel Fisher for `role`.
    pub fn row(self: *const LayerFisher, role: source.ProjRole, e: usize) []const f32 {
        return switch (role) {
            .gate => self.gate[e * self.inter ..][0..self.inter],
            .up => self.up[e * self.inter ..][0..self.inter],
            .down => self.down[e * self.hidden ..][0..self.hidden],
        };
    }
};

/// A little-endian float32 C-order `.npy` array of `want` shape, copied out.
pub fn npyF32(allocator: std.mem.Allocator, bytes: []const u8, want: []const usize) Error![]f32 {
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.FisherFileInvalid;
    const major = bytes[6];
    const header_len: usize, const at: usize = switch (major) {
        1 => .{ std.mem.readInt(u16, bytes[8..10], .little), 10 },
        2, 3 => blk: {
            if (bytes.len < 12) return error.FisherFileInvalid;
            break :blk .{ std.mem.readInt(u32, bytes[8..12], .little), 12 };
        },
        else => return error.FisherFileInvalid,
    };
    if (at + header_len > bytes.len) return error.FisherFileInvalid;
    const header = bytes[at..][0..header_len];
    if (std.mem.indexOf(u8, header, "'descr': '<f4'") == null) return error.FisherFileInvalid;
    if (std.mem.indexOf(u8, header, "'fortran_order': False") == null) return error.FisherFileInvalid;
    const open = std.mem.indexOf(u8, header, "'shape': (") orelse return error.FisherFileInvalid;
    const rest = header[open + "'shape': (".len ..];
    const close = std.mem.indexOfScalar(u8, rest, ')') orelse return error.FisherFileInvalid;
    var it = std.mem.tokenizeAny(u8, rest[0..close], ", ");
    var count: usize = 1;
    var dims: usize = 0;
    while (it.next()) |d| : (dims += 1) {
        const v = std.fmt.parseInt(usize, d, 10) catch return error.FisherFileInvalid;
        if (dims >= want.len or v != want[dims]) return error.FisherShapeMismatch;
        count *= v;
    }
    if (dims != want.len) return error.FisherShapeMismatch;
    const data = bytes[at + header_len ..];
    if (data.len != count * 4) return error.FisherFileInvalid;
    const out = try allocator.alloc(f32, count);
    for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, data[i * 4 ..][0..4], .little));
    return out;
}

fn member(allocator: std.mem.Allocator, io: std.Io, path: []const u8, name: []const u8, want: []const usize) ![]f32 {
    const bytes = calib.zipEntryPrefix(allocator, io, path, name, 1 << 34) catch return error.FisherFileInvalid;
    defer allocator.free(bytes);
    return npyF32(allocator, bytes, want);
}

pub fn layerPath(buf: []u8, dir: []const u8, layer: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/fisher-{d:0>2}.npz", .{ dir, layer });
}

pub fn loadLayer(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, layer: u32, experts: usize, hidden: usize, inter: usize) !LayerFisher {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try layerPath(&path_buf, dir, layer);
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.FisherFileMissing;
    const gate = try member(allocator, io, path, "fisher_gate.npy", &.{ experts, inter });
    errdefer allocator.free(gate);
    const up = try member(allocator, io, path, "fisher_up.npy", &.{ experts, inter });
    errdefer allocator.free(up);
    const down = try member(allocator, io, path, "fisher_down.npy", &.{ experts, hidden });
    return .{ .allocator = allocator, .experts = experts, .hidden = hidden, .inter = inter, .gate = gate, .up = up, .down = down };
}

/// The output scales of weight `w` [in, out] (row-major) under output Fisher `f`, geometric mean 1. A tensor no
/// calibration row reached (`f` all zero) keeps the RMS scales t_j ~ c_j^(1/2).
pub fn scales(w: []const f32, in: usize, out: usize, f: []const f32, t: []f32) void {
    std.debug.assert(w.len == in * out and f.len == out and t.len == out);
    var f_sum: f64 = 0;
    for (f) |v| f_sum += v;
    const f_floor = damp * f_sum / @as(f64, @floatFromInt(out));
    var log_sum: f64 = 0;
    for (0..out) |j| {
        var c: f64 = 0;
        for (0..in) |i| {
            const v: f64 = w[i * out + j];
            c += v * v;
        }
        c = @max(c / @as(f64, @floatFromInt(in)), 1e-30);
        const v = if (f_sum > 0) std.math.pow(f64, c, 0.25) * std.math.pow(f64, @as(f64, f[j]) + f_floor, -0.25) else @sqrt(c);
        t[j] = @floatCast(v);
        log_sum += @log(v);
    }
    const g = @exp(log_sum / @as(f64, @floatFromInt(out)));
    for (t) |*v| v.* = @floatCast(@as(f64, v.*) / g);
}

test "fisher: output scales match the reference formula, and an unreached tensor keeps its RMS scales" {
    const t = std.testing;
    const w = [_]f32{ 0.1, -0.2, 0.3, 0.05, 0.4, -0.1, -0.3, 0.1, 0.2, 0.2, 0.0, -0.05 };
    var got: [3]f32 = undefined;
    scales(&w, 4, 3, &.{ 2.0, 0.5, 1.0 }, &got);
    for ([_]f32{ 0.8157222867012024, 1.265568494796753, 0.968661367893219 }, got) |want, have| try t.expectApproxEqRel(want, have, 2e-6);
    scales(&w, 4, 3, &.{ 0, 0, 0 }, &got);
    for ([_]f32{ 0.9374164938926697, 1.1379804611206055, 0.9374164938926697 }, got) |want, have| try t.expectApproxEqRel(want, have, 2e-6);
}

test "fisher: a float32 .npy of the expected shape reads, any other shape or type is refused" {
    const t = std.testing;
    var bytes: [128 + 24]u8 = undefined;
    const header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
    @memcpy(bytes[0..6], "\x93NUMPY");
    bytes[6] = 1;
    bytes[7] = 0;
    std.mem.writeInt(u16, bytes[8..10], 118, .little);
    @memset(bytes[10..128], ' ');
    @memcpy(bytes[10..][0..header.len], header);
    bytes[127] = '\n';
    for (0..6) |i| std.mem.writeInt(u32, bytes[128 + i * 4 ..][0..4], @bitCast(@as(f32, @floatFromInt(i))), .little);
    const got = try npyF32(t.allocator, &bytes, &.{ 2, 3 });
    defer t.allocator.free(got);
    try t.expectEqualSlices(f32, &.{ 0, 1, 2, 3, 4, 5 }, got);
    try t.expectError(error.FisherShapeMismatch, npyF32(t.allocator, &bytes, &.{ 3, 2 }));
    try t.expectError(error.FisherShapeMismatch, npyF32(t.allocator, &bytes, &.{6}));
    try t.expectError(error.FisherFileInvalid, npyF32(t.allocator, bytes[0..140], &.{ 2, 3 }));
}

test "fisher: a layer's file that is not there is missing, and one that is not a zip is invalid" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = buf[0..try work.dir.realPath(io, &buf)];
    try t.expectError(error.FisherFileMissing, loadLayer(t.allocator, io, dir, 3, 2, 4, 4));
    try work.dir.writeFile(io, .{ .sub_path = "fisher-03.npz", .data = "not a zip" });
    try t.expectError(error.FisherFileInvalid, loadLayer(t.allocator, io, dir, 3, 2, 4, 4));
}
