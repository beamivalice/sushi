const kernel_src = @import("kernel_source.zig");
const log = @import("mlx_host").log;
const mlx = @import("mlx_host").mlx;
const std = @import("std");
const kernels = @import("expert_exl3_kernels.zig");

pub const GpuWindowTable = struct { table: kernels.WindowTable, inverse: mlx.mlx_array };

const GPU_WINDOW_SOURCE: [:0]const u8 =
    \\const uint tid = uint(thread_index_in_threadgroup);
    \\const uint n = uint(count[0]);
    \\const uint capacity = (n + uint(WIN) - 1u) / uint(WIN) + uint(EXPERTS);
    \\threadgroup uint counts[uint(EXPERTS)];
    \\for (uint i = tid; i < capacity; i += uint(EXPERTS)) { starts[i] = 0u; nlives[i] = 0u; }
    \\for (uint i = tid; i < n; i += uint(EXPERTS)) inverse[uint(order[i])] = i;
    \\uint lo = 0u, hi = n;
    \\while (lo < hi) { uint m = (lo + hi) >> 1u; if (uint(eids[m]) < tid) lo = m + 1u; else hi = m; }
    \\const uint first = lo;
    \\hi = n;
    \\while (lo < hi) { uint m = (lo + hi) >> 1u; if (uint(eids[m]) <= tid) lo = m + 1u; else hi = m; }
    \\const uint length = lo - first;
    \\counts[tid] = (length + uint(WIN) - 1u) / uint(WIN);
    \\threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    \\uint offset = 0u;
    \\for (uint e = 0u; e < tid; e++) offset += counts[e];
    \\for (uint i = 0u; i < counts[tid]; i++) {
    \\  starts[offset + i] = first + i * uint(WIN);
    \\  nlives[offset + i] = min(uint(WIN), length - i * uint(WIN));
    \\}
;

const GpuWindowKey = struct { rows: c_int, win: c_int, experts: c_int };

var gpu_window_cfgs: kernel_src.CfgCache(GpuWindowKey, 8) = .{};

var gpu_window_kernel: ?mlx.mlx_fast_metal_kernel = null;

var gpu_window_engaged: bool = false;

/// One thread per expert, in one threadgroup.
pub const GPU_WINDOW_MAX_EXPERTS: c_int = 512;

pub fn buildGpuWindowTable(s: mlx.mlx_stream, eids: mlx.mlx_array, order: mlx.mlx_array, n: c_int, win: c_int, experts: c_int) !GpuWindowTable {
    if (experts < 1 or experts > GPU_WINDOW_MAX_EXPERTS or win < 1 or n < 1) return error.BadExl3Shape;
    const capacity = @divTrunc(n + win - 1, win) + experts;
    const key = GpuWindowKey{ .rows = n, .win = win, .experts = experts };
    const cfg = gpu_window_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{capacity}, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{capacity}, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{n}, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, experts, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, experts, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "WIN", win));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "EXPERTS", experts));
        gpu_window_cfgs.put(key, c);
        break :blk c;
    };
    const nn: u32 = @intCast(n);
    const count = mlx.mlx_array_new_data(&nn, &.{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(count);
    const kernel = try kernels.getNamedKernel(&gpu_window_kernel, "sushi_exl3_gpu_windows", &.{ "eids", "order", "count" }, &.{ "starts", "nlives", "inverse" }, GPU_WINDOW_SOURCE, "");
    const outputs = try kernels.applyOuts(s, kernel, &.{ eids, order, count }, cfg, 3);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var starts = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(starts);
    var nlives = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(nlives);
    var inverse = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inverse);
    try mlx.check(mlx.mlx_vector_array_get(&starts, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&nlives, outputs, 1));
    try mlx.check(mlx.mlx_vector_array_get(&inverse, outputs, 2));
    if (!gpu_window_engaged) {
        gpu_window_engaged = true;
        if (!@import("builtin").is_test) log.info("[exl3-prefill] GPU window metadata engaged experts={d} win={d}\n", .{ experts, win });
    }
    return .{ .table = .{ .starts = starts, .nlives = nlives, .nwin = capacity }, .inverse = inverse };
}

fn gpuWindowCapacityCase(s: mlx.mlx_stream, experts: c_int) !void {
    const ne: usize = @intCast(experts);
    for ([_]usize{ 1, 31, 32, 33, 513, 2049, 16 * 1024 }) |n| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const alloc = arena.allocator();
        const ids = try alloc.alloc(u32, n);
        const ord = try alloc.alloc(u32, n);
        for (ids, ord, 0..) |*id, *o, i| {
            id.* = @intCast(@min((i / 97) * 3, ne - 1));
            o.* = @intCast((i + 7) % n);
        }
        const ea = mlx.mlx_array_new_data(ids.ptr, &.{@intCast(n)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(ea);
        const oa = mlx.mlx_array_new_data(ord.ptr, &.{@intCast(n)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(oa);
        const m = try buildGpuWindowTable(s, ea, oa, @intCast(n), 32, experts);
        defer _ = mlx.mlx_array_free(m.table.starts);
        defer _ = mlx.mlx_array_free(m.table.nlives);
        defer _ = mlx.mlx_array_free(m.inverse);
        try std.testing.expectEqual(@as(c_int, @intCast((n + 31) / 32 + ne)), m.table.nwin);
        try mlx.check(mlx.mlx_array_eval(m.table.starts));
        try mlx.check(mlx.mlx_array_eval(m.table.nlives));
        try mlx.check(mlx.mlx_array_eval(m.inverse));
        const starts = mlx.mlx_array_data_uint32(m.table.starts) orelse return error.Unreadable;
        const lives = mlx.mlx_array_data_uint32(m.table.nlives) orelse return error.Unreadable;
        const inv = mlx.mlx_array_data_uint32(m.inverse) orelse return error.Unreadable;
        var row: usize = 0;
        var wi: usize = 0;
        while (row < n) : (wi += 1) {
            var end = row + 1;
            while (end < n and ids[end] == ids[row] and end - row < 32) : (end += 1) {}
            try std.testing.expectEqual(row, starts[wi]);
            try std.testing.expectEqual(end - row, lives[wi]);
            row = end;
        }
        while (wi < @as(usize, @intCast(m.table.nwin))) : (wi += 1) try std.testing.expectEqual(@as(u32, 0), lives[wi]);
        for (0..n) |i| try std.testing.expectEqual((i + n - 7 % n) % n, inv[i]);
    }
}

const SORTED_INVERSE_SOURCE: [:0]const u8 = blk: {
    const old = "const size_t xb = (size_t)slot * (size_t)(ODIM) + base;";
    const at = std.mem.indexOf(u8, kernels.REDUCE_SOURCE, old).?;
    break :blk kernels.REDUCE_SOURCE[0..at] ++ "const size_t xb = (size_t)inverse[slot] * (size_t)(ODIM) + base;" ++ kernels.REDUCE_SOURCE[at + old.len ..];
};

var sorted_inverse_kernel: ?mlx.mlx_fast_metal_kernel = null;

var sorted_inverse_cfgs: kernel_src.CfgCache(kernels.DecodeReduceKey, 8) = .{};

var sorted_inverse_engaged: bool = false;

pub fn finishSortedInverse(s: mlx.mlx_stream, inner: mlx.mlx_array, inverse: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, dim: c_int, rows: c_int, topk: c_int, dtype: mlx.mlx_dtype) !mlx.mlx_array {
    if (topk < 1 or topk > kernels.REDUCE_MAX_TOPK) return error.Exl3TopkUnsupported;
    const key = kernels.DecodeReduceKey{ .out_dim = dim, .rows = rows, .topk = topk, .dtype = dtype };
    const cfg = sorted_inverse_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ rows, dim }, 2, dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * topk * @divExact(dim, 128), rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32 * topk, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        sorted_inverse_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = try kernels.getNamedKernel(&sorted_inverse_kernel, "sushi_exl3_sorted_inverse_reduce", &.{ "inner", "inverse", "svh", "slots", "sc" }, &.{"y"}, SORTED_INVERSE_SOURCE, "");
    const outputs = try kernels.applyOuts(s, kernel, &.{ inner, inverse, svh, slots, scores }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs, 0));
    if (!sorted_inverse_engaged) {
        sorted_inverse_engaged = true;
        if (!@import("builtin").is_test) log.info("[exl3-prefill] sorted finish/reduce engaged dtype={s}\n", .{@tagName(dtype)});
    }
    return y;
}

pub var gpu_window_prefill_force: ?bool = null;

/// The GPU window metadata and the inverse-indexed finish, at the two served geometries (MiMo, Flash-Next).
pub fn gpuWindowPrefillOn(hidden: c_int, inter: c_int, experts: c_int, topk: c_int) bool {
    if (gpu_window_prefill_force) |v| return v;
    if (experts < 1) return false;
    return (hidden == 4096 and inter == 2048 and experts <= 256 and topk == 8) or
        (hidden == 2560 and inter == 640 and experts <= GPU_WINDOW_MAX_EXPERTS and topk == 10);
}

test "exl3 GPU window metadata has a routing-independent capacity" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    for ([_]c_int{ 256, GPU_WINDOW_MAX_EXPERTS }) |experts| try gpuWindowCapacityCase(s, experts);
}

test "exl3 prefill GPU metadata keys on the served geometries, never on the rate" {
    try std.testing.expect(gpuWindowPrefillOn(4096, 2048, 256, 8));
    try std.testing.expect(gpuWindowPrefillOn(2560, 640, 512, 10));
    try std.testing.expect(!gpuWindowPrefillOn(2560, 640, 513, 10));
    try std.testing.expect(!gpuWindowPrefillOn(2048, 512, 256, 8));
}

test "exl3 Flash-Next prefill reads the sorted down plane through the inverse, to the byte" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const c: kernels.MimoMoeCase = .{ .e = 32, .hidden = 2560, .inter = 640, .topk = 10, .rows = 200, .rate = .{ .n = 48 }, .dec = .{ .codebook = .mcg, .window = .w15 }, .seed = 653 };
    kernels.setDecodeParams(c.dec);
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f = try kernels.mimoMoeFixture(arena.allocator(), c);
    defer f.deinit();
    defer gpu_window_prefill_force = null;
    gpu_window_prefill_force = false;
    const scattered = try kernels.mimoPrefillArm(s, &f, c.topk);
    defer _ = mlx.mlx_array_free(scattered);
    gpu_window_prefill_force = null;
    gpu_window_engaged = false;
    sorted_inverse_engaged = false;
    const served = try kernels.mimoPrefillArm(s, &f, c.topk);
    defer _ = mlx.mlx_array_free(served);
    try std.testing.expect(gpu_window_engaged and sorted_inverse_engaged);
    var ca = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ca);
    var cb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cb);
    const a = try kernels.evalF16(s, scattered, &ca);
    const b = try kernels.evalF16(s, served, &cb);
    const n = c.rows * c.hidden;
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a[0..n]), std.mem.sliceAsBytes(b[0..n]));
}
