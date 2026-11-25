//! Benches that cross model families. `shared.kernel_launch` times the host side of one custom-kernel launch;
//! `shared.shared_expert_fold` times one MoE layer with the shared expert as its own
//! kernels, folded into the routed launches, and absent, for Flash-Next, GLM and DeepSeek-V4.1. Run by `sushi-ubench`,
//! never by a test; the bytes of every arm are asserted by the tests beside the folds. `shared.exl3_trunk_prefill` times
//! one Flash-Next trunk projection at prefill width on the sorted GEMM and the decoded path against affine 8.
//! `shared.read_bandwidth` and `shared.gemm_peak` measure the box's own ceilings: the GPU's sustained streaming read and
//! MLX's dense matmul at Hy4-Flash's prefill shapes.
const std = @import("std");
const mlx = @import("mlx.zig");
const io_util = @import("io_util.zig");
const report = @import("ubench_report.zig");
const api = @import("sushi_exl3");
const arch_registry = @import("arch/registry.zig");

const Arr = mlx.mlx_array;
const kernels = api.kernels;
const exl3_ubench = @import("sushi_exl3_ubench");
const bench = exl3_ubench.bench_util;

pub const benches = [_]report.Bench{ shared_expert_fold, kernel_launch, trunk_prefill, read_bandwidth, gemm_peak };

pub const alloc = std.heap.page_allocator;
pub const max_layers = 32;
pub const default_layers = 8;

pub const FoldArm = enum { separate, fused, routed_only };

pub const shared_expert_fold: report.Bench = .{
    .name = "shared.shared_expert_fold",
    .scope = .shared,
    .what = "one MoE layer with the shared expert separate, folded into the routed launches, and absent: qwen4 (one bank, rate groups; the shared expert affine8 or a 6-bit trellis trunk), glm5 (one bank), dsv41 (rate groups, layer 2); dsv41 also counts launches",
    .unit = .us,
    .steps = default_layers,
    .axes = &.{
        .{ .name = "model", .values = &.{ "qwen4", "glm5", "dsv41" }, .numeric = false },
        .{ .name = "layout", .values = &.{ "bank", "grouped" }, .numeric = false },
        .{ .name = "shared", .values = &.{ "affine", "trellis" }, .numeric = false },
        .{ .name = "rows", .values = &.{ "1", "2", "3", "4", "6", "8" } },
    },
    .arms = &.{ "separate", "fused", "routed_only" },
    .ref = "separate",
    .run = runFold,
};

pub fn listed(values: []const []const u8, v: []const u8) bool {
    for (values) |x| if (std.mem.eql(u8, x, v)) return true;
    return false;
}

pub fn io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

pub fn eval(a: Arr) !void {
    try mlx.check(mlx.mlx_array_eval(a));
}

fn runFold(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    const models = run.axis("model");
    const layouts = run.axis("layout");
    // The bench's own model order, so the rungs run in one order whatever the registry's.
    for (shared_expert_fold.axes[0].values) |model| if (listed(models, model)) inline for (comptime std.meta.tags(arch_registry.Tag)) |t| {
        const M = arch_registry.Mod(t);
        if (comptime @hasDecl(M, "ubench") and @hasDecl(M.ubench, "sharedFold")) {
            if (std.mem.eql(u8, model, @tagName(t))) try M.ubench.sharedFold(run, s, layouts);
        }
    };
    if (listed(run.axis("shared"), "affine") and listed(models, "dsv41") and listed(layouts, "grouped")) try exl3_ubench.ubench.sharedFoldDsv41(run);
}

// ── custom-kernel launch: the host's side of `mlx_fast_metal_kernel_apply` and of its eval ──

const launch_max_steps = 256;
const launch_width = 4096;

pub const kernel_launch: report.Bench = .{
    .name = "shared.kernel_launch",
    .scope = .shared,
    .what = "host us of one custom-kernel launch on tiny inputs (f16 x, w, 4-value scale; 4096 outputs) with a 1k or a 32k header: the apply alone, and apply plus its share of one eval of the batch",
    .unit = .us,
    .steps = 64,
    .axes = &.{.{ .name = "header", .values = &.{ "1k", "32k" }, .numeric = false }},
    .arms = &.{ "apply", "apply_eval" },
    .ref = "apply",
    .run = runLaunch,
};

const launch_source =
    \\uint i = thread_position_in_grid.x;
    \\out[i] = x[i] * w[i] + sc[i & 3u] + T(helper_0(1.0f));
;

const LaunchRung = struct {
    s: mlx.mlx_stream,
    kernel: mlx.mlx_fast_metal_kernel,
    x: Arr,
    w: Arr,
    sc: Arr,
    steps: usize,
};

/// Nanoseconds per launch; the arm decides whether the eval of the batch is inside the clock.
fn launchSample(r: *const LaunchRung, arm: usize) anyerror!f64 {
    const inputs = mlx.mlx_vector_array_new_data(&[_]Arr{ r.x, r.w, r.sc }, 3);
    defer _ = mlx.mlx_vector_array_free(inputs);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{launch_width}, 1, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, launch_width, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .float16));
    const batch = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(batch);
    const sw = io_util.Stopwatch.init(io());
    for (0..r.steps) |_| {
        var outs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, r.kernel, inputs, cfg, r.s));
        var y = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
        try mlx.check(mlx.mlx_vector_array_append_value(batch, y));
    }
    const apply_ns = sw.read();
    try mlx.check(mlx.mlx_eval(batch));
    const ns = if (arm == 0) apply_ns else sw.read();
    return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(r.steps));
}

/// Helper functions to pad a kernel header to `bytes`, the size of a real sushi kernel's source.
fn launchHeader(bytes: usize) ![:0]u8 {
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(alloc);
    var n: usize = 0;
    while (text.items.len < bytes) : (n += 1) try text.print(alloc, "static inline float helper_{d}(float v) {{ return v * 1.0001f + {d}.0f; }}\n", .{ n, n });
    const out = try alloc.allocSentinel(u8, text.items.len, 0);
    @memcpy(out, text.items);
    return out;
}

fn runLaunch(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    const steps = @min(run.steps(64), launch_max_steps);
    var vals: [launch_width]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = 0.001 * @as(f32, @floatFromInt(i % 97));
    const x32 = mlx.mlx_array_new_data(&vals, &[_]c_int{launch_width}, 1, .float32);
    defer _ = mlx.mlx_array_free(x32);
    const sc32 = mlx.mlx_array_new_data(&vals, &[_]c_int{4}, 1, .float32);
    defer _ = mlx.mlx_array_free(sc32);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_astype(&x, x32, .float16, s));
    try mlx.check(mlx.mlx_astype(&sc, sc32, .float16, s));
    try eval(x);
    try eval(sc);
    const names = [_][*:0]const u8{ "x", "w", "sc" };
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    for (run.axis("header")) |size| {
        const header = try launchHeader(if (std.mem.eql(u8, size, "1k")) 1024 else 32 * 1024);
        defer alloc.free(header);
        const name = try std.fmt.allocPrintSentinel(alloc, "sushi_ubench_launch_{s}", .{size}, 0);
        defer alloc.free(name);
        const kernel = mlx.kernelNewSource(name, iv, ov, launch_source, header, true, false);
        if (kernel.ctx == null) return error.MetalKernelCompileFailed;
        defer _ = mlx.mlx_fast_metal_kernel_free(kernel);
        const rung: LaunchRung = .{ .s = s, .kernel = kernel, .x = x, .w = x, .sc = sc, .steps = steps };
        const label = try std.fmt.allocPrint(run.a, "header {s}", .{size});
        try run.interleave(label, &rung, launchSample);
    }
}

// ---- shared.exl3_trunk_prefill: one Flash-Next trunk projection at prefill width ----

/// One prefill projection: `out x in`, and the group its affine-8 twin is stored with.
const TrunkShape = struct { name: []const u8, out: usize, in: usize, group: c_int };

const TRUNK_SHAPES = [_]TrunkShape{
    .{ .name = "wq_b", .out = 32768, .in = 1280, .group = 64 },
    .{ .name = "wo_b", .out = 5120, .in = 8192, .group = 64 },
    .{ .name = "gdn_qkv", .out = 10240, .in = 2560, .group = 128 },
    .{ .name = "gdn_z", .out = 6144, .in = 2560, .group = 128 },
    .{ .name = "gdn_out", .out = 2560, .in = 6144, .group = 128 },
    .{ .name = "attn_q", .out = 12288, .in = 2560, .group = 64 },
    .{ .name = "attn_kv", .out = 512, .in = 2560, .group = 64 },
    .{ .name = "attn_o", .out = 2560, .in = 6144, .group = 64 },
    .{ .name = "idx_q", .out = 640, .in = 2560, .group = 64 },
    .{ .name = "shared_gate", .out = 640, .in = 2560, .group = 128 },
    .{ .name = "shared_down", .out = 2560, .in = 640, .group = 128 },
};

const trunk_shape_names = blk: {
    var names: [TRUNK_SHAPES.len][]const u8 = undefined;
    for (TRUNK_SHAPES, 0..) |sh, i| names[i] = sh.name;
    break :blk names;
};

pub const trunk_prefill: report.Bench = .{
    .name = "shared.exl3_trunk_prefill",
    .scope = .shared,
    .what = "one trunk projection at prefill width: the sorted trellis GEMM and the billed path (decode once to f16 for MLX's GEMM where admitted), at K4, K5 and K6 against affine-8 quantized_matmul and a dense bf16 matmul ceiling",
    .unit = .us,
    .steps = 4,
    .axes = &.{
        .{ .name = "shape", .values = &trunk_shape_names, .numeric = false },
        .{ .name = "k", .values = &.{ "4", "5", "6" } },
        .{ .name = "rows", .values = &.{ "512", "2048", "4096" } },
    },
    .arms = &.{ "affine8", "sorted", "decoded", "dense" },
    .ref = "affine8",
    .run = runTrunkPrefill,
};

/// Copies of each weight a chain cycles through, so a step never reads the previous step's weight from cache.
const TRUNK_COPIES = 2;

const TrunkCtx = struct {
    s: mlx.mlx_stream,
    sh: TrunkShape,
    x0: mlx.mlx_array,
    trellis: [TRUNK_COPIES]api.Proj,
    a8: [TRUNK_COPIES][3]mlx.mlx_array,
    dense: [TRUNK_COPIES]mlx.mlx_array,
    link_ns: f64,
};

fn trunkStep(c: *const TrunkCtx, s: mlx.mlx_stream, i: usize, x: mlx.mlx_array, arm: usize) !mlx.mlx_array {
    const ci = i % TRUNK_COPIES;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    switch (arm) {
        0 => try mlx.check(mlx.mlx_quantized_matmul(&y, x, c.a8[ci][0], c.a8[ci][1], c.a8[ci][2], true, mlx.mlx_optional_int.some(c.sh.group), mlx.mlx_optional_int.some(8), "affine", s)),
        1, 2 => {
            _ = mlx.mlx_array_free(y);
            y = try api.linear.linearWith(s, x, c.trellis[ci], kernels.SPEC1_W15, .{ .rows = .{ .exact = true }, .decoded_billed = arm == 2 });
        },
        3 => {
            var wt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wt);
            try mlx.check(mlx.mlx_transpose(&wt, c.dense[ci], s));
            try mlx.check(mlx.mlx_matmul(&y, x, wt, s));
        },
        else => unreachable,
    }
    return y;
}

const TrunkArmCtx = struct { c: *const TrunkCtx, arm: usize };

fn trunkArmStep(ac: TrunkArmCtx, s: mlx.mlx_stream, i: usize, x: mlx.mlx_array) anyerror!mlx.mlx_array {
    return trunkStep(ac.c, s, i, x, ac.arm);
}

const TrunkSample = struct { c: *const TrunkCtx, steps: usize };

fn trunkSample(smp: TrunkSample, arm: usize) anyerror!f64 {
    kernels.setDecodeParams(kernels.SPEC1_W15);
    const ns = try bench.chainNs(smp.c.s, smp.c.x0, smp.steps, TrunkArmCtx{ .c = smp.c, .arm = arm }, trunkArmStep);
    return (@as(f64, @floatFromInt(ns)) - smp.c.link_ns) / @as(f64, @floatFromInt(smp.steps));
}

fn runTrunkPrefill(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    defer kernels.setDecodeParams(.mul1);
    var prng = std.Random.DefaultPrng.init(0x7f);
    const rnd = prng.random();
    try kernels.SPEC1_W15.requireTable();
    for (TRUNK_SHAPES) |sh| {
        if (!listed(run.axis("shape"), sh.name)) continue;
        for (run.axis("k")) |k_text| {
            const k = try std.fmt.parseInt(u32, k_text, 10);
            if (k < 4 or k > 6) return error.BadK;
            const n = k * 16;
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const a = arena.allocator();
            var owned: std.ArrayList(mlx.mlx_array) = .empty;
            defer for (owned.items) |x| {
                _ = mlx.mlx_array_free(x);
            };
            var c: TrunkCtx = .{ .s = s, .sh = sh, .x0 = undefined, .trellis = undefined, .a8 = undefined, .dense = undefined, .link_ns = 0 };
            const suh_v = try a.alloc(u16, sh.in);
            @memset(suh_v, api.format.f32ToF16Bits(1.0 / @sqrt(@as(f32, @floatFromInt(sh.in)))));
            const svh_v = try a.alloc(u16, sh.out);
            @memset(svh_v, api.format.f32ToF16Bits(1.0));
            const suh = mlx.mlx_array_new_data(suh_v.ptr, &[_]c_int{ 1, @intCast(sh.in) }, 2, .float16);
            const svh = mlx.mlx_array_new_data(svh_v.ptr, &[_]c_int{ 1, @intCast(sh.out) }, 2, .float16);
            try owned.appendSlice(a, &.{ suh, svh });
            for (0..TRUNK_COPIES) |ci| {
                const words = try a.alloc(u16, (sh.in / 16) * (sh.out / 16) * n);
                bench.randomWords(rnd, words);
                const tr = mlx.mlx_array_new_data(words.ptr, &[_]c_int{ 1, @intCast(sh.in / 16), @intCast(sh.out / 16), @intCast(n) }, 4, .uint16);
                try owned.append(a, tr);
                c.trellis[ci] = .{ .trellis = tr, .suh = suh, .svh = svh };
                const codes = try a.alloc(u32, sh.out * sh.in / 4);
                for (codes) |*v| v.* = rnd.int(u32);
                const wb = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ @intCast(sh.out), @intCast(sh.in / 4) }, 2, .uint32);
                const gshape = [_]c_int{ @intCast(sh.out), @intCast(sh.in / @as(usize, @intCast(sh.group))) };
                var sb = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_full(&sb, &gshape, 2, mlx.mlx_array_new_float(0.001), .bfloat16, s));
                var bb = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_full(&bb, &gshape, 2, mlx.mlx_array_new_float(-0.1), .bfloat16, s));
                try owned.appendSlice(a, &.{ wb, sb, bb });
                c.a8[ci] = .{ wb, sb, bb };
                var dw = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_random_normal(&dw, &[_]c_int{ @intCast(sh.out), @intCast(sh.in) }, 2, .bfloat16, 0, 0.02, .{ .ctx = null }, s));
                try owned.append(a, dw);
                c.dense[ci] = dw;
            }
            const ev = mlx.mlx_vector_array_new_data(owned.items.ptr, owned.items.len);
            defer _ = mlx.mlx_vector_array_free(ev);
            try mlx.check(mlx.mlx_eval(ev));
            for (run.axis("rows")) |rows_text| {
                const rows = try std.fmt.parseInt(usize, rows_text, 10);
                const xs = try a.alloc(u16, rows * sh.in);
                for (xs) |*v| v.* = @truncate(@as(u32, @bitCast(rnd.float(f32) - 0.5)) >> 16);
                c.x0 = mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ @intCast(rows), @intCast(sh.in) }, 2, .bfloat16);
                defer _ = mlx.mlx_array_free(c.x0);
                const steps = run.steps(4);
                var t: [5]u64 = undefined;
                for (&t) |*v| v.* = try bench.chainNs(s, c.x0, steps, {}, bench.linkOnly);
                c.link_ns = bench.median(&t);
                const label = try std.fmt.allocPrint(run.a, "{s} [{d}x{d}] K{s} rows {d}", .{ sh.name, sh.out, sh.in, k_text, rows });
                try run.interleave(label, TrunkSample{ .c = &c, .steps = steps }, trunkSample);
                const flop: f64 = 2.0 * @as(f64, @floatFromInt(rows * sh.in * sh.out));
                for ([_][]const u8{ "affine8", "sorted", "decoded", "dense" }) |arm| {
                    if (run.hasArm(arm)) try run.metric(arm, "TFLOP/s", "TFLOP/s", flop / run.lastMedianNs(arm) / 1e3);
                }
            }
        }
    }
}

// ---- shared.read_bandwidth: the GPU's sustained streaming read over a buffer far past the system cache ----

pub const read_bandwidth: report.Bench = .{
    .name = "shared.read_bandwidth",
    .scope = .shared,
    .what = "the GPU's sustained streaming read: one kernel reading a 1, 8 or 16 GB buffer once by coalesced 16-byte loads, 16 per thread, as GB/s",
    .unit = .us,
    .steps = 4,
    .axes = &.{.{ .name = "gb", .values = &.{ "1", "8", "16" } }},
    .arms = &.{"read"},
    .ref = "read",
    .run = runReadBandwidth,
};

const read_loads = 16;
var read_kernel: ?mlx.mlx_fast_metal_kernel = null;
const READ_SOURCE: [:0]const u8 =
    \\const ulong tid = ulong(thread_position_in_grid.x);
    \\const ulong total = ulong(threads_per_grid.x);
    \\const device uint4 *p = (const device uint4 *)w;
    \\uint v = 0u;
    \\for (uint j = 0u; j < LOADS; ++j) { const uint4 a = p[tid + ulong(j) * total]; v += a.x ^ a.y ^ a.z ^ a.w; }
    \\if (tid == 0u && float(xin[0]) == 1.5e30f) v = 0u;
    \\v = simd_sum(v);
    \\if ((tid & 31u) == 0u) y[tid >> 5u] = v;
;

const ReadCtx = struct { w: Arr, bytes: usize, link_ns: f64 };

fn readStep(c: *const ReadCtx, s: mlx.mlx_stream, _: usize, x: Arr) anyerror!Arr {
    const nthreads: c_int = @intCast(c.bytes / 16 / read_loads);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{@divExact(nthreads, 32)}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, nthreads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LOADS", read_loads));
    const k = try kernels.getNamedKernel(&read_kernel, "sushi_ubench_read", &.{ "w", "xin" }, &.{"y"}, READ_SOURCE, "");
    const ov = try kernels.applyOuts(s, k, &.{ c.w, x }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
    return y;
}

const ReadSample = struct { c: *const ReadCtx, x0: Arr, steps: usize };

fn readSample(smp: ReadSample, _: usize) anyerror!f64 {
    const ns = try bench.chainNs(mlx.gpuStream(), smp.x0, smp.steps, smp.c, readStep);
    return (@as(f64, @floatFromInt(ns)) - smp.c.link_ns) / @as(f64, @floatFromInt(smp.steps));
}

fn runReadBandwidth(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    var one = [_]f32{0.5};
    const x0 = mlx.mlx_array_new_data(&one, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(x0);
    const steps = run.steps(4);
    for (run.axis("gb")) |gb_text| {
        const gb = try std.fmt.parseInt(usize, gb_text, 10);
        const bytes = gb << 30;
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        // Written by the GPU, so every page is backed (an untouched buffer reads the shared zero page).
        try mlx.check(mlx.mlx_full(&w, &[_]c_int{ @intCast(gb * 1024), 1 << 18 }, 2, mlx.mlx_array_new_float(7.0), .uint32, s));
        try eval(w);
        var t: [5]u64 = undefined;
        for (&t) |*v| v.* = try bench.chainNs(s, x0, steps, {}, bench.linkOnly);
        const c: ReadCtx = .{ .w = w, .bytes = bytes, .link_ns = bench.median(&t) };
        const label = try std.fmt.allocPrint(run.a, "read {d} GiB", .{gb});
        try run.interleave(label, ReadSample{ .c = &c, .x0 = x0, .steps = steps }, readSample);
        try run.metric("read", "GB/s", "GB/s", @as(f64, @floatFromInt(bytes)) / run.lastMedianNs("read"));
    }
}

// ---- shared.gemm_peak: MLX's dense matmul at Hy4-Flash's prefill shapes ----

const GemmShape = struct { name: []const u8, out: usize, in: usize };

const GEMM_SHAPES = [_]GemmShape{
    .{ .name = "square8192", .out = 8192, .in = 8192 },
    .{ .name = "hy4_q_a", .out = 1024, .in = 3072 },
    .{ .name = "hy4_q_b", .out = 16384, .in = 1024 },
    .{ .name = "hy4_gate", .out = 16384, .in = 3072 },
    .{ .name = "hy4_o", .out = 3072, .in = 16384 },
    .{ .name = "hy4_kv_b", .out = 28672, .in = 512 },
    .{ .name = "hy4_shared_gate", .out = 1536, .in = 3072 },
    .{ .name = "hy4_shared_down", .out = 3072, .in = 1536 },
    .{ .name = "hy4_expert", .out = 1536, .in = 1536 },
};

const gemm_shape_names = blk: {
    var names: [GEMM_SHAPES.len][]const u8 = undefined;
    for (GEMM_SHAPES, 0..) |sh, i| names[i] = sh.name;
    break :blk names;
};

pub const gemm_peak: report.Bench = .{
    .name = "shared.gemm_peak",
    .scope = .shared,
    .what = "MLX's dense matmul (x Wᵀ, the NAX tiles where the GPU has them) at Hy4-Flash's prefill shapes and a square 8192, bf16 and f16, as TFLOP/s",
    .unit = .us,
    .steps = 8,
    .axes = &.{
        .{ .name = "shape", .values = &gemm_shape_names, .numeric = false },
        .{ .name = "rows", .values = &.{ "512", "2048", "4096", "8192" } },
    },
    .arms = &.{ "bf16", "f16" },
    .ref = "bf16",
    .run = runGemmPeak,
};

const GemmCtx = struct { w: [2][2]Arr, x0: [2]Arr, link_ns: f64 };

fn gemmStep(c: GemmArm, s: mlx.mlx_stream, i: usize, x: Arr) anyerror!Arr {
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, c.c.w[c.arm][i % 2], s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_matmul(&y, x, wt, s));
    return y;
}

const GemmArm = struct { c: *const GemmCtx, arm: usize };
const GemmSample = struct { c: *const GemmCtx, steps: usize };

fn gemmSample(smp: GemmSample, arm: usize) anyerror!f64 {
    const ns = try bench.chainNs(mlx.gpuStream(), smp.c.x0[arm], smp.steps, GemmArm{ .c = smp.c, .arm = arm }, gemmStep);
    return (@as(f64, @floatFromInt(ns)) - smp.c.link_ns) / @as(f64, @floatFromInt(smp.steps));
}

fn runGemmPeak(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    for (GEMM_SHAPES) |sh| {
        if (!listed(run.axis("shape"), sh.name)) continue;
        var c: GemmCtx = .{ .w = .{ .{ mlx.mlx_array_new(), mlx.mlx_array_new() }, .{ mlx.mlx_array_new(), mlx.mlx_array_new() } }, .x0 = undefined, .link_ns = 0 };
        defer for (c.w) |pair| for (pair) |w| {
            _ = mlx.mlx_array_free(w);
        };
        for (0..2) |arm| for (0..2) |ci| {
            try mlx.check(mlx.mlx_random_normal(&c.w[arm][ci], &[_]c_int{ @intCast(sh.out), @intCast(sh.in) }, 2, if (arm == 0) .bfloat16 else .float16, 0, 0.02, .{ .ctx = null }, s));
            try eval(c.w[arm][ci]);
        };
        for (run.axis("rows")) |rows_text| {
            const rows = try std.fmt.parseInt(usize, rows_text, 10);
            c.x0 = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
            defer for (c.x0) |x| {
                _ = mlx.mlx_array_free(x);
            };
            for (0..2) |arm| {
                try mlx.check(mlx.mlx_random_normal(&c.x0[arm], &[_]c_int{ @intCast(rows), @intCast(sh.in) }, 2, if (arm == 0) .bfloat16 else .float16, 0, 1, .{ .ctx = null }, s));
                try eval(c.x0[arm]);
            }
            const steps = run.steps(8);
            var t: [5]u64 = undefined;
            for (&t) |*v| v.* = try bench.chainNs(s, c.x0[0], steps, {}, bench.linkOnly);
            c.link_ns = bench.median(&t);
            const label = try std.fmt.allocPrint(run.a, "{s} [{d}x{d}] rows {d}", .{ sh.name, sh.out, sh.in, rows });
            try run.interleave(label, GemmSample{ .c = &c, .steps = steps }, gemmSample);
            const flop: f64 = 2.0 * @as(f64, @floatFromInt(rows * sh.in * sh.out));
            for ([_][]const u8{ "bf16", "f16" }) |arm| if (run.hasArm(arm)) try run.metric(arm, "TFLOP/s", "TFLOP/s", flop / run.lastMedianNs(arm) / 1e3);
        }
    }
}
