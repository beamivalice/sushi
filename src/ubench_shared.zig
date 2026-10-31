//! Benches that cross model families. `shared.kernel_launch` times the host side of one custom-kernel launch;
//! `shared.shared_expert_fold` times one MoE layer with the shared expert as its own
//! kernels, folded into the routed launches, and absent, for Flash-Next, GLM and DeepSeek-V4.1. Run by `sushi-ubench`,
//! never by a test; the bytes of every arm are asserted by the tests beside the folds.
const std = @import("std");
const mlx = @import("mlx.zig");
const io_util = @import("io_util.zig");
const report = @import("ubench_report.zig");
const api = @import("sushi_exl3");
const glm = @import("glm5/glm5_shared.zig");
const qwen = @import("qwen4/qwen4_shared.zig");
const hc_prefill = @import("hc_prefill.zig");

const Arr = mlx.mlx_array;
const Ops = qwen.Ops;
const Bank = @import("glm5/glm5_decode.zig").Bank;

pub const benches = [_]report.Bench{ shared_expert_fold, kernel_launch };

const alloc = std.heap.page_allocator;
const max_layers = 32;
const default_layers = 8;

const FoldArm = enum { separate, fused, routed_only };

pub const shared_expert_fold: report.Bench = .{
    .name = "shared.shared_expert_fold",
    .scope = .shared,
    .what = "one MoE layer with the shared expert separate, folded into the routed launches, and absent: qwen4 (one bank, rate groups), glm5 (one bank), dsv41 (rate groups, layer 2); dsv41 also counts launches",
    .unit = .us,
    .steps = default_layers,
    .axes = &.{
        .{ .name = "model", .values = &.{ "qwen4", "glm5", "dsv41" }, .numeric = false },
        .{ .name = "layout", .values = &.{ "bank", "grouped" }, .numeric = false },
        .{ .name = "rows", .values = &.{ "1", "2", "3", "4", "6", "8" } },
    },
    .arms = &.{ "separate", "fused", "routed_only" },
    .ref = "separate",
    .run = runFold,
};

fn listed(values: []const []const u8, v: []const u8) bool {
    for (values) |x| if (std.mem.eql(u8, x, v)) return true;
    return false;
}

fn io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

fn eval(a: Arr) !void {
    try mlx.check(mlx.mlx_array_eval(a));
}

fn runFold(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    const models = run.axis("model");
    const layouts = run.axis("layout");
    if (listed(models, "qwen4")) for (layouts) |layout| try runQwen(run, s, std.mem.eql(u8, layout, "grouped"));
    if (listed(models, "glm5") and listed(layouts, "bank")) try runGlm(run, s);
    if (listed(models, "dsv41") and listed(layouts, "grouped")) try api.ubench.sharedFoldDsv41(run);
}

// ── glm5: one routed layer at the real geometry (nine experts at n36, 4096 -> 2048 -> 4096), shared A6 ──

const GlmRung = struct {
    s: mlx.mlx_stream,
    rows: usize,
    layers: usize,
    sigmoid: Arr,
    routed: []const api.Bank,
    gate: []const Bank,
    up: []const Bank,
    down: []const Bank,
    ids: []const Arr,
    scores: []const Arr,
};

/// Nanoseconds of one layer of a chain over distinct banks, on a fresh activation.
fn glmSample(r: *const GlmRung, arm: usize) anyerror!f64 {
    var scope = Ops{ .s = r.s };
    defer scope.deinit();
    var prng = std.Random.DefaultPrng.init(0x757);
    const x0 = try scope.own(try glm.randomRow(alloc, prng.random(), r.rows * 4096, 4));
    var x = try scope.reshape(x0, &.{ 1, @intCast(r.rows), 4096 });
    try eval(x);
    const sw = io_util.Stopwatch.init(io());
    for (0..r.layers) |l| {
        x = switch (@as(FoldArm, @enumFromInt(arm))) {
            .separate => if (r.rows == 1)
                try glm.separateChain(&scope, x, r.gate[l], r.up[l], r.down[l], 6, r.routed[l], r.ids[l], r.scores[l], r.sigmoid)
            else
                (try glm.separateRows(&scope, x, r.gate[l], r.up[l], r.down[l], r.routed[l], r.ids[l], r.scores[l])) orelse return error.FoldDeclined,
            .routed_only => if (r.rows == 1)
                try scope.own(try api.moeClamped(r.s, x, r.routed[l], r.ids[l], r.scores[l], glm.test_dec, glm.test_limit))
            else
                try scope.own((try api.glm_group2.moeLayout(r.s, x, r.routed[l], r.ids[l], r.scores[l], glm.test_dec, .serial, .grouped, .lane)) orelse return error.FoldDeclined),
            .fused => try scope.own((try glm.folded(r.s, x, r.gate[l], r.up[l], r.down[l], r.routed[l], r.ids[l], r.scores[l], glm.test_dec, 10, r.sigmoid)) orelse return error.FoldDeclined),
        };
    }
    try eval(x);
    return @as(f64, @floatFromInt(sw.read())) / @as(f64, @floatFromInt(r.layers));
}

fn runGlm(run: *report.Run, s: mlx.mlx_stream) !void {
    const layers = @min(run.steps(default_layers), max_layers);
    var prng = std.Random.DefaultPrng.init(757);
    const rnd = prng.random();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const sigmoid = try hc_prefill.sigmoidTable(s);
    var routed: [max_layers]api.Bank = undefined;
    var fixtures: [max_layers]glm.Fixture = @splat(.{});
    defer for (&fixtures) |*f| f.deinit();
    var gate: [max_layers]Bank = undefined;
    var up: [max_layers]Bank = undefined;
    var down: [max_layers]Bank = undefined;
    for (0..layers) |l| {
        routed[l] = try glm.routedBank(&ops, 36);
        gate[l] = try fixtures[l].bank(alloc, rnd, 2048, 4096, 6, 1);
        up[l] = try fixtures[l].bank(alloc, rnd, 2048, 4096, 6, 1);
        down[l] = try fixtures[l].bank(alloc, rnd, 4096, 2048, 6, 1);
    }
    for (routed[0..layers]) |b| for ([_]Arr{ b.gate.trellis, b.up.trellis, b.down.trellis }) |a| try eval(a);
    for (run.axis("rows")) |rows_text| {
        const rows = try std.fmt.parseInt(usize, rows_text, 10);
        var ids: [max_layers]Arr = undefined;
        var scores: [max_layers]Arr = undefined;
        for (0..layers) |l| {
            const route = try glm.routeInputs(&ops, rnd, rows);
            ids[l] = route.ids;
            scores[l] = route.scores;
        }
        const rung: GlmRung = .{ .s = s, .rows = rows, .layers = layers, .sigmoid = sigmoid, .routed = routed[0..layers], .gate = gate[0..layers], .up = up[0..layers], .down = down[0..layers], .ids = ids[0..layers], .scores = scores[0..layers] };
        const label = try std.fmt.allocPrint(run.a, "glm5 bank rows {d}", .{rows});
        try run.interleave(label, &rung, glmSample);
    }
}

// ── qwen4: a Flash-Next layer (2560 -> 640, top-10, 64 experts), the shared expert affine8 with its gate logit ──

const QwenRung = struct {
    s: mlx.mlx_stream,
    rows: usize,
    layers: usize,
    dec: api.format.Decode,
    weights: []const qwen.Weights,
    chains: []const qwen.Chain,
    ids: []const Arr,
    scores: []const Arr,
};

fn qwenSample(r: *const QwenRung, arm: usize) anyerror!f64 {
    var scope = Ops{ .s = r.s };
    defer scope.deinit();
    var prng = std.Random.DefaultPrng.init(0x913);
    var x = try qwen.randomBf16(&scope, alloc, prng.random(), &.{ 1, @intCast(r.rows), qwen.hidden }, 4);
    try eval(x);
    const sw = io_util.Stopwatch.init(io());
    for (0..r.layers) |l| {
        const chain = r.chains[l];
        x = switch (@as(FoldArm, @enumFromInt(arm))) {
            .separate => try qwen.separate(&scope, x, r.weights[l], chain, r.ids[l], r.scores[l], r.dec),
            .routed_only => try scope.own(switch (chain) {
                .bank => |b| try api.moe(r.s, x, b, r.ids[l], r.scores[l], r.dec, false),
                .ragged => |rb| try api.moeRagged(r.s, x, rb, r.ids[l], r.scores[l], r.dec, false, null, null),
            }),
            .fused => blk: {
                var sh = r.weights[l].shared;
                sh.logit = r.weights[l].logit;
                const got = switch (chain) {
                    .bank => |b| try api.qwen_fold.moeGatedSharedFolded(r.s, x, b, r.ids[l], r.scores[l], r.dec, sh, null),
                    .ragged => |rb| try api.rate_groups.moeGatedSharedFolded(r.s, x, rb, r.ids[l], r.scores[l], r.dec, sh, null),
                };
                break :blk try scope.own(got orelse return error.FoldDeclined);
            },
        };
    }
    try eval(x);
    return @as(f64, @floatFromInt(sw.read())) / @as(f64, @floatFromInt(r.layers));
}

const bank_experts = 64;
const grouped_counts = [_]c_int{ 20, 20, 24 };

fn runQwen(run: *report.Run, s: mlx.mlx_stream, grouped: bool) !void {
    const layers = @min(run.steps(default_layers), max_layers);
    var prng = std.Random.DefaultPrng.init(913);
    const rnd = prng.random();
    var bops: [max_layers]Ops = @splat(.{ .s = s });
    defer for (bops[0..layers]) |*b| b.deinit();
    var ragged: [max_layers]api.Ragged = undefined;
    var made: usize = 0;
    defer for (ragged[0..made]) |*r| r.deinit();
    var weights: [max_layers]qwen.Weights = undefined;
    var chains: [max_layers]qwen.Chain = undefined;
    for (0..layers) |l| {
        weights[l] = try qwen.weights(&bops[l], alloc, rnd, s, qwen.hidden, qwen.inter, 128, 1);
        if (grouped) {
            var parts: [3]api.Bank = undefined;
            for (&parts, grouped_counts, qwen.group_rates, 0..) |*p, count, r, g| p.* = try qwen.routedBank(&bops[l], count, r[0], r[2], 500 + 100 * @as(u64, @intCast(l)) + @as(u64, @intCast(g)));
            ragged[l] = try api.Ragged.init(s, &parts);
            made += 1;
            chains[l] = .{ .ragged = &ragged[l] };
        } else chains[l] = .{ .bank = try qwen.routedBank(&bops[l], bank_experts, 64, 64, 900 + @as(u64, @intCast(l))) };
    }
    const dec = if (grouped) api.format.Decode.tabled(.spec1, .w15) else qwen.test_dec;
    for (run.axis("rows")) |rows_text| {
        const rows = try std.fmt.parseInt(usize, rows_text, 10);
        var rops = Ops{ .s = s };
        defer rops.deinit();
        var ids: [max_layers]Arr = undefined;
        var scores: [max_layers]Arr = undefined;
        for (0..layers) |l| {
            const route = try qwen.routing(&rops, rnd, rows, 60);
            ids[l] = route.ids;
            scores[l] = route.scores;
        }
        const rung: QwenRung = .{ .s = s, .rows = rows, .layers = layers, .dec = dec, .weights = weights[0..layers], .chains = chains[0..layers], .ids = ids[0..layers], .scores = scores[0..layers] };
        const label = try std.fmt.allocPrint(run.a, "qwen4 {s} rows {d}", .{ if (grouped) "grouped" else "bank", rows });
        try run.interleave(label, &rung, qwenSample);
    }
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
        const kernel = mlx.mlx_fast_metal_kernel_new(name, iv, ov, launch_source, header, true, false);
        if (kernel.ctx == null) return error.MetalKernelCompileFailed;
        defer _ = mlx.mlx_fast_metal_kernel_free(kernel);
        const rung: LaunchRung = .{ .s = s, .kernel = kernel, .x = x, .w = x, .sc = sc, .steps = steps };
        const label = try std.fmt.allocPrint(run.a, "header {s}", .{size});
        try run.interleave(label, &rung, launchSample);
    }
}
