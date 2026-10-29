//! GLM-5.3-Flash kernels on synthetic tensors: the verify-row projections, the A6 dense-once trunk, the NAX-shaped
//! arms, one T2048 KDA layer, and MLA attention for a decode row and for a chunk. Run by `sushi-ubench`, never by a
//! test; the exactness of every arm is asserted by the tests beside the kernels.
const std = @import("std");
const mlx = @import("mlx.zig");
const io_util = @import("io_util.zig");
const report = @import("ubench_report.zig");
const transformer = @import("transformer.zig");
const model = @import("model.zig");
const glm = @import("glm5/glm5_model.zig");
const attention = @import("glm5/glm5_attention.zig");
const latent_store = @import("glm5/glm5_latent.zig");
const a6 = @import("glm5/glm5_a6_dense_once.zig");
const dflash_qmm = @import("glm5/glm5_dflash_qmm.zig");
const kda_prework = @import("glm5/glm5_kda_prework.zig");

const Arr = mlx.mlx_array;
const Ops = glm.Ops;
const Linear = glm.Linear;
const Scope = attention.Scope;
const packed_nax = attention.packed_nax;

pub const benches = [_]report.Bench{ verify_qmm, a6_trunk, nax_arms, kda_prefill, mla_decode_attn, mla_chunk_attn, mla_chunk_stages };

fn io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

fn nsPer(sw: io_util.Stopwatch, calls: usize) f64 {
    return @as(f64, @floatFromInt(sw.read())) / @as(f64, @floatFromInt(calls));
}

fn eval(a: Arr) !void {
    try mlx.check(mlx.mlx_array_eval(a));
}

fn evalAll(arrays: []const Arr) !void {
    for (arrays) |a| try eval(a);
}

fn indexOf(values: []const []const u8, v: []const u8) usize {
    return for (values, 0..) |x, i| {
        if (std.mem.eql(u8, x, v)) break i;
    } else unreachable;
}

fn parseCount(text: []const u8) !c_int {
    return std.fmt.parseInt(c_int, text, 10);
}

/// The NAX probe override a bench flips per arm, to force NAX-shaped kernels on a GPU that would not pick them.
const NaxGate = struct {
    saved: ?bool,

    fn take() NaxGate {
        return .{ .saved = transformer.vqmm_nax_probe_override };
    }

    fn restore(g: NaxGate) void {
        transformer.vqmm_nax_probe_override = g.saved;
    }
};

// ── glm5.verify_qmm ─────────────────────────────────────────────────────

const QmmShape = struct { name: []const u8, n: c_int, k: c_int };
const qmm_shape_names = [_][]const u8{ "8192x4096", "4096x8192", "4096x16384", "16384x1536", "2048x4096", "4096x2048", "1536x4096" };
const qmm_shapes = [_]QmmShape{
    .{ .name = "8192x4096", .n = 8192, .k = 4096 },
    .{ .name = "4096x8192", .n = 4096, .k = 8192 },
    .{ .name = "4096x16384", .n = 4096, .k = 16384 },
    .{ .name = "16384x1536", .n = 16384, .k = 1536 },
    .{ .name = "2048x4096", .n = 2048, .k = 4096 },
    .{ .name = "4096x2048", .n = 4096, .k = 2048 },
    .{ .name = "1536x4096", .n = 1536, .k = 4096 },
};

/// An affine bank of random codes at `bits` bits with group-128 scales.
fn affineBank(ops: *Ops, n: c_int, k: c_int, bits: c_int, seed: u64) !Linear {
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, seed));
    const codes = try ops.slot();
    try mlx.check(mlx.mlx_random_bits(codes, &[_]c_int{ n, @divExact(k * bits, 32) }, 2, 4, key.*, ops.s));
    const scales = try ops.slot();
    try mlx.check(mlx.mlx_random_uniform(scales, try ops.scalar(0.001, .bfloat16), try ops.scalar(0.02, .bfloat16), &[_]c_int{ n, @divExact(k, 128) }, 2, .bfloat16, key.*, ops.s));
    const biases = try ops.binary(.mul, scales.*, try ops.scalar(if (bits == 5) -15.5 else -31.5, .bfloat16));
    const linear: Linear = .{ .w = codes.*, .scales = scales.*, .biases = biases, .input = k, .output = n };
    try evalAll(&.{ linear.w, linear.scales, linear.biases });
    return linear;
}

const QmmCase = struct { s: mlx.mlx_stream, linears: []const Linear, x: Arr, rows: usize };

fn sampleQmm(c: QmmCase, arm: usize) anyerror!f64 {
    var ops = Ops{ .s = c.s };
    defer ops.deinit();
    const outs = try std.heap.page_allocator.alloc(Arr, c.linears.len);
    defer std.heap.page_allocator.free(outs);
    const sw = io_util.Stopwatch.init(io());
    for (c.linears, outs) |linear, *out| out.* = switch (arm) {
        0 => blk: {
            var parts: [8]Arr = undefined;
            for (0..c.rows) |r| parts[r] = try linear.apply(&ops, try ops.slice(c.x, 1, @intCast(r), @intCast(r + 1)));
            break :blk try ops.concat(parts[0..c.rows], 1);
        },
        1 => try ops.own((try dflash_qmm.project(c.s, c.x, linear)).?),
        else => try ops.own((try dflash_qmm.tiles(c.s, c.x, linear)).?),
    };
    const group = mlx.mlx_vector_array_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_array_free(group);
    try mlx.check(mlx.mlx_eval(group));
    return nsPer(sw, outs.len);
}

pub const verify_qmm: report.Bench = .{
    .name = "glm5.verify_qmm",
    .scope = .glm5,
    .what = "verify-row projections per stored width: serial qmv rows, the row-streaming kernel, row tiles, timed per projection over distinct banks",
    .unit = .us,
    .steps = 12,
    .axes = &.{
        .{ .name = "bits", .values = &.{ "6", "5" }, .numeric = false },
        .{ .name = "shape", .values = &qmm_shape_names, .numeric = false },
        .{ .name = "rows", .values = &.{"3"} },
    },
    .arms = &.{ "serial", "rows", "tiles" },
    .ref = "serial",
    .run = runVerifyQmm,
};

fn runVerifyQmm(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    const banks = run.steps(12);
    if (banks > 16) return error.TooManyBanks;
    const linears = try run.a.alloc(Linear, banks);
    for (run.axis("bits")) |bits_text| for (run.axis("shape")) |shape_text| {
        const bits = try parseCount(bits_text);
        const shape = qmm_shapes[indexOf(&qmm_shape_names, shape_text)];
        var fx = Ops{ .s = s };
        defer fx.deinit();
        for (linears, 0..) |*linear, i| linear.* = try affineBank(&fx, shape.n, shape.k, bits, i * 7919 + @as(usize, @intCast(shape.n)));
        const key = try fx.slot();
        try mlx.check(mlx.mlx_random_key(key, 31));
        for (run.axis("rows")) |rows_text| {
            const rows: usize = @intCast(try parseCount(rows_text));
            if (rows < 2 or rows > 8) return error.RowsOutOfRange;
            const x = try fx.slot();
            try mlx.check(mlx.mlx_random_normal(x, &[_]c_int{ 1, @intCast(rows), shape.k }, 3, .bfloat16, 0, 1, key.*, s));
            try eval(x.*);
            const label = try std.fmt.allocPrint(run.a, "A{d} {s} T{d}", .{ bits, shape.name, rows });
            try run.interleave(label, QmmCase{ .s = s, .linears = linears, .x = x.*, .rows = rows }, sampleQmm);
        }
    };
}

// ── glm5.a6_trunk ───────────────────────────────────────────────────────

const a6_shape_names = [_][]const u8{ "4096_8192", "8192_4096", "4096_2048", "2048_4096", "4096_12288", "12288_4096", "4096_1536", "1536_16384", "4096_512", "1536_4096" };
const a6_shapes = [_][2]c_int{ .{ 4096, 8192 }, .{ 8192, 4096 }, .{ 4096, 2048 }, .{ 2048, 4096 }, .{ 4096, 12288 }, .{ 12288, 4096 }, .{ 4096, 1536 }, .{ 1536, 16384 }, .{ 4096, 512 }, .{ 1536, 4096 } };

const A6Case = struct { s: mlx.mlx_stream, input: a6.Input, repeats: usize };

fn denseOnce(ops: *Ops, input: a6.Input) !Arr {
    return ops.binary(.mm, input.x, try ops.transpose(try ops.dequant(input.w, input.scales, input.biases), &.{ 1, 0 }));
}

fn sampleA6(c: A6Case, arm: usize) anyerror!f64 {
    var ops = Ops{ .s = c.s };
    defer ops.deinit();
    var outs: [64]Arr = undefined;
    const sw = io_util.Stopwatch.init(io());
    for (outs[0..c.repeats]) |*out| out.* = if (arm == 0) try ops.qmm(c.input.x, c.input.w, c.input.scales, c.input.biases, true) else try denseOnce(&ops, c.input);
    try evalAll(outs[0..c.repeats]);
    return nsPer(sw, c.repeats);
}

/// How many of the dense-once result's bf16 values differ from qmm's, over one call.
fn a6Differ(s: mlx.mlx_stream, input: a6.Input) !usize {
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const want = try ops.qmm(input.x, input.w, input.scales, input.biases, true);
    const got = try denseOnce(&ops, input);
    try evalAll(&.{ want, got });
    const count = mlx.mlx_array_size(want);
    var differ: usize = 0;
    for (mlx.mlx_array_data_bfloat16(want).?[0..count], mlx.mlx_array_data_bfloat16(got).?[0..count]) |x, y| differ += @intFromBool(x != y);
    return differ;
}

pub const a6_trunk: report.Bench = .{
    .name = "glm5.a6_trunk",
    .scope = .glm5,
    .what = "A6 g128 trunk projections at 2048 rows: qmm against dequantize + dense NAX GEMM, and how many bf16 values differ",
    .unit = .us,
    .steps = 6,
    .axes = &.{
        .{ .name = "shape", .values = &a6_shape_names, .numeric = false },
    },
    .arms = &.{ "qmm", "dense_gemm" },
    .ref = "qmm",
    .run = runA6Trunk,
};

fn runA6Trunk(run: *report.Run) anyerror!void {
    if (!glm.naxArms()) return error.NeedsNaxGpu;
    if (run.steps(6) > 64) return error.TooManyCalls;
    const s = mlx.gpuStream();
    for (run.axis("shape")) |name| {
        const idx = indexOf(&a6_shape_names, name);
        const k = a6_shapes[idx][0];
        const n = a6_shapes[idx][1];
        var fx = Ops{ .s = s };
        defer fx.deinit();
        const input = try a6.fixture(&fx, k, n, @intCast(7000 + idx));
        const differ = try a6Differ(s, input);
        const label = try std.fmt.allocPrint(run.a, "{d}->{d}", .{ k, n });
        try run.interleave(label, A6Case{ .s = s, .input = input, .repeats = run.steps(6) }, sampleA6);
        try run.metric("dense_gemm", "differ", "values", @floatFromInt(differ));
    }
}

// ── glm5.nax_arms ───────────────────────────────────────────────────────

const nax_cases = [_][]const u8{ "a6_dense_once", "mla_query_heads", "mla_value_heads", "mla_verify_t3", "kda_cluster" };

const NaxInputs = struct { x: Arr, a6: [3]Arr, head: [3]Arr, q: Arr, v: Arr, cluster_w: Arr };
const NaxCase = struct { s: mlx.mlx_stream, in: *const NaxInputs, case: usize };

fn naxOutput(ops: *Ops, in: *const NaxInputs, case: usize, nax: bool) !Arr {
    const prefill_batch = @import("glm5/glm5_mla_prefill_batch.zig");
    const verify_batch = @import("glm5/glm5_mla_verify_batch.zig");
    const cluster = @import("glm5/glm5_kda_prefill_cluster.zig");
    return switch (case) {
        0 => if (nax) (try a6.tryPrefill(ops, in.x, in.a6[0], in.a6[1], in.a6[2])).? else try ops.qmm(in.x, in.a6[0], in.a6[1], in.a6[2], true),
        1 => if (nax) (try prefill_batch.run(ops, .{ .x = in.q, .w = in.head[0], .scales = in.head[1], .biases = in.head[2] }, .query)).? else try ops.qmm(in.q, in.head[0], in.head[1], in.head[2], false),
        2 => if (nax) (try prefill_batch.run(ops, .{ .x = in.v, .w = in.head[0], .scales = in.head[1], .biases = in.head[2] }, .value)).? else try ops.qmm(in.v, in.head[0], in.head[1], in.head[2], true),
        3 => blk: {
            const rows = try ops.slice(in.q, 0, 0, 3);
            if (nax) break :blk (try verify_batch.run(ops, .{ .x = rows, .w = in.head[0], .scales = in.head[1], .biases = in.head[2] }, .query)).?;
            var parts: [3]Arr = undefined;
            for (0..3) |r| parts[r] = try ops.qmm(try ops.slice(rows, 0, @intCast(r), @intCast(r + 1)), in.head[0], in.head[1], in.head[2], false);
            break :blk try ops.concat(&parts, 0);
        },
        else => blk: {
            if (nax) {
                const banks = try cluster.project(ops, in.cluster_w, in.x);
                break :blk try ops.concat(&banks, -1);
            }
            var parts: [3]Arr = undefined;
            var at: c_int = 0;
            for (cluster.widths, 0..) |n, i| {
                parts[i] = try ops.binary(.mm, in.x, try ops.transpose(try ops.slice(in.cluster_w, 0, at, at + n), &.{ 1, 0 }));
                at += n;
            }
            break :blk try ops.concat(&parts, -1);
        },
    };
}

fn sampleNax(c: NaxCase, arm: usize) anyerror!f64 {
    transformer.vqmm_nax_probe_override = arm == 1;
    var ops = Ops{ .s = c.s };
    defer ops.deinit();
    const sw = io_util.Stopwatch.init(io());
    try eval(try naxOutput(&ops, c.in, c.case, arm == 1));
    return @floatFromInt(sw.read());
}

/// `[differ, relative L2]` of the NAX arm's float32 output against the staged arm's.
fn naxAgreement(s: mlx.mlx_stream, in: *const NaxInputs, case: usize) ![2]f64 {
    var outs: [2]Arr = undefined;
    var ops = Ops{ .s = s };
    defer ops.deinit();
    for (&outs, 0..) |*o, arm| {
        transformer.vqmm_nax_probe_override = arm == 1;
        o.* = try ops.contiguous(try ops.cast(try naxOutput(&ops, in, case, arm == 1), .float32));
        try eval(o.*);
    }
    const n = mlx.mlx_array_size(outs[0]);
    var differ: usize = 0;
    var num: f64 = 0;
    var den: f64 = 0;
    for (mlx.mlx_array_data_float32(outs[0]).?[0..n], mlx.mlx_array_data_float32(outs[1]).?[0..n]) |r, g| {
        if (@as(u32, @bitCast(r)) != @as(u32, @bitCast(g))) differ += 1;
        num += (g - r) * (g - r);
        den += r * r;
    }
    return .{ @floatFromInt(differ), @sqrt(num / den) };
}

pub const nax_arms: report.Bench = .{
    .name = "glm5.nax_arms",
    .scope = .glm5,
    .what = "NAX-shaped arms on this GPU's MLX kernels: staged against NAX, per case, with how far their outputs differ",
    .unit = .us,
    .steps = 1,
    .axes = &.{
        .{ .name = "case", .values = &nax_cases, .numeric = false },
    },
    .arms = &.{ "staged", "nax" },
    .ref = "staged",
    .run = runNaxArms,
};

fn runNaxArms(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    const gate = NaxGate.take();
    defer gate.restore();
    var fx = Ops{ .s = s };
    defer fx.deinit();
    const arr = glm.syntheticArray;
    const in: NaxInputs = .{
        .x = try arr(&fx, &.{ 1, 2048, 4096 }, .bfloat16, 1, -2, 2),
        .a6 = .{ try arr(&fx, &.{ 8192, 768 }, .uint32, 2, 0, 0), try arr(&fx, &.{ 8192, 32 }, .bfloat16, 3, 0.001, 0.01), try arr(&fx, &.{ 8192, 32 }, .bfloat16, 4, -0.3, 0.1) },
        .head = .{ try arr(&fx, &.{ 64, 256, 96 }, .uint32, 5, 0, 0), try arr(&fx, &.{ 64, 256, 4 }, .bfloat16, 6, 0.001, 0.01), try arr(&fx, &.{ 64, 256, 4 }, .bfloat16, 7, -0.3, 0.1) },
        .q = try arr(&fx, &.{ 2048, 64, 1, 256 }, .bfloat16, 8, -1, 1),
        .v = try arr(&fx, &.{ 2048, 64, 1, 512 }, .bfloat16, 9, -1, 1),
        .cluster_w = try arr(&fx, &.{ 320, 4096 }, .bfloat16, 10, -0.05, 0.05),
    };
    for (run.axis("case")) |name| {
        const case = indexOf(&nax_cases, name);
        const agree = try naxAgreement(s, &in, case);
        try run.interleave(name, NaxCase{ .s = s, .in = &in, .case = case }, sampleNax);
        try run.metric("nax", "differ", "values", agree[0]);
        try run.metric("nax", "rel L2", "e-3", agree[1] * 1e3);
    }
}

// ── glm5.kda_prefill ────────────────────────────────────────────────────

const kda_stage_names = [_][]const u8{
    "dequant_qkv",       "qkv_gemms",       "concat",      "dequant_o",      "o_gemm",
    "cluster_gemm",      "fb_gb_gemms",     "prework",     "recurrence",     "post_epilogue",
    "project_qkv_parts", "apply_reference", "gemm_n24576", "gemm_8192_peak", "prework_joined",
};

const KdaMeter = struct {
    layer: glm.KdaLayer,
    x: Arr,
    dq: [3]Arr,
    parts: [3]Arr,
    joined: Arr,
    dout: Arr,
    gated: Arr,
    cluster_banks: [3]Arr,
    raw_a: Arr,
    gate: Arr,
    prework: kda_prework.Result,
    state: Arr,
    y: Arr,
    cfg: model.ModelConfig,
    wide: Arr,
    big: Arr,

    fn run(self: *KdaMeter, ops: *Ops, id: usize) !void {
        switch (id) {
            0 => try evalAll(&.{
                try ops.dequant(self.layer.q.w, self.layer.q.scales, self.layer.q.biases),
                try ops.dequant(self.layer.k.w, self.layer.k.scales, self.layer.k.biases),
                try ops.dequant(self.layer.v.w, self.layer.v.scales, self.layer.v.biases),
            }),
            1 => {
                var out: [3]Arr = undefined;
                for (self.dq, 0..) |w, i| out[i] = try ops.binary(.mm, self.x, try ops.transpose(w, &.{ 1, 0 }));
                try evalAll(&out);
            },
            2 => try eval(try ops.concat(&self.parts, -1)),
            3 => try eval(try ops.dequant(self.layer.out.w, self.layer.out.scales, self.layer.out.biases)),
            4 => try eval(try ops.binary(.mm, self.gated, try ops.transpose(self.dout, &.{ 1, 0 }))),
            5 => {
                const banks = (try self.layer.prefillCluster(ops, self.x)).?;
                try evalAll(&banks);
            },
            6 => try evalAll(&.{ try self.layer.fb.apply(ops, self.cluster_banks[0]), try self.layer.gb.apply(ops, self.cluster_banks[1]) }),
            7, 14 => {
                const split = id == 7;
                const r = (try kda_prework.apply(ops.s, .{ .qkv = if (split) .{ .ctx = null } else try ops.concat(&self.parts, -1), .parts = if (split) self.parts else null, .a = self.raw_a, .beta = self.cluster_banks[2], .conv_weight = self.layer.prepared_conv, .exp_a = self.layer.prepared_decay, .dt_bias = self.layer.dt_bias, .conv_state = null, .heads = 64, .lower = self.cfg.kda_gate_lower_bound })).?;
                defer r.deinit();
                try evalAll(&r.arrays());
            },
            8 => {
                const r = (try @import("glm5/glm5_kda_value_rows.zig").run(.{ .q = self.prework.q, .k = self.prework.k, .v = self.prework.v, .decay = self.prework.decay, .beta = self.prework.beta, .state = self.state }, 4, ops.s)).?;
                defer r.deinit();
                try evalAll(&.{ r.y, r.state });
            },
            9 => try eval((try @import("glm5/glm5_kda_fused.zig").post(ops.s, self.y, self.gate, self.layer.out_norm, self.cfg.rms_norm_eps)).?),
            10 => try evalAll(&(try self.layer.projectQkvParts(ops, self.x))),
            12 => try eval(try ops.binary(.mm, self.x, try ops.transpose(self.wide, &.{ 1, 0 }))),
            13 => try eval(try ops.binary(.mm, self.big, try ops.transpose(self.big, &.{ 1, 0 }))),
            else => {
                var cache = transformer.SSMCacheEntry{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
                defer _ = mlx.mlx_array_free(cache.conv_state);
                defer _ = mlx.mlx_array_free(cache.ssm_state);
                const out = try self.layer.applyReference(ops, self.x, &self.cfg, &cache);
                try evalAll(&.{ out, cache.conv_state, cache.ssm_state });
            },
        }
    }
};

const KdaStage = struct { s: mlx.mlx_stream, m: *KdaMeter, id: usize };

fn sampleKda(c: KdaStage, _: usize) anyerror!f64 {
    var ops = Ops{ .s = c.s };
    defer ops.deinit();
    const sw = io_util.Stopwatch.init(io());
    try c.m.run(&ops, c.id);
    return @floatFromInt(sw.read());
}

pub const kda_prefill: report.Bench = .{
    .name = "glm5.kda_prefill",
    .scope = .glm5,
    .what = "one synthetic T2048 KDA layer: each stage queued with one eval, plus the whole reference layer and two GEMM probes",
    .unit = .us,
    .steps = 1,
    .axes = &.{
        .{ .name = "stage", .values = &kda_stage_names, .numeric = false },
    },
    .arms = &.{"stage"},
    .ref = "stage",
    .run = runKdaPrefill,
};

fn runKdaPrefill(run: *report.Run) anyerror!void {
    if (!glm.naxArms()) return error.NeedsNaxGpu;
    const s = mlx.gpuStream();
    var fx = Ops{ .s = s };
    defer fx.deinit();
    const arr = glm.syntheticArray;
    const bank = struct {
        fn make(o: *Ops, n: c_int, k: c_int, seed: u64) !Linear {
            return .{
                .w = try glm.syntheticArray(o, &.{ n, @divExact(k * 3, 16) }, .uint32, seed, 0, 0),
                .scales = try glm.syntheticArray(o, &.{ n, @divExact(k, 128) }, .bfloat16, seed + 1, 0.0005, 0.004),
                .biases = try glm.syntheticArray(o, &.{ n, @divExact(k, 128) }, .bfloat16, seed + 2, -0.12, 0.04),
                .input = k,
                .output = n,
            };
        }
        fn dense(o: *Ops, n: c_int, k: c_int, seed: u64) !Linear {
            return .{ .w = try glm.syntheticArray(o, &.{ n, k }, .bfloat16, seed, -0.05, 0.05), .input = k, .output = n };
        }
    };
    var layer = glm.KdaLayer{
        .q = try bank.make(&fx, 8192, 4096, 10),
        .k = try bank.make(&fx, 8192, 4096, 20),
        .v = try bank.make(&fx, 8192, 4096, 30),
        .fa = try bank.dense(&fx, 128, 4096, 40),
        .fb = try bank.dense(&fx, 8192, 128, 50),
        .ga = try bank.dense(&fx, 128, 4096, 60),
        .gb = try bank.dense(&fx, 8192, 128, 70),
        .beta = try bank.dense(&fx, 64, 4096, 80),
        .out = try bank.make(&fx, 4096, 8192, 90),
        .conv_q = try arr(&fx, &.{ 8192, 1, 4 }, .bfloat16, 100, -0.5, 0.5),
        .conv_k = try arr(&fx, &.{ 8192, 1, 4 }, .bfloat16, 101, -0.5, 0.5),
        .conv_v = try arr(&fx, &.{ 8192, 1, 4 }, .bfloat16, 102, -0.5, 0.5),
        .a_log = try arr(&fx, &.{64}, .float32, 103, -1, 1),
        .dt_bias = try arr(&fx, &.{8192}, .float32, 104, -1, 1),
        .out_norm = try arr(&fx, &.{128}, .bfloat16, 105, 0.5, 1.5),
    };
    defer layer.deinit();
    try layer.prepare(s);
    try layer.preparePrefillCluster(s);
    const cfg = model.ModelConfig{ .hidden_size = 4096, .linear_key_head_dim = 128, .linear_num_value_heads = 64, .linear_conv_kernel_dim = 4, .kda_gate_lower_bound = -5, .rms_norm_eps = 1e-5 };
    const x = try arr(&fx, &.{ 1, 2048, 4096 }, .bfloat16, 1, -1, 1);
    var m: KdaMeter = undefined;
    m.layer = layer;
    m.x = x;
    m.cfg = cfg;
    for ([_]*const Linear{ &layer.q, &layer.k, &layer.v }, 0..) |l, i| {
        m.dq[i] = try fx.dequant(l.w, l.scales, l.biases);
        m.parts[i] = try fx.binary(.mm, x, try fx.transpose(m.dq[i], &.{ 1, 0 }));
    }
    m.joined = try fx.concat(&m.parts, -1);
    m.wide = try fx.concat(&m.dq, 0);
    m.big = try arr(&fx, &.{ 8192, 8192 }, .bfloat16, 7, -1, 1);
    try eval(m.wide);
    m.dout = try fx.dequant(layer.out.w, layer.out.scales, layer.out.biases);
    m.cluster_banks = (try layer.prefillCluster(&fx, x)).?;
    m.raw_a = try layer.fb.apply(&fx, m.cluster_banks[0]);
    m.gate = try fx.reshape(try layer.gb.apply(&fx, m.cluster_banks[1]), &.{ 1, 2048, 64, 128 });
    m.state = try fx.zeros(&.{ 1, 64, 128, 128 }, .float32);
    m.prework = (try kda_prework.apply(s, .{ .qkv = m.joined, .a = m.raw_a, .beta = m.cluster_banks[2], .conv_weight = layer.prepared_conv, .exp_a = layer.prepared_decay, .dt_bias = layer.dt_bias, .heads = 64 })).?;
    defer m.prework.deinit();
    const rec = (try @import("glm5/glm5_kda_value_rows.zig").run(.{ .q = m.prework.q, .k = m.prework.k, .v = m.prework.v, .decay = m.prework.decay, .beta = m.prework.beta, .state = m.state }, 4, s)).?;
    defer rec.deinit();
    m.y = rec.y;
    m.gated = try fx.reshape((try @import("glm5/glm5_kda_fused.zig").post(s, m.y, m.gate, layer.out_norm, cfg.rms_norm_eps)).?, &.{ 1, 2048, 8192 });
    try evalAll(&.{ m.joined, m.dout, m.raw_a, m.gate, m.state, m.y, m.gated, m.cluster_banks[0], m.cluster_banks[1], m.cluster_banks[2], m.prework.q, m.prework.k, m.prework.v, m.prework.decay, m.prework.beta });

    for (run.axis("stage")) |name| try run.interleave(name, KdaStage{ .s = s, .m = &m, .id = indexOf(&kda_stage_names, name) }, sampleKda);
}

// ── MLA attention state shared by the three benches below ───────────────

/// A kv8 latent state of `n` rows, built as serving builds it: appended in chunks of at most 16384.
fn buildState(scope: *Scope, n: c_int, s: mlx.mlx_stream) !attention.State {
    var state = attention.State{ .latent_bits = latent_store.kv8_bits };
    errdefer state.deinit();
    const ape = try scope.zeros(&.{ 4, 128 }, .bfloat16);
    var filled: c_int = 0;
    while (filled < n) {
        const rows = @min(n - filled, 16384);
        var part = Scope{ .s = s };
        defer part.deinit();
        const seed: u64 = @intCast(filled);
        _ = try state.append(try attention.normal(&part, &.{ rows, 512 }, seed + 1, 1), try attention.normal(&part, &.{ rows, 128 }, seed + 2, 1), try attention.normal(&part, &.{ rows, 128 }, seed + 3, 1), ape, s);
        try state.evaluate();
        filled += rows;
    }
    return state;
}

const sparse_window = 2051;

// ── glm5.mla_decode_attn ────────────────────────────────────────────────

const DecodeStage = enum { null_chain, scores, select, composite_cast, composite_qk, composite_softmax, composite, full };

/// The FP32 composite of one decode row, cut after `stage`.
fn decodeComposite(ops: *Ops, q: Arr, kv_bank: Arr, mask: Arr, scale: f32, stage: DecodeStage) !Arr {
    const kv = try ops.cast(kv_bank, .float32);
    if (stage == .composite_cast) return kv;
    const queries = try ops.binary(.mul, try ops.cast(q, .float32), try ops.scalar(scale, .float32));
    const kt = try ops.transpose(kv, &.{ 0, 2, 1 });
    const blocks = try ops.ones(&.{ 1, 2, 65 }, .bool_);
    const qk = try ops.slot();
    try mlx.check(mlx.mlx_block_masked_mm(qk, queries, kt, 32, blocks, .{ .ctx = null }, .{ .ctx = null }, ops.s));
    const masked = try ops.slot();
    try mlx.check(mlx.mlx_where(masked, try ops.reshape(mask, &.{ 1, 1, sparse_window }), qk.*, try ops.scalar(-std.math.inf(f32), .float32), ops.s));
    if (stage == .composite_qk) return masked.*;
    const probs = try ops.softmax(masked.*, -1);
    if (stage == .composite_softmax) return probs;
    const blocks2 = try ops.ones(&.{ 1, 2, 16 }, .bool_);
    const pv = try ops.slot();
    try mlx.check(mlx.mlx_block_masked_mm(pv, probs, kv, 32, blocks2, .{ .ctx = null }, .{ .ctx = null }, ops.s));
    return ops.cast(pv.*, .bfloat16);
}

const DecodeCase = struct {
    s: mlx.mlx_stream,
    state: *const attention.State,
    q0: Arr,
    bank: Arr,
    mask: Arr,
    pos: usize,
    layers: usize,
};

/// Nanoseconds per layer of a dependent chain: each layer's query and index inputs come from the previous output, and
/// the cumulative stage up to `arm` runs on them.
fn sampleDecode(c: DecodeCase, arm: usize) anyerror!f64 {
    const stage: DecodeStage = @fromBackingInt(@intCast(arm));
    const alloc = std.heap.page_allocator;
    const opses = try alloc.alloc(Ops, c.layers);
    defer alloc.free(opses);
    const scopes = try alloc.alloc(Scope, c.layers);
    defer alloc.free(scopes);
    for (opses, scopes) |*o, *sc| {
        o.* = Ops{ .s = c.s };
        sc.* = Scope{ .s = c.s };
    }
    defer for (opses, scopes) |*o, *sc| {
        o.deinit();
        sc.deinit();
    };
    var q = c.q0;
    const sw = io_util.Stopwatch.init(io());
    for (opses, scopes) |*ops, *sc| {
        const flat = try ops.reshape(q, &.{32768});
        const iq = try ops.reshape(try ops.slice(flat, 0, 0, 4096), &.{ 1, 32, 128 });
        const w = try ops.reshape(try ops.slice(flat, 0, 4096, 4128), &.{ 1, 32 });
        var node = c.state.*;
        node.processed = c.pos + 1;
        const product: Arr = switch (stage) {
            .null_chain => q,
            .scores => try attention.indexScores(sc, &node, iq, w, c.pos),
            .select => try ops.own(try attention.decodeSelected(c.state, iq, w, c.pos, c.s)),
            .composite_cast, .composite_qk, .composite_softmax => try decodeComposite(ops, q, c.bank, c.mask, 1.0 / 16.0, stage),
            .composite => try packed_nax.compositeWith(ops, q, c.bank, c.mask, 1.0 / 16.0, .block_masked),
            .full => try ops.own(try attention.attend(c.state, q, iq, w, c.pos, 1.0 / 16.0, c.s)),
        };
        const first = try ops.cast(try ops.reshape(try ops.slice(try ops.reshape(product, &.{-1}), 0, 0, 1), &.{1}), .float32);
        const token_value = try ops.cast(try ops.binary(.mul, first, try ops.scalar(0, .float32)), .bfloat16);
        q = try ops.binary(.add, q, token_value);
    }
    try eval(q);
    return nsPer(sw, c.layers);
}

pub const mla_decode_attn: report.Bench = .{
    .name = "glm5.mla_decode_attn",
    .scope = .glm5,
    .what = "one serial decode row's MLA attention as a dependent layer chain, cumulative stages (null_chain is the chain's own plumbing)",
    .unit = .us,
    .steps = 16,
    .axes = &.{
        .{ .name = "ctx", .values = &.{ "1024", "8192", "32768", "131072" } },
    },
    .arms = &.{ "null_chain", "scores", "select", "composite_cast", "composite_qk", "composite_softmax", "composite", "full" },
    .ref = "null_chain",
    .run = runMlaDecode,
};

fn runMlaDecode(run: *report.Run) anyerror!void {
    if (!glm.naxArms()) return error.NeedsNaxGpu;
    const s = mlx.gpuStream();
    for (run.axis("ctx")) |ctx_text| {
        const n = try parseCount(ctx_text);
        var scope = Scope{ .s = s };
        defer scope.deinit();
        var state = try buildState(&scope, n, s);
        defer state.deinit();
        const q0 = try attention.normal(&scope, &.{ 1, 64, 512 }, 21, 0.05);
        const bank = try attention.normal(&scope, &.{ 1, sparse_window, 512 }, 22, 1);
        const mask = try scope.slot();
        try mlx.check(mlx.mlx_ones(mask, &.{ 1, sparse_window }, 2, .bool_, s));
        try evalAll(&.{ q0, bank, mask.* });
        const c: DecodeCase = .{ .s = s, .state = &state, .q0 = q0, .bank = bank, .mask = mask.*, .pos = @intCast(n - 1), .layers = run.steps(16) };
        try run.interleave(ctx_text, c, sampleDecode);
    }
}

// ── glm5.mla_chunk_attn ─────────────────────────────────────────────────

const ChunkCase = struct {
    s: mlx.mlx_stream,
    state: *const attention.State,
    q: Arr,
    iq: Arr,
    w: Arr,
    selected: Arr,
    offset: usize,
    rows: c_int,
    gate: NaxGate,
};

/// Whole sparse attention of one short chunk: the scalar kernel, the packed head arm with MLX's FP32 composite or with
/// fused NAX, and the served dispatch. Scalar and packed read the same synthetic window of 2051 latent rows per row.
fn sampleChunk(c: ChunkCase, arm: usize) anyerror!f64 {
    var arm_scope = Scope{ .s = c.s };
    defer arm_scope.deinit();
    var ops = Ops{ .s = c.s };
    defer ops.deinit();
    transformer.vqmm_nax_probe_override = if (arm == 1) false else c.gate.saved;
    const history = c.state.processed;
    const sw = io_util.Stopwatch.init(io());
    const out = switch (arm) {
        0 => try attention.attentionChunk(&arm_scope, c.state, c.q, c.selected, c.offset, 1.0 / 16.0, if (c.rows == 1) 8 else 1, false, null),
        1, 2 => (try packed_nax.run(&ops, c.q, c.state.latentView(), c.selected, c.offset, history, 1.0 / 16.0)).?,
        else => try arm_scope.own(try attention.attend(c.state, c.q, c.iq, c.w, c.offset, 1.0 / 16.0, c.s)),
    };
    try eval(out);
    return @floatFromInt(sw.read());
}

pub const mla_chunk_attn: report.Bench = .{
    .name = "glm5.mla_chunk_attn",
    .scope = .glm5,
    .what = "whole sparse MLA attention of one short chunk on a kv8 state: scalar kernel, packed FP32 composite, packed fused NAX, the served dispatch",
    .unit = .us,
    .steps = 1,
    .axes = &.{
        .{ .name = "ctx", .values = &.{"16384"} },
        .{ .name = "rows", .values = &.{ "1", "8" } },
    },
    .arms = &.{ "scalar", "composite", "nax", "served" },
    .ref = "scalar",
    .run = runMlaChunk,
};

fn runMlaChunk(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    if (run.hasArm("nax") and !glm.naxArms()) return error.NaxArmNeedsNaxGpu;
    const gate = NaxGate.take();
    defer gate.restore();
    const a = std.heap.page_allocator;
    for (run.axis("ctx")) |ctx_text| {
        const n = try parseCount(ctx_text);
        if (n <= sparse_window + 8) return error.ContextInsideDenseWindow;
        var scope = Scope{ .s = s };
        defer scope.deinit();
        var state = try buildState(&scope, n, s);
        defer state.deinit();
        for (run.axis("rows")) |rows_text| {
            const rows = try parseCount(rows_text);
            if (rows > packed_nax.composite_rows) return error.RowsAbovePackedComposite;
            const offset: usize = @intCast(n - rows);
            const ids = try a.alloc(i32, @as(usize, @intCast(rows)) * sparse_window);
            defer a.free(ids);
            for (0..@intCast(rows)) |r| for (0..sparse_window) |k| {
                ids[r * sparse_window + k] = if (k % 37 == 5) -1 else @intCast(offset + r - 2050 + k);
            };
            const selected = try scope.own(mlx.mlx_array_new_data(ids.ptr, &.{ rows, sparse_window }, 2, .int32));
            const q = try attention.normal(&scope, &.{ rows, 64, 512 }, 72 + @as(u64, @intCast(rows)), 2);
            const iq = try attention.normal(&scope, &.{ rows, 32, 128 }, 32, 1);
            const w = try attention.normal(&scope, &.{ rows, 32 }, 33, 0.2);
            try evalAll(&.{ q, iq, w });
            const label = try std.fmt.allocPrint(run.a, "ctx {d} rows {d}", .{ n, rows });
            try run.interleave(label, ChunkCase{ .s = s, .state = &state, .q = q, .iq = iq, .w = w, .selected = selected, .offset = offset, .rows = rows, .gate = gate }, sampleChunk);
        }
    }
}

// ── glm5.mla_chunk_stages ───────────────────────────────────────────────

const Stage = enum { scores, select, attend };

/// One stage of a chunk's sparse attention over `rows` query rows, in packed tiles of at most `packed_nax.max_rows`.
fn runStage(stage: Stage, state: *const attention.State, q: Arr, iq: Arr, w: Arr, offset: usize, s: mlx.mlx_stream) !void {
    const rows: usize = @intCast(mlx.getShape(q)[0]);
    if (stage == .attend) {
        const out = try attention.attend(state, q, iq, w, offset, 1.0 / 16.0, s);
        defer _ = mlx.mlx_array_free(out);
        try eval(out);
        return;
    }
    var start: usize = 0;
    while (start < rows) {
        var scope = Scope{ .s = s };
        defer scope.deinit();
        const outputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs);
        for (0..2) |_| {
            if (start == rows) break;
            const end = @min(rows, start + packed_nax.max_rows);
            var node = state.*;
            if (rows == 1) node.processed = offset + 1;
            const iqc = try scope.cut(iq, @intCast(start), @intCast(end));
            const wc = try scope.cut(w, @intCast(start), @intCast(end));
            const out = if (stage == .scores) try attention.indexScores(&scope, &node, iqc, wc, offset + start) else try attention.selectChunk(&scope, &node, iqc, wc, offset + start);
            try mlx.check(mlx.mlx_vector_array_append_value(outputs, out));
            start = end;
        }
        try mlx.check(mlx.mlx_eval(outputs));
    }
}

const StageCase = struct { s: mlx.mlx_stream, state: *const attention.State, q: Arr, iq: Arr, w: Arr, offset: usize };

fn sampleStage(c: StageCase, arm: usize) anyerror!f64 {
    const sw = io_util.Stopwatch.init(io());
    try runStage(@fromBackingInt(@intCast(arm)), c.state, c.q, c.iq, c.w, c.offset, c.s);
    return @floatFromInt(sw.read());
}

pub const mla_chunk_stages: report.Bench = .{
    .name = "glm5.mla_chunk_stages",
    .scope = .glm5,
    .what = "a long chunk's sparse MLA attention split into index scores, top-512 selection and the whole attention, on a kv8 state",
    .unit = .ms,
    .steps = 1,
    .axes = &.{
        .{ .name = "ctx", .values = &.{ "8192", "32768", "131072" } },
        .{ .name = "rows", .values = &.{"2048"} },
    },
    .arms = &.{ "scores", "select", "attend" },
    .ref = "attend",
    .run = runMlaStages,
};

fn runMlaStages(run: *report.Run) anyerror!void {
    if (!glm.naxArms()) return error.NeedsNaxGpu;
    const s = mlx.gpuStream();
    for (run.axis("ctx")) |ctx_text| {
        const n = try parseCount(ctx_text);
        var scope = Scope{ .s = s };
        defer scope.deinit();
        var state = try buildState(&scope, n, s);
        defer state.deinit();
        for (run.axis("rows")) |rows_text| {
            const rows = try parseCount(rows_text);
            if (n <= sparse_window + rows) return error.ContextInsideDenseWindow;
            const q = try attention.normal(&scope, &.{ rows, 64, 512 }, 31, 0.05);
            const iq = try attention.normal(&scope, &.{ rows, 32, 128 }, 32, 1);
            const w = try attention.normal(&scope, &.{ rows, 32 }, 33, 0.2);
            try evalAll(&.{ q, iq, w });
            const label = try std.fmt.allocPrint(run.a, "ctx {d} rows {d}", .{ n, rows });
            try run.interleave(label, StageCase{ .s = s, .state = &state, .q = q, .iq = iq, .w = w, .offset = @intCast(n - rows) }, sampleStage);
        }
    }
}
