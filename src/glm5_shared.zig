//! One-token GLM shared expert on resident affine5/6/8 g128 banks: gate, up and the clamped BF16 SwiGLU in one
//! dispatch. Each projection keeps MLX qmv_fast's order (`glm5_decode.zig`) and the activation keeps
//! `glm5_activation.zig`'s BF16 rounding boundaries.
const std = @import("std");
const mlx = @import("mlx.zig");
const decode = @import("glm5_decode.zig");
const api = @import("sushi_exl3");
const Arr = mlx.mlx_array;
const Bank = decode.Bank;

/// The clamped SwiGLU of the BF16 `g` and `u` of row `r0 + lane`.
const ACTIVATE: [:0]const u8 =
    \\  g=g>10.0f?10.0f:g;
    \\  u=u>10.0f?10.0f:u;
    \\  u=u<-10.0f?-10.0f:u;
    \\  bfloat16_t cg=bfloat16_t(g);
    \\  bfloat16_t mid=bfloat16_t(g*float(sigmoid[as_type<ushort>(cg)]));
    \\  y[yoff+r0+lane]=bfloat16_t(float(mid)*u);
    \\
;

const GATE_UP: [:0]const u8 =
    \\const int yoff=0;
    \\const uint lane=thread_index_in_simdgroup;
    \\const int which=int(simdgroup_index_in_threadgroup);
    \\const int r0=int(threadgroup_position_in_grid.x)*4;
    \\const device uint8_t* w=(const device uint8_t*)(which==0?wg:wu);
    \\const device T* sc=which==0?sg:su;
    \\const device T* bi=which==0?bg:bu;
    \\float result[4]={0.0f,0.0f,0.0f,0.0f};
    \\glm_qmv_rows<T,K>(w+size_t(r0)*(K*GLM_BITS/8),sc+r0*(K/128),bi+r0*(K/128),x,lane,result);
    \\threadgroup float parts[8];
    \\for(int r=0;r<4;++r) {float v=simd_sum(result[r]);if(lane==0) parts[which*4+r]=float(static_cast<T>(v));}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if(which==0&&lane<4) {
    \\  float g=parts[lane];float u=parts[4+lane];
++ ACTIVATE ++
    \\}
;

/// `GATE_UP` for a launch of 128 threads (the fold's grid): two row groups of four per threadgroup, each the same
/// gate simdgroup, up simdgroup and activation as `GATE_UP`'s one, for the input row of grid y (`x` is BF16 [rows, K],
/// `y` [rows, N]). Each row is the serial one-token kernel's, which is also what the rows kernels' bytes are.
const GATE_UP_128: [:0]const u8 =
    \\const uint lane=thread_index_in_simdgroup;
    \\const int row=int(threadgroup_position_in_grid.y);
    \\const int yoff=row*N;
    \\const device bfloat16_t* xr=x+size_t(row)*K;
    \\const int sgi=int(simdgroup_index_in_threadgroup);
    \\const int which=sgi&1;
    \\const int part=(sgi>>1)*8;
    \\const int r0=int(threadgroup_position_in_grid.x)*8+(sgi>>1)*4;
    \\const device uint8_t* w=(const device uint8_t*)(which==0?wg:wu);
    \\const device bfloat16_t* sc=which==0?sg:su;
    \\const device bfloat16_t* bi=which==0?bg:bu;
    \\float result[4]={0.0f,0.0f,0.0f,0.0f};
    \\glm_qmv_rows<bfloat16_t,K>(w+size_t(r0)*(K*GLM_BITS/8),sc+r0*(K/128),bi+r0*(K/128),xr,lane,result);
    \\threadgroup float parts[16];
    \\for(int r=0;r<4;++r) {float v=simd_sum(result[r]);if(lane==0) parts[part+which*4+r]=float(static_cast<bfloat16_t>(v));}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if(which==0&&lane<4) {
    \\  float g=parts[part+lane];float u=parts[part+4+lane];
++ ACTIVATE ++
    \\}
;

/// One affine projection of each row of a BF16 [rows, K] for a launch of 128 threads: a simdgroup per four outputs, a
/// grid row per input row, in qmv_fast's order; `y` is [rows, N].
const DOWN_128: [:0]const u8 =
    \\const uint lane=thread_index_in_simdgroup;
    \\const int row=int(threadgroup_position_in_grid.y);
    \\const int r0=int(threadgroup_position_in_grid.x)*16+int(simdgroup_index_in_threadgroup)*4;
    \\const device bfloat16_t* xr=x+size_t(row)*K;
    \\float result[4]={0.0f,0.0f,0.0f,0.0f};
    \\glm_qmv_rows<bfloat16_t,K>((const device uint8_t*)w+size_t(r0)*(K*GLM_BITS/8),sc+r0*(K/128),bi+r0*(K/128),xr,lane,result);
    \\for(int r=0;r<4;++r) {float v=simd_sum(result[r]);if(lane==0) y[row*N+r0+r]=static_cast<bfloat16_t>(v);}
;

var kernels: [3]?mlx.mlx_fast_metal_kernel = @splat(null);
var dispatches: usize = 0;
pub fn gateUpDispatches() usize {
    return dispatches;
}

fn kernel(bits: c_int) !mlx.mlx_fast_metal_kernel {
    const slot = &kernels[if (bits == 8) 2 else @as(usize, @intCast(bits - 5))];
    if (slot.*) |k| return k;
    const names = [_][*:0]const u8{ "x", "wg", "sg", "bg", "wu", "su", "bu", "sigmoid" };
    const iv = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&[_][*:0]const u8{"y"}, 1);
    defer _ = mlx.mlx_vector_string_free(ov);
    const label: [*:0]const u8 = switch (bits) {
        5 => "sushi_glm_shared_gate_up_a5",
        6 => "sushi_glm_shared_gate_up_a6",
        else => "sushi_glm_shared_gate_up_a8",
    };
    const k = mlx.mlx_fast_metal_kernel_new(label, iv, ov, GATE_UP, decode.header(bits), true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

const Key = struct { width: c_int, rows: c_int, rank: usize };
const Entry = struct { key: Key, cfg: mlx.mlx_fast_metal_kernel_config };
var cache: [8]?Entry = @splat(null);

/// A cached launch geometry, or a fresh one the caller frees.
fn configuration(key: Key) !struct { value: mlx.mlx_fast_metal_kernel_config, cached: bool } {
    for (cache) |entry| if (entry) |e| {
        if (std.meta.eql(e.key, key)) return .{ .value = e.cfg, .cached = true };
    };
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var out: [3]c_int = @splat(1);
    out[key.rank - 1] = key.rows;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out, key.rank, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(key.rows, 4) * 64, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", key.width));
    for (&cache) |*entry| if (entry.* == null) {
        entry.* = .{ .key = key, .cfg = cfg };
        return .{ .value = cfg, .cached = true };
    };
    return .{ .value = cfg, .cached = false };
}

/// `clamp(gate(x), 10) silu * clamp(up(x), ±10)` for a one-row BF16 input, or null when the input, banks or limit
/// are not served.
pub fn gateUp(s: mlx.mlx_stream, x: Arr, gate: Bank, up: Bank, limit: f32, sigmoid: Arr) !?Arr {
    if (!mlx.streamIsGpu(s) or limit != 10 or x.ctx == null or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const sh = mlx.getShape(x);
    if (sh.len < 2 or sh.len > 3) return null;
    const width = sh[sh.len - 1];
    if (width <= 0 or @rem(width, 256) != 0 or mlx.mlx_array_size(x) != @as(usize, @intCast(width)) or mlx.mlx_array_strides(x)[sh.len - 1] != 1) return null;
    const g = (try decode.checkBank(width, gate)) orelse return null;
    const u = (try decode.checkBank(width, up)) orelse return null;
    if (g.bits != u.bits or g.rows != u.rows) return null;
    const handle = try configuration(.{ .width = width, .rows = g.rows, .rank = sh.len });
    defer if (!handle.cached) {
        _ = mlx.mlx_fast_metal_kernel_config_free(handle.value);
    };
    const inputs = [_]Arr{ x, gate.weight, gate.scales, gate.biases, up.weight, up.scales, up.biases, sigmoid };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try kernel(g.bits), iv, handle.value, s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    dispatches += 1;
    decode.logEngaged(.shared, g.bits);
    return result;
}

/// The routed layer output with this shared expert inside the routed lane chain's launches (`glm_fold.zig`), for one
/// decode row or two to eight verify rows, or null where the shared expert runs as its own kernels.
pub fn folded(s: mlx.mlx_stream, x: Arr, gate: Bank, up: Bank, down: Bank, routed: api.Bank, ids: Arr, scores: Arr, dec: api.format.Decode, limit: f32, sigmoid: Arr) !?Arr {
    if (limit != 10) return null;
    const sh = mlx.getShape(x);
    if (x.ctx == null or sh.len != 3) return null;
    const width = sh[2];
    const g = (try decode.checkBank(width, gate)) orelse return null;
    const u = (try decode.checkBank(width, up)) orelse return null;
    if (g.bits != u.bits or g.rows != u.rows or @rem(g.rows, 256) != 0) return null;
    const d = (try decode.checkBank(g.rows, down)) orelse return null;
    if (d.rows != width) return null;
    const shared: api.glm_fold.Shared = .{
        .gate = .{ .weight = gate.weight, .scales = gate.scales, .biases = gate.biases },
        .up = .{ .weight = up.weight, .scales = up.scales, .biases = up.biases },
        .down = .{ .weight = down.weight, .scales = down.scales, .biases = down.biases },
        .sigmoid = sigmoid,
        .gate_up_body = GATE_UP_128,
        .gate_up_header = decode.header(g.bits),
        .gate_up_bits = g.bits,
        .down_body = DOWN_128,
        .down_header = decode.header(d.bits),
        .down_bits = d.bits,
    };
    const result = (if (sh[1] == 1)
        try api.glm_fold.moeClampedSharedFolded(s, x, routed, ids, scores, dec, 10, shared)
    else
        try api.glm_fold.moeLayoutSharedFolded(s, x, routed, ids, scores, dec, shared)) orelse return null;
    decode.logEngaged(.shared, g.bits);
    return result;
}

pub const Fixture = struct {
    arrays: [18]Arr = undefined,
    made: usize = 0,
    pub fn deinit(self: *Fixture) void {
        for (self.arrays[0..self.made]) |v| {
            _ = mlx.mlx_array_free(v);
        }
    }
    pub fn bf16(v: f32) u16 {
        return @truncate(@as(u32, @bitCast(v)) >> 16);
    }
    pub fn bank(self: *Fixture, rnd: std.Random, rows: usize, k: usize, bits: usize, gain: f32) !Bank {
        const a = std.testing.allocator;
        const codes = try a.alloc(u32, rows * k * bits / 32);
        defer a.free(codes);
        const scales = try a.alloc(u16, rows * k / 128);
        defer a.free(scales);
        const biases = try a.alloc(u16, scales.len);
        defer a.free(biases);
        for (codes) |*v| v.* = rnd.int(u32);
        for (scales, biases) |*sc, *b| {
            sc.* = bf16(gain * (rnd.float(f32) + 0.125) / 128);
            b.* = bf16(gain * (rnd.float(f32) - 0.5));
        }
        const shape = [_]c_int{ @intCast(rows), @intCast(k * bits / 32) };
        const grid = [_]c_int{ @intCast(rows), @intCast(k / 128) };
        self.arrays[self.made] = mlx.mlx_array_new_data(codes.ptr, &shape, 2, .uint32);
        self.made += 1;
        self.arrays[self.made] = mlx.mlx_array_new_data(scales.ptr, &grid, 2, .bfloat16);
        self.made += 1;
        self.arrays[self.made] = mlx.mlx_array_new_data(biases.ptr, &grid, 2, .bfloat16);
        self.made += 1;
        return .{ .weight = self.arrays[self.made - 3], .scales = self.arrays[self.made - 2], .biases = self.arrays[self.made - 1] };
    }
};

pub fn randomRow(rnd: std.Random, n: usize, spread: f32) !Arr {
    const a = std.testing.allocator;
    const values = try a.alloc(u16, n);
    defer a.free(values);
    for (values) |*v| v.* = Fixture.bf16((rnd.float(f32) - 0.5) * spread);
    return mlx.mlx_array_new_data(values.ptr, &[_]c_int{ 1, 1, @intCast(n) }, 3, .bfloat16);
}

pub fn affine(s: mlx.mlx_stream, x: Arr, bank: Bank, bits: usize) !Arr {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_quantized_matmul(&out, x, bank.weight, bank.scales, bank.biases, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(@intCast(bits)), "affine", s));
    return out;
}

pub fn expectSameBits(want: Arr, got: Arr) !void {
    try mlx.check(mlx.mlx_array_eval(want));
    try mlx.check(mlx.mlx_array_eval(got));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    const n = mlx.mlx_array_size(got);
    try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(want).?[0..n], mlx.mlx_array_data_bfloat16(got).?[0..n]);
}

test "GLM shared expert gate/up/activation is bit exact to the staged chain at every stored width" {
    const Ops = @import("glm5_model.zig").Ops;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(731);
    const rnd = random.random();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const sigmoid = try @import("hc_prefill.zig").sigmoidTable(s);
    // The large gain pushes gates and ups through both clamps; the small one keeps most rows inside.
    for ([_]usize{ 6, 5, 8 }) |bits| for ([_]f32{ 1, 0.02 }) |gain| {
        var fx = Fixture{};
        defer fx.deinit();
        const gate = try fx.bank(rnd, 2048, 4096, bits, gain);
        const up = try fx.bank(rnd, 2048, 4096, bits, gain);
        const x = try ops.own(try randomRow(rnd, 4096, 4));
        const before = gateUpDispatches();
        const got = try ops.own((try gateUp(s, x, gate, up, 10, sigmoid)) orelse return error.TestExpectedFusedSharedGateUp);
        try std.testing.expectEqual(before + 1, gateUpDispatches());
        const g = try ops.own(try affine(s, x, gate, bits));
        const u = try ops.own(try affine(s, x, up, bits));
        const hi = try ops.scalar(10, .bfloat16);
        const lo = try ops.scalar(-10, .bfloat16);
        const clipped = try ops.binary(.max, try ops.binary(.min, u, hi), lo);
        const want = try ops.binary(.mul, try ops.silu(try ops.binary(.min, g, hi)), clipped);
        try expectSameBits(want, got);
    };
}

test "GLM shared expert fusion declines what it cannot reproduce" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(733);
    const rnd = random.random();
    var fx = Fixture{};
    defer fx.deinit();
    const bank = try fx.bank(rnd, 16, 256, 6, 1);
    const sigmoid = try @import("hc_prefill.zig").sigmoidTable(s);
    const x = try randomRow(rnd, 256, 2);
    defer _ = mlx.mlx_array_free(x);
    try std.testing.expect((try gateUp(s, x, bank, bank, 9, sigmoid)) == null);
    const wide = try randomRow(rnd, 128, 2);
    defer _ = mlx.mlx_array_free(wide);
    try std.testing.expect((try gateUp(s, wide, bank, bank, 10, sigmoid)) == null);
    const dense = Bank{ .weight = x, .scales = .{ .ctx = null }, .biases = .{ .ctx = null } };
    try std.testing.expect((try gateUp(s, x, dense, dense, 10, sigmoid)) == null);
}

const test_limit: c_int = 10;
const test_dec: api.format.Decode = .{ .codebook = .mcg, .window = .w12 };

/// A resident routed bank at the real GLM geometry: nine experts at rate n=36, 4096 -> 2048 -> 4096.
fn routedBank(ops: *@import("glm5_model.zig").Ops, rate: c_int) !api.Bank {
    var projections: [3]api.Proj = undefined;
    for (&projections, 0..) |*projection, i| {
        const input: c_int = if (i == 2) 2048 else 4096;
        const output: c_int = if (i == 2) 4096 else 2048;
        const key = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key, @intCast(123 + i)));
        const codes = try ops.slot();
        try mlx.check(mlx.mlx_random_bits(codes, &[_]c_int{ 9, @divExact(input, 16), @divExact(output, 16), rate }, 4, 2, key.*, ops.s));
        const su = try ops.slot();
        try mlx.check(mlx.mlx_random_uniform(su, try ops.scalar(-0.2, .float16), try ops.scalar(0.2, .float16), &[_]c_int{ 9, input }, 2, .float16, key.*, ops.s));
        const sv = try ops.slot();
        try mlx.check(mlx.mlx_random_uniform(sv, try ops.scalar(-0.05, .float16), try ops.scalar(0.05, .float16), &[_]c_int{ 9, output }, 2, .float16, key.*, ops.s));
        projection.* = .{ .trellis = codes.*, .suh = su.*, .svh = sv.* };
    }
    return .{ .gate = projections[0], .up = projections[1], .down = projections[2] };
}

fn routeInputs(ops: *@import("glm5_model.zig").Ops, rnd: std.Random, rows: usize) !struct { ids: Arr, scores: Arr } {
    var id_data: [8 * 8]u32 = undefined;
    var score_data: [8 * 8]f32 = undefined;
    for (id_data[0 .. rows * 8], score_data[0 .. rows * 8], 0..) |*id, *score, i| {
        id.* = @intCast((i / 8 * 3 + i % 8 * 2 + rnd.uintLessThan(usize, 2)) % 9);
        score.* = @as(f32, @floatFromInt(i % 8 + 1)) / 36 * 2.5;
    }
    const shape = [_]c_int{ 1, @intCast(rows), 8 };
    return .{
        .ids = try ops.own(mlx.mlx_array_new_data(&id_data, &shape, 3, .uint32)),
        .scores = try ops.own(mlx.mlx_array_new_data(&score_data, &shape, 3, .float32)),
    };
}

/// The two shared kernels, then the routed chain adding their output in its finish reduce: the chain the fold replaces.
fn separateChain(ops: *@import("glm5_model.zig").Ops, x: Arr, gate: Bank, up: Bank, down: Bank, down_bits: usize, routed: api.Bank, ids: Arr, scores: Arr, sigmoid: Arr) !Arr {
    const middle = try ops.own((try gateUp(ops.s, x, gate, up, 10, sigmoid)) orelse return error.TestExpectedFusedSharedGateUp);
    const mid = try ops.own(try affine(ops.s, middle, down, down_bits));
    return ops.own(try api.moeClampedWithShared(ops.s, x, routed, ids, scores, test_dec, test_limit, mid));
}

test "GLM shared expert folded into the routed launches is bit exact to the separate chains at every stored width" {
    const Ops = @import("glm5_model.zig").Ops;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(751);
    const rnd = random.random();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const sigmoid = try @import("hc_prefill.zig").sigmoidTable(s);
    const routed = try routedBank(&ops, 36);
    const route = try routeInputs(&ops, rnd, 1);
    defer api.glm_fold.shared_first = false;
    // Gate/up and down are stored at their own widths; the large gain pushes the activation through both clamps.
    for ([_][3]usize{ .{ 6, 6, 6 }, .{ 5, 5, 5 }, .{ 8, 8, 8 }, .{ 5, 5, 8 }, .{ 8, 8, 6 }, .{ 6, 6, 5 } }) |bits| for ([_]f32{ 1, 0.02 }) |gain| {
        var fx = Fixture{};
        defer fx.deinit();
        const gate = try fx.bank(rnd, 2048, 4096, bits[0], gain);
        const up = try fx.bank(rnd, 2048, 4096, bits[1], gain);
        const down = try fx.bank(rnd, 4096, 2048, bits[2], gain);
        const x = try ops.own(try randomRow(rnd, 4096, 4));
        const gate_up_separate = gateUpDispatches();
        const want = try separateChain(&ops, x, gate, up, down, bits[2], routed, route.ids, route.scores, sigmoid);
        try std.testing.expectEqual(gate_up_separate + 1, gateUpDispatches());
        for ([_]bool{ false, true }) |shared_first| {
            api.glm_fold.shared_first = shared_first;
            const gate_up_before = gateUpDispatches();
            const folded_before = api.glm_fold.foldedCount();
            const launches_before = api.glm_fold.launchCount();
            const got = try ops.own((try folded(s, x, gate, up, down, routed, route.ids, route.scores, test_dec, 10, sigmoid)) orelse return error.TestExpectedFoldedShared);
            try mlx.check(mlx.mlx_array_eval(got));
            try std.testing.expectEqual(gate_up_before, gateUpDispatches());
            try std.testing.expectEqual(folded_before + 1, api.glm_fold.foldedCount());
            // The separate chain launches seven: gate/up, the stock down and the routed five.
            try std.testing.expectEqual(launches_before + 5, api.glm_fold.launchCount());
            try expectSameBits(want, got);
        }
    };
}

test "GLM shared expert fold declines what the separate chains must run" {
    const Ops = @import("glm5_model.zig").Ops;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(753);
    const rnd = random.random();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const sigmoid = try @import("hc_prefill.zig").sigmoidTable(s);
    const routed = try routedBank(&ops, 36);
    const route = try routeInputs(&ops, rnd, 1);
    var fx = Fixture{};
    defer fx.deinit();
    const gate = try fx.bank(rnd, 2048, 4096, 6, 1);
    const up = try fx.bank(rnd, 2048, 4096, 6, 1);
    const down = try fx.bank(rnd, 4096, 2048, 6, 1);
    const x = try ops.own(try randomRow(rnd, 4096, 4));
    const folded_before = api.glm_fold.foldedCount();
    try std.testing.expect((try folded(s, x, gate, up, down, routed, route.ids, route.scores, test_dec, 9, sigmoid)) == null);
    try std.testing.expect((try folded(s, x, gate, up, down, routed, route.ids, route.scores, .{ .codebook = .mul1, .window = .w16 }, 10, sigmoid)) == null);
    const rows2 = try ops.own(try randomRow(rnd, 2 * 4096, 4));
    try std.testing.expect((try folded(s, try ops.reshape(rows2, &.{ 1, 2, 4096 }), gate, up, down, routed, route.ids, route.scores, test_dec, 10, sigmoid)) == null);
    try std.testing.expect((try folded(s, try ops.cast(x, .float32), gate, up, down, routed, route.ids, route.scores, test_dec, 10, sigmoid)) == null);
    const mixed = try fx.bank(rnd, 2048, 4096, 5, 1);
    try std.testing.expect((try folded(s, x, gate, mixed, down, routed, route.ids, route.scores, test_dec, 10, sigmoid)) == null);
    const narrow = try fx.bank(rnd, 4096, 1024, 6, 1);
    try std.testing.expect((try folded(s, x, gate, up, narrow, routed, route.ids, route.scores, test_dec, 10, sigmoid)) == null);
    const dense = Bank{ .weight = x, .scales = .{ .ctx = null }, .biases = .{ .ctx = null } };
    try std.testing.expect((try folded(s, x, dense, dense, dense, routed, route.ids, route.scores, test_dec, 10, sigmoid)) == null);
    var other = routed;
    other.down = other.gate;
    try std.testing.expect((try folded(s, x, gate, up, down, other, route.ids, route.scores, test_dec, 10, sigmoid)) == null);
    try std.testing.expectEqual(folded_before, api.glm_fold.foldedCount());
}

/// The shared rows chain and the grouped routed chain with the shared rows in its finish reduce: what the fold replaces
/// on two to eight verify rows.
fn separateRows(ops: *@import("glm5_model.zig").Ops, x: Arr, gate: Bank, up: Bank, down: Bank, routed: api.Bank, ids: Arr, scores: Arr) !?Arr {
    const Linear = @import("glm5_model.zig").Linear;
    const mlp: @import("glm5_model.zig").DenseMlp = .{
        .gate = Linear{ .w = gate.weight, .scales = gate.scales, .biases = gate.biases, .input = 4096, .output = 2048 },
        .up = Linear{ .w = up.weight, .scales = up.scales, .biases = up.biases, .input = 4096, .output = 2048 },
        .down = Linear{ .w = down.weight, .scales = down.scales, .biases = down.biases, .input = 2048, .output = 4096 },
    };
    const y = try @import("glm5_dflash_ffn.zig").dense(mlp, ops, x, 10);
    const joined = (try api.glm_group2.moeLayoutShared(ops.s, x, routed, ids, scores, test_dec, .serial, .grouped, .lane, y)) orelse return null;
    return try ops.own(joined);
}

test "GLM shared expert folded into the routed rows launches is bit exact to the shared rows chain at every row count and width" {
    const Ops = @import("glm5_model.zig").Ops;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(761);
    const rnd = random.random();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const sigmoid = try @import("hc_prefill.zig").sigmoidTable(s);
    const routed = try routedBank(&ops, 36);
    defer api.glm_fold.shared_first = false;
    for ([_][3]usize{ .{ 6, 6, 6 }, .{ 5, 5, 5 }, .{ 8, 8, 8 }, .{ 5, 5, 8 }, .{ 8, 8, 6 } }) |bits| for ([_]f32{ 1, 0.02 }) |gain| {
        var fx = Fixture{};
        defer fx.deinit();
        const gate = try fx.bank(rnd, 2048, 4096, bits[0], gain);
        const up = try fx.bank(rnd, 2048, 4096, bits[1], gain);
        const down = try fx.bank(rnd, 4096, 2048, bits[2], gain);
        for ([_]usize{ 2, 3, 4, 5, 6, 7, 8 }) |rows| {
            var scope = Ops{ .s = s };
            defer scope.deinit();
            const route = try routeInputs(&scope, rnd, rows);
            const values = try scope.own(try randomRow(rnd, rows * 4096, 4));
            const x = try scope.reshape(values, &.{ 1, @intCast(rows), 4096 });
            const want = (try separateRows(&scope, x, gate, up, down, routed, route.ids, route.scores)) orelse return error.TestExpectedGroup2Shared;
            for ([_]bool{ false, true }) |shared_first| {
                api.glm_fold.shared_first = shared_first;
                const folded_before = api.glm_fold.foldedCount();
                const launches_before = api.glm_fold.launchCount();
                const got = try scope.own((try folded(s, x, gate, up, down, routed, route.ids, route.scores, test_dec, 10, sigmoid)) orelse return error.TestExpectedFoldedShared);
                try std.testing.expectEqual(folded_before + 1, api.glm_fold.foldedCount());
                // The rows chain launches nine: gate, up, activation, down and the routed five.
                try std.testing.expectEqual(launches_before + 5, api.glm_fold.launchCount());
                try expectSameBits(want, got);
            }
        }
    };
}

test "GLM shared expert rows fold declines what the separate chains must run" {
    const Ops = @import("glm5_model.zig").Ops;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(763);
    const rnd = random.random();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const sigmoid = try @import("hc_prefill.zig").sigmoidTable(s);
    const routed = try routedBank(&ops, 36);
    var fx = Fixture{};
    defer fx.deinit();
    const gate = try fx.bank(rnd, 2048, 4096, 6, 1);
    const up = try fx.bank(rnd, 2048, 4096, 6, 1);
    const down = try fx.bank(rnd, 4096, 2048, 6, 1);
    const folded_before = api.glm_fold.foldedCount();
    const route = try routeInputs(&ops, rnd, 4);
    const four = try ops.reshape(try ops.own(try randomRow(rnd, 4 * 4096, 4)), &.{ 1, 4, 4096 });
    try std.testing.expect((try folded(s, four, gate, up, down, routed, route.ids, route.scores, test_dec, 9, sigmoid)) == null);
    try std.testing.expect((try folded(s, four, gate, up, down, routed, route.ids, route.scores, .{ .codebook = .mul1, .window = .w16 }, 10, sigmoid)) == null);
    try std.testing.expect((try folded(s, try ops.cast(four, .float32), gate, up, down, routed, route.ids, route.scores, test_dec, 10, sigmoid)) == null);
    const mixed = try fx.bank(rnd, 2048, 4096, 5, 1);
    try std.testing.expect((try folded(s, four, gate, mixed, down, routed, route.ids, route.scores, test_dec, 10, sigmoid)) == null);
    // Nine rows are past the lane kernels' verify width, one routed row short of the ids.
    const nine = try routeInputs(&ops, rnd, 8);
    const too_many = try ops.reshape(try ops.own(try randomRow(rnd, 9 * 4096, 4)), &.{ 1, 9, 4096 });
    try std.testing.expect((try folded(s, too_many, gate, up, down, routed, nine.ids, nine.scores, test_dec, 10, sigmoid)) == null);
    try std.testing.expect((try folded(s, four, gate, up, down, routed, nine.ids, nine.scores, test_dec, 10, sigmoid)) == null);
    try std.testing.expectEqual(folded_before, api.glm_fold.foldedCount());
}

test "GLM shared fold per layer chain (SUSHI_GLM_FOLD_UBENCH)" {
    if (!@import("transformer.zig").diagEnvOn("SUSHI_GLM_FOLD_UBENCH")) return error.SkipZigTest;
    const Ops = @import("glm5_model.zig").Ops;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    const layers = 8;
    const rounds = 21;
    var random = std.Random.DefaultPrng.init(757);
    const rnd = random.random();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const sigmoid = try @import("hc_prefill.zig").sigmoidTable(s);
    var routed: [layers]api.Bank = undefined;
    var fixtures: [layers]Fixture = @splat(.{});
    defer for (&fixtures) |*f| f.deinit();
    var gate: [layers]Bank = undefined;
    var up: [layers]Bank = undefined;
    var down: [layers]Bank = undefined;
    for (0..layers) |l| {
        routed[l] = try routedBank(&ops, 36);
        gate[l] = try fixtures[l].bank(rnd, 2048, 4096, 6, 1);
        up[l] = try fixtures[l].bank(rnd, 2048, 4096, 6, 1);
        down[l] = try fixtures[l].bank(rnd, 4096, 2048, 6, 1);
    }
    for (routed) |b| for ([_]Arr{ b.gate.trellis, b.up.trellis, b.down.trellis }) |a| try mlx.check(mlx.mlx_array_eval(a));
    defer api.glm_fold.shared_first = false;
    const names = [_][]const u8{ "separate", "fused routed first", "fused shared first", "routed only" };
    for ([_]usize{ 1, 2, 3, 4, 6, 8 }) |rows| {
        var ids: [layers]Arr = undefined;
        var scores: [layers]Arr = undefined;
        for (0..layers) |l| {
            const route = try routeInputs(&ops, rnd, rows);
            ids[l] = route.ids;
            scores[l] = route.scores;
        }
        var times: [4][rounds]f64 = undefined;
        for (0..rounds + 1) |rep| for (0..4) |arm| {
            api.glm_fold.shared_first = arm == 2;
            var scope = Ops{ .s = s };
            defer scope.deinit();
            var x = try scope.reshape(try scope.own(try randomRow(rnd, rows * 4096, 4)), &.{ 1, @intCast(rows), 4096 });
            try mlx.check(mlx.mlx_array_eval(x));
            const sw = @import("io_util.zig").Stopwatch.init(io);
            for (0..layers) |l| {
                x = switch (arm) {
                    0 => if (rows == 1) try separateChain(&scope, x, gate[l], up[l], down[l], 6, routed[l], ids[l], scores[l], sigmoid) else (try separateRows(&scope, x, gate[l], up[l], down[l], routed[l], ids[l], scores[l])).?,
                    3 => if (rows == 1) try scope.own(try api.moeClamped(s, x, routed[l], ids[l], scores[l], test_dec, test_limit)) else try scope.own((try api.glm_group2.moeLayout(s, x, routed[l], ids[l], scores[l], test_dec, .serial, .grouped, .lane)).?),
                    else => try scope.own((try folded(s, x, gate[l], up[l], down[l], routed[l], ids[l], scores[l], test_dec, 10, sigmoid)) orelse return error.TestExpectedFoldedShared),
                };
            }
            try mlx.check(mlx.mlx_array_eval(x));
            if (rep > 0) times[arm][rep - 1] = @as(f64, @floatFromInt(sw.read())) / 1e3 / layers;
        };
        for (&times) |*t| std.mem.sort(f64, t, {}, std.sort.asc(f64));
        for (names, times) |name, t| std.debug.print("[glm-fold-ubench] rows {d} {s}: {d:.1} us per layer\n", .{ rows, name, t[rounds / 2] });
        std.debug.print("[glm-fold-ubench] rows {d} fused / separate: {d:.3} (median of {d} rounds of {d} dependent layers)\n", .{ rows, times[1][rounds / 2] / times[0][rounds / 2], rounds, layers });
    }
}
