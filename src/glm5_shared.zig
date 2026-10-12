//! One-token GLM shared expert on resident affine5/6/8 g128 banks: gate, up and the clamped BF16 SwiGLU in one
//! dispatch. Each projection keeps MLX qmv_fast's order (`glm5_decode.zig`) and the activation keeps
//! `glm5_activation.zig`'s BF16 rounding boundaries.
const std = @import("std");
const mlx = @import("mlx.zig");
const decode = @import("glm5_decode.zig");
const Arr = mlx.mlx_array;
const Bank = decode.Bank;

const GATE_UP: [:0]const u8 =
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
    \\  g=g>10.0f?10.0f:g;
    \\  u=u>10.0f?10.0f:u;
    \\  u=u<-10.0f?-10.0f:u;
    \\  bfloat16_t cg=bfloat16_t(g);
    \\  bfloat16_t mid=bfloat16_t(g*float(sigmoid[as_type<ushort>(cg)]));
    \\  y[r0+lane]=bfloat16_t(float(mid)*u);
    \\}
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

pub const Fixture = struct {
    arrays: [9]Arr = undefined,
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
