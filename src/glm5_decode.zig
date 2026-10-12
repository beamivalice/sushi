//! Copy-free affine5/6/8 group128 GLM decode projections.
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;
pub const Bank = struct { weight: Arr, scales: Arr, biases: Arr };

// Port of oMLX 6745c39c's multi_qmv, Apache-2.0, narrowed to one BF16 token,
// 8 bits and group128. Its qmv_fast arithmetic derives from MLX (MIT). See NOTICE.
const SOURCE: [:0]const u8 =
    \\const uint lane=thread_index_in_simdgroup;
    \\const int sg=int(simdgroup_index_in_threadgroup);
    \\const int row0=int(threadgroup_position_in_grid.x)*8;
    \\const device uint8_t* w;
    \\const device T* sc;
    \\const device T* bi;
    \\device T* y;
    \\int local;
    \\if(row0<N0) {w=(const device uint8_t*)w0;sc=s0;bi=b0;y=y0;local=row0;}
    \\else if(row0<N0+N1) {w=(const device uint8_t*)w1;sc=s1;bi=b1;y=y1;local=row0-N0;}
    \\else {w=(const device uint8_t*)w2;sc=s2;bi=b2;y=y2;local=row0-N0-N1;}
    \\const int r0=local+sg*4;
    \\float result[4]={0.0f,0.0f,0.0f,0.0f};
    \\glm_qmv_rows<T,K>(w+size_t(r0)*(K*GLM_BITS/8),sc+r0*(K/128),bi+r0*(K/128),x,lane,result);
    \\for(int r=0;r<4;++r) {float v=simd_sum(result[r]);if(lane==0) y[r0+r]=static_cast<T>(v);}
;
const HEADER: [:0]const u8 =
    \\#include <metal_simdgroup>
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\constant constexpr int GLM_BITS=8;
    \\template<typename T>
    \\inline float glm_load_vector(const device T* x,thread float* local) {
    \\ float sum=0;
    \\ for(int i=0;i<8;++i) {sum+=x[i];local[i]=x[i];}
    \\ return sum;
    \\}
    \\inline float glm_qdot(const device uint8_t* w,const thread float* x,float scale,float bias,float sum) {
    \\ float accum=0;
    \\ for(int i=0;i<8;++i) accum+=x[i]*w[i];
    \\ return scale*accum+sum*bias;
    \\}
    \\template<typename T,int K>
    \\inline void glm_qmv_rows(const device uint8_t* ws,const device T* scales,const device T* biases,const device T* x,uint lane,thread float* result) {
    \\ thread float local[8];
    \\ ws+=lane*8;scales+=lane/16;biases+=lane/16;x+=lane*8;
    \\ for(int k=0;k<K;k+=256) {
    \\   float sum=glm_load_vector<T>(x,local);
    \\   for(int row=0;row<4;++row) {
    \\     const device uint8_t* wl=ws+row*K;
    \\     const device T* sl=scales+row*(K/128);
    \\     const device T* bl=biases+row*(K/128);
    \\     float s=sl[0];float b=bl[0];
    \\     result[row]+=glm_qdot(wl,local,s,b,sum);
    \\   }
    \\   ws+=256;scales+=2;biases+=2;x+=256;
    \\ }
    \\}
;
// Literal six-bit qmv_fast/load_vector/qdot ordering from pinned MLX quantized.h (MIT).
const HEADER6: [:0]const u8 =
    \\#include <metal_simdgroup>
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\constant constexpr int GLM_BITS=6;
    \\template<typename T>
    \\inline float glm_load_vector(const device T* x,thread float* local) {
    \\ float sum=0;
    \\ for(int i=0;i<8;i+=4) {sum+=x[i]+x[i+1]+x[i+2]+x[i+3];local[i]=x[i];local[i+1]=x[i+1]/64.0f;local[i+2]=x[i+2]/16.0f;local[i+3]=x[i+3]/4.0f;}
    \\ return sum;
    \\}
    \\inline float glm_qdot(const device uint8_t* w,const thread float* x,float scale,float bias,float sum) {
    \\ float accum=0;
    \\ for(int i=0;i<2;++i) {x+=4*i;w+=3*i;accum+=(w[0]&0x3f)*x[0];accum+=(w[0]&0xc0)*x[1];accum+=(w[1]&0x0f)*(x[1]*256.0f);accum+=(w[1]&0xf0)*x[2];accum+=(w[2]&0x03)*(x[2]*256.0f);accum+=(w[2]&0xfc)*x[3];}
    \\ return scale*accum+sum*bias;
    \\}
    \\template<typename T,int K>
    \\inline void glm_qmv_rows(const device uint8_t* ws,const device T* scales,const device T* biases,const device T* x,uint lane,thread float* result) {
    \\ thread float local[8];
    \\ ws+=lane*6;scales+=lane/16;biases+=lane/16;x+=lane*8;
    \\ for(int k=0;k<K;k+=256) {
    \\   float sum=glm_load_vector<T>(x,local);
    \\   for(int row=0;row<4;++row) {
    \\     const device uint8_t* wl=ws+row*(K/4*3);
    \\     const device T* sl=scales+row*(K/128);
    \\     const device T* bl=biases+row*(K/128);
    \\     float s=sl[0];float b=bl[0];
    \\     result[row]+=glm_qdot(wl,local,s,b,sum);
    \\   }
    \\   ws+=192;scales+=2;biases+=2;x+=256;
    \\ }
    \\}
;
// Literal five-bit qmv_fast/load_vector/qdot ordering from pinned MLX quantized.h (MIT):
// sixteen values per lane in two 5-byte packs, so K steps by 512.
const HEADER5: [:0]const u8 =
    \\#include <metal_simdgroup>
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\constant constexpr int GLM_BITS=5;
    \\template<typename T>
    \\inline float glm_load_vector(const device T* x,thread float* local) {
    \\ float sum=0;
    \\ for(int i=0;i<16;i+=8) {sum+=x[i]+x[i+1]+x[i+2]+x[i+3]+x[i+4]+x[i+5]+x[i+6]+x[i+7];local[i]=x[i];local[i+1]=x[i+1]/32.0f;local[i+2]=x[i+2]/4.0f;local[i+3]=x[i+3]/128.0f;local[i+4]=x[i+4]/16.0f;local[i+5]=x[i+5]/2.0f;local[i+6]=x[i+6]/64.0f;local[i+7]=x[i+7]/8.0f;}
    \\ return sum;
    \\}
    \\inline float glm_qdot(const device uint8_t* w,const thread float* x,float scale,float bias,float sum) {
    \\ float accum=0;
    \\ for(int i=0;i<2;++i) {x+=8*i;w+=5*i;accum+=(w[0]&0x1f)*x[0];accum+=(w[0]&0xe0)*x[1];accum+=(w[1]&0x3)*(x[1]*256.0f);accum+=(w[1]&0x7c)*x[2];accum+=(w[1]&0x80)*x[3];accum+=(w[2]&0xf)*(x[3]*256.0f);accum+=(w[2]&0xf0)*x[4];accum+=(w[3]&0x1)*(x[4]*256.0f);accum+=(w[3]&0x3e)*x[5];accum+=(w[3]&0xc0)*x[6];accum+=(w[4]&0x7)*(x[6]*256.0f);accum+=(w[4]&0xf8)*x[7];}
    \\ return scale*accum+sum*bias;
    \\}
    \\template<typename T,int K>
    \\inline void glm_qmv_rows(const device uint8_t* ws,const device T* scales,const device T* biases,const device T* x,uint lane,thread float* result) {
    \\ thread float local[16];
    \\ ws+=lane*10;scales+=lane/8;biases+=lane/8;x+=lane*16;
    \\ for(int k=0;k<K;k+=512) {
    \\   float sum=glm_load_vector<T>(x,local);
    \\   for(int row=0;row<4;++row) {
    \\     const device uint8_t* wl=ws+row*(K/8*5);
    \\     const device T* sl=scales+row*(K/128);
    \\     const device T* bl=biases+row*(K/128);
    \\     float s=sl[0];float b=bl[0];
    \\     result[row]+=glm_qdot(wl,local,s,b,sum);
    \\   }
    \\   ws+=320;scales+=4;biases+=4;x+=512;
    \\ }
    \\}
;
pub const Kernel = enum { qkv, hoist, tiles, shared, kda_project };
var logged: [5][9]bool = @splat(@splat(false));
/// One line per kernel and stored width names the affine paths a pack reaches.
pub fn logEngaged(comptime kernel: Kernel, bits: c_int) void {
    const seen = &logged[@intFromEnum(kernel)][@intCast(bits)];
    if (seen.* or @import("builtin").is_test) return;
    seen.* = true;
    const label = switch (kernel) {
        .qkv => "[glm] copy-free QKV",
        .hoist => "[glm-dflash] coefficient hoist rows",
        .tiles => "[glm-dflash] affine row tiles",
        .shared => "[glm] shared-expert gate/up/activation fusion",
        .kda_project => "[glm] KDA q/k/v/f_a/g_a/beta projection",
    };
    @import("log.zig").info("{s} engaged (A{d}g128)\n", .{ label, bits });
}
var dispatch_count: usize = 0;
/// Successful graph dispatches, read/reset on the sole MLX inference thread.
pub fn dispatchCount() usize {
    return dispatch_count;
}
pub fn resetDispatchCount() void {
    dispatch_count = 0;
}

var cached_kernels: [3]?mlx.mlx_fast_metal_kernel = @splat(null);
fn getKernel(bits: c_int) !mlx.mlx_fast_metal_kernel {
    const index: usize = switch (bits) {
        6 => 1,
        5 => 2,
        else => 0,
    };
    if (cached_kernels[index]) |k| return k;
    const inputs = [_][*:0]const u8{ "x", "w0", "s0", "b0", "w1", "s1", "b1", "w2", "s2", "b2" };
    const outputs = [_][*:0]const u8{ "y0", "y1", "y2" };
    const iv = mlx.mlx_vector_string_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outputs, outputs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const names = [_][*:0]const u8{ "sushi_glm_qkv_a8g128", "sushi_glm_qkv_a6g128", "sushi_glm_qkv_a5g128" };
    const headers = [_][:0]const u8{ HEADER, HEADER6, HEADER5 };
    const k = mlx.mlx_fast_metal_kernel_new(names[index], iv, ov, SOURCE, headers[index], true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    cached_kernels[index] = k;
    return k;
}
// Pinned mlx-c exposes this nonblocking availability query in array.h.
extern "c" fn _mlx_array_is_available(result: *bool, array: Arr) c_int;
fn materialized(a: Arr) !bool {
    if (a.ctx == null) return false;
    var ready = false;
    try mlx.check(_mlx_array_is_available(&ready, a));
    return ready;
}
fn rowMajor(a: Arr) bool {
    const sh = mlx.getShape(a);
    const st = mlx.mlx_array_strides(a);
    return sh.len == 2 and st[1] == 1 and st[0] == @as(usize, @intCast(sh[1]));
}
const Config = struct { value: mlx.mlx_fast_metal_kernel_config, cached: bool };
const Entry = struct { width: c_int, dims: [3]c_int, rank: usize, value: mlx.mlx_fast_metal_kernel_config };
var configs: [8]?Entry = @splat(null);
fn configuration(width: c_int, dims: [3]c_int, rank: usize) !Config {
    for (configs) |entry| if (entry) |e| {
        if (e.width == width and e.rank == rank and std.mem.eql(c_int, &e.dims, &dims)) return .{ .value = e.value, .cached = true };
    };
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (dims) |n| {
        var outshape: [3]c_int = @splat(1);
        outshape[rank - 1] = n;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &outshape, rank, .bfloat16));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(dims[0] + dims[1] + dims[2], 8) * 64, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", width));
    inline for (.{ "N0", "N1", "N2" }, 0..) |name, i| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, dims[i]));
    for (&configs) |*entry| if (entry.* == null) {
        entry.* = .{ .width = width, .dims = dims, .rank = rank, .value = cfg };
        return .{ .value = cfg, .cached = true };
    };
    return .{ .value = cfg, .cached = false };
}

pub const BankShape = struct { bits: c_int, rows: c_int };

/// A loaded, row-major affine g128 bank whose qmv_fast order these kernels reproduce for an input of `width`.
pub fn checkBank(width: c_int, bank: Bank) !?BankShape {
    if (bank.weight.ctx == null or bank.scales.ctx == null or bank.biases.ctx == null) return null;
    const w = mlx.getShape(bank.weight);
    if (w.len != 2 or w[0] <= 0 or @rem(w[0], 8) != 0 or mlx.mlx_array_dtype(bank.weight) != .uint32 or
        !(try materialized(bank.weight)) or !rowMajor(bank.weight)) return null;
    const bits: c_int = if (w[1] == @divExact(width, 4)) 8 else if (w[1] == @divExact(width, 16) * 3) 6 else if (@rem(width, 512) == 0 and w[1] == @divExact(width, 32) * 5) 5 else return null;
    const grid = [_]c_int{ w[0], @divExact(width, 128) };
    // Lazy views can acquire different strides at evaluation; only loaded grids qualify.
    for ([_]Arr{ bank.scales, bank.biases }) |a| if (mlx.mlx_array_dtype(a) != .bfloat16 or !std.mem.eql(c_int, &grid, mlx.getShape(a)) or
        !(try materialized(a)) or !rowMajor(a))
    {
        return null;
    };
    return .{ .bits = bits, .rows = w[0] };
}

/// The qmv_fast helpers (`glm_qmv_rows`) for a stored width.
pub fn header(bits: c_int) [:0]const u8 {
    return switch (bits) {
        6 => HEADER6,
        5 => HEADER5,
        else => HEADER,
    };
}

pub fn qkv(s: mlx.mlx_stream, x: Arr, banks: [3]Bank) !?[3]Arr {
    if (!mlx.streamIsGpu(s) or x.ctx == null or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const sh = mlx.getShape(x);
    if (sh.len < 2 or sh.len > 3) return null;
    const width = sh[sh.len - 1];
    if (width <= 0 or @rem(width, 256) != 0 or mlx.mlx_array_size(x) != @as(usize, @intCast(width)) or mlx.mlx_array_strides(x)[sh.len - 1] != 1) return null;
    var bits: c_int = 0;
    var dims: [3]c_int = undefined;
    for (banks, &dims) |bank, *n| {
        const found = (try checkBank(width, bank)) orelse return null;
        if (bits != 0 and bits != found.bits) return null;
        bits = found.bits;
        n.* = found.rows;
    }
    var total: i64 = 0;
    for (dims) |n| {
        total += n;
        if (@as(i64, n) * @divExact(width, 128) > std.math.maxInt(c_int)) return null;
    }
    if (total > @divTrunc(std.math.maxInt(c_int), 8)) return null;
    const handle = try configuration(width, dims, sh.len);
    defer if (!handle.cached) {
        _ = mlx.mlx_fast_metal_kernel_config_free(handle.value);
    };
    const cfg = handle.value;
    const inputs = [_]Arr{ x, banks[0].weight, banks[0].scales, banks[0].biases, banks[1].weight, banks[1].scales, banks[1].biases, banks[2].weight, banks[2].scales, banks[2].biases };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try getKernel(bits), iv, cfg, s));
    var result: [3]Arr = undefined;
    var made: usize = 0;
    errdefer for (result[0..made]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&result, 0..) |*out, i| {
        out.* = mlx.mlx_array_new();
        made += 1;
        try mlx.check(mlx.mlx_vector_array_get(out, ov, i));
    }
    dispatch_count += 1;
    logEngaged(.qkv, bits);
    return result;
}

test "GLM copy-free QKV is bit exact to affine8 group128 MLX" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(416138);
    const rnd = random.random();
    for ([_]struct { k: usize, n: usize }{ .{ .k = 256, .n = 24 }, .{ .k = 4096, .n = 8192 } }) |shape| {
        var arrays: [9]Arr = undefined;
        var made: usize = 0;
        defer for (arrays[0..made]) |v| {
            _ = mlx.mlx_array_free(v);
        };
        var banks: [3]Bank = undefined;
        for (&banks, 0..) |*bank, p| {
            const rows = shape.n + if (shape.k == 256) p * 8 else @as(usize, 0);
            const codes = try a.alloc(u32, rows * shape.k / 4);
            defer a.free(codes);
            const scales = try a.alloc(u16, rows * shape.k / 128);
            defer a.free(scales);
            const biases = try a.alloc(u16, scales.len);
            defer a.free(biases);
            for (codes) |*v| v.* = rnd.int(u32);
            for (scales, biases) |*sc, *bias| {
                const scale: f32 = (rnd.float(f32) + 0.125) / 128;
                const b: f32 = (rnd.float(f32) - 0.5) * @as(f32, @floatFromInt(p + 1));
                sc.* = @truncate(@as(u32, @bitCast(scale)) >> 16);
                bias.* = @truncate(@as(u32, @bitCast(b)) >> 16);
            }
            arrays[made] = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ @intCast(rows), @intCast(shape.k / 4) }, 2, .uint32);
            made += 1;
            arrays[made] = mlx.mlx_array_new_data(scales.ptr, &[_]c_int{ @intCast(rows), @intCast(shape.k / 128) }, 2, .bfloat16);
            made += 1;
            arrays[made] = mlx.mlx_array_new_data(biases.ptr, &[_]c_int{ @intCast(rows), @intCast(shape.k / 128) }, 2, .bfloat16);
            made += 1;
            bank.* = .{ .weight = arrays[made - 3], .scales = arrays[made - 2], .biases = arrays[made - 1] };
        }
        const input = try a.alloc(u16, shape.k);
        defer a.free(input);
        for (input) |*v| {
            const f: f32 = (rnd.float(f32) - 0.5) * 4;
            v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
        }
        const x = mlx.mlx_array_new_data(input.ptr, &[_]c_int{ 1, 1, @intCast(shape.k) }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        const before = dispatchCount();
        const output = (try qkv(s, x, banks)) orelse return error.TestExpectedFusedQkv;
        try std.testing.expectEqual(before + 1, dispatchCount());
        defer for (output) |v| {
            _ = mlx.mlx_array_free(v);
        };
        for (banks, output) |bank, got| {
            var expected = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(expected);
            try mlx.check(mlx.mlx_quantized_matmul(&expected, x, bank.weight, bank.scales, bank.biases, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(8), "affine", s));
            try mlx.check(mlx.mlx_array_eval(expected));
            try mlx.check(mlx.mlx_array_eval(got));
            try std.testing.expectEqualSlices(c_int, mlx.getShape(expected), mlx.getShape(got));
            const count = mlx.mlx_array_size(got);
            try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(expected).?[0..count], mlx.mlx_array_data_bfloat16(got).?[0..count]);
        }
    }
}

test "GLM copy-free QKV declines unsupported inputs without copying weights" {
    const data: [512]f32 = @splat(0);
    const banks: [3]Bank = @splat(.{ .weight = .{ .ctx = null }, .scales = .{ .ctx = null }, .biases = .{ .ctx = null } });
    const Case = struct { shape: [3]c_int, dtype: mlx.mlx_dtype };
    for ([_]Case{ .{ .shape = .{ 1, 2, 256 }, .dtype = .bfloat16 }, .{ .shape = .{ 1, 1, 128 }, .dtype = .bfloat16 }, .{ .shape = .{ 1, 1, 256 }, .dtype = .float32 }, .{ .shape = .{ 1, 1, 256 }, .dtype = .bfloat16 } }) |case| {
        const x = mlx.mlx_array_new_data(&data, &case.shape, 3, case.dtype);
        defer _ = mlx.mlx_array_free(x);
        try std.testing.expect((try qkv(mlx.gpuStream(), x, banks)) == null);
    }
}

test "GLM copy-free QKV declines dense and strided banks" {
    const s = mlx.gpuStream();
    const xv: [256]u16 = @splat(0x3f80);
    const x = mlx.mlx_array_new_data(&xv, &[_]c_int{ 1, 256 }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const weights: [24 * 256]u16 = @splat(0);
    const dense = mlx.mlx_array_new_data(&weights, &[_]c_int{ 24, 256 }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(dense);
    const absent = Arr{ .ctx = null };
    try std.testing.expect((try qkv(s, x, @splat(.{ .weight = dense, .scales = absent, .biases = absent }))) == null);
    const codes: [64 * 24]u32 = @splat(0x11121314);
    const raw = mlx.mlx_array_new_data(&codes, &[_]c_int{ 64, 24 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(raw);
    var strided = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(strided);
    try mlx.check(mlx.mlx_transpose_axes(&strided, raw, &[_]c_int{ 1, 0 }, 2, s));
    const sv: [24 * 2]u16 = @splat(0x3d00);
    const sc = mlx.mlx_array_new_data(&sv, &[_]c_int{ 24, 2 }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(sc);
    try std.testing.expect((try qkv(s, x, @splat(.{ .weight = strided, .scales = sc, .biases = sc }))) == null);
    try mlx.check(mlx.mlx_array_eval(strided));
    try std.testing.expect((try qkv(s, x, @splat(.{ .weight = strided, .scales = sc, .biases = sc }))) == null);
}

test "GLM copy-free QKV is bit exact to affine6 and affine5 group128 MLX" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(416138);
    const rnd = random.random();
    // Five-bit qmv_fast steps K by 512, so 512 is its narrowest width.
    for ([_]usize{ 6, 5 }) |bits| for ([_]struct { k: usize, n: usize }{ .{ .k = if (bits == 5) 512 else 256, .n = 24 }, .{ .k = 4096, .n = 8192 } }) |shape| {
        var arrays: [9]Arr = undefined;
        var made: usize = 0;
        defer for (arrays[0..made]) |v| {
            _ = mlx.mlx_array_free(v);
        };
        var banks: [3]Bank = undefined;
        for (&banks, 0..) |*bank, p| {
            const rows = shape.n + if (shape.k < 4096) p * 8 else @as(usize, 0);
            const codes = try a.alloc(u32, rows * shape.k * bits / 32);
            defer a.free(codes);
            const scales = try a.alloc(u16, rows * shape.k / 128);
            defer a.free(scales);
            const biases = try a.alloc(u16, scales.len);
            defer a.free(biases);
            for (codes) |*v| v.* = rnd.int(u32);
            for (scales, biases) |*sc, *bias| {
                const scale: f32 = (rnd.float(f32) + 0.125) / 128;
                const b: f32 = (rnd.float(f32) - 0.5) * @as(f32, @floatFromInt(p + 1));
                sc.* = @truncate(@as(u32, @bitCast(scale)) >> 16);
                bias.* = @truncate(@as(u32, @bitCast(b)) >> 16);
            }
            arrays[made] = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ @intCast(rows), @intCast(shape.k * bits / 32) }, 2, .uint32);
            made += 1;
            arrays[made] = mlx.mlx_array_new_data(scales.ptr, &[_]c_int{ @intCast(rows), @intCast(shape.k / 128) }, 2, .bfloat16);
            made += 1;
            arrays[made] = mlx.mlx_array_new_data(biases.ptr, &[_]c_int{ @intCast(rows), @intCast(shape.k / 128) }, 2, .bfloat16);
            made += 1;
            bank.* = .{ .weight = arrays[made - 3], .scales = arrays[made - 2], .biases = arrays[made - 1] };
        }
        const input = try a.alloc(u16, shape.k);
        defer a.free(input);
        for (input) |*v| {
            const f: f32 = (rnd.float(f32) - 0.5) * 4;
            v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
        }
        const x = mlx.mlx_array_new_data(input.ptr, &[_]c_int{ 1, 1, @intCast(shape.k) }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        const before = dispatchCount();
        const output = (try qkv(s, x, banks)) orelse return error.TestExpectedFusedQkv;
        try std.testing.expectEqual(before + 1, dispatchCount());
        defer for (output) |v| {
            _ = mlx.mlx_array_free(v);
        };
        for (banks, output) |bank, got| {
            var expected = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(expected);
            try mlx.check(mlx.mlx_quantized_matmul(&expected, x, bank.weight, bank.scales, bank.biases, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(@intCast(bits)), "affine", s));
            try mlx.check(mlx.mlx_array_eval(expected));
            try mlx.check(mlx.mlx_array_eval(got));
            try std.testing.expectEqualSlices(c_int, mlx.getShape(expected), mlx.getShape(got));
            const count = mlx.mlx_array_size(got);
            try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(expected).?[0..count], mlx.mlx_array_data_bfloat16(got).?[0..count]);
        }
    };
}

test "GLM copy-free QKV rejects mixed six and eight bit banks" {
    const s = mlx.gpuStream();
    const xv: [256]u16 = @splat(0x3f80);
    const x = mlx.mlx_array_new_data(&xv, &.{ 1, 1, 256 }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const words: [24 * 64]u32 = @splat(0xabcdef01);
    const w8 = mlx.mlx_array_new_data(&words, &.{ 24, 64 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(w8);
    const w6 = mlx.mlx_array_new_data(&words, &.{ 24, 48 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(w6);
    const grid: [48]u16 = @splat(0x3c80);
    const sc = mlx.mlx_array_new_data(&grid, &.{ 24, 2 }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(sc);
    const before = dispatchCount();
    try std.testing.expect((try qkv(s, x, .{ .{ .weight = w6, .scales = sc, .biases = sc }, .{ .weight = w8, .scales = sc, .biases = sc }, .{ .weight = w6, .scales = sc, .biases = sc } })) == null);
    // Five-bit rows narrower than 512 are stock qmv, not qmv_fast.
    const w5 = mlx.mlx_array_new_data(&words, &.{ 24, 40 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(w5);
    try std.testing.expect((try qkv(s, x, @splat(.{ .weight = w5, .scales = sc, .biases = sc }))) == null);
    try std.testing.expectEqual(before, dispatchCount());
}
