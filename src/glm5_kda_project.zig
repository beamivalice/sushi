//! One-token KDA projections in one dispatch: q/k/v on the resident affine banks (`glm5_decode.zig`) and the
//! dense BF16 f_a, g_a and beta GEMVs, written as the body's joined `[q|k|v|beta]` row plus the two low-rank
//! gate inputs. The dense rows reproduce MLX's `gemv_bfloat16_bm1_bn8_sm1_sn32_tm4_tn4` (K >= 16 rows):
//! eight K slices of 128 columns per 1024, each summed per lane then by shuffle-down 16..1, then added in slice order.
const std = @import("std");
const mlx = @import("mlx.zig");
const decode = @import("glm5_decode.zig");
const Arr = mlx.mlx_array;
const Bank = decode.Bank;

const SOURCE: [:0]const u8 =
    \\const uint lane=thread_index_in_simdgroup;
    \\const int sg=int(simdgroup_index_in_threadgroup);
    \\const int tg=int(threadgroup_position_in_grid.x);
    \\threadgroup float parts[8][4];
    \\if(tg<DENSE_TGS) {
    \\  const int row0=tg*4;
    \\  const device T* mat;
    \\  if(row0<NA) mat=wfa+size_t(row0)*K;
    \\  else if(row0<2*NA) mat=wga+size_t(row0-NA)*K;
    \\  else mat=wbeta+size_t(row0-2*NA)*K;
    \\  for(int slice=sg*4;slice<sg*4+4;++slice) {
    \\    float result[4]={0.0f,0.0f,0.0f,0.0f};
    \\    int bn=slice*128+int(lane)*4;
    \\    for(int i=0;i<K/1024;++i) {
    \\      float v[4];
    \\      for(int tn=0;tn<4;++tn) v[tn]=float(x[bn+tn]);
    \\      for(int tm=0;tm<4;++tm) for(int tn=0;tn<4;++tn) result[tm]+=float(mat[tm*K+bn+tn])*v[tn];
    \\      bn+=1024;
    \\    }
    \\    for(int tm=0;tm<4;++tm) for(ushort sn=16;sn>=1;sn>>=1) result[tm]+=simd_shuffle_down(result[tm],sn);
    \\    if(lane==0) for(int tm=0;tm<4;++tm) parts[slice][tm]=result[tm];
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if(sg==0&&lane<4) {
    \\    float r=parts[0][lane];
    \\    for(int slice=1;slice<8;++slice) r+=parts[slice][lane];
    \\    const int row=row0+int(lane);
    \\    if(row<NA) a_low[row]=static_cast<T>(r);
    \\    else if(row<2*NA) g_low[row-NA]=static_cast<T>(r);
    \\    else proj[3*N+row-2*NA]=static_cast<T>(r);
    \\  }
    \\} else {
    \\  const int row0=(tg-DENSE_TGS)*8;
    \\  const device uint8_t* w;
    \\  const device T* sc;
    \\  const device T* bi;
    \\  int local;
    \\  if(row0<N) {w=(const device uint8_t*)w0;sc=s0;bi=b0;local=row0;}
    \\  else if(row0<2*N) {w=(const device uint8_t*)w1;sc=s1;bi=b1;local=row0-N;}
    \\  else {w=(const device uint8_t*)w2;sc=s2;bi=b2;local=row0-2*N;}
    \\  const int r0=local+sg*4;
    \\  float result[4]={0.0f,0.0f,0.0f,0.0f};
    \\  glm_qmv_rows<T,K>(w+size_t(r0)*(K*GLM_BITS/8),sc+r0*(K/128),bi+r0*(K/128),x,lane,result);
    \\  const int out0=row0-local+r0;
    \\  for(int r=0;r<4;++r) {float v=simd_sum(result[r]);if(lane==0) proj[out0+r]=static_cast<T>(v);}
    \\}
;

pub const Dense = struct { fa: Arr, ga: Arr, beta: Arr };
pub const Result = struct {
    proj: Arr,
    a_low: Arr,
    gate_low: Arr,
    pub fn deinit(self: Result) void {
        _ = mlx.mlx_array_free(self.proj);
        _ = mlx.mlx_array_free(self.a_low);
        _ = mlx.mlx_array_free(self.gate_low);
    }
};

var kernels: [3]?mlx.mlx_fast_metal_kernel = @splat(null);
var dispatches: usize = 0;
pub fn dispatchCount() usize {
    return dispatches;
}

fn kernel(bits: c_int) !mlx.mlx_fast_metal_kernel {
    const slot = &kernels[if (bits == 8) 2 else @as(usize, @intCast(bits - 5))];
    if (slot.*) |k| return k;
    const names = [_][*:0]const u8{ "x", "w0", "s0", "b0", "w1", "s1", "b1", "w2", "s2", "b2", "wfa", "wga", "wbeta" };
    const iv = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const outs = [_][*:0]const u8{ "proj", "a_low", "g_low" };
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const label: [*:0]const u8 = switch (bits) {
        5 => "sushi_glm_kda_project_a5",
        6 => "sushi_glm_kda_project_a6",
        else => "sushi_glm_kda_project_a8",
    };
    const k = mlx.mlx_fast_metal_kernel_new(label, iv, ov, SOURCE, decode.header(bits), true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

const Key = struct { width: c_int, rows: c_int, low: c_int, heads: c_int };
const Entry = struct { key: Key, cfg: mlx.mlx_fast_metal_kernel_config };
var cache: [4]?Entry = @splat(null);

fn configuration(key: Key) !struct { value: mlx.mlx_fast_metal_kernel_config, cached: bool } {
    for (cache) |entry| if (entry) |e| {
        if (std.meta.eql(e.key, key)) return .{ .value = e.cfg, .cached = true };
    };
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, 1, 3 * key.rows + key.heads }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, 1, key.low }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, 1, key.low }, 3, .bfloat16));
    const dense_groups = @divExact(2 * key.low + key.heads, 4);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, (dense_groups + @divExact(3 * key.rows, 8)) * 64, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    inline for (.{ "K", "N", "NA", "DENSE_TGS" }, .{ key.width, key.rows, key.low, dense_groups }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, value));
    for (&cache) |*entry| if (entry.* == null) {
        entry.* = .{ .key = key, .cfg = cfg };
        return .{ .value = cfg, .cached = true };
    };
    return .{ .value = cfg, .cached = false };
}

fn denseRows(a: Arr, width: c_int) ?c_int {
    if (a.ctx == null or mlx.mlx_array_dtype(a) != .bfloat16) return null;
    const sh = mlx.getShape(a);
    const st = mlx.mlx_array_strides(a);
    if (sh.len != 2 or sh[1] != width or st[1] != 1 or st[0] != @as(usize, @intCast(width)) or @rem(sh[0], 4) != 0) return null;
    return sh[0];
}

/// q/k/v and the f_a, g_a, beta GEMVs of a one-token KDA layer, or null when a bank or width is not served.
pub fn project(s: mlx.mlx_stream, x: Arr, banks: [3]Bank, dense: Dense) !?Result {
    if (!mlx.streamIsGpu(s) or x.ctx == null or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const sh = mlx.getShape(x);
    if (sh.len != 3 or sh[0] != 1 or sh[1] != 1) return null;
    const width = sh[2];
    if (width <= 0 or @rem(width, 1024) != 0 or mlx.mlx_array_strides(x)[2] != 1) return null;
    var bits: c_int = 0;
    var rows: c_int = 0;
    for (banks) |bank| {
        const found = (try decode.checkBank(width, bank)) orelse return null;
        if ((bits != 0 and bits != found.bits) or (rows != 0 and rows != found.rows)) return null;
        bits = found.bits;
        rows = found.rows;
    }
    const low = denseRows(dense.fa, width) orelse return null;
    const heads = denseRows(dense.beta, width) orelse return null;
    if (denseRows(dense.ga, width) != low or @rem(rows, 8) != 0) return null;
    const handle = try configuration(.{ .width = width, .rows = rows, .low = low, .heads = heads });
    defer if (!handle.cached) {
        _ = mlx.mlx_fast_metal_kernel_config_free(handle.value);
    };
    const inputs = [_]Arr{ x, banks[0].weight, banks[0].scales, banks[0].biases, banks[1].weight, banks[1].scales, banks[1].biases, banks[2].weight, banks[2].scales, banks[2].biases, dense.fa, dense.ga, dense.beta };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try kernel(bits), iv, handle.value, s));
    var result = Result{ .proj = mlx.mlx_array_new(), .a_low = mlx.mlx_array_new(), .gate_low = mlx.mlx_array_new() };
    errdefer result.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&result.proj, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&result.a_low, ov, 1));
    try mlx.check(mlx.mlx_vector_array_get(&result.gate_low, ov, 2));
    dispatches += 1;
    decode.logEngaged(.kda_project, bits);
    return result;
}

test "GLM one-token KDA projections are bit exact to the staged q/k/v, f_a, g_a and beta chain" {
    const shared = @import("glm5_shared.zig");
    const Ops = @import("glm5_model.zig").Ops;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(734);
    const rnd = random.random();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    for ([_]usize{ 6, 5, 8 }) |bits| for ([_]usize{ 8192, 512 }) |rows| {
        var fx = shared.Fixture{};
        defer fx.deinit();
        var banks: [3]Bank = undefined;
        for (&banks) |*bank| bank.* = try fx.bank(rnd, rows, 4096, bits, 0.5);
        const x = try ops.own(try shared.randomRow(rnd, 4096, 4));
        var dense: [3]Arr = undefined;
        for (&dense, [_]usize{ 128, 128, 64 }) |*w, n| {
            const values = try std.testing.allocator.alloc(u16, n * 4096);
            defer std.testing.allocator.free(values);
            for (values) |*v| v.* = shared.Fixture.bf16((rnd.float(f32) - 0.5) * 0.25);
            w.* = try ops.own(mlx.mlx_array_new_data(values.ptr, &[_]c_int{ @intCast(n), 4096 }, 2, .bfloat16));
        }
        const before = dispatchCount();
        const got = (try project(s, x, banks, .{ .fa = dense[0], .ga = dense[1], .beta = dense[2] })) orelse return error.TestExpectedFusedKdaProjection;
        defer got.deinit();
        try std.testing.expectEqual(before + 1, dispatchCount());
        var parts: [4]Arr = undefined;
        for (parts[0..3], banks) |*part, bank| part.* = try ops.own(try shared.affine(s, x, bank, bits));
        parts[3] = try ops.binary(.mm, x, try ops.transpose(dense[2], &.{ 1, 0 }));
        try shared.expectSameBits(try ops.concat(&parts, -1), got.proj);
        try shared.expectSameBits(try ops.binary(.mm, x, try ops.transpose(dense[0], &.{ 1, 0 })), got.a_low);
        try shared.expectSameBits(try ops.binary(.mm, x, try ops.transpose(dense[1], &.{ 1, 0 })), got.gate_low);
    };
}

test "GLM one-token KDA projections decline unserved banks and widths" {
    const shared = @import("glm5_shared.zig");
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(735);
    const rnd = random.random();
    var fx = shared.Fixture{};
    defer fx.deinit();
    const bank = try fx.bank(rnd, 16, 1024, 6, 1);
    const x = try shared.randomRow(rnd, 1024, 2);
    defer _ = mlx.mlx_array_free(x);
    const matrix = try shared.randomRow(rnd, 4 * 1024, 1);
    defer _ = mlx.mlx_array_free(matrix);
    // Dense weights must be [rows, width] row-major; a flat vector is not.
    try std.testing.expect((try project(s, x, .{ bank, bank, bank }, .{ .fa = matrix, .ga = matrix, .beta = matrix })) == null);
    const narrow = try shared.randomRow(rnd, 512, 2);
    defer _ = mlx.mlx_array_free(narrow);
    try std.testing.expect((try project(s, narrow, .{ bank, bank, bank }, .{ .fa = matrix, .ga = matrix, .beta = matrix })) == null);
}
