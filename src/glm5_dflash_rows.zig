//! Exact row-streaming A5/A6 projections for two to eight verify rows. Every (row, output, lane) chain is
//! qmv_fast's, so each row equals the serial one-row projection bit for bit; rows are independent chains, so
//! how they are spread over simdgroups changes no bit. Each weight is read once (two simdgroups of a threadgroup
//! share it through L1 at seven or eight rows).
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;
const Ops = @import("glm5_model.zig").Ops;
const Linear = @import("glm5_model.zig").Linear;

pub const min_rows = 2;
pub const max_rows = 8;

// qmv_fast's lane block is V inputs (8 for A6, 16 for A5); a value straddling a byte is two products, kept
// separate. MLX's x*256 on the high part is folded into the coefficient, an exact product either way.
const COEFFICIENTS =
    \\    float coeff[O][C];
    \\    for (int o = 0; o < O; ++o) {
    \\      const device uint8_t* q = codes + size_t(output0 + o) * size_t(K * BITS / 8) + size_t((k + lane * V) * BITS / 8 + pack * B);
    \\      if (BITS == 6) {
    \\        coeff[o][0] = float(q[0] & 0x3f);
    \\        coeff[o][1] = float(q[0] & 0xc0);
    \\        coeff[o][2] = float(q[1] & 0x0f) * 256.0f;
    \\        coeff[o][3] = float(q[1] & 0xf0);
    \\        coeff[o][4] = float(q[2] & 0x03) * 256.0f;
    \\        coeff[o][5] = float(q[2] & 0xfc);
    \\      } else {
    \\        coeff[o][0] = float(q[0] & 0x1f);
    \\        coeff[o][1] = float(q[0] & 0xe0);
    \\        coeff[o][2] = float(q[1] & 0x03) * 256.0f;
    \\        coeff[o][3] = float(q[1] & 0x7c);
    \\        coeff[o][4] = float(q[1] & 0x80);
    \\        coeff[o][5] = float(q[2] & 0x0f) * 256.0f;
    \\        coeff[o][6] = float(q[2] & 0xf0);
    \\        coeff[o][7] = float(q[3] & 0x01) * 256.0f;
    \\        coeff[o][8] = float(q[3] & 0x3e);
    \\        coeff[o][9] = float(q[3] & 0xc0);
    \\        coeff[o][10] = float(q[4] & 0x07) * 256.0f;
    \\        coeff[o][11] = float(q[4] & 0xf8);
    \\      }
    \\    }
;
const PRODUCTS =
    \\        if (BITS == 6) {
    \\          a += coeff[o][0] * xv[0];
    \\          a += coeff[o][1] * xv[1];
    \\          a += coeff[o][2] * xv[1];
    \\          a += coeff[o][3] * xv[2];
    \\          a += coeff[o][4] * xv[2];
    \\          a += coeff[o][5] * xv[3];
    \\        } else {
    \\          a += coeff[o][0] * xv[0];
    \\          a += coeff[o][1] * xv[1];
    \\          a += coeff[o][2] * xv[1];
    \\          a += coeff[o][3] * xv[2];
    \\          a += coeff[o][4] * xv[3];
    \\          a += coeff[o][5] * xv[3];
    \\          a += coeff[o][6] * xv[4];
    \\          a += coeff[o][7] * xv[4];
    \\          a += coeff[o][8] * xv[5];
    \\          a += coeff[o][9] * xv[6];
    \\          a += coeff[o][10] * xv[6];
    \\          a += coeff[o][11] * xv[7];
    \\        }
;
const PROLOGUE =
    \\const int lane = int(thread_index_in_simdgroup);
    \\const device uint8_t* codes = reinterpret_cast<const device uint8_t*>(w);
    \\constexpr int V = BITS == 5 ? 16 : 8;
    \\constexpr int P = BITS == 5 ? 8 : 4;
    \\constexpr int B = BITS == 5 ? 5 : 3;
    \\constexpr int C = BITS == 5 ? 12 : 6;
;

/// STREAM: O outputs per simdgroup, SG simdgroups per threadgroup, every row streamed through the coefficients.
const STREAM_SOURCE = PROLOGUE ++
    \\const int output0 = (int(threadgroup_position_in_grid.x) * SG + int(simdgroup_index_in_threadgroup)) * O;
    \\float result[M][O];
    \\for (int m = 0; m < M; ++m) for (int o = 0; o < O; ++o) result[m][o] = 0.0f;
    \\for (int k = 0; k < K; k += 32 * V) {
    \\  float accum[M][O];
    \\  for (int m = 0; m < M; ++m) for (int o = 0; o < O; ++o) accum[m][o] = 0.0f;
    \\  for (int pack = 0; pack < 2; ++pack) {
++ COEFFICIENTS ++
    \\    for (int m = 0; m < M; ++m) {
    \\      const device float* xs = px + m * K + k + lane * V + pack * P;
    \\      float xv[P];
    \\      for (int i = 0; i < P; ++i) xv[i] = xs[i];
    \\      for (int o = 0; o < O; ++o) {
    \\        float a = accum[m][o];
++ PRODUCTS ++
    \\        accum[m][o] = a;
    \\      }
    \\    }
    \\  }
    \\  for (int o = 0; o < O; ++o) {
    \\    const int group = (output0 + o) * (K / 128) + k / 128 + lane / (128 / V);
    \\    const float scale = float(scales[group]);
    \\    const float bias = float(biases[group]);
    \\    for (int m = 0; m < M; ++m) {
    \\      const float sum = ps[m * (K / V) + k / V + lane];
    \\      result[m][o] += scale * accum[m][o] + sum * bias;
    \\    }
    \\  }
    \\}
    \\for (int m = 0; m < M; ++m) for (int o = 0; o < O; ++o) {
    \\  const float value = simd_sum(result[m][o]);
    \\  if (lane == 0) y[m * N + output0 + o] = bfloat(value);
    \\}
;

/// SPLIT: STREAM with the rows halved over two simdgroups of one threadgroup, which read the same weights.
const SPLIT_SOURCE = PROLOGUE ++
    \\const int output0 = (int(threadgroup_position_in_grid.x) * (SG / 2) + int(simdgroup_index_in_threadgroup) / 2) * O;
    \\const int row0 = (int(simdgroup_index_in_threadgroup) % 2) * MT;
    \\float result[MT][O];
    \\for (int m = 0; m < MT; ++m) for (int o = 0; o < O; ++o) result[m][o] = 0.0f;
    \\for (int k = 0; k < K; k += 32 * V) {
    \\  float accum[MT][O];
    \\  for (int m = 0; m < MT; ++m) for (int o = 0; o < O; ++o) accum[m][o] = 0.0f;
    \\  for (int pack = 0; pack < 2; ++pack) {
++ COEFFICIENTS ++
    \\    for (int m = 0; m < MT; ++m) {
    \\      if (row0 + m >= M) break;
    \\      const device float* xs = px + (row0 + m) * K + k + lane * V + pack * P;
    \\      float xv[P];
    \\      for (int i = 0; i < P; ++i) xv[i] = xs[i];
    \\      for (int o = 0; o < O; ++o) {
    \\        float a = accum[m][o];
++ PRODUCTS ++
    \\        accum[m][o] = a;
    \\      }
    \\    }
    \\  }
    \\  for (int o = 0; o < O; ++o) {
    \\    const int group = (output0 + o) * (K / 128) + k / 128 + lane / (128 / V);
    \\    const float scale = float(scales[group]);
    \\    const float bias = float(biases[group]);
    \\    for (int m = 0; m < MT; ++m) {
    \\      if (row0 + m >= M) break;
    \\      const float sum = ps[(row0 + m) * (K / V) + k / V + lane];
    \\      result[m][o] += scale * accum[m][o] + sum * bias;
    \\    }
    \\  }
    \\}
    \\for (int m = 0; m < MT; ++m) for (int o = 0; o < O; ++o) {
    \\  const float value = simd_sum(result[m][o]);
    \\  if (lane == 0 && row0 + m < M) y[(row0 + m) * N + output0 + o] = bfloat(value);
    \\}
;

/// BOTH (A6): both packs' coefficients live and one row's inputs; no accumulator array across packs.
const BOTH_SOURCE =
    \\const int lane = int(thread_index_in_simdgroup);
    \\const int output0 = (int(threadgroup_position_in_grid.x) * SG + int(simdgroup_index_in_threadgroup)) * O;
    \\const device uint8_t* codes = reinterpret_cast<const device uint8_t*>(w);
    \\float result[M][O];
    \\for (int m = 0; m < M; ++m) for (int o = 0; o < O; ++o) result[m][o] = 0.0f;
    \\for (int k = 0; k < K; k += 256) {
    \\  float coeff[O][12];
    \\  float sc[O];
    \\  float bi[O];
    \\  for (int o = 0; o < O; ++o) {
    \\    const device uint8_t* q = codes + size_t(output0 + o) * size_t(K * 3 / 4) + size_t((k + lane * 8) * 3 / 4);
    \\    for (int pack = 0; pack < 2; ++pack) {
    \\      const int c = pack * 6;
    \\      coeff[o][c] = float(q[pack*3] & 0x3f);
    \\      coeff[o][c+1] = float(q[pack*3] & 0xc0);
    \\      coeff[o][c+2] = float(q[pack*3+1] & 0x0f) * 256.0f;
    \\      coeff[o][c+3] = float(q[pack*3+1] & 0xf0);
    \\      coeff[o][c+4] = float(q[pack*3+2] & 0x03) * 256.0f;
    \\      coeff[o][c+5] = float(q[pack*3+2] & 0xfc);
    \\    }
    \\    const int group = (output0 + o) * (K / 128) + k / 128 + lane / 16;
    \\    sc[o] = float(scales[group]);
    \\    bi[o] = float(biases[group]);
    \\  }
    \\  for (int m = 0; m < M; ++m) {
    \\    float xv[8];
    \\    for (int i = 0; i < 8; ++i) xv[i] = px[m * K + k + lane * 8 + i];
    \\    const float sum = ps[m * (K / 8) + k / 8 + lane];
    \\    for (int o = 0; o < O; ++o) {
    \\      float a = 0.0f;
    \\      for (int pack = 0; pack < 2; ++pack) {
    \\        const int c = pack * 6, i = pack * 4;
    \\        a += coeff[o][c] * xv[i];
    \\        a += coeff[o][c+1] * xv[i+1];
    \\        a += coeff[o][c+2] * xv[i+1];
    \\        a += coeff[o][c+3] * xv[i+2];
    \\        a += coeff[o][c+4] * xv[i+2];
    \\        a += coeff[o][c+5] * xv[i+3];
    \\      }
    \\      result[m][o] += sc[o] * a + sum * bi[o];
    \\    }
    \\  }
    \\}
    \\for (int m = 0; m < M; ++m) for (int o = 0; o < O; ++o) {
    \\  const float value = simd_sum(result[m][o]);
    \\  if (lane == 0) y[m * N + output0 + o] = bfloat(value);
    \\}
;

// Preserve qmv_fast's BF16 sum boundaries and power-of-two scaling once per input.
const PREPARE_SOURCE =
    \\const uint group=thread_position_in_grid.x;
    \\constexpr uint V=BITS==5?16:8;
    \\if(group>=uint(M*K/V))return;
    \\const uint at=group*V;
    \\float sum=0.0f;
    \\if(BITS==5) for(uint i=0;i<16;i+=8) {
    \\ sum+=x[at+i]+x[at+i+1]+x[at+i+2]+x[at+i+3]+x[at+i+4]+x[at+i+5]+x[at+i+6]+x[at+i+7];
    \\ px[at+i]=float(x[at+i]);
    \\ px[at+i+1]=float(x[at+i+1])/32.0f;
    \\ px[at+i+2]=float(x[at+i+2])/4.0f;
    \\ px[at+i+3]=float(x[at+i+3])/128.0f;
    \\ px[at+i+4]=float(x[at+i+4])/16.0f;
    \\ px[at+i+5]=float(x[at+i+5])/2.0f;
    \\ px[at+i+6]=float(x[at+i+6])/64.0f;
    \\ px[at+i+7]=float(x[at+i+7])/8.0f;
    \\} else for(uint i=0;i<8;i+=4) {
    \\ sum+=x[at+i]+x[at+i+1]+x[at+i+2]+x[at+i+3];
    \\ px[at+i]=float(x[at+i]);
    \\ px[at+i+1]=float(x[at+i+1])/64.0f;
    \\ px[at+i+2]=float(x[at+i+2])/16.0f;
    \\ px[at+i+3]=float(x[at+i+3])/4.0f;
    \\}
    \\ps[group]=sum;
;

pub const Kind = enum { stream, split, both };
pub const Plan = struct { kind: Kind, o: c_int, sg: c_int };

/// The variant measured fastest for this width, stored bits and input width (the design's table: the second
/// tile of a row set is cheap once its weights sit in L1, and a wide input row costs more than it saves).
pub fn plan(bits: c_int, rows: c_int, k: c_int) Plan {
    const wide = k >= 16384;
    if (bits == 6) {
        if (rows <= 6) return .{ .kind = .stream, .o = 2, .sg = 4 };
        return .{ .kind = if (wide) .both else .split, .o = 2, .sg = 8 };
    }
    if (rows <= 4 or wide) return .{ .kind = .stream, .o = 2, .sg = 4 };
    return .{ .kind = .split, .o = 2, .sg = 8 };
}

var dispatches: [3]usize = @splat(0);
pub fn dispatchCount(kind: Kind) usize {
    return dispatches[@intFromEnum(kind)];
}
pub fn resetDispatchCount() void {
    dispatches = @splat(0);
}

var kernels: [4]?mlx.mlx_fast_metal_kernel = @splat(null);
const names = [_][*:0]const u8{ "sushi_glm_rows_prepare", "sushi_glm_rows_stream", "sushi_glm_rows_split", "sushi_glm_rows_both" };
const sources = [_][*:0]const u8{ PREPARE_SOURCE, STREAM_SOURCE, SPLIT_SOURCE, BOTH_SOURCE };

fn kernel(index: usize) !mlx.mlx_fast_metal_kernel {
    if (kernels[index]) |k| return k;
    const ins: []const [*:0]const u8 = if (index == 0) &.{"x"} else &.{ "w", "scales", "biases", "px", "ps" };
    const outs: []const [*:0]const u8 = if (index == 0) &.{ "px", "ps" } else &.{"y"};
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(names[index], iv, ov, sources[index], "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernels[index] = k;
    return k;
}

const Key = struct { kind: u8, m: c_int, n: c_int, k: c_int, bits: c_int, o: c_int, sg: c_int };
const Cached = struct { key: Key, config: mlx.mlx_fast_metal_kernel_config };
var configs: [160]?Cached = @splat(null);

/// The configuration for `key`, kept for reuse while a slot is free, else owned by `owned` for this call.
fn configFor(key: Key, owned: *?mlx.mlx_fast_metal_kernel_config) !mlx.mlx_fast_metal_kernel_config {
    for (configs) |item| if (item) |entry| if (std.meta.eql(key, entry.key)) return entry.config;
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    const prepare = key.kind == 0;
    if (prepare) {
        const v: c_int = if (key.bits == 5) 16 else 8;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ 1, key.m, key.k }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ 1, key.m, @divExact(key.k, v) }, 3, .float32));
        const total = @divExact(key.m * key.k, v);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divTrunc(total + 255, 256) * 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 256, 1, 1));
    } else {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ 1, key.m, key.n }, 3, .bfloat16));
        const per_group: c_int = if (key.kind == 2) @divExact(key.sg, 2) else key.sg;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(key.n, key.o * per_group) * key.sg * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, key.sg * 32, 1, 1));
    }
    const common = [_]struct { []const u8, c_int }{ .{ "M", key.m }, .{ "N", key.n }, .{ "K", key.k }, .{ "BITS", key.bits }, .{ "O", key.o }, .{ "SG", key.sg }, .{ "MT", @divTrunc(key.m + 1, 2) } };
    for (common) |item| {
        // A template argument a source never names would still instantiate it; skip what the source does not read.
        const used = switch (key.kind) {
            0 => std.mem.eql(u8, item[0], "M") or std.mem.eql(u8, item[0], "K") or std.mem.eql(u8, item[0], "BITS"),
            1 => !std.mem.eql(u8, item[0], "MT"),
            2 => true,
            else => !std.mem.eql(u8, item[0], "MT") and !std.mem.eql(u8, item[0], "BITS"),
        };
        if (!used) continue;
        var buf: [8:0]u8 = undefined;
        @memcpy(buf[0..item[0].len], item[0]);
        buf[item[0].len] = 0;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, &buf, item[1]));
    }
    for (&configs) |*slot| if (slot.* == null) {
        slot.* = .{ .key = key, .config = c };
        return c;
    };
    owned.* = c;
    return c;
}

fn rowMajorReady(a: Arr) !bool {
    if (a.ctx == null) return false;
    const shape = mlx.getShape(a);
    if (shape.len != 2) return false;
    const strides = mlx.mlx_array_strides(a);
    if (strides[1] != 1 or strides[0] != shape[1]) return false;
    var available = false;
    try mlx.check(mlx._mlx_array_is_available(&available, a));
    return available;
}

/// `x` [1, rows, K] BF16 against the stored A5/A6 group-128 `linear`, rows in 2..8; null when the shape, width or
/// layout is outside what the kernels were proven on.
pub fn project(stream: mlx.mlx_stream, x: Arr, linear: Linear) !?Arr {
    if (!mlx.streamIsGpu(stream) or x.ctx == null or linear.w.ctx == null or linear.scales.ctx == null or linear.biases.ctx == null) return null;
    const sh = mlx.getShape(x);
    const ws = mlx.getShape(linear.w);
    if (sh.len != 3 or sh[0] != 1 or sh[1] < min_rows or sh[1] > max_rows or sh[2] < 256 or @mod(sh[2], 256) != 0 or ws.len != 2 or ws[0] < 8 or @mod(ws[0], 8) != 0) return null;
    const bits: c_int = if (@as(i64, ws[1]) * 16 == @as(i64, sh[2]) * 3) 6 else if (@mod(sh[2], 512) == 0 and @as(i64, ws[1]) * 32 == @as(i64, sh[2]) * 5) 5 else return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(linear.w) != .uint32 or !(try rowMajorReady(linear.w))) return null;
    for ([_]Arr{ linear.scales, linear.biases }) |grid| {
        if (mlx.mlx_array_dtype(grid) != .bfloat16 or !std.mem.eql(c_int, &.{ ws[0], @divExact(sh[2], 128) }, mlx.getShape(grid)) or !(try rowMajorReady(grid))) return null;
    }
    const xs = mlx.mlx_array_strides(x);
    if (xs[2] != 1 or xs[1] != sh[2]) return null;
    const choice = plan(bits, sh[1], sh[2]);
    const per_group: c_int = if (choice.kind == .split) @divExact(choice.sg, 2) else choice.sg;
    if (@mod(ws[0], choice.o * per_group) != 0) return null;

    var owned_prepare: ?mlx.mlx_fast_metal_kernel_config = null;
    defer if (owned_prepare) |c| {
        _ = mlx.mlx_fast_metal_kernel_config_free(c);
    };
    const prep_config = try configFor(.{ .kind = 0, .m = sh[1], .n = 0, .k = sh[2], .bits = bits, .o = 0, .sg = 0 }, &owned_prepare);
    const iv0 = mlx.mlx_vector_array_new_value(x);
    defer _ = mlx.mlx_vector_array_free(iv0);
    var ov0 = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov0);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov0, try kernel(0), iv0, prep_config, stream));
    var px = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(px);
    var ps = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ps);
    try mlx.check(mlx.mlx_vector_array_get(&px, ov0, 0));
    try mlx.check(mlx.mlx_vector_array_get(&ps, ov0, 1));

    var owned_main: ?mlx.mlx_fast_metal_kernel_config = null;
    defer if (owned_main) |c| {
        _ = mlx.mlx_fast_metal_kernel_config_free(c);
    };
    const index: usize = @as(usize, @intFromEnum(choice.kind)) + 1;
    const config = try configFor(.{ .kind = @as(u8, @intFromEnum(choice.kind)) + 1, .m = sh[1], .n = ws[0], .k = sh[2], .bits = bits, .o = choice.o, .sg = choice.sg }, &owned_main);
    const arrays = [_]Arr{ linear.w, linear.scales, linear.biases, px, ps };
    const iv = mlx.mlx_vector_array_new_data(&arrays, arrays.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try kernel(index), iv, config, stream));
    var output = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(output);
    try mlx.check(mlx.mlx_vector_array_get(&output, ov, 0));
    dispatches[@intFromEnum(choice.kind)] += 1;
    @import("glm5_decode.zig").logEngaged(.rows, bits);
    return output;
}

fn mismatches(a: Arr, b: Arr) !usize {
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    const n = mlx.mlx_array_size(a);
    if (n != mlx.mlx_array_size(b)) return n;
    const x = mlx.mlx_array_data_bfloat16(a).?[0..n];
    const y = mlx.mlx_array_data_bfloat16(b).?[0..n];
    var count: usize = 0;
    for (x, y) |u, v| count += @intFromBool(u != v);
    return count;
}

test "GLM row-streaming projections equal serial qmv on every row, width and input magnitude" {
    const s = mlx.gpuStream();
    const Shape = struct { n: c_int, k: c_int };
    resetDispatchCount();
    for ([_]c_int{ 6, 5 }) |bits| for ([_]Shape{
        .{ .n = 32, .k = if (bits == 5) 512 else 256 },
        .{ .n = 1536, .k = 4096 },
        .{ .n = 8192, .k = 4096 },
        .{ .n = 4096, .k = 8192 },
        .{ .n = 4096, .k = 16384 },
        .{ .n = 16384, .k = 1536 },
        .{ .n = 2048, .k = 4096 },
        .{ .n = 4096, .k = 2048 },
        .{ .n = 160, .k = 4096 },
    }) |shape| {
        var fx = Ops{ .s = s };
        defer fx.deinit();
        const key = try fx.slot();
        try mlx.check(mlx.mlx_random_key(key, @intCast(shape.n + shape.k + bits)));
        const codes = try fx.slot();
        try mlx.check(mlx.mlx_random_bits(codes, &[_]c_int{ shape.n, @divExact(shape.k * bits, 32) }, 2, 4, key.*, s));
        const scales = try fx.slot();
        try mlx.check(mlx.mlx_random_uniform(scales, try fx.scalar(0.001, .bfloat16), try fx.scalar(0.02, .bfloat16), &[_]c_int{ shape.n, @divExact(shape.k, 128) }, 2, .bfloat16, key.*, s));
        const biases = try fx.binary(.mul, scales.*, try fx.scalar(if (bits == 5) -15.5 else -31.5, .bfloat16));
        const linear = Linear{ .w = codes.*, .scales = scales.*, .biases = biases, .input = shape.k, .output = shape.n };
        for ([_]Arr{ linear.w, linear.scales, linear.biases }) |a| try mlx.check(mlx.mlx_array_eval(a));
        // The smaller magnitude pushes x/128 into FP32 subnormals, the larger stays finite in BF16 sums.
        for ([_]f32{ 1, 1e-30, 3e4 }) |spread| {
            const x8 = try fx.slot();
            try mlx.check(mlx.mlx_random_normal(x8, &[_]c_int{ 1, 8, shape.k }, 3, .bfloat16, 0, spread, key.*, s));
            try mlx.check(mlx.mlx_array_eval(x8.*));
            for (2..9) |rows_usize| {
                const rows: c_int = @intCast(rows_usize);
                var ops = Ops{ .s = s };
                defer ops.deinit();
                const x = try ops.contiguous(try ops.slice(x8.*, 1, 0, rows));
                try mlx.check(mlx.mlx_array_eval(x));
                const got = try ops.own((try project(s, x, linear)) orelse return error.TestExpectedGlmRows);
                for (0..rows_usize) |row| {
                    const want = try linear.apply(&ops, try ops.slice(x, 1, @intCast(row), @intCast(row + 1)));
                    try std.testing.expectEqual(@as(usize, 0), try mismatches(want, try ops.slice(got, 1, @intCast(row), @intCast(row + 1))));
                }
            }
        }
    };
    for ([_]Kind{ .stream, .split, .both }) |kind| try std.testing.expect(dispatchCount(kind) > 0);
}

test "GLM row-streaming projections decline what they were not proven on" {
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const codes = try ops.zeros(&.{ 64, 192 }, .uint32);
    const scales = try ops.ones(&.{ 64, 8 }, .bfloat16);
    try mlx.check(mlx.mlx_array_eval(codes));
    try mlx.check(mlx.mlx_array_eval(scales));
    const linear = Linear{ .w = codes, .scales = scales, .biases = scales, .input = 1024, .output = 64 };
    const x = try ops.ones(&.{ 1, 4, 1024 }, .bfloat16);
    try std.testing.expect((try project(s, x, linear)) != null);
    for ([_]c_int{ 1, 9 }) |rows| try std.testing.expect((try project(s, try ops.ones(&.{ 1, rows, 1024 }, .bfloat16), linear)) == null);
    try std.testing.expect((try project(s, try ops.cast(x, .float32), linear)) == null);
    try std.testing.expect((try project(s, try ops.ones(&.{ 2, 4, 1024 }, .bfloat16), linear)) == null);
    try std.testing.expect((try project(s, try ops.ones(&.{ 1, 4, 1000 }, .bfloat16), linear)) == null);
    var other = linear;
    other.scales = try ops.ones(&.{ 64, 16 }, .bfloat16);
    other.biases = other.scales;
    try std.testing.expect((try project(s, x, other)) == null);
    const cpu_stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_stream);
    try std.testing.expect((try project(cpu_stream, x, linear)) == null);
}

test "GLM rows plan: each width has one variant that divides every served bank" {
    for ([_]c_int{ 5, 6 }) |bits| for ([_]c_int{ 256, 1536, 2048, 4096, 8192, 16384 }) |k| for (2..9) |rows| {
        const choice = plan(bits, @intCast(rows), k);
        const per_group: c_int = if (choice.kind == .split) @divExact(choice.sg, 2) else choice.sg;
        for ([_]c_int{ 32, 160, 1536, 2048, 4096, 8192, 16384, 154880 }) |n| try std.testing.expectEqual(@as(c_int, 0), @mod(n, choice.o * per_group));
        try std.testing.expect(choice.kind != .both or bits == 6);
    };
}
