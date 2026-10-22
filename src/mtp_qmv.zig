const std = @import("std");
const mlx = @import("mlx.zig");
const dense_rows = @import("mtp_dense_rows.zig");

// Each row retains MLX qmv's lane assignment, affine dot, accumulation order and bf16 rounding.
const SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const uint vec0 = threadgroup_position_in_grid.x * NV;
    \\const uint out_row = threadgroup_position_in_grid.y * (2u * RPS) + sg * RPS;
    \\const int K = int(K_size);
    \\const int N = int(N_size);
    \\constexpr int VPT = FAST ? 8 : 4;
    \\constexpr int BLOCK = VPT * 32;
    \\const int groups = K / GS;
    \\const device uchar* ws = (const device uchar*)w + size_t(out_row) * K + lane * VPT;
    \\const device T* sp = sc + size_t(out_row) * groups + lane / (GS / VPT);
    \\const device T* bp = bi + size_t(out_row) * groups + lane / (GS / VPT);
    \\const device T* xp[NV];
    \\for (int v = 0; v < NV; ++v) xp[v] = x + size_t(min(vec0 + uint(v), uint(M - 1))) * K + lane * VPT;
    \\float result[NV][RPS] = {};
    \\int k = 0;
    \\for (; k < (FAST ? K : K - BLOCK); k += BLOCK) {
    \\  float xt[NV][VPT];
    \\  float sums[NV] = {};
    \\  for (int v = 0; v < NV; ++v) {
    \\    for (int i = 0; i < VPT; ++i) { sums[v] += xp[v][i]; xt[v][i] = xp[v][i]; }
    \\  }
    \\  for (int row = 0; row < RPS; ++row) {
    \\    uchar codes[VPT];
    \\    for (int i = 0; i < VPT; ++i) codes[i] = ws[row * K + i];
    \\    const float scale = sp[row * groups];
    \\    const float bias = bp[row * groups];
    \\    for (int v = 0; v < NV; ++v) {
    \\      float accum = 0.0f;
    \\      for (int i = 0; i < VPT; ++i) accum += xt[v][i] * codes[i];
    \\      result[v][row] += scale * accum + sums[v] * bias;
    \\    }
    \\  }
    \\  ws += BLOCK;
    \\  sp += BLOCK / GS;
    \\  bp += BLOCK / GS;
    \\  for (int v = 0; v < NV; ++v) xp[v] += BLOCK;
    \\}
    \\if (!FAST) {
    \\  const int remaining = clamp(K - k - int(lane) * VPT, 0, VPT);
    \\  if (remaining > 0) {
    \\    float xt[NV][VPT];
    \\    float sums[NV] = {};
    \\    for (int v = 0; v < NV; ++v) {
    \\      for (int i = 0; i < remaining; ++i) { sums[v] += xp[v][i]; xt[v][i] = xp[v][i]; }
    \\    }
    \\    for (int row = 0; row < RPS; ++row) {
    \\      uchar codes[VPT];
    \\      for (int i = 0; i < remaining; ++i) codes[i] = ws[row * K + i];
    \\      const float scale = sp[row * groups];
    \\      const float bias = bp[row * groups];
    \\      for (int v = 0; v < NV; ++v) {
    \\        float accum = 0.0f;
    \\        for (int i = 0; i < remaining; ++i) accum += xt[v][i] * codes[i];
    \\        result[v][row] += scale * accum + sums[v] * bias;
    \\      }
    \\    }
    \\  }
    \\}
    \\for (int v = 0; v < NV; ++v) {
    \\  for (int row = 0; row < RPS; ++row) {
    \\    const float value = simd_sum(result[v][row]);
    \\    if (lane == 0 && vec0 + uint(v) < uint(M)) y[size_t(vec0 + uint(v)) * N + out_row + row] = T(value);
    \\  }
    \\}
;

// Row-streaming arm: O outputs per simdgroup keep their codes in registers while rows stream through, each
// row's per-lane block chain being qmv's. SPLIT pairs simdgroups on the same outputs, MT rows each.
const STREAM_SOURCE =
    \\const int lane = int(thread_index_in_simdgroup);
    \\const int sgi = int(simdgroup_index_in_threadgroup);
    \\const int output0 = (int(threadgroup_position_in_grid.x) * (SPLIT ? SG / 2 : SG) + (SPLIT ? sgi / 2 : sgi)) * O;
    \\const int row0 = SPLIT ? (sgi % 2) * MT : 0;
    \\float result[MT][O];
    \\for (int m = 0; m < MT; ++m) for (int o = 0; o < O; ++o) result[m][o] = 0.0f;
    \\for (int k = 0; k < K; k += 256) {
    \\  float codes[O][8];
    \\  float sc[O];
    \\  float bi[O];
    \\  for (int o = 0; o < O; ++o) {
    \\    const device uint8_t* q = reinterpret_cast<const device uint8_t*>(w) + size_t(output0 + o) * K + k + lane * 8;
    \\    for (int i = 0; i < 8; ++i) codes[o][i] = float(q[i]);
    \\    const int group = (output0 + o) * (K / GS) + (k + lane * 8) / GS;
    \\    sc[o] = float(scales[group]);
    \\    bi[o] = float(biases[group]);
    \\  }
    \\  for (int m = 0; m < MT; ++m) {
    \\    if (row0 + m >= M) break;
    \\    const device bfloat16_t* xp = x + (row0 + m) * K + k + lane * 8;
    \\    float xt[8];
    \\    float sum = 0.0f;
    \\    for (int i = 0; i < 8; ++i) { sum += xp[i]; xt[i] = xp[i]; }
    \\    for (int o = 0; o < O; ++o) {
    \\      float accum = 0.0f;
    \\      for (int i = 0; i < 8; ++i) accum += xt[i] * codes[o][i];
    \\      result[m][o] += sc[o] * accum + sum * bi[o];
    \\    }
    \\  }
    \\}
    \\for (int m = 0; m < MT; ++m) for (int o = 0; o < O; ++o) {
    \\  const float value = simd_sum(result[m][o]);
    \\  if (lane == 0 && row0 + m < M) y[(row0 + m) * N + output0 + o] = bfloat16_t(value);
    \\}
;

const Arm = enum { tile, s1x4, p4x8, p2x8 };

// Measured per shape on the Flash-Next trunk and head (docs/qwen4-perf.md#verify-rows8): the tile arm above is
// flat to 4 rows, then pays a second tile.
fn rowsArm(n: c_int, k: c_int, m: c_int, gs: c_int) Arm {
    if (@mod(k, 256) != 0 or @mod(gs, 8) != 0) return .tile;
    const arm: Arm = switch (m) {
        5 => .s1x4,
        6 => .p4x8,
        7 => if (n == 640) .p2x8 else .s1x4,
        8 => if (n == 10240 or n == 12288) .p2x8 else .tile,
        else => .tile,
    };
    if (arm == .tile) return arm;
    const g = armGeometry(arm);
    return if (@mod(n, g.o * (if (g.split) @divExact(g.sg, 2) else g.sg)) == 0) arm else .tile;
}

fn armGeometry(arm: Arm) struct { o: c_int, sg: c_int, split: bool } {
    return switch (arm) {
        .tile => unreachable,
        .s1x4 => .{ .o = 1, .sg = 4, .split = false },
        .p4x8 => .{ .o = 4, .sg = 8, .split = true },
        .p2x8 => .{ .o = 2, .sg = 8, .split = true },
    };
}

pub var rows_arm_calls: [4]u64 = @splat(0);
var stream_kernel: ?mlx.mlx_fast_metal_kernel = null;

var kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;
const Key = struct { dims: [8]c_int = @splat(0), ndim: usize, n: c_int, gs: c_int };
const Entry = struct { key: Key, arm: Arm, cfg: mlx.mlx_fast_metal_kernel_config, k_size: mlx.mlx_array, n_size: mlx.mlx_array, tick: u64 };
var entries: [128]Entry = undefined;
var count: usize = 0;
var tick: u64 = 0;

fn configuration(xs: []const c_int, n: c_int, k: c_int, m: c_int, gs: c_int) !*const Entry {
    var key = Key{ .ndim = xs.len, .n = n, .gs = gs };
    @memcpy(key.dims[0..xs.len], xs);
    tick +%= 1;
    for (entries[0..count]) |*entry| {
        if (std.meta.eql(key, entry.key)) {
            entry.tick = tick;
            return entry;
        }
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const k_size = mlx.mlx_array_new_int(k);
    errdefer _ = mlx.mlx_array_free(k_size);
    const n_size = mlx.mlx_array_new_int(n);
    errdefer _ = mlx.mlx_array_free(n_size);
    var shape = key.dims;
    shape[xs.len - 1] = n;
    const arm = rowsArm(n, k, m, gs);
    if (arm != .tile) {
        const g = armGeometry(arm);
        const per_group = if (g.split) @divExact(g.sg, 2) else g.sg;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, xs.len, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(n, g.o * per_group) * g.sg * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, g.sg * 32, 1, 1));
        const mt = if (g.split) @divTrunc(m + 1, 2) else m;
        inline for (.{ "M", "N", "K", "GS", "O", "SG", "MT", "SPLIT" }, .{ m, n, k, gs, g.o, g.sg, mt, @as(c_int, @intFromBool(g.split)) }) |name, value|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, value));
        return store(key, arm, cfg, k_size, n_size);
    }
    const tiles = @divTrunc(m + 3, 4);
    const nv = @divTrunc(m + tiles - 1, tiles);
    const fast = @mod(k, 256) == 0;
    // Two output rows per simdgroup on the unrolled path put twice the simdgroups in flight; the tail
    // path keeps four.
    const rps: c_int = if (fast) 2 else 4;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, xs.len, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, tiles * 32, @divExact(n, 2 * rps) * 2, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 2, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "M", m));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", gs));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NV", nv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "FAST", @intFromBool(fast)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RPS", rps));
    return store(key, .tile, cfg, k_size, n_size);
}

fn store(key: Key, arm: Arm, cfg: mlx.mlx_fast_metal_kernel_config, k_size: mlx.mlx_array, n_size: mlx.mlx_array) *const Entry {
    var slot = count;
    if (count == entries.len) {
        slot = 0;
        for (entries[0..count], 0..) |entry, i| if (entry.tick < entries[slot].tick) {
            slot = i;
        };
        _ = mlx.mlx_fast_metal_kernel_config_free(entries[slot].cfg);
        _ = mlx.mlx_array_free(entries[slot].k_size);
        _ = mlx.mlx_array_free(entries[slot].n_size);
    } else count += 1;
    entries[slot] = .{ .key = key, .arm = arm, .cfg = cfg, .k_size = k_size, .n_size = n_size, .tick = tick };
    return &entries[slot];
}

// Every admitted affine8 geometry uses M=1 arithmetic for each of at most 32 rows.
// Shapes outside the shared tile use separate M=1 calls, never a width-dependent qmm.
pub fn matmul(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array) !?mlx.mlx_array {
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    if (xs.len < 2 or xs.len > 8 or ws.len != 2 or ss.len != 2) return null;
    const k = xs[xs.len - 1];
    const n = ws[0];
    if (k <= 0 or n <= 0 or ws[1] <= 0 or ss[1] <= 0 or mlx.mlx_array_dtype(w) != .uint32) return null;
    const geom = @import("expert_quant.zig").affineGeomFromShapes(@intCast(ws[1]), @intCast(ss[1]), @intCast(k)) orelse return null;
    if (geom.bits != 8 or ss[0] != n or !std.mem.eql(c_int, ss, mlx.getShape(bi))) return null;
    const rows = mlx.mlx_array_size(x) / @as(usize, @intCast(k));
    if (rows < 2 or rows > 32) return null;
    const gs: c_int = @intCast(geom.group_size);
    if (!mlx.streamIsGpu(s) or k < 256 or @mod(n, 8) != 0 or
        mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16)
        return try serialRows(s, x, w, sc, bi, gs, @intCast(rows));
    if (kernel == null) {
        const names = [_][*:0]const u8{ "x", "w", "sc", "bi", "K_size", "N_size" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&names, names.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("mtp_serial_qmv8_rows", iv, ov, SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        kernel = value;
    }
    if (!engaged) {
        engaged = true;
        @import("log.zig").info("[mtp-qmv] row-identical affine8 projections engaged\n", .{});
    }
    const cfg = try configuration(xs, n, k, @intCast(rows), gs);
    rows_arm_calls[@intFromEnum(cfg.arm)] +%= 1;
    const tile_inputs = [_]mlx.mlx_array{ x, w, sc, bi, cfg.k_size, cfg.n_size };
    const stream_inputs = [_]mlx.mlx_array{ x, w, sc, bi };
    const inputs: []const mlx.mlx_array = if (cfg.arm == .tile) &tile_inputs else &stream_inputs;
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, if (cfg.arm == .tile) kernel.? else try streamKernel(), iv, cfg.cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

fn streamKernel() !mlx.mlx_fast_metal_kernel {
    if (stream_kernel) |value| return value;
    const names = [_][*:0]const u8{ "x", "w", "scales", "biases" };
    const outs = [_][*:0]const u8{"y"};
    const iv = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const value = mlx.mlx_fast_metal_kernel_new("mtp_serial_qmv8_rows_stream", iv, ov, STREAM_SOURCE, "", true, false);
    if (value.ctx == null) return error.MetalKernelCompileFailed;
    stream_kernel = value;
    return value;
}

pub fn denseMatmul(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array) !?mlx.mlx_array {
    if (x.ctx == null or w.ctx == null) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len < 2 or xs.len > 8 or ws.len != 2 or ws[0] <= 0 or ws[1] <= 0 or xs[xs.len - 1] != ws[0]) return null;
    const rows = mlx.mlx_array_size(x) / @as(usize, @intCast(ws[0]));
    if (rows < 1 or rows > 32) return null;
    // One row takes the same kernel as a verify row where the kernel exists, else MLX's own matmul.
    if (rows >= 2) if (try f32GemvRows(s, x, w)) |out| return out;
    if (dense_rows.enabled()) {
        if (try dense_rows.matmul(s, x, w)) |out| return out;
    }
    if (rows == 1) return null;
    return try serialRows(s, x, w, .{}, .{}, null, @intCast(rows));
}

// MLX v0.32.2 `gemv` (BM1 BN8 SM1 SN32 TM4 TN4) for f32, one threadgroup layer per row.
const F32_GEMV_ROWS_SOURCE =
    \\constexpr int BN = 8;
    \\constexpr int SN = 32;
    \\constexpr int TM = 4;
    \\constexpr int TN = 4;
    \\constexpr int blockM = TM;
    \\constexpr int blockN = BN * SN * TN;
    \\const int thrN = int(thread_index_in_simdgroup);
    \\const int sgN = int(simdgroup_index_in_threadgroup) % BN;
    \\const int simdN = SN * sgN;
    \\const int K = int(K_size);
    \\const int N = int(N_size);
    \\const uint row = threadgroup_position_in_grid.z;
    \\threadgroup float tgp_memory[BN * (blockM + TM)];
    \\float result[TM] = {0};
    \\float inter[TN];
    \\float v_coeff[TN];
    \\int bn = (simdN + thrN) * TN;
    \\const int out_row = int(threadgroup_position_in_grid.x) * blockM;
    \\const device float* mat = w + size_t(out_row) * K;
    \\const device float* in_vec = x + size_t(row) * K;
    \\for (int i = 0; i < K / blockN; ++i) {
    \\  for (int tn = 0; tn < TN; tn++) v_coeff[tn] = in_vec[bn + tn];
    \\  int mat_offset = 0;
    \\  for (int tm = 0; tm < TM; tm++) {
    \\    for (int tn = 0; tn < TN; tn++) inter[tn] = mat[mat_offset + bn + tn];
    \\    for (int tn = 0; tn < TN; tn++) result[tm] += inter[tn] * v_coeff[tn];
    \\    mat_offset += K;
    \\  }
    \\  bn += blockN;
    \\}
    \\for (int tm = 0; tm < TM; tm++) {
    \\  for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) result[tm] += simd_shuffle_down(result[tm], sn);
    \\}
    \\threadgroup float* tgp_results = tgp_memory + sgN * (blockM + TM);
    \\if (thrN == 0) {
    \\  for (int tm = 0; tm < TM; tm++) tgp_results[tm] = result[tm];
    \\  threadgroup_barrier(mem_flags::mem_none);
    \\  if (sgN == 0) {
    \\    for (int sgn = 1; sgn < BN; sgn++) {
    \\      for (int tm = 0; tm < TM; tm++) result[tm] += tgp_results[sgn * (blockM + TM) + tm];
    \\    }
    \\  }
    \\}
    \\if (simdN == 0 && thrN == 0) {
    \\  for (int tm = 0; tm < TM; tm++) y[size_t(row) * N + out_row + tm] = result[tm];
    \\}
;

var f32_gemv_kernel: ?mlx.mlx_fast_metal_kernel = null;
var f32_gemv_engaged = false;
const F32GemvKey = struct { rows: c_int, n: c_int, k: c_int };
var f32_gemv_cfgs: [4]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var f32_gemv_keys: [4]F32GemvKey = undefined;
var f32_gemv_next: usize = 0;

/// Every row of `x` against the transposed f32 `w` [K, N] in one dispatch, each row the M=1 MLX gemv
/// bit for bit. Null outside the one gemv configuration it reproduces (MiMo's router is inside it).
fn f32GemvRows(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or mlx.mlx_array_dtype(x) != .float32 or mlx.mlx_array_dtype(w) != .float32) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const k = ws[0];
    const n = ws[1];
    // MLX picks BM1 BN8 TM4 for a transposed weight when K >= 16 N and N < 4096; the tails are left out.
    if (@mod(k, 1024) != 0 or @mod(n, 4) != 0 or n >= 4096 or k < 16 * n) return null;
    const wst = mlx.mlx_array_strides(w);
    if (wst[0] != 1 or wst[1] != @as(usize, @intCast(k))) return null;
    const rows: c_int = @intCast(mlx.mlx_array_size(x) / @as(usize, @intCast(k)));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    // The kernel reads x row-major and w through its transposed strides, so only x is made contiguous
    // (a lazy array reports row-contiguous strides before it is evaluated).
    var reshaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reshaped);
    try mlx.check(mlx.mlx_reshape(&reshaped, x, &.{ rows, k }, 2, s));
    try mlx.check(mlx.mlx_contiguous(&flat, reshaped, false, s));
    if (f32_gemv_kernel == null) {
        const names = [_][*:0]const u8{ "x", "w", "K_size", "N_size" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&names, names.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("mtp_serial_gemv_f32_rows", iv, ov, F32_GEMV_ROWS_SOURCE, "", false, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        f32_gemv_kernel = value;
    }
    const key = F32GemvKey{ .rows = rows, .n = n, .k = k };
    const slot = for (f32_gemv_keys, f32_gemv_cfgs, 0..) |entry, cfg, i| {
        if (cfg != null and std.meta.eql(entry, key)) break i;
    } else blk: {
        const i = f32_gemv_next;
        f32_gemv_next = (f32_gemv_next + 1) % f32_gemv_cfgs.len;
        if (f32_gemv_cfgs[i]) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        f32_gemv_cfgs[i] = null;
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ rows, n }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(n, 4) * 32, 8, rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 8, 1));
        f32_gemv_cfgs[i] = cfg;
        f32_gemv_keys[i] = key;
        break :blk i;
    };
    const k_size = mlx.mlx_array_new_int(k);
    defer _ = mlx.mlx_array_free(k_size);
    const n_size = mlx.mlx_array_new_int(n);
    defer _ = mlx.mlx_array_free(n_size);
    const inputs = [_]mlx.mlx_array{ flat, w, k_size, n_size };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, f32_gemv_kernel.?, iv, f32_gemv_cfgs[slot].?, s));
    var result = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    if (!f32_gemv_engaged) {
        f32_gemv_engaged = true;
        @import("log.zig").info("[mtp-qmv] row-identical f32 gemv rows engaged K={d} N={d}\n", .{ k, n });
    }
    var shape: [8]c_int = undefined;
    @memcpy(shape[0..xs.len], xs);
    shape[xs.len - 1] = n;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, result, &shape, xs.len, s));
    return out;
}

fn serialRows(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, gs: ?c_int, rows: c_int) !mlx.mlx_array {
    const xs = mlx.getShape(x);
    const k = xs[xs.len - 1];
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &.{ rows, k }, 2, s));
    var parts: [32]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (parts) |part| {
        if (part.ctx != null) _ = mlx.mlx_array_free(part);
    };
    for (0..@intCast(rows)) |row| {
        var xr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xr);
        try mlx.check(mlx.mlx_slice(&xr, flat, &.{ @intCast(row), 0 }, 2, &.{ @intCast(row + 1), k }, 2, &.{ 1, 1 }, 2, s));
        parts[row] = mlx.mlx_array_new();
        if (gs) |group_size| {
            try mlx.check(mlx.mlx_quantized_matmul(&parts[row], xr, w, sc, bi, true, mlx.mlx_optional_int.some(group_size), mlx.mlx_optional_int.some(8), "affine", s));
        } else {
            try mlx.check(mlx.mlx_matmul(&parts[row], xr, w, s));
        }
    }
    const pv = mlx.mlx_vector_array_new_data(&parts, @intCast(rows));
    defer _ = mlx.mlx_vector_array_free(pv);
    var joined = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(joined);
    try mlx.check(mlx.mlx_concatenate_axis(&joined, pv, 0, s));
    var shape: [8]c_int = undefined;
    @memcpy(shape[0..xs.len], xs);
    shape[xs.len - 1] = mlx.getShape(w)[if (gs == null) @as(usize, 1) else 0];
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, joined, &shape, xs.len, s));
    return out;
}

test "qmv8 gs32 shared rows retain every serial qmv output bit" {
    try testSharedRows(32);
}

test "qmv8 gs64 shared rows retain every serial qmv output bit" {
    try testSharedRows(64);
}

test "qmv8 gs128 shared rows retain every serial qmv output bit" {
    try testSharedRows(128);
}

fn testSharedRows(gs: c_int) !void {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const a = std.testing.allocator;
    var random = std.Random.DefaultPrng.init(0x8a57a);
    const rng = random.random();
    for ([_][2]c_int{ .{ 4096, 8192 }, .{ 12288, 2560 }, .{ 2560, 6144 }, .{ 10240, 2560 }, .{ 6144, 2560 }, .{ 640, 2560 }, .{ 512, 640 }, .{ 10240, 384 }, .{ 7, 256 }, .{ 8, 128 }, .{ 1, 128 }, .{ 13, 512 }, .{ 9, gs }, .{ 8, gs * 3 }, .{ 8, gs * 9 }, .{ 48, 2560 } }) |dims| {
        const n = dims[0];
        const k = dims[1];
        const wh = try a.alloc(u32, @intCast(n * @divExact(k, 4)));
        defer a.free(wh);
        const sh = try a.alloc(u16, @intCast(n * @divExact(k, gs)));
        defer a.free(sh);
        const bh = try a.alloc(u16, sh.len);
        defer a.free(bh);
        for (wh) |*v| v.* = rng.int(u32);
        for (sh, bh) |*sc, *bi| {
            sc.* = @truncate(@as(u32, @bitCast(0.001 + rng.float(f32) * 0.001)) >> 16);
            bi.* = @truncate(@as(u32, @bitCast(-0.12 + rng.float(f32) * 0.01)) >> 16);
        }
        const w = mlx.mlx_array_new_data(wh.ptr, &[_]c_int{ n, @divExact(k, 4) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(w);
        const sc = mlx.mlx_array_new_data(sh.ptr, &[_]c_int{ n, @divExact(k, gs) }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(sc);
        const bi = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{ n, @divExact(k, gs) }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(bi);
        for (2..10) |rows| {
            const xh = try a.alloc(u16, rows * @as(usize, @intCast(k)));
            defer a.free(xh);
            for (xh) |*v| v.* = @truncate(@as(u32, @bitCast((rng.float(f32) - 0.5) * 0.4)) >> 16);
            const x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), k }, 2, .bfloat16);
            defer _ = mlx.mlx_array_free(x);
            const tile_before = rows_arm_calls[0];
            const together = (try matmul(s, x, w, sc, bi)) orelse {
                std.debug.print("qmv8 declined gs={d} N={d} K={d} M={d}\n", .{ gs, n, k, rows });
                return error.KernelDeclined;
            };
            defer _ = mlx.mlx_array_free(together);
            // The trunk geometries leave the two-tile arm from five rows on.
            if (rows >= 5 and rows <= 7 and @mod(k, 256) == 0 and @mod(n, 16) == 0) try std.testing.expectEqual(tile_before, rows_arm_calls[0]);
            for (0..rows) |row| {
                var xr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(xr);
                var yr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(yr);
                try mlx.check(mlx.mlx_slice(&xr, x, &[_]c_int{ @intCast(row), 0 }, 2, &[_]c_int{ @intCast(row + 1), k }, 2, &[_]c_int{ 1, 1 }, 2, s));
                try mlx.check(mlx.mlx_slice(&yr, together, &[_]c_int{ @intCast(row), 0 }, 2, &[_]c_int{ @intCast(row + 1), n }, 2, &[_]c_int{ 1, 1 }, 2, s));
                var reference = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(reference);
                try mlx.check(mlx.mlx_quantized_matmul(&reference, xr, w, sc, bi, true, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(8), "affine", s));
                try mlx.check(mlx.mlx_array_eval(reference));
                try mlx.check(mlx.mlx_array_eval(yr));
                const rp = mlx.mlx_array_data_bfloat16(reference) orelse return error.Unreadable;
                const yp = mlx.mlx_array_data_bfloat16(yr) orelse return error.Unreadable;
                for (0..@intCast(n)) |i| {
                    if (rp[i] != yp[i]) {
                        std.debug.print("qmv8 gs={d} N={d} K={d} M={d} row={d} col={d}: serial={x} shared={x}\n", .{ gs, n, k, rows, row, i, rp[i], yp[i] });
                        return error.QmvRowNotIdentical;
                    }
                }
            }
        }
    }
}

test "f32 gemv rows retain every serial MLX gemv output bit on a transposed weight" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const a = std.testing.allocator;
    var random = std.Random.DefaultPrng.init(0x7a11e);
    const rng = random.random();
    for ([_][2]c_int{ .{ 256, 4096 }, .{ 512, 8192 }, .{ 128, 2048 } }) |dims| {
        const n = dims[0];
        const k = dims[1];
        const wh = try a.alloc(f32, @intCast(n * k));
        defer a.free(wh);
        for (wh) |*v| v.* = (rng.float(f32) - 0.5) * 0.08;
        const raw = mlx.mlx_array_new_data(wh.ptr, &[_]c_int{ n, k }, 2, .float32);
        defer _ = mlx.mlx_array_free(raw);
        var wb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wb);
        try mlx.check(mlx.mlx_astype(&wb, raw, .bfloat16, s));
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, wb, s));
        // The served router: a bf16 [out, in] weight transposed at load, then widened to f32.
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        try mlx.check(mlx.mlx_astype(&w, wt, .float32, s));
        try mlx.check(mlx.mlx_array_eval(w));
        for (2..9) |rows| {
            const xh = try a.alloc(f32, rows * @as(usize, @intCast(k)));
            defer a.free(xh);
            for (xh) |*v| v.* = (rng.float(f32) - 0.5) * 12.0;
            const x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ 1, @intCast(rows), k }, 3, .float32);
            defer _ = mlx.mlx_array_free(x);
            const together = (try f32GemvRows(s, x, w)) orelse return error.KernelDeclined;
            defer _ = mlx.mlx_array_free(together);
            try mlx.check(mlx.mlx_array_eval(together));
            const tp = mlx.mlx_array_data_float32(together) orelse return error.Unreadable;
            for (0..rows) |row| {
                const xr = mlx.mlx_array_new_data(xh[row * @as(usize, @intCast(k)) ..].ptr, &[_]c_int{ 1, k }, 2, .float32);
                defer _ = mlx.mlx_array_free(xr);
                var reference = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(reference);
                try mlx.check(mlx.mlx_matmul(&reference, xr, w, s));
                try mlx.check(mlx.mlx_array_eval(reference));
                const rp = mlx.mlx_array_data_float32(reference) orelse return error.Unreadable;
                for (0..@intCast(n)) |i| {
                    if (@as(u32, @bitCast(rp[i])) != @as(u32, @bitCast(tp[row * @as(usize, @intCast(n)) + i]))) {
                        std.debug.print("f32 gemv rows N={d} K={d} M={d} row={d} col={d}: serial={d} shared={d}\n", .{ n, k, rows, row, i, rp[i], tp[row * @as(usize, @intCast(n)) + i] });
                        return error.GemvRowNotIdentical;
                    }
                }
            }
        }
    }
}
