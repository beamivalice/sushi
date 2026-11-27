//! Engram's gate chain (`fwd.engram`) in a few launches, each element the ops' bytes: the per-head norms of the table
//! rows in one launch (`rms_single_row` per head, each head's weight read as stored), the value's bias add and square
//! in one, and the write into the streams in one (the value's RMS, `γ`, each stream's gate from its key·query sum,
//! `σ_c`, the live mask and the add into the streams), all in the ops' f32 order with no fma contraction. The linears
//! and the two row sums (MLX's reduce) stay as they were; the key and query norms are `hy4_norm`'s launches.
const std = @import("std");
const mlx = @import("../mlx.zig");
const Arr = mlx.mlx_array;
const kcfg = @import("hy4_kcfg.zig");

const max_heads = 32;
const max_streams = 8;

/// MLX's Sigmoid and Maximum functors on a float (`unary_ops.h`, `binary_ops.h`; the engine's `sushi_*`).
pub const header: [:0]const u8 =
    \\inline float hy4_sigmoid(float x) {
    \\  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    \\  return (x < 0) ? y : 1 - y;
    \\}
    \\inline float hy4_maximum(float x, float y) {
    \\  if (metal::isnan(x)) return x;
    \\  return x > y ? x : y;
    \\}
;

/// The head norms' body after `HEAD_W` (the head's weight pointer) is chosen: `hy4_norm`'s single-row norm.
pub const heads_body =
    \\constexpr int N_READS = 4;
    \\constexpr int SIMD_SIZE = 32;
    \\uint gid = threadgroup_position_in_grid.x;
    \\uint lid = thread_position_in_threadgroup.x;
    \\uint simd_lane_id = thread_index_in_simdgroup;
    \\uint simd_group_id = simdgroup_index_in_threadgroup;
    \\const uint axis_size = uint(D);
    \\uint t = gid / uint(NH);
    \\uint j = gid % uint(NH);
    \\threadgroup float local_inv_mean[1];
    \\threadgroup float local_sums[SIMD_SIZE];
    \\float acc = 0;
    \\float thread_x[N_READS];
    \\const device T* xr = x + (size_t)t * (size_t)x_strides[0] + (size_t)j * (size_t)x_strides[1];
    \\for (int i = 0; i < N_READS; i++) {
    \\  uint c = lid * N_READS + i;
    \\  thread_x[i] = c < axis_size ? static_cast<float>(xr[(size_t)c * (size_t)x_strides[2]]) : 0;
    \\  acc += thread_x[i] * thread_x[i];
    \\}
    \\acc = simd_sum(acc);
    \\if (simd_group_id == 0) {
    \\  local_sums[simd_lane_id] = 0;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (simd_lane_id == 0) {
    \\  local_sums[simd_group_id] = acc;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (simd_group_id == 0) {
    \\  acc = simd_sum(local_sums[simd_lane_id]);
    \\  if (simd_lane_id == 0) {
    \\    local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + as_type<float>(EPS));
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\device float* o = out + (size_t)gid * (size_t)D;
    \\for (int i = 0; i < N_READS; i++) {
    \\  uint c = lid * N_READS + i;
    \\  if (c < axis_size) {
    \\    o[c] = static_cast<float>(hw[c]) * static_cast<float>(thread_x[i] * local_inv_mean[0]);
    \\  }
    \\}
;

/// `v = lin + b` (the bias widened) and `v · v`.
pub const value_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint i = thread_position_in_grid.x;
    \\uint t = thread_position_in_grid.y;
    \\size_t at = (size_t)t * (size_t)threads_per_grid.x + i;
    \\float v = lin[at] + static_cast<float>(b[i]);
    \\out_v[at] = v;
    \\out_sq[at] = v * v;
;

/// The delta's body (no contraction) after the stream's sum and scale pointers (`ds`, `sc`) are chosen: `INV_N` is
/// `1 / D` as the mean takes it (in f64, stored f32), `INV_SQRT_D` the f32 `1 / √D`, both f32 bits.
pub const delta_body =
    \\uint i = thread_position_in_grid.x;
    \\uint c = thread_position_in_grid.y;
    \\uint t = thread_position_in_grid.z;
    \\float mean = vsum[t] * as_type<float>(INV_N);
    \\float rs = metal::precise::rsqrt(mean + as_type<float>(EPS));
    \\float vr = v[(size_t)t * (size_t)D + i] * rs;
    \\float y = vr * static_cast<float>(gamma[i]);
    \\float dot = ds[t] * as_type<float>(INV_SQRT_D);
    \\float mag = metal::precise::sqrt(hy4_maximum(metal::abs(dot), 1e-6f));
    \\float sg = static_cast<float>((dot > 0.0f) - (dot < 0.0f));
    \\float gate = hy4_sigmoid(mag * sg);
    \\float with_scale = gate + static_cast<float>(sc[i]);
    \\float delta = with_scale * y;
    \\float masked = delta * live[t];
    \\size_t at = ((size_t)t * (size_t)NC + c) * (size_t)D + i;
    \\size_t ht = (size_t)t * (size_t)h_strides[0] + (size_t)c * (size_t)h_strides[1] + (size_t)i * (size_t)h_strides[2];
    \\out[at] = static_cast<float>(h[ht]) + masked;
;

const Made = struct { n: usize = 0, kernel: ?mlx.mlx_fast_metal_kernel = null };
var heads_kernels: [max_heads + 1]Made = @splat(.{});
var delta_kernels: [max_streams + 1]Made = @splat(.{});
var value_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn newKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, header, false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

const gpa = std.heap.c_allocator;

const Kind = enum { heads, value, delta };
/// `a`, `b`, `c` the shape (rows, heads or streams, width), `dt` the rows' dtype, `f` ε's bits.
const Key = struct { kind: Kind, a: c_int, b: c_int, c: c_int = 0, dt: mlx.mlx_dtype = .float32, f: c_int = 0 };
var cfgs: kcfg.Cache(Key) = .{};

fn build(k: Key) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    switch (k.kind) {
        .heads => {
            const group = 32 * @divFloor(@divFloor(k.c + 3, 4) + 31, 32);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, k.b * k.c }, 2, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.a * k.b * group, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, group, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", k.dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", k.c));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NH", k.b));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "EPS", k.f));
        },
        .value => {
            for (0..2) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, k.b }, 2, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.b, k.a, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(k.b, 256), 1, 1));
        },
        .delta => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, k.b, k.c }, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.c, k.b, k.a));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(k.c, 256), 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", k.c));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NC", k.b));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "EPS", k.f));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "INV_N", kcfg.bits(@floatCast(1.0 / @as(f64, @floatFromInt(k.c))))));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "INV_SQRT_D", kcfg.bits(1.0 / @sqrt(@as(f32, @floatFromInt(k.c))))));
        },
    }
    return cfg;
}

/// A kernel whose inputs include `n` numbered ones (`<stem>0..`), the body choosing the one at `index` as `chosen`.
fn numberedKernel(slot: *Made, name: [*:0]const u8, fixed: []const [*:0]const u8, stems: []const []const u8, chosen: []const []const u8, index: []const u8, n: usize, body: []const u8, no_contract: bool) !mlx.mlx_fast_metal_kernel {
    if (slot.kernel) |k| return k;
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    // The pragma opens the body (it must lead its block); the norm's body keeps MLX's own contraction.
    if (no_contract) try src.appendSlice(gpa, "#pragma clang fp contract(off)\n");
    var names: std.ArrayList([*:0]const u8) = .empty;
    defer {
        for (names.items[fixed.len..]) |p| gpa.free(std.mem.span(p));
        names.deinit(gpa);
    }
    try names.appendSlice(gpa, fixed);
    for (stems, chosen) |stem, as| {
        try src.print(gpa, "auto {s} = ", .{as});
        for (0..n) |i| {
            if (i + 1 < n) try src.print(gpa, "{s} == {d} ? {s}{d} : ", .{ index, i, stem, i }) else try src.print(gpa, "{s}{d};\n", .{ stem, i });
            const owned = try std.fmt.allocPrintSentinel(gpa, "{s}{d}", .{ stem, i }, 0);
            names.append(gpa, owned.ptr) catch |e| {
                gpa.free(owned);
                return e;
            };
        }
    }
    try src.appendSlice(gpa, body);
    const text = try gpa.dupeSentinel(u8, src.items, 0);
    defer gpa.free(text);
    const k = try newKernel(name, names.items, &.{"out"}, text);
    slot.* = .{ .n = n, .kernel = k };
    return k;
}

fn applyOuts(s: mlx.mlx_stream, k: mlx.mlx_fast_metal_kernel, inputs: []const Arr, cfg: mlx.mlx_fast_metal_kernel_config, outs: []Arr) !void {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    for (outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(o, ov, i)) catch |e| {
            for (outs[0 .. i + 1]) |prev| _ = mlx.mlx_array_free(prev);
            return e;
        };
    }
}

/// Whether the launches take a forward's Engram: a GPU stream, f32 rows [T, H, hd] and streams [T, C, D], H and C in
/// range, every width within the single-row norm.
pub fn applies(s: mlx.mlx_stream, h: Arr, rows: Arr, heads: usize, streams: usize) bool {
    const hs = mlx.getShape(h);
    const rs = mlx.getShape(rows);
    return mlx.streamIsGpu(s) and hs.len == 3 and rs.len == 3 and mlx.mlx_array_dtype(rows) == .float32 and
        (mlx.mlx_array_dtype(h) == .float32 or mlx.mlx_array_dtype(h) == .bfloat16) and heads >= 1 and heads <= max_heads and
        streams >= 1 and streams <= max_streams and rs[1] == heads and hs[1] == streams and rs[2] <= 4096 and hs[2] <= 4096;
}

/// Each head's rows normed by its own weight (`norm(r_j, w_j, eps)`), joined: `rows` [T, H, hd] → [T, H·hd] f32.
pub fn headNorms(s: mlx.mlx_stream, rows: Arr, weights: []const Arr, eps: f32) !Arr {
    const sh = mlx.getShape(rows);
    const nh = weights.len;
    const k = try numberedKernel(&heads_kernels[nh], "sushi_hy4_engram_heads", &.{"x"}, &.{"w"}, &.{"hw"}, "(threadgroup_position_in_grid.x % uint(NH))", nh, heads_body, false);
    const cfg = try cfgs.get(.{ .kind = .heads, .a = sh[0], .b = sh[1], .c = sh[2], .dt = mlx.mlx_array_dtype(rows), .f = kcfg.bits(eps) }, build);
    var ins: [1 + max_heads]Arr = undefined;
    ins[0] = rows;
    @memcpy(ins[1..][0..nh], weights);
    var out: [1]Arr = undefined;
    try applyOuts(s, k, ins[0 .. 1 + nh], cfg, &out);
    return out[0];
}

/// `v = lin + b` and `v · v`, `lin` [T, D] f32, `b` [D].
pub fn valueSq(s: mlx.mlx_stream, lin: Arr, b: Arr) !struct { v: Arr, sq: Arr } {
    if (value_kernel == null) value_kernel = try newKernel("sushi_hy4_engram_value", &.{ "lin", "b" }, &.{ "out_v", "out_sq" }, value_source);
    const sh = mlx.getShape(lin);
    const cfg = try cfgs.get(.{ .kind = .value, .a = sh[0], .b = sh[1] }, build);
    var out: [2]Arr = undefined;
    try applyOuts(s, value_kernel.?, &.{ lin, b }, cfg, &out);
    return .{ .v = out[0], .sq = out[1] };
}

/// The streams after Engram: `h + live · (σ(sign(d_c)·√max(|d_c|, 1e-6)) + σ_c) · γ ⊙ v·rsqrt(mean(v²) + ε)` with
/// `d_c = dsums[c] / √D`, `h` [T, C, D] → f32. `vsum` and each `dsums[c]` are the row sums [T, 1].
pub fn delta(s: mlx.mlx_stream, h: Arr, v: Arr, vsum: Arr, gamma: Arr, dsums: []const Arr, scales: []const Arr, live: Arr, eps: f32) !Arr {
    const sh = mlx.getShape(h);
    const nc = dsums.len;
    const k = try numberedKernel(&delta_kernels[nc], "sushi_hy4_engram_delta", &.{ "h", "v", "vsum", "gamma", "live" }, &.{ "d", "s" }, &.{ "ds", "sc" }, "thread_position_in_grid.y", nc, delta_body, true);
    const d = sh[2];
    const cfg = try cfgs.get(.{ .kind = .delta, .a = sh[0], .b = sh[1], .c = d, .f = kcfg.bits(eps) }, build);
    var ins: [5 + 2 * max_streams]Arr = undefined;
    for ([_]Arr{ h, v, vsum, gamma, live }, 0..) |a, i| ins[i] = a;
    @memcpy(ins[5..][0..nc], dsums);
    @memcpy(ins[5 + nc ..][0..nc], scales);
    var out: [1]Arr = undefined;
    try applyOuts(s, k, ins[0 .. 5 + 2 * nc], cfg, &out);
    return out[0];
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;
const Ops = @import("../mlx_ops.zig").Ops;

fn randomArr(o: *Ops, shape: []const c_int, dt: mlx.mlx_dtype, seed: u64, scale: f64) !Arr {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(v);
    var r = std.Random.DefaultPrng.init(seed);
    for (v) |*d| d.* = @floatCast(r.random().floatNorm(f64) * scale);
    return o.cast(try o.own(mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32)), dt);
}

fn hostF32(o: *Ops, a: Arr) ![]const u8 {
    const c = try o.contiguous(a);
    try mlx.check(mlx.mlx_array_eval(c));
    return std.mem.sliceAsBytes((mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..mlx.mlx_array_size(c)]);
}

test "hy4 engram: the gate chain in its launches gives the bytes of the ops, at the release's 16 heads of 96, 3,072 wide over 4 streams, and the tiny's, a decode row and a block, a dead row among them" {
    try mlx.requireGpu();
    const fwd = @import("hy4_forward.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    defer fwd.chain_kernels_on = true;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    const Case = struct { t: c_int, heads: usize, hd: c_int, d: c_int };
    for ([_]Case{ .{ .t = 1, .heads = 16, .hd = 96, .d = 3072 }, .{ .t = 6, .heads = 16, .hd = 96, .d = 3072 }, .{ .t = 3, .heads = 4, .hd = 16, .d = 128 } }, 0..) |c, ci| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const seed: u64 = 1000 * (ci + 1);
        const u_dim = @as(c_int, @intCast(c.heads)) * c.hd;
        var head_norms: [16]Arr = undefined;
        for (head_norms[0..c.heads], 0..) |*a, j| a.* = try randomArr(&o, &.{c.hd}, .bfloat16, seed + j, 0.5);
        var key_w: [4]Arr = undefined;
        var key_b: [4]Arr = undefined;
        var norm_key: [4]Arr = undefined;
        var norm_query: [4]Arr = undefined;
        var scales: [4]Arr = undefined;
        for (0..4) |s_| {
            const b = seed + 100 + 10 * s_;
            key_w[s_] = try randomArr(&o, &.{ c.d, u_dim }, .bfloat16, b, 0.05);
            key_b[s_] = try randomArr(&o, &.{c.d}, .bfloat16, b + 1, 0.1);
            norm_key[s_] = try randomArr(&o, &.{c.d}, .bfloat16, b + 2, 0.5);
            norm_query[s_] = try randomArr(&o, &.{c.d}, .bfloat16, b + 3, 0.5);
            scales[s_] = try randomArr(&o, &.{c.d}, .bfloat16, b + 4, 0.2);
        }
        const w: fwd.Engram = .{
            .head_norms = head_norms[0..c.heads],
            .key_w = &key_w,
            .key_b = &key_b,
            .norm_key = &norm_key,
            .norm_query = &norm_query,
            .scales = &scales,
            .value_w = try randomArr(&o, &.{ c.d, u_dim }, .bfloat16, seed + 200, 0.05),
            .value_b = try randomArr(&o, &.{c.d}, .bfloat16, seed + 201, 0.1),
            .gamma = try randomArr(&o, &.{c.d}, .bfloat16, seed + 202, 0.5),
        };
        const h = try randomArr(&o, &.{ c.t, 4, c.d }, .float32, seed + 300, 1);
        const rows = try randomArr(&o, &.{ c.t, @intCast(c.heads), c.hd }, .float32, seed + 301, 1);
        const lv = try testing.allocator.alloc(f32, @intCast(c.t));
        defer testing.allocator.free(lv);
        for (lv, 0..) |*x, i| x.* = if (i == 1) 0 else 1;
        const live = try o.own(mlx.mlx_array_new_data(lv.ptr, &[_]c_int{c.t}, 1, .float32));
        fwd.chain_kernels_on = false;
        const want = try hostF32(&o, try fwd.engramAdd(&o, h, rows, live, w, null));
        fwd.chain_kernels_on = true;
        try testing.expect(applies(es, h, rows, c.heads, 4));
        const before = mlx.engine.stats().dispatches;
        const got = try fwd.engramAdd(&o, h, rows, live, w, null);
        try mlx.check(mlx.mlx_array_eval(got));
        // Head norms, the value linear, its bias and square, its sum; per stream the key linear, its norm, the query
        // norm times the key, the sum; the write. At the release's decode row each linear is its one-row launch.
        if (ci == 0) try testing.expectEqual(@as(u64, 4 + 4 * 4 + 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, want, try hostF32(&o, got));
    }
}
