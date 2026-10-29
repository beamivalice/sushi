//! The draft lattice's candidates in two dispatches: per row, the last `k` entries of MLX's
//! argpartition (a stable ascending merge sort with NaN above every number, so ties keep the
//! higher index) and their logits widened to FP32.
const std = @import("std");
const mlx = @import("../mlx.zig");
const Arr = mlx.mlx_array;

// Keys order floats as unsigned integers (-0 joins +0, every NaN is the maximum; no value keys
// to 0, which marks a taken or padding slot); a (key, index) pair compares lexicographically,
// which is the stable sort's order. Each round takes the largest pair still present.
const ROUND =
    \\uint bk = 0u, bi = 0u;
    \\for (uint e = 0; e < E; ++e) if (keys[e] != 0u && keys[e] >= bk) { bk = keys[e]; bi = ids[e]; }
    \\const uint sk = simd_max(bk);
    \\const uint si = simd_max(bk == sk ? bi : 0u);
    \\threadgroup uint* slot_k = tg_k + (r & 1u) * SG;
    \\threadgroup uint* slot_i = tg_i + (r & 1u) * SG;
    \\if (lane == 0) { slot_k[sg] = sk; slot_i[sg] = si; }
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\uint wk = 0u, wi = 0u;
    \\for (uint j = 0; j < SG; ++j) {
    \\    const uint ck = slot_k[j], ci = slot_i[j];
    \\    if (ck > wk || (ck == wk && ci > wi)) { wk = ck; wi = ci; }
    \\}
    \\for (uint e = 0; e < E; ++e) if (keys[e] == wk && ids[e] == wi) keys[e] = 0u;
;

// Pass 1: each threadgroup ranks one stretch of the row; its top K go to the candidate list.
const PASS1 =
    \\const uint row = threadgroup_position_in_grid.y;
    \\const uint blk = threadgroup_position_in_grid.x;
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\constexpr uint SG = 256 / 32;
    \\threadgroup uint tg_k[2 * SG];
    \\threadgroup uint tg_i[2 * SG];
    \\uint keys[E];
    \\uint ids[E];
    \\for (uint e = 0; e < E; ++e) {
    \\    const uint i = blk * 256u * E + e * 256u + tid;
    \\    ids[e] = i;
    \\    keys[e] = 0u;
    \\    if (i < uint(V)) {
    \\        const float f = float(logits[size_t(row) * V + i]);
    \\        const uint u = as_type<uint>(f == 0.0f ? 0.0f : f);
    \\        keys[e] = isnan(f) ? 0xffffffffu : ((u & 0x80000000u) ? ~u : (u | 0x80000000u));
    \\    }
    \\}
    \\for (uint r = 0; r < K; ++r) {
    ++ ROUND ++
    \\    if (tid == 0) {
    \\        const size_t at = (size_t(row) * NB + blk) * K + r;
    \\        part_keys[at] = wk;
    \\        part_ids[at] = wi;
    \\    }
    \\}
;

// Pass 2: one threadgroup per row ranks the NB * K candidates (at most one per thread).
const PASS2 =
    \\const uint row = threadgroup_position_in_grid.y;
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\constexpr uint SG = 1024 / 32;
    \\threadgroup uint tg_k[2 * SG];
    \\threadgroup uint tg_i[2 * SG];
    \\uint keys[E];
    \\uint ids[E];
    \\keys[0] = tid < NB * K ? part_keys[size_t(row) * NB * K + tid] : 0u;
    \\ids[0] = tid < NB * K ? part_ids[size_t(row) * NB * K + tid] : 0u;
    \\for (uint r = 0; r < K; ++r) {
    ++ ROUND ++
    \\    if (tid == 0) {
    \\        const size_t at = size_t(row) * K + (K - 1 - r);
    \\        cands[at] = int(wi);
    \\        unary[at] = float(logits[size_t(row) * V + wi]);
    \\    }
    \\}
;
var kernels: [2]?mlx.mlx_fast_metal_kernel = @splat(null);

pub const Candidates = struct { ids: Arr, unary: Arr };

fn kernel(slot: usize, name: [:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (kernels[slot]) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernels[slot] = k;
    return k;
}

/// `logits` `[1, m, V]` BF16 → candidate ids (int32) and logits (FP32), both `[1, m, k]`,
/// in argpartition's order; null where the kernels do not serve the shape.
pub fn apply(s: mlx.mlx_stream, logits: Arr, k: u32) !?Candidates {
    if (!mlx.streamIsGpu(s) or logits.ctx == null or mlx.mlx_array_dtype(logits) != .bfloat16) return null;
    const shape = mlx.getShape(logits);
    if (shape.len != 3 or shape[0] != 1 or shape[1] < 1 or shape[1] > 15 or k == 0 or k > 16 or shape[2] < 32 * @as(c_int, @intCast(k)) or shape[2] > 64 * 256 * 32) return null;
    const vocab: u32 = @intCast(shape[2]);
    const per_thread = std.math.divCeil(u32, vocab, 64 * 256) catch unreachable;
    const blocks = std.math.divCeil(u32, vocab, 256 * per_thread) catch unreachable;
    const rows = shape[1];
    const k1 = try kernel(0, "sushi_glm_draft_topk_blocks", &.{"logits"}, &.{ "part_keys", "part_ids" }, PASS1);
    const c1 = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c1);
    const part_shape = [_]c_int{ rows, @intCast(blocks * k) };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c1, &part_shape, 2, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c1, &part_shape, 2, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c1, @intCast(256 * blocks), rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c1, 256, 1, 1));
    inline for (.{ "K", "V", "E", "NB" }, .{ k, vocab, per_thread, blocks }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c1, name, @intCast(value)));
    const in1 = mlx.mlx_vector_array_new_data(&.{logits}, 1);
    defer _ = mlx.mlx_vector_array_free(in1);
    var out1 = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out1);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out1, k1, in1, c1, s));
    var part_keys = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part_keys);
    try mlx.check(mlx.mlx_vector_array_get(&part_keys, out1, 0));
    var part_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part_ids);
    try mlx.check(mlx.mlx_vector_array_get(&part_ids, out1, 1));

    const k2 = try kernel(1, "sushi_glm_draft_topk_merge", &.{ "logits", "part_keys", "part_ids" }, &.{ "cands", "unary" }, PASS2);
    const c2 = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c2);
    const out_shape = [_]c_int{ 1, rows, @intCast(k) };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c2, &out_shape, 3, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c2, &out_shape, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c2, 1024, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c2, 1024, 1, 1));
    inline for (.{ "K", "V", "E", "NB" }, .{ k, vocab, 1, blocks }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c2, name, @intCast(value)));
    const in2 = mlx.mlx_vector_array_new_data(&.{ logits, part_keys, part_ids }, 3);
    defer _ = mlx.mlx_vector_array_free(in2);
    var out2 = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out2);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out2, k2, in2, c2, s));
    var ids = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_vector_array_get(&ids, out2, 0));
    var unary = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(unary);
    try mlx.check(mlx.mlx_vector_array_get(&unary, out2, 1));
    return .{ .ids = ids, .unary = unary };
}

/// The staged reference: argpartition, the top slice, and the gathered logits.
pub fn reference(s: mlx.mlx_stream, logits: Arr, k: u32) !Candidates {
    const Ops = @import("glm5_model.zig").Ops;
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const shape = mlx.getShape(logits);
    const vocab = shape[2];
    const kk: c_int = @intCast(k);
    const part = try ops.slot();
    try mlx.check(mlx.mlx_argpartition_axis(part, logits, vocab - kk, 2, s));
    const top = try ops.slot();
    try mlx.check(mlx.mlx_slice(top, part.*, &.{ 0, 0, vocab - kk }, 3, &.{ 1, shape[1], vocab }, 3, &.{ 1, 1, 1 }, 3, s));
    const ids = try ops.cast(top.*, .int32);
    const taken = try ops.slot();
    try mlx.check(mlx.mlx_take_along_axis(taken, logits, ids, 2, s));
    return .{ .ids = try ops.result(ids), .unary = try ops.result(try ops.cast(taken.*, .float32)) };
}

fn expectSame(s: mlx.mlx_stream, logits: Arr, k: u32) !void {
    const want = try reference(s, logits, k);
    defer for ([_]Arr{ want.ids, want.unary }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const got = (try apply(s, logits, k)) orelse return error.TestExpectedTopk;
    defer for ([_]Arr{ got.ids, got.unary }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for ([_]Arr{ want.ids, want.unary, got.ids, got.unary }) |a| try mlx.check(mlx.mlx_array_eval(a));
    const n = mlx.mlx_array_size(want.ids);
    try std.testing.expectEqualSlices(i32, mlx.mlx_array_data_int32(want.ids).?[0..n], mlx.mlx_array_data_int32(got.ids).?[0..n]);
    const wu: [*]const u32 = @ptrCast(mlx.mlx_array_data_float32(want.unary).?);
    const gu: [*]const u32 = @ptrCast(mlx.mlx_array_data_float32(got.unary).?);
    try std.testing.expectEqualSlices(u32, wu[0..n], gu[0..n]);
}

test "GLM draft top-k equals argpartition's last entries on ties, signed zeros, infinities and NaN" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const vocab = 154880;
    const rows = 2;
    const values = try std.testing.allocator.alloc(u16, rows * vocab);
    defer std.testing.allocator.free(values);
    // BF16 bit patterns: a handful of tied levels near the top, signed zeros, -inf, and noise.
    const levels = [_]u16{ 0x4120, 0x4120, 0x4118, 0x4110, 0x0000, 0x8000, 0xff80, 0x3f80 };
    var prng = std.Random.DefaultPrng.init(17);
    const random = prng.random();
    for ([_]u2{ 0, 1, 2 }) |case| {
        for (values) |*v| {
            const pick = random.uintLessThan(u32, 64);
            v.* = if (pick < levels.len) levels[pick] else random.int(u16) & 0x40ff;
            if (case == 1) v.* = levels[random.uintLessThan(usize, 3)];
        }
        if (case == 2) {
            values[5] = 0x7fc0;
            values[vocab + 77] = 0x7f80;
            values[vocab + 78] = 0xffc1;
        }
        const logits = mlx.mlx_array_new_data(values.ptr, &.{ 1, rows, vocab }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(logits);
        try expectSame(s, logits, 16);
    }
    const small = mlx.mlx_array_new_data(values.ptr, &.{ 1, 1, 512 }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(small);
    try expectSame(s, small, 16);
    const narrow = mlx.mlx_array_new_data(values.ptr, &.{ 1, 1, 511 }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(narrow);
    try std.testing.expect((try apply(s, narrow, 16)) == null);
}
