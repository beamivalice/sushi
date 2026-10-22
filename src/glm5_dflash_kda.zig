//! Vector-gated KDA recurrence over parent-indexed verification rows.
const std = @import("std");
const mlx = @import("mlx.zig");
const primitive = @import("glm5_next.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;

const SOURCE =
    \\constexpr int N = Dk / 32;
    \\const int head = int(thread_position_in_grid.z);
    \\const int dv = int(thread_position_in_grid.y);
    \\const int lane = int(thread_position_in_threadgroup.x);
    \\float saved[W][N];
    \\for (int row = 0; row < W; ++row) {
    \\  const int parent = parents[row];
    \\  const device InT* q_ = q + (row * H + head) * Dk;
    \\  const device InT* k_ = k + (row * H + head) * Dk;
    \\  float state[N];
    \\  float memory = 0.0f;
    \\  for (int i = 0; i < N; ++i) {
    \\    const int key = N * lane + i;
    \\    state[i] = parent < 0 ? state_in[(head * Dv + dv) * Dk + key] : saved[parent][i];
    \\    state[i] = state[i] * decay[(row * H + head) * Dk + key];
    \\    memory += state[i] * k_[key];
    \\  }
    \\  memory = simd_sum(memory);
    \\  const auto delta = (v[(row * H + head) * Dv + dv] - memory) * beta[row * H + head];
    \\  float output = 0.0f;
    \\  for (int i = 0; i < N; ++i) {
    \\    const int key = N * lane + i;
    \\    state[i] = state[i] + k_[key] * delta;
    \\    output += state[i] * q_[key];
    \\    saved[row][i] = state[i];
    \\  }
    \\  output = simd_sum(output);
    \\  if (thread_index_in_simdgroup == 0) y[(row * H + head) * Dv + dv] = OutT(output);
    \\}
;
var kernel: ?mlx.mlx_fast_metal_kernel = null;
var leaf_kernel: ?mlx.mlx_fast_metal_kernel = null;
pub fn cachedLeafRow(parents: []const i32) u32 {
    var row: u32 = 0;
    for (parents[1..], 1..) |parent, child| if (parent == row) {
        row = @intCast(child);
    };
    return row;
}
pub const LeafResult = struct {
    y: Arr,
    state: Arr = .{ .ctx = null },
    row: u32,
    pub fn deinit(self: LeafResult) void {
        _ = mlx.mlx_array_free(self.y);
        if (self.state.ctx != null) _ = mlx.mlx_array_free(self.state);
    }
};
pub fn recurrent(input: primitive.KdaInputs, parents: []const i32, stream: mlx.mlx_stream) !Arr {
    const result = try recurrentImpl(input, parents, stream, false);
    return result.y;
}
pub fn recurrentLeaf(input: primitive.KdaInputs, parents: []const i32, stream: mlx.mlx_stream) !LeafResult {
    if (parents.len == 0 or parents.len > 3) return error.InvalidGlmDraftTree;
    return recurrentImpl(input, parents, stream, true);
}

fn recurrentImpl(input: primitive.KdaInputs, parents: []const i32, stream: mlx.mlx_stream, keep_leaf: bool) !LeafResult {
    if (!mlx.streamIsGpu(stream)) return error.KdaGpuRequired;
    const q = mlx.getShape(input.q);
    const v = mlx.getShape(input.v);
    if (q.len != 4 or v.len != 4 or q[0] != 1 or q[1] < 1 or q[1] > 16 or q[2] < 1 or q[3] < 32 or @mod(q[3], 32) != 0 or v[3] < 4 or @mod(v[3], 4) != 0 or
        !std.mem.eql(c_int, q[0..3], v[0..3]) or !std.mem.eql(c_int, q, mlx.getShape(input.k)) or !std.mem.eql(c_int, q, mlx.getShape(input.decay)) or
        !std.mem.eql(c_int, q[0..3], mlx.getShape(input.beta)) or !std.mem.eql(c_int, &.{ 1, q[2], v[3], q[3] }, mlx.getShape(input.state)) or parents.len != q[1]) return error.InvalidKdaShape;
    const dtype = mlx.mlx_array_dtype(input.q);
    const beta_type = mlx.mlx_array_dtype(input.beta);
    if ((dtype != .bfloat16 and dtype != .float32) or mlx.mlx_array_dtype(input.k) != dtype or mlx.mlx_array_dtype(input.v) != dtype or
        mlx.mlx_array_dtype(input.state) != .float32 or mlx.mlx_array_dtype(input.decay) != .float32 or (beta_type != .bfloat16 and beta_type != .float32)) return error.InvalidKdaDtype;
    if (parents[0] != -1) return error.InvalidGlmDraftTree;
    for (parents[1..], 1..) |parent, row| if (parent < 0 or parent >= row) return error.InvalidGlmDraftTree;
    const selected_kernel = if (keep_leaf) &leaf_kernel else &kernel;
    if (selected_kernel.* == null) {
        const names = [_][*:0]const u8{ "q", "k", "v", "decay", "beta", "state_in", "parents" };
        const outputs: []const [*:0]const u8 = if (keep_leaf) &.{ "y", "leaf_state" } else &.{"y"};
        const iv = mlx.mlx_vector_string_new_data(&names, names.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(outputs.ptr, outputs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const stored: [:0]const u8 = SOURCE ++ "for(int i=0;i<N;++i) leaf_state[(head*Dv+dv)*Dk+N*lane+i]=saved[KEEP_ROW][i];";
        selected_kernel.* = mlx.mlx_fast_metal_kernel_new(if (keep_leaf) "sushi_glm_kda_tree_leaf" else "sushi_glm_kda_tree", iv, ov, if (keep_leaf) stored else SOURCE, "", true, false);
        if (selected_kernel.*.?.ctx == null) {
            selected_kernel.* = null;
            return error.MetalKernelCompileFailed;
        }
    }
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, v.ptr, 4, dtype));
    const row = cachedLeafRow(parents);
    if (keep_leaf) {
        const shape = mlx.getShape(input.state);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, shape.ptr, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KEEP_ROW", @intCast(row)));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, v[3], q[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", q[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", v[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "H", q[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "W", q[1]));
    const par = mlx.mlx_array_new_data(parents.ptr, &[_]c_int{q[1]}, 1, .int32);
    defer _ = mlx.mlx_array_free(par);
    const arrays = [_]Arr{ input.q, input.k, input.v, input.decay, input.beta, input.state, par };
    const inputs = mlx.mlx_vector_array_new_data(&arrays, arrays.len);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, selected_kernel.*.?, inputs, config, stream));
    var result = LeafResult{ .y = mlx.mlx_array_new(), .row = row };
    errdefer result.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&result.y, outputs, 0));
    if (keep_leaf) {
        result.state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&result.state, outputs, 1));
    }
    return result;
}

test "GLM DFlash KDA tree follows per-channel parent state exactly" {
    const s = mlx.gpuStream();
    inline for (.{ .{ 5, 2, 4 }, .{ 16, 64, 128 } }) |geometry| {
        const R = geometry[0];
        const H = geometry[1];
        const DV = geometry[2];
        var parents: [R]i32 = undefined;
        parents[0] = -1;
        for (1..R) |row| parents[row] = @intCast((row - 1) / 2);
        for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dtype| {
            var ops = Ops{ .s = s };
            defer ops.deinit();
            var q_data: [R * H * 128]f32 = undefined;
            var k_data: @TypeOf(q_data) = undefined;
            var g_data: @TypeOf(q_data) = undefined;
            var v_data: [R * H * DV]f32 = undefined;
            var beta_data: [R * H]f32 = undefined;
            for (&q_data, &k_data, &g_data, 0..) |*q, *k, *g, i| {
                q.* = @as(f32, @floatFromInt(i % 13)) / 64 - 0.1;
                k.* = @as(f32, @floatFromInt(i % 17)) / 96 - 0.08;
                g.* = 0.1 + @as(f32, @floatFromInt(i % 29)) / 36;
            }
            for (&v_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 19)) / 32 - 0.2;
            for (&beta_data, 0..) |*v, i| v.* = 0.1 + @as(f32, @floatFromInt(i % 13)) / 16;
            const q = try ops.cast(try ops.own(mlx.mlx_array_new_data(&q_data, &[_]c_int{ 1, R, H, 128 }, 4, .float32)), dtype);
            const k = try ops.cast(try ops.own(mlx.mlx_array_new_data(&k_data, &[_]c_int{ 1, R, H, 128 }, 4, .float32)), dtype);
            const v = try ops.cast(try ops.own(mlx.mlx_array_new_data(&v_data, &[_]c_int{ 1, R, H, DV }, 4, .float32)), dtype);
            const decay = try ops.own(mlx.mlx_array_new_data(&g_data, &[_]c_int{ 1, R, H, 128 }, 4, .float32));
            const beta = try ops.cast(try ops.own(mlx.mlx_array_new_data(&beta_data, &[_]c_int{ 1, R, H }, 3, .float32)), dtype);
            const initial = try ops.binary(.mul, try ops.ones(&.{ 1, H, DV, 128 }, .float32), try ops.scalar(0.07, .float32));
            const all = try ops.own(try recurrent(.{ .q = q, .k = k, .v = v, .decay = decay, .beta = beta, .state = initial }, &parents, s));
            var states: [R]Arr = undefined;
            for (parents, 0..) |parent, row| {
                const start: c_int = @intCast(row);
                const one = try primitive.kda(.{ .q = try ops.slice(q, 1, start, start + 1), .k = try ops.slice(k, 1, start, start + 1), .v = try ops.slice(v, 1, start, start + 1), .decay = try ops.slice(decay, 1, start, start + 1), .beta = try ops.slice(beta, 1, start, start + 1), .state = if (parent < 0) initial else states[@intCast(parent)] }, s);
                states[row] = try ops.own(one.state);
                const y = try ops.cast(try ops.own(one.y), .float32);
                const actual = try ops.cast(try ops.slice(all, 1, start, start + 1), .float32);
                try mlx.check(mlx.mlx_array_eval(y));
                try mlx.check(mlx.mlx_array_eval(actual));
                try std.testing.expectEqualSlices(f32, mlx.mlx_array_data_float32(y).?[0 .. H * DV], mlx.mlx_array_data_float32(actual).?[0 .. H * DV]);
            }
        }
    }
}

pub const ProjectionMode = enum { serial_rows, affine_rows, affine_rows_ffn };

pub fn linearRows(ops: *Ops, linear: @import("glm5_model.zig").Linear, x: Arr, mode: ProjectionMode) !Arr {
    const shape = mlx.getShape(x);
    if (shape.len != 3 or shape[0] != 1 or shape[1] < 1 or shape[1] > 16) return error.InvalidGlmDraftShape;
    if (shape[1] == 1) return linear.apply(ops, x);
    if (mode == .affine_rows or mode == .affine_rows_ffn) {
        if (try @import("glm5_dflash_qmm.zig").project(ops.s, x, linear)) |output| return ops.own(output);
        if (try @import("glm5_dflash_dense_rows.zig").project(ops, linear, x)) |output| return output;
    }
    var rows: [16]Arr = undefined;
    var made: usize = 0;
    defer for (rows[0..made]) |value| {
        _ = mlx.mlx_array_free(value);
    };
    for (0..@intCast(shape[1])) |i| {
        var row = Ops{ .s = ops.s };
        defer row.deinit();
        rows[i] = try row.result(try linear.apply(&row, try row.slice(x, 1, @intCast(i), @intCast(i + 1))));
        made += 1;
    }
    return ops.concat(rows[0..made], 1);
}

pub const Tape = struct {
    inputs: primitive.KdaInputs,
    conv_input: Arr,
    parents: [16]i32 = @splat(-1),
    count: usize = 0,
    retained_state: Arr = .{ .ctx = null },
    retained_row: u32 = std.math.maxInt(u32),
    pub fn deinit(self: *Tape) void {
        if (self.retained_state.ctx != null) _ = mlx.mlx_array_free(self.retained_state);
        for ([_]Arr{ self.inputs.q, self.inputs.k, self.inputs.v, self.inputs.decay, self.inputs.beta, self.inputs.state, self.conv_input }) |value| _ = mlx.mlx_array_free(value);
    }
    pub fn replay(self: *const Tape, path: []const u32, s: mlx.mlx_stream) !@import("transformer.zig").SSMCacheEntry {
        const rows = mlx.getShape(self.inputs.q)[1];
        if (path.len == 0 or path.len > 16 or path[0] != 0 or self.count != rows) return error.InvalidGlmDraftTree;
        for (path, 0..) |row, i| {
            if (row >= rows or (i > 0 and self.parents[row] != path[i - 1])) return error.InvalidGlmDraftTree;
        }
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const indices = try ops.own(mlx.mlx_array_new_data(path.ptr, &[_]c_int{@intCast(path.len)}, 1, .uint32));
        const reused = self.retained_state.ctx != null and path[path.len - 1] == self.retained_row;
        const state = if (reused) try ops.result(self.retained_state) else blk: {
            const output = try primitive.kda(.{ .q = try ops.take(self.inputs.q, indices, 1), .k = try ops.take(self.inputs.k, indices, 1), .v = try ops.take(self.inputs.v, indices, 1), .decay = try ops.take(self.inputs.decay, indices, 1), .beta = try ops.take(self.inputs.beta, indices, 1), .state = self.inputs.state }, s);
            defer output.deinit();
            break :blk try ops.result(output.state);
        };
        errdefer _ = mlx.mlx_array_free(state);
        var tail: [3]u32 = undefined;
        for (&tail, 0..) |*row, i| {
            const position = @as(i32, @intCast(path.len)) - 3 + @as(i32, @intCast(i));
            row.* = if (position < 0) @intCast(3 + position) else path[@intCast(position)] + 3;
        }
        const tail_ids = try ops.own(mlx.mlx_array_new_data(&tail, &[_]c_int{3}, 1, .uint32));
        const conv = try ops.result(try ops.contiguous(try ops.take(self.conv_input, tail_ids, 1)));
        errdefer _ = mlx.mlx_array_free(conv);
        return .{ .conv_state = conv, .ssm_state = state, .initialized = true };
    }
};

pub const LayerResult = struct { output: Arr, tape: Tape };
var force_staged_for_tests = false;
pub fn forceStagedForTest(on: bool) void {
    if (@import("builtin").is_test) force_staged_for_tests = on;
}

const KdaLayer = @import("glm5_model.zig").KdaLayer;
const SSMCacheEntry = @import("transformer.zig").SSMCacheEntry;

/// `.serial_rows` keeps one-row projection geometry; the affine modes batch rows exactly.
pub fn applyLayer(layer: KdaLayer, ops: *Ops, x: Arr, cfg: *const @import("model.zig").ModelConfig, state: *const SSMCacheEntry, parents: []const i32, mode: ProjectionMode) !LayerResult {
    const sh = mlx.getShape(x);
    if (sh.len != 3 or sh[0] != 1 or sh[1] < 1 or sh[1] > 16 or sh[1] != parents.len) return error.InvalidGlmDraftShape;
    const p = try project(layer, ops, x, mode);
    var r = try recur(layer, ops, p, 0, cfg, state, parents);
    errdefer r.tape.deinit();
    return .{ .output = try finish(layer, ops, r.y, p.gate, cfg, mode), .tape = r.tape };
}

pub const Projected = struct { raw: Arr, a: Arr, beta: Arr, gate: Arr };

/// Every projection of the layer over all rows: rows of several requests read each weight once.
pub fn project(layer: KdaLayer, ops: *Ops, x: Arr, mode: ProjectionMode) !Projected {
    const raw = try ops.concat(&.{ try linearRows(ops, layer.q, x, mode), try linearRows(ops, layer.k, x, mode), try linearRows(ops, layer.v, x, mode) }, -1);
    return .{
        .raw = raw,
        .a = try linearRows(ops, layer.fb, try linearRows(ops, layer.fa, x, mode), mode),
        .beta = try linearRows(ops, layer.beta, x, mode),
        .gate = try linearRows(ops, layer.gb, try linearRows(ops, layer.ga, x, mode), mode),
    };
}

pub const Recurred = struct { y: Arr, tape: Tape };

/// One request's tree, rows `from ..` of `p`: conv prework and recurrence from that request's state.
pub fn recur(layer: KdaLayer, ops: *Ops, p: Projected, from: usize, cfg: *const @import("model.zig").ModelConfig, state: *const SSMCacheEntry, parents: []const i32) !Recurred {
    if (parents.len < 1 or parents.len > 16 or cfg.linear_conv_kernel_dim != 4) return error.InvalidGlmDraftShape;
    if (parents[0] != -1) return error.InvalidGlmDraftTree;
    for (parents[1..], 1..) |parent, i| if (parent < 0 or parent >= i) return error.InvalidGlmDraftTree;
    const heads: c_int = @intCast(cfg.linear_num_value_heads);
    const dim: c_int = @intCast(cfg.linear_key_head_dim);
    const width = heads * dim;
    const n: c_int = @intCast(parents.len);
    const begin: c_int = @intCast(from);
    const raw = try ops.slice(p.raw, 1, begin, begin + n);
    const a_raw = try ops.slice(p.a, 1, begin, begin + n);
    const beta_raw = try ops.slice(p.beta, 1, begin, begin + n);
    const dtype = mlx.mlx_array_dtype(raw);
    const old = if (state.initialized) state.conv_state else try ops.zeros(&.{ 1, 3, width * 3 }, dtype);
    const conv_input = try ops.concat(&.{ old, raw }, 1);
    const conv_w = if (layer.prepared_conv.ctx != null) layer.prepared_conv else try ops.contiguous(try ops.transpose(try ops.concat(&.{ layer.conv_q, layer.conv_k, layer.conv_v }, 0), &.{ 0, 2, 1 }));
    const exp_decay = if (layer.prepared_decay.ctx != null) layer.prepared_decay else try ops.unary(.exp, layer.a_log);
    const dims = [_]c_int{ 1, n, heads, dim };
    const fused = if (!force_staged_for_tests and dim == 128) try @import("glm5_kda_prework.zig").applyTree(ops.s, .{
        .qkv = raw,
        .a = a_raw,
        .beta = beta_raw,
        .conv_weight = conv_w,
        .exp_a = exp_decay,
        .dt_bias = layer.dt_bias,
        .conv_state = if (state.initialized) state.conv_state else null,
        .heads = heads,
        .lower = cfg.kda_gate_lower_bound,
    }, parents) else null;
    defer if (fused) |value| value.deinit();
    const work = if (fused) |value| value.arrays() else blk: {
        var window: [16 * 4]i32 = undefined;
        for (0..parents.len) |row| for (0..4) |j| {
            var back = 3 - j;
            var at: i32 = @intCast(row);
            while (back > 0 and at >= 0) {
                at = parents[@intCast(at)];
                back -= 1;
            }
            window[row * 4 + j] = if (at >= 0) 3 + at else 2 - @as(i32, @intCast(back));
        };
        const indices = try ops.own(mlx.mlx_array_new_data(&window, &[_]c_int{@intCast(parents.len * 4)}, 1, .int32));
        const conv_x = try ops.reshape(try ops.take(conv_input, indices, 1), &.{ n, 4, width * 3 });
        const convolved = try ops.reshape(try ops.silu(try ops.conv(conv_x, conv_w, width * 3)), &.{ 1, n, width * 3 });
        const rq = try ops.cast(try ops.reshape(try ops.slice(convolved, 2, 0, width), &dims), .float32);
        const rk = try ops.cast(try ops.reshape(try ops.slice(convolved, 2, width, 2 * width), &dims), .float32);
        const values = try ops.reshape(try ops.slice(convolved, 2, 2 * width, 3 * width), &dims);
        const eps = try ops.scalar(1e-6, .float32);
        const qnorm = try ops.unary(.rsqrt, try ops.binary(.add, try ops.reduce(try ops.binary(.mul, rq, rq), -1, false, true), eps));
        const knorm = try ops.unary(.rsqrt, try ops.binary(.add, try ops.reduce(try ops.binary(.mul, rk, rk), -1, false, true), eps));
        const q = try ops.cast(try ops.binary(.mul, try ops.binary(.mul, rq, qnorm), try ops.scalar(1 / @sqrt(@as(f32, @floatFromInt(dim))), .float32)), dtype);
        const k = try ops.cast(try ops.binary(.mul, rk, knorm), dtype);
        const a = try ops.reshape(try ops.cast(a_raw, .float32), &dims);
        const shift = try ops.reshape(try ops.cast(layer.dt_bias, .float32), &.{ 1, 1, heads, dim });
        const magnitude = try ops.reshape(exp_decay, &.{ 1, 1, heads, 1 });
        const decay = try ops.unary(.exp, try ops.binary(.mul, try ops.unary(.sigmoid, try ops.binary(.mul, magnitude, try ops.binary(.add, a, shift))), try ops.scalar(cfg.kda_gate_lower_bound, .float32)));
        const beta = try ops.unary(.sigmoid, beta_raw);
        break :blk [_]Arr{ q, k, values, decay, beta };
    };
    const initial = if (state.initialized) state.ssm_state else try ops.zeros(&.{ 1, heads, dim, dim }, .float32);
    const inputs = primitive.KdaInputs{ .q = work[0], .k = work[1], .v = work[2], .decay = work[3], .beta = work[4], .state = initial };
    const retained: ?LeafResult = if (parents.len <= 3) try recurrentLeaf(inputs, parents, ops.s) else null;
    defer if (retained) |value| value.deinit();
    const y_bf = try ops.own(if (retained) |value| try ops.result(value.y) else try recurrent(inputs, parents, ops.s));
    var tape = Tape{ .inputs = .{ .q = .{ .ctx = null }, .k = .{ .ctx = null }, .v = .{ .ctx = null }, .decay = .{ .ctx = null }, .beta = .{ .ctx = null }, .state = .{ .ctx = null } }, .conv_input = .{ .ctx = null } };
    errdefer tape.deinit();
    inline for (.{ "q", "k", "v", "decay", "beta", "state" }) |name| @field(tape.inputs, name) = try ops.result(@field(inputs, name));
    tape.conv_input = try ops.result(conv_input);
    if (retained) |value| {
        tape.retained_state = try ops.result(value.state);
        tape.retained_row = value.row;
    }
    tape.count = parents.len;
    @memcpy(tape.parents[0..parents.len], parents);
    return .{ .y = y_bf, .tape = tape };
}

/// Gated output norm and output projection over all rows (`y` and `gate` row-aligned).
pub fn finish(layer: KdaLayer, ops: *Ops, y_bf: Arr, gate: Arr, cfg: *const @import("model.zig").ModelConfig, mode: ProjectionMode) !Arr {
    const heads: c_int = @intCast(cfg.linear_num_value_heads);
    const dim: c_int = @intCast(cfg.linear_key_head_dim);
    const rows = mlx.getShape(y_bf)[1];
    const gate_bf = try ops.reshape(gate, &.{ 1, rows, heads, dim });
    const post = if (!force_staged_for_tests and rows > 1) try @import("glm5_kda_fused.zig").post(ops.s, y_bf, gate_bf, layer.out_norm, cfg.rms_norm_eps) else null;
    const gated = if (post) |value| try ops.own(value) else blk: {
        const y = try ops.cast(y_bf, .float32);
        const variance = try ops.reduce(try ops.binary(.mul, y, y), -1, true, true);
        const normalization = try ops.unary(.rsqrt, try ops.binary(.add, variance, try ops.scalar(cfg.rms_norm_eps, .float32)));
        const normalized = try ops.binary(.mul, try ops.binary(.mul, y, normalization), try ops.cast(layer.out_norm, .float32));
        break :blk try ops.cast(try ops.binary(.mul, normalized, try ops.unary(.sigmoid, try ops.cast(gate_bf, .float32))), mlx.mlx_array_dtype(gate));
    };
    return linearRows(ops, layer.out, try ops.reshape(gated, &.{ 1, rows, heads * dim }), mode);
}

test "GLM DFlash KDA layer tree and replay equal serial ancestor forwards" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    for ([_]u32{ 1, 64 }) |heads| {
        var weights = @import("model.zig").Weights.init(a);
        defer weights.deinit();
        var cfg = try @import("glm5_forward.zig").completeFixture(&weights);
        cfg.linear_num_value_heads = heads;
        var iter = weights.map.iterator();
        var seed: usize = 75;
        while (iter.next()) |entry| {
            if (!std.mem.startsWith(u8, entry.key_ptr.*, "model.language_model.layers.0.self_attn.")) continue;
            const value = entry.value_ptr;
            const original_shape = mlx.getShape(value.*);
            const name = entry.key_ptr.*["model.language_model.layers.0.self_attn.".len..];
            var dimensions: [3]c_int = undefined;
            @memcpy(dimensions[0..original_shape.len], original_shape);
            const width: c_int = @intCast(heads * 128);
            if (std.mem.eql(u8, name, "A_log") or std.mem.eql(u8, name, "dt_bias")) {
                const replacement = mlx.mlx_array_new();
                var mutable = replacement;
                try mlx.check(mlx.mlx_zeros(&mutable, &[_]c_int{if (std.mem.eql(u8, name, "A_log")) @intCast(heads) else width}, 1, .float32, s));
                _ = mlx.mlx_array_free(value.*);
                value.* = mutable;
                continue;
            }
            if (original_shape.len < 2) continue;
            if (std.mem.eql(u8, name, "o_proj.weight")) dimensions[1] = width else if (std.mem.eql(u8, name, "b_proj.weight")) dimensions[0] = @intCast(heads) else if (!std.mem.eql(u8, name, "f_a_proj.weight") and !std.mem.eql(u8, name, "g_a_proj.weight")) dimensions[0] = width;
            const replacement = try @import("dflash.zig").TinyFix.bf16ArrShaped(dimensions[0..original_shape.len], seed, s);
            _ = mlx.mlx_array_free(value.*);
            value.* = replacement;
            seed += 1;
        }
        var layer = try @import("glm5_model.zig").KdaLayer.load(&weights, "model.language_model.layers.0.self_attn", &cfg);
        defer layer.deinit();
        try layer.prepare(s);
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const x = try ops.own(try @import("dflash.zig").TinyFix.bf16ArrShaped(&.{ 1, 5, 128 }, 37, s));
        const initial = @import("transformer.zig").SSMCacheEntry{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer _ = mlx.mlx_array_free(initial.conv_state);
        defer _ = mlx.mlx_array_free(initial.ssm_state);
        const parents = [_]i32{ -1, 0, 0, 1, 2 };
        const pre_before = @import("glm5_kda_prework.zig").treeDispatchCount();
        const post_before = @import("glm5_kda_fused.zig").postDispatchCount();
        var all = try applyLayer(layer, &ops, x, &cfg, &initial, &parents, .serial_rows);
        defer all.tape.deinit();
        try std.testing.expectEqual(pre_before + 1, @import("glm5_kda_prework.zig").treeDispatchCount());
        try std.testing.expectEqual(post_before + 1, @import("glm5_kda_fused.zig").postDispatchCount());
        {
            var staged_ops = Ops{ .s = s };
            defer staged_ops.deinit();
            forceStagedForTest(true);
            defer forceStagedForTest(false);
            var staged = try applyLayer(layer, &staged_ops, x, &cfg, &initial, &parents, .serial_rows);
            defer staged.tape.deinit();
            try equalArray(staged.output, all.output, s);
            inline for (.{ "q", "k", "v", "decay", "beta" }) |field|
                try equalArray(@field(staged.tape.inputs, field), @field(all.tape.inputs, field), s);
            try equalArray(staged.tape.conv_input, all.tape.conv_input, s);
        }
        var states: [5]@import("transformer.zig").SSMCacheEntry = undefined;
        var made: usize = 0;
        defer for (states[0..made]) |state| {
            _ = mlx.mlx_array_free(state.conv_state);
            _ = mlx.mlx_array_free(state.ssm_state);
        };
        for (parents, 0..) |parent, row| {
            states[row] = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = parent >= 0 };
            made += 1;
            if (parent >= 0) {
                try mlx.check(mlx.mlx_array_set(&states[row].conv_state, states[@intCast(parent)].conv_state));
                try mlx.check(mlx.mlx_array_set(&states[row].ssm_state, states[@intCast(parent)].ssm_state));
            }
            const expected = try layer.apply(&ops, try ops.slice(x, 1, @intCast(row), @intCast(row + 1)), &cfg, &states[row]);
            try equalArray(expected, try ops.slice(all.output, 1, @intCast(row), @intCast(row + 1)), s);
        }
        {
            var single_ops = Ops{ .s = s };
            defer single_ops.deinit();
            const before_single = @import("glm5_kda_fused.zig").postDispatchCount();
            var single = try applyLayer(layer, &single_ops, try single_ops.slice(x, 1, 0, 1), &cfg, &initial, &.{-1}, .serial_rows);
            defer single.tape.deinit();
            try std.testing.expectEqual(before_single, @import("glm5_kda_fused.zig").postDispatchCount());
            try equalArray(single.output, try single_ops.slice(all.output, 1, 0, 1), s);
            const one_state = try single.tape.replay(&.{0}, s);
            defer _ = mlx.mlx_array_free(one_state.conv_state);
            defer _ = mlx.mlx_array_free(one_state.ssm_state);
            try equalArray(one_state.conv_state, states[0].conv_state, s);
            try equalArray(one_state.ssm_state, states[0].ssm_state, s);
        }
        try std.testing.expectError(error.InvalidGlmDraftTree, all.tape.replay(&.{ 0, 1, 4 }, s));
        const replay = try all.tape.replay(&.{ 0, 2, 4 }, s);
        defer _ = mlx.mlx_array_free(replay.conv_state);
        defer _ = mlx.mlx_array_free(replay.ssm_state);
        try equalArray(replay.conv_state, states[4].conv_state, s);
        try equalArray(replay.ssm_state, states[4].ssm_state, s);
        var detached_tape = blk: {
            var detached_ops = Ops{ .s = s };
            defer detached_ops.deinit();
            const detached = try applyLayer(layer, &detached_ops, x, &cfg, &initial, &parents, .serial_rows);
            break :blk detached.tape;
        };
        defer detached_tape.deinit();
        const detached_replay = try detached_tape.replay(&.{ 0, 2, 4 }, s);
        defer _ = mlx.mlx_array_free(detached_replay.conv_state);
        defer _ = mlx.mlx_array_free(detached_replay.ssm_state);
        try equalArray(detached_replay.conv_state, states[4].conv_state, s);
        try equalArray(detached_replay.ssm_state, states[4].ssm_state, s);
    }
}

fn equalArray(a: Arr, b: Arr, s: mlx.mlx_stream) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const ca = try ops.contiguous(a);
    const cb = try ops.contiguous(b);
    try mlx.check(mlx.mlx_array_eval(ca));
    try mlx.check(mlx.mlx_array_eval(cb));
    const n = mlx.mlx_array_size(ca);
    if (mlx.mlx_array_dtype(ca) == .float32)
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(ca).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(cb).?[0..n]))
    else
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(ca).?[0..n], mlx.mlx_array_data_bfloat16(cb).?[0..n]);
}

test "GLM DFlash KDA replay keeps raw BF16 one-token history bits" {
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const parents = [_]i32{ -1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7 };
    const bits = [_]u16{ 0x8000, 1, 0x8001, 0x3f80, 0x0080, 0xbf80, 0, 0x007f };
    var raw: [19 * 384]u16 = undefined;
    for (&raw, 0..) |*v, i| v.* = bits[(i + i / 384) % bits.len];
    const source = try ops.own(mlx.mlx_array_new_data(&raw, &.{ 1, 19, 384 }, 3, .bfloat16));
    const zeros = try ops.zeros(&.{ 1, 16, 1, 128 }, .bfloat16);
    var tape = Tape{
        .inputs = .{
            .q = try ops.result(zeros),
            .k = try ops.result(zeros),
            .v = try ops.result(zeros),
            .decay = try ops.result(try ops.ones(&.{ 1, 16, 1, 128 }, .float32)),
            .beta = try ops.result(try ops.zeros(&.{ 1, 16, 1 }, .bfloat16)),
            .state = try ops.result(try ops.zeros(&.{ 1, 1, 128, 128 }, .float32)),
        },
        .conv_input = try ops.result(source),
        .parents = parents,
        .count = 16,
    };
    defer tape.deinit();
    for ([_][]const u32{ &.{0}, &.{ 0, 1 }, &.{ 0, 2, 6 }, &.{ 0, 1, 3, 7, 15 } }) |path| {
        const state = try tape.replay(path, s);
        defer _ = mlx.mlx_array_free(state.conv_state);
        defer _ = mlx.mlx_array_free(state.ssm_state);
        var history: [19]u32 = undefined;
        history[0..3].* = .{ 0, 1, 2 };
        for (path, 0..) |node, i| history[i + 3] = node + 3;
        var expected: [3 * 384]u16 = undefined;
        for (0..3) |j| {
            const row = history[path.len + j];
            @memcpy(expected[j * 384 ..][0..384], raw[row * 384 ..][0..384]);
        }
        try mlx.check(mlx.mlx_array_eval(state.conv_state));
        try std.testing.expectEqualSlices(u16, &expected, mlx.mlx_array_data_bfloat16(state.conv_state).?[0..expected.len]);
    }
}

test "GLM DFlash dense rows preserve integrated serial outputs" {
    const dense = @import("glm5_dflash_dense_rows.zig");
    // 128 and 64 serve KDA's FA/GA/beta, 32 the MLA index weights.
    for ([_]c_int{ 128, 64, 32 }) |width| for ([_]c_int{ 2, 3, 4, 5, 6, 7, 8 }) |rows| {
        var ops = Ops{ .s = mlx.gpuStream() };
        defer ops.deinit();
        const key = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key, @intCast(443 + width + rows)));
        const w = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(w, &.{ width, 4096 }, 2, .bfloat16, 0, 0.05, key.*, ops.s));
        const x = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(x, &.{ 1, rows, 4096 }, 3, .bfloat16, 0, 0.1, key.*, ops.s));
        try mlx.check(mlx.mlx_array_eval(w.*));
        const linear = @import("glm5_model.zig").Linear{ .w = w.*, .input = 4096, .output = width };
        const before = dense.dispatchCount();
        const expected = try linearRows(&ops, linear, x.*, .serial_rows);
        try std.testing.expectEqual(before, dense.dispatchCount());
        const actual = try linearRows(&ops, linear, x.*, .affine_rows_ffn);
        try std.testing.expectEqual(before + 1, dense.dispatchCount());
        try mlx.check(mlx.mlx_array_eval(expected));
        try mlx.check(mlx.mlx_array_eval(actual));
        const n: usize = @intCast(rows * width);
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(expected).?[0..n], mlx.mlx_array_data_bfloat16(actual).?[0..n]);
    };
}
