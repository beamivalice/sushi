//! Layerwise GLM tree verification with bounded branch-state lifetime.
const std = @import("std");
const mlx = @import("mlx.zig");
const forward = @import("glm5_forward.zig");
const base = @import("glm5_model.zig");
const primitive = @import("glm5_next.zig");
const attention = @import("glm5_attention.zig");
const kda = @import("glm5_dflash_kda.zig");
const tree = @import("glm5_dflash_tree.zig");
const adapter = @import("glm5_dflash.zig");
const think_penalty = @import("think_penalty.zig");
const Arr = mlx.mlx_array;
const Ops = base.Ops;
// Layers per asynchronous evaluation group; zero settles every layer synchronously.
threadlocal var async_layers: ?usize = 0;
var async_dispatches: usize = 0;
var sync_dispatches: usize = 0;
pub const ScheduleBinding = struct {
    previous: ?usize,
    pub fn restore(self: ScheduleBinding) void {
        async_layers = self.previous;
    }
};
pub fn bindSchedule(layers: usize) !ScheduleBinding {
    if (layers != 0 and layers != 2 and layers != 4) return error.InvalidGlmVerifySchedule;
    const old = ScheduleBinding{ .previous = async_layers };
    async_layers = layers;
    return old;
}
/// Choose the measured schedule after the grouped verification width is known.
pub fn bindDefaultSchedule() ScheduleBinding {
    const old = ScheduleBinding{ .previous = async_layers };
    async_layers = null;
    return old;
}
fn evaluationCadence(rows: usize, mode: kda.ProjectionMode) usize {
    return async_layers orelse if (rows == 3 and mode == .affine_rows_ffn and base.naxArms()) @as(usize, 2) else 4;
}
pub fn asyncDispatchCount() usize {
    return async_dispatches;
}
pub fn syncDispatchCount() usize {
    return sync_dispatches;
}
fn appendTape(evals: mlx.mlx_vector_array, tape: LayerTape) !void {
    switch (tape) {
        .kda => |t| for ([_]Arr{ t.inputs.q, t.inputs.k, t.inputs.v, t.inputs.decay, t.inputs.beta, t.inputs.state, t.conv_input }) |a| {
            try mlx.check(mlx.mlx_vector_array_append_value(evals, a));
        },
        .mla => |t| for ([_]Arr{ t.latent, t.keys, t.gates }) |a| {
            try mlx.check(mlx.mlx_vector_array_append_value(evals, a));
        },
    }
}
var mla_branch_flushes: usize = 0;
var mla_scratch_bound: usize = 0;
pub fn branchFlushCount() usize {
    return mla_branch_flushes;
}
pub fn scratchBoundBytes() usize {
    return mla_scratch_bound;
}
pub fn resetStats() void {
    async_dispatches = 0;
    sync_dispatches = 0;
    mla_branch_flushes = 0;
    mla_scratch_bound = 0;
}

fn ancestry(parents: []const i32, row: usize, out: *[16]u32) []const u32 {
    var count: usize = 0;
    var at: i32 = @intCast(row);
    while (at >= 0) : (at = parents[@intCast(at)]) {
        out[count] = @intCast(at);
        count += 1;
    }
    std.mem.reverse(u32, out[0..count]);
    return out[0..count];
}

pub const MlaTape = struct {
    latent: Arr,
    keys: Arr,
    gates: Arr,
    ape: Arr,
    fn deinit(self: *MlaTape) void {
        for ([_]Arr{ self.latent, self.keys, self.gates, self.ape }) |a| _ = mlx.mlx_array_free(a);
    }
    fn append(self: *const MlaTape, state: *attention.State, path: []const u32, s: mlx.mlx_stream) !void {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const indices = try ops.own(mlx.mlx_array_new_data(path.ptr, &[_]c_int{@intCast(path.len)}, 1, .uint32));
        _ = try state.append(try ops.take(self.latent, indices, 0), try ops.take(self.keys, indices, 0), try ops.take(self.gates, indices, 0), self.ape, s);
    }
    /// Returns the overlay tail: the path's rows of `readable`.
    fn appendIndex(self: *const MlaTape, state: *attention.State, path: []const u32, readable: Arr, s: mlx.mlx_stream) !Arr {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const indices = try ops.own(mlx.mlx_array_new_data(path.ptr, &.{@intCast(path.len)}, 1, .uint32));
        _ = try state.appendIndexOnly(try ops.take(self.latent, indices, 0), try ops.take(self.keys, indices, 0), try ops.take(self.gates, indices, 0), self.ape, s);
        return ops.result(try ops.take(readable, indices, 0));
    }
};

const MlaRows = struct { qa: Arr, latent: Arr, index_q: Arr, keys: Arr, index_weights: Arr, gates: Arr, broadcast: bool };

/// Every MLA projection over all rows. The absorbed query keeps each row's one-row geometry
/// unless two or more rows take the exact broadcast.
fn mlaProject(layer: *const forward.Mla, ops: *Ops, x: Arr, cfg: *const @import("model.zig").ModelConfig, mode: kda.ProjectionMode) !MlaRows {
    const t: c_int = mlx.getShape(x)[1];
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kd: c_int = @intCast(cfg.mla_qk_nope_head_dim);
    const width: c_int = @intCast(cfg.mla_kv_lora_rank);
    const ih: c_int = @intCast(cfg.indexer_n_heads);
    const iw: c_int = @intCast(cfg.indexer_head_dim);
    const qr = try ops.rms(try kda.linearRows(ops, layer.qa, x, mode), layer.qa_norm, cfg.rms_norm_eps);
    const q = try ops.reshape(try kda.linearRows(ops, layer.qb, qr, mode), &.{ t, heads, 1, kd });
    const verify_batch = @import("glm5_mla_verify_batch.zig");
    const broadcast_q = if (layer.quantized) try verify_batch.run(ops, .{ .x = q, .w = layer.wk, .scales = layer.sk, .biases = layer.bk }, .query) else null;
    const qa = try ops.reshape(if (broadcast_q) |out| out else blk: {
        var absorbed: [16]Arr = undefined;
        for (0..@intCast(t)) |row| {
            const one = try ops.slice(q, 0, @intCast(row), @intCast(row + 1));
            absorbed[row] = if (layer.quantized) try ops.qmm(one, layer.wk, layer.sk, layer.bk, false) else try ops.binary(.mm, one, layer.wk);
        }
        break :blk try ops.concat(absorbed[0..@intCast(t)], 0);
    }, &.{ t, heads, width });
    const index_q = try ops.reshape(try kda.linearRows(ops, layer.iq, qr, mode), &.{ t, ih, iw });
    const compress = base.Linear{ .w = layer.compress, .scales = .{ .ctx = null }, .biases = .{ .ctx = null }, .input = @intCast(cfg.hidden_size), .output = @intCast(cfg.indexer_head_dim) };
    return .{
        .qa = qa,
        .latent = try ops.reshape(try ops.rms(try kda.linearRows(ops, layer.kva, x, mode), layer.kv_norm, cfg.rms_norm_eps), &.{ t, width }),
        .index_q = index_q,
        .keys = try ops.reshape(try ops.layerNorm(try kda.linearRows(ops, layer.ik, x, mode), layer.ik_norm, layer.ik_bias, 1e-6), &.{ t, iw }),
        .index_weights = try ops.reshape(try ops.cast(try ops.binary(.mul, try kda.linearRows(ops, layer.iw, x, mode), try ops.scalar(1 / @sqrt(@as(f32, @floatFromInt(ih * iw))), .float32)), mlx.mlx_array_dtype(index_q)), &.{ t, ih }),
        .gates = try ops.reshape(try kda.linearRows(ops, compress, x, mode), &.{ t, iw }),
        .broadcast = broadcast_q != null,
    };
}

/// One request's tree, rows `from ..` of `rows`: its branches attend its own cache. Each row's
/// `[1, heads, 1, width]` output lands at `attended[row]`.
/// `joined`, when given, receives every row's output as one `[rows, heads, 1, width]` array.
fn mlaAttend(layer: *const forward.Mla, ops: *Ops, rows: MlaRows, from: usize, cfg: *const @import("model.zig").ModelConfig, state: *const attention.State, parents: []const i32, attended: []Arr, joined: ?*?Arr) !MlaTape {
    const native = @import("glm5_attention_decode_batch.zig");
    const dtype = mlx.mlx_array_dtype(rows.latent);
    const native_mode = native.enabled() and native.supportedConfig(cfg, dtype, ops.s);
    if (native_mode and parents.len > @import("glm5_dflash_memory.zig").overlay_rows) return error.GlmDecodeNativeTreeUnsupported;
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kd: c_int = @intCast(cfg.mla_qk_nope_head_dim);
    const width: c_int = @intCast(cfg.mla_kv_lora_rank);
    const iw: c_int = @intCast(cfg.indexer_head_dim);
    const begin: c_int = @intCast(from);
    const end: c_int = begin + @as(c_int, @intCast(parents.len));
    const latent_capacity: usize = if (state.latent.ctx != null) @intCast(mlx.getShape(state.latent)[0]) else 0;
    const pool_capacity: usize = if (state.pooled.ctx != null) @intCast(mlx.getShape(state.pooled)[0]) else 0;
    var scratch = try @import("glm5_dflash_memory.zig").plan(state.processed, latent_capacity, pool_capacity, @intCast(width), @intCast(iw), @intCast(heads), parents.len, mlx.mlx_array_itemsize(rows.latent));
    if (native_mode) {
        const limit = @import("glm5_dflash_memory.zig").limit_bytes;
        if (scratch.common_bytes >= limit - native.scratchLimitFor(parents.len)) return error.GlmTreeScratchLimit;
        scratch.branches = @min(parents.len, (limit - scratch.common_bytes - native.scratchLimitFor(parents.len)) / scratch.per_branch_bytes);
        if (scratch.branches == 0) return error.GlmTreeScratchLimit;
        scratch.live_bytes = scratch.common_bytes + scratch.branches * scratch.per_branch_bytes + native.scratchLimitFor(parents.len);
    }
    mla_scratch_bound = @max(mla_scratch_bound, scratch.live_bytes);
    const latent = try ops.slice(rows.latent, 0, begin, end);
    const qa = try ops.slice(rows.qa, 0, begin, end);
    const index_q = try ops.slice(rows.index_q, 0, begin, end);
    const index_weights = try ops.slice(rows.index_weights, 0, begin, end);
    var tape = MlaTape{ .latent = .{ .ctx = null }, .keys = .{ .ctx = null }, .gates = .{ .ctx = null }, .ape = .{ .ctx = null } };
    errdefer tape.deinit();
    tape.latent = try ops.result(latent);
    tape.keys = try ops.result(try ops.slice(rows.keys, 0, begin, end));
    tape.gates = try ops.result(try ops.slice(rows.gates, 0, begin, end));
    tape.ape = try ops.result(layer.ape);
    // Branch rows are read as serial decode would read them once stored.
    const readable = try ops.own(try @import("glm5_latent.zig").readable(latent, state.latent_bits, ops.s));
    // Dense prefixes do not consume index_q/index_weights. Keep them lazy here;
    // sparse branches naturally bill their necessary indexer work in mla_branches.
    var pending: [16]Arr = undefined;
    var pending_count: usize = 0;
    const batched_native = native_mode and parents.len >= 2 and scratch.branches == parents.len;
    var native_ids: [native.max_tail]Arr = undefined;
    var native_branches: [native.max_tail]native.Branch = undefined;
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(kd)));
    var branch_states: [native.max_tail]attention.State = undefined;
    var select_rows: [native.max_tail]attention.SelectRow = undefined;
    var kept_branches: usize = 0;
    defer for (branch_states[0..kept_branches]) |*kept| kept.deinit();
    for (0..parents.len) |row| {
        var branch = try state.share();
        var keep = false;
        defer if (!keep) branch.deinit();
        var path: [16]u32 = undefined;
        const kept = ancestry(parents, row, &path);
        const overlay = parents.len <= @import("glm5_dflash_memory.zig").overlay_rows;
        const tail = if (overlay) try ops.own(try tape.appendIndex(&branch, kept, readable, ops.s)) else blk: {
            try tape.append(&branch, kept, ops.s);
            break :blk Arr{ .ctx = null };
        };
        const at: c_int = @intCast(row);
        const query = try ops.slice(qa, 0, at, at + 1);
        const index_query = try ops.slice(index_q, 0, at, at + 1);
        const weights = try ops.slice(index_weights, 0, at, at + 1);
        const offset = state.processed + kept.len - 1;
        if (batched_native) {
            branch_states[row] = branch;
            keep = true;
            kept_branches = row + 1;
            select_rows[row] = .{ .state = &branch_states[row], .index_q = index_query, .weights = weights, .offset = offset };
            native_branches[row] = .{ .offset = offset, .length = branch.processed, .path = native.pathOf(&.{}) };
            @memcpy(native_branches[row].path[0..kept.len], kept);
            continue;
        }
        const y = try ops.own(if (overlay)
            try attention.attendOverlay(&branch, query, index_query, weights, offset, scale, .{ .prefix = state.latentView(), .prefix_rows = state.processed, .tail = tail }, ops.s)
        else
            try attention.attend(&branch, query, index_query, weights, offset, scale, ops.s));
        pending[pending_count] = y;
        pending_count += 1;
        // The final group settles at the enclosing layer boundary.
        if (pending_count == scratch.branches and row + 1 < parents.len) {
            const group = mlx.mlx_vector_array_new_data(&pending, pending_count);
            defer _ = mlx.mlx_vector_array_free(group);
            try mlx.check(mlx.mlx_eval(group));
            mla_branch_flushes += 1;
            pending_count = 0;
        }
        attended[row] = try ops.reshape(y, &.{ 1, heads, 1, width });
    }
    if (batched_native) {
        try attention.decodeSelectedRows(select_rows[0..parents.len], native_ids[0..parents.len], ops.s);
        for (native_ids[0..parents.len]) |*id| id.* = try ops.own(id.*);
        const prefix = if (state.processed == 0) attention.Latent{ .data = readable } else state.latentView();
        const selected = try ops.concat(native_ids[0..parents.len], 0);
        const batched = try native.run(ops, qa, prefix, state.processed, readable, native_branches[0..parents.len], selected, scale);
        if (joined) |slot| if (batched) |all| {
            slot.* = try ops.reshape(all, &.{ @intCast(parents.len), heads, 1, width });
        };
        for (0..parents.len) |row| {
            const y = if (batched) |all| try ops.slice(all, 0, @intCast(row), @intCast(row + 1)) else (try native.run(ops, try ops.slice(qa, 0, @intCast(row), @intCast(row + 1)), prefix, state.processed, readable, native_branches[row .. row + 1], native_ids[row], scale)) orelse return error.GlmDecodeNativeUnsupported;
            attended[row] = try ops.reshape(y, &.{ 1, heads, 1, width });
        }
    }
    return tape;
}

/// Value unembedding and output projection over all rows' attention outputs.
/// `joined` is `attended` as one array when the attention produced it so; the value rows then stay joined too.
fn mlaFinish(layer: *const forward.Mla, ops: *Ops, rows: MlaRows, attended: []const Arr, joined: ?Arr, cfg: *const @import("model.zig").ModelConfig, mode: kda.ProjectionMode) !Arr {
    const value_width: c_int = @intCast(cfg.num_attention_heads * cfg.mla_v_head_dim);
    var result_rows: [16]Arr = undefined;
    const broadcast = if (rows.broadcast or (layer.quantized and attended.len >= 2)) try @import("glm5_mla_verify_batch.zig").run(ops, .{ .x = joined orelse try ops.concat(attended, 0), .w = layer.wv, .scales = layer.sv, .biases = layer.bv }, .value) else null;
    if (joined != null) if (broadcast) |all| return kda.linearRows(ops, layer.out, try ops.reshape(all, &.{ 1, @intCast(attended.len), value_width }), mode);
    for (attended, 0..) |y4, row| {
        const values = if (broadcast) |all| try ops.slice(all, 0, @intCast(row), @intCast(row + 1)) else if (layer.quantized) try ops.qmm(y4, layer.wv, layer.sv, layer.bv, true) else try ops.binary(.mm, y4, try ops.transpose(layer.wv, &.{ 0, 2, 1 }));
        result_rows[row] = try ops.reshape(values, &.{ 1, 1, value_width });
    }
    return kda.linearRows(ops, layer.out, try ops.concat(result_rows[0..attended.len], 1), mode);
}

const LayerTape = union(enum) {
    kda: kda.Tape,
    mla: MlaTape,
    fn deinit(self: *LayerTape) void {
        switch (self.*) {
            .kda => |*t| t.deinit(),
            .mla => |*t| t.deinit(),
        }
    }
};

pub const Verified = struct {
    allocator: std.mem.Allocator,
    layers: []?LayerTape,
    captures: adapter.Captures,
    tokens: [16]u32 = undefined,
    parents: [16]i32 = undefined,
    targets: [16]u32 = undefined,
    count: usize,
    offset: usize,
    logits: Arr = .{ .ctx = null },
    pub fn deinit(self: *Verified) void {
        if (self.logits.ctx != null) _ = mlx.mlx_array_free(self.logits);
        for (self.layers) |*maybe| if (maybe.*) |*layer| layer.deinit();
        self.allocator.free(self.layers);
        self.captures.deinit();
    }
    /// Replays only the accepted KDA prework and rebuilds only its IndexPool path.
    pub fn prepareCommit(self: *const Verified, source: *const forward.Request, budget: usize, eos: []const u32, s: mlx.mlx_stream) !adapter.Verification {
        return self.prepareCommitImpl(source, null, budget, eos, s);
    }
    /// Serving commit: the source hands its MLA buffers to the accepted state, so the
    /// append writes in place instead of copying the reserved capacity. The source is
    /// then only valid to be replaced by the result; any later failure marks it failed.
    pub fn prepareCommitConsuming(self: *const Verified, source: *forward.Request, budget: usize, eos: []const u32, s: mlx.mlx_stream) !adapter.Verification {
        return self.prepareCommitImpl(source, source, budget, eos, s);
    }
    fn prepareCommitImpl(self: *const Verified, source: *const forward.Request, owner: ?*forward.Request, budget: usize, eos: []const u32, s: mlx.mlx_stream) !adapter.Verification {
        if (source.offset != self.offset or source.layers.len != self.layers.len) return error.InvalidGlmDraftOffset;
        const accepted = try tree.accept(self.tokens[0..self.count], self.parents[0..self.count], self.targets[0..self.count], budget, eos);
        const path = accepted.rows[0..accepted.count];
        const last = path[path.len - 1];
        var result = adapter.Verification{ .count = self.count, .offset = self.offset, .tokens = self.tokens, .parents = self.parents, .targets = self.targets };
        errdefer result.deinit();
        result.states[last] = try adapter.cloneRequest(source);
        const next = &result.states[last].?;
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const arrays = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(arrays);
        for (self.layers, next.layers) |maybe, *state| switch (maybe orelse return error.IncompleteGlmDraftVerify) {
            .kda => |tape| {
                const updated = try tape.replay(path, s);
                _ = mlx.mlx_array_free(state.recurrent.conv_state);
                _ = mlx.mlx_array_free(state.recurrent.ssm_state);
                state.recurrent = updated;
                try mlx.check(mlx.mlx_vector_array_append_value(arrays, updated.conv_state));
                try mlx.check(mlx.mlx_vector_array_append_value(arrays, updated.ssm_state));
            },
            .mla => |tape| {
                try tape.append(&state.attention, path, s);
                for (state.attention.arrays()) |a| if (a.ctx != null) {
                    try mlx.check(mlx.mlx_vector_array_append_value(arrays, a));
                };
            },
        };
        for (path) |row| {
            result.captures[row] = try adapter.Captures.init(self.allocator, self.captures.hook.ids);
            for (self.captures.hook.out, result.captures[row].?.hook.out) |value, *out| {
                try mlx.check(mlx.mlx_array_set(out, try ops.slice(value, 1, @intCast(row), @intCast(row + 1))));
                try mlx.check(mlx.mlx_vector_array_append_value(arrays, out.*));
            }
        }
        if (owner) |held| {
            errdefer held.failed = true;
            // A command buffer keeps its inputs past `mlx_eval` until it completes; drain it so the
            // pending appends hold the only references and MLX donates the buffers.
            try mlx.check(mlx.mlx_synchronize(s));
            for (held.layers) |*layer| for ([_]*Arr{ &layer.attention.latent, &layer.attention.latent_scales, &layer.attention.latent_biases, &layer.attention.pooled }) |buffer| if (buffer.ctx != null) {
                _ = mlx.mlx_array_free(buffer.*);
                buffer.* = .{ .ctx = null };
            };
            try mlx.check(mlx.mlx_eval(arrays));
        } else try mlx.check(mlx.mlx_eval(arrays));
        next.offset += path.len;
        return result;
    }
};

pub fn verify(target: *const forward.Model, request: *const forward.Request, tokens: []const u32, parents: []const i32, taps: []const u32, mode: kda.ProjectionMode) !Verified {
    var out: [1]Verified = undefined;
    try verifyGroups(target, &.{.{ .request = request, .tokens = tokens, .parents = parents }}, taps, mode, &out);
    return out[0];
}

/// A request's logit policy (static biases, the think penalty) after its committed ids.
pub const Penalty = struct { state: think_penalty.ThinkPenalty, markers: ?Arr };

/// One request's tree in a grouped verification. A penalty shifts its rows' decisions and logits.
pub const Group = struct { request: *const forward.Request, tokens: []const u32, parents: []const i32, penalty: ?Penalty = null };

/// Rows a grouped verification carries: the row kernels' bound.
pub const max_rows = 16;

/// One target forward over several requests' trees, rows in group order. Projections, router,
/// experts, HC and head read each weight once for all rows; the KDA recurrence and MLA attention
/// run per request on its own state. `out[i]` is bit for bit what `verify` returns for `groups[i]`.
pub fn verifyGroups(target: *const forward.Model, groups: []const Group, taps: []const u32, mode: kda.ProjectionMode, out: []Verified) !void {
    if (target.expert_stream != null) return error.GlmStreamingSpecUnsupported;
    if (groups.len == 0 or out.len != groups.len) return error.InvalidGlmDraftTree;
    for (taps, 0..) |id, i| if (id >= target.layers.len or (i > 0 and id <= taps[i - 1])) return error.InvalidGlmCapture;
    var tokens: [max_rows]u32 = undefined;
    var starts: [max_rows + 1]usize = undefined;
    var total: usize = 0;
    for (groups, 0..) |group, g| {
        const request = group.request;
        try tree.validate(group.tokens, group.parents);
        if (request.failed) return error.GlmRequestNeedsReset;
        if (request.capture != null or request.layers.len != target.layers.len) return error.InvalidGlmDraftOffset;
        for (group.tokens) |token| if (token >= target.cfg.vocab_size) return error.InvalidGlmDraftToken;
        for (0..group.parents.len) |row| {
            var path: [16]u32 = undefined;
            const length = ancestry(group.parents, row, &path).len;
            if (request.offset >= target.cfg.max_position_embeddings or length > target.cfg.max_position_embeddings - request.offset) return error.GlmContextExceeded;
        }
        if (total + group.tokens.len > max_rows) return error.InvalidGlmDraftShape;
        starts[g] = total;
        @memcpy(tokens[total .. total + group.tokens.len], group.tokens);
        total += group.tokens.len;
    }
    starts[groups.len] = total;
    var made: usize = 0;
    errdefer for (out[0..made]) |*v| v.deinit();
    for (groups, out) |group, *v| {
        const allocator = group.request.allocator;
        const layers = try allocator.alloc(?LayerTape, target.layers.len);
        @memset(layers, null);
        const captures = adapter.Captures.init(allocator, taps) catch |err| {
            allocator.free(layers);
            return err;
        };
        v.* = .{ .allocator = allocator, .layers = layers, .captures = captures, .count = group.tokens.len, .offset = group.request.offset };
        @memcpy(v.tokens[0..group.tokens.len], group.tokens);
        @memcpy(v.parents[0..group.parents.len], group.parents);
        made += 1;
    }
    const rows: c_int = @intCast(total);
    const cadence = evaluationCadence(total, mode);
    const fold_norm = rows >= 2 and rows <= 8 and base.naxArms() and @import("glm5_hc_collapse_simd32.zig").enabled();
    var h: Arr = undefined;
    {
        var ops = Ops{ .s = target.s };
        defer ops.deinit();
        const ids = try ops.own(mlx.mlx_array_new_data(&tokens, &[_]c_int{ 1, rows }, 2, .uint32));
        const embedding = try ops.own(try target.rawEmbedding(ids));
        h = try ops.result(try ops.contiguous(try ops.broadcast(try ops.reshape(embedding, &.{ 1, rows, 1, @intCast(target.cfg.hidden_size) }), &.{ 1, rows, 4, @intCast(target.cfg.hidden_size) })));
    }
    defer _ = mlx.mlx_array_free(h);
    var h_normalized = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h_normalized);
    var have_normalized = false;
    for (target.layers, 0..) |*layer, index| {
        var ops = Ops{ .s = target.s };
        defer ops.deinit();
        const pre = if (have_normalized) try layer.hc_attn.collapseFromNormalized(&ops, h, &target.cfg, layer.norm_attn, h_normalized) else if (fold_norm) try layer.hc_attn.collapseAndNorm(&ops, h, &target.cfg, layer.norm_attn) else try layer.hc_attn.collapse(&ops, h, &target.cfg);
        defer pre.deinit();
        const x = if (fold_norm) pre.mixed else try ops.rms(pre.mixed, layer.norm_attn, target.cfg.rms_norm_eps);
        const attended = switch (layer.attn) {
            .kda => |weights| blk: {
                const p = try kda.project(weights, &ops, x, mode);
                var ys: [max_rows]Arr = undefined;
                for (groups, out, 0..) |group, *v, g| {
                    const r = try kda.recur(weights, &ops, p, starts[g], &target.cfg, &group.request.layers[index].recurrent, group.parents);
                    v.layers[index] = .{ .kda = r.tape };
                    ys[g] = r.y;
                }
                const y = if (groups.len == 1) ys[0] else try ops.concat(ys[0..groups.len], 1);
                break :blk try kda.finish(weights, &ops, y, p.gate, &target.cfg, mode);
            },
            .mla => |*weights| blk: {
                const projected = try mlaProject(weights, &ops, x, &target.cfg, mode);
                var outputs: [max_rows]Arr = undefined;
                var joined: ?Arr = null;
                for (groups, out, 0..) |group, *v, g| {
                    v.layers[index] = .{ .mla = try mlaAttend(weights, &ops, projected, starts[g], &target.cfg, &group.request.layers[index].attention, group.parents, outputs[starts[g]..starts[g + 1]], if (groups.len == 1) &joined else null) };
                }
                break :blk try mlaFinish(weights, &ops, projected, outputs[0..total], joined, &target.cfg, mode);
            },
        };
        const expanded_attn = if (fold_norm) try @import("glm5_hc_expand_norm.zig").apply(target.s, h, attended, pre.post, pre.comb, target.cfg.rms_norm_eps) else null;
        defer if (expanded_attn) |value| value.deinit();
        const joined = try ops.own(if (expanded_attn) |value| try ops.result(value.expanded) else try primitive.hcExpand(h, attended, pre.post, pre.comb, target.s));
        const ff = if (expanded_attn) |value| try layer.hc_ffn.collapseFromNormalized(&ops, joined, &target.cfg, layer.norm_ffn, value.normalized) else if (fold_norm) try layer.hc_ffn.collapseAndNorm(&ops, joined, &target.cfg, layer.norm_ffn) else try layer.hc_ffn.collapse(&ops, joined, &target.cfg);
        defer ff.deinit();
        const fx = if (fold_norm) ff.mixed else try ops.rms(ff.mixed, layer.norm_ffn, target.cfg.rms_norm_eps);
        const ffout = if (mode == .affine_rows_ffn) try @import("glm5_dflash_ffn.zig").apply(target, index, &ops, fx) else blk: {
            var outputs: [max_rows]Arr = undefined;
            var done: usize = 0;
            defer for (outputs[0..done]) |value| {
                _ = mlx.mlx_array_free(value);
            };
            for (0..total) |row| {
                var one = Ops{ .s = target.s };
                defer one.deinit();
                outputs[row] = try one.result(try target.feedForwardLayer(index, &one, try one.slice(fx, 1, @intCast(row), @intCast(row + 1))));
                done += 1;
            }
            break :blk try ops.concat(outputs[0..done], 1);
        };
        const expanded_ffn = if (fold_norm and index + 1 < target.layers.len) try @import("glm5_hc_expand_norm.zig").apply(target.s, joined, ffout, ff.post, ff.comb, target.cfg.rms_norm_eps) else null;
        defer if (expanded_ffn) |value| value.deinit();
        const next = try ops.own(if (expanded_ffn) |value| try ops.result(value.expanded) else try primitive.hcExpand(joined, ffout, ff.post, ff.comb, target.s));
        have_normalized = expanded_ffn != null;
        if (expanded_ffn) |value| try mlx.check(mlx.mlx_array_set(&h_normalized, value.normalized));
        for (taps, 0..) |id, tap| if (id == index) {
            const mean = try ops.reduce(next, 2, true, false);
            for (out, 0..) |*v, g| {
                const value = if (groups.len == 1) mean else try ops.slice(mean, 1, @intCast(starts[g]), @intCast(starts[g + 1]));
                try mlx.check(mlx.mlx_array_set(&v.captures.hook.out[tap], value));
            }
        };
        // The first layer starts on the GPU while the host builds the rest of its group.
        if (index == 0 and cadence > 1) {
            const evals = mlx.mlx_vector_array_new_value(next);
            defer _ = mlx.mlx_vector_array_free(evals);
            for (out) |v| try appendTape(evals, v.layers[0].?);
            try mlx.check(mlx.mlx_async_eval(evals));
        }
        if (cadence == 0 or (index + 1) % cadence == 0) {
            const evals = mlx.mlx_vector_array_new_value(next);
            defer _ = mlx.mlx_vector_array_free(evals);
            const first = if (cadence == 0) index else index + 1 - cadence;
            for (out) |v| {
                for (v.layers[first .. index + 1]) |tape| try appendTape(evals, tape.?);
                for (taps, 0..) |id, tap| if (id >= first and id <= index) {
                    try mlx.check(mlx.mlx_vector_array_append_value(evals, v.captures.hook.out[tap]));
                };
            }
            if (cadence == 0) {
                try mlx.check(mlx.mlx_eval(evals));
                sync_dispatches += 1;
            } else {
                try mlx.check(mlx.mlx_async_eval(evals));
                async_dispatches += 1;
            }
        }
        try mlx.check(mlx.mlx_array_set(&h, next));
    }
    var ops = Ops{ .s = target.s };
    defer ops.deinit();
    const normalized = try ops.rms(try ops.reduce(h, 2, true, false), target.norm, target.cfg.rms_norm_eps);
    const raw = try kda.linearRows(&ops, target.head, normalized, mode);
    const logits = try target.samplingLogits(&ops, try penalized(&ops, raw, groups, starts[0 .. groups.len + 1]));
    const decisions = try ops.slot();
    try mlx.check(mlx.mlx_argmax_axis(decisions, logits, -1, false, target.s));
    const u = try ops.cast(decisions.*, .uint32);
    // Head dependencies do not necessarily consume every replay/capture array.
    // Settle them explicitly before returning ownership to commit/replay.
    const final = mlx.mlx_vector_array_new_value(u);
    defer _ = mlx.mlx_vector_array_free(final);
    if (cadence != 0) for (out) |v| {
        for (v.layers) |tape| try appendTape(final, tape.?);
        for (v.captures.hook.out) |value| try mlx.check(mlx.mlx_vector_array_append_value(final, value));
    };
    try mlx.check(mlx.mlx_eval(final));
    sync_dispatches += 1;
    const targets = (mlx.mlx_array_data_uint32(u) orelse return error.MlxArrayDataNull)[0..total];
    for (out, 0..) |*v, g| {
        @memcpy(v.targets[0..v.count], targets[starts[g]..starts[g + 1]]);
        v.logits = try ops.result(if (groups.len == 1) logits else try ops.slice(logits, 1, @intCast(starts[g]), @intCast(starts[g + 1])));
    }
}

/// Each penalized group's head rows, every row at the span state its own ancestry reaches, before
/// the reserved-id mask as serial sampling orders them.
fn penalized(ops: *Ops, raw: Arr, groups: []const Group, starts: []const usize) !Arr {
    for (groups) |group| {
        if (group.penalty != null) break;
    } else return raw;
    var parts: [max_rows]Arr = undefined;
    for (groups, 0..) |group, g| {
        const rows = if (groups.len == 1) raw else try ops.slice(raw, 1, @intCast(starts[g]), @intCast(starts[g + 1]));
        const penalty = group.penalty orelse {
            parts[g] = rows;
            continue;
        };
        var inside: [max_rows]bool = undefined;
        for (0..group.tokens.len) |row| {
            var path: [16]u32 = undefined;
            var ids: [16]u32 = undefined;
            const on_path = ancestry(group.parents, row, &path);
            for (on_path, ids[0..on_path.len]) |r, *id| id.* = group.tokens[r];
            inside[row] = penalty.state.phaseAfter(ids[0..on_path.len]) == .inside;
        }
        const out = try ops.slot();
        try think_penalty.shiftDecided(out, rows, penalty.markers, penalty.state, inside[0..group.tokens.len], ops.s);
        parts[g] = out.*;
    }
    return if (groups.len == 1) parts[0] else ops.concat(parts[0..groups.len], 1);
}

/// Commits a single-row verification as one decode step: the request takes the row in place.
pub fn commitPlainRow(verified: *const Verified, request: *forward.Request, s: mlx.mlx_stream) !void {
    if (verified.count != 1) return error.InvalidGlmDraftTree;
    var committed = try verified.prepareCommitConsuming(request, 1, &.{}, s);
    defer committed.deinit();
    request.deinit();
    request.* = committed.states[0].?;
    committed.states[0] = null;
}

test {
    _ = @import("glm5_dflash_memory.zig");
}

fn expectArrayBits(a: Arr, b: Arr, s: mlx.mlx_stream) !void {
    if (a.ctx == null or b.ctx == null) return std.testing.expect(a.ctx == null and b.ctx == null);
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const x = try ops.contiguous(a);
    const y = try ops.contiguous(b);
    try mlx.check(mlx.mlx_array_eval(x));
    try mlx.check(mlx.mlx_array_eval(y));
    const n = mlx.mlx_array_size(x) * mlx.mlx_array_itemsize(x);
    try std.testing.expectEqualSlices(u8, mlx.mlx_array_data_uint8(x).?[0..n], mlx.mlx_array_data_uint8(y).?[0..n]);
}

test "GLM latent overlay three-node verifier commits independent serial ancestry" {
    try verifierCommitCase(0, 3);
}

test "GLM kv8 latent overlay verifier commits independent serial ancestry" {
    try verifierCommitCase(8, 3);
}

test "GLM kv6 latent overlay verifier commits independent serial ancestry" {
    try verifierCommitCase(6, 3);
}

test "GLM overlay verifier past the sparse threshold on a reserved request commits serial ancestry" {
    // 2050 committed rows: the tree's later rows select pools, and their paths complete pool 512.
    try verifierCommitCase(0, 2050);
    try verifierCommitCase(8, 2050);
    try verifierCommitCase(6, 2050);
}

fn verifierCommitCase(latent_bits: u8, prompt: usize) !void {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    var weights = @import("model.zig").Weights.init(a);
    defer weights.deinit();
    var cfg = try forward.completeFixture(&weights);
    cfg.max_position_embeddings = @max(cfg.max_position_embeddings, @as(u32, @intCast(prompt + 16)));
    var iterator = weights.map.iterator();
    var seed: usize = 23;
    while (iterator.next()) |entry| {
        const value = entry.value_ptr;
        const sh = mlx.getShape(value.*);
        if (sh.len < 2 or mlx.mlx_array_dtype(value.*) != .bfloat16 or std.mem.endsWith(u8, entry.key_ptr.*, ".scales") or std.mem.endsWith(u8, entry.key_ptr.*, ".biases")) continue;
        const replacement = try @import("dflash.zig").TinyFix.bf16ArrShaped(sh, seed, s);
        _ = mlx.mlx_array_free(value.*);
        value.* = replacement;
        seed += 1;
    }
    var target = try forward.Model.load(a, cfg, &weights, s);
    defer target.deinit();
    var request = try forward.Request.init(a, target.layers.len);
    defer request.deinit();
    try request.setLatentBits(latent_bits);
    const prompt_ids = try a.alloc(u32, prompt);
    defer a.free(prompt_ids);
    var rng = std.Random.DefaultPrng.init(prompt);
    for (prompt_ids, 0..) |*id, i| id.* = if (prompt <= 3) @intCast(i + 1) else rng.random().uintLessThan(u32, @intCast(cfg.vocab_size));
    const ids = mlx.mlx_array_new_data(prompt_ids.ptr, &.{ 1, @intCast(prompt) }, 2, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const logits = try target.forwardLast(&request, ids, true);
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_eval(logits));
    if (prompt > 3) _ = try @import("glm5_dflash_reserve.zig").reserve(&request, prompt + 65536, std.math.maxInt(usize), s);
    var tokens = [_]u32{ 1, 0, 0 };
    const taps = [_]u32{ 0, 3 };
    for ([_][3]i32{ .{ -1, 0, 1 }, .{ -1, 0, 0 } }) |parents| {
        tokens[2] = if (parents[2] == 0) 2 else 0;
        var oracle = try adapter.verifyTreeOracle(&target, &request, &tokens, &parents, &taps);
        defer oracle.deinit();
        var computed = try verify(&target, &request, &tokens, &parents, &taps, .serial_rows);
        defer computed.deinit();
        try std.testing.expectEqualSlices(u32, oracle.targets[0..oracle.count], computed.targets[0..computed.count]);
        var captures = Ops{ .s = s };
        defer captures.deinit();
        for (0..3) |row| for (computed.captures.hook.out, oracle.captures[row].?.hook.out) |left, right| {
            try expectArrayBits(try captures.slice(left, 1, @intCast(row), @intCast(row + 1)), right, s);
        };
        for ([_]usize{ 1, 2, 3 }) |budget| {
            var committed = try computed.prepareCommit(&request, budget, &.{}, s);
            defer committed.deinit();
            const accepted = try tree.accept(&tokens, &parents, oracle.targets[0..oracle.count], budget, &.{});
            const last = accepted.rows[accepted.count - 1];
            const left = committed.states[last].?;
            const right = oracle.states[last].?;
            try std.testing.expectEqual(right.offset, left.offset);
            try std.testing.expectEqual(@as(mlx.mlx_dtype, if (latent_bits == 0) .bfloat16 else .uint32), mlx.mlx_array_dtype(left.layers[3].attention.latent));
            for (left.layers, right.layers) |x, y| {
                for (x.attention.arrays(), y.attention.arrays()) |u, v| try expectArrayBits(u, v, s);
                try expectArrayBits(x.recurrent.conv_state, y.recurrent.conv_state, s);
                try expectArrayBits(x.recurrent.ssm_state, y.recurrent.ssm_state, s);
            }
        }
    }
}

test "GLM consuming commit appends MLA rows in place with the copying commit's bits" {
    try consumingCommitCase(0, false);
}

test "GLM consuming commit appends kv8 MLA rows in place with the copying commit's bits" {
    try consumingCommitCase(8, false);
}

test "GLM consuming commit appends kv6 MLA rows in place with the copying commit's bits" {
    try consumingCommitCase(6, false);
}

test "GLM consuming commit appends in place while a command buffer still reads the old rows" {
    try consumingCommitCase(0, true);
    try consumingCommitCase(8, true);
    try consumingCommitCase(6, true);
}

/// Queues one long GPU job that reads every MLA buffer: its command buffer holds them until it completes.
fn readInFlight(request: *const forward.Request, s: mlx.mlx_stream) !void {
    var ops = Ops{ .s = s };
    defer ops.deinit();
    var total = try ops.own(mlx.mlx_array_new_float(0));
    for (request.layers) |layer| for ([_]Arr{ layer.attention.latent, layer.attention.latent_scales, layer.attention.latent_biases }) |buffer| if (buffer.ctx != null) {
        const sum = try ops.slot();
        try mlx.check(mlx.mlx_sum(sum, buffer, false, s));
        total = try ops.binary(.add, total, try ops.cast(sum.*, .float32));
    };
    const wide = try ops.slot();
    try mlx.check(mlx.mlx_broadcast_to(wide, total, &[_]c_int{ 1024, 65536 }, 2, s));
    const ones = try ops.slot();
    try mlx.check(mlx.mlx_ones(ones, &[_]c_int{ 65536, 1024 }, 2, .float32, s));
    const evals = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(evals);
    try mlx.check(mlx.mlx_vector_array_append_value(evals, try ops.binary(.mm, wide.*, ones.*)));
    try mlx.check(mlx.mlx_async_eval(evals));
}

fn consumingCommitCase(latent_bits: u8, in_flight: bool) !void {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    var weights = @import("model.zig").Weights.init(a);
    defer weights.deinit();
    const cfg = try forward.completeFixture(&weights);
    var iterator = weights.map.iterator();
    var seed: usize = 41;
    while (iterator.next()) |entry| {
        const value = entry.value_ptr;
        const sh = mlx.getShape(value.*);
        if (sh.len < 2 or mlx.mlx_array_dtype(value.*) != .bfloat16 or std.mem.endsWith(u8, entry.key_ptr.*, ".scales") or std.mem.endsWith(u8, entry.key_ptr.*, ".biases")) continue;
        const replacement = try @import("dflash.zig").TinyFix.bf16ArrShaped(sh, seed, s);
        _ = mlx.mlx_array_free(value.*);
        value.* = replacement;
        seed += 1;
    }
    var target = try forward.Model.load(a, cfg, &weights, s);
    defer target.deinit();
    var request = try forward.Request.init(a, target.layers.len);
    defer request.deinit();
    try request.setLatentBits(latent_bits);
    const ids = mlx.mlx_array_new_data(&[_]u32{ 1, 2, 3 }, &.{ 1, 3 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const logits = try target.forwardLast(&request, ids, true);
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_eval(logits));
    const tokens = [_]u32{ 1, 0, 0 };
    const parents = [_]i32{ -1, 0, 1 };
    const taps = [_]u32{ 0, 3 };
    var computed = try verify(&target, &request, &tokens, &parents, &taps, .serial_rows);
    defer computed.deinit();
    var expected = try computed.prepareCommit(&request, 3, &.{}, s);
    defer expected.deinit();
    var buffers: [16][3]?[*]const u8 = @splat(@splat(null));
    var mla_layers: usize = 0;
    for (request.layers, 0..) |layer, i| if (layer.attention.latent.ctx != null) {
        for (&buffers[i], [_]Arr{ layer.attention.latent, layer.attention.latent_scales, layer.attention.latent_biases }) |*slot, buffer| {
            if (buffer.ctx != null) slot.* = mlx.mlx_array_data_uint8(buffer) else try std.testing.expect(latent_bits == 0);
        }
        mla_layers += 1;
    };
    try std.testing.expect(mla_layers > 0);
    if (in_flight) try readInFlight(&request, s);
    var actual = try computed.prepareCommitConsuming(&request, 3, &.{}, s);
    defer actual.deinit();
    const accepted = try tree.accept(&tokens, &parents, computed.targets[0..computed.count], 3, &.{});
    const last = accepted.rows[accepted.count - 1];
    const left = expected.states[last].?;
    const right = actual.states[last].?;
    try std.testing.expectEqual(left.offset, right.offset);
    for (left.layers, right.layers, request.layers, 0..) |x, y, released, i| {
        for (released.attention.arrays()[0..4]) |buffer| try std.testing.expect(buffer.ctx == null);
        for (buffers[i], [_]Arr{ y.attention.latent, y.attention.latent_scales, y.attention.latent_biases }) |old, now| {
            if (old) |pointer| try std.testing.expectEqual(pointer, mlx.mlx_array_data_uint8(now).?);
        }
        for (x.attention.arrays(), y.attention.arrays()) |u, v| try expectArrayBits(u, v, s);
        try expectArrayBits(x.recurrent.conv_state, y.recurrent.conv_state, s);
        try expectArrayBits(x.recurrent.ssm_state, y.recurrent.ssm_state, s);
    }
    try std.testing.expect(!request.failed);
}

fn randomizedFixture(weights: *@import("model.zig").Weights, s: mlx.mlx_stream) !@import("model.zig").ModelConfig {
    const cfg = try forward.completeFixture(weights);
    var iterator = weights.map.iterator();
    var seed: usize = 41;
    while (iterator.next()) |entry| {
        const value = entry.value_ptr;
        const sh = mlx.getShape(value.*);
        if (sh.len < 2 or mlx.mlx_array_dtype(value.*) != .bfloat16 or std.mem.endsWith(u8, entry.key_ptr.*, ".scales") or std.mem.endsWith(u8, entry.key_ptr.*, ".biases")) continue;
        const replacement = try @import("dflash.zig").TinyFix.bf16ArrShaped(sh, seed, s);
        _ = mlx.mlx_array_free(value.*);
        value.* = replacement;
        seed += 1;
    }
    return cfg;
}

test "GLM grouped verification gives every request exactly what it gets alone" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    var weights = @import("model.zig").Weights.init(a);
    defer weights.deinit();
    const cfg = try randomizedFixture(&weights, s);
    var target = try forward.Model.load(a, cfg, &weights, s);
    defer target.deinit();
    const prompts = [_][]const u32{ &.{ 1, 2, 3 }, &.{ 4, 0 }, &.{ 2, 2, 1, 0 } };
    var requests: [3]forward.Request = undefined;
    for (&requests, prompts, 0..) |*request, prompt, i| {
        request.* = try forward.Request.init(a, target.layers.len);
        try request.setLatentBits((&[_]u8{ 0, 8, 6 })[i]);
        const ids = mlx.mlx_array_new_data(prompt.ptr, &.{ 1, @intCast(prompt.len) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const logits = try target.forwardLast(request, ids, true);
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_eval(logits));
    }
    defer for (&requests) |*request| request.deinit();
    const taps = [_]u32{ 0, 3 };
    const groups = [_]Group{
        .{ .request = &requests[0], .tokens = &.{ 1, 0, 0 }, .parents = &.{ -1, 0, 1 } },
        .{ .request = &requests[1], .tokens = &.{3}, .parents = &.{-1} },
        .{ .request = &requests[2], .tokens = &.{ 2, 1 }, .parents = &.{ -1, 0 } },
    };
    for ([_]kda.ProjectionMode{ .serial_rows, .affine_rows_ffn }) |mode| {
        var together: [groups.len]Verified = undefined;
        try verifyGroups(&target, &groups, &taps, mode, &together);
        defer for (&together) |*v| v.deinit();
        for (groups, &together) |group, *joint| {
            var alone = try verify(&target, group.request, group.tokens, group.parents, &taps, mode);
            defer alone.deinit();
            try std.testing.expectEqual(alone.count, joint.count);
            try std.testing.expectEqualSlices(u32, alone.targets[0..alone.count], joint.targets[0..joint.count]);
            try expectArrayBits(alone.logits, joint.logits, s);
            for (alone.captures.hook.out, joint.captures.hook.out) |x, y| try expectArrayBits(x, y, s);
            var expected = try alone.prepareCommit(group.request, group.tokens.len, &.{}, s);
            defer expected.deinit();
            var actual = try joint.prepareCommit(group.request, group.tokens.len, &.{}, s);
            defer actual.deinit();
            const accepted = try tree.accept(group.tokens, group.parents, alone.targets[0..alone.count], group.tokens.len, &.{});
            const last = accepted.rows[accepted.count - 1];
            const left = expected.states[last].?;
            const right = actual.states[last].?;
            for (left.layers, right.layers) |x, y| {
                for (x.attention.arrays(), y.attention.arrays()) |u, v| try expectArrayBits(u, v, s);
                try expectArrayBits(x.recurrent.conv_state, y.recurrent.conv_state, s);
                try expectArrayBits(x.recurrent.ssm_state, y.recurrent.ssm_state, s);
            }
        }
    }
}

test "GLM one verified row and its commit equal a serial decode step" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    var weights = @import("model.zig").Weights.init(a);
    defer weights.deinit();
    const cfg = try randomizedFixture(&weights, s);
    var target = try forward.Model.load(a, cfg, &weights, s);
    defer target.deinit();
    for ([_]u8{ 0, 8, 6 }) |bits| {
        var request = try forward.Request.initServing(a, target.layers.len);
        defer request.deinit();
        try request.setLatentBits(bits);
        const prompt = [_]u32{ 3, 1, 2, 1, 0 };
        const ids = mlx.mlx_array_new_data(&prompt, &.{ 1, prompt.len }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const prefill = try target.forwardLast(&request, ids, true);
        defer _ = mlx.mlx_array_free(prefill);
        try mlx.check(mlx.mlx_array_eval(prefill));
        for ([_]u32{ 2, 0, 3 }) |token| {
            var verified = try verify(&target, &request, &.{token}, &.{-1}, &.{}, .affine_rows_ffn);
            defer verified.deinit();
            var serial = try adapter.cloneRequest(&request);
            defer serial.deinit();
            const one = mlx.mlx_array_new_data(&token, &.{ 1, 1 }, 2, .uint32);
            defer _ = mlx.mlx_array_free(one);
            const logits = try target.forwardLast(&serial, one, true);
            defer _ = mlx.mlx_array_free(logits);
            try expectArrayBits(logits, verified.logits, s);
            var committed = try verified.prepareCommitConsuming(&request, 1, &.{}, s);
            defer committed.deinit();
            request.deinit();
            request = committed.states[0].?;
            committed.states[0] = null;
            // The next step's logits read every committed MLA row; the recurrent state is compared here.
            try std.testing.expectEqual(serial.offset, request.offset);
            for (serial.layers, request.layers) |x, y| {
                try expectArrayBits(x.recurrent.conv_state, y.recurrent.conv_state, s);
                try expectArrayBits(x.recurrent.ssm_state, y.recurrent.ssm_state, s);
            }
        }
    }
}

test "GLM plain rows of several requests advance each exactly as its serial decode" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    var weights = @import("model.zig").Weights.init(a);
    defer weights.deinit();
    const cfg = try randomizedFixture(&weights, s);
    var target = try forward.Model.load(a, cfg, &weights, s);
    defer target.deinit();
    const prompts = [_][]const u32{ &.{ 3, 1, 2 }, &.{ 0, 2, 2, 1, 3 }, &.{1} };
    var rows: [3]forward.Request = undefined;
    var serial: [3]forward.Request = undefined;
    for (&rows, &serial, prompts, 0..) |*row, *ref, prompt, i| {
        row.* = try forward.Request.initServing(a, target.layers.len);
        ref.* = try forward.Request.initServing(a, target.layers.len);
        for ([_]*forward.Request{ row, ref }) |request| {
            try request.setLatentBits((&[_]u8{ 0, 8, 6 })[i]);
            const ids = mlx.mlx_array_new_data(prompt.ptr, &.{ 1, @intCast(prompt.len) }, 2, .uint32);
            defer _ = mlx.mlx_array_free(ids);
            const logits = try target.forwardLast(request, ids, true);
            defer _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_array_eval(logits));
        }
    }
    defer for (&rows, &serial) |*row, *ref| {
        row.deinit();
        ref.deinit();
    };
    var tokens = [_]u32{ 2, 0, 3 };
    for (0..3) |_| {
        var groups: [3]Group = undefined;
        for (&groups, &rows, 0..) |*g, *row, i| g.* = .{ .request = row, .tokens = tokens[i .. i + 1], .parents = &.{-1} };
        var out: [3]Verified = undefined;
        try verifyGroups(&target, &groups, &.{}, .affine_rows_ffn, &out);
        defer for (&out) |*v| v.deinit();
        for (&out, &rows, &serial, &tokens) |*v, *row, *ref, *token| {
            const one = mlx.mlx_array_new_data(token, &.{ 1, 1 }, 2, .uint32);
            defer _ = mlx.mlx_array_free(one);
            const logits = try target.forwardLast(ref, one, true);
            defer _ = mlx.mlx_array_free(logits);
            try expectArrayBits(logits, v.logits, s);
            try commitPlainRow(v, row, s);
            try std.testing.expectEqual(ref.offset, row.offset);
            token.* = v.targets[0];
        }
    }
}

/// The fixture with distinct embedding and head rows: the stock rows are all alike, so every
/// token embeds the same and every logit ties.
fn distinctRowsFixture(weights: *@import("model.zig").Weights, s: mlx.mlx_stream) !@import("model.zig").ModelConfig {
    const cfg = try randomizedFixture(weights, s);
    for ([_][]const u8{ "model.language_model.embed_tokens.weight", "lm_head.weight" }, 0..) |key, k| {
        var codes: [4 * 32]u32 = undefined;
        var rng = std.Random.DefaultPrng.init(k + 7);
        for (&codes) |*c| c.* = rng.random().int(u32) & 0x07070707;
        const value = weights.map.getPtr(key) orelse return error.TestUnexpectedResult;
        _ = mlx.mlx_array_free(value.*);
        value.* = mlx.mlx_array_new_data(&codes, &[_]c_int{ 4, 32 }, 2, .uint32);
    }
    return cfg;
}

test "GLM verify decides biased tree rows as serial decode under the same bias, per request" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    const Bias = @import("logit_bias.zig").Bias;
    var weights = @import("model.zig").Weights.init(a);
    defer weights.deinit();
    const cfg = try distinctRowsFixture(&weights, s);
    var target = try forward.Model.load(a, cfg, &weights, s);
    defer target.deinit();
    const vocab: usize = cfg.vocab_size;
    // Serial sampling masks reserved ids after the bias; a bias added after the mask rounds in its dtype.
    const reserved = try think_penalty.buildMask(a, &.{0}, vocab);
    defer _ = mlx.mlx_array_free(reserved);
    target.suppress_mask = reserved;
    const markers = try think_penalty.buildMask(a, &.{1}, vocab);
    defer _ = mlx.mlx_array_free(markers);
    var requests: [2]forward.Request = undefined;
    for (&requests, [_][]const u32{ &.{ 3, 1, 2 }, &.{ 2, 0 } }) |*request, prompt| {
        request.* = try forward.Request.init(a, target.layers.len);
        const ids = mlx.mlx_array_new_data(prompt.ptr, &.{ 1, @intCast(prompt.len) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const logits = try target.forwardLast(request, ids, true);
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_eval(logits));
    }
    defer for (&requests) |*request| request.deinit();
    const close: u32 = 3;
    const tokens = [_]u32{ 1, close, 2 };
    var unbiased = try verify(&target, &requests[0], &tokens, &.{ -1, 0, 1 }, &.{}, .serial_rows);
    defer unbiased.deinit();
    const ban = [_]Bias{ .{ .id = unbiased.targets[0], .delta = -100 }, .{ .id = 2, .delta = 0.375 } };
    const scoped = [_]Bias{ .{ .id = 2, .delta = 1, .scope = .reasoning }, .{ .id = 1, .delta = 0.625, .scope = .answer } };
    const cases = [_]think_penalty.ThinkPenalty{
        .{ .biases = &ban },
        .{ .lambda = 1.5, .opener_id = 2, .closer_id = close, .phase = .inside, .file_biases = &scoped, .biases = &ban },
        .{ .lambda = 1.5, .opener_id = 2, .closer_id = close, .phase = .inside },
    };
    for (cases) |case| {
        var p = case;
        try p.prepare(a, vocab, markers, s);
        defer p.deinitPrepared();
        // A chain crosses the closer once; the fork leaves one sibling inside the span.
        for ([_][3]i32{ .{ -1, 0, 1 }, .{ -1, 0, 0 } }) |parents| for ([_]kda.ProjectionMode{ .serial_rows, .affine_rows_ffn }) |mode| {
            const groups = [_]Group{
                .{ .request = &requests[0], .tokens = &tokens, .parents = &parents, .penalty = .{ .state = p, .markers = markers } },
                .{ .request = &requests[1], .tokens = &.{1}, .parents = &.{-1} },
            };
            var out: [2]Verified = undefined;
            try verifyGroups(&target, &groups, &.{}, mode, &out);
            defer for (&out) |*v| v.deinit();
            var plain = try verify(&target, &requests[1], &.{1}, &.{-1}, &.{}, mode);
            defer plain.deinit();
            try std.testing.expectEqual(plain.targets[0], out[1].targets[0]);
            try expectArrayBits(plain.logits, out[1].logits, s);
            if (p.lambda == 0) try std.testing.expect(out[0].targets[0] != unbiased.targets[0]);
            for (0..tokens.len) |row| {
                var serial = try adapter.cloneRequest(&requests[0]);
                defer serial.deinit();
                var ops = Ops{ .s = s };
                defer ops.deinit();
                var path: [16]u32 = undefined;
                var ids: [16]u32 = undefined;
                var raw: Arr = undefined;
                const rows_on_path = ancestry(&parents, row, &path);
                for (rows_on_path, 0..) |r, k| {
                    ids[k] = tokens[r];
                    const one = mlx.mlx_array_new_data(&tokens[r], &.{ 1, 1 }, 2, .uint32);
                    defer _ = mlx.mlx_array_free(one);
                    raw = try ops.own(try target.forwardLast(&serial, one, true));
                }
                var state = p;
                state.phase = p.phaseAfter(ids[0..rows_on_path.len]);
                const shifted = try ops.slot();
                try think_penalty.shiftScoped(shifted, raw, markers, state, null, s);
                const decided = try ops.own(@import("generate.zig").sampleTokenLazy(shifted.*, .{ .temperature = 0, .suppress_mask = reserved }, s));
                try mlx.check(mlx.mlx_array_eval(decided));
                var want: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&want, decided));
                try std.testing.expectEqual(@as(u32, @intCast(want)), out[0].targets[row]);
                try expectArrayBits(try target.samplingLogits(&ops, shifted.*), try ops.slice(out[0].logits, 1, @intCast(row), @intCast(row + 1)), s);
            }
        };
    }
}

test "GLM default verification schedule selects only three NAX rows" {
    const transformer = @import("transformer.zig");
    const saved = transformer.vqmm_nax_probe_override;
    defer transformer.vqmm_nax_probe_override = saved;
    for ([_]bool{ false, true }) |nax| {
        transformer.vqmm_nax_probe_override = nax;
        const automatic = bindDefaultSchedule();
        defer automatic.restore();
        for ([_]usize{ 1, 2, 3, 4, 16 }) |rows| {
            try std.testing.expectEqual(@as(usize, if (nax and rows == 3) 2 else 4), evaluationCadence(rows, .affine_rows_ffn));
            try std.testing.expectEqual(@as(usize, 4), evaluationCadence(rows, .serial_rows));
        }
        for ([_]usize{ 0, 2, 4 }) |cadence| {
            const explicit = try bindSchedule(cadence);
            try std.testing.expectEqual(cadence, evaluationCadence(3, .affine_rows_ffn));
            explicit.restore();
        }
    }
}
