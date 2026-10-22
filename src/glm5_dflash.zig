//! GLM DFlash2 target adapter and exact branch verification oracle.
const std = @import("std");
const mlx = @import("mlx.zig");
const forward = @import("glm5_forward.zig");
const draft = @import("dflash.zig");
const tree = @import("glm5_dflash_tree.zig");
const policy = @import("dflash_policy.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;

test "GLM DFlash request fork keeps rejected pool and recurrence state isolated" {
    var source = try forward.Request.init(std.testing.allocator, 1);
    defer source.deinit();
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const latent = try ops.ones(&.{ 3, 4 }, .float32);
    const keys = try ops.ones(&.{ 3, 2 }, .float32);
    const gates = try ops.zeros(&.{ 3, 2 }, .float32);
    const ape = try ops.zeros(&.{ 4, 2 }, .float32);
    _ = try source.layers[0].attention.append(latent, keys, gates, ape, ops.s);
    try source.layers[0].attention.evaluate();
    source.offset = 3;
    try mlx.check(mlx.mlx_array_set(&source.layers[0].recurrent.ssm_state, try ops.ones(&.{ 1, 1, 2, 2 }, .float32)));
    try mlx.check(mlx.mlx_array_set(&source.layers[0].recurrent.conv_state, try ops.ones(&.{ 1, 3, 6 }, .float32)));
    source.layers[0].recurrent.initialized = true;
    source.dense_prefill = true;
    source.prefill_async = true;
    source.prefill_sync_layers = 4;
    var branch = try cloneRequest(&source);
    defer branch.deinit();
    try std.testing.expectEqual(@as(u8, 4), branch.prefill_sync_layers);
    try std.testing.expect(branch.dense_prefill and branch.prefill_async);
    _ = try branch.layers[0].attention.append(try ops.ones(&.{ 1, 4 }, .float32), try ops.zeros(&.{ 1, 2 }, .float32), try ops.zeros(&.{ 1, 2 }, .float32), ape, ops.s);
    try branch.layers[0].attention.evaluate();
    try mlx.check(mlx.mlx_array_set(&branch.layers[0].recurrent.ssm_state, try ops.zeros(&.{ 1, 1, 2, 2 }, .float32)));
    try std.testing.expectEqual(@as(usize, 3), source.layers[0].attention.processed);
    try std.testing.expectEqual(@as(usize, 4), branch.layers[0].attention.processed);
    try std.testing.expect(source.layers[0].attention.pooled.ctx == null);
    try std.testing.expectEqual(@as(c_int, 3), mlx.getShape(source.layers[0].attention.tail_keys)[0]);
    try mlx.check(mlx.mlx_array_eval(source.layers[0].recurrent.ssm_state));
    try std.testing.expectEqual(@as(f32, 1), mlx.mlx_array_data_float32(source.layers[0].recurrent.ssm_state).?[0]);
    source.failed = true;
    try std.testing.expectError(error.GlmRequestNeedsReset, cloneRequest(&source));
}

/// All arrays are immutable values; retaining handles keeps each parent's history alive.
pub fn cloneRequest(source: *const forward.Request) !forward.Request {
    if (source.failed) return error.GlmRequestNeedsReset;
    var copy = try forward.Request.init(source.allocator, source.layers.len);
    errdefer copy.deinit();
    copy.offset = source.offset;
    copy.decode_async = source.decode_async;
    copy.dense_prefill = source.dense_prefill;
    copy.prefill_async = source.prefill_async;
    copy.prefill_sync_layers = source.prefill_sync_layers;
    copy.latent_bits = source.latent_bits;
    for (copy.layers, source.layers) |*dst, src| {
        if (src.recurrent.initialized) {
            try mlx.check(mlx.mlx_array_set(&dst.recurrent.conv_state, src.recurrent.conv_state));
            try mlx.check(mlx.mlx_array_set(&dst.recurrent.ssm_state, src.recurrent.ssm_state));
            dst.recurrent.initialized = true;
        }
        dst.attention = try src.attention.share();
    }
    return copy;
}

pub const Captures = struct {
    allocator: std.mem.Allocator,
    hook: forward.Capture,
    pub fn init(allocator: std.mem.Allocator, ids: []const u32) !Captures {
        const out = try allocator.alloc(Arr, ids.len);
        for (out) |*a| a.* = mlx.mlx_array_new_float(0);
        return .{ .allocator = allocator, .hook = .{ .ids = ids, .out = out } };
    }
    pub fn deinit(self: *Captures) void {
        for (self.hook.out) |a| _ = mlx.mlx_array_free(a);
        self.allocator.free(self.hook.out);
    }
};

fn argmax(logits: Arr, s: mlx.mlx_stream) !u32 {
    const sh = mlx.getShape(logits);
    if (sh.len != 3 or sh[0] != 1 or sh[1] != 1 or sh[2] < 1) return error.InvalidGlmDraftLogits;
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const idx = try ops.slot();
    try mlx.check(mlx.mlx_argmax_axis(idx, logits, -1, false, s));
    const u = try ops.cast(idx.*, .uint32);
    try mlx.check(mlx.mlx_array_eval(u));
    return (mlx.mlx_array_data_uint32(u) orelse return error.MlxArrayDataNull)[0];
}

/// Correctness oracle: one serial forward per node with an isolated ancestor state.
pub const Verification = struct {
    states: [16]?forward.Request = @splat(null),
    captures: [16]?Captures = @splat(null),
    targets: [16]u32 = undefined,
    tokens: [16]u32 = undefined,
    parents: [16]i32 = undefined,
    count: usize,
    offset: usize,
    pub fn deinit(self: *Verification) void {
        for (&self.states) |*item| if (item.*) |*state| state.deinit();
        for (&self.captures) |*item| if (item.*) |*capture| capture.deinit();
    }
};

pub fn verifyTreeOracle(target: *const forward.Model, request: *const forward.Request, tokens: []const u32, parents: []const i32, taps: []const u32) !Verification {
    try tree.validate(tokens, parents);
    for (tokens) |token| if (token >= target.cfg.vocab_size) return error.InvalidGlmDraftToken;
    if (request.capture != null) return error.GlmCaptureAlreadyActive;
    var result = Verification{ .count = tokens.len, .offset = request.offset };
    errdefer result.deinit();
    @memcpy(result.tokens[0..tokens.len], tokens);
    @memcpy(result.parents[0..parents.len], parents);
    for (tokens, parents, 0..) |token, parent, row| {
        const input = if (parent < 0) request else &result.states[@intCast(parent)].?;
        result.states[row] = try cloneRequest(input);
        result.captures[row] = try Captures.init(request.allocator, taps);
        const state = &result.states[row].?;
        state.capture = &result.captures[row].?.hook;
        defer state.capture = null;
        const ids = mlx.mlx_array_new_data(&token, &[_]c_int{ 1, 1 }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const logits = try target.forwardLast(state, ids, true);
        defer _ = mlx.mlx_array_free(logits);
        result.targets[row] = try argmax(logits, target.s);
        state.capture = null;
    }
    return result;
}

fn cloneContext(assistant: *const draft.DflashModel, source: *const draft.DflashCtx) !draft.DflashCtx {
    var snapshot = try source.cache.snapshot();
    defer snapshot.deinit();
    var copy = try draft.DflashCtx.init(assistant.allocator, assistant, source.base_pos);
    errdefer copy.deinit();
    try copy.cache.restore(&snapshot);
    return copy;
}

fn cropCommitContext(assistant: *const draft.DflashModel, next: *draft.DflashCtx) !void {
    const cfg = &assistant.config;
    if (assistant.layers.len != 5 or cfg.block_size != 8 or cfg.sliding_window != 2048 or next.cache.config.scheme != .off or next.cache.step < 2048) return;
    for (assistant.layers, next.cache.entries) |layer, entry| {
        if (layer.layer_type != .sliding_attention or !entry.initialized or entry.ringed or entry.base != 0 or entry.offset != next.cache.step or mlx.mlx_array_dtype(entry.keys) != .bfloat16 or mlx.mlx_array_dtype(entry.values) != .bfloat16) return;
    }
    const keep = cfg.sliding_window - 1;
    const drop = next.cache.step - keep;
    var ops = Ops{ .s = assistant.s };
    defer ops.deinit();
    for (next.cache.entries) |*entry| {
        const k = try ops.slice(entry.keys, 2, @intCast(drop), @intCast(next.cache.step));
        const v = try ops.slice(entry.values, 2, @intCast(drop), @intCast(next.cache.step));
        try mlx.check(mlx.mlx_array_set(&entry.keys, k));
        try mlx.check(mlx.mlx_array_set(&entry.values, v));
        entry.offset = keep;
    }
    next.cache.step = keep;
    next.base_pos += drop;
}

fn evaluateContext(context: *draft.DflashCtx) !void {
    const arrays = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(arrays);
    context.appendEvalArrays(arrays);
    try mlx.check(mlx.mlx_eval(arrays));
}

/// Both target and assistant publish only after the new assistant state has evaluated.
pub fn commitVerified(assistant: *const draft.DflashModel, context: *draft.DflashCtx, request: *forward.Request, verified: *Verification, budget: usize, eos: []const u32) !tree.Accepted {
    if (request.failed or request.offset != verified.offset or context.absLen() != verified.offset) return error.InvalidGlmDraftOffset;
    const accepted = try tree.accept(verified.tokens[0..verified.count], verified.parents[0..verified.count], verified.targets[0..verified.count], budget, eos);
    const last = accepted.rows[accepted.count - 1];
    if (verified.states[last] == null) return error.GlmDraftAlreadyCommitted;
    for (accepted.rows[0..accepted.count]) |row| {
        const capture = verified.captures[row] orelse return error.InvalidGlmDraftCaptures;
        if (!std.mem.eql(u32, capture.hook.ids, assistant.config.target_layer_ids)) return error.InvalidGlmDraftCaptures;
    }
    if (verified.states[last].?.offset != verified.offset + accepted.count) return error.InvalidGlmDraftOffset;
    var next_context = try cloneContext(assistant, context);
    errdefer next_context.deinit();
    try cropCommitContext(assistant, &next_context);
    var ops = Ops{ .s = assistant.s };
    defer ops.deinit();
    var projected: [16]Arr = undefined;
    if (assistant.config.target_layer_ids.len > projected.len) return error.InvalidGlmDraftCaptures;
    for (0..assistant.config.target_layer_ids.len) |tap| {
        var rows: [16]Arr = undefined;
        for (accepted.rows[0..accepted.count], 0..) |row, i| rows[i] = verified.captures[row].?.hook.out[tap];
        projected[tap] = try ops.concat(rows[0..accepted.count], 1);
    }
    try draft.appendContext(assistant, &next_context, projected[0..assistant.config.target_layer_ids.len], verified.offset);
    try evaluateContext(&next_context);
    request.deinit();
    request.* = verified.states[last].?;
    verified.states[last] = null;
    context.deinit();
    context.* = next_context;
    return accepted;
}

pub fn validatePair(assistant: *const draft.DflashModel, target: *const forward.Model) !void {
    if (target.expert_stream != null) return error.GlmStreamingSpecUnsupported;
    const cfg = &assistant.config;
    if (!target.cfg.isGlm5() or !cfg.isDflash2() or cfg.hidden_size != target.cfg.hidden_size or cfg.mask_token_id >= target.cfg.vocab_size or
        cfg.anchor_row_drafts or assistant.markov != null or assistant.selector == null or cfg.target_layer_ids.len == 0 or cfg.target_layer_ids.len > 16 or
        cfg.block_size < 2 or cfg.block_size > 16 or cfg.conv_kernel_size != 2) return error.GlmDraftTargetMismatch;
    try draft.validateTargetLayers(cfg.target_layer_ids, target.cfg.num_hidden_layers);
    const selector = &assistant.selector.?;
    const pred = mlx.getShape(selector.pred_codebook);
    if (pred.len != 2 or pred[0] != target.cfg.vocab_size or pred[1] != cfg.selector_rank or !std.mem.eql(c_int, pred, mlx.getShape(selector.succ_codebook))) return error.GlmDraftTargetMismatch;
}

pub const Proposal = struct {
    tokens: [16]u32 = undefined,
    parents: [16]i32 = undefined,
    /// The assistant's probability of each draft among its siblings; row 0 has none.
    conf: [16]f32 = undefined,
    count: usize,
};

pub fn proposeTree(assistant: *draft.DflashModel, context: *const draft.DflashCtx, target: *const forward.Model, pending: u32, max_nodes: usize) !Proposal {
    return proposeTreeWithChildren(assistant, context, target, pending, max_nodes, 4);
}

/// The assistant's best-first nodes in the order taken, before any row count is chosen.
pub const Drafted = struct {
    nodes: tree.DraftTree,
    allocator: std.mem.Allocator,
    pending: u32,

    pub fn deinit(self: *Drafted) void {
        self.nodes.deinit(self.allocator);
    }

    /// The verifier's rows over the nodes left, depth-first with the root first.
    pub fn proposal(self: *Drafted) !Proposal {
        try tree.preorder(self.allocator, &self.nodes);
        var result = Proposal{ .count = self.nodes.tokens.len + 1 };
        result.tokens[0] = self.pending;
        result.parents[0] = -1;
        result.conf[0] = 1;
        for (self.nodes.tokens, self.nodes.parents, self.nodes.cond, 1..) |token, parent, conf, row| {
            result.tokens[row] = token;
            result.parents[row] = parent + 1;
            result.conf[row] = conf;
        }
        return result;
    }
};

pub fn draftNodes(assistant: *draft.DflashModel, context: *const draft.DflashCtx, target: *const forward.Model, pending: u32, max_nodes: usize, children: usize) !Drafted {
    try validatePair(assistant, target);
    if (max_nodes == 0 or max_nodes > 15 or children == 0 or children > 16 or pending >= target.cfg.vocab_size) return error.InvalidGlmDraftTree;
    // forwardBlock uses spare cache rows; fork so an evaluation failure cannot alter committed context.
    var work = try cloneContext(assistant, context);
    defer work.deinit();
    var ops = Ops{ .s = assistant.s };
    defer ops.deinit();
    var noise: [16]u32 = @splat(assistant.config.mask_token_id);
    noise[0] = pending;
    const ids = try ops.own(mlx.mlx_array_new_data(&noise, &[_]c_int{ 1, @intCast(assistant.config.block_size) }, 2, .uint32));
    const embeds = try ops.own(try target.rawEmbedding(ids));
    // A tree of n nodes reads only the first n draft rows, plus their anchor.
    const needed: c_int = @intCast(@min(max_nodes + 1, assistant.config.block_size));
    const bounded = needed < assistant.config.block_size;
    const hidden = try ops.own(if (bounded)
        try draft.forwardBlockPrefix(assistant, &work, embeds, context.absLen(), @intCast(needed))
    else
        try draft.forwardBlock(assistant, &work, embeds, context.absLen()));
    const projected = try ops.own(try target.projectHead(try ops.slice(hidden, 1, 1, needed)));
    const logits = try ops.own(try draft.applyLogitTransforms(projected, assistant.config.output_multiplier, assistant.config.logit_softcap, assistant.s));
    const selector_input = try ops.slice(hidden, 1, 0, needed);
    var lattice = try tree.lattice(assistant.allocator, &assistant.selector.?, assistant.config.selector_top_k, selector_input, logits, pending, assistant.s);
    defer lattice.deinit(assistant.allocator);
    return .{ .nodes = try tree.bestFirstNodes(assistant.allocator, &lattice, .{ .max_nodes = max_nodes, .children = children }), .allocator = assistant.allocator, .pending = pending };
}

pub fn proposeTreeWithChildren(assistant: *draft.DflashModel, context: *const draft.DflashCtx, target: *const forward.Model, pending: u32, max_nodes: usize, children: usize) !Proposal {
    var drafted = try draftNodes(assistant, context, target, pending, max_nodes, children);
    defer drafted.deinit();
    return drafted.proposal();
}

fn sameArray(a: Arr, b: Arr, s: mlx.mlx_stream) !void {
    try std.testing.expectEqual(a.ctx == null, b.ctx == null);
    if (a.ctx == null) return;
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    if (mlx.mlx_array_size(a) == 0) return;
    const av = try draft.TinyFix.readF32(a, std.testing.allocator, s);
    defer std.testing.allocator.free(av);
    const bv = try draft.TinyFix.readF32(b, std.testing.allocator, s);
    defer std.testing.allocator.free(bv);
    try std.testing.expectEqualSlices(f32, av, bv);
}

fn sameBits(a: Arr, b: Arr, s: mlx.mlx_stream) !bool {
    if (a.ctx == null or b.ctx == null) return a.ctx == null and b.ctx == null;
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b)) or mlx.mlx_array_dtype(a) != mlx.mlx_array_dtype(b)) return false;
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const x = try ops.contiguous(a);
    const y = try ops.contiguous(b);
    try mlx.check(mlx.mlx_array_eval(x));
    try mlx.check(mlx.mlx_array_eval(y));
    const n = mlx.mlx_array_size(x) * mlx.mlx_array_itemsize(x);
    if (n == 0) return true;
    return std.mem.eql(u8, (mlx.mlx_array_data_uint8(x) orelse return error.MlxArrayDataNull)[0..n], (mlx.mlx_array_data_uint8(y) orelse return error.MlxArrayDataNull)[0..n]);
}

fn sameRequest(a: *const forward.Request, b: *const forward.Request, s: mlx.mlx_stream) !void {
    try std.testing.expectEqual(a.offset, b.offset);
    for (a.layers, b.layers) |left, right| {
        try std.testing.expectEqual(left.recurrent.initialized, right.recurrent.initialized);
        if (left.recurrent.initialized) {
            try sameArray(left.recurrent.conv_state, right.recurrent.conv_state, s);
            try sameArray(left.recurrent.ssm_state, right.recurrent.ssm_state, s);
        }
        try std.testing.expectEqual(left.attention.processed, right.attention.processed);
        for (left.attention.arrays(), right.attention.arrays()) |x, y| try sameArray(x, y, s);
    }
}

fn fixtureLinear(ops: *Ops, input: c_int, output: c_int) !draft.DflashLinear {
    return .{ .w = try ops.binary(.mul, try ops.ones(&.{ input, output }, .bfloat16), try ops.scalar(0.002, .bfloat16)), .scales = try ops.own(mlx.mlx_array_new()), .biases = try ops.own(mlx.mlx_array_new()) };
}

test "GLM DFlash actual branch oracle and commit match independent serial states" {
    const allocator = std.testing.allocator;
    const s = mlx.gpuStream();
    var weights = @import("model.zig").Weights.init(allocator);
    defer weights.deinit();
    const cfg = try forward.completeFixture(&weights);
    var iter = weights.map.iterator();
    var seed: usize = 5;
    while (iter.next()) |entry| {
        const value = entry.value_ptr;
        const shape = mlx.getShape(value.*);
        if (shape.len < 2 or mlx.mlx_array_dtype(value.*) != .bfloat16 or std.mem.endsWith(u8, entry.key_ptr.*, ".scales") or std.mem.endsWith(u8, entry.key_ptr.*, ".biases")) continue;
        const replacement = try draft.TinyFix.bf16ArrShaped(shape, seed, s);
        _ = mlx.mlx_array_free(value.*);
        value.* = replacement;
        seed += 1;
    }
    var embedding_codes: [4 * 32]u32 = undefined;
    for (&embedding_codes, 0..) |*value, i| {
        const row = i / 32;
        const col = i % 32;
        value.* = @as(u32, @intCast(1 + (row + col) % 7)) |
            (@as(u32, @intCast(1 + (3 * row + col) % 5)) << 8) |
            (@as(u32, @intCast(1 + (row + 2 * col) % 11)) << 16) |
            (@as(u32, @intCast(1 + (2 * row + col) % 13)) << 24);
    }
    const embedding = weights.map.getPtr("model.language_model.embed_tokens.weight").?;
    _ = mlx.mlx_array_free(embedding.*);
    embedding.* = mlx.mlx_array_new_data(&embedding_codes, &[_]c_int{ 4, 32 }, 2, .uint32);
    var target = try forward.Model.load(allocator, cfg, &weights, s);
    defer target.deinit();
    var request = try forward.Request.init(allocator, 4);
    defer request.deinit();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    var taps = [_]u32{ 0, 3 };
    var captures = try Captures.init(allocator, &taps);
    defer captures.deinit();
    request.capture = &captures.hook;
    const prefix_ids = try ops.own(mlx.mlx_array_new_data(&[_]u32{ 1, 2, 3 }, &[_]c_int{ 1, 3 }, 2, .uint32));
    _ = try ops.own(try target.forward(&request, prefix_ids));
    request.capture = null;
    var original = try cloneRequest(&request);
    defer original.deinit();
    const norm = try ops.ones(&.{128}, .bfloat16);
    const lin = try fixtureLinear(&ops, 128, 128);
    const conv = draft.DynConv{ .base_kernel = try ops.ones(&.{ 2, 2, 128 }, .bfloat16), .kernel_projection = try fixtureLinear(&ops, 128, 32) };
    var layers = [_]draft.DflashLayer{.{ .layer_type = .sliding_attention, .input_norm = norm, .post_attn_norm = norm, .q = lin, .q_norm = norm, .k = lin, .k_norm = norm, .v = lin, .o = lin, .gate = lin, .up = lin, .down = lin, .attention_conv = conv, .mlp_conv = conv }};
    var types = [_]draft.LayerType{.sliding_attention};
    var assistant = draft.DflashModel{
        .allocator = allocator,
        .s = s,
        .config = .{ .hidden_size = 128, .num_hidden_layers = 1, .num_attention_heads = 1, .num_key_value_heads = 1, .head_dim = 128, .intermediate_size = 128, .rms_norm_eps = 1e-5, .rope_theta = 10000, .sliding_window = 8, .layer_types = &types, .block_size = 4, .mask_token_id = 3, .target_layer_ids = &taps, .selector_rank = 2, .selector_top_k = 2, .conv_kernel_size = 2, .conv_group_size = 16 },
        .fc = try fixtureLinear(&ops, 256, 128),
        .enc_norm = norm,
        .final_norm = norm,
        .layers = &layers,
        .selector = .{ .pred_codebook = try ops.ones(&.{ 4, 2 }, .bfloat16), .succ_codebook = try ops.ones(&.{ 4, 2 }, .bfloat16), .hidden_projection = try fixtureLinear(&ops, 128, 2) },
    };
    var context = try draft.DflashCtx.init(allocator, &assistant, 0);
    defer context.deinit();
    try draft.appendContext(&assistant, &context, captures.hook.out, 0);
    try evaluateContext(&context);
    const proposal = try proposeTree(&assistant, &context, &target, 1, 4);
    try std.testing.expectEqual(@as(usize, 5), proposal.count);
    const explicit_default = try proposeTreeWithChildren(&assistant, &context, &target, 1, 4, 4);
    try std.testing.expectEqualSlices(u32, proposal.tokens[0..proposal.count], explicit_default.tokens[0..explicit_default.count]);
    try std.testing.expectEqualSlices(i32, proposal.parents[0..proposal.count], explicit_default.parents[0..explicit_default.count]);
    const chain = try proposeTreeWithChildren(&assistant, &context, &target, 1, 3, 1);
    try std.testing.expectEqual(@as(usize, 4), chain.count);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1, 2 }, chain.parents[0..chain.count]);
    try std.testing.expectError(error.InvalidGlmDraftTree, proposeTreeWithChildren(&assistant, &context, &target, 1, 3, 0));
    try std.testing.expectError(error.InvalidGlmDraftTree, proposeTreeWithChildren(&assistant, &context, &target, 1, 3, 17));
    try std.testing.expectEqual(@as(usize, 3), context.absLen());
    // The fixture head ties at token zero; its ancestry is root 1 -> token 0 -> token 0.
    const tokens = [_]u32{ 1, 0, 2, 0, 3 };
    const parents = [_]i32{ -1, 0, 0, 1, 2 };
    var verified = try verifyTreeOracle(&target, &request, &tokens, &parents, &taps);
    defer verified.deinit();
    try sameRequest(&request, &original, s);
    const left = try draft.TinyFix.readF32(verified.states[1].?.layers[0].recurrent.ssm_state, allocator, s);
    defer allocator.free(left);
    const right = try draft.TinyFix.readF32(verified.states[2].?.layers[0].recurrent.ssm_state, allocator, s);
    defer allocator.free(right);
    try std.testing.expect(!std.mem.eql(f32, left, right));
    for (0..tokens.len) |row| {
        var serial = try cloneRequest(&original);
        defer serial.deinit();
        var path: [16]usize = undefined;
        var count: usize = 0;
        var at: i32 = @intCast(row);
        while (at >= 0) : (at = parents[@intCast(at)]) {
            path[count] = @intCast(at);
            count += 1;
        }
        var cap = try Captures.init(allocator, &taps);
        defer cap.deinit();
        serial.capture = &cap.hook;
        while (count > 0) {
            count -= 1;
            const id = tokens[path[count]];
            const input = mlx.mlx_array_new_data(&id, &[_]c_int{ 1, 1 }, 2, .uint32);
            defer _ = mlx.mlx_array_free(input);
            const logits = try target.forwardLast(&serial, input, true);
            defer _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_array_eval(logits));
        }
        serial.capture = null;
        try sameRequest(&serial, &verified.states[row].?, s);
        for (cap.hook.out, verified.captures[row].?.hook.out) |x, y| try sameArray(x, y, s);
    }
    @import("glm5_dflash_model.zig").resetStats();
    var layerwise = try @import("glm5_dflash_model.zig").verify(&target, &request, &tokens, &parents, &taps, .serial_rows);
    defer layerwise.deinit();
    try std.testing.expectEqual(@as(usize, 0), @import("glm5_dflash_model.zig").branchFlushCount());
    try std.testing.expect(@import("glm5_dflash_model.zig").scratchBoundBytes() > 0);
    try std.testing.expectEqualSlices(u32, verified.targets[0..verified.count], layerwise.targets[0..layerwise.count]);
    const qmm_before = @import("glm5_dflash_qmm.zig").dispatchCount();
    const ffn_before = @import("glm5_dflash_ffn.zig").batchCount();
    var affine = try @import("glm5_dflash_model.zig").verify(&target, &request, &tokens, &parents, &taps, .affine_rows_ffn);
    defer affine.deinit();
    try std.testing.expectEqual(ffn_before + 1, @import("glm5_dflash_ffn.zig").batchCount());
    try std.testing.expectEqual(qmm_before, @import("glm5_dflash_qmm.zig").dispatchCount());
    try std.testing.expectEqualSlices(u32, layerwise.targets[0..layerwise.count], affine.targets[0..affine.count]);
    // Sampled decisions read these logits, so batched rows must equal serial rows bit for bit.
    try std.testing.expect(try sameBits(layerwise.logits, affine.logits, s));
    var affine_committed = try affine.prepareCommit(&request, 3, &.{}, s);
    defer affine_committed.deinit();
    try sameRequest(&affine_committed.states[3].?, &verified.states[3].?, s);
    var replayed = try layerwise.prepareCommit(&request, 3, &.{}, s);
    defer replayed.deinit();
    try sameRequest(&replayed.states[3].?, &verified.states[3].?, s);
    for ([_]usize{ 0, 1, 3 }) |row| {
        for (replayed.captures[row].?.hook.out, verified.captures[row].?.hook.out) |xcap, ycap| try sameArray(xcap, ycap, s);
    }
    const accepted = try commitVerified(&assistant, &context, &request, &verified, 3, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 3 }, accepted.rows[0..accepted.count]);
    try std.testing.expectEqual(@as(usize, 6), request.offset);
    try std.testing.expectEqual(request.offset, context.absLen());
    try std.testing.expectError(error.InvalidGlmDraftOffset, commitVerified(&assistant, &context, &request, &verified, 3, &.{}));
    request.reset();
    context.deinit();
    context = try draft.DflashCtx.init(allocator, &assistant, 0);
    const pending = try prefill(&assistant, &context, &target, &request, prefix_ids);
    const round = try roundTreeOracle(std.Io.Threaded.global_single_threaded.io(), &assistant, &context, &target, &request, pending, 4, 2, &.{});
    try std.testing.expect(round.count >= 1 and round.count <= 2);
    try std.testing.expectEqual(@as(usize, 5), round.verified_rows);
    try std.testing.expectEqual(@as(usize, 3) + round.count, request.offset);
    try std.testing.expectEqual(request.offset, context.absLen());
    const stopped = try roundTreeLayerwiseWithDecisions(std.Io.Threaded.global_single_threaded.io(), &assistant, &context, &target, &request, 0, 4, 3, &.{0}, .serial_rows, 4, null);
    try std.testing.expect(stopped.stopped and stopped.pending == null);
    try std.testing.expectEqual(@as(usize, 1), stopped.verified_rows);
    target.cfg.max_position_embeddings = 4096;
    for ([_]usize{ 1, 2, 4, 255, 2051 }) |prefix_length| {
        request.reset();
        const prompt = try allocator.alloc(u32, prefix_length);
        defer allocator.free(prompt);
        for (prompt, 0..) |*token, i| token.* = @intCast(i % 4);
        const inputs = mlx.mlx_array_new_data(prompt.ptr, &[_]c_int{ 1, @intCast(prefix_length) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(inputs);
        const logits = try target.forwardLast(&request, inputs, true);
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_eval(logits));
        var reference = try verifyTreeOracle(&target, &request, &tokens, &parents, &taps);
        defer reference.deinit();
        var batch = try @import("glm5_dflash_model.zig").verify(&target, &request, &tokens, &parents, &taps, .serial_rows);
        defer batch.deinit();
        try std.testing.expectEqualSlices(u32, reference.targets[0..reference.count], batch.targets[0..batch.count]);
        for ([_]usize{ 1, 2, 3 }) |budget| {
            var committed = try batch.prepareCommit(&request, budget, &.{}, s);
            defer committed.deinit();
            const keep = try tree.accept(&tokens, &parents, reference.targets[0..reference.count], budget, &.{});
            const last = keep.rows[keep.count - 1];
            try sameRequest(&committed.states[last].?, &reference.states[last].?, s);
            for (keep.rows[0..keep.count]) |row| for (committed.captures[row].?.hook.out, reference.captures[row].?.hook.out) |xcap, ycap| try sameArray(xcap, ycap, s);
        }
    }
}

test "GLM DFlash bounds proposal depth by budget and remaining context before verify" {
    const allocator = std.testing.allocator;
    const s = mlx.gpuStream();
    var weights = @import("model.zig").Weights.init(allocator);
    defer weights.deinit();
    const cfg = try forward.completeFixture(&weights);
    var iter = weights.map.iterator();
    var seed: usize = 5;
    while (iter.next()) |entry| {
        const value = entry.value_ptr;
        const shape = mlx.getShape(value.*);
        if (shape.len < 2 or mlx.mlx_array_dtype(value.*) != .bfloat16 or std.mem.endsWith(u8, entry.key_ptr.*, ".scales") or std.mem.endsWith(u8, entry.key_ptr.*, ".biases")) continue;
        const replacement = try draft.TinyFix.bf16ArrShaped(shape, seed, s);
        _ = mlx.mlx_array_free(value.*);
        value.* = replacement;
        seed += 1;
    }
    var target = try forward.Model.load(allocator, cfg, &weights, s);
    defer target.deinit();
    try std.testing.expectEqual(@as(u32, 16), target.cfg.max_position_embeddings);
    var request = try forward.Request.init(allocator, 4);
    defer request.deinit();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    var taps = [_]u32{ 0, 3 };
    const norm = try ops.ones(&.{128}, .bfloat16);
    const lin = try fixtureLinear(&ops, 128, 128);
    const conv = draft.DynConv{ .base_kernel = try ops.ones(&.{ 2, 2, 128 }, .bfloat16), .kernel_projection = try fixtureLinear(&ops, 128, 32) };
    var layers = [_]draft.DflashLayer{.{ .layer_type = .sliding_attention, .input_norm = norm, .post_attn_norm = norm, .q = lin, .q_norm = norm, .k = lin, .k_norm = norm, .v = lin, .o = lin, .gate = lin, .up = lin, .down = lin, .attention_conv = conv, .mlp_conv = conv }};
    var types = [_]draft.LayerType{.sliding_attention};
    var assistant = draft.DflashModel{
        .allocator = allocator,
        .s = s,
        .config = .{ .hidden_size = 128, .num_hidden_layers = 1, .num_attention_heads = 1, .num_key_value_heads = 1, .head_dim = 128, .intermediate_size = 128, .rms_norm_eps = 1e-5, .rope_theta = 10000, .sliding_window = 8, .layer_types = &types, .block_size = 4, .mask_token_id = 3, .target_layer_ids = &taps, .selector_rank = 2, .selector_top_k = 2, .conv_kernel_size = 2, .conv_group_size = 16 },
        .fc = try fixtureLinear(&ops, 256, 128),
        .enc_norm = norm,
        .final_norm = norm,
        .layers = &layers,
        .selector = .{ .pred_codebook = try ops.ones(&.{ 4, 2 }, .bfloat16), .succ_codebook = try ops.ones(&.{ 4, 2 }, .bfloat16), .hidden_projection = try fixtureLinear(&ops, 128, 2) },
    };
    var context = try draft.DflashCtx.init(allocator, &assistant, 0);
    defer context.deinit();
    var prefix: [14]u32 = undefined;
    for (&prefix, 0..) |*t, i| t.* = @intCast(i % 4);
    const prefix_ids = try ops.own(mlx.mlx_array_new_data(&prefix, &[_]c_int{ 1, 14 }, 2, .uint32));
    const pending = try prefill(&assistant, &context, &target, &request, prefix_ids);
    try std.testing.expectEqual(@as(usize, 14), request.offset);
    try std.testing.expectEqual(request.offset, context.absLen());

    for ([_]usize{ 1, 4 }) |children| {
        for ([_]usize{ 1, 2, 3 }) |budget| {
            var req = try cloneRequest(&request);
            defer req.deinit();
            var ctx = try cloneContext(&assistant, &context);
            defer ctx.deinit();
            const limit = @min(budget, 2);
            const proposal = try proposeRound(&assistant, &ctx, &target, &req, pending, 2, budget, &.{}, children);
            var rows: [16]usize = undefined;
            for (proposal.parents[0..proposal.count], 0..) |parent, i| {
                rows[i] = if (parent < 0) 1 else rows[@intCast(parent)] + 1;
                try std.testing.expect(rows[i] <= limit);
            }
            const result = try roundTreeLayerwiseWithDecisions(std.Io.Threaded.global_single_threaded.io(), &assistant, &ctx, &target, &req, pending, 2, budget, &.{}, .serial_rows, children, null);
            try std.testing.expectEqual(result.count, req.offset - 14);
            try std.testing.expect(result.count >= 1 and result.count <= limit);
            try std.testing.expectEqual(req.offset, ctx.absLen());

            var serial = try cloneRequest(&request);
            defer serial.deinit();
            for (result.tokens[0..result.count]) |token| {
                const input = mlx.mlx_array_new_data(&token, &[_]c_int{ 1, 1 }, 2, .uint32);
                defer _ = mlx.mlx_array_free(input);
                const logits = try target.forwardLast(&serial, input, true);
                defer _ = mlx.mlx_array_free(logits);
                try mlx.check(mlx.mlx_array_eval(logits));
            }
            try sameRequest(&serial, &req, s);
        }
    }

    // The policy picks the rows: free rows buy the widest tree, dear rows keep the narrowest, and the
    // bounds hold either way; a tree is cut by depth first.
    var open_request = try forward.Request.init(allocator, 4);
    defer open_request.deinit();
    var open_context = try draft.DflashCtx.init(allocator, &assistant, 0);
    defer open_context.deinit();
    var open_prefix: [6]u32 = undefined;
    for (&open_prefix, 0..) |*t, i| t.* = @intCast(i % 4);
    const open_ids = try ops.own(mlx.mlx_array_new_data(&open_prefix, &[_]c_int{ 1, 6 }, 2, .uint32));
    const open_pending = try prefill(&assistant, &open_context, &target, &open_request, open_ids);
    var calib = policy.Calibration{};
    var guess: f32 = 0;
    while (guess < 1) : (guess += 0.01) for (0..40) |_| calib.observe(guess, true);
    const cheap = policy.linearLadder(1, 0);
    const dear = policy.linearLadder(1, 100);
    for ([_]usize{ 1, 4 }) |children| {
        const wide = try proposeRoundAdaptive(&assistant, &open_context, &target, &open_request, open_pending, 100, &.{}, children, .{ .min = 2, .max = 4 }, &calib, &cheap);
        try std.testing.expectEqual(@as(usize, 4), wide.count);
        const narrow = try proposeRoundAdaptive(&assistant, &open_context, &target, &open_request, open_pending, 100, &.{}, children, .{ .min = 2, .max = 4 }, &calib, &dear);
        try std.testing.expectEqual(@as(usize, 2), narrow.count);
        const pinned = try proposeRoundAdaptive(&assistant, &open_context, &target, &open_request, open_pending, 100, &.{}, children, .{ .min = 3, .max = 3 }, &calib, &dear);
        try std.testing.expectEqual(@as(usize, 3), pinned.count);
        const short = try proposeRoundAdaptive(&assistant, &open_context, &target, &open_request, open_pending, 2, &.{}, children, .{ .min = 2, .max = 4 }, &calib, &cheap);
        var rows: [16]usize = undefined;
        for (short.parents[0..short.count], 0..) |parent, i| {
            rows[i] = if (parent < 0) 1 else rows[@intCast(parent)] + 1;
            try std.testing.expect(rows[i] <= 2);
        }
        try std.testing.expect(wide.conf[1] > 0 and wide.conf[1] <= 1);
        const plain = try proposeRoundAdaptive(&assistant, &open_context, &target, &open_request, open_pending, 1, &.{}, children, .{ .min = 2, .max = 4 }, &calib, &cheap);
        try std.testing.expectEqual(@as(usize, 1), plain.count);
    }
}

test "trimToDepth keeps the rows within the depth and re-indexes their parents" {
    var proposal = Proposal{ .count = 5 };
    proposal.tokens[0..5].* = .{ 10, 11, 12, 13, 14 };
    proposal.parents[0..5].* = .{ -1, 0, 1, 0, 3 };
    trimToDepth(&proposal, 2);
    try std.testing.expectEqualSlices(u32, &.{ 10, 11, 13 }, proposal.tokens[0..proposal.count]);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0 }, proposal.parents[0..proposal.count]);
    trimToDepth(&proposal, 1);
    try std.testing.expectEqual(@as(usize, 1), proposal.count);
}

/// Stored assistant matrices are used directly: no load-time requantization.
pub fn loadAssistantStored(io: std.Io, allocator: std.mem.Allocator, directory: []const u8, target: *const forward.Model) !draft.DflashModel {
    var dir = try std.Io.Dir.openDirAbsolute(io, directory, .{});
    defer dir.close(io);
    const raw = try dir.readFileAlloc(io, "config.json", allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(raw);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.GlmDraftTargetMismatch;
    const vocab = parsed.value.object.get("vocab_size") orelse return error.GlmDraftTargetMismatch;
    const count = parsed.value.object.get("num_target_layers") orelse return error.GlmDraftTargetMismatch;
    if (vocab != .integer or vocab.integer != target.cfg.vocab_size or count != .integer or count.integer != target.cfg.num_hidden_layers) return error.GlmDraftTargetMismatch;
    var assistant = try draft.loadDflashQuant(io, allocator, target.s, directory, 0);
    errdefer assistant.deinit();
    try validatePair(&assistant, target);
    _ = try assistantStorage(&assistant);
    return assistant;
}

pub const AssistantStorage = struct {
    bits: u32,
    group_size: u32,
    dense_linears: usize = 0,
    affine_linears: usize = 0,
    pub fn label(self: AssistantStorage) []const u8 {
        return if (self.affine_linears == 0) "BF16" else if (self.bits == 4 and self.group_size == 64) "A4g64" else if (self.bits == 8) "A8g128" else "A6g128";
    }
    fn add(self: *AssistantStorage, linear: *const draft.DflashLinear) !void {
        if (linear.w.ctx == null) return error.UnsupportedGlmDraftStorage;
        const ws = mlx.getShape(linear.w);
        if (ws.len != 2 or ws[0] <= 0 or ws[1] <= 0) return error.UnsupportedGlmDraftStorage;
        if (linear.bits == 0) {
            if (mlx.mlx_array_dtype(linear.w) != .bfloat16 or linear.scales.ctx != null or linear.biases.ctx != null) return error.UnsupportedGlmDraftStorage;
            self.dense_linears += 1;
            return;
        }
        if (!((linear.bits == 4 and linear.group_size == 64) or ((linear.bits == 6 or linear.bits == 8) and linear.group_size == 128)) or linear.bits != self.bits or linear.group_size != self.group_size or
            mlx.mlx_array_dtype(linear.w) != .uint32 or linear.scales.ctx == null or linear.biases.ctx == null or
            mlx.mlx_array_dtype(linear.scales) != .bfloat16 or mlx.mlx_array_dtype(linear.biases) != .bfloat16) return error.UnsupportedGlmDraftStorage;
        const ss = mlx.getShape(linear.scales);
        if (ss.len != 2 or ss[0] != ws[0] or ss[1] <= 0 or !std.mem.eql(c_int, ss, mlx.getShape(linear.biases)) or
            @as(u64, @intCast(ws[1])) * 32 != @as(u64, @intCast(ss[1])) * linear.group_size * linear.bits) return error.UnsupportedGlmDraftStorage;
        self.affine_linears += 1;
    }
};

/// Validate every contracted matrix, including dynamic convolutions and selector.
/// Dense small matrices may be retained alongside one uniform stored affine rate.
pub fn assistantStorage(assistant: *const draft.DflashModel) !AssistantStorage {
    var result = AssistantStorage{ .bits = assistant.fc.bits, .group_size = assistant.fc.group_size };
    try result.add(&assistant.fc);
    for (assistant.layers) |*layer| {
        for ([_]*const draft.DflashLinear{ &layer.q, &layer.k, &layer.v, &layer.o, &layer.gate, &layer.up, &layer.down }) |linear| try result.add(linear);
        if (layer.attention_conv) |*conv| try result.add(&conv.kernel_projection);
        if (layer.mlp_conv) |*conv| try result.add(&conv.kernel_projection);
    }
    if (assistant.selector) |*selector| try result.add(&selector.hidden_projection);
    return result;
}

/// Context capture is evaluated per caller-selected prefill chunk, without retaining full hidden history.
pub fn prefill(assistant: *const draft.DflashModel, context: *draft.DflashCtx, target: *const forward.Model, request: *forward.Request, ids: Arr) !u32 {
    try validatePair(assistant, target);
    if (request.offset != context.absLen() or request.capture != null) return error.InvalidGlmDraftOffset;
    var next = try cloneRequest(request);
    errdefer next.deinit();
    var next_context = try cloneContext(assistant, context);
    errdefer next_context.deinit();
    var captures = try Captures.init(request.allocator, assistant.config.target_layer_ids);
    defer captures.deinit();
    next.capture = &captures.hook;
    const logits = try target.forwardLast(&next, ids, true);
    defer _ = mlx.mlx_array_free(logits);
    next.capture = null;
    const pending = try argmax(logits, target.s);
    try draft.appendContext(assistant, &next_context, captures.hook.out, request.offset);
    try evaluateContext(&next_context);
    request.deinit();
    request.* = next;
    context.deinit();
    context.* = next_context;
    return pending;
}

pub const RoundResult = struct {
    tokens: [16]u32 = undefined,
    count: usize,
    pending: ?u32,
    stopped: bool,
    verified_rows: usize,
    accepted_drafts: usize,
    draft_ns: u64,
    verify_ns: u64,
    commit_ns: u64,
    replay_ns: u64 = 0,
    verifier: []const u8 = "serial_branch_oracle",
    /// The proposal rows the round kept, root first.
    rows: [16]u32 = undefined,
};

/// Test oracle round: every proposed node runs the serial target forward.
pub fn roundTreeOracle(io: std.Io, assistant: *draft.DflashModel, context: *draft.DflashCtx, target: *const forward.Model, request: *forward.Request, pending: u32, max_nodes: usize, budget: usize, eos: []const u32) !RoundResult {
    if (budget == 0) return error.InvalidGlmDraftBudget;
    if (request.offset != context.absLen()) return error.InvalidGlmDraftOffset;
    const clock = @import("io_util.zig").Stopwatch;
    var timer = clock.init(io);
    const proposal = if (budget == 1 or std.mem.indexOfScalar(u32, eos, pending) != null) blk: {
        var one = Proposal{ .count = 1 };
        one.tokens[0] = pending;
        one.parents[0] = -1;
        break :blk one;
    } else try proposeTree(assistant, context, target, pending, max_nodes);
    const draft_ns = timer.read();
    timer.reset();
    var verified = try verifyTreeOracle(target, request, proposal.tokens[0..proposal.count], proposal.parents[0..proposal.count], assistant.config.target_layer_ids);
    defer verified.deinit();
    const verify_ns = timer.read();
    timer.reset();
    const kept = try commitVerified(assistant, context, request, &verified, budget, eos);
    var result = RoundResult{ .count = kept.count, .pending = kept.pending, .stopped = kept.stopped, .verified_rows = proposal.count, .accepted_drafts = kept.count - 1, .draft_ns = draft_ns, .verify_ns = verify_ns, .commit_ns = timer.read() };
    for (kept.rows[0..kept.count], 0..) |row, i| result.tokens[i] = proposal.tokens[row];
    result.rows = kept.rows;
    return result;
}

test {
    _ = @import("glm5_dflash_kda.zig");
}

test {
    _ = @import("glm5_dflash_model.zig");
}

pub const DecisionHook = struct {
    ctx: *anyopaque,
    apply: *const fn (*anyopaque, *@import("glm5_dflash_model.zig").Verified, usize, []const u32) anyerror!void,
};

pub fn roundTreeLayerwiseWithDecisions(io: std.Io, assistant: *draft.DflashModel, context: *draft.DflashCtx, target: *const forward.Model, request: *forward.Request, pending: u32, max_nodes: usize, budget: usize, eos: []const u32, mode: @import("glm5_dflash_kda.zig").ProjectionMode, children: usize, decisions: ?DecisionHook) !RoundResult {
    var timer = @import("io_util.zig").Stopwatch.init(io);
    const proposal = try proposeRound(assistant, context, target, request, pending, max_nodes, budget, eos, children);
    const draft_ns = timer.read();
    timer.reset();
    var layerwise = try @import("glm5_dflash_model.zig").verify(target, request, proposal.tokens[0..proposal.count], proposal.parents[0..proposal.count], assistant.config.target_layer_ids, mode);
    defer layerwise.deinit();
    if (decisions) |hook| try hook.apply(hook.ctx, &layerwise, budget, eos);
    const verify_ns = timer.read();
    var result = try finishRound(io, assistant, context, target, request, &proposal, &layerwise, budget, eos);
    result.draft_ns = draft_ns;
    result.verify_ns = verify_ns;
    result.verifier = if (mode == .affine_rows_ffn) "layerwise_tree_affine_ffn_tiles" else if (mode == .affine_rows) "layerwise_tree_affine_row_tiles" else "layerwise_tree_serial_projections";
    return result;
}

/// A round's draft: the assistant's tree cut to the tokens the budget and the target's remaining
/// positions can still take, or the pending token alone at the last one or an EOS.
pub fn proposeRound(assistant: *draft.DflashModel, context: *const draft.DflashCtx, target: *const forward.Model, request: *const forward.Request, pending: u32, max_nodes: usize, budget: usize, eos: []const u32, children: usize) !Proposal {
    if (children == 0 or children > 16) return error.InvalidGlmDraftTree;
    try validatePair(assistant, target);
    if (budget == 0) return error.InvalidGlmDraftBudget;
    if (request.offset != context.absLen()) return error.InvalidGlmDraftOffset;
    const depth = @min(budget, target.cfg.max_position_embeddings -| request.offset);
    if (max_nodes == 0 or depth <= 1 or std.mem.indexOfScalar(u32, eos, pending) != null) {
        var one = Proposal{ .count = 1 };
        one.tokens[0] = pending;
        one.parents[0] = -1;
        return one;
    }
    var proposal = try proposeTreeWithChildren(assistant, context, target, pending, max_nodes, children);
    trimToDepth(&proposal, depth);
    return proposal;
}

/// A round whose row count the policy chooses from the assistant's own confidence: the widest tree
/// the bounds allow is drafted, then cut to the rows with the best expected tokens per ms.
pub fn proposeRoundAdaptive(assistant: *draft.DflashModel, context: *const draft.DflashCtx, target: *const forward.Model, request: *const forward.Request, pending: u32, budget: usize, eos: []const u32, children: usize, bounds: struct { min: u32, max: u32 }, calib: *const policy.Calibration, costs: *const policy.Costs) !Proposal {
    if (children == 0 or children > 16 or bounds.min < 2 or bounds.max < bounds.min or bounds.max > 16) return error.InvalidGlmDraftTree;
    try validatePair(assistant, target);
    if (budget == 0) return error.InvalidGlmDraftBudget;
    if (request.offset != context.absLen()) return error.InvalidGlmDraftOffset;
    const depth = @min(budget, target.cfg.max_position_embeddings -| request.offset);
    if (depth <= 1 or std.mem.indexOfScalar(u32, eos, pending) != null) {
        var one = Proposal{ .count = 1 };
        one.tokens[0] = pending;
        one.parents[0] = -1;
        return one;
    }
    var drafted = try draftNodes(assistant, context, target, pending, bounds.max - 1, children);
    defer drafted.deinit();
    try drafted.nodes.cut(drafted.allocator, depth, bounds.max - 1);
    const rows = policy.pickTreeRows(calib, drafted.nodes.parents, drafted.nodes.cond, bounds.min, bounds.max, costs);
    try drafted.nodes.cut(drafted.allocator, depth, rows - 1);
    return drafted.proposal();
}

/// Drops the nodes whose chain from the root is longer than `depth` rows; parents precede children.
fn trimToDepth(proposal: *Proposal, depth: usize) void {
    var rows: [16]usize = undefined;
    var moved: [16]i32 = undefined;
    var kept: usize = 0;
    for (0..proposal.count) |i| {
        const parent = proposal.parents[i];
        rows[i] = if (parent < 0) 1 else rows[@intCast(parent)] + 1;
        if (rows[i] > depth) continue;
        moved[i] = @intCast(kept);
        proposal.tokens[kept] = proposal.tokens[i];
        proposal.conf[kept] = proposal.conf[i];
        proposal.parents[kept] = if (parent < 0) -1 else moved[@intCast(parent)];
        kept += 1;
    }
    proposal.count = kept;
}

/// Commits a verified round (decisions already applied): target state and assistant context
/// together, or neither.
pub fn finishRound(io: std.Io, assistant: *const draft.DflashModel, context: *draft.DflashCtx, target: *const forward.Model, request: *forward.Request, proposal: *const Proposal, layerwise: *const @import("glm5_dflash_model.zig").Verified, budget: usize, eos: []const u32) !RoundResult {
    var timer = @import("io_util.zig").Stopwatch.init(io);
    var verified = try layerwise.prepareCommitConsuming(request, budget, eos, target.s);
    defer verified.deinit();
    const replay_ns = timer.read();
    timer.reset();
    const kept = commitVerified(assistant, context, request, &verified, budget, eos) catch |err| {
        request.failed = true;
        return err;
    };
    var result = RoundResult{ .count = kept.count, .pending = kept.pending, .stopped = kept.stopped, .verified_rows = proposal.count, .accepted_drafts = kept.count - 1, .draft_ns = 0, .verify_ns = 0, .replay_ns = replay_ns, .commit_ns = timer.read(), .verifier = "layerwise_tree_grouped" };
    for (kept.rows[0..kept.count], 0..) |row, i| result.tokens[i] = proposal.tokens[row];
    result.rows = kept.rows;
    return result;
}

test {
    _ = @import("glm5_dflash_qmm.zig");
    _ = @import("glm5_dflash_rows.zig");
}

test {
    _ = @import("glm5_dflash_ffn.zig");
}

test "GLM draft stored BF16 A4 A6 and A8 validate packed geometry without changing arrays" {
    const nil = mlx.mlx_array{ .ctx = null };
    const codes: [32]u32 = @splat(0);
    const scales: [2]u16 = .{ 0x3f80, 0 };
    for ([_][2]u32{ .{ 4, 64 }, .{ 6, 128 }, .{ 8, 128 } }) |spec| {
        const bits = spec[0];
        const group = spec[1];
        const cols: c_int = @intCast(group * bits / 32);
        const w = mlx.mlx_array_new_data(&codes, &[_]c_int{ 1, cols }, 2, .uint32);
        defer _ = mlx.mlx_array_free(w);
        const scale = mlx.mlx_array_new_data(&scales, &[_]c_int{ 1, 1 }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(scale);
        var linear = draft.DflashLinear{ .w = w, .scales = scale, .biases = scale, .bits = bits, .group_size = group };
        var storage = AssistantStorage{ .bits = bits, .group_size = group };
        try storage.add(&linear);
        try std.testing.expectEqual(@as(usize, 1), storage.affine_linears);
        try std.testing.expectEqual(w.ctx, linear.w.ctx);
        try std.testing.expectEqual(scale.ctx, linear.scales.ctx);
        try std.testing.expectEqual(scale.ctx, linear.biases.ctx);
        try std.testing.expectEqualStrings(if (bits == 4) "A4g64" else if (bits == 6) "A6g128" else "A8g128", storage.label());
        linear.group_size = if (group == 64) 128 else 64;
        try std.testing.expectError(error.UnsupportedGlmDraftStorage, storage.add(&linear));
        linear.group_size = group;
        linear.biases = nil;
        try std.testing.expectError(error.UnsupportedGlmDraftStorage, storage.add(&linear));
        const bad = mlx.mlx_array_new_data(&scales, &[_]c_int{ 1, 2 }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(bad);
        linear.biases = bad;
        try std.testing.expectError(error.UnsupportedGlmDraftStorage, storage.add(&linear));
        linear.scales = bad;
        try std.testing.expectError(error.UnsupportedGlmDraftStorage, storage.add(&linear));
        linear.scales = scale;
        linear.biases = scale;
        linear.bits = if (bits == 6) 8 else 6;
        linear.group_size = 128;
        try std.testing.expectError(error.UnsupportedGlmDraftStorage, storage.add(&linear));
    }
    const w = mlx.mlx_array_new_data(&scales, &[_]c_int{ 1, 2 }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(w);
    const dense = draft.DflashLinear{ .w = w, .scales = nil, .biases = nil };
    var storage = AssistantStorage{ .bits = 0, .group_size = 0 };
    try storage.add(&dense);
    try std.testing.expectEqual(@as(usize, 1), storage.dense_linears);
    try std.testing.expectEqualStrings("BF16", storage.label());
}

test "GLM N2 best-first selection is independent of unused future lattice depths" {
    const k = 4;
    var cands: [7 * k]i32 = undefined;
    var unary: [7 * k]f32 = undefined;
    var anchor: [k]f32 = undefined;
    var edges: [6 * k * k]f32 = undefined;
    for (&cands, &unary, 0..) |*id, *v, i| {
        id.* = @intCast(100 + i);
        v.* = @as(f32, @floatFromInt((i * 7) % 13)) / 8 - 0.5;
    }
    for (&anchor, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 3)) / 4;
    for (&edges, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 3) % 17)) / 8 - 0.75;
    const full = tree.Lattice{ .m = 7, .k = k, .cands = &cands, .unary = &unary, .e0 = &anchor, .e = &edges };
    const short = tree.Lattice{ .m = 2, .k = k, .cands = cands[0 .. 2 * k], .unary = unary[0 .. 2 * k], .e0 = &anchor, .e = edges[0 .. k * k] };
    for ([_]usize{ 1, 2, 4 }) |children| {
        var original = try tree.bestFirstTree(std.testing.allocator, &full, .{ .max_nodes = 2, .children = children });
        defer original.deinit(std.testing.allocator);
        var bounded = try tree.bestFirstTree(std.testing.allocator, &short, .{ .max_nodes = 2, .children = children });
        defer bounded.deinit(std.testing.allocator);
        try std.testing.expectEqualSlices(u32, original.tokens, bounded.tokens);
        try std.testing.expectEqualSlices(i32, original.parents, bounded.parents);
        try std.testing.expectEqualSlices(u32, original.depth, bounded.depth);
    }
}
