//! Native GLM routed experts over Sushi's shared expert cache: BF16 source experts or EXL3 pack banks.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const stream = @import("expert_stream.zig");
const imatrix_capture = @import("imatrix.zig");
const exl3 = @import("sushi_exl3");
const fp8_block = @import("fp8_block.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;

pub const Budget = struct { total: u64, trunk: u64, reserve: u64, cache: u64, fixed: u64, carried: u64 = 0 };
fn bytesPerExpert(g: stream.Geometry) !u64 {
    return stream.expertBytes(try std.math.mul(u32, 2, g.intermediate), g.hidden, g.intermediate);
}

/// Retained lazy trunk metadata must fit this bound before tensor evaluation.
pub fn trunkLimit(cfg: *const model.ModelConfig, total: u64, reserve: u64) !u64 {
    const g = cfg.expertGeometry();
    const bytes = try bytesPerExpert(g);
    const base = try plan(total, 0, reserve, g, bytes);
    const minimum_cache = try std.math.mul(u64, g.layers - g.first_moe_layer, bytes);
    return total - base.fixed - minimum_cache;
}

pub fn plan(total: u64, trunk: u64, reserve: u64, geometry: stream.Geometry, expert_bytes: u64) !Budget {
    if (geometry.layers <= geometry.first_moe_layer or geometry.experts == 0 or expert_bytes == 0 or reserve == 0) return error.InvalidGlmStreamBudget;
    const layers: u64 = geometry.layers - geometry.first_moe_layer;
    const workspace = std.math.mul(u64, geometry.experts, expert_bytes) catch return error.InvalidGlmStreamBudget;
    // Covers page rounding, imported-array metadata and prepared trunk copies.
    const overhead = 64 * 1024 * 1024;
    var fixed = std.math.add(u64, trunk, reserve) catch return error.InvalidGlmStreamBudget;
    for ([_]u64{ workspace, stream.BOUNCE_BYTES, overhead }) |n| fixed = std.math.add(u64, fixed, n) catch return error.InvalidGlmStreamBudget;
    if (fixed >= total) return error.SsdBudgetBelowResident;
    const per_slot = std.math.mul(u64, layers, expert_bytes) catch return error.InvalidGlmStreamBudget;
    const slots = @min((total - fixed) / per_slot, geometry.experts);
    if (slots == 0) return error.SsdBudgetBelowResident;
    return .{ .total = total, .trunk = trunk, .reserve = reserve, .cache = slots * per_slot, .fixed = fixed };
}

test "GLM stream CPU budget refuses underfunding and includes every resident slab" {
    const t = std.testing;
    const g = stream.Geometry{ .layers = 4, .first_moe_layer = 3, .experts = 4, .hidden = 32, .intermediate = 16 };
    const b = try plan(1024 * 1024 * 1024, 100, 200, g, 3072);
    try t.expectEqual(@as(u64, 4 * 3072), b.cache);
    try t.expect(b.fixed >= 100 + 200 + stream.BOUNCE_BYTES + 4 * 3072);
    try t.expect(b.fixed + b.cache <= b.total);
    try t.expectError(error.SsdBudgetBelowResident, plan(100, 100, 200, g, 3072));
    try t.expectError(error.InvalidGlmStreamBudget, plan(std.math.maxInt(u64), std.math.maxInt(u64), 1, g, 3072));
}

test "GLM stream CPU trunk bound admits one slot before trunk materialization" {
    const cfg = model.ModelConfig{ .model_type = "glm5_next", .num_hidden_layers = 4, .first_k_dense_replace = 3, .num_experts = 4, .hidden_size = 32, .moe_intermediate_size = 16 };
    const total = 1024 * 1024 * 1024;
    const reserve = 128 * 1024 * 1024;
    const max_trunk = try trunkLimit(&cfg, total, reserve);
    const g = stream.Geometry{ .layers = 4, .first_moe_layer = 3, .experts = 4, .hidden = 32, .intermediate = 16 };
    const b = try plan(total, max_trunk, reserve, g, 3072);
    try std.testing.expectEqual(@as(u64, 3072), b.cache);
    try std.testing.expectEqual(total, b.fixed + b.cache);
    try std.testing.expectError(error.SsdBudgetBelowResident, plan(total, max_trunk + 1, reserve, g, 3072));
}

/// Conservative one-request live tensor bound. Streamed forwards synchronize each
/// layer; speculative tapes and concurrent requests are outside this admission.
pub fn minimumReserve(cfg: *const model.ModelConfig, tokens: usize, chunk: usize) !u64 {
    if (tokens == 0 or tokens > cfg.max_position_embeddings or chunk == 0 or chunk > 512 or chunk > tokens or cfg.full_attention_interval == 0 or cfg.num_hidden_layers == 0) return error.InvalidGlmStreamBudget;
    const n: u128 = tokens;
    const t: u128 = chunk;
    const h: u128 = cfg.hidden_size;
    const heads: u128 = cfg.num_attention_heads;
    const attention_layers: u128 = cfg.num_hidden_layers / cfg.full_attention_interval;
    const linear_layers: u128 = cfg.num_hidden_layers - attention_layers;
    const d: u128 = cfg.linear_key_head_dim;
    const width: u128 = @as(u128, cfg.linear_num_value_heads) * d;
    const recurrent = linear_layers * (width * d * 4 + width * 3 * 3 * 2);
    // Capacity growth and update may retain both old and new arrays.
    const caches = 4 * n * attention_layers * (@as(u128, cfg.mla_kv_lora_rank) * 2 + @as(u128, cfg.indexer_head_dim) * 8);
    const activations = t * (h * 128 + width * 96 + @as(u128, cfg.vocab_size) * 8 + @as(u128, cfg.num_experts_per_tok) * (@as(u128, cfg.moe_intermediate_size) * 16 + h * 8));
    const attention_scratch = heads * t * n * 8 + t * heads * (@as(u128, cfg.mla_qk_nope_head_dim) + cfg.mla_v_head_dim) * 8;
    const prepared = attention_layers * heads * (@as(u128, cfg.mla_qk_nope_head_dim) + cfg.mla_v_head_dim) * @as(u128, cfg.mla_kv_lora_rank) * 2;
    return std.math.cast(u64, prepared + recurrent * 2 + caches + activations + attention_scratch + 64 * 1024 * 1024) orelse error.InvalidGlmStreamBudget;
}

/// What the NAX arms a capture takes add to `minimumReserve`: the KDA cluster banks resident from load and each
/// arm's scratch for the one layer a synchronous capture keeps pending. Zero under the reference arms.
pub fn armsReserve(cfg: *const model.ModelConfig, chunk: usize) !u64 {
    if (cfg.full_attention_interval == 0) return error.InvalidGlmStreamBudget;
    const cluster = @import("glm5_kda_prefill_cluster.zig");
    const attention = @import("glm5_attention.zig");
    const layers = cfg.num_hidden_layers;
    const mla_layers = layers / cfg.full_attention_interval;
    const resident = if (cluster.enabled()) (layers - mla_layers) * cluster.weight_bytes else 0;
    const scratch = (try @import("glm5_a6_dense_once.zig").transientBudget(chunk, 1)) +
        (try @import("glm5_mla_prefill_batch.zig").transientBudget(chunk, 1)) +
        (try @import("glm5_attention_nax_packed.zig").transientBudget(chunk, 1)) +
        (try attention.packedCadenceTransientBudget(chunk, 1)) +
        (try @import("glm5_indexpool_nax.zig").transientBudget(chunk, 1)) +
        (try cluster.transientBudget(chunk, 1)) +
        (try @import("glm5_attention_decode_batch.zig").transientBudget(1));
    return std.math.add(u64, resident, scratch) catch error.InvalidGlmStreamBudget;
}

/// The BF16 HC residual one window carries between layers.
fn carriedPerWindow(cfg: *const model.ModelConfig, max_tokens: usize) !u64 {
    return std.math.mul(u64, max_tokens, @as(u64, cfg.hc_count) * cfg.hidden_size * 2) catch error.InvalidGlmStreamBudget;
}

/// A layer-major teacher capture: the window-major capture's bill plus every batch window's HC residual. The cache
/// keeps one slot per MoE layer, since the batch reads each layer whole into the union workspace.
pub fn layerMajorBudget(cfg: *const model.ModelConfig, total: u64, trunk: u64, reserve: u64, max_tokens: usize, max_chunk: usize, windows: u32) !Budget {
    if (windows == 0) return error.InvalidGlmStreamBudget;
    const carried = std.math.mul(u64, windows, try carriedPerWindow(cfg, max_tokens)) catch return error.GlmLayerMajorBudgetExceeded;
    const held = std.math.add(u64, reserve, carried) catch return error.GlmLayerMajorBudgetExceeded;
    var b = captureBudget(cfg, total, trunk, held, max_tokens, max_chunk) catch |e| return if (e == error.SsdBudgetBelowResident) error.GlmLayerMajorBudgetExceeded else e;
    const g = cfg.expertGeometry();
    b.reserve = reserve;
    b.carried = carried;
    b.cache = @as(u64, g.layers - g.first_moe_layer) * try bytesPerExpert(g);
    return b;
}

/// The largest batch, at most `cap` windows, whose layer-major bill fits `total`.
pub fn layerMajorWindows(cfg: *const model.ModelConfig, total: u64, trunk: u64, reserve: u64, max_tokens: usize, max_chunk: usize, cap: u32) !u32 {
    const one = try layerMajorBudget(cfg, total, trunk, reserve, max_tokens, max_chunk, 1);
    const fits = (total - one.fixed - one.cache) / one.carried + 1;
    return @intCast(@min(fits, cap));
}

test "GLM teacher reserve bills the NAX arms it takes and nothing under the reference arms" {
    const base = @import("glm5_model.zig");
    const transformer = @import("transformer.zig");
    const gate = transformer.vqmm_nax_probe_override;
    defer {
        transformer.vqmm_nax_probe_override = gate;
        base.leaveTeacher();
    }
    const cfg = model.ModelConfig{ .model_type = "glm5_next", .num_hidden_layers = 45, .full_attention_interval = 4 };
    transformer.vqmm_nax_probe_override = true;
    base.enterTeacher();
    const on = try armsReserve(&cfg, 512);
    // 34 KDA layers keep a prepared 320x4096 BF16 cluster bank; the 512-row MLA head batches and B1 decode scratch are live.
    try std.testing.expect(on >= 34 * 320 * 4096 * 2 + 512 * 196608 + (32 << 20));
    try std.testing.expectEqual(on, try armsReserve(&cfg, 512));
    transformer.vqmm_nax_probe_override = false;
    base.enterTeacher();
    try std.testing.expectEqual(@as(u64, 0), try armsReserve(&cfg, 512));
}

test "GLM layer-major CPU budget bills every batch window's HC residual and refuses by name" {
    const t = std.testing;
    const cfg = model.ModelConfig{
        .model_type = "glm5_next",
        .hidden_size = 4096,
        .vocab_size = 154880,
        .num_hidden_layers = 45,
        .first_k_dense_replace = 3,
        .num_experts = 288,
        .num_experts_per_tok = 8,
        .moe_intermediate_size = 2048,
        .hc_count = 4,
        .num_attention_heads = 64,
        .full_attention_interval = 4,
        .linear_num_value_heads = 64,
        .linear_key_head_dim = 128,
        .mla_kv_lora_rank = 512,
        .mla_qk_nope_head_dim = 256,
        .mla_v_head_dim = 256,
        .indexer_head_dim = 128,
        .max_position_embeddings = 1048576,
    };
    const GiB: u64 = 1 << 30;
    const trunk: u64 = 17_842_600_184;
    const reserve = @max(8 * GiB, try minimumReserve(&cfg, 501, 501));
    const expert: u64 = 3 * 2048 * 4096 * 2;
    const one = try layerMajorBudget(&cfg, 100 * GiB, trunk, reserve, 501, 501, 1);
    const many = try layerMajorBudget(&cfg, 100 * GiB, trunk, reserve, 501, 501, 128);
    // One window carries 501 tokens of four 4096-wide BF16 streams; the cache is one slot per MoE layer.
    try t.expectEqual(@as(u64, 501 * 4 * 4096 * 2), one.carried);
    try t.expectEqual(128 * one.carried, many.carried);
    try t.expectEqual(@as(u64, 42) * expert, many.cache);
    try t.expect(many.fixed >= trunk + reserve + many.carried + 288 * expert + @import("expert_stream.zig").BOUNCE_BYTES);
    try t.expect(many.fixed + many.cache <= many.total);
    try t.expectError(error.GlmLayerMajorBudgetExceeded, layerMajorBudget(&cfg, 40 * GiB, trunk, reserve, 501, 501, 1));
    try t.expectError(error.GlmLayerMajorBudgetExceeded, layerMajorBudget(&cfg, 100 * GiB, trunk, reserve, 501, 501, 10_000));
    try t.expectEqual(@as(u32, 32), try layerMajorWindows(&cfg, 100 * GiB, trunk, reserve, 501, 501, 32));
    const fit = try layerMajorWindows(&cfg, 100 * GiB, trunk, reserve, 501, 501, 1 << 30);
    _ = try layerMajorBudget(&cfg, 100 * GiB, trunk, reserve, 501, 501, fit);
    try t.expectError(error.GlmLayerMajorBudgetExceeded, layerMajorBudget(&cfg, 100 * GiB, trunk, reserve, 501, 501, fit + 1));
    try t.expectError(error.GlmLayerMajorBudgetExceeded, layerMajorWindows(&cfg, 40 * GiB, trunk, reserve, 501, 501, 32));
}

/// A lossless teacher capture's BF16 expert budget: one request of `max_tokens`, chunks of at most 512.
pub fn captureBudget(cfg: *const model.ModelConfig, total: u64, trunk: u64, reserve: u64, max_tokens: usize, max_chunk: usize) !Budget {
    if (!cfg.isGlm5() or max_tokens == 0 or max_tokens > cfg.max_position_embeddings or max_chunk == 0 or max_chunk > max_tokens or max_chunk > 512) return error.InvalidGlmStreamBudget;
    if (reserve < try minimumReserve(cfg, max_tokens, max_chunk)) return error.GlmStreamReserveTooSmall;
    const g = cfg.expertGeometry();
    return plan(total, trunk, reserve, g, try bytesPerExpert(g));
}

/// The caller owns the engine. One inference thread may use a stream at a time;
/// requests own only their caches.
pub const Stream = struct {
    engine: *stream.Engine,
    max_tokens: usize,
    max_chunk: usize,
    request_owner: ?*const anyopaque = null,
    pinned: ?Pinned = null,
    pinned_applies: u64 = 0,
    /// Armed by an imatrix capture: BF16 experts only, every routed row is observed under its global expert id.
    imatrix: ?*imatrix_capture.Collector = null,

    const Pinned = struct { layer: u16, prepared: stream.Prepared };

    /// Makes every expert of `layer` resident until `unpin`, so each later call for that layer reads no SSD
    /// (`prepared.remapped` is indexed by expert id). A layer-major batch reads each layer once this way.
    pub fn pin(self: *Stream, layer: u16) !void {
        self.unpin();
        const ids = try self.engine.allocator.alloc(u16, self.engine.geometry.experts);
        defer self.engine.allocator.free(ids);
        for (ids, 0..) |*id, i| id.* = @intCast(i);
        self.pinned = .{ .layer = layer, .prepared = try self.engine.prepareHost(layer, ids) };
    }
    pub fn unpin(self: *Stream) void {
        if (self.pinned) |*held| held.prepared.deinit();
        self.pinned = null;
    }

    /// Serving admits each request itself; the stream bounds only the model's own context.
    pub fn serving(engine: *stream.Engine, cfg: *const model.ModelConfig) Stream {
        return .{ .engine = engine, .max_tokens = cfg.max_position_embeddings, .max_chunk = cfg.max_position_embeddings };
    }
    pub fn claim(self: *Stream, owner: *const anyopaque) !void {
        if (self.request_owner) |current| if (current != owner) return error.GlmStreamRequestBusy;
        self.request_owner = owner;
    }
    pub fn release(self: *Stream, owner: *const anyopaque) void {
        if (self.request_owner == owner) self.request_owner = null;
    }
    pub fn admit(self: *const Stream, offset: usize, rows: usize) !void {
        if (rows == 0 or rows > self.max_chunk or offset > self.max_tokens or rows > self.max_tokens - offset) return error.GlmStreamRequestBudgetExceeded;
    }
    /// Router ids reach the slabs as slot ids; scores, clamps and the reduction stay the resident path's.
    pub fn apply(self: *Stream, layer: u16, ops: *Ops, x: Arr, ids: Arr, scores: Arr, cfg: *const model.ModelConfig) !Arr {
        if (!mlx.streamIsGpu(ops.s)) return error.GlmStreamRequiresGpu;
        const shape = mlx.getShape(x);
        const ish = mlx.getShape(ids);
        if (shape.len != 3 or shape[0] != 1 or shape[1] < 1 or shape[1] > self.max_chunk or shape[2] != self.engine.geometry.hidden or mlx.mlx_array_dtype(x) != .bfloat16 or
            (mlx.mlx_array_dtype(ids) != .uint32 and mlx.mlx_array_dtype(ids) != .int32) or ish.len != 3 or ish[0] != 1 or ish[1] != shape[1] or ish[2] < 1 or ish[2] > self.engine.geometry.experts or !std.mem.eql(c_int, ish, mlx.getShape(scores)) or mlx.mlx_array_dtype(scores) != .float32 or cfg.glm_swiglu_limit != 10) return error.InvalidGlmStreamInput;
        var scope = Ops{ .s = ops.s };
        defer scope.deinit();
        const raw_ids = try scope.contiguous(try scope.cast(ids, .uint32));
        try mlx.check(mlx.mlx_array_eval(raw_ids));
        const count = mlx.mlx_array_size(raw_ids);
        const data = mlx.mlx_array_data_uint32(raw_ids) orelse return error.InvalidGlmStreamInput;
        const host = try self.engine.allocator.alloc(u16, count);
        defer self.engine.allocator.free(host);
        for (host, 0..) |*v, i| {
            if (data[i] >= self.engine.geometry.experts) return error.ExpertOutOfRange;
            v.* = @intCast(data[i]);
        }
        const held: ?*stream.Prepared = if (self.pinned) |*p| (if (p.layer == layer) &p.prepared else null) else null;
        var fresh: ?stream.Prepared = if (held == null) try self.engine.prepareHost(layer, host) else null;
        defer if (fresh) |*p| p.deinit();
        const prepared = held orelse &fresh.?;
        errdefer _ = mlx.mlx_synchronize(ops.s);
        const local = try self.engine.allocator.alloc(u32, count);
        defer self.engine.allocator.free(local);
        if (held != null) {
            for (local, host) |*v, expert| v.* = prepared.remapped[expert];
            self.pinned_applies += 1;
        } else for (local, prepared.remapped) |*v, remap| v.* = remap;
        const remapped = try scope.own(mlx.mlx_array_new_data(local.ptr, ish.ptr, @intCast(ish.len), .uint32));
        const layout = self.engine.store.layout();
        if (self.imatrix != null and layout != .bf16_individual) return error.GlmImatrixNeedsBf16Experts;
        const tap: ?Tap = if (self.imatrix) |col| .{ .collector = col, .layer = layer, .ids = try scope.reshape(try scope.cast(raw_ids, .int32), &.{ shape[1], ish[2] }) } else null;
        const out = switch (layout) {
            .bf16_individual => try bf16RoutedTapped(&scope, x, prepared.gate, prepared.up, prepared.down, remapped, scores, cfg.glm_swiglu_limit, tap),
            .exl3_k4 => try @import("glm5_forward.zig").routedExl3(&scope, x, exl3Bank(&prepared.quant_raw), remapped, scores, cfg),
            .fp8_individual => try fp8Routed(&scope, self.engine.allocator, x, &prepared.quant_raw, local, ish, scores, cfg.glm_swiglu_limit),
            else => return error.GlmStreamLayoutUnsupported,
        };
        // Complete every slab reader before another layer can refill the union.
        try mlx.check(mlx.mlx_array_eval(out));
        return ops.own(try scope.result(out));
    }
};

fn exl3Bank(raw: *const [stream.quant.component_count]Arr) exl3.Bank {
    const C = stream.quant.Component;
    const proj = struct {
        fn of(r: *const [stream.quant.component_count]Arr, w: C, s: C, b: C) exl3.Proj {
            return .{ .trellis = r[@backingInt(w)], .suh = r[@backingInt(s)], .svh = r[@backingInt(b)] };
        }
    }.of;
    return .{ .gate = proj(raw, .gate_w, .gate_s, .gate_b), .up = proj(raw, .up_w, .up_s, .up_b), .down = proj(raw, .down_w, .down_s, .down_b) };
}

/// FP8 experts by the bf16-dequant route: the routed slots' `[out, in]` e4m3 weights become
/// bf16(code * block scale) (`fp8_block.dequantize`), then the BF16 routed composite. `slots`
/// index `bank`'s leading axis; only the distinct routed slots are dequantized.
pub fn fp8Routed(ops: *Ops, a: std.mem.Allocator, x: Arr, bank: *const [stream.quant.component_count]Arr, slots: []const u32, ids_shape: []const c_int, scores: Arr, limit: f32) !Arr {
    const C = stream.quant.Component;
    const sorted = try a.dupe(u32, slots);
    defer a.free(sorted);
    std.mem.sort(u32, sorted, {}, std.sort.asc(u32));
    var distinct: usize = 0;
    for (sorted) |slot| if (distinct == 0 or sorted[distinct - 1] != slot) {
        sorted[distinct] = slot;
        distinct += 1;
    };
    const used = sorted[0..distinct];
    const position = try a.alloc(u32, used[used.len - 1] + 1);
    defer a.free(position);
    for (used, 0..) |slot, i| position[slot] = @intCast(i);
    const dense = try a.alloc(u32, slots.len);
    defer a.free(dense);
    for (dense, slots) |*d, slot| d.* = position[slot];
    const n: c_int = @intCast(used.len);
    // A union slab holds exactly its routed experts in slots 0..n-1: no gather copy.
    const contiguous = used[used.len - 1] == used.len - 1;
    const pick = try ops.own(mlx.mlx_array_new_data(used.ptr, &.{n}, 1, .uint32));
    var views: [3]Arr = undefined;
    for ([_][2]C{ .{ .gate_w, .gate_s }, .{ .up_w, .up_s }, .{ .down_w, .down_s } }, &views) |pair, *view| {
        var parts: [2]Arr = undefined;
        for (pair, &parts) |c, *part| {
            const all = bank[@backingInt(c)];
            part.* = if (contiguous) try ops.slice(all, 0, 0, n) else try ops.take(all, pick, 0);
        }
        const ws = mlx.getShape(parts[0]);
        const ss = mlx.getShape(parts[1]);
        const rows = n * ws[1];
        var out: [1]Arr = .{.{}};
        try fp8_block.dequantize(ops.s, try ops.reshape(parts[0], &.{ rows, ws[2] }), try ops.reshape(parts[1], &.{ n * ss[1], ss[2] }), fp8_block.RowSplit.dense(@intCast(rows)), &out);
        view.* = try ops.transpose(try ops.reshape(try ops.own(out[0]), &.{ n, ws[1], ws[2] }), &.{ 0, 2, 1 });
    }
    const ids = try ops.own(mlx.mlx_array_new_data(dense.ptr, ids_shape.ptr, @intCast(ids_shape.len), .uint32));
    return bf16Routed(ops, x, views[0], views[1], views[2], ids, scores, limit);
}

/// Banks are [experts,input,output] views. Scores remain FP32 until the final cast.
/// An imatrix capture's view of one routed layer: `ids` is the router's [rows, top_k] GLOBAL expert ids, never slab slots.
const Tap = struct { collector: *imatrix_capture.Collector, layer: u16, ids: Arr };

pub fn bf16Routed(ops: *Ops, x: Arr, gate: Arr, up: Arr, down: Arr, ids: Arr, scores: Arr, limit: f32) !Arr {
    return bf16RoutedTapped(ops, x, gate, up, down, ids, scores, limit, null);
}

fn bf16RoutedTapped(ops: *Ops, x: Arr, gate: Arr, up: Arr, down: Arr, ids: Arr, scores: Arr, limit: f32, tap: ?Tap) !Arr {
    if (!mlx.streamIsGpu(ops.s)) return error.GlmStreamRequiresGpu;
    const xs = mlx.getShape(x);
    const ish = mlx.getShape(ids);
    if (xs.len != 3 or ish.len != 3) return error.InvalidGlmStreamInput;
    const expanded = try ops.reshape(x, &.{ xs[0], xs[1], 1, 1, xs[2] });
    const g = try gather(ops, expanded, gate, ids);
    const u = try gather(ops, expanded, up, ids);
    const hi = try ops.scalar(limit, .bfloat16);
    const lo = try ops.scalar(-limit, .bfloat16);
    const activation = try ops.binary(.mul, try ops.silu(try ops.binary(.min, g, hi)), try ops.binary(.max, try ops.binary(.min, u, hi), lo));
    if (tap) |t| {
        const rows = xs[1] * ish[2];
        try t.collector.observeGateUp(t.layer, try ops.reshape(x, &.{ xs[1], xs[2] }), t.ids);
        const inter = mlx.getShape(activation);
        try t.collector.observeDown(t.layer, try ops.reshape(activation, &.{ rows, inter[inter.len - 1] }), try ops.reshape(t.ids, &.{ rows, 1 }));
    }
    const y = try gather(ops, activation, down, ids);
    const y4 = try ops.reshape(y, &.{ xs[0], xs[1], ish[2], xs[2] });
    const weights = try ops.reshape(scores, &.{ xs[0], xs[1], ish[2], 1 });
    return ops.cast(try ops.reduce(try ops.binary(.mul, y4, weights), -2, false, false), .bfloat16);
}
fn gather(ops: *Ops, x: Arr, w: Arr, ids: Arr) !Arr {
    const out = try ops.slot();
    try mlx.check(mlx.mlx_gather_mm(out, x, w, .{ .ctx = null }, ids, false, ops.s));
    return out.*;
}

test "GLM stream GPU BF16 routed output matches resident through eviction union and retained output" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const fixture = @import("glm_stream_fixture.zig");
    try fixture.write(t.allocator, tmp.dir, .none);
    var path: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(t.io, &path);
    const cpu = mlx.gpuStream();
    const geometry = stream.Geometry{ .layers = 4, .experts = 4, .hidden = 32, .intermediate = 16, .first_moe_layer = 3 };
    var engine = try stream.Engine.initWithOptions(t.allocator, path[0..n], geometry, 2 * 3072, cpu, .{ .layout = .bf16_individual, .bounce_size = 4096, .io_workers = 1 });
    defer engine.deinit();
    var store = Stream{ .engine = &engine, .max_tokens = 16, .max_chunk = 3 };
    const cfg = model.ModelConfig{ .glm_swiglu_limit = 10 };
    var banks: [3]Arr = undefined;
    for (&banks, 0..) |*bank, pi| {
        var raw: [4 * 512]u16 = undefined;
        for (0..4) |e| @memset(raw[e * 512 ..][0..512], fixture.value(e, pi));
        const dims = if (pi == 2) [_]c_int{ 4, 32, 16 } else [_]c_int{ 4, 16, 32 };
        const source = mlx.mlx_array_new_data(&raw, &dims, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(source);
        bank.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_transpose_axes(bank, source, &.{ 0, 2, 1 }, 3, cpu));
    }
    defer for (banks) |b| {
        _ = mlx.mlx_array_free(b);
    };
    const routes = [_][6]u32{ .{ 0, 1, 0, 1, 0, 1 }, .{ 2, 3, 2, 3, 2, 3 }, .{ 3, 2, 1, 0, 3, 0 }, .{ 0, 1, 0, 1, 0, 1 } };
    var old: Arr = .{ .ctx = null };
    defer if (old.ctx != null) {
        _ = mlx.mlx_array_free(old);
    };
    var first: [96]u16 = undefined;
    for (routes, 0..) |route_ids, iteration| {
        var ops = Ops{ .s = cpu };
        defer ops.deinit();
        var values: [96]f32 = undefined;
        for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 5)) / 4;
        const x = try ops.cast(try ops.own(mlx.mlx_array_new_data(&values, &.{ 1, 3, 32 }, 3, .float32)), .bfloat16);
        const ids = try ops.own(mlx.mlx_array_new_data(&route_ids, &.{ 1, 3, 2 }, 3, .uint32));
        const scores = try ops.own(mlx.mlx_array_new_data(&[_]f32{ 0.25, 0.75, 0.4, 0.6, 0.7, 0.3 }, &.{ 1, 3, 2 }, 3, .float32));
        const actual = try store.apply(3, &ops, x, ids, scores, &cfg);
        const expected = try bf16Routed(&ops, x, banks[0], banks[1], banks[2], ids, scores, 10);
        try mlx.check(mlx.mlx_array_eval(expected));
        const bits = mlx.mlx_array_data_bfloat16(actual).?;
        try t.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(expected).?[0..96], bits[0..96]);
        try t.expect(bits[0] != 0);
        if (iteration == 0) {
            old = try ops.result(actual);
            @memcpy(&first, bits[0..96]);
        } else try t.expectEqualSlices(u16, &first, mlx.mlx_array_data_bfloat16(old).?[0..96]);
        try t.expectError(error.ExpertLayerAbsent, store.apply(0, &ops, x, ids, scores, &cfg));
        const bad = try ops.own(mlx.mlx_array_new_data(&[_]u32{ 4, 0, 0, 0, 0, 0 }, &.{ 1, 3, 2 }, 3, .uint32));
        const filled = store.engine.fill_bytes_total;
        try t.expectError(error.ExpertOutOfRange, store.apply(3, &ops, x, bad, scores, &cfg));
        try t.expectEqual(filled, store.engine.fill_bytes_total);
    }
    try t.expect(store.engine.fill_bytes_total > 4 * 3072);
    try t.expectError(error.GlmStreamRequestBudgetExceeded, store.admit(15, 2));
    try t.expectError(error.GlmStreamRequestBudgetExceeded, store.admit(0, 4));
    try store.admit(13, 3);
}

test "GLM stream imatrix tap keys the MLP input and the SwiGLU activation on the router's global expert ids" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const fixture = @import("glm_stream_fixture.zig");
    try fixture.write(t.allocator, tmp.dir, .none);
    var path: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(t.io, &path);
    const geometry = stream.Geometry{ .layers = 4, .experts = 4, .hidden = 32, .intermediate = 16, .first_moe_layer = 3 };
    var engine = try stream.Engine.initWithOptions(t.allocator, path[0..n], geometry, 2 * 3072, s, .{ .layout = .bf16_individual, .bounce_size = 4096, .io_workers = 1 });
    defer engine.deinit();
    const col = try imatrix_capture.Collector.init(t.allocator, s, "/dev/null", 4, 4, .glm5_next);
    defer col.deinit();
    var store = Stream{ .engine = &engine, .max_tokens = 16, .max_chunk = 3, .imatrix = col };
    const cfg = model.ModelConfig{ .glm_swiglu_limit = 10 };
    const route_ids = [6]u32{ 3, 1, 3, 3, 0, 1 };
    var values: [96]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 5)) / 4;
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const x = try ops.cast(try ops.own(mlx.mlx_array_new_data(&values, &.{ 1, 3, 32 }, 3, .float32)), .bfloat16);
    const ids = try ops.own(mlx.mlx_array_new_data(&route_ids, &.{ 1, 3, 2 }, 3, .uint32));
    const scores = try ops.own(mlx.mlx_array_new_data(&[_]f32{ 0.25, 0.75, 0.4, 0.6, 0.7, 0.3 }, &.{ 1, 3, 2 }, 3, .float32));
    _ = try store.apply(3, &ops, x, ids, scores, &cfg);

    // Every fixture weight of one expert and projection is a single constant, so each output channel of gate
    // and up is that constant times the row's sum, and the activation is the same on all 16 channels.
    const weight = struct {
        fn of(e: usize, pi: usize) f32 {
            return @bitCast(@as(u32, fixture.value(e, pi)) << 16);
        }
    }.of;
    var want_gu: [4 * 32]f64 = @splat(0);
    var want_down: [4 * 16]f64 = @splat(0);
    var want_rows: [4]f64 = @splat(0);
    for (0..3) |row| {
        var sum: f64 = 0;
        for (0..32) |c| sum += values[row * 32 + c];
        for (0..2) |slot| {
            const e = route_ids[row * 2 + slot];
            want_rows[e] += 1;
            for (0..32) |c| want_gu[e * 32 + c] += @as(f64, values[row * 32 + c]) * values[row * 32 + c];
            const g = @min(weight(e, 0) * sum, 10);
            const u = std.math.clamp(@min(weight(e, 1) * sum, 10), -10, 10);
            const act = g / (1 + @exp(-g)) * u;
            for (0..16) |c| want_down[e * 16 + c] += act * act;
        }
    }
    const layer = &col.layers[3];
    try mlx.check(mlx.mlx_array_eval(layer.gu));
    try mlx.check(mlx.mlx_array_eval(layer.down));
    try mlx.check(mlx.mlx_array_eval(layer.rows));
    for (want_gu, mlx.mlx_array_data_float32(layer.gu).?[0..want_gu.len]) |want, got| try t.expectApproxEqRel(want, got, 1e-5);
    for (want_rows, mlx.mlx_array_data_float32(layer.rows).?[0..4]) |want, got| try t.expectEqual(want, got);
    for (want_down, mlx.mlx_array_data_float32(layer.down).?[0..want_down.len]) |want, got| try t.expectApproxEqAbs(want, got, 0.03 * @max(1, want));
    try t.expectEqual(@as(u64, 3), layer.tokens);
    try t.expect(col.layers[0].gu.ctx == null);
}

/// The fixture's FP8 bank resident: `[experts, out, in]` codes and `[experts, out/128, in/128]` scales.
fn fp8FixtureBank() [stream.quant.component_count]Arr {
    const fixture = @import("glm_stream_fixture.zig");
    const C = stream.quant.Component;
    var bank: [stream.quant.component_count]Arr = @splat(.{ .ctx = null });
    var codes: [4 * 128 * 128]u8 = undefined;
    var scales: [4]f32 = undefined;
    for ([_][2]C{ .{ .gate_w, .gate_s }, .{ .up_w, .up_s }, .{ .down_w, .down_s } }, 0..) |pair, pi| {
        for (0..4) |e| {
            for (0..128 * 128) |i| codes[e * 128 * 128 + i] = fixture.fp8Code(e, pi, i);
            scales[e] = fixture.fp8Scale(e, pi, 0);
        }
        bank[@backingInt(pair[0])] = mlx.mlx_array_new_data(&codes, &[_]c_int{ 4, 128, 128 }, 3, .uint8);
        bank[@backingInt(pair[1])] = mlx.mlx_array_new_data(&scales, &[_]c_int{ 4, 1, 1 }, 3, .float32);
    }
    return bank;
}

test "GLM stream GPU FP8 routed output matches the resident FP8 bank through eviction and the union" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try @import("glm_stream_fixture.zig").writeStorage(t.allocator, tmp.dir, .none, 128, 128, .fp8);
    var path: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(t.io, &path);
    const geometry = stream.Geometry{ .layers = 4, .experts = 4, .hidden = 128, .intermediate = 128, .first_moe_layer = 3 };
    const per_expert = try stream.expertBytesFor(t.allocator, path[0..n], geometry, .fp8_individual);
    var engine = try stream.Engine.initWithOptions(t.allocator, path[0..n], geometry, 2 * per_expert, s, .{ .layout = .fp8_individual, .bounce_size = 1 << 20, .io_workers = 1 });
    defer engine.deinit();
    var store = Stream{ .engine = &engine, .max_tokens = 16, .max_chunk = 3 };
    const cfg = model.ModelConfig{ .glm_swiglu_limit = 10 };
    var bank = fp8FixtureBank();
    defer for (bank) |b| if (b.ctx != null) {
        _ = mlx.mlx_array_free(b);
    };
    var union_seen = false;
    for ([_][6]u32{ .{ 0, 1, 2, 3, 2, 0 }, .{ 2, 3, 2, 3, 2, 3 }, .{ 3, 2, 1, 0, 3, 0 }, .{ 0, 1, 0, 1, 0, 1 }, .{ 1, 3, 1, 3, 3, 1 } }) |route_ids| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        var values: [3 * 128]f32 = undefined;
        for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 8;
        const x = try ops.cast(try ops.own(mlx.mlx_array_new_data(&values, &.{ 1, 3, 128 }, 3, .float32)), .bfloat16);
        const ids = try ops.own(mlx.mlx_array_new_data(&route_ids, &.{ 1, 3, 2 }, 3, .uint32));
        const scores = try ops.own(mlx.mlx_array_new_data(&[_]f32{ 0.25, 0.75, 0.4, 0.6, 0.7, 0.3 }, &.{ 1, 3, 2 }, 3, .float32));
        const actual = try store.apply(3, &ops, x, ids, scores, &cfg);
        union_seen = union_seen or engine.last.union_members > 0 or engine.layers[3].stats.union_members > 0;
        const expected = try fp8Routed(&ops, t.allocator, x, &bank, &route_ids, &.{ 1, 3, 2 }, scores, 10);
        try mlx.check(mlx.mlx_array_eval(expected));
        const want = mlx.mlx_array_data_bfloat16(expected).?[0..3 * 128];
        try t.expectEqualSlices(u16, want, mlx.mlx_array_data_bfloat16(actual).?[0..3 * 128]);
        var nonzero = false;
        for (want) |bits| nonzero = nonzero or bits & 0x7fff != 0;
        try t.expect(nonzero);
    }
    try t.expect(union_seen);
    try t.expect(engine.fill_experts_total > 4);
}

test "GLM FP8 expert dequant is bf16 of the e4m3 code times its block scale, against an f32 oracle" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(0xf8f8);
    const rnd = prng.random();
    var codes: [2 * 256 * 256]u8 = undefined;
    for (&codes) |*c| {
        c.* = rnd.int(u8);
        if (c.* & 0x7f == 0x7f) c.* ^= 1;
    }
    var scales: [2 * 2 * 2]f32 = undefined;
    for (&scales) |*v| v.* = (0.5 + rnd.float(f32)) * 4.0e-5;
    var bank: [stream.quant.component_count]Arr = @splat(.{ .ctx = null });
    defer for (bank) |b| if (b.ctx != null) {
        _ = mlx.mlx_array_free(b);
    };
    const C = stream.quant.Component;
    for ([_]C{ .gate_w, .up_w, .down_w }) |c| bank[@backingInt(c)] = mlx.mlx_array_new_data(&codes, &[_]c_int{ 2, 256, 256 }, 3, .uint8);
    for ([_]C{ .gate_s, .up_s, .down_s }) |c| bank[@backingInt(c)] = mlx.mlx_array_new_data(&scales, &[_]c_int{ 2, 2, 2 }, 3, .float32);
    var ops = Ops{ .s = s };
    defer ops.deinit();
    var out: [1]Arr = .{.{}};
    try fp8_block.dequantize(s, try ops.reshape(bank[@backingInt(C.gate_w)], &.{ 512, 256 }), try ops.reshape(bank[@backingInt(C.gate_s)], &.{ 4, 2 }), fp8_block.RowSplit.dense(512), &out);
    defer _ = mlx.mlx_array_free(out[0]);
    try mlx.check(mlx.mlx_array_eval(out[0]));
    const got = mlx.mlx_array_data_bfloat16(out[0]).?;
    for (0..2) |e| for (0..256) |r| for (0..256) |col| {
        const code = codes[(e * 256 + r) * 256 + col];
        const sign: f32 = if (code & 0x80 != 0) -1 else 1;
        const exp: i32 = @intCast((code >> 3) & 0xf);
        const man: f32 = @floatFromInt(code & 7);
        const magnitude: f32 = if (exp == 0) man / 8 * std.math.pow(f32, 2, -6) else (1 + man / 8) * std.math.pow(f32, 2, @floatFromInt(exp - 7));
        const scale = scales[(e * 2 + r / 128) * 2 + col / 128];
        const bits: u32 = @bitCast(sign * magnitude * scale);
        const want: u16 = @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
        try t.expectEqual(want, got[(e * 256 + r) * 256 + col]);
    };
}

test "GLM stream CPU only one request holds the admitted cache reserve" {
    var store = Stream{ .engine = undefined, .max_tokens = 16, .max_chunk = 4 };
    var a: u8 = 0;
    var b: u8 = 0;
    try store.claim(&a);
    try store.claim(&a);
    try std.testing.expectError(error.GlmStreamRequestBusy, store.claim(&b));
    store.release(&b);
    try std.testing.expectError(error.GlmStreamRequestBusy, store.claim(&b));
    store.release(&a);
    try store.claim(&b);
    store.release(&b);
    try std.testing.expect(store.request_owner == null);
}

test "GLM stream CPU refuses unsupported BF16 gather before reading arrays" {
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var store = Stream{ .engine = undefined, .max_tokens = 16, .max_chunk = 4 };
    var ops = Ops{ .s = cpu };
    defer ops.deinit();
    const nil = Arr{ .ctx = null };
    try std.testing.expectError(error.GlmStreamRequiresGpu, store.apply(0, &ops, nil, nil, nil, &.{ .glm_swiglu_limit = 10 }));
    try std.testing.expectError(error.GlmStreamRequiresGpu, bf16Routed(&ops, nil, nil, nil, nil, nil, nil, 10));
    try std.testing.expectEqual(@as(usize, 0), ops.count);
}
