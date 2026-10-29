const trf = @import("../transformer.zig");
const kv_cache = @import("../kv_cache.zig");
const std = @import("std");
const qwen4_mod = @import("qwen4_exp.zig");
const expert_stream_mod = @import("../expert_stream.zig");
const mlx = @import("../mlx.zig");
const mrope = @import("../mrope.zig");
const kv_quant = @import("../kv_quant.zig");
const qwen4_forward = @import("qwen4_forward.zig");
const verifyLmHeadProjection = qwen4_forward.verifyLmHeadProjection;
const KVQuantConfig = kv_quant.KVQuantConfig;
const model_mod = @import("../model.zig");
const io_util_mod = @import("../io_util.zig");
const diagEnvOn = io_util_mod.diagEnvOn;
const testing = std.testing;
const qwen4_test_support = @import("qwen4_test_support.zig");
const Qwen4TestSlot = qwen4_test_support.Qwen4TestSlot;
const expectRowsByteEqual = qwen4_test_support.expectRowsByteEqual;
const expectSlotStateEqual = qwen4_test_support.expectSlotStateEqual;
const logicalSsmCapture = qwen4_test_support.logicalSsmCapture;
const qwen4BatchedPleForward = qwen4_test_support.qwen4BatchedPleForward;
const qwen4ExpectMoeBitsEqual = qwen4_test_support.qwen4ExpectMoeBitsEqual;
const rollbackRowToAccepted = qwen4_test_support.rollbackRowToAccepted;
const test_support = @import("../test_support.zig");
const attn256RandBf16 = test_support.attn256RandBf16;
const mlxSettledActiveBytes = test_support.mlxSettledActiveBytes;
const qkvExpectBitsEqual = test_support.qkvExpectBitsEqual;
const qkvIdentityArray = test_support.qkvIdentityArray;
const qsaArraysAllEqual = test_support.qsaArraysAllEqual;
const qwen4ReadF32 = test_support.qwen4ReadF32;
const CaptureLayers = trf.CaptureLayers;
const ForwardCtx = trf.ForwardCtx;
const KVCache = trf.KVCache;
const MoeMlpWeights = trf.MoeMlpWeights;
const QuantParams = trf.QuantParams;
const SSMCacheEntry = trf.SSMCacheEntry;
const SSMCacheEntrySnapshot = trf.SSMCacheEntrySnapshot;
const Transformer = trf.Transformer;
const affineParamsFromGeometry = trf.affineParamsFromGeometry;
const materializedOwnedCopy = trf.materializedOwnedCopy;
const qmatmulBits = trf.qmatmulBits;
const sliceAttentionSeq = trf.sliceAttentionSeq;
const ssmFreeQsaState = trf.ssmFreeQsaState;
const ssmFreeSpecCapture = trf.ssmFreeSpecCapture;
const ssmHasSpecCapture = trf.ssmHasSpecCapture;
const ssmLiveAux = trf.ssmLiveAux;
const ssmRestore = trf.ssmRestore;
const ssmRollbackFromCapture = trf.ssmRollbackFromCapture;
const ssmSnapshot = trf.ssmSnapshot;
const ssmSnapshotDeinit = trf.ssmSnapshotDeinit;
const standinRef = trf.standinRef;
const transposeBf16Weight = trf.transposeBf16Weight;
const verifySharedHardware = trf.verifySharedHardware;

fn qwen4TransposeExpertBank(source: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_transpose_axes(&view, source, &[_]c_int{ 0, 2, 1 }, 3, s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_contiguous(&result, view, false, s));
    try mlx.check(mlx.mlx_array_eval(result));
    return result;
}

fn qwen4StreamedResidentArm(xfm: *Transformer, resident_mw: *const MoeMlpWeights, source_dir: []const u8, cache_slots: u16, batch: c_int, seq: c_int) !void {
    const cfg = &xfm.config;
    const expert_bytes = (@as(u64, 2) * cfg.moe_intermediate_size * cfg.hidden_size + @as(u64, cfg.hidden_size) * cfg.moe_intermediate_size) * 2;
    const cache_bytes = expert_bytes * cache_slots * cfg.num_hidden_layers;
    var engine = try expert_stream_mod.Engine.initWithOptions(testing.allocator, source_dir, .{
        .layers = @intCast(cfg.num_hidden_layers),
        .experts = @intCast(cfg.num_experts),
        .hidden = cfg.hidden_size,
        .intermediate = cfg.moe_intermediate_size,
    }, cache_bytes, xfm.s, .{ .bounce_size = 1024 * 1024 });
    defer engine.deinit();
    const count: usize = @intCast(batch * seq * @as(c_int, @intCast(cfg.hidden_size)));
    const data = try testing.allocator.alloc(u16, count);
    defer testing.allocator.free(data);
    for (data, 0..) |*value, i| value.* = @intCast(0x3c00 + i % 257);
    const x = mlx.mlx_array_new_data(data.ptr, &[_]c_int{ batch, seq, @intCast(cfg.hidden_size) }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const resident = try xfm.moeMLP2(x, x, resident_mw);
    defer _ = mlx.mlx_array_free(resident);
    xfm.expert_stream = &engine;
    defer xfm.expert_stream = null;
    var ctx = xfm.defaultCtx();
    const streamed = try xfm.moeMLP2WithRouter(x, x, resident_mw, null, false, .{ .ctx = &ctx, .layer = 0 }, null);
    defer _ = mlx.mlx_array_free(streamed);
    try qwen4ExpectMoeBitsEqual(resident, streamed);
    if (cache_slots == cfg.num_experts) {
        engine.beginForward();
        const cached = try xfm.moeMLP2WithRouter(x, x, resident_mw, null, false, .{ .ctx = &ctx, .layer = 0 }, null);
        defer _ = mlx.mlx_array_free(cached);
        try qwen4ExpectMoeBitsEqual(resident, cached);
        try testing.expectEqual(@as(u64, cfg.num_experts), engine.layerHits(0));
    } else {
        try testing.expect(engine.layerUnionMembers(0) > cache_slots);
    }
}

/// Packed `[E, 2M, X]` gate/up expert tensor (HF layout, gate rows first) → contiguous halves.
fn splitPackedGateUp(arr: mlx.mlx_array, s: mlx.mlx_stream) !struct { gate: mlx.mlx_array, up: mlx.mlx_array } {
    const shape = mlx.getShape(arr);
    if (shape.len != 3) return error.BadPackedGateUpShape;
    const two_m = shape[1];
    const m = @divExact(two_m, 2);
    const strides = [_]c_int{ 1, 1, 1 };

    var gate_view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_view);
    const g_start = [_]c_int{ 0, 0, 0 };
    const g_stop = [_]c_int{ shape[0], m, shape[2] };
    try mlx.check(mlx.mlx_slice(&gate_view, arr, &g_start, 3, &g_stop, 3, &strides, 3, s));

    var up_view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up_view);
    const u_start = [_]c_int{ 0, m, 0 };
    const u_stop = [_]c_int{ shape[0], two_m, shape[2] };
    try mlx.check(mlx.mlx_slice(&up_view, arr, &u_start, 3, &u_stop, 3, &strides, 3, s));

    // gather_qmm reads the expert weight buffers as ROW-CONTIGUOUS: a lazy slice view
    // has parent strides and silently produces zeros.
    var gate = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_contiguous(&gate, gate_view, false, s));
    var up = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(up);
    try mlx.check(mlx.mlx_contiguous(&up, up_view, false, s));

    return .{ .gate = gate, .up = up };
}

fn qwen4StreamedResidentFixtureParity(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8, source_dir: []const u8) !void {
    var config = try model_mod.parseConfig(io, allocator, model_dir);
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, model_dir);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    xfm.expert_bf16_kernels = false;
    var source = try model_mod.loadWeights(io, allocator, source_dir);
    defer source.deinit();
    const packed_gate_up = source.get("model.language_model.layers.0.mlp.experts.gate_up_proj") orelse return error.MissingFixtureTensor;
    const down_source = source.get("model.language_model.layers.0.mlp.experts.down_proj") orelse return error.MissingFixtureTensor;
    const pair = try splitPackedGateUp(packed_gate_up, xfm.s);
    defer _ = mlx.mlx_array_free(pair.gate);
    defer _ = mlx.mlx_array_free(pair.up);
    const gate = try qwen4TransposeExpertBank(pair.gate, xfm.s);
    defer _ = mlx.mlx_array_free(gate);
    const up = try qwen4TransposeExpertBank(pair.up, xfm.s);
    defer _ = mlx.mlx_array_free(up);
    const down = try qwen4TransposeExpertBank(down_source, xfm.s);
    defer _ = mlx.mlx_array_free(down);
    var resident_mw = xfm.moe_layers.?[0].mlp.moe;
    resident_mw.switch_gate_w = gate;
    resident_mw.switch_gate_s = mlx.mlx_array_new();
    resident_mw.switch_gate_b = mlx.mlx_array_new();
    resident_mw.switch_up_w = up;
    resident_mw.switch_up_s = mlx.mlx_array_new();
    resident_mw.switch_up_b = mlx.mlx_array_new();
    resident_mw.switch_down_w = down;
    resident_mw.switch_down_s = mlx.mlx_array_new();
    resident_mw.switch_down_b = mlx.mlx_array_new();
    const experts: u16 = @intCast(config.num_experts);
    try qwen4StreamedResidentArm(&xfm, &resident_mw, source_dir, experts, 1, 1);
    try qwen4StreamedResidentArm(&xfm, &resident_mw, source_dir, experts, 1, 64);
    try qwen4StreamedResidentArm(&xfm, &resident_mw, source_dir, experts, 2, 4);
    try qwen4StreamedResidentArm(&xfm, &resident_mw, source_dir, @max(experts / 2, 1), 1, 1);
    try qwen4StreamedResidentArm(&xfm, &resident_mw, source_dir, @max(experts / 2, 1), 1, 64);
}

const Qwen4RowCmp = struct { min_cos: f64, argmax_agree: usize, rows: usize };

/// Per-row cosine + argmax agreement between `a` and `b`, both `[rows][v]`.
/// The tiny random oracle's near ties, acquitted by the REFERENCE's own
/// numbers, never ours: a row whose QSA block selection (`qsa_gap`, the
/// relative margin between the last selected and first rejected block score,
/// min over the QSA layers — relu leaves exact zeros) sits under 3% flips
/// under 1% kernel noise and its whole row differs; a row whose top-2 logit
/// margin (`logit_margin`) is under 3% of the row's RMS logit has a coin-flip
/// argmax. Rows without a tensor are never acquitted.
const QWEN4_TIE_REL: f32 = 0.03;
/// Expert-selection acquittal: the router sees a bf16 residual (noisier than
/// the f32 QSA scores) — a k=2 tiny fixture flipped a row at a 3.5% margin.
const QWEN4_TIE_REL_ROUTE: f32 = 0.05;

const Qwen4Ties = struct {
    qsa_gap: ?[]const f32,
    /// Min over MoE layers of the relative softmax margin at the top-k
    /// boundary (k < E fixtures; absent or 1.0 when every expert runs).
    route_gap: ?[]const f32,
    logit_margin: ?[]const f32,
    fn read(alloc: std.mem.Allocator, fx: *const model_mod.Weights, s: mlx.mlx_stream) !Qwen4Ties {
        return .{
            .qsa_gap = if (fx.get("qsa_gap")) |g| try qwen4ReadF32(alloc, g, s) else null,
            .route_gap = if (fx.get("route_gap")) |g| try qwen4ReadF32(alloc, g, s) else null,
            .logit_margin = if (fx.get("logit_margin")) |g| try qwen4ReadF32(alloc, g, s) else null,
        };
    }
    pub fn deinit(self: Qwen4Ties, alloc: std.mem.Allocator) void {
        if (self.qsa_gap) |g| alloc.free(g);
        if (self.route_gap) |g| alloc.free(g);
        if (self.logit_margin) |g| alloc.free(g);
    }
    /// Row-level acquittal: a QSA block selection OR an expert selection
    /// near-tie in the reference.
    fn rowTied(self: Qwen4Ties, row: usize) bool {
        if (self.qsa_gap) |g| if (g[row] < QWEN4_TIE_REL) return true;
        if (self.route_gap) |g| if (g[row] < QWEN4_TIE_REL_ROUTE) return true;
        return false;
    }
    fn margin(self: Qwen4Ties, row: usize) f32 {
        var m: f32 = 1.0;
        if (self.qsa_gap) |g| m = @min(m, g[row]);
        if (self.route_gap) |g| m = @min(m, g[row]);
        return m;
    }
    /// Argmax acquittal: row-tied OR a near-tie top-2 logit.
    fn argmaxTied(self: Qwen4Ties, row: usize, ref_row: []const f32) bool {
        if (self.rowTied(row)) return true;
        const mg = self.logit_margin orelse return false;
        var ss: f64 = 0;
        for (ref_row) |x| ss += @as(f64, x) * x;
        const rms: f32 = @floatCast(@sqrt(ss / @as(f64, @floatFromInt(ref_row.len))));
        return mg[row] < QWEN4_TIE_REL * rms;
    }
};

const Qwen4TiedCmp = struct { min_cos: f64, agree: usize, decisive: usize, tied: usize };

/// Row-wise compare over `rows` starting at fixture row `row0`: min cos over
/// non-tied rows, argmax agreement over decisive rows.
fn qwen4CompareTied(tag: []const u8, a: []const f32, b: []const f32, rows: usize, v: usize, row0: usize, ties: Qwen4Ties) Qwen4TiedCmp {
    var out: Qwen4TiedCmp = .{ .min_cos = 1.0, .agree = 0, .decisive = 0, .tied = 0 };
    for (0..rows) |r| {
        const ref_row = b[r * v ..][0..v];
        const rc = qwen4CompareRows(a[r * v ..][0..v], ref_row, 1, v);
        if (ties.rowTied(row0 + r)) {
            out.tied += 1;
            continue;
        }
        if (rc.min_cos < out.min_cos) out.min_cos = rc.min_cos;
        if (diagEnvOn("QWEN4_FIXTURE_ROWS")) std.debug.print("    {s} row {d}: cos {d:.4} margin {d:.4}\n", .{ tag, row0 + r, rc.min_cos, ties.margin(row0 + r) });
        if (ties.argmaxTied(row0 + r, ref_row)) continue;
        out.decisive += 1;
        out.agree += rc.argmax_agree;
    }
    return out;
}

fn qwen4CompareRows(a: []const f32, b: []const f32, rows: usize, v: usize) Qwen4RowCmp {
    var r: Qwen4RowCmp = .{ .min_cos = 1.0, .argmax_agree = 0, .rows = rows };
    for (0..rows) |i| {
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        var ia: usize = 0;
        var ib: usize = 0;
        for (0..v) |j| {
            const x: f64 = a[i * v + j];
            const y: f64 = b[i * v + j];
            dot += x * y;
            na += x * x;
            nb += y * y;
            if (x > a[i * v + ia]) ia = j;
            if (y > b[i * v + ib]) ib = j;
        }
        const cos = dot / (@sqrt(na) * @sqrt(nb) + 1e-30);
        if (cos < r.min_cos) r.min_cos = cos;
        if (ia == ib) r.argmax_agree += 1;
    }
    return r;
}

fn qwen4Rows(arr: mlx.mlx_array) c_int {
    return if (arr.ctx != null) mlx.getShape(arr)[1] else -1;
}

fn qwen4BitsEqual(alloc: std.mem.Allocator, a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    if (a.ctx == null or b.ctx == null) return a.ctx == b.ctx;
    const ah = try qwen4ReadF32(alloc, a, s);
    defer alloc.free(ah);
    const bh = try qwen4ReadF32(alloc, b, s);
    defer alloc.free(bh);
    return std.mem.eql(u8, std.mem.sliceAsBytes(ah), std.mem.sliceAsBytes(bh));
}

/// Value bar for a deferred-PLE forward: everything it publishes must hold the
/// same bytes as the eager arm, not the graph as it stood while the PLE leaf
/// was still zero. Reports every arm before the test fails, so one run names
/// which tensors a regression poisoned.
fn captureMismatch(a: []const f32, b: []const f32, width: usize, label: []const u8) !bool {
    try testing.expectEqual(a.len, b.len);
    var differ: usize = 0;
    var max_abs: f32 = 0;
    for (a, b) |x, y| {
        if (x == y) continue;
        differ += 1;
        const d = @abs(x - y);
        if (d > max_abs) max_abs = d;
    }
    if (differ == 0) return false;
    std.debug.print("[S={d} {s}] {d}/{d} values differ, max abs diff {d}\n", .{ width, label, differ, a.len, max_abs });
    return true;
}

fn qwen4ReplayPrompt(xfm: *Transformer, slot: *Qwen4TestSlot, prompt: []const i32) !void {
    const prefill = try slot.forward(xfm, prompt[0 .. prompt.len - 1]);
    defer _ = mlx.mlx_array_free(prefill);
    try mlx.check(mlx.mlx_array_eval(prefill));
    const last = try slot.forward(xfm, prompt[prompt.len - 1 ..]);
    defer _ = mlx.mlx_array_free(last);
    try mlx.check(mlx.mlx_array_eval(last));
}

fn qwen4Top2(values: []const f32) !struct { id: usize, gap: f32 } {
    for (values) |v| try testing.expect(std.math.isFinite(v));
    var best: usize = 0;
    var second: f32 = -std.math.inf(f32);
    for (values[1..], 1..) |v, i| {
        if (v > values[best]) {
            second = values[best];
            best = i;
        } else if (v > second) second = v;
    }
    return .{ .id = best, .gap = values[best] - second };
}

const L10VerifySlots = struct {
    solo: [2]*Qwen4TestSlot,
    group: [2]*Qwen4TestSlot,

    pub fn init(alloc: std.mem.Allocator, xfm: *Transformer, n_layers: u32, prompts: [2][]const i32) !L10VerifySlots {
        var out: L10VerifySlots = undefined;
        for (0..2) |i| {
            out.solo[i] = try Qwen4TestSlot.init(alloc, n_layers);
            out.group[i] = try Qwen4TestSlot.init(alloc, n_layers);
            _ = mlx.mlx_array_free(try out.solo[i].forward(xfm, prompts[i]));
            _ = mlx.mlx_array_free(try out.group[i].forward(xfm, prompts[i]));
        }
        return out;
    }
    pub fn deinit(self: *L10VerifySlots, alloc: std.mem.Allocator) void {
        for (0..2) |i| {
            self.solo[i].deinit(alloc);
            self.group[i].deinit(alloc);
        }
    }
};

const SingleVerifyTestResult = struct {
    logits: mlx.mlx_array,
    last: mlx.mlx_array,
    all: mlx.mlx_array,

    pub fn deinit(self: *@This()) void {
        _ = mlx.mlx_array_free(self.logits);
        _ = mlx.mlx_array_free(self.last);
        _ = mlx.mlx_array_free(self.all);
    }
};

fn singleVerifyTestRound(xfm: *Transformer, slot: *Qwen4TestSlot, row: mlx.mlx_array) !SingleVerifyTestResult {
    const Gen = @import("../generate.zig").Generator;
    var gen: Gen = undefined;
    gen.xfm = xfm;
    gen.ctx = slot.ctx;
    gen.sampling = .{};
    defer slot.ctx = gen.ctx;
    var state: Gen.MtpRoundState = undefined;
    state.tracing = false;
    state.verify_input = row;
    try gen.mtpRoundVerify(&state);
    var result: SingleVerifyTestResult = .{ .logits = state.verify_logits, .last = state.new_hidden, .all = state.verify_hidden_all };
    errdefer result.deinit();
    const values = mlx.mlx_vector_array_new_data(&.{ result.logits, result.last, result.all }, 3);
    defer _ = mlx.mlx_vector_array_free(values);
    try mlx.check(mlx.mlx_eval(values));
    return result;
}

fn qwen4RowDiff(s: mlx.mlx_stream, serial: mlx.mlx_array, verify: mlx.mlx_array, width: c_int, row_index: c_int, label: []const u8) !bool {
    if (serial.ctx == null or verify.ctx == null) {
        try testing.expectEqual(serial.ctx == null, verify.ctx == null);
        return false;
    }
    const shape = mlx.getShape(verify);
    var begin: [5]c_int = @splat(0);
    var end: [5]c_int = @splat(1);
    const step: [5]c_int = @splat(1);
    @memcpy(end[0..shape.len], shape);
    if (shape.len >= 3 and shape[0] == 1 and shape[1] == width) {
        begin[1] = row_index;
        end[1] = row_index + 1;
    } else if (shape.len == 2 and shape[0] == width) {
        begin[0] = row_index;
        end[0] = row_index + 1;
    } else return error.BadTraceShape;
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    try mlx.check(mlx.mlx_slice(&row, verify, &begin, shape.len, &end, shape.len, &step, shape.len, s));
    var ac = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ac);
    var bc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bc);
    try mlx.check(mlx.mlx_contiguous(&ac, serial, false, s));
    try mlx.check(mlx.mlx_contiguous(&bc, row, false, s));
    const ah = try qwen4ReadF32(testing.allocator, ac, s);
    defer testing.allocator.free(ah);
    const bh = try qwen4ReadF32(testing.allocator, bc, s);
    defer testing.allocator.free(bh);
    try testing.expectEqual(ah.len, bh.len);
    var differences: usize = 0;
    var maximum: f32 = 0;
    var first: usize = 0;
    for (ah, bh, 0..) |x, y, i| {
        try testing.expect(std.math.isFinite(x) and std.math.isFinite(y));
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) {
            if (differences == 0) first = i;
            differences += 1;
            maximum = @max(maximum, @abs(x - y));
        }
    }
    if (differences == 0) return false;
    std.debug.print("{s}: {d}/{d} differ; first {d}: {d} vs {d}; max {d}\n", .{ label, differences, ah.len, first, ah[first], bh[first], maximum });
    return true;
}

const DenseVerifyReplay = struct {
    logits: mlx.mlx_array,
    last: mlx.mlx_array,
    all: mlx.mlx_array,
    ns: u64,

    pub fn deinit(self: DenseVerifyReplay) void {
        _ = mlx.mlx_array_free(self.logits);
        _ = mlx.mlx_array_free(self.last);
        _ = mlx.mlx_array_free(self.all);
    }
};

fn denseVerifyReplay(xfm: *Transformer, slot: *Qwen4TestSlot, tokens: []const i32, io: std.Io) !DenseVerifyReplay {
    const ids = mlx.mlx_array_new_data(tokens.ptr, &.{ 1, @intCast(tokens.len) }, 2, .int32);
    defer _ = mlx.mlx_array_free(ids);
    var last = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(last);
    var all = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(all);
    slot.ctx.capture_ssm_seq = true;
    slot.ctx.ple_defer = true;
    var watch = @import("../io_util.zig").Stopwatch.init(io);
    const logits = try xfm.forwardWithCaptureAll(&slot.ctx, ids, &last, &all);
    errdefer _ = mlx.mlx_array_free(logits);
    slot.ctx.capture_ssm_seq = false;
    slot.ctx.ple_defer = false;
    try xfm.flushDeferredPle(&slot.ctx);
    const evals = mlx.mlx_vector_array_new_data(&.{ logits, last, all }, 3);
    defer _ = mlx.mlx_vector_array_free(evals);
    for (slot.entries) |entry| {
        inline for (.{ "conv_state", "ssm_state", "spec_state_seq", "spec_conv_input", "spec_ple_input", "aux_state", "qsa_pooled" }) |field| {
            const arr = @field(entry, field);
            if (arr.ctx != null) try mlx.check(mlx.mlx_vector_array_append_value(evals, arr));
        }
    }
    try mlx.check(mlx.mlx_eval(evals));
    return .{ .logits = logits, .last = last, .all = all, .ns = watch.read() };
}

fn qwen4LayerIdentityCase(capture_layers: bool, capture_hidden: bool, width: c_int, prefix: usize, row_index: c_int) !void {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();
    var config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |path| a.free(path);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    const serial = try Qwen4TestSlot.init(a, config.num_hidden_layers);
    defer serial.deinit(a);
    const verify = try Qwen4TestSlot.init(a, config.num_hidden_layers);
    defer verify.deinit(a);
    const prompt = [_]i32{ 248045, 846, 198, 7734, 264, 2716, 3255, 883, 264, 11952, 6618, 310, 5984, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };
    const history = [_]i32{ 4413, 220, 22, 18, 19, 11, 852, 14459, 310, 381, 2512, 328, 57517, 1288, 1467, 524, 3418, 3069, 279, 12313, 36811, 948, 781, 6749, 506, 279, 41564, 13, 1946, 28970, 11, 264 };
    for ([_]*Qwen4TestSlot{ serial, verify }) |slot| {
        try slot.cache.reinit(config.num_hidden_layers, KVQuantConfig.affine(8));
        try qwen4ReplayPrompt(&xfm, slot, &prompt);
        var begin: usize = 0;
        while (begin < prefix) {
            const stop = @min(prefix, begin + if (slot == serial) @as(usize, 1) else @as(usize, @intCast(width)));
            const ids = mlx.mlx_array_new_data(history[begin..stop].ptr, &.{ 1, @intCast(stop - begin) }, 2, .int32);
            defer _ = mlx.mlx_array_free(ids);
            var hidden = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(hidden);
            var all = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(all);
            slot.ctx.ple_defer = slot == verify;
            slot.ctx.capture_ssm_seq = slot == verify;
            const out = if (slot == serial) try slot.forward(&xfm, history[begin..stop]) else try xfm.forwardWithCaptureAll(&slot.ctx, ids, &hidden, &all);
            defer _ = mlx.mlx_array_free(out);
            try xfm.flushDeferredPle(&slot.ctx);
            try mlx.check(mlx.mlx_array_eval(out));
            for (slot.entries) |*entry| ssmFreeSpecCapture(entry);
            begin = stop;
        }
    }
    const layer_ids = try a.alloc(u32, config.num_hidden_layers);
    defer a.free(layer_ids);
    const sa = try a.alloc(mlx.mlx_array, layer_ids.len);
    defer a.free(sa);
    const va = try a.alloc(mlx.mlx_array, layer_ids.len);
    defer a.free(va);
    for (layer_ids, sa, va, 0..) |*id, *x, *y, i| {
        id.* = @intCast(i);
        x.* = mlx.mlx_array_new();
        y.* = mlx.mlx_array_new();
    }
    defer for (sa, va) |x, y| {
        _ = mlx.mlx_array_free(x);
        _ = mlx.mlx_array_free(y);
    };
    var sc = CaptureLayers{ .ids = layer_ids, .out = sa };
    var vc = CaptureLayers{ .ids = layer_ids, .out = va };
    if (capture_layers) {
        serial.ctx.capture_layers = &sc;
        verify.ctx.capture_layers = &vc;
    }
    var st = Transformer.Qwen4Trace{};
    defer st.deinit();
    var vt = Transformer.Qwen4Trace{};
    defer vt.deinit();
    for (history[prefix..][0..@intCast(row_index)]) |token| {
        const out = try serial.forward(&xfm, &.{token});
        defer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));
    }
    Transformer.qwen4_trace = &st;
    defer Transformer.qwen4_trace = null;
    const sr = try serial.forward(&xfm, &.{history[prefix + @as(usize, @intCast(row_index))]});
    defer _ = mlx.mlx_array_free(sr);
    try mlx.check(mlx.mlx_array_eval(sr));
    Transformer.qwen4_trace = &vt;
    verify.ctx.capture_ssm_seq = true;
    verify.ctx.ple_defer = capture_hidden;
    var hidden = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden);
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    const tokens = history[prefix..][0..@intCast(width)];
    const ids = mlx.mlx_array_new_data(tokens.ptr, &[_]c_int{ 1, width }, 2, .int32);
    defer _ = mlx.mlx_array_free(ids);
    const vr = if (capture_hidden) try xfm.forwardWithCaptureAll(&verify.ctx, ids, &hidden, &all) else try verify.forward(&xfm, tokens);
    defer _ = mlx.mlx_array_free(vr);
    try xfm.flushDeferredPle(&verify.ctx);
    try mlx.check(mlx.mlx_array_eval(vr));
    var mismatch = false;
    inline for (.{ "mixed_attn", "inj_attn", "attn_out", "mixed_mlp", "inj_mlp", "mlp_out", "ple_emb", "ple_out", "attn3_input", "q3_raw", "k3_raw", "v3_raw", "gate3", "attn3_out" }) |name| {
        if (try qwen4RowDiff(s, @field(st, name), @field(vt, name), width, row_index, name)) mismatch = true;
    }
    inline for (.{ "q3_rope", "k3_rope", "v3_t", "kv3_k", "kv3_v", "attn3_pre_tail" }) |name| {
        const ref = @field(st, name);
        const begin = if (comptime std.mem.startsWith(u8, name, "kv")) 0 else row_index;
        const row = try sliceAttentionSeq(s, @field(vt, name), begin, begin + mlx.getShape(ref)[2]);
        defer _ = mlx.mlx_array_free(row);
        qkvExpectBitsEqual(s, ref, row) catch {
            std.debug.print("attention stage differs: {s}\n", .{name});
            mismatch = true;
        };
    }
    if (capture_layers) for (sa, va, 0..) |x, y, i| {
        var buf: [48]u8 = undefined;
        const label = try std.fmt.bufPrint(&buf, "layer {d}", .{i});
        if (try qwen4RowDiff(s, x, y, width, row_index, label)) mismatch = true;
    };
    if (!capture_layers) for (xfm.moe_layers.?, serial.entries, verify.entries, 0..) |layer, se, *ve, i| {
        if (layer.attn != .linear) continue;
        const sequence = try logicalSsmCapture(s, ve);
        defer _ = mlx.mlx_array_free(sequence);
        const zero = mlx.mlx_array_new_int(row_index);
        defer _ = mlx.mlx_array_free(zero);
        var first = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(first);
        try mlx.check(mlx.mlx_take_axis(&first, sequence, zero, 0, s));
        qkvExpectBitsEqual(s, se.ssm_state, first) catch {
            std.debug.print("recurrent state differs: layer={d} row={d}\n", .{ i, row_index });
            mismatch = true;
        };
    };
    for (serial.cache.entries, verify.cache.entries, 0..) |se, ve, layer| {
        if (!se.initialized) continue;
        inline for (.{ "key_view", "value_view", "key_scales_view", "key_biases_view", "value_scales_view", "value_biases_view" }) |field| {
            const ref = @field(se, field);
            if (ref.ctx != null) {
                const row = try sliceAttentionSeq(s, @field(ve, field), 0, mlx.getShape(ref)[2]);
                defer _ = mlx.mlx_array_free(row);
                if (!try qsaArraysAllEqual(ref, row, s)) {
                    std.debug.print("KV prefix differs: layer={d} field={s}\n", .{ layer, field });
                    mismatch = true;
                }
            }
        }
    }
    if (try qwen4RowDiff(s, sr, vr, width, row_index, "logits")) mismatch = true;
    if (mismatch) return error.VerifyLayerRowDivergence;
}

test "qwen4 MTP head: a load that fails at any allocation frees what it took (QWEN4_TEST_MODEL)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, t.allocator, std.mem.span(model_dir));
    defer config.deinit(t.allocator);
    var weights = try model_mod.loadWeights(io, t.allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var name_buf: [256]u8 = undefined;
    var fail_index: usize = 0;
    while (true) : (fail_index += 1) {
        var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = fail_index });
        const head = Transformer.loadQwen4Mtp(failing.allocator(), config, &weights, &name_buf, s) catch continue;
        var h = head orelse return error.SkipZigTest;
        h.deinit(failing.allocator());
        break;
    }
    try t.expect(fail_index > 0);
}

test "qwen4 MTP head: N=1 FORCE_BATCHED draft ids and logits mlx_equal solo at depths 1/2/4" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const t = std.testing;
    const allocator = t.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    qwen4_forward.mtp_head_force_batched_override = true;
    defer qwen4_forward.mtp_head_force_batched_override = null;

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try t.expect(xfm.qwen4_mtp != null);
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();

    var host_ids: [16]i32 = undefined;
    for (&host_ids, 0..) |*v, i| v.* = @intCast(1 + i);
    const slot = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
    defer slot.deinit(allocator);
    var stream = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stream);
    slot.ctx.capture_stream_all = &stream;
    _ = mlx.mlx_array_free(try slot.forward(&xfm, host_ids[0..8]));

    const H = struct {
        fn row(st: mlx.mlx_stream, arr: mlx.mlx_array, at: usize) !mlx.mlx_array {
            const sh = mlx.getShape(arr);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&out, arr, &[_]c_int{ 0, @intCast(at), 0 }, 3, &[_]c_int{ sh[0], @intCast(at + 1), sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, st));
            return out;
        }
        fn tok(v: i32) mlx.mlx_array {
            const sh = [_]c_int{ 1, 1 };
            var x: i32 = v;
            return mlx.mlx_array_new_data(@ptrCast(&x), &sh, 2, .int32);
        }
        fn argmax(alloc: std.mem.Allocator, logits: mlx.mlx_array, st: mlx.mlx_stream) !i32 {
            const f = try qwen4ReadF32(alloc, logits, st);
            defer alloc.free(f);
            var best: f32 = -std.math.inf(f32);
            var id: i32 = 0;
            for (f, 0..) |x, i| if (x > best) {
                best = x;
                id = @intCast(i);
            };
            return id;
        }
    };
    const h0 = try H.row(s, stream, 0);
    defer _ = mlx.mlx_array_free(h0);
    var tok_s: i32 = host_ids[1];
    var tok_b: i32 = host_ids[1];

    const depths = [_]u32{ 1, 2, 4 };
    for (depths) |depth| {
        var solo = try xfm.qwen4MtpStateNew();
        defer solo.deinit();
        var bat = try xfm.qwen4MtpStateNew();
        defer bat.deinit();
        var h_s = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&h_s, h0));
        defer _ = mlx.mlx_array_free(h_s);
        var h_b = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&h_b, h0));
        defer _ = mlx.mlx_array_free(h_b);
        tok_s = host_ids[1];
        tok_b = host_ids[1];
        var d: u32 = 0;
        while (d < depth) : (d += 1) {
            const po_s: c_int = if (solo.seq_offset == 0) 1 else solo.pos_base + @as(c_int, @intCast(solo.seq_offset));
            const po_b: c_int = if (bat.seq_offset == 0) 1 else bat.pos_base + @as(c_int, @intCast(bat.seq_offset));
            const id_s = H.tok(tok_s);
            defer _ = mlx.mlx_array_free(id_s);
            const id_b = H.tok(tok_b);
            defer _ = mlx.mlx_array_free(id_b);
            const os = try xfm.qwen4MtpForwardOn(&solo, h_s, id_s, po_s, null, .last_row);
            defer _ = mlx.mlx_array_free(os.logits);
            defer _ = mlx.mlx_array_free(os.stream);
            qwen4_forward.qwen4_mtp_head_graphs = 0;
            const brows = [_]Transformer.Qwen4MtpBatchRow{
                .{ .st = &bat, .stream = h_b, .token_ids = id_b, .pos_offset = po_b },
            };
            const bout = try xfm.qwen4MtpForwardBatched(&brows, .last_row, depth);
            defer {
                for (bout) |o| {
                    if (o.logits.ctx != null) _ = mlx.mlx_array_free(o.logits);
                    if (o.stream.ctx != null) _ = mlx.mlx_array_free(o.stream);
                    if (o.mixed.ctx != null) _ = mlx.mlx_array_free(o.mixed);
                }
                allocator.free(bout);
            }
            try t.expectEqual(@as(usize, 1), qwen4_forward.qwen4_mtp_head_graphs);
            try t.expect(try qsaArraysAllEqual(os.logits, bout[0].logits, s));
            const id_solo = try H.argmax(allocator, os.logits, s);
            const id_bat = try H.argmax(allocator, bout[0].logits, s);
            try t.expectEqual(id_solo, id_bat);
            tok_s = id_solo;
            tok_b = id_bat;
            _ = mlx.mlx_array_free(h_s);
            _ = mlx.mlx_array_free(h_b);
            h_s = mlx.mlx_array_new();
            h_b = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&h_s, os.stream));
            try mlx.check(mlx.mlx_array_set(&h_b, bout[0].stream));
        }
    }
}

test "MTP head: N=2/4 mlx_equal drafts/logits vs solo" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const t = std.testing;
    const allocator = t.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try t.expect(xfm.qwen4_mtp != null);
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();

    var host_ids: [32]i32 = undefined;
    for (&host_ids, 0..) |*v, i| v.* = @intCast(1 + i);
    const slot_a = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
    defer slot_a.deinit(allocator);
    const slot_b = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
    defer slot_b.deinit(allocator);
    var stream_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stream_a);
    var stream_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stream_b);
    slot_a.ctx.capture_stream_all = &stream_a;
    slot_b.ctx.capture_stream_all = &stream_b;
    _ = mlx.mlx_array_free(try slot_a.forward(&xfm, host_ids[0..8]));
    _ = mlx.mlx_array_free(try slot_b.forward(&xfm, host_ids[3..8]));

    const H = struct {
        fn row(st: mlx.mlx_stream, arr: mlx.mlx_array, at: usize) !mlx.mlx_array {
            return span(st, arr, at, 1);
        }
        fn span(st: mlx.mlx_stream, arr: mlx.mlx_array, at: usize, count: usize) !mlx.mlx_array {
            const sh = mlx.getShape(arr);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&out, arr, &[_]c_int{ 0, @intCast(at), 0 }, 3, &[_]c_int{ sh[0], @intCast(at + count), sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, st));
            return out;
        }
        fn tok(v: i32) mlx.mlx_array {
            const sh = [_]c_int{ 1, 1 };
            var x: i32 = v;
            return mlx.mlx_array_new_data(@ptrCast(&x), &sh, 2, .int32);
        }
        fn top2(alloc: std.mem.Allocator, logits: mlx.mlx_array, st: mlx.mlx_stream) !struct { id: i32 } {
            const f = try qwen4ReadF32(alloc, logits, st);
            defer alloc.free(f);
            var b0: f32 = -std.math.inf(f32);
            var id: i32 = 0;
            for (f, 0..) |x, i| {
                if (x > b0) {
                    b0 = x;
                    id = @intCast(i);
                }
            }
            return .{ .id = id };
        }
        fn pos(st: *Transformer.Qwen4MtpState) c_int {
            return if (st.seq_offset == 0) st.pos_base else st.pos_base + @as(c_int, @intCast(st.seq_offset));
        }
    };
    const ha = try H.row(s, stream_a, 0);
    defer _ = mlx.mlx_array_free(ha);
    const hb = try H.row(s, stream_b, 0);
    defer _ = mlx.mlx_array_free(hb);
    {
        var warm = try xfm.qwen4MtpStateNew();
        defer warm.deinit();
        const idw = H.tok(host_ids[1]);
        defer _ = mlx.mlx_array_free(idw);
        const o = try xfm.qwen4MtpForwardOn(&warm, ha, idw, 1, null, .last_row);
        try mlx.check(mlx.mlx_array_eval(o.logits));
        if (o.logits.ctx != null) _ = mlx.mlx_array_free(o.logits);
        _ = mlx.mlx_array_free(o.stream);
        try mlx.check(mlx.mlx_synchronize(s));
    }

    const ns = [_]usize{ 2, 4 };
    const depths = [_]u32{ 1, 2, 4 };
    for ([_]bool{ false, true }) |with_stash| {
        for (ns) |N| {
            const n_use: usize = N;
            for (depths) |depth| {
                var solo: [4]Transformer.Qwen4MtpState = undefined;
                var bat: [4]Transformer.Qwen4MtpState = undefined;
                for (0..n_use) |k| {
                    solo[k] = try xfm.qwen4MtpStateNew();
                    bat[k] = try xfm.qwen4MtpStateNew();
                    solo[k].pos_base = @intCast(1 + (k % 2) * 7);
                    bat[k].pos_base = solo[k].pos_base;
                }
                defer for (0..n_use) |k| {
                    solo[k].deinit();
                    bat[k].deinit();
                };
                var hs: [4]mlx.mlx_array = undefined;
                var hb2: [4]mlx.mlx_array = undefined;
                const srcs = [_]mlx.mlx_array{ ha, hb, ha, hb };
                const tok0 = [_]i32{ host_ids[1], host_ids[4], host_ids[2], host_ids[5] };
                for (0..n_use) |k| {
                    hs[k] = mlx.mlx_array_new();
                    const initial = if (with_stash) try H.span(s, if (k % 2 == 0) stream_a else stream_b, 0, 2 + k % 2) else try H.row(s, srcs[k], 0);
                    defer _ = mlx.mlx_array_free(initial);
                    try mlx.check(mlx.mlx_array_set(&hs[k], initial));
                    hb2[k] = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_array_set(&hb2[k], hs[k]));
                }
                defer for (hs[0..n_use], hb2[0..n_use]) |a, c| {
                    _ = mlx.mlx_array_free(a);
                    _ = mlx.mlx_array_free(c);
                };
                var tok_s: [4]i32 = tok0;
                var tok_b: [4]i32 = tok0;
                var steps_s: [4]u32 = @splat(0);
                var steps_b: [4]u32 = @splat(0);
                var d: u32 = 0;
                while (d < depth) : (d += 1) {
                    var ids_s: [4]mlx.mlx_array = undefined;
                    var ids_b: [4]mlx.mlx_array = undefined;
                    for (0..n_use) |k| {
                        if (with_stash and d == 0) {
                            const count = 2 + k % 2;
                            const shape = [_]c_int{ 1, @intCast(count) };
                            ids_s[k] = mlx.mlx_array_new_data(host_ids[k + 1 ..][0..count].ptr, &shape, 2, .int32);
                            ids_b[k] = mlx.mlx_array_new_data(host_ids[k + 1 ..][0..count].ptr, &shape, 2, .int32);
                        } else {
                            ids_s[k] = H.tok(tok_s[k]);
                            ids_b[k] = H.tok(tok_b[k]);
                        }
                    }
                    defer for (0..n_use) |k| {
                        _ = mlx.mlx_array_free(ids_s[k]);
                        _ = mlx.mlx_array_free(ids_b[k]);
                    };
                    var solo_out: [4]Transformer.Qwen4MtpOut = undefined;
                    for (0..n_use) |k| {
                        solo_out[k] = try xfm.qwen4MtpForwardOn(&solo[k], hs[k], ids_s[k], H.pos(&solo[k]), null, .last_row);
                    }
                    defer for (0..n_use) |k| {
                        if (solo_out[k].logits.ctx != null) _ = mlx.mlx_array_free(solo_out[k].logits);
                        _ = mlx.mlx_array_free(solo_out[k].stream);
                    };
                    var brows_buf: [4]Transformer.Qwen4MtpBatchRow = undefined;
                    for (0..n_use) |k| {
                        brows_buf[k] = .{ .st = &bat[k], .stream = hb2[k], .token_ids = ids_b[k], .pos_offset = H.pos(&bat[k]) };
                    }
                    const bout = try xfm.qwen4MtpForwardBatched(brows_buf[0..n_use], .last_row, depth);
                    defer {
                        for (bout) |o| {
                            if (o.logits.ctx != null) _ = mlx.mlx_array_free(o.logits);
                            if (o.stream.ctx != null) _ = mlx.mlx_array_free(o.stream);
                            if (o.mixed.ctx != null) _ = mlx.mlx_array_free(o.mixed);
                        }
                        allocator.free(bout);
                    }
                    for (0..n_use) |k| {
                        const eq_logits = try qsaArraysAllEqual(solo_out[k].logits, bout[k].logits, s);
                        if (!eq_logits) {
                            const eq_stream = try qsaArraysAllEqual(solo_out[k].stream, bout[k].stream, s);
                            var row_mix = try xfm.hcRead(bout[k].stream, &xfm.qwen4_mtp.?.mixer, 1, 1);
                            defer row_mix.deinit();
                            const row_logits = try xfm.lmHeadProject(row_mix.mixed, false);
                            defer _ = mlx.mlx_array_free(row_logits);
                            const eq_readout = try qsaArraysAllEqual(row_logits, bout[k].logits, s);
                            std.debug.print("[mtp mlx_equal] N={d} depth={d} step={d} row={d} logits=NE stream={} readout={}\n", .{ N, depth, d, k, eq_stream, eq_readout });
                        }
                        try t.expect(eq_logits);
                        const ts = try H.top2(allocator, solo_out[k].logits, s);
                        const tb = try H.top2(allocator, bout[k].logits, s);
                        try t.expectEqual(ts.id, tb.id);
                        tok_s[k] = ts.id;
                        tok_b[k] = tb.id;
                        steps_s[k] += 1;
                        steps_b[k] += 1;
                        _ = mlx.mlx_array_free(hs[k]);
                        _ = mlx.mlx_array_free(hb2[k]);
                        hs[k] = mlx.mlx_array_new();
                        hb2[k] = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_array_set(&hs[k], solo_out[k].stream));
                        try mlx.check(mlx.mlx_array_set(&hb2[k], bout[k].stream));
                    }
                }
            }
        }
    }
}

test "verify vocabulary projections preserve solo logits across request widths (MTP_LMHEAD_TEST_FILE)" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E);
    const path = std.c.getenv("MTP_LMHEAD_TEST_FILE") orelse return error.SkipZigTest;
    var weights = try model_mod.loadWeightsSingleFile(testing.allocator, std.mem.span(path));
    defer weights.deinit();
    const w = weights.get("language_model.lm_head.weight") orelse return error.MissingLmHead;
    const sc = weights.get("language_model.lm_head.scales") orelse return error.MissingLmHead;
    const bi = weights.get("language_model.lm_head.biases") orelse return error.MissingLmHead;
    const k: c_int = 2560;
    const n: c_int = 248320;
    {
        for ([_]usize{ 2, 3, 4, 8 }) |count| for ([_][8]c_int{ @splat(2), @splat(3), @splat(4), @splat(5), @splat(6), @splat(7), .{ 5, 3, 2, 4, 2, 2, 3, 3 } }) |lengths| {
            var total: c_int = 0;
            for (lengths[0..count]) |width| total += width;
            const x = try attn256RandBf16(prng.random(), &.{ 1, total, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (total > 12 or lengths[0] > 6) {
                try testing.expect((try verifyLmHeadProjection(s, x, w, sc, bi, lengths[0..count])) == null);
                continue;
            }
            const grouped = (try verifyLmHeadProjection(s, x, w, sc, bi, lengths[0..count])) orelse return error.LmHeadProjectionDeclined;
            defer _ = mlx.mlx_array_free(grouped);
            var offset: c_int = 0;
            for (lengths[0..count]) |width| {
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, k }, 3, &.{ 1, 1, 1 }, 3, s));
                const expected = try qmatmulBits(row, w, sc, bi, 8, 64, .affine, s);
                defer _ = mlx.mlx_array_free(expected);
                var actual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_slice(&actual, grouped, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, n }, 3, &.{ 1, 1, 1 }, 3, s));
                try testing.expect(try qsaArraysAllEqual(expected, actual, s));
                offset += width;
            }
        };
        const x = try attn256RandBf16(prng.random(), &.{ 1, 8, k }, s);
        defer _ = mlx.mlx_array_free(x);
        for ([_][]const c_int{ &.{ 1, 7 }, &.{8}, &.{ 2, 2 }, &.{ 0, 8 } }) |widths| {
            try testing.expect((try verifyLmHeadProjection(s, x, w, sc, bi, widths)) == null);
        }
        try testing.expect((try verifyLmHeadProjection(s, x, w, sc, .{}, &.{ 4, 4 })) == null);
        const wide = try attn256RandBf16(prng.random(), &.{ 1, 25, k }, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expect((try verifyLmHeadProjection(s, wide, w, sc, bi, &.{ 5, 5, 5, 5, 5 })) == null);
        const old = trf.vqmm_mixed_plain_test_force;
        trf.vqmm_mixed_plain_test_force = true;
        defer trf.vqmm_mixed_plain_test_force = old;
        try testing.expect((try verifyLmHeadProjection(s, x, w, sc, bi, &.{ 4, 4 })) == null);
    }
}

test "qwen4 fixture vision: tower parity, pre-tile splice + M-RoPE prefill/decode, QSA mask at 3-D block positions (QWEN4_FIXTURE_VISION)" {
    // `tests/qwen4/dump_qwen4_exp_fixtures.py build|dump --vision` + the converter's
    // `--add-vision --src`. Proves, together: the tower on this arch's names,
    // the splice BEFORE the hyper-connection tile, the M-RoPE chunk tables
    // reaching attention AND the QSA indexer, and the indexer's pooled block
    // keys roped at 3-D block-start positions (the layer-3 mask is EXACT).
    const qwen_vision = @import("qwen_vision.zig");
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("QWEN4_FIXTURE_VISION") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    try testing.expect(config.qwen_vision and config.image_token_id != 0);
    var weights = try model_mod.loadWeightsWithVision(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    xfm.qwen4_stream_f32 = true;
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    var qv = try qwen_vision.QwenVision.init(allocator, config, &weights);
    defer qv.deinit();

    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    const T: usize = mlx.mlx_array_size(ids_arr);
    const ids_src = mlx.mlx_array_data_int32(ids_arr) orelse return error.Unreadable;
    const v: usize = @intCast(config.vocab_size);
    const ref_full = try qwen4ReadF32(allocator, fx.get("logits_full") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_full);
    const ref_last = try qwen4ReadF32(allocator, fx.get("logits_prefill_last") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_last);
    const ref_dec = try qwen4ReadF32(allocator, fx.get("logits_decode") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_dec);
    const ties = try Qwen4Ties.read(allocator, &fx, s);
    defer ties.deinit(allocator);
    const t_dec: usize = ref_dec.len / v;
    const t_pre: usize = T - t_dec;
    // Our own full-forward logits, the bar for the chunked arm below.
    const ours_full = try allocator.alloc(f32, T * v);
    defer allocator.free(ours_full);

    // [a] tower: pixel_values → post-merger rows vs the reference pooler_output.
    const grid_arr = fx.get("image_grid_thw") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(grid_arr));
    const grid = mlx.mlx_array_data_int32(grid_arr) orelse return error.Unreadable;
    const gt: u32 = @intCast(grid[0]);
    const gh: u32 = @intCast(grid[1]);
    const gw: u32 = @intCast(grid[2]);
    const px = fx.get("pixel_values") orelse return error.MissingFixtureTensor;
    const emb = try qv.forward(px, gh, gw); // [1, N/merge², out_hidden]
    defer _ = mlx.mlx_array_free(emb);
    const ref_emb = try qwen4ReadF32(allocator, fx.get("image_embeds") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_emb);
    const our_emb = try qwen4ReadF32(allocator, emb, s);
    defer allocator.free(our_emb);
    try testing.expectEqual(ref_emb.len, our_emb.len);
    const hid: usize = @intCast(config.qv_out_hidden);
    const n_img: usize = ref_emb.len / hid;
    const re = qwen4CompareRows(our_emb, ref_emb, n_img, hid);
    errdefer std.debug.print("[qwen4 fixture vision] tower {d}x{d}x{d} -> {d} rows: min row cos {d:.5}\n", .{ gt, gh, gw, n_img, re.min_cos });
    try testing.expect(re.min_cos > 0.99);

    // [b] 3-D positions: our get_rope_index == the reference table + delta.
    const ref_pos_arr = fx.get("position_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ref_pos_arr));
    const ref_pos = mlx.mlx_array_data_int32(ref_pos_arr) orelse return error.Unreadable;
    const ref_delta_arr = fx.get("rope_delta") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ref_delta_arr));
    const ref_delta = (mlx.mlx_array_data_int32(ref_delta_arr) orelse return error.Unreadable)[0];
    const host_ids = try allocator.alloc(u32, T);
    defer allocator.free(host_ids);
    for (0..T) |i| host_ids[i] = @intCast(ids_src[i]);
    const grids = [_]mrope.ImageGrid{.{ .t = gt, .h = gh, .w = gw }};
    var ri = try mrope.getRopeIndex(allocator, host_ids, &grids, &.{}, config.image_token_id, config.video_token_id, config.vision_start_token_id, config.qv_merge);
    defer ri.deinit();
    try testing.expectEqual(ref_delta, ri.delta);
    const pos_flat = try allocator.alloc(i32, 3 * T);
    defer allocator.free(pos_flat);
    for (0..3) |axis| for (0..T) |i| {
        pos_flat[axis * T + i] = ri.pos[axis][i];
        try testing.expectEqual(ref_pos[axis * T + i], ri.pos[axis][i]);
    };
    var n_pad_total: usize = 0;
    for (host_ids) |t| {
        if (t == config.image_token_id) n_pad_total += 1;
    }
    try testing.expectEqual(n_img, n_pad_total);

    // [c] one full-prompt forward: streams, final logits, layer-3 QSA mask EXACT.
    {
        const shape = [_]c_int{ 1, @intCast(T) };
        const ids = mlx.mlx_array_new_data(@ptrCast(ids_src), &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(ids);
        var ctx = xfm.defaultCtx();
        ctx.vision_embeddings = emb;
        ctx.mrope_pos = pos_flat;
        ctx.mrope_total = T;
        ctx.mrope_delta = ri.delta;
        var trace: Transformer.Qwen4Trace = .{};
        defer trace.deinit();
        Transformer.qwen4_trace = &trace;
        defer Transformer.qwen4_trace = null;
        const logits = try xfm.forwardWith(&ctx, ids);
        defer _ = mlx.mlx_array_free(logits);
        // The reference's own cos/sin table [T, rope_dims] vs our chunk build.
        if (fx.get("rope_cos")) |ref_cos_arr| {
            const ref_cos = try qwen4ReadF32(allocator, ref_cos_arr, s);
            defer allocator.free(ref_cos);
            const ours_cs = try xfm.mropeCosSinAt(.{ .pos = pos_flat, .total = T, .delta = ri.delta }, 0, 1, T, .float32);
            defer _ = mlx.mlx_array_free(ours_cs.cos);
            defer _ = mlx.mlx_array_free(ours_cs.sin);
            const our_cos = try qwen4ReadF32(allocator, ours_cs.cos, s);
            defer allocator.free(our_cos);
            try testing.expectEqual(ref_cos.len, our_cos.len);
            var max_d: f32 = 0;
            for (ref_cos, our_cos) |a, b| max_d = @max(max_d, @abs(a - b));
            errdefer std.debug.print("[qwen4 fixture vision] M-RoPE cos table vs reference: max |d| {e:.3}\n", .{max_d});
            try testing.expect(max_d < 1e-4);
        }
        // The mask is the direct guard for the pooled-key angles: a scalar
        // rope on the image blocks picks different blocks.
        const ref_mask = try qwen4ReadF32(allocator, fx.get("l3_qsa_mask") orelse return error.MissingFixtureTensor, s);
        defer allocator.free(ref_mask);
        try testing.expect(trace.qsa_mask.ctx != null);
        const our_mask = try qwen4ReadF32(allocator, trace.qsa_mask, s);
        defer allocator.free(our_mask);
        try testing.expectEqual(ref_mask.len, our_mask.len);
        var mask_bad: usize = 0;
        var mask_tied_rows: usize = 0;
        for (0..T) |row| {
            if (ties.rowTied(row)) {
                mask_tied_rows += 1;
                continue;
            }
            for (0..T) |c| {
                if (ref_mask[row * T + c] != our_mask[row * T + c]) mask_bad += 1;
            }
        }
        errdefer std.debug.print("[qwen4 fixture vision] layer-3 QSA mask: {d} mismatched over {d} decisive rows ({d} acquitted)\n", .{ mask_bad, T - mask_tied_rows, mask_tied_rows });
        try testing.expect(mask_tied_rows < T / 2);
        try testing.expectEqual(@as(usize, 0), mask_bad);
        const ours = try qwen4ReadF32(allocator, logits, s);
        defer allocator.free(ours);
        try testing.expectEqual(T * v, ours.len);
        @memcpy(ours_full, ours);
        const r = qwen4CompareTied("logits", ours, ref_full, T, v, 0, ties);
        errdefer std.debug.print("[qwen4 fixture vision] full prefill T={d}: min cos {d:.5} ({d} rows acquitted) argmax {d}/{d} decisive rows\n", .{ T, r.min_cos, r.tied, r.agree, r.decisive });
        try testing.expect(r.min_cos > 0.995);
        // A k < E fixture (`build --topk`) ties in most trunk rows across
        // every MoE layer; its selection coverage is the MTP head's ONE
        // MoE layer (decisive floor above). k = E fixtures keep the T/2 floor.
        try testing.expect(r.decisive >= (if (ties.route_gap != null) @as(usize, 2) else T / 2) and r.agree == r.decisive);
    }

    // [d] fresh state: prefill in TWO chunks split inside the image span
    // (`vision_splice_offset` + pooled block keys built across chunks), then
    // decode past the table (positions abs + delta) vs the reference.
    {
        var cache = try KVCache.init(allocator, config.num_hidden_layers);
        defer cache.deinit();
        var off: usize = 0;
        const entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
        defer allocator.free(entries);
        for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer for (entries) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        };
        var ctx: ForwardCtx = .{ .cache = &cache, .moe_seq_offset = &off, .ssm_entries = entries, .capture_hidden = null, .vision_embeddings = emb };
        ctx.mrope_pos = pos_flat;
        ctx.mrope_total = T;
        ctx.mrope_delta = ri.delta;
        var first_pad: usize = 0;
        while (first_pad < T and host_ids[first_pad] != config.image_token_id) first_pad += 1;
        const t1: usize = first_pad + n_img / 2;
        try testing.expect(t1 < t_pre);
        const c1 = mlx.mlx_array_new_data(@ptrCast(ids_src), &[_]c_int{ 1, @intCast(t1) }, 2, .int32);
        defer _ = mlx.mlx_array_free(c1);
        const l1 = try xfm.forwardWith(&ctx, c1);
        _ = mlx.mlx_array_free(l1);
        ctx.vision_splice_offset = t1 - first_pad;
        const c2 = mlx.mlx_array_new_data(@ptrCast(ids_src + t1), &[_]c_int{ 1, @intCast(t_pre - t1) }, 2, .int32);
        defer _ = mlx.mlx_array_free(c2);
        const plog = try xfm.forwardWith(&ctx, c2);
        defer _ = mlx.mlx_array_free(plog);
        ctx.vision_splice_offset = 0;
        const pours = try qwen4ReadF32(allocator, plog, s);
        defer allocator.free(pours);
        // The chunking invariant is OUR full forward: the split lands inside
        // the image and the last row may be a reference near-tie, but chunked
        // == unchunked is ours to keep (chunk-boundary rounding only).
        const last = pours[(t_pre - t1 - 1) * v ..][0..v];
        const rself = qwen4CompareRows(last, ours_full[(t_pre - 1) * v ..][0..v], 1, v);
        const rp = qwen4CompareTied("prefill-last", last, ref_last, 1, v, t_pre - 1, ties);
        errdefer std.debug.print("[qwen4 fixture vision] chunked prefill {d}+{d}: last-row cos vs our full forward {d:.5}, vs reference {d:.5}\n", .{ t1, t_pre - t1, rself.min_cos, rp.min_cos });
        try testing.expect(rself.min_cos > 0.999);
        try testing.expect(rp.min_cos > 0.995);

        var dec: Qwen4TiedCmp = .{ .min_cos = 1.0, .agree = 0, .decisive = 0, .tied = 0 };
        for (0..t_dec) |k| {
            const dshape = [_]c_int{ 1, 1 };
            const did = mlx.mlx_array_new_data(@ptrCast(ids_src + t_pre + k), &dshape, 2, .int32);
            defer _ = mlx.mlx_array_free(did);
            const dlog = try xfm.forwardWith(&ctx, did);
            defer _ = mlx.mlx_array_free(dlog);
            const dours = try qwen4ReadF32(allocator, dlog, s);
            defer allocator.free(dours);
            const rd = qwen4CompareTied("decode", dours, ref_dec[k * v ..][0..v], 1, v, t_pre + k, ties);
            if (rd.min_cos < dec.min_cos) dec.min_cos = rd.min_cos;
            dec.agree += rd.agree;
            dec.decisive += rd.decisive;
            dec.tied += rd.tied;
        }
        errdefer std.debug.print("[qwen4 fixture vision] decode x{d}: min cos {d:.5} ({d} acquitted) argmax {d}/{d} decisive rows\n", .{ t_dec, dec.min_cos, dec.tied, dec.agree, dec.decisive });
        try testing.expect(dec.min_cos > 0.995);
        try testing.expect(dec.decisive >= (if (ties.route_gap != null) @as(usize, 0) else 2) and dec.agree == dec.decisive);
    }
}

test "qwen4 fixture bf16: residual streams track a bf16 reference (QWEN4_FIXTURE_BF16)" {
    // `tests/qwen4/dump_qwen4_exp_fixtures.py` with QWEN4_REF_DTYPE=bf16. The shipped
    // checkpoint runs bf16 residual streams; against the f32 fixture that path
    // reads 0.905 at the logits (bf16 rounding compounds through the tiny
    // random model), against a bf16 reference the streams agree to 0.9998.
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("QWEN4_FIXTURE_BF16") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    const ids_shape = mlx.getShape(ids_arr);
    const T: usize = @intCast(ids_shape[ids_shape.len - 1]);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    const tshape = [_]c_int{ 1, @intCast(T) };
    try mlx.check(mlx.mlx_reshape(&ids, ids_arr, &tshape, 2, s));

    var ctx = xfm.defaultCtx();
    const n_layers: usize = config.num_hidden_layers;
    const cap_ids = try allocator.alloc(u32, n_layers);
    defer allocator.free(cap_ids);
    const cap_out = try allocator.alloc(mlx.mlx_array, n_layers);
    defer allocator.free(cap_out);
    for (0..n_layers) |i| {
        cap_ids[i] = @intCast(i);
        cap_out[i] = mlx.mlx_array_new();
    }
    defer for (cap_out) |o| {
        _ = mlx.mlx_array_free(o);
    };
    var cap_layers: CaptureLayers = .{ .ids = cap_ids, .out = cap_out };
    ctx.capture_layers = &cap_layers;
    const logits = try xfm.forwardWith(&ctx, ids);
    defer _ = mlx.mlx_array_free(logits);
    var checked: usize = 0;
    for (1..n_layers) |i| {
        var nb: [32]u8 = undefined;
        const key = std.fmt.bufPrint(&nb, "stream_{d}", .{i}) catch unreachable;
        const ref = fx.get(key) orelse continue;
        const ref_h = try qwen4ReadF32(allocator, ref, s);
        defer allocator.free(ref_h);
        const our_h = try qwen4ReadF32(allocator, cap_out[i - 1], s);
        defer allocator.free(our_h);
        if (our_h.len != ref_h.len) continue;
        const rc = qwen4CompareRows(our_h, ref_h, T, ref_h.len / T);
        errdefer std.debug.print("[qwen4 fixture bf16] after layer {d}: min row cos {d:.5}\n", .{ i - 1, rc.min_cos });
        try testing.expect(rc.min_cos > 0.999);
        checked += 1;
    }
    try testing.expect(checked >= n_layers - 2);
}

test "qwen4 fixture bf16 streamed experts are bit-identical to resident experts" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    const source_dir = std.c.getenv("QWEN4_TEST_HF_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    try qwen4StreamedResidentFixtureParity(std.testing.allocator, std.testing.io, std.mem.span(model_dir), std.mem.span(source_dir));
}

test "qwen4 hc_mean capture is the mean of the four HC streams at the layer output (QWEN4_TEST_MODEL)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();

    const hc: usize = 4;
    const hidden: usize = @intCast(config.hidden_size);
    const prompt = [_]i32{ 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000 };
    const layer_ids = [_]u32{ 5, 15 };
    var raw_out = [_]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    var mean_out = [_]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (raw_out ++ mean_out) |a| {
        _ = mlx.mlx_array_free(a);
    };
    inline for (.{ &raw_out, &mean_out }, .{ 0, hc }) |outs, mean_streams| {
        const slot = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
        defer slot.deinit(allocator);
        var cl: CaptureLayers = .{ .ids = &layer_ids, .out = outs, .hc_mean = mean_streams };
        slot.ctx.capture_layers = &cl;
        const logits = try slot.forward(&xfm, &prompt);
        _ = mlx.mlx_array_free(logits);
    }
    for (raw_out, mean_out) |raw_arr, mean_arr| {
        const raw = try qwen4ReadF32(allocator, raw_arr, s);
        defer allocator.free(raw);
        const got = try qwen4ReadF32(allocator, mean_arr, s);
        defer allocator.free(got);
        try testing.expectEqual(prompt.len * hc * hidden, raw.len);
        try testing.expectEqual(prompt.len * hidden, got.len);
        var worst: f32 = 0;
        for (0..prompt.len) |row| {
            for (0..hidden) |c| {
                var acc: f32 = 0;
                for (0..hc) |k| acc += raw[(row * hc + k) * hidden + c];
                const want = acc / @as(f32, hc);
                const err = @abs(got[row * hidden + c] - want) / @max(1.0, @abs(want));
                worst = @max(worst, err);
            }
        }
        // bf16 output of an f32-accumulated mean of bf16 streams.
        try testing.expect(worst < 0.01);
    }
}

test "qwen4 batched decode: one forwardMoeBatchedDecode tick == two serial ticks (GDN + PLE history + QSA keys/mask)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("QWEN4_FIXTURE") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.supportsBatchedGdnDecode());
    xfm.qwen4_stream_f32 = true;
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();

    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    const T: usize = mlx.mlx_array_size(ids_arr);
    const ids = (mlx.mlx_array_data_int32(ids_arr) orelse return error.Unreadable)[0..T];
    const v: usize = @intCast(config.vocab_size);
    const n_layers = config.num_hidden_layers;

    // Two prompts of DIFFERENT length (pads the shorter slot's KV) whose
    // continuations run every slot past the indexer budget.
    const pa: usize = T / 2;
    const pb: usize = pa - 3;
    const K: usize = T - pa;
    const a_ids = ids[0 .. pa + K];
    const b_ids = ids[3 .. 3 + pb + K];

    const a_s = try Qwen4TestSlot.init(allocator, n_layers);
    defer a_s.deinit(allocator);
    const b_s = try Qwen4TestSlot.init(allocator, n_layers);
    defer b_s.deinit(allocator);
    const a_b = try Qwen4TestSlot.init(allocator, n_layers);
    defer a_b.deinit(allocator);
    const b_b = try Qwen4TestSlot.init(allocator, n_layers);
    defer b_b.deinit(allocator);
    _ = mlx.mlx_array_free(try a_s.forward(&xfm, a_ids[0..pa]));
    _ = mlx.mlx_array_free(try b_s.forward(&xfm, b_ids[0..pb]));
    _ = mlx.mlx_array_free(try a_b.forward(&xfm, a_ids[0..pa]));
    _ = mlx.mlx_array_free(try b_b.forward(&xfm, b_ids[0..pb]));

    var min_cos: f64 = 1.0;
    var checked: usize = 0;
    var agreed: usize = 0;
    for (0..K) |k| {
        const ta = a_ids[pa + k];
        const tb = b_ids[pb + k];
        const la = try a_s.forward(&xfm, &.{ta});
        defer _ = mlx.mlx_array_free(la);
        const lb = try b_s.forward(&xfm, &.{tb});
        defer _ = mlx.mlx_array_free(lb);

        const ctxs = [_]*ForwardCtx{ &a_b.ctx, &b_b.ctx };
        const offs = [_]u32{ @intCast(a_b.off), @intCast(b_b.off) };
        const out = try xfm.forwardMoeBatchedDecode(&.{ @intCast(ta), @intCast(tb) }, &ctxs, &offs, null);
        defer {
            for (out) |o| _ = mlx.mlx_array_free(o);
            allocator.free(out);
        }
        // The scheduler advances each slot's own position after the tick.
        a_b.off += 1;
        b_b.off += 1;
        try testing.expectEqual(a_s.off, a_b.off);
        try testing.expectEqual(b_s.off, b_b.off);

        inline for (.{ la, lb }, .{ out[0], out[1] }) |ser, bat| {
            const sh = try qwen4ReadF32(allocator, ser, s);
            defer allocator.free(sh);
            const bh = try qwen4ReadF32(allocator, bat, s);
            defer allocator.free(bh);
            try testing.expectEqual(sh.len, bh.len);
            const rc = qwen4CompareRows(sh, bh, 1, v);
            if (rc.min_cos < min_cos) min_cos = rc.min_cos;
            // argmax must agree wherever the serial top-2 is decisive (3%).
            var top: f32 = -std.math.inf(f32);
            var second: f32 = -std.math.inf(f32);
            for (sh) |x| {
                if (x > top) {
                    second = top;
                    top = x;
                } else if (x > second) second = x;
            }
            if (@abs(top - second) > 0.03 * @abs(top)) {
                checked += 1;
                agreed += rc.argmax_agree;
            }
        }
        // Per-slot state moved identically: PLE history, QSA key rows, pooled blocks.
        for (0..n_layers) |l| {
            inline for (.{ .{ a_s, a_b }, .{ b_s, b_b } }) |pair| {
                const es = &pair[0].entries[l];
                const eb = &pair[1].entries[l];
                try testing.expectEqual(es.ple_prev_valid, eb.ple_prev_valid);
                try testing.expectEqualSlices(u32, &es.ple_prev, &eb.ple_prev);
                try testing.expectEqual(qwen4Rows(es.aux_state), qwen4Rows(ssmLiveAux(eb)));
                try testing.expectEqual(qwen4Rows(es.qsa_pooled), qwen4Rows(eb.qsa_pooled));
            }
        }
    }
    errdefer std.debug.print("[qwen4 batched] {d} ticks x 2 slots: min cos {d:.5}, argmax {d}/{d} decisive rows\n", .{ K, min_cos, agreed, checked });
    try testing.expect(min_cos > 0.995);
    try testing.expect(checked >= 2 and agreed == checked);
    // Both slots crossed the budget: the batched QSA mask path ran.
    var pooled = false;
    for (b_b.entries) |*e| if (e.qsa_pooled.ctx != null) {
        pooled = true;
    };
    try testing.expect(pooled);
}

test "qwen4 per-row batched decode == solo ticks bit for bit (ragged KV, join/leave, M-RoPE)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const old_path = diagEnvOn("QWEN4_ROWS_OLD");
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.supportsBatchedQwen4Rows());
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    const n_layers = config.num_hidden_layers;

    const lens = [_]usize{ 2300, 3100, 5200, 2100 };
    const mrope_slot = 3;
    const delta: i32 = -3;
    var table: [3 * 2100]i32 = undefined;
    for (0..3) |axis| for (0..2100) |i| {
        table[axis * 2100 + i] = @intCast(i);
    };
    var solo: [4]*Qwen4TestSlot = undefined;
    var batch: [4]*Qwen4TestSlot = undefined;
    var made: usize = 0;
    defer for (0..made) |i| {
        solo[i].deinit(allocator);
        batch[i].deinit(allocator);
    };
    var prompt: [5200]i32 = undefined;
    for (0..4) |slot| {
        var x: u64 = 0x9e3779b97f4a7c15 +% slot;
        for (prompt[0..lens[slot]]) |*id| {
            x = x *% 6364136223846793005 +% 1442695040888963407;
            id.* = @intCast(1000 + (x >> 33) % 100000);
        }
        solo[slot] = try Qwen4TestSlot.initWith(allocator, n_layers, KVQuantConfig.affine(8));
        batch[slot] = try Qwen4TestSlot.initWith(allocator, n_layers, KVQuantConfig.affine(8));
        made += 1;
        for ([_]*Qwen4TestSlot{ solo[slot], batch[slot] }) |sl| {
            if (slot == mrope_slot) {
                sl.ctx.mrope_pos = &table;
                sl.ctx.mrope_total = 2100;
            }
            _ = mlx.mlx_array_free(try sl.forward(&xfm, prompt[0..lens[slot]]));
            if (slot == mrope_slot) sl.ctx.mrope_delta = delta;
        }
    }

    // Slots 0 and 1 run together, 2 joins, 0 leaves as 3 joins, 2 is cancelled, 0 comes back.
    const active = [_][]const usize{
        &.{ 0, 1 },    &.{ 0, 1 },    &.{ 0, 1, 2 }, &.{ 0, 1, 2 }, &.{ 1, 2, 3 },
        &.{ 1, 2, 3 }, &.{ 1, 3 },    &.{ 0, 1, 3 }, &.{ 0, 1, 3 }, &.{ 0, 1, 2, 3 },
    };
    var mismatched: usize = 0;
    for (active, 0..) |set, tick| {
        var ctxs: [4]*ForwardCtx = undefined;
        var toks: [4]u32 = undefined;
        var offs: [4]u32 = undefined;
        for (set, 0..) |slot, i| {
            ctxs[i] = &batch[slot].ctx;
            toks[i] = @intCast(5000 + (slot * 131 + tick * 17) % 90000);
            offs[i] = @intCast(batch[slot].off);
        }
        const out = if (old_path)
            try xfm.forwardMoeBatchedDecode(toks[0..set.len], ctxs[0..set.len], offs[0..set.len], null)
        else
            try xfm.forwardQwen4DecodeRows(toks[0..set.len], ctxs[0..set.len], null);
        defer {
            for (out) |o| _ = mlx.mlx_array_free(o);
            allocator.free(out);
        }
        for (set, 0..) |slot, i| {
            if (old_path) batch[slot].off += 1;
            const id: i32 = @intCast(toks[i]);
            const ref = try solo[slot].forward(&xfm, &.{id});
            defer _ = mlx.mlx_array_free(ref);
            if (!try qwen4BitsEqual(allocator, ref, out[i], s)) {
                mismatched += 1;
                std.debug.print("[qwen4 rows] tick {d} slot {d} (rows {d}): logits differ\n", .{ tick, slot, set.len });
            }
            try testing.expectEqual(solo[slot].off, batch[slot].off);
        }
    }
    try testing.expectEqual(@as(usize, 0), mismatched);
    if (old_path) return;

    for (0..4) |slot| {
        for (0..n_layers) |l| {
            const es = &solo[slot].entries[l];
            const eb = &batch[slot].entries[l];
            try testing.expectEqual(es.ple_prev_valid, eb.ple_prev_valid);
            try testing.expectEqualSlices(u32, &es.ple_prev, &eb.ple_prev);
            inline for (.{ "conv_state", "ssm_state", "aux_state", "qsa_pooled" }) |field| {
                try testing.expect(try qwen4BitsEqual(allocator, @field(es, field), @field(eb, field), s));
            }
            if (xfm.moe_layers.?[l].attn == .full) {
                var ds = try solo[slot].cache.denseView(@intCast(l), s);
                defer ds.deinit();
                var db = try batch[slot].cache.denseView(@intCast(l), s);
                defer db.deinit();
                try testing.expect(try qwen4BitsEqual(allocator, ds.k, db.k, s));
                try testing.expect(try qwen4BitsEqual(allocator, ds.v, db.v, s));
            }
        }
    }
}

test "qwen4 deferred PLE: pipelined decode AND MTP verify widths match the direct forward (QWEN4_TEST_MODEL)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    // The profiler's synchronizing lap fills the leaf inside the build, so a
    // deferral test has nothing left to observe under it.
    if (diagEnvOn("QWEN4_PROFILE_FWD")) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();

    const State = struct {
        cache: KVCache,
        off: usize = 0,
        entries: []SSMCacheEntry,
        pub fn init(a: std.mem.Allocator, n_layers: u32) !@This() {
            const entries = try a.alloc(SSMCacheEntry, n_layers);
            for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
            const cache = try KVCache.init(a, n_layers);
            return .{ .cache = cache, .entries = entries };
        }
        pub fn deinit(self: *@This(), a: std.mem.Allocator) void {
            for (self.entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                ssmFreeQsaState(e);
            }
            a.free(self.entries);
            self.cache.deinit();
        }
        fn ctx(self: *@This()) ForwardCtx {
            return .{ .cache = &self.cache, .moe_seq_offset = &self.off, .ssm_entries = self.entries, .capture_hidden = null, .vision_embeddings = null };
        }
    };
    var direct = try State.init(allocator, config.num_hidden_layers);
    defer direct.deinit(allocator);
    var deferred = try State.init(allocator, config.num_hidden_layers);
    defer deferred.deinit(allocator);
    var dctx = direct.ctx();
    var lctx = deferred.ctx();

    const prompt = [_]i32{ 5, 17, 42, 9, 23, 8, 31, 2 };
    const pshape = [_]c_int{ 1, prompt.len };
    for ([_]*ForwardCtx{ &dctx, &lctx }) |c| {
        const pids = mlx.mlx_array_new_data(&prompt, &pshape, 2, .int32);
        defer _ = mlx.mlx_array_free(pids);
        const l = try xfm.forwardWith(c, pids);
        try mlx.check(mlx.mlx_array_eval(l));
        _ = mlx.mlx_array_free(l);
    }

    // Two decode steps: the deferred arm builds on an unfilled leaf, then
    // fills it before eval. Both steps must match bit for bit (the second one
    // proves the n-gram history advanced at flush time).
    const steps = [_]i32{ 11, 4 };
    for (steps) |tok| {
        const dshape = [_]c_int{ 1, 1 };
        const did = mlx.mlx_array_new_data(&tok, &dshape, 2, .int32);
        defer _ = mlx.mlx_array_free(did);
        const dl = try xfm.forwardWith(&dctx, did);
        defer _ = mlx.mlx_array_free(dl);

        lctx.ple_defer = true;
        const ll = try xfm.forwardWith(&lctx, did);
        defer _ = mlx.mlx_array_free(ll);
        lctx.ple_defer = false;
        try testing.expect(lctx.ple_pending != null);
        try xfm.flushDeferredPle(&lctx);
        try testing.expect(lctx.ple_pending == null);

        const a = try qwen4ReadF32(allocator, dl, s);
        defer allocator.free(a);
        const b = try qwen4ReadF32(allocator, ll, s);
        defer allocator.free(b);
        try testing.expectEqualSlices(f32, a, b);
    }

    // Verify-width arm: an MTP round builds `[1, 1+m]` over a chain of LAZY
    // drafts with per-position SSM capture on. Deferred must match eager in
    // logits AND in the rollback state the capture leaves behind — the PLE
    // layer's own conv capture runs later in the same forward and keys on
    // `spec_ple_len`, which the deferred gather has not written yet.
    {
        const rows = [_]i32{ 13, 6, 27, 3, 19, 40, 12 };
        for ([_]usize{ 4, 7 }) |w| {
            const vshape = [_]c_int{ 1, @intCast(w) };
            const vids = mlx.mlx_array_new_data(&rows, &vshape, 2, .int32);
            defer _ = mlx.mlx_array_free(vids);

            dctx.capture_ssm_seq = true;
            const dl = try xfm.forwardWith(&dctx, vids);
            defer _ = mlx.mlx_array_free(dl);
            dctx.capture_ssm_seq = false;

            lctx.capture_ssm_seq = true;
            lctx.ple_defer = true;
            const ll = try xfm.forwardWith(&lctx, vids);
            defer _ = mlx.mlx_array_free(ll);
            lctx.ple_defer = false;
            lctx.capture_ssm_seq = false;
            try testing.expect(lctx.ple_pending != null);
            try xfm.flushDeferredPle(&lctx);
            try testing.expect(lctx.ple_pending == null);

            const a = try qwen4ReadF32(allocator, dl, s);
            defer allocator.free(a);
            const b = try qwen4ReadF32(allocator, ll, s);
            defer allocator.free(b);
            try testing.expectEqualSlices(f32, a, b);

            var saw_ple = false;
            for (direct.entries, deferred.entries) |*de, *le| {
                try testing.expectEqual(de.spec_ple_len, le.spec_ple_len);
                try testing.expectEqual(de.ple_prev_valid, le.ple_prev_valid);
                try testing.expectEqualSlices(u32, &de.ple_prev, &le.ple_prev);
                // A deferred gather that never claimed the slot leaves this
                // null while eager holds a capture: `ssmRollbackFromCapture`
                // would then silently take the non-PLE branch.
                try testing.expectEqual(de.spec_ple_input.ctx == null, le.spec_ple_input.ctx == null);
                if (de.spec_ple_len > 0) {
                    saw_ple = true;
                    try testing.expect(de.spec_ple_len >= @as(u32, @intCast(w)));
                    const n: usize = de.spec_ple_len;
                    try testing.expectEqualSlices(u32, de.spec_ple_tokens[0..n], le.spec_ple_tokens[0..n]);
                    try testing.expect(de.spec_ple_input.ctx != null);
                }
            }
            try testing.expect(saw_ple);
            for (direct.entries) |*e| ssmFreeSpecCapture(e);
            for (deferred.entries) |*e| ssmFreeSpecCapture(e);
        }
    }

    // Load-bearing arm: an unflushed leaf is NOT the direct forward.
    {
        const tok: i32 = 7;
        const dshape = [_]c_int{ 1, 1 };
        const did = mlx.mlx_array_new_data(&tok, &dshape, 2, .int32);
        defer _ = mlx.mlx_array_free(did);
        const dl = try xfm.forwardWith(&dctx, did);
        defer _ = mlx.mlx_array_free(dl);
        lctx.ple_defer = true;
        const ll = try xfm.forwardWith(&lctx, did);
        defer _ = mlx.mlx_array_free(ll);
        lctx.ple_defer = false;
        const a = try qwen4ReadF32(allocator, dl, s);
        defer allocator.free(a);
        const b = try qwen4ReadF32(allocator, ll, s);
        defer allocator.free(b);
        xfm.discardDeferredPle(&lctx);
        try testing.expect(!std.mem.eql(f32, a, b));
    }
}

test "qwen4 deferred PLE: the captured hidden the MTP head reads matches the eager capture (QWEN4_TEST_MODEL)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();

    const eager = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
    defer eager.deinit(allocator);
    const lazy = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
    defer lazy.deinit(allocator);

    const prompt = [_]i32{ 5, 17, 42, 9, 23, 8, 31, 2 };
    for ([_]*Qwen4TestSlot{ eager, lazy }) |sl| {
        const l = try sl.forward(&xfm, &prompt);
        defer _ = mlx.mlx_array_free(l);
        try mlx.check(mlx.mlx_array_eval(l));
    }

    // A prefill-width chunk (past the eval cadence, which evaluates the stream
    // mid-layer-loop) and a verify build (per-position SSM capture on, the
    // shape an MTP round runs). Both captures feed the MTP head.
    var rows: [40]i32 = undefined;
    for (&rows, 0..) |*r, i| r.* = @intCast(3 + (i * 37) % 900);
    var mismatched = false;
    for ([_]struct { w: usize, ssm: bool }{ .{ .w = 4, .ssm = true }, .{ .w = 40, .ssm = false } }) |arm| {
        const shape = [_]c_int{ 1, @intCast(arm.w) };
        const ids = mlx.mlx_array_new_data(&rows, &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(ids);

        var e_last = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e_last);
        var e_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e_all);
        eager.ctx.capture_ssm_seq = arm.ssm;
        const e_logits = try xfm.forwardWithCaptureAll(&eager.ctx, ids, &e_last, &e_all);
        defer _ = mlx.mlx_array_free(e_logits);
        eager.ctx.capture_ssm_seq = false;
        try testing.expect(eager.ctx.ple_pending == null);

        var d_last = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d_last);
        var d_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d_all);
        lazy.ctx.capture_ssm_seq = arm.ssm;
        lazy.ctx.ple_defer = true;
        const d_logits = try xfm.forwardWithCaptureAll(&lazy.ctx, ids, &d_last, &d_all);
        defer _ = mlx.mlx_array_free(d_logits);
        lazy.ctx.ple_defer = false;
        lazy.ctx.capture_ssm_seq = false;
        // Filled by an eval inside the build or here: either way the capture
        // must not be holding the zero leaf.
        try xfm.flushDeferredPle(&lazy.ctx);
        try testing.expect(lazy.ctx.ple_pending == null);

        for ([_]struct { e: mlx.mlx_array, d: mlx.mlx_array, label: []const u8 }{
            .{ .e = e_logits, .d = d_logits, .label = "logits" },
            .{ .e = e_all, .d = d_all, .label = "hidden all" },
            .{ .e = e_last, .d = d_last, .label = "hidden last" },
        }) |cmp| {
            const a = try qwen4ReadF32(allocator, cmp.e, s);
            defer allocator.free(a);
            const b = try qwen4ReadF32(allocator, cmp.d, s);
            defer allocator.free(b);
            if (try captureMismatch(a, b, arm.w, cmp.label)) mismatched = true;
        }
        for (eager.entries) |*e| ssmFreeSpecCapture(e);
        for (lazy.entries) |*e| ssmFreeSpecCapture(e);
    }
    if (mismatched) return error.DeferredPleCaptureMismatch;
}

test "qwen4 grow to the reservation before the prefill keeps greedy logits bit-identical (QWEN4_TEST_MODEL)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();

    const grown = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
    defer grown.deinit(allocator);
    const lazy = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
    defer lazy.deinit(allocator);

    // The warm shape: the prompt's buffers hold 256 rows, the reservation 1024. One slot grows
    // before the next chunk; the other at the decode step past 256.
    const prompt = [_]i32{ 5, 17, 42, 9, 23, 8, 31, 2 };
    for ([_]*Qwen4TestSlot{ grown, lazy }) |sl| {
        const l = try sl.forward(&xfm, &prompt);
        defer _ = mlx.mlx_array_free(l);
        try mlx.check(mlx.mlx_array_eval(l));
        sl.cache.reserve(1024);
    }
    try grown.cache.growToReservation(s);
    for (grown.cache.entries) |*e| {
        if (e.initialized and !e.ringed) try testing.expect(KVCache.bufferCapacity(e.keys) >= 1024);
    }

    var chunk: [40]i32 = undefined;
    for (&chunk, 0..) |*r, i| r.* = @intCast(3 + (i * 37) % 900);
    var ids: []const i32 = &chunk;
    var next: [1]i32 = undefined;
    var step: usize = 0;
    while (step < 220) : (step += 1) {
        const g = try grown.forward(&xfm, ids);
        defer _ = mlx.mlx_array_free(g);
        const l = try lazy.forward(&xfm, ids);
        defer _ = mlx.mlx_array_free(l);
        const a = try qwen4ReadF32(allocator, g, s);
        defer allocator.free(a);
        const b = try qwen4ReadF32(allocator, l, s);
        defer allocator.free(b);
        try testing.expectEqual(a.len, b.len);
        for (a, b) |x, y| try testing.expectEqual(@as(u32, @bitCast(x)), @as(u32, @bitCast(y)));
        const vocab: usize = @intCast(mlx.getShape(g)[2]);
        const last = a[a.len - vocab ..];
        var best: usize = 0;
        for (last, 0..) |x, i| {
            if (x > last[best]) best = i;
        }
        next[0] = @intCast(best);
        ids = &next;
    }
}

test "qwen4 MTP verify token values equal serial ticks without near-tie acquittals (QWEN4_TEST_MODEL)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const stream = mlx.gpuStream();
    var config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |path| a.free(path);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    const prompt = [_]i32{ 248045, 846, 198, 7734, 264, 2716, 3255, 883, 264, 11952, 6618, 310, 5984, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };
    const prefix_len = if (std.c.getenv("QWEN4_TEST_PREFIX_TOKENS")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 0;
    const replay_prompt = try a.alloc(i32, prefix_len + prompt.len);
    defer a.free(replay_prompt);
    for (replay_prompt[0..prefix_len], 0..) |*token, i| token.* = prompt[3 + i % 10];
    @memcpy(replay_prompt[prefix_len..], &prompt);
    const continuation = [_]i32{ 4413, 220, 22, 18, 19, 11, 852, 14459, 310, 381, 2512, 328, 57517, 1288, 1467, 524, 3418, 3069, 279, 12313, 36811, 948, 781, 6749, 506, 279, 41564, 13, 1946, 28970, 11, 264, 41564, 557, 15756, 264, 6206, 303, 42900, 69377, 11, 264, 17921, 303, 87840, 9039, 11, 321, 264, 3050, 11013, 314, 2438, 310, 18016, 89661, 13, 1049, 557, 795, 13, 1049, 557, 42642, 13, 271, 3850, 28970, 995, 978, 12244, 310, 279, 5001, 64568, 33377, 314, 279, 39303, 7869, 6272, 15849, 11, 321, 1141, 5839, 32740, 557, 310, 3827, 13, 271, 760, 1118, 17225, 557, 440, 3851, 4897, 11, 264, 83713, 440, 5984, 5304, 2540, 18645, 321, 6213, 421, 9047, 310, 3222, 777, 4602, 1056, 10926, 13, 2838, 8826 };
    var mismatch = false;
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |kv| {
        const serial = try Qwen4TestSlot.init(a, config.num_hidden_layers);
        defer serial.deinit(a);
        try serial.cache.reinit(config.num_hidden_layers, kv);
        try qwen4ReplayPrompt(&xfm, serial, replay_prompt);
        var expected: [continuation.len]usize = undefined;
        var gaps: [continuation.len]f32 = undefined;
        for (continuation, 0..) |token, row| {
            const out = try serial.forward(&xfm, &.{token});
            defer _ = mlx.mlx_array_free(out);
            const values = try qwen4ReadF32(a, out, stream);
            defer a.free(values);
            const top = try qwen4Top2(values);
            expected[row] = top.id;
            gaps[row] = top.gap;
        }
        for (2..9) |width| {
            const verify = try Qwen4TestSlot.init(a, config.num_hidden_layers);
            defer verify.deinit(a);
            try verify.cache.reinit(config.num_hidden_layers, kv);
            try qwen4ReplayPrompt(&xfm, verify, replay_prompt);
            var begin: usize = 0;
            var reported: usize = 0;
            while (begin < continuation.len) {
                const end = @min(begin + width, continuation.len);
                const rows = end - begin;
                const ids = mlx.mlx_array_new_data(continuation[begin..end].ptr, &[_]c_int{ 1, @intCast(rows) }, 2, .int32);
                defer _ = mlx.mlx_array_free(ids);
                var hidden = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(hidden);
                var all = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(all);
                verify.ctx.ple_defer = true;
                verify.ctx.capture_ssm_seq = true;
                const out = try xfm.forwardWithCaptureAll(&verify.ctx, ids, &hidden, &all);
                defer _ = mlx.mlx_array_free(out);
                try xfm.flushDeferredPle(&verify.ctx);
                const values = try qwen4ReadF32(a, out, stream);
                defer a.free(values);
                const vocab: usize = config.vocab_size;
                try testing.expectEqual(rows * vocab, values.len);
                for (0..rows) |row| {
                    const got = try qwen4Top2(values[row * vocab ..][0..vocab]);
                    const position = begin + row;
                    if (expected[position] != got.id) {
                        mismatch = true;
                        if (reported < 3) std.debug.print("MTP value divergence: kv={d} width={d} token={d} serial={d} verify={d} serial_gap={d} verify_gap={d}\n", .{ kv.bits, width, position, expected[position], got.id, gaps[position], got.gap });
                        reported += 1;
                    }
                }
                for (verify.entries) |*entry| ssmFreeSpecCapture(entry);
                begin = end;
            }
        }
    }
    if (mismatch) return error.MtpSerialTokenDivergence;
}

test "batched PLE defer engages at N=2 and matches eager at S=1 and S=4" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    xfm.qwen4_stream_f32 = true;
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();

    const n_layers = config.num_hidden_layers;
    const prompt_a = [_]i32{ 5, 17, 42, 9, 23, 8, 31, 2 };
    const prompt_b = [_]i32{ 11, 4, 19, 7, 28, 3 };
    const a_e = try Qwen4TestSlot.init(allocator, n_layers);
    defer a_e.deinit(allocator);
    const b_e = try Qwen4TestSlot.init(allocator, n_layers);
    defer b_e.deinit(allocator);
    const a_d = try Qwen4TestSlot.init(allocator, n_layers);
    defer a_d.deinit(allocator);
    const b_d = try Qwen4TestSlot.init(allocator, n_layers);
    defer b_d.deinit(allocator);
    _ = mlx.mlx_array_free(try a_e.forward(&xfm, &prompt_a));
    _ = mlx.mlx_array_free(try b_e.forward(&xfm, &prompt_b));
    _ = mlx.mlx_array_free(try a_d.forward(&xfm, &prompt_a));
    _ = mlx.mlx_array_free(try b_d.forward(&xfm, &prompt_b));

    const widths = [_]c_int{ 1, 4 };
    const decode_ids = [_]i32{ 13, 6 };
    const verify_ids = [_]i32{ 13, 6, 27, 3, 19, 40, 12, 8 };
    for (widths) |w| {
        const ids: []const i32 = if (w == 1) &decode_ids else &verify_ids;
        const capture = w > 1;
        const eager_slots = [_]*Qwen4TestSlot{ a_e, b_e };
        const defer_slots = [_]*Qwen4TestSlot{ a_d, b_d };
        const eager = try qwen4BatchedPleForward(&xfm, &eager_slots, ids, w, capture, false, true);
        defer _ = mlx.mlx_array_free(eager.logits);
        try testing.expect(!eager.pending);
        const deferred = try qwen4BatchedPleForward(&xfm, &defer_slots, ids, w, capture, true, true);
        defer _ = mlx.mlx_array_free(deferred.logits);
        try testing.expect(deferred.pending);
        const ea = try qwen4ReadF32(allocator, eager.logits, s);
        defer allocator.free(ea);
        const da = try qwen4ReadF32(allocator, deferred.logits, s);
        defer allocator.free(da);
        try testing.expectEqualSlices(f32, ea, da);
        for (a_e.entries, a_d.entries) |*ee, *de| {
            try testing.expectEqual(ee.ple_prev_valid, de.ple_prev_valid);
            try testing.expectEqualSlices(u32, &ee.ple_prev, &de.ple_prev);
            try testing.expectEqual(ee.spec_ple_len, de.spec_ple_len);
            const n: usize = ee.spec_ple_len;
            try testing.expectEqualSlices(u32, ee.spec_ple_tokens[0..n], de.spec_ple_tokens[0..n]);
        }
        for (b_e.entries, b_d.entries) |*ee, *de| {
            try testing.expectEqual(ee.ple_prev_valid, de.ple_prev_valid);
            try testing.expectEqualSlices(u32, &ee.ple_prev, &de.ple_prev);
            try testing.expectEqual(ee.spec_ple_len, de.spec_ple_len);
            const n: usize = ee.spec_ple_len;
            try testing.expectEqualSlices(u32, ee.spec_ple_tokens[0..n], de.spec_ple_tokens[0..n]);
        }
        if (capture) {
            const acc: u32 = 1;
            const vl: u32 = @intCast(w);
            for (a_e.entries, a_d.entries) |*ee, *de| {
                try ssmRollbackFromCapture(ee, acc, vl, s);
                try ssmRollbackFromCapture(de, acc, vl, s);
                try testing.expectEqualSlices(u32, &ee.ple_prev, &de.ple_prev);
            }
            for (b_e.entries, b_d.entries) |*ee, *de| {
                try ssmRollbackFromCapture(ee, acc, vl, s);
                try ssmRollbackFromCapture(de, acc, vl, s);
                try testing.expectEqualSlices(u32, &ee.ple_prev, &de.ple_prev);
            }
            for (a_e.entries) |*e| ssmFreeSpecCapture(e);
            for (b_e.entries) |*e| ssmFreeSpecCapture(e);
            for (a_d.entries) |*e| ssmFreeSpecCapture(e);
            for (b_d.entries) |*e| ssmFreeSpecCapture(e);
        }
    }
}

test "qwen4 fixture: full prefill, chunked prefill + decode past the QSA budget match the HF reference" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("QWEN4_FIXTURE") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    // A `build --yarn` pack carries `rope_type: "yarn"` in its config: the same
    // fixture then measures the context extension end to end (attention, QSA
    // indexer, M-RoPE table, MTP head) at positions past the trained window.
    // Red probe: the YaRN fixture against the no-YaRN pack (same weights)
    // fails at the first check (MTP head cos 0.971 vs the 0.995 bar).
    try testing.expectEqual(config.rope_yarn, xfm.yarnActive());
    // The fixture renders the reference in f32; run our streams in f32 too so
    // the bar measures the math. The shipped bf16 path is pinned by the
    // `qwen4 fixture bf16` test below (streams vs a bf16 reference) — a tiny
    // random model's final logits are chaotic between any two bf16 renderings.
    xfm.qwen4_stream_f32 = true;
    xfm.compileQwen4Hc(); // the served path: fused hyper-connection tails
    xfm.compileGdnGate();
    xfm.compileMoeRouting();

    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    const ref_full = try qwen4ReadF32(allocator, fx.get("logits_full") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_full);
    const ref_last = try qwen4ReadF32(allocator, fx.get("logits_prefill_last") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_last);
    const ref_dec = try qwen4ReadF32(allocator, fx.get("logits_decode") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_dec);
    const ties = try Qwen4Ties.read(allocator, &fx, s);
    defer ties.deinit(allocator);
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    const T: usize = mlx.mlx_array_size(ids_arr);
    const ids_src = mlx.mlx_array_data_int32(ids_arr) orelse return error.Unreadable;
    const v: usize = @intCast(config.vocab_size);
    const t_dec: usize = ref_dec.len / v;
    const t_pre: usize = T - t_dec;

    // [1] one full-prompt forward.
    {
        const shape = [_]c_int{ 1, @intCast(T) };
        const ids = mlx.mlx_array_new_data(@ptrCast(ids_src), &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(ids);
        var ctx = xfm.defaultCtx();
        var stream_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stream_all);
        ctx.capture_stream_all = &stream_all;
        if (fx.get("l1_ngram_ids")) |ref_ids_arr| {
            const st = xfm.qwen4.?;
            var f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f);
            try mlx.check(mlx.mlx_astype(&f, ref_ids_arr, .int32, s));
            try mlx.check(mlx.mlx_array_eval(f));
            const ref_ids = mlx.mlx_array_data_int32(f) orelse return error.Unreadable;
            const nh: usize = st.hash.n_heads;
            const host_ids = try allocator.alloc(u32, T);
            defer allocator.free(host_ids);
            for (0..T) |i| host_ids[i] = @intCast(ids_src[i]);
            const rows = try allocator.alloc(i64, T * nh);
            defer allocator.free(rows);
            var prev: [8]u32 = @splat(st.hash.eos);
            st.hash.rowIds(prev[0 .. st.hash.ngram_size - 1], host_ids, rows);
            for (0..T * nh) |i| try testing.expectEqual(@as(i64, ref_ids[i]), rows[i]);
        }
        const logits = try xfm.forwardWith(&ctx, ids);
        defer _ = mlx.mlx_array_free(logits);
        // MTP head: stream rows 0..T-2 + tokens 1..T-1 at positions 1.. → logits for 2..T.
        if (fx.get("mtp_logits")) |ref_mtp_arr| if (xfm.qwen4_mtp != null) {
            const ref_mtp = try qwen4ReadF32(allocator, ref_mtp_arr, s);
            defer allocator.free(ref_mtp);
            const tm: c_int = @intCast(T - 1);
            var stream_prev = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(stream_prev);
            const st_shape = mlx.getShape(stream_all);
            try mlx.check(mlx.mlx_slice(&stream_prev, stream_all, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, tm, st_shape[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            const tshape = [_]c_int{ 1, tm };
            const next_ids = mlx.mlx_array_new_data(@ptrCast(ids_src + 1), &tshape, 2, .int32);
            defer _ = mlx.mlx_array_free(next_ids);
            try xfm.qwen4MtpReset();
            const mo = try xfm.qwen4MtpForward(stream_prev, next_ids, 1, null, .all_rows);
            defer _ = mlx.mlx_array_free(mo.logits);
            defer _ = mlx.mlx_array_free(mo.stream);
            const ours_m = try qwen4ReadF32(allocator, mo.logits, s);
            defer allocator.free(ours_m);
            // The head's own router ties are acquitted by its own margins.
            const head_ties = Qwen4Ties{
                .qsa_gap = null,
                .route_gap = if (fx.get("mtp_route_gap")) |g| try qwen4ReadF32(allocator, g, s) else null,
                .logit_margin = null,
            };
            defer head_ties.deinit(allocator);
            const rm = qwen4CompareTied("mtp", ours_m, ref_mtp, T - 1, v, 0, head_ties);
            errdefer std.debug.print("[qwen4 fixture] MTP head T-1={d}: min cos {d:.5} ({d} rows acquitted) argmax {d}/{d} decisive rows\n", .{ T - 1, rm.min_cos, rm.tied, rm.agree, rm.decisive });
            try testing.expect(rm.min_cos > 0.995);
            try testing.expect(rm.decisive >= (T - 1) / 2);
        };
        const ours = try qwen4ReadF32(allocator, logits, s);
        defer allocator.free(ours);
        try testing.expectEqual(T * v, ours.len);
        const r = qwen4CompareTied("logits", ours, ref_full, T, v, 0, ties);
        errdefer std.debug.print("[qwen4 fixture] full prefill T={d}: min cos {d:.5} ({d} rows acquitted) argmax {d}/{d} decisive rows\n", .{ T, r.min_cos, r.tied, r.agree, r.decisive });
        try testing.expect(r.min_cos > 0.995);
        // A k < E fixture (`build --topk`) ties in most trunk rows across
        // every MoE layer; its selection coverage is the MTP head's ONE
        // MoE layer (decisive floor above). k = E fixtures keep the T/2 floor.
        try testing.expect(r.decisive >= (if (ties.route_gap != null) @as(usize, 2) else T / 2) and r.agree == r.decisive);
    }

    // [2] fresh state: prefill the first t_pre tokens, then decode one at a
    // time past the indexer budget (the sparse mask + key history path).
    {
        var cache = try KVCache.init(allocator, config.num_hidden_layers);
        defer cache.deinit();
        var off: usize = 0;
        const entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
        defer allocator.free(entries);
        for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer for (entries) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        };
        var ctx: ForwardCtx = .{ .cache = &cache, .moe_seq_offset = &off, .ssm_entries = entries, .capture_hidden = null, .vision_embeddings = null };

        const pshape = [_]c_int{ 1, @intCast(t_pre) };
        const pids = mlx.mlx_array_new_data(@ptrCast(ids_src), &pshape, 2, .int32);
        defer _ = mlx.mlx_array_free(pids);
        const plog = try xfm.forwardWith(&ctx, pids);
        defer _ = mlx.mlx_array_free(plog);
        const pours = try qwen4ReadF32(allocator, plog, s);
        defer allocator.free(pours);
        const rp = qwen4CompareTied("prefill-last", pours[(t_pre - 1) * v ..], ref_last, 1, v, t_pre - 1, ties);
        errdefer std.debug.print("[qwen4 fixture] prefill {d}: last-row cos {d:.5}\n", .{ t_pre, rp.min_cos });
        try testing.expect(rp.min_cos > 0.995);

        var dec: Qwen4TiedCmp = .{ .min_cos = 1.0, .agree = 0, .decisive = 0, .tied = 0 };
        for (0..t_dec) |k| {
            const dshape = [_]c_int{ 1, 1 };
            const did = mlx.mlx_array_new_data(@ptrCast(ids_src + t_pre + k), &dshape, 2, .int32);
            defer _ = mlx.mlx_array_free(did);
            const dlog = try xfm.forwardWith(&ctx, did);
            defer _ = mlx.mlx_array_free(dlog);
            const dours = try qwen4ReadF32(allocator, dlog, s);
            defer allocator.free(dours);
            const rd = qwen4CompareTied("decode", dours, ref_dec[k * v ..][0..v], 1, v, t_pre + k, ties);
            if (rd.min_cos < dec.min_cos) dec.min_cos = rd.min_cos;
            dec.agree += rd.agree;
            dec.decisive += rd.decisive;
            dec.tied += rd.tied;
        }
        errdefer std.debug.print("[qwen4 fixture] decode x{d}: min cos {d:.5} ({d} acquitted) argmax {d}/{d} decisive rows\n", .{ t_dec, dec.min_cos, dec.tied, dec.agree, dec.decisive });
        try testing.expect(dec.min_cos > 0.995);
        try testing.expect(dec.decisive >= (if (ties.route_gap != null) @as(usize, 0) else 2) and dec.agree == dec.decisive);
    }

    // [3] spec-verify rollback: prefill P, a CAPTURING verify of 4 tokens,
    // roll back to 1 accepted (GDN state + PLE window/tokens + QSA key
    // histories), continue 4 tokens — must match a fresh forward.
    {
        const P: usize = 10;
        const VL: usize = 4;
        const ACC: u32 = 1;
        const cont: usize = 4;
        var cache = try KVCache.init(allocator, config.num_hidden_layers);
        defer cache.deinit();
        var off: usize = 0;
        const entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
        defer allocator.free(entries);
        for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer for (entries) |*e| {
            ssmFreeSpecCapture(e);
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        };
        var ctx: ForwardCtx = .{ .cache = &cache, .moe_seq_offset = &off, .ssm_entries = entries, .capture_hidden = null, .vision_embeddings = null };
        const pids = mlx.mlx_array_new_data(@ptrCast(ids_src), &[_]c_int{ 1, @intCast(P) }, 2, .int32);
        defer _ = mlx.mlx_array_free(pids);
        _ = mlx.mlx_array_free(try xfm.forwardWith(&ctx, pids));
        const kv_step_snap = cache.step;
        ctx.capture_ssm_seq = true;
        const vids = mlx.mlx_array_new_data(@ptrCast(ids_src + P), &[_]c_int{ 1, @intCast(VL) }, 2, .int32);
        defer _ = mlx.mlx_array_free(vids);
        _ = mlx.mlx_array_free(try xfm.forwardWith(&ctx, vids));
        ctx.capture_ssm_seq = false;
        try cache.truncate(P + 1 + ACC, s);
        cache.step = kv_step_snap;
        for (entries) |*e| try ssmRollbackFromCapture(e, ACC, @intCast(VL), s);
        for (entries) |*e| ssmFreeSpecCapture(e);
        off = P + 1 + ACC;
        const kept = P + 1 + ACC;
        const cids = mlx.mlx_array_new_data(@ptrCast(ids_src + kept), &[_]c_int{ 1, @intCast(cont) }, 2, .int32);
        defer _ = mlx.mlx_array_free(cids);
        const clog = try xfm.forwardWith(&ctx, cids);
        defer _ = mlx.mlx_array_free(clog);
        const ours_c = try qwen4ReadF32(allocator, clog, s);
        defer allocator.free(ours_c);

        var cache2 = try KVCache.init(allocator, config.num_hidden_layers);
        defer cache2.deinit();
        var off2: usize = 0;
        const entries2 = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
        defer allocator.free(entries2);
        for (entries2) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer for (entries2) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        };
        var ctx2: ForwardCtx = .{ .cache = &cache2, .moe_seq_offset = &off2, .ssm_entries = entries2, .capture_hidden = null, .vision_embeddings = null };
        const fids = mlx.mlx_array_new_data(@ptrCast(ids_src), &[_]c_int{ 1, @intCast(kept + cont) }, 2, .int32);
        defer _ = mlx.mlx_array_free(fids);
        const flog = try xfm.forwardWith(&ctx2, fids);
        defer _ = mlx.mlx_array_free(flog);
        const fresh = try qwen4ReadF32(allocator, flog, s);
        defer allocator.free(fresh);
        const rr = qwen4CompareRows(ours_c, fresh[kept * v ..], cont, v);
        errdefer std.debug.print("[qwen4 fixture] rollback (P={d}, verify {d}, accept {d}, +{d}): min cos {d:.5} argmax {d}/{d}\n", .{ P, VL, ACC, cont, rr.min_cos, rr.argmax_agree, rr.rows });
        try testing.expect(rr.min_cos > 0.999);
        try testing.expect(rr.argmax_agree == rr.rows);
    }

    // [4] cross-round MTP head bookkeeping (mirrors mtpChainBuild / Phase 5a
    // stash / consume-time truncate): prefill history, two rounds of
    // draft → verify(capture) → accept a → trunk rollback → stash, then the
    // third round's merged (stash ++ t1) head forward must match a fresh head
    // over the full committed rows. Head KV + aux keys + pooled blocks all
    // ride qwen4MtpTruncate.
    if (xfm.qwen4_mtp != null) {
        const P: usize = 8;
        const M: usize = 2; // drafts per round
        const accs = [_]u32{ 1, 0 };
        const H = struct {
            fn rows(st: mlx.mlx_stream, arr: mlx.mlx_array, from: usize, to: usize) !mlx.mlx_array {
                const sh = mlx.getShape(arr);
                var out = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(&out, arr, &[_]c_int{ 0, @intCast(from), 0 }, 3, &[_]c_int{ sh[0], @intCast(to), sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, st));
                return out;
            }
            fn cat(st: mlx.mlx_stream, parts: []const mlx.mlx_array) !mlx.mlx_array {
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                for (parts) |p| _ = mlx.mlx_vector_array_append_value(vec, p);
                var out = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, st));
                return out;
            }
            fn ids(vals: []const i32) mlx.mlx_array {
                const sh = [_]c_int{ 1, @intCast(vals.len) };
                return mlx.mlx_array_new_data(@ptrCast(vals.ptr), &sh, 2, .int32);
            }
            fn argmaxLast(alloc: std.mem.Allocator, logits: mlx.mlx_array, st: mlx.mlx_stream, vv: usize) !i32 {
                const f = try qwen4ReadF32(alloc, logits, st);
                defer alloc.free(f);
                const row = f[f.len - vv ..];
                var best: usize = 0;
                for (row, 0..) |x, i| if (x > row[best]) {
                    best = i;
                };
                return @intCast(best);
            }
        };
        // Trunk: prefill P with the pre-mixer stream captured.
        var cache = try KVCache.init(allocator, config.num_hidden_layers);
        defer cache.deinit();
        var off: usize = 0;
        const entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
        defer allocator.free(entries);
        for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer for (entries) |*e| {
            ssmFreeSpecCapture(e);
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        };
        var ctx: ForwardCtx = .{ .cache = &cache, .moe_seq_offset = &off, .ssm_entries = entries, .capture_hidden = null, .vision_embeddings = null };
        var h_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_all);
        ctx.capture_hidden_all = &h_all;
        const pids = H.ids(ids_src[0..P]);
        defer _ = mlx.mlx_array_free(pids);
        _ = mlx.mlx_array_free(try xfm.forwardWith(&ctx, pids));

        // Head history like the prefill chunk loop: rows (h[j], id[j+1]),
        // j < P-1, at position 1 (the last prompt row is round 1's job).
        try xfm.qwen4MtpReset();
        {
            const hp = try H.rows(s, h_all, 0, P - 1);
            defer _ = mlx.mlx_array_free(hp);
            const tp = H.ids(ids_src[1..P]);
            defer _ = mlx.mlx_array_free(tp);
            const o = try xfm.qwen4MtpForward(hp, tp, 1, null, .all_rows);
            _ = mlx.mlx_array_free(o.logits);
            _ = mlx.mlx_array_free(o.stream);
        }
        // Committed rows for the fresh-head replay: hidden + token per row.
        var fresh_h: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (fresh_h.items) |a| _ = mlx.mlx_array_free(a);
            fresh_h.deinit(allocator);
        }
        var fresh_ids: std.ArrayList(i32) = .empty;
        defer fresh_ids.deinit(allocator);
        try fresh_h.append(allocator, try H.rows(s, h_all, 0, P - 1));
        try fresh_ids.appendSlice(allocator, ids_src[1..P]);

        var last_hidden = try H.rows(s, h_all, P - 1, P);
        defer _ = mlx.mlx_array_free(last_hidden);
        var t1: i32 = ids_src[P];
        var next_free: usize = P + 1; // fixture ids handed out as "sampled" tokens
        var stash_ids: std.ArrayList(i32) = .empty;
        defer stash_ids.deinit(allocator);
        var stash_h: ?mlx.mlx_array = null;
        defer if (stash_h) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var stash_off0: usize = 0;
        var off0: usize = xfm.qwen4_mtp.?.seq_offset;
        var last_logits: mlx.mlx_array = .{ .ctx = null };
        defer if (last_logits.ctx != null) {
            _ = mlx.mlx_array_free(last_logits);
        };

        for (accs, 0..) |acc, round| {
            // Chain build (i = 0 merges the pending stash; i > 0 chains the
            // head's own stream).
            var drafts: [M]i32 = undefined;
            var h_chain: ?mlx.mlx_array = null;
            defer if (h_chain) |a| {
                _ = mlx.mlx_array_free(a);
            };
            for (0..M) |i| {
                const prev_tok: i32 = if (i == 0) t1 else drafts[i - 1];
                const o = if (i == 0 and stash_h != null) blk: {
                    try xfm.qwen4MtpTruncate(stash_off0);
                    try stash_ids.append(allocator, prev_tok);
                    const mids = H.ids(stash_ids.items);
                    defer _ = mlx.mlx_array_free(mids);
                    const mh = try H.cat(s, &.{ stash_h.?, last_hidden });
                    defer _ = mlx.mlx_array_free(mh);
                    _ = mlx.mlx_array_free(stash_h.?);
                    stash_h = null;
                    stash_ids.clearRetainingCapacity();
                    break :blk try xfm.qwen4MtpForward(mh, mids, @intCast(stash_off0 + 1), null, .all_rows);
                } else blk: {
                    const one = [_]i32{prev_tok};
                    const tid = H.ids(&one);
                    defer _ = mlx.mlx_array_free(tid);
                    break :blk try xfm.qwen4MtpForward(if (i == 0) last_hidden else h_chain.?, tid, @intCast(off0 + i + 1), null, .all_rows);
                };
                if (h_chain) |a| _ = mlx.mlx_array_free(a);
                h_chain = try H.rows(s, o.stream, @intCast(mlx.getShape(o.stream)[1] - 1), @intCast(mlx.getShape(o.stream)[1]));
                _ = mlx.mlx_array_free(o.stream);
                drafts[i] = try H.argmaxLast(allocator, o.logits, s, v);
                _ = mlx.mlx_array_free(o.logits);
            }
            // Verify [t1, drafts] with capture, roll the trunk back to acc.
            var vids_buf: [1 + M]i32 = undefined;
            vids_buf[0] = t1;
            for (drafts, 0..) |d, i| vids_buf[1 + i] = d;
            const kv_step_snap = cache.step;
            ctx.capture_ssm_seq = true;
            var vh_all = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(vh_all);
            ctx.capture_hidden_all = &vh_all;
            const vids = H.ids(&vids_buf);
            defer _ = mlx.mlx_array_free(vids);
            _ = mlx.mlx_array_free(try xfm.forwardWith(&ctx, vids));
            ctx.capture_ssm_seq = false;
            ctx.capture_hidden_all = null;
            try cache.truncate(kv_step_snap + 1 + acc, s);
            cache.step = kv_step_snap;
            for (entries) |*e| try ssmRollbackFromCapture(e, acc, @intCast(1 + M), s);
            for (entries) |*e| ssmFreeSpecCapture(e);
            off = kv_step_snap + 1 + acc;
            // Phase 5a stash: ids [t1, drafts[0..acc]], hidden [last_hidden, vh[0..acc]].
            try stash_ids.append(allocator, t1);
            for (drafts[0..acc]) |d| try stash_ids.append(allocator, d);
            const vh_acc = try H.rows(s, vh_all, 0, acc);
            defer _ = mlx.mlx_array_free(vh_acc);
            stash_h = if (acc == 0) try H.cat(s, &.{last_hidden}) else try H.cat(s, &.{ last_hidden, vh_acc });
            stash_off0 = off0;
            off0 = stash_off0 + 1 + acc;
            // Fresh replay rows mirror the stash.
            try fresh_h.append(allocator, try H.cat(s, &.{last_hidden}));
            try fresh_ids.append(allocator, t1);
            if (acc > 0) {
                try fresh_h.append(allocator, try H.rows(s, vh_all, 0, acc));
                try fresh_ids.appendSlice(allocator, drafts[0..acc]);
            }
            // Phase 5b: last_hidden = vh[acc], t1 = the "sampled" next token.
            _ = mlx.mlx_array_free(last_hidden);
            last_hidden = try H.rows(s, vh_all, acc, acc + 1);
            t1 = ids_src[next_free];
            next_free += 1;
            _ = round;
        }
        // Round 3's merged forward: stash ++ [t1].
        try xfm.qwen4MtpTruncate(stash_off0);
        try stash_ids.append(allocator, t1);
        const n_merged = stash_ids.items.len;
        {
            const mids = H.ids(stash_ids.items);
            defer _ = mlx.mlx_array_free(mids);
            const mh = try H.cat(s, &.{ stash_h.?, last_hidden });
            defer _ = mlx.mlx_array_free(mh);
            const o = try xfm.qwen4MtpForward(mh, mids, @intCast(stash_off0 + 1), null, .all_rows);
            _ = mlx.mlx_array_free(o.stream);
            last_logits = o.logits;
        }
        const ours_x = try qwen4ReadF32(allocator, last_logits, s);
        defer allocator.free(ours_x);
        try testing.expectEqual(n_merged * v, ours_x.len);
        // Fresh head over every committed row in one forward.
        try fresh_h.append(allocator, try H.cat(s, &.{last_hidden}));
        try fresh_ids.append(allocator, t1);
        try xfm.qwen4MtpReset();
        const fh = try H.cat(s, fresh_h.items);
        defer _ = mlx.mlx_array_free(fh);
        const fids = H.ids(fresh_ids.items);
        defer _ = mlx.mlx_array_free(fids);
        try testing.expectEqual(@as(c_int, @intCast(fresh_ids.items.len)), mlx.getShape(fh)[1]);
        const fo = try xfm.qwen4MtpForward(fh, fids, 1, null, .all_rows);
        defer _ = mlx.mlx_array_free(fo.logits);
        defer _ = mlx.mlx_array_free(fo.stream);
        const fresh_x = try qwen4ReadF32(allocator, fo.logits, s);
        defer allocator.free(fresh_x);
        const n_rows = fresh_ids.items.len;
        const rx = qwen4CompareRows(ours_x, fresh_x[(n_rows - n_merged) * v ..], n_merged, v);
        errdefer std.debug.print("[qwen4 fixture] MTP cross-round (P={d}, rounds {d}, acc {d}/{d}, merged {d} rows of {d}): min cos {d:.5} argmax {d}/{d}\n", .{ P, accs.len, accs[0], accs[1], n_merged, n_rows, rx.min_cos, rx.argmax_agree, rx.rows });
        try testing.expect(rx.min_cos > 0.999);
        try testing.expect(rx.argmax_agree == rx.rows);
    }
}

test "qwen4 MTP head: last-row and no-logits projections match the full block (QWEN4_TEST_MODEL)" {
    // `MtpHeadRef.qwen4Step` consumes ONE row of logits, and the merged
    // history step after a partial accept hands the head `1 + accepted` rows —
    // so the mixer read + the vocab-wide lm_head used to run over the whole
    // block for a single row. `.last_row` cuts the stream first, `.none`
    // (appendHistory) skips both stages.
    //
    // Bars: the STREAM is a pure slice / a pure skip, so it is bit-identical
    // on both. The last row's LOGITS are not asserted bit-identical past S=1 —
    // a quantized projection at M=1 and at M=S are different kernels, and only
    // the S=1 arm shares one — so the multi-row arm takes the fixture bar.
    // That is the right bar here: these logits only steer which DRAFTS the
    // head proposes, and the trunk's verify forward decides every byte that
    // ships, so a last-ulp difference can move acceptance and never output.
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    if (xfm.qwen4_mtp == null) return error.SkipZigTest;
    xfm.compileQwen4Hc();
    xfm.compileMoeRouting();

    const hcH: usize = @as(usize, config.hc_count) * @as(usize, config.hidden_size);
    const dt = mlx.mlx_array_dtype(xfm.qwen4_mtp.?.pre_norm_hidden);

    for ([_]usize{ 1, 5 }) |S| {
        // Distinct rows: identical rows cannot tell a last-row slice from a
        // first-row one.
        const host = try allocator.alloc(f32, S * hcH);
        defer allocator.free(host);
        var rs: u64 = 0x9E3779B97F4A7C15;
        for (host) |*x| {
            rs = rs *% 6364136223846793005 +% 1442695040888963407;
            x.* = (@as(f32, @floatFromInt((rs >> 40) & 0xFFFF)) / 32768.0) - 1.0;
        }
        const hshape = [_]c_int{ 1, @intCast(S), @intCast(hcH) };
        const sp_f32 = mlx.mlx_array_new_data(host.ptr, &hshape, 3, .float32);
        defer _ = mlx.mlx_array_free(sp_f32);
        var stream_prev = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stream_prev);
        try mlx.check(mlx.mlx_astype(&stream_prev, sp_f32, dt, s));

        var id_buf: [5]i32 = undefined;
        for (0..S) |i| id_buf[i] = @intCast(7 + i * 13);
        const ishape = [_]c_int{ 1, @intCast(S) };
        const ids = mlx.mlx_array_new_data(&id_buf, &ishape, 2, .int32);
        defer _ = mlx.mlx_array_free(ids);

        try xfm.qwen4MtpReset();
        const full = try xfm.qwen4MtpForward(stream_prev, ids, 1, null, .all_rows);
        defer _ = mlx.mlx_array_free(full.logits);
        defer _ = mlx.mlx_array_free(full.stream);
        const full_l = try qwen4ReadF32(allocator, full.logits, s);
        defer allocator.free(full_l);
        const full_h = try qwen4ReadF32(allocator, full.stream, s);
        defer allocator.free(full_h);
        try testing.expectEqual(S * hcH, full_h.len);
        const v: usize = full_l.len / S;

        try xfm.qwen4MtpReset();
        const last = try xfm.qwen4MtpForward(stream_prev, ids, 1, null, .last_row);
        defer _ = mlx.mlx_array_free(last.logits);
        defer _ = mlx.mlx_array_free(last.stream);
        const last_l = try qwen4ReadF32(allocator, last.logits, s);
        defer allocator.free(last_l);
        const last_h = try qwen4ReadF32(allocator, last.stream, s);
        defer allocator.free(last_h);
        try testing.expectEqual(v, last_l.len);
        try testing.expectEqualSlices(f32, full_h[(S - 1) * hcH ..], last_h);
        if (S == 1) {
            try testing.expectEqualSlices(f32, full_l, last_l);
        } else {
            const c = qwen4CompareRows(last_l, full_l[(S - 1) * v ..], 1, v);
            errdefer std.debug.print("[qwen4] MTP last-row projection S={d}: cos {d:.6} argmax {d}/1\n", .{ S, c.min_cos, c.argmax_agree });
            try testing.expect(c.min_cos > 0.9999);
            try testing.expect(c.argmax_agree == 1);
        }

        try xfm.qwen4MtpReset();
        const bare = try xfm.qwen4MtpForward(stream_prev, ids, 1, null, .none);
        defer _ = mlx.mlx_array_free(bare.stream);
        try testing.expect(bare.logits.ctx == null);
        const bare_h = try qwen4ReadF32(allocator, bare.stream, s);
        defer allocator.free(bare_h);
        try testing.expectEqualSlices(f32, full_h, bare_h);
    }
}

test "qwen4 MTP head: quantized KV rollback + re-append matches dense draft ids (QWEN4_TEST_MODEL)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    if (xfm.qwen4_mtp == null) return error.SkipZigTest;
    xfm.compileQwen4Hc();
    xfm.compileMoeRouting();

    const saved = Transformer.mtp_head_kv_quant_override;
    defer Transformer.mtp_head_kv_quant_override = saved;
    Transformer.mtp_head_kv_quant_override = true;

    const hcH: usize = @as(usize, config.hc_count) * @as(usize, config.hidden_size);
    const dt = mlx.mlx_array_dtype(xfm.qwen4_mtp.?.pre_norm_hidden);
    const S: usize = 4;
    const keep: usize = 2;
    const host = try allocator.alloc(f32, S * hcH);
    defer allocator.free(host);
    var rs: u64 = 0x9E3779B97F4A7C15;
    for (host) |*x| {
        rs = rs *% 6364136223846793005 +% 1442695040888963407;
        x.* = (@as(f32, @floatFromInt((rs >> 40) & 0xFFFF)) / 32768.0) - 1.0;
    }
    const hshape = [_]c_int{ 1, @intCast(S), @intCast(hcH) };
    const sp_f32 = mlx.mlx_array_new_data(host.ptr, &hshape, 3, .float32);
    defer _ = mlx.mlx_array_free(sp_f32);
    var stream_prev = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stream_prev);
    try mlx.check(mlx.mlx_astype(&stream_prev, sp_f32, dt, s));
    var id_buf: [4]i32 = .{ 7, 20, 33, 46 };
    const ishape = [_]c_int{ 1, @intCast(S) };
    const ids = mlx.mlx_array_new_data(&id_buf, &ishape, 2, .int32);
    defer _ = mlx.mlx_array_free(ids);

    const Run = struct {
        fn lastLogits(xf: *Transformer, st: mlx.mlx_stream, alloc: std.mem.Allocator, stream: mlx.mlx_array, tok: mlx.mlx_array, kkeep: usize) ![]f32 {
            try xf.qwen4MtpReset();
            const full = try xf.qwen4MtpForward(stream, tok, 1, null, .all_rows);
            defer _ = mlx.mlx_array_free(full.logits);
            defer _ = mlx.mlx_array_free(full.stream);
            try xf.qwen4MtpTruncate(kkeep);
            const sh = mlx.getShape(stream);
            var tail = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(tail);
            try mlx.check(mlx.mlx_slice(&tail, stream, &[_]c_int{ 0, @intCast(kkeep), 0 }, 3, &[_]c_int{ sh[0], sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, st));
            const tsh = mlx.getShape(tok);
            var ttail = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ttail);
            try mlx.check(mlx.mlx_slice(&ttail, tok, &[_]c_int{ 0, @intCast(kkeep) }, 2, &[_]c_int{ tsh[0], tsh[1] }, 2, &[_]c_int{ 1, 1 }, 2, st));
            const more = try xf.qwen4MtpForward(tail, ttail, @intCast(kkeep + 1), null, .last_row);
            defer _ = mlx.mlx_array_free(more.logits);
            defer _ = mlx.mlx_array_free(more.stream);
            return qwen4ReadF32(alloc, more.logits, st);
        }
        fn argmaxGap(row: []const f32) struct { id: usize, gap: f32 } {
            var b0: usize = 0;
            var b1: usize = 0;
            for (row, 0..) |x, i| {
                if (x > row[b0]) {
                    b1 = b0;
                    b0 = i;
                } else if (i != b0 and (b1 == b0 or x > row[b1])) {
                    b1 = i;
                }
            }
            return .{ .id = b0, .gap = row[b0] - row[b1] };
        }
    };

    const dense_l = try Run.lastLogits(&xfm, s, allocator, stream_prev, ids, keep);
    defer allocator.free(dense_l);
    try xfm.qwen4MtpApplyKvQuant(kv_quant.KVQuantConfig.affine(8));
    try testing.expectEqual(kv_quant.Scheme.affine, xfm.qwen4_mtp.?.cache.config.scheme);
    try testing.expectEqual(@as(u8, 8), xfm.qwen4_mtp.?.cache.config.bits);
    const quant_l = try Run.lastLogits(&xfm, s, allocator, stream_prev, ids, keep);
    defer allocator.free(quant_l);
    try testing.expectEqual(dense_l.len, quant_l.len);
    const d = Run.argmaxGap(dense_l);
    const q = Run.argmaxGap(quant_l);
    if (d.id != q.id) {
        try testing.expect(d.gap <= 0.15);
    }
}

test "qwen4MtpForward: every error path of the MTP head forward releases its stream (fault-injected, QWEN4_TEST_MODEL)" {
    // The claim under proof: `h` is born with ONE function-scope errdefer, and
    // every later rebind either consumes the old handle (`hcWrite`, which
    // frees its input only past its last `try`) or frees it itself (the
    // `.last_row` slice) — so no error path leaks it and none frees it twice.
    // Argument cannot settle that; a sweep can.
    //
    // Method: fault the k-th checked mlx-c call for every k the forward
    // issues, on every projection, at S > 1 so the slice path is live, and
    // require live Metal bytes back at the pre-call baseline each time. The
    // head's own state (KV, aux, seq_offset) is reset on BOTH sides of the
    // measurement, so only ownership can move the number.
    //
    // What the oracle can and cannot see: MLX is lazy, so a handle leaked
    // before anything downstream of it has been evaluated holds no buffer and
    // is invisible here — the equality is a NECESSARY condition, sharp for
    // every k at or past the first evaluation inside the head (the QSA host
    // reads, the eval cadence, the projection) and weak before it. The static
    // scan above covers the whole function unconditionally, and the DOUBLE
    // free half is unconditional at every k: it aborts this binary rather
    // than failing an assertion.
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    if (xfm.qwen4_mtp == null) return error.SkipZigTest;
    xfm.compileQwen4Hc();
    xfm.compileMoeRouting();

    const S: usize = 5; // > 1: keeps the `.last_row` / `.mixed_last_row` slice live
    const hcH: usize = @as(usize, config.hc_count) * @as(usize, config.hidden_size);
    const dt = mlx.mlx_array_dtype(xfm.qwen4_mtp.?.pre_norm_hidden);
    const host = try allocator.alloc(f32, S * hcH);
    defer allocator.free(host);
    var rs: u64 = 0x243F6A8885A308D3;
    for (host) |*x| {
        rs = rs *% 6364136223846793005 +% 1442695040888963407;
        x.* = (@as(f32, @floatFromInt((rs >> 40) & 0xFFFF)) / 32768.0) - 1.0;
    }
    const hshape = [_]c_int{ 1, @intCast(S), @intCast(hcH) };
    const sp_f32 = mlx.mlx_array_new_data(host.ptr, &hshape, 3, .float32);
    defer _ = mlx.mlx_array_free(sp_f32);
    var stream_prev = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stream_prev);
    try mlx.check(mlx.mlx_astype(&stream_prev, sp_f32, dt, s));
    var id_buf: [S]i32 = undefined;
    for (0..S) |i| id_buf[i] = @intCast(7 + i * 13);
    const ishape = [_]c_int{ 1, @intCast(S) };
    const ids = mlx.mlx_array_new_data(&id_buf, &ishape, 2, .int32);
    defer _ = mlx.mlx_array_free(ids);

    inline for ([_]Transformer.Qwen4MtpProject{ .all_rows, .last_row, .mixed_last_row, .none }) |proj| {
        // Un-faulted warm-up: builds every lazy artefact the forward touches
        // and counts the checked ops it issues. Its outputs are freed HERE, or
        // every later baseline would carry them and the equality would be a
        // statement about nothing.
        try xfm.qwen4MtpReset();
        const c0 = mlx.op_count.load(.monotonic);
        const warm = try xfm.qwen4MtpForward(stream_prev, ids, 1, null, proj);
        const n_ops = mlx.op_count.load(.monotonic) - c0;
        try testing.expect(warm.stream.ctx != null);
        if (warm.logits.ctx != null) _ = mlx.mlx_array_free(warm.logits);
        if (warm.mixed.ctx != null) _ = mlx.mlx_array_free(warm.mixed);
        _ = mlx.mlx_array_free(warm.stream);

        var errs: usize = 0;
        var k: u64 = 1;
        while (k <= n_ops) : (k += 1) {
            try xfm.qwen4MtpReset();
            const base = mlxSettledActiveBytes(s);
            mlx.fault.arm(k);
            const r = xfm.qwen4MtpForward(stream_prev, ids, 1, null, proj);
            mlx.fault.disarm();
            if (r) |o| {
                // Reachable only where a fallback arm below the head swallowed
                // the injected error; the outputs are still ours to free.
                if (o.logits.ctx != null) _ = mlx.mlx_array_free(o.logits);
                if (o.mixed.ctx != null) _ = mlx.mlx_array_free(o.mixed);
                if (o.stream.ctx != null) _ = mlx.mlx_array_free(o.stream);
            } else |_| {
                errs += 1;
            }
            try xfm.qwen4MtpReset();
            try testing.expectEqual(base, mlxSettledActiveBytes(s));
        }
        errdefer std.debug.print("[qwen4] MTP fault sweep {s}: {d} checked ops, {d} error paths, live bytes back at baseline on all\n", .{ @tagName(proj), n_ops, errs });
        try testing.expect(errs >= 1);
    }
    try xfm.qwen4MtpReset();
}

test "row-axis verify: every row of a group verify is byte-identical to its solo verify" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    const n_layers = config.num_hidden_layers;

    var ids: [640]i32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 40000);
    const prompts = [2][]const i32{ ids[0..320], ids[64..232] };
    const indexed_before = trf.mtp_verify_indexed_input_calls;
    const reduce_before = trf.mtp_verify_expert_reduce_calls;
    const expert_before = trf.mtp_verify_expert_pairs_calls;
    const rows_before = trf.mtp_verify_expert_rows_calls;
    const attn_before = qwen4_forward.mtp_verify_attn_rows_calls;
    const lmhead_before = qwen4_forward.mtp_verify_lmhead_rows_calls;
    const gdn_before = qwen4_forward.mtp_verify_gdn_rows_calls;
    const shared_gate_before = qwen4_forward.mtp_verify_shared_gate_rows_calls;
    const grouped_before = qwen4_forward.mtp_verify_moe_group_graphs;

    for ([_][2]usize{ .{ 2, 2 }, .{ 5, 5 }, .{ 2, 3 }, .{ 3, 2 }, .{ 9, 9 }, .{ 2, 9 }, .{ 9, 3 } }) |S_row| {
        var slots = try L10VerifySlots.init(allocator, &xfm, n_layers, prompts);
        defer slots.deinit(allocator);

        var rows: [2]mlx.mlx_array = undefined;
        defer for (rows) |r| {
            _ = mlx.mlx_array_free(r);
        };
        for (0..2) |i| {
            const shape = [_]c_int{ 1, @intCast(S_row[i]) };
            rows[i] = mlx.mlx_array_new_data(@ptrCast(ids[400 + i * 32 ..][0..S_row[i]].ptr), &shape, 2, .int32);
        }

        var solo_logits: [2]mlx.mlx_array = undefined;
        var solo_last: [2]mlx.mlx_array = undefined;
        var solo_all: [2]mlx.mlx_array = undefined;
        defer for (0..2) |i| {
            _ = mlx.mlx_array_free(solo_logits[i]);
            _ = mlx.mlx_array_free(solo_last[i]);
            _ = mlx.mlx_array_free(solo_all[i]);
        };
        for (0..2) |i| {
            const one_row = [_]mlx.mlx_array{rows[i]};
            const one_ctx = [_]*ForwardCtx{&slots.solo[i].ctx};
            try xfm.forwardRowAxisVerify(&one_row, &one_ctx, solo_logits[i .. i + 1], solo_last[i .. i + 1], solo_all[i .. i + 1], null, null);
        }

        var grp_logits: [2]mlx.mlx_array = undefined;
        var grp_last: [2]mlx.mlx_array = undefined;
        var grp_all: [2]mlx.mlx_array = undefined;
        defer for (0..2) |i| {
            _ = mlx.mlx_array_free(grp_logits[i]);
            _ = mlx.mlx_array_free(grp_last[i]);
            _ = mlx.mlx_array_free(grp_all[i]);
        };
        const grp_ctxs = [_]*ForwardCtx{ &slots.group[0].ctx, &slots.group[1].ctx };
        try xfm.forwardRowAxisVerify(&rows, &grp_ctxs, &grp_logits, &grp_last, &grp_all, null, null);

        for (0..2) |i| {
            try expectRowsByteEqual(allocator, s, solo_logits[i], grp_logits[i], "logits");
            try expectRowsByteEqual(allocator, s, solo_all[i], grp_all[i], "hidden_all");
            try expectRowsByteEqual(allocator, s, solo_last[i], grp_last[i], "hidden_last");
            try testing.expectEqual(prompts[i].len + S_row[i], slots.group[i].off);
            try testing.expectEqual(slots.solo[i].off, slots.group[i].off);
        }
    }
    try testing.expectEqual(@as(u64, 7), qwen4_forward.mtp_verify_moe_group_graphs - grouped_before);
    try testing.expectEqual(indexed_before, trf.mtp_verify_indexed_input_calls);
    try testing.expectEqual(reduce_before, trf.mtp_verify_expert_reduce_calls);
    try testing.expectEqual(expert_before, trf.mtp_verify_expert_pairs_calls);
    try testing.expect(trf.mtp_verify_expert_rows_calls > rows_before);
    try testing.expect(qwen4_forward.mtp_verify_gdn_rows_calls > gdn_before);
    try testing.expect(qwen4_forward.mtp_verify_shared_gate_rows_calls > shared_gate_before);
    try testing.expect(qwen4_forward.mtp_verify_attn_rows_calls > attn_before);
    try testing.expect(qwen4_forward.mtp_verify_lmhead_rows_calls > lmhead_before);
    try testing.expect(qwen4_forward.mtp_verify_moe_group_last_ops > 0);
}

test "per-row rollback: different accept counts across rows restore exactly the solo states" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    const n_layers = config.num_hidden_layers;

    var ids: [640]i32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 40000);
    const prompts = [2][]const i32{ ids[0..320], ids[64..232] };

    const S: usize = 5;
    const accepts = [2]u32{ 0, 3 };
    var slots = try L10VerifySlots.init(allocator, &xfm, n_layers, prompts);
    defer slots.deinit(allocator);

    var rows: [2]mlx.mlx_array = undefined;
    defer for (rows) |r| {
        _ = mlx.mlx_array_free(r);
    };
    for (0..2) |i| {
        const shape = [_]c_int{ 1, @intCast(S) };
        rows[i] = mlx.mlx_array_new_data(@ptrCast(ids[400 + i * 32 ..][0..S].ptr), &shape, 2, .int32);
    }

    var off_snap: [2]usize = undefined;
    var step_snap: [2]usize = undefined;
    for (0..2) |i| {
        off_snap[i] = slots.solo[i].off;
        step_snap[i] = slots.solo[i].cache.step;
        try testing.expectEqual(off_snap[i], slots.group[i].off);
        try testing.expectEqual(step_snap[i], slots.group[i].cache.step);
    }

    var solo_logits: [2]mlx.mlx_array = undefined;
    var solo_last: [2]mlx.mlx_array = undefined;
    var solo_all: [2]mlx.mlx_array = undefined;
    defer for (0..2) |i| {
        _ = mlx.mlx_array_free(solo_logits[i]);
        _ = mlx.mlx_array_free(solo_last[i]);
        _ = mlx.mlx_array_free(solo_all[i]);
    };
    for (0..2) |i| {
        const one_row = [_]mlx.mlx_array{rows[i]};
        const one_ctx = [_]*ForwardCtx{&slots.solo[i].ctx};
        try xfm.forwardRowAxisVerify(&one_row, &one_ctx, solo_logits[i .. i + 1], solo_last[i .. i + 1], solo_all[i .. i + 1], null, null);
    }

    var grp_logits: [2]mlx.mlx_array = undefined;
    var grp_last: [2]mlx.mlx_array = undefined;
    var grp_all: [2]mlx.mlx_array = undefined;
    defer for (0..2) |i| {
        _ = mlx.mlx_array_free(grp_logits[i]);
        _ = mlx.mlx_array_free(grp_last[i]);
        _ = mlx.mlx_array_free(grp_all[i]);
    };
    const grp_ctxs = [_]*ForwardCtx{ &slots.group[0].ctx, &slots.group[1].ctx };
    try xfm.forwardRowAxisVerify(&rows, &grp_ctxs, &grp_logits, &grp_last, &grp_all, null, null);

    for (0..2) |i| {
        const a_start = [_]c_int{ 0, @intCast(accepts[i]), 0 };
        const a_stop = [_]c_int{ 1, @as(c_int, @intCast(accepts[i])) + 1, mlx.getShape(solo_all[i])[2] };
        const stride = [_]c_int{ 1, 1, 1 };
        var h_solo = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_solo);
        var h_grp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_grp);
        try mlx.check(mlx.mlx_slice(&h_solo, solo_all[i], &a_start, 3, &a_stop, 3, &stride, 3, s));
        try mlx.check(mlx.mlx_slice(&h_grp, grp_all[i], &a_start, 3, &a_stop, 3, &stride, 3, s));
        try expectRowsByteEqual(allocator, s, h_solo, h_grp, "next-round hidden");
    }

    for (0..2) |i| {
        try rollbackRowToAccepted(slots.solo[i], accepts[i], @intCast(S), off_snap[i], step_snap[i], s);
        try rollbackRowToAccepted(slots.group[i], accepts[i], @intCast(S), off_snap[i], step_snap[i], s);
        try testing.expectEqual(off_snap[i] + 1 + accepts[i], slots.group[i].off);
        try expectSlotStateEqual(allocator, s, slots.solo[i], slots.group[i], i);
    }

    for (0..2) |i| {
        const next = [_]i32{ids[500 + i]};
        const la = try slots.solo[i].forward(&xfm, &next);
        defer _ = mlx.mlx_array_free(la);
        const lb = try slots.group[i].forward(&xfm, &next);
        defer _ = mlx.mlx_array_free(lb);
        try expectRowsByteEqual(allocator, s, la, lb, "next-token logits");
    }
}

test "deferred PLE at N rows: a failed row abandons every row's claim and leaves the slots usable" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    const n_layers = config.num_hidden_layers;

    var ids: [640]i32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 40000);
    const prompts = [2][]const i32{ ids[0..192], ids[64..200] };
    const S: usize = 4;

    var slots = try L10VerifySlots.init(allocator, &xfm, n_layers, prompts);
    defer slots.deinit(allocator);
    var rows: [2]mlx.mlx_array = undefined;
    defer for (rows) |r| {
        _ = mlx.mlx_array_free(r);
    };
    for (0..2) |i| {
        const shape = [_]c_int{ 1, @intCast(S) };
        rows[i] = mlx.mlx_array_new_data(@ptrCast(ids[400 + i * 32 ..][0..S].ptr), &shape, 2, .int32);
    }
    const ctxs = [_]*ForwardCtx{ &slots.group[0].ctx, &slots.group[1].ctx };

    var logits: [2]mlx.mlx_array = undefined;
    var last: [2]mlx.mlx_array = undefined;
    var all: [2]mlx.mlx_array = undefined;
    const c0 = mlx.op_count.load(.monotonic);
    {
        const one_row = [_]mlx.mlx_array{rows[0]};
        const one_ctx = [_]*ForwardCtx{&slots.solo[0].ctx};
        try xfm.forwardRowAxisVerify(&one_row, &one_ctx, logits[0..1], last[0..1], all[0..1], null, null);
    }
    const n_ops = mlx.op_count.load(.monotonic) - c0;
    for (0..1) |i| {
        _ = mlx.mlx_array_free(logits[i]);
        _ = mlx.mlx_array_free(last[i]);
        _ = mlx.mlx_array_free(all[i]);
    }
    try testing.expect(n_ops > 8);

    // A round that never commits must not move any row's n-gram history.
    const ple_layer = blk: {
        for (xfm.moe_layers.?, 0..) |*lw, l| {
            if (lw.ple != null) break :blk l;
        }
        return error.SkipZigTest;
    };
    var prev_before: [2][qwen4_mod.MAX_NGRAM_SIZE]u32 = undefined;
    var prev_valid_before: [2]bool = undefined;
    for (ctxs, 0..) |c, i| {
        prev_before[i] = c.ssm_entries.?[ple_layer].ple_prev;
        prev_valid_before[i] = c.ssm_entries.?[ple_layer].ple_prev_valid;
    }

    var offsets_before: [2]usize = undefined;
    var cache_steps_before: [2]usize = undefined;
    var kv_offsets_before: [2][]usize = undefined;
    for (ctxs, 0..) |ctx, row| {
        offsets_before[row] = ctx.moe_seq_offset.*;
        cache_steps_before[row] = ctx.cache.step;
        kv_offsets_before[row] = try allocator.alloc(usize, ctx.cache.entries.len);
        for (ctx.cache.entries, 0..) |entry, layer| kv_offsets_before[row][layer] = entry.offset;
    }
    defer for (kv_offsets_before) |offsets| allocator.free(offsets);

    if (xfm.qwen4VerifyRowsEligible(&rows, &ctxs)) {
        qwen4_forward.verify_fault_after_ple_row = 1;
    } else mlx.fault.arm(n_ops + 8);
    defer qwen4_forward.verify_fault_after_ple_row = null;
    const r = xfm.forwardRowAxisVerify(&rows, &ctxs, &logits, &last, &all, null, null);
    mlx.fault.disarm();
    qwen4_forward.verify_fault_after_ple_row = null;
    try testing.expect(mlx.fault.didFire());
    try testing.expectError(error.MlxError, r);

    for (ctxs, 0..) |c, row| {
        errdefer std.debug.print("[mtp failure state] row={d} before={d} after={d}\n", .{ row, offsets_before[row], c.moe_seq_offset.* });
        try testing.expectEqual(offsets_before[row], c.moe_seq_offset.*);
        try testing.expectEqual(cache_steps_before[row], c.cache.step);
        for (c.cache.entries, 0..) |entry, layer| try testing.expectEqual(kv_offsets_before[row][layer], entry.offset);
        try testing.expect(c.ple_pending == null);
        for (c.ssm_entries.?) |*e| {
            try testing.expectEqual(@as(u32, 0), e.spec_ple_len);
            try testing.expect(e.spec_state_seq.ctx == null);
            try testing.expect(e.spec_conv_input.ctx == null);
            try testing.expect(e.spec_ple_input.ctx == null);
        }
    }
    for (ctxs, 0..) |c, i| {
        const e = &c.ssm_entries.?[ple_layer];
        try testing.expectEqualSlices(u32, &prev_before[i], &e.ple_prev);
        try testing.expectEqual(prev_valid_before[i], e.ple_prev_valid);
    }

    const sync_before = kv_cache.deferred_ple_group_syncs;
    kv_cache.verify_fail_after_flush = 1;
    const flushed_failure = xfm.forwardRowAxisVerify(&rows, &ctxs, &logits, &last, &all, null, null);
    kv_cache.verify_fail_after_flush = null;
    try testing.expectError(error.InjectedGroupFlushFailure, flushed_failure);
    try testing.expectEqual(@as(u64, 1), kv_cache.deferred_ple_group_syncs - sync_before);
    for (ctxs, 0..) |ctx, row| {
        try testing.expectEqual(offsets_before[row], ctx.moe_seq_offset.*);
        for (ctx.cache.entries, 0..) |entry, layer| try testing.expectEqual(kv_offsets_before[row][layer], entry.offset);
        const entry = &ctx.ssm_entries.?[ple_layer];
        try testing.expectEqualSlices(u32, &prev_before[row], &entry.ple_prev);
        try testing.expectEqual(prev_valid_before[row], entry.ple_prev_valid);
        try testing.expect(ctx.ple_pending == null);
    }

    try xfm.forwardRowAxisVerify(&rows, &ctxs, &logits, &last, &all, null, null);
    defer for (0..2) |i| {
        _ = mlx.mlx_array_free(logits[i]);
        _ = mlx.mlx_array_free(last[i]);
        _ = mlx.mlx_array_free(all[i]);
    };
    for (0..2) |i| {
        const h = try qwen4ReadF32(allocator, logits[i], s);
        defer allocator.free(h);
        for (h) |x| try testing.expect(std.math.isFinite(x));
    }
    for (0..2) |i| {
        const reference = try Qwen4TestSlot.init(allocator, n_layers);
        defer reference.deinit(allocator);
        _ = mlx.mlx_array_free(try reference.forward(&xfm, prompts[i]));
        var ref_logits: [1]mlx.mlx_array = undefined;
        var ref_last: [1]mlx.mlx_array = undefined;
        var ref_all: [1]mlx.mlx_array = undefined;
        try xfm.forwardRowAxisVerify(rows[i..][0..1], &.{&reference.ctx}, &ref_logits, &ref_last, &ref_all, null, null);
        defer {
            _ = mlx.mlx_array_free(ref_logits[0]);
            _ = mlx.mlx_array_free(ref_last[0]);
            _ = mlx.mlx_array_free(ref_all[0]);
        }
        try testing.expect(try qsaArraysAllEqual(ref_logits[0], logits[i], s));
        try testing.expect(try qsaArraysAllEqual(ref_all[0], all[i], s));
    }
}

test "group round: a row's own argmax and budget decide it, and nothing else does" {
    const gen_mod = @import("../generate.zig");
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    const n_layers = config.num_hidden_layers;

    var ids: [640]i32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 40000);
    const prompts = [2][]const i32{ ids[0..288], ids[64..200] };
    const S_row = [2]usize{ 5, 3 };

    var slots = try L10VerifySlots.init(allocator, &xfm, n_layers, prompts);
    defer slots.deinit(allocator);
    var rows: [2]mlx.mlx_array = undefined;
    defer for (rows) |r| {
        _ = mlx.mlx_array_free(r);
    };
    var drafts: [2][8]u32 = undefined;
    for (0..2) |i| {
        const shape = [_]c_int{ 1, @intCast(S_row[i]) };
        const row_ids = ids[400 + i * 32 ..][0..S_row[i]];
        rows[i] = mlx.mlx_array_new_data(@ptrCast(row_ids.ptr), &shape, 2, .int32);
        for (row_ids[1..], 0..) |t, k| drafts[i][k] = @intCast(t);
    }

    var so: [2]mlx.mlx_array = undefined;
    var sl: [2]mlx.mlx_array = undefined;
    var sa: [2]mlx.mlx_array = undefined;
    var go: [2]mlx.mlx_array = undefined;
    var gl: [2]mlx.mlx_array = undefined;
    var ga: [2]mlx.mlx_array = undefined;
    defer for (0..2) |i| {
        _ = mlx.mlx_array_free(so[i]);
        _ = mlx.mlx_array_free(sl[i]);
        _ = mlx.mlx_array_free(sa[i]);
        _ = mlx.mlx_array_free(go[i]);
        _ = mlx.mlx_array_free(gl[i]);
        _ = mlx.mlx_array_free(ga[i]);
    };
    for (0..2) |i| {
        const one_row = [_]mlx.mlx_array{rows[i]};
        const one_ctx = [_]*ForwardCtx{&slots.solo[i].ctx};
        try xfm.forwardRowAxisVerify(&one_row, &one_ctx, so[i .. i + 1], sl[i .. i + 1], sa[i .. i + 1], null, null);
    }
    const gctxs = [_]*ForwardCtx{ &slots.group[0].ctx, &slots.group[1].ctx };
    try xfm.forwardRowAxisVerify(&rows, &gctxs, &go, &gl, &ga, null, null);

    var solo_am: [2][]i32 = undefined;
    var grp_am: [2][]i32 = undefined;
    defer for (0..2) |i| {
        allocator.free(solo_am[i]);
        allocator.free(grp_am[i]);
    };
    for ([_]*[2]mlx.mlx_array{ &so, &go }, [_]*[2][]i32{ &solo_am, &grp_am }) |logits, out| {
        for (0..2) |i| {
            var am = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(am);
            try mlx.check(mlx.mlx_argmax_axis(&am, logits[i], 2, false, s));
            try mlx.check(mlx.mlx_array_eval(am));
            const n = mlx.mlx_array_size(am);
            const src = mlx.mlx_array_data_int32(am) orelse return error.Unreadable;
            const buf = try allocator.alloc(i32, n);
            @memcpy(buf, src[0..n]);
            out[i] = buf;
        }
    }
    for (0..2) |i| try testing.expectEqualSlices(i32, solo_am[i], grp_am[i]);

    const accept = gen_mod.Generator.mtpGreedyVerdict;
    var base: [2]gen_mod.Generator.MtpRowVerdict = undefined;
    for (0..2) |i| {
        const solo_v = accept(solo_am[i], drafts[i][0 .. S_row[i] - 1], 0, 400);
        const grp_v = accept(grp_am[i], drafts[i][0 .. S_row[i] - 1], 0, 400);
        try testing.expectEqual(solo_v.accepted, grp_v.accepted);
        try testing.expectEqual(solo_v.next_pending, grp_v.next_pending);
        base[i] = grp_v;
    }

    const clipped = accept(grp_am[0], drafts[0][0 .. S_row[0] - 1], 399, 400);
    try testing.expectEqual(@as(u32, 0), clipped.accepted);
    try testing.expectEqual(@as(u32, @intCast(grp_am[0][0])), clipped.next_pending);
    try testing.expectEqual(@as(u32, 1), gen_mod.Generator.mtpRetained(clipped.accepted));
    const row1_again = accept(grp_am[1], drafts[1][0 .. S_row[1] - 1], 0, 400);
    try testing.expectEqual(base[1].accepted, row1_again.accepted);
    try testing.expectEqual(base[1].next_pending, row1_again.next_pending);

    const retained = [2]u32{ gen_mod.Generator.mtpRetained(base[0].accepted), gen_mod.Generator.mtpRetained(base[1].accepted) };
    for (0..2) |i| try testing.expect(retained[i] >= 1 and retained[i] <= S_row[i]);
}

test "N=4 rollback: rejects, partial accepts and full accept restore solo states" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    const n_layers = config.num_hidden_layers;

    var ids: [640]i32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 40000);
    const prompts = [4][]const i32{ ids[0..320], ids[64..232], ids[16..128], ids[128..384] };

    const S: usize = 5;
    const accepts = [4]u32{ 0, 3, 1, 4 };
    var left = try L10VerifySlots.init(allocator, &xfm, n_layers, prompts[0..2].*);
    defer left.deinit(allocator);
    var right = try L10VerifySlots.init(allocator, &xfm, n_layers, prompts[2..4].*);
    defer right.deinit(allocator);
    const slots = .{
        .solo = [_]*Qwen4TestSlot{ left.solo[0], left.solo[1], right.solo[0], right.solo[1] },
        .group = [_]*Qwen4TestSlot{ left.group[0], left.group[1], right.group[0], right.group[1] },
    };

    var rows: [4]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (rows) |r| {
        _ = mlx.mlx_array_free(r);
    };
    for (0..4) |i| {
        const shape = [_]c_int{ 1, @intCast(S) };
        rows[i] = mlx.mlx_array_new_data(@ptrCast(ids[400 + i * 32 ..][0..S].ptr), &shape, 2, .int32);
    }

    var off_snap: [4]usize = undefined;
    var step_snap: [4]usize = undefined;
    for (0..4) |i| {
        off_snap[i] = slots.solo[i].off;
        step_snap[i] = slots.solo[i].cache.step;
        try testing.expectEqual(off_snap[i], slots.group[i].off);
        try testing.expectEqual(step_snap[i], slots.group[i].cache.step);
    }

    var solo_logits: [4]mlx.mlx_array = @splat(.{ .ctx = null });
    var solo_last: [4]mlx.mlx_array = @splat(.{ .ctx = null });
    var solo_all: [4]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (0..4) |i| {
        _ = mlx.mlx_array_free(solo_logits[i]);
        _ = mlx.mlx_array_free(solo_last[i]);
        _ = mlx.mlx_array_free(solo_all[i]);
    };
    for (0..4) |i| {
        const one_row = [_]mlx.mlx_array{rows[i]};
        const one_ctx = [_]*ForwardCtx{&slots.solo[i].ctx};
        try xfm.forwardRowAxisVerify(&one_row, &one_ctx, solo_logits[i .. i + 1], solo_last[i .. i + 1], solo_all[i .. i + 1], null, null);
    }

    var grp_logits: [4]mlx.mlx_array = @splat(.{ .ctx = null });
    var grp_last: [4]mlx.mlx_array = @splat(.{ .ctx = null });
    var grp_all: [4]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (0..4) |i| {
        _ = mlx.mlx_array_free(grp_logits[i]);
        _ = mlx.mlx_array_free(grp_last[i]);
        _ = mlx.mlx_array_free(grp_all[i]);
    };
    var grp_ctxs: [4]*ForwardCtx = undefined;
    for (0..4) |i| grp_ctxs[i] = &slots.group[i].ctx;
    try xfm.forwardRowAxisVerify(&rows, &grp_ctxs, &grp_logits, &grp_last, &grp_all, null, null);

    for (0..4) |i| {
        const a_start = [_]c_int{ 0, @intCast(accepts[i]), 0 };
        const a_stop = [_]c_int{ 1, @as(c_int, @intCast(accepts[i])) + 1, mlx.getShape(solo_all[i])[2] };
        const stride = [_]c_int{ 1, 1, 1 };
        var h_solo = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_solo);
        var h_grp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_grp);
        try mlx.check(mlx.mlx_slice(&h_solo, solo_all[i], &a_start, 3, &a_stop, 3, &stride, 3, s));
        try mlx.check(mlx.mlx_slice(&h_grp, grp_all[i], &a_start, 3, &a_stop, 3, &stride, 3, s));
        try expectRowsByteEqual(allocator, s, h_solo, h_grp, "next-round hidden");
    }

    for (0..4) |i| {
        try rollbackRowToAccepted(slots.solo[i], accepts[i], @intCast(S), off_snap[i], step_snap[i], s);
        try rollbackRowToAccepted(slots.group[i], accepts[i], @intCast(S), off_snap[i], step_snap[i], s);
        try testing.expectEqual(off_snap[i] + 1 + accepts[i], slots.group[i].off);
        try expectSlotStateEqual(allocator, s, slots.solo[i], slots.group[i], i);
    }

    for (0..4) |i| {
        const next = [_]i32{ids[500 + i]};
        const la = try slots.solo[i].forward(&xfm, &next);
        defer _ = mlx.mlx_array_free(la);
        const lb = try slots.group[i].forward(&xfm, &next);
        defer _ = mlx.mlx_array_free(lb);
        try expectRowsByteEqual(allocator, s, la, lb, "next-token logits");
    }
}

test "single-stream verify optimizations preserve native outputs and captures (QWEN4_TEST_MODEL)" {
    const path = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const old_features = trf.single_verify_test_features;
    defer trf.single_verify_test_features = old_features;
    var config = try model_mod.parseConfig(io, a, std.mem.span(path));
    defer if (config.ngram_table_path) |name| a.free(name);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(path));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    const prefix_len = if (std.c.getenv("SINGLE_MTP_TEST_CONTEXT")) |value| try std.fmt.parseInt(usize, std.mem.span(value), 10) else 4101;
    const ids = try a.alloc(i32, prefix_len + 64);
    defer a.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(1 + (i * 7919) % 40000);
    var slots: [2]*Qwen4TestSlot = undefined;
    var initialized: usize = 0;
    defer for (slots[0..initialized]) |slot| slot.deinit(a);
    const prefill_routes = trf.mtp_verify_kernel_calls[2];
    const prefill_hc = qwen4_forward.mtp_verify_hc_prepared_calls;
    for (&slots, 0..) |*target, arm| {
        trf.single_verify_test_features = @splat(arm == 1);
        target.* = try Qwen4TestSlot.init(a, config.num_hidden_layers);
        initialized += 1;
        target.*.cache.config = KVQuantConfig.affine(8);
        target.*.ctx.skip_lm_head = true;
        var start: usize = 0;
        while (start < prefix_len) {
            const end = @min(start + 4096, prefix_len);
            const output = try target.*.forward(&xfm, ids[start..end]);
            try mlx.check(mlx.mlx_array_eval(output));
            _ = mlx.mlx_array_free(output);
            start = end;
        }
        target.*.ctx.skip_lm_head = false;
    }
    try testing.expectEqual(prefill_routes, trf.mtp_verify_kernel_calls[2]);
    try testing.expectEqual(prefill_hc, qwen4_forward.mtp_verify_hc_prepared_calls);
    var position = prefix_len;
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8 }) |width| {
        const input = mlx.mlx_array_new_data(@ptrCast(ids[position..][0..width].ptr), &.{ 1, @as(c_int, @intCast(width)) }, 2, .int32);
        defer _ = mlx.mlx_array_free(input);
        trf.single_verify_test_features = @splat(false);
        var reference = try singleVerifyTestRound(&xfm, slots[0], input);
        defer reference.deinit();
        const before = [_]u64{ trf.mtp_verify_kernel_calls[2], trf.mtp_verify_indexed_input_calls, trf.mtp_verify_expert_pairs_calls, trf.mtp_verify_expert_reduce_calls, qwen4_forward.mtp_verify_hc_prepared_calls };
        trf.single_verify_test_features = @splat(true);
        var candidate = try singleVerifyTestRound(&xfm, slots[1], input);
        defer candidate.deinit();
        const after = [_]u64{ trf.mtp_verify_kernel_calls[2], trf.mtp_verify_indexed_input_calls, trf.mtp_verify_expert_pairs_calls, trf.mtp_verify_expert_reduce_calls, qwen4_forward.mtp_verify_hc_prepared_calls };
        inline for (.{ "logits", "last", "all" }) |field| try expectRowsByteEqual(a, xfm.s, @field(reference, field), @field(candidate, field), "single verify " ++ field);
        for (slots[0].entries, slots[1].entries) |*left, *right| {
            inline for (.{ "conv_state", "ssm_state", "aux_state", "qsa_pooled", "spec_conv_input", "spec_ple_input" }) |field| {
                const x = @field(left.*, field);
                const y = @field(right.*, field);
                if (x.ctx == null or y.ctx == null) {
                    try testing.expect(x.ctx == null and y.ctx == null);
                } else try expectRowsByteEqual(a, xfm.s, x, y, "single verify " ++ field);
            }
            if (ssmHasSpecCapture(left)) {
                const x = try logicalSsmCapture(xfm.s, left);
                defer _ = mlx.mlx_array_free(x);
                const y = try logicalSsmCapture(xfm.s, right);
                defer _ = mlx.mlx_array_free(y);
                try expectRowsByteEqual(a, xfm.s, x, y, "single verify logical capture");
            } else try testing.expect(!ssmHasSpecCapture(right));
            try testing.expectEqual(left.ple_prev_valid, right.ple_prev_valid);
            try testing.expectEqualSlices(u32, &left.ple_prev, &right.ple_prev);
            ssmFreeSpecCapture(left);
            ssmFreeSpecCapture(right);
        }
        errdefer std.debug.print("[single-verify-exact] context={d} width={d} before={any} after={any}\n", .{ prefix_len, width, before, after });
        for (before, after, 0..) |old, new, feature| {
            if (feature == 4 and verifySharedHardware() and width >= 2 and width <= 8) {
                try testing.expect(new > old);
            } else try testing.expectEqual(old, new);
        }
        position += width;
    }
}

test "qwen4 verify first-row layer identity (QWEN4_TEST_MODEL)" {
    try qwen4LayerIdentityCase(true, false, 2, 0, 0);
}

test "qwen4 verify unflushed stream and recurrent state identity (QWEN4_TEST_MODEL)" {
    try qwen4LayerIdentityCase(false, true, 2, 0, 0);
}

test "qwen4 verify history row and cache identity (QWEN4_TEST_MODEL)" {
    try qwen4LayerIdentityCase(false, true, 8, 24, 1);
}

test "qwen4 dense rows evaluated verification replay (QWEN4_VERIFY_BENCH)" {
    const cases_path = std.c.getenv("QWEN4_VERIFY_BENCH") orelse return error.SkipZigTest;
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    if (diagEnvOn("QWEN4_PROFILE_FWD")) return error.ProfilerChangesReplay;
    const dense = @import("../mtp_dense_rows.zig");
    const old = dense.override;
    defer dense.override = old;
    dense.override = false;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();
    const body = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(cases_path), a, .limited(8 << 20));
    defer a.free(body);
    const cases = try std.json.parseFromSlice(std.json.Value, a, body, .{});
    defer cases.deinit();
    var config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |path| a.free(path);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    for (cases.value.array.items) |case| {
        const name = case.object.get("name").?.string;
        const selected_width = if (case.object.get("width")) |v| v.integer else 0;
        const repetitions: usize = if (case.object.get("reps")) |v| @intCast(std.math.clamp(v.integer, 1, 64)) else 8;
        const reverse = if (case.object.get("reverse")) |v| v.bool else false;
        const ids_json = case.object.get("ids").?.array.items;
        const tokens = try a.alloc(i32, ids_json.len);
        defer a.free(tokens);
        for (tokens, ids_json) |*out, item| out.* = @intCast(item.integer);
        try testing.expect(tokens.len > 8);
        const prefix = tokens.len - 8;
        const seed = try Qwen4TestSlot.init(a, config.num_hidden_layers);
        defer seed.deinit(a);
        try seed.cache.reinit(config.num_hidden_layers, KVQuantConfig.affine(8));
        seed.ctx.skip_lm_head = true;
        dense.override = false;
        var at: usize = 0;
        while (at < prefix) {
            const end = @min(prefix, at + 2048);
            const out = try seed.forward(&xfm, tokens[at..end]);
            defer _ = mlx.mlx_array_free(out);
            try Transformer.evalCadencePoint(out, seed.entries);
            at = end;
            _ = mlx.mlx_clear_cache();
        }
        var kv = try seed.cache.snapshot();
        defer kv.deinit();
        const snaps = try a.alloc(SSMCacheEntrySnapshot, seed.entries.len);
        defer a.free(snaps);
        for (snaps, seed.entries) |*snap, *entry| snap.* = ssmSnapshot(entry);
        defer for (snaps) |*snap| ssmSnapshotDeinit(snap);
        for ([_]usize{ 2, 3, 4, 5, 8 }) |width| {
            if (selected_width != 0 and selected_width != width) continue;
            for (0..repetitions + 1) |rep| {
                var results: [2]DenseVerifyReplay = undefined;
                var slots: [2]?*Qwen4TestSlot = .{ null, null };
                var evaluated: [2]bool = .{ false, false };
                defer for (0..2) |arm| {
                    if (evaluated[arm]) results[arm].deinit();
                    if (slots[arm]) |slot| slot.deinit(a);
                };
                for (0..2) |order| {
                    const arm = if ((rep % 2 == 0) != reverse) order else 1 - order;
                    const slot = try Qwen4TestSlot.init(a, config.num_hidden_layers);
                    slots[arm] = slot;
                    try slot.cache.reinit(config.num_hidden_layers, kv.config);
                    try slot.cache.restore(&kv);
                    // Remove restore-only KV copies from the evaluated-forward interval.
                    for (slot.cache.entries) |*entry| {
                        if (!entry.initialized) continue;
                        try testing.expect(mlx.getShape(entry.keys)[2] >= prefix + width);
                        inline for (.{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" }) |field| {
                            const old_arr = @field(entry, field);
                            if (old_arr.ctx != null) {
                                const own = try materializedOwnedCopy(s, old_arr);
                                _ = mlx.mlx_array_free(old_arr);
                                @field(entry, field) = own;
                            }
                        }
                        entry.shared_view = false;
                    }
                    for (slot.entries, snaps) |*entry, *snap| try ssmRestore(entry, snap);
                    slot.off = prefix;
                    try mlx.check(mlx.mlx_synchronize(s));
                    dense.override = arm == 1;
                    const before = dense.calls;
                    results[arm] = try denseVerifyReplay(&xfm, slot, tokens[prefix..][0..width], io);
                    evaluated[arm] = true;
                    if (arm == 1) {
                        try testing.expect(dense.calls[0] >= before[0] + config.num_hidden_layers);
                        try testing.expect(dense.calls[1] >= before[1] + config.num_hidden_layers);
                    }
                }
                try qkvExpectBitsEqual(s, results[0].logits, results[1].logits);
                try qkvExpectBitsEqual(s, results[0].last, results[1].last);
                try qkvExpectBitsEqual(s, results[0].all, results[1].all);
                for (slots[0].?.entries, slots[1].?.entries) |left, right| {
                    inline for (.{ "conv_state", "ssm_state", "spec_conv_input", "spec_ple_input", "aux_state", "qsa_pooled" }) |field| {
                        const l = @field(left, field);
                        const r = @field(right, field);
                        try testing.expectEqual(l.ctx == null, r.ctx == null);
                        if (l.ctx != null) try qkvExpectBitsEqual(s, l, r);
                    }
                    try testing.expectEqual(ssmHasSpecCapture(&left), ssmHasSpecCapture(&right));
                    if (ssmHasSpecCapture(&left)) {
                        const l = try logicalSsmCapture(s, &left);
                        defer _ = mlx.mlx_array_free(l);
                        const r = try logicalSsmCapture(s, &right);
                        defer _ = mlx.mlx_array_free(r);
                        try qkvExpectBitsEqual(s, l, r);
                    }
                    try testing.expectEqual(left.spec_ple_len, right.spec_ple_len);
                    try testing.expectEqualSlices(u32, &left.spec_ple_tokens, &right.spec_ple_tokens);
                }
                for (slots) |maybe_slot| {
                    const slot = maybe_slot.?;
                    try slot.cache.truncate(prefix + 1, s);
                    for (slot.entries) |*entry| try ssmRollbackFromCapture(entry, 0, @intCast(width), s);
                }
                for (slots[0].?.entries, slots[1].?.entries) |left, right| {
                    inline for (.{ "conv_state", "ssm_state", "aux_state", "qsa_pooled" }) |field| {
                        const l = @field(left, field);
                        const r = @field(right, field);
                        try testing.expectEqual(l.ctx == null, r.ctx == null);
                        if (l.ctx != null) try qkvExpectBitsEqual(s, l, r);
                    }
                    try testing.expectEqual(left.ple_prev_valid, right.ple_prev_valid);
                    try testing.expectEqualSlices(u32, &left.ple_prev, &right.ple_prev);
                    inline for (.{ "qsa_hist_rows", "qsa_key_rows", "qsa_pooled_blocks" }) |field|
                        try testing.expectEqual(@field(left, field), @field(right, field));
                }
                if (rep > 0) std.debug.print("[dense-verify] case={s} prefix={d} S={d} rep={d} baseline_ns={d} candidate_ns={d} bits=equal\n", .{ name, prefix, width, rep, results[0].ns, results[1].ns });
            }
        }
    }
}

test "qwen4 checkpoint affine8 and dense projections are row-identical (QWEN4_TEST_MODEL)" {
    const path = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();
    const config = try model_mod.parseConfig(io, a, std.mem.span(path));
    defer if (config.ngram_table_path) |p| a.free(p);
    if (config.quant_bits != 8 or config.quant_mode != .affine) return error.SkipZigTest;
    var weights = try model_mod.loadWeights(io, a, std.mem.span(path));
    defer weights.deinit();
    var checked: usize = 0;
    var it = weights.map.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const quantized = std.mem.endsWith(u8, name, ".scales");
        const router = std.mem.endsWith(u8, name, ".mlp.gate.weight") or std.mem.endsWith(u8, name, ".mlp.shared_expert_gate.weight");
        if (!quantized and !router) continue;
        const sc = if (quantized) entry.value_ptr.* else mlx.mlx_array{};
        if (quantized and mlx.getShape(sc).len != 2) continue;
        const stem = name[0 .. name.len - (if (quantized) @as(usize, ".scales".len) else ".weight".len)];
        const wn = try std.fmt.allocPrint(a, "{s}.weight", .{stem});
        defer a.free(wn);
        const bn = try std.fmt.allocPrint(a, "{s}.biases", .{stem});
        defer a.free(bn);
        const raw = weights.get(wn) orelse return error.MissingProjectionWeight;
        if (!quantized and mlx.mlx_array_dtype(raw) == .uint32) continue;
        const bi = weights.get(bn) orelse if (quantized) return error.MissingProjectionBiases else mlx.mlx_array{};
        const w = if (quantized) try standinRef(raw) else try transposeBf16Weight(raw, s);
        defer _ = mlx.mlx_array_free(w);
        const ws = mlx.getShape(w);
        try testing.expectEqual(@as(usize, 2), ws.len);
        const k = if (quantized) ws[1] * 4 else ws[0];
        const qp = if (quantized) affineParamsFromGeometry(w, sc, @intCast(k)) orelse return error.InvalidAffine8Geometry else QuantParams{ .bits = 0, .group_size = 64, .mode = .affine };
        for ([_]c_int{ 2, 9 }) |rows| {
            const x = try qkvIdentityArray(s, 1, rows, k, 0.039 * @as(f32, @floatFromInt(rows)), mlx.mlx_array_dtype(if (quantized) sc else w));
            defer _ = mlx.mlx_array_free(x);
            const together = try qmatmulBits(x, w, sc, bi, qp.bits, qp.group_size, .affine, s);
            defer _ = mlx.mlx_array_free(together);
            for (0..@intCast(rows)) |row| {
                const begin: c_int = @intCast(row);
                const xr = try sliceAttentionSeq(s, x, begin, begin + 1);
                defer _ = mlx.mlx_array_free(xr);
                const single = try qmatmulBits(xr, w, sc, bi, qp.bits, qp.group_size, .affine, s);
                defer _ = mlx.mlx_array_free(single);
                const candidate = try sliceAttentionSeq(s, together, begin, begin + 1);
                defer _ = mlx.mlx_array_free(candidate);
                qkvExpectBitsEqual(s, single, candidate) catch |err| {
                    std.debug.print("projection {s} gs={d} rows={d} row={d}\n", .{ stem, qp.group_size, rows, row });
                    return err;
                };
            }
        }
        checked += 1;
    }
    try testing.expect(checked > 0);
}
