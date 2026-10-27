//! Shared MTP (multi-token prediction) machinery: draft depth bounds, per-chip
//! depth caps, the cost-profile classifiers the adaptive planner reads, and the
//! draft-rerank scheme (coarse low-bit readout, exact top-32 re-score through
//! the TARGET's lm_head).
//!
//! The heads themselves live with their models: qwen4_exp's in-checkpoint head
//! in `qwen4_forward.zig`, MiMo's in `mimo_mtp.zig`, DeepSeek's DSpark stages in
//! the dsv41 modules. `Generator` (src/generate.zig) drives them through
//! `MtpHeadRef`.

const std = @import("std");
const mlx = @import("mlx.zig");
const mrope = @import("mrope.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");
const log = @import("log.zig");
const Transformer = transformer_mod.Transformer;
const expert_quant = @import("expert_quant.zig");

/// Default draft depth (tokens drafted per round). Flipped 1 -> 3 after the
/// round-v2 rebuild made rejected drafts ~free (scalar-anchor rollback + the
/// 3-bit draft-only lm_head replaced the old full trunk re-forward): the old
/// cost model was what made depth 1 optimal. 2026-07-12 validation matrix on
/// Qwen3.6-27B 4-bit (M4 Max, adaptive controller active, decode tok/s
/// depth-3 vs depth-1): code 54.3 vs 43.3 (+25%), coding-agent ladder 2K
/// 52.1 vs 41.4 (+26%) and 16K 43.9 vs 38.1 (+15%), 2-turn pi agentic
/// (weighted) 48.6 vs 40.9 (+19%), creative temp-0.8 39.1 vs 37.9 (+3% —
/// the class that REGRESSED under the old cost model now holds even at ~30%
/// per-draft acceptance because the controller demotes without churn).
/// Users can cap rounds with `--mtp-max-depth`; the Generator's adaptive
/// controller demotes/promotes within [1, configured].
pub const DEFAULT_DEPTH: u32 = 3;
pub const MAX_DEPTH: u32 = 8;

/// `--max-mtp-ctx`: MTP stays off past this many prompt tokens (inclusive; 0 = unlimited).
/// A machine limit: `enable_mtp:true` in the body does not lift it. MTP only.
pub fn mtpCtxWithinLimit(max: u32, ctx_tokens: usize) bool {
    if (max == 0) return true;
    return ctx_tokens <= @as(usize, max);
}

/// Per-silicon adaptive depth cap for machines on the `.generic` cost
/// surface. The cap is a MACHINE measurement, so each row is one, never
/// interpolated between chips; an unmeasured chip keeps the default row.
///   M1 Pro: 4 (2026-08-20, Qwen3.8-27B iQ-3.8bpw, forced-depth sweep:
///     13.01 tok/s at depth 4 vs 10.78/9.63 at 5/6 — the verify width 6
///     cliff; auto at cap 6 measured 10.64, barely over --no-mtp's 10.57).
/// `chip` is sysctl machdep.cpu.brand_string via `chip.chipBrand`
/// (the GPU arch string cannot tell Ultra from Max); "" lands on default.
/// The row carries its own LABEL so the resolve site can say which one it
/// applied: a bare depth=4 in the spec-stats line is indistinguishable from
/// the EV controller having picked 4 on its own, or from `--mtp-max-depth 4`.
/// `measured` marks a row a HUMAN swept as realized throughput. Those beat the
/// boot probe's cost ladder, which cannot see acceptance or the extension sync
/// — see `generate.mtpDepthCapResolved`.
pub const DepthCap = struct { cap: u32, label: []const u8, measured: bool = false };

pub fn adaptiveDepthCapForMachine(chip: []const u8, default_cap: u32) DepthCap {
    if (std.mem.indexOf(u8, chip, "M1 Pro") != null) return .{ .cap = 4, .label = "m1-pro", .measured = true };
    // Base M4 (2026-08-22, tester report, Qwen3.5-9B-MTPLX 6-bit, 16 GB):
    // saturated echo, median of 5, decode tok/s by forced depth —
    //   3: 47.44   4: 55.50   5: 47.50   6: 47.28
    // Depth 4 is the only width where the cap BINDS (m_lo == cap), so the
    // plan collapses to one chunk and pays no extension sync; from 5 on every
    // round pays it to buy ~1 more accepted token (4.00 -> 4.96/round) and
    // that trade is a 17% net LOSS there. Novel content sits at depth ~2
    // whatever the cap is, so capping at 4 costs nothing outside the echo
    // regime. The boot probe picks 6 on this chip — a cost ladder cannot see
    // either half of that sentence.
    if (std.mem.indexOf(u8, chip, "M4") != null and
        std.mem.indexOf(u8, chip, "M4 Pro") == null and
        std.mem.indexOf(u8, chip, "M4 Max") == null and
        std.mem.indexOf(u8, chip, "M4 Ultra") == null) return .{ .cap = 4, .label = "m4-base", .measured = true };
    if (std.mem.indexOf(u8, chip, "M4 Max") != null) return .{ .cap = default_cap, .label = "m4-max", .measured = true };
    // Base M5 only — the Pro/Max/Ultra dies are their own (unmeasured) rows.
    if (std.mem.indexOf(u8, chip, "M5") != null and
        std.mem.indexOf(u8, chip, "M5 Pro") == null and
        std.mem.indexOf(u8, chip, "M5 Max") == null and
        std.mem.indexOf(u8, chip, "M5 Ultra") == null) return .{ .cap = 4, .label = "m5", .measured = true };
    return .{ .cap = default_cap, .label = "default" };
}

/// Exact full-round cost surfaces known to the adaptive MTP controller.
/// Selection is based on runtime tensor geometry, never a model/repository
/// name. `generic` retains the conservative M1-M4 surface and auto cap.
pub const MtpCostProfile = enum {
    generic,
    g17_nax_q8_gs32,
    g17_nax_q4_gs32,
    g17_nax_q4_gs64,
    g17_nax_q6_gs64,
    g17_nax_q8_gs64,
    g17_nax_oq4e_q4_gs64,
    g17_nax_qwen4_q4_gs64,
    g17_nax_qwen4_mixed_4_8_gs64,
    /// MiMo's three heads over the MCG K2.5 EXL3 pack: a verify row is ~45% of a forward.
    mimo_exl3,
};

/// Target-side tensors that contribute materially to a complete MTP round.
/// This is deliberately a single classification rather than independent
/// booleans: one bound target can match at most one measured cost surface.
pub const MtpNaxTargetSurface = enum {
    none,
    uniform_quantized_embedding,
    uniform_bf16_embedding,
    uniform_q6_quantized_embedding,
    uniform_q8_bf16_embedding,
    oqe_quantized_embedding,
};

/// Pure first-stage classifier for a complete head/target fingerprint.
/// Runtime tensor-shape validation must still pass before the returned
/// calibrated profile is used.
pub fn m5NaxCostProfileForFingerprint(
    bits: u32,
    group_size: u32,
    target_surface: MtpNaxTargetSurface,
) MtpCostProfile {
    return switch (target_surface) {
        .uniform_quantized_embedding => if (group_size == 32) switch (bits) {
            8 => .g17_nax_q8_gs32,
            4 => .g17_nax_q4_gs32,
            else => .generic,
        } else .generic,
        .uniform_bf16_embedding => if (bits == 4 and group_size == 64)
            .g17_nax_q4_gs64
        else
            .generic,
        .uniform_q6_quantized_embedding => if (bits == 6 and group_size == 64)
            .g17_nax_q6_gs64
        else
            .generic,
        .uniform_q8_bf16_embedding => if (bits == 8 and group_size == 64)
            .g17_nax_q8_gs64
        else
            .generic,
        .oqe_quantized_embedding => if (bits == 4 and group_size == 64)
            .g17_nax_oq4e_q4_gs64
        else
            .generic,
        .none => .generic,
    };
}

/// Pure first-stage classifier for the qwen4_exp in-checkpoint head. The
/// trunk is MoE, so the fingerprint path above can never match it;
/// this is its own measured surface, not a widening of the others.
pub fn qwen4G17CostProfileForFingerprint(
    trunk_affine: bool,
    trunk_bits: u32,
    trunk_group_size: u32,
    head_packs_match: bool,
    nax_lane_live: bool,
    expert_layout: expert_quant.Layout,
) MtpCostProfile {
    if (expert_layout != .quantized_split) return .generic;
    if (!trunk_affine or trunk_bits != 4 or trunk_group_size != 64) return .generic;
    if (!head_packs_match) return .generic;
    if (!nax_lane_live) return .generic;
    return .g17_nax_qwen4_q4_gs64;
}

const Qwen4MixedGeometry = struct {
    hidden: u32 = 2560,
    layers: u32 = 48,
    hc: u32 = 4,
    experts: u32 = 512,
    top_k: u32 = 10,
    expert_width: u32 = 640,
    vocab: u32 = 248320,
    ple_layer: i32 = 1,
    kv_bits: u32 = 8,
    kv_group: u32 = 64,
    head_kv_bits: u32 = 8,
    head_kv_group: u32 = 64,
};

fn qwen4MixedCostProfileForFingerprint(geometry: Qwen4MixedGeometry, packs_match: bool, rerank_ready: bool, nax_live: bool) MtpCostProfile {
    if (!std.meta.eql(geometry, Qwen4MixedGeometry{}) or !packs_match or !rerank_ready or !nax_live) return .generic;
    return .g17_nax_qwen4_mixed_4_8_gs64;
}

fn qwen4MixedPackMatches(w: mlx.mlx_array, s: mlx.mlx_array, b: mlx.mlx_array, logical: []const c_int, bits: u32) bool {
    if (logical.len < 2 or logical.len > 3 or (bits != 4 and bits != 8)) return false;
    if (w.ctx == null or s.ctx == null or b.ctx == null) return false;
    if (mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(s) != .bfloat16 or mlx.mlx_array_dtype(b) != .bfloat16) return false;
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(s);
    const bs = mlx.getShape(b);
    if (ws.len != logical.len or ss.len != logical.len or !std.mem.eql(c_int, ss, bs)) return false;
    const last = logical.len - 1;
    for (logical[0..last], ws[0..last], ss[0..last]) |want, wc, sc| {
        if (want <= 0 or wc != want or sc != want) return false;
    }
    const k = logical[last];
    if (k <= 0 or @mod(k, 64) != 0) return false;
    return @as(i64, ws[last]) * 32 == @as(i64, k) * bits and ss[last] == @divExact(k, 64);
}

test "Qwen4 mixed pack matcher pins actual quantization geometry" {
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    for ([_]u32{ 4, 8 }) |bits| for ([_]bool{ false, true }) |stacked| {
        const words: c_int = @intCast(bits * 2);
        const logical: []const c_int = if (stacked) &.{ 2, 16, 64 } else &.{ 16, 64 };
        const ws: []const c_int = if (stacked) &.{ 2, 16, words } else &.{ 16, words };
        const ss: []const c_int = if (stacked) &.{ 2, 16, 1 } else &.{ 16, 1 };
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var scales = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scales);
        var biases = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(biases);
        try mlx.check(mlx.mlx_zeros(&w, ws.ptr, @intCast(ws.len), .uint32, stream));
        try mlx.check(mlx.mlx_zeros(&scales, ss.ptr, @intCast(ss.len), .bfloat16, stream));
        try mlx.check(mlx.mlx_zeros(&biases, ss.ptr, @intCast(ss.len), .bfloat16, stream));
        try testing.expect(qwen4MixedPackMatches(w, scales, biases, logical, bits));
        try testing.expect(!qwen4MixedPackMatches(w, scales, biases, logical, if (bits == 4) 8 else 4));
        try testing.expect(!qwen4MixedPackMatches(w, scales, .{}, logical, bits));
        try testing.expect(!qwen4MixedPackMatches(w, scales, biases, &.{ 16, 128 }, bits));
        var wrong_dtype = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wrong_dtype);
        try mlx.check(mlx.mlx_zeros(&wrong_dtype, ss.ptr, @intCast(ss.len), .float32, stream));
        try testing.expect(!qwen4MixedPackMatches(w, wrong_dtype, biases, logical, bits));
        try testing.expect(!qwen4MixedPackMatches(w, scales, wrong_dtype, logical, bits));
    };
}

test "Qwen4 mixed MTP profile requires its measured geometry and packs" {
    try testing.expectEqual(MtpCostProfile.g17_nax_qwen4_mixed_4_8_gs64, qwen4MixedCostProfileForFingerprint(.{}, true, true, true));
    inline for (.{ "hidden", "layers", "hc", "experts", "top_k", "expert_width", "vocab", "ple_layer", "kv_bits", "kv_group", "head_kv_bits", "head_kv_group" }) |field| {
        var geometry: Qwen4MixedGeometry = .{};
        @field(geometry, field) += 1;
        try testing.expectEqual(MtpCostProfile.generic, qwen4MixedCostProfileForFingerprint(geometry, true, true, true));
    }
    for ([_]bool{ false, true }) |packs| for ([_]bool{ false, true }) |rerank| for ([_]bool{ false, true }) |nax| {
        if (packs and rerank and nax) continue;
        try testing.expectEqual(MtpCostProfile.generic, qwen4MixedCostProfileForFingerprint(.{}, packs, rerank, nax));
    };
}

test "Qwen4 mixed MTP profile validates the live checkpoint (QWEN4_TEST_MODEL)" {
    const path = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or !transformer_mod.verifyQmmNaxAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(path));
    defer if (config.ngram_table_path) |name| a.free(name);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(path));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    const old_kv_flag = Transformer.mtp_head_kv_quant_flag;
    defer Transformer.mtp_head_kv_quant_flag = old_kv_flag;
    Transformer.mtp_head_kv_quant_flag = true;
    try xfm.cache.reinit(config.num_hidden_layers, transformer_mod.KVQuantConfig.affine(8));
    try xfm.qwen4MtpApplyKvQuant(transformer_mod.KVQuantConfig.affine(8));
    try testing.expect(xfm.qwen4BuildDraftRerank());
    try testing.expectEqual(MtpCostProfile.g17_nax_qwen4_mixed_4_8_gs64, qwen4G17CostProfile(&xfm));
    try testing.expectEqual(MtpCostProfile.generic, qwen4G17CostProfileForKv(&xfm, transformer_mod.KVQuantConfig.dense));
    try testing.expectEqual(MtpCostProfile.generic, qwen4G17CostProfileForKv(&xfm, transformer_mod.KVQuantConfig.affine(4)));
    const old_bias = xfm.qwen4_mtp.?.fc_emb_b;
    defer xfm.qwen4_mtp.?.fc_emb_b = old_bias;
    xfm.qwen4_mtp.?.fc_emb_b = .{};
    try testing.expectEqual(MtpCostProfile.generic, qwen4G17CostProfile(&xfm));
}

var qwen4_g17_profile_logged: bool = false;
var qwen4_mixed_profile_logged: bool = false;

fn qwen4MixedHcMatches(hc: anytype) bool {
    return qwen4MixedPackMatches(hc.down_w, hc.down_s, hc.down_b, &.{ 320, 10240 }, 8) and
        qwen4MixedPackMatches(hc.up_w, hc.up_s, hc.up_b, &.{ 10240, 320 }, 8);
}

fn qwen4MixedLayerMatches(layer: anytype) bool {
    const mw = switch (layer.mlp) {
        .moe => |value| value,
        else => return false,
    };
    if (!qwen4MixedPackMatches(mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, &.{ 512, 640, 2560 }, 4) or
        !qwen4MixedPackMatches(mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, &.{ 512, 640, 2560 }, 4) or
        !qwen4MixedPackMatches(mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, &.{ 512, 2560, 640 }, 4) or
        !qwen4MixedPackMatches(mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b, &.{ 640, 2560 }, 8) or
        !qwen4MixedPackMatches(mw.shared_up_w, mw.shared_up_s, mw.shared_up_b, &.{ 640, 2560 }, 8) or
        !qwen4MixedPackMatches(mw.shared_down_w, mw.shared_down_s, mw.shared_down_b, &.{ 2560, 640 }, 8) or
        !qwen4MixedPackMatches(mw.router_w, mw.router_s, mw.router_b, &.{ 512, 2560 }, 8)) return false;
    if (!qwen4MixedHcMatches(layer.hc_attn orelse return false) or !qwen4MixedHcMatches(layer.hc_mlp orelse return false)) return false;
    if (layer.ple) |ple| {
        if (!qwen4MixedPackMatches(ple.key_w, ple.key_s, ple.key_b, &.{ 10240, 2560 }, 8) or
            !qwen4MixedPackMatches(ple.value_w, ple.value_s, ple.value_b, &.{ 2560, 2560 }, 8)) return false;
    }
    return switch (layer.attn) {
        .full => |f| qwen4MixedPackMatches(f.q_w, f.q_s, f.q_b, &.{ 12288, 2560 }, 8) and
            qwen4MixedPackMatches(f.k_w, f.k_s, f.k_b, &.{ 512, 2560 }, 8) and
            qwen4MixedPackMatches(f.v_w, f.v_s, f.v_b, &.{ 512, 2560 }, 8) and
            qwen4MixedPackMatches(f.o_w, f.o_s, f.o_b, &.{ 2560, 6144 }, 8) and
            qwen4MixedPackMatches(f.idx_qk_w, f.idx_qk_s, f.idx_qk_b, &.{ 640, 2560 }, 8),
        .linear => |l| qwen4MixedPackMatches(l.qkv_w, l.qkv_s, l.qkv_b, &.{ 10240, 2560 }, 8) and
            qwen4MixedPackMatches(l.z_w, l.z_s, l.z_b, &.{ 6144, 2560 }, 8) and
            qwen4MixedPackMatches(l.a_w, l.a_s, l.a_b, &.{ 48, 2560 }, 8) and
            qwen4MixedPackMatches(l.b_w, l.b_s, l.b_b, &.{ 48, 2560 }, 8) and
            qwen4MixedPackMatches(l.out_w, l.out_s, l.out_b, &.{ 2560, 6144 }, 8),
    };
}

fn qwen4MixedModelPacksMatch(target: *const Transformer) bool {
    if (target.qwen4 == null or target.qwen4_stream_f32 or target.config.quant_mode != .affine or target.config.quant_group_size != 64) return false;
    const cfg = &target.config;
    if (cfg.linear_key_head_dim != 128 or cfg.linear_value_head_dim != 128 or cfg.linear_num_key_heads != 16 or
        cfg.linear_num_value_heads != 48 or cfg.linear_conv_kernel_dim != 4 or cfg.ple_embed_dim != 2560) return false;
    const head = target.qwen4_mtp orelse return false;
    if (!qwen4MixedPackMatches(target.emb_w, target.emb_s, target.emb_b, &.{ 248320, 2560 }, 4) or
        !qwen4MixedPackMatches(target.lm_head_w, target.lm_head_s, target.lm_head_b, &.{ 248320, 2560 }, 8) or
        !qwen4MixedPackMatches(head.fc_emb_w, head.fc_emb_s, head.fc_emb_b, &.{ 2560, 2560 }, 8) or
        !qwen4MixedPackMatches(head.fc_hid_w, head.fc_hid_s, head.fc_hid_b, &.{ 2560, 2560 }, 8)) return false;
    if (!qwen4MixedHcMatches(target.qwen4_mixer orelse return false) or !qwen4MixedHcMatches(head.mixer) or
        head.layer.is_linear or !qwen4MixedLayerMatches(&head.layer)) return false;
    const layers = target.moe_layers orelse return false;
    if (layers.len != 48) return false;
    for (layers, 0..) |*layer, i| {
        if (layer.is_linear != (i % 4 != 3) or (layer.ple != null) != (i == 1) or !qwen4MixedLayerMatches(layer)) return false;
    }
    return true;
}

/// Live resolver for the qwen4_exp head: validates the runtime fingerprint
/// the G17 surface was measured on — uniform affine-4/gs-64 config and both
/// head fc packs at that width (K derived from the scales, so no input-dim
/// formula is assumed). Revoked without matrix units like the
/// other profiles: auto depth never assumes a lane the hardware lacks.
pub fn qwen4G17CostProfile(target: *const Transformer) MtpCostProfile {
    return qwen4G17CostProfileForKv(target, target.cache.config);
}

/// Mixed-pack prices use the request's actual KV format, including a slot
/// override. The original all-4-bit classifier retains its existing scope.
pub fn qwen4G17CostProfileForKv(target: *const Transformer, kv: transformer_mod.KVQuantConfig) MtpCostProfile {
    if (target.config.expert_layout == .exl3_k4) return .generic;
    const head = if (target.qwen4_mtp) |*h| h else return .generic;
    const cfg = &target.config;
    const geometry = Qwen4MixedGeometry{
        .hidden = cfg.hidden_size,
        .layers = cfg.num_hidden_layers,
        .hc = cfg.hc_count,
        .experts = cfg.num_experts,
        .top_k = cfg.num_experts_per_tok,
        .expert_width = cfg.moe_intermediate_size,
        .vocab = cfg.vocab_size,
        .ple_layer = cfg.ple_layer_idx,
        .kv_bits = if (kv.scheme == .affine) kv.bits else 0,
        .kv_group = kv.group_size,
        .head_kv_bits = if (head.cache.config.scheme == .affine) head.cache.config.bits else 0,
        .head_kv_group = head.cache.config.group_size,
    };
    const rerank_ready = if (head.rerank) |r| r.bits == 3 and r.group_size == 64 and r.rows == 248320 else false;
    const mixed = qwen4MixedCostProfileForFingerprint(geometry, std.meta.eql(geometry, Qwen4MixedGeometry{}) and qwen4MixedModelPacksMatch(target), rerank_ready, mlx.streamIsGpu(target.s) and transformer_mod.verifyQmmNaxAvailable());
    if (mixed != .generic) {
        if (!qwen4_mixed_profile_logged) {
            qwen4_mixed_profile_logged = true;
            log.info("[mtp] cost profile g17-nax-qwen4-mixed-4-8-gs64 engaged (affine8 target/head KV)\n", .{});
        }
        return mixed;
    }
    if (cfg.hidden_size == 0 or cfg.hidden_size > std.math.maxInt(c_int)) return .generic;
    const hidden: c_int = @intCast(cfg.hidden_size);
    const packs = qwen4PackTripleMatches(head.fc_emb_w, head.fc_emb_s, head.fc_emb_b, hidden) and
        qwen4PackTripleMatches(head.fc_hid_w, head.fc_hid_s, head.fc_hid_b, hidden);
    const profile = qwen4G17CostProfileForFingerprint(
        cfg.quant_mode == .affine,
        cfg.quant_bits,
        cfg.quant_group_size,
        packs,
        transformer_mod.verifyQmmNaxAvailable(),
        cfg.expert_layout,
    );
    if (profile != .generic and !qwen4_g17_profile_logged) {
        qwen4_g17_profile_logged = true;
        log.info("[mtp] cost profile g17-nax-qwen4-q4-gs64 engaged (SUSHI_FORCE_GPU_FAMILY_FALLBACK=1 restores generic)\n", .{});
    }
    return profile;
}

/// One packed projection of the qwen4 head: uint32 codes [N, K*4/32] with
/// bf16 scales and biases [N, K/64], N == hidden. K comes from the scales.
fn qwen4PackTripleMatches(w: mlx.mlx_array, s: mlx.mlx_array, b: mlx.mlx_array, hidden: c_int) bool {
    if (w.ctx == null or s.ctx == null or b.ctx == null) return false;
    if (mlx.mlx_array_dtype(w) != .uint32) return false;
    if (mlx.mlx_array_dtype(s) != .bfloat16 or mlx.mlx_array_dtype(b) != .bfloat16) return false;
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(s);
    const bs = mlx.getShape(b);
    if (ws.len != 2 or ss.len != 2 or bs.len != 2) return false;
    if (ws[0] != hidden or ss[0] != hidden or bs[0] != hidden) return false;
    if (ss[1] <= 0 or ss[1] != bs[1]) return false;
    const k = @as(u64, @intCast(ss[1])) * 64;
    return @as(u64, @intCast(ws[1])) * 32 == k * 4;
}

/// One linear: quantized (w packed u32, s/b bf16) when `s.ctx != null`,
/// otherwise a pre-transposed bf16 weight `[in, out]` for plain matmul.
pub const QLinear = struct {
    w: mlx.mlx_array,
    s: mlx.mlx_array,
    b: mlx.mlx_array,

    pub fn deinit(self: *QLinear) void {
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.s);
        _ = mlx.mlx_array_free(self.b);
    }
};

/// Coarse-head width AT BUILD TIME. The default is `draftHeadBitsFromEnv` (3),
/// so SUSHI_MTP_DRAFT_HEAD_BITS retunes it and "0"/"off" disables the build
/// entirely. 2 is what the scheme was first measured at and is the obvious
/// second A/B arm; a coarser head is fewer bytes and a longer tail of
/// shortlist misses, and a miss costs acceptance, never output.
///
/// Read exactly ONCE, where a head is built. The width a head is READ at is a
/// property of its packed weights (`RerankCoarse.bits`), never of the
/// environment at draft time — packed geometry implies K, so a re-read that
/// disagrees with the build reads a different vocab entirely.
pub fn rerankCoarseBits() u32 {
    const p = std.c.getenv("SUSHI_MTP_DRAFT_HEAD_BITS");
    return draftHeadBitsFrom(if (p) |v| std.mem.span(v) else null, 3);
}

/// SUSHI_MTP_DRAFT_HEAD_BITS as read against a head family's unset default.
pub fn draftHeadBitsFrom(raw: ?[]const u8, default: u32) u32 {
    const v = std.fmt.parseInt(u32, raw orelse return default, 10) catch return 0;
    return switch (v) {
        2, 3, 4, 6, 8 => v,
        else => 0,
    };
}

/// Group size every coarse head is packed at.
pub const RERANK_COARSE_GROUP: u32 = 64;

/// A BUILT coarse readout head: the packed weights plus the geometry they
/// were packed with. Both live together because `mlx_quantized_matmul` needs
/// all three and only the build knows them.
pub const RerankCoarse = struct {
    q: QLinear,
    bits: u32,
    group_size: u32,
    rows: c_int,

    pub fn deinit(self: *RerankCoarse) void {
        self.q.deinit();
    }
};

/// The coarse readout head: a low-bit/gs64 requantized copy of the target's
/// lm_head, materialized so the first draft does not pay for it. Infallible —
/// every refusal returns null and the caller keeps its full-readout path.
pub fn buildRerankCoarse(s: mlx.mlx_stream, target: *Transformer, bits: u32) ?RerankCoarse {
    if (bits == 0) return null;
    if (target.lm_head_s.ctx == null) return null; // dense bf16 - row-gather rerank unbuilt/unmeasured
    const w_shape = mlx.getShape(target.lm_head_w);
    if (w_shape.len != 2 or w_shape[0] < TOP32_MIN_ROWS) return null;
    const head_qp = headQuantParams(&target.config, target.lm_head_w, target.lm_head_s);
    if (bits >= head_qp.bits) return null; // no byte saving over the head we already read

    var rc = requantizeRows(
        s,
        target.lm_head_w,
        target.lm_head_s,
        target.lm_head_b,
        head_qp.group_size,
        head_qp.bits,
        head_qp.mode.cstr(),
        RERANK_COARSE_GROUP,
        bits,
        32768,
    ) catch |err| {
        log.warn("[mtp] draft rerank coarse build failed ({s}) - keeping the full-readout path\n", .{@errorName(err)});
        return null;
    };
    const eval_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(eval_vec);
    _ = mlx.mlx_vector_array_append_value(eval_vec, rc.w);
    _ = mlx.mlx_vector_array_append_value(eval_vec, rc.s);
    _ = mlx.mlx_vector_array_append_value(eval_vec, rc.b);
    mlx.check(mlx.mlx_eval(eval_vec)) catch {
        rc.deinit();
        log.warn("[mtp] draft rerank coarse eval failed - keeping the full-readout path\n", .{});
        return null;
    };
    return .{ .q = rc, .bits = bits, .group_size = RERANK_COARSE_GROUP, .rows = w_shape[0] };
}

/// Resident bytes of a coarse head over `rows` x `hidden` at `bits`/gs64:
/// packed weights plus bf16 scales AND biases. Reported at build so the log
/// line carries the number the change is claiming.
pub fn rerankCoarseBytes(rows: c_int, hidden: c_int, bits: u32) u64 {
    const r: u64 = @intCast(rows);
    const h: u64 = @intCast(hidden);
    const packed_bytes = r * h * @as(u64, bits) / 8;
    const groups = h / 64;
    return packed_bytes + 2 * (r * groups * 2);
}

/// `argmax(where(mask, -inf, logits))` as a lazy [.., 1] int32 id - the tail
/// every full-readout draft path shares.
pub fn maskAndArgmax(s: mlx.mlx_stream, logits_in: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_set(&logits, logits_in));
    if (suppress_mask) |m| {
        const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(neg_inf);
        var masked = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits, s));
        _ = mlx.mlx_array_free(logits);
        logits = masked;
    }
    var amax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(amax);
    try mlx.check(mlx.mlx_argmax_axis(&amax, logits, -1, false, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, amax, .int32, s));
    return out;
}

/// Full trunk-lm_head readout + argmax. The qwen4 arm's fallback: there is no
/// low-bit draft head to prefer.
pub fn fullReadoutArgmax(
    s: mlx.mlx_stream,
    target: *Transformer,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !mlx.mlx_array {
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    if (target.lm_head_s.ctx == null) {
        const axes = [_]c_int{ 1, 0 };
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose_axes(&wt, target.lm_head_w, &axes, 2, s));
        try mlx.check(mlx.mlx_matmul(&logits, x, wt, s));
    } else {
        const qp = headQuantParams(&target.config, target.lm_head_w, target.lm_head_s);
        try mlx.check(mlx.mlx_quantized_matmul(
            &logits,
            x,
            target.lm_head_w,
            target.lm_head_s,
            target.lm_head_b,
            true,
            mlx.mlx_optional_int.some(@intCast(qp.group_size)),
            mlx.mlx_optional_int.some(@intCast(qp.bits)),
            qp.mode.cstr(),
            s,
        ));
    }
    return maskAndArgmax(s, logits, suppress_mask);
}

/// `-inf` on every suppressed id of a coarse readout, in place.
fn rerankMaskCoarse(s: mlx.mlx_stream, logits: *mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !void {
    // The mask goes on the COARSE row: a reserved id that never enters the
    // shortlist cannot be re-scored back into the draft.
    const m = suppress_mask orelse return;
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    var masked = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits.*, s));
    _ = mlx.mlx_array_free(logits.*);
    logits.* = masked;
}

/// The exact re-scored shortlist behind every rerank draft: the coarse head's
/// top-32 candidate ids plus their trunk-head logits. A greedy draft is the
/// argmax over `exact`; a sampled draft is drawn from the filtered softmax over
/// it, so q's support is exactly these ids.
pub const Shortlist = struct {
    /// `[32]` uint32 vocab ids.
    cands: mlx.mlx_array,
    /// `[.., 32]` trunk-head logits for those ids.
    exact: mlx.mlx_array,
    /// The head's vocab rows — the width a dense proposal row must have.
    rows: c_int,

    pub fn deinit(self: *Shortlist) void {
        _ = mlx.mlx_array_free(self.cands);
        _ = mlx.mlx_array_free(self.exact);
    }
};

/// Shortlist one coarse row `flat` `[rows]` and re-score it through the trunk head.
/// Null retires the coarse head, so every later step lands on the caller's fallback.
fn rerankShortlistRow(
    s: mlx.mlx_stream,
    target: *Transformer,
    coarse: *?RerankCoarse,
    flat: mlx.mlx_array,
    rows: c_int,
    x_row: mlx.mlx_array,
) !?Shortlist {
    const cands = draftTop32(s, flat, rows) catch |err| {
        log.warn("[mtp] draft rerank shortlist failed ({s}) - dropping to the full readout\n", .{@errorName(err)});
        var dead = coarse.*.?;
        dead.deinit();
        coarse.* = null;
        return null;
    };
    errdefer _ = mlx.mlx_array_free(cands);

    // Re-score the shortlist through the trunk head's own rows: gathered
    // packed rows are self-contained (w [V, K*bits/32], scales/biases
    // [V, K/gs]), so a 32-row quantized matmul is exact.
    const qp = headQuantParams(&target.config, target.lm_head_w, target.lm_head_s);
    var w32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w32);
    var s32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(s32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_take_axis(&w32, target.lm_head_w, cands, 0, s));
    try mlx.check(mlx.mlx_take_axis(&s32, target.lm_head_s, cands, 0, s));
    if (target.lm_head_b.ctx != null)
        try mlx.check(mlx.mlx_take_axis(&b32, target.lm_head_b, cands, 0, s));

    var exact = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(exact);
    try mlx.check(mlx.mlx_quantized_matmul(
        &exact,
        x_row,
        w32,
        s32,
        b32,
        true,
        mlx.mlx_optional_int.some(@intCast(qp.group_size)),
        mlx.mlx_optional_int.some(@intCast(qp.bits)),
        qp.mode.cstr(),
        s,
    ));
    return .{ .cands = cands, .exact = exact, .rows = rows };
}

/// The greedy draft's tail: argmax over the re-scored shortlist, mapped back to
/// a vocab id as a lazy int32 array.
pub fn shortlistArgmax(s: mlx.mlx_stream, sl: Shortlist) !mlx.mlx_array {
    var amax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(amax);
    try mlx.check(mlx.mlx_argmax_axis(&amax, sl.exact, -1, false, s));
    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    try mlx.check(mlx.mlx_take_axis(&picked, sl.cands, amax, 0, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, picked, .int32, s));
    return out;
}

/// The coarse full-vocab readout of one row as a flat `[rows]` array, reserved
/// ids already at `-inf` — the front half every rerank draft shares.
fn rerankCoarseFlat(
    s: mlx.mlx_stream,
    rc: *const RerankCoarse,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !mlx.mlx_array {
    var coarse_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(coarse_logits);
    try mlx.check(mlx.mlx_quantized_matmul(
        &coarse_logits,
        x,
        rc.q.w,
        rc.q.s,
        rc.q.b,
        true,
        mlx.mlx_optional_int.some(@intCast(rc.group_size)),
        // The BUILT width and group, never `rerankCoarseBits()`: the packed
        // geometry implies K, so a per-step env re-read that disagrees with the
        // build does not read a coarser vocab - it reads a different one.
        mlx.mlx_optional_int.some(@intCast(rc.bits)),
        "affine",
        s,
    ));
    try rerankMaskCoarse(s, &coarse_logits, suppress_mask);
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    const flat_shape = [_]c_int{rc.rows};
    try mlx.check(mlx.mlx_reshape(&flat, coarse_logits, &flat_shape, 1, s));
    return flat;
}

/// `rerankSelect` without its argmax tail: the caller owns the shortlist and
/// decides the proposal over it (argmax for a greedy request, a draw from the
/// draft sampler's filtered softmax for a sampled one). Null means the coarse
/// head is unavailable, exactly as in `rerankSelect`.
pub fn rerankShortlist(
    s: mlx.mlx_stream,
    target: *Transformer,
    coarse: *?RerankCoarse,
    logged: *bool,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !?Shortlist {
    const rc = if (coarse.*) |*p| p else return null;
    const coarse_bits = rc.bits;
    const rows = rc.rows;

    const flat = try rerankCoarseFlat(s, rc, x, suppress_mask);
    defer _ = mlx.mlx_array_free(flat);

    const sl = (try rerankShortlistRow(s, target, coarse, flat, rows, x)) orelse return null;
    if (!logged.*) {
        log.info("[mtp] draft rerank engaged ({d}-bit coarse -> top-32 -> trunk re-score)\n", .{coarse_bits});
        logged.* = true;
    }
    return sl;
}

/// One greedy draft proposal via the rerank scheme: coarse full-vocab readout
/// -> exact top-32 shortlist (two dispatches) -> the trunk head's own 32 rows
/// re-score -> argmax. Returns a lazy [1,1] int32 token id, or NULL when the
/// coarse head is unavailable - never built, or a shortlist-kernel failure
/// just dropped it (`coarse` is freed and nulled in that case, so every later
/// step of the same chain lands on the caller's fallback too).
///
/// PROPOSAL SIDE ONLY: verification still reads the trunk forward's logits, so
/// a coarse miss costs acceptance, never output.
pub fn rerankSelect(
    s: mlx.mlx_stream,
    target: *Transformer,
    coarse: *?RerankCoarse,
    logged: *bool,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !?mlx.mlx_array {
    var sl = (try rerankShortlist(s, target, coarse, logged, x, suppress_mask)) orelse return null;
    defer sl.deinit();
    return try shortlistArgmax(s, sl);
}

fn rerankCoarseRows(s: mlx.mlx_stream, x: mlx.mlx_array, rc: *const RerankCoarse) !mlx.mlx_array {
    const shape = mlx.getShape(x);
    if (shape.len < 2 or shape.len > 3 or shape[0] < 1 or shape[0] > 32) return error.MtpRerankBatchTooWide;
    if (shape.len == 3 and shape[1] != 1) return error.MtpRerankInputShape;
    if (try transformer_mod.mtpCoarsePairs(s, x, rc.q.w, rc.q.s, rc.q.b, rc.bits, rc.group_size)) |out| return out;
    if (try @import("transformer.zig").msvQmvRows(s, x, rc.q.w, rc.q.s, rc.q.b, rc.bits, rc.group_size)) |out| return out;
    const n: usize = @intCast(shape[0]);
    var parts: [32]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (parts[0..n]) |p| {
        if (p.ctx != null) _ = mlx.mlx_array_free(p);
    };
    for (0..n) |i| {
        var start: [3]c_int = @splat(0);
        var stop: [3]c_int = @splat(0);
        const strides: [3]c_int = @splat(1);
        @memcpy(stop[0..shape.len], shape);
        start[0] = @intCast(i);
        stop[0] = @intCast(i + 1);
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_slice(&row, x, &start, @intCast(shape.len), &stop, @intCast(shape.len), &strides, @intCast(shape.len), s));
        parts[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(&parts[i], row, rc.q.w, rc.q.s, rc.q.b, true, mlx.mlx_optional_int.some(@intCast(rc.group_size)), mlx.mlx_optional_int.some(@intCast(rc.bits)), "affine", s));
    }
    const vec = mlx.mlx_vector_array_new_data(parts[0..n].ptr, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, s));
    return out;
}
/// Per-row exact re-scored shortlists from mixer rows `x` `[N,1,H]`: ONE coarse
/// readout over N, then the top-32 select and the 32-row re-score per row.
/// False = the coarse head is unavailable (nothing is written to `out`).
pub fn rerankShortlistsBatched(
    s: mlx.mlx_stream,
    target: *Transformer,
    coarse: *?RerankCoarse,
    logged: *bool,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
    out: []Shortlist,
) !bool {
    const rc = if (coarse.*) |*p| p else return false;
    const xsh = mlx.getShape(x);
    const N: usize = @intCast(xsh[0]);
    if (N != out.len) return error.MtpRerankBatchLengthMismatch;
    if (N > 32) return error.MtpRerankBatchTooWide;

    var coarse_logits = try rerankCoarseRows(s, x, rc);
    defer _ = mlx.mlx_array_free(coarse_logits);
    try rerankMaskCoarse(s, &coarse_logits, suppress_mask);
    if (!logged.*) {
        log.info("[mtp] draft rerank engaged ({d}-bit coarse -> top-32 -> trunk re-score)\n", .{rc.bits});
        logged.* = true;
    }

    const rows = rc.rows;
    var built: usize = 0;
    errdefer for (out[0..built]) |*sl| sl.deinit();
    const csh = mlx.getShape(coarse_logits);
    const last = csh[csh.len - 1];
    for (0..N) |i| {
        const i_c: c_int = @intCast(i);
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        if (csh.len == 3) {
            try mlx.check(mlx.mlx_slice(&row, coarse_logits, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, 1, last }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        } else {
            try mlx.check(mlx.mlx_slice(&row, coarse_logits, &[_]c_int{ i_c, 0 }, 2, &[_]c_int{ i_c + 1, last }, 2, &[_]c_int{ 1, 1 }, 2, s));
        }
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, row, &[_]c_int{rows}, 1, s));
        var xi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xi);
        try mlx.check(mlx.mlx_slice(&xi, x, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, 1, xsh[xsh.len - 1] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        // A shortlist failure retires the coarse head mid-loop; the caller's
        // fallback answers for the whole group, so the rows already built go.
        out[i] = (try rerankShortlistRow(s, target, coarse, flat, rows, xi)) orelse {
            for (out[0..built]) |*sl| sl.deinit();
            return false;
        };
        built = i + 1;
    }
    return true;
}

/// Batched greedy draft ids from mixer rows `x` `[N,1,H]`. Same scheme as
/// `rerankSelect` per row; one coarse readout over N.
pub fn rerankSelectBatched(
    s: mlx.mlx_stream,
    target: *Transformer,
    coarse: *?RerankCoarse,
    logged: *bool,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !?mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const N: usize = @intCast(xsh[0]);
    if (N <= 1) return rerankSelect(s, target, coarse, logged, x, suppress_mask);
    if (N > 32) return error.MtpRerankBatchTooWide;

    var shortlists: [32]Shortlist = undefined;
    if (!try rerankShortlistsBatched(s, target, coarse, logged, x, suppress_mask, shortlists[0..N])) return null;
    defer for (shortlists[0..N]) |*sl| sl.deinit();

    var id_parts: [32]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (id_parts[0..N]) |p| {
        if (p.ctx != null) _ = mlx.mlx_array_free(p);
    };
    for (0..N) |i| {
        const id = try shortlistArgmax(s, shortlists[i]);
        defer _ = mlx.mlx_array_free(id);
        id_parts[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&id_parts[i], id, &[_]c_int{ 1, 1 }, 2, s));
    }
    const vec = mlx.mlx_vector_array_new_data(id_parts[0..N].ptr, N);
    defer _ = mlx.mlx_vector_array_free(vec);
    var stacked = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&stacked, vec, 0, s));
    return stacked;
}

/// Sanity bound for index.json / safetensors headers (headers of the largest
/// checkpoints stay well under this).
const checkpoint_header_limit: usize = 64 * 1024 * 1024;

/// Read a file under `dir` fully (bounded); null on absence or overflow.
fn readDirFileAlloc(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, sub_path: []const u8, limit: usize) ?[]u8 {
    const f = dir.openFile(io, sub_path, .{}) catch return null;
    defer f.close(io);
    var rb: [8192]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(allocator, .limited(limit)) catch null;
}

/// Qwen3.8-Flash-Next's head is the checkpoint's own layer, loaded by the
/// trunk (`loadQwen4Mtp`).
const qwen4_mtp_marker = "\"language_model.mtp.fc_hidden.weight\"";

fn indexJsonHasQwen4Mtp(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir) bool {
    const bytes = readDirFileAlloc(io, allocator, dir, "model.safetensors.index.json", checkpoint_header_limit) orelse return false;
    defer allocator.free(bytes);
    return std.mem.indexOf(u8, bytes, qwen4_mtp_marker) != null;
}

/// Advertisement probe: does the checkpoint carry the qwen4_exp head.
pub fn dirAdvertisesMtp(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir) bool {
    return indexJsonHasQwen4Mtp(io, allocator, dir);
}

/// Requantize a row-quantized affine weight `(w, scales, biases)` from
/// `(from_gs, from_bits)` to `(to_gs, to_bits)`, chunk-wise over rows so the
/// dequantized bf16 transient stays bounded (~chunk_rows × in_features × 2 B
/// instead of the whole matrix — a 248K×5120 lm_head would otherwise
/// materialize 2.5 GB). Rows quantize independently in MLX's affine packing,
/// so per-chunk triples concatenate along axis 0 into a valid whole.
///
/// A DENSE source (`scales.ctx == null`, e.g. a bf16 lm_head) skips the
/// dequantize and quantizes the row chunk directly — same chunking, one
/// requantizer.
pub fn requantizeRows(
    s: mlx.mlx_stream,
    w: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    from_gs: u32,
    from_bits: u32,
    from_mode: [*:0]const u8,
    to_gs: u32,
    to_bits: u32,
    chunk_rows: c_int,
) !QLinear {
    const w_shape = mlx.getShape(w);
    if (w_shape.len != 2) return error.UnsupportedDraftHeadShape;
    const rows: c_int = w_shape[0];

    const wv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(wv);
    const sv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(sv);
    const bv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(bv);

    var r0: c_int = 0;
    while (r0 < rows) : (r0 += chunk_rows) {
        const r1: c_int = @min(rows, r0 + chunk_rows);

        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        if (scales.ctx == null) {
            var raw = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(raw);
            try sliceRows(&raw, w, r0, r1, s);
            try mlx.check(mlx.mlx_astype(&dense, raw, .bfloat16, s));
        } else {
            var wq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wq);
            var sq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sq);
            var bq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bq);
            try sliceRows(&wq, w, r0, r1, s);
            try sliceRows(&sq, scales, r0, r1, s);
            if (biases.ctx != null) try sliceRows(&bq, biases, r0, r1, s);
            try mlx.check(mlx.mlx_dequantize(
                &dense,
                wq,
                sq,
                bq,
                mlx.mlx_optional_int.some(@intCast(from_gs)),
                mlx.mlx_optional_int.some(@intCast(from_bits)),
                from_mode,
                .{}, // global_scale
                .{ .value = .bfloat16, .has_value = true },
                s,
            ));
        }

        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(
            &triple,
            dense,
            mlx.mlx_optional_int.some(@intCast(to_gs)),
            mlx.mlx_optional_int.some(@intCast(to_bits)),
            "affine",
            .{}, // global_scale
            s,
        ));
        if (mlx.mlx_vector_array_size(triple) != 3) return error.UnexpectedQuantizeOutput;
        var part = [3]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        try mlx.check(mlx.mlx_vector_array_get(&part[0], triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&part[1], triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&part[2], triple, 2));
        // Realize the chunk so its dense transient can be reclaimed before
        // the next chunk builds (lazy eval would otherwise stack them all).
        {
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            for (part) |p| _ = mlx.mlx_vector_array_append_value(ev, p);
            try mlx.check(mlx.mlx_eval(ev));
        }
        _ = mlx.mlx_vector_array_append_value(wv, part[0]);
        _ = mlx.mlx_vector_array_append_value(sv, part[1]);
        _ = mlx.mlx_vector_array_append_value(bv, part[2]);
        for (part) |p| _ = mlx.mlx_array_free(p);
    }

    var out = QLinear{
        .w = mlx.mlx_array_new(),
        .s = mlx.mlx_array_new(),
        .b = mlx.mlx_array_new(),
    };
    errdefer out.deinit();
    try mlx.check(mlx.mlx_concatenate_axis(&out.w, wv, 0, s));
    try mlx.check(mlx.mlx_concatenate_axis(&out.s, sv, 0, s));
    try mlx.check(mlx.mlx_concatenate_axis(&out.b, bv, 0, s));
    return out;
}

fn sliceRows(out: *mlx.mlx_array, src: mlx.mlx_array, r0: c_int, r1: c_int, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(src);
    const start = [_]c_int{ r0, 0 };
    const stop = [_]c_int{ r1, shape[1] };
    const strides = [_]c_int{ 1, 1 };
    try mlx.check(mlx.mlx_slice(out, src, &start, 2, &stop, 2, &strides, 2, s));
}

/// The trunk lm_head's TRUE quant params, via the same dtype-gated resolver
/// the trunk itself uses: uint8 scales → the config's non-affine mode (a raw
/// geometry solve mis-reads mxfp8 8-bit/gs32 as AFFINE 8-bit/gs32 — issue
/// #81, "Biases must be provided" crash on biasless heads); float scales →
/// exact per-geometry affine solve (the head's bits routinely differ from
/// the trunk global on mixed checkpoints — an 8-bit head over a 2-bit trunk).
fn headQuantParams(config: *const model_mod.ModelConfig, w: mlx.mlx_array, sc: mlx.mlx_array) transformer_mod.QuantParams {
    return transformer_mod.computeQuantParams(config, w, sc, config.hidden_size);
}

pub const StepOut = struct {
    /// `[1, 1, vocab]` LAST-row logits, or `.ctx == null` when the caller did
    /// not ask for `.logits` (multi-row calls never project the history rows —
    /// only the last row feeds the draft chain).
    logits: mlx.mlx_array,
    /// MTP post-norm hidden — the next depth's `hidden` input. `[1, 1, H]`
    /// (last row) under `.logits`, the full `[1, L, H]` otherwise.
    hidden_next: mlx.mlx_array,
    /// The vector the TARGET's lm_head consumes, for `.mixed` steps. Null on
    /// every arm where that is just `hidden_next`; non-null on
    /// qwen4_exp, whose `hidden_next` is the PRE-mixer stream and whose
    /// lm_head input is the mixer output. Owned by the caller when non-null.
    rerank_x: mlx.mlx_array = .{ .ctx = null },
};

/// What a head step owes its caller. `want_logits: bool` could not tell the
/// two zero-logits cases apart: a history append wants nothing past the
/// stream, while a rerank draft still needs the vector the lm_head consumes.
pub const StepWant = enum {
    /// Last-row logits (and the last-row hidden).
    logits,
    /// No logits, but the lm_head's INPUT vector for the last row — the
    /// rerank draft path.
    mixed,
    /// Nothing past the stream: committed-history appends.
    none,
};

pub const MropeContext = mrope.PositionContext;

// ── Draft-rerank shortlist (port of the mlx.fast challenge draft-rerank
// scheme, submission 942e5ab2 in mlx.fast-qwen-3.8-mtp-challenge) ──
//
// PROPOSAL SIDE ONLY: a coarse 2-bit requant of the trunk lm_head scores the
// full vocab, an exact top-32 shortlist is taken in TWO dispatches (MLX's
// GPU argpartition is a full multi-block argsort — 14 dependent dispatches
// for 32 read values), and the trunk head's own 32 rows re-score the
// shortlist. The draft is the trunk's argmax over those rows, so whenever
// the coarse top-32 contains the trunk argmax the draft IS the trunk's
// choice — acceptance rides the trunk's order at ~2-bit readout cost.
// Verification is untouched: a coarse miss costs acceptance, never output.
//
// Ordinal: a monotone map from float into uint32 inducing (value asc, NaN
// above every number, -0.0 == +0.0). Real values never map to 0, so the
// zero-initialized empty slots below can never be selected while a
// simdgroup still has real candidates (guaranteed by rows >= TILES*TG).
const TOP32_ORDINAL_HEADER =
    \\inline uint sushi_top32_ordinal(float v) {
    \\    if (isnan(v))  { return 0xFFFFFFFFu; }
    \\    if (v == 0.0f) { return 0x80000000u; }
    \\    uint u = as_type<uint>(v);
    \\    return (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
    \\}
;

// Stage 1: TILES threadgroups partition [0, RC); each emits its local top 32
// as (ordinal, index) pairs — TILES * 32 candidates. Selection is 32 rounds
// of simd_max over per-thread slots with a `taken` bitmask (hence the
// PER_THREAD <= 32 static_assert), then one simdgroup reduces the
// threadgroup's per-simd lists the same way.
const TOP32_PARTIAL_SRC =
    \\constexpr uint REAL_COUNT = (uint)RC;
    \\constexpr uint TG_SIZE    = 256;
    \\constexpr uint STRIDE     = 64u * 256u;
    \\constexpr uint PER_THREAD = (REAL_COUNT + STRIDE - 1u) / STRIDE;
    \\constexpr uint TOPK       = 32;
    \\constexpr uint SIMD_SIZE  = 32;
    \\constexpr uint NSIMD      = TG_SIZE / SIMD_SIZE;
    \\constexpr uint PB         = (NSIMD * TOPK) / SIMD_SIZE;
    \\static_assert(PER_THREAD <= 32, "PER_THREAD exceeds taken-bitmask width");
    \\static_assert(PB <= 32, "PB exceeds tk2-bitmask width");
    \\
    \\uint tile = threadgroup_position_in_grid.x;
    \\uint tid  = thread_position_in_threadgroup.x;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg   = simdgroup_index_in_threadgroup;
    \\
    \\uint ord[PER_THREAD];
    \\uint idx[PER_THREAD];
    \\for (uint t = 0; t < PER_THREAD; ++t) { ord[t] = 0u; idx[t] = 0u; }
    \\uint n = 0;
    \\for (uint i = tile * TG_SIZE + tid; i < REAL_COUNT; i += STRIDE) {
    \\    ord[n] = sushi_top32_ordinal(float(logits[i]));
    \\    idx[n] = i;
    \\    n++;
    \\}
    \\
    \\threadgroup uint sc_ord[NSIMD * TOPK];
    \\threadgroup uint sc_idx[NSIMD * TOPK];
    \\
    \\uint taken = 0u;
    \\for (uint r = 0; r < TOPK; ++r) {
    \\    uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
    \\    for (uint t = 0; t < PER_THREAD; ++t) {
    \\        if ((taken & (1u << t)) != 0u) { continue; }
    \\        if (ord[t] > bo || (ord[t] == bo && idx[t] > bi)) {
    \\            bo = ord[t]; bi = idx[t]; bs = t;
    \\        }
    \\    }
    \\    uint mo = simd_max(bo);
    \\    uint mi = simd_max((bo == mo) ? bi : 0u);
    \\    if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {
    \\        taken |= (1u << bs);
    \\    }
    \\    if (lane == 0) {
    \\        sc_ord[sg * TOPK + r] = mo;
    \\        sc_idx[sg * TOPK + r] = mi;
    \\    }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\if (sg == 0) {
    \\    uint o2[PB];
    \\    uint i2[PB];
    \\    for (uint t = 0; t < PB; ++t) {
    \\        uint p = t * SIMD_SIZE + lane;
    \\        o2[t] = sc_ord[p];
    \\        i2[t] = sc_idx[p];
    \\    }
    \\    uint tk2 = 0u;
    \\    for (uint r = 0; r < TOPK; ++r) {
    \\        uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
    \\        for (uint t = 0; t < PB; ++t) {
    \\            if ((tk2 & (1u << t)) != 0u) { continue; }
    \\            if (o2[t] > bo || (o2[t] == bo && i2[t] > bi)) {
    \\                bo = o2[t]; bi = i2[t]; bs = t;
    \\            }
    \\        }
    \\        uint mo = simd_max(bo);
    \\        uint mi = simd_max((bo == mo) ? bi : 0u);
    \\        if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {
    \\            tk2 |= (1u << bs);
    \\        }
    \\        if (lane == 0) {
    \\            cand_ord[tile * TOPK + r] = mo;
    \\            cand_idx[tile * TOPK + r] = mi;
    \\        }
    \\    }
    \\}
;

// Stage 2: one threadgroup reduces the TILES*32 candidates to the final 32
// ids (written ascending by (value, index) — consumers are order-blind).
const TOP32_FINAL_SRC =
    \\constexpr uint TG_SIZE    = 256;
    \\constexpr uint PER_THREAD = 8;
    \\constexpr uint TOPK       = 32;
    \\constexpr uint SIMD_SIZE  = 32;
    \\constexpr uint NSIMD      = TG_SIZE / SIMD_SIZE;
    \\constexpr uint PB         = (NSIMD * TOPK) / SIMD_SIZE;
    \\
    \\uint tid  = thread_position_in_threadgroup.x;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg   = simdgroup_index_in_threadgroup;
    \\
    \\uint ord[PER_THREAD];
    \\uint idx[PER_THREAD];
    \\for (uint t = 0; t < PER_THREAD; ++t) {
    \\    uint p = t * TG_SIZE + tid;
    \\    ord[t] = cand_ord[p];
    \\    idx[t] = cand_idx[p];
    \\}
    \\
    \\threadgroup uint sc_ord[NSIMD * TOPK];
    \\threadgroup uint sc_idx[NSIMD * TOPK];
    \\
    \\uint taken = 0u;
    \\for (uint r = 0; r < TOPK; ++r) {
    \\    uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
    \\    for (uint t = 0; t < PER_THREAD; ++t) {
    \\        if ((taken & (1u << t)) != 0u) { continue; }
    \\        if (ord[t] > bo || (ord[t] == bo && idx[t] > bi)) {
    \\            bo = ord[t]; bi = idx[t]; bs = t;
    \\        }
    \\    }
    \\    uint mo = simd_max(bo);
    \\    uint mi = simd_max((bo == mo) ? bi : 0u);
    \\    if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {
    \\        taken |= (1u << bs);
    \\    }
    \\    if (lane == 0) {
    \\        sc_ord[sg * TOPK + r] = mo;
    \\        sc_idx[sg * TOPK + r] = mi;
    \\    }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\if (sg == 0) {
    \\    uint o2[PB];
    \\    uint i2[PB];
    \\    for (uint t = 0; t < PB; ++t) {
    \\        uint p = t * SIMD_SIZE + lane;
    \\        o2[t] = sc_ord[p];
    \\        i2[t] = sc_idx[p];
    \\    }
    \\    uint tk2 = 0u;
    \\    for (uint r = 0; r < TOPK; ++r) {
    \\        uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
    \\        for (uint t = 0; t < PB; ++t) {
    \\            if ((tk2 & (1u << t)) != 0u) { continue; }
    \\            if (o2[t] > bo || (o2[t] == bo && i2[t] > bi)) {
    \\                bo = o2[t]; bi = i2[t]; bs = t;
    \\            }
    \\        }
    \\        uint mo = simd_max(bo);
    \\        uint mi = simd_max((bo == mo) ? bi : 0u);
    \\        if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {
    \\            tk2 |= (1u << bs);
    \\        }
    \\        if (lane == 0) { token_ids[TOPK - 1u - r] = mi; }
    \\    }
    \\}
;

const TOP32_K: c_int = 32;
const TOP32_TG: c_int = 256;
const TOP32_TILES: c_int = 64;
const TOP32_CANDS: c_int = TOP32_TILES * TOP32_K;
/// Minimum row count: below TILES*TG some simdgroups hold fewer than 32 real
/// candidates and the zero-initialized empty slots become reachable.
pub const TOP32_MIN_ROWS: c_int = TOP32_TILES * TOP32_TG;

var top32_partial_cached: ?mlx.mlx_fast_metal_kernel = null;
var top32_final_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getTop32Partial() !mlx.mlx_fast_metal_kernel {
    if (top32_partial_cached) |k| return k;
    const input_names = [_][*:0]const u8{"logits"};
    const output_names = [_][*:0]const u8{ "cand_ord", "cand_idx" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_mtp_top32_partial",
        in_vec,
        out_vec,
        TOP32_PARTIAL_SRC,
        TOP32_ORDINAL_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    top32_partial_cached = kernel;
    return kernel;
}

fn getTop32Final() !mlx.mlx_fast_metal_kernel {
    if (top32_final_cached) |k| return k;
    const input_names = [_][*:0]const u8{ "cand_ord", "cand_idx" };
    const output_names = [_][*:0]const u8{"token_ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_mtp_top32_finalize",
        in_vec,
        out_vec,
        TOP32_FINAL_SRC,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    top32_final_cached = kernel;
    return kernel;
}

/// Exact top-32 ids of `row` ([rows], any float dtype) as a lazy [32] uint32
/// array, ascending by (value, index) — ties break toward the higher index,
/// NaN above every number. `rows` rides as a template int (stable per model,
/// so MLX caches one specialization per vocab width).
pub fn draftTop32(s: mlx.mlx_stream, row: mlx.mlx_array, rows: c_int) !mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return error.MetalKernelNeedsGpuStream;
    if (rows < TOP32_MIN_ROWS) return error.UnsupportedTop32Shape;
    const per_thread = @divTrunc(rows + TOP32_TILES * TOP32_TG - 1, TOP32_TILES * TOP32_TG);
    if (per_thread > 32) return error.UnsupportedTop32Shape;

    const pk = try getTop32Partial();
    var cand_ord = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ord);
    var cand_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_idx);
    {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const cand_shape = [_]c_int{TOP32_CANDS};
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &cand_shape, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &cand_shape, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, TOP32_TILES * TOP32_TG, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, TOP32_TG, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RC", rows));
        const inputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(inputs);
        _ = mlx.mlx_vector_array_append_value(inputs, row);
        var outputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, pk, inputs, cfg, s));
        try mlx.check(mlx.mlx_vector_array_get(&cand_ord, outputs, 0));
        try mlx.check(mlx.mlx_vector_array_get(&cand_idx, outputs, 1));
    }

    const fk = try getTop32Final();
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_shape = [_]c_int{TOP32_K};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, TOP32_TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, TOP32_TG, 1, 1));
    const inputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(inputs);
    _ = mlx.mlx_vector_array_append_value(inputs, cand_ord);
    _ = mlx.mlx_vector_array_append_value(inputs, cand_idx);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, fk, inputs, cfg, s));
    var token_ids = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&token_ids, outputs, 0));
    return token_ids;
}

// ── Tests ──

const testing = std.testing;

test "mtp: requantizeRows round-trips through a finer re-encode (chunked)" {
    const s = mlx.gpuStream();
    const rows: usize = 64;
    const cols: usize = 256;

    var prng = std.Random.DefaultPrng.init(42);
    const buf = try testing.allocator.alloc(f32, rows * cols);
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32);
    const shape = [_]c_int{ @intCast(rows), @intCast(cols) };
    const dense_f32 = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(dense_f32);
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_astype(&dense, dense_f32, .bfloat16, s));

    // "Trunk" 4-bit/gs64 triple.
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, dense, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var q4 = QLinear{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new(), .b = mlx.mlx_array_new() };
    defer q4.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&q4.w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q4.s, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&q4.b, triple, 2));

    // Requantize to 8-bit/gs64, chunked at 16 rows (4 chunks → exercises concat).
    var q8 = try requantizeRows(s, q4.w, q4.s, q4.b, 64, 4, "affine", 64, 8, 16);
    defer q8.deinit();

    // Dequantize both and compare — an 8-bit re-encode of 4-bit-quantized
    // values is near-lossless, so cosine must be ~1.
    var deq = [2]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (deq) |d| {
        _ = mlx.mlx_array_free(d);
    };
    try mlx.check(mlx.mlx_dequantize(&deq[0], q4.w, q4.s, q4.b, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, .{ .value = .float32, .has_value = true }, s));
    try mlx.check(mlx.mlx_dequantize(&deq[1], q8.w, q8.s, q8.b, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, .{ .value = .float32, .has_value = true }, s));
    for (deq) |d| try mlx.check(mlx.mlx_array_eval(d));

    const a = mlx.mlx_array_data_float32(deq[0]).?;
    const b = mlx.mlx_array_data_float32(deq[1]).?;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..rows * cols) |i| {
        dot += @as(f64, a[i]) * b[i];
        na += @as(f64, a[i]) * a[i];
        nb += @as(f64, b[i]) * b[i];
    }
    const cos = dot / (@sqrt(na) * @sqrt(nb));
    try testing.expect(cos > 0.999);

    // Shape sanity: 8-bit packs 4 in-features per u32 → cols/4 packed cols.
    const w8_shape = mlx.getShape(q8.w);
    try testing.expectEqual(@as(c_int, @intCast(rows)), w8_shape[0]);
    try testing.expectEqual(@as(c_int, @intCast(cols / 4)), w8_shape[1]);
}

test "mtp: requantizeRows accepts a non-affine (mxfp8) source — the issue-#81 draft-head path" {
    // With headQuantParams fixed, buildDraftHead on an mxfp8 trunk passes
    // mode="mxfp8" and the load path's null-ctx biases into requantizeRows.
    // The source dequantize must ride that without demanding affine biases
    // (else the crash just moves from the first forward to bind time), and
    // the 3-bit affine draft re-encode must stay correlated.
    const s = mlx.gpuStream();
    const rows: usize = 64;
    const cols: usize = 256;

    var prng = std.Random.DefaultPrng.init(7);
    const buf = try testing.allocator.alloc(f32, rows * cols);
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32);
    const shape = [_]c_int{ @intCast(rows), @intCast(cols) };
    const dense_f32 = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(dense_f32);
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_astype(&dense, dense_f32, .bfloat16, s));

    // mxfp8 "trunk head": a (w, scales) pair — no biases tensor, by design.
    var pair = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(pair);
    try mlx.check(mlx.mlx_quantize(&pair, dense, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", .{}, s));
    var qmx = QLinear{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new(), .b = mlx.mlx_array_new() };
    defer qmx.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&qmx.w, pair, 0));
    try mlx.check(mlx.mlx_vector_array_get(&qmx.s, pair, 1));
    // qmx.b stays null-ctx — exactly what the load path hands buildDraftHead.

    // The live buildDraftHead call shape: 3-bit/gs64 draft re-encode, chunked.
    var q3 = try requantizeRows(s, qmx.w, qmx.s, qmx.b, 32, 8, "mxfp8", 64, 3, 16);
    defer q3.deinit();

    var deq = [2]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (deq) |d| {
        _ = mlx.mlx_array_free(d);
    };
    try mlx.check(mlx.mlx_dequantize(&deq[0], qmx.w, qmx.s, qmx.b, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", .{}, .{ .value = .float32, .has_value = true }, s));
    try mlx.check(mlx.mlx_dequantize(&deq[1], q3.w, q3.s, q3.b, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(3), "affine", .{}, .{ .value = .float32, .has_value = true }, s));
    for (deq) |d| try mlx.check(mlx.mlx_array_eval(d));

    const a = mlx.mlx_array_data_float32(deq[0]).?;
    const b = mlx.mlx_array_data_float32(deq[1]).?;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..rows * cols) |i| {
        dot += @as(f64, a[i]) * b[i];
        na += @as(f64, a[i]) * a[i];
        nb += @as(f64, b[i]) * b[i];
    }
    const cos = dot / (@sqrt(na) * @sqrt(nb));
    try testing.expect(cos > 0.95);
}

test "mtp: headQuantParams never mis-resolves a non-affine lm_head as affine (issue #81)" {
    // An mxfp8 8-bit/gs32 lm_head's GEOMETRY coincidentally solves as a valid
    // affine 8-bit/gs32 (w [V, H/4] u32, scales [V, H/32]) — but the scales
    // are fp8-encoded uint8 and no biases tensor exists, so a mode="affine"
    // matmul/dequantize throws "Biases must be provided for affine
    // quantization" and kills the first MTP forward. The scales-dtype gate
    // (uint8 → the config's non-affine mode) must win over the geometry
    // shortcut, exactly as computeQuantParams resolves the trunk.
    const s = mlx.gpuStream();
    const H = 512;
    const V = 8;

    const mk = struct {
        fn arr(shape: []const c_int, dt: mlx.mlx_dtype, st: mlx.mlx_stream) !mlx.mlx_array {
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&a, shape.ptr, shape.len, dt, st));
            return a;
        }
    };

    // mxfp8 head: w [V, H*8/32] u32, scales [V, H/32] u8.
    const w8 = try mk.arr(&.{ V, H * 8 / 32 }, .uint32, s);
    defer _ = mlx.mlx_array_free(w8);
    const s8 = try mk.arr(&.{ V, H / 32 }, .uint8, s);
    defer _ = mlx.mlx_array_free(s8);

    var cfg = model_mod.ModelConfig{};
    cfg.hidden_size = H;
    cfg.quant_bits = 8;
    cfg.quant_group_size = 32;
    cfg.quant_mode = .mxfp8;

    const qp8 = headQuantParams(&cfg, w8, s8);
    try testing.expectEqual(model_mod.QuantMode.mxfp8, qp8.mode);
    try testing.expectEqual(@as(u32, 8), qp8.bits);
    try testing.expectEqual(@as(u32, 32), qp8.group_size);

    // mxfp4 (same class, 4-bit/gs32 — also a false-positive affine geometry).
    const w4 = try mk.arr(&.{ V, H * 4 / 32 }, .uint32, s);
    defer _ = mlx.mlx_array_free(w4);
    cfg.quant_bits = 4;
    cfg.quant_mode = .mxfp4;
    const qp4 = headQuantParams(&cfg, w4, s8);
    try testing.expectEqual(model_mod.QuantMode.mxfp4, qp4.mode);

    // Characterization (green before AND after): the mixed-AFFINE shape this
    // function exists for — an 8-bit/gs32 head (bf16 scales) over a
    // 2-bit/gs64 trunk still resolves per-geometry, never per-config.
    const sb = try mk.arr(&.{ V, H / 32 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(sb);
    cfg.quant_bits = 2;
    cfg.quant_group_size = 64;
    cfg.quant_mode = .affine;
    const qpa = headQuantParams(&cfg, w8, sb);
    try testing.expectEqual(model_mod.QuantMode.affine, qpa.mode);
    try testing.expectEqual(@as(u32, 8), qpa.bits);
    try testing.expectEqual(@as(u32, 32), qpa.group_size);
}

/// A bare Transformer carrying nothing but a quantized lm_head + the config
/// fields `headQuantParams` reads. Every function under test touches exactly
/// those, which is the property that let the qwen4_exp head reuse the scheme.
/// A minimal target for the rerank scheme: a Transformer carrying nothing
/// but a quantized `lm_head`, which is all the scheme ever reads. `pub` so
/// the qwen4 arm's tests in generate.zig build the same target.
pub const RerankFixture = struct {
    xfm: Transformer,
    dense: mlx.mlx_array,
    vocab: c_int,
    hidden: c_int,

    pub fn init(s: mlx.mlx_stream, vocab: c_int, hidden: c_int, bits: u32, gs: u32, seed: u64) !RerankFixture {
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        const n: usize = @intCast(vocab * hidden);
        const buf = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(buf);
        // bf16-exact levels: the dense reference and the requantized source
        // must not differ by a rounding the test would read as a rerank miss.
        for (buf) |*v| v.* = @as(f32, @floatFromInt(rand.intRangeAtMost(i32, -128, 127))) * 0.0078125;
        const shape = [_]c_int{ vocab, hidden };
        const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var dense = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&dense, f32_arr, .bfloat16, s));

        var qv = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(qv);
        try mlx.check(mlx.mlx_quantize(
            &qv,
            dense,
            mlx.mlx_optional_int.some(@intCast(gs)),
            mlx.mlx_optional_int.some(@intCast(bits)),
            "affine",
            .{ .ctx = null },
            s,
        ));
        var xfm: Transformer = undefined;
        xfm.config = .{};
        xfm.config.hidden_size = @intCast(hidden);
        xfm.config.quant_mode = .affine;
        xfm.config.quant_bits = bits;
        xfm.config.quant_group_size = gs;
        xfm.s = s;
        xfm.lm_head_w = mlx.mlx_array_new();
        xfm.lm_head_s = mlx.mlx_array_new();
        xfm.lm_head_b = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&xfm.lm_head_w, qv, 0));
        try mlx.check(mlx.mlx_vector_array_get(&xfm.lm_head_s, qv, 1));
        try mlx.check(mlx.mlx_vector_array_get(&xfm.lm_head_b, qv, 2));
        return .{ .xfm = xfm, .dense = dense, .vocab = vocab, .hidden = hidden };
    }

    pub fn deinit(self: *RerankFixture) void {
        _ = mlx.mlx_array_free(self.dense);
        _ = mlx.mlx_array_free(self.xfm.lm_head_w);
        _ = mlx.mlx_array_free(self.xfm.lm_head_s);
        _ = mlx.mlx_array_free(self.xfm.lm_head_b);
    }

    /// `[1, 1, hidden]` bf16 activation.
    fn randomX(self: *const RerankFixture, s: mlx.mlx_stream, seed: u64) !mlx.mlx_array {
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        const buf = try testing.allocator.alloc(f32, @intCast(self.hidden));
        defer testing.allocator.free(buf);
        for (buf) |*v| v.* = @as(f32, @floatFromInt(rand.intRangeAtMost(i32, -64, 63))) * 0.015625;
        const shape = [_]c_int{ 1, 1, self.hidden };
        const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&out, f32_arr, .bfloat16, s));
        return out;
    }

    /// The full-vocab logits the real head would produce — the exact answer
    /// the rerank is judged against.
    fn exactLogits(self: *const RerankFixture, s: mlx.mlx_stream, x: mlx.mlx_array) !mlx.mlx_array {
        const qp = headQuantParams(&self.xfm.config, self.xfm.lm_head_w, self.xfm.lm_head_s);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(
            &out,
            x,
            self.xfm.lm_head_w,
            self.xfm.lm_head_s,
            self.xfm.lm_head_b,
            true,
            mlx.mlx_optional_int.some(@intCast(qp.group_size)),
            mlx.mlx_optional_int.some(@intCast(qp.bits)),
            qp.mode.cstr(),
            s,
        ));
        return out;
    }
};

fn readIdScalar(arr: mlx.mlx_array) !u32 {
    try mlx.check(mlx.mlx_array_eval(arr));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, arr));
    return @intCast(v);
}

fn readRow(allocator: std.mem.Allocator, s: mlx.mlx_stream, arr: mlx.mlx_array, n: usize) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const ptr = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    const out = try allocator.alloc(f32, n);
    @memcpy(out, ptr[0..n]);
    return out;
}

test "mtp: draft rerank returns the exact argmax whenever it is inside the coarse shortlist" {
    // The scheme's whole contract. The coarse head only SHORTLISTS; the 32
    // survivors are re-scored through the real head's own rows, so the pick is
    // exact given the shortlist — and matches the full-vocab argmax exactly
    // when the coarse pass kept it. A miss costs ACCEPTANCE (the trunk verify
    // rejects a bad draft), never output, which is why the bar is conditional
    // rather than an equality on every draw.
    //
    // The bar is on the drafted token's exact LOGIT, not its id. A random
    // small head ties constantly — 256-wide dot products of bf16-exact levels
    // through an 8-bit quantizer collide, and the tie rate is scale-invariant
    // — and under a tie the host's first-max and a GPU argmax over the
    // shortlist's own ordering disagree on the id while agreeing on the value.
    // Two ids with the same logit are the same draft as far as acceptance is
    // concerned; convicting on the id would be judging our output against our
    // output's ordering. Id equality is still demanded wherever the reference's
    // OWN margin says the winner is unique.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    // Vocab must clear TOP32_MIN_ROWS with a ragged tail, like the top32 test.
    var fx = try RerankFixture.init(s, TOP32_MIN_ROWS + 96, 256, 8, 64, 0x5EED);
    defer fx.deinit();

    const coarse = buildRerankCoarse(s, &fx.xfm, 3) orelse return error.CoarseBuildDeclined;
    var slot: ?RerankCoarse = coarse;
    defer if (slot) |*q| q.deinit();
    var logged = false;

    // Two different fp paths read the same rows (a full-width qmm vs a 32-row
    // gathered one), so equality between them is approximate by construction.
    const eps: f32 = 1e-3;
    var kept: u32 = 0;
    var decisive: u32 = 0;
    var draws: u32 = 0;
    var seed: u64 = 1;
    while (seed <= 8) : (seed += 1) {
        const x = try fx.randomX(s, seed);
        defer _ = mlx.mlx_array_free(x);

        const exact = try fx.exactLogits(s, x);
        defer _ = mlx.mlx_array_free(exact);
        const ex = try readRow(allocator, s, exact, @intCast(fx.vocab));
        defer allocator.free(ex);
        var want: u32 = 0;
        for (ex, 0..) |v, i| if (v > ex[want]) {
            want = @intCast(i);
        };

        const picked_arr = (try rerankSelect(s, &fx.xfm, &slot, &logged, x, null)) orelse
            return error.RerankDeclined;
        defer _ = mlx.mlx_array_free(picked_arr);
        const picked = try readIdScalar(picked_arr);
        draws += 1;

        // What the coarse pass actually shortlisted, read the same way the
        // scheme reads it.
        var coarse_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(coarse_logits);
        try mlx.check(mlx.mlx_quantized_matmul(
            &coarse_logits,
            x,
            slot.?.q.w,
            slot.?.q.s,
            slot.?.q.b,
            true,
            mlx.mlx_optional_int.some(64),
            mlx.mlx_optional_int.some(3),
            "affine",
            s,
        ));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const flat_shape = [_]c_int{fx.vocab};
        try mlx.check(mlx.mlx_reshape(&flat, coarse_logits, &flat_shape, 1, s));
        const cands = try draftTop32(s, flat, fx.vocab);
        defer _ = mlx.mlx_array_free(cands);
        try mlx.check(mlx.mlx_array_eval(cands));
        const cand_ptr = mlx.mlx_array_data_uint32(cands) orelse return error.NoData;

        var picked_in_list = false;
        var best_in_list: u32 = cand_ptr[0];
        var runner_up: f32 = -std.math.inf(f32);
        for (cand_ptr[0..32]) |c| {
            if (c == picked) picked_in_list = true;
            if (ex[c] > ex[best_in_list]) best_in_list = c;
        }
        // The reference's OWN margin: the gap between the shortlist's best and
        // the best OTHER value in it. Zero means the winner is not unique and
        // no id assertion is decidable.
        for (cand_ptr[0..32]) |c| {
            if (c == best_in_list) continue;
            if (ex[c] > runner_up) runner_up = ex[c];
        }
        const unique = (ex[best_in_list] - runner_up) > eps;

        // Unconditional: the pick is always FROM the shortlist and always
        // carries the shortlist's best exact logit. That is the re-scoring
        // step's own bar and it holds whether or not the coarse pass kept the
        // true argmax.
        try testing.expect(picked_in_list);
        try testing.expectApproxEqAbs(ex[best_in_list], ex[picked], eps);
        // Under a unique winner the id is decidable, so demand it.
        if (unique) {
            decisive += 1;
            try testing.expectEqual(best_in_list, picked);
        }

        // Conditional: the shortlist kept a row scoring the global max => the
        // drafted token scores the global max. This is the property acceptance
        // actually depends on — the trunk verify compares TOKENS, and two rows
        // with the same logit are the same draft.
        if (@abs(ex[best_in_list] - ex[want]) <= eps) {
            try testing.expectApproxEqAbs(ex[want], ex[picked], eps);
            kept += 1;
        }
    }
    // A coarse head that never keeps the argmax would satisfy every assertion
    // above while proposing garbage every step — the acceptance collapse this
    // change would otherwise ship silently.
    try testing.expect(kept * 2 >= draws);
    // And a fixture that ties on every draw would make the id bar vacuous.
    try testing.expect(decisive > 0);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "mtp: the coarse rerank head drafts at its BUILT width, not the env's current one" {
    // The width a coarse head is READ at is a property of the BUILT weights.
    // `rerankCoarseBits()` is the env-backed BUILD default; re-reading it per
    // draft step means a head packed at one width gets dispatched at another
    // the moment the variable moves — a retune between build and draft, or any
    // caller passing `buildRerankCoarse` an explicit width, which both the
    // qwen4 and MiMo arms do. Packed geometry implies K, so a mismatched
    // width does not degrade gracefully: it reads a different vocab entirely.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    var fx = try RerankFixture.init(s, TOP32_MIN_ROWS + 96, 256, 8, 64, 0x5EED);
    defer fx.deinit();

    // Built at 3 bits...
    const coarse = buildRerankCoarse(s, &fx.xfm, 3) orelse return error.CoarseBuildDeclined;
    var slot: ?RerankCoarse = coarse;
    defer if (slot) |*q| q.deinit();
    // ...and the head SAYS so, without asking the environment.
    try testing.expectEqual(@as(u32, 3), slot.?.bits);
    try testing.expectEqual(@as(u32, 64), slot.?.group_size);
    try testing.expectEqual(fx.vocab, slot.?.rows);

    // ...while the environment now says 2. Every draft below must still be
    // dispatched at 3.
    _ = setenv("SUSHI_MTP_DRAFT_HEAD_BITS", "2", 1);
    defer _ = unsetenv("SUSHI_MTP_DRAFT_HEAD_BITS");
    try testing.expectEqual(@as(u32, 2), rerankCoarseBits());

    var logged = false;
    const eps: f32 = 1e-3;
    var kept: u32 = 0;
    var draws: u32 = 0;
    var seed: u64 = 1;
    while (seed <= 8) : (seed += 1) {
        const x = try fx.randomX(s, seed);
        defer _ = mlx.mlx_array_free(x);

        const exact = try fx.exactLogits(s, x);
        defer _ = mlx.mlx_array_free(exact);
        const ex = try readRow(allocator, s, exact, @intCast(fx.vocab));
        defer allocator.free(ex);
        var want: u32 = 0;
        for (ex, 0..) |v, i| if (v > ex[want]) {
            want = @intCast(i);
        };

        const picked_arr = (try rerankSelect(s, &fx.xfm, &slot, &logged, x, null)) orelse
            return error.RerankDeclined;
        defer _ = mlx.mlx_array_free(picked_arr);
        const picked = try readIdScalar(picked_arr);
        draws += 1;

        // The shortlist the 3-bit head actually produces — read at 3 bits,
        // which is the whole point.
        var coarse_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(coarse_logits);
        try mlx.check(mlx.mlx_quantized_matmul(
            &coarse_logits,
            x,
            slot.?.q.w,
            slot.?.q.s,
            slot.?.q.b,
            true,
            mlx.mlx_optional_int.some(64),
            mlx.mlx_optional_int.some(3),
            "affine",
            s,
        ));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const flat_shape = [_]c_int{fx.vocab};
        try mlx.check(mlx.mlx_reshape(&flat, coarse_logits, &flat_shape, 1, s));
        const cands = try draftTop32(s, flat, fx.vocab);
        defer _ = mlx.mlx_array_free(cands);
        try mlx.check(mlx.mlx_array_eval(cands));
        const cand_ptr = mlx.mlx_array_data_uint32(cands) orelse return error.NoData;

        var picked_in_list = false;
        var best_in_list: u32 = cand_ptr[0];
        for (cand_ptr[0..32]) |c| {
            if (c == picked) picked_in_list = true;
            if (ex[c] > ex[best_in_list]) best_in_list = c;
        }
        // The draft came from the BUILT head's shortlist and carries its best
        // exact logit. Dispatched at any other width it comes from somewhere
        // else entirely.
        try testing.expect(picked_in_list);
        try testing.expectApproxEqAbs(ex[best_in_list], ex[picked], eps);
        if (@abs(ex[best_in_list] - ex[want]) <= eps) kept += 1;
    }
    // And the 3-bit head is still a USEFUL shortlister — a mis-dispatched read
    // that happened to stay in range would not be.
    try testing.expect(kept * 2 >= draws);
}

test "mtp: batched rerank ids equal N solo rerankSelect" {
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    var fx = try RerankFixture.init(s, TOP32_MIN_ROWS + 96, 256, 8, 64, 0xBA7C);
    defer fx.deinit();
    const coarse = buildRerankCoarse(s, &fx.xfm, 3) orelse return error.CoarseBuildDeclined;
    var slot: ?RerankCoarse = coarse;
    defer if (slot) |*q| q.deinit();
    var logged = false;
    const xa = try fx.randomX(s, 11);
    defer _ = mlx.mlx_array_free(xa);
    const xb = try fx.randomX(s, 23);
    defer _ = mlx.mlx_array_free(xb);
    const sa = (try rerankSelect(s, &fx.xfm, &slot, &logged, xa, null)) orelse return error.SoloRerankDeclined;
    defer _ = mlx.mlx_array_free(sa);
    const sb = (try rerankSelect(s, &fx.xfm, &slot, &logged, xb, null)) orelse return error.SoloRerankDeclined;
    defer _ = mlx.mlx_array_free(sb);
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, xa);
    _ = mlx.mlx_vector_array_append_value(vec, xb);
    var stacked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stacked);
    try mlx.check(mlx.mlx_concatenate_axis(&stacked, vec, 0, s));
    const bat = (try rerankSelectBatched(s, &fx.xfm, &slot, &logged, stacked, null)) orelse return error.BatchedRerankDeclined;
    defer _ = mlx.mlx_array_free(bat);
    var a0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a0);
    var a1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a1);
    try mlx.check(mlx.mlx_slice(&a0, bat, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ 1, 1 }, 2, &[_]c_int{ 1, 1 }, 2, s));
    try mlx.check(mlx.mlx_slice(&a1, bat, &[_]c_int{ 1, 0 }, 2, &[_]c_int{ 2, 1 }, 2, &[_]c_int{ 1, 1 }, 2, s));
    try testing.expectEqual(try readIdScalar(sa), try readIdScalar(a0));
    try testing.expectEqual(try readIdScalar(sb), try readIdScalar(a1));
}

test "mtp: rerankSelect declines without a coarse head, and the full readout is the exact argmax" {
    // The fallback is not decoration: `canRerankDrafts` false, a build refusal
    // and a mid-chain shortlist failure all land here, and the answer must
    // still be a legal draft — the full-vocab argmax, which is what the qwen4
    // head did before this change.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    var fx = try RerankFixture.init(s, TOP32_MIN_ROWS + 96, 256, 8, 64, 0xA11CE);
    defer fx.deinit();

    var none: ?RerankCoarse = null;
    var logged = false;
    const x = try fx.randomX(s, 7);
    defer _ = mlx.mlx_array_free(x);
    try testing.expect((try rerankSelect(s, &fx.xfm, &none, &logged, x, null)) == null);
    try testing.expect(!logged);

    const exact = try fx.exactLogits(s, x);
    defer _ = mlx.mlx_array_free(exact);
    const ex = try readRow(allocator, s, exact, @intCast(fx.vocab));
    defer allocator.free(ex);
    var want: u32 = 0;
    for (ex, 0..) |v, i| if (v > ex[want]) {
        want = @intCast(i);
    };

    const got = try fullReadoutArgmax(s, &fx.xfm, x, null);
    defer _ = mlx.mlx_array_free(got);
    try testing.expectEqual(want, try readIdScalar(got));
}

test "mtp: a suppressed id is never drafted, through the coarse shortlist or the full readout" {
    // The mask goes on the COARSE row, not the re-scored 32: a reserved id
    // that never enters the shortlist cannot be re-scored back into the draft.
    // Both paths are checked because a rerank build refusal silently routes
    // every draft through the other one.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    var fx = try RerankFixture.init(s, TOP32_MIN_ROWS + 96, 256, 8, 64, 0xBEEF);
    defer fx.deinit();
    const x = try fx.randomX(s, 3);
    defer _ = mlx.mlx_array_free(x);

    const exact = try fx.exactLogits(s, x);
    defer _ = mlx.mlx_array_free(exact);
    const ex = try readRow(allocator, s, exact, @intCast(fx.vocab));
    defer allocator.free(ex);
    var want: u32 = 0;
    for (ex, 0..) |v, i| if (v > ex[want]) {
        want = @intCast(i);
    };

    // Suppress exactly the winner.
    const mask_buf = try allocator.alloc(bool, @intCast(fx.vocab));
    defer allocator.free(mask_buf);
    @memset(mask_buf, false);
    mask_buf[want] = true;
    const mask_shape = [_]c_int{fx.vocab};
    const mask = mlx.mlx_array_new_data(mask_buf.ptr, &mask_shape, 1, .bool_);
    defer _ = mlx.mlx_array_free(mask);

    const coarse = buildRerankCoarse(s, &fx.xfm, 3) orelse return error.CoarseBuildDeclined;
    var slot: ?RerankCoarse = coarse;
    defer if (slot) |*q| q.deinit();
    var logged = false;
    const picked_arr = (try rerankSelect(s, &fx.xfm, &slot, &logged, x, mask)) orelse
        return error.RerankDeclined;
    defer _ = mlx.mlx_array_free(picked_arr);
    try testing.expect((try readIdScalar(picked_arr)) != want);

    const full = try fullReadoutArgmax(s, &fx.xfm, x, mask);
    defer _ = mlx.mlx_array_free(full);
    try testing.expect((try readIdScalar(full)) != want);
}

test "mtp: coarse draft width is the env's when set, else the family default; an unreadable value turns it off" {
    try testing.expectEqual(@as(u32, 3), draftHeadBitsFrom(null, 3));
    try testing.expectEqual(@as(u32, 2), draftHeadBitsFrom(null, 2));
    try testing.expectEqual(@as(u32, 3), draftHeadBitsFrom("3", 2));
    try testing.expectEqual(@as(u32, 4), draftHeadBitsFrom("4", 2));
    try testing.expectEqual(@as(u32, 0), draftHeadBitsFrom("0", 2));
    try testing.expectEqual(@as(u32, 0), draftHeadBitsFrom("off", 2));
    try testing.expectEqual(@as(u32, 0), draftHeadBitsFrom("5", 3));
}

test "mtp: rerankCoarseBytes prices the packed weights AND both bf16 group tables" {
    // The build log claims a size; a bill that forgot scales+biases would
    // under-report the qwen4 coarse head by 40 MB.
    // 248320 x 2560 at 3 bits = 238.4 MiB packed, + 2 x (248320 x 40 x 2 B).
    const packed_mib: u64 = 248320 * 2560 * 3 / 8;
    const tables: u64 = 2 * (248320 * (2560 / 64) * 2);
    try testing.expectEqual(packed_mib + tables, rerankCoarseBytes(248320, 2560, 3));
    // 2-bit is exactly two thirds of the packed half, tables unchanged.
    try testing.expectEqual(248320 * 2560 * 2 / 8 + tables, rerankCoarseBytes(248320, 2560, 2));
}

test "mtp: draftTop32 matches a host top-32 reference (bf16, ties, -inf)" {
    const allocator = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();

    // Two widths: the guard edge (just past TILES*TG with a ragged tail) and
    // the live qwen4 vocab.
    const counts = [_]c_int{ TOP32_MIN_ROWS + 7, 248320 };
    var prng = std.Random.DefaultPrng.init(0x7031);
    const rand = prng.random();

    for (counts) |n| {
        const un: usize = @intCast(n);
        // bf16-exact values with deliberate duplicates (ties) and a few -inf
        // (the suppress-mask spelling).
        const vals = try allocator.alloc(f32, un);
        defer allocator.free(vals);
        for (vals) |*v| {
            const level: f32 = @floatFromInt(rand.intRangeAtMost(i32, -512, 511));
            v.* = level * 0.125; // exact in bf16
        }
        vals[0] = -std.math.inf(f32);
        vals[un / 2] = -std.math.inf(f32);

        const shape = [_]c_int{n};
        const row_f32 = mlx.mlx_array_new_data(vals.ptr, &shape, 1, .float32);
        defer _ = mlx.mlx_array_free(row_f32);
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_astype(&row, row_f32, .bfloat16, s));

        const ids = try draftTop32(s, row, n);
        defer _ = mlx.mlx_array_free(ids);
        try mlx.check(mlx.mlx_array_eval(ids));
        const ids_ptr = mlx.mlx_array_data_uint32(ids) orelse return error.NoData;

        // Host reference: (value asc, index asc), take the tail 32.
        const Entry = struct { v: f32, i: u32 };
        const entries = try allocator.alloc(Entry, un);
        defer allocator.free(entries);
        for (vals, 0..) |v, i| entries[i] = .{ .v = v, .i = @intCast(i) };
        std.mem.sort(Entry, entries, {}, struct {
            fn lt(_: void, a: Entry, b: Entry) bool {
                if (a.v != b.v) return a.v < b.v;
                return a.i < b.i;
            }
        }.lt);

        var expect_set = std.AutoHashMap(u32, void).init(allocator);
        defer expect_set.deinit();
        for (entries[un - 32 ..]) |e| try expect_set.put(e.i, {});

        for (0..32) |k| {
            const id = ids_ptr[k];
            try testing.expect(expect_set.contains(id));
            _ = expect_set.remove(id); // no duplicates either
        }
        try testing.expectEqual(@as(u32, 0), expect_set.count());
    }
}

test "adaptiveDepthCapForMachine names the row it applied" {
    // The cap must be nameable at runtime or an M1 Pro's depth=4 is
    // indistinguishable from the EV controller having chosen 4 by itself —
    // same reason `dflash.blockCapForMachine` carries a label.
    try testing.expectEqualStrings("m1-pro", adaptiveDepthCapForMachine("Apple M1 Pro", 6).label);
    try testing.expectEqualStrings("m4-max", adaptiveDepthCapForMachine("Apple M4 Max", 6).label);
    try testing.expectEqualStrings("default", adaptiveDepthCapForMachine("", 6).label);
}

test "base M4 caps at 4 where M4 Pro/Max keep the default (2026-08-22 sweep)" {
    // Depth 4 is the only width where the cap BINDS, collapsing the plan to
    // one chunk; from 5 on every round pays an extension sync for ~1 more
    // accepted token and loses 17%. The probe's cost cliff says 6 here.
    try testing.expectEqual(@as(u32, 4), adaptiveDepthCapForMachine("Apple M4", 6).cap);
    try testing.expectEqualStrings("m4-base", adaptiveDepthCapForMachine("Apple M4", 6).label);
    try testing.expectEqual(@as(u32, 6), adaptiveDepthCapForMachine("Apple M4 Pro", 6).cap);
    try testing.expectEqual(@as(u32, 6), adaptiveDepthCapForMachine("Apple M4 Max", 6).cap);
    // Every row above is a HUMAN sweep and outranks the boot probe.
    for ([_][]const u8{ "Apple M4", "Apple M4 Max", "Apple M1 Pro", "Apple M5" }) |c| {
        try testing.expect(adaptiveDepthCapForMachine(c, 6).measured);
    }
    try testing.expect(!adaptiveDepthCapForMachine("Apple M2 Pro", 6).measured);
}

test "adaptiveDepthCapForMachine: measured rows only, unmeasured chips keep the default" {
    try testing.expectEqual(@as(u32, 4), adaptiveDepthCapForMachine("Apple M1 Pro", 6).cap);
    try testing.expectEqual(@as(u32, 6), adaptiveDepthCapForMachine("Apple M1 Max", 6).cap);
    try testing.expectEqual(@as(u32, 6), adaptiveDepthCapForMachine("", 6).cap);
}

test "adaptiveDepthCapForMachine: base M5 caps at 4, Pro/Max/Ultra keep the default" {
    try testing.expectEqual(@as(u32, 4), adaptiveDepthCapForMachine("Apple M5", 6).cap);
    try testing.expectEqualStrings("m5", adaptiveDepthCapForMachine("Apple M5", 6).label);
    try testing.expectEqual(@as(u32, 6), adaptiveDepthCapForMachine("Apple M5 Pro", 6).cap);
    try testing.expectEqual(@as(u32, 6), adaptiveDepthCapForMachine("Apple M5 Max", 6).cap);
    try testing.expectEqual(@as(u32, 6), adaptiveDepthCapForMachine("Apple M5 Ultra", 6).cap);
}

test "mtpCtxWithinLimit: 0 is unlimited and the ceiling is inclusive" {
    try testing.expect(mtpCtxWithinLimit(0, 0));
    try testing.expect(mtpCtxWithinLimit(0, 1_000_000));

    try testing.expect(mtpCtxWithinLimit(4096, 4095));
    try testing.expect(mtpCtxWithinLimit(4096, 4096));
    try testing.expect(!mtpCtxWithinLimit(4096, 4097));
    try testing.expect(!mtpCtxWithinLimit(4096, 1_000_000));

    try testing.expect(mtpCtxWithinLimit(1, 1));
    try testing.expect(!mtpCtxWithinLimit(1, 2));
}

test "mtp: row-axis coarse logits equal each solo readout" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var fx = try RerankFixture.init(s, TOP32_MIN_ROWS + 96, 2560, 8, 64, 0x3B17);
    defer fx.deinit();
    var rc = buildRerankCoarse(s, &fx.xfm, 3) orelse return error.CoarseBuildDeclined;
    defer rc.deinit();
    const tx = @import("transformer.zig");
    {
        for ([_]usize{ 2, 4, 8, 17 }) |n| {
            var parts: [32]mlx.mlx_array = @splat(.{ .ctx = null });
            defer for (parts[0..n]) |x| {
                if (x.ctx != null) _ = mlx.mlx_array_free(x);
            };
            for (0..n) |i| parts[i] = try fx.randomX(s, 17 + i);
            const vec = mlx.mlx_vector_array_new_data(parts[0..n].ptr, n);
            defer _ = mlx.mlx_vector_array_free(vec);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_concatenate_axis(&x, vec, 0, s));
            const before = tx.mtp_head_row_dispatches;
            const got = try rerankCoarseRows(s, x, &rc);
            defer _ = mlx.mlx_array_free(got);
            try testing.expectEqual(@as(usize, if (n <= 16) 1 else 0), tx.mtp_head_row_dispatches - before);
            for (parts[0..n], 0..) |input, i| {
                var want = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(want);
                try mlx.check(mlx.mlx_quantized_matmul(&want, input, rc.q.w, rc.q.s, rc.q.b, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(3), "affine", s));
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, got, &[_]c_int{ @intCast(i), 0, 0 }, 3, &[_]c_int{ @intCast(i + 1), 1, rc.rows }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
                var same = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(same);
                try mlx.check(mlx.mlx_array_equal(&same, row, want, false, s));
                try mlx.check(mlx.mlx_array_eval(same));
                var equal: bool = false;
                try mlx.check(mlx.mlx_array_item_bool(&equal, same));
                if (!equal) std.debug.print("coarse logits differ: N={d} row={d}\n", .{ n, i });
                try testing.expect(equal);
            }
        }
    }
}
