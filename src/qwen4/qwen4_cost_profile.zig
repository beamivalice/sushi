//! Flash-Next MTP cost profiles: the runtime fingerprint of the packs the G17 surfaces were measured on.
const mtp_mod = @import("../mtp.zig");
const mtp_policy = @import("../mtp_policy.zig");
const MtpEvCosts = mtp_policy.MtpEvCosts;
const MtpCostProfile = mtp_mod.MtpCostProfile;
const trunk_mod = @import("../trunk.zig");
const Trunk = trunk_mod.Trunk;
const arch_state = @import("../arch/arch_state.zig");
const expert_quant = @import("../expert_quant.zig");
const log = @import("../log.zig");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const qwen4_state = @import("qwen4_state.zig");
const std = @import("std");
const testing = std.testing;

/// Pure first-stage classifier for the qwen4_exp in-checkpoint head.
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
    if (mlx.noGpuBackend() or !trunk_mod.verifyQmmNaxAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(path));
    defer if (config.ngram_table_path) |name| a.free(name);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(path));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const old_kv_flag = Trunk.mtp_head_kv_quant_flag;
    defer Trunk.mtp_head_kv_quant_flag = old_kv_flag;
    Trunk.mtp_head_kv_quant_flag = true;
    try xfm.cache.reinit(config.num_hidden_layers, trunk_mod.KVQuantConfig.affine(8));
    try arch_state.applyHeadKvQuant(&xfm, trunk_mod.KVQuantConfig.affine(8));
    try testing.expect(arch_state.buildDraftRerank(&xfm));
    try testing.expectEqual(MtpCostProfile.g17_nax_qwen4_mixed_4_8_gs64, qwen4G17CostProfile(&xfm));
    try testing.expectEqual(MtpCostProfile.generic, qwen4G17CostProfileForKv(&xfm, trunk_mod.KVQuantConfig.dense));
    try testing.expectEqual(MtpCostProfile.generic, qwen4G17CostProfileForKv(&xfm, trunk_mod.KVQuantConfig.affine(4)));
    const head = qwen4_state.mtpOf(&xfm).?;
    const old_bias = head.fc_emb_b;
    defer head.fc_emb_b = old_bias;
    head.fc_emb_b = .{};
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

fn qwen4MixedModelPacksMatch(target: *const Trunk) bool {
    const q4 = qwen4_state.stateOfConst(target);
    if (q4.model == null or q4.stream_f32 or target.config.quant_mode != .affine or target.config.quant_group_size != 64) return false;
    const cfg = &target.config;
    if (cfg.linear_key_head_dim != 128 or cfg.linear_value_head_dim != 128 or cfg.linear_num_key_heads != 16 or
        cfg.linear_num_value_heads != 48 or cfg.linear_conv_kernel_dim != 4 or cfg.ple_embed_dim != 2560) return false;
    const head = if (q4.mtp) |*h| h else return false;
    if (!qwen4MixedPackMatches(target.emb_w, target.emb_s, target.emb_b, &.{ 248320, 2560 }, 4) or
        !qwen4MixedPackMatches(target.lm_head_w, target.lm_head_s, target.lm_head_b, &.{ 248320, 2560 }, 8) or
        !qwen4MixedPackMatches(head.fc_emb_w, head.fc_emb_s, head.fc_emb_b, &.{ 2560, 2560 }, 8) or
        !qwen4MixedPackMatches(head.fc_hid_w, head.fc_hid_s, head.fc_hid_b, &.{ 2560, 2560 }, 8)) return false;
    if (!qwen4MixedHcMatches(q4.mixer orelse return false) or !qwen4MixedHcMatches(head.mixer) or
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
pub fn qwen4G17CostProfile(target: *const Trunk) MtpCostProfile {
    return qwen4G17CostProfileForKv(target, target.cache.config);
}

/// Mixed-pack prices use the request's actual KV format, including a slot
/// override. The original all-4-bit classifier retains its existing scope.
pub fn qwen4G17CostProfileForKv(target: *const Trunk, kv: trunk_mod.KVQuantConfig) MtpCostProfile {
    if (target.config.expert_layout == .exl3_k4) return .generic;
    const head = qwen4_state.mtpOfConst(target) orelse return .generic;
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
    const mixed = qwen4MixedCostProfileForFingerprint(geometry, std.meta.eql(geometry, Qwen4MixedGeometry{}) and qwen4MixedModelPacksMatch(target), rerank_ready, mlx.streamIsGpu(target.s) and trunk_mod.verifyQmmNaxAvailable());
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
        trunk_mod.verifyQmmNaxAvailable(),
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

/// qwen4_exp (Qwen3.8-Flash-Next 125B-A6B) on G17, M5 Max 40-core refit
/// (2026-08-27/28). Fitted at ~8.5k context like every prior surface
/// (a short-context fit planned too shallow at 8.5k, measured -4% —
/// every constant this controller ever shipped was fitted at ONE
/// context length, and 8K is the house one). Forced-depth saturated
/// echo (the test_mtp_equivalence echo appended to ~8.3k of neutral
/// filler, temp 0, `--prefix-cache-entries 0`, round-cost persist off),
/// depths {1,2,3,4,6} x two reversed passes x 3 reps, medians of
/// per-request round_ms: T(1)=25.41, T(2)=31.18, T(3)=35.70,
/// T(4)=41.33, T(6)=53.58 ms -> 20.34 ms floor, composite
/// (draft + per-position) marginals .257/.303 (SSE 0.28), shipped
/// house-style below as `.draft = 0.02` plus per-position .237/.283 —
/// composite minus draft, the same split every calibrated surface
/// carries. Short-context fit for the record: T(1..4,6,8)=
/// 23.61/28.12/33.91/38.87/50.16/62.70 -> floor 18.24, composite
/// .283/.310/.344 (the .344 ships as per-position `.325`).
/// A verify row on this arch is BYTES — a second row reads its own
/// experts — so the marginals sit 3-5x above the dense-sidecar surfaces
/// and there is NO NAX flattening: routed expert banks never ride the
/// vqmm NAX lane (that is the M5 plan's grouped-expert item, and this
/// table is its before-number). `per_pos_nax` carries the short-context
/// k>=7 STEEPENING (+11%/pos, a measurement not a discount) and is
/// inert under this profile's default cap of 6. sync stays nominal per
/// the dry-spell-gate doctrine (measured chunk-boundary sync ~1.7 ms is
/// governed by the realized-rate gate, not the prior). draft/verify
/// split arbitrary but positive; only the sums enter the controller.
pub const MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS: MtpEvCosts = .{
    .draft = 0.02,
    .per_pos_lo = 0.237,
    .per_pos_hi = 0.283,
    .flat_max = 4,
    .sync = 0.02,
    .nax_from = 7,
    .per_pos_nax = 0.325,
};

/// 2026-09-13 M5 Max mixed4/8 refit after the single-stream verifier
/// fast paths. Affine8 target/head KV, 3-bit draft shortlist, ~8.6k
/// context; forced depths {1,2,3,4,5,6,8}, three echo repetitions in
/// each of two reversed boot orders. Median full-round EMAs (ms):
/// 24.145/28.970/34.035/38.470/44.310/51.970/61.600.
/// Fit floor=19.037 and position slopes=4.983/8.018/4.815 ms;
/// max relative residual 1.30%. Depth6 verifies seven positions and
/// leaves the optimized width2..6 envelope, hence flat_max=5.
/// Only composite marginals are identifiable; retain draft=.02 and
/// nominal sync=.02. Runtime fingerprint and KV guards live in mtp.zig.
/// Evidence and reproduction: notes/MTP-SINGLE-VERIFY.md.
pub const MTP_EV_G17_NAX_QWEN4_MIXED_4_8_GS64_COSTS: MtpEvCosts = .{
    .draft = 0.02,
    .per_pos_lo = 0.241753,
    .per_pos_hi = 0.401180,
    .flat_max = 5,
    .sync = 0.02,
    .nax_from = 7,
    .per_pos_nax = 0.232929,
};

/// The EV cost surface of each Flash-Next profile.
pub fn evCosts(profile: MtpCostProfile) ?MtpEvCosts {
    return switch (profile) {
        .g17_nax_qwen4_q4_gs64 => MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS,
        .g17_nax_qwen4_mixed_4_8_gs64 => MTP_EV_G17_NAX_QWEN4_MIXED_4_8_GS64_COSTS,
        else => null,
    };
}

test "mtpEvCostsForProfile: G17 profiles are explicit" {
    const generic = mtp_policy.mtpEvCostsForProfile(.generic);
    try std.testing.expectEqual(mtp_policy.MTP_EV_DEFAULT_COSTS, generic);

    const qwen4 = mtp_policy.mtpEvCostsForProfile(.g17_nax_qwen4_q4_gs64);
    try std.testing.expectEqual(MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS, qwen4);
    try std.testing.expect(!std.meta.eql(generic, qwen4));
}
