//! Hy4-Flash's tensor census: every checkpoint name mapped to the role the loader reads it as, checked against the
//! geometry (a layer past the count, an indexer on a sliding layer, experts on the dense layer or Engram where the config
//! puts none are refused by name, as is any name the release does not ship). MLX-free.
const std = @import("std");
const log = @import("../log.zig");
const hy4_config = @import("hy4_config.zig");

pub const Role = enum {
    embedding,
    lm_head,
    final_norm,
    /// The head's hyper-connection collapse (`model.hc_head.*`).
    hc_head,
    /// A layer's two pre-norms.
    layer_norm,
    /// The attention and FFN hyper-connection mixes (`hc_{attn,mlp}_layer.hc_pre.*`).
    hc_mix,
    attention,
    indexer,
    dense_ffn,
    router,
    routed_experts,
    /// The routed experts' shared down/up projections into and out of their 1,536-wide latent.
    latent_proj,
    shared_experts,
    /// `mlp.combine_output_norm`, and `post_fc2_norm` on the dense layer.
    ffn_norm,
    engram_table,
    /// Engram's token map, projections, norms and mix scales: everything of a module but its table.
    engram,
    dfly2,
    vision,
};

/// The parts the arch doc's storage table sums.
pub const Part = enum { routed, engram_table, sliding_attn, sparse_attn, dfly2, shared, vision, latent, embedding, lm_head, dense, small };

pub const Entry = struct {
    role: Role,
    /// The backbone layer, for a `model.layers.N.` name.
    layer: ?u32 = null,

    pub fn part(self: Entry, c: *const hy4_config.Config) Part {
        return switch (self.role) {
            .routed_experts => .routed,
            .engram_table => .engram_table,
            .attention => if (c.kinds[self.layer.?] == .sparse) .sparse_attn else .sliding_attn,
            .dfly2 => .dfly2,
            .shared_experts => .shared,
            .vision => .vision,
            .latent_proj => .latent,
            .embedding => .embedding,
            .lm_head => .lm_head,
            .dense_ffn => .dense,
            .final_norm, .hc_head, .layer_norm, .hc_mix, .indexer, .router, .ffn_norm, .engram => .small,
        };
    }
};

pub const Error = error{UnknownHy4Tensor};

/// The role of checkpoint tensor `name` under geometry `c`.
pub fn roleOf(c: *const hy4_config.Config, name: []const u8) Error!Entry {
    // An STQ pack (docs/pack-format.md#hy4-stq): a routed rate group's bank, or a trellis bank standing in for the
    // release's `<base>.weight`, which takes that weight's place.
    if (std.mem.indexOf(u8, name, ".mlp.switch_mlp.")) |at_sw| {
        const leaf = name[at_sw + ".mlp.switch_mlp.".len ..];
        if (!stqExpertLeaf(leaf)) return refuse(name, "not a routed rate-group bank");
        var base_buf: [160]u8 = undefined;
        const base = std.fmt.bufPrint(&base_buf, "{s}.mlp.experts.down_proj", .{name[0..at_sw]}) catch return refuse(name, "too long");
        return roleOf(c, base);
    }
    inline for (.{ ".trellis", ".suh", ".svh" }) |suffix| if (std.mem.endsWith(u8, name, suffix)) {
        var base_buf: [168]u8 = undefined;
        const base = std.fmt.bufPrint(&base_buf, "{s}.weight", .{name[0 .. name.len - suffix.len]}) catch return refuse(name, "too long");
        return roleOf(c, base);
    };
    if (std.mem.startsWith(u8, name, "vit.")) {
        if (c.vision == null) return refuse(name, "the config declares no vision tower");
        return .{ .role = .vision };
    }
    if (std.mem.startsWith(u8, name, "model.dfly_layers.")) return dfly(c, name);
    if (std.mem.eql(u8, name, "lm_head.weight")) return .{ .role = .lm_head };
    if (std.mem.eql(u8, name, "model.embed_tokens.weight")) return .{ .role = .embedding };
    if (std.mem.eql(u8, name, "model.norm.weight")) return .{ .role = .final_norm };
    if (oneOf(name, "model.hc_head.", &.{ "hc_head_fn", "hc_head_base", "hc_head_scale" })) return .{ .role = .hc_head };
    const rest = stripPrefix(name, "model.layers.") orelse return refuse(name, "not a name the release ships");
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return refuse(name, "no layer index");
    const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return refuse(name, "no layer index");
    if (layer >= c.layers) return refuse(name, "a layer past the config's count");
    const r = rest[dot + 1 ..];
    const at: Entry = .{ .role = .attention, .layer = layer };
    const sparse = c.kinds[layer] == .sparse;
    const dense = layer < c.dense_layers;
    if (oneOf(r, "", &.{ "input_layernorm.weight", "post_attention_layernorm.weight" })) return with(at, .layer_norm);
    if (oneOf(r, "hc_attn_layer.hc_pre.", &.{ "hc_fn", "hc_base", "hc_scale" }) or oneOf(r, "hc_mlp_layer.hc_pre.", &.{ "hc_fn", "hc_base", "hc_scale" }))
        return with(at, .hc_mix);
    if (stripPrefix(r, "self_attn.compressed_indexer.")) |ix| {
        if (!sparse) return refuse(name, "an indexer on a sliding layer");
        if (!oneOf(ix, "", &indexer_names)) return refuse(name, "not a name the release ships");
        return with(at, .indexer);
    }
    if (stripPrefix(r, "self_attn.")) |a| {
        if (!oneOf(a, "", &attention_names)) return refuse(name, "not a name the release ships");
        return at;
    }
    if (stripPrefix(r, "engram_layer.")) |e| {
        if (!engramAt(c, layer)) return refuse(name, "Engram where the config puts no module");
        if (std.mem.eql(u8, e, "multi_head_embedding.embedding.weight")) return with(at, .engram_table);
        if (!engramName(c, e)) return refuse(name, "not a name the release ships");
        return with(at, .engram);
    }
    const m = stripPrefix(r, "mlp.") orelse return refuse(name, "not a name the release ships");
    if (dense) {
        if (oneOf(m, "", &.{ "gate_proj.weight", "up_proj.weight", "down_proj.weight" })) return with(at, .dense_ffn);
        if (std.mem.eql(u8, m, "post_fc2_norm.weight")) return with(at, .ffn_norm);
        return refuse(name, "an MoE tensor on a dense layer");
    }
    if (oneOf(m, "experts.", &.{ "gate_up_proj", "down_proj" })) return with(at, .routed_experts);
    if (oneOf(m, "", &.{ "routed_expert_down_proj.weight", "routed_expert_up_proj.weight" })) return with(at, .latent_proj);
    if (oneOf(m, "shared_experts.", &.{ "gate_proj.weight", "up_proj.weight", "down_proj.weight" })) return with(at, .shared_experts);
    if (oneOf(m, "gate.", &.{ "weight", "e_score_correction_bias" })) return with(at, .router);
    if (std.mem.eql(u8, m, "combine_output_norm.weight")) return with(at, .ffn_norm);
    return refuse(name, if (oneOf(m, "", &.{ "gate_proj.weight", "up_proj.weight", "down_proj.weight", "post_fc2_norm.weight" })) "a dense FFN tensor on an MoE layer" else "not a name the release ships");
}

/// A tower matrix as its `[out, in]` linear: each block's attention projections and MLP, and the merger's two
/// convolutions over their (C, kh, kw) values. Null for any other tensor (the patch embedding is not one).
pub fn visionMatrix(v: hy4_config.Vision, name: []const u8) ?[2]u64 {
    const d: u64 = v.hidden;
    const cells = d * v.merge * v.merge;
    if (std.mem.eql(u8, name, "vit.perceive.proj.0.weight")) return .{ cells, cells };
    if (std.mem.eql(u8, name, "vit.perceive.proj.2.weight")) return .{ v.out_hidden, cells };
    const rest = stripPrefix(name, "vit.layers.") orelse return null;
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
    if (layer >= v.layers) return null;
    const leaf = rest[dot + 1 ..];
    if (oneOf(leaf, "self_attn.", &.{ "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight" })) return .{ d, d };
    if (std.mem.eql(u8, leaf, "mlp.dense_h_to_4h.weight")) return .{ v.inter, d };
    if (std.mem.eql(u8, leaf, "mlp.dense_4h_to_h.weight")) return .{ d, v.inter };
    return null;
}

/// A tower matrix a pack stores as MLX affine codes at `hy4_config.visionGroup` (docs/pack-format.md#hy4-stq): every
/// `visionMatrix` whose input the group divides. Null for any other tensor.
pub fn visionAffine(v: hy4_config.Vision, name: []const u8) ?[2]u64 {
    const g = hy4_config.visionGroup(v.hidden) orelse return null;
    const m = visionMatrix(v, name) orelse return null;
    return if (m[1] % g == 0) m else null;
}

/// A tower matrix an affine pack keeps BF16 because no MLX group divides its input (the release's 4304-wide MLP down
/// projections); `vision_quant.bf16_kept` names each one.
pub fn visionKept(v: hy4_config.Vision, name: []const u8) bool {
    return visionMatrix(v, name) != null and visionAffine(v, name) == null;
}

const attention_names = [_][]const u8{
    "q_a_proj.weight",       "q_a_layernorm.weight", "q_b_proj.weight", "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight", "kv_b_proj.weight",     "o_proj.weight",   "linear_gate.weight",
    "post_proj_norm.weight", "learnable_sink_param",
};

const indexer_names = [_][]const u8{
    "query_projection.weight",    "token_key_projection.weight", "token_key_norm.weight",         "key_value_projection.weight",
    "key_gate_projection.weight", "key_norm.weight",             "head_weight_projection.weight", "intra_group_ape",
};

const dfly_layer_names = attention_names ++ [_][]const u8{
    "input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
};

const dfly_top_names = [_][]const u8{
    "final_norm.weight",                   "confidence_head.proj.weight",          "confidence_head.proj.bias",
    "hidden_correction.embed_norm.weight", "hidden_correction.hidden_norm.weight", "hidden_correction.gate_proj.weight",
    "hidden_correction.up_proj.weight",    "hidden_correction.down_proj.weight",   "mtp_fusion.eh_proj.weight",
    "mtp_fusion.enorm.weight",             "mtp_fusion.hnorm.weight",              "mtp_fusion.gate",
};

fn dfly(c: *const hy4_config.Config, name: []const u8) Error!Entry {
    const rest = name["model.dfly_layers.".len..];
    if (oneOf(rest, "", &dfly_top_names)) return .{ .role = .dfly2 };
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return refuse(name, "not a name the release ships");
    const i = std.fmt.parseInt(u32, rest[0..dot], 10) catch return refuse(name, "not a name the release ships");
    if (i >= c.dfly.layers) return refuse(name, "a DFly2 layer past the config's count");
    const r = rest[dot + 1 ..];
    if (stripPrefix(r, "self_attn.")) |a| if (oneOf(a, "", &attention_names)) return .{ .role = .dfly2 };
    if (oneOf(r, "", dfly_layer_names[attention_names.len..])) return .{ .role = .dfly2 };
    return refuse(name, "not a name the release ships");
}

fn engramAt(c: *const hy4_config.Config, layer: u32) bool {
    for (c.engram.layers[0..c.engram.n]) |l| if (l == layer) return true;
    return false;
}

/// An Engram module's tensors but its table: per-head norms for each of its heads, and per hyper-connection stream a
/// key projection, key and query norm and a mix scale.
fn engramName(c: *const hy4_config.Config, e: []const u8) bool {
    if (oneOf(e, "", &.{ "lookup_table", "unified_pre_gamma", "value_proj.weight", "value_proj.bias" })) return true;
    const families = [_]struct { prefix: []const u8, suffix: []const u8, n: u32 }{
        .{ .prefix = "lookup_per_head_norms.", .suffix = ".weight", .n = c.engram.primeCount() },
        .{ .prefix = "key_projs.", .suffix = ".weight", .n = c.hc_mult },
        .{ .prefix = "key_projs.", .suffix = ".bias", .n = c.hc_mult },
        .{ .prefix = "norm_key.", .suffix = ".weight", .n = c.hc_mult },
        .{ .prefix = "norm_query.", .suffix = ".weight", .n = c.hc_mult },
        .{ .prefix = "unified_per_hc_scales.", .suffix = "", .n = c.hc_mult },
    };
    for (families) |f| {
        const mid = stripPrefix(e, f.prefix) orelse continue;
        if (!std.mem.endsWith(u8, mid, f.suffix)) continue;
        const digits = mid[0 .. mid.len - f.suffix.len];
        const i = std.fmt.parseInt(u32, digits, 10) catch continue;
        if (i < f.n) return true;
    }
    return false;
}

fn with(e: Entry, role: Role) Entry {
    return .{ .role = role, .layer = e.layer };
}

/// `{gate,up,down}_proj.g<N>.{trellis_pm,suh,svh}`: the routed banks are pair-major; a pack-order `trellis` is no name of the census.
fn stqExpertLeaf(leaf: []const u8) bool {
    var it = std.mem.splitScalar(u8, leaf, '.');
    const proj = it.next() orelse return false;
    const group = it.next() orelse return false;
    const part = it.next() orelse return false;
    if (it.next() != null) return false;
    if (!oneOf(proj, "", &.{ "gate_proj", "up_proj", "down_proj" })) return false;
    if (group.len < 2 or group[0] != 'g') return false;
    _ = std.fmt.parseInt(u8, group[1..], 10) catch return false;
    return oneOf(part, "", &.{ "trellis_pm", "suh", "svh" });
}

fn stripPrefix(s: []const u8, prefix: []const u8) ?[]const u8 {
    return if (std.mem.startsWith(u8, s, prefix)) s[prefix.len..] else null;
}

/// `s` is `prefix` followed by exactly one of `names`.
fn oneOf(s: []const u8, prefix: []const u8, names: []const []const u8) bool {
    const r = stripPrefix(s, prefix) orelse return false;
    for (names) |n| if (std.mem.eql(u8, r, n)) return true;
    return false;
}

fn refuse(name: []const u8, why: []const u8) Error {
    if (!@import("builtin").is_test) log.err("hy_v4_flash: tensor {s}: {s}\n", .{ name, why });
    return error.UnknownHy4Tensor;
}

const testing = std.testing;

/// The release's 1,964 tensors as `name dtype dim,dim,...`, one a line, read from its shard headers.
const release_tensors = @embedFile("fixtures/hy4_tensors.txt");

fn dtypeBytes(d: []const u8) !u64 {
    if (std.mem.eql(u8, d, "BF16")) return 2;
    if (std.mem.eql(u8, d, "F32") or std.mem.eql(u8, d, "I32")) return 4;
    return error.TestUnknownDtype;
}

test "hy4 census: every one of the release's 1,964 tensors has a role, and the parts sum to the doc's table" {
    const c = try hy4_config.parseText(hy4_config.release_config);
    var bytes: std.EnumArray(Part, u64) = .initFill(0);
    var roles: std.EnumArray(Role, u32) = .initFill(0);
    var count: u32 = 0;
    var lines = std.mem.tokenizeScalar(u8, release_tensors, '\n');
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const name = f.next().?;
        var n = try dtypeBytes(f.next().?);
        var dims = std.mem.tokenizeScalar(u8, f.next().?, ',');
        while (dims.next()) |d| n *= try std.fmt.parseInt(u64, d, 10);
        const e = roleOf(&c, name) catch |err| {
            std.debug.print("unmapped: {s}\n", .{name});
            return err;
        };
        bytes.getPtr(e.part(&c)).* += n;
        roles.getPtr(e.role).* += 1;
        count += 1;
    }
    try testing.expectEqual(@as(u32, 1964), count);
    // Every role the census names is one the release ships.
    for (std.enums.values(Role)) |r| try testing.expect(roles.get(r) > 0);
    // docs/hy4-flash-arch.md's storage table, to the byte.
    const want = [_]struct { Part, u64 }{
        .{ .routed, 297_158_049_792 },    .{ .engram_table, 126_721_231_872 }, .{ .sliding_attn, 8_222_430_720 },
        .{ .sparse_attn, 1_703_523_840 }, .{ .dfly2, 1_691_372_866 },          .{ .shared, 1_160_773_632 },
        .{ .vision, 933_258_720 },        .{ .latent, 773_849_088 },           .{ .embedding, 742_391_808 },
        .{ .lm_head, 742_391_808 },       .{ .dense, 254_803_968 },            .{ .small, 416_755_796 },
    };
    var total: u64 = 0;
    for (want) |w| {
        try testing.expectEqual(w[1], bytes.get(w[0]));
        total += w[1];
    }
    try testing.expectEqual(@as(u64, 440_520_833_910), total);
}

test "hy4 census: a name the geometry puts nowhere is refused by name" {
    const c = try hy4_config.parseText(hy4_config.release_config);
    const refused = [_][]const u8{
        "model.layers.42.input_layernorm.weight", // past the 42 layers
        "model.layers.2.self_attn.compressed_indexer.query_projection.weight", // layer 2 is sliding
        "model.layers.0.mlp.experts.down_proj", // layer 0 is dense
        "model.layers.3.mlp.gate_proj.weight", // a dense FFN on an MoE layer
        "model.layers.2.engram_layer.lookup_table", // Engram sits at 1, 11, 21, 31
        "model.layers.1.engram_layer.key_projs.4.weight", // four hyper-connection streams
        "model.layers.1.engram_layer.lookup_per_head_norms.16.weight", // sixteen heads
        "model.layers.1.self_attn.q_proj.weight", // MLA has no q_proj
        "model.dfly_layers.5.input_layernorm.weight", // five DFly2 layers
        "model.layers.x.input_layernorm.weight",
        "model.mtp.0.eh_proj.weight",
        "lm_head.bias",
    };
    for (refused) |name| {
        if (roleOf(&c, name)) |e| {
            std.debug.print("took {s} as {t}\n", .{ name, e.role });
            return error.TestUnexpectedRole;
        } else |err| try testing.expectEqual(error.UnknownHy4Tensor, err);
    }
    try testing.expectEqual(Role.indexer, (try roleOf(&c, "model.layers.1.self_attn.compressed_indexer.intra_group_ape")).role);
    try testing.expectEqual(Role.ffn_norm, (try roleOf(&c, "model.layers.0.mlp.post_fc2_norm.weight")).role);
    var no_vision = c;
    no_vision.vision = null;
    try testing.expectError(error.UnknownHy4Tensor, roleOf(&no_vision, "vit.perceive.norm.weight"));
}

test "hy4 census: an STQ pack's banks take the roles of the release's weights they stand for, and a bank no weight stands behind is refused" {
    const c = try hy4_config.parseText(hy4_config.release_config);
    const took = [_]struct { []const u8, Role }{
        .{ "model.layers.3.mlp.switch_mlp.gate_proj.g0.trellis_pm", .routed_experts },
        .{ "model.layers.3.mlp.switch_mlp.down_proj.g12.svh", .routed_experts },
        .{ "model.layers.3.self_attn.q_b_proj.trellis", .attention },
        .{ "model.layers.3.self_attn.o_proj.suh", .attention },
        .{ "model.layers.0.mlp.down_proj.trellis", .dense_ffn },
        .{ "model.layers.3.mlp.routed_expert_up_proj.svh", .latent_proj },
        .{ "model.layers.3.mlp.shared_experts.gate_proj.trellis", .shared_experts },
        .{ "lm_head.trellis", .lm_head },
        .{ "model.embed_tokens.suh", .embedding },
    };
    for (took) |t| try testing.expectEqual(t[1], (try roleOf(&c, t[0])).role);
    const refused = [_][]const u8{
        "model.layers.0.mlp.switch_mlp.gate_proj.g0.trellis_pm", // layer 0 is dense
        "model.layers.3.mlp.switch_mlp.gate_up_proj.g0.trellis_pm", // the pack splits gate_up
        "model.layers.3.mlp.switch_mlp.gate_proj.trellis_pm", // no rate group
        "model.layers.3.mlp.switch_mlp.gate_proj.g0.trellis", // the pack order: migrated by `sushi repack`, never read
        "model.layers.3.mlp.switch_mlp.gate_proj.g0.weight",
        "model.layers.3.self_attn.q_proj.trellis", // MLA has no q_proj
        "model.layers.1.self_attn.learnable_sink_param.trellis", // not a .weight
    };
    for (refused) |name| {
        if (roleOf(&c, name)) |e| {
            std.debug.print("took {s} as {t}\n", .{ name, e.role });
            return error.TestUnexpectedRole;
        } else |err| try testing.expectEqual(error.UnknownHy4Tensor, err);
    }
}

test "hy4 census: the release's tower stores 137 matrices affine 8 at group 128 and keeps its 27 4304-wide down projections BF16 (0.27 GB), 0.63 GB in place of 0.93" {
    const c = try hy4_config.parseText(hy4_config.release_config);
    const v = c.vision.?;
    try testing.expectEqual(@as(?u32, 128), hy4_config.visionGroup(v.hidden));
    var count: u32 = 0;
    var kept: u32 = 0;
    var kept_bytes: u64 = 0;
    var stored: u64 = 0;
    var lines = std.mem.tokenizeScalar(u8, release_tensors, '\n');
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const name = f.next().?;
        if (!std.mem.startsWith(u8, name, "vit.")) continue;
        var n = try dtypeBytes(f.next().?);
        var dims = std.mem.tokenizeScalar(u8, f.next().?, ',');
        var elems: u64 = 1;
        while (dims.next()) |dim| elems *= try std.fmt.parseInt(u64, dim, 10);
        n *= elems;
        if (visionAffine(v, name)) |m| {
            try testing.expectEqual(m[0] * m[1], elems);
            count += 1;
            // A code byte per weight, a BF16 scale and bias per 128 inputs.
            n = m[0] * m[1] + 2 * 2 * m[0] * (m[1] / 128);
        }
        if (visionKept(v, name)) {
            try testing.expect(std.mem.endsWith(u8, name, ".mlp.dense_4h_to_h.weight"));
            kept += 1;
            kept_bytes += n;
        }
        stored += n;
    }
    try testing.expectEqual(@as(u32, 27 * 5 + 2), count);
    try testing.expectEqual(@as(u32, 27), kept);
    try testing.expectEqual(@as(u64, 267_743_232), kept_bytes);
    try testing.expectEqual(@as(u64, 630_438_816), stored);
    try testing.expectEqual(@as(?[2]u64, null), visionAffine(v, "vit.layers.0.mlp.dense_4h_to_h.weight"));
    try testing.expectEqual(@as(?[2]u64, null), visionAffine(v, "vit.layers.27.self_attn.q_proj.weight"));
    try testing.expectEqual(@as(?[2]u64, null), visionAffine(v, "vit.embeddings.patch_embedding.weight"));
    // The tiny's 32-wide tower takes group 32, which its 64-wide down projection takes too.
    const tiny: hy4_config.Vision = .{ .layers = 2, .hidden = 32, .heads = 2, .inter = 64, .patch = 16, .merge = 2, .out_hidden = 128 };
    try testing.expectEqual(@as(?[2]u64, .{ 32, 64 }), visionAffine(tiny, "vit.layers.1.mlp.dense_4h_to_h.weight"));
    try testing.expect(!visionKept(tiny, "vit.layers.1.mlp.dense_4h_to_h.weight"));
}
