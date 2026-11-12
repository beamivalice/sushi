//! The config side of the arch seam: the arch a parsed config carries, each arch's parse, and what a config states
//! about its arch without loading it. An arch declares its part in the `slot:config` lines of its `arch.zig`.
const std = @import("std");
const model = @import("../model.zig");
const registry = @import("registry.zig");

pub const Tag = registry.Tag;

/// The arch of a config, resolved once by the parse. A config built by hand names its arch by `model_type` alone.
/// `model_type` must not be reassigned after the parse: the tag and the string would then disagree.
pub const ArchSpec = struct {
    tag: ?Tag = null,
};

/// Which context grid an arch's round-cost table is measured on (`round_cost.Layout`).
pub const CostLayout = @import("../round_cost.zig").Layout;

/// How a pack stores the routed experts the streamed loader reads; the layout probe of its weight map follows it. The
/// first two are banks under the checkpoint's `language_model.model.layers.` and `model.layers.` nesting.
pub const ExpertSource = enum { none, language_model_banks, flat_banks, individual_source, mxfp4_w123 };

/// Constants an arch states about its config; the defaults are what an arch with nothing special says.
pub const Facts = struct {
    /// A request that names no thinking preference renders with thinking on.
    thinks_by_default: bool = false,
    /// Its packs may store pair-major expert tiles.
    reads_pair_major: bool = false,
    /// The routed layers start here until the config's own dense prefix is read.
    stub_first_moe: u16 = 0,
    /// The config's `moe_layer_freq` names the dense prefix.
    dense_prefix_in_layer_freq: bool = false,
    /// Video input is wired.
    serves_video: bool = true,
    /// It renders its own chat template, so a pack with no template file still chats.
    renders_own_chat_template: bool = false,
    /// The residual is a set of hyper-connection streams: the layers carry no norms (the stream read normalizes),
    /// each loads its hyper-connection weights and the PLE, and the trunk keeps a weight-less RMS norm's ones.
    hyper_connected: bool = false,
    /// A dense embedding table's dtype is the residual's, where others widen or narrow it to bf16.
    residual_from_embedding: bool = false,
    /// The MoE routing scores stay f32 into the expert sum, which is narrowed back to the activations' dtype.
    moe_scores_f32: bool = false,
    /// The attention binder takes a fused QKV split by per-head K and V widths (or the source FP8 QKV whole).
    qkv_split_rows: bool = false,
    /// The routed layers route by sigmoid with a selection-only bias, carry no shared expert, and bank biasless.
    bias_routed_moe: bool = false,
    /// Where its routed experts live in a pack that streams them; `none`: they cannot stream.
    expert_source: ExpertSource = .none,
    /// A streamed pack may group its EXL3 layers by rate.
    streams_rate_groups: bool = false,
    /// The width its prefill starts at, resolved per request by the admission bill; 0 = not a per-request arch of this kind.
    prefill_start_width: u32 = 0,
    /// The MLX buffer pool its streamed decode holds, billed in the SSD-budget ledger.
    expert_pool_bytes: u64 = 0,
    /// Its routed experts always stream.
    streams_experts: bool = false,
    /// Its MTP head draws its drafts at this temperature; null takes the shared sidecar head's.
    mtp_draft_temp: ?f32 = null,
    /// The adaptive MTP controller's cost surface was calibrated for its head, though the head is not one the trunk carries.
    mtp_adaptive_calibrated: bool = false,
    /// The quoted tensor key whose presence in `model.safetensors.index.json` says a pack carries the arch's MTP head.
    mtp_head_marker: []const u8 = "",
    /// The context length below which the adaptive switch does not exist; 0 takes the shared floor.
    mtp_adaptive_min_kv: u32 = 0,
    /// A speculative round rolls the cache back by truncating it: the layers keep no recurrent state to restore.
    mtp_rollback_by_truncate: bool = false,
    /// The round-cost grid its persisted table uses.
    cost_layout: CostLayout = .legacy,
    /// Its requests keep state beside the KV cache that a prefix entry restores whole: the cache is hybrid, and its
    /// checkpoints thin by recency.
    state_beside_kv: bool = false,
    /// It runs no MTP head: the engine's `[mtp]` line and head default do not apply.
    no_mtp_head: bool = false,
    /// The MTP settings and flags do not reach it.
    mtp_unserved: bool = false,
    /// Its DFlash2 assistant loads only when `--drafter` names it, is billed by the shared assistant terms and stays
    /// resident on a streamed load.
    dflash_by_flag: bool = false,
    /// A streamed load keeps images on: its vision tower is resident beside the streamed experts.
    vision_on_streamed: bool = false,
    /// The KV cache holds an MLA latent per layer, not per-head K and V; the pooled index keeps key/gate pairs and no QSA history.
    latent_kv: bool = false,
    /// A recurrent state of KDA's own: f32 whatever the GDN state width, owned by the request, not by SSM entries.
    kda_state: bool = false,
    /// Its sliding layers keep a ring of the window.
    sliding_ring: bool = false,
    /// Every long-context mechanism (KV reservation, pad-waste cap, checkpoint thinning, admission terms, chunk bar) is on.
    long_ctx_gated: bool = false,
    /// Admission credits the hot cache and evicts it to admit a prefill though the arch is not long-context gated.
    evicts_hot_cache: bool = false,
    /// A resident EXL3 pack has the measured load envelope and component bills.
    sushi_quant_memory_bill: bool = false,
    /// A checkpoint can declare thinking off: the table keeps no default word, and a request's on switch alone leaves it off.
    thinking_off_checkpoint: bool = false,
    /// Its MTP tensors load only when the head will run.
    skips_mtp_weights_when_off: bool = false,
    /// `--kv-quant` has nothing to quantize: it keeps its own compressed state.
    own_kv_state: bool = false,
    /// Attention scores only each row's picked keys, billed in its own transient: no dense score sheet narrows its prefill chunk.
    scores_picked_keys: bool = false,
    /// Its resident pack fills the machine: the wired ceiling is the declared limit, not what other apps leave free.
    declared_wired_ceiling: bool = false,
    /// Its DFlash2 assistant is billed by the arch's own bill, not the shared sliding-assistant terms.
    own_dflash_bill: bool = false,
    /// Streamed experts fill preallocated slabs in place: the fill stages nothing beyond the workspace and bounce.
    fills_slabs_in_place: bool = false,
    /// The drafter's block size is capped by the chip's wide verify lane, and the load line names the cap; false where
    /// the arch's `dflashRowBounds` decides the rows alone.
    chip_capped_drafter_block: bool = true,
    /// Its weight loader reads and checks the pack itself; the shared EXL3 pack index check does not apply.
    reads_own_source: bool = false,
    /// Its RoPE reads `rope_parameters` of type yarn, so a per-model `yarn_factor` reaches it.
    reads_yarn: bool = false,
};

/// The config data only `name`'s arch reads, parsed once with the config: its `OwnConfig`, or an empty struct where the
/// arch is not built or declares none.
pub fn OwnConfig(comptime name: []const u8) type {
    if (@hasField(Tag, name)) {
        const M = registry.Mod(@field(Tag, name));
        if (@hasDecl(M, "OwnConfig")) return M.OwnConfig;
    }
    return struct {};
}

fn releaseMod(comptime name: []const u8) ?type {
    if (!@hasField(Tag, name)) return null;
    const M = registry.Mod(@field(Tag, name));
    return if (@hasDecl(M, "releaseOwn")) M else null;
}

/// For tests of shared code at an arch's geometry: the `OwnConfig` of its release checkpoint; null where the arch is not built.
pub fn releaseOwn(comptime name: []const u8) !?OwnConfig(name) {
    return if (comptime releaseMod(name)) |M| try M.releaseOwn() else null;
}

/// For tests of shared code at an arch's geometry: what the arch serves with no flag or setting.
pub fn releaseServeDefaults(comptime name: []const u8) model.ServeDefaults {
    return if (comptime releaseMod(name)) |M| M.serveDefaults(model.ServeDefaults) else .{};
}

/// `Facts.hyper_connected` of the config's arch; false for an unclaimed config.
pub fn hyperConnected(config: *const model.ModelConfig) bool {
    return if (facts(config)) |f| f.hyper_connected else false;
}

/// One of the config's arch's `Facts`; false for an unclaimed config.
pub fn fact(config: *const model.ModelConfig, comptime name: []const u8) bool {
    return if (facts(config)) |f| @field(f, name) else false;
}

/// The `mtp_head_marker` of every built arch that has one.
pub const mtp_head_markers = blk: {
    var out: []const []const u8 = &.{};
    for (std.enums.values(Tag)) |t| {
        const marker = registry.Mod(t).config_facts.mtp_head_marker;
        if (marker.len != 0) out = out ++ &[_][]const u8{marker};
    }
    const final = out[0..out.len].*;
    break :blk final;
};

pub fn tagOf(config: *const model.ModelConfig) ?Tag {
    return config.arch.tag orelse registry.ofModelType(config.model_type);
}

/// `is` for a config held by value or by pointer.
pub fn isOf(config: anytype, comptime name: []const u8) bool {
    return switch (@typeInfo(@TypeOf(config))) {
        .pointer => is(config, name),
        else => is(&config, name),
    };
}

pub fn is(config: *const model.ModelConfig, comptime name: []const u8) bool {
    const want = comptime std.meta.stringToEnum(Tag, name);
    return want != null and tagOf(config) == want.?;
}

fn claims(comptime types: anytype, model_type: []const u8) bool {
    inline for (types) |t| if (std.mem.eql(u8, t, model_type)) return true;
    return false;
}

/// The arch whose config parse reads `model_type`: its served types and the text-only twins of them.
pub fn ofConfigType(model_type: []const u8) ?Tag {
    inline for (comptime std.enums.values(Tag)) |tag| {
        if (claims(registry.Mod(tag).config_types, model_type)) return tag;
    }
    return null;
}

/// Whether the arch that parses `model_type` reads YaRN `rope_parameters`.
pub fn readsYarn(model_type: []const u8) bool {
    const tag = ofConfigType(model_type) orelse return false;
    return factsOf(tag).reads_yarn;
}

/// Whether discovery lists a directory declaring `model_type`.
pub fn discovers(model_type: []const u8) bool {
    inline for (comptime std.enums.values(Tag)) |tag| {
        if (claims(registry.Mod(tag).discovery_types, model_type)) return true;
    }
    return false;
}

/// The facts of the arch that serves `model_type`; null when no enabled arch does.
pub fn factsOfType(model_type: []const u8) ?Facts {
    const tag = registry.ofModelType(model_type) orelse return null;
    return factsOf(tag);
}

fn factsOf(tag: Tag) Facts {
    return switch (tag) {
        inline else => |t| registry.Mod(t).config_facts,
    };
}

pub fn facts(config: *const model.ModelConfig) ?Facts {
    return factsOf(tagOf(config) orelse return null);
}

/// The pool the ledger bills beside the model: the arch's own answer for this config where it has one, else its fact.
pub fn expertPoolBytes(config: *const model.ModelConfig) u64 {
    const tag = tagOf(config) orelse return 0;
    switch (tag) {
        inline else => |t| return if (@hasDecl(registry.Mod(t), "expertPoolBytes")) registry.Mod(t).expertPoolBytes(config) else registry.Mod(t).config_facts.expert_pool_bytes,
    }
}

/// Whether the arch's load keeps its streamed experts' scale vectors in host memory; an arch that says nothing keeps none.
pub fn expertKeepsScales(config: *const model.ModelConfig) bool {
    const tag = tagOf(config) orelse return false;
    switch (tag) {
        inline else => |t| return if (@hasDecl(registry.Mod(t), "expertKeepsScales")) registry.Mod(t).expertKeepsScales(config) else false,
    }
}

/// Tokens the prompt's chunk loop holds back for its last forward, from an arch that places the prompt-end snapshot
/// by its own template (`tok` looks its markers up); null is the shared `prefill_plan.SSM_SNAPSHOT_BACKOFF`.
pub fn snapshotBackoff(config: *const model.ModelConfig, tok: anytype, prompt_ids: []const u32) ?usize {
    const tag = tagOf(config) orelse return null;
    switch (tag) {
        inline else => |t| return if (@hasDecl(registry.Mod(t), "snapshotBackoff")) registry.Mod(t).snapshotBackoff(tok, prompt_ids) else null,
    }
}

/// The arch's part of the parse, on top of the fields every arch shares.
pub fn parseFamily(config: *model.ModelConfig, tag: Tag, root: std.json.ObjectMap, cfg_obj: std.json.ObjectMap) !void {
    config.arch = .{ .tag = tag };
    switch (tag) {
        inline else => |t| try registry.Mod(t).parseConfig(config, root, cfg_obj),
    }
}

/// What the arch adds once the pack directory is known (paths, environment).
pub fn parseDir(config: *model.ModelConfig, allocator: std.mem.Allocator, model_dir: []const u8) !void {
    const tag = tagOf(config) orelse return;
    switch (tag) {
        inline else => |t| if (@hasDecl(registry.Mod(t), "parseDir")) try registry.Mod(t).parseDir(config, allocator, model_dir),
    }
}

/// What the arch changes once the pack's expert layout is known.
pub fn adoptLayout(config: *model.ModelConfig) void {
    const tag = tagOf(config) orelse return;
    switch (tag) {
        inline else => |t| if (@hasDecl(registry.Mod(t), "adoptLayout")) registry.Mod(t).adoptLayout(config),
    }
}

/// The family's documented sampling truncation, for a checkpoint that ships no generation_config.json.
pub fn samplingDefaults(config: *model.ModelConfig) void {
    const tag = tagOf(config) orelse return;
    switch (tag) {
        inline else => |t| if (@hasDecl(registry.Mod(t), "samplingDefaults")) registry.Mod(t).samplingDefaults(config),
    }
}

/// The stored-state layout name of the arch's disk cache namespace; null when the arch has none of its own.
pub fn cacheLayoutNamespace(config: *const model.ModelConfig) ?[]const u8 {
    const tag = tagOf(config) orelse return null;
    return switch (tag) {
        inline else => |t| if (@hasDecl(registry.Mod(t), "cacheLayoutNamespace")) registry.Mod(t).cacheLayoutNamespace(config) else null,
    };
}

/// The bytes of per-token state the arch keeps for one cached token; null when the arch bills it by other terms.
pub fn kvBytesPerToken(config: *const model.ModelConfig) ?u64 {
    const tag = tagOf(config) orelse return null;
    return switch (tag) {
        inline else => |t| if (@hasDecl(registry.Mod(t), "kvBytesPerToken")) registry.Mod(t).kvBytesPerToken(config) else null,
    };
}

/// Hands the Sushi Cooler choice (`on`; when off, `by_flag` says the launch flag asked, else the model setting) to an
/// arch that has the engine; the other archs have nothing to stamp.
pub fn stampCooler(config: *model.ModelConfig, on: bool, by_flag: bool) void {
    const tag = tagOf(config) orelse return;
    switch (tag) {
        inline else => |t| if (@hasDecl(registry.Mod(t), "stampCooler")) registry.Mod(t).stampCooler(config, on, by_flag),
    }
}

/// Whether this load keeps the arch's DSpark drafter stages resident.
pub fn dsparkOn(config: *const model.ModelConfig) bool {
    const tag = tagOf(config) orelse return false;
    return switch (tag) {
        inline else => |t| if (@hasDecl(registry.Mod(t), "dsparkOn")) registry.Mod(t).dsparkOn(config) else false,
    };
}

/// What a streamed load keeps resident when the arch reads its own source layout; null takes the indexed scan.
pub fn streamedResidentSplit(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const model.ModelConfig) !?model.ResidentSplit {
    const tag = tagOf(config) orelse return null;
    return switch (tag) {
        inline else => |t| if (@hasDecl(registry.Mod(t), "streamedResidentSplit"))
            try registry.Mod(t).streamedResidentSplit(io, allocator, model_dir, config)
        else
            null,
    };
}

/// The layers a pack of `config`'s arch stores after the trunk's (its MTP layers under the trunk's key prefix); 0 when
/// the arch declares none.
pub fn trailingPackLayers(config: *const model.ModelConfig) u32 {
    const tag = tagOf(config) orelse return 0;
    return switch (tag) {
        inline else => |t| if (@hasDecl(registry.Mod(t), "trailingPackLayers")) registry.Mod(t).trailingPackLayers(config) else 0,
    };
}

/// Where a pack's routed experts sit and what its preflight (`pack_index.validateExl3Source`) checks. An arch states it
/// as `pack_layout` in its `arch.zig`; the defaults are the layout of an unclaimed pack.
pub const PackLayout = struct {
    /// Key prefix of the trunk's layers.
    prefix: []const u8 = "language_model.model",
    /// Every shard's codebook stamp must agree with the config.
    stamped: bool = false,
    /// The MTP head's routed layer: a key prefix of its own, the `trailingPackLayers` under `prefix`, or none.
    head: union(enum) { none, layer: []const u8, trailing } = .{ .layer = "language_model.mtp.layers.0" },
    /// A rate-grouped layer streams group by group.
    streams_groups: bool = false,
    /// A rate-grouped layer is refused.
    refuses_rate_groups: bool = false,
};

pub fn packLayout(config: *const model.ModelConfig) PackLayout {
    const tag = tagOf(config) orelse return .{};
    return switch (tag) {
        inline else => |t| if (@hasDecl(registry.Mod(t), "pack_layout")) registry.Mod(t).pack_layout else .{},
    };
}

/// The committed config of the arch registered as `name` (`fixtures.config` in its chat face), for a shared test;
/// null when the build left the arch out.
pub fn fixtureConfig(comptime name: []const u8) ?[]const u8 {
    if (comptime !@hasField(Tag, name)) return null;
    return registry.Chat(@field(Tag, name)).fixtures.config;
}

/// A config no arch claims, which a shared fixture parses in place of an arch the build left out.
pub const unbuilt_config = "{\"model_type\":\"unbuilt\",\"hidden_size\":64,\"vocab_size\":32}";

/// The names the arch gate's refusal lists: the enabled archs, in the order they were first documented.
pub const gate_text = blk: {
    var text: []const u8 = "";
    for (registry.registered_model_types) |t| {
        if (registry.ofModelType(t) != null) text = text ++ (if (text.len > 0) ", " else "") ++ t;
    }
    break :blk text;
};

test "every enabled arch declares its config side" {
    inline for (comptime std.enums.values(Tag)) |tag| {
        const M = registry.Mod(tag);
        inline for (.{ "config_types", "discovery_types", "config_facts", "parseConfig" }) |decl| {
            if (!@hasDecl(M, decl)) @compileError(@tagName(tag) ++ "/arch.zig lacks " ++ decl);
        }
        for (M.model_types) |t| try std.testing.expect(claims(M.config_types, t) and claims(M.discovery_types, t));
    }
}

test "the arch gate text lists every enabled arch's model types and no other" {
    for (registry.served_model_types) |t| try std.testing.expect(std.mem.indexOf(u8, gate_text, t) != null);
    for (registry.registered_model_types) |t| {
        if (registry.ofModelType(t) == null) try std.testing.expect(std.mem.indexOf(u8, gate_text, t) == null);
    }
}

test "a config type resolves to the arch that parses it and an unknown one to none" {
    try std.testing.expectEqual(@as(?Tag, null), ofConfigType("mistral"));
    try std.testing.expectEqual(@as(?Tag, null), ofConfigType(""));
    inline for (comptime std.enums.values(Tag)) |tag| {
        for (registry.Mod(tag).config_types) |t| try std.testing.expectEqual(@as(?Tag, tag), ofConfigType(t));
    }
}

test "the arch gate text keeps the order its refusal has always listed" {
    if (registry.served_model_types.len != 4) return;
    try std.testing.expectEqualStrings("qwen4_exp, mimo_v2, glm5_next, deepseek_v41", gate_text);
}

test "a per-request-prefill arch states its start width and the others state none" {
    if (factsOfType("glm5_next")) |f| try std.testing.expectEqual(@as(u32, 2048), f.prefill_start_width);
    if (factsOfType("qwen4_exp")) |f| try std.testing.expectEqual(@as(u32, 0), f.prefill_start_width);
}

test "a config takes its arch's pack layout; an unclaimed one the default" {
    inline for (comptime std.enums.values(Tag)) |t| {
        const M = registry.Mod(t);
        const c = model.ModelConfig{ .model_type = M.config_types[0] };
        try std.testing.expectEqualDeep(if (@hasDecl(M, "pack_layout")) M.pack_layout else PackLayout{}, packLayout(&c));
    }
    try std.testing.expectEqualDeep(PackLayout{}, packLayout(&.{ .model_type = "mistral" }));
}
