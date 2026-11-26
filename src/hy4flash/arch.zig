//! Hy4-Flash (`hy_v4_flash`) as the registry sees it: its one public face. The composed forward (`hy4_model`) serves a
//! load whose tensors are all resident; streamed experts (1k2) and Engram read by row (1k3) come next, and `cooler` (no
//! opt-out) with its contract (docs/hy4-flash-arch.md).
const std = @import("std");
const mlx = @import("../mlx.zig");
const trunk = @import("../trunk.zig");
const admission = @import("../arch/admission.zig");
const arch_config = @import("../arch/config.zig");
const registry = @import("../arch/registry.zig");
const config_file = @import("hy4_config.zig");
const Trunk = trunk.Trunk;

pub const model_types = chat.model_types;
pub const Config = @import("../model.zig").ModelConfig;

pub fn kvBytesPerToken(config: *const Config) ?u64 {
    const c = config.hy4flash orelse return null;
    return bill.latentBytesPerToken(&c);
}
/// The SSD namespace of Hy4's stored points: v2 splits a point into its small state and the entry's row segments.
pub fn cacheLayoutNamespace(config: *const Config) ?[]const u8 {
    _ = config;
    return "hy4-state-v2";
}
pub const Model = *Trunk;
pub const State = @import("hy4_state.zig").State;
pub const Ctx = @import("hy4_state.zig").Ctx;

pub const packs = struct { // slot:packs (S2h)
    pub const product = "Hy4-Flash";
    pub const aliases = [_]@import("../arch/packs.zig").Alias{
        .{ .name = "hy4-flash", .tag = "4bpw", .repo = "beamster/Hy4-Flash-Sushi-STQ-4bpw", .is_default = true, .tier_gb = 96, .recommend_gb = 96 },
        .{ .name = "hy4-flash", .tag = "2.25bpw", .repo = "beamster/Hy4-Flash-Sushi-STQ-2.25bpw", .tier_gb = 64, .recommend_gb = 48 },
    };
};
pub const flags = @import("hy4_flags.zig"); // slot:flags (S2h)
pub const sched = @import("hy4_sched.zig"); // slot:sched (S2b)
pub const chat = @import("hy4_chat.zig"); // slot:chat (S2g)
pub const prefix = @import("hy4_prefix.zig"); // slot:prefix (S2c)
pub const bill = @import("hy4_ledger.zig"); // slot:bill (S2d)
/// DFly2 drafts in the shared drafter rounds over the model's own state (`hy4_drafter`).
pub const drafter = @import("hy4_drafter.zig"); // slot:drafter
pub const media = @import("hy4_media.zig"); // slot:media (S2f)
pub const load = @import("hy4_load.zig");
pub const kld = @import("hy4_kld_capture.zig");
/// Its STQ packs' routed experts take the pair-major layout (`sushi repack`); its trunk and DFly2 banks are dense and stay.
/// The experts run at `routed_expert_hidden_size`, not the trunk's width.
pub const repack: arch_config.Repack = .{ .hidden_key = "routed_expert_hidden_size" };

/// The arch's microbenches, which `sushi-ubench` lists; nothing else reads them.
pub const ubench = @import("ubench.zig");

/// What a streamed load keeps resident, which the preflight sizes the load by (`hy4_load.streamedSplit`).
pub fn streamedResidentSplit(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const Config) !?@import("../model.zig").ResidentSplit {
    return try load.streamedSplit(io, allocator, model_dir, config);
}
/// Every Hy4-Flash load runs on Sushi Cooler: there is no MLX arm to opt out to.
pub const cooler: arch_config.Cooler = .{ .opt_out = null };
/// What the shared Sushi Cooler contract suite asks of this arch (`src/cooler_contract_test.zig`).
pub const cooler_contract = @import("hy4_contract.zig");

pub const config_types = model_types;
pub const discovery_types = model_types;
pub const samplingDefaults = @import("hy4_config.zig").samplingDefaults;
/// An STQ pack's routed banks sit under the release's own names (`model.layers.{L}.mlp.switch_mlp`); its trunk shards
/// carry the trunk's SPEC1 stamp beside the experts' PAIR1, so the shard stamps are not held to the expert block.
pub const pack_layout: arch_config.PackLayout = .{ .prefix = "model", .head = .none, .streams_groups = true };
pub const config_facts: arch_config.Facts = .{ .thinks_by_default = true, .streams_experts = true, .vision_on_streamed = true, .expert_source = .fused_banks, .prefill_start_width = 4096, .static_chunk_max = @import("hy4_config.zig").widest_chunk, .static_chunk_default = @import("hy4_config.zig").default_chunk, .streams_rate_groups = true, .reads_pair_major = true, .reads_pack_order = false, .exclusive_decode_of = joinsNoRows, .ssd_first_capable = true, .bias_unserved = true, .ssd_tables = &.{".engram_layer.multi_head_embedding."} };
/// A load that joins no rows runs one request at a time; one that joins them is held to its row count.
fn joinsNoRows(config: *const @import("../model.zig").ModelConfig) bool {
    return @import("hy4_plan.zig").joinedRows(config) <= 1;
}
pub const OwnConfig = config_file.Config;
pub const serveDefaults = config_file.Config.serveDefaults;
/// The family's fields, then an STQ pack's trunk rate (`trunk_quant`), which the MLX-free family parse does not read:
/// the bill prices the GPU embedding gather by it, and the pack label names it.
pub fn parseConfig(config: anytype, root: std.json.ObjectMap, cfg_obj: std.json.ObjectMap) !void {
    try config_file.parseFamily(config, root, cfg_obj);
    if (root.get("trunk_quant")) |b| config.trunk_quant_rate = (try @import("sushi_exl3").parseQuantBlock(b)).rate;
}

pub fn releaseOwn() !OwnConfig {
    return config_file.parseText(config_file.release_file);
}

pub fn init(io: std.Io, allocator: std.mem.Allocator, config: Config, weights: *trunk.Weights, s: mlx.mlx_stream) !Trunk {
    const own = config.hy4flash orelse return error.Hy4ConfigMissing;
    const model_stream = @import("../cooler_side.zig").stream(&config) orelse s;
    var shell = try @import("../layer_weights.zig").initModuleShell(allocator, config, model_stream);
    errdefer shell.deinitTrunk();
    // A streamed load's banks stay on SSD; the shell owns the engine from here and frees it with itself.
    if (config.expert_streaming) shell.expert_stream = try Trunk.newExpertEngine(allocator, &config, model_stream);
    const st = &shell.arch_state.hy4flash;
    errdefer st.deinit(allocator);
    st.weights = try @import("hy4_state.zig").takeWeights(allocator, weights, &own);
    st.loader_weights = weights;
    // Engram's tables are read by row from the checkpoint's shards on every load (`hy4_load.kept`).
    const dir = config.expert_source_dir orelse return error.MissingExpertSourceDir;
    // An STQ pack's stamps (`trunk_quant`, `expert_quant`, `synthetic`); none for the BF16 release.
    const quant = try @import("hy4_stq.zig").readQuant(io, allocator, dir);
    // Each NAX GEMM proven where it runs before any recording asks for one: a dry recording cannot prove it.
    if (@import("../cooler_side.zig").onEngine(&config)) {
        const kernels = @import("sushi_exl3").kernels;
        if (quant.expert) |dec| try kernels.prepareGemmNax(dec, if (config.expert_tile_order == .pair_major) .tile_major else .pack, model_stream);
        if (quant.trunk) |dec| try kernels.prepareGemmNax(dec, .pack, model_stream);
    }
    st.model = try @import("hy4_model.zig").Model.initWith(allocator, own, st.weights.?, model_stream, .{ .experts = shell.expert_stream, .dir = dir, .quant = quant, .served = true });
    // The shell's head is the model's: the shared Cooler seam reads its width for a wide forward's readout.
    // An STQ head is its trellis bank, as the shared trunk holds one (its width read by `trunk_stq.outDim`).
    const head = if (st.model.?.lm_head_stq) |b| b.w else st.model.?.lm_head;
    try mlx.check(mlx.mlx_array_set(&shell.lm_head_w, head));
    // Freed by `deinit` once loaded; a step after this that fails gives it back here (the trunk's own frees skip it).
    errdefer {
        _ = mlx.mlx_array_free(shell.lm_head_w);
        shell.lm_head_w = .{ .ctx = null };
    }
    // DFly2, when the load keeps it: the shared drafter rounds over the model's own state (`hy4_drafter`).
    if (drafterOn(&config)) st.drafter = try @import("hy4_dfly.zig").Drafter.init(allocator, st.model.?, st.weights.?);
    return shell;
}

/// DFly2 stays resident beside streamed experts in either of Hy4's expert layouts: the release's fused BF16 banks and an
/// STQ pack's trellis groups.
pub const streamed_mtp_head = struct {
    pub const phrase = "a Hy4-Flash pack that ships DFly2";
    pub fn supported(layout: @import("../expert_quant.zig").Layout) bool {
        return layout == .bf16_fused or layout == .exl3_k4;
    }
};

/// Whether this load keeps DFly2: the load resolved its MTP on and the pack ships the layers (`stream_mtp_head`, set by
/// the loader on a resident load and a streamed one alike).
pub fn drafterOn(config: *const Config) bool {
    const c = config.hy4flash orelse return false;
    return config.stream_mtp_head and c.dfly.layers > 0;
}

pub fn deinit(xfm: Model) void {
    xfm.arch_state.hy4flash.deinit(xfm.allocator);
    // The shell's handle on the model's head (`init`): the trunk frees no head it does not load.
    if (xfm.lm_head_w.ctx != null) _ = mlx.mlx_array_free(xfm.lm_head_w);
    xfm.lm_head_w = .{ .ctx = null };
    xfm.deinitTrunk();
}

pub fn resetCache(xfm: Model) !void {
    @import("hy4_state.zig").dropVerify(&xfm.arch_state.hy4flash, false);
    if (xfm.arch_state.hy4flash.model) |m| {
        const own: ?@import("hy4_model.zig").Model.Lent = if (m.serves_requests and m.bound == null) m.lendOwn() else null;
        defer if (own) |l| l.end();
        m.reset();
    }
    return xfm.resetTrunk();
}

/// What a load's pipeline manifest is keyed by: the facts that decide which kernels its forwards compile (the shapes,
/// the attention and indexer geometry, the expert layout and whether experts stream, the first chunk's width).
pub fn coolerPipelineScope(config: *const Config) [16]u8 {
    const manifest = @import("../pipeline_manifest.zig");
    const c = config.hy4flash orelse return @splat('0');
    var d: manifest.Digest = .{};
    d.add("model type", config.model_type);
    for ([_]struct { []const u8, u64 }{
        .{ "layers", c.layers },                                 .{ "dim", c.dim },
        .{ "vocab", config.vocab_size },                         .{ "hc mult", c.hc_mult },
        .{ "sparse heads", c.sparse_heads },                     .{ "sliding heads", c.sliding_heads },
        .{ "kv lora", c.kv_lora },                               .{ "q lora", c.q_lora },
        .{ "rope", c.rope_dim },                                 .{ "window", c.window },
        .{ "index group", c.indexer.group },                     .{ "index topk", c.indexer.group_topk },
        .{ "index heads", c.indexer.heads },                     .{ "experts", c.experts },
        .{ "top k", c.top_k },                                   .{ "expert width", c.moe_inter },
        .{ "expert layout", @backingInt(config.expert_layout) }, .{ "tile order", @intFromEnum(config.expert_tile_order) },
        .{ "streamed", @intFromBool(config.expert_streaming) },  .{ "dfly2", @intFromBool(drafterOn(config)) },
        .{ "start width", config.prefillStartWidth() },
    }) |f| d.addInt(f[0], f[1]);
    @import("../arch/cooler_pipelines.zig").addRecipeEnv(&d);
    return d.hex();
}

pub fn warmup(xfm: Model) !void {
    if (xfm.arch_state.hy4flash.model) |m| try m.warmKernels();
    if (xfm.arch_state.hy4flash.model) |m| try m.warmJoined(@import("hy4_plan.zig").joinedRows(&xfm.config), @import("../bills.zig").configuredKvQuantFor(&xfm.config));
    return xfm.warmupTrunk();
}

/// The load's plans recorded before its warm (D98, `hy4_plan.record`), its samples' with them, kept on the model
/// (`Model.recorded`) until the warm's end makes their arena (`planBuild`). A load whose plans are recorded or built
/// already records nothing.
pub fn planRecord(xfm: Model, cache: ?@import("../tensor/plan/plan_book.zig").Cache) !void {
    const m = xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    if (m.plans != null or m.recorded != null) return;
    const bills = @import("../bills.zig");
    const dfly: ?*const @import("hy4_dfly.zig").Drafter = if (xfm.arch_state.hy4flash.drafter) |*d| d else null;
    // A load that keeps the vision tower plans its encode beside its forwards, over the tower's own tensors.
    var tower: ?@import("hy4_vision.zig").Tower = null;
    defer if (tower) |*t| t.deinit();
    const v = m.config.vision;
    if (v != null) if (xfm.arch_state.hy4flash.loader_weights) |w| if (w.get("vit.embeddings.patch_embedding.weight") != null) {
        tower = try @import("hy4_vision.zig").Tower.init(xfm.allocator, v.?, w);
    };
    // The ledger holds the encode where the load recorded that it keeps the tower (`ModelConfig.keeps_tower`).
    if ((tower != null) != xfm.config.keeps_tower) return error.Hy4TowerUnbilled;
    const encode: ?@import("hy4_plan.zig").Encode = if (tower) |*t| .{ .tower = t, .max_patches = media.maxImagePatches(media.preproc(&xfm.config)) } else null;
    const p = try @import("hy4_plan.zig").record(m, xfm.allocator, bills.planReach(&xfm.config), dfly, bills.configuredKvQuantFor(&xfm.config), cache, encode, @import("hy4_plan.zig").joinedRows(&xfm.config));
    errdefer p.deinit(xfm.allocator);
    p.pack = try packKey(xfm);
    try @import("hy4_sample_plan.zig").record(m, p, .{ .suppress = xfm.suppress_mask });
    m.recorded = p;
}

/// The pack's identity, which names the region a load keeps (`hy4_plan.Plans.pack`).
fn packKey(xfm: Model) ![16]u8 {
    const manifest = @import("../pipeline_manifest.zig");
    var d: manifest.Digest = .{};
    try manifest.addPack(&d, xfm.allocator, std.Io.Threaded.global_single_threaded.io(), xfm.config.expert_source_dir orelse return error.MissingExpertSourceDir);
    return d.hex();
}

/// The arena a load whose plans are recorded or built holds with prefill chunks of `chunk` rows: its plans' and the
/// region the chunk asks (`hy4_plan.regionAt`, dry), each part the larger of the two. null for a load that planned none.
pub fn planArena(xfm: Model, chunk: u32) !?u64 {
    const m = xfm.arch_state.hy4flash.model orelse return null;
    const p = m.recorded orelse m.plans orelse return null;
    return p.layoutWith(try @import("hy4_plan.zig").regionAt(m, p, @min(chunk, p.reach))).arenaBytes();
}

/// The load's plans, at its warm's end (`arch/cooler.zig` `planLoad`): those recorded before the warm (`planRecord`),
/// else recorded now, and the arena made for them and the region of the load's widest prefill chunk (D97). Every load
/// plans, a streamed one with `--expert-cache-bias` too (D102).
pub fn planBuild(xfm: Model, cache: ?@import("../tensor/plan/plan_book.zig").Cache) !void {
    const m = xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    if (m.plans != null) return;
    try planRecord(xfm, cache);
    const p = m.recorded.?;
    try @import("hy4_plan.zig").makeArena(m, p, @min(@import("../prefill_plan.zig").widestPrefillWidth(&xfm.config), p.reach));
    m.recorded = null;
    m.plans = p;
}

/// A draw of a served request begins (`arch/cooler.zig` `Sample`): an execution of its mix's root where the load planned
/// it (D24).
pub fn sampleBegin(xfm: Model, mix: @import("../sampler.zig").Mix) !bool {
    const m = xfm.arch_state.hy4flash.model orelse return false;
    return @import("hy4_sample_plan.zig").begin(m, mix);
}

pub fn sampleToken(xfm: Model, lazy: mlx.mlx_array) !mlx.mlx_array {
    return @import("hy4_sample_plan.zig").token(xfm.arch_state.hy4flash.model.?, lazy);
}

/// A forced call's token in the form the load's draws read a token in flight, null where it plans no samples.
pub fn sampleForced(xfm: Model, id: u32) !?mlx.mlx_array {
    const m = xfm.arch_state.hy4flash.model orelse return null;
    return @import("hy4_sample_plan.zig").forced(m, id);
}

pub fn sampleEnd(xfm: Model) !void {
    return @import("hy4_sample_plan.zig").end(xfm.arch_state.hy4flash.model.?);
}

pub fn sampleFail(xfm: Model) error{PlanMiss}!void {
    return @import("hy4_sample_plan.zig").fail(xfm.arch_state.hy4flash.model.?);
}

/// A planned load's forward input: `ids` written into the model's input block (`Model.idsInput`); null where the load
/// plans nothing, whose input is the generator's own array.
pub fn forwardIds(xfm: Model, ids: []const i32) !?mlx.mlx_array {
    const m = xfm.arch_state.hy4flash.model orelse return null;
    if (m.plans == null) return null;
    return try m.idsInput(ids);
}

/// The rows `ids` [1, n] at the request's position `ctx.cache.step`, extending the model's state; the last row's logits
/// [1, 1, vocab] in f32. A position 0 starts the request over.
pub fn forward(xfm: Model, ctx: *trunk.ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
    const m = xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    const scope = m.enterForward(@import("hy4_state.zig").requestOf(ctx));
    defer scope.endForward();
    const r = m.req();
    if (ctx.cache.step == 0) {
        // A cold request: the last one's live record no longer stands behind this state, nor an open verify's mark.
        @import("hy4_state.zig").dropVerifyReq(m, r, false);
        m.reset();
        m.dropLive();
        r.kv = try kvWidth(ctx.cache.config);
    }
    if (r.pos != ctx.cache.step) return error.Hy4CachePositionMismatch;
    const shape = mlx.getShape(ids);
    if (shape.len != 2 or shape[0] != 1) return error.InvalidHy4Input;
    // The request's state made at its end before the forward runs, so the forward asks for none of its own; never
    // short of the rows this forward writes.
    if (ctx.cache.end_tokens > 0) try m.reserveFor(@max(ctx.cache.end_tokens, r.pos + @as(usize, @intCast(shape[1]))));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_astype(&flat, ids, .uint32, xfm.s));
    try mlx.check(mlx.mlx_array_eval(flat));
    const n: usize = @intCast(shape[1]);
    const host = (mlx.mlx_array_data_uint32(flat) orelse return error.InvalidHy4Input)[0..n];
    if (xfm.expert_stream) |engine| engine.beginForward();
    const vision: ?@import("hy4_model.zig").Model.VisionRows = if (ctx.vision_embeddings) |ve| .{ .rows = ve, .offset = ctx.vision_splice_offset } else null;
    const spliced = if (vision) |v| m.splicesImage(host, v) else false;
    const logits = try @import("hy4_plan.zig").run(m, host, vision, if (ctx.skip_lm_head) .hidden_last else .last);
    if (xfm.expert_stream) |engine| engine.finishForward(n);
    ctx.cache.step = r.pos;
    // The image rows a forward splices are consumed, one an image token: a later forward splices past them, and a
    // decode past the image is the text forward.
    if (spliced) ctx.vision_splice_offset += std.mem.count(u32, host, &.{m.config.image_id});
    if (ctx.skip_lm_head) return logits;
    defer _ = mlx.mlx_array_free(logits);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, logits, &[_]c_int{ 1, 1, mlx.getShape(logits)[1] }, 3, xfm.s));
    return out;
}

/// The latent rows' width a request's cache names: the engine's affine kv8 (the default) or BF16; any other is refused.
fn kvWidth(kv: @import("../kv_quant.zig").KVQuantConfig) !@import("../kv_quant.zig").KVQuantConfig {
    if (flags.storesKv(kv)) return kv;
    @import("../log.zig").err("hy_v4_flash: KV at {s} {d} bits is not served: 8-bit affine (the default) or 16 (BF16)\n", .{ @tagName(kv.scheme), kv.bits });
    return error.Hy4KvWidthUnsupported;
}

pub fn lmHeadForDraft(xfm: Model, x: mlx.mlx_array) !mlx.mlx_array {
    _ = .{ xfm, x };
    return error.DraftHeadUnsupported;
}

pub fn ownsModuleDecodeState(xfm: Model) bool {
    const m = xfm.arch_state.hy4flash.model orelse return true;
    return !m.serves_requests;
}

pub fn expertStream(config: *const Config) admission.ExpertStreamTraits {
    return .{ .stq_experts = config.expert_layout == .exl3_k4, .tile_major = config.expert_layout == .exl3_k4, .cache_bias = true };
}

test "a model stamped as hy4flash is dispatched to hy4flash" {
    const tag = registry.ofModelType(model_types[0]) orelse return;
    var xfm: Trunk = undefined;
    xfm.arch = tag;
    try std.testing.expectEqual(tag, std.meta.activeTag(registry.handleOf(&xfm)));
}
