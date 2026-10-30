const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const generate_mod = @import("generate.zig");
const mtp_mod = @import("mtp.zig");
const depth_bounds = @import("mtp_depth_bounds.zig");
const dspark_rows = @import("dsv41/dspark_rows.zig");
const mimo_mtp = @import("mimo2/mimo_mtp.zig");
const chip_mod = @import("chip.zig");
const model_mod = @import("model.zig");
const vision_mod = @import("vision.zig");
const chat_mod = @import("chat.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const dsv41_prefix = @import("dsv41/dsv41_prefix.zig");
const metrics_mod = @import("metrics.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const disk_placement = @import("disk_placement.zig");
const tokenize_cache_mod = @import("tokenize_cache.zig");
const model_registry_mod = @import("model_registry.zig");
const model_settings = @import("model_settings.zig");
const mtp_acceptance_mod = @import("mtp_acceptance.zig");
const model_discovery = @import("model_discovery.zig");
const log = @import("log.zig");
const io_util = @import("io_util.zig");
const status = @import("status.zig");
const Transformer = transformer_mod.Transformer;
const ModelConfig = model_mod.ModelConfig;
const Tokenizer = tokenizer_mod.Tokenizer;
const dflash_mod = @import("dflash.zig");
const round_cost_mod = @import("round_cost.zig");
const expert_stream_mod = @import("expert_stream.zig");
const expert_usage = @import("dsv41/expert_usage.zig");
const DflashModel = dflash_mod.DflashModel;
const VisionEncoder = vision_mod.VisionEncoder;
const Weights = model_mod.Weights;
const ChatConfig = chat_mod.ChatConfig;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;
const glm5_rows_tick = @import("glm5/glm5_rows_tick.zig");
const glmRowsUbench = glm5_rows_tick.glmRowsUbench;
const glmRoundUbench = glm5_rows_tick.glmRoundUbench;
const mtp_plan = @import("mtp_plan.zig");
const sampler = @import("sampler.zig");
const testing = std.testing;
const scheduler_mod = @import("scheduler.zig");
const BudgetRevise = scheduler_mod.BudgetRevise;
const LoadRequest = scheduler_mod.LoadRequest;
const Scheduler = scheduler_mod.Scheduler;
const UnloadRequest = scheduler_mod.UnloadRequest;
const hotCacheMemForLoad = scheduler_mod.hotCacheMemForLoad;
const publishDiskStats = scheduler_mod.publishDiskStats;
const publishHotCacheResidency = scheduler_mod.publishHotCacheResidency;

const exl3_kernels = @import("sushi_exl3").kernels;
const request_media = @import("request_media.zig");

/// Phase A1: model-load plan executed on the scheduler's inference thread.
///
/// mlx 0.31.2 uses thread-local GPU streams: every `mlx_*` op binds to the
/// stream of the calling thread, and JIT-compiled closures are tied to that
/// stream too. If main loads the model and the scheduler later calls forward,
/// mlx aborts with "no Stream(gpu, N) in current thread". Solution: the
/// scheduler's inference thread does the load itself so the stream is bound
/// on the right thread from t0.
///
/// CPU-only state (parsed config, tokenizer, chat config) is loaded by main
/// and passed in by reference. The mlx-allocating pieces (weights tensors,
/// Transformer, vision encoder, drafter, JIT compile, warmup) all run on the
/// inference thread before `init()` returns.
pub const LoadParams = struct {
    /// Registry that owns the entry to populate + provides snapshot/eviction
    /// bookkeeping. Outlives the scheduler.
    registry: *ModelRegistry,
    /// The (pre-registered) LoadedModel stub that will be promoted to
    /// `.ready` by the inference-thread load. Holds id/path on entry;
    /// `loadModelOnInferenceThread` installs weights/transformer/vision/
    /// drafter/tokenizer/chat_config/config into this slot. Lifetime
    /// matches the registry.
    entry: *LoadedModel,
    /// Heap-allocated parsed config. Ownership transfers to `entry` on
    /// successful load; caller must NOT deinit/free externally after
    /// `Scheduler.init` returns.
    config: *ModelConfig,
    /// Heap-allocated tokenizer. Ownership transfers to `entry`.
    tok: *Tokenizer,
    /// Heap-allocated chat config. Ownership transfers to `entry`.
    chat_config: *ChatConfig,
    /// Path to the model directory. Borrowed; outlive scheduler.
    model_dir: []const u8,
    /// Path to the assistant drafter checkpoint. Empty disables the drafter.
    /// Borrowed; outlive scheduler.
    drafter_dir: []const u8 = "",
    /// `--no-drafter`: never load a drafter, including one MERGED into the
    /// checkpoint. `drafter_dir == ""` stopped meaning "off" the moment a
    /// checkpoint could carry its own, so the opt-out needs its own bit.
    no_drafter: bool = false,
    /// Auto-load the Qwen native MTP sidecar when the model dir ships one.
    mtp_enabled: bool = true,
    /// `--mtp` / `--no-mtp` was given: `mtp_enabled` then outranks the per-model `mtp`.
    mtp_explicit: bool = false,
    mtp_head_kv_quant: bool = false,
    /// Max MTP draft depth (CLI --mtp-max-depth; 0 = auto, resolved by
    /// generate_mod.resolveMtpDepthCap at load/Generator init).
    mtp_depth: u32 = 0,
    /// Whether to also load vision-tower weights. Combined with
    /// `config.has_vision` — false here disables vision regardless of config.
    load_vision: bool = false,
    /// Eager warmup: fault weight pages + run a tiny forward to JIT-compile
    /// the decode path on the inference thread. Adds ~600-900 ms at boot but
    /// keeps the first user request fast.
    warmup_eager: bool = true,
    /// KV-cache storage backend: `--kv-quant`, else the kv8 engine default. Stored on every
    /// per-slot KVCache and consulted at every read/write boundary.
    kv_quant_config: transformer_mod.KVQuantConfig = transformer_mod.KVQuantConfig.engine_default,
    /// `--kv-quant` was given: the flag then outranks the per-model `kv_quant`.
    kv_quant_explicit: bool = false,
    /// Per-model hot prefix cache capacity (count). 0 disables.
    prefix_cache_capacity: u32 = 1,
    prefix_cache_ram_enabled: bool = true,
    /// Per-model hot prefix cache KV-bytes budget. 0 disables the byte cap.
    prefix_cache_mem_bytes: u64 = 0,
    /// Clamp the hot-cache byte budget against live post-load headroom
    /// (`server.prefixCacheMemForLoad`) — a pointer because the scheduler
    /// deliberately has no server.zig import. Null = no clamp (tests).
    prefix_cache_mem_resolver: ?*const fn (*model_mod.ModelConfig, u64, BudgetRevise, *u64) u64 = null,
    /// SSD tier byte budget for the hot prefix cache (`--prefix-cache-disk`).
    /// 0 disables persistence. Attached per model at load for pure-attention
    /// archs; entries live under `~/.sushi/kv-cache/<fingerprint>`.
    prefix_cache_disk_bytes: u64 = 0,
    /// Sizes the SSD budget for this model and logs the `Allocating` line (`server.prefixCacheDiskForLoad`);
    /// takes the cache directory. Null = `prefix_cache_disk_bytes` as given (tests).
    prefix_cache_disk_resolver: ?*const fn (*model_mod.ModelConfig, []const u8, u64) ?disk_placement.DefaultDiskBudget = null,
    ssd_budget_bytes: u64 = 0,
    expert_cache_fit_resolver: ?*const fn (*const model_mod.ModelConfig, u64) anyerror!void = null,
    /// Phase 1 (perf-plan): SSM/conv state snapshot stride during prefill.
    /// 0 = disabled (hybrid models bypass the hot prefix cache). Non-zero
    /// enables hybrid in `HotPrefixCache.shouldUse` and triggers per-stride
    /// snapshots in the Generator's prefill loop. Default 0 here so callers
    /// that don't set it (legacy paths) preserve pre-Phase-1 behavior;
    /// `main.zig` overrides via `--ssm-checkpoint-stride` for the serve path.
    ssm_checkpoint_stride: u32 = 0,
    /// Phase 1: cap on snapshots retained per request.
    ssm_checkpoint_max: u32 = 16,
    /// Iteration 2 (perf-plan Phase 4 #3): per-LoadedModel LRU cache
    /// of chat-template render+tokenize results. 0 disables the cache
    /// (useful for ablation benches / debugging). Default 4 matches
    /// `prefix_cache_capacity` — most warm-reuse benches exercise a
    /// handful of repeated prompts, and full chat conversations bump
    /// this counter anyway via LRU as new turns arrive.
    tokenize_cache_entries: u32 = 4,
    /// Headless boot: skip the startup load entirely and run the inference
    /// loop idle. The registry holds discovery stubs (or nothing); the first
    /// model — chat or media — loads on demand via `/v1/load-model`. `entry`/
    /// `config`/`tok`/`chat_config` are still required (they seed the
    /// scheduler's borrowed-view fields) but are never installed on an entry.
    no_initial_load: bool = false,
    /// Optional metrics sink. Null when --metrics is off (the default).
    /// Stored on the Scheduler and read by the `finishSlot` per-request funnel.
    metrics: ?*metrics_mod.Metrics = null,
    /// The --ctx-size launch flag (0 = not given).
    ctx_size: u32 = 0,
};

/// Signal to the parent waiting in `init()` that the inference thread is done
/// with its load (or has failed). After `started` flips, the parent reads
/// `load_failed` to decide whether to surface a startup error.
pub fn signalStarted(sch: *Scheduler) void {
    sch.started_mu.lockUncancelable(sch.io);
    defer sch.started_mu.unlock(sch.io);
    sch.started.store(true, .release);
    sch.started_cond.broadcast(sch.io);
}
/// Set the load_error_name + flip load_failed. Best-effort dupe; on OOM the
/// parent still sees `load_failed=true` and surfaces "unknown".
pub fn recordLoadError(sch: *Scheduler, err_name: []const u8) void {
    if (sch.load_error_name) |old| sch.allocator.free(old);
    sch.load_error_name = sch.allocator.dupe(u8, err_name) catch null;
    sch.load_failed.store(true, .release);
}

/// Both load construction sites (here and main.zig's startup load) stamp the
/// per-model settings onto the config the bills and defaults read.
pub fn applyModelSettings(config: *ModelConfig, o: model_settings.Override) void {
    config.ctx_override = o.ctx_size orelse 0;
    config.kv_quant_override = o.kv_quant;
    config.mtp_override = o.mtp;
    config.mtp_acceptance_override = o.mtp_acceptance;
    config.mtp_greedy_tail_override = o.mtp_greedy_tail;
    config.ssd_budget_gb_override = o.ssd_budget_gb orelse 0;
    config.preserve_thinking_override = o.preserve_thinking;
    config.think_penalty_override = o.think_penalty;
    config.logit_bias_file_override = o.logit_bias_file;
    config.vision_override = o.vision;
    if (resolveSsdBudget(0, config.ssd_budget_gb_override, config.supportsExpertStreaming()).setting_ignored)
        log.warn("[model-settings] ssd_budget_gb ignored: this checkpoint does not stream experts from SSD\n", .{});
}

/// A resident MiMo warms its verify rows with or without its heads: a PLD request verifies on them too.
pub fn mimoVerifyWarmWanted(config: *const ModelConfig) bool {
    return config.isMimo() and !config.expert_streaming;
}

fn mimoSpecWarmup(io: std.Io, xfm: *Transformer, head: ?*mimo_mtp.Head, kv_config: transformer_mod.KVQuantConfig) void {
    const start = std.Io.Timestamp.now(io, .awake);
    const rows = xfm.warmupMimoVerify(kv_config) catch |err| {
        log.warn("[spec-warmup] MiMo verify failed ({s}); the first round at each width pays its kernel compile inside the round.\n", .{@errorName(err)});
        return;
    };
    const steps = if (head) |h| h.warmup(xfm) catch |err| {
        log.warn("[spec-warmup] MiMo heads failed ({s}); the first round pays their kernel compile inside the round.\n", .{@errorName(err)});
        return;
    } else 0;
    const ms: u64 = @as(u64, @intCast(start.untilNow(io, .awake).nanoseconds)) / std.time.ns_per_ms;
    log.info("[spec-warmup] MiMo verify rows 0x{x}, {d} head steps ({d} ms).\n", .{ rows, steps, ms });
}

/// MiMo's MTP heads from the checkpoint's own shard, bound to the trunk they share.
fn loadMimoHeads(sch: *Scheduler, model_dir: []const u8, config: *const ModelConfig, xfm: *Transformer) !?*mimo_mtp.Head {
    var weights = try @import("mimo2/mimo_source.zig").loadMtpWeights(sch.io, sch.allocator, model_dir);
    defer weights.deinit();
    var head = (try mimo_mtp.Head.load(sch.allocator, mlx.gpuStream(), config, &weights)) orelse return null;
    errdefer head.deinit();
    head.target = xfm;
    const ptr = try sch.allocator.create(mimo_mtp.Head);
    ptr.* = head;
    // The coarse lm_head copy is a load cost, never the first draft's.
    const rerank = ptr.canRerankDrafts();
    log.info("[mimo-mtp] {d} heads loaded ({d:.2} GB resident); draft rerank {s}\n", .{ head.heads, @as(f64, @floatFromInt(head.residentBytes())) / 1e9, if (rerank) "on" else "off" });
    return ptr;
}

/// A load's MTP decision: `--mtp`/`--no-mtp` > the per-model `mtp` > on.
pub fn mtpChoiceFor(mtp_enabled: bool, mtp_explicit: bool, config: *const ModelConfig) model_settings.MtpChoice {
    return model_settings.MtpChoice.resolve(model_settings.launchFlag(bool, mtp_enabled, mtp_explicit), config.mtp_override, true);
}

/// The streaming gate's verdict on a load's MTP choice (`expert_stream.mtpUnderStreaming`). The gate
/// runs before the load marks its config streamed.
pub fn mtpStreamingVerdict(choice: model_settings.MtpChoice, head_supported: bool) expert_stream_mod.MtpUnderStreaming {
    return expert_stream_mod.mtpUnderStreaming(choice.on, choice.source == .model_settings, choice.source == .default, head_supported);
}

pub fn streamedHeadSupportedAt(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8) bool {
    const geometry = config.expertGeometry();
    const layout = expert_stream_mod.quant.streamingLayoutOfDir(allocator, io, config.model_type, model_dir, geometry.layers, geometry.first_moe_layer) catch return false;
    if (config.isDsv41() and !@import("dsv41/dsv41_source.zig").shipsDspark(io, allocator, model_dir)) return false;
    return expert_stream_mod.streamedMtpHeadSupported(config.isQwen4(), config.isDsv41(), layout);
}

/// An engine-default MTP under expert streaming resolves off (`expert_stream.mtpUnderStreaming`):
/// the head is not loaded and the load log reads `off (streaming; default)`.
pub fn mtpDefaultOffUnderStreaming(choice: model_settings.MtpChoice, expert_streaming: bool, head_kept: bool) bool {
    return expert_streaming and !head_kept and choice.on and choice.source == .default;
}

test "a streamed load drops only the engine-default MTP, never an asked-for one" {
    try testing.expect(mtpDefaultOffUnderStreaming(.{ .on = true, .source = .default }, true, false));
    try testing.expect(!mtpDefaultOffUnderStreaming(.{ .on = true, .source = .default }, true, true));
    try testing.expect(!mtpDefaultOffUnderStreaming(.{ .on = true, .source = .default }, false, false));
    try testing.expect(!mtpDefaultOffUnderStreaming(.{ .on = true, .source = .flag }, true, false));
    try testing.expect(!mtpDefaultOffUnderStreaming(.{ .on = false, .source = .default }, true, false));
}

/// What a load line and `/props` read for the MTP, acceptance, greedy-tail and KV keys, under the launch globals.
const LaunchPicks = struct {
    mtp: model_settings.MtpChoice,
    acceptance: model_settings.Pick(mtp_acceptance_mod.Mode),
    greedy_tail: model_settings.Pick(bool),
    kv: transformer_mod.KvCacheChoice,

    fn of(mtp_flag: ?bool, kv_flag: ?transformer_mod.KVQuantConfig, config: *const ModelConfig) LaunchPicks {
        return .{
            .mtp = mtpChoiceFor(mtp_flag orelse true, mtp_flag != null, config),
            .acceptance = mtp_plan.mtpAcceptanceFor(config.mtp_acceptance_override),
            .greedy_tail = mtp_plan.mtpGreedyTailFor(config.mtp_greedy_tail_override),
            .kv = transformer_mod.KvCacheChoice.resolve(config.kv_quant_override, kv_flag orelse transformer_mod.KVQuantConfig.engine_default, kv_flag != null),
        };
    }
};

const LaunchGlobals = struct {
    acceptance: mtp_acceptance_mod.Mode,
    acceptance_explicit: bool,
    greedy_tail_explicit: bool,

    fn save() LaunchGlobals {
        return .{
            .acceptance = generate_mod.mtp_acceptance_default,
            .acceptance_explicit = generate_mod.mtp_acceptance_explicit,
            .greedy_tail_explicit = generate_mod.mtp_greedy_tail_explicit,
        };
    }

    fn restore(self: LaunchGlobals) void {
        generate_mod.mtp_acceptance_default = self.acceptance;
        generate_mod.mtp_acceptance_explicit = self.acceptance_explicit;
        generate_mod.mtp_greedy_tail_explicit = self.greedy_tail_explicit;
    }
};

test "an explicit launch flag outranks model-settings.json for MTP, acceptance, greedy tail and KV" {
    const saved = LaunchGlobals.save();
    defer saved.restore();
    (LaunchGlobals{ .acceptance = .exact, .acceptance_explicit = false, .greedy_tail_explicit = false }).restore();
    var file = ModelConfig{};
    file.mtp_override = false;
    file.mtp_acceptance_override = .{ .tokenv3 = 0.95 };
    file.mtp_greedy_tail_override = true;
    file.kv_quant_override = .dense;

    const unflagged = LaunchPicks.of(null, null, &file);
    try testing.expect(!unflagged.mtp.on);
    try testing.expectEqual(model_settings.Source.model_settings, unflagged.mtp.source);
    try testing.expectEqual(model_settings.Source.model_settings, unflagged.acceptance.source);
    try testing.expectEqual(model_settings.Pick(bool){ .value = true, .source = .model_settings }, unflagged.greedy_tail);
    try testing.expectEqualStrings("model-settings.json", unflagged.kv.sourceName());

    generate_mod.mtp_acceptance_default = .{ .typical = .{ .delta = 0.1 } };
    generate_mod.mtp_acceptance_explicit = true;
    generate_mod.mtp_greedy_tail_explicit = true;
    const flagged = LaunchPicks.of(true, transformer_mod.KVQuantConfig.affine(4), &file);
    try testing.expect(flagged.mtp.on);
    try testing.expectEqualStrings("--mtp", flagged.mtp.sourceName());
    try testing.expectEqual(@as(f32, 0.1), flagged.acceptance.value.typical.delta);
    try testing.expectEqual(model_settings.Source.flag, flagged.acceptance.source);
    try testing.expectEqual(model_settings.Source.flag, flagged.greedy_tail.source);
    try testing.expectEqualStrings("kv4", flagged.kv.label());
    try testing.expectEqualStrings("--kv-quant", flagged.kv.sourceName());
}

test "an explicit --mtp keeps the head on a streamed load, or refuses when it cannot" {
    const asked = mtpChoiceFor(true, true, &ModelConfig{});
    try testing.expectEqual(expert_stream_mod.MtpUnderStreaming.keep, mtpStreamingVerdict(asked, true));
    try testing.expectEqual(expert_stream_mod.MtpUnderStreaming.refuse, mtpStreamingVerdict(asked, false));
    var streamed = ModelConfig{};
    streamed.expert_streaming = true;
    try testing.expect(mtpChoiceFor(true, true, &streamed).on);
}

pub const SsdBudgetChoice = struct {
    bytes: u64,
    from_setting: bool,
    setting_ignored: bool,
};

/// The launch flag wins over the per-model setting; a non-streaming model ignores the
/// setting (warned once).
pub fn resolveSsdBudget(flag_bytes: u64, setting_gb: u32, streaming: bool) SsdBudgetChoice {
    if (!streaming) return .{ .bytes = 0, .from_setting = false, .setting_ignored = setting_gb > 0 };
    if (flag_bytes > 0) return .{ .bytes = flag_bytes, .from_setting = false, .setting_ignored = false };
    return .{ .bytes = @as(u64, setting_gb) << 30, .from_setting = setting_gb > 0, .setting_ignored = false };
}

/// Plan 05 Phase D: pre-loaded CPU state bundle. Built by the conn thread
/// (CPU only — file I/O + parse, no mlx) ahead of posting a LoadRequest.
/// Ownership transfers to the entry on successful load; on failure the
/// conn thread frees via `freeCpuState`.
pub const CpuState = struct {
    config: *ModelConfig,
    tok: *Tokenizer,
    chat_config: *ChatConfig,
};

/// Phase D: parse config.json, tokenizer, and chat config from `model_dir`
/// into heap pointers ready to hand off to `LoadRequest`. Mirrors the
/// pre-load that main.zig does for the startup model in serve mode.
///
/// Errdefer pattern: for each `try ... else error`, the `deinit` errdefer
/// is registered AFTER the successful init so a downstream failure doesn't
/// call deinit on uninitialized memory. `ModelConfig.deinit` frees the config's
/// one owned field; `allocator.destroy` alone would leak it.
pub fn preloadCpuState(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !CpuState {
    // An unsupported file format is refused by name before any config.json
    // read (such a dir has none).
    if (model_discovery.isGgufModelPath(io, model_dir)) return error.ModelFormatUnsupported;

    const config = try allocator.create(ModelConfig);
    errdefer allocator.destroy(config);
    config.* = try model_mod.parseConfig(io, allocator, model_dir);

    const tok = try allocator.create(Tokenizer);
    errdefer allocator.destroy(tok);
    tok.* = try tokenizer_mod.loadTokenizer(io, allocator, model_dir);
    errdefer tok.deinit();

    const cc = try allocator.create(ChatConfig);
    errdefer allocator.destroy(cc);
    cc.* = try chat_mod.loadChatConfig(io, allocator, model_dir);
    errdefer cc.deinit();

    // Same EOS-resolution as main.zig — merge the tokenizer's chat-terminator
    // EOS into the stop set ALWAYS, even when config.json already specified an
    // eos_token_id. Some checkpoints (e.g. Qwen2.5-Coder-7B) set config.json
    // eos_token_id to <|endoftext|> but end chat turns with <|im_end|>; gating
    // on `num_eos_tokens == 0` left <|im_end|> out of the stop set and it leaked
    // into output. Additive + dedup-guarded: only ever ADDS a declared stop.
    if (cc.eos_token) |eos_str| {
        if (tok.special_tokens.get(eos_str)) |eos_id| {
            if (!config.isEosToken(eos_id)) config.addEosToken(eos_id);
        }
    }
    if (tok.special_tokens.get("<|endoftext|>")) |eot_id| {
        if (!config.isEosToken(eot_id)) config.addEosToken(eot_id);
    }
    if (tok.special_tokens.get("<pad>")) |pad_id| {
        if (pad_id > 0 and !config.isEosToken(pad_id)) {
            config.addEosToken(pad_id);
        }
    }

    return .{ .config = config, .tok = tok, .chat_config = cc };
}

pub const StreamingPlan = struct {
    layout: expert_stream_mod.quant.Layout,
    /// `vision` is the tower only when the load keeps it.
    split: model_mod.ResidentSplit,
    resolved: ExpertCacheResolution,
    cache: expert_stream_mod.CachePlan,
    vision: VisionCost,
};

/// What the pack's tower costs a streamed load: its weights in the budget, its largest image
/// encode beside it. Slot counts come from the ssd-budget ledger: `slots_without` 0 = no
/// ledger, `slots_with` null = the budget cannot hold the tower.
pub const VisionCost = struct {
    on: bool = false,
    tower_bytes: u64 = 0,
    encode_bytes: u64 = 0,
    slots_without: u16 = 0,
    slots_with: ?u16 = null,
};

/// The stored-state layout the SSD namespace is keyed by; DeepSeek's names the expert drop list and the lossy pick too.
fn diskLayoutKey(allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8) !?[]u8 {
    if (!config.isDsv41()) return if (config.cacheLayoutNamespace()) |tag| try allocator.dupe(u8, tag) else null;
    var drop: ?expert_usage.Drop = if (config.drop_expert == 0) null else try expert_usage.load(allocator, model_dir, config.expertDropRequest());
    defer if (drop) |*d| d.deinit();
    return try dsv41_prefix.layoutKey(allocator, config.dsparkOn(), if (drop) |*d| d else null, expert_stream_mod.pick_tolerance);
}

/// The ONE streamed-load plan, read by the serving load, its registry gate and `kld`: the
/// on-disk layout, what stays resident, and the expert cache the budget leaves.
pub fn planExpertStreaming(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8, budget_bytes: u64, vision: bool) !StreamingPlan {
    return planExpertStreamingWith(io, allocator, config, model_dir, budget_bytes, vision, 0);
}

/// `assistant_bytes` is a resident DFlash2 drafter: it comes out of the budget the expert cache is cut from.
pub fn planExpertStreamingWith(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8, budget_bytes: u64, vision: bool, assistant_bytes: u64) !StreamingPlan {
    var geometry = config.expertGeometry();
    if (config.drop_expert != 0 and !config.isDsv41()) return error.LaunchFlagUnsupported;
    var drop: ?expert_usage.Drop = if (config.drop_expert == 0) null else expert_usage.load(allocator, model_dir, config.expertDropRequest()) catch |err| {
        log.err("{s}\n", .{expert_usage.describe(err)});
        return error.ExpertDropRefused;
    };
    defer if (drop) |*d| d.deinit();
    if (drop) |*d| geometry.drop = d;
    var streamed = config.*;
    streamed.expert_layout = try expert_stream_mod.quant.streamingLayoutOfDir(allocator, io, config.model_type, model_dir, geometry.layers, geometry.first_moe_layer);
    if (config.stream_mtp_head and !expert_stream_mod.streamedMtpHeadSupported(config.isQwen4(), config.isDsv41(), streamed.expert_layout)) {
        log.err("[expert-stream] {s}; drop --mtp\n", .{expert_stream_mod.MTP_UNSUPPORTED});
        return error.ExpertStreamingMtpUnsupported;
    }
    var split = try model_mod.streamingResidentSplit(io, allocator, model_dir, &streamed);
    split.trunk +|= mimoCoarseHeadBytes(&streamed) +| fp8ExpertScratchBytes(streamed.expert_layout, geometry) +| assistant_bytes +|
        (if (streamed.dsv41) |c| @import("dsv41/dsv41_engram.zig").cacheBytes(c.engram) else 0);
    if (!config.has_vision) split.vision = 0;
    if (!config.stream_mtp_head) split.mtp = 0;
    const bill = expert_stream_mod.expertBillFor(allocator, model_dir, geometry, streamed.expert_layout) catch |err| {
        if (err == error.DropEmptiesRateGroup) log.err("--drop-expert {d} would leave a rate group of some layer with no expert; use a smaller n\n", .{config.drop_expert});
        return err;
    };
    const per_expert = bill.expert_bytes;
    streamed.expert_shape = bill.shape;
    streamed.expert_shape.pool_bytes = config.expertPoolBytes();
    const picked = resolveStreamedVision(budget_bytes, &streamed, split, per_expert, vision) catch |err| {
        if (err == error.SsdBudgetBelowVision) log.err("[vision] --ssd-budget-gb {d} cannot hold the vision tower ({d:.2} GB) beside two expert slots per layer; raise the budget or pass --no-vision\n", .{
            budget_bytes >> 30,
            @as(f64, @floatFromInt(split.vision)) / 1e9,
        });
        return err;
    };
    if (!picked.cost.on) split.vision = 0;
    const cache = try expert_stream_mod.cachePlanBytesForGeometry(picked.resolved.cache_bytes, streamed.expertGeometry(), per_expert);
    return .{ .layout = streamed.expert_layout, .split = split, .resolved = picked.resolved, .cache = cache, .vision = picked.cost };
}

/// PURE: the expert cache with the pack's tower (`split.vision`) kept or left off, and what
/// keeping it costs. Asked for on a budget that cannot hold it, it is refused by name.
pub fn resolveStreamedVision(budget_bytes: u64, config: *const ModelConfig, split: model_mod.ResidentSplit, per_expert: u64, want: bool) !struct { resolved: ExpertCacheResolution, cost: VisionCost } {
    var text = split;
    text.vision = 0;
    const without = try resolveExpertCache(budget_bytes, config, text, config.stream_mtp_head, per_expert);
    if (split.vision == 0) return .{ .resolved = without, .cost = .{} };
    const with: ?ExpertCacheResolution = resolveExpertCache(budget_bytes, config, split, config.stream_mtp_head, per_expert) catch |err| switch (err) {
        error.SsdBudgetBelowResident => null,
        else => return err,
    };
    if (want and with == null) return error.SsdBudgetBelowVision;
    const cost = VisionCost{
        .on = want,
        .tower_bytes = split.vision,
        .encode_bytes = request_media.largestImageEncodeBytes(config),
        .slots_without = if (without.ledger) |l| l.slots_per_layer else 0,
        .slots_with = if (with) |w| (if (w.ledger) |l| l.slots_per_layer else null) else null,
    };
    return .{ .resolved = if (want) with.? else without, .cost = cost };
}

/// One `[vision]` line per load: its choice and source, and on a streamed load what the tower costs.
fn visionLoadLine(buf: []u8, choice: model_settings.VisionChoice, streamed: bool, cost: VisionCost) []const u8 {
    const tower = @as(f64, @floatFromInt(cost.tower_bytes)) / 1e9;
    const encode = @as(f64, @floatFromInt(cost.encode_bytes)) / 1e9;
    if (!streamed or (!choice.on and choice.source != .default))
        return std.fmt.bufPrint(buf, "[vision] {s} ({s})", .{ if (choice.on) "on" else "off", choice.sourceName() }) catch "[vision]";
    if (cost.tower_bytes == 0) return "[vision] off: this checkpoint ships no vision tower weights";
    const lost = cost.slots_without -| (cost.slots_with orelse 0);
    return (if (cost.on and cost.slots_without == 0)
        std.fmt.bufPrint(buf, "[vision] on ({s}): tower {d:.2} GB resident beside the expert cache; an image encode up to {d:.2} GB is billed per request", .{ choice.sourceName(), tower, encode })
    else if (cost.on)
        std.fmt.bufPrint(buf, "[vision] on ({s}): tower {d:.2} GB in the ssd budget (-{d} slots/layer); an image encode up to {d:.2} GB is billed per request", .{ choice.sourceName(), tower, lost, encode })
    else if (cost.slots_without == 0)
        std.fmt.bufPrint(buf, "[vision] off under expert streaming; pass --vision to load the tower (+{d:.2} GB resident)", .{tower})
    else if (cost.slots_with == null)
        std.fmt.bufPrint(buf, "[vision] off under expert streaming; pass --vision to load the tower (+{d:.2} GB: more than this ssd budget holds)", .{tower})
    else
        std.fmt.bufPrint(buf, "[vision] off under expert streaming; pass --vision to load the tower (+{d:.2} GB, -{d} slots/layer)", .{ tower, lost })) catch "[vision]";
}

/// FP8 experts compute from bf16 copies of a layer's routed slots (`glm5_stream.fp8Routed`): at most
/// every expert, plus the gathered codes when the routed slots of a cache slab are not contiguous.
pub fn fp8ExpertScratchBytes(layout: expert_stream_mod.quant.Layout, geometry: expert_stream_mod.Geometry) u64 {
    if (layout != .fp8_individual) return 0;
    const codes = 3 *| @as(u64, geometry.hidden) *| geometry.intermediate;
    return @as(u64, geometry.experts) *| codes *| (2 + 1);
}

/// The expert fill peak a streamed load and its requests bill. Flash-Next's streamed prefill peaked no higher than
/// its resident one, the union workspace and bounce already inside the budget (engine-expert-streaming.md#budget).
/// DeepSeek fills preallocated slabs in place: its fill stages nothing beyond the workspace and bounce the budget holds.
pub fn billedFillPeakBytes(config: *const ModelConfig, prefill_peak_bytes: u64) u64 {
    return if (config.longCtxGated() or config.isDsv41()) 0 else prefill_peak_bytes;
}

/// What a streamed load holds beside its cache and bounce buffers while it fills. A request bill never carries the
/// workspace: it is allocated at boot and already counted in the active memory a request reads.
pub fn loadStagingBytes(config: *const ModelConfig, billed_fill_peak_bytes: u64, workspace_bytes: u64) u64 {
    return billed_fill_peak_bytes +| (if (config.isDsv41()) workspace_bytes else 0) +| config.expertPoolBytes();
}

/// Marks the config streamed: the fields every streamed forward and bill reads.
pub fn applyStreamingPlan(config: *ModelConfig, plan: StreamingPlan, budget_bytes: u64) void {
    config.expert_layout = plan.layout;
    config.expert_streaming = true;
    config.expert_cache_bytes = plan.cache.cache_bytes;
    config.expert_ssd_budget_bytes = if (plan.resolved.ledger != null) budget_bytes else 0;
    config.expert_workspace_bytes = plan.cache.workspace_bytes;
    config.expert_bounce_bytes = plan.cache.bounce_bytes;
    config.expert_fill_peak_bytes = billedFillPeakBytes(config, plan.cache.prefill_peak_bytes);
    config.expert_borrow_bytes = if (config.dsv41) |c| @import("dsv41/dsv41_borrow.zig").plannedBytes(&c, plan.layout == .exl3_k4, plan.cache.cache_bytes) else 0;
    config.expert_vision_tower_bytes = plan.split.vision;
    config.expert_vision_encode_bytes = if (plan.vision.on) plan.vision.encode_bytes else 0;
}

pub const ExpertCacheResolution = struct {
    cache_bytes: u64,
    ledger: ?expert_stream_mod.BudgetLedger,
};

/// PURE: the expert-cache byte budget for this load. `--ssd-budget-gb` is a TOTAL resident target and the cache is what is left of it
/// after the resident trunk, the MTP head (only when it stays resident), the vision tower the
/// split carries, the whole-layer prefill union, the selected-expert slab and the fill bounce buffers.
pub fn resolveExpertCache(
    budget_bytes: u64,
    config: *const model_mod.ModelConfig,
    split: model_mod.ResidentSplit,
    mtp_resident: bool,
    per_expert: u64,
) !ExpertCacheResolution {
    if (budget_bytes == 0) return .{ .cache_bytes = 0, .ledger = null };
    const ledger = try expert_stream_mod.budgetLedger(
        budget_bytes,
        split.trunk,
        if (mtp_resident) split.mtp else 0,
        split.vision,
        @intCast(config.expertLayerCount()),
        @intCast(config.keptExperts()),
        @intCast(config.num_experts_per_tok),
        per_expert,
        expert_stream_mod.BOUNCE_BYTES,
        config.expert_shape,
    );
    return .{ .cache_bytes = ledger.cache_bytes, .ledger = ledger };
}

/// The 2-bit lm_head copy a MiMo source trunk keeps, resident or streamed: the heads draft on it
/// and a greedy readout shortlists on it.
pub fn mimoCoarseHeadBytes(config: *const ModelConfig) u64 {
    if (!config.usesMimoSourceTrunk() or mimo_mtp.rerankBits() == 0) return 0;
    return mtp_mod.rerankCoarseBytes(@intCast(config.vocab_size), @intCast(config.hidden_size), mimo_mtp.rerankBits());
}

/// Frees the three CPU-state pointers.
pub fn freeCpuState(allocator: std.mem.Allocator, s: *CpuState) void {
    s.config.deinit(allocator);
    allocator.destroy(s.config);
    s.tok.deinit();
    allocator.destroy(s.tok);
    s.chat_config.deinit();
    allocator.destroy(s.chat_config);
}

/// The scheduler's borrowed-view seed for a headless boot (`no_initial_load`):
/// never installed on an entry and never loaded.
pub fn headlessStubCpuState(allocator: std.mem.Allocator) !CpuState {
    return stubCpuState(allocator, .{
        .model_type = "headless",
        .weight_prefix = "model",
        .num_hidden_layers = 1,
        .hidden_size = 1,
        .head_dim = 1,
        .num_attention_heads = 1,
        .num_key_value_heads = 1,
        .max_position_embeddings = 4096,
    });
}

/// Heap CPU state around `config` with an empty byte-level tokenizer and an
/// empty chat template: a caller that never tokenizes only needs the shapes.
fn stubCpuState(allocator: std.mem.Allocator, config_value: ModelConfig) !CpuState {
    const config = try allocator.create(ModelConfig);
    errdefer allocator.destroy(config);
    config.* = config_value;

    const tok = try allocator.create(Tokenizer);
    errdefer allocator.destroy(tok);
    var byte_map: [256]u21 = undefined;
    for (0..256) |b| byte_map[b] = @intCast(b);
    tok.* = .{
        .vocab = std.StringHashMap(u32).init(allocator),
        .id_to_token = std.AutoHashMap(u32, []const u8).init(allocator),
        .merge_ranks = @TypeOf(tok.merge_ranks).init(allocator),
        .allocator = allocator,
        .special_tokens = std.StringHashMap(u32).init(allocator),
        .byte_to_unicode = byte_map,
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = null,
        .eos_id = null,
        .parsed_json = null,
    };
    errdefer tok.deinit();

    const cc = try allocator.create(ChatConfig);
    errdefer allocator.destroy(cc);
    cc.* = .{
        .chat_template = try allocator.dupe(u8, ""),
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    return .{ .config = config, .tok = tok, .chat_config = cc };
}

/// The post-load residency bill the eviction gate reserves, in bytes: the
/// weights plus 10% headroom for KV / vision / drafter overhead.
pub fn gateEstimateBytes(bytes_on_disk: ?u64, num_hidden_layers: u32, hidden_size: u32) u64 {
    const base: u64 = if (bytes_on_disk) |b|
        b
    else
        @as(u64, num_hidden_layers) * @as(u64, hidden_size) * 4 * 4;
    return base + base / 10;
}

/// A boot `--model` entry carries no discovery `bytes_on_disk`, so the measured shard sum is
/// the answer before the layers x hidden guess (which read 1966080 for a ~64 GiB pack).
pub fn residentWeightBytes(billed: ?u64, bytes_on_disk: ?u64, measured_disk: u64, num_hidden_layers: u32, hidden_size: u32) u64 {
    if (billed) |b| return b;
    if (bytes_on_disk) |b| return b;
    if (measured_disk > 0) return measured_disk;
    return @as(u64, num_hidden_layers) * @as(u64, hidden_size) * 4 * 4;
}

test "residentWeightBytes: a boot entry without a disk hint reports its measured shards" {
    const gib: u64 = 1 << 30;
    try testing.expectEqual(64 * gib, residentWeightBytes(null, null, 64 * gib, 48, 2560));
    try testing.expectEqual(@as(u64, 97) * gib, residentWeightBytes(97 * gib, 64 * gib, 64 * gib, 48, 2560));
    try testing.expectEqual(@as(u64, 5), residentWeightBytes(null, 5, 64 * gib, 48, 2560));
    try testing.expectEqual(@as(u64, 48 * 2560 * 16), residentWeightBytes(null, null, 0, 48, 2560));
}

pub fn expertStreamingGateBytes(resident_bytes: u64, cache_bytes: u64, fill_peak_bytes: u64, bounce_bytes: u64) u64 {
    return resident_bytes +| cache_bytes +| fill_peak_bytes +| bounce_bytes;
}

/// Sum of `*.safetensors` bytes in `model_dir` — the MLX weight footprint used
/// by the load pre-flight. Returns 0 if the dir can't be read (treated as
/// "unknown" by the caller, which then skips the check). Symlinked weights
/// count (statFile follows links) — an HF hub-cache snapshot is ALL symlinks.
fn modelDiskBytes(io: std.Io, model_dir: []const u8) u64 {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    // A pack's index names the shards the loader reads; a stray shard beside
    // them (issue #274) is dead weight and must not be billed.
    var referenced: ?std.StringHashMapUnmanaged(void) = model_discovery.indexShardSet(io, dir);
    defer if (referenced) |*r| model_discovery.freeShardSet(r);
    var it = dir.iterate();
    var total: u64 = 0;
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        if (referenced) |r| if (!r.contains(entry.name)) continue;
        const st = dir.statFile(io, entry.name, .{}) catch continue;
        if (st.kind != .file) continue;
        total += @intCast(st.size);
    }
    return total;
}

test "modelDiskBytes follows HF-cache symlinks (a snapshot dir measured ZERO)" {
    // A model served straight out of the HuggingFace hub cache is a snapshot
    // dir of SYMLINKS into ../../blobs. Skipping .sym_link entries measured a
    // 121 GB checkpoint at 0 bytes, and memInsufficientForLoad treats 0 as
    // "unknown" → the preflight waved the load through into 34 GB of swap.
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.createDirPath(io, "blobs");
    try tmp.dir.writeFile(io, .{ .sub_path = "blobs/abc123", .data = "0123456789abcdef" });
    try tmp.dir.createDirPath(io, "snapshots/rev");
    try tmp.dir.symLink(io, "../../blobs/abc123", "snapshots/rev/model.safetensors", .{});
    // A dangling link (blob pruned) is skipped, never an error…
    try tmp.dir.symLink(io, "../../blobs/gone", "snapshots/rev/model-00002.safetensors", .{});
    // …and a symlink to a DIRECTORY must not be summed (statFile follows it).
    try tmp.dir.symLink(io, "../../blobs", "snapshots/rev/dir.safetensors", .{});

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const snap = try std.fmt.allocPrint(std.testing.allocator, "{s}/.zig-cache/tmp/{s}/snapshots/rev", .{ cwd, tmp.sub_path });
    defer std.testing.allocator.free(snap);

    try std.testing.expectEqual(@as(u64, 16), modelDiskBytes(io, snap));
}

test "modelDiskBytes bills only the shards the index names (issue #274)" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "m");
    try tmp.dir.writeFile(io, .{ .sub_path = "m/model-00001-of-00002.safetensors", .data = "0123456789" });
    try tmp.dir.writeFile(io, .{ .sub_path = "m/model-00002-of-00002.safetensors", .data = "01234" });
    // Dead weight: present on disk, referenced by nothing.
    try tmp.dir.writeFile(io, .{ .sub_path = "m/stray.safetensors", .data = "0123456789abcdef0123456789abcdef" });
    try tmp.dir.writeFile(io, .{ .sub_path = "m/model.safetensors.index.json", .data =
        \\{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors","c":"model-00001-of-00002.safetensors"}}
    });

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const dir = try std.fmt.allocPrint(std.testing.allocator, "{s}/.zig-cache/tmp/{s}/m", .{ cwd, tmp.sub_path });
    defer std.testing.allocator.free(dir);
    try std.testing.expectEqual(@as(u64, 15), modelDiskBytes(io, dir));
}

/// The full requested-context bill and the largest context fitting the same budget.
pub const LoadServingBill = struct {
    requested_context: u32,
    needed: u64,
    max_context: u32,
    kv_bits: u64,
    chunk: u64,
};

/// The drafter is quantized at load, so it stays resident at its quantized bytes, not its file bytes.
fn sushiAssistantLoadBytes(io: std.Io, allocator: std.mem.Allocator, directory: []const u8) !u64 {
    if (directory.len == 0) return 0;
    return (try dflash_mod.assistantLoadBytes(io, allocator, directory, dflash_mod.quantBitsFromEnv())).resident;
}

fn sushiResidentLoadBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, load_vision: bool, mtp_on: bool) !u64 {
    if (config.isMimo()) return mimoResidentLoadBytes(io, allocator, model_dir, config, load_vision, mtp_on);
    const split = try model_mod.qwenResidentWeightBytes(io, allocator, model_dir, load_vision, mtp_on);
    const head_bytes = split.mtp;
    const coarse_bits = mtp_mod.rerankCoarseBits();
    const coarse = if (mtp_on and head_bytes > 0 and coarse_bits > 0)
        mtp_mod.rerankCoarseBytes(@intCast(config.vocab_size), @intCast(config.hidden_size), coarse_bits)
    else
        0;
    return split.trunk +| head_bytes +| coarse;
}

/// The resident bytes a MiMo source-trunk load holds: trunk and vision tower as stored, the heads
/// when MTP is on, and the coarse lm_head copy the heads and the greedy readout share.
fn mimoResidentLoadBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, load_vision: bool, mtp_on: bool) !u64 {
    var bytes = try model_mod.mimoSourceResidentBytes(io, allocator, model_dir, load_vision and config.mimo_vision);
    if (mtp_on) bytes += try model_mod.mimoMtpResidentBytes(io, allocator, model_dir);
    return bytes + mimoCoarseHeadBytes(config);
}

/// A streamed GLM forward keeps no speculative state: a found assistant stays unloaded,
/// an asked-for one is refused by name.
pub fn glmStreamedDrafterOff(config: *const ModelConfig, drafter_dir: []const u8) !bool {
    if (!config.isGlm5() or !config.expert_streaming) return false;
    if (drafter_dir.len > 0) return error.GlmStreamingSpecUnsupported;
    log.info("[glm-dflash] off: unsupported under expert streaming\n", .{});
    return true;
}

/// The drafter a load binds: `--no-drafter` wins, then an explicit dir, then one shipped in the model dir.
/// Flash-Next keeps MTP unless `--drafter` asks for DFlash2, so a drafter beside its pack is never loaded.
pub const LoadDrafterDir = struct {
    dir: []const u8,
    owned: ?[]u8 = null,

    pub fn resolve(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, no_drafter: bool, drafter_dir: []const u8, model_dir: []const u8) LoadDrafterDir {
        if (no_drafter) return .{ .dir = "" };
        if (drafter_dir.len > 0) return .{ .dir = drafter_dir };
        if (config.isQwen4()) return .{ .dir = "" };
        const in_dir = dflash_mod.resolveInDirDrafter(io, allocator, model_dir) orelse return .{ .dir = "" };
        return .{ .dir = in_dir, .owned = in_dir };
    }

    pub fn deinit(self: LoadDrafterDir, allocator: std.mem.Allocator) void {
        if (self.owned) |p| allocator.free(p);
    }
};

/// A streamed Flash-Next load keeps the found DFlash2 drafter resident beside the trunk.
pub fn streamedAssistantBytes(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, no_drafter: bool, drafter_dir: []const u8, model_dir: []const u8) !u64 {
    if (!config.isQwen4()) return 0;
    const found = LoadDrafterDir.resolve(io, allocator, config, no_drafter, drafter_dir, model_dir);
    defer found.deinit(allocator);
    return sushiAssistantLoadBytes(io, allocator, found.dir);
}

/// The serving bill a cold load reserves, read on the config the load will stamp with the drafter's request terms.
fn coldServingBillBytes(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, drafter_dir: []const u8) u64 {
    const bill = scheduler_mod.load_serving_bill orelse return 0;
    var stamped = config.*;
    if (config.isQwen4() and drafter_dir.len > 0) stampDflashRequestTerms(io, allocator, &stamped, drafter_dir) catch {};
    return if (bill(&stamped, std.math.maxInt(u64))) |b| b.needed else 0;
}

/// What a resident cold load reserves in the registry: the requirement its load preflight
/// compares with free memory, so the gate and the preflight read one bill.
pub fn residentColdLoadBillBytes(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8, load_vision: bool, mtp_on: bool, no_drafter: bool, drafter_dir: []const u8) !u64 {
    var weights = if (model_mod.usesSushiQuantMemoryBill(config))
        try sushiResidentLoadBytes(io, allocator, model_dir, config, load_vision, mtp_on)
    else
        try mimoResidentLoadBytes(io, allocator, model_dir, config, load_vision, mtp_on);
    const drafter = LoadDrafterDir.resolve(io, allocator, config, no_drafter, drafter_dir, model_dir);
    defer drafter.deinit(allocator);
    var context: u64 = 0;
    if (model_mod.usesSushiQuantMemoryBill(config)) {
        weights +|= try sushiAssistantLoadBytes(io, allocator, drafter.dir);
        context = coldServingBillBytes(io, allocator, config, drafter.dir);
    }
    return loadRequirementBytes(config, weights, mtp_on, context);
}

test "a resident MiMo cold load reserves its load preflight's requirement" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var fixture = try @import("mimo2/mimo_source.zig").makeTinySourceFixture(io, a, &tmp);
    defer fixture.deinit();
    fixture.config.expert_layout = .mxfp4_individual;
    const weights = try mimoResidentLoadBytes(io, a, fixture.path, &fixture.config, false, false);
    const preflight = loadRequirementBytes(&fixture.config, weights, false, 0);
    try testing.expectEqual(preflight, try residentColdLoadBillBytes(io, a, &fixture.config, fixture.path, false, false, false, ""));
    try testing.expect(weights > 0);
}

/// Which drafter directory a COLD load should use.
///
/// `--no-drafter` is a policy — it silences every model, including one whose
/// own dir ships a sidecar `dflash.resolveInDirDrafter` would otherwise find.
/// `--drafter <path>`, by contrast, names a sidecar for the checkpoint it was
/// passed beside: handing it to whatever model is swapped in next would load a
/// mismatched assistant, so it applies only when the entry being loaded IS the
/// launch model (which happens on a reload after eviction). Every other model
/// is served by the in-dir probe.
pub fn coldLoadDrafterDir(
    no_drafter: bool,
    primary_model_dir: []const u8,
    drafter_dir: []const u8,
    entry_path: []const u8,
) []const u8 {
    if (no_drafter) return "";
    if (drafter_dir.len == 0) return "";
    if (!std.mem.eql(u8, primary_model_dir, entry_path)) return "";
    return drafter_dir;
}

/// The load's vision choice: `--vision`/`--no-vision` (process-wide, so cold loads honour it
/// too) > the per-model `vision` > on, except off on a streamed load that is not DeepSeek-V4.1
/// (its tower comes out of the expert cache; DeepSeek-V4.1 always streams and keeps images on).
pub fn visionChoiceFor(config: *const ModelConfig, streamed: bool) model_settings.VisionChoice {
    return model_settings.VisionChoice.resolve(model_settings.vision_flag, config.vision_override, !streamed or config.isDsv41());
}

/// Should a cold load bring up the checkpoint's vision tower?
pub fn coldLoadVision(config: *const ModelConfig, streamed: bool) bool {
    return config.has_vision and visionChoiceFor(config, streamed).on;
}

test "coldLoadDrafterDir: --no-drafter wins, an explicit --drafter belongs to its OWN model" {
    // `--drafter <path>` names a sidecar for the checkpoint it was passed
    // with; handing it to whatever model gets swapped in next would load a
    // mismatched assistant. Other models are served by the in-dir probe.
    // Reloading the launch model AFTER an eviction must still get it back.
    try testing.expectEqualStrings("/d", coldLoadDrafterDir(false, "/m", "/d", "/m"));
    try testing.expectEqualStrings("", coldLoadDrafterDir(false, "/m", "/d", "/other"));
    // --no-drafter is a policy, not a path: it silences every model, including
    // one whose own dir ships a sidecar the in-dir probe would find.
    try testing.expectEqualStrings("", coldLoadDrafterDir(true, "/m", "/d", "/m"));
    try testing.expectEqualStrings("", coldLoadDrafterDir(true, "/m", "/d", "/other"));
    // No --drafter at launch: nothing to carry, the in-dir probe decides.
    try testing.expectEqualStrings("", coldLoadDrafterDir(false, "/m", "", "/m"));
}

test "the expert cache resolves from --ssd-budget-gb" {
    const t = testing;
    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 48;
    cfg.num_experts = 512;
    cfg.num_experts_per_tok = 10;
    cfg.hidden_size = 2560;
    cfg.moe_intermediate_size = 640;
    const split = model_mod.ResidentSplit{ .trunk = 9_900_000_000, .mtp = 5_200_000_000 };
    const GiB: u64 = 1 << 30;
    const fused: u64 = 9_830_400;

    const from_budget = try resolveExpertCache(60 * GiB, &cfg, split, false, fused);
    try t.expect(from_budget.ledger != null);
    try t.expectEqual(@as(u16, 103), from_budget.ledger.?.slots_per_layer);
    try t.expectEqual(from_budget.ledger.?.cache_bytes, from_budget.cache_bytes);

    const with_mtp = try resolveExpertCache(60 * GiB, &cfg, split, true, fused);
    try t.expectEqual(@as(u16, 92), with_mtp.ledger.?.slots_per_layer);

    const no_budget = try resolveExpertCache(0, &cfg, split, false, fused);
    try t.expectEqual(@as(u64, 0), no_budget.cache_bytes);
    try t.expect(no_budget.ledger == null);

    try t.expectError(error.SsdBudgetBelowResident, resolveExpertCache(14 * GiB, &cfg, split, false, fused));
}

test "streamed vision: the tower comes out of the ssd budget only when asked, refused by name when it does not fit" {
    const t = testing;
    const cfg = model_mod.ModelConfig{
        .num_hidden_layers = 48,
        .num_experts = 512,
        .num_experts_per_tok = 10,
        .hidden_size = 2560,
        .moe_intermediate_size = 640,
        .has_vision = true,
        .qwen_vision = true,
        .qv_heads = 16,
        .qv_hidden = 1152,
        .qv_intermediate = 4304,
        .qv_out_hidden = 2560,
        .qv_patch = 16,
        .qv_temporal_patch = 2,
        .qv_merge = 2,
    };
    const split = model_mod.ResidentSplit{ .trunk = 9_900_000_000, .mtp = 0, .vision = 897_862_112 };
    const GiB: u64 = 1 << 30;
    const fused: u64 = 9_830_400;
    const encode = request_media.largestImageEncodeBytes(&cfg);
    try t.expect(encode > 0);

    const off = try resolveStreamedVision(60 * GiB, &cfg, split, fused, false);
    try t.expect(!off.cost.on);
    try t.expectEqual(@as(u64, 0), off.resolved.ledger.?.vision_bytes);
    const on = try resolveStreamedVision(60 * GiB, &cfg, split, fused, true);
    try t.expect(on.cost.on);
    try t.expectEqual(split.vision, on.resolved.ledger.?.vision_bytes);
    try t.expectEqual(encode, on.cost.encode_bytes);
    try t.expectEqual(off.resolved.ledger.?.slots_per_layer, on.cost.slots_without);
    try t.expectEqual(on.resolved.ledger.?.slots_per_layer, on.cost.slots_with.?);
    try t.expect(on.cost.slots_with.? < on.cost.slots_without);
    // Off, the plan still prices what --vision would take.
    try t.expectEqual(on.cost.slots_with, off.cost.slots_with);

    const fixed = split.trunk + 522 * fused + expert_stream_mod.BOUNCE_BYTES;
    const tight = fixed + 2 * 48 * fused;
    try t.expectEqual(@as(?u16, null), (try resolveStreamedVision(tight, &cfg, split, fused, false)).cost.slots_with);
    try t.expectError(error.SsdBudgetBelowVision, resolveStreamedVision(tight, &cfg, split, fused, true));
    try t.expectError(error.SsdBudgetBelowResident, resolveStreamedVision(14 * GiB, &cfg, split, fused, true));

    var bare = split;
    bare.vision = 0;
    try t.expect(!(try resolveStreamedVision(60 * GiB, &cfg, bare, fused, true)).cost.on);
}

test "the vision load line names its source, and what --vision would cost a streamed load" {
    const t = testing;
    var buf: [256]u8 = undefined;
    const cost = VisionCost{ .tower_bytes = 900_000_000, .encode_bytes = 1_700_000_000, .slots_without = 103, .slots_with = 101 };
    const off = visionLoadLine(&buf, .{ .on = false, .source = .default }, true, cost);
    try t.expectEqualStrings("[vision] off under expert streaming; pass --vision to load the tower (+0.90 GB, -2 slots/layer)", off);
    var on_cost = cost;
    on_cost.on = true;
    try t.expectEqualStrings("[vision] on (--vision): tower 0.90 GB in the ssd budget (-2 slots/layer); an image encode up to 1.70 GB is billed per request", visionLoadLine(&buf, .{ .on = true, .source = .flag }, true, on_cost));
    var cache_cost = cost;
    cache_cost.slots_with = null;
    cache_cost.slots_without = 0;
    try t.expectEqualStrings("[vision] off under expert streaming; pass --vision to load the tower (+0.90 GB resident)", visionLoadLine(&buf, .{ .on = false, .source = .default }, true, cache_cost));
    cache_cost.on = true;
    try t.expectEqualStrings("[vision] on (model-settings.json): tower 0.90 GB resident beside the expert cache; an image encode up to 1.70 GB is billed per request", visionLoadLine(&buf, .{ .on = true, .source = .model_settings }, true, cache_cost));
    cache_cost.on = false;
    cache_cost.slots_without = 103;
    try t.expectEqualStrings("[vision] off under expert streaming; pass --vision to load the tower (+0.90 GB: more than this ssd budget holds)", visionLoadLine(&buf, .{ .on = false, .source = .default }, true, cache_cost));
    try t.expectEqualStrings("[vision] off (--no-vision)", visionLoadLine(&buf, .{ .on = false, .source = .flag }, true, cost));
    try t.expectEqualStrings("[vision] on (default)", visionLoadLine(&buf, .{ .on = true, .source = .default }, false, .{}));
    try t.expectEqualStrings("[vision] off (model-settings.json)", visionLoadLine(&buf, .{ .on = false, .source = .model_settings }, false, .{}));
}

test "the ssd budget falls back to the per-model setting, and a non-streaming model ignores it" {
    const t = testing;
    const GiB: u64 = 1 << 30;

    const flag_wins = resolveSsdBudget(60 * GiB, 40, true);
    try t.expectEqual(60 * GiB, flag_wins.bytes);
    try t.expect(!flag_wins.from_setting);
    try t.expect(!flag_wins.setting_ignored);

    const from_setting = resolveSsdBudget(0, 60, true);
    try t.expectEqual(60 * GiB, from_setting.bytes);
    try t.expect(from_setting.from_setting);
    try t.expect(!from_setting.setting_ignored);

    const neither = resolveSsdBudget(0, 0, true);
    try t.expectEqual(@as(u64, 0), neither.bytes);
    try t.expect(!neither.from_setting);

    const not_streaming = resolveSsdBudget(0, 60, false);
    try t.expectEqual(@as(u64, 0), not_streaming.bytes);
    try t.expect(not_streaming.setting_ignored);
    try t.expect(!resolveSsdBudget(0, 0, false).setting_ignored);

    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 48;
    cfg.num_experts = 512;
    cfg.num_experts_per_tok = 10;
    cfg.hidden_size = 2560;
    cfg.moe_intermediate_size = 640;
    const split = model_mod.ResidentSplit{ .trunk = 9_900_000_000, .mtp = 5_200_000_000 };
    const via_setting = try resolveExpertCache(resolveSsdBudget(0, 60, true).bytes, &cfg, split, false, 9_830_400);
    try t.expectEqual(@as(u16, 103), via_setting.ledger.?.slots_per_layer);
}

test "EXL3 streaming CPU settings budget engages the EXL3 loader" {
    // The streamed load refuses an EXL3 layout by name; the budget used to engage it anyway, so a
    // settings-file budget made the pack unloadable with a "re-convert the pack" 503.
    const t = std.testing;
    var q4 = ModelConfig{
        .model_type = "qwen4_exp",
        .num_hidden_layers = 4,
        .num_experts = 4,
        .num_experts_per_tok = 2,
        .hidden_size = 128,
        .moe_intermediate_size = 128,
        .expert_layout = .exl3_k4,
    };
    q4.ssd_budget_gb_override = 60;
    try t.expect(q4.supportsExpertStreaming());
    const budget = resolveSsdBudget(0, q4.ssd_budget_gb_override, q4.supportsExpertStreaming());
    try t.expect(!budget.setting_ignored);
    try t.expectEqual(@as(u64, 60 << 30), budget.bytes);
    try t.expect(expert_stream_mod.expertStreamingEngaged(q4.supportsExpertStreaming(), q4.expertStreamingRequired(), budget.bytes));
}

test "bf16 streaming registry gate uses resident plan not checkpoint disk size" {
    try testing.expectEqual(@as(u64, 81), expertStreamingGateBytes(10, 60, 5, 6));
    try testing.expect(expertStreamingGateBytes(15_000_000_000, 60_000_000_000, 5_033_164_800, 536_870_912) < 360_000_000_000);
}

test "coldLoadVision honors --no-vision, --vision and the per-model vision, resident and streamed" {
    const saved = model_settings.vision_flag;
    defer model_settings.vision_flag = saved;
    var cfg = ModelConfig{ .has_vision = true };
    model_settings.vision_flag = null;
    try std.testing.expect(coldLoadVision(&cfg, false));
    try std.testing.expect(!coldLoadVision(&cfg, true));
    try std.testing.expect(!coldLoadVision(&ModelConfig{}, false));
    model_settings.vision_flag = false;
    try std.testing.expect(!coldLoadVision(&cfg, false));
    model_settings.vision_flag = true;
    try std.testing.expect(coldLoadVision(&cfg, true));
    try std.testing.expect(!coldLoadVision(&ModelConfig{}, true));
    model_settings.vision_flag = null;
    cfg.vision_override = true;
    try std.testing.expect(coldLoadVision(&cfg, true));
    cfg.vision_override = false;
    try std.testing.expect(!coldLoadVision(&cfg, false));
}

/// Which "available memory" figure the preflight should trust. On iOS the
/// host-wide number is meaningless — the OS keeps RAM full (file cache,
/// jetsam-evictable background apps) and will evict on our behalf, so the
/// per-PROCESS jetsam headroom (`os_proc_available_memory`, nonzero only on
/// iOS) is the figure that decides whether the load survives. Live bug: an
/// 8 GB iPhone reported ~4 GB host-free and the preflight refused a 3.6 GB
/// model that fit comfortably inside the ~6.4 GB process limit.
/// `gpu_limit` = Metal's working-set limit (0 = unknown): a lowered `iogpu.wired_limit_mb` binds
/// below free RAM, and weights past it OOM in warmup instead of refusing by name.
fn effectiveAvailableBytes(host_avail: u64, proc_avail: u64, gpu_limit: u64) u64 {
    const avail = if (proc_avail > 0) proc_avail else host_avail;
    return if (gpu_limit > 0) @min(avail, gpu_limit) else avail;
}

/// A just-exited process returns its Metal memory over seconds: while a short reading keeps
/// rising, poll every 500 ms; stop at enough, after 2 s without growth, or after 15 s.
fn settleAvailableBytes(first: u64, needed: u64, reader: anytype) u64 {
    var avail = first;
    var stalled: u32 = 0;
    var polls: u32 = 0;
    while (avail > 0 and avail < needed and stalled < 4 and polls < 30) : (polls += 1) {
        reader.sleep();
        const next = reader.read();
        stalled = if (next > avail) 0 else stalled + 1;
        avail = next;
    }
    return avail;
}

/// A resident Sushi load shares the working-set cap with already active model buffers. A declared
/// wired limit (`wired_floor`, 0 = none) lifts the free-RAM figure to what that limit leaves the
/// GPU, the same floor request admission reads.
fn effectiveSushiAvailableBytes(host_avail: u64, proc_avail: u64, gpu_limit: u64, active: u64, wired_floor: u64) u64 {
    const host = if (wired_floor > 0) @max(host_avail, wired_floor -| active) else host_avail;
    const available = effectiveAvailableBytes(host, proc_avail, gpu_limit);
    return if (gpu_limit > 0) @min(available, gpu_limit -| active) else available;
}

test "load preflight waits for a prior process's memory to come back, and only while it does" {
    const gib: u64 = 1 << 30;
    const Fake = struct {
        seq: []const u64,
        i: usize = 0,
        fn read(f: *@This()) u64 {
            const v = f.seq[@min(f.i, f.seq.len - 1)];
            f.i += 1;
            return v;
        }
        fn sleep(_: *@This()) void {}
    };
    var rising = Fake{ .seq = &.{ 80 * gib, 95 * gib, 110 * gib } };
    try testing.expectEqual(@as(u64, 110 * gib), settleAvailableBytes(69 * gib, 100 * gib, &rising));
    var short = Fake{ .seq = &.{ 70 * gib, 70 * gib, 70 * gib, 70 * gib, 70 * gib, 200 * gib } };
    try testing.expectEqual(@as(u64, 70 * gib), settleAvailableBytes(69 * gib, 100 * gib, &short));
    try testing.expectEqual(@as(usize, 5), short.i);
    var never = Fake{ .seq = &.{200 * gib} };
    try testing.expectEqual(@as(u64, 120 * gib), settleAvailableBytes(120 * gib, 100 * gib, &never));
    try testing.expectEqual(@as(usize, 0), never.i);
}

test "Sushi quant memory subtracts existing resident buffers from the GPU limit" {
    const gib: u64 = 1 << 30;
    try testing.expectEqual(@as(u64, 54 * gib), effectiveSushiAvailableBytes(88 * gib, 0, 94 * gib, 40 * gib, 0));
    try testing.expectEqual(@as(u64, 30 * gib), effectiveSushiAvailableBytes(30 * gib, 0, 94 * gib, 40 * gib, 0));
    try testing.expectEqual(@as(u64, 0), effectiveSushiAvailableBytes(88 * gib, 0, 40 * gib, 40 * gib, 0));
    try testing.expectEqual(@as(u64, 88 * gib), effectiveSushiAvailableBytes(88 * gib, 0, 0, 40 * gib, 0));
}

test "a declared wired limit lifts the load preflight's free-RAM figure, never past the working set" {
    const gib: u64 = 1 << 30;
    // 32 GB box, iogpu.wired_limit_mb raised: file cache and other apps' pages read as gone,
    // but the declared limit says the GPU may claim the floor.
    try testing.expectEqual(@as(u64, 28 * gib), effectiveSushiAvailableBytes(25 * gib, 0, 29 * gib, 0, 28 * gib));
    // Resident buffers come off the floor, as they come off the working set.
    try testing.expectEqual(@as(u64, 18 * gib), effectiveSushiAvailableBytes(15 * gib, 0, 29 * gib, 10 * gib, 28 * gib));
    // More free RAM than the floor keeps the larger figure; the working set still caps it.
    try testing.expectEqual(@as(u64, 29 * gib), effectiveSushiAvailableBytes(40 * gib, 0, 29 * gib, 0, 28 * gib));
    // A floor above the working set never raises the cap.
    try testing.expectEqual(@as(u64, 29 * gib), effectiveSushiAvailableBytes(25 * gib, 0, 29 * gib, 0, 31 * gib));
}

test "effectiveAvailableBytes is capped by the GPU working-set limit" {
    const GB: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(36 * GB, effectiveAvailableBytes(98 * GB, 0, 36 * GB));
    try std.testing.expect(memInsufficientForLoad(70 * GB, effectiveAvailableBytes(98 * GB, 0, 36 * GB), 71 * GB));
}

test "effectiveAvailableBytes prefers the per-process jetsam headroom when present" {
    const GB: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(6 * GB, effectiveAvailableBytes(4 * GB, 6 * GB, 0)); // iOS: proc wins
    try std.testing.expectEqual(4 * GB, effectiveAvailableBytes(4 * GB, 0, 0)); // macOS: proc query = 0 → host
    try std.testing.expectEqual(@as(u64, 0), effectiveAvailableBytes(0, 0, 0)); // both unknown → 0 (never blocks)
}

/// Free memory a load keeps beyond everything it bills: allocator slack and pages macOS has not yet handed back.
const LOAD_SAFETY_NET_BYTES: u64 = 1 << 30;

/// What a load's warmup allocates above its billed weights: the measured peak over the billed bytes at
/// each architecture's worst setting, rounded up (engine-memory-admission.md#load-warmup).
fn loadWarmupBytes(config: *const ModelConfig, mtp_on: bool) u64 {
    const gib: u64 = 1 << 30;
    // GLM warms T=1 and T=8, then resets; its KDA state, rings and first latent rows exist from then on.
    if (config.isGlm5()) return config.ssmCheckpointBytes() +| config.qsaRingBytes() +|
        (config.kvBytesPerToken() +| config.qsaHistoryBytesPerToken()) *| 1024 +| gib * 3 / 8;
    if (config.isMimo()) return gib / 4;
    // Flash-Next: reading the MTP head peaks about a GiB above its resident bytes, and quantizing the
    // drafter holds its dense linears beside the quantized ones until the load ends.
    return gib / 8 +| (if (mtp_on) gib * 9 / 8 else 0) +| config.dflash_load_transient_bytes;
}

/// A streamed load holds the cache, the whole-layer fill union and the bounce buffers beside the resident trunk.
pub fn streamedLoadRequirementBytes(resident_bytes: u64, cache_bytes: u64, fill_peak_bytes: u64, bounce_bytes: u64) u64 {
    return expertStreamingGateBytes(resident_bytes, cache_bytes, fill_peak_bytes, bounce_bytes) +| LOAD_SAFETY_NET_BYTES;
}

/// Free memory a load demands: its billed weights, then the larger of the warmup transient plus the safety net and
/// the requested context's admission bill (the warmup ends before any request allocates).
pub fn loadRequirementBytes(config: *const ModelConfig, weights_bytes: u64, mtp_on: bool, context_bill: u64) u64 {
    if (config.expert_streaming) return streamedLoadRequirementBytes(weights_bytes, config.expert_cache_bytes, loadStagingBytes(config, config.expert_fill_peak_bytes, config.expert_workspace_bytes), config.expert_bounce_bytes);
    return weights_bytes +| @max(loadWarmupBytes(config, mtp_on) +| LOAD_SAFETY_NET_BYTES, context_bill);
}

/// Unknown figures (a failed memory query, an unmeasurable size) never block a load.
fn memInsufficientForLoad(weights_bytes: u64, avail_bytes: u64, required_bytes: u64) bool {
    if (weights_bytes == 0 or avail_bytes == 0) return false;
    return avail_bytes < required_bytes;
}

/// Stamp the per-request terms of a sliding DFlash2 assistant on the target's config.
fn stampDflashRequestTerms(io: std.Io, allocator: std.mem.Allocator, config: *ModelConfig, directory: []const u8) !void {
    if (!dflash_mod.probeIsDflash(io, allocator, directory)) return error.DflashAssistantRequired;
    var cfg = try dflash_mod.parseConfig(io, allocator, directory);
    defer cfg.deinit(allocator);
    if (!cfg.isDflash2() or cfg.hidden_size != config.hidden_size or cfg.sliding_window < 2) return error.DflashTargetMismatch;
    for (cfg.layer_types) |kind| if (kind != .sliding_attention) return error.GlmDflashWindowUnsupported;
    config.dflash_loaded = true;
    if (config.isQwen4()) config.dflash_load_transient_bytes = (try dflash_mod.assistantLoadBytes(io, allocator, directory, dflash_mod.quantBitsFromEnv())).transient;
    const per_token = @as(u64, cfg.num_hidden_layers) * cfg.num_key_value_heads * cfg.head_dim * 4;
    // A draft block grows a 2047-row window to the next 256-row allocation.
    config.dflash_window_bytes = per_token * ((@as(u64, cfg.sliding_window) + cfg.block_size + 255) / 256 * 256);
    config.dflash_capture_bytes_per_token = (@as(u64, cfg.target_layer_ids.len) * cfg.hidden_size + @as(u64, cfg.hidden_size) * 2) * 2 + per_token;
}

fn glmDflashLoadBytes(io: std.Io, allocator: std.mem.Allocator, config: *ModelConfig, model_dir: []const u8, directory: []const u8) !u64 {
    if (directory.len == 0) return 0;
    stampDflashRequestTerms(io, allocator, config, directory) catch |err| return switch (err) {
        error.DflashAssistantRequired => error.GlmDflashAssistantRequired,
        error.DflashTargetMismatch => error.GlmDraftTargetMismatch,
        else => err,
    };
    const runtime_cache = @import("glm5/glm5_dflash_cache.zig");
    return if (runtime_cache.isShippedSource(directory, model_dir))
        try runtime_cache.plannedResidentBytes(io, allocator, directory)
    else
        try @import("glm5/glm5_diagnostic.zig").assistantResidentBytes(io, allocator, directory);
}

pub fn glmColdLoadBillBytes(io: std.Io, allocator: std.mem.Allocator, config: *ModelConfig, model_dir: []const u8, load_vision: bool, no_drafter: bool, drafter_dir: []const u8) !u64 {
    const drafter = LoadDrafterDir.resolve(io, allocator, config, no_drafter, drafter_dir, model_dir);
    defer drafter.deinit(allocator);
    const weights = try @import("glm5/glm5_diagnostic.zig").residentBytesWithVision(io, allocator, model_dir, config.num_hidden_layers, load_vision and config.glm5_vision);
    return loadRequirementBytes(config, weights +| try glmDflashLoadBytes(io, allocator, config, model_dir, drafter.dir), false, 0);
}

test "Sushi quant memory production pack CPU header audit" {
    const raw = std.c.getenv("SUSHI_MEMORY_AUDIT_MODEL") orelse return error.SkipZigTest;
    const path = std.mem.span(raw);
    const io = std.testing.io;
    const a = std.testing.allocator;
    var cfg = try model_mod.parseConfig(io, a, path);
    defer cfg.deinit(a);
    try testing.expect(model_mod.usesSushiQuantMemoryBill(&cfg));
    const text = try sushiResidentLoadBytes(io, a, path, &cfg, false, false);
    const vision = try sushiResidentLoadBytes(io, a, path, &cfg, true, false);
    const all = try sushiResidentLoadBytes(io, a, path, &cfg, true, true);
    try testing.expect(all > vision);
    try testing.expect(vision > text);
    try testing.expectEqual(loadRequirementBytes(&cfg, all, true, 0), try residentColdLoadBillBytes(io, a, &cfg, path, true, true, true, ""));
}

pub const LoadBillFixtures = struct {
    qwen: ModelConfig,
    mimo: ModelConfig,
    glm: ModelConfig,

    pub fn init() !LoadBillFixtures {
        const a = testing.allocator;
        var qwen = try model_mod.parseConfigFromJson(a, @embedFile("fixtures/model-configs/qwen4_exp.json"));
        errdefer qwen.deinit(a);
        var mimo = try model_mod.parseConfigFromJson(a, @embedFile("fixtures/model-configs/mimo_v2.json"));
        errdefer mimo.deinit(a);
        return .{ .qwen = qwen, .mimo = mimo, .glm = try model_mod.parseConfigFromJson(a, @embedFile("fixtures/glm5_config.json")) };
    }

    pub fn deinit(self: *LoadBillFixtures) void {
        self.qwen.deinit(testing.allocator);
        self.mimo.deinit(testing.allocator);
        self.glm.deinit(testing.allocator);
    }
};

test "the drafter is billed at its quantized resident bytes and its load transient joins the warmup" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    const gib: u64 = 1 << 30;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // fc [1024, 2048] bf16 quantizes at gs 64; the codebook and the norm stay dense.
    const header =
        \\{"fc.weight":{"dtype":"BF16","shape":[1024,2048],"data_offsets":[0,4194304]},
        \\"candidate_selector.predecessor_codebook":{"dtype":"BF16","shape":[256,128],"data_offsets":[4194304,4259840]},
        \\"norm.weight":{"dtype":"BF16","shape":[2048],"data_offsets":[4259840,4263936]}}
    ;
    const body = try a.alloc(u8, 8 + header.len + 4263936);
    defer a.free(body);
    @memset(body, 0);
    std.mem.writeInt(u64, body[0..8], header.len, .little);
    @memcpy(body[8 .. 8 + header.len], header);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = body });
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];

    const stored = try @import("glm5/glm5_diagnostic.zig").assistantResidentBytes(io, a, dir);
    const billed = try sushiAssistantLoadBytes(io, a, dir);
    try testing.expect(billed < stored);
    try testing.expectEqual((try dflash_mod.assistantLoadBytes(io, a, dir, dflash_mod.quantBitsFromEnv())).resident, billed);

    var f = try LoadBillFixtures.init();
    defer f.deinit();
    const bare = loadRequirementBytes(&f.qwen, 44 * gib, false, 0);
    f.qwen.dflash_load_transient_bytes = gib / 2;
    try testing.expectEqual(bare + gib / 2, loadRequirementBytes(&f.qwen, 44 * gib, false, 0));
    // An MTP-on load pays both peaks, and a context bill still replaces them.
    try testing.expectEqual(44 * gib + gib * 5 / 4 + gib / 2 + gib, loadRequirementBytes(&f.qwen, 44 * gib, true, 0));
    try testing.expectEqual(44 * gib + 6 * gib, loadRequirementBytes(&f.qwen, 44 * gib, true, 6 * gib));
}

test "a cold load reads the serving bill on the config the load stamps with the drafter's request terms" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    const gib: u64 = 1 << 30;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var f = try LoadBillFixtures.init();
    defer f.deinit();
    const Hook = struct {
        fn bill(config: *const ModelConfig, _: u64) ?LoadServingBill {
            return .{ .requested_context = 0, .needed = if (config.dflash_loaded) 9 * (1 << 30) else 8 * (1 << 30), .max_context = 0, .kv_bits = 8, .chunk = 0 };
        }
    };
    const saved = scheduler_mod.load_serving_bill;
    defer scheduler_mod.load_serving_bill = saved;
    scheduler_mod.load_serving_bill = Hook.bill;
    var cfg = f.qwen;
    cfg.hidden_size = 64;
    try testing.expectEqual(8 * gib, coldServingBillBytes(io, a, &cfg, ""));

    try tmp.dir.createDirPath(io, "drafter");
    var dd = try tmp.dir.openDir(io, "drafter", .{});
    defer dd.close(io);
    var path_buf: [512]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const drafter = try std.fmt.allocPrint(a, "{s}/drafter", .{root});
    defer a.free(drafter);
    try dflash_mod.TinyFix.writeAssistant2(io, dd, drafter, mlx.gpuStream());
    // The stamp refuses a full-attention layer; the fixture's config ships one.
    const sliding = try std.mem.replaceOwned(u8, a, dflash_mod.TinyFix.ASSISTANT2_CONFIG, "\"full_attention\"", "\"sliding_attention\"");
    defer a.free(sliding);
    try dd.writeFile(io, .{ .sub_path = "config.json", .data = sliding });
    try testing.expectEqual(9 * gib, coldServingBillBytes(io, a, &cfg, drafter));
    try testing.expect(!cfg.dflash_loaded);
}

test "a streamed load bills its trunk, expert cache, fill union and bounce buffers plus the net" {
    const gib: u64 = 1 << 30;
    const cfg = ModelConfig{ .model_type = "qwen4_exp", .expert_streaming = true, .expert_cache_bytes = 12 * gib, .expert_fill_peak_bytes = 2 * gib, .expert_bounce_bytes = gib / 2 };
    try testing.expectEqual(6 * gib + 12 * gib + 2 * gib + gib / 2 + gib, loadRequirementBytes(&cfg, 6 * gib, true, 0));
    try testing.expectEqual(loadRequirementBytes(&cfg, 6 * gib, true, 0), streamedLoadRequirementBytes(6 * gib, 12 * gib, 2 * gib, gib / 2));
}

test "the exact load bill admits a MiMo load the flat 7 GiB headroom refused" {
    const gib: u64 = 1 << 30;
    var f = try LoadBillFixtures.init();
    defer f.deinit();
    const weights = 90 * gib;
    const available = 92 * gib;
    // The retired headroom demanded weights + min(weights / 8, 6 GiB) + 1 GiB.
    try testing.expect(memInsufficientForLoad(weights, available, weights + 7 * gib));
    const exact = loadRequirementBytes(&f.mimo, weights, true, 0);
    try testing.expect(!memInsufficientForLoad(weights, available, exact));
    try testing.expect(!memInsufficientForLoad(weights, exact, exact));
    try testing.expect(memInsufficientForLoad(weights, exact - 1, exact));
    // Unknown figures (a failed memory query, an unmeasurable size) never block a load.
    try testing.expect(!memInsufficientForLoad(0, available, exact));
    try testing.expect(!memInsufficientForLoad(weights, 0, exact));
}

test "the eviction gate bills weights plus 10% headroom" {
    try testing.expectEqual(@as(u64, 1100), gateEstimateBytes(1000, 0, 0));
    try testing.expectEqual(@as(u64, 0), gateEstimateBytes(0, 0, 0));
    // No bytes_on_disk → the layers × hidden × 16 fallback.
    const fallback: u64 = 32 * 4096 * 16;
    try testing.expectEqual(fallback + fallback / 10, gateEstimateBytes(null, 32, 4096));
}

/// Phase A1 → Plan 05: do the full model load on the inference thread.
/// mlx ops here bind to this thread's GPU stream from t0; subsequent
/// forwards stay on the same thread.
///
/// `params` is duck-typed (`anytype`): both `LoadParams` (startup) and
/// `*LoadRequest` (on-demand) supply the same field set — `entry`,
/// `config`/`tok`/`chat_config` (heap pointers), `model_dir`,
/// `drafter_dir`, `load_vision`, `warmup_eager`, `kv_quant_config`,
/// `prefix_cache_capacity`, `prefix_cache_mem_bytes`. The function reads them by name.
///
/// On any error the partial state has already been freed via errdefer; the
/// caller decides how to surface (startup → recordLoadError + signal
/// started; on-demand → req.error_name + done broadcast).
pub fn doLoadOnInferenceThread(sch: *Scheduler, params: anytype) !void {
    var streaming_resident_bytes: ?u64 = null;
    var expert_source_assigned = false;
    errdefer if (expert_source_assigned) {
        sch.allocator.free(params.config.expert_source_dir.?);
        params.config.expert_source_dir = null;
    };
    if (params.config.isGlm5()) {
        const kv = transformer_mod.KvCacheChoice.resolve(params.config.kv_quant_override, params.kv_quant_config, params.kv_quant_explicit);
        if (kv.config.glmLatentBits() == null) {
            log.err("[glm] native MLA cache supports --kv-quant 8, 6 or off\n", .{});
            return error.GlmKvQuantUnsupported;
        }
    }
    if (@import("server.zig").archFlagRefusal(params.config, .{
        .kv_quant = params.kv_quant_explicit,
        .qwen_gdn_16 = !model_mod.qwen_gdn_f32,
        .mtp_head_kv_quant = transformer_mod.Transformer.mtp_head_kv_quant_flag,
        .drop_expert = params.config.drop_expert != 0,
        .mtp_typical = generate_mod.mtp_acceptance_explicit and generate_mod.mtp_acceptance_default == .typical,
        .mtp_tokenv3 = generate_mod.mtp_acceptance_explicit and generate_mod.mtp_acceptance_default == .tokenv3,
        .mtp_greedy_tail = generate_mod.mtp_greedy_tail_explicit,
    })) |why| {
        log.err("{s}\n", .{why});
        return error.LaunchFlagUnsupported;
    }
    if (@import("server.zig").settingsKvQuantIgnoredLine(params.config)) |line| log.info("{s}\n", .{line});
    if (@import("server.zig").settingsMtpIgnoredLine(params.config)) |line| log.info("{s}\n", .{line});
    const streaming_budget = resolveSsdBudget(params.ssd_budget_bytes, params.config.ssd_budget_gb_override, params.config.supportsExpertStreaming());
    const streamed = expert_stream_mod.expertStreamingEngaged(
        params.config.supportsExpertStreaming(),
        params.config.expertStreamingRequired(),
        streaming_budget.bytes,
    );
    const vision_choice = visionChoiceFor(params.config, streamed);
    var load_vision = params.load_vision and vision_choice.on;
    var vision_cost: VisionCost = .{};
    if (streamed) {
        const budget = streaming_budget;
        if (budget.bytes == 0) return error.ExpertStreamingRequired;
        const mtp = mtpChoiceFor(params.mtp_enabled, params.mtp_explicit, params.config);
        const verdict = mtpStreamingVerdict(mtp, streamedHeadSupportedAt(sch.io, sch.allocator, params.config, params.model_dir));
        switch (verdict) {
            .refuse => {
                log.err("[expert-stream] {s}; MTP is on ({s}), pass --no-mtp\n", .{ expert_stream_mod.MTP_UNSUPPORTED, mtp.sourceName() });
                return error.ExpertStreamingMtpUnsupported;
            },
            .keep => {
                log.info("[expert-stream] MTP head resident ({s}); routed experts stream\n", .{mtp.sourceName()});
                params.config.stream_mtp_head = true;
            },
            .drop_settings => {
                log.info("[expert-stream] model-settings mtp=true ignored: {s}\n", .{expert_stream_mod.MTP_UNSUPPORTED});
                params.config.mtp_override = false;
            },
            .drop_default, .off => {},
        }
        if (budget.from_setting)
            log.info("[expert-stream] ssd budget {d} GiB from model-settings.json\n", .{budget.bytes >> 30});
        const assistant_bytes = try streamedAssistantBytes(sch.io, sch.allocator, params.config, params.no_drafter, params.drafter_dir, params.model_dir);
        const plan = try planExpertStreamingWith(sch.io, sch.allocator, params.config, params.model_dir, budget.bytes, load_vision, assistant_bytes);
        vision_cost = plan.vision;
        load_vision = plan.vision.on;
        if (plan.resolved.ledger) |led| log.info("[expert-stream] ssd budget {d} GiB: trunk {d:.2} GB, mtp {d:.2} GB, vision {d:.2} GB, workspace {d:.2} GB, selected {d:.2} GB, bounce {d:.2} GB, mlx pool {d:.2} GB -> expert cache {d:.2} GB = {d} slots/layer{s}\n", .{
            led.budget_bytes >> 30,
            @as(f64, @floatFromInt(led.trunk_bytes)) / 1e9,
            @as(f64, @floatFromInt(led.mtp_bytes)) / 1e9,
            @as(f64, @floatFromInt(led.vision_bytes)) / 1e9,
            @as(f64, @floatFromInt(led.workspace_bytes)) / 1e9,
            @as(f64, @floatFromInt(led.selected_bytes)) / 1e9,
            @as(f64, @floatFromInt(led.bounce_bytes)) / 1e9,
            @as(f64, @floatFromInt(led.pool_bytes)) / 1e9,
            @as(f64, @floatFromInt(led.cache_bytes)) / 1e9,
            plan.cache.slots_per_layer,
            if (led.resident) ", every routed expert resident" else "",
        });
        applyStreamingPlan(params.config, plan, budget.bytes);
        if (plan.resolved.ledger) |led| if (led.pool_bytes > 0) @import("server.zig").capMlxPool(led.pool_bytes);
        if (params.config.expert_source_dir == null) {
            params.config.expert_source_dir = try sch.allocator.dupe(u8, params.model_dir);
            expert_source_assigned = true;
        }
        streaming_resident_bytes = plan.split.trunk +| plan.split.mtp +| plan.split.vision;
        if (params.expert_cache_fit_resolver) |fit| try fit(params.config, streaming_resident_bytes.?);
    } else if (params.config.isGlm5()) {
        streaming_resident_bytes = try @import("glm5/glm5_diagnostic.zig").residentBytesWithVision(sch.io, sch.allocator, params.model_dir, params.config.num_hidden_layers, load_vision and params.config.glm5_vision);
    } else if (model_mod.usesSushiQuantMemoryBill(params.config)) {
        streaming_resident_bytes = try sushiResidentLoadBytes(sch.io, sch.allocator, params.model_dir, params.config, load_vision, mtpChoiceFor(params.mtp_enabled, params.mtp_explicit, params.config).on);
    } else if (params.config.usesMimoSourceTrunk()) {
        streaming_resident_bytes = try mimoResidentLoadBytes(sch.io, sch.allocator, params.model_dir, params.config, load_vision, mtpChoiceFor(params.mtp_enabled, params.mtp_explicit, params.config).on);
    }
    if (params.config.has_vision) {
        var line_buf: [256]u8 = undefined;
        log.info("{s}\n", .{visionLoadLine(&line_buf, vision_choice, streamed, vision_cost)});
    } else if (model_settings.vision_flag == true) log.warn("[vision] --vision ignored: this checkpoint has no vision tower\n", .{});

    // Resolve the sidecar before preflight so billing and loading see the same dependency.
    const drafter = LoadDrafterDir.resolve(sch.io, sch.allocator, params.config, params.no_drafter or try glmStreamedDrafterOff(params.config, params.drafter_dir), params.drafter_dir, params.model_dir);
    defer drafter.deinit(sch.allocator);
    var drafter_dir = drafter.dir;
    var prepared_drafter_path: ?[]u8 = null;
    defer if (prepared_drafter_path) |path| sch.allocator.free(path);
    if (drafter.owned != null) log.info("[dflash] auto-detected assistant: {s}\n", .{drafter_dir});
    if (dflash_mod.drafterDepthExplicit() and drafter_dir.len == 0) log.warn("[dflash] --drafter-min-depth/--drafter-max-depth ignored: no drafter is loaded for this model\n", .{});
    if (model_mod.usesSushiQuantMemoryBill(params.config) and drafter_dir.len > 0 and !streamed)
        streaming_resident_bytes = streaming_resident_bytes.? +| try sushiAssistantLoadBytes(sch.io, sch.allocator, drafter_dir);
    if (params.config.isQwen4() and drafter_dir.len > 0) try stampDflashRequestTerms(sch.io, sch.allocator, params.config, drafter_dir);
    if (params.config.isQwen4() and drafter_dir.len == 0) {
        const why: []const u8 = if (params.no_drafter) "--no-drafter" else "DFlash2: --drafter <path>";
        const mtp_on = mtpChoiceFor(params.mtp_enabled, params.mtp_explicit, params.config).on;
        log.info("[spec] Qwen3.8-Flash-Next decodes with {s} ({s})\n", .{ if (mtp_on) "MTP" else "plain ticks", why });
    }
    if (params.config.isGlm5()) {
        const assistant_bytes = try glmDflashLoadBytes(sch.io, sch.allocator, params.config, params.model_dir, drafter_dir);
        streaming_resident_bytes = streaming_resident_bytes.? +| assistant_bytes;
        if (assistant_bytes > 0) log.info("[glm-dflash] resident assistant {d:.3} GiB; window {d} MiB; BF16 context; verification scratch billed per request\n", .{ @as(f64, @floatFromInt(assistant_bytes)) / (1024 * 1024 * 1024), params.config.dflash_window_bytes >> 20 });
        if (drafter_dir.len == 0 and !params.no_drafter and !params.config.expert_streaming)
            log.warn("[glm-dflash] off: no assistant found (expected {s}/{s}, or an intact {s}/{s} cache); re-run `sushi pull` to fetch it\n", .{ params.model_dir, dflash_mod.SHIPPED_GLM_SUBDIR, params.model_dir, dflash_mod.DFLASH2_IN_DIR_SUBDIR });
    }

    // GPU-memory pre-flight (MLX path). A Metal OOM during weight load / warmup
    // is thrown by MLX as a C++ exception that can't be caught across the C ABI,
    // so it terminates the whole process. Refuse the load up front instead, with
    // an actionable error, when free RAM clearly can't hold the weights + warmup
    // headroom — catches the common "restarted before the prior server released
    // its memory" case. Bypass with --skip-mem-preflight.
    if (!scheduler_mod.skip_mem_preflight) {
        const weights_bytes = streaming_resident_bytes orelse modelDiskBytes(sch.io, params.model_dir);
        const gpu_limit = mlx.maxRecommendedWorkingSet();
        const Reader = struct {
            io: std.Io,
            gpu_limit: usize,
            sushi: bool,
            wired_floor: u64,
            fn read(r: *@This()) u64 {
                if (!r.sushi) return effectiveAvailableBytes(status.getAvailableMemBytes(), status.getProcAvailableMemBytes(), r.gpu_limit);
                var active: usize = 0;
                _ = mlx.mlx_get_active_memory(&active);
                return effectiveSushiAvailableBytes(status.getAvailableMemBytes(), status.getProcAvailableMemBytes(), r.gpu_limit, active, r.wired_floor);
            }
            fn sleep(r: *@This()) void {
                std.Io.sleep(r.io, .fromMilliseconds(500), .real) catch {};
            }
        };
        var reader = Reader{ .io = sch.io, .gpu_limit = gpu_limit, .sushi = model_mod.usesSushiQuantMemoryBill(params.config), .wired_floor = @import("server.zig").wiredCeilingFloorFor(params.config) };
        var avail_bytes = reader.read();
        const mtp_on = mtpChoiceFor(params.mtp_enabled, params.mtp_explicit, params.config).on;
        var serving = if (scheduler_mod.load_serving_bill) |bill| bill(params.config, avail_bytes -| weights_bytes) else null;
        var needed = loadRequirementBytes(params.config, weights_bytes, mtp_on, if (serving) |b| b.needed else 0);
        if (memInsufficientForLoad(weights_bytes, avail_bytes, needed)) {
            log.info("[preflight] {d:.2} GB available of {d:.2} GB needed; waiting while memory is still being released\n", .{
                @as(f64, @floatFromInt(avail_bytes)) / (1024.0 * 1024.0 * 1024.0),
                @as(f64, @floatFromInt(needed)) / (1024.0 * 1024.0 * 1024.0),
            });
            avail_bytes = settleAvailableBytes(avail_bytes, needed, &reader);
            serving = if (scheduler_mod.load_serving_bill) |bill| bill(params.config, avail_bytes -| weights_bytes) else null;
            needed = loadRequirementBytes(params.config, weights_bytes, mtp_on, if (serving) |b| b.needed else 0);
        }
        var active: usize = 0;
        if (reader.sushi) _ = mlx.mlx_get_active_memory(&active);
        log.info("[preflight] weights ~{d:.2} GB, needs ~{d:.2} GB, available {d:.2} GB (warmup + net or context ~{d:.2} GB)\n", .{
            @as(f64, @floatFromInt(weights_bytes)) / (1024.0 * 1024.0 * 1024.0),
            @as(f64, @floatFromInt(needed)) / (1024.0 * 1024.0 * 1024.0),
            @as(f64, @floatFromInt(avail_bytes)) / (1024.0 * 1024.0 * 1024.0),
            @as(f64, @floatFromInt(needed -| weights_bytes)) / (1024.0 * 1024.0 * 1024.0),
        });
        const gpu_exhausted = model_mod.usesSushiQuantMemoryBill(params.config) and gpu_limit > 0 and active >= gpu_limit;
        if (weights_bytes > 0 and (memInsufficientForLoad(weights_bytes, avail_bytes, needed) or gpu_exhausted)) {
            const gb = 1024.0 * 1024.0 * 1024.0;
            if (serving) |b| {
                const maximum: u32 = if (avail_bytes < loadRequirementBytes(params.config, weights_bytes, mtp_on, 0)) 0 else b.max_context;
                sch.registry.mutex.lockUncancelable(sch.io);
                const refusal = model_registry_mod.MemoryContextRefusal{ .requested = b.requested_context, .maximum = maximum, .needed = needed, .available = avail_bytes, .kv_bits = b.kv_bits, .chunk = b.chunk };
                params.entry.memory_context_refusal = refusal;
                if (@TypeOf(params) == *LoadRequest) params.memory_context_refusal = refusal;
                sch.registry.mutex.unlock(sch.io);
                log.err("Insufficient memory for requested context: {d} tokens at KV{d} and prefill chunk {d} need ~{d:.2} GiB including {d:.2} GiB of enabled model/vision/MTP weights, but only {d:.2} GiB is available. Maximum context with this memory budget: {d} tokens. Lower --ctx-size or free memory; --skip-mem-preflight overrides this check.\n", .{
                    b.requested_context,                         b.kv_bits,                                 b.chunk,                                                                              @as(f64, @floatFromInt(needed)) / gb,
                    @as(f64, @floatFromInt(weights_bytes)) / gb, @as(f64, @floatFromInt(avail_bytes)) / gb, maximum,
                });
                return error.InsufficientMemory;
            }
            log.err("Insufficient memory to load model: needs ~{d:.1} GB free ({d:.1} GB of weights plus headroom for warmup buffers and a baseline KV cache) but only {d:.1} GB is available (free RAM, capped at the GPU working-set limit that iogpu.wired_limit_mb sets). Close other models/apps (or wait for a prior sushi to fully exit) and retry; pass --skip-mem-preflight to override.\n", .{
                @as(f64, @floatFromInt(needed)) / gb,
                @as(f64, @floatFromInt(weights_bytes)) / gb,
                @as(f64, @floatFromInt(avail_bytes)) / gb,
            });
            return error.InsufficientMemory;
        }
    }

    if (params.config.isGlm5() and @import("glm5/glm5_dflash_cache.zig").isShippedSource(drafter_dir, params.model_dir)) {
        const prepared = try @import("glm5/glm5_dflash_cache.zig").prepare(sch.io, sch.allocator, drafter_dir, mlx.gpuStream());
        prepared_drafter_path = prepared.path;
        if (prepared.fallback) {
            const cached_bytes = try @import("glm5/glm5_dflash_cache.zig").plannedResidentBytes(sch.io, sch.allocator, drafter_dir);
            const bf16_bytes = try @import("glm5/glm5_diagnostic.zig").assistantResidentBytes(sch.io, sch.allocator, prepared.path);
            streaming_resident_bytes = streaming_resident_bytes.? -| cached_bytes +| bf16_bytes;
            if (!scheduler_mod.skip_mem_preflight) {
                const available = effectiveAvailableBytes(status.getAvailableMemBytes(), status.getProcAvailableMemBytes(), mlx.maxRecommendedWorkingSet());
                const needed = loadRequirementBytes(params.config, streaming_resident_bytes.?, false, 0);
                log.info("[preflight] BF16 DFlash2 fallback: weights ~{d:.2} GiB, needs ~{d:.2} GiB, available {d:.2} GiB\n", .{ @as(f64, @floatFromInt(streaming_resident_bytes.?)) / (1 << 30), @as(f64, @floatFromInt(needed)) / (1 << 30), @as(f64, @floatFromInt(available)) / (1 << 30) });
                if (available > 0 and needed > available) return error.InsufficientMemory;
            }
        }
        drafter_dir = prepared.path;
        _ = mlx.mlx_clear_cache();
    }

    // Allocate the drafter_path dupe up front so the post-publish step
    // (lower down) has no fallible operations — once we start assigning
    // pointers onto `params.entry`, an OOM during a dupe would leave the
    // entry holding pointers that the per-ptr errdefers would double-free.
    var drafter_path_owned: []u8 = &[_]u8{};
    errdefer if (drafter_path_owned.len > 0) sch.allocator.free(drafter_path_owned);
    if (params.drafter_dir.len > 0) {
        drafter_path_owned = try sch.allocator.dupe(u8, drafter_dir);
    }

    // Weights — first mlx call. Binds the stream on this thread.
    const weights_ptr = try sch.allocator.create(Weights);
    errdefer sch.allocator.destroy(weights_ptr);
    var load_config = params.config.*;
    // The resolved launch choice gates lazy native-head tensors before Transformer.init binds them.
    load_config.mtp_override = mtpChoiceFor(params.mtp_enabled, params.mtp_explicit, params.config).on;
    weights_ptr.* = try model_mod.loadWeightsForConfig(sch.io, sch.allocator, params.model_dir, &load_config, load_vision);
    errdefer weights_ptr.deinit();
    model_mod.resolveWeightPrefix(params.config, weights_ptr);

    // Transformer — owns the bulk of the GPU memory.
    const xfm_ptr = try sch.allocator.create(Transformer);
    errdefer sch.allocator.destroy(xfm_ptr);
    xfm_ptr.* = try Transformer.init(sch.io, sch.allocator, params.config.*, weights_ptr);
    errdefer xfm_ptr.deinit();
    if (xfm_ptr.expert_stream) |engine| {
        engine.borrow_publish = &params.config.expert_borrow_bytes;
        engine.publishBorrowable();
    }

    // Reserved-token suppression mask (never sample `<|fim_hole|>`-class
    // specials): derived per model from tokenizer + template + eos.
    params.config.mergeTokenizerStops(params.tok);
    sampler.installSuppressMask(xfm_ptr, params.tok, params.chat_config.chat_template, params.config.eosTokenSlice());
    generate_mod.installThinkMarkers(xfm_ptr, params.tok);
    try generate_mod.installLogitBias(sch.io, xfm_ptr, params.tok);

    // Propagate the kv-quant config to the Transformer's own cache. Slot
    // caches in serve mode honor this independently in `Slot.init`; this
    // call covers any path that still touches `xfm.cache` directly (legacy
    // single-slot fallbacks, prompt-cache reuse).
    // An explicit launch flag outranks the per-model settings stamped on the config at BOTH construction sites.
    const kv_cache = @import("server.zig").kvCacheForLaunch(params.config, params.kv_quant_config, params.kv_quant_explicit);
    const load_ctx = model_settings.contextPick(sch.ctx_size_flag, params.config.ctx_override);
    var ctx_buf: [16]u8 = undefined;
    log.info("[kv-cache] {s} ({s}); ctx {s} ({s})\n", .{
        kv_cache.label(),                                      kv_cache.sourceName(),
        model_settings.contextLabel(&ctx_buf, load_ctx.value), model_settings.sourceLabel(load_ctx.source, "--ctx-size"),
    });
    const kv_quant_config = kv_cache.config;
    const mtp = mtpChoiceFor(params.mtp_enabled, params.mtp_explicit, params.config);
    const mtp_streaming_off = mtpDefaultOffUnderStreaming(mtp, params.config.expert_streaming, params.config.stream_mtp_head);
    const acceptance = mtp_plan.mtpAcceptanceFor(params.config.mtp_acceptance_override);
    const greedy_tail = mtp_plan.mtpGreedyTailFor(params.config.mtp_greedy_tail_override);
    // GLM has no MTP head; the server's `MTP: off` line already says so.
    if (!params.config.isGlm5() and !params.config.isDsv41()) log.info("[mtp] {s} ({s}{s}); acceptance {s} ({s}); greedy tail {s} ({s})\n", .{
        if (mtp_streaming_off) "off" else mtp.label(), if (mtp_streaming_off) "streaming; " else "", mtp.sourceName(),
        mtp_acceptance_mod.name(acceptance.value),     model_settings.sourceLabel(acceptance.source, model_settings.acceptanceFlagName(acceptance.value)),
        if (greedy_tail.value) "on" else "off",        model_settings.sourceLabel(greedy_tail.source, "--mtp-greedy-tail"),
    });
    const mtp_enabled = mtp.on and !mtp_streaming_off and !params.config.isGlm5() and !params.config.isDsv41();
    if (params.config.isDsv41()) {
        const c = &params.config.dsv41.?;
        log.info("[dsv41] {d} bytes/token of compressed state\n", .{params.config.kvBytesPerToken()});
        if (params.config.dsparkOn()) {
            log.info("[spec] DeepSeek-V4.1-Flash decodes with DSpark ({d} stages, block {d}, resident; {s})\n", .{ c.n_mtp, c.dspark.block, mtp.sourceName() });
            if (depth_bounds.active.explicit()) {
                const range = dspark_rows.Range.of(depth_bounds.active, c.dspark.block);
                log.info("[spec] DSpark verifies {d}..{d} rows a round (--mtp-min-depth/--mtp-max-depth{s})\n", .{ range.min_rows, range.max_rows, if (range.clamped) "; a depth past the block clamps to it" else "" });
            }
        } else {
            const why: []const u8 = if (c.dspark.block == 0 or !@import("dsv41/dsv41_source.zig").shipsDspark(sch.io, sch.allocator, params.model_dir)) "no DSpark stages in this checkpoint" else "--no-mtp or model-settings mtp:false";
            log.info("[spec] DeepSeek-V4.1-Flash decodes with plain ticks ({s})\n", .{why});
        }
    }
    if (params.config.isGlm5()) {
        const latent_bits: u64 = if (kv_quant_config.isQuant()) kv_quant_config.bits else 16;
        log.info("[glm] native {s} MLA: {d} latent + {d} pooled-index bytes/token; serial decode\n", .{ if (!kv_quant_config.isQuant()) "BF16" else if (kv_quant_config.bits == 6) "kv6" else "kv8", @import("server.zig").kvBytesPerTokenAtBits(params.config.kvBytesPerToken(), latent_bits), params.config.qsaHistoryBytesPerToken() });
        log.info("[glm] NAX arms {s}\n", .{if (@import("glm5/glm5_model.zig").naxArms()) "on: packed sparse and B1/B3 attention, NAX index scores, KDA cluster, A6 dense-once, MLA head batches" else "off: FP32 composite sparse and B1/B3 attention, scalar index scores, staged KDA cluster, affine QMM trunk"});
        if (mtp.on) log.warn("[glm] MTP head is not integrated with serving; MTP off\n", .{});
        if (params.prefix_cache_capacity > 0 and params.ssm_checkpoint_stride > 0) log.info("[glm] prefix cache: KDA checkpoints every 8192 tokens and at the prompt end (<= {d} held in memory, {d} MiB each)\n", .{ @import("glm5/glm5_prefix.zig").checkpointMax(params.ssm_checkpoint_max), params.config.ssmCheckpointBytes() >> 20 });
    }
    if (std.mem.indexOf(u8, params.chat_config.chat_template, "preserve_thinking") != null) {
        const keep = model_settings.pick(bool, model_settings.preserve_thinking_flag, params.config.preserve_thinking_override, true);
        log.info("[chat] preserve_thinking {s} ({s})\n", .{ if (keep.value) "on" else "off", model_settings.sourceLabel(keep.source, "--preserve-thinking") });
    }
    const think = model_settings.pick(f32, model_settings.think_penalty_flag, params.config.think_penalty_override, 0);
    if (think.value > 0) log.info("[think-penalty] lambda {d} ({s}); a request's think_penalty outranks it\n", .{ think.value, model_settings.sourceLabel(think.source, "--think-penalty") });
    if (kv_quant_config.scheme != .off) {
        try xfm_ptr.cache.reinit(params.config.num_hidden_layers, kv_quant_config);
    }
    Transformer.mtp_head_kv_quant_flag = params.mtp_head_kv_quant;
    try xfm_ptr.qwen4MtpApplyKvQuant(kv_quant_config);

    // Wire model weights into GPU memory (prevents paging, matches mlx-lm).
    // Policy in mlx.applyWiredPolicy; re-applied in runLoadRequest /
    // runUnloadRequest so `fit` capacity tracks the live set across
    // load/unload churn (this early call covers warmup's forwards).
    logWiredPolicy(mlx.applyWiredPolicy());

    // JIT-compile activation kernels. These are bound to THIS thread's mlx
    // stream — that's exactly the point of doing them here. Skipped entirely
    // when there's no GPU backend (iOS Simulator's CPU-only MLX): mlx_compile
    // is Metal kernel fusion and requests a GPU stream that doesn't exist; the
    // runtime falls back to the uncompiled paths when compiled_* stays null.
    if (!mlx.noGpuBackend()) {
        if (params.config.hidden_act == .gelu_approx) {
            xfm_ptr.compileGelu();
            xfm_ptr.compileGeglu();
        }
        if (xfm_ptr.moe_layers != null) {
            xfm_ptr.compileMoeRouting();
        }
        if (params.config.linear_num_key_heads > 0) {
            xfm_ptr.compileGdnGate();
        }
        if (params.config.isQwen4()) {
            xfm_ptr.compileQwen4Hc();
        }
    }

    if (transformer_mod.diagEnvOn("SUSHI_PREFILL_UBENCH")) prefillUbench(sch.allocator, xfm_ptr, params.config, params.tok);
    if (transformer_mod.diagEnvOn("SUSHI_GLM_ROWS_UBENCH")) if (xfm_ptr.glm5) |target| glmRowsUbench(sch.allocator, target, kv_quant_config.glmLatentBits() orelse 0, params.tok);

    // DIAGNOSTIC (SUSHI_DECODE_FWD_UBENCH=N): time N decode-width forward
    // passes back to back, with NO sampling, detokenization, stop-checking or
    // cache bookkeeping around them. The server reports `predicted_ms` around
    // the whole decode LOOP, so this is the only way to say how much of a
    // token is the model and how much is everything else. Resets the KV cache
    // afterwards so the probe cannot pollute real requests.
    if (std.c.getenv("SUSHI_DECODE_FWD_UBENCH")) |raw| {
        const n = std.fmt.parseInt(usize, std.mem.sliceTo(raw, 0), 10) catch 0;
        if (n > 0) {
            const io_u = @import("io_util.zig");
            const tio = std.Io.Threaded.global_single_threaded.io();
            var ctx = xfm_ptr.defaultCtx();
            // SUSHI_DECODE_FWD_UBENCH_S=<rows>[,<rows>...]: verify-width forwards
            // (per-position SSM capture on, as spec verify runs them), one pass per width.
            // SUSHI_DECODE_FWD_UBENCH_KV=<tokens>: prefill that many
            // tokens first so the meter runs at a real context length.
            // SUSHI_DECODE_FWD_UBENCH_PROFILE=1: one more pass per width under the
            // sub-block profiler (`[decode-prof]`).
            var widths: [16]usize = @splat(1);
            var n_widths: usize = 1;
            if (std.c.getenv("SUSHI_DECODE_FWD_UBENCH_S")) |r| {
                n_widths = 0;
                var it = std.mem.tokenizeScalar(u8, std.mem.sliceTo(r, 0), ',');
                while (it.next()) |w| {
                    if (n_widths == widths.len) break;
                    widths[n_widths] = @max(1, std.fmt.parseInt(usize, w, 10) catch 1);
                    n_widths += 1;
                }
                n_widths = @max(n_widths, 1);
            }
            const profile_pass = transformer_mod.diagEnvOn("SUSHI_DECODE_FWD_UBENCH_PROFILE");
            const kv_pre: usize = blk: {
                const r = std.c.getenv("SUSHI_DECODE_FWD_UBENCH_KV") orelse break :blk 0;
                break :blk std.fmt.parseInt(usize, std.mem.sliceTo(r, 0), 10) catch 0;
            };
            if (kv_pre > 0) {
                var done_pre: usize = 0;
                const pre_buf = try sch.allocator.alloc(i32, 2048);
                defer sch.allocator.free(pre_buf);
                for (pre_buf, 0..) |*v, i| v.* = @intCast(1 + (i % 1000));
                while (done_pre < kv_pre) {
                    const n_chunk = @min(2048, kv_pre - done_pre);
                    const psh = [_]c_int{ 1, @intCast(n_chunk) };
                    const ti = mlx.mlx_array_new_data(pre_buf.ptr, &psh, 2, .int32);
                    defer _ = mlx.mlx_array_free(ti);
                    const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                    _ = mlx.mlx_array_eval(lg);
                    _ = mlx.mlx_array_free(lg);
                    done_pre += n_chunk;
                }
                log.info("[fwd-ubench] prefilled {d} tokens\n", .{done_pre});
            }
            for (widths[0..n_widths]) |rows| {
            const tok_slice = try sch.allocator.alloc(i32, @min(rows, 4096));
            defer sch.allocator.free(tok_slice);
            for (tok_slice, 0..) |*v, i| v.* = @intCast(1 + (i % 997));
            const tok = tok_slice.ptr;
            const tsh = [_]c_int{ 1, @intCast(tok_slice.len) };
            ctx.capture_ssm_seq = rows > 1 and rows <= 16 and ctx.ssm_entries != null; // verify widths capture, prefill chunks do not
            ctx.verify_rows = xfm_ptr.config.isMimo() and rows > 1 and rows <= transformer_mod.MIMO_VERIFY_ROWS_MAX;
            log.info("[fwd-ubench] rows={d} capture={} verify_rows={}\n", .{ tok_slice.len, ctx.capture_ssm_seq, ctx.verify_rows });
            // Warm: first forward pays kernel JIT + lazy weight materialization.
            for (0..3) |_| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                _ = mlx.mlx_array_eval(lg);
                _ = mlx.mlx_array_free(lg);
            }
            // Split CPU graph CONSTRUCTION from GPU execution. MLX is lazy, so
            // `forwardWith` only issues ops — if that half dominates, the token
            // is bounded by op count / FFI overhead, not by memory bandwidth,
            // and no kernel-level optimization can reach it.
            var sw = io_u.Stopwatch.init(tio);
            var build_ns: u64 = 0;
            var eval_ns: u64 = 0;
            var submit_ns: u64 = 0;
            var submit_cpu_ns: u64 = 0;
            var ops_total: u64 = 0;
            var done: usize = 0;
            for (0..n) |_| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const ops_before = mlx.op_count.load(.monotonic);
                var swb = io_u.Stopwatch.init(tio);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                build_ns += swb.read();
                ops_total += mlx.op_count.load(.monotonic) - ops_before;
                var swe = io_u.Stopwatch.init(tio);
                const lg_vec = mlx.mlx_vector_array_new_value(lg);
                var cpu0: std.c.timespec = undefined;
                _ = std.c.clock_gettime(.THREAD_CPUTIME_ID, &cpu0);
                _ = mlx.mlx_async_eval(lg_vec);
                var cpu1: std.c.timespec = undefined;
                _ = std.c.clock_gettime(.THREAD_CPUTIME_ID, &cpu1);
                submit_cpu_ns += @intCast((cpu1.sec - cpu0.sec) * std.time.ns_per_s + (cpu1.nsec - cpu0.nsec));
                _ = mlx.mlx_vector_array_free(lg_vec);
                submit_ns += swe.read();
                _ = mlx.mlx_array_eval(lg);
                eval_ns += swe.read();
                _ = mlx.mlx_array_free(lg);
                done += 1;
            }
            const dn: f64 = @floatFromInt(@max(done, 1));
            const ms = @as(f64, @floatFromInt(sw.read())) / 1.0e6 / dn;
            log.info("[fwd-ubench] {d} decode forwards, eval-per-step: {d:.3} ms/forward (build {d:.3} ms CPU + eval {d:.3} ms GPU of which submit {d:.3} ms ({d:.3} ms thread CPU), {d:.0} ops/forward)\n", .{
                done,
                ms,
                @as(f64, @floatFromInt(build_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(eval_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(submit_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(submit_cpu_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(ops_total)) / dn,
            });
            // SUSHI_DECODE_FWD_GRAPH=<abs path>: one forward's unevaluated tape, one primitive per
            // line; a primitive is at most one dispatch, so a histogram of it is the dispatch plan.
            if (std.c.getenv("SUSHI_DECODE_FWD_GRAPH")) |path| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const lg = try xfm_ptr.forwardWith(&ctx, ti);
                defer _ = mlx.mlx_array_free(lg);
                if (std.c.fopen(path, "w")) |f| {
                    const outs = mlx.mlx_vector_array_new_value(lg);
                    defer _ = mlx.mlx_vector_array_free(outs);
                    const namer = mlx.mlx_node_namer_new();
                    defer _ = mlx.mlx_node_namer_free(namer);
                    _ = mlx.mlx_print_graph(f, namer, outs);
                    _ = std.c.fclose(f);
                    log.info("[fwd-ubench] graph written to {s}\n", .{std.mem.sliceTo(path, 0)});
                }
                _ = mlx.mlx_array_eval(lg);
            }

            // Same forward with the vocab projection suppressed. lm_head is
            // terminal — nothing downstream depends on it — so dropping it
            // cannot change the work the rest of the graph does, which makes
            // this the one sound ablation in the probe.
            ctx.skip_lm_head = true;
            for (0..3) |_| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                _ = mlx.mlx_array_eval(lg);
                _ = mlx.mlx_array_free(lg);
            }
            var sw_nolm = io_u.Stopwatch.init(tio);
            var done_nolm: usize = 0;
            for (0..n) |_| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                _ = mlx.mlx_array_eval(lg);
                _ = mlx.mlx_array_free(lg);
                done_nolm += 1;
            }
            const ms_nolm = @as(f64, @floatFromInt(sw_nolm.read())) / 1.0e6 / @as(f64, @floatFromInt(@max(done_nolm, 1)));
            ctx.skip_lm_head = false;
            log.info("[fwd-ubench] without lm_head: {d:.3} ms/forward  => lm_head = {d:.3} ms\n", .{ ms_nolm, ms - ms_nolm });
            if (profile_pass) {
                transformer_mod.decodeProfileSession(@intCast(rows));
                for (0..n) |_| {
                    const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                    defer _ = mlx.mlx_array_free(ti);
                    const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                    _ = mlx.mlx_array_eval(lg);
                    _ = mlx.mlx_array_free(lg);
                }
                transformer_mod.decodeProfileSession(0);
            }
            }
            log.info("[fwd-ubench] done\n", .{});
            xfm_ptr.resetCache() catch {};
        }
    }

    // Vision encoder if requested. `MissingVisionWeights` is a benign opt-out
    // (model declares vision in config but the safetensors didn't ship the
    // tower); other errors fail the whole load.
    var vision_ptr: ?*VisionEncoder = null;
    if (load_vision) {
        const v = try sch.allocator.create(VisionEncoder);
        if (VisionEncoder.init(sch.allocator, params.config.*, weights_ptr)) |encoder| {
            v.* = encoder;
            vision_ptr = v;
        } else |err| {
            sch.allocator.destroy(v);
            if (err == error.MissingVisionWeights) {
                log.warn("Vision weights missing — vision disabled (model may have been quantized without vision tower)\n", .{});
            } else {
                return err;
            }
        }
    }
    errdefer if (vision_ptr) |v| {
        v.deinit();
        sch.allocator.destroy(v);
    };

    // The measured round-cost table (`round_cost.zig`) is keyed per (chip,
    // model, quant, OS build); restored here, written at request end.
    {
        var quant_buf: [32]u8 = undefined;
        const quant = std.fmt.bufPrint(&quant_buf, "q{d}g{d}", .{
            params.config.quant_bits,
            params.config.quant_group_size,
        }) catch "q?";
        var os_buf: [64]u8 = undefined;
        const os_build = transformer_mod.macosProductVersion(&os_buf) orelse "";
        // The measured round-cost table rides the same identity: restored
        // here, written at the end of any request that folded new samples.
        // The bucket grid and store version are the arch's (only qwen4_exp gets the long
        // grid); every other arch keeps the `rc1` table 26.9.1 wrote and boots warm.
        const rc_layout: round_cost_mod.Layout = round_cost_mod.layoutFor(params.config);
        xfm_ptr.round_cost.layout = rc_layout;
        const rc_key = round_cost_mod.cacheKey(&xfm_ptr.round_cost_key_buf, chip_mod.chipBrand(), params.model_dir, quant, os_build, rc_layout, round_cost_mod.engineBuildId());
        xfm_ptr.round_cost_key_len = @intCast(rc_key.len);
        if (round_cost_mod.loadCached(sch.allocator, sch.io, rc_key, rc_layout)) |t| {
            xfm_ptr.round_cost = t;
            log.info("[spec-cost] round-cost table restored ({d} width cells, {d} serial cells)\n", .{ t.restored, t.restored_serial });
            if (t.restored_dropped > 0) log.info("[spec-cost] dropped {d} implausible persisted cell(s)\n", .{t.restored_dropped});
        }
    }

    // Assistant sidecar (optional). Loaded only when `drafter_dir` is
    // non-empty. The sidecar KIND is decided by its config CONTRACT: a
    // config declaring block_size + mask_token_id + target_layer_ids is a
    // DFlash block-drafter (any `*_assistant` family); anything else is refused.
    var dflash_ptr: ?*DflashModel = null;
    if (drafter_dir.len > 0 and dflash_mod.probeIsDflash(sch.io, sch.allocator, drafter_dir)) {
        const d = try sch.allocator.create(DflashModel);
        d.* = (if (params.config.isGlm5()) @import("glm5/glm5_dflash.zig").loadAssistantStored(sch.io, sch.allocator, drafter_dir, xfm_ptr.glm5.?) else dflash_mod.loadDflash(sch.io, sch.allocator, mlx.gpuStream(), drafter_dir)) catch |err| {
            sch.allocator.destroy(d);
            log.err("Failed to load DFlash assistant at {s}: {s}\n", .{ drafter_dir, @errorName(err) });
            return err;
        };
        if (params.config.isGlm5()) {
            d.sliding_serving = true;
            log.info("[glm-dflash] native DFlash2 loaded; greedy and sampled target decisions use layerwise tree verification\n", .{});
            if (transformer_mod.diagEnvOn("SUSHI_GLM_ROUND_UBENCH")) glmRoundUbench(sch.allocator, xfm_ptr.glm5.?, d, kv_quant_config.glmLatentBits() orelse 0, params.tok);
        } else if (params.config.isQwen4()) {
            d.sliding_serving = true;
            d.bind(xfm_ptr) catch |err| {
                d.deinit();
                sch.allocator.destroy(d);
                log.err("DFlash assistant at {s} is incompatible with target: {s}\n", .{ drafter_dir, @errorName(err) });
                return err;
            };
        } else {
            d.deinit();
            sch.allocator.destroy(d);
            log.err("--drafter: a DFlash2 assistant serves Qwen3.8-Flash-Next and GLM-5.3-Flash only\n", .{});
            return error.DflashTargetMismatch;
        }
        dflash_ptr = d;
        const wide_lane = dflash_mod.wideVerifyLaneAvailable();
        const block_cap = dflash_mod.blockCapForMachine(chip_mod.chipBrand());
        sch.drafter_block_size = dflash_mod.resolveBlockSize(d.config.block_size, wide_lane, block_cap.cap);
        if (params.config.isQwen4()) {
            const bounds = dflash_mod.qwenRowBounds(dflash_mod.drafter_min_depth, dflash_mod.drafter_max_depth, d.config.block_size) catch |err| {
                d.deinit();
                sch.allocator.destroy(d);
                log.err("--drafter-min-depth/--drafter-max-depth: this drafter serves {d}..{d} rows per round on Qwen3.8-Flash-Next\n", .{ dflash_mod.QWEN_MIN_ROWS, @min(d.config.block_size, dflash_mod.QWEN_MAX_ROWS) });
                return err;
            };
            sch.drafter_block_size = bounds.max;
        } else if (params.config.isGlm5()) {
            const bounds = dflash_mod.glmRowBounds(dflash_mod.drafter_min_depth, dflash_mod.drafter_max_depth) catch |err| {
                d.deinit();
                sch.allocator.destroy(d);
                log.err("--drafter-min-depth/--drafter-max-depth: GLM-5.3-Flash verifies {d}..{d} rows per round\n", .{ dflash_mod.GLM_MIN_ROWS, dflash_mod.GLM_MAX_ROWS });
                return err;
            };
            sch.drafter_block_size = bounds.max;
        }
        var cap_note_buf: [96]u8 = undefined;
        const cap_note: []const u8 = if (!wide_lane and !params.config.isQwen4() and !dflash_mod.drafterDepthExplicit() and d.config.block_size > sch.drafter_block_size)
            std.fmt.bufPrint(&cap_note_buf, ", capped ({s} cap {d})", .{
                block_cap.label,
                block_cap.cap,
            }) catch ", capped"
        else
            "";
        log.info("DFlash drafter ready (block_size={d}{s}, wide_verify_lane={}, targets={any}).\n", .{
            sch.drafter_block_size,
            cap_note,
            wide_lane,
            d.config.target_layer_ids,
        });
        if (params.config.isQwen4()) log.info("[spec] Qwen3.8-Flash-Next decodes with DFlash2 ({d}..{d} rows per round; --drafter outranks MTP, --no-drafter selects it)\n", .{ dflash_mod.qwenMinRows(), sch.drafter_block_size });
        if (params.config.isGlm5()) log.info("[spec] GLM-5.3-Flash decodes with DFlash2 ({d}..{d} rows per round)\n", .{ dflash_mod.drafter_min_depth orelse dflash_mod.GLM_MIN_ROWS, sch.drafter_block_size });
    } else if (drafter_dir.len > 0) {
        log.err("Drafter at {s} is not a DFlash assistant (its config lacks block_size, mask_token_id and target_layer_ids)\n", .{drafter_dir});
        return error.UnsupportedDrafter;
    }
    errdefer if (dflash_ptr) |d| {
        d.deinit();
        sch.allocator.destroy(d);
    };

    const mtp_cost_profile: mtp_mod.MtpCostProfile = .generic;
    // MiMo's trained heads, resident beside a resident trunk (streaming refuses MTP above).
    var mimo_head: ?*mimo_mtp.Head = null;
    if (mtp_enabled and params.config.isMimo() and !params.config.expert_streaming) {
        mimo_head = loadMimoHeads(sch, params.model_dir, params.config, xfm_ptr) catch |err| blk: {
            log.warn("[mimo-mtp] heads not loaded ({s}) — MTP off\n", .{@errorName(err)});
            break :blk null;
        };
    }
    errdefer if (mimo_head) |h| {
        h.deinit();
        sch.allocator.destroy(h);
    };
    // The heads built the coarse copy for their drafts; without them the trunk's greedy readout still takes one.
    // Same predicate as the resident bill that prices this copy.
    if (mimoCoarseHeadBytes(params.config) > 0 and xfm_ptr.lm_head_coarse == null)
        xfm_ptr.lm_head_coarse = mtp_mod.buildRerankCoarse(mlx.gpuStream(), xfm_ptr, mimo_mtp.rerankBits());
    // The qwen4_exp in-checkpoint head's coarse rerank build: a
    // `requantizeRows` of the whole
    // trunk lm_head plus a synchronous eval of ~240 MB. Lazily it ran inside
    // the FIRST request's draft chain — on this thread, mid-round, with the
    // stream drained — i.e. first-token latency for whoever loaded the model.
    // Gated exactly like `entry.mtp`'s `.qwen4` arm below: `--no-mtp` never
    // drafts, so it never pays. One-shot, so the draft path's ask stays a
    // pure read; if this ever does not run, that ask still builds.
    if (mtp_enabled) _ = xfm_ptr.qwen4BuildDraftRerank();

    if (xfm_ptr.expert_stream) |engine| warm: {
        if (engine.plan.resident) break :warm;
        const slots = engine.warmSlotsPerLayer();
        const layers = expert_stream_mod.moeLayerCount(engine.geometry);
        const ranked = engine.warm_usage != null;
        const bytes = if (ranked) engine.plan.cache_bytes else @as(u64, slots) * layers * engine.plan.expert_bytes;
        log.info("[expert-stream] cache warm: preloading {d:.3} GB ({s})\n", .{ @as(f64, @floatFromInt(bytes)) / 1e9, if (ranked) "every pool, most used experts first" else "80% of cache slots, lowest expert IDs" });
        const start = std.Io.Timestamp.now(sch.io, .awake);
        try engine.warmCache();
        const elapsed_ns: u64 = @intCast(start.untilNow(sch.io, .awake).nanoseconds);
        log.info("[expert-stream] cache warm complete: slots={d}/{d} per layer, layers={d}, bytes={d}, elapsed_ms={d}\n", .{
            slots, engine.plan.slots_per_layer, layers, bytes, elapsed_ns / std.time.ns_per_ms,
        });
    }

    // Eager warmup: faults weight pages + compiles the decode-path kernels
    // on this thread's stream. ~600-900 ms at boot but the first user request
    // skips a cold path.
    if (params.warmup_eager) {
        const warmup_start = std.Io.Timestamp.now(sch.io, .awake);
        xfm_ptr.warmup() catch |err| {
            log.warn("Warmup failed ({s}); continuing without it — first request may be slow.\n", .{@errorName(err)});
        };
        const warmup_ns: u64 = @intCast(warmup_start.untilNow(sch.io, .awake).nanoseconds);
        log.info("Warmup complete ({d} ms).\n", .{warmup_ns / std.time.ns_per_ms});
        if (mtp_enabled and xfm_ptr.qwen4_mtp != null) {
            const cap = generate_mod.Generator.resolveMtpDepthCapForProfile(params.mtp_depth, mtp_cost_profile);
            xfm_ptr.warmupSpecVerify(cap, params.kv_quant_config) catch |err| {
                log.warn("[spec-warmup] failed ({s}); the first round at each width pays its kernel compile inside the round.\n", .{@errorName(err)});
            };
        }
        if (mimoVerifyWarmWanted(params.config)) mimoSpecWarmup(sch.io, xfm_ptr, mimo_head, params.kv_quant_config);
    }

    // ── Phase 05: install everything onto the LoadedModel entry, mark
    //    ready, and update the scheduler's borrowed views. The registry
    //    mutex guards the state transition + `current_resident_bytes`
    //    accounting; `state_cond.broadcast` (inside markReadyLocked) wakes
    //    any waiter blocked in ensureLoaded.
    //
    //    Everything below this comment must be infallible — once we begin
    //    assigning to `params.entry`, the per-ptr errdefers above would
    //    double-free if we error-return. `drafter_path_owned` was alloc'd
    //    up front for exactly this reason; the per-model prefix cache
    //    init is a struct literal (no fallible alloc).
    const entry = params.entry;
    entry.weights = weights_ptr;
    entry.transformer = xfm_ptr;
    entry.vision_encoder = vision_ptr;
    entry.dflash = dflash_ptr;
    entry.drafter_block_size = sch.drafter_block_size;
    entry.mtp = if (mtp_enabled and xfm_ptr.qwen4_mtp != null)
        generate_mod.MtpHeadRef{ .qwen4 = xfm_ptr }
    else if (xfm_ptr.dsv41 != null and xfm_ptr.dsv41.?.dspark != null)
        generate_mod.MtpHeadRef{ .dsv41 = xfm_ptr }
    else if (mimo_head) |h|
        generate_mod.MtpHeadRef{ .mimo = h }
    else
        null;
    // Resolve the auto (0) cap here so every downstream reader of
    // `lm.mtp_depth` (server log lines, slot params) sees the real value.
    entry.mtp_depth = generate_mod.Generator.resolveMtpDepthCapForProfile(params.mtp_depth, mtp_cost_profile);
    xfm_ptr.mtp_depth_free = generate_mod.Generator.mtpDepthCapFree(params.mtp_depth);
    if (mimo_head) |h| {
        // The head count AND the widest verify whose rows stay byte-identical
        // to serial decode: a deeper round would hand the forward more rows
        // than the decode-shaped MiMo verify serves.
        const rows_max: u32 = generate_mod.Generator.mtpVerifyDraftsMax(true);
        entry.mtp_depth = @min(entry.mtp_depth, @min(@as(u32, @intCast(h.heads)), rows_max));
        xfm_ptr.mtp_depth_free = @min(xfm_ptr.mtp_depth_free, @min(@as(u32, @intCast(h.heads)), rows_max));
    }
    if (entry.mtp != null and !params.config.isDsv41()) log.info("[mtp] depth range {d}..{d} ({s})\n", .{
        depth_bounds.floorFor(depth_bounds.active, entry.mtp_depth),
        entry.mtp_depth,
        if (depth_bounds.active.explicit()) "--mtp-min-depth/--mtp-max-depth" else "default",
    });
    // A MERGED drafter has no `--drafter` to echo, so the reported path comes
    // from what was actually resolved — `drafter_loaded` and `drafter_path`
    // must not disagree about the same sidecar.
    if (drafter_path_owned.len == 0 and drafter_dir.len > 0 and
        dflash_ptr != null)
    {
        drafter_path_owned = try sch.allocator.dupe(u8, drafter_dir);
    }
    entry.drafter_path = drafter_path_owned;
    drafter_path_owned = &[_]u8{}; // disarm the errdefer
    // Transfer ownership of the heap-allocated CPU state from `params` to
    // the entry. The caller (main.zig) MUST NOT free these — `LoadedModel.deinit`
    // walks them in the same `*X` pointer form they came in.
    entry.releaseRetainedCpuState();
    entry.config = params.config;
    entry.tokenizer = params.tok;
    entry.chat_config = params.chat_config;
    // Per-model hot prefix cache (Plan 03 → Plan 05 move). Hybrid recurrent
    // archs are accepted iff `ssm_checkpoint_stride > 0` (Phase 1 of the
    // performance plan): with per-stride SSM checkpoints we can rewind both
    // KV and SSM to a snapshotted prefix; without them, divergence forces a
    // full reset, so we keep the legacy single-slot path for hybrid.
    const enable_ssm_cps = params.ssm_checkpoint_stride > 0;
    const ram_prefix_cache = params.prefix_cache_ram_enabled;
    // Phase 3 persists hybrid recurrent state too: the disk tier is allowed whenever the RAM tier
    // accepts the arch — pure-attention always, hybrid iff SSM checkpoints are enabled.
    const has_ssm_layers = params.config.full_attention_interval > 0;
    const disk_ok = !has_ssm_layers or enable_ssm_cps;
    const cache_applies = params.prefix_cache_capacity > 0 and
        prefix_cache_mod.HotPrefixCache.shouldUse(params.config, enable_ssm_cps);
    // The SSD tier comes up BEFORE anything is sized for it: whether this model has one decides its
    // RAM semantics (SSD-first or not), its bills and its checkpoint capture. A tier that was wanted
    // and did not come up (no room, an unreadable fingerprint, a failed init) is recorded on the config,
    // so every bill reads the same answer as `--prefix-cache-disk off`.
    var disk_tier: ?kv_disk_cache.DiskTier = null;
    if (cache_applies and params.prefix_cache_disk_bytes > 0 and disk_ok) attach: {
        const layout = diskLayoutKey(sch.allocator, params.config, entry.path) catch |err| {
            log.warn("[disk-cache] fingerprint failed: {s} — persistence off for this model\n", .{@errorName(err)});
            break :attach;
        };
        defer if (layout) |l| sch.allocator.free(l);
        const fp = disk_placement.modelFingerprintWithLayout(sch.allocator, sch.io, entry.path, layout) catch |err| {
            log.warn("[disk-cache] fingerprint failed: {s} — persistence off for this model\n", .{@errorName(err)});
            break :attach;
        };
        defer sch.allocator.free(fp);
        const base = disk_placement.defaultBaseDir(sch.allocator) catch break :attach;
        defer sch.allocator.free(base);
        const plan: disk_placement.DefaultDiskBudget = if (params.prefix_cache_disk_resolver) |resolve|
            resolve(params.config, base, disk_placement.tierBytes(sch.io, base, fp)) orelse break :attach
        else
            .{ .bytes = params.prefix_cache_disk_bytes, .bound = .flag, .cap = params.prefix_cache_disk_bytes };
        disk_tier = kv_disk_cache.DiskTier.init(
            sch.allocator,
            sch.io,
            base,
            fp,
            plan.bytes,
            kv_disk_cache.DEFAULT_CHUNK_TOKENS,
        ) catch |err| {
            log.warn("[disk-cache] init failed: {s} — persistence off for this model\n", .{@errorName(err)});
            break :attach;
        };
        disk_tier.?.operator_cap = plan.cap;
    }
    params.config.prefix_cache_disk_declined = params.prefix_cache_disk_bytes > 0 and disk_tier == null;
    const disk_prefix_cache = disk_tier != null;
    if (cache_applies and (ram_prefix_cache or disk_prefix_cache)) {
        // The weights are resident here, so the resolver's active-memory read
        // is honest; the raw launch budget never reaches initWithMem (a 40 GB
        // cap beside a ~70 GB pack was the 2026-08-30 uncatchable Metal OOM).
        // Disk-only mode retains no reusable KV in RAM, so it asks for no RAM budget.
        var ssd_idle_mem: u64 = 0;
        const clamped_prefix_mem = hotCacheMemForLoad(ram_prefix_cache, params.prefix_cache_mem_resolver, params.config, params.prefix_cache_mem_bytes, &ssd_idle_mem);
        if (ram_prefix_cache) {
            // On an SSD-first arch `--prefix-cache-mem` is the idle allowance beside the live session.
            const ram_bytes = if (prefix_cache_mod.ssdFirstActive(params.config, disk_prefix_cache, ram_prefix_cache)) ssd_idle_mem else clamped_prefix_mem;
            log.info("Allocating {d:.1} GB RAM for the prefix cache (--prefix-cache-mem)\n", .{@as(f64, @floatFromInt(ram_bytes)) / (1024.0 * 1024.0 * 1024.0)});
        }
        entry.prefix_cache = prefix_cache_mod.HotPrefixCache.initWithMem(
            sch.allocator,
            if (ram_prefix_cache) params.prefix_cache_capacity else 0,
            clamped_prefix_mem,
        );
        entry.prefix_cache.?.qsa_history_required = params.config.indexer_budget != 0 and !params.config.isGlm5();
        // Checkpoint-retention arch gate, mirrored once: `HotPrefixCache`/`DiskTier` never
        // see a ModelConfig. The ungated value names the previous behaviour at each site.
        entry.prefix_cache.?.cp_thin = if (params.config.longCtxGated() or params.config.isGlm5() or params.config.isDsv41()) .min_span_recency else .min_span;
        entry.prefix_cache.?.ssd_idle_mem = ssd_idle_mem;
        entry.prefix_cache.?.hybrid = has_ssm_layers or params.config.isGlm5() or params.config.isDsv41();
        if (disk_tier) |tier| {
            entry.prefix_cache.?.disk = tier;
            entry.prefix_cache.?.disk.?.cp_thin =
                if (params.config.longCtxGated() or !ram_prefix_cache or params.config.isGlm5() or params.config.isDsv41()) .min_span_recency else .oldest;
            entry.prefix_cache.?.disk.?.ssm_max_per_entry = if (params.config.isDsv41())
                @import("dsv41/dsv41_prefix.zig").disk_checkpoint_cap
            else if (params.config.isGlm5() or params.config.longCtxGated() or !ram_prefix_cache)
                kv_disk_cache.SSM_DISK_MAX_PER_ENTRY
            else
                kv_disk_cache.SSM_DISK_MAX_PER_ENTRY_LEGACY;
            if (params.config.dsv41) |c| {
                const dsv41_pfx = @import("dsv41/dsv41_prefix.zig");
                const divs = try dsv41_pfx.rowDivisors(sch.allocator, &c, params.config.dsparkOn());
                defer sch.allocator.free(divs);
                try entry.prefix_cache.?.disk.?.setGrownSpec(dsv41_pfx.row_span, divs, c.stateBytesPerToken());
            }
        }
        // SSD-first: arch + env switch + a live disk tier (attached above); without one qwen4_exp takes the RAM arm.
        entry.prefix_cache.?.ssd_first = prefix_cache_mod.ssdFirstActive(
            params.config,
            entry.prefix_cache.?.disk != null,
            ram_prefix_cache,
        );
        // Every tier writes off the inference thread, so a commit lands whole after the response.
        if (entry.prefix_cache.?.disk) |*d| {
            d.writer_permit_bytes = @import("server.zig").disk_writer_permit_bytes;
            d.enableBackgroundWriter();
        }
        if (entry.prefix_cache.?.ssd_first) {
            entry.prefix_cache.?.disk.?.ssd_first = true;
            // Startup sweep of strays + root-wide LRU across sibling fingerprints.
            entry.prefix_cache.?.disk.?.sweepSiblings();
        }
        entry.ssm_checkpoint_stride = params.ssm_checkpoint_stride;
        const cp_max = if (params.config.isGlm5())
            @import("glm5/glm5_prefix.zig").checkpointMax(params.ssm_checkpoint_max)
        else if (params.config.isDsv41())
            @import("dsv41/dsv41_prefix.zig").checkpointMax(params.ssm_checkpoint_max)
        else
            params.ssm_checkpoint_max;
        entry.ssm_checkpoint_max = cp_max;
        // The cache re-applies the cap after a replace-path merge; without this
        // it defaults to 0 (unlimited) and multi-turn entries grow unbounded.
        entry.prefix_cache.?.ssm_checkpoint_max = cp_max;
    }
    // Iteration 2 (perf-plan Phase 4 #3): tokenize cache for warm-path
    // chat-template renders (`chat_mod.formatChat` at the handler boundary).
    // Default capacity is small (4 entries) because chat conversations
    // mutate the messages list every turn; the goal is to catch warm
    // reuse benches and repeated agent-loop probes, not to memoize a
    // full session.
    if (params.tokenize_cache_entries > 0) {
        entry.tokenize_cache = tokenize_cache_mod.TokenizeCache.init(
            sch.allocator,
            params.tokenize_cache_entries,
        );
    }

    // The weights the load preflight billed. Drives the registry's resident-memory gate and
    // `/v1/models` `bytes_resident`.
    const bytes_resident: u64 = if (params.config.expert_streaming)
        expertStreamingGateBytes(
            streaming_resident_bytes.?,
            params.config.expert_cache_bytes,
            loadStagingBytes(params.config, params.config.expert_fill_peak_bytes, params.config.expert_workspace_bytes),
            params.config.expert_bounce_bytes,
        )
    else
        residentWeightBytes(streaming_resident_bytes, entry.bytes_on_disk, modelDiskBytes(sch.io, params.model_dir), params.config.num_hidden_layers, params.config.hidden_size);

    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.markReadyLocked(entry, bytes_resident);
    sch.registry.mutex.unlock(sch.io);

    // Set borrowed views for scheduler-internal code (and `current_model`
    // for ensureLoaded → request handlers in Phase C).
    sch.current_model = entry;
    sch.weights = weights_ptr;
    sch.xfm = xfm_ptr;
    sch.vision_encoder = vision_ptr;
    sch.dflash = dflash_ptr;
    if (entry.prefix_cache) |*hc| sch.hot_prefix_cache = hc;
    publishHotCacheResidency(sch);
}

/// Plan 05 Phase D: service a cold-load work item on the inference thread.
/// Conn thread has already:
///   * Marked `req.entry.state = .loading` (transitioned from .unloaded).
///   * (Optionally) marked `req.evict_entry.state = .evicting` and drained
///     its refcount to 0.
///   * Parsed CPU-only state (config/tok/chat_config) and handed pointers
///     in via the request.
/// We unload the victim (if any), run the load body, install everything on
/// `req.entry`, mark ready, and broadcast `req.done_cond` so the conn
/// thread wakes. On failure, mark `.error_state` so future ensureLoaded
/// calls fail fast; the conn thread surfaces a 500.
fn logWiredPolicy(limit: ?usize) void {
    if (limit) |bytes| {
        log.info("[wired] limit={d} MB\n", .{bytes / (1024 * 1024)});
    } else {
        log.debug("[wired] declined (no gpu)\n", .{});
    }
}

pub fn runLoadRequest(sch: *Scheduler, req: *LoadRequest) void {
    // Step 1: evict victims (if any) BEFORE the load, so peak GPU residency
    // never holds the old + new model at once. unloadResident() drops
    // mlx_arrays — same thread-stream invariant as cleanup_queue drain.
    const wired_before = if (req.evict_entries.len > 0) status.getWiredMemBytes() else 0;
    var evicted_bytes: u64 = 0;
    for (req.evict_entries) |victim| {
        const victim_bytes = victim.bytes_resident; // unloadResident zeroes it
        log.info("[registry] evicting model id={s} ({d:.2} GB resident)\n", .{
            victim.id,
            @as(f64, @floatFromInt(victim_bytes)) / 1_073_741_824.0,
        });
        victim.unloadResident();
        if (victim.config) |c| @import("dsv41/dsv41_borrow.zig").unloadModelMlx(c);
        evicted_bytes +|= victim_bytes;
        sch.registry.mutex.lockUncancelable(sch.io);
        sch.registry.accountEvictedLocked(victim_bytes);
        sch.registry.finalizeEvictionLocked(victim);
        sch.registry.mutex.unlock(sch.io);
    }
    // The preflight below reads free memory: the victims' must be back first.
    if (evicted_bytes > 0) {
        _ = mlx.mlx_clear_cache();
        waitForUnwire(sch.io, wired_before, evicted_bytes);
    }

    // Step 2: the actual load. On error, mark .error_state and signal done
    // (conn thread frees pre-parsed CPU state — ownership stays on req on
    // the failure path).
    const had_error = mlx.errorPending();
    doLoadOnInferenceThread(sch, req) catch |err| {
        failLoadRequest(sch, req, err, had_error);
        return;
    };

    log.info("[registry] model id={s} ready ({d:.2} GB resident)\n", .{
        req.entry.id,
        @as(f64, @floatFromInt(req.entry.bytes_resident)) / 1_073_741_824.0,
    });
    // Re-apply so `fit` capacity covers everything this load brought in
    // (MTP head / drafter / vision land after the mid-load apply).
    logWiredPolicy(mlx.applyWiredPolicy());
    finishLoadRequest(sch, req, null);
}

fn failLoadRequest(sch: *Scheduler, req: *LoadRequest, err: anyerror, had_error: bool) void {
    log.err("[registry] load failed for model id={s}: {s}\n", .{ req.entry.id, @errorName(err) });
    // The load's own MLX error is reported here; latched, it would fail the next request.
    mlx.dropLatchedErrorUnless(had_error);
    sch.registry.mutex.lockUncancelable(sch.io);
    if (err == error.InsufficientMemory) {
        // A memory-preflight refusal is transient, not a property of the
        // checkpoint: the 503 tells the user to free memory and retry, so
        // the entry must go back to .unloaded — stuck in .error_state the
        // retry would fail fast until a server restart (#144).
        sch.registry.markUnloadedLocked(req.entry);
    } else {
        // markErrorLocked dupes the error name onto the entry; the
        // conn thread reads it back from the entry, not from req.
        sch.registry.markErrorLocked(req.entry, @errorName(err));
    }
    sch.registry.mutex.unlock(sch.io);
    finishLoadRequest(sch, req, @errorName(err));
}

fn finishLoadRequest(sch: *Scheduler, req: *LoadRequest, err_name: ?[]const u8) void {
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    if (err_name) |name| {
        if (req.error_name) |old| req.allocator.free(old);
        req.error_name = req.allocator.dupe(u8, name) catch null;
    }
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

/// Free a model's resident mlx state on the inference thread (stream-bound,
/// same invariant as the cleanup-queue drain) and finalize the eviction
/// accounting. The conn thread already marked the entry `.evicting` and
/// drained its refcount in `unloadModel`.
pub fn runUnloadRequest(sch: *Scheduler, req: *UnloadRequest) void {
    const entry = req.entry;
    const bytes = entry.bytes_resident; // unloadResident zeroes it
    log.info("[registry] unloading model id={s} ({d:.2} GB resident)\n", .{
        entry.id,
        @as(f64, @floatFromInt(bytes)) / 1_073_741_824.0,
    });
    const wired_before = status.getWiredMemBytes();
    entry.unloadResident();
    if (entry.config) |c| @import("dsv41/dsv41_borrow.zig").unloadModelMlx(c);
    // unloadResident freed the arrays into MLX's allocator cache — clear it
    // so the unload actually returns the memory to the OS (the whole point
    // of the load→generate→unload flow).
    _ = mlx.mlx_clear_cache();
    waitForUnwire(sch.io, wired_before, bytes);
    // Drop any borrowed views that pointed at this entry so post-unload reads
    // don't dangle (gen entries leave xfm null already, but an LLM unload
    // must clear them).
    if (sch.current_model == entry) {
        sch.current_model = null;
        sch.xfm = null;
        sch.weights = null;
        sch.vision_encoder = null;
        sch.dflash = null;
        sch.hot_prefix_cache = null;
        publishHotCacheResidency(sch);
        if (scheduler_mod.hot_cache_budget_invalidate) |f| f();
    }
    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.accountEvictedLocked(bytes);
    sch.registry.finalizeEvictionLocked(entry);
    sch.registry.mutex.unlock(sch.io);
    publishDiskStats(sch);

    // Shrink `fit` capacity back to the surviving live set — leaving the
    // freed model's headroom in place is exactly the per-transient-commit
    // configuration the policy exists to avoid.
    logWiredPolicy(mlx.applyWiredPolicy());

    req.done_mu.lockUncancelable(sch.io);
    req.done = true;
    req.done_cond.broadcast(sch.io);
    req.done_mu.unlock(sch.io);
}

/// The kernel unwires a freed Metal buffer asynchronously (measured ~0.5 s for 50 GB), and a
/// load right behind the unload read that memory as taken (preflight 45 GB free, 95 GB half a
/// second later). The unload answers once most of what it freed is back, or after a bound.
fn waitForUnwire(io: std.Io, wired_before: u64, freed: u64) void {
    if (wired_before == 0 or freed == 0) return;
    var waited_ms: u32 = 0;
    while (!unwireSettled(wired_before, status.getWiredMemBytes(), freed) and waited_ms < UNWIRE_WAIT_MAX_MS) : (waited_ms += 50) {
        std.Io.sleep(io, .fromMilliseconds(50), .real) catch return;
    }
    if (waited_ms > 0) log.info("[registry] waited {d} ms for the unloaded model's memory to unwire\n", .{waited_ms});
}

const UNWIRE_WAIT_MAX_MS: u32 = 3000;

/// Nine tenths of the freed bytes back from the wired set; other processes wire and unwire too.
pub fn unwireSettled(wired_before: u64, wired_now: u64, freed: u64) bool {
    return wired_before -| wired_now >= freed / 10 * 9;
}

test "unwireSettled: the unload waits until most of the freed bytes left the wired set" {
    const gib: u64 = 1 << 30;
    try testing.expect(!unwireSettled(53 * gib, 37 * gib, 50 * gib)); // t+0.1 s in the measured unload
    try testing.expect(unwireSettled(53 * gib, 7 * gib, 50 * gib)); // t+0.7 s
    try testing.expect(unwireSettled(53 * gib, 60 * gib, 0)); // nothing freed, nothing to wait for
    try testing.expect(!unwireSettled(10 * gib, 12 * gib, 5 * gib)); // wired grew: never underflows
}

/// DIAGNOSTIC (SUSHI_PREFILL_UBENCH=N): N cold one-chunk prefills per arm at load, each from an
/// empty cache, logits never projected (the chunk loop never reads them). `_ROWS` (default 2025)
/// is capped at the chunk admission would pick for such a prompt; `_TEXT=<abs path>` tokenized
/// for real routing; `_ARMS=0,1,1,0` runs a pass per arm, 0 on the reference EXL3 NAX GEMM body.
fn prefillUbench(alloc: std.mem.Allocator, xfm: *Transformer, cfg: *const ModelConfig, tok: *Tokenizer) void {
    const tio = std.Io.Threaded.global_single_threaded.io();
    const n = @max(1, std.fmt.parseInt(usize, std.mem.sliceTo(std.c.getenv("SUSHI_PREFILL_UBENCH").?, 0), 10) catch 3);
    const asked: usize = if (std.c.getenv("SUSHI_PREFILL_UBENCH_ROWS")) |r| std.fmt.parseInt(usize, std.mem.sliceTo(r, 0), 10) catch 2025 else 2025;
    const pick = scheduler_mod.prefill_ubench_chunk orelse {
        log.warn("[prefill-ubench] skipped: no admission hook to size the chunk\n", .{});
        return;
    };
    // A wider forward than admission would run allocates outside the load's memory bill.
    const rows = @max(1, @min(asked, pick(cfg, asked + 1, 1, xfm.cache.config, false, 0, 0, false, false)));
    const ids = alloc.alloc(i32, rows) catch return;
    defer alloc.free(ids);
    for (ids, 0..) |*v, i| v.* = @intCast(1 + (i % 997));
    if (std.c.getenv("SUSHI_PREFILL_UBENCH_TEXT")) |path| blk: {
        const text = std.Io.Dir.cwd().readFileAlloc(tio, std.mem.sliceTo(path, 0), alloc, .limited(64 << 20)) catch break :blk;
        defer alloc.free(text);
        const enc = tok.encode(alloc, text) catch break :blk;
        defer alloc.free(enc);
        if (enc.len == 0) break :blk;
        for (ids, 0..) |*v, i| v.* = @intCast(enc[i % enc.len]);
    }
    var arms: [16]?bool = @splat(null);
    const n_arms = if (std.c.getenv("SUSHI_PREFILL_UBENCH_ARMS")) |r| parseUbenchArms(std.mem.sliceTo(r, 0), &arms) else 1;
    defer exl3_kernels.nax_reference_override = false;
    const had_error = mlx.errorPending();
    defer xfm.resetCache() catch {};
    const shape = [_]c_int{ 1, @intCast(rows) };
    const times = alloc.alloc(f64, n) catch return;
    defer alloc.free(times);
    for (arms[0..n_arms]) |arm| {
        exl3_kernels.nax_reference_override = if (arm) |a| !a else false;
        for (0..n + 1) |iter| {
            xfm.resetCache() catch return;
            var ctx = xfm.defaultCtx();
            ctx.skip_lm_head = true;
            const ti = mlx.mlx_array_new_data(ids.ptr, &shape, 2, .int32);
            defer _ = mlx.mlx_array_free(ti);
            var sw = io_util.Stopwatch.init(tio);
            const out = xfm.forwardWith(&ctx, ti) catch |err| {
                mlx.dropLatchedErrorUnless(had_error);
                log.warn("[prefill-ubench] forward failed: {s}\n", .{@errorName(err)});
                return;
            };
            defer _ = mlx.mlx_array_free(out);
            mlx.check(mlx.mlx_array_eval(out)) catch |err| {
                mlx.dropLatchedErrorUnless(had_error);
                log.warn("[prefill-ubench] eval failed: {s}\n", .{@errorName(err)});
                return;
            };
            if (iter > 0) times[iter - 1] = @as(f64, @floatFromInt(sw.read())) / 1.0e6;
        }
        std.mem.sortUnstable(f64, times, {}, std.sort.asc(f64));
        log.info("[prefill-ubench] arm={s} rows={d} n={d} median {d:.2} ms min {d:.2} max {d:.2} ({d:.1} tok/s)\n", .{
            if (arm) |a| (if (a) "1" else "0") else "shipped", rows, n, times[n / 2], times[0], times[n - 1],
            @as(f64, @floatFromInt(rows)) / times[n / 2] * 1000.0,
        });
    }
}

/// `_ARMS` list: each comma-separated entry is an arm, `0` the reference, anything else the shipped one.
fn parseUbenchArms(raw: []const u8, out: *[16]?bool) usize {
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, raw, ',');
    while (it.next()) |a| {
        if (n == out.len) break;
        out[n] = a[0] != '0';
        n += 1;
    }
    return @max(n, 1);
}

test "prefill ubench arms parse one entry per comma, 0 the reference and an empty list the shipped arm" {
    var arms: [16]?bool = @splat(null);
    try std.testing.expectEqual(@as(usize, 4), parseUbenchArms("0,1,1,0", &arms));
    try std.testing.expectEqualSlices(?bool, &.{ false, true, true, false }, arms[0..4]);
    arms = @splat(null);
    try std.testing.expectEqual(@as(usize, 1), parseUbenchArms("", &arms));
    try std.testing.expectEqual(@as(?bool, null), arms[0]);
    try std.testing.expectEqual(@as(usize, 16), parseUbenchArms("1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0", &arms));
}

test "preloadCpuState refuses an unsupported checkpoint format by name" {
    // The verdict is a NAMED load error the client sees as a 503, never a
    // stub config.
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "g");
    try tmp.dir.writeFile(io, .{ .sub_path = "g/model-Q4_K_M.gguf", .data = "GGUF" });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/g", .{buf[0..root_len]});
    defer testing.allocator.free(path);
    try testing.expectError(error.ModelFormatUnsupported, preloadCpuState(testing.allocator, io, path));
}

test "every real streamed pack and source on this box plans a streamed load" {
    const t = std.testing;
    const Case = struct { name: []const u8, layout: expert_stream_mod.quant.Layout, tower: u64 };
    for ([_]Case{
        .{ .name = "Qwen3.8-Flash-Next-Sushi-2.6bpw", .layout = .exl3_k4, .tower = 597_327_776 },
        .{ .name = "Qwen3.8-Flash-Next-Sushi-4bpw", .layout = .exl3_k4, .tower = 597_327_776 },
        .{ .name = "Qwen/Qwen3.8-Flash-Next", .layout = .bf16_fused, .tower = 597_327_776 },
        .{ .name = "MiMo-V2.6-Flash-Sushi-2.3bpw", .layout = .exl3_k4, .tower = 1_457_188_864 },
        .{ .name = "GLM-5.3-Flash-Sushi-2.3bpw", .layout = .exl3_k4, .tower = 493_389_824 },
        .{ .name = "GLM-5.3-Flash-Sushi-2.5bpw", .layout = .exl3_k4, .tower = 493_389_824 },
        .{ .name = "GLM-5.3-Flash-BF16", .layout = .bf16_individual, .tower = 1_127_254_016 },
        .{ .name = "GLM-5.3-Flash-FP8", .layout = .fp8_individual, .tower = 1_127_254_016 },
    }) |case| {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try @import("test_models.zig").packPath(&path_buf, case.name);
        var config = model_mod.parseConfig(t.io, t.allocator, path) catch continue;
        defer config.deinit(t.allocator);
        const plan = try planExpertStreaming(t.io, t.allocator, &config, path, 60 << 30, false);
        try t.expectEqual(case.layout, plan.layout);
        try t.expect(plan.cache.slots_per_layer > 1);
        try t.expect(plan.split.trunk > 0 and plan.split.trunk < 40 << 30);
        try t.expectEqual(@as(u64, 0), plan.split.vision);
        try t.expectEqual(case.tower, plan.vision.tower_bytes);
        const seen = try planExpertStreaming(t.io, t.allocator, &config, path, 60 << 30, true);
        try t.expectEqual(case.tower, seen.split.vision);
        try t.expectEqual(plan.split.trunk, seen.split.trunk);
        try t.expectEqual(seen.vision.slots_with.?, seen.cache.slots_per_layer);
        // Slots are whole per layer: a tower smaller than one slot across the layers, or a pack that fits in full, costs none.
        try t.expect(seen.cache.slots_per_layer <= plan.cache.slots_per_layer);
    }
}

test "planExpertStreamingWith takes a resident drafter out of the expert cache" {
    const t = testing;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try @import("test_models.zig").packPath(&path_buf, "Qwen3.8-Flash-Next-Sushi-2.6bpw");
    var config = model_mod.parseConfig(t.io, t.allocator, path) catch return error.SkipZigTest;
    defer config.deinit(t.allocator);
    const drafter: u64 = 2 << 30;
    const without = try planExpertStreamingWith(t.io, t.allocator, &config, path, 32 << 30, false, 0);
    const with = try planExpertStreamingWith(t.io, t.allocator, &config, path, 32 << 30, false, drafter);
    try t.expectEqual(without.split.trunk + drafter, with.split.trunk);
    try t.expect(with.cache.slots_per_layer < without.cache.slots_per_layer);
    try t.expect(without.cache.cache_bytes - with.cache.cache_bytes >= drafter - drafter / 8);
}

test "a failed cold load drops the MLX error it latched and keeps one latched before it" {
    try mlx.requireGpu();
    mlx.installErrorHandler();
    var reg = try ModelRegistry.init(testing.allocator, testing.io, null, 3, 0, null);
    defer reg.deinit();
    const entry = try reg.registerStub("m", "/m", 1);
    var sch: Scheduler = undefined;
    sch.registry = reg;
    sch.io = testing.io;
    var req = LoadRequest{ .entry = entry, .config = undefined, .tok = undefined, .chat_config = undefined, .model_dir = "", .allocator = testing.allocator };
    defer if (req.error_name) |name| testing.allocator.free(name);

    try testing.expect(!mlx.errorPending());
    mlx.latchErrorForTest("[load] the load's own failure");
    failLoadRequest(&sch, &req, error.LoadFailed, false);
    try testing.expect(!mlx.errorPending());

    mlx.latchErrorForTest("[decode] another request's failure");
    failLoadRequest(&sch, &req, error.LoadFailed, true);
    try testing.expect(mlx.errorPending());
    mlx.dropLatchedErrorUnless(false);
}
