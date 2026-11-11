const std = @import("std");
const settings_resolve = @import("settings_resolve.zig");
const server_state = @import("server_state.zig");
const arch_flags = @import("arch/flags.zig");
const arch_config = @import("arch/config.zig");
const mlx = @import("mlx.zig");
const trunk = @import("trunk.zig");
const tokenizer_mod = @import("tokenizer.zig");
const generate_mod = @import("generate.zig");
const drafter_mod = @import("arch/drafter.zig");
const mtp_mod = @import("mtp.zig");
const depth_bounds = @import("mtp_depth_bounds.zig");
const chip_mod = @import("chip.zig");
const model_mod = @import("model.zig");
const admission = @import("arch/admission.zig");
const vision_mod = @import("vision.zig");
const chat_mod = @import("chat.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const prefix_arch = @import("arch/prefix.zig");
const arch_serve = @import("arch/serve.zig");
const metrics_mod = @import("metrics.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const disk_placement = @import("disk_placement.zig");
const tokenize_cache_mod = @import("tokenize_cache.zig");
const model_registry_mod = @import("model_registry.zig");
const model_settings = @import("model_settings.zig");
const mtp_acceptance_mod = @import("mtp_acceptance.zig");
const model_discovery = @import("model_discovery.zig");
const log = @import("log.zig");
const load_clock = @import("load_clock.zig");
const gpu_mem = @import("gpu_mem.zig");
const io_util = @import("io_util.zig");
const status = @import("status.zig");
const Trunk = trunk.Trunk;
const arch_state = @import("arch/arch_state.zig");
const arch_handle = @import("arch/handle.zig");
const arch_load = @import("arch/load.zig");
const arch_packs = @import("arch/packs.zig");
const ModelConfig = model_mod.ModelConfig;
const Tokenizer = tokenizer_mod.Tokenizer;
const dflash_mod = @import("dflash.zig");
const round_cost_mod = @import("round_cost.zig");
const expert_stream_mod = @import("expert_stream.zig");
const expert_usage = @import("expert_usage.zig");
const DflashModel = dflash_mod.DflashModel;
const VisionEncoder = vision_mod.VisionEncoder;
const Weights = model_mod.Weights;
const ChatConfig = chat_mod.ChatConfig;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;
const sampler = @import("sampler.zig");
const testing = std.testing;
const bills = @import("bills.zig");

// The load-time bills live in bills.zig; these are the names the loader and its callers use.
pub const billedFillPeakBytes = bills.billedFillPeakBytes;
const expertStreamingGateBytes = bills.expertStreamingGateBytes;
const fp8ExpertScratchBytes = bills.fp8ExpertScratchBytes;
pub const gateEstimateBytes = bills.gateEstimateBytes;
pub const loadRequirementBytes = bills.loadRequirementBytes;
pub const loadStagingBytes = bills.loadStagingBytes;
pub const coarseHeadBytes = bills.coarseHeadBytes;
const residentWeightBytes = bills.residentWeightBytes;
pub const streamedLoadRequirementBytes = bills.streamedLoadRequirementBytes;
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
/// Trunk, vision encoder, drafter, JIT compile, warmup) all run on the
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
    /// a successful load.
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
    /// `--mtp` / `--no-mtp`; null = not given. A flag outranks the per-model `mtp`.
    mtp: ?bool = null,
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
    /// `--kv-quant`; null = not given. A flag outranks the per-model `kv_quant`, which outranks the kv8 engine default.
    kv_quant: ?trunk.KVQuantConfig = null,
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
    prefix_cache_disk_resolver: ?DiskResolver = null,
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
/// CPU time of every thread of this process, user and system.
fn processCpuNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.PROCESS_CPUTIME_ID, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub fn signalStarted(sch: *Scheduler) void {
    sch.started_mu.lockUncancelable(sch.io);
    defer sch.started_mu.unlock(sch.io);
    sch.started.store(true, .release);
    sch.started_cond.broadcast(sch.io);
}
/// The startup load passed every refusal that runs before its weights: `Scheduler.init` returns and the server
/// listens while the rest of the load runs.
pub fn signalPrechecked(sch: *Scheduler) void {
    sch.started_mu.lockUncancelable(sch.io);
    defer sch.started_mu.unlock(sch.io);
    sch.prechecked = true;
    sch.started_cond.broadcast(sch.io);
}

/// `SUSHI_TEST_LOAD_DELAY_MS=<ms>` holds the startup load after its refusals, so a test can observe the server
/// while the model loads (tests/test_early_listener.sh).
fn startupLoadDelayMs() u64 {
    if (!io_util.diagEnvOn("SUSHI_TEST_LOAD_DELAY_MS")) return 0;
    return std.fmt.parseInt(u64, std.mem.span(std.c.getenv("SUSHI_TEST_LOAD_DELAY_MS").?), 10) catch 0;
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
    if (o.kv_quant) |k| if (arch_flags.kvQuantRequestRefusal(config, k) != null) {
        config.kv_quant_override = null;
        log.warn("[model-settings] kv_quant {s} ignored: the MLA latent is stored at off, 6 or 8 bits\n", .{k.wireName()});
    };
    config.mtp_override = o.mtp;
    config.mtp_acceptance_override = o.mtp_acceptance;
    config.mtp_greedy_tail_override = o.mtp_greedy_tail;
    config.ssd_budget_gb_override = o.ssd_budget_gb orelse 0;
    config.ssd_prefetch_override = o.ssd_prefetch;
    config.cooler_override = o.ds41_cooler;
    config.stampCooler();
    config.preserve_thinking_override = o.preserve_thinking;
    config.think_penalty_override = o.think_penalty;
    config.logit_bias_file_override = o.logit_bias_file;
    config.vision_override = o.vision;
    config.max_tokens_override = o.max_tokens;
    if (yarnFactorIgnoredLine(config, o.yarn_factor)) |line| log.warn("{s}\n", .{line});
    if (resolveSsdBudget(0, config.ssd_budget_gb_override, config.supportsExpertStreaming()).setting_ignored)
        log.warn("[model-settings] ssd_budget_gb ignored: this checkpoint does not stream experts from SSD\n", .{});
}

/// A saved `yarn_factor` the parse did not apply is ignored with this one line, never a refusal: the file outlives the arch it was saved for.
pub fn yarnFactorIgnoredLine(config: *const ModelConfig, saved: ?u8) ?[]const u8 {
    if (saved == null or config.yarn_factor_override != null) return null;
    return model_settings.unavailable("yarn_factor", arch_flags.settingsFit(config)) orelse "model-settings.json yarn_factor is ignored: the config.json was parsed without it";
}

/// The line that refuses a load whose `--ctx-size`, or else model-settings.json `ctx_size`, is above the window the config
/// has under the settings and `--config-overrides`; null when it fits.
pub fn contextWindowRefusal(buf: []u8, config: *const ModelConfig, ctx_flag: u32) ?[]const u8 {
    const pick = model_settings.contextPick(ctx_flag, config.ctx_override);
    return model_settings.ctxAboveWindowLine(buf, pick.value, config.contextCap(), config.yarn_factor_override, pick.source);
}

pub const LaunchRefusal = struct { err: error{ GlmKvQuantUnsupported, LaunchFlagUnsupported }, line: []const u8 };

/// The launch flag, or the saved kv_quant, this model cannot act on; the cold-load pre-check and the load itself ask the same question.
pub fn launchRefusal(buf: []u8, config: *const ModelConfig, kv_quant: ?trunk.KVQuantConfig) ?LaunchRefusal {
    const launch_kv = trunk.KvCacheChoice.resolve(kv_quant, config.kv_quant_override);
    if (arch_flags.kvQuantRequestRefusal(config, launch_kv.config)) |why| {
        const line = std.fmt.bufPrint(buf, "[kv-cache] {s}", .{why}) catch why;
        return .{ .err = error.GlmKvQuantUnsupported, .line = line };
    }
    if (arch_flags.archFlagRefusal(config, .{
        .kv_quant = kv_quant != null,
        .qwen_gdn_16 = !model_mod.qwen_gdn_f32,
        .mtp_head_kv_quant = trunk.Trunk.mtp_head_kv_quant_flag,
        .drop_expert = config.drop_expert != 0,
        .ssd_prefetch = config.ssd_prefetch != null,
        .cooler_off = config.cooler_flag != null,
        .mtp_typical = if (settings_resolve.launch.mtp_acceptance) |m| m == .typical else false,
        .mtp_tokenv3 = if (settings_resolve.launch.mtp_acceptance) |m| m == .tokenv3 else false,
        .mtp_greedy_tail = settings_resolve.launch.mtp_greedy_tail != null,
        .mtp_min_depth = if (depth_bounds.active.min > 1) depth_bounds.active.min else 0,
        .mtp_max_depth = depth_bounds.active.max,
    })) |why| return .{ .err = error.LaunchFlagUnsupported, .line = why };
    return null;
}

/// A load's MTP decision: `--mtp`/`--no-mtp` > the per-model `mtp` > on.
pub fn mtpChoiceFor(flag: ?bool, config: *const ModelConfig) model_settings.MtpChoice {
    return model_settings.MtpChoice.resolve(flag, config.mtp_override, model_settings.default_mtp);
}

/// The streaming gate's verdict on a load's MTP choice (`expert_stream.mtpUnderStreaming`). The gate
/// runs before the load marks its config streamed.
pub fn mtpStreamingVerdict(choice: model_settings.MtpChoice, head_supported: bool) expert_stream_mod.MtpUnderStreaming {
    return expert_stream_mod.mtpUnderStreaming(choice.on, choice.source == .model_settings, choice.source == .default, head_supported);
}

/// The layouts whose experts stream beside a resident drafting head; the pack's own head is read from disk only after this.
fn headLayoutSupported(config: *const ModelConfig, layout: expert_stream_mod.quant.Layout) bool {
    return admission.streamedMtpHead(config, layout);
}

pub fn streamedHeadSupportedAt(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8) bool {
    const geometry = config.expertGeometry();
    const layout = expert_stream_mod.quant.streamingLayoutOfDir(allocator, io, config.model_type, model_dir, geometry.layers, geometry.first_moe_layer) catch return false;
    if (!headLayoutSupported(config, layout)) return false;
    return arch_load.shipsHead(io, allocator, config, model_dir) orelse true;
}

test "a drafting head streams beside the experts only for the layouts its arch reads" {
    const rows = [_]struct { model_type: []const u8, exl3: bool, mxfp4_w123: bool }{
        .{ .model_type = "qwen4_exp", .exl3 = true, .mxfp4_w123 = false },
        .{ .model_type = "deepseek_v41", .exl3 = true, .mxfp4_w123 = true },
        .{ .model_type = "mimo_v2", .exl3 = false, .mxfp4_w123 = false },
        .{ .model_type = "glm5_next", .exl3 = false, .mxfp4_w123 = false },
    };
    for (rows) |r| {
        if (@import("arch/registry.zig").ofModelType(r.model_type) == null) continue;
        const config = ModelConfig{ .model_type = r.model_type };
        for (std.enums.values(expert_stream_mod.quant.Layout)) |layout| {
            const want = (layout == .exl3_k4 and r.exl3) or (layout == .mxfp4_w123 and r.mxfp4_w123);
            try testing.expectEqual(want, headLayoutSupported(&config, layout));
        }
    }
}

/// Does the pack in `model_dir` ship the head its arch drafts with: Flash-Next's MTP head, MiMo's heads, DeepSeek's DSpark stages?
pub fn packShipsMtpHead(io: std.Io, allocator: std.mem.Allocator, model_type: []const u8, model_dir: []const u8) bool {
    const probe = ModelConfig{ .model_type = model_type };
    if (arch_config.fact(&probe, "dflash_by_flag")) return model_discovery.readStubMeta(io, allocator, model_dir).has_mtp;
    if (arch_load.shipsHead(io, allocator, &probe, model_dir)) |ships| return ships;
    return false;
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
    acceptance: settings_resolve.Setting(mtp_acceptance_mod.Mode),
    greedy_tail: settings_resolve.Setting(bool),
    kv: trunk.KvCacheChoice,

    fn of(mtp_flag: ?bool, kv_flag: ?trunk.KVQuantConfig, config: *const ModelConfig) LaunchPicks {
        return .{
            .mtp = mtpChoiceFor(mtp_flag, config),
            .acceptance = settings_resolve.acceptanceFor(config.mtp_acceptance_override),
            .greedy_tail = settings_resolve.greedyTailFor(config.mtp_greedy_tail_override),
            .kv = trunk.KvCacheChoice.resolve(kv_flag, config.kv_quant_override),
        };
    }
};

test "an explicit launch flag outranks model-settings.json for MTP, acceptance, greedy tail and KV" {
    const saved = settings_resolve.launch;
    defer settings_resolve.launch = saved;
    settings_resolve.launch = .{};
    var file = ModelConfig{};
    file.mtp_override = false;
    file.mtp_acceptance_override = .{ .tokenv3 = 0.95 };
    file.mtp_greedy_tail_override = true;
    file.kv_quant_override = .dense;

    const unflagged = LaunchPicks.of(null, null, &file);
    try testing.expect(!unflagged.mtp.on);
    try testing.expectEqual(settings_resolve.Source.model_settings, unflagged.mtp.source);
    try testing.expectEqual(settings_resolve.Source.model_settings, unflagged.acceptance.source);
    try testing.expectEqual(settings_resolve.Setting(bool){ .value = true, .source = .model_settings }, unflagged.greedy_tail);
    try testing.expectEqualStrings("model-settings.json", unflagged.kv.sourceName());

    settings_resolve.launch = .{ .mtp_acceptance = .{ .typical = .{ .delta = 0.1 } }, .mtp_greedy_tail = true };
    const flagged = LaunchPicks.of(true, trunk.KVQuantConfig.affine(4), &file);
    try testing.expect(flagged.mtp.on);
    try testing.expectEqualStrings("--mtp", flagged.mtp.sourceName());
    try testing.expectEqual(@as(f32, 0.1), flagged.acceptance.value.typical.delta);
    try testing.expectEqual(settings_resolve.Source.flag, flagged.acceptance.source);
    try testing.expectEqual(settings_resolve.Source.flag, flagged.greedy_tail.source);
    try testing.expectEqualStrings("kv4", flagged.kv.label());
    try testing.expectEqualStrings("--kv-quant", flagged.kv.sourceName());
}

test "the width a load warms its spec verify at is the one it resolved, a file kv_quant included" {
    var cfg = ModelConfig{ .model_type = "qwen4_exp" };
    try testing.expectEqual(trunk.KVQuantConfig.engine_default, bills.kvCacheForLaunch(&cfg, null).config);
    cfg.kv_quant_override = trunk.KVQuantConfig.affine(4);
    try testing.expectEqual(trunk.KVQuantConfig.affine(4), bills.kvCacheForLaunch(&cfg, null).config);
    try testing.expectEqual(trunk.KVQuantConfig.affine(6), bills.kvCacheForLaunch(&cfg, trunk.KVQuantConfig.affine(6)).config);
}

test "a model-settings kv_quant the MLA latent cannot store is dropped at stamping, never a refused load" {
    var glm = ModelConfig{ .model_type = "glm5_next" };
    applyModelSettings(&glm, .{ .kv_quant = trunk.KVQuantConfig.affine(4) });
    try testing.expect(glm.kv_quant_override == null);
    applyModelSettings(&glm, .{ .kv_quant = trunk.KVQuantConfig.affine(6) });
    try testing.expectEqual(@as(u8, 6), glm.kv_quant_override.?.bits);
    var qwen = ModelConfig{ .model_type = "qwen4_exp" };
    applyModelSettings(&qwen, .{ .kv_quant = trunk.KVQuantConfig.affine(4) });
    try testing.expectEqual(@as(u8, 4), qwen.kv_quant_override.?.bits);
}

test "a saved yarn_factor an arch cannot read is ignored with a line naming why; one the parse applied says nothing" {
    var glm = ModelConfig{ .model_type = "glm5_next" };
    try testing.expect(std.mem.indexOf(u8, yarnFactorIgnoredLine(&glm, 4).?, "YaRN") != null);
    try testing.expect(yarnFactorIgnoredLine(&glm, null) == null);
    var qwen = ModelConfig{ .model_type = "qwen4_exp", .yarn_factor_override = 4 };
    try testing.expect(yarnFactorIgnoredLine(&qwen, 4) == null);
}

test "an explicit --mtp keeps the head on a streamed load, or refuses when it cannot" {
    const asked = mtpChoiceFor(true, &ModelConfig{});
    try testing.expectEqual(expert_stream_mod.MtpUnderStreaming.keep, mtpStreamingVerdict(asked, true));
    try testing.expectEqual(expert_stream_mod.MtpUnderStreaming.refuse, mtpStreamingVerdict(asked, false));
    var streamed = ModelConfig{};
    streamed.expert_streaming = true;
    try testing.expect(mtpChoiceFor(true, &streamed).on);
}

pub const SsdBudgetOrigin = enum { none, flag, model_settings, ram_default };

pub const SsdBudgetChoice = struct {
    bytes: u64,
    origin: SsdBudgetOrigin,
    setting_ignored: bool,
    /// The `iogpu.wired_limit_mb` that earned a 128 GB Mac the larger default; 0 = none did.
    wired_limit_mb: u64 = 0,
};

/// The launch flag wins over the per-model setting; a non-streaming model ignores the
/// setting (warned once). 0 bytes with origin `none` leaves the RAM default to the load.
pub fn resolveSsdBudget(flag_bytes: u64, setting_gb: u32, streaming: bool) SsdBudgetChoice {
    if (!streaming) return .{ .bytes = 0, .origin = .none, .setting_ignored = setting_gb > 0 };
    if (flag_bytes > 0) return .{ .bytes = flag_bytes, .origin = .flag, .setting_ignored = false };
    if (setting_gb > 0) return .{ .bytes = @as(u64, setting_gb) << 30, .origin = .model_settings, .setting_ignored = false };
    return .{ .bytes = 0, .origin = .none, .setting_ignored = false };
}

/// The Macs a DeepSeek load picks its own `--ssd-budget-gb` on, by physical RAM.
pub const RamTier = enum {
    gb96,
    gb128,
    gb192,
    gb256,

    fn gib(self: RamTier) u64 {
        return switch (self) {
            .gb96 => 96,
            .gb128 => 128,
            .gb192 => 192,
            .gb256 => 256,
        };
    }
};

/// The tier `ram_bytes` belongs to, within 3% of its nominal size, or null.
pub fn ramTier(ram_bytes: u64) ?RamTier {
    for ([_]RamTier{ .gb96, .gb128, .gb192, .gb256 }) |tier| {
        const nominal = tier.gib() << 30;
        if (ram_bytes *| 100 >= nominal * 97 and ram_bytes <= nominal * 103 / 100) return tier;
    }
    return null;
}

/// The tier a load with no flag and no setting takes its default budget from: a model that asks for one, on a tiered Mac.
pub fn ramDefaultTierFor(config: *const ModelConfig) ?RamTier {
    if (!config.serve_defaults.ram_budget) return null;
    return ramTier(status.getTotalMemBytes());
}

/// What bounds a default budget: the residency limit and the user-wire cap, each with the bytes kept unplanned under it.
pub const BudgetLimits = struct {
    /// `iogpu.wired_limit_mb` in bytes; 0 = unset, the macOS default.
    wired_limit: u64 = 0,
    ceiling: u64,
    ceiling_name: []const u8,
    margin: u64,
    user_wire: u64,
    user_wire_name: []const u8,
    user_wire_margin: u64,
};

/// A 128 GB Mac takes the larger default only with the wired limit raised to at least this (`iogpu.wired_limit_mb`).
const wired_limit_for_108: u64 = 120_000 << 20;

pub const RamDefault = struct {
    wired_limit_mb: u64 = 0,
    bytes: u64,
    resident: bool,
    /// What the tier asked for, 0 when the pack cannot load resident on a resident tier.
    wanted_bytes: u64,
    /// The limit that stepped the budget down, empty when nothing did.
    bound_by: []const u8,
};

/// PURE: the budget a DeepSeek load takes with no flag and no setting. A 96 GB Mac streams at 80 GiB and a 128 GB Mac at
/// 96, or 108 once `iogpu.wired_limit_mb` is at least 120000;
/// 192 and 256 GB Macs load fully resident (`resident_need`, the planner's bytes for it, 0 = the pack cannot).
/// A budget the limits cannot hold beside `beside` (the 1M request bill and the image encode) steps down to the
/// largest whole-GiB streamed budget they do.
pub fn ramDefaultBudget(tier: RamTier, resident_need: u64, beside: u64, limits: BudgetLimits) RamDefault {
    const GiB: u64 = 1 << 30;
    const wanted: u64 = switch (tier) {
        .gb96 => 80 * GiB,
        .gb128 => if (limits.wired_limit >= wired_limit_for_108) 108 * GiB else 96 * GiB,
        .gb192, .gb256 => resident_need,
    };
    const by_ceiling = limits.ceiling -| limits.margin -| beside;
    const by_wire = if (limits.user_wire > 0) limits.user_wire -| limits.user_wire_margin else std.math.maxInt(u64);
    const room = @min(by_ceiling, by_wire);
    if (wanted > 0 and wanted <= room)
        return .{
            .bytes = wanted,
            .resident = tier == .gb192 or tier == .gb256,
            .wanted_bytes = wanted,
            .bound_by = "",
            .wired_limit_mb = if (tier == .gb128 and wanted == 108 * GiB) limits.wired_limit >> 20 else 0,
        };
    return .{
        .bytes = room / GiB * GiB,
        .resident = false,
        .wanted_bytes = wanted,
        .bound_by = if (wanted == 0) "" else if (by_wire < by_ceiling) limits.user_wire_name else limits.ceiling_name,
    };
}

/// The load line naming the arm and where the budget came from.
fn budgetOriginLine(buf: []u8, choice: SsdBudgetChoice, resident: bool) []const u8 {
    const gib = choice.bytes >> 30;
    return (switch (choice.origin) {
        .flag => std.fmt.bufPrint(buf, "[expert-stream] ssd budget {d} GiB (--ssd-budget-gb)", .{gib}),
        .model_settings => std.fmt.bufPrint(buf, "[expert-stream] ssd budget {d} GiB (model-settings.json)", .{gib}),
        .ram_default => if (resident)
            std.fmt.bufPrint(buf, "[expert-stream] ssd budget: resident (RAM default)", .{})
        else if (choice.wired_limit_mb > 0)
            std.fmt.bufPrint(buf, "[expert-stream] ssd budget: streamed {d} GB (RAM default, iogpu.wired_limit_mb {d})", .{ gib, choice.wired_limit_mb })
        else
            std.fmt.bufPrint(buf, "[expert-stream] ssd budget: streamed {d} GB (RAM default)", .{gib}),
        .none => std.fmt.bufPrint(buf, "[expert-stream] ssd budget {d} GiB", .{gib}),
    }) catch "[expert-stream] ssd budget";
}

/// Why a step-down happened, naming the default, what was chosen and the limit that bound it.
fn stepDownLine(buf: []u8, d: RamDefault) []const u8 {
    return std.fmt.bufPrint(buf, "[expert-stream] default ssd budget {d} GiB does not fit {s}; using {d} GiB streamed", .{
        d.wanted_bytes >> 30,
        d.bound_by,
        d.bytes >> 30,
    }) catch "[expert-stream] default ssd budget stepped down";
}

/// The refusal for a load that needs a budget and has no default: names the flag, the setting and the two tiers' defaults.
pub const budget_required_message = "This checkpoint streams its experts from SSD and needs --ssd-budget-gb <n> or \"ssd_budget_gb\" in model-settings.json; with neither, a 96 GB Mac takes 80, a 128 GB Mac 96 (108 with iogpu.wired_limit_mb at 120000 or more), and a 192 or 256 GB Mac loads the pack fully resident.";

/// The budget a DeepSeek load takes from its Mac's RAM tier. The planner is asked what holding every routed expert
/// resident needs (a budget no pack exhausts), and the limits then decide whether that, or a streamed budget, fits.
/// The registry gate and the load bill the same figure; it calls no MLX, so a connection thread may run it.
pub fn ramDefaultFor(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8, tier: RamTier, load_vision: bool, assistant_bytes: u64) RamDefault {
    const unbounded: u64 = 1 << 50;
    const probe = planExpertStreamingWith(io, allocator, config, model_dir, unbounded, load_vision, assistant_bytes) catch null;
    var resident_need: u64 = 0;
    var encode: u64 = 0;
    if (probe) |p| {
        encode = if (p.vision.on) p.vision.encode_bytes else 0;
        if (p.resolved.ledger) |led| if (led.resident) {
            resident_need = led.trunk_bytes +| led.mtp_bytes +| led.vision_bytes +| led.pool_bytes +| led.cache_bytes;
        };
    }
    return ramDefaultBudget(tier, resident_need, bills.defaultRequestBill(config) +| encode, bills.defaultBudgetLimits());
}

fn ramDefaultChoice(sch: *Scheduler, params: anytype, tier: RamTier, load_vision: bool, assistant_bytes: u64) !SsdBudgetChoice {
    const d = ramDefaultFor(sch.io, sch.allocator, params.config, params.model_dir, tier, load_vision, assistant_bytes);
    var buf: [256]u8 = undefined;
    if (d.bound_by.len > 0) log.warn("{s}\n", .{stepDownLine(&buf, d)});
    return .{ .bytes = d.bytes, .origin = .ram_default, .setting_ignored = false, .wired_limit_mb = d.wired_limit_mb };
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

/// The stored-state layout the SSD namespace is keyed by: the loaded arch's own key, else the config's tag.
fn diskLayoutKey(allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8, xfm: ?*trunk.Trunk) !?[]u8 {
    if (xfm) |x| if (try prefix_arch.diskLayoutKey(allocator, config, model_dir, x)) |key| return key;
    return if (config.cacheLayoutNamespace()) |tag| try allocator.dupe(u8, tag) else null;
}

/// The ONE streamed-load plan, read by the serving load, its registry gate and `kld`: the
/// on-disk layout, what stays resident, and the expert cache the budget leaves.
pub fn planExpertStreaming(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8, budget_bytes: u64, vision: bool) !StreamingPlan {
    return planExpertStreamingWith(io, allocator, config, model_dir, budget_bytes, vision, 0);
}

/// `assistant_bytes` is a resident DFlash2 drafter: it comes out of the budget the expert cache is cut from.
pub fn planExpertStreamingWith(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8, budget_bytes: u64, vision: bool, assistant_bytes: u64) !StreamingPlan {
    var geometry = config.expertGeometry();
    if (config.drop_expert != 0 and !admission.expertStream(config).drop_expert) return error.LaunchFlagUnsupported;
    var drop: ?expert_usage.Drop = if (config.drop_expert == 0) null else expert_usage.load(allocator, model_dir, config.expertDropRequest()) catch |err| {
        log.err("{s}\n", .{expert_usage.describe(err)});
        return error.ExpertDropRefused;
    };
    defer if (drop) |*d| d.deinit();
    if (drop) |*d| geometry.drop = d;
    var streamed = config.*;
    streamed.expert_layout = try expert_stream_mod.quant.streamingLayoutOfDir(allocator, io, config.model_type, model_dir, geometry.layers, geometry.first_moe_layer);
    if (config.stream_mtp_head and !admission.streamedMtpHead(config, streamed.expert_layout)) {
        log.err("[expert-stream] {s}; drop --mtp\n", .{expert_stream_mod.MTP_UNSUPPORTED});
        return error.ExpertStreamingMtpUnsupported;
    }
    var split = try model_mod.streamingResidentSplit(io, allocator, model_dir, &streamed);
    split.trunk +|= coarseHeadBytes(&streamed) +| fp8ExpertScratchBytes(streamed.expert_layout, geometry) +| assistant_bytes +|
        admission.streamedResidentExtra(&streamed);
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

/// Marks the config streamed: the fields every streamed forward and bill reads.
pub fn applyStreamingPlan(config: *ModelConfig, plan: StreamingPlan, budget_bytes: u64) void {
    config.expert_layout = plan.layout;
    config.expert_streaming = true;
    config.expert_cache_bytes = plan.cache.cache_bytes;
    config.expert_ssd_budget_bytes = if (plan.resolved.ledger != null) budget_bytes else 0;
    config.expert_workspace_bytes = plan.cache.workspace_bytes;
    config.expert_bounce_bytes = plan.cache.bounce_bytes;
    config.expert_fill_peak_bytes = billedFillPeakBytes(config, plan.cache.prefill_peak_bytes);
    config.expert_borrow_bytes = arch_serve.plannedBorrowBytes(config, plan.layout == .exl3_k4, plan.cache.cache_bytes);
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
    const split = try model_mod.residentWeightBytes(io, allocator, model_dir, load_vision, mtp_on);
    return split.trunk +| split.mtp +| bills.rerankCoarseBill(split, config, mtp_on, mtp_mod.rerankCoarseBits());
}

/// The drafter a load binds: `--no-drafter` wins, then an explicit dir, then one shipped in the model dir.
/// Flash-Next keeps MTP unless `--drafter` asks for DFlash2, so a drafter beside its pack is never loaded.
pub const LoadDrafterDir = struct {
    dir: []const u8,
    owned: ?[]u8 = null,

    pub fn resolve(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, no_drafter: bool, drafter_dir: []const u8, model_dir: []const u8) LoadDrafterDir {
        if (no_drafter) return .{ .dir = "" };
        if (drafter_dir.len > 0) return .{ .dir = drafter_dir };
        if (arch_config.fact(config, "dflash_by_flag")) return .{ .dir = "" };
        const in_dir = dflash_mod.resolveInDirDrafter(io, allocator, model_dir) orelse return .{ .dir = "" };
        return .{ .dir = in_dir, .owned = in_dir };
    }

    pub fn deinit(self: LoadDrafterDir, allocator: std.mem.Allocator) void {
        if (self.owned) |p| allocator.free(p);
    }
};

/// A streamed Flash-Next load keeps the found DFlash2 drafter resident beside the trunk.
pub fn streamedAssistantBytes(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, no_drafter: bool, drafter_dir: []const u8, model_dir: []const u8) !u64 {
    if (!arch_config.fact(config, "dflash_by_flag")) return 0;
    const found = LoadDrafterDir.resolve(io, allocator, config, no_drafter, drafter_dir, model_dir);
    defer found.deinit(allocator);
    return sushiAssistantLoadBytes(io, allocator, found.dir);
}

/// The serving bill a cold load reserves, read on the config the load will stamp with the drafter's request terms.
fn coldServingBillBytes(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, drafter_dir: []const u8) u64 {
    const bill = scheduler_mod.load_serving_bill orelse return 0;
    var stamped = config.*;
    if (arch_config.fact(config, "dflash_by_flag") and drafter_dir.len > 0) stampDflashRequestTerms(io, allocator, &stamped, drafter_dir) catch {};
    return if (bill(&stamped, std.math.maxInt(u64))) |b| b.needed else 0;
}

/// What a resident cold load reserves in the registry: the requirement its load preflight
/// compares with free memory, so the gate and the preflight read one bill.
pub fn residentColdLoadBillBytes(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8, load_vision: bool, mtp_on: bool, no_drafter: bool, drafter_dir: []const u8) !u64 {
    var weights = (try arch_load.residentLoadBytes(io, allocator, model_dir, config, load_vision, mtp_on)) orelse
        try sushiResidentLoadBytes(io, allocator, model_dir, config, load_vision, mtp_on);
    const drafter = LoadDrafterDir.resolve(io, allocator, config, no_drafter, drafter_dir, model_dir);
    defer drafter.deinit(allocator);
    var context: u64 = 0;
    if (model_mod.usesSushiQuantMemoryBill(config)) {
        weights +|= try sushiAssistantLoadBytes(io, allocator, drafter.dir);
        context = coldServingBillBytes(io, allocator, config, drafter.dir);
    }
    return loadRequirementBytes(config, weights, mtp_on, context);
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
    return model_settings.VisionChoice.resolve(model_settings.vision_flag, config.vision_override, visionDefaultOn(config, streamed));
}

pub fn visionDefaultOn(config: *const ModelConfig, streamed: bool) bool {
    return !streamed or arch_config.fact(config, "vision_on_streamed");
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

test "a load with no flag and no setting takes its RAM tier's budget" {
    const t = testing;
    const GiB: u64 = 1 << 30;
    const GB: u64 = 1_000_000_000;
    // Tiers read from physical RAM within a few percent, never from exact bytes.
    try t.expectEqual(RamTier.gb96, ramTier(96 * GiB).?);
    try t.expectEqual(RamTier.gb128, ramTier(137_438_953_472).?);
    try t.expectEqual(RamTier.gb128, ramTier(127 * GiB).?);
    try t.expectEqual(RamTier.gb192, ramTier(192 * GiB).?);
    try t.expectEqual(RamTier.gb256, ramTier(256 * GiB).?);
    // Any other size has no default: the budget stays required.
    try t.expectEqual(@as(?RamTier, null), ramTier(64 * GiB));
    try t.expectEqual(@as(?RamTier, null), ramTier(48 * GiB));
    try t.expectEqual(@as(?RamTier, null), ramTier(512 * GiB));
    try t.expectEqual(@as(?RamTier, null), ramTier(0));

    // Roomy limits: 96 GB takes 80; 128 GB takes 108 only with the wired limit raised to 120000 MB or more.
    const roomy = BudgetLimits{ .wired_limit = 125_000 << 20, .ceiling = 125 * GB, .ceiling_name = "iogpu.wired_limit_mb", .margin = GiB, .user_wire = 125 * GB, .user_wire_name = "vm.user_wire_limit", .user_wire_margin = GiB };
    const beside = 5 * GB;
    const d128 = ramDefaultBudget(.gb128, 0, beside, roomy);
    try t.expectEqual(108 * GiB, d128.bytes);
    try t.expectEqual(@as(u64, 125_000), d128.wired_limit_mb);
    try t.expect(!d128.resident);
    try t.expectEqualStrings("", d128.bound_by);
    try t.expectEqual(80 * GiB, ramDefaultBudget(.gb96, 0, beside, roomy).bytes);
    var at = roomy;
    at.wired_limit = 120_000 << 20;
    try t.expectEqual(108 * GiB, ramDefaultBudget(.gb128, 0, beside, at).bytes);
    for ([_]u64{ 0, 119_999 << 20 }) |below| {
        var unset = roomy;
        unset.wired_limit = below;
        const d = ramDefaultBudget(.gb128, 0, beside, unset);
        try t.expectEqual(96 * GiB, d.bytes);
        try t.expectEqual(@as(u64, 0), d.wired_limit_mb);
        // The 96 GB tier does not read the wired limit.
        try t.expectEqual(80 * GiB, ramDefaultBudget(.gb96, 0, beside, unset).bytes);
    }

    // The flag wins, and the setting outranks the default; neither reaches `ramDefaultBudget`.
    try t.expectEqual(60 * GiB, resolveSsdBudget(60 * GiB, 40, true).bytes);
    const settings = resolveSsdBudget(0, 40, true);
    try t.expectEqual(40 * GiB, settings.bytes);
    try t.expectEqual(SsdBudgetOrigin.model_settings, settings.origin);
    // A 64 GB Mac has no tier: the choice stays unset and the load refuses by name.
    try t.expectEqual(SsdBudgetOrigin.none, resolveSsdBudget(0, 0, true).origin);

    // A low wire limit steps the default down to the largest whole GiB the limit holds, and names the limit.
    const low = BudgetLimits{ .wired_limit = 125_000 << 20, .ceiling = 100 * GB, .ceiling_name = "the GPU working set", .margin = GiB, .user_wire = 0, .user_wire_name = "", .user_wire_margin = GiB };
    const stepped = ramDefaultBudget(.gb128, 0, beside, low);
    const room = 100 * GB - GiB - beside;
    try t.expectEqual(room / GiB * GiB, stepped.bytes);
    try t.expect(stepped.bytes < 108 * GiB);
    try t.expectEqual(108 * GiB, stepped.wanted_bytes);
    try t.expectEqualStrings("the GPU working set", stepped.bound_by);
    // The user-wire cap binds when it is the lower one.
    const wired = BudgetLimits{ .wired_limit = 125_000 << 20, .ceiling = 125 * GB, .ceiling_name = "iogpu.wired_limit_mb", .margin = GiB, .user_wire = 100 * GB, .user_wire_name = "vm.global_user_wire_limit", .user_wire_margin = GiB };
    const by_wire = ramDefaultBudget(.gb128, 0, beside, wired);
    try t.expectEqual((100 * GB - GiB) / GiB * GiB, by_wire.bytes);
    try t.expectEqualStrings("vm.global_user_wire_limit", by_wire.bound_by);
    // The step-down never lands on a budget the wired-limit preflight refuses.
    try t.expect(by_wire.bytes + beside + GiB <= 125 * GB);
}

test "192 and 256 GB Macs load the shipped pack fully resident, a pack too big for the limits streams instead" {
    const t = testing;
    const GiB: u64 = 1 << 30;
    const GB: u64 = 1_000_000_000;
    // The shipped STQ pack: trunk 4.25 + DSpark 3.92 + tower 0.50 + Engram rows 0.57, the 1 GiB decode pool, 119.6 GB of experts.
    const shipped: u64 = 9_240_000_000 + GiB + 119_600_000_000;
    const beside = 3_400_000_000 + 1_600_000_000;
    for ([_]struct { tier: RamTier, ceiling: u64, wire: u64 }{
        .{ .tier = .gb192, .ceiling = 172 * GB, .wire = 175 * GB },
        .{ .tier = .gb256, .ceiling = 230 * GB, .wire = 233 * GB },
    }) |c| {
        const limits = BudgetLimits{ .ceiling = c.ceiling, .ceiling_name = "the GPU working set", .margin = GiB, .user_wire = c.wire, .user_wire_name = "vm.global_user_wire_limit", .user_wire_margin = GiB };
        const d = ramDefaultBudget(c.tier, shipped, beside, limits);
        try t.expect(d.resident);
        try t.expectEqual(shipped, d.bytes);
        try t.expectEqualStrings("", d.bound_by);
        // A pack that cannot be resident here streams at the largest budget that fits.
        const big = ramDefaultBudget(c.tier, c.ceiling, beside, limits);
        try t.expect(!big.resident);
        try t.expect(big.bytes < c.ceiling);
        try t.expectEqual(c.ceiling, big.wanted_bytes);
        try t.expectEqualStrings("the GPU working set", big.bound_by);
        // A pack with no resident layout (need 0) streams with nothing to warn about.
        const none = ramDefaultBudget(c.tier, 0, beside, limits);
        try t.expect(!none.resident);
        try t.expectEqualStrings("", none.bound_by);
    }
    // 128 and 96 GB never take the resident arm, whatever the pack.
    const roomy = BudgetLimits{ .ceiling = 400 * GB, .ceiling_name = "x", .margin = GiB, .user_wire = 0, .user_wire_name = "", .user_wire_margin = GiB };
    try t.expect(!ramDefaultBudget(.gb128, shipped, beside, roomy).resident);
    try t.expect(!ramDefaultBudget(.gb96, shipped, beside, roomy).resident);
    // The flag still wins on any tier.
    try t.expectEqual(50 * GiB, resolveSsdBudget(50 * GiB, 0, true).bytes);
}

test "only a model with serve defaults takes a RAM-tier default, and the load lines name the arm" {
    const t = testing;
    try t.expectEqual(@as(?RamTier, null), ramDefaultTierFor(&ModelConfig{ .model_type = "qwen4_exp" }));
    try t.expectEqual(@as(?RamTier, null), ramDefaultTierFor(&ModelConfig{ .model_type = "glm5_next" }));
    var buf: [160]u8 = undefined;
    const GiB: u64 = 1 << 30;
    try t.expectEqualStrings("[expert-stream] ssd budget: resident (RAM default)", budgetOriginLine(&buf, .{ .bytes = 130 * GiB, .origin = .ram_default, .setting_ignored = false }, true));
    try t.expectEqualStrings("[expert-stream] ssd budget: streamed 108 GB (RAM default)", budgetOriginLine(&buf, .{ .bytes = 108 * GiB, .origin = .ram_default, .setting_ignored = false }, false));
    try t.expectEqualStrings("[expert-stream] ssd budget: streamed 108 GB (RAM default, iogpu.wired_limit_mb 125000)", budgetOriginLine(&buf, .{ .bytes = 108 * GiB, .origin = .ram_default, .setting_ignored = false, .wired_limit_mb = 125_000 }, false));
    try t.expectEqualStrings("[expert-stream] ssd budget 60 GiB (--ssd-budget-gb)", budgetOriginLine(&buf, .{ .bytes = 60 * GiB, .origin = .flag, .setting_ignored = false }, false));
    try t.expectEqualStrings("[expert-stream] ssd budget 40 GiB (model-settings.json)", budgetOriginLine(&buf, .{ .bytes = 40 * GiB, .origin = .model_settings, .setting_ignored = false }, false));
    const d = RamDefault{ .bytes = 102 * GiB, .resident = false, .wanted_bytes = 108 * GiB, .bound_by = "the GPU working-set limit" };
    try t.expectEqualStrings("[expert-stream] default ssd budget 108 GiB does not fit the GPU working-set limit; using 102 GiB streamed", stepDownLine(&buf, d));
}

test "the ssd budget falls back to the per-model setting, and a non-streaming model ignores it" {
    const t = testing;
    const GiB: u64 = 1 << 30;

    const flag_wins = resolveSsdBudget(60 * GiB, 40, true);
    try t.expectEqual(60 * GiB, flag_wins.bytes);
    try t.expectEqual(SsdBudgetOrigin.flag, flag_wins.origin);
    try t.expect(!flag_wins.setting_ignored);

    const from_setting = resolveSsdBudget(0, 60, true);
    try t.expectEqual(60 * GiB, from_setting.bytes);
    try t.expectEqual(SsdBudgetOrigin.model_settings, from_setting.origin);
    try t.expect(!from_setting.setting_ignored);

    const neither = resolveSsdBudget(0, 0, true);
    try t.expectEqual(@as(u64, 0), neither.bytes);
    try t.expectEqual(SsdBudgetOrigin.none, neither.origin);

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

/// Stamp the per-request terms of a sliding DFlash2 assistant on the target's config.
pub fn stampDflashRequestTerms(io: std.Io, allocator: std.mem.Allocator, config: *ModelConfig, directory: []const u8) !void {
    if (!dflash_mod.probeIsDflash(io, allocator, directory)) return error.DflashAssistantRequired;
    var cfg = try dflash_mod.parseConfig(io, allocator, directory);
    defer cfg.deinit(allocator);
    if (!cfg.isDflash2() or cfg.hidden_size != config.hidden_size or cfg.sliding_window < 2) return error.DflashTargetMismatch;
    for (cfg.layer_types) |kind| if (kind != .sliding_attention) return error.DflashWindowUnsupported;
    config.dflash_loaded = true;
    if (arch_config.fact(config, "dflash_by_flag")) config.dflash_load_transient_bytes = (try dflash_mod.assistantLoadBytes(io, allocator, directory, dflash_mod.quantBitsFromEnv())).transient;
    const per_token = @as(u64, cfg.num_hidden_layers) * cfg.num_key_value_heads * cfg.head_dim * 4;
    // A draft block grows a 2047-row window to the next 256-row allocation.
    config.dflash_window_bytes = per_token * ((@as(u64, cfg.sliding_window) + cfg.block_size + 255) / 256 * 256);
    config.dflash_capture_bytes_per_token = (@as(u64, cfg.target_layer_ids.len) * cfg.hidden_size + @as(u64, cfg.hidden_size) * 2) * 2 + per_token;
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
    // Data bytes the header declares: the fc matrix, the codebook and the norm, all bf16.
    const stored: u64 = (1024 * 2048 + 256 * 128 + 2048) * 2;
    const body = try a.alloc(u8, 8 + header.len + stored);
    defer a.free(body);
    @memset(body, 0);
    std.mem.writeInt(u64, body[0..8], header.len, .little);
    @memcpy(body[8 .. 8 + header.len], header);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = body });
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];

    const billed = try sushiAssistantLoadBytes(io, a, dir);
    try testing.expect(billed < stored);
    try testing.expectEqual((try dflash_mod.assistantLoadBytes(io, a, dir, dflash_mod.quantBitsFromEnv())).resident, billed);

    var f = try @import("bills_fixtures_test.zig").LoadBillFixtures.init();
    defer f.deinit();
    const bare = loadRequirementBytes(&f.qwen, 44 * gib, false, 0);
    f.qwen.dflash_load_transient_bytes = gib / 2;
    try testing.expectEqual(bare + gib / 2, loadRequirementBytes(&f.qwen, 44 * gib, false, 0));
    // An MTP-on load pays both peaks, and a context bill still replaces them.
    try testing.expectEqual(44 * gib + gib * 5 / 4 + gib / 2 + gib, loadRequirementBytes(&f.qwen, 44 * gib, true, 0));
    try testing.expectEqual(44 * gib + 6 * gib, loadRequirementBytes(&f.qwen, 44 * gib, true, 6 * gib));
}

test "a cold load reads the serving bill on the config the load stamps with the drafter's request terms" {
    if (@import("arch/registry.zig").ofModelType("qwen4_exp") == null) return error.SkipZigTest;
    const io = std.testing.io;
    const a = std.testing.allocator;
    const gib: u64 = 1 << 30;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var f = try @import("bills_fixtures_test.zig").LoadBillFixtures.init();
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

/// Sizes the SSD budget for a model under a cache directory, null when the tier is off. `quiet` keeps the sizing
/// and refusal lines for a load: a re-arm check logs only when it arms.
pub const DiskResolver = *const fn (*model_mod.ModelConfig, []const u8, u64, bool) ?disk_placement.DefaultDiskBudget;

/// What a model's prefix-cache attach is sized from: the launch's cache settings, read alike by the load and by a
/// later re-arm of a tier the load declined.
pub const CacheAttach = struct {
    capacity: u32,
    ram_enabled: bool,
    mem_bytes: u64,
    mem_resolver: ?*const fn (*model_mod.ModelConfig, u64, BudgetRevise, *u64) u64,
    disk_resolver: ?DiskResolver,
    ssm_checkpoint_stride: u32,
    ssm_checkpoint_max: u32,
    /// How a re-arm reads the cache volume's free space.
    space_probe: disk_placement.SpaceProbeFn = disk_placement.volumeSpaceNear,
};

pub const AttachMode = enum { load, rearm };

/// Brings up the model's SSD tier and the hot prefix cache hanging off it, and bills the tier on the config.
/// `entry` has its config, path and handle; `disk_bytes` is the SSD ask. A re-arm of a declined tier logs nothing
/// when it fails and leaves the config declined; one that finds the cache built (a RAM tier) attaches the tier to it.
fn attachPrefixCache(allocator: std.mem.Allocator, io: std.Io, entry: *LoadedModel, p: CacheAttach, disk_bytes: u64, mode: AttachMode) !void {
    const config = entry.config.?;
    const xfm: ?*trunk.Trunk = if (entry.handle) |h| arch_handle.trunk(h) else null;
    const failed: log.Level = if (mode == .rearm) .debug else .warn;
    // Per-model hot prefix cache (Plan 03 → Plan 05 move). Hybrid recurrent
    // archs are accepted iff `ssm_checkpoint_stride > 0` (Phase 1 of the
    // performance plan): with per-stride SSM checkpoints we can rewind both
    // KV and SSM to a snapshotted prefix; without them, divergence forces a
    // full reset, so we keep the legacy single-slot path for hybrid.
    const enable_ssm_cps = p.ssm_checkpoint_stride > 0;
    const ram_prefix_cache = p.ram_enabled;
    // Phase 3 persists hybrid recurrent state too: the disk tier is allowed whenever the RAM tier
    // accepts the arch — pure-attention always, hybrid iff SSM checkpoints are enabled.
    const has_ssm_layers = config.full_attention_interval > 0;
    const disk_ok = !has_ssm_layers or enable_ssm_cps;
    const cache_applies = p.capacity > 0 and
        prefix_cache_mod.HotPrefixCache.shouldUse(config, enable_ssm_cps);
    // The SSD tier comes up BEFORE anything is sized for it: whether this model has one decides its
    // RAM semantics (SSD-first or not), its bills and its checkpoint capture. A tier that was wanted
    // and did not come up (no room, an unreadable fingerprint, a failed init) is recorded on the config,
    // so every bill reads the same answer as `--prefix-cache-disk off`.
    var disk_tier: ?kv_disk_cache.DiskTier = null;
    if (mode == .load and p.capacity > 0 and disk_bytes == 0) {
        const ask = bills.diskAsk();
        var off_buf: [96]u8 = undefined;
        if (ask.explicit()) log.info("{s}\n", .{disk_placement.diskOffLine(&off_buf, 0, "", ask.sourceName())});
    }
    if (cache_applies and disk_bytes > 0 and disk_ok) attach: {
        const layout = diskLayoutKey(allocator, config, entry.path, xfm) catch |err| {
            log.atLevel(failed, "[disk-cache] fingerprint failed: {s} — persistence off for this model\n", .{@errorName(err)});
            break :attach;
        };
        defer if (layout) |l| allocator.free(l);
        const fp = disk_placement.modelFingerprintFor(allocator, io, entry.path, layout, config.yarn_factor_override) catch |err| {
            log.atLevel(failed, "[disk-cache] fingerprint failed: {s} — persistence off for this model\n", .{@errorName(err)});
            break :attach;
        };
        defer allocator.free(fp);
        const base = disk_placement.defaultBaseDir(allocator) catch break :attach;
        defer allocator.free(base);
        const plan: disk_placement.DefaultDiskBudget = if (p.disk_resolver) |resolve|
            resolve(config, base, disk_placement.tierBytes(io, base, fp), mode == .rearm) orelse break :attach
        else
            .{ .bytes = disk_bytes, .bound = .flag, .cap = disk_bytes };
        disk_tier = kv_disk_cache.DiskTier.init(
            allocator,
            io,
            base,
            fp,
            plan.bytes,
            kv_disk_cache.DEFAULT_CHUNK_TOKENS,
        ) catch |err| {
            log.atLevel(failed, "[disk-cache] init failed: {s} — persistence off for this model\n", .{@errorName(err)});
            break :attach;
        };
        disk_tier.?.operator_cap = plan.cap;
    }
    if (mode == .rearm and disk_tier == null) return;
    errdefer if (disk_tier) |*t| t.deinit();
    if (disk_tier) |*tier| {
        tier.cp_thin =
            if (config.longCtxGated() or !ram_prefix_cache or arch_config.fact(config, "state_beside_kv")) .min_span_recency else .oldest;
        tier.ssm_max_per_entry = if (prefix_arch.diskCheckpointCap(config)) |cap|
            cap
        else if (arch_config.fact(config, "kda_state") or config.longCtxGated() or !ram_prefix_cache)
            kv_disk_cache.SSM_DISK_MAX_PER_ENTRY
        else
            kv_disk_cache.SSM_DISK_MAX_PER_ENTRY_LEGACY;
        try prefix_arch.setDiskGrownSpec(allocator, config, tier);
    }
    config.prefix_cache_disk_declined = disk_bytes > 0 and disk_tier == null;
    const disk_prefix_cache = disk_tier != null;
    if (cache_applies and (ram_prefix_cache or disk_prefix_cache)) {
        if (entry.prefix_cache == null) {
            // The weights are resident here, so the resolver's active-memory read
            // is honest; the raw launch budget never reaches initWithMem (a 40 GB
            // cap beside a ~70 GB pack was the 2026-08-30 uncatchable Metal OOM).
            // Disk-only mode retains no reusable KV in RAM, so it asks for no RAM budget.
            var ssd_idle_mem: u64 = 0;
            const clamped_prefix_mem = hotCacheMemForLoad(ram_prefix_cache, p.mem_resolver, config, p.mem_bytes, &ssd_idle_mem);
            if (ram_prefix_cache) {
                // On an SSD-first arch `--prefix-cache-mem` is the idle allowance beside the live session.
                const ram_bytes = if (prefix_cache_mod.ssdFirstActive(config, disk_prefix_cache, ram_prefix_cache)) ssd_idle_mem else clamped_prefix_mem;
                log.info("Allocating {d:.1} GB RAM for the prefix cache (--prefix-cache-mem)\n", .{@as(f64, @floatFromInt(ram_bytes)) / (1024.0 * 1024.0 * 1024.0)});
            }
            entry.prefix_cache = prefix_cache_mod.HotPrefixCache.initWithMem(
                allocator,
                if (ram_prefix_cache) p.capacity else 0,
                clamped_prefix_mem,
            );
            entry.prefix_cache.?.qsa_history_required = config.indexer_budget != 0 and !arch_config.fact(config, "latent_kv");
            // Checkpoint-retention arch gate, mirrored once: `HotPrefixCache`/`DiskTier` never
            // see a ModelConfig. The ungated value names the previous behaviour at each site.
            entry.prefix_cache.?.cp_thin = if (config.longCtxGated() or arch_config.fact(config, "state_beside_kv")) .min_span_recency else .min_span;
            entry.prefix_cache.?.ssd_idle_mem = ssd_idle_mem;
            entry.prefix_cache.?.hybrid = has_ssm_layers or arch_config.fact(config, "state_beside_kv");
        }
        if (disk_tier) |tier| entry.prefix_cache.?.disk = tier;
        entry.prefix_cache.?.arrays_on = prefix_arch.arraysOn(config);
        if (entry.prefix_cache.?.disk) |*d| {
            // MLX reads and writes the plain-file backend's safetensors, so that tier's arrays stay MLX's.
            if (d.fs.vfs != null) d.arrays_on = entry.prefix_cache.?.arrays_on else if (entry.prefix_cache.?.arrays_on != null)
                log.warn("[disk-cache] SUSHI_PREFIX_FS=real: this model's state is not MLX's, and the tier's files pass through MLX arrays\n", .{});
        }
        // SSD-first: arch + env switch + a live disk tier (attached above); without one qwen4_exp takes the RAM arm.
        entry.prefix_cache.?.ssd_first = prefix_cache_mod.ssdFirstActive(
            config,
            entry.prefix_cache.?.disk != null,
            ram_prefix_cache,
        );
        // Every tier writes off the inference thread, so a commit lands whole after the response.
        if (entry.prefix_cache.?.disk) |*d| {
            d.writer_permit_bytes = server_state.disk_writer_permit_bytes;
            d.enableBackgroundWriter();
        }
        if (entry.prefix_cache.?.ssd_first) {
            entry.prefix_cache.?.disk.?.ssd_first = true;
            // Startup sweep of strays + root-wide LRU across sibling fingerprints.
            entry.prefix_cache.?.disk.?.sweepSiblings();
        }
        entry.ssm_checkpoint_stride = p.ssm_checkpoint_stride;
        const cp_max = prefix_arch.checkpointMax(config, p.ssm_checkpoint_max);
        entry.ssm_checkpoint_max = cp_max;
        // The cache re-applies the cap after a replace-path merge; without this
        // it defaults to 0 (unlimited) and multi-turn entries grow unbounded.
        entry.prefix_cache.?.ssm_checkpoint_max = cp_max;
    }
}

/// Attaches the SSD tier a load declined, on the inference thread with no request holding the model: the tier, the
/// cache that hangs off it if the load built none, and the bill, which reads `prefix_cache_disk_declined`. Quiet
/// until it arms; true when it did. A non-explicit ask waits for the volume to leave room above the reserve.
pub fn rearmDiskTier(allocator: std.mem.Allocator, io: std.Io, entry: *LoadedModel, p: CacheAttach, why: []const u8) bool {
    const config = entry.config orelse return false;
    if (!config.prefix_cache_disk_declined) return false;
    const ask = bills.diskAsk();
    if (ask.bytes == 0) return false;
    if (!ask.explicit()) {
        const base = disk_placement.defaultBaseDir(allocator) catch return false;
        defer allocator.free(base);
        if (p.space_probe(base)) |v| if (disk_placement.freeDiskRoom(v.free) == 0) return false;
    }
    attachPrefixCache(allocator, io, entry, p, ask.bytes, .rearm) catch |err| {
        log.debug("[disk-cache] re-arm failed: {s}\n", .{@errorName(err)});
        return false;
    };
    if (config.prefix_cache_disk_declined) return false;
    const d = &entry.prefix_cache.?.disk.?;
    log.info("[disk-cache] re-armed ({s}): {d:.1} GB at {s}\n", .{ why, @as(f64, @floatFromInt(d.max_bytes)) / (1024.0 * 1024.0 * 1024.0), d.root });
    return true;
}

/// Phase A1 → Plan 05: do the full model load on the inference thread.
/// mlx ops here bind to this thread's GPU stream from t0; subsequent
/// forwards stay on the same thread.
///
/// `params` is duck-typed (`anytype`): both `LoadParams` (startup) and
/// `*LoadRequest` (on-demand) supply the same field set — `entry`,
/// `config`/`tok`/`chat_config` (heap pointers), `model_dir`,
/// `drafter_dir`, `load_vision`, `warmup_eager`, `kv_quant`,
/// `prefix_cache_capacity`, `prefix_cache_mem_bytes`. The function reads them by name.
///
/// On any error the partial state has already been freed via errdefer; the
/// caller decides how to surface (startup → recordLoadError + signal
/// started; on-demand → req.error_name + done broadcast).
/// The MLX message behind a failed load step, for its log line.
fn latchedCause(buf: *[512]u8) []const u8 {
    return mlx.peekErrorMessage(buf) orelse "no MLX message";
}

/// An MLX error the load raised and nothing handled is said once and cleared, so it cannot become the first request's
/// `MlxFailure`. A step whose result matters returns `error.MlxError` through `mlx.check` and stops the load; what is
/// latched here came from a call nobody checked, after the weights bound and the warmups ran.
fn settleLoadLatch(had_error: bool) void {
    if (had_error) return; // raised before the load began: it belongs to someone else
    var buf: [512]u8 = undefined;
    const msg = mlx.takeError(&buf) orelse return;
    log.warn("[load] MLX raised an error nothing handled during the load; cleared so the first request does not fail on it: {s}\n", .{msg});
}

pub fn doLoadOnInferenceThread(sch: *Scheduler, params: anytype) !void {
    load_clock.begin();
    defer load_clock.end();
    // The tally at both ends of the load: what lies between is the load's own.
    generate_mod.logMlxCalls();
    defer generate_mod.logMlxCalls();
    const had_error_at_start = mlx.errorPending();
    try model_mod.archGate(params.config);
    if (params.config.unsupported_note) |note| log.warn("{s}\n", .{note});
    var streaming_resident_bytes: ?u64 = null;
    var expert_source_assigned = false;
    errdefer if (expert_source_assigned) {
        sch.allocator.free(params.config.expert_source_dir.?);
        params.config.expert_source_dir = null;
    };
    var launch_buf: [384]u8 = undefined;
    if (launchRefusal(&launch_buf, params.config, params.kv_quant)) |r| {
        log.err("{s}\n", .{r.line});
        return r.err;
    }
    var window_buf: [256]u8 = undefined;
    if (contextWindowRefusal(&window_buf, params.config, sch.ctx_size_flag)) |line| {
        log.err("{s}\n", .{line});
        return error.ContextSizeAboveWindow;
    }
    if (arch_flags.settingsSsdPrefetchIgnoredLine(params.config)) |line| log.info("{s}\n", .{line});
    if (arch_flags.settingsCoolerIgnoredLine(params.config)) |line| log.info("{s}\n", .{line});
    if (arch_flags.settingsKvQuantIgnoredLine(params.config)) |line| log.info("{s}\n", .{line});
    if (arch_flags.settingsMtpIgnoredLine(params.config)) |line| log.info("{s}\n", .{line});
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
        var budget = streaming_budget;
        const tier = ramDefaultTierFor(params.config);
        if (budget.bytes == 0 and tier == null) {
            log.err("{s}\n", .{budget_required_message});
            return error.ExpertStreamingRequired;
        }
        const mtp = mtpChoiceFor(params.mtp, params.config);
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
        const assistant_bytes = try streamedAssistantBytes(sch.io, sch.allocator, params.config, params.no_drafter, params.drafter_dir, params.model_dir);
        if (budget.bytes == 0) budget = try ramDefaultChoice(sch, params, tier.?, load_vision, assistant_bytes);
        const plan = try planExpertStreamingWith(sch.io, sch.allocator, params.config, params.model_dir, budget.bytes, load_vision, assistant_bytes);
        vision_cost = plan.vision;
        load_vision = plan.vision.on;
        var origin_buf: [160]u8 = undefined;
        log.info("{s}\n", .{budgetOriginLine(&origin_buf, budget, if (plan.resolved.ledger) |led| led.resident else false)});
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
        if (plan.resolved.ledger) |led| if (led.pool_bytes > 0) gpu_mem.capMlxPool(led.pool_bytes);
        if (params.config.expert_source_dir == null) {
            params.config.expert_source_dir = try sch.allocator.dupe(u8, params.model_dir);
            expert_source_assigned = true;
        }
        streaming_resident_bytes = plan.split.trunk +| plan.split.mtp +| plan.split.vision;
        if (params.expert_cache_fit_resolver) |fit| try fit(params.config, streaming_resident_bytes.?);
    } else if (try arch_load.residentLoadBytes(sch.io, sch.allocator, params.model_dir, params.config, load_vision, mtpChoiceFor(params.mtp, params.config).on)) |bytes| {
        streaming_resident_bytes = bytes;
    } else if (model_mod.usesSushiQuantMemoryBill(params.config)) {
        streaming_resident_bytes = try sushiResidentLoadBytes(sch.io, sch.allocator, params.model_dir, params.config, load_vision, mtpChoiceFor(params.mtp, params.config).on);
    }
    if (params.config.has_vision) {
        var line_buf: [256]u8 = undefined;
        log.info("{s}\n", .{visionLoadLine(&line_buf, vision_choice, streamed, vision_cost)});
    } else if (model_settings.vision_flag == true) log.warn("[vision] --vision ignored: this checkpoint has no vision tower\n", .{});

    // Resolve the sidecar before preflight so billing and loading see the same dependency.
    const drafter = LoadDrafterDir.resolve(sch.io, sch.allocator, params.config, params.no_drafter or try arch_load.drafterOff(params.config, params.drafter_dir), params.drafter_dir, params.model_dir);
    defer drafter.deinit(sch.allocator);
    var drafter_dir = drafter.dir;
    var prepared_drafter_path: ?[]u8 = null;
    defer if (prepared_drafter_path) |path| sch.allocator.free(path);
    if (drafter.owned != null) log.info("[dflash] auto-detected assistant: {s}\n", .{drafter_dir});
    if (dflash_mod.drafterDepthExplicit() and drafter_dir.len == 0) log.warn("[dflash] --drafter-min-depth/--drafter-max-depth ignored: no drafter is loaded for this model\n", .{});
    if (model_mod.usesSushiQuantMemoryBill(params.config) and drafter_dir.len > 0 and !streamed)
        streaming_resident_bytes = streaming_resident_bytes.? +| try sushiAssistantLoadBytes(sch.io, sch.allocator, drafter_dir);
    try arch_load.drafterChosen(sch.io, sch.allocator, params.config, drafter_dir, params.no_drafter, mtpChoiceFor(params.mtp, params.config).on);
    if (try arch_load.stageDrafter(sch.io, sch.allocator, params.config, params.model_dir, drafter_dir, params.no_drafter)) |assistant_bytes|
        streaming_resident_bytes = streaming_resident_bytes.? +| assistant_bytes;

    // GPU-memory pre-flight (MLX path). A Metal OOM during weight load / warmup
    // is thrown by MLX as a C++ exception that can't be caught across the C ABI,
    // so it terminates the whole process. Refuse the load up front instead, with
    // an actionable error, when free RAM clearly can't hold the weights + warmup
    // headroom — catches the common "restarted before the prior server released
    // its memory" case. Bypass with --skip-mem-preflight.
    if (!scheduler_mod.skipMemPreflight()) {
        const weights_bytes = streaming_resident_bytes orelse modelDiskBytes(sch.io, params.model_dir);
        const gpu_limit = mlx.maxRecommendedWorkingSet();
        const Reader = struct {
            io: std.Io,
            gpu_limit: usize,
            sushi: bool,
            wired_floor: u64,
            pub fn read(r: *@This()) u64 {
                if (!r.sushi) return bills.effectiveAvailableBytes(status.getAvailableMemBytes(), status.getProcAvailableMemBytes(), r.gpu_limit);
                return bills.effectiveSushiAvailableBytes(status.getAvailableMemBytes(), status.getProcAvailableMemBytes(), r.gpu_limit, gpu_mem.activeBytes(), r.wired_floor);
            }
            pub fn sleep(r: *@This()) void {
                std.Io.sleep(r.io, .fromMilliseconds(500), .real) catch {};
            }
        };
        var reader = Reader{ .io = sch.io, .gpu_limit = gpu_limit, .sushi = model_mod.usesSushiQuantMemoryBill(params.config), .wired_floor = bills.wiredCeilingFloorFor(params.config) };
        var avail_bytes = reader.read();
        const mtp_on = mtpChoiceFor(params.mtp, params.config).on;
        var serving = if (scheduler_mod.load_serving_bill) |bill| bill(params.config, avail_bytes -| weights_bytes) else null;
        var needed = loadRequirementBytes(params.config, weights_bytes, mtp_on, if (serving) |b| b.needed else 0);
        if (bills.memInsufficientForLoad(weights_bytes, avail_bytes, needed)) {
            log.info("[preflight] {d:.2} GB available of {d:.2} GB needed; waiting while memory is still being released\n", .{
                @as(f64, @floatFromInt(avail_bytes)) / (1024.0 * 1024.0 * 1024.0),
                @as(f64, @floatFromInt(needed)) / (1024.0 * 1024.0 * 1024.0),
            });
            avail_bytes = bills.settleAvailableBytes(avail_bytes, needed, &reader);
            serving = if (scheduler_mod.load_serving_bill) |bill| bill(params.config, avail_bytes -| weights_bytes) else null;
            needed = loadRequirementBytes(params.config, weights_bytes, mtp_on, if (serving) |b| b.needed else 0);
        }
        const active: usize = if (reader.sushi) gpu_mem.activeBytes() else 0;
        log.info("[preflight] weights ~{d:.2} GB, needs ~{d:.2} GB, available {d:.2} GB (warmup + net or context ~{d:.2} GB)\n", .{
            @as(f64, @floatFromInt(weights_bytes)) / (1024.0 * 1024.0 * 1024.0),
            @as(f64, @floatFromInt(needed)) / (1024.0 * 1024.0 * 1024.0),
            @as(f64, @floatFromInt(avail_bytes)) / (1024.0 * 1024.0 * 1024.0),
            @as(f64, @floatFromInt(needed -| weights_bytes)) / (1024.0 * 1024.0 * 1024.0),
        });
        const gpu_exhausted = model_mod.usesSushiQuantMemoryBill(params.config) and gpu_limit > 0 and active >= gpu_limit;
        if (weights_bytes > 0 and (bills.memInsufficientForLoad(weights_bytes, avail_bytes, needed) or gpu_exhausted)) {
            const gb = 1024.0 * 1024.0 * 1024.0;
            if (serving) |b| {
                const maximum: u32 = if (avail_bytes < loadRequirementBytes(params.config, weights_bytes, mtp_on, 0)) 0 else b.max_context;
                sch.registry.mutex.lockUncancelable(sch.io);
                const refusal = model_registry_mod.MemoryContextRefusal{ .requested = b.requested_context, .maximum = maximum, .needed = needed, .available = avail_bytes, .kv_bits = b.kv_bits, .chunk = b.chunk };
                params.entry.memory_context_refusal = refusal;
                if (@TypeOf(params) == *LoadRequest) params.memory_context_refusal = refusal;
                sch.registry.mutex.unlock(sch.io);
                log.err("Insufficient memory for requested context: {d} tokens at KV{d} and prefill chunk {d} need ~{d:.2} GiB including {d:.2} GiB of enabled model/vision/MTP weights, but only {d:.2} GiB is available. Maximum context with this memory budget: {d} tokens. Lower --ctx-size or free memory; --skip-mem-preflight, or \"Skip the memory check before loading\" in Settings, overrides this check.\n", .{
                    b.requested_context,                         b.kv_bits,                                 b.chunk,                                                                              @as(f64, @floatFromInt(needed)) / gb,
                    @as(f64, @floatFromInt(weights_bytes)) / gb, @as(f64, @floatFromInt(avail_bytes)) / gb, maximum,
                });
                return error.InsufficientMemory;
            }
            log.err("Insufficient memory to load model: needs ~{d:.1} GB free ({d:.1} GB of weights plus headroom for warmup buffers and a baseline KV cache) but only {d:.1} GB is available ({s}). Close other models/apps (or wait for a prior sushi to fully exit) and retry; pass --skip-mem-preflight (or turn on \"Skip the memory check before loading\" in Settings) to override.\n", .{
                @as(f64, @floatFromInt(needed)) / gb,
                @as(f64, @floatFromInt(weights_bytes)) / gb,
                @as(f64, @floatFromInt(avail_bytes)) / gb,
                if (reader.sushi and reader.wired_floor > 0) "the iogpu.wired_limit_mb ceiling less what other processes hold on the GPU" else "free RAM, capped at the GPU working-set limit that iogpu.wired_limit_mb sets",
            });
            return error.InsufficientMemory;
        }
    }

    if (try arch_load.prepareDrafter(sch.io, sch.allocator, params.config, params.model_dir, drafter_dir, streaming_resident_bytes)) |prepared| {
        prepared_drafter_path = prepared.path;
        streaming_resident_bytes = prepared.resident_bytes;
        drafter_dir = prepared.path;
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

    if (@TypeOf(params) == LoadParams) {
        signalPrechecked(sch);
        const hold_ms = startupLoadDelayMs();
        if (hold_ms > 0) {
            log.info("[load] SUSHI_TEST_LOAD_DELAY_MS: holding the startup load {d} ms\n", .{hold_ms});
            std.Io.sleep(sch.io, .fromMilliseconds(@intCast(hold_ms)), .awake) catch {};
        }
    }

    // Weights — first mlx call. Binds the stream on this thread.
    const weights_ptr = try sch.allocator.create(Weights);
    errdefer sch.allocator.destroy(weights_ptr);
    var load_config = params.config.*;
    // The resolved launch choice gates lazy native-head tensors before Trunk.init binds them.
    load_config.mtp_override = mtpChoiceFor(params.mtp, params.config).on;
    weights_ptr.* = try model_mod.loadWeightsForConfig(sch.io, sch.allocator, params.model_dir, &load_config, load_vision);
    errdefer weights_ptr.deinit();
    model_mod.resolveWeightPrefix(params.config, weights_ptr);

    // Trunk — owns the bulk of the GPU memory.
    const xfm_ptr = try sch.allocator.create(Trunk);
    errdefer sch.allocator.destroy(xfm_ptr);
    const build_t0 = load_clock.now();
    xfm_ptr.* = try Trunk.init(sch.io, sch.allocator, params.config.*, weights_ptr);
    load_clock.buildDone(build_t0);
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

    // Propagate the kv-quant config to the Trunk's own cache. Slot
    // caches in serve mode honor this independently in `Slot.init`; this
    // call covers any path that still touches `xfm.cache` directly (legacy
    // single-slot fallbacks, prompt-cache reuse).
    // An explicit launch flag outranks the per-model settings stamped on the config at BOTH construction sites.
    const kv_cache = bills.kvCacheForLaunch(params.config, params.kv_quant);
    const load_ctx = model_settings.contextPick(sch.ctx_size_flag, params.config.ctx_override);
    var ctx_buf: [16]u8 = undefined;
    log.info("[kv-cache] {s} ({s}); ctx {s} ({s})\n", .{
        kv_cache.label(),                                      kv_cache.sourceName(),
        model_settings.contextLabel(&ctx_buf, load_ctx.value), settings_resolve.sourceLabel(load_ctx.source, "--ctx-size"),
    });
    const kv_quant_config = kv_cache.config;
    const mtp = mtpChoiceFor(params.mtp, params.config);
    const mtp_streaming_off = mtpDefaultOffUnderStreaming(mtp, params.config.expert_streaming, params.config.stream_mtp_head);
    const acceptance = settings_resolve.acceptanceFor(params.config.mtp_acceptance_override);
    const greedy_tail = settings_resolve.greedyTailFor(params.config.mtp_greedy_tail_override);
    // GLM has no MTP head; the server's `MTP: off` line already says so.
    if (!arch_config.fact(params.config, "no_mtp_head")) log.info("[mtp] {s} ({s}{s}); acceptance {s} ({s}); greedy tail {s} ({s})\n", .{
        if (mtp_streaming_off) "off" else mtp.label(), if (mtp_streaming_off) "streaming; " else "", mtp.sourceName(),
        mtp_acceptance_mod.name(acceptance.value),     settings_resolve.sourceLabel(acceptance.source, model_settings.acceptanceFlagName(acceptance.value)),
        if (greedy_tail.value) "on" else "off",        settings_resolve.sourceLabel(greedy_tail.source, "--mtp-greedy-tail"),
    });
    const mtp_enabled = mtp.on and !mtp_streaming_off and !arch_config.fact(params.config, "no_mtp_head");
    arch_serve.loadLines(params.config, .{ .io = sch.io, .allocator = sch.allocator, .model_dir = params.model_dir, .mtp_source = mtp.sourceName() });
    arch_load.logLoad(params.config, kv_quant_config, mtp.on, params);
    if (@import("chat.zig").templateTakesPreserveThinking(params.chat_config.chat_template)) {
        const keep = settings_resolve.resolve(bool, model_settings.preserve_thinking_flag, params.config.preserve_thinking_override, model_settings.default_preserve_thinking);
        log.info("[chat] preserve_thinking {s} ({s})\n", .{ if (keep.value) "on" else "off", settings_resolve.sourceLabel(keep.source, "--preserve-thinking") });
    }
    const think = settings_resolve.resolve(f32, model_settings.think_penalty_flag, params.config.think_penalty_override, model_settings.default_think_penalty);
    if (think.value > 0) log.info("[think-penalty] lambda {d} ({s}); a request's think_penalty outranks it\n", .{ think.value, settings_resolve.sourceLabel(think.source, "--think-penalty") });
    if (kv_quant_config.scheme != .off) {
        try xfm_ptr.cache.reinit(params.config.num_hidden_layers, kv_quant_config);
    }
    Trunk.mtp_head_kv_quant_flag = params.mtp_head_kv_quant;
    try arch_state.applyHeadKvQuant(xfm_ptr, kv_quant_config);

    // Wire model weights into GPU memory (prevents paging, matches mlx-lm).
    // Policy in mlx.applyWiredPolicy; re-applied in runLoadRequest /
    // runUnloadRequest so `fit` capacity tracks the live set across
    // load/unload churn (this early call covers warmup's forwards).
    const wired_t0 = load_clock.now();
    logWiredPolicy(mlx.applyWiredPolicy());
    load_clock.wiredDone(wired_t0);

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
        arch_state.compile(xfm_ptr, .all);
    }

    if (trunk.diagEnvOn("SUSHI_PREFILL_UBENCH")) prefillUbench(sch.allocator, xfm_ptr, params.config, params.tok);
    arch_load.warmUbench(sch.allocator, xfm_ptr, kv_quant_config, params.tok);

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
            const profile_pass = trunk.diagEnvOn("SUSHI_DECODE_FWD_UBENCH_PROFILE");
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
            const verify_rows = rows > 1 and rows <= (arch_state.verifyRowsMax(&xfm_ptr.config) orelse 0);
            arch_state.setVerifyRows(&xfm_ptr.config, &ctx.arch_ctx, verify_rows);
            log.info("[fwd-ubench] rows={d} capture={} verify_rows={}\n", .{ tok_slice.len, ctx.capture_ssm_seq, verify_rows });
            // Decode widths run as a decode tick does; a prefill width keeps the size budget.
            const tick: ?mlx.DecodeBuffers = if (rows <= 16) mlx.DecodeBuffers.begin() else null;
            defer if (tick) |k| k.end();
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
            const proc_cpu0 = processCpuNs();
            const engine0 = mlx.engine.stats();
            mlx.engine.resetKernelProfile();
            var buffers0: usize = 0;
            _ = mlx.mlx_metal_command_buffer_count(&buffers0);
            var active0: usize = 0;
            _ = mlx.mlx_reset_peak_memory();
            _ = mlx.mlx_get_active_memory(&active0);
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
            const proc_cpu_ns = processCpuNs() - proc_cpu0;
            var buffers1: usize = 0;
            _ = mlx.mlx_metal_command_buffer_count(&buffers1);
            var peak1: usize = 0;
            _ = mlx.mlx_get_peak_memory(&peak1);
            const dn: f64 = @floatFromInt(@max(done, 1));
            const ms = @as(f64, @floatFromInt(sw.read())) / 1.0e6 / dn;
            log.info("[fwd-ubench] {d} decode forwards, eval-per-step: {d:.3} ms/forward, process CPU {d:.3} ms/forward on every thread, {d:.0} command buffers/forward, MLX peak {d} MB over the start (build {d:.3} ms CPU + eval {d:.3} ms GPU of which submit {d:.3} ms ({d:.3} ms thread CPU), {d:.0} ops/forward)\n", .{
                done,
                ms,
                @as(f64, @floatFromInt(proc_cpu_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(buffers1 - buffers0)) / dn,
                (peak1 -| active0) >> 20,
                @as(f64, @floatFromInt(build_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(eval_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(submit_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(submit_cpu_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(ops_total)) / dn,
            });
            if (mlx.engine.enabled) {
                const e = mlx.engine.stats();
                log.info("[fwd-ubench] engine: {d:.0} dispatches behind {d:.0} barriers, {d:.1} command buffers, {d:.1} waits, {d:.0} imports, {d:.1} lends per forward; slice updates {d} in place, {d} copied; pool {d} MB; GPU waits {d:.3} ms, commits {d:.3} ms per forward; GPU busy {d:.3} ms and idle between passes {d:.3} ms per forward ({d} buffers untimed); pauses per forward under 20 us / 100 us / 1 ms / longer: {d:.1} ({d:.3} ms) / {d:.1} ({d:.3} ms) / {d:.1} ({d:.3} ms) / {d:.1} ({d:.3} ms)\n", .{
                    @as(f64, @floatFromInt(e.dispatches - engine0.dispatches)) / dn,
                    @as(f64, @floatFromInt(e.barriers - engine0.barriers)) / dn,
                    @as(f64, @floatFromInt(e.submits - engine0.submits)) / dn,
                    @as(f64, @floatFromInt(e.waits - engine0.waits)) / dn,
                    @as(f64, @floatFromInt(e.imports - engine0.imports)) / dn,
                    @as(f64, @floatFromInt(e.lends - engine0.lends)) / dn,
                    e.updates_in_place - engine0.updates_in_place,
                    e.updates_copied - engine0.updates_copied,
                    e.pool_bytes >> 20,
                    @as(f64, @floatFromInt(e.wait_ns - engine0.wait_ns)) / 1.0e6 / dn,
                    @as(f64, @floatFromInt(e.commit_ns - engine0.commit_ns)) / 1.0e6 / dn,
                    @as(f64, @floatFromInt(e.gpu_busy_ns - engine0.gpu_busy_ns)) / 1.0e6 / dn,
                    @as(f64, @floatFromInt(e.gpu_gap_ns - engine0.gpu_gap_ns)) / 1.0e6 / dn,
                    e.gpu_untimed - engine0.gpu_untimed,
                    @as(f64, @floatFromInt(e.gpu_gap_count[0] - engine0.gpu_gap_count[0])) / dn,
                    @as(f64, @floatFromInt(e.gpu_gap_sum[0] - engine0.gpu_gap_sum[0])) / 1.0e6 / dn,
                    @as(f64, @floatFromInt(e.gpu_gap_count[1] - engine0.gpu_gap_count[1])) / dn,
                    @as(f64, @floatFromInt(e.gpu_gap_sum[1] - engine0.gpu_gap_sum[1])) / 1.0e6 / dn,
                    @as(f64, @floatFromInt(e.gpu_gap_count[2] - engine0.gpu_gap_count[2])) / dn,
                    @as(f64, @floatFromInt(e.gpu_gap_sum[2] - engine0.gpu_gap_sum[2])) / 1.0e6 / dn,
                    @as(f64, @floatFromInt(e.gpu_gap_count[3] - engine0.gpu_gap_count[3])) / dn,
                    @as(f64, @floatFromInt(e.gpu_gap_sum[3] - engine0.gpu_gap_sum[3])) / 1.0e6 / dn,
                });
                mlx.engine.logKernelProfile(dn);
            }
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
                trunk.decodeProfileSession(@intCast(rows));
                for (0..n) |_| {
                    const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                    defer _ = mlx.mlx_array_free(ti);
                    const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                    _ = mlx.mlx_array_eval(lg);
                    _ = mlx.mlx_array_free(lg);
                }
                trunk.decodeProfileSession(0);
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
        const os_build = trunk.macosProductVersion(&os_buf) orelse "";
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
        if (!drafter_mod.supportsDflash(xfm_ptr)) {
            log.err("DFlash assistant at {s} refused: {s} models have no DFlash2 drafter\n", .{ drafter_dir, params.config.model_type });
            return error.DrafterUnsupported;
        }
        const d = try sch.allocator.create(DflashModel);
        d.* = arch_load.loadDflash(sch.io, sch.allocator, xfm_ptr, drafter_dir) catch |err| {
            sch.allocator.destroy(d);
            log.err("Failed to load DFlash assistant at {s}: {s}\n", .{ drafter_dir, @errorName(err) });
            return err;
        };
        const bound = arch_load.bindDflash(sch.allocator, xfm_ptr, d, kv_quant_config, params.tok) catch |err| {
            d.deinit();
            sch.allocator.destroy(d);
            log.err("DFlash assistant at {s} is incompatible with target: {s}\n", .{ drafter_dir, @errorName(err) });
            return err;
        };
        if (!bound) {
            d.deinit();
            sch.allocator.destroy(d);
            log.err("--drafter: a DFlash2 assistant serves {s} only\n", .{arch_packs.dflash_targets_and});
            return error.DflashTargetMismatch;
        }
        dflash_ptr = d;
        const wide_lane = dflash_mod.wideVerifyLaneAvailable();
        const block_cap = dflash_mod.blockCapForMachine(chip_mod.chipBrand());
        sch.drafter_block_size = dflash_mod.resolveBlockSize(d.config.block_size, wide_lane, block_cap.cap);
        if (arch_load.dflashRowBounds(params.config, dflash_mod.drafter_min_depth, dflash_mod.drafter_max_depth, d.config.block_size) catch |err| {
            d.deinit();
            sch.allocator.destroy(d);
            return err;
        }) |rows| {
            sch.drafter_block_size = rows;
        }
        var cap_note_buf: [96]u8 = undefined;
        const cap_note: []const u8 = if (!wide_lane and arch_config.fact(params.config, "chip_capped_drafter_block") and !dflash_mod.drafterDepthExplicit() and d.config.block_size > sch.drafter_block_size)
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
        arch_load.logDflashReady(params.config, sch.drafter_block_size);
    } else if (drafter_dir.len > 0) {
        log.err("Drafter at {s} is not a DFlash assistant (its config lacks block_size, mask_token_id and target_layer_ids)\n", .{drafter_dir});
        return error.UnsupportedDrafter;
    }
    errdefer if (dflash_ptr) |d| {
        d.deinit();
        sch.allocator.destroy(d);
    };

    const mtp_cost_profile: mtp_mod.MtpCostProfile = .generic;
    const aux_head = arch_load.loadHead(sch.io, sch.allocator, params.model_dir, params.config, xfm_ptr, mtp_enabled);
    errdefer if (aux_head) |h| arch_load.freeHead(params.config, sch.allocator, h);
    arch_load.prepare(params.config, xfm_ptr);
    // The qwen4_exp in-checkpoint head's coarse rerank build: a
    // `requantizeRows` of the whole
    // trunk lm_head plus a synchronous eval of ~240 MB. Lazily it ran inside
    // the FIRST request's draft chain — on this thread, mid-round, with the
    // stream drained — i.e. first-token latency for whoever loaded the model.
    // Gated exactly like `entry.mtp`'s `.qwen4` arm below: `--no-mtp` never
    // drafts, so it never pays. One-shot, so the draft path's ask stays a
    // pure read; if this ever does not run, that ask still builds.
    if (mtp_enabled) _ = arch_state.buildDraftRerank(xfm_ptr);

    if (xfm_ptr.expert_stream) |engine| warm: {
        if (engine.plan.resident) break :warm;
        const layers = expert_stream_mod.moeLayerCount(engine.geometry);
        const ranked = engine.warm_usage != null;
        const slots = expert_stream_mod.warmedSlots(engine.plan.slots_per_layer, ranked);
        const bytes = if (ranked) engine.plan.cache_bytes else @as(u64, slots) * layers * engine.plan.expert_bytes;
        log.info("[expert-stream] cache warm: preloading {d:.3} GB ({s})\n", .{ @as(f64, @floatFromInt(bytes)) / 1e9, if (ranked) "every pool, most used experts first" else "80% of cache slots, lowest expert IDs" });
        const start = std.Io.Timestamp.now(sch.io, .awake);
        const cache_t0 = load_clock.now();
        try engine.warmCache();
        load_clock.cacheDone(cache_t0);
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
            var cause: [512]u8 = undefined;
            log.warn("Warmup failed ({s}: {s}); continuing without it — first request may be slow.\n", .{ @errorName(err), latchedCause(&cause) });
        };
        const warmup_ns: u64 = @intCast(warmup_start.untilNow(sch.io, .awake).nanoseconds);
        log.info("Warmup complete ({d} ms).\n", .{warmup_ns / std.time.ns_per_ms});
        if (mtp_enabled) {
            const cap = generate_mod.Generator.resolveMtpDepthCapForProfile(params.mtp_depth, mtp_cost_profile);
            arch_load.warmVerify(params.config, xfm_ptr, cap, kv_quant_config) catch |err| {
                var cause: [512]u8 = undefined;
                log.warn("[spec-warmup] failed ({s}: {s}); the first round at each width pays its kernel compile inside the round.\n", .{ @errorName(err), latchedCause(&cause) });
            };
        }
        arch_load.warmSpec(sch.io, params.config, xfm_ptr, aux_head, kv_quant_config);
    }
    settleLoadLatch(had_error_at_start);

    // A MERGED drafter has no `--drafter` to echo, so the reported path comes
    // from what was actually resolved — `drafter_loaded` and `drafter_path`
    // must not disagree about the same sidecar.
    if (drafter_path_owned.len == 0 and drafter_dir.len > 0 and
        dflash_ptr != null)
    {
        drafter_path_owned = try sch.allocator.dupe(u8, drafter_dir);
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
    entry.handle = arch_handle.handleOf(xfm_ptr);
    entry.vision_encoder = vision_ptr;
    entry.dflash = dflash_ptr;
    entry.drafter_block_size = sch.drafter_block_size;
    entry.mtp = generate_mod.MtpHeadRef.fromModel(xfm_ptr, mtp_enabled, aux_head);
    // Resolve the auto (0) cap here so every downstream reader of
    // `lm.mtp_depth` (server log lines, slot params) sees the real value.
    entry.mtp_depth = generate_mod.Generator.resolveMtpDepthCapForProfile(params.mtp_depth, mtp_cost_profile);
    xfm_ptr.mtp_depth_free = generate_mod.Generator.mtpDepthCapFree(params.mtp_depth);
    if (aux_head) |h| if (arch_load.headDepthCap(params.config, h)) |cap| {
        entry.mtp_depth = @min(entry.mtp_depth, cap);
        xfm_ptr.mtp_depth_free = @min(xfm_ptr.mtp_depth_free, cap);
    };
    if (entry.mtp != null and !arch_config.fact(params.config, "no_mtp_head")) log.info("[mtp] depth range {d}..{d} ({s})\n", .{
        depth_bounds.floorFor(depth_bounds.active, entry.mtp_depth),
        entry.mtp_depth,
        if (depth_bounds.active.explicit()) "--mtp-min-depth/--mtp-max-depth" else "default",
    });
    entry.drafter_path = drafter_path_owned;
    drafter_path_owned = &[_]u8{}; // disarm the errdefer
    // Transfer ownership of the heap-allocated CPU state from `params` to
    // the entry. The caller (main.zig) MUST NOT free these — `LoadedModel.deinit`
    // walks them in the same `*X` pointer form they came in.
    entry.releaseRetainedCpuState();
    entry.config = params.config;
    entry.tokenizer = params.tok;
    entry.chat_config = params.chat_config;
    try attachPrefixCache(sch.allocator, sch.io, entry, .{
        .capacity = params.prefix_cache_capacity,
        .ram_enabled = params.prefix_cache_ram_enabled,
        .mem_bytes = params.prefix_cache_mem_bytes,
        .mem_resolver = params.prefix_cache_mem_resolver,
        .disk_resolver = params.prefix_cache_disk_resolver,
        .ssm_checkpoint_stride = params.ssm_checkpoint_stride,
        .ssm_checkpoint_max = params.ssm_checkpoint_max,
    }, params.prefix_cache_disk_bytes, .load);
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
    load_clock.end();
    if (params.config.dsv41 != null) {
        var line_buf: [512]u8 = undefined;
        log.info("{s}\n", .{load_clock.line(&line_buf, &load_clock.load)});
    }
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
        if (victim.config) |c| arch_serve.unloadModel(c);
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
    sch.registry.mutex.lockUncancelable(sch.io);
    const idle = sch.registry.countLoadedLocked() == 0;
    sch.registry.mutex.unlock(sch.io);
    if (idle) gpu_mem.refreshIdleGpuFloor();

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
    sch.registry.markLoadFailedLocked(req.entry, @errorName(err));
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
    scheduler_mod.flushLiveState(entry, "unload");
    entry.unloadResident();
    if (entry.config) |c| arch_serve.unloadModel(c);
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
/// for real routing; `_ARMS=0,1,1,0` runs a pass per arm, 0 on the reference EXL3 NAX GEMM body, or with
/// `_LEVER=decoded` 0 on the trunk's sorted GEMM instead of its decoded path.
fn prefillUbench(alloc: std.mem.Allocator, xfm: *Trunk, cfg: *const ModelConfig, tok: *Tokenizer) void {
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
    defer @import("sushi_exl3").trunk_gemm.enabled = true;
    const decoded_lever = if (std.c.getenv("SUSHI_PREFILL_UBENCH_LEVER")) |l| std.mem.eql(u8, std.mem.sliceTo(l, 0), "decoded") else false;
    const had_error = mlx.errorPending();
    defer xfm.resetCache() catch {};
    const shape = [_]c_int{ 1, @intCast(rows) };
    const times = alloc.alloc(f64, n) catch return;
    defer alloc.free(times);
    var first_ms: f64 = 0;
    for (arms[0..n_arms]) |arm| {
        if (decoded_lever)
            @import("sushi_exl3").trunk_gemm.enabled = if (arm) |a| a else true
        else
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
            const ms = @as(f64, @floatFromInt(sw.read())) / 1.0e6;
            if (iter > 0) times[iter - 1] = ms else first_ms = ms;
        }
        std.mem.sortUnstable(f64, times, {}, std.sort.asc(f64));
        log.info("[prefill-ubench] arm={s} rows={d} n={d} median {d:.2} ms min {d:.2} max {d:.2} ({d:.1} tok/s), untimed first {d:.2} ms\n", .{
            if (arm) |a| (if (a) "1" else "0") else "shipped", rows, n, times[n / 2], times[0], times[n - 1],
            @as(f64, @floatFromInt(rows)) / times[n / 2] * 1000.0, first_ms,
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

test "a cold load refused for memory or a setting returns the entry to unloaded, any other failure keeps its name" {
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

    for ([_]anyerror{ error.InsufficientMemory, error.OutOfMemory, error.SsdBudgetBelowResident, error.ExpertCacheDoesNotFit }) |memory| {
        failLoadRequest(&sch, &req, memory, false);
        try testing.expectEqual(model_registry_mod.LoadState.unloaded, entry.state);
    }
    failLoadRequest(&sch, &req, error.LoadFailed, false);
    try testing.expectEqual(model_registry_mod.LoadState.error_state, entry.state);
}

// ── Prefix-cache attach ──

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// A one-shard model dir and a private cache dir (`SUSHI_PREFIX_CACHE_DIR`) under a tmp dir, and a stub entry.
const AttachFixture = struct {
    tmp: std.testing.TmpDir,
    reg: *ModelRegistry,
    entry: *LoadedModel,
    config: ModelConfig = .{},
    cache_dir: [:0]u8,

    fn init(self: *AttachFixture) !void {
        const io = testing.io;
        const a = testing.allocator;
        self.tmp = testing.tmpDir(.{});
        errdefer self.tmp.cleanup();
        try self.tmp.dir.createDirPath(io, "m");
        try self.tmp.dir.createDirPath(io, "c");
        try self.tmp.dir.writeFile(io, .{ .sub_path = "m/w.safetensors", .data = "x" });
        var buf: [512]u8 = undefined;
        const root = buf[0..try self.tmp.dir.realPath(io, &buf)];
        const model_dir = try std.fmt.allocPrint(a, "{s}/m", .{root});
        defer a.free(model_dir);
        self.cache_dir = try std.fmt.allocPrintSentinel(a, "{s}/c", .{root}, 0);
        errdefer a.free(self.cache_dir);
        self.reg = try ModelRegistry.init(a, io, null, 3, 0, null);
        errdefer self.reg.deinit();
        self.entry = try self.reg.registerStub("m", model_dir, 1);
        self.config = .{};
        self.entry.config = &self.config;
        _ = setenv("SUSHI_PREFIX_CACHE_DIR", self.cache_dir.ptr, 1);
    }

    fn deinit(self: *AttachFixture) void {
        _ = unsetenv("SUSHI_PREFIX_CACHE_DIR");
        if (self.entry.prefix_cache) |*hc| hc.deinit();
        self.entry.prefix_cache = null;
        self.entry.config = null;
        self.reg.deinit();
        testing.allocator.free(self.cache_dir);
        self.tmp.cleanup();
    }

    fn settings(ram: bool) CacheAttach {
        return .{ .capacity = 4, .ram_enabled = ram, .mem_bytes = 0, .mem_resolver = null, .disk_resolver = null, .ssm_checkpoint_stride = 0, .ssm_checkpoint_max = 16 };
    }
};

test "attachPrefixCache: SSD-only builds a tier-only cache with the tier's policy and a started writer" {
    var f: AttachFixture = undefined;
    try f.init();
    defer f.deinit();
    try attachPrefixCache(testing.allocator, testing.io, f.entry, AttachFixture.settings(false), 1 << 30, .load);

    try testing.expect(!f.config.prefix_cache_disk_declined);
    const hc = &f.entry.prefix_cache.?;
    const disk = &hc.disk.?;
    try testing.expect(!hc.ram_enabled);
    try testing.expectEqual(@as(u64, 1 << 30), disk.max_bytes);
    try testing.expectEqual(@as(u64, 1 << 30), disk.operator_cap);
    try testing.expectEqual(prefix_cache_mod.ssdFirstActive(&f.config, true, false), hc.ssd_first);
    try testing.expectEqual(hc.ssd_first, disk.ssd_first);
    try testing.expect(disk.writer != null);
    try testing.expectEqual(server_state.disk_writer_permit_bytes, disk.writer_permit_bytes);
    try testing.expectEqual(@as(usize, kv_disk_cache.SSM_DISK_MAX_PER_ENTRY), disk.ssm_max_per_entry);
    try testing.expectEqual(trunk.ThinPolicy.min_span_recency, disk.cp_thin);
    try testing.expectEqual(trunk.ThinPolicy.min_span, hc.cp_thin);
    try testing.expectEqual(@as(u32, 16), f.entry.ssm_checkpoint_max);
    try testing.expectEqual(@as(u32, 16), hc.ssm_checkpoint_max);
}

test "attachPrefixCache: RAM beside the tier keeps the RAM cache and the legacy checkpoint cap" {
    var f: AttachFixture = undefined;
    try f.init();
    defer f.deinit();
    try attachPrefixCache(testing.allocator, testing.io, f.entry, AttachFixture.settings(true), 1 << 30, .load);

    try testing.expect(!f.config.prefix_cache_disk_declined);
    const hc = &f.entry.prefix_cache.?;
    try testing.expect(hc.ram_enabled);
    try testing.expectEqual(@as(usize, kv_disk_cache.SSM_DISK_MAX_PER_ENTRY_LEGACY), hc.disk.?.ssm_max_per_entry);
    try testing.expectEqual(trunk.ThinPolicy.oldest, hc.disk.?.cp_thin);
    try testing.expect(!hc.ssd_first);
}

test "attachPrefixCache: a tier the resolver refuses is recorded as declined, and SSD-only then builds no cache" {
    const Refuse = struct {
        fn resolve(_: *model_mod.ModelConfig, _: []const u8, _: u64, _: bool) ?disk_placement.DefaultDiskBudget {
            return null;
        }
    };
    var f: AttachFixture = undefined;
    try f.init();
    defer f.deinit();
    var s = AttachFixture.settings(false);
    s.disk_resolver = Refuse.resolve;
    try attachPrefixCache(testing.allocator, testing.io, f.entry, s, 1 << 30, .load);
    try testing.expect(f.config.prefix_cache_disk_declined);
    try testing.expect(f.entry.prefix_cache == null);

    s.ram_enabled = true;
    try attachPrefixCache(testing.allocator, testing.io, f.entry, s, 1 << 30, .load);
    try testing.expect(f.config.prefix_cache_disk_declined);
    try testing.expect(f.entry.prefix_cache.?.ram_enabled);
    try testing.expect(f.entry.prefix_cache.?.disk == null);
    try testing.expect(!f.entry.prefix_cache.?.ssd_first);
}

test "attachPrefixCache: an SSD ask of zero declines nothing and builds no SSD-only cache" {
    var f: AttachFixture = undefined;
    try f.init();
    defer f.deinit();
    try attachPrefixCache(testing.allocator, testing.io, f.entry, AttachFixture.settings(false), 0, .load);
    try testing.expect(!f.config.prefix_cache_disk_declined);
    try testing.expect(f.entry.prefix_cache == null);
}

const RoomProbe = struct {
    var free: u64 = 0;
    fn probe(_: []const u8) ?disk_placement.VolumeSpace {
        return .{ .free = free, .total = 1 << 40 };
    }
    fn refuse(_: *model_mod.ModelConfig, _: []const u8, _: u64, _: bool) ?disk_placement.DefaultDiskBudget {
        return null;
    }
};

/// A load that declined its SSD tier (the resolver refused), with the ask `source` and a volume probe of `free`.
fn declinedFixture(f: *AttachFixture, ram: bool, source: @import("settings_resolve.zig").Source, free: u64) !CacheAttach {
    server_state.setDiskAsk(1 << 30, source);
    var s = AttachFixture.settings(ram);
    s.disk_resolver = RoomProbe.refuse;
    try attachPrefixCache(testing.allocator, testing.io, f.entry, s, 1 << 30, .load);
    try testing.expect(f.config.prefix_cache_disk_declined);
    RoomProbe.free = free;
    s.disk_resolver = null;
    s.space_probe = RoomProbe.probe;
    return s;
}

test "rearmDiskTier: a declined SSD tier stays off without room above the reserve and arms when it returns" {
    const saved = bills.diskAsk();
    defer server_state.setDiskAsk(saved.bytes, saved.source);
    var f: AttachFixture = undefined;
    try f.init();
    defer f.deinit();
    var s = try declinedFixture(&f, false, .default, disk_placement.DISK_FREE_RESERVE);

    try testing.expect(!rearmDiskTier(testing.allocator, testing.io, f.entry, s, "test"));
    try testing.expect(f.config.prefix_cache_disk_declined);
    try testing.expect(f.entry.prefix_cache == null);

    RoomProbe.free = disk_placement.DISK_FREE_RESERVE + (8 << 30);
    s.disk_resolver = RoomProbe.refuse;
    try testing.expect(!rearmDiskTier(testing.allocator, testing.io, f.entry, s, "test"));
    try testing.expect(f.config.prefix_cache_disk_declined);
    try testing.expect(f.entry.prefix_cache == null);

    s.disk_resolver = null;
    try testing.expect(rearmDiskTier(testing.allocator, testing.io, f.entry, s, "test"));
    try testing.expect(!f.config.prefix_cache_disk_declined);
    const disk = &f.entry.prefix_cache.?.disk.?;
    try testing.expectEqual(@as(u64, 1 << 30), disk.max_bytes);
    try testing.expect(disk.writer != null);
    try testing.expectEqual(f.entry.prefix_cache.?.ssd_first, disk.ssd_first);
    try testing.expect(!rearmDiskTier(testing.allocator, testing.io, f.entry, s, "test"));
}

test "rearmDiskTier: an SSD size named in settings arms a declined tier on a full volume, a size of zero does not" {
    const saved = bills.diskAsk();
    defer server_state.setDiskAsk(saved.bytes, saved.source);
    var f: AttachFixture = undefined;
    try f.init();
    defer f.deinit();
    const s = try declinedFixture(&f, false, .default, 0);
    try testing.expect(!rearmDiskTier(testing.allocator, testing.io, f.entry, s, "test"));

    server_state.setDiskAsk(0, .model_settings);
    try testing.expect(!rearmDiskTier(testing.allocator, testing.io, f.entry, s, "test"));
    try testing.expect(f.config.prefix_cache_disk_declined);

    server_state.setDiskAsk(2 << 30, .model_settings);
    try testing.expect(rearmDiskTier(testing.allocator, testing.io, f.entry, s, "test"));
    try testing.expect(!f.config.prefix_cache_disk_declined);
    try testing.expectEqual(@as(u64, 2 << 30), f.entry.prefix_cache.?.disk.?.max_bytes);
}

test "rearmDiskTier: a RAM cache the load built keeps its object and entries when the tier joins it" {
    const saved = bills.diskAsk();
    defer server_state.setDiskAsk(saved.bytes, saved.source);
    var f: AttachFixture = undefined;
    try f.init();
    defer f.deinit();
    const s = try declinedFixture(&f, true, .default, disk_placement.DISK_FREE_RESERVE + (8 << 30));
    const before = &f.entry.prefix_cache.?;
    try testing.expect(before.disk == null);

    try testing.expect(rearmDiskTier(testing.allocator, testing.io, f.entry, s, "test"));
    try testing.expectEqual(before, &f.entry.prefix_cache.?);
    try testing.expect(before.ram_enabled);
    try testing.expect(before.disk != null);
    try testing.expectEqual(prefix_cache_mod.ssdFirstActive(&f.config, true, true), before.ssd_first);
}

// ── Bill pins ──
// One test per load-time bill function, on the committed fixture config of every served pack.


fn readLoadLogForTest(path: [*:0]const u8, buf: []u8) []const u8 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return buf[0..0];
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, buf.ptr, buf.len);
    return buf[0..if (n > 0) @intCast(n) else 0];
}

test "an MLX error latched during a load is named in the log and does not reach the first request" {
    const dir = "/tmp/sushi-load-latch";
    const path = dir ++ "/s.log";
    _ = std.c.mkdir(dir, @as(std.c.mode_t, 0o755));
    _ = std.c.unlink(path);
    try log.openFile(path, 0);
    defer {
        log.closeFile();
        _ = std.c.unlink(path);
    }
    mlx.installErrorHandler();
    var scratch: [512]u8 = undefined;
    _ = mlx.takeError(&scratch);

    mlx.latchErrorForTest("[array] expected a non-empty mlx_array");
    try testing.expect(mlx.errorPending());
    settleLoadLatch(false);
    try testing.expect(!mlx.errorPending());
    try mlx.checkError();
    mlx.latchErrorForTest("[decode] another request's failure");
    settleLoadLatch(true);
    try testing.expect(mlx.errorPending());
    _ = mlx.takeError(&scratch);

    log.closeFile();
    var buf: [2048]u8 = undefined;
    const text = readLoadLogForTest(path, &buf);
    try testing.expect(std.mem.indexOf(u8, text, "expected a non-empty mlx_array") != null);
}

test "a saved ctx_size above the window the loaded config has is refused by name; a flag, an auto context and a fitting value are not" {
    var buf: [256]u8 = undefined;
    var c = ModelConfig{ .model_type = "qwen4_exp", .max_position_embeddings = 262144, .ctx_override = 1048576 };
    const line = contextWindowRefusal(&buf, &c, 0).?;
    try testing.expectEqualStrings("[model-settings] ctx_size 1048576 is above the 262144-token window this model has without yarn_factor; lower ctx_size or set yarn_factor", line);

    c.rope_yarn = true;
    c.yarn_factor = 2;
    c.yarn_orig_max_pos = 262144;
    c.max_position_embeddings = 524288;
    c.yarn_factor_override = 2;
    try testing.expectEqualStrings("[model-settings] ctx_size 1048576 is above the 524288-token window this model has with yarn_factor 2; lower ctx_size or yarn_factor", contextWindowRefusal(&buf, &c, 0).?);

    c.ctx_override = 524288;
    try testing.expect(contextWindowRefusal(&buf, &c, 0) == null);
    c.ctx_override = 0;
    try testing.expect(contextWindowRefusal(&buf, &c, 0) == null);
    // An explicit launch flag outranks the file, so the file's value is not what loads.
    c.ctx_override = 1048576;
    try testing.expect(contextWindowRefusal(&buf, &c, 100000) == null);
}

test "an explicit --ctx-size above the window is refused by name, under the yarn_factor in force; 0 and a fitting value are not" {
    var buf: [256]u8 = undefined;
    var c = ModelConfig{ .model_type = "qwen4_exp", .max_position_embeddings = 262144 };
    try testing.expectEqualStrings("--ctx-size 1048576 is above the 262144-token window this model has without yarn_factor; lower --ctx-size or set yarn_factor", contextWindowRefusal(&buf, &c, 1048576).?);
    try testing.expect(contextWindowRefusal(&buf, &c, 262144) == null);
    try testing.expect(contextWindowRefusal(&buf, &c, 0) == null);

    c.rope_yarn = true;
    c.yarn_factor = 2;
    c.yarn_orig_max_pos = 262144;
    c.max_position_embeddings = 524288;
    c.yarn_factor_override = 2;
    try testing.expectEqualStrings("--ctx-size 1048576 is above the 524288-token window this model has with yarn_factor 2; lower --ctx-size or yarn_factor", contextWindowRefusal(&buf, &c, 1048576).?);
    try testing.expect(contextWindowRefusal(&buf, &c, 524288) == null);

    // The flag outranks the file: a fitting flag loads whatever the file holds, a too-large one is the flag's refusal.
    c.ctx_override = 1048576;
    try testing.expect(contextWindowRefusal(&buf, &c, 100000) == null);
    try testing.expect(std.mem.startsWith(u8, contextWindowRefusal(&buf, &c, 2000000).?, "--ctx-size 2000000 "));
}

test "a launch flag or saved kv_quant the arch cannot act on is refused with its line, the same answer for the pre-check and the load" {
    if (@import("arch/registry.zig").ofModelType("deepseek_v41") == null) return error.SkipZigTest;
    var buf: [384]u8 = undefined;
    const dsv = ModelConfig{ .model_type = "deepseek_v41" };
    const r = launchRefusal(&buf, &dsv, trunk.KVQuantConfig.affine(8)).?;
    try testing.expect(r.err == error.LaunchFlagUnsupported);
    try testing.expect(std.mem.startsWith(u8, r.line, "--kv-quant has no effect on "));
    try testing.expect(launchRefusal(&buf, &dsv, null) == null);
}
