const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const qwen4_qsa_select = @import("qwen4/qwen4_qsa_select.zig");
const qwen4_qsa_gather = @import("qwen4/qwen4_qsa_gather.zig");
const qwen4_qsa_attn = @import("qwen4/qwen4_qsa_attn.zig");
const tokenizer_mod = @import("tokenizer.zig");
const generate_mod = @import("generate.zig");
const mtp_mod = @import("mtp.zig");
const mimo_mtp = @import("mimo2/mimo_mtp.zig");
const mtp_acceptance_mod = @import("mtp_acceptance.zig");
const chat_mod = @import("chat.zig");
const stop_seq_mod = @import("stop_sequences.zig");
const rp_mod = @import("reasoning_protocol.zig");
const token_mask = @import("token_mask.zig");
const expert_stream_mod = @import("expert_stream.zig");
const fp8_block = @import("fp8_block.zig");
const model_mod = @import("model.zig");
const dsv41_config = @import("dsv41/dsv41_config.zig");
const qwen_vision = @import("qwen4/qwen_vision.zig");
const vision_common = @import("vision_common.zig");
const mimo_vision = @import("mimo2/mimo_vision.zig");
const glm5_vision = @import("glm5/glm5_vision.zig");
const dsv41_vision = @import("dsv41/dsv41_vision.zig");
const glm5_prefix = @import("glm5/glm5_prefix.zig");
const dsv41_prefix = @import("dsv41/dsv41_prefix.zig");
const dsv41_borrow = @import("dsv41/dsv41_borrow.zig");
const mrope_mod = @import("mrope.zig");
const vision_mod = @import("vision.zig");
const log = @import("log.zig");
const responses_mod = @import("responses.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const kv_disk_writer = @import("kv_disk_writer.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const tokenize_cache_mod = @import("tokenize_cache.zig");
const scheduler_mod = @import("scheduler.zig");
const model_registry_mod = @import("model_registry.zig");
const model_discovery = @import("model_discovery.zig");
const model_settings = @import("model_settings.zig");
const stb = @import("stb");
const webp = @import("webp");
const metrics = @import("status.zig");
const instr = @import("metrics.zig");
const qwen4_mod = @import("qwen4/qwen4_exp.zig");
const update_mod = @import("update.zig");

const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const Generator = generate_mod.Generator;
const VisionEncoder = vision_mod.VisionEncoder;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;
const anthropic = @import("anthropic.zig");
pub const buildAnthropicStreamError = anthropic.buildAnthropicStreamError;
pub const isAnthropicRoute = anthropic.isAnthropicRoute;
const anthropicImageUrl = anthropic.anthropicImageUrl;
const readAnthropicMessages = anthropic.readAnthropicMessages;
pub const sendAnthropicError = anthropic.sendAnthropicError;
pub const sendAnthropicEvent = anthropic.sendAnthropicEvent;
const handleAnthropicMessages = anthropic.handleAnthropicMessages;
const handleAnthropicCountTokens = anthropic.handleAnthropicCountTokens;
const completions = @import("completions.zig");
pub const completionShowsToken = completions.completionShowsToken;
pub const completionText = completions.completionText;
pub const handleCompletions = completions.handleCompletions;
const responses_api = @import("responses_api.zig");
pub const getOrInitResponseStore = responses_api.getOrInitResponseStore;
const deinitGlobalResponseStore = responses_api.deinitGlobalResponseStore;
pub const buildResponsesStreamError = responses_api.buildResponsesStreamError;
pub const splitConstrainedResponse = responses_api.splitConstrainedResponse;
pub const sendResponsesEvent = responses_api.sendResponsesEvent;
const handleResponsesGet = responses_api.handleResponsesGet;
const handleResponsesDelete = responses_api.handleResponsesDelete;
const handleResponsesCompact = responses_api.handleResponsesCompact;
pub const handleResponses = responses_api.handleResponses;
const generation_stream = @import("generation_stream.zig");
const PREFILL_NOFIT_MSG = generation_stream.PREFILL_NOFIT_MSG;
pub const EMPTY_PROMPT_MSG = generation_stream.EMPTY_PROMPT_MSG;
const GenErrorWire = generation_stream.GenErrorWire;
pub const mapGenerationError = generation_stream.mapGenerationError;
const sendGenerationError = generation_stream.sendGenerationError;
const visibleToolPreamble = generation_stream.visibleToolPreamble;
const nonStreamingViaScheduler = generation_stream.nonStreamingViaScheduler;
const handleNonStreamingGeneration = generation_stream.handleNonStreamingGeneration;
pub const StreamMode = generation_stream.StreamMode;
const StreamingTokenStream = generation_stream.StreamingTokenStream;
pub const requestSpecModes = generation_stream.requestSpecModes;
const noteMtpRefused = generation_stream.noteMtpRefused;
pub const pickStreamMode = generation_stream.pickStreamMode;
const handleStreamingGeneration = generation_stream.handleStreamingGeneration;
pub const loadRefusalFor = generation_stream.loadRefusalFor;
const sendNamedLoadRefusal = generation_stream.sendNamedLoadRefusal;
const sendLoadFailedResponse = generation_stream.sendLoadFailedResponse;
const contextOverflowMessage = generation_stream.contextOverflowMessage;
const formatTimingsObject = generation_stream.formatTimingsObject;
const formatPerfBracket = generation_stream.formatPerfBracket;
pub const contentTokenRange = generation_stream.contentTokenRange;
pub const sendStreamIdle = generation_stream.sendStreamIdle;
const beatStreamKeepalive = generation_stream.beatStreamKeepalive;
const http_conn = @import("http_conn.zig");
const startListener = http_conn.startListener;
const WsBridge = http_conn.WsBridge;
pub const Conn = http_conn.Conn;
const closeEmbeddedListener = http_conn.closeEmbeddedListener;
const routeRequestModel = http_conn.routeRequestModel;
const crossOriginRefused = http_conn.crossOriginRefused;
const routeExists = http_conn.routeExists;
const ConnThreadArgs = http_conn.ConnThreadArgs;
const readRequestWithin = http_conn.readRequestWithin;
const isStatusRoute = http_conn.isStatusRoute;
const isTextGenRoute = http_conn.isTextGenRoute;
const sendSseHeaders = http_conn.sendSseHeaders;
const apiKeyGateApplies = http_conn.apiKeyGateApplies;
const findHeaderValueCI = http_conn.findHeaderValueCI;
const apiKeyAuthorized = http_conn.apiKeyAuthorized;
const sendUnauthorized = http_conn.sendUnauthorized;
const serveOneForTest = http_conn.serveOneForTest;
const handleResponsesWebSocket = http_conn.handleResponsesWebSocket;
const request_media = @import("request_media.zig");
const MediaFault = request_media.MediaFault;
const responsesMediaRefusal = request_media.responsesMediaRefusal;
const decodeImageUrlInto = request_media.decodeImageUrlInto;
const openAiImageUrl = request_media.openAiImageUrl;
const flattenMedia = request_media.flattenMedia;
pub const visionEncodeBill = request_media.visionEncodeBill;
pub const largestImageEncodeBytes = request_media.largestImageEncodeBytes;
const visionEncodeBillBytes = request_media.visionEncodeBillBytes;
pub const visionEncodeAvailable = request_media.visionEncodeAvailable;
const PreparedMedia = request_media.PreparedMedia;
const placeRequestMedia = request_media.placeRequestMedia;
const prepareRequestMedia = request_media.prepareRequestMedia;
const visionPreprocFromConfig = request_media.visionPreprocFromConfig;
pub const decodeEncodedImage = request_media.decodeEncodedImage;
const RequestMedia = request_media.RequestMedia;
const request_policy = @import("request_policy.zig");
const resolveSamplingDefault = request_policy.resolveSamplingDefault;
const resolveThinkPenalty = request_policy.resolveThinkPenalty;
const armThinkPenalty = request_policy.armThinkPenalty;
const omittedMaxTokensDefault = request_policy.omittedMaxTokensDefault;
const launchMaxTokensDefault = request_policy.launchMaxTokensDefault;
const resolveRequestMaxTokens = request_policy.resolveRequestMaxTokens;
const maxTokensOrigin = request_policy.maxTokensOrigin;
const describeMaxTokens = request_policy.describeMaxTokens;
const clampMaxTokens = request_policy.clampMaxTokens;
pub const chatTemplateSupportsThinking = request_policy.chatTemplateSupportsThinking;
pub const optSamplingRecJson = request_policy.optSamplingRecJson;
const effortWordOnly = request_policy.effortWordOnly;
const parseReasoningEffort = request_policy.parseReasoningEffort;
const reasoningEffortFromWord = request_policy.reasoningEffortFromWord;
const EffortNotice = request_policy.EffortNotice;
const effortModelName = request_policy.effortModelName;
const clearEffortNotice = request_policy.clearEffortNotice;
const holdEffortNotice = request_policy.holdEffortNotice;
const writeEffortNoticeHeader = request_policy.writeEffortNoticeHeader;
const withEffortNoticeJson = request_policy.withEffortNoticeJson;
pub const effortRefusal = request_policy.effortRefusal;
pub const reasoningEffortsJson = request_policy.reasoningEffortsJson;
const effortRefusalFor = request_policy.effortRefusalFor;
const responsesEnableThinking = request_policy.responsesEnableThinking;
const checkThinkingSupport = request_policy.checkThinkingSupport;
pub const resolveEnableThinking = request_policy.resolveEnableThinking;
const schemaMasksThinking = request_policy.schemaMasksThinking;
const resolvePreserveThinking = request_policy.resolvePreserveThinking;
const parseRepeatPenalty = request_policy.parseRepeatPenalty;
const parseAdditivePenalty = request_policy.parseAdditivePenalty;
const models_api = @import("models_api.zig");
pub const renderModelEntry = models_api.renderModelEntry;
const handleModels = models_api.handleModels;
const handleModelsRescan = models_api.handleModelsRescan;
const settingsPropsJson = models_api.settingsPropsJson;
const handleProps = models_api.handleProps;
const handlePropsNoModel = models_api.handlePropsNoModel;
const logprobs = @import("logprobs.zig");
const StreamLogprobs = logprobs.StreamLogprobs;
const formatCompletionsLogprobs = logprobs.formatCompletionsLogprobs;
const formatLogprobsObject = logprobs.formatLogprobsObject;
/// Global flag set by signal handler for graceful shutdown.
pub var shutdown_requested = std.atomic.Value(bool).init(false);
/// Number of live per-connection threads. Incremented before each spawn,
/// decremented when the thread's handler returns. On shutdown, `serve` waits
/// for this to reach 0 before returning (which runs `scheduler.deinit`) — a
/// detached conn thread still in `Scheduler.complete` would otherwise race
/// deinit's teardown of the slot queues into a use-after-free (SIGSEGV).
var active_conn_threads = std.atomic.Value(u32).init(0);
pub var completion_id_counter = std.atomic.Value(i64).init(0);
/// Optional global API key (`--api-key`). When set, every NON-LOOPBACK request
/// (i.e. from another machine over the network) except the `/health` probe and
/// CORS preflight requires the key — the OpenAI/Anthropic APIs AND the
/// metrics feed. Loopback (127.0.0.0/8, ::1) is TRUSTED and
/// exempt, so the local app + a local browser keep working with no credentials;
/// the key protects the surface that actually matters — network exposure
/// (`--host 0.0.0.0`). Accepted as `Authorization: Bearer <key>`, `x-api-key:
/// <key>` (Anthropic), HTTP Basic (browser pages — the key is the password), or
/// an `?api_key=` / `?key=` query param. null ⇒ fully open (default, unchanged).
pub var g_api_key: ?[]const u8 = null;

/// `--api-key-strict`: drop the loopback exemption, so the key is required
/// from 127.0.0.1 too (`/health` and CORS preflight stay open). For hosts
/// where localhost is NOT one trust domain — an embedding application that
/// generates a fresh key per start wants "only the process holding the key
/// can drive inference", and loopback exemption made any local process (and
/// any browser page via a simple no-preflight POST) a key holder. No effect
/// unless `--api-key` is set.
pub var g_api_key_strict: bool = false;

/// Tool-call ARGUMENT auto-correct. When true (default), parsed tool-call
/// arguments are coerced to the types the request's tool schema declares
/// (`chat.coerceToolArgsToSchema` — e.g. Python's `False` → JSON `false`, an
/// `edits` array shipped as a string → a real array). `--no-tool-autocorrect`
/// turns it OFF: arguments pass through exactly as the model emitted them (still
/// valid JSON — the parse-repair + safety net always run; this gates ONLY the
/// schema-type coercion, the opinionated layer). Off means a weak model's
/// mistyped value reaches the client verbatim, which strict clients reject.
pub var g_tool_autocorrect: bool = true;

/// `--parent-pid`: a host runs this engine, so the host (not `/v1/update`) updates it.
pub var host_managed: bool = false;
/// The bind `serve` listens on; `/v1/update` checks the page's Origin against it.
pub var listen_host: []const u8 = default_host;
pub var listen_port: u16 = default_port;

pub const default_host: []const u8 = "127.0.0.1";
pub const default_port: u16 = 12345;

pub const BindConfig = struct {
    host: []const u8,
    port: u16,
};

pub fn resolveBind(host_flag: ?[]const u8, port_flag: ?u16) BindConfig {
    return .{
        .host = host_flag orelse default_host,
        .port = port_flag orelse default_port,
    };
}

pub fn bindAddress(host: []const u8, port: u16) error{InvalidHost}!std.Io.net.IpAddress {
    var ip4_bytes: [4]u8 = undefined;
    if (std.mem.eql(u8, host, "localhost")) {
        ip4_bytes = .{ 127, 0, 0, 1 };
    } else if (std.mem.eql(u8, host, "0.0.0.0")) {
        ip4_bytes = .{ 0, 0, 0, 0 };
    } else {
        // Parse dotted-decimal IP
        var parts = std.mem.splitScalar(u8, host, '.');
        var idx: usize = 0;
        while (parts.next()) |part| {
            if (idx >= 4) return error.InvalidHost;
            ip4_bytes[idx] = std.fmt.parseInt(u8, part, 10) catch return error.InvalidHost;
            idx += 1;
        }
        if (idx != 4) return error.InvalidHost;
    }
    return .{ .ip4 = .{ .bytes = ip4_bytes, .port = port } };
}

pub fn ensurePortFree(io: std.Io, host: []const u8, port: u16) (error{InvalidHost} || std.Io.net.IpAddress.ListenError)!void {
    const addr = try bindAddress(host, port);
    if (addr.connect(io, .{ .mode = .stream })) |s| {
        s.close(io);
        return error.AddressInUse;
    } else |_| {}
    var probe = try startListener(host, port);
    probe.deinit(io);
}

pub fn startupRefusal(err: anyerror, host: []const u8, port: u16, buf: []u8) ?[]const u8 {
    switch (err) {
        error.AddressInUse => return std.fmt.bufPrint(buf, "port {d} is already in use", .{port}) catch unreachable,
        error.InvalidHost => return std.fmt.bufPrint(buf, "invalid --host '{s}'", .{host}) catch "invalid --host",
        else => return null,
    }
}

const io_util = @import("io_util.zig");
const ws_mod = @import("ws.zig");
const build_options = @import("build_options");
const glm5_generate = @import("glm5/glm5_generate.zig");
const mtp_plan = @import("mtp_plan.zig");
const prefill_plan = @import("prefill_plan.zig");
const sampler = @import("sampler.zig");
const think_penalty = @import("think_penalty.zig");
const nowSecs = io_util.nowSecs;
const nowMs = io_util.nowMs;
const nowMsMonotonic = io_util.nowMsMonotonic;
const Stopwatch = io_util.Stopwatch;

const nextCompletionId = io_util.nextCompletionId;

fn signalHandler(_: std.posix.SIG) callconv(.c) void {
    shutdown_requested.store(true, .release);
}


// Plan 05: drafter state moved to `LoadedModel` (per-model `drafter`,
// `drafter_block_size`, `drafter_path`). The previous module-level
// `default_drafter` / `default_enable_drafter` / `default_draft_block_size`
// / `global_drafter_path` singletons were removed; handlers now read
// these fields off the request's resolved `*LoadedModel` (`lm.drafter`,
// etc.). `default_enable_drafter` semantics (`drafter != null and !isMoe()`)
// is re-computed per-request as `lm.drafter != null and !config.isMoe()`.

pub fn requestShutdown() void {
    shutdown_requested.store(true, .release);
}

/// Server-level configuration. Single source of truth for all process-wide
/// defaults that handlers might consult. Populated once by `serve()` from
/// its CLI args; read-only afterwards (no synchronization required, the
/// values don't change for the server's lifetime).
pub const ServerConfig = struct {
    /// Starts an embedded client only after this process owns its listening socket.
    on_ready: ?*const fn (u16) anyerror!std.Thread = null,
    /// Maximum context size (0 = unlimited). `--ctx-size N`.
    max_context_size: u32 = 0,
    /// Request timeout in seconds (0 = no timeout). `--timeout N`.
    request_timeout_sec: u32 = 300,
    /// Default reasoning budget in tokens (-1 = unlimited).
    /// `--reasoning-budget N`. Per-request body fields override.
    default_reasoning_budget: i32 = -1,
    /// Defaults for sampling fields the request OMITS, set by `--temp` /
    /// `--top-p` / `--top-k` in serve mode (the macOS app passes its Settings
    /// values so external clients like Claude Code — which send no sampling
    /// params at all — inherit them). null = flag not given; resolution falls
    /// through to the model's generation_config.json recommendation, then the
    /// hardcoded fallback. Explicit request body fields always win. See
    /// `resolveSamplingDefault`.
    default_temperature: ?f32 = null,
    default_top_p: ?f32 = null,
    default_top_k: ?u32 = null,
    /// Default `max_tokens` for a request that omits the field (`--max-tokens N` in serve mode;
    /// 0 = flag not given). Agent clients send nothing, and an omitted budget is the rest of
    /// the window, which the admission bill reserves for. Read through `launchMaxTokensDefault`.
    default_max_tokens: u32 = 0,
    /// `--mtp`: force the native MTP head ON for MoE targets too. The
    /// per-request default is otherwise `sidecar loaded and !isMoe()` (the
    /// verify-forward expert-routing caution the drafter shares), which makes
    /// a MoE MTP checkpoint unreachable from any client that doesn't send
    /// `enable_mtp:true` in the body. Per-request `enable_mtp` still wins.
    default_force_mtp: bool = false,
};

/// The per-request `enable_mtp` default, for a request that omitted the field.
/// The ONE place this policy lives — every HTTP surface calls it, so a new
/// surface can't silently ship a different default (the drafter-dispatch-hole
/// lesson: an output-equality test cannot see a spec path that never engaged).
///
/// MoE targets default OFF because the verify forward pays the expert-routing
/// penalty; `--mtp` (`default_force_mtp`) overrides that for operators who
/// measured otherwise — the 35B-A3B sidecar holds ~73% per-draft.
///
/// `native_measured`: same exemption, for an arch whose head ships inside the
/// checkpoint AND has been measured no-worse-than-serial across the context
/// ladder (`Transformer.nativeMoeMtpHeadMeasured`, which carries the bar). It
/// still needs a head LOADED — the claim is about the head, not the arch.
/// `--mtp` process-wide, or the model's own `"mtp": true` in `model-settings.json` when unflagged.
pub fn forceMtpFor(config: *const model_mod.ModelConfig) bool {
    return mtpChoiceFor(config).forced();
}

/// Will a request that omits `enable_mtp` run the head on this model? The load-time bills
/// (`sessionBytesPerToken`) price its KV whenever it does, the request-time bill per request.
fn mtpHeadDefaultOn(config: *const model_mod.ModelConfig) bool {
    if (config.isGlm5() or config.isDsv41()) return false;
    // A loaded DFlash2 drafter outranks the head: the default round never builds its cache.
    if (config.dflash_loaded) return false;
    const choice = mtpChoiceFor(config);
    if (choice.forced()) return true;
    return choice.on and model_mod.isServedArch(config.model_type) and (!config.expert_streaming or config.stream_mtp_head);
}

/// Explicit launch choice, available to memory bills before Scheduler.init returns.
var configured_mtp: ?bool = null;

/// THIS model's MTP decision and its source, the answer the load log and `/props` read.
pub fn mtpChoiceFor(config: *const model_mod.ModelConfig) model_settings.MtpChoice {
    if (global_scheduler) |sch| return scheduler_mod.mtpChoiceFor(sch.mtp_enabled, sch.mtp_explicit, config);
    const flag: ?bool = configured_mtp orelse (if (server_config.default_force_mtp) @as(?bool, true) else null);
    return scheduler_mod.mtpChoiceFor(flag orelse true, flag != null, config);
}

///
/// `served`: `model.served_model_types` default ON with a head loaded (owner policy: MTP is always
/// on for the served packs). The expert-streaming path keeps the `force` rule.
pub fn defaultEnableMtp(mtp_loaded: bool, is_moe: bool, force: bool, native_measured: bool, served: bool) bool {
    if (!mtp_loaded) return false;
    return !is_moe or force or native_measured or served;
}

pub fn defaultEnableMtpForStreaming(mtp_loaded: bool, is_moe: bool, force: bool, native_measured: bool, served: bool, expert_streaming: bool) bool {
    if (expert_streaming) return mtp_loaded and force;
    return defaultEnableMtp(mtp_loaded, is_moe, force, native_measured, served);
}

/// Does this model's MTP head carry the measured native-MoE exemption above?
/// A NAMED per-arch capability read once here, so the call sites can never disagree.
pub fn nativeMeasuredMoeHead(lm: *LoadedModel) bool {
    const x = lm.transformer orelse return false;
    return x.nativeMoeMtpHeadMeasured();
}

/// Can this model run an MTP-flagged request speculatively? Every per-surface
/// `enable_mtp` conjunct must use THIS, not a bare `lm.mtp != null`.
pub fn mtpCapable(lm: *LoadedModel) bool {
    return lm.mtp != null;
}

/// DSpark verifies each row under that row's grammar mask, so a constrained request may speculate with it.
pub fn rowMaskedDrafter(lm: *LoadedModel) bool {
    return if (lm.mtp) |head| head == .dsv41 else false;
}

/// `--max-mtp-ctx` admission: MTP is refused past the operator's context ceiling. Called once
/// per surface after tokenization; a machine limit, so it outranks an explicit `enable_mtp:true`.
pub fn admitMtpForCtx(enable_mtp: bool, prompt_tokens: usize) bool {
    if (!enable_mtp) return false;
    const max = generate_mod.max_mtp_ctx;
    if (mtp_mod.mtpCtxWithinLimit(max, prompt_tokens)) return true;
    log.info("  mtp=off (ctx {d} > --max-mtp-ctx {d})\n", .{ prompt_tokens, max });
    return false;
}

/// parseJsonFloat variant that distinguishes "omitted / wrong type" (null)
/// from an explicit value, for fields whose default comes from the
/// resolution chain above.
pub fn parseJsonFloatOpt(root: std.json.ObjectMap, key: []const u8, min: f32, max: f32) ?f32 {
    const v = root.get(key) orelse return null;
    const raw: f32 = switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => return null,
    };
    return std.math.clamp(raw, min, max);
}

/// Optional top_k body-field parse (positive integer, capped at 1000).
/// null = omitted or unusable; explicit 0 means "disable top-k".
pub fn parseJsonTopKOpt(root: std.json.ObjectMap, key: []const u8) ?u32 {
    const v = root.get(key) orelse return null;
    return switch (v) {
        .integer => |i| if (i > 0) @intCast(@min(i, 1000)) else 0,
        .float => |f| if (f > 0) @intFromFloat(@min(f, 1000)) else 0,
        else => null,
    };
}

/// Live process-wide config. Mutated by `serve()` at startup from its
/// parameters; never written from a handler after that. Public so main.zig
/// can supply defaults (notably the per-request override CLI flags don't
/// flow through `serve()` arguments).
pub var server_config: ServerConfig = .{};

pub fn getTimeoutNs() u64 {
    if (server_config.request_timeout_sec == 0) return 0;
    return @as(u64, server_config.request_timeout_sec) * std.time.ns_per_s;
}

/// Plan 01 Phase 2 — continuous-batching scheduler. Always non-null in
/// serve mode (set by `serve()`); null only before `serve()` runs. Every
/// inference request handler routes through this; the scheduler's
/// inference thread is the single mlx-call site.
pub var global_scheduler: ?*scheduler_mod.Scheduler = null;

/// The boot's explicit `--kv-quant`, published by `serve()` before `Scheduler.init` so a
/// load-time bill sees it (the scheduler does not exist yet during the load; asking it billed
/// the SSD-first session at bf16, ~8.6 GB of idle allowance lost at 786k). Null = unflagged.
pub var configured_kv_quant: ?transformer_mod.KVQuantConfig = null;

/// THIS model's KV scheme and its source, the one answer the load log, `/props` and every bill read.
pub fn kvCacheFor(config: *const model_mod.ModelConfig) transformer_mod.KvCacheChoice {
    if (global_scheduler) |sch| return kvCacheForLaunch(config, sch.kv_quant_config, sch.kv_quant_explicit);
    const launch = configured_kv_quant orelse transformer_mod.KVQuantConfig.engine_default;
    return kvCacheForLaunch(config, launch, configured_kv_quant != null);
}

/// `kvCacheFor` under a given launch choice. deepseek_v41 keeps its own compressed state; `--kv-quant` has nothing to quantize.
pub fn kvCacheForLaunch(config: *const model_mod.ModelConfig, launch: transformer_mod.KVQuantConfig, explicit: bool) transformer_mod.KvCacheChoice {
    if (config.isDsv41()) return .{ .config = .dense, .source = .default };
    return transformer_mod.KvCacheChoice.resolve(config.kv_quant_override, launch, explicit);
}

pub fn configuredKvQuantFor(config: *const model_mod.ModelConfig) transformer_mod.KVQuantConfig {
    return kvCacheFor(config).config;
}

/// The explicit context for THIS model: `--ctx-size`, else its `model-settings.json` `ctx_size`.
/// 0 = auto. Every reader of the manual context goes through here, never `server_config.max_context_size`.
pub fn manualContext(config: *const model_mod.ModelConfig) u32 {
    return model_settings.contextPick(server_config.max_context_size, config.ctx_override).value;
}

/// Plan 05 — model registry. Always non-null in serve mode (set by
/// `serve()`); handleConnection resolves `model` body fields against this
/// per request via `ensureLoaded`/`release`.
pub var global_registry: ?*ModelRegistry = null;

/// Plan 05 — extract the `"model":"..."` value from a JSON request body
/// without doing a full parse. Returns a borrowed slice into `body` (valid
/// until the body buffer is freed) or null if the field is missing or
/// malformed. Used by `handleConnection` to route each POST to the right
/// LoadedModel before invoking the handler.
///
/// Cheap (single linear scan, ~tens of nanoseconds for typical bodies);
/// the per-handler full JSON parse runs immediately after and rejects any
/// truly malformed body, so robustness here is fine.
pub fn parseModelFromBody(body: []const u8) ?[]const u8 {
    var idx: usize = 0;
    while (idx < body.len) {
        const key_start = std.mem.indexOfPos(u8, body, idx, "\"model\"") orelse return null;
        // Must be JSON object key: preceded by `{` or `,` (skipping whitespace).
        var prev = key_start;
        while (prev > 0) {
            prev -= 1;
            const c = body[prev];
            if (c == ' ' or c == '\t' or c == '\n' or c == '\r') continue;
            if (c == '{' or c == ',') break;
            // Not a top-level key — keep searching after this match.
            idx = key_start + 1;
            return parseModelFromBody(body[idx..]);
        }
        // Skip past the key + the colon.
        var pos = key_start + "\"model\"".len;
        while (pos < body.len and (body[pos] == ' ' or body[pos] == '\t')) pos += 1;
        if (pos >= body.len or body[pos] != ':') return null;
        pos += 1;
        while (pos < body.len and (body[pos] == ' ' or body[pos] == '\t' or body[pos] == '\n' or body[pos] == '\r')) pos += 1;
        if (pos >= body.len) return null;
        // Accept "string" only — null/numbers/etc fall back to default.
        if (body[pos] != '"') return null;
        pos += 1;
        const val_start = pos;
        while (pos < body.len and body[pos] != '"') {
            if (body[pos] == '\\' and pos + 1 < body.len) pos += 1; // skip escape
            pos += 1;
        }
        if (pos >= body.len) return null;
        return body[val_start..pos];
    }
    return null;
}

/// The browser chat page: one self-contained file that talks to this server's own API.
const chat_page_html = @embedFile("webui/index.html");

/// `--edit on|off`: whether a new chat on the page starts with its write tools on. Each chat's
/// Edit chip then switches them for itself.
pub var g_web_edit: bool = false;

/// The browser orchestrates calls; this local-only bridge shares the REPL's restrictions.
fn handleWebTools(allocator: std.mem.Allocator, stream: *Conn, headers: []const u8, body: []const u8) !void {
    const origin = findHeaderValueCI(headers, "origin") orelse "";
    if (!peerIsLoopback(stream) or !update_mod.isLoopbackHost(listen_host) or
        !update_mod.originMatches(origin, listen_host, listen_port))
    {
        return sendErrorResponse(allocator, stream, "403 Forbidden", "invalid_request_error", "Tools require this Mac's chat page Origin and a loopback server bind", 403);
    }
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch
        return sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Expected a JSON object", 400);
    defer parsed.deinit();
    if (parsed.value != .object) return sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Expected a JSON object", 400);
    const cwd = try std.Io.Dir.cwd().realPathFileAlloc(stream.io, ".", allocator);
    defer allocator.free(cwd);
    const pack = @import("repl_tools.zig");
    const directory = parsed.value.object.get("directory");
    if (directory) |d| {
        if (d != .string or !std.fs.path.isAbsolute(d.string))
            return sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Directory must be an absolute folder path", 400);
    }
    const root = switch (try pack.changeRoot(allocator, stream.io, cwd, cwd, if (directory) |d| d.string else cwd)) {
        .ok => |path| path,
        .refused => |reason| return sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400),
    };
    defer allocator.free(root);
    if (parsed.value.object.get("browse")) |browse| {
        if (browse != .bool) return sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Browse must be a boolean", 400);
        if (browse.bool) return sendWebToolFolders(allocator, stream, root);
    }
    const vision = if (parsed.value.object.get("vision")) |v| v == .bool and v.bool else false;
    if (parsed.value.object.get("write")) |w| {
        if (w != .bool) return sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Write must be a boolean", 400);
    }
    const writable = if (parsed.value.object.get("write")) |w| w.bool else false;
    const name = parsed.value.object.get("name");
    if (name == null) {
        const defs = try std.json.parseFromSlice(std.json.Value, allocator, pack.definitionsJson(vision, writable), .{});
        defer defs.deinit();
        const json = try std.json.Stringify.valueAlloc(allocator, .{ .tools = defs.value, .root = root, .edit_default = g_web_edit }, .{});
        defer allocator.free(json);
        return sendResponse(stream, "200 OK", "application/json", json);
    }
    const args = parsed.value.object.get("arguments");
    if (name.? != .string or args == null or args.? != .string)
        return sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Tool name and arguments must be strings", 400);
    const result = try pack.run(.{ .allocator = allocator, .io = stream.io, .root = root, .vision = vision, .writable = writable }, name.?.string, args.?.string);
    defer result.deinit(allocator);
    const json = try std.json.Stringify.valueAlloc(allocator, result, .{});
    defer allocator.free(json);
    try sendResponse(stream, "200 OK", "application/json", json);
}

/// Explicit folder-picker navigation, separate from the model's confined file tools.
fn sendWebToolFolders(allocator: std.mem.Allocator, stream: *Conn, root: []const u8) !void {
    var dir = std.Io.Dir.openDirAbsolute(stream.io, root, .{ .iterate = true }) catch
        return sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Cannot browse this folder", 400);
    defer dir.close(stream.io);
    var directories = std.ArrayList([]u8).empty;
    defer {
        for (directories.items) |name| allocator.free(name);
        directories.deinit(allocator);
    }
    var iterator = dir.iterate();
    var truncated = false;
    while (iterator.next(stream.io) catch return sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Cannot read this folder", 400)) |entry| {
        if (entry.kind != .directory or entry.name.len == 0 or entry.name[0] == '.' or @import("repl_tools.zig").isSecretName(entry.name)) continue;
        if (directories.items.len == 1000) {
            truncated = true;
            break;
        }
        const name = try allocator.dupe(u8, entry.name);
        errdefer allocator.free(name);
        try directories.append(allocator, name);
    }
    std.mem.sort([]u8, directories.items, {}, struct {
        fn less(_: void, a: []u8, b: []u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.less);
    const json = try std.json.Stringify.valueAlloc(allocator, .{
        .root = root,
        .parent = std.fs.path.dirname(root) orelse root,
        .directories = directories.items,
        .truncated = truncated,
    }, .{});
    defer allocator.free(json);
    try sendResponse(stream, "200 OK", "application/json", json);
}

fn isChatPagePath(path: []const u8) bool {
    return std.mem.eql(u8, path, "/") or std.mem.eql(u8, path, "/chat") or std.mem.eql(u8, path, "/dashboard");
}

/// The chat page URL printed at startup. A wildcard bind is shown as loopback,
/// because browsers refuse to open 0.0.0.0.
pub fn chatPageUrl(buf: []u8, host: []const u8, port: u16) []const u8 {
    const shown = if (std.mem.eql(u8, host, "0.0.0.0")) "127.0.0.1" else host;
    return std.fmt.bufPrint(buf, "http://{s}:{d}/", .{ shown, port }) catch "http://127.0.0.1/";
}

pub const max_request_bytes: usize = 64 * 1024 * 1024;

/// `?model=<id>` for bodiless GETs (`/props`), percent-decoded into `buf`.
pub fn queryModel(buf: []u8, raw_path: []const u8) ?[]const u8 {
    const q = raw_path[(std.mem.indexOfScalar(u8, raw_path, '?') orelse return null) + 1 ..];
    var it = std.mem.splitScalar(u8, q, '&');
    while (it.next()) |kv| {
        if (!std.mem.startsWith(u8, kv, "model=") or kv.len == "model=".len) continue;
        const v = kv["model=".len..];
        if (v.len > buf.len) return null;
        @memcpy(buf[0..v.len], v);
        return std.Uri.percentDecodeInPlace(buf[0..v.len]);
    }
    return null;
}

/// Module-level capacity for plan 03 hot prefix cache. main.zig writes this
/// from `--prefix-cache-entries N` before calling `serve()`.
// Default 32: agent clients (Claude Code, the app's agent loop) interleave
// requests from several conversation roots — main thread, subagents, title
// generation — and a single-entry cache gets its long system-prompt prefix
// evicted by every interleaved request, forcing a full re-prefill each
// turn. The count cap is pure retention metadata; the SSD byte budget
// (`prefixCacheDiskForLoad`) is what bounds the bytes. 0 disables.
pub var prefix_cache_capacity: u32 = 32;
/// RAM retention is opt-in: `--prefix-cache-mem` turns it on, `0` or `off` keeps it off.
pub var prefix_cache_ram_enabled: bool = false;

/// The RAM tier's byte budget, read only where RAM retention is on (`--prefix-cache-mem`). The cache
/// evicts LRU entries on commit until `current_kv_bytes + new_bytes <= prefix_cache_mem_bytes`. 0
/// disables the byte budget (the machine's headroom and the count cap from `--prefix-cache-entries`
/// still apply).
pub var prefix_cache_mem_bytes: u64 = PREFIX_CACHE_MEM_DEFAULT;
pub const PREFIX_CACHE_MEM_DEFAULT: u64 = 2 * 1024 * 1024 * 1024;
/// Set by `--prefix-cache-mem`: an operator's number is used as given, even when it equals the default.
pub var prefix_cache_mem_explicit = false;

/// `--prefix-cache-mem 0` (or `off`) is "no RAM retention".
pub fn ramRetentionForMemArg(mem_bytes: u64) bool {
    return mem_bytes > 0;
}

test "ramRetentionForMemArg: a zero budget keeps RAM retention off" {
    try testing.expect(!ramRetentionForMemArg(0));
    try testing.expect(ramRetentionForMemArg(1 << 30));
}

/// What the hot cache was actually given for the loaded model, after `clampedPrefixCacheMem`.
/// Every post-load reserve reads it through `resolvedPrefixCacheMem()`. Atomic: written on the
/// inference thread, read from connection threads. `maxInt(u64)` is the unresolved sentinel,
/// not 0 (`--prefix-cache-mem 0` under SSD-first publishes a real zero).
const HOT_CACHE_MEM_UNRESOLVED: u64 = std.math.maxInt(u64);
var hot_cache_mem_resolved = std.atomic.Value(u64).init(HOT_CACHE_MEM_UNRESOLVED);

/// The hot-cache byte budget every post-load reserve must bill: the clamp's answer once loaded, the raw ask before.
pub fn resolvedPrefixCacheMem() u64 {
    if (prefix_cache_capacity == 0 or !prefix_cache_ram_enabled) return 0;
    const v = hot_cache_mem_resolved.load(.monotonic);
    return if (v != HOT_CACHE_MEM_UNRESOLVED) v else prefix_cache_mem_bytes;
}

/// Publish the clamp's answer (inference thread, end of a load); `clearResolvedPrefixCacheMem`
/// resets it on unload so the next model does not inherit this one's budget.
pub fn publishResolvedPrefixCacheMem(v: u64) void {
    hot_cache_mem_resolved.store(v, .monotonic);
}

pub fn clearResolvedPrefixCacheMem() void {
    hot_cache_mem_resolved.store(HOT_CACHE_MEM_UNRESOLVED, .monotonic);
}

/// The SSD tier's byte ask (`--prefix-cache-disk`, 0 = off). Committed KV prefixes persist as chunked
/// safetensors under `~/.sushi/kv-cache/<model-fingerprint>` and are restored across restarts instead
/// of recomputed, LRU-evicted to the budget. Unnamed, the ask is the 20 GB ceiling and the tier is on;
/// `prefixCacheDiskForLoad` sizes the real budget per model (the Swift app's Settings toggle passes the flag).
pub var prefix_cache_disk_bytes: u64 = kv_disk_cache.DEFAULT_DISK_CAP;
/// Set by `--prefix-cache-disk`: the operator's number is the budget as given.
pub var prefix_cache_disk_explicit = false;

/// Does THIS model run with an SSD tier? The launch asked for one and its load did not decline it
/// (`ModelConfig.prefix_cache_disk_declined`); every bill, budget and checkpoint policy reads this.
pub fn diskTierOn(config: *const model_mod.ModelConfig) bool {
    return prefix_cache_disk_bytes > 0 and !config.prefix_cache_disk_declined;
}

/// The SSD writer's permit, sized once from host RAM at startup (`kv_disk_writer.permitForRam`); the
/// scheduler arms every writer with it.
pub var disk_writer_permit_bytes: u64 = kv_disk_writer.DEFAULT_PERMIT_BYTES;

/// The disk bytes the SSD-tier-dependent policies read: the launch ask, or 0 for a model whose tier is off.
fn diskBytesFor(config: *const model_mod.ModelConfig) u64 {
    return if (diskTierOn(config)) prefix_cache_disk_bytes else 0;
}

/// Phase 1 (performance-plan): SSM/conv state snapshot stride during prefill,
/// in tokens. Non-zero values enable multi-turn warm reuse on hybrid SSM
/// architectures (Flash-Next GatedDeltaNet). Set to 0 to disable (fall back to pre-Phase-1 behavior:
/// hybrid models bypass the hot prefix cache entirely). Override via
/// `--ssm-checkpoint-stride N`.
///
/// Default 256. The stride forces a prefill CHUNK boundary at every multiple
/// (to snapshot SSM/sliding-window state mid-prompt). On dense / non-MoE-hybrid
/// models prefill is compute-bound, so a fine 256 stride is ~free and buys
/// finer warm mid-prompt reuse. **MoE models are different**: their prefill is
/// memory-bound on the per-expert weights, and every extra chunk re-streams
/// ~all expert weights from HBM, so a fine stride silently costs cold prefill
/// on a MoE (an 850-token prompt = 4 chunks = ~4x expert-weight traffic). To
/// avoid that, the prefill loop
/// **coarsens this stride to >= PREFILL_CHUNK for MoE models only** (see
/// `generate.effectiveSsmCheckpointStride`), so MoE prefill is never
/// over-chunked at any prompt length while non-MoE keeps the fine stride.
/// The always-on end-of-prompt snapshot preserves append-growth multi-turn
/// reuse for MoE regardless. 0 disables (hybrid models then bypass the hot cache);
/// only tests move it off the default.
pub const SSM_CHECKPOINT_STRIDE_DEFAULT: u32 = 256;
pub var ssm_checkpoint_stride: u32 = SSM_CHECKPOINT_STRIDE_DEFAULT;

/// Phase 1: cap on the number of checkpoints retained per request. Snapshots
/// past this drop the oldest (front of list) to keep memory bounded on very
/// long prompts. 0 = unlimited (rely on the prefix-cache byte budget alone).
pub var ssm_checkpoint_max: u32 = 16;

/// SSM checkpoints feed reusable prefix tiers; zero capacity disables all reuse.
pub fn effectiveSsmCheckpointStride(stride: u32, cache_capacity: u32, ram_enabled: bool, disk_bytes: u64) u32 {
    if (cache_capacity == 0 or (!ram_enabled and disk_bytes == 0)) return 0;
    return stride;
}

test "effectiveSsmCheckpointStride: zero capacity disables both tiers, RAM-off preserves disk" {
    try std.testing.expectEqual(@as(u32, 0), effectiveSsmCheckpointStride(256, 0, true, 4 << 30));
    try std.testing.expectEqual(@as(u32, 0), effectiveSsmCheckpointStride(256, 0, false, 4 << 30));
    try std.testing.expectEqual(@as(u32, 256), effectiveSsmCheckpointStride(256, 32, true, 0));
    try std.testing.expectEqual(@as(u32, 256), effectiveSsmCheckpointStride(256, 32, false, 4 << 30));
    try std.testing.expectEqual(@as(u32, 0), effectiveSsmCheckpointStride(256, 32, false, 0));
    try std.testing.expectEqual(@as(u32, 0), effectiveSsmCheckpointStride(0, 32, true, 0));
}

/// LRU capacity of the per-LoadedModel chat-template tokenize cache. Small
/// because real chat conversations mutate the messages list every turn; it
/// catches warm-reuse benches and repeated agent probes without hoarding
/// token buffers across a long session.
pub const tokenize_cache_entries: u32 = 4;

pub fn parseOnOff(word: []const u8) ?bool {
    if (std.mem.eql(u8, word, "on")) return true;
    if (std.mem.eql(u8, word, "off")) return false;
    return null;
}

/// Plan 01 — continuous batching: maximum concurrent in-flight requests sharing
/// the inference thread's batched-decode pass. Set via `--max-concurrent N`.
/// Default 1 = legacy single-slot behavior (no scheduler engagement, every
/// existing test bit-identical). Values >1 require Phase 2's scheduler wiring
/// (handler refactor + per-connection threads), which the `serve()` startup
/// guard checks before enabling.
pub var max_concurrent: u32 = 1;

// Plan 05: vision encoder and model id moved to `LoadedModel.vision_encoder`
// and `LoadedModel.id`. Handlers read them off `lm`. `global_vision_encoder`
// and `global_model_id` singletons were removed. The `discovered_models`
// slice was also removed — `/v1/models` iterates `registry.entries` directly.

/// Assistant-history reasoning field on an incoming chat message:
/// `reasoning_content` (our own SSE/vLLM field, what pi rounds-trips) with
/// `reasoning` (the vLLM request spelling some templates read first) as
/// the fallback. Non-empty strings only — an empty string would render an
/// empty <think></think> block, which is exactly the nothink signature this
/// field exists to avoid.
fn messageReasoningFromObj(obj: std.json.ObjectMap) ?[]const u8 {
    inline for (.{ "reasoning_content", "reasoning" }) |key| {
        if (obj.get(key)) |v| {
            if (v == .string and v.string.len > 0) return v.string;
        }
    }
    return null;
}

/// True when the rendered generation prompt ends inside a template-opened
/// think block (Qwen 3.5/3.6 render `…assistant\n<think>\n` when thinking is
/// on). Decodes the last few prompt tokens — cheap, engine-agnostic, and
/// independent of the tokenize cache. Drives the unclosed-think split policy
/// in `chat.splitThinkBlock` so a length-truncated thought never leaks into
/// visible content.
pub fn promptOpensThink(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
) bool {
    if (prompt_ids.len == 0) return false;
    const n = @min(prompt_ids.len, 8);
    const tail = tok.decode(allocator, prompt_ids[prompt_ids.len - n ..]) catch return false;
    defer allocator.free(tail);
    return chat_mod.promptTailOpensThink(tail);
}

/// One token id for `text`, or null when the tokenizer spells it in pieces.
pub fn atomicTokenId(allocator: std.mem.Allocator, tok: *const Tokenizer, text: []const u8) ?u32 {
    const ids = tok.encode(allocator, text) catch return null;
    defer allocator.free(ids);
    return if (ids.len == 1) ids[0] else null;
}

/// Qwen's own thinking-budget recipe: the model is told time is up, then the
/// closer is committed and the answer follows.
const THINK_BOUND_EARLY_STOP = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n";

/// Arm a decode-time thinking bound for this request, or null when nothing
/// bounds it (no budget, thinking off, markers not atomic). `forced` is
/// allocated; the caller frees it after generation.
pub fn armThinkBound(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer, prompt_ids: []const u32, enable_thinking: bool, budget: i32) ?mtp_plan.ThinkBound {
    if (budget < 0) return null;
    return thinkBoundWithBudget(allocator, lm, tok, prompt_ids, enable_thinking, @intCast(budget), true);
}

/// The bound a loop convicted inside a thought closes it with, built when the loop is
/// convicted: a request carrying a `think_bound` from the start is refused DFlash2.
pub fn loopCloseBound(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer, prompt_ids: []const u32, enable_thinking: bool) ?mtp_plan.ThinkBound {
    return thinkBoundWithBudget(allocator, lm, tok, prompt_ids, enable_thinking, std.math.maxInt(u32), false);
}

fn thinkBoundWithBudget(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer, prompt_ids: []const u32, enable_thinking: bool, budget: u32, announce: bool) ?mtp_plan.ThinkBound {
    if (!enable_thinking or lm.transformer == null) return null;
    const closer = atomicTokenId(allocator, tok, chat_mod.BARE_THINK_CLOSER) orelse {
        if (announce) log.info("  reasoning budget {d}: think markers are not atomic tokens, delivery cap only\n", .{budget});
        return null;
    };
    const opener = atomicTokenId(allocator, tok, chat_mod.BARE_THINK_OPENER);
    const opened = promptOpensThink(allocator, tok, prompt_ids);
    if (opener == null and !opened) return null;

    const stop_ids = tok.encode(allocator, THINK_BOUND_EARLY_STOP) catch return null;
    defer allocator.free(stop_ids);
    const sep_ids = tok.encode(allocator, "\n\n") catch return null;
    defer allocator.free(sep_ids);
    const forced = allocator.alloc(u32, stop_ids.len + 1 + sep_ids.len) catch return null;
    @memcpy(forced[0..stop_ids.len], stop_ids);
    forced[stop_ids.len] = closer;
    @memcpy(forced[stop_ids.len + 1 ..], sep_ids);
    if (announce) log.info("  reasoning budget {d}: enforced in-stream\n", .{budget});
    return .{ .budget = budget, .opener_id = opener, .closer_id = closer, .forced = forced, .in_think = opened };
}

pub const TOOL_CHOICE_UNDECLARED = "tool_choice names a function that is not in tools";

/// The decode-time half of tool_choice "none": the template's call opener, banned when it is one
/// token, so a model that sees earlier calls in the history cannot write one as plain text.
fn noneCallBan(allocator: std.mem.Allocator, tok: *const Tokenizer, chat_config: *const chat_mod.ChatConfig) ?@import("logit_bias.zig").Bias {
    const opener = (chat_mod.callOpener(allocator, chat_config) catch null) orelse return null;
    defer allocator.free(opener);
    const id = atomicTokenId(allocator, tok, opener) orelse return null;
    return .{ .id = id, .delta = -100 };
}

/// Arm the decode-time half of a forced tool_choice (required or named), or
/// null when the choice forces nothing or the call cannot be committed: no
/// call markup the template writes after a closed thought, a prompt-opened
/// thought whose closer is not one token, or `required` with no declared name.
/// The prompt instruction then stands alone. The caller `deinit`s the result.
pub fn armCallForce(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    prompt_ids: []const u32,
    enable_thinking: bool,
    choice: chat_mod.ToolChoice,
    tools_json: ?[]const u8,
) ?generate_mod.CallForce {
    const name: ?[]const u8 = switch (choice) {
        .auto, .none => return null,
        .required => null,
        .named => |n| n,
    };
    if (prompt_ids.len == 0) return null;
    var text = (chat_mod.forcedCallText(allocator, chat_config, name) catch null) orelse {
        log.info("  [tool-choice] forced call is prompt-only: no call markup after a closed thought in the template\n", .{});
        return null;
    };
    defer text.deinit(allocator);
    const closer = atomicTokenId(allocator, tok, chat_mod.BARE_THINK_CLOSER);
    const opener = atomicTokenId(allocator, tok, chat_mod.BARE_THINK_OPENER);
    const phase: generate_mod.CallForce.Phase = if (promptOpensThink(allocator, tok, prompt_ids)) blk: {
        if (closer == null) {
            log.info("  [tool-choice] forced call is prompt-only: the think closer is not one token\n", .{});
            return null;
        }
        break :blk .thinking;
    } else if (enable_thinking and opener != null and closer != null and templateThinkOpener(chat_config.chat_template) != null)
        .open
    else
        .answer;

    var names: NameIds = .{};
    var armed = false;
    defer if (!armed) names.deinit(allocator);
    if (name == null) {
        names = nameIds(allocator, tok, tools_json orelse "[]", text.name_tail) catch return null;
        if (names.slices.len == 0) {
            log.info("  [tool-choice] forced call is prompt-only: no declared function to name\n", .{});
            return null;
        }
    }
    const sep_ids: []u32 = if (text.sep.len == 0) &.{} else tok.encode(allocator, text.sep) catch return null;
    defer if (sep_ids.len > 0) allocator.free(sep_ids);
    const call_ids = tok.encode(allocator, text.call) catch return null;
    defer allocator.free(call_ids);
    const forced = std.mem.concat(allocator, u32, &.{ sep_ids, call_ids }) catch return null;
    const choice_buf = allocator.alloc(u32, names.slices.len) catch {
        allocator.free(forced);
        return null;
    };
    armed = true;
    log.info("  [tool-choice] forced call {s}: committed {s}\n", .{ name orelse "(any declared)", switch (phase) {
        .open => "after the opener at token 0 and the closed thought",
        .thinking => "once the thought closes",
        .answer, .name, .done => "at token 0",
    } });
    return .{
        .forced = forced,
        .call_at = sep_ids.len,
        .closer_id = closer,
        .opener = .{opener orelse 0},
        .phase = phase,
        .last = prompt_ids[prompt_ids.len - 1],
        .names = names.slices,
        .names_flat = names.flat,
        .choice_buf = choice_buf,
    };
}

const NameIds = struct {
    flat: []const u32 = &.{},
    slices: []const []const u32 = &.{},

    fn deinit(self: *NameIds, allocator: std.mem.Allocator) void {
        allocator.free(self.flat);
        allocator.free(self.slices);
    }
};

/// Every declared function name followed by `tail`, tokenized.
fn nameIds(allocator: std.mem.Allocator, tok: *const Tokenizer, tools_json: []const u8, tail: []const u8) !NameIds {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const declared = try chat_mod.declaredToolNames(arena, tools_json);
    const per_name = try arena.alloc([]u32, declared.len);
    var total: usize = 0;
    for (declared, per_name) |n, *ids| {
        const text = try std.mem.concat(arena, u8, &.{ n, tail });
        const encoded = try tok.encode(allocator, text);
        defer allocator.free(encoded);
        ids.* = try arena.dupe(u32, encoded);
        total += ids.len;
    }
    const flat = try allocator.alloc(u32, total);
    errdefer allocator.free(flat);
    const slices = try allocator.alloc([]const u32, declared.len);
    var at: usize = 0;
    for (per_name, slices) |ids, *slice| {
        @memcpy(flat[at..][0..ids.len], ids);
        slice.* = flat[at..][0..ids.len];
        at += ids.len;
    }
    return .{ .flat = flat, .slices = slices };
}

/// A generated tag needs one unambiguous literal spelling in the template.
/// Dynamic or conflicting spellings retain the unsupported-format fallback.
fn templateThinkOpener(template: []const u8) ?[]const u8 {
    var found: ?[]const u8 = null;
    var offset: usize = 0;
    while (std.mem.indexOfPos(u8, template, offset, "<think")) |start| {
        const len = chat_mod.thinkOpenTagLenAt(template[start..]) orelse return null;
        const tag = template[start..][0..len];
        if (found) |previous| {
            if (!std.mem.eql(u8, previous, tag)) return null;
        }
        found = tag;
        offset = start + len;
    }
    return found;
}

/// Resolve a reasoning protocol from the rendered prompt and template. Tokenizer
/// indexes and exact recovery suffixes are cached on the loaded model.
pub fn resolveReasoningProtocol(
    allocator: std.mem.Allocator,
    io: std.Io,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    prompt_ids: []const u32,
    proto_out: *rp_mod.Protocol,
    allow_reasoning: bool,
) bool {
    if (prompt_ids.len == 0 or lm.transformer == null) return false;
    const tail_n = @min(prompt_ids.len, 64);
    const tail = tok.decode(allocator, prompt_ids[prompt_ids.len - tail_n ..]) catch return false;
    defer allocator.free(tail);
    const trimmed = std.mem.trimEnd(u8, tail, "\n\r\t ");

    const template = chat_config.chat_template;
    const kind: rp_mod.Kind = if (std.mem.indexOf(u8, template, "<|channel>") != null)
        .gemma
    else
        .bare_think;
    proto_out.* = .{ .kind = kind };
    const proto = proto_out;
    if (kind != .bare_think) {
        proto.configureChannels();
        if (!proto.startFromPrompt(tail)) return false;
    } else {
        switch (chat_mod.promptThinkTailClass(trimmed)) {
            .bare => {
                proto.kind = .bare_think;
                if (!proto.setCloser(chat_mod.BARE_THINK_CLOSER)) return false;
            },
            .none => {
                const opener = templateThinkOpener(template) orelse return false;
                var closer_buf: [rp_mod.MAX_MARKER_BYTES]u8 = undefined;
                const close = std.fmt.bufPrint(&closer_buf, "</{s}", .{opener[1..]}) catch return false;
                if (!proto.setOpener(opener) or !proto.setCloser(close)) return false;
            },
        }
    }

    var final_header: [rp_mod.MAX_MARKER_BYTES]u8 = undefined;
    if (!allow_reasoning and !proto.finalOnly(&final_header)) return false;

    const closer = lm.reasoningMarker(allocator, io, proto.closerText()) catch return false;
    proto.closer_atomic = closer.atomic;
    proto.closer_suffix = closer.suffix;
    proto.closer_suffix_buf = closer.tokens;
    proto.closer_span_candidates = closer.close_candidates;
    if (closer.atomic == null) {
        const run = closer.suffix[0];
        if (run.len == 0 or !proto.setForced(closer.tokens[run.offset..][0..run.len])) return false;
    }
    if (proto.openerText()) |text| {
        const opener = lm.reasoningMarker(allocator, io, text) catch return false;
        proto.opener_atomic = opener.atomic;
        proto.opener_suffix = opener.suffix;
        proto.opener_suffix_buf = opener.tokens;
        proto.opener_candidates = opener.open_candidates;
        if (opener.atomic == null and opener.open_candidates.len == 0) return false;
    }
    for (proto.headers[0..proto.header_len]) |*rule| {
        const marker = lm.reasoningMarker(allocator, io, rule.text) catch return false;
        if (marker.suffix[0].len == 0) return false;
        rule.text = marker.text;
        rule.candidates = marker.open_candidates;
        rule.suffix = marker.suffix;
        rule.tokens = marker.tokens;
    }
    // Structural special ids remain separate from JSON token bytes.
    if (proto.closer_atomic) |id| {
        proto.specials[proto.special_len] = .{ .id = id, .text = closer.text };
        proto.special_len += 1;
    }
    for ([_][]const u8{
        "<|channel>", "<channel|>",
    }) |text| {
        if (tok.specialTokenId(text)) |id| {
            proto.specials[proto.special_len] = .{ .id = id, .text = text };
            proto.special_len += 1;
        }
    }
    return true;
}

/// A request's `seed`; negative is the llama.cpp "random" convention and means unseeded.
pub fn parseRequestSeed(v: ?std.json.Value) ?u64 {
    const val = v orelse return null;
    return switch (val) {
        .integer => |i| if (i < 0) null else @as(u64, @intCast(i)),
        else => null,
    };
}

/// A client-supplied JSON integer narrowed to i32 by clamping, never by `@intCast`.
pub fn clampJsonI32(i: i64) i32 {
    return @intCast(std.math.clamp(i, std.math.minInt(i32), std.math.maxInt(i32)));
}

/// Widest string `describeMaxTokens` writes: a 10-digit budget plus
/// " (launch default)".
pub const MAX_TOKENS_DESC_LEN: usize = 27;

pub fn ignoreEosRequested(root: std.json.ObjectMap) bool {
    const v = root.get("ignore_eos") orelse return false;
    return v == .bool and v.bool;
}

/// Chat refuses `ignore_eos`: past its end of turn the model writes another turn, whose think
/// block the non-stream reply merges into the reasoning and the live stream cannot.
pub fn chatIgnoreEosRejectReason(root: std.json.ObjectMap) ?[]const u8 {
    if (!ignoreEosRequested(root)) return null;
    return "'ignore_eos' is supported on /v1/completions only: a chat reply ends at its end of turn";
}

/// The auto budget's own tightness question: under a quarter of the window left.
pub fn autoBudgetWindowTight(remaining: u32, effective_ctx: u32) bool {
    return remaining < effective_ctx / 4;
}

/// Finish reason for a response whose generated text yielded tool calls.
/// A TRUNCATED generation ("length": max_tokens or the request timeout) must
/// keep reporting "length" even when the cut-off text salvaged a partial tool
/// call (the truncated-opener recovery in chat.parseToolCalls) — clients key
/// their truncation recovery on it (the app's chunk-and-retry nudge fires on
/// finish_reason "length" + tool calls present). Overriding to "tool_calls"
/// hides the cut: the client executes a half-argument call, blames the model
/// ("you sent no content"), and the model re-emits the same doomed mega-call.
pub fn toolCallFinishReason(pre_parse: []const u8) []const u8 {
    return if (std.mem.eql(u8, pre_parse, "length")) "length" else "tool_calls";
}

/// A repetition-loop cut may land inside an otherwise recognizable tool call.
/// Never emit that buffer as executable work: unlike a genuine max-token cut,
/// this intentional stop must not ask clients to recover by compacting/retrying.
pub fn shouldParseToolCalls(finish_details: ?[]const u8) bool {
    const d = finish_details orelse return true;
    return !std.mem.eql(u8, d, "repetition_loop");
}

/// The `finish_details` object emitted BESIDE `finish_reason` on a choice,
/// or "" when there is nothing to say. Comes with its leading comma so call
/// sites splice it straight into the choice literal.
///
/// The loop guard uses `finish_reason: "stop"`; the sibling preserves why the
/// server stopped without overloading "length" (which makes agents run output-
/// or context-exhaustion recovery). OpenAI's deprecated `finish_details` is the
/// closest precedent, and conforming clients ignore keys they don't know.
pub fn finishDetailsField(reason: []const u8, details: ?[]const u8) []const u8 {
    const d = details orelse return "";
    // Every emitter can override the slot's reason after the fact. Gate on the
    // reason actually being emitted so a cause never accompanies a genuine
    // length exhaustion, tool completion, or client disconnect.
    if (!std.mem.eql(u8, reason, "stop")) return "";
    // One known value today; a switch here keeps an unknown string from
    // reaching the wire as an unescaped literal.
    if (std.mem.eql(u8, d, "repetition_loop")) return ",\"finish_details\":{\"type\":\"repetition_loop\"}";
    return "";
}

/// Debug-only corpus-harvest aid (`SUSHI_RAW_DUMP_FILE`). Appends ONE framed
/// record per tools request: the DECLARED TOOLS SCHEMA and the raw pre-parse text
/// together, written at the one site where both are in scope. Correlating them
/// any other way is unsound — the debug log truncates lines at 16 KB (so a large
/// request body never re-parses, and a harvester silently pairs a model output
/// with some EARLIER request's schema), and concurrent requests interleave.
///
/// Framing is by byte COUNT, never a delimiter: both payloads are arbitrary bytes
/// and any sentinel can occur inside them. libc `write(2)` like log.zig's sink
/// (callers here carry no `Io`); O_APPEND keeps conn threads from interleaving.
pub fn appendRawToolDump(path: []const u8, tools_json: ?[]const u8, text: []const u8) void {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len == 0 or path.len >= path_buf.len) return;
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;
    const path_z: [*:0]const u8 = @ptrCast(&path_buf);

    const fd = std.c.open(path_z, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return;
    defer _ = std.c.close(fd);

    const tools = tools_json orelse "";
    var hdr: [96]u8 = undefined;
    const h = std.fmt.bufPrint(&hdr, "\n===MLX_RAW_DUMP tools={d} raw={d}===\n", .{ tools.len, text.len }) catch return;
    _ = std.c.write(fd, h.ptr, h.len);
    if (tools.len > 0) _ = std.c.write(fd, tools.ptr, tools.len);
    if (text.len > 0) _ = std.c.write(fd, text.ptr, text.len);
}

/// The ONE tool-call extraction path for every HTTP surface: parse, fall back to
/// bare-JSON inference, then coerce the arguments to the schema the request
/// declared. Every surface must go through this — a site that calls
/// `chat_mod.parseToolCalls` directly silently reintroduces the value-spelling
/// type-inference bug (a boolean param arriving as the string "False") on that
/// surface alone, which no output-equality test can see.
pub fn parseToolCallsForRequest(
    allocator: std.mem.Allocator,
    text: []const u8,
    tools_json: ?[]const u8,
    /// OpenAI `parallel_tool_calls` (Anthropic spelling:
    /// !tool_choice.disable_parallel_tool_use). false = the client accepts AT
    /// MOST ONE call per response, so the chokepoint keeps the first parsed
    /// call and drops the rest (the model re-issues them next round after the
    /// first result). Deliberately NOT gated by --no-tool-autocorrect:
    /// client-requested behavior, not a repair heuristic.
    allow_parallel: bool,
) !?[]chat_mod.ParsedToolCall {
    var parsed_calls = try chat_mod.parseToolCalls(allocator, text);
    // Heuristically-inferred raw-JSON calls must name a DECLARED tool — a
    // truncated data object ({"name": "George Washington", …}) is not a call.
    // Deliberately NOT gated on g_tool_autocorrect: this corrects our own
    // heuristic's false positive, not the model's output.
    if (parsed_calls) |c| {
        if (tools_json) |tj| parsed_calls = try chat_mod.filterInferredBySchema(allocator, c, tj);
    }
    var calls = parsed_calls orelse
        (if (tools_json) |tj| try chat_mod.inferBareJsonToolCalls(allocator, text, tj) else null) orelse
        return null;
    if (!allow_parallel and calls.len > 1) {
        log.info("  parallel_tool_calls=false: clamping {d} parsed calls to the first\n", .{calls.len});
        for (calls[1..]) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        const kept = calls[0];
        allocator.free(calls);
        const one = try allocator.alloc(chat_mod.ParsedToolCall, 1);
        one[0] = kept;
        calls = one;
    }
    if (g_tool_autocorrect) {
        if (tools_json) |tj| {
            // Order matters: put a buried required param back where the schema
            // says it lives BEFORE coercing types, so the hoisted value is
            // type-checked like any other top-level arg.
            try chat_mod.hoistMisplacedRequiredParams(allocator, calls, tj);
            try chat_mod.coerceToolArgsToSchema(allocator, calls, tj);
        }
    }
    return calls;
}

/// Start the HTTP server on the given host and port.
///
/// `cfg` carries all process-wide defaults (context size, timeouts, PLD
/// defaults, reasoning budget). It's copied into the module-level
/// `server_config` before the listen loop starts.
pub fn serve(
    io: std.Io,
    allocator: std.mem.Allocator,
    /// Phase A1: model load happens on the scheduler's inference thread. The
    /// caller (main.zig) is responsible only for CPU-side setup (config parse,
    /// tokenizer load, EOS resolution, chat-template load); everything that
    /// touches mlx is done by `Scheduler.init` before this fn proceeds.
    load_params: scheduler_mod.LoadParams,
    config: *const model_mod.ModelConfig,
    host: []const u8,
    port: u16,
    cfg: ServerConfig,
) !void {
    server_config = cfg;
    // Before `Scheduler.init`: that call is the model load and every load-time bill inside it
    // asks for the KV width.
    configured_kv_quant = if (load_params.kv_quant_explicit) load_params.kv_quant_config else null;
    defer configured_kv_quant = null;
    configured_mtp = model_settings.launchFlag(bool, load_params.mtp_enabled, load_params.mtp_explicit);
    defer configured_mtp = null;
    scheduler_mod.load_serving_bill = &loadServingBill;
    defer scheduler_mod.load_serving_bill = null;

    scheduler_mod.prefill_ubench_chunk = &requestPrefillChunkNow;
    defer scheduler_mod.prefill_ubench_chunk = null;
    // ── Phase A1: spin up the scheduler. Its inference thread does the
    //    Transformer/vision/drafter load, JIT compile, and warmup before
    //    `Scheduler.init` returns. From this point on, mlx ops are bound
    //    to the inference thread's GPU stream.
    var scheduler = try scheduler_mod.Scheduler.init(
        allocator,
        io,
        load_params,
        max_concurrent,
    );
    defer scheduler.deinit();
    global_scheduler = scheduler;
    defer global_scheduler = null;
    // The inference thread's evict-or-refuse hook (#353); the scheduler has no server import.
    scheduler_mod.prefill_admission_fits = &prefillFitsNow;
    defer scheduler_mod.prefill_admission_fits = null;
    scheduler_mod.prefill_admission_numbers = &prefillBillNumbersNow;
    defer scheduler_mod.prefill_admission_numbers = null;
    scheduler_mod.prefill_request_chunk = &requestPrefillChunkNow;
    defer scheduler_mod.prefill_request_chunk = null;
    // The published budget is per model; retire it whenever the scheduler drops the cache.
    scheduler_mod.hot_cache_budget_invalidate = &clearResolvedPrefixCacheMem;
    defer scheduler_mod.hot_cache_budget_invalidate = null;
    scheduler_mod.prefill_admission_refused_log = &logPrefillRefusal;
    defer scheduler_mod.prefill_admission_refused_log = null;
    // Asked by the prefill loop: the width of the next chunk, re-priced at every boundary.
    scheduler_mod.prefill_chunk_adapt = &adaptivePrefillWidthNow;
    defer scheduler_mod.prefill_chunk_adapt = null;
    dsv41_borrow.headroom_fn = &dsv41Headroom;
    defer dsv41_borrow.headroom_fn = null;
    dsv41_borrow.pool_ram_cap = mlxCacheLimitBytes(metrics.getTotalMemBytes());
    defer dsv41_borrow.pool_ram_cap = 0;
    scheduler_mod.prefill_chunk_widen_ok = &adaptivePrefillWidenStillFits;
    defer scheduler_mod.prefill_chunk_widen_ok = null;
    // Whether the hook above can move anything for a given model (the installs are process-wide, so presence is not a gate).
    scheduler_mod.prefill_chunk_adaptive_enabled = &adaptivePrefillChunkEnabled;
    defer scheduler_mod.prefill_chunk_adaptive_enabled = null;
    scheduler_mod.vision_encode_available = &visionEncodeAvailable;
    scheduler_mod.vision_encode_bill = &visionEncodeBillBytes;
    scheduler_mod.vision_image_decode = &decodeEncodedImage;
    scheduler_mod.vision_fit_fault = &towerFitFaultText;
    defer scheduler_mod.vision_encode_available = null;

    if (instr.current() != null) {
        startGaugeSampler() catch |err| log.err("metrics gauge sampler failed to start: {s}\n", .{@errorName(err)});
        log.info("Prometheus metrics: ENABLED — GET http://{s}:{d}/metrics and /metrics.json\n", .{ host, port });
    }
    // Registered after the scheduler's deinit defer, so it runs first and the sampler never reads a freed scheduler.
    defer stopGaugeSampler();

    // Idle eviction (--idle-evict-secs). Same lifetime shape as the sampler,
    // and never in `Scheduler.deinit` — see `IdleEvictCtx.stop`.
    var idle_evict_thread: ?std.Thread = null;
    var idle_evict_stop = std.atomic.Value(bool).init(false);
    if (scheduler.registry.idle_evict_secs) |secs| {
        idle_evict_thread = try std.Thread.spawn(.{}, idleEvictLoop, .{IdleEvictCtx{
            .scheduler = scheduler,
            .window_ms = @as(i64, secs) * 1000,
            .stop = &idle_evict_stop,
        }});
        log.info("[registry] idle eviction ON — unloading models idle for {d}s\n", .{secs});
    }
    defer if (idle_evict_thread) |t| t.join();
    defer idle_evict_stop.store(true, .monotonic);

    global_registry = scheduler.registry;
    defer global_registry = null;

    // Plan 05: the hot prefix cache lives on the LoadedModel
    // (entry.prefix_cache) and is set up by `loadModelOnInferenceThread`
    // using the per-LoadParams capacity + byte budget. Surface a friendly
    // log line so users see whether the cache engaged.
    if (scheduler.hot_prefix_cache != null) {
        if (!prefix_cache_ram_enabled and diskTierOn(config)) {
            log.info("Prefix cache: SSD ONLY (RAM retention off)\n", .{});
        } else if (resolvedPrefixCacheMem() > 0) {
            const cap_mb = @as(f64, @floatFromInt(resolvedPrefixCacheMem())) / (1024.0 * 1024.0);
            log.info("Hot prefix cache: ENABLED (capacity={d}, mem-cap={d:.1} MB)\n", .{ prefix_cache_capacity, cap_mb });
        } else {
            log.info("Hot prefix cache: ENABLED (capacity={d}, mem-cap=unlimited)\n", .{prefix_cache_capacity});
        }
    } else if (prefix_cache_capacity > 0) {
        if (config.full_attention_interval > 0) {
            log.info("Hot prefix cache: requested capacity={d} but model uses full-attention-interval — disabled\n", .{prefix_cache_capacity});
        } else {
            log.info("Hot prefix cache: requested capacity={d} but disabled by scheduler\n", .{prefix_cache_capacity});
        }
    }

    // `--max-concurrent N` is honored when the model can ride the batched
    // decode kernel (pure-attention, no SSM/MoE/encoder). Hybrid / MoE /
    // encoder architectures clamp to 1 — they need single-slot serial
    // because the batched kernel doesn't model their state. DSV4 is
    // honored (per-slot LatentKVCache landed in Plan 04 Section 4
    // Phase A+B; Section B fix 2026-05-13 excludes DSV4 from
    // `Scheduler.batchable` so two DSV4 slots fall through to per-slot
    // `runSingleDecodeTick` — sequential through the inference thread,
    // safe).
    // `--max-concurrent` sizes the submit queue; requests decode together at
    // any value. Whether they share ONE forward is the model's answer, printed
    // at every value so a default-1 boot does not read as "one at a time".
    const startup_entry = scheduler.registry.resolveEntry("") catch null;
    const verdict = if (startup_entry) |e| batchVerdictFor(e) else batchVerdictOf(config, false);
    log.info("Concurrency: --max-concurrent={d}, {s} (arch: {s})\n", .{ max_concurrent, concurrencyClause(verdict), config.model_type });
    if (verdict == .ok and prefix_cache_capacity > 0 and prefix_cache_capacity < max_concurrent) prefix_cache_capacity = max_concurrent;
    // Install signal handlers for graceful shutdown
    const sigact = std.posix.Sigaction{
        .handler = .{ .handler = signalHandler },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    };
    std.posix.sigaction(std.posix.SIG.INT, &sigact, null);
    std.posix.sigaction(std.posix.SIG.TERM, &sigact, null);

    listen_host = host;
    listen_port = port;
    // Parse host address
    var server = startListener(host, port) catch |err| {
        var msg_buf: [512]u8 = undefined;
        if (startupRefusal(err, host, port, &msg_buf)) |msg| {
            log.err("{s}\n", .{msg});
            std.process.exit(1);
        }
        return err;
    };
    var client_thread: ?std.Thread = null;
    defer closeEmbeddedListener(io, &server, client_thread);
    listen_port = server.socket.address.ip4.port;
    client_thread = if (server_config.on_ready) |ready| try ready(listen_port) else null;

    // Freeze the auto-context NOW, at startup, while the model is freshly
    // loaded and nothing else has taken RAM. Clients read this number once
    // (pi/opencode bake it into a config file) and budget against it for the
    // whole session, so it must not drift with system load. `--ctx-size` wins.
    const pinned = pinAutoContext(@constCast(config));
    warnUserWireLimit();
    if (manualContext(config) > 0) {
        const ctx = model_settings.contextPick(server_config.max_context_size, config.ctx_override);
        log.info("Context size: {d} tokens ({s})\n", .{ ctx.value, model_settings.sourceLabel(ctx.source, "--ctx-size") });
    } else {
        const memory_ctx = computeMemoryContext(config);
        const memory_allows = safeAutoContext(memory_ctx);
        const rope_ctx = config.contextCap();
        if (rope_ctx > 0 and pinned >= rope_ctx) {
            // The checkpoint's own maximum binds; memory had room to spare.
            log.info("Context size: {d} tokens (auto: the model's maximum; memory would allow {d}) [pinned]\n", .{ pinned, memory_allows });
        } else {
            log.info("Context size: {d} tokens (auto: {d}% of the {d}-token memory ceiling, reserving headroom) [pinned]\n", .{ pinned, auto_ctx_safety_pct, memory_ctx });
        }
    }

    if (server_config.request_timeout_sec > 0) {
        log.info("Request timeout: {d}s\n", .{server_config.request_timeout_sec});
    }
    const model_ctx = config.contextCap();
    if (model_ctx > 0) {
        log.info("Model context length: {d} tokens\n", .{model_ctx});
    }
    if (server_config.default_reasoning_budget >= 0) {
        log.info("Reasoning budget: {d} tokens\n", .{server_config.default_reasoning_budget});
    } else {
        log.info("Reasoning budget: unlimited\n", .{});
    }
    if (launchMaxTokensDefault() > 0) {
        log.info("default max_tokens for omitted requests: {d}\n", .{launchMaxTokensDefault()});
    }
    if (scheduler.dflash != null) {
        log.info("DFlash speculative decoding: ENABLED (block_size={d}; default for new requests)\n", .{scheduler.drafter_block_size});
    }
    if (config.isGlm5()) {
        log.info("MTP: off (GLM native serving has no integrated MTP head)\n", .{});
    } else if (config.expert_streaming and !config.stream_mtp_head) {
        log.info("MTP: off under expert streaming (--mtp keeps the head resident)\n", .{});
    } else if (server_config.default_force_mtp) {
        log.info("MTP: forced ON for MoE targets (--mtp; default for new requests)\n", .{});
    }
    if (mtpHeadKvLine(config, transformer_mod.Transformer.mtp_head_kv_quant_flag)) |line| log.info("{s}\n", .{line});
    if (qwenGdnLine(config, model_mod.qwen_gdn_f32)) |line| log.info("{s}\n", .{line});
    if (generate_mod.max_mtp_ctx != 0) {
        log.info("MTP context ceiling: {d} tokens (--max-mtp-ctx; requests past it decode serially)\n", .{generate_mod.max_mtp_ctx});
    }
    log.info("\nServer listening on http://{s}:{d}\n", .{ host, listen_port });
    if (g_api_key != null) {
        log.info("API key auth: ENABLED for non-loopback requests (localhost is trusted; /health and the chat page stay open)\n", .{});
    }
    log.info("  GET  /, /chat (chat page)\n", .{});
    log.info("  GET  /health\n", .{});
    log.info("  GET  /props\n", .{});
    log.info("  GET  /v1/models\n", .{});
    log.info("  POST /v1/chat/completions\n", .{});
    log.info("  POST /v1/completions\n", .{});
    log.info("  POST /v1/messages (Anthropic)\n", .{});
    log.info("  POST /v1/messages/count_tokens (Anthropic)\n", .{});
    log.info("  POST /v1/responses (OpenAI Responses)\n", .{});
    log.info("  POST /v1/responses/compact\n", .{});
    log.info("  GET  /v1/responses/{{id}}\n", .{});
    log.info("  DEL  /v1/responses/{{id}}\n", .{});
    log.info("  POST /tokenize\n", .{});
    log.info("  POST /detokenize\n\n", .{});
    var chat_url_buf: [96]u8 = undefined;
    log.info("chat in your browser: {s}\n\n", .{chatPageUrl(&chat_url_buf, host, listen_port)});

    // Print system metrics once at startup
    const rss = metrics.getAppRssMb();
    if (rss >= 1024) {
        log.info("RSS: {d}.{d}G  Mem: {d}%  CPU: {d}%  GPU: {d}%\n", .{
            rss / 1024,             (rss % 1024) * 10 / 1024,
            metrics.getSysMemPct(), metrics.getCpuPct(),
            metrics.getGpuPct(),
        });
    } else {
        log.info("RSS: {d}M  Mem: {d}%  CPU: {d}%  GPU: {d}%\n", .{
            rss, metrics.getSysMemPct(), metrics.getCpuPct(), metrics.getGpuPct(),
        });
    }

    var poll_fds = [_]std.posix.pollfd{.{
        .fd = server.socket.handle,
        .events = std.posix.POLL.IN,
        .revents = 0,
    }};

    while (!shutdown_requested.load(.acquire)) {
        // Poll with 1-second timeout so we can check shutdown flag
        const poll_result = std.posix.poll(&poll_fds, 1000) catch |err| {
            if (shutdown_requested.load(.acquire)) break;
            log.err("poll error: {}\n", .{err});
            continue;
        };
        if (poll_result == 0) continue; // timeout, re-check shutdown flag
        if (shutdown_requested.load(.acquire)) break;

        const accepted_stream = server.accept(io) catch |err| {
            if (shutdown_requested.load(.acquire)) break;
            log.err("accept error: {}\n", .{err});
            continue;
        };

        // When the scheduler is engaged AND the model is pure-attention,
        // Spawn a per-connection thread so HTTP I/O for one request can
        // overlap with another request's generation. mlx ops (forward
        // passes, eval, etc.) live exclusively on the scheduler's inference
        // thread, so the connection thread NEVER touches `xfm.s` directly —
        // it parses HTTP, encodes the prompt, calls `scheduler.submit`,
        // reads tokens via `slot.waitNext`, and writes the response.
        const args = allocator.create(ConnThreadArgs) catch |err| {
            log.err("conn thread args alloc failed: {}\n", .{err});
            accepted_stream.close(io);
            continue;
        };
        args.* = .{
            .allocator = allocator,
            .accepted_stream = accepted_stream,
            .io = io,
        };
        // Count the thread before spawning so the shutdown drain can't miss a
        // thread that's about to start; the thread decrements on return.
        _ = active_conn_threads.fetchAdd(1, .acq_rel);
        const conn_thread = std.Thread.spawn(.{}, handleConnectionThread, .{args}) catch |err| {
            log.err("spawn conn thread failed: {}\n", .{err});
            _ = active_conn_threads.fetchSub(1, .acq_rel);
            accepted_stream.close(io);
            allocator.destroy(args);
            continue;
        };
        // Detach: a joinable handle dropped per request leaks a stack region; no spawn site may discard its handle.
        conn_thread.detach();
    }

    // Shutdown ordering (prevents a SIGSEGV in Scheduler.complete): the accept
    // loop has exited. Cancel every in-flight slot so its connection thread
    // unblocks from waitNext and runs its `defer complete(...)`, then WAIT for
    // all connection threads to finish before returning — `serve`'s caller runs
    // `scheduler.deinit` on return, which frees the slot queues that an
    // in-flight `complete()` is still touching. The inference thread is still
    // alive here (deinit joins it only after we return), so cancelled slots
    // settle promptly. Bounded so a wedged thread can't hang shutdown forever.
    if (global_scheduler) |sch| sch.cancelAllInFlight();
    {
        var waited_ms: u64 = 0;
        var idle_fds = [_]std.posix.pollfd{};
        while (active_conn_threads.load(.acquire) > 0 and waited_ms < 30_000) {
            _ = std.posix.poll(&idle_fds, 20) catch {}; // empty fd set → 20ms sleep
            waited_ms += 20;
        }
        const remaining = active_conn_threads.load(.acquire);
        if (remaining > 0)
            log.warn("shutdown: {d} connection thread(s) still active after {d}ms — proceeding\n", .{ remaining, waited_ms });
    }

    deinitGlobalResponseStore();

    log.info("\nShutting down gracefully...\n", .{});
}

fn handleConnectionThread(args: *ConnThreadArgs) void {
    // Decrement LAST (declared first → runs after every other defer), so the
    // shutdown drain in `serve` only sees this thread leave once it has fully
    // returned from `complete()` and closed its socket.
    defer _ = active_conn_threads.fetchSub(1, .acq_rel);
    var conn: Conn = undefined;
    Conn.init(&conn, args.accepted_stream, args.io);
    defer conn.close();
    handleConnection(args.allocator, &conn) catch |err| {
        switch (err) {
            error.WriteFailed, error.ReadFailed => {
                // error.WriteFailed/ReadFailed collapses BrokenPipe + ConnectionResetByPeer
                // + other low-level errors. Surface the actual cause from write_state.err /
                // read_state.err so debug logs distinguish "client hung up" from real bugs.
                if (err == error.WriteFailed) {
                    if (conn.write_state.err) |we| {
                        log.debug("  -> client disconnected (write: {s})\n", .{@errorName(we)});
                    } else {
                        log.debug("  -> client disconnected (write)\n", .{});
                    }
                } else {
                    if (conn.read_state.err) |re| {
                        log.debug("  -> client disconnected (read: {s})\n", .{@errorName(re)});
                    } else {
                        log.debug("  -> client disconnected (read)\n", .{});
                    }
                }
            },
            else => log.err("  -> error: {}\n", .{err}),
        }
    };
    args.allocator.destroy(args);
}

/// A client gets this long between bytes of its request, and in all for its head: an idle or
/// dripping connection otherwise holds its thread forever.
const REQUEST_READ_LIMIT_MS: i64 = 30_000;

pub fn handleConnection(
    allocator: std.mem.Allocator,
    stream: *Conn,
) !void {
    // Plan 05: resolve which model this request targets. The registry was
    // set up in `serve()`; per-POST routing happens after we read the body
    // and parse out the optional `"model"` field. For Phase C we use the
    // default model for everything (only one is loaded), but the plumbing
    // is in place for Phase D's hot-load path.
    const registry = global_registry orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Server not ready", 503);
        return;
    };
    const raw = (try readRequestWithin(allocator, stream, REQUEST_READ_LIMIT_MS)) orelse return;
    defer allocator.free(raw.buf);
    const total_read = raw.len;
    const header_end_pos = raw.header_end;
    const request = raw.buf[0..total_read];
    const first_line_end = std.mem.indexOf(u8, request, "\r\n") orelse return;
    const first_line = request[0..first_line_end];

    var line_iter = std.mem.splitScalar(u8, first_line, ' ');
    const method = line_iter.next() orelse return;
    const raw_path = line_iter.next() orelse return;
    // Strip query string for route matching (e.g. /v1/messages?beta=true -> /v1/messages)
    const path = if (std.mem.indexOf(u8, raw_path, "?")) |qpos| raw_path[0..qpos] else raw_path;
    const request_body = if (total_read > header_end_pos) request[header_end_pos..total_read] else "";
    logHttpRequest(method, raw_path, request_body);

    // ── API-key auth gate. When --api-key is set, every NON-LOOPBACK request
    //    requires the key (the OpenAI/Anthropic APIs AND the metrics feed) —
    //    EXCEPT `/health`, CORS preflight and the chat page (it holds no data and asks for the key itself),
    //    which load balancers and browsers must reach unauthenticated. Loopback is
    //    trusted (the local app connects via 127.0.0.1), so it's exempt: the app
    //    + a local browser never need credentials, and the key protects network
    //    exposure. Browser-facing pages get a `WWW-Authenticate: Basic`
    //    challenge so a browser prompts for the key; API clients
    //    pass Bearer / x-api-key. null key ⇒ fully open. See `apiKeyAuthorized`.
    if (apiKeyGateApplies(g_api_key != null, g_api_key_strict, peerIsLoopback(stream)) and
        !std.mem.eql(u8, method, "OPTIONS") and
        !(std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/health")) and
        !(std.mem.eql(u8, method, "GET") and isChatPagePath(path)) and
        !apiKeyAuthorized(request[0..header_end_pos], raw_path))
    {
        log.debug("{s} {s} -> 401 (missing/invalid API key)\n", .{ method, path });
        try sendUnauthorized(stream);
        return;
    }

    if (chat_mod.jsonNestingExceeds(request_body, chat_mod.max_json_nesting)) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "JSON nesting is too deep", 400);
        return;
    }

    if (std.mem.eql(u8, path, "/v1/tools")) {
        if (!std.mem.eql(u8, method, "POST")) {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "Use POST for tools", 405);
            return;
        }
        try handleWebTools(allocator, stream, request[0..header_end_pos], request_body);
        return;
    }

    // ── Plan 05: routes that don't depend on a loaded model (connectivity
    //    probes + CORS preflight + listing endpoints). Handle these BEFORE
    //    `scheduler.ensureLoaded` so they don't trigger a cold load of the
    //    default model just to read metadata. `/v1/models` and the GET-side
    //    of the Responses API are pure registry/store reads — no model
    //    needed.
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/health")) {
        log.debug("GET  /health -> 200\n", .{});
        try sendResponse(stream, "200 OK", "application/json", "{\"status\":\"ok\"}");
        return;
    }
    // Prometheus scrape endpoint. 503 when --metrics is off. Behind the global
    // API-key gate above when --api-key is set (auth already enforced here).
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/metrics")) {
        if (instr.current()) |m| {
            var out: std.Io.Writer.Allocating = .init(allocator);
            defer out.deinit();
            try instr.renderPrometheus(m, &out.writer);
            // Quiet: don't dump the ~3 KB scrape body every 2 s at debug level.
            try sendResponseQuiet(stream, "200 OK", "text/plain; version=0.0.4; charset=utf-8", out.written());
        } else {
            try sendResponse(stream, "503 Service Unavailable", "text/plain", "metrics not enabled (start with --metrics, or POST /metrics/enable)");
        }
        return;
    }
    // JSON feed of the same counters. Behind the global API-key gate above
    // when --api-key is set.
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/metrics.json")) {
        if (instr.current()) |m| {
            var out: std.Io.Writer.Allocating = .init(allocator);
            defer out.deinit();
            var sessions: [2 * instr.MAX_SESSIONS]instr.Session = undefined;
            try instr.renderJson(m, liveSessions(registry, &sessions), &out.writer);
            // Quiet: dashboards poll this ~1 Hz — don't log the body.
            try sendResponseQuiet(stream, "200 OK", "application/json", out.written());
        } else {
            try sendResponse(stream, "503 Service Unavailable", "application/json", "{\"error\":\"metrics not enabled — start with --metrics, or POST /metrics/enable\"}");
        }
        return;
    }
    if (std.mem.eql(u8, method, "OPTIONS")) {
        log.debug("OPTIONS {s} -> 204\n", .{path});
        try sendResponse(stream, "204 No Content", "text/plain", "");
        return;
    }
    if (crossOriginRefused(method, path, request[0..header_end_pos], listen_host, listen_port)) {
        log.debug("{s} {s} -> 403 (foreign Origin)\n", .{ method, path });
        try sendErrorResponse(allocator, stream, "403 Forbidden", "invalid_request_error", "This route accepts browser requests from this server's own page only", 403);
        return;
    }
    if (std.mem.eql(u8, path, "/metrics/enable")) {
        if (!std.mem.eql(u8, method, "POST")) {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "/metrics/enable answers POST only", 405);
            return;
        }
        try handleMetricsEnable(stream);
        return;
    }
    if (std.mem.eql(u8, path, "/v1/update")) {
        if (!std.mem.eql(u8, method, "POST")) {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "/v1/update answers POST only", 405);
            return;
        }
        try handleUpdate(allocator, stream, request[0..header_end_pos], raw_path);
        return;
    }
    // Every method is answered here: past this point a request resolves (and may cold-load) a model.
    if (isChatPagePath(path)) {
        if (std.mem.eql(u8, method, "GET")) {
            try sendResponseQuiet(stream, "200 OK", "text/html; charset=utf-8", chat_page_html);
        } else {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "The chat page answers GET only", 405);
        }
        return;
    }
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/models")) {
        log.debug("GET  /v1/models -> 200\n", .{});
        try handleModels(allocator, stream);
        return;
    }
    if (std.mem.eql(u8, method, "GET") and std.mem.startsWith(u8, path, "/v1/responses/")) {
        const id = path["/v1/responses/".len..];
        try handleResponsesGet(allocator, stream, id);
        return;
    }
    if (std.mem.eql(u8, method, "DELETE") and std.mem.startsWith(u8, path, "/v1/responses/")) {
        const id = path["/v1/responses/".len..];
        try handleResponsesDelete(allocator, stream, id);
        return;
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/responses/compact")) {
        // Compaction is a pure data transformation — no model load needed.
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleResponsesCompact(allocator, stream, body);
        return;
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/load-model")) {
        // Phase E: explicit cold-load. Keep strict semantics — unknown ids
        // get a 404 here even though the main dispatch path below falls
        // back to default for SDK compatibility.
        log.debug("POST /v1/load-model -> 200\n", .{});
        try handleLoadModelStrict(allocator, stream, request_body);
        return;
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/unload-model")) {
        // Free a model's resident GPU state (the app's load→generate→unload
        // default). The stub stays registered so it can reload. Idempotent.
        log.debug("POST /v1/unload-model -> 200\n", .{});
        try handleUnloadModelStrict(allocator, stream, request_body);
        return;
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/models/rescan")) {
        // Absorb models downloaded AFTER boot (the Model Browser pulls while
        // the server runs; discovery only walks the roots at startup).
        try handleModelsRescan(allocator, stream);
        return;
    }

    // ── Plan 05 Phase D: resolve the request's target model via the
    //    scheduler (which delegates the fast path to registry.ensureLoaded
    //    and handles cold-load + eviction internally). Absent `model`
    //    field → default. Unknown id → 404. Unloaded id with no room and
    //    no LRU victim → 503 (NotEnoughMemory). Block-until-loaded is
    //    intentional; clients targeting an unloaded model should set a
    //    longer request timeout (or hit /v1/load-model first).
    const scheduler = global_scheduler orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    // Strip whatever the client passed in `"model":"..."` — except when the
    // id or path matches one we've registered (`routeRequestModel`): we honor
    // it and route. The OpenAI / Anthropic ecosystem commonly sends marketing
    // names like "gpt-4" or "claude-opus-4-x" expecting the local server to
    // just respond with whatever it has loaded; the multi-model registry's
    // strict-id semantics are opt-in by sending an id we registered.
    // Swift's JSON writers escape '/' as '\\/' (Foundation does this
    // unconditionally, both JSONSerialization and JSONEncoder). That is legal
    // JSON, but the id is read out of the RAW body, so an HF-style `org/repo`
    // id arrived as `Runpod\\/FLUX.2-…`, matched nothing in the registry, and
    // silently fell through to the default model — a 400 "does not support
    // this media modality" on the gen endpoints, and a wrong-model answer on
    // chat (live from the iPhone app, 2026-07-25). Canonicalise once, here,
    // so every consumer below (peek, ensureLoaded) sees the real id.
    var model_id_buf: [512]u8 = undefined;
    var query_model_buf: [512]u8 = undefined;
    var requested_model_id = unescapeJsonSlashes(
        &model_id_buf,
        parseModelFromBody(request_body) orelse queryModel(&query_model_buf, raw_path) orelse "",
    );
    const home: ?[]const u8 = if (std.c.getenv("HOME")) |h| std.mem.span(h) else null;
    switch (routeRequestModel(registry, requested_model_id, home)) {
        .entry => |id| requested_model_id = id,
        .default => requested_model_id = "",
        .not_found => {
            const msg = "No model is registered at that path; POST /v1/load-model with the absolute path registers it";
            if (isAnthropicRoute(path)) {
                try sendAnthropicError(allocator, stream, "not_found_error", msg, 404);
            } else {
                try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", msg, 404);
            }
            return;
        },
    }
    // Text-gen route aimed at a KNOWN non-text model: reject before
    // ensureLoaded, or the request cold-loads a multi-GB media model just
    // to earn its 400. The post-load `text_gen_reject` below stays the
    // authoritative crash barrier (this peek can't see `--model` primaries
    // with no arch hint until they're resident).
    if (requested_model_id.len > 0 and isTextGenRoute(method, path)) {
        // Entry and snapshot under ONE hold — plain `peek` unlocks before it
        // returns. `arch_hint` outlives the unlock: entry-owned, freed in
        // `deinit`.
        const target: ?TextGenTarget = blk: {
            registry.mutex.lockUncancelable(stream.io);
            defer registry.mutex.unlock(stream.io);
            const peeked = registry.peekLocked(requested_model_id) orelse break :blk null;
            break :blk textGenTargetOf(peeked);
        };
        if (target) |t| {
            if (textGenRejectReason(t)) |reason| {
                if (isAnthropicRoute(path)) {
                    try sendAnthropicError(allocator, stream, "invalid_request_error", reason, 400);
                } else {
                    try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
                }
                return;
            }
        }
    }
    // An endpoint that doesn't exist is a 404, and it must cost NOTHING to say
    // so. Resolution runs ahead of dispatch, so this has to be answered here
    // rather than inside one of `ensureLoaded`'s error arms: with a resolvable
    // `model` in the body, resolution SUCCEEDS — cold-loading the checkpoint —
    // and the unknown path only 404s afterwards. `POST /v1/load` (the route is
    // /v1/load-model) cost 2m42s and 121 GB resident for a typo, and the same
    // one-liner pins the box for anyone who sends it. Placed above every load.
    if (!routeExists(path)) {
        try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "Unknown endpoint", 404);
        return;
    }
    // `GET /props` is a STATUS read, and falling into `ensureLoaded` makes a
    // poll cold-load the model. With idle eviction on that is a loop: the sweep
    // hands the memory back and the next poll takes it straight again (the app's
    // tray polls every 3s). Nothing resident means nothing to report — the same
    // body the `NoDefaultModel` arm below already sends.
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/props")) {
        const resident = if (registry.resolveEntry(requested_model_id)) |e| blk: {
            registry.mutex.lockUncancelable(stream.io);
            defer registry.mutex.unlock(stream.io);
            break :blk e.state == .ready;
        } else |_| false;
        if (!resident) {
            try handlePropsNoModel(allocator, stream);
            return;
        }
    }
    const lm = scheduler.ensureLoaded(requested_model_id) catch |err| switch (err) {
        error.UnknownModelId => {
            try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "Unknown model id", 404);
            return;
        },
        error.NotLoaded => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "model_not_loaded", "Requested model is not currently loaded", 503);
            return;
        },
        error.NoDefaultModel => {
            // Headless gen-only boot (media models resident, no default chat
            // model): /props keeps answering with the live memory counters —
            // the app's tray polls it, and a 503 here read as "0 MB".
            if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/props")) {
                try handlePropsNoModel(allocator, stream);
                return;
            }
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "no_model", "No default model configured", 503);
            return;
        },
        error.LoadFailed => {
            try sendLoadFailedResponse(allocator, stream, scheduler, requested_model_id);
            return;
        },
        error.Shutdown => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "shutting_down", "Server is shutting down", 503);
            return;
        },
        // Other errors (CPU-side preload failures like FileNotFound on a
        // missing tokenizer.json, JSON parse errors on a malformed
        // config.json, etc.) bubble out of `preloadCpuState`. A refusal the
        // registry preserves by name answers 503 under that name; anything
        // else is a 500 with the error name, never a hung connection.
        else => {
            if (try sendNamedLoadRefusal(allocator, stream, scheduler, requested_model_id, err)) return;
            log.warn("  -> 500 ({s}) while resolving model\n", .{@errorName(err)});
            const msg = std.fmt.allocPrint(allocator, "Failed to load model: {s}", .{@errorName(err)}) catch {
                try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", "Failed to load model", 500);
                return;
            };
            defer allocator.free(msg);
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", msg, 500);
            return;
        },
    };
    // A status read must not look like use — see `isStatusRoute`.
    const status_read = isStatusRoute(method, path);
    defer if (status_read) scheduler.releaseStatus(lm) else scheduler.release(lm);

    // A model loaded on demand freezes its own auto-context here, for the same
    // reason the `--model` primary does at startup: the number we advertise is
    // what clients budget against, and each model has its own dims. Idempotent,
    // and a no-op under `--ctx-size`.
    if (lm.config) |c| _ = pinAutoContext(c);
    warnUserWireLimit();

    // ── Phase C: handlers take `lm` directly and extract their own
    //    locals. handleConnection only needs one decision for every
    //    text-generation route: media (image/audio/video/3D) models get a clean 400 instead of reaching a prefill that has
    //    no transformer to run (the SIGSEGV class documented on
    //    textGenRejectReason).
    const text_gen_reject: ?[]const u8 = textGenRejectReason(textGenTargetOf(lm));

    // ── WebSocket upgrade for /v1/responses ──
    // Detect Upgrade: websocket BEFORE the regular route dispatch so the
    // GET method (used for the upgrade handshake) doesn't fall through to
    // the 404 branch.
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/responses") and ws_mod.isUpgrade(request[0..header_end_pos])) {
        if (text_gen_reject) |reason| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        try handleResponsesWebSocket(allocator, stream, request[0..header_end_pos], lm);
        return;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/props")) {
        log.debug("GET  /props -> 200\n", .{});
        try handleProps(allocator, stream, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/chat/completions")) {
        if (text_gen_reject) |reason| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleChatCompletions(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/completions")) {
        if (text_gen_reject) |reason| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleCompletions(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/messages")) {
        if (text_gen_reject) |reason| {
            try sendAnthropicError(allocator, stream, "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleAnthropicMessages(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/messages/count_tokens")) {
        if (text_gen_reject) |reason| {
            try sendAnthropicError(allocator, stream, "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleAnthropicCountTokens(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/responses")) {
        if (text_gen_reject) |reason| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleResponses(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/tokenize")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleTokenize(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/detokenize")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleDetokenize(allocator, stream, body, lm);
    } else {
        log.warn("{s} {s} -> 404\n", .{ method, path });
        try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "The requested endpoint does not exist", null);
    }
}

/// Percentage of the memory-derived ceiling we actually admit as context when
/// `--ctx-size` is absent.
///
/// `computeMaxSafeContext` returns the largest context that FITS right now.
/// Serving at exactly that leaves nothing for the prefix cache to grow into, a
/// second model to load beside this one, or another app to take RAM — and the
/// Metal OOM it would eventually hit is uncatchable (see the auto-context
/// gotcha). Reserve headroom instead.
const auto_ctx_safety_pct: u32 = 93;

/// PURE: apply the safety margin and round down to a 1024 boundary so the
/// number reads sanely in logs and client configs. Never returns 0.
///
/// Takes the MEMORY-derived ceiling only. The model's own `max_position_embeddings`
/// is applied afterwards, un-margined: when the checkpoint's max is the binding
/// constraint there is nothing to reserve memory headroom against, and shaving
/// the margin off a 131,072-token model that comfortably fits in RAM just throws
/// context away.
fn safeAutoContext(raw: u32) u32 {
    const scaled: u64 = (@as(u64, raw) * auto_ctx_safety_pct) / 100;
    const rounded: u64 = (scaled / 1024) * 1024;
    if (rounded == 0) return @intCast(@max(scaled, 1));
    return @intCast(rounded);
}

/// This model's auto-context: memory ceiling with headroom, then capped by what
/// the checkpoint actually supports. `contextCap` is the rope-derived window:
/// `max_position_embeddings`, or `original_max_position_embeddings × factor` for
/// a YaRN-scaled checkpoint — the same derivation vLLM applies to
/// `max_model_len`, because a position past the scaled window aliases back
/// inside the ramp rather than reading as a longer distance.
/// The context the server advertises, from the largest context memory alone allows (93%
/// margin on the memory number, checkpoint cap afterwards). Shared by the load-time session
/// bill and the boot-time sizer so the two cannot spell the relation two ways.
fn autoContextFrom(memory_ctx: u32, ctx_cap: u32) u32 {
    const with_headroom = safeAutoContext(memory_ctx);
    return if (ctx_cap > 0) @min(with_headroom, ctx_cap) else with_headroom;
}

fn autoContextFor(config: *const model_mod.ModelConfig) u32 {
    if (billedAutoContext(config)) return billedAdvertisedContext(config, computeMemoryContext(config));
    return autoContextFrom(computeMemoryContext(config), config.contextCap());
}

/// Freeze this model's auto-context at load time. Idempotent; a no-op (and
/// never a write) when `--ctx-size` pinned it explicitly. Returns the value
/// `getEffectiveContextLength` will report from now on.
///
/// Called once per model, at startup for the `--model` primary and right after
/// an on-demand load. The advertised context must not drift: agent CLIs read it
/// once (pi/opencode never re-read `/v1/models`) and budget their own
/// `max_tokens` against it for the rest of the session.
fn pinAutoContext(config: *model_mod.ModelConfig) u32 {
    // Order is load-bearing: the sizer bills THIS chunk's transient reserve and
    // the admission guard bills the same frozen value, so the width has to be
    // resolved before any context is computed from it. Pinned even under an
    // explicit `--ctx-size`, because the guard reads it on every request.
    _ = pinPrefillChunk(config);
    if (manualContext(config) > 0) return manualContext(config);
    if (config.pinned_context == 0) {
        config.pinned_context = autoContextFor(config);
    }
    return config.pinned_context;
}

pub fn getEffectiveContextLength(config: *const model_mod.ModelConfig) u32 {
    if (manualContext(config) > 0) return manualContext(config);
    if (config.pinned_context > 0) return config.pinned_context;
    // Not pinned yet (a discovery stub that was never loaded): compute from
    // current GPU memory rather than a fixed 16K cap.
    return autoContextFor(config);
}

/// Live requests, then each hot-cache entry no live row restored from, each row stamped with its
/// model's effective context limit. The connection-thread reader copies the inference thread's
/// `queue_mu` and `digest_mu` snapshots under those same locks: a
/// Session is POD with an inline model buffer, so the copy stays valid after
/// its slot frees. Rows carry no pointers into scheduler or registry state.
fn liveSessions(registry: *ModelRegistry, buf: *[2 * instr.MAX_SESSIONS]instr.Session) []instr.Session {
    const sch = global_scheduler orelse return buf[0..0];
    sch.queue_mu.lockUncancelable(sch.io);
    const live = sch.live_session_count;
    @memcpy(buf[0..live], sch.live_sessions[0..live]);
    sch.queue_mu.unlock(sch.io);

    sch.digest_mu.lockUncancelable(sch.io);
    const n = instr.appendUnclaimedCached(buf, live, sch.cached_sessions[0..sch.cached_session_count]);
    sch.digest_mu.unlock(sch.io);

    registry.mutex.lockUncancelable(registry.io);
    defer registry.mutex.unlock(registry.io);
    for (buf[0..n]) |*s| {
        const entry = registry.peekLocked(s.model()) orelse continue;
        if (entry.state != .ready) continue;
        if (entry.config) |cfg| s.context_length = getEffectiveContextLength(cfg);
    }
    return buf[0..n];
}

/// Metal's recommended max working-set size for the default device — the real
/// ceiling whose breach throws `[METAL] … Insufficient Memory` from the
/// command-buffer completion handler (which terminates the process: ggml's
/// global std::terminate handler prints the backtrace, but the throw is MLX's).
/// `getMetalBufferLimit()` (75% of physical RAM) over-estimates this on
/// small-RAM Macs — a 16 GB Mac reports ~11.9 GB recommended vs the 12 GB that
/// hw.memsize×0.75 yields — so budgeting against hw.memsize lets auto-context
/// oversubscribe. Falls back to `getMetalBufferLimit()` when the device query
/// is unavailable (CI / non-Metal hosts).
fn getGpuWorkingSetLimit() u64 {
    var dev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&dev);
    defer _ = mlx.mlx_device_free(dev);
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) == 0) {
        var max_rec: usize = 0;
        if (mlx.mlx_device_info_get_size(&max_rec, info, "max_recommended_working_set_size") == 0 and max_rec > 0) {
            return @as(u64, max_rec);
        }
    }
    return getMetalBufferLimit();
}

/// PURE (unit-testable): the real ceiling a NEW MLX allocation must fit under.
///
/// `working_set_limit` (Metal's `max_recommended_working_set_size`) is a STATIC
/// device maximum — it assumes the whole GPU working set is MLX's to claim and
/// is BLIND to memory held by anything else on the machine. `mlx_footprint`
/// (MLX active + reclaimable cache) + `free_system` (physically free RAM) is
/// what MLX can actually reach RIGHT NOW; when another process holds a big
/// chunk of unified memory the second term binds and the ceiling collapses.
///
/// #64 (2026-07): a Claude Code session on a 128 GB Mac ran a docker-compose
/// stack (firecrawl/rabbitmq/postgres/playwright) holding tens of GB. The guard
/// budgeted against the static 115 GB max, admitted a 90 K-token MoE prefill,
/// and Metal threw `Insufficient Memory` from the command-buffer completion
/// handler — an UNCATCHABLE async C++ throw that terminates the process.
/// Capping by real free RAM makes the two prefill guards
/// reject/clamp before that allocation is ever submitted.
fn physicalMemoryCeiling(working_set_limit: u64, mlx_footprint: u64, free_system: u64) u64 {
    return @min(working_set_limit, mlx_footprint +| free_system);
}

/// How far under the enforced wired limit a plan may reach: past the limit Metal returns
/// zeros before an uncatchable abort, so a real transient's worth of margin stays unplanned.
pub const WIRED_LIMIT_MARGIN_BYTES: u64 = 1 << 30;

/// PURE: the floor an explicitly raised `iogpu.wired_limit_mb` puts under the ceiling; 0 when
/// the sysctl is absent or at the macOS default (75% of RAM), which leaves the ceiling as is.
/// A floor, not a cap: the working-set term already tracks the sysctl. The term it lifts is
/// the free-RAM one, which counts other processes' pages as gone; the operator who raised the
/// enforced limit has said the GPU may claim that much. `margin` (`--wired-margin-gib`)
/// is what stays unplanned under the limit: other processes' wired GPU memory, the
/// allocator cache and the estimator's slack; past the limit Metal returns zeros.
pub fn wiredLimitFloor(wired_limit: u64, total_ram: u64, margin: u64) u64 {
    if (wired_limit == 0 or total_ram == 0) return 0;
    if (wired_limit <= total_ram * 75 / 100) return 0; // the macOS default: no declaration
    return @min(wired_limit -| margin, total_ram -| margin);
}

pub fn parseWiredMarginGib(raw: []const u8) error{InvalidWiredMargin}!u64 {
    const n = std.fmt.parseInt(u32, raw, 10) catch return error.InvalidWiredMargin;
    if (n < 1 or n > 32) return error.InvalidWiredMargin;
    return @as(u64, n) << 30;
}

pub fn parseSsdBudgetGb(raw: []const u8) error{InvalidSsdBudgetGb}!u64 {
    const gb = std.fmt.parseInt(u64, raw, 10) catch return error.InvalidSsdBudgetGb;
    if (gb == 0 or gb > 512) return error.InvalidSsdBudgetGb;
    return std.math.mul(u64, gb, 1 << 30) catch return error.InvalidSsdBudgetGb;
}

/// PURE: the residency limit the SSD budget is admitted against — `iogpu.wired_limit_mb`
/// when the owner declared one, else Metal's recommended working set.
pub fn ssdBudgetCeiling(wired_limit: u64, working_set: u64) u64 {
    return if (wired_limit > 0) wired_limit else working_set;
}

pub fn ssdBudgetAdmits(budget_bytes: u64, kv_bytes: u64, limit: u64) bool {
    return budget_bytes +| kv_bytes <= limit;
}

/// PURE: the `iogpu.wired_limit_mb` value that would admit this plan — quoted by the refusal.
pub fn wiredLimitMbForBudget(budget_bytes: u64, kv_bytes: u64) u64 {
    const MB: u64 = 1024 * 1024;
    return (budget_bytes +| kv_bytes +| (MB - 1)) / MB;
}

/// `--wired-margin-gib` in bytes: what stays unplanned under a raised wired limit.
pub var wired_limit_margin_bytes: u64 = WIRED_LIMIT_MARGIN_BYTES;

/// GPU memory the wired limit counts that MLX does not own: other processes' Metal allocations
/// (WindowServer, every app with a layer) and the driver's. `AGXAccelerator` "In use system
/// memory" is system-wide, so this is that counter less MLX's active + cache, read once at
/// startup before MLX holds a model (`measureGpuFloor`). A command buffer fails once MLX's peak
/// passes the wired limit less this floor; above it the counter is elastic.
pub var gpu_floor_bytes = std.atomic.Value(u64).init(0);

/// What the bill may still be wrong by once the floor is planned, including sushi's own non-MLX
/// Metal objects that arrive after the startup reading.
pub const GPU_FLOOR_SLACK_BYTES: u64 = 512 << 20;

/// PURE: what stays unplanned under a declared wired limit: the operator's margin, or the
/// measured floor plus the bill's slack when that is larger.
pub fn wiredUnplannedBytes(margin: u64, gpu_floor: u64) u64 {
    if (gpu_floor == 0) return margin;
    return @max(margin, gpu_floor +| GPU_FLOOR_SLACK_BYTES);
}

/// PURE: the floor from one idle reading; 0 when IOKit gave nothing.
pub fn gpuFloorFrom(in_use_system: u64, mlx_active: u64, mlx_cache: u64) u64 {
    if (in_use_system == 0) return 0;
    return in_use_system -| (mlx_active +| mlx_cache);
}

/// Read the floor once, at startup, before MLX holds a model: the only moment the counter is
/// clean both ways: right after a load it lags low while the weights are still being wired, and
/// after an unload it lags high while freed pages unwire.
pub fn measureGpuFloor() void {
    var active: usize = 0;
    var cache: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_cache_memory(&cache);
    const in_use = metrics.getGpuInUseSystemMemBytes();
    const floor = gpuFloorFrom(in_use, active, cache);
    gpu_floor_bytes.store(floor, .monotonic);
    log.info("[mem] GPU floor {d} MB (system GPU in use {d} MB, MLX active {d} MB + cache {d} MB)\n", .{
        floor >> 20,
        in_use >> 20,
        active >> 20,
        cache >> 20,
    });
}

/// The MB-per-buffer the prefill terms were measured at (`prefillStreamBytesPerToken` on an
/// M4 Max, MLX's Max/Ultra default). MLX commits a command buffer once its distinct inputs pass
/// `MLX_MAX_MB_PER_BUFFER`, and every op's inputs live until its buffer completes, up to 10
/// buffers in flight (`mlx/backend/metal/eval.cpp`, `MAX_ACTIVE_TASKS`).
pub const BILL_FIT_MB_PER_BUFFER: u64 = 50;

/// MLX's per-buffer defaults by architecture suffix (`mlx/backend/metal/device.cpp`).
pub fn mlxDefaultMbPerBuffer(arch: []const u8) u64 {
    if (arch.len == 0) return 40;
    return switch (arch[arch.len - 1]) {
        'p', 'g' => 40,
        's', 'd' => 50,
        else => 40,
    };
}

pub fn mlxDefaultOpsPerBuffer(arch: []const u8) u64 {
    if (arch.len == 0) return 40;
    return switch (arch[arch.len - 1]) {
        'p' => 20,
        'g' => 40,
        's', 'd' => 50,
        else => 40,
    };
}

/// PURE: MLX's reading of its env override (a positive integer replaces the default).
pub fn mlxPerBufferFromEnv(raw: ?[]const u8, default: u64) u64 {
    const s = raw orelse return default;
    const n = std.fmt.parseInt(u64, std.mem.trim(u8, s, " \t"), 10) catch return default;
    return if (n > 0) n else default;
}

/// PURE: prefill bytes the command-buffer cadence keeps alive beyond what the bill's fitted
/// terms already hold; ops per buffer does not move the peak. A narrower cap is not credited.
pub fn cadenceInflightBytes(mb_per_buffer: u64) u64 {
    return (mb_per_buffer -| BILL_FIT_MB_PER_BUFFER) * (1 << 20) * 9 / 8;
}

test "the bill plans the non-MLX GPU floor and the command-buffer size" {
    const t = std.testing;
    const mb: u64 = 1 << 20;
    // An idle reading: 28.96 GB in use, 28.28 active + 0.06 cache -> a 0.62 GB floor.
    try t.expectEqual(@as(u64, 620 * mb), gpuFloorFrom(28_960 * mb, 28_280 * mb, 60 * mb));
    try t.expectEqual(@as(u64, 0), gpuFloorFrom(0, 28_280 * mb, 0));
    try t.expectEqual(@as(u64, 0), gpuFloorFrom(100 * mb, 200 * mb, 0));
    // No reading keeps the operator's margin; a floor above margin - slack widens it.
    try t.expectEqual(WIRED_LIMIT_MARGIN_BYTES, wiredUnplannedBytes(WIRED_LIMIT_MARGIN_BYTES, 0));
    try t.expectEqual(WIRED_LIMIT_MARGIN_BYTES, wiredUnplannedBytes(WIRED_LIMIT_MARGIN_BYTES, 256 * mb));
    try t.expectEqual(620 * mb + GPU_FLOOR_SLACK_BYTES, wiredUnplannedBytes(WIRED_LIMIT_MARGIN_BYTES, 620 * mb));
    // The default cadence bills nothing extra; 1000 MB bills about 1.07 GB.
    try t.expectEqual(@as(u64, 0), cadenceInflightBytes(BILL_FIT_MB_PER_BUFFER));
    try t.expectEqual(@as(u64, 0), cadenceInflightBytes(40));
    try t.expectEqual(@as(u64, 950 * mb * 9 / 8), cadenceInflightBytes(1000));
    // MLX's table and env reading.
    try t.expectEqual(@as(u64, 50), mlxDefaultMbPerBuffer("applegpu_g13s"));
    try t.expectEqual(@as(u64, 50), mlxDefaultMbPerBuffer("applegpu_g13d"));
    try t.expectEqual(@as(u64, 40), mlxDefaultMbPerBuffer("applegpu_g13g"));
    try t.expectEqual(@as(u64, 20), mlxDefaultOpsPerBuffer("applegpu_g14p"));
    try t.expectEqual(@as(u64, 1000), mlxPerBufferFromEnv("1000", 50));
    try t.expectEqual(@as(u64, 50), mlxPerBufferFromEnv("0", 50));
    try t.expectEqual(@as(u64, 50), mlxPerBufferFromEnv("lots", 50));
    try t.expectEqual(@as(u64, 50), mlxPerBufferFromEnv(null, 50));
}

/// The effective `MLX_MAX_MB_PER_BUFFER`, resolved once at startup.
pub var mlx_mb_per_buffer: u64 = BILL_FIT_MB_PER_BUFFER;

/// Main thread, before other threads: resolve MLX's command-buffer cadence the way MLX does
/// and say what it is, since MLX reads the env once and reports nothing.
pub fn resolveMlxCadence() void {
    var arch_z: [*:0]const u8 = "";
    var dev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&dev);
    defer _ = mlx.mlx_device_free(dev);
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) == 0) _ = mlx.mlx_device_info_get_string(&arch_z, info, "architecture");
    const arch = std.mem.span(arch_z);
    const env_mb: ?[]const u8 = if (std.c.getenv("MLX_MAX_MB_PER_BUFFER")) |v| std.mem.span(v) else null;
    const env_ops: ?[]const u8 = if (std.c.getenv("MLX_MAX_OPS_PER_BUFFER")) |v| std.mem.span(v) else null;
    mlx_mb_per_buffer = mlxPerBufferFromEnv(env_mb, mlxDefaultMbPerBuffer(arch));
    const ops = mlxPerBufferFromEnv(env_ops, mlxDefaultOpsPerBuffer(arch));
    log.info("[mem] MLX command buffers: {d} ops / {d} MB ({s}, {s}); prefill bill +{d} MB in flight\n", .{
        ops,
        mlx_mb_per_buffer,
        if (env_mb != null or env_ops != null) "env" else "default",
        if (arch.len > 0) arch else "unknown arch",
        cadenceInflightBytes(mlx_mb_per_buffer) >> 20,
    });
    measureGpuFloor();
}

/// PURE: the ceiling with the wired-limit floor applied. `wired_floor == 0` returns
/// `physicalMemoryCeiling` byte for byte. A declared limit IS the ceiling, its margin included:
/// the working-set term is the limit itself, so on a quiet box it would plan into the margin.
pub fn gpuCeilingWithWiredFloor(
    working_set_limit: u64,
    mlx_footprint: u64,
    free_system: u64,
    wired_floor: u64,
) u64 {
    if (wired_floor > 0) return wired_floor;
    return physicalMemoryCeiling(working_set_limit, mlx_footprint, free_system);
}

/// Test seam: `iogpu.wired_limit_mb` in MB; `null` reads the machine.
pub var wired_limit_mb_override: ?u64 = null;

var wired_limit_bytes_cached: u64 = 0;
var wired_limit_read: bool = false;

/// `iogpu.wired_limit_mb` in bytes, read ONCE per process: re-reading it per admission would
/// let the ceiling move under a live request. 0 when the OID is absent.
pub fn wiredLimitBytes() u64 {
    if (wired_limit_mb_override) |mb| return mb *| (1024 * 1024);
    if (wired_limit_read) return wired_limit_bytes_cached;
    var v: u32 = 0;
    var len: usize = @sizeOf(u32);
    wired_limit_bytes_cached = if (sysctlbyname("iogpu.wired_limit_mb", @ptrCast(&v), &len, null, 0) == 0)
        @as(u64, v) * 1024 * 1024
    else
        0;
    wired_limit_read = true;
    return wired_limit_bytes_cached;
}

/// Bytes MLX wires beside the owned weights and slabs: the decode buffer pool (`dsv41_pool_bytes`), the one
/// ledger term the owned-bytes count lacks.
pub const USER_WIRE_MARGIN_BYTES: u64 = 1 << 30;

/// RAM a suggested user-wire limit leaves to the rest of the system.
const USER_WIRE_RAM_FLOOR_BYTES: u64 = 12_000_000_000;

/// PURE: the one-line warning for a load whose wired bytes pass macOS's user-wire cap, else null.
/// `limits` are `vm.global_user_wire_limit` and `vm.user_wire_limit` (0 = unreadable); the lower binds.
/// Past it prefill slows down with no error. The suggested value is the need or the declared
/// `iogpu.wired_limit_mb`, rounded up to a whole GB, never above `total_ram` less a floor.
pub fn userWireWarning(buf: []u8, owned_bytes: u64, limits: [2]u64, wired_limit: u64, total_ram: u64) ?[]const u8 {
    const names = [2][]const u8{ "vm.global_user_wire_limit", "vm.user_wire_limit" };
    const bind: usize = if (limits[0] == 0 or (limits[1] != 0 and limits[1] < limits[0])) 1 else 0;
    const limit = limits[bind];
    if (limit == 0) return null;
    const need = owned_bytes +| USER_WIRE_MARGIN_BYTES;
    if (need <= limit) return null;
    const GB: u64 = 1_000_000_000;
    const cap = total_ram -| USER_WIRE_RAM_FLOOR_BYTES;
    const suggested = @min((@max(need, wired_limit) +| (GB - 1)) / GB * GB, cap / GB * GB);
    const head = "[mem] warning: this load wires ~{d:.1} GB but {s} is {d:.1} GB: prefill will run slower. ";
    const args = .{ @as(f64, @floatFromInt(need)) / 1e9, names[bind], @as(f64, @floatFromInt(limit)) / 1e9 };
    if (suggested < need)
        return std.fmt.bufPrint(buf, head ++ "This Mac cannot wire that much: use a smaller pack or a lower --ssd-budget-gb", args) catch null;
    return std.fmt.bufPrint(buf, head ++ "Raise it: sudo sysctl vm.global_user_wire_limit={d} vm.user_wire_limit={d} (resets at reboot)", args ++ .{ suggested, suggested }) catch null;
}

fn readSysctlU64(name: [*:0]const u8) u64 {
    var v: u64 = 0;
    var len: usize = @sizeOf(u64);
    return if (sysctlbyname(name, @ptrCast(&v), &len, null, 0) == 0 and len == @sizeOf(u64)) v else 0;
}

var user_wire_warned = std.atomic.Value(bool).init(false);

/// Warns once per process, after the weights and expert slabs are mapped, when they pass the user-wire cap.
pub fn warnUserWireLimit() void {
    if (user_wire_warned.cmpxchgStrong(false, true, .monotonic, .monotonic) != null) return;
    var buf: [384]u8 = undefined;
    const limits = [2]u64{ readSysctlU64("vm.global_user_wire_limit"), readSysctlU64("vm.user_wire_limit") };
    if (userWireWarning(&buf, model_mod.ownedBytesTotal(), limits, wiredLimitBytes(), metrics.getTotalMemBytes())) |line|
        log.warn("{s}\n", .{line});
}

var wired_floor_logged = std.atomic.Value(bool).init(false);

/// Long-context models and engaged expert streams honor the operator's wired limit.
/// `null` config (no model resolved yet) takes no floor.
pub fn wiredCeilingFloorFor(config: ?*const model_mod.ModelConfig) u64 {
    return wiredCeilingFloorForRam(config, metrics.getTotalMemBytes());
}

test "MiMo honors the declared wired limit streamed or resident, without enabling long-context paths" {
    const prior = wired_limit_mb_override;
    wired_limit_mb_override = 120000;
    defer wired_limit_mb_override = prior;
    const floor = wiredLimitFloor(120000 * 1024 * 1024, 128 << 30, wired_limit_margin_bytes);
    var c = model_mod.ModelConfig{ .model_type = "mimo_v2", .expert_streaming = true };
    try std.testing.expect(!c.longCtxGated());
    try std.testing.expectEqual(floor, wiredCeilingFloorForRam(&c, 128 << 30));
    c.expert_streaming = false;
    try std.testing.expectEqual(floor, wiredCeilingFloorForRam(&c, 128 << 30));
    // An arch the engine does not serve keeps the free-RAM ceiling.
    const other = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    try std.testing.expectEqual(@as(u64, 0), wiredCeilingFloorForRam(&other, 128 << 30));
}

/// PURE: the floor for a machine with `total_ram` bytes; the wrapper above reads the machine.
fn wiredCeilingFloorForRam(config: ?*const model_mod.ModelConfig, total_ram: u64) u64 {
    const c = config orelse return 0;
    // Resident MiMo fills a 128 GB Mac, so its ceiling is the declared limit, not what other apps leave free.
    if (!c.longCtxGated() and !c.expert_streaming and !c.isMimo() and !c.isGlm5()) return 0;
    const unplanned = wiredUnplannedBytes(wired_limit_margin_bytes, gpu_floor_bytes.load(.monotonic));
    const floor = wiredLimitFloor(wiredLimitBytes(), total_ram, unplanned);
    if (floor > 0 and wired_floor_logged.cmpxchgStrong(false, true, .monotonic, .monotonic) == null) {
        log.info("[mem] ceiling {d} MB from iogpu.wired_limit_mb={d} (working set {d} MB, margin {d} MB, GPU floor {d} MB, unplanned {d} MB)\n", .{
            floor >> 20,
            wiredLimitBytes() >> 20,
            getGpuWorkingSetLimit() >> 20,
            wired_limit_margin_bytes >> 20,
            gpu_floor_bytes.load(.monotonic) >> 20,
            unplanned >> 20,
        });
    }
    return floor;
}

/// PURE (unit-testable): the cap to put on MLX's reclaimable buffer pool for a
/// machine with `total_ram` bytes of physical memory. 0 when the RAM query
/// failed — never clamp on bad data.
///
/// MLX's own default is `min(1.5 x working_set, 0.95 x RAM)` with a GC limit at
/// `0.95 x working_set` (`backend/metal/allocator.cpp`) — ~121 GB / ~91 GB on a
/// 128 GB Mac, i.e. it does not trim until the machine is already dead. Freed
/// buffers are parked in that pool instead of going back to the OS, so anything
/// that frees never-repeating sizes in a loop (KV growth, decode transients)
/// grows the process footprint without moving `active_bytes` at all. Issue #110:
/// 81.4 GB in Activity Monitor against 19.6 GB in the panel.
///
/// RAM/16 is big enough for the step-to-step buffer reuse the pool exists for
/// (8 GB on a 128 GB Mac) and small enough that it can never be the footprint.
/// The 2 GB floor keeps 8/16 GB machines from thrashing the allocator.
pub fn mlxCacheLimitBytes(total_ram: u64) u64 {
    if (total_ram == 0) return 0;
    const GB: u64 = 1 << 30;
    return @max(2 * GB, @min(8 * GB, total_ram / 16));
}

/// Impure wrapper: apply the cap to the live MLX allocator. Called ONCE from
/// `main()`, above every subcommand branch — not per serve path, because a
/// hand-rolled per-path config is exactly how `runHeadlessServe` (the mode the
/// app always launches) came to silently eat launch flags.
///
/// Never RAISES a tighter existing cap.
pub fn applyMlxCacheLimit() void {
    const cap = mlxCacheLimitBytes(metrics.getTotalMemBytes());
    if (cap == 0) return;
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, @intCast(cap));
    if (prev < cap) {
        var tmp: usize = 0;
        _ = mlx.mlx_set_cache_limit(&tmp, prev);
        return;
    }
    log.info("[mem] MLX buffer-pool cap {d} MB (was {d} MB)\n", .{ cap >> 20, prev >> 20 });
}

/// Lowers MLX's buffer-pool cap to a per-model ledger term; never raises the RAM-wide cap.
pub fn capMlxPool(bytes: u64) void {
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, @intCast(bytes));
    if (prev < bytes) {
        var tmp: usize = 0;
        _ = mlx.mlx_set_cache_limit(&tmp, prev);
        return;
    }
    dsv41_borrow.poolCapped(prev);
    log.info("[mem] MLX buffer-pool cap {d} MB from the ssd-budget ledger (was {d} MB)\n", .{ bytes >> 20, prev >> 20 });
}

/// Impure wrapper: current GPU allocation ceiling from live MLX + system
/// counters. `mlx_footprint` = active (in-use) + cache (reclaimable) so an
/// idle machine's ceiling stays ≈ the static device max (no auto-context
/// regression), while external pressure — reflected in `getAvailableMemBytes`
/// (free + file-backed pages, capped by the RAM outside wired + compressor) — tightens it. See
/// `physicalMemoryCeiling`.
/// The ceiling term that does not move with instantaneous free RAM: Metal's recommended
/// working set (or the wired limit). The load-time hot-cache clamp bills against this and
/// nothing else: two boots 11 minutes apart resolved the same ask to 1076 and 9757 MB off the
/// live term. Request-time admission still reads live memory.
/// Tests only: stand in for the machine's working-set limit (CI runners have 7 GB).
pub var static_ceiling_override: ?u64 = null;

pub fn staticGpuMemoryCeiling() u64 {
    return static_ceiling_override orelse getGpuWorkingSetLimit();
}

/// THE ceiling helper: `available`, `/props`, the hot-cache clamp, the auto-context pin and the
/// admission bill all read it, so the wired-limit floor cannot reach some of them and not others.
pub fn currentGpuMemoryCeiling(config: ?*const model_mod.ModelConfig, active_mem: u64) u64 {
    var cache_mem: usize = 0;
    _ = mlx.mlx_get_cache_memory(&cache_mem);
    return gpuCeilingWithWiredFloor(
        getGpuWorkingSetLimit(),
        active_mem +| @as(u64, cache_mem),
        metrics.getAvailableMemBytes(),
        wiredCeilingFloorFor(config),
    );
}

pub fn expertStreamingLoadFits(ceiling: u64, active_mem: u64, resident_bytes: u64, cache_bytes: u64, bounce_bytes: u64, fill_peak_bytes: u64, serving_bytes: u64) bool {
    const needed = active_mem +| resident_bytes +| cache_bytes +| bounce_bytes +| fill_peak_bytes +| serving_bytes;
    return needed <= ceiling;
}

fn expertStreamingPlannedSeq(config: *const model_mod.ModelConfig) u64 {
    const manual = manualContext(config);
    return if (manual > 0) manual else @min(config.contextCap(), 65_536);
}

/// The KV a streamed load plans to hold: the second term of the
/// budget-vs-wired-limit admission.
fn expertStreamingPlannedKvBytes(config: *const model_mod.ModelConfig) u64 {
    const kv_bits = defaultKvBits(config);
    return sessionBytesPerToken(config, kv_bits) *| expertStreamingPlannedSeq(config) +| slotRingBytes(config, kv_bits);
}

fn expertStreamingServingBytes(config: *const model_mod.ModelConfig) u64 {
    const seq: u64 = expertStreamingPlannedSeq(config);
    const explicit = explicitPrefillChunk();
    // A request picks its own rung against free memory, and an unpinned arch's load-time pin is memory-sized after
    // this check, so the load must only prove the ladder's floor fits.
    const floor: u64 = PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];
    const chunk: u64 = if (explicit > 0) (perRequestFloorWidth(config) orelse explicit) else if (config.pinned_prefill_chunk > 0) config.pinned_prefill_chunk else floor;
    // The load's own staging term already holds the decode pool.
    return prefillNeededAtChunk(config, @max(seq, chunk), 2048, defaultKvBits(config), chunk, .{}) -| config.expert_fill_peak_bytes -| config.expertPoolBytes();
}

/// What a streamed load holds beside its ssd budget: the planned KV and, under `--vision`, its largest image encode.
fn expertStreamingBesideBudgetBytes(config: *const model_mod.ModelConfig) u64 {
    return expertStreamingPlannedKvBytes(config) +| config.expert_vision_encode_bytes;
}

pub fn expertCacheFitForLoad(config: *const model_mod.ModelConfig, resident_bytes: u64) !void {
    if (!config.expert_streaming) return;
    if (config.expert_ssd_budget_bytes > 0) {
        const beside = expertStreamingBesideBudgetBytes(config);
        const limit = ssdBudgetCeiling(wiredLimitBytes(), staticGpuMemoryCeiling());
        if (!ssdBudgetAdmits(config.expert_ssd_budget_bytes, beside, limit)) {
            log.err("--ssd-budget-gb {d} plus {d:.2} GB of planned KV and {d:.2} GB of image encode exceeds the {d:.2} GB residency limit; raise iogpu.wired_limit_mb to {d} or lower the budget\n", .{
                config.expert_ssd_budget_bytes >> 30,
                @as(f64, @floatFromInt(expertStreamingPlannedKvBytes(config))) / 1e9,
                @as(f64, @floatFromInt(config.expert_vision_encode_bytes)) / 1e9,
                @as(f64, @floatFromInt(limit)) / 1e9,
                wiredLimitMbForBudget(config.expert_ssd_budget_bytes, beside),
            });
            return error.SsdBudgetExceedsWiredLimit;
        }
    }
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    const ceiling = currentGpuMemoryCeiling(config, active_mem);
    if (!expertStreamingConfigFits(config, ceiling, active_mem, resident_bytes)) {
        log.err("the streamed load bills {d} MB (active {d}, resident {d}, cache {d}, workspace and bounce {d}, serving {d}) against {d} MB; lower --ssd-budget-gb\n", .{
            (active_mem + resident_bytes + config.expert_cache_bytes + config.expert_bounce_bytes + scheduler_mod.loadStagingBytes(config, config.expert_fill_peak_bytes, config.expert_workspace_bytes) + @max(expertStreamingServingBytes(config), config.expert_vision_encode_bytes)) >> 20,
            active_mem >> 20,
            resident_bytes >> 20,
            config.expert_cache_bytes >> 20,
            (config.expert_bounce_bytes + scheduler_mod.loadStagingBytes(config, config.expert_fill_peak_bytes, config.expert_workspace_bytes)) >> 20,
            @max(expertStreamingServingBytes(config), config.expert_vision_encode_bytes) >> 20,
            ceiling >> 20,
        });
        return error.ExpertCacheDoesNotFit;
    }
}

fn expertStreamingConfigFits(config: *const model_mod.ModelConfig, ceiling: u64, active_mem: u64, resident_bytes: u64) bool {
    return expertStreamingLoadFits(
        ceiling,
        active_mem,
        resident_bytes,
        config.expert_cache_bytes,
        config.expert_bounce_bytes,
        scheduler_mod.loadStagingBytes(config, config.expert_fill_peak_bytes, config.expert_workspace_bytes),
        @max(expertStreamingServingBytes(config), config.expert_vision_encode_bytes),
    );
}

pub fn rowSsdBudgetGb(effective_bytes: u64, setting_gb: u32) u32 {
    if (effective_bytes > 0) return @intCast(effective_bytes >> 30);
    return setting_gb;
}

/// PURE budget math (no MLX/Metal calls — unit-testable): the largest context
/// length whose KV cache + per-token working set fits under the GPU
/// working-set ceiling, after subtracting what's already resident
/// (`active_mem` — model weights + any resident hot-cache KV) AND the hot
/// prefix cache's FULL byte budget (`hot_cache_reserve`).
///
/// Reserving the whole cache budget — not just its current residency — budgets
/// for STEADY STATE. Over an agentic session the hot cache fills to its cap, so
/// an auto-context computed against an empty cache (24k tokens on a 16 GB Mac)
/// later collides with the filled 2 GB cache plus a
/// large cold MoE prefill and the command buffer OOMs. Subtracting it up front
/// keeps the reported context stable across the session and within the ceiling
/// once the cache is full. The 0.64 factor (two ×4/5 margins) absorbs the
/// prefill compute spike that lands on top of `active_mem`.
/// PURE: the raw largest context the budget fits — no 1024 floor, no
/// `max_pos` clamp. Split out of `safeContextForBudget` because
/// `resolvePrefillChunk` compares rungs against each other and the floor makes
/// a rung that fits NOTHING look identical to one that fits 1024 tokens.
/// The 0.64 factor is the same two x4/5 margins documented below.
fn contextForBudget(working_set_limit: u64, active_mem: u64, hot_cache_reserve: u64, per_tok: u64) u64 {
    if (per_tok == 0) return 0;
    const spoken_for: u64 = active_mem +| hot_cache_reserve;
    const usable: u64 = if (working_set_limit > spoken_for) working_set_limit - spoken_for else 0;
    return ((usable * 4 / 5) * 4 / 5) / per_tok;
}

fn safeContextForBudget(
    working_set_limit: u64,
    active_mem: u64,
    hot_cache_reserve: u64,
    per_tok: u64,
    max_pos: u32,
) u32 {
    if (per_tok == 0) return 1024;
    const max_seq: u64 = contextForBudget(working_set_limit, active_mem, hot_cache_reserve, per_tok);
    if (max_seq < 1024) return 1024;

    var result: u32 = if (max_seq > std.math.maxInt(u32)) std.math.maxInt(u32) else @intCast(max_seq);
    // Cap at model's max position embeddings.
    if (max_pos > 0) result = @min(result, max_pos);
    return result;
}

/// The process-wide KV width a request gets when it names none — the same
/// expression `checkAttentionMemory` falls back to, so the sizer and the
/// admission guard read one answer.
///
/// Through `configuredKvQuant`, so a load-time bill gets the boot's `--kv-quant`, not the dense fallback.
fn defaultKvBits(config: *const model_mod.ModelConfig) u64 {
    const cfg: transformer_mod.KVQuantConfig = configuredKvQuantFor(config);
    return if (cfg.scheme == .off) 16 else cfg.bits;
}

/// The prefill working set that is NOT the KV cache: score scratch, the
/// quantized-KV dequant, and the QKV/MLP transient envelope — all bounded by
/// the prefill CHUNK, at the widest chunk any prompt on this model can run.
///
/// It is a one-off reserve, not a per-token cost. `computeMemoryContext` used
/// to bill the MLP envelope (`8 × max(hidden, ffn) × 2`) against every token of
/// CONTEXT: 272 KB/token beside a 64 KB/token KV cache, i.e.
/// 81% of the budget spent on a transient that never scales with the context
/// length, which caps a 16 GB Mac under 4k tokens whatever the weights cost.
/// The peak above steady state is flat in prompt length and tracks the chunk.
///
/// Same estimator as the admission guard with the KV term zeroed, because the
/// KV cache is exactly what the sizer is solving for.
pub fn prefillTransientReserve(config: *const model_mod.ModelConfig, kv_bits: u64, chunk: u64) u64 {
    return prefillTransientReserveAtKv(config, kv_bits, chunk, chunk);
}

/// `prefillTransientReserve` with the KV length made explicit. The load-time sizer asks at
/// `kv = chunk`; a per-chunk decision has a KV, and the attention score sheet and the QSA
/// sheet are linear in it (~0.5 GB under-billed at the default sheet, ~9 GB at 4096 MB).
pub fn prefillTransientReserveAtKv(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    chunk: u64,
    kv_len: u64,
) u64 {
    const seq: u64 = @max(kv_len, chunk);
    if (config.dsv41) |c| return c.chunkCap(chunk, seq) +| (if (config.expert_streaming) config.expert_fill_peak_bytes else 0);
    if (config.isGlm5()) return glm5TransientBytes(config, seq, chunk, kv_bits);
    return prefillMemoryNeeded(
        seq,
        config.num_attention_heads,
        config.num_key_value_heads,
        0,
        config.head_dim,
        config.prefillScoreHeadDim(),
        config.hidden_size,
        prefillFfnWidth(config),
        kv_bits,
        chunk,
        config.prefillAttnKeys(seq),
        prefillStreamBytesPerToken(config),
        prefillDequantWeightBytes(config),
        .{
            .state_bytes = if (mtpHeadDefaultOn(config)) mimo_mtp.State.billedBytes(config) else 0,
            .qsa_ring_bytes = slotRingBytes(config, kv_bits),
            .dequant_scratch_bytes = kvDequantScratchBytes(config, seq, @min(chunk, seq)),
            .uplift_16ths = billUplift16ths(config, seq),
        },
    ) + qsaMaskBytes(config, @min(chunk, @max(seq, 1)), seq) +
        slidingBandScoreBytes(config, @min(chunk, @max(seq, 1))) +
        fp8DequantScratchBytes(config, @min(chunk, @max(seq, 1))) +
        (if (config.expert_streaming) config.expert_fill_peak_bytes else 0) +
        cadenceInflightBytes(mlx_mb_per_buffer);
}

/// The composed score sheet a ringed arch's SLIDING layers build where
/// `sushi_attn_pd` does not serve their band + sink attention. Their attention
/// is a band, so one layer's sheet is `[heads, fwd, window + fwd - 1]` whatever
/// the prompt length. Billed at `MOE_PREFILL_COEXIST` layers, the same
/// coexistence the other per-layer prefill transients are billed at. Zero
/// exactly where the dispatch's own predicate (`slidingPrefillFused`) holds.
pub fn slidingBandScoreBytes(config: *const model_mod.ModelConfig, fwd: u64) u64 {
    if (config.swaRingTokens() == 0) return 0;
    if (transformer_mod.slidingPrefillFused(config, fwd)) return 0;
    const keys: u64 = @as(u64, config.sliding_window) +| fwd -| 1;
    return MOE_PREFILL_COEXIST *| @as(u64, config.num_attention_heads) *| fwd *| keys *| 2;
}

/// The bf16 copy `fp8_block.project` dequantizes a source FP8 linear into when
/// a forward is wider than its GEMV: the widest linear's, and one more for the
/// copy MLX's buffer cache still holds when the next linear dequantizes.
pub fn fp8DequantScratchBytes(config: *const model_mod.ModelConfig, fwd: u64) u64 {
    if (!config.usesMimoSourceTrunk() or fwd <= @as(u64, @intCast(fp8_block.gemv_max_rows))) return 0;
    var widest: u64 = config.intermediate_size;
    for (0..config.num_hidden_layers) |i| {
        const l: u32 = @intCast(i);
        const qkv: u64 = @as(u64, config.layerNumHeads(l)) * config.layerHeadDim(l) +
            @as(u64, config.layerKVHeads(l)) * (config.layerHeadDim(l) + config.layerVHeadDim(l));
        widest = @max(widest, qkv);
    }
    return 2 *| widest *| config.hidden_size *| 2;
}

/// Dense bytes ONE layer's `KVCache.denseView` rebuild costs at this prompt
/// length — the quantized-KV prefill transient, which the eval cadence bounds
/// to a layer at a time. Zero unless the generic `2 · seq · kv_heads · hdim ·
/// 2` cannot say it: on a ringed arch the sliding layers rebuild a WINDOW, and
/// the layers that rebuild the sequence have their own head count and two
/// different K and V widths (mimo_v2: 4 heads of 192+128, against the generic
/// 8 heads of 192 twice — 2.4x).
pub fn kvDequantScratchBytes(config: *const model_mod.ModelConfig, seq: u64, fwd: u64) u64 {
    // A QSA layer rebuilds only what a forward of this width cannot gather in place.
    const qsa_rows: ?u64 = if (config.longCtxGated())
        transformer_mod.qsaDenseRebuildRows(config.indexer_budget, config.indexer_compress_ratio, fwd)
    else
        null;
    if (config.swaRingTokens() == 0 and qsa_rows == null) return 0;
    // A layer whose prefill attention reads the PACKED cache one dispatch at a
    // time (`transformer.fusedSdpaPrefillKv`) never rebuilds more than a
    // dispatch's keys, at `MOE_PREFILL_COEXIST` of them — the dispatches are
    // serially dependent through the softmax carry, so this is the ceiling.
    // The kill switch takes the width off `prefillHeadDimFused` and the whole
    // rebuild comes back with it. A decode rebuilds whole below the packed arms' floor.
    const seq_rows: u64 = if (qsa_rows) |rows| rows else if (transformer_mod.prefillHeadDimFused(config.prefillScoreHeadDim()))
        @max(MOE_PREFILL_COEXIST *| @as(u64, @intCast(transformer_mod.PACKED_KV_SLICE_MAX)), transformer_mod.mimoGlobalDecodeRebuildMaxKeys())
    else
        seq;
    var widest: u64 = 0;
    var li: u32 = 0;
    while (li < config.num_hidden_layers) : (li += 1) {
        const rows: u64 = if (config.isKvPerTokenLayer(li)) @min(seq, seq_rows) else @min(seq, config.swaRingTokens());
        widest = @max(widest, rows *| config.layerKvBytes(li));
    }
    return widest;
}

/// Bytes per (query, key) the QSA prefill holds for ONE live layer past the
/// indexer budget: the `[S, kv]` bool mask, its additive copy for the sdpa
/// arm, and the f32 indexer scores over kv/4 blocks (4 bytes/block = 1/key).
const QSA_MASK_BYTES_PER_KEY: u64 = 4;

/// qwen4_exp sparse attention materializes a `[S, kv]` mask per layer that
/// no envelope term models: 4096 x 25k keys = 410 MB, and a tight box met it
/// as a Metal OOM at 25k+. Zero for every arch without an indexer. The sizer
/// bills it at kv = chunk (no prompt yet); the admission guard at the real
/// prompt length.
pub fn qsaMaskBytes(config: *const model_mod.ModelConfig, fwd: u64, kv: u64) u64 {
    if (config.indexer_budget == 0) return 0;
    // Prefill widths gather by block index (row-chunked score sheet, no
    // [S, kv] mask); decode/verify widths still build the dense mask.
    if (fwd >= transformer_mod.FUSED256_MIN_Q_LEN and transformer_mod.qsaGatherEnabled())
        return transformer_mod.qsaPrefillTransientBytes(config.indexer_n_heads, fwd, kv, config.indexer_compress_ratio, config.indexer_head_dim);
    return QSA_MASK_BYTES_PER_KEY * fwd * kv * 5 / 4;
}

/// Widths `resolvePrefillChunk` will step down through. Descending, floored at
/// `generate.PREFILL_CHUNK_FLOOR` — below that the score-budget path refuses to
/// go either, and a 256-token forward stops amortizing the per-chunk sweeps.
pub const PREFILL_CHUNK_LADDER = [_]u32{ 8192, 4096, 2048, 1024, 512 };

/// How much of the post-weights serving budget a ONE-OFF prefill transient may
/// claim before the chunk steps down. A quarter: past that the machine is
/// trading the whole session's context for one forward's speed, which is the
/// trade that reported a 1024-token context on a 16 GB Mac.
const PREFILL_RESERVE_BUDGET_SHARE: u64 = 4;

/// The hot prefix cache's smallest useful budget: a prefill chunk that would price the cache
/// under this steps down instead. Stops `requested == 0` from resolving to the 1-byte floor.
const HOT_CACHE_FLOOR_BYTES: u64 = 1024 * 1024 * 1024;

/// PURE: the widest prefill chunk THIS machine can afford for THIS model.
///
/// The chunk multiplies the biggest term in the memory bill, and until now
/// nothing tied it to the machine: `--prefill-chunk` defaults to 8192, the app
/// never passes the flag, so a 16 GB Mac reserved the same 5-7 GB MLP envelope
/// a 128 GB one does. Measured on a 16 GB profile (Mistral-7B-4bit, weights
/// 3.80 GB against an 11.9 GB Metal ceiling): the reserve at chunk 8192 alone
/// exceeded what was left after the weights, so the sizer reported 1024 tokens
/// and the admission guard refused a 10k-token prompt for "needing" 9.2 GB —
/// a prompt whose measured peak is 2.39 GB. Stepping to chunk 512 costs prefill
/// throughput and buys 22k tokens of context back.
///
/// Never raises anything: `effectivePrefillChunk` takes the MIN of this and the
/// launch chunk, and an explicit `--prefill-chunk` outranks it entirely.
///
/// It does not read the hot-cache ask: the clamp downstream subtracts this chunk's reserve
/// from the cache's headroom, so feeding the ask in here was non-monotone (a bigger
/// `--prefix-cache-mem` stepped the chunk down and handed the cache MORE bytes). The chunk is
/// sized against the serving budget alone; the cache is the residual claimant. `ctx_kv_bytes`
/// is the KV a pinned context has already spoken for (0 under auto-context).
/// The byte cap one prefill forward's transient may claim, and the one place the sizer's two
/// changes (dropping the ask, adding the ctx bar) are gated by arch; every other arch keeps
/// the previous arithmetic.
pub fn prefillChunkCap(
    config: *const model_mod.ModelConfig,
    ceiling: u64,
    active_mem: u64,
    ctx_kv_bytes: u64,
    hot_cache_ask: u64,
) u64 {
    if (!config.longCtxGated()) {
        // Ungated: the ask is spoken for, and a quarter of what is left is the whole cap.
        const spoken_for: u64 = active_mem +| hot_cache_ask;
        return (ceiling -| spoken_for) / PREFILL_RESERVE_BUDGET_SHARE;
    }
    const serving_budget: u64 = ceiling -| active_mem;
    // Two ask-independent bars: a share of the serving budget, and whatever the pinned context leaves less a cache worth having.
    const share_bar: u64 = serving_budget / PREFILL_RESERVE_BUDGET_SHARE;
    // The ctx bar can only ever narrow; `--ctx-size` is a cap, not an allocation, so it can pin
    // chunk 512 for the life of a process. The narrowing is logged.
    const ctx_bar: u64 = (serving_budget -| ctx_kv_bytes) -| HOT_CACHE_FLOOR_BYTES;
    return @min(share_bar, ctx_bar);
}

pub fn resolvePrefillChunk(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    ceiling: u64,
    active_mem: u64,
    ctx_kv_bytes: u64,
    hot_cache_ask: u64,
) u32 {
    if (config.isGlm5()) return contextNeutralChunk(config, kv_bits, ceiling, active_mem, @import("glm5/glm5_forward.zig").prefill_chunk);
    if (config.isDsv41()) return contextNeutralChunk(config, kv_bits, ceiling, active_mem, dsv41_config.prefill_width);
    const cap: u64 = prefillChunkCap(config, ceiling, active_mem, ctx_kv_bytes, hot_cache_ask);
    for (PREFILL_CHUNK_LADDER) |chunk| {
        if (prefillTransientReserve(config, kv_bits, chunk) <= cap) return chunk;
    }
    // Nothing fits the share — the model barely fits at all. Take the narrowest
    // rung: it is the smallest bill this box can be asked for, and returning 0
    // would read as "not pinned" and hand the forward the launch width.
    return PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];
}

/// The widest rung, at most `width` (the arch's numerics width), that advertises the same auto context as the
/// narrowest: the bill `max_safe_context` and admission charge decides, not a share of free memory. GLM and DeepSeek,
/// whose per-token state is small beside their prefill transient.
fn contextNeutralChunk(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, width: u32) u32 {
    const narrowest = PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];
    const floor = advertisedContextAt(config, kv_bits, ceiling, active_mem, narrowest);
    for (PREFILL_CHUNK_LADDER) |chunk| {
        if (chunk > width) continue;
        if (advertisedContextAt(config, kv_bits, ceiling, active_mem, chunk) >= floor) return chunk;
    }
    return narrowest;
}

fn advertisedContextAt(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, chunk: u64) u32 {
    return autoContextFrom(memoryContextAtChunk(config, kv_bits, ceiling, active_mem, chunk), config.contextCap());
}

/// The width `--prefill-chunk` asked for, or 0. The bill must let it outrank the pin as the
/// forward does: `--prefill-chunk 4096` used to run against a 512-token reserve.
pub fn explicitPrefillChunk() u32 {
    if (!generate_mod.prefill_chunk_explicit) return 0;
    const raw: usize = @min(generate_mod.prefill_chunk_override, @as(usize, std.math.maxInt(u32)));
    return @intCast(raw);
}

/// The chunk this model is billed at, and therefore runs at. `chunk_override` (0 = none)
/// mirrors `generate.effectivePrefillChunk`'s precedence.
pub fn billedPrefillChunk(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    ceiling: u64,
    active_mem: u64,
    ctx_kv_bytes: u64,
    hot_cache_ask: u64,
    chunk_override: u32,
) u32 {
    if (chunk_override > 0) return chunk_override;
    return resolvePrefillChunk(config, kv_bits, ceiling, active_mem, ctx_kv_bytes, hot_cache_ask);
}

/// The context KV the chunk sizer must leave standing: the pinned context's bill under an
/// explicit `--ctx-size`, 0 while the context is auto (the auto sizer adapts to the rung).
pub fn sizerCtxKvBytes(config: *const model_mod.ModelConfig, kv_bits: u64) u64 {
    if (manualContext(config) == 0) return 0;
    return sessionBytesPerToken(config, kv_bits) *| manualContext(config) +| slotRingBytes(config, kv_bits);
}

/// The full-context admission bill at the width a requested launch can run.
/// An explicit prefill width is priced as requested; auto picks the per-request floor.
fn sushiContextBytesAt(config: *const model_mod.ModelConfig, context: u32, kv_bits: u64) struct { bytes: u64, chunk: u64 } {
    const explicit = explicitPrefillChunk();
    const rung: u32 = if (explicit > 0) explicit else perRequestFloorWidth(config) orelse
        (if (config.pinned_prefill_chunk > 0) config.pinned_prefill_chunk else 8192);
    const chunk = rungWidth(config, context, rung, config.longCtxGated());
    return .{
        .bytes = prefillNeededAtChunk(config, context, 0, kv_bits, chunk, .{ .mtp_on = mtpHeadDefaultOn(config) }),
        .chunk = chunk,
    };
}

/// PURE with respect to memory: solve the same admission bill used to check the requested context.
pub fn maxSushiContextForBudget(config: *const model_mod.ModelConfig, kv_bits: u64, budget: u64) u32 {
    return maxSushiContextAtWidth(config, kv_bits, budget, null, .{ .mtp_on = mtpHeadDefaultOn(config) });
}

fn maxSushiContextAtWidth(config: *const model_mod.ModelConfig, kv_bits: u64, budget: u64, fixed_chunk: ?u64, warm: WarmPrefix) u32 {
    var low: u32 = 0;
    var high: u32 = if (config.contextCap() > 0) config.contextCap() else std.math.maxInt(u32);
    // Before the auto context is pinned the bill's reservation reads `getEffectiveContextLength`,
    // which is this very search: bill each probe as the context it is asking about.
    const unpinned = manualContext(config) == 0 and config.pinned_context == 0;
    while (low < high) {
        const mid = low + @as(u32, @intCast((@as(u64, high) - low + 1) / 2));
        var probe = config.*;
        if (unpinned) probe.pinned_context = mid;
        const chunk = fixed_chunk orelse sushiContextBytesAt(&probe, mid, kv_bits).chunk;
        if (prefillNeededAtChunk(&probe, mid, 0, kv_bits, chunk, warm) <= budget)
            low = mid
        else
            high = mid - 1;
    }
    return low;
}

fn requestContextRefusal(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv: transformer_mod.KVQuantConfig, bill: AdmissionBill, warm: WarmPrefix) ?model_registry_mod.MemoryContextRefusal {
    if (!model_mod.usesSushiQuantMemoryBill(config)) return null;
    const bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
    const budget = bill.available +| bill.evictionCredit();
    return .{
        .requested = @as(u64, @intCast(prompt_len)) +| @min(max_tokens, getEffectiveContextLength(config) -| @as(u64, @intCast(prompt_len))),
        .maximum = maxSushiContextAtWidth(config, bits, budget, bill.chunk, warm),
        .needed = bill.needed,
        .available = budget,
        .kv_bits = bits,
        .chunk = bill.chunk,
    };
}

pub fn memoryContextRefusalMessage(buf: []u8, refusal: model_registry_mod.MemoryContextRefusal) []const u8 {
    return std.fmt.bufPrint(buf, "Context {d} does not fit available GPU memory at KV{d}, prefill chunk {d}. Maximum context: {d} tokens. Reduce context or free memory.", .{
        refusal.requested, refusal.kv_bits, refusal.chunk, refusal.maximum,
    }) catch "The requested context does not fit in available GPU memory.";
}

pub fn loadServingBill(config: *const model_mod.ModelConfig, budget: u64) ?scheduler_mod.LoadServingBill {
    if (!model_mod.usesSushiQuantMemoryBill(config)) return null;
    const context = manualContext(config);
    if (context == 0) return null;
    const bits = defaultKvBits(config);
    const full = sushiContextBytesAt(config, context, bits);
    return .{
        .requested_context = context,
        .needed = full.bytes,
        .max_context = maxSushiContextForBudget(config, bits, budget),
        .kv_bits = bits,
        .chunk = full.chunk,
    };
}

test "Sushi quant memory production context CPU audit" {
    const raw = std.c.getenv("SUSHI_MEMORY_AUDIT_MODEL") orelse return error.SkipZigTest;
    const path = std.mem.span(raw);
    var cfg = try model_mod.parseConfig(std.testing.io, std.testing.allocator, path);
    defer cfg.deinit(std.testing.allocator);
    const saved_ctx = server_config.max_context_size;
    defer server_config.max_context_size = saved_ctx;
    const saved_width = generate_mod.prefill_chunk_override;
    defer generate_mod.prefill_chunk_override = saved_width;
    const saved_explicit = generate_mod.prefill_chunk_explicit;
    defer generate_mod.prefill_chunk_explicit = saved_explicit;
    server_config.max_context_size = 500000;
    generate_mod.prefill_chunk_override = 2048;
    generate_mod.prefill_chunk_explicit = true;
    const budget: u64 = 16 << 30;
    const bill = loadServingBill(&cfg, budget).?;
    try std.testing.expectEqual(@as(u64, 8), bill.kv_bits);
    try std.testing.expect(sushiContextBytesAt(&cfg, bill.max_context, 8).bytes <= budget);
    if (bill.max_context < cfg.contextCap()) try std.testing.expect(sushiContextBytesAt(&cfg, bill.max_context + 1, 8).bytes > budget);
}

test "Sushi quant memory request refusal carries the actual KV width MTP and maximum to the wire" {
    const saved_ctx = server_config.max_context_size;
    defer server_config.max_context_size = saved_ctx;
    const saved_mtp = configured_mtp;
    defer configured_mtp = saved_mtp;
    server_config.max_context_size = 500000;
    configured_mtp = null;
    var cfg = qwen4RequestTestConfig();
    cfg.expert_layout = .exl3_k4;
    const chunk: u64 = 1024;
    const budget = prefillNeededAtChunk(&cfg, 64000, 0, 4, chunk, .{});
    const bill = AdmissionBill{ .needed = prefillNeededAtChunk(&cfg, 128000, 32000, 4, chunk, .{}), .available = budget, .chunk = chunk };
    const refusal = requestContextRefusal(&cfg, 128000, 32000, transformer_mod.KVQuantConfig.affine(4), bill, .{}).?;
    try std.testing.expectEqual(@as(u64, 4), refusal.kv_bits);
    try std.testing.expectEqual(@as(u32, 64000), refusal.maximum);
    try std.testing.expectEqual(@as(u64, 160000), refusal.requested);
    const mtp = requestContextRefusal(&cfg, 128000, 32000, transformer_mod.KVQuantConfig.affine(4), bill, .{ .mtp_on = true }).?;
    try std.testing.expect(mtp.maximum < refusal.maximum);
    var buf: [256]u8 = undefined;
    const message = memoryContextRefusalMessage(&buf, refusal);
    try std.testing.expect(std.mem.indexOf(u8, message, "Maximum context: 64000 tokens") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "KV4") != null);
    slot_memory_context_refusal = refusal;
    const wire = mapGenerationError(error.PrefillDoesNotFit, &buf);
    try std.testing.expectEqual(@as(u32, 400), wire.code);
    try std.testing.expect(std.mem.indexOf(u8, wire.message, "Maximum context: 64000 tokens") != null);
    try std.testing.expectEqual(@as(?model_registry_mod.MemoryContextRefusal, null), slot_memory_context_refusal);
    try std.testing.expectEqualStrings(PREFILL_NOFIT_MSG, mapGenerationError(error.PrefillDoesNotFit, &buf).message);
}

pub fn testDsv41Vp() !chat_mod.VisionPreproc {
    var cfg: model_mod.ModelConfig = .{};
    cfg.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    return visionPreprocFromConfig(&cfg);
}

test "Sushi quant memory rejects an oversized context with an exact maximum at KV8" {
    const saved_ctx = server_config.max_context_size;
    defer server_config.max_context_size = saved_ctx;
    const saved_mtp = configured_mtp;
    defer configured_mtp = saved_mtp;
    const saved_kv = configured_kv_quant;
    defer configured_kv_quant = saved_kv;
    const saved_width = generate_mod.prefill_chunk_override;
    defer generate_mod.prefill_chunk_override = saved_width;
    const saved_explicit = generate_mod.prefill_chunk_explicit;
    defer generate_mod.prefill_chunk_explicit = saved_explicit;
    server_config.max_context_size = 500000;
    configured_kv_quant = null;
    configured_mtp = false;
    generate_mod.prefill_chunk_override = 2048;
    generate_mod.prefill_chunk_explicit = true;
    var cfg = qwen4RequestTestConfig();
    cfg.expert_layout = .exl3_k4;
    const budget = sushiContextBytesAt(&cfg, 128000, 8).bytes;
    const bill = loadServingBill(&cfg, budget).?;
    try std.testing.expectEqual(@as(u64, 8), bill.kv_bits);
    try std.testing.expectEqual(@as(u32, 500000), bill.requested_context);
    try std.testing.expect(bill.needed > budget);
    try std.testing.expectEqual(@as(u32, 128000), bill.max_context);
    try std.testing.expect(sushiContextBytesAt(&cfg, bill.max_context, 8).bytes <= budget);
    try std.testing.expect(sushiContextBytesAt(&cfg, bill.max_context + 1, 8).bytes > budget);
    try std.testing.expectEqual(@as(u64, 2048), sushiContextBytesAt(&cfg, 128000, 8).chunk);
    try std.testing.expect(maxSushiContextForBudget(&cfg, 4, budget) > bill.max_context);
    try std.testing.expectEqual(@as(u32, 0), maxSushiContextForBudget(&cfg, 8, 0));
    configured_mtp = true;
    try std.testing.expect(loadServingBill(&cfg, budget).?.max_context < bill.max_context);
    var mimo = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/model-configs/mimo_v2.json"));
    defer mimo.deinit(std.testing.allocator);
    mimo.expert_layout = .exl3_k4;
    const mimo_budget = sushiContextBytesAt(&mimo, 64000, 8).bytes;
    const mimo_bill = loadServingBill(&mimo, mimo_budget).?;
    try std.testing.expectEqual(@as(u32, 64000), mimo_bill.max_context);
    server_config.max_context_size = 0;
    cfg.ctx_override = 4096;
    cfg.kv_quant_override = transformer_mod.KVQuantConfig.affine(4);
    const settings = loadServingBill(&cfg, budget).?;
    try std.testing.expectEqual(@as(u32, 4096), settings.requested_context);
    try std.testing.expectEqual(@as(u64, 4), settings.kv_bits);
    server_config.max_context_size = 8192;
    configured_kv_quant = transformer_mod.KVQuantConfig.affine(8);
    try std.testing.expectEqual(@as(u32, 8192), loadServingBill(&cfg, budget).?.requested_context);
    try std.testing.expectEqual(@as(u64, 8), loadServingBill(&cfg, budget).?.kv_bits);
    cfg.expert_layout = .mxfp4_individual;
    try std.testing.expectEqual(@as(?scheduler_mod.LoadServingBill, null), loadServingBill(&cfg, budget));
    cfg.expert_layout = .exl3_k4;
    cfg.expert_streaming = true;
    try std.testing.expectEqual(@as(?scheduler_mod.LoadServingBill, null), loadServingBill(&cfg, budget));
}

/// Freeze this model's prefill chunk at load, from live memory. Idempotent.
/// Must run BEFORE the auto-context sizer: the sizer bills this chunk's
/// transient reserve, and `checkAttentionMemory` and
/// `generate.effectivePrefillChunk` then read the same frozen value, so the
/// bill and the forward cannot drift.
pub fn pinPrefillChunk(config: *model_mod.ModelConfig) u32 {
    if (config.pinned_prefill_chunk == 0) {
        var active_mem: usize = 0;
        _ = mlx.mlx_get_active_memory(&active_mem);
        const kv_bits: u64 = defaultKvBits(config);
        // Which ask: the gated arch reads the resolved budget; every other arch reads the raw
        // ask as before (the accessor would pin a second model against the first's budget).
        const hot_cache_ask = if (config.longCtxGated()) resolvedPrefixCacheMem() else legacyPrefixCacheAsk(config);
        // The override's precedence is the second gated input: ungated, `--prefill-chunk`
        // still outranks the pin only at forward time, as before (a known inconsistency).
        const pin_override: u32 = if (config.longCtxGated()) explicitPrefillChunk() else 0;
        config.pinned_prefill_chunk = billedPrefillChunk(
            config,
            kv_bits,
            currentGpuMemoryCeiling(config, active_mem),
            active_mem,
            sizerCtxKvBytes(config, kv_bits),
            hot_cache_ask,
            pin_override,
        );
        {
            var cache_mem: usize = 0;
            _ = mlx.mlx_get_cache_memory(&cache_mem);
            log.debug("[prefill] pinned {d}: ceiling {d} MB, active {d} MB, mlx cache {d} MB, system available {d} MB, working set {d} MB\n", .{
                config.pinned_prefill_chunk,
                currentGpuMemoryCeiling(config, active_mem) >> 20,
                active_mem >> 20,
                cache_mem >> 20,
                metrics.getAvailableMemBytes() >> 20,
                getGpuWorkingSetLimit() >> 20,
            });
        }
        // Say when the ctx bar is the binding one.
        if (config.longCtxGated() and explicitPrefillChunk() == 0) {
            const share_only = resolvePrefillChunk(config, kv_bits, currentGpuMemoryCeiling(config, active_mem), active_mem, 0, hot_cache_ask);
            if (share_only > config.pinned_prefill_chunk) {
                log.info("[prefill] chunk {d} (ctx bar: the --ctx-size KV bill; share-only would allow {d})\n", .{ config.pinned_prefill_chunk, share_only });
            }
        }
        // Say it once per model, wherever the model was pinned from (startup
        // primary or on-demand load) — a narrowed prefill otherwise reads as an
        // unexplained slowdown.
        if (!generate_mod.prefill_chunk_explicit) {
            var buf: [256]u8 = undefined;
            if (prefillChunkLoadLine(&buf, config, config.pinned_prefill_chunk, generate_mod.prefill_chunk_override)) |line| log.info("{s}", .{line});
        }
    }
    return config.pinned_prefill_chunk;
}

/// The load line for the frozen width, null when there is nothing to say. On a per-request arch every
/// request prices its own rung against live memory, so the pin is only the fallback.
pub fn prefillChunkLoadLine(buf: []u8, config: *const model_mod.ModelConfig, pinned: u32, launch: usize) ?[]const u8 {
    if (perRequestPrefillChunkEnabled(config)) {
        const widest = rungWidth(config, 1, cappedRung(PREFILL_CHUNK_LADDER[0], ladderTop(config, explicitPrefillChunk())), config.longCtxGated());
        return std.fmt.bufPrint(buf, "Prefill chunk: per request, up to {d} at a short prompt (the widest rung each request's bill admits); load-time fallback {d}\n", .{ widest, pinned }) catch null;
    }
    if (config.isDsv41() and launch > dsv41_config.prefill_width) {
        if (pinned == dsv41_config.prefill_width)
            return std.fmt.bufPrint(buf, "Prefill chunk: {d} tokens (DeepSeek's default width; --prefill-chunk overrides)\n", .{pinned}) catch null;
        return std.fmt.bufPrint(buf, "Prefill chunk: {d} tokens (memory-sized down from DeepSeek's {d}; --prefill-chunk overrides)\n", .{ pinned, dsv41_config.prefill_width }) catch null;
    }
    if (pinned >= launch) return null;
    if (config.isGlm5() and pinned == @import("glm5/glm5_forward.zig").prefill_chunk)
        return std.fmt.bufPrint(buf, "Prefill chunk: {d} tokens (GLM's numerics width; --prefill-chunk overrides)\n", .{pinned}) catch null;
    return std.fmt.bufPrint(buf, "Prefill chunk: {d} tokens (memory-sized down from {d}; --prefill-chunk overrides)\n", .{ pinned, launch }) catch null;
}

/// PURE: clamp the hot prefix cache's byte budget to what the loaded weights
/// leave under the GPU ceiling after the serving context's KV and the prefill
/// transient reserve. A 40 GB `--prefix-cache-mem` beside a ~70 GB pack was
/// never validated against this headroom: the cache filled toward its cap and
/// a 143k prefill died in an uncatchable Metal OOM (2026-08-30, Flash-Next on
/// 128 GB). `requested == 0` (byte cap disabled) is bounded too — an uncapped
/// cache beside a large model is exactly that crash. Never returns 0:
/// `initWithMem` reads 0 as "no byte cap", the opposite of no headroom.
pub fn clampedPrefixCacheMem(
    requested: u64,
    gpu_ceiling: u64,
    active_weights: u64,
    ctx_kv_bytes: u64,
    transient_reserve: u64,
) u64 {
    const headroom = @max(gpu_ceiling -| (active_weights +| ctx_kv_bytes +| transient_reserve), 1);
    if (requested == 0) return headroom;
    return @min(requested, headroom);
}

/// What the hot cache gets and what it was charged for getting it.
pub const HotCachePlan = struct {
    /// The width the sizer pinned.
    chunk: u32,
    /// The width the clamp reserved for; not `chunk` on an arch that re-bills per request (`clampReserveWidth`).
    reserve_chunk: u32,
    reserve: u64,
    ctx_kv: u64,
    /// The clamped byte budget handed to `initWithMem`.
    budget: u64,
};

/// The whole chain in the one order that is monotone in the ask: chunk first
/// (ask-independent), then the cache takes the residual.
/// The prefill width the load-time clamp reserves for: the ladder floor on an arch that
/// re-bills its width per request (the load-time reserve is a promise to the first request
/// only). Billing the sizer's rung made the clamp non-monotone in the ceiling (a bigger
/// ceiling bought a wider rung and a smaller cache: 3873 vs 1076 MB). An explicit
/// `--prefill-chunk` only lowers that floor; ungated archs keep the sizer's rung.
pub fn clampReserveWidth(config: *const model_mod.ModelConfig, pinned_width: u32) u32 {
    return perRequestFloorWidth(config) orelse pinned_width;
}

/// The narrowest width a request can step down to on the per-request ladder: its floor, or a
/// narrower `--prefill-chunk`. Null where the width is pinned (no ladder).
fn perRequestFloorWidth(config: *const model_mod.ModelConfig) ?u32 {
    if (!perRequestPrefillChunkEnabled(config)) return null;
    return cappedRung(PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1], explicitPrefillChunk());
}

pub fn planHotCache(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    ceiling: u64,
    active_weights: u64,
    ctx_tokens: u64,
    sizer_ctx_kv: u64,
    requested: u64,
    chunk_override: u32,
) HotCachePlan {
    const chunk = billedPrefillChunk(config, kv_bits, ceiling, active_weights, sizer_ctx_kv, requested, chunk_override);
    const reserve_chunk = clampReserveWidth(config, chunk);
    const reserve = prefillTransientReserve(config, kv_bits, reserve_chunk);
    const ctx_kv: u64 = sessionBytesPerToken(config, kv_bits) *| ctx_tokens +| slotRingBytes(config, kv_bits);
    return .{
        .chunk = chunk,
        .reserve_chunk = reserve_chunk,
        .reserve = reserve,
        .ctx_kv = ctx_kv,
        .budget = clampedPrefixCacheMem(requested, ceiling, active_weights, ctx_kv, reserve),
    };
}

/// SSD-first budget semantics. The RAM-first formula treats the hot cache as what is left
/// after the live session's reserve, which double-counts (the resident entry's arrays are
/// shared with the live KV). Here the budget floors at one entry at the working context and
/// `--prefix-cache-mem` is the RAM allowance for idle entries on top (0 = none idle).
pub fn ssdFirstPrefixCacheMem(
    idle_requested: u64,
    gpu_ceiling: u64,
    active_weights: u64,
    ctx_kv_bytes: u64,
    transient_reserve: u64,
) u64 {
    const headroom = gpu_ceiling -| (active_weights +| ctx_kv_bytes +| transient_reserve);
    const idle = @min(idle_requested, headroom);
    // Never 0: `initWithMem` reads 0 as "no byte cap".
    return @max(ctx_kv_bytes +| idle, 1);
}

/// The context a load-time bill must use, before `pinAutoContext` has run: on an auto boot
/// `getEffectiveContextLength` re-derives through `computeMemoryContext`, which reserves the
/// very ask being resolved and answered a 1024-token placeholder. `cache_reserve` is
/// `CTX_SIZING_CACHE_RESERVE` on both arms so the billed session is the advertised one; it
/// must never be the ask or anything derived from it. Runs `autoContextFrom`, the same helper
/// `autoContextFor` runs.
pub fn resolvedContextForLoad(
    explicit_ctx: u32,
    pinned_ctx: u32,
    ceiling: u64,
    active_mem: u64,
    cache_reserve: u64,
    transient_reserve: u64,
    per_tok: u64,
    ctx_cap: u32,
) u32 {
    if (explicit_ctx > 0) return explicit_ctx;
    if (pinned_ctx > 0) return pinned_ctx;
    return autoContextFrom(safeContextForBudget(
        ceiling,
        active_mem,
        cache_reserve +| transient_reserve,
        per_tok,
        0,
    ), ctx_cap);
}

/// The SSD-first call site. Takes the static ceiling (the budget is a property of the machine),
/// the same reserve constant the RAM arm and `computeMemoryContext` pass, and the boot's
/// configured KV width (billing at bf16 on a `--kv-quant 8` boot was the 22,464-vs-13,824 MB defect).
fn ssdFirstSessionTokensNow(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, chunk: u32) u32 {
    if (billedAutoContext(config)) return billedAdvertisedContext(config, memoryContextAtChunk(config, kv_bits, ceiling, active_mem, chunk));
    return resolvedContextForLoad(
        manualContext(config),
        config.pinned_context,
        ceiling,
        active_mem,
        ctxCacheShare(ceiling, active_mem, ctxSizingCacheReserve(config)),
        prefillTransientReserve(config, kv_bits, chunk),
        sessionBytesPerToken(config, kv_bits),
        config.contextCap(),
    );
}

/// The bytes the SSD-first budget floors at: one session at the working context, at the width the cache stores.
fn ssdFirstSessionKvBytes(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, chunk: u32) u64 {
    return sessionBytesPerToken(config, kv_bits) *|
        ssdFirstSessionTokensNow(config, kv_bits, ceiling, active_mem, chunk) +| slotRingBytes(config, kv_bits);
}

/// The SSD-first arm of `prefixCacheMemForLoad`, gate and log included. Null when the arch
/// gate is off; the caller falls through to `clampedPrefixCacheMem`.
fn ssdFirstBudgetForLoad(
    config: *model_mod.ModelConfig,
    requested: u64,
    ceiling: u64,
    active_mem: usize,
    ctx_kv: u64,
    transient_reserve: u64,
    idle_out: *u64,
    quiet: bool,
) ?u64 {
    // The predicate, shared with the spill site: without a disk tier the mode's floor would be RAM the server cannot use.
    if (prefix_cache_capacity == 0 or !prefix_cache_mod.ssdFirstActive(config, diskTierOn(config), prefix_cache_ram_enabled)) return null;
    const budget = ssdFirstPrefixCacheMem(
        requested,
        ceiling,
        active_mem,
        ctx_kv,
        transient_reserve,
    );
    // Publish the cache's cost BEYOND the live session, not the whole budget: `budget` includes
    // `ctx_kv`, and `computeMemoryContext` subtracts the published figure, so publishing the
    // whole would bill the session twice and halve the auto-context.
    publishResolvedPrefixCacheMem(budget -| ctx_kv);
    // The idle allowance, handed to the cache for `spillIdleEntries` to enforce.
    idle_out.* = budget -| ctx_kv;
    // On this arch `--prefix-cache-mem` is the idle allowance, not the whole cache; say so once.
    if (quiet) return budget;
    log.info("[hot-cache] SSD-first budget {d} MB = one session at the working context ({d} MB) + {d} MB idle (--prefix-cache-mem {d} MB = the IDLE allowance on this arch; 0 = no idle entries)\n", .{
        budget >> 20,
        ctx_kv >> 20,
        (budget -| ctx_kv) >> 20,
        requested >> 20,
    });
    return budget;
}

/// Impure wrapper for the model-load site (`Scheduler.doLoadOnInferenceThread`,
/// reached through the LoadParams/LoadRequest resolver pointer — the scheduler
/// deliberately has no server.zig import): the weights are resident there, so
/// `mlx_get_active_memory` is honest. Logs one line when the clamp bites, and publishes the
/// answer so the auto-context sizer stops reserving a budget the cache never got.
/// The hot-cache reserve the context is sized against at load time on the RAM-first arm: a
/// constant (the server's own `--prefix-cache-mem` default), never the ask. Reading the ask
/// let a 60GB ask collapse the advertised context to 870 tokens; context is the primary
/// claimant and the cache the residual.
const CTX_SIZING_CACHE_RESERVE: u64 = 2 * 1024 * 1024 * 1024;

/// The previous context-sizing cache reserve: the raw `--prefix-cache-mem` ask. Kept for
/// ungated archs so their advertised `context_length` does not move. None when RAM retention
/// is off or the model's hot cache never loads.
fn legacyPrefixCacheAsk(config: *const model_mod.ModelConfig) u64 {
    if (prefix_cache_capacity == 0 or !prefix_cache_ram_enabled) return 0;
    if (!prefix_cache_mod.HotPrefixCache.shouldUse(config, ssm_checkpoint_stride > 0)) return 0;
    return prefix_cache_mem_bytes; // legacy_ask_read
}

/// An arch whose auto context is the largest one its request bill admits, and that has none pinned yet.
/// Flash-Next's advertised context from the largest one its bill admits: within the measured range the bill
/// (at least 1.1x the live peak) is the one reserve, so no margin goes on top; past it the margin stays.
fn billedAdvertisedContext(config: *const model_mod.ModelConfig, memory_ctx: u32) u32 {
    const measured: u32 = @intCast(@min(memory_ctx, MEASURED_NET_MAX_TOKENS) / 1024 * 1024);
    const ctx = @max(safeAutoContext(memory_ctx), measured);
    return if (config.contextCap() > 0) @min(ctx, config.contextCap()) else ctx;
}

fn billedAutoContext(config: *const model_mod.ModelConfig) bool {
    return config.longCtxGated() and manualContext(config) == 0 and config.pinned_context == 0;
}

/// The context the RAM-first clamp bills, through the one shared pure resolver.
fn ramFirstContextForLoad(config: *const model_mod.ModelConfig, kv_bits: u64, active_mem: u64, chunk: u32) u32 {
    const ceiling = currentGpuMemoryCeiling(config, active_mem);
    if (billedAutoContext(config)) return billedAdvertisedContext(config, memoryContextAtChunk(config, kv_bits, ceiling, active_mem, chunk));
    return resolvedContextForLoad(
        manualContext(config),
        config.pinned_context,
        ceiling,
        active_mem,
        ctxCacheShare(ceiling, active_mem, ctxSizingCacheReserve(config)),
        prefillTransientReserve(config, kv_bits, chunk),
        sessionBytesPerToken(config, kv_bits),
        config.contextCap(),
    );
}

/// The hot-cache ask when nobody named one: `session` (`OneSession.ask`), never under 2 GB. Below
/// one session the cache keeps only a prefix of the longest conversations, the ones whose reuse
/// saves the most prefill. Any other ask (an operator's, even one equal to the default) stands.
pub fn defaultPrefixCacheAsk(requested: u64, explicit: bool, session: u64) u64 {
    if (explicit or requested != PREFIX_CACHE_MEM_DEFAULT) return requested;
    return @max(requested, session);
}

/// GLM's unnamed RAM tier: the last few turns, given back to a long prefill by admission.
/// Long reuse belongs on the SSD tier (`--prefix-cache-disk`).
pub const GLM_PREFIX_CACHE_MEM_DEFAULT: u64 = 1 << 30;

/// `defaultPrefixCacheAsk`, with GLM's unnamed ask pinned at `GLM_PREFIX_CACHE_MEM_DEFAULT`.
pub fn defaultPrefixCacheAskFor(config: *const model_mod.ModelConfig, requested: u64, explicit: bool, session: u64) u64 {
    if (config.isGlm5() and !explicit and requested == PREFIX_CACHE_MEM_DEFAULT) return GLM_PREFIX_CACHE_MEM_DEFAULT;
    return defaultPrefixCacheAsk(requested, explicit, session);
}

/// Bytes one cached session at `ctx_tokens` holds: the capacity a request there reserves (one
/// widest rung past it, `reservedCacheTokens`), the rings, every ring checkpoint an entry keeps,
/// and the SSM checkpoints a cold prefill of that length retains.
pub fn oneSessionEntryBytes(config: *const model_mod.ModelConfig, kv_bits: u64, ctx_tokens: u64, chunk: u64) u64 {
    const ring = config.swaRingBytes() +| @as(u64, prefix_cache_mod.RING_CHECKPOINT_MAX) *| config.swaRingCheckpointBytes();
    return sessionBytesPerToken(config, kv_bits) *| (ctx_tokens +| @as(u64, PREFILL_CHUNK_LADDER[0])) +|
        config.qsaRingBytes() +| kvBytesPerTokenAtBits(ring, kv_bits) +| retainedSsmCheckpointBytes(config, ctx_tokens, 0, chunk);
}

/// What an unnamed hot-cache budget grows toward, and the room the machine leaves it.
pub const OneSession = struct {
    /// One cached session at the working context (`oneSessionEntryBytes`).
    session: u64,
    /// The ceiling less the weights, `claim` and `bill`.
    room: u64,
    /// Admission's bill for a cold prompt that fills the working context.
    bill: u64,
    /// Page cache that must stay resident beside the weights.
    claim: u64,

    /// The session when the room holds it. Else the room, at most half the bill: a session that
    /// outgrows the cache is trimmed by a real copy beside the entry it replaces and its live KV.
    pub fn ask(self: OneSession) u64 {
        if (self.session <= self.room) return self.session;
        return @min(self.room, self.bill / 2);
    }
};

/// `claim` (the qwen4_exp n-gram table, `qwen4_exp.page_cache_claim`) is billed as GPU memory: the
/// wired KV would otherwise evict the rows every decode step gathers. `chunk` is the width the
/// clamp reserves, so a full cache still admits a cold full-context prompt there.
pub fn oneSessionFor(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, claim: u64, ctx_tokens: u64, chunk: u64) OneSession {
    const bill = prefillNeededAtChunk(config, ctx_tokens, 0, kv_bits, chunk, .{ .mtp_on = mtpHeadDefaultOn(config) });
    return .{
        .session = oneSessionEntryBytes(config, kv_bits, ctx_tokens, chunk),
        .room = ceiling -| (active_mem +| claim +| bill),
        .bill = bill,
        .claim = claim,
    };
}

fn logOneSessionAsk(ask: u64, ctx_tokens: u64, one: OneSession) void {
    log.info("[hot-cache] budget {d} MB for one session at the working context ({d} tokens, {d} MB; room {d} MB beside the weights, {d} MB of n-gram page cache and a full-context prompt's {d} MB) — --prefix-cache-mem overrides\n", .{
        ask >> 20, ctx_tokens, one.session >> 20, one.room >> 20, one.claim >> 20, one.bill >> 20,
    });
}

/// `revise` marks a post-load re-clamp (#364): the cache's own resident bytes come off the
/// machine read so it never bills itself, and the budget lines stay quiet.
pub fn prefixCacheMemForLoad(config: *model_mod.ModelConfig, requested: u64, revise: scheduler_mod.BudgetRevise, idle_out: *u64) u64 {
    // RAM-first default: no idle allowance. Written first so every early return carries it.
    idle_out.* = 0;
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    active_mem -|= @as(usize, @intCast(@min(revise.exclude_bytes, std.math.maxInt(usize))));
    const kv_bits: u64 = defaultKvBits(config);
    // Arch gate: the budget resolver changed three inputs at once (static ceiling, the
    // load-time context resolver, the floor-width reserve), all measured on qwen4_exp alone.
    // Ungated is the previous `prefixCacheMemForLoad`.
    if (!config.longCtxGated()) {
        const chunk: u64 = pinPrefillChunk(config);
        const ctx_tokens: u64 = getEffectiveContextLength(config);
        const ceiling = currentGpuMemoryCeiling(config, active_mem);
        // `statePerTokenBilled` is 0 off qwen4_exp.
        const ctx_kv: u64 = (kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(config)) *|
            ctx_tokens +| slotRingBytes(config, kv_bits);
        // Context sizing and the chunk pin read the raw ask, so neither moves with this one.
        const one = oneSessionFor(config, kv_bits, ceiling, active_mem, qwen4_mod.page_cache_claim.load(.acquire), ctx_tokens, chunk);
        const ask = defaultPrefixCacheAskFor(config, requested, prefix_cache_mem_explicit, one.ask());
        const clamped = clampedPrefixCacheMem(
            ask,
            ceiling,
            active_mem,
            ctx_kv,
            prefillTransientReserve(config, kv_bits, chunk),
        );
        publishResolvedPrefixCacheMem(clamped);
        if (revise.quiet) return clamped;
        if (ask != requested) logOneSessionAsk(ask, ctx_tokens, one);
        if (ask > 0 and clamped < ask) {
            log.info("[hot-cache] budget clamped {d} -> {d} MB (weights + ctx KV + prefill reserve vs GPU ceiling)\n", .{ ask >> 20, clamped >> 20 });
        } else if (requested == 0) {
            log.info("[hot-cache] budget capped at {d} MB (no --prefix-cache-mem; weights + ctx KV + prefill reserve vs GPU ceiling)\n", .{clamped >> 20});
        }
        return clamped;
    }
    // SSD-first takes the budget before the RAM-first plan: its resident entry IS the live KV.
    const ssd_chunk: u64 = pinPrefillChunk(config);
    // Not `getEffectiveContextLength`: on an auto boot it is still unpinned here.
    const ssd_ctx_kv: u64 = ssdFirstSessionKvBytes(config, kv_bits, staticGpuMemoryCeiling(), active_mem, @intCast(ssd_chunk));
    // Static ceiling on both arms (the budget is a property of the machine), and the clamp
    // reserve at the same width `planHotCache` reserves for: the ladder floor on an arch that
    // re-bills per request. The session is billed at the pinned width, the budget at the floor.
    const ssd_clamp_reserve: u64 = prefillTransientReserve(config, kv_bits, clampReserveWidth(config, @intCast(ssd_chunk)));
    if (ssdFirstBudgetForLoad(config, requested, staticGpuMemoryCeiling(), active_mem, ssd_ctx_kv, ssd_clamp_reserve, idle_out, revise.quiet)) |b| return b;
    // Pin first, then hand the pinned width in as the override.
    const pinned: u32 = pinPrefillChunk(config);
    // Not `getEffectiveContextLength`: still a placeholder on an auto boot.
    const ctx_tokens = ramFirstContextForLoad(config, kv_bits, active_mem, pinned);
    // Static ceiling, not the live one: the budget must be reproducible boot to boot.
    const one = oneSessionFor(config, kv_bits, staticGpuMemoryCeiling(), active_mem, qwen4_mod.page_cache_claim.load(.acquire), ctx_tokens, clampReserveWidth(config, pinned));
    const ask = defaultPrefixCacheAskFor(config, requested, prefix_cache_mem_explicit, one.ask());
    const plan = planHotCache(
        config,
        kv_bits,
        staticGpuMemoryCeiling(),
        active_mem,
        ctx_tokens,
        sizerCtxKvBytes(config, kv_bits),
        ask,
        pinned,
    );
    publishResolvedPrefixCacheMem(plan.budget);
    if (revise.quiet) return plan.budget;
    // Say so when the box is under external pressure right now, but do not shrink the budget for it.
    const live_ceiling: u64 = currentGpuMemoryCeiling(config, active_mem);
    if (live_ceiling < staticGpuMemoryCeiling()) {
        const free_gb = @as(f64, @floatFromInt(live_ceiling -| active_mem)) / (1024.0 * 1024.0 * 1024.0);
        log.info("[hot-cache] budget {d} MB (static ceiling); free at load {d:.1} GB — live admission will evict as needed\n", .{ plan.budget >> 20, free_gb });
    }
    if (ask != requested) logOneSessionAsk(ask, ctx_tokens, one);
    if (ask > 0 and plan.budget < ask) {
        log.info("[hot-cache] budget clamped {d} -> {d} MB (chunk {d}, reserve at width {d} = {d} MB, ctx KV {d} MB)\n", .{ ask >> 20, plan.budget >> 20, plan.chunk, plan.reserve_chunk, plan.reserve >> 20, plan.ctx_kv >> 20 });
    } else if (requested == 0) {
        log.info("[hot-cache] budget capped at {d} MB (no --prefix-cache-mem; chunk {d}, reserve at width {d} = {d} MB, ctx KV {d} MB)\n", .{ plan.budget >> 20, plan.chunk, plan.reserve_chunk, plan.reserve >> 20, plan.ctx_kv >> 20 });
    }
    return plan.budget;
}

/// The SSD bytes one cached token costs this model at these settings, from the serialized geometry:
/// the KV rows at the served width, the one pooled QSA history the entry writes, and the MTP head's KV
/// and history where the head is on. Execution state (the QSA score bank) is
/// RAM-only and never enters it. A recurrent state is kept on every `POINT_GRID` tokens, so its share per token
/// is billed; DFlash2's window and MiMo's rings are fixed per entry.
pub fn diskBytesPerToken(config: *const model_mod.ModelConfig, kv_bits: u64) u64 {
    if (config.dsv41) |c| return dsv41_prefix.diskBytesPerToken(&c, config.dsparkOn());
    const points = (config.ssmCheckpointBytes() +| kv_disk_cache.POINT_GRID - 1) / kv_disk_cache.POINT_GRID;
    const rows = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +| config.qsaHistoryBytesPerToken() +| points;
    if (config.isGlm5() or !config.longCtxGated() or !mtpHeadDefaultOn(config)) return rows;
    const layers = config.attnCacheLayerCount();
    if (layers == 0) return rows;
    return rows +| mtpHeadKvBytesPerToken(config) +| config.qsaHistoryBytesPerToken() / layers;
}

/// The context the SSD budget is sized for: the model's pinned served context (`pinAutoContext`, idempotent
/// and the one startup and `/v1/load-model` run anyway), so the Allocating line, `/props` and `/v1/models`
/// name the same number. The weights are resident at this point, so the memory read is the one they would take.
fn diskContextForLoad(config: *model_mod.ModelConfig) u64 {
    return pinAutoContext(config);
}

/// The SSD budget the model gets at load, in bytes (0 = tier off); logs the `Allocating` line, or why
/// there is none. Runs on the inference thread beside `prefixCacheMemForLoad`.
/// `held` is what this model's tier already occupies under `dir`: it counts as available, or a restart on a
/// nearly full volume would evict what the last boot stored.
pub fn prefixCacheDiskForLoad(config: *model_mod.ModelConfig, dir: []const u8, held: u64) ?kv_disk_cache.DefaultDiskBudget {
    const kv_bits = defaultKvBits(config);
    const bpt = diskBytesPerToken(config, kv_bits);
    const ctx = if (prefix_cache_disk_explicit) 0 else diskContextForLoad(config);
    // An unreadable volume is no reason to refuse: the tier re-checks before every store.
    const free: u64 = if (kv_disk_cache.volumeSpaceNear(dir)) |v| v.free else std.math.maxInt(u64);
    var buf: [384]u8 = undefined;
    const plan = kv_disk_cache.resolveDiskBudget(prefix_cache_disk_explicit, prefix_cache_disk_bytes, prefix_cache_capacity, ctx, bpt, free, held) orelse {
        log.info("{s}\n", .{kv_disk_cache.diskOffLine(&buf, free, dir)});
        return null;
    };
    log.info("{s}\n", .{kv_disk_cache.allocatingDiskLine(&buf, plan, prefix_cache_capacity, ctx, bpt, dir)});
    return plan;
}

test "diskBytesPerToken: the serialized geometry, whatever the QSA execution switches say" {
    const t = std.testing;
    const saved_mtp = configured_mtp;
    defer configured_mtp = saved_mtp;
    const saved_fused = qwen4_qsa_select.qsa_score_fused_override;
    defer qwen4_qsa_select.qsa_score_fused_override = saved_fused;
    var q = qwen4DeployedTestConfig();
    // kv8 rows plus the one pooled history; the head is off.
    configured_mtp = false;
    // Plus its share of the recurrent state kept on every grid point.
    const points = (q.ssmCheckpointBytes() + kv_disk_cache.POINT_GRID - 1) / kv_disk_cache.POINT_GRID;
    try t.expect(points > 0);
    try t.expectEqual(@as(u64, 13_824) + points, diskBytesPerToken(&q, 8));
    const head = mtpHeadKvBytesPerToken(&q) + q.qsaHistoryBytesPerToken() / q.attnCacheLayerCount();
    try t.expect(head > 0);
    configured_mtp = true;
    try t.expectEqual(@as(u64, 13_824) + points + head, diskBytesPerToken(&q, 8));
    try t.expect(diskBytesPerToken(&q, 16) > diskBytesPerToken(&q, 8));
    // The score bank is execution state, never in an entry.
    for ([_]?bool{ true, false }) |fused| {
        qwen4_qsa_select.qsa_score_fused_override = fused;
        for ([_]bool{ false, true }) |mtp| {
            configured_mtp = mtp;
            try t.expectEqual(@as(u64, 13_824) + points + (if (mtp) head else 0), diskBytesPerToken(&q, 8));
        }
    }
    qwen4_qsa_select.qsa_score_fused_override = false;
    configured_mtp = false;
    try t.expect(sessionBytesPerToken(&q, 8) > diskBytesPerToken(&q, 8) - points);

    // A ringed arch pays per token only on its global layers; its head is a fixed window.
    const mimo = mimoV2FlashBillConfig();
    for ([_]bool{ false, true }) |mtp| {
        configured_mtp = mtp;
        try t.expectEqual(kvBytesPerTokenAtBits(mimo.kvBytesPerToken(), 8), diskBytesPerToken(&mimo, 8));
        try t.expectEqual(kvBytesPerTokenAtBits(mimo.kvBytesPerToken(), 16), diskBytesPerToken(&mimo, 16));
    }

    // GLM: the latent rows (6,688 at kv8, 11,968 BF16) and the pooled index; DFlash2's window is fixed per entry.
    var glm = try model_mod.parseConfigFromJson(t.allocator, @embedFile("fixtures/glm5_config.json"));
    glm.dflash_loaded = false;
    const glm_points = (glm.ssmCheckpointBytes() + kv_disk_cache.POINT_GRID - 1) / kv_disk_cache.POINT_GRID;
    try t.expect(glm_points > 0);
    try t.expectEqual(@as(u64, 6_688) + glm_points, diskBytesPerToken(&glm, 8));
    try t.expectEqual(@as(u64, 5_280) + glm_points, diskBytesPerToken(&glm, 6));
    try t.expectEqual(@as(u64, 11_968) + glm_points, diskBytesPerToken(&glm, 16));
    glm.dflash_loaded = true;
    glm.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    try t.expectEqual(@as(u64, 6_688) + glm_points, diskBytesPerToken(&glm, 8));
}

test "diskContextForLoad pins the served context, so the Allocating line names what /props and /v1/models name" {
    const t = std.testing;
    var cfg = qwen4DeployedTestConfig();
    cfg.pinned_context = 0;
    const saved_ctx = server_config.max_context_size;
    defer server_config.max_context_size = saved_ctx;
    server_config.max_context_size = 0;
    const ctx = diskContextForLoad(&cfg);
    try t.expect(cfg.pinned_context != 0);
    try t.expectEqual(@as(u64, cfg.pinned_context), ctx);
    try t.expectEqual(@as(u64, getEffectiveContextLength(&cfg)), ctx);
}

test "a model whose SSD tier did not come up bills and budgets as with --prefix-cache-disk off" {
    const t = std.testing;
    var glm = try model_mod.parseConfigFromJson(t.allocator, @embedFile("fixtures/glm5_config.json"));
    glm.dflash_loaded = true;
    glm.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    const saved = .{ prefix_cache_capacity, ssm_checkpoint_stride, ssm_checkpoint_max };
    defer {
        prefix_cache_capacity = saved[0];
        ssm_checkpoint_stride = saved[1];
        ssm_checkpoint_max = saved[2];
    }
    prefix_cache_capacity = 32;
    ssm_checkpoint_stride = 256;
    ssm_checkpoint_max = 16;
    const warm: WarmPrefix = .{ .checkpoints = 4 };

    // RAM off: the SSD tier is the only retention, so with it GLM keeps checkpoints and without it none.
    const tiers = TierGuard.pin(false, 20 << 30);
    defer tiers.restore();
    try t.expect(diskTierOn(&glm));
    const with_tier = glmCommitStateBytes(&glm, 65536, warm, 8);
    try t.expect(with_tier > 0);
    // Declined at load (no room, a failed init): the same answers as an explicit `--prefix-cache-disk off`.
    glm.prefix_cache_disk_declined = true;
    try t.expect(!diskTierOn(&glm));
    prefix_cache_disk_bytes = 0;
    glm.prefix_cache_disk_declined = false;
    const off_flag = .{ glmCommitStateBytes(&glm, 65536, warm, 8), glmPrefixStateBytes(&glm, 65536, warm) };
    prefix_cache_disk_bytes = 20 << 30;
    glm.prefix_cache_disk_declined = true;
    try t.expectEqual(off_flag[0], glmCommitStateBytes(&glm, 65536, warm, 8));
    try t.expectEqual(off_flag[1], glmPrefixStateBytes(&glm, 65536, warm));
    try t.expectEqual(@as(u64, 0), off_flag[0]);
    // The RAM sizing no longer takes the SSD-first arm for this model either.
    var cfg = qwen4DeployedTestConfig();
    const saved_over = prefix_cache_mod.ssd_first_override;
    defer prefix_cache_mod.ssd_first_override = saved_over;
    prefix_cache_mod.ssd_first_override = true;
    prefix_cache_ram_enabled = true;
    var idle: u64 = 0;
    try t.expect(ssdFirstBudgetForLoad(&cfg, 1 << 30, 100 << 30, 40 << 30, 4 << 30, 1 << 30, &idle, true) != null);
    cfg.prefix_cache_disk_declined = true;
    try t.expect(ssdFirstBudgetForLoad(&cfg, 1 << 30, 100 << 30, 40 << 30, 4 << 30, 1 << 30, &idle, true) == null);
}

test "admission's headroom is the ceiling less MLX's bytes, and never wraps" {
    const t = std.testing;
    const gib: u64 = 1 << 30;
    try t.expectEqual(@as(u64, 21 * gib), admissionAvailable(100 * gib, 79 * gib));
    try t.expectEqual(@as(u64, 0), admissionAvailable(100 * gib, 101 * gib));
}

test "defaultPrefixCacheAsk: an unnamed budget holds one session at the working context, never under 2 GB" {
    const t = std.testing;
    const gb: u64 = 1024 * 1024 * 1024;
    const session: u64 = 262144 * 24576; // 262k-token qwen4_exp session, ~6 GB
    try t.expectEqual(session, defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, false, session));
    // A short working context keeps the old floor.
    try t.expectEqual(PREFIX_CACHE_MEM_DEFAULT, defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, false, gb));
    // An operator's number stands, even one equal to the default.
    try t.expectEqual(PREFIX_CACHE_MEM_DEFAULT, defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, true, session));
    try t.expectEqual(32 * gb, defaultPrefixCacheAsk(32 * gb, true, session));
    try t.expectEqual(@as(u64, 0), defaultPrefixCacheAsk(0, true, session));
    // Any other ask is someone's choice and stands.
    try t.expectEqual(256 * 1024 * 1024, defaultPrefixCacheAsk(256 * 1024 * 1024, false, session));
}

test "defaultPrefixCacheAsk: the machine's headroom still caps the defaulted ask" {
    const t = std.testing;
    const gb: u64 = 1024 * 1024 * 1024;
    const session: u64 = 262144 * 24576;
    const ask = defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, false, session);
    // 16 GB ceiling, 11 GB weights, 1 GB live context, 1.5 GB prefill reserve: 2.5 GB left.
    try t.expectEqual(gb * 5 / 2, clampedPrefixCacheMem(ask, 16 * gb, 11 * gb, gb, gb * 3 / 2));
    // A big machine gets the whole session.
    try t.expectEqual(session, clampedPrefixCacheMem(ask, 200 * gb, 80 * gb, gb, 2 * gb));
}

test "oneSessionEntryBytes: a cached session is billed with its SSM checkpoints and the capacity it reserved" {
    const t = std.testing;
    const guard = qsaScoreFusedOffGuard();
    defer guard.deinit();
    var cfg = qwen4DeployedTestConfig();
    const ctx: u64 = 262_144;
    const chunk: u64 = 8192;
    cfg.pinned_context = @intCast(ctx);
    const entry = oneSessionEntryBytes(&cfg, 16, ctx, chunk);
    try t.expect(cfg.ssmCheckpointBytes() > 0);
    const fixed = cfg.qsaRingBytes() + retainedSsmCheckpointBytes(&cfg, ctx, 0, chunk);
    // A request at the working context reserves past it (`reservedCacheTokens`), and the entry
    // shares that capacity: no session the context admits outgrows the default.
    for ([_]u64{ ctx - 4000, ctx }) |seq| {
        const reserved = reservedCacheTokens(seq, 8192, PREFILL_CHUNK_LADDER[0], ctx);
        try t.expect(reserved > ctx);
        try t.expect(sessionBytesPerToken(&cfg, 16) * reserved + fixed <= entry);
    }
    try t.expect(defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, false, entry) >= entry);
}

test "oneSessionEntryBytes: a ringed session is billed with its ring and every checkpoint an entry keeps" {
    const t = std.testing;
    var cfg = mimoV2FlashBillConfig();
    const ctx: u64 = 454_656;
    cfg.pinned_context = @intCast(ctx);
    const ring = cfg.swaRingBytes() + prefix_cache_mod.RING_CHECKPOINT_MAX * cfg.swaRingCheckpointBytes();
    try t.expect(ring > 0);
    const entry = oneSessionEntryBytes(&cfg, 8, ctx, 2048);
    try t.expectEqual(sessionBytesPerToken(&cfg, 8) * (ctx + PREFILL_CHUNK_LADDER[0]) + kvBytesPerTokenAtBits(ring, 8), entry);
    // Past the 2 GB default at kv8: the sessions that cold-prefilled every turn.
    try t.expect(entry > PREFIX_CACHE_MEM_DEFAULT);
}

test "oneSessionFor: an unnamed budget grows only into room the n-gram table and a full-context prompt leave" {
    const t = std.testing;
    const guard = qsaScoreFusedOffGuard();
    defer guard.deinit();
    const GiB: u64 = 1 << 30;
    var cfg = qwen4DeployedTestConfig();
    const ctx: u64 = 1_048_576;
    cfg.pinned_context = @intCast(ctx);
    const chunk: u64 = 512;
    const weights: u64 = 51 * GiB;
    const table: u64 = 30 * GiB;

    const roomy = oneSessionFor(&cfg, 8, 400 * GiB, weights, table, ctx, chunk);
    try t.expectEqual(roomy.session, roomy.ask());

    // This box: the session fits beside the weights, not beside the table as well.
    const ceiling: u64 = 111_808 * (1 << 20);
    const bare = oneSessionFor(&cfg, 8, ceiling, weights, 0, ctx, chunk);
    try t.expectEqual(bare.session, bare.ask());
    const with_table = oneSessionFor(&cfg, 8, ceiling, weights, table, ctx, chunk);
    const ask = with_table.ask();
    try t.expect(ask > PREFIX_CACHE_MEM_DEFAULT and ask < with_table.session);
    try t.expect(weights + table + ask + with_table.bill <= ceiling);

    // A room past half the prompt's bill stops at the half: the trim a session makes when it
    // outgrows the cache copies up to the budget beside the entry it replaces and the live KV.
    const wide = oneSessionFor(&cfg, 8, weights + roomy.bill + roomy.session - 1, weights, 0, ctx, chunk);
    try t.expect(wide.bill / 2 < wide.room);
    try t.expectEqual(wide.bill / 2, wide.ask());

    // No room: nothing grows and the 2 GB default stands.
    const none = oneSessionFor(&cfg, 8, weights + table, weights, table, ctx, chunk);
    try t.expectEqual(@as(u64, 0), none.ask());
    try t.expectEqual(PREFIX_CACHE_MEM_DEFAULT, defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, false, none.ask()));
}

test "oneSessionFor: a full MiMo cache leaves a cold full-context prompt its bill at the pinned width" {
    const t = std.testing;
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    const GiB: u64 = 1 << 30;
    var cfg = mimoV2FlashBillConfig();
    const ctx: u64 = 454_656;
    cfg.pinned_context = @intCast(ctx);
    const chunk: u64 = 2048;
    const weights: u64 = 97 * GiB;
    const ceiling: u64 = weights + 14 * GiB;
    const saved = server_config.default_force_mtp;
    defer server_config.default_force_mtp = saved;
    server_config.default_force_mtp = false;
    // Off, then the served default: the heads on for every request that leaves MTP out.
    for ([_]?bool{ false, null }) |mtp| {
        cfg.mtp_override = mtp;
        const ask = defaultPrefixCacheAsk(PREFIX_CACHE_MEM_DEFAULT, false, oneSessionFor(&cfg, 8, ceiling, weights, 0, ctx, chunk).ask());
        try t.expect(ask > PREFIX_CACHE_MEM_DEFAULT);
        // The ungated arch refuses rather than evicts to admit, so the cache must not take that bill.
        try t.expect(prefillNeededAtChunk(&cfg, ctx, 0, 8, chunk, .{ .mtp_on = mtp == null }) <= ceiling - weights - ask);
    }
    try t.expect(oneSessionFor(&cfg, 8, ceiling, weights, 0, ctx, chunk).bill > prefillNeededAtChunk(&cfg, ctx, 0, 8, chunk, .{}));
}

test "a qwen4_exp SSD restore is credited: its first grow bills one eval window of the restored rows" {
    // The SSD tier fills buffers the slot owns at exactly the restored length (`LookupResult.slot_owned`),
    // so the first append grows every layer beside its old buffer and frees that buffer at the layer's
    // eval window. A share is billed the whole copy instead.
    const t = std.testing;
    const guard = qsaScoreFusedOffGuard();
    defer guard.deinit();
    var cfg = qwen4DeployedTestConfig();
    cfg.pinned_context = 1 << 20;
    const seq: u64 = 400_000;
    const restored: u64 = seq - 2000;
    const disk = WarmPrefix{ .matched_tokens = restored, .capacity_tokens = restored, .will_donate = true };
    const shared = WarmPrefix{ .matched_tokens = restored, .capacity_tokens = restored };
    const d = prefillRequestTerms(&cfg, seq, 2048, 8, 1024, disk);
    try t.expectEqual(restored * kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8), d.shared_resident_bytes);
    // Twelve caching layers, one inside each four-layer eval window.
    try t.expectEqual(d.shared_resident_bytes / 12, d.grow_coexist_bytes);
    try t.expectEqual(
        prefillNeededAtChunk(&cfg, seq, 2048, 8, 1024, shared) - (d.shared_resident_bytes - d.grow_coexist_bytes) * 5 / 4,
        prefillNeededAtChunk(&cfg, seq, 2048, 8, 1024, disk),
    );
}

test "clampedPrefixCacheMem: the budget never exceeds what the weights leave under the ceiling" {
    const t = std.testing;
    const GB: u64 = 1 << 30;
    // The crash's numbers: 96 GB ceiling, ~70 GB weights, ~6.4 GB of 262k-ctx
    // KV, ~4 GB prefill reserve, --prefix-cache-mem 40 GB → ~15.6 GB budget.
    const crash = clampedPrefixCacheMem(40 * GB, 96 * GB, 70 * GB, 6 * GB + (400 << 20), 4 * GB);
    try t.expect(crash >= 10 * GB and crash < 16 * GB);
    // Small model with plenty of headroom: the request is untouched.
    try t.expectEqual(8 * GB, clampedPrefixCacheMem(8 * GB, 96 * GB, 20 * GB, 2 * GB, 2 * GB));
    // requested == 0 (byte cap disabled today): headroom still bounds it.
    try t.expectEqual(96 * GB - 80 * GB, clampedPrefixCacheMem(0, 96 * GB, 70 * GB, 6 * GB, 4 * GB));
    // No headroom at all: never 0 — initWithMem reads 0 as "no byte cap".
    try t.expectEqual(@as(u64, 1), clampedPrefixCacheMem(40 * GB, 64 * GB, 70 * GB, 6 * GB, 4 * GB));
}

test "bf16 expert cache admission bills cache workspace bounce and resident trunk" {
    const gb: u64 = 1_000_000_000;
    const fill_peak: u64 = 2 * 5_033_164_800 + 127 * 9_830_400 + 536_870_912;
    const serving: u64 = 9 * gb;
    const needed = 10 * gb + 16 * gb + 60 * gb + 536_870_912 + fill_peak + serving;
    try std.testing.expect(expertStreamingLoadFits(needed, 10 * gb, 16 * gb, 60 * gb, 536_870_912, fill_peak, serving));
    try std.testing.expect(!expertStreamingLoadFits(needed - 1, 10 * gb, 16 * gb, 60 * gb, 536_870_912, fill_peak, serving));
}

test "DeepSeek's streamed request bill brackets the prefill peaks measured on the M5 Max" {
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.max_position_embeddings = 1_048_576;
    config.expert_fill_peak_bytes = scheduler_mod.billedFillPeakBytes(&config, 9_911 << 20);
    // No decoder pool to lend: the bill is the working set's, as measured.
    config.expert_borrow_bytes = 0;
    try std.testing.expectEqual(prefill_plan.TAIL_MERGE_MAX - 1, dsv41_config.tail_merge_rows);
    // Live per-chunk `peak_bytes - active_bytes` (`SUSHI_PREFILL_CHUNK_PEAK`), CED prefill, `--ssd-budget-gb 94`, the STQ d4
    // pack, DSpark on: `chunk` is the configured width, `seq` where the chunk ended (a last chunk absorbs its tail).
    const measured = [_]struct { seq: u64, chunk: u64, peak_mb: u64 }{
        .{ .seq = 7168, .chunk = 1024, .peak_mb = 3_560 },
        .{ .seq = 8675, .chunk = 1024, .peak_mb = 4_049 },
        .{ .seq = 18432, .chunk = 512, .peak_mb = 2_635 },
        .{ .seq = 34625, .chunk = 512, .peak_mb = 2_997 },
        .{ .seq = 108032, .chunk = 512, .peak_mb = 2_600 },
        .{ .seq = 108675, .chunk = 512, .peak_mb = 2_740 },
        // Transient peaks of MLX's own bytes, state included, binary 26a28040, budget 108.
        .{ .seq = 2048, .chunk = 2048, .peak_mb = 4_907 },
        .{ .seq = 225_000, .chunk = 2048, .peak_mb = 5_124 },
    };
    for (measured) |m| {
        const bill = prefillNeededAtChunk(&config, m.seq, 16, 16, m.chunk, .{});
        const peak = m.peak_mb * 1_000_000;
        // Under-billing is a Metal OOM: at least a tenth over the peak. The upper bar keeps the net small.
        try std.testing.expect(bill >= peak + peak / 10);
        try std.testing.expect(bill <= peak * 4);
    }
}

test "DeepSeek prices its chunk per request: 2048 while the bill fits, then 1024, then 512, never wider" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.max_position_embeddings = 1_048_576;
    config.num_attention_heads = 64;
    config.expert_fill_peak_bytes = scheduler_mod.billedFillPeakBytes(&config, 9_911 << 20);
    // No decoder pool to lend: the bill is the working set's, as measured.
    config.expert_borrow_bytes = 0;
    const MB: u64 = 1 << 20;
    // What `--ssd-budget-gb 96` on the 128 GB M5 Max leaves to serve, as its pin measured it.
    const at96: u64 = 110_100 * MB - 97_080 * MB;
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&config, 8192, 2048, 16, at96, 512, 0, .{}));
    // Its attention scores only the picked keys, so a long prompt keeps the width while the bill fits.
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&config, 400_000, 2048, 16, at96, 512, 0, .{}));
    const long: u64 = 524_288;
    const at1024 = prefillNeededAtChunk(&config, long, 2048, 16, 1024, .{});
    const only_narrow = prefillNeededAtChunk(&config, long, 2048, 16, 512, .{});
    try t.expect(prefillNeededAtChunk(&config, long, 2048, 16, 2048, .{}) > at1024 and at1024 > only_narrow);
    try t.expectEqual(@as(u32, 1024), chooseRequestPrefillChunk(&config, long, 2048, 16, at1024, 512, 0, .{}));
    try t.expectEqual(@as(u32, 512), chooseRequestPrefillChunk(&config, long, 2048, 16, only_narrow, 512, 0, .{}));
    try t.expectEqual(@as(u32, 512), chooseRequestPrefillChunk(&config, 8192, 2048, 16, at96, 512, 512, .{}));
    var buf: [256]u8 = undefined;
    try t.expectEqualStrings("Prefill chunk: per request, up to 2048 at a short prompt (the widest rung each request's bill admits); load-time fallback 512\n", prefillChunkLoadLine(&buf, &config, 512, 8192).?);
    // The ladder never narrows the advertised context: it is priced at the 512 floor.
    try t.expectEqual(@as(u32, 974_848), advertisedContextAt(&config, 16, 110_100 * MB, 97_080 * MB, 512));
}

test "DeepSeek advertises the longest context its own request bill admits" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.max_position_embeddings = 1_048_576;
    config.num_attention_heads = 64;
    config.expert_fill_peak_bytes = scheduler_mod.billedFillPeakBytes(&config, 9_911 << 20);
    // No decoder pool to lend: the bill is the working set's, as measured.
    config.expert_borrow_bytes = 0;
    const GB: u64 = 1_000_000_000;
    // The request bill grows with the prompt (the state and the rows' growth), so a serving budget of exactly one
    // prompt's bill advertises that prompt and not one token less.
    for ([_]u64{ 8192, 65_536, 131_072, 262_144, 524_288, 1_048_576 }) |seq| {
        for ([_]u64{ 512, 1024 }) |chunk| {
            const bill = prefillNeededAtChunk(&config, seq, 0, 16, chunk, .{});
            try t.expect(memoryContextAtChunk(&config, 16, 100 * GB + bill, 100 * GB, chunk) >= seq);
            try t.expect(memoryContextAtChunk(&config, 16, 100 * GB + bill - 1, 100 * GB, chunk) < seq);
        }
    }
    // At 96 the whole 1M context fits the fallback width's bill.
    const MB: u64 = 1 << 20;
    try t.expectEqual(@as(u32, 1_048_576), memoryContextAtChunk(&config, 16, 110_100 * MB, 97_080 * MB, 512));
}

test "a streamed DeepSeek bills its workspace at load and no fill peak at load or per request" {
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.max_position_embeddings = 1_048_576;
    config.expert_cache_bytes = 93_993 << 20;
    config.expert_bounce_bytes = 512 << 20;
    config.expert_workspace_bytes = 3_525 << 20;
    config.expert_fill_peak_bytes = scheduler_mod.billedFillPeakBytes(&config, 9_911 << 20);
    const mib: u64 = 1 << 20;
    // No decoder pool to lend: the bill is the working set's, as measured.
    config.expert_borrow_bytes = 0;
    // A 100-token request pays its own transient, not the cache layer and a second workspace.
    try std.testing.expect(prefillNeededAtChunk(&config, 100, 512, 16, 512, .{}) < 2048 * mib);
    const resident: u64 = 8_295 * mib;
    const held = resident + config.expert_cache_bytes + config.expert_bounce_bytes + config.expert_workspace_bytes + config.expertPoolBytes();
    const serving = expertStreamingServingBytes(&config);
    try std.testing.expect(expertStreamingConfigFits(&config, held + serving, 0, resident));
    try std.testing.expect(!expertStreamingConfigFits(&config, held + serving - 1, 0, resident));
}

test "DeepSeek's live headroom and its load fit each count the decode pool the ledger parks once" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.max_position_embeddings = 1_048_576;
    // The parked pool is in MLX's cache, not its active bytes, so `ceiling - active` does not see it.
    try t.expectEqual(@as(u64, 10 << 30) - config.expertPoolBytes(), dsv41HeadroomOf(10 << 30, &config));
    try t.expectEqual(@as(u64, 0), dsv41HeadroomOf(1 << 20, &config));
    // The load fit holds the pool in its staging term, so the serving term it adds is the bill without it.
    config.expert_borrow_bytes = 60_000_000_000;
    const bill = prefillNeededAtChunk(&config, expertStreamingPlannedSeq(&config), 2048, defaultKvBits(&config), 512, .{});
    try t.expectEqual(bill - config.expertPoolBytes(), expertStreamingServingBytes(&config));
}

test "DeepSeek's request bill takes its prefill working set out of the decoder pools a prefill may release" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.max_position_embeddings = 1_048_576;
    const c = config.dsv41.?;
    const seq: u64 = 524_288;
    const state = ((seq + 2048) * c.stateBytesPerToken() + c.stateFixedBytes(true) + dsv41CheckpointReserve(&config, seq, 2048)) * 5 / 4;
    const work = c.chunkCap(2048, seq) + config.expertPoolBytes();
    const decode = c.forwardCap(dsv41_borrow.verify_rows, seq);
    // No pool to lend: the whole working set is billed, as before.
    try t.expectEqual(state + work, prefillNeededAtChunk(&config, seq, 2048, 16, 2048, .{}));
    // Pools that cover part of it: admission pays only the rest.
    config.expert_borrow_bytes = work / 2;
    try t.expectEqual(state + work - work / 2, prefillNeededAtChunk(&config, seq, 2048, 16, 2048, .{}));
    // Pools that cover all of it: the state, a decode forward and the decode pool it leaves parked, at any chunk.
    config.expert_borrow_bytes = 60_000_000_000;
    for ([_]u64{ 512, 1024, 2048 }) |chunk| {
        const ck = ((seq + 2048) * c.stateBytesPerToken() + c.stateFixedBytes(true) + dsv41CheckpointReserve(&config, seq, chunk)) * 5 / 4;
        try t.expectEqual(ck + decode + config.expertPoolBytes(), prefillNeededAtChunk(&config, seq, 2048, 16, chunk, .{}));
    }
    // The widest rung is then always affordable, and a chunk boundary never steps down for the working set.
    const MB: u64 = 1 << 20;
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&config, seq, 2048, 16, state + decode + config.expertPoolBytes(), 512, 0, .{}));
    var st: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&config, 16, seq, 64 * MB, 2048, 2048, &st));
    config.expert_borrow_bytes = 0;
    var st0: mtp_plan.AdaptiveWidthState = .{};
    try t.expect(adaptivePrefillWidth(&config, 16, seq, 64 * MB, 2048, 2048, &st0) < 2048);
}

test "with its decoder pools lendable, DeepSeek at --ssd-budget-gb 117 with DSpark on wired 124000 advertises its whole context" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.max_position_embeddings = 1_048_576;
    config.num_attention_heads = 64;
    const saved = .{ prefix_cache_ram_enabled, prefix_cache_disk_bytes };
    defer {
        prefix_cache_ram_enabled = saved[0];
        prefix_cache_disk_bytes = saved[1];
    }
    // The served default: no RAM tier, the SSD tier on, so restore points go to SSD as a prefill takes them.
    prefix_cache_ram_enabled = false;
    prefix_cache_disk_bytes = 20 << 30;
    const MB: u64 = 1 << 20;
    // The budget-100 boot on wired 124000 read a 122,170 MiB ceiling and 100,328 MiB active; 17 GiB more at 117.
    const ceiling: u64 = 122_170 * MB;
    // The budget now holds the decode pool, parked in MLX's cache: the same budget leaves active that much lower.
    const active: u64 = (100_328 + 17 * 1024) * MB - (1 << 30);
    // Its expert cache, 13.57 GB under the budget (94% of the experts), split evenly: the decoder half is lendable.
    config.expert_borrow_bytes = (117 * (1 << 30) - 13_570_000_000) / 2;
    try t.expectEqual(@as(u32, 974_848), advertisedContextAt(&config, 16, ceiling, active, 2048));
    // Billing the working set to the request, with nothing to lend, advertises well under the whole context.
    config.expert_borrow_bytes = 0;
    try t.expectEqual(@as(u32, 390_144), advertisedContextAt(&config, 16, ceiling, active, 512));
}

test "bf16 expert load admission includes 65k KV and QSA serving terms" {
    const fused_guard = qsaScoreFusedOffGuard();
    defer fused_guard.deinit();
    var config = qwen4RequestTestConfig();
    config.expert_streaming = true;
    config.expert_fill_peak_bytes = 11_852_000_000;
    const saved_ctx = server_config.max_context_size;
    defer server_config.max_context_size = saved_ctx;
    server_config.max_context_size = 65_536;
    const saved_kv = configured_kv_quant;
    defer configured_kv_quant = saved_kv;
    configured_kv_quant = transformer_mod.KVQuantConfig.affine(8);
    const serving = expertStreamingServingBytes(&config);
    try std.testing.expect(serving >= kvBytesPerTokenAtBits(config.kvBytesPerToken(), 8) * 65_536);
    const fixed: u64 = 10_000_000_000 + 16_000_000_000 + 60_000_000_000 + 536_870_912 + config.expert_fill_peak_bytes;
    try std.testing.expect(expertStreamingLoadFits(fixed + serving, 10_000_000_000, 16_000_000_000, 60_000_000_000, 536_870_912, config.expert_fill_peak_bytes, serving));
    try std.testing.expect(!expertStreamingLoadFits(fixed + serving - 1, 10_000_000_000, 16_000_000_000, 60_000_000_000, 536_870_912, config.expert_fill_peak_bytes, serving));
}

test "a streamed load bills the narrowest rung the per-request ladder can take" {
    const fused_guard = qsaScoreFusedOffGuard();
    defer fused_guard.deinit();
    var config = qwen4RequestTestConfig();
    config.expert_streaming = true;
    // The fit check runs before the post-load sizer pins a chunk.
    config.pinned_prefill_chunk = 0;
    const saved_ctx = server_config.max_context_size;
    defer server_config.max_context_size = saved_ctx;
    server_config.max_context_size = 32_768;
    const saved_kv = configured_kv_quant;
    defer configured_kv_quant = saved_kv;
    configured_kv_quant = transformer_mod.KVQuantConfig.affine(8);
    const saved_explicit = generate_mod.prefill_chunk_explicit;
    const saved_chunk = generate_mod.prefill_chunk_override;
    defer {
        generate_mod.prefill_chunk_explicit = saved_explicit;
        generate_mod.prefill_chunk_override = saved_chunk;
    }
    const saved_per_request = per_request_chunk_override;
    defer per_request_chunk_override = saved_per_request;
    generate_mod.prefill_chunk_explicit = false;
    per_request_chunk_override = true;
    const floor: u64 = PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];
    const at = struct {
        fn bill(c: *const model_mod.ModelConfig, chunk: u64) u64 {
            return prefillNeededAtChunk(c, @max(32_768, chunk), 2048, 8, chunk, .{}) -| c.expert_fill_peak_bytes;
        }
    }.bill;
    try std.testing.expectEqual(at(&config, floor), expertStreamingServingBytes(&config));
    try std.testing.expect(at(&config, floor) < at(&config, 8192));
    // An explicit width caps the ladder, so the load proves the floor, or a narrower flag.
    generate_mod.prefill_chunk_explicit = true;
    generate_mod.prefill_chunk_override = 4096;
    try std.testing.expectEqual(at(&config, floor), expertStreamingServingBytes(&config));
    generate_mod.prefill_chunk_override = 256;
    try std.testing.expectEqual(at(&config, 256), expertStreamingServingBytes(&config));
    // With the ladder off the flag is a pin again.
    generate_mod.prefill_chunk_override = 4096;
    per_request_chunk_override = false;
    try std.testing.expectEqual(at(&config, 4096), expertStreamingServingBytes(&config));
    per_request_chunk_override = true;
    generate_mod.prefill_chunk_explicit = false;
    config.pinned_prefill_chunk = 1024;
    try std.testing.expectEqual(at(&config, 1024), expertStreamingServingBytes(&config));
    config.pinned_prefill_chunk = 0;
    per_request_chunk_override = false;
    try std.testing.expectEqual(at(&config, floor), expertStreamingServingBytes(&config));
}

test "a loaded streamed row reports its effective budget, an unloaded one the setting" {
    try std.testing.expectEqual(@as(u32, 60), rowSsdBudgetGb(60 << 30, 0));
    try std.testing.expectEqual(@as(u32, 48), rowSsdBudgetGb(0, 48));
    try std.testing.expectEqual(@as(u32, 60), rowSsdBudgetGb(60 << 30, 48));
    try std.testing.expectEqual(@as(u32, 0), rowSsdBudgetGb(0, 0));
}

test "the user-wire warning fires when the wired bytes pass the lower macOS limit" {
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    // M5 Max 128 GB: `vm.global_user_wire_limit` defaults to memsize less 20.62 GB.
    const ram: u64 = 137_438_953_472;
    const default_limit: u64 = 116_820_000_000;
    // A streamed DeepSeek load's owned bytes are its ssd budget less the 1 GiB MLX pool.
    const owned = struct {
        fn at(budget_gib: u64) u64 {
            return (budget_gib - 1) * (1 << 30);
        }
    }.at;
    var buf: [384]u8 = undefined;
    try t.expect(userWireWarning(&buf, owned(108), .{ default_limit, default_limit }, 125_000 << 20, ram) == null);
    for ([_]u64{ 111, 114 }) |b| {
        const line = userWireWarning(&buf, owned(b), .{ default_limit, default_limit }, 125_000 << 20, ram) orelse return error.TestExpectedWarning;
        try t.expect(std.mem.indexOf(u8, line, "vm.global_user_wire_limit is 116.8 GB") != null);
        try t.expect(std.mem.indexOf(u8, line, "sudo sysctl vm.global_user_wire_limit=125000000000 vm.user_wire_limit=125000000000") != null);
    }
    const line = userWireWarning(&buf, owned(111), .{ default_limit, default_limit }, 0, ram).?;
    try t.expect(std.mem.indexOf(u8, line, "wires ~119.2 GB") != null);
    try t.expect(std.mem.indexOf(u8, line, "limit=120000000000 ") != null);
    try t.expect(userWireWarning(&buf, owned(114), .{ 125_000_000_000, 125_000_000_000 }, 125_000 << 20, ram) == null);
    // The per-process limit binds when it is the lower one; unreadable limits never warn.
    try t.expect(std.mem.indexOf(u8, userWireWarning(&buf, owned(111), .{ 125_000_000_000, default_limit }, 0, ram).?, "vm.user_wire_limit is") != null);
    try t.expect(userWireWarning(&buf, owned(114), .{ 0, 0 }, 0, ram) == null);
    // A 96 GB Mac: the default limit is 15% under memsize; the suggestion stays 12 GB under RAM.
    const ram96: u64 = 103_079_215_104;
    const limit96: u64 = 87_600_000_000;
    try t.expect(userWireWarning(&buf, 80 * GiB, .{ limit96, limit96 }, 0, ram96) == null);
    const l96 = userWireWarning(&buf, 83 * GiB, .{ limit96, limit96 }, 0, ram96).?;
    try t.expect(std.mem.indexOf(u8, l96, "limit=91000000000 ") != null);
    // Past what the Mac may wire, raising the limit cannot help: the line says so.
    const none = userWireWarning(&buf, 95 * GiB, .{ limit96, limit96 }, 0, ram96).?;
    try t.expect(std.mem.indexOf(u8, none, "sudo") == null);
    try t.expect(std.mem.indexOf(u8, none, "--ssd-budget-gb") != null);
}

test "ssd budget parses GiB and admits against the wired limit" {
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    try t.expectEqual(100 * GiB, try parseSsdBudgetGb("100"));
    try t.expectError(error.InvalidSsdBudgetGb, parseSsdBudgetGb("0"));
    try t.expectError(error.InvalidSsdBudgetGb, parseSsdBudgetGb("513"));
    try t.expectError(error.InvalidSsdBudgetGb, parseSsdBudgetGb("100.5"));
    // The sysctl is the limit when it is set; otherwise Metal's recommended working set.
    try t.expectEqual(120 * GiB, ssdBudgetCeiling(120 * GiB, 96 * GiB));
    try t.expectEqual(96 * GiB, ssdBudgetCeiling(0, 96 * GiB));
    try t.expect(ssdBudgetAdmits(80 * GiB, 8 * GiB, 96 * GiB));
    try t.expect(!ssdBudgetAdmits(100 * GiB, 8 * GiB, 96 * GiB));
    try t.expectEqual(@as(u64, 110_592), wiredLimitMbForBudget(100 * GiB, 8 * GiB));
    try t.expectEqual(@as(u64, 1), wiredLimitMbForBudget(1, 0));
}

/// A stand-in for the deployed long-context pack: a GDN+MoE trunk with a QSA
/// indexer. Declares `qwen4_exp`: every long-context mechanism is gated on it.
fn longCtxTestConfig() model_mod.ModelConfig {
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "qwen4_exp"; // arch-scan-exempt: test fixture, not a gate
    cfg.num_hidden_layers = 40;
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 256;
    cfg.hidden_size = 2560;
    cfg.intermediate_size = 9216;
    cfg.intermediate_size_declared = true;
    cfg.quant_bits = 4;
    cfg.max_position_embeddings = 1048576;
    cfg.indexer_budget = 2048;
    cfg.indexer_n_heads = 4;
    cfg.indexer_head_dim = 128;
    cfg.indexer_compress_ratio = 4;
    return cfg;
}

test "planHotCache: the budget is MONOTONE in the ask" {
    // The defect: a bigger ask stepped the chunk down, shrank the reserve and bought MORE cache
    // (10GB -> 3873 MB, 24GB -> 5703 MB, same box).
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const cfg = longCtxTestConfig();
    const kv_bits: u64 = 8;
    const ctx_tokens: u64 = 262144;
    const weights: u64 = 69 * GiB;
    const ceiling: u64 = weights + 28_909 * (1 << 20);

    var prev: u64 = 0;
    var ask: u64 = GiB;
    while (ask <= 64 * GiB) : (ask += GiB) {
        const plan = planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, 0, ask, 0);
        try t.expect(plan.budget >= prev);
        try t.expect(plan.budget <= @max((ceiling -| weights) -| plan.ctx_kv -| plan.reserve, 1));
        prev = plan.budget;
    }

    // The two live asks now agree: the sizer no longer sees the ask.
    const at10 = planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, 0, 10 * GiB, 0);
    const at24 = planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, 0, 24 * GiB, 0);
    try t.expect(at24.budget >= at10.budget);
    try t.expectEqual(at10.chunk, at24.chunk);
}

test "planHotCache: an out-of-the-box boot gets the whole remaining headroom, never the 1-byte floor" {
    // One boot with no flags: `requested == 0` used to hand the sizer a zero reserve, so it took
    // a wide rung and the clamp returned its 1-byte floor.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const cfg = longCtxTestConfig();
    const kv_bits: u64 = 8;
    const weights: u64 = 69 * GiB;
    const ceiling: u64 = weights + 28_909 * (1 << 20);
    const ctx_tokens: u64 = 262144;
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) + statePerTokenBilled(&cfg);
    const sizer_ctx_kv: u64 = per_tok * ctx_tokens + cfg.qsaRingBytes();

    const boot = planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, sizer_ctx_kv, 0, 0);
    // Everything left over: the cache is the residual claimant.
    try t.expectEqual(@max((ceiling -| weights) -| boot.ctx_kv -| boot.reserve, 1), boot.budget);
    // A cache worth having, or the ladder bottomed out and the box is too small.
    try t.expect(boot.budget >= HOT_CACHE_FLOOR_BYTES or
        boot.chunk == PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1]);

    // The rung is the widest that clears both ask-independent bars.
    const serving: u64 = ceiling - weights;
    const cap: u64 = @min(serving / PREFILL_RESERVE_BUDGET_SHARE, (serving -| sizer_ctx_kv) -| HOT_CACHE_FLOOR_BYTES);
    for (PREFILL_CHUNK_LADDER) |c| {
        if (c <= boot.chunk) break;
        try t.expect(prefillTransientReserve(&cfg, kv_bits, c) > cap);
    }

    // An explicit ask can only take a slice of the same headroom.
    const huge = planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, sizer_ctx_kv, 1024 * GiB, 0);
    try t.expectEqual(boot.budget, huge.budget);
    try t.expect(planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, sizer_ctx_kv, 4 * GiB, 0).budget <= boot.budget);
}

test "the clamp's live inversion, as arithmetic" {
    // The reported boot reduced to `clampedPrefixCacheMem`: ceiling - weights = 28,909 MiB, ctx KV 20,736 MiB.
    const t = std.testing;
    const MiB: u64 = 1 << 20;
    const GiB: u64 = 1 << 30;
    const serving: u64 = 28_909 * MiB;
    const ctx_kv: u64 = 20_736 * MiB;
    const reserve_1024: u64 = 4_300 * MiB;
    const reserve_512: u64 = 2_470 * MiB;

    const ask10 = clampedPrefixCacheMem(10 * GiB, serving, 0, ctx_kv, reserve_1024);
    const ask24 = clampedPrefixCacheMem(24 * GiB, serving, 0, ctx_kv, reserve_512);
    try t.expectEqual(3_873 * MiB, ask10);
    try t.expectEqual(5_703 * MiB, ask24);
    try t.expect(ask10 < ask24); // the inversion, in the numbers that shipped

    // One ask-independent reserve: the clamp is `@min(ask, headroom)` over a headroom no ask can move.
    var prev: u64 = 0;
    var ask: u64 = GiB;
    while (ask <= 64 * GiB) : (ask += GiB) {
        const got = clampedPrefixCacheMem(ask, serving, 0, ctx_kv, reserve_1024);
        try t.expect(got >= prev);
        try t.expect(got <= @max((serving -| ctx_kv) -| reserve_1024, 1));
        prev = got;
    }
    try t.expectEqual(3_873 * MiB, clampedPrefixCacheMem(0, serving, 0, ctx_kv, reserve_1024));
}

test "an explicit --prefill-chunk is the chunk that gets BILLED" {
    // `--prefill-chunk 4096` used to run a 4096-token forward against a 512-token reserve.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    var cfg = longCtxTestConfig();
    cfg.head_dim = 128; // fused dim: boundedPrefillChunk returns its base untouched
    const kv_bits: u64 = 8;
    const weights: u64 = 69 * GiB;
    // Squeezed so the machine sizer would step below the flag.
    const ceiling: u64 = weights + 6 * GiB;

    generate_mod.prefill_chunk_override = 4096;
    generate_mod.prefill_chunk_explicit = true;
    defer {
        generate_mod.prefill_chunk_override = 8192;
        generate_mod.prefill_chunk_explicit = false;
    }
    try t.expectEqual(@as(u32, 4096), explicitPrefillChunk());
    const sized = resolvePrefillChunk(&cfg, kv_bits, ceiling, weights, 0, 0);
    try t.expect(sized < 4096); // the machine would have narrowed it

    const billed = billedPrefillChunk(&cfg, kv_bits, ceiling, weights, 0, 0, explicitPrefillChunk());
    try t.expectEqual(@as(u32, 4096), billed);

    // The other reader agrees.
    try t.expectEqual(@as(usize, 4096), prefill_plan.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        40_000,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        billed,
    ));

    // With the per-request ladder off the flag is a pin, and the clamp reserves it; on the ladder it
    // only caps a width every request re-bills (`clampReserveWidth`).
    per_request_chunk_override = false;
    defer per_request_chunk_override = null;
    const plan = planHotCache(&cfg, kv_bits, ceiling, weights, 8192, 0, 2 * GiB, billed);
    try t.expectEqual(@as(u32, 4096), plan.chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, 4096), plan.reserve);
    try t.expect(plan.reserve > prefillTransientReserve(&cfg, kv_bits, sized));
}

/// The deployed long-context shape at the live per-token widths: 12 caching layers, hd 256,
/// 2 KV heads (13,056 B/tok at 8-bit); a 128-wide QSA indexer at ratio 4.
pub fn qsaScoreFusedOffGuard() struct {
    pub fn deinit(_: @This()) void {
        qwen4_qsa_select.qsa_score_fused_override = null;
    }
} {
    qwen4_qsa_select.qsa_score_fused_override = false;
    return .{};
}

/// The incidents these bills reproduce ran an engine that rebuilt the whole kv8 cache per layer.
fn qsaWholeRebuildGuard() struct {
    pub fn deinit(_: @This()) void {
        qwen4_qsa_attn.qsa_attn_kernel_override = null;
    }
} {
    qwen4_qsa_attn.qsa_attn_kernel_override = false;
    return .{};
}

fn qwen4RequestTestConfig() model_mod.ModelConfig {
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "qwen4_exp"; // arch-scan-exempt: test fixture, not a gate
    cfg.num_hidden_layers = 48;
    cfg.full_attention_interval = 4;
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 256;
    // No sliding-band layers: the field defaults to true, which routes a different chunk policy.
    cfg.has_sliding_window = false;
    cfg.hidden_size = 2560;
    cfg.moe_intermediate_size = 1024;
    cfg.num_experts_per_tok = 8;
    cfg.num_experts = 512;
    cfg.quant_bits = 4;
    cfg.max_position_embeddings = 1048576;
    cfg.indexer_budget = 2048;
    cfg.indexer_n_heads = 4;
    cfg.indexer_head_dim = 128;
    cfg.indexer_compress_ratio = 4;
    cfg.pinned_prefill_chunk = 1024; // what load-time sizing gives at 1M ctx
    return cfg;
}

/// The geometry the served pack actually ships (`Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`):
/// a narrower expert with a shared one beside it, and the GatedDeltaNet head counts
/// `qwen4RequestTestConfig` leaves at zero. Both chunk-scaled terms of the bill read them, so
/// the fixture's step is not the deployed one.
fn qwen4DeployedTestConfig() model_mod.ModelConfig {
    var cfg = qwen4RequestTestConfig();
    cfg.moe_intermediate_size = 640;
    cfg.num_experts_per_tok = 10;
    cfg.shared_expert_intermediate_size = 640;
    cfg.linear_num_key_heads = 16;
    cfg.linear_num_value_heads = 48;
    cfg.linear_key_head_dim = 128;
    cfg.linear_value_head_dim = 128;
    cfg.linear_conv_kernel_dim = 4;
    cfg.quant_group_size = 64;
    cfg.quant_mode = .affine;
    return cfg;
}

test "qwen4RequestTestConfig does not leak qsa_score_fused_override" {
    const saved = qwen4_qsa_select.qsa_score_fused_override;
    defer qwen4_qsa_select.qsa_score_fused_override = saved;
    qwen4_qsa_select.qsa_score_fused_override = null;
    _ = qwen4RequestTestConfig();
    try std.testing.expectEqual(@as(?bool, null), qwen4_qsa_select.qsa_score_fused_override);
}

/// The width a ladder rung actually forwards at for this prompt.
fn widthForRung(cfg: *const model_mod.ModelConfig, seq: u64, rung: u32) u32 {
    return @intCast(rungWidth(cfg, seq, rung, cfg.longCtxGated()));
}

/// The served Flash-Next geometry at `--qwen-gdn 16`, as the M1 Max runs measured it.
fn qwen4MeasuredTestConfig() model_mod.ModelConfig {
    var cfg = qwen4DeployedTestConfig();
    cfg.gdn_state_f32 = false;
    return cfg;
}

test "Flash-Next's request bill brackets the peaks measured on the M1 Max, with no padding beyond a small net" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const saved_ctx = server_config.max_context_size;
    defer server_config.max_context_size = saved_ctx;
    server_config.max_context_size = 100000;
    const saved_width = generate_mod.prefill_chunk_override;
    defer generate_mod.prefill_chunk_override = saved_width;
    generate_mod.prefill_chunk_override = 512;
    const cfg = qwen4MeasuredTestConfig();
    // Live `peak_bytes - active_bytes` after one prompt per boot (512 rung, kv8, no MTP, SSD tier on, 16 new
    // tokens), with each landed checkpoint already moved to the SSD.
    const measured = [_]struct { tokens: u64, peak_mib: u64 }{
        .{ .tokens = 2857, .peak_mib = 935 },
        .{ .tokens = 8505, .peak_mib = 984 },
        .{ .tokens = 16874, .peak_mib = 1196 },
        .{ .tokens = 33958, .peak_mib = 1381 },
        .{ .tokens = 59314, .peak_mib = 1976 },
    };
    for (measured) |m| {
        const bill = prefillNeededAtChunk(&cfg, m.tokens, 16, 8, 512, .{});
        const peak = m.peak_mib * (1 << 20);
        // Under-billing is a Metal OOM: at least a tenth over the peak. The upper bar keeps the net small.
        try std.testing.expect(bill >= peak + peak / 10);
        try std.testing.expect(bill <= peak + peak * 2 / 5);
    }
}

test "Flash-Next's streamed request bill brackets the streamed peaks measured on the M1 Max" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const saved_ctx = server_config.max_context_size;
    defer server_config.max_context_size = saved_ctx;
    server_config.max_context_size = 49152;
    var cfg = qwen4MeasuredTestConfig();
    cfg.expert_streaming = true;
    cfg.expert_fill_peak_bytes = scheduler_mod.billedFillPeakBytes(&cfg, 1_848_000_000);
    // Live `peak_bytes - active_bytes` after one cold prompt per width, `--ssd-budget-gb 23`, kv8, no MTP, SSD tier on.
    const measured = [_]struct { tokens: u64, chunk: u64, peak_mib: u64 }{
        .{ .tokens = 7987, .chunk = 1024, .peak_mib = 1224 },
        .{ .tokens = 45106, .chunk = 512, .peak_mib = 1683 },
    };
    for (measured) |m| {
        const bill = prefillNeededAtChunk(&cfg, m.tokens, 16, 8, m.chunk, .{});
        const peak = m.peak_mib << 20;
        try std.testing.expect(bill >= peak + peak / 10);
        try std.testing.expect(bill <= peak + peak * 2 / 5);
    }
}


test "the advertised Flash-Next context is the largest one the request bill admits" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const saved_ctx = server_config.max_context_size;
    defer server_config.max_context_size = saved_ctx;
    server_config.max_context_size = 0;
    const saved_disk = prefix_cache_disk_bytes;
    defer prefix_cache_disk_bytes = saved_disk;
    prefix_cache_disk_bytes = 20 << 30;
    const saved_mtp = configured_mtp;
    defer configured_mtp = saved_mtp;
    configured_mtp = false;
    var cfg = qwen4MeasuredTestConfig();
    cfg.pinned_prefill_chunk = 512;
    const MiB: u64 = 1 << 20;
    const ceiling: u64 = 28976 * MiB; // 30000 MiB wired limit less the 1 GiB margin
    const active: u64 = 27_378_205_274; // the measured lean pack's resident bytes
    // The SSD writer's host bytes sit outside the wired limit: the 1 GiB margin under it is the only reserve.
    const budget = ceiling - active;
    const raw = memoryContextAt(&cfg, ceiling, active);
    const fits = struct {
        fn at(c: *const model_mod.ModelConfig, ctx: u64, avail: u64) bool {
            var sized = c.*;
            sized.pinned_context = @intCast(ctx);
            return prefillNeededAtChunk(&sized, ctx, 0, 8, 512, .{}) <= avail;
        }
    };
    // What admission accepts is what the sizer advertises, to within the search step.
    try std.testing.expect(raw > 0);
    try std.testing.expect(fits.at(&cfg, raw, budget));
    try std.testing.expect(!fits.at(&cfg, @as(u64, raw) + 1024, budget));
}


test "a 32 GB Mac's lean Flash-Next budget advertises and admits an 80k prompt" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const saved_ctx = server_config.max_context_size;
    defer server_config.max_context_size = saved_ctx;
    server_config.max_context_size = 0;
    const saved_disk = prefix_cache_disk_bytes;
    defer prefix_cache_disk_bytes = saved_disk;
    prefix_cache_disk_bytes = 20 << 30;
    const saved_mtp = configured_mtp;
    defer configured_mtp = saved_mtp;
    configured_mtp = false;
    var cfg = qwen4MeasuredTestConfig();
    cfg.pinned_prefill_chunk = 512;
    const ceiling: u64 = 28976 << 20; // 30000 MiB wired limit less the 1 GiB margin
    const active: u64 = 27_378_205_274; // the measured lean pack's resident bytes
    const prompt: u64 = 80_000;
    const new_tokens: u32 = 16;
    // What the boot advertises: within the measured range the bill is the one reserve, no margin on top.
    var sized = cfg;
    sized.pinned_context = ssdFirstSessionTokensNow(&cfg, 8, ceiling, active, 512);
    try std.testing.expect(sized.pinned_context >= prompt + new_tokens);
    const budget = admissionAvailable(ceiling, active);
    try std.testing.expect(prefillNeededAtChunk(&sized, prompt, new_tokens, 8, 512, .{}) <= budget);
}

test "the load-time budget is reproducible: free RAM at load does not move it" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Two boots 11 minutes apart resolved the same 10 GB ask to 1076 and 9757 MB off the live ceiling.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const GB: u64 = 1000 * 1000 * 1000;
    const weights: u64 = 70 * GB;
    const ctx_tokens: u64 = 1_048_576; // the deployed --ctx-size
    const static_ceiling: u64 = 120_000 * 1024 * 1024; // the wired limit, both boots
    // `requested == 0` is the arm where the clamp alone decides: the budget IS the headroom.
    const uncapped: u64 = 0;

    const busy = physicalMemoryCeiling(static_ceiling, weights, 108_700 * 1000 * 1000 -| weights);
    const quiet = physicalMemoryCeiling(static_ceiling, weights, 116_600 * 1000 * 1000 -| weights);
    try t.expect(busy < quiet);

    const on_busy = planHotCache(&cfg, kv_bits, busy, weights, ctx_tokens, 0, uncapped, 0);
    const on_quiet = planHotCache(&cfg, kv_bits, quiet, weights, ctx_tokens, 0, uncapped, 0);
    try t.expect(on_busy.budget != on_quiet.budget);
    try t.expect(on_quiet.budget > on_busy.budget);

    // Billed against the static term, both boots agree.
    const a = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, uncapped, 0);
    const b = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, uncapped, 0);
    try t.expectEqual(a.budget, b.budget);
    try t.expect(a.budget >= on_quiet.budget);

    const capped_a = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, 24 * GB, 0);
    const capped_b = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, 24 * GB, 0);
    try t.expectEqual(capped_a.budget, capped_b.budget);
    try t.expect(capped_a.budget <= a.budget);

    try t.expectEqual(static_ceiling, physicalMemoryCeiling(static_ceiling, weights, 400 * GB));
}

test "an auto boot sizes the SAME context whatever the cache ask (live check #6)" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `--prefix-cache-mem 60GB` collapsed the advertised context to 870 tokens (ctx KV billed at 26 MB).
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB; // backed out of the live clamp line
    const live_ceiling: u64 = 109_395 * MiB;
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);
    const transient: u64 = prefillTransientReserve(&cfg, kv_bits, 4096);
    const cap: u32 = cfg.contextCap();

    // The sizing reserve is a constant, so the answer cannot move with the ask.
    const ctx_default = resolvedContextForLoad(0, 0, live_ceiling, active, CTX_SIZING_CACHE_RESERVE, transient, per_tok, cap);
    const ctx_big_ask = resolvedContextForLoad(0, 0, live_ceiling, active, CTX_SIZING_CACHE_RESERVE, transient, per_tok, cap);
    try t.expectEqual(ctx_default, ctx_big_ask);
    try t.expect(ctx_default > 900_000); // a real context, not the floor

    // The defect, both spellings: sizing against the granted budget...
    const vs_granted = resolvedContextForLoad(0, 0, live_ceiling, active, 48_673 * MiB, transient, per_tok, cap);
    try t.expect(vs_granted <= 1024);
    // ...and against the raw ask, which saturates usable to zero.
    const vs_raw_ask = resolvedContextForLoad(0, 0, live_ceiling, active, 60 * 1024 * MiB, transient, per_tok, cap);
    try t.expect(vs_raw_ask <= 1024);
    try t.expect(ctx_default > vs_granted * 500);
}

test "the clamp bills the context that will be SERVED, not the placeholder" {
    // With a placeholder context the clamp hands back nearly the whole ask.
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    var cfg = qwen4RequestTestConfig();
    cfg.mtp_override = false; // the MTP head's own term: `the load-time bill prices the MTP head ...`
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const static_ceiling: u64 = 120_000 * MiB;
    const ask: u64 = 60 * 1024 * MiB;

    const placeholder = planHotCache(&cfg, kv_bits, static_ceiling, active, 1024, 0, ask, 4096);
    const real = planHotCache(&cfg, kv_bits, static_ceiling, active, 1_048_576, 0, ask, 4096);

    try t.expect(placeholder.budget > 45_000 * MiB); // reproduces the live 48,673
    // 13,056 KV + 5,376 state (one history copy + the f32 score bank).
    try t.expectEqual(@as(u64, 1_048_576) * (13_056 + 2_304) + cfg.qsaRingBytes(), real.ctx_kv);

    // Both budgets are their headroom, so the whole difference is the session each billed.
    try t.expectEqual(placeholder.budget - real.budget, real.ctx_kv - placeholder.ctx_kv);
    try t.expect(real.budget < placeholder.budget);
    try t.expect(placeholder.ctx_kv < 30 * MiB);
    try t.expect(real.ctx_kv > 15_000 * MiB);
    try t.expect(real.budget > 0);
}

test "the SSD-first budget bills the FLOOR reserve, at the deployed pack's live numbers" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `ssdFirstBudgetForLoad` short-circuits above `planHotCache`, so the floor-width reserve never ran on qwen4_exp.
    const t = std.testing;
    var cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const ceiling: u64 = 109_395 * MiB;

    const orig_ctx = server_config.max_context_size;
    defer server_config.max_context_size = orig_ctx;
    server_config.max_context_size = 0;
    cfg.pinned_context = 0;
    per_request_chunk_override = true;
    defer per_request_chunk_override = null;
    const orig_explicit = generate_mod.prefill_chunk_explicit;
    defer generate_mod.prefill_chunk_explicit = orig_explicit;
    generate_mod.prefill_chunk_explicit = false;

    // The pure chunk resolver: `pinPrefillChunk` reads live memory.
    const chunk: u32 = @intCast(prefill_plan.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        0,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        cfg.pinned_prefill_chunk,
    ));
    try t.expect(perRequestPrefillChunkEnabled(&cfg));
    const floor_w = clampReserveWidth(&cfg, chunk);
    try t.expectEqual(PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1], floor_w);
    try t.expect(floor_w < chunk);

    const reserve_pinned = prefillTransientReserve(&cfg, kv_bits, chunk);
    const reserve_floor = prefillTransientReserve(&cfg, kv_bits, floor_w);
    try t.expect(reserve_floor < reserve_pinned);

    const ctx_kv: u64 = (kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg)) *|
        ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk);
    const ask: u64 = 10 * 1024 * MiB;

    // The idle allowance is the residual after the reserve, so the whole difference lands there.
    const now = ssdFirstPrefixCacheMem(ask, ceiling, active, ctx_kv, reserve_floor);
    const before = ssdFirstPrefixCacheMem(ask, ceiling, active, ctx_kv, reserve_pinned);
    try t.expect(now >= before);

    // The RAM arm reserves the same floor for the same pack.
    const plan = planHotCache(&cfg, kv_bits, ceiling, active, 1_048_576, sizerCtxKvBytes(&cfg, kv_bits), ask, chunk);
    try t.expectEqual(floor_w, plan.reserve_chunk);
    try t.expectEqual(reserve_floor, plan.reserve);

    // The session the floor holds is the one the server advertises.
    const advertised = autoContextFrom(memoryContextAt(&cfg, ceiling, active), cfg.contextCap());
    try t.expectEqual(advertised, ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk));
}

test "SSD-first is gated on a DISK TIER: with --prefix-cache-disk off, qwen4_exp takes the RAM arm" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `ssdFirstEnabled()` never checked that a tier exists, and `--prefix-cache-disk` is off by default.
    const t = std.testing;
    var cfg = qwen4RequestTestConfig();

    const orig_disk = prefix_cache_disk_bytes;
    defer prefix_cache_disk_bytes = orig_disk;
    // `prefixCacheMemForLoad` publishes into a process global; put it back or a later test reads it.
    defer clearResolvedPrefixCacheMem();
    const orig_over = prefix_cache_mod.ssd_first_override;
    defer prefix_cache_mod.ssd_first_override = orig_over;
    prefix_cache_mod.ssd_first_override = true; // the env switch is not what is on trial
    // The budget is a property of the machine; the test's machine is a 128 GB one, not the runner.
    const orig_ceiling = static_ceiling_override;
    defer static_ceiling_override = orig_ceiling;
    static_ceiling_override = 109_395 * (1 << 20);

    try t.expect(!prefix_cache_mod.ssdFirstActive(&cfg, false, true));
    try t.expect(prefix_cache_mod.ssdFirstActive(&cfg, true, true));
    var dense = model_mod.ModelConfig{};
    dense.model_type = "not_a_served_arch";
    try t.expect(!prefix_cache_mod.ssdFirstActive(&dense, true, true));
    try t.expect(prefix_cache_mod.ssdFirstActive(&dense, true, false));
    prefix_cache_mod.ssd_first_override = false;
    try t.expect(!prefix_cache_mod.ssdFirstActive(&cfg, true, true));
    prefix_cache_mod.ssd_first_override = true;

    // The SSD arm floors at one session's KV, the RAM arm is a residual bounded by the ask.
    const orig_ctx = server_config.max_context_size;
    defer server_config.max_context_size = orig_ctx;
    server_config.max_context_size = 1_048_576; // explicit: no auto-sizing in the way
    cfg.pinned_context = 0;
    const ask: u64 = 4 * 1024 * 1024 * 1024;

    prefix_cache_disk_bytes = 0;
    var idle_off: u64 = 12345;
    const ram_arm = prefixCacheMemForLoad(&cfg, ask, .{}, &idle_off);
    try t.expectEqual(@as(u64, 0), idle_off);
    try t.expect(ram_arm <= ask);

    prefix_cache_disk_bytes = 64 * 1024 * 1024 * 1024;
    var idle_on: u64 = 0;
    const ssd_arm = prefixCacheMemForLoad(&cfg, ask, .{}, &idle_on);
    try t.expect(ssd_arm > ask);
    try t.expect(ssd_arm > ram_arm);
    try t.expect(idle_on > 0 and idle_on <= ask);
    try t.expect(ssd_arm -| idle_on > ask); // the one-session floor
}

test "an auto boot advertises the session the SSD-first budget floor was billed for" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The load-time bill and the pinned context must be one session: the SSD arm billed with
    // `cache_reserve = 0` while `pinAutoContext` sized against `CTX_SIZING_CACHE_RESERVE`, so
    // the floor held RAM for a larger session than the one advertised.
    const t = std.testing;
    var cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const ceiling: u64 = 109_395 * MiB;

    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 0;
    cfg.pinned_context = 0;

    const chunk: u32 = @intCast(prefill_plan.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        0,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        cfg.pinned_prefill_chunk,
    ));
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);
    const transient: u64 = prefillTransientReserve(&cfg, kv_bits, chunk);

    const advertised = autoContextFrom(memoryContextAt(&cfg, ceiling, active), cfg.contextCap());
    const billed = ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk);
    try t.expectEqual(advertised, billed);

    // And on a tighter box where the checkpoint cap does not bind.
    const tight: u64 = active + 24_000 * MiB;
    const billed_t = ssdFirstSessionTokensNow(&cfg, kv_bits, tight, active, chunk);
    try t.expectEqual(autoContextFrom(memoryContextAt(&cfg, tight, active), cfg.contextCap()), billed_t);
    try t.expect(billed > 500_000);
    try t.expect(billed_t > 300_000);
    try t.expect(billed_t < billed);

    // The floor holds RAM for exactly the advertised session.
    const idle: u64 = 4 * 1024 * MiB;
    const floor_t = ssdFirstPrefixCacheMem(idle, tight, active, per_tok *| @as(u64, billed_t), transient);
    const floor_more = ssdFirstPrefixCacheMem(idle, tight, active, per_tok *| @as(u64, billed_t + 1024), transient);
    try t.expectEqual(per_tok * 1024, floor_more - floor_t);
}

test "the load-time session bill is billed at the boot's --kv-quant, not bf16" {
    // Live: the SSD-first budget billed the session at 22,464 MB with `--kv-quant 8` and with
    // `off` alike (dense 29,952 B/tok): `defaultKvBits` asked `global_scheduler`, which `serve`
    // assigns only after `Scheduler.init` performs the load.
    const t = std.testing;
    var cfg = qwen4RequestTestConfig();
    cfg.mtp_override = false; // the MTP head's own term: `the load-time bill prices the MTP head ...`
    const MiB: u64 = 1 << 20;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    try t.expectEqual(@as(u64, 2_304), statePerTokenBilled(&cfg));
    try t.expectEqual(@as(u64, 13_056), kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8));
    try t.expectEqual(@as(u64, 24_576), kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 16));

    // An explicit `--ctx-size`, so the two numbers differ in the width and nothing else.
    const saved_ctx = server_config.max_context_size;
    defer server_config.max_context_size = saved_ctx;
    server_config.max_context_size = 786_432;
    const ceiling: u64 = 109_395 * MiB;
    const active: u64 = 69_827 * MiB;
    const chunk: u32 = cfg.pinned_prefill_chunk;

    const saved_kv = configured_kv_quant;
    defer configured_kv_quant = saved_kv;

    // `--kv-quant 8`, no scheduler yet: the state the load runs in.
    configured_kv_quant = transformer_mod.KVQuantConfig.affine(8);
    try t.expectEqual(@as(u64, 8), defaultKvBits(&cfg));
    try t.expectEqual(
        (13_056 + 2_304) * 786_432 + cfg.qsaRingBytes(),
        ssdFirstSessionKvBytes(&cfg, defaultKvBits(&cfg), ceiling, active, chunk),
    );

    configured_kv_quant = transformer_mod.KVQuantConfig.dense;
    try t.expectEqual(@as(u64, 16), defaultKvBits(&cfg));
    try t.expectEqual(
        (24_576 + 2_304) * 786_432 + cfg.qsaRingBytes(),
        ssdFirstSessionKvBytes(&cfg, defaultKvBits(&cfg), ceiling, active, chunk),
    );
    configured_kv_quant = null;
    try t.expectEqual(@as(u64, 8), defaultKvBits(&cfg));

    const idle: u64 = 10 * 1024 * MiB;
    const transient: u64 = prefillTransientReserve(&cfg, 8, chunk);
    const at8 = ssdFirstPrefixCacheMem(idle, ceiling, active, 13_824 * MiB, transient);
    const at16 = ssdFirstPrefixCacheMem(idle, ceiling, active, 22_464 * MiB, transient);
    try t.expectEqual(@as(u64, 8_640 * MiB), at16 -| at8);
}

test "manualContext: an explicit --ctx-size outranks the model's ctx_size, which outranks auto" {
    const t = std.testing;
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    var cfg = model_mod.ModelConfig{};
    cfg.ctx_override = 4096;
    server_config.max_context_size = 16384;
    try t.expectEqual(@as(u32, 16384), manualContext(&cfg));
    server_config.max_context_size = 0;
    try t.expectEqual(@as(u32, 4096), manualContext(&cfg));
    cfg.ctx_override = 0;
    try t.expectEqual(@as(u32, 0), manualContext(&cfg));
}

test "with no --kv-quant, the per-token KV bill and the auto-context follow the kv8 engine default" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    var cfg = qwen4RequestTestConfig();
    const saved_kv = configured_kv_quant;
    defer configured_kv_quant = saved_kv;

    // One fixed memory reading for every arm: live free RAM moves between two reads.
    const ceiling: u64 = 96 << 30;
    const active: u64 = 90 << 30; // tight enough that the checkpoint's own cap does not bind
    configured_kv_quant = null;
    try t.expectEqual(@as(u64, 8), defaultKvBits(&cfg));
    try t.expectEqual(sessionBytesPerToken(&cfg, 8), sessionBytesPerToken(&cfg, defaultKvBits(&cfg)));
    const ctx_default = memoryContextAt(&cfg, ceiling, active);

    configured_kv_quant = transformer_mod.KVQuantConfig.affine(8);
    try t.expectEqual(ctx_default, memoryContextAt(&cfg, ceiling, active));
    configured_kv_quant = transformer_mod.KVQuantConfig.dense;
    try t.expect(memoryContextAt(&cfg, ceiling, active) < ctx_default);

    // A per-model setting still outranks the default.
    configured_kv_quant = null;
    cfg.kv_quant_override = transformer_mod.KVQuantConfig.dense;
    try t.expectEqual(@as(u64, 16), defaultKvBits(&cfg));
}

test "an explicit --ctx-size boot never consults the session reserve" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `--ctx-size` wins in the resolver's first branch; asserted as invariance in the reserve.
    const t = std.testing;
    var cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);

    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;

    server_config.max_context_size = 262_144;
    cfg.pinned_context = 0;
    for ([_]u64{ 0, 40_000 * MiB, 109_395 * MiB, 900_000 * MiB }) |ceiling| {
        for ([_]u64{ 0, 69_827 * MiB }) |active| {
            for ([_]u32{ 512, 1024, 4096, 8192 }) |chunk| {
                try t.expectEqual(@as(u32, 262_144), ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk));
            }
        }
    }
    for ([_]u64{ 0, CTX_SIZING_CACHE_RESERVE, 99_000 * MiB }) |reserve| {
        try t.expectEqual(
            @as(u32, 262_144),
            resolvedContextForLoad(262_144, 0, 109_395 * MiB, 69_827 * MiB, reserve, 3 * 1024 * MiB, per_tok, cfg.contextCap()),
        );
    }

    server_config.max_context_size = 0;
    cfg.pinned_context = 131_072;
    try t.expectEqual(@as(u32, 131_072), ssdFirstSessionTokensNow(&cfg, kv_bits, 109_395 * MiB, 69_827 * MiB, 4096));
}

test "an explicit --ctx-size keeps the load-time context byte-identical" {
    const t = std.testing;
    const MiB: u64 = 1 << 20;
    const per_tok: u64 = 20_736;
    try t.expectEqual(@as(u32, 1_048_576), resolvedContextForLoad(1_048_576, 0, 0, 0, 99_000 * MiB, 99_000 * MiB, per_tok, 262_144));
    try t.expectEqual(@as(u32, 262_144), resolvedContextForLoad(0, 262_144, 0, 0, 99_000 * MiB, 0, per_tok, 1_048_576));
}

test "clampReserveWidth: the load-time reserve is a promise to the FIRST request" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The ceiling inversion: same ask, `iogpu.wired_limit_mb 120000` got a 3.6x smaller cache
    // because the wider rung's reserve grew faster than the headroom. Billing the ladder floor
    // makes the budget monotone in the ceiling.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const GiB: u64 = 1 << 30;
    const weights: u64 = 70 * GiB;
    const ctx_tokens: u64 = 1_048_576;
    const floor_rung = PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];

    const low = planHotCache(&cfg, kv_bits, weights + 28_909 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    const high = planHotCache(&cfg, kv_bits, weights + 43_000 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    try t.expectEqual(floor_rung, low.reserve_chunk);
    try t.expectEqual(floor_rung, high.reserve_chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, floor_rung), low.reserve);
    // A bigger ceiling never buys a smaller cache.
    try t.expect(high.budget >= low.budget);
    try t.expect(low.chunk >= low.reserve_chunk);

    var prev: u64 = 0;
    var mb: u64 = 24_000;
    while (mb <= 60_000) : (mb += 1_000) {
        const p = planHotCache(&cfg, kv_bits, weights + mb * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
        try t.expect(p.budget >= prev);
        prev = p.budget;
    }

    // An explicit --prefill-chunk caps the ladder, so the clamp still reserves the floor.
    generate_mod.prefill_chunk_override = 4096;
    generate_mod.prefill_chunk_explicit = true;
    const capped = planHotCache(&cfg, kv_bits, weights + 43_000 * MiB, weights, ctx_tokens, 0, 10 * GiB, explicitPrefillChunk());
    try t.expectEqual(@as(u32, 4096), capped.chunk);
    try t.expectEqual(floor_rung, capped.reserve_chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, floor_rung), capped.reserve);
    // With the ladder off it is a pin, billed as-is.
    per_request_chunk_override = false;
    const pinned = planHotCache(&cfg, kv_bits, weights + 43_000 * MiB, weights, ctx_tokens, 0, 10 * GiB, explicitPrefillChunk());
    per_request_chunk_override = null;
    generate_mod.prefill_chunk_override = 8192;
    generate_mod.prefill_chunk_explicit = false;
    try t.expectEqual(@as(u32, 4096), pinned.reserve_chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, 4096), pinned.reserve);

    // Gate off: the clamp reserves at the sizer's rung.
    per_request_chunk_override = false;
    defer per_request_chunk_override = null;
    const off = planHotCache(&cfg, kv_bits, weights + 28_909 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    try t.expectEqual(off.chunk, off.reserve_chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, off.chunk), off.reserve);
    per_request_chunk_override = null;

    var other = qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";
    const neutral = planHotCache(&other, kv_bits, weights + 28_909 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    try t.expectEqual(neutral.chunk, neutral.reserve_chunk);
}

test "the per-token widths the request chooser is built on" {
    // Pin the two constants every number in the story is derived from.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    try t.expectEqual(@as(u32, 12), cfg.attnCacheLayerCount());
    try t.expectEqual(@as(u64, 13_056), kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8));
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    try t.expectEqual(@as(u64, 768), cfg.qsaHistoryBytesPerToken());
    try t.expectEqual(@as(u64, 1_536), cfg.qsaScoreBankBytesPerToken());
    try t.expectEqual(@as(u64, 2_304), statePerTokenBilled(&cfg));
    try t.expectEqual(@as(u64, 15_360 * (1 << 20)), (13_056 + 2_304) * @as(u64, 1_048_576));
}

// Bar: the ring is billed once per slot, independent of prompt length.
test "a 512k QSA session drops the raw-key overbill and bills the ring once" {
    // These pins price a cache with no SSD tier behind it: every checkpoint stays in RAM.
    const no_tier = TierGuard.pin(false, 0);
    defer no_tier.restore();
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const cfg = qwen4ExpOomConfig();
    const seq: u64 = 512 * 1024;
    const short = prefillRequestTerms(&cfg, 1024, 2048, 8, 4096, .{});
    const long = prefillRequestTerms(&cfg, seq, 2048, 8, 4096, .{});
    try t.expectEqual(cfg.qsaRingBytes(), short.qsa_ring_bytes);
    try t.expectEqual(cfg.qsaRingBytes(), long.qsa_ring_bytes);
    const reserved = @max(reservedCacheTokens(seq, 2048, 4096, getEffectiveContextLength(&cfg)), seq);
    const new_needed = prefillNeededAtChunk(&cfg, seq, 2048, 8, 4096, .{});
    const old_needed = new_needed +| (reserved * 3_072 * 5 / 4);
    const ceiling: u64 = 26_000 * 1024 * 1024;
    try t.expect(old_needed > ceiling);
    try t.expect(new_needed <= ceiling);
}

test "chooseRequestPrefillChunk: an ordinary prompt buys the wide chunk a 1M session cannot" {
    // `--ctx-size 1048576` with no chunk or cache flags must still prefill ordinary prompts at
    // the wide chunk; load-time sizing reserved for the whole session (rung 1024 for the boot).
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const available: u64 = 28_909 * MiB;
    const pin = cfg.pinned_prefill_chunk;

    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, 4096, 2048, kv_bits, available, pin, 0, .{}));
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, available, pin, 0, .{}));
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, 384_000, 2048, kv_bits, available, pin, 0, .{}));
    try t.expect(4096 > pin);

    // The full 1M context fits at 4096 on this box under the one-copy history bill.
    const full = chooseRequestPrefillChunk(&cfg, 1_048_576, 2048, kv_bits, available, pin, 0, .{});
    try t.expectEqual(@as(u32, 4096), full);
    try t.expect(prefillNeededAtChunk(&cfg, 1_048_576, 2048, kv_bits, 2048, .{}) <= available);
    try t.expect(prefillNeededAtChunk(&cfg, 1_048_576, 2048, kv_bits, 4096, .{}) <= available);

    // A shorter prompt can afford at least as wide a forward.
    var prev: u32 = 0;
    var i: usize = 0;
    const seqs = [_]u64{ 1_048_576, 384_000, 300_000, 60_000, 4096 };
    while (i < seqs.len) : (i += 1) {
        const w = chooseRequestPrefillChunk(&cfg, seqs[i], 2048, kv_bits, available, pin, 0, .{});
        try t.expect(w >= prev);
        prev = w;
    }
}

test "chooseRequestPrefillChunk: WIDEST that fits, at the boundary" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // "Widest that fits": priced at the exact bill of a width that width comes out; one byte
    // under, a strictly narrower one (except at the floor). Walks the config's own ladder.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 300_000;
    const pin = cfg.pinned_prefill_chunk;
    const floor_width: u32 = widthForRung(&cfg, seq, PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1]);

    var prev_width: u32 = 0;
    var distinct: usize = 0;
    for (PREFILL_CHUNK_LADDER) |rung| {
        if (rung > ladderTop(&cfg, 0)) continue; // above the arch's start width
        const width = widthForRung(&cfg, seq, rung);
        if (width == prev_width) continue; // a rung this arch narrows onto a width already walked
        prev_width = width;
        distinct += 1;
        const bill = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, width, .{});
        // A rung the long-context gate widened is taken only past its margin.
        const needed = if (width > rungWidth(&cfg, seq, rung, false))
            bill * (100 + PREFILL_WIDE_RUNG_MARGIN_PCT) / 100
        else
            bill;
        try t.expectEqual(width, chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, needed, pin, 0, .{}));
        const under = chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, needed - 1, pin, 0, .{});
        if (width == floor_width) {
            try t.expectEqual(floor_width, under);
        } else {
            try t.expect(under < width);
        }
    }
    try t.expect(distinct >= 4);

    // Nothing fits: the ladder floor, never 0 and never the load-time pin.
    try t.expectEqual(floor_width, chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, 0, pin, 0, .{}));
}

test "chooseRequestPrefillChunk: the rung the gate widened clears its bill by the estimator's margin" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 300_000;
    const pin = cfg.pinned_prefill_chunk;

    const wide_bill = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 8192, .{});
    const narrow_bill = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 4096, .{});
    try t.expect(wide_bill > narrow_bill);
    // 8192 is a width only the gate offers: ungated, this rung forwards at 4096.
    try t.expectEqual(@as(u32, 8192), widthForRung(&cfg, seq, 8192));
    try t.expectEqual(@as(u64, 4096), rungWidth(&cfg, seq, 8192, false));

    // Only an explicit `--prefill-chunk 8192` offers the rung past the 4096 start width.
    const with_margin = wide_bill * (100 + PREFILL_WIDE_RUNG_MARGIN_PCT) / 100;
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, with_margin, pin, 0, .{}));
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, wide_bill, pin, 8192, .{}));
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, with_margin - 1, pin, 8192, .{}));
    try t.expectEqual(@as(u32, 8192), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, with_margin, pin, 8192, .{}));
    // The rungs the gate did not widen are still taken at their plain bill.
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, narrow_bill, pin, 8192, .{}));
}

test "prefillMemoryNeeded: the wider chunk bills every chunk-scaled term at the wider width" {
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 300_000;
    const ffn: u64 = prefillFfnWidth(&cfg);
    const stream: u64 = prefillStreamBytesPerToken(&cfg);
    const dq: u64 = prefillDequantWeightBytes(&cfg);
    const keys: u64 = cfg.prefillAttnKeys(seq);

    const narrow = prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, ffn, kv_bits, 4096, keys, stream, dq, .{});
    const wide = prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, ffn, kv_bits, 8192, keys, stream, dq, .{});

    const mlp_step: u64 = 8 * 4096 * @max(@as(u64, cfg.hidden_size), ffn) * 2;
    const envelope_step: u64 = @max(3 * mlp_step, mlp_step + 4096 * stream);
    try t.expect(wide >= narrow + envelope_step * 5 / 4);
    try t.expect(qsaMaskBytes(&cfg, 8192, seq) > qsaMaskBytes(&cfg, 4096, seq));
}

test "prefillNeededAtChunk: the deployed pack's 8192 step is at least the measured peak step" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = qwen4DeployedTestConfig();
    const kv_bits: u64 = 8;

    for ([_]u64{ 300_000, 326_351 }) |seq| {
        for ([_]bool{ false, true }) |mtp_on| {
            const narrow = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 4096, .{ .mtp_on = mtp_on });
            const wide = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 8192, .{ .mtp_on = mtp_on });
            // The widest active+cache step the 4096 -> 8192 trade measured on this pack was
            // +3.3 GB; the bill for the step must not sit under what the box pays for it.
            try t.expect(wide - narrow >= 3_300_000_000);
        }
    }
}

test "chooseRequestPrefillChunk: the explicit flag caps it, the gate outranks it" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const roomy: u64 = 200 * (@as(u64, 1) << 30);
    const pin = cfg.pinned_prefill_chunk;

    // An explicit `--prefill-chunk` sets the ladder's top: never wider however roomy, the floor when nothing fits.
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, roomy, pin, 0, .{}));
    try t.expectEqual(@as(u32, 8192), chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, roomy, pin, 8192, .{}));
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, roomy, pin, 4096, .{}));
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, roomy, pin, 2048, .{}));
    try t.expectEqual(widthForRung(&cfg, 300_000, 512), chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, 0, pin, 2048, .{}));

    // Kill switch: the load-time pin for every request.
    per_request_chunk_override = false;
    defer per_request_chunk_override = null;
    try t.expectEqual(pin, chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, roomy, pin, 0, .{}));
    per_request_chunk_override = null;

    // Another arch keeps the load-time pin exactly.
    var other = qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";
    try t.expect(!other.perRequestPrefillChunk());
    try t.expectEqual(pin, chooseRequestPrefillChunk(&other, 300_000, 2048, kv_bits, roomy, pin, 0, .{}));
    try t.expect(cfg.perRequestPrefillChunk());
}

test "an explicit --prefill-chunk is the widest width a request may run, not a pin" {
    const t = std.testing;
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    const saved_chunk = generate_mod.prefill_chunk_override;
    const saved_explicit = generate_mod.prefill_chunk_explicit;
    defer {
        generate_mod.prefill_chunk_override = saved_chunk;
        generate_mod.prefill_chunk_explicit = saved_explicit;
    }
    // The live shape: MiMo, `--prefill-chunk 4096`, a 139,717-token turn.
    const cfg = mimoV2FlashBillConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 139_717;
    const pin: u32 = 2048;
    generate_mod.prefill_chunk_override = 4096;
    generate_mod.prefill_chunk_explicit = true;
    const n = explicitPrefillChunk();
    const at_n = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 4096, .{});
    const at_2048 = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 2048, .{});
    const at_floor = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 512, .{});
    try t.expect(at_2048 < at_n);

    // N when it fits, however roomy: never wider.
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, at_n, pin, n, .{}));
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, 200 << 30, pin, n, .{}));
    // Memory that fits only 2048 runs 2048 instead of refusing.
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, at_n - 1, pin, n, .{}));
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, at_2048, pin, n, .{}));
    // Nothing fits: the floor, whose bill the guard then refuses.
    const floor_w = chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, at_floor - 1, pin, n, .{});
    try t.expectEqual(@as(u32, 512), floor_w);
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(.{ .needed = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, floor_w, .{}), .available = at_floor - 1 }));

    // The bill and the forward run the one width the pick names.
    const pick = requestPrefillPick(&cfg, seq, 2048, kv_bits, at_2048, false, .{});
    try t.expectEqual(@as(u32, 2048), pick);
    try t.expectEqual(@as(usize, pick), prefill_plan.requestPrefillChunk(cfg.prefillScoreHeadDim(), cfg.num_attention_heads, seq, cfg.has_sliding_window, cfg.isMoe(), cfg.longCtxGated(), pick));

    // A narrower N is the cap for every rung; one below the floor is the floor.
    generate_mod.prefill_chunk_override = 3000;
    try t.expectEqual(@as(u32, 3000), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, 200 << 30, pin, explicitPrefillChunk(), .{}));
    generate_mod.prefill_chunk_override = 256;
    try t.expectEqual(@as(u32, 256), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, 0, pin, explicitPrefillChunk(), .{}));

    // Per-request picking off: the explicit width is the pin again, on the bill and the forward.
    generate_mod.prefill_chunk_override = 4096;
    per_request_chunk_override = false;
    defer per_request_chunk_override = null;
    const off = requestPrefillPick(&cfg, seq, 2048, kv_bits, 0, false, .{});
    try t.expectEqual(@as(u32, 4096), off);
    try t.expectEqual(@as(usize, 4096), prefill_plan.requestPrefillChunk(cfg.prefillScoreHeadDim(), cfg.num_attention_heads, seq, cfg.has_sliding_window, cfg.isMoe(), cfg.longCtxGated(), off));
}

test "admission tries the ladder DOWN before refusing: a prompt that fits only at 512" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The probe used to price the load-time pin while the scheduler chose a per-request width,
    // so a prompt that fit only at the floor was refused for a forward that never ran.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 384_000;
    const pin = cfg.pinned_prefill_chunk;
    const floor_width = widthForRung(&cfg, seq, PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1]);
    const pin_width = widthForRung(&cfg, seq, pin);

    try t.expect(pin_width > floor_width);
    const at_floor = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, floor_width, .{});
    const at_pin = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, pin_width, .{});
    try t.expect(at_floor < at_pin);

    // Exactly enough for the floor width and not a byte more.
    const available = at_floor;

    try t.expectEqual(floor_width, chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, available, pin, 0, .{}));
    try t.expect(prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, floor_width, .{}) <= available);

    // Gate off: the load-time pin does not fit.
    per_request_chunk_override = false;
    defer per_request_chunk_override = null;
    try t.expectEqual(pin, chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, available, pin, 0, .{}));
    try t.expect(prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, pin_width, .{}) > available);
    per_request_chunk_override = null;

    // An explicit `--prefill-chunk` caps the ladder, it does not pin it: the same step down.
    generate_mod.prefill_chunk_override = 4096;
    generate_mod.prefill_chunk_explicit = true;
    defer {
        generate_mod.prefill_chunk_override = 8192;
        generate_mod.prefill_chunk_explicit = false;
    }
    const capped = chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, available, pin, explicitPrefillChunk(), .{});
    try t.expectEqual(floor_width, capped);
    try t.expect(prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, capped, .{}) <= available);
}

test "the post-eviction re-ask never exceeds what live memory affords, and never runs on an arch that has no per-request width" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The composition the scheduler runs: the chooser is asked before and after the eviction
    // pass, and the second reading (live memory) runs.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 383_000;
    const pin: u32 = 512;
    const floor_width = widthForRung(&cfg, seq, PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1]);

    // Pre-eviction: nothing on the ladder fits, the chooser hands back the floor.
    const admitted = chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, 0, pin, 0, .{});
    try t.expectEqual(floor_width, admitted);

    // Post-eviction, with the cache's bytes returned: a wider rung fits.
    const roomy: u64 = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, widthForRung(&cfg, seq, 4096), .{});
    const reasked = chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, roomy, pin, 0, .{});
    try t.expect(reasked > admitted);
    try t.expectEqual(reasked, scheduler_mod.postEvictionPrefillChunk(admitted, reasked).width);

    try t.expectEqual(admitted, scheduler_mod.postEvictionPrefillChunk(0, admitted).width);

    // The reverse reading (memory moved between the asks) runs at the floor.
    try t.expectEqual(admitted, scheduler_mod.postEvictionPrefillChunk(reasked, admitted).width);

    // A non-qwen4 arch is untouched end to end.
    var other = qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";
    const other_pre = chooseRequestPrefillChunk(&other, seq, 2048, kv_bits, 0, pin, 0, .{});
    const other_post = chooseRequestPrefillChunk(&other, seq, 2048, kv_bits, roomy, pin, 0, .{});
    try t.expectEqual(pin, other_pre);
    try t.expectEqual(pin, other_post);
    try t.expectEqual(pin, scheduler_mod.postEvictionPrefillChunk(other_pre, other_post).width);

    per_request_chunk_override = false;
    defer per_request_chunk_override = null;
    const off_pre = chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, 0, pin, 0, .{});
    const off_post = chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, roomy, pin, 0, .{});
    try t.expectEqual(pin, scheduler_mod.postEvictionPrefillChunk(off_pre, off_post).width);
}

test "admissionLogLevel: free on the first request after a load and at the edge, quiet in the middle" {
    const t = std.testing;
    const GiB: u64 = 1 << 30;

    // A fresh load is always free.
    try t.expectEqual(log.Level.info, admissionLogLevel(1 * GiB, 100 * GiB, true));

    // Inside 10% of what is available.
    try t.expectEqual(log.Level.info, admissionLogLevel(95 * GiB, 100 * GiB, false));
    try t.expectEqual(log.Level.info, admissionLogLevel(120 * GiB, 100 * GiB, false)); // already over
    // The boundary itself: exactly 0.9 is quiet.
    try t.expectEqual(log.Level.debug, admissionLogLevel(90 * GiB, 100 * GiB, false));
    try t.expectEqual(log.Level.info, admissionLogLevel(90 * GiB + 1, 100 * GiB, false));

    try t.expectEqual(log.Level.debug, admissionLogLevel(1 * GiB, 100 * GiB, false));

    try t.expectEqual(log.Level.info, admissionLogLevel(1 * GiB, 0, false));

    try t.expectEqual(log.Level.info, admissionLogLevel(std.math.maxInt(u64), 1, false));
    try t.expectEqual(log.Level.debug, admissionLogLevel(1, std.math.maxInt(u64), false));
}

test "admissionVerdict names the same three arms the guard acts on" {
    const t = std.testing;
    try t.expectEqual(AdmissionVerdict.admit, admissionVerdict(.{ .needed = 10, .available = 20 }));
    try t.expectEqual(AdmissionVerdict.admit, admissionVerdict(.{ .needed = 20, .available = 20 }));
    // Fits once the cache gives back what it can prove: `reclaimable`, never `evictable`.
    try t.expectEqual(AdmissionVerdict.evict, admissionVerdict(.{ .needed = 30, .available = 20, .reclaimable = 10 }));
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(.{ .needed = 30, .available = 20, .evictable = 1000, .reclaimable = 0 }));
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(.{ .needed = 100, .available = 20, .reclaimable = 10 }));

    try t.expectEqualStrings("admit", AdmissionVerdict.admit.name());
    try t.expectEqualStrings("evict", AdmissionVerdict.evict.name());
    try t.expectEqualStrings("refuse", AdmissionVerdict.refuse.name());
}

/// The largest context this model's per-token footprint fits into RAM right now, ignoring the checkpoint's own maximum.
pub fn computeMemoryContext(config: *const model_mod.ModelConfig) u32 {
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    return memoryContextAt(config, currentGpuMemoryCeiling(config, active_mem), active_mem);
}

/// `computeMemoryContext` at one reading of the ceiling and MLX's active memory.
fn memoryContextAt(config: *const model_mod.ModelConfig, ceiling: u64, active_mem: u64) u32 {
    // `total_ctx = 0` asks for the unshrunk cap: the widest forward any prompt can run.
    const chunk: u64 = @intCast(prefill_plan.effectivePrefillChunk(
        config.prefillScoreHeadDim(),
        config.num_attention_heads,
        0,
        config.has_sliding_window,
        config.isMoe(),
        config.longCtxGated(),
        config.pinned_prefill_chunk,
    ));
    return memoryContextAtChunk(config, defaultKvBits(config), ceiling, active_mem, chunk);
}

/// `memoryContextAt` for a given prefill width, before the checkpoint's position cap.
fn memoryContextAtChunk(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, chunk: u64) u32 {
    if (config.num_attention_heads == 0) return 16384;
    // Flash-Next advertises the context its own request bill admits at the floor rung, from the
    // headroom admission reads (the writers' host bytes included), so the two cannot disagree.
    if (config.longCtxGated()) {
        const budget = admissionAvailable(ceiling, active_mem) -| ctxCacheShare(ceiling, active_mem, ctxSizingCacheReserve(config));
        return maxSushiContextAtWidth(config, kv_bits, budget, null, .{ .mtp_on = mtpHeadDefaultOn(config) });
    }
    // DeepSeek's request bill grows with the prompt (the indexer's scores), so its context is the longest that bill admits.
    if (config.isDsv41()) return maxSushiContextAtWidth(config, kv_bits, admissionAvailable(ceiling, active_mem) -| ctxCacheShare(ceiling, active_mem, ctxSizingCacheReserve(config)), chunk, .{});
    //   KV cache: the arch's own caching-layer count and K/V widths, billed at the active kv-quant width.
    const per_tok: u64 = sessionBytesPerToken(config, kv_bits);
    return safeContextForBudget(
        // Real reachable ceiling, so auto-context shrinks under external memory pressure (#64).
        ceiling,
        active_mem,
        // The hot prefix cache fills to this cap over a session, and a prefill has to land on
        // top of whatever context we report. `CTX_SIZING_CACHE_RESERVE`, not the resolved
        // budget and not the ask: the ask starved the context, the resolved budget was a
        // one-step loop. Gated: off the gate this is the raw ask, so `/v1/models` is unchanged.
        ctxCacheShare(ceiling, active_mem, ctxSizingCacheReserve(config)) +| prefillTransientReserve(config, kv_bits, chunk),
        per_tok,
        // 0 = do not clamp to the checkpoint's max here; the caller applies it after the margin.
        0,
    );
}

/// The share of the room `reserve` may claim: the context is the primary claimant, so the cache never takes more than half.
fn ctxCacheShare(ceiling: u64, active_mem: u64, reserve: u64) u64 {
    return @min(reserve, admissionAvailable(ceiling, active_mem) / 2);
}

/// The cache reserve the context sizer bills. Gated: the ask-independent constant. Ungated:
/// the raw `--prefix-cache-mem`. Both load-time wrappers must pass the same value.
fn ctxSizingCacheReserve(config: *const model_mod.ModelConfig) u64 {
    // GLM's admission evicts its RAM tier and prices each request's checkpoints instead.
    if (config.isGlm5()) return 0;
    if (prefix_cache_capacity == 0 or !prefix_cache_ram_enabled) return 0;
    return if (config.longCtxGated()) CTX_SIZING_CACHE_RESERVE else legacyPrefixCacheAsk(config);
}

/// Pure memory model behind checkAttentionMemory. All quantities in bytes,
/// 25% safety margin included. `chunk` must be the SAME prefill chunk
/// generate.zig will pick (generate_mod.effectivePrefillChunk) so the
/// admission guard and the real prefill cannot drift. Terms:
///   - kv: persistent for the request; billed at the ACTIVE kv-quant width
///     (+0.5 bit/elem group scale/bias overhead when quantized), not a
///     hardcoded fp16.
///   - scores: the composed-SDPA scratch [heads, chunk, seq] — materialized
///     only for head_dims no fused kernel covers (transformer.
///     prefillHeadDimFused: <= 128 via MLX, 256 via sushi_attn_p256 — unfused
///     only for an exotic dim).
///     Bounded to ~one layer by the adaptive eval cadence
///     (transformer.prefillEvalCadence).
///   - dequant: dense-fp16 rebuild of the FULL quantized cache each layer
///     (KVCache.denseView under --kv-quant).
///   - mlp: QKV/MLP transient envelope; ~3 layers coexist between eval
///     points, and it is CHUNK-bounded, never seq-scaled — the old
///     8×seq×ffn envelope over-billed a 255K prompt by ~60 GB and rejected
///     requests that fit comfortably (PR #69).
///
/// `attn_keys` is how many keys one query actually reads (config.
/// prefillAttnKeys): `seq` for dense causal attention, a much smaller bound for
/// a sparse arch. It scales the SCORE term only — KV/dequant are storage, and
/// a sparse reader still stores every latent it might later attend to.
/// `kv_per_tok` is the whole-model DENSE (f16) KV bytes for one token —
/// `ModelConfig.kvBytesPerToken()`. It replaced a `layers` parameter because
/// neither the layer count nor the per-head width is uniform once an arch
/// interleaves linear-attention layers or stores keys wider than values.
///
/// `hdim` and `score_hdim` are two DIFFERENT widths and only coincide on a
/// symmetric arch. `hdim` is the STORED width the quantized-KV dequant
/// transient is read at; `score_hdim` is the width the score is contracted at
/// (`ModelConfig.prefillScoreHeadDim`), which is what decides whether a fused
/// kernel exists — MLA scores over 192 while storing values at 128, and billing
/// the score term against `head_dim` 128 declared it fused and billed ZERO for
/// a composed path that really does materialize [heads, chunk, seq].
/// KV bytes for ONE token at the width the cache actually STORES. `dense` is
/// `ModelConfig.kvBytesPerToken()` — the arch's own caching-layer count and its
/// own K and V widths, at fp16. A quantized cache stores `kv_bits` per element
/// plus the group scale/bias pair, which is the `+1` in `(2·bits + 1)/32`
/// (~0.5 bit/elem at group 64 with bf16 params).
///
/// Both the auto-context sizer and the prefill admission guard bill through
/// here. The sizer used to bill fp16 unconditionally, so a `--kv-quant 4`
/// server reported — and served — under a third of the context it can hold.
pub fn kvBytesPerTokenAtBits(dense: u64, kv_bits: u64) u64 {
    if (kv_bits >= 16) return dense;
    return dense * (2 * kv_bits + 1) / 32;
}

/// Context-INDEPENDENT bytes one live slot holds beside its per-token KV: the
/// qwen4 QSA raw-key ring (not kv-quantized) and a ringed sliding layer's
/// retained window plus every checkpoint a slot holds (`SLOT_RING_CHECKPOINTS`;
/// stored like any other cache row, so they follow `kv_bits`).
/// Every sizer that used to add `qsaRingBytes` adds this instead — a second
/// per-slot constant billed at only some of them is the under-bill class.
pub fn slotRingBytes(config: *const model_mod.ModelConfig, kv_bits: u64) u64 {
    if (config.dsv41) |c| return c.stateFixedBytes(config.dsparkOn());
    if (config.isGlm5()) return config.qsaRingBytes() + config.ssmCheckpointBytes();
    const swa = config.swaRingBytes() +| prefix_cache_mod.SLOT_RING_CHECKPOINTS *| config.swaRingCheckpointBytes();
    return config.qsaRingBytes() +| kvBytesPerTokenAtBits(swa, kv_bits) +| batchedDecodeRowsBytes(config);
}

/// What the other rows of a batched MiMo decode hold beside this slot's tick: each row
/// rebuilds one global layer dense below the packed arms' floor, where a solo tick's rebuild
/// was alone. Zero where decode does not batch.
pub fn batchedDecodeRowsBytes(config: *const model_mod.ModelConfig) u64 {
    if (!config.supportsBatchedMimoDecode()) return 0;
    const keys = transformer_mod.mimoGlobalDecodeRebuildMaxKeys();
    var widest: u64 = 0;
    var li: u32 = 0;
    while (li < config.num_hidden_layers) : (li += 1) {
        if (config.isKvPerTokenLayer(li)) widest = @max(widest, config.layerKvBytes(li));
    }
    return @as(u64, scheduler_mod.batchGroupCap(config) - 1) *| keys *| widest;
}

/// The chunk-independent floor every prefill pays: MLX runtime scratch, the
/// KV cache's proportional capacity growth (old + new buffer coexist across a
/// grow), and the graph live-set no other term models. MEASURED as the
/// intercept of peak-above-steady-state against the chunk. It
/// does NOT scale with the weights — a dense bf16 checkpoint pays it too — so
/// it is billed as what it measures as: a constant.
const PREFILL_RUNTIME_FLOOR_BYTES: u64 = 512 * 1024 * 1024;

/// Prompt length up to which Flash-Next's summed terms stand without a net: measured at least a tenth over
/// the live peak (engine-memory-admission.md#bill-net). Past it the bill keeps the quarter its terms were fitted under.
const MEASURED_NET_MAX_TOKENS: u64 = 86_016;

/// The net the bill adds over its summed terms, in sixteenths (16 = none, 20 = the quarter).
fn billUplift16ths(config: *const model_mod.ModelConfig, seq: u64) u64 {
    return if (config.longCtxGated() and seq <= MEASURED_NET_MAX_TOKENS) 16 else 20;
}

/// How many MoE layers' gather-sort expansions coexist between eval points.
/// Mirrors `Transformer.MOE_EVAL_EVERY_N_LAYERS` — the MoE prefill loop evals
/// every 4th layer, so 4 layers' worth is what the peak holds.
const MOE_PREFILL_COEXIST: u64 = transformer_mod.Transformer.PREFILL_EVAL_CADENCE_DEFAULT;

/// What a warm turn already holds. A RAM hot-cache restore hands the slot the entry's KV
/// buffers without allocating another copy, so `available` is already net of them and
/// billing them again refused a 786k warm turn whose own entry was resident. KV only: the
/// QSA history is privatised on restore. Credited only while the prefix fits the destination
/// capacity (a grow allocates a fresh buffer beside the protected entry).
pub const WarmPrefix = struct {
    /// Prompt rows the hot cache restored. 0 = cold.
    matched_tokens: u64 = 0,
    /// Tokens of capacity the restored buffers hold (`KVCache.residentCapacityTokens`). The
    /// connection thread passes 0 and defers the warm decision to the inference thread.
    capacity_tokens: u64 = 0,
    /// Did the restore check out its entry (`prefix_cache.checkoutEligible`)? A shared restore
    /// is copied whole by the first append, so crediting it is an under-bill.
    will_donate: bool = false,
    mtp_on: bool = false,
    /// KDA checkpoints a GLM request keeps for the prefix cache. Only the inference thread's
    /// admission pass bills them, and it keeps fewer rather than refuse.
    checkpoints: u32 = 0,
    /// The prefill moves each checkpoint to SSD as it is produced and frees the RAM copy, so at most
    /// `glm5_prefix.checkpoint_ram_held` states are in RAM at once.
    released: bool = false,
    /// The request takes a thought point mid-decode: one more checkpoint state, held to its commit.
    thought: bool = false,

    /// Rows of KV this request will not allocate: the rows the restored buffer already holds.
    /// Capacity is a ceiling on the credit, never a switch: a growing chain outruns its capacity
    /// on nearly every turn. The grow's own cost is `reserved - credited` plus
    /// `PrefillRequestTerms.grow_coexist_bytes`. `will_donate` gates the whole credit.
    pub fn creditedRows(self: WarmPrefix, reserved: u64) u64 {
        if (!self.will_donate) return 0;
        if (self.matched_tokens == 0) return 0;
        return @min(@min(self.matched_tokens, self.capacity_tokens), reserved);
    }

    /// Does this append grow the buffers at all? The per-layer grow fires on `offset + new_len
    /// > capacity`; a reservation above the prompt allocates nothing by itself.
    pub fn grows(self: WarmPrefix, seq: u64) bool {
        if (!self.will_donate or self.matched_tokens == 0) return false;
        return seq > self.capacity_tokens;
    }

    /// Does the reservation outgrow restored buffers that hold the whole prompt? Then no append
    /// in the prefill grows them; `KVCache.growToReservation` does, before the first chunk.
    pub fn decodeOutgrows(self: WarmPrefix, seq: u64, reserved: u64) bool {
        if (self.matched_tokens == 0) return false;
        return seq <= self.capacity_tokens and reserved > self.capacity_tokens;
    }
};

/// PURE: the most KV-caching layers that fall inside one window of `window` consecutive layers,
/// i.e. how many old buffers can be alive at once at that eval cadence.
fn attnLayersPerEvalWindow(config: *const model_mod.ModelConfig, window: u32) u32 {
    const n = config.num_hidden_layers;
    if (window == 0 or n == 0) return 0;
    if (window >= n) return config.attnCacheLayerCount();
    var best: u32 = 0;
    var start: u32 = 0;
    while (start + window <= n) : (start += 1) {
        var count: u32 = 0;
        var i: u32 = start;
        while (i < start + window) : (i += 1) {
            if (config.isKvPerTokenLayer(i)) count += 1;
        }
        if (count > best) best = count;
    }
    return @min(best, config.kvPerTokenLayerCount());
}

/// PURE: bytes of OLD KV buffer alive beside the new ones while a warm append grows the cache.
/// `growQuantBuf` drops our reference to the old buffer at once, but the graph node holds it
/// until it EVALUATES, so the coexistence window is the prefill loop's eval cadence rather than
/// the size of the cache. A forward narrower than `prefillEvalCadenceApplies` runs one eval for
/// the whole layer loop and therefore does pay the whole old cache.
fn growCoexistBytes(config: *const model_mod.ModelConfig, warm: WarmPrefix, seq: u64, reserved: u64, kv_per_tok: u64) u64 {
    if (warm.grows(seq)) return oldBuffersInEvalWindow(config, warm.capacity_tokens, seq -| warm.matched_tokens, kv_per_tok);
    if (warm.decodeOutgrows(seq, reserved)) {
        // The prefill grows each layer first (`KVCache.growToReservation`), one eval at a
        // time; a share's old rows stay the entry's.
        return if (warm.will_donate) oldBuffersInEvalWindow(config, warm.capacity_tokens, 1 << 20, kv_per_tok) else 0;
    }
    // An SSD restore installs each layer at exactly its restored rows, so the first append grows
    // it beside the restored buffer. A ringed arch's admission never sees the restore (it bills a
    // warm request cold), so with a disk tier it bills that coexistence at the prompt's length.
    if (diskTierOn(config) and config.swaRingTokens() > 0 and !config.longCtxGated())
        return oldBuffersInEvalWindow(config, seq, seq, kv_per_tok);
    return 0;
}

/// Bytes of `tokens`-row old buffers alive beside their grown copies: one eval window of the
/// caching layers when a `span`-wide forward runs the cadence, every caching layer otherwise.
fn oldBuffersInEvalWindow(config: *const model_mod.ModelConfig, tokens: u64, span: u64, kv_per_tok: u64) u64 {
    // The layers `kv_per_tok` is the sum over, not every caching layer: on a
    // ringed arch those are nine of forty-eight.
    const attn = config.kvPerTokenLayerCount();
    if (attn == 0) return 0;
    const window: u32 = if (transformer_mod.Transformer.prefillEvalCadenceApplies(@intCast(@min(span, 1 << 20))))
        attnLayersPerEvalWindow(config, transformer_mod.Transformer.MOE_EVAL_EVERY_N_LAYERS)
    else
        attn;
    if (window == 0) return 0;
    // The old buffer's own size, spread over the layers that cache.
    return tokens *| kv_per_tok / attn *| window;
}

/// Per-request bytes that scale with the prompt rather than the chunk.
pub const PrefillRequestTerms = struct {
    /// Cache capacity reserved at admission beyond `seq * kv_per_tok`: the generation headroom
    /// plus a chunk of slack, so no buffer grows during the prefill (#353).
    reserved_kv_bytes: u64 = 0,
    /// Per-request state the arch keeps outside the KV cache (the qwen4_exp QSA indexer history), at its reserved size.
    state_bytes: u64 = 0,
    /// SSM checkpoints the prefill is still holding when it reaches its last chunk.
    checkpoint_bytes: u64 = 0,
    /// Bytes of the KV terms above already resident in the buffer the restore handed this slot
    /// (`WarmPrefix`). KV only; subtracted once, in `prefillMemoryNeeded`.
    shared_resident_bytes: u64 = 0,
    /// Old KV buffers that coexist with the new ones while a warm append, or the grow to the
    /// reservation before it, GROWS the cache. Zero when nothing grows, and on every arch that
    /// does not reserve its capacity.
    grow_coexist_bytes: u64 = 0,
    qsa_ring_bytes: u64 = 0,
    mtp_head_kv_bytes: u64 = 0,
    /// The DFlash2 assistant's window and capture rows (`dflashRequestBytes`).
    dflash_bytes: u64 = 0,
    /// The bill's net over the summed terms, in sixteenths (20 = the historical quarter).
    uplift_16ths: u64 = 20,
    /// The dense rebuild ONE quantized-KV layer really makes at this prompt
    /// length (`kvDequantScratchBytes`). Zero = the generic
    /// `2 · seq · kv_heads · hdim · 2` below, which is exact for every arch
    /// whose caching layers share one head count and one width.
    dequant_scratch_bytes: u64 = 0,
};

pub fn prefillMemoryNeeded(seq: u64, heads: u64, kv_heads: u64, kv_per_tok: u64, hdim: u64, score_hdim: u64, hidden: u64, ffn: u64, kv_bits: u64, chunk: u64, attn_keys: u64, stream_per_tok: u64, dequant_weights: u64, req: PrefillRequestTerms) u64 {
    // A forward is never wider than the prompt: a prompt shorter than the
    // chunk runs ONE seq-wide forward, so the per-chunk transient envelope
    // scales with min(chunk, seq) — billing the raw chunk cap rejected a
    // 31-token prompt for "needing" 6.2 GB on a memory-squeezed host
    // (live 2026-07-14). No-op for prompts >= one chunk.
    const fwd: u64 = @min(chunk, @max(seq, 1));
    const kv_bytes: u64 = seq * kvBytesPerTokenAtBits(kv_per_tok, kv_bits);
    const scores: u64 = if (!transformer_mod.prefillHeadDimFused(@intCast(score_hdim))) heads * fwd * @min(attn_keys, seq) * 2 else 0;
    const dequant: u64 = if (kv_bits >= 16) 0 else if (req.dequant_scratch_bytes > 0)
        req.dequant_scratch_bytes
    else
        2 * seq * kv_heads * hdim * 2;
    const mlp: u64 = 8 * fwd * @max(hidden, ffn) * 2;
    // The MLP envelope alone is not the whole per-token working set: an arch
    // with its own prefill streams (linear-attention layers, MoE gather-sort
    // expansion — `prefillStreamBytesPerToken`) adds to it. Floored at the
    // historical 3-envelope bill so no arch's admission ever LOOSENS: that
    // figure is conservative on plain attention archs and would be a 33%
    // UNDER-bill on a GatedDeltaNet hybrid.
    const envelope: u64 = @max(3 * mlp, mlp + fwd * stream_per_tok);
    // The dequant+GEMM prefill route (transformer.prefillDqGemm) materializes
    // a bf16 copy of the weight it is about to multiply plus its transpose,
    // and only at forwards wide enough to take that route. Measured against
    // the route off: ~0 at chunk 8192 where the envelope already dominates.
    const dq_weights: u64 = if (fwd >= transformer_mod.PREFILL_DQ_GEMM_MIN_M) dequant_weights else 0;
    const gross: u64 = kv_bytes + req.reserved_kv_bytes + req.state_bytes + req.checkpoint_bytes +
        req.grow_coexist_bytes + req.qsa_ring_bytes + req.mtp_head_kv_bytes + req.dflash_bytes + scores + dequant + envelope + dq_weights + PREFILL_RUNTIME_FLOOR_BYTES;
    // The one place a warm turn's resident rows leave the bill, inside the 5/4.
    return (gross -| req.shared_resident_bytes) * req.uplift_16ths / 16;
}

/// Per-TOKEN bytes of prefill working set that live OUTSIDE the MLP envelope,
/// because the arch runs streams the envelope does not model. MEASURED as the
/// slope of peak-above-steady-state against the prefill chunk.
///
/// Two families need it. A linear-attention hybrid holds one chunk-wide q/k/v
/// stream per LINEAR layer — all of them, not the ~3 the eval cadence bounds.
/// A MoE prefill sorts and gathers,
/// which replicates the hidden stream `top_k` times per layer alongside the
/// expert rows, for the 4 layers the MoE eval cadence lets coexist.
///
/// The linear term bills only `cadence + 1` layers: since the conv_state tail is
/// an owned copy evaluated at each cadence point, older layers' chunk inputs are
/// released inside the loop (the old all-layers figure WAS that pin).
///
/// Attention-only archs return 0 and keep exactly the bill they had.
fn prefillStreamBytesPerToken(config: *const model_mod.ModelConfig) u64 {
    var per_tok: u64 = 0;
    const linear_layers: u64 = @as(u64, config.num_hidden_layers) -| config.attnCacheLayerCount();
    if (config.linear_num_value_heads > 0 and linear_layers > 0) {
        const qkv: u64 = 2 * @as(u64, config.linear_num_key_heads) * config.linear_key_head_dim +
            @as(u64, config.linear_num_value_heads) * config.linear_value_head_dim;
        const cadence_window: u64 = @as(u64, transformer_mod.Transformer.PREFILL_EVAL_CADENCE_DEFAULT) + 1;
        per_tok += @min(linear_layers, cadence_window) * qkv * 2;
    }
    if (config.isMoe()) {
        const top_k: u64 = @max(@as(u64, config.num_experts_per_tok), 1);
        per_tok += MOE_PREFILL_COEXIST * top_k * 2 *
            (@as(u64, config.hidden_size) + config.moe_intermediate_size) * 2;
    }
    // A ringed sliding layer stages the WHOLE chunk before it compacts back to
    // its window, so the rows exist for the width of the forward and the
    // per-token KV term (global layers only) does not carry them. Bounded by
    // the same eval-cadence window as the linear term above: older layers'
    // staging is released inside the loop, so five coexist, not thirty-nine.
    per_tok += config.swaStreamBytesPerToken(
        @as(u64, transformer_mod.Transformer.PREFILL_EVAL_CADENCE_DEFAULT) + 1,
    );
    return per_tok;
}

/// The dequantized-weight working set `transformer.prefillDqGemm` materializes
/// (a bf16 copy of the weight plus its transpose, ~3 alive across the layer
/// loop). Chunk-INDEPENDENT — it is weights, not activations — and billed only
/// where the route can fire: affine-quantized weights, and (inside
/// `prefillMemoryNeeded`) forwards at least `PREFILL_DQ_GEMM_MIN_M` wide.
/// Reads the same kill switch the route does, so an A/B with the route off
/// does not get billed for it.
fn prefillDequantWeightBytes(config: *const model_mod.ModelConfig) u64 {
    if (config.quant_bits == 0 or config.quant_mode != .affine) return 0;
    if (!transformer_mod.prefillDqGemmEnabled()) return 0;
    return 3 * prefillFfnWidth(config) * @as(u64, config.hidden_size) * 2;
}

fn prefillFfnWidth(config: *const model_mod.ModelConfig) u64 {
    const per_tok: u64 = @max(@as(u64, config.num_experts_per_tok), 1);
    const moe_w: u64 = @as(u64, config.moe_intermediate_size) * per_tok +
        config.shared_expert_intermediate_size;
    const dense_w: u64 = if (config.intermediate_size_declared) config.intermediate_size else 0;
    const w = @max(dense_w, moe_w);
    return if (w > 0) w else config.intermediate_size;
}

/// Estimate peak GPU memory for prefill and reject if it would exceed the Metal
/// working-set ceiling. Exceeding it throws an uncatchable C++ exception on a
/// Metal completion-handler thread and kills the process, so PREVENTION is the
/// only lever — this guard must track what the prefill actually allocates
/// (prefillMemoryNeeded above; chunk choice from generate.effectivePrefillChunk).
/// `kv_override` is the per-request `kv_quant` body field where the surface
/// parses one (chat/messages/responses); null falls back to the process default.
/// Per-token bytes of the arch's out-of-cache request state as billed: the QSA indexer history
/// at the one copy a live slot holds, plus the f32 block-score bank. One helper because the
/// auto-context sizer and the admission guard must agree.
pub fn statePerTokenBilled(config: *const model_mod.ModelConfig) u64 {
    const one = config.qsaHistoryBytesPerToken();
    if (config.isGlm5()) return one;
    if (one == 0) return 0;
    return one + config.qsaScoreBankBytesPerToken();
}

/// Tokens of cache capacity a request reserves at admission (prompt + generation + one chunk
/// of slack); `KVCache.reserve_tokens` allocates it on the first grow so the prefill never
/// runs a second grow. The guard and the allocator are the same number by construction.
pub fn reservedCacheTokens(seq: u64, max_tokens: u64, chunk: u64, ctx: u64) u64 {
    return transformer_mod.KVCache.reservedTokens(seq, max_tokens, chunk, ctx);
}

/// Bytes of SSM checkpoints a prefill is still holding at its last chunk (~1.9 GB on a 36-layer
/// GDN trunk, billed at zero before #353). Mirrors the prefill loop's stride. `matched` is the
/// restored prefix: a warm prefill captures only over `seq - matched` (the entry's own
/// checkpoints are already resident and merged, never re-allocated).
pub fn retainedSsmCheckpointBytes(config: *const model_mod.ModelConfig, seq: u64, matched: u64, chunk: u64) u64 {
    const per_cp = config.ssmCheckpointBytes();
    if (per_cp == 0 or ssm_checkpoint_stride == 0) return 0;
    if (config.isGlm5()) return glmPrefillSchedule(seq, matched, glm5_prefix.checkpointMax(ssm_checkpoint_max)).peak * per_cp;
    const stride: u64 = prefill_plan.effectiveSsmCheckpointStride(ssm_checkpoint_stride, @max(@as(usize, @intCast(chunk)), generate_mod.prefill_chunk_override));
    if (stride == 0) return 0;
    // Stride-aligned capture positions in `(matched, seq]` plus the always-on end-of-prompt
    // snapshot, the only capture a warm append shorter than one stride makes.
    const span: u64 = seq -| @min(matched, seq);
    var n: u64 = seq / stride -| @min(matched, seq) / stride;
    if (span > prefill_plan.SSM_SNAPSHOT_BACKOFF) n += 1;
    if (ssm_checkpoint_max > 0) n = @min(n, ssm_checkpoint_max);
    if (ssmResidentCap(config)) |cap| n = @min(n, cap);
    return n * per_cp;
}

/// The checkpoint state a thinking request holds past its prefill's: the restore point behind its thought's closer, taken
/// mid-decode and held to the commit (`generate.ThoughtPoint`).
fn qwenThoughtPointBytes(config: *const model_mod.ModelConfig, warm: WarmPrefix) u64 {
    if (!warm.thought or !config.isQwen4() or ssm_checkpoint_stride == 0) return 0;
    return config.ssmCheckpointBytes();
}

/// GLM's twin: held from the closer to the commit beside the checkpoints the commit lists, only for a request that keeps any.
fn glmThoughtPointBytes(config: *const model_mod.ModelConfig, warm: WarmPrefix) u64 {
    if (!warm.thought or effectiveSsmCheckpointStride(ssm_checkpoint_stride, prefix_cache_capacity, prefix_cache_ram_enabled, diskBytesFor(config)) == 0 or warm.checkpoints == 0) return 0;
    return config.ssmCheckpointBytes();
}

/// The most checkpoint states a Qwen request holds at once when its model runs an SSD tier: the write-through
/// frees each one once its file is written, and a request it cannot serve is thinned to the same number
/// (`scheduler.qwenCheckpointCap`). Null = every checkpoint stays in RAM.
fn ssmResidentCap(config: *const model_mod.ModelConfig) ?u64 {
    if (!config.isQwen4() or !diskTierOn(config)) return null;
    return transformer_mod.SSM_RESIDENT_MAX;
}

/// The checkpoints a GLM prefill of `seq` tokens takes after restoring `matched`: the one schedule the generator runs.
fn glmPrefillSchedule(seq: u64, matched: u64, cap: u32) glm5_generate.GlmCaptureSchedule {
    const offset: usize = @intCast(@min(matched, seq));
    return glm5_generate.glmCaptureSchedule(generate_mod.glm_capture_stride, cap, offset, @as(usize, @intCast(seq)) - offset);
}

/// The restore points a DeepSeek prefill holds in RAM while it runs: with a RAM tier the capped list plus the copy
/// taken before a thin, each a full copy of the state at most `seq` tokens deep; SSD-first only what is in flight to
/// the tier. Zero with no prefix cache at all.
fn dsv41CheckpointReserve(config: *const model_mod.ModelConfig, seq: u64, chunk: u64) u64 {
    const c = config.dsv41 orelse return 0;
    const base = effectiveSsmCheckpointStride(ssm_checkpoint_stride, prefix_cache_capacity, prefix_cache_ram_enabled, diskBytesFor(config));
    const stride = prefill_plan.effectiveSsmCheckpointStride(base, @max(@as(usize, @intCast(chunk)), generate_mod.prefill_chunk_override));
    if (stride == 0) return 0;
    const one = c.stateFixedBytes(config.dsparkOn()) +| seq *| c.stateBytesPerToken();
    // SSD-first, a point the tier takes shares the request's rows and leaves RAM as the writer copies it: what is in
    // flight is the writer's bounded host buffer, and the fixed rings each point copies; the deepest few of those
    // stay in RAM as the next turn's rollback points (`dsv41_prefix.Live`).
    if (seq >= kv_disk_cache.MIN_PERSIST_TOKENS and prefix_cache_mod.ssdFirstActive(config, diskTierOn(config), prefix_cache_ram_enabled))
        return @min(disk_writer_permit_bytes, one) +| (1 + @as(u64, dsv41_prefix.live_point_cap)) *| c.stateFixedBytes(config.dsparkOn());
    const held = @min(@as(u64, dsv41_prefix.checkpointMax(ssm_checkpoint_max)), seq / stride + 1) + 1;
    return held *| one;
}

/// What a GLM request holds for the prefix cache while it runs: its prefill's KDA checkpoints (at
/// most `warm.checkpoints`, plus the copy taken before each thin; a released prefill holds
/// `glm5_prefix.checkpoint_ram_held`) and the assistant window it keeps from prefill end.
fn glmPrefixStateBytes(config: *const model_mod.ModelConfig, seq: u64, warm: WarmPrefix) u64 {
    if (effectiveSsmCheckpointStride(ssm_checkpoint_stride, prefix_cache_capacity, prefix_cache_ram_enabled, diskBytesFor(config)) == 0) return 0;
    if (warm.checkpoints == 0) return 0;
    const window: u64 = if (config.dflash_loaded) config.dflash_window_bytes else 0;
    const peak = glmPrefillSchedule(seq, warm.matched_tokens, warm.checkpoints).peak;
    const held = @as(u64, if (warm.released) @min(peak, glm5_prefix.checkpoint_ram_held) else peak) * config.ssmCheckpointBytes();
    return held +| window;
}

/// What the finished request holds while its commit runs, beside its live cache: the MLA rows copied, the
/// checkpoints, the window and the SSD writer's staged flush. Only the RAM tier copies rows, at most its
/// budget (`HotPrefixCache.glmCommitLen`); the SSD flush reads the request's own (`glmDiskLen`).
fn glmCommitStateBytes(config: *const model_mod.ModelConfig, seq: u64, warm: WarmPrefix, kv_bits: u64) u64 {
    if (effectiveSsmCheckpointStride(ssm_checkpoint_stride, prefix_cache_capacity, prefix_cache_ram_enabled, diskBytesFor(config)) == 0 or warm.checkpoints == 0) return 0;
    const sched = glmPrefillSchedule(seq, warm.matched_tokens, warm.checkpoints);
    const rows = @as(u64, sched.newest) * sessionBytesPerToken(config, kv_bits);
    const bound = resolvedPrefixCacheMem();
    const copied: u64 = if (!prefix_cache_ram_enabled) 0 else if (bound == 0) rows else @min(rows, bound);
    const window: u64 = if (config.dflash_loaded) config.dflash_window_bytes else 0;
    const listed = @min(sched.events, warm.checkpoints);
    const kept = @as(u64, if (warm.released) @min(listed, glm5_prefix.checkpoint_ram_held) else listed) * config.ssmCheckpointBytes();
    return copied +| kept +| window;
}

/// What `--mtp-head-kv-quant` does on this arch, for the boot log; null when unset.
pub fn mtpHeadKvLine(config: *const model_mod.ModelConfig, flag: bool) ?[]const u8 {
    if (!flag) return null;
    if (config.isMimo()) return "MTP head KV: --mtp-head-kv-quant has no effect on mimo_v2 (its heads keep a dense sliding window)";
    return "MTP head KV: following --kv-quant (--mtp-head-kv-quant)";
}

/// The launch flags a load can be refused for, as the process was started.
pub const LaunchFlagUse = struct { kv_quant: bool = false, qwen_gdn_16: bool = false, mtp_head_kv_quant: bool = false, drop_expert: bool = false, mtp_typical: bool = false, mtp_tokenv3: bool = false, mtp_greedy_tail: bool = false };

/// A launch flag with nothing to act on in this arch, named with the reason; null when every flag given applies.
/// An explicit flag is refused rather than ignored, before the weights load.
pub fn archFlagRefusal(config: *const model_mod.ModelConfig, use: LaunchFlagUse) ?[]const u8 {
    if (use.drop_expert and !config.isDsv41()) return "--drop-expert has no effect on this model: only deepseek_v41 packs carry an expert usage list; drop the flag";
    if (!config.isDsv41()) return null;
    if (use.kv_quant) return "--kv-quant has no effect on deepseek_v41: its compressed state is stored as the QAT round trips left it, so there is no KV cache to quantize; drop the flag";
    if (use.qwen_gdn_16) return "--qwen-gdn has no effect on deepseek_v41: it has no GatedDeltaNet layers; drop the flag";
    if (use.mtp_head_kv_quant) return "--mtp-head-kv-quant has no effect on deepseek_v41: its DSpark stages keep their own windows; drop the flag";
    if (use.mtp_typical) return "--mtp-typical has no effect on deepseek_v41: DSpark accepts drafts by the exact rule only; drop the flag";
    if (use.mtp_tokenv3) return "--mtp-tokenv3 has no effect on deepseek_v41: DSpark accepts drafts by the exact rule only; drop the flag";
    if (use.mtp_greedy_tail) return "--mtp-greedy-tail has no effect on deepseek_v41: DSpark drafts one block from the trunk's own state, with no MTP tail; drop the flag";
    return null;
}

/// A model-settings.json `kv_quant` on an arch with no KV cache is ignored with this one line, never a refusal:
/// the file outlives the flags that wrote it, and only an explicit launch flag is a request.
pub fn settingsKvQuantIgnoredLine(config: *const model_mod.ModelConfig) ?[]const u8 {
    if (!config.isDsv41() or config.kv_quant_override == null) return null;
    return "model-settings.json kv_quant is ignored on deepseek_v41 (no KV cache to quantize)";
}

/// A model-settings.json `mtp_acceptance` or `mtp_greedy_tail` on deepseek_v41 is ignored with this one line, as `kv_quant` is.
pub fn settingsMtpIgnoredLine(config: *const model_mod.ModelConfig) ?[]const u8 {
    if (!config.isDsv41()) return null;
    const acceptance = config.mtp_acceptance_override != null;
    const tail = config.mtp_greedy_tail_override != null;
    if (acceptance and tail) return "model-settings.json mtp_acceptance and mtp_greedy_tail are ignored on deepseek_v41 (DSpark accepts drafts by the exact rule only, with no MTP tail)";
    if (acceptance) return "model-settings.json mtp_acceptance is ignored on deepseek_v41 (DSpark accepts drafts by the exact rule only)";
    if (tail) return "model-settings.json mtp_greedy_tail is ignored on deepseek_v41 (DSpark has no MTP tail)";
    return null;
}

/// What `--qwen-gdn 16` does on this arch, for the boot log; null at the f32 default.
/// GLM and MiMo have no GatedDeltaNet arm, so the flag has no effect there.
pub fn qwenGdnLine(config: *const model_mod.ModelConfig, f32_state: bool) ?[]const u8 {
    if (f32_state) return null;
    if (config.isGlm5()) return "GDN state: --qwen-gdn 16 has no effect on GLM (its KDA state is f32 already)";
    if (config.isMimo()) return "GDN state: --qwen-gdn 16 has no effect on mimo_v2 (no GatedDeltaNet layers)";
    return "GDN state: bf16 between tokens (--qwen-gdn 16; f32 is the default; the decay gate is f32 either way)";
}

/// Per-token bytes of the qwen4 MTP head's own KV at the scheme it was LOADED with,
/// billed once per request when MTP is on. It was resident and unbilled before.
pub fn mtpHeadKvBytesPerToken(config: *const model_mod.ModelConfig) u64 {
    if (!config.longCtxGated()) return 0;
    const n = config.attnCacheLayerCount();
    if (n == 0) return 0;
    const head_cfg = transformer_mod.Transformer.qwen4MtpHeadKvConfig(configuredKvQuantFor(config));
    const bits: u64 = if (head_cfg.scheme == .off) 16 else head_cfg.bits;
    return kvBytesPerTokenAtBits(config.kvBytesPerToken() / n, bits);
}

fn mtpHeadQsaRingBytes(config: *const model_mod.ModelConfig) u64 {
    if (!config.longCtxGated()) return 0;
    const n = config.attnCacheLayerCount();
    if (n == 0) return 0;
    return config.qsaRingBytes() / n;
}

fn mtpHeadStateBytesPerToken(config: *const model_mod.ModelConfig) u64 {
    if (!config.longCtxGated()) return 0;
    const n = config.attnCacheLayerCount();
    if (n == 0) return 0;
    return statePerTokenBilled(config) / n;
}

fn sessionBytesPerToken(config: *const model_mod.ModelConfig, kv_bits: u64) u64 {
    if (config.dsv41) |c| return c.stateBytesPerToken();
    if (config.isGlm5()) return kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) + config.qsaHistoryBytesPerToken();
    const head: u64 = if (mtpHeadDefaultOn(config)) mtpHeadKvBytesPerToken(config) +| mtpHeadStateBytesPerToken(config) else 0;
    return kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(config) +| head;
}

/// Most MLA layers among any `pending` consecutive layers: MLA-only prefill scratch is held
/// once per MLA layer in flight, not per pending layer.
fn glmMlaLayersPending(config: *const model_mod.ModelConfig, pending: u32) u32 {
    var most: u32 = 0;
    var start: u32 = 0;
    while (start + pending <= config.num_hidden_layers) : (start += 1) {
        var count: u32 = 0;
        for (start..start + pending) |i| count += @intFromBool(!config.isLinearLayer(@intCast(i)));
        most = @max(most, count);
    }
    return most;
}

/// Native MLA fuses attention and bounds index scores internally. It stores no
/// per-head K/V, affine rebuild, QSA score bank or generic SSM checkpoints.
fn glm5TransientBytes(config: *const model_mod.ModelConfig, seq: u64, chunk: u64, kv_bits: u64) u64 {
    const rows = @min(chunk, @max(seq, 1));
    const width = @as(u64, config.linear_num_value_heads) * config.linear_value_head_dim;
    const per_row = @as(u64, config.hidden_size) * 64 + width * 64 +
        @as(u64, config.num_experts_per_tok) * (@as(u64, config.moe_intermediate_size) * 8 + @as(u64, config.hidden_size) * 4);
    // A growing latent/pool layer retains its old allocation through evaluation.
    const latent_row = kvBytesPerTokenAtBits(@as(u64, config.mla_kv_lora_rank) * 2, kv_bits);
    const grow = seq * (latent_row + @as(u64, config.indexer_head_dim) * 2 / @max(config.indexer_compress_ratio, 1));
    const pending = 2; // glm5_forward.Request.prefill_sync_layers
    const mla_pending = glmMlaLayersPending(config, pending);
    const n = std.math.cast(usize, rows) orelse return std.math.maxInt(u64);
    const quant_scratch: u64 = if (kv_bits < 16) @import("glm5/glm5_latent.zig").scratchBytes(config.mla_kv_lora_rank, n, mla_pending, @intCast(kv_bits)) else 0;
    const native = (@import("glm5/glm5_a6_dense_once.zig").transientBudget(n, pending) catch return std.math.maxInt(u64)) +|
        (@import("glm5/glm5_mla_prefill_batch.zig").transientBudget(n, mla_pending) catch return std.math.maxInt(u64)) +|
        (@import("glm5/glm5_attention_nax_packed.zig").transientBudget(n, mla_pending) catch return std.math.maxInt(u64)) +|
        (@import("glm5/glm5_attention.zig").packedCadenceTransientBudget(n, mla_pending) catch return std.math.maxInt(u64)) +|
        (@import("glm5/glm5_indexpool_nax.zig").transientBudget(n, mla_pending) catch return std.math.maxInt(u64)) +|
        (@import("glm5/glm5_kda_prefill_cluster.zig").transientBudget(n, pending) catch return std.math.maxInt(u64)) +|
        (@import("glm5/glm5_attention_decode_batch.zig").transientBudget(4) catch return std.math.maxInt(u64));
    const dflash = if (config.dflash_loaded) glmDflashRequestBytes(config, rows) else 0;
    const attention = @import("glm5/glm5_attention.zig");
    const history = std.math.cast(usize, seq) orelse return std.math.maxInt(u64);
    const selection = @max(attention.score_scratch_bytes, attention.selectionScratchBytes(history) catch return std.math.maxInt(u64));
    return rows *| per_row +| grow +| quant_scratch +| slotRingBytes(config, 16) +| native +| dflash +| glmFp8DequantScratchBytes(config, rows, pending) +|
        selection +| attention.attention_scratch_bytes +| PREFILL_RUNTIME_FLOOR_BYTES;
}

/// MiMo's raw-FP8 prefill route retains BF16 scratch for each projection until
/// the pending layer graph is evaluated. Decode uses its direct GEMV, with none.
pub fn glmFp8DequantScratchBytes(config: *const model_mod.ModelConfig, rows: u64, pending_layers: u64) u64 {
    if (!config.glm_fp8_trunk or rows <= @as(u64, @intCast(fp8_block.gemv_max_rows))) return 0;
    const hidden: u64 = config.hidden_size;
    const dense = 3 *| hidden *| config.intermediate_size;
    const shared = 3 *| hidden *| config.moe_intermediate_size;
    const mla = @as(u64, config.mla_q_lora_rank) *| (hidden +| @as(u64, config.num_attention_heads) *| config.mla_qk_nope_head_dim) +|
        @as(u64, config.mla_kv_lora_rank) *| hidden +| hidden *| config.num_attention_heads *| config.mla_v_head_dim +| shared;
    return @max(dense, mla) *| 2 *| pending_layers;
}

/// Dense sliding assistant window, capture/encoder/projection rows, target
/// replay states and the native verifier's bounded MLA work. Model weights
/// are already resident and are never charged again per request.
pub fn glmDflashRequestBytes(config: *const model_mod.ModelConfig, chunk: u64) u64 {
    if (!config.dflash_loaded) return 0;
    return config.dflash_window_bytes *| 4 +|
        config.dflash_capture_bytes_per_token *| chunk +|
        config.ssmCheckpointBytes() *| 3 +|
        @import("glm5/glm5_dflash_memory.zig").limit_bytes +| 64 * 1024 * 1024;
}

/// The sliding DFlash2 assistant's per-request bytes on a non-GLM target: the physical window and
/// its trim copies, plus the tap and encoder rows of one prefill chunk.
pub fn dflashRequestBytes(config: *const model_mod.ModelConfig, chunk: u64) u64 {
    if (!config.dflash_loaded or config.isGlm5()) return 0;
    return config.dflash_window_bytes *| 4 +| config.dflash_capture_bytes_per_token *| chunk +| 64 * 1024 * 1024;
}

/// What a GLM DFlash2 request may still allocate when it reserves its verifier capacity right after
/// prefill: its admission bill at the narrowest width (no width bills less) less what it holds.
pub fn glmDflashReserveBudget(config: *const model_mod.ModelConfig, seq: u64, max_tokens: u32, kv_bits: u64, held: u64) u64 {
    // `max_tokens` is already clamped to admission's context; the caller's config copy is not pinned to it.
    var sized = config.*;
    sized.pinned_context = @intCast(@min(seq +| max_tokens, std.math.maxInt(u32)));
    return prefillNeededAtChunk(&sized, seq, max_tokens, kv_bits, 1, .{}) -| held;
}

/// A native GLM request's latent and pooled-index capacity at its eventual length.
fn glmCacheBytes(config: *const model_mod.ModelConfig, prompt_len: u64, max_tokens: u32, kv_bits: u64) u64 {
    const ctx = getEffectiveContextLength(config);
    const reserved = @max(prompt_len, @min(prompt_len +| max_tokens, ctx));
    // Native appendRows rounds latent and pooled capacities separately to 256 rows.
    const rows = (reserved +| 1023) / 1024 * 1024;
    return sessionBytesPerToken(config, kv_bits) *| rows;
}

/// The per-request terms of the admission bill, in one place.
pub fn prefillRequestTerms(config: *const model_mod.ModelConfig, seq: u64, max_tokens: u64, kv_bits: u64, chunk: u64, warm: WarmPrefix) PrefillRequestTerms {
    // Arch gate for every term (all new, all measured on qwen4_exp alone; the reservation's
    // allocator twin is gated too, so an ungated guard billed memory never reserved). `.{}` is
    // the identity: `prefillMemoryNeeded` then reduces to the previous expression. A ringed arch
    // joins through the narrower `reservesKvCapacity`; every term it does not have stays zero on
    // its own (no QSA history, no SSM checkpoints); its MTP heads bill `mimo_mtp.State.billedBytes`.
    if (!config.reservesKvCapacity()) return .{};
    // `reservedTokens` returns 0 below its length threshold; floor the reserved length at `seq`.
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(config)), seq);
    // Only the headroom is new here: the prompt's own rows are already billed.
    const kv_per_tok = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits);
    const mtp_on = warm.mtp_on or (forceMtpFor(config) and !config.dflash_loaded);
    const head_per_tok: u64 = if (mtp_on) mtpHeadKvBytesPerToken(config) else 0;
    const head_qsa_ring: u64 = if (mtp_on) mtpHeadQsaRingBytes(config) else 0;
    const head_state_per_tok: u64 = if (mtp_on) mtpHeadStateBytesPerToken(config) else 0;
    // Rows the restore hands over without allocating another copy. KV only: the history is
    // privatised on restore (`seedCapBuf`), so crediting it cancelled the one-copy bill outright.
    const credited = warm.creditedRows(reserved);
    return .{
        .reserved_kv_bytes = (reserved -| seq) * kv_per_tok,
        .state_bytes = reserved * (statePerTokenBilled(config) +| head_state_per_tok) +|
            (if (mtp_on) mimo_mtp.State.billedBytes(config) else 0),
        // The warm span, not the prompt; read regardless of `will_donate` (a shared restore skips the same rows).
        .checkpoint_bytes = retainedSsmCheckpointBytes(config, seq, warm.matched_tokens, chunk) +| qwenThoughtPointBytes(config, warm),
        .shared_resident_bytes = credited *| kv_per_tok,
        .grow_coexist_bytes = growCoexistBytes(config, warm, seq, reserved, kv_per_tok),
        .qsa_ring_bytes = slotRingBytes(config, kv_bits) +| head_qsa_ring,
        .mtp_head_kv_bytes = reserved *| head_per_tok,
        .dflash_bytes = dflashRequestBytes(config, @min(chunk, @max(seq, 1))),
        .dequant_scratch_bytes = kvDequantScratchBytes(config, seq, @min(chunk, seq)),
        .uplift_16ths = billUplift16ths(config, seq),
    };
}

// Numeric slot diagnostics cross to error mapping only on this connection's thread.
pub threadlocal var slot_memory_context_refusal: ?model_registry_mod.MemoryContextRefusal = null;
pub threadlocal var slot_media_refusal: ?scheduler_mod.MediaRefusal = null;

/// The message `error.GenerationOutOfMemory` sends on every surface and both paths.
pub const GEN_OOM_MSG = "The engine ran out of GPU memory during this request and it was abandoned. The server is still running. Reduce the prompt length, lower --ctx-size, or free memory on the machine.";

/// The body of the pre-flight memory 400. Two arms: with the credits present the message
/// quotes them; with them structurally zero (every arch outside `admissionEvictsHotCache`)
/// the previous sentence, which named no cache. The discriminator is the arch gate, not `bill.evictable`,
/// so the qwen4_exp bytes stay exactly as they were. Caller owns the returned bytes.
fn memoryRefusalMessage(
    allocator: std.mem.Allocator,
    prompt_len: usize,
    needed_mb: u64,
    avail_mb: u64,
    bill: AdmissionBill,
    carries_cache: bool,
) ![]u8 {
    if (!carries_cache) {
        return std.fmt.allocPrint(allocator, "Prompt ({d} tokens) requires ~{d}MB GPU memory but only ~{d}MB available. Reduce prompt size or use a smaller model.", .{ prompt_len, needed_mb, avail_mb });
    }
    return std.fmt.allocPrint(allocator, "Prompt ({d} tokens) requires ~{d}MB GPU memory but only ~{d}MB is available and the hot prefix cache holds ~{d}MB more, all of which can be reclaimed (~{d}MB — the shortfall is elsewhere), which is not enough. Reduce prompt size, lower --ctx-size, or use a smaller model.", .{ prompt_len, needed_mb, avail_mb, bill.evictable / (1024 * 1024), bill.evictionCredit() / (1024 * 1024) });
}

/// The admission bill and the headroom it is compared against, so the connection thread's
/// guard and the inference thread's evict-or-refuse decision cannot compute them differently.
pub const AdmissionBill = struct {
    needed: u64 = 0,
    available: u64 = 0,
    /// Bytes the hot prefix cache is holding. Reported in a refusal, never credited (see `reclaimable`).
    evictable: u64 = 0,
    /// The part of `evictable` an eviction can prove it will return: residency minus the
    /// largest entry (a restore pins at most one).
    reclaimable: u64 = 0,
    /// The prefill width this bill was taken at, so a refusal can name it.
    chunk: u64 = 0,

    pub fn fits(self: AdmissionBill) bool {
        return self.needed <= self.available;
    }

    /// The one credit the decision and both sentences read: `reclaimable`, never `evictable`
    /// (a refusal once quoted the slot's own checked-out prefix as "evicted").
    pub fn evictionCredit(self: AdmissionBill) u64 {
        return self.reclaimable;
    }

    /// Fits once the cache has given back what it provably can. Credits `reclaimable`, not
    /// `evictable`: a 383k prompt was admitted on one entry it then restored from and refused
    /// after the eviction pass returned nothing.
    pub fn fitsAfterEviction(self: AdmissionBill) bool {
        return self.needed <= self.available +| self.evictionCredit();
    }
};

/// Unloads idle resident models, least recently used first, until `fits` holds; false when
/// nothing idle is left. An idle model is memory its next request reloads.
pub fn fitAfterUnloadingIdle(ctx: anytype, fits: fn (@TypeOf(ctx)) bool, unloadOne: fn (@TypeOf(ctx)) bool) bool {
    while (!fits(ctx)) if (!unloadOne(ctx)) return false;
    return true;
}

test "a prompt that does not fit unloads idle models, oldest first, until it fits" {
    const t = std.testing;
    const Fake = struct {
        needed: u64,
        available: u64,
        idle: []const u64, // resident bytes, least recently used first
        unloaded: usize = 0,
        fn fits(f: *@This()) bool {
            return f.needed <= f.available;
        }
        fn unloadOne(f: *@This()) bool {
            if (f.unloaded == f.idle.len) return false;
            f.available += f.idle[f.unloaded];
            f.unloaded += 1;
            return true;
        }
    };
    // Already fits: nothing is unloaded.
    var a = Fake{ .needed = 5, .available = 9, .idle = &.{ 15, 3 } };
    try t.expect(fitAfterUnloadingIdle(&a, Fake.fits, Fake.unloadOne));
    try t.expectEqual(@as(usize, 0), a.unloaded);
    // One idle model is enough: the second stays resident.
    var b = Fake{ .needed = 10_558, .available = 9_466, .idle = &.{ 14_730, 8_000 } };
    try t.expect(fitAfterUnloadingIdle(&b, Fake.fits, Fake.unloadOne));
    try t.expectEqual(@as(usize, 1), b.unloaded);
    // Every idle model goes and it still does not fit: refused.
    var c = Fake{ .needed = 50, .available = 9, .idle = &.{ 15, 3 } };
    try t.expect(!fitAfterUnloadingIdle(&c, Fake.fits, Fake.unloadOne));
    try t.expectEqual(@as(usize, 2), c.unloaded);
}

test "a refusal names the credit it COMPARED, never the residency it cannot take" {
    const t = std.testing;
    const mb: u64 = 1024 * 1024;

    // The live shape: the whole residency is the checked-out entry.
    const pinned_only = AdmissionBill{
        .needed = 13_664 * mb,
        .available = 13_374 * mb,
        .evictable = 7_802 * mb,
        .reclaimable = 0,
        .chunk = 512,
    };
    try t.expectEqual(@as(u64, 0), pinned_only.evictionCredit());
    try t.expectEqual(7_802 * mb, pinnedResidentBytes(pinned_only));
    try t.expect(!pinned_only.fits());
    try t.expect(!pinned_only.fitsAfterEviction());
    try t.expectEqual(
        pinned_only.needed <= pinned_only.available +| pinned_only.evictionCredit(),
        pinned_only.fitsAfterEviction(),
    );

    // The same residency, genuinely reclaimable: the credit is real and the request is admitted to evict.
    var reclaimable = pinned_only;
    reclaimable.reclaimable = 7_802 * mb;
    try t.expectEqual(7_802 * mb, reclaimable.evictionCredit());
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(reclaimable));
    try t.expect(reclaimable.fitsAfterEviction());
}

/// Everything the admission guard bills for one request at one prefill width, so the
/// per-request chunk chooser prices a rung with the same function that admits it. `chunk` is
/// the width the forward will run, already through `generate.effectivePrefillChunk`.
pub fn prefillNeededAtChunk(
    config: *const model_mod.ModelConfig,
    seq: u64,
    max_tokens: u32,
    kv_bits: u64,
    chunk: u64,
    warm: WarmPrefix,
) u64 {
    if (config.dsv41) |c| {
        const held = (@as(u64, seq) +| max_tokens) *| c.stateBytesPerToken() +| c.stateFixedBytes(config.dsparkOn());
        // The transient cap a forward runs under (`Config.forwardCap`) is its bill. The prefill's comes out of the decoder
        // pools it may release (`dsv41_borrow`); a decode forward's never does. The decode pool is parked beside either.
        const work = @max(c.forwardCap(dsv41_borrow.verify_rows, seq), c.chunkCap(chunk, seq) -| dsv41BorrowableBytes(config)) +| config.expertPoolBytes();
        return (held +| dsv41CheckpointReserve(config, seq, chunk)) *| 5 / 4 +| work +| (if (config.expert_streaming) config.expert_fill_peak_bytes else 0);
    }
    if (config.isGlm5()) {
        return (glmCacheBytes(config, seq, max_tokens, kv_bits) +| @max(
            glm5TransientBytes(config, seq, chunk, kv_bits) +| glmPrefixStateBytes(config, seq, warm),
            glmCommitStateBytes(config, seq, warm, kv_bits) +| glmThoughtPointBytes(config, warm),
        )) *| 5 / 4 +|
            (if (config.expert_streaming) config.expert_fill_peak_bytes else 0) +|
            cadenceInflightBytes(mlx_mb_per_buffer);
    }
    const heads: u64 = config.num_attention_heads;
    const kv_heads: u64 = config.num_key_value_heads;
    const hdim: u64 = config.head_dim;
    const hidden: u64 = config.hidden_size;
    const ffn: u64 = prefillFfnWidth(config);
    return prefillMemoryNeeded(seq, heads, kv_heads, config.kvBytesPerToken(), hdim, config.prefillScoreHeadDim(), hidden, ffn, kv_bits, chunk, config.prefillAttnKeys(seq), prefillStreamBytesPerToken(config), prefillDequantWeightBytes(config), prefillRequestTerms(config, seq, max_tokens, kv_bits, chunk, warm)) +
        qsaMaskBytes(config, @min(chunk, @max(seq, 1)), seq) +
        slidingBandScoreBytes(config, @min(chunk, @max(seq, 1))) +
        fp8DequantScratchBytes(config, @min(chunk, @max(seq, 1))) +
        (if (config.expert_streaming) config.expert_fill_peak_bytes else 0) +
        cadenceInflightBytes(mlx_mb_per_buffer);
}

/// Headroom a rung must clear beyond its own bill when the long-context gate is what widened
/// it: past that width the machine cap frozen at load no longer bounds the forward, and
/// `prefillMemoryNeeded` is documented to UNDER-bill quantized prefill by about this much.
/// Narrower rungs keep the plain bill, so no other admission decision moves.
pub const PREFILL_WIDE_RUNG_MARGIN_PCT: u64 = 22;

/// Test seam: false makes the load-time pin the width again.
pub var per_request_chunk_override: ?bool = null;

pub fn perRequestPrefillChunkEnabled(config: *const model_mod.ModelConfig) bool {
    if (!config.perRequestPrefillChunk()) return false;
    return per_request_chunk_override orelse true;
}

/// The widest ladder rung this request can afford, priced by the same estimator that admits
/// it against post-eviction memory. Candidates are priced at the width the forward will run
/// (`effectivePrefillChunk` caps by arch), and the width is returned, not the rung. An explicit
/// `--prefill-chunk` (`chunk_override`) caps every rung; nothing fitting means the capped floor.
pub fn chooseRequestPrefillChunk(
    config: *const model_mod.ModelConfig,
    seq: u64,
    max_tokens: u32,
    kv_bits: u64,
    available: u64,
    load_time_pin: u32,
    chunk_override: u32,
    warm: WarmPrefix,
) u32 {
    // The arch gate first: an ungated arch's answer is the load-time pin, as before.
    if (!perRequestPrefillChunkEnabled(config)) return load_time_pin;
    const top = ladderTop(config, chunk_override);
    // GLM's checkpoints yield to the width: a request keeps those the chosen width leaves room for.
    var rung_warm = warm;
    rung_warm.checkpoints = 0;
    for (PREFILL_CHUNK_LADDER) |r| {
        const rung = cappedRung(r, top);
        const width: u64 = rungWidth(config, seq, rung, config.longCtxGated());
        const bill = prefillNeededAtChunk(config, seq, max_tokens, kv_bits, width, rung_warm);
        const needed = if (width > rungWidth(config, seq, rung, false))
            bill *| (100 + PREFILL_WIDE_RUNG_MARGIN_PCT) / 100
        else
            bill;
        if (needed <= available) return @intCast(width);
    }
    return @intCast(rungWidth(config, seq, cappedRung(PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1], top), config.longCtxGated()));
}

fn cappedRung(rung: u32, cap: u32) u32 {
    return if (cap == 0) rung else @min(rung, cap);
}

/// The widest rung a request may take: an explicit `--prefill-chunk`, else the arch's start width.
fn ladderTop(config: *const model_mod.ModelConfig, chunk_override: u32) u32 {
    return if (chunk_override > 0) chunk_override else config.prefillStartWidth();
}

/// The width one ladder rung forwards at. `long_ctx_gated = false` asks what the rung would be
/// without the long-context gate, i.e. the width the load-time machine cap still bounds.
fn rungWidth(config: *const model_mod.ModelConfig, seq: u64, rung: u32, long_ctx_gated: bool) u64 {
    // `--prefill-chunk` is unbounded at its source; clamp before narrowing.
    return @min(
        @as(u64, prefill_plan.requestPrefillChunk(
            config.prefillScoreHeadDim(),
            config.num_attention_heads,
            @intCast(seq),
            config.has_sliding_window,
            config.isMoe(),
            long_ctx_gated,
            rung,
        )),
        @as(u64, std.math.maxInt(u32)),
    );
}

/// The width handed to the forward (`Generator.pinned_prefill_chunk`): one rule for the
/// connection thread's bill and the scheduler's pick. Off the ladder (the vision kill switch, an
/// arch that does not pick per request) an explicit width outranks the load-time pin.
pub fn requestPrefillPick(config: *const model_mod.ModelConfig, seq: u64, max_tokens: u32, kv_bits: u64, available: u64, unchunked_prefill: bool, warm: WarmPrefix) u32 {
    const explicit = explicitPrefillChunk();
    if (unchunked_prefill or !perRequestPrefillChunkEnabled(config))
        return if (explicit > 0) explicit else config.pinned_prefill_chunk;
    return chooseRequestPrefillChunk(config, seq, max_tokens, kv_bits, available, config.pinned_prefill_chunk, explicit, warm);
}

/// Impure wrapper the scheduler reaches through `prefill_request_chunk`, mirroring
/// `prefillFitsNow`'s inputs. Reads live memory, so it must run after the eviction pass.
pub fn requestPrefillChunkNow(
    config: *const model_mod.ModelConfig,
    prompt_len: usize,
    max_tokens: u32,
    kv_cfg: transformer_mod.KVQuantConfig,
    unchunked_prefill: bool,
    warm_matched: u64,
    warm_capacity: u64,
    warm_will_donate: bool,
    enable_mtp: bool,
) u32 {
    const pin: u32 = config.pinned_prefill_chunk;
    const seq: u64 = @intCast(prompt_len);
    const kv_bits: u64 = if (kv_cfg.scheme == .off) 16 else kv_cfg.bits;
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    const available: u64 = currentGpuMemoryCeiling(config, active_mem) -| active_mem;
    const warm = WarmPrefix{ .matched_tokens = warm_matched, .capacity_tokens = warm_capacity, .will_donate = warm_will_donate, .mtp_on = enable_mtp };
    const chosen = requestPrefillPick(config, seq, max_tokens, kv_bits, available, unchunked_prefill, warm);
    if (chosen != pin and !unchunked_prefill and perRequestPrefillChunkEnabled(config)) {
        const width: u64 = chosen;
        const terms = prefillRequestTerms(config, seq, max_tokens, kv_bits, width, warm);
        const kv_bytes: u64 = seq *| kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +| terms.reserved_kv_bytes;
        log.debug("[prefill] chunk {d} for this request (reserve {d} MB beside KV {d} MB)\n", .{ width, prefillTransientReserve(config, kv_bits, width) >> 20, kv_bytes >> 20 });
    }
    return chosen;
}

/// Test seam: false turns the per-chunk adaptive width off.
pub var adaptive_chunk_override: ?bool = null;

/// The per-chunk adaptive width: subordinate to the per-request gate, plus "no operator pinned a width".
pub fn adaptivePrefillChunkEnabled(config: *const model_mod.ModelConfig) bool {
    // Arch first: the cheapest and most selective, and this runs once per chunk boundary.
    if (!config.perRequestPrefillChunk()) return false;
    if (explicitPrefillChunk() > 0) return false;
    if (!perRequestPrefillChunkEnabled(config)) return false;
    return adaptive_chunk_override orelse true;
}

/// Consecutive supporting probes a widening needs: the probe is taken after `mlx_clear_cache`
/// and never contains a chunk's own peak, so the second reading pays for that blind spot.
pub const PREFILL_WIDEN_PROBES: u8 = 2;

/// What one chunk of `width` costs beyond what is already resident: its transient envelope
/// plus the per-token state the chunk is about to write. The growth term is billed even under
/// the reservation (short prompts do grow per chunk).
pub fn prefillChunkCost(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    width: u64,
    kv_len: u64,
) u64 {
    const per_tok: u64 = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +|
        statePerTokenBilled(config);
    return (prefillTransientReserveAtKv(config, kv_bits, width, kv_len) -| dsv41BorrowableBytes(config)) +| (width *| per_tok);
}

/// The decoder pools a DeepSeek prefill may still release for its working set (`dsv41_borrow`); 0 elsewhere.
fn dsv41BorrowableBytes(config: *const model_mod.ModelConfig) u64 {
    return @atomicLoad(u64, &config.expert_borrow_bytes, .monotonic);
}

/// `dsv41_borrow.headroom_fn`: what admission reads as available, live.
fn dsv41Headroom(config: *const model_mod.ModelConfig) u64 {
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    return dsv41HeadroomOf(admissionAvailable(currentGpuMemoryCeiling(config, active_mem), active_mem), config);
}

/// The decode pool the ledger parks in MLX's cache is no part of its active bytes, so every live check (decode and
/// prefill width) takes it off here once; `prefillNeededAtChunk` bills it once for the same reason.
fn dsv41HeadroomOf(available: u64, config: *const model_mod.ModelConfig) u64 {
    return available -| config.expertPoolBytes();
}

/// The width the next chunk runs at. Asymmetric because a Metal abort is uncatchable: HOLD
/// while `headroom >= cost(current)` (the estimator carries its own 5/4); STEP DOWN at once,
/// by as many rungs as it takes, floored at `PREFILL_CHUNK_FLOOR`; WIDEN one rung only at
/// 1.25x its cost, after `PREFILL_WIDEN_PROBES` supporting probes, up to `cap`, never after a
/// step-down. `cap` is the widest width this arch forwards for this prompt, not a raw rung.
pub fn adaptivePrefillWidth(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    kv_len: u64,
    headroom: u64,
    current: u32,
    cap: u32,
    st: *mtp_plan.AdaptiveWidthState,
) u32 {
    if (cap == 0 or current == 0) return current;
    const floor: u32 = @min(cap, @as(u32, @intCast(prefill_plan.PREFILL_CHUNK_FLOOR)));

    if (prefillChunkCost(config, kv_bits, current, kv_len) > headroom) {
        st.supporting = 0;
        st.ratcheted = true;
        var best: u32 = floor;
        for (PREFILL_CHUNK_LADDER) |rung| {
            const w: u32 = @min(rung, cap);
            if (w >= current) continue;
            if (prefillChunkCost(config, kv_bits, w, kv_len) <= headroom) {
                best = w;
                break;
            }
        }
        return @min(best, current);
    }

    var next: u32 = 0;
    for (PREFILL_CHUNK_LADDER) |rung| {
        const w: u32 = @min(rung, cap);
        if (w > current and (next == 0 or w < next)) next = w;
    }
    if (next == 0 or st.ratcheted) {
        st.supporting = 0;
        return current;
    }
    const asked: u64 = prefillChunkCost(config, kv_bits, next, kv_len) *| 5 / 4;
    if (asked > headroom) {
        st.supporting = 0;
        return current;
    }
    st.supporting +|= 1;
    if (st.supporting < PREFILL_WIDEN_PROBES) return current;
    st.supporting = 0;
    return next;
}

/// The live headroom one prefill chunk has to fit in: the GPU ceiling less what MLX's
/// allocator holds, read once per chunk boundary after its `mlx_clear_cache`. The same
/// expression `prefillAdmissionBill` compares against.
pub fn prefillHeadroomNow(config: *const model_mod.ModelConfig, staged_host: u64) u64 {
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    // The SSD write-through stages chunk blobs as host bytes behind a ~1 GiB permit that
    // `mlx_get_active_memory` does not count; on unified memory they are the same pool. Read at
    // the hook site (inference thread) and passed in.
    return prefillHeadroomFrom(currentGpuMemoryCeiling(config, active_mem), active_mem, staged_host);
}

/// Pure half of `prefillHeadroomNow`.
pub fn prefillHeadroomFrom(ceiling: u64, active: u64, staged_host: u64) u64 {
    return ceiling -| active -| staged_host;
}

/// Does a widen the pre-tick probe chose still fit after the interleave tick allocated for
/// co-tenant slots? Same 1.25 bar, same estimator, same live KV.
pub fn adaptivePrefillWidenStillFits(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    pos: usize,
    want: u32,
    staged_host: u64,
) bool {
    if (!adaptivePrefillChunkEnabled(config)) return false;
    const headroom = prefillHeadroomNow(config, staged_host);
    const asked: u64 = prefillChunkCost(config, kv_bits, want, pos) *| 5 / 4;
    if (asked <= headroom) return true;
    log.info("[prefill] widen to {d} withdrawn at pos {d} (headroom {d} MB after the decode tick, cost {d} MB)\n", .{
        want,
        pos,
        headroom >> 20,
        asked >> 20,
    });
    return false;
}

/// Impure wrapper the prefill loop reaches through `generate.prefill_chunk_adapt`. Logs one
/// line per change, never per boundary (the bench harness greps it).
pub fn adaptivePrefillWidthNow(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    pos: usize,
    current: u32,
    cap: u32,
    st: *mtp_plan.AdaptiveWidthState,
    staged_host: u64,
) u32 {
    if (!adaptivePrefillChunkEnabled(config)) return current;
    const headroom = prefillHeadroomNow(config, staged_host);
    // `pos` is the live KV this chunk will attend to (see `prefillTransientReserveAtKv`).
    const next = adaptivePrefillWidth(config, kv_bits, pos, headroom, current, cap, st);
    if (next != current) {
        // Quote the number it compared: `prefillChunkCost` at this KV.
        log.info("[prefill] width {d} -> {d} at pos {d} (headroom {d} MB, cost {d} MB)\n", .{
            current,
            next,
            pos,
            headroom >> 20,
            prefillChunkCost(config, kv_bits, next, pos) >> 20,
        });
    }
    return next;
}

/// What a request may allocate: the ceiling less MLX's allocations. The SSD writers' host bytes are
/// pageable memory outside the wired limit, so the ceiling's margin under it is the only reserve.
pub fn admissionAvailable(total_limit: u64, active_mem: u64) u64 {
    return total_limit -| active_mem;
}

/// The hot cache is read once, as the published evictable atomic, never billed on top.
pub fn prefillAdmissionBill(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_override: ?transformer_mod.KVQuantConfig, unchunked_prefill: bool, prompt_tokens: ?[]const u32, warm: WarmPrefix) AdmissionBill {
    const heads = config.num_attention_heads;
    if (heads == 0) return .{ .needed = 0, .available = std.math.maxInt(u64) };
    const seq: u64 = @intCast(prompt_len);

    const kv_cfg: transformer_mod.KVQuantConfig = kv_override orelse configuredKvQuantFor(config);
    const kv_bits: u64 = if (kv_cfg.scheme == .off) 16 else kv_cfg.bits;
    // Available = the lesser of Metal's static working-set max and what is physically reachable
    // now (#64). Read before the width is chosen: on a per-request arch the width is a function of it.
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    const total_limit: u64 = currentGpuMemoryCeiling(config, active_mem);
    const available = admissionAvailable(total_limit, active_mem);

    // A vision prefill chunks like text since issue #197 (the splice resumes
    // its row index across chunks), so it bills the chunk-bounded envelope —
    // UNLESS the SUSHI_VISION_CHUNKED=0 kill switch restored the
    // whole-prompt forward, in which case `unchunked_prefill` bills the real
    // width. Call sites pass generate_mod.visionPrefillUnchunked(has_vision)
    // so the guard and the prefill loop cannot disagree.
    //
    // Otherwise the width comes from `requestPrefillPick`, the same rule the scheduler pins with.
    const chosen: u32 = if (unchunked_prefill) 0 else requestPrefillPick(config, seq, max_tokens, kv_bits, available, false, warm);
    const chunk: u64 = if (unchunked_prefill)
        @max(seq, 1)
    else
        @intCast(prefill_plan.requestPrefillChunk(config.prefillScoreHeadDim(), config.num_attention_heads, prompt_len, config.has_sliding_window, config.isMoe(), config.longCtxGated(), chosen));
    // RAM hot-cache restores rebind MLX array handles by refcount; they do not
    // allocate another copy of the cached buffers. `active_mem` above already
    // includes the resident entry, while `prefillMemoryNeeded` bills the full
    // destination KV capacity that may be allocated when the restored cache
    // grows. Adding the resident entry here again would invent a third copy
    // and reject long warm prompts (and even cache misses) spuriously.
    const needed: u64 = prefillNeededAtChunk(config, seq, max_tokens, kv_bits, chunk, warm);
    // Published number, never the pointer: `hot_prefix_cache` is inference-thread state, freed
    // on every model switch. Stale by at most one entry, which is all a hint has to be.
    const evictable: u64 = if (global_scheduler) |sch|
        sch.resident_hot_cache_bytes.load(.monotonic)
    else
        0;
    // Prompt-aware when the caller has the ids: an entry this prompt could not restore from is
    // genuinely evictable. Still a published snapshot, never the cache pointer.
    const reclaimable: u64 = if (global_scheduler) |sch|
        if (prompt_tokens) |toks|
            scheduler_mod.reclaimableHotCacheBytesFor(sch, toks)
        else
            sch.reclaimable_hot_cache_bytes.load(.monotonic)
    else
        0;
    return creditedAdmissionBill(config, needed, available, evictable, reclaimable, chunk);
}

/// PURE: the bill with the hot-cache credits on an arch that evicts to admit
/// (`admissionEvictsHotCache`). Elsewhere both credits are zero, so both admit arms go dead and
/// the single `needed > available` refusal stands. The publishers stay on every arch.
pub fn creditedAdmissionBill(config: *const model_mod.ModelConfig, needed: u64, available: u64, evictable: u64, reclaimable: u64, chunk: u64) AdmissionBill {
    if (!config.admissionEvictsHotCache()) {
        return .{ .needed = needed, .available = available, .chunk = chunk };
    }
    return .{ .needed = needed, .available = available, .evictable = evictable, .reclaimable = reclaimable, .chunk = chunk };
}

/// Bytes of the resident hot cache this prompt pins: residency less what an eviction can prove
/// it will return. Non-zero means an entry this prompt would restore from is holding memory
/// that neither goes back nor has to be allocated again; the connection thread cannot price
/// how much, so it defers to the inference thread's `WarmPrefix` bill.
pub fn pinnedResidentBytes(bill: AdmissionBill) u64 {
    return bill.evictable -| bill.evictionCredit();
}

/// What the admission guard decided, from one bill so the log line and the arm agree: fits
/// outright, fits once the hot cache gives back what it provably can, or does not fit.
pub const AdmissionVerdict = enum {
    admit,
    evict,
    refuse,

    pub fn name(self: AdmissionVerdict) []const u8 {
        return switch (self) {
            .admit => "admit",
            .evict => "evict",
            .refuse => "refuse",
        };
    }
};

pub fn admissionVerdict(bill: AdmissionBill) AdmissionVerdict {
    if (bill.fits()) return .admit;
    if (bill.fitsAfterEviction()) return .evict;
    return .refuse;
}

/// The generation of `markReadyLocked` the last info-level admission line was emitted for.
var admission_logged_generation = std.atomic.Value(u64).init(std.math.maxInt(u64));

/// Has a model become resident since the last time this returned true? Consuming, so N racing connection threads produce one line.
fn firstRequestAfterLoad() bool {
    const gen = model_registry_mod.load_generation.load(.monotonic);
    const last = admission_logged_generation.load(.monotonic);
    if (gen == last) return false;
    return admission_logged_generation.cmpxchgStrong(last, gen, .monotonic, .monotonic) == null;
}

/// The level one admission line is emitted at: info on the first request after a load and
/// whenever the bill is within 10% of what is available; debug otherwise.
pub fn admissionLogLevel(needed: u64, available: u64, first_after_load: bool) log.Level {
    if (first_after_load) return .info;
    // `needed > 0.9 * available` without floating point or overflow.
    if (needed *| 10 > available *| 9) return .info;
    return .debug;
}

/// One line per request at the admission decision, on admitted requests too. Every field comes
/// off the same `bill` the arms act on. The level check runs first.
fn logAdmissionDecision(bill: AdmissionBill) void {
    // `info` is the loudest level this line is emitted at, and `firstRequestAfterLoad` consumes its token.
    if (!log.enabled(.info)) return;
    const level = admissionLogLevel(bill.needed, bill.available, firstRequestAfterLoad());
    if (!log.enabled(level)) return;
    const mb = 1024 * 1024;
    log.atLevel(level, "[admission] needed={d} MB available={d} MB reclaimable={d} MB width={d} verdict={s}\n", .{
        bill.needed / mb,
        bill.available / mb,
        bill.evictionCredit() / mb,
        bill.chunk,
        admissionVerdict(bill).name(),
    });
}

/// Inference-thread predicate behind the scheduler's admission hook: does this request fit
/// right now, with live memory re-read? Every input is forwarded, none defaulted (a hardcoded
/// `kv_override = null` priced a `kv_quant: 4` request at fp16 and evicted a cache for nothing).
pub fn prefillFitsNow(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_cfg: transformer_mod.KVQuantConfig, unchunked_prefill: bool, warm_matched: u64, warm_capacity: u64, warm_will_donate: bool, enable_mtp: bool, checkpoints: scheduler_mod.CheckpointPlan) bool {
    return prefillAdmissionBill(config, prompt_len, max_tokens, kv_cfg, unchunked_prefill, null, .{
        .matched_tokens = warm_matched,
        .capacity_tokens = warm_capacity,
        .will_donate = warm_will_donate,
        .mtp_on = enable_mtp,
        .checkpoints = checkpoints.keep,
        .released = checkpoints.released,
        .thought = checkpoints.thought,
    }).fits();
}

/// The cold bill, live memory re-read, for the scheduler's sibling hold.
pub fn prefillBillNumbersNow(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_cfg: transformer_mod.KVQuantConfig, unchunked_prefill: bool, enable_mtp: bool) scheduler_mod.MemoryBill {
    const bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_cfg, unchunked_prefill, null, .{ .mtp_on = enable_mtp });
    const kv_bits: u64 = if (kv_cfg.scheme == .off) 16 else kv_cfg.bits;
    return .{ .needed = bill.needed, .available = bill.available, .commit = admissionCommitBytes(config, prompt_len, max_tokens, kv_bits) };
}

/// Bytes the bill prices for a request's cache that it allocates AFTER admission. A native GLM
/// MLA cache grows row by row (a DFlash2 request reserves it after prefill); a Qwen or MiMo
/// request takes its reserved capacity at its first grow, so it is resident before the next admission.
pub fn admissionCommitBytes(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_bits: u64) u64 {
    if (!config.isGlm5()) return 0;
    return glmCacheBytes(config, prompt_len, max_tokens, kv_bits) +| slotRingBytes(config, kv_bits);
}

/// The inference thread's refusal, quoting the numbers it compared.
pub fn logPrefillRefusal(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_cfg: transformer_mod.KVQuantConfig, unchunked_prefill: bool, warm_matched: u64, warm_capacity: u64, warm_will_donate: bool, enable_mtp: bool, checkpoints: scheduler_mod.CheckpointPlan) ?model_registry_mod.MemoryContextRefusal {
    // The same warm inputs the probe was refused on, the checkout decision included.
    const warm = WarmPrefix{
        .matched_tokens = warm_matched,
        .capacity_tokens = warm_capacity,
        .will_donate = warm_will_donate,
        .mtp_on = enable_mtp,
        .checkpoints = checkpoints.keep,
        .released = checkpoints.released,
        .thought = checkpoints.thought,
    };
    const bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_cfg, unchunked_prefill, null, warm);
    const mb = 1024 * 1024;
    // "Narrowest width tried" is only true where the ladder was walked.
    const width_note: []const u8 = if (perRequestPrefillChunkEnabled(config)) "the narrowest width tried" else "the pinned width";
    // The comparison first (`fitsAfterEviction` spelled out), then the residency as context, with
    // the part eviction cannot take named.
    log.warn("  prompt {d} tokens needs ~{d}MB at prefill chunk {d} ({s}), ~{d}MB available + ~{d}MB the hot cache can give back (~{d}MB resident, ~{d}MB of it pinned by this request's own prefix) — refused before prefill\n", .{
        prompt_len,
        bill.needed / mb,
        bill.chunk,
        width_note,
        bill.available / mb,
        bill.evictionCredit() / mb,
        bill.evictable / mb,
        pinnedResidentBytes(bill) / mb,
    });
    const context = requestContextRefusal(config, prompt_len, max_tokens, kv_cfg, bill, warm);
    if (context) |r| log.warn("  maximum context at KV{d}, prefill chunk {d}: {d} tokens\n", .{ r.kv_bits, r.chunk, r.maximum });
    return context;
}

/// `fitAfterUnloadingIdle`'s context for one request: each unload re-bills the same prompt.
const IdleUnloadFit = struct {
    sch: *scheduler_mod.Scheduler,
    keep_id: []const u8,
    bill: *AdmissionBill,
    config: *const model_mod.ModelConfig,
    prompt_ids: []const u32,
    max_tokens: u32,
    kv_override: ?transformer_mod.KVQuantConfig,
    unchunked_prefill: bool,
    enable_mtp: bool,

    fn fits(self: *IdleUnloadFit) bool {
        return self.bill.fitsAfterEviction();
    }

    fn unloadOne(self: *IdleUnloadFit) bool {
        const gone = self.sch.unloadLruIdleModel(self.keep_id) orelse return false;
        log.info("  prompt {d} tokens does not fit: unloaded idle model {s} ({d} MB)\n", .{ self.prompt_ids.len, gone.id, gone.bytes / (1024 * 1024) });
        self.bill.* = prefillAdmissionBill(self.config, self.prompt_ids.len, self.max_tokens, self.kv_override, self.unchunked_prefill, self.prompt_ids, .{ .mtp_on = self.enable_mtp });
        return true;
    }
};

pub fn checkAttentionMemory(allocator: std.mem.Allocator, stream: *Conn, prompt_ids: []const u32, max_tokens: u32, config: *const model_mod.ModelConfig, keep_id: []const u8, is_anthropic: bool, kv_override: ?transformer_mod.KVQuantConfig, unchunked_prefill: bool, enable_mtp: bool) !bool {
    if (config.isGlm5() and (kv_override orelse configuredKvQuantFor(config)).glmLatentBits() == null) {
        const msg = "GLM native MLA cache supports kv_quant 8, 6 or off.";
        if (is_anthropic) {
            try sendAnthropicError(allocator, stream, "invalid_request_error", msg, 400);
        } else {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, null);
        }
        return false;
    }
    const prompt_len: usize = prompt_ids.len;
    if (config.num_attention_heads == 0) return true; // unknown architecture, skip check
    // The connection thread has no slot and no cache: it bills cold and defers a warm prompt.
    var bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_override, unchunked_prefill, prompt_ids, .{ .mtp_on = enable_mtp });
    logAdmissionDecision(bill);
    const needed = bill.needed;
    const available = bill.available;
    // Evict to admit (#353): when the bill only fails because the cache is holding memory, admit;
    // the inference thread evicts LRU entries until it fits and refuses by name only if it still does not.
    if (needed > available and bill.fitsAfterEviction()) {
        log.info("  prompt {d} tokens needs ~{d}MB, ~{d}MB available + ~{d}MB reclaimable hot cache (of ~{d}MB resident) — admitting, the prefill will evict\n", .{
            prompt_len, needed / (1024 * 1024), available / (1024 * 1024), bill.evictionCredit() / (1024 * 1024), bill.evictable / (1024 * 1024),
        });
        return true;
    }
    // Warm deferral. The bill above is cold; a non-zero `pinnedResidentBytes` means an entry
    // this prompt would restore from is holding bytes that will neither be allocated nor
    // reclaimed. How many is knowable only on the inference thread, which bills with `WarmPrefix`.
    if (needed > available and pinnedResidentBytes(bill) > 0) {
        log.info("  prompt {d} tokens needs ~{d}MB cold, ~{d}MB available, and ~{d}MB of hot cache is a prefix this prompt would SHARE — deferring to the prefill, which bills the warm request\n", .{
            prompt_len, needed / (1024 * 1024), available / (1024 * 1024), pinnedResidentBytes(bill) / (1024 * 1024),
        });
        return true;
    }
    // Idle resident models are memory too: a connection thread asks the inference thread to unload
    // them, least recently used first, before it refuses.
    if (needed > available) if (global_scheduler) |sch| {
        var unloading = IdleUnloadFit{ .sch = sch, .keep_id = keep_id, .bill = &bill, .config = config, .prompt_ids = prompt_ids, .max_tokens = max_tokens, .kv_override = kv_override, .unchunked_prefill = unchunked_prefill, .enable_mtp = enable_mtp };
        if (fitAfterUnloadingIdle(&unloading, IdleUnloadFit.fits, IdleUnloadFit.unloadOne)) return true;
    };
    if (bill.needed > bill.available) {
        const needed_mb = bill.needed / (1024 * 1024);
        const avail_mb = bill.available / (1024 * 1024);
        log.warn("  prompt {d} tokens needs ~{d}MB (KV+working+margin) at prefill chunk {d}, ~{d}MB available + ~{d}MB reclaimable of ~{d}MB resident hot cache — rejecting\n", .{ prompt_len, needed_mb, bill.chunk, avail_mb, bill.evictionCredit() / (1024 * 1024), bill.evictable / (1024 * 1024) });
        // A refusal quotes the numbers it compared, hot cache included; everything the cache holds
        // is evictable here by construction (the withheld case took the deferral arm). The cache
        // clause is only true where the bill carries the cache; `memoryRefusalMessage` is the one formatter.
        const base = try memoryRefusalMessage(allocator, prompt_len, needed_mb, avail_mb, bill, config.admissionEvictsHotCache());
        defer allocator.free(base);
        const kv = kv_override orelse configuredKvQuantFor(config);
        const refusal = requestContextRefusal(config, prompt_len, max_tokens, kv, bill, .{ .mtp_on = enable_mtp });
        const msg = if (refusal) |r|
            try std.fmt.allocPrint(allocator, "{s} Maximum context at KV{d} and prefill chunk {d}: {d} tokens.", .{ base, r.kv_bits, r.chunk, r.maximum })
        else
            try allocator.dupe(u8, base);
        defer allocator.free(msg);
        countRejected();
        if (is_anthropic) {
            try sendAnthropicError(allocator, stream, "invalid_request_error", msg, 400);
        } else {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, 400);
        }
        return false;
    }
    return true;
}

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// Get the Metal max buffer allocation limit (~75% of system unified memory).
fn getMetalBufferLimit() u64 {
    var mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    _ = sysctlbyname("hw.memsize", @ptrCast(&mem), &len, null, 0);
    if (mem == 0) return 8 * 1024 * 1024 * 1024; // fallback 8GB
    return mem * 75 / 100;
}

/// Facts about a request's target model that decide whether a TEXT-
/// GENERATION route (/v1/chat/completions, /v1/completions, /v1/messages,
/// /v1/responses + WS) may serve it. Extracted
/// from a LoadedModel by `textGenTargetOf`; kept as plain facts so the
/// decision is hermetically testable.
const TextGenTarget = struct {
    /// A text-capable LM is resident (transformer) —
    /// or the entry isn't loaded yet, in which case stubs default to
    /// "assume text until the arch hint or a load says otherwise".
    has_text_lm: bool = true,
};

/// Reason a text-generation route must reject this model with a 400, or null
/// when it can serve text: a request that reached prefill with no language
/// model resident would dereference a null transformer.
fn textGenRejectReason(t: TextGenTarget) ?[]const u8 {
    if (!t.has_text_lm) return "This model cannot serve text generation (no language model resident).";
    return null;
}

fn textGenTargetOf(lm: *LoadedModel) TextGenTarget {
    return .{
        .has_text_lm = lm.state != .ready or lm.transformer != null,
    };
}

pub fn modelQuantizationLabel(allocator: std.mem.Allocator, config: *const model_mod.ModelConfig) ![]u8 {
    if (config.expert_layout == .exl3_k4) {
        var rate_buf: [8]u8 = undefined;
        if (config.glm_fp8_trunk) return std.fmt.allocPrint(allocator, "EXL3 {s}bpw experts, raw FP8 E4M3FN block128 trunk", .{config.expert_quant_rate.kText(&rate_buf)});
        return std.fmt.allocPrint(allocator, "EXL3 {s}bpw experts, {d}-bit dense", .{ config.expert_quant_rate.kText(&rate_buf), config.quant_bits });
    }
    return std.fmt.allocPrint(allocator, "{d}-bit", .{config.quant_bits});
}

/// Plan 05 Phase E: `POST /v1/load-model`. Renders a status payload for the
/// resolved-and-loaded `lm` (the dispatcher already called
/// `scheduler.ensureLoaded` against the body's `model` field, so the load
/// has actually happened by the time we get here). Returns the single
/// loaded entry with resident bytes + state so clients can confirm.
/// Plan 05 Phase E: explicit cold-load. Strict on unknown ids (404 — the
/// caller asked for a specific id, not "anything"). Routes through
/// scheduler.ensureLoaded so eviction/back-pressure kicks in.
fn handleLoadModelStrict(allocator: std.mem.Allocator, stream: *Conn, request_body: []const u8) !void {
    const scheduler = global_scheduler orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    // Parse the model field with std.json (NOT the quick scanner): clients
    // like Swift's JSONSerialization legally escape '/' as '\/', and the raw
    // scanner slice would read "\/Users\/…" — missing the absolute-path
    // branch below and 404ing on the mangled id (live failure 2026-06-12).
    var requested_id: []const u8 = "";
    // `"default": true` — promote the loaded model to the server default
    // (requests that omit `model` / the "sushi" alias, and /v1/models'
    // default-first sort). Explicit opt-in: the app's model SWITCH sends it;
    // media-gen side-loads must never steal the chat default.
    var make_default = false;
    var parsed_body: ?std.json.Parsed(std.json.Value) = null;
    defer if (parsed_body) |*p| p.deinit();
    if (std.json.parseFromSlice(std.json.Value, allocator, request_body, .{})) |parsed| {
        parsed_body = parsed;
        if (parsed.value == .object) {
            if (parsed.value.object.get("model")) |m| {
                if (m == .string) requested_id = m.string;
            }
            if (parsed.value.object.get("default")) |d| {
                if (d == .bool) make_default = d.bool;
            }
        }
    } else |_| {
        requested_id = parseModelFromBody(request_body) orelse "";
    }
    // Register-by-path: an absolute path to a model directory OUTSIDE the
    // --model-dir scan (e.g. a pack in the app's download root).
    // The dir is validated exactly like discovery (config.json, supported
    // model_type + quant mode) before anything is registered; the load then
    // proceeds under the directory-basename id.
    if (std.mem.startsWith(u8, requested_id, "/")) {
        const registry = global_registry orelse {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503);
            return;
        };
        requested_id = registry.registerByPath(stream.io, requested_id) catch |err| switch (err) {
            error.ModelDirNotFound, error.InvalidModelPath => {
                try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "No loadable model directory at that path", 404);
                return;
            },
            error.UnsupportedArch => {
                try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Model at that path has an unsupported model_type", 400);
                return;
            },
            error.UnsupportedQuantMode => {
                try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Model at that path has an unsupported quantization mode", 400);
                return;
            },
            else => return err,
        };
    }
    const lm = scheduler.ensureLoaded(requested_id) catch |err| switch (err) {
        error.UnknownModelId => {
            try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "Unknown model id", 404);
            return;
        },
        error.NotLoaded => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "model_not_loaded", "Requested model is not currently loaded", 503);
            return;
        },
        error.NoDefaultModel => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "no_model", "No default model configured", 503);
            return;
        },
        error.LoadFailed => {
            try sendLoadFailedResponse(allocator, stream, scheduler, requested_id);
            return;
        },
        else => {
            if (try sendNamedLoadRefusal(allocator, stream, scheduler, requested_id, err)) return;
            log.warn("  -> 500 ({s}) on /v1/load-model\n", .{@errorName(err)});
            const msg = std.fmt.allocPrint(allocator, "Failed to load model: {s}", .{@errorName(err)}) catch {
                try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", "Failed to load model", 500);
                return;
            };
            defer allocator.free(msg);
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", msg, 500);
            return;
        },
    };
    defer scheduler.release(lm);
    // Promotion happens only past ensureLoaded's error switch: a load that
    // FAILED must never steal the default from a model that answers.
    if (make_default) {
        if (global_registry) |registry| {
            registry.setDefault(lm.id) catch {};
            log.info("[registry] default model -> {s} (load-model request)\n", .{lm.id});
        }
    }
    const config = lm.config orelse {
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_not_ready", "Loaded model has no parsed config", 500);
        return;
    };
    const model_id: []const u8 = if (lm.id.len > 0) lm.id else config.model_type;
    const bytes_resident = lm.bytes_resident;
    const bytes_on_disk_str = if (lm.bytes_on_disk) |b|
        try std.fmt.allocPrint(allocator, "{d}", .{b})
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(bytes_on_disk_str);
    const drafter_loaded = lm.dflash != null;
    const drafter_path_json = if (drafter_loaded)
        try jsonEscape(allocator, lm.drafter_path)
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(drafter_path_json);

    const body = try std.fmt.allocPrint(allocator,
        \\{{"model":{{"id":"{s}","object":"model","loaded":true,"state":"ready","bytes_resident":{d},"bytes_on_disk":{s},"drafter_loaded":{s},"drafter_path":{s}}}}}
    , .{
        model_id,
        bytes_resident,
        bytes_on_disk_str,
        if (drafter_loaded) "true" else "false",
        drafter_path_json,
    });
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// Does an unload body carry keys but no usable `"model"`? `{"model_id": id}`
/// and `{"id": id}` parse fine, leave the id empty, and so resolve to the
/// DEFAULT model — which, when that default is already unloaded, answers 200
/// with `{"id":"sushi","state":"unloaded"}`. That is a success-shaped
/// payload for an unload that never happened, and no client can tell it apart
/// from a real one (and on a server whose default IS resident, it unloads the
/// wrong model). Same class as the context-overflow 400: an unload that did
/// not do what was asked has to say so.
///
/// An EMPTY object (or no body at all) is left alone — "the default model" is
/// the documented shorthand this endpoint has always taken.
fn unloadBodyNamesNoModel(obj: std.json.ObjectMap) bool {
    if (obj.count() == 0) return false;
    if (obj.get("model")) |m| return m != .string;
    return true;
}

test "an unload body naming an unrecognised key is a 400, not the default model" {
    const Case = struct { body: []const u8, rejected: bool };
    for ([_]Case{
        .{ .body = "{\"model\":\"org/repo\"}", .rejected = false },
        .{ .body = "{\"model\":\"/abs/path/org-repo\"}", .rejected = false },
        // The shorthand every other endpoint honours must keep working.
        .{ .body = "{}", .rejected = false },
        // The live bug: 200 + "sushi" with the named model still resident.
        .{ .body = "{\"model_id\":\"org/repo\"}", .rejected = true },
        .{ .body = "{\"id\":\"org/repo\"}", .rejected = true },
        // A present-but-unusable `model` took the same silent path.
        .{ .body = "{\"model\":123}", .rejected = true },
        .{ .body = "{\"model\":null}", .rejected = true },
    }) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.body, .{});
        defer parsed.deinit();
        try testing.expectEqual(c.rejected, unloadBodyNamesNoModel(parsed.value.object));
    }
}

/// `POST /v1/unload-model` `{"model": id}`. Frees the model's resident GPU
/// state (the stub stays registered so it can reload). Idempotent. Returns a
/// small status payload confirming the model is unloaded.
fn handleUnloadModelStrict(allocator: std.mem.Allocator, stream: *Conn, request_body: []const u8) !void {
    const scheduler = global_scheduler orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    // Parse the model id with std.json (handles '\/'-escaped absolute paths
    // the same way /v1/load-model does), then reduce an absolute path to its
    // basename — the registry keys media + by-path models by dir basename.
    var requested_id: []const u8 = "";
    var parsed_body: ?std.json.Parsed(std.json.Value) = null;
    defer if (parsed_body) |*p| p.deinit();
    if (std.json.parseFromSlice(std.json.Value, allocator, request_body, .{})) |parsed| {
        parsed_body = parsed;
        if (parsed.value == .object) {
            if (unloadBodyNamesNoModel(parsed.value.object)) {
                try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Unload requires the model id as a string under the \"model\" key", 400);
                return;
            }
            if (parsed.value.object.get("model")) |m| {
                if (m == .string) requested_id = m.string;
            }
        }
    } else |_| {
        requested_id = parseModelFromBody(request_body) orelse "";
    }
    if (std.mem.startsWith(u8, requested_id, "/")) {
        var trimmed = requested_id;
        while (trimmed.len > 0 and trimmed[trimmed.len - 1] == '/') trimmed = trimmed[0 .. trimmed.len - 1];
        // A discovered entry is keyed `org/name`, so the path is the only exact handle.
        const by_path = if (global_registry) |r| r.peekByPath(trimmed) else null;
        requested_id = if (by_path) |e| e.id else std.fs.path.basename(trimmed);
    }

    scheduler.unloadModel(requested_id) catch |err| switch (err) {
        error.UnknownModelId => {
            try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "Unknown model id", 404);
            return;
        },
        error.NoDefaultModel => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "no_model", "No default model configured", 503);
            return;
        },
        error.Shutdown => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "shutting_down", "Server is shutting down", 503);
            return;
        },
        else => {
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_unload_failed", "Model unload failed", 500);
            return;
        },
    };

    const id_for_body = if (requested_id.len > 0) requested_id else "sushi";
    const body = try std.fmt.allocPrint(allocator,
        \\{{"model":{{"id":"{s}","object":"model","loaded":false,"state":"unloaded"}}}}
    , .{id_for_body});
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// The model-level half of `Scheduler.batchVerdict`: does this loaded model
/// batch decode at all? Per-slot arms (spec, grammar, logprobs) come later.
pub fn batchVerdictFor(entry: *const LoadedModel) scheduler_mod.BatchVerdict {
    const exclusive = if (entry.transformer) |t| t.ownsModuleDecodeState() else false;
    return batchVerdictOf(entry.config, exclusive);
}

pub fn batchVerdictOf(cfg: ?*const model_mod.ModelConfig, exclusive: bool) scheduler_mod.BatchVerdict {
    if (exclusive) return .exclusive;
    const c = cfg orelse return .arch;
    return if (scheduler_mod.configBatchesDecode(c)) .ok else .arch;
}

/// The boot line's account of concurrent requests, from the verdict `/props` reports.
pub fn concurrencyClause(verdict: scheduler_mod.BatchVerdict) []const u8 {
    return switch (verdict) {
        .ok => "batched decode on",
        .exclusive => "batched decode off; concurrent requests queue, one at a time",
        else => "batched decode off; concurrent requests interleave serially",
    };
}

/// A request decoding or queued, or a model loading or evicting: an update now would cut it off.
fn busyForUpdate(io: std.Io) bool {
    if (global_scheduler) |sch| {
        sch.queue_mu.lockUncancelable(sch.io);
        const in_flight = sch.in_flight;
        sch.queue_mu.unlock(sch.io);
        if (in_flight > 0) return true;
    }
    const reg = global_registry orelse return false;
    reg.mutex.lockUncancelable(io);
    defer reg.mutex.unlock(io);
    var it = reg.entries.valueIterator();
    while (it.next()) |e| if (e.*.state == .loading or e.*.state == .evicting) return true;
    return false;
}

/// `POST /v1/update`: `update.guard` decides. On accept the answer is 202 and the server shuts down through its
/// SIGTERM path; main then replaces the process with the updater (`update.relaunchIfRequested`).
/// Turns metrics on for the running server; 200 whether this call did it or it was already on. Counters start
/// from zero at the moment of the first enable.
fn handleMetricsEnable(stream: *Conn) !void {
    const changed = instr.process.enable();
    startGaugeSampler() catch |err| {
        log.err("metrics gauge sampler failed to start: {s}\n", .{@errorName(err)});
        return sendResponse(stream, "500 Internal Server Error", "application/json", "{\"error\":\"metrics enabled but the gauge sampler did not start; retry\"}");
    };
    if (changed) log.info("Prometheus metrics: ENABLED at runtime (POST /metrics/enable)\n", .{});
    try sendResponse(stream, "200 OK", "application/json", if (changed) "{\"metrics\":true,\"already_enabled\":false}" else "{\"metrics\":true,\"already_enabled\":true}");
}

fn handleUpdate(allocator: std.mem.Allocator, stream: *Conn, headers: []const u8, raw_path: []const u8) !void {
    var ebuf: [std.fs.max_path_bytes]u8 = undefined;
    var rbuf: [512]u8 = undefined;
    var vbuf: [64]u8 = undefined;
    const latest = update_mod.availableVersion(&vbuf);
    const ask: update_mod.Ask = .{
        .bind_host = listen_host,
        .port = listen_port,
        .peer_loopback = peerIsLoopback(stream),
        .origin = findHeaderValueCI(headers, "origin"),
        .key_set = g_api_key != null,
        .key_ok = apiKeyAuthorized(headers, raw_path),
        .host_managed = host_managed,
        .busy = busyForUpdate(stream.io),
        .install = update_mod.selfInstallRefusal(stream.io, &ebuf, &rbuf),
        .available = latest != null,
    };
    if (update_mod.guard(ask)) |r| {
        log.info("[update] POST /v1/update -> {d}: {s}\n", .{ r.code, r.message });
        if (r.code == 401) return sendUnauthorized(stream);
        return sendErrorResponse(allocator, stream, r.status, r.kind, r.message, r.code);
    }
    const body = try std.fmt.allocPrint(allocator, "{{\"status\":\"updating\",\"from\":\"{s}\",\"to\":\"{s}\"}}", .{ update_mod.version, latest.? });
    defer allocator.free(body);
    try sendResponse(stream, "202 Accepted", "application/json", body);
    log.info("[update] updating {s} -> {s} from the chat page; the server restarts\n", .{ update_mod.version, latest.? });
    update_mod.relaunch_requested.store(true, .release);
    shutdown_requested.store(true, .release);
}

fn handleTokenize(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    const tok = lm.tokenizer.?;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON", 400);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;

    const content = if (root.get("content")) |v| (if (v == .string) v.string else null) else null;
    if (content == null) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'content' is required", 400);
        return;
    }

    const ids = try tok.encode(allocator, content.?);
    defer allocator.free(ids);

    var result = std.ArrayList(u8).empty;
    defer result.deinit(allocator);
    try result.appendSlice(allocator, "{\"tokens\":[");
    for (ids, 0..) |id, i| {
        if (i > 0) try result.append(allocator, ',');
        var num_buf: [12]u8 = undefined;
        const num = std.fmt.bufPrint(&num_buf, "{d}", .{id}) catch continue;
        try result.appendSlice(allocator, num);
    }
    try result.appendSlice(allocator, "]}");

    log.debug("POST /tokenize -> {d} tokens\n", .{ids.len});
    try sendResponse(stream, "200 OK", "application/json", result.items);
}

fn handleDetokenize(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    const tok = lm.tokenizer.?;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON", 400);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;

    const tokens_val = root.get("tokens") orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'tokens' is required", 400);
        return;
    };
    if (tokens_val != .array) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'tokens' must be an array", 400);
        return;
    }

    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(allocator);
    for (tokens_val.array.items) |item| {
        if (item == .integer) try ids.append(allocator, @intCast(item.integer));
    }

    const text = try tok.decode(allocator, ids.items);
    defer allocator.free(text);

    const result = try detokenizeResponseJson(allocator, text);
    defer allocator.free(result);

    log.debug("POST /detokenize -> {d} tokens -> {d} chars\n", .{ ids.items.len, text.len });
    try sendResponse(stream, "200 OK", "application/json", result);
}

/// `{"content": <text>}` with the text escaped by the SHARED escaper.
///
/// This handler used to hand-roll a five-character escape table and pass every
/// other byte through raw, so any token whose bytes are below 0x20 produced a
/// body no JSON parser accepts — found live 2026-08-04 detokenizing single ids
/// while cross-checking a vocabulary. Same class as the control-byte rule on
/// the chat render path: a decoded token is arbitrary bytes, so it goes through
/// `appendJsonString`, never a local switch.
fn detokenizeResponseJson(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);
    try result.appendSlice(allocator, "{\"content\":");
    try chat_mod.appendJsonString(allocator, &result, text);
    try result.append(allocator, '}');
    return result.toOwnedSlice(allocator);
}

/// OpenAI `n` (choice count): this is a single-choice engine and n>1 is
/// deliberately unimplemented (YAGNI — no agent client sends it). Silently
/// returning one choice for n=2 is the silent-no-op class llmprobe flags;
/// reject with an honest 400 instead. Absent, null, 1, or 1.0 all mean
/// "one choice" and pass.
/// jsonEscape with an OOM fallback to the literal `""`. Ownership is reported
/// by PROVENANCE (`owned`), never inferred from content — escaping an empty
/// string also yields `""` but that one is allocated (2-byte leak per
/// all-reasoning request under the old content-equality defer).
const EscapedText = struct { slice: []const u8, owned: bool };
pub fn jsonEscapeOrEmpty(allocator: std.mem.Allocator, text: []const u8) EscapedText {
    const s = jsonEscape(allocator, text) catch return .{ .slice = "\"\"", .owned = false };
    return .{ .slice = s, .owned = true };
}

/// `continue_final_message`: extend the trailing assistant message instead of
/// answering after it. vLLM's spelling, because a client that already knows how
/// to ask another local server for this should not have to learn a second name.
///
/// The flag alone is not enough — the conversation has to BE continuable
/// (`chat_mod.continuationRequested`). A flag set on a user-final request is
/// ignored rather than 400'd: the request is perfectly answerable as an
/// ordinary turn, and refusing it would break the obvious client that sets the
/// field once for the whole session.
fn continueFinalMessageRequested(root: std.json.ObjectMap, messages: []const chat_mod.Message) bool {
    const v = root.get("continue_final_message") orelse return false;
    if (v != .bool or !v.bool) return false;
    return chat_mod.continuationRequested(messages);
}

const JoinedText = struct { text: []const u8, owned: bool };

/// Collect every `{"type":"text"}` part of a content array into one string,
/// in order, '\n'-joined. A single (or no) text part borrows the JSON's
/// bytes (`owned=false`); 2+ parts allocate (`owned=true`, caller frees).
///
/// Multiple text parts are real traffic: opencode plan mode appends its
/// "Plan Mode - System Reminder" as a SECOND text part on the user message,
/// and last-wins here dropped the user's actual prompt — the model answered
/// "What would you like to accomplish?" to every first plan-mode message
/// (issue #195).
pub fn joinedTextParts(allocator: std.mem.Allocator, parts: []const std.json.Value) !JoinedText {
    var single: []const u8 = "";
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    var n: usize = 0;
    for (parts) |part| {
        if (part != .object) continue;
        const ptype = part.object.get("type") orelse continue;
        if (ptype != .string or !std.mem.eql(u8, ptype.string, "text")) continue;
        const tv = part.object.get("text") orelse continue;
        if (tv != .string or tv.string.len == 0) continue;
        n += 1;
        if (n == 1) {
            single = tv.string;
            continue;
        }
        if (n == 2) try buf.appendSlice(allocator, single);
        try buf.append(allocator, '\n');
        try buf.appendSlice(allocator, tv.string);
    }
    if (n <= 1) return .{ .text = single, .owned = false };
    return .{ .text = try buf.toOwnedSlice(allocator), .owned = true };
}

/// Workload key for hot-cache eviction (#378): `prompt_cache_key` (OpenAI's own
/// routing field) > `metadata.user_id` (Anthropic; Claude Code sends its session
/// id) > the system prompt (OpenAI `messages[0]`, Anthropic `system`, Responses
/// `instructions`) > 0 = anonymous. Text parts hash in order like `joinedTextParts`.
pub fn requestCacheKey(root: std.json.ObjectMap) u64 {
    if (root.get("prompt_cache_key")) |v| if (v == .string and v.string.len > 0)
        return std.hash.Wyhash.hash(1, v.string);
    if (root.get("metadata")) |mv| if (mv == .object) {
        if (mv.object.get("user_id")) |v| if (v == .string and v.string.len > 0)
            return std.hash.Wyhash.hash(2, v.string);
    };
    if (root.get("messages")) |mv| if (mv == .array and mv.array.items.len > 0) {
        const first = mv.array.items[0];
        if (first == .object) if (first.object.get("role")) |r| if (r == .string and std.mem.eql(u8, chat_mod.canonicalRole(r.string), "system")) {
            if (first.object.get("content")) |c| return hashTextValue(c) orelse 0;
        };
    };
    if (root.get("system")) |v| return hashTextValue(v) orelse 0;
    if (root.get("instructions")) |v| return hashTextValue(v) orelse 0;
    return 0;
}

fn hashTextValue(v: std.json.Value) ?u64 {
    var h = std.hash.Wyhash.init(3);
    var n: usize = 0;
    switch (v) {
        .string => |t| {
            if (t.len == 0) return null;
            h.update(t);
            n = 1;
        },
        .array => |parts| for (parts.items) |part| {
            if (part != .object) continue;
            const ptype = part.object.get("type") orelse continue;
            if (ptype != .string or !std.mem.eql(u8, ptype.string, "text")) continue;
            const tv = part.object.get("text") orelse continue;
            if (tv != .string or tv.string.len == 0) continue;
            if (n > 0) h.update("\n");
            h.update(tv.string);
            n += 1;
        },
        else => {},
    }
    return if (n == 0) null else h.final();
}

pub fn nChoicesRejectReason(root: std.json.ObjectMap) ?[]const u8 {
    const v = root.get("n") orelse return null;
    switch (v) {
        .null => return null,
        .integer => |i| if (i == 1) return null,
        .float => |f| if (f == 1.0) return null,
        else => {},
    }
    return "'n' > 1 is not supported: this server returns a single choice per request; omit 'n' or set it to 1";
}

pub fn handleChatCompletions(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    const tok = lm.tokenizer.?;
    const chat_config = lm.chat_config.?;
    const config = lm.config.?;
    // Parse JSON body
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/chat/completions -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();

    // A valid-JSON body that isn't an object (e.g. `42` or `[1,2]`) would panic
    // on the `.object` field access below and take the whole server down, so
    // reject it as a 400 like any other malformed request.
    if (parsed.value != .object) {
        log.warn("POST /v1/chat/completions -> 400 (body is not a JSON object)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;

    if (nChoicesRejectReason(root)) |reason| {
        log.warn("POST /v1/chat/completions -> 400 (unsupported n)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
        return;
    }
    if (chatIgnoreEosRejectReason(root)) |reason| {
        log.warn("POST /v1/chat/completions -> 400 (ignore_eos)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
        return;
    }

    // Extract messages
    const messages_val = root.get("messages") orelse {
        log.warn("POST /v1/chat/completions -> 400 (missing messages)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'messages' is a required field", 400);
        return;
    };

    // `messages` present but not an array (e.g. a string) would panic on the
    // `.array` access below. Reject rather than crash.
    if (messages_val != .array) {
        log.warn("POST /v1/chat/completions -> 400 (messages is not an array)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'messages' must be an array", 400);
        return;
    }

    var wire = WireMessages.init(allocator);
    defer wire.deinit();
    try readOpenAiMessages(&wire, messages_val.array.items, visionPreprocFromConfig(config));
    const messages = &wire.messages;
    if (wire.media.refusal()) |fault| {
        log.warn("POST /v1/chat/completions -> 400 ({s})\n", .{fault.text()});
        try sendErrorResponse(allocator, stream, fault.status(), fault.errorType(), fault.text(), fault.code());
        return;
    }
    if (messages.items.len == 0) {
        log.warn("POST /v1/chat/completions -> 400 (no valid messages)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "No valid messages found in request", 400);
        return;
    }

    // Support both max_tokens and max_completion_tokens (OpenAI alias). Absent
    // or <= 0 → auto (peg to remaining context).
    const requested_max_tokens = root.get("max_tokens") orelse root.get("max_completion_tokens");
    const max_tokens: u32 = resolveRequestMaxTokens(
        requested_max_tokens,
        omittedMaxTokensDefault(getEffectiveContextLength(config)),
    );
    // For the request log line only: which of the three sources this came from.
    const max_tokens_origin = maxTokensOrigin(requested_max_tokens, launchMaxTokensDefault());

    const is_stream = if (root.get("stream")) |v| v == .bool and v.bool else false;

    const temperature = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "temperature", 0.0, 2.0), server_config.default_temperature, config.gen_temperature, 1.0);
    const top_p = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "top_p", 0.0, 1.0), server_config.default_top_p, config.gen_top_p, 1.0);
    const top_k = resolveSamplingDefault(u32, parseJsonTopKOpt(root, "top_k"), server_config.default_top_k, config.gen_top_k, 0);

    const repeat_penalty = parseRepeatPenalty(root);

    const presence_penalty = parseAdditivePenalty(root, "presence_penalty");
    const frequency_penalty = parseAdditivePenalty(root, "frequency_penalty");

    const seed: ?u64 = parseRequestSeed(root.get("seed"));

    // Parse logprobs: "logprobs": true, "top_logprobs": N (0-20)
    const logprobs_n: u32 = blk: {
        const lp = root.get("logprobs") orelse break :blk 0;
        if (lp != .bool or !lp.bool) break :blk 0;
        // logprobs=true without top_logprobs defaults to 0 (just the chosen token's logprob)
        const tlp = root.get("top_logprobs") orelse break :blk 1;
        break :blk switch (tlp) {
            .integer => |i| @intCast(@min(@max(i, 0), 20)),
            else => 1,
        };
    };

    // Extract tools JSON from request body for chat template injection
    var tools_json: ?[]const u8 = null;
    var has_tools = root.get("tools") != null;
    const tool_choice = chat_mod.parseToolChoice(root.get("tool_choice"));
    if (tool_choice == .none) has_tools = false;
    const tool_choice_instruction: ?[]const u8 = if (has_tools) try tool_choice.instruction(allocator) else null;
    defer if (tool_choice_instruction) |tci| allocator.free(tci);

    // OpenAI parallel_tool_calls: only an explicit false clamps to one call
    // per response (the SDK sets false in strict structured-output mode).
    const allow_parallel_tools = if (root.get("parallel_tool_calls")) |v|
        !(v == .bool and !v.bool)
    else
        true;

    if (has_tools) {
        // Find the tools array in the raw JSON body and extract it
        if (extractJsonField(body, "tools")) |tools_str| {
            tools_json = tools_str;
        }
        if (chat_mod.toolChoiceNamesUndeclared(allocator, tool_choice, tools_json)) {
            log.warn("POST /v1/chat/completions -> 400 (tool_choice names an undeclared function)\n", .{});
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", TOOL_CHOICE_UNDECLARED, 400);
            return;
        }
    }

    // Parse stop sequences
    var stop_sequences = std.ArrayList([]const u8).empty;
    defer stop_sequences.deinit(allocator);
    if (root.get("stop")) |stop_val| {
        switch (stop_val) {
            .string => |s| if (s.len > 0) try stop_sequences.append(allocator, s),
            .array => |arr| {
                for (arr.items) |item| {
                    if (item == .string and item.string.len > 0) try stop_sequences.append(allocator, item.string);
                }
            },
            else => {},
        }
    }

    // Parse model name from request (use for response, fallback to config)
    const model_name = if (root.get("model")) |v|
        (if (v == .string) v.string else config.model_type)
    else
        config.model_type;

    // Track allocations from response_format injection so we can free them
    var rf_allocs = std.ArrayList([]const u8).empty;
    defer {
        for (rf_allocs.items) |a| allocator.free(a);
        rf_allocs.deinit(allocator);
    }

    // Parse response_format — inject JSON schema constraint into system message,
    // and capture the schema value for grammar-constrained sampling below.
    var grammar_schema_val: ?std.json.Value = null;
    // Backing storage for the synthetic `{"type":"object"}` schema used for
    // `response_format: {type: "json_object"}`. Lives for the request and is
    // freed at the bottom; the schema parser arena copies what it needs.
    var json_object_schema_holder: ?std.json.Parsed(std.json.Value) = null;
    defer if (json_object_schema_holder) |*p| p.deinit();
    if (root.get("response_format")) |rf| {
        if (rf == .object) {
            const rf_type = if (rf.object.get("type")) |t| (if (t == .string) t.string else "") else "";
            if (std.mem.eql(u8, rf_type, "json_schema")) {
                const declared: ?std.json.Value = if (rf.object.get("json_schema")) |js| (if (js == .object) js.object.get("schema") else null) else null;
                if (declared == null or declared.? != .object) {
                    try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "response_format.json_schema.schema must be a JSON object (use {\"type\":\"json_object\"} for unconstrained JSON)", 400);
                    return;
                }
                // Extract the schema JSON string from the raw body
                var schema_instruction = std.ArrayList(u8).empty;
                defer schema_instruction.deinit(allocator);
                try schema_instruction.appendSlice(allocator, "Respond with valid JSON only. No other text, no markdown, no explanation. ");

                if (rf.object.get("json_schema")) |js| {
                    if (js == .object) {
                        if (js.object.get("schema")) |schema_val| {
                            grammar_schema_val = schema_val;
                            var out: std.Io.Writer.Allocating = .init(allocator);
                            defer out.deinit();
                            var jws: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
                            schema_val.jsonStringify(&jws) catch {};
                            try schema_instruction.appendSlice(allocator, "Your response MUST conform to this JSON schema:\n");
                            try schema_instruction.appendSlice(allocator, out.written());
                        }
                    }
                }

                const instruction = try allocator.dupe(u8, schema_instruction.items);
                try rf_allocs.append(allocator, instruction);
                if (messages.items.len > 0 and std.mem.eql(u8, messages.items[0].role, "system")) {
                    const combined = try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ messages.items[0].content, instruction });
                    try rf_allocs.append(allocator, combined);
                    messages.items[0].content = combined;
                } else {
                    try messages.insert(allocator, 0, .{ .role = "system", .content = instruction, .tool_calls = null, .tool_call_id = null });
                }
            } else if (std.mem.eql(u8, rf_type, "json_object")) {
                const instruction = "Respond with valid JSON only. No other text, no markdown fences (no ``` or ```json), no explanation. Begin your response with `{` or `[`.";
                if (messages.items.len > 0 and std.mem.eql(u8, messages.items[0].role, "system")) {
                    const combined = try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ messages.items[0].content, instruction });
                    try rf_allocs.append(allocator, combined);
                    messages.items[0].content = combined;
                } else {
                    try messages.insert(allocator, 0, .{ .role = "system", .content = instruction, .tool_calls = null, .tool_call_id = null });
                }
                // Belt + braces: also constrain decoding with a permissive
                // object schema so the very first token cannot be a leading
                // backtick (Gemma 4 ignores the prompt-side instruction
                // otherwise). `additionalProperties: true` allows any keys
                // and values — `json_object` only constrains JSON-ness, not
                // a specific shape.
                if (json_object_schema_holder == null) {
                    json_object_schema_holder = std.json.parseFromSlice(std.json.Value, allocator, "{\"type\":\"object\",\"additionalProperties\":true}", .{}) catch null;
                }
                if (json_object_schema_holder) |*p| {
                    grammar_schema_val = p.value;
                }
            }
        }
    }

    // Parse stream_options
    const include_usage = if (root.get("stream_options")) |so| blk: {
        if (so != .object) break :blk false;
        if (so.object.get("include_usage")) |iu| {
            break :blk iu == .bool and iu.bool;
        }
        break :blk false;
    } else false;

    // Thinking opt-ins: the OpenAI-standard `reasoning_effort` string and the
    // vendor `enable_thinking` bool (top-level, or nested in `chat_template_kwargs`).
    // Either switch turns thinking on; effort "none" alone never does.
    // A request naming NEITHER takes the arch default (off for every arch but
    // the ones whose vendor documents thinking-on).
    const effort_cfg = parseReasoningEffort(root, server_config.default_reasoning_budget, effortWordOnly(allocator, lm, tok), model_mod.effortArms(config.model_type)) catch {
        const msg = try effortRefusalFor(allocator, lm, root.get("reasoning_effort").?.string);
        defer allocator.free(msg);
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, 400);
        return;
    };
    var effort_notice: ?EffortNotice = null;
    defer if (effort_notice) |n| n.deinit(allocator);
    defer clearEffortNotice(stream);
    if (effort_cfg) |e| effort_notice = try holdEffortNotice(allocator, stream, effortModelName(lm), e);
    var enable_thinking = resolveEnableThinking(root, effort_cfg, config.defaultEnableThinking());
    if (!try checkThinkingSupport(allocator, stream, config, enable_thinking, false)) return;

    // Reasoning budget (max tokens in <think> block, -1 = unlimited):
    const default_effort_budget = model_mod.defaultThinkBudget(config, server_config.default_reasoning_budget);
    const effort_budget: i32 = if (effort_cfg) |e| e.budget else default_effort_budget;
    const reasoning_budget: i32 = if (root.get("reasoning_budget_tokens")) |v| switch (v) {
        .integer => |i| clampJsonI32(i),
        else => effort_budget,
    } else effort_budget;

    // Wave 1.A: per-request KV-quant override. When unset, falls back to the
    // process-level --kv-quant default carried on the scheduler. Cross-scheme
    // hot-prefix-cache hits never happen — entries record their scheme and
    // findBestMatch filters on it.
    const kv_quant_override = parseKvQuantOverride(root);

    // Parse enable_drafter: per-request override of the --drafter default.
    // Auto-disabled when:
    //   - the server didn't load a drafter (`default_drafter == null`)
    //   - logprobs are requested (drafter doesn't expose realized log-probs)
    //   - hybrid SSM architecture (the drafter would hit the SSM-state issue on
    //     the verify forward).
    // Tools do NOT disable the drafter.
    // Plan 05: per-model drafter — when a drafter is loaded for this model,
    // requests default to ON unless the target is MoE (where verify-forward
    // routing penalty overwhelms the win). Per-request `enable_drafter` JSON
    // overrides either way.
    const lm_default_enable_drafter: bool = lm.dflash != null;
    var enable_drafter: bool = if (root.get("enable_drafter")) |v|
        (v == .bool and v.bool)
    else
        lm_default_enable_drafter;
    if (enable_drafter and lm.dflash == null) {
        enable_drafter = false; // no drafter loaded; quietly fall through
    }
    if (enable_drafter and logprobs_n > 0) {
        log.info("  drafter=disabled (logprobs requested)\n", .{});
        enable_drafter = false;
    }
    // Qwen native MTP head: defaults ON whenever the model loaded one (the
    // sidecar only loads when it binds to this trunk; MoE mirrors the
    // drafter caution). Priority DFlash > MTP at
    // dispatch (`requestSpecModes`). The trained head holds ~73%
    // per-draft acceptance even on fully novel content.
    const allow_batch_mtp = if (root.get("enable_batch_mtp")) |v| v != .bool or v.bool else true;
    if (expert_stream_mod.mtpRefusal(config.expert_streaming, if (root.get("enable_mtp")) |v| (v == .bool and v.bool) else false, lm.mtp != null)) |why| {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", why, 400);
        return;
    }
    var enable_mtp: bool = if (root.get("enable_mtp")) |v|
        (v == .bool and v.bool)
    else
        defaultEnableMtpForStreaming(lm.mtp != null, config.isMoe(), forceMtpFor(config), nativeMeasuredMoeHead(lm), model_mod.isServedArch(config.model_type), config.expert_streaming and !config.stream_mtp_head);
    if (enable_mtp and lm.mtp == null) enable_mtp = false;
    if (enable_mtp and logprobs_n > 0) {
        log.info("  mtp=disabled (logprobs requested)\n", .{});
        enable_mtp = false;
    }
    if (enable_drafter) {
        log.info("  drafter=enabled (block_size={d})\n", .{lm.drafter_block_size});
    }

    // Log the request
    const last_msg = messages.items[messages.items.len - 1];
    const preview_len = @min(last_msg.content.len, 80);

    // Compute sizes for debug info
    var system_chars: usize = 0;
    var user_chars: usize = 0;
    var tool_msg_count: usize = 0;
    for (messages.items) |msg| {
        if (std.mem.eql(u8, msg.role, "system")) {
            system_chars += msg.content.len;
        } else if (std.mem.eql(u8, msg.role, "user")) {
            user_chars += msg.content.len;
        } else if (std.mem.eql(u8, msg.role, "tool")) {
            tool_msg_count += 1;
        }
    }
    const tools_len = if (tools_json) |tj| tj.len else 0;

    // Format chat template: Jinja render + BPE tokenize, threading
    // `tools_json` + `tool_choice_instruction` through.
    //
    // Phase 4 instrumentation + Iteration 2 cache: time the render+tokenize
    // step.
    // Continuation is EXPLICIT on this surface (`continue_final_message`, the
    // spelling vLLM uses). It cannot be implied by a trailing assistant
    // message the way /v1/messages implies it: agent frameworks legitimately
    // POST assistant-last history here and expect a fresh turn, and turning
    // that into a prefill would change what every one of them gets back.
    const continue_final = continueFinalMessageRequested(root, messages.items);
    var tokenize_sw = Stopwatch.init(stream.io);
    const render_config = renderConfigFor(chat_config, root, model_settings.preserve_thinking_flag, config.preserve_thinking_override);
    var prompt_ids_raw = cachedFormatChat(allocator, stream.io, lm, tok, &render_config, messages.items, tools_json, tool_choice_instruction, enable_thinking, if (effort_cfg) |e| e.effort else model_mod.defaultEffortWord(config), continue_final) catch |err| return sendGenerationError(allocator, stream, err, .openai);
    var schema_proto: rp_mod.Protocol = undefined;
    var schema_proto_active = false;
    if (grammar_schema_val != null and !has_tools and enable_thinking and reasoning_budget < 0) {
        schema_proto_active = resolveReasoningProtocol(allocator, stream.io, lm, tok, chat_config, prompt_ids_raw, &schema_proto, true);
    }
    switch (schemaMasksThinking(
        grammar_schema_val != null,
        has_tools,
        enable_thinking,
        reasoning_budget >= 0,
        schema_proto_active,
    )) {
        .deferred => {},
        .fallback_thinking_off => {
            schema_proto_active = false;
            allocator.free(prompt_ids_raw);
            enable_thinking = false;
            prompt_ids_raw = cachedFormatChat(allocator, stream.io, lm, tok, &render_config, messages.items, tools_json, tool_choice_instruction, false, if (effort_cfg) |e| e.effort else model_mod.defaultEffortWord(config), continue_final) catch |err| return sendGenerationError(allocator, stream, err, .openai);
            log.info("[grammar] {s}; rerendered with thinking off\n", .{if (reasoning_budget >= 0) "finite reasoning budget" else "reasoning protocol unsupported for this model/prompt"});
        },
        .no_mask, .token_zero => schema_proto_active = false,
    }
    if (grammar_schema_val != null and !has_tools and !enable_thinking and lm.transformer != null) {
        if (!resolveReasoningProtocol(allocator, stream.io, lm, tok, chat_config, prompt_ids_raw, &schema_proto, false)) {
            schema_proto = rp_mod.Protocol.jsonOnly();
        }
        schema_proto_active = true;
    }
    const tokenize_ns = tokenize_sw.read();

    var max_tokens_desc_buf: [MAX_TOKENS_DESC_LEN]u8 = undefined;
    log.info("POST /v1/chat/completions ({d} msgs, max_tokens={s}, temp={d:.2}, top_p={d:.2}, top_k={d}, stream={}, thinking={}, sys={d}b, user={d}b, tools={d}b, tool_msgs={d}) \n", .{ messages.items.len, describeMaxTokens(&max_tokens_desc_buf, max_tokens, max_tokens_origin), temperature, top_p, top_k, is_stream, enable_thinking, system_chars, user_chars, tools_len, tool_msg_count });
    log.info("  > \"{s}{s}\"\n", .{ last_msg.content[0..preview_len], if (last_msg.content.len > 80) "..." else "" });

    // Every message's media, placed and encoded; the slot takes the
    // embeddings and M-RoPE table at submit, `deinit` frees what it did not.
    const cache_key = requestCacheKey(root);
    var req_media: PreparedMedia = .{};
    defer req_media.deinit(allocator);
    switch (try prepareRequestMedia(allocator, lm, messages.items, &prompt_ids_raw, getEffectiveContextLength(config))) {
        .ready => |r| req_media = r,
        .refused => |fault| {
            allocator.free(prompt_ids_raw);
            log.warn("POST /v1/chat/completions -> {d} ({s})\n", .{ fault.code(), fault.text() });
            try sendErrorResponse(allocator, stream, fault.status(), fault.errorType(), fault.text(), fault.code());
            return;
        },
    }
    const prompt_ids = prompt_ids_raw;
    defer allocator.free(prompt_ids);
    enable_mtp = admitMtpForCtx(enable_mtp, prompt_ids.len);

    // Enforce context size limit
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        log.warn("POST /v1/chat/completions -> 400 (prompt {d} tokens exceeds ctx_size {d})\n", .{ prompt_ids.len, effective_ctx });
        countRejected();
        var ovf_buf: [160]u8 = undefined;
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", contextOverflowMessage(&ovf_buf, prompt_ids.len, effective_ctx), 400);
        return;
    }

    // Clamp max_tokens first: the memory guard bills the reserved generation headroom, and an
    // omitted max_tokens is the maxInt(u32)/4 sentinel.
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len, effective_ctx);

    // Check if attention computation would exceed GPU memory.
    const bill_head = requestBillsMtpHead(enable_drafter, enable_mtp, lm.dflash != null, mtpCapable(lm), .{ .repeat_penalty = repeat_penalty, .presence_penalty = presence_penalty, .frequency_penalty = frequency_penalty }, logprobs_n);
    if (!try checkAttentionMemory(allocator, stream, prompt_ids, effective_max_tokens, config, lm.id, false, kv_quant_override, prefill_plan.visionPrefillUnchunked(req_media.deferred != null), bill_head)) return;

    log.info("  prompt={d} tokens, max_gen={d}, ctx={d}\n", .{ prompt_ids.len, effective_max_tokens, effective_ctx });

    // Prompt caching: reuse KV cache for shared prefix.
    // Force invalidation when images are present — image tokens have identical IDs
    // but different vision embeddings, so prefix matching would reuse stale features.
    const eos_slice = config.eosTokenSlice();

    var sampling = sampler.SamplingParams{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .repeat_penalty = repeat_penalty,
        .presence_penalty = presence_penalty,
        .frequency_penalty = frequency_penalty,
        .seed = seed,
    };

    // Build grammar-constrained sampling state if a JSON schema was supplied.
    // Lifetime is scoped to this request — the SchemaConstraint must NOT be
    // moved (the embedded Constraint holds pointers into it).
    var sc: generate_mod.SchemaConstraint = undefined;
    var sc_init = false;
    defer if (sc_init) sc.deinit();

    if (grammar_schema_val) |sv| {
        if (has_tools) {
            log.info("[grammar] skipped JSON schema mask while tools are available (tool calls must remain reachable)\n", .{});
        } else {
            const tb = try lm.grammarTokenBytes(allocator, stream.io);
            if (sc.initFromValue(allocator, sv, tb)) {
                sc_init = true;
                var deferred = false;
                if (schema_proto_active) {
                    sc.deferWithProtocol(&schema_proto);
                    deferred = schema_proto.startState().phase != .json_body;
                }
                sc.constraint.row_masks = rowMaskedDrafter(lm);
                sampling.constraint = &sc.constraint;
                if (deferred) {
                    log.info("[grammar] deferring JSON schema until the reasoning boundary (vocab={d}, mask={d}b)\n", .{ tb.bytes.len, sc.mask_buf.len });
                } else {
                    log.info("[grammar] enforcing JSON schema from token 0 (vocab={d}, mask={d}b)\n", .{ tb.bytes.len, sc.mask_buf.len });
                }
            } else |err| {
                log.warn("[grammar] schema parse failed ({s}); falling back to prompt-only enforcement\n", .{@errorName(err)});
            }
        }
    }

    // A decode-time bound owns the budget; the surfaces then deliver the
    // whole (closed) thought instead of trimming it.
    sampling.think_penalty = armThinkPenalty(allocator, lm, tok, prompt_ids, enable_thinking, resolveThinkPenalty(root, model_settings.think_penalty_flag, config.think_penalty_override));
    const request_bias: []const @import("logit_bias.zig").Bias = if (root.get("logit_bias")) |bias_value| @import("logit_bias.zig").request(allocator, bias_value, @min(config.vocab_size, tok.definedVocabSize())) catch |err| {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", @errorName(err), null);
        return;
    } else &.{};
    defer allocator.free(request_bias);
    const none_ban = if (tool_choice == .none and lm.transformer != null) noneCallBan(allocator, tok, chat_config) else null;
    const biases = if (none_ban) |ban| try std.mem.concat(allocator, @import("logit_bias.zig").Bias, &.{ request_bias, &.{ban} }) else request_bias;
    defer if (none_ban != null) allocator.free(biases);
    sampling.think_penalty.biases = biases;
    var think_bound = armThinkBound(allocator, lm, tok, prompt_ids, enable_thinking, reasoning_budget);
    defer if (think_bound) |tb| allocator.free(tb.forced);
    const surface_budget: i32 = if (think_bound != null) -1 else reasoning_budget;
    if (think_bound) |*tb| sampling.think_bound = tb;
    var call_force = if (has_tools and lm.transformer != null and !continue_final)
        armCallForce(allocator, tok, chat_config, prompt_ids, enable_thinking, tool_choice, tools_json)
    else
        null;
    defer if (call_force) |*cf| cf.deinit(allocator);
    if (call_force) |*cf| sampling.call_force = cf;

    // Hand vision ownership off to the sub-handler, which transfers it to
    // the slot at submit time.
    const sub_ve = req_media.deferred;
    req_media.deferred = null;
    const sub_mrope = req_media.mrope;
    req_media.mrope = .{}; // ownership transferred to the sub-handler → slot
    if (is_stream) {
        handleStreamingGeneration(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, include_usage, has_tools, tools_json, allow_parallel_tools, logprobs_n, enable_thinking, surface_budget, enable_drafter, enable_mtp, allow_batch_mtp, sub_ve, req_media.chain, cache_key, sub_mrope, kv_quant_override, tokenize_ns) catch |err| {
            log.err("  -> streaming error: {}\n", .{err});
            // One mapping with the non-streaming arm: an HTTP status before the SSE head, an SSE `error` event after.
            sendGenerationError(allocator, stream, err, .openai) catch {};
        };
    } else {
        handleNonStreamingGeneration(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, has_tools, tools_json, allow_parallel_tools, logprobs_n, enable_thinking, surface_budget, enable_drafter, enable_mtp, allow_batch_mtp, sub_ve, req_media.chain, cache_key, sub_mrope, kv_quant_override, tokenize_ns) catch |err| {
            log.err("  -> {s}\n", .{@errorName(err)});
            sendGenerationError(allocator, stream, err, .openai) catch {};
        };
    }
}

const BudgetedReasoning = struct {
    text: []const u8,
    owned: bool,
};

pub fn reasoningWithinBudget(allocator: std.mem.Allocator, tok: *const Tokenizer, reasoning: []const u8, budget: i32) !BudgetedReasoning {
    if (budget < 0) return .{ .text = reasoning, .owned = false };
    const ids = try tok.encode(allocator, reasoning);
    defer allocator.free(ids);
    const n: usize = @intCast(budget);
    if (ids.len <= n) return .{ .text = reasoning, .owned = false };
    return .{ .text = try tok.decode(allocator, ids[0..n]), .owned = true };
}

/// Will this request build the MTP head's cache? The admission bill prices the head's KV, state and QSA ring
/// only then: a request a DFlash2 drafter serves never allocates them.
pub fn requestBillsMtpHead(
    enable_drafter: bool,
    enable_mtp: bool,
    dflash_loaded: bool,
    mtp_loaded: bool,
    sampling: sampler.SamplingParams,
    logprobs_n: u32,
) bool {
    return requestSpecModes(enable_drafter, enable_mtp, dflash_loaded, mtp_loaded, sampling, logprobs_n).use_mtp;
}

test "the bill prices the MTP head only for a request that runs it: a DFlash2 request never builds the head's cache" {
    const t = std.testing;
    const saved_force = server_config.default_force_mtp;
    defer server_config.default_force_mtp = saved_force;
    server_config.default_force_mtp = false;
    const plain = sampler.SamplingParams{};
    // A loaded drafter outranks the head; `enable_drafter:false` hands the request back to it.
    try t.expect(!requestBillsMtpHead(true, true, true, true, plain, 0));
    try t.expect(requestBillsMtpHead(false, true, true, true, plain, 0));
    // A penalty keeps the drafter off, so the head runs.
    try t.expect(requestBillsMtpHead(true, true, true, true, .{ .presence_penalty = 1.5 }, 0));
    try t.expect(requestBillsMtpHead(true, true, false, true, plain, 0));
    try t.expect(!requestBillsMtpHead(true, false, false, true, plain, 0));
    var cfg = qwen4ExpOomConfig();
    cfg.mtp_override = null;
    cfg.dflash_loaded = true;
    const seq: u64 = 100_000;
    const dflash = prefillRequestTerms(&cfg, seq, 4096, 8, 4096, .{ .mtp_on = requestBillsMtpHead(true, true, true, true, plain, 0) });
    const head = prefillRequestTerms(&cfg, seq, 4096, 8, 4096, .{ .mtp_on = requestBillsMtpHead(false, true, true, true, plain, 0) });
    try t.expectEqual(@as(u64, 0), dflash.mtp_head_kv_bytes);
    try t.expect(head.mtp_head_kv_bytes > 0);
    try t.expect(head.state_bytes > dflash.state_bytes);
    try t.expect(head.qsa_ring_bytes > dflash.qsa_ring_bytes);
}

// ── Shared utilities ──

pub fn sendResponse(stream: *Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
    logHttpResponse(status, content_type, body);
    try sendResponseFramed(stream, status, content_type, body);
}

/// Like `sendResponse` but does NOT dump the response body to the debug log —
/// only a one-line status summary. For high-frequency, self-describing bodies
/// where the full dump is pure noise: the Prometheus scrape (`/metrics`, hit
/// every couple seconds) and the panel's JSON feed (`/metrics.json`, ~1 Hz).
fn sendResponseQuiet(stream: *Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
    if (log.isDebug()) log.debug("[http] <- {s} {s} body={d}b (scrape; body not logged)\n", .{ status, content_type, body.len });
    try sendResponseFramed(stream, status, content_type, body);
}

pub fn sendResponseFramed(stream: *Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
    const extra = stream.effort_notice_json orelse "";
    const inject = extra.len > 0 and body.len > 0 and body[body.len - 1] == '}' and std.mem.indexOf(u8, body, "\"effort_notice\":") == null;
    const out_len = body.len + if (inject) extra.len else 0;
    if (stream.ws_mode) |bridge| {
        if (body.len > 0) try bridge.sendText(body);
        return;
    }

    var hdr_buf: [512]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n", .{
        status,
        content_type,
        out_len,
    }) catch return error.Overflow;
    try stream.writeAllNoFlush(hdr);
    try writeEffortNoticeHeader(stream);
    try stream.writeAllNoFlush("\r\n");
    stream.length_framed = true;
    if (inject) {
        try stream.writeAllNoFlush(body[0 .. body.len - 1]);
        try stream.writeAllNoFlush(extra);
        try stream.writeAll("}");
    } else if (body.len > 0) {
        try stream.writeAll(body);
    } else try stream.flush();
}

/// `raw_path` with the value of every `api_key` / `key` query parameter replaced by `***`; a query too long for
/// `buf` is dropped whole rather than risk logging a key.
pub fn redactQueryKeys(buf: []u8, raw_path: []const u8) []const u8 {
    const qpos = std.mem.indexOfScalar(u8, raw_path, '?') orelse return raw_path;
    const fail_closed = raw_path[0 .. qpos + 1];
    if (qpos + 1 > buf.len) return fail_closed[0..@min(fail_closed.len, buf.len)];
    @memcpy(buf[0 .. qpos + 1], fail_closed);
    var n = qpos + 1;
    var it = std.mem.splitScalar(u8, raw_path[qpos + 1 ..], '&');
    var first = true;
    while (it.next()) |pair| {
        const eq = std.mem.indexOfScalar(u8, pair, '=');
        const secret = eq != null and (std.mem.eql(u8, pair[0..eq.?], "api_key") or std.mem.eql(u8, pair[0..eq.?], "key"));
        const kept = if (secret) pair[0 .. eq.? + 1] else pair;
        const tail = if (secret) "***" else "";
        const need = @intFromBool(!first) + kept.len + tail.len;
        if (n + need > buf.len) return fail_closed;
        if (!first) {
            buf[n] = '&';
            n += 1;
        }
        @memcpy(buf[n..][0..kept.len], kept);
        n += kept.len;
        @memcpy(buf[n..][0..tail.len], tail);
        n += tail.len;
        first = false;
    }
    return buf[0..n];
}

fn logHttpRequest(method: []const u8, raw_path: []const u8, body: []const u8) void {
    if (!log.isDebug()) return;
    var buf: [1024]u8 = undefined;
    log.debug("[http] -> {s} {s} body={d}b\n", .{ method, redactQueryKeys(&buf, raw_path), body.len });
    logHttpBody("[http] request body", body);
}

pub fn logHttpResponse(status: []const u8, content_type: []const u8, body: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- {s} {s} body={d}b\n", .{ status, content_type, body.len });
    logHttpBody("[http] response body", body);
}

/// The `Access-Control-Allow-Headers` value each SSE surface advertises.
pub const SSE_ALLOW_HEADERS_DEFAULT = "Content-Type, Authorization";

pub fn logHttpSseData(data: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- sse data={d}b\n", .{data.len});
    logHttpBody("[http] sse data", data);
}

/// Cap on how much of a BINARY body reaches the debug log.
const BODY_LOG_LIMIT = 4096;

/// True when every byte is printable or ordinary whitespace. Multibyte UTF-8
/// (≥0x80) counts as text, so an emoji in a chat body doesn't demote it.
fn bodyIsText(body: []const u8) bool {
    for (body) |c| {
        if (c == '\n' or c == '\r' or c == '\t') continue;
        if (c < 0x20 or c == 0x7f) return false;
    }
    return true;
}

/// Bounded, strictly-printable view of a body, copied into `out`.
fn bodyPreview(out: []u8, body: []const u8) []const u8 {
    const n = @min(@min(body.len, out.len), BODY_LOG_LIMIT);
    for (body[0..n], 0..) |c, i| {
        out[i] = if (c == '\n' or c == '\r' or c == '\t' or (c >= 0x20 and c < 0x7f)) c else '.';
    }
    return out[0..n];
}

pub fn logHttpBody(label: []const u8, body: []const u8) void {
    if (body.len == 0) return;
    // Text bodies go out WHOLE — reading a full request body out of the debug
    // log is the documented way to reproduce a tool-calling bug.
    if (bodyIsText(body)) {
        log.debug("{s} ({d}b):\n{s}\n", .{ label, body.len, body });
        return;
    }
    // Binary is a different story. `/v1/images/edits` is multipart, so one image
    // upload wrote raw PNG bytes into the log, and a body near the 64 MB request
    // cap blew through the 32 MB rotation — taking the post-mortem file with it.
    var buf: [BODY_LOG_LIMIT]u8 = undefined;
    const preview = bodyPreview(&buf, body);
    log.debug("{s} ({d}b binary, first {d} sanitized):\n{s}…\n", .{ label, body.len, preview.len, preview });
}

const GaugeSamplerCtx = struct {
    metrics: *instr.Metrics,
    scheduler: *scheduler_mod.Scheduler,
    /// Dedicated stop flag — set before joining the thread (LIFO defers ensure
    /// this is written before join() blocks). A separate flag (not
    /// scheduler.shutdown) avoids a LIFO deadlock: scheduler.deinit() (which
    /// sets shutdown) is the FIRST-registered defer and runs LAST, after join.
    stop: *std.atomic.Value(bool),
};

/// Sample instantaneous system/queue state into the gauges. Non-blocking
/// syscalls plus a brief scheduler-counter lock. Called by the sampler thread.
/// The live-token gauge reads the scheduler's `inflight_generated_tokens`
/// aggregate (published once per decode tick) — it NEVER touches per-slot
/// fields off-thread, so the inference path stays lock-free per token.
fn sampleGauges(ctx: GaugeSamplerCtx) void {
    // System gauges (non-blocking syscalls).
    ctx.metrics.gpu_utilization_pct.set(@as(u64, metrics.getGpuPct()));
    ctx.metrics.memory_mb.set(@as(u64, metrics.getAppMemFootprintMb()));
    // The two halves of the MLX allocator, so `memory_mb`'s gap has a name.
    var mlx_active: usize = 0;
    var mlx_cache: usize = 0;
    _ = mlx.mlx_get_active_memory(&mlx_active);
    _ = mlx.mlx_get_cache_memory(&mlx_cache);
    ctx.metrics.mlx_active_bytes.set(@as(u64, mlx_active));
    ctx.metrics.mlx_cache_bytes.set(@as(u64, mlx_cache));
    // qwen4 n-gram table warm (F8) — same lock-free published-atomic pattern.
    ctx.metrics.ngram_warm_bytes.set(qwen4_mod.live_warm_bytes.load(.monotonic));

    // Request queue depth — brief lock to read two scheduler counters only.
    ctx.scheduler.queue_mu.lockUncancelable(ctx.scheduler.io);
    const running = @as(u64, ctx.scheduler.in_flight);
    const waiting = @as(u64, ctx.scheduler.pending.items.len);
    ctx.scheduler.queue_mu.unlock(ctx.scheduler.io);
    ctx.metrics.requests_running.set(running);
    ctx.metrics.requests_waiting.set(waiting);

    // Real-time throughput source: completed generation tokens plus the tokens
    // generated so far by still-decoding slots (the scheduler publishes that
    // aggregate atomically once per decode tick — race-free, no per-token
    // write). Non-zero even mid-request, unlike the completion-only counter.
    const inflight = ctx.scheduler.inflight_generated_tokens.load(.monotonic);
    ctx.metrics.generation_tokens_live.set(instr.liveGenerationTokens(ctx.metrics, inflight));

    // The other phase. Prompt-token counters and the prefill-time histogram
    // only advance at request COMPLETION, so a long prefill was invisible: GPU
    // pinned, decode reading 0, prefill reading "—". This gauge moves per
    // prefill chunk and returns to 0 the moment the prefill ends.
    ctx.metrics.prefill_tokens_live.set(ctx.scheduler.inflight_prefill_tokens.load(.monotonic));
    ctx.metrics.prefill_tokens_expected.set(ctx.scheduler.inflight_prefill_expected.load(.monotonic));
    ctx.metrics.requests_prefilling.set(ctx.scheduler.requests_prefilling.load(.monotonic));
}

const IdleEvictCtx = struct {
    scheduler: *scheduler_mod.Scheduler,
    /// `--idle-evict-secs`, in ms.
    window_ms: i64,
    /// Dedicated stop flag, for the same reason `GaugeSamplerCtx` has one:
    /// gating on `scheduler.shutdown` would never be observed, because
    /// `scheduler.deinit()` (which sets it) is the FIRST-registered defer and
    /// runs LAST, after this thread's join. Worse here than for the sampler —
    /// a sweep blocked inside `unloadModel` is waiting on the inference thread,
    /// and `deinit` only rescues orphaned unload requests AFTER joining it, so
    /// a join placed inside `deinit` would hang forever.
    stop: *std.atomic.Value(bool),
};

/// `--idle-evict-secs`: unload models that have served nothing for the window.
///
/// Goes through `Scheduler.unloadModelIfIdle`, the same path
/// `POST /v1/unload-model` uses, so the eviction itself is unchanged and
/// nothing here touches mlx directly.
fn idleEvictLoop(ctx: IdleEvictCtx) void {
    const sch = ctx.scheduler;
    const tick_ms = scheduler_mod.idleEvictTickMs(ctx.window_ms);
    // Wake every 500 ms to check the stop flag; sweep once per tick interval.
    // `std.c.nanosleep` for the same reason as the gauge sampler: `std.time.sleep`
    // is gone in Zig 0.16, and a blocking Io timer here could outlive the stop.
    const poll_ts = std.c.timespec{ .sec = 0, .nsec = 500_000_000 };
    const ticks_per_sweep: u64 = @max(1, @as(u64, @intCast(@divTrunc(tick_ms, 500))));
    var tick: u64 = 0;
    while (!ctx.stop.load(.monotonic)) {
        _ = std.c.nanosleep(&poll_ts, null);
        tick += 1;
        if (tick < ticks_per_sweep) continue;
        tick = 0;
        if (ctx.stop.load(.monotonic)) return;

        // One victim per sweep. `unloadModelIfIdle` blocks on the inference
        // thread, so holding the registry mutex across it would stall every
        // request; the next sweep takes anything else that has gone idle.
        const now_ms = io_util.nowMsMonotonic(sch.io);
        sch.registry.mutex.lockUncancelable(sch.io);
        const victim = sch.registry.pickIdleEvictable(now_ms, ctx.window_ms);
        const idle_ms = if (victim) |v| now_ms - v.last_used_ms.load(.acquire) else 0;
        const bytes = if (victim) |v| v.bytes_resident else 0;
        sch.registry.mutex.unlock(sch.io);

        if (victim) |entry| {
            log.info("[registry] idle-evicting model id={s} ({d:.2} GB resident, idle {d}s)\n", .{
                entry.id,
                @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0),
                @divTrunc(idle_ms, 1000),
            });
            // Age and refcount are re-checked under the mutex inside, so a
            // request arriving since the pick cancels the eviction instead of
            // pinning the model in `.evicting` behind a live stream.
            sch.unloadModelIfIdle(entry.id, ctx.window_ms) catch |err| {
                log.warn("[registry] idle eviction failed for {s}: {t}\n", .{ entry.id, err });
            };
        }
    }
}

const sampler_idle = 0;
const sampler_spawning = 1;
const sampler_running = 2;
const sampler_closed = 3;
var sampler_state = std.atomic.Value(u8).init(sampler_idle);
var sampler_thread: std.Thread = undefined;
var sampler_stop = std.atomic.Value(bool).init(false);

/// Starts the gauge sampler once, from `--metrics` at boot or from `POST /metrics/enable`, whichever gets
/// there first. A no-op while metrics are off, before the scheduler exists, and after shutdown began.
fn startGaugeSampler() !void {
    const m = instr.current() orelse return;
    const sch = global_scheduler orelse return;
    if (sampler_state.cmpxchgStrong(sampler_idle, sampler_spawning, .acquire, .monotonic) != null) return;
    errdefer sampler_state.store(sampler_idle, .release);
    sampler_thread = try std.Thread.spawn(.{}, gaugeSamplerLoop, .{GaugeSamplerCtx{ .metrics = m, .scheduler = sch, .stop = &sampler_stop }});
    sampler_state.store(sampler_running, .release);
}

fn stopGaugeSampler() void {
    sampler_stop.store(true, .monotonic);
    while (true) switch (sampler_state.load(.acquire)) {
        sampler_idle => if (sampler_state.cmpxchgStrong(sampler_idle, sampler_closed, .acq_rel, .monotonic) == null) return,
        sampler_spawning => std.atomic.spinLoopHint(),
        sampler_running => {
            sampler_thread.join();
            sampler_state.store(sampler_closed, .release);
            return;
        },
        else => return,
    };
}

fn gaugeSamplerLoop(ctx: GaugeSamplerCtx) void {
    // Wake every 500 ms to check the stop flag; sample system state once per
    // 2-second interval. The 2 s cadence is what makes generation_tokens_live a
    // real-time tok/s source — the panel windows over a few of these samples.
    // Uses std.c.nanosleep (POSIX) because std.time.sleep was removed in Zig
    // 0.16 — this thread has no Io handle.
    const SAMPLE_INTERVAL_TICKS: u64 = 4; // 4 × 500 ms = 2 s
    const poll_ts = std.c.timespec{ .sec = 0, .nsec = 500_000_000 };
    // Sample once immediately so the panel / Grafana show real values from the
    // first scrape instead of zeros for the first interval.
    sampleGauges(ctx);
    var tick: u64 = 0;
    while (!ctx.stop.load(.monotonic)) {
        _ = std.c.nanosleep(&poll_ts, null);
        tick += 1;
        if (tick < SAMPLE_INTERVAL_TICKS) continue;
        tick = 0;
        sampleGauges(ctx);
    }
}

// ── API-key auth helpers (used only when --api-key / g_api_key is set) ──

/// True if the connection's peer is a loopback address (127.0.0.0/8, ::1, or an
/// IPv4-mapped-IPv6 loopback). Loopback is the local machine → trusted and
/// exempt from the API key, so the same-host app never needs credentials.
fn ipIsLoopback(addr: std.Io.net.IpAddress) bool {
    return switch (addr) {
        .ip4 => |a4| a4.bytes[0] == 127, // 127.0.0.0/8
        .ip6 => |a6| blk: {
            const b = a6.bytes; // big-endian 16 bytes
            const zeros10: [10]u8 = @splat(0);
            // ::1
            const zeros15: [15]u8 = @splat(0);
            if (std.mem.eql(u8, b[0..15], &zeros15) and b[15] == 1) break :blk true;
            // ::ffff:127.x.x.x  (IPv4-mapped loopback)
            if (std.mem.eql(u8, b[0..10], &zeros10) and b[10] == 0xff and b[11] == 0xff and b[12] == 127) break :blk true;
            break :blk false;
        },
    };
}

fn peerIsLoopback(conn: *const Conn) bool {
    return ipIsLoopback(conn.stream.socket.address);
}

/// Collapse JSON's optional `\/` escape to `/`. Swift's JSONSerialization
/// (and PHP's json_encode) escape every slash, and a model id is read out of
/// the RAW body, so `org\/name` would miss the registry. Returns the input
/// verbatim (zero-copy) when there is nothing to collapse or the scratch
/// buffer is too small.
fn unescapeJsonSlashes(buf: []u8, s: []const u8) []const u8 {
    if (std.mem.indexOf(u8, s, "\\/") == null or s.len > buf.len) return s;
    var n: usize = 0;
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] == '\\' and i + 1 < s.len and s[i + 1] == '/') continue;
        buf[n] = s[i];
        n += 1;
    }
    return buf[0..n];
}

test "JSON-escaped slashes in a model id canonicalize (Swift clients send org\\/name)" {
    const t = std.testing;
    var buf: [256]u8 = undefined;
    try t.expectEqualStrings("ddalcu/gemma-e2b", unescapeJsonSlashes(&buf, "ddalcu\\/gemma-e2b"));
    // No escapes → the INPUT slice comes back verbatim (zero-copy).
    const plain = "ddalcu/gemma-e2b";
    try t.expect(unescapeJsonSlashes(&buf, plain).ptr == plain.ptr);
    // Only the two-byte sequence `\/` collapses; other backslashes survive.
    try t.expectEqualStrings("a\\b/c", unescapeJsonSlashes(&buf, "a\\b\\/c"));
    // Oversized input degrades to verbatim rather than truncating.
    var tiny: [4]u8 = undefined;
    try t.expectEqualStrings("x\\/y", unescapeJsonSlashes(tiny[0..2], "x\\/y"));
}

pub fn responseBody(response: []const u8) []const u8 {
    const at = std.mem.indexOf(u8, response, "\r\n\r\n") orelse return "";
    return response[at + 4 ..];
}

test "metrics enable: POST /metrics/enable turns the feeds on, idempotently" {
    const t = std.testing;
    instr.process = .{};
    defer instr.process = .{};
    for ([_][]const u8{ "/metrics", "/metrics.json" }) |path| {
        const request = try std.fmt.allocPrint(t.allocator, "GET {s} HTTP/1.1\r\n\r\n", .{path});
        defer t.allocator.free(request);
        const off = try serveOneForTest(request);
        defer t.allocator.free(off);
        try t.expect(std.mem.startsWith(u8, off, "HTTP/1.1 503 "));
        try t.expect(std.mem.indexOf(u8, responseBody(off), "POST /metrics/enable") != null);
    }
    const first = try serveOneForTest("POST /metrics/enable HTTP/1.1\r\nContent-Length: 0\r\n\r\n");
    defer t.allocator.free(first);
    try t.expect(std.mem.startsWith(u8, first, "HTTP/1.1 200 OK\r\n"));
    try t.expectEqualStrings("{\"metrics\":true,\"already_enabled\":false}", responseBody(first));
    try t.expect(instr.current() != null);
    for ([_][]const u8{ "/metrics", "/metrics.json" }) |path| {
        const request = try std.fmt.allocPrint(t.allocator, "GET {s} HTTP/1.1\r\n\r\n", .{path});
        defer t.allocator.free(request);
        const on = try serveOneForTest(request);
        defer t.allocator.free(on);
        try t.expect(std.mem.startsWith(u8, on, "HTTP/1.1 200 OK\r\n"));
    }
    const core = instr.current().?;
    core.recordRejected();
    const again = try serveOneForTest("POST /metrics/enable HTTP/1.1\r\nContent-Length: 0\r\n\r\n");
    defer t.allocator.free(again);
    try t.expect(std.mem.startsWith(u8, again, "HTTP/1.1 200 OK\r\n"));
    try t.expectEqualStrings("{\"metrics\":true,\"already_enabled\":true}", responseBody(again));
    try t.expect(instr.current().? == core);
    try t.expectEqual(@as(u64, 1), core.requests_rejected_total.load());
}

test "an unknown endpoint is a 404 that never reaches model resolution" {
    const t = std.testing;
    var sch: scheduler_mod.Scheduler = undefined;
    const prev_sch = global_scheduler;
    defer global_scheduler = prev_sch;
    global_scheduler = &sch;
    const response = try serveOneForTest("POST /v1/load HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 13\r\n\r\n{\"model\":\"m\"}");
    defer t.allocator.free(response);
    try t.expect(std.mem.startsWith(u8, response, "HTTP/1.1 404 Not Found\r\n"));
    try t.expect(std.mem.indexOf(u8, responseBody(response), "Unknown endpoint") != null);
}

test "ipIsLoopback exempts local addresses only" {
    // IPv4 loopback (whole 127.0.0.0/8) is exempt; a LAN address is not.
    try std.testing.expect(ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 0 } }));
    try std.testing.expect(ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 127, 3, 2, 1 }, .port = 0 } }));
    try std.testing.expect(!ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 192, 168, 1, 42 }, .port = 0 } }));
    try std.testing.expect(!ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 10, 0, 0, 5 }, .port = 0 } }));
    // IPv6 ::1 is exempt; a routable v6 address is not.
    try std.testing.expect(ipIsLoopback(.{ .ip6 = .{ .bytes = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, .port = 0 } }));
    // IPv4-mapped loopback ::ffff:127.0.0.1
    try std.testing.expect(ipIsLoopback(.{ .ip6 = .{ .bytes = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 127, 0, 0, 1 }, .port = 0 } }));
    try std.testing.expect(!ipIsLoopback(.{ .ip6 = .{ .bytes = .{ 0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, .port = 0 } }));
}

/// Extract a JSON field's raw value from a JSON body string.
/// Returns the raw substring for the field value (e.g., the array or object).
fn extractJsonField(body: []const u8, field: []const u8) ?[]const u8 {
    // Search for "field": or "field" :
    var pos: usize = 0;
    while (pos < body.len) {
        const quote_pos = std.mem.indexOf(u8, body[pos..], "\"") orelse return null;
        const key_start = pos + quote_pos + 1;
        if (key_start + field.len >= body.len) return null;

        if (std.mem.eql(u8, body[key_start .. key_start + field.len], field) and
            body[key_start + field.len] == '"')
        {
            // Found the key, skip to colon
            var i = key_start + field.len + 1;
            while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\n' or body[i] == '\r' or body[i] == '\t')) {
                i += 1;
            }
            if (i >= body.len) return null;

            // Now extract the value - find matching bracket/brace
            const start = i;
            const open = body[start];
            const close: u8 = if (open == '[') ']' else if (open == '{') '}' else return null;
            var depth: usize = 1;
            var j = start + 1;
            var in_string = false;
            while (j < body.len and depth > 0) {
                if (body[j] == '\\' and in_string) {
                    j += 1; // skip escaped char
                } else if (body[j] == '"') {
                    in_string = !in_string;
                } else if (!in_string) {
                    if (body[j] == open) depth += 1;
                    if (body[j] == close) depth -= 1;
                }
                j += 1;
            }
            if (depth == 0) return body[start..j];
            return null;
        }
        pos = key_start;
    }
    return null;
}

/// Body text for a context-overflow 400, in a caller-owned buffer.
///
/// The COUNTS are the point. "Prompt exceeds maximum context length" tells a
/// client only that it failed; with both numbers the client can say how far
/// over it went and offer the one action that fixes it (raise the context to
/// N). Our own app renders exactly that, and the numbers are only knowable
/// here — the request is rejected before any usage is reported.
///
/// The legacy sentence stays the PREFIX: `APIError.looksLikeContextOverflow`
/// and third-party clients key on it, so appending must not move it. Output is
/// digits and spaces, so it carries no escaping hazard into the JSON sink, and
/// a bufPrint failure falls back to the bare sentence rather than sending no
/// body at all (the media-gen fixed-buffer class).
/// The `error.NotEnoughMemory` refusal, in ONE place because two identical
/// copies of it is how they drift.
///
/// The old wording ("retry after current requests complete") named concurrency,
/// which is the one thing it is never waiting for: the gate refuses against a
/// STATIC cap, so on an idle server with nothing loaded there is nothing to
/// retry after (#126). The counts are knowable only server-side and go to the
/// log; what the client needs is the knob.
///
/// ONE gate reaches this arm: the resident-model budget with no evictable
/// LRU victim (scheduler's `planEvictionsLocked` refusal). The free-memory
/// PREFLIGHT refuses as `error.InsufficientMemory` and gets its own message
/// below (#144) — blaming free memory here sends a user quitting apps when
/// the fix is unloading a model or moving the cap.
pub const not_enough_memory_message =
    "Not enough memory to load model: it would exceed the resident-model budget and no loaded model can be evicted. " ++
    "Unload the model you are chatting with (tray > Models > eject), " ++
    "or raise or disable the cap with --max-resident-mem <size>|0. " ++
    "The server log names the exact figures and the current cap.";

/// #144: the memory PREFLIGHT refusal (free RAM can't hold weights + warmup
/// headroom) is distinct from the resident-budget gate above and has a
/// different knob. Counts go to the log (#126); the client gets the remedy.
pub const insufficient_free_memory_message =
    "Not enough free memory to load model: weights + warmup headroom exceed what is currently available. " ++
    "Close other apps or models and retry (the server log names the peak estimate and available memory), or pass --skip-mem-preflight to override.";

/// A request refused before it owned a slot. A null sink (metrics off) is one branch.
pub fn countRejected() void {
    if (instr.current()) |m| m.recordRejected();
}

pub fn sendErrorResponse(allocator: std.mem.Allocator, stream: *Conn, status: []const u8, err_type: []const u8, message: []const u8, code: ?u32) !void {
    const escaped_msg = try jsonEscape(allocator, message);
    defer allocator.free(escaped_msg);

    var code_buf: [16]u8 = undefined;
    const code_str = if (code) |c|
        std.fmt.bufPrint(&code_buf, "{d}", .{c}) catch "null"
    else
        "null";

    const body = try std.fmt.allocPrint(allocator,
        \\{{"error":{{"message":{s},"type":"{s}","param":null,"code":{s}}}}}
    , .{ escaped_msg, err_type, code_str });
    defer allocator.free(body);
    try sendResponse(stream, status, "application/json", body);
}

/// Build a llama.cpp-style `timings` JSON object (no surrounding key) from
/// raw nanosecond counts and token totals. Caller frees. Returns an empty
/// string when `prefill_ns`, `decode_ns`, AND `tokenize_ns` are all zero
/// (legacy paths that don't measure timing) so the field can be omitted.
///
/// `tokenize_ns` is the wall-clock cost of the synchronous
/// `renderChatTemplate + tokenizer.encode` step on the request thread.
/// Phase 4 #1 of `performance-plan.md` calls for instrumentation-first:
/// before we move tokenize onto a worker thread, we need numbers per
/// engine / prompt length. Pass 0 for legacy paths that don't measure it
/// (the field is then omitted from the JSON; existing callers stay shape-
/// compatible while the new ones surface the metric).
///
/// Iteration 2: chat-template render+tokenize wrapper that consults a
/// per-LoadedModel LRU cache before calling the underlying engine's
/// encoder. On hit the returned slice is a fresh allocation owned by the
/// caller (drop-in replacement for the engine encoders). On miss it
/// stores a copy of the result in the cache.
/// A bool the client set in `chat_template_kwargs` (the vLLM/SGLang spelling for
/// template variables); null when absent or not a bool. The one reader of that object.
pub fn chatTemplateKwargBool(root: std.json.ObjectMap, key: []const u8) ?bool {
    const kwargs = root.get("chat_template_kwargs") orelse return null;
    if (kwargs != .object) return null;
    const v = kwargs.object.get(key) orelse return null;
    return if (v == .bool) v.bool else null;
}

/// The model's chat config as this request renders it.
pub fn renderConfigFor(chat_config: *const chat_mod.ChatConfig, root: std.json.ObjectMap, flag: ?bool, setting: ?bool) chat_mod.ChatConfig {
    var cfg = chat_config.*;
    cfg.preserve_thinking = resolvePreserveThinking(chatTemplateKwargBool(root, "preserve_thinking"), flag, setting);
    return cfg;
}

pub fn cachedFormatChat(
    allocator: std.mem.Allocator,
    io: std.Io,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    messages: []const chat_mod.Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
    reasoning_effort: ?[]const u8,
    /// Extend the trailing assistant message rather than answering after it.
    continue_final: bool,
) ![]u32 {
    const cache_ptr: ?*tokenize_cache_mod.TokenizeCache = if (lm.tokenize_cache) |*tc| tc else null;
    const key_opt: ?u64 = if (cache_ptr != null)
        tokenize_cache_mod.TokenizeCache.keyFor(messages, tools_json, tool_choice_instruction, enable_thinking, reasoning_effort, continue_final, chat_config.preserve_thinking)
    else
        null;
    if (cache_ptr) |cache| if (key_opt) |key| {
        if (try cache.get(io, key, allocator)) |cached| return cached;
    };
    const ids = try chat_mod.formatChat(allocator, tok, messages, chat_config, tools_json, tool_choice_instruction, enable_thinking, reasoning_effort, continue_final);
    if (cache_ptr) |cache| if (key_opt) |key| {
        // Insert is best-effort; an OOM in the cache shouldn't fail the
        // request — the user already has their tokenized prompt.
        cache.put(io, key, ids) catch |err| {
            log.warn("[tokenize-cache] put failed: {s}\n", .{@errorName(err)});
        };
    };
    return ids;
}

/// THE JSON string escaper for every response body. Every string here is model bytes
/// somewhere and a token is a BPE fragment, so invalid UTF-8 is sanitized INSIDE the
/// escaper (one U+FFFD per bad sequence); valid input passes through byte for byte.
pub fn jsonEscape(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    if (std.unicode.utf8ValidateSlice(input)) return jsonEscapeValid(allocator, input);
    const clean = try chat_mod.utf8Sanitize(allocator, input);
    defer allocator.free(clean);
    return jsonEscapeValid(allocator, clean);
}

fn jsonEscapeValid(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);

    try result.append(allocator, '"');
    for (input) |c| {
        switch (c) {
            '"' => try result.appendSlice(allocator, "\\\""),
            '\\' => try result.appendSlice(allocator, "\\\\"),
            '\n' => try result.appendSlice(allocator, "\\n"),
            '\r' => try result.appendSlice(allocator, "\\r"),
            '\t' => try result.appendSlice(allocator, "\\t"),
            else => {
                if (c < 0x20) {
                    var esc_buf: [6]u8 = undefined;
                    const s = std.fmt.bufPrint(&esc_buf, "\\u{x:0>4}", .{c}) catch unreachable;
                    try result.appendSlice(allocator, s);
                } else {
                    try result.append(allocator, c);
                }
            },
        }
    }
    try result.append(allocator, '"');
    return result.toOwnedSlice(allocator);
}

test "EVERY JSON string escaper survives bytes that are not valid UTF-8" {
    // Class guard over the three escapers (OpenAI/Anthropic, Responses, the prompt
    // render): one passing bytes through loses a whole reply.
    const a = std.testing.allocator;
    const cases = [_][]const u8{
        "near\xe8\x91\xe4\xb8\x89 in", // 3-byte lead, one continuation, then another lead
        "\xff\xfe", // never-valid lead bytes
        "tail\xf0\x9f", // first half of a 4-byte emoji, cut by max_tokens
    };
    for (cases) |bad| {
        const a1 = try jsonEscape(a, bad);
        defer a.free(a1);
        try std.testing.expect(std.unicode.utf8ValidateSlice(a1));

        const a2 = try responses_mod.jsonEscape(a, bad);
        defer a.free(a2);
        try std.testing.expect(std.unicode.utf8ValidateSlice(a2));

        var buf = std.ArrayList(u8).empty;
        defer buf.deinit(a);
        try chat_mod.appendJsonString(a, &buf, bad);
        try std.testing.expect(std.unicode.utf8ValidateSlice(buf.items));
    }
    // Valid input is passed through byte for byte.
    const good = try jsonEscape(a, "héllo \u{1F600}");
    defer a.free(good);
    try std.testing.expectEqualStrings("\"héllo \u{1F600}\"", good);
}

/// Wave 1.A — parse the optional per-request `kv_quant` body field. Accepts
/// the string forms `"off"`, `"0"`, `"4"`, `"6"`, `"8"` and the integer forms `0`,
/// `4`, `6`, `8`. Returns null when the field is absent or unrecognized (the
/// caller falls back to the process-level `--kv-quant` default carried on
/// the scheduler). Returns `KVQuantConfig.dense` for "off"/0.
pub fn parseKvQuantOverride(root: std.json.ObjectMap) ?transformer_mod.KVQuantConfig {
    const v = root.get("kv_quant") orelse return null;
    const kq = transformer_mod.KVQuantConfig.fromJsonValue(v) orelse return null;
    switch (kq.scheme) {
        .off => log.info("  kv-quant override: off (per-request)\n", .{}),
        .affine => log.info("  kv-quant override: affine {d}-bit (per-request)\n", .{kq.bits}),
    }
    return kq;
}

// ── Vision Processing ──

/// Where a request body keeps the message a media fault belongs to
/// (`messages[3]`, `input[1]`).
pub const WirePos = struct { list: []const u8, index: usize };

const WireVocab = enum { openai, anthropic };

/// One message's content array read in order.
pub const ReadContent = struct {
    text: []const u8 = "",
    images: ?[]const chat_mod.ImageData = null,
    videos: ?[]const chat_mod.VideoData = null,
    audio: ?[]const chat_mod.AudioData = null,
    media_parts: ?[]const chat_mod.MediaPart = null,
};

/// Read a content array: text parts join as `joinedTextParts` does, and each
/// media part is decoded into `media` and placed at the text offset it sat at,
/// so the template renders its placeholder there.
pub fn readContentParts(
    allocator: std.mem.Allocator,
    media: *RequestMedia,
    texts: *std.ArrayList([]const u8),
    parts: []const std.json.Value,
    vocab: WireVocab,
    pos: WirePos,
    vp: chat_mod.VisionPreproc,
) !ReadContent {
    const img_slot = try media.openImages();
    const vid_slot = try media.openVideos();
    const aud_slot = try media.openAudio();
    var placed = std.ArrayList(chat_mod.MediaPart).empty;
    defer placed.deinit(allocator);
    // The length the joined text has reached: a media part renders there.
    var joined_len: usize = 0;
    for (parts) |part| {
        if (part != .object) continue;
        const tv = part.object.get("type") orelse continue;
        if (tv != .string) continue;
        const ptype = tv.string;
        if (std.mem.eql(u8, ptype, "text")) {
            const t = part.object.get("text") orelse continue;
            if (t != .string or t.string.len == 0) continue;
            joined_len += t.string.len + @intFromBool(joined_len > 0);
        } else if (vocab == .openai and std.mem.eql(u8, ptype, "image_url")) {
            if (media.addImage(img_slot, openAiImageUrl(part), vp, pos))
                try placed.append(allocator, .{ .at = joined_len, .kind = .image });
        } else if (vocab == .anthropic and std.mem.eql(u8, ptype, "image")) {
            const url = try anthropicImageUrl(allocator, part);
            defer if (url) |u| allocator.free(u);
            if (media.addImage(img_slot, url, vp, pos))
                try placed.append(allocator, .{ .at = joined_len, .kind = .image });
        } else if (vocab == .openai and std.mem.eql(u8, ptype, "video_url")) {
            // A video is, on the wire, an ordered array of frame image data
            // URLs: frame extraction is the client's job.
            var frame_urls = std.ArrayList([]const u8).empty;
            defer frame_urls.deinit(allocator);
            var fps: f64 = 24;
            if (part.object.get("video_url")) |vid_obj| if (vid_obj == .object) {
                if (vid_obj.object.get("fps")) |v| {
                    const rate: f64 = if (v == .float) v.float else if (v == .integer) @floatFromInt(v.integer) else 0;
                    if (std.math.isFinite(rate) and rate > 0) fps = rate;
                }
                if (vid_obj.object.get("frames")) |frames| if (frames == .array) {
                    for (frames.array.items) |f| if (f == .string) try frame_urls.append(allocator, f.string);
                };
            };
            if (media.addVideo(vid_slot, frame_urls.items, vp, pos)) {
                const videos = media.videos(vid_slot);
                videos.items[videos.items.len - 1].fps = fps;
                try placed.append(allocator, .{ .at = joined_len, .kind = .video });
            }
        } else if (vocab == .openai and std.mem.eql(u8, ptype, "input_audio")) {
            const data: ?[]const u8 = if (part.object.get("input_audio")) |a| (if (a == .object) (if (a.object.get("data")) |d| (if (d == .string) d.string else null) else null) else null) else null;
            media.addAudio(aud_slot, data, pos);
        }
    }
    const joined = try joinedTextParts(allocator, parts);
    if (joined.owned) try texts.append(allocator, joined.text);
    return .{
        .text = joined.text,
        .images = media.imagesSlice(img_slot),
        .videos = media.videosSlice(vid_slot),
        .audio = media.audioSlice(aud_slot),
        .media_parts = try media.ownParts(&placed),
    };
}

/// A request body's messages and every buffer they borrow, owned in one place.
pub const WireMessages = struct {
    allocator: std.mem.Allocator,
    messages: std.ArrayList(chat_mod.Message) = .empty,
    media: RequestMedia,
    tool_call_lists: std.ArrayList([]const chat_mod.ToolCall) = .empty,
    /// Joined text, serialized tool arguments and other strings messages borrow.
    texts: std.ArrayList([]const u8) = .empty,

    pub fn init(allocator: std.mem.Allocator) WireMessages {
        return .{ .allocator = allocator, .media = RequestMedia.init(allocator) };
    }

    pub fn deinit(self: *WireMessages) void {
        for (self.tool_call_lists.items) |tcs| self.allocator.free(tcs);
        self.tool_call_lists.deinit(self.allocator);
        for (self.texts.items) |s| self.allocator.free(s);
        self.texts.deinit(self.allocator);
        self.media.deinit();
        self.messages.deinit(self.allocator);
    }
};

/// Every message of a chat-completions body, each with its media decoded and
/// placed where it sat — user, tool or any other role.
fn readOpenAiMessages(wire: *WireMessages, items: []const std.json.Value, vp: chat_mod.VisionPreproc) !void {
    const allocator = wire.allocator;
    for (items, 0..) |msg_val, raw_index| {
        // A malformed element is skipped rather than crashing on `.object`.
        if (msg_val != .object) continue;
        const obj = msg_val.object;
        const role_val = obj.get("role") orelse continue;
        if (role_val != .string) continue;

        var read: ReadContent = .{};
        if (obj.get("content")) |cv| switch (cv) {
            .string => |s| read.text = s,
            .array => |arr| read = try readContentParts(allocator, &wire.media, &wire.texts, arr.items, .openai, .{ .list = "messages", .index = raw_index }, vp),
            else => {},
        };

        var msg_tool_calls: ?[]const chat_mod.ToolCall = null;
        if (std.mem.eql(u8, role_val.string, "assistant")) {
            if (obj.get("tool_calls")) |tc_val| {
                if (tc_val == .array) {
                    var tcs = std.ArrayList(chat_mod.ToolCall).empty;
                    for (tc_val.array.items) |tc_item| {
                        if (tc_item != .object) continue;
                        const tc_id = if (tc_item.object.get("id")) |v| (if (v == .string) v.string else "") else "";
                        const func = tc_item.object.get("function") orelse continue;
                        if (func != .object) continue;
                        const fn_name = if (func.object.get("name")) |v| (if (v == .string) v.string else "") else "";
                        const fn_args = if (func.object.get("arguments")) |v| (if (v == .string) v.string else "{}") else "{}";
                        try tcs.append(allocator, .{ .id = tc_id, .name = fn_name, .arguments = fn_args });
                    }
                    if (tcs.items.len > 0) {
                        const owned = try tcs.toOwnedSlice(allocator);
                        try wire.tool_call_lists.append(allocator, owned);
                        msg_tool_calls = owned;
                    } else {
                        tcs.deinit(allocator);
                    }
                }
            }
        }

        const tool_call_id: ?[]const u8 = if (std.mem.eql(u8, role_val.string, "tool"))
            (if (obj.get("tool_call_id")) |v| (if (v == .string) v.string else null) else null)
        else
            null;

        // Reasoning the client round-trips on assistant history. Dropping it
        // starves templates that persist reasoning across turns: every
        // prior turn renders the empty <think></think> nothink signature and
        // the model stops thinking from turn 2 of a session.
        const msg_reasoning: ?[]const u8 = if (std.mem.eql(u8, role_val.string, "assistant"))
            messageReasoningFromObj(obj)
        else
            null;

        if (read.text.len == 0 and msg_tool_calls == null and read.images == null and read.videos == null and read.audio == null and msg_reasoning == null and !std.mem.eql(u8, role_val.string, "tool")) continue;

        try wire.messages.append(allocator, .{
            .role = chat_mod.canonicalRole(role_val.string),
            .content = read.text,
            .tool_calls = msg_tool_calls,
            .tool_call_id = tool_call_id,
            .images = read.images,
            .videos = read.videos,
            .audio = read.audio,
            .media_parts = read.media_parts,
            .reasoning_content = msg_reasoning,
        });
    }
}

/// The encode's fit check, before it runs: `visionEncodeBill` must fit what the
/// GPU has left.
pub fn towerFitFault(config: *const model_mod.ModelConfig, images: []const chat_mod.ImageData, videos: []const chat_mod.VideoData, rows: usize, available: u64) ?MediaFault {
    if (config.qv_heads == 0) return null;
    const bill = visionEncodeBill(config, images, videos, rows);
    const worst = bill.largest_group_patches;
    const needed = bill.bytes;
    if (needed <= available) return null;
    const mb: u64 = 1024 * 1024;
    return MediaFault.init(false, "encoding the media needs ~{d} MB of GPU memory (its largest block has {d} patches), ~{d} MB is available", .{ needed / mb, worst, available / mb });
}

fn towerFitFaultText(out: []u8, config: *const model_mod.ModelConfig, images: []const chat_mod.ImageData, videos: []const chat_mod.VideoData, rows: usize, available: u64) ?[]const u8 {
    const fault = towerFitFault(config, images, videos, rows, available) orelse return null;
    const text = fault.text();
    const n = @min(text.len, out.len);
    @memcpy(out[0..n], text[0..n]);
    return out[0..n];
}

/// Qwen3-VL interleaved M-RoPE position-id table for a request. `pos` (owned) is
/// the flat [3 × total] i32 table threaded to the slot; null for non-Qwen / no
/// images. Pass-through bundle so sub-handlers thread one value, not three.
pub const MropeData = struct {
    pos: ?[]const i32 = null,
    total: usize = 0,
    delta: i32 = 0,
};

// 8x8 solid PNGs: red and blue.
pub const test_png_red = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEUlEQVR42mP4z8CAFTEMLQkAKP8/wc53yE8AAAAASUVORK5CYII=";
pub const test_png_blue = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEElEQVR42mNgYPiPAw0pCQCpcD/B/MtF/AAAAABJRU5ErkJggg==";
const test_qwen_vp: chat_mod.VisionPreproc = .{ .mode = .qwen, .patch = 16, .tps = 2, .merge = 2 };

pub fn testWire(body: []const u8, vocab: WireVocab) !struct { parsed: std.json.Parsed(std.json.Value), wire: WireMessages } {
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, body, .{});
    errdefer parsed.deinit();
    var wire = WireMessages.init(std.testing.allocator);
    errdefer wire.deinit();
    const items = parsed.value.object.get("messages").?.array.items;
    switch (vocab) {
        .openai => try readOpenAiMessages(&wire, items, test_qwen_vp),
        .anthropic => try readAnthropicMessages(&wire, items, test_qwen_vp),
    }
    return .{ .parsed = parsed, .wire = wire };
}

test "an undecodable image is refused naming its message and the reason" {
    const Case = struct { url: []const u8, reason: []const u8 };
    const cases = [_]Case{
        .{ .url = "https://example.com/shot.png", .reason = "remote URLs are not fetched" },
        .{ .url = "data:image/png;base64,@@@@", .reason = "base64 payload is malformed" },
        .{ .url = "data:image/png;base64,AAAA", .reason = "not a readable JPEG, PNG, GIF, BMP or WebP" },
    };
    for (cases) |c| {
        const body = try std.fmt.allocPrint(std.testing.allocator, "{{\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}},{{\"role\":\"assistant\",\"content\":\"ok\"}},{{\"role\":\"user\",\"content\":[{{\"type\":\"image_url\",\"image_url\":{{\"url\":\"{s}\"}}}}]}}]}}", .{c.url});
        defer std.testing.allocator.free(body);
        var t = try testWire(body, .openai);
        defer t.parsed.deinit();
        defer t.wire.deinit();
        const fault = t.wire.media.refusal() orelse return error.TestExpectedRefusal;
        try std.testing.expect(!fault.server_error);
        try std.testing.expect(std.mem.startsWith(u8, fault.text(), "messages[2]: image could not be decoded"));
        try std.testing.expect(std.mem.indexOf(u8, fault.text(), c.reason) != null);
    }
    // An Anthropic image block with no data is refused too, not dropped.
    var t = try testWire("{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"\"}}]}]}", .anthropic);
    defer t.parsed.deinit();
    defer t.wire.deinit();
    try std.testing.expect(std.mem.startsWith(u8, t.wire.media.refusal().?.text(), "messages[0]: image could not be decoded"));
}

test "a body past the image limit is refused with both counts" {
    var body = std.ArrayList(u8).empty;
    defer body.deinit(std.testing.allocator);
    try body.appendSlice(std.testing.allocator, "{\"messages\":[{\"role\":\"user\",\"content\":[");
    for (0..chat_mod.MAX_REQUEST_IMAGES + 1) |i| {
        if (i > 0) try body.append(std.testing.allocator, ',');
        try body.appendSlice(std.testing.allocator, "{\"type\":\"image_url\",\"image_url\":{\"url\":\"http://x/y.png\"}}");
    }
    try body.appendSlice(std.testing.allocator, "]}]}");
    var t = try testWire(body.items, .openai);
    defer t.parsed.deinit();
    defer t.wire.deinit();
    var want_buf: [96]u8 = undefined;
    const want = try std.fmt.bufPrint(&want_buf, "the request carries {d} images; at most {d} are accepted", .{ chat_mod.MAX_REQUEST_IMAGES + 1, chat_mod.MAX_REQUEST_IMAGES });
    try std.testing.expectEqualStrings(want, t.wire.media.refusal().?.text());
}

test "a long agent session's 65 images pass the image limit" {
    var body = std.ArrayList(u8).empty;
    defer body.deinit(std.testing.allocator);
    try body.appendSlice(std.testing.allocator, "{\"messages\":[{\"role\":\"user\",\"content\":[");
    for (0..65) |i| {
        if (i > 0) try body.append(std.testing.allocator, ',');
        try body.appendSlice(std.testing.allocator, "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACQAAAAUCAIAAABqGAhiAAAALUlEQVR42mM4ISdHBpIjSxvDqGWjlo1aNswsI0/fhw8iZKBRy0YtG7VsmFkGAGi0hC5b6HVfAAAAAElFTkSuQmCC\"}}");
    }
    try body.appendSlice(std.testing.allocator, "]}]}");
    var t = try testWire(body.items, .openai);
    defer t.parsed.deinit();
    defer t.wire.deinit();
    try std.testing.expect(t.wire.media.refusal() == null);
}

/// Serialize a std.json.Value to JSON text, appending to buf.
pub fn serializeJsonValue(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), value: std.json.Value) !void {
    switch (value) {
        .null => try buf.appendSlice(allocator, "null"),
        .bool => |b| try buf.appendSlice(allocator, if (b) "true" else "false"),
        .integer => |i| {
            var num_buf: [24]u8 = undefined;
            const s = std.fmt.bufPrint(&num_buf, "{d}", .{i}) catch "0";
            try buf.appendSlice(allocator, s);
        },
        .float => |f| {
            var num_buf: [32]u8 = undefined;
            const s = std.fmt.bufPrint(&num_buf, "{d}", .{f}) catch "0";
            try buf.appendSlice(allocator, s);
        },
        .string => |s| {
            const escaped = try jsonEscape(allocator, s);
            defer allocator.free(escaped);
            try buf.appendSlice(allocator, escaped);
        },
        .array => |arr| {
            try buf.append(allocator, '[');
            for (arr.items, 0..) |item, idx| {
                if (idx > 0) try buf.append(allocator, ',');
                try serializeJsonValue(allocator, buf, item);
            }
            try buf.append(allocator, ']');
        },
        .object => |obj| {
            try buf.append(allocator, '{');
            var iter = obj.iterator();
            var first = true;
            while (iter.next()) |entry| {
                if (!first) try buf.append(allocator, ',');
                first = false;
                const ek = try jsonEscape(allocator, entry.key_ptr.*);
                defer allocator.free(ek);
                try buf.appendSlice(allocator, ek);
                try buf.append(allocator, ':');
                try serializeJsonValue(allocator, buf, entry.value_ptr.*);
            }
            try buf.append(allocator, '}');
        },
        .number_string => |s| try buf.appendSlice(allocator, s),
    }
}

/// Convert Anthropic tools format to OpenAI tools format for chat template compatibility.
pub fn buildOpenAIToolsJson(allocator: std.mem.Allocator, tools_array: std.json.Array) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    try buf.append(allocator, '[');
    for (tools_array.items, 0..) |tool_val, i| {
        if (i > 0) try buf.append(allocator, ',');
        if (tool_val != .object) continue;
        const tool = tool_val.object;
        const name = if (tool.get("name")) |v| (if (v == .string) v.string else "") else "";
        const desc = if (tool.get("description")) |v| (if (v == .string) v.string else "") else "";
        const esc_n = try jsonEscape(allocator, name);
        defer allocator.free(esc_n);
        const esc_d = try jsonEscape(allocator, desc);
        defer allocator.free(esc_d);
        try buf.appendSlice(allocator, "{\"type\":\"function\",\"function\":{\"name\":");
        try buf.appendSlice(allocator, esc_n);
        try buf.appendSlice(allocator, ",\"description\":");
        try buf.appendSlice(allocator, esc_d);
        try buf.appendSlice(allocator, ",\"parameters\":");
        if (tool.get("input_schema")) |schema_val| {
            try serializeJsonValue(allocator, &buf, schema_val);
        } else {
            try buf.appendSlice(allocator, "{}");
        }
        try buf.appendSlice(allocator, "}}");
    }
    try buf.append(allocator, ']');
    return try buf.toOwnedSlice(allocator);
}

// ── Tests ──

const testing = std.testing;

test "debug body log bounds BINARY but never truncates text (multipart PNG class)" {
    // `/v1/images/edits` is the first binary body we serve. `logHttpBody` dumped
    // whatever it got verbatim, so at --log-level debug one image upload wrote
    // raw PNG bytes into the log — and a body near the 64 MB request cap blew
    // straight through the 32 MB rotation, taking the post-mortem file with it.
    //
    // Text bodies must stay WHOLE: reading a full request body out of the debug
    // log is the documented way to reproduce a tool-calling bug.
    try testing.expect(bodyIsText("{\"model\":\"x\",\"prompt\":\"hi\"}"));
    try testing.expect(bodyIsText("multi\nline\twith\r\nwhitespace"));
    try testing.expect(bodyIsText("utf8 stays text: é ✓ 🎉")); // multibyte, not control
    try testing.expect(!bodyIsText("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"));
    // The real shape: text multipart framing wrapped around a binary payload.
    try testing.expect(!bodyIsText("--B\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\n\x89PNG\x00\x01\r\n--B--"));

    // A binary preview is bounded AND strictly printable ASCII.
    var buf: [64]u8 = undefined;
    const png = "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\xffbinary tail";
    const p = bodyPreview(&buf, png);
    try testing.expect(p.len <= buf.len);
    for (p) |c| try testing.expect(c == '\n' or c == '\r' or c == '\t' or (c >= 0x20 and c < 0x7f));

    // Larger than the sink → truncated, never streamed.
    const big = try testing.allocator.alloc(u8, 1024 * 1024);
    defer testing.allocator.free(big);
    @memset(big, 0);
    try testing.expectEqual(@as(usize, buf.len), bodyPreview(&buf, big).len);
    // The cap is the SMALLER of the caller's buffer and the byte limit, so a
    // generous buffer can't reintroduce the megabyte dump.
    var huge: [BODY_LOG_LIMIT * 2]u8 = undefined;
    try testing.expectEqual(@as(usize, BODY_LOG_LIMIT), bodyPreview(&huge, big).len);
}

test "extractJsonField extracts array" {
    const body =
        \\{"messages":[{"role":"user","content":"hi"}],"temperature":0.7}
    ;
    const result = extractJsonField(body, "messages").?;
    try testing.expect(std.mem.startsWith(u8, result, "["));
    try testing.expect(std.mem.endsWith(u8, result, "]"));
}

test "extractJsonField extracts nested object" {
    const body =
        \\{"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}}
    ;
    const result = extractJsonField(body, "response_format").?;
    try testing.expect(std.mem.startsWith(u8, result, "{"));
    try testing.expect(std.mem.endsWith(u8, result, "}"));
}

test "extractJsonField returns null for missing field" {
    const body = "{\"messages\":[]}";
    try testing.expect(extractJsonField(body, "tools") == null);
}

test "extractJsonField handles escaped quotes in strings" {
    const body =
        \\{"tools":[{"type":"function","function":{"name":"say_\"hello\""}}]}
    ;
    const result = extractJsonField(body, "tools").?;
    try testing.expect(std.mem.startsWith(u8, result, "["));
    try testing.expect(std.mem.endsWith(u8, result, "]"));
}

test "jsonEscape basic string" {
    const allocator = testing.allocator;
    const result = try jsonEscape(allocator, "hello");
    defer allocator.free(result);
    try testing.expectEqualStrings("\"hello\"", result);
}

test "jsonEscape special characters" {
    const allocator = testing.allocator;
    const result = try jsonEscape(allocator, "line1\nline2\t\"quoted\"\\back");
    defer allocator.free(result);
    try testing.expectEqualStrings("\"line1\\nline2\\t\\\"quoted\\\"\\\\back\"", result);
}

test "jsonEscape control characters" {
    const allocator = testing.allocator;
    const input = &[_]u8{ 0x01, 0x02 };
    const result = try jsonEscape(allocator, input);
    defer allocator.free(result);
    try testing.expectEqualStrings("\"\\u0001\\u0002\"", result);
}

test "jsonEscape empty string" {
    const allocator = testing.allocator;
    const result = try jsonEscape(allocator, "");
    defer allocator.free(result);
    try testing.expectEqualStrings("\"\"", result);
}

test "safeAutoContext leaves headroom below the memory ceiling and rounds to 1024" {
    // The raw ceiling is the largest context that FITS. Running at exactly that
    // leaves nothing for the prefix cache to grow into, a second model, or
    // another app — so admit only `auto_ctx_safety_pct` of it.
    try testing.expectEqual(@as(u32, 88064), safeAutoContext(94729)); // 93% = 88098 -> 1024-floor
    try testing.expectEqual(@as(u32, 29696), safeAutoContext(32768)); // 93% = 30474 -> 1024-floor
    // Never rounds down to zero on a tiny ceiling.
    try testing.expect(safeAutoContext(1000) > 0);
    try testing.expect(safeAutoContext(1) > 0);
    // Monotonic, and always strictly inside the ceiling for real-sized contexts.
    var prev: u32 = 0;
    for ([_]u32{ 4096, 8192, 16384, 32768, 65536, 94729, 262144 }) |raw| {
        const got = safeAutoContext(raw);
        try testing.expect(got <= raw);
        try testing.expect(got >= prev);
        prev = got;
    }
}

test "autoContextFor: the safety margin applies to MEMORY, never to the model's own max" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 0;

    // A tiny model whose `max_position_embeddings` is far below anything memory
    // could constrain: it must get its FULL declared context, un-margined.
    // (Regression: applying the margin AFTER the model-max clamp shaved context off a
    // 131,072-token checkpoint that fits in RAM with room to spare.)
    var small = model_mod.ModelConfig{};
    small.max_position_embeddings = 4096;
    small.num_attention_heads = 8;
    small.num_hidden_layers = 4;
    small.num_key_value_heads = 2;
    small.head_dim = 64;
    small.hidden_size = 512;
    small.intermediate_size = 1024;
    try testing.expectEqual(@as(u32, 4096), autoContextFor(&small));

    // No declared max: fall back to the margined memory ceiling.
    var unbounded = small;
    unbounded.max_position_embeddings = 0;
    try testing.expect(autoContextFor(&unbounded) > 0);
    const memory_ctx = memoryContextAt(&unbounded, 96 << 30, 40 << 30);
    try testing.expectEqual(safeAutoContext(memory_ctx), autoContextFrom(memory_ctx, unbounded.contextCap()));
}

test "every served model advertises 93% of the memory ceiling, capped at its own maximum" {
    const t = testing;
    var qwen = try model_mod.parseConfigFromJson(t.allocator, @embedFile("fixtures/model-configs/qwen4_exp.json"));
    defer qwen.deinit(t.allocator);
    var mimo = try model_mod.parseConfigFromJson(t.allocator, @embedFile("fixtures/model-configs/mimo_v2.json"));
    defer mimo.deinit(t.allocator);
    var glm = try model_mod.parseConfigFromJson(t.allocator, @embedFile("fixtures/glm5_config.json"));
    defer glm.deinit(t.allocator);
    for ([_]*const model_mod.ModelConfig{ &qwen, &mimo, &glm }) |cfg| {
        try t.expectEqual(@as(u32, 1_048_576), cfg.contextCap());
        try t.expectEqual(@as(u32, 88064), autoContextFrom(94729, cfg.contextCap()));
        try t.expectEqual(cfg.contextCap(), autoContextFrom(4_000_000, cfg.contextCap()));
        try t.expectEqual(@as(u32, 974_848), autoContextFrom(1_048_576, cfg.contextCap()));
    }
}

test "getEffectiveContextLength returns the PINNED value, not a fresh memory reading" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 0; // auto

    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 131072;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 42;
    config.num_key_value_heads = 2;
    config.head_dim = 256;

    // Once pinned, the advertised context must NOT move — clients budget
    // against it, and a fresh memory reading drifts with system load.
    config.pinned_context = 12345;
    try testing.expectEqual(@as(u32, 12345), getEffectiveContextLength(&config));
    try testing.expectEqual(@as(u32, 12345), getEffectiveContextLength(&config));

    // ...but an explicit --ctx-size still wins.
    server_config.max_context_size = 4096;
    try testing.expectEqual(@as(u32, 4096), getEffectiveContextLength(&config));
}

test "pinAutoContext is idempotent and a no-op under --ctx-size" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;

    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 131072;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 42;
    config.num_key_value_heads = 2;
    config.head_dim = 256;

    server_config.max_context_size = 0;
    const first = pinAutoContext(&config);
    try testing.expect(first > 0);
    try testing.expectEqual(first, config.pinned_context);
    // Second call must not re-read memory and shift the value.
    try testing.expectEqual(first, pinAutoContext(&config));

    // Explicit --ctx-size: never overwrite, always report the override.
    var manual = model_mod.ModelConfig{};
    manual.max_position_embeddings = 131072;
    server_config.max_context_size = 8192;
    try testing.expectEqual(@as(u32, 8192), pinAutoContext(&manual));
    try testing.expectEqual(@as(u32, 0), manual.pinned_context);
    try testing.expect(manual.pinned_prefill_chunk > 0);
}

test "getEffectiveContextLength uses ctx_size override" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;

    server_config.max_context_size = 4096;
    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 32768;
    try testing.expectEqual(@as(u32, 4096), getEffectiveContextLength(&config));
}

test "getEffectiveContextLength computes safe default from GPU memory" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;

    server_config.max_context_size = 0;
    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 131072;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 42;
    config.num_key_value_heads = 2;
    config.head_dim = 256;
    // Should compute a value based on GPU memory, not hardcoded 16K
    const computed = getEffectiveContextLength(&config);
    try testing.expect(computed > 0);
    try testing.expect(computed <= config.max_position_embeddings);

    // Explicit --ctx-size overrides the computed default
    server_config.max_context_size = 32768;
    try testing.expectEqual(@as(u32, 32768), getEffectiveContextLength(&config));
}

test "safeContextForBudget reserves the hot-cache budget (2026-06-19 OOM regression)" {
    const GB: u64 = 1 << 30;
    // A footprint that OOM'd a 16 GB Mac: 32 layers, 4 kv heads, head_dim 256.
    const kv_per_tok: u64 = 32 * 2 * 4 * 256 * 2; // 131072
    const work_per_tok: u64 = 8 * 9216 * 2; //        147456
    const per_tok: u64 = kv_per_tok + work_per_tok;

    // Mid-session: ~6 GB resident (model + a partly-filled hot cache), 2 GB cache cap.
    const ws: u64 = 12 * GB; // ~Metal recommended working set on a 16 GB Mac
    const active: u64 = 6 * GB;
    const cache: u64 = 2 * GB;

    const with_reserve = safeContextForBudget(ws, active, cache, per_tok, 0);
    const without_reserve = safeContextForBudget(ws, active, 0, per_tok, 0);

    // The whole point of the fix: subtracting the hot-cache budget shrinks the
    // reported context so it still fits once the cache fills (it didn't before
    // — auto-ctx was computed against an empty cache and later collided with it).
    try testing.expect(with_reserve < without_reserve);
    // Still usable, never floored to nothing.
    try testing.expect(with_reserve > 1024);
}

test "an explicitly raised iogpu.wired_limit_mb is a FLOOR under the ceiling" {
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    // Bar: a raised sysctl lifts the free-RAM term and nothing else; every other arch and the
    // default sysctl keep the old expression byte for byte.
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const mb: u64 = 1024 * 1024;
    const total_ram: u64 = 131_072 * mb;
    const working_set: u64 = 120_000 * mb;
    const saved = wired_limit_mb_override;
    defer wired_limit_mb_override = saved;
    wired_limit_mb_override = 120_000;

    try t.expectEqual(@as(u64, 1 << 30), WIRED_LIMIT_MARGIN_BYTES);
    try t.expectEqual(@as(u64, 118_976), wiredLimitFloor(120_000 * mb, total_ram, WIRED_LIMIT_MARGIN_BYTES) / mb);
    // The macOS default (75% of RAM) and anything under it declares nothing; and however
    // absurd the sysctl, never plan within the margin of physical RAM.
    try t.expectEqual(@as(u64, 0), wiredLimitFloor(98_304 * mb, total_ram, WIRED_LIMIT_MARGIN_BYTES));
    try t.expectEqual(@as(u64, 0), wiredLimitFloor(0, total_ram, WIRED_LIMIT_MARGIN_BYTES));
    try t.expectEqual(@as(u64, 0), wiredLimitFloor(120_000 * mb, 0, WIRED_LIMIT_MARGIN_BYTES));
    try t.expectEqual(total_ram - WIRED_LIMIT_MARGIN_BYTES, wiredLimitFloor(400 * (1 << 30), total_ram, WIRED_LIMIT_MARGIN_BYTES));

    const box96: u64 = 96_000 * mb;
    const six: u64 = 6 << 30;
    try t.expectEqual(@as(u64, 90_000) * mb - six, wiredLimitFloor(90_000 * mb, box96, six));
    try t.expectEqual(box96 - six, wiredLimitFloor(96_000 * mb, box96, six));
    try t.expectEqual(@as(u64, 90_000) * mb - WIRED_LIMIT_MARGIN_BYTES, wiredLimitFloor(90_000 * mb, box96, WIRED_LIMIT_MARGIN_BYTES));

    // Gated: every other arch takes floor 0, which is `physicalMemoryCeiling` exactly.
    var cfg = qwen4ExpLive364kConfig();
    try t.expect(wiredCeilingFloorForRam(&cfg, total_ram) > 0);
    try t.expectEqual(@as(u64, 0), wiredCeilingFloorForRam(null, total_ram));
    for ([_][]const u8{"not_a_served_arch"}) |mt| {
        var other = cfg;
        other.model_type = mt;
        try t.expect(!other.longCtxGated());
        try t.expectEqual(@as(u64, 0), wiredCeilingFloorForRam(&other, total_ram));
    }
    const footprint: u64 = 87_722 * mb;
    for ([_]u64{ 0, 12_934 * mb, 400 * (1 << 30) }) |free| {
        try t.expectEqual(
            physicalMemoryCeiling(working_set, footprint, free),
            gpuCeilingWithWiredFloor(working_set, footprint, free, 0),
        );
    }

    // A 448k warm append that the free-RAM term refused fits under the floor.
    const kv_bits: u64 = 8;
    const seq: u64 = 448_604;
    const max_tokens: u32 = @intCast(getEffectiveContextLength(&cfg) - seq);
    const shared = WarmPrefix{ .matched_tokens = 448_531, .capacity_tokens = 448_531, .will_donate = false };
    const needed = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, 512, shared);
    try t.expectEqual(@as(u64, 11_456), needed / mb);
    try t.expect(needed > gpuCeilingWithWiredFloor(working_set, footprint, 10_000 * mb, 0) -| footprint);
    const lifted = gpuCeilingWithWiredFloor(working_set, footprint, 12_934 * mb, wiredCeilingFloorForRam(&cfg, total_ram));
    try t.expectEqual(@as(u64, 31_254), (lifted -| footprint) / mb);
    try t.expect(needed <= lifted -| footprint);

    // Under a declared limit the margin holds however much RAM is free: a load on a quiet box saw the
    // working set, the margin gone, and proved a context its own requests then refused.
    const floor = wiredCeilingFloorForRam(&cfg, total_ram);
    try t.expectEqual(floor, gpuCeilingWithWiredFloor(working_set, 0, working_set, floor));
    try t.expectEqual(floor, gpuCeilingWithWiredFloor(working_set, footprint, 400 * (1 << 30), floor));
}

test "parseWiredMarginGib accepts 1..32 and names a bad value" {
    const t = std.testing;
    try t.expectEqual(@as(u64, 1) << 30, try parseWiredMarginGib("1"));
    try t.expectEqual(@as(u64, 8) << 30, try parseWiredMarginGib("8"));
    try t.expectEqual(@as(u64, 6) << 30, try parseWiredMarginGib("6"));
    try t.expectEqual(@as(u64, 2) << 30, try parseWiredMarginGib("2"));
    try t.expectEqual(@as(u64, 32) << 30, try parseWiredMarginGib("32"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib("0"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib("33"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib("nope"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib(""));
}

test "a warm append is billed the rows it ALLOCATES, not the rows it already holds" {
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    // Bar: a checked-out restore credits the rows it holds even when the append outgrows the
    // buffer; the grow bills the new capacity plus one eval window of old buffers.
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const mb: u64 = 1024 * 1024;
    const cfg = qwen4ExpLive364kConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 448_604;
    const matched: u64 = 448_531;
    const chunk: u64 = 512;
    const max_tokens: u32 = @intCast(getEffectiveContextLength(&cfg) - seq);
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg)), seq);

    // Pessimal reading of the entry: capacity exactly the rows it matched, so the append DOES
    // grow every attention layer.
    const grows = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = matched, .will_donate = true };
    const g = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, grows);
    try t.expectEqual(matched *| kv_per_tok, g.shared_resident_bytes);
    try t.expectEqual((reserved - matched) *| kv_per_tok, g.reserved_kv_bytes +| (seq - matched) *| kv_per_tok);
    // ...plus one eval window of old buffers: qwen4_exp caches 12 of 48 layers, one per window.
    try t.expectEqual(@as(u32, 1), attnLayersPerEvalWindow(&cfg, transformer_mod.Transformer.MOE_EVAL_EVERY_N_LAYERS));
    try t.expectEqual(@as(u64, 465), g.grow_coexist_bytes / mb);
    // The bill at the narrowest and a wider width.
    try t.expectEqual(@as(u64, 5_057), prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, grows) / mb);
    try t.expectEqual(@as(u64, 6_548), prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, 2048, grows) / mb);

    // A buffer the previous turn sized above this prompt but below the reservation is grown to
    // it before the prefill writes, one attention layer per eval: one window of old rows.
    const fits = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = 450_048, .will_donate = true };
    const f = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, fits);
    try t.expect(reserved > 450_048);
    try t.expectEqual(450_048 *| kv_per_tok / 12, f.grow_coexist_bytes);
    try t.expectEqual(@as(u64, 5_059), prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, fits) / mb);

    // Byte-identical arms: a SHARED restore is copied whole by the first append, so it credits
    // nothing and bills no window either; a COLD prompt has nothing to credit.
    const shared = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = matched, .will_donate = false };
    const sh = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, shared);
    try t.expectEqual(@as(u64, 0), sh.shared_resident_bytes);
    try t.expectEqual(@as(u64, 0), sh.grow_coexist_bytes);
    const cold = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, .{});
    try t.expectEqual(@as(u64, 0), cold.shared_resident_bytes);
    try t.expectEqual(@as(u64, 0), cold.grow_coexist_bytes);
}

test "physicalMemoryCeiling caps the static GPU max by real free RAM (#64 docker OOM)" {
    const GB: u64 = 1 << 30;
    const static_max: u64 = 115 * GB; // Metal max_recommended_working_set_size, 128 GB Mac

    // Idle: MLX holds 26 GB (active) + 5 GB (reclaimable cache), ~84 GB free.
    // 31 + 84 = 115 ⇒ the static device max still binds — NO regression when the
    // machine is uncontended (auto-context stays generous).
    try testing.expectEqual(static_max, physicalMemoryCeiling(static_max, 31 * GB, 84 * GB));

    // #64: a docker-compose stack (firecrawl/rabbitmq/postgres/playwright) holds
    // ~45 GB. MLX holds 26 GB, only ~35 GB is physically free. The static max is
    // BLIND to docker, but the ceiling must collapse to 26 + 35 = 61 GB — otherwise
    // a large MoE prefill is admitted and Metal throws an uncatchable OOM.
    try testing.expectEqual(@as(u64, 61 * GB), physicalMemoryCeiling(static_max, 26 * GB, 35 * GB));

    // Saturating add: a bogus/huge footprint reading can never wrap to a tiny ceiling.
    try testing.expectEqual(static_max, physicalMemoryCeiling(static_max, std.math.maxInt(u64), GB));
}

test "auto-context tightens under external memory pressure vs the pre-fix static ceiling (#64)" {
    const GB: u64 = 1 << 30;
    // A 35B-A3B MoE footprint (per-token KV + working envelope).
    const per_tok: u64 = 328 * 1024;
    const static_max: u64 = 115 * GB;
    const active: u64 = 26 * GB;
    const mlx_cache: u64 = 4 * GB;
    const hot_cache_reserve: u64 = 2 * GB;
    const max_pos: u32 = 262144;

    // Pre-fix behaviour: budget straight against the static device max, blind to
    // free RAM — the exact code path that OOM'd #64.
    const prefix_ctx = safeContextForBudget(static_max, active, hot_cache_reserve, per_tok, max_pos);

    // Post-fix, idle (~84 GB free): ceiling ≈ static max ⇒ context essentially unchanged.
    const idle_ceiling = physicalMemoryCeiling(static_max, active + mlx_cache, 84 * GB);
    const idle_ctx = safeContextForBudget(idle_ceiling, active, hot_cache_reserve, per_tok, max_pos);
    try testing.expect(idle_ctx >= prefix_ctx * 9 / 10); // within 10% of the old generous value

    // Post-fix, docker holds ~45 GB (~35 GB free): ceiling collapses, so the safe
    // context is far smaller than the pre-fix static-max computation would allow.
    const busy_ceiling = physicalMemoryCeiling(static_max, active + mlx_cache, 35 * GB);
    const busy_ctx = safeContextForBudget(busy_ceiling, active, hot_cache_reserve, per_tok, max_pos);
    try testing.expect(busy_ctx < prefix_ctx / 2);
    try testing.expect(busy_ctx > 1024); // still usable, never floored to nothing
}

test "safeContextForBudget floors when memory is exhausted and caps at max_pos" {
    const GB: u64 = 1 << 30;
    const per_tok: u64 = 278528;
    // working set already below what's resident → no room → minimum floor.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(4 * GB, 5 * GB, 0, per_tok, 0));
    // active + reserve exceeds the ceiling → floor.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(8 * GB, 6 * GB, 4 * GB, per_tok, 0));
    // Plenty of memory, but capped at the model's max position embeddings.
    try testing.expectEqual(@as(u32, 4096), safeContextForBudget(256 * GB, 0, 0, per_tok, 4096));
    // Degenerate per_tok never divides by zero.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(8 * GB, 0, 0, 0, 0));
}

test "kvBytesPerTokenAtBits: the KV bill follows the CONFIGURED quant width" {
    const t = std.testing;
    // Dense: 16 caching layers x 4 kv heads x 256 head_dim x (K+V) x 2 B.
    const dense: u64 = 16 * 4 * 256 * 2 * 2; // 65536

    // Dense/off bills fp16 verbatim — no change for anyone not using --kv-quant.
    try t.expectEqual(dense, kvBytesPerTokenAtBits(dense, 16));
    try t.expectEqual(dense, kvBytesPerTokenAtBits(dense, 32));

    // Quantized widths carry the group scale/bias overhead (+0.5 bit/elem), the
    // SAME expression prefillMemoryNeeded bills — the sizer and the admission
    // guard must not disagree about what one token costs.
    try t.expectEqual(dense * 9 / 32, kvBytesPerTokenAtBits(dense, 4)); // 18432
    try t.expectEqual(dense * 13 / 32, kvBytesPerTokenAtBits(dense, 6));
    try t.expectEqual(dense * 17 / 32, kvBytesPerTokenAtBits(dense, 8));
    // The bill is what the cache stores: `bits` per element plus a bf16 scale and bias per group of 64.
    for ([_]u64{ 4, 6, 8 }) |bits| {
        const elems = dense / 2;
        try t.expectEqual(elems * bits / 8 + elems / 64 * 4, kvBytesPerTokenAtBits(dense, bits));
    }
    // 4-bit is what makes the 16 GB tier reachable: under a third of dense.
    try t.expect(kvBytesPerTokenAtBits(dense, 4) * 3 < dense);
}

test "auto-context on a 16 GB profile: activations are a chunk reserve, not a per-token multiplier" {
    const t = std.testing;
    // Qwen3.8-27B on a 16 GB Mac: ~10.6 GB reachable Metal working set, an
    // 8.6 GB iQ-MLX pack resident, --kv-quant 4, and the small prefill chunk
    // the machine can actually afford.
    const ceiling: u64 = 10_600 * 1000 * 1000;
    const weights: u64 = 8_600 * 1000 * 1000;
    const dense_kv: u64 = 16 * 4 * 256 * 2 * 2;
    const per_tok = kvBytesPerTokenAtBits(dense_kv, 4);

    var cfg = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 4;
    cfg.head_dim = 256;
    cfg.hidden_size = 5120;
    cfg.intermediate_size = 17408;
    cfg.intermediate_size_declared = true;
    // The real checkpoint's hybrid geometry: 64 layers, every 4th full
    // attention, 48 GatedDeltaNet layers whose chunk-wide q/k/v streams are
    // two thirds of this model's measured prefill transient.
    cfg.num_hidden_layers = 64;
    cfg.full_attention_interval = 4;
    cfg.linear_num_key_heads = 16;
    cfg.linear_num_value_heads = 48;

    const reserve = prefillTransientReserve(&cfg, 4, 512);
    const got = safeContextForBudget(ceiling, weights, reserve, per_tok, 262144);

    // The pre-fix formula billed 8 x max(hidden, ffn) x 2 = 272 KB PER TOKEN on
    // top of a 64 KB/token dense KV — 344 KB/token, which caps this machine at
    // under 4k tokens no matter what the weights cost.
    const old_per_tok: u64 = dense_kv + 8 * 17408 * 2;
    const old = safeContextForBudget(ceiling, weights, 0, old_per_tok, 262144);
    try t.expect(old < 4096);

    // Corrected, the same machine holds a real working context. The band is
    // what the RESERVE is worth, and the reserve is measured: this checkpoint
    // peaks 1.06 GB above steady state at chunk 512 (M4 Max, 2026-08-14), so
    // a sizer that reported the ~51k tokens the first cut of this fix did was
    // spending memory the prefill needs. Halve the chunk and it comes back.
    try t.expect(got > 15_000);
    try t.expect(got < 40_000);

    // The reserve is a CONSTANT of the chunk, so halving the chunk buys context
    // back rather than being invisible.
    const narrow = safeContextForBudget(ceiling, weights, prefillTransientReserve(&cfg, 4, 256), per_tok, 262144);
    try t.expect(narrow > got);
}

test "auto-context never sizes past the ceiling as weights grow" {
    const t = std.testing;
    const ceiling: u64 = 10_600 * 1000 * 1000;
    const dense_kv: u64 = 16 * 4 * 256 * 2 * 2;
    const per_tok = kvBytesPerTokenAtBits(dense_kv, 4);
    var cfg = model_mod.ModelConfig{};
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 4;
    cfg.head_dim = 256;
    cfg.hidden_size = 5120;
    cfg.intermediate_size = 17408;
    cfg.intermediate_size_declared = true;
    const reserve = prefillTransientReserve(&cfg, 4, 512);

    // Under-billing here ends in an uncatchable Metal OOM, not a 400: at every
    // weight size the reported context plus what is already spoken for has to
    // stay inside the ceiling, and the reported figure must fall as the pack
    // grows rather than staying pinned to the checkpoint's max.
    var prev: u32 = std.math.maxInt(u32);
    var weights: u64 = 2_000 * 1000 * 1000;
    while (weights < ceiling) : (weights += 1_000 * 1000 * 1000) {
        const ctx = safeContextForBudget(ceiling, weights, reserve, per_tok, 262144);
        try t.expect(weights + reserve + @as(u64, ctx) * per_tok <= ceiling or ctx == 1024);
        try t.expect(ctx <= prev);
        prev = ctx;
    }
    // Squeezed to nothing it floors rather than reporting a context that cannot exist.
    try t.expectEqual(@as(u32, 1024), safeContextForBudget(ceiling, ceiling, reserve, per_tok, 262144));
}

test "getTimeoutNs computes correctly" {
    const original = server_config.request_timeout_sec;
    defer server_config.request_timeout_sec = original;

    server_config.request_timeout_sec = 300;
    try testing.expectEqual(@as(u64, 300_000_000_000), getTimeoutNs());

    server_config.request_timeout_sec = 0;
    try testing.expectEqual(@as(u64, 0), getTimeoutNs());
}

pub fn renderQwenForResponsesTest(a: std.mem.Allocator, messages: []const chat_mod.Message) ![]const u8 {
    const config = chat_mod.ChatConfig{
        .chat_template = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"),
        .bos_token = null,
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = a,
    };
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\",\"properties\":{\"id\":{\"type\":\"integer\"}}}}}]";
    return chat_mod.renderChatTemplate(a, messages, &config, tools, null, true, null, false);
}

test "Responses: a stored reasoning turn renders as the model's own think block" {
    const a = testing.allocator;
    const calls = [_]chat_mod.ToolCall{.{ .id = "c1", .name = "lookup", .arguments = "{\"id\":1}" }};
    // The Qwen3.8 template reads only reasoning_content; an inline <think> in content renders as literal text.
    const inline_think = [_]chat_mod.Message{
        .{ .role = "user", .content = "compare" },
        .{ .role = "assistant", .content = "<think>inspect it</think>", .tool_calls = &calls },
        .{ .role = "tool", .content = "done", .tool_call_id = "c1" },
    };
    const field = [_]chat_mod.Message{
        inline_think[0],
        .{ .role = "assistant", .content = "", .reasoning_content = "inspect it", .tool_calls = &calls },
        inline_think[2],
    };
    const r_inline = try renderQwenForResponsesTest(a, &inline_think);
    defer a.free(r_inline);
    const r_field = try renderQwenForResponsesTest(a, &field);
    defer a.free(r_field);
    try testing.expect(std.mem.indexOf(u8, r_field, "<think>\ninspect it\n</think>\n\n<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, r_inline, "<think>\ninspect it\n</think>") == null);
}

test "content-array text parts JOIN in order — plan-mode's [prompt, reminder] shape (issue #195)" {
    // opencode plan mode appends its system-reminder as a SECOND text part on
    // the user message. Last-wins parsing kept only the reminder, so the model
    // saw no user prompt at all.
    const body =
        \\[{"type":"text","text":"analyze project"},
        \\ {"type":"image_url","image_url":{"url":"data:x"}},
        \\ {"type":"text","text":"<system-reminder>plan mode</system-reminder>"}]
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const j = try joinedTextParts(testing.allocator, parsed.value.array.items);
    defer if (j.owned) testing.allocator.free(j.text);
    try testing.expect(j.owned);
    try testing.expectEqualStrings("analyze project\n<system-reminder>plan mode</system-reminder>", j.text);
}

test "joinedTextParts: single text part borrows; empty and non-text parts ignored" {
    const body =
        \\[{"type":"text","text":""}, {"type":"input_audio"}, {"type":"text","text":"hello"}, {"nope":1}]
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const j = try joinedTextParts(testing.allocator, parsed.value.array.items);
    try testing.expect(!j.owned);
    try testing.expectEqualStrings("hello", j.text);

    const none = try joinedTextParts(testing.allocator, &.{});
    try testing.expect(!none.owned);
    try testing.expectEqualStrings("", none.text);
}

fn cacheKeyOf(body: []const u8) !u64 {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    return requestCacheKey(parsed.value.object);
}

test "canonicalRole: developer is the system turn; every other role is itself" {
    try testing.expectEqualStrings("system", chat_mod.canonicalRole("developer"));
    try testing.expectEqualStrings("system", chat_mod.canonicalRole("system"));
    try testing.expectEqualStrings("tool", chat_mod.canonicalRole("tool"));
    try testing.expectEqualStrings("assistant", chat_mod.canonicalRole("assistant"));
}

test "requestCacheKey: prompt_cache_key > metadata.user_id > system prompt > anonymous" {
    const sys_str = try cacheKeyOf(
        \\{"messages":[{"role":"system","content":"You are S."},{"role":"user","content":"hi"}]}
    );
    const sys_parts = try cacheKeyOf(
        \\{"messages":[{"role":"system","content":[{"type":"text","text":"You are S."}]}]}
    );
    const anth = try cacheKeyOf(
        \\{"system":[{"type":"text","text":"You are S."}],"messages":[{"role":"user","content":"hi"}]}
    );
    const resp = try cacheKeyOf(
        \\{"instructions":"You are S.","input":"hi"}
    );
    try testing.expect(sys_str != 0);
    try testing.expectEqual(sys_str, sys_parts);
    try testing.expectEqual(sys_str, anth);
    try testing.expectEqual(sys_str, resp);
    // OpenAI's reasoning-model spelling of the system turn (pi sends it).
    try testing.expectEqual(sys_str, try cacheKeyOf(
        \\{"messages":[{"role":"developer","content":"You are S."},{"role":"user","content":"hi"}]}
    ));
    try testing.expect(sys_str != try cacheKeyOf(
        \\{"messages":[{"role":"system","content":"You are T."}]}
    ));

    const uid = try cacheKeyOf(
        \\{"metadata":{"user_id":"sess-1"},"messages":[{"role":"system","content":"You are S."}]}
    );
    try testing.expect(uid != 0 and uid != sys_str);
    const pck = try cacheKeyOf(
        \\{"prompt_cache_key":"batch","metadata":{"user_id":"sess-1"},"messages":[{"role":"system","content":"You are S."}]}
    );
    try testing.expect(pck != 0 and pck != uid);
    try testing.expectEqual(pck, try cacheKeyOf(
        \\{"prompt_cache_key":"batch","prompt":"doc"}
    ));

    try testing.expectEqual(@as(u64, 0), try cacheKeyOf(
        \\{"messages":[{"role":"user","content":"hi"}]}
    ));
    try testing.expectEqual(@as(u64, 0), try cacheKeyOf(
        \\{"prompt_cache_key":"","prompt":"doc"}
    ));
}

test "settingsPropsJson: /props names the greedy tail and where it came from" {
    for ([_]model_settings.Pick(bool){ .{ .value = false, .source = .default }, .{ .value = true, .source = .model_settings }, .{ .value = true, .source = .flag } }, [_][]const u8{ "default", "model-settings.json", "--mtp-greedy-tail" }) |tail, source| {
        const frag = try settingsPropsJson(testing.allocator, .{ .engine = "mlx", .kv_quant = "8", .prefill_chunk = 8192, .mtp_loaded = true, .mtp_default_on = true, .mtp_acceptance = .exact, .mtp_greedy_tail = tail, .mtp_depth = 0, .mtp_adaptive = true, .max_mtp_ctx = 0, .drafter = "none", .max_concurrent = 1, .prefix_cache_mem_bytes = 0, .prefix_cache_disk_bytes = 0 });
        defer testing.allocator.free(frag);
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, frag[",\"settings\":".len..], .{});
        defer parsed.deinit();
        const mtp = parsed.value.object.get("mtp").?.object;
        try testing.expectEqual(tail.value, mtp.get("greedy_tail").?.bool);
        try testing.expectEqualStrings(source, mtp.get("greedy_tail_source").?.string);
    }
}

test "expertPoolBytes: only DeepSeek-V4.1 takes a ledger pool, within the RAM-wide cap" {
    const ds = model_mod.ModelConfig{ .model_type = "deepseek_v41" };
    try testing.expect(ds.expertPoolBytes() > 0);
    try testing.expect(ds.expertPoolBytes() <= mlxCacheLimitBytes(128 << 30));
    for ([_][]const u8{ "qwen4_exp", "mimo_v2", "glm5_next" }) |arch| {
        const c = model_mod.ModelConfig{ .model_type = arch };
        try testing.expectEqual(@as(u64, 0), c.expertPoolBytes());
    }
}

test "mlxCacheLimitBytes: RAM-proportional cap, 2 GB floor, 8 GB ceiling" {
    const GB: u64 = 1 << 30;
    // MLX's own default is min(1.5 x working_set, 0.95 x RAM) — ~121 GB on a
    // 128 GB Mac (`backend/metal/allocator.cpp`), i.e. it will not trim until
    // the machine is already dead. RAM/16 keeps the pool big enough for
    // step-to-step buffer reuse without letting it become the footprint.
    try testing.expectEqual(8 * GB, mlxCacheLimitBytes(128 * GB));
    try testing.expectEqual(4 * GB, mlxCacheLimitBytes(64 * GB));
    try testing.expectEqual(2 * GB, mlxCacheLimitBytes(32 * GB)); // 2 GB exactly
    try testing.expectEqual(2 * GB, mlxCacheLimitBytes(16 * GB)); // floor
    try testing.expectEqual(2 * GB, mlxCacheLimitBytes(8 * GB)); // floor
    // Ceiling holds above 128 GB (Mac Studio / M3 Ultra 512 GB).
    try testing.expectEqual(8 * GB, mlxCacheLimitBytes(512 * GB));
    // A failed `hw.memsize` read is 0 — never clamp on bad data.
    try testing.expectEqual(@as(u64, 0), mlxCacheLimitBytes(0));
}

/// Pins the cache tiers a bill test is about. The defaults (no RAM tier, an SSD tier) bill differently:
/// SSD-first stages the writer's flush and a ringed arch bills its restore's coexistence.
const TierGuard = struct {
    ram: bool,
    disk: u64,

    fn pin(ram: bool, disk: u64) TierGuard {
        const saved: TierGuard = .{ .ram = prefix_cache_ram_enabled, .disk = prefix_cache_disk_bytes };
        prefix_cache_ram_enabled = ram;
        prefix_cache_disk_bytes = disk;
        return saved;
    }

    fn restore(self: TierGuard) void {
        prefix_cache_ram_enabled = self.ram;
        prefix_cache_disk_bytes = self.disk;
    }
};

test "prefix cache default capacity covers interleaved agent flows" {
    // Claude Code-style clients interleave several conversation roots (main
    // thread, subagents, title generation). With capacity 1, every
    // interleaved request evicted the long system-prompt prefix and forced a
    // full re-prefill per turn.
    try testing.expect(prefix_cache_capacity >= 4);
}

test "prefix cache defaults: no RAM retention, an unnamed SSD tier on" {
    try testing.expect(!prefix_cache_ram_enabled);
    try testing.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
    try testing.expect(!prefix_cache_mem_explicit and !prefix_cache_disk_explicit);
    try testing.expectEqual(kv_disk_cache.DEFAULT_DISK_CAP, prefix_cache_disk_bytes);
}

test "parseToolCallsForRequest coerces args to the schema (server-side chokepoint wiring)" {
    // The chokepoint every HTTP surface routes through must actually invoke the
    // schema coercion — not just parse. This pins the WIRING (a mis-wire that
    // parsed but skipped coerce would reintroduce the string-"False" bug on every
    // surface at once). Uses the exact Hermes-XML `replace_all=False` shape from
    // the live Claude Code failure.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;
    const text = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
        "<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const ra = parsed.value.object.get("replace_all").?;
    try std.testing.expect(ra == .bool); // coerced from the STRING "False"
    try std.testing.expectEqual(false, ra.bool);
    // Null tools_json path must still parse (no coercion, no crash).
    const calls2 = try parseToolCallsForRequest(allocator, text, null, true);
    defer if (calls2) |cs| {
        for (cs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(cs);
    };
    try std.testing.expect(calls2 != null);
}

test "parseToolCallsForRequest hoists a misplaced required param (server-side chokepoint wiring)" {
    // Live 2026-07-13 pi session, gemma-4-26B-A4B: the model buried `path` inside
    // each edits[] item and emitted none at the top level, so pi rejected three
    // consecutive multi-thousand-token generations with
    //   Validation failed for tool "edit": - path: must have required properties path
    // This pins the WIRING — every HTTP surface reaches the hoist through this one
    // function, and the hoist must run BEFORE coercion so the lifted value is
    // type-checked like any other top-level arg.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},"required":["oldText","newText"]}}},"required":["path","edits"]}}}]
    ;
    const text = "<tool_call>{\"name\":\"edit\",\"arguments\":{\"edits\":[{\"newText\":\"new\",\"oldText\":\"old\"," ++
        "\"path\":\"us_presidents/generate_site.sh\"}]}}</tool_call>";

    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const path = parsed.value.object.get("path") orelse return error.PathNotHoisted;
    try std.testing.expectEqualStrings("us_presidents/generate_site.sh", path.string);
    try std.testing.expect(parsed.value.object.get("edits").?.array.items[0].object.get("path") == null);
    try std.testing.expect(chat_mod.toolCallConformsToSchema(allocator, calls[0], tools));

    // The escape hatch covers this correction too: it repairs the MODEL's output
    // (unlike the inferred-name filter, which corrects our own heuristic).
    g_tool_autocorrect = false;
    defer g_tool_autocorrect = true;
    const raw_calls = (try parseToolCallsForRequest(allocator, text, tools, true)).?;
    defer {
        for (raw_calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(raw_calls);
    }
    const raw_parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw_calls[0].arguments, .{});
    defer raw_parsed.deinit();
    try std.testing.expect(raw_parsed.value.object.get("path") == null); // verbatim: still buried
}

test "parseToolCallsForRequest: --no-tool-autocorrect leaves args verbatim" {
    // The escape hatch: with g_tool_autocorrect off, the schema coercion is
    // skipped and the model's value passes through EXACTLY (here the STRING
    // "False", which is what a strict client would then reject — the user's
    // choice). Parse-repair + valid-JSON safety net still run, so args stay
    // well-formed JSON regardless.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;
    const text = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
        "<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>";

    g_tool_autocorrect = false;
    defer g_tool_autocorrect = true; // restore the default for other tests
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const ra = parsed.value.object.get("replace_all").?;
    try std.testing.expect(ra == .string); // NOT coerced — verbatim
    try std.testing.expectEqualStrings("False", ra.string);
}

test "parseToolCallsForRequest: truncated DATA object is not promoted to a tool call (George Washington class)" {
    // Live pi capture 2026-07-13 (Qwen3.6-35B-A3B distilled): generation hit
    // max_tokens midway through a presidents data script. The raw-JSON fallback
    // found the first balanced object — {"name": "George Washington", "num": 1,
    // …} — and the flat-shape synthesis promoted it to a TOOL CALL named
    // "George Washington" (args = every key but "name"). pi answered "Tool
    // George Washington not found", the model retried the identical mega-write,
    // and the session burned two full 16K-token turns making zero progress.
    // A HEURISTICALLY inferred call (no tag syntax — the model never said
    // "tool call") must carry a name the request actually declared; otherwise
    // the text stays visible and finish_reason="length" reaches the client
    // untouched, so its truncation recovery fires instead of a bogus tool loop.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    const text = "Now let me build a generator script that creates all 46 president pages with real historical data:\n\n" ++
        "presidents = [\n" ++
        "  {\"name\": \"George Washington\", \"num\": 1, \"party\": \"None (Federalist-leaning)\", \"term\": \"1789\u{2013}1797\", \"vice\": \"John Adams\"},\n" ++
        "  {\"name\": \"John Adams\", \"num\": 2, \"party\": \"Federalist\",";
    const calls = try parseToolCallsForRequest(allocator, text, tools, true);
    defer if (calls) |cs| {
        for (cs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(cs);
    };
    try std.testing.expect(calls == null);
}

test "parseToolCallsForRequest: raw-JSON call with a DECLARED name still parses" {
    // The counterweight to the George Washington guard: models without a
    // trained tool format (Gemma 3) emit fenced raw-JSON calls, and those must
    // keep working when the name matches a declared tool.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]
    ;
    const text = "```json\n{\"name\": \"write\", \"arguments\": {\"path\": \"a.txt\", \"content\": \"hi\"}}\n```";
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)) orelse
        return error.ExpectedToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqualStrings("write", calls[0].name);
}

test "parseToolCallsForRequest: tag-format call with an UNDECLARED name is kept" {
    // An EXPLICIT tag-format call (<tool_call>…) to a name the request never
    // declared still goes to the client — "tool not found" is model-visible
    // feedback the model can correct from. Only HEURISTIC raw-JSON inference
    // gets schema-name validation; this pins the filter against over-reach.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    const text = "<tool_call>{\"name\":\"searchWeb\",\"arguments\":{\"q\":\"zig\"}}</tool_call>";
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)) orelse
        return error.ExpectedToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqualStrings("searchWeb", calls[0].name);
}

test "parseToolCallsForRequest: parallel_tool_calls=false clamps to the FIRST call" {
    // OpenAI contract: parallel_tool_calls=false means AT MOST ONE call per
    // response (the SDK sets it in strict structured-output mode; those
    // clients execute tool_calls[0] and assume length 1). We can't constrain
    // generation, so the chokepoint clamps post-parse: keep the first, drop
    // the rest — the model re-issues them next round after seeing the first
    // result. Whole valid calls only; nothing partial ships.
    const allocator = std.testing.allocator;
    const text = "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}</tool_call>\n" ++
        "<tool_call>{\"name\":\"write\",\"arguments\":{\"path\":\"b\"}}</tool_call>";

    // allow_parallel=true (default wiring): both calls survive.
    const both = (try parseToolCallsForRequest(allocator, text, null, true)).?;
    defer {
        for (both) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(both);
    }
    try std.testing.expectEqual(@as(usize, 2), both.len);

    // allow_parallel=false: only the FIRST survives, intact.
    const one = (try parseToolCallsForRequest(allocator, text, null, false)).?;
    defer {
        for (one) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(one);
    }
    try std.testing.expectEqual(@as(usize, 1), one.len);
    try std.testing.expectEqualStrings("read", one[0].name);

    // A single call with the flag off passes through untouched.
    const single_text = "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}</tool_call>";
    const single = (try parseToolCallsForRequest(allocator, single_text, null, false)).?;
    defer {
        for (single) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(single);
    }
    try std.testing.expectEqual(@as(usize, 1), single.len);
}

test "parseToolCallsForRequest: MiniCPM5 V3 XML through the real server chokepoint" {
    // Exercises the EXACT function every HTTP dispatch site in this file calls,
    // proving the new dialect is wired all the way through — not just reachable
    // from chat.parseToolCalls in isolation. Uses the Agent shell tool's real
    // declared schema shape.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"shell","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
    ;
    const text = "<function name=\"shell\">\n  <param name=\"command\">git status</param>\n</function>";
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true)) orelse
        return error.ExpectedToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("git status", parsed.value.object.get("command").?.string);
}

test "parseToolCallsForRequest: coercion fires across think on/off × qwen/gemma" {
    // The tool-call parse strips the leading think block BEFORE parsing args, and
    // the two families strip DIFFERENTLY (Qwen `<think>…</think>`, Gemma
    // `<|channel>thought…<channel|>`). Pin that a preceding reasoning block never
    // blocks the schema coercion, for BOTH families, WITH and WITHOUT thinking.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;
    const gtools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;

    const Case = struct { name: []const u8, text: []const u8, tools: []const u8 };
    const cases = [_]Case{
        // Qwen3.6-MoE thinking OFF — Hermes XML, no reasoning block.
        .{ .name = "qwen think-off", .tools = tools, .text = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>" },
        // Qwen3.6-MoE thinking ON — a `<think>…</think>` block precedes the call.
        .{ .name = "qwen think-on", .tools = tools, .text = "<think>\nThe user wants replace_all disabled, so I'll pass it false.\n</think>\n<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>" },
        // Gemma thinking OFF — custom `call:` form, no channel block.
        .{ .name = "gemma think-off", .tools = gtools, .text = "<|tool_call>call:Edit{file_path:<|\"|>/tmp/x<|\"|>,replace_all:False}<tool_call|>" },
        // Gemma thinking ON — a `<|channel>thought…<channel|>` block precedes.
        .{ .name = "gemma think-on", .tools = gtools, .text = "<|channel>thought\nI should disable replace_all for a single edit.<channel|>\n<|tool_call>call:Edit{file_path:<|\"|>/tmp/x<|\"|>,replace_all:False}<tool_call|>" },
    };

    for (cases) |c| {
        const calls = (try parseToolCallsForRequest(allocator, c.text, c.tools, true)) orelse {
            std.debug.print("\n[{s}] no tool call parsed\n", .{c.name});
            return error.NoToolCall;
        };
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try std.testing.expectEqualStrings("Edit", calls[0].name);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        const ra = parsed.value.object.get("replace_all") orelse {
            std.debug.print("\n[{s}] replace_all missing in {s}\n", .{ c.name, calls[0].arguments });
            return error.ArgMissing;
        };
        if (ra != .bool or ra.bool != false) {
            std.debug.print("\n[{s}] replace_all NOT coerced to bool false: {s}\n", .{ c.name, calls[0].arguments });
            return error.NotCoerced;
        }
    }
}

test "toolCallFinishReason preserves truncation over parsed tool calls" {
    // A truncated generation ("length": max_tokens OR request timeout) whose
    // cut-off text still salvaged a partial tool call must KEEP reporting
    // "length" — clients key their truncation recovery (chunk-and-retry
    // nudges) on it. Reporting "tool_calls" hides the cut: the client
    // executes a half-argument call and tells the model IT made a mistake.
    // Live capture 2026-07-03: Qwen3.6-27B 33KB writeFile guillotined by the
    // 300s default timeout arrived as {"path":...} with finish "tool_calls".
    try std.testing.expectEqualStrings("length", toolCallFinishReason("length"));
    // Everything else stays the normal tool-call finish.
    try std.testing.expectEqualStrings("tool_calls", toolCallFinishReason("stop"));
    try std.testing.expectEqualStrings("tool_calls", toolCallFinishReason("tool_calls"));
    try std.testing.expectEqualStrings("tool_calls", toolCallFinishReason("client_disconnect"));
}

test "repetition-loop cuts decline tool-call parsing" {
    // The loop detector can fire inside a tool argument. The response must
    // remain a safe terminal stop without emitting an executable fragment.
    try std.testing.expect(shouldParseToolCalls(null));
    try std.testing.expect(!shouldParseToolCalls("repetition_loop"));
    try std.testing.expect(shouldParseToolCalls("unknown_future_detail"));
}

test "finishDetailsField: the loop cause rides beside finish_reason, and only a known cause reaches the wire" {
    // Absent = the field is not emitted at all, so every ordinary response
    // is byte-identical to what it was before this existed.
    try std.testing.expectEqualStrings("", finishDetailsField("stop", null));
    try std.testing.expectEqualStrings(
        ",\"finish_details\":{\"type\":\"repetition_loop\"}",
        finishDetailsField("stop", "repetition_loop"),
    );
    // An unknown value is DROPPED rather than interpolated: this string is
    // spliced into a JSON literal, and a literal is arbitrary bytes too (the
    // media-gen `sendError` class). A future cause adds an arm here.
    try std.testing.expectEqualStrings("", finishDetailsField("stop", "something new"));
    try std.testing.expectEqualStrings("", finishDetailsField("stop", "\",\"x\":\""));
    // The cause belongs only beside the intentional loop stop.
    try std.testing.expectEqualStrings("", finishDetailsField("length", "repetition_loop"));
    try std.testing.expectEqualStrings("", finishDetailsField("tool_calls", "repetition_loop"));
    try std.testing.expectEqualStrings("", finishDetailsField("client_disconnect", "repetition_loop"));
}

test "nChoicesRejectReason: n>1 earns an honest 400, single-choice spellings pass" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, rejected: bool }{
        // Absent, explicit null, and the two spellings of "one choice" pass.
        .{ .body = "{}", .rejected = false },
        .{ .body = "{\"n\":null}", .rejected = false },
        .{ .body = "{\"n\":1}", .rejected = false },
        .{ .body = "{\"n\":1.0}", .rejected = false },
        // Anything else is a request for multi-choice (or garbage) — reject
        // instead of the silent one-choice no-op.
        .{ .body = "{\"n\":2}", .rejected = true },
        .{ .body = "{\"n\":0}", .rejected = true },
        .{ .body = "{\"n\":\"2\"}", .rejected = true },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const reason = nChoicesRejectReason(parsed.value.object);
        try std.testing.expectEqual(case.rejected, reason != null);
    }
}

test "jsonEscapeOrEmpty: escaping an empty string is OWNED (leak class 2026-07-19)" {
    // The old inline pattern decided ownership by CONTENT
    // (`if (!mem.eql(escaped, "\"\"")) free`), so a legitimately-allocated
    // escape of "" — every all-reasoning generation with empty visible
    // content — matched the OOM-fallback literal and leaked 2 bytes per
    // request. Ownership keys on provenance; std.testing.allocator fails
    // this test on any leak.
    const a = std.testing.allocator;
    const empty = jsonEscapeOrEmpty(a, "");
    try std.testing.expect(empty.owned);
    try std.testing.expectEqualStrings("\"\"", empty.slice);
    if (empty.owned) a.free(empty.slice);

    const text = jsonEscapeOrEmpty(a, "hi \"there\"");
    try std.testing.expect(text.owned);
    try std.testing.expectEqualStrings("\"hi \\\"there\\\"\"", text.slice);
    if (text.owned) a.free(text.slice);
}

test "detokenizeResponseJson escapes arbitrary token bytes (control-byte class)" {
    const allocator = testing.allocator;
    // A decoded token is arbitrary bytes. The old hand-rolled escaper covered
    // five characters and passed everything else through, so a single id whose
    // bytes are below 0x20 shipped a body no JSON parser accepts — live
    // 2026-08-04 while cross-checking a vocabulary id by id.
    const nasty = "a\x01b\x1fc\"d\\e\nf\tg";
    const out = try detokenizeResponseJson(allocator, nasty);
    defer allocator.free(out);
    for (out) |c| try testing.expect(c >= 0x20);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, out, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings(nasty, parsed.value.object.get("content").?.string);

    // Ordinary text is untouched apart from the quoting.
    const plain = try detokenizeResponseJson(allocator, "hello");
    defer allocator.free(plain);
    try testing.expectEqualStrings("{\"content\":\"hello\"}", plain);
}

test "textGenRejectReason: LM-less entries rejected, chat models pass" {
    // Chat-capable targets pass — resident MLX LM, embedded engines, and
    // unloaded stubs (benefit of the doubt until load).
    try std.testing.expect(textGenRejectReason(.{}) == null);
    // A READY entry with no LM resident rejects instead of crashing.
    try std.testing.expect(textGenRejectReason(.{ .has_text_lm = false }) != null);
}

test "kvBytesPerToken bills only the CACHING layers, at the arch's own K and V widths" {
    const t = std.testing;
    // Uniform arch: every layer caches, K and V are both head_dim wide.
    var dense = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    dense.num_hidden_layers = 24;
    dense.num_key_value_heads = 8;
    dense.head_dim = 128;
    try t.expectEqual(@as(u32, 24), dense.attnCacheLayerCount());
    try t.expectEqual(@as(u64, 24 * 8 * 256 * 2), dense.kvBytesPerToken());

    // A hybrid with 6 of 24 layers attention, whose MLA stores
    // a 192-wide key against a 128-wide value. Billing it as a uniform arch
    // charged 196608 B/token against a real 61440 — a 3.2x over-bill that
    // pinned auto-context to under a third of what fits.
    var ling = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    ling.num_hidden_layers = 24;
    ling.num_attention_heads = 16;
    ling.num_key_value_heads = 16;
    ling.head_dim = 128;
    ling.full_attention_interval = 4;
    ling.mla_kv_lora_rank = 512;
    ling.mla_qk_nope_head_dim = 128;
    ling.mla_qk_rope_head_dim = 64;
    ling.mla_v_head_dim = 128;
    try t.expectEqual(@as(u32, 6), ling.attnCacheLayerCount());
    try t.expectEqual(@as(u64, 6 * 16 * (192 + 128) * 2), ling.kvBytesPerToken());
    try t.expect(ling.kvBytesPerToken() * 3 < 24 * 2 * 16 * 128 * 2);

    // MLA caches per ATTENTION head — it decompresses the latent to all of
    // them, so there is no grouping to bill against. Ling 3.0 ships 16/16, so
    // only an asymmetric config can tell the two spellings apart, and reading
    // `num_key_value_heads` here UNDER-bills (2 of 16 heads = 8x light) —
    // which surfaces as an uncatchable Metal OOM, not a refusal.
    var mla_grouped = ling;
    mla_grouped.num_key_value_heads = 2;
    try t.expectEqual(@as(u64, 6 * 16 * (192 + 128) * 2), mla_grouped.kvBytesPerToken());
    // ...while a NON-MLA arch still bills its KV heads, grouping and all.
    var gqa = dense;
    gqa.num_attention_heads = 32;
    gqa.num_key_value_heads = 8;
    try t.expectEqual(@as(u64, 24 * 8 * 256 * 2), gqa.kvBytesPerToken());

    // A hybrid WITHOUT MLA still drops its linear layers from the bill.
    var gdn = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    gdn.num_hidden_layers = 40;
    gdn.num_key_value_heads = 2;
    gdn.head_dim = 256;
    gdn.full_attention_interval = 4;
    try t.expectEqual(@as(u32, 10), gdn.attnCacheLayerCount());
    try t.expectEqual(@as(u64, 10 * 2 * 512 * 2), gdn.kvBytesPerToken());
}

test "prefillMemoryNeeded: a prompt shorter than the chunk bills the prompt-width forward, not the chunk cap" {
    // Live 2026-07-14: with 111 GB of weights resident (~4 GB headroom),
    // a 31-TOKEN prompt was rejected "needs ~6246MB" — the MLP envelope was
    // billed at the hd-128 8192 chunk cap while the actual prefill runs ONE
    // 31-token forward (~25 MB). A forward is never wider than the prompt:
    // the envelope must use min(chunk, seq). Guard-tracks-reality rule.
    const small = prefillMemoryNeeded(31, 64, 8, 327680, 128, 128, 4096, 13312, 8, 31, 31, 0, 0, .{});
    const capped = prefillMemoryNeeded(31, 64, 8, 327680, 128, 128, 4096, 13312, 8, 8192, 31, 0, 0, .{});
    try std.testing.expectEqual(small, capped);
    // Sanity: the clamped estimate for the live 31-token case sits far below
    // the ~4 GB that was actually available. It is not zero — every prefill
    // pays PREFILL_RUNTIME_FLOOR_BYTES — but a tiny prompt pays little else,
    // and the dequant-weight term never fires this far under the route's
    // minimum forward width.
    try std.testing.expect(capped < 1024 * 1024 * 1024);
}

test "prefillMemoryNeeded: quantized KV is billed at its real width, not fp16" {
    const t = std.testing;
    transformer_mod.fused256_override = false;
    defer transformer_mod.fused256_override = null;
    // 26B-A4B shape at 100K ctx, chunk 512 (what the bounded chunk picks there).
    // fp16: kv 30x2x100000x8x256x2 = 24.576 GB; scores 16x512x100000x2 = 1.6384 GB;
    //       mlp 3x8x512x2816x2 = 69.2 MB; x1.25 margin.
    try t.expectEqual(
        @as(u64, 32_854_507_520 + 671_088_640),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{}),
    );
    // 4-bit: kv billed at (2*4+1)/16 bytes/elem (payload + group scale/bias)
    // = 6.912 GB, plus the 0.8192 GB per-layer dense dequant transient.
    try t.expectEqual(
        @as(u64, 11_798_507_520 + 671_088_640),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 4, 512, 100_000, 0, 0, .{}),
    );
    // Direction: quantized admission must be under half the fp16 bill.
    try t.expect(prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 4, 512, 100_000, 0, 0, .{}) * 2 <
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{}));
}

test "prefillMemoryNeeded: working set is chunk-bounded — the 255K MoE prompt is admittable" {
    const t = std.testing;
    transformer_mod.fused256_override = false;
    defer transformer_mod.fused256_override = null;
    // The PR-#69 failure case: Qwen3.6-35B-A3B shape (whose checkpoint OMITS
    // intermediate_size, so the struct default 15360 leaks into ffn) at 262144
    // tokens with 4-bit KV. kv 6.04 GB + scores 4.29 GB + dequant 0.54 GB +
    // mlp 0.377 GB, x1.25 = ~13.1 GiB — admittable on a big Mac.
    const needed = prefillMemoryNeeded(262_144, 16, 2, 81920, 256, 256, 2048, 15360, 4, 512, 262_144, 0, 0, .{});
    try t.expectEqual(@as(u64, 14_061_404_160 + 671_088_640), needed);
    try t.expect(needed < 16 << 30);
    // A seq-scaled envelope (8 x seq x ffn x 2 working bytes, ~64 GB, on top
    // of fp16 KV, ~21.5 GB) would refuse this prompt with a spurious 400.
    const old_estimate: u64 = (40 * 2 * 262_144 * 2 * 256 * 2 + 8 * 262_144 * 15360 * 2) * 5 / 4;
    try t.expect(needed < old_estimate / 7);
}

test "prefillMemoryNeeded: unfused head dims bill the materialized score scratch" {
    const t = std.testing;
    transformer_mod.fused256_override = false;
    defer transformer_mod.fused256_override = null;
    // Same shape, head_dim 128 vs 256 (kv scaled by hdim too, so compare the
    // full bills computed by hand): hd=128 fp16 -> kv 12.288 GB + 0 scores +
    // mlp 69.2 MB, x1.25.
    try t.expectEqual(
        @as(u64, 15_446_507_520 + 671_088_640),
        prefillMemoryNeeded(100_000, 16, 8, 122880, 128, 128, 2816, 2112, 16, 512, 100_000, 0, 0, .{}),
    );
    // hd=256 adds the [heads, chunk, seq] score tensor (the guard must see
    // what the composed SDPA path actually allocates). Decomposition: the
    // hd-128 bill + the KV doubling from 128->256 (12.288 GB x1.25) + the
    // score tensor (16x512x100000x2 = 1.6384 GB x1.25).
    const with_scores = prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{});
    try t.expectEqual(@as(u64, 15_446_507_520 + 671_088_640 + 15_360_000_000 + 2_048_000_000), with_scores);
}

test "prefillMemoryNeeded: the SCORE width decides the score term, not the stored width" {
    // MLA contracts scores over qk_head_dim 192 while storing
    // values at v_head_dim 128 and declaring head_dim 128. mlx has a fused
    // vector kernel for that pair at DECODE and falls back to the composed
    // path at prefill widths, so the [heads, chunk, seq] score tensor is real —
    // but `prefillHeadDimFused(128)` is true, so passing the stored width for
    // both billed it at ZERO. Under-billing ends in an uncatchable Metal OOM
    // instead of a 400, which is the whole reason this guard exists.
    const t = std.testing;
    const kv_per_tok: u64 = 6 * 16 * (192 + 128) * 2; // 6 caching layers of 24
    const stored: u64 = 128;
    const scored: u64 = 192;
    // `sushi_attn_pd` serves qk 192 now, so the composed sheet survives only
    // under the kill switch — which is precisely where confusing the two
    // widths would zero a sheet that is really there.
    transformer_mod.fused256_override = false;
    defer transformer_mod.fused256_override = null;
    const honest = prefillMemoryNeeded(32_768, 16, 16, kv_per_tok, stored, scored, 1536, 4608, 16, 4096, 32_768, 0, 0, .{});
    const blind = prefillMemoryNeeded(32_768, 16, 16, kv_per_tok, stored, stored, 1536, 4608, 16, 4096, 32_768, 0, 0, .{});
    // The difference is exactly the score tensor: heads x chunk x seq x 2, x1.25.
    const score_bytes: u64 = 16 * 4096 * 32_768 * 2;
    try t.expectEqual(honest - blind, score_bytes * 5 / 4);
    try t.expect(honest > 2 * blind);
    // The two widths are independent: at fp16 (no dequant transient) the
    // stored width touches nothing the score term reads.
    try t.expectEqual(
        honest,
        prefillMemoryNeeded(32_768, 16, 16, kv_per_tok, scored, scored, 1536, 4608, 16, 4096, 32_768, 0, 0, .{}),
    );
    // With the kernel serving the width there is no sheet to attribute, so the
    // two widths agree — at zero, not by accident.
    transformer_mod.fused256_override = true;
    try t.expectEqual(
        blind,
        prefillMemoryNeeded(32_768, 16, 16, kv_per_tok, stored, scored, 1536, 4608, 16, 4096, 32_768, 0, 0, .{}),
    );
}

test "prefillMemoryNeeded: fused hd-256 kernel drops the score bill, keeps KV + dequant" {
    const t = std.testing;
    // Default (sushi_attn_p256 active): the composed score scratch never exists,
    // so hd 256 bills exactly the hd-256 KV + mlp — the unfused bill minus the
    // 1.6384 GB x1.25 score term from the test above.
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    try t.expectEqual(
        @as(u64, 32_854_507_520 + 671_088_640 - 2_048_000_000),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{}),
    );
    // The quantized-KV dequant transient is a denseView property, NOT a score
    // property — it must survive the fused kernel (fires on every kv-quant
    // request regardless of head_dim).
    try t.expectEqual(
        @as(u64, 11_798_507_520 + 671_088_640 - 2_048_000_000),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 4, 512, 100_000, 0, 0, .{}),
    );
}

test "resolvePrefillChunk: a squeezed box steps the chunk down; a roomy one keeps it" {
    // The reserve at the widest chunk must never claim more than a quarter of
    // what is left after the weights. Numbers are the measured 16 GB profile: Metal's working
    // set ~11.9 GiB, Mistral-7B-4bit 3.80 GiB resident. The hot-cache budget is no longer an
    // input; the former `2 * GiB` argument is the pinned-context KV bar and is 0 under auto-context.
    const t = std.testing;
    const GiB: u64 = 1024 * 1024 * 1024;
    var mistral = model_mod.ModelConfig{};
    mistral.num_hidden_layers = 32;
    mistral.num_attention_heads = 32;
    mistral.num_key_value_heads = 8;
    mistral.head_dim = 128;
    mistral.hidden_size = 4096;
    mistral.intermediate_size = 14336;
    mistral.intermediate_size_declared = true;
    mistral.quant_bits = 4;
    mistral.max_position_embeddings = 32768;

    const ceiling_16: u64 = 11_918_000_000;
    const weights: u64 = 4_080_000_000;
    const narrowed = resolvePrefillChunk(&mistral, 16, ceiling_16, weights, 0, 0);
    try t.expect(narrowed < 8192);
    try t.expect(narrowed >= 512);
    // It is the WIDEST rung that fits, not simply the floor.
    try t.expect(prefillTransientReserve(&mistral, 16, narrowed) <=
        (ceiling_16 - weights) / 4);
    for (PREFILL_CHUNK_LADDER) |c| {
        if (c <= narrowed) break;
        try t.expect(prefillTransientReserve(&mistral, 16, c) > (ceiling_16 - weights) / 4);
    }

    // Same model, a machine with room: nothing narrows.
    try t.expectEqual(@as(u32, 8192), resolvePrefillChunk(&mistral, 16, 95 * GiB, weights, 0, 0));

    // A model that barely fits gets the narrowest rung rather than a wide one
    // it cannot pay for — and never 0, which would read as "not pinned".
    try t.expectEqual(@as(u32, 512), resolvePrefillChunk(&mistral, 16, weights + 64 * 1024 * 1024, weights, 0, 0));

    // Every rung is a real ladder entry, descending, floored at generate's own
    // minimum — a chunk generate would refuse is a chunk the bill cannot model.
    var prev: u32 = std.math.maxInt(u32);
    for (PREFILL_CHUNK_LADDER) |c| {
        try t.expect(c < prev);
        try t.expect(c >= prefill_plan.PREFILL_CHUNK_FLOOR);
        prev = c;
    }
}

test "qsaMaskBytes: a qwen4_exp twin bills the QSA mask and steps a rung the same trunk without qwen4_exp keeps" {
    const t = std.testing;
    var twin = model_mod.ModelConfig{};
    twin.num_hidden_layers = 40;
    twin.num_attention_heads = 24;
    twin.num_key_value_heads = 2;
    twin.head_dim = 256;
    twin.hidden_size = 2560;
    twin.intermediate_size = 9216;
    twin.intermediate_size_declared = true;
    twin.quant_bits = 4;
    twin.max_position_embeddings = 262144;
    var q4 = twin;
    q4.indexer_budget = 2048;
    q4.indexer_n_heads = 4;
    q4.indexer_head_dim = 128;
    q4.indexer_compress_ratio = 4;
    try t.expectEqual(@as(u64, 0), qsaMaskBytes(&twin, 4096, 25000));
    // Decode/verify widths build the dense mask; prefill widths gather by
    // block and bill the bounded score sheet instead of 4 B x rows x keys.
    try t.expectEqual(@as(u64, 4 * 8 * 25000 * 5 / 4), qsaMaskBytes(&q4, 8, 25000));
    qwen4_qsa_gather.qsa_gather_override = true;
    defer qwen4_qsa_gather.qsa_gather_override = null;
    const saved_fused = qwen4_qsa_select.qsa_score_fused_override;
    const saved_nax = transformer_mod.vqmm_nax_probe_override;
    defer {
        qwen4_qsa_select.qsa_score_fused_override = saved_fused;
        transformer_mod.vqmm_nax_probe_override = saved_nax;
    }
    transformer_mod.vqmm_nax_probe_override = false;
    qwen4_qsa_select.qsa_score_fused_override = true;
    const gathered_composed = qsaMaskBytes(&q4, 4096, 250_000);
    try t.expect(gathered_composed > 0);
    try t.expect(gathered_composed < 4 * 4096 * 250_000);
    try t.expectEqual(transformer_mod.qsaPrefillTransientBytes(4, 4096, 250_000, 4, 128), gathered_composed);
    transformer_mod.vqmm_nax_probe_override = null;
    qwen4_qsa_select.qsa_score_fused_override = true;
    const gathered_fused = qsaMaskBytes(&q4, 4096, 250_000);
    try t.expectEqual(transformer_mod.qsaPrefillTransientBytes(4, 4096, 250_000, 4, 128), gathered_fused);
    if (transformer_mod.qsaScoreFusedActive()) {
        try t.expect(gathered_fused < gathered_composed);
    }
    qwen4_qsa_select.qsa_score_fused_override = false;
    transformer_mod.vqmm_nax_probe_override = false;
    const gathered = qsaMaskBytes(&q4, 4096, 250_000);
    try t.expectEqual(gathered_composed, gathered);
    const twin_bill = prefillTransientReserve(&twin, 16, 8192);
    const q4_bill = prefillTransientReserve(&q4, 16, 8192);
    try t.expectEqual(twin_bill + qsaMaskBytes(&q4, 8192, 8192) + q4.qsaRingBytes() * 5 / 4, q4_bill);
    // A ceiling whose quarter-share sits between the two bills at 8192.
    const weights: u64 = 70_000_000_000;
    const ceiling = weights + (twin_bill + q4_bill) / 2 * 4;
    try t.expectEqual(@as(u32, 8192), resolvePrefillChunk(&twin, 16, ceiling, weights, 0, 0));
    try t.expectEqual(@as(u32, 4096), resolvePrefillChunk(&q4, 16, ceiling, weights, 0, 0));
}

test "resolvePrefillChunk: the sizer and the guard bill the chunk that was pinned" {
    // The whole point of pinning: `prefillTransientReserve` (sizer) and
    // `prefillMemoryNeeded` (guard) must move together with the resolved width.
    // A narrower pin has to produce a SMALLER bill for the same prompt, or the
    // step-down bought nothing; and generate must resolve to the same width or
    // the forward runs wider than the bill and the Metal OOM is uncatchable.
    const t = std.testing;
    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 32;
    cfg.num_attention_heads = 32;
    cfg.num_key_value_heads = 8;
    cfg.head_dim = 128;
    cfg.hidden_size = 4096;
    cfg.intermediate_size = 14336;
    cfg.intermediate_size_declared = true;

    const wide = prefillTransientReserve(&cfg, 16, 8192);
    const narrow = prefillTransientReserve(&cfg, 16, 512);
    try t.expect(narrow * 2 < wide);

    cfg.pinned_prefill_chunk = 512;
    try t.expectEqual(@as(usize, 512), prefill_plan.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        40_000,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        cfg.pinned_prefill_chunk,
    ));
    // Unpinned keeps the launch width — a model the sizer never ran on must
    // not silently get a narrower forward than the guard bills.
    const unpinned_prefill_chunk: usize = 0;
    try t.expectEqual(@as(usize, 8192), prefill_plan.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        40_000,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        unpinned_prefill_chunk,
    ));
    // An explicit --prefill-chunk outranks the pin in BOTH directions.
    generate_mod.prefill_chunk_override = 2048;
    generate_mod.prefill_chunk_explicit = true;
    defer {
        generate_mod.prefill_chunk_override = 8192;
        generate_mod.prefill_chunk_explicit = false;
    }
    try t.expectEqual(@as(usize, 2048), prefill_plan.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        40_000,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        cfg.pinned_prefill_chunk,
    ));
}

test "prefillMemoryNeeded: a GDN trunk's prefill peaks stay covered" {
    // Peak GPU bytes above steady state for ONE request on a clean boot
    // (`--prefix-cache-entries 0 --no-mtp --no-drafter`, /props
    // peak_bytes minus active_bytes, M4 Max) — byte-identical across repeat
    // boots, so these are exact figures, not samples. An estimate BELOW a peak
    // we have actually seen ends in an uncatchable Metal OOM instead of a 400.
    // The rows are the measured peaks minus the exact conv_state pin the
    // compaction removed ((linear_layers - 5) x chunk x qkv x 2).
    const t = std.testing;
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;

    // Hidden 5120, ffn 17408, 64 layers / 16 caching, hd 256,
    // 48 GatedDeltaNet layers x (2x16x128 + 48x128) elems x 2 B of stream.
    const q27_stream: u64 = 5 * (2 * 16 * 128 + 48 * 128) * 2;
    const q27_dq: u64 = 3 * 17408 * 5120 * 2;
    try t.expect(prefillMemoryNeeded(9320, 24, 4, 65536, 256, 256, 5120, 17408, 16, 2048, 9320, q27_stream, q27_dq, .{}) >= 2_174_449_280);
    try t.expect(prefillMemoryNeeded(9270, 24, 4, 65536, 256, 256, 5120, 17408, 16, 8192, 9270, q27_stream, q27_dq, .{}) >= 3_623_797_120);
    // Hidden 2560, ffn 9216, 32 layers / 8 caching, 24 GDN layers.
    const q4_stream: u64 = 5 * (2 * 16 * 128 + 32 * 128) * 2;
    const q4_dq: u64 = 3 * 9216 * 2560 * 2;
    try t.expect(prefillMemoryNeeded(9340, 16, 4, 32768, 256, 256, 2560, 9216, 16, 1024, 9340, q4_stream, q4_dq, .{}) >= 1_329_232_896);
    try t.expect(prefillMemoryNeeded(9270, 16, 4, 32768, 256, 256, 2560, 9216, 16, 8192, 9270, q4_stream, q4_dq, .{}) >= 2_559_863_168);
    // A MoE trunk: hidden 2816, top_k 8 x moe 704.
    const g4_stream: u64 = 4 * 8 * 2 * (2816 + 704) * 2;
    const g4_dq: u64 = 3 * 5632 * 2816 * 2;
    try t.expect(prefillMemoryNeeded(8204, 16, 8, 245760, 256, 256, 2816, 5632, 16, 4096, 8204, g4_stream, g4_dq, .{}) >= 3_960_000_000);
}

test "prefillMemoryNeeded: the new terms fire only where the measurement put them" {
    const t = std.testing;
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    const q27 = .{ .seq = 9320, .h = 24, .kvh = 4, .kvpt = 65536, .hd = 256, .hidden = 5120, .ffn = 17408 };
    const stream: u64 = 48 * (2 * 16 * 128 + 48 * 128) * 2;
    const dq: u64 = 3 * 17408 * 5120 * 2;

    // The stream term is per CHUNK-TOKEN: it scales with the forward width and
    // not with the prompt, which is what the whole chunk-bounded model says.
    const at2k = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, stream, 0, .{});
    const at8k = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 8192, q27.seq, stream, 0, .{});
    const flat2k = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, 0, 0, .{});
    const mlp2k: u64 = 8 * 2048 * q27.ffn * 2;
    const floor: u64 = 512 * 1024 * 1024;
    // The stream arm wins on this arch — that IS the fix — so the bill is one
    // MLP envelope plus the streams, not the three-envelope floor.
    try t.expectEqual(@as(u64, (q27.seq * q27.kvpt + mlp2k + 2048 * stream + floor) * 5 / 4), at2k);
    try t.expectEqual(@as(u64, (q27.seq * q27.kvpt + 3 * mlp2k + floor) * 5 / 4), flat2k);
    try t.expect(at2k > flat2k);
    try t.expect(at8k > at2k);

    // A longer prompt at the same chunk adds KV only — the stream term does
    // not move, or it would be the per-token multiplier this class is about.
    const long = prefillMemoryNeeded(4 * q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, 4 * q27.seq, stream, 0, .{});
    try t.expectEqual(long - at2k, 3 * q27.seq * q27.kvpt * 5 / 4);

    // The dequant-weight term is chunk-INDEPENDENT (it is weights) but only
    // bills where the route can fire: at or above PREFILL_DQ_GEMM_MIN_M.
    const dq_wide = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, 0, dq, .{});
    try t.expectEqual(dq_wide - flat2k, dq * 5 / 4);
    const narrow_with = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 1024, q27.seq, 0, dq, .{});
    const narrow_without = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 1024, q27.seq, 0, 0, .{});
    try t.expectEqual(narrow_with, narrow_without);
    // A dense checkpoint passes 0 and is billed nothing for it.
    try t.expectEqual(flat2k, prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, 0, 0, .{}));

    // An arch with neither keeps the 3-envelope bill it had, plus the floor —
    // no plain-attention arch's admission tightens by more than the floor.
    const dense_only = prefillMemoryNeeded(9827, 32, 2, 53248, 128, 128, 6656, 19968, 16, 2048, 9827, 0, 0, .{});
    const pre_fix: u64 = (9827 * 53248 + 3 * 8 * 2048 * 19968 * 2) * 5 / 4;
    try t.expectEqual(dense_only - pre_fix, 512 * 1024 * 1024 * 5 / 4);
}

test "prefillStreamBytesPerToken: keyed on the arch's own geometry, zero for plain attention" {
    const t = std.testing;
    // 64 layers, every 4th full — 48 GatedDeltaNet layers, each
    // holding a chunk-wide q/k/v stream (2 x 16 x 128 key + 48 x 128 value).
    var q27 = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    q27.num_hidden_layers = 64;
    q27.full_attention_interval = 4;
    q27.linear_num_key_heads = 16;
    q27.linear_num_value_heads = 48;
    try t.expectEqual(@as(u64, 5 * (2 * 16 * 128 + 48 * 128) * 2), prefillStreamBytesPerToken(&q27));

    var q4 = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    q4.num_hidden_layers = 32;
    q4.full_attention_interval = 4;
    q4.linear_num_key_heads = 16;
    q4.linear_num_value_heads = 32;
    try t.expectEqual(@as(u64, 5 * (2 * 16 * 128 + 32 * 128) * 2), prefillStreamBytesPerToken(&q4));

    // No linear layers, but a MoE prefill gathers a top_k-fold
    // copy of the hidden stream beside the expert rows, for the 4 layers the
    // MoE eval cadence lets coexist.
    var g4 = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    g4.num_hidden_layers = 30;
    g4.hidden_size = 2816;
    g4.num_experts = 128;
    g4.num_experts_per_tok = 8;
    g4.moe_intermediate_size = 704;
    try t.expectEqual(@as(u64, 4 * 8 * 2 * (2816 + 704) * 2), prefillStreamBytesPerToken(&g4));

    // A plain attention arch declares neither and is
    // billed exactly what it was billed before this term existed.
    var dense = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    dense.num_hidden_layers = 52;
    dense.hidden_size = 6656;
    try t.expectEqual(@as(u64, 0), prefillStreamBytesPerToken(&dense));

    // Non-attention layers without linear head geometry are NOT
    // linear-attention layers: the term must not invent a stream for them.
    var conv = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    conv.num_hidden_layers = 30;
    conv.full_attention_interval = 4;
    conv.hidden_size = 2048;
    try t.expectEqual(@as(u64, 0), prefillStreamBytesPerToken(&conv));
}

test "prefillDequantWeightBytes: affine-quantized weights only, and it reads the route's kill switch" {
    const t = std.testing;
    var cfg = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    cfg.hidden_size = 6656;
    cfg.intermediate_size = 19968;
    cfg.intermediate_size_declared = true;
    cfg.quant_bits = 4;
    cfg.quant_mode = .affine;
    try t.expectEqual(@as(u64, 3 * 19968 * 6656 * 2), prefillDequantWeightBytes(&cfg));

    // Dense bf16: the route never runs, and the measurement agrees — a bf16
    // checkpoint's prefill peak sits below the quantized one by this term.
    var dense = cfg;
    dense.quant_bits = 0;
    try t.expectEqual(@as(u64, 0), prefillDequantWeightBytes(&dense));

    // Non-affine (mxfp4/mxfp8) stays in mlx_quantized_matmul — no dequantized
    // copy exists to bill.
    var mxfp8 = cfg;
    mxfp8.quant_mode = .mxfp8;
    try t.expectEqual(@as(u64, 0), prefillDequantWeightBytes(&mxfp8));

    // The estimator must not bill an allocation the kill switch removed.
    transformer_mod.prefill_dq_gemm_override = false;
    defer transformer_mod.prefill_dq_gemm_override = null;
    try t.expectEqual(@as(u64, 0), prefillDequantWeightBytes(&cfg));
}

test "prefillMemoryNeeded: a sparse-attention arch bills its KEY BOUND, not the whole prompt" {
    const t = std.testing;
    // DeepSeek-V4-Flash shape (live 2026-07-31): 64 heads, ONE latent kv head,
    // 43 layers, head_dim 512 — outside every fused kernel, so the score term
    // fires. But DSV4 attends to sliding_window(128) + index_topk(512) + sink
    // keys per query, NEVER the whole prompt (`tk = wk + n_sel`). Billing a dense 5806-wide key axis put a 5806-token prompt at
    // ~10.3 GB and 400-rejected it on a 1M-context model with 8.6 GB free.
    // The live request verbatim: 5806 tokens at kv-quant 8, chunk 5632, with
    // 8629 MB free. Pre-fix it billed 10277 MB — a dense 5806-wide key axis
    // PLUS the 15360 intermediate_size struct default — and 400'd.
    const available: u64 = 8629 * 1024 * 1024;
    const before = prefillMemoryNeeded(5806, 64, 1, 88064, 512, 512, 4096, 15360, 8, 5632, 5806, 0, 0, .{});
    try t.expectEqual(@as(u64, 10277 + 640), before / (1024 * 1024));
    try t.expect(before > available);
    // Fixed: 641 keys (window 128 + top-512 + sink) and the width the config
    // actually states (top_k 6 x moe_intermediate 2048 = 12288, what
    // prefillFfnWidth resolves for this checkpoint) — 4848 MB, admitted.
    const after = prefillMemoryNeeded(5806, 64, 1, 88064, 512, 512, 4096, 12288, 8, 5632, 641, 0, 0, .{});
    try t.expectEqual(@as(u64, 4848 + 640), after / (1024 * 1024));
    try t.expect(after < available);
    // The score term is the ONLY one the key bound may touch: same call with a
    // fused head_dim has no score scratch, so the bound changes nothing.
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    try t.expectEqual(
        prefillMemoryNeeded(5806, 64, 1, 44032, 256, 256, 4096, 2048, 8, 5632, 5806, 0, 0, .{}),
        prefillMemoryNeeded(5806, 64, 1, 44032, 256, 256, 4096, 2048, 8, 5632, 641, 0, 0, .{}),
    );
}

test "resolvePrefillChunk: the 16 GB tier gets a working context instead of the 1024 floor" {
    // The reported bug, end to end. Mistral-7B-4bit on a 16 GB Mac: ~11.9 GiB
    // Metal working set, 3.80 GiB of weights (measured), the hot prefix cache
    // at its 2 GB default. At the launch chunk the transient reserve alone is
    // larger than what is left, so the sizer floored the advertised context at
    // 1024 and the guard 400'd a 10,348-token prompt for "needing" 9.18 GiB —
    // a prompt whose peak measures 2.391 GiB (M4 Max, one request per boot).
    const t = std.testing;
    const GiB: u64 = 1024 * 1024 * 1024;
    const ceiling: u64 = 11_918_000_000;
    const weights: u64 = 4_080_000_000;
    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 32;
    cfg.num_attention_heads = 32;
    cfg.num_key_value_heads = 8;
    cfg.head_dim = 128;
    cfg.hidden_size = 4096;
    cfg.intermediate_size = 14336;
    cfg.intermediate_size_declared = true;
    cfg.quant_bits = 4;
    cfg.max_position_embeddings = 32768;
    const per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 16);

    const launch = safeContextForBudget(ceiling, weights, 2 * GiB +| prefillTransientReserve(&cfg, 16, 8192), per_tok, 32768);
    try t.expectEqual(@as(u32, 1024), launch); // the floor: nothing fits

    // The hot-cache budget is no longer an input; auto-context means the pinned-context bar is 0.
    const chunk = resolvePrefillChunk(&cfg, 16, ceiling, weights, 0, 0);
    const sized = safeContextForBudget(ceiling, weights, 2 * GiB +| prefillTransientReserve(&cfg, 16, chunk), per_tok, 32768);
    try t.expect(sized > 12_000);

    // And the guard now admits a real agent prompt at that context, billed
    // against the same chunk. 8.10 GiB is what a 16 GB Mac has free with only
    // the weights resident.
    const needed = prefillMemoryNeeded(
        10_348,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.kvBytesPerToken(),
        cfg.head_dim,
        cfg.prefillScoreHeadDim(),
        cfg.hidden_size,
        prefillFfnWidth(&cfg),
        16,
        chunk,
        cfg.prefillAttnKeys(10_348),
        prefillStreamBytesPerToken(&cfg),
        prefillDequantWeightBytes(&cfg),
        .{},
    );
    try t.expect(needed < ceiling - weights);
    // Still comfortably above the measured 2.391 GiB peak — narrowing the chunk
    // must not turn a conservative bill into an optimistic one.
    try t.expect(needed > 2_391_000_000);
}

test "prefillFfnWidth: a declared MoE width beats the struct default, which is only a fallback" {
    const t = std.testing;
    // DSV4's config ships NO intermediate_size at all, so the struct default
    // (15360) was billed as though declared. The width the config DOES state is
    // top_k(6) x moe_intermediate(2048) — a chunk gathers that many expert rows
    // per token, so billing the bare 2048 would UNDER-bill 3x, and under-
    // billing ends in an uncatchable Metal OOM rather than a 400.
    var moe = model_mod.ModelConfig{};
    moe.intermediate_size = 15360; // struct default, never seen in the JSON
    moe.intermediate_size_declared = false;
    moe.moe_intermediate_size = 2048;
    moe.num_experts_per_tok = 6;
    moe.shared_expert_intermediate_size = 0;
    try t.expectEqual(@as(u64, 12288), prefillFfnWidth(&moe));

    // The shared expert runs on every token too, so it adds rather than maxes.
    var shared = moe;
    shared.shared_expert_intermediate_size = 1024;
    try t.expectEqual(@as(u64, 13312), prefillFfnWidth(&shared));

    // A dense checkpoint is unaffected: no MoE width, declared ffn wins.
    var dense = model_mod.ModelConfig{};
    dense.intermediate_size = 13312;
    dense.intermediate_size_declared = true;
    dense.moe_intermediate_size = 0;
    dense.num_experts_per_tok = 0;
    try t.expectEqual(@as(u64, 13312), prefillFfnWidth(&dense));

    // A checkpoint that DECLARES a dense ffn keeps the conservative @max — a
    // hybrid with dense layers really does run the wider one.
    var declared = moe;
    declared.intermediate_size_declared = true;
    try t.expectEqual(@as(u64, 15360), prefillFfnWidth(&declared));

    // Declares nothing: the struct default is all we have, so it still wins.
    var bare = model_mod.ModelConfig{};
    bare.intermediate_size = 15360;
    bare.intermediate_size_declared = false;
    bare.moe_intermediate_size = 0;
    bare.shared_expert_intermediate_size = 0;
    try t.expectEqual(@as(u64, 15360), prefillFfnWidth(&bare));
}

test "defaultEnableMtp: --mtp forces the native head on for MoE targets" {
    const t = std.testing;
    // No sidecar loaded → never on, whatever the operator asked for.
    try t.expect(!defaultEnableMtp(false, false, false, false, false));
    try t.expect(!defaultEnableMtp(false, true, true, false, false));
    // Dense target with a sidecar → on by default (unchanged behavior).
    try t.expect(defaultEnableMtp(true, false, false, false, false));
    try t.expect(defaultEnableMtp(true, false, true, false, false));
    // MoE target → OFF by default (the verify-forward routing caution) ...
    try t.expect(!defaultEnableMtp(true, true, false, false, false));
    // ... but ON when the operator passed --mtp. Without this, a MoE MTP
    // checkpoint is unreachable from any client that doesn't send
    // `enable_mtp:true` in the body (llmprobe, Claude Code, curl).
    try t.expect(defaultEnableMtp(true, true, true, false, false));
    // A MEASURED native MoE head defaults ON despite is_moe — the
    // caution above is about a bolted-on sidecar paying expert routing it was
    // never designed around, and this arch was measured no-worse-than-serial
    // at every context rung on two prompt shapes.
    try t.expect(defaultEnableMtp(true, true, false, true, false));
    // The claim is about the HEAD, so it still needs one loaded.
    try t.expect(!defaultEnableMtp(false, true, false, true, false));
    try t.expect(!defaultEnableMtpForStreaming(true, false, false, true, false, true));
    try t.expect(defaultEnableMtpForStreaming(true, true, true, false, false, true));
}

test "defaultEnableMtp: a served MoE pack with its head loaded defaults ON; streaming keeps --mtp" {
    const t = std.testing;
    try t.expect(defaultEnableMtp(true, true, false, false, true));
    try t.expect(!defaultEnableMtp(false, true, false, false, true)); // still needs a head
    try t.expect(!defaultEnableMtp(true, true, false, false, false)); // an inherited MoE arch keeps the caution
    try t.expect(!defaultEnableMtpForStreaming(true, true, false, false, true, true));
    try t.expect(defaultEnableMtpForStreaming(true, true, true, false, true, true));
    try t.expect(defaultEnableMtpForStreaming(true, true, false, false, true, false));
    try t.expect(model_mod.isServedArch("qwen4_exp") and model_mod.isServedArch("mimo_v2"));
}

test "the load-time bill prices the MTP head a served pack runs by default" {
    const t = std.testing;
    const saved_force = server_config.default_force_mtp;
    defer server_config.default_force_mtp = saved_force;
    server_config.default_force_mtp = false;
    var cfg = qwen4ExpOomConfig();
    cfg.mtp_override = null;
    const head = mtpHeadKvBytesPerToken(&cfg) + mtpHeadStateBytesPerToken(&cfg);
    try t.expect(head > 0);
    try t.expect(mtpHeadDefaultOn(&cfg));
    var off = cfg;
    off.mtp_override = false;
    try t.expect(!mtpHeadDefaultOn(&off));
    try t.expectEqual(sessionBytesPerToken(&off, 8) + head, sessionBytesPerToken(&cfg, 8));
    var streamed = cfg;
    streamed.expert_streaming = true;
    try t.expect(!mtpHeadDefaultOn(&streamed));
    // A streamed load that kept its head bills it like a resident one.
    streamed.stream_mtp_head = true;
    try t.expect(mtpHeadDefaultOn(&streamed));
    try t.expectEqual(sessionBytesPerToken(&off, 8) + head, sessionBytesPerToken(&streamed, 8));
}

test "the load-time bill drops the MTP head when Qwen's default round is DFlash2" {
    const t = std.testing;
    const saved_force = server_config.default_force_mtp;
    defer server_config.default_force_mtp = saved_force;
    server_config.default_force_mtp = false;
    var cfg = qwen4ExpOomConfig();
    cfg.mtp_override = null;
    const head = mtpHeadKvBytesPerToken(&cfg) + mtpHeadStateBytesPerToken(&cfg);
    const with_head = sessionBytesPerToken(&cfg, 8);
    var dflash = cfg;
    dflash.dflash_loaded = true;
    try t.expect(!mtpHeadDefaultOn(&dflash));
    try t.expectEqual(with_head - head, sessionBytesPerToken(&dflash, 8));
    // The auto-context bill reads the same default: no head terms at the rung it prices.
    const a = sushiContextBytesAt(&cfg, 100_000, 8).bytes;
    const b = sushiContextBytesAt(&dflash, 100_000, 8).bytes;
    try t.expect(b + 100_000 * head <= a + dflashRequestBytes(&dflash, 4096));
    try t.expect(b < a + dflashRequestBytes(&dflash, 8192));
    // A forced head still yields to the drafter by default, and `enable_drafter:false` bills it again.
    server_config.default_force_mtp = true;
    try t.expectEqual(@as(u64, 0), prefillRequestTerms(&dflash, 100_000, 4096, 8, 4096, .{}).mtp_head_kv_bytes);
    try t.expect(prefillRequestTerms(&dflash, 100_000, 4096, 8, 4096, .{ .mtp_on = true }).mtp_head_kv_bytes > 0);
    try t.expect(prefillRequestTerms(&cfg, 100_000, 4096, 8, 4096, .{}).mtp_head_kv_bytes > 0);
}

test "reasoningWithinBudget cuts a thought at the budget and keeps a shorter one whole" {
    const a = std.testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.vocab.deinit();
    defer tok.id_to_token.deinit();
    defer tok.merge_ranks.deinit();
    defer tok.special_tokens.deinit();
    defer tok.unicode_to_byte.deinit();
    try tok.special_tokens.put("aa", 1);
    try tok.special_tokens.put("bb", 2);
    try tok.special_tokens.put("cc", 3);
    try tok.id_to_token.put(1, "aa");
    try tok.id_to_token.put(2, "bb");
    try tok.id_to_token.put(3, "cc");
    const thought = "aabbcc";
    const under = try reasoningWithinBudget(a, &tok, thought, 3);
    defer if (under.owned) a.free(under.text);
    try std.testing.expect(!under.owned);
    try std.testing.expectEqualStrings(thought, under.text);
    const over = try reasoningWithinBudget(a, &tok, thought, 2);
    defer if (over.owned) a.free(over.text);
    try std.testing.expectEqualStrings("aabb", over.text);
}

test "the out-of-memory 503 names the cap's flag and never blames concurrency" {
    // #126: an idle server with zero models loaded and 21 MB RSS answered
    // "retry after current requests complete", which is unactionable advice
    // pointing at the wrong subsystem — the cause is the static
    // --max-resident-mem cap, a flag the message never mentioned and the app
    // does not pass. Both 503 sites read the one constant.
    const m = not_enough_memory_message;
    try testing.expect(std.mem.indexOf(u8, m, "--max-resident-mem") != null);
    try testing.expect(std.mem.indexOf(u8, m, "retry after current requests") == null);
    // Only the resident-budget gate raises NotEnoughMemory; the free-memory
    // preflight has its own arm (insufficient_free_memory_message). Blaming
    // free memory here sends the user quitting apps when the fix is the cap.
    try testing.expect(std.mem.indexOf(u8, m, "free memory") == null);
    try testing.expect(std.mem.indexOf(u8, m, "resident-model budget") != null);

    // One mapping serves both load switches; a second literal is the drift.
    try testing.expectEqualStrings("out_of_memory", loadRefusalFor(error.NotEnoughMemory).?.type);
    try testing.expectEqualStrings(not_enough_memory_message, loadRefusalFor(error.NotEnoughMemory).?.message);
    try testing.expectEqualStrings("out_of_memory", loadRefusalFor(error.InsufficientMemory).?.type);
    try testing.expectEqualStrings(insufficient_free_memory_message, loadRefusalFor(error.InsufficientMemory).?.message);
}

test "messageReasoningFromObj: reasoning_content round-trip, reasoning fallback, empty dropped" {
    const t = std.testing;
    const allocator = t.allocator;
    const cases = [_]struct { body: []const u8, want: ?[]const u8 }{
        // pi/vLLM SSE spelling (live 2026-07-29).
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning_content\":\"two plus two\"}", .want = "two plus two" },
        // vLLM request spelling.
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning\":\"twice two\"}", .want = "twice two" },
        // reasoning_content wins when both are present.
        .{ .body = "{\"role\":\"assistant\",\"reasoning_content\":\"a\",\"reasoning\":\"b\"}", .want = "a" },
        // Empty string is NOT reasoning — it would render the empty
        // <think></think> nothink signature the field exists to avoid.
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning_content\":\"\"}", .want = null },
        // Non-string shapes are ignored, never crash.
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning_content\":{\"x\":1}}", .want = null },
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\"}", .want = null },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const got = messageReasoningFromObj(parsed.value.object);
        if (case.want) |w| {
            try t.expectEqualStrings(w, got.?);
        } else {
            try t.expect(got == null);
        }
    }
}

test "admitMtpForCtx: the --max-mtp-ctx ceiling outranks the request's own flag" {
    const saved = generate_mod.max_mtp_ctx;
    defer generate_mod.max_mtp_ctx = saved;

    generate_mod.max_mtp_ctx = 0;
    try std.testing.expect(admitMtpForCtx(true, 1_000_000));
    try std.testing.expect(!admitMtpForCtx(false, 1));

    // Inclusive at the ceiling; an explicit `enable_mtp:true` loses to it.
    generate_mod.max_mtp_ctx = 4096;
    try std.testing.expect(admitMtpForCtx(true, 4096));
    try std.testing.expect(!admitMtpForCtx(true, 4097));

    // It never turns MTP on for a request that had it off for another reason.
    try std.testing.expect(!admitMtpForCtx(false, 1));
}

/// The qwen4_exp shape of the live #353 report: Qwen3.8-Flash-Next mixed-4-8bit, `--ctx-size 1048576 --kv-quant 8`.
pub fn qwen4ExpOomConfig() model_mod.ModelConfig {
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "qwen4_exp"; // arch-scan-exempt: test fixture, not a gate
    cfg.num_hidden_layers = 48;
    cfg.full_attention_interval = 4; // 12 caching layers, 36 GatedDeltaNet
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 256;
    cfg.hidden_size = 2560;
    cfg.moe_intermediate_size = 640;
    cfg.num_experts = 512;
    cfg.num_experts_per_tok = 10;
    cfg.shared_expert_intermediate_size = 640;
    cfg.linear_num_key_heads = 16;
    cfg.linear_key_head_dim = 128;
    cfg.linear_num_value_heads = 48;
    cfg.linear_value_head_dim = 128;
    cfg.linear_conv_kernel_dim = 4;
    cfg.indexer_budget = 512;
    cfg.indexer_head_dim = 128;
    cfg.indexer_compress_ratio = 4;
    // Pinned: with nothing pinned `getEffectiveContextLength` reads live GPU memory and every
    // expectation below depends on the box.
    cfg.pinned_context = 1_048_576;
    return cfg;
}

test "qwen4ExpOomConfig does not leak qsa_score_fused_override" {
    const saved = qwen4_qsa_select.qsa_score_fused_override;
    defer qwen4_qsa_select.qsa_score_fused_override = saved;
    qwen4_qsa_select.qsa_score_fused_override = null;
    _ = qwen4ExpOomConfig();
    try std.testing.expectEqual(@as(?bool, null), qwen4_qsa_select.qsa_score_fused_override);
}

test "the 458k prefill's two unbilled terms are billed: retained checkpoints and reserved capacity" {
    // These pins price a cache with no SSD tier behind it: every checkpoint stays in RAM.
    const no_tier = TierGuard.pin(false, 0);
    defer no_tier.restore();
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    // A 458,832-token cold prefill was admitted against 26.6 GB of headroom and died mid-prefill.
    const cfg = qwen4ExpOomConfig();
    const seq: u64 = 458_832;
    const chunk: u64 = 4096;

    // 1. SSM checkpoints: 36 GDN layers x (state + conv window), up to --ssm-checkpoint-max.
    const per_cp = cfg.ssmCheckpointBytes();
    // 36 x (3,145,728 + 61,440) B = 115.46 MB: the state is f32.
    try t.expect(per_cp > 115_000_000 and per_cp < 116_000_000);
    const saved_cp_max = ssm_checkpoint_max;
    ssm_checkpoint_max = 32;
    defer ssm_checkpoint_max = saved_cp_max;
    const held = retainedSsmCheckpointBytes(&cfg, seq, 0, chunk);
    try t.expectEqual(@as(u64, ssm_checkpoint_max) * per_cp, held);
    try t.expect(held > 1_800_000_000);
    // It saturates: 393k holds the same 32 as 458k.
    try t.expectEqual(held, retainedSsmCheckpointBytes(&cfg, 393_216, 0, chunk));
    // An arch with no linear layers is billed nothing.
    var dense = model_mod.ModelConfig{};
    dense.num_hidden_layers = 32;
    dense.num_attention_heads = 32;
    try t.expectEqual(@as(u64, 0), dense.ssmCheckpointBytes());
    try t.expectEqual(@as(u64, 0), retainedSsmCheckpointBytes(&dense, seq, 0, chunk));

    // 2. Reserved capacity: prompt + max_tokens + one chunk; the guard bills the headroom beyond
    // the prompt's own rows, not a second copy of the cache.
    const terms = prefillRequestTerms(&cfg, seq, 2048, 8, chunk, .{});
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8);
    try t.expectEqual((2048 + chunk) * kv_per_tok, terms.reserved_kv_bytes);
    try t.expect(terms.reserved_kv_bytes * 4 < seq * kv_per_tok); // headroom, not a copy
    // The indexer history is billed at one copy plus the score bank (`statePerTokenBilled`).
    try t.expectEqual(cfg.qsaHistoryBytesPerToken() + cfg.qsaScoreBankBytesPerToken(), statePerTokenBilled(&cfg));
    try t.expectEqual((seq + 2048 + chunk) * statePerTokenBilled(&cfg), terms.state_bytes);
    try t.expectEqual(held, terms.checkpoint_bytes);

    // 3. The bill moved by more than 3 GB, toward a 503 instead of a corpse.
    const with_terms = prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, prefillFfnWidth(&cfg), 8, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), terms);
    const without = prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, prefillFfnWidth(&cfg), 8, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), .{});
    try t.expect(with_terms > without + 3_000_000_000);
    // The sentinel runs through the same path: an omitted max_tokens must not move the bill past the context.
    const sentinel: u64 = omittedMaxTokensDefault(@intCast(@min(getEffectiveContextLength(&cfg), 1_048_576)));
    const with_sentinel = prefillRequestTerms(&cfg, seq, sentinel, 8, chunk, .{});
    const with_explicit = prefillRequestTerms(&cfg, seq, 2048, 8, chunk, .{});
    try t.expect(with_sentinel.reserved_kv_bytes < 8 * 1024 * 1024 * 1024);
    try t.expect(with_sentinel.state_bytes < 4 * with_explicit.state_bytes);
    try t.expectEqual(with_explicit.checkpoint_bytes, with_sentinel.checkpoint_bytes);

    // A short prompt is untouched.
    const short = prefillRequestTerms(&cfg, 4096, 2048, 8, chunk, .{});
    try t.expectEqual(@as(u64, 0), short.reserved_kv_bytes);
    try t.expectEqual(4096 * statePerTokenBilled(&cfg), short.state_bytes);
}

/// The generic admission bill for `qwen4ExpOomConfig`, minus the live-memory read a unit test cannot make.
fn oomBillFor(cfg: *const model_mod.ModelConfig, seq: u64, chunk: u64, terms: PrefillRequestTerms) u64 {
    return prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, prefillFfnWidth(cfg), 8, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(cfg), prefillDequantWeightBytes(cfg), terms) +
        qsaMaskBytes(cfg, @min(chunk, @max(seq, 1)), seq);
}

test "an omitted max_tokens reserves a generation HEADROOM, not the rest of the context" {
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    // A 383,067-token prompt with no `max_tokens` was admitted by eviction and then refused on
    // the inference thread: `clampMaxTokens` made the reservation cover the whole context
    // (~13 GB for a generation that would never run). The reservation exists to stop buffer
    // coexistence during prefill, not to pre-buy a generation.
    const KVCache = transformer_mod.KVCache;
    const MB: u64 = 1024 * 1024;
    const cfg = qwen4ExpOomConfig();
    const ctx: u64 = getEffectiveContextLength(&cfg);
    try t.expectEqual(@as(u64, 1_048_576), ctx);
    const seq: u64 = 383_067;
    const chunk: u64 = 4096;
    const clamped: u64 = clampMaxTokens(@intCast(omittedMaxTokensDefault(@intCast(ctx))), @intCast(seq), @intCast(ctx));
    try t.expectEqual(ctx - seq, clamped); // 665,509 — the live number

    // The reservation is bounded by the headroom, not by the context.
    const reserved = KVCache.reservedTokens(seq, clamped, chunk, ctx);
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + chunk, reserved);
    try t.expect(reserved < seq + ctx / 8);
    try t.expectEqual(seq + 512 + chunk, KVCache.reservedTokens(seq, 512, chunk, ctx));
    try t.expectEqual(ctx + chunk, KVCache.reservedTokens(ctx - 16, clamped, chunk, ctx));

    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8);
    const terms = prefillRequestTerms(&cfg, seq, clamped, 8, chunk, .{});
    try t.expectEqual((KVCache.RESERVE_GEN_HEADROOM + chunk) * kv_per_tok, terms.reserved_kv_bytes);
    try t.expectEqual(reserved * statePerTokenBilled(&cfg), terms.state_bytes);

    // ...and the whole bill drops by more than 10 GB, from above the live headroom to under it.
    const ctx_bounded: PrefillRequestTerms = .{
        .reserved_kv_bytes = (ctx + chunk - seq) * kv_per_tok,
        .state_bytes = (ctx + chunk) * statePerTokenBilled(&cfg),
        .checkpoint_bytes = terms.checkpoint_bytes,
    };
    const bill = oomBillFor(&cfg, seq, chunk, terms);
    const old_bill = oomBillFor(&cfg, seq, chunk, ctx_bounded);
    try t.expect(old_bill > bill + 10 * 1024 * MB);
    try t.expect(old_bill > 28_000 * MB);
    try t.expect(bill < 30_000 * MB); // under the live "~37130MB available"
}

test "the guard credits only PROVABLY reclaimable cache bytes, never the entry a restore will pin" {
    const t = std.testing;
    // The guard admitted on `available + evictable >= needed`, but the evictable 1,564 MB was
    // the one entry this prompt then restored from; the provable credit is residency minus the largest entry.
    const MB: u64 = 1024 * 1024;
    var bill = AdmissionBill{ .needed = 37_394 * MB, .available = 37_130 * MB, .evictable = 1_564 * MB, .reclaimable = 0 };
    try t.expect(!bill.fits());
    try t.expect(!bill.fitsAfterEviction()); // it DID, before this fix
    bill.reclaimable = 1_564 * MB;
    try t.expect(bill.fitsAfterEviction());
}

test "the admission probe bills the request's OWN kv-quant and chunking, not the process defaults" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The probe used to hardcode `kv_override = null`, pricing a `kv_quant: 4` request at fp16 on the inference thread.
    const t = std.testing;
    const cfg = qwen4ExpOomConfig();
    const seq: usize = 200_000;
    const q4 = transformer_mod.KVQuantConfig.affine(4);
    const q8 = transformer_mod.KVQuantConfig.affine(8);
    const bill4 = prefillAdmissionBill(&cfg, seq, 2048, q4, false, null, .{});
    const bill8 = prefillAdmissionBill(&cfg, seq, 2048, q8, false, null, .{});
    const bill16 = prefillAdmissionBill(&cfg, seq, 2048, transformer_mod.KVQuantConfig.dense, false, null, .{});
    try t.expect(bill4.needed < bill8.needed);
    try t.expect(bill8.needed < bill16.needed);
}

test "the QSA history is billed at the one copy a slot HOLDS, plus the score bank" {
    const t = std.testing;
    const cfg = qwen4ExpOomConfig();
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const one = cfg.qsaHistoryBytesPerToken();
    const bank = cfg.qsaScoreBankBytesPerToken();
    try t.expect(one > 0);
    try t.expect(bank > 0);
    try t.expectEqual(one + bank, statePerTokenBilled(&cfg));

    const seq: u64 = 458_832;
    const terms = prefillRequestTerms(&cfg, seq, 2048, 8, 4096, .{});
    try t.expectEqual((seq + 2048 + 4096) * (one + bank), terms.state_bytes);

    var dense = model_mod.ModelConfig{};
    dense.num_hidden_layers = 32;
    dense.num_attention_heads = 32;
    try t.expectEqual(@as(u64, 0), statePerTokenBilled(&dense));
}

test "ssdFirstPrefixCacheMem: the active session's KV is billed ONCE and is the floor" {
    const t = std.testing;
    const GB: u64 = 1 << 30;
    // The 1M qwen4 case: 107.5 GB ceiling, ~70 GB weights, ~24 GB of 1M-ctx KV, ~3 GB reserve.
    const ceiling: u64 = 107 * GB;
    const weights: u64 = 70 * GB;
    const ctx_kv: u64 = 24 * GB;
    const transient: u64 = 3 * GB;

    // RAM-first clamps the cache to the leftover: less than half an entry.
    const ram_first = clampedPrefixCacheMem(0, ceiling, weights, ctx_kv, transient);
    try t.expect(ram_first < ctx_kv);

    // SSD-first floors at one entry and hands `--prefix-cache-mem` to idle entries on top.
    try t.expectEqual(ctx_kv, ssdFirstPrefixCacheMem(0, ceiling, weights, ctx_kv, transient));
    try t.expectEqual(ctx_kv + 4 * GB, ssdFirstPrefixCacheMem(4 * GB, ceiling, weights, ctx_kv, transient));
    const huge = ssdFirstPrefixCacheMem(500 * GB, ceiling, weights, ctx_kv, transient);
    try t.expectEqual(ctx_kv + (ceiling - weights - ctx_kv - transient), huge);
    try t.expectEqual(ctx_kv, ssdFirstPrefixCacheMem(8 * GB, 80 * GB, weights, ctx_kv, transient));
    try t.expect(ssdFirstPrefixCacheMem(0, 0, 0, 0, 0) >= 1);
}

/// The widest width the adaptive chooser may take for a prompt on this config.
fn adaptCapFor(cfg: *const model_mod.ModelConfig, seq: u64) u32 {
    return widthForRung(cfg, seq, PREFILL_CHUNK_LADDER[0]);
}

test "adaptivePrefillWidth: a step-down is immediate, by as many rungs as it takes, and never 0" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The down direction does not wait for a trend or a margin.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 1_048_576;
    const kv: u64 = seq;
    const cap = adaptCapFor(&cfg, seq);

    {
        var st: mtp_plan.AdaptiveWidthState = .{};
        const h = prefillChunkCost(&cfg, kv_bits, 2048, kv);
        try t.expect(h < prefillChunkCost(&cfg, kv_bits, cap, kv));
        try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, h, cap, cap, &st));
        try t.expect(st.ratcheted);
    }
    // Several rungs in one call.
    {
        var st: mtp_plan.AdaptiveWidthState = .{};
        const h = prefillChunkCost(&cfg, kv_bits, 512, kv);
        try t.expectEqual(@as(u32, 512), adaptivePrefillWidth(&cfg, kv_bits, kv, h, cap, cap, &st));
    }
    // No headroom at all: the ladder floor, never 0 (the loop reads 0 as "no change").
    {
        var st: mtp_plan.AdaptiveWidthState = .{};
        const got = adaptivePrefillWidth(&cfg, kv_bits, kv, 0, cap, cap, &st);
        try t.expectEqual(@as(u32, @intCast(prefill_plan.PREFILL_CHUNK_FLOOR)), got);
        try t.expect(got != 0);
    }
    {
        var st: mtp_plan.AdaptiveWidthState = .{};
        const floor: u32 = @intCast(prefill_plan.PREFILL_CHUNK_FLOOR);
        try t.expectEqual(floor, adaptivePrefillWidth(&cfg, kv_bits, kv, 0, floor, cap, &st));
    }
}

test "adaptivePrefillWidth: HOLD is margin 1.0 — the admitted width is not second-guessed" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `prefillMemoryNeeded` already carries its own 5/4; a second 1.25 on the hold stepped the
    // 768k prompt down from the width admission admitted.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const kv: u64 = 786_432;
    const cap = adaptCapFor(&cfg, 786_432);
    const cost_2k = prefillChunkCost(&cfg, kv_bits, 2048, kv);

    var st: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, cost_2k, 2048, cap, &st));
    try t.expect(!st.ratcheted);
    try t.expect(cost_2k * 5 / 4 > cost_2k);
    var st2: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, cost_2k * 5 / 4 - 1, 2048, cap, &st2));
    var st3: mtp_plan.AdaptiveWidthState = .{};
    try t.expect(adaptivePrefillWidth(&cfg, kv_bits, kv, cost_2k - 1, 2048, cap, &st3) < 2048);
}

test "adaptivePrefillWidth: a widen costs 1.25x AND two consecutive supporting probes" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Holding is free, growing is a bet: two consecutive probes.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const kv: u64 = 524_288;
    const cap = adaptCapFor(&cfg, 524_288);
    // Starts at half the cap so one widen step lands exactly on the cap.
    const from: u32 = cap / 2;
    const cost_up = prefillChunkCost(&cfg, kv_bits, cap, kv);

    var st: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st));
    try t.expectEqual(@as(u8, 1), st.supporting);
    try t.expectEqual(cap, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st));
    try t.expectEqual(@as(u8, 0), st.supporting); // and the count restarts at the new width

    // Room for the wider chunk's bill but not its margin: never.
    var st2: mtp_plan.AdaptiveWidthState = .{};
    const between = cost_up * 5 / 4 - 1;
    try t.expect(between >= cost_up);
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, between, from, cap, &st2));
    }

    // A single unsupporting probe resets the run.
    var st3: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st3));
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, between, from, cap, &st3));
    try t.expectEqual(@as(u8, 0), st3.supporting);
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st3));
}

test "the tail-merge gate reads the ARCH, not the installed hook (serve installs it for everyone)" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `serve` installs `prefill_chunk_adapt` unconditionally, so `chunk_width_hook != null` is
    // true on every arch and reading it as the gate put the scaled bound back everywhere. This
    // builds the slot state the way `serve` does and asserts the two answers differ.
    const t = std.testing;
    scheduler_mod.prefill_chunk_adapt = &adaptivePrefillWidthNow;
    defer scheduler_mod.prefill_chunk_adapt = null;
    scheduler_mod.prefill_chunk_widen_ok = &adaptivePrefillWidenStillFits;
    defer scheduler_mod.prefill_chunk_widen_ok = null;
    scheduler_mod.prefill_chunk_adaptive_enabled = &adaptivePrefillChunkEnabled;
    defer scheduler_mod.prefill_chunk_adaptive_enabled = null;

    const hook_installed = scheduler_mod.prefill_chunk_adapt != null;
    try t.expect(hook_installed);

    var other = qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";
    const adaptive = scheduler_mod.adaptiveChunkWidthFor(&other);
    try t.expect(!adaptive);
    try t.expect(hook_installed != adaptive);

    // A slot without the adaptive hook keeps the flat bound.
    try t.expectEqual(prefill_plan.TAIL_MERGE_MAX, prefill_plan.tailMergeMaxFor(512, adaptive));
    try t.expectEqual(@as(usize, 812), prefill_plan.nextChunkEnd(0, 812, 512, false, 0, 0, adaptive));
    try t.expectEqual(@as(usize, 1), prefill_plan.prefillChunkCount(812, 512, false, 0, 0, adaptive));

    const cfg = qwen4RequestTestConfig();
    per_request_chunk_override = true;
    defer per_request_chunk_override = null;
    adaptive_chunk_override = true;
    defer adaptive_chunk_override = null;
    const unpinned = explicitPrefillChunk() == 0;
    try t.expectEqual(unpinned, scheduler_mod.adaptiveChunkWidthFor(&cfg));
    if (unpinned) {
        try t.expectEqual(@as(usize, 64), prefill_plan.tailMergeMaxFor(512, scheduler_mod.adaptiveChunkWidthFor(&cfg)));
        try t.expectEqual(@as(usize, 512), prefill_plan.nextChunkEnd(0, 812, 512, false, 0, 0, scheduler_mod.adaptiveChunkWidthFor(&cfg)));
    }
    try t.expect(!scheduler_mod.adaptiveChunkWidthFor(null));
    scheduler_mod.prefill_chunk_adaptive_enabled = null;
    try t.expect(!scheduler_mod.adaptiveChunkWidthFor(&cfg));
    try t.expect(scheduler_mod.prefill_chunk_adapt != null); // ...with the hook still installed
}

test "the tail-merge bound scales ONLY where the per-chunk adaptive width is live" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Every arch but an adaptive-width qwen4_exp keeps the flat `TAIL_MERGE_MAX` at every chunk width.
    const t = std.testing;

    var other = qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";
    try t.expect(!other.perRequestPrefillChunk());
    try t.expect(!adaptivePrefillChunkEnabled(&other));
    const off = adaptivePrefillChunkEnabled(&other);
    try t.expectEqual(prefill_plan.TAIL_MERGE_MAX, prefill_plan.tailMergeMaxFor(512, off));
    try t.expectEqual(prefill_plan.TAIL_MERGE_MAX, prefill_plan.tailMergeMaxFor(2048, off));
    try t.expectEqual(@as(usize, 812), prefill_plan.nextChunkEnd(0, 812, 512, false, 0, 0, off));

    const cfg = qwen4RequestTestConfig();
    try t.expect(cfg.perRequestPrefillChunk());
    per_request_chunk_override = true;
    defer per_request_chunk_override = null;
    adaptive_chunk_override = true;
    defer adaptive_chunk_override = null;
    const unpinned = explicitPrefillChunk() == 0;
    try t.expectEqual(unpinned, adaptivePrefillChunkEnabled(&cfg));
    try t.expectEqual(@as(usize, 64), prefill_plan.tailMergeMaxFor(512, true));
    try t.expectEqual(@as(usize, 512), prefill_plan.nextChunkEnd(0, 812, 512, false, 0, 0, true));
    // An operator pin turns the scaling back off.
    adaptive_chunk_override = false;
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
    try t.expectEqual(prefill_plan.TAIL_MERGE_MAX, prefill_plan.tailMergeMaxFor(512, adaptivePrefillChunkEnabled(&cfg)));
}

test "adaptivePrefillWidth: the ratchet is one-way, and the cap is a ceiling" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const kv: u64 = 1_048_576;
    const cap = adaptCapFor(&cfg, 1_048_576);

    // Step down once, then hand it the whole machine: it stays (a re-widen would be a metronome).
    var st: mtp_plan.AdaptiveWidthState = .{};
    const narrowed = adaptivePrefillWidth(&cfg, kv_bits, kv, prefillChunkCost(&cfg, kv_bits, 1024, kv), cap, cap, &st);
    try t.expect(narrowed < cap);
    try t.expect(st.ratcheted);
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(narrowed, adaptivePrefillWidth(&cfg, kv_bits, kv, std.math.maxInt(u64) / 2, narrowed, cap, &st));
    }

    // Nothing above the arch cap, ever.
    var st2: mtp_plan.AdaptiveWidthState = .{};
    i = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(cap, adaptivePrefillWidth(&cfg, kv_bits, kv, std.math.maxInt(u64) / 2, cap, cap, &st2));
    }
    // A cap of 0 (the unchunked vision arm) is "do not touch the width".
    var st3: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, 0, 2048, 0, &st3));
}

test "adaptivePrefillWidth: a 1M prompt walks DOWN as the box fills, and a small one never moves" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Sweep: the width is the widest rung that fits, only narrows within one prefill, and is always forwardable.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const kv: u64 = 1_048_576;
    const cap = adaptCapFor(&cfg, 1_048_576);

    var st: mtp_plan.AdaptiveWidthState = .{};
    var width: u32 = cap;
    var seen_narrower: usize = 0;
    var headroom: u64 = 50 * 1024 * MiB; // the wired-limit box, quiet
    var step: usize = 0;
    while (step < 64) : (step += 1) {
        const next = adaptivePrefillWidth(&cfg, kv_bits, kv, headroom, width, cap, &st);
        try t.expect(next <= width); // never widens after the first step-down
        try t.expect(next >= @as(u32, @intCast(prefill_plan.PREFILL_CHUNK_FLOOR)));
        try t.expect(next <= cap);
        try t.expect(prefillChunkCost(&cfg, kv_bits, next, kv) <= headroom or
            next == @as(u32, @intCast(prefill_plan.PREFILL_CHUNK_FLOOR)));
        if (next < width) seen_narrower += 1;
        width = next;
        headroom -|= 1024 * MiB; // a co-tenant, or the accreting state
    }
    try t.expect(seen_narrower >= 2);
    try t.expectEqual(@as(u32, @intCast(prefill_plan.PREFILL_CHUNK_FLOOR)), width);

    // A 32k prompt on the same box: the width never moves.
    var st2: mtp_plan.AdaptiveWidthState = .{};
    const cap32 = adaptCapFor(&cfg, 32_768);
    var w32: u32 = cap32;
    step = 0;
    while (step < 16) : (step += 1) {
        w32 = adaptivePrefillWidth(&cfg, kv_bits, 32_768, 50 * 1024 * MiB, w32, cap32, &st2);
        try t.expectEqual(cap32, w32);
    }
    try t.expect(!st2.ratcheted);
}

test "prefillHeadroomNow subtracts the SSD writer's staged HOST bytes" {
    // Staged host bytes come out of the same unified pool the widen bar reads.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    try t.expectEqual(90 * GiB, prefillHeadroomFrom(120 * GiB, 30 * GiB, 0));
    try t.expectEqual(89 * GiB, prefillHeadroomFrom(120 * GiB, 30 * GiB, GiB));
    // Saturating: a permit larger than the headroom refuses every widen.
    try t.expectEqual(@as(u64, 0), prefillHeadroomFrom(120 * GiB, 119 * GiB, 8 * GiB));
}

test "adaptivePrefillWidth: the widen prices the QSA sheet at the LIVE KV, not at one chunk" {
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `prefillTransientReserve` bills the QSA sheet at `kv = chunk`; past the indexer budget the
    // sheet costs its whole budget, so a widen decided on that number is unguarded.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const cap = adaptCapFor(&cfg, 786_432);
    const short_kv: u64 = 4096;
    const long_kv: u64 = 400_000;

    // The bill is strictly larger at the live KV.
    const at_short = prefillChunkCost(&cfg, kv_bits, cap, short_kv);
    const at_long = prefillChunkCost(&cfg, kv_bits, cap, long_kv);
    try t.expect(at_long > at_short);
    // `prefillTransientReserve` is exactly the `kv = chunk` case.
    try t.expectEqual(
        prefillTransientReserve(&cfg, kv_bits, cap),
        prefillTransientReserveAtKv(&cfg, kv_bits, cap, cap),
    );

    // Headroom that clears the short bill's 1.25 bar but not the long one: widen at 4k, refuse at 400k.
    const headroom: u64 = at_short * 5 / 4 + 1;
    try t.expect(headroom < at_long * 5 / 4);

    // Starts at half the cap so one widen step lands exactly on the cap.
    const from: u32 = cap / 2;
    var st_short: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, short_kv, headroom, from, cap, &st_short));
    try t.expectEqual(cap, adaptivePrefillWidth(&cfg, kv_bits, short_kv, headroom, from, cap, &st_short));

    var st_long: mtp_plan.AdaptiveWidthState = .{};
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, long_kv, headroom, from, cap, &st_long));
    }
    try t.expectEqual(@as(u8, 0), st_long.supporting);

    // The hold keeps its margin-1.0 semantics at the live KV.
    const hold = prefillChunkCost(&cfg, kv_bits, 2048, long_kv);
    var st_hold: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, long_kv, hold, 2048, cap, &st_hold));
    var st_down: mtp_plan.AdaptiveWidthState = .{};
    try t.expect(adaptivePrefillWidth(&cfg, kv_bits, long_kv, hold - 1, 2048, cap, &st_down) < 2048);
}

test "adaptivePrefillChunkEnabled: the arch, both kill switches, and any pinned width" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    var other = qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";

    try t.expect(adaptivePrefillChunkEnabled(&cfg));
    // Another arch never adapts; one switch disables both.
    try t.expect(!adaptivePrefillChunkEnabled(&other));

    per_request_chunk_override = false;
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
    per_request_chunk_override = null;
    try t.expect(adaptivePrefillChunkEnabled(&cfg));

    adaptive_chunk_override = false;
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
    adaptive_chunk_override = null;

    // An operator who pinned a width pinned every forward.
    generate_mod.prefill_chunk_override = 4096;
    generate_mod.prefill_chunk_explicit = true;
    defer {
        generate_mod.prefill_chunk_override = 8192;
        generate_mod.prefill_chunk_explicit = false;
    }
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
}

test "resolvedContextForLoad: an auto boot bills the session it will serve, not the placeholder" {
    // On an auto boot the effective-context accessor answered a 1024-token placeholder during the load.
    const t = std.testing;
    const MiB: u64 = 1 << 20;
    // This box: ~28,909 MiB after the weights, 20,736 B/token at 1M/kv8, ~3 GiB transient.
    const per_tok: u64 = 20_736;
    const active: u64 = 70_000 * MiB;
    const ceiling: u64 = active + 28_909 * MiB;
    const transient: u64 = 3 * 1024 * MiB;
    const cap: u32 = 1_048_576;

    // Auto boot, nothing pinned: the session is what the machine can serve. `cache_reserve = 0`
    // is a test isolation.
    const auto = resolvedContextForLoad(0, 0, ceiling, active, 0, transient, per_tok, cap);
    try t.expect(auto > 100_000);
    try t.expect(auto <= cap);

    // The placeholder bills ~26 MB of session and hands the idle allowance almost the whole machine.
    const placeholder_kv: u64 = per_tok * 1024;
    const real_kv: u64 = per_tok * auto;
    try t.expect(placeholder_kv < 30 * MiB);
    try t.expect(real_kv > 100 * placeholder_kv);
    const ask: u64 = 24 * 1024 * MiB;
    const bogus = ssdFirstPrefixCacheMem(ask, ceiling, active, placeholder_kv, transient);
    const fixed = ssdFirstPrefixCacheMem(ask, ceiling, active, real_kv, transient);
    try t.expect(bogus -| placeholder_kv == ask);
    // Under the real session bill the same ask is clamped to what is left after the weights, one
    // session and the transient (the helper's own contract, not a hand-picked byte count).
    const idle_headroom: u64 = ceiling - (active + real_kv + transient);
    try t.expectEqual(real_kv + idle_headroom, fixed);
    try t.expect(fixed -| real_kv < ask);
    try t.expect(fixed -| real_kv < (bogus -| placeholder_kv) / 2);

    // An explicit --ctx-size wins outright, and a pinned context is used as-is.
    try t.expectEqual(@as(u32, 262_144), resolvedContextForLoad(262_144, 0, ceiling, active, 0, transient, per_tok, cap));
    try t.expectEqual(@as(u32, 131_072), resolvedContextForLoad(0, 131_072, ceiling, active, 0, transient, per_tok, cap));
    try t.expectEqual(cap, resolvedContextForLoad(0, 0, active + 900_000 * MiB, active, 0, transient, per_tok, cap));
}

test "the advertised context does not move with the cache ask" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Sizing the context against the ask collapsed it (60GB ask -> 870 tokens); sizing against the
    // resolved budget was a one-step loop. A constant reserve closes both.
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const live_ceiling: u64 = 109_395 * MiB;
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);
    const transient: u64 = prefillTransientReserve(&cfg, kv_bits, 4096);

    // `max_pos` is 0: the checkpoint cap is applied by the caller after the margin.
    const memory_ctx = safeContextForBudget(live_ceiling, active, CTX_SIZING_CACHE_RESERVE +| transient, per_tok, 0);
    try t.expect(memory_ctx > 900_000);

    // The two defect spellings.
    const vs_ask = safeContextForBudget(live_ceiling, active, 60 * 1024 * MiB +| transient, per_tok, 0);
    const vs_resolved = safeContextForBudget(live_ceiling, active, 48_673 * MiB +| transient, per_tok, 0);
    try t.expect(vs_ask <= 1024);
    try t.expect(vs_resolved <= 1024);

    // The pair agrees at the advertised number: both sides go through `autoContextFrom`.
    const advertised = autoContextFrom(memory_ctx, cfg.contextCap());
    const clamp_ctx = resolvedContextForLoad(0, 0, live_ceiling, active, CTX_SIZING_CACHE_RESERVE, transient, per_tok, cfg.contextCap());
    try t.expectEqual(advertised, clamp_ctx);
}

test "the published hot-cache budget is retired when the cache is dropped" {
    // The budget is per model, the global is process-wide.
    const t = std.testing;
    const tiers = TierGuard.pin(true, 0);
    defer tiers.restore();
    publishResolvedPrefixCacheMem(4096);
    try t.expectEqual(@as(u64, 4096), resolvedPrefixCacheMem());
    clearResolvedPrefixCacheMem();
    // Back to the ask, the honest pre-load answer.
    try t.expectEqual(prefix_cache_mem_bytes, resolvedPrefixCacheMem());

    // A resolved zero is a real budget (`--prefix-cache-mem 0` under SSD-first); the sentinel is maxInt.
    publishResolvedPrefixCacheMem(0);
    try t.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
    clearResolvedPrefixCacheMem();
    try t.expectEqual(prefix_cache_mem_bytes, resolvedPrefixCacheMem());
}

test "a WARM turn is not billed for the prefix it is about to SHARE (786,707 @ 19,032 MB)" {
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    const t = std.testing;
    // Live: a 786,707-token warm turn restored its own 786,676-token entry and was then refused
    // by name; the restore hands over the entry's buffers without a second copy, so those
    // ~14.5 GB were already inside `mlx_get_active_memory` and `needed` billed them again.
    const cfg = qwen4ExpOomConfig();
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const seq: u64 = 786_707;
    const chunk: u64 = 512; // the width the ladder came down to that evening
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 200;
    const mb: u64 = 1024 * 1024;
    const available: u64 = 19_032 * mb;

    // 13,056 B of 8-bit KV and 5,376 B of state per token (one history copy + the score bank).
    // The credit prices KV only: the history is privatised on restore.
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    try t.expectEqual(@as(u64, 13_056), kv_per_tok);
    try t.expectEqual(@as(u64, 768), cfg.qsaHistoryBytesPerToken());
    try t.expectEqual(@as(u64, 2_304), statePerTokenBilled(&cfg));

    // Cold: the refusal, reproduced.
    const cold = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{});
    try t.expect(cold / mb > 18_000 and cold / mb < 23_000);
    try t.expect(cold > available);

    // Warm: the entry's buffers hold the capacity the previous turn reserved, which still covers
    // this turn's reservation, so nothing grows.
    const matched: u64 = 786_676;
    const capacity: u64 = 787_456;
    const reserved = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    try t.expectEqual(@as(u64, 787_419), reserved);
    try t.expect(reserved <= capacity);

    // `will_donate`: this restore checked out its entry, so the buffers are the slot's own.
    const warm_prefix = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = capacity, .will_donate = true };
    try t.expectEqual(matched, warm_prefix.creditedRows(reserved));

    // The credit is the handed-over rows at this arch's KV width and nothing else, inside the 5/4.
    const credit = matched * kv_per_tok;
    try t.expectEqual(@as(u64, 786_676 * 13_056), credit);
    try t.expect(credit < matched * (kv_per_tok + cfg.qsaHistoryBytesPerToken()));
    const warm = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, warm_prefix);
    // Isolated against a shared restore of the same span: `checkpoint_bytes` follows `matched` too.
    const shared_same_span = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{ .matched_tokens = matched, .capacity_tokens = capacity });
    try t.expect(shared_same_span < cold);
    try t.expectEqual(shared_same_span - credit * 5 / 4, warm);

    // The verdict flips: admitted outright, no eviction pass. Under-crediting is the safe direction.
    try t.expect(warm < available);
    try t.expect(warm / mb > 6_500 and warm / mb < 8_500);

    // A cold prompt on the same box keeps the old refusal.
    try t.expect(prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{}) > available);
    try t.expect(prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{ .matched_tokens = 0, .capacity_tokens = capacity, .will_donate = true }) > available);
}

test "a chain extension that OUTGROWS the resident entry is credited the rows it holds" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    // Past the restored capacity the cache grows per layer: the resident rows stay credited and
    // the grow bills the new capacity beyond the old plus one eval window of old buffers.
    const cfg = qwen4ExpOomConfig();
    const seq: u64 = 1_047_556;
    const chunk: u64 = 512;
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 200;
    const capacity: u64 = 787_456; // what the 786k turn reserved

    const reserved = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    try t.expectEqual(@as(u64, 1_048_268), reserved);
    const extend = WarmPrefix{ .matched_tokens = 786_707, .capacity_tokens = capacity, .will_donate = true };
    try t.expect(reserved > capacity);
    try t.expect(extend.grows(seq));
    try t.expectEqual(@as(u64, 786_707), extend.creditedRows(reserved));
    const per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const ext_terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, extend);
    const window = attnLayersPerEvalWindow(&cfg, transformer_mod.Transformer.MOE_EVAL_EVERY_N_LAYERS);
    try t.expectEqual(capacity * per_tok / cfg.attnCacheLayerCount() * window, ext_terms.grow_coexist_bytes);
    try t.expectEqual(
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{}) -
            (786_707 * per_tok - ext_terms.grow_coexist_bytes) * 5 / 4,
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, extend),
    );

    // A buffer that already covers the reservation credits the full shared prefix: the gate is capacity, not prompt length.
    const roomy = WarmPrefix{ .matched_tokens = 786_707, .capacity_tokens = 1_048_576, .will_donate = true };
    try t.expectEqual(@as(u64, 786_707), roomy.creditedRows(reserved));
    const credit = 786_707 * kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    try t.expectEqual(
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{}) - credit * 5 / 4,
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, roomy),
    );
}

/// The live 768k scenario's checkout inputs, as `prefix_cache.checkoutEligible` takes them.
const CheckoutCase = struct {
    ssd_first: bool = true,
    move_enabled: bool = true,
    pending_disk: bool = false,
    entry_tokens: usize = 786_695,
    shared: usize = 786_695,
    prompt_len: usize = 786_707,
    has_slot: bool = true,

    /// The WarmPrefix the scheduler builds, with `will_donate` from restore-move's own predicate.
    fn warm(self: CheckoutCase, matched: u64, capacity: u64) WarmPrefix {
        return .{
            .matched_tokens = matched,
            .capacity_tokens = capacity,
            .will_donate = prefix_cache_mod.HotPrefixCache.checkoutEligible(
                self.ssd_first,
                self.move_enabled,
                self.pending_disk,
                self.entry_tokens,
                self.shared,
                self.prompt_len,
                self.has_slot,
            ),
        };
    }
};

test "the warm KV credit fires ONLY where the restore checked its entry out" {
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    // `KVCache.restore` binds by refcount and mlx donates only at `use_count() == 1`, so on a
    // shared restore the first `writeAtOffset` copies the whole prefix: the credited bytes are
    // allocated. Only the checkout makes the credit true; it used to fire on `matched > 0` alone.
    const cfg = qwen4ExpOomConfig();
    const seq: u64 = 786_707;
    const chunk: u64 = 512;
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 200;
    const mb: u64 = 1024 * 1024;
    const matched: u64 = 786_676;
    const capacity: u64 = 787_456;
    const reserved = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    const cold = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{});

    // (1) qwen4_exp, SSD-first, full-prefix hit, switch on: the credit is real.
    const donating = (CheckoutCase{}).warm(matched, capacity);
    try t.expect(donating.will_donate);
    try t.expectEqual(matched, donating.creditedRows(reserved));
    const warm = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, donating);
    // Baseline = a shared restore of the same span, so the two credits are not folded into one comparison.
    const shared_same_span = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{ .matched_tokens = matched, .capacity_tokens = capacity });
    try t.expectEqual(shared_same_span - matched * kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) * 5 / 4, warm);
    try t.expect(warm / mb > 6_500 and warm / mb < 8_500);

    // Every arm below is the same restore minus one conjunct of the checkout, and every one shares.
    const shared_arms = [_]WarmPrefix{
        // (2) a partial hit: the entry describes a prefix this prompt left.
        (CheckoutCase{ .shared = 700_000 }).warm(matched, capacity),
        // (3) a non-qwen4 arch: a checkout can never be taken there.
        (CheckoutCase{ .ssd_first = false }).warm(matched, capacity),
        // (4) restore by move off: the refcount share.
        (CheckoutCase{ .move_enabled = false }).warm(matched, capacity),
        // (5) a pending disk record shares the same buffers.
        (CheckoutCase{ .pending_disk = true }).warm(matched, capacity),
        // (6) no slot to hand the buffers to (the connection thread's shape).
        (CheckoutCase{ .has_slot = false }).warm(matched, capacity),
    };
    for (shared_arms, 0..) |arm, i| {
        errdefer std.debug.print("shared arm {d} credited\n", .{i});
        try t.expect(!arm.will_donate);
        try t.expectEqual(@as(u64, 0), arm.creditedRows(reserved));
        try t.expectEqual(shared_same_span, prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, arm));
    }
    // ...and `cold` is still the number a prompt with NO resident prefix pays,
    // which is what the `.{}` arm above is for.
    try t.expect(cold > shared_same_span);

    // Capacity is a CEILING on the credit, not a switch: a buffer holding 700,000 rows donates
    // them and the request is billed the growth on top.
    const outgrown = (CheckoutCase{}).warm(matched, 700_000);
    try t.expect(outgrown.will_donate);
    try t.expectEqual(@as(u64, 700_000), outgrown.creditedRows(reserved));
    try t.expect(outgrown.grows(seq));
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const og_terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, outgrown);
    // 31 tokens wide, BELOW `prefillEvalCadenceApplies`: one eval for the whole layer loop, so
    // every old buffer is alive at once and the coexistence term cancels the credit exactly.
    try t.expect(seq - matched < 32);
    try t.expectEqual(700_000 * kv_per_tok, og_terms.grow_coexist_bytes);
    try t.expectEqual(shared_same_span, prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, outgrown));
    // An append wide enough to take the cadence pays ONE eval window instead, and that is the
    // difference between the old bill and the new one.
    const wide = WarmPrefix{ .matched_tokens = 700_000, .capacity_tokens = 700_000, .will_donate = true };
    const wide_terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, wide);
    const window = attnLayersPerEvalWindow(&cfg, transformer_mod.Transformer.MOE_EVAL_EVERY_N_LAYERS);
    try t.expect(window < cfg.attnCacheLayerCount());
    try t.expectEqual(700_000 * kv_per_tok / cfg.attnCacheLayerCount() * window, wide_terms.grow_coexist_bytes);
    try t.expect(prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, wide) < shared_same_span);
}

test "pinnedResidentBytes names the entry a restore would share, and only that" {
    const t = std.testing;
    // Residency the eviction pass cannot return: the connection thread's one signal that this prompt is warm.
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(.{ .evictable = 0, .reclaimable = 0 }));
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(.{ .evictable = 4096, .reclaimable = 4096 }));
    // The live 768k reading: 24,826 MB resident, 10,294 MB reclaimable.
    const mb: u64 = 1024 * 1024;
    try t.expectEqual(@as(u64, 14_532 * mb), pinnedResidentBytes(.{ .evictable = 24_826 * mb, .reclaimable = 10_294 * mb }));
    // Saturating: the two numbers are published from different sites.
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(.{ .evictable = 1, .reclaimable = 2 }));
}

test "prefillChunkCap: the chunk sizer's two long-context changes are qwen4_exp-only" {
    // Both sizer changes (dropping the ask, the ctx bar) moved the prefill width on every arch;
    // the ungated arithmetic is the previous `resolvePrefillChunk`.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const ceiling: u64 = 120 * GiB;
    const weights: u64 = 69 * GiB;
    const ask: u64 = 10 * GiB;
    const ctx_kv: u64 = 20 * GiB;

    // Every other arch: the ask is spoken for, the context is not, no HOT_CACHE_FLOOR.
    for ([_][]const u8{ "glm5_next", "not_a_served_arch" }) |mt| {
        var cfg = longCtxTestConfig();
        cfg.model_type = mt;
        const baseline: u64 = (ceiling - (weights + ask)) / PREFILL_RESERVE_BUDGET_SHARE;
        try t.expectEqual(baseline, prefillChunkCap(&cfg, ceiling, weights, ctx_kv, ask));
        try t.expectEqual(baseline, prefillChunkCap(&cfg, ceiling, weights, 48 * GiB, ask));
        try t.expect(prefillChunkCap(&cfg, ceiling, weights, ctx_kv, 0) > baseline);
    }

    // qwen4_exp: the ask is ignored, and the ctx bar binds.
    var q4 = longCtxTestConfig();
    q4.model_type = "qwen4_exp";
    const serving: u64 = ceiling - weights;
    const share_bar: u64 = serving / PREFILL_RESERVE_BUDGET_SHARE;
    const want: u64 = @min(share_bar, (serving - ctx_kv) -| HOT_CACHE_FLOOR_BYTES);
    try t.expectEqual(want, prefillChunkCap(&q4, ceiling, weights, ctx_kv, ask));
    try t.expectEqual(
        prefillChunkCap(&q4, ceiling, weights, ctx_kv, 0),
        prefillChunkCap(&q4, ceiling, weights, ctx_kv, 24 * GiB),
    );
    try t.expect(prefillChunkCap(&q4, ceiling, weights, 48 * GiB, ask) < share_bar);
}

test "prefillRequestTerms: the admission bill's new terms are qwen4_exp-only" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The allocator side of the reservation is gated (`generate.reservedPrefillTokens`); the
    // guard side was not, so off qwen4_exp the bill charged headroom never reserved.
    const t = std.testing;
    const seq: u64 = 200_000; // well past RESERVE_MIN_TOKENS
    const chunk: u64 = 4096;
    const max_tokens: u64 = 8192;
    const kv_bits: u64 = 8;

    // A 27B-class GDN pack: `ssmCheckpointBytes()` is non-zero here.
    var q35 = qwen4ExpOomConfig();
    q35.model_type = "not_a_served_arch";
    try t.expect(q35.ssmCheckpointBytes() > 0);
    try t.expect(retainedSsmCheckpointBytes(&q35, seq, 0, chunk) > 0);

    // Supplies the donating case so the qwen4_exp assertion below is falsifiable; the ungated loop reads 0 through `.{}`.
    const warm = WarmPrefix{ .matched_tokens = 100_000, .capacity_tokens = 400_000, .will_donate = true };
    for ([_][]const u8{ "glm5_next", "not_a_served_arch" }) |mt| {
        var cfg = qwen4ExpOomConfig();
        cfg.model_type = mt;
        const terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, warm);
        try t.expectEqual(@as(u64, 0), terms.reserved_kv_bytes);
        try t.expectEqual(@as(u64, 0), terms.checkpoint_bytes);
        try t.expectEqual(@as(u64, 0), terms.state_bytes);
        try t.expectEqual(@as(u64, 0), terms.shared_resident_bytes);
        // `.{}` is the identity: the whole bill is the previous expression.
        try t.expectEqual(
            prefillMemoryNeeded(seq, 24, 2, cfg.kvBytesPerToken(), 256, 256, 2560, 9216, kv_bits, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), .{}),
            prefillMemoryNeeded(seq, 24, 2, cfg.kvBytesPerToken(), 256, 256, 2560, 9216, kv_bits, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), terms),
        );
    }

    const q4 = qwen4ExpOomConfig();
    const gated = prefillRequestTerms(&q4, seq, max_tokens, kv_bits, chunk, warm);
    try t.expect(gated.reserved_kv_bytes > 0);
    try t.expect(gated.checkpoint_bytes > 0);
    try t.expect(gated.state_bytes > 0);
    try t.expect(gated.shared_resident_bytes > 0);
}

test "prefillRequestTerms: a loaded DFlash2 assistant bills its window and chunk rows on qwen4 only" {
    const t = std.testing;
    var cfg = qwen4ExpOomConfig();
    const seq: u64 = 200_000;
    const chunk: u64 = 2048;
    const before = prefillRequestTerms(&cfg, seq, 8192, 8, chunk, .{});
    try t.expectEqual(@as(u64, 0), before.dflash_bytes);
    cfg.dflash_loaded = true;
    cfg.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    cfg.dflash_capture_bytes_per_token = (5 * 2560 + 2 * 2560) * 2 + 5 * 8 * 128 * 4;
    const on = prefillRequestTerms(&cfg, seq, 8192, 8, chunk, .{});
    try t.expectEqual(cfg.dflash_window_bytes * 4 + cfg.dflash_capture_bytes_per_token * chunk + 64 * 1024 * 1024, on.dflash_bytes);
    const short = prefillRequestTerms(&cfg, 100, 8192, 8, chunk, .{});
    try t.expectEqual(cfg.dflash_window_bytes * 4 + cfg.dflash_capture_bytes_per_token * 100 + 64 * 1024 * 1024, short.dflash_bytes);
    var other = cfg;
    other.model_type = "not_a_served_arch";
    try t.expectEqual(@as(u64, 0), prefillRequestTerms(&other, seq, 8192, 8, chunk, .{}).dflash_bytes);
}

test "prefillRequestTerms: qwen4 MTP head KV is billed when MTP is on and zero when off" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const saved_force = server_config.default_force_mtp;
    defer server_config.default_force_mtp = saved_force;
    server_config.default_force_mtp = false;
    const saved_ov = transformer_mod.Transformer.mtp_head_kv_quant_override;
    defer transformer_mod.Transformer.mtp_head_kv_quant_override = saved_ov;
    transformer_mod.Transformer.mtp_head_kv_quant_override = false;
    const saved_boot = configured_kv_quant;
    defer configured_kv_quant = saved_boot;
    configured_kv_quant = transformer_mod.KVQuantConfig.affine(8);
    var cfg = qwen4ExpOomConfig();
    cfg.mtp_override = null;
    try t.expect(!forceMtpFor(&cfg));
    const seq: u64 = 200_000;
    const chunk: u64 = 4096;
    const max_tokens: u64 = 8192;
    const kv_bits: u64 = 8;
    try t.expectEqual(@as(u64, 2048), mtpHeadKvBytesPerToken(&cfg));
    transformer_mod.Transformer.mtp_head_kv_quant_override = true;
    try t.expectEqual(@as(u64, 1088), mtpHeadKvBytesPerToken(&cfg));
    transformer_mod.Transformer.mtp_head_kv_quant_override = false;
    const off = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, .{});
    try t.expectEqual(@as(u64, 0), off.mtp_head_kv_bytes);
    const on = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, .{ .mtp_on = true });
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg)), seq);
    try t.expectEqual(reserved * 2048, on.mtp_head_kv_bytes);
    const n: u64 = cfg.attnCacheLayerCount();
    try t.expectEqual(cfg.qsaRingBytes() + cfg.qsaRingBytes() / n, on.qsa_ring_bytes);
    try t.expectEqual(reserved * (statePerTokenBilled(&cfg) + statePerTokenBilled(&cfg) / n), on.state_bytes);
    try t.expectEqual(cfg.qsaRingBytes(), off.qsa_ring_bytes);
    try t.expectEqual(reserved * statePerTokenBilled(&cfg), off.state_bytes);
    var other = cfg;
    other.model_type = "not_a_served_arch";
    const other_on = prefillRequestTerms(&other, seq, max_tokens, kv_bits, chunk, .{ .mtp_on = true });
    try t.expectEqual(@as(u64, 0), other_on.mtp_head_kv_bytes);
}

test "mtpHeadKvLine: --mtp-head-kv-quant is reported per arch" {
    const t = std.testing;
    const qwen4 = model_mod.ModelConfig{ .model_type = "qwen4_exp" };
    const mimo = model_mod.ModelConfig{ .model_type = "mimo_v2" };
    try t.expect(mtpHeadKvLine(&qwen4, false) == null);
    try t.expect(mtpHeadKvLine(&mimo, false) == null);
    try t.expectEqualStrings("MTP head KV: following --kv-quant (--mtp-head-kv-quant)", mtpHeadKvLine(&qwen4, true).?);
    // MiMo's heads keep their own dense window (`mimo_mtp.RowCache`); the flag never reaches them.
    try t.expectEqualStrings("MTP head KV: --mtp-head-kv-quant has no effect on mimo_v2 (its heads keep a dense sliding window)", mtpHeadKvLine(&mimo, true).?);
}

test "qwenGdnLine: --qwen-gdn 16 is reported per arch, and has no effect where there is no GDN arm" {
    const t = std.testing;
    const qwen4 = model_mod.ModelConfig{ .model_type = "qwen4_exp" };
    const glm = model_mod.ModelConfig{ .model_type = "glm5_next" };
    const mimo = model_mod.ModelConfig{ .model_type = "mimo_v2" };
    try t.expect(qwenGdnLine(&qwen4, true) == null);
    try t.expect(qwenGdnLine(&glm, true) == null);
    try t.expectEqualStrings("GDN state: bf16 between tokens (--qwen-gdn 16; f32 is the default; the decay gate is f32 either way)", qwenGdnLine(&qwen4, false).?);
    // GLM's recurrent state is KDA, not GDN: the flag names a layer it does not have.
    try t.expectEqualStrings("GDN state: --qwen-gdn 16 has no effect on GLM (its KDA state is f32 already)", qwenGdnLine(&glm, false).?);
    try t.expectEqualStrings("GDN state: --qwen-gdn 16 has no effect on mimo_v2 (no GatedDeltaNet layers)", qwenGdnLine(&mimo, false).?);
}

test "deepseek_v41 refuses the launch flags it cannot act on, and ignores a model-settings kv_quant with one line" {
    const t = std.testing;
    var dsv41 = model_mod.ModelConfig{ .model_type = "deepseek_v41" };
    const qwen4 = model_mod.ModelConfig{ .model_type = "qwen4_exp" };
    try t.expect(archFlagRefusal(&dsv41, .{}) == null);
    const all = LaunchFlagUse{ .kv_quant = true, .qwen_gdn_16 = true, .mtp_head_kv_quant = true, .mtp_typical = true, .mtp_tokenv3 = true, .mtp_greedy_tail = true };
    try t.expect(archFlagRefusal(&qwen4, all) == null);
    for ([_]struct { use: LaunchFlagUse, flag: []const u8 }{
        .{ .use = .{ .kv_quant = true }, .flag = "--kv-quant" },
        .{ .use = .{ .qwen_gdn_16 = true }, .flag = "--qwen-gdn" },
        .{ .use = .{ .mtp_head_kv_quant = true }, .flag = "--mtp-head-kv-quant" },
        .{ .use = .{ .mtp_typical = true }, .flag = "--mtp-typical" },
        .{ .use = .{ .mtp_tokenv3 = true }, .flag = "--mtp-tokenv3" },
        .{ .use = .{ .mtp_greedy_tail = true }, .flag = "--mtp-greedy-tail" },
    }) |c| {
        const why = archFlagRefusal(&dsv41, c.use) orelse return error.TestExpectedRefusal;
        try t.expect(std.mem.startsWith(u8, why, c.flag));
        try t.expect(std.mem.indexOf(u8, why, "deepseek_v41") != null);
    }
    try t.expectEqualStrings("off", kvCacheForLaunch(&dsv41, transformer_mod.KVQuantConfig.affine(4), true).label());
    try t.expectEqualStrings("kv4", kvCacheForLaunch(&qwen4, transformer_mod.KVQuantConfig.affine(4), true).label());
    try t.expect(settingsKvQuantIgnoredLine(&dsv41) == null);
    dsv41.kv_quant_override = transformer_mod.KVQuantConfig.affine(8);
    try t.expect(std.mem.indexOf(u8, settingsKvQuantIgnoredLine(&dsv41).?, "model-settings.json kv_quant is ignored") != null);
    try t.expect(settingsKvQuantIgnoredLine(&qwen4) == null);
    try t.expectEqualStrings("off", kvCacheForLaunch(&dsv41, transformer_mod.KVQuantConfig.engine_default, false).label());
}

test "deepseek_v41 ignores a model-settings mtp_acceptance or mtp_greedy_tail with one line naming the keys set" {
    const t = std.testing;
    var dsv41 = model_mod.ModelConfig{ .model_type = "deepseek_v41" };
    var qwen4 = model_mod.ModelConfig{ .model_type = "qwen4_exp" };
    try t.expect(settingsMtpIgnoredLine(&dsv41) == null);
    dsv41.mtp_acceptance_override = .{ .typical = .{ .delta = 0.3 } };
    try t.expectEqualStrings("model-settings.json mtp_acceptance is ignored on deepseek_v41 (DSpark accepts drafts by the exact rule only)", settingsMtpIgnoredLine(&dsv41).?);
    dsv41.mtp_acceptance_override = null;
    dsv41.mtp_greedy_tail_override = true;
    try t.expectEqualStrings("model-settings.json mtp_greedy_tail is ignored on deepseek_v41 (DSpark has no MTP tail)", settingsMtpIgnoredLine(&dsv41).?);
    dsv41.mtp_acceptance_override = .{ .tokenv3 = 0.95 };
    try t.expectEqualStrings("model-settings.json mtp_acceptance and mtp_greedy_tail are ignored on deepseek_v41 (DSpark accepts drafts by the exact rule only, with no MTP tail)", settingsMtpIgnoredLine(&dsv41).?);
    qwen4.mtp_acceptance_override = .{ .tokenv3 = 0.95 };
    qwen4.mtp_greedy_tail_override = true;
    try t.expect(settingsMtpIgnoredLine(&qwen4) == null);
}

test "--drop-expert is refused on every arch but deepseek_v41, before the weights load" {
    const t = std.testing;
    const use = LaunchFlagUse{ .drop_expert = true };
    const dsv41 = model_mod.ModelConfig{ .model_type = "deepseek_v41" };
    try t.expect(archFlagRefusal(&dsv41, use) == null);
    for ([_][]const u8{ "qwen4_exp", "mimo_v2", "glm5_next" }) |arch| {
        const cfg = model_mod.ModelConfig{ .model_type = arch };
        const why = archFlagRefusal(&cfg, use) orelse return error.TestExpectedRefusal;
        try t.expect(std.mem.startsWith(u8, why, "--drop-expert"));
        try t.expect(archFlagRefusal(&cfg, .{}) == null);
    }
}

test "the 64 GB Sushi-STQ-2.75bpw recipe's bills at both GDN widths: 250000 context, kv8, MTP, 32000 new tokens" {
    const t = std.testing;
    const mb: u64 = 1024 * 1024;
    var cfg = try model_mod.parseConfigFromJson(t.allocator, @embedFile("fixtures/model-configs/qwen4_exp.json"));
    defer cfg.deinit(t.allocator);
    cfg.expert_layout = .exl3_k4;
    const saved_head = transformer_mod.Transformer.mtp_head_kv_quant_flag;
    defer transformer_mod.Transformer.mtp_head_kv_quant_flag = saved_head;
    transformer_mod.Transformer.mtp_head_kv_quant_flag = true;
    const saved_ctx = server_config.max_context_size;
    defer server_config.max_context_size = saved_ctx;
    server_config.max_context_size = 250_000;
    const saved_kv = configured_kv_quant;
    defer configured_kv_quant = saved_kv;
    configured_kv_quant = transformer_mod.KVQuantConfig.affine(8);
    const saved_mtp = configured_mtp;
    defer configured_mtp = saved_mtp;
    configured_mtp = true;
    const warm = WarmPrefix{ .mtp_on = true };
    // The load bill's context term is the narrowest rung; at this pack's weights the 59,000 MB ceiling admits
    // the 2048 rung. With the SSD tier up (the default) the prefill holds 3 checkpoints, not the 16 it captures.
    try t.expectEqual(@as(u64, 7_844), prefillNeededAtChunk(&cfg, 250_000, 32_000, 8, 512, warm) / mb);
    try t.expectEqual(@as(u64, 9_337), prefillNeededAtChunk(&cfg, 250_000, 32_000, 8, 2048, warm) / mb);
    try t.expectEqual(@as(u64, 11_157), prefillNeededAtChunk(&cfg, 250_000, 32_000, 8, 4096, warm) / mb);
    cfg.gdn_state_f32 = false;
    try t.expectEqual(@as(u64, 7_641), prefillNeededAtChunk(&cfg, 250_000, 32_000, 8, 512, warm) / mb);
    try t.expectEqual(@as(u64, 9_135), prefillNeededAtChunk(&cfg, 250_000, 32_000, 8, 2048, warm) / mb);
    try t.expectEqual(@as(u64, 10_954), prefillNeededAtChunk(&cfg, 250_000, 32_000, 8, 4096, warm) / mb);
}

test "an SSD tier bounds the checkpoints a Qwen prefill holds, at both GDN widths" {
    const t = std.testing;
    var cfg = try model_mod.parseConfigFromJson(t.allocator, @embedFile("fixtures/model-configs/qwen4_exp.json"));
    defer cfg.deinit(t.allocator);
    const saved = .{ ssm_checkpoint_stride, ssm_checkpoint_max };
    defer {
        ssm_checkpoint_stride = saved[0];
        ssm_checkpoint_max = saved[1];
    }
    ssm_checkpoint_stride = 256;
    ssm_checkpoint_max = 16;
    const seq: u64 = 250_000;
    for ([_]bool{ true, false }) |f32_state| {
        cfg.gdn_state_f32 = f32_state;
        const per_cp = cfg.ssmCheckpointBytes();
        try t.expectEqual(if (f32_state) @as(u64, 115_458_048) else 58_834_944, per_cp);

        const no_tier = TierGuard.pin(false, 0);
        try t.expectEqual(@as(u64, 16) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 0, 512));
        no_tier.restore();

        // A write-through prefill holds two states; one it cannot serve is thinned to two and holds a third
        // while one is captured: the bill covers the larger.
        const ssd_only = TierGuard.pin(false, 20 << 30);
        try t.expectEqual(@as(u64, 3) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 0, 512));
        // A prompt under one stride captures its prompt-end snapshot alone and bills just that.
        try t.expectEqual(per_cp, retainedSsmCheckpointBytes(&cfg, 300, 0, 512));
        // A tier that did not come up bills as without one.
        cfg.prefix_cache_disk_declined = true;
        try t.expectEqual(@as(u64, 16) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 0, 512));
        cfg.prefix_cache_disk_declined = false;
        ssd_only.restore();

        // A RAM tier beside the SSD one keeps its entry's newest two out of the same states.
        const both = TierGuard.pin(true, 20 << 30);
        try t.expectEqual(@as(u64, 3) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 0, 512));
        both.restore();
    }
}

test "prefillRequestTerms: MTP head KV bills the boot scheme, not a request kv_quant override" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const saved_ov = transformer_mod.Transformer.mtp_head_kv_quant_override;
    defer transformer_mod.Transformer.mtp_head_kv_quant_override = saved_ov;
    transformer_mod.Transformer.mtp_head_kv_quant_override = true;
    const saved_boot = configured_kv_quant;
    defer configured_kv_quant = saved_boot;
    configured_kv_quant = transformer_mod.KVQuantConfig.affine(8);
    var cfg = qwen4ExpOomConfig();
    cfg.mtp_override = null;
    const seq: u64 = 200_000;
    const chunk: u64 = 4096;
    const max_tokens: u64 = 8192;
    const request_bits: u64 = 4;
    const n: u64 = cfg.attnCacheLayerCount();
    const request_head = kvBytesPerTokenAtBits(cfg.kvBytesPerToken() / n, request_bits);
    try t.expectEqual(@as(u64, 576), request_head);
    try t.expectEqual(@as(u64, 1088), mtpHeadKvBytesPerToken(&cfg));
    const on = prefillRequestTerms(&cfg, seq, max_tokens, request_bits, chunk, .{ .mtp_on = true });
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg)), seq);
    try t.expectEqual(reserved * 1088, on.mtp_head_kv_bytes);
}

test "ctxSizingCacheReserve: the advertised context is unchanged on every other arch" {
    // The constant changes the advertised `context_length` at every ask except the 2 GiB default,
    // and an agent CLI budgets against that number for a whole session.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const saved = prefix_cache_mem_bytes;
    defer prefix_cache_mem_bytes = saved;
    const tiers = TierGuard.pin(true, 0);
    defer tiers.restore();

    var q4 = longCtxTestConfig(); // declares qwen4_exp
    var other = longCtxTestConfig();
    other.model_type = "not_a_served_arch";

    for ([_]u64{ 0, 2 * GiB, 10 * GiB, 60 * GiB }) |ask| {
        prefix_cache_mem_bytes = ask;
        try t.expectEqual(CTX_SIZING_CACHE_RESERVE, ctxSizingCacheReserve(&q4));
        try t.expectEqual(ask, ctxSizingCacheReserve(&other));
    }

    // At the default ask the two agree: a default boot is unchanged on every arch.
    prefix_cache_mem_bytes = CTX_SIZING_CACHE_RESERVE;
    try t.expectEqual(ctxSizingCacheReserve(&q4), ctxSizingCacheReserve(&other));

    // The reserve really moves the advertised context on the ungated arch.
    const kv_bits: u64 = 8;
    const per_tok = kvBytesPerTokenAtBits(other.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&other);
    const ceiling: u64 = 120 * GiB;
    const active: u64 = 69 * GiB;
    const transient = prefillTransientReserve(&other, kv_bits, 4096);
    const small = safeContextForBudget(ceiling, active, 2 * GiB +| transient, per_tok, 0);
    const big = safeContextForBudget(ceiling, active, 40 * GiB +| transient, per_tok, 0);
    try t.expect(big < small);
}

test "prefillAdmissionBill: the evict-to-admit arms are driven by the two credits alone" {
    // Both admit arms are driven entirely by the two credits, so zeroing them restores the
    // previous single `needed > available` refusal.
    const t = std.testing;
    const MB: u64 = 1024 * 1024;

    const gated = AdmissionBill{ .needed = 30 * MB, .available = 20 * MB, .evictable = 15 * MB, .reclaimable = 15 * MB };
    try t.expect(!gated.fits());
    try t.expect(gated.fitsAfterEviction()); // evict-and-admit
    try t.expectEqual(AdmissionVerdict.evict, admissionVerdict(gated));

    const deferral = AdmissionBill{ .needed = 30 * MB, .available = 20 * MB, .evictable = 15 * MB, .reclaimable = 0 };
    try t.expect(!deferral.fitsAfterEviction());
    try t.expect(pinnedResidentBytes(deferral) > 0); // warm deferral

    // Where the arch does not evict, the bill carries neither credit, and both arms go dead.
    const ungated = AdmissionBill{ .needed = 30 * MB, .available = 20 * MB };
    try t.expectEqual(ungated.fits(), ungated.fitsAfterEviction());
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(ungated));
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(ungated));
    // A bill that fits still admits on both arms.
    const roomy = AdmissionBill{ .needed = 10 * MB, .available = 20 * MB };
    try t.expectEqual(AdmissionVerdict.admit, admissionVerdict(roomy));
}

test "memoryRefusalMessage: the 400 names the hot cache only where the bill carries it" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // A non-qwen4 refusal read "the hot prefix cache holds ~0MB ..." on a box with a multi-GB resident cache.
    const t = std.testing;
    const MB: u64 = 1024 * 1024;

    // Ungated: the previous sentence, byte for byte.
    const bare = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{ .needed = 30 * MB, .available = 20 * MB }, false);
    defer t.allocator.free(bare);
    try t.expectEqualStrings(
        "Prompt (12000 tokens) requires ~4096MB GPU memory but only ~2048MB available. Reduce prompt size or use a smaller model.",
        bare,
    );

    // Gated: the message quotes what the guard compared.
    const rich = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{
        .needed = 30 * MB,
        .available = 20 * MB,
        .evictable = 1500 * MB,
        .reclaimable = 900 * MB,
    }, true);
    defer t.allocator.free(rich);
    try t.expect(std.mem.indexOf(u8, rich, "holds ~1500MB more") != null);
    try t.expect(std.mem.indexOf(u8, rich, "(~900MB") != null);

    // The gated arch reaches this arm with an empty cache too and keeps its bytes, zeroes and all.
    const q4_cold = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{ .needed = 30 * MB, .available = 20 * MB }, true);
    defer t.allocator.free(q4_cold);
    try t.expectEqualStrings(
        "Prompt (12000 tokens) requires ~4096MB GPU memory but only ~2048MB is available and the hot prefix cache holds ~0MB more, all of which can be reclaimed (~0MB — the shortfall is elsewhere), which is not enough. Reduce prompt size, lower --ctx-size, or use a smaller model.",
        q4_cold,
    );
    const ungated_rich = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{
        .needed = 30 * MB,
        .available = 20 * MB,
        .evictable = 1500 * MB,
        .reclaimable = 900 * MB,
    }, false);
    defer t.allocator.free(ungated_rich);
    try t.expectEqualStrings(bare, ungated_rich);
}

/// The live 364k agent session: `qwen4ExpOomConfig` plus the deployed pack's indexer budget 2048,
/// four indexer heads, affine-4 weights and `--ctx-size 786432`; reproduces the incident's bills to the MB.
fn qwen4ExpLive364kConfig() model_mod.ModelConfig {
    var cfg = qwen4ExpOomConfig();
    cfg.pinned_context = 786_432;
    cfg.indexer_budget = 2048;
    cfg.indexer_n_heads = 4;
    cfg.quant_bits = 4;
    cfg.quant_mode = .affine;
    cfg.max_position_embeddings = 1_048_576;
    return cfg;
}

test "a WARM append bills the checkpoints its prefill CAPTURES, not the prompt's" {
    // These pins price a cache with no SSD tier behind it: every checkpoint stays in RAM.
    const no_tier = TierGuard.pin(false, 0);
    defer no_tier.restore();
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    // Turn B: prompt 368,208, restored 364,478 by move, refused by 290 MB at width 512. 1,795 MB
    // of the bill was 32 SSM checkpoints; a warm prefill forwards a 3,730-token tail and captures
    // at most one, and the entry's own checkpoints are already resident and merged at commit.
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const cfg = qwen4ExpLive364kConfig();
    const mb: u64 = 1024 * 1024;
    const per_cp = cfg.ssmCheckpointBytes();
    const seq: u64 = 368_208;
    const matched: u64 = 364_478;
    const chunk: u64 = 512;
    // The stride follows the uncapped base chunk (`prefill_chunk_override`), not the adaptive width.
    const saved_override = generate_mod.prefill_chunk_override;
    defer generate_mod.prefill_chunk_override = saved_override;
    generate_mod.prefill_chunk_override = 8192;
    const stride: u64 = 8192;
    try t.expectEqual(stride, prefill_plan.effectiveSsmCheckpointStride(ssm_checkpoint_stride, 8192));

    // Cold: `seq / stride` saturates at --ssm-checkpoint-max as before.
    try t.expectEqual(@as(u64, ssm_checkpoint_max) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 0, chunk));
    // Warm: a 3,730-token append at stride 8192 crosses no boundary, so the end snapshot is the whole of it.
    try t.expectEqual(@as(u64, 1) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, matched, chunk));
    // A chain extension long enough to fill the ring still bills the cap: the
    // credit is proportional to the SPAN, not a blanket warm discount.
    try t.expectEqual(@as(u64, ssm_checkpoint_max) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 60_000, chunk));
    // A span shorter than the backoff window snapshots nothing.
    try t.expectEqual(@as(u64, 0), retainedSsmCheckpointBytes(&cfg, seq, seq - 4, chunk));

    // The bill term by term, from the same inputs `prefillMemoryNeeded` reads.
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 418_224;
    // The restored buffer is smaller than the reservation, so it grows and the rows it holds
    // are still credited. This block keeps the original turn-B decomposition by pricing the
    // SHARED restore of the same span, whose credit is zero for the other reason.
    const turn_b_donating = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = 365_568, .will_donate = true };
    const turn_b = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = 365_568, .will_donate = false };
    const reserve_rows = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    try t.expectEqual(seq + transformer_mod.KVCache.RESERVE_GEN_HEADROOM + chunk, reserve_rows);
    try t.expect(reserve_rows > turn_b.capacity_tokens);
    try t.expectEqual(@as(u64, 0), turn_b.creditedRows(reserve_rows));
    try t.expectEqual(@as(u64, 364_478), turn_b_donating.creditedRows(reserve_rows));

    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, turn_b);
    // On a GDN MoE the envelope is the stream arm, not 3 x MLP.
    const mlp: u64 = 8 * chunk * @max(cfg.hidden_size, prefillFfnWidth(&cfg)) * 2;
    const envelope: u64 = @max(3 * mlp, mlp + chunk * prefillStreamBytesPerToken(&cfg));
    try t.expect(envelope > 3 * mlp);
    const gross =
        seq * kv_per_tok +
        terms.reserved_kv_bytes +
        terms.state_bytes +
        terms.checkpoint_bytes +
        terms.qsa_ring_bytes +
        2 * seq * cfg.num_key_value_heads * cfg.head_dim * 2 +
        envelope +
        PREFILL_RUNTIME_FLOOR_BYTES;
    // No score scratch (hd 256 is fused) and no dequant-GEMM weights (fwd 512
    // is below PREFILL_DQ_GEMM_MIN_M) at this width.
    try t.expect(transformer_mod.prefillHeadDimFused(@intCast(cfg.prefillScoreHeadDim())));
    try t.expect(chunk < transformer_mod.PREFILL_DQ_GEMM_MIN_M);
    const needed = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, turn_b);
    try t.expectEqual(gross * 5 / 4 + qsaMaskBytes(&cfg, chunk, seq), needed);

    // AND THE VERDICT MOVES. 11,854 MB refused against 10,000 MB available;
    // the honest checkpoint term brings it to 9,789 MB, so the turn is served.
    const available: u64 = 10_000 * mb;
    try t.expect(needed < available);
    try t.expectEqual(@as(u64, 9_789), needed / mb);
    const before = gross - terms.checkpoint_bytes + @as(u64, ssm_checkpoint_max) * per_cp;
    try t.expectEqual(@as(u64, 11_854), (before * 5 / 4 + qsaMaskBytes(&cfg, chunk, seq)) / mb);
    try t.expect((before * 5 / 4 + qsaMaskBytes(&cfg, chunk, seq)) > available);
}

test "the live 364k session's admission numbers reproduce, to the megabyte" {
    // These pins price a cache with no SSD tier behind it: every checkpoint stays in RAM.
    const no_tier = TierGuard.pin(false, 0);
    defer no_tier.restore();
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    // The anchor: fed the deployed config and the incident's prompt lengths, the estimator prints
    // the incident's own numbers (Turn A admitted at 16,233 MB; Turn B refused at 13,664 MB).
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const cfg = qwen4ExpLive364kConfig();
    const mb: u64 = 1024 * 1024;
    const kv_bits: u64 = 8;
    // Turn A, width 2048, prompt 364,509, max_gen 421,923.
    try t.expectEqual(
        @as(u64, 13_269),
        prefillNeededAtChunk(&cfg, 364_509, 421_923, kv_bits, 2048, .{}) / mb,
    );
    try t.expectEqual(
        @as(u64, 11_854),
        prefillNeededAtChunk(&cfg, 368_208, 418_224, kv_bits, 512, .{}) / mb,
    );
}

test "the MLX buffer pool is inside `available` by construction, so a refusal cannot buy it back" {
    // `currentGpuMemoryCeiling` adds the pool to the footprint before the physical minimum, so on
    // the physical arm `available` is `cache_bytes + free_system` and a clear moves it by zero.
    const t = std.testing;
    const wired: u64 = 120 << 30;
    const active: u64 = 71_310 << 20;
    const pool: u64 = 8_268 << 20;
    const free_system: u64 = 12_546 << 20;
    const ceiling = physicalMemoryCeiling(wired, active + pool, free_system);
    try t.expect(ceiling < wired); // the physical arm, as on the box
    try t.expectEqual(pool + free_system, ceiling - active);
    const cleared = physicalMemoryCeiling(wired, active, free_system + pool);
    try t.expectEqual(ceiling, cleared);
}

fn mimoV2BillConfig() model_mod.ModelConfig {
    var c = model_mod.ModelConfig{};
    c.model_type = "mimo_v2";
    c.num_hidden_layers = 4;
    c.hidden_size = 384;
    c.num_attention_heads = 4;
    c.has_sliding_window = true;
    c.has_explicit_layer_types = true;
    c.sliding_window = 128;
    c.layer_is_global[0] = true;
    c.layer_is_global[3] = true;
    c.head_dim = 192;
    c.v_head_dim = 128;
    c.num_key_value_heads = 3;
    c.global_head_dim = 192;
    c.global_v_head_dim = 128;
    c.num_global_key_value_heads = 2;
    // Pinned as a load pins it: unpinned, the reservation's headroom follows an auto-context
    // sized from live GPU memory and from the attention switches a test flips between bills.
    c.pinned_context = 1 << 20;
    return c;
}

test "a MiMo slot bills the decode rebuilds the other rows of its batched decode hold" {
    const t = std.testing;
    var cfg = mimoV2BillConfig();
    // One global layer's dense K/V below the packed arms' floor, for every other row of a full group.
    const rebuild = transformer_mod.mimoGlobalDecodeRebuildMaxKeys() * cfg.layerKvBytes(0);
    try t.expect(rebuild > 0);
    try t.expectEqual(@as(u64, @intCast(scheduler_mod.batchGroupCap(&cfg) - 1)) * rebuild, batchedDecodeRowsBytes(&cfg));
    try t.expectEqual(@as(usize, 4), scheduler_mod.batchGroupCap(&cfg));
    // A streamed load decodes serial and bills nothing for it.
    cfg.expert_streaming = true;
    try t.expectEqual(@as(u64, 0), batchedDecodeRowsBytes(&cfg));
    var plain = mimoV2BillConfig();
    plain.model_type = "not_a_served_arch";
    try t.expectEqual(@as(u64, 0), batchedDecodeRowsBytes(&plain));
}

test "the sliding ring is billed once per slot and staged per chunk token" {
    const t = std.testing;
    const cfg = mimoV2BillConfig();
    // Once per slot, at the width the cache stores — the twin of the QSA ring,
    // which this arch does not have.
    try t.expectEqual(@as(u64, 0), cfg.qsaRingBytes());
    // The slot also holds its ring checkpoints until the commit takes them: where it restored
    // to, the message starts its prefill crossed, and its prompt end.
    const ring_and_cp = cfg.swaRingBytes() + prefix_cache_mod.SLOT_RING_CHECKPOINTS * cfg.swaRingCheckpointBytes();
    const rows = batchedDecodeRowsBytes(&cfg);
    try t.expectEqual(ring_and_cp + rows, slotRingBytes(&cfg, 16));
    try t.expectEqual(kvBytesPerTokenAtBits(ring_and_cp, 8) + rows, slotRingBytes(&cfg, 8));
    try t.expect(slotRingBytes(&cfg, 8) < slotRingBytes(&cfg, 16));
    // Per token of context only the two global layers count.
    try t.expectEqual(@as(u64, 2 * 2 * (192 + 128) * 2), sessionBytesPerToken(&cfg, 16));
    // The chunk the ring has not compacted yet rides the prefill envelope.
    try t.expect(prefillStreamBytesPerToken(&cfg) >= cfg.swaStreamBytesPerToken(5));
    // At a 48-layer geometry the cadence window is the bound: five sliding
    // layers stage a chunk at once, not forty.
    var full = cfg;
    full.num_hidden_layers = 48;
    var li: u32 = 0;
    while (li < 48) : (li += 1) full.layer_is_global[li] = (li % 6) == 0;
    try t.expectEqual(
        @as(u64, 5 * 3 * (192 + 128) * 2),
        full.swaStreamBytesPerToken(@as(u64, transformer_mod.Transformer.PREFILL_EVAL_CADENCE_DEFAULT) + 1),
    );
    // The ring itself still holds every sliding layer.
    try t.expectEqual(full.swaRingTokens() * 40 * 3 * (192 + 128) * 2, full.swaRingBytes());
    // An arch that stores every layer full-length keeps a zero ring.
    var plain = cfg;
    plain.model_type = "not_a_served_arch";
    try t.expectEqual(@as(u64, 0), slotRingBytes(&plain, 16));
    try t.expectEqual(@as(u64, 0), prefillStreamBytesPerToken(&plain));
}

test "a QSA arch's kv-quant dequant scratch follows the widest forward that still gathers from a rebuild" {
    const t = std.testing;
    const cfg = qwen4RequestTestConfig();
    const seq: u64 = 1024 * 1024;
    const layer_bytes: u64 = 2 * (256 + 256) * 2;
    const span: u64 = 2048 + 4 - 1 + transformer_mod.FUSED256_MIN_Q_LEN - 1;
    const split_k: u64 = @intCast(transformer_mod.QSA_ATTN_PACKED_MAX_S);
    // Split-K widths read the packed cache at every kv: only the dense span rebuilds.
    try t.expectEqual(span * layer_bytes, kvDequantScratchBytes(&cfg, seq, split_k));
    try t.expectEqual(span * layer_bytes, kvDequantScratchBytes(&cfg, 2 * seq, 16));
    // A wider forward rebuilds only while its rows would stage each packed row over 4 times.
    try t.expectEqual(512 * 2048 / 4 * layer_bytes, kvDequantScratchBytes(&cfg, seq, 512));
    try t.expectEqual(seq * layer_bytes, kvDequantScratchBytes(&cfg, seq, 8192));
    // A short prompt never rebuilds more than it stores.
    try t.expectEqual(@as(u64, 1000) * layer_bytes, kvDequantScratchBytes(&cfg, 1000, 16));
    // Either kill switch brings the whole-cache rebuild (the generic term) back.
    qwen4_qsa_attn.qsa_attn_kernel_override = false;
    try t.expectEqual(@as(u64, 0), kvDequantScratchBytes(&cfg, seq, 16));
    qwen4_qsa_attn.qsa_attn_kernel_override = null;
    qwen4_qsa_gather.qsa_gather_override = false;
    defer qwen4_qsa_gather.qsa_gather_override = null;
    try t.expectEqual(@as(u64, 0), kvDequantScratchBytes(&cfg, seq, 16));
}

test "a ringed arch's kv-quant dequant scratch is one layer's rebuild, not the model's" {
    const t = std.testing;
    const cfg = mimoV2BillConfig();
    const seq: u64 = 512 * 1024;

    // Composed: the widest rebuild is a GLOBAL layer at the prompt length,
    // 2 kv heads of qk 192 + v 128. The sliding layers rebuild their ring.
    transformer_mod.fused256_override = false;
    defer transformer_mod.fused256_override = null;
    try t.expectEqual(seq * 2 * (192 + 128) * 2, kvDequantScratchBytes(&cfg, seq, 1024));
    // The generic expression this replaces reads the SLIDING head count and
    // doubles the qk width: 1.8x here, 2.4x at the shipped 4/8-head geometry.
    const generic: u64 = 2 * seq * cfg.num_key_value_heads * cfg.head_dim * 2;
    try t.expect(generic > kvDequantScratchBytes(&cfg, seq, 1024));

    // Fused: the global layers read the packed cache per dispatch, so their
    // rebuild stops scaling with the prompt at all.
    transformer_mod.fused256_override = true;
    const fused = kvDequantScratchBytes(&cfg, seq, 1024);
    try t.expectEqual(
        MOE_PREFILL_COEXIST * @as(u64, @intCast(transformer_mod.PACKED_KV_SLICE_MAX)) * 2 * (192 + 128) * 2,
        fused,
    );
    try t.expectEqual(fused, kvDequantScratchBytes(&cfg, 2 * seq, 1024));
    transformer_mod.fused256_override = false;

    // A decode forward bills whatever the global-layer decode rebuilds whole: a floor's worth of keys.
    transformer_mod.fused256_override = true;
    try t.expectEqual(fused, kvDequantScratchBytes(&cfg, seq, 1));
    transformer_mod.fused256_override = false;

    // A short prompt never rebuilds more than it stores, and below the ring the
    // widest layer is a SLIDING one: it carries more KV heads than a global.
    try t.expectEqual(@as(u64, 256) * 3 * (192 + 128) * 2, kvDequantScratchBytes(&cfg, 256, 1024));

    // Every arch whose caching layers share one geometry keeps the generic term.
    var plain = cfg;
    plain.model_type = "not_a_served_arch";
    try t.expectEqual(@as(u64, 0), kvDequantScratchBytes(&plain, seq, 1024));

    // fp16 KV never rebuilds anything, whatever the arch.
    const dense_kv = prefillMemoryNeeded(seq, 4, 3, cfg.kvBytesPerToken(), 192, 192, 384, 1024, 16, 1024, seq, 0, 0, .{ .dequant_scratch_bytes = 1 << 30 });
    try t.expectEqual(
        dense_kv,
        prefillMemoryNeeded(seq, 4, 3, cfg.kvBytesPerToken(), 192, 192, 384, 1024, 16, 1024, seq, 0, 0, .{}),
    );
}

test "a ringed arch reserves its KV capacity once and bills the ring it holds" {
    const t = std.testing;
    const tiers = TierGuard.pin(true, 0);
    defer tiers.restore();
    const cfg = mimoV2BillConfig();
    const seq: u64 = 512 * 1024;
    const chunk: u64 = 1024;

    // The allocator side: the prefill loop asks for the whole request's capacity
    // before the first chunk writes, so no buffer grows mid-prefill.
    try t.expect(prefill_plan.reservedPrefillTokens(&cfg, seq, 2048, chunk) > seq);
    try t.expectEqual(@as(u64, 0), prefill_plan.reservedPrefillTokens(&cfg, 1024, 2048, chunk));

    // The bill side: headroom and the per-slot ring, nothing else acquired.
    const terms = prefillRequestTerms(&cfg, seq, 2048, 8, chunk, .{});
    try t.expect(terms.reserved_kv_bytes > 0);
    try t.expectEqual(slotRingBytes(&cfg, 8), terms.qsa_ring_bytes);
    try t.expect(terms.qsa_ring_bytes > 0);
    try t.expectEqual(@as(u64, 0), terms.state_bytes);
    try t.expectEqual(@as(u64, 0), terms.checkpoint_bytes);
    try t.expectEqual(@as(u64, 0), terms.mtp_head_kv_bytes);
    // A cold prefill reserves; nothing coexists because nothing grows.
    try t.expectEqual(@as(u64, 0), terms.grow_coexist_bytes);

    // A warm append that outruns its restored capacity DOES grow, and the old
    // buffer it carries is spread over the layers that store the sequence —
    // nine of forty-eight here, not all of them.
    try t.expectEqual(@as(u32, 2), cfg.kvPerTokenLayerCount());
    try t.expectEqual(cfg.num_hidden_layers, cfg.attnCacheLayerCount());
    const warm = WarmPrefix{ .matched_tokens = seq / 2, .capacity_tokens = seq / 2, .will_donate = true };
    const grown = prefillRequestTerms(&cfg, seq, 2048, 8, chunk, warm);
    try t.expect(grown.grow_coexist_bytes > 0);
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8);
    try t.expectEqual(
        (seq / 2) * kv_per_tok / cfg.kvPerTokenLayerCount() * attnLayersPerEvalWindow(&cfg, transformer_mod.Transformer.MOE_EVAL_EVERY_N_LAYERS),
        grown.grow_coexist_bytes,
    );

    // Every arch outside the gate keeps exactly the terms it had.
    var plain = cfg;
    plain.model_type = "not_a_served_arch";
    try t.expect(!plain.reservesKvCapacity());
    try t.expectEqual(PrefillRequestTerms{}, prefillRequestTerms(&plain, seq, 2048, 8, chunk, .{}));
    try t.expectEqual(@as(u64, 0), prefill_plan.reservedPrefillTokens(&plain, seq, 2048, chunk));
}

test "mimo_v2 prefill bill: fusing qk 192 drops the global and the band score sheets" {
    const t = std.testing;
    const tiers = TierGuard.pin(true, 0);
    defer tiers.restore();
    const cfg = mimoV2BillConfig();
    const seq: u64 = 512 * 1024;
    const chunk: u64 = 1024;
    // Pin the arm, not the host: the guard and the dispatch read one switch.
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;

    // One global layer's composed sheet at this prompt, the term the kernel removes.
    const global_sheet: u64 = @as(u64, cfg.num_attention_heads) * chunk * seq * 2;
    const fused_bill = prefillNeededAtChunk(&cfg, seq, 2048, 8, chunk, .{});
    try t.expect(fused_bill < global_sheet);

    // The sliding layers' band + sink attention runs in the same kernel: no sheet left.
    try t.expectEqual(@as(u64, 0), slidingBandScoreBytes(&cfg, chunk));

    // Kill switch: composed attention restores both sheets, and the bill with them. The
    // band sheet is prompt-length independent, `window + fwd - 1` keys wide, at the same
    // coexistence the other per-layer prefill transients are billed at.
    transformer_mod.fused256_override = false;
    try t.expectEqual(MOE_PREFILL_COEXIST * @as(u64, cfg.num_attention_heads) * chunk * (128 + chunk - 1) * 2, slidingBandScoreBytes(&cfg, chunk));
    try t.expect(prefillNeededAtChunk(&cfg, seq, 2048, 8, chunk, .{}) > fused_bill +| global_sheet);

    // An arch with no ring never acquires the band term.
    var plain = cfg;
    plain.model_type = "not_a_served_arch";
    transformer_mod.fused256_override = true;
    try t.expectEqual(@as(u64, 0), slidingBandScoreBytes(&plain, chunk));
}

test "mimo_v2 prefill bill carries the FP8 trunk's dequant scratch past the GEMV width" {
    const t = std.testing;
    var cfg = mimoV2BillConfig();
    cfg.intermediate_size = 1536;
    cfg.expert_layout = .exl3_k4;
    try t.expect(cfg.usesMimoSourceTrunk());
    // The widest source FP8 linear is a sliding layer's QKV: 4*192 + 3*192 + 3*128 rows.
    const scratch: u64 = 2 * 1728 * 384 * 2;
    try t.expectEqual(scratch, fp8DequantScratchBytes(&cfg, 512));
    try t.expectEqual(@as(u64, 0), fp8DequantScratchBytes(&cfg, @intCast(fp8_block.gemv_max_rows)));
    // Both admission paths carry it, and a trunk with no FP8 linears does not.
    var dense = cfg;
    dense.expert_layout = .bf16_fused;
    try t.expectEqual(@as(u64, 0), fp8DequantScratchBytes(&dense, 512));
    try t.expectEqual(scratch, prefillNeededAtChunk(&cfg, 4096, 256, 8, 512, .{}) - prefillNeededAtChunk(&dense, 4096, 256, 8, 512, .{}));
    try t.expectEqual(scratch, prefillTransientReserveAtKv(&cfg, 8, 512, 4096) - prefillTransientReserveAtKv(&dense, 8, 512, 4096));
}

/// The Flash geometry the band sheet scales with: 64 query heads, 39 of 48 layers sliding.
fn mimoV2FlashBillConfig() model_mod.ModelConfig {
    var cfg = mimoV2BillConfig();
    cfg.num_hidden_layers = 48;
    cfg.hidden_size = 4096;
    cfg.num_attention_heads = 64;
    cfg.num_key_value_heads = 8;
    cfg.num_global_key_value_heads = 4;
    cfg.layer_is_global[3] = false;
    for ([_]u32{ 0, 5, 11, 17, 23, 29, 35, 41, 47 }) |li| cfg.layer_is_global[li] = true;
    return cfg;
}

test "mimo_v2 with an SSD tier bills a restored global layer beside its first grow" {
    // An SSD restore installs each global layer at exactly the restored rows, and the first
    // append grows it while the restored buffer is still alive. MiMo's admission never sees
    // the restore, so with a disk tier it bills one eval window of that coexistence.
    const t = std.testing;
    const cfg = mimoV2FlashBillConfig();
    const saved = prefix_cache_disk_bytes;
    defer prefix_cache_disk_bytes = saved;
    const seq: u64 = 400_000;
    const kv_bits: u64 = 8;
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const one_layer = seq * kv_per_tok / cfg.kvPerTokenLayerCount();
    prefix_cache_disk_bytes = 0;
    try t.expectEqual(@as(u64, 0), prefillRequestTerms(&cfg, seq, 8192, kv_bits, 2048, .{}).grow_coexist_bytes);
    prefix_cache_disk_bytes = 20 << 30;
    const window = attnLayersPerEvalWindow(&cfg, transformer_mod.Transformer.MOE_EVAL_EVERY_N_LAYERS);
    try t.expect(window >= 1);
    try t.expectEqual(one_layer * window, prefillRequestTerms(&cfg, seq, 8192, kv_bits, 2048, .{}).grow_coexist_bytes);
}

test "mimo_v2 prefills at the widest width its request bill admits, not at the load-time pin" {
    const t = std.testing;
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    const cfg = mimoV2FlashBillConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 500_000;
    // What load-time sizing pins beside the default hot-cache ask.
    const pin: u32 = 512;
    const roomy: u64 = 200 << 30;

    try t.expect(cfg.perRequestPrefillChunk());
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, 4096, 256, kv_bits, roomy, pin, 0, .{}));
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, roomy, pin, 0, .{}));

    // Admitted at its exact bill; a byte under steps down the ladder, not back to the pin.
    const need = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 2048, .{});
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, need, pin, 0, .{}));
    try t.expectEqual(@as(u32, 1024), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, need - 1, pin, 0, .{}));

    // The forward starts at 2048 and steps down per chunk only where a chunk stops fitting.
    try t.expectEqual(@as(usize, 2048), prefill_plan.effectivePrefillChunk(cfg.prefillScoreHeadDim(), cfg.num_attention_heads, seq, cfg.has_sliding_window, cfg.isMoe(), cfg.longCtxGated(), 0));
    try t.expectEqual(explicitPrefillChunk() == 0, adaptivePrefillChunkEnabled(&cfg));

    // 2048 is a default: an explicit `--prefill-chunk 4096` raises the ladder's top rung, billed at that width.
    const saved_explicit = generate_mod.prefill_chunk_explicit;
    const saved_chunk = generate_mod.prefill_chunk_override;
    defer {
        generate_mod.prefill_chunk_explicit = saved_explicit;
        generate_mod.prefill_chunk_override = saved_chunk;
    }
    generate_mod.prefill_chunk_explicit = true;
    generate_mod.prefill_chunk_override = 4096;
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, roomy, pin, 4096, .{}));
    const need4096 = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 4096, .{});
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, need4096 - 1, pin, 4096, .{}));
}

test "mimo_v2: a 64k prompt keeps 2048 rows at every chunk where its box has room" {
    // The per-chunk estimator once stepped a 64k MiMo prompt down to 512 on a box that had room.
    const t = std.testing;
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    const cfg = mimoV2FlashBillConfig();
    const kv_bits: u64 = 8;
    const GiB: u64 = 1 << 30;
    // A 93 GiB resident MiMo under the 120,000 MB wired floor, beside its 64k request's reserved
    // KV, ring and MTP state: the live headroom each chunk boundary reads.
    const ceiling: u64 = 115_904 * 1024 * 1024;
    const resident: u64 = 93 * GiB;
    const seq: u64 = 65_536;
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, ceiling - resident, 2048, 0, .{}));
    const terms = prefillRequestTerms(&cfg, seq, 2048, kv_bits, 2048, .{});
    const held = seq * kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) + terms.reserved_kv_bytes + terms.state_bytes + terms.qsa_ring_bytes;
    const headroom = ceiling - resident - held;
    var st: mtp_plan.AdaptiveWidthState = .{};
    var pos: u64 = 0;
    while (pos < seq) : (pos += 2048) try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, pos, headroom, 2048, 2048, &st));
}

test "the load line calls a per-request arch's pin the fallback, not the width it prefills at" {
    const t = std.testing;
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    const saved = per_request_chunk_override;
    defer per_request_chunk_override = saved;
    const cfg = mimoV2FlashBillConfig();
    var buf: [256]u8 = undefined;

    per_request_chunk_override = true;
    const line = prefillChunkLoadLine(&buf, &cfg, 1024, 8192).?;
    try t.expect(std.mem.indexOf(u8, line, "per request, up to 2048 at a short prompt") != null);
    try t.expect(std.mem.indexOf(u8, line, "fallback 1024") != null);

    // With the ladder off the pin is the width every request runs.
    per_request_chunk_override = false;
    try t.expectEqualStrings("Prefill chunk: 1024 tokens (memory-sized down from 8192; --prefill-chunk overrides)\n", prefillChunkLoadLine(&buf, &cfg, 1024, 8192).?);
    try t.expectEqual(@as(?[]const u8, null), prefillChunkLoadLine(&buf, &cfg, 8192, 8192));
}

test "mimo_v2 admission credits the hot cache: a gap the cache covers is an evict, not a refusal" {
    const t = std.testing;
    const MB: u64 = 1024 * 1024;
    const cfg = mimoV2FlashBillConfig();
    // The live refusal: 8924 MB needed, 8027 MB available, 1995 MB of hot cache this prompt does not pin.
    const bill = creditedAdmissionBill(&cfg, 8924 * MB, 8027 * MB, 1995 * MB, 1995 * MB, 4096);
    try t.expectEqual(1995 * MB, bill.evictionCredit());
    try t.expectEqual(AdmissionVerdict.evict, admissionVerdict(bill));
    // A credit short of the gap is still a refusal.
    const short = creditedAdmissionBill(&cfg, 8924 * MB, 8027 * MB, 1995 * MB, 800 * MB, 4096);
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(short));

    // Both halves read one predicate: the credits ride exactly where the inference thread evicts.
    var other = cfg;
    other.model_type = "not_a_served_arch";
    const q4 = qwen4RequestTestConfig();
    for ([_]*const model_mod.ModelConfig{ &cfg, &other, &q4 }) |c| {
        const b = creditedAdmissionBill(c, 8924 * MB, 8027 * MB, 1995 * MB, 1995 * MB, 4096);
        try t.expectEqual(scheduler_mod.admissionPassArmed(c), b.evictionCredit() > 0);
    }
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(creditedAdmissionBill(&other, 8924 * MB, 8027 * MB, 1995 * MB, 1995 * MB, 4096)));
}

test "a warm restore whose buffers hold the prompt but not the reservation bills one window of old rows" {
    // seq <= C < R: the prefill grows each KV layer to R before it writes, one eval at a time
    // (`KVCache.growToReservation`), so only one window's C-row buffers are alive beside the new.
    const t = std.testing;
    const saved_disk = prefix_cache_disk_bytes;
    defer prefix_cache_disk_bytes = saved_disk;
    prefix_cache_disk_bytes = 0;
    var q4 = qwen4RequestTestConfig();
    q4.pinned_context = 1 << 20;
    const mimo = mimoV2FlashBillConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 130_000;
    const matched: u64 = 125_000;
    const capacity: u64 = 131_072;
    const chunk: u64 = 2048;
    for ([_]*const model_mod.ModelConfig{ &q4, &mimo }) |cfg| {
        const kv = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
        const reserved = @max(reservedCacheTokens(seq, 32_000, chunk, getEffectiveContextLength(cfg)), seq);
        try t.expect(seq <= capacity and capacity < reserved);
        const window = capacity * kv / cfg.kvPerTokenLayerCount() * attnLayersPerEvalWindow(cfg, transformer_mod.Transformer.MOE_EVAL_EVERY_N_LAYERS);
        try t.expect(window > 0 and window < capacity * kv);
        for ([_]bool{ true, false }) |donate| {
            const warm = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = capacity, .will_donate = donate };
            const terms = prefillRequestTerms(cfg, seq, 32_000, kv_bits, chunk, warm);
            const kv_bill = seq * kv + terms.reserved_kv_bytes + terms.grow_coexist_bytes - terms.shared_resident_bytes;
            // A donated grow frees each old buffer after its window; a share's stay with the entry.
            const peak = if (donate) reserved * kv - capacity * kv + window else reserved * kv;
            try t.expect(kv_bill >= peak);
            try t.expectEqual(if (donate) window else 0, terms.grow_coexist_bytes);
        }
        // Nothing grows for a restore that already holds the reservation.
        const roomy = prefillRequestTerms(cfg, seq, 32_000, kv_bits, chunk, .{ .matched_tokens = matched, .capacity_tokens = reserved, .will_donate = true });
        try t.expectEqual(@as(u64, 0), roomy.grow_coexist_bytes);
    }
}

test "mimo_v2 warm bill credits only the restored global rows, never the ring" {
    const t = std.testing;
    const cfg = mimoV2FlashBillConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 140_000;
    const matched: u64 = 120_000;
    const chunk: u64 = 4096;
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    // `kvBytesPerToken` is the nine global layers; `residentCapacityTokens` skips the ringed ones.
    try t.expectEqual(@as(u32, 9), cfg.kvPerTokenLayerCount());

    const cold = prefillRequestTerms(&cfg, seq, 32_000, kv_bits, chunk, .{});
    const donated = prefillRequestTerms(&cfg, seq, 32_000, kv_bits, chunk, .{ .matched_tokens = matched, .capacity_tokens = matched, .will_donate = true });
    try t.expectEqual(matched * kv_per_tok, donated.shared_resident_bytes);
    // The restored ring holds only its retained rows; the first append regrows it, so the ring
    // and both checkpoint copies are billed whole on a warm turn too.
    try t.expect(donated.qsa_ring_bytes > 0);
    try t.expectEqual(cold.qsa_ring_bytes, donated.qsa_ring_bytes);
    // A share is copied by its first append: nothing credited.
    const shared = prefillRequestTerms(&cfg, seq, 32_000, kv_bits, chunk, .{ .matched_tokens = matched, .capacity_tokens = matched, .will_donate = false });
    try t.expectEqual(@as(u64, 0), shared.shared_resident_bytes);
}

test "mimo_v2 at 500k, kv8, chunk 2048: the fused sliding arm drops exactly the band sheet" {
    const t = std.testing;
    const cfg = mimoV2FlashBillConfig();
    const seq: u64 = 500_000;
    const chunk: u64 = 2048;
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    defer transformer_mod.sliding_prefill_fused_override = null;

    transformer_mod.sliding_prefill_fused_override = false;
    const band = slidingBandScoreBytes(&cfg, chunk);
    try t.expectEqual(MOE_PREFILL_COEXIST * 64 * chunk * (128 + chunk - 1) * 2, band);
    const composed = prefillNeededAtChunk(&cfg, seq, 2048, 8, chunk, .{});
    transformer_mod.sliding_prefill_fused_override = null;
    try t.expectEqual(composed - band, prefillNeededAtChunk(&cfg, seq, 2048, 8, chunk, .{}));

    // One predicate decides the dispatch and the bill, at every width and switch state.
    for ([_]?bool{ null, false }) |ov| {
        transformer_mod.sliding_prefill_fused_override = ov;
        for ([_]u64{ 1, 8, 15, 16, 512, 2048, 4096 }) |fwd| {
            try t.expectEqual(transformer_mod.slidingPrefillFused(&cfg, fwd), slidingBandScoreBytes(&cfg, fwd) == 0);
        }
    }
}

test "generated think tags require an unambiguous literal template opener" {
    try std.testing.expectEqualStrings("<think>", templateThinkOpener("{{ '<think>' }} <think>").?);
    try std.testing.expect(templateThinkOpener("{{ '<think:opensource>' }}") == null);
    try std.testing.expect(templateThinkOpener("<think> <think:opensource>") == null);
    try std.testing.expect(templateThinkOpener("<think:{{ suffix }}>") == null);
    try std.testing.expect(templateThinkOpener("assistant") == null);
}

test "resolveBind: no flags bind 127.0.0.1:12345" {
    const d = resolveBind(null, null);
    try std.testing.expectEqualStrings("127.0.0.1", d.host);
    try std.testing.expectEqual(@as(u16, 12345), d.port);
    const e = resolveBind("0.0.0.0", 23817);
    try std.testing.expectEqualStrings("0.0.0.0", e.host);
    try std.testing.expectEqual(@as(u16, 23817), e.port);
    const p = resolveBind(null, 23817);
    try std.testing.expectEqualStrings("127.0.0.1", p.host);
    try std.testing.expectEqual(@as(u16, 23817), p.port);
    const h = resolveBind("192.168.7.9", null);
    try std.testing.expectEqualStrings("192.168.7.9", h.host);
    try std.testing.expectEqual(@as(u16, 12345), h.port);
}

test "startupRefusal: AddressInUse refuses naming the port" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("port 12345 is already in use", startupRefusal(error.AddressInUse, "127.0.0.1", 12345, &buf).?);
    try std.testing.expectEqualStrings("port 23817 is already in use", startupRefusal(error.AddressInUse, "127.0.0.1", 23817, &buf).?);
    try std.testing.expect(startupRefusal(error.AccessDenied, "127.0.0.1", 23817, &buf) == null);
    try std.testing.expect(startupRefusal(error.SystemResources, "127.0.0.1", 23817, &buf) == null);
}

test "bindAddress: localhost is loopback; wildcards stay" {
    const l = try bindAddress("localhost", 12345);
    try std.testing.expect(std.mem.eql(u8, &l.ip4.bytes, &.{ 127, 0, 0, 1 }));
    try std.testing.expectEqual(@as(u16, 12345), l.ip4.port);
    const e = try bindAddress("127.0.0.1", 23817);
    try std.testing.expect(std.mem.eql(u8, &e.ip4.bytes, &.{ 127, 0, 0, 1 }));
    try std.testing.expectEqual(@as(u16, 23817), e.ip4.port);
    const w = try bindAddress("0.0.0.0", 12345);
    try std.testing.expect(std.mem.eql(u8, &w.ip4.bytes, &.{ 0, 0, 0, 0 }));
    try std.testing.expectEqual(@as(u16, 12345), w.ip4.port);
}

test "bindAddress: garbage is refused by name" {
    try std.testing.expectError(error.InvalidHost, bindAddress("example.com", 80));
    try std.testing.expectError(error.InvalidHost, bindAddress("999.1.2.3", 80));
    try std.testing.expectError(error.InvalidHost, bindAddress("1.2.3", 80));
    var buf: [512]u8 = undefined;
    try std.testing.expectEqualStrings("invalid --host 'example.com'", startupRefusal(error.InvalidHost, "example.com", 80, &buf).?);
    try std.testing.expectEqualStrings("invalid --host '999.1.2.3'", startupRefusal(error.InvalidHost, "999.1.2.3", 80, &buf).?);
}

test "MiMo MTP state bill is constant and follows retained storage" {
    const t = std.testing;
    const saved = server_config.default_force_mtp;
    defer server_config.default_force_mtp = saved;
    server_config.default_force_mtp = false;
    var cfg = mimoV2BillConfig();
    cfg.hidden_size = 4096;
    cfg.num_key_value_heads = 8;
    cfg.mtp_override = false;
    const kv_row: u64 = 8 * (192 + 128) * 2;
    const kv: u64 = 3 * (2 * 128 - 1) * kv_row;
    const hiddens: u64 = 256 * 4096 * 2;
    const catchup: u64 = 3 * 128 * 4096 * 2;
    const expected: u64 = 2 * (kv + hiddens) + catchup;
    for ([_]u64{ 1024, 131072 }) |seq| {
        const off = prefillRequestTerms(&cfg, seq, 2048, 8, 512, .{});
        const on = prefillRequestTerms(&cfg, seq, 2048, 8, 512, .{ .mtp_on = true });
        try t.expectEqual(@as(u64, 0), off.state_bytes);
        try t.expectEqual(expected, on.state_bytes);
        try t.expectEqual(@as(u64, 0), on.mtp_head_kv_bytes);
    }
    const off_reserve = prefillTransientReserve(&cfg, 8, 512);
    const off_per_token = sessionBytesPerToken(&cfg, 8);
    cfg.mtp_override = true;
    try t.expectEqual(expected * 5 / 4, prefillTransientReserve(&cfg, 8, 512) - off_reserve);
    try t.expectEqual(off_per_token, sessionBytesPerToken(&cfg, 8));
    cfg.hidden_size = 2048;
    cfg.num_key_value_heads = 4;
    cfg.sliding_window = 64;
    const smaller: u64 = 2 * (3 * (2 * 64 - 1) * 4 * (192 + 128) * 2 + 256 * 2048 * 2) + 3 * 64 * 2048 * 2;
    try t.expectEqual(smaller, prefillRequestTerms(&cfg, 1024, 2048, 4, 512, .{}).state_bytes);
}

const byteTokenizerForTests = tokenizer_mod.byteTokenizerForTests;
const deinitTestTokenizer = tokenizer_mod.deinitTestTokenizer;

/// One surface's forced tool_choice, end to end without a model: its wire
/// shapes parse, the prompt renders through each family's template, the force
/// arms, and a model that thinks "hmm" (when the turn thinks) meets the call.
pub fn expectSurfaceForcesCall(tool_choice_json: []const u8, openai_tools: []const u8, forced_name: ?[]const u8) !void {
    const a = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    var tok = try byteTokenizerForTests(a, arena_state.allocator(), &.{ "<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_start|>", "<|im_end|>" });
    defer deinitTestTokenizer(&tok);

    const parsed = try std.json.parseFromSlice(std.json.Value, a, tool_choice_json, .{});
    defer parsed.deinit();
    const choice = chat_mod.parseToolChoice(parsed.value);
    const instruction = try choice.instruction(a);
    defer if (instruction) |i| a.free(i);

    const Family = struct { tpl: []const u8, head: []const u8, name_tail: []const u8, sep: []const u8 };
    const families = [_]Family{
        .{ .tpl = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), .head = "<tool_call>\n<function=", .name_tail = ">\n", .sep = "\n\n" },
        .{ .tpl = @embedFile("fixtures/mimo_v26_chat_template.jinja"), .head = "<tool_call><function=", .name_tail = ">", .sep = "" },
    };
    const messages = [_]chat_mod.Message{.{ .role = "user", .content = "What time is it?" }};
    for (families) |f| {
        var config = chat_mod.ChatConfig{ .chat_template = f.tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
        // `required` commits the only declared name too: the call names a declared function.
        const call = try std.mem.concat(a, u8, &.{ f.head, forced_name orelse "get_time", f.name_tail });
        defer a.free(call);
        for ([_]bool{ true, false }) |thinking| {
            const rendered = try chat_mod.renderChatTemplate(a, &messages, &config, openai_tools, instruction, thinking, null, false);
            defer a.free(rendered);
            try std.testing.expect(std.mem.indexOf(u8, rendered, "You MUST call") != null);
            const prompt_ids = try tok.encode(a, rendered);
            defer a.free(prompt_ids);

            var cf = armCallForce(a, &tok, &config, prompt_ids, thinking, choice, openai_tools) orelse return error.ForceNotArmed;
            defer cf.deinit(a);
            var published = std.ArrayList(u32).empty;
            defer published.deinit(a);
            var thought = false;
            while (true) {
                const next = cf.decide(published.items);
                if (next == .force) {
                    try published.append(a, next.force);
                } else if (next == .choose) {
                    try published.append(a, next.choose[0]);
                } else if (!cf.pending()) {
                    break;
                } else {
                    if (thought) return error.ForceNeverFired;
                    thought = true;
                    const ids = try tok.encode(a, "hmm</think>");
                    defer a.free(ids);
                    try published.appendSlice(a, ids);
                }
            }
            const text = try tok.decode(a, published.items);
            defer a.free(text);
            // Qwen's template opens the thought in the prompt; MiMo's leaves the opener to the model.
            const opener: []const u8 = if (thinking and f.sep.len == 0) "<think>" else "";
            const want = if (thinking)
                try std.mem.concat(a, u8, &.{ opener, "hmm</think>", f.sep, call })
            else
                try a.dupe(u8, call);
            defer a.free(want);
            try std.testing.expectEqualStrings(want, text);
        }
    }
}

test "chat completions: a forced tool_choice is committed at the answer boundary on both families" {
    const tools =
        \\[{"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}}}}},
        \\ {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]
    ;
    try expectSurfaceForcesCall("{\"type\":\"function\",\"function\":{\"name\":\"get_time\"}}", tools, "get_time");
    try expectSurfaceForcesCall("\"required\"", tools, null);
}

test "responses: a forced tool_choice is committed at the answer boundary on both families" {
    const a = std.testing.allocator;
    const wire =
        \\[{"type":"function","name":"get_time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}}}},
        \\ {"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, wire, .{});
    defer parsed.deinit();
    const tools = try responses_mod.buildToolsJson(a, parsed.value.array);
    defer a.free(tools);
    try expectSurfaceForcesCall("{\"type\":\"function\",\"name\":\"get_time\"}", tools, "get_time");
    try expectSurfaceForcesCall("\"required\"", tools, null);
}

test "tool_choice auto and none arm no forced call" {
    const a = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    var tok = try byteTokenizerForTests(a, arena_state.allocator(), &.{ "<think>", "</think>" });
    defer deinitTestTokenizer(&tok);
    var config = chat_mod.ChatConfig{ .chat_template = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
    const ids = try tok.encode(a, "<|im_start|>assistant\n<think>\n");
    defer a.free(ids);
    try std.testing.expect(armCallForce(a, &tok, &config, ids, true, .auto, null) == null);
    try std.testing.expect(armCallForce(a, &tok, &config, ids, true, .none, null) == null);
}

test "tool_choice none bans each family's one-token call opener, and nothing else" {
    const a = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    var tok = try byteTokenizerForTests(a, arena_state.allocator(), &.{ "<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_start|>", "<|im_end|>" });
    defer deinitTestTokenizer(&tok);
    var split = try byteTokenizerForTests(a, arena_state.allocator(), &.{ "<think>", "</think>" });
    defer deinitTestTokenizer(&split);
    for ([_][]const u8{
        @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"),
        @embedFile("fixtures/mimo_v26_chat_template.jinja"),
        @embedFile("fixtures/glm53_chat_template.jinja"),
    }) |tpl| {
        var config = chat_mod.ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
        const ban = noneCallBan(a, &tok, &config) orelse return error.OpenerNotBanned;
        try std.testing.expectEqual(tok.special_tokens.get("<tool_call>").?, ban.id);
        try std.testing.expectEqual(@as(f32, -100), ban.delta);
        // An opener the tokenizer splits would ban its first piece, a plain `<`, everywhere.
        try std.testing.expect(noneCallBan(a, &split, &config) == null);
    }
}

test "chat_template_kwargs is read in one place, and only a bool counts" {
    const a = std.testing.allocator;
    const Case = struct { body: []const u8, want: ?bool };
    const cases = [_]Case{
        .{ .body = "{\"chat_template_kwargs\":{\"preserve_thinking\":false}}", .want = false },
        .{ .body = "{\"chat_template_kwargs\":{\"preserve_thinking\":true}}", .want = true },
        .{ .body = "{\"chat_template_kwargs\":{\"preserve_thinking\":\"off\"}}", .want = null },
        .{ .body = "{\"chat_template_kwargs\":[1]}", .want = null },
        .{ .body = "{\"preserve_thinking\":false}", .want = null },
        .{ .body = "{}", .want = null },
    };
    for (cases) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, c.body, .{});
        defer parsed.deinit();
        try std.testing.expectEqual(c.want, chatTemplateKwargBool(parsed.value.object, "preserve_thinking"));
    }
}

test "settingsPropsJson: /props reports the prefill decode share" {
    const frag = try settingsPropsJson(testing.allocator, .{ .engine = "mlx", .kv_quant = "off", .prefill_chunk = 8192, .mtp_loaded = false, .mtp_default_on = false, .mtp_acceptance = .exact, .mtp_depth = 0, .mtp_adaptive = false, .max_mtp_ctx = 0, .drafter = "none", .max_concurrent = 8, .prefix_cache_mem_bytes = 0, .prefix_cache_disk_bytes = 0, .prefill_decode_share = 0.5 });
    defer testing.allocator.free(frag);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, frag[",\"settings\":".len..], .{});
    defer parsed.deinit();
    const v = parsed.value.object.get("prefill_decode_share") orelse return error.MissingShare;
    try testing.expectApproxEqAbs(@as(f64, 0.5), v.float, 1e-6);
}

test "liveSessions copies the queue snapshot and stamps the effective context limit" {
    const t = std.testing;
    var sch: scheduler_mod.Scheduler = undefined;
    sch.io = t.io;
    sch.queue_mu = .init;
    sch.live_sessions = undefined;
    sch.live_session_count = 2;
    var row = instr.Session.init("org/live-test", .decode, 1700, 1200, 200, 4096);
    row.request_id = 42;
    row.max_tokens = 32000;
    row.elapsed_seconds = 12.5;
    row.entry_id = 7;
    sch.live_sessions[0] = row;
    sch.live_sessions[1] = instr.Session.init("org/never-loaded", .prefill, 512, 0, 0, 1024);
    sch.digest_mu = .init;
    var restored = instr.Session.init("org/live-test", .cached, 1200, 1200, 0, 2048);
    restored.entry_id = 7;
    var idle = instr.Session.init("org/live-test", .cached, 300, 300, 0, 512);
    idle.entry_id = 9;
    sch.cached_sessions[0] = restored;
    sch.cached_sessions[1] = idle;
    sch.cached_session_count = 2;
    const prev_sch = global_scheduler;
    defer global_scheduler = prev_sch;
    global_scheduler = &sch;

    const reg = try ModelRegistry.init(t.allocator, t.io, null, 1, 0, null);
    defer reg.deinit();
    const lm = try reg.registerStub("org/live-test", "/path/to/m", null);
    var cfg = std.mem.zeroes(model_mod.ModelConfig);
    lm.state = .ready;
    lm.config = &cfg;
    defer lm.config = null; // LoadedModel.deinit destroys a non-null config — detach first
    const prev_ctx = server_config.max_context_size;
    defer server_config.max_context_size = prev_ctx;
    server_config.max_context_size = 8192; // manualContext wins: no GPU sizing

    var buf: [2 * instr.MAX_SESSIONS]instr.Session = undefined;
    const rows = liveSessions(reg, &buf);
    // The entry the decode row restored from is listed once, as that row.
    try t.expectEqual(@as(usize, 3), rows.len);
    try t.expectEqual(instr.Session.Phase.cached, rows[2].phase);
    try t.expectEqual(@as(u32, 300), rows[2].context_tokens);
    try t.expectEqual(@as(u32, 8192), rows[2].context_length);
    try t.expectEqualStrings("org/live-test", rows[0].model());
    try t.expectEqual(@as(u64, 42), rows[0].request_id);
    try t.expectEqual(@as(u32, 32000), rows[0].max_tokens);
    try t.expectEqual(@as(f64, 12.5), rows[0].elapsed_seconds);
    // The ready model's row carries its EFFECTIVE limit (`--ctx-size` ranks
    // above the model default, auto sizing never runs here).
    try t.expectEqual(@as(u32, 8192), rows[0].context_length);
    // An id the registry never resolved keeps 0 = unknown — a placeholder,
    // never a modeled guess — and the queue snapshot itself is untouched:
    // the reader copies, it never mutates the inference thread's rows.
    try t.expectEqual(@as(u32, 0), rows[1].context_length);
    try t.expectEqual(@as(u32, 0), sch.live_sessions[0].context_length);

    // Before `serve()` wires a scheduler: an empty slice, not a crash.
    global_scheduler = null;
    try t.expectEqual(@as(usize, 0), liveSessions(reg, &buf).len);
}

test "resolvedPrefixCacheMem: RAM-off bills zero despite a configured disk tier" {
    const capacity = prefix_cache_capacity;
    const ram = prefix_cache_ram_enabled;
    defer prefix_cache_capacity = capacity;
    defer prefix_cache_ram_enabled = ram;
    prefix_cache_capacity = 8;
    prefix_cache_ram_enabled = false;
    try testing.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
    prefix_cache_ram_enabled = true;
    prefix_cache_capacity = 0;
    try testing.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
}

test "disabled prefix cache: sizing releases the cache reserve on every arch" {
    const saved_capacity = prefix_cache_capacity;
    const saved_ask = prefix_cache_mem_bytes;
    defer prefix_cache_capacity = saved_capacity;
    defer prefix_cache_mem_bytes = saved_ask;
    const saved_ram = prefix_cache_ram_enabled;
    defer prefix_cache_ram_enabled = saved_ram;
    var gated = longCtxTestConfig();
    var other = longCtxTestConfig();
    other.model_type = "not_a_served_arch";
    const configs = [_]*model_mod.ModelConfig{ &gated, &other };
    for ([_]u64{ 0, 2 << 30, 60 << 30 }) |ask| {
        prefix_cache_mem_bytes = ask;
        prefix_cache_capacity = 0;
        for (configs) |cfg| {
            try testing.expectEqual(@as(u64, 0), ctxSizingCacheReserve(cfg));
        }
        try testing.expectEqual(@as(u64, 0), legacyPrefixCacheAsk(&other));
        prefix_cache_capacity = 32;
        prefix_cache_ram_enabled = false;
        for (configs) |cfg| {
            try testing.expectEqual(@as(u64, 0), ctxSizingCacheReserve(cfg));
        }
        try testing.expectEqual(@as(u64, 0), legacyPrefixCacheAsk(&other));
        prefix_cache_ram_enabled = true;
    }
    prefix_cache_capacity = 32;
    prefix_cache_ram_enabled = true;
    prefix_cache_mem_bytes = 10 << 30;
    try testing.expectEqual(CTX_SIZING_CACHE_RESERVE, ctxSizingCacheReserve(&gated));
    try testing.expectEqual(prefix_cache_mem_bytes, ctxSizingCacheReserve(&other));
    try testing.expectEqual(prefix_cache_mem_bytes, legacyPrefixCacheAsk(&other));
    // GLM's context is sized with no cache reserve (its admission evicts); its unnamed ask is
    // its 1 GiB tier and a named one stands.
    const glm = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    try testing.expectEqual(@as(u64, 0), ctxSizingCacheReserve(&glm));
    try testing.expectEqual(GLM_PREFIX_CACHE_MEM_DEFAULT, defaultPrefixCacheAskFor(&glm, PREFIX_CACHE_MEM_DEFAULT, false, 8 << 30));
    try testing.expectEqual(PREFIX_CACHE_MEM_DEFAULT, defaultPrefixCacheAskFor(&glm, PREFIX_CACHE_MEM_DEFAULT, true, 8 << 30));
    try testing.expectEqual(@as(u64, 6 << 30), defaultPrefixCacheAskFor(&glm, 6 << 30, true, 8 << 30));
    try testing.expectEqual(@as(u64, 8 << 30), defaultPrefixCacheAskFor(&other, PREFIX_CACHE_MEM_DEFAULT, false, 8 << 30));
    // `--prefix-cache-mem off --prefix-cache-disk 12GB` keeps no idle RAM.
    const saved_disk = prefix_cache_disk_bytes;
    defer prefix_cache_disk_bytes = saved_disk;
    prefix_cache_disk_bytes = 12 << 30;
    prefix_cache_ram_enabled = false;
    try testing.expectEqual(@as(u64, 0), legacyPrefixCacheAsk(&other));
}

test "GLM serving memory bills one latent at the request's KV width and pooled index per token" {
    const cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    const saved = configured_kv_quant;
    defer configured_kv_quant = saved;
    configured_kv_quant = null;
    try std.testing.expectEqual(@as(u64, 8), defaultKvBits(&cfg));
    try std.testing.expectEqual(@as(u64, 11968), sessionBytesPerToken(&cfg, 16));
    // kv8: 11 layers x (512 codes + 8 BF16 scale/bias pairs) beside the BF16 pooled index.
    try std.testing.expectEqual(@as(u64, 11 * 544 + 704), sessionBytesPerToken(&cfg, 8));
    configured_kv_quant = transformer_mod.KVQuantConfig.dense;
    try std.testing.expectEqual(@as(u64, 16), defaultKvBits(&cfg));
    configured_kv_quant = null;
    // kv8 retains 480 fewer growth bytes per token and adds the pending MLA layer's dequant and quantizer scratch.
    try std.testing.expectEqual(prefillTransientReserveAtKv(&cfg, 16, 2048, 65536) + (2051 * 1024 + 2048 * 544), prefillTransientReserveAtKv(&cfg, 8, 2048, 65536) + 65536 * 480);
    try std.testing.expect(prefillNeededAtChunk(&cfg, 65536, 1024, 8, 2048, .{}) < prefillNeededAtChunk(&cfg, 65536, 1024, 16, 2048, .{}));
    try std.testing.expectEqual(cfg.ssmCheckpointBytes() + cfg.qsaRingBytes(), slotRingBytes(&cfg, 16));
    try std.testing.expectEqual(@as(u64, 0), ctxSizingCacheReserve(&cfg));
    try std.testing.expectEqual(@as(u64, 704), statePerTokenBilled(&cfg));
    try std.testing.expectEqual(@as(u64, 0), cfg.layerKvBytes(0));
    try std.testing.expectEqual(@as(u64, 1024), cfg.layerKvBytes(3));
    try std.testing.expect(prefillTransientReserveAtKv(&cfg, 16, 2048, 500000) < 6 * 1024 * 1024 * 1024);
}

test "a second GLM admission leaves free the cache growth the first was admitted to allocate" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.pinned_context = 1_048_576;
    const gib: u64 = 1 << 30;
    const prompt: usize = 16;
    const max_tokens: u32 = 1_048_560;
    for ([_]u64{ 16, 8 }) |kv_bits| {
        const bill = prefillAdmissionBill(&cfg, prompt, max_tokens, if (kv_bits == 16) transformer_mod.KVQuantConfig.dense else transformer_mod.KVQuantConfig.affine(8), false, null, .{});
        const nums = prefillBillNumbersNow(&cfg, prompt, max_tokens, if (kv_bits == 16) transformer_mod.KVQuantConfig.dense else transformer_mod.KVQuantConfig.affine(8), false, false);
        const commit = admissionCommitBytes(&cfg, prompt, max_tokens, kv_bits);
        try std.testing.expectEqual(commit, nums.commit);
        // What the request retains is most of its bill, and it is not resident at admission.
        try std.testing.expect(commit > bill.needed / 2 and commit <= bill.needed);
        // Headroom that admits A alone and, a tick later with A's growth unallocated, B.
        const headroom = bill.needed + gib;
        const b = scheduler_mod.MemoryBill{ .needed = bill.needed, .available = headroom };
        try std.testing.expectEqual(@as(usize, 1), scheduler_mod.admitsWithinMemory(&.{b}, true, 0));
        try std.testing.expectEqual(@as(usize, 0), scheduler_mod.admitsWithinMemory(&.{b}, true, commit));
    }
    // kv8 audit shape: one request's eventual state is 6.669 GiB.
    const kv8 = admissionCommitBytes(&cfg, prompt, max_tokens, 8);
    try std.testing.expect(kv8 > 6 * gib + gib * 2 / 3 and kv8 < 6 * gib + gib * 7 / 10);
    try std.testing.expect(admissionCommitBytes(&cfg, prompt, max_tokens, 16) > kv8);
}

test "a Qwen or MiMo request owes no growth after admission, its reserved capacity is taken at its first grow" {
    for ([_][]const u8{ "qwen4_exp", "mimo_v2" }) |model_type| {
        const cfg = model_mod.ModelConfig{ .model_type = model_type };
        try std.testing.expectEqual(@as(u64, 0), admissionCommitBytes(&cfg, 40_000, 4096, 8));
    }
}

test "GLM prefill bills the packed selection planes of its history" {
    const cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    const attention = @import("glm5/glm5_attention.zig");
    const short: u64 = 1 << 19;
    const long: u64 = 1 << 20;
    // Beyond selection planes, only the growing kv8 latent (544) and pooled index (64) scale with history.
    const growth = (long - short) * (544 + 64);
    const planes = try attention.selectionScratchBytes(long) - try attention.selectionScratchBytes(short);
    try std.testing.expectEqual(growth + planes, prefillTransientReserveAtKv(&cfg, 8, 2048, long) - prefillTransientReserveAtKv(&cfg, 8, 2048, short));
}

test "a streamed GLM bills its fill peak per request and its latent at the KV width plus the pooled index per planned token" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    const saved = configured_kv_quant;
    defer configured_kv_quant = saved;
    var planned: [2]u64 = undefined;
    for ([_]transformer_mod.KVQuantConfig{ .dense, transformer_mod.KVQuantConfig.affine(8) }, [_]u64{ 16, 8 }, &planned) |kv, bits, *bytes| {
        configured_kv_quant = kv;
        try std.testing.expectEqual(bits, defaultKvBits(&cfg));
        cfg.expert_streaming = false;
        const resident = prefillNeededAtChunk(&cfg, 4096, 2048, bits, 2048, .{});
        cfg.expert_streaming = true;
        cfg.expert_fill_peak_bytes = 3 << 30;
        try std.testing.expectEqual(resident + (3 << 30), prefillNeededAtChunk(&cfg, 4096, 2048, bits, 2048, .{}));
        bytes.* = expertStreamingPlannedKvBytes(&cfg);
        try std.testing.expectEqual(sessionBytesPerToken(&cfg, bits) * expertStreamingPlannedSeq(&cfg) + slotRingBytes(&cfg, bits), bytes.*);
    }
    try std.testing.expect(planned[1] < planned[0]);
}

test "a streamed Qwen plans its QSA history beside its KV for every planned token" {
    const fused_guard = qsaScoreFusedOffGuard();
    defer fused_guard.deinit();
    var cfg = qwen4RequestTestConfig();
    cfg.expert_streaming = true;
    const saved = configured_kv_quant;
    defer configured_kv_quant = saved;
    configured_kv_quant = transformer_mod.KVQuantConfig.affine(8);
    const state = statePerTokenBilled(&cfg);
    try std.testing.expect(state > 0);
    const per_token = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8) + state;
    try std.testing.expectEqual(per_token * expertStreamingPlannedSeq(&cfg) + slotRingBytes(&cfg, 8), expertStreamingPlannedKvBytes(&cfg));
}

test "a streamed --vision load proves its largest image encode beside the budget, with the planned KV" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.expert_streaming = true;
    const kv = expertStreamingPlannedKvBytes(&cfg);
    try std.testing.expectEqual(kv, expertStreamingBesideBudgetBytes(&cfg));
    cfg.expert_vision_encode_bytes = largestImageEncodeBytes(&cfg);
    try std.testing.expect(cfg.expert_vision_encode_bytes > 1 << 30);
    try std.testing.expectEqual(kv + cfg.expert_vision_encode_bytes, expertStreamingBesideBudgetBytes(&cfg));
    try std.testing.expect(!ssdBudgetAdmits(32 << 30, expertStreamingBesideBudgetBytes(&cfg), (32 << 30) + kv));
}

fn glmAssistantFixture() !model_mod.ModelConfig {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    cfg.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    cfg.dflash_capture_bytes_per_token = (5 * 4096 + 2 * 4096) * 2 + 5 * 8 * 128 * 4;
    return cfg;
}

test "GLM MLA prefill scratch is held by the MLA layers a pending window can contain" {
    const cfg = try glmAssistantFixture();
    // Layers 3, 7, ..., 43: two consecutive layers never hold two of them.
    try std.testing.expectEqual(@as(u32, 1), glmMlaLayersPending(&cfg, 2));
    try std.testing.expectEqual(@as(u32, 2), glmMlaLayersPending(&cfg, 5));
    var adjacent = cfg;
    adjacent.full_attention_interval = 1;
    try std.testing.expectEqual(@as(u32, 2), glmMlaLayersPending(&adjacent, 2));
}

test "GLM release defaults advertise the full context at the widest prefill chunk that keeps it" {
    var cfg = try glmAssistantFixture();
    cfg.pinned_prefill_chunk = 0;
    const kv_bits: u64 = 8;
    const cap = cfg.contextCap();
    // Sushi-2.5bpw with vision and the A4 assistant, as its boot measured: 104.35 GB active, 115.9 GB ceiling.
    const ceiling: u64 = 115_904 * 1024 * 1024;
    const active: u64 = 104_350_000_000;
    try std.testing.expectEqual(@as(u32, 2048), resolvePrefillChunk(&cfg, kv_bits, ceiling, active, 0, 0));
    try std.testing.expectEqual(cap, advertisedContextAt(&cfg, kv_bits, ceiling, active, 2048));
    // A 1 GiB idle cache reserve takes the same bytes as a 1 GiB lower ceiling: 2048 rows would cost
    // advertised context there, 1024 keeps all of it.
    const reserved = ceiling - 1024 * 1024 * 1024;
    try std.testing.expect(advertisedContextAt(&cfg, kv_bits, reserved, active, 2048) < cap);
    try std.testing.expectEqual(@as(u32, 1024), resolvePrefillChunk(&cfg, kv_bits, reserved, active, 0, 0));
    try std.testing.expectEqual(cap, advertisedContextAt(&cfg, kv_bits, reserved, active, 1024));
    // The least free memory at which 1024 rows still advertise the position cap keeps 1024.
    var low: u64 = 0;
    var high: u64 = ceiling - active;
    while (low < high) {
        const mid = low + (high - low) / 2;
        if (advertisedContextAt(&cfg, kv_bits, active + mid, active, 1024) >= cap) high = mid else low = mid + 1;
    }
    try std.testing.expect(advertisedContextAt(&cfg, kv_bits, active + low, active, 2048) < cap);
    try std.testing.expectEqual(@as(u32, 1024), resolvePrefillChunk(&cfg, kv_bits, active + low, active, 0, 0));
    // Far less memory: every widening costs advertised context.
    try std.testing.expectEqual(@as(u32, 512), resolvePrefillChunk(&cfg, kv_bits, active + 8 * 1024 * 1024 * 1024, active, 0, 0));
}

test "GLM requests prefill at 2048 and step down a rung only where a chunk stops fitting" {
    const t = std.testing;
    const cfg = try glmAssistantFixture();
    const kv_bits: u64 = 8;
    try t.expect(cfg.perRequestPrefillChunk());
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, 4096, 256, kv_bits, 1 << 40, 2048, 0, .{}));
    const unpinned = explicitPrefillChunk() == 0;
    try t.expectEqual(unpinned, adaptivePrefillChunkEnabled(&cfg));

    // Release defaults: a 1M request is admitted with a 2048-row tail, and 2048 still fits there.
    const available: u64 = 115_904 * 1024 * 1024 - 104_350_000_000;
    const top: u64 = 1_048_576 - 256;
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, top, 256, kv_bits, available, 2048, 0, .{}));
    var st: mtp_plan.AdaptiveWidthState = .{};
    const kv_top = sessionBytesPerToken(&cfg, kv_bits) * top;
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, top, available - kv_top, 2048, 2048, &st));

    // A tighter box: the top only fits at 1024 rows, and that tail's bill is what admits it.
    const at_2048 = prefillNeededAtChunk(&cfg, top, 256, kv_bits, 2048, .{});
    try t.expectEqual(@as(u32, 1024), chooseRequestPrefillChunk(&cfg, top, 256, kv_bits, at_2048 - 1, 2048, 0, .{}));
    try t.expect(prefillNeededAtChunk(&cfg, top, 256, kv_bits, 1024, .{}) <= at_2048 - 1);
    // Early chunks keep 2048; the chunk that no longer fits beside the KV steps down one rung.
    st = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, 4096, prefillChunkCost(&cfg, kv_bits, 2048, 4096), 2048, 2048, &st));
    try t.expectEqual(@as(u32, 1024), adaptivePrefillWidth(&cfg, kv_bits, top, prefillChunkCost(&cfg, kv_bits, 2048, top) - 1, 2048, 2048, &st));
    // Once stepped down, a request never widens again.
    try t.expectEqual(@as(u32, 1024), adaptivePrefillWidth(&cfg, kv_bits, top, available, 1024, 2048, &st));

    // A small box narrows from the first chunk.
    st = .{};
    try t.expectEqual(@as(u32, 512), adaptivePrefillWidth(&cfg, kv_bits, 0, prefillChunkCost(&cfg, kv_bits, 512, 0), 2048, 2048, &st));
}

test "GLM prefill chunk never widens past the width its numerics are built for" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    const roomy: u64 = 1 << 40;
    for ([_]bool{ false, true }) |assistant| {
        cfg.dflash_loaded = assistant;
        for ([_]u64{ 16, 8 }) |kv_bits| try std.testing.expectEqual(@as(u32, 2048), resolvePrefillChunk(&cfg, kv_bits, roomy, 0, 0, 0));
    }
}

test "GLM serving DFlash2 bill includes the bounded window captures replay and scratch" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    try std.testing.expectEqual(@as(u64, 0), glmDflashRequestBytes(&cfg, 2048));
    cfg.dflash_loaded = true;
    cfg.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    cfg.dflash_capture_bytes_per_token = (5 * 4096 + 2 * 4096) * 2 + 5 * 8 * 128 * 4;
    const needed = glmDflashRequestBytes(&cfg, 2048);
    try std.testing.expect(needed >= cfg.dflash_window_bytes * 4 + 256 * 1024 * 1024);
    try std.testing.expect(needed >= cfg.ssmCheckpointBytes() * 3);
    try std.testing.expect(needed < 2 * 1024 * 1024 * 1024);
    try std.testing.expect(glmDflashRequestBytes(&cfg, 1024) < needed);
}

test "the concurrency verdict: a resident GLM and qwen4 batch, an exclusive model queues, another MoE interleaves" {
    const glm = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    const qwen4 = model_mod.ModelConfig{ .model_type = "qwen4_exp", .full_attention_interval = 4, .num_experts = 8, .num_experts_per_tok = 2 };
    const moe = model_mod.ModelConfig{ .model_type = "qwen3_moe", .num_experts = 8, .num_experts_per_tok = 2 };
    try std.testing.expectEqual(scheduler_mod.BatchVerdict.ok, batchVerdictOf(&glm, false));
    try std.testing.expectEqual(scheduler_mod.BatchVerdict.arch, batchVerdictOf(&moe, false));
    try std.testing.expectEqual(scheduler_mod.BatchVerdict.exclusive, batchVerdictOf(&glm, true));
    try std.testing.expectEqual(scheduler_mod.BatchVerdict.ok, batchVerdictOf(&qwen4, false));
    try std.testing.expectEqual(scheduler_mod.BatchVerdict.arch, batchVerdictOf(null, false));
    try std.testing.expectEqualStrings("batched decode off; concurrent requests interleave serially", concurrencyClause(.arch));
    try std.testing.expectEqualStrings("batched decode off; concurrent requests queue, one at a time", concurrencyClause(.exclusive));
    try std.testing.expectEqualStrings("batched decode on", concurrencyClause(.ok));
}

test "GLM DFlash2 reserve at prefill end fits the admission bill at every request shape" {
    const reserve = @import("glm5/glm5_dflash_reserve.zig");
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    const mla_layers: usize = cfg.num_hidden_layers / cfg.full_attention_interval;
    const pool_row: usize = @as(usize, cfg.indexer_head_dim) * 2;
    const shapes = [_][2]u64{ .{ 16, std.math.maxInt(u32) }, .{ 18, 32 }, .{ 100, 920 }, .{ 1000, 24 }, .{ 1022, 2 }, .{ 2008, 2400 }, .{ 32768, 32768 }, .{ 131072, 4096 }, .{ 2008, std.math.maxInt(u32) }, .{ 500000, std.math.maxInt(u32) } };
    for ([_]u32{ 1024, 8192, 65536, 1048576 }) |ctx| {
        cfg.pinned_context = ctx;
        for ([_]u64{ 16, 8, 6 }) |kv_bits| {
            const latent_row: usize = if (kv_bits == 16) @as(usize, cfg.mla_kv_lora_rank) * 2 else @import("glm5/glm5_latent.zig").rowBytes(cfg.mla_kv_lora_rank, @intCast(kv_bits));
            for (shapes) |shape| {
                const seq: usize = @intCast(shape[0]);
                if (seq >= ctx) continue;
                const max_tokens: u32 = @intCast(@min(shape[1], ctx - seq));
                const lc = try reserve.capacity(seq);
                const pc = try reserve.capacity(seq / 4);
                const p = try reserve.plan(seq, lc, pc, latent_row, cfg.indexer_head_dim, 2, seq + max_tokens + 3);
                const held: u64 = mla_layers * (lc * latent_row + pc * pool_row) + cfg.ssmCheckpointBytes();
                try std.testing.expect(try reserve.statesPeak(mla_layers, p) <= glmDflashReserveBudget(&cfg, seq, max_tokens, kv_bits, held));
            }
        }
    }
}

test "GLM prefill transient bill follows the latent width: kv6 sheds 128 bytes per latent row against kv8" {
    const cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    const seq: u64 = 4096;
    const chunk: u64 = 2048;
    const kv8 = glm5TransientBytes(&cfg, seq, chunk, 8);
    const kv6 = glm5TransientBytes(&cfg, seq, chunk, 6);
    const pending: u64 = glmMlaLayersPending(&cfg, 2);
    try std.testing.expectEqual(seq * 128 + pending * chunk * 128, kv8 - kv6);
}

test "GLM DFlash2 reserve fits the admission bill when max_tokens fills the advertised context" {
    const reserve = @import("glm5/glm5_dflash_reserve.zig");
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    var mla_layers: usize = 0;
    for (0..cfg.num_hidden_layers) |l| mla_layers += @intFromBool(!cfg.isLinearLayer(@intCast(l)));
    const pool_row: usize = @as(usize, cfg.indexer_head_dim) * 2;
    for ([_]struct { bits: u64, ctx: u32 }{ .{ .bits = 8, .ctx = 1048576 }, .{ .bits = 6, .ctx = 1048576 }, .{ .bits = 16, .ctx = 524288 } }) |arm| {
        cfg.pinned_context = arm.ctx;
        const latent_row: usize = if (arm.bits == 16) @as(usize, cfg.mla_kv_lora_rank) * 2 else @import("glm5/glm5_latent.zig").rowBytes(cfg.mla_kv_lora_rank, @intCast(arm.bits));
        for ([_]usize{ 16, 2008, 100000 }) |seq| {
            const max_tokens: u32 = @intCast(arm.ctx - seq);
            const lc = try reserve.capacity(seq);
            const pc = try reserve.capacity(seq / 4);
            const p = try reserve.plan(seq, lc, pc, latent_row, cfg.indexer_head_dim, 2, seq + max_tokens + 3);
            const held: u64 = mla_layers * (lc * latent_row + pc * pool_row) + cfg.ssmCheckpointBytes();
            const peak = try reserve.statesPeak(mla_layers, p);
            // The generator's config copy is never pinned: it resolves its own, smaller context.
            var unpinned = cfg;
            unpinned.pinned_context = arm.ctx / 4;
            try std.testing.expect(peak <= glmDflashReserveBudget(&unpinned, seq, max_tokens, arm.bits, held));
        }
    }
}

test "GLM raw FP8 scratch bills all pending projections only beyond direct decode" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    try std.testing.expectEqual(@as(u64, 0), glmFp8DequantScratchBytes(&cfg, 2048, 2));
    cfg.glm_fp8_trunk = true;
    try std.testing.expectEqual(@as(u64, 0), glmFp8DequantScratchBytes(&cfg, @intCast(fp8_block.gemv_max_rows), 2));
    try std.testing.expectEqual(@as(u64, 576 * 1024 * 1024), glmFp8DequantScratchBytes(&cfg, 2048, 2));
    try std.testing.expectEqual(glmFp8DequantScratchBytes(&cfg, 17, 1) * 2, glmFp8DequantScratchBytes(&cfg, 2048, 2));
}

test "GLM raw FP8 honors the declared wired limit with the same reserve as MiMo" {
    const prior = wired_limit_mb_override;
    wired_limit_mb_override = 120000;
    defer wired_limit_mb_override = prior;
    const cfg = model_mod.ModelConfig{ .model_type = "glm5_next", .glm_fp8_trunk = true };
    try std.testing.expectEqual(wiredLimitFloor(120000 * 1024 * 1024, 128 << 30, wired_limit_margin_bytes), wiredCeilingFloorForRam(&cfg, 128 << 30));
    wired_limit_mb_override = 98304;
    try std.testing.expectEqual(@as(u64, 0), wiredCeilingFloorForRam(&cfg, 128 << 30));
}

test "GLM admission bills the checkpoints and assistant window a request keeps; the context sizer reserves none of them" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    cfg.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    cfg.dflash_capture_bytes_per_token = (5 * 4096 + 2 * 4096) * 2 + 5 * 8 * 128 * 4;
    const saved = .{ prefix_cache_capacity, ssm_checkpoint_stride, ssm_checkpoint_max };
    const tiers = TierGuard.pin(true, 0);
    defer tiers.restore();
    defer {
        prefix_cache_capacity = saved[0];
        ssm_checkpoint_stride = saved[1];
        ssm_checkpoint_max = saved[2];
    }
    ssm_checkpoint_stride = 256;
    ssm_checkpoint_max = 16;
    const per_cp = cfg.ssmCheckpointBytes();
    try std.testing.expectEqual(@as(u64, 147_619_840), per_cp);
    prefix_cache_capacity = 0;
    const off = prefillNeededAtChunk(&cfg, 65536, 1024, 8, 2048, .{ .checkpoints = 8 });
    const sizer_off = prefillTransientReserve(&cfg, 8, 2048);
    const ctx_off = memoryContextAtChunk(&cfg, 8, 120 << 30, 100 << 30, 2048);
    const cap = glm5_prefix.checkpointMax(16);
    try std.testing.expectEqual(@as(u64, 0), glmPrefixStateBytes(&cfg, 65536, .{ .checkpoints = cap }));
    prefix_cache_capacity = 32;
    // Every 8192-token grid point and the prompt end, thinned to the GLM cap; the newest
    // copy exists before each thin.
    const held = @as(u64, glmPrefillSchedule(65536, 0, cap).peak) * per_cp + cfg.dflash_window_bytes;
    try std.testing.expectEqual(held, glmPrefixStateBytes(&cfg, 65536, .{ .checkpoints = cap }));
    const on = prefillNeededAtChunk(&cfg, 65536, 1024, 8, 2048, .{ .checkpoints = cap });
    try std.testing.expect(on - off >= held * 5 / 4 - 1 and on - off <= held * 5 / 4 + 1);
    // Fewer checkpoints hold fewer copies; none hold no window either.
    try std.testing.expectEqual(2 * per_cp + cfg.dflash_window_bytes, glmPrefixStateBytes(&cfg, 65536, .{ .checkpoints = 1 }));
    try std.testing.expectEqual(off, prefillNeededAtChunk(&cfg, 65536, 1024, 8, 2048, .{}));
    // The advertised context and the RAM tier's clamp are sized as with the cache off.
    try std.testing.expectEqual(sizer_off, prefillTransientReserve(&cfg, 8, 2048));
    try std.testing.expectEqual(@as(u64, 0), ctxSizingCacheReserve(&cfg));
    try std.testing.expectEqual(ctx_off, memoryContextAtChunk(&cfg, 8, 120 << 30, 100 << 30, 2048));
    // A warm restore at 61440 captures only over its own span: the prompt-end checkpoint (the grid point 65536 is past the loop's end).
    try std.testing.expectEqual(per_cp, retainedSsmCheckpointBytes(&cfg, 65536, 61440, 2048));
    try std.testing.expect(prefillNeededAtChunk(&cfg, 65536, 1024, 8, 2048, .{ .matched_tokens = 61440, .checkpoints = cap }) < on);
}

test "GLM billed checkpoint peak equals the checkpoints its prefill captures, whatever stride is configured" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.dflash_loaded = false;
    const saved = .{ prefix_cache_capacity, ssm_checkpoint_stride, prefix_cache_disk_bytes, prefix_cache_ram_enabled };
    defer {
        prefix_cache_capacity = saved[0];
        ssm_checkpoint_stride = saved[1];
        prefix_cache_disk_bytes = saved[2];
        prefix_cache_ram_enabled = saved[3];
    }
    prefix_cache_capacity = 32;
    prefix_cache_disk_bytes = 0;
    prefix_cache_ram_enabled = true;
    const per_cp = cfg.ssmCheckpointBytes();
    const Case = struct { seq: u64, matched: u64 };
    const cases = [_]Case{
        .{ .seq = 16, .matched = 0 },       .{ .seq = 31, .matched = 0 },          .{ .seq = 40, .matched = 0 },
        .{ .seq = 2049, .matched = 0 },     .{ .seq = 5000, .matched = 0 },        .{ .seq = 18432, .matched = 0 },
        .{ .seq = 70000, .matched = 0 },    .{ .seq = 65536, .matched = 61440 },   .{ .seq = 18432, .matched = 16384 },
        .{ .seq = 9000, .matched = 8188 },  .{ .seq = 9100, .matched = 8188 },
    };
    for ([_]u32{ 256, 2048, 8192, 65536 }) |stride| {
        ssm_checkpoint_stride = stride;
        for (cases) |c| for (0..9) |cap_usize| {
            const cap: u32 = @intCast(cap_usize);
            const oracle = glm5_generate.glmCaptureOracle(generate_mod.glm_checkpoint_stride, generate_mod.glm_capture_stride, cap, c.matched, c.seq - c.matched, 2048);
            const billed = glmPrefixStateBytes(&cfg, c.seq, .{ .matched_tokens = c.matched, .checkpoints = cap });
            const want: u64 = if (cap == 0) 0 else @as(u64, oracle.peak) * per_cp;
            std.testing.expectEqual(want, billed) catch |e| {
                std.debug.print("stride {d} seq {d} matched {d} cap {d}: billed {d} cps, captured {d}\n", .{ stride, c.seq, c.matched, cap, billed / per_cp, want / per_cp });
                return e;
            };
            // A prefill that releases landed states holds the newest two, or all it captures if fewer.
            const released = glmPrefixStateBytes(&cfg, c.seq, .{ .matched_tokens = c.matched, .checkpoints = cap, .released = true });
            try std.testing.expectEqual(@min(want, 2 * per_cp), released);
        };
    }
}

test "GLM admission bills the commit's row copy, so a long generation cannot commit above the ceiling, and the copy never costs a long prompt its checkpoints" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    cfg.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    const saved = .{ prefix_cache_capacity, ssm_checkpoint_stride, prefix_cache_disk_bytes, prefix_cache_ram_enabled, prefix_cache_mem_bytes };
    defer {
        prefix_cache_capacity = saved[0];
        ssm_checkpoint_stride = saved[1];
        prefix_cache_disk_bytes = saved[2];
        prefix_cache_ram_enabled = saved[3];
        prefix_cache_mem_bytes = saved[4];
    }
    ssm_checkpoint_stride = 256;
    prefix_cache_capacity = 32;
    prefix_cache_mem_bytes = GLM_PREFIX_CACHE_MEM_DEFAULT;
    const per_cp = cfg.ssmCheckpointBytes();
    const row_bytes = sessionBytesPerToken(&cfg, 8);
    const cap = glm5_prefix.checkpointMax(16);
    const Arm = struct { ram: bool, disk: u64 };
    for ([_]Arm{ .{ .ram = true, .disk = 0 }, .{ .ram = false, .disk = 12 << 30 }, .{ .ram = true, .disk = 12 << 30 } }) |arm| {
        const ram = arm.ram;
        prefix_cache_ram_enabled = ram;
        prefix_cache_disk_bytes = arm.disk;
        const staged: u64 = 0;
        for ([_]u64{ 64, 131_072, 900_000, 1_040_000 }) |seq| {
            const max_tokens: u32 = @intCast(getEffectiveContextLength(&cfg) - seq);
            const oracle = glm5_generate.glmCaptureOracle(generate_mod.glm_checkpoint_stride, generate_mod.glm_capture_stride, cap, 0, seq, 2048);
            // Eight spread checkpoints below the newest, each at its real size.
            var positions: [8]usize = undefined;
            const bytes: [8]u64 = @splat(per_cp);
            for (&positions, 1..) |*p, i| p.* = if (i == 8) oracle.newest else @max(oracle.newest * i / 8 / 2048 * 2048, 4);
            const kept: usize = @min(@as(usize, oracle.events), 8);
            // What the commit copies: the RAM tier's rows within its budget, chosen before the copy.
            // The SSD tier copies none: its flush reads the request's own rows.
            const copied_len = if (!ram) 0 else prefix_cache_mod.HotPrefixCache.glmCommitLenFor(
                .min_span_recency,
                resolvedPrefixCacheMem(),
                false,
                oracle.newest,
                row_bytes,
                cfg.dflash_window_bytes,
                positions[8 - kept ..],
                bytes[8 - kept ..],
            ) orelse 0;
            const commit_peak = glmCacheBytes(&cfg, seq, max_tokens, 8) + copied_len * row_bytes + @as(u64, @intCast(kept)) * per_cp + cfg.dflash_window_bytes + staged;
            const needed = prefillNeededAtChunk(&cfg, seq, max_tokens, 8, 2048, .{ .checkpoints = cap });
            std.testing.expect(needed >= commit_peak) catch |e| {
                std.debug.print("ram={any} seq {d}: admitted {d} bytes, commit peaks at {d}\n", .{ ram, seq, needed, commit_peak });
                return e;
            };
            // A long prompt's checkpoints are priced by its prefill alone: the copy is smaller than its transient.
            const warm = WarmPrefix{ .checkpoints = cap };
            try std.testing.expect(glmCommitStateBytes(&cfg, seq, warm, 8) <= glm5TransientBytes(&cfg, seq, 2048, 8) +| glmPrefixStateBytes(&cfg, seq, warm));
            // SSD-only: checkpoints, window and the writer's staged bytes, no rows at any length.
            if (!ram) try std.testing.expectEqual(@as(u64, @min(oracle.events, cap)) * per_cp + cfg.dflash_window_bytes + staged, glmCommitStateBytes(&cfg, seq, warm, 8));
            // Both tiers: the RAM tier's copy; the disk tier adds no term of its own here.
            if (ram and arm.disk > 0) {
                prefix_cache_disk_bytes = 0;
                const ram_only = glmCommitStateBytes(&cfg, seq, warm, 8);
                prefix_cache_disk_bytes = arm.disk;
                try std.testing.expectEqual(ram_only, glmCommitStateBytes(&cfg, seq, warm, 8));
            }
        }
    }
}

test "GLM SSD-only bills its checkpoints and window, no writer term (the shared admission headroom has it), and sizes no cache" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    cfg.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    const saved = .{ prefix_cache_capacity, ssm_checkpoint_stride, ssm_checkpoint_max, prefix_cache_disk_bytes, prefix_cache_ram_enabled };
    defer {
        prefix_cache_capacity = saved[0];
        ssm_checkpoint_stride = saved[1];
        ssm_checkpoint_max = saved[2];
        prefix_cache_disk_bytes = saved[3];
        prefix_cache_ram_enabled = saved[4];
    }
    ssm_checkpoint_stride = 256;
    ssm_checkpoint_max = 16;
    prefix_cache_capacity = 0;
    const sizer_off = prefillTransientReserve(&cfg, 8, 2048);
    prefix_cache_capacity = 32;
    prefix_cache_disk_bytes = 12 << 30;
    prefix_cache_ram_enabled = false;
    const held = @as(u64, glmPrefillSchedule(65536, 0, glm5_prefix.checkpointMax(16)).peak) * cfg.ssmCheckpointBytes() + cfg.dflash_window_bytes;
    try std.testing.expectEqual(held, glmPrefixStateBytes(&cfg, 65536, .{ .checkpoints = glm5_prefix.checkpointMax(16) }));
    try std.testing.expectEqual(@as(u64, 0), glmPrefixStateBytes(&cfg, 65536, .{}));
    try std.testing.expectEqual(@as(u64, 0), ctxSizingCacheReserve(&cfg));
    try std.testing.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
    try std.testing.expectEqual(sizer_off, prefillTransientReserve(&cfg, 8, 2048));
}

test "GLM bills two KDA states, not eight, while its prefill moves each to SSD and frees the RAM copy" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    cfg.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    const saved = .{ prefix_cache_capacity, ssm_checkpoint_stride, ssm_checkpoint_max, prefix_cache_disk_bytes, prefix_cache_ram_enabled };
    defer {
        prefix_cache_capacity = saved[0];
        ssm_checkpoint_stride = saved[1];
        ssm_checkpoint_max = saved[2];
        prefix_cache_disk_bytes = saved[3];
        prefix_cache_ram_enabled = saved[4];
    }
    ssm_checkpoint_stride = 256;
    ssm_checkpoint_max = 16;
    prefix_cache_capacity = 32;
    prefix_cache_disk_bytes = 12 << 30;
    prefix_cache_ram_enabled = false;
    const per_cp = cfg.ssmCheckpointBytes();
    const window = cfg.dflash_window_bytes;
    const cap = glm5_prefix.checkpointMax(16);
    const keeps = WarmPrefix{ .checkpoints = cap };
    const releases = WarmPrefix{ .checkpoints = cap, .released = true };
    // 65,536 tokens capture 7 grid points and the prompt end: eight held before, two now.
    try std.testing.expectEqual(8 * per_cp + window, glmPrefixStateBytes(&cfg, 65536, keeps));
    try std.testing.expectEqual(2 * per_cp + window, glmPrefixStateBytes(&cfg, 65536, releases));
    // A prompt that captures one checkpoint holds one either way.
    try std.testing.expectEqual(per_cp + window, glmPrefixStateBytes(&cfg, 100, releases));
    // The commit after the response: the checkpoints it lists, its two newest still in RAM.
    try std.testing.expectEqual(8 * per_cp + window, glmCommitStateBytes(&cfg, 65536, keeps, 8));
    try std.testing.expectEqual(2 * per_cp + window, glmCommitStateBytes(&cfg, 65536, releases, 8));
    // Admission follows the storage: the difference is the six states that are no longer held (a 5/4 margin rides the checkpoints).
    const kept_bill = prefillNeededAtChunk(&cfg, 65536, 1024, 8, 2048, keeps);
    const released_bill = prefillNeededAtChunk(&cfg, 65536, 1024, 8, 2048, releases);
    try std.testing.expect(kept_bill - released_bill >= 6 * per_cp * 5 / 4 - 1 and kept_bill - released_bill <= 6 * per_cp * 5 / 4 + 1);
}

test "GLM near the top of its context keeps fewer checkpoints at the start width instead of being refused" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    cfg.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    const saved = .{ prefix_cache_capacity, ssm_checkpoint_stride, ssm_checkpoint_max };
    defer {
        prefix_cache_capacity = saved[0];
        ssm_checkpoint_stride = saved[1];
        ssm_checkpoint_max = saved[2];
    }
    ssm_checkpoint_stride = 256;
    ssm_checkpoint_max = 16;
    prefix_cache_capacity = 32;
    const seq: u64 = 1_040_000;
    const width: u64 = rungWidth(&cfg, seq, cfg.prefillStartWidth(), false);
    // Free memory admits the prompt with its prompt-end checkpoint and one grid copy, no more.
    const Fits = struct {
        cfg: *const model_mod.ModelConfig,
        seq: u64,
        width: u64,
        available: u64,
        fn at(ctx: ?*anyopaque, n: u32) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            return prefillNeededAtChunk(self.cfg, self.seq, 2048, 8, self.width, .{ .checkpoints = n }) <= self.available;
        }
    };
    var probe = Fits{ .cfg = &cfg, .seq = seq, .width = width, .available = prefillNeededAtChunk(&cfg, seq, 2048, 8, width, .{ .checkpoints = 1 }) };
    const cap = glm5_prefix.checkpointMax(16);
    try std.testing.expect(prefillNeededAtChunk(&cfg, seq, 2048, 8, width, .{ .checkpoints = cap }) > probe.available);
    try std.testing.expectEqual(@as(?u32, 1), scheduler_mod.fewerCheckpointsToAdmit(cap, &probe, Fits.at));
    // The checkpoints yield to the width: the prefill still starts at the start width.
    try std.testing.expectEqual(@as(u32, @intCast(width)), chooseRequestPrefillChunk(&cfg, seq, 2048, 8, probe.available, cfg.pinned_prefill_chunk, 0, .{ .checkpoints = cap }));
    // Room for the prompt alone keeps none; less than that is refused for the prompt, not the hold.
    probe.available = prefillNeededAtChunk(&cfg, seq, 2048, 8, width, .{ .checkpoints = 0 });
    try std.testing.expectEqual(@as(?u32, 0), scheduler_mod.fewerCheckpointsToAdmit(cap, &probe, Fits.at));
    probe.available -= 1;
    try std.testing.expectEqual(@as(?u32, null), scheduler_mod.fewerCheckpointsToAdmit(cap, &probe, Fits.at));
}

test "GLM admission credits the hot cache, so an eviction lets a long prefill in" {
    const cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    const gb: u64 = 1 << 30;
    const bill = creditedAdmissionBill(&cfg, 20 * gb, 19 * gb, 3 * gb / 2, 3 * gb / 2, 2048);
    try std.testing.expectEqual(AdmissionVerdict.evict, admissionVerdict(bill));
    try std.testing.expectEqual(AdmissionVerdict.refuse, admissionVerdict(creditedAdmissionBill(&cfg, 21 * gb, 19 * gb, 3 * gb / 2, 3 * gb / 2, 2048)));
}

/// Plays the inference thread's decode tick: publishes `n` tokens, then accounts them one by one
/// with `gap_ms` between steps, all inside one `in_pass`.
pub const TickPlayer = struct {
    slot: *scheduler_mod.Slot,
    n: u32,
    gap_ms: i64,

    pub fn run(self: *TickPlayer) void {
        const io = self.slot.io;
        var i: u32 = 0;
        while (i < self.n) : (i += 1) self.slot.pushTokenWithLogprob(i + 1, null);
        std.Io.sleep(io, .fromMilliseconds(self.gap_ms), .real) catch {};
        i = 0;
        while (i < self.n) : (i += 1) {
            self.slot.completion_tokens += 1;
            std.Io.sleep(io, .fromMilliseconds(self.gap_ms), .real) catch {};
        }
        self.slot.decode_ns +|= 777;
        _ = self.slot.in_pass.fetchSub(1, .acq_rel);
    }
};

pub const FakeSlot = struct {
    tokens: []const u32,
    next: usize = 0,
    cancelled: bool = false,
    stop_hit: bool = false,

    pub fn waitNextTimeout(self: *FakeSlot, _: i64) ?scheduler_mod.NextResult {
        if (self.cancelled or self.next == self.tokens.len) return .{ .done = {} };
        defer self.next += 1;
        return .{ .token = self.tokens[self.next] };
    }

    pub fn cancel(self: *FakeSlot) void {
        self.cancelled = true;
    }

    pub fn cancelOnStop(self: *FakeSlot) void {
        self.stop_hit = true;
        self.cancel();
    }
};

test "countRejected moves only the rejected counter, and does nothing with metrics off" {
    instr.process = .{};
    defer instr.process = .{};
    countRejected();

    _ = instr.process.enable();
    const m = instr.current().?;
    countRejected();
    try std.testing.expectEqual(@as(u64, 1), m.requests_rejected_total.load());
    try std.testing.expectEqual(@as(u64, 0), m.requests_failed_total.load());
    try std.testing.expectEqual(@as(u64, 0), m.requests_cancelled_total.load());
}

test "dsv41 bills: a prefill holds its restore points beside the state with a RAM tier, and only the writer's in-flight bytes SSD-first (DSV41_TINY)" {
    try mlx.requireGpu();
    const dir = try @import("dsv41/dsv41_model.zig").fixtureDir();
    const t = std.testing;
    const io = std.Io.Threaded.global_single_threaded.io();
    var cfg = try model_mod.parseConfig(io, t.allocator, dir);
    defer cfg.deinit(t.allocator);
    const saved = .{ prefix_cache_ram_enabled, ssm_checkpoint_stride, ssm_checkpoint_max, prefix_cache_disk_bytes, prefix_cache_capacity };
    defer {
        prefix_cache_ram_enabled = saved[0];
        ssm_checkpoint_stride = saved[1];
        ssm_checkpoint_max = saved[2];
        prefix_cache_disk_bytes = saved[3];
        prefix_cache_capacity = saved[4];
    }
    ssm_checkpoint_stride = 256;
    ssm_checkpoint_max = 16;
    prefix_cache_capacity = 32;
    const c = cfg.dsv41.?;
    const seq: u64 = 20_000;
    const one = c.stateFixedBytes(cfg.dsparkOn()) + seq * c.stateBytesPerToken();
    prefix_cache_ram_enabled = false;
    prefix_cache_disk_bytes = 0;
    const off = prefillNeededAtChunk(&cfg, seq, 0, 8, 512, .{});
    try t.expectEqual(@as(u64, 0), dsv41CheckpointReserve(&cfg, seq, 512));
    for ([_][2]bool{ .{ true, false }, .{ true, true } }) |tiers| {
        prefix_cache_ram_enabled = tiers[0];
        prefix_cache_disk_bytes = if (tiers[1]) 20 << 30 else 0;
        // One checkpoint per 8192-token stride and the prompt end, plus the copy taken before a thin.
        try t.expectEqual(@as(u64, 4) * one, dsv41CheckpointReserve(&cfg, seq, 512));
        try t.expectEqual(off + 4 * one * 5 / 4, prefillNeededAtChunk(&cfg, seq, 0, 8, 512, .{}));
        // The list is capped, whatever the prompt.
        try t.expectEqual(@as(u64, 5) * (c.stateFixedBytes(cfg.dsparkOn()) + 1_000_000 * c.stateBytesPerToken()), dsv41CheckpointReserve(&cfg, 1_000_000, 512));
    }
    // SSD-first: each point goes to the tier as it is taken, so the bill is the writer's bounded host buffer and one
    // point's fixed rings, whatever the prompt; under the tier's floor the points wait for the commit in RAM.
    prefix_cache_ram_enabled = false;
    prefix_cache_disk_bytes = 20 << 30;
    const rings = c.stateFixedBytes(cfg.dsparkOn());
    // The rollback points the next turn keeps in RAM are billed with it: each is one set of fixed rings.
    const kept = @as(u64, dsv41_prefix.live_point_cap) * rings;
    try t.expectEqual(one + rings + kept, dsv41CheckpointReserve(&cfg, seq, 512));
    const big: u64 = 1 << 40;
    try t.expectEqual(@min(disk_writer_permit_bytes, rings + big * c.stateBytesPerToken()) + rings + kept, dsv41CheckpointReserve(&cfg, big, 512));
    try t.expect(dsv41CheckpointReserve(&cfg, 100, 512) == 2 * (rings + 100 * c.stateBytesPerToken()));
    // An entry on disk keeps its rows once and the fixed rings of one point per grid step.
    try t.expectEqual(c.stateBytesPerToken() + (rings + dsv41_prefix.row_span - 1) / dsv41_prefix.row_span, diskBytesPerToken(&cfg, 8));
    ssm_checkpoint_stride = 0;
    try t.expectEqual(@as(u64, 0), dsv41CheckpointReserve(&cfg, seq, 512));
}

pub fn readAllFromPair(fd: std.posix.fd_t, buf: []u8) []u8 {
    var got: usize = 0;
    while (got < buf.len) {
        const n = std.c.recv(fd, buf[got..].ptr, buf.len - got, 0);
        if (n <= 0) break;
        got += @intCast(n);
    }
    return buf[0..got];
}

test "auto context on a room of about 1.8 GB neither advertises a few tokens nor lets the cache reserve eat it" {
    const GiB: u64 = 1 << 30;
    const per_tok: u64 = 278528;
    const room: u64 = GiB * 18 / 10;
    // The 2 GiB cache reserve alone exceeds the room: the context, not the cache, gets the half.
    const share = ctxCacheShare(room + 5 * GiB, 5 * GiB, 2 * GiB);
    try testing.expectEqual(room / 2, share);
    try testing.expectEqual(@as(u64, GiB), ctxCacheShare(64 * GiB, 5 * GiB, GiB));
    const auto = resolvedContextForLoad(0, 0, room + 5 * GiB, 5 * GiB, share, 0, per_tok, 0);
    try testing.expectEqual(@as(u32, 2048), auto);
    // A room that fits a few tokens past the reserve floors at 1024 instead of advertising them.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(2 * GiB + 100 * (1 << 20), 0, 2 * GiB, per_tok, 0));
}

test "a thought point is one more checkpoint state in the bill: every Qwen request, a GLM request that keeps checkpoints" {
    const t = std.testing;
    const saved = .{ prefix_cache_capacity, ssm_checkpoint_stride, ssm_checkpoint_max, prefix_cache_disk_bytes, prefix_cache_ram_enabled };
    defer {
        prefix_cache_capacity = saved[0];
        ssm_checkpoint_stride = saved[1];
        ssm_checkpoint_max = saved[2];
        prefix_cache_disk_bytes = saved[3];
        prefix_cache_ram_enabled = saved[4];
    }
    ssm_checkpoint_stride = 256;
    ssm_checkpoint_max = 16;
    prefix_cache_capacity = 32;
    prefix_cache_disk_bytes = 12 << 30;
    prefix_cache_ram_enabled = false;

    var qwen = try model_mod.parseConfigFromJson(t.allocator, @embedFile("fixtures/model-configs/qwen4_exp.json"));
    defer qwen.deinit(t.allocator);
    qwen.expert_layout = .exl3_k4;
    const q_per = qwen.ssmCheckpointBytes();
    try t.expect(q_per > 0);
    try t.expectEqual(q_per, qwenThoughtPointBytes(&qwen, .{ .thought = true }));
    try t.expectEqual(@as(u64, 0), qwenThoughtPointBytes(&qwen, .{}));
    const q_terms = prefillRequestTerms(&qwen, 4096, 1024, 8, 512, .{ .thought = true });
    try t.expectEqual(retainedSsmCheckpointBytes(&qwen, 4096, 0, 512) + q_per, q_terms.checkpoint_bytes);
    try t.expectEqual(retainedSsmCheckpointBytes(&qwen, 4096, 0, 512), prefillRequestTerms(&qwen, 4096, 1024, 8, 512, .{}).checkpoint_bytes);
    ssm_checkpoint_stride = 0;
    try t.expectEqual(@as(u64, 0), qwenThoughtPointBytes(&qwen, .{ .thought = true }));
    ssm_checkpoint_stride = 256;

    var glm = try model_mod.parseConfigFromJson(t.allocator, @embedFile("fixtures/glm5_config.json"));
    defer glm.deinit(t.allocator);
    glm.dflash_loaded = true;
    const g_per = glm.ssmCheckpointBytes();
    try t.expectEqual(@as(u64, 147_619_840), g_per);
    try t.expectEqual(g_per, glmThoughtPointBytes(&glm, .{ .checkpoints = 8, .thought = true }));
    try t.expectEqual(g_per, glmThoughtPointBytes(&glm, .{ .checkpoints = 1, .released = true, .thought = true }));
    // A request that takes no thought point, keeps no checkpoints, or has no tier to take one bills nothing.
    try t.expectEqual(@as(u64, 0), glmThoughtPointBytes(&glm, .{ .checkpoints = 8 }));
    try t.expectEqual(@as(u64, 0), glmThoughtPointBytes(&glm, .{ .thought = true }));
    prefix_cache_capacity = 0;
    prefix_cache_disk_bytes = 0;
    try t.expectEqual(@as(u64, 0), glmThoughtPointBytes(&glm, .{ .checkpoints = 8, .thought = true }));
}
