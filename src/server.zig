const std = @import("std");
const sched_policy = @import("sched_policy.zig");
const settings_resolve = @import("settings_resolve.zig");
const mlx = @import("mlx.zig");
const trunk = @import("trunk.zig");
const verify_moe = @import("verify_moe.zig");
const attn_kernels = @import("attn_kernels.zig");
const nax_gate = @import("nax_gate.zig");
const tokenizer_mod = @import("tokenizer.zig");
const generate_mod = @import("generate.zig");
const mtp_mod = @import("mtp.zig");
const mtp_acceptance_mod = @import("mtp_acceptance.zig");
const chat_mod = @import("chat.zig");
const think_tags = @import("think_tags.zig");
const tool_calls_mod = @import("tool_calls.zig");
const tool_schema = @import("tool_schema.zig");
const stop_seq_mod = @import("stop_sequences.zig");
const rp_mod = @import("reasoning_protocol.zig");
const token_mask = @import("token_mask.zig");
const expert_stream_mod = @import("expert_stream.zig");
const fp8_block = @import("fp8_block.zig");
const sushi_exl3 = @import("sushi_exl3");
const model_mod = @import("model.zig");
const effort_mod = @import("effort.zig");
const vision_mod = @import("vision.zig");
const log = @import("log.zig");
const server_state = @import("server_state.zig");
const http = @import("http.zig");
const request_parse = @import("request_parse.zig");
const arch_flags = @import("arch/flags.zig");
const arch_serve = @import("arch/serve.zig");
const responses_mod = @import("responses.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const kv_disk_writer = @import("kv_disk_writer.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const disk_placement = @import("disk_placement.zig");
const tokenize_cache_mod = @import("tokenize_cache.zig");
const scheduler_mod = @import("scheduler.zig");
const model_load = @import("model_load.zig");
const slot_admission = @import("slot_admission.zig");
const vision_queue = @import("vision_queue.zig");
const model_registry_mod = @import("model_registry.zig");
const model_discovery = @import("model_discovery.zig");
const model_settings = @import("model_settings.zig");
const stb = @import("stb");
const webp = @import("webp");
const metrics = @import("status.zig");
const instr = @import("metrics.zig");
const update_mod = @import("update.zig");

const Trunk = trunk.Trunk;
const Tokenizer = tokenizer_mod.Tokenizer;
const Generator = generate_mod.Generator;
const VisionEncoder = vision_mod.VisionEncoder;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;
const anthropic = @import("anthropic.zig");
const completions = @import("completions.zig");
const responses_api = @import("responses_api.zig");
const generation_stream = @import("generation_stream.zig");
const http_conn = @import("http_conn.zig");
const boot_gate = @import("boot_gate.zig");
const request_media = @import("request_media.zig");
const arch_media = @import("arch/media.zig");
const request_policy = @import("request_policy.zig");
const models_api = @import("models_api.zig");
const prefix_cache_admin = @import("prefix_cache_admin.zig");
const downloads = @import("downloads.zig");
const metrics_history = @import("metrics_history.zig");
const gpu_clock_live = @import("gpu_clock_live.zig");
const json_text = @import("json_text.zig");
const settings_api = @import("settings_api.zig");
const logprobs = @import("logprobs.zig");
const bills = @import("bills.zig");
const fixtures = @import("bills_fixtures_test.zig");
const gpu_mem = @import("gpu_mem.zig");

// Names other files import from here; the bill itself lives in bills.zig.
pub const forceMtpFor = bills.forceMtpFor;
pub const mtpChoiceFor = bills.mtpChoiceFor;
pub const kvCacheFor = bills.kvCacheFor;
pub const kvCacheForLaunch = bills.kvCacheForLaunch;
const configuredKvQuantFor = bills.configuredKvQuantFor;
const manualContext = bills.manualContext;
pub const resolvedPrefixCacheMem = bills.resolvedPrefixCacheMem;
const clearResolvedPrefixCacheMem = bills.clearResolvedPrefixCacheMem;
pub const diskAsk = bills.diskAsk;
pub const setDiskAsk = server_state.setDiskAsk;
const diskTierOn = bills.diskTierOn;
pub const effectiveSsmCheckpointStride = bills.effectiveSsmCheckpointStride;
pub const getEffectiveContextLength = bills.getEffectiveContextLength;
pub const parseWiredMarginGib = bills.parseWiredMarginGib;
pub const parseSsdBudgetGb = bills.parseSsdBudgetGb;
const ssdBudgetAdmits = bills.ssdBudgetAdmits;
const warnUserWireLimit = bills.warnUserWireLimit;
const mlxCacheLimitBytes = bills.mlxCacheLimitBytes;
pub const currentGpuMemoryCeiling = bills.currentGpuMemoryCeiling;
pub const expertCacheFitForLoad = bills.expertCacheFitForLoad;
pub const rowSsdBudgetGb = bills.rowSsdBudgetGb;
pub const memoryContextRefusalMessage = bills.memoryContextRefusalMessage;
const loadServingBill = bills.loadServingBill;
pub const prefixCacheMemForLoad = bills.prefixCacheMemForLoad;
pub const prefixCacheDiskForLoad = bills.prefixCacheDiskForLoad;
pub const computeMemoryContext = bills.computeMemoryContext;
pub const glmDflashReserveBudget = bills.glmDflashReserveBudget;
pub const prefillRequestTerms = bills.prefillRequestTerms;
const AdmissionBill = bills.AdmissionBill;
const fitAfterUnloadingIdle = bills.fitAfterUnloadingIdle;
const prefillNeededAtChunk = bills.prefillNeededAtChunk;
const requestPrefillChunkNow = bills.requestPrefillChunkNow;
const adaptivePrefillChunkEnabled = bills.adaptivePrefillChunkEnabled;
const adaptivePrefillWidenStillFits = bills.adaptivePrefillWidenStillFits;
const adaptivePrefillWidthNow = bills.adaptivePrefillWidthNow;
const prefillAdmissionBill = bills.prefillAdmissionBill;
const pinnedResidentBytes = bills.pinnedResidentBytes;
const prefillFitsNow = bills.prefillFitsNow;
const prefillBillNumbersNow = bills.prefillBillNumbersNow;
const logPrefillRefusal = bills.logPrefillRefusal;
pub const resolveMlxCadence = gpu_mem.resolveMlxCadence;
pub const applyMlxCacheLimit = gpu_mem.applyMlxCacheLimit;
pub const capMlxPool = gpu_mem.capMlxPool;

// The transport and the request readers live in http.zig and request_parse.zig; the handlers here call them by name.
const startupRefusal = http.startupRefusal;
const sendResponse = http.sendResponse;
const sendResponseQuiet = http.sendResponseQuiet;
const logHttpRequest = http.logHttpRequest;
const peerIsLoopback = http.peerIsLoopback;
const responseBody = http.responseBody;
const not_enough_memory_message = http.not_enough_memory_message;
const insufficient_free_memory_message = http.insufficient_free_memory_message;
const countRejected = http.countRejected;
const sendErrorResponse = http.sendErrorResponse;
pub const jsonEscape = http.jsonEscape;
const parseJsonFloatOpt = request_parse.parseJsonFloatOpt;
const parseJsonTopKOpt = request_parse.parseJsonTopKOpt;
const parseModelFromBody = request_parse.parseModelFromBody;
const queryModel = request_parse.queryModel;
const messageReasoningFromObj = request_parse.messageReasoningFromObj;
const parseRequestSeed = request_parse.parseRequestSeed;
const MAX_TOKENS_DESC_LEN = request_parse.MAX_TOKENS_DESC_LEN;
const chatIgnoreEosRejectReason = request_parse.chatIgnoreEosRejectReason;
const parseCancelRequestId = request_parse.parseCancelRequestId;
const joinedTextParts = request_parse.joinedTextParts;
pub const requestCacheKey = request_parse.requestCacheKey;
const nChoicesRejectReason = request_parse.nChoicesRejectReason;
const extractJsonField = request_parse.extractJsonField;
const chatTemplateKwargBool = request_parse.chatTemplateKwargBool;
const parseKvQuantOverride = request_parse.parseKvQuantOverride;
const unescapeJsonSlashes = request_parse.unescapeJsonSlashes;
/// Number of live per-connection threads. Incremented before each spawn,
/// decremented when the thread's handler returns. On shutdown, `serve` waits
/// for this to reach 0 before returning (which runs `scheduler.deinit`) — a
/// detached conn thread still in `Scheduler.complete` would otherwise race
/// deinit's teardown of the slot queues into a use-after-free (SIGSEGV).
var active_conn_threads = std.atomic.Value(u32).init(0);






const io_util = @import("io_util.zig");
const ws_mod = @import("ws.zig");
const build_options = @import("build_options");
const mtp_plan = @import("mtp_plan.zig");
const prefill_plan = @import("prefill_plan.zig");
const sampler = @import("sampler.zig");
const think_penalty = @import("think_penalty.zig");

fn signalHandler(_: std.posix.SIG) callconv(.c) void {
    server_state.shutdown_requested.store(true, .release);
}

// Plan 05: drafter state moved to `LoadedModel` (per-model `drafter`,
// `drafter_block_size`, `drafter_path`). The previous module-level
// `default_drafter` / `default_enable_drafter` / `default_draft_block_size`
// / `global_drafter_path` singletons were removed; handlers now read
// these fields off the request's resolved `*LoadedModel` (`lm.drafter`,
// etc.). `default_enable_drafter` semantics (`drafter != null and !isMoe()`)
// is re-computed per-request as `lm.drafter != null and !config.isMoe()`.

pub fn requestShutdown() void {
    server_state.shutdown_requested.store(true, .release);
}


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
/// ladder (`Trunk.nativeMoeMtpHeadMeasured`, which carries the bar). It
/// still needs a head LOADED — the claim is about the head, not the arch.


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
    return if (lm.mtp) |head| head.kind() == .dspark else false;
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




pub fn getTimeoutNs() u64 {
    if (server_state.server_config.request_timeout_sec == 0) return 0;
    return @as(u64, server_state.server_config.request_timeout_sec) * std.time.ns_per_s;
}





/// The browser chat page: one self-contained file that talks to this server's own API.
const chat_page_html = @embedFile("webui/index.html");


/// The browser orchestrates calls; this local-only bridge shares the REPL's restrictions.
fn handleWebTools(allocator: std.mem.Allocator, stream: *http_conn.Conn, headers: []const u8, body: []const u8) !void {
    const origin = http_conn.findHeaderValueCI(headers, "origin") orelse "";
    if (!peerIsLoopback(stream) or !update_mod.isLoopbackHost(server_state.listen_host) or
        !update_mod.originMatches(origin, server_state.listen_host, server_state.listen_port))
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
        const json = try std.json.Stringify.valueAlloc(allocator, .{ .tools = defs.value, .root = root, .edit_default = server_state.g_web_edit }, .{});
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
fn sendWebToolFolders(allocator: std.mem.Allocator, stream: *http_conn.Conn, root: []const u8) !void {
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
    return std.mem.eql(u8, path, "/") or std.mem.eql(u8, path, "/chat") or std.mem.eql(u8, path, "/dashboard") or
        std.mem.eql(u8, path, "/models") or std.mem.eql(u8, path, "/settings") or std.mem.eql(u8, path, "/downloads");
}

/// The chat page URL printed at startup. A wildcard bind is shown as loopback,
/// because browsers refuse to open 0.0.0.0.
pub fn chatPageUrl(buf: []u8, host: []const u8, port: u16) []const u8 {
    const shown = if (std.mem.eql(u8, host, "0.0.0.0")) "127.0.0.1" else host;
    return std.fmt.bufPrint(buf, "http://{s}:{d}/", .{ shown, port }) catch "http://127.0.0.1/";
}

pub const max_request_bytes: usize = 64 * 1024 * 1024;







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


// Plan 05: vision encoder and model id moved to `LoadedModel.vision_encoder`
// and `LoadedModel.id`. Handlers read them off `lm`. `global_vision_encoder`
// and `global_model_id` singletons were removed. The `discovered_models`
// slice was also removed — `/v1/models` iterates `registry.entries` directly.


/// True when the rendered generation prompt ends inside a template-opened
/// think block (Qwen 3.5/3.6 render `…assistant\n<think>\n` when thinking is
/// on). Decodes the last few prompt tokens — cheap, engine-agnostic, and
/// independent of the tokenize cache. Drives the unclosed-think split policy
/// in `think_tags.splitThinkBlock` so a length-truncated thought never leaks into
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
    return think_tags.promptTailOpensThink(tail);
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
    const closer = atomicTokenId(allocator, tok, think_tags.BARE_THINK_CLOSER) orelse {
        if (announce) log.info("  reasoning budget {d}: think markers are not atomic tokens, delivery cap only\n", .{budget});
        return null;
    };
    const opener = atomicTokenId(allocator, tok, think_tags.BARE_THINK_OPENER);
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
    choice: tool_calls_mod.ToolChoice,
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
    const closer = atomicTokenId(allocator, tok, think_tags.BARE_THINK_CLOSER);
    const opener = atomicTokenId(allocator, tok, think_tags.BARE_THINK_OPENER);
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
    const declared = try tool_calls_mod.declaredToolNames(arena, tools_json);
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
        const len = think_tags.thinkOpenTagLenAt(template[start..]) orelse return null;
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
        switch (think_tags.promptThinkTailClass(trimmed)) {
            .bare => {
                proto.kind = .bare_think;
                if (!proto.setCloser(think_tags.BARE_THINK_CLOSER)) return false;
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






/// The auto budget's own tightness question: under a quarter of the window left.
pub fn autoBudgetWindowTight(remaining: u32, effective_ctx: u32) bool {
    return remaining < effective_ctx / 4;
}

/// Finish reason for a response whose generated text yielded tool calls.
/// A TRUNCATED generation ("length": max_tokens or the request timeout) must
/// keep reporting "length" even when the cut-off text salvaged a partial tool
/// call (the truncated-opener recovery in tool_calls.parseToolCalls) — clients key
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
/// `tool_calls.parseToolCalls` directly silently reintroduces the value-spelling
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
) !?[]tool_schema.ParsedToolCall {
    var parsed_calls = try tool_calls_mod.parseToolCalls(allocator, text);
    // Heuristically-inferred raw-JSON calls must name a DECLARED tool — a
    // truncated data object ({"name": "George Washington", …}) is not a call.
    // Deliberately NOT gated on g_tool_autocorrect: this corrects our own
    // heuristic's false positive, not the model's output.
    if (parsed_calls) |c| {
        if (tools_json) |tj| parsed_calls = try tool_schema.filterInferredBySchema(allocator, c, tj);
    }
    var calls = parsed_calls orelse
        (if (tools_json) |tj| try tool_schema.inferBareJsonToolCalls(allocator, text, tj) else null) orelse
        return null;
    if (!allow_parallel and calls.len > 1) {
        log.info("  parallel_tool_calls=false: clamping {d} parsed calls to the first\n", .{calls.len});
        for (calls[1..]) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        const kept = calls[0];
        allocator.free(calls);
        const one = try allocator.alloc(tool_schema.ParsedToolCall, 1);
        one[0] = kept;
        calls = one;
    }
    if (server_state.g_tool_autocorrect) {
        if (tools_json) |tj| {
            // Order matters: put a buried required param back where the schema
            // says it lives BEFORE coercing types, so the hoisted value is
            // type-checked like any other top-level arg.
            try tool_schema.hoistMisplacedRequiredParams(allocator, calls, tj);
            try tool_schema.coerceToolArgsToSchema(allocator, calls, tj);
        }
    }
    return calls;
}

/// Start the HTTP server on the given host and port.
///
/// `cfg` carries all process-wide defaults (context size, timeouts, PLD
/// defaults, reasoning budget). It's copied into the module-level
/// `server_config` before the listen loop starts.
///
/// Order: bind; start the launch model's load and wait out its refusals (`Scheduler.init`); serve while the rest of
/// the load runs; then the post-load lines and hooks, which a request that got the model waits for (`boot_gate`).
pub fn serve(
    io: std.Io,
    allocator: std.mem.Allocator,
    /// Phase A1: model load happens on the scheduler's inference thread. The
    /// caller (main.zig) is responsible only for CPU-side setup (config parse,
    /// tokenizer load, EOS resolution, chat-template load); everything that
    /// touches mlx is done on the inference thread.
    load_params: model_load.LoadParams,
    config: *model_mod.ModelConfig,
    host: []const u8,
    port: u16,
    cfg: server_state.ServerConfig,
) !void {
    server_state.server_config = cfg;
    // Before `Scheduler.init`: that call starts the model load and every load-time bill inside it
    // asks for the KV width.
    server_state.configured_kv_quant = load_params.kv_quant;
    defer server_state.configured_kv_quant = null;
    server_state.configured_mtp = load_params.mtp;
    defer server_state.configured_mtp = null;
    scheduler_mod.load_serving_bill = &loadServingBill;
    defer scheduler_mod.load_serving_bill = null;

    scheduler_mod.prefill_ubench_chunk = &requestPrefillChunkNow;
    defer scheduler_mod.prefill_ubench_chunk = null;

    server_state.listen_host = host;
    server_state.listen_port = port;
    var listener: ?std.Io.net.Server = if (cfg.embedded_client != null) null else http_conn.startListener(host, port) catch |err| {
        var msg_buf: [512]u8 = undefined;
        if (startupRefusal(err, host, port, &msg_buf)) |msg| {
            log.err("{s}\n", .{msg});
            std.process.exit(1);
        }
        return err;
    };
    if (listener) |l| server_state.listen_port = l.socket.address.ip4.port;

    const boots_model = !load_params.no_initial_load;
    if (boots_model) boot_gate.process.begin();
    defer if (!boot_gate.process.opened.isSet()) boot_gate.process.fail(io);

    // ── Phase A1: spin up the scheduler. Its inference thread does the
    //    Trunk/vision/drafter load, JIT compile, and warmup; mlx ops
    //    are bound to that thread's GPU stream. `init` returns once the load
    //    has passed its refusals, so a refusal still ends the boot unserved.
    var scheduler = scheduler_mod.Scheduler.init(
        allocator,
        io,
        load_params,
        server_state.max_concurrent,
    ) catch |err| {
        if (listener) |*l| l.deinit(io);
        return err;
    };
    defer scheduler.deinit();
    server_state.global_scheduler = scheduler;
    defer server_state.global_scheduler = null;
    server_state.global_registry = scheduler.registry;
    defer server_state.global_registry = null;
    scheduler.registry.load_gate = downloads.loadGate;

    if (instr.current() != null) {
        startGaugeSampler() catch |err| log.err("metrics gauge sampler failed to start: {s}\n", .{@errorName(err)});
        log.info("Prometheus metrics: ENABLED — GET http://{s}:{d}/metrics and /metrics.json\n", .{ host, port });
    }
    // Registered after the scheduler's deinit defer, so it runs first and the sampler never reads a freed scheduler.
    defer stopGaugeSampler();

    var client_thread: ?std.Thread = null;
    defer if (listener) |*l| http_conn.closeEmbeddedListener(io, l, client_thread) else if (client_thread) |t| t.join();
    var accept_thread: ?std.Thread = null;
    defer stopServing(&accept_thread);
    if (listener != null) {
        log.info("\nServer listening on http://{s}:{d}\n", .{ host, server_state.listen_port });
        if (boots_model) log.info("  {s} is loading: the page, /health, /v1/models, /props, /v1/settings and the metrics answer now; generation waits for the model\n", .{load_params.entry.id});
        accept_thread = try std.Thread.spawn(.{}, acceptLoop, .{ allocator, io, &listener.? });
    }

    scheduler.awaitStartupLoad() catch |err| {
        boot_gate.process.fail(io);
        return err;
    };

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
    arch_serve.install(mlxCacheLimitBytes(metrics.getTotalMemBytes()));
    defer arch_serve.uninstall();
    scheduler_mod.prefill_chunk_widen_ok = &adaptivePrefillWidenStillFits;
    defer scheduler_mod.prefill_chunk_widen_ok = null;
    // Whether the hook above can move anything for a given model (the installs are process-wide, so presence is not a gate).
    scheduler_mod.prefill_chunk_adaptive_enabled = &adaptivePrefillChunkEnabled;
    defer scheduler_mod.prefill_chunk_adaptive_enabled = null;
    scheduler_mod.vision_encode_available = &request_media.visionEncodeAvailable;
    scheduler_mod.vision_encode_bill = &request_media.visionEncodeBillBytes;
    scheduler_mod.vision_image_decode = &request_media.decodeEncodedImage;
    scheduler_mod.vision_fit_fault = &towerFitFaultText;
    defer scheduler_mod.vision_encode_available = null;

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

    // Plan 05: the hot prefix cache lives on the LoadedModel
    // (entry.prefix_cache) and is set up by `loadModelOnInferenceThread`
    // using the per-LoadParams capacity + byte budget. Surface a friendly
    // log line so users see whether the cache engaged.
    if (scheduler.hot_prefix_cache != null) {
        if (!server_state.prefix_cache_ram_enabled and diskTierOn(config)) {
            log.info("Prefix cache: SSD ONLY (RAM retention off)\n", .{});
        } else if (resolvedPrefixCacheMem() > 0) {
            const cap_mb = @as(f64, @floatFromInt(resolvedPrefixCacheMem())) / (1024.0 * 1024.0);
            log.info("Hot prefix cache: ENABLED (capacity={d}, mem-cap={d:.1} MB)\n", .{ server_state.prefix_cache_capacity, cap_mb });
        } else {
            log.info("Hot prefix cache: ENABLED (capacity={d}, mem-cap=unlimited)\n", .{server_state.prefix_cache_capacity});
        }
    } else if (server_state.prefix_cache_capacity > 0) {
        if (config.full_attention_interval > 0) {
            log.info("Hot prefix cache: requested capacity={d} but model uses full-attention-interval — disabled\n", .{server_state.prefix_cache_capacity});
        } else {
            log.info("Hot prefix cache: requested capacity={d} but disabled by scheduler\n", .{server_state.prefix_cache_capacity});
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
    log.info("Concurrency: --max-concurrent={d}, {s} (arch: {s})\n", .{ server_state.max_concurrent, concurrencyClause(verdict), config.model_type });
    if (verdict == .ok and server_state.prefix_cache_capacity > 0 and server_state.prefix_cache_capacity < server_state.max_concurrent) server_state.prefix_cache_capacity = server_state.max_concurrent;
    // Installed only now: until the load ends, a signal's default action ends the process mid-load.
    const sigact = std.posix.Sigaction{
        .handler = .{ .handler = signalHandler },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    };
    std.posix.sigaction(std.posix.SIG.INT, &sigact, null);
    std.posix.sigaction(std.posix.SIG.TERM, &sigact, null);

    client_thread = if (server_state.server_config.on_ready) |ready| try ready(server_state.listen_port) else null;

    // Freeze the auto-context NOW, at startup, while the model is freshly
    // loaded and nothing else has taken RAM. Clients read this number once
    // (pi/opencode bake it into a config file) and budget against it for the
    // whole session, so it must not drift with system load. `--ctx-size` wins.
    const pinned = bills.pinAutoContext(config);
    warnUserWireLimit();
    if (manualContext(config) > 0) {
        const ctx = model_settings.contextPick(server_state.server_config.max_context_size, config.ctx_override);
        log.info("Context size: {d} tokens ({s})\n", .{ ctx.value, settings_resolve.sourceLabel(ctx.source, "--ctx-size") });
    } else {
        const memory_ctx = computeMemoryContext(config);
        const memory_allows = bills.safeAutoContext(memory_ctx);
        const rope_ctx = config.contextCap();
        if (config.defaultContext() > 0 and pinned == config.defaultContext()) {
            log.info("Context size: {d} tokens (auto: the model's default; memory would allow {d}) [pinned]\n", .{ pinned, memory_ctx });
        } else if (rope_ctx > 0 and pinned >= rope_ctx) {
            // The checkpoint's own maximum binds; memory had room to spare.
            log.info("Context size: {d} tokens (auto: the model's maximum; memory would allow {d}) [pinned]\n", .{ pinned, memory_allows });
        } else {
            log.info("Context size: {d} tokens (auto: {d}% of the {d}-token memory ceiling, reserving headroom) [pinned]\n", .{ pinned, bills.auto_ctx_safety_pct, memory_ctx });
        }
    }

    if (server_state.server_config.request_timeout_sec > 0) {
        log.info("Request timeout: {d}s\n", .{server_state.server_config.request_timeout_sec});
    }
    const model_ctx = config.contextCap();
    if (model_ctx > 0) {
        log.info("Model context length: {d} tokens\n", .{model_ctx});
    }
    if (request_policy.filledMaxTokens(config) > 0) {
        log.info("default max_tokens for omitted requests: {d} ({s})\n", .{ request_policy.filledMaxTokens(config), if (request_policy.launchMaxTokensDefault() > 0) "--max-tokens" else "model default" });
    }
    if (scheduler.dflash != null) {
        log.info("DFlash speculative decoding: ENABLED (block_size={d}; default for new requests)\n", .{scheduler.drafter_block_size});
    }
    if (arch_flags.mtpOffLine(config)) |line| {
        log.info("{s}\n", .{line});
    } else if (config.expert_streaming and !config.stream_mtp_head) {
        log.info("MTP: off under expert streaming (--mtp keeps the head resident)\n", .{});
    } else if (server_state.server_config.default_force_mtp) {
        log.info("MTP: forced ON for MoE targets (--mtp; default for new requests)\n", .{});
    }
    const launch_use = arch_flags.LaunchFlagUse{ .qwen_gdn_16 = !model_mod.qwen_gdn_f32, .mtp_head_kv_quant = trunk.Trunk.mtp_head_kv_quant_flag };
    var boot_lines: [arch_flags.options.len]?[]const u8 = undefined;
    for (arch_flags.bootLines(config, launch_use, &boot_lines)) |line| if (line) |l| log.info("{s}\n", .{l});
    if (generate_mod.max_mtp_ctx != 0) {
        log.info("MTP context ceiling: {d} tokens (--max-mtp-ctx; requests past it decode serially)\n", .{generate_mod.max_mtp_ctx});
    }
    if (listener != null) {
        if (server_state.g_api_key != null) {
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
        log.info("chat in your browser: {s}chat\n\n", .{chatPageUrl(&chat_url_buf, host, server_state.listen_port)});
    }

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

    scheduler.releaseStartupPin();
    boot_gate.process.open(io);

    if (cfg.embedded_client) |start| client_thread = try start();

    if (accept_thread) |t| {
        t.join();
        accept_thread = null;
    } else acceptLoop(allocator, io, null);
    stopServing(&accept_thread);

    responses_api.deinitGlobalResponseStore();

    log.info("\nShutting down gracefully...\n", .{});
}

/// Accepts until shutdown, one detached thread per connection. With no listener (an embedded client) it only
/// watches the shutdown flag: poll ignores a negative fd and sleeps out the timeout.
fn acceptLoop(allocator: std.mem.Allocator, io: std.Io, listener: ?*std.Io.net.Server) void {
    var poll_fds = [_]std.posix.pollfd{.{
        .fd = if (listener) |l| l.socket.handle else -1,
        .events = std.posix.POLL.IN,
        .revents = 0,
    }};

    while (!server_state.shutdown_requested.load(.acquire)) {
        // Poll with 1-second timeout so we can check shutdown flag
        const poll_result = std.posix.poll(&poll_fds, 1000) catch |err| {
            if (server_state.shutdown_requested.load(.acquire)) break;
            log.err("poll error: {}\n", .{err});
            continue;
        };
        if (poll_result == 0) continue; // timeout, re-check shutdown flag
        if (server_state.shutdown_requested.load(.acquire)) break;

        const accepted_stream = listener.?.accept(io) catch |err| {
            if (server_state.shutdown_requested.load(.acquire)) break;
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
        const args = allocator.create(http_conn.ConnThreadArgs) catch |err| {
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
}

/// Shutdown ordering (prevents a SIGSEGV in Scheduler.complete): stop the accept
/// loop, cancel every in-flight slot so its connection thread unblocks from
/// waitNext and runs its `defer complete(...)`, then WAIT for all connection
/// threads to finish — `serve` runs `scheduler.deinit` on return, which frees
/// the slot queues that an in-flight `complete()` is still touching. The
/// inference thread is still alive here (deinit joins it later), so cancelled
/// slots settle promptly. Bounded so a wedged thread can't hang shutdown forever.
/// Idempotent; also the cleanup of a boot whose load failed with the listener up.
fn stopServing(accept_thread: *?std.Thread) void {
    server_state.shutdown_requested.store(true, .release);
    downloads.stop();
    if (accept_thread.*) |t| {
        t.join();
        accept_thread.* = null;
    }
    if (server_state.global_scheduler) |sch| sch.cancelAllInFlight();
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

fn handleConnectionThread(args: *http_conn.ConnThreadArgs) void {
    // Decrement LAST (declared first → runs after every other defer), so the
    // shutdown drain in `serve` only sees this thread leave once it has fully
    // returned from `complete()` and closed its socket.
    defer _ = active_conn_threads.fetchSub(1, .acq_rel);
    var conn: http_conn.Conn = undefined;
    http_conn.Conn.init(&conn, args.accepted_stream, args.io);
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
    stream: *http_conn.Conn,
) !void {
    // Plan 05: resolve which model this request targets. The registry was
    // set up in `serve()`; per-POST routing happens after we read the body
    // and parse out the optional `"model"` field. For Phase C we use the
    // default model for everything (only one is loaded), but the plumbing
    // is in place for Phase D's hot-load path.
    const registry = server_state.global_registry orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Server not ready", 503);
        return;
    };
    const raw = (try http_conn.readRequestWithin(allocator, stream, REQUEST_READ_LIMIT_MS)) orelse return;
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
    if (http_conn.apiKeyGateApplies(server_state.g_api_key != null, server_state.g_api_key_strict, peerIsLoopback(stream)) and
        !std.mem.eql(u8, method, "OPTIONS") and
        !(std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/health")) and
        !(std.mem.eql(u8, method, "GET") and isChatPagePath(path)) and
        !http_conn.apiKeyAuthorized(request[0..header_end_pos], raw_path))
    {
        log.debug("{s} {s} -> 401 (missing/invalid API key)\n", .{ method, path });
        try http_conn.sendUnauthorized(stream);
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
        const health = boot_gate.process.health();
        log.debug("GET  /health -> {s}\n", .{health.status});
        try sendResponse(stream, health.status, "application/json", health.body);
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
            var history: std.Io.Writer.Allocating = .init(allocator);
            defer history.deinit();
            if (metrics_history.sinceParam(raw_path)) |since| {
                try metrics_history.process.writeSince(stream.io, since, &history.writer);
                try history.writer.writeAll(",");
                try m.finished.writeSince(stream.io, since, &history.writer);
            }
            try instr.renderJsonWith(m, liveSessions(registry, &sessions), if (history.written().len > 0) history.written() else null, &out.writer);
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
    if (http_conn.crossOriginRefused(method, path, request[0..header_end_pos], server_state.listen_host, server_state.listen_port)) {
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
        try models_api.handleModels(allocator, stream);
        return;
    }
    if (std.mem.eql(u8, method, "GET") and std.mem.startsWith(u8, path, "/v1/responses/")) {
        const id = path["/v1/responses/".len..];
        try responses_api.handleResponsesGet(allocator, stream, id);
        return;
    }
    if (std.mem.eql(u8, method, "DELETE") and std.mem.startsWith(u8, path, "/v1/responses/")) {
        const id = path["/v1/responses/".len..];
        try responses_api.handleResponsesDelete(allocator, stream, id);
        return;
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/responses/compact")) {
        // Compaction is a pure data transformation — no model load needed.
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try responses_api.handleResponsesCompact(allocator, stream, body);
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
    if (std.mem.eql(u8, path, "/v1/settings")) {
        if (std.mem.eql(u8, method, "GET")) {
            try settings_api.handleGet(allocator, stream);
        } else if (std.mem.eql(u8, method, "POST")) {
            try settings_api.handlePost(allocator, stream, request_body);
        } else {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "Use GET to read the settings and POST to change them", 405);
        }
        return;
    }
    if (std.mem.eql(u8, path, "/v1/prefix-cache")) {
        if (!std.mem.eql(u8, method, "GET")) {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "Use GET for the prefix cache listing", 405);
            return;
        }
        try prefix_cache_admin.handleList(allocator, stream);
        return;
    }
    if (std.mem.eql(u8, path, "/v1/prefix-cache/clear")) {
        if (!std.mem.eql(u8, method, "POST")) {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "Use POST to clear the prefix cache", 405);
            return;
        }
        try prefix_cache_admin.handleClear(allocator, stream, request_body);
        return;
    }
    if (std.mem.eql(u8, path, "/v1/downloads")) {
        if (std.mem.eql(u8, method, "GET")) {
            try downloads.handleGet(allocator, stream, raw_path);
        } else if (std.mem.eql(u8, method, "POST")) {
            if (@import("download_progress.zig").startRefused(peerIsLoopback(stream), server_state.g_api_key != null)) {
                try sendErrorResponse(allocator, stream, "403 Forbidden", "invalid_request_error", "A download from another machine needs the server started with --api-key", 403);
                return;
            }
            try downloads.handleStart(allocator, stream, request_body);
        } else {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "Use GET for the packs and downloads and POST to start one", 405);
        }
        return;
    }
    if (std.mem.eql(u8, path, "/v1/downloads/cancel")) {
        if (!std.mem.eql(u8, method, "POST")) {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "Use POST to cancel a download", 405);
            return;
        }
        try downloads.handleCancel(allocator, stream, request_body);
        return;
    }
    if (std.mem.eql(u8, path, "/v1/slots/cancel")) {
        if (!std.mem.eql(u8, method, "POST")) {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "Use POST to cancel an in-flight slot", 405);
            return;
        }
        try handleSlotCancel(allocator, stream, request_body);
        return;
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/models/rescan")) {
        // Absorb models downloaded AFTER boot (the Model Browser pulls while
        // the server runs; discovery only walks the roots at startup).
        try models_api.handleModelsRescan(allocator, stream);
        return;
    }

    // ── Plan 05 Phase D: resolve the request's target model via the
    //    scheduler (which delegates the fast path to registry.ensureLoaded
    //    and handles cold-load + eviction internally). Absent `model`
    //    field → default. Unknown id → 404. Unloaded id with no room and
    //    no LRU victim → 503 (NotEnoughMemory). Block-until-loaded is
    //    intentional; clients targeting an unloaded model should set a
    //    longer request timeout (or hit /v1/load-model first).
    const scheduler = server_state.global_scheduler orelse {
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
    switch (http_conn.routeRequestModel(registry, requested_model_id, home)) {
        .entry => |id| requested_model_id = id,
        .default => requested_model_id = "",
        .not_found => {
            const msg = "No model is registered at that path; POST /v1/load-model with the absolute path registers it";
            if (anthropic.isAnthropicRoute(path)) {
                try anthropic.sendAnthropicError(allocator, stream, "not_found_error", msg, 404);
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
    if (requested_model_id.len > 0 and http_conn.isTextGenRoute(method, path)) {
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
                if (anthropic.isAnthropicRoute(path)) {
                    try anthropic.sendAnthropicError(allocator, stream, "invalid_request_error", reason, 400);
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
    if (!http_conn.routeExists(path)) {
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
            try models_api.handlePropsNoModel(allocator, stream);
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
                try models_api.handlePropsNoModel(allocator, stream);
                return;
            }
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "no_model", "No default model configured", 503);
            return;
        },
        error.LoadFailed => {
            try generation_stream.sendLoadFailedResponse(allocator, stream, scheduler, requested_model_id);
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
            if (try generation_stream.sendNamedLoadRefusal(allocator, stream, scheduler, requested_model_id, err)) return;
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
    const status_read = http_conn.isStatusRoute(method, path);
    defer if (status_read) scheduler.releaseStatus(lm) else scheduler.release(lm);
    // The launch model turns ready before its boot pins its context and installs the request hooks.
    boot_gate.process.waitOpen(stream.io);

    // A model loaded on demand freezes its own auto-context here, for the same
    // reason the `--model` primary does at startup: the number we advertise is
    // what clients budget against, and each model has its own dims. Idempotent,
    // and a no-op under `--ctx-size`.
    if (lm.config) |c| _ = bills.pinAutoContext(c);
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
        try http_conn.handleResponsesWebSocket(allocator, stream, request[0..header_end_pos], lm);
        return;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/props")) {
        log.debug("GET  /props -> 200\n", .{});
        try models_api.handleProps(allocator, stream, lm);
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
        try completions.handleCompletions(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/messages")) {
        if (text_gen_reject) |reason| {
            try anthropic.sendAnthropicError(allocator, stream, "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try anthropic.handleAnthropicMessages(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/messages/count_tokens")) {
        if (text_gen_reject) |reason| {
            try anthropic.sendAnthropicError(allocator, stream, "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try anthropic.handleAnthropicCountTokens(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/responses")) {
        if (text_gen_reject) |reason| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try responses_api.handleResponses(allocator, stream, body, lm);
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

/// Live requests, then each hot-cache entry no live row restored from, each row stamped with its
/// model's effective context limit. The connection-thread reader copies the inference thread's
/// `queue_mu` and `digest_mu` snapshots under those same locks: a
/// Session is POD with an inline model buffer, so the copy stays valid after
/// its slot frees. Rows carry no pointers into scheduler or registry state.
fn liveSessions(registry: *ModelRegistry, buf: *[2 * instr.MAX_SESSIONS]instr.Session) []instr.Session {
    const sch = server_state.global_scheduler orelse return buf[0..0];
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


test "Sushi quant memory request refusal carries the actual KV width MTP and maximum to the wire" {
    const saved_ctx = server_state.server_config.max_context_size;
    defer server_state.server_config.max_context_size = saved_ctx;
    const saved_mtp = server_state.configured_mtp;
    defer server_state.configured_mtp = saved_mtp;
    server_state.server_config.max_context_size = 500000;
    server_state.configured_mtp = null;
    var cfg = fixtures.qwen4RequestTestConfig();
    cfg.expert_layout = .exl3_k4;
    const chunk: u64 = 1024;
    const budget = prefillNeededAtChunk(&cfg, 64000, 0, 4, chunk, .{});
    const bill = AdmissionBill{ .needed = prefillNeededAtChunk(&cfg, 128000, 32000, 4, chunk, .{}), .available = budget, .chunk = chunk };
    const refusal = bills.requestContextRefusal(&cfg, 128000, 32000, trunk.KVQuantConfig.affine(4), bill, .{}).?;
    try std.testing.expectEqual(@as(u64, 4), refusal.kv_bits);
    try std.testing.expectEqual(@as(u32, 64000), refusal.maximum);
    try std.testing.expectEqual(@as(u64, 160000), refusal.requested);
    const mtp = bills.requestContextRefusal(&cfg, 128000, 32000, trunk.KVQuantConfig.affine(4), bill, .{ .mtp_on = true }).?;
    try std.testing.expect(mtp.maximum < refusal.maximum);
    var buf: [256]u8 = undefined;
    const message = memoryContextRefusalMessage(&buf, refusal);
    try std.testing.expect(std.mem.indexOf(u8, message, "Maximum context: 64000 tokens") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "KV4") != null);
    server_state.slot_memory_context_refusal = refusal;
    const wire = generation_stream.mapGenerationError(error.PrefillDoesNotFit, &buf);
    try std.testing.expectEqual(@as(u32, 400), wire.code);
    try std.testing.expect(std.mem.indexOf(u8, wire.message, "Maximum context: 64000 tokens") != null);
    try std.testing.expectEqual(@as(?model_registry_mod.MemoryContextRefusal, null), server_state.slot_memory_context_refusal);
    try std.testing.expectEqualStrings(generation_stream.PREFILL_NOFIT_MSG, generation_stream.mapGenerationError(error.PrefillDoesNotFit, &buf).message);
}


/// The message `error.GenerationOutOfMemory` sends on every surface and both paths.
pub const GEN_OOM_MSG = "The engine ran out of GPU memory during this request and it was abandoned. The server is still running. Reduce the prompt length, lower --ctx-size, or free memory on the machine.";

/// `fitAfterUnloadingIdle`'s context for one request: each unload re-bills the same prompt.
const IdleUnloadFit = struct {
    sch: *scheduler_mod.Scheduler,
    keep_id: []const u8,
    bill: *AdmissionBill,
    config: *const model_mod.ModelConfig,
    prompt_ids: []const u32,
    max_tokens: u32,
    kv_override: ?trunk.KVQuantConfig,
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

pub fn checkAttentionMemory(allocator: std.mem.Allocator, stream: *http_conn.Conn, prompt_ids: []const u32, max_tokens: u32, config: *const model_mod.ModelConfig, keep_id: []const u8, is_anthropic: bool, kv_override: ?trunk.KVQuantConfig, unchunked_prefill: bool, enable_mtp: bool) !bool {
    if (arch_flags.kvQuantRequestRefusal(config, kv_override orelse configuredKvQuantFor(config))) |msg| {
        if (is_anthropic) {
            try anthropic.sendAnthropicError(allocator, stream, "invalid_request_error", msg, 400);
        } else {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, null);
        }
        return false;
    }
    const prompt_len: usize = prompt_ids.len;
    if (config.num_attention_heads == 0) return true; // unknown architecture, skip check
    // The connection thread has no slot and no cache: it bills cold and defers a warm prompt.
    var bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_override, unchunked_prefill, prompt_ids, .{ .mtp_on = enable_mtp });
    bills.logAdmissionDecision(bill);
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
    if (needed > available) if (server_state.global_scheduler) |sch| {
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
        const base = try bills.memoryRefusalMessage(allocator, prompt_len, needed_mb, avail_mb, bill, config.admissionEvictsHotCache());
        defer allocator.free(base);
        const kv = kv_override orelse configuredKvQuantFor(config);
        const refusal = bills.requestContextRefusal(config, prompt_len, max_tokens, kv, bill, .{ .mtp_on = enable_mtp });
        const msg = if (refusal) |r|
            try std.fmt.allocPrint(allocator, "{s} Maximum context at KV{d} and prefill chunk {d}: {d} tokens.", .{ base, r.kv_bits, r.chunk, r.maximum })
        else
            try allocator.dupe(u8, base);
        defer allocator.free(msg);
        countRejected();
        if (is_anthropic) {
            try anthropic.sendAnthropicError(allocator, stream, "invalid_request_error", msg, 400);
        } else {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, 400);
        }
        return false;
    }
    return true;
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
fn handleLoadModelStrict(allocator: std.mem.Allocator, stream: *http_conn.Conn, request_body: []const u8) !void {
    const scheduler = server_state.global_scheduler orelse {
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
        const registry = server_state.global_registry orelse {
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
            try generation_stream.sendLoadFailedResponse(allocator, stream, scheduler, requested_id);
            return;
        },
        else => {
            if (try generation_stream.sendNamedLoadRefusal(allocator, stream, scheduler, requested_id, err)) return;
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
    boot_gate.process.waitOpen(stream.io);
    // Promotion happens only past ensureLoaded's error switch: a load that
    // FAILED must never steal the default from a model that answers.
    if (make_default) {
        if (server_state.global_registry) |registry| {
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
        \\{{"model":{{"id":{f},"object":"model","loaded":true,"state":"ready","bytes_resident":{d},"bytes_on_disk":{s},"drafter_loaded":{s},"drafter_path":{s}}}}}
    , .{
        json_text.Str{ .s = model_id },
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
fn handleUnloadModelStrict(allocator: std.mem.Allocator, stream: *http_conn.Conn, request_body: []const u8) !void {
    const scheduler = server_state.global_scheduler orelse {
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
        const by_path = if (server_state.global_registry) |r| r.peekByPath(trimmed) else null;
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
        \\{{"model":{{"id":{f},"object":"model","loaded":false,"state":"unloaded"}}}}
    , .{json_text.Str{ .s = id_for_body }});
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// The model-level half of `Scheduler.batchVerdict`: does this loaded model
/// batch decode at all? Per-slot arms (spec, grammar, logprobs) come later.
pub fn batchVerdictFor(entry: *const LoadedModel) sched_policy.BatchVerdict {
    const exclusive = if (entry.transformer) |t| t.ownsModuleDecodeState() else false;
    return batchVerdictOf(entry.config, exclusive);
}

pub fn batchVerdictOf(cfg: ?*const model_mod.ModelConfig, exclusive: bool) sched_policy.BatchVerdict {
    if (exclusive) return .exclusive;
    const c = cfg orelse return .arch;
    return if (slot_admission.configBatchesDecode(c)) .ok else .arch;
}

/// The boot line's account of concurrent requests, from the verdict `/props` reports.
pub fn concurrencyClause(verdict: sched_policy.BatchVerdict) []const u8 {
    return switch (verdict) {
        .ok => "batched decode on",
        .exclusive => "batched decode off; concurrent requests queue, one at a time",
        else => "batched decode off; concurrent requests interleave serially",
    };
}

/// A request decoding or queued, or a model loading or evicting: an update now would cut it off.
fn busyForUpdate(io: std.Io) bool {
    if (server_state.global_scheduler) |sch| {
        sch.queue_mu.lockUncancelable(sch.io);
        const in_flight = sch.in_flight;
        sch.queue_mu.unlock(sch.io);
        if (in_flight > 0) return true;
    }
    const reg = server_state.global_registry orelse return false;
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
fn handleMetricsEnable(stream: *http_conn.Conn) !void {
    const changed = instr.process.enable();
    startGaugeSampler() catch |err| {
        log.err("metrics gauge sampler failed to start: {s}\n", .{@errorName(err)});
        return sendResponse(stream, "500 Internal Server Error", "application/json", "{\"error\":\"metrics enabled but the gauge sampler did not start; retry\"}");
    };
    if (changed) log.info("Prometheus metrics: ENABLED at runtime (POST /metrics/enable)\n", .{});
    try sendResponse(stream, "200 OK", "application/json", if (changed) "{\"metrics\":true,\"already_enabled\":false}" else "{\"metrics\":true,\"already_enabled\":true}");
}


fn handleSlotCancel(allocator: std.mem.Allocator, stream: *http_conn.Conn, request_body: []const u8) !void {
    const target_id = parseCancelRequestId(allocator, request_body) orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Send {\"id\": <request_id>}", 400);
        return;
    };
    const scheduler = server_state.global_scheduler orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "server_error", "Server is not ready to process requests", 503);
        return;
    };
    if (!scheduler.cancelSlotById(target_id)) {
        try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "No in-flight request matches that id", 404);
        return;
    }
    log.info("[cancel] slot request_id={d} cancelled via API\n", .{target_id});
    try sendResponse(stream, "200 OK", "application/json", "{\"ok\":true}");
}

fn handleUpdate(allocator: std.mem.Allocator, stream: *http_conn.Conn, headers: []const u8, raw_path: []const u8) !void {
    var ebuf: [std.fs.max_path_bytes]u8 = undefined;
    var rbuf: [512]u8 = undefined;
    var vbuf: [64]u8 = undefined;
    const latest = update_mod.availableVersion(&vbuf);
    const ask: update_mod.Ask = .{
        .bind_host = server_state.listen_host,
        .port = server_state.listen_port,
        .peer_loopback = peerIsLoopback(stream),
        .origin = http_conn.findHeaderValueCI(headers, "origin"),
        .key_set = server_state.g_api_key != null,
        .key_ok = http_conn.apiKeyAuthorized(headers, raw_path),
        .host_managed = server_state.host_managed,
        .busy = busyForUpdate(stream.io),
        .install = update_mod.selfInstallRefusal(stream.io, &ebuf, &rbuf),
        .available = latest != null,
    };
    if (update_mod.guard(ask)) |r| {
        log.info("[update] POST /v1/update -> {d}: {s}\n", .{ r.code, r.message });
        if (r.code == 401) return http_conn.sendUnauthorized(stream);
        return sendErrorResponse(allocator, stream, r.status, r.kind, r.message, r.code);
    }
    const body = try std.fmt.allocPrint(allocator, "{{\"status\":\"updating\",\"from\":\"{s}\",\"to\":\"{s}\"}}", .{ update_mod.version, latest.? });
    defer allocator.free(body);
    try sendResponse(stream, "202 Accepted", "application/json", body);
    log.info("[update] updating {s} -> {s} from the chat page; the server restarts\n", .{ update_mod.version, latest.? });
    update_mod.relaunch_requested.store(true, .release);
    server_state.shutdown_requested.store(true, .release);
}

fn handleTokenize(
    allocator: std.mem.Allocator,
    stream: *http_conn.Conn,
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
    stream: *http_conn.Conn,
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
/// (`think_tags.continuationRequested`). A flag set on a user-final request is
/// ignored rather than 400'd: the request is perfectly answerable as an
/// ordinary turn, and refusing it would break the obvious client that sets the
/// field once for the whole session.
fn continueFinalMessageRequested(root: std.json.ObjectMap, messages: []const chat_mod.Message) bool {
    const v = root.get("continue_final_message") orelse return false;
    if (v != .bool or !v.bool) return false;
    return think_tags.continuationRequested(messages);
}






pub fn handleChatCompletions(
    allocator: std.mem.Allocator,
    stream: *http_conn.Conn,
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
    try readOpenAiMessages(&wire, messages_val.array.items, request_media.visionPreprocFromConfig(config));
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
    const max_tokens: u32 = request_policy.resolveRequestMaxTokens(
        requested_max_tokens,
        request_policy.omittedMaxTokensDefault(config),
    );
    // For the request log line only: which of the three sources this came from.
    const max_tokens_origin = request_policy.maxTokensOrigin(requested_max_tokens, request_policy.launchMaxTokensDefault(), config.defaultMaxTokens());

    const is_stream = if (root.get("stream")) |v| v == .bool and v.bool else false;

    const temperature = request_policy.samplingTemperature(config, parseJsonFloatOpt(root, "temperature", 0.0, 2.0));
    const top_p = request_policy.samplingTopP(config, parseJsonFloatOpt(root, "top_p", 0.0, 1.0));
    const top_k = request_policy.samplingTopK(config, parseJsonTopKOpt(root, "top_k"));

    const repeat_penalty = request_policy.parseRepeatPenalty(root);

    const presence_penalty = request_policy.parseAdditivePenalty(root, "presence_penalty");
    const frequency_penalty = request_policy.parseAdditivePenalty(root, "frequency_penalty");

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
    const tool_choice = tool_calls_mod.parseToolChoice(root.get("tool_choice"));
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
        if (tool_calls_mod.toolChoiceNamesUndeclared(allocator, tool_choice, tools_json)) {
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
    const model_name_raw = if (root.get("model")) |v|
        (if (v == .string) v.string else config.model_type)
    else
        config.model_type;
    const model_name = try json_text.innerAlloc(allocator, model_name_raw);
    defer allocator.free(model_name);

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
    const effort_snap = effort_mod.snapshot();
    if (request_policy.nonWordEffort(root, .chat)) |v| return request_policy.sendEffortValueRefusal(allocator, stream, lm, v, &effort_snap, false);
    const served_arms = effort_mod.servedArmsIn(&effort_snap, config.model_type);
    const effort_cfg = request_policy.parseReasoningEffort(root, request_policy.effortWordOnly(allocator, lm, tok), if (served_arms) |*a| a.slice() else null) catch {
        const msg = try request_policy.effortRefusalFor(allocator, lm, root.get("reasoning_effort").?.string, &effort_snap);
        defer allocator.free(msg);
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, 400);
        return;
    };
    var effort_notice: ?request_policy.EffortNotice = null;
    defer if (effort_notice) |n| n.deinit(allocator);
    defer request_policy.clearEffortNotice(stream);
    if (effort_cfg) |e| effort_notice = try request_policy.holdEffortNotice(allocator, stream, request_policy.effortModelName(lm), e);
    const thinking = request_policy.chatThinking(root, effort_cfg, config, &effort_snap);
    var enable_thinking = thinking.enable;
    if (!try request_policy.checkThinkingSupport(allocator, stream, lm, &effort_snap, enable_thinking, false)) return;
    // Reasoning budget (max tokens in <think> block, -1 = unlimited).
    const reasoning_budget: i32 = thinking.budget;
    const template_effort = thinking.template;

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
    var tokenize_sw = io_util.Stopwatch.init(stream.io);
    const render_config = renderConfigFor(chat_config, root, model_settings.preserve_thinking_flag, config.preserve_thinking_override);
    var prompt_ids_raw = cachedFormatChat(allocator, stream.io, lm, tok, &render_config, messages.items, tools_json, tool_choice_instruction, enable_thinking, template_effort, continue_final) catch |err| return generation_stream.sendGenerationError(allocator, stream, err, .openai);
    var schema_proto: rp_mod.Protocol = undefined;
    var schema_proto_active = false;
    if (grammar_schema_val != null and !has_tools and enable_thinking and reasoning_budget < 0) {
        schema_proto_active = resolveReasoningProtocol(allocator, stream.io, lm, tok, chat_config, prompt_ids_raw, &schema_proto, true);
    }
    switch (request_policy.schemaMasksThinking(
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
            prompt_ids_raw = cachedFormatChat(allocator, stream.io, lm, tok, &render_config, messages.items, tools_json, tool_choice_instruction, false, template_effort, continue_final) catch |err| return generation_stream.sendGenerationError(allocator, stream, err, .openai);
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
    log.info("POST /v1/chat/completions ({d} msgs, max_tokens={s}, temp={d:.2}, top_p={d:.2}, top_k={d}, stream={}, thinking={}, sys={d}b, user={d}b, tools={d}b, tool_msgs={d}) \n", .{ messages.items.len, request_policy.describeMaxTokens(&max_tokens_desc_buf, max_tokens, max_tokens_origin), temperature, top_p, top_k, is_stream, enable_thinking, system_chars, user_chars, tools_len, tool_msg_count });
    log.info("  > \"{s}{s}\"\n", .{ last_msg.content[0..preview_len], if (last_msg.content.len > 80) "..." else "" });

    // Every message's media, placed and encoded; the slot takes the
    // embeddings and M-RoPE table at submit, `deinit` frees what it did not.
    const cache_key = requestCacheKey(root);
    var req_media: request_media.PreparedMedia = .{};
    defer req_media.deinit(allocator);
    switch (try request_media.prepareRequestMedia(allocator, lm, messages.items, &prompt_ids_raw, getEffectiveContextLength(config))) {
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
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", generation_stream.contextOverflowMessage(&ovf_buf, prompt_ids.len, effective_ctx), 400);
        return;
    }

    // Clamp max_tokens first: the memory guard bills the reserved generation headroom, and an
    // omitted max_tokens is the maxInt(u32)/4 sentinel.
    const effective_max_tokens = request_policy.clampMaxTokens(max_tokens, prompt_ids.len, effective_ctx);

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
    sampling.think_penalty = request_policy.armThinkPenalty(allocator, lm, tok, prompt_ids, enable_thinking, request_policy.resolveThinkPenalty(root, model_settings.think_penalty_flag, config.think_penalty_override));
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
        generation_stream.handleStreamingGeneration(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, include_usage, has_tools, tools_json, allow_parallel_tools, logprobs_n, enable_thinking, surface_budget, enable_drafter, enable_mtp, allow_batch_mtp, sub_ve, req_media.chain, cache_key, sub_mrope, kv_quant_override, tokenize_ns) catch |err| {
            log.err("  -> streaming error: {}\n", .{err});
            // One mapping with the non-streaming arm: an HTTP status before the SSE head, an SSE `error` event after.
            generation_stream.sendGenerationError(allocator, stream, err, .openai) catch {};
        };
    } else {
        generation_stream.handleNonStreamingGeneration(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, has_tools, tools_json, allow_parallel_tools, logprobs_n, enable_thinking, surface_budget, enable_drafter, enable_mtp, allow_batch_mtp, sub_ve, req_media.chain, cache_key, sub_mrope, kv_quant_override, tokenize_ns) catch |err| {
            log.err("  -> {s}\n", .{@errorName(err)});
            generation_stream.sendGenerationError(allocator, stream, err, .openai) catch {};
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
    return generation_stream.requestSpecModes(enable_drafter, enable_mtp, dflash_loaded, mtp_loaded, sampling, logprobs_n).use_mtp;
}

test "the bill prices the MTP head only for a request that runs it: a DFlash2 request never builds the head's cache" {
    const t = std.testing;
    const saved_force = server_state.server_config.default_force_mtp;
    defer server_state.server_config.default_force_mtp = saved_force;
    server_state.server_config.default_force_mtp = false;
    const plain = sampler.SamplingParams{};
    // A loaded drafter outranks the head; `enable_drafter:false` hands the request back to it.
    try t.expect(!requestBillsMtpHead(true, true, true, true, plain, 0));
    try t.expect(requestBillsMtpHead(false, true, true, true, plain, 0));
    // A penalty keeps the drafter off, so the head runs.
    try t.expect(requestBillsMtpHead(true, true, true, true, .{ .presence_penalty = 1.5 }, 0));
    try t.expect(requestBillsMtpHead(true, true, false, true, plain, 0));
    try t.expect(!requestBillsMtpHead(true, false, false, true, plain, 0));
    var cfg = fixtures.qwen4ExpOomConfig();
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
fn sampleGauges(ctx: GaugeSamplerCtx) void {
    // System gauges (non-blocking syscalls).
    ctx.metrics.gpu_utilization_pct.set(@as(u64, metrics.getGpuPct()));
    ctx.metrics.gpu_clock_mhz.set(gpu_clock_live.sampleMhz());
    ctx.metrics.gpu_clock_max_mhz.set(gpu_clock_live.maxMhz());
    ctx.metrics.memory_mb.set(@as(u64, metrics.getAppMemFootprintMb()));
    // The two halves of the MLX allocator, so `memory_mb`'s gap has a name.
    var mlx_active: usize = 0;
    var mlx_cache: usize = 0;
    _ = mlx.mlx_get_active_memory(&mlx_active);
    _ = mlx.mlx_get_cache_memory(&mlx_cache);
    ctx.metrics.mlx_active_bytes.set(@as(u64, mlx_active));
    ctx.metrics.mlx_cache_bytes.set(@as(u64, mlx_cache));
    // qwen4 n-gram table warm (F8) — same lock-free published-atomic pattern.
    ctx.metrics.ngram_warm_bytes.set(arch_serve.warmBytes(.monotonic));

    // Request queue depth — brief lock to read two scheduler counters only.
    ctx.scheduler.queue_mu.lockUncancelable(ctx.scheduler.io);
    const running = @as(u64, ctx.scheduler.in_flight);
    const waiting = @as(u64, ctx.scheduler.pending.items.len);
    ctx.scheduler.queue_mu.unlock(ctx.scheduler.io);
    ctx.metrics.requests_running.set(running);
    ctx.metrics.requests_waiting.set(waiting);

    // The other phase. Prompt-token counters and the prefill-time histogram
    // only advance at request COMPLETION, so a long prefill was invisible: GPU
    // pinned, decode reading 0, prefill reading "—". This gauge moves per
    // prefill chunk and returns to 0 the moment the prefill ends.
    ctx.metrics.prefill_tokens_live.set(ctx.scheduler.inflight_prefill_tokens.load(.monotonic));
    ctx.metrics.prefill_tokens_expected.set(ctx.scheduler.inflight_prefill_expected.load(.monotonic));
    ctx.metrics.requests_prefilling.set(ctx.scheduler.requests_prefilling.load(.monotonic));

    metrics_history.process.push(ctx.scheduler.io, .{
        .t_ms = io_util.nowMs(ctx.scheduler.io),
        .gen_tokens = ctx.metrics.generation_tokens_live.load(),
        .prefill_tokens = ctx.metrics.prefill_tokens_live.load(),
        .mlx_active = mlx_active,
        .mlx_cache = mlx_cache,
        .memory_mb = @intCast(@min(ctx.metrics.memory_mb.load(), std.math.maxInt(u32))),
        .gpu_pct = @intCast(@min(ctx.metrics.gpu_utilization_pct.load(), 100)),
        .gpu_mhz = @intCast(@min(ctx.metrics.gpu_clock_mhz.load(), std.math.maxInt(u16))),
        .running = @intCast(@min(running, std.math.maxInt(u16))),
        .waiting = @intCast(@min(waiting, std.math.maxInt(u16))),
        .prefilling = @intCast(@min(ctx.metrics.requests_prefilling.load(), std.math.maxInt(u16))),
        .batched = @intCast(@min(ctx.metrics.batched_group_size.load(), std.math.maxInt(u16))),
    });
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
    const tick_ms = @import("sched_policy.zig").idleEvictTickMs(ctx.window_ms);
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
    const sch = server_state.global_scheduler orelse return;
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






test "metrics enable: POST /metrics/enable turns the feeds on, idempotently" {
    const t = std.testing;
    instr.process = .{};
    defer instr.process = .{};
    for ([_][]const u8{ "/metrics", "/metrics.json" }) |path| {
        const request = try std.fmt.allocPrint(t.allocator, "GET {s} HTTP/1.1\r\n\r\n", .{path});
        defer t.allocator.free(request);
        const off = try http_conn.serveOneForTest(request);
        defer t.allocator.free(off);
        try t.expect(std.mem.startsWith(u8, off, "HTTP/1.1 503 "));
        try t.expect(std.mem.indexOf(u8, responseBody(off), "POST /metrics/enable") != null);
    }
    const first = try http_conn.serveOneForTest("POST /metrics/enable HTTP/1.1\r\nContent-Length: 0\r\n\r\n");
    defer t.allocator.free(first);
    try t.expect(std.mem.startsWith(u8, first, "HTTP/1.1 200 OK\r\n"));
    try t.expectEqualStrings("{\"metrics\":true,\"already_enabled\":false}", responseBody(first));
    try t.expect(instr.current() != null);
    for ([_][]const u8{ "/metrics", "/metrics.json" }) |path| {
        const request = try std.fmt.allocPrint(t.allocator, "GET {s} HTTP/1.1\r\n\r\n", .{path});
        defer t.allocator.free(request);
        const on = try http_conn.serveOneForTest(request);
        defer t.allocator.free(on);
        try t.expect(std.mem.startsWith(u8, on, "HTTP/1.1 200 OK\r\n"));
    }
    const core = instr.current().?;
    core.recordRejected();
    const again = try http_conn.serveOneForTest("POST /metrics/enable HTTP/1.1\r\nContent-Length: 0\r\n\r\n");
    defer t.allocator.free(again);
    try t.expect(std.mem.startsWith(u8, again, "HTTP/1.1 200 OK\r\n"));
    try t.expectEqualStrings("{\"metrics\":true,\"already_enabled\":true}", responseBody(again));
    try t.expect(instr.current().? == core);
    try t.expectEqual(@as(u64, 1), core.requests_rejected_total.load());
}

test "an unknown endpoint is a 404 that never reaches model resolution" {
    const t = std.testing;
    var sch: scheduler_mod.Scheduler = undefined;
    const prev_sch = server_state.global_scheduler;
    defer server_state.global_scheduler = prev_sch;
    server_state.global_scheduler = &sch;
    const response = try http_conn.serveOneForTest("POST /v1/load HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 13\r\n\r\n{\"model\":\"m\"}");
    defer t.allocator.free(response);
    try t.expect(std.mem.startsWith(u8, response, "HTTP/1.1 404 Not Found\r\n"));
    try t.expect(std.mem.indexOf(u8, responseBody(response), "Unknown endpoint") != null);
}








/// The model's chat config as this request renders it.
pub fn renderConfigFor(chat_config: *const chat_mod.ChatConfig, root: std.json.ObjectMap, flag: ?bool, setting: ?bool) chat_mod.ChatConfig {
    var cfg = chat_config.*;
    cfg.preserve_thinking = request_policy.resolvePreserveThinking(chatTemplateKwargBool(root, "preserve_thinking"), flag, setting);
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
    media: *request_media.RequestMedia,
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
            if (media.addImage(img_slot, request_media.openAiImageUrl(part), vp, pos))
                try placed.append(allocator, .{ .at = joined_len, .kind = .image });
        } else if (vocab == .anthropic and std.mem.eql(u8, ptype, "image")) {
            const url = try anthropic.anthropicImageUrl(allocator, part);
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
    media: request_media.RequestMedia,
    tool_call_lists: std.ArrayList([]const chat_mod.ToolCall) = .empty,
    /// Joined text, serialized tool arguments and other strings messages borrow.
    texts: std.ArrayList([]const u8) = .empty,

    pub fn init(allocator: std.mem.Allocator) WireMessages {
        return .{ .allocator = allocator, .media = request_media.RequestMedia.init(allocator) };
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
pub fn towerFitFault(config: *const model_mod.ModelConfig, images: []const chat_mod.ImageData, videos: []const chat_mod.VideoData, rows: usize, available: u64) ?request_media.MediaFault {
    if (config.qv_heads == 0) return null;
    const bill = request_media.visionEncodeBill(config, images, videos, rows);
    const worst = bill.largest_group_patches;
    const needed = bill.bytes;
    if (needed <= available) return null;
    const mb: u64 = 1024 * 1024;
    return request_media.MediaFault.init(false, "encoding the media needs ~{d} MB of GPU memory (its largest block has {d} patches), ~{d} MB is available", .{ needed / mb, worst, available / mb });
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
const test_wire_vp: chat_mod.VisionPreproc = .{ .mode = arch_media.modeNamed("qwen") orelse .gemma, .patch = 16, .tps = 2, .merge = 2 };

pub fn testWire(body: []const u8, vocab: WireVocab) !struct { parsed: std.json.Parsed(std.json.Value), wire: WireMessages } {
    // The processor is the arch's: a build without it cannot decode these images.
    if (!arch_media.servesMode(test_wire_vp.mode)) return error.SkipZigTest;
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, body, .{});
    errdefer parsed.deinit();
    var wire = WireMessages.init(std.testing.allocator);
    errdefer wire.deinit();
    const items = parsed.value.object.get("messages").?.array.items;
    switch (vocab) {
        .openai => try readOpenAiMessages(&wire, items, test_wire_vp),
        .anthropic => try anthropic.readAnthropicMessages(&wire, items, test_wire_vp),
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



// ── Tests ──

const testing = std.testing;










test "getTimeoutNs computes correctly" {
    const original = server_state.server_config.request_timeout_sec;
    defer server_state.server_config.request_timeout_sec = original;

    server_state.server_config.request_timeout_sec = 300;
    try testing.expectEqual(@as(u64, 300_000_000_000), getTimeoutNs());

    server_state.server_config.request_timeout_sec = 0;
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






test "settingsPropsJson: /props names the greedy tail and where it came from" {
    for ([_]settings_resolve.Setting(bool){ .{ .value = false, .source = .default }, .{ .value = true, .source = .model_settings }, .{ .value = true, .source = .flag } }, [_][]const u8{ "default", "model-settings.json", "--mtp-greedy-tail" }) |tail, source| {
        const frag = try models_api.settingsPropsJson(testing.allocator, .{ .engine = "mlx", .kv_quant = "8", .prefill_chunk = 8192, .mtp_loaded = true, .mtp_default_on = true, .mtp_acceptance = .exact, .mtp_greedy_tail = tail, .mtp_depth = 0, .mtp_adaptive = true, .max_mtp_ctx = 0, .drafter = "none", .max_concurrent = 1, .prefix_cache_mem_bytes = 0, .prefix_cache_disk_bytes = 0 });
        defer testing.allocator.free(frag);
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, frag[",\"settings\":".len..], .{});
        defer parsed.deinit();
        const mtp = parsed.value.object.get("mtp").?.object;
        try testing.expectEqual(tail.value, mtp.get("greedy_tail").?.bool);
        try testing.expectEqualStrings(source, mtp.get("greedy_tail_source").?.string);
    }
}

test "prefix cache default capacity covers interleaved agent flows" {
    // Claude Code-style clients interleave several conversation roots (main
    // thread, subagents, title generation). With capacity 1, every
    // interleaved request evicted the long system-prompt prefix and forced a
    // full re-prefill per turn.
    try testing.expect(server_state.prefix_cache_capacity >= 4);
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
    try std.testing.expect(tool_schema.toolCallConformsToSchema(allocator, calls[0], tools));

    // The escape hatch covers this correction too: it repairs the MODEL's output
    // (unlike the inferred-name filter, which corrects our own heuristic).
    server_state.g_tool_autocorrect = false;
    defer server_state.g_tool_autocorrect = true;
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

    server_state.g_tool_autocorrect = false;
    defer server_state.g_tool_autocorrect = true; // restore the default for other tests
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
    // from tool_calls.parseToolCalls in isolation. Uses the Agent shell tool's real
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

    // A dashboard user learns where the switch is; a CLI user keeps the flag.
    const free = insufficient_free_memory_message;
    try testing.expect(std.mem.indexOf(u8, free, "Skip the memory check before loading") != null);
    try testing.expect(std.mem.indexOf(u8, free, "--skip-mem-preflight") != null);

    // One mapping serves both load switches; a second literal is the drift.
    try testing.expectEqualStrings("out_of_memory", generation_stream.loadRefusalFor(error.NotEnoughMemory).?.type);
    try testing.expectEqualStrings(not_enough_memory_message, generation_stream.loadRefusalFor(error.NotEnoughMemory).?.message);
    try testing.expectEqualStrings("out_of_memory", generation_stream.loadRefusalFor(error.InsufficientMemory).?.type);
    try testing.expectEqualStrings(insufficient_free_memory_message, generation_stream.loadRefusalFor(error.InsufficientMemory).?.message);
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

test "generated think tags require an unambiguous literal template opener" {
    try std.testing.expectEqualStrings("<think>", templateThinkOpener("{{ '<think>' }} <think>").?);
    try std.testing.expect(templateThinkOpener("{{ '<think:opensource>' }}") == null);
    try std.testing.expect(templateThinkOpener("<think> <think:opensource>") == null);
    try std.testing.expect(templateThinkOpener("<think:{{ suffix }}>") == null);
    try std.testing.expect(templateThinkOpener("assistant") == null);
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
    const choice = tool_calls_mod.parseToolChoice(parsed.value);
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
    const frag = try models_api.settingsPropsJson(testing.allocator, .{ .engine = "mlx", .kv_quant = "off", .prefill_chunk = 8192, .mtp_loaded = false, .mtp_default_on = false, .mtp_acceptance = .exact, .mtp_depth = 0, .mtp_adaptive = false, .max_mtp_ctx = 0, .drafter = "none", .max_concurrent = 8, .prefix_cache_mem_bytes = 0, .prefix_cache_disk_bytes = 0, .prefill_decode_share = 0.5 });
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
    const prev_sch = server_state.global_scheduler;
    defer server_state.global_scheduler = prev_sch;
    server_state.global_scheduler = &sch;

    const reg = try ModelRegistry.init(t.allocator, t.io, null, 1, 0, null);
    defer reg.deinit();
    const lm = try reg.registerStub("org/live-test", "/path/to/m", null);
    var cfg = std.mem.zeroes(model_mod.ModelConfig);
    lm.state = .ready;
    lm.config = &cfg;
    defer lm.config = null; // LoadedModel.deinit destroys a non-null config — detach first
    const prev_ctx = server_state.server_config.max_context_size;
    defer server_state.server_config.max_context_size = prev_ctx;
    server_state.server_config.max_context_size = 8192; // manualContext wins: no GPU sizing

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
    server_state.global_scheduler = null;
    try t.expectEqual(@as(usize, 0), liveSessions(reg, &buf).len);
}

test "a streamed --vision load proves its largest image encode beside the budget, with the planned KV" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    if (!arch_media.claimed(&cfg)) return;
    cfg.expert_streaming = true;
    const kv = bills.expertStreamingPlannedKvBytes(&cfg);
    try std.testing.expectEqual(kv, bills.expertStreamingBesideBudgetBytes(&cfg));
    cfg.expert_vision_encode_bytes = request_media.largestImageEncodeBytes(&cfg);
    try std.testing.expect(cfg.expert_vision_encode_bytes > 1 << 30);
    try std.testing.expectEqual(kv + cfg.expert_vision_encode_bytes, bills.expertStreamingBesideBudgetBytes(&cfg));
    try std.testing.expect(!ssdBudgetAdmits(32 << 30, bills.expertStreamingBesideBudgetBytes(&cfg), (32 << 30) + kv));
}

test "the concurrency verdict: a resident GLM and qwen4 batch, an exclusive model queues, another MoE interleaves" {
    const glm = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    const qwen4 = model_mod.ModelConfig{ .model_type = "qwen4_exp", .full_attention_interval = 4, .num_experts = 8, .num_experts_per_tok = 2 };
    const moe = model_mod.ModelConfig{ .model_type = "qwen3_moe", .num_experts = 8, .num_experts_per_tok = 2 };
    try std.testing.expectEqual(sched_policy.BatchVerdict.ok, batchVerdictOf(&glm, false));
    try std.testing.expectEqual(sched_policy.BatchVerdict.arch, batchVerdictOf(&moe, false));
    try std.testing.expectEqual(sched_policy.BatchVerdict.exclusive, batchVerdictOf(&glm, true));
    try std.testing.expectEqual(sched_policy.BatchVerdict.ok, batchVerdictOf(&qwen4, false));
    try std.testing.expectEqual(sched_policy.BatchVerdict.arch, batchVerdictOf(null, false));
    try std.testing.expectEqualStrings("batched decode off; concurrent requests interleave serially", concurrencyClause(.arch));
    try std.testing.expectEqualStrings("batched decode off; concurrent requests queue, one at a time", concurrencyClause(.exclusive));
    try std.testing.expectEqualStrings("batched decode on", concurrencyClause(.ok));
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

pub fn readAllFromPair(fd: std.posix.fd_t, buf: []u8) []u8 {
    var got: usize = 0;
    while (got < buf.len) {
        const n = std.c.recv(fd, buf[got..].ptr, buf.len - got, 0);
        if (n <= 0) break;
        got += @intCast(n);
    }
    return buf[0..got];
}

