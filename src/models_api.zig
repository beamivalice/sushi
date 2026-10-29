const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const generate_mod = @import("generate.zig");
const mtp_acceptance_mod = @import("mtp_acceptance.zig");
const chat_mod = @import("chat.zig");
const model_mod = @import("model.zig");
const expert_quant = @import("expert_quant.zig");
const effort_mod = @import("effort.zig");
const log = @import("log.zig");
const scheduler_mod = @import("scheduler.zig");
const slot_admission = @import("slot_admission.zig");
const model_load = @import("model_load.zig");
const model_registry_mod = @import("model_registry.zig");
const model_discovery = @import("model_discovery.zig");
const model_settings = @import("model_settings.zig");
const metrics = @import("status.zig");
const qwen4_mod = @import("qwen4/qwen4_exp.zig");
const update_mod = @import("update.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const io_util = @import("io_util.zig");
const build_options = @import("build_options");
const mtp_plan = @import("mtp_plan.zig");
const nowSecs = io_util.nowSecs;
const testing = std.testing;
const server = @import("server.zig");
const http_conn = @import("http_conn.zig");
const request_policy = @import("request_policy.zig");
const Conn = http_conn.Conn;
const batchVerdictFor = server.batchVerdictFor;
const chatTemplateSupportsThinking = request_policy.chatTemplateSupportsThinking;
const computeMemoryContext = server.computeMemoryContext;
const defaultEnableMtpForStreaming = server.defaultEnableMtpForStreaming;
const forceMtpFor = server.forceMtpFor;
const getEffectiveContextLength = server.getEffectiveContextLength;
const jsonEscape = server.jsonEscape;
const kvCacheFor = server.kvCacheFor;
const modelQuantizationLabel = server.modelQuantizationLabel;
const mtpCapable = server.mtpCapable;
const mtpChoiceFor = server.mtpChoiceFor;
const nativeMeasuredMoeHead = server.nativeMeasuredMoeHead;
const optSamplingRecJson = request_policy.optSamplingRecJson;
const queryModel = server.queryModel;
const reasoningEffortsJson = request_policy.reasoningEffortsJson;
const resolvedPrefixCacheMem = server.resolvedPrefixCacheMem;
const rowSsdBudgetGb = server.rowSsdBudgetGb;
const sendErrorResponse = server.sendErrorResponse;
const sendResponse = server.sendResponse;

pub fn streamingStubMarker(architecture: []const u8, geometry_ok: bool, index_complete: bool) bool {
    return expert_quant.isExpertStreamingArch(architecture) and geometry_ok and index_complete;
}

/// A dense qwen4_exp checkpoint has no resident arm at all; a quantized pack
/// streams only when a budget asks it to. A MiMo EXL3 pack names no config-wide
/// width, so the width alone would read it as dense — `quantized_experts` is
/// the pack's own `expert_quant` block.
pub fn streamingRequiredMarker(architecture: []const u8, quant_bits: u32, quantized_experts: bool, geometry_ok: bool, index_complete: bool) bool {
    return streamingStubMarker(architecture, geometry_ok, index_complete) and quant_bits == 0 and !quantized_experts;
}

pub fn streamingRowPart(buf: []u8, candidate: bool, required: bool, ssd_budget_gb: u32) []const u8 {
    if (!candidate) return "";
    const head = if (required) ",\"streaming\":true,\"streaming_required\":true" else ",\"streaming\":true";
    if (ssd_budget_gb == 0) return head;
    return std.fmt.bufPrint(buf, "{s},\"ssd_budget_gb\":{d}", .{ head, ssd_budget_gb }) catch head;
}

/// Compute the maximum safe context length based on the GPU working-set
/// ceiling, the model's per-token footprint, and the hot prefix cache budget.
/// Linear in seq (per_tok × seq ≤ budget) — no seq² term, MLX's fused SDPA
/// tiles over seq and never materializes [heads, seq, seq]. See
/// `safeContextForBudget` for the steady-state reservation rationale.
/// `computeMemoryContext` capped by the checkpoint's `max_position_embeddings`.
/// This is the raw "largest context that fits" figure — reported in logs and
/// `/props` for diagnostics. The context we actually SERVE is `autoContextFor`,
/// which reserves headroom below it.
fn computeMaxSafeContext(config: *const model_mod.ModelConfig) u32 {
    const memory_ctx = computeMemoryContext(config);
    const max_pos = config.contextCap();
    return if (max_pos > 0) @min(memory_ctx, max_pos) else memory_ctx;
}

test "bf16 streaming marker is a property of the checkpoint, not of the launch flags" {
    try std.testing.expect(streamingStubMarker("qwen4_exp", true, true));
    try std.testing.expect(streamingStubMarker("qwen4_exp", true, true));
    try std.testing.expect(!streamingStubMarker("qwen4_exp", false, true));
    try std.testing.expect(!streamingStubMarker("qwen4_exp", true, false));
    try std.testing.expect(!streamingStubMarker("not_a_served_arch", true, true));
}

test "a quantized pack is streaming capable and only the dense one is streaming required" {
    try std.testing.expect(streamingStubMarker("qwen4_exp", true, true));
    try std.testing.expect(streamingRequiredMarker("qwen4_exp", 0, false, true, true));
    try std.testing.expect(!streamingRequiredMarker("qwen4_exp", 4, false, true, true));
    try std.testing.expect(!streamingRequiredMarker("qwen4_exp", 8, false, true, true));
    try std.testing.expect(!streamingRequiredMarker("not_a_served_arch", 0, false, true, true));
    // A pack with no `quantization` block names no config-wide width, so the
    // width alone cannot tell it from a dense checkpoint.
    try std.testing.expect(!streamingRequiredMarker("mimo_v2", 0, true, true, true));
    try std.testing.expect(streamingStubMarker("mimo_v2", true, true));
}

test "the streaming markers read the ONE arch gate: no other arch carries them" {
    const t = std.testing;
    var buf: [96]u8 = undefined;
    for ([_][]const u8{ "qwen4_exp_text", "not_a_served_arch", "gguf", "" }) |arch| {
        try t.expect(!streamingStubMarker(arch, true, true));
        try t.expect(!streamingRequiredMarker(arch, 0, false, true, true));
        try t.expectEqualStrings("", streamingRowPart(&buf, streamingStubMarker(arch, true, true), false, 60));
    }
    try t.expect(streamingStubMarker("qwen4_exp", true, true));
    try t.expectEqualStrings(",\"streaming\":true,\"ssd_budget_gb\":60", streamingRowPart(&buf, streamingStubMarker("qwen4_exp", true, true), false, 60));
}

test "an unloaded qwen4 stub is text-only only when the checkpoint must stream" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "chat_template.jinja", .data = "{{ messages }}" });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    const path = path_buf[0..path_len];
    const geometry = "\"hidden_size\":2560,\"num_hidden_layers\":48,\"num_experts\":512,\"num_experts_per_tok\":10,\"moe_intermediate_size\":640,\"max_position_embeddings\":262144";
    const quantized = try std.fmt.allocPrint(t.allocator, "{{\"model_type\":\"qwen4_exp\",\"quantization\":{{\"bits\":4,\"group_size\":64}},\"vision_config\":{{\"hidden_size\":1152}},{s}}}", .{geometry});
    defer t.allocator.free(quantized);
    const dense = try std.fmt.allocPrint(t.allocator, "{{\"model_type\":\"qwen4_exp\",\"vision_config\":{{\"hidden_size\":1152}},{s}}}", .{geometry});
    defer t.allocator.free(dense);

    var entry = model_registry_mod.LoadedModel{
        .allocator = t.allocator,
        .id = "stub",
        .path = path,
        .bytes_on_disk = 0,
        .arch_hint = "qwen4_exp",
        .config = null,
        .weights = null,
        .transformer = null,
        .tokenizer = null,
        .chat_config = null,
        .vision_encoder = null,
        .drafter_path = "",
        .drafter_block_size = 0,
        .prefix_cache = null,
        .refcount = std.atomic.Value(u32).init(0),
        .last_used_ns = 0,
        .bytes_resident = 0,
        .state = .unloaded,
        .error_name = null,
        .streaming_index_complete = true,
    };

    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = quantized });
    const quant_row = try renderModelEntry(t.allocator, io, &entry);
    defer t.allocator.free(quant_row);
    try t.expect(std.mem.indexOf(u8, quant_row, "\"streaming\":true") != null);
    try t.expect(std.mem.indexOf(u8, quant_row, "\"streaming_required\":true") == null);
    try t.expect(std.mem.indexOf(u8, quant_row, "\"vision\"") != null);
    try t.expect(std.mem.indexOf(u8, quant_row, "\"image\"") != null);

    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = dense });
    const dense_row = try renderModelEntry(t.allocator, io, &entry);
    defer t.allocator.free(dense_row);
    try t.expect(std.mem.indexOf(u8, dense_row, "\"streaming_required\":true") != null);
    try t.expect(std.mem.indexOf(u8, dense_row, "\"vision\"") == null);
    try t.expectEqual(@as(?usize, null), std.mem.indexOf(u8, dense_row, "\"image\""));
}

test "a quantized pack loaded resident advertises no streaming, the same pack streamed does" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);

    var cfg = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .quant_bits = 4,
        .num_hidden_layers = 48,
        .num_experts = 512,
        .num_experts_per_tok = 10,
        .hidden_size = 2560,
        .moe_intermediate_size = 640,
        .max_position_embeddings = 262144,
    };
    var chat_cfg = chat_mod.ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = t.allocator,
    };
    var entry = model_registry_mod.LoadedModel{
        .allocator = t.allocator,
        .id = "resident",
        .path = path_buf[0..path_len],
        .bytes_on_disk = 0,
        .arch_hint = "qwen4_exp",
        .config = &cfg,
        .weights = null,
        .transformer = null,
        .tokenizer = null,
        .chat_config = &chat_cfg,
        .vision_encoder = null,
        .drafter_path = "",
        .drafter_block_size = 0,
        .prefix_cache = null,
        .refcount = std.atomic.Value(u32).init(0),
        .last_used_ns = 0,
        .bytes_resident = 0,
        .state = .ready,
        .error_name = null,
    };
    const resident_row = try renderModelEntry(t.allocator, io, &entry);
    defer t.allocator.free(resident_row);
    try t.expectEqual(@as(?usize, null), std.mem.indexOf(u8, resident_row, "\"streaming\":true"));

    cfg.expert_streaming = true;
    const streamed_row = try renderModelEntry(t.allocator, io, &entry);
    defer t.allocator.free(streamed_row);
    try t.expect(std.mem.indexOf(u8, streamed_row, "\"streaming\":true") != null);
    try t.expectEqual(@as(?usize, null), std.mem.indexOf(u8, streamed_row, "\"streaming_required\":true"));

    // A ready row names the load's KV scheme and where it came from.
    const saved_kv = server.configured_kv_quant;
    defer server.configured_kv_quant = saved_kv;
    server.configured_kv_quant = null;
    const default_row = try renderModelEntry(t.allocator, io, &entry);
    defer t.allocator.free(default_row);
    try t.expect(std.mem.indexOf(u8, default_row, "\"kv_quant\":\"8\",\"kv_cache\":{\"scheme\":\"kv8\",\"source\":\"default\"}") != null);
    cfg.kv_quant_override = transformer_mod.KVQuantConfig.dense;
    const setting_row = try renderModelEntry(t.allocator, io, &entry);
    defer t.allocator.free(setting_row);
    try t.expect(std.mem.indexOf(u8, setting_row, "\"kv_quant\":\"off\",\"kv_cache\":{\"scheme\":\"off\",\"source\":\"model-settings.json\"}") != null);
}

test "a streaming row carries the marker and its ssd budget at top level" {
    var buf: [96]u8 = undefined;
    try std.testing.expectEqualStrings("", streamingRowPart(&buf, false, false, 0));
    try std.testing.expectEqualStrings("", streamingRowPart(&buf, false, false, 60));
    try std.testing.expectEqualStrings(",\"streaming\":true", streamingRowPart(&buf, true, false, 0));
    try std.testing.expectEqualStrings(",\"streaming\":true,\"ssd_budget_gb\":60", streamingRowPart(&buf, true, false, 60));
    try std.testing.expectEqualStrings(",\"streaming\":true,\"ssd_budget_gb\":4294967295", streamingRowPart(&buf, true, false, std.math.maxInt(u32)));
    try std.testing.expectEqualStrings(",\"streaming\":true,\"streaming_required\":true", streamingRowPart(&buf, true, true, 0));
    try std.testing.expectEqualStrings(",\"streaming\":true,\"streaming_required\":true,\"ssd_budget_gb\":60", streamingRowPart(&buf, true, true, 60));
}

/// Render the JSON metadata fragment for one entry. For `.ready` entries
/// Capability flags for a READY registry entry, kept as plain booleans so the
/// JSON assembly below is hermetically testable without a LoadedModel.
const ReadyCaps = struct {
    has_chat: bool = false,
    has_vision: bool = false,
    has_audio: bool = false,
    has_reasoning: bool = false,
};

/// `capabilities` JSON array for a ready model. Caller deinits.
fn readyCapsJson(allocator: std.mem.Allocator, c: ReadyCaps) !std.ArrayList(u8) {
    var caps = std.ArrayList(u8).empty;
    errdefer caps.deinit(allocator);
    try caps.append(allocator, '[');
    var n_caps: usize = 0;
    const append_cap = struct {
        fn call(a: std.mem.Allocator, b: *std.ArrayList(u8), n: *usize, name: []const u8) !void {
            if (n.* > 0) try b.append(a, ',');
            try b.append(a, '"');
            try b.appendSlice(a, name);
            try b.append(a, '"');
            n.* += 1;
        }
    }.call;
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "chat");
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "tool_use");
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "streaming");
    if (c.has_vision) try append_cap(allocator, &caps, &n_caps, "vision");
    if (c.has_audio) try append_cap(allocator, &caps, &n_caps, "audio");
    if (c.has_reasoning) try append_cap(allocator, &caps, &n_caps, "reasoning");
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "json_schema");
    try caps.append(allocator, ']');
    return caps;
}

/// pulls full capabilities/dimensions off the resident config/chat_config;
/// for non-ready entries renders a lightweight stub with state +
/// bytes_on_disk only. Returns an allocator-owned string; caller frees.
/// Called from `handleModels` per registry entry. Registry mutex must be
/// held by the caller (entry fields are read directly).
pub fn renderModelEntry(
    allocator: std.mem.Allocator,
    io: std.Io,
    entry: *LoadedModel,
) ![]u8 {
    if (entry.state == .ready and entry.config != null and entry.chat_config != null) {
        const config = entry.config.?;
        const chat_config = entry.chat_config.?;
        const quantization = try modelQuantizationLabel(allocator, config);
        defer allocator.free(quantization);
        const ctx_len = getEffectiveContextLength(config);
        const ctx_str = if (ctx_len > 0)
            try std.fmt.allocPrint(allocator, "{d}", .{ctx_len})
        else
            try std.fmt.allocPrint(allocator, "null", .{});
        defer allocator.free(ctx_str);

        const has_chat = chat_config.chat_template.len > 0;
        const has_vision = entry.vision_encoder != null;
        var caps = try readyCapsJson(allocator, .{
            .has_chat = has_chat,
            .has_vision = has_vision,
            .has_reasoning = has_chat and chatTemplateSupportsThinking(chat_config.chat_template),
        });
        defer caps.deinit(allocator);

        var mods = std.ArrayList(u8).empty;
        defer mods.deinit(allocator);
        try mods.appendSlice(allocator, "[\"text\"");
        if (has_vision) try mods.appendSlice(allocator, ",\"image\"");
        if (has_vision and config.video_token_id != 0) try mods.appendSlice(allocator, ",\"video\"");
        try mods.append(allocator, ']');

        const model_id: []const u8 = if (entry.id.len > 0) entry.id else config.model_type;
        const efforts_part = try reasoningEffortsJson(allocator, config);
        defer allocator.free(efforts_part);
        const drafter_loaded = entry.dflash != null;
        const mtp_loaded = entry.mtp != null;
        const drafter_path_json = if (drafter_loaded)
            try jsonEscape(allocator, entry.drafter_path)
        else
            try allocator.dupe(u8, "null");
        defer allocator.free(drafter_path_json);
        const bytes_on_disk_str = if (entry.bytes_on_disk) |b|
            try std.fmt.allocPrint(allocator, "{d}", .{b})
        else
            try allocator.dupe(u8, "null");
        defer allocator.free(bytes_on_disk_str);

        // Model-author sampling recommendations from generation_config.json,
        // surfaced so the Swift Settings UI can show "model recommends" pills
        // next to the per-request sampling sliders. `null` when the model
        // ships no value for that field.
        const gen_temp_str = try optSamplingRecJson(allocator, f32, config.gen_temperature);
        defer allocator.free(gen_temp_str);
        const gen_top_p_str = try optSamplingRecJson(allocator, f32, config.gen_top_p);
        defer allocator.free(gen_top_p_str);
        const gen_top_k_str = try optSamplingRecJson(allocator, u32, config.gen_top_k);
        defer allocator.free(gen_top_k_str);

        var streaming_buf: [96]u8 = undefined;
        const streaming_part = streamingRowPart(
            &streaming_buf,
            config.expert_streaming or config.expertStreamingRequired(),
            config.expertStreamingRequired(),
            rowSsdBudgetGb(config.expert_ssd_budget_bytes, config.ssd_budget_gb_override),
        );

        return std.fmt.allocPrint(allocator,
            \\{{"id":"{s}","object":"model","created":{d},"owned_by":"sushi","loaded":true,"state":"ready","bytes_resident":{d},"bytes_on_disk":{s},"context_length":{s},"max_model_len":{s}{s},"batched_decode":{s},"capabilities":{s}{s},"input_modalities":{s},"meta":{{"architecture":"{s}","engine":"{s}","vocab_size":{d},"hidden_size":{d},"num_layers":{d},"quantization":"{s}","context_length":{s},"model_max_tokens":{d},"is_moe":{s},"drafter_loaded":{s},"drafter_path":{s},"mtp_loaded":{s},"mtp_available":{s},"kv_quant":"{s}","kv_cache":{{"scheme":"{s}","source":"{s}"}},"gen_temperature":{s},"gen_top_p":{s},"gen_top_k":{s}}}}}
        , .{
            model_id,
            nowSecs(io),
            entry.bytes_resident,
            bytes_on_disk_str,
            // Top-level twins of meta.context_length: openai-models-list
            // discovery clients (oh-my-pi, vLLM-shaped readers) look for
            // row-level max_model_len/context_length and substitute their own
            // defaults when absent (issue #188).
            ctx_str,
            ctx_str,
            streaming_part,
            if (batchVerdictFor(entry) == .ok) "true" else "false",
            caps.items,
            efforts_part,
            mods.items,
            config.model_type,
            "mlx",
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
            quantization,
            ctx_str,
            config.max_position_embeddings,
            if (config.isMoe()) "true" else "false",
            if (drafter_loaded) "true" else "false",
            drafter_path_json,
            if (mtp_loaded) "true" else "false",
            if (mtp_loaded or model_discovery.readStubMeta(io, allocator, entry.path).has_mtp) "true" else "false",
            kvCacheFor(config).config.wireName(),
            kvCacheFor(config).label(),
            kvCacheFor(config).sourceName(),
            gen_temp_str,
            gen_top_p_str,
            gen_top_k_str,
        });
    }

    // Non-ready entry: stub form with state + bytes_on_disk.
    const state_str = switch (entry.state) {
        .unloaded => "unloaded",
        .loading => "loading",
        .ready => "ready",
        .error_state => "error",
        .evicting => "evicting",
    };
    const bytes_on_disk_str = if (entry.bytes_on_disk) |b|
        try std.fmt.allocPrint(allocator, "{d}", .{b})
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(bytes_on_disk_str);
    const err_part: []const u8 = if (entry.error_name) |name| blk: {
        // Inline escape to avoid double allocation; the names we emit
        // never contain quotes/backslashes (they're @errorName output).
        break :blk try std.fmt.allocPrint(
            allocator,
            ",\"error\":\"{s}\"",
            .{name},
        );
    } else &[_]u8{};
    defer if (err_part.len > 0) allocator.free(err_part);
    // Stub metadata sourced from config.json (+ chat-template presence) WITHOUT
    // faulting in weights: dims, context window, MoE-ness, and capabilities.
    // `/v1/models` isn't hot, so reading a small JSON per unloaded entry is fine.
    const sm = model_discovery.readStubMeta(io, allocator, entry.path);
    const streaming_geometry = sm.found and sm.hidden_size > 0 and sm.num_hidden_layers > 0 and
        sm.num_experts > 0 and sm.num_experts_per_tok > 0 and sm.moe_intermediate_size > 0;
    const streaming_precheck = expert_quant.isExpertStreamingArch(entry.arch_hint) and streaming_geometry;
    const streaming_candidate = streamingStubMarker(
        entry.arch_hint,
        streaming_geometry,
        streaming_precheck and entry.streaming_index_complete,
    );
    const streaming_required = streamingRequiredMarker(
        entry.arch_hint,
        sm.quant_bits,
        sm.quantized_experts,
        streaming_geometry,
        streaming_precheck and entry.streaming_index_complete,
    );
    var streaming_buf: [96]u8 = undefined;
    const streaming_part = streamingRowPart(&streaming_buf, streaming_candidate, streaming_required, if (streaming_candidate) blk: {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        var settings = model_settings.load(allocator, io, model_settings.defaultPath(&path_buf));
        defer settings.deinit();
        break :blk settings.lookup(entry.path).ssd_budget_gb orelse 0;
    } else 0);

    // Capabilities: chat/tool_use/streaming/json_schema (gated on chat-template
    // presence, the same rule the loaded path uses) plus "vision" when a vision
    // config is present. Reasoning/audio stay load-gated (they need the live
    // template / encoder), so an unloaded stub may under-report those two.
    const caps_part: []const u8 = blk: {
        if (!sm.found or !(sm.has_chat or sm.has_vision)) break :blk try allocator.dupe(u8, "");
        var b = std.ArrayList(u8).empty;
        errdefer b.deinit(allocator);
        try b.appendSlice(allocator, ",\"capabilities\":[");
        var n: usize = 0;
        const add = struct {
            fn call(a: std.mem.Allocator, list: *std.ArrayList(u8), cnt: *usize, name: []const u8) !void {
                if (cnt.* > 0) try list.append(a, ',');
                try list.append(a, '"');
                try list.appendSlice(a, name);
                try list.append(a, '"');
                cnt.* += 1;
            }
        }.call;
        if (sm.has_chat) {
            try add(allocator, &b, &n, "chat");
            try add(allocator, &b, &n, "tool_use");
            try add(allocator, &b, &n, "streaming");
            try add(allocator, &b, &n, "json_schema");
        }
        if (sm.has_vision and !streaming_required) try add(allocator, &b, &n, "vision");
        try b.append(allocator, ']');
        break :blk try b.toOwnedSlice(allocator);
    };
    defer allocator.free(caps_part);
    const efforts_part = try reasoningEffortsJson(allocator, &.{ .model_type = entry.arch_hint, .gen_enable_thinking = sm.enable_thinking });
    defer allocator.free(efforts_part);

    const mods_part: []const u8 = if (streaming_required)
        ",\"input_modalities\":[\"text\"]"
    else if (sm.found and sm.has_vision and sm.has_video)
        ",\"input_modalities\":[\"text\",\"image\",\"video\"]"
    else if (sm.found and sm.has_vision)
        ",\"input_modalities\":[\"text\",\"image\"]"
    else if (sm.found and sm.has_chat)
        ",\"input_modalities\":[\"text\"]"
    else
        "";

    const arch_part: []const u8 = if (entry.arch_hint.len > 0) blk: {
        // arch_hint comes from config.json's model_type via discovery — the
        // supported-type allowlist guarantees no JSON metacharacters.
        break :blk try std.fmt.allocPrint(allocator, "\"architecture\":\"{s}\",", .{entry.arch_hint});
    } else &[_]u8{};
    defer if (arch_part.len > 0) allocator.free(arch_part);

    const engine_part = "\"engine\":\"mlx\",";

    // Top-level twins of meta.context_length (issue #188): openai-models-list
    // discovery clients read row-level max_model_len/context_length. Unloaded
    // stubs advertise the architectural max — the same number meta carries here.
    const top_ctx_part: []const u8 = if (sm.found) blk: {
        break :blk try std.fmt.allocPrint(allocator, ",\"context_length\":{d},\"max_model_len\":{d}", .{
            sm.max_position_embeddings,
            sm.max_position_embeddings,
        });
    } else &[_]u8{};
    defer if (top_ctx_part.len > 0) allocator.free(top_ctx_part);

    // Dimensions/context/quant/MoE — emitted only when config.json was readable.
    const dims_part: []const u8 = if (sm.found) blk: {
        const quantization = try modelQuantizationLabel(allocator, &.{
            .quant_bits = sm.quant_bits,
            .expert_layout = if (sm.expert_quant_rate != null) .exl3_k4 else .bf16_fused,
            .expert_quant_rate = sm.expert_quant_rate orelse .{ .n = 64 },
        });
        defer allocator.free(quantization);
        break :blk try std.fmt.allocPrint(allocator, "\"vocab_size\":{d},\"hidden_size\":{d},\"num_layers\":{d},\"quantization\":\"{s}\",\"context_length\":{d},\"model_max_tokens\":{d},\"is_moe\":{s},\"mtp_available\":{s},", .{
            sm.vocab_size,
            sm.hidden_size,
            sm.num_hidden_layers,
            quantization,
            sm.max_position_embeddings,
            sm.max_position_embeddings,
            if (sm.is_moe) "true" else "false",
            if (sm.has_mtp) "true" else "false",
        });
    } else &[_]u8{};
    defer if (dims_part.len > 0) allocator.free(dims_part);

    return std.fmt.allocPrint(allocator,
        \\{{"id":"{s}","object":"model","created":0,"owned_by":"sushi","loaded":false,"state":"{s}","bytes_resident":0,"bytes_on_disk":{s}{s}{s}{s}{s}{s}{s},"meta":{{{s}{s}{s}"bytes_on_disk":{s}}}}}
    , .{ entry.id, state_str, bytes_on_disk_str, streaming_part, err_part, top_ctx_part, caps_part, efforts_part, mods_part, arch_part, engine_part, dims_part, bytes_on_disk_str });
}

pub fn handleModels(allocator: std.mem.Allocator, stream: *Conn) !void {
    // Plan 05 Phase E: emit every registry entry (loaded + unloaded), not
    // just the default model + flat discovery list. Default model is sorted
    // first so single-model clients reading `data[0]` continue to work.
    // The mutex is held for the whole render so eviction can't fire under
    // us; each rendered entry is at most a few hundred bytes. This handler
    // does NOT route through `scheduler.ensureLoaded` — listing metadata
    // shouldn't trigger a cold load of the default model.
    const registry = server.global_registry orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503);
        return;
    };

    var entries_buf = std.ArrayList(u8).empty;
    defer entries_buf.deinit(allocator);

    // Collect entries while holding the mutex so iteration is safe.
    // Sort: default first, then by last_used_ns desc.
    var ordered = std.ArrayList(*LoadedModel).empty;
    defer ordered.deinit(allocator);
    {
        registry.mutex.lockUncancelable(stream.io);
        defer registry.mutex.unlock(stream.io);
        try ordered.ensureTotalCapacity(allocator, registry.entries.count());
        var it = registry.entries.valueIterator();
        while (it.next()) |entry_ptr| ordered.appendAssumeCapacity(entry_ptr.*);
        const default_id = registry.default_id;
        const Cmp = struct {
            fn lt(ctx: []const u8, a: *LoadedModel, b: *LoadedModel) bool {
                const a_def = std.mem.eql(u8, a.id, ctx);
                const b_def = std.mem.eql(u8, b.id, ctx);
                if (a_def != b_def) return a_def;
                return a.last_used_ns > b.last_used_ns;
            }
        };
        std.sort.pdq(*LoadedModel, ordered.items, default_id, Cmp.lt);

        for (ordered.items) |entry| {
            if (entries_buf.items.len > 0) try entries_buf.append(allocator, ',');
            const json = try renderModelEntry(allocator, stream.io, entry);
            defer allocator.free(json);
            try entries_buf.appendSlice(allocator, json);
        }
    }

    const body = try std.fmt.allocPrint(allocator,
        \\{{"object":"list","data":[{s}]}}
    , .{entries_buf.items});
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// `POST /v1/models/rescan`: re-walk the boot roots and register stubs for
/// new dirs and clear failed loads (`ModelRegistry.rescan`). Answers `{"added":N}`.
pub fn handleModelsRescan(allocator: std.mem.Allocator, stream: *Conn) !void {
    const registry = server.global_registry orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503);
        return;
    };
    const added = registry.rescan() catch |err| {
        log.warn("/v1/models/rescan failed: {s}\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "internal_error", "Model rescan failed", 500);
        return;
    };
    if (added > 0) log.info("[registry] rescan added {d} model(s)\n", .{added});
    const body = try std.fmt.allocPrint(allocator, "{{\"added\":{d}}}", .{added});
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// Pure body-builder for `GET /props`. Split out from `handleProps` so it
/// can be unit-tested without standing up a real `LoadedModel`. Caller owns
/// the returned slice.
///
/// Note: the `chat_template` field that upstream llama-server exposes here
/// is intentionally omitted. Our Swift app never read it, and stock chat
/// templates are 10–100 KB of Jinja — that turned every /props poll into
/// wasted bandwidth. Capabilities still come from `/v1/models` per entry;
/// clients that genuinely need the template can still get it from the
/// model's `config.json` / `chat_template.jinja` on disk, or by rendering
/// through `/v1/chat/completions` server-side.
fn renderPropsBody(
    allocator: std.mem.Allocator,
    config: *const model_mod.ModelConfig,
    ctx_str: []const u8,
    active_mem: usize,
    peak_mem: usize,
    available_mem: u64,
    safe_ctx: u32,
    cache_mem: usize,
    /// KV of the hot prefix caches + live slots.
    kv_cache_mem: u64,
    /// Leading-comma JSON fragments spliced before the root close (the
    /// qwen4 n-gram warm object, ...). Concatenated by the handler.
    extra_json: []const u8,
) ![]u8 {
    // `available_bytes` is free SYSTEM RAM, computed with the SAME formula the
    // model-load pre-flight uses (`metrics.getAvailableMemBytes`), so the tray's
    // "Free RAM" line can never drift from the number that gates a load. Distinct
    // axis from `active_bytes` (the MLX GPU-allocator footprint).
    //
    // `cache_bytes` is MLX's reclaimable buffer pool — memory the process HOLDS
    // but is not using. It is a third axis again, and its absence is why #110
    // was invisible: the panel read 19.6 GB of `active_bytes` while the process
    // sat at 81.4 GB, and nothing we served named the other 61.
    return std.fmt.allocPrint(allocator,
        \\{{"default_generation_settings":{{"model":"{s}","n_ctx":{s}}},"total_slots":1,"model_info":{{"vocab_size":{d},"hidden_size":{d},"num_hidden_layers":{d},"num_attention_heads":{d},"num_key_value_heads":{d},"head_dim":{d},"quantization_bits":{d},"quantization_group_size":{d},"max_position_embeddings":{d}}},"memory":{{"active_bytes":{d},"peak_bytes":{d},"available_bytes":{d},"max_safe_context":{d},"cache_bytes":{d},"kv_cache_bytes":{d},"owned_bytes":{d}}}{s}}}
    , .{
        config.model_type,              ctx_str,
        config.vocab_size,              config.hidden_size,
        config.num_hidden_layers,       config.num_attention_heads,
        config.num_key_value_heads,     config.head_dim,
        config.quant_bits,              config.quant_group_size,
        config.max_position_embeddings, active_mem,
        peak_mem,                       available_mem,
        safe_ctx,                       cache_mem,
        kv_cache_mem,                   model_mod.ownedBytesTotal(),
        extra_json,
    });
}

/// The /props "batching" object: whether the loaded model rides the batched
/// decode kernel, and why not when it does not.
fn batchingPropsJson(allocator: std.mem.Allocator, why: slot_admission.BatchVerdict, max_group: usize) ![]u8 {
    return std.fmt.allocPrint(allocator, ",\"batching\":{{\"supported\":{s},\"reason\":\"{s}\",\"max_group\":{d}}}", .{
        if (why == .ok) "true" else "false",
        @tagName(why),
        max_group,
    });
}

/// The /props "template_fallbacks" count: renders where the model's own chat template raised and the
/// generic format answered instead. Above zero the prompts are degraded, never a normal state.
fn templateFallbacksPropsJson(allocator: std.mem.Allocator, count: u64) ![]u8 {
    return std.fmt.allocPrint(allocator, ",\"template_fallbacks\":{d}", .{count});
}

/// The /props "ngram_warm" object; absent when no table is warming.
fn ngramWarmPropsJson(allocator: std.mem.Allocator, bytes: u64, total: u64) ![]u8 {
    if (total == 0) return allocator.dupe(u8, "");
    return std.fmt.allocPrint(allocator, ",\"ngram_warm\":{{\"bytes\":{d},\"total\":{d}}}", .{ bytes, total });
}

/// Effective per-model serving settings, resolved the way a request that sends no overrides sees them.
const PropsSettings = struct {
    engine: []const u8,
    kv_quant: []const u8,
    kv_cache: transformer_mod.KvCacheChoice = .{ .config = transformer_mod.KVQuantConfig.engine_default, .source = .default },
    prefill_chunk: usize,
    prefill_decode_share: f32 = 0,
    mtp_loaded: bool,
    mtp_default_on: bool,
    mtp_choice: model_settings.MtpChoice = .{ .on = true, .source = .default },
    mtp_acceptance: mtp_acceptance_mod.Mode,
    mtp_acceptance_source: []const u8 = "default",
    mtp_greedy_tail: model_settings.Pick(bool) = .{ .value = false, .source = .default },
    /// 0 = auto.
    mtp_depth: u32,
    mtp_adaptive: bool,
    /// 0 = no ceiling.
    max_mtp_ctx: u32,
    drafter: []const u8,
    max_concurrent: u32,
    prefix_cache_mem_bytes: u64,
    prefix_cache_disk_bytes: u64,
    /// The SSD tier's bytes in use and entry count, as the inference thread last published them.
    prefix_cache_disk_used_bytes: u64 = 0,
    prefix_cache_disk_entries: u64 = 0,
    prefix_cache_ram_enabled: bool = true,
    /// The byte figures are a streamed load's (0 resident or off).
    vision: struct { loaded: bool = false, source: []const u8 = "default", tower_bytes: u64 = 0, encode_bytes: u64 = 0 } = .{},
};

fn propsSettingsFor(lm: *LoadedModel) PropsSettings {
    return mlxPropsSettings(lm);
}

fn mlxPropsSettings(lm: *LoadedModel) PropsSettings {
    const config = lm.config.?;
    const kv_cache = kvCacheFor(config);
    const kv = kv_cache.config;
    const acceptance = mtp_plan.mtpAcceptanceFor(config.mtp_acceptance_override);
    // This model's own tier, the four numbers from one publish.
    const disk = lm.disk_stats.snapshot();
    return .{
        .engine = "mlx",
        .kv_quant = kv.wireName(),
        .kv_cache = kv_cache,
        .prefill_chunk = generate_mod.prefill_chunk_override,
        .prefill_decode_share = scheduler_mod.prefillDecodeShare(),
        .mtp_loaded = mtpCapable(lm),
        .mtp_default_on = defaultEnableMtpForStreaming(lm.mtp != null, config.isMoe(), forceMtpFor(config), nativeMeasuredMoeHead(lm), model_mod.isServedArch(config.model_type), config.expert_streaming and !config.stream_mtp_head),
        .mtp_choice = mtpChoiceFor(config),
        .mtp_acceptance = acceptance.value,
        .mtp_acceptance_source = model_settings.sourceLabel(acceptance.source, model_settings.acceptanceFlagName(acceptance.value)),
        .mtp_greedy_tail = mtp_plan.mtpGreedyTailFor(config.mtp_greedy_tail_override),
        .mtp_depth = lm.mtp_depth,
        .mtp_adaptive = generate_mod.Generator.mtpAdaptiveEnabled(),
        .max_mtp_ctx = generate_mod.max_mtp_ctx,
        .drafter = if (lm.dflash != null) "dflash" else "none",
        .max_concurrent = server.max_concurrent,
        .prefix_cache_mem_bytes = resolvedPrefixCacheMem(),
        .prefix_cache_ram_enabled = server.prefix_cache_capacity > 0 and server.prefix_cache_ram_enabled,
        .prefix_cache_disk_bytes = disk.budget,
        .prefix_cache_disk_used_bytes = disk.used,
        .prefix_cache_disk_entries = disk.entries,
        .vision = .{
            .loaded = lm.vision_encoder != null,
            .source = model_load.visionChoiceFor(config, config.expert_streaming).sourceName(),
            .tower_bytes = config.expert_vision_tower_bytes,
            .encode_bytes = config.expert_vision_encode_bytes,
        },
    };
}

/// The /props "settings" object: leading-comma fragment spliced before the props root close.
pub fn settingsPropsJson(allocator: std.mem.Allocator, st: PropsSettings) ![]u8 {
    var param_buf: [32]u8 = undefined;
    const param: []const u8 = switch (st.mtp_acceptance) {
        .exact => "null",
        .typical => |t| try std.fmt.bufPrint(&param_buf, "{d}", .{t.delta}),
        .tokenv3 => |a| try std.fmt.bufPrint(&param_buf, "{d}", .{a}),
    };
    var vision_buf: [160]u8 = undefined;
    const vision = try std.fmt.bufPrint(&vision_buf, "{{\"loaded\":{},\"source\":\"{s}\",\"streamed_tower_bytes\":{d},\"streamed_encode_bytes\":{d}}}", .{ st.vision.loaded, st.vision.source, st.vision.tower_bytes, st.vision.encode_bytes });
    return std.fmt.allocPrint(allocator, ",\"settings\":{{\"version\":\"{s}\",\"engine\":\"{s}\",\"kv_quant\":\"{s}\",\"kv_cache\":{{\"scheme\":\"{s}\",\"source\":\"{s}\"}},\"prefill_chunk\":{d},\"mtp\":{{\"loaded\":{},\"default_on\":{},\"source\":\"{s}\",\"acceptance_source\":\"{s}\",\"acceptance\":\"{s}\",\"acceptance_param\":{s},\"greedy_tail\":{},\"greedy_tail_source\":\"{s}\",\"depth\":{d},\"adaptive\":{},\"max_ctx\":{d}}},\"drafter\":\"{s}\",\"max_concurrent\":{d},\"prefill_decode_share\":{d},\"prefix_cache\":{{\"ram_enabled\":{},\"mem_bytes\":{d},\"disk_bytes\":{d},\"disk_used_bytes\":{d},\"disk_entries\":{d}}},\"vision\":{s}}}", .{
        build_options.version,                      st.engine,
        st.kv_quant,                                st.kv_cache.label(),
        st.kv_cache.sourceName(),
        st.prefill_chunk,
        st.mtp_loaded,                              st.mtp_default_on,
        st.mtp_choice.sourceName(),                 st.mtp_acceptance_source,
        mtp_acceptance_mod.name(st.mtp_acceptance), param,
        st.mtp_greedy_tail.value,                   model_settings.sourceLabel(st.mtp_greedy_tail.source, "--mtp-greedy-tail"),
        st.mtp_depth,                               st.mtp_adaptive,
        st.max_mtp_ctx,                             st.drafter,
        st.max_concurrent,                          st.prefill_decode_share,
        st.prefix_cache_ram_enabled,                st.prefix_cache_mem_bytes,
        st.prefix_cache_disk_bytes,                 st.prefix_cache_disk_used_bytes,
        st.prefix_cache_disk_entries,               vision,
    });
}

pub fn handleProps(allocator: std.mem.Allocator, stream: *Conn, lm: *LoadedModel) !void {
    const config = lm.config.?;
    const ctx_len = getEffectiveContextLength(config);
    const ctx_str = if (ctx_len > 0) blk: {
        break :blk try std.fmt.allocPrint(allocator, "{d}", .{ctx_len});
    } else try std.fmt.allocPrint(allocator, "0", .{});
    defer allocator.free(ctx_str);

    const safe_ctx = computeMaxSafeContext(config);

    // Query memory usage from mlx's allocator counters, plus its reclaimable
    // buffer pool.
    var active_mem: usize = 0;
    var peak_mem: usize = 0;
    var cache_mem: usize = 0;
    _ = mlx.mlx_get_cache_memory(&cache_mem);
    _ = mlx.mlx_get_active_memory(&active_mem);
    _ = mlx.mlx_get_peak_memory(&peak_mem);

    // Free system RAM — same calc as the model-load pre-flight, so the app's
    // "Free RAM" line stays in lockstep with what gates a load.
    const available_mem = metrics.getAvailableMemBytes();

    // qwen4 n-gram table warm (F8) — the two module-level atomics are zero
    // whenever no table is warming, so the object is absent off qwen4_exp.
    const ngram_json = try ngramWarmPropsJson(allocator, qwen4_mod.live_warm_bytes.load(.acquire), qwen4_mod.live_warm_total.load(.acquire));
    defer allocator.free(ngram_json);
    const batching_json = try batchingPropsJson(allocator, batchVerdictFor(lm), if (lm.config) |c| slot_admission.batchGroupCap(c) else slot_admission.MAX_BATCH_GROUP);
    defer allocator.free(batching_json);
    const settings_json = try settingsPropsJson(allocator, propsSettingsFor(lm));
    defer allocator.free(settings_json);
    const update_json = try update_mod.propsJson(allocator);
    defer allocator.free(update_json);
    const fallbacks_json = try templateFallbacksPropsJson(allocator, chat_mod.template_fallbacks.load(.monotonic));
    defer allocator.free(fallbacks_json);
    const extra_json = try std.fmt.allocPrint(allocator, "{s}{s}{s}{s}{s}", .{ ngram_json, batching_json, settings_json, update_json, fallbacks_json });
    defer allocator.free(extra_json);

    const kv_cache_mem: u64 = if (server.global_scheduler) |sch|
        sch.resident_hot_cache_bytes.load(.monotonic) + sch.resident_live_kv_bytes.load(.monotonic)
    else
        0;
    const body = try renderPropsBody(allocator, config, ctx_str, active_mem, peak_mem, available_mem, safe_ctx, cache_mem, kv_cache_mem, extra_json);
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// Memory-only `/props` for a boot with no default chat model (headless
/// media-gen serving). Same `memory` object shape as `renderPropsBody` —
/// `MemoryInfo.parse` client-side reads only that key — with the model
/// fields omitted (there is no model config to describe).
pub fn handlePropsNoModel(allocator: std.mem.Allocator, stream: *Conn) !void {
    var active_mem: usize = 0;
    var peak_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    _ = mlx.mlx_get_peak_memory(&peak_mem);
    const available_mem = metrics.getAvailableMemBytes();
    const update_json = try update_mod.propsJson(allocator);
    defer allocator.free(update_json);
    const body = try std.fmt.allocPrint(allocator,
        \\{{"total_slots":1,"memory":{{"active_bytes":{d},"peak_bytes":{d},"available_bytes":{d},"max_safe_context":0}},"batching":{{"supported":false,"reason":"no_model","max_group":{d}}}{s}}}
    , .{ active_mem, peak_mem, available_mem, slot_admission.MAX_BATCH_GROUP, update_json });
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

test "a video-capable vision model advertises video beside image, loaded or not" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    var chat_cfg = chat_mod.ChatConfig{ .chat_template = "{{ messages }}", .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = t.allocator };
    const Case = struct { config_json: []const u8, video_token_id: u32, want: []const u8 };
    for ([_]Case{
        .{ .config_json = "{\"model_type\":\"qwen4_exp\",\"vision_config\":{},\"video_token_id\":151656}", .video_token_id = 151656, .want = "\"input_modalities\":[\"text\",\"image\",\"video\"]" },
        .{ .config_json = "{\"model_type\":\"qwen4_exp\",\"vision_config\":{}}", .video_token_id = 0, .want = "\"input_modalities\":[\"text\",\"image\"]" },
    }) |c| {
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = c.config_json });
        var cfg = model_mod.ModelConfig{ .model_type = "qwen4_exp", .video_token_id = c.video_token_id };
        var entry = model_registry_mod.LoadedModel{
            .allocator = t.allocator,
            .id = "m",
            .path = path,
            .bytes_on_disk = 0,
            .arch_hint = "qwen4_exp",
            .config = &cfg,
            .weights = null,
            .transformer = null,
            .tokenizer = null,
            .chat_config = &chat_cfg,
            .vision_encoder = @ptrFromInt(0x1000),
            .drafter_path = "",
            .drafter_block_size = 0,
            .prefix_cache = null,
            .refcount = std.atomic.Value(u32).init(0),
            .last_used_ns = 0,
            .bytes_resident = 0,
            .state = .ready,
            .error_name = null,
        };
        const loaded = try renderModelEntry(t.allocator, io, &entry);
        defer t.allocator.free(loaded);
        try t.expect(std.mem.indexOf(u8, loaded, c.want) != null);
        entry.state = .unloaded;
        entry.config = null;
        entry.chat_config = null;
        entry.vision_encoder = null;
        const stub = try renderModelEntry(t.allocator, io, &entry);
        defer t.allocator.free(stub);
        try t.expect(std.mem.indexOf(u8, stub, c.want) != null);
    }
}

// --- /props payload regression ------------------------------------------------
//
// The `chat_template` field used to be emitted by /props for llama.cpp
// server-compat clients. Our own Swift app never read it, and stock chat
// templates are 10–100 KB of Jinja — that's wasted bandwidth on every poll.
// `renderPropsBody` is the pure body-builder behind `handleProps`; these
// tests pin the schema so a future "let's add it back" can't slip through
// without flipping these assertions intentionally.

test "renderPropsBody omits chat_template" {
    var config = model_mod.ModelConfig{};
    config.vocab_size = 32000;
    config.hidden_size = 4096;
    config.num_hidden_layers = 32;
    config.num_attention_heads = 32;
    config.num_key_value_heads = 8;
    config.head_dim = 128;
    config.quant_bits = 4;
    config.quant_group_size = 64;
    config.max_position_embeddings = 8192;
    config.model_type = "not_a_served_arch";

    const body = try renderPropsBody(testing.allocator, &config, "4096", 1234, 5678, 9_000_000_000, 16384, 4321, 0, "");
    defer testing.allocator.free(body);

    try testing.expect(std.mem.indexOf(u8, body, "\"chat_template\"") == null);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    parsed.deinit();
}

test "queryModel: GET /props?model=<id> routes to that model, percent-decoded" {
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings("mlx-community/Qwen3.5-0.8B", queryModel(&buf, "/props?x=1&model=mlx-community%2FQwen3.5-0.8B").?);
    try testing.expect(queryModel(&buf, "/props") == null);
    try testing.expect(queryModel(&buf, "/props?models=a") == null);
}

test "settingsPropsJson: /props names the vision choice and a streamed tower's bill" {
    const frag = try settingsPropsJson(testing.allocator, .{ .engine = "mlx", .kv_quant = "8", .prefill_chunk = 2048, .mtp_loaded = false, .mtp_default_on = false, .mtp_acceptance = .exact, .mtp_depth = 0, .mtp_adaptive = false, .max_mtp_ctx = 0, .drafter = "none", .max_concurrent = 1, .prefix_cache_mem_bytes = 0, .prefix_cache_disk_bytes = 0, .vision = .{ .loaded = true, .source = "--vision", .tower_bytes = 493_389_824, .encode_bytes = 5_000_000_000 } });
    defer testing.allocator.free(frag);
    var config = model_mod.ModelConfig{};
    const body = try renderPropsBody(testing.allocator, &config, "4096", 1, 2, 3, 4, 5, 0, frag);
    defer testing.allocator.free(body);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const vision = parsed.value.object.get("settings").?.object.get("vision").?.object;
    try testing.expect(vision.get("loaded").?.bool);
    try testing.expectEqualStrings("--vision", vision.get("source").?.string);
    try testing.expectEqual(@as(i64, 493_389_824), vision.get("streamed_tower_bytes").?.integer);
    try testing.expectEqual(@as(i64, 5_000_000_000), vision.get("streamed_encode_bytes").?.integer);
}

test "settingsPropsJson: /props names the effective serving settings a benchmark ran under" {
    const frag = try settingsPropsJson(testing.allocator, .{
        .engine = "mlx",
        .kv_quant = "4",
        .kv_cache = transformer_mod.KvCacheChoice.resolve(transformer_mod.KVQuantConfig.affine(4), transformer_mod.KVQuantConfig.engine_default, false),
        .prefill_chunk = 8192,
        .mtp_loaded = true,
        .mtp_default_on = true,
        .mtp_choice = model_settings.MtpChoice.resolve(true, false, true),
        .mtp_acceptance = .{ .tokenv3 = 0.95 },
        .mtp_acceptance_source = "--mtp-tokenv3",
        .mtp_depth = 0,
        .mtp_adaptive = true,
        .max_mtp_ctx = 32768,
        .drafter = "none",
        .max_concurrent = 4,
        .prefix_cache_mem_bytes = 2048,
        .prefix_cache_disk_bytes = 0,
    });
    defer testing.allocator.free(frag);
    var config = model_mod.ModelConfig{};
    config.model_type = "not_a_served_arch";
    const body = try renderPropsBody(testing.allocator, &config, "4096", 1, 2, 3, 4, 5, 0, frag);
    defer testing.allocator.free(body);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const st = (parsed.value.object.get("settings") orelse return error.MissingSettings).object;
    try testing.expectEqualStrings(build_options.version, st.get("version").?.string);
    try testing.expectEqualStrings("4", st.get("kv_quant").?.string);
    const kv_cache = st.get("kv_cache").?.object;
    try testing.expectEqualStrings("kv4", kv_cache.get("scheme").?.string);
    try testing.expectEqualStrings("model-settings.json", kv_cache.get("source").?.string);
    try testing.expect(st.get("kv_attn_mode") == null);
    try testing.expect(st.get("decode_attn_quant") == null);
    const mtp = st.get("mtp").?.object;
    try testing.expect(mtp.get("default_on").?.bool);
    try testing.expectEqualStrings("--mtp", mtp.get("source").?.string);
    try testing.expectEqualStrings("--mtp-tokenv3", mtp.get("acceptance_source").?.string);
    try testing.expectEqualStrings("tokenv3", mtp.get("acceptance").?.string);
    try testing.expectApproxEqAbs(@as(f64, 0.95), mtp.get("acceptance_param").?.float, 1e-6);
    try testing.expectEqual(@as(i64, 32768), mtp.get("max_ctx").?.integer);
    try testing.expect(st.get("pld") == null);
    try testing.expectEqual(@as(i64, 4), st.get("max_concurrent").?.integer);
    try testing.expect(st.get("prefix_cache").?.object.get("ram_enabled").?.bool);

    const used = try settingsPropsJson(testing.allocator, .{ .engine = "mlx", .kv_quant = "8", .prefill_chunk = 4096, .mtp_loaded = false, .mtp_default_on = false, .mtp_acceptance = .exact, .mtp_depth = 0, .mtp_adaptive = false, .max_mtp_ctx = 0, .drafter = "none", .max_concurrent = 1, .prefix_cache_mem_bytes = 0, .prefix_cache_disk_bytes = 20 << 30, .prefix_cache_disk_used_bytes = 3 << 30, .prefix_cache_disk_entries = 5, .prefix_cache_ram_enabled = false });
    defer testing.allocator.free(used);
    var up = try std.json.parseFromSlice(std.json.Value, testing.allocator, used[",\"settings\":".len..], .{});
    defer up.deinit();
    const pc = up.value.object.get("prefix_cache").?.object;
    try testing.expectEqual(@as(i64, 20 << 30), pc.get("disk_bytes").?.integer);
    try testing.expectEqual(@as(i64, 3 << 30), pc.get("disk_used_bytes").?.integer);
    try testing.expectEqual(@as(i64, 5), pc.get("disk_entries").?.integer);

    const exact = try settingsPropsJson(testing.allocator, .{ .engine = "mlx", .kv_quant = "q4", .prefill_chunk = 4096, .mtp_loaded = false, .mtp_default_on = false, .mtp_acceptance = .exact, .mtp_depth = 3, .mtp_adaptive = false, .max_mtp_ctx = 0, .drafter = "dflash", .max_concurrent = 1, .prefix_cache_mem_bytes = 0, .prefix_cache_disk_bytes = 4096, .prefix_cache_ram_enabled = false });
    defer testing.allocator.free(exact);
    var ep = try std.json.parseFromSlice(std.json.Value, testing.allocator, exact[",\"settings\":".len..], .{});
    defer ep.deinit();
    try testing.expect(ep.value.object.get("mtp").?.object.get("acceptance_param").? == .null);
    try testing.expect(!ep.value.object.get("prefix_cache").?.object.get("ram_enabled").?.bool);
    try testing.expectEqual(@as(i64, 0), ep.value.object.get("prefix_cache").?.object.get("mem_bytes").?.integer);
}

test "templateFallbacksPropsJson: /props counts renders that fell back to the generic format" {
    const frag = try templateFallbacksPropsJson(testing.allocator, 3);
    defer testing.allocator.free(frag);
    var config = model_mod.ModelConfig{};
    const body = try renderPropsBody(testing.allocator, &config, "4096", 1, 2, 3, 4, 5, 0, frag);
    defer testing.allocator.free(body);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(i64, 3), parsed.value.object.get("template_fallbacks").?.integer);
}

test "ngramWarmPropsJson: /props names how far the qwen4 ngram warm has got" {
    const none = try ngramWarmPropsJson(testing.allocator, 0, 0);
    defer testing.allocator.free(none);
    try testing.expectEqualStrings("", none);

    const frag = try ngramWarmPropsJson(testing.allocator, 17_179_869_184, 54_975_581_388);
    defer testing.allocator.free(frag);
    try testing.expectEqualStrings(",\"ngram_warm\":{\"bytes\":17179869184,\"total\":54975581388}", frag);

    var config = model_mod.ModelConfig{};
    config.model_type = "qwen4_exp";
    const body = try renderPropsBody(testing.allocator, &config, "4096", 1, 2, 3, 4, 5, 0, frag);
    defer testing.allocator.free(body);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const w = (parsed.value.object.get("ngram_warm") orelse return error.MissingNgramWarm).object;
    try testing.expectEqual(@as(i64, 17_179_869_184), w.get("bytes").?.integer);
    try testing.expectEqual(@as(i64, 54_975_581_388), w.get("total").?.integer);

}

test "renderPropsBody keeps fields the Swift app + integration tests rely on" {
    var config = model_mod.ModelConfig{};
    config.model_type = "not_a_served_arch";
    config.vocab_size = 32000;
    config.hidden_size = 4096;
    config.num_hidden_layers = 32;
    config.num_attention_heads = 32;
    config.num_key_value_heads = 8;
    config.head_dim = 128;
    config.quant_bits = 4;
    config.quant_group_size = 64;
    config.max_position_embeddings = 8192;

    const body = try renderPropsBody(testing.allocator, &config, "4096", 1234, 5678, 9_000_000_000, 16384, 4321, 8888, "");
    defer testing.allocator.free(body);

    // Hit every field a known consumer reads.
    try testing.expect(std.mem.indexOf(u8, body, "\"n_ctx\":4096") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"total_slots\":1") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"model_info\"") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"memory\"") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"active_bytes\":1234") != null); // Swift fetchProps
    try testing.expect(std.mem.indexOf(u8, body, "\"peak_bytes\":5678") != null); // Swift fetchProps
    try testing.expect(std.mem.indexOf(u8, body, "\"available_bytes\":9000000000") != null); // Swift fetchProps (Free RAM line)
    try testing.expect(std.mem.indexOf(u8, body, "\"max_safe_context\":16384") != null); // Swift fetchProps
    // #110: the panel showed 19.6 GB (active) while the process held 81 GB.
    // The missing 61 GB was MLX's reclaimable buffer pool, which nothing we
    // expose reported — so the bug was invisible from every surface.
    try testing.expect(std.mem.indexOf(u8, body, "\"cache_bytes\":4321") != null); // Swift fetchProps
    try testing.expect(std.mem.indexOf(u8, body, "\"kv_cache_bytes\":8888") != null); // llmtop
    try testing.expect(std.mem.indexOf(u8, body, "\"owned_bytes\":") != null);
}

test "/v1/models rows, loaded or not, name each model's effort words and the one a silent request runs at" {
    const t = std.testing;
    const io = t.io;
    const saved = effort_mod.think_effort_flag;
    defer effort_mod.think_effort_flag = saved;
    effort_mod.think_effort_flag = null;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "chat_template.jinja", .data = "{{ messages }}" });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    var chat_cfg = chat_mod.ChatConfig{ .chat_template = "{{ messages }}", .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = t.allocator };
    const Case = struct { arch: []const u8, gen: []const u8, want: []const u8 };
    for ([_]Case{
        .{ .arch = "qwen4_exp", .gen = "{}", .want = ",\"reasoning_efforts\":[\"off\",\"low\",\"medium\",\"xhigh\"],\"reasoning_effort_budgets\":{\"low\":2048,\"medium\":8192},\"default_reasoning_effort\":\"medium\"" },
        .{ .arch = "qwen4_exp", .gen = "{\"default_chat_template_kwargs\":{\"enable_thinking\":true}}", .want = ",\"default_reasoning_effort\":\"medium\"" },
        .{ .arch = "qwen4_exp", .gen = "{\"default_chat_template_kwargs\":{\"enable_thinking\":false}}", .want = ",\"default_reasoning_effort\":\"off\"" },
        .{ .arch = "mimo_v2", .gen = "{}", .want = ",\"reasoning_efforts\":[\"off\",\"on\"],\"reasoning_effort_budgets\":{},\"default_reasoning_effort\":\"on\"" },
        .{ .arch = "glm5_next", .gen = "{}", .want = ",\"reasoning_efforts\":[\"low\",\"high\",\"max\"],\"reasoning_effort_budgets\":{},\"default_reasoning_effort\":\"high\"" },
        .{ .arch = "deepseek_v41", .gen = "{}", .want = ",\"reasoning_efforts\":[\"off\",\"low\",\"high\",\"max\"],\"reasoning_effort_budgets\":{},\"default_reasoning_effort\":\"high\"" },
    }) |c| {
        const config_json = try std.fmt.allocPrint(t.allocator, "{{\"model_type\":\"{s}\"}}", .{c.arch});
        defer t.allocator.free(config_json);
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config_json });
        try tmp.dir.writeFile(io, .{ .sub_path = "generation_config.json", .data = c.gen });
        var cfg = model_mod.ModelConfig{ .model_type = c.arch, .gen_enable_thinking = model_mod.parseGenerationDefaultsFromJson(c.gen).enable_thinking };
        var entry = model_registry_mod.LoadedModel{
            .allocator = t.allocator,
            .id = "m",
            .path = path,
            .bytes_on_disk = 0,
            .arch_hint = c.arch,
            .config = &cfg,
            .weights = null,
            .transformer = null,
            .tokenizer = null,
            .chat_config = &chat_cfg,
            .vision_encoder = null,
                .drafter_path = "",
            .drafter_block_size = 0,
            .prefix_cache = null,
            .refcount = std.atomic.Value(u32).init(0),
            .last_used_ns = 0,
            .bytes_resident = 0,
            .state = .ready,
            .error_name = null,
        };
        const loaded = try renderModelEntry(t.allocator, io, &entry);
        defer t.allocator.free(loaded);
        try t.expect(std.mem.indexOf(u8, loaded, c.want) != null);
        entry.state = .unloaded;
        entry.config = null;
        entry.chat_config = null;
        const stub = try renderModelEntry(t.allocator, io, &entry);
        defer t.allocator.free(stub);
        try t.expect(std.mem.indexOf(u8, stub, c.want) != null);
    }
}

fn testQuantizationModelRow(k: ?u8, expected: []const u8, loaded: bool) !void {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    const path = path_buf[0..path_len];
    const expert = if (k) |rate| try std.fmt.allocPrint(t.allocator, ",\"expert_quant\":{{\"format\":\"exl3\",\"codebook\":\"mcg\",\"k\":{d}}}", .{rate}) else try t.allocator.dupe(u8, "");
    defer t.allocator.free(expert);
    const raw = try std.fmt.allocPrint(t.allocator, "{{\"model_type\":\"qwen4_exp\",\"quantization\":{{\"bits\":8,\"group_size\":64}}{s}}}", .{expert});
    defer t.allocator.free(raw);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = raw });
    const cfg = model_mod.ModelConfig{ .model_type = "qwen4_exp", .quant_bits = 8, .expert_layout = if (k != null) .exl3_k4 else .quantized_split, .expert_quant_rate = .{ .n = if (k) |rate| rate * 16 else 64 } };
    if (loaded) {
        const label = try modelQuantizationLabel(t.allocator, &cfg);
        defer t.allocator.free(label);
        try t.expectEqualStrings(expected, label);
        return;
    }
    var entry = model_registry_mod.LoadedModel{
        .allocator = t.allocator,
        .id = "test-pack",
        .path = path,
        .bytes_on_disk = 0,
        .arch_hint = "qwen4_exp",
        .config = null,
        .weights = null,
        .transformer = null,
        .tokenizer = null,
        .chat_config = null,
        .vision_encoder = null,
        .drafter_path = "",
        .drafter_block_size = 0,
        .prefix_cache = null,
        .refcount = std.atomic.Value(u32).init(0),
        .last_used_ns = 0,
        .bytes_resident = 0,
        .state = .unloaded,
        .error_name = null,
    };
    const row = try renderModelEntry(t.allocator, io, &entry);
    defer t.allocator.free(row);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, row, .{});
    defer parsed.deinit();
    try t.expectEqualStrings(expected, parsed.value.object.get("meta").?.object.get("quantization").?.string);
}

test "model quantization label K3 loaded and unloaded" {
    try testQuantizationModelRow(3, "EXL3 3bpw experts, 8-bit dense", true);
    try testQuantizationModelRow(3, "EXL3 3bpw experts, 8-bit dense", false);
}

test "model quantization label K4 loaded and unloaded" {
    try testQuantizationModelRow(4, "EXL3 4bpw experts, 8-bit dense", true);
    try testQuantizationModelRow(4, "EXL3 4bpw experts, 8-bit dense", false);
}

test "model quantization label affine unchanged" {
    try testQuantizationModelRow(null, "8-bit", true);
    try testQuantizationModelRow(null, "8-bit", false);
}
