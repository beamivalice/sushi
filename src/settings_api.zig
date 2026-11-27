//! `GET/POST /v1/settings`: the settings file (`~/.sushi/model-settings.json`) as the web UI edits it. Server-wide keys
//! live under its `"server"` object, a model's under its absolute path. A launch flag outranks the file, so a key
//! a flag pins is reported locked; the file still takes the write, for the next start without that flag.
const std = @import("std");
const http = @import("http.zig");
const log = @import("log.zig");
const model_settings = @import("model_settings.zig");
const sched_policy = @import("sched_policy.zig");
const scheduler = @import("scheduler.zig");
const default_port = @import("default_port.zig");
const model_registry_mod = @import("model_registry.zig");
const disk_placement = @import("disk_placement.zig");
const http_conn = @import("http_conn.zig");
const server = @import("server.zig");
const server_state = @import("server_state.zig");
const arch_flags = @import("arch/flags.zig");
const model_discovery = @import("model_discovery.zig");
const expert_quant = @import("expert_quant.zig");
const model_mod = @import("model.zig");
const arch_config = @import("arch/config.zig");
const model_load = @import("model_load.zig");
const models_api = @import("models_api.zig");
const request_policy = @import("request_policy.zig");
const chat = @import("chat.zig");
const kv_quant_mod = @import("kv_quant.zig");
const update_mod = @import("update.zig");
const pack_update = @import("pack_update.zig");
const effort_mod = @import("effort.zig");
const bills = @import("bills.zig");
const settings_resolve = @import("settings_resolve.zig");
const status_mod = @import("status.zig");

const Edit = model_settings.Edit;

/// A running SSD tier as the inference thread last published it.
pub const TierView = struct { cap_bytes: u64, used_bytes: u64 };

pub const ModelRow = struct {
    id: []const u8,
    path: []const u8,
    state: []const u8,
    bytes: ?u64,
    /// What the key gates and value limits read (`model_settings.limit`).
    fit: model_settings.Fit = .{ .arch = "", .streams_experts = false, .has_vision = false },
    /// A stub's discovery found every shard its expert index names.
    index_complete: bool = false,
    /// The file values a ready model's load stamped; null for a model that is not loaded.
    loaded: ?model_settings.Override = null,
    /// What each key resolves to with no flag and no file value, for a ready model; a null field = not knowable.
    defaults: ?model_settings.Override = null,
    tier: ?TierView = null,
    /// The RoPE the pack's own config.json declares; `native` 0 = not read.
    rope: model_discovery.RopeFacts = .{},
};

pub const ServerView = struct {
    default_models_dir: []const u8,
    boot_roots: []const []const u8,
    /// The server-wide ask, which loads without a number of their own start from.
    ask_bytes: u64,
    /// `--think <word>`, which pins each family's default word it binds.
    think: ?effort_mod.Effort = null,
    /// This Mac's RAM and the `iogpu.wired_limit_mb` this process read at start, in bytes (0 = unknown / undeclared).
    total_ram: u64 = 0,
    wired_in_use: u64 = 0,
    wire_now: bills.WireLimits = .{},
};

/// The file value of one key as a JSON value, or null when the file had none.
fn overrideValue(o: model_settings.Override, name: []const u8) ?std.json.Value {
    const eql = std.mem.eql;
    if (eql(u8, name, "ctx_size")) return if (o.ctx_size) |v| .{ .integer = v } else null;
    if (eql(u8, name, "kv_quant")) return if (o.kv_quant) |v| .{ .string = v.wireName() } else null;
    if (eql(u8, name, "mtp")) return if (o.mtp) |v| .{ .bool = v } else null;
    if (eql(u8, name, "ssd_budget_gb")) return if (o.ssd_budget_gb) |v| .{ .integer = v } else null;
    if (eql(u8, name, "ssd_prefetch")) return if (o.ssd_prefetch) |v| .{ .integer = v } else null;
    if (eql(u8, name, "preserve_thinking")) return if (o.preserve_thinking) |v| .{ .bool = v } else null;
    if (eql(u8, name, "vision")) return if (o.vision) |v| .{ .bool = v } else null;
    if (eql(u8, name, "ds41_cooler")) return if (o.ds41_cooler) |v| .{ .bool = v } else null;
    if (eql(u8, name, "qwen_cooler")) return if (o.qwen_cooler) |v| .{ .bool = v } else null;
    if (eql(u8, name, "max_tokens")) return if (o.max_tokens) |v| .{ .integer = v } else null;
    if (eql(u8, name, "yarn_factor")) return if (o.yarn_factor) |v| .{ .string = if (v == 2) "2" else "4" } else null;
    if (eql(u8, name, "expert_cache_bias")) return if (o.expert_cache_bias) |v| .{ .float = v } else null;
    if (eql(u8, name, "prefill_chunk_max")) return if (o.prefill_chunk_max) |v| .{ .integer = v } else null;
    if (eql(u8, name, "prefill_chunk_min")) return if (o.prefill_chunk_min) |v| .{ .integer = v } else null;
    if (eql(u8, name, "max_concurrent")) return if (o.max_concurrent) |v| .{ .integer = v } else null;
    if (eql(u8, name, "max_kv_tokens")) return if (o.max_kv_tokens) |v| .{ .integer = @intCast(v) } else null;
    return null;
}

/// The settings a ready model's load stamped on its config.
fn loadedWith(config: *const model_mod.ModelConfig) model_settings.Override {
    var o: model_settings.Override = .{
        .ctx_size = if (config.ctx_override > 0) config.ctx_override else null,
        .kv_quant = config.kv_quant_override,
        .mtp = config.mtp_override,
        .ssd_budget_gb = if (config.ssd_budget_gb_override > 0) config.ssd_budget_gb_override else null,
        .ssd_prefetch = config.ssd_prefetch_override,
        .preserve_thinking = config.preserve_thinking_override,
        .vision = config.vision_override,
        .max_tokens = config.max_tokens_override,
        .yarn_factor = config.yarn_factor_override,
        .expert_cache_bias = config.expert_cache_bias_override,
        .prefill_chunk_max = config.prefill_chunk_max_override,
        .prefill_chunk_min = config.prefill_chunk_min_override,
        .max_concurrent = config.max_concurrent_override,
        .max_kv_tokens = config.max_kv_tokens_override,
    };
    arch_flags.putCoolerSetting(config, &o, config.cooler_override);
    return o;
}

/// What a loaded model's keys resolve to when the file and the launch flags say nothing, read from the loader's own
/// resolvers on a config with its overrides cleared. `ctx_size` and `ssd_budget_gb` stay null: they are measured at load.
pub fn resolvedDefaults(config: *const model_mod.ModelConfig) model_settings.Override {
    var bare = config.*;
    model_load.applyModelSettings(&bare, .{});
    bare.ssd_prefetch = null;
    bare.cooler_flag = null;
    var o: model_settings.Override = .{
        .kv_quant = server.kvCacheForLaunch(&bare, null).config,
        .mtp = model_load.mtpChoiceFor(null, &bare).on,
        .ssd_prefetch = bare.ssdPrefetchPct(),
        .preserve_thinking = model_settings.default_preserve_thinking,
        .vision = model_settings.VisionChoice.resolve(null, null, model_load.visionDefaultOn(&bare, bare.expert_streaming)).on,
        .max_tokens = request_policy.filledMaxTokensWith(0, bare.defaultMaxTokens()),
        .expert_cache_bias = 0,
    };
    const widest = settings_resolve.defaultChunkMax(&bare);
    o.prefill_chunk_max = widest;
    o.prefill_chunk_min = settings_resolve.defaultChunkMin(widest);
    o.max_concurrent = bare.max_concurrent;
    o.max_kv_tokens = bare.max_kv_tokens;
    arch_flags.putCoolerSetting(&bare, &o, bare.coolerChoice().on);
    return o;
}

fn writeJson(w: *std.Io.Writer, v: anytype) !void {
    try std.json.Stringify.value(v, .{}, w);
}

fn writeSpec(w: *std.Io.Writer, spec: model_settings.KeySpec) !void {
    try w.print("{{\"name\":\"{s}\",\"type\":\"{s}\"", .{ spec.name, @tagName(spec.kind) });
    switch (spec.kind) {
        .int, .number => try w.print(",\"min\":{d},\"max\":{d}", .{ spec.min, spec.max }),
        .choice => if (spec.choices.len > 0) {
            try w.writeAll(",\"choices\":");
            try writeJson(w, spec.choices);
        } else try w.writeAll(",\"path\":true"),
        .bool => {},
    }
    try w.writeAll("}");
}

/// One key's state: the file's value (if the loader would read it), the flag pinning it, and which wins.
fn writeKey(w: *std.Io.Writer, spec: model_settings.KeySpec, file_value: ?std.json.Value, lock: ?*const model_settings.Lock) !void {
    const valid = if (file_value) |v| (if (model_settings.validate(spec, v)) true else |_| false) else false;
    try w.writeAll("{\"file\":");
    if (valid) try writeJson(w, model_settings.canonicalValue(spec, file_value.?)) else try w.writeAll("null");
    const pinned = if (lock) |l| l.set else false;
    try w.writeAll(",\"flag\":");
    if (pinned) try writeJson(w, lock.?.text()) else try w.writeAll("null");
    try w.print(",\"source\":\"{s}\",\"locked\":{}", .{ if (pinned) "flag" else if (valid) "file" else "default", pinned });
    // A saved value the loader ignores is named where the person sets it: the key, the value and what the key takes.
    if (file_value) |v| if (!valid and v != .null) {
        var expected_buf: [160]u8 = undefined;
        try w.print(",\"invalid\":{{\"key\":\"{s}\",\"value\":", .{spec.name});
        try writeJson(w, v);
        try w.writeAll(",\"expected\":");
        try writeJson(w, model_settings.expectedText(&expected_buf, spec));
        try w.writeAll("}");
    };
    try w.writeAll("}");
}

/// What a model's keys may be set to, for every key with a constraint: `disabled` (the key does nothing), `choices` (the
/// subset it stores), `max` and `unset` (keys that must stay unset), each with a `why`.
fn writeLimits(w: *std.Io.Writer, fit: model_settings.Fit) !void {
    try w.writeAll("{");
    var first = true;
    for (model_settings.model_keys) |k| {
        const l = model_settings.limit(k.name, fit);
        var kept: [8][]const u8 = undefined;
        var n_kept: usize = 0;
        for (k.choices) |c| if (model_settings.choiceAllowed(k.name, c, fit) and n_kept < kept.len) {
            kept[n_kept] = c;
            n_kept += 1;
        };
        const narrowed = k.choices.len > 0 and n_kept < k.choices.len;
        if (l.isEmpty() and !narrowed) continue;
        if (!first) try w.writeAll(",");
        first = false;
        try w.print("\"{s}\":{{", .{k.name});
        if (l.disabled) try w.writeAll("\"disabled\":true,");
        if (narrowed) {
            try w.writeAll("\"choices\":");
            try writeJson(w, kept[0..n_kept]);
            try w.writeAll(",");
        }
        if (l.max > 0) try w.print("\"max\":{d},", .{l.max});
        if (l.unset.len > 0) {
            try w.writeAll("\"unset\":");
            try writeJson(w, l.unset);
            try w.writeAll(",");
        }
        var buf: [96]u8 = undefined;
        const why = if (l.why.len > 0) l.why else if (l.max > 0) model_settings.maxWhy(&buf, l.max) else "This model's cache stores only these widths.";
        try w.writeAll("\"why\":");
        try writeJson(w, why);
        try w.writeAll("}");
    }
    try w.writeAll("}");
}

const port_applies = "the next start; a running server keeps its port";

/// A server-wide on/off key: the file's value, the pinning flag, and the default it falls back to.
fn writeToggle(w: *std.Io.Writer, name: []const u8, file: ?bool, lock: *const model_settings.Lock, default: bool) !void {
    try w.print(",\"{s}\":{{\"file\":", .{name});
    try writeJson(w, file);
    try w.writeAll(",\"flag\":");
    if (lock.set) try writeJson(w, lock.text()) else try w.writeAll("null");
    try w.print(",\"default\":{},\"source\":\"{s}\",\"locked\":{},\"applies\":\"applies at once\"}}", .{
        default, if (lock.set) "flag" else if (file != null) "file" else "default", lock.set,
    });
}

fn writeSends(w: *std.Io.Writer, s: effort_mod.Sends) !void {
    switch (s) {
        .none => try w.writeAll("null"),
        .word => |x| try w.print("\"{s}\"", .{@tagName(x)}),
        .scale => |n| try w.print("{d}", .{n}),
    }
}

fn writeRow(w: *std.Io.Writer, r: effort_mod.Row) !void {
    try w.print("{{\"offered\":{},\"sends\":", .{r.offered});
    try writeSends(w, r.sends);
    try w.writeAll(",\"cap\":");
    try writeJson(w, r.cap);
    try w.writeAll(",\"fold_to\":");
    try writeJson(w, if (r.fold_to) |f| @tagName(f) else null);
    try w.writeAll("}");
}

/// The three wire limits now, the GPU limit this process uses, and the one command that sets the standard one
/// (all three limits, `bills.wireLimitsCommand`).
fn writeGpuLimit(w: *std.Io.Writer, view: ServerView) !void {
    const standard = bills.standardWiredLimitMb(view.total_ram);
    if (standard == 0) return w.writeAll(",\"gpu_limit\":null");
    var cmd: [192]u8 = undefined;
    const now = view.wire_now;
    try w.print(",\"gpu_limit\":{{\"total_ram_bytes\":{d},\"iogpu_mb\":{d},\"global_user_wire_bytes\":{d},\"user_wire_bytes\":{d},\"in_use_mb\":{d},\"standard_mb\":{d},\"command\":", .{
        view.total_ram, now.iogpu_mb, now.global_user_bytes, now.user_bytes, view.wired_in_use >> 20, standard,
    });
    try writeJson(w, bills.wireLimitsCommand(&cmd, standard));
    try w.writeAll(",\"applies\":\"run it in Terminal, then restart sushi: the limit is read at start, and macOS resets it at reboot\"}");
}

/// The effort tables as the page edits them: per family the built-in row, the stored fields and the served row, the
/// default word and the `--think` that pins it. Read from the served tables, so a save shows at once.
fn writeEffort(w: *std.Io.Writer, view: ServerView) !void {
    try w.writeAll(",\"effort\":{\"applies\":\"applies at once\",\"families\":[");
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    for (effort_mod.families, 0..) |f, fi| {
        if (fi > 0) try w.writeAll(",");
        const edits = effort_mod.publishedEdits(f);
        const b = effort_mod.builtin(f);
        const t = effort_mod.merge(f, edits);
        try w.print("{{\"family\":\"{s}\",\"sends\":", .{@tagName(f)});
        if (f.takesScale()) {
            try w.writeAll("{\"kind\":\"scale\",\"min\":1,\"max\":100,\"presets\":{");
            for (f.spec().presets, 0..) |p, i| try w.print("{s}\"{s}\":{d}", .{ if (i > 0) "," else "", @tagName(p.word), p.value });
            try w.writeAll("}}");
        } else {
            try w.writeAll("{\"kind\":\"word\",\"choices\":[");
            for (f.sendsWords(), 0..) |x, i| try w.print("{s}\"{s}\"", .{ if (i > 0) "," else "", @tagName(x) });
            try w.writeAll("]}");
        }
        const flag_arm: ?effort_mod.EffortArm = if (view.think) |e| blk: {
            const arms = t.offeredArms();
            break :blk effort_mod.armForWord(arms.slice(), @tagName(e));
        } else null;
        try w.print(",\"off_offerable\":{},\"default\":{{\"builtin\":\"{s}\",\"file\":", .{ f.offOfferable(), @tagName(b.default) });
        try writeJson(w, if (edits.default) |d| @tagName(d) else null);
        try w.print(",\"value\":\"{s}\",\"flag\":", .{@tagName(if (flag_arm) |a| a.effort else t.default)});
        if (flag_arm != null) {
            try w.print("\"--think {s}\"", .{@tagName(view.think.?)});
        } else try w.writeAll("null");
        try w.print(",\"locked\":{},\"source\":\"{s}\"}},\"rows\":[", .{ flag_arm != null, if (flag_arm != null) "flag" else if (edits.default != null) "file" else "default" });
        for (effort_mod.table_words, 0..) |word, i| {
            if (i > 0) try w.writeAll(",");
            try w.print("{{\"word\":\"{s}\",\"builtin\":", .{@tagName(word)});
            try writeRow(w, b.rows[i]);
            try w.writeAll(",\"file\":");
            if (edits.rows[i].isEmpty()) try w.writeAll("null") else try writeJson(w, try effort_mod.rowJson(arena_state.allocator(), f, edits.rows[i]));
            try w.writeAll(",\"value\":");
            try writeRow(w, t.rows[i]);
            try w.writeAll("}");
        }
        try w.writeAll("]}");
    }
    try w.writeAll("]}");
}

pub fn render(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    file: *const model_settings.Settings,
    rows: []const ModelRow,
    locks: *const model_settings.LaunchLocks,
    view: ServerView,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"file\":");
    try writeJson(w, file_path);
    try w.writeAll(",\"model_keys\":[");
    for (model_settings.model_keys, 0..) |k, i| {
        if (i > 0) try w.writeAll(",");
        try writeSpec(w, k);
    }
    try w.writeAll("],\"applies\":\"next load of the model\",\"server\":{");

    const srv = model_settings.server(file);
    try w.writeAll("\"models_dir\":{\"default\":");
    try writeJson(w, view.default_models_dir);
    try w.writeAll(",\"boot_roots\":");
    try writeJson(w, view.boot_roots);
    try w.writeAll(",");
    const dir_pinned = locks.models_dir.set;
    try w.writeAll("\"file\":");
    try writeJson(w, srv.models_dir);
    try w.writeAll(",\"flag\":");
    if (dir_pinned) try writeJson(w, locks.models_dir.text()) else try w.writeAll("null");
    try w.print(",\"source\":\"{s}\",\"locked\":{},\"applies\":\"models are rescanned at once; the next start discovers from it\"}}", .{
        if (dir_pinned) "flag" else if (srv.models_dir != null) "file" else "default", dir_pinned,
    });

    const cache_pinned = locks.prefix_cache_disk_gb.set;
    try w.writeAll(",\"prefix_cache_disk_gb\":{\"file\":");
    try writeJson(w, srv.prefix_cache_disk_gb);
    try w.writeAll(",\"flag\":");
    if (cache_pinned) try writeJson(w, locks.prefix_cache_disk_gb.text()) else try w.writeAll("null");
    var effective = view.ask_bytes;
    var running = false;
    for (rows) |row| if (row.tier) |t| {
        effective = if (running) @max(effective, t.cap_bytes) else t.cap_bytes;
        running = true;
    };
    try w.print(",\"source\":\"{s}\",\"locked\":{},\"effective_bytes\":{d},\"ask_bytes\":{d},\"default_bytes\":{d},\"running_tiers\":[", .{
        if (cache_pinned) "flag" else if (srv.prefix_cache_disk_gb != null) "file" else "default", cache_pinned, effective, view.ask_bytes, disk_placement.DEFAULT_DISK_CAP,
    });
    var first_tier = true;
    for (rows) |row| if (row.tier) |t| {
        if (!first_tier) try w.writeAll(",");
        first_tier = false;
        try w.writeAll("{\"model\":");
        try writeJson(w, row.id);
        try w.print(",\"cap_bytes\":{d},\"used_bytes\":{d}}}", .{ t.cap_bytes, t.used_bytes });
    };
    try w.writeAll("],\"applies\":\"a running tier takes a new size at once and evicts down to it; 0 (off) and removing the key apply at the next load\"}");
    try writeToggle(w, "update_check", srv.update_check, &locks.update_check, true);
    try writeToggle(w, "pack_update_check", srv.pack_update_check, &locks.pack_update_check, true);
    try writeToggle(w, "skip_mem_preflight", srv.skip_mem_preflight, &locks.skip_mem_preflight, false);
    try writeToggle(w, "continue_after_thought", srv.continue_after_thought, &locks.continue_after_thought, true);
    try w.writeAll(",\"max_request_mb\":{\"file\":");
    try writeJson(w, srv.max_request_mb);
    try w.print(",\"default\":{d},\"min\":1,\"max\":{d},\"source\":\"{s}\",\"locked\":false,\"applies\":\"applies at once to new requests; uploads already in progress keep their limit\"}}", .{
        model_settings.default_max_request_mb, model_settings.max_request_mb_limit, if (srv.max_request_mb != null) "file" else "default",
    });
    const margin_pinned = locks.wired_margin_gib.set;
    try w.writeAll(",\"wired_margin_gib\":{\"file\":");
    try writeJson(w, srv.wired_margin_gib);
    try w.writeAll(",\"flag\":");
    if (margin_pinned) try writeJson(w, locks.wired_margin_gib.text()) else try w.writeAll("null");
    try w.print(",\"default\":{d},\"min\":{d},\"max\":{d},\"source\":\"{s}\",\"locked\":{},\"applies\":\"{s}\"}}", .{
        model_settings.default_wired_margin_gib, model_settings.wired_margin_gib_min, model_settings.wired_margin_gib_max,
        if (margin_pinned) "flag" else if (srv.wired_margin_gib != null) "file" else "default",
        margin_pinned, if (margin_pinned) wired_margin_flag_applies else wired_margin_applies,
    });
    const concurrent_pinned = locks.max_concurrent.set;
    try w.writeAll(",\"max_concurrent\":{\"file\":");
    try writeJson(w, srv.max_concurrent);
    try w.writeAll(",\"flag\":");
    if (concurrent_pinned) try writeJson(w, locks.max_concurrent.text()) else try w.writeAll("null");
    try w.print(",\"default\":{d},\"min\":1,\"max\":{d},\"source\":\"{s}\",\"locked\":{},\"applies\":\"{s}\"}}", .{
        default_max_concurrent,                                                           sched_policy.MAX_BATCH_GROUP,
        if (concurrent_pinned) "flag" else if (srv.max_concurrent != null) "file" else "default", concurrent_pinned,
        if (concurrent_pinned) max_concurrent_flag_applies else max_concurrent_applies,
    });
    const port_pinned = locks.port.set;
    try w.writeAll(",\"port\":{\"file\":");
    try writeJson(w, srv.port);
    try w.writeAll(",\"flag\":");
    if (port_pinned) try writeJson(w, locks.port.text()) else try w.writeAll("null");
    try w.print(",\"default\":{d},\"source\":\"{s}\",\"locked\":{},\"applies\":\"{s}\"}}", .{
        default_port.port, if (port_pinned) "flag" else if (srv.port != null) "file" else "default", port_pinned, port_applies,
    });
    try w.writeAll("}");
    try writeEffort(w, view);
    try writeGpuLimit(w, view);

    try w.writeAll(",\"models\":[");
    for (rows, 0..) |row, i| {
        if (i > 0) try w.writeAll(",");
        try w.writeAll("{\"id\":");
        try writeJson(w, row.id);
        try w.writeAll(",\"path\":");
        try writeJson(w, row.path);
        try w.print(",\"state\":\"{s}\",\"bytes_on_disk\":", .{row.state});
        try writeJson(w, row.bytes);
        try w.writeAll(",\"keys\":[");
        var first_key = true;
        for (model_settings.model_keys) |k| if (model_settings.applies(k.name, row.fit)) {
            if (!first_key) try w.writeAll(",");
            first_key = false;
            try writeJson(w, k.name);
        };
        try w.writeAll("],\"limits\":");
        try writeLimits(w, row.fit);
        try w.writeAll(",\"rope\":");
        if (row.rope.native == 0) {
            try w.writeAll("null");
        } else try w.print("{{\"method\":\"{s}\",\"factor\":{d},\"native\":{d},\"window\":{d}}}", .{
            if (row.rope.yarn) "yarn" else "native", row.rope.factor, row.rope.native, row.rope.window,
        });
        try w.writeAll(",\"loaded_with\":");
        if (row.loaded) |o| {
            try w.writeAll("{");
            for (model_settings.model_keys, 0..) |k, ki| {
                if (ki > 0) try w.writeAll(",");
                try w.print("\"{s}\":", .{k.name});
                try writeJson(w, overrideValue(o, k.name));
            }
            try w.writeAll("}");
        } else try w.writeAll("null");
        try w.writeAll(",\"defaults\":");
        if (row.defaults) |o| {
            try w.writeAll("{");
            var first_default = true;
            for (model_settings.model_keys) |k| if (overrideValue(o, k.name)) |v| {
                if (!first_default) try w.writeAll(",");
                first_default = false;
                try w.print("\"{s}\":", .{k.name});
                try writeJson(w, v);
            };
            try w.writeAll("}");
        } else try w.writeAll("null");
        try w.writeAll(",\"settings\":{");
        const obj = model_settings.modelObject(file, row.path);
        for (model_settings.model_keys, 0..) |k, ki| {
            if (ki > 0) try w.writeAll(",");
            try w.print("\"{s}\":", .{k.name});
            try writeKey(w, k, if (obj) |o| o.get(k.name) else null, &locks.models[ki]);
        }
        try w.writeAll("}}");
    }
    try w.writeAll("]}");
    return out.toOwnedSlice();
}

pub const Rejection = struct { status: u16, err_type: []const u8, message: []const u8 };

/// One family's part of a save. A named row or default replaces the stored one; null takes the built-in back.
pub const EffortPlan = struct {
    named: bool = false,
    /// The whole table back to built-in.
    reset: bool = false,
    /// Outer null: not named.
    default: ??effort_mod.Effort = null,
    rows: [effort_mod.table_words.len]??effort_mod.RowEdit = @splat(null),
};

pub const Plan = struct {
    server_edits: []const Edit = &.{},
    model_id: ?[]const u8 = null,
    model_edits: []const Edit = &.{},
    effort: [effort_mod.family_count]EffortPlan = @splat(.{}),

    fn namesEffort(self: Plan) bool {
        for (self.effort) |e| if (e.named) return true;
        return false;
    }
};

pub const Parsed = union(enum) { ok: Plan, rejected: Rejection };

fn reject(arena: std.mem.Allocator, status: u16, err_type: []const u8, comptime fmt: []const u8, args: anytype) Parsed {
    const msg = std.fmt.allocPrint(arena, fmt, args) catch "invalid request";
    return .{ .rejected = .{ .status = status, .err_type = err_type, .message = msg } };
}

fn collect(arena: std.mem.Allocator, specs: []const model_settings.KeySpec, obj: std.json.ObjectMap, edits: *std.ArrayList(Edit)) ?Parsed {
    var it = obj.iterator();
    while (it.next()) |kv| {
        const spec = model_settings.findKey(specs, kv.key_ptr.*) orelse {
            if (model_settings.removedWhy(kv.key_ptr.*)) |why| return reject(arena, 400, "unknown_setting", "\"{s}\" is removed: {s}", .{ kv.key_ptr.*, why });
            return reject(arena, 400, "unknown_setting", "\"{s}\" is not a setting this server has", .{kv.key_ptr.*});
        };
        if (kv.value_ptr.* == .null) {
            edits.append(arena, .{ .key = spec.name, .value = null }) catch return reject(arena, 500, "internal_error", "out of memory", .{});
            continue;
        }
        model_settings.validate(spec, kv.value_ptr.*) catch
            return reject(arena, 400, "invalid_setting", "\"{s}\" does not take that value", .{spec.name});
        edits.append(arena, .{ .key = spec.name, .value = model_settings.canonicalValue(spec, kv.value_ptr.*) }) catch return reject(arena, 500, "internal_error", "out of memory", .{});
    }
    return null;
}

fn effortRejection(arena: std.mem.Allocator, f: effort_mod.Family, p: effort_mod.Problem) Parsed {
    return reject(arena, 400, "invalid_setting", "effort {s}.{s}{s}{s}: {s}", .{ @tagName(f), p.key, if (p.field.len > 0) "." else "", p.field, p.why });
}

/// `{"<family>": null | {"default": word|null, "<word>": row|null}}`, each value checked against the family's template.
fn collectEffort(arena: std.mem.Allocator, obj: std.json.ObjectMap, plan: *Plan) ?Parsed {
    var it = obj.iterator();
    while (it.next()) |kv| {
        const f = effort_mod.Family.of(kv.key_ptr.*) orelse
            return reject(arena, 400, "unknown_setting", "\"{s}\" has no effort table", .{kv.key_ptr.*});
        const fp = &plan.effort[@intFromEnum(f)];
        fp.named = true;
        const v = kv.value_ptr.*;
        if (v == .null) {
            fp.reset = true;
            continue;
        }
        if (v != .object) return reject(arena, 400, "invalid_request_error", "\"effort.{s}\" must be an object or null", .{@tagName(f)});
        var rows = v.object.iterator();
        while (rows.next()) |r| {
            const name = r.key_ptr.*;
            var problem: ?effort_mod.Problem = null;
            if (std.mem.eql(u8, name, "default")) {
                fp.default = if (r.value_ptr.* == .null) @as(?effort_mod.Effort, null) else effort_mod.parseDefault(r.value_ptr.*, &problem);
            } else if (std.meta.stringToEnum(effort_mod.Effort, name)) |word| {
                const i = effort_mod.rowIndex(word) orelse return effortRejection(arena, f, .{ .key = name, .why = "not an effort word" });
                fp.rows[i] = if (r.value_ptr.* == .null) @as(?effort_mod.RowEdit, null) else effort_mod.parseRow(f, word, r.value_ptr.*, &problem);
            } else problem = .{ .key = name, .why = "not an effort word" };
            if (problem) |p| return effortRejection(arena, f, p);
        }
    }
    return null;
}

/// `{"server": {key: value|null}, "model": "<id>", "set": {key: value|null}, "effort": {...}}`; every key is checked before anything is written.
pub fn parseRequest(arena: std.mem.Allocator, body: []const u8) Parsed {
    const tree = std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{}) catch
        return reject(arena, 400, "invalid_request_error", "Body must be JSON", .{});
    if (tree != .object) return reject(arena, 400, "invalid_request_error", "Body must be a JSON object", .{});
    var plan: Plan = .{};
    var it = tree.object.iterator();
    var set_obj: ?std.json.ObjectMap = null;
    while (it.next()) |kv| {
        const k = kv.key_ptr.*;
        const v = kv.value_ptr.*;
        if (std.mem.eql(u8, k, "server")) {
            if (v != .object) return reject(arena, 400, "invalid_request_error", "\"server\" must be an object", .{});
            var edits = std.ArrayList(Edit).empty;
            if (collect(arena, &model_settings.server_keys, v.object, &edits)) |r| return r;
            plan.server_edits = edits.items;
        } else if (std.mem.eql(u8, k, "model")) {
            if (v != .string or v.string.len == 0) return reject(arena, 400, "invalid_request_error", "\"model\" must be a model id", .{});
            plan.model_id = v.string;
        } else if (std.mem.eql(u8, k, "set")) {
            if (v != .object) return reject(arena, 400, "invalid_request_error", "\"set\" must be an object", .{});
            set_obj = v.object;
        } else if (std.mem.eql(u8, k, "effort")) {
            if (v != .object) return reject(arena, 400, "invalid_request_error", "\"effort\" must be an object", .{});
            if (collectEffort(arena, v.object, &plan)) |r| return r;
        } else {
            return reject(arena, 400, "unknown_setting", "\"{s}\" is not a field of this request", .{k});
        }
    }
    if (set_obj) |obj| {
        if (plan.model_id == null) return reject(arena, 400, "invalid_request_error", "\"set\" needs a \"model\"", .{});
        var edits = std.ArrayList(Edit).empty;
        if (collect(arena, &model_settings.model_keys, obj, &edits)) |r| return r;
        plan.model_edits = edits.items;
    }
    if (plan.server_edits.len == 0 and plan.model_edits.len == 0 and !plan.namesEffort()) return reject(arena, 400, "invalid_request_error", "Nothing to change", .{});
    return .{ .ok = plan };
}

/// Checks every named family's table against the file `file` holds and builds the sections that write it (only what
/// differs from built-in); returns the refusal text, or null with `next` and `sections` filled.
fn effortSections(
    arena: std.mem.Allocator,
    file: *const model_settings.Settings,
    plans: [effort_mod.family_count]EffortPlan,
    next: *[effort_mod.family_count]effort_mod.Edits,
    sections: *std.ArrayList(model_settings.SectionEdits),
) !?[]const u8 {
    for (effort_mod.families) |f| {
        const p = plans[@intFromEnum(f)];
        const n = &next[@intFromEnum(f)];
        n.* = model_settings.effortEdits(file, f, false);
        if (!p.named) continue;
        if (p.reset) n.* = .{};
        if (p.default) |d| n.default = d;
        for (p.rows, &n.rows) |r, *row| if (r) |given| {
            row.* = given orelse .{};
        };
        if (effort_mod.checkTable(effort_mod.merge(f, n.*))) |problem| return try std.fmt.allocPrint(arena, "effort {s}.{s}: {s}", .{ @tagName(f), problem.key, problem.why });
        // An unoffered --think word would stop binding now and refuse the next start with the same flag.
        if (effort_mod.think_effort_flag) |think| if (effort_mod.servedArms(@tagName(f))) |served| {
            const merged = effort_mod.merge(f, n.*).offeredArms();
            if (effort_mod.resolve(served.slice(), think) != null and effort_mod.resolve(merged.slice(), think) == null)
                return try std.fmt.allocPrint(arena, "effort {s}.{s}: --think serves this word; a start without --think may unoffer it", .{ @tagName(f), @tagName(think) });
        };
        n.* = effort_mod.normalize(f, n.*);
    }
    for (effort_mod.families) |f| {
        if (!plans[@intFromEnum(f)].named) continue;
        const n = next[@intFromEnum(f)];
        const edits = try arena.alloc(Edit, effort_mod.table_words.len + 1);
        edits[0] = .{ .key = "default", .value = if (n.default) |d| .{ .string = @tagName(d) } else null };
        for (effort_mod.table_words, n.rows, 1..) |word, row, i| edits[i] = .{ .key = @tagName(word), .value = if (row.isEmpty()) null else try effort_mod.rowJson(arena, f, row) };
        var buf: [64]u8 = undefined;
        const name = try arena.dupe(u8, effort_mod.sectionName(&buf, f));
        try sections.append(arena, .{ .section = name, .edits = edits });
    }
    return null;
}

fn effortMessage(think: ?effort_mod.Effort, plans: [effort_mod.family_count]EffortPlan) []const u8 {
    if (think != null) for (plans) |p| if (p.default != null) return "applies at once, from the next request; --think still picks the default word until a start without it";
    return "applies at once, from the next request";
}

// ── HTTP ──

fn stateName(s: model_registry_mod.LoadState) []const u8 {
    return @tagName(s);
}

/// What an unloaded model's config.json says, read without the registry lock.
fn stubFit(io: std.Io, allocator: std.mem.Allocator, row: *const ModelRow) model_settings.Fit {
    const arch = row.fit.arch;
    const sm = model_discovery.readStubMeta(io, allocator, row.path);
    const geometry = sm.found and sm.hidden_size > 0 and sm.num_hidden_layers > 0 and sm.num_experts > 0 and
        sm.num_experts_per_tok > 0 and sm.moe_intermediate_size > 0;
    const probe = model_mod.ModelConfig{ .model_type = arch, .expert_layout = if (sm.quantized_experts) .exl3_k4 else .mxfp4_w123 };
    var fit = arch_flags.settingsFit(&probe);
    fit.streams_experts = geometry and expert_quant.isExpertStreamingArch(arch);
    fit.has_vision = sm.has_vision;
    fit.streaming_required = arch_config.fact(&probe, "streams_experts") or models_api.streamingRequiredMarker(arch, sm.quant_bits, sm.quantized_experts, geometry, row.index_complete);
    fit.stq_experts = sm.quantized_experts;
    fit.context_cap = sm.context_cap;
    // An unloaded pack's score geometry is unread: its load refuses a min above its width, and retries the file.
    fit.prefill_chunk_ceiling = 0;
    return fit;
}

/// The facts only the pack's files hold, read for a loaded model as for a stub: its head, and for a stub its chat template.
fn readPackFacts(io: std.Io, allocator: std.mem.Allocator, row: *ModelRow) void {
    if (row.loaded == null) row.fit = stubFit(io, allocator, row);
    row.fit.mtp_head = model_load.packShipsMtpHead(io, allocator, row.fit.arch, row.path);
    row.rope = model_discovery.readRopeFacts(io, allocator, row.path);
    row.fit.window_native = row.rope.native;
    row.fit.window_default = row.rope.cap();
    if (row.loaded != null) return;
    var chat_config = chat.peekChatConfig(io, allocator, row.path, if (row.fit.arch.len > 0) row.fit.arch else null) catch return;
    defer chat_config.deinit();
    row.fit.template_preserves_thinking = chat.templateTakesPreserveThinking(&chat_config);
}

/// An entry's row from what the registry holds; the caller holds its lock.
fn heldRow(e: *model_registry_mod.LoadedModel) ModelRow {
    var row: ModelRow = .{ .id = e.id, .path = e.path, .state = stateName(e.state), .bytes = e.bytes_on_disk, .index_complete = e.streaming_index_complete };
    row.fit.arch = e.archHint();
    if (e.state == .ready and e.config != null) {
        const c = e.config.?;
        row.fit = arch_flags.settingsFit(c);
        row.loaded = loadedWith(c);
        row.defaults = resolvedDefaults(c);
        if (e.chat_config) |cc| {
            row.fit.template_preserves_thinking = chat.templateTakesPreserveThinking(cc);
        }
        const snap = e.disk_stats.snapshot();
        if (snap.namespace != 0) row.tier = .{ .cap_bytes = snap.budget, .used_bytes = snap.used };
    }
    return row;
}

fn collectRows(allocator: std.mem.Allocator, registry: *model_registry_mod.ModelRegistry) ![]ModelRow {
    var rows = std.ArrayList(ModelRow).empty;
    errdefer rows.deinit(allocator);
    {
        registry.mutex.lockUncancelable(registry.io);
        defer registry.mutex.unlock(registry.io);
        var it = registry.entries.valueIterator();
        while (it.next()) |p| try rows.append(allocator, heldRow(p.*));
    }
    for (rows.items) |*row| readPackFacts(registry.io, allocator, row);
    std.mem.sort(ModelRow, rows.items, {}, struct {
        fn lt(_: void, a: ModelRow, b: ModelRow) bool {
            return std.mem.lessThan(u8, a.id, b.id);
        }
    }.lt);
    return rows.toOwnedSlice(allocator);
}

pub fn handleGet(allocator: std.mem.Allocator, stream: *http_conn.Conn) !void {
    const registry = server_state.global_registry orelse {
        try http.sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", http.not_ready_message, 503);
        return;
    };
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = model_settings.defaultPath(&path_buf);
    var file = model_settings.load(allocator, stream.io, path);
    defer file.deinit();
    const rows = try collectRows(allocator, registry);
    defer allocator.free(rows);
    const home = std.mem.span(std.c.getenv("HOME") orelse "/tmp");
    const default_dir = try std.fmt.allocPrint(allocator, "{s}/.sushi/models", .{home});
    defer allocator.free(default_dir);
    const roots: []const []const u8 = if (registry.discovery) |d| d.roots else &.{};
    const body = try render(allocator, path, &file, rows, &model_settings.launch_locks, .{
        .default_models_dir = default_dir,
        .boot_roots = roots,
        .ask_bytes = server.diskAsk().bytes,
        .think = effort_mod.think_effort_flag,
        .total_ram = status_mod.getTotalMemBytes(),
        .wired_in_use = bills.ceilingNow().declared orelse 0,
        .wire_now = bills.wireLimitsNow(),
    });
    defer allocator.free(body);
    try http.sendResponse(stream, "200 OK", "application/json", body);
}

/// Why `path` cannot be the models folder, by name, or null when it can be listed.
fn modelsDirRefusal(io: std.Io, path: []const u8) error{Canceled}!?[]const u8 {
    var d = std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true }) catch |err| return try @import("repl_tools.zig").folderOpenRefusal(err);
    d.close(io);
    return null;
}

/// The 400 text for the first edit this model cannot honour, or null.
fn refusalMessage(arena: std.mem.Allocator, fit: model_settings.Fit, existing: ?std.json.ObjectMap, edits: []const Edit) ?[]const u8 {
    var why_buf: [192]u8 = undefined;
    const r = model_settings.refuseEdits(fit, existing, edits, &why_buf) orelse return null;
    return std.fmt.allocPrint(arena, "\"{s}\" cannot be set for this model: {s}", .{ r.key, r.why }) catch "a setting this model cannot honour";
}

pub fn handlePost(allocator: std.mem.Allocator, stream: *http_conn.Conn, request_body: []const u8) !void {
    const sch = server_state.global_scheduler orelse {
        try http.sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", http.not_ready_message, 503);
        return;
    };
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const plan = switch (parseRequest(arena, request_body)) {
        .ok => |p| p,
        .rejected => |r| {
            var status_buf: [32]u8 = undefined;
            const status = std.fmt.bufPrint(&status_buf, "{d} Bad Request", .{r.status}) catch "400 Bad Request";
            try http.sendErrorResponse(allocator, stream, status, r.err_type, r.message, r.status);
            return;
        },
    };

    for (plan.server_edits) |e| if (std.mem.eql(u8, e.key, "models_dir")) if (e.value) |v| {
        if (try modelsDirRefusal(stream.io, v.string)) |why| {
            try http.sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_setting", try std.fmt.allocPrint(arena, "models_dir: {s}", .{why}), 400);
            return;
        }
    };
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = model_settings.defaultPath(&path_buf);
    var model_path: []const u8 = "";
    var model_entry: ?*model_registry_mod.LoadedModel = null;
    var model_loaded = false;
    var model_fit: ?model_settings.Fit = null;
    if (plan.model_id) |id| {
        const entry = sch.registry.resolveEntry(id) catch {
            try http.sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "Unknown model id", 404);
            return;
        };
        model_path = entry.path;
        model_entry = entry;
        sch.registry.mutex.lockUncancelable(sch.io);
        model_loaded = entry.state == .ready;
        var row = heldRow(entry);
        sch.registry.mutex.unlock(sch.io);
        if (plan.model_edits.len > 0) {
            readPackFacts(stream.io, allocator, &row);
            model_fit = row.fit;
        }
    }

    const live: Live = .{ .sch = sch, .io = stream.io, .path = path, .model = model_entry };
    const saved = save(live, arena, stream.io, path, plan, model_path, model_fit) catch |err| return writeFailed(allocator, stream, err);
    const messages = switch (saved) {
        .refused => |message| {
            try http.sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_setting", message, 400);
            return;
        },
        .saved => |m| m,
    };

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"ok\":true,\"applied\":{");
    var first = true;
    for (plan.server_edits, messages) |e, message| {
        if (!first) try w.writeAll(",");
        first = false;
        try w.print("\"{s}\":", .{e.key});
        try writeJson(w, message);
    }
    if (plan.namesEffort()) {
        if (!first) try w.writeAll(",");
        first = false;
        try w.writeAll("\"effort\":");
        try writeJson(w, effortMessage(effort_mod.think_effort_flag, plan.effort));
    }
    if (plan.model_edits.len > 0) {
        if (!first) try w.writeAll(",");
        try w.writeAll("\"model\":");
        try writeJson(w, if (model_loaded) "saved; the loaded model uses it after a reload" else "saved; applies at the model's next load");
    }
    try w.print("}},\"model_loaded\":{}}}", .{model_loaded});
    try http.sendResponse(stream, "200 OK", "application/json", out.written());
}

/// What a save reaches beyond its own file: the scheduler's SSD tiers and model rescan, and the registry's stored load
/// failures. `Live` is the running server's; a test stands in its own.
const Live = struct {
    sch: *scheduler.Scheduler,
    io: std.Io,
    path: []const u8,
    /// The entry the request resolved: a save reaches it with no second lookup by path.
    model: ?*model_registry_mod.LoadedModel,

    fn write(self: Live, arena: std.mem.Allocator, sections: []const model_settings.SectionEdits) !void {
        return model_settings.writeSectionsLocked(arena, self.io, self.path, sections);
    }

    fn forgetAll(self: Live) void {
        self.sch.registry.forgetLoadFailures();
    }

    fn forgetModel(self: Live) void {
        if (self.model) |entry| self.sch.registry.forgetLoadFailure(entry);
    }

    fn cacheBudget(self: Live, value: ?std.json.Value) []const u8 {
        return applyCacheBudget(self.sch, value);
    }

    fn modelsDir(self: Live, arena: std.mem.Allocator) []const u8 {
        return applyModelsDir(self.sch, arena.alloc(u8, 160) catch return "saved; models rescanned");
    }

    fn rearmDisk(self: Live) void {
        self.sch.requestDiskRearm();
    }
};

pub const Outcome = union(enum) {
    /// The save broke a rule of the effort tables or of the model's settings; nothing was written.
    refused: []const u8,
    /// One answer per server edit, in the plan's order.
    saved: []const []const u8,
};

fn differs(comptime T: type, was: ?T, now: ?T, default: T) bool {
    return (was orelse default) != (now orelse default);
}

fn sameText(a: ?[]const u8, b: ?[]const u8) bool {
    if (a == null or b == null) return a == null and b == null;
    return std.mem.eql(u8, a.?, b.?);
}

/// Whether saving `e` moves what the server runs on: a value other than the one the file held (an absent key is its
/// default), and never under a flag that pins the key. Keys no load reads never move one.
fn serverKeyMoves(locks: *const model_settings.LaunchLocks, was: model_settings.Server, e: Edit) bool {
    const eql = std.mem.eql;
    const now_int: ?u32 = if (e.value) |v| if (v == .integer) std.math.cast(u32, v.integer) else null else null;
    if (eql(u8, e.key, "models_dir")) return !locks.models_dir.set and !sameText(was.models_dir, if (e.value) |v| v.string else null);
    if (eql(u8, e.key, "prefix_cache_disk_gb")) return !locks.prefix_cache_disk_gb.set and differs(u32, was.prefix_cache_disk_gb, now_int, @intCast(disk_placement.DEFAULT_DISK_CAP >> 30));
    if (eql(u8, e.key, "skip_mem_preflight")) return !locks.skip_mem_preflight.set and differs(bool, was.skip_mem_preflight, if (e.value) |v| v.bool else null, false);
    if (eql(u8, e.key, "max_request_mb")) return differs(u32, was.max_request_mb, now_int, model_settings.default_max_request_mb);
    if (eql(u8, e.key, "wired_margin_gib")) return !locks.wired_margin_gib.set and differs(u32, was.wired_margin_gib, now_int, model_settings.default_wired_margin_gib);
    if (eql(u8, e.key, "max_concurrent")) return !locks.max_concurrent.set and differs(u32, was.max_concurrent, now_int, default_max_concurrent);
    return false;
}

/// A model's key moves what its next load reads when the file's value changes and no flag pins the key; a key left
/// out of the file reads as the model's own default, so only the file's text is compared.
fn modelKeyMoves(arena: std.mem.Allocator, locks: *const model_settings.LaunchLocks, was: ?std.json.ObjectMap, e: Edit) bool {
    if (locks.model(e.key)) |lock| if (lock.set) return false;
    const old: ?std.json.Value = if (was) |o| o.get(e.key) else null;
    if (old == null or e.value == null) return old != null or e.value != null;
    const a = std.json.Stringify.valueAlloc(arena, old.?, .{}) catch return true;
    const b = std.json.Stringify.valueAlloc(arena, e.value.?, .{}) catch return true;
    return !std.mem.eql(u8, a, b);
}

/// Applies one saved server key; `parseRequest` has checked its value, so nothing here fails after the write.
fn applyServerEdit(host: anytype, arena: std.mem.Allocator, locks: *const model_settings.LaunchLocks, e: Edit, moved: bool) []const u8 {
    const eql = std.mem.eql;
    if (eql(u8, e.key, "prefix_cache_disk_gb")) return host.cacheBudget(e.value);
    if (eql(u8, e.key, "update_check") or eql(u8, e.key, "pack_update_check")) return applyUpdateCheck(locks, e.key, e.value);
    if (eql(u8, e.key, "skip_mem_preflight")) return applySkipMemPreflight(locks, e.value);
    if (eql(u8, e.key, "continue_after_thought")) return applyContinueAfterThought(locks, e.value);
    if (eql(u8, e.key, "max_request_mb")) {
        server_state.setMaxRequestMb(if (e.value) |v| std.math.cast(u32, v.integer) orelse unreachable else null);
        return "applies to the next request";
    }
    if (eql(u8, e.key, "wired_margin_gib")) return applyWiredMargin(locks, e.value, moved);
    if (eql(u8, e.key, "port")) return portMessage(locks);
    if (eql(u8, e.key, "max_concurrent")) return applyMaxConcurrent(locks, e.value);
    return host.modelsDir(arena);
}

/// The one save path: every section in one rewrite of the file, then each server key applied to the running server,
/// then the stored load failures its moved keys end. A write that fails returns before anything is applied.
/// `fit`: the model's, when the plan edits a model: its edits are judged against the file this save rewrites.
fn save(host: anytype, arena: std.mem.Allocator, io: std.Io, path: []const u8, plan: Plan, model_path: []const u8, fit: ?model_settings.Fit) !Outcome {
    const locks = &model_settings.launch_locks;
    // Read, check, write and apply under the write lock: a save never merges into a file another one is rewriting,
    // and the file's last save is the server's last apply.
    model_settings.lockWrites(io);
    defer model_settings.unlockWrites(io);
    var file = model_settings.load(arena, io, path);
    defer file.deinit();
    // Under the lock: a save checked against a file another save then rewrote could land a pair either alone is refused.
    if (fit) |f| if (plan.model_edits.len > 0) {
        var judged = f;
        if (locks.model("ctx_size")) |lock| if (lock.set) {
            judged.ctx_flag = std.fmt.parseInt(u32, lock.text(), 10) catch 0;
        };
        if (refusalMessage(arena, judged, model_settings.modelObject(&file, model_path), plan.model_edits)) |message| return .{ .refused = message };
    };
    var next: [effort_mod.family_count]effort_mod.Edits = undefined;
    var sections = std.ArrayList(model_settings.SectionEdits).empty;
    if (plan.namesEffort()) {
        if (try effortSections(arena, &file, plan.effort, &next, &sections)) |refused| return .{ .refused = refused };
    }
    if (plan.server_edits.len > 0) try sections.append(arena, .{ .section = model_settings.server_section, .edits = plan.server_edits });
    if (plan.model_edits.len > 0) try sections.append(arena, .{ .section = model_path, .edits = plan.model_edits });

    const messages = try arena.alloc([]const u8, plan.server_edits.len);
    try host.write(arena, sections.items);

    const was = model_settings.server(&file);
    var forget_all = false;
    for (plan.server_edits, messages) |e, *message| {
        const moved = serverKeyMoves(locks, was, e);
        message.* = applyServerEdit(host, arena, locks, e, moved);
        if (moved and reachesLoad(e.key)) forget_all = true;
    }
    for (effort_mod.families) |f| {
        if (!plan.effort[@intFromEnum(f)].named) continue;
        effort_mod.publish(f, next[@intFromEnum(f)]);
        log.info("[effort] {s}: table saved, serving from the next request\n", .{@tagName(f)});
    }
    // Last, so a load that starts after it reads the new values, and one already running is marked to forget its failure.
    if (forget_all) {
        host.forgetAll();
    } else for (plan.model_edits) |e| {
        if (reachesLoad(e.key) and modelKeyMoves(arena, locks, model_settings.modelObject(&file, model_path), e)) {
            host.forgetModel();
            break;
        }
    }
    if (savesSsdTier(plan)) host.rearmDisk();
    return .{ .saved = messages };
}

/// A saved key that sizes an SSD tier: the SSD prefix cache's size, or the expert-streaming SSD budget.
pub fn reachesSsdTier(key: []const u8) bool {
    return std.mem.eql(u8, key, "prefix_cache_disk_gb") or std.mem.eql(u8, key, "ssd_budget_gb");
}

/// Runs after the save is applied, so the re-arm reads the new ask.
fn savesSsdTier(plan: Plan) bool {
    for (plan.server_edits) |e| if (reachesSsdTier(e.key)) return true;
    for (plan.model_edits) |e| if (reachesSsdTier(e.key)) return true;
    return false;
}

/// Whether saving `key` changes what a load or its preflight reads, so a model that failed to load deserves another try.
pub fn reachesLoad(key: []const u8) bool {
    const reads = [_][]const u8{ "ctx_size", "kv_quant", "mtp", "vision", "ds41_cooler", "qwen_cooler", "ssd_budget_gb", "ssd_prefetch", "yarn_factor", "expert_cache_bias", "prefill_chunk_max", "prefill_chunk_min", "max_concurrent", "max_kv_tokens", "models_dir", "prefix_cache_disk_gb", "skip_mem_preflight", "wired_margin_gib" };
    for (reads) |n| if (std.mem.eql(u8, n, key)) return true;
    return false;
}

test "settings api: every saved key is classified, and the ones a load reads clear a stored load failure" {
    for ([_][]const u8{ "ctx_size", "kv_quant", "mtp", "vision", "ds41_cooler", "qwen_cooler", "ssd_budget_gb", "ssd_prefetch", "yarn_factor", "expert_cache_bias", "prefill_chunk_max", "prefill_chunk_min", "max_concurrent", "max_kv_tokens", "models_dir", "prefix_cache_disk_gb", "skip_mem_preflight", "wired_margin_gib" }) |key| {
        try std.testing.expect(reachesLoad(key));
    }
    for ([_][]const u8{ "preserve_thinking", "max_tokens", "update_check", "pack_update_check", "port", "max_request_mb", "continue_after_thought" }) |key| {
        try std.testing.expect(!reachesLoad(key));
    }
    // A new key must be classified above before it ships.
    for (model_settings.model_keys) |k| try std.testing.expect(classified(k.name));
    for (model_settings.server_keys) |k| try std.testing.expect(classified(k.name));
}

test "settings api: only the two SSD size keys re-arm a declined SSD tier, and both reach a load" {
    for ([_][]const u8{ "prefix_cache_disk_gb", "ssd_budget_gb" }) |key| {
        try std.testing.expect(reachesSsdTier(key));
        try std.testing.expect(reachesLoad(key));
    }
    for ([_][]const u8{ "ctx_size", "kv_quant", "mtp", "models_dir", "skip_mem_preflight", "port" }) |key| try std.testing.expect(!reachesSsdTier(key));
}

fn classified(key: []const u8) bool {
    const known = [_][]const u8{ "ctx_size", "kv_quant", "mtp", "vision", "ds41_cooler", "qwen_cooler", "ssd_budget_gb", "ssd_prefetch", "yarn_factor", "expert_cache_bias", "prefill_chunk_max", "prefill_chunk_min", "max_concurrent", "max_kv_tokens", "models_dir", "prefix_cache_disk_gb", "skip_mem_preflight", "preserve_thinking", "max_tokens", "update_check", "pack_update_check", "port", "max_request_mb", "wired_margin_gib", "continue_after_thought" };
    for (known) |n| if (std.mem.eql(u8, n, key)) return true;
    return false;
}

fn writeFailed(allocator: std.mem.Allocator, stream: *http_conn.Conn, err: anyerror) !void {
    if (err == error.MalformedSettingsFile) {
        try http.sendErrorResponse(allocator, stream, "409 Conflict", "settings_file_malformed", "The settings file is not a JSON object; fix or remove it before saving", 409);
        return;
    }
    log.warn("[settings] write failed: {s}\n", .{@errorName(err)});
    try http.sendErrorResponse(allocator, stream, "500 Internal Server Error", "settings_write_failed", "Could not write the settings file; nothing was saved or applied", 500);
}

fn cacheBudgetMessage(running_tiers_resized: usize) []const u8 {
    return if (running_tiers_resized > 0) "applied to the running tiers" else "saved; applies at the next model load";
}

/// The new SSD tier size reaches the running tiers at once and every later load; the launch flag, when given, keeps the number it set.
fn applyCacheBudget(sch: *@import("scheduler.zig").Scheduler, value: ?std.json.Value) []const u8 {
    if (model_settings.launch_locks.prefix_cache_disk_gb.set) return "saved; the --prefix-cache-disk flag still sets the size until a start without it";
    const gb: ?u64 = if (value) |v| @intCast(v.integer) else null;
    server_state.setDiskAsk(if (gb) |g| g << 30 else disk_placement.DEFAULT_DISK_CAP, if (gb != null) .model_settings else .default);
    const bytes = (gb orelse 0) << 30;
    if (gb == null or bytes == 0) return cacheBudgetMessage(0);
    var tiers = std.ArrayList(*model_registry_mod.LoadedModel).empty;
    defer tiers.deinit(sch.allocator);
    {
        sch.registry.mutex.lockUncancelable(sch.io);
        defer sch.registry.mutex.unlock(sch.io);
        var it = sch.registry.entries.valueIterator();
        while (it.next()) |p| {
            if (p.*.state == .ready and p.*.disk_stats.snapshot().namespace != 0) tiers.append(sch.allocator, p.*) catch {};
        }
    }
    var resized: usize = 0;
    for (tiers.items) |e| {
        const r = sch.prefixAdmin(.{ .entry = e, .action = .set_cap, .cap_bytes = bytes }) catch break;
        if (r.outcome == .cleared) resized += 1;
    }
    return cacheBudgetMessage(resized);
}

/// The loops read the published value before each look, so a toggle needs no restart; null (key removed) is the default, on.
fn applyUpdateCheck(locks: *const model_settings.LaunchLocks, key: []const u8, value: ?std.json.Value) []const u8 {
    const is_sushi = std.mem.eql(u8, key, "update_check");
    if ((if (is_sushi) locks.update_check else locks.pack_update_check).set) return "saved; --no-update-check still keeps it off until a start without it";
    const on = if (value) |v| v.bool else true;
    if (is_sushi) update_mod.setAutoCheck(on) else pack_update.setAuto(on);
    return "applies at once";
}

/// The next load reads the published value; null (key removed) is the default, the check on.
fn applySkipMemPreflight(locks: *const model_settings.LaunchLocks, value: ?std.json.Value) []const u8 {
    if (locks.skip_mem_preflight.set) return "saved; --skip-mem-preflight still skips the check until a start without it";
    scheduler.setSkipMemPreflight(if (value) |v| v.bool else false);
    return "applies at once";
}

/// The next reply to end inside its thought reads the published value; null (key removed) is the default, on.
fn applyContinueAfterThought(locks: *const model_settings.LaunchLocks, value: ?std.json.Value) []const u8 {
    if (locks.continue_after_thought.set) return "saved; --no-continue-after-thought still keeps it off until a start without it";
    server_state.setContinueAfterThought(if (value) |v| v.bool else true);
    return "applies at once";
}

const wired_margin_applies = "applies at once: the next request and the next load use it";
const wired_margin_flag_applies = "set by --wired-margin-gib until a start without it";

/// The launch flag outranks the saved value inside `wiredMarginBytes`, so a save under it is kept and not applied.
fn applyWiredMargin(locks: *const model_settings.LaunchLocks, value: ?std.json.Value, moved: bool) []const u8 {
    server_state.setWiredMarginFile(if (value) |v| std.math.cast(u32, v.integer) else null);
    if (moved) bills.relogCeiling();
    return if (locks.wired_margin_gib.set) "saved; --wired-margin-gib still sets the margin until a start without it" else wired_margin_applies;
}

const default_max_concurrent: u32 = 1;
const max_concurrent_applies = "applies to each model's next load; a model's own value outranks it";
const max_concurrent_flag_applies = "set by --max-concurrent until a start without it";

/// The launch flag outranks the saved value inside `resolveConcurrency`, so a save under it is kept and not applied.
fn applyMaxConcurrent(locks: *const model_settings.LaunchLocks, value: ?std.json.Value) []const u8 {
    server_state.setMaxConcurrentFile(if (value) |v| std.math.cast(u32, v.integer) else null);
    return if (locks.max_concurrent.set) "saved; --max-concurrent still sets it until a start without it" else max_concurrent_applies;
}

fn portMessage(locks: *const model_settings.LaunchLocks) []const u8 {
    return if (locks.port.set) "saved; the --port flag still sets the port until a start without it" else "saved; applies at the next start";
}

fn modelsDirMessage(buf: []u8, rescan: anyerror!u32) []const u8 {
    const added = rescan catch return "saved; the rescan failed, use /v1/models/rescan";
    return std.fmt.bufPrint(buf, "saved; models rescanned, {d} added", .{added}) catch "saved; models rescanned";
}

fn applyModelsDir(sch: *@import("scheduler.zig").Scheduler, buf: []u8) []const u8 {
    if (model_settings.launch_locks.models_dir.set) return "saved; the --model-dir flag still sets the directories until a start without it";
    return modelsDirMessage(buf, if (sch.registry.rescan()) |r| r.added else |e| e);
}

// ── Tests ──

const testing = std.testing;

fn expectRejected(body: []const u8, status: u16, err_type: []const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    switch (parseRequest(arena_state.allocator(), body)) {
        .ok => return error.TestExpectedRejection,
        .rejected => |r| {
            try testing.expectEqual(status, r.status);
            try testing.expectEqualStrings(err_type, r.err_type);
        },
    }
}

test "settings api: a removed model key is refused by name as removed" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    switch (parseRequest(arena_state.allocator(), "{\"model\":\"m\",\"set\":{\"mtp_greedy_tail\":true}}")) {
        .ok => return error.TestExpectedRejection,
        .rejected => |r| {
            try testing.expectEqualStrings("unknown_setting", r.err_type);
            try testing.expectEqualStrings("\"mtp_greedy_tail\" is removed: MTP accepts drafts by the exact rule only", r.message);
        },
    }
}

test "settings api: a request names real keys with valid values, or is refused with a named 400" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const ok = parseRequest(arena_state.allocator(),
        \\{"server":{"prefix_cache_disk_gb":40,"models_dir":null},"model":"m","set":{"kv_quant":"8","mtp":null}}
    );
    try testing.expectEqual(@as(usize, 2), ok.ok.server_edits.len);
    try testing.expectEqual(@as(usize, 2), ok.ok.model_edits.len);
    try testing.expect(ok.ok.model_edits[1].value == null);
    try testing.expectEqualStrings("m", ok.ok.model_id.?);

    try expectRejected("{\"model\":\"m\",\"set\":{\"temperature\":1}}", 400, "unknown_setting");
    try expectRejected("{\"server\":{\"colour\":1}}", 400, "unknown_setting");
    try expectRejected("{\"model\":\"m\",\"set\":{\"ssd_prefetch\":101}}", 400, "invalid_setting");
    try expectRejected("{\"server\":{\"models_dir\":\"relative\"}}", 400, "invalid_setting");
    try expectRejected("{\"set\":{\"mtp\":true}}", 400, "invalid_request_error");
    try expectRejected("{\"colour\":1}", 400, "unknown_setting");
    try expectRejected("{}", 400, "invalid_request_error");
    try expectRejected("[1]", 400, "invalid_request_error");
    try expectRejected("nope", 400, "invalid_request_error");
}

test "settings api: a models_dir that cannot be listed is refused with its own reason, a folder it may not read by permission" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "Models");
    try tmp.dir.createDirPath(io, "locked");
    try tmp.dir.writeFile(io, .{ .sub_path = "notes.txt", .data = "hi\n" });
    try tmp.dir.setFilePermissions(io, "locked", .fromMode(0o000), .{});
    defer tmp.dir.setFilePermissions(io, "locked", .fromMode(0o755), .{}) catch {};
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    try testing.expectEqual(@as(?[]const u8, null), try modelsDirRefusal(io, try std.fmt.bufPrint(&pb, "{s}/Models/", .{base})));
    try testing.expectEqualStrings("no such folder", (try modelsDirRefusal(io, try std.fmt.bufPrint(&pb, "{s}/missing", .{base}))).?);
    try testing.expectEqualStrings("not a folder", (try modelsDirRefusal(io, try std.fmt.bufPrint(&pb, "{s}/notes.txt", .{base}))).?);
    try testing.expect(std.mem.startsWith(u8, (try modelsDirRefusal(io, try std.fmt.bufPrint(&pb, "{s}/locked", .{base}))).?, "no permission to read this folder"));
}

test "settings api: the update-check keys take a boolean and nothing else" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const ok = parseRequest(arena_state.allocator(), "{\"server\":{\"update_check\":false,\"pack_update_check\":null}}");
    try testing.expectEqual(@as(usize, 2), ok.ok.server_edits.len);
    try testing.expect(!ok.ok.server_edits[0].value.?.bool);
    try testing.expect(ok.ok.server_edits[1].value == null);
    try expectRejected("{\"server\":{\"update_check\":\"off\"}}", 400, "invalid_setting");
    try expectRejected("{\"server\":{\"pack_update_check\":0}}", 400, "invalid_setting");
}

test "settings api: render reports both update-check keys with their default, file value, and the flag that pins them" {
    var file = try model_settings.parse(testing.allocator, "{\"server\": {\"update_check\": false}}");
    defer file.deinit();
    var locks: model_settings.LaunchLocks = .{};
    const view: ServerView = .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30 };
    {
        const body = try render(testing.allocator, "/h/s.json", &file, &.{}, &locks, view);
        defer testing.allocator.free(body);
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
        defer parsed.deinit();
        const srv = parsed.value.object.get("server").?.object;
        const u = srv.get("update_check").?.object;
        try testing.expect(!u.get("file").?.bool);
        try testing.expect(u.get("default").?.bool);
        try testing.expectEqualStrings("file", u.get("source").?.string);
        try testing.expect(!u.get("locked").?.bool);
        try testing.expect(u.get("flag").? == .null);
        const p = srv.get("pack_update_check").?.object;
        try testing.expect(p.get("file").? == .null);
        try testing.expectEqualStrings("default", p.get("source").?.string);
    }
    locks.update_check.pin("--no-update-check", .{});
    locks.pack_update_check.pin("--no-update-check", .{});
    const body = try render(testing.allocator, "/h/s.json", &file, &.{}, &locks, view);
    defer testing.allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    for ([_][]const u8{ "update_check", "pack_update_check" }) |k| {
        const o = parsed.value.object.get("server").?.object.get(k).?.object;
        try testing.expectEqualStrings("flag", o.get("source").?.string);
        try testing.expect(o.get("locked").?.bool);
        try testing.expectEqualStrings("--no-update-check", o.get("flag").?.string);
    }
}

test "settings api: render reports the default port with the file's value, the pinning flag and when it applies" {
    const view: ServerView = .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30 };
    var locks: model_settings.LaunchLocks = .{};
    {
        var file = try model_settings.parse(testing.allocator, "{\"server\": {\"port\": 8080}}");
        defer file.deinit();
        const body = try render(testing.allocator, "/h/s.json", &file, &.{}, &locks, view);
        defer testing.allocator.free(body);
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
        defer parsed.deinit();
        const p = parsed.value.object.get("server").?.object.get("port").?.object;
        try testing.expectEqual(@as(i64, 8080), p.get("file").?.integer);
        try testing.expectEqual(@as(i64, 12345), p.get("default").?.integer);
        try testing.expectEqualStrings("file", p.get("source").?.string);
        try testing.expect(!p.get("locked").?.bool);
        try testing.expect(std.mem.indexOf(u8, p.get("applies").?.string, "next start") != null);
    }
    var empty = try model_settings.parse(testing.allocator, "{}");
    defer empty.deinit();
    locks.port.pin("9000", .{});
    const body = try render(testing.allocator, "/h/s.json", &empty, &.{}, &locks, view);
    defer testing.allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const p = parsed.value.object.get("server").?.object.get("port").?.object;
    try testing.expect(p.get("file").? == .null);
    try testing.expectEqualStrings("flag", p.get("source").?.string);
    try testing.expect(p.get("locked").?.bool);
    try testing.expectEqualStrings("9000", p.get("flag").?.string);
}

test "settings api: server.port takes a whole port from 1 to 65535 or null, and says a flag still wins" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const ok = parseRequest(arena_state.allocator(), "{\"server\":{\"port\":8080}}");
    try testing.expectEqual(@as(i64, 8080), ok.ok.server_edits[0].value.?.integer);
    try testing.expect(parseRequest(arena_state.allocator(), "{\"server\":{\"port\":null}}").ok.server_edits[0].value == null);
    for ([_][]const u8{ "0", "65536", "-1", "\"80\"", "80.5", "true" }) |bad| {
        const body = try std.fmt.allocPrint(arena_state.allocator(), "{{\"server\":{{\"port\":{s}}}}}", .{bad});
        try expectRejected(body, 400, "invalid_setting");
    }
    var locks: model_settings.LaunchLocks = .{};
    try testing.expect(std.mem.indexOf(u8, portMessage(&locks), "next start") != null);
    locks.port.pin("9000", .{});
    try testing.expect(std.mem.indexOf(u8, portMessage(&locks), "--port flag still sets") != null);
}

test "settings api: a toggle reaches the loops at once, and a pinning flag keeps the key off" {
    defer update_mod.setAutoCheck(true);
    defer pack_update.setAuto(true);
    var locks: model_settings.LaunchLocks = .{};
    try testing.expectEqualStrings("applies at once", applyUpdateCheck(&locks, "update_check", .{ .bool = false }));
    try testing.expect(!update_mod.autoCheckOn());
    try testing.expect(pack_update.autoOn());
    _ = applyUpdateCheck(&locks, "update_check", null);
    try testing.expect(update_mod.autoCheckOn());
    _ = applyUpdateCheck(&locks, "pack_update_check", .{ .bool = false });
    try testing.expect(!pack_update.autoOn());
    locks.pack_update_check.pin("--no-update-check", .{});
    _ = applyUpdateCheck(&locks, "pack_update_check", .{ .bool = true });
    try testing.expect(!pack_update.autoOn());
}

test "settings api: render reports the three wire limits now, the GPU limit in use and the standard command" {
    var locks: model_settings.LaunchLocks = .{};
    var file = try model_settings.parse(testing.allocator, "{}");
    defer file.deinit();
    const view: ServerView = .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30, .total_ram = 48 << 30, .wired_in_use = 40960 << 20, .wire_now = .{
        .iogpu_mb = 41667,
        .global_user_bytes = 38_000_000_000,
        .user_bytes = 36_000_000_000,
    } };
    const body = try render(testing.allocator, "/h/s.json", &file, &.{}, &locks, view);
    defer testing.allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const g = parsed.value.object.get("gpu_limit").?.object;
    try testing.expectEqual(@as(i64, 48 << 30), g.get("total_ram_bytes").?.integer);
    // The sysctls as they read now; the in-use number differs until sushi restarts.
    try testing.expectEqual(@as(i64, 41667), g.get("iogpu_mb").?.integer);
    try testing.expectEqual(@as(i64, 38_000_000_000), g.get("global_user_wire_bytes").?.integer);
    try testing.expectEqual(@as(i64, 36_000_000_000), g.get("user_wire_bytes").?.integer);
    try testing.expectEqual(@as(i64, 40960), g.get("in_use_mb").?.integer);
    try testing.expectEqual(@as(i64, 45056), g.get("standard_mb").?.integer);
    try testing.expectEqualStrings("sudo sysctl iogpu.wired_limit_mb=45056 vm.global_user_wire_limit=47244640256 vm.user_wire_limit=47244640256", g.get("command").?.string);
    // Unknown RAM: no limit to suggest.
    const blind = try render(testing.allocator, "/h/s.json", &file, &.{}, &locks, .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30 });
    defer testing.allocator.free(blind);
    const p2 = try std.json.parseFromSlice(std.json.Value, testing.allocator, blind, .{});
    defer p2.deinit();
    try testing.expect(p2.value.object.get("gpu_limit").? == .null);
}

test "settings api: render reports skip_mem_preflight off by default, with the file value and the flag that pins it" {
    const view: ServerView = .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30 };
    var locks: model_settings.LaunchLocks = .{};
    {
        var file = try model_settings.parse(testing.allocator, "{\"server\": {\"skip_mem_preflight\": true}}");
        defer file.deinit();
        const body = try render(testing.allocator, "/h/s.json", &file, &.{}, &locks, view);
        defer testing.allocator.free(body);
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
        defer parsed.deinit();
        const o = parsed.value.object.get("server").?.object.get("skip_mem_preflight").?.object;
        try testing.expect(o.get("file").?.bool);
        try testing.expect(!o.get("default").?.bool);
        try testing.expectEqualStrings("file", o.get("source").?.string);
        try testing.expect(!o.get("locked").?.bool);
        try testing.expect(o.get("flag").? == .null);
        try testing.expectEqualStrings("applies at once", o.get("applies").?.string);
        const update = parsed.value.object.get("server").?.object.get("update_check").?.object;
        try testing.expect(update.get("default").?.bool);
    }
    var empty = try model_settings.parse(testing.allocator, "{}");
    defer empty.deinit();
    locks.skip_mem_preflight.pin("--skip-mem-preflight", .{});
    const body = try render(testing.allocator, "/h/s.json", &empty, &.{}, &locks, view);
    defer testing.allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const o = parsed.value.object.get("server").?.object.get("skip_mem_preflight").?.object;
    try testing.expectEqualStrings("flag", o.get("source").?.string);
    try testing.expect(o.get("locked").?.bool);
    try testing.expectEqualStrings("--skip-mem-preflight", o.get("flag").?.string);
}

test "settings api: continue_after_thought is on by default, takes a boolean or null at once, and the flag keeps it off" {
    defer server_state.setContinueAfterThought(true);
    const view: ServerView = .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30 };
    var locks: model_settings.LaunchLocks = .{};
    for ([_]struct { file: []const u8, source: []const u8 }{
        .{ .file = "{}", .source = "default" },
        .{ .file = "{\"server\": {\"continue_after_thought\": false}}", .source = "file" },
    }) |c| {
        var file = try model_settings.parse(testing.allocator, c.file);
        defer file.deinit();
        const body = try render(testing.allocator, "/h/s.json", &file, &.{}, &locks, view);
        defer testing.allocator.free(body);
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
        defer parsed.deinit();
        const o = parsed.value.object.get("server").?.object.get("continue_after_thought").?.object;
        try testing.expect(o.get("default").?.bool);
        try testing.expectEqualStrings(c.source, o.get("source").?.string);
        try testing.expect(!o.get("locked").?.bool);
    }
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    try testing.expect(!parseRequest(arena_state.allocator(), "{\"server\":{\"continue_after_thought\":false}}").ok.server_edits[0].value.?.bool);
    try testing.expect(parseRequest(arena_state.allocator(), "{\"server\":{\"continue_after_thought\":null}}").ok.server_edits[0].value == null);
    try expectRejected("{\"server\":{\"continue_after_thought\":\"off\"}}", 400, "invalid_setting");
    try expectRejected("{\"server\":{\"continue_after_thought\":0}}", 400, "invalid_setting");
    try testing.expectEqualStrings("applies at once", applyContinueAfterThought(&locks, .{ .bool = false }));
    try testing.expect(!server_state.continueAfterThought());
    _ = applyContinueAfterThought(&locks, null);
    try testing.expect(server_state.continueAfterThought());

    // The launch flag outranks the file: off whatever was saved, and a save under it is kept, not applied.
    try testing.expect(!model_settings.continueAfterThought(true, true).value);
    try testing.expect(model_settings.continueAfterThought(false, null).value);
    try testing.expect(!model_settings.continueAfterThought(false, false).value);
    server_state.setContinueAfterThought(false);
    locks.continue_after_thought.pin("--no-continue-after-thought", .{});
    try testing.expect(std.mem.startsWith(u8, applyContinueAfterThought(&locks, .{ .bool = true }), "saved; --no-continue-after-thought"));
    try testing.expect(!server_state.continueAfterThought());
    var empty = try model_settings.parse(testing.allocator, "{\"server\": {\"continue_after_thought\": true}}");
    defer empty.deinit();
    const body = try render(testing.allocator, "/h/s.json", &empty, &.{}, &locks, view);
    defer testing.allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const o = parsed.value.object.get("server").?.object.get("continue_after_thought").?.object;
    try testing.expectEqualStrings("flag", o.get("source").?.string);
    try testing.expect(o.get("locked").?.bool);
    try testing.expectEqualStrings("--no-continue-after-thought", o.get("flag").?.string);
}

test "settings api: request size limit accepts MiB or null and rejects invalid values" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const plan = parseRequest(arena_state.allocator(), "{\"server\":{\"max_request_mb\":5}}");
    try testing.expect(plan == .ok);
    try testing.expectEqual(@as(i64, 5), plan.ok.server_edits[0].value.?.integer);
    const reset = parseRequest(arena_state.allocator(), "{\"server\":{\"max_request_mb\":null}}");
    try testing.expect(reset == .ok);
    try testing.expect(reset.ok.server_edits[0].value == null);
    try testing.expect(parseRequest(arena_state.allocator(), "{\"server\":{\"max_request_mb\":1024}}") == .ok);
    for ([_][]const u8{ "0", "-1", "1025", "1.5", "\"5\"", "true" }) |value| {
        const body = try std.fmt.allocPrint(arena_state.allocator(), "{{\"server\":{{\"max_request_mb\":{s}}}}}", .{value});
        try expectRejected(body, 400, "invalid_setting");
    }
}

test "settings api: wired_margin_gib accepts 0..32 or null and rejects every other value" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const plan = parseRequest(arena_state.allocator(), "{\"server\":{\"wired_margin_gib\":8}}");
    try testing.expectEqual(@as(i64, 8), plan.ok.server_edits[0].value.?.integer);
    try testing.expect(parseRequest(arena_state.allocator(), "{\"server\":{\"wired_margin_gib\":null}}").ok.server_edits[0].value == null);
    for ([_][]const u8{ "0", "1", "32" }) |value| {
        const body = try std.fmt.allocPrint(arena_state.allocator(), "{{\"server\":{{\"wired_margin_gib\":{s}}}}}", .{value});
        try testing.expect(parseRequest(arena_state.allocator(), body) == .ok);
    }
    for ([_][]const u8{ "-1", "33", "1.5", "\"4\"", "true" }) |value| {
        const body = try std.fmt.allocPrint(arena_state.allocator(), "{{\"server\":{{\"wired_margin_gib\":{s}}}}}", .{value});
        try expectRejected(body, 400, "invalid_setting");
    }
}

test "settings api: a saved safety margin reaches the margin the ceiling reads at once, and the launch flag outranks it" {
    const flag_was = server_state.wired_margin_flag_bytes;
    defer server_state.wired_margin_flag_bytes = flag_was;
    defer server_state.setWiredMarginFile(null);
    server_state.wired_margin_flag_bytes = null;
    var locks: model_settings.LaunchLocks = .{};
    try testing.expectEqual(@as(u64, 1) << 30, server_state.wiredMarginBytes());
    try testing.expectEqualStrings("applies at once: the next request and the next load use it", applyWiredMargin(&locks, .{ .integer = 6 }, false));
    try testing.expectEqual(@as(u64, 6) << 30, server_state.wiredMarginBytes());
    _ = applyWiredMargin(&locks, null, false);
    try testing.expectEqual(@as(u64, 1) << 30, server_state.wiredMarginBytes());

    server_state.wired_margin_flag_bytes = 3 << 30;
    locks.wired_margin_gib.pin("3", .{});
    try testing.expectEqualStrings("saved; --wired-margin-gib still sets the margin until a start without it", applyWiredMargin(&locks, .{ .integer = 6 }, false));
    try testing.expectEqual(@as(u64, 3) << 30, server_state.wiredMarginBytes());
}

test "a saved margin of 0 is 0 in the file, the readback and a fresh process" {
    const globals = SavedGlobals.take();
    defer globals.restore();
    server_state.wired_margin_flag_bytes = null;
    model_settings.launch_locks = .{};
    server_state.setWiredMarginFile(null);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [600]u8 = undefined;
    const path = try savedFile(&tmp, &path_buf, null);
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var host: TestHost = .{ .io = testing.io, .path = path };
    try testing.expect(try save(&host, a, testing.io, path, parseRequest(a, "{\"server\":{\"wired_margin_gib\":0}}").ok, "", null) == .saved);
    try testing.expectEqual(@as(u64, 0), server_state.wiredMarginBytes());
    const text = try tmp.dir.readFileAlloc(testing.io, "model-settings.json", a, .limited(1 << 16));
    const on_disk = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer on_disk.deinit();
    try testing.expectEqual(@as(i64, 0), on_disk.value.object.get("server").?.object.get("wired_margin_gib").?.integer);
    var reloaded = try model_settings.parse(a, text);
    defer reloaded.deinit();
    try testing.expectEqual(@as(?u32, 0), model_settings.server(&reloaded).wired_margin_gib);
    server_state.setWiredMarginFile(null);
    try testing.expectEqual(@as(u64, 1) << 30, server_state.wiredMarginBytes());
    server_state.setWiredMarginFile(model_settings.server(&reloaded).wired_margin_gib);
    try testing.expectEqual(@as(u64, 0), server_state.wiredMarginBytes());
    const body = try render(a, path, &reloaded, &.{}, &model_settings.launch_locks, .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30 });
    defer a.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
    defer parsed.deinit();
    const o = parsed.value.object.get("server").?.object.get("wired_margin_gib").?.object;
    try testing.expectEqual(@as(i64, 0), o.get("file").?.integer);
    try testing.expectEqualStrings("file", o.get("source").?.string);
    try testing.expectEqual(@as(i64, 0), o.get("min").?.integer);
}

test "settings api: render reports server.max_concurrent from the file, the default of 1 or the launch flag that pins it" {
    const view: ServerView = .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30 };
    var locks: model_settings.LaunchLocks = .{};
    const cases = [_]struct { file: []const u8, flag: ?[]const u8, source: []const u8 }{
        .{ .file = "{\"server\": {\"max_concurrent\": 3}}", .flag = null, .source = "file" },
        .{ .file = "{}", .flag = null, .source = "default" },
        .{ .file = "{\"server\": {\"max_concurrent\": 3}}", .flag = "4", .source = "flag" },
    };
    for (cases) |c| {
        locks.max_concurrent = .{};
        if (c.flag) |f| locks.max_concurrent.pin("{s}", .{f});
        var file = try model_settings.parse(testing.allocator, c.file);
        defer file.deinit();
        const body = try render(testing.allocator, "/h/s.json", &file, &.{}, &locks, view);
        defer testing.allocator.free(body);
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
        defer parsed.deinit();
        const o = parsed.value.object.get("server").?.object.get("max_concurrent").?.object;
        try testing.expectEqualStrings(c.source, o.get("source").?.string);
        try testing.expectEqual(c.flag != null, o.get("locked").?.bool);
        try testing.expectEqual(@as(i64, 1), o.get("default").?.integer);
        try testing.expectEqual(@as(i64, 1), o.get("min").?.integer);
        try testing.expectEqual(@as(i64, sched_policy.MAX_BATCH_GROUP), o.get("max").?.integer);
        if (c.flag) |f| try testing.expectEqualStrings(f, o.get("flag").?.string) else try testing.expect(o.get("flag").? == .null);
        if (std.mem.eql(u8, c.source, "file")) try testing.expectEqual(@as(i64, 3), o.get("file").?.integer) else try testing.expect(o.get("file").? == .integer or o.get("file").? == .null);
    }
}

test "settings api: a saved server.max_concurrent reaches the next load and ends a stored load failure; the launch flag keeps it from moving anything" {
    const globals = SavedGlobals.take();
    defer globals.restore();
    const flag_was = server_state.max_concurrent_flag;
    defer server_state.max_concurrent_flag = flag_was;
    server_state.wired_margin_flag_bytes = null;
    server_state.max_concurrent_flag = null;
    server_state.setMaxConcurrentFile(null);
    model_settings.launch_locks = .{};
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [600]u8 = undefined;
    const path = try savedFile(&tmp, &path_buf, null);
    var host: TestHost = .{ .io = testing.io, .path = path };
    try testing.expect(try save(&host, a, testing.io, path, parseRequest(a, "{\"server\":{\"max_concurrent\":3}}").ok, "", null) == .saved);
    try testing.expectEqualSlices(TestHost.Event, &.{ .write, .forget_all }, host.events[0..host.n_events]);
    try testing.expectEqual(@as(?u32, 3), server_state.maxConcurrentFile());

    var held: TestHost = .{ .io = testing.io, .path = path };
    model_settings.launch_locks.max_concurrent.pin("4", .{});
    const kept = try save(&held, a, testing.io, path, parseRequest(a, "{\"server\":{\"max_concurrent\":2}}").ok, "", null);
    try testing.expect(kept == .saved);
    try testing.expect(std.mem.startsWith(u8, kept.saved[0], "saved; --max-concurrent still"));
    try testing.expectEqual(@as(usize, 0), held.count(.forget_all));
    model_settings.launch_locks = .{};

    var cleared: TestHost = .{ .io = testing.io, .path = path };
    try testing.expect(try save(&cleared, a, testing.io, path, parseRequest(a, "{\"server\":{\"max_concurrent\":null}}").ok, "", null) == .saved);
    try testing.expectEqual(@as(?u32, null), server_state.maxConcurrentFile());
    try testing.expectEqual(@as(usize, 1), cleared.count(.forget_all));
}

test "settings api: render reports the safety margin from the file, the default or the launch flag that pins it" {
    const view: ServerView = .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30 };
    var locks: model_settings.LaunchLocks = .{};
    const cases = [_]struct { file: []const u8, flag: ?[]const u8, source: []const u8 }{
        .{ .file = "{\"server\": {\"wired_margin_gib\": 8}}", .flag = null, .source = "file" },
        .{ .file = "{}", .flag = null, .source = "default" },
        .{ .file = "{\"server\": {\"wired_margin_gib\": 8}}", .flag = "3", .source = "flag" },
    };
    for (cases) |c| {
        locks.wired_margin_gib = .{};
        if (c.flag) |f| locks.wired_margin_gib.pin("{s}", .{f});
        var file = try model_settings.parse(testing.allocator, c.file);
        defer file.deinit();
        const body = try render(testing.allocator, "/h/s.json", &file, &.{}, &locks, view);
        defer testing.allocator.free(body);
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
        defer parsed.deinit();
        const o = parsed.value.object.get("server").?.object.get("wired_margin_gib").?.object;
        try testing.expectEqualStrings(c.source, o.get("source").?.string);
        try testing.expectEqual(c.flag != null, o.get("locked").?.bool);
        try testing.expectEqual(@as(i64, model_settings.default_wired_margin_gib), o.get("default").?.integer);
        try testing.expectEqual(@as(i64, model_settings.wired_margin_gib_min), o.get("min").?.integer);
        try testing.expectEqual(@as(i64, model_settings.wired_margin_gib_max), o.get("max").?.integer);
        if (c.flag) |f| try testing.expectEqualStrings(f, o.get("flag").?.string) else try testing.expect(o.get("flag").? == .null);
        if (std.mem.eql(u8, c.source, "default")) try testing.expect(o.get("file").? == .null) else try testing.expectEqual(@as(i64, 8), o.get("file").?.integer);
        try testing.expectEqualStrings(if (c.flag != null) "set by --wired-margin-gib until a start without it" else "applies at once: the next request and the next load use it", o.get("applies").?.string);
    }
}

test "settings api: skip_mem_preflight takes a boolean or null, reaches the next load at once, and a pinning flag keeps it on" {
    defer scheduler.setSkipMemPreflight(false);
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    try testing.expect(parseRequest(arena_state.allocator(), "{\"server\":{\"skip_mem_preflight\":true}}").ok.server_edits[0].value.?.bool);
    try expectRejected("{\"server\":{\"skip_mem_preflight\":\"on\"}}", 400, "invalid_setting");
    var locks: model_settings.LaunchLocks = .{};
    try testing.expectEqualStrings("applies at once", applySkipMemPreflight(&locks, .{ .bool = true }));
    try testing.expect(scheduler.skipMemPreflight());
    _ = applySkipMemPreflight(&locks, null);
    try testing.expect(!scheduler.skipMemPreflight());
    scheduler.setSkipMemPreflight(true);
    locks.skip_mem_preflight.pin("--skip-mem-preflight", .{});
    _ = applySkipMemPreflight(&locks, .{ .bool = false });
    try testing.expect(scheduler.skipMemPreflight());
}

test "settings api: a kv_quant request in any spelling is stored as its choice name; a value the loader would not read is refused" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    for ([_][]const u8{ "0", "16", "\"16\"", "\"off\"" }) |spelling| {
        const body = try std.fmt.allocPrint(arena_state.allocator(), "{{\"model\":\"m\",\"set\":{{\"kv_quant\":{s}}}}}", .{spelling});
        const parsed = parseRequest(arena_state.allocator(), body);
        try testing.expectEqualStrings("off", parsed.ok.model_edits[0].value.?.string);
    }
    const eight = parseRequest(arena_state.allocator(), "{\"model\":\"m\",\"set\":{\"kv_quant\":8}}");
    try testing.expectEqualStrings("8", eight.ok.model_edits[0].value.?.string);
    try expectRejected("{\"model\":\"m\",\"set\":{\"kv_quant\":5}}", 400, "invalid_setting");
}

test "settings api: the SSD message says running tiers only when one was resized; models_dir reports what the rescan added" {
    try testing.expectEqualStrings("applied to the running tiers", cacheBudgetMessage(1));
    try testing.expectEqualStrings("saved; applies at the next model load", cacheBudgetMessage(0));
    var buf: [160]u8 = undefined;
    try testing.expectEqualStrings("saved; models rescanned, 2 added", modelsDirMessage(&buf, 2));
    try testing.expectEqualStrings("saved; the rescan failed, use /v1/models/rescan", modelsDirMessage(&buf, error.OutOfMemory));
}

test "settings api: render names the keys a model honours, what a loaded model loaded with, and the tiers in force" {
    var file = try model_settings.parse(testing.allocator,
        \\{"/m/a": {"kv_quant": 0}, "/m/b": {"kv_quant": "8"}}
    );
    defer file.deinit();
    const locks: model_settings.LaunchLocks = .{};
    const rows = [_]ModelRow{
        .{ .id = "a", .path = "/m/a", .state = "ready", .bytes = 5, .fit = .{ .arch = "prefetcher", .prefetch = true, .streams_experts = true, .has_vision = true }, .loaded = .{ .ctx_size = 4096, .kv_quant = null, .ssd_budget_gb = 60 }, .tier = .{ .cap_bytes = 2 << 30, .used_bytes = 7 } },
        .{ .id = "b", .path = "/m/b", .state = "unloaded", .bytes = 5, .fit = .{ .arch = "qwen4_exp", .streams_experts = false, .has_vision = false } },
    };
    const body = try render(testing.allocator, "/h/s.json", &file, &rows, &locks, .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30 });
    defer testing.allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    const cache = root.get("server").?.object.get("prefix_cache_disk_gb").?.object;
    try testing.expectEqual(@as(i64, 2 << 30), cache.get("effective_bytes").?.integer);
    try testing.expectEqual(@as(i64, 20 << 30), cache.get("ask_bytes").?.integer);
    const tier = cache.get("running_tiers").?.array.items[0].object;
    try testing.expectEqualStrings("a", tier.get("model").?.string);
    try testing.expectEqual(@as(i64, 7), tier.get("used_bytes").?.integer);

    const a = root.get("models").?.array.items[0].object;
    var has_prefetch = false;
    for (a.get("keys").?.array.items) |k| {
        try testing.expect(!std.mem.eql(u8, k.string, "kv_quant"));
        if (std.mem.eql(u8, k.string, "ssd_prefetch")) has_prefetch = true;
    }
    try testing.expect(has_prefetch);
    const lw = a.get("loaded_with").?.object;
    try testing.expectEqual(@as(i64, 4096), lw.get("ctx_size").?.integer);
    try testing.expect(lw.get("kv_quant").? == .null);
    try testing.expectEqualStrings("off", a.get("settings").?.object.get("kv_quant").?.object.get("file").?.string);

    const b = root.get("models").?.array.items[1].object;
    try testing.expect(b.get("loaded_with").? == .null);
    for (b.get("keys").?.array.items) |k| try testing.expect(!std.mem.startsWith(u8, k.string, "ssd_"));
}

test "settings api: a saved value the loader ignores is reported on its key as invalid, with the value and what the key takes" {
    // A cache bias saved under the old 0 to 8 range, beside a valid saved value and a key never saved.
    var file = try model_settings.parse(testing.allocator,
        \\{"/m/a": {"expert_cache_bias": 3, "kv_quant": "8"}}
    );
    defer file.deinit();
    const locks: model_settings.LaunchLocks = .{};
    const rows = [_]ModelRow{
        .{ .id = "a", .path = "/m/a", .state = "unloaded", .bytes = 5, .fit = .{ .arch = "prefetcher", .prefetch = true, .cache_bias = true, .streams_experts = true, .has_vision = true } },
    };
    const body = try render(testing.allocator, "/h/s.json", &file, &rows, &locks, .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30 });
    defer testing.allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const settings = parsed.value.object.get("models").?.array.items[0].object.get("settings").?.object;
    const bias = settings.get("expert_cache_bias").?.object;
    try testing.expect(bias.get("file").? == .null);
    try testing.expectEqualStrings("default", bias.get("source").?.string);
    const bad = bias.get("invalid").?.object;
    try testing.expectEqualStrings("expert_cache_bias", bad.get("key").?.string);
    try testing.expectEqual(@as(i64, 3), bad.get("value").?.integer);
    try testing.expectEqualStrings("a number from 0 to 2", bad.get("expected").?.string);
    try testing.expect(settings.get("kv_quant").?.object.get("invalid") == null);
    try testing.expect(settings.get("ssd_budget_gb").?.object.get("invalid") == null);
}

test "settings api: each model lists the limits its arch and pack put on its keys, and nothing for a key it does not constrain" {
    var file = try model_settings.parse(testing.allocator, "{}");
    defer file.deinit();
    const locks: model_settings.LaunchLocks = .{};
    const rows = [_]ModelRow{
        .{ .id = "latent", .path = "/m/a", .state = "unloaded", .bytes = 5, .fit = .{ .arch = "a", .streams_experts = false, .has_vision = false, .kv_choices = &.{ "off", "0", "16", "6", "8" }, .serves_mtp = false, .context_cap = 131072 } },
        .{ .id = "dropped", .path = "/m/b", .state = "unloaded", .bytes = 5, .fit = .{ .arch = "b", .streams_experts = true, .has_vision = false, .head_streams = false } },
        .{ .id = "plain", .path = "/m/c", .state = "unloaded", .bytes = 5, .fit = .{ .arch = "c", .streams_experts = true, .has_vision = true } },
        .{ .id = "bare", .path = "/m/d", .state = "unloaded", .bytes = 5, .fit = .{ .arch = "prefetcher", .prefetch = true, .streams_experts = true, .has_vision = true, .template_preserves_thinking = false } },
    };
    const body = try render(testing.allocator, "/h/s.json", &file, &rows, &locks, .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30 });
    defer testing.allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const models = parsed.value.object.get("models").?.array.items;

    const latent = models[0].object.get("limits").?.object;
    const kv = latent.get("kv_quant").?.object;
    try testing.expectEqual(@as(usize, 3), kv.get("choices").?.array.items.len);
    try testing.expectEqualStrings("6", kv.get("choices").?.array.items[1].string);
    try testing.expect(latent.get("mtp").?.object.get("disabled").?.bool);
    try testing.expect(latent.get("mtp").?.object.get("why").?.string.len > 0);
    try testing.expectEqual(@as(i64, 131072), latent.get("ctx_size").?.object.get("max").?.integer);

    const dropped = models[1].object.get("limits").?.object;
    try testing.expectEqualStrings("ssd_budget_gb", dropped.get("mtp").?.object.get("unset").?.array.items[0].string);
    try testing.expect(dropped.get("kv_quant") == null);

    const plain = models[2].object.get("limits").?.object;
    for ([_][]const u8{ "kv_quant", "mtp", "ctx_size", "ssd_budget_gb", "vision", "preserve_thinking" }) |k| try testing.expect(plain.get(k) == null);
    try testing.expect(plain.get("ssd_prefetch").?.object.get("disabled").?.bool);

    const bare = models[3].object.get("limits").?.object;
    try testing.expect(bare.get("kv_quant").?.object.get("disabled").?.bool);
    try testing.expect(bare.get("preserve_thinking").?.object.get("disabled").?.bool);
    try testing.expect(bare.get("ssd_prefetch") == null);
}

test "settings api: a save the model cannot honour is refused with its reason, a removal never" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const latent: model_settings.Fit = .{ .arch = "a", .streams_experts = false, .has_vision = false, .kv_choices = &.{ "off", "0", "16", "6", "8" }, .serves_mtp = false };
    const four = refusalMessage(arena, latent, null, &.{.{ .key = "kv_quant", .value = .{ .string = "4" } }}).?;
    try testing.expect(std.mem.indexOf(u8, four, "\"kv_quant\"") != null);
    try testing.expect(refusalMessage(arena, latent, null, &.{.{ .key = "kv_quant", .value = .{ .string = "off" } }}) == null);
    const mtp = refusalMessage(arena, latent, null, &.{.{ .key = "mtp", .value = .{ .bool = false } }}).?;
    try testing.expect(std.mem.indexOf(u8, mtp, "speculative head") != null);
    try testing.expect(refusalMessage(arena, latent, null, &.{.{ .key = "mtp", .value = null }}) == null);
}

test "settings api: render reports the file's value, the pinning flag and the winner per key" {
    var file = try model_settings.parse(testing.allocator,
        \\{"/m/a": {"kv_quant": "4", "ssd_prefetch": 200, "mtp": false}, "server": {"prefix_cache_disk_gb": 30}}
    );
    defer file.deinit();
    var locks: model_settings.LaunchLocks = .{};
    locks.modelMut("mtp").pin("on", .{});
    const rows = [_]ModelRow{.{ .id = "a", .path = "/m/a", .state = "ready", .bytes = 5 }};
    const body = try render(testing.allocator, "/h/model-settings.json", &file, &rows, &locks, .{ .default_models_dir = "/h/.sushi/models", .boot_roots = &.{"/h/.sushi/models"}, .ask_bytes = 30 << 30 });
    defer testing.allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const cache = root.get("server").?.object.get("prefix_cache_disk_gb").?.object;
    try testing.expectEqual(@as(i64, 30), cache.get("file").?.integer);
    try testing.expectEqualStrings("file", cache.get("source").?.string);
    try testing.expect(!cache.get("locked").?.bool);
    // A saved size leaves the size "Use default" goes back to.
    try testing.expectEqual(@as(i64, @intCast(disk_placement.DEFAULT_DISK_CAP)), cache.get("default_bytes").?.integer);
    const s = root.get("models").?.array.items[0].object.get("settings").?.object;
    try testing.expectEqualStrings("file", s.get("kv_quant").?.object.get("source").?.string);
    try testing.expectEqualStrings("4", s.get("kv_quant").?.object.get("file").?.string);
    // The loader ignores 200 for ssd_prefetch, so the editor sees no file value.
    try testing.expect(s.get("ssd_prefetch").?.object.get("file").? == .null);
    try testing.expectEqualStrings("default", s.get("ssd_prefetch").?.object.get("source").?.string);
    const mtp = s.get("mtp").?.object;
    try testing.expectEqualStrings("flag", mtp.get("source").?.string);
    try testing.expect(mtp.get("locked").?.bool);
    try testing.expect(!mtp.get("file").?.bool);
    try testing.expectEqual(@as(usize, model_settings.model_keys.len), root.get("model_keys").?.array.items.len);
}

test "settings api: a ready model reports the default each key resolves to; an unloaded one reports none" {
    if (arch_config.ofConfigType("deepseek_v41") == null) return error.SkipZigTest;
    var file = try model_settings.parse(testing.allocator, "{}");
    defer file.deinit();
    const locks: model_settings.LaunchLocks = .{};
    const dsv41 = model_mod.ModelConfig{ .model_type = "deepseek_v41", .max_position_embeddings = 1_048_576, .serve_defaults = arch_config.releaseServeDefaults("dsv41") };
    const rows = [_]ModelRow{
        .{ .id = "ready", .path = "/m/a", .state = "ready", .bytes = 5, .defaults = resolvedDefaults(&dsv41) },
        .{ .id = "cold", .path = "/m/b", .state = "unloaded", .bytes = 5 },
    };
    const body = try render(testing.allocator, "/h/s.json", &file, &rows, &locks, .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30 });
    defer testing.allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const models = parsed.value.object.get("models").?.array.items;
    const d = models[0].object.get("defaults").?.object;
    try testing.expectEqual(@as(i64, 256_000), d.get("max_tokens").?.integer);
    try testing.expect(d.get("mtp").?.bool);
    try testing.expect(d.get("preserve_thinking").?.bool);
    try testing.expectEqual(@as(i64, 0), d.get("ssd_prefetch").?.integer);
    // Measured at load, so never guessed.
    try testing.expect(d.get("ctx_size") == null);
    try testing.expect(d.get("ssd_budget_gb") == null);
    try testing.expect(models[1].object.get("defaults").? == .null);
    // Every model's prefill chunk keys are a choice of five widths, defaulting to the model's own.
    const keys = parsed.value.object.get("model_keys").?.array.items;
    for (keys) |k| {
        const key = k.object;
        if (!std.mem.startsWith(u8, key.get("name").?.string, "prefill_chunk_")) continue;
        try testing.expectEqualStrings("choice", key.get("type").?.string);
        try testing.expectEqual(@as(usize, 5), key.get("choices").?.array.items.len);
    }
    try testing.expectEqual(@as(i64, @intCast(settings_resolve.defaultChunkMax(&dsv41))), d.get("prefill_chunk_max").?.integer);
    try testing.expectEqual(@as(i64, 512), d.get("prefill_chunk_min").?.integer);
}

test "settings api: max_concurrent and max_kv_tokens are offered with their defaults, and an exclusive model refuses more than one" {
    if (arch_config.ofConfigType("deepseek_v41") == null) return error.SkipZigTest;
    var file = try model_settings.parse(testing.allocator, "{}");
    defer file.deinit();
    const locks: model_settings.LaunchLocks = .{};
    const dsv41 = model_mod.ModelConfig{ .model_type = "deepseek_v41", .max_position_embeddings = 1_048_576, .serve_defaults = arch_config.releaseServeDefaults("dsv41") };
    const rows = [_]ModelRow{.{ .id = "ready", .path = "/m/a", .state = "ready", .bytes = 5, .defaults = resolvedDefaults(&dsv41), .fit = arch_flags.settingsFit(&dsv41) }};
    const body = try render(testing.allocator, "/h/s.json", &file, &rows, &locks, .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30 });
    defer testing.allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    var seen_concurrent = false;
    var seen_kv = false;
    for (root.get("model_keys").?.array.items) |k| {
        const key = k.object;
        if (std.mem.eql(u8, key.get("name").?.string, "max_concurrent")) {
            seen_concurrent = true;
            try testing.expectEqualStrings("int", key.get("type").?.string);
            try testing.expectEqual(@as(i64, 1), key.get("min").?.integer);
        }
        if (std.mem.eql(u8, key.get("name").?.string, "max_kv_tokens")) {
            seen_kv = true;
            try testing.expectEqualStrings("int", key.get("type").?.string);
            try testing.expectEqual(@as(i64, 0), key.get("min").?.integer);
        }
    }
    try testing.expect(seen_concurrent and seen_kv);
    const d = root.get("models").?.array.items[0].object.get("defaults").?.object;
    // No flag, no key and no saved server key: one request at a time.
    try testing.expectEqual(@as(i64, 1), d.get("max_concurrent").?.integer);
    try testing.expectEqual(@as(i64, 0), d.get("max_kv_tokens").?.integer);

    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    // A resident DeepSeek load batches rows, so its config is not "fixed at 1".
    const fit = arch_flags.settingsFit(&dsv41);
    try testing.expect(refusalMessage(arena_state.allocator(), fit, null, &.{.{ .key = "max_concurrent", .value = .{ .integer = 2 } }}) == null);
    try testing.expect(refusalMessage(arena_state.allocator(), fit, null, &.{.{ .key = "max_kv_tokens", .value = .{ .integer = 4096 } }}) == null);
    // Only an arch whose decode can never batch (Hy4-Flash) is fixed at 1 with its reason.
    const hy4 = model_mod.ModelConfig{ .model_type = "hy_v4_flash" };
    const hfit = arch_flags.settingsFit(&hy4);
    try testing.expectEqual(@as(u32, 1), model_settings.limit("max_concurrent", hfit).max);
    const why = refusalMessage(arena_state.allocator(), hfit, null, &.{.{ .key = "max_concurrent", .value = .{ .integer = 2 } }}).?;
    try testing.expect(std.mem.indexOf(u8, why, "one request at a time") != null);

    // A non-exclusive model takes no more rows than its own batched tick carries.
    const qwen = model_mod.ModelConfig{ .model_type = "qwen4_exp" };
    const qfit = arch_flags.settingsFit(&qwen);
    try testing.expectEqual(@as(u32, 8), model_settings.limit("max_concurrent", qfit).max);
    const cap_why = refusalMessage(arena_state.allocator(), qfit, null, &.{.{ .key = "max_concurrent", .value = .{ .integer = 9 } }}).?;
    try testing.expect(std.mem.indexOf(u8, cap_why, "joins at most this many") != null);
    try testing.expect(refusalMessage(arena_state.allocator(), qfit, null, &.{.{ .key = "max_concurrent", .value = .{ .integer = 8 } }}) == null);
}

test "settings api: a model's default max_concurrent is the value it inherits from the launch flag or the saved server key" {
    const flag_was = server_state.max_concurrent_flag;
    defer server_state.max_concurrent_flag = flag_was;
    defer server_state.setMaxConcurrentFile(null);
    server_state.max_concurrent_flag = null;
    const qwen = model_mod.ModelConfig{ .model_type = "qwen4_exp", .max_concurrent_override = 2 };
    server_state.setMaxConcurrentFile(null);
    try testing.expectEqual(@as(?u32, 1), resolvedDefaults(&qwen).max_concurrent);
    server_state.setMaxConcurrentFile(3);
    try testing.expectEqual(@as(?u32, 3), resolvedDefaults(&qwen).max_concurrent);
    server_state.max_concurrent_flag = 4;
    try testing.expectEqual(@as(?u32, 4), resolvedDefaults(&qwen).max_concurrent);
}

test "settings api: the removed prefix-cache entry key is refused by name" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const parsed = parseRequest(arena_state.allocator(), "{\"model\":\"m\",\"set\":{\"prefix_cache_entries\":32}}");
    try testing.expectEqual(@as(u32, 400), parsed.rejected.status);
    try testing.expect(std.mem.indexOf(u8, parsed.rejected.message, "prefix_cache_entries") != null);
    try testing.expect(std.mem.indexOf(u8, parsed.rejected.message, "removed") != null);
}

test "settings api: the resolved defaults ignore the file's values the config was stamped with" {
    var stamped = model_mod.ModelConfig{ .model_type = "qwen4_exp", .max_position_embeddings = 262_144 };
    model_load.applyModelSettings(&stamped, .{ .max_tokens = 999, .mtp = false, .kv_quant = kv_quant_mod.KVQuantConfig.affine(4), .vision = false, .ssd_prefetch = 40 });
    const d = resolvedDefaults(&stamped);
    try testing.expectEqual(@as(?u32, request_policy.ENGINE_DEFAULT_MAX_TOKENS), d.max_tokens);
    try testing.expectEqual(@as(?bool, true), d.mtp);
    try testing.expectEqual(@as(u8, 8), d.kv_quant.?.bits);
    try testing.expectEqual(@as(?bool, true), d.vision);
    try testing.expectEqual(@as(?u8, 0), d.ssd_prefetch);
    // What it was stamped with is still what the model loaded with.
    try testing.expectEqual(@as(?u32, 999), loadedWith(&stamped).max_tokens);
}

test "settings api: a model save is judged against the file it rewrites, so two concurrent saves never land a pair either alone is refused for" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const saved_globals = SavedGlobals.take();
    defer saved_globals.restore();
    // A streamed model whose MTP head cannot run while its experts stream: `mtp` is refused while `ssd_budget_gb` is set.
    const fit: model_settings.Fit = .{ .arch = "a", .streams_experts = true, .has_vision = false, .head_streams = false };
    // Save A asked for `mtp` when the file had no budget; save B's budget landed before A took the write lock.
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [600]u8 = undefined;
    const path = try savedFile(&tmp, &path_buf, "{\"/m/a\":{\"ssd_budget_gb\":20}}");
    var host: TestHost = .{ .io = testing.io, .path = path };
    const outcome = try save(&host, a, testing.io, path, parseRequest(a, "{\"model\":\"m\",\"set\":{\"mtp\":true}}").ok, "/m/a", fit);
    try testing.expect(outcome == .refused);
    try testing.expect(std.mem.indexOf(u8, outcome.refused, "\"mtp\"") != null);
    try testing.expectEqual(@as(usize, 0), host.count(.write));
    // The same save against a file with no budget lands.
    const path2 = try savedFile(&tmp, &path_buf, "{}");
    var host2: TestHost = .{ .io = testing.io, .path = path2 };
    try testing.expect(try save(&host2, a, testing.io, path2, parseRequest(a, "{\"model\":\"m\",\"set\":{\"mtp\":true}}").ok, "/m/a", fit) == .saved);
}

test "settings api: a max_kv_tokens save below the --ctx-size in force is refused naming the flag, with or without a file ctx_size" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const saved_globals = SavedGlobals.take();
    defer saved_globals.restore();
    const fit: model_settings.Fit = .{ .arch = "a", .streams_experts = false, .has_vision = false };
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [600]u8 = undefined;
    const path = try savedFile(&tmp, &path_buf, "{}");
    var host: TestHost = .{ .io = testing.io, .path = path };
    const body = "{\"model\":\"m\",\"set\":{\"max_kv_tokens\":4096}}";
    // No flag and no file ctx_size: nothing to be above.
    try testing.expect(try save(&host, a, testing.io, path, parseRequest(a, body).ok, "/m/a", fit) == .saved);
    model_settings.launch_locks.modelMut("ctx_size").pin("{d}", .{65536});
    const refused = try save(&host, a, testing.io, path, parseRequest(a, body).ok, "/m/a", fit);
    try testing.expect(refused == .refused);
    try testing.expect(std.mem.indexOf(u8, refused.refused, "\"max_kv_tokens\"") != null);
    try testing.expect(std.mem.indexOf(u8, refused.refused, "--ctx-size 65536 is above this model's max_kv_tokens 4096") != null);
    // A budget the flag fits, and removing the budget, are never blocked by it.
    try testing.expect(try save(&host, a, testing.io, path, parseRequest(a, "{\"model\":\"m\",\"set\":{\"max_kv_tokens\":65536}}").ok, "/m/a", fit) == .saved);
    try testing.expect(try save(&host, a, testing.io, path, parseRequest(a, "{\"model\":\"m\",\"set\":{\"max_kv_tokens\":null}}").ok, "/m/a", fit) == .saved);
}

test "settings api: max_tokens round-trips as a positive integer and is refused above the context window" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const ok = parseRequest(arena_state.allocator(), "{\"model\":\"m\",\"set\":{\"max_tokens\":32000}}");
    try testing.expectEqual(@as(i64, 32000), ok.ok.model_edits[0].value.?.integer);
    try expectRejected("{\"model\":\"m\",\"set\":{\"max_tokens\":0}}", 400, "invalid_setting");
    try expectRejected("{\"model\":\"m\",\"set\":{\"max_tokens\":\"big\"}}", 400, "invalid_setting");
    try expectRejected("{\"model\":\"m\",\"set\":{\"max_tokens\":1.5}}", 400, "invalid_setting");
    const fit: model_settings.Fit = .{ .arch = "a", .streams_experts = false, .has_vision = false, .context_cap = 131072 };
    const edit = [_]Edit{.{ .key = "max_tokens", .value = .{ .integer = 200_000 } }};
    const msg = refusalMessage(arena_state.allocator(), fit, null, &edit).?;
    try testing.expect(std.mem.indexOf(u8, msg, "\"max_tokens\"") != null);
    try testing.expect(std.mem.indexOf(u8, msg, "131072") != null);
    try testing.expect(refusalMessage(arena_state.allocator(), fit, null, &.{.{ .key = "max_tokens", .value = .{ .integer = 131072 } }}) == null);
}

fn renderJson(file: *const model_settings.Settings, view: ServerView) !std.json.Parsed(std.json.Value) {
    const locks: model_settings.LaunchLocks = .{};
    const body = try render(testing.allocator, "/h/s.json", file, &.{}, &locks, view);
    defer testing.allocator.free(body);
    return std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
}

fn effortFamily(root: std.json.Value, family: []const u8) std.json.ObjectMap {
    for (root.object.get("effort").?.object.get("families").?.array.items) |f| {
        if (std.mem.eql(u8, f.object.get("family").?.string, family)) return f.object;
    }
    unreachable;
}

fn effortRow(fam: std.json.ObjectMap, word: []const u8) std.json.ObjectMap {
    for (fam.get("rows").?.array.items) |r| if (std.mem.eql(u8, r.object.get("word").?.string, word)) return r.object;
    unreachable;
}

test "settings api: an effort save names a family and its rows, and a value its template cannot take is refused by row" {
    if (!effort_mod.all_built) return;
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const ok = parseRequest(a,
        \\{"effort":{"qwen4_exp":{"default":"low","high":{"offered":true,"sends":"medium","cap":64},"max":null},"glm5_next":null}}
    ).ok;
    const q = ok.effort[@intFromEnum(effort_mod.Family.qwen4_exp)];
    try testing.expect(q.named and !q.reset);
    try testing.expectEqual(@as(?effort_mod.Effort, .low), q.default.?);
    try testing.expectEqualDeep(effort_mod.RowEdit{ .offered = true, .sends = .{ .word = .medium }, .cap = @as(?i32, 64) }, q.rows[effort_mod.rowIndex(.high).?].?.?);
    try testing.expect(q.rows[effort_mod.rowIndex(.max).?].? == null);
    try testing.expect(q.rows[effort_mod.rowIndex(.low).?] == null);
    try testing.expect(ok.effort[@intFromEnum(effort_mod.Family.glm5_next)].reset);
    try testing.expect(!ok.effort[@intFromEnum(effort_mod.Family.deepseek_v41)].named);
    try expectRejected("{\"effort\":{\"mimo_v2\":{\"default\":\"on\"}}}", 400, "unknown_setting");
    try expectRejected("{\"effort\":{\"qwen4_exp\":{\"ultra\":{}}}}", 400, "invalid_setting");
    try expectRejected("{\"effort\":{\"qwen4_exp\":{\"high\":{\"sends\":\"high\"}}}}", 400, "invalid_setting");
    try expectRejected("{\"effort\":{\"deepseek_v41\":{\"medium\":{\"sends\":0}}}}", 400, "invalid_setting");
    try expectRejected("{\"effort\":{\"glm5_next\":{\"low\":{\"cap\":-1}}}}", 400, "invalid_setting");
    try expectRejected("{\"effort\":[]}", 400, "invalid_request_error");
    switch (parseRequest(a, "{\"effort\":{\"qwen4_exp\":{\"high\":{\"sends\":\"high\"}}}}")) {
        .ok => return error.TestExpectedRejection,
        .rejected => |r| try testing.expectEqualStrings("effort qwen4_exp.high.sends: Qwen takes low, medium or xhigh", r.message),
    }
}

test "settings api: GET carries each family's table, built-in and served value per field, and the --think lock" {
    if (!effort_mod.all_built) return;
    defer for (effort_mod.families) |f| effort_mod.publish(f, .{});
    var file = try model_settings.parse(testing.allocator, "{}");
    defer file.deinit();
    var e: effort_mod.Edits = .{};
    e.rows[effort_mod.rowIndex(.medium).?] = .{ .sends = .{ .scale = 40 }, .cap = @as(?i32, 64) };
    effort_mod.publish(.deepseek_v41, e);
    const view: ServerView = .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 0, .think = .xhigh };
    const parsed = try renderJson(&file, view);
    defer parsed.deinit();
    const ef = parsed.value.object.get("effort").?.object;
    try testing.expectEqualStrings("applies at once", ef.get("applies").?.string);
    try testing.expectEqual(effort_mod.families.len, ef.get("families").?.array.items.len);
    const ds = effortFamily(parsed.value, "deepseek_v41");
    try testing.expectEqualStrings("scale", ds.get("sends").?.object.get("kind").?.string);
    try testing.expectEqual(@as(i64, 50), ds.get("sends").?.object.get("presets").?.object.get("low").?.integer);
    const med = effortRow(ds, "medium");
    try testing.expectEqual(@as(i64, 62), med.get("builtin").?.object.get("sends").?.integer);
    try testing.expectEqual(@as(i64, 40), med.get("file").?.object.get("sends").?.integer);
    try testing.expectEqual(@as(i64, 64), med.get("value").?.object.get("cap").?.integer);
    try testing.expect(effortRow(ds, "low").get("file").? == .null);
    const ds_default = ds.get("default").?.object;
    try testing.expect(ds_default.get("locked").?.bool);
    try testing.expectEqualStrings("--think xhigh", ds_default.get("flag").?.string);
    try testing.expectEqualStrings("xhigh", ds_default.get("value").?.string);
    try testing.expectEqualStrings("high", ds_default.get("builtin").?.string);
    const qwen = effortFamily(parsed.value, "qwen4_exp");
    const high = effortRow(qwen, "high").get("builtin").?.object;
    try testing.expect(!high.get("offered").?.bool);
    try testing.expectEqualStrings("xhigh", high.get("fold_to").?.string);
    try testing.expectEqual(@as(i64, 2048), effortRow(qwen, "low").get("builtin").?.object.get("cap").?.integer);
    try testing.expect(effortRow(qwen, "off").get("builtin").?.object.get("sends").? == .null);
    const glm = effortFamily(parsed.value, "glm5_next");
    try testing.expect(!glm.get("off_offerable").?.bool);
    // --think xhigh on GLM is served as max, so it pins GLM's default too.
    try testing.expectEqualStrings("max", glm.get("default").?.object.get("value").?.string);
    const plain = try renderJson(&file, .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 0 });
    defer plain.deinit();
    try testing.expect(!effortFamily(plain.value, "qwen4_exp").get("default").?.object.get("locked").?.bool);
}

test "settings api: an effort save lands on the table it names whichever families are built" {
    defer for (effort_mod.families) |f| effort_mod.publish(f, .{});
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    var path_buf: [600]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/model-settings.json", .{dir});
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    for (effort_mod.families) |f| {
        const body = try std.fmt.allocPrint(a, "{{\"effort\":{{\"{s}\":{{\"high\":{{\"cap\":64}}}}}}}}", .{@tagName(f)});
        const plan = parseRequest(a, body).ok;
        for (effort_mod.families) |g| try testing.expectEqual(g == f, plan.effort[@intFromEnum(g)].named);
        try testing.expect(try applyEffort(a, io, path, plan.effort) == null);
        for (effort_mod.families) |g| {
            const cap = effort_mod.publishedEdits(g).rows[effort_mod.rowIndex(.high).?].cap;
            try testing.expectEqual(if (g == f) @as(??i32, @as(?i32, 64)) else null, cap);
        }
        for (effort_mod.families) |g| effort_mod.publish(g, .{});
    }
}

test "settings api: an effort save writes only what differs from built-in, takes hold at once, and a reset clears it" {
    if (!effort_mod.all_built) return;
    defer for (effort_mod.families) |f| effort_mod.publish(f, .{});
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    var path_buf: [600]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/model-settings.json", .{dir});
    try tmp.dir.writeFile(io, .{ .sub_path = "model-settings.json", .data = "{\"/m/a\":{\"ctx_size\":64}}" });
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    const set = parseRequest(a, "{\"effort\":{\"qwen4_exp\":{\"high\":{\"offered\":true,\"sends\":\"medium\",\"cap\":64},\"low\":{\"offered\":true,\"cap\":2048}}}}").ok;
    try testing.expect(try applyEffort(a, io, path, set.effort) == null);
    try testing.expectEqualDeep(effort_mod.RowEdit{ .offered = true, .sends = .{ .word = .medium }, .cap = @as(?i32, 64) }, effort_mod.publishedEdits(.qwen4_exp).rows[effort_mod.rowIndex(.high).?]);
    {
        var s = model_settings.load(testing.allocator, io, path);
        defer s.deinit();
        const sec = s.parsed.?.value.object.get("effort:qwen4_exp").?.object;
        try testing.expectEqual(@as(usize, 1), sec.count());
        try testing.expectEqual(@as(i64, 64), sec.get("high").?.object.get("cap").?.integer);
        try testing.expectEqual(@as(?u32, 64), s.lookup("/m/a").ctx_size);
    }
    // The default word must be offered; nothing is written when it is not.
    const bad = parseRequest(a, "{\"effort\":{\"qwen4_exp\":{\"medium\":{\"offered\":false}}}}").ok;
    try testing.expectEqualStrings("effort qwen4_exp.default: the default word must be offered", (try applyEffort(a, io, path, bad.effort)).?);
    try testing.expect(effort_mod.publishedEdits(.qwen4_exp).rows[effort_mod.rowIndex(.medium).?].offered == null);
    // A row named again replaces the stored row; a row not named keeps it.
    const row = parseRequest(a, "{\"effort\":{\"qwen4_exp\":{\"default\":\"high\",\"low\":{\"cap\":null}}}}").ok;
    try testing.expect(try applyEffort(a, io, path, row.effort) == null);
    try testing.expectEqual(effort_mod.Effort.high, effort_mod.table(.qwen4_exp).default);
    try testing.expectEqual(@as(?i32, null), effort_mod.table(.qwen4_exp).row(.low).cap);
    try testing.expectEqual(@as(?i32, 64), effort_mod.table(.qwen4_exp).row(.high).cap);
    // Reset: the section goes, the model's section stays.
    const reset = parseRequest(a, "{\"effort\":{\"qwen4_exp\":null}}").ok;
    try testing.expect(try applyEffort(a, io, path, reset.effort) == null);
    try testing.expectEqualDeep(effort_mod.builtin(.qwen4_exp), effort_mod.table(.qwen4_exp));
    var s = model_settings.load(testing.allocator, io, path);
    defer s.deinit();
    try testing.expect(s.parsed.?.value.object.get("effort:qwen4_exp") == null);
    try testing.expectEqual(@as(?u32, 64), s.lookup("/m/a").ctx_size);
}

test "settings api: an effort save cannot unoffer the word --think serves, which would refuse the next start" {
    if (!effort_mod.all_built) return;
    defer for (effort_mod.families) |f| effort_mod.publish(f, .{});
    const saved = effort_mod.think_effort_flag;
    defer effort_mod.think_effort_flag = saved;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    var path_buf: [600]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/model-settings.json", .{dir});
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    effort_mod.think_effort_flag = .xhigh;
    const unoffer = parseRequest(a, "{\"effort\":{\"qwen4_exp\":{\"xhigh\":{\"offered\":false}}}}").ok;
    try testing.expect(try applyEffort(a, io, path, unoffer.effort) != null);
    try testing.expect(effort_mod.thinkFlagArm("qwen4_exp") != null);
    // A word --think never resolved on a family is no reason to refuse that family's save.
    effort_mod.think_effort_flag = .off;
    const glm = parseRequest(a, "{\"effort\":{\"glm5_next\":{\"low\":{\"cap\":64}}}}").ok;
    try testing.expect(try applyEffort(a, io, path, glm.effort) == null);
    // Without --think the row may go.
    effort_mod.think_effort_flag = null;
    try testing.expect(try applyEffort(a, io, path, unoffer.effort) == null);
}

test "settings api: two effort saves at once each keep their row, in the file and in the served table" {
    if (!effort_mod.all_built) return;
    defer for (effort_mod.families) |f| effort_mod.publish(f, .{});
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    var path_buf: [600]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/model-settings.json", .{dir});
    const Saver = struct {
        fn run(p: []const u8, body: []const u8, go: *std.atomic.Value(u32), round: u32) void {
            while (go.load(.acquire) < round) std.atomic.spinLoopHint();
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const a = arena.allocator();
            _ = applyEffort(a, testing.io, p, parseRequest(a, body).ok.effort) catch {};
        }
    };
    var go = std.atomic.Value(u32).init(0);
    var round: u32 = 1;
    while (round <= 60) : (round += 1) {
        var low_buf: [96]u8 = undefined;
        var medium_buf: [96]u8 = undefined;
        const low = try std.fmt.bufPrint(&low_buf, "{{\"effort\":{{\"qwen4_exp\":{{\"low\":{{\"cap\":{d}}}}}}}}}", .{round});
        const medium = try std.fmt.bufPrint(&medium_buf, "{{\"effort\":{{\"qwen4_exp\":{{\"medium\":{{\"cap\":{d}}}}}}}}}", .{round});
        const t1 = try std.Thread.spawn(.{}, Saver.run, .{ path, low, &go, round });
        const t2 = try std.Thread.spawn(.{}, Saver.run, .{ path, medium, &go, round });
        go.store(round, .release);
        t1.join();
        t2.join();
        var file = model_settings.load(testing.allocator, io, path);
        defer file.deinit();
        const saved = model_settings.effortEdits(&file, .qwen4_exp, false);
        const served = effort_mod.publishedEdits(.qwen4_exp);
        for ([_]effort_mod.Edits{ saved, served }) |e| {
            try testing.expectEqual(@as(??i32, @as(i32, @intCast(round))), e.rows[effort_mod.rowIndex(.low).?].cap);
            try testing.expectEqual(@as(??i32, @as(i32, @intCast(round))), e.rows[effort_mod.rowIndex(.medium).?].cap);
        }
    }
}

test "settings api: a start reads the file's tables, skipping each bad field with a log line" {
    if (!effort_mod.all_built) return;
    defer for (effort_mod.families) |f| effort_mod.publish(f, .{});
    var file = try model_settings.parse(testing.allocator,
        \\{"effort:glm5_next":{"low":{"cap":64,"sends":"medium"},"off":{"offered":true}},"effort:deepseek_v41":{"default":"medium","medium":{"offered":false}}}
    );
    defer file.deinit();
    model_settings.publishEffortTables(&file);
    try testing.expectEqualDeep(effort_mod.Row{ .offered = true, .sends = .{ .word = .low }, .cap = 64 }, effort_mod.table(.glm5_next).row(.low));
    try testing.expect(!effort_mod.table(.glm5_next).row(.off).offered);
    // An unoffered default drops the file's default first.
    try testing.expectEqual(effort_mod.Effort.high, effort_mod.table(.deepseek_v41).default);
    try testing.expect(!effort_mod.table(.deepseek_v41).row(.medium).offered);
    try testing.expectEqualDeep(effort_mod.builtin(.qwen4_exp), effort_mod.table(.qwen4_exp));
}

test "settings api: yarn_factor is offered where the arch reads YaRN, carries the config's own RoPE, and reports the launch flag that outranks it" {
    var file = try model_settings.parse(testing.allocator, "{\"/m/a\":{\"yarn_factor\":2}}");
    defer file.deinit();
    var locks: model_settings.LaunchLocks = .{};
    const rows = [_]ModelRow{
        .{ .id = "qwen", .path = "/m/a", .state = "unloaded", .bytes = 5, .rope = .{ .native = 262144, .window = 262144 }, .fit = .{ .arch = "qwen4_exp", .streams_experts = false, .has_vision = false, .reads_yarn = true } },
        .{ .id = "glm", .path = "/m/b", .state = "unloaded", .bytes = 5, .fit = .{ .arch = "glm5_next", .streams_experts = false, .has_vision = false, .reads_yarn = false } },
    };
    const view: ServerView = .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30 };
    {
        const body = try render(testing.allocator, "/h/s.json", &file, &rows, &locks, view);
        defer testing.allocator.free(body);
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
        defer parsed.deinit();
        const models = parsed.value.object.get("models").?.array.items;
        var offered = false;
        for (models[0].object.get("keys").?.array.items) |k| offered = offered or std.mem.eql(u8, k.string, "yarn_factor");
        try testing.expect(offered);
        const rope = models[0].object.get("rope").?.object;
        try testing.expectEqual(@as(i64, 262144), rope.get("native").?.integer);
        try testing.expectEqualStrings("native", rope.get("method").?.string);
        try testing.expectEqualStrings("2", models[0].object.get("settings").?.object.get("yarn_factor").?.object.get("file").?.string);
        try testing.expect(!models[0].object.get("settings").?.object.get("yarn_factor").?.object.get("locked").?.bool);
        try testing.expect(models[1].object.get("rope").? == .null);
        try testing.expect(models[1].object.get("limits").?.object.get("yarn_factor").?.object.get("disabled").?.bool);
    }
    locks.modelMut("yarn_factor").pin("--config-overrides", .{});
    const body = try render(testing.allocator, "/h/s.json", &file, &rows, &locks, view);
    defer testing.allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const pinned = parsed.value.object.get("models").?.array.items[0].object.get("settings").?.object.get("yarn_factor").?.object;
    try testing.expect(pinned.get("locked").?.bool);
    try testing.expectEqualStrings("--config-overrides", pinned.get("flag").?.string);
}

test "settings api: a save of yarn_factor for an arch that does not read YaRN is refused with the reason" {
    const other: model_settings.Fit = .{ .arch = "glm5_next", .streams_experts = false, .has_vision = false, .reads_yarn = false };
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const text = refusalMessage(arena.allocator(), other, null, &.{.{ .key = "yarn_factor", .value = .{ .string = "4" } }}).?;
    try testing.expectEqualStrings("\"yarn_factor\" cannot be set for this model: This model's RoPE has no YaRN context extension.", text);
    try testing.expect(refusalMessage(arena.allocator(), other, null, &.{.{ .key = "yarn_factor", .value = null }}) == null);
}

test "settings api: ds41_cooler is offered only where the arch has Sushi Cooler, disabled with the reason where the pack cannot use it, locked by --ds41-mlx-engine, and defaults on" {
    var file = try model_settings.parse(testing.allocator, "{\"/m/a\": {\"ds41_cooler\": false}}");
    defer file.deinit();
    var locks: model_settings.LaunchLocks = .{};
    locks.modelMut("ds41_cooler").pin("--ds41-mlx-engine", .{});
    const rows = [_]ModelRow{
        .{ .id = "ds", .path = "/m/a", .state = "ready", .bytes = 5, .fit = .{ .arch = "d", .streams_experts = true, .has_vision = false, .cooler_setting = "ds41_cooler" }, .loaded = .{ .ds41_cooler = false }, .defaults = .{ .ds41_cooler = true } },
        .{ .id = "dense", .path = "/m/b", .state = "unloaded", .bytes = 5, .fit = .{ .arch = "g", .streams_experts = false, .has_vision = false } },
        .{ .id = "qwen", .path = "/m/c", .state = "unloaded", .bytes = 5, .fit = .{ .arch = "q", .streams_experts = false, .has_vision = false, .cooler_setting = "qwen_cooler" } },
    };
    const body = try render(testing.allocator, "/h/s.json", &file, &rows, &locks, .{ .default_models_dir = "/h", .boot_roots = &.{}, .ask_bytes = 20 << 30 });
    defer testing.allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const models = parsed.value.object.get("models").?.array.items;
    const has = struct {
        fn key(m: std.json.Value, name: []const u8) bool {
            for (m.object.get("keys").?.array.items) |k| if (std.mem.eql(u8, k.string, name)) return true;
            return false;
        }
    }.key;
    try testing.expect(has(models[0], "ds41_cooler") and !has(models[0], "qwen_cooler"));
    try testing.expect(!has(models[1], "ds41_cooler") and !has(models[1], "qwen_cooler"));
    try testing.expect(has(models[2], "qwen_cooler") and !has(models[2], "ds41_cooler"));
    const set = models[0].object.get("settings").?.object.get("ds41_cooler").?.object;
    try testing.expect(!set.get("file").?.bool);
    try testing.expectEqualStrings("--ds41-mlx-engine", set.get("flag").?.string);
    try testing.expect(set.get("locked").?.bool);
    try testing.expect(!models[0].object.get("loaded_with").?.object.get("ds41_cooler").?.bool);
    try testing.expect(models[0].object.get("defaults").?.object.get("ds41_cooler").?.bool);
    try testing.expect(models[0].object.get("limits").?.object.get("ds41_cooler") == null);
    try testing.expect(reachesLoad("ds41_cooler") and reachesLoad("qwen_cooler"));
}

/// A save's reach in a test: it writes the real file, answers the tier keys with fixed text, and records the order of
/// what the save did and what the running server read at the moment the stored load failures were forgotten.
const TestHost = struct {
    io: std.Io,
    path: []const u8,
    /// A write that carries this section fails, as a full disk fails it.
    fail_section: ?[]const u8 = null,
    events: [16]Event = undefined,
    n_events: usize = 0,
    margin_at_forget: u64 = 0,
    skip_at_forget: bool = false,
    max_request_at_forget: usize = 0,

    const Event = enum { write, cache_budget, models_dir, forget_all, forget_model, rearm };

    fn note(self: *TestHost, e: Event) void {
        self.events[self.n_events] = e;
        self.n_events += 1;
    }

    fn count(self: *const TestHost, e: Event) usize {
        var n: usize = 0;
        for (self.events[0..self.n_events]) |x| n += @intFromBool(x == e);
        return n;
    }

    fn write(self: *TestHost, arena: std.mem.Allocator, sections: []const model_settings.SectionEdits) !void {
        self.note(.write);
        if (self.fail_section) |bad| for (sections) |one| if (std.mem.eql(u8, one.section, bad)) return error.NoSpaceLeft;
        return model_settings.writeSectionsLocked(arena, self.io, self.path, sections);
    }

    fn sawServer(self: *TestHost) void {
        self.margin_at_forget = bills.ceilingNow().margin;
        self.skip_at_forget = scheduler.skipMemPreflight();
        self.max_request_at_forget = server_state.max_request_bytes.load(.monotonic);
    }

    fn forgetAll(self: *TestHost) void {
        self.sawServer();
        self.note(.forget_all);
    }

    fn forgetModel(self: *TestHost) void {
        self.sawServer();
        self.note(.forget_model);
    }

    fn cacheBudget(self: *TestHost, _: ?std.json.Value) []const u8 {
        self.note(.cache_budget);
        return "tier";
    }

    fn modelsDir(self: *TestHost, _: std.mem.Allocator) []const u8 {
        self.note(.models_dir);
        return "rescanned";
    }

    fn rearmDisk(self: *TestHost) void {
        self.note(.rearm);
    }
};

/// The process-wide values a save sets, put back after a test that saves.
const SavedGlobals = struct {
    skip: bool,
    max_request: usize,
    flag: ?u64,
    locks: model_settings.LaunchLocks,

    fn take() SavedGlobals {
        return .{
            .skip = scheduler.skipMemPreflight(),
            .max_request = server_state.max_request_bytes.load(.monotonic),
            .flag = server_state.wired_margin_flag_bytes,
            .locks = model_settings.launch_locks,
        };
    }

    fn restore(self: SavedGlobals) void {
        scheduler.setSkipMemPreflight(self.skip);
        server_state.max_request_bytes.store(self.max_request, .monotonic);
        server_state.wired_margin_flag_bytes = self.flag;
        server_state.setWiredMarginFile(null);
        server_state.setMaxConcurrentFile(null);
        model_settings.launch_locks = self.locks;
    }
};

fn savedFile(tmp: *testing.TmpDir, path_buf: []u8, body: ?[]const u8) ![]const u8 {
    var dir_buf: [512]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(testing.io, &dir_buf)];
    if (body) |text| try tmp.dir.writeFile(testing.io, .{ .sub_path = "model-settings.json", .data = text });
    return std.fmt.bufPrint(path_buf, "{s}/model-settings.json", .{dir});
}

/// The effort tables alone, saved to the real file the way a request that names nothing else saves them.
fn applyEffort(arena: std.mem.Allocator, io: std.Io, path: []const u8, plans: [effort_mod.family_count]EffortPlan) !?[]const u8 {
    var host: TestHost = .{ .io = io, .path = path };
    return switch (try save(&host, arena, io, path, .{ .effort = plans }, "", null)) {
        .refused => |message| message,
        .saved => null,
    };
}

test "settings api: a save writes the file, applies every key, then forgets the stored load failures" {
    const globals = SavedGlobals.take();
    defer globals.restore();
    server_state.wired_margin_flag_bytes = null;
    model_settings.launch_locks = .{};
    scheduler.setSkipMemPreflight(false);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [600]u8 = undefined;
    const path = try savedFile(&tmp, &path_buf, null);
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var host: TestHost = .{ .io = testing.io, .path = path };
    const plan = parseRequest(a,
        \\{"server":{"wired_margin_gib":8,"skip_mem_preflight":true,"max_request_mb":64,"prefix_cache_disk_gb":40,"models_dir":"/tmp"}}
    ).ok;
    try testing.expect(try save(&host, a, testing.io, path, plan, "", null) == .saved);
    try testing.expectEqualSlices(TestHost.Event, &.{ .write, .cache_budget, .models_dir, .forget_all, .rearm }, host.events[0..host.n_events]);
    try testing.expectEqual(@as(u64, 8) << 30, host.margin_at_forget);
    try testing.expect(host.skip_at_forget);
    try testing.expectEqual(@as(usize, 64) << 20, host.max_request_at_forget);
}

test "settings api: a save whose write fails leaves the file and the running server as they were" {
    const globals = SavedGlobals.take();
    defer globals.restore();
    server_state.wired_margin_flag_bytes = null;
    model_settings.launch_locks = .{};
    server_state.setWiredMarginFile(2);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const before = "{\"server\":{\"wired_margin_gib\":2}}";
    var path_buf: [600]u8 = undefined;
    const path = try savedFile(&tmp, &path_buf, before);
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var host: TestHost = .{ .io = testing.io, .path = path, .fail_section = "/m/a" };
    const plan = parseRequest(a, "{\"server\":{\"wired_margin_gib\":8},\"model\":\"m\",\"set\":{\"ctx_size\":4096}}").ok;
    try testing.expectError(error.NoSpaceLeft, save(&host, a, testing.io, path, plan, "/m/a", null));
    const kept = try tmp.dir.readFileAlloc(testing.io, "model-settings.json", a, .limited(1 << 16));
    try testing.expectEqualStrings(before, kept);
    try testing.expectEqual(@as(u64, 2) << 30, server_state.wiredMarginBytes());
    try testing.expectEqual(@as(usize, 0), host.count(.forget_all) + host.count(.forget_model));
}

test "settings api: a save forgets the stored load failures only for a key whose running value moved" {
    const globals = SavedGlobals.take();
    defer globals.restore();
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const Case = struct {
        file: []const u8,
        body: []const u8,
        /// `--wired-margin-gib` pins the margin.
        flag: bool = false,
        forgets: enum { none, all, model },
    };
    const cases = [_]Case{
        .{ .file = "{\"server\":{\"wired_margin_gib\":8}}", .body = "{\"server\":{\"wired_margin_gib\":8}}", .forgets = .none },
        .{ .file = "{\"server\":{\"wired_margin_gib\":2}}", .body = "{\"server\":{\"wired_margin_gib\":8}}", .forgets = .all },
        .{ .file = "{\"server\":{\"wired_margin_gib\":2}}", .body = "{\"server\":{\"wired_margin_gib\":8}}", .flag = true, .forgets = .none },
        .{ .file = "{}", .body = "{\"server\":{\"wired_margin_gib\":1}}", .forgets = .none },
        .{ .file = "{\"server\":{\"wired_margin_gib\":4}}", .body = "{\"server\":{\"wired_margin_gib\":null}}", .forgets = .all },
        .{ .file = "{\"server\":{\"max_request_mb\":64}}", .body = "{\"server\":{\"port\":9000}}", .forgets = .none },
        .{ .file = "{\"/m/a\":{\"ctx_size\":4096}}", .body = "{\"model\":\"m\",\"set\":{\"ctx_size\":4096}}", .forgets = .none },
        .{ .file = "{\"/m/a\":{\"ctx_size\":4096}}", .body = "{\"model\":\"m\",\"set\":{\"ctx_size\":8192}}", .forgets = .model },
        .{ .file = "{}", .body = "{\"model\":\"m\",\"set\":{\"ctx_size\":4096}}", .forgets = .model },
        .{ .file = "{\"/m/a\":{\"ctx_size\":4096}}", .body = "{\"model\":\"m\",\"set\":{\"ctx_size\":null}}", .forgets = .model },
        .{ .file = "{}", .body = "{\"model\":\"m\",\"set\":{\"ctx_size\":null}}", .forgets = .none },
    };
    for (cases) |c| {
        server_state.wired_margin_flag_bytes = if (c.flag) 3 << 30 else null;
        model_settings.launch_locks = .{};
        if (c.flag) model_settings.launch_locks.wired_margin_gib.pin("3", .{});
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        var path_buf: [600]u8 = undefined;
        const path = try savedFile(&tmp, &path_buf, c.file);
        var host: TestHost = .{ .io = testing.io, .path = path };
        try testing.expect(try save(&host, a, testing.io, path, parseRequest(a, c.body).ok, "/m/a", null) == .saved);
        try testing.expectEqual(c.forgets == .all, host.count(.forget_all) == 1);
        try testing.expectEqual(c.forgets == .model, host.count(.forget_model) == 1);
        try testing.expectEqual(@as(usize, if (c.forgets == .none) 0 else 1), host.count(.forget_all) + host.count(.forget_model));
    }
}

test "settings api: a save that moves the margin logs one new ceiling line, and one that moves nothing logs none" {
    const globals = SavedGlobals.take();
    defer globals.restore();
    const limit_was = bills.wired_limit_mb_override;
    const floor_was = bills.gpu_floor_bytes.load(.monotonic);
    const logged_was = bills.wired_floor_logged.load(.monotonic);
    defer {
        bills.wired_limit_mb_override = limit_was;
        bills.gpu_floor_bytes.store(floor_was, .monotonic);
        bills.wired_floor_logged.store(logged_was, .monotonic);
        log.closeFile();
    }
    bills.wired_limit_mb_override = 120_000;
    bills.gpu_floor_bytes.store(0, .monotonic);
    server_state.wired_margin_flag_bytes = null;
    model_settings.launch_locks = .{};
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [600]u8 = undefined;
    const path = try savedFile(&tmp, &path_buf, "{\"server\":{\"wired_margin_gib\":2}}");
    var log_buf: [640]u8 = undefined;
    const log_path = try std.fmt.bufPrint(&log_buf, "{s}.log", .{path});
    try log.openFile(log_path, 0);
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const status = @import("status.zig");
    const total_was = status.total_mem_override;
    const static_was = bills.static_ceiling_override;
    defer {
        status.total_mem_override = total_was;
        bills.static_ceiling_override = static_was;
    }
    status.total_mem_override = 128 << 30;
    bills.static_ceiling_override = 100 << 30;
    var host: TestHost = .{ .io = testing.io, .path = path };
    const bodies = [_][]const u8{
        "{\"server\":{\"wired_margin_gib\":2}}", // unchanged
        "{\"server\":{\"wired_margin_gib\":8}}", // moved
        "{\"server\":{\"wired_margin_gib\":8}}", // unchanged again
    };
    bills.wired_floor_logged.store(false, .monotonic);
    _ = bills.ceilingNow();
    var seen: usize = 1;
    const expect_new = [_]usize{ 0, 1, 0 };
    for (bodies, expect_new) |body, new| {
        try testing.expect(try save(&host, a, testing.io, path, parseRequest(a, body).ok, "", null) == .saved);
        _ = bills.ceilingNow();
        _ = bills.ceilingNow();
        const text = try tmp.dir.readFileAlloc(testing.io, "model-settings.json.log", a, .limited(1 << 16));
        const lines = std.mem.count(u8, text, "[mem] ceiling");
        try testing.expectEqual(seen + new, lines);
        seen = lines;
    }
    const text = try tmp.dir.readFileAlloc(testing.io, "model-settings.json.log", a, .limited(1 << 16));
    try testing.expect(std.mem.indexOf(u8, text, "margin 8192 MB") != null);
}
