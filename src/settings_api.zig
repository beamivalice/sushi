//! `GET/POST /v1/settings`: the settings file (`~/.sushi/model-settings.json`) as the web UI edits it. Server-wide keys
//! live under its `"server"` object, a model's under its absolute path. A launch flag outranks the file, so a key
//! a flag pins is reported locked; the file still takes the write, for the next start without that flag.
const std = @import("std");
const log = @import("log.zig");
const model_settings = @import("model_settings.zig");
const scheduler = @import("scheduler.zig");
const default_port = @import("default_port.zig");
const model_registry_mod = @import("model_registry.zig");
const disk_placement = @import("disk_placement.zig");
const http_conn = @import("http_conn.zig");
const server = @import("server.zig");
const model_discovery = @import("model_discovery.zig");
const expert_quant = @import("expert_quant.zig");
const model_mod = @import("model.zig");
const model_load = @import("model_load.zig");
const models_api = @import("models_api.zig");
const expert_stream = @import("expert_stream.zig");
const chat = @import("chat.zig");
const kv_quant_mod = @import("kv_quant.zig");
const update_mod = @import("update.zig");
const pack_update = @import("pack_update.zig");
const effort_mod = @import("effort.zig");

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
};

pub const ServerView = struct {
    default_models_dir: []const u8,
    boot_roots: []const []const u8,
    /// The server-wide ask, which loads without a number of their own start from.
    ask_bytes: u64,
    /// `--think <word>`, which pins each family's default word it binds.
    think: ?effort_mod.Effort = null,
};

/// The file value of one key as a JSON value, or null when the file had none.
fn overrideValue(o: model_settings.Override, name: []const u8) ?std.json.Value {
    const eql = std.mem.eql;
    if (eql(u8, name, "ctx_size")) return if (o.ctx_size) |v| .{ .integer = v } else null;
    if (eql(u8, name, "kv_quant")) return if (o.kv_quant) |v| .{ .string = v.wireName() } else null;
    if (eql(u8, name, "mtp")) return if (o.mtp) |v| .{ .bool = v } else null;
    if (eql(u8, name, "mtp_acceptance")) return if (o.mtp_acceptance) |v| .{ .string = @import("mtp_acceptance.zig").name(v) } else null;
    if (eql(u8, name, "mtp_greedy_tail")) return if (o.mtp_greedy_tail) |v| .{ .bool = v } else null;
    if (eql(u8, name, "ssd_budget_gb")) return if (o.ssd_budget_gb) |v| .{ .integer = v } else null;
    if (eql(u8, name, "ssd_prefetch")) return if (o.ssd_prefetch) |v| .{ .integer = v } else null;
    if (eql(u8, name, "preserve_thinking")) return if (o.preserve_thinking) |v| .{ .bool = v } else null;
    if (eql(u8, name, "think_penalty")) return if (o.think_penalty) |v| .{ .float = v } else null;
    if (eql(u8, name, "vision")) return if (o.vision) |v| .{ .bool = v } else null;
    if (eql(u8, name, "max_tokens")) return if (o.max_tokens) |v| .{ .integer = v } else null;
    return null;
}

/// The settings a ready model's load stamped on its config.
fn loadedWith(config: *const model_mod.ModelConfig) model_settings.Override {
    return .{
        .ctx_size = if (config.ctx_override > 0) config.ctx_override else null,
        .kv_quant = config.kv_quant_override,
        .mtp = config.mtp_override,
        .mtp_acceptance = config.mtp_acceptance_override,
        .mtp_greedy_tail = config.mtp_greedy_tail_override,
        .ssd_budget_gb = if (config.ssd_budget_gb_override > 0) config.ssd_budget_gb_override else null,
        .ssd_prefetch = config.ssd_prefetch_override,
        .preserve_thinking = config.preserve_thinking_override,
        .think_penalty = config.think_penalty_override,
        .vision = config.vision_override,
        .max_tokens = config.max_tokens_override,
    };
}

/// What a loaded model's keys resolve to when the file and the launch flags say nothing, read from the loader's own
/// resolvers on a config with its overrides cleared. `ctx_size` and `ssd_budget_gb` stay null: they are measured at load.
pub fn resolvedDefaults(config: *const model_mod.ModelConfig) model_settings.Override {
    var bare = config.*;
    model_load.applyModelSettings(&bare, .{});
    bare.ssd_prefetch = null;
    return .{
        .kv_quant = server.kvCacheForLaunch(&bare, kv_quant_mod.KVQuantConfig.engine_default, false).config,
        .mtp = model_load.mtpChoiceFor(false, false, &bare).on,
        .mtp_acceptance = model_settings.default_mtp_acceptance,
        .mtp_greedy_tail = model_settings.default_mtp_greedy_tail,
        .ssd_prefetch = bare.ssdPrefetchPct(),
        .preserve_thinking = model_settings.default_preserve_thinking,
        .think_penalty = model_settings.default_think_penalty,
        .vision = model_settings.VisionChoice.resolve(null, null, model_load.visionDefaultOn(&bare, bare.expert_streaming)).on,
        .max_tokens = bare.defaultMaxTokens(),
    };
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
    try w.print(",\"source\":\"{s}\",\"locked\":{}}}", .{ if (pinned) "flag" else if (valid) "file" else "default", pinned });
}

/// What a model's keys may be set to, for every key with a constraint: `disabled` (the key does nothing), `choices` (the
/// subset it stores), `max`, `requires` (another key's value it needs) and `unset` (keys that must stay unset), each with a `why`.
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
        if (l.requires_mtp) try w.writeAll("\"requires\":{\"mtp\":true},");
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
    effort: [effort_mod.families.len]EffortPlan = @splat(.{}),

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
        const spec = model_settings.findKey(specs, kv.key_ptr.*) orelse
            return reject(arena, 400, "unknown_setting", "\"{s}\" is not a setting this server has", .{kv.key_ptr.*});
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

/// Writes the named families' tables (only what differs from built-in) and publishes them for the next request. Every
/// family is checked before any is written; returns the refusal text, or null.
pub fn applyEffort(arena: std.mem.Allocator, io: std.Io, path: []const u8, plans: [effort_mod.families.len]EffortPlan) !?[]const u8 {
    // Read, check, write and publish under the write lock: a save never merges into a file another one is rewriting.
    model_settings.lockWrites(io);
    defer model_settings.unlockWrites(io);
    var file = model_settings.load(arena, io, path);
    defer file.deinit();
    var next: [effort_mod.families.len]effort_mod.Edits = undefined;
    for (effort_mod.families, plans, &next) |f, p, *n| {
        n.* = model_settings.effortEdits(&file, f, false);
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
    // Every family in one rewrite, so a failed write leaves none of them saved.
    var sections: [effort_mod.families.len]model_settings.SectionEdits = undefined;
    var n_sections: usize = 0;
    for (effort_mod.families, plans, next) |f, p, n| {
        if (!p.named) continue;
        const edits = try arena.alloc(Edit, effort_mod.table_words.len + 1);
        edits[0] = .{ .key = "default", .value = if (n.default) |d| .{ .string = @tagName(d) } else null };
        for (effort_mod.table_words, n.rows, 1..) |word, row, i| edits[i] = .{ .key = @tagName(word), .value = if (row.isEmpty()) null else try effort_mod.rowJson(arena, f, row) };
        var buf: [64]u8 = undefined;
        sections[n_sections] = .{ .section = try arena.dupe(u8, effort_mod.sectionName(&buf, f)), .edits = edits };
        n_sections += 1;
    }
    try model_settings.writeSectionsLocked(arena, io, path, sections[0..n_sections]);
    for (effort_mod.families, plans, next) |f, p, n| {
        if (!p.named) continue;
        effort_mod.publish(f, n);
        log.info("[effort] {s}: table saved, serving from the next request\n", .{@tagName(f)});
    }
    return null;
}

fn effortMessage(think: ?effort_mod.Effort, plans: [effort_mod.families.len]EffortPlan) []const u8 {
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
    const probe = model_mod.ModelConfig{ .model_type = arch };
    const latent = probe.isGlm5();
    const layout: expert_quant.Layout = if (sm.quantized_experts) .exl3_k4 else .mxfp4_w123;
    return .{
        .arch = arch,
        .streams_experts = geometry and expert_quant.isExpertStreamingArch(arch),
        .has_vision = sm.has_vision,
        .serves_mtp = !latent,
        .streaming_required = probe.isDsv41() or models_api.streamingRequiredMarker(arch, sm.quant_bits, sm.quantized_experts, geometry, row.index_complete),
        .head_streams = expert_stream.streamedMtpHeadSupported(probe.isQwen4(), probe.isDsv41(), layout),
        .stq_experts = sm.quantized_experts,
        .latent_kv = latent,
        .context_cap = sm.context_cap,
    };
}

/// The facts only the pack's files hold, read for a loaded model as for a stub: its head, and for a stub its chat template.
fn readPackFacts(io: std.Io, allocator: std.mem.Allocator, row: *ModelRow) void {
    if (row.loaded == null) row.fit = stubFit(io, allocator, row);
    row.fit.mtp_head = model_load.packShipsMtpHead(io, allocator, row.fit.arch, row.path);
    if (row.loaded != null) return;
    var chat_config = chat.loadChatConfig(io, allocator, row.path) catch return;
    defer chat_config.deinit();
    row.fit.template_preserves_thinking = chat.templateTakesPreserveThinking(chat_config.chat_template);
    row.fit.template_thinks = chat.templateHasThinkSpan(chat_config.chat_template);
}

/// An entry's row from what the registry holds; the caller holds its lock.
fn heldRow(e: *model_registry_mod.LoadedModel) ModelRow {
    var row: ModelRow = .{ .id = e.id, .path = e.path, .state = stateName(e.state), .bytes = e.bytes_on_disk, .index_complete = e.streaming_index_complete };
    row.fit.arch = e.arch_hint;
    if (e.state == .ready and e.config != null) {
        const c = e.config.?;
        row.fit = server.settingsFit(c);
        row.loaded = loadedWith(c);
        row.defaults = resolvedDefaults(c);
        if (e.chat_config) |cc| {
            row.fit.template_preserves_thinking = chat.templateTakesPreserveThinking(cc.chat_template);
            row.fit.template_thinks = chat.templateHasThinkSpan(cc.chat_template);
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
    const registry = server.global_registry orelse {
        try server.sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503);
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
    });
    defer allocator.free(body);
    try server.sendResponse(stream, "200 OK", "application/json", body);
}

fn dirReadable(io: std.Io, path: []const u8) bool {
    var d = std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true }) catch return false;
    d.close(io);
    return true;
}

/// The 400 text for the first edit this model cannot honour, or null.
fn refusalMessage(arena: std.mem.Allocator, fit: model_settings.Fit, existing: ?std.json.ObjectMap, edits: []const Edit) ?[]const u8 {
    var why_buf: [128]u8 = undefined;
    const r = model_settings.refuseEdits(fit, existing, edits, &why_buf) orelse return null;
    return std.fmt.allocPrint(arena, "\"{s}\" cannot be set for this model: {s}", .{ r.key, r.why }) catch "a setting this model cannot honour";
}

pub fn handlePost(allocator: std.mem.Allocator, stream: *http_conn.Conn, request_body: []const u8) !void {
    const sch = server.global_scheduler orelse {
        try server.sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
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
            try server.sendErrorResponse(allocator, stream, status, r.err_type, r.message, r.status);
            return;
        },
    };

    for (plan.server_edits) |e| if (std.mem.eql(u8, e.key, "models_dir")) if (e.value) |v| {
        if (!dirReadable(stream.io, v.string)) {
            try server.sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_setting", "models_dir does not exist or cannot be read", 400);
            return;
        }
    };
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = model_settings.defaultPath(&path_buf);
    var model_path: []const u8 = "";
    var model_loaded = false;
    if (plan.model_id) |id| {
        const entry = sch.registry.resolveEntry(id) catch {
            try server.sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "Unknown model id", 404);
            return;
        };
        model_path = entry.path;
        sch.registry.mutex.lockUncancelable(sch.io);
        model_loaded = entry.state == .ready;
        var row = heldRow(entry);
        sch.registry.mutex.unlock(sch.io);
        if (plan.model_edits.len > 0) {
            readPackFacts(stream.io, allocator, &row);
            var held = model_settings.load(allocator, stream.io, path);
            defer held.deinit();
            if (refusalMessage(arena, row.fit, model_settings.modelObject(&held, model_path), plan.model_edits)) |message| {
                try server.sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_setting", message, 400);
                return;
            }
        }
    }

    var message_buf: [160]u8 = undefined;
    const write = struct {
        fn run(a: std.mem.Allocator, io: std.Io, p: []const u8, section: []const u8, edits: []const Edit) !void {
            if (edits.len > 0) try model_settings.writeEdits(a, io, p, section, edits);
        }
    }.run;
    if (plan.namesEffort()) {
        const refused = applyEffort(arena, stream.io, path, plan.effort) catch |err| return writeFailed(allocator, stream, err);
        if (refused) |message| {
            try server.sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_setting", message, 400);
            return;
        }
    }
    write(allocator, stream.io, path, model_settings.server_section, plan.server_edits) catch |err| return writeFailed(allocator, stream, err);
    write(allocator, stream.io, path, model_path, plan.model_edits) catch |err| return writeFailed(allocator, stream, err);

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"ok\":true,\"applied\":{");
    var first = true;
    for (plan.server_edits) |e| {
        if (!first) try w.writeAll(",");
        first = false;
        try w.print("\"{s}\":", .{e.key});
        if (std.mem.eql(u8, e.key, "prefix_cache_disk_gb")) {
            try writeJson(w, applyCacheBudget(sch, e.value));
        } else if (std.mem.eql(u8, e.key, "update_check") or std.mem.eql(u8, e.key, "pack_update_check")) {
            try writeJson(w, applyUpdateCheck(&model_settings.launch_locks, e.key, e.value));
        } else if (std.mem.eql(u8, e.key, "skip_mem_preflight")) {
            try writeJson(w, applySkipMemPreflight(&model_settings.launch_locks, e.value));
        } else if (std.mem.eql(u8, e.key, "port")) {
            try writeJson(w, portMessage(&model_settings.launch_locks));
        } else {
            try writeJson(w, applyModelsDir(sch, &message_buf));
        }
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
    try server.sendResponse(stream, "200 OK", "application/json", out.written());
}

fn writeFailed(allocator: std.mem.Allocator, stream: *http_conn.Conn, err: anyerror) !void {
    if (err == error.MalformedSettingsFile) {
        try server.sendErrorResponse(allocator, stream, "409 Conflict", "settings_file_malformed", "The settings file is not a JSON object; fix or remove it before saving", 409);
        return;
    }
    log.warn("[settings] write failed: {s}\n", .{@errorName(err)});
    try server.sendErrorResponse(allocator, stream, "500 Internal Server Error", "settings_write_failed", "Could not write the settings file", 500);
}

fn cacheBudgetMessage(running_tiers_resized: usize) []const u8 {
    return if (running_tiers_resized > 0) "applied to the running tiers" else "saved; applies at the next model load";
}

/// The new SSD tier size reaches the running tiers at once and every later load; the launch flag, when given, keeps the number it set.
fn applyCacheBudget(sch: *@import("scheduler.zig").Scheduler, value: ?std.json.Value) []const u8 {
    if (model_settings.launch_locks.prefix_cache_disk_gb.set) return "saved; the --prefix-cache-disk flag still sets the size until a start without it";
    const gb: ?u64 = if (value) |v| @intCast(v.integer) else null;
    server.setDiskAsk(if (gb) |g| g << 30 else disk_placement.DEFAULT_DISK_CAP, gb != null);
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

fn portMessage(locks: *const model_settings.LaunchLocks) []const u8 {
    return if (locks.port.set) "saved; the --port flag still sets the port until a start without it" else "saved; applies at the next start";
}

fn modelsDirMessage(buf: []u8, rescan: anyerror!u32) []const u8 {
    const added = rescan catch return "saved; the rescan failed, use /v1/models/rescan";
    return std.fmt.bufPrint(buf, "saved; models rescanned, {d} added", .{added}) catch "saved; models rescanned";
}

fn applyModelsDir(sch: *@import("scheduler.zig").Scheduler, buf: []u8) []const u8 {
    if (model_settings.launch_locks.models_dir.set) return "saved; the --model-dir flag still sets the directories until a start without it";
    return modelsDirMessage(buf, sch.registry.rescan());
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
        .{ .id = "a", .path = "/m/a", .state = "ready", .bytes = 5, .fit = .{ .arch = model_settings.prefetch_arch, .streams_experts = true, .has_vision = true }, .loaded = .{ .ctx_size = 4096, .kv_quant = null, .ssd_budget_gb = 60, .mtp_acceptance = .exact }, .tier = .{ .cap_bytes = 2 << 30, .used_bytes = 7 } },
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
    try testing.expectEqualStrings("exact", lw.get("mtp_acceptance").?.string);
    try testing.expectEqualStrings("off", a.get("settings").?.object.get("kv_quant").?.object.get("file").?.string);

    const b = root.get("models").?.array.items[1].object;
    try testing.expect(b.get("loaded_with").? == .null);
    for (b.get("keys").?.array.items) |k| try testing.expect(!std.mem.startsWith(u8, k.string, "ssd_"));
}

test "settings api: each model lists the limits its arch and pack put on its keys, and nothing for a key it does not constrain" {
    var file = try model_settings.parse(testing.allocator, "{}");
    defer file.deinit();
    const locks: model_settings.LaunchLocks = .{};
    const rows = [_]ModelRow{
        .{ .id = "latent", .path = "/m/a", .state = "unloaded", .bytes = 5, .fit = .{ .arch = "a", .streams_experts = false, .has_vision = false, .latent_kv = true, .serves_mtp = false, .context_cap = 131072 } },
        .{ .id = "dropped", .path = "/m/b", .state = "unloaded", .bytes = 5, .fit = .{ .arch = "b", .streams_experts = true, .has_vision = false, .head_streams = false } },
        .{ .id = "plain", .path = "/m/c", .state = "unloaded", .bytes = 5, .fit = .{ .arch = "c", .streams_experts = true, .has_vision = true } },
        .{ .id = "bare", .path = "/m/d", .state = "unloaded", .bytes = 5, .fit = .{ .arch = model_settings.prefetch_arch, .streams_experts = true, .has_vision = true, .template_preserves_thinking = false } },
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
    try testing.expect(latent.get("mtp_acceptance").?.object.get("disabled").?.bool);
    try testing.expect(latent.get("mtp").?.object.get("why").?.string.len > 0);
    try testing.expectEqual(@as(i64, 131072), latent.get("ctx_size").?.object.get("max").?.integer);

    const dropped = models[1].object.get("limits").?.object;
    try testing.expectEqualStrings("ssd_budget_gb", dropped.get("mtp").?.object.get("unset").?.array.items[0].string);
    try testing.expect(dropped.get("mtp_greedy_tail").?.object.get("requires").?.object.get("mtp").?.bool);
    try testing.expect(dropped.get("kv_quant") == null);

    const plain = models[2].object.get("limits").?.object;
    for ([_][]const u8{ "kv_quant", "mtp", "ctx_size", "ssd_budget_gb", "vision", "preserve_thinking", "think_penalty" }) |k| try testing.expect(plain.get(k) == null);
    try testing.expect(plain.get("ssd_prefetch").?.object.get("disabled").?.bool);
    try testing.expect(plain.get("mtp_acceptance").?.object.get("unset") == null);

    const bare = models[3].object.get("limits").?.object;
    try testing.expect(bare.get("kv_quant").?.object.get("disabled").?.bool);
    try testing.expect(bare.get("preserve_thinking").?.object.get("disabled").?.bool);
    try testing.expect(bare.get("ssd_prefetch") == null);
}

test "settings api: a save the model cannot honour is refused with its reason, a removal never" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const latent: model_settings.Fit = .{ .arch = "a", .streams_experts = false, .has_vision = false, .latent_kv = true, .serves_mtp = false };
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
    var file = try model_settings.parse(testing.allocator, "{}");
    defer file.deinit();
    const locks: model_settings.LaunchLocks = .{};
    const dsv41 = model_mod.ModelConfig{ .model_type = "deepseek_v41", .max_position_embeddings = 1_048_576, .serve_defaults = @import("dsv41/dsv41_config.zig").Config.serveDefaults(model_mod.ServeDefaults) };
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
    try testing.expectEqualStrings("exact", d.get("mtp_acceptance").?.string);
    try testing.expect(d.get("mtp").?.bool);
    try testing.expect(d.get("preserve_thinking").?.bool);
    try testing.expectEqual(@as(i64, 0), d.get("think_penalty").?.integer);
    try testing.expectEqual(@as(i64, 0), d.get("ssd_prefetch").?.integer);
    // Measured at load, so never guessed.
    try testing.expect(d.get("ctx_size") == null);
    try testing.expect(d.get("ssd_budget_gb") == null);
    try testing.expect(models[1].object.get("defaults").? == .null);
}

test "settings api: the resolved defaults ignore the file's values the config was stamped with" {
    var stamped = model_mod.ModelConfig{ .model_type = "qwen4_exp", .max_position_embeddings = 262_144 };
    model_load.applyModelSettings(&stamped, .{ .max_tokens = 999, .mtp = false, .kv_quant = kv_quant_mod.KVQuantConfig.affine(4), .vision = false, .ssd_prefetch = 40 });
    const d = resolvedDefaults(&stamped);
    try testing.expectEqual(@as(?u32, 0), d.max_tokens);
    try testing.expectEqual(@as(?bool, true), d.mtp);
    try testing.expectEqual(@as(u8, 8), d.kv_quant.?.bits);
    try testing.expectEqual(@as(?bool, true), d.vision);
    try testing.expectEqual(@as(?u8, 0), d.ssd_prefetch);
    // What it was stamped with is still what the model loaded with.
    try testing.expectEqual(@as(?u32, 999), loadedWith(&stamped).max_tokens);
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
    try testing.expectEqual(@as(usize, 3), ef.get("families").?.array.items.len);
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

test "settings api: an effort save writes only what differs from built-in, takes hold at once, and a reset clears it" {
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
