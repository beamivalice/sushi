//! Per-model settings (`~/.sushi/model-settings.json`): context size, KV
//! quant, MTP and `preserve_thinking` that follow the MODEL, applied at every load construction
//! site. Keyed by the model's absolute path.
//! The app edits the file; the server owns applying it. A malformed file is
//! logged and treated as empty: a settings typo must never stop a load.
const std = @import("std");
const kv_quant = @import("kv_quant.zig");
const log = @import("log.zig");
const default_port = @import("default_port.zig");
const mtp_acceptance = @import("mtp_acceptance.zig");
const effort = @import("effort.zig");

/// Where a load's value for one key came from. A request's own field outranks all of them.
pub const Source = enum { flag, model_settings, default };

pub fn Pick(comptime T: type) type {
    return struct { value: T, source: Source };
}

/// An explicit launch flag outranks the file, which outranks the default. `flag` null = not given.
pub fn pick(comptime T: type, flag: ?T, setting: ?T, default: T) Pick(T) {
    if (flag) |f| return .{ .value = f, .source = .flag };
    if (setting) |s| return .{ .value = s, .source = .model_settings };
    return .{ .value = default, .source = .default };
}

pub fn sourceLabel(source: Source, flag_name: []const u8) []const u8 {
    return switch (source) {
        .flag => flag_name,
        .model_settings => "model-settings.json",
        .default => "default",
    };
}

/// `--preserve-thinking on|off`; null = not given. Read per render, where a
/// request's `chat_template_kwargs.preserve_thinking` outranks it.
pub var preserve_thinking_flag: ?bool = null;

/// `--think-penalty <lambda>`; null = not given. A request's `think_penalty` outranks it.
pub var think_penalty_flag: ?f32 = null;
/// The largest think penalty any source may set, in logits.
pub const think_penalty_max: f32 = 20;

/// A launch value counts as a flag only when the operator passed it.
pub fn launchFlag(comptime T: type, value: T, explicit: bool) ?T {
    return if (explicit) value else null;
}

/// What a load picks for a key with no flag and no file value; the load sites and the settings API read these.
pub const default_mtp = true;
pub const default_mtp_acceptance: mtp_acceptance.Mode = .exact;
pub const default_mtp_greedy_tail = false;
pub const default_preserve_thinking = true;
pub const default_think_penalty: f32 = 0;

/// The manual context: `--ctx-size` > `ctx_size` > auto. 0 = auto, i.e. not given, on both sides.
pub fn contextPick(flag: u32, setting: u32) Pick(u32) {
    return pick(u32, if (flag > 0) flag else null, if (setting > 0) setting else null, 0);
}

/// Whether a load arms the MTP head, and whether requests then default to it on any arch.
pub const MtpChoice = struct {
    on: bool,
    source: Source,

    pub fn resolve(flag: ?bool, setting: ?bool, default: bool) MtpChoice {
        const p = pick(bool, flag, setting, default);
        return .{ .on = p.value, .source = p.source };
    }

    /// The engine default arms the head but leaves the request default to the arch
    /// (`server.defaultEnableMtp`); an operator's `on` forces it for MoE targets too.
    pub fn forced(self: MtpChoice) bool {
        return self.on and self.source != .default;
    }

    pub fn label(self: MtpChoice) []const u8 {
        return if (self.on) "on" else "off";
    }

    pub fn sourceName(self: MtpChoice) []const u8 {
        return sourceLabel(self.source, if (self.on) "--mtp" else "--no-mtp");
    }
};

/// `--vision` (true) / `--no-vision` (false); null = neither given.
pub var vision_flag: ?bool = null;

pub fn visionFlagFrom(vision: bool, no_vision: bool) error{VisionFlagConflict}!?bool {
    if (vision and no_vision) return error.VisionFlagConflict;
    return if (vision) true else if (no_vision) false else null;
}

/// Whether a load brings up the checkpoint's vision tower, when it has one.
pub const VisionChoice = struct {
    on: bool,
    source: Source,

    /// `default_on` is false for a streamed load, whose tower and encode reserve come out of the expert cache.
    pub fn resolve(flag: ?bool, setting: ?bool, default_on: bool) VisionChoice {
        const p = pick(bool, flag, setting, default_on);
        return .{ .on = p.value, .source = p.source };
    }

    pub fn sourceName(self: VisionChoice) []const u8 {
        return sourceLabel(self.source, if (self.on) "--vision" else "--no-vision");
    }
};

test "model_settings: writeEdits edits a section whose key the hand-edited file spelled with a trailing slash" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    var path_buf: [600]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/model-settings.json", .{dir});
    try tmp.dir.writeFile(io, .{ .sub_path = "model-settings.json", .data = "{\"/m/a/\":{\"ctx_size\":64,\"mtp\":false}}" });

    try writeEdits(t.allocator, io, path, "/m/a", &.{.{ .key = "ctx_size", .value = .{ .integer = 128 } }});
    {
        var s = load(t.allocator, io, path);
        defer s.deinit();
        try t.expectEqual(@as(?u32, 128), s.lookup("/m/a").ctx_size);
        try t.expectEqual(@as(?bool, false), s.lookup("/m/a").mtp);
        try t.expectEqual(@as(usize, 1), s.parsed.?.value.object.count());
    }
    try writeEdits(t.allocator, io, path, "/m/a", &.{ .{ .key = "ctx_size", .value = null }, .{ .key = "mtp", .value = null } });
    var s = load(t.allocator, io, path);
    defer s.deinit();
    try t.expectEqual(@as(usize, 0), s.parsed.?.value.object.count());
}

test "model_settings: writeEdits writes through a symlinked settings file and leaves the link in place" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    try tmp.dir.createDirPath(io, "real");
    try tmp.dir.writeFile(io, .{ .sub_path = "real/settings.json", .data = "{}" });
    var target_buf: [600]u8 = undefined;
    const target = try std.fmt.bufPrintSentinel(&target_buf, "{s}/real/settings.json", .{dir}, 0);
    var link_buf: [600]u8 = undefined;
    const link = try std.fmt.bufPrintSentinel(&link_buf, "{s}/model-settings.json", .{dir}, 0);
    try t.expectEqual(@as(c_int, 0), std.c.symlink(target, link));

    try writeEdits(t.allocator, io, link, "/m/a", &.{.{ .key = "ctx_size", .value = .{ .integer = 128 } }});
    var target_settings = load(t.allocator, io, target);
    defer target_settings.deinit();
    try t.expectEqual(@as(?u32, 128), target_settings.lookup("/m/a").ctx_size);
    var link_target: [600]u8 = undefined;
    try t.expectEqualStrings(target, link_target[0..try tmp.dir.readLink(io, "model-settings.json", &link_target)]);
}

/// Log token for a manual context; 0 is auto.
pub fn contextLabel(buf: []u8, ctx: u32) []const u8 {
    if (ctx == 0) return "auto";
    return std.fmt.bufPrint(buf, "{d}", .{ctx}) catch "auto";
}

/// The flag an acceptance mode would have been launched with.
pub fn acceptanceFlagName(mode: mtp_acceptance.Mode) []const u8 {
    return switch (mode) {
        .exact => "default",
        .typical => "--mtp-typical",
        .tokenv3 => "--mtp-tokenv3",
    };
}

/// The acceptance flag as typed, value included; null when neither was given.
pub fn acceptanceFlagText(buf: []u8, typical: ?[]const u8, tokenv3: ?[]const u8) ?[]const u8 {
    const flag = if (typical != null) "--mtp-typical" else if (tokenv3 != null) "--mtp-tokenv3" else return null;
    return std.fmt.bufPrint(buf, "{s} {s}", .{ flag, typical orelse tokenv3.? }) catch null;
}

pub const Override = struct {
    ctx_size: ?u32 = null,
    kv_quant: ?kv_quant.KVQuantConfig = null,
    mtp: ?bool = null,
    mtp_acceptance: ?mtp_acceptance.Mode = null,
    mtp_greedy_tail: ?bool = null,
    ssd_budget_gb: ?u32 = null,
    ssd_prefetch: ?u8 = null,
    preserve_thinking: ?bool = null,
    think_penalty: ?f32 = null,
    logit_bias_file: ?@import("logit_bias.zig").FilePath = null,
    vision: ?bool = null,
    /// The omitted-`max_tokens` budget. In a defaults row, 0 = the rest of the window.
    max_tokens: ?u32 = null,

    pub fn isEmpty(o: Override) bool {
        return o.ctx_size == null and o.kv_quant == null and o.mtp == null and
            o.mtp_acceptance == null and o.mtp_greedy_tail == null and o.ssd_budget_gb == null and o.ssd_prefetch == null and o.preserve_thinking == null and
            o.think_penalty == null and o.logit_bias_file == null and o.vision == null and o.max_tokens == null;
    }
};

pub const Settings = struct {
    parsed: ?std.json.Parsed(std.json.Value) = null,

    pub fn deinit(self: *Settings) void {
        if (self.parsed) |*p| p.deinit();
        self.parsed = null;
    }

    pub fn lookup(self: *const Settings, model_path: []const u8) Override {
        const p = self.parsed orelse return .{};
        const root = switch (p.value) {
            .object => |o| o,
            else => return .{},
        };
        const want = trimSlash(model_path);
        var it = root.iterator();
        while (it.next()) |kv| {
            if (!std.mem.eql(u8, trimSlash(kv.key_ptr.*), want)) continue;
            return fromValue(kv.value_ptr.*);
        }
        return .{};
    }
};

fn trimSlash(p: []const u8) []const u8 {
    var s = p;
    while (s.len > 1 and s[s.len - 1] == '/') s = s[0 .. s.len - 1];
    return s;
}

/// The editor's range for a key: the loader applies what `validate` accepts, no more.
fn accepts(key: []const u8, v: std.json.Value) bool {
    validate(findKey(&model_keys, key).?, v) catch return false;
    return true;
}

fn fromValue(v: std.json.Value) Override {
    const obj = switch (v) {
        .object => |o| o,
        else => return .{},
    };
    var o: Override = .{};
    if (obj.get("ctx_size")) |c| if (accepts("ctx_size", c)) {
        o.ctx_size = @intCast(c.integer);
    };
    if (obj.get("kv_quant")) |k| if (accepts("kv_quant", k)) {
        o.kv_quant = kv_quant.KVQuantConfig.fromJsonValue(k);
    };
    if (obj.get("mtp")) |m| switch (m) {
        .bool => |b| o.mtp = b,
        else => {},
    };
    if (obj.get("mtp_acceptance")) |a| if (accepts("mtp_acceptance", a)) {
        o.mtp_acceptance = mtp_acceptance.fromName(a.string);
    };
    if (obj.get("mtp_greedy_tail")) |g| switch (g) {
        .bool => |b| o.mtp_greedy_tail = b,
        else => {},
    };
    if (obj.get("ssd_budget_gb")) |g| if (accepts("ssd_budget_gb", g)) {
        o.ssd_budget_gb = @intCast(g.integer);
    };
    if (obj.get("ssd_prefetch")) |g| if (accepts("ssd_prefetch", g)) {
        o.ssd_prefetch = @intCast(g.integer);
    };
    if (obj.get("preserve_thinking")) |p| switch (p) {
        .bool => |b| o.preserve_thinking = b,
        else => {},
    };
    if (obj.get("vision")) |p| switch (p) {
        .bool => |b| o.vision = b,
        else => {},
    };
    if (obj.get("max_tokens")) |g| if (accepts("max_tokens", g)) {
        o.max_tokens = @intCast(g.integer);
    };
    if (obj.get("logit_bias_file")) |path_value| switch (path_value) {
        .string => |path| o.logit_bias_file = @import("logit_bias.zig").FilePath.from(path),
        else => {},
    };
    if (obj.get("think_penalty")) |t| if (accepts("think_penalty", t)) {
        o.think_penalty = @floatCast(switch (t) {
            .float => |f| f,
            .integer => |i| @as(f64, @floatFromInt(i)),
            else => unreachable,
        });
    };
    return o;
}

pub fn parse(alloc: std.mem.Allocator, body: []const u8) !Settings {
    return .{ .parsed = try std.json.parseFromSlice(std.json.Value, alloc, body, .{}) };
}

/// Missing file = empty. Unreadable or malformed = empty, logged.
pub fn load(alloc: std.mem.Allocator, io: std.Io, path: []const u8) Settings {
    const body = std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 20)) catch |err| {
        if (err != error.FileNotFound) log.warn("[model-settings] {s}: unreadable ({s}), ignored\n", .{ path, @errorName(err) });
        return .{};
    };
    defer alloc.free(body);
    return parse(alloc, body) catch |err| {
        log.warn("[model-settings] {s}: malformed ({s}), ignored\n", .{ path, @errorName(err) });
        return .{};
    };
}

/// The port a server binds and a client targets without `--port`: the file's `server.port`, else the built-in one.
pub fn defaultPort(alloc: std.mem.Allocator, io: std.Io) u16 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var file = load(alloc, io, defaultPath(&buf));
    defer file.deinit();
    return server(&file).port orelse default_port.port;
}

pub fn defaultPath(buf: []u8) []const u8 {
    const home = std.mem.span(std.c.getenv("HOME") orelse "/tmp");
    return std.fmt.bufPrint(buf, "{s}/.sushi/model-settings.json", .{home}) catch "";
}

/// The one call load sites make: read the default file, look the model up, log a hit.
pub fn overrideFor(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8) Override {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var s = load(alloc, io, defaultPath(&buf));
    defer s.deinit();
    const o = s.lookup(model_path);
    if (!o.isEmpty()) log.info("[model-settings] {s}: ctx={d} kv={s} mtp={s} accept={s} greedy_tail={s} ssd_budget_gb={d} ssd_prefetch={?d} preserve_thinking={s} think_penalty={d} vision={s} max_tokens={d}\n", .{
        model_path,
        o.ctx_size orelse 0,
        if (o.kv_quant) |k| k.wireName() else "default",
        if (o.mtp) |m| (if (m) "on" else "off") else "default",
        if (o.mtp_acceptance) |a| mtp_acceptance.name(a) else "default",
        if (o.mtp_greedy_tail) |g| (if (g) "on" else "off") else "default",
        o.ssd_budget_gb orelse 0,
        o.ssd_prefetch,
        if (o.preserve_thinking) |p| (if (p) "on" else "off") else "default",
        o.think_penalty orelse 0,
        if (o.vision) |v| (if (v) "on" else "off") else "default",
        o.max_tokens orelse 0,
    });
    return o;
}

// ── Editing ──

/// One editable per-model key. Every key is stamped at load, so a change reaches a loaded model at its next load.
pub const KeySpec = struct {
    name: []const u8,
    kind: enum { int, bool, number, choice },
    min: f64 = 0,
    max: f64 = 0,
    choices: []const []const u8 = &.{},
    /// Maps every spelling the loader reads to its choice name; null = not a value.
    normalize: ?*const fn (std.json.Value) ?[]const u8 = null,
};

fn kvQuantChoice(v: std.json.Value) ?[]const u8 {
    return (kv_quant.KVQuantConfig.fromJsonValue(v) orelse return null).wireName();
}

pub const model_keys = [_]KeySpec{
    .{ .name = "ctx_size", .kind = .int, .min = 1, .max = std.math.maxInt(u32) },
    .{ .name = "kv_quant", .kind = .choice, .choices = &.{ "off", "4", "6", "8" }, .normalize = kvQuantChoice },
    .{ .name = "mtp", .kind = .bool },
    .{ .name = "mtp_acceptance", .kind = .choice, .choices = &.{ "exact", "typical", "tokenv3" } },
    .{ .name = "mtp_greedy_tail", .kind = .bool },
    .{ .name = "ssd_budget_gb", .kind = .int, .min = 1, .max = 512 },
    .{ .name = "ssd_prefetch", .kind = .int, .min = 0, .max = 100 },
    .{ .name = "preserve_thinking", .kind = .bool },
    .{ .name = "think_penalty", .kind = .number, .min = 0, .max = think_penalty_max },
    .{ .name = "vision", .kind = .bool },
    .{ .name = "max_tokens", .kind = .int, .min = 1, .max = std.math.maxInt(u32) },
};

/// Server-wide keys, under the file's `"server"` object (a model's key is its absolute path, so the two never meet).
pub const server_keys = [_]KeySpec{
    .{ .name = "models_dir", .kind = .choice },
    .{ .name = "prefix_cache_disk_gb", .kind = .int, .min = 0, .max = 1 << 20 },
    .{ .name = "update_check", .kind = .bool },
    .{ .name = "pack_update_check", .kind = .bool },
    .{ .name = "skip_mem_preflight", .kind = .bool },
    .{ .name = "port", .kind = .int, .min = 1, .max = 65535 },
};

pub const server_section = "server";

pub const EditError = error{ UnknownSetting, InvalidSetting };

pub fn findKey(specs: []const KeySpec, name: []const u8) ?KeySpec {
    for (specs) |k| if (std.mem.eql(u8, k.name, name)) return k;
    return null;
}

/// What decides which keys and values a model honours; the loader's own gates read the same answers. The fields past
/// `has_vision` default to "not ruled out", so a caller that cannot read one never hides a key.
pub const Fit = struct {
    arch: []const u8,
    streams_experts: bool,
    has_vision: bool,
    /// The arch runs a speculative head at all.
    serves_mtp: bool = true,
    /// The pack ships that head.
    mtp_head: bool = true,
    /// The pack cannot load without streaming its experts.
    streaming_required: bool = false,
    /// The head survives a streamed load.
    head_streams: bool = true,
    /// The pack's experts are STQ, the only stream with an expert prefetcher.
    stq_experts: bool = true,
    /// The cache stores only what the MLA latent holds (`KVQuantConfig.glmLatentBits`).
    latent_kv: bool = false,
    /// The model's context window; 0 = unknown.
    context_cap: u32 = 0,
    template_preserves_thinking: bool = true,
    template_thinks: bool = true,
};

/// The arch with an expert prefetcher and no KV cache or MTP tail: its drafts accept by the exact rule only.
pub const prefetch_arch = "deepseek_v41";

/// Why the loader ignores `key` for this model, as a sentence for the person editing; null when it takes effect.
pub fn unavailable(key: []const u8, fit: Fit) ?[]const u8 {
    const eql = std.mem.eql;
    const special = eql(u8, fit.arch, prefetch_arch);
    if (eql(u8, key, "ssd_budget_gb")) return if (fit.streams_experts) null else "This model does not stream its experts from SSD.";
    if (eql(u8, key, "ssd_prefetch")) {
        if (!special) return "This model has no expert prefetcher.";
        return if (fit.stq_experts) null else "This pack has no STQ experts, so there is no expert prefetcher.";
    }
    if (eql(u8, key, "vision")) return if (fit.has_vision) null else "This checkpoint has no vision tower.";
    if (eql(u8, key, "kv_quant")) return if (special) "This model keeps no KV cache to quantize." else null;
    if (eql(u8, key, "mtp")) {
        if (!fit.serves_mtp) return "This model serves without a speculative head.";
        if (!fit.mtp_head) return "This pack ships no MTP head.";
        if (fit.streaming_required and !fit.head_streams) return "This pack streams its experts, and its MTP head cannot run while they stream.";
        return null;
    }
    if (eql(u8, key, "mtp_acceptance") or eql(u8, key, "mtp_greedy_tail")) {
        if (special) return "Drafts are accepted by the exact rule only, with no MTP tail.";
        return unavailable("mtp", fit);
    }
    if (eql(u8, key, "preserve_thinking")) return if (fit.template_preserves_thinking) null else "This model's chat template has no preserve-thinking switch.";
    if (eql(u8, key, "think_penalty")) return if (fit.template_thinks) null else "This model's chat template has no thinking span to penalise.";
    return null;
}

pub fn applies(key: []const u8, fit: Fit) bool {
    return unavailable(key, fit) == null;
}

/// A pack that does not require streaming is started streaming by `ssd_budget_gb`, which drops a head that cannot stream.
fn budgetDropsHead(fit: Fit) bool {
    return fit.streams_experts and !fit.streaming_required and !fit.head_streams;
}

/// Whether the model's cache stores `choice` of `key`.
pub fn choiceAllowed(key: []const u8, choice: []const u8, fit: Fit) bool {
    if (!fit.latent_kv or !std.mem.eql(u8, key, "kv_quant")) return true;
    const stored = kv_quant.KVQuantConfig.fromJsonValue(.{ .string = choice }) orelse return false;
    return stored.glmLatentBits() != null;
}

pub const Limit = struct {
    /// The key does nothing for this model.
    disabled: bool = false,
    /// The largest value the model takes; 0 = no bound.
    max: u32 = 0,
    /// Takes effect only while `mtp` is not false (unset counts as on).
    requires_mtp: bool = false,
    /// Takes effect only while none of these keys is set.
    unset: []const []const u8 = &.{},
    why: []const u8 = "",

    pub fn isEmpty(self: Limit) bool {
        return !self.disabled and self.max == 0 and !self.requires_mtp and self.unset.len == 0;
    }
};

const unset_budget = [_][]const u8{"ssd_budget_gb"};

/// What the settings page may offer for `key` on this model, besides the choices `choiceAllowed` keeps.
pub fn limit(key: []const u8, fit: Fit) Limit {
    const eql = std.mem.eql;
    if (unavailable(key, fit)) |why| return .{ .disabled = true, .why = why };
    var l: Limit = .{};
    const tail = eql(u8, key, "mtp_acceptance") or eql(u8, key, "mtp_greedy_tail");
    if ((tail or eql(u8, key, "mtp")) and budgetDropsHead(fit)) {
        l.unset = &unset_budget;
        l.why = "An SSD budget makes this model stream its experts, and its MTP head cannot run while they stream.";
    }
    if (tail) {
        l.requires_mtp = true;
        if (l.why.len == 0) l.why = "Used only while MTP is on.";
    }
    if (eql(u8, key, "ctx_size") or eql(u8, key, "max_tokens")) l.max = fit.context_cap;
    return l;
}

pub fn maxWhy(buf: []u8, max: u32) []const u8 {
    return std.fmt.bufPrint(buf, "This model's context window is {d} tokens.", .{max}) catch "Above this model's context window.";
}

pub const Refusal = struct { key: []const u8, why: []const u8 };

/// The first of `edits` the model cannot honour, judged against the settings the file would hold after them. Only what an
/// edit sets is judged: a removal is always allowed, and a value already in the file stays.
pub fn refuseEdits(fit: Fit, existing: ?std.json.ObjectMap, edits: []const Edit, buf: []u8) ?Refusal {
    for (edits) |e| {
        const v = e.value orelse continue;
        const lim = limit(e.key, fit);
        if (lim.disabled) return .{ .key = e.key, .why = lim.why };
        if (v == .string and !choiceAllowed(e.key, v.string, fit)) return .{ .key = e.key, .why = "This model's cache cannot store that width." };
        if (lim.max > 0 and v == .integer and v.integer > lim.max) return .{ .key = e.key, .why = maxWhy(buf, lim.max) };
        if (lim.requires_mtp) if (stateOf(existing, edits, "mtp")) |m| if (m == .bool and !m.bool) return .{ .key = e.key, .why = "Used only while MTP is on." };
        for (lim.unset) |other| if (stateOf(existing, edits, other) != null) return .{ .key = e.key, .why = lim.why };
    }
    return null;
}

/// A key's value once `edits` land on `existing`; null = unset.
fn stateOf(existing: ?std.json.ObjectMap, edits: []const Edit, key: []const u8) ?std.json.Value {
    for (edits) |e| if (std.mem.eql(u8, e.key, key)) return e.value;
    if (existing) |o| if (o.get(key)) |v| return if (v == .null) null else v;
    return null;
}

/// The one stored spelling of a value, for a key that has several (`kv_quant`: 0, 16, "16" and "off" are one choice).
pub fn canonicalValue(spec: KeySpec, v: std.json.Value) std.json.Value {
    if (spec.normalize) |norm| if (norm(v)) |text| return .{ .string = text };
    return v;
}

/// Is `v` a value the key takes? A `choice` with no list (models_dir) takes an absolute path.
pub fn validate(spec: KeySpec, v: std.json.Value) EditError!void {
    switch (spec.kind) {
        .bool => if (v != .bool) return error.InvalidSetting,
        .int => switch (v) {
            .integer => |i| if (@as(f64, @floatFromInt(i)) < spec.min or @as(f64, @floatFromInt(i)) > spec.max) return error.InvalidSetting,
            else => return error.InvalidSetting,
        },
        .number => {
            const x: f64 = switch (v) {
                .integer => |i| @floatFromInt(i),
                .float => |f| f,
                else => return error.InvalidSetting,
            };
            if (!(x >= spec.min and x <= spec.max)) return error.InvalidSetting;
        },
        .choice => {
            const text: []const u8 = if (spec.normalize) |norm| (norm(v) orelse return error.InvalidSetting) else switch (v) {
                .string => |t| t,
                else => return error.InvalidSetting,
            };
            if (spec.choices.len == 0) {
                if (text.len == 0 or text.len > std.fs.max_path_bytes or !std.fs.path.isAbsolute(text)) return error.InvalidSetting;
                return;
            }
            for (spec.choices) |c| if (std.mem.eql(u8, c, text)) return;
            return error.InvalidSetting;
        },
    }
}

pub const Edit = struct {
    key: []const u8,
    /// Null removes the key.
    value: ?std.json.Value,
};

var write_mu: std.Io.Mutex = .init;

/// Held by an edit that reads the file, merges into it and writes it (`writeSectionsLocked`), so no other write lands between.
pub fn lockWrites(io: std.Io) void {
    write_mu.lockUncancelable(io);
}

pub fn unlockWrites(io: std.Io) void {
    write_mu.unlock(io);
}

pub const SectionEdits = struct { section: []const u8, edits: []const Edit };

/// Applies `edits` to the object keyed `section` (a model path, or `server_section`) and rewrites the file through a
/// temp file and a rename. Keys and sections it does not touch are kept as they are; an emptied section goes.
pub fn writeEdits(alloc: std.mem.Allocator, io: std.Io, link_path: []const u8, section: []const u8, edits: []const Edit) !void {
    lockWrites(io);
    defer unlockWrites(io);
    return writeSectionsLocked(alloc, io, link_path, &.{.{ .section = section, .edits = edits }});
}

/// `writeEdits` over several sections in one rewrite; the caller holds `lockWrites`.
pub fn writeSectionsLocked(alloc: std.mem.Allocator, io: std.Io, link_path: []const u8, sections: []const SectionEdits) !void {
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // A rename onto a symlink replaces the link, so the write goes to the file it points at.
    const path: []const u8 = std.Io.Dir.cwd().realPathFileAlloc(io, link_path, arena) catch link_path;

    const existing = std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(1 << 20)) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    var root: std.json.ObjectMap = .empty;
    if (existing) |body| {
        const tree = std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{}) catch return error.MalformedSettingsFile;
        if (tree != .object) return error.MalformedSettingsFile;
        root = tree.object;
    }
    for (sections) |one| {
        // The loader matches section keys without trailing slashes; the editor must edit the key it reads.
        const key = for (root.keys()) |k| {
            if (std.mem.eql(u8, trimSlash(k), trimSlash(one.section))) break k;
        } else one.section;
        var sec: std.json.ObjectMap = if (root.get(key)) |v| (if (v == .object) v.object else .empty) else .empty;
        for (one.edits) |e| {
            if (e.value) |v| {
                try sec.put(arena, try arena.dupe(u8, e.key), v);
            } else {
                _ = sec.orderedRemove(e.key);
            }
        }
        if (sec.count() == 0) {
            _ = root.orderedRemove(key);
        } else {
            try root.put(arena, key, .{ .object = sec });
        }
    }
    const out = try std.json.Stringify.valueAlloc(arena, std.json.Value{ .object = root }, .{ .whitespace = .indent_2 });

    if (std.fs.path.dirname(path)) |dir| try std.Io.Dir.cwd().createDirPath(io, dir);
    const tmp = try std.fmt.allocPrint(arena, "{s}.tmp{d}", .{ path, std.c.getpid() });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = tmp, .data = out });
    errdefer std.Io.Dir.cwd().deleteFile(io, tmp) catch {};
    try std.Io.Dir.renameAbsolute(tmp, path, io);
}

pub const Server = struct {
    models_dir: ?[]const u8 = null,
    prefix_cache_disk_gb: ?u32 = null,
    /// The automatic sushi release check; null = the file says nothing.
    update_check: ?bool = null,
    /// The automatic pack revision check.
    pack_update_check: ?bool = null,
    /// Load a model without the free-memory check; null = the file says nothing (check on).
    skip_mem_preflight: ?bool = null,
    /// The port a server binds and the clients target without `--port`.
    port: ?u16 = null,
};

/// An automatic network look is on unless `--no-update-check` or the file turns it off.
pub fn autoCheck(no_update_check: bool, file: ?bool) Pick(bool) {
    return pick(bool, if (no_update_check) false else null, file, true);
}

/// The load-time memory check runs unless `--skip-mem-preflight` or the file turns it off.
pub fn skipMemPreflight(flag: bool, file: ?bool) Pick(bool) {
    return pick(bool, if (flag) true else null, file, false);
}

/// The file's `"server"` section. `models_dir` borrows from `self`.
pub fn server(self: *const Settings) Server {
    const p = self.parsed orelse return .{};
    const root = switch (p.value) {
        .object => |o| o,
        else => return .{},
    };
    const sec = switch (root.get(server_section) orelse return .{}) {
        .object => |o| o,
        else => return .{},
    };
    var out: Server = .{};
    if (sec.get("models_dir")) |v| if (v == .string and std.fs.path.isAbsolute(v.string)) {
        out.models_dir = v.string;
    };
    if (sec.get("prefix_cache_disk_gb")) |v| if (v == .integer and v.integer >= 0 and v.integer <= server_keys[1].max) {
        out.prefix_cache_disk_gb = @intCast(v.integer);
    };
    if (sec.get("update_check")) |v| if (v == .bool) {
        out.update_check = v.bool;
    };
    if (sec.get("pack_update_check")) |v| if (v == .bool) {
        out.pack_update_check = v.bool;
    };
    if (sec.get("skip_mem_preflight")) |v| if (v == .bool) {
        out.skip_mem_preflight = v.bool;
    };
    if (sec.get("port")) |v| out.port = default_port.fromValue(v);
    return out;
}

/// The raw per-model object (for the editor), or null.
pub fn modelObject(self: *const Settings, model_path: []const u8) ?std.json.ObjectMap {
    const p = self.parsed orelse return null;
    const root = switch (p.value) {
        .object => |o| o,
        else => return null,
    };
    const want = trimSlash(model_path);
    var it = root.iterator();
    while (it.next()) |kv| {
        if (!std.mem.eql(u8, trimSlash(kv.key_ptr.*), want)) continue;
        return switch (kv.value_ptr.*) {
            .object => |o| o,
            else => null,
        };
    }
    return null;
}

/// The family's effort table as the file holds it, checked as a load checks it: a bad field or a default left
/// unoffered is skipped, each with a log line when `log_problems`.
pub fn effortEdits(file: *const Settings, f: effort.Family, log_problems: bool) effort.Edits {
    const p = file.parsed orelse return .{};
    if (p.value != .object) return .{};
    var buf: [64]u8 = undefined;
    const v = p.value.object.get(effort.sectionName(&buf, f)) orelse return .{};
    const read = effort.parseSection(f, v, log_problems).edits;
    const kept = effort.servable(f, read);
    if (log_problems and !std.meta.eql(kept, read)) log.warn("[effort] model-settings.json effort:{s}: the default word must be offered; the edit that unoffered it is ignored\n", .{@tagName(f)});
    return effort.normalize(f, kept);
}

/// Publishes every family's table from the file: what a start serves until a settings save replaces it.
pub fn publishEffortTables(file: *const Settings) void {
    for (effort.families) |f| {
        const e = effortEdits(file, f, true);
        effort.publish(f, e);
        if (!std.meta.eql(e, effort.Edits{})) log.info("[effort] {s}: table from model-settings.json\n", .{@tagName(f)});
    }
}

/// Reads the default settings file and publishes its effort tables.
pub fn loadEffortTables(alloc: std.mem.Allocator, io: std.Io) void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var s = load(alloc, io, defaultPath(&buf));
    defer s.deinit();
    publishEffortTables(&s);
}

/// A launch flag that pins a key: set by `main.zig` once the arguments are parsed, read by the settings API.
pub const Lock = struct {
    set: bool = false,
    buf: [std.fs.max_path_bytes]u8 = undefined,
    len: u16 = 0,

    pub fn pin(self: *Lock, comptime fmt: []const u8, args: anytype) void {
        self.set = true;
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = @intCast(w.end);
    }

    pub fn text(self: *const Lock) []const u8 {
        return self.buf[0..self.len];
    }
};

/// A byte count in the launch flag's own size syntax (`512MB`), so it reads back as typed.
pub fn sizeText(buf: []u8, bytes: u64) []const u8 {
    const units = [_]struct { suffix: []const u8, size: u64 }{ .{ .suffix = "GB", .size = 1 << 30 }, .{ .suffix = "MB", .size = 1 << 20 }, .{ .suffix = "KB", .size = 1 << 10 } };
    if (bytes == 0) return "0";
    for (units) |u| if (bytes % u.size == 0) return std.fmt.bufPrint(buf, "{d}{s}", .{ bytes / u.size, u.suffix }) catch "";
    return std.fmt.bufPrint(buf, "{d}B", .{bytes}) catch "";
}

pub const LaunchLocks = struct {
    models: [model_keys.len]Lock = @splat(.{}),
    models_dir: Lock = .{},
    prefix_cache_disk_gb: Lock = .{},
    update_check: Lock = .{},
    pack_update_check: Lock = .{},
    skip_mem_preflight: Lock = .{},
    port: Lock = .{},

    pub fn model(self: *const LaunchLocks, name: []const u8) ?*const Lock {
        for (model_keys, 0..) |k, i| if (std.mem.eql(u8, k.name, name)) return &self.models[i];
        return null;
    }

    pub fn modelMut(self: *LaunchLocks, name: []const u8) *Lock {
        for (model_keys, 0..) |k, i| if (std.mem.eql(u8, k.name, name)) return &self.models[i];
        unreachable;
    }
};

pub var launch_locks: LaunchLocks = .{};

test "model_settings: the editor accepts exactly the values the loader applies" {
    const t = std.testing;
    const cases = [_]struct { key: []const u8, body: []const u8 }{
        .{ .key = "ssd_budget_gb", .body = "600" },
        .{ .key = "ssd_budget_gb", .body = "512" },
        .{ .key = "ssd_budget_gb", .body = "0" },
        .{ .key = "ssd_prefetch", .body = "101" },
        .{ .key = "ctx_size", .body = "0" },
        .{ .key = "think_penalty", .body = "21" },
        .{ .key = "think_penalty", .body = "2.5" },
        .{ .key = "kv_quant", .body = "\"16\"" },
        .{ .key = "kv_quant", .body = "\"0\"" },
        .{ .key = "kv_quant", .body = "16" },
        .{ .key = "kv_quant", .body = "0" },
        .{ .key = "kv_quant", .body = "\"5\"" },
        .{ .key = "mtp_acceptance", .body = "\"fast\"" },
        .{ .key = "mtp_acceptance", .body = "\"typical\"" },
        .{ .key = "mtp", .body = "1" },
    };
    for (cases) |c| {
        const doc = try std.fmt.allocPrint(t.allocator, "{{\"/m/a\":{{\"{s}\":{s}}}}}", .{ c.key, c.body });
        defer t.allocator.free(doc);
        var s = try parse(t.allocator, doc);
        defer s.deinit();
        const value = s.parsed.?.value.object.get("/m/a").?.object.get(c.key).?;
        const editor = if (validate(findKey(&model_keys, c.key).?, value)) true else |_| false;
        try t.expectEqual(!s.lookup("/m/a").isEmpty(), editor);
    }
}

test "model_settings: applies: a key shows only for a model the loader would honour it on" {
    const t = std.testing;
    const dsv: Fit = .{ .arch = prefetch_arch, .streams_experts = true, .has_vision = true };
    for ([_][]const u8{ "kv_quant", "mtp_acceptance", "mtp_greedy_tail" }) |k| try t.expect(!applies(k, dsv));
    for ([_][]const u8{ "ssd_budget_gb", "ssd_prefetch", "vision", "mtp", "ctx_size", "preserve_thinking", "think_penalty" }) |k| try t.expect(applies(k, dsv));

    const resident: Fit = .{ .arch = "qwen4_exp", .streams_experts = false, .has_vision = false };
    for ([_][]const u8{ "ssd_budget_gb", "ssd_prefetch", "vision" }) |k| try t.expect(!applies(k, resident));
    for ([_][]const u8{ "kv_quant", "mtp", "mtp_acceptance", "mtp_greedy_tail", "ctx_size" }) |k| try t.expect(applies(k, resident));

    const qwen_streamed: Fit = .{ .arch = "qwen4_exp", .streams_experts = true, .has_vision = true };
    try t.expect(applies("ssd_budget_gb", qwen_streamed));
    try t.expect(!applies("ssd_prefetch", qwen_streamed));
    for (model_keys) |k| _ = applies(k.name, dsv);
}

test "model_settings: kv_quant is stored as its choice name whatever spelling the request used" {
    const spec = findKey(&model_keys, "kv_quant").?;
    const t = std.testing;
    try t.expectEqualStrings("off", canonicalValue(spec, .{ .integer = 0 }).string);
    try t.expectEqualStrings("off", canonicalValue(spec, .{ .integer = 16 }).string);
    try t.expectEqualStrings("off", canonicalValue(spec, .{ .string = "16" }).string);
    try t.expectEqualStrings("8", canonicalValue(spec, .{ .integer = 8 }).string);
    try t.expectEqualStrings("4", canonicalValue(spec, .{ .string = "4" }).string);
    try t.expectEqual(@as(i64, 5), canonicalValue(findKey(&model_keys, "ssd_prefetch").?, .{ .integer = 5 }).integer);
}

test "model_settings: a launch lock keeps a long path whole" {
    var long: [100]u8 = undefined;
    @memset(&long, 'a');
    var l: Lock = .{};
    l.pin("{s}", .{@as([]const u8, &long)});
    try std.testing.expect(l.set);
    try std.testing.expectEqualStrings(&long, l.text());
}

test "model_settings: the acceptance lock text is the flag as typed, never a placeholder" {
    var b: [64]u8 = undefined;
    try std.testing.expectEqualStrings("--mtp-typical 0.3", acceptanceFlagText(&b, "0.3", null).?);
    try std.testing.expectEqualStrings("--mtp-tokenv3 0.5", acceptanceFlagText(&b, null, "0.5").?);
    try std.testing.expect(acceptanceFlagText(&b, null, null) == null);
    var l: Lock = .{};
    l.pin("{s}", .{acceptanceFlagText(&b, null, "0.5").?});
    try std.testing.expectEqualStrings("--mtp-tokenv3 0.5", l.text());
}

test "model_settings: sizeText reads back in the flag's own syntax, sub-GB sizes included" {
    var b: [32]u8 = undefined;
    try std.testing.expectEqualStrings("40GB", sizeText(&b, 40 << 30));
    try std.testing.expectEqualStrings("512MB", sizeText(&b, 512 << 20));
    try std.testing.expectEqualStrings("1500MB", sizeText(&b, 1500 << 20));
    try std.testing.expectEqualStrings("3KB", sizeText(&b, 3 << 10));
    try std.testing.expectEqualStrings("1000B", sizeText(&b, 1000));
    try std.testing.expectEqualStrings("0", sizeText(&b, 0));
}

test "model_settings: parse + lookup with and without trailing slash" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a/": {"ctx_size": 65536, "kv_quant": "8", "mtp": false}, "/m/b": {"kv_quant": 4}, "/m/c": {"kv_quant": "6"}}
    );
    defer s.deinit();
    const a = s.lookup("/m/a");
    try std.testing.expectEqual(@as(?u32, 65536), a.ctx_size);
    try std.testing.expectEqual(@as(u8, 8), a.kv_quant.?.bits);
    try std.testing.expectEqual(@as(?bool, false), a.mtp);
    const b = s.lookup("/m/b/");
    try std.testing.expectEqual(@as(?u32, null), b.ctx_size);
    try std.testing.expectEqual(@as(u8, 4), b.kv_quant.?.bits);
    try std.testing.expectEqual(@as(?bool, null), b.mtp);
    try std.testing.expectEqual(@as(u8, 6), s.lookup("/m/c").kv_quant.?.bits);
    try std.testing.expect(s.lookup("/m/d").isEmpty());
}

test "model_settings: mtp_acceptance names a mode at its default threshold" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"mtp_acceptance": "typical"}, "/m/b": {"mtp_acceptance": "tokenv3"}, "/m/c": {"mtp_acceptance": "exact"}, "/m/d": {"mtp_acceptance": "fast"}}
    );
    defer s.deinit();
    try std.testing.expectEqual(@as(f32, 0.2), s.lookup("/m/a").mtp_acceptance.?.typical.delta);
    try std.testing.expectEqual(@as(f32, 0.95), s.lookup("/m/b").mtp_acceptance.?.tokenv3);
    try std.testing.expect(s.lookup("/m/c").mtp_acceptance.? == .exact);
    try std.testing.expect(s.lookup("/m/d").isEmpty());
}

test "model_settings: the greedy tail (mtp_greedy_tail) is a bool, anything else is unset" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"mtp_greedy_tail": true}, "/m/b": {"mtp_greedy_tail": false}, "/m/c": {"mtp_greedy_tail": "on"}}
    );
    defer s.deinit();
    try std.testing.expectEqual(@as(?bool, true), s.lookup("/m/a").mtp_greedy_tail);
    try std.testing.expectEqual(@as(?bool, false), s.lookup("/m/b").mtp_greedy_tail);
    try std.testing.expect(!s.lookup("/m/b").isEmpty());
    try std.testing.expect(s.lookup("/m/c").isEmpty());
}

test "model_settings: bad values ignored, bad JSON = empty" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"ctx_size": 0, "kv_quant": "3", "mtp": "yes", "future": 1}}
    );
    defer s.deinit();
    try std.testing.expect(s.lookup("/m/a").isEmpty());
    try std.testing.expectError(error.SyntaxError, parse(std.testing.allocator, "{nope"));
    var empty = load(std.testing.allocator, std.testing.io, "/nonexistent/model-settings.json");
    defer empty.deinit();
    try std.testing.expect(empty.lookup("/m/a").isEmpty());
}

test "model_settings: ssd_budget_gb rides the same file as the other keys" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"ssd_budget_gb": 60}, "/m/b": {"ssd_budget_gb": 0}, "/m/c": {"ssd_budget_gb": "60"}, "/m/d": {"ctx_size": 4096}}
    );
    defer s.deinit();
    try std.testing.expectEqual(@as(?u32, 60), s.lookup("/m/a").ssd_budget_gb);
    try std.testing.expect(s.lookup("/m/b").isEmpty());
    try std.testing.expect(s.lookup("/m/c").isEmpty());
    try std.testing.expectEqual(@as(?u32, null), s.lookup("/m/d").ssd_budget_gb);
    try std.testing.expect(!s.lookup("/m/a").isEmpty());
}

test "model_settings: ssd_prefetch is 0..100, a zero is a setting" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"ssd_prefetch": 40}, "/m/b": {"ssd_prefetch": 0}, "/m/c": {"ssd_prefetch": 101}, "/m/d": {"ssd_prefetch": "40"}, "/m/e": {"ssd_prefetch": -1}}
    );
    defer s.deinit();
    try std.testing.expectEqual(@as(?u8, 40), s.lookup("/m/a").ssd_prefetch);
    try std.testing.expectEqual(@as(?u8, 0), s.lookup("/m/b").ssd_prefetch);
    try std.testing.expect(!s.lookup("/m/b").isEmpty());
    for ([_][]const u8{ "/m/c", "/m/d", "/m/e" }) |path| try std.testing.expect(s.lookup(path).isEmpty());
}

test "an explicit --mtp / --no-mtp outranks the per-model mtp, which outranks the default" {
    const t = std.testing;
    const no_mtp = MtpChoice.resolve(false, true, true);
    try t.expect(!no_mtp.on);
    try t.expectEqualStrings("--no-mtp", no_mtp.sourceName());
    try t.expect(!no_mtp.forced());
    const mtp = MtpChoice.resolve(true, false, true);
    try t.expect(mtp.on and mtp.forced());
    try t.expectEqualStrings("--mtp", mtp.sourceName());
    const file = MtpChoice.resolve(null, true, true);
    try t.expect(file.on and file.forced());
    try t.expectEqualStrings("model-settings.json", file.sourceName());
    const engine = MtpChoice.resolve(null, null, true);
    try t.expect(engine.on and !engine.forced());
    try t.expectEqualStrings("default", engine.sourceName());
    try t.expectEqualStrings("off", MtpChoice.resolve(null, false, true).label());
}

test "an explicit --ctx-size outranks the per-model ctx_size; 0 is auto on both sides" {
    const t = std.testing;
    const flag = contextPick(16384, 4096);
    try t.expectEqual(@as(u32, 16384), flag.value);
    try t.expectEqualStrings("--ctx-size", sourceLabel(flag.source, "--ctx-size"));
    const file = contextPick(0, 4096);
    try t.expectEqual(@as(u32, 4096), file.value);
    try t.expectEqual(Source.model_settings, file.source);
    const auto = contextPick(0, 0);
    try t.expectEqual(@as(u32, 0), auto.value);
    try t.expectEqual(Source.default, auto.source);
}

test "an explicit --mtp-typical / --mtp-tokenv3 outranks the per-model mtp_acceptance" {
    const t = std.testing;
    const Mode = mtp_acceptance.Mode;
    const typical: Mode = .{ .typical = .{ .delta = 0.3 } };
    const flag = pick(Mode, typical, .{ .tokenv3 = 0.95 }, .exact);
    try t.expectEqual(@as(f32, 0.3), flag.value.typical.delta);
    try t.expectEqualStrings("--mtp-typical", sourceLabel(flag.source, acceptanceFlagName(flag.value)));
    const file = pick(Mode, null, .{ .tokenv3 = 0.95 }, .exact);
    try t.expectEqual(@as(f32, 0.95), file.value.tokenv3);
    try t.expectEqual(Source.model_settings, file.source);
    const engine = pick(Mode, null, null, .exact);
    try t.expect(engine.value == .exact);
    try t.expectEqual(Source.default, engine.source);
}

test "model_settings: think_penalty is a non-negative number, anything else is unset" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"think_penalty": 1.5}, "/m/b": {"think_penalty": 2}, "/m/c": {"think_penalty": -1}, "/m/d": {"think_penalty": "1"}, "/m/e": {"think_penalty": 0}}
    );
    defer s.deinit();
    try std.testing.expectEqual(@as(?f32, 1.5), s.lookup("/m/a").think_penalty);
    try std.testing.expectEqual(@as(?f32, 2), s.lookup("/m/b").think_penalty);
    try std.testing.expect(s.lookup("/m/c").isEmpty());
    try std.testing.expect(s.lookup("/m/d").isEmpty());
    try std.testing.expectEqual(@as(?f32, 0), s.lookup("/m/e").think_penalty);
    try std.testing.expect(!s.lookup("/m/e").isEmpty());
}

test "model_settings: preserve_thinking is a bool, anything else is unset" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"preserve_thinking": false}, "/m/b": {"preserve_thinking": true}, "/m/c": {"preserve_thinking": "off"}}
    );
    defer s.deinit();
    try std.testing.expectEqual(@as(?bool, false), s.lookup("/m/a").preserve_thinking);
    try std.testing.expect(!s.lookup("/m/a").isEmpty());
    try std.testing.expectEqual(@as(?bool, true), s.lookup("/m/b").preserve_thinking);
    try std.testing.expect(s.lookup("/m/c").isEmpty());
}

test "logit bias CPU: file setting is a populated override" {
    var s = try parse(std.testing.allocator, "{\"/m/a\":{\"logit_bias_file\":\"bias.json\"}}");
    defer s.deinit();
    try std.testing.expect(!s.lookup("/m/a").isEmpty());
}

test "vision: a streamed load leaves the tower off unless --vision or the per-model vision asks; resident keeps it on" {
    const t = std.testing;
    try t.expect(VisionChoice.resolve(null, null, true).on);
    try t.expect(!VisionChoice.resolve(null, null, false).on);
    try t.expectEqualStrings("default", VisionChoice.resolve(null, null, false).sourceName());
    const asked = VisionChoice.resolve(true, null, false);
    try t.expect(asked.on);
    try t.expectEqualStrings("--vision", asked.sourceName());
    const from_file = VisionChoice.resolve(null, true, false);
    try t.expect(from_file.on);
    try t.expectEqualStrings("model-settings.json", from_file.sourceName());
    try t.expect(!VisionChoice.resolve(false, true, true).on);
    try t.expectEqualStrings("--no-vision", VisionChoice.resolve(false, true, true).sourceName());
    try t.expect(!VisionChoice.resolve(null, false, true).on);
}

test "vision: --vision and --no-vision together are refused; either alone is the flag" {
    const t = std.testing;
    try t.expectEqual(@as(?bool, null), try visionFlagFrom(false, false));
    try t.expectEqual(@as(?bool, true), try visionFlagFrom(true, false));
    try t.expectEqual(@as(?bool, false), try visionFlagFrom(false, true));
    try t.expectError(error.VisionFlagConflict, visionFlagFrom(true, true));
}

test "model_settings: vision is a bool, anything else is unset" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"vision": true}, "/m/b": {"vision": false}, "/m/c": {"vision": "on"}}
    );
    defer s.deinit();
    try std.testing.expectEqual(@as(?bool, true), s.lookup("/m/a").vision);
    try std.testing.expect(!s.lookup("/m/a").isEmpty());
    try std.testing.expectEqual(@as(?bool, false), s.lookup("/m/b").vision);
    try std.testing.expect(s.lookup("/m/c").isEmpty());
}

test "model_settings: validate takes the values the loader reads and names the rest" {
    const t = std.testing;
    const ok = [_]struct { k: []const u8, v: std.json.Value }{
        .{ .k = "ctx_size", .v = .{ .integer = 32768 } },
        .{ .k = "kv_quant", .v = .{ .string = "8" } },
        .{ .k = "kv_quant", .v = .{ .integer = 0 } },
        .{ .k = "mtp", .v = .{ .bool = false } },
        .{ .k = "mtp_acceptance", .v = .{ .string = "typical" } },
        .{ .k = "ssd_prefetch", .v = .{ .integer = 0 } },
        .{ .k = "think_penalty", .v = .{ .integer = 2 } },
        .{ .k = "think_penalty", .v = .{ .float = 0.5 } },
    };
    for (ok) |c| try validate(findKey(&model_keys, c.k).?, c.v);
    const bad = [_]struct { k: []const u8, v: std.json.Value }{
        .{ .k = "ctx_size", .v = .{ .integer = 0 } },
        .{ .k = "kv_quant", .v = .{ .string = "5" } },
        .{ .k = "mtp", .v = .{ .string = "on" } },
        .{ .k = "mtp_acceptance", .v = .{ .string = "fast" } },
        .{ .k = "ssd_budget_gb", .v = .{ .integer = 513 } },
        .{ .k = "ssd_prefetch", .v = .{ .integer = 101 } },
        .{ .k = "think_penalty", .v = .{ .float = 20.5 } },
        .{ .k = "vision", .v = .{ .integer = 1 } },
    };
    for (bad) |c| try t.expectError(error.InvalidSetting, validate(findKey(&model_keys, c.k).?, c.v));
    try t.expect(findKey(&model_keys, "logit_bias_file") == null);
    try validate(findKey(&server_keys, "models_dir").?, .{ .string = "/Users/x/models" });
    try t.expectError(error.InvalidSetting, validate(findKey(&server_keys, "models_dir").?, .{ .string = "models" }));
    try t.expectError(error.InvalidSetting, validate(findKey(&server_keys, "prefix_cache_disk_gb").?, .{ .integer = -1 }));
}

test "model_settings: the update-check keys are booleans, read from the file, and a launch flag outranks the file" {
    const t = std.testing;
    for ([_][]const u8{ "update_check", "pack_update_check" }) |name| {
        const spec = findKey(&server_keys, name).?;
        try validate(spec, .{ .bool = false });
        try t.expectError(error.InvalidSetting, validate(spec, .{ .string = "off" }));
        try t.expectError(error.InvalidSetting, validate(spec, .{ .integer = 0 }));
    }
    var s = try parse(t.allocator, "{\"server\":{\"update_check\":false,\"pack_update_check\":\"no\"}}");
    defer s.deinit();
    try t.expectEqual(@as(?bool, false), server(&s).update_check);
    try t.expectEqual(@as(?bool, null), server(&s).pack_update_check);

    try t.expect(autoCheck(false, null).value);
    try t.expectEqual(Source.default, autoCheck(false, null).source);
    try t.expect(!autoCheck(false, false).value);
    try t.expectEqual(Source.model_settings, autoCheck(false, false).source);
    try t.expect(autoCheck(false, true).value);
    try t.expect(!autoCheck(true, true).value);
    try t.expectEqual(Source.flag, autoCheck(true, true).source);
}

test "model_settings: server.skip_mem_preflight is a boolean read from the file, off when absent or mistyped" {
    const t = std.testing;
    const spec = findKey(&server_keys, "skip_mem_preflight").?;
    try validate(spec, .{ .bool = true });
    try t.expectError(error.InvalidSetting, validate(spec, .{ .string = "on" }));
    var on = try parse(t.allocator, "{\"server\":{\"skip_mem_preflight\":true}}");
    defer on.deinit();
    try t.expectEqual(@as(?bool, true), server(&on).skip_mem_preflight);
    var bad = try parse(t.allocator, "{\"server\":{\"skip_mem_preflight\":1}}");
    defer bad.deinit();
    try t.expectEqual(@as(?bool, null), server(&bad).skip_mem_preflight);
    try t.expect(!launch_locks.skip_mem_preflight.set);
}

test "model_settings: the memory check is skipped when the flag is given or the file says so, and the flag is never overridden" {
    const t = std.testing;
    try t.expect(!skipMemPreflight(false, null).value);
    try t.expectEqual(Source.default, skipMemPreflight(false, null).source);
    try t.expect(skipMemPreflight(false, true).value);
    try t.expectEqual(Source.model_settings, skipMemPreflight(false, true).source);
    try t.expect(!skipMemPreflight(false, false).value);
    try t.expect(skipMemPreflight(true, false).value);
    try t.expectEqual(Source.flag, skipMemPreflight(true, false).source);
}

test "model_settings: writeEdits keeps what it does not touch, round-trips through lookup, and removes an emptied section" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    var path_buf: [600]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/sub/model-settings.json", .{dir});

    try writeEdits(t.allocator, io, path, "/m/a", &.{
        .{ .key = "kv_quant", .value = .{ .string = "4" } },
        .{ .key = "mtp", .value = .{ .bool = false } },
    });
    try writeEdits(t.allocator, io, path, server_section, &.{
        .{ .key = "prefix_cache_disk_gb", .value = .{ .integer = 40 } },
        .{ .key = "models_dir", .value = .{ .string = "/models" } },
    });
    try writeEdits(t.allocator, io, path, "/m/a", &.{.{ .key = "ctx_size", .value = .{ .integer = 8192 } }});
    {
        var s = load(t.allocator, io, path);
        defer s.deinit();
        const o = s.lookup("/m/a");
        try t.expectEqual(@as(?u32, 8192), o.ctx_size);
        try t.expectEqual(@as(?bool, false), o.mtp);
        try t.expectEqualStrings("4", o.kv_quant.?.wireName());
        try t.expectEqual(@as(?u32, 40), server(&s).prefix_cache_disk_gb);
        try t.expectEqualStrings("/models", server(&s).models_dir.?);
    }
    try writeEdits(t.allocator, io, path, "/m/a", &.{
        .{ .key = "kv_quant", .value = null },
        .{ .key = "mtp", .value = null },
        .{ .key = "ctx_size", .value = null },
    });
    var s = load(t.allocator, io, path);
    defer s.deinit();
    try t.expect(s.lookup("/m/a").isEmpty());
    try t.expect(modelObject(&s, "/m/a") == null);
    try t.expectEqual(@as(?u32, 40), server(&s).prefix_cache_disk_gb);

    try tmp.dir.writeFile(io, .{ .sub_path = "sub/model-settings.json", .data = "[1,2" });
    try t.expectError(error.MalformedSettingsFile, writeEdits(t.allocator, io, path, "/m/a", &.{.{ .key = "mtp", .value = .{ .bool = true } }}));
}

test "model_settings: limits: each arch and pack variant is offered only what the loader honours" {
    const t = std.testing;
    const latent: Fit = .{ .arch = "a", .streams_experts = false, .has_vision = false, .latent_kv = true, .serves_mtp = false, .context_cap = 4096 };
    try t.expect(choiceAllowed("kv_quant", "off", latent));
    try t.expect(choiceAllowed("kv_quant", "6", latent));
    try t.expect(choiceAllowed("kv_quant", "8", latent));
    try t.expect(!choiceAllowed("kv_quant", "4", latent));
    try t.expect(choiceAllowed("kv_quant", "4", .{ .arch = "a", .streams_experts = false, .has_vision = false }));
    for ([_][]const u8{ "mtp", "mtp_acceptance", "mtp_greedy_tail" }) |k| {
        try t.expect(!applies(k, latent));
        try t.expect(limit(k, latent).disabled);
        try t.expect(limit(k, latent).why.len > 0);
    }
    try t.expectEqual(@as(u32, 4096), limit("ctx_size", latent).max);

    const headless: Fit = .{ .arch = "a", .streams_experts = false, .has_vision = false, .mtp_head = false };
    for ([_][]const u8{ "mtp", "mtp_acceptance", "mtp_greedy_tail" }) |k| try t.expect(!applies(k, headless));

    const dsv: Fit = .{ .arch = prefetch_arch, .streams_experts = true, .has_vision = true, .streaming_required = true, .stq_experts = false };
    try t.expect(!applies("ssd_prefetch", dsv));
    try t.expect(applies("mtp", dsv));
    try t.expect(applies("ssd_prefetch", .{ .arch = prefetch_arch, .streams_experts = true, .has_vision = true }));

    const dropped: Fit = .{ .arch = "a", .streams_experts = true, .has_vision = false, .head_streams = false };
    try t.expectEqualStrings("ssd_budget_gb", limit("mtp", dropped).unset[0]);
    try t.expectEqual(@as(usize, 1), limit("mtp_acceptance", dropped).unset.len);
    try t.expect(limit("mtp_acceptance", dropped).requires_mtp);
    try t.expect(!limit("mtp", dropped).requires_mtp);
    try t.expect(limit("mtp", .{ .arch = "a", .streams_experts = true, .has_vision = false }).isEmpty());

    var forced = dropped;
    forced.streaming_required = true;
    for ([_][]const u8{ "mtp", "mtp_acceptance", "mtp_greedy_tail" }) |k| try t.expect(!applies(k, forced));

    const plain: Fit = .{ .arch = "a", .streams_experts = false, .has_vision = false, .template_preserves_thinking = false, .template_thinks = false };
    try t.expect(!applies("preserve_thinking", plain));
    try t.expect(!applies("think_penalty", plain));
}

test "model_settings: refuseEdits: a value or pair the model cannot honour is refused, a removal or a kept file value never" {
    const t = std.testing;
    var buf: [128]u8 = undefined;
    const latent: Fit = .{ .arch = "a", .streams_experts = false, .has_vision = false, .latent_kv = true, .serves_mtp = false, .context_cap = 4096 };
    try t.expectEqualStrings("kv_quant", refuseEdits(latent, null, &.{.{ .key = "kv_quant", .value = .{ .string = "4" } }}, &buf).?.key);
    try t.expect(refuseEdits(latent, null, &.{.{ .key = "kv_quant", .value = .{ .string = "6" } }}, &buf) == null);
    try t.expectEqualStrings("mtp", refuseEdits(latent, null, &.{.{ .key = "mtp", .value = .{ .bool = true } }}, &buf).?.key);
    try t.expect(refuseEdits(latent, null, &.{ .{ .key = "mtp", .value = null }, .{ .key = "mtp_acceptance", .value = null } }, &buf) == null);
    const over = refuseEdits(latent, null, &.{.{ .key = "ctx_size", .value = .{ .integer = 4097 } }}, &buf).?;
    try t.expect(std.mem.indexOf(u8, over.why, "4096") != null);
    try t.expect(refuseEdits(latent, null, &.{.{ .key = "ctx_size", .value = .{ .integer = 4096 } }}, &buf) == null);

    const dropped: Fit = .{ .arch = "a", .streams_experts = true, .has_vision = false, .head_streams = false };
    var file = try parse(t.allocator, "{\"/m\":{\"mtp\":false,\"ssd_budget_gb\":40}}");
    defer file.deinit();
    const held = modelObject(&file, "/m");
    const accept = Edit{ .key = "mtp_acceptance", .value = .{ .string = "typical" } };
    try t.expectEqualStrings("mtp_acceptance", refuseEdits(dropped, held, &.{accept}, &buf).?.key);
    try t.expectEqualStrings("mtp_acceptance", refuseEdits(dropped, null, &.{ accept, .{ .key = "mtp", .value = .{ .bool = false } } }, &buf).?.key);
    try t.expect(refuseEdits(dropped, held, &.{.{ .key = "ssd_budget_gb", .value = .{ .integer = 50 } }}, &buf) == null);
    try t.expectEqualStrings("mtp", refuseEdits(dropped, held, &.{.{ .key = "mtp", .value = .{ .bool = true } }}, &buf).?.key);
    try t.expect(refuseEdits(dropped, held, &.{ .{ .key = "mtp", .value = .{ .bool = true } }, .{ .key = "ssd_budget_gb", .value = null } }, &buf) == null);
    const resident: Fit = .{ .arch = "a", .streams_experts = false, .has_vision = false };
    try t.expect(refuseEdits(resident, held, &.{accept}, &buf) != null);
    try t.expect(refuseEdits(resident, null, &.{accept}, &buf) == null);
}

test "model_settings: max_tokens is a positive integer the model's window bounds" {
    const t = std.testing;
    var s = try parse(t.allocator,
        \\{"/m/a": {"max_tokens": 4096}, "/m/b": {"max_tokens": 0}, "/m/c": {"max_tokens": -5}, "/m/d": {"max_tokens": "9"}, "/m/e": {"max_tokens": 1.5}}
    );
    defer s.deinit();
    try t.expectEqual(@as(?u32, 4096), s.lookup("/m/a").max_tokens);
    for ([_][]const u8{ "/m/b", "/m/c", "/m/d", "/m/e" }) |m| try t.expectEqual(@as(?u32, null), s.lookup(m).max_tokens);
    const fit: Fit = .{ .arch = "x", .streams_experts = false, .has_vision = false, .context_cap = 65536 };
    try t.expect(applies("max_tokens", fit));
    try t.expectEqual(@as(u32, 65536), limit("max_tokens", fit).max);
    var buf: [128]u8 = undefined;
    const over = refuseEdits(fit, null, &.{.{ .key = "max_tokens", .value = .{ .integer = 65537 } }}, &buf).?;
    try t.expectEqualStrings("max_tokens", over.key);
    try t.expect(refuseEdits(fit, null, &.{.{ .key = "max_tokens", .value = .{ .integer = 65536 } }}, &buf) == null);
    try t.expect((Override{ .max_tokens = 1 }).isEmpty() == false);
}

test "model_settings: server.port is a whole port from 1 to 65535, anything else is ignored" {
    const t = std.testing;
    for ([_]struct { body: []const u8, want: ?u16 }{
        .{ .body = "{\"server\":{\"port\":8080}}", .want = 8080 },
        .{ .body = "{\"server\":{\"port\":65535}}", .want = 65535 },
        .{ .body = "{\"server\":{\"port\":1}}", .want = 1 },
        .{ .body = "{\"server\":{\"port\":0}}", .want = null },
        .{ .body = "{\"server\":{\"port\":65536}}", .want = null },
        .{ .body = "{\"server\":{\"port\":\"8080\"}}", .want = null },
        .{ .body = "{\"server\":{\"port\":80.5}}", .want = null },
        .{ .body = "{\"server\":{}}", .want = null },
        .{ .body = "{}", .want = null },
    }) |c| {
        var s = try parse(t.allocator, c.body);
        defer s.deinit();
        try t.expectEqual(c.want, server(&s).port);
    }
    const spec = findKey(&server_keys, "port").?;
    try t.expectEqual(@as(f64, 1), spec.min);
    try t.expectEqual(@as(f64, 65535), spec.max);
}
