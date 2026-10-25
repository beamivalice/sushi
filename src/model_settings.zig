//! Per-model settings (`~/.sushi/model-settings.json`): context size, KV
//! quant, MTP and `preserve_thinking` that follow the MODEL, applied at every load construction
//! site. Keyed by the model's absolute path (dir, or the `.gguf` file).
//! The app edits the file; the server owns applying it. A malformed file is
//! logged and treated as empty: a settings typo must never stop a load.
const std = @import("std");
const kv_quant = @import("kv_quant.zig");
const log = @import("log.zig");
const mtp_acceptance = @import("mtp_acceptance.zig");

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

pub const Override = struct {
    ctx_size: ?u32 = null,
    kv_quant: ?kv_quant.KVQuantConfig = null,
    mtp: ?bool = null,
    mtp_acceptance: ?mtp_acceptance.Mode = null,
    mtp_greedy_tail: ?bool = null,
    ssd_budget_gb: ?u32 = null,
    preserve_thinking: ?bool = null,
    think_penalty: ?f32 = null,
    logit_bias_file: ?@import("logit_bias.zig").FilePath = null,
    vision: ?bool = null,

    pub fn isEmpty(o: Override) bool {
        return o.ctx_size == null and o.kv_quant == null and o.mtp == null and
            o.mtp_acceptance == null and o.mtp_greedy_tail == null and o.ssd_budget_gb == null and o.preserve_thinking == null and
            o.think_penalty == null and o.logit_bias_file == null and o.vision == null;
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

fn fromValue(v: std.json.Value) Override {
    const obj = switch (v) {
        .object => |o| o,
        else => return .{},
    };
    var o: Override = .{};
    if (obj.get("ctx_size")) |c| switch (c) {
        .integer => |i| if (i > 0 and i <= std.math.maxInt(u32)) {
            o.ctx_size = @intCast(i);
        },
        else => {},
    };
    if (obj.get("kv_quant")) |k| o.kv_quant = kv_quant.KVQuantConfig.fromJsonValue(k);
    if (obj.get("mtp")) |m| switch (m) {
        .bool => |b| o.mtp = b,
        else => {},
    };
    if (obj.get("mtp_acceptance")) |a| switch (a) {
        .string => |name| o.mtp_acceptance = mtp_acceptance.fromName(name),
        else => {},
    };
    if (obj.get("mtp_greedy_tail")) |g| switch (g) {
        .bool => |b| o.mtp_greedy_tail = b,
        else => {},
    };
    if (obj.get("ssd_budget_gb")) |g| switch (g) {
        .integer => |i| if (i > 0 and i <= std.math.maxInt(u32)) {
            o.ssd_budget_gb = @intCast(i);
        },
        else => {},
    };
    if (obj.get("preserve_thinking")) |p| switch (p) {
        .bool => |b| o.preserve_thinking = b,
        else => {},
    };
    if (obj.get("vision")) |p| switch (p) {
        .bool => |b| o.vision = b,
        else => {},
    };
    if (obj.get("logit_bias_file")) |path_value| switch (path_value) {
        .string => |path| o.logit_bias_file = @import("logit_bias.zig").FilePath.from(path),
        else => {},
    };
    const lambda: ?f64 = if (obj.get("think_penalty")) |t| switch (t) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => null,
    } else null;
    if (lambda) |l| if (l >= 0 and l <= think_penalty_max) {
        o.think_penalty = @floatCast(l);
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
    if (!o.isEmpty()) log.info("[model-settings] {s}: ctx={d} kv={s} mtp={s} accept={s} greedy_tail={s} ssd_budget_gb={d} preserve_thinking={s} think_penalty={d} vision={s}\n", .{
        model_path,
        o.ctx_size orelse 0,
        if (o.kv_quant) |k| k.wireName() else "default",
        if (o.mtp) |m| (if (m) "on" else "off") else "default",
        if (o.mtp_acceptance) |a| mtp_acceptance.name(a) else "default",
        if (o.mtp_greedy_tail) |g| (if (g) "on" else "off") else "default",
        o.ssd_budget_gb orelse 0,
        if (o.preserve_thinking) |p| (if (p) "on" else "off") else "default",
        o.think_penalty orelse 0,
        if (o.vision) |v| (if (v) "on" else "off") else "default",
    });
    return o;
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
