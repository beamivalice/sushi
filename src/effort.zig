const std = @import("std");
const log = @import("log.zig");
const testing = std.testing;

/// The engine's thinking-effort vocabulary. Each served arch accepts a subset
/// (`effortArms`); a word outside it is refused. `selectEffortPick` folds Qwen
/// `high`/`max` to `xhigh`, GLM `medium` to `high` and `xhigh` to `max`, and
/// MiMo's thinking words to `on` (that fold is silent).
pub const Effort = enum { off, on, minimal, low, medium, high, xhigh, max };

/// Explicit --think default for serving, CLI and later model loads.
pub var think_effort_flag: ?Effort = null;

/// The arm `--think` selects on this arch from its served table; null leaves the model on its own default.
pub fn thinkFlagArm(model_type: []const u8) ?EffortArm {
    const snap = snapshot();
    return thinkFlagArmIn(&snap, model_type);
}

pub fn thinkFlagArmIn(snap: *const Snapshot, model_type: []const u8) ?EffortArm {
    const e = think_effort_flag orelse return null;
    const arms = servedArmsIn(snap, model_type) orelse return null;
    return armForWord(arms.slice(), @tagName(e));
}

/// One accepted effort word on one arch. `budget` is the decode-time thinking
/// cap in tokens; null = none (a request's own budget still applies). `sends`
/// is what the template's `reasoning_effort` receives (`templateValue`). A fold arm is served as `fold_to`'s arm, with a
/// notice, and is not listed (`isListed`).
pub const EffortArm = struct { effort: Effort, budget: ?i32 = null, sends: Sends = .none, fold_to: ?Effort = null };

pub fn isListed(a: EffortArm) bool {
    return a.fold_to == null;
}

/// What a row puts in the template's `reasoning_effort`: nothing (off), one of the family's words, or DeepSeek's 1-100 scale.
pub const Sends = union(enum) { none, word: Effort, scale: u8 };

/// A served family with an editable effort table (model-settings.json `effort:<family>`). MiMo has only on/off.
pub const Family = enum {
    qwen4_exp,
    deepseek_v41,
    glm5_next,

    pub fn of(model_type: []const u8) ?Family {
        return std.meta.stringToEnum(Family, model_type);
    }

    pub fn spec(f: Family) *const Spec {
        return &specs[@intFromEnum(f)];
    }

    /// Its template has a thinking-off switch.
    pub fn offOfferable(f: Family) bool {
        return f.spec().off_offerable;
    }

    /// The words its template takes for `reasoning_effort`; empty for a 1-100 scale.
    pub fn sendsWords(f: Family) []const Effort {
        return f.spec().words;
    }

    pub fn takesScale(f: Family) bool {
        return f.spec().words.len == 0;
    }
};

/// A scale value the template takes as a word of its own (`chat.dsv41EffortFor`).
pub const Preset = struct { word: Effort, value: u8 };

/// One family's facts, in `Family` order: what its template takes and its built-in table.
pub const Spec = struct {
    label: []const u8,
    words: []const Effort,
    presets: []const Preset = &.{},
    off_offerable: bool = true,
    table: Table,
    sends_why: []const u8,
    off_why: []const u8,
};

fn makeSpec(comptime label: []const u8, comptime words: []const Effort, comptime presets: []const Preset, comptime off_offerable: bool, comptime t: Table) Spec {
    comptime var list: []const u8 = "";
    inline for (words, 0..) |w, i| list = list ++ (if (i == 0) "" else if (i + 1 == words.len) " or " else ", ") ++ @tagName(w);
    return .{
        .label = label,
        .words = words,
        .presets = presets,
        .off_offerable = off_offerable,
        .table = t,
        .sends_why = if (words.len == 0) label ++ " takes a whole number from 1 to 100" else label ++ " takes " ++ list,
        .off_why = label ++ "'s template has no thinking-off switch",
    };
}

pub const families = std.enums.values(Family);

/// A table's rows, in the order the settings page shows them.
pub const table_words = [_]Effort{ .off, .low, .medium, .high, .xhigh, .max };

pub fn rowIndex(e: Effort) ?usize {
    return std.mem.indexOfScalar(Effort, &table_words, e);
}

/// `offered`: served as itself. A row that is not offered is refused, unless it is a built-in fold (`fold_to`), which a
/// file's `offered`, true or false, replaces.
pub const Row = struct { offered: bool, sends: Sends = .none, cap: ?i32 = null, fold_to: ?Effort = null };

pub const Table = struct {
    rows: [table_words.len]Row,
    default: Effort,

    pub fn row(self: Table, e: Effort) Row {
        return self.rows[rowIndex(e).?];
    }

    /// The fold target a not-offered row is served at now, or null when it is refused.
    pub fn foldsTo(self: Table, e: Effort) ?Effort {
        const r = self.row(e);
        if (r.offered) return null;
        const to = r.fold_to orelse return null;
        return if (self.row(to).offered) to else null;
    }

    /// Every word a request may name: the offered rows, then fold arms for the folds whose target is offered.
    pub fn offeredArms(self: Table) Arms {
        var out: Arms = .{};
        for (self.rows, table_words) |r, w| if (r.offered) {
            out.buf[out.len] = .{ .effort = w, .budget = r.cap, .sends = r.sends };
            out.len += 1;
        } else if (self.foldsTo(w)) |to| {
            out.buf[out.len] = .{ .effort = w, .fold_to = to };
            out.len += 1;
        };
        return out;
    }
};

/// A served table's offered arms, by value.
pub const Arms = struct {
    buf: [std.enums.values(Effort).len]EffortArm = undefined,
    len: usize = 0,

    pub fn slice(self: *const Arms) []const EffortArm {
        return self.buf[0..self.len];
    }
};

/// The engine's own tables, in `Family` order. Qwen high/max and GLM medium/xhigh are folds: served at a near word, with
/// a notice. The 1-100 family's own words are low 50, high 75 and max 100.
const specs = [_]Spec{
    makeSpec("Qwen", &.{ .low, .medium, .xhigh }, &.{}, true, .{ .default = .medium, .rows = .{
        .{ .offered = true },
        .{ .offered = true, .sends = .{ .word = .low }, .cap = 2048 },
        .{ .offered = true, .sends = .{ .word = .medium }, .cap = 8192 },
        .{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh },
        .{ .offered = true, .sends = .{ .word = .xhigh } },
        .{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh },
    } }),
    makeSpec("DeepSeek", &.{}, &.{ .{ .word = .low, .value = 50 }, .{ .word = .high, .value = 75 }, .{ .word = .max, .value = 100 } }, true, .{ .default = .high, .rows = .{
        .{ .offered = true },
        .{ .offered = true, .sends = .{ .scale = 50 } },
        .{ .offered = true, .sends = .{ .scale = 62 } },
        .{ .offered = true, .sends = .{ .scale = 75 } },
        .{ .offered = true, .sends = .{ .scale = 87 } },
        .{ .offered = true, .sends = .{ .scale = 100 } },
    } }),
    makeSpec("GLM", &.{ .low, .high, .max }, &.{}, false, .{ .default = .high, .rows = .{
        .{ .offered = false },
        .{ .offered = true, .sends = .{ .word = .low } },
        .{ .offered = false, .sends = .{ .word = .high }, .fold_to = .high },
        .{ .offered = true, .sends = .{ .word = .high } },
        .{ .offered = false, .sends = .{ .word = .max }, .fold_to = .max },
        .{ .offered = true, .sends = .{ .word = .max } },
    } }),
};

pub fn builtin(f: Family) Table {
    return f.spec().table;
}

const builtin_arms = blk: {
    var out: [families.len]Arms = undefined;
    for (families, &out) |f, *a| a.* = builtin(f).offeredArms();
    break :blk out;
};

// MiMo's template has only on/off; do not advertise artificial effort levels.
const mimo_v2_efforts = [_]EffortArm{
    .{ .effort = .off },
    .{ .effort = .on },
};

/// The built-in offered arms; null = an inherited arch: its effort words keep `responses.effortBudget`.
pub fn effortArms(model_type: []const u8) ?[]const EffortArm {
    if (Family.of(model_type)) |f| return builtin_arms[@intFromEnum(f)].slice();
    if (std.mem.eql(u8, model_type, "mimo_v2")) return &mimo_v2_efforts;
    return null;
}

/// The arms a request is served from now: the family's merged table, MiMo's on/off, null for an inherited arch.
pub fn servedArms(model_type: []const u8) ?Arms {
    const snap = snapshot();
    return servedArmsIn(&snap, model_type);
}

pub fn servedArmsIn(snap: *const Snapshot, model_type: []const u8) ?Arms {
    if (Family.of(model_type)) |f| return merge(f, snap[@intFromEnum(f)]).offeredArms();
    const built = effortArms(model_type) orelse return null;
    var out: Arms = .{ .len = built.len };
    @memcpy(out.buf[0..built.len], built);
    return out;
}

const scale_text = blk: {
    @setEvalBranchQuota(100_000);
    var t: [101][]const u8 = undefined;
    for (&t, 0..) |*s, i| s.* = std.fmt.comptimePrint("{d}", .{i});
    break :blk t;
};

/// `n` (0-100) as decimal text with static lifetime.
pub fn scaleText(n: u8) []const u8 {
    return scale_text[@min(n, 100)];
}

/// The value the template's `reasoning_effort` receives for `arm` (`chat.serializeExtraContext` maps it per family).
pub fn templateValue(arm: EffortArm) []const u8 {
    return switch (arm.sends) {
        .none => @tagName(arm.effort),
        .word => |w| @tagName(w),
        .scale => |n| scaleText(n),
    };
}

// ── The file's edits ──

pub const RowEdit = struct {
    offered: ?bool = null,
    sends: ?Sends = null,
    /// Outer null: not edited; inner null: no cap.
    cap: ??i32 = null,

    pub fn isEmpty(self: RowEdit) bool {
        return self.offered == null and self.sends == null and self.cap == null;
    }
};

/// One family's changes from its built-in table; a null field keeps the built-in value.
pub const Edits = struct {
    rows: [table_words.len]RowEdit = @splat(.{}),
    default: ?Effort = null,
};

pub fn merge(f: Family, edits: Edits) Table {
    var t = builtin(f);
    for (&t.rows, edits.rows) |*r, e| {
        if (e.offered) |o| {
            r.offered = o;
            r.fold_to = null;
        }
        if (e.sends) |s| r.sends = s;
        if (e.cap) |c| r.cap = c;
    }
    if (edits.default) |d| t.default = d;
    return t;
}

/// Only the fields that differ from the built-in table: what the file stores.
pub fn normalize(f: Family, edits: Edits) Edits {
    const b = builtin(f);
    var out = edits;
    for (&out.rows, b.rows) |*e, r| {
        if (e.offered) |o| if (o == r.offered and r.fold_to == null) {
            e.offered = null;
        };
        if (e.sends) |s| if (std.meta.eql(s, r.sends)) {
            e.sends = null;
        };
        if (e.cap) |c| if (c == r.cap) {
            e.cap = null;
        };
    }
    if (out.default) |d| if (d == b.default) {
        out.default = null;
    };
    return out;
}

/// A field the file or a save names that the family's template cannot take. `key` is the row word or "default".
pub const Problem = struct { key: []const u8, field: []const u8 = "", why: []const u8 };

pub fn checkOffered(f: Family, word: Effort, offered: bool) ?[]const u8 {
    if (word == .off and offered and !f.offOfferable()) return f.spec().off_why;
    return null;
}

pub fn checkSends(f: Family, word: Effort, s: Sends) ?[]const u8 {
    if (word == .off) return if (s == .none) null else "off turns thinking off and sends no effort";
    const ok = if (f.takesScale()) s == .scale and s.scale >= 1 and s.scale <= 100 else s == .word and std.mem.indexOfScalar(Effort, f.sendsWords(), s.word) != null;
    return if (ok) null else f.spec().sends_why;
}

pub fn checkCap(word: Effort, cap: ?i32) ?[]const u8 {
    if (word == .off) return if (cap == null) null else "off has no thinking to cap";
    if (cap) |c| if (c < 1) return "a cap is a whole number of tokens, 1 or more, or blank for none";
    return null;
}

/// Why a merged table cannot be served, or null.
pub fn checkTable(t: Table) ?Problem {
    if (!t.row(t.default).offered) return .{ .key = "default", .why = "the default word must be offered" };
    return null;
}

/// `edits` with whatever leaves the default word unoffered dropped: the file's default first, then its `offered: false`.
pub fn servable(f: Family, edits: Edits) Edits {
    var out = edits;
    if (checkTable(merge(f, out)) == null) return out;
    out.default = null;
    if (checkTable(merge(f, out)) == null) return out;
    out.rows[rowIndex(builtin(f).default).?].offered = null;
    return out;
}

fn wordKey(key: []const u8) ?Effort {
    const e = std.meta.stringToEnum(Effort, key) orelse return null;
    return if (rowIndex(e) != null) e else null;
}

fn note(problem: *?Problem, p: Problem) void {
    if (problem.* == null) problem.* = p;
}

/// One row object; a field the template cannot take is skipped and the first one lands in `problem`.
pub fn parseRow(f: Family, word: Effort, v: std.json.Value, problem: *?Problem) RowEdit {
    const key = @tagName(word);
    var out: RowEdit = .{};
    if (v != .object) {
        note(problem, .{ .key = key, .why = "a row is an object of offered, sends and cap" });
        return out;
    }
    var it = v.object.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        const x = kv.value_ptr.*;
        if (std.mem.eql(u8, name, "offered")) {
            if (x != .bool) {
                note(problem, .{ .key = key, .field = "offered", .why = "offered is true or false" });
            } else if (checkOffered(f, word, x.bool)) |why| {
                note(problem, .{ .key = key, .field = "offered", .why = why });
            } else out.offered = x.bool;
        } else if (std.mem.eql(u8, name, "sends")) {
            const s: ?Sends = switch (x) {
                .integer => |i| if (f.takesScale() and i >= 1 and i <= 100) Sends{ .scale = @intCast(i) } else null,
                .string => |text| if (!f.takesScale()) (if (std.meta.stringToEnum(Effort, text)) |e| Sends{ .word = e } else null) else null,
                else => null,
            };
            if (s == null or checkSends(f, word, s.?) != null) {
                note(problem, .{ .key = key, .field = "sends", .why = checkSends(f, word, s orelse .{ .word = .on }) orelse "not a value this template takes" });
            } else out.sends = s;
        } else if (std.mem.eql(u8, name, "cap")) {
            const cap: ?i32 = switch (x) {
                .null => null,
                .integer => |i| if (i >= 1 and i <= std.math.maxInt(i32)) @intCast(i) else 0,
                else => 0,
            };
            if (checkCap(word, cap)) |why| {
                note(problem, .{ .key = key, .field = "cap", .why = why });
            } else out.cap = cap;
        } else note(problem, .{ .key = key, .field = name, .why = "not a field of an effort row" });
    }
    return out;
}

/// The `default` value: a table word; null with a problem otherwise.
pub fn parseDefault(v: std.json.Value, problem: *?Problem) ?Effort {
    if (v == .string) if (wordKey(v.string)) |e| return e;
    note(problem, .{ .key = "default", .why = "the default is one of off, low, medium, high, xhigh, max" });
    return null;
}

pub const Parsed = struct { edits: Edits, problem: ?Problem };

/// One family's file section. Every bad field is skipped; the first is returned for a save to refuse, and `log_problems`
/// (a load) logs each one.
pub fn parseSection(f: Family, v: std.json.Value, log_problems: bool) Parsed {
    var out: Parsed = .{ .edits = .{}, .problem = null };
    if (v != .object) {
        out.problem = .{ .key = "", .why = "the section is an object" };
        if (log_problems) log.warn("[effort] model-settings.json effort:{s}: not an object, ignored\n", .{@tagName(f)});
        return out;
    }
    var it = v.object.iterator();
    while (it.next()) |kv| {
        var problem: ?Problem = null;
        const name = kv.key_ptr.*;
        if (std.mem.eql(u8, name, "default")) {
            out.edits.default = parseDefault(kv.value_ptr.*, &problem);
        } else if (wordKey(name)) |word| {
            out.edits.rows[rowIndex(word).?] = parseRow(f, word, kv.value_ptr.*, &problem);
        } else problem = .{ .key = name, .why = "not an effort word" };
        if (problem) |p| {
            if (log_problems) log.warn("[effort] model-settings.json effort:{s}.{s}{s}{s}: {s}; ignored\n", .{ @tagName(f), p.key, if (p.field.len > 0) "." else "", p.field, p.why });
            note(&out.problem, p);
        }
    }
    return out;
}

/// A row's stored fields as the file spells them.
pub fn rowJson(arena: std.mem.Allocator, f: Family, e: RowEdit) !std.json.Value {
    _ = f;
    var obj: std.json.ObjectMap = .empty;
    if (e.offered) |o| try obj.put(arena, "offered", .{ .bool = o });
    if (e.sends) |s| try obj.put(arena, "sends", switch (s) {
        .none => .null,
        .word => |w| .{ .string = @tagName(w) },
        .scale => |n| .{ .integer = n },
    });
    if (e.cap) |c| try obj.put(arena, "cap", if (c) |n| .{ .integer = n } else .null);
    return .{ .object = obj };
}

/// The model-settings.json section a family's table lives in; a model's key is an absolute path, so the two never meet.
pub fn sectionName(buf: []u8, f: Family) []const u8 {
    return std.fmt.bufPrint(buf, "effort:{s}", .{@tagName(f)}) catch unreachable;
}

// ── The served tables ──

var store_lock: std.atomic.Mutex = .unlocked;
var store: [families.len]Edits = @splat(.{});

fn lockStore() void {
    while (!store_lock.tryLock()) std.atomic.spinLoopHint();
}

/// Makes `edits` the family's table from the next request on; the caller has checked it (`servable`).
pub fn publish(f: Family, edits: Edits) void {
    lockStore();
    defer store_lock.unlock();
    store[@intFromEnum(f)] = edits;
}

pub fn publishedEdits(f: Family) Edits {
    lockStore();
    defer store_lock.unlock();
    return store[@intFromEnum(f)];
}

/// Every family's edits in one read: a request resolves against one table throughout while a save publishes another.
pub const Snapshot = [families.len]Edits;

pub fn snapshot() Snapshot {
    lockStore();
    defer store_lock.unlock();
    return store;
}

pub fn table(f: Family) Table {
    return merge(f, publishedEdits(f));
}

/// The default word the file names for this arch's family, if it names one.
pub fn fileDefault(model_type: []const u8) ?Effort {
    const snap = snapshot();
    return fileDefaultIn(&snap, model_type);
}

pub fn fileDefaultIn(snap: *const Snapshot, model_type: []const u8) ?Effort {
    return snap[@intFromEnum(Family.of(model_type) orelse return null)].default;
}

/// `none` is an alias of off. No served table lists `minimal`; inherited arches
/// keep its legacy budget (`responses.effortBudget`).
pub fn parseEffort(word: []const u8) ?Effort {
    if (std.mem.eql(u8, word, "none")) return .off;
    return std.meta.stringToEnum(Effort, word);
}

pub fn findEffortArm(arms: []const EffortArm, effort: Effort) ?EffortArm {
    for (arms) |a| if (a.effort == effort) return a;
    return null;
}

/// The effort `e` selects on a model offering `offered`; null = refused.
pub const EffortPick = struct {
    effort: Effort,
    /// Set when the asked word was re-routed. MiMo's on-fold leaves this null.
    rerouted_from: ?Effort = null,
};

pub fn selectEffortPick(offered: []const Effort, e: Effort) ?EffortPick {
    if (std.mem.indexOfScalar(Effort, offered, e) != null) return .{ .effort = e };
    if ((e == .high or e == .max) and std.mem.indexOfScalar(Effort, offered, .xhigh) != null)
        return .{ .effort = .xhigh, .rerouted_from = e };
    if (e == .medium and std.mem.indexOfScalar(Effort, offered, .high) != null)
        return .{ .effort = .high, .rerouted_from = e };
    if (e == .xhigh and std.mem.indexOfScalar(Effort, offered, .max) != null)
        return .{ .effort = .max, .rerouted_from = e };
    if (e != .off and std.mem.indexOfScalar(Effort, offered, .on) != null) return .{ .effort = .on };
    return null;
}

pub fn selectEffort(offered: []const Effort, e: Effort) ?Effort {
    const pick = selectEffortPick(offered, e) orelse return null;
    return pick.effort;
}

/// `{model} has no `{asked}` effort; served at `{served}`.
pub const effort_reroute_notice = "{s} has no `{s}` effort; served at `{s}`";

pub fn formatEffortRerouteNotice(allocator: std.mem.Allocator, model_name: []const u8, from: Effort, to: Effort) ![]u8 {
    return std.fmt.allocPrint(allocator, effort_reroute_notice, .{ model_name, @tagName(from), @tagName(to) });
}

pub const Resolved = struct { arm: EffortArm, rerouted_from: ?Effort = null };

/// The arm a request for `e` is served from: its own, its fold's target (with the asked word as `rerouted_from`), or
/// on MiMo `on` for every thinking word, silently. Null = refused.
pub fn resolve(arms: []const EffortArm, e: Effort) ?Resolved {
    if (findEffortArm(arms, e)) |a| {
        const to = a.fold_to orelse return .{ .arm = a };
        const target = findEffortArm(arms, to) orelse return null;
        return if (isListed(target)) .{ .arm = target, .rerouted_from = e } else null;
    }
    if (e != .off) if (findEffortArm(arms, .on)) |a| return .{ .arm = a };
    return null;
}

/// The arm a client's effort word selects (`resolve`); null = refused.
pub fn armForWord(arms: []const EffortArm, word: []const u8) ?EffortArm {
    return (resolve(arms, parseEffort(word) orelse return null) orelse return null).arm;
}

/// The words a request may name as themselves, as `/v1/models` lists them.
pub fn listedWords(arms: []const EffortArm, buf: []Effort) []const Effort {
    var n: usize = 0;
    for (arms) |a| if (isListed(a)) {
        buf[n] = a.effort;
        n += 1;
    };
    return buf[0..n];
}

test "DeepSeek-V4.1 serves medium and xhigh as their own words, without a re-route" {
    const arms = effortArms("deepseek_v41").?;
    var offered: [std.enums.values(Effort).len]Effort = undefined;
    for (arms, 0..) |a, i| offered[i] = a.effort;
    for ([_]Effort{ .off, .low, .medium, .high, .xhigh, .max }) |e| {
        const pick = selectEffortPick(offered[0..arms.len], e).?;
        try testing.expectEqual(e, pick.effort);
        try testing.expect(pick.rerouted_from == null);
    }
}

test "selectEffort: Qwen high and max re-route to xhigh; other unknowns stay refused" {
    const qwen = [_]Effort{ .off, .low, .medium, .xhigh };
    try testing.expectEqual(@as(?Effort, .xhigh), selectEffort(&qwen, .high));
    try testing.expectEqual(@as(?Effort, .xhigh), selectEffort(&qwen, .max));
    try testing.expectEqual(@as(?Effort, .xhigh), selectEffort(&qwen, .xhigh));
    try testing.expectEqual(@as(?Effort, null), selectEffort(&qwen, .minimal));
    const glm = [_]Effort{ .low, .high, .max };
    try testing.expectEqual(@as(?Effort, .high), selectEffort(&glm, .high));
    try testing.expectEqual(@as(?Effort, .max), selectEffort(&glm, .max));
    try testing.expectEqual(@as(?Effort, .high), selectEffort(&glm, .medium));
    try testing.expectEqual(@as(?Effort, .max), selectEffort(&glm, .xhigh));
    const mimo = [_]Effort{ .off, .on };
    try testing.expectEqual(@as(?Effort, .on), selectEffort(&mimo, .high));
    try testing.expectEqual(@as(?Effort, .on), selectEffort(&mimo, .max));
    try testing.expectEqual(@as(?Effort, .off), selectEffort(&mimo, .off));
}

test "selectEffort: GLM medium re-routes to high and xhigh to max; other missing words stay refused" {
    const glm = [_]Effort{ .low, .high, .max };
    try testing.expectEqual(@as(?Effort, .high), selectEffort(&glm, .medium));
    try testing.expectEqual(@as(?Effort, .max), selectEffort(&glm, .xhigh));
    try testing.expectEqual(Effort.medium, selectEffortPick(&glm, .medium).?.rerouted_from.?);
    try testing.expectEqual(Effort.xhigh, selectEffortPick(&glm, .xhigh).?.rerouted_from.?);
    try testing.expect(selectEffortPick(&glm, .high).?.rerouted_from == null);
    try testing.expect(selectEffortPick(&glm, .max).?.rerouted_from == null);
    try testing.expect(selectEffort(&glm, .off) == null);
    try testing.expect(selectEffort(&glm, .minimal) == null);
    try testing.expect(selectEffort(&glm, .on) == null);
    const mimo = [_]Effort{ .off, .on };
    try testing.expect(selectEffortPick(&mimo, .medium).?.rerouted_from == null);
    try testing.expect(selectEffortPick(&mimo, .xhigh).?.rerouted_from == null);
}

test "selectEffortPick: only Qwen high and max carry a re-route; MiMo's on-fold stays quiet" {
    const qwen = [_]Effort{ .off, .low, .medium, .xhigh };
    const high = selectEffortPick(&qwen, .high).?;
    try testing.expectEqual(Effort.xhigh, high.effort);
    try testing.expectEqual(Effort.high, high.rerouted_from.?);
    try testing.expect(selectEffortPick(&qwen, .xhigh).?.rerouted_from == null);
    try testing.expect(selectEffortPick(&qwen, .minimal) == null);
    const mimo = [_]Effort{ .off, .on };
    try testing.expect(selectEffortPick(&mimo, .max).?.rerouted_from == null);
    try testing.expectEqual(Effort.on, selectEffortPick(&mimo, .max).?.effort);
    const glm = [_]Effort{ .low, .high, .max };
    try testing.expect(selectEffortPick(&glm, .high).?.rerouted_from == null);
    try testing.expectEqual(Effort.max, selectEffortPick(&glm, .xhigh).?.effort);
    try testing.expectEqual(Effort.xhigh, selectEffortPick(&glm, .xhigh).?.rerouted_from.?);
    const notice = try formatEffortRerouteNotice(testing.allocator, "Qwen3.8-Flash-Next", .high, .xhigh);
    defer testing.allocator.free(notice);
    try testing.expectEqualStrings("Qwen3.8-Flash-Next has no `high` effort; served at `xhigh`", notice);
}

fn parseJson(text: []const u8) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, testing.allocator, text, .{});
}

test "effort table: the built-in tables are the arms each family served before the table" {
    for ([_]Family{ .qwen4_exp, .deepseek_v41, .glm5_next }) |f| {
        const arms = servedArms(@tagName(f)).?;
        const before = effortArms(@tagName(f)).?;
        try testing.expectEqual(before.len, arms.len);
        for (arms.slice(), before) |a, b| try testing.expectEqualDeep(b, a);
    }
    const q = builtin(.qwen4_exp);
    try testing.expectEqual(Effort.medium, q.default);
    try testing.expectEqualDeep(Row{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh }, q.row(.high));
    try testing.expectEqualDeep(Row{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh }, q.row(.max));
    try testing.expectEqualDeep(Row{ .offered = true, .sends = .{ .word = .low }, .cap = 2048 }, q.row(.low));
    const d = builtin(.deepseek_v41);
    for ([_]Effort{ .low, .medium, .high, .xhigh, .max }, [_]u8{ 50, 62, 75, 87, 100 }) |w, n| try testing.expectEqualDeep(Row{ .offered = true, .sends = .{ .scale = n } }, d.row(w));
    const g = builtin(.glm5_next);
    try testing.expect(!g.row(.off).offered);
    try testing.expectEqualDeep(Row{ .offered = false, .sends = .{ .word = .high }, .fold_to = .high }, g.row(.medium));
    try testing.expectEqualDeep(Row{ .offered = false, .sends = .{ .word = .max }, .fold_to = .max }, g.row(.xhigh));
    try testing.expect(Family.of("mimo_v2") == null);
    try testing.expectEqual(@as(usize, 2), servedArms("mimo_v2").?.len);
    try testing.expect(servedArms("not_a_served_arch") == null);
}

test "effort table: a row's template value is its word, or its number on the 1-100 scale" {
    try testing.expectEqualStrings("medium", templateValue(.{ .effort = .high, .sends = .{ .word = .medium } }));
    try testing.expectEqualStrings("62", templateValue(.{ .effort = .medium, .sends = .{ .scale = 62 } }));
    try testing.expectEqualStrings("100", templateValue(.{ .effort = .max, .sends = .{ .scale = 100 } }));
    try testing.expectEqualStrings("on", templateValue(.{ .effort = .on }));
}

test "effort table: the file's fields merge row by row over the built-in table" {
    const parsed = try parseJson(
        \\{"default":"low","high":{"offered":true,"sends":"medium","cap":64},"low":{"cap":null},"medium":{"offered":false}}
    );
    defer parsed.deinit();
    const got = parseSection(.qwen4_exp, parsed.value, false);
    try testing.expect(got.problem == null);
    const t = merge(.qwen4_exp, got.edits);
    try testing.expectEqual(Effort.low, t.default);
    try testing.expectEqualDeep(Row{ .offered = true, .sends = .{ .word = .medium }, .cap = 64 }, t.row(.high));
    try testing.expectEqualDeep(Row{ .offered = true, .sends = .{ .word = .low }, .cap = null }, t.row(.low));
    try testing.expect(!t.row(.medium).offered);
    try testing.expectEqualDeep(builtin(.qwen4_exp).row(.xhigh), t.row(.xhigh));
    const arms = t.offeredArms();
    var listed: [8]Effort = undefined;
    try testing.expectEqualSlices(Effort, &.{ .off, .low, .high, .xhigh }, listedWords(arms.slice(), &listed));
    try testing.expectEqual(Effort.xhigh, findEffortArm(arms.slice(), .max).?.fold_to.?);
    try testing.expectEqualDeep(EffortArm{ .effort = .high, .budget = 64, .sends = .{ .word = .medium } }, findEffortArm(arms.slice(), .high).?);
}

test "effort table: each family's template takes only its own values, on save and on load" {
    const Case = struct { f: Family, body: []const u8, field: []const u8 };
    for ([_]Case{
        .{ .f = .qwen4_exp, .body = "{\"high\":{\"sends\":\"high\"}}", .field = "sends" },
        .{ .f = .qwen4_exp, .body = "{\"low\":{\"sends\":62}}", .field = "sends" },
        .{ .f = .glm5_next, .body = "{\"low\":{\"sends\":\"medium\"}}", .field = "sends" },
        .{ .f = .glm5_next, .body = "{\"off\":{\"offered\":true}}", .field = "offered" },
        .{ .f = .deepseek_v41, .body = "{\"medium\":{\"sends\":0}}", .field = "sends" },
        .{ .f = .deepseek_v41, .body = "{\"medium\":{\"sends\":101}}", .field = "sends" },
        .{ .f = .deepseek_v41, .body = "{\"medium\":{\"sends\":\"high\"}}", .field = "sends" },
        .{ .f = .deepseek_v41, .body = "{\"medium\":{\"sends\":62.5}}", .field = "sends" },
        .{ .f = .deepseek_v41, .body = "{\"off\":{\"sends\":40}}", .field = "sends" },
        .{ .f = .qwen4_exp, .body = "{\"low\":{\"cap\":0}}", .field = "cap" },
        .{ .f = .qwen4_exp, .body = "{\"low\":{\"cap\":-5}}", .field = "cap" },
        .{ .f = .qwen4_exp, .body = "{\"low\":{\"cap\":\"64\"}}", .field = "cap" },
        .{ .f = .qwen4_exp, .body = "{\"off\":{\"cap\":64}}", .field = "cap" },
        .{ .f = .qwen4_exp, .body = "{\"low\":{\"offered\":\"yes\"}}", .field = "offered" },
        .{ .f = .qwen4_exp, .body = "{\"low\":{\"colour\":1}}", .field = "colour" },
        .{ .f = .qwen4_exp, .body = "{\"ultra\":{\"offered\":true}}", .field = "" },
        .{ .f = .qwen4_exp, .body = "{\"default\":\"ultra\"}", .field = "" },
        .{ .f = .qwen4_exp, .body = "{\"low\":7}", .field = "" },
    }) |c| {
        const parsed = try parseJson(c.body);
        defer parsed.deinit();
        const got = parseSection(c.f, parsed.value, false);
        const p = got.problem orelse return error.TestExpectedProblem;
        try testing.expectEqualStrings(c.field, p.field);
        try testing.expect(p.why.len > 0);
        // The bad field is skipped, so the table it loads is the built-in one.
        try testing.expectEqualDeep(builtin(c.f), merge(c.f, got.edits));
    }
    const good = try parseJson("{\"medium\":{\"sends\":40,\"cap\":64},\"off\":{\"offered\":false}}");
    defer good.deinit();
    const ds = parseSection(.deepseek_v41, good.value, false);
    try testing.expect(ds.problem == null);
    try testing.expectEqualDeep(Row{ .offered = true, .sends = .{ .scale = 40 }, .cap = 64 }, merge(.deepseek_v41, ds.edits).row(.medium));
}

test "effort table: a load keeps the good fields of a row beside a bad one" {
    const parsed = try parseJson("{\"high\":{\"offered\":true,\"sends\":\"ultra\",\"cap\":64}}");
    defer parsed.deinit();
    const got = parseSection(.qwen4_exp, parsed.value, false);
    try testing.expect(got.problem != null);
    try testing.expectEqualDeep(Row{ .offered = true, .sends = .{ .word = .xhigh }, .cap = 64 }, merge(.qwen4_exp, got.edits).row(.high));
}

test "effort table: the default word must be offered; a load that breaks it drops the edit that did" {
    var e: Edits = .{};
    e.rows[rowIndex(.medium).?].offered = false;
    try testing.expectEqualStrings("default", checkTable(merge(.qwen4_exp, e)).?.key);
    const kept = servable(.qwen4_exp, e);
    try testing.expect(merge(.qwen4_exp, kept).row(.medium).offered);
    var d: Edits = .{ .default = .high };
    try testing.expect(checkTable(merge(.qwen4_exp, d)) != null);
    try testing.expect(servable(.qwen4_exp, d).default == null);
    d.rows[rowIndex(.high).?].offered = true;
    try testing.expect(checkTable(merge(.qwen4_exp, d)) == null);
    try testing.expectEqual(@as(?Effort, .high), servable(.qwen4_exp, d).default);
    // GLM offers no off, so off can never be its default.
    try testing.expect(checkTable(merge(.glm5_next, .{ .default = .off })) != null);
    try testing.expect(checkTable(merge(.deepseek_v41, .{ .default = .off })) == null);
}

test "effort table: the file keeps only the fields that differ from the built-in table" {
    var e: Edits = .{ .default = .medium };
    e.rows[rowIndex(.low).?] = .{ .offered = true, .sends = .{ .word = .low }, .cap = @as(?i32, 2048) };
    e.rows[rowIndex(.high).?] = .{ .offered = true, .sends = .{ .word = .xhigh }, .cap = @as(?i32, 64) };
    const n = normalize(.qwen4_exp, e);
    try testing.expect(n.default == null);
    try testing.expect(n.rows[rowIndex(.low).?].isEmpty());
    try testing.expectEqualDeep(RowEdit{ .offered = true, .cap = @as(?i32, 64) }, n.rows[rowIndex(.high).?]);
    var a = std.heap.ArenaAllocator.init(testing.allocator);
    defer a.deinit();
    const v = try rowJson(a.allocator(), .deepseek_v41, .{ .sends = .{ .scale = 40 }, .cap = @as(?i32, null) });
    const text = try std.json.Stringify.valueAlloc(a.allocator(), v, .{});
    try testing.expectEqualStrings("{\"sends\":40,\"cap\":null}", text);
    const back = try parseJson(text);
    defer back.deinit();
    var problem: ?Problem = null;
    try testing.expectEqualDeep(RowEdit{ .sends = .{ .scale = 40 }, .cap = @as(?i32, null) }, parseRow(.deepseek_v41, .medium, back.value, &problem));
    try testing.expect(problem == null);
}

test "effort table: a published table reaches the next lookup, and a reset brings the built-in one back" {
    defer publish(.glm5_next, .{});
    var e: Edits = .{};
    e.rows[rowIndex(.low).?].cap = @as(?i32, 64);
    publish(.glm5_next, e);
    try testing.expectEqual(@as(?i32, 64), findEffortArm(servedArms("glm5_next").?.slice(), .low).?.budget);
    publish(.glm5_next, .{});
    try testing.expectEqual(@as(?i32, null), findEffortArm(servedArms("glm5_next").?.slice(), .low).?.budget);
}

test "effort table: --think picks its word from the served table" {
    const saved = think_effort_flag;
    defer think_effort_flag = saved;
    defer publish(.qwen4_exp, .{});
    think_effort_flag = .high;
    // Not offered: the old fold serves it at xhigh.
    try testing.expectEqual(Effort.xhigh, thinkFlagArm("qwen4_exp").?.effort);
    var e: Edits = .{};
    e.rows[rowIndex(.high).?] = .{ .offered = true, .sends = .{ .word = .medium }, .cap = @as(?i32, 64) };
    publish(.qwen4_exp, e);
    try testing.expectEqualDeep(EffortArm{ .effort = .high, .budget = 64, .sends = .{ .word = .medium } }, thinkFlagArm("qwen4_exp").?);
}

test "effort table: only the built-in folds fold; a file's offered false refuses the word" {
    const arms = builtin(.qwen4_exp).offeredArms();
    const high = resolve(arms.slice(), .high).?;
    try testing.expectEqual(Effort.xhigh, high.arm.effort);
    try testing.expectEqual(Effort.high, high.rerouted_from.?);
    try testing.expect(resolve(arms.slice(), .minimal) == null);
    var e: Edits = .{};
    e.rows[rowIndex(.high).?].offered = false;
    try testing.expect(!normalize(.qwen4_exp, e).rows[rowIndex(.high).?].isEmpty());
    const refused = merge(.qwen4_exp, e).offeredArms();
    try testing.expect(resolve(refused.slice(), .high) == null);
    try testing.expectEqual(Effort.max, resolve(refused.slice(), .max).?.rerouted_from.?);
    // A fold whose target is no longer offered is refused too.
    var x: Edits = .{};
    x.rows[rowIndex(.xhigh).?].offered = false;
    const no_target = merge(.qwen4_exp, x).offeredArms();
    try testing.expect(resolve(no_target.slice(), .high) == null);
    try testing.expect(resolve(no_target.slice(), .xhigh) == null);
    // DeepSeek has no folds: a word turned off is refused.
    var d: Edits = .{};
    d.rows[rowIndex(.medium).?].offered = false;
    try testing.expect(resolve(merge(.deepseek_v41, d).offeredArms().slice(), .medium) == null);
    // MiMo's on-fold stays quiet.
    const mimo = resolve(effortArms("mimo_v2").?, .high).?;
    try testing.expectEqual(Effort.on, mimo.arm.effort);
    try testing.expect(mimo.rerouted_from == null);
}
