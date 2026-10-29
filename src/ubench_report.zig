//! The microbench harness without MLX: the CLI grammar, the registry's types, the `Run` a bench body drives, the
//! result tables, the JSON and `compare`. `sushi-ubench` (ubench_main.zig) supplies the box and the GPU; a bench
//! body reaches this file through its host root as `report`. It holds the only tests a microbench has: a bench file
//! never carries a `test` block ([process-ubench](../docs/process-ubench.md)).
const std = @import("std");

pub const Scope = enum { qwen4, mimo2, glm5, dsv41, enc, shared };
pub const Kind = enum { kernel, pack };
pub const Unit = enum {
    ns,
    us,
    ms,

    pub fn perNs(u: Unit) f64 {
        return switch (u) {
            .ns => 1,
            .us => 1e3,
            .ms => 1e6,
        };
    }
};

/// A rung axis: a numeric one takes any positive integer, a categorical one only its listed values.
pub const Axis = struct { name: []const u8, values: []const []const u8, numeric: bool = true };
pub const Opt = struct { name: []const u8, help: []const u8 };

pub const Bench = struct {
    /// `<scope>.<name>`.
    name: []const u8,
    scope: Scope,
    kind: Kind = .kernel,
    what: []const u8,
    unit: Unit = .us,
    /// Dependent calls per sample when `--steps` is not given.
    steps: usize = 16,
    axes: []const Axis = &.{},
    arms: []const []const u8,
    ref: []const u8,
    opts: []const Opt = &.{},
    run: *const fn (*Run) anyerror!void,

    fn armIndex(b: *const Bench, name: []const u8) ?usize {
        for (b.arms, 0..) |arm, i| if (std.mem.eql(u8, arm, name)) return i;
        return null;
    }

    fn axisNamed(b: *const Bench, name: []const u8) ?Axis {
        for (b.axes) |ax| if (std.mem.eql(u8, ax.name, name)) return ax;
        return null;
    }

    fn hasOpt(b: *const Bench, name: []const u8) bool {
        for (b.opts) |o| if (std.mem.eql(u8, o.name, name)) return true;
        return false;
    }
};

pub const max_arms = 16;
pub const max_rounds = 1000;
pub const default_rounds = 7;

// ── CLI ─────────────────────────────────────────────────────────────────

pub const Selection = struct { name: []const u8, values: []const []const u8 };
pub const KV = struct { key: []const u8, value: []const u8 };

pub const RunArgs = struct {
    target: []const u8,
    rungs: []const Selection = &.{},
    rounds: ?usize = null,
    steps: ?usize = null,
    arms: ?[]const []const u8 = null,
    ref: ?[]const u8 = null,
    pack: ?[]const u8 = null,
    opts: []const KV = &.{},
    out: ?[]const u8 = null,
    contended: bool = false,
};

pub const Command = union(enum) {
    help,
    list: ?[]const u8,
    run: RunArgs,
    compare: [2][]const u8,
};

pub const USAGE =
    \\usage: sushi-ubench <command>
    \\
    \\  list [scope]                       the registry: one bench per line with its axes, arms and opts
    \\  run <name|scope:all> [flags]       time a bench on a quiet box and print rungs x arms
    \\  compare <a.json> <b.json>          two results of one bench, cell by cell
    \\
    \\run flags:
    \\  --rungs <axis>=<v,v..>             rungs to time on one axis (repeat for several axes)
    \\  --rounds N                         same-process rounds per rung (default 7, at most 1000)
    \\  --steps N                          dependent calls per round
    \\  --arms a,b                         arms to time (default: all)
    \\  --ref arm                          the arm the ratios are read against (default: the bench's)
    \\  --pack <dir>                       a pack for the benches that need one
    \\  --opt <k>=<v>                      a knob the bench declares (repeat for several)
    \\  --out <dir>                        where the JSON goes (default ~/.sushi/runs/ubench)
    \\  --contended                        keep only the GPU lock; the result is stamped CONTENDED
    \\
    \\A bench is its own binary run, never a test: the exit code says whether it measured, never how fast.
    \\Quiet-box steps (lock, QUIET, power, no live sushi) are taken by the binary; run it under `taskpolicy -a`.
    \\
;

pub const ParseError = error{ BadArgs, OutOfMemory };

fn bad(a: std.mem.Allocator, msg: *[]const u8, comptime fmt: []const u8, args: anytype) ParseError {
    msg.* = try std.fmt.allocPrint(a, fmt, args);
    return error.BadArgs;
}

fn positive(text: []const u8, limit: usize) ?usize {
    const n = std.fmt.parseInt(usize, text, 10) catch return null;
    return if (n >= 1 and n <= limit) n else null;
}

fn splitList(a: std.mem.Allocator, text: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |part| try out.append(a, part);
    return out.items;
}

/// `args` excludes argv[0]. On `BadArgs`, `msg` says which argument and why.
pub fn parse(a: std.mem.Allocator, args: []const []const u8, msg: *[]const u8) ParseError!Command {
    if (args.len == 0) return error.BadArgs;
    for (args) |arg| if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) return .help;
    const cmd = args[0];
    if (std.mem.eql(u8, cmd, "list")) {
        if (args.len > 2) return bad(a, msg, "list takes at most one scope", .{});
        return .{ .list = if (args.len == 2) args[1] else null };
    }
    if (std.mem.eql(u8, cmd, "compare")) {
        if (args.len != 3) return bad(a, msg, "compare takes two result files", .{});
        return .{ .compare = .{ args[1], args[2] } };
    }
    if (!std.mem.eql(u8, cmd, "run")) return bad(a, msg, "unknown command {s}", .{cmd});
    if (args.len < 2 or std.mem.startsWith(u8, args[1], "--")) return bad(a, msg, "run needs a bench name or scope:all", .{});
    var ra: RunArgs = .{ .target = args[1] };
    var rungs: std.ArrayList(Selection) = .empty;
    var opts: std.ArrayList(KV) = .empty;
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        if (std.mem.eql(u8, flag, "--contended")) {
            ra.contended = true;
            continue;
        }
        const known = for ([_][]const u8{ "--rungs", "--rounds", "--steps", "--arms", "--ref", "--pack", "--opt", "--out" }) |k| {
            if (std.mem.eql(u8, flag, k)) break true;
        } else false;
        if (!known) return bad(a, msg, "unknown argument {s}", .{flag});
        i += 1;
        if (i >= args.len) return bad(a, msg, "{s} needs a value", .{flag});
        const value = args[i];
        if (std.mem.eql(u8, flag, "--rounds")) {
            ra.rounds = positive(value, max_rounds) orelse return bad(a, msg, "--rounds {s}: a count from 1 to {d}", .{ value, max_rounds });
        } else if (std.mem.eql(u8, flag, "--steps")) {
            ra.steps = positive(value, 1 << 20) orelse return bad(a, msg, "--steps {s}: a positive count", .{value});
        } else if (std.mem.eql(u8, flag, "--arms")) {
            ra.arms = try splitList(a, value);
        } else if (std.mem.eql(u8, flag, "--ref")) {
            ra.ref = value;
        } else if (std.mem.eql(u8, flag, "--pack")) {
            ra.pack = value;
        } else if (std.mem.eql(u8, flag, "--out")) {
            ra.out = value;
        } else {
            const eq = std.mem.indexOfScalar(u8, value, '=') orelse return bad(a, msg, "{s} {s}: expected <name>=<value>", .{ flag, value });
            if (eq == 0 or eq + 1 == value.len) return bad(a, msg, "{s} {s}: expected <name>=<value>", .{ flag, value });
            if (std.mem.eql(u8, flag, "--opt")) {
                try opts.append(a, .{ .key = value[0..eq], .value = value[eq + 1 ..] });
            } else {
                try rungs.append(a, .{ .name = value[0..eq], .values = try splitList(a, value[eq + 1 ..]) });
            }
        }
    }
    ra.rungs = rungs.items;
    ra.opts = opts.items;
    return .{ .run = ra };
}

// ── Selecting benches ───────────────────────────────────────────────────

/// One bench's resolved run: the arms in force as indices into `bench.arms`, and only the flags it declares.
pub const Plan = struct {
    bench: *const Bench,
    arms: []const usize,
    ref: usize,
    rungs: []const Selection,
    opts: []const KV,
};

pub const Selected = struct { plans: []const Plan, skipped: []const []const u8 };

fn scopeOf(name: []const u8) ?Scope {
    return std.meta.stringToEnum(Scope, name);
}

/// Everything that can be wrong with the flags, found before anything runs.
pub fn select(a: std.mem.Allocator, registry: []const Bench, ra: RunArgs, msg: *[]const u8) ParseError!Selected {
    var plans: std.ArrayList(Plan) = .empty;
    var skipped: std.ArrayList([]const u8) = .empty;
    var group = false;
    if (std.mem.endsWith(u8, ra.target, ":all")) {
        group = true;
        const scope = scopeOf(ra.target[0 .. ra.target.len - ":all".len]) orelse return bad(a, msg, "unknown scope in {s}", .{ra.target});
        for (registry) |*b| {
            if (b.scope != scope) continue;
            if (b.kind == .pack and ra.pack == null) {
                try skipped.append(a, b.name);
                continue;
            }
            try plans.append(a, .{ .bench = b, .arms = &.{}, .ref = 0, .rungs = &.{}, .opts = &.{} });
        }
        if (plans.items.len == 0) return bad(a, msg, "{s}: no bench to run", .{ra.target});
    } else {
        const b = for (registry) |*b| {
            if (std.mem.eql(u8, b.name, ra.target)) break b;
        } else return bad(a, msg, "unknown bench {s} (sushi-ubench list)", .{ra.target});
        if (b.kind == .pack and ra.pack == null) return bad(a, msg, "{s} needs --pack", .{b.name});
        try plans.append(a, .{ .bench = b, .arms = &.{}, .ref = 0, .rungs = &.{}, .opts = &.{} });
    }
    if (group and (ra.arms != null or ra.ref != null)) return bad(a, msg, "--arms and --ref name one bench's arms; run a single bench", .{});

    for (ra.rungs) |sel| {
        var known = false;
        for (plans.items) |p| {
            const ax = p.bench.axisNamed(sel.name) orelse continue;
            known = true;
            for (sel.values) |v| {
                if (ax.numeric) {
                    if (positive(v, 1 << 40) == null) return bad(a, msg, "--rungs {s}: {s} is not a positive integer", .{ sel.name, v });
                } else if (!listed(ax.values, v)) return bad(a, msg, "--rungs {s}: {s} is not one of the axis values", .{ sel.name, v });
            }
        }
        if (!known) return bad(a, msg, "--rungs: no selected bench has an axis {s}", .{sel.name});
    }
    for (ra.opts) |kv| {
        var known = false;
        for (plans.items) |p| known = known or p.bench.hasOpt(kv.key);
        if (!known) return bad(a, msg, "--opt: no selected bench declares {s}", .{kv.key});
    }

    for (plans.items) |*p| {
        const b = p.bench;
        if (b.arms.len == 0 or b.arms.len > max_arms) return bad(a, msg, "{s}: {d} arms, a bench has 1 to {d}", .{ b.name, b.arms.len, max_arms });
        var arms: std.ArrayList(usize) = .empty;
        if (ra.arms) |names| {
            for (names) |n| {
                const idx = b.armIndex(n) orelse return bad(a, msg, "--arms: {s} has no arm {s}", .{ b.name, n });
                if (std.mem.indexOfScalar(usize, arms.items, idx) != null) return bad(a, msg, "--arms: {s} twice", .{n});
                try arms.append(a, idx);
            }
        } else for (0..b.arms.len) |i| try arms.append(a, i);
        const ref_name = ra.ref orelse b.ref;
        const ref = b.armIndex(ref_name) orelse return bad(a, msg, "--ref: {s} has no arm {s}", .{ b.name, ref_name });
        if (std.mem.indexOfScalar(usize, arms.items, ref) == null) return bad(a, msg, "the reference arm {s} is not among the arms run", .{ref_name});
        var rungs: std.ArrayList(Selection) = .empty;
        for (ra.rungs) |s| if (b.axisNamed(s.name) != null) try rungs.append(a, s);
        var opts: std.ArrayList(KV) = .empty;
        for (ra.opts) |kv| if (b.hasOpt(kv.key)) try opts.append(a, kv);
        p.* = .{ .bench = b, .arms = arms.items, .ref = ref, .rungs = rungs.items, .opts = opts.items };
    }
    return .{ .plans = plans.items, .skipped = skipped.items };
}

fn listed(values: []const []const u8, v: []const u8) bool {
    for (values) |x| if (std.mem.eql(u8, x, v)) return true;
    return false;
}

pub fn writeList(w: *std.Io.Writer, registry: []const Bench, scope: ?[]const u8) !void {
    for (registry) |b| {
        if (scope) |s| if (!std.mem.eql(u8, @tagName(b.scope), s)) continue;
        try w.print("{s}  [{s}]  {s}\n", .{ b.name, @tagName(b.kind), b.what });
        for (b.axes) |ax| {
            try w.print("    axis {s}{s}: ", .{ ax.name, if (ax.numeric) "" else " (categorical)" });
            for (ax.values, 0..) |v, i| try w.print("{s}{s}", .{ if (i == 0) "" else ",", v });
            try w.writeByte('\n');
        }
        try w.writeAll("    arms: ");
        for (b.arms, 0..) |arm, i| try w.print("{s}{s}{s}", .{ if (i == 0) "" else ", ", arm, if (std.mem.eql(u8, arm, b.ref)) " (ref)" else "" });
        try w.writeByte('\n');
        for (b.opts) |o| try w.print("    opt {s}: {s}\n", .{ o.name, o.help });
    }
}

// ── Statistics ──────────────────────────────────────────────────────────

pub const Stats = struct { median: f64, min: f64, max: f64 };

/// Median of an even count is the mean of the middle two. `vals` is never empty and holds at most `max_rounds`.
pub fn stats(vals: []const f64) Stats {
    var buf: [max_rounds]f64 = undefined;
    const sorted = buf[0..vals.len];
    @memcpy(sorted, vals);
    std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
    const n = sorted.len;
    const med = if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
    return .{ .median = med, .min = sorted[0], .max = sorted[n - 1] };
}

/// The arm order of one round: a rotation by the round number, reversed on odd rounds, so no arm
/// always runs first, last or straight after the same neighbour.
pub fn armOrder(round: usize, out: []usize) void {
    const n = out.len;
    for (out, 0..) |*o, i| o.* = (i + round) % n;
    if (round % 2 == 1) std.mem.reverse(usize, out);
}

// ── The result ──────────────────────────────────────────────────────────

pub const Stamps = struct {
    commit: []const u8,
    dirty: bool,
    sha256: []const u8,
    mtime: []const u8,
    optimize: []const u8 = "ReleaseFast",
    chip: []const u8,
    qos: ?i32 = null,
    power: []const u8,
    die_start_c: ?f64 = null,
    die_end_c: ?f64 = null,
    lock: []const u8,
    contended: bool,
    time: []const u8,
};
pub const Cell = struct { arm: []const u8, ns: []const f64 };
pub const Row = struct { label: []const u8, cells: []const Cell };
pub const Metric = struct { row: []const u8, arm: []const u8, name: []const u8, unit: []const u8, value: f64 };
pub const Result = struct {
    bench: []const u8,
    kind: []const u8,
    what: []const u8,
    unit: Unit,
    stamps: Stamps,
    rounds: usize,
    steps: usize,
    ref: []const u8,
    arms: []const []const u8,
    axes: []const Selection,
    opts: []const KV,
    rows: []const Row,
    metrics: []const Metric,
    /// `valid`, `contended`, or `invalid:<reason>`.
    validity: []const u8,

    fn cell(r: Result, row: Row, arm: []const u8) ?Cell {
        _ = r;
        for (row.cells) |c| if (std.mem.eql(u8, c.arm, arm)) return c;
        return null;
    }
};

pub fn toJson(a: std.mem.Allocator, r: Result) ![]u8 {
    return std.json.Stringify.valueAlloc(a, r, .{ .whitespace = .indent_2 });
}

pub fn fromJson(a: std.mem.Allocator, bytes: []const u8) !Result {
    const parsed = try std.json.parseFromSlice(Result, a, bytes, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    return parsed.value;
}

// ── The Run a bench body drives ─────────────────────────────────────────

const RowAcc = struct { label: []const u8, samples: []std.ArrayList(f64) };

pub const Run = struct {
    a: std.mem.Allocator,
    plan: Plan,
    rounds: usize,
    steps_override: ?usize,
    pack: ?[]const u8,
    cancel: *const std.atomic.Value(bool),
    after_sample: ?*const fn () void = null,
    rows: std.ArrayList(RowAcc) = .empty,
    metrics: std.ArrayList(Metric) = .empty,
    invalid_reason: ?[]const u8 = null,

    /// The values of an axis in force: the `--rungs` selection, else the axis's defaults.
    pub fn axis(run: *const Run, name: []const u8) []const []const u8 {
        for (run.plan.rungs) |s| if (std.mem.eql(u8, s.name, name)) return s.values;
        return run.plan.bench.axisNamed(name).?.values;
    }

    pub fn steps(run: *const Run, default: usize) usize {
        return run.steps_override orelse default;
    }

    pub fn opt(run: *const Run, key: []const u8) ?[]const u8 {
        for (run.plan.opts) |kv| if (std.mem.eql(u8, kv.key, key)) return kv.value;
        return null;
    }

    /// Whether an arm of this bench is in force.
    pub fn hasArm(run: *const Run, name: []const u8) bool {
        const idx = run.plan.bench.armIndex(name) orelse return false;
        return std.mem.indexOfScalar(usize, run.plan.arms, idx) != null;
    }

    /// One rung: every arm in force is sampled once untimed, then `rounds` times in the rotated order of `armOrder`.
    /// `f` returns the nanoseconds of one sample of arm `arm` (an index into `bench.arms`).
    pub fn interleave(run: *Run, label: []const u8, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize) anyerror!f64) !void {
        const n = run.plan.arms.len;
        const slots = try run.a.alloc(std.ArrayList(f64), run.plan.bench.arms.len);
        @memset(slots, .empty);
        for (run.plan.arms) |arm| _ = try f(ctx, arm);
        var order: [max_arms]usize = undefined;
        for (0..run.rounds) |round| {
            if (run.cancel.load(.acquire)) return error.Interrupted;
            armOrder(round, order[0..n]);
            for (order[0..n]) |k| {
                const arm = run.plan.arms[k];
                try slots[arm].append(run.a, try f(ctx, arm));
            }
            if (run.after_sample) |hook| hook();
        }
        try run.rows.append(run.a, .{ .label = label, .samples = slots });
    }

    /// Median ns of the latest rung's arm, for a metric computed from the timing.
    pub fn lastMedianNs(run: *const Run, arm: []const u8) f64 {
        const row = run.rows.items[run.rows.items.len - 1];
        return stats(row.samples[run.plan.bench.armIndex(arm).?].items).median;
    }

    /// A column beside the timings for the latest rung.
    pub fn metric(run: *Run, arm: []const u8, name: []const u8, unit: []const u8, value: f64) !void {
        const row = run.rows.items[run.rows.items.len - 1];
        try run.metrics.append(run.a, .{ .row = row.label, .arm = arm, .name = name, .unit = unit, .value = value });
    }

    /// The arms' outputs disagreed or a validity check failed: the result is kept and the run exits non-zero.
    pub fn invalid(run: *Run, reason: []const u8) void {
        if (run.invalid_reason == null) run.invalid_reason = reason;
    }

    pub fn finish(run: *const Run, stamps: Stamps) !Result {
        const b = run.plan.bench;
        var arms: std.ArrayList([]const u8) = .empty;
        for (run.plan.arms) |i| try arms.append(run.a, b.arms[i]);
        var rows: std.ArrayList(Row) = .empty;
        for (run.rows.items) |acc| {
            var cells: std.ArrayList(Cell) = .empty;
            for (run.plan.arms) |i| try cells.append(run.a, .{ .arm = b.arms[i], .ns = acc.samples[i].items });
            try rows.append(run.a, .{ .label = acc.label, .cells = cells.items });
        }
        var axes: std.ArrayList(Selection) = .empty;
        for (b.axes) |ax| try axes.append(run.a, .{ .name = ax.name, .values = run.axis(ax.name) });
        return .{
            .bench = b.name,
            .kind = @tagName(b.kind),
            .what = b.what,
            .unit = b.unit,
            .stamps = stamps,
            .rounds = run.rounds,
            .steps = run.steps(b.steps),
            .ref = b.arms[run.plan.ref],
            .arms = arms.items,
            .axes = axes.items,
            .opts = run.plan.opts,
            .rows = rows.items,
            .metrics = run.metrics.items,
            .validity = if (run.invalid_reason) |why| try std.fmt.allocPrint(run.a, "invalid:{s}", .{why}) else if (stamps.contended) "contended" else "valid",
        };
    }
};

// ── Tables ──────────────────────────────────────────────────────────────

fn writeGrid(w: *std.Io.Writer, a: std.mem.Allocator, header: []const []const u8, body: []const []const []const u8) !void {
    const widths = try a.alloc(usize, header.len);
    for (header, widths) |h, *wd| wd.* = h.len;
    for (body) |row| for (row, 0..) |c, i| {
        widths[i] = @max(widths[i], c.len);
    };
    try writeRow(w, header, widths);
    for (body) |row| try writeRow(w, row, widths);
}

fn writeRow(w: *std.Io.Writer, cells: []const []const u8, widths: []const usize) !void {
    for (cells, widths) |c, wd| {
        try w.writeAll("| ");
        try w.writeAll(c);
        try w.splatByteAll(' ', wd - c.len + 1);
    }
    try w.writeAll("|\n");
}

fn cellText(a: std.mem.Allocator, unit: Unit, ns: []const f64) ![]const u8 {
    const s = stats(ns);
    const d = unit.perNs();
    return std.fmt.allocPrint(a, "{d:.1} ({d:.1}-{d:.1})", .{ s.median / d, s.min / d, s.max / d });
}

fn writeDie(w: *std.Io.Writer, v: ?f64, suffix: []const u8) !void {
    if (v) |x| try w.print("{d:.0} C {s}", .{ x, suffix }) else try w.print("unread {s}", .{suffix});
}

pub fn renderTable(w: *std.Io.Writer, a: std.mem.Allocator, r: Result) !void {
    const st = r.stamps;
    try w.print("sushi-ubench {s}   {s}, {s}\n", .{ r.bench, r.kind, r.what });
    try w.print("commit {s} {s} | sushi-ubench sha256 {s} mtime {s} | {s} | {s}\n", .{ st.commit, if (st.dirty) "dirty" else "clean", st.sha256, st.mtime, st.optimize, st.chip });
    if (st.qos) |q| try w.print("pri {d}", .{q}) else try w.writeAll("pri unread");
    try w.print(" | {s} | ", .{st.power});
    try w.writeAll("die ");
    try writeDie(w, st.die_start_c, "start");
    try w.writeAll(", ");
    try writeDie(w, st.die_end_c, "end");
    try w.print(" | lock {s}{s}\n", .{ st.lock, if (st.contended) " | CONTENDED" else "" });
    try w.print("{d} rounds of {d} steps, arms rotated per round, ref = {s}\n", .{ r.rounds, r.steps, r.ref });
    try w.print("{s} per call, median (min-max)\n", .{@tagName(r.unit)});
    if (std.mem.startsWith(u8, r.validity, "invalid:")) try w.print("INVALID RUN: {s}\n", .{r.validity["invalid:".len..]});

    var header: std.ArrayList([]const u8) = .empty;
    try header.append(a, "rung");
    for (r.arms) |arm| try header.append(a, arm);
    for (r.arms) |arm| if (!std.mem.eql(u8, arm, r.ref)) try header.append(a, try std.fmt.allocPrint(a, "{s}/{s}", .{ arm, r.ref }));
    const metric_cols = try metricColumns(a, r);
    for (metric_cols) |m| try header.append(a, m.header);

    var body: std.ArrayList([]const []const u8) = .empty;
    for (r.rows) |row| {
        var cells: std.ArrayList([]const u8) = .empty;
        try cells.append(a, row.label);
        for (r.arms) |arm| try cells.append(a, if (r.cell(row, arm)) |c| try cellText(a, r.unit, c.ns) else "");
        const ref_ns = if (r.cell(row, r.ref)) |c| stats(c.ns).median else 0;
        for (r.arms) |arm| {
            if (std.mem.eql(u8, arm, r.ref)) continue;
            const c = r.cell(row, arm) orelse {
                try cells.append(a, "");
                continue;
            };
            try cells.append(a, if (ref_ns > 0) try std.fmt.allocPrint(a, "{d:.3}", .{stats(c.ns).median / ref_ns}) else "");
        }
        for (metric_cols) |m| {
            var text: []const u8 = "";
            for (r.metrics) |mm| {
                if (std.mem.eql(u8, mm.row, row.label) and std.mem.eql(u8, mm.arm, m.arm) and std.mem.eql(u8, mm.name, m.name))
                    text = if (@abs(mm.value) < 10) try std.fmt.allocPrint(a, "{d:.2}", .{mm.value}) else try std.fmt.allocPrint(a, "{d:.1}", .{mm.value});
            }
            try cells.append(a, text);
        }
        try body.append(a, cells.items);
    }
    try writeGrid(w, a, header.items, body.items);
}

const MetricCol = struct { arm: []const u8, name: []const u8, header: []const u8 };

fn metricColumns(a: std.mem.Allocator, r: Result) ![]const MetricCol {
    var cols: std.ArrayList(MetricCol) = .empty;
    for (r.metrics) |m| {
        const seen = for (cols.items) |c| {
            if (std.mem.eql(u8, c.arm, m.arm) and std.mem.eql(u8, c.name, m.name)) break true;
        } else false;
        if (seen) continue;
        const label = if (std.mem.eql(u8, m.name, m.unit)) m.name else try std.fmt.allocPrint(a, "{s} [{s}]", .{ m.name, m.unit });
        try cols.append(a, .{ .arm = m.arm, .name = m.name, .header = if (r.arms.len > 1) try std.fmt.allocPrint(a, "{s} {s}", .{ m.arm, label }) else label });
    }
    return cols.items;
}

// ── compare ─────────────────────────────────────────────────────────────

pub const CompareError = error{ DifferentBench, ContendedMix, OutOfMemory, WriteFailed };

fn rowNamed(r: Result, label: []const u8) ?Row {
    for (r.rows) |row| if (std.mem.eql(u8, row.label, label)) return row;
    return null;
}

pub fn compare(w: *std.Io.Writer, a: std.mem.Allocator, x: Result, y: Result) CompareError!void {
    if (!std.mem.eql(u8, x.bench, y.bench)) return error.DifferentBench;
    if (x.stamps.contended != y.stamps.contended) return error.ContendedMix;
    try w.print("sushi-ubench compare {s}, {s} per call, median; A {s} {s}, B {s} {s}\n", .{ x.bench, @tagName(x.unit), x.stamps.commit, x.stamps.time, y.stamps.commit, y.stamps.time });
    if (!std.mem.eql(u8, x.stamps.chip, y.stamps.chip)) try w.print("warning: chip differs: {s} vs {s}\n", .{ x.stamps.chip, y.stamps.chip });
    if (x.stamps.qos != y.stamps.qos) try w.writeAll("warning: QoS differs\n");
    if (x.stamps.dirty or y.stamps.dirty) try w.writeAll("warning: a result was built from a dirty tree\n");
    if (x.stamps.contended) try w.writeAll("warning: both results are CONTENDED\n");
    for ([_]Result{ x, y }, [_][]const u8{ "A", "B" }) |r, who| {
        if (r.stamps.die_end_c) |t| if (t > 50) try w.print("warning: {s} ended with the die at {d:.0} C\n", .{ who, t });
        if (std.mem.startsWith(u8, r.validity, "invalid:")) try w.print("warning: {s} is an invalid run ({s})\n", .{ who, r.validity });
    }
    if (x.steps != y.steps or x.rounds != y.rounds) try w.print("warning: rounds x steps differ: {d}x{d} vs {d}x{d}\n", .{ x.rounds, x.steps, y.rounds, y.steps });

    var dropped: std.ArrayList([]const u8) = .empty;
    var body: std.ArrayList([]const []const u8) = .empty;
    for (x.rows) |rx| {
        const ry = rowNamed(y, rx.label) orelse {
            try dropped.append(a, try std.fmt.allocPrint(a, "rung {s} (only in A)", .{rx.label}));
            continue;
        };
        for (rx.cells) |cx| {
            const cy = y.cell(ry, cx.arm) orelse {
                try dropped.append(a, try std.fmt.allocPrint(a, "{s} / {s} (only in A)", .{ rx.label, cx.arm }));
                continue;
            };
            const sx = stats(cx.ns);
            const sy = stats(cy.ns);
            const d = x.unit.perNs();
            const overlap = sx.min <= sy.max and sy.min <= sx.max;
            const cells = try a.dupe([]const u8, &.{
                rx.label,
                cx.arm,
                try std.fmt.allocPrint(a, "{d:.1}", .{sx.median / d}),
                try std.fmt.allocPrint(a, "{d:.1}", .{sy.median / d}),
                try std.fmt.allocPrint(a, "{d:.3}", .{sy.median / sx.median}),
                if (overlap) "~" else "",
            });
            try body.append(a, cells);
        }
    }
    for (y.rows) |ry| {
        if (rowNamed(x, ry.label) == null) try dropped.append(a, try std.fmt.allocPrint(a, "rung {s} (only in B)", .{ry.label}));
    }
    try writeGrid(w, a, &.{ "rung", "arm", "A", "B", "B/A", "~" }, body.items);
    try w.writeAll("~ marks overlapping min-max ranges: the noise call, not a verdict.\n");
    for (dropped.items) |d| try w.print("dropped: {s}\n", .{d});
}

// ── Time ────────────────────────────────────────────────────────────────

/// `YYYYmmddTHHMMSS` (UTC) for file names, or `YYYY-MM-DD HH:MM UTC` when `human`.
pub fn formatTime(a: std.mem.Allocator, secs: i64, human: bool) ![]u8 {
    const es: std.time.epoch.EpochSeconds = .{ .secs = @intCast(@max(secs, 0)) };
    const ymd = es.getEpochDay().calculateYearDay();
    const md = ymd.calculateMonthDay();
    const ds = es.getDaySeconds();
    return if (human)
        std.fmt.allocPrint(a, "{d:0>4}-{d:0>2}-{d:0>2} {d:0>2}:{d:0>2} UTC", .{ ymd.year, md.month.numeric(), md.day_index + 1, ds.getHoursIntoDay(), ds.getMinutesIntoHour() })
    else
        std.fmt.allocPrint(a, "{d:0>4}{d:0>2}{d:0>2}T{d:0>2}{d:0>2}{d:0>2}", .{ ymd.year, md.month.numeric(), md.day_index + 1, ds.getHoursIntoDay(), ds.getMinutesIntoHour(), ds.getSecondsIntoMinute() });
}

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;

fn fakeRun(_: *Run) anyerror!void {}

const fake_axes = [_]Axis{
    .{ .name = "rows", .values = &.{ "1", "4" } },
    .{ .name = "mode", .values = &.{ "uniform", "lowest" }, .numeric = false },
};
const fake_registry = [_]Bench{
    .{ .name = "dsv41.experts", .scope = .dsv41, .what = "chain", .axes = &fake_axes, .arms = &.{ "a", "b" }, .ref = "a", .opts = &.{.{ .name = "text", .help = "x" }}, .run = fakeRun },
    .{ .name = "dsv41.packed", .scope = .dsv41, .kind = .pack, .what = "p", .arms = &.{"only"}, .ref = "only", .run = fakeRun },
    .{ .name = "shared.other", .scope = .shared, .what = "o", .arms = &.{"x"}, .ref = "x", .run = fakeRun },
};

fn parseOne(a: std.mem.Allocator, argv: []const []const u8) !Command {
    var msg: []const u8 = "";
    return parse(a, argv, &msg);
}

test "ubench cli: every flag parses, and a bad one is refused with its reason" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ok = try parseOne(a, &.{ "run", "dsv41.experts", "--rungs", "rows=1,4", "--rungs", "mode=lowest", "--rounds", "3", "--steps", "8", "--arms", "a,b", "--ref", "b", "--pack", "/p", "--opt", "k=v", "--out", "/o", "--contended" });
    const ra = ok.run;
    try testing.expectEqualStrings("dsv41.experts", ra.target);
    try testing.expectEqual(@as(usize, 2), ra.rungs.len);
    try testing.expectEqualStrings("4", ra.rungs[0].values[1]);
    try testing.expectEqual(@as(?usize, 3), ra.rounds);
    try testing.expectEqual(@as(?usize, 8), ra.steps);
    try testing.expectEqual(@as(usize, 2), ra.arms.?.len);
    try testing.expectEqualStrings("b", ra.ref.?);
    try testing.expectEqualStrings("k", ra.opts[0].key);
    try testing.expect(ra.contended);
    try testing.expect((try parseOne(a, &.{ "run", "x", "--help" })) == .help);
    try testing.expectEqualStrings("shape", (try parseOne(a, &.{ "list", "shape" })).list.?);
    try testing.expectEqualStrings("b.json", (try parseOne(a, &.{ "compare", "a.json", "b.json" })).compare[1]);

    const refused = [_][]const []const u8{
        &.{},
        &.{"frobnicate"},
        &.{"run"},
        &.{ "run", "--rounds", "3" },
        &.{ "run", "x", "--rounds", "0" },
        &.{ "run", "x", "--rounds", "1001" },
        &.{ "run", "x", "--rounds", "many" },
        &.{ "run", "x", "--steps", "-1" },
        &.{ "run", "x", "--rounds" },
        &.{ "run", "x", "--rungs", "rows" },
        &.{ "run", "x", "--opt", "=v" },
        &.{ "run", "x", "--opt", "k=" },
        &.{ "run", "x", "--bogus" },
        &.{ "compare", "a.json" },
        &.{ "list", "a", "b" },
    };
    for (refused) |argv| try testing.expectError(error.BadArgs, parseOne(a, argv));
}

test "ubench select: names, scopes, axes, arms and opts are checked before anything runs" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var msg: []const u8 = "";
    const one = try select(a, &fake_registry, .{ .target = "dsv41.experts", .rungs = &.{.{ .name = "rows", .values = &.{"7"} }}, .arms = &.{ "b", "a" }, .ref = "b" }, &msg);
    try testing.expectEqual(@as(usize, 1), one.plans.len);
    try testing.expectEqual(@as(usize, 1), one.plans[0].ref);
    try testing.expectEqualSlices(usize, &.{ 1, 0 }, one.plans[0].arms);

    const group = try select(a, &fake_registry, .{ .target = "dsv41:all" }, &msg);
    try testing.expectEqual(@as(usize, 1), group.plans.len);
    try testing.expectEqualStrings("dsv41.packed", group.skipped[0]);
    const with_pack = try select(a, &fake_registry, .{ .target = "dsv41:all", .pack = "/p" }, &msg);
    try testing.expectEqual(@as(usize, 2), with_pack.plans.len);

    const refused = [_]RunArgs{
        .{ .target = "nope" },
        .{ .target = "bogus:all" },
        .{ .target = "enc:all" },
        .{ .target = "dsv41.packed" },
        .{ .target = "dsv41.experts", .arms = &.{"c"} },
        .{ .target = "dsv41.experts", .arms = &.{ "a", "a" } },
        .{ .target = "dsv41.experts", .ref = "c" },
        .{ .target = "dsv41.experts", .arms = &.{"b"} },
        .{ .target = "dsv41:all", .arms = &.{"a"} },
        .{ .target = "dsv41.experts", .rungs = &.{.{ .name = "ctx", .values = &.{"1"} }} },
        .{ .target = "dsv41.experts", .rungs = &.{.{ .name = "rows", .values = &.{"x"} }} },
        .{ .target = "dsv41.experts", .rungs = &.{.{ .name = "mode", .values = &.{"sideways"} }} },
        .{ .target = "dsv41.experts", .opts = &.{.{ .key = "graph", .value = "p" }} },
    };
    for (refused) |ra| try testing.expectError(error.BadArgs, select(a, &fake_registry, ra, &msg));
}

test "ubench stats: median, min and max, and the arm order of each round" {
    const odd = stats(&.{ 5, 1, 3 });
    try testing.expectEqual(@as(f64, 3), odd.median);
    try testing.expectEqual(@as(f64, 1), odd.min);
    try testing.expectEqual(@as(f64, 5), odd.max);
    try testing.expectEqual(@as(f64, 2.5), stats(&.{ 4, 1, 2, 3 }).median);
    try testing.expectEqual(@as(f64, 7), stats(&.{7}).median);
    const want = [_][3]usize{ .{ 0, 1, 2 }, .{ 0, 2, 1 }, .{ 2, 0, 1 }, .{ 2, 1, 0 }, .{ 1, 2, 0 } };
    for (want, 0..) |w, round| {
        var got: [3]usize = undefined;
        armOrder(round, &got);
        try testing.expectEqualSlices(usize, &w, &got);
    }
}

const SampleCtx = struct {
    seen: *std.ArrayList(usize),
    a: std.mem.Allocator,
};

fn sampleArm(c: SampleCtx, arm: usize) anyerror!f64 {
    try c.seen.append(c.a, arm);
    return @as(f64, @floatFromInt(arm + 1)) * 1000;
}

fn sampleResult(a: std.mem.Allocator, contended: bool) !Result {
    var cancel: std.atomic.Value(bool) = .init(false);
    var seen: std.ArrayList(usize) = .empty;
    var run: Run = .{ .a = a, .plan = .{ .bench = &fake_registry[0], .arms = &.{ 0, 1 }, .ref = 0, .rungs = &.{}, .opts = &.{} }, .rounds = 3, .steps_override = null, .pack = null, .cancel = &cancel };
    try run.interleave("rows 1", SampleCtx{ .seen = &seen, .a = a }, sampleArm);
    try run.metric("a", "MB read", "MB", 12.5);
    try run.metric("a", "GB/s", "GB/s", 400);
    const stamps: Stamps = .{ .commit = "abc1234", .dirty = false, .sha256 = "4b7e..19", .mtime = "2026-10-29 14:02 UTC", .chip = "Apple M5 Max", .qos = 31, .power = "AC", .die_start_c = 46, .lock = "ubench:dsv41.experts", .contended = contended, .time = "20261029T140200" };
    return run.finish(stamps);
}

test "ubench run: every arm is warmed once, then sampled in the rotated order each round" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cancel: std.atomic.Value(bool) = .init(false);
    var seen: std.ArrayList(usize) = .empty;
    var run: Run = .{ .a = a, .plan = .{ .bench = &fake_registry[0], .arms = &.{ 0, 1 }, .ref = 0, .rungs = &.{}, .opts = &.{} }, .rounds = 3, .steps_override = null, .pack = null, .cancel = &cancel };
    try run.interleave("rows 1", SampleCtx{ .seen = &seen, .a = a }, sampleArm);
    try testing.expectEqualSlices(usize, &.{ 0, 1, 0, 1, 0, 1, 0, 1 }, seen.items);
    try testing.expectEqual(@as(f64, 2000), run.lastMedianNs("b"));
    cancel.store(true, .release);
    try testing.expectError(error.Interrupted, run.interleave("rows 4", SampleCtx{ .seen = &seen, .a = a }, sampleArm));
    try testing.expectEqual(@as(usize, 1), run.rows.items.len);
}

test "ubench json: a result reads back as written, and the table carries ratios and metrics" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try sampleResult(a, true);
    const back = try fromJson(a, try toJson(a, r));
    try testing.expectEqualStrings(r.bench, back.bench);
    try testing.expectEqualStrings("contended", back.validity);
    try testing.expectEqual(@as(?i32, 31), back.stamps.qos);
    try testing.expectEqual(@as(?f64, null), back.stamps.die_end_c);
    try testing.expectEqual(@as(usize, 2), back.rows[0].cells.len);
    try testing.expectEqualSlices(f64, r.rows[0].cells[1].ns, back.rows[0].cells[1].ns);
    try testing.expectEqual(@as(f64, 12.5), back.metrics[0].value);

    var aw: std.Io.Writer.Allocating = .init(a);
    try renderTable(&aw.writer, a, back);
    const text = aw.written();
    try testing.expect(std.mem.indexOf(u8, text, "CONTENDED") != null);
    try testing.expect(std.mem.indexOf(u8, text, "| b/a ") != null);
    try testing.expect(std.mem.indexOf(u8, text, "2.000") != null);
    try testing.expect(std.mem.indexOf(u8, text, "MB read [MB]") != null);
    try testing.expect(std.mem.indexOf(u8, text, "a GB/s |") != null);
    try testing.expect(std.mem.indexOf(u8, text, "[GB/s]") == null);
    try testing.expect(std.mem.indexOf(u8, text, "1.0 (1.0-1.0)") != null);
}

test "ubench compare: the intersection, the overlap mark, and a refusal for another bench or a contended mix" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const x = try sampleResult(a, true);
    var y = x;
    y.rows = &.{
        .{ .label = "rows 1", .cells = &.{ .{ .arm = "a", .ns = &.{ 900, 1000, 1100 } }, .{ .arm = "b", .ns = &.{ 4000, 4100, 4200 } } } },
        .{ .label = "rows 4", .cells = &.{} },
    };
    var aw: std.Io.Writer.Allocating = .init(a);
    try compare(&aw.writer, a, x, y);
    const text = aw.written();
    try testing.expect(std.mem.indexOf(u8, text, "dropped: rung rows 4 (only in B)") != null);
    try testing.expect(std.mem.indexOf(u8, text, "2.050") != null);
    var lines = std.mem.splitScalar(u8, text, '\n');
    var a_line: ?[]const u8 = null;
    var b_line: ?[]const u8 = null;
    while (lines.next()) |l| {
        if (std.mem.startsWith(u8, l, "| rows 1 | a ")) a_line = l;
        if (std.mem.startsWith(u8, l, "| rows 1 | b ")) b_line = l;
    }
    try testing.expect(std.mem.indexOf(u8, a_line.?, "~") != null);
    try testing.expect(std.mem.indexOf(u8, b_line.?, "~") == null);

    var other = x;
    other.bench = "shared.other";
    try testing.expectError(error.DifferentBench, compare(&aw.writer, a, x, other));
    var quiet = x;
    quiet.stamps.contended = false;
    try testing.expectError(error.ContendedMix, compare(&aw.writer, a, x, quiet));
}

test "ubench time: the file stamp and the header stamp" {
    const a = testing.allocator;
    const compact = try formatTime(a, 1793282520, false);
    defer a.free(compact);
    try testing.expectEqualStrings("20261029T140200", compact);
    const human = try formatTime(a, 1793282520, true);
    defer a.free(human);
    try testing.expectEqualStrings("2026-10-29 14:02 UTC", human);
}
