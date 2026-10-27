//! `sushi bench`: the release bench client. It drives a server somebody else booted (tests/bench.sh does the
//! hygiene and the boot) with two fixed token-id corpora, a code arm and a prose arm, one after the other: each is
//! a 4k cold request, then 16 windows of 8192 tokens, each restoring the previous request's prompt-end point. Docs: process-measurement.md#release-bench.

const std = @import("std");
const io_util = @import("io_util.zig");
const log = @import("log.zig");
const gpu_clock = @import("gpu_clock.zig");
const testing = std.testing;

pub const window: usize = 8192;
pub const first_rung: usize = 4096;
pub const ladder_steps: usize = 16;
pub const samples_per_rung: usize = 3;
pub const corpus_seed: u64 = 0x5a5a_c0de;
/// Pride and Prejudice, public domain (Project Gutenberg #1342) with the Project Gutenberg header and footer removed.
const prose_text = @embedFile("fixtures/prose_pride_and_prejudice.txt");
/// The ladder must reach 128k tokens, plus a margin so the last window is never short.
const corpus_tokens: usize = 132_000;
/// A request counts as restored when its cached tokens fall this close below the point it should restore
/// (the prompt-end checkpoint trails the prompt end by up to 34 tokens).
const restore_slack: usize = 64;
const rungs = [_]usize{ 4096, 8192, 16384, 32768, 65536, 131072 };

pub const USAGE =
    \\Usage: sushi bench --out <dir> --log <server.log> [options]
    \\
    \\Drives a running server through the release bench, once on a code corpus and once on prose: a cold 4k
    \\request, 16 windows of 8192 tokens each restoring the previous prompt end, and 3 decode samples at
    \\4k, 8k, 16k, 32k, 64k and 128k. It reads the GPU clock and thermal pressure through `sudo -n powermetrics`
    \\beside every request and stops with an error when the busy clock is throttled (at most 1100 MHz); the
    \\pressure level is recorded only. Without a passwordless sudo rule it says so and runs unmonitored.
    \\Boot the server and take the box rules with tests/bench.sh.
    \\
    \\Options:
    \\  --url <host:port>      Server to drive (default 127.0.0.1:8080)
    \\  --model <id>           Model id to send (default: the first id of /v1/models)
    \\  --out <dir>            Directory for bench.json, the raw samples (required)
    \\  --log <path>           The server's log, for the speculative mode and the restore time (required)
    \\  --report <path>        Write the docs/bench/v<ver>/<model>.md report here
    \\  --version <ver>        Release version for the report title
    \\  --stamp <text>         Header lines for the report (commit, binary, hardware); \n separates lines
    \\  --decode-tokens <n>    Tokens decoded per code-arm sample (default 256; the prose arm decodes at most 128)
    \\  --help                 Show this help
    \\
;

pub const Options = struct {
    help: bool = false,
    url: []const u8 = "127.0.0.1:8080",
    model: []const u8 = "",
    out_dir: []const u8 = "",
    log_path: []const u8 = "",
    report: []const u8 = "",
    version: []const u8 = "",
    stamp: []const u8 = "",
    decode_tokens: u32 = 256,
};

pub const ParseError = error{ UnknownArgument, MissingValue, BadValue, MissingRequired };

/// Every flag that takes a value, for the parser and the help-lists-every-flag test.
const value_flags = [_][]const u8{ "--url", "--model", "--out", "--log", "--report", "--version", "--stamp", "--decode-tokens" };

pub fn parseArgs(args: []const []const u8) ParseError!Options {
    var o: Options = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) {
            o.help = true;
            continue;
        }
        const idx = for (value_flags, 0..) |f, n| {
            if (std.mem.eql(u8, a, f)) break n;
        } else return error.UnknownArgument;
        i += 1;
        if (i >= args.len) return error.MissingValue;
        const v = args[i];
        switch (idx) {
            0 => {
                if (std.mem.indexOfScalar(u8, v, ':') == null or std.mem.indexOfScalar(u8, v, '/') != null) return error.BadValue;
                o.url = v;
            },
            1 => o.model = v,
            2 => o.out_dir = v,
            3 => o.log_path = v,
            4 => o.report = v,
            5 => o.version = v,
            6 => o.stamp = v,
            7 => {
                o.decode_tokens = std.fmt.parseInt(u32, v, 10) catch return error.BadValue;
                if (o.decode_tokens < 2) return error.BadValue;
            },
            else => unreachable,
        }
    }
    if (!o.help and (o.out_dir.len == 0 or o.log_path.len == 0)) return error.MissingRequired;
    return o;
}

// ── The corpus ──────────────────────────────────────────────────────────

const Rng = struct {
    s: u64,

    fn next(self: *Rng) u64 {
        self.s +%= 0x9e3779b97f4a7c15;
        var z = self.s;
        z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }

    fn below(self: *Rng, n: usize) usize {
        return @intCast(self.next() % n);
    }

    fn pick(self: *Rng, comptime list: []const []const u8) []const u8 {
        return list[self.below(list.len)];
    }
};

const nouns = [_][]const u8{ "order", "invoice", "session", "token", "cart", "user", "ledger", "queue", "cache", "route", "tenant", "webhook", "report", "schedule", "policy", "metric", "stream", "batch", "index", "profile" };
const verbs = [_][]const u8{ "load", "save", "parse", "merge", "drop", "fetch", "retry", "emit", "build", "verify", "resolve", "flush", "rotate", "sync", "encode" };
const ts_types = [_][]const u8{ "string", "number", "boolean", "Date", "Uint8Array", "Record<string, unknown>", "string[]", "number | null" };

fn cap(buf: []u8, word: []const u8) []const u8 {
    @memcpy(buf[0..word.len], word);
    buf[0] = std.ascii.toUpper(buf[0]);
    return buf[0..word.len];
}

/// A seeded stream of synthetic TypeScript modules in an agent session's style: imports, interfaces,
/// a service class, helpers. The same seed gives the same bytes on every box.
pub const Corpus = struct {
    rng: Rng,
    module: usize = 0,

    pub fn init(seed: u64) Corpus {
        return .{ .rng = .{ .s = seed } };
    }

    /// Whole modules up to about 24 KB; the caller frees.
    pub fn nextChunk(self: *Corpus, allocator: std.mem.Allocator) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();
        while (out.written().len < 24 * 1024) {
            try self.writeModule(&out.writer);
            try out.writer.writeAll("\n\n");
        }
        return out.toOwnedSlice();
    }

    fn writeModule(self: *Corpus, w: *std.Io.Writer) !void {
        const r = &self.rng;
        const n = self.module;
        self.module += 1;
        const noun = r.pick(&nouns);
        const other = r.pick(&nouns);
        var nb: [32]u8 = undefined;
        var ob: [32]u8 = undefined;
        const Noun = cap(&nb, noun);
        const Other = cap(&ob, other);
        try w.print("// src/{s}/{s}-{d}.ts\n", .{ noun, noun, n });
        try w.print("import {{ {s}Record{d}, {s}Options }} from \"../{s}/{s}-{d}\";\n", .{ Other, r.below(900), Other, other, other, r.below(900) });
        try w.writeAll("import type { Logger } from \"../core/logger\";\n\n");
        try w.print("export interface {s}Record{d} {{\n  id: string;\n", .{ Noun, n });
        for (0..3 + r.below(4)) |_| try w.print("  {s}{s}: {s};\n", .{ r.pick(&verbs), r.pick(&nouns), r.pick(&ts_types) });
        try w.writeAll("}\n\n");
        try w.print("export class {s}Service{d} {{\n  private readonly cache = new Map<string, {s}Record{d}>();\n", .{ Noun, n, Noun, n });
        try w.print("  constructor(private readonly log: Logger, private readonly limit = {d}) {{}}\n", .{ 16 + r.below(4096) });
        for (0..3 + r.below(3)) |_| {
            const verb = r.pick(&verbs);
            var vb: [32]u8 = undefined;
            const Verb = cap(&vb, verb);
            try w.print("\n  async {s}{s}(id: string, attempt = {d}): Promise<{s}Record{d} | undefined> {{\n", .{ verb, Noun, r.below(5), Noun, n });
            try w.print("    const hit = this.cache.get(id);\n    if (hit && attempt < {d}) return hit;\n", .{ 1 + r.below(9) });
            try w.print("    try {{\n      const next = await {s}{s}(id, {{ timeoutMs: {d} }});\n", .{ verb, Other, 50 * (1 + r.below(200)) });
            try w.print("      this.cache.set(id, next);\n      this.log.info(`{s}{s} ${{id}} size=${{this.cache.size}}`);\n      return next;\n", .{ Verb, Noun });
            try w.print("    }} catch (err) {{\n      if (attempt >= {d}) throw new Error(`{s} failed for ${{id}}: ${{String(err)}}`);\n", .{ 2 + r.below(6), verb });
            try w.print("      return this.{s}{s}(id, attempt + 1);\n    }}\n  }}\n", .{ verb, Noun });
        }
        try w.writeAll("}\n\n");
        try w.print("export function {s}Key{d}(parts: string[]): string {{\n  return parts.map((p) => p.trim().toLowerCase()).filter(Boolean).join(\"{s}\");\n}}", .{ noun, n, r.pick(&.{ ":", "/", "-", "|" }) });
    }
};

/// The text of the prose arm in pieces of at most about 24 KB, each ending at a blank line, so the server tokenizes
/// the same bytes on every box.
pub const ProseChunks = struct {
    text: []const u8,
    pos: usize = 0,

    pub fn next(self: *ProseChunks) ?[]const u8 {
        if (self.pos >= self.text.len) return null;
        const limit = 24 * 1024;
        var end = @min(self.pos + limit, self.text.len);
        if (end < self.text.len) {
            if (std.mem.lastIndexOf(u8, self.text[self.pos..end], "\n\n")) |at| end = self.pos + at + 2;
        }
        const piece = self.text[self.pos..end];
        self.pos = end;
        return piece;
    }
};

pub const Arm = enum { code, prose };

// ── The plan ────────────────────────────────────────────────────────────

pub const Kind = enum { ladder, repeat };

pub const Request = struct {
    prompt_len: usize,
    max_tokens: u32,
    kind: Kind,
    /// The prompt length whose end point this request should restore; 0 = cold.
    base: usize,
};

fn isRung(len: usize) bool {
    return std.mem.indexOfScalar(usize, &rungs, len) != null;
}

/// The 4k cold request, then every 8192-token window; a rung is followed by its two repeats at once so
/// a checkpoint cap cannot thin the point a repeat restores.
/// The prose arm decodes half the code arm's tokens a sample: its rounds are the slowest of the run.
pub const prose_decode_tokens: u32 = 128;

pub fn armDecodeTokens(arm: Arm, decode_tokens: u32) u32 {
    return if (arm == .prose) @min(decode_tokens, prose_decode_tokens) else decode_tokens;
}

pub fn plan(allocator: std.mem.Allocator, decode_tokens: u32) ![]Request {
    var list: std.ArrayList(Request) = .empty;
    errdefer list.deinit(allocator);
    var prev: usize = 0;
    var len: usize = first_rung;
    var step: usize = 0;
    while (true) : (step += 1) {
        const decode = isRung(len);
        try list.append(allocator, .{ .prompt_len = len, .max_tokens = if (decode) decode_tokens else 1, .kind = .ladder, .base = prev });
        if (decode) for (1..samples_per_rung) |_| try list.append(allocator, .{ .prompt_len = len, .max_tokens = decode_tokens, .kind = .repeat, .base = len });
        prev = len;
        if (step >= ladder_steps) break;
        len = window * (step + 1);
    }
    return list.toOwnedSlice(allocator);
}

// ── Samples ─────────────────────────────────────────────────────────────

pub const Sample = struct {
    arm: Arm = .code,
    prompt_len: usize,
    kind: Kind,
    base: usize,
    max_tokens: u32,
    prompt_n: u32,
    cached_n: u32,
    completion_n: u32,
    prompt_ms: f64,
    predicted_ms: f64,
    wall_ms: f64,
    restore_ms: f64,
    restore_logged: bool,
    mode: []const u8,
    /// Median active GPU clock of the request's busy samples (0 = none) and its worst thermal pressure (null = not monitored).
    gpu_mhz: f64 = 0,
    gpu_pressure: ?gpu_clock.Pressure = null,
    /// The server ended the request through its repetition-loop stop, so fewer tokens than asked is expected.
    loop_stop: bool = false,

    pub fn newTokens(s: Sample) u32 {
        return s.prompt_n -| s.cached_n;
    }

    /// The server's prefill time includes the restore; the restore is logged apart and never counted.
    pub fn prefillMs(s: Sample) f64 {
        return @max(s.prompt_ms - s.restore_ms, 0);
    }

    pub fn prefillTps(s: Sample) f64 {
        const ms = s.prefillMs();
        return if (ms > 0) @as(f64, @floatFromInt(s.newTokens())) * 1000.0 / ms else 0;
    }

    pub fn decodeTps(s: Sample) f64 {
        return if (s.predicted_ms > 0) @as(f64, @floatFromInt(s.completion_n)) * 1000.0 / s.predicted_ms else 0;
    }

    /// Request wall time up to the first token: everything but the decode after it.
    pub fn ttftMs(s: Sample) f64 {
        return @max(s.wall_ms - s.predicted_ms, 0);
    }
};

pub const Problem = enum { restore_missed, no_restore_line, short_decode, gpu_throttled };

pub fn check(s: Sample) ?Problem {
    if (gpu_clock.verdict(s.gpu_mhz, s.gpu_pressure) != .ok) return .gpu_throttled;
    if (s.base == 0) {
        if (s.cached_n != 0) return .restore_missed;
    } else if (s.cached_n + restore_slack < s.base or s.cached_n > s.base) return .restore_missed;
    if (!s.restore_logged) return .no_restore_line;
    if (s.max_tokens > 1 and s.completion_n < s.max_tokens and !s.loop_stop) return .short_decode;
    return null;
}

pub fn median(vals: []const f64) f64 {
    var buf: [16]f64 = undefined;
    const n = @min(vals.len, buf.len);
    if (n == 0) return 0;
    @memcpy(buf[0..n], vals[0..n]);
    std.mem.sort(f64, buf[0..n], {}, std.sort.asc(f64));
    return if (n % 2 == 1) buf[n / 2] else (buf[n / 2 - 1] + buf[n / 2]) / 2;
}

// ── Wire formats ────────────────────────────────────────────────────────

pub fn promptBody(allocator: std.mem.Allocator, model: []const u8, ids: []const u32, max_tokens: u32) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeAll("{");
    if (model.len > 0) {
        try w.writeAll("\"model\":");
        try std.json.Stringify.value(model, .{}, w);
        try w.writeAll(",");
    }
    try w.print("\"max_tokens\":{d},\"temperature\":0,\"ignore_eos\":true,\"stream\":false,\"prompt\":[", .{max_tokens});
    for (ids, 0..) |id, i| {
        if (i > 0) try w.writeAll(",");
        try w.print("{d}", .{id});
    }
    try w.writeAll("]}");
    return out.toOwnedSlice();
}

pub const Completion = struct {
    prompt_n: u32,
    cached_n: u32,
    completion_n: u32,
    prompt_ms: f64,
    predicted_ms: f64,
    loop_stop: bool = false,
};

fn jsonNum(v: ?std.json.Value) ?f64 {
    const x = v orelse return null;
    return switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

/// `choices[0].finish_details.type == "repetition_loop"`, the shape server.zig's `finishDetailsField` emits.
fn loopStopped(root: std.json.ObjectMap) bool {
    const choices = root.get("choices") orelse return false;
    if (choices != .array or choices.array.items.len == 0 or choices.array.items[0] != .object) return false;
    const d = choices.array.items[0].object.get("finish_details") orelse return false;
    if (d != .object) return false;
    const ty = d.object.get("type") orelse return false;
    return ty == .string and std.mem.eql(u8, ty.string, "repetition_loop");
}

pub fn parseCompletion(allocator: std.mem.Allocator, body: []const u8) !Completion {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadCompletion;
    const t = parsed.value.object.get("timings") orelse return error.NoTimings;
    if (t != .object) return error.NoTimings;
    const o = t.object;
    return .{
        .loop_stop = loopStopped(parsed.value.object),
        .prompt_n = @intFromFloat(jsonNum(o.get("prompt_n")) orelse return error.NoTimings),
        .cached_n = @intFromFloat(jsonNum(o.get("cached_n")) orelse return error.NoTimings),
        .completion_n = @intFromFloat(jsonNum(o.get("predicted_n")) orelse return error.NoTimings),
        .prompt_ms = jsonNum(o.get("prompt_ms")) orelse return error.NoTimings,
        .predicted_ms = jsonNum(o.get("predicted_ms")) orelse return error.NoTimings,
    };
}

pub const Restore = struct { matched: usize, total: usize, ms: f64 };

/// The last `... reused N of M prompt tokens from T, prefilling K; lookup and restore X ms` line of a log segment.
pub fn parseRestore(segment: []const u8) ?Restore {
    const key = " reused ";
    const at = std.mem.lastIndexOf(u8, segment, key) orelse return null;
    const line_end = std.mem.indexOfScalarPos(u8, segment, at, '\n') orelse segment.len;
    const line = segment[at + key.len .. line_end];
    var it = std.mem.tokenizeAny(u8, line, " ,;");
    const matched = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
    if (!std.mem.eql(u8, it.next() orelse return null, "of")) return null;
    const total = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
    const marker = "lookup and restore ";
    const m = std.mem.indexOf(u8, line, marker) orelse return null;
    var ms_it = std.mem.tokenizeScalar(u8, line[m + marker.len ..], ' ');
    const ms = std.fmt.parseFloat(f64, ms_it.next() orelse return null) catch return null;
    return .{ .matched = matched, .total = total, .ms = ms };
}

/// The speculative mode of the last `[spec-stats] mode=<m>` line of a log segment.
pub fn parseSpecMode(segment: []const u8) ?[]const u8 {
    const key = "[spec-stats] mode=";
    const at = std.mem.lastIndexOf(u8, segment, key) orelse return null;
    const rest = segment[at + key.len ..];
    var n: usize = 0;
    while (n < rest.len and std.ascii.isAlphanumeric(rest[n])) n += 1;
    return if (n == 0) null else rest[0..n];
}

// ── The report ──────────────────────────────────────────────────────────

pub const Report = struct {
    model: []const u8,
    version: []const u8,
    stamp: []const u8,
    decode_tokens: u32,
    samples: []const Sample,
    clock: gpu_clock.RunSummary = .{},
    /// Why the run stopped early; empty for a complete run.
    aborted: []const u8 = "",
    cooldown: ?Cooldown = null,
};

/// The target and the cap of the idle between the arms; the same rule `tests/bench.sh` applies before a boot.
pub const cooldown_target_c: f64 = 50;
pub const cooldown_cap_s: u32 = 180;
const cooldown_poll_s: u32 = 10;

pub const Cooldown = struct {
    waited_s: u32,
    start_c: ?f64 = null,
    end_c: ?f64 = null,
};

/// No reading means the fixed wait: only the cap ends it.
pub fn cooldownDone(die_c: ?f64, waited_s: u32) bool {
    if (waited_s >= cooldown_cap_s) return true;
    return if (die_c) |t| t < cooldown_target_c else false;
}

/// The hottest `topTemps` sensor of a `fanctl status` document; null when it holds none or is not that document.
pub fn hottestDie(allocator: std.mem.Allocator, status: []const u8) ?f64 {
    const parsed = std.json.parseFromSlice(struct { topTemps: []const struct { celsius: f64 } }, allocator, status, .{ .ignore_unknown_fields = true }) catch return null;
    defer parsed.deinit();
    var hottest: ?f64 = null;
    for (parsed.value.topTemps) |t| hottest = if (hottest) |h| @max(h, t.celsius) else t.celsius;
    return hottest;
}

fn fanctlRun(a: std.mem.Allocator, io: std.Io, argv: []const []const u8) ?[]u8 {
    const r = std.process.run(a, io, .{ .argv = argv, .stdout_limit = .limited(1 << 20) }) catch return null;
    a.free(r.stderr);
    const ok = switch (r.term) {
        .exited => |code| code == 0,
        else => false,
    };
    if (ok) return r.stdout;
    a.free(r.stdout);
    return null;
}

fn readDie(a: std.mem.Allocator, io: std.Io) ?f64 {
    const out = fanctlRun(a, io, &.{ "fanctl", "status" }) orelse return null;
    defer a.free(out);
    return hottestDie(a, out);
}

/// Fans to max, then idles until the hottest die is under the target or the cap passes; the server stays up.
fn cooldownBetweenArms(a: std.mem.Allocator, io: std.Io) Cooldown {
    if (fanctlRun(a, io, &.{ "fanctl", "max", "--ttl", "900" })) |out| a.free(out);
    var clock = io_util.Stopwatch.init(io);
    const start_c = readDie(a, io);
    var die = start_c;
    while (true) {
        const waited: u32 = @intCast(clock.read() / std.time.ns_per_s);
        if (cooldownDone(die, waited)) break;
        std.Io.sleep(io, .fromSeconds(cooldown_poll_s), .real) catch {};
        die = readDie(a, io);
    }
    const waited: u32 = @intCast(clock.read() / std.time.ns_per_s);
    if (start_c) |s| {
        log.info("[bench] arms cooldown: {d:.0} C -> {d:.0} C in {d} s\n", .{ s, die orelse s, waited });
    } else {
        log.info("[bench] arms cooldown: {d} s (no fanctl)\n", .{waited});
    }
    return .{ .waited_s = waited, .start_c = start_c, .end_c = die };
}

fn writeThousands(w: *std.Io.Writer, v: f64) !void {
    const n: u64 = @intFromFloat(@round(@max(v, 0)));
    var buf: [24]u8 = undefined;
    const digits = std.fmt.bufPrint(&buf, "{d}", .{n}) catch unreachable;
    for (digits, 0..) |c, i| {
        if (i > 0 and (digits.len - i) % 3 == 0) try w.writeAll(",");
        try w.writeByte(c);
    }
}

fn writeRungLabel(w: *std.Io.Writer, len: usize) !void {
    try w.print("{d}k", .{len / 1024});
}

const RungRow = struct {
    len: usize,
    n: usize = 0,
    tps: [samples_per_rung]f64 = @splat(0),
    cached: [samples_per_rung]u32 = @splat(0),
    modes: [samples_per_rung][]const u8 = @splat(""),
    mhz: [samples_per_rung]f64 = @splat(0),
    pressure: [samples_per_rung]?gpu_clock.Pressure = @splat(null),
};

/// The worst of the pressure levels, null when none was read.
fn worstPressure(levels: []const ?gpu_clock.Pressure) ?gpu_clock.Pressure {
    var worst: ?gpu_clock.Pressure = null;
    for (levels) |l| {
        const p = l orelse continue;
        if (worst == null or @intFromEnum(p) > @intFromEnum(worst.?)) worst = p;
    }
    return worst;
}

fn writeGpuCells(w: *std.Io.Writer, mhz: f64, pressure: ?gpu_clock.Pressure) !void {
    if (mhz > 0) try w.print(" {d:.0} |", .{mhz}) else try w.writeAll(" · |");
    if (pressure) |p| try w.print(" {s} |", .{gpu_clock.pressureLabel(p)}) else try w.writeAll(" · |");
}

fn armHasGpu(samples: []const Sample, arm: Arm) bool {
    for (samples) |s| if (s.arm == arm and s.gpu_pressure != null) return true;
    return false;
}

fn modeLabel(row: RungRow) []const u8 {
    const first = if (row.modes[0].len == 0) "none" else row.modes[0];
    for (row.modes[1..row.n]) |m| {
        const mm = if (m.len == 0) "none" else m;
        if (!std.mem.eql(u8, mm, first)) return "mixed";
    }
    return first;
}

fn rungRows(allocator: std.mem.Allocator, samples: []const Sample, arm: Arm) !std.ArrayList(RungRow) {
    var rows: std.ArrayList(RungRow) = .empty;
    errdefer rows.deinit(allocator);
    for (samples) |s| {
        if (s.arm != arm or s.max_tokens <= 1) continue;
        if (rows.items.len == 0 or rows.items[rows.items.len - 1].len != s.prompt_len) try rows.append(allocator, .{ .len = s.prompt_len });
        const row = &rows.items[rows.items.len - 1];
        if (row.n >= samples_per_rung) continue;
        row.tps[row.n] = s.decodeTps();
        row.cached[row.n] = s.cached_n;
        row.modes[row.n] = s.mode;
        row.mhz[row.n] = s.gpu_mhz;
        row.pressure[row.n] = s.gpu_pressure;
        row.n += 1;
    }
    return rows;
}

fn writeDecodeCells(w: *std.Io.Writer, row: ?RungRow, gpu: bool) !void {
    const r = row orelse {
        try w.writeAll(" · | · | · | · | · |");
        if (gpu) try w.writeAll(" · | · |");
        return;
    };
    for (0..samples_per_rung) |i| {
        if (i < r.n) try w.print(" {d:.1} |", .{r.tps[i]}) else try w.writeAll(" · |");
    }
    try w.print(" {d:.1} | {s} |", .{ median(r.tps[0..r.n]), modeLabel(r) });
    if (gpu) {
        var seen: [samples_per_rung]f64 = undefined;
        var n: usize = 0;
        for (r.mhz[0..r.n]) |m| if (m > 0) {
            seen[n] = m;
            n += 1;
        };
        try writeGpuCells(w, median(seen[0..n]), worstPressure(r.pressure[0..r.n]));
    }
}

fn findRow(rows: []const RungRow, len: usize) ?RungRow {
    for (rows) |r| if (r.len == len) return r;
    return null;
}

fn writeWindows(w: *std.Io.Writer, samples: []const Sample, arm: Arm) !void {
    const gpu = armHasGpu(samples, arm);
    try w.print("\n## Prefill by window, {s} (tok/s)\n\n| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s |{s}\n|---|---:|---:|---:|---:|---:|{s}\n", .{ @tagName(arm), if (gpu) " GPU MHz | Pressure |" else "", if (gpu) "---:|---|" else "" });
    for (samples) |s| {
        if (s.arm != arm or s.kind != .ladder) continue;
        try w.writeAll("| ");
        try writeThousands(w, @floatFromInt(s.cached_n));
        try w.writeAll(" to ");
        try writeThousands(w, @floatFromInt(s.prompt_n));
        try w.writeAll(" | ");
        try writeThousands(w, @floatFromInt(s.cached_n));
        try w.writeAll(" | ");
        try writeThousands(w, @floatFromInt(s.newTokens()));
        try w.writeAll(" | ");
        try writeThousands(w, s.prefillTps());
        try w.print(" | {d:.0} | {d:.2} |", .{ s.restore_ms, s.ttftMs() / 1000.0 });
        if (gpu) try writeGpuCells(w, s.gpu_mhz, s.gpu_pressure);
        try w.writeAll("\n");
    }
}

fn writeSummaryRows(w: *std.Io.Writer, name: []const u8, rows: []const RungRow, samples: []const Sample, arm: Arm) !void {
    try w.print("| **{s}** | Prefill |", .{name});
    for (rows) |row| {
        try w.writeAll(" ");
        for (samples) |s| if (s.arm == arm and s.kind == .ladder and s.prompt_len == row.len) try writeThousands(w, s.prefillTps());
        try w.writeAll(" |");
    }
    try w.writeAll("\n|  | Decode |");
    for (rows) |row| try w.print(" {d:.1} |", .{median(row.tps[0..row.n])});
    try w.writeAll("\n");
}

pub fn render(allocator: std.mem.Allocator, rep: Report) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const w = &out.writer;

    var code = try rungRows(allocator, rep.samples, .code);
    defer code.deinit(allocator);
    var prose = try rungRows(allocator, rep.samples, .prose);
    defer prose.deinit(allocator);
    const has_prose = prose.items.len > 0;

    try w.print("# {s} bench", .{rep.model});
    if (rep.version.len > 0) try w.print(" (Sushi {s})", .{rep.version});
    try w.writeAll("\n\n");
    var lines = std.mem.splitSequence(u8, rep.stamp, "\\n");
    while (lines.next()) |l| if (l.len > 0) try w.print("{s}\n\n", .{l});
    if (rep.aborted.len > 0) try w.print("**Aborted, not a measurement:** {s}\n\n", .{rep.aborted});
    try gpu_clock.writeRunHeader(w, rep.clock);
    try w.print(
        "Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, {d} decoded tokens a sample ({d} on prose). The code arm is one fixed seeded TypeScript corpus; " ++
            "the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. " ++
            "Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. " ++
            "The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on.",
        .{ rep.decode_tokens, armDecodeTokens(.prose, rep.decode_tokens) },
    );
    if (rep.cooldown) |c| {
        try w.print(" The server idled {d} s between the arms (", .{c.waited_s});
        if (c.start_c) |s| try w.print("die {d:.0} C to {d:.0} C", .{ s, c.end_c orelse s }) else try w.writeAll("die not read");
        try w.writeAll(").");
    }
    try w.writeAll("\n\n");

    const code_gpu = armHasGpu(rep.samples, .code);
    const prose_gpu = armHasGpu(rep.samples, .prose);
    try w.writeAll("## Decode (tok/s)\n\n| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode |");
    if (code_gpu) try w.writeAll(" Code MHz | Code pressure |");
    if (has_prose) try w.writeAll(" Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode |");
    if (prose_gpu) try w.writeAll(" Prose MHz | Prose pressure |");
    if (has_prose) try w.writeAll(" Prose / code |");
    try w.writeAll("\n|---|---:|---:|---:|---:|---|");
    if (code_gpu) try w.writeAll("---:|---|");
    if (has_prose) try w.writeAll("---:|---:|---:|---:|---|");
    if (prose_gpu) try w.writeAll("---:|---|");
    if (has_prose) try w.writeAll("---:|");
    try w.writeAll("\n");
    for (code.items) |row| {
        try w.writeAll("| ");
        try writeRungLabel(w, row.len);
        try w.writeAll(" |");
        try writeDecodeCells(w, row, code_gpu);
        if (has_prose) {
            const p = findRow(prose.items, row.len);
            try writeDecodeCells(w, p, prose_gpu);
            const cm = median(row.tps[0..row.n]);
            if (p != null and cm > 0) try w.print(" {d:.2} |", .{median(p.?.tps[0..p.?.n]) / cm}) else try w.writeAll(" · |");
        }
        try w.writeAll("\n");
    }

    try writeWindows(w, rep.samples, .code);
    if (has_prose) try writeWindows(w, rep.samples, .prose);

    try w.writeAll("\n## Summary rows\n\nPrefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.\n\n| Sushi |  |");
    for (code.items) |row| {
        try w.writeAll(" ");
        try writeRungLabel(w, row.len);
        try w.writeAll(" |");
    }
    try w.writeAll("\n|---|---|");
    for (code.items) |_| try w.writeAll("---:|");
    try w.writeAll("\n");
    try writeSummaryRows(w, rep.model, code.items, rep.samples, .code);
    if (has_prose) {
        const pname = try std.fmt.allocPrint(allocator, "{s} (prose)", .{rep.model});
        defer allocator.free(pname);
        try writeSummaryRows(w, pname, prose.items, rep.samples, .prose);
    }

    var any = false;
    for (rep.samples) |s| {
        const p = check(s) orelse continue;
        if (!any) try w.writeAll("\n## Warnings\n\nA row with a warning is not a valid measurement.\n\n");
        any = true;
        try w.print("- {s}, {d} tokens ({s}): {s}\n", .{ @tagName(s.arm), s.prompt_len, @tagName(s.kind), switch (p) {
            .restore_missed => "restored fewer tokens than the previous prompt end holds",
            .no_restore_line => "no restore line in the server log, so the restore time was not taken out",
            .short_decode => "decoded fewer tokens than asked",
            .gpu_throttled => "the GPU clock or thermal pressure failed the abort rule (see the GPU columns)",
        } });
    }

    var noted = false;
    for (rep.samples) |s| {
        if (!s.loop_stop or s.completion_n >= s.max_tokens) continue;
        if (!noted) try w.writeAll("\n## Notes\n\nThe server's loop stop cut a greedy repetition loop. The sample counts: its tok/s is a real decode.\n\n");
        noted = true;
        try w.print("- {s}, {d} tokens ({s}): the loop stop ended it at {d} tokens\n", .{ @tagName(s.arm), s.prompt_len, @tagName(s.kind), s.completion_n });
    }
    return out.toOwnedSlice();
}

// ── The run ─────────────────────────────────────────────────────────────

fn post(allocator: std.mem.Allocator, client: *std.http.Client, url: []const u8, body: []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const res = try client.fetch(.{
        .location = .{ .url = url },
        .method = .POST,
        .payload = body,
        .headers = .{ .content_type = .{ .override = "application/json" } },
        .keep_alive = false,
        .response_writer = &out.writer,
    });
    if (res.status != .ok) {
        log.err("sushi bench: {s} answered HTTP {d}: {s}\n", .{ url, @backingInt(res.status), out.written() });
        return error.BenchHttpStatus;
    }
    return out.toOwnedSlice();
}

fn get(allocator: std.mem.Allocator, client: *std.http.Client, url: []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const res = try client.fetch(.{ .location = .{ .url = url }, .keep_alive = false, .response_writer = &out.writer });
    if (res.status != .ok) return error.BenchHttpStatus;
    return out.toOwnedSlice();
}

fn readLog(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 30));
}

fn tokenizeChunk(allocator: std.mem.Allocator, client: *std.http.Client, base: []const u8, text: []const u8, ids: *std.ArrayList(u32)) !void {
    const url = try std.fmt.allocPrint(allocator, "{s}/tokenize", .{base});
    const body = try std.json.Stringify.valueAlloc(allocator, .{ .content = text }, .{});
    const resp = try post(allocator, client, url, body);
    const parsed = try std.json.parseFromSlice(struct { tokens: []const u32 }, allocator, resp, .{ .ignore_unknown_fields = true });
    try ids.appendSlice(allocator, parsed.value.tokens);
}

fn writeFile(io: std.Io, path: []const u8, body: []const u8) !void {
    if (std.fs.path.dirname(path)) |dir| try std.Io.Dir.cwd().createDirPath(io, dir);
    var f = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer f.close(io);
    var buf: [16 * 1024]u8 = undefined;
    var fw = f.writer(io, &buf);
    try fw.interface.writeAll(body);
    try fw.interface.flush();
}

fn printOut(io: std.Io, text: []const u8) !void {
    var buf: [4096]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    try w.interface.writeAll(text);
    try w.interface.flush();
}

/// One request of the plan: send the prompt, then read what the server logged for it (the restore line always
/// lands, the speculative mode only on a decode) a moment after the response.
fn runRequest(a: std.mem.Allocator, io: std.Io, client: *std.http.Client, url: []const u8, model: []const u8, log_path: []const u8, ids: []const u32, req: Request, arm: Arm, log_len: *usize, sampler: ?*gpu_clock.Sampler) !Sample {
    const body = try promptBody(a, model, ids[0..req.prompt_len], req.max_tokens);
    var wall = io_util.Stopwatch.init(io);
    const started_ms = io_util.nowMsMonotonic(io);
    const resp = try post(a, client, url, body);
    const ended_ms = io_util.nowMsMonotonic(io);
    const wall_ms = @as(f64, @floatFromInt(wall.read())) / std.time.ns_per_ms;
    const c = try parseCompletion(a, resp);

    var restore: ?Restore = null;
    var mode: ?[]const u8 = null;
    var tries: usize = 0;
    while (tries < 20) : (tries += 1) {
        const text = try readLog(a, io, log_path);
        const seg = text[@min(log_len.*, text.len)..];
        restore = parseRestore(seg);
        mode = parseSpecMode(seg);
        if (restore != null and (mode != null or req.max_tokens <= 1)) break;
        std.Io.sleep(io, .fromMilliseconds(500), .real) catch {};
    }
    log_len.* = (try readLog(a, io, log_path)).len;

    var gpu: gpu_clock.Summary = .{};
    if (sampler) |s| {
        // The sample that covers the request's last moments is printed at the end of its interval.
        _ = s.waitPast(ended_ms, 2 * gpu_clock.interval_ms);
        gpu = try s.summarizeWindow(a, started_ms, ended_ms);
    }

    return .{
        .arm = arm,
        .prompt_len = req.prompt_len,
        .kind = req.kind,
        .base = req.base,
        .max_tokens = req.max_tokens,
        .prompt_n = c.prompt_n,
        .cached_n = c.cached_n,
        .completion_n = c.completion_n,
        .prompt_ms = c.prompt_ms,
        .predicted_ms = c.predicted_ms,
        .wall_ms = wall_ms,
        .restore_ms = if (restore) |r| r.ms else 0,
        .restore_logged = restore != null,
        .mode = mode orelse "",
        .gpu_mhz = gpu.mhz,
        .gpu_pressure = gpu.pressure,
        .loop_stop = c.loop_stop,
    };
}

fn startSampler(gpa: std.mem.Allocator, io: std.Io) ?*gpu_clock.Sampler {
    const s = gpu_clock.Sampler.start(gpa, io, &gpu_clock.powermetrics_argv) catch |e| {
        log.warn("[bench] the GPU clock is not monitored: cannot run powermetrics ({s})\n", .{@errorName(e)});
        return null;
    };
    if (!s.waitFirst(5000)) {
        s.stop();
        log.warn("[bench] the GPU clock is not monitored: `sudo -n powermetrics` gave no reading (it needs a passwordless sudo rule)\n", .{});
        return null;
    }
    return s;
}

fn throttleMessage(a: std.mem.Allocator, s: Sample) ![]u8 {
    return std.fmt.allocPrint(a, "the GPU was throttled during the {s} {d}-token {s} request: median {d:.0} MHz (abort at {d:.0} MHz or under), thermal pressure {s}", .{
        @tagName(s.arm), s.prompt_len, @tagName(s.kind), s.gpu_mhz, gpu_clock.throttle_mhz, gpu_clock.pressureLabel(s.gpu_pressure orelse .nominal),
    });
}

pub fn cmdBench(gpa: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    const opts = parseArgs(args) catch |err| {
        log.err("sushi bench: {s}\n\n{s}", .{ @errorName(err), USAGE });
        return err;
    };
    if (opts.help) {
        try printOut(io, USAGE);
        return;
    }
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    const base = try std.fmt.allocPrint(a, "http://{s}", .{opts.url});
    var client: std.http.Client = .{ .allocator = a, .io = io };
    defer client.deinit();

    _ = get(a, &client, try std.fmt.allocPrint(a, "{s}/health", .{base})) catch {
        log.err("sushi bench: no server answers at {s}\n", .{base});
        return error.BenchServerDown;
    };
    var model = opts.model;
    if (model.len == 0) {
        const body = try get(a, &client, try std.fmt.allocPrint(a, "{s}/v1/models", .{base}));
        const parsed = try std.json.parseFromSlice(struct { data: []const struct { id: []const u8 } }, a, body, .{ .ignore_unknown_fields = true });
        if (parsed.value.data.len == 0) return error.BenchNoModel;
        model = parsed.value.data[0].id;
    }

    var samples: std.ArrayList(Sample) = .empty;
    const url = try std.fmt.allocPrint(a, "{s}/v1/completions", .{base});
    var log_len: usize = (try readLog(a, io, opts.log_path)).len;
    const sampler = startSampler(gpa, io);
    defer if (sampler) |s| s.stop();
    var aborted: []const u8 = "";
    var cooldown: ?Cooldown = null;
    arms: for ([_]Arm{ .code, .prose }) |arm| {
        if (arm == .prose) cooldown = cooldownBetweenArms(a, io);
        var ids: std.ArrayList(u32) = .empty;
        switch (arm) {
            .code => {
                var corpus = Corpus.init(corpus_seed);
                while (ids.items.len < corpus_tokens) try tokenizeChunk(a, &client, base, try corpus.nextChunk(a), &ids);
            },
            .prose => {
                var chunks: ProseChunks = .{ .text = prose_text };
                while (ids.items.len < corpus_tokens) {
                    const piece = chunks.next() orelse {
                        log.err("sushi bench: the prose corpus holds {d} tokens, short of the {d} the ladder needs\n", .{ ids.items.len, corpus_tokens });
                        return error.BenchProseTooShort;
                    };
                    try tokenizeChunk(a, &client, base, piece, &ids);
                }
            },
        }
        log.info("[bench] {s} corpus: {d} tokens, model {s}\n", .{ @tagName(arm), ids.items.len, model });
        const reqs = try plan(a, armDecodeTokens(arm, opts.decode_tokens));
        for (reqs) |req| {
            const s = try runRequest(a, io, &client, url, model, opts.log_path, ids.items, req, arm, &log_len, sampler);
            try samples.append(a, s);
            log.info("[bench] {s} {d} {s}: restored {d}, new {d}, prefill {d:.0} tok/s, ttft {d:.2} s, decode {d:.1} tok/s, mode {s}, gpu {d:.0} MHz {s}{s}\n", .{
                @tagName(arm),                     s.prompt_len,  @tagName(s.kind), s.cached_n, s.newTokens(), s.prefillTps(), s.ttftMs() / 1000.0, s.decodeTps(),
                if (s.mode.len > 0) s.mode else "none", s.gpu_mhz, if (s.gpu_pressure) |p| gpu_clock.pressureLabel(p) else "unmonitored", if (check(s) != null) " WARNING" else "",
            });
            if (check(s) == .gpu_throttled) {
                aborted = try throttleMessage(a, s);
                break :arms;
            }
        }
    }

    const rep: Report = .{
        .model = model,
        .version = opts.version,
        .stamp = opts.stamp,
        .decode_tokens = opts.decode_tokens,
        .samples = samples.items,
        .clock = if (sampler) |s| try s.summarizeAll(a) else .{},
        .aborted = aborted,
        .cooldown = cooldown,
    };
    const raw = try std.json.Stringify.valueAlloc(a, .{ .model = model, .url = opts.url, .decode_tokens = opts.decode_tokens, .corpus_seed = corpus_seed, .stamp = opts.stamp, .aborted = aborted, .samples = samples.items }, .{ .whitespace = .indent_2 });
    try writeFile(io, try std.fs.path.join(a, &.{ opts.out_dir, "bench.json" }), raw);
    const md = try render(a, rep);
    try printOut(io, md);
    if (aborted.len > 0) {
        // The partial report stays in the run directory: a docs/bench report is a finished measurement.
        try writeFile(io, try std.fs.path.join(a, &.{ opts.out_dir, "bench-aborted.md" }), md);
        log.err("sushi bench: aborted, {s}\n", .{aborted});
        return error.BenchGpuThrottled;
    }
    if (opts.report.len > 0) try writeFile(io, opts.report, md);
    for (samples.items) |s| if (check(s) != null) return error.BenchInvalidSamples;
}

// ── Tests ───────────────────────────────────────────────────────────────

test "bench args: defaults, every flag, and a refusal for the rest" {
    const o = try parseArgs(&.{ "--out", "/tmp/o", "--log", "/tmp/l", "--decode-tokens", "256", "--url", "127.0.0.1:9", "--version", "1.4.0" });
    try testing.expectEqual(@as(u32, 256), o.decode_tokens);
    try testing.expectEqualStrings("127.0.0.1:9", o.url);
    try testing.expectEqual(@as(u32, 256), (try parseArgs(&.{ "--out", "o", "--log", "l" })).decode_tokens);
    try testing.expectError(error.UnknownArgument, parseArgs(&.{ "--out", "o", "--log", "l", "--fast" }));
    try testing.expectError(error.MissingValue, parseArgs(&.{ "--out", "o", "--log" }));
    try testing.expectError(error.BadValue, parseArgs(&.{ "--out", "o", "--log", "l", "--decode-tokens", "0" }));
    try testing.expectError(error.BadValue, parseArgs(&.{ "--out", "o", "--log", "l", "--decode-tokens", "x" }));
    try testing.expectError(error.BadValue, parseArgs(&.{ "--out", "o", "--log", "l", "--url", "http://h:1" }));
    try testing.expectError(error.MissingRequired, parseArgs(&.{ "--url", "h:1" }));
    try testing.expect((try parseArgs(&.{"--help"})).help);
}

test "bench: --help lists every flag the parser accepts" {
    for (value_flags) |f| try testing.expect(std.mem.indexOf(u8, USAGE, f) != null);
    try testing.expect(std.mem.indexOf(u8, USAGE, "--help") != null);
}

test "bench corpus: the same seed gives the same bytes, a chunk is whole modules" {
    const a = testing.allocator;
    var c1 = Corpus.init(corpus_seed);
    var c2 = Corpus.init(corpus_seed);
    var c3 = Corpus.init(corpus_seed + 1);
    const x = try c1.nextChunk(a);
    defer a.free(x);
    const y = try c2.nextChunk(a);
    defer a.free(y);
    const z = try c3.nextChunk(a);
    defer a.free(z);
    try testing.expectEqualStrings(x, y);
    try testing.expect(!std.mem.eql(u8, x, z));
    try testing.expect(x.len >= 24 * 1024 and x.len < 40 * 1024);
    try testing.expect(std.mem.indexOf(u8, x, "export class") != null);
    try testing.expect(std.mem.endsWith(u8, x, "\n\n"));
    const next = try c1.nextChunk(a);
    defer a.free(next);
    try testing.expect(!std.mem.eql(u8, x, next));
}

test "bench plan: a cold 4k, 16 windows, two repeats after each of the six decode rungs" {
    const reqs = try plan(testing.allocator, 512);
    defer testing.allocator.free(reqs);
    try testing.expectEqual(@as(usize, 1 + ladder_steps + 6 * 2), reqs.len);
    try testing.expectEqual(@as(usize, 4096), reqs[0].prompt_len);
    try testing.expectEqual(@as(usize, 0), reqs[0].base);
    try testing.expectEqual(@as(u32, 512), reqs[0].max_tokens);
    // The 4k repeats come before the first window, which restores the 4k request's point.
    try testing.expectEqual(Kind.repeat, reqs[1].kind);
    try testing.expectEqual(@as(usize, 4096), reqs[2].base);
    try testing.expectEqual(@as(usize, 8192), reqs[3].prompt_len);
    try testing.expectEqual(@as(usize, 4096), reqs[3].base);
    // A non-rung window decodes one token and restores the window before it.
    const w3 = for (reqs) |r| {
        if (r.prompt_len == 3 * window) break r;
    } else unreachable;
    try testing.expectEqual(@as(u32, 1), w3.max_tokens);
    try testing.expectEqual(2 * window, w3.base);
    const last = reqs[reqs.len - 1];
    try testing.expectEqual(@as(usize, 131072), last.prompt_len);
    try testing.expectEqual(Kind.repeat, last.kind);
    var ladder: usize = 0;
    for (reqs) |r| {
        if (r.kind == .ladder) ladder += 1;
        try testing.expect(r.prompt_len <= window * ladder_steps);
    }
    try testing.expectEqual(@as(usize, 1 + ladder_steps), ladder);
}

test "bench samples: the restore comes out of the prefill time, and a missed restore is flagged" {
    var s: Sample = .{
        .prompt_len = 16384, .kind = .ladder, .base = 8192, .max_tokens = 1, .prompt_n = 16384, .cached_n = 8161,
        .completion_n = 1, .prompt_ms = 5000, .predicted_ms = 50, .wall_ms = 5100, .restore_ms = 1000, .restore_logged = true, .mode = "",
    };
    try testing.expectEqual(@as(u32, 8223), s.newTokens());
    try testing.expectApproxEqAbs(@as(f64, 2055.75), s.prefillTps(), 0.01);
    try testing.expectApproxEqAbs(@as(f64, 5050), s.ttftMs(), 0.01);
    try testing.expect(check(s) == null);
    s.cached_n = 4000;
    try testing.expectEqual(Problem.restore_missed, check(s).?);
    s.cached_n = 8161;
    s.restore_logged = false;
    try testing.expectEqual(Problem.no_restore_line, check(s).?);
    s.restore_logged = true;
    s.max_tokens = 512;
    try testing.expectEqual(Problem.short_decode, check(s).?);
    s.loop_stop = true;
    try testing.expect(check(s) == null);
    s.loop_stop = false;
    s.max_tokens = 1;
    s.base = 0;
    try testing.expectEqual(Problem.restore_missed, check(s).?);
}

test "bench median: odd, even and empty" {
    try testing.expectEqual(@as(f64, 3), median(&.{ 5, 1, 3 }));
    try testing.expectEqual(@as(f64, 2.5), median(&.{ 4, 1 }));
    try testing.expectEqual(@as(f64, 0), median(&.{}));
}

test "bench wire: a prompt body is valid JSON with greedy ignore_eos token ids" {
    const a = testing.allocator;
    const body = try promptBody(a, "m\"x", &.{ 1, 22, 333 }, 512);
    defer a.free(body);
    const p = try std.json.parseFromSlice(std.json.Value, a, body, .{});
    defer p.deinit();
    const o = p.value.object;
    try testing.expectEqualStrings("m\"x", o.get("model").?.string);
    try testing.expectEqual(@as(i64, 512), o.get("max_tokens").?.integer);
    try testing.expect(o.get("ignore_eos").?.bool);
    try testing.expectEqual(@as(usize, 3), o.get("prompt").?.array.items.len);
    const bare = try promptBody(a, "", &.{7}, 1);
    defer a.free(bare);
    try testing.expect(std.mem.indexOf(u8, bare, "\"model\"") == null);
}

test "bench wire: a completion's timings, and a reply without them is refused" {
    const a = testing.allocator;
    const c = try parseCompletion(a, "{\"usage\":{},\"timings\":{\"prompt_n\":8223,\"cached_n\":31,\"prompt_ms\":4000.5,\"predicted_n\":512,\"predicted_ms\":8000.0,\"tokenize_ms\":1.0}}");
    try testing.expectEqual(@as(u32, 8223), c.prompt_n);
    try testing.expectEqual(@as(u32, 31), c.cached_n);
    try testing.expectEqual(@as(u32, 512), c.completion_n);
    try testing.expectApproxEqAbs(@as(f64, 4000.5), c.prompt_ms, 0.001);
    try testing.expectError(error.NoTimings, parseCompletion(a, "{\"usage\":{}}"));
    try testing.expect(!c.loop_stop);
}

test "bench plan: the prose arm decodes at most 128 tokens a sample, the code arm what was asked" {
    try testing.expectEqual(@as(u32, 256), armDecodeTokens(.code, 256));
    try testing.expectEqual(@as(u32, 128), armDecodeTokens(.prose, 256));
    try testing.expectEqual(@as(u32, 64), armDecodeTokens(.prose, 64));
}

test "bench wire: a loop stop is read from the choice's finish_details" {
    const a = testing.allocator;
    const timings = "\"timings\":{\"prompt_n\":8223,\"cached_n\":31,\"prompt_ms\":4000.5,\"predicted_n\":390,\"predicted_ms\":6000.0}";
    const loop = try parseCompletion(a, "{\"choices\":[{\"index\":0,\"text\":\"x\",\"finish_reason\":\"stop\",\"finish_details\":{\"type\":\"repetition_loop\"}}]," ++ timings ++ "}");
    try testing.expect(loop.loop_stop);
    try testing.expectEqual(@as(u32, 390), loop.completion_n);
    const plain = try parseCompletion(a, "{\"choices\":[{\"index\":0,\"text\":\"x\",\"finish_reason\":\"stop\"}]," ++ timings ++ "}");
    try testing.expect(!plain.loop_stop);
}

test "bench log: the restore line the scheduler writes and the last spec mode parse back" {
    const prefix_cache = @import("prefix_cache.zig");
    var buf: [192]u8 = undefined;
    const line = try prefix_cache.restoreLogLine(&buf, "[prefix]", 8161, 16384, "SSD", 42);
    const seg = try std.fmt.allocPrint(testing.allocator, "noise\n{s}\n  [spec-stats] mode=mtp attempts=1\n  [spec-stats] mode=dspark attempts=2 x\n", .{line});
    defer testing.allocator.free(seg);
    const r = parseRestore(seg).?;
    try testing.expectEqual(@as(usize, 8161), r.matched);
    try testing.expectEqual(@as(usize, 16384), r.total);
    try testing.expectEqual(@as(f64, 42), r.ms);
    try testing.expectEqualStrings("dspark", parseSpecMode(seg).?);
    try testing.expect(parseRestore("nothing here\n") == null);
    try testing.expect(parseSpecMode("nothing here\n") == null);
    const dsv = try prefix_cache.restoreLogLine(&buf, "[dsv41-prefix]", 0, 4096, "nothing", 0);
    try testing.expectEqual(@as(usize, 0), parseRestore(dsv).?.matched);
}

test "bench report: decode rows, the window table, the summary row and the warnings" {
    const a = testing.allocator;
    const mk = struct {
        fn s(len: usize, kind: Kind, base: usize, max: u32, cached: u32, tps_ms: f64) Sample {
            return .{ .prompt_len = len, .kind = kind, .base = base, .max_tokens = max, .prompt_n = @intCast(len), .cached_n = cached, .completion_n = if (max > 1) 512 else 1, .prompt_ms = 4000, .predicted_ms = tps_ms, .wall_ms = 4500 + tps_ms, .restore_ms = 0, .restore_logged = true, .mode = "dspark" };
        }
    }.s;
    const samples = [_]Sample{
        mk(4096, .ladder, 0, 512, 0, 10_000),
        mk(4096, .repeat, 4096, 512, 4065, 8_000),
        mk(4096, .repeat, 4096, 512, 4065, 16_000),
        mk(8192, .ladder, 4096, 512, 4065, 10_000),
    };
    const md = try render(a, .{ .model = "Pack-X", .version = "1.4.0", .stamp = "commit abc\\nM5 Max", .decode_tokens = 512, .samples = &samples });
    defer a.free(md);
    try testing.expect(std.mem.indexOf(u8, md, "# Pack-X bench (Sushi 1.4.0)") != null);
    try testing.expect(std.mem.indexOf(u8, md, "commit abc\n\nM5 Max") != null);
    // 512 tokens in 10 s, 8 s and 16 s is 51.2, 64.0 and 32.0 tok/s; the median is 51.2.
    try testing.expect(std.mem.indexOf(u8, md, "| 4k | 51.2 | 64.0 | 32.0 | 51.2 | dspark |") != null);
    try testing.expect(std.mem.indexOf(u8, md, "| 4,065 to 8,192 | 4,065 | 4,127 | 1,032 |") != null);
    try testing.expect(std.mem.indexOf(u8, md, "| **Pack-X** | Prefill |") != null);
    try testing.expect(std.mem.indexOf(u8, md, "## Warnings") == null);
    try testing.expect(std.mem.indexOf(u8, md, "Prose") == null);
    var bad = samples;
    bad[3].cached_n = 0;
    const md2 = try render(a, .{ .model = "Pack-X", .version = "", .stamp = "", .decode_tokens = 512, .samples = &bad });
    defer a.free(md2);
    try testing.expect(std.mem.indexOf(u8, md2, "## Warnings") != null);
    try testing.expect(std.mem.indexOf(u8, md, "## Notes") == null);

    var looped = samples;
    looped[1].completion_n = 390;
    looped[1].loop_stop = true;
    const md3 = try render(a, .{ .model = "Pack-X", .version = "", .stamp = "", .decode_tokens = 512, .samples = &looped });
    defer a.free(md3);
    try testing.expect(std.mem.indexOf(u8, md3, "## Warnings") == null);
    try testing.expect(std.mem.indexOf(u8, md3, "## Notes") != null);
    try testing.expect(std.mem.indexOf(u8, md3, "the loop stop ended it at 390 tokens") != null);
}

test "bench prose: the embedded novel is clean, whole, and cut into paragraph-aligned pieces" {
    try testing.expect(prose_text.len > 600 * 1024);
    // The Project Gutenberg header, footer and trademark terms are not part of the corpus.
    try testing.expect(std.mem.indexOf(u8, prose_text, "Gutenberg") == null);
    try testing.expect(std.mem.indexOf(u8, prose_text, "It is a truth universally acknowledged") != null);
    try testing.expect(std.mem.indexOfScalar(u8, prose_text, '\r') == null);
    var chunks: ProseChunks = .{ .text = prose_text };
    var total: usize = 0;
    var n: usize = 0;
    var rebuilt = std.ArrayList(u8).empty;
    defer rebuilt.deinit(testing.allocator);
    while (chunks.next()) |piece| : (n += 1) {
        try testing.expect(piece.len > 0 and piece.len <= 24 * 1024);
        total += piece.len;
        try rebuilt.appendSlice(testing.allocator, piece);
        // A cut lands on a blank line unless it is the last piece.
        if (total < prose_text.len) try testing.expect(std.mem.endsWith(u8, piece, "\n\n"));
    }
    try testing.expectEqual(prose_text.len, total);
    try testing.expectEqualStrings(prose_text, rebuilt.items);
    try testing.expect(n > 25);
    try testing.expect(chunks.next() == null);
}

test "bench report: a prose arm sits beside the code arm and gets its own summary rows" {
    const a = testing.allocator;
    const mk = struct {
        fn s(arm: Arm, len: usize, kind: Kind, base: usize, max: u32, cached: u32, decode_ms: f64) Sample {
            return .{ .arm = arm, .prompt_len = len, .kind = kind, .base = base, .max_tokens = max, .prompt_n = @intCast(len), .cached_n = cached, .completion_n = if (max > 1) 512 else 1, .prompt_ms = 4000, .predicted_ms = decode_ms, .wall_ms = 4500 + decode_ms, .restore_ms = 0, .restore_logged = true, .mode = "dspark" };
        }
    }.s;
    const samples = [_]Sample{
        mk(.code, 4096, .ladder, 0, 512, 0, 10_000),
        mk(.code, 4096, .repeat, 4096, 512, 4065, 10_000),
        mk(.code, 4096, .repeat, 4096, 512, 4065, 10_000),
        mk(.prose, 4096, .ladder, 0, 512, 0, 20_000),
        mk(.prose, 4096, .repeat, 4096, 512, 4065, 20_000),
        mk(.prose, 4096, .repeat, 4096, 512, 4065, 20_000),
        mk(.prose, 8192, .ladder, 4096, 1, 4065, 50),
    };
    const md = try render(a, .{ .model = "Pack-X", .version = "", .stamp = "", .decode_tokens = 512, .samples = &samples });
    defer a.free(md);
    // 512 tokens in 10 s is 51.2 tok/s on code and 25.6 on prose: a ratio of 0.50.
    try testing.expect(std.mem.indexOf(u8, md, "| 4k | 51.2 | 51.2 | 51.2 | 51.2 | dspark | 25.6 | 25.6 | 25.6 | 25.6 | dspark | 0.50 |") != null);
    try testing.expect(std.mem.indexOf(u8, md, "## Prefill by window, code") != null);
    try testing.expect(std.mem.indexOf(u8, md, "## Prefill by window, prose") != null);
    try testing.expect(std.mem.indexOf(u8, md, "| **Pack-X** | Prefill |") != null);
    try testing.expect(std.mem.indexOf(u8, md, "| **Pack-X (prose)** | Prefill |") != null);
    // The code row keeps its place ahead of the prose row.
    try testing.expect(std.mem.indexOf(u8, md, "| **Pack-X** |").? < std.mem.indexOf(u8, md, "| **Pack-X (prose)** |").?);
    var bad = samples;
    bad[6].cached_n = 0;
    const md2 = try render(a, .{ .model = "Pack-X", .version = "", .stamp = "", .decode_tokens = 512, .samples = &bad });
    defer a.free(md2);
    try testing.expect(std.mem.indexOf(u8, md2, "- prose, 8192 tokens (ladder)") != null);
}

test "bench report: GPU clock columns beside the tok/s, the header line, and an abort notice" {
    const a = testing.allocator;
    const mk = struct {
        fn s(arm: Arm, len: usize, kind: Kind, base: usize, max: u32, cached: u32, decode_ms: f64, mhz: f64, p: gpu_clock.Pressure) Sample {
            return .{ .arm = arm, .prompt_len = len, .kind = kind, .base = base, .max_tokens = max, .prompt_n = @intCast(len), .cached_n = cached, .completion_n = if (max > 1) 512 else 1, .prompt_ms = 4000, .predicted_ms = decode_ms, .wall_ms = 4500 + decode_ms, .restore_ms = 0, .restore_logged = true, .mode = "dspark", .gpu_mhz = mhz, .gpu_pressure = p };
        }
    }.s;
    const samples = [_]Sample{
        mk(.code, 4096, .ladder, 0, 512, 0, 10_000, 1600, .nominal),
        mk(.code, 4096, .repeat, 4096, 512, 4065, 10_000, 1580, .nominal),
        mk(.code, 4096, .repeat, 4096, 512, 4065, 10_000, 1620, .nominal),
        mk(.code, 8192, .ladder, 4096, 1, 4065, 50, 1050, .heavy),
    };
    try testing.expectEqual(Problem.gpu_throttled, check(samples[3]).?);
    try testing.expect(check(samples[0]) == null);
    // Heavy pressure with the clock held is a measurement, not a throttled request.
    try testing.expect(check(mk(.prose, 4096, .repeat, 4096, 512, 4065, 10_000, 1539, .heavy)) == null);
    const msg = try throttleMessage(a, samples[3]);
    defer a.free(msg);
    try testing.expectEqualStrings("the GPU was throttled during the code 8192-token ladder request: median 1050 MHz (abort at 1100 MHz or under), thermal pressure Heavy", msg);
    const clock: gpu_clock.RunSummary = .{ .monitored = true, .min_mhz = 1050, .median_mhz = 1590, .busy_samples = 40, .seen = 0b101 };
    const md = try render(a, .{ .model = "Pack-X", .version = "", .stamp = "", .decode_tokens = 512, .samples = &samples, .clock = clock, .aborted = msg });
    defer a.free(md);
    try testing.expect(std.mem.indexOf(u8, md, "**Aborted, not a measurement:** the GPU was throttled during the code 8192-token") != null);
    try testing.expect(std.mem.indexOf(u8, md, "GPU clock, sampled every 500 ms: median 1590 MHz, minimum 1050 MHz over 40 busy samples; thermal pressure seen: Nominal, Heavy.") != null);
    try testing.expect(std.mem.indexOf(u8, md, "| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure |") != null);
    try testing.expect(std.mem.indexOf(u8, md, "| 4k | 51.2 | 51.2 | 51.2 | 51.2 | dspark | 1600 | Nominal |") != null);
    try testing.expect(std.mem.indexOf(u8, md, "| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |") != null);
    try testing.expect(std.mem.indexOf(u8, md, "| 4,065 to 8,192 | 4,065 | 4,127 | 1,032 | 0 | 4.50 | 1050 | Heavy |\n") != null);
    try testing.expect(std.mem.indexOf(u8, md, "- code, 8192 tokens (ladder): the GPU clock or thermal pressure failed the abort rule") != null);
    // Without a reading nothing is added: no columns, and the header says it was not monitored.
    var plain = samples;
    for (&plain) |*s| {
        s.gpu_mhz = 0;
        s.gpu_pressure = null;
    }
    const md2 = try render(a, .{ .model = "Pack-X", .version = "", .stamp = "", .decode_tokens = 512, .samples = &plain });
    defer a.free(md2);
    try testing.expect(std.mem.indexOf(u8, md2, "GPU clock: not monitored") != null);
    try testing.expect(std.mem.indexOf(u8, md2, "Code MHz") == null);
    try testing.expect(std.mem.indexOf(u8, md2, "GPU MHz") == null);
    try testing.expect(std.mem.indexOf(u8, md2, "## Warnings") == null);
}

test "bench cooldown: stops under 50 C, is capped at 180 s, and waits the full cap without a reading" {
    try testing.expect(cooldownDone(49.9, 0));
    try testing.expect(!cooldownDone(50.0, 0));
    try testing.expect(!cooldownDone(87, 170));
    try testing.expect(cooldownDone(87, 180));
    try testing.expect(!cooldownDone(null, 0));
    try testing.expect(!cooldownDone(null, 170));
    try testing.expect(cooldownDone(null, 180));
}

test "bench cooldown: the hottest die in a fanctl status document, and nothing from a bad one" {
    const status =
        \\{
        \\  "fans" : [ { "actualRPM" : 5356, "index" : 0, "mode" : 0 } ],
        \\  "state" : "max",
        \\  "topTemps" : [
        \\    { "celsius" : 55.165626525878906, "key" : "TCMb" },
        \\    { "celsius" : 87, "key" : "TCHP" },
        \\    { "celsius" : 47.078125, "key" : "TCDX" }
        \\  ]
        \\}
    ;
    try testing.expectEqual(@as(f64, 87), hottestDie(testing.allocator, status).?);
    try testing.expectEqual(@as(?f64, null), hottestDie(testing.allocator, "{\"topTemps\":[]}"));
    try testing.expectEqual(@as(?f64, null), hottestDie(testing.allocator, "not json"));
    try testing.expectEqual(@as(?f64, null), hottestDie(testing.allocator, "{\"state\":\"max\"}"));
}

test "bench report: the protocol line names the cooldown between the arms" {
    const a = testing.allocator;
    const md = try render(a, .{ .model = "Pack-X", .version = "", .stamp = "", .decode_tokens = 512, .samples = &.{}, .cooldown = .{ .waited_s = 140, .start_c = 87, .end_c = 49 } });
    defer a.free(md);
    try testing.expect(std.mem.indexOf(u8, md, "The server idled 140 s between the arms (die 87 C to 49 C).") != null);
    const blind = try render(a, .{ .model = "Pack-X", .version = "", .stamp = "", .decode_tokens = 512, .samples = &.{}, .cooldown = .{ .waited_s = 180 } });
    defer a.free(blind);
    try testing.expect(std.mem.indexOf(u8, blind, "The server idled 180 s between the arms (die not read).") != null);
    const none = try render(a, .{ .model = "Pack-X", .version = "", .stamp = "", .decode_tokens = 512, .samples = &.{} });
    defer a.free(none);
    try testing.expect(std.mem.indexOf(u8, none, "between the arms") == null);
}
