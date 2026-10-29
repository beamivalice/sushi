//! `sushi-ubench`: the microbench binary. It owns the registry, the CLI and the quiet-box steps; the bench bodies
//! are `exl3/ubench.zig` and its siblings, and the pure half (tables, JSON, compare) is ubench_report.zig. A bench
//! is never a test: the exit code says whether it measured, never how fast.
//! Docs: process-ubench.md.
const std = @import("std");
const builtin = @import("builtin");
const stamp = @import("ubench_stamp");
// pub: src/exl3 reaches mlx, log, io_util and the bench types through its host root.
pub const mlx = @import("mlx.zig");
pub const io_util = @import("io_util.zig");
pub const log = @import("log.zig");
pub const report = @import("ubench_report.zig");
const box = @import("bench_box.zig");
const exl3 = @import("sushi_exl3");
const dsv41 = @import("ubench_dsv41.zig");
const glm5 = @import("ubench_glm5.zig");
const shared = @import("ubench_shared.zig");
const encoder = @import("sushi_encoder");
const ubench_mimo2 = @import("ubench_mimo2.zig");

const registry = [_]report.Bench{
    exl3.ubench.dsv41_experts,
    dsv41.index_decode,
    dsv41.index_peak,
    dsv41.index_block_peak,
    dsv41.rows_growth,
    dsv41.attn_span,
    dsv41.topk,
    exl3.ubench.decode_chain,
    exl3.ubench.grouped_decode,
    exl3.ubench.moe_layer,
    exl3.ubench.prefill_gemm,
    exl3.ubench.trunk_rows,
    exl3.ubench.routed_t1,
    ubench_mimo2.fp8_gemv,
    ubench_mimo2.vision_encode,
    exl3.ubench.exl3_rate_groups,
    exl3.ubench.exl3_rate_groups_rows,
    encoder.ubench.factor,
    encoder.ubench.search,
} ++ glm5.benches ++ shared.benches;

const exit_bad_args = 2;
const exit_refused = 3;
const fan_renew_s = 300;
const queue_log_s = 30;

var cancelled: std.atomic.Value(bool) = .init(false);
var fan_io: ?std.Io = null;
var fan_renew_at_ms: i64 = 0;
var fan_alloc: std.mem.Allocator = undefined;

fn onSignal(_: std.posix.SIG) callconv(.c) void {
    cancelled.store(true, .release);
    const pid = box.acquiring_pid.load(.acquire);
    if (pid > 0) std.posix.kill(pid, .TERM) catch {};
}

fn installSignals() void {
    const act: std.posix.Sigaction = .{ .handler = .{ .handler = onSignal }, .mask = std.posix.sigemptyset(), .flags = 0 };
    std.posix.sigaction(std.posix.SIG.INT, &act, null);
    std.posix.sigaction(std.posix.SIG.TERM, &act, null);
}

/// A request for max fans lasts 15 minutes; a long run renews it every 5.
fn renewFans() void {
    const io = fan_io orelse return;
    const now = io_util.nowMsMonotonic(io);
    if (now < fan_renew_at_ms) return;
    box.fansMax(fan_alloc, io);
    fan_renew_at_ms = now + fan_renew_s * 1000;
}

fn writeStdout(io: std.Io, text: []const u8) !void {
    var buf: [4096]u8 = undefined;
    var w = std.Io.File.stdout().writerStreaming(io, &buf);
    try w.interface.writeAll(text);
    try w.interface.flush();
}

fn fail(code: u8, comptime fmt: []const u8, args: anytype) noreturn {
    log.err("sushi-ubench: " ++ fmt ++ "\n", args);
    std.process.exit(code);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena_state = std.heap.ArenaAllocator.init(init.gpa);
    const a = arena_state.allocator();

    var argv: std.ArrayList([]const u8) = .empty;
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, a);
    while (it.next()) |arg| try argv.append(a, try a.dupe(u8, arg));
    const args = argv.items[1..];

    var msg: []const u8 = "";
    const cmd = report.parse(a, args, &msg) catch |e| switch (e) {
        error.BadArgs => {
            if (args.len == 0) {
                try writeStdout(io, report.USAGE);
                std.process.exit(exit_bad_args);
            }
            log.err("sushi-ubench: {s}\n\n{s}", .{ msg, report.USAGE });
            std.process.exit(exit_bad_args);
        },
        else => return e,
    };
    switch (cmd) {
        .help => try writeStdout(io, report.USAGE),
        .list => |scope| {
            var aw: std.Io.Writer.Allocating = .init(a);
            try report.writeList(&aw.writer, &registry, scope);
            try writeStdout(io, aw.written());
        },
        .compare => |files| try runCompare(a, io, files),
        .run => |ra| try runBenches(a, io, ra),
    }
}

fn readResult(a: std.mem.Allocator, io: std.Io, path: []const u8) report.Result {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28)) catch |e| fail(1, "{s}: {s}", .{ path, @errorName(e) });
    return report.fromJson(a, bytes) catch |e| fail(1, "{s}: not a sushi-ubench result ({s})", .{ path, @errorName(e) });
}

fn runCompare(a: std.mem.Allocator, io: std.Io, files: [2][]const u8) !void {
    const x = readResult(a, io, files[0]);
    const y = readResult(a, io, files[1]);
    var aw: std.Io.Writer.Allocating = .init(a);
    report.compare(&aw.writer, a, x, y) catch |e| switch (e) {
        error.DifferentBench => fail(1, "the results are of different benches: {s} and {s}", .{ x.bench, y.bench }),
        error.ContendedMix => fail(1, "refusing to compare a contended result with a quiet one", .{}),
        else => return e,
    };
    try writeStdout(io, aw.written());
}

const Box = struct {
    a: std.mem.Allocator,
    io: std.Io,
    repo: []const u8,
    lock_script: []const u8,
    quiet: []const u8,
    contended: bool,

    /// The facts steps 1-3 and 6 of the quiet box read; a contended run reads none of them.
    fn refusal(self: Box, buf: []u8) ?[]const u8 {
        if (self.contended) return null;
        var facts: box.Facts = .{ .dirty = stamp.dirty, .pri = box.priority(self.a, self.io), .gpu_jobs = box.liveGpuJobs(self.a, self.io), .power = box.power(self.a, self.io) };
        const head = box.runOut(self.a, self.io, &.{ "git", "-C", self.repo, "rev-parse", "HEAD" });
        const now = if (head) |h| std.mem.trim(u8, h, " \n\r\t") else "unreadable";
        if (!std.mem.eql(u8, now, stamp.commit)) facts.stale = std.fmt.allocPrint(self.a, "built from {s}, HEAD is {s}", .{ shortSha(stamp.commit), shortSha(now) }) catch "commit differs";
        return box.firstRefusal(facts, buf);
    }

    fn waitForBuilds(self: Box) void {
        var waited: u32 = 0;
        while (box.buildsRunning(self.a, self.io) and !cancelled.load(.acquire)) {
            if (waited % queue_log_s == 0) log.info("[ubench] waiting for zig/clang/ld to end (QUIET is set)\n", .{});
            std.Io.sleep(self.io, .fromSeconds(5), .real) catch {};
            waited += 5;
        }
    }
};

fn shortSha(sha: []const u8) []const u8 {
    return sha[0..@min(sha.len, 8)];
}

fn runBenches(a: std.mem.Allocator, io: std.Io, ra: report.RunArgs) !void {
    if (builtin.mode != .fast) fail(exit_refused, "built in {s}: a Debug or safe build gives fake regressions, build with `zig build ubench`", .{@tagName(builtin.mode)});
    var msg: []const u8 = "";
    const selected = report.select(a, &registry, ra, &msg) catch |e| switch (e) {
        error.BadArgs => fail(exit_bad_args, "{s}", .{msg}),
        else => return e,
    };
    for (selected.skipped) |name| log.warn("[ubench] SKIPPED {s}: it needs a pack and the run has no --pack\n", .{name});

    const home = std.mem.span(std.c.getenv("HOME") orelse fail(1, "HOME is not set", .{}));
    const b: Box = .{
        .a = a,
        .io = io,
        .repo = stamp.repo,
        .lock_script = try std.fs.path.join(a, &.{ stamp.repo, "scripts", "gpu-lock.sh" }),
        .quiet = try box.quietPath(a),
        .contended = ra.contended,
    };
    var refusal_buf: [512]u8 = undefined;
    if (b.refusal(&refusal_buf)) |why| fail(exit_refused, "{s}", .{why});

    mlx.installErrorHandler();
    if (!mlx.streamIsGpu(mlx.gpuStream())) fail(exit_refused, "no Metal GPU stream", .{});
    installSignals();
    fan_io = io;
    fan_alloc = a;

    const out_dir = ra.out orelse try std.fs.path.join(a, &.{ home, ".sushi", "runs", "ubench" });
    var all_valid = true;
    for (selected.plans) |plan| {
        const result = runOne(a, io, b, ra, plan) catch |e| switch (e) {
            error.Interrupted => std.process.exit(130),
            else => fail(1, "{s}: {s}", .{ plan.bench.name, @errorName(e) }),
        };
        var aw: std.Io.Writer.Allocating = .init(a);
        try report.renderTable(&aw.writer, a, result);
        try aw.writer.writeByte('\n');
        try writeStdout(io, aw.written());
        const file = try std.fmt.allocPrint(a, "{s}-{s}-{s}.json", .{ result.bench, result.stamps.commit[0..@min(7, result.stamps.commit.len)], result.stamps.time });
        const path = try std.fs.path.join(a, &.{ out_dir, file });
        try std.Io.Dir.cwd().createDirPath(io, out_dir);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = try report.toJson(a, result) });
        log.info("[ubench] {s}\n", .{path});
        if (std.mem.startsWith(u8, result.validity, "invalid:")) all_valid = false;
    }
    if (!all_valid) fail(1, "an invalid run: the arms disagreed or a check failed (the JSON is kept)", .{});
}

/// One bench under the quiet box: lock, then (unless contended) QUIET, builds gone, AC, cool die, then the body.
fn runOne(a: std.mem.Allocator, io: std.Io, b: Box, ra: report.RunArgs, plan: report.Plan) !report.Result {
    const bench = plan.bench;
    const owner = try std.fmt.allocPrint(a, "ubench:{s}", .{bench.name});
    box.lockAcquire(io, b.lock_script, owner) catch |e| {
        if (cancelled.load(.acquire)) return error.Interrupted;
        return e;
    };
    defer box.lockRelease(io, b.lock_script, owner);

    var die_start: ?f64 = null;
    if (!b.contended) {
        try box.quietTouch(io, b.quiet);
        defer box.quietRemove(io, b.quiet);
        b.waitForBuilds();
        if (cancelled.load(.acquire)) return error.Interrupted;
        if (box.power(a, io) != .ac) fail(exit_refused, "not on AC power", .{});
        const cool = box.cooldown(a, io);
        die_start = cool.end_c;
        fan_renew_at_ms = io_util.nowMsMonotonic(io) + fan_renew_s * 1000;
        defer box.fansAuto(a, io);
        return measure(a, io, b, ra, plan, owner, die_start);
    }
    return measure(a, io, b, ra, plan, owner, box.readDie(a, io));
}

fn measure(a: std.mem.Allocator, io: std.Io, b: Box, ra: report.RunArgs, plan: report.Plan, owner: []const u8, die_start: ?f64) !report.Result {
    var run: report.Run = .{
        .a = a,
        .plan = plan,
        .rounds = ra.rounds orelse report.default_rounds,
        .steps_override = ra.steps,
        .pack = ra.pack,
        .cancel = &cancelled,
        .after_sample = if (b.contended) null else renewFans,
    };
    try plan.bench.run(&run);
    const die_end = box.readDie(a, io);
    const st = try stamps(a, io, b, owner, die_start, die_end);
    return run.finish(st);
}

fn stamps(a: std.mem.Allocator, io: std.Io, b: Box, owner: []const u8, die_start: ?f64, die_end: ?f64) !report.Stamps {
    const exe = try std.process.executablePathAlloc(io, a);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, exe, a, .limited(1 << 30));
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    const file_stat = try std.Io.Dir.cwd().statFile(io, exe, .{});
    const secs = file_stat.mtime.toSeconds();
    return .{
        .commit = try a.dupe(u8, shortSha(stamp.commit)),
        .dirty = stamp.dirty,
        .sha256 = try a.dupe(u8, hex[0..12]),
        .mtime = try report.formatTime(a, secs, true),
        .chip = try chipText(a, io),
        .qos = box.priority(a, io),
        .power = @tagName(box.power(a, io)),
        .die_start_c = die_start,
        .die_end_c = die_end,
        .lock = owner,
        .contended = b.contended,
        .time = try report.formatTime(a, io_util.nowSecs(io), false),
    };
}

fn chipText(a: std.mem.Allocator, io: std.Io) ![]const u8 {
    const brand = if (box.runOut(a, io, &.{ "sysctl", "-n", "machdep.cpu.brand_string" })) |t| std.mem.trim(u8, t, " \n") else "unknown chip";
    const mem = if (box.runOut(a, io, &.{ "sysctl", "-n", "hw.memsize" })) |t| std.fmt.parseInt(u64, std.mem.trim(u8, t, " \n"), 10) catch 0 else 0;
    const os = if (box.runOut(a, io, &.{ "sw_vers", "-productVersion" })) |t| std.mem.trim(u8, t, " \n") else "?";
    return std.fmt.allocPrint(a, "{s} {d} GB, macOS {s}", .{ brand, mem >> 30, os });
}
