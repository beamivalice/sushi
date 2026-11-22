//! Sushi's test runner: Zig 0.17.1's default runner (`lib/compiler/test_runner.zig`, MIT), unchanged but for `runTest`,
//! which asks after every test whether the engine is at rest (`sushi_after_each`, exported where the engine is built)
//! and fails the test that left it otherwise, by name, before the next one starts, for the per-test log that
//! `main` reads from `SUSHI_TEST_LOG` and `SUSHI_TEST_STEP` (`zig build test -Dtest-log=<path>`), and for the run step's
//! selection, `SUSHI_TEST_SELECT`.
const builtin = @import("builtin");

const std = @import("std");
const Io = std.Io;
const fatal = std.process.fatal;
const testing = std.testing;
const assert = std.debug.assert;
const panic = std.debug.panic;
const fuzz_abi = std.Build.abi.fuzz;

fn noHook() callconv(.c) ?[*:0]const u8 {
    return null;
}
comptime {
    @export(&noHook, .{ .name = "sushi_after_each", .linkage = .weak });
    @export(&noHook, .{ .name = "sushi_after_all", .linkage = .weak });
    @export(&noHook, .{ .name = "sushi_test_note", .linkage = .weak });
}
/// After each test: what of it did not run (null: all of it ran), written on its log line.
extern fn sushi_test_note() callconv(.c) ?[*:0]const u8;
extern fn sushi_after_each() callconv(.c) ?[*:0]const u8;
/// As the process ends: what the loads a run step kept across its tests left once unloaded (null: nothing).
extern fn sushi_after_all() callconv(.c) ?[*:0]const u8;

/// The step's end: a held load's unload that left anything fails the step, logged as a test of its own.
fn afterAll() bool {
    const why = sushi_after_all() orelse return true;
    std.debug.print("after the step's last test, the loads it held: {s}\n", .{why});
    _ = logTest("(the step's held loads, unloaded)", .FAIL);
    return false;
}

/// The per-test log: after each test `<UTC>\t<step>\t<PASS|SKIP|FAIL>\t<ms>\t<name>`, and once a process has run a
/// test, `<UTC>\t<step>\tDONE\t<ms>\t<p> pass, <s> skip, <f> fail` as it exits. A test's ms hold its at-rest check;
/// DONE's are the process's own wall from its first test. Nothing goes to the test's own streams.
const TestLog = struct {
    path: []const u8,
    step: []const u8,
    counts: [3]u32 = @splat(0),
    first: ?Io.Timestamp = null,

    const Status = enum { PASS, SKIP, FAIL };

    /// A test leaking or logging an error fails its step as a failed test does, so it is a FAIL here too.
    fn status(passed: bool, skipped: bool, leaked: bool, logged_errors: bool) Status {
        if (skipped and !leaked and !logged_errors) return .SKIP;
        return if (passed and !leaked and !logged_errors) .PASS else .FAIL;
    }

    /// `note`: the parts of the test that did not run, a sixth field after the name.
    fn record(self: *TestLog, name: []const u8, s: Status, ms: i64, note: ?[*:0]const u8) !void {
        self.counts[@backingInt(s)] += 1;
        if (note) |n| return self.append("{t}\t{d}\t{s}\tparts skipped: {s}", .{ s, ms, name, std.mem.span(n) });
        try self.append("{t}\t{d}\t{s}", .{ s, ms, name });
    }

    fn done(self: *TestLog) !void {
        const first = self.first orelse return;
        const ms = first.untilNow(runner_threaded_io, .awake).toMilliseconds();
        const c = self.counts;
        try self.append("DONE\t{d}\t{d} pass, {d} skip, {d} fail", .{ ms, c[0], c[1], c[2] });
    }

    /// The binaries run as parallel processes, so a line is written under the file's exclusive lock at its current end.
    fn append(self: *TestLog, comptime fmt: []const u8, args: anytype) !void {
        var line_buf: [4096]u8 = undefined;
        var w = std.Io.Writer.fixed(&line_buf);
        writeUtc(&w, Io.Timestamp.now(runner_threaded_io, .real)) catch return error.TestNameTooLong;
        w.print("\t{s}\t" ++ fmt ++ "\n", .{self.step} ++ args) catch return error.TestNameTooLong;
        const file = try Io.Dir.createFileAbsolute(runner_threaded_io, self.path, .{ .truncate = false, .lock = .exclusive });
        defer file.close(runner_threaded_io);
        try file.writePositionalAll(runner_threaded_io, w.buffered(), try file.length(runner_threaded_io));
    }
};

/// `2026-11-19T07:13:02.123Z`.
fn writeUtc(w: *std.Io.Writer, t: Io.Timestamp) !void {
    const ms: u64 = @intCast(@divTrunc(t.nanoseconds, std.time.ns_per_ms));
    const secs: std.time.epoch.EpochSeconds = .{ .secs = ms / 1000 };
    const year_day = secs.getEpochDay().calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day = secs.getDaySeconds();
    try w.print("{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}.{d:0>3}Z", .{
        year_day.year,         month_day.month.numeric(), month_day.day_index + 1,
        day.getHoursIntoDay(), day.getMinutesIntoHour(),  day.getSecondsIntoMinute(),
        ms % 1000,
    });
}

var test_log: ?TestLog = null;

/// `SUSHI_TEST_SELECT`: one test-name prefix a line, and `!<prefix>` lines that drop what they match; the process runs
/// only the tests it selects, so several run steps of one binary each run their own. Unset, every test runs.
var select: ?[]const u8 = null;
/// The selected tests' indexes in `builtin.test_functions`, in the order the build runner was told them.
var selected_tests: []u32 = &.{};

fn isSelected(name: []const u8) bool {
    const lines = select orelse return true;
    var kept = false;
    var it = std.mem.splitScalar(u8, lines, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        if (line[0] == '!') {
            if (std.mem.startsWith(u8, name, line[1..])) return false;
        } else if (std.mem.startsWith(u8, name, line)) kept = true;
    }
    return kept;
}

/// The runner's own copy of the process environment, array and strings. A test's setenv or unsetenv rebuilds
/// libc's environment and can free the array (and the strings libc made) that the start block points into, so every
/// later read the runner and each test's Io make goes through this copy.
fn ownedEnviron(start: std.process.Environ) std.process.Environ {
    if (comptime std.process.Environ.Block != std.process.Environ.PosixBlock) return start;
    const gpa = std.heap.page_allocator;
    var n: usize = 0;
    while (std.c.environ[n] != null) n += 1;
    const copy = gpa.allocSentinel(?[*:0]const u8, n, null) catch @panic("OOM");
    for (copy, std.c.environ[0..n]) |*out, entry| out.* = (gpa.dupeSentinel(u8, std.mem.span(entry.?), 0) catch @panic("OOM")).ptr;
    return .{ .block = .{ .slice = copy } };
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn getdtablesize() c_int;
extern "c" fn close(fd: c_int) c_int;

/// Closes every descriptor past stdio except the progress pipe the build runner passes on purpose (`ZIG_PROGRESS`).
/// The build runner starts steps from several threads, and a pipe's CLOEXEC is a second call after `pipe()` on macOS,
/// so a process can inherit another step's pipe; that step's stdout then reaches EOF only when this process ends, and
/// the build runner kills it as unresponsive. The runner opens nothing before this.
fn closeInheritedFds() void {
    const progress: c_int = if (std.c.getenv("ZIG_PROGRESS")) |v| std.fmt.parseInt(c_int, std.mem.span(v), 10) catch -1 else -1;
    var fd: c_int = 3;
    while (fd < getdtablesize()) : (fd += 1) {
        if (fd != progress) _ = close(fd);
    }
}

/// The run step's own variables arrive as `--env=NAME=VALUE`, not in the environment the build runner hands the process:
/// a run step's environment is a snapshot taken at configure, and a cached configure replays it, so a variable set at
/// this invocation would never reach the test. The process keeps the invocation's environment as it is now and takes
/// the step's on top; a name the invocation also sets to another value is refused, naming it.
fn applyStepEnv(args: []const [:0]const u8) void {
    for (args) |arg| {
        const pair = std.mem.cutPrefix(u8, arg, "--env=") orelse continue;
        const eq = std.mem.indexOfScalar(u8, pair, '=') orelse panic("{s}: expected --env=NAME=VALUE", .{arg});
        const name = std.heap.page_allocator.dupeSentinel(u8, pair[0..eq], 0) catch @panic("OOM");
        const value = std.heap.page_allocator.dupeSentinel(u8, pair[eq + 1 ..], 0) catch @panic("OOM");
        if (std.c.getenv(name)) |set| if (!std.mem.eql(u8, std.mem.span(set), value)) {
            std.debug.print("{s} is set to '{s}' at this invocation, but the run step sets it to '{s}': unset it (the step owns it)\n", .{ name, std.mem.span(set), value });
            std.process.exit(1);
        };
        if (setenv(name, value, 1) != 0) panic("setenv {s} failed", .{name});
    }
}

/// A step that can load the Qwen cut gets `SUSHI_TEST_FIXTURE_CUT`: `1` when the build declared the cut's memory
/// (`-Dfixture-cut=true`), `0` when not, so a `0` step that sees QWEN4_TEST_MODEL fails before its first test.
fn refuseUndeclaredCut(environ: std.process.Environ, step: ?[]const u8) void {
    const declared = environ.getPosix("SUSHI_TEST_FIXTURE_CUT") orelse return;
    const model = environ.getPosix("QWEN4_TEST_MODEL") orelse return;
    if (model.len == 0 or !std.mem.eql(u8, declared, "0")) return;
    std.debug.print("step '{s}': QWEN4_TEST_MODEL is set, but the build did not declare it: zig build test -Dfixture-cut=true\n", .{step orelse "(no SUSHI_TEST_STEP)"});
    std.process.exit(1);
}

/// Fills `selected_tests`; a selection that keeps no test fails the process, naming its step.
fn selectTests(gpa: std.mem.Allocator, step: ?[]const u8) void {
    var kept: std.ArrayList(u32) = .empty;
    for (builtin.test_functions, 0..) |test_fn, i| if (isSelected(test_fn.name))
        kept.append(gpa, @intCast(i)) catch @panic("OOM");
    if (select != null and kept.items.len == 0) {
        std.debug.print("step '{s}': SUSHI_TEST_SELECT selects no test of this binary\n", .{step orelse "(no SUSHI_TEST_STEP)"});
        std.process.exit(1);
    }
    selected_tests = kept.items;
}
/// The milliseconds of the last `runTest`, its at-rest check included.
var last_test_ms: i64 = 0;

/// The test, then the engine put back at rest; a test that passed but left it otherwise fails, naming what it left.
fn runTest(func: *const fn () anyerror!void) anyerror!void {
    const start: Io.Timestamp = .now(runner_threaded_io, .awake);
    if (test_log) |*l| if (l.first == null) {
        l.first = start;
    };
    const result = func();
    const left = sushi_after_each();
    last_test_ms = start.untilNow(runner_threaded_io, .awake).toMilliseconds();
    try result;
    if (left) |why| {
        std.debug.print("{s}\n", .{why});
        return error.EngineNotAtRest;
    }
}

/// Logs a test that ended as `s`; a log that cannot be written fails the test, naming the file.
fn logTest(name: []const u8, s: TestLog.Status) bool {
    const note = sushi_test_note();
    const l = &(test_log orelse return true);
    l.record(name, s, last_test_ms, note) catch |err| {
        std.debug.print("SUSHI_TEST_LOG={s}: {t}\n", .{ l.path, err });
        return false;
    };
    return true;
}

fn logDone() void {
    if (test_log) |*l| l.done() catch |err| std.debug.print("SUSHI_TEST_LOG={s}: {t}\n", .{ l.path, err });
}

pub const std_options: std.Options = .{
    .logFn = log,
};

var log_err_count: usize = 0;
var stdin_buffer: [4096]u8 = undefined;
var stdout_buffer: [4096]u8 = undefined;
var stdin_reader: Io.File.Reader = undefined;
var stdout_writer: Io.File.Writer = undefined;
const runner_threaded_io: Io = Io.Threaded.global_single_threaded.io();

/// Keep in sync with logic in `std.Build.addRunArtifact` which decides whether
/// the test runner will communicate with the build runner via `std.zig.Server`.
const need_simple = switch (builtin.zig_backend) {
    .stage2_aarch64,
    .stage2_loongarch,
    .stage2_powerpc,
    .stage2_riscv64,
    => true,
    else => false,
};

pub fn main(start: std.process.Init.Minimal) void {
    @disableInstrumentation();
    if (builtin.cpu.arch.isSpirV()) {
        // SPIR-V needs an special test-runner
        return;
    }

    if (need_simple) {
        return mainSimple() catch |err| panic("test failure: {t}", .{err});
    }

    closeInheritedFds();
    const args = start.args.toSlice(std.heap.page_allocator) catch |err| panic("unable to parse command line args: {t}", .{err});
    applyStepEnv(args[1..]);
    const init: std.process.Init.Minimal = .{ .environ = ownedEnviron(start.environ), .args = start.args };

    var listen = false;
    var opt_cache_dir: ?[]const u8 = null;

    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--listen=-")) {
            listen = true;
        } else if (std.mem.startsWith(u8, arg, "--seed=")) {
            testing.random_seed = std.fmt.parseUnsigned(u32, arg["--seed=".len..], 0) catch
                @panic("unable to parse --seed command line argument");
        } else if (std.mem.startsWith(u8, arg, "--cache-dir")) {
            opt_cache_dir = arg["--cache-dir=".len..];
        } else if (std.mem.startsWith(u8, arg, "--env=")) {
            // Taken by `applyStepEnv` above.
        } else {
            panic("unrecognized command line argument: {s}", .{arg});
        }
    }

    if (init.environ.getPosix("SUSHI_TEST_LOG")) |path| {
        if (!std.fs.path.isAbsolute(path)) panic("SUSHI_TEST_LOG must be an absolute path, not {s}", .{path});
        const step = init.environ.getPosix("SUSHI_TEST_STEP") orelse
            panic("SUSHI_TEST_LOG names no step: zig build test -Dtest-log=<path> sets SUSHI_TEST_STEP beside it", .{});
        test_log = .{ .path = path, .step = step };
    }
    select = init.environ.getPosix("SUSHI_TEST_SELECT");
    selectTests(std.heap.page_allocator, init.environ.getPosix("SUSHI_TEST_STEP"));

    if (builtin.fuzz) {
        const cache_dir = opt_cache_dir orelse @panic("missing --cache-dir=[path] argument");
        fuzz_abi.fuzzer_init(.fromSlice(cache_dir));
    }

    if (listen) {
        return mainServer(init) catch |err| panic("internal test runner failure: {t}", .{err});
    } else {
        return mainTerminal(init);
    }
}

fn mainServer(init: std.process.Init.Minimal) !void {
    @disableInstrumentation();
    stdin_reader = .initStreaming(.stdin(), runner_threaded_io, &stdin_buffer);
    stdout_writer = .initStreaming(.stdout(), runner_threaded_io, &stdout_buffer);
    var server: std.zig.Server = .{
        .in = &stdin_reader.interface,
        .out = &stdout_writer.interface,
    };
    try server.serveStringMessage(.zig_version, builtin.zig_version_string);

    while (true) {
        const hdr = try server.receiveMessage();
        switch (hdr.tag) {
            .exit => {
                const ok = afterAll();
                logDone();
                return std.process.exit(if (ok) 0 else 1);
            },
            .query_test_metadata => {
                var sa: std.heap.SafeAllocator = .init(std.heap.page_allocator, .{});
                defer if (sa.deinit() != 0) @panic("internal test runner memory leak");
                const gpa = sa.allocator();

                var string_bytes: std.ArrayList(u8) = .empty;
                defer string_bytes.deinit(gpa);
                try string_bytes.append(gpa, 0); // Reserve 0 for null.

                const names = try gpa.alloc(u32, selected_tests.len);
                defer gpa.free(names);
                const expected_panic_msgs = try gpa.alloc(u32, selected_tests.len);
                defer gpa.free(expected_panic_msgs);

                for (selected_tests, names, expected_panic_msgs) |fn_index, *name, *expected_panic_msg| {
                    const test_fn = builtin.test_functions[fn_index];
                    name.* = @intCast(string_bytes.items.len);
                    try string_bytes.ensureUnusedCapacity(gpa, test_fn.name.len + 1);
                    string_bytes.appendSliceAssumeCapacity(test_fn.name);
                    string_bytes.appendAssumeCapacity(0);
                    expected_panic_msg.* = 0;
                }

                try server.serveTestMetadata(.{
                    .names = names,
                    .expected_panic_msgs = expected_panic_msgs,
                    .string_bytes = string_bytes.items,
                });
            },

            .run_test => {
                testing.environ = init.environ;
                testing.allocator_instance = .init(std.heap.page_allocator, .{
                    .canary = 0xc3a701ba,
                    .check_write_after_free = true,
                });
                testing.io_instance = .init(testing.allocator, .{
                    .argv0 = .init(init.args),
                    .environ = init.environ,
                });
                log_err_count = 0;
                const index = try server.receiveBody_u32();
                refuseUndeclaredCut(init.environ, init.environ.getPosix("SUSHI_TEST_STEP"));
                const test_fn = builtin.test_functions[selected_tests[index]];
                is_fuzz_test = false;

                // let the build server know we're starting the test now
                try server.serveStringMessage(.test_started, &.{});

                const TestResults = std.zig.Server.Message.TestResults;
                var status: TestResults.Status = if (runTest(test_fn.func)) |v| s: {
                    v;
                    break :s .pass;
                } else |err| switch (err) {
                    error.SkipZigTest => .skip,
                    else => s: {
                        if (@errorReturnTrace()) |trace| {
                            std.debug.dumpErrorReturnTrace(trace);
                        }
                        break :s .fail;
                    },
                };
                testing.io_instance.deinit();
                const leak_count = testing.allocator_instance.deinit();
                if (!logTest(test_fn.name, TestLog.status(status == .pass, status == .skip, leak_count != 0, log_err_count != 0)))
                    status = .fail;
                try server.serveTestResults(.{
                    .index = index,
                    .flags = .{
                        .status = status,
                        .fuzz = is_fuzz_test,
                        .log_err_count = std.math.lossyCast(
                            @FieldType(TestResults.Flags, "log_err_count"),
                            log_err_count,
                        ),
                        .leak_count = std.math.lossyCast(
                            @FieldType(TestResults.Flags, "leak_count"),
                            leak_count,
                        ),
                    },
                });
            },
            .start_fuzzing => {
                // This ensures that this code won't be analyzed and hence reference fuzzer symbols
                // since they are not present.
                if (!builtin.fuzz) unreachable;

                var gpa_instance: std.heap.SafeAllocator = .init(std.heap.page_allocator, .{});
                defer if (gpa_instance.deinit() != 0) {
                    @panic("internal test runner memory leak");
                };
                const gpa = gpa_instance.allocator();
                var io_instance: Io.Threaded = .init(gpa, .{
                    .argv0 = .init(init.args),
                    .environ = init.environ,
                });
                defer io_instance.deinit();

                const mode: fuzz_abi.LimitKind = @fromBackingInt(@intCast(try server.receiveBody_u8()));
                const amount_or_instance = try server.receiveBody_u64();
                const main_instance = mode == .iterations or amount_or_instance == 0;

                if (main_instance) {
                    const coverage = fuzz_abi.fuzzer_coverage();
                    try server.serveCoverageIdMessage(
                        coverage.id,
                        coverage.runs,
                        coverage.unique,
                        coverage.seen,
                    );
                }

                const n_tests: u32 = try server.receiveBody_u32();
                const test_indexes = try gpa.alloc(u32, n_tests);
                defer gpa.free(test_indexes);
                fuzz_runner = .{
                    .indexes = test_indexes,
                    .server = &server,
                    .gpa = gpa,
                    .threaded_io = &io_instance,
                    .input_poller = undefined,
                };

                {
                    var large_name_buf: std.ArrayList(u8) = .empty;
                    defer large_name_buf.deinit(gpa);
                    for (test_indexes) |*i| {
                        const name_len = try server.receiveBody_u32();
                        const name = if (name_len <= server.in.buffer.len)
                            try server.in.take(name_len)
                        else large_name: {
                            try large_name_buf.resize(gpa, name_len);
                            try server.in.readSliceAll(large_name_buf.items);
                            break :large_name large_name_buf.items;
                        };

                        for (0.., builtin.test_functions) |test_i, test_fn| {
                            if (std.mem.eql(u8, name, test_fn.name)) {
                                i.* = @intCast(test_i);
                                break;
                            }
                        } else {
                            panic("fuzz test {s} no longer exists", .{name});
                        }

                        if (main_instance) {
                            const relocated_entry_addr = @intFromPtr(builtin.test_functions[i.*].func);
                            const entry_addr = fuzz_abi.fuzzer_unslide_address(relocated_entry_addr);
                            try server.serveU64Message(.fuzz_start_addr, entry_addr);
                        }
                    }
                }

                fuzz_abi.fuzzer_main(n_tests, testing.random_seed, mode, amount_or_instance);

                assert(mode != .forever);
                std.process.exit(0);
            },

            else => {
                std.debug.print("unsupported message: {x}\n", .{@backingInt(hdr.tag)});
                std.process.exit(1);
            },
        }
    }
}

fn mainTerminal(init: std.process.Init.Minimal) void {
    @disableInstrumentation();
    if (builtin.fuzz) @panic("fuzz test requires server");

    var ok_count: usize = 0;
    var skip_count: usize = 0;
    var fail_count: usize = 0;
    var fuzz_count: usize = 0;
    const root_node = if (builtin.fuzz) std.Progress.Node.none else std.Progress.start(runner_threaded_io, .{
        .root_name = "Test",
        .estimated_total_items = selected_tests.len,
    });
    const have_tty = Io.File.stderr().isTty(runner_threaded_io) catch unreachable;

    var leaks: usize = 0;
    for (selected_tests, 0..) |fn_index, i| {
        const test_fn = builtin.test_functions[fn_index];
        testing.allocator_instance = .init(std.heap.page_allocator, .{
            .canary = 0xc3a701ba,
            .check_write_after_free = true,
        });
        testing.io_instance = .init(testing.allocator, .{
            .argv0 = .init(init.args),
            .environ = init.environ,
        });
        var ran: TestLog.Status = .FAIL;
        const errors_before = log_err_count;
        defer {
            testing.io_instance.deinit();
            const leaked = testing.allocator_instance.deinit() != 0;
            if (leaked) leaks += 1;
            if (!logTest(test_fn.name, TestLog.status(ran == .PASS, ran == .SKIP, leaked, log_err_count != errors_before)))
                fail_count += 1;
        }
        testing.log_level = .warn;
        testing.environ = init.environ;

        const test_node = root_node.start(test_fn.name, 0);
        if (!have_tty) {
            std.debug.print("{d}/{d} {s}...", .{ i + 1, selected_tests.len, test_fn.name });
        }
        is_fuzz_test = false;
        if (runTest(test_fn.func)) |_| {
            ran = .PASS;
            ok_count += 1;
            test_node.end();
            if (!have_tty) std.debug.print("OK\n", .{});
        } else |err| switch (err) {
            error.SkipZigTest => {
                ran = .SKIP;
                skip_count += 1;
                if (have_tty) {
                    std.debug.print("{d}/{d} {s}...SKIP\n", .{ i + 1, selected_tests.len, test_fn.name });
                } else {
                    std.debug.print("SKIP\n", .{});
                }
                test_node.end();
            },
            else => {
                fail_count += 1;
                if (have_tty) {
                    std.debug.print("{d}/{d} {s}...FAIL ({t})\n", .{
                        i + 1, selected_tests.len, test_fn.name, err,
                    });
                } else {
                    std.debug.print("FAIL ({t})\n", .{err});
                }
                if (@errorReturnTrace()) |trace| {
                    std.debug.dumpErrorReturnTrace(trace);
                }
                test_node.end();
            },
        }
        fuzz_count += @intFromBool(is_fuzz_test);
    }
    root_node.end();
    if (!afterAll()) fail_count += 1;
    logDone();
    if (ok_count == selected_tests.len) {
        std.debug.print("All {d} tests passed.\n", .{ok_count});
    } else {
        std.debug.print("{d} passed; {d} skipped; {d} failed.\n", .{ ok_count, skip_count, fail_count });
    }
    if (log_err_count != 0) {
        std.debug.print("{d} errors were logged.\n", .{log_err_count});
    }
    if (leaks != 0) {
        std.debug.print("{d} tests leaked memory.\n", .{leaks});
    }
    if (fuzz_count != 0) {
        std.debug.print("{d} fuzz tests found.\n", .{fuzz_count});
    }
    if (leaks != 0 or log_err_count != 0 or fail_count != 0) {
        std.process.exit(1);
    }
}

pub fn log(
    comptime message_level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime format: []const u8,
    args: anytype,
) void {
    @disableInstrumentation();
    if (@backingInt(message_level) <= @backingInt(std.log.Level.err)) {
        log_err_count +|= 1;
    }
    if (@backingInt(message_level) <= @backingInt(testing.log_level)) {
        std.debug.print(
            "[" ++ @tagName(scope) ++ "] (" ++ @tagName(message_level) ++ "): " ++ format ++ "\n",
            args,
        );
    }
}

/// Simpler main(), exercising fewer language features, so that
/// work-in-progress backends can handle it.
pub fn mainSimple() anyerror!void {
    @disableInstrumentation();
    // is the backend capable of calling `Io.File.writeAll`?
    const enable_write = switch (builtin.zig_backend) {
        .stage2_aarch64, .stage2_riscv64 => true,
        else => false,
    };
    // is the backend capable of calling `Io.Writer.print`?
    const enable_print = switch (builtin.zig_backend) {
        .stage2_aarch64, .stage2_riscv64 => true,
        else => false,
    };

    testing.allocator_instance = .init(std.heap.page_allocator, .{});
    testing.io_instance = .init(testing.allocator, .{});

    var passed: u64 = 0;
    var skipped: u64 = 0;
    var failed: u64 = 0;

    // we don't want to bring in File and Writer if the backend doesn't support it
    const stdout = if (enable_write) Io.File.stdout() else {};

    for (builtin.test_functions) |test_fn| {
        if (enable_write) {
            stdout.writeStreamingAll(runner_threaded_io, test_fn.name) catch {};
            stdout.writeStreamingAll(runner_threaded_io, "... ") catch {};
        }
        if (runTest(test_fn.func)) |_| {
            if (enable_write) stdout.writeStreamingAll(runner_threaded_io, "PASS\n") catch {};
        } else |err| {
            if (err != error.SkipZigTest) {
                if (enable_write) stdout.writeStreamingAll(runner_threaded_io, "FAIL\n") catch {};
                failed += 1;
                if (!enable_write) return err;
                continue;
            }
            if (enable_write) stdout.writeStreamingAll(runner_threaded_io, "SKIP\n") catch {};
            skipped += 1;
            continue;
        }
        passed += 1;
    }
    if (enable_print) {
        var unbuffered_stdout_writer = stdout.writer(runner_threaded_io, &.{});
        unbuffered_stdout_writer.interface.print(
            "{} passed, {} skipped, {} failed\n",
            .{ passed, skipped, failed },
        ) catch {};
    }
    if (failed != 0) std.process.exit(1);
}

var is_fuzz_test: bool = undefined;
var fuzz_runner: if (builtin.fuzz) struct {
    indexes: []u32,
    server: *std.zig.Server,
    gpa: std.mem.Allocator,
    threaded_io: *Io.Threaded,
    input_poller: Io.Future(Io.Cancelable!void),

    comptime {
        assert(builtin.fuzz); // `fuzz_runner` was analyzed in non-fuzzing compilation
    }

    export fn runner_test_run(i: u32) void {
        @disableInstrumentation();

        fuzz_runner.server.serveU32Message(.fuzz_test_change, i) catch |e| switch (e) {
            error.WriteFailed => panic("failed to write to stdout: {t}", .{stdout_writer.err.?}),
        };

        testing.allocator_instance = .init(std.heap.page_allocator, .{
            .canary = 0xc3a701ba,
            .check_write_after_free = true,
        });
        defer if (testing.allocator_instance.deinit() != 0) std.process.exit(1);
        is_fuzz_test = false;

        testing.io_instance = .init(testing.allocator, .{
            .argv0 = fuzz_runner.threaded_io.argv0,
            .environ = fuzz_runner.threaded_io.environ.process_environ,
        });
        defer testing.io_instance.deinit();

        builtin.test_functions[fuzz_runner.indexes[i]].func() catch |err| switch (err) {
            error.SkipZigTest => return,
            else => {
                if (@errorReturnTrace()) |trace| {
                    std.debug.dumpErrorReturnTrace(trace);
                }
                std.debug.print("failed with error.{t}\n", .{err});
                std.process.exit(1);
            },
        };

        if (!is_fuzz_test) @panic("missed call to std.testing.fuzz");
        if (log_err_count != 0) @panic("error logs detected");
    }

    export fn runner_test_name(i: u32) fuzz_abi.Slice {
        @disableInstrumentation();
        return .fromSlice(builtin.test_functions[fuzz_runner.indexes[i]].name);
    }

    export fn runner_broadcast_input(test_i: u32, bytes_slice: fuzz_abi.Slice) void {
        @disableInstrumentation();
        const bytes = bytes_slice.toSlice();
        fuzz_runner.server.serveBroadcastFuzzInputMessage(test_i, bytes) catch |e| switch (e) {
            error.WriteFailed => panic("failed to write to stdout: {t}", .{stdout_writer.err.?}),
        };
    }

    export fn runner_start_input_poller() void {
        @disableInstrumentation();
        const io = fuzz_runner.threaded_io.io();
        const future = io.concurrent(inputPoller, .{}) catch |e| switch (e) {
            error.ConcurrencyUnavailable => @panic("failed to spawn concurrent fuzz input poller"),
        };
        fuzz_runner.input_poller = future;
    }

    export fn runner_stop_input_poller() void {
        @disableInstrumentation();
        const io = fuzz_runner.threaded_io.io();
        assert(fuzz_runner.input_poller.cancel(io) == error.Canceled);
    }

    export fn runner_futex_wait(ptr: *const u32, expected: u32) bool {
        @disableInstrumentation();
        const io = fuzz_runner.threaded_io.io();
        return io.futexWait(u32, ptr, expected) == error.Canceled;
    }

    export fn runner_futex_wake(ptr: *const u32, waiters: u32) void {
        @disableInstrumentation();
        const io = fuzz_runner.threaded_io.io();
        io.futexWake(u32, ptr, waiters);
    }

    fn inputPoller() Io.Cancelable!void {
        @disableInstrumentation();
        switch (inputPollerInner()) {
            error.Canceled => |e| return e,
            error.ReadFailed => {
                if (stdin_reader.err.? == error.Canceled) return error.Canceled;
                panic("failed to read from stdin: {t}", .{stdin_reader.err.?});
            },
            error.EndOfStream => @panic("unexpected end of stdin"),
        }
    }

    fn inputPollerInner() (Io.Cancelable || Io.Reader.Error) {
        @disableInstrumentation();
        const server = fuzz_runner.server;
        var large_bytes_list: std.ArrayList(u8) = .empty;
        defer large_bytes_list.deinit(fuzz_runner.gpa);
        while (true) {
            const hdr = try server.receiveMessage();
            if (hdr.tag != .new_fuzz_input) {
                panic("unexpected message: {x}\n", .{@backingInt(hdr.tag)});
            }
            const test_i = try server.receiveBody_u32();
            const input_len = hdr.bytes_len - 4;
            const bytes = if (input_len <= server.in.buffer.len)
                try server.in.take(input_len)
            else bytes: {
                large_bytes_list.resize(fuzz_runner.gpa, @intCast(input_len)) catch @panic("OOM");
                try server.in.readSliceAll(large_bytes_list.items);
                break :bytes large_bytes_list.items;
            };
            if (fuzz_abi.fuzzer_receive_input(test_i, .fromSlice(bytes))) {
                return error.Canceled;
            }
        }
    }
} else void = undefined;

pub fn fuzz(
    context: anytype,
    comptime testOne: fn (context: @TypeOf(context), *std.testing.Smith) anyerror!void,
    options: testing.FuzzInputOptions,
) anyerror!void {
    // Prevent this function from confusing the fuzzer by omitting its own code
    // coverage from being considered.
    @disableInstrumentation();

    // Some compiler backends are not capable of handling fuzz testing yet but
    // we still want CI test coverage enabled.
    if (need_simple) return;

    // Smoke test to ensure the test did not use conditional compilation to
    // contradict itself by making it not actually be a fuzz test when the test
    // is built in fuzz mode.
    is_fuzz_test = true;

    // Ensure no test failure occurred before starting fuzzing.
    if (log_err_count != 0) @panic("error logs detected");

    // libfuzzer is in a separate compilation unit so that its own code can be
    // excluded from code coverage instrumentation. It needs a function pointer
    // it can call for checking exactly one input. Inside this function we do
    // our standard unit test checks such as memory leaks, and interaction with
    // error logs.
    const global = struct {
        var ctx: @TypeOf(context) = undefined;

        fn test_one() callconv(.c) bool {
            @disableInstrumentation();
            testing.allocator_instance = .init(std.heap.page_allocator, .{
                .canary = 0xcacce5e0,
                .check_write_after_free = true,
            });
            defer if (testing.allocator_instance.deinit() != 0) std.process.exit(1);
            log_err_count = 0;
            testOne(ctx, @constCast(&testing.Smith{ .in = null })) catch |err| switch (err) {
                error.SkipZigTest => return true,
                else => {
                    const stderr = std.debug.lockStderr(&.{}).terminal();
                    p: {
                        if (@errorReturnTrace()) |trace| {
                            std.debug.writeErrorReturnTrace(trace, stderr) catch break :p;
                        }
                        stderr.writer.print("failed with error.{t}\n", .{err}) catch break :p;
                    }
                    std.process.exit(1);
                },
            };
            if (log_err_count != 0) {
                const stderr = std.debug.lockStderr(&.{}).terminal();
                stderr.writer.print("error logs detected\n", .{}) catch {};
                std.process.exit(1);
            }
            return false;
        }
    };

    if (builtin.fuzz) {
        // Preserve the calling test's allocator state
        const prev_allocator_state = testing.allocator_instance;
        defer testing.allocator_instance = prev_allocator_state;

        global.ctx = context;
        fuzz_abi.fuzzer_set_test(&global.test_one);
        for (options.corpus) |elem|
            fuzz_abi.fuzzer_new_input(.fromSlice(elem));
        fuzz_abi.fuzzer_start_test();
        return;
    }

    // When the unit test executable is not built in fuzz mode, only run the
    // provided corpus.
    for (options.corpus) |input| {
        var smith: testing.Smith = .{ .in = input };
        try testOne(context, &smith);
    }

    // In case there is no provided corpus, also use an empty
    // string as a smoke test.
    var smith: testing.Smith = .{ .in = "" };
    try testOne(context, &smith);
}
