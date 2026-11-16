//! The Sushi Cooler contract: what every load on the engine of our own owes, whatever its arch. An arch on Sushi
//! Cooler declares `cooler_contract` (its tiny fixtures by env var and the reads only it can make: `prepare`,
//! `prompt`, `weightsOutsideRegions`, `headLoaded`, optionally `specTokens` and `specTokensWith` (sampled), and
//! `greedyRows`, required when its `coolerRows` gives more than one (the rows test fails by name without it); each test
//! runs on every such fixture present and is reported skipped when any is missing or when no fixture's load applied to
//! it. Loads, requests and unloads take the turns and hooks serve takes.
const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const Trunk = @import("trunk.zig").Trunk;
const registry = @import("arch/registry.zig");
const arch_config = @import("arch/config.zig");
const arch_handle = @import("arch/handle.zig");
const test_support = @import("test_support.zig");
const model_load = @import("model_load.zig");
const arch_serve = @import("arch/serve.zig");
const arch_sched = @import("arch/sched.zig");
const arch_cooler = @import("arch/cooler.zig");
const prefill_plan = @import("prefill_plan.zig");
const JobTurn = model_load.JobTurn;
const Generator = @import("generate.zig").Generator;
const Tokenizer = @import("tokenizer.zig").Tokenizer;

const io = std.Io.Threaded.global_single_threaded.io();

/// One test load of a fixture, on the side the test names. Its load and its unload run in the job turn a served load
/// and unload take (`model_load.JobTurn`).
const Load = struct {
    a: std.mem.Allocator,
    config: model_mod.ModelConfig,
    weights: model_mod.Weights,
    xfm: *Trunk,
    h: arch_handle.Handle,
    /// Off only for the MLX reference (`openMlxReference`).
    compile: bool,

    fn prepare(self: *Load, comptime C: type, dir: []const u8, cooler: bool) !void {
        return self.prepareOn(testing.allocator, C, dir, cooler);
    }

    /// A load on MLX is the one the user's opt-out flag makes, through the parse and the settings stamp a served load
    /// takes; a Cooler load lifts the tests' MLX default.
    fn prepareOn(self: *Load, a: std.mem.Allocator, comptime C: type, dir: []const u8, cooler: bool) !void {
        self.a = a;
        self.compile = true;
        const off = model_mod.cooler_off;
        defer model_mod.cooler_off = off;
        model_mod.cooler_off = if (cooler) .empty else .full;
        self.config = try model_mod.parseConfig(io, a, dir);
        errdefer self.config.deinit(a);
        model_load.applyModelSettings(&self.config, .{});
        try C.prepare(a, &self.config, dir);
        arch_config.coolerForTest(&self.config, true);
    }

    /// The load after `prepare`; on an error nothing of the load stays.
    fn load(self: *Load, dir: []const u8) !void {
        const a = self.a;
        errdefer self.config.deinit(a);
        const job = JobTurn.begin(null, &self.config);
        defer job.end();
        self.weights = try model_mod.loadWeightsForConfig(io, a, dir, &self.config, false);
        errdefer self.weights.deinit();
        self.xfm = try a.create(Trunk);
        errdefer a.destroy(self.xfm);
        self.xfm.* = try Trunk.init(io, a, self.config, &self.weights);
        if (self.compile) @import("model_load.zig").compileClosures(self.xfm, .all);
        self.h = arch_handle.adopt(self.xfm);
    }

    fn open(self: *Load, comptime C: type, dir: []const u8, cooler: bool) !void {
        try self.prepare(C, dir, cooler);
        try self.load(dir);
    }

    /// The MLX load a Cooler load's tokens are held to. MLX's compiled f32 GDN gate is ulps off its own unfused chain,
    /// which the engine equals, so the reference compiles no closure.
    fn openMlxReference(self: *Load, comptime C: type, dir: []const u8) !void {
        try self.prepare(C, dir, false);
        self.compile = false;
        try self.load(dir);
    }

    fn onEngine(self: *Load) bool {
        return arch_handle.onEngine(self.h, null);
    }

    /// The unload serve runs (`model_load.retire`, in the model's job): the model, its weights, its arch's hook. Returns
    /// the MLX calls the teardown made before the job's end.
    fn close(self: *Load) u64 {
        const calls = mlx.engine.mlxCalls();
        const asked = blk: {
            const job = JobTurn.begin(self.h, &self.config);
            defer job.end();
            arch_handle.destroy(self.h, self.a);
            self.weights.deinit();
            arch_serve.unloadModel(&self.config);
            break :blk mlx.engine.mlxCalls() - calls;
        };
        self.config.deinit(self.a);
        return asked;
    }
};

/// The engine's pool and mappings, the pool's free buffers taken back first, and the owned regions weights live in.
const Sheet = struct {
    in_use: u64,
    asked: u64,
    mapped: u64,
    regions: usize,
    owned: u64,

    fn now() Sheet {
        mlx.engine.close();
        const s = mlx.engine.stats();
        return .{
            .in_use = s.pool_bytes - s.pool_free,
            .asked = s.pool_asked,
            .mapped = s.mapped_bytes,
            .regions = model_mod.ownedRegionCount(),
            .owned = model_mod.owned_bytes.load(.monotonic),
        };
    }
};

/// What `eachFixture` saw over the fixtures. A body that skips (a load the test does not apply to) moves on to the next
/// fixture; any other error ends the test. The test is a skip when a fixture is missing or no body ran.
const Tally = struct {
    ran: bool = false,
    missing: bool = false,

    fn absent(self: *Tally) void {
        self.missing = true;
    }

    fn took(self: *Tally, result: anyerror!void) anyerror!void {
        result catch |e| {
            if (e == error.SkipZigTest) return;
            return e;
        };
        self.ran = true;
    }

    fn verdict(self: Tally) error{SkipZigTest}!void {
        if (!self.ran or self.missing) return error.SkipZigTest;
    }
};

/// `body` on each fixture of each arch that declares `cooler_contract` (`Tally` says what a skip or a missing one does).
fn eachFixture(comptime body: fn (comptime C: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    var tally: Tally = .{};
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime @hasDecl(registry.Mod(t), "cooler_contract")) {
            const C = registry.Mod(t).cooler_contract;
            for (C.fixtures) |env| {
                const dir = std.mem.span(std.c.getenv(env) orelse {
                    tally.absent();
                    continue;
                });
                const ids = C.prompt(testing.allocator, dir) catch |e| return named(env, dir, e);
                defer testing.allocator.free(ids);
                tally.took(body(C, dir, ids)) catch |e| return named(env, dir, e);
            }
        }
    }
    try tally.verdict();
}

test "cooler contract runner: a fixture whose body skips does not stop the next; the test skips only when none ran or one is missing" {
    var first_skips: Tally = .{};
    try first_skips.took(error.SkipZigTest);
    try first_skips.took({});
    try first_skips.verdict();
    var all_skip: Tally = .{};
    try all_skip.took(error.SkipZigTest);
    try all_skip.took(error.SkipZigTest);
    try testing.expectError(error.SkipZigTest, all_skip.verdict());
    var one_missing: Tally = .{};
    try one_missing.took({});
    one_missing.absent();
    try testing.expectError(error.SkipZigTest, one_missing.verdict());
    var fails: Tally = .{};
    try testing.expectError(error.TestUnexpectedResult, fails.took(error.TestUnexpectedResult));
}

/// `e`, after naming the fixture it came from.
fn named(env: []const u8, dir: []const u8, e: anyerror) anyerror {
    var msg: [512]u8 = undefined;
    std.debug.print("cooler contract: fixture {s} ({s}): {s}\n", .{ env, dir, mlx.takeError(&msg) orelse "no MLX or engine error latched" });
    return e;
}

/// `n` greedy tokens over `ids` in serial steps, one request in the model's turn, ended as serve ends one.
fn greedy(ld: *Load, ids: []const u32, n: u32) !std.ArrayList(u32) {
    const a = testing.allocator;
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    const xfm = ld.xfm;
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try xfm.resetCache();
    var gen = try Generator.initWithOptions(io, a, xfm, &tok, ids, n, .{ .temperature = 0.0 }, &.{}, .{});
    defer gen.deinit(a);
    var got: std.ArrayList(u32) = .empty;
    errdefer got.deinit(a);
    while (try gen.next(a)) |t| try got.append(a, t);
    return got;
}

test "cooler contract: a Cooler load that serves more than one row gives each row its solo request's tokens" {
    // Each row's tokens equal its request run alone: rows of ragged lengths decoded as one forward, each against the same
    // prompt run alone. A load of one row has nothing to compare and is skipped.
    try eachFixture(struct {
        fn body(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.prepare(C, dir, true);
            const rows = arch_sched.coolerRows(&ld.config);
            if (rows.rows <= 1) {
                ld.config.deinit(a);
                return error.SkipZigTest;
            }
            if (comptime !@hasDecl(C, "greedyRows")) {
                ld.config.deinit(a);
                return error.CoolerRowsWithoutGreedyRows;
            } else try rowsAgainstSolo(C, &ld, dir, ids, rows.rows);
        }

        fn rowsAgainstSolo(comptime C: type, ld: *Load, dir: []const u8, ids: []const u32, rows: usize) !void {
            const a = testing.allocator;
            try ld.load(dir);
            defer _ = ld.close();
            const n = @min(rows, 3);
            var prompts: [3][]const u32 = undefined;
            for (0..n) |i| prompts[i] = ids[0 .. ids.len - @min(ids.len - 1, i * 2)];
            // The rows run in the model's turn and end as serve ends a request: the arch's decl takes neither.
            const got = blk: {
                const turn = arch_handle.turnFor(ld.h, &ld.config);
                defer turn.end();
                defer arch_sched.endRequest(&ld.config);
                break :blk try C.greedyRows(a, ld.xfm, prompts[0..n], 6);
            };
            defer {
                for (got) |*g| g.deinit(a);
                a.free(got);
            }
            for (prompts[0..n], got) |p, row| {
                var solo = try greedy(ld, p, 6);
                defer solo.deinit(a);
                try testing.expectEqualSlices(u32, solo.items, row.items);
            }
        }
    }.body);
}

test "cooler contract: a Cooler load runs on the engine and holds every weight in an owned region, none in the forward pool" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            try testing.expect(cooler.onEngine());
            try testing.expectEqual(@as(usize, 0), try C.weightsOutsideRegions(cooler.xfm));
            try testing.expect(C.headLoaded(cooler.xfm));
        }
    }.run);
}

test "cooler contract: a Cooler load's request and unload ask nothing of MLX, and a load the user opted out to MLX says the same tokens" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const prompt = ids[0..@min(ids.len, 30)];
            var want: std.ArrayList(u32) = .empty;
            defer want.deinit(testing.allocator);
            {
                var on_mlx: Load = undefined;
                try on_mlx.openMlxReference(C, dir);
                defer _ = on_mlx.close();
                try testing.expect(!on_mlx.onEngine());
                want = try greedy(&on_mlx, prompt, 12);
            }
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            var closed = false;
            defer if (!closed) {
                _ = cooler.close();
            };
            const calls = mlx.engine.mlxCalls();
            var got = try greedy(&cooler, prompt, 12);
            defer got.deinit(testing.allocator);
            try testing.expectEqual(calls, mlx.engine.mlxCalls());
            try testing.expectEqualSlices(u32, want.items, got.items);
            // The job's end then pays MLX what it was owed while at rest: not the teardown's.
            closed = true;
            try testing.expectEqual(@as(u64, 0), cooler.close());
        }
    }.run);
}

test "cooler contract: a Cooler load refused at pool buffers sampled over its whole load frees what it made, and the next load serves the same tokens" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            defer mlx.engine.fault.disarm();
            const prompt = ids[0..@min(ids.len, 30)];
            var want: std.ArrayList(u32) = .empty;
            defer want.deinit(testing.allocator);
            {
                var first: Load = undefined;
                try first.open(C, dir, true);
                defer _ = first.close();
                want = try greedy(&first, prompt, 8);
            }
            const base = Sheet.now();
            const c0 = mlx.engine.fault.count();
            {
                var counted: Load = undefined;
                try counted.open(C, dir, true);
                _ = counted.close();
            }
            const total = mlx.engine.fault.count() - c0;
            try testing.expectEqual(base, Sheet.now());
            var msg: [300]u8 = undefined;
            var refused: usize = 0;
            var k: u64 = 1;
            while (k <= total) : (k += @max(1, total / 12)) {
                var fx: Load = undefined;
                try fx.prepare(C, dir, true);
                mlx.engine.fault.arm(k);
                const r = fx.load(dir);
                mlx.engine.fault.disarm();
                if (r) {
                    _ = fx.close();
                    errdefer std.debug.print("cooler contract: the load refused at pool buffer {d} of {d} loaded and left an error latched: {s}\n", .{ k, total, mlx.takeError(&msg) orelse "" });
                    try testing.expect(!mlx.errorPending());
                } else |_| {
                    refused += 1;
                    _ = mlx.takeError(&msg);
                }
                try testing.expect(!mlx.engine.holding());
                errdefer std.debug.print("cooler contract: the load refused at pool buffer {d} of {d} ({s}) left {any} -> {any}\n", .{ k, total, if (r) |_| "it loaded" else |_| "it failed", base, Sheet.now() });
                try testing.expectEqual(base, Sheet.now());
            }
            try testing.expect(refused >= 10);
            var after: Load = undefined;
            try after.open(C, dir, true);
            defer _ = after.close();
            var got = try greedy(&after, prompt, 8);
            defer got.deinit(testing.allocator);
            try testing.expectEqualSlices(u32, want.items, got.items);
        }
    }.run);
}

test "cooler contract: a Cooler load's speculative requests say the same requests' tokens on MLX and ask nothing of MLX, and the unload gives back everything" {
    try eachFixture(struct {
        fn spec(ld: *Load, comptime C: type, prompt: []const u32) !std.ArrayList(u32) {
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            defer arch_sched.endRequest(&ld.config);
            return C.specTokens(testing.allocator, ld.xfm, prompt, 12);
        }

        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specTokens")) return error.SkipZigTest;
            const a = testing.allocator;
            var want: [C.prompt_lengths.len]std.ArrayList(u32) = @splat(.empty);
            defer for (&want) |*w| w.deinit(a);
            {
                var on_mlx: Load = undefined;
                try on_mlx.openMlxReference(C, dir);
                defer _ = on_mlx.close();
                for (C.prompt_lengths, &want) |len, *w| w.* = try spec(&on_mlx, C, ids[0..@min(ids.len, len)]);
            }
            const base = Sheet.now();
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            {
                errdefer _ = cooler.close();
                try testing.expect(C.headLoaded(cooler.xfm));
                for (C.prompt_lengths, want) |len, w| {
                    const prompt = ids[0..@min(ids.len, len)];
                    for (0..2) |round| {
                        const calls = mlx.engine.mlxCalls();
                        var got = try spec(&cooler, C, prompt);
                        defer got.deinit(a);
                        errdefer std.debug.print("cooler contract: speculative request {d} over a prompt of {d} tokens\n", .{ round, prompt.len });
                        try testing.expectEqualSlices(u32, w.items, got.items);
                        try testing.expectEqual(calls, mlx.engine.mlxCalls());
                    }
                }
            }
            _ = cooler.close();
            try testing.expectEqual(base, Sheet.now());
        }
    }.run);
}

test "cooler contract: a Cooler load's warm that ends seals the engine, and one refused at a pool buffer fails the load" {
    try eachFixture(struct {
        fn warm(ld: *Load) !void {
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            return @import("model_load.zig").warmLoad(io, &ld.config, ld.xfm, null, null, ld.xfm.cache.config);
        }

        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            defer mlx.engine.fault.disarm();
            var total: u64 = 0;
            {
                var ld: Load = undefined;
                try ld.open(C, dir, true);
                defer _ = ld.close();
                const c0 = mlx.engine.fault.count();
                try warm(&ld);
                total = mlx.engine.fault.count() - c0;
                try testing.expect(mlx.engine.sealed());
            }
            // An arch that warms inside its init (DeepSeek) fails its load there; its served warm makes nothing more.
            if (total < 2) return;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            mlx.engine.fault.arm(total / 2);
            const r = warm(&ld);
            mlx.engine.fault.disarm();
            var msg: [300]u8 = undefined;
            _ = mlx.takeError(&msg);
            errdefer std.debug.print("cooler contract: the warm refused at pool buffer {d} of {d} served on\n", .{ total / 2, total });
            try testing.expectError(error.MlxError, r);
        }
    }.run);
}

test "cooler contract: a Cooler prefill's chunks before the tail project no logits" {
    // An eager engine evaluates what MLX's lazy graph drops: a chunk's logits are rows x vocab x 2 bytes.
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            var long: [700]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
            const before = @import("trunk.zig").head_rows_for_test;
            var got = try greedy(&ld, &long, 2);
            got.deinit(testing.allocator);
            const rows = @import("trunk.zig").head_rows_for_test - before;
            // An arch with a head of its own projects none through the trunk's.
            if (rows == 0) return error.SkipZigTest;
            errdefer std.debug.print("cooler contract: a 700-token prompt and 2 tokens projected {d} head rows\n", .{rows});
            try testing.expect(rows <= prefill_plan.SSM_SNAPSHOT_BACKOFF + 1 + 2);
        }
    }.run);
}

test "cooler contract: a Cooler request ends with its pass closed and at most its resting cap free in the pool, and the unload leaves the pool as the load found it, nothing free" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const before = Sheet.now();
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            {
                errdefer _ = ld.close();
                // Lengths rising: each new one takes buffers of new sizes, which a pool without an owner keeps.
                var long: [2000]u32 = undefined;
                for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
                for ([_]usize{ 1, 40, 300, 700, 1500, 2000 }) |len| {
                    var got = try greedy(&ld, long[0..len], 4);
                    got.deinit(testing.allocator);
                    const st = mlx.engine.stats();
                    const rest = arch_cooler.restCap(&ld.config);
                    errdefer std.debug.print("cooler contract: after a request over {d} tokens, {d} dispatches open, {d} B free in the pool beside {d} B asked, resting cap {d} B\n", .{ len, mlx.engine.openDispatches(), st.pool_free, st.pool_asked, rest });
                    try testing.expectEqual(@as(u32, 0), mlx.engine.openDispatches());
                    try testing.expect(st.pool_bytes <= st.pool_asked +| rest);
                }
            }
            _ = ld.close();
            const st = mlx.engine.stats();
            errdefer std.debug.print("cooler contract: after the unload the pool holds {d} B, {d} B of them free, {d} B in use before the load\n", .{ st.pool_bytes, st.pool_free, before.in_use });
            try testing.expectEqual(@as(u64, 0), st.pool_free);
            try testing.expectEqual(before.in_use, st.pool_bytes);
        }
    }.run);
}

test "cooler contract: a Cooler load's prefix cache keeps its arrays on the engine, and an MLX load's on MLX" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            const prefix_arch = @import("arch/prefix.zig");
            inline for (.{ true, false }) |cool| {
                var ld: Load = undefined;
                try ld.open(C, dir, cool);
                defer _ = ld.close();
                const on = prefix_arch.arraysOn(&ld.config);
                errdefer std.debug.print("cooler contract: a load {s} keeps its prefix arrays on {s}\n", .{ if (cool) "on Cooler" else "on MLX", if (on == null) "MLX" else "the engine" });
                if (cool) try testing.expect(on != null and mlx.engine.isStream(on.?)) else try testing.expect(on == null);
            }
        }
    }.run);
}

test "cooler contract: a Cooler load's sampled speculative requests, seeded or not, serve and ask nothing of MLX" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specTokensWith")) return error.SkipZigTest;
            const a = testing.allocator;
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            const cases = [_]@import("sampler.zig").SamplingParams{
                .{ .temperature = 0.8, .top_p = 0.95 },
                .{ .temperature = 0.8, .top_p = 0.95, .seed = 7 },
                .{ .temperature = 1.0, .top_k = 20 },
            };
            for (cases) |sampling| for (C.prompt_lengths) |len| {
                const prompt = ids[0..@min(ids.len, len)];
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                const calls = mlx.engine.mlxCalls();
                errdefer std.debug.print("cooler contract: sampled speculative request over {d} tokens, seed {?d}, top_k {d}\n", .{ prompt.len, sampling.seed, sampling.top_k });
                var got = try C.specTokensWith(a, cooler.xfm, prompt, 12, sampling);
                defer got.deinit(a);
                try testing.expectEqual(@as(usize, 12), got.items.len);
                try testing.expectEqual(calls, mlx.engine.mlxCalls());
            };
        }
    }.run);
}

test "cooler contract: after a Cooler load, requests at the prompt lengths the arch declares compile no kernel" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            const late = mlx.engine.latePipelines();
            for (C.prompt_lengths) |len| {
                const prompt = ids[0..@min(ids.len, len)];
                var got = try greedy(&cooler, prompt, 6);
                got.deinit(testing.allocator);
                if (@hasDecl(C, "specTokens")) {
                    const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                    defer turn.end();
                    defer arch_sched.endRequest(&cooler.config);
                    var spec = try C.specTokens(testing.allocator, cooler.xfm, prompt, 6);
                    spec.deinit(testing.allocator);
                }
                errdefer std.debug.print("cooler contract: prompt of {d} tokens\n", .{prompt.len});
                try testing.expectEqual(late, mlx.engine.latePipelines());
            }
        }
    }.run);
}

/// One Cooler load of `dir` with its `k`-th checked MLX call failed, on an allocator of its own: whether anything of it
/// stays (the allocator's, the engine's or a hold), and whether the fault fired.
fn faultedLoad(comptime C: type, dir: []const u8, k: u64) !struct { fired: bool, left: bool } {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const base = Sheet.now();
    var fx: Load = undefined;
    try fx.prepareOn(gpa.allocator(), C, dir, true);
    mlx.fault.arm(k);
    const r = fx.load(dir);
    mlx.fault.disarm();
    const fired = mlx.fault.didFire();
    if (r) {
        _ = fx.close();
    } else |_| {
        var msg: [300]u8 = undefined;
        _ = mlx.takeError(&msg);
    }
    const leaked = gpa.deinit() == .leak;
    const after = Sheet.now();
    const left = leaked or mlx.engine.holding() or !std.meta.eql(base, after);
    if (left) std.debug.print("cooler contract: the load failed at checked MLX call {d} left: allocator {}, hold {}, {any} -> {any}\n", .{ k, leaked, mlx.engine.holding(), base, after });
    return .{ .fired = fired, .left = left };
}

test "cooler contract: a Cooler load failed at any of its checked MLX calls leaves nothing of it behind" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            defer mlx.fault.disarm();
            {
                var warm: Load = undefined;
                try warm.open(C, dir, true);
                _ = warm.close();
            }
            var calls: u64 = 1;
            while ((try faultedLoad(C, dir, calls)).fired) calls *= 2;
            var left: usize = 0;
            var k: u64 = 1;
            while (k <= calls) : (k += if (k < 64) 1 else @max(1, calls / 96)) {
                if ((try faultedLoad(C, dir, k)).left) left += 1;
            }
            try testing.expectEqual(@as(usize, 0), left);
        }
    }.run);
}

test "cooler contract: requests, repeated, leave the engine as they found it, and an unload gives back everything, load after load" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const prompt = ids[0..@min(ids.len, 30)];
            for (0..3) |_| {
                const before = Sheet.now();
                var cooler: Load = undefined;
                try cooler.open(C, dir, true);
                {
                    errdefer _ = cooler.close();
                    for (0..2) |_| {
                        var g = try greedy(&cooler, prompt, 6);
                        g.deinit(testing.allocator);
                    }
                    const warm = Sheet.now();
                    for (0..3) |_| {
                        var g = try greedy(&cooler, prompt, 6);
                        g.deinit(testing.allocator);
                        try testing.expectEqual(warm, Sheet.now());
                    }
                }
                _ = cooler.close();
                try testing.expectEqual(before, Sheet.now());
            }
        }
    }.run);
}

test "cooler contract: a Cooler request failed at any of its checked MLX calls leaves the engine as a whole request does, and the next says the same tokens" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            defer mlx.fault.disarm();
            const prompt = ids[0..@min(ids.len, 30)];
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            var want = try greedy(&cooler, prompt, 6);
            defer want.deinit(testing.allocator);
            {
                var again = try greedy(&cooler, prompt, 6);
                again.deinit(testing.allocator);
            }
            const base = Sheet.now();
            var calls: u64 = 1;
            while (true) : (calls *= 2) {
                mlx.fault.arm(calls);
                var got = greedy(&cooler, prompt, 6);
                mlx.fault.disarm();
                if (got) |*g| g.deinit(testing.allocator) else |_| {}
                var msg: [300]u8 = undefined;
                _ = mlx.takeError(&msg);
                if (!mlx.fault.didFire()) break;
            }
            var k: u64 = 1;
            while (k <= calls) : (k += if (k < 64) 1 else @max(1, calls / 256)) {
                mlx.fault.arm(k);
                var faulted = greedy(&cooler, prompt, 6);
                mlx.fault.disarm();
                if (faulted) |*g| g.deinit(testing.allocator) else |_| {}
                var msg: [300]u8 = undefined;
                _ = mlx.takeError(&msg);
                var got = try greedy(&cooler, prompt, 6);
                defer got.deinit(testing.allocator);
                errdefer std.debug.print("cooler contract: a request failed at checked MLX call {d}\n", .{k});
                try testing.expectEqualSlices(u32, want.items, got.items);
                try testing.expectEqual(base, Sheet.now());
            }
        }
    }.run);
}

/// The logits of a prefill of `ids` and of the one-token step after it, as bytes.
fn stepLogits(ld: *Load, ids: []const u32, step: u32, out: *std.ArrayList(u8)) !void {
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    try ld.xfm.resetCache();
    const tail = [_]u32{step};
    for ([_][]const u32{ ids, &tail }) |part| {
        const arr = mlx.mlx_array_new_data(@ptrCast(part.ptr), &[_]c_int{ 1, @intCast(part.len) }, 2, .int32);
        defer _ = mlx.mlx_array_free(arr);
        const lg = try @import("model_handle.zig").forward(ld.xfm, arr);
        defer _ = mlx.mlx_array_free(lg);
        var c: mlx.mlx_array = .{};
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_contiguous(&c, lg, false, ld.xfm.s));
        try mlx.check(mlx.mlx_array_eval(c));
        const n = mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c);
        try out.appendSlice(testing.allocator, mlx.mlx_array_data_uint8(c).?[0..n]);
    }
}

test "cooler contract: a Cooler load's prefill and next step give the uncompiled MLX load's logits byte for byte at every prompt length, and an exact speculative request its greedy tokens" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var want: [C.prompt_lengths.len]std.ArrayList(u8) = @splat(.empty);
            defer for (&want) |*w| w.deinit(a);
            {
                var on_mlx: Load = undefined;
                try on_mlx.openMlxReference(C, dir);
                defer _ = on_mlx.close();
                for (C.prompt_lengths, &want) |len, *w| try stepLogits(&on_mlx, ids[0..@min(ids.len, len)], ids[ids.len - 1], w);
            }
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            for (C.prompt_lengths, want) |len, w| {
                const prompt = ids[0..@min(ids.len, len)];
                errdefer std.debug.print("cooler contract: a prompt of {d} tokens\n", .{prompt.len});
                var got: std.ArrayList(u8) = .empty;
                defer got.deinit(a);
                try stepLogits(&ld, prompt, ids[ids.len - 1], &got);
                try testing.expectEqualSlices(u8, w.items, got.items);
                if (comptime @hasDecl(C, "spec_is_greedy")) {
                    var g = try greedy(&ld, prompt, 12);
                    defer g.deinit(a);
                    const turn = arch_handle.turnFor(ld.h, &ld.config);
                    defer turn.end();
                    defer arch_sched.endRequest(&ld.config);
                    var sp = try C.specTokens(a, ld.xfm, prompt, 12);
                    defer sp.deinit(a);
                    try testing.expectEqualSlices(u32, g.items, sp.items);
                }
            }
        }
    }.run);
}
