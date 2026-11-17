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
        @import("cooler_side.zig").forTest(&self.config, true);
    }

    /// The load after `prepare`; on an error nothing of the load stays.
    fn load(self: *Load, dir: []const u8) !void {
        const a = self.a;
        errdefer self.config.deinit(a);
        const job = JobTurn.begin(null, &self.config);
        defer job.end();
        self.weights = try model_mod.loadWeightsForConfig(io, a, dir, &self.config, false);
        errdefer self.weights.deinit();
        model_mod.resolveWeightPrefix(&self.config, &self.weights);
        self.xfm = try a.create(Trunk);
        errdefer a.destroy(self.xfm);
        self.xfm.* = try Trunk.init(io, a, self.config, &self.weights);
        if (self.compile) @import("model_load.zig").compileClosures(self.xfm, .all);
        self.h = arch_handle.adopt(self.xfm);
    }

    fn open(self: *Load, comptime C: type, dir: []const u8, cooler: bool) !void {
        try self.prepare(C, dir, cooler);
        try self.loadServed(dir);
    }

    /// `load` and then `serve`, as a served load runs them; on an error nothing of the load stays.
    fn loadServed(self: *Load, dir: []const u8) !void {
        try self.load(dir);
        errdefer _ = self.close();
        try self.serve();
    }

    /// What a served load runs once its trunk is up: the KV scheme a launch with no flag gets, the MTP head's draft
    /// rerank, and the warm (`model_load.warmLoad`, fatal on Cooler, which ends sealed). No aux head is loaded.
    fn serve(self: *Load) !void {
        const job = JobTurn.begin(self.h, &self.config);
        defer job.end();
        const kv = @import("bills.zig").kvCacheForLaunch(&self.config, null).config;
        try self.xfm.cache.reinit(self.config.num_hidden_layers, kv);
        const mtp = model_load.mtpChoiceFor(null, &self.config);
        const mtp_on = mtp.on and !model_load.mtpDefaultOffUnderStreaming(mtp, self.config.expert_streaming, self.config.stream_mtp_head) and
            !@import("arch/config.zig").fact(&self.config, "no_mtp_head");
        if (mtp_on) _ = try @import("arch/arch_state.zig").buildDraftRerank(self.xfm);
        const policy = @import("mtp_policy.zig");
        const verify_cap: ?u32 = if (mtp_on) @min(policy.resolveMtpDepthCapForProfile(0, .generic), policy.mtpVerifyDraftsMax(&self.config)) else null;
        try model_load.warmLoad(io, self.a, &self.config, self.xfm, verify_cap, null, kv);
    }

    /// The MLX load a Cooler load's tokens are held to. MLX's compiled f32 GDN gate is ulps off its own unfused chain,
    /// which the engine equals, so the reference compiles no closure.
    fn openMlxReference(self: *Load, comptime C: type, dir: []const u8) !void {
        try self.prepare(C, dir, false);
        self.compile = false;
        try self.loadServed(dir);
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

/// The engine's pool and mappings, the pool's free buffers taken back first, and the owned regions weights live in; and
/// the dispatches a pass still held open when the sheet was read, which at rest are none.
const Sheet = struct {
    open: u32,
    in_use: u64,
    asked: u64,
    mapped: u64,
    regions: usize,
    owned: u64,

    fn now() Sheet {
        const open = mlx.engine.openDispatches();
        mlx.engine.close();
        const s = mlx.engine.stats();
        return .{
            .open = open,
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
    std.debug.print("cooler contract: fixture {s} ({s}): {s}: {s}\n", .{ env, dir, @errorName(e), mlx.takeError(&msg) orelse "no MLX or engine error latched" });
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

test "cooler contract: a Cooler load that drops experts by usage routes only the kept ones and says the MLX load's tokens" {
    // A route to a dropped expert fails the request (`ExpertDropped`): the router must never pick one.
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "dropExperts")) return error.SkipZigTest;
            const a = testing.allocator;
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            var root: ?[:0]u8 = null;
            defer if (root) |r| a.free(r);
            var said: [2]std.ArrayList(u32) = undefined;
            var made: usize = 0;
            defer for (said[0..made]) |*t| t.deinit(a);
            for ([_]bool{ false, true }) |cooler| {
                var ld: Load = undefined;
                try ld.prepare(C, dir, cooler);
                ld.compile = cooler;
                if (root == null) root = try @import("test_support.zig").usagePack(&tmp, dir, ld.config.num_hidden_layers, ld.config.num_experts, ld.config.num_experts_per_tok);
                const n = (C.dropExperts(a, &ld.config, root.?) catch |e| {
                    ld.config.deinit(a);
                    return e;
                }) orelse {
                    ld.config.deinit(a);
                    return error.SkipZigTest;
                };
                ld.config.drop_expert = n;
                try ld.load(root.?);
                defer _ = ld.close();
                try testing.expectEqual(cooler, ld.onEngine());
                // Every expert the router can pick is resident: a one-row request reads no route on the host.
                const engine = ld.xfm.expert_stream orelse return error.TestExpectedExpertStream;
                if (engine.plan.resident) {
                    const before = engine.hostRoutes();
                    const handed = if (cooler) mlx.engine.stats().handovers else 0;
                    const tokens = 4;
                    var one = try greedy(&ld, ids[0..1], tokens);
                    one.deinit(a);
                    try testing.expectEqual(before, engine.hostRoutes());
                    // Nor does a layer hand its pass over early: each boundary leaves the GPU idle, and the kicks start it.
                    // A token hands over two: the step's own next forward, and its sample's pass before the next
                    // forward reads the ids (`qwen4_hc.handOverIdsWriter`).
                    if (cooler) try testing.expect(mlx.engine.stats().handovers - handed <= 2 * tokens);
                }
                said[made] = try greedy(&ld, ids[0..@min(ids.len, 9)], 6);
                made += 1;
            }
            try testing.expectEqualSlices(u32, said[0].items, said[1].items);
        }
    }.run);
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
            try ld.loadServed(dir);
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

test "cooler contract: a prompt token id outside the model's vocabulary is refused before any forward reads past the embedding" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            const past: u32 = @intCast(cooler.config.vocab_size);
            try testing.expectError(error.PromptTokenOutOfVocab, greedy(&cooler, &.{ ids[0], past }, 1));
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
                const r = fx.loadServed(dir);
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
            var asked: u64 = 0;
            return specAsked(ld, C, prompt, &asked);
        }

        /// `spec`, and the MLX calls the request made to its end. The turn's end is not counted: returning to a side
        /// not at rest pays MLX what it is owed for its own earlier launches (`mlx.payOwed`), which no request asked.
        fn specAsked(ld: *Load, comptime C: type, prompt: []const u32, asked: *u64) !std.ArrayList(u32) {
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            const calls = mlx.engine.mlxCalls();
            defer asked.* = mlx.engine.mlxCalls() - calls;
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
                        var asked: u64 = 0;
                        var got = try specAsked(&cooler, C, prompt, &asked);
                        defer got.deinit(a);
                        errdefer std.debug.print("cooler contract: speculative request {d} over a prompt of {d} tokens\n", .{ round, prompt.len });
                        try testing.expectEqualSlices(u32, w.items, got.items);
                        try testing.expectEqual(@as(u64, 0), asked);
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
            return @import("model_load.zig").warmLoad(io, testing.allocator, &ld.config, ld.xfm, null, null, ld.xfm.cache.config);
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
            // The load doesn't serve; the refusal is named by whichever op or host maker met it.
            if (r) |_| return error.TestUnexpectedResult else |_| {}
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

test "cooler contract: a second Cooler load beside a sealed one, of another arch, warms unsealed and ends sealed" {
    // The load scope is shared: the second load's warm makes its own pipelines, none of them counted as late.
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    var ran = false;
    inline for (comptime std.meta.tags(registry.Tag)) |ta| {
        if (comptime !@hasDecl(registry.Mod(ta), "cooler_contract")) continue;
        inline for (comptime std.meta.tags(registry.Tag)) |tb| {
            if (comptime ta == tb or !@hasDecl(registry.Mod(tb), "cooler_contract")) continue;
            ran = (try besideSealed(registry.Mod(ta).cooler_contract, registry.Mod(tb).cooler_contract)) or ran;
        }
    }
    if (!ran) return error.SkipZigTest;
}

/// A load of `A`'s first fixture, warmed and sealed, then a load of `B`'s beside it; false when either is missing.
fn besideSealed(comptime A: type, comptime B: type) !bool {
    const da = std.mem.span(std.c.getenv(A.fixtures[0]) orelse return false);
    const db = std.mem.span(std.c.getenv(B.fixtures[0]) orelse return false);
    const Warm = struct {
        fn run(ld: *Load) !void {
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            return @import("model_load.zig").warmLoad(io, testing.allocator, &ld.config, ld.xfm, null, null, ld.xfm.cache.config);
        }
    };
    var first: Load = undefined;
    try first.open(A, da, true);
    defer _ = first.close();
    try Warm.run(&first);
    try testing.expect(mlx.engine.sealed());
    const late = mlx.engine.latePipelines();
    mlx.engine.late_names.clear();
    var second: Load = undefined;
    try second.open(B, db, true);
    defer _ = second.close();
    try Warm.run(&second);
    errdefer std.debug.print("cooler contract: the second load's warm ({s} beside {s}) counted late:\n{s}", .{ B.fixtures[0], A.fixtures[0], mlx.engine.late_names.text() });
    try testing.expectEqual(late, mlx.engine.latePipelines());
    try testing.expect(mlx.engine.sealed());
    return true;
}

test "cooler contract: a Cooler request ends with its pass closed and at most its resting cap free in the pool, and the unload leaves the pool empty" {
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
            // Every array a cache kept for the engine's side goes with its last model (`side_cache.dropAll`): the
            // pool holds only what was in use before the load, nothing when this test runs alone.
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

test "cooler contract: after a Cooler load's served warm, requests of every prompt length to 40, greedy, speculative and sampled, compile no kernel" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                const verify_cap: ?u32 = if (@hasDecl(C, "specTokens")) @import("generate.zig").Generator.mtpVerifyDraftsMax(&cooler.config) else null;
                try @import("model_load.zig").warmLoad(io, testing.allocator, &cooler.config, cooler.xfm, verify_cap, null, cooler.xfm.cache.config);
            }
            const late = mlx.engine.latePipelines();
            mlx.engine.late_names.clear();
            var long: [40]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
            var first_late: ?usize = null;
            for (1..long.len + 1) |len| {
                const prompt = long[0..len];
                var got = try greedy(&cooler, prompt, 6);
                got.deinit(testing.allocator);
                if (@hasDecl(C, "specTokens")) {
                    const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                    defer turn.end();
                    defer arch_sched.endRequest(&cooler.config);
                    var spec = try C.specTokens(testing.allocator, cooler.xfm, prompt, 6);
                    spec.deinit(testing.allocator);
                }
                if (@hasDecl(C, "specTokensWith")) {
                    const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                    defer turn.end();
                    defer arch_sched.endRequest(&cooler.config);
                    var sampled = try C.specTokensWith(testing.allocator, cooler.xfm, prompt, 6, .{ .temperature = 0.7, .top_p = 0.9, .seed = 7 });
                    sampled.deinit(testing.allocator);
                }
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = len;
            }
            errdefer std.debug.print("cooler contract: from a prompt of {?d} tokens on, requests compiled:\n{s}", .{ first_late, mlx.engine.late_names.text() });
            try testing.expectEqual(late, mlx.engine.latePipelines());
        }
    }.run);
}

test "cooler contract: a Cooler load's widest prefill chunk grows the engine's pool by no more than the chunk's bill" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            const a = testing.allocator;
            // A 4097-token prompt prefills as one 4096-row chunk, the served ladder's top, and a row. The ids repeat the
            // fixture's own prompt: past its vocabulary the embedding would read another table's bytes.
            const wide = try a.alloc(u32, 4097);
            defer a.free(wide);
            for (wide, 0..) |*t, i| t.* = ids[i % ids.len];
            mlx.engine.trim();
            mlx.engine.resetPoolPeak(.process);
            const held = mlx.engine.poolTotal();
            var got = try greedy(&cooler, wide, 1);
            got.deinit(a);
            const bills = @import("bills.zig");
            const kv = bills.configuredKvQuantFor(&cooler.config);
            const bill = bills.prefillChunkCost(&cooler.config, if (kv.scheme == .off) 16 else kv.bits, 4096, wide.len);
            const grew = mlx.engine.poolHeldPeak(.process) -| held;
            if (grew > bill) std.debug.print("cooler contract: the widest chunk grew the pool {d} MB over its {d} MB bill\n", .{ grew >> 20, bill >> 20 });
            try testing.expect(grew <= bill);
        }
    }.run);
}

test "cooler contract: after a Cooler load's served warm, a prompt at the widest chunk and a repetitive one past a sparse indexer's budget, greedy, speculative and sampled, compile no kernel" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            const budget: usize = cooler.config.indexer_budget;
            if (budget == 0) return;
            var widest: u32 = 0;
            {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                // The served cap: the adaptive depth row of this machine, and the widest a prompt-lookup chain drafts.
                const G = @import("generate.zig").Generator;
                const depth_cap = @import("mtp_policy.zig").mtpDepthCapForProfile(0, true, .generic);
                const verify_cap: ?u32 = if (@hasDecl(C, "specTokens")) G.mtpWarmVerifyCap(&cooler.config, depth_cap) else null;
                widest = @min(@max(depth_cap, G.mtpLookupDraftCap(&cooler.config)), G.mtpVerifyDraftsMax(&cooler.config));
                try @import("model_load.zig").warmLoad(io, testing.allocator, &cooler.config, cooler.xfm, verify_cap, null, cooler.xfm.cache.config);
            }
            const late = mlx.engine.latePipelines();
            mlx.engine.late_names.clear();
            const a = testing.allocator;
            // A chunk of 4096 rows is what a short prompt's request takes; a cycle of 61 tokens is what prompt lookup drafts.
            const wide = try a.alloc(u32, 4097);
            defer a.free(wide);
            for (wide, 0..) |*t, i| t.* = ids[i % ids.len] +% @as(u32, @intCast(i / ids.len));
            var first_late: ?[]const u8 = null;
            var got = try greedy(&cooler, wide, 4);
            got.deinit(a);
            if (first_late == null and mlx.engine.latePipelines() != late) first_late = "the widest chunk";
            const past = try a.alloc(u32, budget + 400);
            defer a.free(past);
            for (past, 0..) |*t, i| t.* = ids[i % 61 % ids.len] +% @as(u32, @intCast(i % 61));
            got = try greedy(&cooler, past, 24);
            got.deinit(a);
            if (first_late == null and mlx.engine.latePipelines() != late) first_late = "greedy past the budget";
            {
                // A prompt restored from the prefix cache prefills its tail past the budget, at any narrow width.
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                try @import("cooler_warm.zig").narrowAfter(io, a, cooler.xfm, past[0 .. budget + 61], 0);
            }
            if (first_late == null and mlx.engine.latePipelines() != late) first_late = "every narrow width past the budget";
            if (@hasDecl(C, "specTokensAt")) {
                // Each draft width a round can take, past the budget: the planner picks them by the acceptance it measures.
                const depth_bounds = @import("mtp_depth_bounds.zig");
                const saved = depth_bounds.active;
                defer depth_bounds.active = saved;
                var width: u32 = 1;
                while (width <= widest) : (width += 1) {
                    depth_bounds.active = .{ .min = width, .max = width };
                    const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                    defer turn.end();
                    defer arch_sched.endRequest(&cooler.config);
                    var spec = try C.specTokensAt(a, cooler.xfm, past, width + 2, .{ .temperature = 0 }, widest);
                    spec.deinit(a);
                }
                depth_bounds.active = saved;
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = "each pinned width past the budget";
                for ([_]@import("sampler.zig").SamplingParams{ .{ .temperature = 0 }, .{ .temperature = 0.7, .top_p = 0.9, .seed = 7 } }) |sp| {
                    const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                    defer turn.end();
                    defer arch_sched.endRequest(&cooler.config);
                    var spec = try C.specTokensWith(a, cooler.xfm, past, 48, sp);
                    spec.deinit(a);
                    if (first_late == null and mlx.engine.latePipelines() != late) first_late = if (sp.temperature == 0) "speculative past the budget" else "sampled speculative past the budget";
                }
                // Every round shape at the widest chunk's prompt: the planner opens them from the costs it measures.
                {
                    const policy = @import("mtp_policy.zig");
                    var plans: [64]policy.MtpRoundPlan = undefined;
                    policy.warm_plans = policy.everyPlan(&plans, widest);
                    defer policy.warm_plans = &.{};
                    const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                    defer turn.end();
                    defer arch_sched.endRequest(&cooler.config);
                    const rounds_before = policy.warm_plan_at;
                    var spec = try C.specTokensWith(a, cooler.xfm, wide, 96, .{ .temperature = 0 });
                    spec.deinit(a);
                    try testing.expect(policy.warm_plan_at - rounds_before >= policy.warm_plans.len);
                    if (first_late == null and mlx.engine.latePipelines() != late) first_late = "every round shape at the widest chunk";
                }
                // The release bench's request: the widest chunk's prompt, greedy with a nucleus and a top-k given.
                for ([_]@import("sampler.zig").SamplingParams{ .{ .temperature = 0 }, .{ .temperature = 0, .top_p = 0.95, .top_k = 20 } }) |sp| {
                    const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                    defer turn.end();
                    defer arch_sched.endRequest(&cooler.config);
                    var spec = try C.specTokensWith(a, cooler.xfm, wide, 48, sp);
                    spec.deinit(a);
                    if (first_late == null and mlx.engine.latePipelines() != late) first_late = if (sp.top_k == 0) "speculative at the widest chunk" else "speculative at the widest chunk with top-p and top-k";
                }
            }
            errdefer std.debug.print("cooler contract: from {?s} on, requests compiled:\n{s}", .{ first_late, mlx.engine.late_names.text() });
            try testing.expectEqual(late, mlx.engine.latePipelines());
        }
    }.run);
}

test "cooler contract: a pipeline a Cooler request compiles late is written to the load's manifest, for the next load to make ahead" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            const shared = @import("arch/cooler_pipelines.zig");
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            shared.root_hook = tmp.dir;
            defer shared.root_hook = null;
            defer shared.forget();
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                try @import("model_load.zig").warmLoad(io, testing.allocator, &cooler.config, cooler.xfm, null, null, cooler.xfm.cache.config);
            }
            const key = shared.writtenKey() orelse return error.NoManifestWritten;
            {
                // A kernel no warm makes: a unary op over int16, inside a request.
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                const vals = [_]i16{ -3, 4, -5, 6 };
                const x = mlx.mlx_array_new_data(&vals, &[_]c_int{4}, 1, .int16);
                defer _ = mlx.mlx_array_free(x);
                var y = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(y);
                try mlx.check(mlx.mlx_abs(&y, x, mlx.engine.stream()));
                try mlx.check(mlx.mlx_array_eval(y));
            }
            var found = try @import("pipeline_manifest.zig").lookup(testing.allocator, io, tmp.dir, key);
            defer found.deinit();
            const recipes = switch (found) {
                .same => |l| l.recipes,
                else => return error.NoManifestWritten,
            };
            for (recipes) |r| if (std.mem.indexOf(u8, r.function, "Abs") != null and std.mem.indexOf(u8, r.function, "int16") != null) return;
            return error.LatePipelineNotWritten;
        }
    }.run);
}

test "cooler contract: a Cooler load whose manifest made every pipeline runs no served warm" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            const shared = @import("arch/cooler_pipelines.zig");
            const cooler_warm = @import("cooler_warm.zig");
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            shared.root_hook = tmp.dir;
            defer shared.root_hook = null;
            defer shared.forget();
            var runs: [2]u32 = undefined;
            var scoped = false;
            for (&runs) |*r| {
                // `open` is the served load, its warm included.
                const before = cooler_warm.runs_for_test;
                var cooler: Load = undefined;
                try cooler.open(C, dir, true);
                defer _ = cooler.close();
                r.* = cooler_warm.runs_for_test - before;
                scoped = switch (@import("arch/config.zig").tagOf(&cooler.config).?) {
                    inline else => |t| @hasDecl(registry.Mod(t), "coolerPipelineScope"),
                };
            }
            // An arch that settles its manifest in its own load (no scope hook) warms every load.
            try testing.expectEqual(@as(u32, 1), runs[0]);
            try testing.expectEqual(@as(u32, if (scoped) 0 else 1), runs[1]);
        }
    }.run);
}

/// One Cooler load of `dir` with its `k`-th checked MLX call failed, on an allocator of its own: whether anything of it
/// stays (the allocator's, the engine's or a hold), and whether the fault fired.
/// A served load's two phases, each swept from its own root: `load` brings the trunk up, `serve` warms it.
const Phase = enum { load, serve };

/// A served load whose `phase` fails at its checked call `k` from the phase's start (0: none fails), on an allocator that
/// counts leaks: whether the fault fired, and whether anything of the load was left.
fn faultedLoad(comptime C: type, dir: []const u8, phase: Phase, k: u64) !struct { fired: bool, left: bool } {
    // Safety on whatever the build: it is what counts a leak at `deinit`, and ReleaseFast turns it off by default.
    var gpa: std.heap.DebugAllocator(.{ .safety = true }) = .init;
    const base = Sheet.now();
    var fx: Load = undefined;
    try fx.prepareOn(gpa.allocator(), C, dir, true);
    if (phase == .load) mlx.fault.arm(k);
    const loaded = fx.load(dir);
    if (phase == .load) mlx.fault.disarm();
    const r = if (loaded) |_| blk: {
        if (phase == .serve) mlx.fault.arm(k);
        const served = fx.serve();
        if (phase == .serve) mlx.fault.disarm();
        if (served) |_| {} else |_| _ = fx.close();
        break :blk served;
    } else |e| e;
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
    if (left) std.debug.print("cooler contract: the load failed at checked MLX call {d} of its {s} left: allocator {}, hold {}, {any} -> {any}\n", .{ k, @tagName(phase), leaked, mlx.engine.holding(), base, after });
    return .{ .fired = fired, .left = left };
}

/// Frames from each phase's root a load's sweep tells sites by. `load`: every step of bringing the trunk up, to the
/// weights and layers it binds. `serve`: the warm's own steps, the cache, the draft rerank, the warm-up, the verify and
/// served warms; the forwards inside them are the request sweeps'.
const load_sweep_frames = std.enums.EnumArray(Phase, usize).init(.{ .load = 4, .serve = 3 });

test "cooler contract: a Cooler load failed at any of its checked MLX calls leaves nothing of it behind" {
    var paths = std.AutoHashMap(u64, void).init(testing.allocator);
    defer paths.deinit();
    const S = struct {
        var seen: *std.AutoHashMap(u64, void) = undefined;
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            defer mlx.fault.disarm();
            defer mlx.fault.sites = .{};
            {
                var warm: Load = undefined;
                try warm.open(C, dir, true);
                _ = warm.close();
            }
            var left: usize = 0;
            for ([_]Phase{ .load, .serve }) |phase| {
                // One clean load records the phase's sites; each is failed at its first call. A fixture whose phase
                // makes the very calls another's did is that path again, and is not swept twice.
                const R = struct {
                    dir: []const u8,
                    phase: Phase,
                    fn go(r: @This()) anyerror!void {
                        if ((try faultedLoad(C, r.dir, r.phase, 0)).left) return error.CleanLoadLeft;
                    }
                };
                mlx.fault.sites = .{ .from = .root, .frames = load_sweep_frames.get(phase), .lasts = false };
                const points = try mlx.sweepPoints(testing.allocator, &mlx.fault.sites, R{ .dir = dir, .phase = phase }, R.go);
                defer testing.allocator.free(points);
                const path = std.hash.Wyhash.hash(@intFromEnum(phase), std.mem.sliceAsBytes(points));
                if ((try seen.getOrPut(path)).found_existing) continue;
                for (points) |k| {
                    const r = try faultedLoad(C, dir, phase, k);
                    if (!r.fired) {
                        std.debug.print("cooler contract: the load's {s} never reached its checked call {d}: it makes other calls than the one recorded\n", .{ @tagName(phase), k });
                        return error.TestUnexpectedResult;
                    }
                    if (r.left) {
                        left += 1;
                        mlx.fault.trace = true;
                        defer mlx.fault.trace = false;
                        _ = try faultedLoad(C, dir, phase, k);
                    }
                }
            }
            try testing.expectEqual(@as(usize, 0), left);
        }
    };
    S.seen = &paths;
    try eachFixture(S.run);
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
            const Req = struct {
                ld: *Load,
                prompt: []const u32,
                fn go(r: @This()) anyerror!void {
                    var got = try greedy(r.ld, r.prompt, 6);
                    got.deinit(testing.allocator);
                }
            };
            // Each op under its callers fails once, at its first call: a request is short, and its sites many.
            mlx.fault.sites = .{ .lasts = false };
            defer mlx.fault.sites = .{};
            const points = try mlx.sweepPoints(testing.allocator, &mlx.fault.sites, Req{ .ld = &cooler, .prompt = prompt }, Req.go);
            defer testing.allocator.free(points);
            for (points) |k| {
                mlx.fault.arm(k);
                var faulted = greedy(&cooler, prompt, 6);
                mlx.fault.disarm();
                if (faulted) |*g| g.deinit(testing.allocator) else |_| {}
                if (!mlx.fault.didFire()) {
                    std.debug.print("cooler contract: the request's checked call {d} was never reached: a request makes other calls than the one recorded\n", .{k});
                    return error.TestUnexpectedResult;
                }
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
                if (comptime @hasDecl(C, "specIsGreedy")) {
                    if (C.specIsGreedy(&ld.config)) {
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
        }
    }.run);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// A contract load served by the scheduler's own slot path: a registry entry holding the load's handle and config, the
/// SSD-only tier a served load arms for it (`attachPrefixCache`, its cache dir under a tmp dir), and a test scheduler.
const Served = struct {
    const scheduler = @import("scheduler.zig");
    const registry_mod = @import("model_registry.zig");

    tmp: testing.TmpDir,
    cache_dir: [:0]u8,
    reg: *registry_mod.ModelRegistry,
    entry: *registry_mod.LoadedModel,
    tok: Tokenizer,
    sch: scheduler.Scheduler,
    /// The last request's prompt tokens taken from a cache tier or the live state.
    reused: u32 = 0,

    fn init(self: *Served, ld: *Load, dir: []const u8) !void {
        const a = testing.allocator;
        self.tmp = testing.tmpDir(.{});
        errdefer self.tmp.cleanup();
        var buf: [512]u8 = undefined;
        self.cache_dir = try std.fmt.allocPrintSentinel(a, "{s}", .{buf[0..try self.tmp.dir.realPath(io, &buf)]}, 0);
        errdefer a.free(self.cache_dir);
        self.reg = try registry_mod.ModelRegistry.init(a, io, null, 3, 0, null);
        errdefer self.reg.deinit();
        self.entry = try self.reg.registerStub("contract", dir, 1);
        self.tok = Tokenizer.initEmpty(a);
        self.entry.config = &ld.config;
        self.entry.handle = ld.h;
        self.entry.tokenizer = &self.tok;
        errdefer self.release();
        _ = setenv("SUSHI_PREFIX_CACHE_DIR", self.cache_dir.ptr, 1);
        defer _ = unsetenv("SUSHI_PREFIX_CACHE_DIR");
        const server_state = @import("server_state.zig");
        try model_load.attachPrefixCache(a, io, self.entry, .{
            .capacity = 4,
            .ram_enabled = false,
            .mem_bytes = 0,
            .mem_resolver = null,
            .disk_resolver = null,
            .ssm_checkpoint_stride = server_state.ssm_checkpoint_stride,
            .ssm_checkpoint_max = server_state.ssm_checkpoint_max,
        }, 1 << 30, .load);
        if (self.entry.prefix_cache == null or self.entry.prefix_cache.?.disk == null) return error.SkipZigTest;
        self.entry.prefix_cache.?.disk.?.armTestSpace(256 << 30, 1 << 40);
        self.sch = scheduler.submitTestScheduler(a, 2);
        self.sch.kv_quant = null;
        self.sch.metrics = null;
        self.sch.hot_prefix_cache = null;
        self.sch.requests_prefilling = .init(0);
        self.sch.inflight_prefill_tokens = .init(0);
        self.sch.inflight_prefill_expected = .init(0);
        self.sch.resident_hot_cache_bytes = .init(0);
        self.sch.reclaimable_hot_cache_bytes = .init(0);
        self.sch.resident_live_kv_bytes = .init(0);
        self.sch.live_session_count = 0;
        self.sch.registry = self.reg;
        self.sch.digest_mu = .init;
        self.sch.hot_cache_digests = &.{};
        self.sch.digest_residency = 0;
        self.sch.tiers_declined = .init(0);
    }

    /// The entry's cache, and the entry's hold on the load, which stays the contract's.
    fn release(self: *Served) void {
        if (self.entry.prefix_cache) |*hc| hc.deinit();
        self.entry.prefix_cache = null;
        self.entry.config = null;
        self.entry.handle = null;
        self.entry.tokenizer = null;
        self.tok.deinit();
    }

    fn deinit(self: *Served) void {
        self.sch.pending.deinit(testing.allocator);
        self.sch.decoding.deinit(testing.allocator);
        self.sch.cleanup_queue.deinit(testing.allocator);
        self.sch.prefilling.deinit(testing.allocator);
        self.release();
        self.reg.deinit();
        testing.allocator.free(self.cache_dir);
        self.tmp.cleanup();
    }

    /// One greedy request through the scheduler's slot path (submit, `runPrefill`, decode ticks, the finish that
    /// commits), its writes to the tier drained; the tokens it said, and whether it restored from SSD and how far.
    fn request(self: *Served, prompt: []const u32, n: u32, restored: *u32) !std.ArrayList(u32) {
        return self.requestWith(prompt, n, .{ .temperature = 0.0 }, restored);
    }

    /// `restored`: the prompt tokens an SSD restore gave; `reused` holds what any tier or the live state gave.
    fn requestWith(self: *Served, prompt: []const u32, n: u32, sampling: @import("sampler.zig").SamplingParams, restored: *u32) !std.ArrayList(u32) {
        const a = testing.allocator;
        const sch = &self.sch;
        const slot = try sch.submit(.{ .model = self.entry, .prompt_ids = prompt, .sampling = sampling, .eos_token_ids = &.{}, .max_tokens = n });
        defer {
            sch.complete(slot);
            for (sch.cleanup_queue.items) |s| s.deinit();
            sch.cleanup_queue.clearRetainingCapacity();
        }
        _ = sch.pending.orderedRemove(0);
        try scheduler.runPrefill(sch, slot);
        // The prefill's end counts the hit and clears the slot's tier; the cache still names it.
        const hc = &self.entry.prefix_cache.?;
        restored.* = if (slot.cached_tokens > 0 and hc.lastRestoredTier() == .ssd) slot.cached_tokens else 0;
        self.reused = slot.cached_tokens;
        if (!scheduler.postPrefillTerminal(slot)) {
            try sch.decoding.append(a, slot);
            var one = [_]*scheduler.Slot{slot};
            while (slot.joinsTick()) try scheduler.runDecodeTick(sch, &one);
            scheduler.cullDecoding(sch);
        }
        if (slot.error_code) |e| {
            std.debug.print("cooler contract: the served request failed ({s})\n", .{e});
            return error.RequestFailed;
        }

        var got: std.ArrayList(u32) = .empty;
        errdefer got.deinit(a);
        try got.appendSlice(a, slot.legacy_gen.?.generated_ids.items);
        hc.flushPendingDisk(arch_handle.trunk(self.entry.handle.?).s);
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
        return got;
    }
};

test "cooler contract: a served Cooler request that restores from the SSD tier says a cold request's tokens and asks nothing of MLX" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            // Long enough that the tier holds restore points under it.
            var long: [1300]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            var cold_from: u32 = 0;
            var cold = try sv.request(&long, 12, &cold_from);
            defer cold.deinit(a);
            try testing.expectEqual(@as(u32, 0), cold_from);
            // Another conversation takes the model, so the first one's state stands only on SSD.
            var other_from: u32 = 0;
            var other = try sv.request(ids[0..@min(ids.len, 20)], 4, &other_from);
            other.deinit(a);
            const calls = mlx.engine.mlxCalls();
            var warm_from: u32 = 0;
            var warm = try sv.request(&long, 12, &warm_from);
            defer warm.deinit(a);
            try testing.expectEqual(calls, mlx.engine.mlxCalls());
            errdefer std.debug.print("cooler contract: the warm request restored {d} tokens from SSD\n", .{warm_from});
            try testing.expect(warm_from > 0);
            try testing.expectEqualSlices(u32, cold.items, warm.items);
        }
    }.run);
}

test "cooler contract: after a Cooler load's served warm, a served request that restores the live state behind its reply compiles no kernel" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            // Long enough that the request holds restore points under it, as the SSD contract's prompt does.
            var long: [1300]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            var from: u32 = 0;
            var first = try sv.request(&long, 24, &from);
            first.deinit(a);
            const late = mlx.engine.latePipelines();
            mlx.engine.late_names.clear();
            // The bench's repeat: the same prompt again, its state rolled back from behind the first reply.
            var again = try sv.request(&long, 24, &from);
            again.deinit(a);
            errdefer std.debug.print("cooler contract: the repeat reused {d} tokens and compiled:\n{s}", .{ sv.reused, mlx.engine.late_names.text() });
            try testing.expect(sv.reused > 0);
            try testing.expectEqual(late, mlx.engine.latePipelines());
        }
    }.run);
}

test "cooler contract: after a Cooler load's served warm, the release bench's first request through the scheduler compiles no kernel" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            // The bench's first: a 4096-token prompt, greedy with a nucleus and a top-k given, its rounds as the planner
            // picks them.
            const prompt = try a.alloc(u32, 4096);
            defer a.free(prompt);
            for (prompt, 0..) |*t, i| t.* = ids[i % ids.len];
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            const late = mlx.engine.latePipelines();
            mlx.engine.late_names.clear();
            var from: u32 = 0;
            var got = try sv.requestWith(prompt, 64, .{ .temperature = 0, .top_p = 0.95, .top_k = 20 }, &from);
            got.deinit(a);
            errdefer std.debug.print("cooler contract: the bench's first request compiled:\n{s}", .{mlx.engine.late_names.text()});
            try testing.expectEqual(late, mlx.engine.latePipelines());
        }
    }.run);
}

/// A greedy request, plain or through the arch's speculative path, on a load that stays.
fn requestOf(comptime C: type, ld: *Load, prompt: []const u32, n: u32, spec: bool) !std.ArrayList(u32) {
    if (!spec) return greedy(ld, prompt, n);
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    try ld.xfm.resetCache();
    return C.specTokens(testing.allocator, ld.xfm, prompt, n);
}

test "cooler contract: a Cooler request refused at any pool buffer fails by its error, and every later request on the same load says the same tokens" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            defer mlx.engine.fault.disarm();
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            var msg: [300]u8 = undefined;
            for ([_]usize{ 1, 9 }) |plen| for ([_]bool{ false, true }) |spec| {
                if (spec and !@hasDecl(C, "specTokens")) continue;
                const prompt = ids[0..@min(ids.len, plen)];
                var want = try requestOf(C, &ld, prompt, 6, spec);
                defer want.deinit(a);
                const Req = struct {
                    ld: *Load,
                    prompt: []const u32,
                    spec: bool,
                    fn go(q: @This()) anyerror!void {
                        var got = try requestOf(C, q.ld, q.prompt, 6, q.spec);
                        got.deinit(testing.allocator);
                    }
                };
                // One clean request records every pool buffer's site; each site is refused at its first and last ask. A
                // speculative request's asks do not come in the same order every time, so a point is a site's n-th ask.
                const points = try mlx.sweepSites(a, &mlx.engine.fault.sites, Req{ .ld = &ld, .prompt = prompt, .spec = spec }, Req.go);
                defer a.free(points);
                const base = Sheet.now();
                for (points, 0..) |pt, at| {
                    errdefer std.debug.print("cooler contract: a {s} request of {d} tokens refused at point {d} (a site's ask {d})\n", .{ if (spec) "speculative" else "plain", prompt.len, at, pt.n });
                    mlx.engine.fault.armAt(pt);
                    const r = requestOf(C, &ld, prompt, 6, spec);
                    const fired = !mlx.engine.fault.armed();
                    mlx.engine.fault.disarm();
                    if (r) |got| {
                        var g = got;
                        g.deinit(a);
                        try testing.expect(!mlx.errorPending());
                    } else |_| _ = mlx.takeError(&msg);
                    if (!fired and pt.strict) {
                        std.debug.print("cooler contract: a {s} request never made ask {d} of a site its recording made\n", .{ if (spec) "speculative" else "plain", pt.n });
                        return error.TestUnexpectedResult;
                    }
                    var after = try requestOf(C, &ld, prompt, 6, spec);
                    defer after.deinit(a);
                    try testing.expectEqualSlices(u32, want.items, after.items);
                    try testing.expectEqual(base, Sheet.now());
                }
            };
        }
    }.run);
}

test "cooler contract: a Cooler prefill hands each chunk's free buffers back, so a long prompt holds no more free than one chunk" {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const Q = registry.Mod(.qwen4).cooler_contract;
    const dir = std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const ids = try Q.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(Q, dir, true);
    defer _ = ld.close();
    var long: [3600]u32 = undefined;
    for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
    var free_after: [2]u64 = undefined;
    for ([_]usize{ 512, long.len }, &free_after) |len, *free| {
        const turn = arch_handle.turnFor(ld.h, &ld.config);
        defer turn.end();
        try ld.xfm.resetCache();
        mlx.engine.close();
        mlx.engine.trim();
        var tok = Tokenizer.initEmpty(a);
        defer tok.deinit();
        var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, long[0..len], 1, .{ .temperature = 0.0 }, &.{}, .{ .pinned_prefill_chunk = 512, .skip_lazy_preforward = true });
        defer gen.deinit(a);
        mlx.engine.close();
        free.* = mlx.engine.stats().pool_free;
        arch_sched.endRequest(&ld.config);
    }
    // Beyond one chunk's, only what the whole prompt sizes may stay free: its ids, a pool class of their bytes.
    try testing.expect(free_after[1] <= free_after[0] + std.math.ceilPowerOfTwoAssert(usize, long.len * @sizeOf(u32)));
}
