//! The Sushi Cooler contract: what every load on the engine of our own owes, whatever its arch. An arch on Sushi
//! Cooler declares `cooler_contract` (its tiny fixtures by env var and the reads only it can make: `prepare`,
//! `prompt`, `weightsOutsideRegions`, `headLoaded`, optionally `specTokens` and `specTokensWith` (sampled), and
//! `greedyRows`, required when its `coolerRows` gives more than one (the rows test fails by name without it), and
//! `reads_out_last_row_only` where its forward reads out a wide forward's last row alone; each test
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
const SamplingParams = @import("sampler.zig").SamplingParams;

const io = std.Io.Threaded.global_single_threaded.io();

/// One test load of a fixture, on the side the test names. Its load and its unload run in the job turn a served load
/// and unload take (`model_load.JobTurn`).
/// A Load stamps the model settings file's row for its directory, as a served load does (`model_settings.overrideFor`),
/// instead of none: the cells that hold what the web UI saves (`withUiSettings`).
var settings_from_file = false;

const Load = struct {
    a: std.mem.Allocator,
    config: model_mod.ModelConfig,
    weights: model_mod.Weights,
    xfm: *Trunk,
    h: arch_handle.Handle,

    fn prepare(self: *Load, comptime C: type, dir: []const u8, cooler: bool) !void {
        return self.prepareOn(testing.allocator, C, dir, cooler);
    }

    /// A load on MLX is the one the user's opt-out flag makes, through the parse and the settings stamp a served load
    /// takes; a Cooler load lifts the tests' MLX default.
    fn prepareOn(self: *Load, a: std.mem.Allocator, comptime C: type, dir: []const u8, cooler: bool) !void {
        self.a = a;
        const off = model_mod.cooler_off;
        defer model_mod.cooler_off = off;
        model_mod.cooler_off = if (cooler) .empty else .full;
        self.config = try model_mod.parseConfig(io, a, dir);
        errdefer self.config.deinit(a);
        model_load.applyModelSettings(&self.config, if (settings_from_file) @import("model_settings.zig").overrideFor(a, io, dir) else .{});
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
        @import("model_load.zig").compileClosures(self.xfm, .all);
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

/// Each pool-buffer site `run(ctx)` reaches, at its first ask, but a site already in `seen`, which takes the rest: a
/// site is a code location with its own unwinding, so one refusal each walks every error path once. Sites are told
/// apart per arch, whose forward unwinds above them.
fn firstAsks(a: std.mem.Allocator, seen: *std.AutoHashMapUnmanaged(u64, void), comptime C: type, ctx: anytype, comptime run: fn (@TypeOf(ctx)) anyerror!void) ![]mlx.FaultSites.SitePoint {
    const sites = &mlx.engine.fault.sites;
    sites.lasts = false;
    defer sites.* = .{};
    const all = try mlx.sweepSites(a, sites, ctx, run);
    defer a.free(all);
    var fresh: std.ArrayList(mlx.FaultSites.SitePoint) = .empty;
    errdefer fresh.deinit(a);
    for (all) |pt| if (!(try seen.getOrPut(a, archSite(C, pt.site))).found_existing) try fresh.append(a, pt);
    return fresh.toOwnedSlice(a);
}

fn archSite(comptime C: type, site: u64) u64 {
    return std.hash.Wyhash.hash(site, @typeName(C));
}

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

/// `SUSHI_CONTRACT_FIXTURE`: the fixture envs a run takes, comma-separated (`QWEN4_STQ_TINY,DSV41_STQ_TINY`); unset, all.
fn selection() ?[]const u8 {
    return if (std.c.getenv("SUSHI_CONTRACT_FIXTURE")) |v| std.mem.span(v) else null;
}

/// Whether a run under `sel` takes the fixture named by `env`.
fn fixtureChosen(sel: ?[]const u8, env: []const u8) bool {
    const names = sel orelse return true;
    var it = std.mem.tokenizeScalar(u8, names, ',');
    while (it.next()) |n| if (std.mem.eql(u8, std.mem.trim(u8, n, " "), env)) return true;
    return false;
}

/// Whether a cell skips before any load: with no selection, the gate's rule that every fixture is set, so a cell never
/// pays for the fixtures present only to report a skip. A selection skips per named fixture instead (`Tally`).
fn skipsUpFront(sel: ?[]const u8, set: []const bool) bool {
    if (sel != null) return false;
    for (set) |is_set| if (!is_set) return true;
    return false;
}

/// Every contract fixture env, and whether it is set.
fn fixturesSet(buf: *[16]bool) []const bool {
    var n: usize = 0;
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime @hasDecl(registry.Mod(t), "cooler_contract")) {
            for (registry.Mod(t).cooler_contract.fixtures) |env| {
                buf[n] = std.c.getenv(env) != null;
                n += 1;
            }
        }
    }
    return buf[0..n];
}

/// `body` on each chosen fixture of each arch that declares `cooler_contract` (`Tally` says what a skip or a missing one
/// does). For a property of the fixture's format or geometry; one that holds for any fixture of an arch is `eachArch`.
fn eachFixture(comptime body: fn (comptime C: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return eachOf(.every, body);
}

/// `body` on one fixture per arch: its first chosen and set fixture (the STQ tiny when set). For a property of the arch
/// and the engine, not of a pack's format.
fn eachArch(comptime body: fn (comptime C: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return eachOf(.first, body);
}

/// `body` on one fixture per arch and per expert residency, the first chosen of each: resident, and streamed (the BF16
/// checkpoint, MXFP4 experts). For a pool, memory or cleanup property, which a streamed plan changes and a pack's
/// geometry or tile order does not.
fn eachResidency(comptime body: fn (comptime C: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return eachOf(.per_residency, body);
}

/// `body` on one fixture per arch and per expert layout, the first chosen of each: its residency and its tile order
/// (the pair-major pack). For a property of the loader, its cleanup or the calls a request makes, which a pack's
/// geometry (the cut beside the tiny) does not change.
fn eachLayout(comptime body: fn (comptime C: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return eachOf(.per_layout, body);
}

/// A contract load's expert layout, from its config as the arch prepares it; nothing loads. Bit 0: the experts stream;
/// bit 1: they are stored pair-major.
fn layoutOf(comptime C: type, dir: []const u8) !u2 {
    const a = testing.allocator;
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    try C.prepare(a, &config, dir);
    const streamed = config.expert_streaming and !config.expert_resident;
    return @as(u2, @intFromBool(streamed)) | @as(u2, @intFromBool(config.expert_tile_order == .pair_major)) << 1;
}

fn eachOf(comptime which: enum { every, first, per_residency, per_layout }, comptime body: fn (comptime C: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    try mlx.requireGpu();
    const sel = selection();
    var set_buf: [16]bool = undefined;
    if (skipsUpFront(sel, fixturesSet(&set_buf))) return error.SkipZigTest;
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    var tally: Tally = .{};
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime @hasDecl(registry.Mod(t), "cooler_contract")) {
            const C = registry.Mod(t).cooler_contract;
            var took_one = false;
            // The layouts taken (`layoutOf`); per residency only its streamed bit counts.
            var took: [4]bool = @splat(false);
            for (C.fixtures) |env| {
                if (!fixtureChosen(sel, env)) continue;
                if (which == .first and took_one) break;
                const dir = std.mem.span(std.c.getenv(env) orelse {
                    tally.absent();
                    continue;
                });
                if (which == .per_residency or which == .per_layout) {
                    const layout = layoutOf(C, dir) catch |e| return named(env, dir, e);
                    const key = if (which == .per_residency) layout & 1 else layout;
                    if (took[key]) continue;
                    took[key] = true;
                }
                const ids = C.prompt(testing.allocator, dir) catch |e| return named(env, dir, e);
                defer testing.allocator.free(ids);
                tally.took(body(C, dir, ids)) catch |e| return named(env, dir, e);
                took_one = true;
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

test "cooler contract runner: SUSHI_CONTRACT_FIXTURE names the fixtures a run takes, and without it a missing fixture skips the test before any load" {
    // No selection: every fixture is taken, and one unset skips the cell up front.
    try testing.expect(fixtureChosen(null, "QWEN4_STQ_TINY"));
    try testing.expect(skipsUpFront(null, &.{ true, false, true }));
    try testing.expect(!skipsUpFront(null, &.{ true, true, true }));
    // A selection takes only the fixtures it names, comma-separated, and an unnamed unset one is no reason to skip.
    try testing.expect(fixtureChosen("QWEN4_STQ_TINY", "QWEN4_STQ_TINY"));
    try testing.expect(!fixtureChosen("QWEN4_STQ_TINY", "QWEN4_TEST_MODEL"));
    try testing.expect(fixtureChosen("DSV41_TINY,QWEN4_STQ_TINY", "QWEN4_STQ_TINY"));
    try testing.expect(!fixtureChosen("QWEN4_STQ_TINY_X", "QWEN4_STQ_TINY"));
    try testing.expect(!skipsUpFront("QWEN4_STQ_TINY", &.{ true, false, false }));
    // A named fixture left unset is still missing: the run says skipped, never passed.
    var named_unset: Tally = .{};
    named_unset.absent();
    try testing.expectError(error.SkipZigTest, named_unset.verdict());
}

/// `e`, after naming the fixture it came from.
fn named(env: []const u8, dir: []const u8, e: anyerror) anyerror {
    var msg: [512]u8 = undefined;
    std.debug.print("cooler contract: fixture {s} ({s}): {s}: {s}\n", .{ env, dir, @errorName(e), mlx.takeError(&msg) orelse "no MLX or engine error latched" });
    return e;
}

/// `n` greedy tokens over `ids` in serial steps, one request in the model's turn, ended as serve ends one.
fn greedy(ld: *Load, ids: []const u32, n: u32) !std.ArrayList(u32) {
    return serial(ld, ids, n, .{ .temperature = 0.0 });
}

/// `greedy` under `sampling`.
fn serial(ld: *Load, ids: []const u32, n: u32, sampling: SamplingParams) !std.ArrayList(u32) {
    return serialWith(ld, ids, n, sampling, .{});
}

/// `serial` with the generator's options, as a served request's own.
fn serialWith(ld: *Load, ids: []const u32, n: u32, sampling: SamplingParams, options: Generator.InitOptions) !std.ArrayList(u32) {
    const a = testing.allocator;
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    const xfm = ld.xfm;
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try xfm.resetCache();
    var gen = try Generator.initWithOptions(io, a, xfm, &tok, ids, n, sampling, &.{}, options);
    defer gen.deinit(a);
    var got: std.ArrayList(u32) = .empty;
    errdefer got.deinit(a);
    while (try gen.next(a)) |t| try got.append(a, t);
    return got;
}

test "cooler contract: a Cooler load that drops experts by usage serves, routing only the kept ones on the GPU" {
    // A route to a dropped expert fails the request (`ExpertDropped`): the router must never pick one.
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "dropExperts")) return error.SkipZigTest;
            const a = testing.allocator;
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            var ld: Load = undefined;
            try ld.prepare(C, dir, true);
            const root = @import("test_support.zig").usagePack(&tmp, dir, ld.config.num_hidden_layers, ld.config.num_experts, ld.config.num_experts_per_tok) catch |e| {
                ld.config.deinit(a);
                return e;
            };
            defer a.free(root);
            const n = (C.dropExperts(a, &ld.config, root) catch |e| {
                ld.config.deinit(a);
                return e;
            }) orelse {
                ld.config.deinit(a);
                return error.SkipZigTest;
            };
            ld.config.drop_expert = n;
            try ld.load(root);
            defer _ = ld.close();
            try testing.expect(ld.onEngine());
            // Every expert the router can pick is resident: a one-row request reads no route on the host.
            const engine = ld.xfm.expert_stream orelse return error.TestExpectedExpertStream;
            if (engine.plan.resident) {
                const before = engine.hostRoutes();
                const handed = mlx.engine.stats().handovers;
                const tokens = 4;
                var one = try greedy(&ld, ids[0..1], tokens);
                one.deinit(a);
                try testing.expectEqual(before, engine.hostRoutes());
                // Nor does a layer hand its pass over early: each boundary leaves the GPU idle, and the kicks start it.
                // A token hands over two: the step's own next forward, and its sample's pass before the next
                // forward reads the ids (`qwen4_hc.handOverIdsWriter`).
                try testing.expect(mlx.engine.stats().handovers - handed <= 2 * tokens);
            }
            var said = try greedy(&ld, ids[0..@min(ids.len, 9)], 6);
            said.deinit(a);
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
    try eachLayout(struct {
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

test "cooler contract: a Cooler load opens under whatever engine cap was in force before it, and its unload leaves the pool as it found it" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            const base = Sheet.now();
            // A cap an earlier forward left in force, below the smallest buffer the load makes.
            try mlx.engine.budget(4 << 10);
            defer mlx.engine.budget(std.math.maxInt(u64)) catch {};
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            _ = ld.close();
            try testing.expectEqual(base, Sheet.now());
        }
    }.run);
}

test "cooler contract: a prompt token id outside the model's vocabulary is refused before any forward reads past the embedding" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            const past: u32 = @intCast(cooler.config.vocab_size);
            try testing.expectError(error.PromptTokenOutOfVocab, greedy(&cooler, &.{ ids[0], past }, 1));
        }
    }.run);
}

test "cooler contract: a Cooler load's request and unload ask nothing of MLX" {
    try eachLayout(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const prompt = ids[0..@min(ids.len, 30)];
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            var closed = false;
            defer if (!closed) {
                _ = cooler.close();
            };
            const calls = mlx.engine.mlxCalls();
            var got = try greedy(&cooler, prompt, 12);
            got.deinit(testing.allocator);
            try testing.expectEqual(calls, mlx.engine.mlxCalls());
            // The job's end then pays MLX what it was owed while at rest: not the teardown's.
            closed = true;
            try testing.expectEqual(@as(u64, 0), cooler.close());
        }
    }.run);
}

test "cooler contract: a Cooler load refused at a pool buffer where its cleanup changes frees what it made, and the next load serves the same tokens" {
    try eachLayout(struct {
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
            // The asks of the trunk's bring-up (`Load.load`, whose errdefers free the weights and the trunk) and of the
            // served warm (`serve`, whose failure closes the whole load), counted on a clean load after the first.
            var asks: [2]u64 = undefined;
            {
                var counted: Load = undefined;
                try counted.prepare(C, dir, true);
                const c0 = mlx.engine.fault.count();
                try counted.load(dir);
                defer _ = counted.close();
                const c1 = mlx.engine.fault.count();
                try counted.serve();
                asks = .{ c1 - c0, mlx.engine.fault.count() - c1 };
            }
            try testing.expectEqual(base, Sheet.now());
            try testing.expect(asks[0] > 0 and asks[1] > 0);
            // The bring-up's first and last ask (nothing of it made yet; all of it but one buffer), and the warm's first:
            // a warm failed anywhere closes the whole load, and its forwards' own unwinding is the request sweeps'.
            const points = [_]u64{ 1, asks[0], asks[0] + 1 };
            var msg: [300]u8 = undefined;
            for (points) |k| {
                var fx: Load = undefined;
                try fx.prepare(C, dir, true);
                mlx.engine.fault.arm(k);
                const r = fx.loadServed(dir);
                const fired = !mlx.engine.fault.armed();
                mlx.engine.fault.disarm();
                errdefer std.debug.print("cooler contract: the load refused at pool buffer {d} of {d} ({s}) left {any} -> {any}\n", .{ k, asks[0] + asks[1], if (r) |_| "it loaded" else |_| "it failed", base, Sheet.now() });
                if (r) |_| {
                    _ = fx.close();
                    try testing.expect(!mlx.errorPending());
                } else |_| _ = mlx.takeError(&msg);
                try testing.expect(fired);
                try testing.expect(!mlx.engine.holding());
                try testing.expectEqual(base, Sheet.now());
            }
            var after: Load = undefined;
            try after.open(C, dir, true);
            defer _ = after.close();
            var got = try greedy(&after, prompt, 8);
            defer got.deinit(testing.allocator);
            try testing.expectEqualSlices(u32, want.items, got.items);
        }
    }.run);
}

test "cooler contract: a Cooler warm's image splice refused at any of its pool buffer sites frees what it made" {
    try eachArch(struct {
        fn go(ld: *Load) anyerror!void {
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            return @import("cooler_warm.zig").spliceAhead(ld.xfm);
        }

        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            const a = testing.allocator;
            defer mlx.engine.fault.disarm();
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            if (ld.config.image_token_id == 0) return error.SkipZigTest;
            var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
            defer seen.deinit(a);
            const points = try firstAsks(a, &seen, C, &ld, go);
            defer a.free(points);
            try testing.expect(points.len > 1);
            const base = Sheet.now();
            var msg: [300]u8 = undefined;
            for (points, 0..) |pt, at| {
                mlx.engine.fault.armAt(pt);
                const r = go(&ld);
                const fired = !mlx.engine.fault.armed();
                mlx.engine.fault.disarm();
                _ = mlx.takeError(&msg);
                errdefer std.debug.print("cooler contract: the image splice refused at point {d} of {d} left {any} -> {any}\n", .{ at, points.len, base, Sheet.now() });
                try testing.expect(fired);
                try testing.expect(std.meta.isError(r));
                // What the refused splice encoded is lazy, not held: the next splice runs it.
                try go(&ld);
                try testing.expectEqual(base, Sheet.now());
            }
        }
    }.run);
}

test "cooler contract: a Cooler image request refused at any pool buffer site a text request lacks frees what it made, the embedding its rows splice into too" {
    try eachArch(struct {
        const Req = struct {
            ld: *Load,
            prompt: []const u32,
            fn go(q: @This()) anyerror!void {
                var got = try greedy(q.ld, q.prompt, 2);
                got.deinit(testing.allocator);
            }
        };

        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            defer mlx.engine.fault.disarm();
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            // A cut fixture's vocabulary may end before its image placeholder: no prompt can carry one.
            if (ld.config.image_token_id == 0 or ld.config.image_token_id >= ld.config.vocab_size) return error.SkipZigTest;
            var prompt: [12]u32 = undefined;
            for (&prompt, 0..) |*t, i| t.* = if (i >= 3 and i < 7) ld.config.image_token_id else ids[i % ids.len];
            {
                const turn = arch_handle.turnFor(ld.h, &ld.config);
                defer turn.end();
                const scope = mlx.engine.enter(ld.xfm.s);
                defer scope.end();
                var rows = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(rows);
                try mlx.check(mlx.mlx_zeros(&rows, &[_]c_int{ 1, 4, @intCast(ld.config.hidden_size) }, 3, .bfloat16, ld.xfm.s));
                try mlx.check(mlx.mlx_array_eval(rows));
                ld.xfm.vision_embeddings = rows;
            }
            defer {
                const turn = arch_handle.turnFor(ld.h, &ld.config);
                defer turn.end();
                _ = mlx.mlx_array_free(ld.xfm.vision_embeddings.?);
                ld.xfm.vision_embeddings = null;
            }
            // A text request's sites are the request sweep's; this one refuses those the image's rows add.
            var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
            defer seen.deinit(a);
            var text: [12]u32 = undefined;
            for (&text, 0..) |*t, i| t.* = ids[i % ids.len];
            a.free(try firstAsks(a, &seen, C, Req{ .ld = &ld, .prompt = &text }, Req.go));
            const points = try firstAsks(a, &seen, C, Req{ .ld = &ld, .prompt = &prompt }, Req.go);
            defer a.free(points);
            const base = Sheet.now();
            var msg: [300]u8 = undefined;
            for (points, 0..) |pt, at| {
                mlx.engine.fault.armAt(pt);
                const r = Req.go(.{ .ld = &ld, .prompt = &prompt });
                const fired = !mlx.engine.fault.armed();
                mlx.engine.fault.disarm();
                if (r) |_| try testing.expect(!mlx.errorPending()) else |_| _ = mlx.takeError(&msg);
                errdefer std.debug.print("cooler contract: an image request refused at point {d} of {d} left {any} -> {any}\n", .{ at, points.len, base, Sheet.now() });
                try testing.expect(fired or !pt.strict);
                // A request leaves the load's caches as it found them only once one runs whole after the refusal.
                try Req.go(.{ .ld = &ld, .prompt = &prompt });
                try testing.expectEqual(base, Sheet.now());
            }
        }
    }.run);
}

test "cooler contract: a Cooler load's speculative requests ask nothing of MLX, and the unload gives back everything" {
    try eachLayout(struct {
        /// A speculative request, and the MLX calls it made to its end. The turn's end is not counted: returning to a
        /// side not at rest pays MLX what it is owed for its own earlier launches (`mlx.payOwed`), which no request asked.
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
            const base = Sheet.now();
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            {
                errdefer _ = cooler.close();
                try testing.expect(C.headLoaded(cooler.xfm));
                // One row, past the verify edge, a chunk: the forwards a prompt length can take.
                for ([_]usize{ 1, 9, 40 }) |len| {
                    const prompt = ids[0..@min(ids.len, len)];
                    var asked: u64 = 0;
                    var got = try specAsked(&cooler, C, prompt, &asked);
                    got.deinit(testing.allocator);
                    errdefer std.debug.print("cooler contract: a speculative request over a prompt of {d} tokens asked MLX {d} times\n", .{ prompt.len, asked });
                    try testing.expectEqual(@as(u64, 0), asked);
                }
            }
            _ = cooler.close();
            try testing.expectEqual(base, Sheet.now());
        }
    }.run);
}

test "cooler contract: a Cooler load's warm that ends seals the engine, and one refused at a pool buffer fails the load" {
    try eachArch(struct {
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
    try eachArch(struct {
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
    try eachResidency(struct {
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
                    try testing.expect(st.pool_free <= rest);
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
    try eachArch(struct {
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
    try eachArch(struct {
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
            // The sampler's draw follows no prompt length: each length's forward is the greedy cell's.
            const prompt = ids[0..@min(ids.len, 9)];
            for (cases) |sampling| {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                const calls = mlx.engine.mlxCalls();
                errdefer std.debug.print("cooler contract: sampled speculative request over {d} tokens, seed {?d}, top_k {d}\n", .{ prompt.len, sampling.seed, sampling.top_k });
                var got = try C.specTokensWith(a, cooler.xfm, prompt, 12, sampling);
                defer got.deinit(a);
                try testing.expectEqual(@as(usize, 12), got.items.len);
                try testing.expectEqual(calls, mlx.engine.mlxCalls());
            }
        }
    }.run);
}

test "cooler contract: a seeded sampled speculative request says the seeded serial request's tokens, again after another request and at every pinned depth" {
    try eachFixture(struct {
        fn spec(ld: *Load, comptime C: type, prompt: []const u32, n: u32, sampling: SamplingParams, depth: u32) !std.ArrayList(u32) {
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            defer arch_sched.endRequest(&ld.config);
            return C.specTokensAt(testing.allocator, ld.xfm, prompt, n, sampling, depth);
        }

        fn differs(what: []const u8, depth: u32, sampling: SamplingParams, len: usize, want: []const u32, got: []const u32) bool {
            const at = std.mem.indexOfDiff(u32, want, got) orelse return false;
            std.debug.print("cooler contract: seed {?d} over {d} tokens: {s} (pinned depth {d}) differs from token {d}\n", .{ sampling.seed, len, what, depth, at });
            return true;
        }

        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specTokensAt") or !@hasDecl(C, "specIsGreedy")) return error.SkipZigTest;
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            // Verify rows another plan computes at each row count give no position one token.
            if (!C.specIsGreedy(&ld.config)) return error.SkipZigTest;
            const depth_bounds = @import("mtp_depth_bounds.zig");
            const saved = depth_bounds.active;
            defer depth_bounds.active = saved;
            var long: [300]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[(i * 7 + i / ids.len) % ids.len];
            const cases = [_]SamplingParams{
                .{ .temperature = 0.8, .top_k = 20, .seed = 42 },
                .{ .temperature = 1.0, .top_p = 0.95, .top_k = 40, .min_p = 0.05, .seed = 11 },
            };
            var bad: u32 = 0;
            const len = long.len;
            const prompt: []const u32 = &long;
            for (cases) |sampling| {
                var want = try serial(&ld, prompt, 32, sampling);
                defer want.deinit(a);
                var first = try spec(&ld, C, prompt, 32, sampling, 0);
                defer first.deinit(a);
                if (differs("the speculative request against the serial one", 0, sampling, len, want.items, first.items)) bad += 1;
                // Another request between publishes the acceptance the next one plans from.
                {
                    var other = try spec(&ld, C, long[0..100], 24, .{ .temperature = 0 }, 0);
                    other.deinit(a);
                    var again = try spec(&ld, C, prompt, 32, sampling, 0);
                    defer again.deinit(a);
                    if (differs("the speculative request after another against the first", 0, sampling, len, first.items, again.items)) bad += 1;
                }
                for ([_]u32{ 1, 3, 6 }) |depth| {
                    depth_bounds.active = .{ .min = depth, .max = depth };
                    defer depth_bounds.active = saved;
                    var pinned = try spec(&ld, C, prompt, 32, sampling, depth);
                    defer pinned.deinit(a);
                    if (differs("the speculative request against the first", depth, sampling, len, first.items, pinned.items)) bad += 1;
                }
            }
            try testing.expectEqual(@as(u32, 0), bad);
        }
    }.run);
}

/// One greedy request under a thinking bound, ticked in the scheduler's order (`loopGuardTick` before every tick,
/// `commitThoughtClose` when it fires).
const BoundRun = struct {
    /// `drained` decodes serial but leaves the pipeline before the thought's last token, as an MTP serial block's exit
    /// does, so the bound fires with no token in flight whatever the model accepts; `mtp` runs rounds at a pinned depth.
    const Mode = union(enum) { serial, drained, mtp: u32 };

    tokens: std.ArrayList(u32) = .empty,
    /// The logits the forced commit leaves for the token after it.
    logits: []f32 = &.{},
    /// The bound fired on a speculative round's exit (no pipelined token in flight).
    at_exit: bool = false,

    fn deinit(self: *BoundRun, a: std.mem.Allocator) void {
        self.tokens.deinit(a);
        a.free(self.logits);
    }

    fn run(ld: *Load, comptime C: type, prompt: []const u32, n: u32, budget: u32, forced: []const u32, mode: Mode) !BoundRun {
        const a = testing.allocator;
        const scheduler = @import("scheduler.zig");
        const turn = arch_handle.turnFor(ld.h, &ld.config);
        defer turn.end();
        defer arch_sched.endRequest(&ld.config);
        var bound = @import("mtp_plan.zig").ThinkBound{ .budget = budget, .opener_id = null, .closer_id = forced[forced.len - 2], .forced = forced, .in_think = true };
        var tok = Tokenizer.initEmpty(a);
        defer tok.deinit();
        try ld.xfm.resetCache();
        const options: Generator.InitOptions = if (mode == .mtp) C.specOptions(ld.xfm, mode.mtp) else .{};
        var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, prompt, n, .{ .temperature = 0.0, .think_bound = &bound }, &.{}, options);
        defer gen.deinit(a);
        var out: BoundRun = .{};
        errdefer out.deinit(a);
        while (true) {
            bound.observe(gen.generated_ids.items);
            if (bound.due()) {
                bound.fired = true;
                out.at_exit = gen.pending == null;
                const r = try gen.commitForcedTokens(a, forced);
                defer a.free(r.emitted);
                try out.tokens.appendSlice(a, r.emitted);
                if (r.stopped) break;
                out.logits = try test_support.qwen4ReadF32(a, gen.pending.?.logits, ld.xfm.s);
                if (!try gen.mtpResumeAfterForced(a)) {
                    gen.spec_disabled_runtime = true;
                    gen.spec_disable_reason = .think_bound;
                }
                continue;
            }
            if (mode == .mtp and !scheduler.thinkBoundForcesPlain(&gen)) {
                gen.mtp_group_cap = 0;
                scheduler.capMtpRoundToThinkBound(&gen);
                const r = (try gen.nextMtp(a)) orelse break;
                defer a.free(r.tokens);
                try out.tokens.appendSlice(a, r.tokens);
            } else if (mode == .drained and bound.room() == 1) switch (try gen.drainPipelineForSpec(a)) {
                .drained => |t| try out.tokens.append(a, t),
                .already_clean => return error.NoPipelineToDrain,
                .stopped => break,
            } else try out.tokens.append(a, (try gen.next(a)) orelse break);
        }
        if (!bound.fired) return error.BoundNeverFired;
        return out;
    }
};

test "cooler contract: a greedy request's thought closed by its budget says the serial request's tokens, whether the close lands on a serial tick or a speculative round's exit" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specOptions") or !@hasDecl(C, "specIsGreedy")) return error.SkipZigTest;
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            if (!C.specIsGreedy(&ld.config)) return error.SkipZigTest;
            const depth_bounds = @import("mtp_depth_bounds.zig");
            const saved = depth_bounds.active;
            defer depth_bounds.active = saved;
            // As wide as a served close (an early-stop line, the closer, a separator), past the verify rows.
            var forced: [24]u32 = undefined;
            for (&forced, 0..) |*t, i| t.* = ids[(i * 5 + 3) % ids.len];
            forced[forced.len - 2] = @intCast(ld.config.vocab_size - 1);
            var bad: u32 = 0;
            for ([_]u32{ 3, 6, 10, 15 }) |budget| {
                var want = try BoundRun.run(&ld, C, ids, 64, budget, &forced, .serial);
                defer want.deinit(a);
                try testing.expect(!want.at_exit);
                const modes = [_]BoundRun.Mode{ .drained, .{ .mtp = 1 }, .{ .mtp = 2 }, .{ .mtp = 3 }, .{ .mtp = 6 } };
                for (modes) |mode| {
                    if (mode == .mtp) depth_bounds.active = .{ .min = mode.mtp, .max = mode.mtp };
                    defer depth_bounds.active = saved;
                    var got = try BoundRun.run(&ld, C, ids, 64, budget, &forced, mode);
                    defer got.deinit(a);
                    if (mode == .drained) try testing.expect(got.at_exit);
                    const tokens_at = std.mem.indexOfDiff(u32, want.tokens.items, got.tokens.items);
                    const logits_same = std.mem.eql(u8, std.mem.sliceAsBytes(want.logits), std.mem.sliceAsBytes(got.logits));
                    if (tokens_at == null and logits_same) continue;
                    bad += 1;
                    const depth = if (mode == .mtp) mode.mtp else 0;
                    const flight = if (got.at_exit) "no token in flight" else "a token in flight";
                    const logits = if (logits_same) "equal" else "differ";
                    std.debug.print("cooler contract: budget {d}, {s} depth {d}, closed with {s}: tokens differ from {?d}, logits after the close {s}\n", .{ budget, @tagName(mode), depth, flight, tokens_at, logits });
                }
            }
            try testing.expectEqual(@as(u32, 0), bad);
        }
    }.run);
}

test "cooler contract: after a Cooler load's served warm, greedy requests of every prompt length to 40, and speculative and sampled ones, compile no kernel" {
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
                // Each length is its own prefill shape; a speculative request's verify and draw take the same shapes
                // after any of them, so one length runs them.
                if (len != long.len) {
                    if (first_late == null and mlx.engine.latePipelines() != late) first_late = len;
                    continue;
                }
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

/// A request of `ids` whose prefill runs at `chunk` rows, its Generator made (the prefill runs in it) and handed to `body`.
fn withPrefilled(ld: *Load, ids: []const u32, n: u32, chunk: usize, ctx: anytype, comptime body: fn (@TypeOf(ctx), *Generator) anyerror!void) !void {
    const a = testing.allocator;
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, ids, n, .{ .temperature = 0.0 }, &.{}, .{ .pinned_prefill_chunk = chunk });
    defer gen.deinit(a);
    try body(ctx, &gen);
}

test "cooler contract: an exact Cooler prefill bill is what each kind of layer asks at its peak, within the one margin either way" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (!@hasDecl(C, "prefill_kind_sets")) return error.SkipZigTest;
            const bills = @import("bills.zig");
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            if (bills.coolerPrefillRowBytes(&ld.config) == 0) return error.SkipZigTest;
            const kv = bills.configuredKvQuantFor(&ld.config);
            const kv_bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
            const a = testing.allocator;
            defer C.standInAllBut(null);
            defer bills.cooler_rows_for_test = null;
            defer bills.qsa_off_for_test = false;
            for (C.prefill_kind_sets) |kinds| {
                C.standInAllBut(kinds);
                bills.cooler_rows_for_test = C.prefillRowsOf(&ld.config, kinds);
                bills.qsa_off_for_test = !C.selectsBlocks(kinds);
                for ([_]usize{ 512, 8192 }) |width| {
                    const wide = try a.alloc(u32, width);
                    defer a.free(wide);
                    for (wide, 0..) |*t, i| t.* = ids[i % ids.len];
                    const grew = try chunkAsked(&ld, wide);
                    const bill = bills.prefillChunkCost(&ld.config, kv_bits, width, width);
                    // The margin covers the small tensors no term names, and no term may bill more than a margin it does not
                    // hold beyond the allowance the bill names: the rows' in-flight planes, and the f16 weight a decoded
                    // trunk linear holds at its own matmul (`bills.trunkDecodeScratchBytes`, NAX only), never at a chunk's
                    // peak: the stq4 cut's 8192-row attention chunk asks 1,933,856 KB on the M5 and on the M1 alike.
                    const allowance: u64 = (if (@hasDecl(C, "prefillAllowanceRows")) width * C.prefillAllowanceRows(&ld.config) else 0) +
                        bills.trunkDecodeScratchBytes(&ld.config, width);
                    if (grew > bill + bills.BILL_MARGIN_BYTES or bill > grew + allowance + bills.BILL_MARGIN_BYTES) std.debug.print("cooler contract: a {d}-row chunk running {any} asked {d} KB against its {d} KB bill\n", .{ width, kinds, grew >> 10, bill >> 10 });
                    try testing.expect(grew <= bill + bills.BILL_MARGIN_BYTES);
                    try testing.expect(bill <= grew + allowance + bills.BILL_MARGIN_BYTES);
                }
            }
        }
    }.run);
}

/// The bytes the engine's pool has in use at its fullest over one prefill chunk of `ids` from an empty cache, past what
/// it had in use before.
fn chunkAsked(ld: *Load, ids: []const u32) !u64 {
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    try ld.xfm.resetCache();
    mlx.engine.close();
    mlx.engine.trim();
    const in0 = mlx.engine.poolTotal() - mlx.engine.poolFree();
    mlx.engine.resetPoolPeak(.process);
    const arr = mlx.mlx_array_new_data(@ptrCast(ids.ptr), &[_]c_int{ 1, @intCast(ids.len) }, 2, .int32);
    defer _ = mlx.mlx_array_free(arr);
    var ctx = ld.xfm.defaultCtx();
    // A served chunk reads out no row of a prompt it continues.
    ctx.skip_lm_head = true;
    const hidden = try @import("model_handle.zig").forwardWith(ld.xfm, &ctx, arr);
    defer _ = mlx.mlx_array_free(hidden);
    try mlx.check(mlx.mlx_array_eval(hidden));
    mlx.engine.close();
    return mlx.engine.poolPeak(.process) -| in0;
}

test "cooler contract: a wide forward that reads out every row's logits, as `sushi kld` and the decode meter run one, runs under its cap" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            const a = testing.allocator;
            for ([_]usize{ 418, 2048 }) |width| {
                const wide = try a.alloc(u32, width);
                defer a.free(wide);
                for (wide, 0..) |*t, i| t.* = ids[i % ids.len];
                const turn = arch_handle.turnFor(ld.h, &ld.config);
                defer turn.end();
                defer arch_sched.endRequest(&ld.config);
                try ld.xfm.resetCache();
                const arr = mlx.mlx_array_new_data(@ptrCast(wide.ptr), &[_]c_int{ 1, @intCast(width) }, 2, .int32);
                defer _ = mlx.mlx_array_free(arr);
                var ctx = ld.xfm.defaultCtx();
                errdefer std.debug.print("cooler contract: a {d}-row forward reading out every row\n", .{width});
                const logits = try @import("model_handle.zig").forwardWith(ld.xfm, &ctx, arr);
                defer _ = mlx.mlx_array_free(logits);
                try mlx.check(mlx.mlx_array_eval(logits));
                const read: usize = if (@hasDecl(C, "reads_out_last_row_only")) 1 else width;
                try testing.expectEqual(@as(c_int, @intCast(read)), mlx.getShape(logits)[1]);
            }
        }
    }.run);
}

test "cooler contract: a decode after a long Cooler prefill grows the engine's pool by no more than its forward's cap" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            const a = testing.allocator;
            const len: usize = 4097;
            const long = try a.alloc(u32, len);
            defer a.free(long);
            for (long, 0..) |*t, i| t.* = ids[i % ids.len];
            const Body = struct {
                fn decode(ld: *Load, gen: *Generator) anyerror!void {
                    // From what the prefill left in use: each step's scratch, within a one-row forward's cap.
                    mlx.engine.close();
                    const in0 = Sheet.now().in_use;
                    mlx.engine.resetPoolPeak(.process);
                    var steps: usize = 0;
                    while (try gen.next(testing.allocator)) |_| steps += 1;
                    mlx.engine.close();
                    const grew = mlx.engine.poolHeldPeak(.process) -| in0;
                    const cap = @import("arch/cooler.zig").forwardCap(&ld.config, 1, len + steps);
                    if (grew > cap) std.debug.print("cooler contract: {d} decode steps after a {d}-token prefill grew the pool {d} KB over its {d} KB cap\n", .{ steps, len, grew >> 10, cap >> 10 });
                    try testing.expect(grew <= cap);
                }
            };
            try withPrefilled(&cooler, long, 16, 4096, &cooler, Body.decode);
        }
    }.run);
}

test "cooler contract: a Cooler prefill chunk, the widest on the load's first request and any width on a later one, grows the engine's pool by no more than the chunk's bill" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            const bills = @import("bills.zig");
            const kv = bills.configuredKvQuantFor(&ld.config);
            const top: usize = 8192;
            const long = try a.alloc(u32, top + 1);
            defer a.free(long);
            for (long, 0..) |*t, i| t.* = ids[i % ids.len];
            const cfg = &ld.xfm.config;
            // The widest is the load's first request; a narrower width, then the widest again, run after another ended.
            for ([_]usize{ top, 1024, top }, 0..) |pin, at| {
                const prompt = long[0 .. pin + 1];
                const width = prefill_plan.requestPrefillChunk(cfg.prefillScoreHeadDim(), cfg.num_attention_heads, prompt.len, cfg.has_sliding_window, cfg.isMoe(), cfg.longCtxGated(), pin);
                const bill = bills.prefillChunkHeldMax(&ld.config, if (kv.scheme == .off) 16 else kv.bits, width, prompt.len);
                mlx.engine.trim();
                mlx.engine.resetPoolPeak(.process);
                const held = mlx.engine.poolTotal();
                errdefer std.debug.print("cooler contract: request {d}, {d} tokens pinned at {d} rows a chunk ({d} served), bill {d} MB\n", .{ at, prompt.len, pin, width, bill >> 20 });
                var got = try serialWith(&ld, prompt, 1, .{ .temperature = 0.0 }, .{ .pinned_prefill_chunk = pin });
                got.deinit(a);
                const grew = mlx.engine.poolHeldPeak(.process) -| held;
                if (grew > bill) std.debug.print("cooler contract: the pool grew {d} MB\n", .{grew >> 20});
                try testing.expect(grew <= bill);
            }
        }
    }.run);
}

test "cooler contract: a served cold prefill past the reservation floor, whose first chunk grows the request's whole reserved cache, runs under its chunks' caps" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            // The owner's agent turn, 40,388 tokens cold at 4,096 a chunk. A cut's one attention layer reserves the
            // context's whole rest to outgrow its first chunk's cap, as the real pack's twelve do at 256,000.
            const len: usize = 40388;
            const max_tokens: u32 = 1_000_000;
            if (!ld.config.reservesKvCapacity() or ld.config.max_position_embeddings < len + max_tokens) return error.SkipZigTest;
            const long = try a.alloc(u32, len);
            defer a.free(long);
            for (long, 0..) |*t, i| t.* = ids[i % ids.len];
            const Body = struct {
                fn reserved(l: *Load, _: *Generator) anyerror!void {
                    try testing.expect(l.xfm.cache.reserve_tokens >= len + max_tokens);
                }
            };
            errdefer std.debug.print("cooler contract: a cold {d}-token prefill at 4096 rows a chunk reserving {d} more\n", .{ len, max_tokens });
            try withPrefilled(&ld, long, max_tokens, 4096, &ld, Body.reserved);
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
            // A long chunk's remainder, or a restored prompt's tail, past the budget: wider than the narrow sweep,
            // narrower than the widest chunk.
            got = try serialWith(&cooler, past, 1, .{ .temperature = 0 }, .{ .pinned_prefill_chunk = 100 });
            got.deinit(a);
            if (first_late == null and mlx.engine.latePipelines() != late) first_late = "a 100-row chunk past the budget";
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
                // Each penalty keeps MTP on: greedy and sampled, every penalty a request may set.
                for ([_]@import("sampler.zig").SamplingParams{
                    .{ .temperature = 0, .repeat_penalty = 1.3 },
                    .{ .temperature = 0, .presence_penalty = 1.0, .frequency_penalty = 0.7 },
                    .{ .temperature = 0.8, .top_k = 20, .seed = 3, .repeat_penalty = 1.3, .presence_penalty = 1.0, .frequency_penalty = 0.7 },
                }) |sp| {
                    const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                    defer turn.end();
                    defer arch_sched.endRequest(&cooler.config);
                    var spec = try C.specTokensWith(a, cooler.xfm, past, 48, sp);
                    spec.deinit(a);
                    if (first_late == null and mlx.engine.latePipelines() != late) first_late = "speculative with penalties";
                }
                // A context that crosses the budget: attention stays dense to there, over keys enough for two passes.
                if (budget > 16) for ([_]SamplingParams{ .{ .temperature = 0 }, .{ .temperature = 0.7, .top_p = 0.9, .seed = 7 } }) |sp| {
                    const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                    defer turn.end();
                    defer arch_sched.endRequest(&cooler.config);
                    var spec = try C.specTokensWith(a, cooler.xfm, past[0 .. budget - 8], 48, sp);
                    spec.deinit(a);
                    if (first_late == null and mlx.engine.latePipelines() != late) first_late = "speculative across the budget";
                };
                // Concurrent requests whose drafter rounds join, two rows and three, greedy and sampled.
                const drafter = @import("generate.zig").MtpHeadRef.fromModel(cooler.xfm, true, null).?;
                for ([_]SamplingParams{ .{ .temperature = 0 }, .{ .temperature = 0.7, .top_p = 0.9, .seed = 7 } }) |sp| {
                    const prompts = [_][]const u32{ past[0..40], past[0..101], past[0..77] };
                    for ([_]usize{ 2, 3 }) |rows| {
                        const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                        defer turn.end();
                        defer arch_sched.endRequest(&cooler.config);
                        try @import("cooler_warm.zig").joinedRounds(io, a, cooler.xfm, drafter, prompts[0..rows], sp, 24);
                        if (first_late == null and mlx.engine.latePipelines() != late) first_late = "joined speculative rows";
                    }
                }
                // An image request: its prefill and its verify rows splice the tower's rows and rotate by M-RoPE.
                if (cooler.config.image_token_id != 0 and cooler.config.image_token_id < cooler.config.vocab_size) {
                    for ([_]?@import("generate.zig").MtpHeadRef{ drafter, null }) |d| {
                        const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                        defer turn.end();
                        defer arch_sched.endRequest(&cooler.config);
                        try @import("cooler_warm.zig").imageRequest(io, a, cooler.xfm, d, .{ .temperature = 0 }, 300, 64, 24);
                        if (first_late == null and mlx.engine.latePipelines() != late) first_late = "an image request";
                    }
                }
                // Through the scheduler: two requests decoded at once with the MTP head, whose ticks join their rows,
                // and a request that restores a longer one's prompt past the budget and drafts from there.
                var sv: Served = undefined;
                try sv.init(&cooler, dir);
                defer sv.deinit();
                try sv.concurrent(&.{ wide[0..40], wide[0..101] }, 64, .{ .temperature = 0 }, true);
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = "two concurrent requests";
                // Long enough that the tier holds restore points under it, as the SSD contract's prompt is.
                const restored = @max(budget, 1300) + 300;
                try sv.concurrent(&.{wide[0..restored]}, 8, .{ .temperature = 0 }, true);
                try sv.concurrent(&.{wide[0 .. restored + 400]}, 32, .{ .temperature = 0 }, true);
                try testing.expect(sv.reused > 0);
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = "a request restored past the budget";
            }
            if (cooler.config.image_token_id != 0) {
                // An image request's prefill splices the tower's rows in at its placeholders, in the trunk's forward.
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                const s = cooler.xfm.s;
                const rows: c_int = 300;
                const hidden: c_int = @intCast(cooler.config.hidden_size);
                var tids: [300]i32 = undefined;
                for (&tids, 0..) |*t, i| t.* = if (i >= 40 and i < 104) @intCast(cooler.config.image_token_id) else @intCast(ids[i % ids.len]);
                const scope = mlx.engine.enter(s);
                defer scope.end();
                const tid = try mlx.newData(&tids, &[_]c_int{ 1, rows }, .int32);
                defer _ = mlx.mlx_array_free(tid);
                var ve = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ve);
                try mlx.check(mlx.mlx_zeros(&ve, &[_]c_int{ 1, 64, hidden }, 3, .bfloat16, s));
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_zeros(&h, &[_]c_int{ 1, rows, hidden }, 3, .bfloat16, s));
                const spliced = try @import("trunk.zig").Trunk.spliceVisionRows(h, tid, ve, cooler.config.image_token_id, cooler.config.video_token_id, 0, s);
                defer _ = mlx.mlx_array_free(spliced);
                try mlx.check(mlx.mlx_array_eval(spliced));
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = "an image's rows spliced into the prompt";
            }
            errdefer std.debug.print("cooler contract: from {?s} on, requests compiled:\n{s}", .{ first_late, mlx.engine.late_names.text() });
            try testing.expectEqual(late, mlx.engine.latePipelines());
        }
    }.run);
}

test "cooler contract: a pipeline a Cooler request compiles late is written to the load's manifest, for the next load to make ahead" {
    try eachArch(struct {
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
    try eachArch(struct {
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

test "cooler contract: a Cooler load whose widest served request holds more than its bill is refused by name" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            const shared = @import("arch/cooler_pipelines.zig");
            const cooler_warm = @import("cooler_warm.zig");
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            // A fresh manifest, so the load runs its served warm.
            shared.root_hook = tmp.dir;
            defer shared.root_hook = null;
            defer shared.forget();
            cooler_warm.bill_for_test = 1 << 20;
            defer cooler_warm.bill_for_test = null;
            var cooler: Load = undefined;
            if (cooler.open(C, dir, true)) |_| {
                _ = cooler.close();
                return error.TestUnexpectedResult;
            } else |e| try testing.expectEqual(error.WarmOverBill, e);
            var msg: [512]u8 = undefined;
            _ = mlx.takeError(&msg);
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

/// Frames from each phase's root a load's sweep tells sites by: one site per step of the phase. `load`: the weights,
/// the trunk's init, its closures, the handle. `serve`: the cache, the draft rerank, the warm; what runs inside a step
/// unwinds through that step's cleanup, and the forwards inside the warm are the request sweeps'.
const load_sweep_frames = std.enums.EnumArray(Phase, usize).init(.{ .load = 3, .serve = 3 });

test "cooler contract: a Cooler load failed at the first checked MLX call of any step, or the last of a bring-up step, leaves nothing of it behind" {
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
                // One clean load records the phase's steps; each is failed at its first call (nothing of it made), and a
                // bring-up step at its last too (all of it made but that call); the warm's tail varies by a call from load
                // to load. A fixture whose phase makes the very calls another's did is that path again, and is not swept
                // twice.
                const R = struct {
                    dir: []const u8,
                    phase: Phase,
                    fn go(r: @This()) anyerror!void {
                        if ((try faultedLoad(C, r.dir, r.phase, 0)).left) return error.CleanLoadLeft;
                    }
                };
                mlx.fault.sites = .{ .from = .root, .frames = load_sweep_frames.get(phase), .lasts = phase == .load };
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
    try eachLayout(S.run);
}

test "cooler contract: requests, repeated, leave the engine as they found it, and an unload gives back everything, load after load" {
    try eachResidency(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const prompt = ids[0..@min(ids.len, 30)];
            // A second load is what meets a first load's leftovers.
            for (0..2) |_| {
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
                    var g = try greedy(&cooler, prompt, 6);
                    g.deinit(testing.allocator);
                    try testing.expectEqual(warm, Sheet.now());
                }
                _ = cooler.close();
                try testing.expectEqual(before, Sheet.now());
            }
        }
    }.run);
}

test "cooler contract: a Cooler request failed at any of its checked MLX call sites leaves the engine as a whole request does, and the next says the same tokens" {
    var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer seen.deinit(testing.allocator);
    const S = struct {
        var sites: *std.AutoHashMapUnmanaged(u64, void) = undefined;
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
            // Each op under its callers fails once, at its first call, on the first fixture of the arch that reaches it.
            var list: std.ArrayList(u64) = .empty;
            defer list.deinit(testing.allocator);
            mlx.fault.sites = .{ .list = &list, .gpa = testing.allocator };
            const recorded = Req.go(.{ .ld = &cooler, .prompt = prompt });
            const unrecorded = mlx.fault.sites.failed;
            mlx.fault.sites = .{};
            try recorded;
            if (unrecorded) return error.OutOfMemory;
            var points: std.ArrayList(u64) = .empty;
            defer points.deinit(testing.allocator);
            for (list.items, 1..) |site, k| {
                if (!(try sites.getOrPut(testing.allocator, archSite(C, site))).found_existing) try points.append(testing.allocator, k);
            }
            for (points.items) |k| {
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
    };
    S.sites = &seen;
    try eachLayout(S.run);
}


test "cooler contract: an exact Cooler speculative request says its greedy tokens, over one row, past the verify edge and a chunk" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specIsGreedy")) return error.SkipZigTest;
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            if (!C.specIsGreedy(&ld.config)) return error.SkipZigTest;
            for ([_]usize{ 1, 9, 40 }) |len| {
                const prompt = ids[0..@min(ids.len, len)];
                errdefer std.debug.print("cooler contract: a prompt of {d} tokens\n", .{prompt.len});
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
    /// A fault sweep refuses requests on purpose: their failures are expected, not news.
    quiet_failures: bool = false,

    fn init(self: *Served, ld: *Load, dir: []const u8) !void {
        const a = testing.allocator;
        self.reused = 0;
        self.quiet_failures = false;
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
        self.sch.force_batched = false;
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
            if (!self.quiet_failures) std.debug.print("cooler contract: the served request failed ({s})\n", .{e});
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

    /// Requests through the scheduler, each prefilled in turn and then decoded together, their rows joined in its ticks
    /// as concurrent requests are, with the model's MTP head when `mtp`. `reused`: what the last took from a tier.
    fn concurrent(self: *Served, prompts: []const []const u32, n: u32, sampling: SamplingParams, mtp: bool) !void {
        const a = testing.allocator;
        const sch = &self.sch;
        var slots: [4]*scheduler.Slot = undefined;
        var made: usize = 0;
        defer {
            for (slots[0..made]) |slot| sch.complete(slot);
            for (sch.cleanup_queue.items) |s| s.deinit();
            sch.cleanup_queue.clearRetainingCapacity();
        }
        for (prompts) |p| {
            const head = if (mtp) @import("generate.zig").MtpHeadRef.fromModel(arch_handle.trunk(self.entry.handle.?), true, null) else null;
            const slot = try sch.submit(.{ .model = self.entry, .prompt_ids = p, .sampling = sampling, .eos_token_ids = &.{}, .max_tokens = n, .enable_mtp = mtp, .mtp = head });
            slots[made] = slot;
            made += 1;
            _ = sch.pending.orderedRemove(0);
            try scheduler.runPrefill(sch, slot);
            self.reused = slot.cached_tokens;
            if (!scheduler.postPrefillTerminal(slot)) try sch.decoding.append(a, slot);
        }
        while (true) {
            var live: [4]*scheduler.Slot = undefined;
            var k: usize = 0;
            for (slots[0..made]) |slot| if (slot.joinsTick()) {
                live[k] = slot;
                k += 1;
            };
            if (k == 0) break;
            try scheduler.runDecodeTick(sch, live[0..k]);
            scheduler.cullDecoding(sch);
        }
        for (slots[0..made]) |slot| if (slot.error_code) |e| {
            std.debug.print("cooler contract: a served request failed ({s})\n", .{e});
            return error.RequestFailed;
        };
        const hc = &self.entry.prefix_cache.?;
        hc.flushPendingDisk(arch_handle.trunk(self.entry.handle.?).s);
        hc.disk.?.drainWriter();
    }
};

test "cooler contract: a served Cooler request that restores from the SSD tier says a cold request's tokens and asks nothing of MLX" {
    try eachArch(struct {
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
            // Held restore points would answer from RAM; with none held every point is on the SSD tier.
            sv.entry.prefix_cache.?.disk.?.held_cap_for_test = 0;
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

test "cooler contract: a Cooler load's /props names Sushi Cooler as the engine its forwards run on" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            _ = ids;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            try testing.expectEqualStrings("sushi-cooler", @import("models_api.zig").propsSettingsFor(sv.entry).engine);
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

test "cooler contract: a Cooler request refused at any pool buffer site fails by its error, and every later request on the same load says the same tokens" {
    var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer seen.deinit(testing.allocator);
    const S = struct {
        var sites: *std.AutoHashMapUnmanaged(u64, void) = undefined;
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            defer mlx.engine.fault.disarm();
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            var msg: [300]u8 = undefined;
            // A speculative round's plan follows measured costs and live sync times, so its draft count, and the asks
            // with it, change from request to request; pinned (as --mtp-min-depth 3 --mtp-max-depth 3) every request
            // makes the same asks and the sweep proves each one.
            const depth_bounds = @import("mtp_depth_bounds.zig");
            const bounds = depth_bounds.active;
            defer depth_bounds.active = bounds;
            depth_bounds.active = .{ .min = 3, .max = 3 };
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
                // One clean request records every pool buffer's site; each is refused at its first ask, on the first arm
                // and fixture of the arch that reaches it. A speculative request's asks do not come in the same order
                // every time, so a point is a site's n-th ask.
                const points = try firstAsks(a, sites, C, Req{ .ld = &ld, .prompt = prompt, .spec = spec }, Req.go);
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
    };
    S.sites = &seen;
    try eachLayout(S.run);
}


/// The Cooler contract of the arch registered as `name`, or an empty struct where the build left it out: a shared test
/// of one arch's load checks `@hasDecl` and skips, so the drop of that arch still compiles it.
fn ContractOf(comptime name: []const u8) type {
    if (comptime !@hasField(registry.Tag, name)) return struct {};
    return registry.Mod(@field(registry.Tag, name)).cooler_contract;
}

test "cooler contract: a Cooler prefill hands each chunk's free buffers back, so a long prompt holds no more free than one chunk" {
    const Q = ContractOf("qwen4");
    if (comptime @hasDecl(Q, "prompt")) return prefillHandsChunksBack(Q);
    return error.SkipZigTest;
}

fn prefillHandsChunksBack(comptime Q: type) !void {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
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

test "cooler contract: a decode past its cache's capacity grows it a buffer at a time, holding no more than the new rows past its growth" {
    const Q = ContractOf("qwen4");
    if (comptime @hasDecl(Q, "prompt")) return decodeGrowsInPlace(Q);
    return error.SkipZigTest;
}

fn decodeGrowsInPlace(comptime Q: type) !void {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try Q.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(Q, dir, true);
    defer _ = ld.close();
    // A prompt below the reservation's threshold takes no reservation, so decode grows the cache by its +25% steps.
    const len: usize = 4097;
    const n: u32 = 3072;
    const long = try a.alloc(u32, len);
    defer a.free(long);
    for (long, 0..) |*t, i| t.* = ids[i % ids.len];
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, long, n, .{ .temperature = 0.0 }, &.{}, .{});
    defer gen.deinit(a);
    var steps: usize = 0;
    var in0: u64 = 0;
    var grows0: usize = 0;
    // The decode's own scratch: its peak over what is in use at the step before its first grow.
    var scratch: u64 = 0;
    var before: u64 = 0;
    var grown = false;
    while (try gen.next(a)) |_| {
        steps += 1;
        if (steps > 1 and !grown) {
            if (@import("kv_cache.zig").KVCache.kv_cap_buf_grows > grows0) {
                grown = true;
                scratch = before;
            } else before = mlx.engine.poolHeldPeak(.process) -| in0;
        }
        if (steps == 1) {
            // From the first token on: the prefill's own transients are its bill's.
            mlx.engine.close();
            in0 = Sheet.now().in_use;
            mlx.engine.resetPoolPeak(.process);
            grows0 = @import("kv_cache.zig").KVCache.kv_cap_buf_grows;
        }
    }
    mlx.engine.close();
    try testing.expect(@import("kv_cache.zig").KVCache.kv_cap_buf_grows > grows0);
    const grew = Sheet.now().in_use -| in0;
    var widest: u64 = 0;
    for (ld.xfm.cache.entries) |entry| if (entry.initialized) {
        widest = @max(widest, mlx.engine.poolBytes(mlx.mlx_array_size(entry.keys) * mlx.mlx_array_itemsize(entry.keys)), mlx.engine.poolBytes(mlx.mlx_array_size(entry.values) * mlx.mlx_array_itemsize(entry.values)));
    };
    // What the grows held past the decode's own scratch and the growth itself: the zero tail of the buffer being
    // grown, at most a quarter of it (the cache grows by a quarter). A grow that built a zeros buffer of the whole
    // capacity first held the widest buffer once more (4096 KB on the tiny STQ fixture, against 448 KB).
    const transient = (mlx.engine.poolHeldPeak(.process) -| in0) -| scratch -| grew;
    if (transient > widest / 4) std.debug.print("cooler contract: a decode's KV grows held {d} KB past its growth and scratch, over a quarter of its {d} KB widest buffer\n", .{ transient >> 10, widest >> 10 });
    try testing.expect(transient <= widest / 4);
}

test "cooler contract: a long reserved request holds no more than its admission bill, prefill and decode" {
    const Q = ContractOf("qwen4");
    if (comptime @hasDecl(Q, "prompt")) return requestWithinBill(Q, false);
    return error.SkipZigTest;
}

test "cooler contract: a long reserved speculative request holds no more than its admission bill, prefill and decode" {
    const Q = ContractOf("qwen4");
    if (comptime @hasDecl(Q, "specOptions")) return requestWithinBill(Q, true);
    return error.SkipZigTest;
}

fn requestWithinBill(comptime Q: type, spec: bool) !void {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    var ran = false;
    for ([_][:0]const u8{ "QWEN4_STQ_TINY", "QWEN4_TEST_MODEL" }) |env| {
        if (!fixtureChosen(selection(), env)) continue;
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        try requestWithinBillOn(Q, dir, spec);
        ran = true;
    }
    if (!ran) return error.SkipZigTest;
}

fn requestWithinBillOn(comptime Q: type, dir: []const u8, spec: bool) !void {
    const a = testing.allocator;
    const ids = try Q.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(Q, dir, true);
    defer _ = ld.close();
    const bills = @import("bills.zig");
    // Past the threshold, so the cache reserves the prompt and the whole reply up front.
    const len: usize = @intCast(@import("kv_cache.zig").KVCache.RESERVE_MIN_TOKENS + 1024);
    const n: u32 = 1024;
    const long = try a.alloc(u32, len);
    defer a.free(long);
    for (long, 0..) |*t, i| t.* = ids[i % ids.len];
    const c = &ld.config;
    const kv = bills.configuredKvQuantFor(c);
    const kv_bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
    const chunk = @import("prefill_plan.zig").requestPrefillChunk(c.prefillScoreHeadDim(), c.num_attention_heads, len, c.has_sliding_window, c.isMoe(), c.longCtxGated(), 0);
    const bill = bills.prefillNeededAtChunk(c, len, n, kv_bits, chunk, .{ .mtp_on = spec });
    const turn = arch_handle.turnFor(ld.h, c);
    defer turn.end();
    defer arch_sched.endRequest(c);
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    mlx.engine.close();
    mlx.engine.trim();
    const held0 = mlx.engine.poolTotal();
    mlx.engine.resetPoolPeak(.process);
    var mlx0: usize = 0;
    _ = mlx.mlx_get_active_memory(&mlx0);
    _ = mlx.mlx_reset_peak_memory();
    var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, long, n, .{ .temperature = 0.0 }, &.{}, if (spec) Q.specOptions(ld.xfm, 0) else .{});
    defer gen.deinit(a);
    if (spec) {
        while (try gen.nextMtp(a)) |r| a.free(r.tokens);
    } else while (try gen.next(a)) |_| {}
    mlx.engine.close();
    var mlx_peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&mlx_peak);
    const held = (mlx.engine.poolHeldPeak(.process) -| held0) + (@as(u64, mlx_peak) -| mlx0);
    if (held > bill) std.debug.print("cooler contract: {s}: a {d}-token request with {d} new tokens (MTP {}) held {d} KB over its {d} KB bill\n", .{ dir, len, n, spec, (held - bill) >> 10, bill >> 10 });
    try testing.expect(held <= bill);
}

test "cooler contract: a served request restored from a shorter prompt's SSD record takes it, and says the same tokens every time" {
    // Bytewise cold holds only for a restore at the request's own point (the SSD contract above): another prompt's
    // record sits off this prompt's cold split ([0, N - 31) then the tail), so its state reaches N - 31 in two
    // spans where cold forwards one, and bf16 near-ties may flip. What holds is that the restore is taken and is
    // deterministic.
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var long: [1300]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[(i * 7 + i / ids.len) % ids.len];
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            var said: [2]std.ArrayList(u32) = undefined;
            for (&said) |*got| {
                // A fresh tier each time: the shorter prompt's record is the only one the longer request can restore.
                var sv: Served = undefined;
                try sv.init(&ld, dir);
                defer sv.deinit();
                var from: u32 = 0;
                var shorter = try sv.request(long[0..1299], 24, &from);
                shorter.deinit(a);
                got.* = try sv.request(&long, 24, &from);
                errdefer std.debug.print("cooler contract: the longer request reused {d} tokens of the shorter's\n", .{sv.reused});
                try testing.expectEqual(@as(u32, 1299 - 31), sv.reused);
            }
            defer for (&said) |*got| got.deinit(a);
            try testing.expectEqualSlices(u32, said[0].items, said[1].items);
        }
    }.run);
}

/// The Qwen fixtures a served-path cell runs on: the tiny STQ and, when set, the cut.
fn eachQwenServed(comptime body: fn (comptime Q: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return qwenServedOn(&.{ "QWEN4_STQ_TINY", "QWEN4_TEST_MODEL" }, body);
}

/// The tiny STQ alone: for a sweep, whose sites the cut's load reaches as well.
fn qwenTinyServed(comptime body: fn (comptime Q: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return qwenServedOn(&.{"QWEN4_STQ_TINY"}, body);
}

fn qwenServedOn(envs: []const [:0]const u8, comptime body: fn (comptime Q: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    defer mlx.engine.fault.disarm();
    const Q = ContractOf("qwen4");
    if (comptime !@hasDecl(Q, "prompt")) return error.SkipZigTest;
    var ran = false;
    for (envs) |env| {
        if (!fixtureChosen(selection(), env)) continue;
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        const ids = try Q.prompt(testing.allocator, dir);
        defer testing.allocator.free(ids);
        body(Q, dir, ids) catch |e| return named(env, dir, e);
        ran = true;
    }
    if (!ran) return error.SkipZigTest;
}

/// One served conversation on a fresh tier: `long` cold, another conversation, then `long` again, restored from SSD.
const SsdRestore = struct {
    ld: *Load,
    dir: []const u8,
    long: []const u32,
    other: []const u32,
    restored: ?*std.ArrayList(u32) = null,
    /// The two cold requests alone, no restore.
    cold_only: bool = false,

    fn go(self: SsdRestore) anyerror!void {
        const a = testing.allocator;
        var sv: Served = undefined;
        try sv.init(self.ld, self.dir);
        defer sv.deinit();
        sv.quiet_failures = true;
        var f: u32 = 0;
        var cold = try sv.request(self.long, 12, &f);
        cold.deinit(a);
        var o = try sv.request(self.other, 4, &f);
        o.deinit(a);
        if (self.cold_only) return;
        const got = try sv.request(self.long, 12, &f);
        if (self.restored) |out| out.* = got else {
            var g = got;
            g.deinit(a);
        }
    }
};

test "cooler contract: a served SSD restore refused at any pool buffer site its cold requests lack fails or says the cold tokens, and the next conversation says them" {
    try qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var long: [1300]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var cold: std.ArrayList(u32) = undefined;
            const plain = SsdRestore{ .ld = &ld, .dir = dir, .long = &long, .other = ids[0..@min(ids.len, 20)], .restored = &cold };
            try plain.go();
            defer cold.deinit(a);
            // Each point replays the whole conversation on a fresh tier, so every recorded point is reached again. The
            // restore's own sites only: the cold requests' are the request sweep's.
            var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
            defer seen.deinit(a);
            a.free(try firstAsks(a, &seen, Q, SsdRestore{ .ld = &ld, .dir = dir, .long = &long, .other = ids[0..@min(ids.len, 20)], .cold_only = true }, SsdRestore.go));
            const points = try firstAsks(a, &seen, Q, SsdRestore{ .ld = &ld, .dir = dir, .long = &long, .other = ids[0..@min(ids.len, 20)] }, SsdRestore.go);
            defer a.free(points);
            var msg: [300]u8 = undefined;
            for (points, 0..) |pt, at| {
                var got: std.ArrayList(u32) = .empty;
                mlx.engine.fault.armAt(pt);
                const r = (SsdRestore{ .ld = &ld, .dir = dir, .long = &long, .other = ids[0..@min(ids.len, 20)], .restored = &got }).go();
                const fired = !mlx.engine.fault.armed();
                mlx.engine.fault.disarm();
                if (r) |_| {
                    defer got.deinit(a);
                    errdefer std.debug.print("cooler contract: SSD restore point {d} of {d}: the refused conversation said other tokens\n", .{ at, points.len });
                    try testing.expectEqualSlices(u32, cold.items, got.items);
                } else |_| _ = mlx.takeError(&msg);
                errdefer std.debug.print("cooler contract: SSD restore point {d} of {d} (fired {})\n", .{ at, points.len, fired });
                if (pt.strict) try testing.expect(fired);
                var after: std.ArrayList(u32) = undefined;
                try (SsdRestore{ .ld = &ld, .dir = dir, .long = &long, .other = ids[0..@min(ids.len, 20)], .restored = &after }).go();
                defer after.deinit(a);
                try testing.expectEqualSlices(u32, cold.items, after.items);
            }
        }
    }.run);
}

/// Two served requests decoded as rows of one tick through the scheduler, as serve runs them.
fn servedRows(sv: *Served, prompts: [2][]const u32, n: u32, out: *[2]std.ArrayList(u32)) !void {
    const scheduler = @import("scheduler.zig");
    const a = testing.allocator;
    const sch = &sv.sch;
    var slots: [2]*scheduler.Slot = undefined;
    var made: usize = 0;
    defer {
        for (slots[0..made]) |s| sch.complete(s);
        for (sch.cleanup_queue.items) |s| s.deinit();
        sch.cleanup_queue.clearRetainingCapacity();
    }
    for (0..2) |i| {
        slots[i] = try sch.submit(.{ .model = sv.entry, .prompt_ids = prompts[i], .sampling = .{ .temperature = 0.0 }, .eos_token_ids = &.{}, .max_tokens = n });
        made += 1;
    }
    sch.pending.clearRetainingCapacity();
    var failed: ?anyerror = null;
    for (slots) |s| {
        scheduler.runPrefill(sch, s) catch |e| {
            failed = e;
            s.markError(@errorName(e));
        };
        if (!scheduler.postPrefillTerminal(s)) try sch.decoding.append(a, s);
    }
    while (true) {
        var act: [2]*scheduler.Slot = undefined;
        var k: usize = 0;
        for (sch.decoding.items) |s| if (s.joinsTick()) {
            act[k] = s;
            k += 1;
        };
        if (k == 0) break;
        scheduler.runDecodeTick(sch, act[0..k]) catch |e| {
            failed = e;
            for (act[0..k]) |s| s.markError(@errorName(e));
        };
        scheduler.cullDecoding(sch);
    }
    scheduler.cullDecoding(sch);
    if (failed) |e| return e;
    for (slots, 0..) |s, i| {
        if (s.error_code) |_| return error.RequestFailed;
        out[i] = .empty;
        try out[i].appendSlice(a, s.legacy_gen.?.generated_ids.items);
    }
    const hc = &sv.entry.prefix_cache.?;
    hc.flushPendingDisk(arch_handle.trunk(sv.entry.handle.?).s);
    hc.disk.?.drainWriter();
}

test "cooler contract: two served requests as rows say their solo tokens, and a rows tick refused at any pool buffer site the solo requests lack leaves the next rows theirs and the pool as it was" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            var p1: [37]u32 = undefined;
            for (&p1, 0..) |*t, i| t.* = ids[(i * 3) % ids.len];
            const prompts = [2][]const u32{ ids, &p1 };
            var solo: [2]std.ArrayList(u32) = undefined;
            for (0..2) |i| {
                var f: u32 = 0;
                solo[i] = try sv.request(prompts[i], 16, &f);
            }
            defer for (&solo) |*s| s.deinit(a);
            const Req = struct {
                sv: *Served,
                p: [2][]const u32,
                fn go(q: @This()) anyerror!void {
                    var o: [2]std.ArrayList(u32) = undefined;
                    try servedRows(q.sv, q.p, 16, &o);
                    for (&o) |*x| x.deinit(testing.allocator);
                }
            };
            var rows: [2]std.ArrayList(u32) = undefined;
            try servedRows(&sv, prompts, 16, &rows);
            for (&rows, 0..) |*x, i| {
                defer x.deinit(a);
                try testing.expectEqualSlices(u32, solo[i].items, x.items);
            }
            // The sweep's sites are the cut's too: it runs on the tiny, over the rows tick's own sites.
            if (!std.mem.eql(u8, dir, std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse ""))) return;
            var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
            defer seen.deinit(a);
            const Solo = struct {
                sv: *Served,
                p: [2][]const u32,
                fn go(q: @This()) anyerror!void {
                    for (q.p) |p| {
                        var f: u32 = 0;
                        var got = try q.sv.request(p, 16, &f);
                        got.deinit(testing.allocator);
                    }
                }
            };
            a.free(try firstAsks(a, &seen, Q, Solo{ .sv = &sv, .p = prompts }, Solo.go));
            const points = try firstAsks(a, &seen, Q, Req{ .sv = &sv, .p = prompts }, Req.go);
            defer a.free(points);
            const base = Sheet.now();
            var msg: [300]u8 = undefined;
            for (points, 0..) |pt, at| {
                errdefer std.debug.print("cooler contract: rows point {d} of {d}\n", .{ at, points.len });
                mlx.engine.fault.armAt(pt);
                var o: [2]std.ArrayList(u32) = undefined;
                const r = servedRows(&sv, prompts, 16, &o);
                mlx.engine.fault.disarm();
                if (r) |_| {
                    for (&o, 0..) |*x, i| {
                        defer x.deinit(a);
                        try testing.expectEqualSlices(u32, solo[i].items, x.items);
                    }
                } else |_| _ = mlx.takeError(&msg);
                try testing.expect(!mlx.errorPending());
                var o2: [2]std.ArrayList(u32) = undefined;
                try servedRows(&sv, prompts, 16, &o2);
                for (&o2, 0..) |*x, i| {
                    defer x.deinit(a);
                    try testing.expectEqualSlices(u32, solo[i].items, x.items);
                }
                const now = Sheet.now();
                try testing.expectEqual(base.in_use, now.in_use);
                try testing.expectEqual(@as(u64, 0), now.open);
            }
        }
    }.run);
}


/// `n` ids of real text: Pride and Prejudice (public domain, the bench's corpus) through the fixture's own tokenizer.
fn realTextIds(a: std.mem.Allocator, dir: []const u8, n: usize) ![]u32 {
    const prose = @embedFile("fixtures/prose_pride_and_prejudice.txt");
    var tok = try @import("tokenizer.zig").loadTokenizer(io, a, dir);
    defer tok.deinit();
    const text = try tok.encode(a, prose[0..@min(prose.len, n * 8)]);
    defer a.free(text);
    try testing.expect(text.len >= n);
    return a.dupe(u32, text[0..n]);
}

fn greedyOn(ld: *Load, ids: []const u32, n: u32) !std.ArrayList(u32) {
    const a = testing.allocator;
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, ids, n, .{ .temperature = 0.0 }, &.{}, .{});
    defer gen.deinit(a);
    var got: std.ArrayList(u32) = .empty;
    errdefer got.deinit(a);
    while (try gen.next(a)) |t| try got.append(a, t);
    return got;
}

test "cooler contract: on real text, greedy speculative at depths 1, 3 and 6 says greedy's tokens, short and past the two-pass context (cut)" {
    const Q = ContractOf("qwen4");
    if (comptime !@hasDecl(Q, "prompt")) return error.SkipZigTest;
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    errdefer {
        var msg: [512]u8 = undefined;
        std.debug.print("cooler contract: fixture {s}: {s}\n", .{ dir, mlx.takeError(&msg) orelse "no MLX or engine error latched" });
    }
    const a = testing.allocator;
    const ids = try realTextIds(a, dir, 4097);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(Q, dir, true);
    defer _ = ld.close();
    // Below and past the 2048 context where verify takes the two-pass kernel, the second a chunked prefill; the short
    // prompt arms are the tiny's.
    for ([_]usize{ 17, 4097 }) |len| {
        var greedy_said = try greedyOn(&ld, ids[0..len], 40);
        defer greedy_said.deinit(a);
        // The shallowest and deepest round and one between: each depth re-prefills the prompt.
        for ([_]u32{ 1, 3, 6 }) |depth| {
            var spec = blk: {
                const turn = arch_handle.turnFor(ld.h, &ld.config);
                defer turn.end();
                defer arch_sched.endRequest(&ld.config);
                break :blk try Q.specTokensAt(a, ld.xfm, ids[0..len], 40, .{ .temperature = 0.0 }, depth);
            };
            defer spec.deinit(a);
            errdefer std.debug.print("cooler contract: real text of {d} tokens at depth {d}: speculative differs from greedy at {?d}\n", .{ len, depth, std.mem.indexOfDiff(u32, greedy_said.items, spec.items) });
            try testing.expectEqualSlices(u32, greedy_said.items, spec.items);
        }
    }
}

test "cooler contract: on real text, joined decode rows say their solo rows' tokens for a long chunked prompt beside a short one (cut)" {
    const Q = ContractOf("qwen4");
    if (comptime !@hasDecl(Q, "prompt")) return error.SkipZigTest;
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    errdefer {
        var msg: [512]u8 = undefined;
        std.debug.print("cooler contract: fixture {s}: {s}\n", .{ dir, mlx.takeError(&msg) orelse "no MLX or engine error latched" });
    }
    const a = testing.allocator;
    const ids = try realTextIds(a, dir, 4200);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(Q, dir, true);
    defer _ = ld.close();
    // A long chunked row past the two-pass context beside a short one; three short rows are the tiny's.
    const sets = [_][]const usize{&.{ 4097, 17 }};
    for (sets) |lens| {
        var prompts: [8][]const u32 = undefined;
        for (lens, 0..) |l, i| prompts[i] = ids[i * 7 .. i * 7 + l];
        const got = blk: {
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            defer arch_sched.endRequest(&ld.config);
            break :blk try Q.greedyRows(a, ld.xfm, prompts[0..lens.len], 40);
        };
        defer {
            for (got) |*g| g.deinit(a);
            a.free(got);
        }
        for (prompts[0..lens.len], got) |p, row| {
            const one = blk: {
                const turn = arch_handle.turnFor(ld.h, &ld.config);
                defer turn.end();
                defer arch_sched.endRequest(&ld.config);
                break :blk try Q.greedyRows(a, ld.xfm, &.{p}, 40);
            };
            defer {
                for (one) |*g| g.deinit(a);
                a.free(one);
            }
            errdefer std.debug.print("cooler contract: rows {any}: the row of {d} tokens differs from its solo run at {?d}\n", .{ lens, p.len, std.mem.indexOfDiff(u32, one[0].items, row.items) });
            try testing.expectEqualSlices(u32, one[0].items, row.items);
        }
    }
}

test "cooler contract: a served Qwen conversation keeps its newest off-grid restore points in RAM, writes none while the next turn extends them, and a flush writes the held ones" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const dws = @import("disk_write_stats.zig");
            const ssm = @intFromEnum(dws.Kind.ssm);
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            var prompt: std.ArrayList(u32) = .empty;
            defer prompt.deinit(a);
            for (0..3000) |i| try prompt.append(a, ids[(i * 7 + i / ids.len) % ids.len]);
            const disk = &sv.entry.prefix_cache.?.disk.?;
            const before = dws.process.snapshot().written[ssm];
            var f: u32 = 0;
            var last_point: u32 = 0;
            // Four turns under the 8192 grid: every restore point is off-grid, and each turn restores the one before.
            for (0..4) |turn| {
                var reply = try sv.request(prompt.items, 16, &f);
                defer reply.deinit(a);
                disk.drainWriter();
                if (turn > 0) {
                    try testing.expectEqual(last_point, sv.reused);
                    // A held point is restored from RAM: the hit counts on the RAM tier, never the SSD's.
                    try testing.expectEqual(@import("cache_stats.zig").Tier.hot, sv.entry.prefix_cache.?.lastRestoredTier());
                }
                last_point = @intCast(prompt.items.len - 31);
                try prompt.appendSlice(a, reply.items);
                for (0..1500) |i| try prompt.append(a, ids[(i * 13 + turn * 101) % ids.len]);
            }
            const during = dws.process.snapshot().written[ssm] - before;
            if (during != 0) std.debug.print("cooler contract: the conversation wrote {d} KB of restore points while each turn extended the last\n", .{during >> 10});
            try testing.expectEqual(@as(u64, 0), during);
            @import("scheduler.zig").flushLiveState(sv.entry, "test");
            disk.drainWriter();
            try testing.expect(dws.process.snapshot().written[ssm] - before > 0);
        }
    }.run);
}

/// A Qwen conversation of `n` tokens, its own: `seed` picks the ids.
fn heldPrompt(a: std.mem.Allocator, ids: []const u32, n: usize, seed: usize) !std.ArrayList(u32) {
    var p: std.ArrayList(u32) = .empty;
    errdefer p.deinit(a);
    for (0..n) |i| try p.append(a, ids[(i * 7 + i / ids.len + seed * 131) % ids.len]);
    return p;
}

fn ssmWritten() u64 {
    return @import("disk_write_stats.zig").process.snapshot().written[@intFromEnum(@import("disk_write_stats.zig").Kind.ssm)];
}

test "cooler contract: a held Qwen restore point another conversation pushes out of RAM is written, not dropped, and restores" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            const disk = &sv.entry.prefix_cache.?.disk.?;
            var prompts: [3]std.ArrayList(u32) = undefined;
            for (&prompts, 0..) |*p, k| p.* = try heldPrompt(a, ids, 3000, k + 1);
            defer for (&prompts) |*p| p.deinit(a);
            var f: u32 = 0;
            var r0 = try sv.request(prompts[0].items, 8, &f);
            r0.deinit(a);
            disk.drainWriter();
            try testing.expectEqual(@as(usize, 1), disk.held.items.len);
            const one = disk.held_bytes;
            disk.held_cap_for_test = 2 * one + one / 2;
            var r1 = try sv.request(prompts[1].items, 8, &f);
            r1.deinit(a);
            disk.drainWriter();
            try testing.expectEqual(@as(usize, 2), disk.held.items.len);
            const before = ssmWritten();
            // A third conversation's point: the oldest (the first conversation's) is written, not dropped.
            var r2 = try sv.request(prompts[2].items, 8, &f);
            r2.deinit(a);
            disk.drainWriter();
            try testing.expectEqual(@as(usize, 2), disk.held.items.len);
            try testing.expectEqual(one, ssmWritten() - before);
            var again = try sv.request(prompts[0].items, 8, &f);
            again.deinit(a);
            try testing.expectEqual(@as(u32, 3000 - 31), sv.reused);
        }
    }.run);
}

test "cooler contract: a held Qwen restore point restores the same tokens as its file after a flush" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            const disk = &sv.entry.prefix_cache.?.disk.?;
            var prompt = try heldPrompt(a, ids, 3000, 7);
            defer prompt.deinit(a);
            var f: u32 = 0;
            var cold = try sv.request(prompt.items, 8, &f);
            cold.deinit(a);
            disk.drainWriter();
            const before = ssmWritten();
            var from_ram = try sv.request(prompt.items, 24, &f);
            defer from_ram.deinit(a);
            disk.drainWriter();
            try testing.expectEqual(@as(u32, 3000 - 31), sv.reused);
            try testing.expectEqual(before, ssmWritten());
            // The clean unload's flush: the held point is written, and the next request restores it from its file.
            @import("scheduler.zig").flushLiveState(sv.entry, "test");
            disk.drainWriter();
            try testing.expectEqual(@as(usize, 0), disk.held.items.len);
            try testing.expect(ssmWritten() > before);
            var from_file = try sv.request(prompt.items, 24, &f);
            defer from_file.deinit(a);
            try testing.expectEqual(@as(u32, 3000 - 31), sv.reused);
            try testing.expectEqualSlices(u32, from_ram.items, from_file.items);
        }
    }.run);
}

test "cooler contract: a regenerated Qwen turn restores from its own prompt end, live and after a flush" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            const disk = &sv.entry.prefix_cache.?.disk.?;
            var prompt = try heldPrompt(a, ids, 3000, 11);
            defer prompt.deinit(a);
            var f: u32 = 0;
            var first = try sv.request(prompt.items, 16, &f);
            first.deinit(a);
            disk.drainWriter();
            // The same conversation's next point extends the prompt end (a reply end, or the turn after).
            var longer = try prompt.clone(a);
            defer longer.deinit(a);
            for (0..64) |i| try longer.append(a, ids[(i * 13 + 5) % ids.len]);
            var next = try sv.request(longer.items, 16, &f);
            next.deinit(a);
            disk.drainWriter();
            // Regenerate: the first prompt again restores from its own end, which the newer point did not take away.
            var live = try sv.request(prompt.items, 16, &f);
            defer live.deinit(a);
            disk.drainWriter();
            try testing.expectEqual(@as(u32, 3000 - 31), sv.reused);
            // After the flush an unload or a restart runs, it restores from the written point.
            @import("scheduler.zig").flushLiveState(sv.entry, "test");
            disk.drainWriter();
            try testing.expectEqual(@as(usize, 0), disk.held.items.len);
            var after = try sv.request(prompt.items, 16, &f);
            defer after.deinit(a);
            try testing.expectEqual(@as(u32, 3000 - 31), sv.reused);
            try testing.expectEqualSlices(u32, live.items, after.items);
        }
    }.run);
}

/// Every restore point the SSD tier's index names restores from where it lives: its held bytes or its file.
fn expectListedPointsLive(disk: *@import("kv_disk_cache.zig").DiskTier) !void {
    disk.drainWriter();
    for (disk.entries.items) |e| for (e.ssm_positions) |p| {
        if (disk.heldIndex(e.id, p) != null) continue;
        var buf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/e{d}/s{d:0>7}.safetensors", .{ disk.root, e.id, p });
        errdefer std.debug.print("cooler contract: e{d} lists a restore point at {d} that is neither held nor on disk\n", .{ e.id, p });
        try testing.expect(disk.fs.size(path) != null);
    };
}

test "cooler contract: a Qwen prompt that re-renders an earlier reply restores from that turn's own point, written when the held points pass the cap" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            const disk = &sv.entry.prefix_cache.?.disk.?;
            // Over the 8192 grid, so a written grid point lies under the turns' held points.
            var conv = try heldPrompt(a, ids, 9000, 41);
            defer conv.deinit(a);
            var f: u32 = 0;
            // The second turn's prompt and the first 8 tokens of its reply: an agent's later prompt that re-renders
            // the rest of that reply (a coerced tool call) shares this much.
            var shared: usize = 0;
            for (0..3) |turn| {
                var reply = try sv.request(conv.items, 16, &f);
                defer reply.deinit(a);
                try expectListedPointsLive(disk);
                // Room for one point and a half: a commit keeps its newest point held and writes the spare it supersedes.
                if (turn == 0) disk.held_cap_for_test = disk.held_bytes / disk.held.items.len * 3 / 2;
                if (turn == 1) shared = conv.items.len + 8;
                try conv.appendSlice(a, reply.items);
                for (0..600) |i| try conv.append(a, ids[(i * 13 + turn * 101) % ids.len]);
            }
            var next = try conv.clone(a);
            defer next.deinit(a);
            next.shrinkRetainingCapacity(shared);
            for (0..200) |i| try next.append(a, ids[(i * 17 + 3) % ids.len]);
            var last = try sv.request(next.items, 16, &f);
            last.deinit(a);
            // The second turn's prompt-end point, not the 8192 grid point under it.
            try testing.expectEqual(@as(u32, @intCast(shared - 8 - 31)), sv.reused);
        }
    }.run);
}

test "cooler contract: a held Qwen restore point whose write fails as it gives way to a staged write mid-commit leaves no listed point without bytes" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            const disk = &sv.entry.prefix_cache.?.disk.?;
            var conv = try heldPrompt(a, ids, 3000, 51);
            defer conv.deinit(a);
            var f: u32 = 0;
            var first = try sv.request(conv.items, 16, &f);
            try conv.appendSlice(a, first.items);
            first.deinit(a);
            try expectListedPointsLive(disk);
            const one = disk.held_bytes;
            try testing.expect(one > 0);
            // The next turn's commit stages its chunks with too little permit beside the held points, and their writes fail.
            const permit = disk.writer.?.permit_bytes;
            defer disk.writer.?.permit_bytes = permit;
            disk.writer.?.permit_bytes = one + (1 << 20);
            disk.fail_held_flush_for_test = true;
            defer disk.fail_held_flush_for_test = false;
            for (0..1500) |i| try conv.append(a, ids[(i * 13 + 7) % ids.len]);
            var next = try sv.request(conv.items, 16, &f);
            next.deinit(a);
            try testing.expectEqual(@as(usize, 0), disk.held.items.len);
            try expectListedPointsLive(disk);
        }
    }.run);
}

test "cooler contract: held Qwen restore points give way to a staged write that needs the writer's permit" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            const disk = &sv.entry.prefix_cache.?.disk.?;
            var p0 = try heldPrompt(a, ids, 3000, 11);
            defer p0.deinit(a);
            var p1 = try heldPrompt(a, ids, 3000, 12);
            defer p1.deinit(a);
            var f: u32 = 0;
            var r0 = try sv.request(p0.items, 8, &f);
            r0.deinit(a);
            disk.drainWriter();
            const one = disk.held_bytes;
            try testing.expect(one > 0);
            // Room for the held points, but not beside the next conversation's chunk writes.
            disk.held_cap_for_test = 4 * one;
            const permit = disk.writer.?.permit_bytes;
            defer disk.writer.?.permit_bytes = permit;
            disk.writer.?.permit_bytes = one + (1 << 20);
            const before = ssmWritten();
            var r1 = try sv.request(p1.items, 8, &f);
            r1.deinit(a);
            disk.drainWriter();
            try testing.expectEqual(@as(usize, 0), disk.held.items.len);
            try testing.expect(ssmWritten() - before >= one);
        }
    }.run);
}

/// A four-turn Qwen conversation over the 8192 grid on a fresh tier, then a rewind to its second turn's prompt, whose
/// own held point the turns after it superseded: the rewind's tokens, and the prompt tokens it took from a tier.
fn rewindPastHeld(ld: *Load, dir: []const u8, ids: []const u32, reused: *u32) !std.ArrayList(u32) {
    const a = testing.allocator;
    var conv = try heldPrompt(a, ids, 7000, 21);
    defer conv.deinit(a);
    var rewind: std.ArrayList(u32) = .empty;
    defer rewind.deinit(a);
    var sv: Served = undefined;
    try sv.init(ld, dir);
    defer sv.deinit();
    const disk = &sv.entry.prefix_cache.?.disk.?;
    var f: u32 = 0;
    for (0..4) |turn| {
        var reply = try sv.request(conv.items, 8, &f);
        defer reply.deinit(a);
        disk.drainWriter();
        try conv.appendSlice(a, reply.items);
        for (0..1500) |i| try conv.append(a, ids[(i * 13 + turn * 101) % ids.len]);
        if (turn == 1) try rewind.appendSlice(a, conv.items[0 .. conv.items.len - 1500]);
    }
    const back = try sv.request(rewind.items, 16, &f);
    reused.* = sv.reused;
    return back;
}

test "cooler contract: a rewind past a Qwen conversation's held restore points restores from the grid point, the same tokens every time" {
    // Its tokens are not the cold run's: the grid point's state came from a prefill that started at the turn before's
    // point, off a cold prefill's split, as any restore from another prompt's record is (the shorter-prompt SSD cell).
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var said: [2]std.ArrayList(u32) = undefined;
            for (&said) |*got| {
                var reused: u32 = 0;
                got.* = try rewindPastHeld(&ld, dir, ids, &reused);
                try testing.expectEqual(@as(u32, 8192), reused);
            }
            defer for (&said) |*got| got.deinit(a);
            try testing.expectEqualSlices(u32, said[0].items, said[1].items);
        }
    }.run);
}

/// Runs `body` with `$HOME/.sushi/model-settings.json` holding `row` for `dir`, as the web UI's model settings save it, and
/// each Load stamping it; the parse reads the file's `yarn_factor` itself.
fn withUiSettings(dir: []const u8, row: []const u8, comptime body: fn () anyerror!void) !void {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, ".sushi");
    const text = try std.fmt.allocPrint(a, "{{\"{s}\": {s}}}", .{ dir, row });
    defer a.free(text);
    try tmp.dir.writeFile(io, .{ .sub_path = ".sushi/model-settings.json", .data = text });
    var buf: [512]u8 = undefined;
    const home = try std.fmt.allocPrintSentinel(a, "{s}", .{buf[0..try tmp.dir.realPath(io, &buf)]}, 0);
    defer a.free(home);
    const was = try a.dupeSentinel(u8, std.mem.span(std.c.getenv("HOME") orelse "/tmp"), 0);
    defer a.free(was);
    _ = setenv("HOME", home.ptr, 1);
    defer _ = setenv("HOME", was.ptr, 1);
    settings_from_file = true;
    defer settings_from_file = false;
    try body();
}

test "cooler contract: a Cooler load under each KV precision and context extension the web UI offers takes it and loads on the engine" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            _ = ids;
            const rows = [_]struct { []const u8, []const u8, []const u8 }{
                .{ "{\"kv_quant\": \"off\"}", "kv_quant", "off" },
                .{ "{\"kv_quant\": \"4\"}", "kv_quant", "4" },
                .{ "{\"kv_quant\": \"6\"}", "kv_quant", "6" },
                .{ "{\"yarn_factor\": 2}", "yarn_factor", "2" },
            };
            inline for (rows) |r| {
                const row = r[0];
                const Cell = struct {
                    var d: []const u8 = undefined;
                    fn body() !void {
                        var ld: Load = undefined;
                        try ld.open(C, d, true);
                        defer _ = ld.close();
                        // A row the web UI does not offer for this model (DeepSeek keeps no KV cache to quantize, a
                        // RoPE with no YaRN) has nothing to hold; one it offers must take effect, never skip.
                        const fit = @import("arch/flags.zig").settingsFit(&ld.config);
                        const model_settings = @import("model_settings.zig");
                        if (!model_settings.applies(r[1], fit) or !model_settings.choiceAllowed(r[1], r[2], fit)) return;
                        errdefer std.debug.print("cooler contract: {s} is offered for this model and did not take effect\n", .{row});
                        if (comptime std.mem.eql(u8, r[1], "kv_quant")) {
                            try testing.expect(ld.config.kv_quant_override != null);
                        } else try testing.expect(ld.config.yarn_factor_override != null);
                        try testing.expect(ld.onEngine());
                    }
                };
                Cell.d = dir;
                try withUiSettings(dir, row, Cell.body);
            }
        }
    }.run);
}
