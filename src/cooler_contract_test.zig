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
const JobTurn = model_load.JobTurn;
const Generator = @import("generate.zig").Generator;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const SamplingParams = @import("sampler.zig").SamplingParams;

const static_bill = @import("static_bill.zig");
const io = std.Io.Threaded.global_single_threaded.io();

/// One test load of a fixture, on the side the test names. Its load and its unload run in the job turn a served load
/// and unload take (`model_load.JobTurn`).
/// A Load stamps the model settings file's row for its directory, as a served load does (`model_settings.overrideFor`),
/// instead of none: the cells that hold what the web UI saves (`withUiSettings`).
var settings_from_file = false;

pub const Load = struct {
    a: std.mem.Allocator,
    config: model_mod.ModelConfig,
    weights: model_mod.Weights,
    xfm: *Trunk,
    h: arch_handle.Handle,
    /// Whether the load keeps the arch's vision tower in its weights, as a served load asked for images does.
    vision: bool,
    /// The engine's counts before the load's weights: a load on the ledger bills what they gained (`static_bill`).
    before: static_bill.Before,

    pub fn prepare(self: *Load, comptime C: type, dir: []const u8, cooler: bool) !void {
        return self.prepareOn(testing.allocator, C, dir, cooler);
    }

    /// A load on MLX is the one the user's opt-out flag makes, through the parse and the settings stamp a served load
    /// takes; a Cooler load lifts the tests' MLX default.
    pub fn prepareOn(self: *Load, a: std.mem.Allocator, comptime C: type, dir: []const u8, cooler: bool) !void {
        self.a = a;
        self.vision = false;
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
    pub fn load(self: *Load, dir: []const u8) !void {
        const a = self.a;
        errdefer self.config.deinit(a);
        const job = JobTurn.begin(null, &self.config);
        defer job.end();
        self.before = static_bill.Before.now();
        self.config.keeps_tower = self.vision and self.config.has_vision;
        self.weights = try model_mod.loadWeightsForConfig(io, a, dir, &self.config, self.vision);
        errdefer self.weights.deinit();
        model_mod.resolveWeightPrefix(&self.config, &self.weights);
        self.xfm = try a.create(Trunk);
        errdefer a.destroy(self.xfm);
        // As a served load makes it: the config every cell bills from takes what the load recorded past the parse.
        self.xfm.* = try model_load.initTrunk(io, a, &self.config, &self.weights);
        model_load.publishEngine(self.xfm, &self.config);
        @import("model_load.zig").compileClosures(self.xfm, .all);
        self.h = arch_handle.adopt(self.xfm);
    }

    pub fn open(self: *Load, comptime C: type, dir: []const u8, cooler: bool) !void {
        try self.prepare(C, dir, cooler);
        try self.loadServed(dir);
    }

    /// `open` with the warm skipped, for a cell that asserts nothing the warm compiles or grows.
    pub fn openLight(self: *Load, comptime C: type, dir: []const u8, cooler: bool) !void {
        try self.prepare(C, dir, cooler);
        self.config.test_skip_warm = true;
        try self.loadServed(dir);
    }

    /// `loadServed` with the warm skipped.
    pub fn loadLight(self: *Load, dir: []const u8) !void {
        self.config.test_skip_warm = true;
        try self.loadServed(dir);
    }

    /// `load` and then `serve`, as a served load runs them; on an error nothing of the load stays.
    pub fn loadServed(self: *Load, dir: []const u8) !void {
        try self.load(dir);
        errdefer self.discard();
        try self.serve();
    }

    /// The teardown of a served load that failed (`model_load.failLoadRequest` after the load's own errdefers): the
    /// model and its weights go, and no unload hook runs.
    pub fn discard(self: *Load) void {
        {
            const job = JobTurn.begin(self.h, &self.config);
            defer job.end();
            arch_handle.destroy(self.h, self.a);
            self.weights.deinit();
        }
        static_bill.close(&self.config);
        self.config.deinit(self.a);
    }

    /// What a served load runs once its trunk is up: the KV scheme a launch with no flag gets, the MTP head's draft
    /// rerank, and the warm (`model_load.warmLoad`, fatal on Cooler, which ends sealed). No aux head is loaded.
    pub fn serve(self: *Load) !void {
        const job = JobTurn.begin(self.h, &self.config);
        defer job.end();
        const kv = @import("bills.zig").kvCacheForLaunch(&self.config, null).config;
        try self.xfm.cache.reinit(self.config.num_hidden_layers, kv);
        const mtp_on = self.mtpOn();
        if (mtp_on) _ = try @import("arch/arch_state.zig").buildDraftRerank(self.xfm);
        const policy = @import("mtp_policy.zig");
        const verify_cap: ?u32 = if (mtp_on) @min(policy.resolveMtpDepthCapForProfile(0, .generic), policy.mtpVerifyDraftsMax(&self.config)) else null;
        // An arch on the ledger opens it, fixes its chunk and arena, warms and seals, as a served load does.
        errdefer static_bill.close(&self.config);
        try model_load.openLedger(&self.config, self.xfm, self.before, true, null);
        try model_load.warmLoad(io, self.a, &self.config, self.xfm, verify_cap, null, kv);
        try model_load.sealLedger(&self.config, self.xfm, self.before);
    }

    /// Whether a launch with no MTP flag drafts with the arch's MTP head.
    pub fn mtpOn(self: *Load) bool {
        const mtp = model_load.mtpChoiceFor(null, &self.config);
        return mtp.on and !model_load.mtpDefaultOffUnderStreaming(mtp, self.config.expert_streaming, self.config.stream_mtp_head) and
            !@import("arch/config.zig").fact(&self.config, "no_mtp_head");
    }

    pub fn onEngine(self: *Load) bool {
        return arch_handle.onEngine(self.h, null);
    }

    /// The unload serve runs (`model_load.retire`, in the model's job): the model, its weights, its arch's hook. Returns
    /// the MLX calls the teardown made before the job's end.
    pub fn close(self: *Load) u64 {
        const calls = mlx.engine.mlxCalls();
        const asked = blk: {
            const job = JobTurn.begin(self.h, &self.config);
            defer job.end();
            arch_handle.destroy(self.h, self.a);
            self.weights.deinit();
            arch_serve.unloadModel(&self.config);
            break :blk mlx.engine.mlxCalls() - calls;
        };
        static_bill.close(&self.config);
        self.config.deinit(self.a);
        return asked;
    }
};

/// A run step's served Cooler loads, one per fixture folder, opened by the first request-level cell that asks
/// (`heldLoad`) and kept for the next. After every test the runner reads them back (`heldRest`): each model put
/// between requests, its config unchanged, no dispatch open and the weights' mappings as the last open left them; a
/// test that left otherwise fails by name.
/// The process's end unloads them and asks for an empty pool (`heldClose`).
pub const Held = struct { dir: []u8, ld: *Load, config_hash: u64 };
var held_loads: [8]Held = undefined;
var held_count: usize = 0;
/// The engine with every held load at rest, read after the last open.
var held_sheet: Sheet = undefined;

pub fn heldLoad(comptime C: type, dir: []const u8) !*Load {
    for (held_loads[0..held_count]) |*h| if (std.mem.eql(u8, h.dir, dir)) return h.ld;
    if (held_count == held_loads.len) return error.TooManyHeldLoads;
    unloadHeldOnLedger();
    const a = std.heap.smp_allocator;
    const ld = try a.create(Load);
    errdefer a.destroy(ld);
    try ld.prepareOn(a, C, dir, true);
    ld.config.test_skip_warm = true;
    try ld.loadServed(dir);
    errdefer _ = ld.close();
    const name = try a.dupe(u8, dir);
    held_loads[held_count] = .{ .dir = name, .ld = ld, .config_hash = configHash(&ld.config) };
    held_count += 1;
    const rest = @import("test_rest.zig");
    rest.held = .{ .holds = rest.held.holds + 1, .rest = heldRest, .close = heldClose };
    var none = std.Io.Writer.fixed(&.{});
    _ = heldRest(&none);
    held_sheet = Sheet.now();
    return ld;
}

/// A cell of the pool-cap regime skips a load on the ledger, whose forwards run under no cap: its narrow phases on its
/// plans, its wide ones in its region (D97), its pool its turn's gate (D64). The cell names the ledger's test of the same.
pub fn requirePoolCap(config: *const model_mod.ModelConfig) error{SkipZigTest}!void {
    if (!@import("bills.zig").ledgerArch(config)) return;
    @import("test_note.zig").notePartSkipped("a load on the ledger: its forwards run under no pool cap");
    return error.SkipZigTest;
}

/// The refusals a load beside a load on the ledger may meet: what is left of the ceiling does not hold it.
const memory_refusals = [_]anyerror{ error.InsufficientMemory, error.ExpertCacheDoesNotFit, error.WarmChunkDoesNotFit, error.ModelBillPastBudget, error.PrefillChunkDoesNotFit, error.ContextDoesNotFit };

/// A served load of `dir` beside a load on the ledger, which holds its whole budget (D90): refused by name, before it
/// holds anything.
pub fn expectRefusedBesideLedger(comptime C: type, dir: []const u8) !void {
    var second: Load = undefined;
    second.open(C, dir, true) catch |e| {
        errdefer std.debug.print("cooler contract: a load of {s} beside one on the ledger failed {s}, not a refusal by name\n", .{ std.fs.path.basename(dir), @errorName(e) });
        return testing.expect(for (memory_refusals) |r| {
            if (e == r) break true;
        } else false);
    };
    _ = second.close();
    return error.TestUnexpectedResult;
}

/// A load on the ledger holds its whole budget (D90), so no load opens beside one: the held loads on it are unloaded
/// before any other opens, and opened again where a cell asks for them.
fn unloadHeldOnLedger() void {
    const rest = @import("test_rest.zig");
    var i: usize = 0;
    while (i < held_count) {
        const h = held_loads[i];
        if (h.ld.config.static_ledger == null) {
            i += 1;
            continue;
        }
        _ = h.ld.close();
        std.heap.smp_allocator.destroy(h.ld);
        std.heap.smp_allocator.free(h.dir);
        std.mem.copyForwards(Held, held_loads[i .. held_count - 1], held_loads[i + 1 .. held_count]);
        held_count -= 1;
        rest.held.holds -= 1;
    }
}

fn configHash(config: *const model_mod.ModelConfig) u64 {
    return std.hash.Wyhash.hash(0, std.mem.asBytes(config));
}

/// Each held model put between requests as serve leaves it (its cache reset, its request ended), then read: what a
/// test left of them is written to `w`.
pub fn heldRest(w: *std.Io.Writer) bool {
    var left = false;
    const open = mlx.engine.openDispatches();
    if (open != 0) {
        w.print(" {d} dispatches open beside the held loads;", .{open}) catch {};
        left = true;
    }
    for (held_loads[0..held_count]) |*h| {
        if (configHash(&h.ld.config) != h.config_hash) {
            w.print(" the held load of {s} with its config changed;", .{std.fs.path.basename(h.dir)}) catch {};
            left = true;
        }
        const turn = arch_handle.turnFor(h.ld.h, &h.ld.config);
        defer turn.end();
        // A cache keeps the capacity its longest request grew it to: the load's own, made again as serve made it.
        h.ld.xfm.resetCache() catch {
            w.writeAll(" a held load's cache not reset;") catch {};
            left = true;
        };
        h.ld.xfm.cache.reinit(h.ld.config.num_hidden_layers, @import("bills.zig").kvCacheForLaunch(&h.ld.config, null).config) catch {
            w.writeAll(" a held load's cache not made again;") catch {};
            left = true;
        };
        arch_sched.endRequest(&h.ld.config);
    }
    // What is in use is not read: a model keeps caches its longest request grew, which no cell made.
    const now = Sheet.now();
    if (now.mapped != held_sheet.mapped or now.regions != held_sheet.regions) {
        w.print(" beside the held loads: {d} B mapped ({d} held), {d} owned regions ({d});", .{ now.mapped, held_sheet.mapped, now.regions, held_sheet.regions }) catch {};
        return true;
    }
    return left;
}

/// The held loads unloaded, newest first; the pool must then hold nothing in use.
fn heldClose(w: *std.Io.Writer) bool {
    const rest = @import("test_rest.zig");
    while (held_count > 0) {
        held_count -= 1;
        const h = held_loads[held_count];
        _ = h.ld.close();
        std.heap.smp_allocator.destroy(h.ld);
        std.heap.smp_allocator.free(h.dir);
    }
    rest.held = .{};
    const now = Sheet.now();
    if (now.in_use == 0 and now.regions == 0 and now.open == 0) return false;
    w.print(" {d} B in use, {d} owned regions, {d} dispatches open after the held loads' unload;", .{ now.in_use, now.regions, now.open }) catch {};
    return true;
}

/// The engine's pool and mappings, the pool's free buffers taken back first, and the owned regions weights live in; and
/// the dispatches a pass still held open when the sheet was read, which at rest are none.
pub const Sheet = struct {
    open: u32,
    in_use: u64,
    asked: u64,
    mapped: u64,
    regions: usize,
    owned: u64,

    pub fn now() Sheet {
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
pub fn firstAsks(a: std.mem.Allocator, seen: *std.AutoHashMapUnmanaged(u64, void), comptime C: type, ctx: anytype, comptime run: fn (@TypeOf(ctx)) anyerror!void) ![]mlx.FaultSites.SitePoint {
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

pub fn archSite(comptime C: type, site: u64) u64 {
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
pub fn selection() ?[]const u8 {
    return if (std.c.getenv("SUSHI_CONTRACT_FIXTURE")) |v| std.mem.span(v) else null;
}

/// Whether a run under `sel` takes the fixture named by `env`.
pub fn fixtureChosen(sel: ?[]const u8, env: []const u8) bool {
    const names = sel orelse return true;
    var it = std.mem.tokenizeScalar(u8, names, ',');
    while (it.next()) |n| if (std.mem.eql(u8, std.mem.trim(u8, n, " "), env)) return true;
    return false;
}

/// Whether an ordered pair of arches runs: both first fixtures set, and the selection names a fixture of either.
pub fn pairChosen(sel: ?[]const u8, first: []const [:0]const u8, second: []const [:0]const u8, first_set: bool, second_set: bool) bool {
    if (!first_set or !second_set) return false;
    return anyChosen(sel, first) or anyChosen(sel, second);
}

fn anyChosen(sel: ?[]const u8, envs: []const [:0]const u8) bool {
    for (envs) |env| if (fixtureChosen(sel, env)) return true;
    return false;
}

/// Whether a cell skips before any load: with no selection, the gate's rule that every fixture is set, so a cell never
/// pays for the fixtures present only to report a skip. A selection skips per named fixture instead (`Tally`).
fn skipsUpFront(sel: ?[]const u8, set: []const bool) bool {
    if (sel != null) return false;
    for (set) |is_set| if (!is_set) return true;
    return false;
}

/// Every contract fixture env that `which` takes, and whether it is set.
fn fixturesSet(comptime which: Which, buf: *[16]bool) []const bool {
    var n: usize = 0;
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime @hasDecl(registry.Mod(t), "cooler_contract")) {
            for (registry.Mod(t).cooler_contract.fixtures) |env| {
                if (!takes(which, env)) continue;
                buf[n] = std.c.getenv(env) != null;
                n += 1;
            }
        }
    }
    return buf[0..n];
}

/// `body` on each chosen fixture of each arch that declares `cooler_contract` (`Tally` says what a skip or a missing one
/// does). For a property of the fixture's format or geometry; one that holds for any fixture of an arch is `eachArch`.
pub fn eachFixture(comptime body: fn (comptime C: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return eachOf(.every, body);
}

/// Whether the arch whose contract is `C` has an MLX arm (its user's opt-out): one that runs only on Sushi Cooler says
/// so with `mlx_arm = false`.
pub fn hasMlxArm(comptime C: type) bool {
    return !@hasDecl(C, "mlx_arm") or C.mlx_arm;
}

/// `body` on one fixture per arch: its first chosen and set fixture (the STQ tiny when set). For a property of the arch
/// and the engine, not of a pack's format.
pub fn eachArch(comptime body: fn (comptime C: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return eachOf(.first, body);
}

/// `body` on one fixture per arch and per expert residency, the first chosen of each: resident, and streamed (the BF16
/// checkpoint, MXFP4 experts). For a pool, memory or cleanup property, which a streamed plan changes and a pack's
/// geometry or tile order does not.
pub fn eachResidency(comptime body: fn (comptime C: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return eachOf(.per_residency, body);
}

/// `body` on one fixture per arch and per expert layout, the first chosen of each: its residency and its tile order
/// (the pair-major pack). For a property of the loader, its cleanup or the calls a request makes, which a pack's
/// geometry (the cut beside the tiny) does not change.
pub fn eachLayout(comptime body: fn (comptime C: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return eachOf(.per_layout, body);
}

/// `body` on each chosen fixture but the Qwen cut, and `onTheCut` on the cut alone: the two halves of a late cell, each a
/// run step of its own, so the cut's load runs beside the tinies' instead of after them.
pub fn eachTinyFixture(comptime body: fn (comptime C: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return eachOf(.tinies, body);
}

pub fn onTheCut(comptime body: fn (comptime C: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return eachOf(.cut, body);
}

const Which = enum { every, first, per_residency, per_layout, tinies, cut };

/// The fixture env of the Qwen cut (`QWEN4_TEST_MODEL`).
const cut_env = "QWEN4_TEST_MODEL";

pub fn takes(comptime which: Which, env: []const u8) bool {
    return switch (which) {
        .tinies => !std.mem.eql(u8, env, cut_env),
        .cut => std.mem.eql(u8, env, cut_env),
        .every, .first, .per_residency, .per_layout => true,
    };
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

fn eachOf(comptime which: Which, comptime body: fn (comptime C: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    try mlx.requireGpu();
    const sel = selection();
    var set_buf: [16]bool = undefined;
    if (skipsUpFront(sel, fixturesSet(which, &set_buf))) return error.SkipZigTest;
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
                if (!takes(which, env) or !fixtureChosen(sel, env)) continue;
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

test "cooler contract runner: a late cell's tinies half takes every fixture but the cut, and its cut half the cut alone" {
    for ([_][]const u8{ "QWEN4_STQ_TINY", "QWEN4_BF16_TINY", "DSV41_STQ_TINY", "DSV41_STQ_TINY_PM", "DSV41_TINY" }) |env| {
        try testing.expect(takes(.tinies, env));
        try testing.expect(!takes(.cut, env));
    }
    try testing.expect(!takes(.tinies, "QWEN4_TEST_MODEL"));
    try testing.expect(takes(.cut, "QWEN4_TEST_MODEL"));
    try testing.expect(takes(.every, "QWEN4_TEST_MODEL"));
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

test "cooler contract runner: a pair runs when the selection names either arch's fixture, and only where both fixtures are set" {
    const dsv41 = [_][:0]const u8{ "DSV41_STQ_TINY", "DSV41_STQ_TINY_PM", "DSV41_TINY" };
    const qwen4 = [_][:0]const u8{ "QWEN4_STQ_TINY", "QWEN4_TEST_MODEL", "QWEN4_BF16_TINY" };
    const hy4 = [_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY" };
    try testing.expect(pairChosen(null, &dsv41, &hy4, true, true));
    try testing.expect(!pairChosen(null, &dsv41, &hy4, true, false));
    try testing.expect(!pairChosen(null, &dsv41, &hy4, false, true));
    try testing.expect(pairChosen("HY4_STQ_TINY", &dsv41, &hy4, true, true));
    try testing.expect(pairChosen("HY4_STQ_TINY", &hy4, &dsv41, true, true));
    try testing.expect(!pairChosen("HY4_STQ_TINY", &dsv41, &qwen4, true, true));
    try testing.expect(pairChosen("DSV41_STQ_TINY_PM", &dsv41, &qwen4, true, true));
    try testing.expect(!pairChosen("HY4_STQ_TINY", &dsv41, &hy4, true, false));
}

test "cooler contract: test_skip_warm skips the warm on a light load and never reaches a served load; a light load on the ledger opens, plans and seals it" {
    try testing.expectEqual(false, (model_mod.ModelConfig{}).test_skip_warm);
    try testing.expectEqual(false, (@import("server_state.zig").ServerConfig{}).test_skip_warm);
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const cooler_warm = @import("cooler_warm.zig");
            const prompt = ids[0..@min(ids.len, 16)];
            const served_before = cooler_warm.runs_for_test;
            const arch_before = cooler_warm.arch_warm_runs_for_test;
            var want: std.ArrayList(u32) = .empty;
            defer want.deinit(testing.allocator);
            var served_ledger = false;
            var served_gate: @FieldType(mlx.engine.BudgetSlot, "gate") = .open;
            {
                var served: Load = undefined;
                try served.open(C, dir, true);
                defer _ = served.close();
                try testing.expectEqual(false, served.config.test_skip_warm);
                try testing.expectEqual(served_before + 1, cooler_warm.runs_for_test);
                if (@import("arch/config.zig").isOf(&served.config, "dsv41"))
                    try testing.expectEqual(arch_before + 1, cooler_warm.arch_warm_runs_for_test);
                served_ledger = served.config.static_ledger != null;
                served_gate = served.xfm.cooler_budget.gate;
                want = try greedy(&served, prompt, 4);
            }
            const light_before = cooler_warm.runs_for_test;
            const light_arch_before = cooler_warm.arch_warm_runs_for_test;
            var light: Load = undefined;
            try light.openLight(C, dir, true);
            defer _ = light.close();
            try testing.expectEqual(light_before, cooler_warm.runs_for_test);
            try testing.expectEqual(light_arch_before, cooler_warm.arch_warm_runs_for_test);
            try testing.expectEqual(served_ledger, light.config.static_ledger != null);
            if (light.config.static_ledger) |l| {
                try testing.expect(l.context.load(.acquire) > 0);
                try testing.expect(light.config.static_prefill_chunk > 0);
            }
            try testing.expectEqual(served_gate, light.xfm.cooler_budget.gate);
            var got = try greedy(&light, prompt, 4);
            defer got.deinit(testing.allocator);
            try testing.expectEqualSlices(u32, want.items, got.items);
        }
    }.run);
}

/// `e`, after naming the fixture it came from.
pub fn named(env: []const u8, dir: []const u8, e: anyerror) anyerror {
    var msg: [512]u8 = undefined;
    std.debug.print("cooler contract: fixture {s} ({s}): {s}: {s}\n", .{ env, dir, @errorName(e), mlx.takeError(&msg) orelse "no MLX or engine error latched" });
    return e;
}

/// `n` greedy tokens over `ids` in serial steps, one request in the model's turn, ended as serve ends one.
pub fn greedy(ld: *Load, ids: []const u32, n: u32) !std.ArrayList(u32) {
    return serial(ld, ids, n, .{ .temperature = 0.0 });
}

/// `greedy` under `sampling`.
pub fn serial(ld: *Load, ids: []const u32, n: u32, sampling: SamplingParams) !std.ArrayList(u32) {
    return serialWith(ld, ids, n, sampling, .{});
}

/// `serial` with the generator's options, as a served request's own.
pub fn serialWith(ld: *Load, ids: []const u32, n: u32, sampling: SamplingParams, options: Generator.InitOptions) !std.ArrayList(u32) {
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

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// A contract load served by the scheduler's own slot path: a registry entry holding the load's handle and config, the
/// SSD-only tier a served load arms for it (`attachPrefixCache`, its cache dir under a tmp dir), and a test scheduler.
pub const Served = struct {
    pub const scheduler = @import("scheduler.zig");
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

    pub fn init(self: *Served, ld: *Load, dir: []const u8) !void {
        return self.initTiers(ld, dir, 0);
    }

    /// `ram_bytes`: the RAM tier's budget, 0 for the SSD tier alone.
    pub fn initTiers(self: *Served, ld: *Load, dir: []const u8, ram_bytes: u64) !void {
        try self.initWith(ld, dir, ram_bytes, 1 << 30);
        if (self.entry.prefix_cache == null or self.entry.prefix_cache.?.disk == null) {
            self.deinit();
            return error.SkipZigTest;
        }
        self.entry.prefix_cache.?.disk.?.armTestSpace(256 << 30, 1 << 40);
    }

    /// A RAM tier of `ram_bytes` and no SSD tier, as `--prefix-cache-disk off` leaves a load; the entry has no cache at
    /// all where the load keeps none.
    pub fn initRamOnly(self: *Served, ld: *Load, dir: []const u8, ram_bytes: u64) !void {
        try self.initWith(ld, dir, ram_bytes, 0);
    }

    fn initWith(self: *Served, ld: *Load, dir: []const u8, ram_bytes: u64, disk_bytes: u64) !void {
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
        model_load.attachPrefixCache(a, io, self.entry, .{
            .capacity = 4,
            .ram_enabled = ram_bytes > 0,
            .mem_bytes = ram_bytes,
            .mem_resolver = null,
            .disk_resolver = null,
            .ssm_checkpoint_stride = server_state.ssm_checkpoint_stride,
            .ssm_checkpoint_max = server_state.ssm_checkpoint_max,
        }, disk_bytes, .load);
        self.sch = scheduler.submitTestScheduler(a, 2);
        self.sch.kv_quant = null;
        self.sch.metrics = null;
        self.sch.requests_prefilling = .init(0);
        self.sch.inflight_prefill_tokens = .init(0);
        self.sch.inflight_prefill_expected = .init(0);
        self.sch.resident_live_kv_bytes = .init(0);
        self.sch.live_session_count = 0;
        self.sch.registry = self.reg;
        self.sch.digest_mu = .init;
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

    pub fn deinit(self: *Served) void {
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
    pub fn request(self: *Served, prompt: []const u32, n: u32, restored: *u32) !std.ArrayList(u32) {
        return self.requestWith(prompt, n, .{ .temperature = 0.0 }, false, restored);
    }

    /// `restored`: the prompt tokens an SSD restore gave; `reused` holds what any tier or the live state gave.
    /// `mtp`: with the model's MTP head, its rounds solo, as one speculative request decodes.
    pub fn requestWith(self: *Served, prompt: []const u32, n: u32, sampling: @import("sampler.zig").SamplingParams, mtp: bool, restored: *u32) !std.ArrayList(u32) {
        const a = testing.allocator;
        const sch = &self.sch;
        const head = if (mtp) @import("generate.zig").MtpHeadRef.fromModel(arch_handle.trunk(self.entry.handle.?), true, null) else null;
        const slot = try sch.submit(.{ .model = self.entry, .prompt_ids = prompt, .sampling = sampling, .eos_token_ids = &.{}, .max_tokens = n, .enable_mtp = mtp, .mtp = head });
        defer {
            sch.complete(slot);
            scheduler.drainSlots(sch, sch.cleanup_queue.items);
            sch.cleanup_queue.clearRetainingCapacity();
        }
        _ = sch.pending.orderedRemove(0);
        try scheduler.runPrefill(sch, slot);
        // The prefill's end counts the hit and clears the slot's tier; the cache still names it.
        const hc = &self.entry.prefix_cache.?;
        restored.* = if (slot.cached_tokens > 0 and hc.lastRestoredTier() == .ssd) slot.cached_tokens else 0;
        self.reused = slot.cached_tokens;
        if (!scheduler.postPrefillTerminal(sch, slot)) {
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

    /// One request as the inference loop runs it, a failure settled by `failPrefill` or `failTick`, then completed and
    /// taken by the loop's cleanup drain (`scheduler.drainSlots`): the dispatches still open once the slot is gone.
    pub fn requestToCleanup(self: *Served, prompt: []const u32, n: u32, mtp: bool) !u32 {
        const sch = &self.sch;
        const head = if (mtp) @import("generate.zig").MtpHeadRef.fromModel(arch_handle.trunk(self.entry.handle.?), true, null) else null;
        const slot = try sch.submit(.{ .model = self.entry, .prompt_ids = prompt, .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = n, .enable_mtp = mtp, .mtp = head });
        _ = sch.pending.orderedRemove(0);
        scheduler.runPrefill(sch, slot) catch |e| scheduler.failPrefill(slot, e);
        if (!scheduler.postPrefillTerminal(sch, slot)) {
            try sch.decoding.append(testing.allocator, slot);
            var one = [_]*scheduler.Slot{slot};
            while (slot.joinsTick()) scheduler.runDecodeTick(sch, &one) catch |e| scheduler.failTick(&one, e, "[scheduler]");
            scheduler.cullDecoding(sch);
        }
        sch.complete(slot);
        scheduler.drainSlots(sch, sch.cleanup_queue.items);
        sch.cleanup_queue.clearRetainingCapacity();
        return mlx.engine.openDispatches();
    }

    /// Requests through the scheduler, each prefilled in turn and then decoded together, their rows joined in its ticks
    /// as concurrent requests are, with the model's MTP head when `mtp`. `reused`: what the last took from a tier.
    pub fn concurrent(self: *Served, prompts: []const []const u32, n: u32, sampling: SamplingParams, mtp: bool) !void {
        return self.concurrentInto(prompts, n, sampling, mtp, null);
    }

    /// `concurrent`, with each request's tokens appended to `out[i]` when given.
    pub fn concurrentInto(self: *Served, prompts: []const []const u32, n: u32, sampling: SamplingParams, mtp: bool, out: ?[]std.ArrayList(u32)) !void {
        const a = testing.allocator;
        const sch = &self.sch;
        var slots: [4]*scheduler.Slot = undefined;
        var made: usize = 0;
        defer {
            for (slots[0..made]) |slot| sch.complete(slot);
            scheduler.drainSlots(sch, sch.cleanup_queue.items);
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
            if (!scheduler.postPrefillTerminal(sch, slot)) try sch.decoding.append(a, slot);
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
        if (out) |o| for (slots[0..made], o) |slot, *into| try into.appendSlice(a, slot.legacy_gen.?.generated_ids.items);
        const hc = &self.entry.prefix_cache.?;
        hc.flushPendingDisk(arch_handle.trunk(self.entry.handle.?).s);
        hc.disk.?.drainWriter();
    }

    /// A request asked for `n` tokens that its client cancels after `ticks` decode ticks, ended as a cancel ends it (the
    /// slot culled, completed and deinitialised); the tokens it said before.
    pub fn requestCancelled(self: *Served, prompt: []const u32, n: u32, ticks: usize) !std.ArrayList(u32) {
        const a = testing.allocator;
        const sch = &self.sch;
        const slot = try sch.submit(.{ .model = self.entry, .prompt_ids = prompt, .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = n });
        defer {
            sch.complete(slot);
            scheduler.drainSlots(sch, sch.cleanup_queue.items);
            sch.cleanup_queue.clearRetainingCapacity();
        }
        _ = sch.pending.orderedRemove(0);
        try scheduler.runPrefill(sch, slot);
        if (!scheduler.postPrefillTerminal(sch, slot)) {
            try sch.decoding.append(a, slot);
            var one = [_]*scheduler.Slot{slot};
            for (0..ticks) |_| if (slot.joinsTick()) try scheduler.runDecodeTick(sch, &one);
            slot.cancelled.store(true, .release);
            scheduler.cullDecoding(sch);
        }
        var got: std.ArrayList(u32) = .empty;
        errdefer got.deinit(a);
        try got.appendSlice(a, slot.legacy_gen.?.generated_ids.items);
        return got;
    }
};

/// The Cooler contract of the arch registered as `name`, or an empty struct where the build left it out: a shared test
/// of one arch's load checks `@hasDecl` and skips, so the drop of that arch still compiles it.
pub fn ContractOf(comptime name: []const u8) type {
    if (comptime !@hasField(registry.Tag, name)) return struct {};
    return registry.Mod(@field(registry.Tag, name)).cooler_contract;
}

/// The Qwen fixtures a served-path cell runs on: the tiny STQ and, when set, the cut.
pub fn eachQwenServed(comptime body: fn (comptime Q: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return qwenServedOn(&.{ "QWEN4_STQ_TINY", "QWEN4_TEST_MODEL" }, body);
}

/// The tiny STQ alone: for a sweep, whose sites the cut's load reaches as well.
pub fn qwenTinyServed(comptime body: fn (comptime Q: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return qwenServedOn(&.{"QWEN4_STQ_TINY"}, body);
}

fn qwenServedOn(envs: []const [:0]const u8, comptime body: fn (comptime Q: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return servedOn("qwen4", envs, body);
}

pub fn servedOn(comptime arch: []const u8, envs: []const [:0]const u8, comptime body: fn (comptime Q: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    defer mlx.engine.fault.disarm();
    const Q = ContractOf(arch);
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

/// One served request through the scheduler as serve runs it (prefill, then its ticks), drafting with `head` when set:
/// its tokens, and how many drafts its rounds verified (an MTP head's or DSpark's).
pub fn servedDrafts(sv: *Served, prompt: []const u32, n: u32, sampling: @import("sampler.zig").SamplingParams, head: ?@import("generate.zig").MtpHeadRef, drafted: *u64) !std.ArrayList(u32) {
    const a = testing.allocator;
    const sch = &sv.sch;
    const scheduler = Served.scheduler;
    const slot = try sch.submit(.{ .model = sv.entry, .prompt_ids = prompt, .sampling = sampling, .eos_token_ids = &.{}, .max_tokens = n, .enable_mtp = head != null, .mtp = head });
    defer {
        sch.complete(slot);
        scheduler.drainSlots(sch, sch.cleanup_queue.items);
        sch.cleanup_queue.clearRetainingCapacity();
    }
    _ = sch.pending.orderedRemove(0);
    scheduler.runPrefill(sch, slot) catch |e| {
        scheduler.failPrefill(slot, e);
        return e;
    };
    if (!scheduler.postPrefillTerminal(sch, slot)) {
        try sch.decoding.append(a, slot);
        var one = [_]*scheduler.Slot{slot};
        while (slot.joinsTick()) try scheduler.runDecodeTick(sch, &one);
        scheduler.cullDecoding(sch);
    }
    if (slot.error_code) |e| {
        if (!sv.quiet_failures) std.debug.print("cooler contract: the served request failed ({s})\n", .{e});
        return error.RequestFailed;
    }
    const g = slot.legacy_gen.?;
    drafted.* = g.mtp_drafted_tokens + g.drafter_stats.drafted;
    var got: std.ArrayList(u32) = .empty;
    errdefer got.deinit(a);
    try got.appendSlice(a, g.generated_ids.items);
    // The turn's SSD write lands before the next request looks it up, as a served pause between turns gives it.
    if (sv.entry.prefix_cache) |*hc| {
        hc.flushPendingDisk(@import("arch/handle.zig").trunk(sv.entry.handle.?).s);
        if (hc.disk) |*d| d.drainWriter();
    }
    return got;
}

/// Two served requests decoded as rows of one tick through the scheduler, as serve runs them.
pub fn servedRows(sv: *Served, prompts: [2][]const u32, n: u32, mtp: bool, out: *[2]std.ArrayList(u32)) !void {
    const scheduler = @import("scheduler.zig");
    const a = testing.allocator;
    const sch = &sv.sch;
    var slots: [2]*scheduler.Slot = undefined;
    var made: usize = 0;
    defer {
        for (slots[0..made]) |s| sch.complete(s);
        scheduler.drainSlots(sch, sch.cleanup_queue.items);
        sch.cleanup_queue.clearRetainingCapacity();
    }
    for (0..2) |i| {
        const head = if (mtp) @import("generate.zig").MtpHeadRef.fromModel(arch_handle.trunk(sv.entry.handle.?), true, null) else null;
        slots[i] = try sch.submit(.{ .model = sv.entry, .prompt_ids = prompts[i], .sampling = .{ .temperature = 0.0 }, .eos_token_ids = &.{}, .max_tokens = n, .enable_mtp = mtp, .mtp = head });
        made += 1;
    }
    sch.pending.clearRetainingCapacity();
    var failed: ?anyerror = null;
    for (slots) |s| {
        scheduler.runPrefill(sch, s) catch |e| {
            failed = e;
            s.markError(@errorName(e));
        };
        if (!scheduler.postPrefillTerminal(sch, s)) try sch.decoding.append(a, s);
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
    for (slots) |s| if (s.error_code) |_| return error.RequestFailed;
    var filled: usize = 0;
    errdefer for (out[0..filled]) |*o| o.deinit(a);
    for (slots, 0..) |s, i| {
        out[i] = .empty;
        try out[i].appendSlice(a, s.legacy_gen.?.generated_ids.items);
        filled += 1;
    }
    const hc = &sv.entry.prefix_cache.?;
    hc.flushPendingDisk(arch_handle.trunk(sv.entry.handle.?).s);
    hc.disk.?.drainWriter();
}

/// Requests through one scheduler as the inference loop runs them, with the loop's own failure and cleanup steps: all
/// submitted at once and prefilled in turn (a second one after a tick of the first, as `interleaveDecodeTick` runs),
/// then joined in each decode tick; a client that leaves is completed and drained at its tick.
pub const MixedTick = struct {
    pub const scheduler = Served.scheduler;
    const max = 4;

    const Spec = struct {
        prompt: []const u32,
        n: u32,
        sampling: SamplingParams = .{ .temperature = 0.0 },
        mtp: bool = false,
        /// Decode ticks after its admission at which its client leaves; null: it runs to its end.
        cancel_at: ?u32 = null,

        fn deterministic(s: Spec) bool {
            return s.sampling.temperature == 0.0 or s.sampling.seed != null;
        }
    };

    /// A request's tokens, or the name it failed by.
    const Out = struct {
        tokens: std.ArrayList(u32) = .empty,
        err: [48]u8 = undefined,
        err_len: usize = 0,

        fn failed(o: *const Out) bool {
            return o.err_len > 0;
        }
    };

    pub fn tick(sch: *scheduler.Scheduler) void {
        var act: [max]*scheduler.Slot = undefined;
        var k: usize = 0;
        for (sch.decoding.items) |s| if (s.joinsTick()) {
            act[k] = s;
            k += 1;
        };
        if (k > 0) scheduler.runDecodeTick(sch, act[0..k]) catch |e| scheduler.failTick(act[0..k], e, "[scheduler]");
        scheduler.cullDecoding(sch);
    }

    pub fn drain(sch: *scheduler.Scheduler) void {
        scheduler.drainSlots(sch, sch.cleanup_queue.items);
        sch.cleanup_queue.clearRetainingCapacity();
    }

    fn run(sv: *Served, specs: []const Spec, out: []Out) !void {
        const a = testing.allocator;
        const sch = &sv.sch;
        // `submit` waits for room in the queue; every request here is in flight at once.
        sch.queue_cap = max;
        var slots: [max]*scheduler.Slot = undefined;
        var gone: [max]bool = @splat(false);
        var made: usize = 0;
        for (out[0..specs.len]) |*o| o.* = .{};
        errdefer for (out[0..specs.len]) |*o| o.tokens.deinit(a);
        defer {
            for (slots[0..made], gone[0..made]) |s, g| if (!g) sch.complete(s);
            drain(sch);
        }
        for (specs, 0..) |sp, i| {
            if (i > 0) tick(sch);
            const head = if (sp.mtp) @import("generate.zig").MtpHeadRef.fromModel(arch_handle.trunk(sv.entry.handle.?), true, null) else null;
            slots[i] = try sch.submit(.{ .model = sv.entry, .prompt_ids = sp.prompt, .sampling = sp.sampling, .eos_token_ids = &.{}, .max_tokens = sp.n, .enable_mtp = sp.mtp, .mtp = head });
            made += 1;
            for (sch.pending.items, 0..) |p, k| if (p == slots[i]) {
                _ = sch.pending.orderedRemove(k);
                break;
            };
            scheduler.runPrefill(sch, slots[i]) catch |e| scheduler.failPrefill(slots[i], e);
            if (!scheduler.postPrefillTerminal(sch, slots[i])) try sch.decoding.append(a, slots[i]);
        }
        var t: u32 = 0;
        while (true) : (t += 1) {
            for (specs, 0..) |sp, i| {
                if (sp.cancel_at != t or gone[i]) continue;
                if (slots[i].legacy_gen) |*g| try out[i].tokens.appendSlice(a, g.generated_ids.items);
                sch.complete(slots[i]);
                gone[i] = true;
                drain(sch);
            }
            var live = false;
            for (sch.decoding.items) |s| live = live or s.joinsTick();
            if (!live) break;
            tick(sch);
        }
        scheduler.cullDecoding(sch);
        for (slots[0..made], gone[0..made], out[0..made]) |s, g, *o| {
            if (g) continue;
            if (s.error_code) |e| {
                o.err_len = @min(e.len, o.err.len);
                @memcpy(o.err[0..o.err_len], e[0..o.err_len]);
            } else try o.tokens.appendSlice(a, s.legacy_gen.?.generated_ids.items);
        }
        const hc = &sv.entry.prefix_cache.?;
        hc.flushPendingDisk(arch_handle.trunk(sv.entry.handle.?).s);
        hc.disk.?.drainWriter();
    }

    /// `specs` refused at each pool-buffer site of their joined run that `seen` lacks, at its first ask: at most one
    /// request fails, and by its settled name; every other one says its solo tokens, the next joined run says them all
    /// and the pool is as it was.
    pub fn faultSweep(comptime C: type, sv: *Served, specs: []const Spec, seen: *std.AutoHashMapUnmanaged(u64, void)) !void {
        const a = testing.allocator;
        var solo: [max]Out = undefined;
        var made: usize = 0;
        defer for (solo[0..made]) |*s| s.tokens.deinit(a);
        for (specs) |sp| {
            var one = [_]Spec{sp};
            one[0].cancel_at = null;
            try run(sv, &one, solo[made..][0..1]);
            made += 1;
            try testing.expect(!solo[made - 1].failed());
        }
        const Joined = struct {
            sv: *Served,
            specs: []const Spec,
            fn go(q: @This()) anyerror!void {
                var o: [max]Out = undefined;
                try run(q.sv, q.specs, &o);
                for (o[0..q.specs.len]) |*x| x.tokens.deinit(testing.allocator);
            }
        };
        const points = try firstAsks(a, seen, C, Joined{ .sv = sv, .specs = specs }, Joined.go);
        defer a.free(points);
        try (Joined{ .sv = sv, .specs = specs }).go();
        const base = Sheet.now();
        for (points, 0..) |pt, at| {
            errdefer std.debug.print("cooler contract: mixed tick refused at pool point {d} of {d}\n", .{ at, points.len });
            var o: [max]Out = undefined;
            mlx.engine.fault.armAt(pt);
            const r = run(sv, specs, &o);
            mlx.engine.fault.disarm();
            try r;
            defer for (o[0..specs.len]) |*x| x.tokens.deinit(a);
            try testing.expect(!mlx.errorPending());
            var failed: usize = 0;
            for (specs, o[0..specs.len], solo[0..specs.len]) |sp, *x, *want| {
                if (x.failed()) {
                    failed += 1;
                    try testing.expectEqualStrings("TransientCapExceeded", x.err[0..x.err_len]);
                } else if (sp.deterministic() and sp.cancel_at == null) try testing.expectEqualSlices(u32, want.tokens.items, x.tokens.items);
            }
            try testing.expect(failed <= 1);
            var again: [max]Out = undefined;
            try run(sv, specs, &again);
            defer for (again[0..specs.len]) |*x| x.tokens.deinit(a);
            for (specs, again[0..specs.len], solo[0..specs.len]) |sp, *x, *want| {
                try testing.expect(!x.failed());
                if (sp.deterministic() and sp.cancel_at == null) try testing.expectEqualSlices(u32, want.tokens.items, x.tokens.items);
            }
            const now = Sheet.now();
            try testing.expectEqual(base.in_use, now.in_use);
            try testing.expectEqual(@as(u32, 0), now.open);
        }
    }
};

/// Runs `body` with `$HOME/.sushi/model-settings.json` holding `row` for `dir`, as the web UI's model settings save it, and
/// each Load stamping it; the parse reads the file's `yarn_factor` itself.
pub fn withUiSettings(dir: []const u8, row: []const u8, comptime body: fn () anyerror!void) !void {
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

/// The engine pool's bytes in use at its peak over `body`, past those in use before it.
pub fn inUsePeak(ctx: anytype, comptime body: fn (@TypeOf(ctx)) anyerror!void) !u64 {
    mlx.engine.close();
    mlx.engine.trim();
    const in0 = mlx.engine.poolTotal() - mlx.engine.poolFree();
    mlx.engine.resetPoolPeak(.process);
    try body(ctx);
    mlx.engine.close();
    return mlx.engine.poolPeak(.process) -| in0;
}
