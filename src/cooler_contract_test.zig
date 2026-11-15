//! The Sushi Cooler contract: what every load on the engine of our own owes, whatever its arch. An arch on Sushi
//! Cooler declares `cooler_contract` (its tiny fixtures by env var and the reads only it can make: `prepare`,
//! `prompt`, `weightsOutsideRegions`, `headLoaded`, and optionally `specTokens`); each test runs on every such fixture
//! present and is reported skipped when any is missing. Loads, requests and unloads take the turns and hooks serve
//! takes.
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
        self.h = arch_handle.adopt(self.xfm);
    }

    fn open(self: *Load, comptime C: type, dir: []const u8, cooler: bool) !void {
        try self.prepare(C, dir, cooler);
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

/// `body` on each fixture of each arch that declares `cooler_contract`; a skip when none is present.
fn eachFixture(comptime body: fn (comptime C: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    var ran = false;
    var missing = false;
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime @hasDecl(registry.Mod(t), "cooler_contract")) {
            const C = registry.Mod(t).cooler_contract;
            for (C.fixtures) |env| {
                const dir = std.mem.span(std.c.getenv(env) orelse {
                    missing = true;
                    continue;
                });
                ran = true;
                errdefer std.debug.print("cooler contract: fixture {s} ({s})\n", .{ env, dir });
                const ids = try C.prompt(testing.allocator, dir);
                defer testing.allocator.free(ids);
                try body(C, dir, ids);
            }
        }
    }
    if (!ran or missing) return error.SkipZigTest;
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
                try on_mlx.open(C, dir, false);
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
                } else |_| {
                    refused += 1;
                    _ = mlx.takeError(&msg);
                }
                try testing.expect(!mlx.engine.holding());
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
