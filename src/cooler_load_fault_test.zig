//! The Sushi Cooler contract's load fault sweeps: a load or warm failed or refused at each of its sites leaves the
//! engine and the pool as they were. A step of its own beside `cooler_load_test.zig`'s; the runner and the shared
//! helpers are in `cooler_contract_test.zig`.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const arch_handle = @import("arch/handle.zig");
const io = std.Io.Threaded.global_single_threaded.io();
const contract = @import("cooler_contract_test.zig");
const Load = contract.Load;
const Sheet = contract.Sheet;
const eachArch = contract.eachArch;
const eachFixture = contract.eachFixture;
const eachLayout = contract.eachLayout;
const firstAsks = contract.firstAsks;
const greedy = contract.greedy;

test "cooler contract: a Cooler load refused at a pool buffer where its cleanup changes frees what it made, and the next load serves the same tokens" {
    try eachLayout(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            defer mlx.engine.fault.disarm();
            const prompt = ids[0..@min(ids.len, 30)];
            var want: std.ArrayList(u32) = .empty;
            defer want.deinit(testing.allocator);
            {
                var first: Load = undefined;
                try first.openLight(C, dir, true);
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
            try after.openLight(C, dir, true);
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
            try ld.openLight(C, dir, true);
            defer _ = ld.close();
            if (ld.config.image_token_id == 0) return error.SkipZigTest;
            // The splice is a step of the warm, which asks the pool by design (D64); the sealed load's gate opens as a warm's does.
            @import("arch/cooler.zig").warmBegins(ld.xfm);
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
                const path = std.hash.Wyhash.hash(@backingInt(phase), std.mem.sliceAsBytes(points));
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

test "cooler contract: a Cooler load whose warm fails, torn down as a served load's failure is, leaves the pool holding no more than before the load" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            defer mlx.engine.fault.disarm();
            // The pool buffers a warm asks for, counted on a load that serves.
            var total: u64 = 0;
            {
                var fx: Load = undefined;
                try fx.prepare(C, dir, true);
                try fx.load(dir);
                const c0 = mlx.engine.fault.count();
                fx.serve() catch |e| {
                    fx.discard();
                    return e;
                };
                total = mlx.engine.fault.count() - c0;
                _ = fx.close();
            }
            mlx.engine.close();
            const before = mlx.engine.stats().pool_bytes;
            for ([_]u64{ total / 4, total / 2, total * 3 / 4, total - 2 }) |k| {
                var fx: Load = undefined;
                try fx.prepare(C, dir, true);
                try fx.load(dir);
                mlx.engine.fault.arm(k);
                const r = fx.serve();
                mlx.engine.fault.disarm();
                if (r) |_| {
                    _ = fx.close();
                    continue;
                } else |_| {}
                var msg: [300]u8 = undefined;
                _ = mlx.takeError(&msg);
                fx.discard();
                mlx.engine.close();
                const held = mlx.engine.stats().pool_bytes;
                errdefer std.debug.print("{s}: a warm failed at pool buffer {d} of {d} leaves the pool at {d} MB, {d} MB before the load\n", .{ std.fs.path.basename(dir), k, total, held >> 20, before >> 20 });
                try testing.expect(held <= before);
            }
        }
    }.run);
}

test "cooler contract: a second Cooler load beside a sealed one whose warm fails leaves the engine sealed, as the second load's success would" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            defer mlx.engine.fault.disarm();
            var first: Load = undefined;
            try first.open(C, dir, true);
            defer _ = first.close();
            try testing.expect(mlx.engine.sealed());
            // Beside a load on the ledger the second is refused by name (D90), and the first stays sealed.
            if (first.config.static_ledger != null) {
                try contract.expectRefusedBesideLedger(C, dir);
                return testing.expect(mlx.engine.sealed());
            }
            // The warm's pool asks, counted on a second load that serves: it ends sealed.
            var total: u64 = 0;
            {
                var fx: Load = undefined;
                try fx.prepare(C, dir, true);
                try fx.load(dir);
                try testing.expect(!mlx.engine.sealed());
                const c0 = mlx.engine.fault.count();
                fx.serve() catch |e| {
                    fx.discard();
                    return e;
                };
                total = mlx.engine.fault.count() - c0;
                _ = fx.close();
            }
            try testing.expect(mlx.engine.sealed());
            if (total < 2) return error.SkipZigTest;
            var fx: Load = undefined;
            try fx.prepare(C, dir, true);
            try fx.load(dir);
            mlx.engine.fault.arm(total / 2);
            const r = fx.serve();
            mlx.engine.fault.disarm();
            if (r) |_| {
                _ = fx.close();
                return error.SkipZigTest;
            } else |_| {}
            var msg: [300]u8 = undefined;
            _ = mlx.takeError(&msg);
            fx.discard();
            errdefer std.debug.print("{s}: the first load serves on with the engine unsealed after a second load's warm failed at pool buffer {d} of {d}\n", .{ std.fs.path.basename(dir), total / 2, total });
            try testing.expect(mlx.engine.sealed());
        }
    }.run);
}

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
        if (served) |_| {} else |_| fx.discard();
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

/// One Cooler load of `dir` with its `k`-th checked MLX call failed, on an allocator of its own: whether anything of it
/// stays (the allocator's, the engine's or a hold), and whether the fault fired.
/// A served load's two phases, each swept from its own root: `load` brings the trunk up, `serve` warms it.
const Phase = enum { load, serve };
