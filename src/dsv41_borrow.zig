//! DeepSeek-V4.1's prefill takes its working set from the expert cache. A forward that does not fit the free memory
//! first releases whole decoder-layer pools, latest layer first: under CED those layers run only a chunk's last
//! `window` rows, and a released layer's picks are read into the union workspace. The first decode forward after the
//! prefill reads the pools back. Every forward runs under a transient cap (`Config.forwardCap`), which is what admission
//! bills: a request's state, a decode forward's cap and the part of a prefill's the pools cannot cover.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const status = @import("status.zig");
const stream = @import("expert_stream.zig");
const cfg_mod = @import("dsv41_config.zig");
const ModelConfig = @import("model.zig").ModelConfig;
const Config = cfg_mod.Config;

/// The live headroom (the GPU ceiling less MLX's active bytes), installed by the server. Unset (tests, `sushi kld`),
/// nothing is ever released.
pub var headroom_fn: ?*const fn (*const ModelConfig) u64 = null;

/// The widest verify block: up to it a forward is a decode forward, whose cap admission always reserves.
pub const verify_rows = cfg_mod.verify_rows;

/// What a prefill-width forward must find free before it runs: the most it may allocate (`Config.forwardCap`).
pub fn prefillNeed(c: *const Config, rows: u64, seq: u64) u64 {
    return c.forwardCap(rows, seq);
}

/// PURE: the MLX buffer-pool cap of a forward of `rows`: the decode cap up to a verify block, the RAM-wide cap above
/// it. The transient cap bounds the pool with the forward's allocations, so a wide forward needs no cap of its own.
pub fn poolCap(rows: u64, decode_cap: u64, ram_cap: u64) u64 {
    const decode = @min(ram_cap, decode_cap);
    return if (rows <= verify_rows) decode else @max(decode, ram_cap);
}

/// The RAM-wide pool cap, installed by the server. Unset (tests, `sushi kld`), neither cap is ever moved.
pub var pool_ram_cap: u64 = 0;
/// The cap in force while it is above the decode cap; null at the decode cap the ledger set at load.
var pool_now: ?u64 = null;
/// Whether a transient cap is installed.
var transient_now: bool = false;
/// The process-wide cap the load replaced with the decode cap; null while none is lowered.
var ram_limit: ?u64 = null;

pub const PoolOps = struct {
    set_limit: *const fn (bytes: u64) void,
    clear: *const fn () void,
    /// Caps what MLX may allocate beyond what it holds now, its freed-buffer pool included; 0 lifts the cap.
    set_transient: *const fn (bytes: u64) void,
};

fn rowsCap(rows: u64, config: *const ModelConfig) u64 {
    return poolCap(rows, config.expertPoolBytes(), if (pool_ram_cap == 0) std.math.maxInt(u64) else pool_ram_cap);
}

/// What a phase may take: its own bytes and the decode pool the ledger parks beside every phase.
fn phaseLimit(bytes: u64, config: *const ModelConfig) u64 {
    return bytes +| config.expertPoolBytes();
}

fn moveCap(want: u64, config: *const ModelConfig, ops: PoolOps) bool {
    const decode = config.expertPoolBytes();
    const have = pool_now orelse decode;
    if (want == have) return false;
    pool_now = if (want > decode) want else null;
    ops.set_limit(want);
    // A lower cap trims lazily; the clear makes the parked pool what was billed.
    if (want < have) ops.clear();
    return true;
}

fn setTransient(bytes: u64, ops: PoolOps) void {
    transient_now = bytes > 0;
    ops.set_transient(bytes);
}

/// Moves the pool cap to the one `rows` takes and caps the forward's allocations at what it bills, on the inference
/// thread. True when the pool cap moved.
pub fn enterForward(rows: u64, seq: u64, config: *const ModelConfig, c: *const Config, ops: PoolOps) bool {
    if (pool_ram_cap == 0 or !config.isDsv41()) return false;
    setTransient(phaseLimit(c.forwardCap(rows, seq), config), ops);
    return moveCap(rowsCap(rows, config), config, ops);
}

/// An image encode of `bill` bytes (`server.visionEncodeBill`) starts: its allocations are capped at it. No bill, no cap.
pub fn enterVision(bill: u64, config: *const ModelConfig, ops: PoolOps) void {
    if (pool_ram_cap == 0 or bill == 0 or !config.isDsv41()) return;
    setTransient(phaseLimit(bill, config), ops);
}

/// The encode is over; the next forward installs its own cap.
pub fn leaveVision(config: *const ModelConfig, ops: PoolOps) void {
    if (pool_ram_cap == 0 or !config.isDsv41() or !transient_now) return;
    setTransient(0, ops);
}

/// A request is over: whatever forward ran last, the transient cap is lifted (a restore allocates the next request's
/// state before its first forward) and the pool cap drops back to the decode cap.
pub fn endRequest(config: *const ModelConfig, ops: PoolOps) bool {
    if (pool_ram_cap == 0 or !config.isDsv41()) return false;
    const was_on = transient_now;
    if (was_on) setTransient(0, ops);
    return moveCap(config.expertPoolBytes(), config, ops) or was_on;
}

/// The load lowered the process-wide cap from `prev` to the ledger's decode cap.
pub fn poolCapped(prev: u64) void {
    if (ram_limit == null) ram_limit = prev;
}

/// A DeepSeek model is unloaded: the next model gets the process-wide cap back, and the raised cap is forgotten.
pub fn unloadModel(config: *const ModelConfig, ops: PoolOps) bool {
    if (!config.isDsv41()) return false;
    pool_now = null;
    if (transient_now) setTransient(0, ops);
    const prev = ram_limit orelse return false;
    ram_limit = null;
    ops.set_limit(prev);
    return true;
}

/// The unload hook on the real MLX pool.
pub fn unloadModelMlx(config: *const ModelConfig) void {
    _ = unloadModel(config, mlx_ops);
}

/// The request-end hook on the real MLX pool.
pub fn endRequestMlx(config: *const ModelConfig) void {
    _ = endRequest(config, mlx_ops);
}

fn mlxSetLimit(bytes: u64) void {
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, @intCast(bytes));
}

fn mlxClear() void {
    _ = mlx.mlx_clear_cache();
}

/// The limit counts MLX's own active bytes and its pool, so it is installed over what is live now.
fn mlxSetTransient(bytes: u64) void {
    if (bytes == 0) {
        _ = mlx.setTransientLimit(0);
        return;
    }
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.setTransientLimit(@intCast(@as(u64, active) -| mlx.externalMemory() +| bytes));
}

const mlx_ops: PoolOps = .{ .set_limit = &mlxSetLimit, .clear = &mlxClear, .set_transient = &mlxSetTransient };

/// The image-encode hooks on the real MLX allocator.
pub fn enterVisionMlx(bill: u64, config: *const ModelConfig) void {
    enterVision(bill, config, mlx_ops);
}

pub fn leaveVisionMlx(config: *const ModelConfig) void {
    leaveVision(config, mlx_ops);
}

/// The pools a prefill may release when `cache_bytes` is split evenly over the layers: the load-time estimate the
/// engine's own count replaces once it is built.
pub fn plannedBytes(c: *const Config, stq: bool, cache_bytes: u64) u64 {
    if (!stq) return 0;
    const split = c.cedSplit() orelse return 0;
    return cache_bytes / c.layers * (c.layers - split);
}

/// PURE: how many `pools` (bytes each, in release order) must go before `headroom` covers `need`; null when even all
/// of them leave it short.
pub fn releaseCount(pools: []const u64, need: u64, headroom: u64) ?usize {
    var have = headroom;
    for (pools, 0..) |bytes, i| {
        if (have >= need) return i;
        have +|= bytes;
    }
    return if (have >= need) pools.len else null;
}

/// Before a forward of `rows` that ends at `seq`. A prefill-width forward releases decoder pools, latest first, until
/// its need fits the live headroom, and is refused when all of them do not cover it. A decode forward reads released
/// pools back while each fits beside its own working set.
pub fn beforeForward(engine: *stream.Engine, config: *const ModelConfig, c: *const Config, rows: u64, seq: u64) !void {
    _ = enterForward(rows, seq, config, c, mlx_ops);
    const headroom = headroom_fn orelse return;
    if (engine.borrow_from >= engine.layers.len) return;
    if (rows <= verify_rows) return restore(engine, config, headroom, c.forwardCap(rows, seq));
    const need = prefillNeed(c, rows, seq);
    if (headroom(config) >= need) return;
    _ = mlx.mlx_clear_cache();
    const before = headroom(config);
    const gpa = engine.allocator;
    const order = try gpa.alloc(u16, engine.layers.len - engine.borrow_from);
    defer gpa.free(order);
    const pools = try gpa.alloc(u64, order.len);
    defer gpa.free(pools);
    var n: usize = 0;
    var li = engine.layers.len;
    while (li > engine.borrow_from) {
        li -= 1;
        if (engine.isBorrowed(@intCast(li))) continue;
        order[n] = @intCast(li);
        pools[n] = engine.poolBytes(@intCast(li));
        n += 1;
    }
    // A resident pack pays for the union workspace out of its first release.
    const count = releaseCount(pools[0..n], need +| engine.unionPending(), before) orelse {
        log.warn("[dsv41-borrow] a {d}-row forward at {d} needs {d:.2} GB; the headroom and every decoder pool hold {d:.2} GB\n", .{ rows, seq, gb(need), gb((before +| sum(pools[0..n])) -| engine.unionPending()) });
        return error.PrefillDoesNotFit;
    };
    const wired = status.getWiredMemBytes();
    var released: u64 = 0;
    var have = before;
    var i: usize = 0;
    // The plan counts what the pools hold; the live headroom decides, so a short release takes the next pool too.
    while (i < n and (i < count or have < need)) : (i += 1) {
        released += try engine.borrowLayer(order[i]);
        have = headroom(config);
    }
    if (released > 0) {
        const waited = waitForUnwire(wired, released);
        log.info("[dsv41-borrow] released layers {d}..{d} ({d:.2} GB) for a {d}-row forward at {d}: need {d:.2} GB, headroom {d:.2} -> {d:.2} GB, unwired in {d} ms\n", .{
            order[i - 1], order[0], gb(released), rows, seq, gb(need), gb(before), gb(have), waited,
        });
    }
    if (have < need) return error.PrefillDoesNotFit;
}

fn restore(engine: *stream.Engine, config: *const ModelConfig, headroom: *const fn (*const ModelConfig) u64, work: u64) !void {
    var any = false;
    for (engine.borrow_from..engine.layers.len) |li| any = any or engine.isBorrowed(@intCast(li));
    if (!any) return;
    _ = mlx.mlx_clear_cache();
    const io = std.Io.Threaded.global_single_threaded.io();
    const t0 = std.Io.Timestamp.now(io, .awake);
    var bytes: u64 = 0;
    var count: usize = 0;
    var left: usize = 0;
    for (engine.borrow_from..engine.layers.len) |li| {
        if (!engine.isBorrowed(@intCast(li))) continue;
        if (left > 0 or headroom(config) < work +| engine.lentBytes(@intCast(li))) {
            left += 1;
            continue;
        }
        bytes += engine.lentBytes(@intCast(li));
        try engine.restoreLayer(@intCast(li));
        count += 1;
    }
    const ms: u64 = @intCast(@divTrunc(t0.untilNow(io, .awake).nanoseconds, std.time.ns_per_ms));
    if (count > 0) log.info("[dsv41-borrow] restored {d} layers ({d:.2} GB) in {d} ms before decode\n", .{ count, gb(bytes), ms });
    if (left > 0) log.info("[dsv41-borrow] {d} layers stay released: no room beside the decode forward\n", .{left});
}

/// The kernel unwires a freed Metal buffer asynchronously; the forward waits until most of what was released has left
/// the wired set, at most a second. Returns the ms waited.
fn waitForUnwire(wired_before: u64, freed: u64) u64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    var waited: u64 = 0;
    while (wired_before -| status.getWiredMemBytes() < freed / 10 * 9 and waited < 1000) : (waited += 10) {
        std.Io.sleep(io, .fromMilliseconds(10), .real) catch break;
    }
    return waited;
}

fn sum(xs: []const u64) u64 {
    var total: u64 = 0;
    for (xs) |x| total +|= x;
    return total;
}

fn gb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1e9;
}

const testing = std.testing;

test "dsv41 borrow: a forward releases pools in order until the headroom covers its need, and no more" {
    const pools = [_]u64{ 3, 3, 2, 4 };
    try testing.expectEqual(@as(?usize, 0), releaseCount(&pools, 10, 10));
    try testing.expectEqual(@as(?usize, 1), releaseCount(&pools, 10, 7));
    try testing.expectEqual(@as(?usize, 2), releaseCount(&pools, 10, 6));
    try testing.expectEqual(@as(?usize, 4), releaseCount(&pools, 12, 0));
    // Every pool short of the need: the forward is refused, never run over the edge.
    try testing.expectEqual(@as(?usize, null), releaseCount(&pools, 13, 0));
}

test "dsv41 borrow: a forward's need is its cap, a decode forward's far smaller" {
    const c = try cfg_mod.parseText(cfg_mod.release_config);
    const ctx: u64 = 1 << 20;
    try testing.expectEqual(c.forwardCap(2048, ctx), prefillNeed(&c, 2048, ctx));
    // A last chunk that absorbed its tail is priced at its own rows.
    try testing.expect(prefillNeed(&c, 2048 + 511, ctx) > prefillNeed(&c, 2048, ctx));
    try testing.expect(c.forwardCap(verify_rows, ctx) * 3 < prefillNeed(&c, 512, ctx) * 2);
}

test "dsv41 borrow: the load-time estimate is the decoder layers' even share of the cache, STQ packs only" {
    const c = try cfg_mod.parseText(cfg_mod.release_config);
    try testing.expectEqual(@as(u64, 20 * 2_000_000_000), plannedBytes(&c, true, 80_000_000_000));
    try testing.expectEqual(@as(u64, 0), plannedBytes(&c, false, 80_000_000_000));
}

test "dsv41 borrow: the pool cap is the decode cap up to a verify block and the RAM-wide cap above it" {
    const decode: u64 = 1 << 30;
    const ram: u64 = 8 << 30;
    try testing.expectEqual(decode, poolCap(1, decode, ram));
    try testing.expectEqual(decode, poolCap(verify_rows, decode, ram));
    try testing.expectEqual(ram, poolCap(verify_rows + 1, decode, ram));
    try testing.expectEqual(ram, poolCap(1 << 20, decode, ram));
    // Never past the RAM-wide cap, and never under the decode cap it is the floor of.
    try testing.expectEqual(@as(u64, 4 << 30), poolCap(2048, decode, 4 << 30));
    try testing.expectEqual(@as(u64, 512 << 20), poolCap(31, decode, 512 << 20));
    try testing.expectEqual(@as(u64, 1 << 20), poolCap(4096, decode, 1 << 20));
}

test "dsv41 borrow: at the budget-115 geometry no chunk releases a pool, and at 117 a 2048-row chunk releases one" {
    const c = try cfg_mod.parseText(cfg_mod.release_config);
    // Live at `--ssd-budget-gb 115` (280 slots a layer): the decoder pools a prefill may release, and the headroom there.
    const pools = [_]u64{ 2_700_000_000, 2_700_000_000, 2_760_000_000 };
    const headroom: u64 = 8_950_000_000;
    try testing.expectEqual(@as(?usize, 0), releaseCount(&pools, prefillNeed(&c, 31, 14766), headroom));
    try testing.expectEqual(@as(?usize, 0), releaseCount(&pools, prefillNeed(&c, 2048, 2048), headroom));
    try testing.expectEqual(@as(?usize, 0), releaseCount(&pools, prefillNeed(&c, 2048 + 511, 1 << 20), headroom));
    // At 117 the headroom is ~4.6 GB.
    try testing.expectEqual(@as(?usize, 1), releaseCount(&pools, prefillNeed(&c, 2048, 2048), 4_600_000_000));
}

const PoolProbe = struct {
    var limits: [8]u64 = undefined;
    var n: usize = 0;
    var cleared: usize = 0;
    var transient: u64 = 0;
    var transient_sets: usize = 0;
    fn reset() void {
        n = 0;
        cleared = 0;
        transient = 0;
        transient_sets = 0;
    }
    fn capTransient(bytes: u64) void {
        transient = bytes;
        transient_sets += 1;
    }
    fn set(bytes: u64) void {
        limits[n] = bytes;
        n += 1;
    }
    fn clear() void {
        cleared += 1;
    }
    const ops: PoolOps = .{ .set_limit = &set, .clear = &clear, .set_transient = &capTransient };
};

fn withPoolState(ram_cap: u64, body: *const fn () anyerror!void) !void {
    const saved = .{ pool_ram_cap, pool_now, ram_limit, transient_now };
    defer {
        pool_ram_cap = saved[0];
        pool_now = saved[1];
        ram_limit = saved[2];
        transient_now = saved[3];
    }
    pool_ram_cap = ram_cap;
    pool_now = null;
    ram_limit = null;
    transient_now = false;
    PoolProbe.reset();
    try body();
}

fn dsv41Config() !ModelConfig {
    var config = ModelConfig{ .model_type = "deepseek_v41" };
    config.dsv41 = try cfg_mod.parseText(cfg_mod.release_config);
    return config;
}

test "dsv41 borrow: a forward moves the pool cap to its rows' cap, and lowering it clears the pool" {
    const Body = struct {
        fn run() anyerror!void {
            const config = try dsv41Config();
            const c = &config.dsv41.?;
            const decode = config.expertPoolBytes();
            // A decode forward finds the ledger's cap in force: nothing to move.
            try testing.expect(!enterForward(1, 9_000, &config, c, PoolProbe.ops));
            try testing.expect(!enterForward(17, 9_000, &config, c, PoolProbe.ops));
            try testing.expectEqual(@as(usize, 0), PoolProbe.n);
            try testing.expect(enterForward(2048, 2048, &config, c, PoolProbe.ops));
            try testing.expect(!enterForward(2048, 4096, &config, c, PoolProbe.ops));
            try testing.expectEqual(@as(u64, 8 << 30), PoolProbe.limits[0]);
            try testing.expectEqual(@as(usize, 0), PoolProbe.cleared);
            // A narrower forward after a wide one cannot leave the wide pool parked.
            try testing.expect(enterForward(1, 4097, &config, c, PoolProbe.ops));
            try testing.expectEqual(decode, PoolProbe.limits[PoolProbe.n - 1]);
            try testing.expectEqual(@as(usize, 1), PoolProbe.cleared);
        }
    };
    try withPoolState(8 << 30, &Body.run);
    // Unset (tests, `sushi kld`), nothing is moved or capped.
    const config = try dsv41Config();
    PoolProbe.reset();
    try testing.expect(!enterForward(2048, 2048, &config, &config.dsv41.?, PoolProbe.ops));
    try testing.expectEqual(@as(usize, 0), PoolProbe.n);
    try testing.expectEqual(@as(usize, 0), PoolProbe.transient_sets);
}

test "dsv41 borrow: every phase installs the cap it bills and the decode pool beside it" {
    const Body = struct {
        fn run() anyerror!void {
            const config = try dsv41Config();
            const c = &config.dsv41.?;
            const pool = config.expertPoolBytes();
            _ = enterForward(1, 9_000, &config, c, PoolProbe.ops);
            try testing.expectEqual(c.forwardCap(1, 9_000) + pool, PoolProbe.transient);
            _ = enterForward(17, 225_000, &config, c, PoolProbe.ops);
            try testing.expectEqual(c.forwardCap(17, 225_000) + pool, PoolProbe.transient);
            _ = enterForward(2048, 225_000, &config, c, PoolProbe.ops);
            try testing.expectEqual(c.forwardCap(2048, 225_000) + pool, PoolProbe.transient);
            // The same rows further into the context cost more: the cap follows the position.
            const before = PoolProbe.transient;
            _ = enterForward(2048, 1 << 20, &config, c, PoolProbe.ops);
            try testing.expect(PoolProbe.transient > before);
            // An image encode is capped at its own bill.
            enterVision(1_100_000_000, &config, PoolProbe.ops);
            try testing.expectEqual(@as(u64, 1_100_000_000) + pool, PoolProbe.transient);
            leaveVision(&config, PoolProbe.ops);
            try testing.expectEqual(@as(u64, 0), PoolProbe.transient);
            // An encode with no bill is not capped to the pool alone.
            const none = PoolProbe.transient_sets;
            enterVision(0, &config, PoolProbe.ops);
            try testing.expectEqual(none, PoolProbe.transient_sets);
            // Another architecture is never capped.
            const qwen = ModelConfig{ .model_type = "qwen4_exp" };
            const sets = PoolProbe.transient_sets;
            enterVision(1 << 30, &qwen, PoolProbe.ops);
            try testing.expectEqual(sets, PoolProbe.transient_sets);
        }
    };
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: a request that ends lifts the transient cap and lowers the pool cap, even with no decode forward" {
    const Body = struct {
        fn run() anyerror!void {
            const config = try dsv41Config();
            _ = enterForward(2048, 2048, &config, &config.dsv41.?, PoolProbe.ops);
            try testing.expect(PoolProbe.transient > 0);
            try testing.expect(endRequest(&config, PoolProbe.ops));
            try testing.expectEqual(@as(u64, 0), PoolProbe.transient);
            try testing.expectEqual(config.expertPoolBytes(), PoolProbe.limits[PoolProbe.n - 1]);
            try testing.expectEqual(@as(usize, 1), PoolProbe.cleared);
            // Already at rest: nothing to do.
            const sets = PoolProbe.transient_sets;
            try testing.expect(!endRequest(&config, PoolProbe.ops));
            try testing.expectEqual(sets, PoolProbe.transient_sets);
            // A decode request that ends: the cap is lifted though the pool cap never moved.
            _ = enterForward(1, 9_000, &config, &config.dsv41.?, PoolProbe.ops);
            try testing.expect(endRequest(&config, PoolProbe.ops));
            try testing.expectEqual(@as(u64, 0), PoolProbe.transient);
            // Another architecture's request never touches either cap.
            _ = enterForward(2048, 2048, &config, &config.dsv41.?, PoolProbe.ops);
            const qwen = ModelConfig{ .model_type = "qwen4_exp" };
            try testing.expect(!endRequest(&qwen, PoolProbe.ops));
            try testing.expect(PoolProbe.transient > 0);
        }
    };
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: unloading DeepSeek gives the next model the process-wide pool cap and forgets the raised cap" {
    const Body = struct {
        fn run() anyerror!void {
            const config = try dsv41Config();
            const ram: u64 = 8 << 30;
            poolCapped(ram);
            // A second load keeps the cap the first one replaced.
            poolCapped(config.expertPoolBytes());
            try testing.expect(enterForward(2048, 2048, &config, &config.dsv41.?, PoolProbe.ops));
            try testing.expect(pool_now != null);
            try testing.expect(unloadModel(&config, PoolProbe.ops));
            try testing.expectEqual(@as(u64, 0), PoolProbe.transient);
            try testing.expectEqual(ram, PoolProbe.limits[PoolProbe.n - 1]);
            try testing.expectEqual(@as(?u64, null), pool_now);
            try testing.expectEqual(@as(?u64, null), ram_limit);
            // Nothing to give back twice, and another architecture's unload never touches the cap.
            const n = PoolProbe.n;
            try testing.expect(!unloadModel(&config, PoolProbe.ops));
            poolCapped(ram);
            const qwen = ModelConfig{ .model_type = "qwen4_exp" };
            try testing.expect(!unloadModel(&qwen, PoolProbe.ops));
            try testing.expectEqual(n, PoolProbe.n);
        }
    };
    try withPoolState(8 << 30, &Body.run);
}
