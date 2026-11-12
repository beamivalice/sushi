//! DeepSeek-V4.1's prefill takes its working set from the expert cache. A forward that does not fit the free memory
//! first releases whole decoder-layer pools, latest layer first: under CED those layers run only a chunk's last
//! `window` rows, and a released layer's picks are read into the union workspace. The first decode forward after the
//! prefill reads the pools back. Every forward runs under a transient cap (`Config.forwardCap`), which is what admission
//! bills: a request's state, a decode forward's cap and the part of a prefill's the pools cannot cover.
const std = @import("std");
const builtin = @import("builtin");
const arch_config = @import("../arch/config.zig");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const status = @import("../status.zig");
const stream = @import("../expert_stream.zig");
const cfg_mod = @import("dsv41_config.zig");
const ModelConfig = @import("../model.zig").ModelConfig;
const Config = cfg_mod.Config;
const MlxAsk = cfg_mod.MlxAsk;

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
/// Loads on MLX that hold the lowered cap: the last one to unload gives the process-wide cap back.
var ram_cappers: u32 = 0;

/// Whether `config`'s forwards allocate on the engine of our own. Such a model moves the engine's budget and nothing
/// of MLX's; a model on MLX moves MLX's caps and nothing of the engine's. Each hook below takes one branch.
fn ownsEngine(config: *const ModelConfig) bool {
    return if (config.dsv41) |*c| ownDecode(c) else false;
}

pub const PoolOps = struct {
    set_limit: *const fn (bytes: u64) void,
    clear: *const fn () void,
    /// Caps what MLX may allocate beyond what it holds now, its freed-buffer pool included; 0 lifts the cap.
    set_transient: *const fn (bytes: u64) void,
    /// Caps what the engine of our own may hold beyond what its tensors in use ask for, its free buffers included.
    set_own: *const fn (bytes: u64) void,
};

/// Where a pack's decode forwards (one row, DSpark draft, verify rows) run, and why not on the engine of our own.
pub const DecodeSide = enum {
    engine,
    /// The engine has no `quantized_matmul`, which an FP8 trunk and MXFP4 experts launch.
    mlx_not_stq,
    /// Every kernel of the engine is held to MLX's bytes on a GPU with matrix units; its other matmul tiles have
    /// not run on one without.
    mlx_no_matrix_units,
    /// `--ds41-mlx-engine`.
    mlx_by_flag,
    /// The model's `ds41_cooler` is off.
    mlx_by_setting,
};

/// PURE: the side for a pack that is or is not STQ throughout, on a GPU with or without matrix units. A pack or
/// Mac the engine cannot serve keeps its own reason; the user's ask only moves a decode the engine would have run.
pub fn decodeSide(stq: bool, matrix_units: bool, ask: MlxAsk) DecodeSide {
    if (!stq) return .mlx_not_stq;
    if (!matrix_units) return .mlx_no_matrix_units;
    return switch (ask) {
        .none => .engine,
        .flag => .mlx_by_flag,
        .setting => .mlx_by_setting,
    };
}

/// The load line for a side. The engine of our own ships as Sushi Cooler.
pub fn decodeLine(side: DecodeSide) []const u8 {
    return switch (side) {
        .engine => "[dsv41] decode forwards (one row, DSpark draft and verify rows): on Sushi Cooler, the engine of our own",
        .mlx_not_stq => "[dsv41] decode forwards: on MLX (Sushi Cooler has no quantized_matmul for an FP8 trunk or MXFP4 experts)",
        .mlx_no_matrix_units => "[dsv41] decode forwards: on MLX (Sushi Cooler is proven on a GPU with matrix units only)",
        .mlx_by_flag => "[dsv41] decode forwards: on MLX because --ds41-mlx-engine was given (Sushi Cooler is off)",
        .mlx_by_setting => "[dsv41] decode forwards: on MLX because model-settings.json ds41_cooler is off (Sushi Cooler is off)",
    };
}

/// Set while the load's warm runs. `serve` installs the caps after the load, so each engine forward of the warm takes the
/// budget a request's forward of the same rows and position has (`enterWarm`), or it would hold any amount.
pub var warm_budget: bool = false;

/// A test names the engine budget a warm forward gets.
pub var warm_budget_override: ?u64 = null;

/// Whether a warm forward installed a budget, which `endWarm` lifts.
var warm_budget_set: bool = false;

/// A test names the side its model decodes on.
pub var own_override: ?bool = null;

/// The load line of a pack whose decode is on the engine: its prefill chunks run there too, at every width.
pub const prefill_line = "[dsv41] prefill chunks wider than 16 rows: on Sushi Cooler, the engine of our own";

pub fn decodeSideOf(c: *const Config) DecodeSide {
    return decodeSide(c.expert_dec != null and c.trunk_dec != null, mlx.engine.matrixUnits(), c.mlx_ask);
}

/// Whether this pack's decode forwards run on the engine of our own. The load and every bill ask here.
pub fn ownDecode(c: *const Config) bool {
    if (own_override) |own| return own;
    // The tests written against MLX's decode say nothing and keep it; the engine's own tests turn it on.
    if (comptime builtin.is_test) return false;
    return decodeSideOf(c) == .engine;
}

/// What the engine of our own may hold during a request at `seq`, beside MLX's cap: a decode forward's cap again,
/// since a forward of ours makes on its own queue what MLX's would have made on MLX's.
pub fn ownCap(c: *const Config, seq: u64) u64 {
    return if (ownDecode(c)) c.forwardCap(verify_rows, seq) else 0;
}

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
    if (!arch_config.isOf(config, "dsv41")) return false;
    if (pool_ram_cap == 0) {
        enterWarm(rows, seq, c, ops);
        return false;
    }
    const split = splitOf(rows, seq, c);
    if (ownDecode(c)) {
        ops.set_own(split.own);
        return false;
    }
    setTransient(phaseLimit(split.mlx, config), ops);
    return moveCap(rowsCap(rows, config), config, ops);
}

/// A forward of the load's warm: the engine's share of the cap a request's forward of `rows` ending at `seq` has. Nothing
/// outside the warm.
pub fn enterWarm(rows: u64, seq: u64, c: *const Config, ops: PoolOps) void {
    if (!warm_budget or !ownDecode(c)) return;
    warm_budget_set = true;
    ops.set_own(warm_budget_override orelse splitOf(rows, seq, c).own);
}

pub fn enterWarmMlx(rows: u64, seq: u64, c: *const Config) void {
    enterWarm(rows, seq, c, mlx_ops);
}

/// The warm is over: the engine runs under no budget until a request's forward installs its own.
pub fn endWarm(ops: PoolOps) void {
    warm_budget = false;
    if (!warm_budget_set) return;
    warm_budget_set = false;
    ops.set_own(std.math.maxInt(u64));
}

pub fn endWarmMlx() void {
    endWarm(mlx_ops);
}

/// What a forward's cap leaves to MLX and to the engine of our own. A wide chunk on the engine takes the whole cap its
/// bill reserves: the state rows it appends grow in the engine's pool with the rest, and MLX allocates nothing.
fn splitOf(rows: u64, seq: u64, c: *const Config) struct { mlx: u64, own: u64 } {
    const cap = c.forwardCap(rows, seq);
    if (!ownDecode(c)) return .{ .mlx = cap, .own = 0 };
    return .{ .mlx = 0, .own = if (rows > verify_rows) cap else ownCap(c, seq) };
}

/// An image encode of `bill` bytes (`request_media.visionEncodeBill`) starts: its allocations are capped at it. No bill, no cap.
pub fn enterVision(bill: u64, config: *const ModelConfig, ops: PoolOps) void {
    if (pool_ram_cap == 0 or bill == 0 or !arch_config.isOf(config, "dsv41") or ownsEngine(config)) return;
    setTransient(phaseLimit(bill, config), ops);
}

/// The encode is over; the next forward installs its own cap.
pub fn leaveVision(config: *const ModelConfig, ops: PoolOps) void {
    if (pool_ram_cap == 0 or !arch_config.isOf(config, "dsv41") or ownsEngine(config) or !transient_now) return;
    setTransient(0, ops);
}

/// A request is over: whatever forward ran last, the transient cap is lifted (a restore allocates the next request's
/// state before its first forward) and the pool cap drops back to the decode cap.
pub fn endRequest(config: *const ModelConfig, ops: PoolOps) bool {
    if (pool_ram_cap == 0 or !arch_config.isOf(config, "dsv41")) return false;
    if (ownsEngine(config)) {
        // The next request bills the engine's cap from its own length: the shortest one covers what stays.
        ops.set_own(ownCap(&config.dsv41.?, 1));
        return false;
    }
    const was_on = transient_now;
    if (was_on) setTransient(0, ops);
    return moveCap(config.expertPoolBytes(), config, ops) or was_on;
}

/// The load lowered the process-wide cap from `prev` to the ledger's decode cap.
pub fn poolCapped(prev: u64) void {
    if (ram_limit == null) ram_limit = prev;
    ram_cappers += 1;
}

/// A DeepSeek model is unloaded: the next model gets the process-wide cap back, and the raised cap is forgotten.
pub fn unloadModel(config: *const ModelConfig, ops: PoolOps) bool {
    if (!arch_config.isOf(config, "dsv41")) return false;
    if (ownsEngine(config)) {
        // The engine's free buffers go, and then its cap: a cap is a forward's, and the next load makes tensors before any.
        ops.set_own(0);
        ops.set_own(std.math.maxInt(u64));
        return false;
    }
    pool_now = null;
    if (transient_now) setTransient(0, ops);
    const prev = ram_limit orelse return false;
    ram_cappers -|= 1;
    if (ram_cappers > 0) return false;
    ram_limit = null;
    ops.set_limit(prev);
    return true;
}

/// The unload hook on the real MLX pool.
pub fn unloadModelMlx(config: *const ModelConfig) void {
    mlx.engine.close();
    _ = unloadModel(config, mlx_ops);
}

/// The request-end hook on the real MLX pool.
pub fn endRequestMlx(config: *const ModelConfig) void {
    // An idle server holds no open pass, so a stop between requests ends its thread cleanly.
    mlx.engine.close();
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

fn ownSet(bytes: u64) void {
    mlx.engine.budget(bytes) catch |e| log.warn("[engine] pool: the cap of {d} MB was not installed: {s}\n", .{ bytes >> 20, @errorName(e) });
}

const mlx_ops: PoolOps = .{ .set_limit = &mlxSetLimit, .clear = &mlxClear, .set_transient = &mlxSetTransient, .set_own = &ownSet };

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
    var own: u64 = 0;
    var own_sets: usize = 0;
    fn reset() void {
        n = 0;
        cleared = 0;
        transient = 0;
        transient_sets = 0;
        own = 0;
        own_sets = 0;
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
    fn capOwn(bytes: u64) void {
        own = bytes;
        own_sets += 1;
    }
    const ops: PoolOps = .{ .set_limit = &set, .clear = &clear, .set_transient = &capTransient, .set_own = &capOwn };
};

fn withPoolState(ram_cap: u64, body: *const fn () anyerror!void) !void {
    const saved = .{ pool_ram_cap, pool_now, ram_limit, transient_now, ram_cappers };
    defer {
        pool_ram_cap = saved[0];
        pool_now = saved[1];
        ram_limit = saved[2];
        transient_now = saved[3];
        ram_cappers = saved[4];
    }
    pool_ram_cap = ram_cap;
    pool_now = null;
    ram_limit = null;
    transient_now = false;
    ram_cappers = 0;
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

test "dsv41 borrow: a wide chunk on the engine gives the engine's pool the whole cap it bills, the state's growth included" {
    const Body = struct {
        fn run() anyerror!void {
            const config = try dsv41Config();
            const c = &config.dsv41.?;
            own_override = true;
            defer own_override = null;
            for ([_][2]u64{ .{ 2048, 2048 }, .{ 1100, 1 << 20 }, .{ 31, 4096 } }) |chunk| {
                _ = enterForward(chunk[0], chunk[1], &config, c, PoolProbe.ops);
                try testing.expectEqual(c.forwardCap(chunk[0], chunk[1]), PoolProbe.own);
            }
            // Nothing of MLX's is a Cooler load's to move: its caps belong to the models that allocate there.
            try testing.expectEqual(@as(usize, 0), PoolProbe.n);
            try testing.expectEqual(@as(usize, 0), PoolProbe.transient_sets);
        }
    };
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: a wide chunk of a pack that decodes on MLX keeps the whole cap on MLX" {
    const Body = struct {
        fn run() anyerror!void {
            const config = try dsv41Config();
            const c = &config.dsv41.?;
            own_override = false;
            defer own_override = null;
            _ = enterForward(2048, 2048, &config, c, PoolProbe.ops);
            try testing.expectEqual(c.forwardCap(2048, 2048), PoolProbe.transient - config.expertPoolBytes());
            try testing.expectEqual(@as(u64, 0), PoolProbe.own);
        }
    };
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: the load's warm gives each engine forward, of any width, the budget serve gives it, and nothing else" {
    const Body = struct {
        fn run() anyerror!void {
            const config = try dsv41Config();
            const c = &config.dsv41.?;
            own_override = true;
            defer {
                own_override = null;
                warm_budget = false;
                warm_budget_set = false;
            }
            // Outside the warm a forward is not capped before `serve` installs the caps, as `sushi kld` and the tests are not.
            _ = enterForward(2048, 2048, &config, c, PoolProbe.ops);
            enterWarm(1, 4096, c, PoolProbe.ops);
            try testing.expectEqual(@as(u64, 0), PoolProbe.own);
            warm_budget = true;
            // A wide chunk takes the one cap whole; a decode-width forward, whatever its rows, the decode cap at its position.
            try testing.expect(!enterForward(2048, 2048, &config, c, PoolProbe.ops));
            try testing.expectEqual(c.forwardCap(2048, 2048), PoolProbe.own);
            for ([_]u64{ 1, 3, verify_rows }) |rows| {
                enterWarm(rows, 4096 + rows, c, PoolProbe.ops);
                try testing.expectEqual(ownCap(c, 4096 + rows), PoolProbe.own);
            }
            try testing.expectEqual(@as(usize, 0), PoolProbe.n);
            try testing.expectEqual(@as(usize, 0), PoolProbe.transient_sets);
            // The warm's end lifts what it set, once, and leaves another load's cap alone.
            endWarm(PoolProbe.ops);
            try testing.expect(!warm_budget);
            try testing.expectEqual(@as(u64, std.math.maxInt(u64)), PoolProbe.own);
            PoolProbe.reset();
            endWarm(PoolProbe.ops);
            try testing.expectEqual(@as(u64, 0), PoolProbe.own);
            // Another architecture's forward is never capped.
            warm_budget = true;
            const qwen = ModelConfig{ .model_type = "qwen4_exp" };
            _ = enterForward(2048, 2048, &qwen, c, PoolProbe.ops);
            try testing.expectEqual(@as(u64, 0), PoolProbe.own);
        }
    };
    try withPoolState(0, &Body.run);
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
            defer ram_cappers = 0;
            const qwen = ModelConfig{ .model_type = "qwen4_exp" };
            try testing.expect(!unloadModel(&qwen, PoolProbe.ops));
            try testing.expectEqual(n, PoolProbe.n);
        }
    };
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: with decode on the engine of our own every forward installs its cap there, and a request's end the shortest one" {
    const Body = struct {
        fn run() anyerror!void {
            const config = try dsv41Config();
            const c = &config.dsv41.?;
            const was = own_override;
            defer own_override = was;
            own_override = false;
            _ = enterForward(1, 9_000, &config, c, PoolProbe.ops);
            try testing.expectEqual(@as(u64, 0), PoolProbe.own);
            own_override = true;
            _ = enterForward(1, 9_000, &config, c, PoolProbe.ops);
            try testing.expectEqual(c.forwardCap(verify_rows, 9_000), PoolProbe.own);
            // A prefill chunk takes the cap less MLX's share.
            _ = enterForward(2048, 225_000, &config, c, PoolProbe.ops);
            try testing.expectEqual(c.forwardCap(2048, 225_000), PoolProbe.own);
            _ = endRequest(&config, PoolProbe.ops);
            try testing.expectEqual(c.forwardCap(verify_rows, 1), PoolProbe.own);
            // The unload trims the engine's pool and leaves it without a cap for the next load.
            _ = unloadModel(&config, PoolProbe.ops);
            try testing.expectEqual(@as(u64, std.math.maxInt(u64)), PoolProbe.own);
        }
    };
    PoolProbe.reset();
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: decode runs on the engine for an STQ pack on a GPU with matrix units, and on MLX otherwise, by name" {
    try testing.expectEqual(DecodeSide.engine, decodeSide(true, true, .none));
    try testing.expectEqual(DecodeSide.mlx_not_stq, decodeSide(false, true, .none));
    try testing.expectEqual(DecodeSide.mlx_not_stq, decodeSide(false, false, .none));
    try testing.expectEqual(DecodeSide.mlx_no_matrix_units, decodeSide(true, false, .none));
}

test "dsv41 borrow: the user's ask for MLX moves a pack the engine could serve, and a pack or Mac it cannot serve keeps its own reason" {
    try testing.expectEqual(DecodeSide.mlx_by_flag, decodeSide(true, true, .flag));
    try testing.expectEqual(DecodeSide.mlx_by_setting, decodeSide(true, true, .setting));
    try testing.expectEqual(DecodeSide.mlx_not_stq, decodeSide(false, true, .flag));
    try testing.expectEqual(DecodeSide.mlx_no_matrix_units, decodeSide(true, false, .setting));
    var c: Config = .{};
    c.mlx_ask = .setting;
    try testing.expectEqual(DecodeSide.mlx_not_stq, decodeSideOf(&c));
}

test "dsv41 borrow: the load line names the side and why, with Sushi Cooler as the engine's name" {
    try testing.expectEqualStrings("[dsv41] decode forwards (one row, DSpark draft and verify rows): on Sushi Cooler, the engine of our own", decodeLine(.engine));
    try testing.expectEqualStrings("[dsv41] decode forwards: on MLX because --ds41-mlx-engine was given (Sushi Cooler is off)", decodeLine(.mlx_by_flag));
    try testing.expectEqualStrings("[dsv41] decode forwards: on MLX because model-settings.json ds41_cooler is off (Sushi Cooler is off)", decodeLine(.mlx_by_setting));
    try testing.expect(std.mem.indexOf(u8, decodeLine(.mlx_not_stq), "FP8 trunk or MXFP4 experts") != null);
    try testing.expect(std.mem.indexOf(u8, decodeLine(.mlx_no_matrix_units), "matrix units") != null);
}

test "dsv41 borrow: beside a Cooler model, a model on MLX has its transient cap lifted at its request's end" {
    try mlx.requireGpu();
    const Body = struct {
        fn run() anyerror!void {
            const config = try dsv41Config();
            const c = &config.dsv41.?;
            const was_enabled = mlx.engine.enabled;
            mlx.engine.enabled = true;
            defer @import("../test_support.zig").restoreEngine(was_enabled);
            defer mlx.engine.budget(std.math.maxInt(u64)) catch {};
            defer own_override = null;
            mlx.engine.hold();
            defer mlx.engine.unhold();
            {
                const turn = mlx.engine.turnTo(false);
                defer turn.end();
                try testing.expectEqual(@as(usize, 0), mlx.setTransientLimit(0));
                own_override = false;
                _ = enterForward(1, 9_000, &config, c, mlx_ops);
            }
            {
                const turn = mlx.engine.turnTo(true);
                defer turn.end();
                own_override = true;
                _ = endRequest(&config, mlx_ops);
            }
            const turn = mlx.engine.turnTo(false);
            defer turn.end();
            own_override = false;
            _ = endRequest(&config, mlx_ops);
            try testing.expectEqual(@as(usize, 0), mlx.setTransientLimit(0));
        }
    };
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: beside a Cooler model's forward, a model on MLX drops its pool cap back for its decode" {
    try mlx.requireGpu();
    const Body = struct {
        fn run() anyerror!void {
            const config = try dsv41Config();
            const c = &config.dsv41.?;
            const was_enabled = mlx.engine.enabled;
            mlx.engine.enabled = true;
            defer @import("../test_support.zig").restoreEngine(was_enabled);
            defer mlx.engine.budget(std.math.maxInt(u64)) catch {};
            defer own_override = null;
            var orig: usize = 0;
            try mlx.check(mlx.mlx_set_cache_limit(&orig, 1 << 40));
            defer {
                var unused: usize = 0;
                _ = mlx.mlx_set_cache_limit(&unused, orig);
                _ = mlx.setTransientLimit(0);
            }
            mlx.engine.hold();
            defer mlx.engine.unhold();
            {
                const turn = mlx.engine.turnTo(false);
                defer turn.end();
                own_override = false;
                try testing.expect(enterForward(2048, 2048, &config, c, mlx_ops));
            }
            {
                const turn = mlx.engine.turnTo(true);
                defer turn.end();
                own_override = true;
                _ = enterForward(1, 4097, &config, c, mlx_ops);
            }
            const turn = mlx.engine.turnTo(false);
            defer turn.end();
            own_override = false;
            _ = enterForward(1, 2049, &config, c, mlx_ops);
            var now: usize = 0;
            try mlx.check(mlx.mlx_set_cache_limit(&now, 1 << 40));
            try testing.expectEqual(config.expertPoolBytes(), now);
        }
    };
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: a Cooler model's unload leaves MLX's lowered pool cap to the model on MLX, whose unload gives it back" {
    try mlx.requireGpu();
    const Body = struct {
        fn run() anyerror!void {
            const config = try dsv41Config();
            const was_enabled = mlx.engine.enabled;
            mlx.engine.enabled = true;
            defer @import("../test_support.zig").restoreEngine(was_enabled);
            defer mlx.engine.budget(std.math.maxInt(u64)) catch {};
            defer own_override = null;
            var orig: usize = 0;
            try mlx.check(mlx.mlx_set_cache_limit(&orig, 1 << 40));
            defer {
                var unused: usize = 0;
                _ = mlx.mlx_set_cache_limit(&unused, orig);
            }
            mlx.engine.hold();
            {
                // The model on MLX loads: the ledger lowers MLX's pool cap in its turn.
                const turn = mlx.engine.turnTo(false);
                defer turn.end();
                var prev: usize = 0;
                try mlx.check(mlx.mlx_set_cache_limit(&prev, 3 << 30));
                poolCapped(prev);
            }
            own_override = true;
            unloadModelMlx(&config);
            mlx.engine.unhold();
            var now: usize = 0;
            try mlx.check(mlx.mlx_set_cache_limit(&now, 3 << 30));
            try testing.expectEqual(@as(usize, 3 << 30), now);
            own_override = false;
            unloadModelMlx(&config);
            try mlx.check(mlx.mlx_set_cache_limit(&now, 1 << 40));
            try testing.expectEqual(@as(usize, 1 << 40), now);
        }
    };
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: a model on MLX never moves the engine's budget, at a forward, a request's end or its unload" {
    const Body = struct {
        fn run() anyerror!void {
            const config = try dsv41Config();
            const c = &config.dsv41.?;
            own_override = false;
            defer own_override = null;
            const sets = PoolProbe.own_sets;
            _ = enterForward(1, 9_000, &config, c, PoolProbe.ops);
            _ = enterForward(2048, 2048, &config, c, PoolProbe.ops);
            _ = endRequest(&config, PoolProbe.ops);
            _ = unloadModel(&config, PoolProbe.ops);
            try testing.expectEqual(sets, PoolProbe.own_sets);
        }
    };
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: two models on MLX: the lowered pool cap goes back when the second of them unloads" {
    const Body = struct {
        fn run() anyerror!void {
            const config = try dsv41Config();
            own_override = false;
            defer own_override = null;
            poolCapped(8 << 30);
            poolCapped(1 << 30);
            try testing.expect(!unloadModel(&config, PoolProbe.ops));
            try testing.expectEqual(@as(usize, 0), PoolProbe.n);
            try testing.expect(unloadModel(&config, PoolProbe.ops));
            try testing.expectEqual(@as(u64, 8 << 30), PoolProbe.limits[PoolProbe.n - 1]);
        }
    };
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: the ledger bills the pool that is parked: MLX's decode pool, or a Cooler load's engine buffers under its idle cap" {
    const config = try dsv41Config();
    const c = &config.dsv41.?;
    defer own_override = null;
    own_override = false;
    try testing.expectEqual(cfg_mod.dsv41_pool_bytes, config.expertPoolBytes());
    own_override = true;
    try testing.expectEqual(c.forwardCap(verify_rows, 1), config.expertPoolBytes());
    try testing.expect(config.expertPoolBytes() < cfg_mod.dsv41_pool_bytes);
    // What a request's end leaves the engine is what the ledger billed.
    const Body = struct {
        fn run() anyerror!void {
            const cfg = try dsv41Config();
            _ = endRequest(&cfg, PoolProbe.ops);
            try testing.expectEqual(cfg.expertPoolBytes(), PoolProbe.own);
        }
    };
    try withPoolState(8 << 30, &Body.run);
}
