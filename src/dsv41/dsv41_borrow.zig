//! DeepSeek-V4.1's prefill takes its working set from the expert cache. A forward that does not fit the free memory
//! first releases whole decoder-layer pools, latest layer first: under CED those layers run only a chunk's last
//! `window` rows, and a released layer's picks are read into the union workspace. The first decode forward after the
//! prefill reads the pools back. Every forward runs under a transient cap (`Config.forwardCap`), which is what admission
//! bills: a request's state, a decode forward's cap and the part of a prefill's the pools cannot cover.
const std = @import("std");
const arch_config = @import("../arch/config.zig");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const status = @import("../status.zig");
const stream = @import("../expert_stream.zig");
const cfg_mod = @import("dsv41_config.zig");
const ModelConfig = @import("../model.zig").ModelConfig;
const Config = cfg_mod.Config;
const cooler_side = @import("../cooler_side.zig");
const Mem = @import("../bills.zig").Mem;

/// The live headroom (`bills.headroomNow`), installed by the server. Unset (tests, `sushi kld`), nothing is ever released.
pub var headroom_fn: ?*const fn (*const ModelConfig) Headroom = null;
const Headroom = @import("../bills.zig").Headroom;
/// The need of the wide forward `beforeForward` is sizing, which a test's `headroom_fn` reads.
pub var forward_need: u64 = 0;

/// The widest verify block: up to it a forward is a decode forward, whose cap admission always reserves.
pub const verify_rows = cfg_mod.verify_rows;

/// What a prefill-width forward must find free before it runs: the most it may allocate on its side
/// (`Config.forwardCapOn`).
pub fn prefillNeed(c: *const Config, engine: bool, rows: u64, seq: u64) u64 {
    return c.forwardCapOn(engine, rows, seq);
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
    return config.dsv41 != null and cooler_side.onEngine(config);
}

pub const PoolOps = struct {
    set_limit: *const fn (bytes: u64) void,
    clear: *const fn () void,
    /// Caps what MLX may allocate beyond what it holds now, its freed-buffer pool included; 0 lifts the cap.
    set_transient: *const fn (bytes: u64) void,
    /// Caps what the engine of our own may hold beyond what its tensors in use ask for, its free buffers included.
    set_own: *const fn (bytes: u64) void,
    /// Gives MLX's pool and the engine's free buffers the process-wide cap the load lowered back.
    restore_ram: *const fn (bytes: u64) void,
};

/// Set while the load's warm runs. `serve` installs the caps after the load, so each engine forward of the warm takes the
/// budget a request's forward of the same rows and position has (`enterWarm`), or it would hold any amount.
pub var warm_budget: bool = false;

/// A test names the engine budget a warm forward gets.
pub var warm_budget_override: ?u64 = null;

/// Whether a warm forward installed a budget, which `endWarm` lifts.
var warm_budget_set: bool = false;

/// The load line of a pack whose decode is on the engine: its prefill chunks run there too, at every width.
pub const prefill_line = "[dsv41] prefill chunks wider than 16 rows: on Sushi Cooler, the engine of our own";

/// Whether the load keeps each expert's scale vectors in host memory once read (`expert_stream.KeptScales`): an STQ
/// pack on Sushi Cooler. The load turns it on and the ssd-budget ledger bills it from this one answer.
pub fn keepsScales(config: *const ModelConfig) bool {
    return config.dsv41.?.expert_dec != null and cooler_side.onEngine(config);
}

/// What the engine of our own may hold during a request at `seq`, beside MLX's cap: a decode forward's cap again,
/// since a forward of ours makes on its own queue what MLX's would have made on MLX's.
pub fn ownCap(c: *const Config, own: bool, seq: u64) u64 {
    return if (own) c.forwardCap(verify_rows, seq) else 0;
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
    const own = cooler_side.onEngine(config);
    const split = splitOf(rows, seq, c, own);
    // The engine's cap is the forward's own whether or not `serve` installed MLX's: the rest a request's end leaves in
    // force would refuse a wide chunk mid-forward, and none would let it hold past its bill.
    if (own) {
        ops.set_own(ownBudget(split.own));
        return false;
    }
    if (pool_ram_cap == 0) return false;
    setTransient(phaseLimit(split.mlx, config), ops);
    return moveCap(rowsCap(rows, config), config, ops);
}

/// `enterForward` of a joined decode forward, a row per request at `seqs`: its cap is `Config.joinedCap`.
pub fn enterRows(seqs: []const u64, config: *const ModelConfig, c: *const Config, ops: PoolOps) bool {
    if (!arch_config.isOf(config, "dsv41")) return false;
    const cap = c.joinedCap(seqs);
    if (cooler_side.onEngine(config)) {
        ops.set_own(ownBudget(cap));
        return false;
    }
    if (pool_ram_cap == 0) return false;
    setTransient(phaseLimit(cap, config), ops);
    return moveCap(rowsCap(seqs.len, config), config, ops);
}

/// `beforeForward` of a joined decode forward (`enterRows`).
pub fn beforeRowsForward(engine: *stream.Engine, config: *const ModelConfig, c: *const Config, seqs: []const u64) !void {
    try engine.healReleased();
    _ = enterRows(seqs, config, c, mlx_ops);
    const headroom = headroom_fn orelse return;
    if (engine.borrow_from >= engine.layers.len) return;
    return restore(engine, config, headroom, c.joinedCap(seqs));
}

/// The engine's budget of a forward that bills `bytes`; the warm's, which `endWarm` lifts, while it runs.
fn ownBudget(bytes: u64) u64 {
    if (!warm_budget) return bytes;
    warm_budget_set = true;
    return warm_budget_override orelse bytes;
}

/// A forward of the load's warm: the engine's share of the cap a request's forward of `rows` ending at `seq` has. Nothing
/// outside the warm.
pub fn enterWarm(rows: u64, seq: u64, c: *const Config, own: bool, ops: PoolOps) void {
    if (!warm_budget or !own) return;
    ops.set_own(ownBudget(splitOf(rows, seq, c, own).own));
}

pub fn enterWarmMlx(rows: u64, seq: u64, c: *const Config, own: bool) void {
    enterWarm(rows, seq, c, own, mlx_ops);
}

/// A joined decode forward of the load's warm, under the cap the served one installs (`enterRows`).
pub fn enterWarmRowsMlx(seqs: []const u64, c: *const Config) void {
    if (!warm_budget) return;
    mlx_ops.set_own(ownBudget(c.joinedCap(seqs)));
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
fn splitOf(rows: u64, seq: u64, c: *const Config, own: bool) struct { mlx: u64, own: u64 } {
    const cap = c.forwardCapOn(own, rows, seq);
    if (!own) return .{ .mlx = cap, .own = 0 };
    return .{ .mlx = 0, .own = if (rows > verify_rows) cap else ownCap(c, own, seq) };
}

/// Whether an image encode on the engine runs under its bill (`enterVision`).
var vision_billed: bool = false;

/// Whether the engine's cap is an image encode's bill: each pass of the tower keeps it, the bill holding the worst pass.
pub fn visionBilled() bool {
    return vision_billed;
}

/// An image encode of `bill` bytes (`request_media.visionEncodeBill`) starts: its allocations are capped at it, on the
/// engine its tower's passes, the patch arrays between them and the join of the spans. No bill, no cap.
pub fn enterVision(bill: u64, config: *const ModelConfig, ops: PoolOps) void {
    if (pool_ram_cap == 0 or bill == 0 or !arch_config.isOf(config, "dsv41")) return;
    if (ownsEngine(config)) {
        vision_billed = true;
        ops.set_own(bill);
        return;
    }
    setTransient(phaseLimit(bill, config), ops);
}

/// The encode is over; the next forward installs its own cap.
pub fn leaveVision(config: *const ModelConfig, ops: PoolOps) void {
    if (pool_ram_cap == 0 or !arch_config.isOf(config, "dsv41")) return;
    if (ownsEngine(config)) {
        vision_billed = false;
        return;
    }
    if (!transient_now) return;
    setTransient(0, ops);
}

/// A request is over: whatever forward ran last, the transient cap is lifted (a restore allocates the next request's
/// state before its first forward) and the pool cap drops back to the decode cap.
pub fn endRequest(config: *const ModelConfig, ops: PoolOps) bool {
    if (pool_ram_cap == 0 or !arch_config.isOf(config, "dsv41")) return false;
    // The engine's rest is the shared Cooler seam's (`arch/cooler.zig`): the decode pool the ledger parks.
    if (ownsEngine(config)) return false;
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
    ops.restore_ram(prev);
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

const mlx_ops: PoolOps = .{ .set_limit = &mlxSetLimit, .clear = &mlxClear, .set_transient = &mlxSetTransient, .set_own = &ownSet, .restore_ram = &@import("../gpu_mem.zig").restoreMlxPool };

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

/// Before a forward of `rows` that ends at `seq`. A released layer left with no union workspace is healed first. A
/// prefill-width forward releases decoder pools, latest first, until its need fits the live headroom, and is refused
/// when all of them do not cover it. A decode forward reads released pools back while each fits beside its own
/// working set.
/// `rest`: the prompt tokens the prefill runs after this forward (`ForwardCtx.prefill_rest`), 0 past it.
pub fn beforeForward(engine: *stream.Engine, config: *const ModelConfig, c: *const Config, rows: u64, seq: u64, rest: u64) !void {
    try engine.healReleased();
    _ = enterForward(rows, seq, config, c, mlx_ops);
    const headroom = headroom_fn orelse return;
    if (engine.borrow_from >= engine.layers.len) return;
    if (rows <= verify_rows) return restore(engine, config, headroom, c.forwardCap(rows, seq));
    const need = prefillNeed(c, cooler_side.onEngine(config), rows, seq);
    forward_need = need;
    if (headroom(config).forNeed(need) >= need) return;
    _ = mlx.mlx_clear_cache();
    const before = headroom(config).forNeed(need);
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
        const have = (before +| sum(pools[0..n])) -| engine.unionPending();
        const u = Mem.unitOf(need, have);
        log.warn("[dsv41-borrow] a {d}-row forward at {d} needs {f}; the headroom and every decoder pool hold {f}\n", .{ rows, seq, Mem.need(need, u), Mem.have(have, u) });
        return refuse(c, cooler_side.onEngine(config), rows, seq, rest, need, have);
    };
    var m: LiveBorrow = .{ .engine = engine, .order = order[0..n], .config = config, .read = headroom, .need = need, .wired = status.getWiredMemBytes() };
    const r = releaseUntil(&m, n, count, need) catch |e| {
        log.warn("[dsv41-borrow] a release for a {d}-row forward at {d} failed after {d:.2} GB: {s}\n", .{ rows, seq, gb(m.released), @errorName(e) });
        return e;
    };
    if (m.released > 0) {
        log.info("[dsv41-borrow] released layers {d}..{d} ({d:.2} GB) for a {d}-row forward at {d}: need {d:.2} GB, headroom {d:.2} -> {d:.2} GB, unwired in {d} ms\n", .{
            order[r.layers - 1], order[0], gb(m.released), rows, seq, gb(need), gb(before), gb(r.have), m.waited,
        });
    }
    if (r.have < need) return refuse(c, cooler_side.onEngine(config), rows, seq, rest, need, r.have);
}

/// The widest `--prefill-chunk` below the refused forward's whose forwards fit `have` to the prompt's end, `rest` tokens
/// past `seq`, beside the state those tokens grow; halving to 64, 0 when none does. An explicit width's forward takes up
/// to a whole merged tail beside it (`prefill_plan.TAIL_MERGE_MAX`), and its cap grows with the position it ends at.
pub fn chunkThatFits(c: *const Config, engine: bool, rows: u64, seq: u64, rest: u64, have: u64) u64 {
    if (rows < 2) return 0;
    var w = std.math.floorPowerOfTwo(u64, rows / 2);
    while (w >= 64) : (w /= 2) if (widestAt(c, engine, w, seq, rest) <= have) return w;
    return 0;
}

/// The dearest forward of the prompt's prefill rerun at `--prefill-chunk w` (`prefill_plan.nextChunkEnd`, with and
/// without the checkpoint stride, the tail merged as the loop merges it), each at the position it ends with the state
/// grown past `seq` to there: a tail merges only where the prompt has the rows.
fn widestAt(c: *const Config, engine: bool, w: u64, seq: u64, rest: u64) u64 {
    const plan = @import("../prefill_plan.zig");
    const loop_end: usize = @intCast(seq +| rest);
    var widest: u64 = 0;
    for ([_]usize{ 0, plan.effectiveSsmCheckpointStride(@import("../server_state.zig").ssm_checkpoint_stride, @intCast(w)) }) |stride| {
        var pos: usize = 0;
        while (pos < loop_end) {
            const end = plan.nextChunkEnd(pos, loop_end, @intCast(w), stride > 0, stride, 0, false);
            widest = @max(widest, prefillNeed(c, engine, end - pos, end) +| (end -| seq) *| c.stateBytesPerToken());
            pos = end;
        }
    }
    return widest;
}

/// What the client reads when a forward is refused mid-request: what did not fit, by how much, and what to change. `seq`
/// is the position the forward ends at.
pub fn refusalText(buf: []u8, rows: u64, seq: u64, need: u64, have: u64, chunk_fits: u64) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    writeRefusal(&w, rows, seq, need, have, chunk_fits) catch return "A prefill chunk of this prompt did not fit in GPU memory; the server log has the figures.";
    return w.buffered();
}

fn writeRefusal(w: *std.Io.Writer, rows: u64, seq: u64, need: u64, have: u64, chunk_fits: u64) std.Io.Writer.Error!void {
    const u = Mem.unitOf(need, have);
    try w.print("This prompt ran out of GPU memory partway through its prefill.\n", .{});
    try w.print("Its {d}-token chunk, tokens {d} to {d}, needs {f}, and the free memory, with the expert cache lent to the prefill, holds {f}: {f} short. The chunks before it ran.\n", .{
        rows, seq -| rows, seq, Mem.need(need, u), Mem.have(have, u), Mem.short(need, have, u),
    });
    try w.print("To serve it:\n", .{});
    if (chunk_fits > 0) try w.print("- Restart sushi with --prefill-chunk {d} (a launch flag; Settings has none): a narrower chunk needs less memory at this length.\n", .{chunk_fits});
    try w.print("- Quit other apps or unload another model on the Models page, then retry: the memory they hold comes back.", .{});
    // A first chunk has no shorter prompt the chunks before it prove.
    if (seq > rows) try w.print("\n- Shorten the prompt to {d} tokens or fewer: the chunks up to there fit.", .{seq - rows});
}

fn refuse(c: *const Config, engine: bool, rows: u64, seq: u64, rest: u64, need: u64, have: u64) error{PrefillChunkDoesNotFit} {
    var buf: [1024]u8 = undefined;
    const text = refusalText(&buf, rows, seq, need, have, chunkThatFits(c, engine, rows, seq, rest, have));
    // The console carries the client's text whole, its actions with it.
    log.err("{s}\n", .{text});
    @import("../forward_refusal.zig").record(text);
    return error.PrefillChunkDoesNotFit;
}

/// The engine's side of `releaseUntil`: a release lends a layer's pool; the wait is the kernel's unwire of everything
/// released so far.
const LiveBorrow = struct {
    engine: *stream.Engine,
    order: []const u16,
    config: *const ModelConfig,
    read: *const fn (*const ModelConfig) Headroom,
    /// The forward's need: the engine's free buffers it reuses under its cap count up to it.
    need: u64,
    wired: u64,
    released: u64 = 0,
    waited: u64 = 0,

    fn release(self: *LiveBorrow, i: usize) !u64 {
        const bytes = try self.engine.borrowLayer(self.order[i]);
        self.released += bytes;
        return bytes;
    }
    fn wait(self: *LiveBorrow) void {
        self.waited += waitForUnwire(self.wired, self.released);
    }
    fn headroom(self: *LiveBorrow) u64 {
        return self.read(self.config).forNeed(self.need);
    }
};

/// Releases the first `count` of `n` pools (the plan, counted from their bytes), waits for the kernel to unwire them,
/// then reads the headroom: freed pages count in it only once unwired. Still short, it releases one more and waits
/// again. A release that fails ends the borrow. `m` gives `release(i)`, `wait()` and `headroom()`.
fn releaseUntil(m: anytype, n: usize, count: usize, need: u64) !struct { layers: usize, have: u64 } {
    var i: usize = 0;
    while (i < @min(count, n)) : (i += 1) _ = try m.release(i);
    if (i > 0) m.wait();
    var have = m.headroom();
    // The plan counts what the pools hold; the headroom read after the unwire decides, so a short one takes the next.
    while (i < n and have < need) : (i += 1) {
        _ = try m.release(i);
        m.wait();
        have = m.headroom();
    }
    return .{ .layers = i, .have = have };
}

fn restore(engine: *stream.Engine, config: *const ModelConfig, headroom: *const fn (*const ModelConfig) Headroom, work: u64) !void {
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
        // The forward reuses free buffers for its own work; a pool read back takes memory of its own.
        if (left > 0 or headroom(config).forNeed(work) < work +| engine.lentBytes(@intCast(li))) {
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

/// A machine whose headroom grows only once the kernel has unwired what was released, as macOS does.
const LateUnwire = struct {
    pools: []const u64,
    free: u64,
    released: u64 = 0,
    unwired: u64 = 0,
    count: usize = 0,

    fn release(self: *LateUnwire, i: usize) !u64 {
        self.count += 1;
        self.released += self.pools[i];
        return self.pools[i];
    }
    fn wait(self: *LateUnwire) void {
        self.unwired = self.released;
    }
    fn headroom(self: *LateUnwire) u64 {
        return self.free + self.unwired;
    }
};

test "issue 105: a borrow whose freed pools unwire late releases only the pools its plan counts, and the forward fits once they have" {
    // The report's run: 20 decoder pools of 2.33 GB, 4.11 GB of headroom, a 2048-row chunk needing 4.35 GB.
    const pools: [20]u64 = @splat(2_326_500_000);
    var m: LateUnwire = .{ .pools = &pools, .free = 4_110_000_000 };
    const need: u64 = 4_350_000_000;
    const count = releaseCount(&pools, need, m.headroom()).?;
    try testing.expectEqual(@as(usize, 1), count);
    const r = try releaseUntil(&m, pools.len, count, need);
    try testing.expectEqual(@as(usize, 1), m.count);
    try testing.expectEqual(@as(usize, 1), r.layers);
    try testing.expect(r.have >= need);
}

test "issue 105: a borrow still short after its planned pools have unwired releases one more at a time, and a forward nothing covers is refused" {
    const pools = [_]u64{ 2, 2, 2 };
    var m: LateUnwire = .{ .pools = &pools, .free = 1 };
    // The plan counted 1 pool, but the machine gives back half of what each frees: the next pools go one by one.
    const r = try releaseUntil(&m, pools.len, 1, 4);
    try testing.expectEqual(@as(usize, 2), r.layers);
    try testing.expect(r.have >= 4);
    var short: LateUnwire = .{ .pools = &pools, .free = 0 };
    const all = try releaseUntil(&short, pools.len, 3, 7);
    try testing.expectEqual(@as(usize, 3), all.layers);
    try testing.expect(all.have < 7);
}

/// A machine whose release of pool `fail_at` fails.
const FailingRelease = struct {
    pools: []const u64,
    fail_at: usize,
    released: u64 = 0,
    count: usize = 0,

    fn release(self: *FailingRelease, i: usize) !u64 {
        self.count += 1;
        if (i == self.fail_at) return error.ImportFailed;
        self.released += self.pools[i];
        return self.pools[i];
    }
    fn wait(_: *FailingRelease) void {}
    fn headroom(self: *FailingRelease) u64 {
        return self.released;
    }
};

test "dsv41 borrow: a release that fails stops the borrow, planned or one at a time: no pool after it is released" {
    const pools = [_]u64{ 2, 2, 2 };
    var planned: FailingRelease = .{ .pools = &pools, .fail_at = 0 };
    try testing.expectError(error.ImportFailed, releaseUntil(&planned, pools.len, 2, 4));
    try testing.expectEqual(@as(usize, 1), planned.count);
    var late: FailingRelease = .{ .pools = &pools, .fail_at = 1 };
    try testing.expectError(error.ImportFailed, releaseUntil(&late, pools.len, 1, 6));
    try testing.expectEqual(@as(usize, 2), late.count);
}

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
    const c = try cfg_mod.releaseStqForTest();
    const ctx: u64 = 1 << 20;
    try testing.expectEqual(c.forwardCap(2048, ctx), prefillNeed(&c, true, 2048, ctx));
    // A last chunk that absorbed its tail is priced at its own rows.
    try testing.expect(prefillNeed(&c, true, 2048 + 511, ctx) > prefillNeed(&c, true, 2048, ctx));
    try testing.expect(c.forwardCap(verify_rows, ctx) * 3 < prefillNeed(&c, true, 512, ctx) * 2);
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

test "dsv41 borrow: at the budget-115 and 117 geometries no chunk releases a pool, and under a 2048-row chunk's cap one goes" {
    const c = try cfg_mod.releaseStqForTest();
    // Live at `--ssd-budget-gb 115` (280 slots a layer): the decoder pools a prefill may release, and the headroom there.
    const pools = [_]u64{ 2_700_000_000, 2_700_000_000, 2_760_000_000 };
    const headroom: u64 = 8_950_000_000;
    try testing.expectEqual(@as(?usize, 0), releaseCount(&pools, prefillNeed(&c, true, 31, 14766), headroom));
    try testing.expectEqual(@as(?usize, 0), releaseCount(&pools, prefillNeed(&c, true, 2048, 2048), headroom));
    try testing.expectEqual(@as(?usize, 0), releaseCount(&pools, prefillNeed(&c, true, 2048 + 511, 1 << 20), headroom));
    // At 117 the headroom is ~4.6 GB, over the 4.38 GB a 2048-row chunk holds on the engine; under it, one pool goes.
    try testing.expectEqual(@as(?usize, 0), releaseCount(&pools, prefillNeed(&c, true, 2048, 2048), 4_600_000_000));
    try testing.expectEqual(@as(?usize, 1), releaseCount(&pools, prefillNeed(&c, true, 2048, 2048), 4_000_000_000));
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
    const ops: PoolOps = .{ .set_limit = &set, .clear = &clear, .set_transient = &capTransient, .set_own = &capOwn, .restore_ram = &set };
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
            var config = try dsv41Config();
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
    var config = try dsv41Config();
    PoolProbe.reset();
    try testing.expect(!enterForward(2048, 2048, &config, &config.dsv41.?, PoolProbe.ops));
    try testing.expectEqual(@as(usize, 0), PoolProbe.n);
    try testing.expectEqual(@as(usize, 0), PoolProbe.transient_sets);
}

test "dsv41 borrow: a wide chunk on the engine gives the engine's pool the whole cap it bills, the state's growth included" {
    const Body = struct {
        fn run() anyerror!void {
            var config = try dsv41Config();
            const c = &config.dsv41.?;
            config.cooler_for_test = true;
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

test "dsv41 borrow: a joined decode forward of one request installs that request's decode cap, of more each one's with the slack once" {
    const Body = struct {
        fn run() anyerror!void {
            var config = try dsv41Config();
            const c = &config.dsv41.?;
            for ([_]bool{ true, false }) |cooler| {
                config.cooler_for_test = cooler;
                PoolProbe.reset();
                _ = enterForward(1, 4097, &config, c, PoolProbe.ops);
                const solo = .{ PoolProbe.own, PoolProbe.transient };
                PoolProbe.reset();
                _ = enterRows(&.{4097}, &config, c, PoolProbe.ops);
                try testing.expectEqual(solo, .{ PoolProbe.own, PoolProbe.transient });
                PoolProbe.reset();
                _ = enterRows(&.{ 4097, 70_000 }, &config, c, PoolProbe.ops);
                const two = c.forwardCap(1, 4097) + c.forwardCap(1, 70_000) - cfg_mod.cap_fixed_bytes;
                if (cooler) try testing.expectEqual(two, PoolProbe.own) else try testing.expectEqual(two, PoolProbe.transient - config.expertPoolBytes());
            }
        }
    };
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: a forward on the engine runs under its own cap whether or not serve installed MLX's, never under the rest a request's end left" {
    const Body = struct {
        fn run() anyerror!void {
            var config = try dsv41Config();
            const c = &config.dsv41.?;
            config.cooler_for_test = true;
            PoolProbe.own = @import("../arch/cooler.zig").restCap(&config);
            _ = enterForward(4097, 4097, &config, c, PoolProbe.ops);
            try testing.expectEqual(c.forwardCap(4097, 4097), PoolProbe.own);
            _ = enterForward(1, 4098, &config, c, PoolProbe.ops);
            try testing.expectEqual(c.forwardCap(verify_rows, 4098), PoolProbe.own);
            _ = enterRows(&.{ 4099, 9_000 }, &config, c, PoolProbe.ops);
            try testing.expectEqual(c.joinedCap(&.{ 4099, 9_000 }), PoolProbe.own);
            // MLX's caps stay serve's to move.
            try testing.expectEqual(@as(usize, 0), PoolProbe.n);
            try testing.expectEqual(@as(usize, 0), PoolProbe.transient_sets);
        }
    };
    try withPoolState(0, &Body.run);
}

test "dsv41 borrow: a wide chunk of a pack that decodes on MLX keeps the whole cap on MLX" {
    const Body = struct {
        fn run() anyerror!void {
            var config = try dsv41Config();
            const c = &config.dsv41.?;
            config.cooler_for_test = false;
            _ = enterForward(2048, 2048, &config, c, PoolProbe.ops);
            try testing.expectEqual(c.mlxForwardCap(2048, 2048), PoolProbe.transient - config.expertPoolBytes());
            try testing.expectEqual(@as(u64, 0), PoolProbe.own);
        }
    };
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: the load's warm gives each engine forward, of any width, the budget serve gives it, and nothing else" {
    const Body = struct {
        fn run() anyerror!void {
            var config = try dsv41Config();
            const c = &config.dsv41.?;
            config.cooler_for_test = true;
            defer {
                warm_budget = false;
                warm_budget_set = false;
            }
            enterWarm(1, 4096, c, true, PoolProbe.ops);
            try testing.expectEqual(@as(u64, 0), PoolProbe.own);
            warm_budget = true;
            // A wide chunk takes the one cap whole; a decode-width forward, whatever its rows, the decode cap at its position.
            try testing.expect(!enterForward(2048, 2048, &config, c, PoolProbe.ops));
            try testing.expectEqual(c.forwardCap(2048, 2048), PoolProbe.own);
            for ([_]u64{ 1, 3, verify_rows }) |rows| {
                enterWarm(rows, 4096 + rows, c, true, PoolProbe.ops);
                try testing.expectEqual(ownCap(c, true, 4096 + rows), PoolProbe.own);
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
            var config = try dsv41Config();
            const c = &config.dsv41.?;
            const pool = config.expertPoolBytes();
            _ = enterForward(1, 9_000, &config, c, PoolProbe.ops);
            try testing.expectEqual(c.forwardCap(1, 9_000) + pool, PoolProbe.transient);
            _ = enterForward(17, 225_000, &config, c, PoolProbe.ops);
            try testing.expectEqual(c.forwardCap(17, 225_000) + pool, PoolProbe.transient);
            _ = enterForward(2048, 225_000, &config, c, PoolProbe.ops);
            try testing.expectEqual(c.mlxForwardCap(2048, 225_000) + pool, PoolProbe.transient);
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
            // On the engine the bill is the engine's cap for the whole encode, and nothing of MLX's moves.
            config.cooler_for_test = true;
            enterVision(1_100_000_000, &config, PoolProbe.ops);
            try testing.expectEqual(@as(u64, 1_100_000_000), PoolProbe.own);
            try testing.expect(visionBilled());
            leaveVision(&config, PoolProbe.ops);
            try testing.expect(!visionBilled());
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
            var config = try dsv41Config();
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
            var config = try dsv41Config();
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

test "dsv41 borrow: with decode on the engine of our own every forward installs its cap there, and a request's end leaves it to the shared seam" {
    const Body = struct {
        fn run() anyerror!void {
            var config = try dsv41Config();
            const c = &config.dsv41.?;
            config.cooler_for_test = false;
            _ = enterForward(1, 9_000, &config, c, PoolProbe.ops);
            try testing.expectEqual(@as(u64, 0), PoolProbe.own);
            config.cooler_for_test = true;
            _ = enterForward(1, 9_000, &config, c, PoolProbe.ops);
            try testing.expectEqual(c.forwardCap(verify_rows, 9_000), PoolProbe.own);
            // A prefill chunk takes the cap less MLX's share.
            _ = enterForward(2048, 225_000, &config, c, PoolProbe.ops);
            try testing.expectEqual(c.forwardCap(2048, 225_000), PoolProbe.own);
            const sets = PoolProbe.own_sets;
            _ = endRequest(&config, PoolProbe.ops);
            try testing.expectEqual(sets, PoolProbe.own_sets);
            try testing.expectEqual(c.forwardCap(verify_rows, 1), @import("../arch/cooler.zig").restCap(&config));
            // The unload trims the engine's pool and leaves it without a cap for the next load.
            _ = unloadModel(&config, PoolProbe.ops);
            try testing.expectEqual(@as(u64, std.math.maxInt(u64)), PoolProbe.own);
        }
    };
    PoolProbe.reset();
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: decode runs on the engine for every pack, the release's own as an STQ one, unless the user asks for MLX by name" {
    var config = try dsv41Config();
    // A config with no trellis decode is the release's own layout.
    try testing.expect(config.dsv41.?.expert_dec == null and config.dsv41.?.trunk_dec == null);
    config.cooler_for_test = true;
    try testing.expect(cooler_side.sideOf(&config) == .engine);
    config.cooler_ask = .flag;
    try testing.expect(cooler_side.sideOf(&config) == .mlx_by_flag);
    config.cooler_ask = .setting;
    try testing.expect(cooler_side.sideOf(&config) == .mlx_by_setting);
}

test "dsv41 borrow: the load line names the side and why, with Sushi Cooler as the engine's name" {
    const what = "[dsv41] decode forwards (one row, DSpark draft and verify rows)";
    var buf: [256]u8 = undefined;
    var config = try dsv41Config();
    config.cooler_for_test = true;
    try testing.expectEqualStrings(what ++ ": on Sushi Cooler, the engine of our own", cooler_side.line(&buf, what, &config));
    config.cooler_ask = .flag;
    try testing.expectEqualStrings(what ++ ": on MLX because --ds41-mlx-engine was given (Sushi Cooler is off)", cooler_side.line(&buf, what, &config));
    config.cooler_ask = .setting;
    try testing.expectEqualStrings(what ++ ": on MLX because model-settings.json ds41_cooler is off (Sushi Cooler is off)", cooler_side.line(&buf, what, &config));
}

test "dsv41 borrow: beside a Cooler model, a model on MLX has its transient cap lifted at its request's end" {
    try mlx.requireGpu();
    const Body = struct {
        fn run() anyerror!void {
            var config = try dsv41Config();
            const c = &config.dsv41.?;
            const was_enabled = mlx.engine.enabled;
            mlx.engine.enabled = true;
            defer @import("../test_support.zig").restoreEngine(was_enabled);
            defer mlx.engine.budget(std.math.maxInt(u64)) catch {};
            mlx.engine.hold();
            defer @import("../arch/handle.zig").engineHoldEnd();
            {
                const turn = mlx.engine.turnTo(false);
                defer turn.end();
                try testing.expectEqual(@as(usize, 0), mlx.setTransientLimit(0));
                config.cooler_for_test = false;
                _ = enterForward(1, 9_000, &config, c, mlx_ops);
            }
            {
                const turn = mlx.engine.turnTo(true);
                defer turn.end();
                config.cooler_for_test = true;
                _ = endRequest(&config, mlx_ops);
            }
            const turn = mlx.engine.turnTo(false);
            defer turn.end();
            config.cooler_for_test = false;
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
            var config = try dsv41Config();
            const c = &config.dsv41.?;
            const was_enabled = mlx.engine.enabled;
            mlx.engine.enabled = true;
            defer @import("../test_support.zig").restoreEngine(was_enabled);
            defer mlx.engine.budget(std.math.maxInt(u64)) catch {};
            var orig: usize = 0;
            try mlx.check(mlx.mlx_set_cache_limit(&orig, 1 << 40));
            defer {
                var unused: usize = 0;
                _ = mlx.mlx_set_cache_limit(&unused, orig);
                _ = mlx.setTransientLimit(0);
            }
            mlx.engine.hold();
            defer @import("../arch/handle.zig").engineHoldEnd();
            {
                const turn = mlx.engine.turnTo(false);
                defer turn.end();
                config.cooler_for_test = false;
                try testing.expect(enterForward(2048, 2048, &config, c, mlx_ops));
            }
            {
                const turn = mlx.engine.turnTo(true);
                defer turn.end();
                config.cooler_for_test = true;
                _ = enterForward(1, 4097, &config, c, mlx_ops);
            }
            const turn = mlx.engine.turnTo(false);
            defer turn.end();
            config.cooler_for_test = false;
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
            var config = try dsv41Config();
            const was_enabled = mlx.engine.enabled;
            mlx.engine.enabled = true;
            defer @import("../test_support.zig").restoreEngine(was_enabled);
            defer mlx.engine.budget(std.math.maxInt(u64)) catch {};
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
            config.cooler_for_test = true;
            unloadModelMlx(&config);
            @import("../arch/handle.zig").engineHoldEnd();
            var now: usize = 0;
            try mlx.check(mlx.mlx_set_cache_limit(&now, 3 << 30));
            try testing.expectEqual(@as(usize, 3 << 30), now);
            config.cooler_for_test = false;
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
            var config = try dsv41Config();
            const c = &config.dsv41.?;
            config.cooler_for_test = false;
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
            var config = try dsv41Config();
            config.cooler_for_test = false;
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
    var config = try dsv41Config();
    const c = &config.dsv41.?;
    config.cooler_for_test = false;
    try testing.expectEqual(cfg_mod.dsv41_pool_bytes, config.expertPoolBytes());
    config.cooler_for_test = true;
    try testing.expectEqual(c.forwardCap(verify_rows, 1), config.expertPoolBytes());
    try testing.expect(config.expertPoolBytes() < cfg_mod.dsv41_pool_bytes);
    // What a request's end leaves the engine is what the ledger billed.
    try testing.expectEqual(config.expertPoolBytes(), @import("../arch/cooler.zig").restCap(&config));
}

test "the opt-out flag reaches a DeepSeek load's decode side through the parse and the settings stamp, and outranks a test's side" {
    const dir = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const model_load = @import("../model_load.zig");
    const model_mod = @import("../model.zig");
    const saved = model_mod.cooler_off;
    defer model_mod.cooler_off = saved;
    {
        model_mod.cooler_off = .empty;
        var cfg = try model_mod.parseConfig(io, a, dir);
        defer cfg.deinit(a);
        model_load.applyModelSettings(&cfg, .{});
        try testing.expect(cooler_side.sideOf(&cfg) == .mlx_in_test);
        cfg.cooler_for_test = true;
        try testing.expect(cooler_side.onEngine(&cfg));
    }
    {
        model_mod.cooler_off = .empty;
        model_mod.cooler_off.insert(.dsv41);
        var cfg = try model_mod.parseConfig(io, a, dir);
        defer cfg.deinit(a);
        model_load.applyModelSettings(&cfg, .{ .ds41_cooler = true });
        cfg.cooler_for_test = true;
        try testing.expect(cooler_side.sideOf(&cfg) == .mlx_by_flag);
    }
    {
        model_mod.cooler_off = .empty;
        var cfg = try model_mod.parseConfig(io, a, dir);
        defer cfg.deinit(a);
        model_load.applyModelSettings(&cfg, .{ .ds41_cooler = false });
        try testing.expect(cooler_side.sideOf(&cfg) == .mlx_by_setting);
    }
}

test "dsv41 borrow: a load's served warm runs its forwards under the warm budget, which its end lifts" {
    const arch_state = @import("../arch/arch_state.zig");
    defer {
        warm_budget = false;
        warm_budget_set = false;
    }
    var s: arch_state.State = .{};
    arch_state.servedWarmBegin(&s);
    try testing.expect(warm_budget);
    arch_state.servedWarmEnd(&s);
    try testing.expect(!warm_budget);
}


/// A load's fit check that refuses it, noting the engine's free cap it found: the load's plan has capped the pool by then.
const RefusingFit = struct {
    var free_cap_seen: ?u64 = null;
    fn fit(_: *const ModelConfig, _: u64) ?@import("../bills.zig").ExpertCacheRefusal {
        free_cap_seen = @import("../tensor/core.zig").free_cap;
        return .{ .err = error.ExpertCacheDoesNotFit, .refusal = .{ .needed = 1, .avail = .{ .bytes = 0, .free = 0 } }, .readings = .{} };
    }
};

test "dsv41 borrow: a DeepSeek load on MLX that fails after its plan capped the pool gives both caps back, and so does an unload (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const Body = struct {
        fn run() anyerror!void {
            const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
            const model_mod = @import("../model.zig");
            const model_load = @import("../model_load.zig");
            const core = @import("../tensor/core.zig");
            const a = testing.allocator;
            const io = std.Io.Threaded.global_single_threaded.io();
            const was_enabled = mlx.engine.enabled;
            mlx.engine.enabled = true;
            defer @import("../test_support.zig").restoreEngine(was_enabled);
            const free_was = core.free_cap;
            defer core.free_cap = free_was;
            const ram: usize = 8 << 30;
            var orig: usize = 0;
            try mlx.check(mlx.mlx_set_cache_limit(&orig, ram));
            defer {
                var unused: usize = 0;
                _ = mlx.mlx_set_cache_limit(&unused, orig);
            }
            core.free_cap = ram;

            // The user's opt-out: the load is MLX's, whose plan caps MLX's pool and the engine's free buffers.
            const off = model_mod.cooler_off;
            defer model_mod.cooler_off = off;
            model_mod.cooler_off = .full;
            var config = try model_mod.parseConfig(io, a, dir);
            defer config.deinit(a);
            var sch: @import("../scheduler.zig").Scheduler = undefined;
            sch.io = io;
            sch.allocator = a;
            sch.ctx_size_flag = 0;
            RefusingFit.free_cap_seen = null;
            const params: model_load.LoadParams = .{ .registry = undefined, .entry = undefined, .config = &config, .tok = undefined, .chat_config = undefined, .model_dir = dir, .ssd_budget_bytes = 16 << 30, .expert_cache_fit_resolver = &RefusingFit.fit };
            try testing.expectError(error.ExpertCacheDoesNotFit, model_load.doLoadOnInferenceThread(&sch, params));
            try testing.expect(RefusingFit.free_cap_seen.? < ram);
            var now: usize = 0;
            try mlx.check(mlx.mlx_set_cache_limit(&now, ram));
            try testing.expectEqual(ram, now);
            try testing.expectEqual(@as(?u64, ram), core.free_cap);
            try testing.expectEqual(@as(u32, 0), ram_cappers);

            // A load that serves and is unloaded gives both back too.
            try testing.expect(@import("../gpu_mem.zig").capMlxPool(1 << 30));
            unloadModelMlx(&config);
            try mlx.check(mlx.mlx_set_cache_limit(&now, ram));
            try testing.expectEqual(ram, now);
            try testing.expectEqual(@as(?u64, ram), core.free_cap);
        }
    };
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: a forward refused mid-prefill names its chunk, the shortfall and the narrower chunk that fits, never the admission refusal" {
    const c = try cfg_mod.parseText(cfg_mod.release_config);
    const gib: u64 = 1 << 30;
    const rows: u64 = 2048;
    const seq: u64 = 6144;
    const need = prefillNeed(&c, true, rows, seq);
    // What is free holds a 1024-row forward at the prompt's end.
    const have = prefillNeed(&c, true, 1024 + 511, seq) + (1 << 20);
    try testing.expect(have < need);
    try testing.expectEqual(@as(u64, 1024), chunkThatFits(&c, true, rows, seq, 0, have));
    // The refused forward is the prompt's last: at 1024 its 2048 rows run as two whole chunks, no tail to merge.
    try testing.expectEqual(@as(u64, 512), chunkThatFits(&c, true, rows, seq, 0, prefillNeed(&c, true, 1024, seq) - 1));
    try testing.expectEqual(@as(u64, 0), chunkThatFits(&c, true, rows, seq, 0, 0));
    var buf: [1024]u8 = undefined;
    const text = refusalText(&buf, rows, seq, 5 * gib, 4 * gib + gib / 2, 1024);
    for ([_][]const u8{
        "This prompt ran out of GPU memory partway through its prefill.\n",
        "Its 2048-token chunk, tokens 4096 to 6144, needs 5.0 GB",
        "holds 4.5 GB: 0.5 GB short. The chunks before it ran.\n",
        "To serve it:\n- Restart sushi with --prefill-chunk 1024 (a launch flag; Settings has none)",
        "- Quit other apps or unload another model on the Models page, then retry",
        "- Shorten the prompt to 4096 tokens or fewer",
    }) |part| {
        errdefer std.debug.print("missing \"{s}\" in:\n{s}\n", .{ part, text });
        try testing.expect(std.mem.indexOf(u8, text, part) != null);
    }
    try testing.expect(std.mem.indexOf(u8, refusalText(&buf, rows, seq, 5 * gib, 0, 0), "--prefill-chunk") == null);
}

test "dsv41 borrow: a mid-prefill refusal a byte short shows its need above what the borrow holds, in one unit" {
    var buf: [1024]u8 = undefined;
    const need: u64 = 290_000_000;
    const text = refusalText(&buf, 569, 569, need, need - 1, 0);
    errdefer std.debug.print("{s}\n", .{text});
    try testing.expect(std.mem.indexOf(u8, text, "needs 277 MB, and ") != null);
    try testing.expect(std.mem.indexOf(u8, text, " holds 276 MB: 1 MB short.") != null);
}

test "dsv41 borrow: a first chunk refused mid-request names no shorter prompt (it would be 0 tokens)" {
    const gib: u64 = 1 << 30;
    var buf: [1024]u8 = undefined;
    const text = refusalText(&buf, 2048, 2048, 5 * gib, 4 * gib, 1024);
    errdefer std.debug.print("{s}\n", .{text});
    try testing.expect(std.mem.indexOf(u8, text, "Shorten the prompt") == null);
    try testing.expect(std.mem.endsWith(u8, text, "the memory they hold comes back."));
}

test "dsv41 borrow: the --prefill-chunk a mid-prefill refusal names fits the prompt's later chunks, not only the one it refused" {
    const c = try cfg_mod.parseText(cfg_mod.release_config);
    const tail: u64 = @import("../prefill_plan.zig").TAIL_MERGE_MAX - 1;
    const MiB: u64 = 1 << 20;
    // A 32k-token prompt refused at its 2048-row chunk ending at 8192; what is free then holds a 1024 chunk there.
    const rows: u64 = 2048;
    const seq: u64 = 8192;
    const prompt: u64 = 32768;
    const have = prefillNeed(&c, true, 1024 + tail, seq) + MiB;
    const w = chunkThatFits(&c, true, rows, seq, prompt - seq, have);
    try testing.expect(w > 0);
    // At --prefill-chunk w the same request's last forward ends at the prompt's end; the state between grows too.
    const growth = (prompt - seq) * c.stateBytesPerToken();
    // Its last forward: the prompt's chunks at `w`, the tail merged as the loop merges it.
    var pos: usize = 0;
    var last: u64 = 0;
    while (pos < prompt) {
        const end = @import("../prefill_plan.zig").nextChunkEnd(pos, prompt, @intCast(w), false, 0, 0, false);
        last = end - pos;
        pos = end;
    }
    errdefer std.debug.print("named --prefill-chunk {d}: its last forward of {d} rows at {d} needs {d} MB beside {d} MB of state growth, {d} MB free\n", .{ w, last, prompt, prefillNeed(&c, true, last, prompt) / MiB, growth / MiB, have / MiB });
    try testing.expect(prefillNeed(&c, true, last, prompt) + growth <= have);
}

test "dsv41 borrow: a short prompt refused mid-prefill is offered the narrower --prefill-chunk its own remaining rows fit" {
    const c = try cfg_mod.parseText(cfg_mod.release_config);
    // An 1,100-token prompt's whole forward refused; at --prefill-chunk 512 its prefill runs 512 rows, then the 588
    // left as one forward (its 76-row tail merged): never the 1,023 a full tail past 512 would make.
    const rows: u64 = 1100;
    const seq: u64 = 1100;
    const have = prefillNeed(&c, true, 588, seq);
    try testing.expect(have < prefillNeed(&c, true, rows, seq));
    try testing.expectEqual(@as(u64, 512), chunkThatFits(&c, true, rows, seq, 0, have));
}
