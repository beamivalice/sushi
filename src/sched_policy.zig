//! The scheduler's decisions, apart from the slots and arrays they act on: which slots join a tick and in
//! which bucket, how a batched group is capped, the order pending requests admit in, the slot's end-state
//! transitions, and the interleave and idle-park arithmetic. Every function reads flags, counts and lengths
//! only, so the host tier runs them (`server-lifecycle.md#scheduler-policy`).

const std = @import("std");
const testing = std.testing;

/// Slots one batched forward can carry; the tail decodes serial this tick.
pub const MAX_BATCH_GROUP = 32;

pub const SlotState = enum { pending_prefill, decoding, finished, errored };

/// The state `markFinished` moves a slot to; null when it already finished (idempotent). An earlier error does not block it.
pub fn finishState(finished: bool) ?SlotState {
    return if (finished) null else .finished;
}

/// `markError` latches only on a slot that has not ended, whichever way.
pub fn errorLands(errored: bool, finished: bool) bool {
    return !(errored or finished);
}

/// A slot the tick loop still drives.
pub fn joinsTick(cancelled: bool, finished: bool, errored: bool) bool {
    return !(cancelled or finished or errored);
}

/// A cancel nothing else has ended yet.
pub fn cancelOpen(cancelled: bool, finished: bool, errored: bool) bool {
    return cancelled and !finished and !errored;
}

/// Until the inference thread has released it, a slot that has not ended keeps blocking exclusive
/// admission, cancelled or not.
pub fn holdsAdmission(finished: bool, errored: bool) bool {
    return !finished and !errored;
}

pub const BatchVerdict = enum {
    ok,
    spec_active,
    head_release_pending,
    grammar,
    forced_call,
    logprobs,
    penalty,
    arch,
    pad_waste,
    row_cap,
    /// The model owns its decode state (`slotExclusiveDecode`): requests queue, never co-decode.
    exclusive,
};

/// What a slot's decode tick reads to decide whether it rides the batched kernel.
pub const BatchFacts = struct {
    /// Its next tick runs the regular (non-speculative) path: read from dispatch (`specTickMode`), never from
    /// the armed `enable_drafter` / `enable_mtp` flags.
    ticks_regular: bool,
    /// A module-head release is armed but not landed, so the slot still holds the head.
    release_pending: bool,
    grammar: bool,
    /// A forced call decides its own successor, and the batched tick samples every slot's.
    forced_call_pending: bool,
    logprobs: bool,
    penalty: bool,
    /// The transformer's batching answer, never a config compare; null when the slot has no transformer.
    model_batches: ?bool,
};

/// Why a slot does or does not ride the batched kernel, first match wins.
pub fn batchVerdictOf(f: BatchFacts) BatchVerdict {
    if (!f.ticks_regular) return .spec_active;
    if (f.release_pending) return .head_release_pending;
    if (f.grammar) return .grammar;
    if (f.forced_call_pending) return .forced_call;
    if (f.logprobs) return .logprobs;
    if (f.penalty) return .penalty;
    return if (f.model_batches orelse false) .ok else .arch;
}

/// How a slot is ticked while it speculates.
pub const TickClass = enum {
    /// Runs its round alone.
    none,
    /// Its MTP round may join the grouped verify, or drop to a plain batched tick when crowded.
    mtp_group,
    /// Decodes as a plain row of the grouped tick when its model has company.
    rows_company,
};

pub const TickBucket = enum { batch, mtp, rows, serial };

/// How many slots each bucket holds so far this tick.
pub const BucketFill = struct { batch: usize = 0, mtp: usize = 0, rows: usize = 0 };

/// Where a slot goes this tick. A full bucket sends the slot to the serial path.
pub fn bucketOf(why: BatchVerdict, cls: TickClass, fill: BucketFill) TickBucket {
    if (why == .ok and fill.batch < MAX_BATCH_GROUP) return .batch;
    if (why == .spec_active and cls == .mtp_group and fill.mtp < MAX_BATCH_GROUP) return .mtp;
    if (why == .spec_active and cls == .rows_company and fill.rows < MAX_BATCH_GROUP) return .rows;
    return .serial;
}

/// A crowd of MTP slots on one model beats its sub-grouped verify rounds with one plain batched tick when
/// it reaches the threshold and still fits the batch.
pub fn crowdFits(run: usize, threshold: usize, batch_n: usize) bool {
    return run >= threshold and batch_n + run <= MAX_BATCH_GROUP;
}

/// How many of a group's rows one batched forward takes: the arch's bound, where it declares one.
pub fn rowsKept(len: usize, row_cap: ?usize) usize {
    return @min(len, row_cap orelse MAX_BATCH_GROUP);
}

/// Items sorted by `model` pointer, so one model's slots run contiguously.
pub fn sortByModel(comptime T: type, items: []T) void {
    std.sort.pdq(T, items, {}, struct {
        fn lt(_: void, a: T, b: T) bool {
            return @intFromPtr(a.model) < @intFromPtr(b.model);
        }
    }.lt);
}

/// The end of the run that starts at `start` and shares its `model` (items sorted by it).
pub fn modelRunEnd(comptime T: type, items: []const T, start: usize) usize {
    var end = start + 1;
    while (end < items.len and items[end].model == items[start].model) end += 1;
    return end;
}

/// Stable ascending insertion sort of `items` by their parallel `keys`. Stable, since off a long-context trunk
/// the keys all tie and the order decides which slots fall out of the group.
pub fn sortByKeyStable(comptime T: type, items: []T, keys: []u32) void {
    var i: usize = 1;
    while (i < items.len) : (i += 1) {
        const item_i = items[i];
        const key_i = keys[i];
        var j = i;
        while (j > 0 and keys[j - 1] > key_i) : (j -= 1) {
            items[j] = items[j - 1];
            keys[j] = keys[j - 1];
        }
        items[j] = item_i;
        keys[j] = key_i;
    }
}

/// Batched decode pads every slot's KV to the group's LONGEST (`padAndStackBatchedKV`),
/// so the tensor it builds is `N x kv_max`, not `sum(kv_len)`. A group mixing one
/// 100k-token stream with three 1k ones therefore materializes ~100x the bytes the
/// short slots need on a long-context trunk, and NOTHING bills it: it is a per-tick
/// transient, invisible to `prefillTransientReserve` and to the load-time gate. An
/// uncatchable Metal OOM is the failure mode.
///
/// So the group is capped by PADDING WASTE, which is the quantity that actually hurts,
/// rather than by a length ratio: keep the largest prefix of the ascending-sorted
/// lengths whose padded tensor stays within `MAX_PAD_WASTE` of its useful bytes. The
/// slots that fall out are the LONGEST ones — the ones dominating kv_max — and they
/// decode serially this tick, so every slot still advances.
///
/// Returns how many of `kv_lens_asc` may batch together (0 or 1 = nobody batches).
/// The lengths are the arch's true attention KV length (`batchKvLenOf`), pre-tick counts,
/// so this is a heuristic bar on the padding the forward will build, not an accounting identity.
/// The padded tensor may be at most this multiple of the bytes the group
/// actually needs. It must stay BELOW 2.0 or a two-slot group can never be
/// vetoed: one 1-token slot beside one 200k slot pads to exactly 2x, which is
/// the worst case for N=2 and the pathological pair this cap exists for.
pub const MAX_PAD_WASTE: f64 = 1.5;

pub fn batchedKvKeepCount(kv_lens_asc: []const u32) usize {
    if (kv_lens_asc.len < 2) return 0;
    var k: usize = kv_lens_asc.len;
    while (k >= 2) : (k -= 1) {
        var sum: u64 = 0;
        for (kv_lens_asc[0..k]) |l| sum += l;
        if (sum == 0) return k; // nothing prefilled yet: no padding to waste
        const padded: f64 = @floatFromInt(@as(u64, k) * kv_lens_asc[k - 1]);
        if (padded <= MAX_PAD_WASTE * @as(f64, @floatFromInt(sum))) return k;
    }
    return 0;
}

/// The padding waste the whole group would pay; reported by the cap's log.
pub fn batchedPadWaste(kv_lens_asc: []const u32) f64 {
    var sum: u64 = 0;
    for (kv_lens_asc) |l| sum += l;
    if (sum == 0 or kv_lens_asc.len == 0) return 1.0;
    const padded: f64 = @floatFromInt(@as(u64, kv_lens_asc.len) * kv_lens_asc[kv_lens_asc.len - 1]);
    return padded / @as(f64, @floatFromInt(sum));
}

/// A group whose longest billed slot is at most this long pads a small tensor, and splitting it sends a slot
/// through a whole serial forward instead: under a gather arm every long slot bills at the indexer
/// budget, so a short sub-agent beside long streams reads as a 2x pad.
pub const PAD_FREE_KV: u32 = 4096;

/// The floor only engages once a slot holds this much context: below it the serial forward it saves is cheap.
pub const PAD_FREE_MIN_CTX: u32 = 131072;

/// `batchedKvKeepCount` for a group whose longest TRUE context is `ctx_max`.
pub fn groupKeepCount(kv_lens_asc: []const u32, ctx_max: u32) usize {
    if (kv_lens_asc.len >= 2 and ctx_max >= PAD_FREE_MIN_CTX and kv_lens_asc[kv_lens_asc.len - 1] <= PAD_FREE_KV) return kv_lens_asc.len;
    return batchedKvKeepCount(kv_lens_asc);
}

/// Pure core of `slotExclusiveDecode`. `model_owns_state` wins outright and is never
/// released; the head clause is a module-owned head, released for good once the
/// adaptive switch moved the slot to serial (`Generator.mtpModuleHeadReleased`).
pub fn headExclusiveFor(
    model_owns_state: bool,
    head_module_owned: bool,
    slot_enable_mtp: bool,
    head_released: bool,
) bool {
    if (model_owns_state) return true;
    if (!head_module_owned or !slot_enable_mtp) return false;
    return !head_released;
}

/// A slot whose drafter outranks the head (`specTickMode`) never touches the module-owned head.
pub fn slotDrivesHead(enable_mtp: bool, enable_drafter: bool, has_dflash: bool) bool {
    return enable_mtp and !(enable_drafter and has_dflash);
}

/// One pending-drain candidate (or live decoding slot), reduced to what
/// admission needs: an opaque model identity + the exclusive-decode bit.
pub const AdmitCand = struct { model: usize, exclusive: bool };

/// FIFO admission for one drain tick. An EXCLUSIVE candidate admits only if
/// no EXCLUSIVE `active` slot (live decoding) holds its model and no earlier
/// admitted exclusive candidate claimed it this tick
/// (a same-tick sibling is not in `decoding` yet — the claim covers the
/// window). Non-exclusive candidates always admit — no head-of-line
/// blocking behind a held exclusive request. Writes admitted candidate
/// indices to `out` (queue order preserved), returns the count. Held
/// candidates stay where they are and retry next tick; no release
/// bookkeeping exists — the busy signal IS presence in `decoding`.
pub fn admitPendingTick(cands: []const AdmitCand, active: []const AdmitCand, out: []usize) usize {
    var n: usize = 0;
    outer: for (cands, 0..) |c, i| {
        if (n >= out.len) break;
        if (c.exclusive) {
            for (active) |a| if (a.exclusive and a.model == c.model) continue :outer;
            for (out[0..n]) |j| {
                if (cands[j].exclusive and cands[j].model == c.model) continue :outer;
            }
        }
        out[n] = i;
        n += 1;
    }
    return n;
}

/// What a slot's request wishes for speculation, and what the generator armed, decide which
/// modes `runPrefill` arms in its Generator init options.
pub const SpecInitWiring = struct {
    use_mtp: bool,
    use_dflash: bool,
};

/// Per-site spec gating for one slot, as a pure function.
///
/// Speculative decode must be able to ROLL BACK a rejected tail, and the shell
/// rolls back what it owns: the slot's KVCache and ssm_entries. A MODULE-OWNED
/// arch uses neither — by the time a verify forward returns, its own state has
/// already absorbed every draft token, and the shell's snapshot/truncate run
/// over an empty entries array. So every shell-driven spec mode is off there.
pub fn specInitWiring(
    owns_module_state: bool,
    module_spec_rollback: bool,
    enable_mtp: bool,
    has_mtp: bool,
    enable_drafter: bool,
    has_dflash: bool,
) SpecInitWiring {
    if (owns_module_state) {
        // A module-owned arch that CAN rewind its state across a verify runs
        // a DFlash2 drafter or its own MTP head like any other target. Without
        // rollback every spec mode stays off.
        const use_dflash = module_spec_rollback and enable_drafter and has_dflash;
        return .{
            .use_mtp = !use_dflash and module_spec_rollback and enable_mtp and has_mtp,
            .use_dflash = use_dflash,
        };
    }
    // enable_drafter is the request-level "assistant sidecar" switch for BOTH
    // sidecar kinds; the loader guarantees at most one of drafter/dflash is
    // loaded per model. Priority: dflash > MTP — a
    // loaded DFlash sidecar is an explicit choice (`--drafter` or the pack's
    // own `drafter/`) while the MTP head ships with the checkpoint;
    // `--no-drafter` (or `enable_drafter:false`) hands the round back to MTP.
    const use_dflash = enable_drafter and has_dflash;
    const use_mtp = !use_dflash and enable_mtp and has_mtp;
    return .{
        .use_mtp = use_mtp,
        .use_dflash = use_dflash,
    };
}

/// Which spec mode a decode tick drives for a slot.
pub const SpecTickMode = enum { mtp, dflash, regular };

/// Pure decode-tick dispatch decision. The slot flags carry the REQUEST's
/// wish; the generator-side values carry what `Generator.initWithOptions`
/// actually armed. Every arm must require BOTH.
pub fn specTickMode(
    slot_enable_mtp: bool,
    gen_has_mtp: bool,
    slot_enable_drafter: bool,
    gen_has_dflash: bool,
) SpecTickMode {
    if (slot_enable_drafter and gen_has_dflash) return .dflash;
    if (slot_enable_mtp and gen_has_mtp) return .mtp;
    return .regular;
}

pub const PREFILL_DECODE_SHARE_MAX: f32 = 0.9;
pub fn parseDecodeShare(text: []const u8) error{InvalidDecodeShare}!f32 {
    const v = std.fmt.parseFloat(f32, text) catch return error.InvalidDecodeShare;
    if (!(v >= 0 and v <= PREFILL_DECODE_SHARE_MAX)) return error.InvalidDecodeShare;
    return v;
}
pub fn decodeShareBudgetNs(chunk_ns: u64, share: f32) u64 {
    if (std.math.isNan(share) or share <= 0) return 0;
    const fraction: f64 = @min(share, PREFILL_DECODE_SHARE_MAX);
    const budget = @as(f64, @floatFromInt(chunk_ns)) * fraction / (1 - fraction);
    if (budget >= @as(f64, @floatFromInt(std.math.maxInt(u64)))) return std.math.maxInt(u64);
    return @intFromFloat(budget);
}
pub const OwedTicks = struct { ticks: u32, spent_ns: u64 };
pub fn runOwedDecodeTicks(share: f32, chunk_ns: u64, first_ns: u64, ctx: *anyopaque, tick: *const fn (*anyopaque) u64) OwedTicks {
    var result = OwedTicks{ .ticks = 1, .spent_ns = first_ns };
    if (first_ns == 0) return result;
    const budget = decodeShareBudgetNs(chunk_ns, share);
    while (result.spent_ns < budget) {
        const ns = tick(ctx);
        if (ns == 0) break;
        result.ticks +|= 1;
        result.spent_ns +|= ns;
    }
    return result;
}

/// Shorter than the idle time after which the state is gone.
pub const GPU_WARM_TICK_NS: u64 = 500 * std.time.ns_per_ms;

/// PURE: whether an idle thread ticks now. `idle_ns` is null until the thread has run a
/// prefill or decode; a thread with work never ticks.
pub fn gpuWarmTickDue(has_work: bool, idle_ns: ?u64, window_ns: u64) bool {
    const idle = idle_ns orelse return false;
    return !has_work and idle < window_ns;
}

/// PURE: how long an idle thread parks before its next tick, or null to park until work arrives.
pub fn gpuWarmParkNs(idle_ns: ?u64, window_ns: u64) ?u64 {
    const idle = idle_ns orelse return null;
    if (idle >= window_ns) return null;
    return @min(GPU_WARM_TICK_NS, window_ns - idle);
}

/// How often the idle or busy loop looks for room to re-arm a declined SSD tier, and how soon it looks again
/// when a request held the model.
pub const REARM_PROBE_NS: u64 = 60 * std.time.ns_per_s;
pub const REARM_BUSY_RETRY_NS: u64 = 5 * std.time.ns_per_s;

/// PURE: whether the re-arm check runs now. Only a declined tier is looked after, once per `wait_ns`; a settings
/// save (`forced`) skips the wait.
pub fn rearmDue(forced: bool, declined: u32, since_ns: u64, wait_ns: u64) bool {
    return forced or (declined > 0 and since_ns >= wait_ns);
}

/// PURE: how long until the next check after one that found a model a request still held.
pub fn rearmWaitNs(busy: bool) u64 {
    return if (busy) REARM_BUSY_RETRY_NS else REARM_PROBE_NS;
}

pub const GpuWarmWindow = enum { keep, restart, close };

/// PURE: the window as the thread is about to park. The last pass's prefill or decode restarts it;
/// an unload closes it, since no model may be left to keep warm.
pub fn gpuWarmBeforePark(worked: bool, unloaded: bool) GpuWarmWindow {
    if (unloaded) return .close;
    return if (worked) .restart else .keep;
}

/// Poll interval for the idle-eviction sweep, given the configured window.
///
/// A quarter of the window, so a model is evicted within ~1.25x the configured
/// idle time rather than up to 2x it. Floored at a second so `--idle-evict-secs 1`
/// does not spin, capped at 30s so a long window still costs ~nothing.
pub fn idleEvictTickMs(window_ms: i64) i64 {
    return @max(1000, @min(@divTrunc(window_ms, 4), 30_000));
}

// Tests.

test "a slot ends once: a second finish, and an error after any end, are ignored" {
    try testing.expectEqual(@as(?SlotState, .finished), finishState(false));
    try testing.expectEqual(@as(?SlotState, null), finishState(true));
    try testing.expect(errorLands(false, false));
    try testing.expect(!errorLands(true, false));
    try testing.expect(!errorLands(false, true));
    try testing.expect(!errorLands(true, true));
}

test "tick membership: a cancelled, finished or errored slot leaves the tick, and a cancel is open only until another end" {
    try testing.expect(joinsTick(false, false, false));
    try testing.expect(!joinsTick(true, false, false));
    try testing.expect(!joinsTick(false, true, false));
    try testing.expect(!joinsTick(false, false, true));
    try testing.expect(cancelOpen(true, false, false));
    try testing.expect(!cancelOpen(false, false, false));
    try testing.expect(!cancelOpen(true, true, false));
    try testing.expect(!cancelOpen(true, false, true));
    // A cancelled slot keeps blocking exclusive admission until the inference thread has released it.
    try testing.expect(holdsAdmission(false, false));
    try testing.expect(!holdsAdmission(true, false));
    try testing.expect(!holdsAdmission(false, true));
}

test "batchVerdictOf: the first reason in the gate's order wins, and a model with no answer does not batch" {
    const base = BatchFacts{
        .ticks_regular = true,
        .release_pending = false,
        .grammar = false,
        .forced_call_pending = false,
        .logprobs = false,
        .penalty = false,
        .model_batches = true,
    };
    try testing.expectEqual(BatchVerdict.ok, batchVerdictOf(base));

    var f = base;
    f.model_batches = false;
    try testing.expectEqual(BatchVerdict.arch, batchVerdictOf(f));
    f.model_batches = null;
    try testing.expectEqual(BatchVerdict.arch, batchVerdictOf(f));

    // Each reason alone, then stacked on all later ones: the earlier arm names the slot.
    const order = [_]struct { field: []const u8, why: BatchVerdict }{
        .{ .field = "release_pending", .why = .head_release_pending },
        .{ .field = "grammar", .why = .grammar },
        .{ .field = "forced_call_pending", .why = .forced_call },
        .{ .field = "logprobs", .why = .logprobs },
        .{ .field = "penalty", .why = .penalty },
    };
    var all = base;
    inline for (order) |o| @field(all, o.field) = true;
    inline for (order, 0..) |o, i| {
        var one = base;
        @field(one, o.field) = true;
        try testing.expectEqual(o.why, batchVerdictOf(one));
        try testing.expectEqual(o.why, batchVerdictOf(blk: {
            var rest = base;
            inline for (order[i..]) |later| @field(rest, later.field) = true;
            break :blk rest;
        }));
    }
    all.ticks_regular = false;
    try testing.expectEqual(BatchVerdict.spec_active, batchVerdictOf(all));
}

test "bucketOf: a plain slot batches, a speculating one by its class, and a full bucket sends the slot serial" {
    const empty = BucketFill{};
    try testing.expectEqual(TickBucket.batch, bucketOf(.ok, .none, empty));
    try testing.expectEqual(TickBucket.batch, bucketOf(.ok, .mtp_group, empty));
    try testing.expectEqual(TickBucket.mtp, bucketOf(.spec_active, .mtp_group, empty));
    try testing.expectEqual(TickBucket.rows, bucketOf(.spec_active, .rows_company, empty));
    try testing.expectEqual(TickBucket.serial, bucketOf(.spec_active, .none, empty));
    inline for (.{ .grammar, .logprobs, .penalty, .arch, .forced_call, .head_release_pending }) |why| {
        try testing.expectEqual(TickBucket.serial, bucketOf(why, .none, empty));
    }

    const full = MAX_BATCH_GROUP;
    try testing.expectEqual(TickBucket.batch, bucketOf(.ok, .none, .{ .batch = full - 1 }));
    try testing.expectEqual(TickBucket.serial, bucketOf(.ok, .none, .{ .batch = full }));
    try testing.expectEqual(TickBucket.serial, bucketOf(.spec_active, .mtp_group, .{ .mtp = full }));
    try testing.expectEqual(TickBucket.serial, bucketOf(.spec_active, .rows_company, .{ .rows = full }));
    // Another bucket's fill never blocks this one.
    try testing.expectEqual(TickBucket.mtp, bucketOf(.spec_active, .mtp_group, .{ .batch = full, .rows = full }));
}

test "crowdFits: a crowd goes plain at the threshold and only while the batch has room" {
    try testing.expect(!crowdFits(2, 4, 0));
    try testing.expect(crowdFits(4, 4, 0));
    try testing.expect(crowdFits(4, 4, MAX_BATCH_GROUP - 4));
    try testing.expect(!crowdFits(4, 4, MAX_BATCH_GROUP - 3));
}

test "rowsKept: the arch's bound caps a group, none declared means the batch width" {
    try testing.expectEqual(@as(usize, 3), rowsKept(5, 3));
    try testing.expectEqual(@as(usize, 2), rowsKept(2, 3));
    try testing.expectEqual(@as(usize, 5), rowsKept(5, null));
    try testing.expectEqual(@as(usize, MAX_BATCH_GROUP), rowsKept(MAX_BATCH_GROUP + 8, null));
}

test "a group is ordered by model, and a model's slots form one run" {
    const Item = struct { model: *const u8, id: u8 };
    var a: u8 = 0;
    var b: u8 = 0;
    const lo: *const u8, const hi: *const u8 = if (@intFromPtr(&a) < @intFromPtr(&b)) .{ &a, &b } else .{ &b, &a };
    var items = [_]Item{ .{ .model = hi, .id = 1 }, .{ .model = lo, .id = 2 }, .{ .model = hi, .id = 3 }, .{ .model = lo, .id = 4 } };
    sortByModel(Item, &items);
    try testing.expect(items[0].model == lo and items[1].model == lo);
    try testing.expect(items[2].model == hi and items[3].model == hi);
    try testing.expectEqual(@as(usize, 2), modelRunEnd(Item, &items, 0));
    try testing.expectEqual(@as(usize, 4), modelRunEnd(Item, &items, 2));
}

test "sortByKeyStable: ascending, and equal keys keep their order" {
    var items = [_]u8{ 'a', 'b', 'c', 'd', 'e' };
    var keys = [_]u32{ 7, 0, 7, 0, 3 };
    sortByKeyStable(u8, &items, &keys);
    try testing.expectEqualSlices(u8, "bdeac", &items);
    try testing.expectEqualSlices(u32, &.{ 0, 0, 3, 7, 7 }, &keys);

    var all_tied = [_]u8{ 'x', 'y', 'z' };
    var zero = [_]u32{ 0, 0, 0 };
    sortByKeyStable(u8, &all_tied, &zero);
    try testing.expectEqualSlices(u8, "xyz", &all_tied);
}

test "batchedKvKeepCount: padding waste caps the group, and the long slots are the ones dropped" {
    // Even lengths: no padding waste, everybody batches.
    try testing.expectEqual(@as(usize, 4), batchedKvKeepCount(&[_]u32{ 1000, 1000, 1000, 1000 }));
    try testing.expectEqual(@as(usize, 4), batchedKvKeepCount(&[_]u32{ 900, 1000, 1100, 1200 }));

    // The case this exists for: three short streams and one long one. Batching
    // all four pads to 4 x 100000 = 400k against 103k useful (3.9x); dropping
    // the long one leaves 3 x 1000 vs 3000 (1.0x).
    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&[_]u32{ 1000, 1000, 1000, 100_000 }));

    // Two long ones: the pair still batches together, since 2 x 100000 against
    // 200000 useful wastes nothing — the veto is about the padding, not length.
    try testing.expectEqual(@as(usize, 2), batchedKvKeepCount(&[_]u32{ 100_000, 100_000 }));
    // One short slot among two long ones still batches: 3 x 100000 padded
    // against 200010 useful is 1.5x, inside the bar. The veto is about the
    // WASTE the padding creates, not about any slot being an outlier.
    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&[_]u32{ 10, 100_000, 100_000 }));

    // One slot never "batches", and neither does an empty group.
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&[_]u32{1000}));
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&[_]u32{}));

    // Nothing prefilled yet: no padding to waste, so nothing is vetoed (the
    // ready gate, not this one, is what keeps unprefilled slots out).
    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&[_]u32{ 0, 0, 0 }));

    // A pathological pair degrades to no batch at all rather than making the
    // 1-token slot build a 200k-wide tensor nobody billed. This is why the bar
    // has to sit below 2.0 — at 2.0 a pair is unvetoable by construction.
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&[_]u32{ 1, 200_000 }));
}

test "groupKeepCount: the small-pad floor keeps a group whole only at long context" {
    // A long stream billed at the gather arm's 2052-row cap beside two short sub-agents: 3x pad.
    const billed = [_]u32{ 10, 10, 2052 };
    try testing.expectEqual(@as(usize, 2), groupKeepCount(&billed, PAD_FREE_MIN_CTX - 1));
    try testing.expectEqual(@as(usize, 3), groupKeepCount(&billed, PAD_FREE_MIN_CTX));
    // Past the floor's billed bound the cap holds at any context.
    try testing.expectEqual(@as(usize, 0), groupKeepCount(&[_]u32{ 1, PAD_FREE_KV + 1 }, 300_000));
    try testing.expectEqual(@as(usize, 2), groupKeepCount(&[_]u32{ 1, PAD_FREE_KV }, 300_000));
    // A group of one never batches.
    try testing.expectEqual(@as(usize, 0), groupKeepCount(&[_]u32{2052}, 300_000));
}

test "a slot running a DFlash drafter does not hold the module-owned MTP head" {
    try testing.expect(slotDrivesHead(true, false, true));
    try testing.expect(slotDrivesHead(true, true, false));
    try testing.expect(!slotDrivesHead(true, true, true));
    try testing.expect(!slotDrivesHead(false, false, false));
}

test "S21: a released module head drops the slot's exclusivity, and the MODEL bit is untouched" {
    try testing.expect(headExclusiveFor(false, true, true, false));
    try testing.expect(!headExclusiveFor(false, true, true, true));
    try testing.expect(!headExclusiveFor(false, true, false, false));
    try testing.expect(!headExclusiveFor(false, true, false, true));
    try testing.expect(!headExclusiveFor(false, false, true, false));
    try testing.expect(!headExclusiveFor(false, false, true, true));
    try testing.expect(headExclusiveFor(true, false, false, true));
    try testing.expect(headExclusiveFor(true, true, true, true));
}

test "admitPendingTick: exclusive single-flight FIFO contract" {
    const A: usize = 0xA0;
    const B: usize = 0xB0;
    var out: [16]usize = undefined;

    // Held while a live slot on the same exclusive model is decoding.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 0), admitPendingTick(&cands, &active, &out));
    }
    // Admits once no live slot holds the model.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &.{}, &out));
        try testing.expectEqual(@as(usize, 0), out[0]);
    }
    // Two exclusive candidates on the SAME model in one tick: only the
    // first admits — the tick-claim covers slots not yet in `decoding`.
    {
        const cands = [_]AdmitCand{
            .{ .model = A, .exclusive = true },
            .{ .model = A, .exclusive = true },
        };
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &.{}, &out));
        try testing.expectEqual(@as(usize, 0), out[0]);
    }
    // Distinct exclusive models admit independently.
    {
        const cands = [_]AdmitCand{
            .{ .model = A, .exclusive = true },
            .{ .model = B, .exclusive = true },
        };
        try testing.expectEqual(@as(usize, 2), admitPendingTick(&cands, &.{}, &out));
        try testing.expectEqual(@as(usize, 0), out[0]);
        try testing.expectEqual(@as(usize, 1), out[1]);
    }
}

test "admitPendingTick: non-exclusive concurrency and queue order preserved" {
    const A: usize = 0xA0;
    const L: usize = 0x10;
    var out: [16]usize = undefined;

    // Non-exclusive candidates (per-slot state) admit freely
    // even beside live slots — their serial-tick interleave is safe.
    {
        const cands = [_]AdmitCand{
            .{ .model = L, .exclusive = false },
            .{ .model = L, .exclusive = false },
        };
        const active = [_]AdmitCand{.{ .model = L, .exclusive = false }};
        try testing.expectEqual(@as(usize, 2), admitPendingTick(&cands, &active, &out));
    }
    // A held exclusive candidate must not head-of-line-block a later
    // candidate on another model (requests for OTHER models keep flowing).
    {
        const cands = [_]AdmitCand{
            .{ .model = A, .exclusive = true },
            .{ .model = L, .exclusive = false },
        };
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &active, &out));
        try testing.expectEqual(@as(usize, 1), out[0]);
    }
    // `out` caps the admitted count (mirrors to_prefill's 16), in order.
    {
        const cands = [_]AdmitCand{
            .{ .model = L, .exclusive = false },
            .{ .model = L, .exclusive = false },
            .{ .model = L, .exclusive = false },
        };
        var small: [2]usize = undefined;
        try testing.expectEqual(@as(usize, 2), admitPendingTick(&cands, &.{}, &small));
        try testing.expectEqual(@as(usize, 0), small[0]);
        try testing.expectEqual(@as(usize, 1), small[1]);
    }
}

test "idleEvictTickMs: sweeps well inside the window without spinning" {
    // Bar: a quarter of the window, clamped to [1s, 30s] — eviction lands near
    // the configured time without the sweep becoming a busy loop.
    try testing.expectEqual(@as(i64, 5_000), idleEvictTickMs(20_000));
    try testing.expectEqual(@as(i64, 15_000), idleEvictTickMs(60_000));
    try testing.expectEqual(@as(i64, 30_000), idleEvictTickMs(900_000));
    try testing.expectEqual(@as(i64, 1000), idleEvictTickMs(1000));
    try testing.expectEqual(@as(i64, 1000), idleEvictTickMs(2000));
    try testing.expectEqual(@as(i64, 30_000), idleEvictTickMs(28_800_000));
    try testing.expect(idleEvictTickMs(0) >= 1000);
}

test "re-arm: a declined tier is looked at once a minute, a settings save at once, a tier that is up never" {
    const s = std.time.ns_per_s;
    try testing.expect(!rearmDue(false, 1, 59 * s, REARM_PROBE_NS));
    try testing.expect(rearmDue(false, 1, 60 * s, REARM_PROBE_NS));
    try testing.expect(!rearmDue(false, 0, 600 * s, REARM_PROBE_NS));
    try testing.expect(rearmDue(true, 0, 0, REARM_PROBE_NS));
    try testing.expectEqual(REARM_PROBE_NS, rearmWaitNs(false));
    try testing.expect(rearmWaitNs(true) < rearmWaitNs(false));
}

test "gpu warm: ticks every half second inside the window after work, never past it or with work" {
    const w: u64 = 60 * std.time.ns_per_s;
    try testing.expectEqual(@as(?u64, null), gpuWarmParkNs(null, w));
    try testing.expectEqual(@as(?u64, GPU_WARM_TICK_NS), gpuWarmParkNs(0, w));
    try testing.expectEqual(@as(?u64, GPU_WARM_TICK_NS), gpuWarmParkNs(30 * std.time.ns_per_s, w));
    try testing.expectEqual(@as(?u64, 100), gpuWarmParkNs(w - 100, w));
    try testing.expectEqual(@as(?u64, null), gpuWarmParkNs(w, w));
    try testing.expectEqual(@as(?u64, null), gpuWarmParkNs(0, 0));
    try testing.expect(gpuWarmTickDue(false, 0, w));
    try testing.expect(!gpuWarmTickDue(true, 0, w));
    try testing.expect(!gpuWarmTickDue(false, null, w));
    try testing.expect(!gpuWarmTickDue(false, w, w));
    try testing.expect(!gpuWarmTickDue(false, 0, 0));
    try testing.expectEqual(GpuWarmWindow.restart, gpuWarmBeforePark(true, false));
    try testing.expectEqual(GpuWarmWindow.keep, gpuWarmBeforePark(false, false));
    try testing.expectEqual(GpuWarmWindow.close, gpuWarmBeforePark(false, true));
    try testing.expectEqual(GpuWarmWindow.close, gpuWarmBeforePark(true, true));
}

test "specInitWiring: a module-owned arch only gets the spec modes it can roll back" {
    // Spec decode must be able to ROLL BACK a rejected tail. The shell rolls
    // back its KVCache/ssm_entries — which a module-owned arch does not use, so
    // by the time the verify returns, the module has already absorbed every
    // draft and the shell's snapshot/truncate run over an EMPTY entries array.
    // The predicate is per-ARCH CAPABILITY (`moduleStateSpecRollback`), not ownership.
    // Args: (owns_module_state, module_spec_rollback,
    //        enable_mtp, has_mtp, enable_drafter, has_dflash)

    // Plain arch: MTP wins over a missing sidecar.
    {
        const w = specInitWiring(false, false, true, true, true, false);
        try testing.expect(w.use_mtp and !w.use_dflash);
    }
    // A flag with no loaded handle never arms.
    {
        const w = specInitWiring(false, false, true, false, false, false);
        try testing.expect(!w.use_mtp and !w.use_dflash);
    }

    // DFlash rides the enable_drafter switch: dflash > MTP.
    {
        const w = specInitWiring(false, false, false, false, true, true);
        try testing.expect(!w.use_mtp and w.use_dflash);
    }
    // A loaded drafter outranks the checkpoint's own MTP head: the sidecar
    // is an explicit choice (`--drafter` or the in-dir `drafter/`), the head
    // ships with every pack; `--no-drafter` restores MTP by not loading it.
    {
        const w = specInitWiring(false, false, true, true, true, true);
        try testing.expect(w.use_dflash and !w.use_mtp);
    }
    // enable_drafter:false on the request hands the round back to MTP.
    {
        const w = specInitWiring(false, false, true, true, false, true);
        try testing.expect(w.use_mtp and !w.use_dflash);
    }
    // enable_drafter:false opts the sidecar out.
    {
        const w = specInitWiring(false, false, false, false, false, true);
        try testing.expect(!w.use_dflash);
    }

    // Module-owned with NO rollback: everything off.
    {
        const w = specInitWiring(true, false, true, true, true, true);
        try testing.expect(!w.use_mtp and !w.use_dflash);
    }

    // Module-owned WITH rollback: a loaded, requested drafter outranks the head;
    // without one the head arms.
    {
        const w = specInitWiring(true, true, true, true, true, true);
        try testing.expect(w.use_dflash and !w.use_mtp);
    }
    {
        const w = specInitWiring(true, true, true, true, false, true);
        try testing.expect(w.use_mtp and !w.use_dflash);
    }
    {
        const w = specInitWiring(true, false, false, false, true, true);
        try testing.expect(!w.use_mtp and !w.use_dflash);
    }
    // Rollback capability alone never arms a head or a drafter that is not loaded.
    {
        const w = specInitWiring(true, true, true, false, true, false);
        try testing.expect(!w.use_mtp and !w.use_dflash);
    }
    // ...nor one the request opted out of.
    {
        const w = specInitWiring(true, true, false, true, true, false);
        try testing.expect(!w.use_mtp);
    }
    // An image request keeps the head when the drafter is off for it.
    {
        const w = specInitWiring(true, true, true, true, false, true);
        try testing.expect(w.use_mtp and !w.use_dflash);
    }
}

test "specTickMode: every spec arm requires the GENERATOR's armed state, not the slot flag alone" {
    // mtp and dflash each need the slot's wish AND the generator's handle.
    try testing.expectEqual(SpecTickMode.mtp, specTickMode(true, true, false, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(true, false, false, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, true, false));
    // Generator armed but the request opted enable_mtp off → serial.
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, true, false, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, false));

    // DFlash: slot's enable_drafter + generator's dflash handle; outranks the MTP
    // head (a loaded sidecar is the explicit choice). A generator handle alone never
    // resurrects it, and a dflash generator with the slot flag off stays regular, or
    // MTP when that is armed.
    try testing.expectEqual(SpecTickMode.dflash, specTickMode(false, false, true, true));
    try testing.expectEqual(SpecTickMode.dflash, specTickMode(true, true, true, true));
    try testing.expectEqual(SpecTickMode.mtp, specTickMode(true, true, false, true));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, true));
}

test "decode share: parse takes 0 to 0.9 and rejects anything else" {
    try testing.expectEqual(@as(f32, 0), try parseDecodeShare("0"));
    try testing.expectEqual(@as(f32, 0.5), try parseDecodeShare("0.5"));
    try testing.expectEqual(PREFILL_DECODE_SHARE_MAX, try parseDecodeShare("0.9"));
    for ([_][]const u8{ "", "abc", "nan", "inf", "0.95", "1", "-0.3", "0.5x" }) |bad| {
        try testing.expectError(error.InvalidDecodeShare, parseDecodeShare(bad));
    }
}

test "decode share: the budget is chunk * S / (1 - S), zero when off" {
    const ms = std.time.ns_per_ms;
    try testing.expectEqual(@as(u64, 0), decodeShareBudgetNs(400 * ms, 0));
    try testing.expectApproxEqAbs(@as(f64, 400 * ms), @as(f64, @floatFromInt(decodeShareBudgetNs(400 * ms, 0.5))), 1e3);
    try testing.expectApproxEqAbs(@as(f64, 3600 * ms), @as(f64, @floatFromInt(decodeShareBudgetNs(400 * ms, 0.9))), 1e4);
    try testing.expectApproxEqAbs(@as(f64, 400 * ms * 3 / 7), @as(f64, @floatFromInt(decodeShareBudgetNs(400 * ms, 0.3))), 1e4);
}

const FakeDecodeTicks = struct {
    left: u32,
    ns: u64,
    calls: u32 = 0,
    fn tick(ctx: *anyopaque) u64 {
        const f: *FakeDecodeTicks = @ptrCast(@alignCast(ctx));
        f.calls += 1;
        if (f.left == 0) return 0;
        f.left -= 1;
        return f.ns;
    }
};

test "decode share: ticks to the budget, keeps one tick at zero, stops when decoders run out" {
    const ms = std.time.ns_per_ms;
    // S=0.5 on a 400 ms chunk with 8 ms ticks: 50 ticks, 400 ms of decode.
    var full = FakeDecodeTicks{ .left = 1000, .ns = 8 * ms };
    const r = runOwedDecodeTicks(0.5, 400 * ms, 8 * ms, &full, FakeDecodeTicks.tick);
    try testing.expectEqual(@as(u32, 50), r.ticks);
    try testing.expectEqual(@as(u64, 400 * ms), r.spent_ns);
    // Share zero retains sushi's single boundary tick.
    for ([_]u64{ 100, 300, 2300, 8000 }) |c| {
        var legacy = FakeDecodeTicks{ .left = 1000, .ns = 8 * ms };
        const l = runOwedDecodeTicks(0, c * ms, 8 * ms, &legacy, FakeDecodeTicks.tick);
        try testing.expectEqual(@as(u32, 1), l.ticks);
    }
    // The decoders finish after 3 more ticks: the 4th call returns 0 and the loop stops.
    var dry = FakeDecodeTicks{ .left = 3, .ns = 8 * ms };
    const d = runOwedDecodeTicks(0.5, 400 * ms, 8 * ms, &dry, FakeDecodeTicks.tick);
    try testing.expectEqual(@as(u32, 4), d.ticks);
    try testing.expectEqual(@as(u32, 4), dry.calls);
    try testing.expectEqual(@as(u64, 32 * ms), d.spent_ns);
    // Nobody decoding at the boundary: the first tick measured 0, no more are tried.
    var none = FakeDecodeTicks{ .left = 1000, .ns = 8 * ms };
    const n = runOwedDecodeTicks(0.5, 2300 * ms, 0, &none, FakeDecodeTicks.tick);
    try testing.expectEqual(@as(u32, 1), n.ticks);
    try testing.expectEqual(@as(u32, 0), none.calls);
}

test "decode share: 0.25 owes a third of the chunk time in ticks" {
    const ms = std.time.ns_per_ms;
    var f = FakeDecodeTicks{ .left = 1000, .ns = 10 * ms };
    const r = runOwedDecodeTicks(0.25, 3000 * ms, 10 * ms, &f, FakeDecodeTicks.tick);
    try testing.expectEqual(@as(u32, 100), r.ticks);
    try testing.expectEqual(@as(u64, 1000 * ms), r.spent_ns);
}
