const std = @import("std");
const test_note = @import("test_note.zig");
const arch_handle = @import("arch/handle.zig");
const arch_kernels = @import("arch/kernels.zig");
const trunk = @import("trunk.zig");
const model_mod = @import("model.zig");
const model_registry_mod = @import("model_registry.zig");
const log = @import("log.zig");
const Trunk = trunk.Trunk;
const KVCache = trunk.KVCache;
const ModelConfig = model_mod.ModelConfig;
const LoadedModel = model_registry_mod.LoadedModel;
const prefill_plan = @import("prefill_plan.zig");
const testing = std.testing;
const scheduler_mod = @import("scheduler.zig");
const Scheduler = scheduler_mod.Scheduler;
const Slot = scheduler_mod.Slot;
const submitTestScheduler = scheduler_mod.submitTestScheduler;
const sched = @import("arch/sched.zig");
const sched_policy = @import("sched_policy.zig");
const static_bill = @import("static_bill.zig");
const forward_refusal_mod = @import("forward_refusal.zig");

pub const PostEvictionWidth = struct {
    /// The width the prefill runs at. Always the re-ask.
    width: u32,
    widened: bool,
    /// The re-ask came back narrower than the admitted width: memory moved between the reads.
    moved: bool,
};

/// The prefill width to run, given the width admission was billed at before the eviction
/// pass and the width re-asked against live memory after it. The re-ask wins in both
/// directions: `@max` would widen on memory that is gone. `admitted == 0` = no pass ran.
pub fn postEvictionPrefillChunk(admitted: u32, reasked: u32) PostEvictionWidth {
    return .{
        .width = reasked,
        .widened = admitted != 0 and reasked > admitted,
        .moved = admitted != 0 and reasked < admitted,
    };
}

/// The growth a request commits once the eviction pass re-asked its width: the cache it reserves at the width it runs
/// (`bills.admissionCommitBytes`), re-priced where the re-ask moved the width or the request restores a cache (the admitted
/// commit priced a cold one: a restore's decode regrow starts from the capacity it holds).
pub fn growthCommitAt(cfg: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_bits: u64, admitted_commit: u64, decision: PostEvictionWidth, warm: @import("bills.zig").WarmPrefix) u64 {
    if (!decision.widened and !decision.moved and warm.matched_tokens == 0) return admitted_commit;
    return @import("bills.zig").admissionCommitBytes(cfg, prompt_len, max_tokens, kv_bits, decision.width, warm);
}

/// A GLM request whose prefix-cache checkpoints do not fit even with the hot cache evicted keeps
/// fewer rather than be refused: the most below `cap` that `fits` admits, 0 = none; null = not
/// even that.
pub fn fewerCheckpointsToAdmit(cap: u32, ctx: ?*anyopaque, fits: *const fn (?*anyopaque, u32) bool) ?u32 {
    var n = cap;
    while (n > 0) {
        n -= 1;
        if (fits(ctx, n)) return n;
    }
    return null;
}

/// Does the inference thread run the evict-to-admit pass for this model? The connection
/// thread's credits (`server.creditedAdmissionBill`) read the same predicate. A load on the ledger runs none: its
/// admission releases the parked state itself (`static_bill.admit`).
pub fn admissionPassArmed(cfg: ?*const ModelConfig) bool {
    const c = cfg orelse return false;
    return c.static_ledger == null and c.admissionEvictsHotCache();
}

/// What `Generator.InitOptions.adaptive_chunk_width` gets for a slot.
pub fn adaptiveChunkWidthFor(cfg: ?*const model_mod.ModelConfig) bool {
    const c = cfg orelse return false;
    const enabled = scheduler_mod.prefill_chunk_adaptive_enabled orelse return false;
    return enabled(c);
}

// The pure decisions live in `sched_policy.zig`; these keep their old names for callers.

/// Does the loaded model's config batch at all? The arch half of `batchVerdict`,
/// shared with `/props`, `/v1/models` and the serve-mode startup line.
pub fn configBatchesDecode(cfg: *const model_mod.ModelConfig) bool {
    return sched.batches(cfg);
}

/// The rows one batched forward carries: the arch's bound, where it declares one (`sched.rowCap`).
pub fn batchGroupCap(cfg: *const model_mod.ModelConfig) usize {
    return sched.rowCap(cfg) orelse sched_policy.MAX_BATCH_GROUP;
}

/// One line per slot the first time it decodes serial beside live company;
/// the counter moves every tick so the rate is visible under `--metrics`.
pub fn noteSerial(sch: *Scheduler, slot: *Slot, why: sched_policy.BatchVerdict) void {
    if (sch.metrics) |m| m.decode_serial_total[@backingInt(why)].inc();
    if (slot.serial_reason_logged) return;
    slot.serial_reason_logged = true;
    log.info("[batched] slot serial: {s} (model={s})\n", .{ @tagName(why), slot.model.id });
}

pub fn padWasteKeepCount(group: []*Slot, gate_batch_kv_len: bool, kv_lens: *[32]u32) usize {
    var ctx_max: u32 = 0;
    // The stable insertion sort is part of the change: `std.sort.pdq` is unstable and
    // off qwen4_exp every key is `cache.step` == 0, so the sort decides the ordering.
    if (gate_batch_kv_len) {
        var caches_buf: [sched_policy.MAX_BATCH_GROUP]*const KVCache = undefined;
        for (group, 0..) |g, i| {
            caches_buf[i] = &g.cache;
        }
        fillGroupPadWasteKvLens(caches_buf[0..group.len], group[0].model.config, 1, kv_lens[0..group.len]);
        ctx_max = groupCtxMax(caches_buf[0..group.len]);
        sched_policy.sortByKeyStable(*Slot, group, kv_lens[0..group.len]);
    } else {
        std.sort.pdq(*Slot, group, {}, struct {
            fn lt(_: void, a: *Slot, b: *Slot) bool {
                return a.cache.step < b.cache.step;
            }
        }.lt);
        for (group, 0..) |g, i| kv_lens[i] = @intCast(g.cache.step);
    }
    return sched_policy.groupKeepCount(kv_lens[0..group.len], ctx_max);
}

pub fn groupCtxMax(caches: []const *const KVCache) u32 {
    var m: usize = 0;
    for (caches) |c| m = @max(m, c.kvLenForBatching());
    return @intCast(@min(m, std.math.maxInt(u32)));
}

/// One source for the length the batched group is sorted and capped by. `cache.step`
/// advances only on global layer 0, so on a linear-layer-0 trunk (GDN, gated-conv, Mamba2,
/// KDA) it is 0 forever and the pad-waste cap never fired. `KVCache.kvLenForBatching` reads
/// the first attention layer's own offset there.
pub fn batchKvLenOf(cache: *const KVCache, cfg: ?*const model_mod.ModelConfig) u32 {
    return batchKvLenOfWith(cache, cfg, 1);
}

pub fn batchKvLenOfWith(cache: *const KVCache, cfg: ?*const model_mod.ModelConfig, seq_len: c_int) u32 {
    // Arch gate: the multi-stream batched wins on the 27B were measured with the cap dead,
    // so every other arch keeps `cache.step` (and the dead cap) pending a measurement.
    const c = cfg orelse return @intCast(cache.step);
    if (!c.longCtxGated()) return @intCast(cache.step);
    const raw: u32 = @intCast(cache.kvLenForBatching());
    const gather_on = arch_kernels.batchedGatherOn(c, seq_len);
    const min_kv: u32 = @intCast(arch_kernels.batchedGatherFloor(c, seq_len, cache.config.scheme == .affine));
    return c.batchedEffectiveKvLen(raw, gather_on, min_kv);
}

pub fn fillGroupPadWasteKvLens(
    caches: []const *const KVCache,
    cfg: ?*const model_mod.ModelConfig,
    seq_len: c_int,
    out: []u32,
) void {
    for (caches, 0..) |c, i| out[i] = batchKvLenOfWith(c, cfg, seq_len);
}

/// A model whose per-request decode state is MODULE-OWNED (one per model,
/// not per slot) — at most ONE in-flight request may touch it. Ask the
/// transformer which archs own their decode state; never key on
/// a batching verdict and never name one arch here.
pub fn modelExclusiveDecode(model: *const model_registry_mod.LoadedModel) bool {
    const t = arch_handle.trunk(model.handle orelse return false);
    return t.ownsModuleDecodeState();
}

/// Per-SLOT exclusivity: the model's own bit, OR a slot that will drive a
/// module-owned MTP head (qwen4: `Qwen4Mtp.cache` is one per model). Plain
/// slots on the same model keep interleaving/batching beside it; two MTP
/// slots serialize, until one of them releases the head.
pub fn slotExclusiveDecode(slot: *const Slot) bool {
    const head = slot.model.mtp;
    return sched_policy.headExclusiveFor(
        modelExclusiveDecode(slot.model),
        if (head) |h| h.moduleOwned() else false,
        sched_policy.slotDrivesHead(slot.enable_mtp, slot.enable_drafter, slot.dflash != null),
        if (slot.legacy_gen) |*g| g.mtpModuleHeadReleased() else false,
    );
}

/// A cancel can land after the tick's cull, and `complete()` moves a slot from `decoding` to
/// `cleanup_queue` without a cull in between: until the inference thread has released it, the slot
/// still owns its native state and keeps blocking exclusive admission.
fn slotHoldsAdmission(slot: *const Slot) bool {
    return sched_policy.holdsAdmission(slot.finished, slot.error_code != null);
}

/// The prompts plus `max_tokens` a live slot reserved.
pub fn slotReservation(s: *const Slot) u64 {
    return @as(u64, s.full_prompt.len) +| s.max_tokens;
}

/// Live slots of `model` (decoding, prefilling, or queued for cleanup and not freed) and the tokens they reserved.
pub const ModelLive = struct { slots: u32 = 0, reserved: u64 = 0 };

pub fn liveCountFor(sch: *Scheduler, model: usize) u32 {
    var n: u32 = 0;
    for ([_][]const *Slot{ sch.decoding.items, sch.prefilling.items, sch.cleanup_queue.items }) |list| for (list) |s| {
        if (@intFromPtr(s.model) == model and slotHoldsAdmission(s)) n +|= 1;
    };
    return n;
}

pub fn liveFor(sch: *Scheduler, model: usize) ModelLive {
    var live: ModelLive = .{};
    for ([_][]const *Slot{ sch.decoding.items, sch.prefilling.items, sch.cleanup_queue.items }) |list| for (list) |s| {
        if (@intFromPtr(s.model) != model or !slotHoldsAdmission(s)) continue;
        live.slots +|= 1;
        live.reserved +|= slotReservation(s);
    };
    return live;
}

/// Entry ids the model's live slots restored from (shared buffers or a checkout): each is inside that slot's own
/// reservation, so the token pool counts it once and no eviction may take it.
pub fn liveEntryIds(sch: *Scheduler, model: usize, out: []u64) usize {
    var n: usize = 0;
    for ([_][]const *Slot{ sch.decoding.items, sch.prefilling.items, sch.cleanup_queue.items }) |list| for (list) |s| {
        if (@intFromPtr(s.model) != model or !slotHoldsAdmission(s) or s.restored_entry == 0) continue;
        if (std.mem.indexOfScalar(u64, out[0..n], s.restored_entry) != null) continue;
        if (n < out.len) {
            out[n] = s.restored_entry;
            n += 1;
        }
    };
    return n;
}

/// Tokens the model's retained hot-cache entries hold in RAM, never an entry a live slot uses (its slot's
/// reservation holds it) and never a checked-out one. The SSD tier's records hold nothing in the token pool.
pub fn retainedTokensFor(sch: *Scheduler, model: *const model_registry_mod.LoadedModel) u64 {
    const hc = model.prefix_cache orelse return 0;
    var ids: [64]u64 = undefined;
    const n = liveEntryIds(sch, @intFromPtr(model), &ids);
    var tokens: u64 = 0;
    for (hc.entries.items) |e| {
        if (e.checkout != null) continue;
        if (std.mem.indexOfScalar(u64, ids[0..n], e.id) != null) continue;
        tokens +|= e.tokens.len;
    }
    return tokens;
}

/// Inference thread, `queue_mu` held: publishes what `/props` reads without touching the hot cache.
pub fn publishModelReservedLocked(sch: *Scheduler, model: *model_registry_mod.LoadedModel) void {
    const n = liveFor(sch, @intFromPtr(model)).reserved +| retainedTokensFor(sch, model);
    model.reserved_tokens.store(n, .monotonic);
}

pub fn publishModelReserved(sch: *Scheduler, model: *model_registry_mod.LoadedModel) void {
    sch.queue_mu.lockUncancelable(sch.io);
    defer sch.queue_mu.unlock(sch.io);
    publishModelReservedLocked(sch, model);
}

/// A request's admission refusal for a `max_kv_tokens` its own reservation cannot fit, as the client reads it.
pub fn maxKvTokensRefusal(buf: []u8, cap: u64, reservation: u64, prompt: usize, max_tokens: u32) []const u8 {
    return std.fmt.bufPrint(buf, "This request reserves {d} tokens (prompt {d} + max_tokens {d}), above the model's max_kv_tokens {d}; lower the prompt or max_tokens, or raise max_kv_tokens", .{ reservation, prompt, max_tokens, cap }) catch "This request is above the model's max_kv_tokens.";
}

pub const AdmitTick = struct { admits: usize = 0, refused: usize = 0 };

/// The `[queue]` line of a candidate one of its model's caps holds this tick; null when neither cap holds it (an
/// earlier request of its model, an exclusive model's running one, or memory).
pub fn capWaitLine(buf: []u8, c: sched_policy.AdmitCand, cands: []const sched_policy.AdmitCand, verdicts: []const sched_policy.AdmitVerdict) ?[]const u8 {
    var tick_live: u32 = 0;
    var tick_reserved: u64 = 0;
    for (cands, verdicts) |o, v| if (v == .admit and o.model == c.model) {
        tick_live += 1;
        tick_reserved +|= o.reservation;
    };
    if (c.max_concurrent > 0 and c.live +| tick_live >= c.max_concurrent)
        return std.fmt.bufPrint(buf, "[queue] waiting for max_concurrent {d}: {d} of this model's requests run", .{ c.max_concurrent, c.live +| tick_live }) catch null;
    const held = c.reserved +| tick_reserved +| (c.retained -| c.own_retained);
    if (c.max_kv_tokens > 0 and held +| c.reservation > c.max_kv_tokens)
        return std.fmt.bufPrint(buf, "[queue] waiting for max_kv_tokens {d}: {d} tokens held, this request reserves {d}", .{ c.max_kv_tokens, held, c.reservation }) catch null;
    return null;
}

/// `queue_mu` held: this tick's admits and refusals from `sch.pending`, under each model's `max_concurrent` and
/// `max_kv_tokens`. A candidate held only by retained entries first evicts them oldest-first (never one a live slot
/// holds or has checked out) and is asked again; one over the cap alone is refused with the number it needs. Fills
/// `to_prefill` and `refused` with slot pointers; the caller removes them from `pending`.
pub fn admitWithinCaps(sch: *Scheduler, to_prefill: []*Slot, refused: []*Slot) AdmitTick {
    const CAND_MAX = 64;
    var cand_buf: [CAND_MAX]sched_policy.AdmitCand = undefined;
    var verdicts: [CAND_MAX]sched_policy.AdmitVerdict = undefined;
    var admit_idx: [16]usize = undefined;
    const n_cands = @min(sch.pending.items.len, CAND_MAX);
    var tick: AdmitTick = .{};
    var pass: u8 = 0;
    while (true) : (pass += 1) {
        var live_buf: [32]sched_policy.AdmitCand = undefined;
        const n_live = heldExclusive(&live_buf, &.{ sch.decoding.items, sch.cleanup_queue.items }, slotExclusiveDecode);
        for (sch.pending.items[0..n_cands], 0..) |s, i| {
            const model = @as(usize, @intFromPtr(s.model));
            const cfg = s.model.config;
            const count_gate = if (cfg) |c| c.max_concurrent > 0 else false;
            const kv_cap = if (cfg) |c| c.max_kv_tokens > 0 else false;
            const live = if (kv_cap) liveFor(sch, model) else ModelLive{ .slots = if (count_gate) liveCountFor(sch, model) else 0 };
            const retained = if (kv_cap) retainedTokensFor(sch, s.model) else 0;
            var own: u64 = 0;
            var own_id: u64 = 0;
            if (kv_cap) if (s.model.prefix_cache) |*cache| {
                const match = cache.restoreEntry(s.full_prompt);
                var ids: [64]u64 = undefined;
                const n_ids = liveEntryIds(sch, model, &ids);
                if (match.id != 0 and std.mem.indexOfScalar(u64, ids[0..n_ids], match.id) == null) {
                    own = match.tokens;
                    own_id = match.id;
                }
            };
            cand_buf[i] = .{
                .model = model,
                .exclusive = slotExclusiveDecode(s),
                .live = live.slots,
                .reserved = live.reserved,
                .retained = retained,
                .own_retained = own,
                .own_id = own_id,
                .reservation = slotReservation(s),
                .max_concurrent = if (cfg) |c| c.max_concurrent else 0,
                .max_kv_tokens = if (cfg) |c| c.max_kv_tokens else 0,
            };
        }
        _ = sched_policy.admitPendingTick(cand_buf[0..n_cands], live_buf[0..n_live], &admit_idx, verdicts[0..n_cands]);
        // Retained entries are the only obstacle: give the pool room, then ask again.
        var reclaimed = false;
        for (sch.pending.items[0..n_cands], verdicts[0..n_cands]) |s, v| {
            if (v != .reclaim) continue;
            const cfg = s.model.config orelse continue;
            const hc = if (s.model.prefix_cache) |*h| h else continue;
            const live = liveFor(sch, @intFromPtr(s.model));
            var protect: [64]u64 = undefined;
            const n_ids = liveEntryIds(sch, @intFromPtr(s.model), &protect);
            const match = hc.restoreEntry(s.full_prompt);
            const own = if (match.id != 0 and std.mem.indexOfScalar(u64, protect[0..n_ids], match.id) == null) match.tokens else 0;
            var n_protect = n_ids;
            if (match.id != 0 and n_protect < protect.len) {
                protect[n_protect] = match.id;
                n_protect += 1;
            }
            // A candidate this tick admits is not live yet: its reservation is held and its restore match kept.
            var sib_reserved: u64 = 0;
            var sib_own: u64 = 0;
            for (sch.pending.items[0..n_cands], verdicts[0..n_cands]) |o, ov| {
                if (ov != .admit or o.model != s.model) continue;
                sib_reserved +|= slotReservation(o);
                const m = hc.restoreEntry(o.full_prompt);
                if (m.id == 0 or std.mem.indexOfScalar(u64, protect[0..n_protect], m.id) != null) continue;
                sib_own +|= m.tokens;
                if (n_protect < protect.len) {
                    protect[n_protect] = m.id;
                    n_protect += 1;
                }
            }
            const held = live.reserved +| sib_reserved +| (retainedTokensFor(sch, s.model) -| own -| sib_own) +| slotReservation(s);
            const need = held -| cfg.max_kv_tokens;
            if (need == 0 or hc.evictOldestForTokens(need, protect[0..n_protect]) == 0) continue;
            reclaimed = true;
        }
        if (!reclaimed or pass >= 8) break;
    }
    // Policy first (exclusive, then the two caps), memory second: exactly the admits the policy picked are billed
    // together, and `admitted_held` is what later admissions and `/props` see.
    var admitted_idx: [64]usize = undefined;
    var n_admits: usize = 0;
    for (verdicts[0..n_cands], 0..) |v, i| {
        if (v != .admit or n_admits >= admitted_idx.len) continue;
        admitted_idx[n_admits] = i;
        n_admits += 1;
    }
    const n_memory = memoryAdmitCount(sch, admitted_idx[0..n_admits]);
    for (sch.pending.items[0..n_cands], 0..) |s, i| {
        switch (verdicts[i]) {
            .admit => {
                if (std.mem.indexOfScalar(usize, admitted_idx[0..n_memory], i) == null) continue;
                if (tick.admits < to_prefill.len) {
                    to_prefill[tick.admits] = s;
                    tick.admits += 1;
                }
            },
            .refuse => {
                const cfg = s.model.config orelse continue;
                var text = forward_refusal_mod.Text{};
                const line = maxKvTokensRefusal(&text.buf, cfg.max_kv_tokens, slotReservation(s), s.full_prompt.len, s.max_tokens);
                text.len = line.len;
                forward_refusal_mod.record(line);
                s.markError("PrefillDoesNotFit");
                log.info("[admission] refused: {s}\n", .{line});
                if (tick.refused < refused.len) {
                    refused[tick.refused] = s;
                    tick.refused += 1;
                }
            },
            .wait, .reclaim => {
                if (s.queue_logged) continue;
                var buf: [192]u8 = undefined;
                const line = capWaitLine(&buf, cand_buf[i], cand_buf[0..n_cands], verdicts[0..n_cands]) orelse continue;
                s.queue_logged = true;
                log.info("{s}\n", .{line});
            },
        }
    }
    var published: [64]usize = undefined;
    var n_published: usize = 0;
    for (sch.pending.items[0..n_cands]) |s| {
        const id = @intFromPtr(s.model);
        if (std.mem.indexOfScalar(usize, published[0..n_published], id) != null) continue;
        if (n_published >= published.len) continue;
        published[n_published] = id;
        n_published += 1;
        publishModelReservedLocked(sch, s.model);
    }
    return tick;
}

/// Exclusive slots in `lists` (decoding, then queued for cleanup) that still hold admission.
pub fn heldExclusive(out: []sched_policy.AdmitCand, lists: []const []const *Slot, exclusive: *const fn (*const Slot) bool) usize {
    var n: usize = 0;
    for (lists) |list| for (list) |s| {
        if (n >= out.len) return n;
        if (!slotHoldsAdmission(s) or !exclusive(s)) continue;
        out[n] = .{ .model = @intFromPtr(s.model), .exclusive = true };
        n += 1;
    };
    return n;
}

/// What a pending request waits behind. `running_*` describe the oldest live request; its
/// `running_max` less `running_generated` is an upper bound on the wait.
pub const QueueWait = struct {
    ahead: u32,
    running_prompt: u32 = 0,
    running_generated: u32 = 0,
    running_max: u32 = 0,
    /// Tokens per second the running request decoded so far; 0 when unmeasured.
    tps: f64 = 0,
};

/// Caller holds `queue_mu`. Null unless `slot` is still pending.
fn queueWaitLocked(sch: *Scheduler, slot: *const Slot) ?QueueWait {
    const pos = for (sch.pending.items, 0..) |p, i| {
        if (p == slot) break i;
    } else return null;
    var w = QueueWait{ .ahead = @intCast(pos) };
    var running: ?*const Slot = null;
    for ([_][]const *Slot{ sch.decoding.items, sch.prefilling.items }) |list| for (list) |s| {
        if (!slotHoldsAdmission(s) or s.cancelled.load(.acquire)) continue;
        w.ahead += 1;
        if (running == null) running = s;
    };
    if (running) |s| {
        const generated = @atomicLoad(u32, &s.completion_tokens, .monotonic);
        const decode_ns = @atomicLoad(u64, &s.decode_ns, .monotonic);
        w.running_prompt = @intCast(s.full_prompt.len);
        w.running_generated = generated;
        w.running_max = s.max_tokens;
        if (generated > 0 and decode_ns > 0) w.tps = @as(f64, @floatFromInt(generated)) * std.time.ns_per_s / @as(f64, @floatFromInt(decode_ns));
    }
    return w;
}

/// Connection thread: what `slot` waits behind, or null once it is admitted.
pub fn queueWait(sch: *Scheduler, slot: *const Slot) ?QueueWait {
    sch.queue_mu.lockUncancelable(sch.io);
    defer sch.queue_mu.unlock(sch.io);
    return queueWaitLocked(sch, slot);
}

/// The `[queue]` log line (`log_line`) or the SSE comment line a streamed client gets, without
/// its line ending.
pub fn queueWaitText(buf: []u8, w: QueueWait, comptime log_line: bool) []const u8 {
    var out = std.Io.Writer.fixed(buf);
    out.print(if (log_line) "[queue] waiting behind {d} request(s)" else ": queued behind {d} request(s)", .{w.ahead}) catch return out.buffered();
    if (w.running_max == 0) return out.buffered();
    (if (log_line)
        out.print("; running: {d}+{d} of max {d} tokens", .{ w.running_prompt, w.running_generated, w.running_max })
    else
        out.print("; running {d} of {d} tokens", .{ w.running_generated, w.running_max })) catch return out.buffered();
    if (w.tps > 0) {
        const left: f64 = @floatFromInt(w.running_max -| w.running_generated);
        out.print(" (~{d} s at {d:.1} tok/s)", .{ @as(u64, @intFromFloat(@round(left / w.tps))), w.tps }) catch {};
    }
    return out.buffered();
}

/// Inference thread, `queue_mu` held: the one `[queue]` line a pending slot gets; null after it.
pub fn queueLogLine(sch: *Scheduler, buf: []u8, slot: *Slot) ?[]const u8 {
    if (slot.queue_logged) return null;
    const w = queueWaitLocked(sch, slot) orelse return null;
    slot.queue_logged = true;
    return queueWaitText(buf, w, true);
}

/// The line for a slot that was logged as waiting and now starts; null when it never waited.
pub fn queueStartLine(buf: []u8, slot: *const Slot, waited_ns: u64) ?[]const u8 {
    if (!slot.queue_logged) return null;
    const secs = @as(f64, @floatFromInt(waited_ns)) / std.time.ns_per_s;
    return std.fmt.bufPrint(buf, "[queue] started after {d:.1} s", .{secs}) catch null;
}

/// `commit` is the part of `needed` the request allocates after admission (cache growth the bill
/// prices up front); until it is resident, later admissions must leave it free. `held` is `needed`
/// less the bill's safety margin: what the request holds at its peak.
pub const MemoryBill = struct { needed: u64, available: u64, commit: u64 = 0, held: u64 = 0 };

/// How many of one tick's admits (queue order) go on to prefill. Each connection-thread bill
/// ran before any sibling allocated, so an ungated arch is re-billed here: an admit that does
/// not fit beside live requests plus this tick's earlier admits stays pending, and so does
/// every admit after it. Alone it proceeds: nobody would ever free memory for it. The gated arch
/// is billed here too, so a Qwen and a MiMo admit see each other, and again inside `runPrefill`.
/// `outstanding` is what live requests were admitted to allocate and have not yet.
pub fn admitsWithinMemory(bills: []const ?MemoryBill, live_company: bool, outstanding: u64) usize {
    var promised: u64 = outstanding;
    for (bills, 0..) |bill, i| {
        const b = bill orelse continue;
        if ((live_company or i > 0) and b.needed +| promised > b.available) return i;
        promised +|= b.needed;
    }
    return bills.len;
}

/// What a live slot was admitted to allocate and has not: its commitment less its resident
/// cache. A DFlash2 slot reserves its capacity after prefill, so its share drains at once.
pub fn outstandingGrowth(commit: u64, resident: u64) u64 {
    return commit -| resident;
}

fn slotOutstandingGrowth(s: *const Slot) u64 {
    var resident = s.cache.residentBytes();
    // A running request's own state (DeepSeek's decode state too) exists and `requestBytes` bills it.
    resident +|= sched.requestBytes(s);
    return outstandingGrowth(s.growth_commit, resident);
}

/// The requests admitted and not yet freed, and the bytes their bills priced them to hold
/// (`Slot.admitted_held`): what comes back once every one of them has finished.
pub const LiveHold = struct { requests: u32 = 0, bytes: u64 = 0 };

/// Connection thread: `Scheduler.live_hold`, under `queue_mu`.
pub fn liveHold(sch: *Scheduler) LiveHold {
    sch.queue_mu.lockUncancelable(sch.io);
    defer sch.queue_mu.unlock(sch.io);
    return sch.live_hold;
}

/// `queue_mu` held: `slot` is admitted, and holds what its bill priced until `releaseHoldLocked` after its free.
pub fn countHoldLocked(sch: *Scheduler, slot: *Slot) void {
    slot.hold_counted = true;
    sch.live_hold.requests += 1;
    sch.live_hold.bytes +|= slot.admitted_held;
}

/// What `slots` were counted holding; read before they are freed.
pub fn heldBy(slots: []const *Slot) LiveHold {
    var hold: LiveHold = .{};
    for (slots) |s| if (s.hold_counted) {
        hold.requests += 1;
        hold.bytes +|= s.admitted_held;
    };
    return hold;
}

/// `queue_mu` held: `freed` (`heldBy`) has been freed.
pub fn releaseHoldLocked(sch: *Scheduler, freed: LiveHold) void {
    sch.live_hold.requests -|= freed.requests;
    sch.live_hold.bytes -|= freed.bytes;
}

/// `admitsWithinMemory` over this tick's admits; called under `queue_mu`.
pub fn memoryAdmitCount(sch: *Scheduler, admit_idx: []const usize) usize {
    const numbers_fn = scheduler_mod.prefill_admission_numbers;
    // A finished request whose slot is not freed yet still holds its memory: it is company too.
    var live = sch.live_hold.requests > 0;
    var outstanding: u64 = 0;
    for (sch.decoding.items) |s| {
        if (!s.joinsTick()) continue;
        live = true;
        outstanding +|= slotOutstandingGrowth(s);
    }
    var bills: [16]?MemoryBill = undefined;
    const n = @min(admit_idx.len, bills.len);
    for (admit_idx[0..n], 0..) |idx, i| {
        bills[i] = null;
        const s = sch.pending.items[idx];
        const cfg = s.model.config orelse continue;
        // A model with no handle is not loaded, except where a test's own bill hook stands in for the load.
        if (s.model.handle == null and numbers_fn == null) continue;
        bills[i] = static_bill.holdBill(cfg, s.full_prompt.len, s.max_tokens, s.deferred_media) orelse if (numbers_fn) |f|
            f(cfg, s.full_prompt.len, s.max_tokens, s.cache.config, prefill_plan.visionPrefillUnchunked(s.vision_embeddings != null or s.deferred_media != null), s.enable_mtp)
        else
            null;
    }
    const admitted = admitsWithinMemory(bills[0..n], live, outstanding);
    for (admit_idx[0..admitted], bills[0..admitted]) |idx, bill| {
        const s = sch.pending.items[idx];
        s.growth_commit = if (bill) |b| b.commit else 0;
        s.admitted_held = if (bill) |b| b.held else 0;
    }
    if (admitted < n) {
        const held = sch.pending.items[admit_idx[admitted]];
        if (!held.memory_hold_logged) {
            held.memory_hold_logged = true;
            log.info("[admission] held: {d} tokens need ~{d}MB, ~{d}MB available beside live requests (~{d}MB of their growth outstanding); waiting for one to finish\n", .{
                held.full_prompt.len, bills[admitted].?.needed >> 20, bills[admitted].?.available >> 20, outstanding >> 20,
            });
        }
    }
    return admitted;
}

test "admitsWithinMemory: siblings are billed together, a lone request always proceeds" {
    const gb: u64 = 1 << 30;
    const b: ?MemoryBill = .{ .needed = 8 * gb, .available = 20 * gb };
    // Four arrivals each fit alone against the same free memory; only two fit together.
    try testing.expectEqual(@as(usize, 2), admitsWithinMemory(&.{ b, b, b, b }, false, 0));
    // Beside a live request the first must fit by itself.
    const big: ?MemoryBill = .{ .needed = 30 * gb, .available = 20 * gb };
    try testing.expectEqual(@as(usize, 0), admitsWithinMemory(&.{ big, b }, true, 0));
    // Alone it proceeds whatever the bill: nothing would ever free memory for it.
    try testing.expectEqual(@as(usize, 1), admitsWithinMemory(&.{big}, false, 0));
    // A gated admit carries no bill here and never blocks the queue.
    try testing.expectEqual(@as(usize, 3), admitsWithinMemory(&.{ null, b, b }, false, 0));
}

test "admissionPassArmed: the evict-to-admit pass runs for every served arch, never an unserved one" {

    try test_note.needArchs(&.{ "qwen4", "mimo2", "glm5" });
    const mimo = ModelConfig{ .model_type = "mimo_v2", .num_hidden_layers = 1, .has_sliding_window = true, .sliding_window = 128, .head_dim = 192 };
    const qwen4 = ModelConfig{ .model_type = "qwen4_exp" };
    const glm = ModelConfig{ .model_type = "glm5_next" };
    const unserved = ModelConfig{ .model_type = "not_a_served_arch", .has_sliding_window = true, .sliding_window = 128, .head_dim = 192 };
    try testing.expect(admissionPassArmed(&mimo));
    try testing.expect(admissionPassArmed(&qwen4));
    try testing.expect(admissionPassArmed(&glm));
    try testing.expect(!admissionPassArmed(&unserved));
    try testing.expect(!admissionPassArmed(null));
}

test "a resident MiMo batches decode in groups of the FP8 GEMV's row-identical width; a streamed one stays serial" {

    try test_note.needArchs(&.{"mimo2"});
    var cfg = std.mem.zeroes(model_mod.ModelConfig);
    cfg.model_type = "mimo_v2";
    cfg.num_experts = 256;
    try testing.expect(configBatchesDecode(&cfg));
    try testing.expectEqual(@as(usize, 4), batchGroupCap(&cfg));
    cfg.expert_streaming = true;
    try testing.expect(!configBatchesDecode(&cfg));
    try testing.expectEqual(sched_policy.MAX_BATCH_GROUP, batchGroupCap(&cfg));
}

test "the batched group is capped by padding waste: the longest slot sorts last and falls out" {
    var slots: [3]Slot = undefined;
    var group: [3]*Slot = undefined;
    for (&slots, &group, [_]usize{ 90_000, 1_000, 1_200 }) |*slot, *g, step| {
        slot.cache = .{ .entries = &.{}, .step = step, .allocator = testing.allocator, .config = .dense };
        g.* = slot;
    }
    var kv_lens: [32]u32 = undefined;
    try testing.expectEqual(@as(usize, 2), padWasteKeepCount(&group, false, &kv_lens));
    try testing.expectEqualSlices(*Slot, &.{ &slots[1], &slots[2], &slots[0] }, &group);
    try testing.expectEqualSlices(u32, &.{ 1_000, 1_200, 90_000 }, kv_lens[0..3]);
}

test "the pad-waste cap reads the arch's TRUE attention KV length, not cache.step" {

    try test_note.needArchs(&.{"qwen4"});
    // `cache.step` is 0 forever on a linear-layer-0 trunk, so the cap never fired there.
    const lens = [_]usize{ 1_000, 1_000, 60_000 };
    const first_attn = [_]u32{ 3, 2, 7 }; // GDN / gated-conv / KDA spacings
    var caches: [3]KVCache = undefined;
    var built: usize = 0;
    defer for (caches[0..built]) |*c| c.deinit();
    for (lens, first_attn, 0..) |len, fa, i| {
        caches[i] = try KVCache.init(testing.allocator, 32);
        built += 1;
        caches[i].entries[fa].initialized = true;
        caches[i].entries[fa].offset = len;
        try testing.expectEqual(@as(usize, 0), caches[i].step); // the trap
    }

    var q4 = model_mod.ModelConfig{ .model_type = "qwen4_exp" };
    var kv_lens: [3]u32 = undefined;
    for (caches[0..], 0..) |*c, i| kv_lens[i] = batchKvLenOf(c, &q4);
    try testing.expectEqualSlices(u32, &[_]u32{ 1_000, 1_000, 60_000 }, &kv_lens);

    // Every other hybrid keeps `cache.step` (the cap stays dead there, pending a multi-stream measurement).
    for ([_][]const u8{ "mimo_v2", "glm5_next", "not_a_served_arch" }) |mt| {
        var cfg = model_mod.ModelConfig{ .model_type = mt };
        for (caches[0..]) |*c| try testing.expectEqual(@as(u32, 0), batchKvLenOf(c, &cfg));
    }
    for (caches[0..]) |*c| try testing.expectEqual(@as(u32, 0), batchKvLenOf(c, null));

    try testing.expectEqual(@as(usize, 2), sched_policy.batchedKvKeepCount(&kv_lens));
    try testing.expect(sched_policy.batchedPadWaste(&kv_lens) > sched_policy.MAX_PAD_WASTE);

    try testing.expectEqual(@as(usize, 3), sched_policy.batchedKvKeepCount(&[_]u32{ 0, 0, 0 }));
    try testing.expectEqual(@as(f64, 1.0), sched_policy.batchedPadWaste(&[_]u32{ 0, 0, 0 }));
}

test "an attention-first trunk's batching lengths are unchanged by the fix" {
    // Dense / attention-first archs are byte-identical across this change.
    const lens = [_]usize{ 1_000, 1_000, 1_000, 100_000 };
    var unserved = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    var caches: [4]KVCache = undefined;
    var built: usize = 0;
    defer for (caches[0..built]) |*c| c.deinit();
    var kv_lens: [4]u32 = undefined;
    for (lens, 0..) |len, i| {
        caches[i] = try KVCache.init(testing.allocator, 8);
        built += 1;
        caches[i].step = len;
        caches[i].entries[0].initialized = true;
        caches[i].entries[0].offset = len;
        kv_lens[i] = batchKvLenOf(&caches[i], &unserved);
    }
    try testing.expectEqualSlices(u32, &[_]u32{ 1_000, 1_000, 1_000, 100_000 }, &kv_lens);
    var q4b = model_mod.ModelConfig{ .model_type = "qwen4_exp" };
    for (caches[0..], 0..) |*c, i| try testing.expectEqual(kv_lens[i], batchKvLenOf(c, &q4b));
    try testing.expectEqual(@as(usize, 3), sched_policy.batchedKvKeepCount(&kv_lens));
    try testing.expectEqual(
        sched_policy.batchedKvKeepCount(&[_]u32{ 1_000, 1_000, 1_000, 100_000 }),
        sched_policy.batchedKvKeepCount(&kv_lens),
    );
}

test "a cancelled or queued-for-cleanup slot still blocks exclusive admission until released" {
    const Fake = struct {
        fn exclusive(_: *const Slot) bool {
            return true;
        }
    };
    var model: LoadedModel = undefined;
    var live: Slot = undefined;
    var cancelled: Slot = undefined;
    var queued: Slot = undefined;
    var finished: Slot = undefined;
    for ([_]*Slot{ &live, &cancelled, &queued, &finished }) |s| {
        s.model = &model;
        s.finished = false;
        s.error_code = null;
        s.cancelled = .init(false);
    }
    cancelled.cancelled.store(true, .release);
    finished.finished = true;
    var out: [8]sched_policy.AdmitCand = undefined;
    const decoding = [_]*Slot{ &live, &cancelled, &finished };
    const cleanup = [_]*Slot{&queued};
    try testing.expectEqual(@as(usize, 3), heldExclusive(&out, &.{ &decoding, &cleanup }, Fake.exclusive));
    try testing.expectEqual(@as(usize, 0), heldExclusive(&out, &.{ &.{}, &.{} }, Fake.exclusive));
}

test "postEvictionPrefillChunk: the re-asked width is the one that runs, in BOTH directions" {
    // 1. Admitted at 1024, affords 4096 after the pass: runs at 4096, logs the widen.
    {
        const d = postEvictionPrefillChunk(1024, 4096);
        try testing.expectEqual(@as(u32, 4096), d.width);
        try testing.expect(d.widened);
        try testing.expect(!d.moved);
    }
    try testing.expectEqual(@as(u32, 8192), postEvictionPrefillChunk(512, 8192).width);

    // 2. No eviction pass ran: the re-ask is the only ask, nothing reportable.
    {
        const d = postEvictionPrefillChunk(0, 1024);
        try testing.expectEqual(@as(u32, 1024), d.width);
        try testing.expect(!d.widened);
        try testing.expect(!d.moved);
    }
    try testing.expectEqual(@as(u32, 512), postEvictionPrefillChunk(0, 512).width);
    try testing.expectEqual(@as(u32, 0), postEvictionPrefillChunk(0, 0).width);

    // 3. A narrower re-ask still wins (`@max` would widen on memory that is gone).
    {
        const d = postEvictionPrefillChunk(4096, 1024);
        try testing.expectEqual(@as(u32, 1024), d.width);
        try testing.expect(d.moved);
        try testing.expect(!d.widened);
    }
    try testing.expectEqual(@as(u32, 512), postEvictionPrefillChunk(2048, 512).width);

    // 4. Unchanged memory is a no-op in both directions.
    inline for (.{ 512, 1024, 2048, 4096, 8192 }) |w| {
        const d = postEvictionPrefillChunk(w, w);
        try testing.expectEqual(@as(u32, w), d.width);
        try testing.expect(!d.widened and !d.moved);
    }

    // 5. Never below the ladder floor: this function computes no width of its own.
    inline for (.{ 512, 1024, 2048, 4096, 8192 }) |floor| {
        try testing.expectEqual(@as(u32, floor), postEvictionPrefillChunk(8192, floor).width);
    }
}

test "admitsWithinMemory: a live request's outstanding growth is not free headroom" {
    const gb: u64 = 1 << 30;
    const b: ?MemoryBill = .{ .needed = 9 * gb, .available = 10 * gb };
    try testing.expectEqual(@as(usize, 1), admitsWithinMemory(&.{b}, true, 0));
    try testing.expectEqual(@as(usize, 0), admitsWithinMemory(&.{b}, true, 2 * gb));
    // Alone nothing is outstanding, and the lone request proceeds whatever the bill.
    try testing.expectEqual(@as(usize, 1), admitsWithinMemory(&.{b}, false, 0));
}

test "a staggered admission waits for the growth an earlier request was admitted to allocate, and for a finished one's slot to be freed" {
    const gb: u64 = 1 << 30;
    const Probe = struct {
        var available: u64 = 10 * gb;
        fn numbers(_: *const ModelConfig, _: usize, _: u32, _: trunk.KVQuantConfig, _: bool, _: bool) MemoryBill {
            return .{ .needed = 9 * gb, .available = available, .commit = 6 * gb };
        }
    };
    const saved = scheduler_mod.prefill_admission_numbers;
    scheduler_mod.prefill_admission_numbers = Probe.numbers;
    defer scheduler_mod.prefill_admission_numbers = saved;
    var cfg = ModelConfig{ .model_type = "glm5_next" };
    var xfm: Trunk = undefined;
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.handle = arch_handle.fixtureHandle(&xfm);
    var a: Slot = undefined;
    a.model = &model;
    a.full_prompt = &.{};
    a.max_tokens = 1;
    a.cache = .{ .entries = &.{}, .step = 0, .allocator = testing.allocator, .config = .dense };
    a.request = null;
    a.vision_embeddings = null;
    a.enable_mtp = false;
    a.memory_hold_logged = true;
    a.cancelled = std.atomic.Value(bool).init(false);
    a.finished = false;
    a.error_code = null;
    a.growth_commit = 0;
    var b = a;
    var sch: Scheduler = undefined;
    sch.pending = .empty;
    sch.decoding = .empty;
    sch.live_hold = .{};
    defer sch.pending.deinit(testing.allocator);
    defer sch.decoding.deinit(testing.allocator);

    try sch.pending.append(testing.allocator, &a);
    try testing.expectEqual(@as(usize, 1), memoryAdmitCount(&sch, &.{0}));
    try testing.expectEqual(6 * gb, a.growth_commit);

    // A decodes with nothing of its growth resident yet; B arrives a tick later.
    _ = sch.pending.orderedRemove(0);
    try sch.decoding.append(testing.allocator, &a);
    try sch.pending.append(testing.allocator, &b);
    try testing.expectEqual(@as(usize, 0), memoryAdmitCount(&sch, &.{0}));
    try testing.expectEqual(@as(u64, 0), b.growth_commit);

    // A finished A holds its memory until its slot is freed: B, short of it, waits instead of going on alone.
    a.finished = true;
    Probe.available = 8 * gb;
    defer Probe.available = 10 * gb;
    sch.live_hold = .{ .requests = 1, .bytes = 9 * gb };
    try testing.expectEqual(@as(usize, 0), memoryAdmitCount(&sch, &.{0}));
    // Freed, its commitment is gone and B proceeds.
    Probe.available = 10 * gb;
    sch.live_hold = .{};
    try testing.expectEqual(@as(usize, 1), memoryAdmitCount(&sch, &.{0}));
}

test "a commitment is released as the cache allocates it" {
    try testing.expectEqual(@as(u64, 6), outstandingGrowth(6, 0));
    try testing.expectEqual(@as(u64, 2), outstandingGrowth(6, 4));
    // A DFlash2 request reserves its whole capacity after prefill.
    try testing.expectEqual(@as(u64, 0), outstandingGrowth(6, 6));
    try testing.expectEqual(@as(u64, 0), outstandingGrowth(6, 7));
}

/// A slot reduced to the fields the queue report reads; no mlx state.
fn queueTestSlot(prompt_len: usize, generated: u32, max: u32, decode_s: u64) Slot {
    var s: Slot = undefined;
    s.full_prompt = scheduler_mod.queue_test_prompt[0..prompt_len];
    s.completion_tokens = generated;
    s.max_tokens = max;
    s.decode_ns = decode_s * std.time.ns_per_s;
    s.finished = false;
    s.error_code = null;
    s.cancelled = .init(false);
    s.queue_logged = false;
    return s;
}

test "a request held behind a running one logs what it waits behind, once" {
    const a = testing.allocator;
    var sch = submitTestScheduler(a, 8);
    defer sch.pending.deinit(a);
    defer sch.decoding.deinit(a);
    defer sch.prefilling.deinit(a);
    var running = queueTestSlot(3000, 8000, 32768, 200);
    var first = queueTestSlot(10, 0, 100, 0);
    var second = queueTestSlot(10, 0, 100, 0);
    try sch.decoding.append(a, &running);
    try sch.pending.append(a, &first);
    try sch.pending.append(a, &second);

    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings(
        "[queue] waiting behind 1 request(s); running: 3000+8000 of max 32768 tokens (~619 s at 40.0 tok/s)",
        queueLogLine(&sch, &buf, &first).?,
    );
    try testing.expect(queueLogLine(&sch, &buf, &first) == null);
    try testing.expectEqualStrings(
        "[queue] waiting behind 2 request(s); running: 3000+8000 of max 32768 tokens (~619 s at 40.0 tok/s)",
        queueLogLine(&sch, &buf, &second).?,
    );
}

test "the queue report leaves out a decode rate nobody has measured, and a slot that is not queued has none" {
    const a = testing.allocator;
    var sch = submitTestScheduler(a, 8);
    defer sch.pending.deinit(a);
    defer sch.decoding.deinit(a);
    defer sch.prefilling.deinit(a);
    var running = queueTestSlot(3000, 0, 32768, 0);
    var waiting = queueTestSlot(10, 0, 100, 0);
    var other = queueTestSlot(10, 0, 100, 0);
    try sch.prefilling.append(a, &running);
    try sch.pending.append(a, &waiting);
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings(
        "[queue] waiting behind 1 request(s); running: 3000+0 of max 32768 tokens",
        queueLogLine(&sch, &buf, &waiting).?,
    );
    try testing.expect(queueWaitLocked(&sch, &other) == null);
    running.finished = true;
    const w = queueWaitLocked(&sch, &waiting).?;
    try testing.expectEqual(@as(u32, 0), w.ahead);
    try testing.expectEqualStrings("[queue] waiting behind 0 request(s)", queueWaitText(&buf, w, true));
}

test "a request that waited reports how long once it starts, and one that did not stays silent" {
    var buf: [64]u8 = undefined;
    var waited = queueTestSlot(10, 0, 100, 0);
    waited.queue_logged = true;
    try testing.expectEqualStrings("[queue] started after 12.3 s", queueStartLine(&buf, &waited, 12_340_000_000).?);
    var direct = queueTestSlot(10, 0, 100, 0);
    try testing.expect(queueStartLine(&buf, &direct, 12_340_000_000) == null);
}

test "the streamed queue notice is one SSE comment line with the same numbers" {
    var buf: [256]u8 = undefined;
    const w = QueueWait{ .ahead = 2, .running_prompt = 3000, .running_generated = 8000, .running_max = 32768, .tps = 40 };
    try testing.expectEqualStrings(
        ": queued behind 2 request(s); running 8000 of 32768 tokens (~619 s at 40.0 tok/s)",
        queueWaitText(&buf, w, false),
    );
}

fn admissionTestEntry(allocator: std.mem.Allocator, tokens: []const u32, used: u64) !@import("prefix_cache.zig").Entry {
    return .{
        .tokens = try allocator.dupe(u32, tokens),
        .has_tools = false,
        .snapshot = .{ .entries = try allocator.alloc(trunk.KVCacheEntry, 0), .step = 0, .allocator = allocator, .config = trunk.KVQuantConfig.dense },
        .last_used = used,
        .quant_config = @import("kv_quant.zig").KVQuantConfig.dense,
        .kv_bytes = tokens.len * 10,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
    };
}

fn admissionTestModel(cfg: *ModelConfig, ram: bool, model: *model_registry_mod.LoadedModel) void {
    model.* = undefined;
    model.config = cfg;
    model.handle = null;
    model.mtp = null;
    model.prefix_cache = if (ram) @import("prefix_cache.zig").HotPrefixCache.initWithMem(testing.allocator, @import("prefix_cache.zig").unlimited_entries, 0) else null;
}

test "a request a cap holds says which cap in its queue line, and one only behind its model's earlier request says nothing" {
    const t = testing;
    var buf: [192]u8 = undefined;
    const A: usize = 0xA0;
    const cands = [_]sched_policy.AdmitCand{
        .{ .model = A, .exclusive = false, .live = 1, .max_concurrent = 2, .reservation = 10 },
        .{ .model = A, .exclusive = false, .live = 1, .max_concurrent = 2, .reservation = 10 },
    };
    const verdicts = [_]sched_policy.AdmitVerdict{ .admit, .wait };
    try t.expectEqualStrings("[queue] waiting for max_concurrent 2: 2 of this model's requests run", capWaitLine(&buf, cands[1], &cands, &verdicts).?);
    const kv = [_]sched_policy.AdmitCand{.{ .model = A, .exclusive = false, .reserved = 3410, .reservation = 3410, .max_kv_tokens = 6000 }};
    try t.expectEqualStrings("[queue] waiting for max_kv_tokens 6000: 3410 tokens held, this request reserves 3410", capWaitLine(&buf, kv[0], &kv, &.{.wait}).?);
    const fifo = [_]sched_policy.AdmitCand{.{ .model = A, .exclusive = false, .reserved = 10, .reservation = 10, .max_kv_tokens = 6000 }};
    try t.expect(capWaitLine(&buf, fifo[0], &fifo, &.{.wait}) == null);
}

test "with max_kv_tokens set, a candidate evicts the oldest retained entry and admits instead of waiting" {
    const a = testing.allocator;
    var cfg = ModelConfig{ .model_type = "qwen4_exp", .num_hidden_layers = 0 };
    cfg.max_concurrent = 4;
    cfg.max_kv_tokens = 200;
    var model: model_registry_mod.LoadedModel = undefined;
    admissionTestModel(&cfg, true, &model);
    defer model.prefix_cache.?.deinit();
    const hc = &model.prefix_cache.?;
    try hc.entries.append(a, try admissionTestEntry(a, &.{ 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030 }, 1));
    try hc.entries.append(a, try admissionTestEntry(a, &.{ 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030 }, 2));
    var sch = submitTestScheduler(a, 8);
    defer sch.pending.deinit(a);
    defer sch.decoding.deinit(a);
    defer sch.cleanup_queue.deinit(a);
    defer sch.prefilling.deinit(a);
    var prompt: [80]u32 = undefined;
    for (&prompt, 0..) |*p, i| p.* = @intCast(i + 1);
    const slot = try sch.submit(.{ .model = &model, .prompt_ids = &prompt, .sampling = .{}, .eos_token_ids = &.{}, .max_tokens = 70 });
    defer {
        sch.complete(slot);
        for (sch.cleanup_queue.items) |s| s.deinit();
    }
    var planned: [16]*Slot = undefined;
    var refused: [16]*Slot = undefined;
    const tick = admitWithinCaps(&sch, &planned, &refused);
    try testing.expectEqual(@as(usize, 1), tick.admits);
    try testing.expectEqual(@as(usize, 0), tick.refused);
    try testing.expect(planned[0] == slot);
    // The oldest entry paid; the newer one stays.
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(u32, 2001), hc.entries.items[0].tokens[0]);
}

test "a candidate over max_kv_tokens alone is refused once, naming the cap and its reservation" {
    const a = testing.allocator;
    var cfg = ModelConfig{ .model_type = "qwen4_exp", .num_hidden_layers = 0 };
    cfg.max_concurrent = 4;
    cfg.max_kv_tokens = 200;
    var model: model_registry_mod.LoadedModel = undefined;
    admissionTestModel(&cfg, false, &model);
    var sch = submitTestScheduler(a, 8);
    defer sch.pending.deinit(a);
    defer sch.decoding.deinit(a);
    defer sch.cleanup_queue.deinit(a);
    defer sch.prefilling.deinit(a);
    var prompt: [120]u32 = undefined;
    for (&prompt, 0..) |*p, i| p.* = @intCast(i + 1);
    const slot = try sch.submit(.{ .model = &model, .prompt_ids = &prompt, .sampling = .{}, .eos_token_ids = &.{}, .max_tokens = 100 });
    defer {
        sch.complete(slot);
        for (sch.cleanup_queue.items) |s| s.deinit();
    }
    var planned: [16]*Slot = undefined;
    var refused: [16]*Slot = undefined;
    const tick = admitWithinCaps(&sch, &planned, &refused);
    try testing.expectEqual(@as(usize, 0), tick.admits);
    try testing.expectEqual(@as(usize, 1), tick.refused);
    try testing.expect(refused[0] == slot);
    try testing.expect(slot.errorNameIs("PrefillDoesNotFit"));
    try testing.expect(std.mem.indexOf(u8, slot.forward_refusal.?.text(), "max_kv_tokens 200") != null);
    try testing.expect(std.mem.indexOf(u8, slot.forward_refusal.?.text(), "220 tokens") != null);
}

test "the caps pick the candidates and memory admits from them: two that fit alone admit one, with their hold billed" {
    const a = testing.allocator;
    const Probe = struct {
        fn numbers(_: *const ModelConfig, _: usize, _: u32, _: trunk.KVQuantConfig, _: bool, _: bool) MemoryBill {
            return .{ .needed = 8, .available = 10, .commit = 6, .held = 5 };
        }
    };
    const saved = scheduler_mod.prefill_admission_numbers;
    scheduler_mod.prefill_admission_numbers = Probe.numbers;
    defer scheduler_mod.prefill_admission_numbers = saved;
    var cfg = ModelConfig{ .model_type = "qwen4_exp", .num_hidden_layers = 0 };
    cfg.max_concurrent = 4;
    cfg.max_kv_tokens = 0;
    var model: model_registry_mod.LoadedModel = undefined;
    model.config = &cfg;
    model.handle = null;
    model.prefix_cache = null;
    model.mtp = null;
    var prompt = [_]u32{1};
    var slots: [2]Slot = undefined;
    for (&slots, 0..) |*s, i| {
        s.model = &model;
        s.full_prompt = prompt[0..];
        s.max_tokens = 1;
        s.cache = .{ .entries = &.{}, .step = 0, .allocator = a, .config = trunk.KVQuantConfig.dense };
        s.request = null;
        s.vision_embeddings = null;
        s.deferred_media = null;
        s.enable_mtp = false;
        s.enable_drafter = false;
        s.dflash = null;
        s.legacy_gen = null;
        s.memory_hold_logged = true;
        s.queue_logged = true;
        s.cancelled = std.atomic.Value(bool).init(false);
        s.finished = false;
        s.error_code = null;
        s.growth_commit = 0;
        s.admitted_held = 0;
        s.restored_entry = 0;
        if (i == 1) prompt[0] = 2;
    }
    var sch = submitTestScheduler(a, 8);
    defer sch.pending.deinit(a);
    defer sch.decoding.deinit(a);
    defer sch.cleanup_queue.deinit(a);
    defer sch.prefilling.deinit(a);
    try sch.pending.append(a, &slots[0]);
    try sch.pending.append(a, &slots[1]);
    var planned: [16]*Slot = undefined;
    var refused: [16]*Slot = undefined;
    const tick = admitWithinCaps(&sch, &planned, &refused);
    try testing.expectEqual(@as(usize, 1), tick.admits);
    try testing.expect(planned[0] == &slots[0]);
    try testing.expectEqual(@as(u64, 5), slots[0].admitted_held);
    try testing.expectEqual(@as(u64, 6), slots[0].growth_commit);
    try testing.expectEqual(@as(u64, 0), slots[1].admitted_held);
}

fn kvCapSlot(model: *model_registry_mod.LoadedModel, prompt: []u32, max_tokens: u32, restored_entry: u64) Slot {
    var s: Slot = undefined;
    s.model = model;
    s.full_prompt = prompt;
    s.max_tokens = max_tokens;
    s.cache = .{ .entries = &.{}, .step = 0, .allocator = testing.allocator, .config = trunk.KVQuantConfig.dense };
    s.request = null;
    s.vision_embeddings = null;
    s.deferred_media = null;
    s.enable_mtp = false;
    s.enable_drafter = false;
    s.dflash = null;
    s.legacy_gen = null;
    s.memory_hold_logged = true;
    s.queue_logged = true;
    s.cancelled = std.atomic.Value(bool).init(false);
    s.finished = false;
    s.error_code = null;
    s.growth_commit = 0;
    s.admitted_held = 0;
    s.restored_entry = restored_entry;
    return s;
}

fn kvCapModel(cfg: *ModelConfig, model: *model_registry_mod.LoadedModel) *@import("prefix_cache.zig").HotPrefixCache {
    cfg.max_concurrent = 0;
    model.config = cfg;
    model.handle = null;
    model.mtp = null;
    model.prefix_cache = @import("prefix_cache.zig").HotPrefixCache.initWithMem(testing.allocator, @import("prefix_cache.zig").unlimited_entries, 0);
    return &model.prefix_cache.?;
}

test "a stale last-restored entry never blocks eviction: nothing live, the candidate admits" {
    const a = testing.allocator;
    var cfg = ModelConfig{ .model_type = "qwen4_exp", .num_hidden_layers = 0 };
    cfg.max_kv_tokens = 200;
    var model: model_registry_mod.LoadedModel = undefined;
    const hc = kvCapModel(&cfg, &model);
    defer model.prefix_cache.?.deinit();
    var entry_tokens: [100]u32 = undefined;
    for (&entry_tokens, 0..) |*t, i| t.* = @intCast(i + 1);
    try hc.entries.append(a, try admissionTestEntry(a, &entry_tokens, 1));
    hc.entries.items[0].id = 1;
    hc.last_restored_used = 1; // the PREVIOUS request's restore, stale
    var prompt: [80]u32 = undefined;
    for (&prompt, 0..) |*p, i| p.* = @intCast(101 + i);
    var slot = kvCapSlot(&model, &prompt, 70, 0);
    var sch = submitTestScheduler(a, 8);
    defer sch.pending.deinit(a);
    defer sch.decoding.deinit(a);
    defer sch.cleanup_queue.deinit(a);
    defer sch.prefilling.deinit(a);
    try sch.pending.append(a, &slot);
    var planned: [16]*Slot = undefined;
    var refused: [16]*Slot = undefined;
    const tick = admitWithinCaps(&sch, &planned, &refused);
    try testing.expectEqual(@as(usize, 1), tick.admits);
    try testing.expectEqual(@as(usize, 0), hc.entries.items.len);
}

test "the candidate's own restore match is the last entry evicted, never the first" {
    const a = testing.allocator;
    var cfg = ModelConfig{ .model_type = "qwen4_exp", .num_hidden_layers = 0 };
    cfg.max_kv_tokens = 150;
    var model: model_registry_mod.LoadedModel = undefined;
    const hc = kvCapModel(&cfg, &model);
    defer model.prefix_cache.?.deinit();
    var own: [30]u32 = undefined;
    for (&own, 0..) |*t, i| t.* = @intCast(i + 1);
    try hc.entries.append(a, try admissionTestEntry(a, &own, 1));
    hc.entries.items[0].id = 1;
    var other: [60]u32 = undefined;
    for (&other, 0..) |*t, i| t.* = @intCast(1001 + i);
    try hc.entries.append(a, try admissionTestEntry(a, &other, 2));
    hc.entries.items[1].id = 2;
    var slot = kvCapSlot(&model, &own, 90, 0);
    var sch = submitTestScheduler(a, 8);
    defer sch.pending.deinit(a);
    defer sch.decoding.deinit(a);
    defer sch.cleanup_queue.deinit(a);
    defer sch.prefilling.deinit(a);
    try sch.pending.append(a, &slot);
    var planned: [16]*Slot = undefined;
    var refused: [16]*Slot = undefined;
    const tick = admitWithinCaps(&sch, &planned, &refused);
    try testing.expectEqual(@as(usize, 1), tick.admits);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(u32, 1), hc.entries.items[0].tokens[0]);
}

test "a later candidate's reclaim never evicts the entry a candidate admitted in the same tick restores from" {
    const a = testing.allocator;
    var cfg = ModelConfig{ .model_type = "qwen4_exp", .num_hidden_layers = 0 };
    cfg.max_kv_tokens = 250;
    var model: model_registry_mod.LoadedModel = undefined;
    const hc = kvCapModel(&cfg, &model);
    defer model.prefix_cache.?.deinit();
    var first: [80]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast(i + 1);
    try hc.entries.append(a, try admissionTestEntry(a, first[0..50], 1));
    hc.entries.items[0].id = 1;
    var other: [60]u32 = undefined;
    for (&other, 0..) |*t, i| t.* = @intCast(1001 + i);
    try hc.entries.append(a, try admissionTestEntry(a, &other, 2));
    hc.entries.items[1].id = 2;
    var second: [80]u32 = undefined;
    for (&second, 0..) |*t, i| t.* = @intCast(5001 + i);
    var slot_a = kvCapSlot(&model, &first, 20, 0);
    var slot_b = kvCapSlot(&model, &second, 70, 0);
    var sch = submitTestScheduler(a, 8);
    defer sch.pending.deinit(a);
    defer sch.decoding.deinit(a);
    defer sch.cleanup_queue.deinit(a);
    defer sch.prefilling.deinit(a);
    try sch.pending.append(a, &slot_a);
    try sch.pending.append(a, &slot_b);
    var planned: [16]*Slot = undefined;
    var refused: [16]*Slot = undefined;
    const tick = admitWithinCaps(&sch, &planned, &refused);
    // 100 + 150 reserved fills the 250 cap once the unrelated 60 goes; the first request's 50-token match stays.
    try testing.expectEqual(@as(usize, 2), tick.admits);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(u32, 1), hc.entries.items[0].tokens[0]);
}

test "an entry a live slot restored from counts once and is never evicted" {
    const a = testing.allocator;
    var cfg = ModelConfig{ .model_type = "qwen4_exp", .num_hidden_layers = 0 };
    cfg.max_kv_tokens = 200;
    var model: model_registry_mod.LoadedModel = undefined;
    const hc = kvCapModel(&cfg, &model);
    defer model.prefix_cache.?.deinit();
    var live_tokens: [60]u32 = undefined;
    for (&live_tokens, 0..) |*t, i| t.* = @intCast(1001 + i);
    try hc.entries.append(a, try admissionTestEntry(a, &live_tokens, 1)); // oldest
    hc.entries.items[0].id = 1;
    var other: [30]u32 = undefined;
    for (&other, 0..) |*t, i| t.* = @intCast(2001 + i);
    try hc.entries.append(a, try admissionTestEntry(a, &other, 2));
    hc.entries.items[1].id = 2;
    var live_prompt: [50]u32 = undefined;
    for (&live_prompt, 0..) |*p, i| p.* = @intCast(3001 + i);
    var live = kvCapSlot(&model, &live_prompt, 50, 1);
    var prompt: [80]u32 = undefined;
    for (&prompt, 0..) |*p, i| p.* = @intCast(4001 + i);
    var slot = kvCapSlot(&model, &prompt, 10, 0);
    var sch = submitTestScheduler(a, 8);
    defer sch.pending.deinit(a);
    defer sch.decoding.deinit(a);
    defer sch.cleanup_queue.deinit(a);
    defer sch.prefilling.deinit(a);
    try sch.decoding.append(a, &live);
    try sch.pending.append(a, &slot);
    var planned: [16]*Slot = undefined;
    var refused: [16]*Slot = undefined;
    const tick = admitWithinCaps(&sch, &planned, &refused);
    try testing.expectEqual(@as(usize, 1), tick.admits);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(u32, 1001), hc.entries.items[0].tokens[0]);
}

test "the reserved-token gauge is what /props reads, published without touching the hot cache" {
    const a = testing.allocator;
    var cfg = ModelConfig{ .model_type = "qwen4_exp", .num_hidden_layers = 0 };
    cfg.max_kv_tokens = 1000;
    var model: model_registry_mod.LoadedModel = undefined;
    const hc = kvCapModel(&cfg, &model);
    defer model.prefix_cache.?.deinit();
    var entry_tokens: [40]u32 = undefined;
    for (&entry_tokens, 0..) |*t, i| t.* = @intCast(1001 + i);
    try hc.entries.append(a, try admissionTestEntry(a, &entry_tokens, 1));
    var prompt: [60]u32 = undefined;
    for (&prompt, 0..) |*p, i| p.* = @intCast(3001 + i);
    var slot = kvCapSlot(&model, &prompt, 40, 0);
    var sch = submitTestScheduler(a, 8);
    defer sch.pending.deinit(a);
    defer sch.decoding.deinit(a);
    defer sch.cleanup_queue.deinit(a);
    defer sch.prefilling.deinit(a);
    try sch.decoding.append(a, &slot);
    publishModelReserved(&sch, &model);
    try testing.expectEqual(@as(u64, 60 + 40 + 40), model.reserved_tokens.load(.monotonic));
}
