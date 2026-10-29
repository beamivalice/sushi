const std = @import("std");
const mlx = @import("mlx.zig");
const generate_mod = @import("generate.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const metrics_mod = @import("metrics.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const disk_placement = @import("disk_placement.zig");
const model_registry_mod = @import("model_registry.zig");
const log = @import("log.zig");
const Generator = generate_mod.Generator;
const LoadedModel = model_registry_mod.LoadedModel;
const degenerate = @import("degenerate.zig");
const testing = std.testing;
const scheduler_mod = @import("scheduler.zig");
const Scheduler = scheduler_mod.Scheduler;
const Slot = scheduler_mod.Slot;
const commitSlotIfApplicable = scheduler_mod.commitSlotIfApplicable;
const expectOutcomes = scheduler_mod.expectOutcomes;
const publishHotCacheResidency = scheduler_mod.publishHotCacheResidency;
const releaseNativeState = scheduler_mod.releaseNativeState;
const specTickMode = scheduler_mod.specTickMode;

/// Phase A6: finalize a slot. Commits to hot prefix cache (if applicable)
/// before signaling completion. The order matters: commit first while the
/// slot is alive, then markFinished — the conn thread's waitNext might
/// return immediately after the broadcast and call complete()→deinit, so
/// we cannot reach into the slot afterwards.
/// Pure decision for the degenerate-tail-loop guard: when the generated tail
/// has collapsed into a short repeating cycle, returns the finish reason to
/// cut the request with; null while generation is healthy.
///
/// The reason is "stop", not "length": this guard fired far below the requested
/// output cap, and clients such as pi treat "length" as context-overflow recovery
/// and compact unnecessarily. The server separately suppresses tool-call parsing
/// for this cause, so a cut fragment cannot be promoted to a completed call.
pub fn loopStopReason(generated_ids: []const u32) ?[]const u8 {
    const d = loopStopDecision(generated_ids) orelse return null;
    return d.finish_reason;
}

/// What a loop cut tells the rest of the server. `finish_reason` is the wire
/// value (always "stop" — see above); `finish_details` is the sibling
/// signal that names the CAUSE, and `trim_start` is where the client's copy
/// of the answer should end.
pub const LoopStop = struct {
    finish_reason: []const u8 = "stop",
    /// The `finish_details.type` value. One string for all three tiers: a
    /// client's decision ("this turn is unusable, don't feed it back") is the
    /// same whichever tier convicted, and the tier is in the log.
    finish_details: []const u8 = "repetition_loop",
    tier: degenerate.DegenerateTail.Tier,
    trim_start: usize,
};

/// Pure decision + trim point. The three tiers live in degenerate.zig
/// (`degenerateTail`); this is where their verdict becomes server behaviour.
pub fn loopStopDecision(generated_ids: []const u32) ?LoopStop {
    const d = degenerate.degenerateTail(generated_ids) orelse return null;
    return .{ .tier = d.tier, .trim_start = d.start };
}

var loop_trim_env: ?bool = null;
/// `SUSHI_LOOP_TRIM=0` keeps the whole degenerate tail in the response —
/// the A/B arm, and the escape hatch for anyone who needs to see exactly what
/// the model emitted. The cut itself is unaffected either way.
pub fn loopTrimEnabled() bool {
    if (loop_trim_env) |v| return v;
    const raw = std.c.getenv("SUSHI_LOOP_TRIM");
    const enabled = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    loop_trim_env = enabled;
    return enabled;
}

/// The terminator a finishing slot publishes, and the only caller of `Slot.markFinished`.
/// `latched` is a peek (`mlx.peekErrorName`): a decode forward that failed can still hand
/// back a plausible EOS, and the request must not finish 200 on it. The tick wrapper still
/// owns the latch and fails the rest of a batched group with it.
fn publishSlotTerminator(slot: anytype, reason: []const u8, latched: ?[]const u8) void {
    if (latched) |name| {
        slot.markError(name);
        return;
    }
    slot.markFinished(reason);
}

/// Decoded bytes of the answer that ride the `[short-gen]` line.
const SHORT_GEN_TEXT_CAP = 200;

/// `emitted` (what the client got) and `realized` (what the generator appended) diverge only on a block path.
fn formatShortGen(
    out: []u8,
    reason: []const u8,
    emitted: u32,
    ids: []const u32,
    text: []const u8,
    path: []const u8,
) []const u8 {
    var n: usize = 0;
    n += (std.fmt.bufPrint(out[n..], "[short-gen] reason={s} emitted={d} realized={d} ids=[", .{ reason, emitted, ids.len }) catch return out[0..n]).len;
    for (ids, 0..) |id, i| {
        const sep: []const u8 = if (i == 0) "" else ",";
        n += (std.fmt.bufPrint(out[n..], "{s}{d}", .{ sep, id }) catch break).len;
    }
    n += (std.fmt.bufPrint(out[n..], "] bytes=", .{}) catch return out[0..n]).len;
    n = log.appendQuoted(out, n, text, SHORT_GEN_TEXT_CAP);
    n += (std.fmt.bufPrint(out[n..], " path={s}", .{path}) catch return out[0..n]).len;
    return out[0..n];
}

/// Decoded bytes of a token that ride the `[stop]` line.
const STOP_TEXT_CAP = 48;

const StopLine = struct {
    kind: []const u8,
    token: u32,
    token_text: []const u8,
    after: ?u32,
    after_text: []const u8,
    completion: u32,
    path: []const u8,
};

/// Which token ended a turn that finished "stop": the logs otherwise cannot tell an EOS from a pad run or a stop sequence.
fn formatStop(out: []u8, line: StopLine) []const u8 {
    var n: usize = 0;
    n += (std.fmt.bufPrint(out[n..], "[stop] reason={s} token={d} ", .{ line.kind, line.token }) catch return out[0..n]).len;
    n = log.appendQuoted(out, n, line.token_text, STOP_TEXT_CAP);
    if (line.after) |id| {
        n += (std.fmt.bufPrint(out[n..], " after={d} ", .{id}) catch return out[0..n]).len;
    } else {
        n += (std.fmt.bufPrint(out[n..], " after=none ", .{}) catch return out[0..n]).len;
    }
    n = log.appendQuoted(out, n, line.after_text, STOP_TEXT_CAP);
    n += (std.fmt.bufPrint(out[n..], " completion={d} path={s}", .{ line.completion, line.path }) catch return out[0..n]).len;
    return out[0..n];
}

/// An EOS ends the turn by itself; token 0 ends it only as the third pad in a row.
fn stopKind(token: u32, eos: []const u32) []const u8 {
    return if (generate_mod.isEosId(token, eos)) "eos" else "pad-run";
}

/// Names the token that ended a turn finishing "stop" on an EOS or a pad run. Decodes into stack
/// scratch: a log line never allocates and never fails the request.
fn logStop(slot: *Slot, reason: []const u8) void {
    if (!std.mem.eql(u8, reason, "stop")) return;
    const gen: ?*Generator = if (slot.legacy_gen) |*g| g else null;
    const token = slot.stop_token orelse (if (gen) |g| g.stop_token else null) orelse return;
    const after: ?u32 = blk: {
        slot.out_mu.lockUncancelable(slot.io);
        defer slot.out_mu.unlock(slot.io);
        break :blk slot.out_buf.getLastOrNull();
    };
    var scratch: [512]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&scratch);
    const tok = slot.model.tokenizer;
    const token_text: []const u8 = if (tok) |t| (t.decode(fba.allocator(), &.{token}) catch "") else "";
    const after_text: []const u8 = if (tok != null and after != null) (tok.?.decode(fba.allocator(), &.{after.?}) catch "") else "";
    var path = slot.stop_path;
    if (path.len == 0) {
        path = if (gen) |g| (if (specTickMode(slot.enable_mtp, g.mtp != null or g.dspark, slot.enable_drafter, g.dflash != null) == .regular) "serial" else "spec") else "serial";
    }
    var buf: [384]u8 = undefined;
    log.info("{s}\n", .{formatStop(&buf, .{
        .kind = stopKind(token, slot.eos_token_ids),
        .token = token,
        .token_text = token_text,
        .after = after,
        .after_text = after_text,
        .completion = slot.completion_tokens,
        .path = path,
    })});
}

/// A one-word answer and a block-path handling bug look identical until the ids are named.
fn logShortGen(slot: *Slot, reason: []const u8) void {
    const gen = if (slot.legacy_gen) |*g| g else return;
    const mode = specTickMode(
        slot.enable_mtp,
        gen.mtp != null or gen.dspark,
        slot.enable_drafter,
        gen.dflash != null,
    );
    const ids = gen.generated_ids.items;
    const decoded: ?[]u8 = if (slot.model.tokenizer) |tok|
        (tok.decode(slot.allocator, ids) catch null)
    else
        null;
    defer if (decoded) |d| slot.allocator.free(d);
    var buf: [1024]u8 = undefined;
    log.info("{s}\n", .{formatShortGen(
        &buf,
        reason,
        slot.completion_tokens,
        ids,
        decoded orelse "",
        if (mode == .regular) "serial" else @tagName(mode),
    )});
}

/// Outcome of a slot that reached `finishSlot`. A latched MLX failure turns the finish into an error.
fn finishOutcome(reason: []const u8, latched: ?[]const u8) metrics_mod.Outcome {
    if (latched != null) return .failed;
    if (std.mem.eql(u8, reason, "cancelled")) return .cancelled;
    return .success;
}

/// Outcome of a slot the cleanup drain sees without `finishSlot` having recorded it, read only from
/// state the inference thread set. `cancelled` is never consulted: `Scheduler.complete` sets it on
/// every completion, normal ones included.
fn cleanupOutcome(slot: anytype) metrics_mod.Outcome {
    if (slot.error_code != null) return .failed;
    if (slot.finished) return finishOutcome(slot.finish_reason, null);
    if (slot.stop_hit.load(.acquire)) return .success;
    return .cancelled;
}

/// Count a slot's outcome once; the first path to reach the slot wins.
fn recordSlotEnd(metrics: ?*metrics_mod.Metrics, slot: anytype, outcome: metrics_mod.Outcome) void {
    const m = metrics orelse return;
    if (slot.metrics_recorded) return;
    slot.metrics_recorded = true;
    m.recordRequest(
        outcome,
        slot.first_token_ns,
        slot.prefill_ns,
        slot.decode_ns,
        slot.prompt_tokens,
        slot.completion_tokens,
        slot.cached_tokens,
    );
}

/// The cleanup drain's count: a slot refused before its first forward is a rejection; anything
/// `finishSlot` already counted is skipped.
pub fn recordSlotCleanup(metrics: ?*metrics_mod.Metrics, slot: anytype) void {
    const m = metrics orelse return;
    if (slot.metrics_recorded) return;
    if (slot.error_code) |name| if (std.mem.eql(u8, name, "PrefillDoesNotFit") or std.mem.eql(u8, name, "MediaRefused")) {
        slot.metrics_recorded = true;
        m.recordRejected();
        return;
    };
    recordSlotEnd(m, slot, cleanupOutcome(slot));
}

pub fn finishSlot(sch: *Scheduler, slot: *Slot, reason: []const u8) void {
    // Emit the `[spec-stats]` summary (no-op for non-speculative slots).
    // The legacy generate() path logs this itself; scheduler-driven slots
    // finalize here instead.
    if (slot.legacy_gen) |*g| {
        g.logSpecStats();
        // `[qsa-arms]` rides the same seam: the SERVE path finalizes here, so
        // wiring it only beside generate.zig's own logSpecStats() calls (the
        // legacy/CLI path) makes it dead on every served request.
        g.logQsaArms();
        g.persistRoundCost();
    }
    const latched: ?[]const u8 = mlx.peekErrorName();
    if (latched) |name| {
        log.err("[scheduler] finish suppressed: the last forward failed ({s}) but produced a \"{s}\" — failing this request rather than answering 200 with what Metal never wrote\n", .{ name, reason });
    }
    if (slot.completion_tokens <= 2) logShortGen(slot, reason);
    if (latched == null) logStop(slot, reason);
    commitSlotIfApplicable(sch, slot);
    // Restore by move: a slot that ended without committing still holds its checkout, and
    // the record now describes bytes that die with `slot.cache`. Above every early return.
    if (slot.model.prefix_cache) |*hc| hc.releaseCheckout(@intFromPtr(slot), reason);
    releaseNativeState(slot);
    // SSD flush runs AFTER markFinished so the client never waits on the
    // chunk-append — but everything it needs must be captured BEFORE the
    // broadcast: the conn thread may complete()+free the slot immediately.
    // The prefix cache and stream live on the registry-owned LoadedModel,
    // which outlives the slot (unload also runs on this thread).
    const hc_opt: ?*prefix_cache_mod.HotPrefixCache =
        if (slot.model.prefix_cache) |*p| p else null;
    const stream_opt: ?mlx.mlx_stream =
        if (slot.model.transformer) |x| x.s else null;
    // Record per-request metrics while slot fields are still live (before markFinished broadcasts:
    // the conn thread may complete()+free the slot immediately after). real_ttft = first_token_ns
    // (queue+prefill, captured at prefill completion); recordRequest derives e2e = first_token_ns + decode_ns.
    recordSlotEnd(sch.metrics, slot, finishOutcome(reason, latched));
    publishSlotTerminator(slot, reason, latched);
    if (hc_opt) |hc| {
        if (stream_opt) |s| {
            hc.flushPendingDisk(s);
            // After the flush, so this turn's entry is the MRU one and any entry spilled
            // already has a complete copy to spill into.
            hc.spillIdleEntries(s);
            publishHotCacheResidency(sch);
            logDiskUsage(slot.model);
        }
    }
    // Return this turn's transients to the OS. The per-`CACHE_CLEAR_INTERVAL`
    // clear inside `Generator.advanceStep` can't cover the tail of a turn, and
    // MLX parks freed buffers in a size-keyed pool rather than releasing them —
    // so without this a short turn hands everything it stranded to the next one
    // and the process footprint ratchets across a session (issue #110).
    _ = mlx.mlx_clear_cache();
}

/// One `[disk-cache] usage` line per finished turn, from the finishing model's own tier.
fn logDiskUsage(model: *const LoadedModel) void {
    const snap = model.disk_stats.snapshot();
    if (snap.budget == 0) return;
    var buf: [128]u8 = undefined;
    log.info("{s}\n", .{disk_placement.diskUsageLine(&buf, snap.used, snap.budget, snap.entries)});
}

test "a finish over a latched MLX failure ends the request as an ERROR, never a 200" {
    const Stub = struct {
        finished: ?[]const u8 = null,
        errored: ?[]const u8 = null,
        fn markFinished(self: *@This(), reason: []const u8) void {
            self.finished = reason;
        }
        fn markError(self: *@This(), name: []const u8) void {
            self.errored = name;
        }
    };

    var clean = Stub{};
    publishSlotTerminator(&clean, "stop", null);
    try testing.expectEqualStrings("stop", clean.finished.?);
    try testing.expect(clean.errored == null);

    // Latched: no "stop" is published at all.
    var poisoned = Stub{};
    publishSlotTerminator(&poisoned, "stop", "OutOfMemory");
    try testing.expect(poisoned.finished == null);
    try testing.expectEqualStrings("OutOfMemory", poisoned.errored.?);
    try testing.expect(Slot.errorNameIsMemory(poisoned.errored.?));

    var shape = Stub{};
    publishSlotTerminator(&shape, "length", "MlxFailure");
    try testing.expect(shape.finished == null);
    try testing.expectEqualStrings("MlxFailure", shape.errored.?);
    try testing.expect(!Slot.errorNameIsMemory(shape.errored.?));
}

const OutcomeStub = struct {
    finished: bool = false,
    error_code: ?[]const u8 = null,
    finish_reason: []const u8 = "",
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    stop_hit: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    metrics_recorded: bool = false,
    first_token_ns: u64 = 5_000_000,
    prefill_ns: u64 = 4_000_000,
    decode_ns: u64 = 20_000_000,
    prompt_tokens: u32 = 100,
    completion_tokens: u32 = 10,
    cached_tokens: u32 = 0,

    /// `finishSlot` on the inference thread: records, then publishes the terminator.
    fn finish(self: *OutcomeStub, m: ?*metrics_mod.Metrics, reason: []const u8) void {
        recordSlotEnd(m, self, finishOutcome(reason, null));
        self.finished = true;
        self.finish_reason = reason;
    }
    /// `Slot.markError` mid-decode: no finishSlot follows.
    fn fail(self: *OutcomeStub, name: []const u8) void {
        if (self.error_code != null or self.finished) return;
        self.error_code = name;
    }
    /// The connection thread's `Scheduler.complete`: sets `cancelled` on EVERY completion.
    fn complete(self: *OutcomeStub) void {
        self.cancelled.store(true, .release);
    }
};

test "outcome: a disconnect while the request waits in pending counts cancelled once" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 1, 0, 0);
}

test "outcome: a disconnect during prefill is counted once, not again by the drain" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.complete();
    s.finish(&m, "cancelled");
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 1, 0, 0);
}

test "outcome: a disconnect during decode never reaches finishSlot and the drain counts it" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.completion_tokens = 7;
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 1, 0, 0);
    try testing.expectEqual(@as(u64, 0), m.ttft_ns.count.load(.monotonic));
    try testing.expectEqual(@as(u64, 0), m.generation_tokens_total.load());
}

test "outcome: a normal completion is success once even though complete() sets cancelled, with its totals" {
    var m = metrics_mod.Metrics.init();
    for ([_][]const u8{ "stop", "length" }) |reason| {
        var s = OutcomeStub{};
        s.finish(&m, reason);
        s.complete();
        recordSlotCleanup(&m, &s);
    }
    try expectOutcomes(&m, 2, 0, 0, 0);
    try testing.expectEqual(@as(u64, 20), m.generation_tokens_total.load());
    try testing.expectEqual(@as(u64, 200), m.prompt_tokens_total.load());
    try testing.expectEqual(@as(u64, 2), m.e2e_latency_ns.count.load(.monotonic));
    try testing.expectEqual(@as(u64, 50_000_000), m.e2e_latency_ns.sum.load(.monotonic));
}

test "outcome: a mid-decode generation error counts failed and leaves the histograms alone" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.fail("MlxFailure");
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 0, 1, 0);
    try testing.expectEqual(@as(u64, 0), m.ttft_ns.count.load(.monotonic));
    try testing.expectEqual(@as(u64, 0), m.e2e_latency_ns.count.load(.monotonic));
    try testing.expectEqual(@as(u64, 0), m.generation_tokens_total.load());
}

test "outcome: a finish over a latched MLX failure counts failed, not success" {
    try testing.expectEqual(metrics_mod.Outcome.failed, finishOutcome("stop", "OutOfMemory"));
    try testing.expectEqual(metrics_mod.Outcome.success, finishOutcome("stop", null));
    try testing.expectEqual(metrics_mod.Outcome.cancelled, finishOutcome("cancelled", null));
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    recordSlotEnd(&m, &s, finishOutcome("stop", "OutOfMemory"));
    s.fail("OutOfMemory");
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 0, 1, 0);
}

test "outcome: an error followed by a client disconnect is one failure, never two outcomes" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.fail("MlxFailure");
    recordSlotCleanup(&m, &s);
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 0, 1, 0);
}

test "outcome: a slot refused by admission before its first forward is a rejection, not a failure" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.fail("PrefillDoesNotFit");
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 0, 0, 1);
}

test "outcome: metrics off does no work and leaves the slot untouched" {
    var s = OutcomeStub{};
    s.complete();
    recordSlotCleanup(null, &s);
    s.finish(null, "stop");
    try testing.expect(!s.metrics_recorded);
}

test "outcome: a request ended by its own stop sequence is a success, a disconnect stays cancelled" {
    var m = metrics_mod.Metrics.init();
    var stopped = OutcomeStub{};
    stopped.stop_hit.store(true, .release);
    stopped.complete();
    recordSlotCleanup(&m, &stopped);
    try expectOutcomes(&m, 1, 0, 0, 0);
    try testing.expectEqual(@as(u64, 10), m.generation_tokens_total.load());

    var gone = OutcomeStub{};
    gone.complete();
    recordSlotCleanup(&m, &gone);
    try expectOutcomes(&m, 1, 1, 0, 0);

    var stop_then_error = OutcomeStub{};
    stop_then_error.stop_hit.store(true, .release);
    stop_then_error.fail("MlxFailure");
    recordSlotCleanup(&m, &stop_then_error);
    try expectOutcomes(&m, 1, 1, 1, 0);
}

test "loopStopReason: a degenerate tail cut reports stop, a healthy tail is not cut" {
    // A loop guard is an intentional server stop, not exhaustion of the
    // requested output budget. Reporting "length" makes clients such as pi
    // run context-overflow recovery and compact a mostly-empty context. Tool
    // calls from this truncated buffer are suppressed separately at emission.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);

    // Healthy varied tail: never cut.
    for (0..40) |i| try ids.append(testing.allocator, @as(u32, @intCast(i * 7 + 3)));
    try testing.expect(loopStopReason(ids.items) == null);

    // Collapse into a short cycle (the php.html shape: "server-side scripting
    // language, " ≈ a 6-token cycle) past the guard's span threshold.
    for (0..degenerate.degenerate_loop_min_span / 6 + 1) |_| {
        for ([_]u32{ 101, 202, 303, 404, 505, 606 }) |t| {
            try ids.append(testing.allocator, t);
        }
    }
    const reason = loopStopReason(ids.items) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", reason);
}

test "loopStopReason: a LONG-period sentence loop is cut at the second tier" {
    // The 2026-08-02 shooter wrap-up failure: a two-sentence cycle ("The game
    // is complete. Let me do a final review... Let me verify main.js...") of
    // ~58 tokens repeated 26 times sailed through the 8-token-period tier and
    // was never cut. Tier 2 scans periods 9..64 and requires 10 exact
    // repetitions — verbatim-identical long cycles at that count are
    // degeneration, not content.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    for (0..30) |i| try ids.append(testing.allocator, @as(u32, @intCast(i * 3 + 11)));

    // 58-token cycle, one rep short of the tier's span bar: NOT cut.
    var cycle: [58]u32 = undefined;
    for (&cycle, 0..) |*v, i| v.* = @as(u32, @intCast(1000 + i));
    for (0..degenerate.degenerate_loop_long_min_span / cycle.len) |_| try ids.appendSlice(testing.allocator, &cycle);
    try testing.expect(loopStopReason(ids.items) == null);

    // The next repetition crosses it — cut as an intentional stop.
    try ids.appendSlice(testing.allocator, &cycle);
    const reason = loopStopReason(ids.items) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", reason);
}

test "loopStopDecision: the wire reason is stop and the CAUSE rides beside it" {
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    try ids.appendSlice(testing.allocator, &[_]u32{ 5, 6, 7 });
    for (0..degenerate.degenerate_loop_min_span / 3 + 1) |_| {
        try ids.appendSlice(testing.allocator, &[_]u32{ 101, 102, 103 });
    }

    const stop = loopStopDecision(ids.items) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", stop.finish_reason);
    try testing.expectEqualStrings("repetition_loop", stop.finish_details);
    try testing.expectEqual(degenerate.DegenerateTail.Tier.exact_cycle, stop.tier);
    // Trimmed to the honest prefix plus one copy of the cycle.
    try testing.expectEqual(@as(usize, 6), stop.trim_start);

    // Once a deferred boundary has been committed, the old reasoning loop is
    // outside the guard window and cannot stop the very next answer tick.
    try ids.append(testing.allocator, 99);
    const answer_start = ids.items.len;
    try testing.expect(loopStopDecision(ids.items[answer_start..]) == null);

    // A new loop wholly inside the constrained answer retains the existing
    // stop/repetition result, with an absolute trim point for response code.
    for (0..degenerate.degenerate_loop_min_span / 3 + 1) |_| {
        try ids.appendSlice(testing.allocator, &[_]u32{ 7, 8, 9 });
    }
    const answer_loop = loopStopDecision(ids.items[answer_start..]) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", answer_loop.finish_reason);
    try testing.expect(answer_loop.trim_start + answer_start >= answer_start);

    // Healthy output decides nothing at all — no reason, and nothing to trim.
    var healthy: [512]u32 = undefined;
    for (&healthy, 0..) |*v, i| v.* = @intCast(i);
    try testing.expect(loopStopDecision(&healthy) == null);
    try testing.expect(loopStopReason(&healthy) == null);
}

test "loopStopReason: periods past the long tier stay uncut" {
    // A 70-token exact cycle (> long-tier max 64) repeated many times is
    // outside both tiers — the guard stays scoped rather than judging whole
    // repeated paragraphs.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    var cycle: [70]u32 = undefined;
    for (&cycle, 0..) |*v, i| v.* = @as(u32, @intCast(2000 + i));
    for (0..12) |_| try ids.appendSlice(testing.allocator, &cycle);
    try testing.expect(loopStopReason(ids.items) == null);
}

test "[short-gen] carries both token counts, the ids and what they decode to" {
    // Bar: the line carries emitted vs realized and survives quotes, newlines and a 300-byte answer.
    var buf: [1024]u8 = undefined;
    const ids = [_]u32{ 151645, 17 };
    try testing.expectEqualStrings(
        "[short-gen] reason=stop emitted=0 realized=2 ids=[151645,17] bytes=\"hi\" path=mtp",
        formatShortGen(&buf, "stop", 0, &ids, "hi", "mtp"),
    );

    const esc = formatShortGen(&buf, "length", 2, &.{}, "a\"b\nc", "serial");
    try testing.expectEqualStrings(
        "[short-gen] reason=length emitted=2 realized=0 ids=[] bytes=\"a\\\"b\\nc\" path=serial",
        esc,
    );

    var long: [300]u8 = undefined;
    @memset(&long, 'x');
    const capped = formatShortGen(&buf, "stop", 1, &.{}, &long, "pld");
    try testing.expectEqual(@as(usize, SHORT_GEN_TEXT_CAP), std.mem.count(u8, capped, "x"));
}

test "[stop] names the token that ended the turn" {
    // Bar: the line names the token and the one before it, escapes both, and has an honest "none".
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings(
        "[stop] reason=eos token=248046 \"<|im_end|>\" after=1510 \"Ġ`\" completion=2784 path=serial",
        formatStop(&buf, .{ .kind = "eos", .token = 248046, .token_text = "<|im_end|>", .after = 1510, .after_text = "Ġ`", .completion = 2784, .path = "serial" }),
    );
    try testing.expectEqualStrings(
        "[stop] reason=pad-run token=0 \"\" after=none \"\" completion=0 path=batch",
        formatStop(&buf, .{ .kind = "pad-run", .token = 0, .token_text = "", .after = null, .after_text = "", .completion = 0, .path = "batch" }),
    );
    const esc = formatStop(&buf, .{ .kind = "eos", .token = 2, .token_text = "a\"\nb", .after = 7, .after_text = "x", .completion = 1, .path = "spec" });
    try testing.expectEqualStrings("[stop] reason=eos token=2 \"a\\\"\\nb\" after=7 \"x\" completion=1 path=spec", esc);
    try testing.expectEqualStrings("eos", stopKind(248046, &.{ 248046, 248044 }));
    try testing.expectEqualStrings("pad-run", stopKind(0, &.{248046}));
}
