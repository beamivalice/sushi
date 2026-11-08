//! The scheduler's view of an arch. `scheduler.zig` and `slot_admission.zig` keep the ticks, the queues and the
//! admission order; what differs per arch (what batches, what a request owns, how it commits, restores and writes
//! through) is declared by the arch's `sched` module (`<arch>_sched.zig`) and reached here, `inline else` over the
//! enabled archs. A module omits a hook to get the default stated on its dispatch function. Inference thread only,
//! except `facts`, `batches` and `rowCap`, which read a config.
//!
//! Dispatch reads the arch tag of the slot's (or model's) config, never the loaded handle, so a slot built by a
//! test needs only its config.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const registry = @import("registry.zig");
const arch_config = @import("config.zig");
const pc = @import("../prefix_cache.zig");
const kv_quant = @import("../kv_quant.zig");
const transformer = @import("../transformer.zig");
const model_handle = @import("../model_handle.zig");
const trunk = @import("../trunk.zig");
const model_registry = @import("../model_registry.zig");
const generate = @import("../generate.zig");
const scheduler = @import("../scheduler.zig");

const Tag = registry.Tag;
const Slot = scheduler.Slot;
const Scheduler = scheduler.Scheduler;
const Generator = generate.Generator;
const Transformer = trunk.Transformer;
const ForwardCtx = model_handle.ForwardCtx;
const ModelConfig = model.ModelConfig;

/// The request state a slot owns beside its KV cache (`arch_state.SlotRequest`).
pub const NativeRequest = @import("arch_state.zig").SlotRequest;

pub const ThoughtPoint = enum {
    off,
    always,
    /// Only a request that runs the arch's MTP head.
    with_mtp,
};

/// Per-config facts the scheduler reads on every tick, so a slot holds them from admission (`Slot.facts`).
pub const Facts = struct {
    /// Batched rows pad to the attention layers' true kv length, and the pad-waste cap sorts by it.
    long_ctx: bool = false,
    /// With an SSD tier behind the cache, a request lists at most `SSM_RESIDENT_MAX` checkpoints.
    caps_ssm_checkpoints: bool = false,
    thought_point: ThoughtPoint = .off,
    /// Admission bills the request's recurrent checkpoints, and fewer where they do not fit.
    bills_checkpoints: bool = false,
    /// A write-through prefill moves each checkpoint to SSD and drops its RAM copy.
    releases_checkpoints: bool = false,
    /// A restore point is a full copy of the state, so it goes to SSD however short the new span.
    always_write_through: bool = false,
    /// The rows one batched forward carries, when the arch bounds it (`rowCap`).
    row_cap: ?usize = null,
    /// A batched group's forward pads every slot to the longest (`padsBatch`).
    pads_batch: bool = true,
};

pub const TickClass = @import("../sched_policy.zig").TickClass;

/// The batched kernel a group takes this tick. `arm` is the arch's own id for it, handed back to `batchedForward`.
pub const BatchArm = struct {
    arm: u8 = 0,
    /// Rows are positioned by each slot's `moe_seq_offset`, not by `cache.step`.
    moe_positions: bool = true,
    /// The batched forward leaves each slot's `moe_seq_offset` where it was; the tick moves it by one.
    advances_offset: bool = true,
};

/// Where a hot-cache lookup restores an arch whose state is not the slot's SSM entries.
pub const RestoreTarget = struct {
    rec: pc.Recurrent,
    /// The restore carries the assistant context.
    takes_dflash: bool = false,
    /// The restore carries the MTP head's history.
    takes_mtp: bool = false,
    /// The restore log line's prefix.
    log_tag: []const u8 = "[prefix]",
};

/// What a loaded model's cold-load bill needs from the loader.
pub const ColdLoad = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    config: *ModelConfig,
    path: []const u8,
    load_vision: bool,
    mtp_on: bool,
    no_drafter: bool,
    drafter_dir: []const u8,
};

/// The committed MTP head of an arch whose head rides the trunk's own cache.
pub const HeadParts = struct {
    entry: ?*const trunk.SSMCacheEntry = null,
    pos_base: c_int = 0,
    marks: []const transformer.QsaHeadMark = &.{},
};

fn S(comptime t: Tag) type {
    return registry.Mod(t).sched;
}

fn has(comptime t: Tag, comptime name: []const u8) bool {
    return @hasDecl(S(t), name);
}

fn tagOfConfig(config: ?*const ModelConfig) ?Tag {
    return arch_config.tagOf(config orelse return null);
}

fn tagOfSlot(slot: *const Slot) ?Tag {
    if (slot.model.transformer) |xfm| return tagOfModel(xfm);
    return tagOfConfig(slot.model.config);
}

fn tagOfModel(xfm: *const Transformer) ?Tag {
    return tagOfConfig(&xfm.config);
}

pub fn facts(config: *const ModelConfig) Facts {
    const tag = tagOfConfig(config) orelse return .{};
    var f: Facts = switch (tag) {
        inline else => |t| if (comptime has(t, "facts")) S(t).facts(config) else .{},
    };
    f.row_cap = rowCap(config);
    f.pads_batch = padsBatch(config);
    return f;
}

/// Does this config decode concurrent slots as one batch?
pub fn batches(config: *const ModelConfig) bool {
    const tag = tagOfConfig(config) orelse return false;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "batches")) S(t).batches(config) else false,
    };
}

/// The rows one batched forward carries for this config, when the arch bounds it below `MAX_BATCH_GROUP`.
pub fn rowCap(config: *const ModelConfig) ?usize {
    const tag = tagOfConfig(config) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "rowCap")) S(t).rowCap(config) else null,
    };
}

/// The loaded model batches (the config's answer, and the layers it built).
pub fn modelBatches(xfm: *const Transformer) bool {
    const tag = tagOfModel(xfm) orelse return false;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "modelBatches")) S(t).modelBatches(xfm) else false,
    };
}

/// Whether the slot whose context this is can merge into this tick's batch; one that cannot yet decodes serially.
pub fn readyToBatch(xfm: *const Transformer, ctx: *ForwardCtx) bool {
    const tag = tagOfModel(xfm) orelse return true;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "readyToBatch")) S(t).readyToBatch(xfm, ctx) else true,
    };
}

/// A batched group's slots each carry their own length; the forward pads them to the longest unless the arch's
/// rows read each slot's own cache.
pub fn padsBatch(config: *const ModelConfig) bool {
    const tag = tagOfConfig(config) orelse return true;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "padsBatch")) S(t).padsBatch(config) else true,
    };
}

/// The kernel the group takes this tick, null when the arch has none ready.
pub fn batchArm(xfm: *Transformer, ctxs: []const *ForwardCtx) ?BatchArm {
    const tag = tagOfModel(xfm) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "batchArm")) S(t).batchArm(xfm, ctxs) else null,
    };
}

/// One batched forward over `batch`. Returns each remaining row's logits, in the order of the (possibly shortened)
/// `batch`.
pub fn batchedForward(
    sch: *Scheduler,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    batch: *[]*Slot,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
    arm: BatchArm,
    hidden_rows: ?*?[]mlx.mlx_array,
) ![]mlx.mlx_array {
    const tag = tagOfModel(xfm) orelse return error.NoBatchedKernel;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "batchedForward"))
            S(t).batchedForward(sch, allocator, xfm, batch, next_tokens, ctxs, rope_offsets, arm, hidden_rows)
        else
            error.NoBatchedKernel,
    };
}

/// Plain decode rows are priced by the grouped-MTP planner as they run (its cost table learns from them).
pub fn plannerObservesPlainRows(xfm: *const Transformer) bool {
    const tag = tagOfModel(xfm) orelse return false;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "plannerObservesPlainRows")) S(t).plannerObservesPlainRows(xfm) else false,
    };
}

pub fn tickClass(slot: *const Slot) TickClass {
    const tag = tagOfSlot(slot) orelse return .none;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "tickClass")) S(t).tickClass(slot) else .none,
    };
}

/// Can this slot's MTP round join the grouped verify?
pub fn mtpGroupable(slot: *const Slot) bool {
    const tag = tagOfSlot(slot) orelse return false;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "mtpGroupable")) S(t).mtpGroupable(slot) else false,
    };
}

/// Does another slot of this one's model decode this tick (a drafting slot or a plain batchable one)?
pub fn rowsInCompany(slot: *const Slot, drafting: []const *Slot, plain: []const *Slot) bool {
    var company: usize = 0;
    for (drafting) |other| company += @intFromBool(other.model == slot.model);
    for (plain) |other| company += @intFromBool(other.model == slot.model);
    return company >= 2;
}

/// The state a new slot owns beside its KV cache; null where the arch keeps none per request.
pub fn initRequest(allocator: std.mem.Allocator, config: *const ModelConfig) !?NativeRequest {
    const tag = tagOfConfig(config) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "initRequest")) try S(t).initRequest(allocator, config) else null,
    };
}

/// The arch refuses a KV scheme it does not hold.
pub fn checkKvQuant(config: *const ModelConfig, kv: kv_quant.KVQuantConfig) !void {
    const tag = tagOfConfig(config) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "checkKvQuant")) try S(t).checkKvQuant(kv),
    }
}

/// The arch's cold-load bill; null leaves the generic one.
pub fn coldLoadBill(load: ColdLoad) !?u64 {
    const tag = tagOfConfig(load.config) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "coldLoadBill")) try S(t).coldLoadBill(load) else null,
    };
}

/// Resident bytes of the request state the slot owns beside its KV cache.
pub fn requestBytes(slot: *const Slot) u64 {
    return if (slot.request) |*request| request.residentBytes() else 0;
}

/// Bytes of request state that live in the model, not in the slot.
pub fn modelStateBytes(slot: *const Slot) u64 {
    const tag = tagOfSlot(slot) orelse return 0;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "modelStateBytes")) S(t).modelStateBytes(slot) else 0,
    };
}

/// A slot the inference thread stops touching releases the state a later admission would otherwise read as held.
pub fn releaseState(slot: *Slot) void {
    if (slot.request) |*request| request.reset();
}

/// A cancelled decode-phase slot commits at the cull when its request state is reset right after, or is the model's
/// own (an arch's `commitsAtCull`): the next exclusive request takes that state over.
pub fn commitsAtCull(slot: *const Slot) bool {
    if (slot.request != null) return true;
    const tag = tagOfSlot(slot) orelse return false;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "commitsAtCull")) S(t).commitsAtCull(slot) else false,
    };
}

/// Readies the slot's state for a commit; an error leaves the request uncommitted.
pub fn prepareCommit(slot: *Slot) !void {
    const tag = tagOfSlot(slot) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "prepareCommit")) try S(t).prepareCommit(slot),
    }
}

/// The request over the arch's model state finished with `tokens`.
pub fn publishLive(sch: *Scheduler, slot: *Slot, hc: *pc.HotPrefixCache, tokens: []const u32) void {
    const tag = tagOfSlot(slot) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "publishLive")) S(t).publishLive(sch, slot, hc, tokens),
    }
}

/// The arch commits the slot its own way (true), or leaves the generic commit to the caller (false). `gen` is null
/// for a prefill the client cancelled.
pub fn commitOwn(
    hc: *pc.HotPrefixCache,
    slot: *Slot,
    gen: ?*Generator,
    tokens: []const u32,
    cps: ?[]trunk.SSMCheckpoint,
    prompt_len: usize,
) bool {
    const tag = tagOfSlot(slot) orelse return false;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "commitOwn")) S(t).commitOwn(hc, slot, gen, tokens, cps, prompt_len) else false,
    };
}

/// A prefill the client cancelled keeps nothing unless it salvaged checkpoints.
pub fn cancelledNeedsCheckpoints(slot: *const Slot) bool {
    const tag = tagOfSlot(slot) orelse return false;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "cancelledNeedsCheckpoints")) S(t).cancelledNeedsCheckpoints() else false,
    };
}

/// The newest checkpoint takes the slot's live indexer history as a view of its capacity buffer.
pub fn handoffHistory(slot: *Slot, cps: []trunk.SSMCheckpoint) void {
    const tag = tagOfSlot(slot) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "handoffHistory")) S(t).handoffHistory(slot, cps),
    }
}

/// The in-checkpoint head's rows a commit stores beside the KV; null refuses the head's history.
pub fn mtpHeadCommit(slot: *const Slot, head: *const Transformer) ?HeadParts {
    const tag = tagOfSlot(slot) orelse return .{};
    return switch (tag) {
        inline else => |t| if (comptime has(t, "mtpHeadCommit")) S(t).mtpHeadCommit(head) else HeadParts{},
    };
}

/// Where the lookup restores this slot's state, when the arch keeps it outside the SSM entries.
pub fn restoreTarget(slot: *Slot, xfm: *Transformer) ?RestoreTarget {
    const tag = tagOfSlot(slot) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "restoreTarget")) S(t).restoreTarget(slot, xfm) else null,
    };
}

/// A write-through prefill chunk the arch stores its own way (true), or the generic hybrid write (false).
pub fn writeThrough(
    hc: *pc.HotPrefixCache,
    slot: *Slot,
    entry_id: *?u64,
    prompt: []const u32,
    media: pc.MediaStamp,
    cps: []trunk.SSMCheckpoint,
    s: mlx.mlx_stream,
) bool {
    const tag = tagOfSlot(slot) orelse return false;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "writeThrough")) S(t).writeThrough(hc, slot, entry_id, prompt, media, cps, s) else false,
    };
}

/// The assistant sidecar drafts for this request by the arch's own rule.
pub fn nativeDflashArmed(slot: *const Slot) bool {
    const tag = tagOfSlot(slot) orelse return false;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "nativeDflashArmed")) S(t).nativeDflashArmed(slot) else false,
    };
}

/// The pool grid of the arch's thought point; null leaves the head-driven grid.
pub fn thoughtGrid(slot: *const Slot, gen: *const Generator) ?u32 {
    const tag = tagOfSlot(slot) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "thoughtGrid")) S(t).thoughtGrid(gen) else null,
    };
}

/// The arch's own thought point after prefill.
pub fn armThoughtPoint(slot: *Slot, gen: *Generator, tier_takes_points: bool) void {
    const tag = tagOfSlot(slot) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "armThoughtPoint")) S(t).armThoughtPoint(slot, gen, tier_takes_points),
    }
}

/// A speculative block was published.
pub fn afterSpecBlock(slot: *Slot, gen: *Generator, block: []const u32) void {
    const tag = tagOfSlot(slot) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "afterSpecBlock")) S(t).afterSpecBlock(slot, gen, block),
    }
}

/// A request ended on this model.
pub fn endRequest(config: *const ModelConfig) void {
    const tag = tagOfConfig(config) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "endRequest")) S(t).endRequest(config),
    }
}

/// A prefill is retried cold: what the arch holds across requests goes back to its start.
pub fn resetForRetry(slot: *const Slot, xfm: *Transformer) void {
    const tag = tagOfSlot(slot) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "resetForRetry")) S(t).resetForRetry(slot, xfm),
    }
}

/// The live restore points of a resident model reach SSD.
pub fn flushResident(entry: *model_registry.LoadedModel, why: []const u8) void {
    const tag = tagOfConfig(entry.config) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "flushResident")) S(t).flushResident(entry, why),
    }
}

const known_hooks = [_][]const u8{
    "facts",           "batches",       "rowCap",         "modelBatches",             "padsBatch",       "batchArm",
    "readyToBatch",
    "batchedForward",  "plannerObservesPlainRows",        "tickClass",                "mtpGroupable",    "initRequest",
    "checkKvQuant",    "coldLoadBill",  "modelStateBytes",                          "commitsAtCull",
    "prepareCommit",   "publishLive",   "commitOwn",      "cancelledNeedsCheckpoints", "handoffHistory",  "mtpHeadCommit",
    "restoreTarget",   "writeThrough",  "nativeDflashArmed",                          "thoughtGrid",     "armThoughtPoint",
    "afterSpecBlock",  "endRequest",    "resetForRetry",  "flushResident",
};

fn isHook(comptime name: []const u8) bool {
    @setEvalBranchQuota(100_000);
    inline for (known_hooks) |h| if (std.mem.eql(u8, h, name)) return true;
    return false;
}

test "every enabled arch publishes a sched module, and each of its declarations is a hook the dispatch calls" {
    inline for (comptime std.meta.tags(Tag)) |t| {
        const M = S(t);
        inline for (comptime std.meta.declarations(M)) |d| {
            if (comptime !isHook(d)) @compileError(@tagName(t) ++ " sched module declares " ++ d ++ ", which no dispatch function reads");
        }
    }
}

test "an unclaimed config has no facts, batches nothing and caps no rows" {
    var config = ModelConfig{ .model_type = "mistral" };
    try std.testing.expectEqual(Facts{}, facts(&config));
    try std.testing.expect(!batches(&config));
    try std.testing.expectEqual(@as(?usize, null), rowCap(&config));
}

test "batching per model type: what batches, and the rows one forward carries" {
    const Row = struct { model_type: []const u8, batches: bool, cap: ?usize };
    const rows = [_]Row{
        .{ .model_type = "qwen4_exp", .batches = true, .cap = 8 },
        .{ .model_type = "mimo_v2", .batches = true, .cap = 4 },
        .{ .model_type = "glm5_next", .batches = true, .cap = 4 },
        .{ .model_type = "deepseek_v41", .batches = false, .cap = null },
        .{ .model_type = "mistral", .batches = false, .cap = null },
    };
    for (rows) |r| {
        if (registry.ofModelType(r.model_type) == null and model.isServedArch(r.model_type)) continue;
        var config = ModelConfig{ .model_type = r.model_type, .full_attention_interval = 4 };
        try std.testing.expectEqual(r.batches, batches(&config));
        try std.testing.expectEqual(r.cap, rowCap(&config));
    }
}

test "only Flash-Next gates its long-context rules and caps its checkpoints; only the thinking-point archs arm one" {
    const Row = struct { model_type: []const u8, long_ctx: bool, caps: bool, thought: ThoughtPoint, release: bool, always: bool };
    const rows = [_]Row{
        .{ .model_type = "qwen4_exp", .long_ctx = true, .caps = true, .thought = .with_mtp, .release = false, .always = false },
        .{ .model_type = "glm5_next", .long_ctx = false, .caps = false, .thought = .always, .release = true, .always = false },
        .{ .model_type = "deepseek_v41", .long_ctx = false, .caps = false, .thought = .off, .release = false, .always = true },
        .{ .model_type = "mimo_v2", .long_ctx = false, .caps = false, .thought = .off, .release = false, .always = false },
    };
    for (rows) |r| {
        if (registry.ofModelType(r.model_type) == null) continue;
        var config = ModelConfig{ .model_type = r.model_type };
        const f = facts(&config);
        try std.testing.expectEqual(r.long_ctx, f.long_ctx);
        try std.testing.expectEqual(config.longCtxGated(), f.long_ctx);
        try std.testing.expectEqual(r.caps, f.caps_ssm_checkpoints);
        try std.testing.expectEqual(r.thought, f.thought_point);
        try std.testing.expectEqual(r.release, f.releases_checkpoints);
        try std.testing.expectEqual(r.always, f.always_write_through);
        try std.testing.expectEqual(rowCap(&config), f.row_cap);
        try std.testing.expectEqual(padsBatch(&config), f.pads_batch);
    }
}

/// A slot of `config` over a transformer, for a face's own test.
pub fn testSlot(xfm: *Transformer, lm: *model_registry.LoadedModel, config: ModelConfig) Slot {
    xfm.config = config;
    lm.transformer = xfm;
    lm.config = null;
    var slot: Slot = undefined;
    slot.model = lm;
    slot.request = null;
    slot.legacy_gen = null;
    return slot;
}

test "an arch with no request state restores into the SSM entries and commits the generic way" {
    inline for (.{ "qwen4_exp", "mimo_v2" }) |model_type| {
        if (registry.ofModelType(model_type) != null) {
            var xfm: Transformer = undefined;
            var lm: model_registry.LoadedModel = undefined;
            var slot = testSlot(&xfm, &lm, .{ .model_type = model_type });
            try std.testing.expect(restoreTarget(&slot, &xfm) == null);
            try std.testing.expect(!cancelledNeedsCheckpoints(&slot));
            try std.testing.expect(!commitsAtCull(&slot));
            try std.testing.expectEqual(@as(u64, 0), requestBytes(&slot));
            try std.testing.expectEqual(TickClass.none, tickClass(&slot));
        }
    }
}

test "a slot's own request state is billed, released and culled with no config to name its arch" {
    if (comptime !@hasField(Tag, "glm5")) return error.SkipZigTest;
    var lm: model_registry.LoadedModel = undefined;
    lm.transformer = null;
    lm.config = null;
    var slot: Slot = undefined;
    slot.model = &lm;
    slot.request = null;
    try std.testing.expectEqual(@as(u64, 0), requestBytes(&slot));
    try std.testing.expect(!commitsAtCull(&slot));
    slot.request = try NativeRequest.init(std.testing.allocator, 2);
    defer slot.request.?.deinit();
    slot.request.?.offset = 4;
    try std.testing.expectEqual(slot.request.?.residentBytes(), requestBytes(&slot));
    try std.testing.expect(commitsAtCull(&slot));
    releaseState(&slot);
    try std.testing.expectEqual(@as(usize, 0), slot.request.?.offset);
}
