//! The scheduler's view of an arch. `scheduler.zig` and `slot_admission.zig` keep the ticks, the queues and the
//! admission order; what differs per arch (what batches, what a request owns, how it commits, restores and writes
//! through) is declared by the arch's `sched` module (`<arch>_sched.zig`) and reached here, `inline else` over the
//! enabled archs. A module omits a hook to get the default stated on its dispatch function. Inference thread only,
//! except `facts`, `batches` and `rowCap`, which read a config.
//!
//! Dispatch reads the arch tag of the slot's (or model's) config, never the loaded handle, so a slot built by a
//! test needs only its config.
const std = @import("std");
const arch_handle = @import("handle.zig");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const registry = @import("registry.zig");
const arch_config = @import("config.zig");
const arch_state = @import("arch_state.zig");
const pc = @import("../prefix_cache.zig");
const kv_quant = @import("../kv_quant.zig");
const model_handle = @import("../model_handle.zig");
const trunk = @import("../trunk.zig");
const model_registry = @import("../model_registry.zig");
const generate = @import("../generate.zig");
const scheduler = @import("../scheduler.zig");

const Tag = registry.Tag;
const Slot = scheduler.Slot;
const Scheduler = scheduler.Scheduler;
const Generator = generate.Generator;
const Trunk = trunk.Trunk;
const ForwardCtx = model_handle.ForwardCtx;
const ModelConfig = model.ModelConfig;

/// The request state a slot owns beside its KV cache (`arch_state.SlotRequest`).
pub const NativeRequest = arch_state.SlotRequest;

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
    marks: []const trunk.QsaHeadMark = &.{},
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
    if (slot.model.handle) |h| return tagOfModel(arch_handle.trunk(h));
    return tagOfConfig(slot.model.config);
}

fn tagOfModel(xfm: *const Trunk) ?Tag {
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

/// What one forward of a Sushi Cooler load carries: its rows, and at one, why.
pub const CoolerRows = struct {
    rows: usize,
    /// Why the load serves one row; null when it serves more.
    why: ?[]const u8 = null,
    /// What serving more rows costs or changes, for the load line; null when nothing.
    note: ?[]const u8 = null,
};

const no_cooler_rows = "its arch declares no Cooler rows";

/// The rows of a Cooler load of `config`. An arch with Sushi Cooler declares them (a compile error otherwise,
/// `arch/config.zig`); only an arch that is never a Cooler load takes the one-row default.
pub fn coolerRows(config: *const ModelConfig) CoolerRows {
    const tag = tagOfConfig(config) orelse return .{ .rows = 1, .why = no_cooler_rows };
    return switch (tag) {
        inline else => |t| if (comptime has(t, "coolerRows")) S(t).coolerRows(config) else .{ .rows = 1, .why = no_cooler_rows },
    };
}

/// A load on Sushi Cooler batches when its arch batches and its Cooler rows are more than one; on MLX the arch answers.
pub fn coolerBatches(on_engine: bool, rows: CoolerRows, arch_batches: bool) bool {
    return arch_batches and (!on_engine or rows.rows > 1);
}

/// A load on Sushi Cooler carries at most its Cooler rows, under the arch's own cap; on MLX the arch's cap stands.
pub fn coolerRowCap(on_engine: bool, rows: CoolerRows, arch_cap: ?usize) ?usize {
    if (!on_engine) return arch_cap;
    return @min(arch_cap orelse rows.rows, rows.rows);
}

/// The load line of a Cooler load's rows; null for a load on MLX.
pub fn coolerRowsLine(buf: []u8, config: *const ModelConfig) ?[]const u8 {
    return coolerRowsLineOf(buf, arch_handle.onEngine(null, config), coolerRows(config));
}

pub fn coolerRowsLineOf(buf: []u8, on_engine: bool, rows: CoolerRows) ?[]const u8 {
    if (!on_engine) return null;
    if (rows.rows > 1) return std.fmt.bufPrint(buf, "[cooler] one forward carries {d} rows{s}{s}", .{ rows.rows, if (rows.note != null) "; " else "", rows.note orelse "" }) catch null;
    return std.fmt.bufPrint(buf, "[cooler] one forward carries one row: {s}", .{rows.why orelse no_cooler_rows}) catch null;
}

/// Does this config decode concurrent slots as one batch? A Cooler load answers through its Cooler rows.
pub fn batches(config: *const ModelConfig) bool {
    const tag = tagOfConfig(config) orelse return false;
    const arch_batches = switch (tag) {
        inline else => |t| if (comptime has(t, "batches")) S(t).batches(config) else false,
    };
    if (!arch_batches) return false;
    return coolerBatches(arch_handle.onEngine(null, config), coolerRows(config), true);
}

/// The rows one batched forward carries for this config, when the arch bounds it below `MAX_BATCH_GROUP`; a Cooler
/// load is bounded by its Cooler rows as well.
pub fn rowCap(config: *const ModelConfig) ?usize {
    const tag = tagOfConfig(config) orelse return null;
    const cap = switch (tag) {
        inline else => |t| if (comptime has(t, "rowCap")) S(t).rowCap(config) else null,
    };
    if (!arch_handle.onEngine(null, config)) return cap;
    return coolerRowCap(true, coolerRows(config), cap);
}

/// The loaded model batches (the config's answer, and the layers it built); a Cooler load through its Cooler rows.
pub fn modelBatches(xfm: *const Trunk) bool {
    const tag = tagOfModel(xfm) orelse return false;
    const arch_batches = switch (tag) {
        inline else => |t| if (comptime has(t, "modelBatches")) S(t).modelBatches(xfm) else false,
    };
    if (!arch_batches) return false;
    return coolerBatches(arch_handle.onEngine(null, &xfm.config), coolerRows(&xfm.config), true);
}

/// Whether the slot whose context this is can merge into this tick's batch; one that cannot yet decodes serially.
pub fn readyToBatch(xfm: *const Trunk, ctx: *ForwardCtx) bool {
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
pub fn batchArm(xfm: *Trunk, ctxs: []const *ForwardCtx) ?BatchArm {
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
    xfm: *Trunk,
    batch: *[]*Slot,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
    arm: BatchArm,
    hidden_rows: ?*?[]mlx.mlx_array,
) ![]mlx.mlx_array {
    const tag = tagOfModel(xfm) orelse return error.NoBatchedKernel;
    enterJoinedRows(xfm);
    return switch (tag) {
        inline else => |t| if (comptime has(t, "batchedForward"))
            S(t).batchedForward(sch, allocator, xfm, batch, next_tokens, ctxs, rope_offsets, arm, hidden_rows)
        else
            error.NoBatchedKernel,
    };
}

/// A joined forward of the model of `xfm` begins (`arch_cooler.enterJoined`).
pub fn enterJoinedRows(xfm: *Trunk) void {
    @import("cooler.zig").enterJoined(&xfm.config, &xfm.cooler_budget);
}

/// A joined tick's draws on a load whose arch plans its samples: each row through its own request's planned root.
/// True when the arch drew every row; false leaves the tick to `sampler.sampleRows`.
pub fn sampleJoinedRows(xfm: *Trunk, gens: []const *generate.Generator, rows: []const mlx.mlx_array, params: []@import("../sampler.zig").SamplingParams, out: []i32) !bool {
    const tag = tagOfModel(xfm) orelse return false;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "sampleJoinedRows")) S(t).sampleJoinedRows(xfm, gens, rows, params, out) else false,
    };
}

/// Whether the arch runs joined decode rows outside a scheduler (`rowsForward`); one whose rows need a request's
/// own state from the scheduler does not.
pub fn hasRowsForward(xfm: *const Trunk) bool {
    const tag = tagOfModel(xfm) orelse return false;
    return switch (tag) {
        inline else => |t| comptime has(t, "rowsForward"),
    };
}

/// One forward of joined decode rows outside a scheduler (a Cooler load's served warm), as `batchedForward` runs
/// them for `arm`, for an arch with `hasRowsForward`.
pub fn rowsForward(xfm: *Trunk, next_tokens: []const u32, ctxs: []const *ForwardCtx, rope_offsets: []const u32, arm: BatchArm, hidden_rows: ?*?[]mlx.mlx_array) ![]mlx.mlx_array {
    const tag = tagOfModel(xfm) orelse return error.NoBatchedKernel;
    enterJoinedRows(xfm);
    return switch (tag) {
        inline else => |t| if (comptime has(t, "rowsForward")) S(t).rowsForward(xfm, next_tokens, ctxs, rope_offsets, arm, hidden_rows) else error.NoBatchedKernel,
    };
}

/// Plain decode rows are priced by the grouped-MTP planner as they run (its cost table learns from them).
pub fn plannerObservesPlainRows(xfm: *const Trunk) bool {
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
    return company >= rowsCompanyMin(slot);
}

/// How many slots of one model decoding at once make a drafting slot ride a plain row: the arch's crossover, else two.
fn rowsCompanyMin(slot: *const Slot) usize {
    const tag = tagOfSlot(slot) orelse return 2;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "rows_company_min")) S(t).rows_company_min else 2,
    };
}

/// The fewest rows one joined forward of the model runs, one or two: a group the tick's filter leaves narrower ticks its
/// row as a single slot.
pub fn joinedRowsMin(xfm: *const Trunk) usize {
    const tag = tagOfModel(xfm) orelse return 1;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "joined_rows_min")) S(t).joined_rows_min else 1,
    };
}

/// The state a new slot owns beside its KV cache; null where the arch keeps none per request.
pub fn initRequest(allocator: std.mem.Allocator, config: *const ModelConfig) !?NativeRequest {
    const tag = tagOfConfig(config) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "initRequest") and arch_state.keepsRequest(t))
            NativeRequest.of(t, try S(t).initRequest(allocator, config) orelse return null)
        else
            null,
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

/// Bytes of the state a model keeps between requests for the next one to take over; a live slot of the model bills
/// them as `modelStateBytes`.
pub fn parkedStateBytes(entry: *model_registry.LoadedModel) u64 {
    const tag = tagOfConfig(entry.config) orelse return 0;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "parkedStateBytes")) S(t).parkedStateBytes(entry) else 0,
    };
}

/// Admission takes the room of the state a model parks: it reaches SSD where it keeps restore points, and is released.
pub fn releaseParked(entry: *model_registry.LoadedModel) void {
    const tag = tagOfConfig(entry.config) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "releaseParked")) S(t).releaseParked(entry),
    }
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

/// What the arch's `prepareCommit` readies, for the line that says it failed.
pub fn prepareCommitLabel(slot: *const Slot) []const u8 {
    const tag = tagOfSlot(slot) orelse return "commit preparation";
    return switch (tag) {
        inline else => |t| if (comptime has(t, "prepare_commit_label")) S(t).prepare_commit_label else "commit preparation",
    };
}

/// Who the cancelled-prefill line says would restore as a miss.
pub fn cancelledEntryLabel(slot: *const Slot) []const u8 {
    const tag = tagOfSlot(slot) orelse return "an entry";
    return switch (tag) {
        inline else => |t| if (comptime has(t, "cancelled_entry_label")) S(t).cancelled_entry_label else "an entry",
    };
}

/// A request keeps `kept` of the `max` checkpoints its admission would bill: the arch says so in the log.
pub fn logCheckpointsKept(slot: *const Slot, kept: u32, max: u32) void {
    const tag = tagOfSlot(slot) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "logCheckpointsKept")) S(t).logCheckpointsKept(kept, max, slot.full_prompt.len),
    }
}

/// The newest checkpoint takes the slot's live indexer history as a view of its capacity buffer.
pub fn handoffHistory(slot: *Slot, cps: []trunk.SSMCheckpoint) void {
    const tag = tagOfSlot(slot) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "handoffHistory")) S(t).handoffHistory(slot, cps),
    }
}

/// The in-checkpoint head's rows a commit stores beside the KV; null refuses the head's history.
pub fn mtpHeadCommit(slot: *const Slot, head: *const Trunk) ?HeadParts {
    const tag = tagOfSlot(slot) orelse return .{};
    return switch (tag) {
        inline else => |t| if (comptime has(t, "mtpHeadCommit")) S(t).mtpHeadCommit(head) else HeadParts{},
    };
}

/// Where the lookup restores this slot's state, when the arch keeps it outside the SSM entries.
pub fn restoreTarget(slot: *Slot, xfm: *Trunk) ?RestoreTarget {
    const tag = tagOfSlot(slot) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "restoreTarget")) S(t).restoreTarget(slot, xfm) else null,
    };
}

/// The restore `restoreTarget` named is over: an arch that lent it a state takes the lend back.
pub fn restoreEnd(slot: *Slot, xfm: *Trunk) void {
    const tag = tagOfModel(xfm) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "restoreEnd")) S(t).restoreEnd(slot, xfm),
    }
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
    closeRequest(config);
    @import("cooler.zig").rest(config);
}

/// A request ended on this model, its slot still to be freed: the pool comes to rest once it is (`cooler.rest`).
pub fn closeRequest(config: *const ModelConfig) void {
    defer @import("cooler.zig").closeRequest(config);
    const tag = tagOfConfig(config) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "endRequest")) S(t).endRequest(config),
    }
}

/// A prefill is retried cold: what the arch holds across requests goes back to its start.
pub fn resetForRetry(slot: *const Slot, xfm: *Trunk) void {
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
    "readyToBatch",    "coolerRows",    "sampleJoinedRows",
    "batchedForward",  "rowsForward",   "plannerObservesPlainRows",        "tickClass",                "mtpGroupable",    "initRequest",
    "checkKvQuant",    "coldLoadBill",  "modelStateBytes", "parkedStateBytes",      "releaseParked", "commitsAtCull",
    "prepareCommit",   "publishLive",   "commitOwn",      "cancelledNeedsCheckpoints", "handoffHistory",  "mtpHeadCommit",
    "restoreTarget",   "restoreEnd",    "writeThrough",  "nativeDflashArmed",                          "thoughtGrid",     "armThoughtPoint",
    "afterSpecBlock",  "endRequest",    "resetForRetry",  "flushResident",
    "logCheckpointsKept", "prepare_commit_label", "cancelled_entry_label", "rows_company_min",
    "joined_rows_min",
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

test "Cooler rows: an arch that declares none serves one row on Sushi Cooler, and says why" {
    var config = ModelConfig{ .model_type = "mistral" };
    const r = coolerRows(&config);
    try std.testing.expectEqual(@as(usize, 1), r.rows);
    try std.testing.expectEqualStrings("its arch declares no Cooler rows", r.why.?);
}

test "Cooler rows gate admission on a Cooler load only: one row batches nothing, more rows cap the arch's own" {
    const one: CoolerRows = .{ .rows = 1, .why = "one" };
    const four: CoolerRows = .{ .rows = 4 };
    // On MLX the arch's answers stand, whatever its Cooler rows.
    try std.testing.expect(coolerBatches(false, one, true));
    try std.testing.expect(!coolerBatches(false, four, false));
    try std.testing.expectEqual(@as(?usize, 8), coolerRowCap(false, one, 8));
    try std.testing.expectEqual(@as(?usize, null), coolerRowCap(false, four, null));
    // On Sushi Cooler a batch needs the arch's batching and more than one row; the cap is the smaller.
    try std.testing.expect(!coolerBatches(true, one, true));
    try std.testing.expect(!coolerBatches(true, four, false));
    try std.testing.expect(coolerBatches(true, four, true));
    try std.testing.expectEqual(@as(?usize, 4), coolerRowCap(true, four, 8));
    try std.testing.expectEqual(@as(?usize, 4), coolerRowCap(true, four, null));
    try std.testing.expectEqual(@as(?usize, 2), coolerRowCap(true, four, 2));
    try std.testing.expectEqual(@as(?usize, 1), coolerRowCap(true, one, 8));
}

test "Cooler rows: the load line names the rows, and at one row why; a load on MLX has none" {
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("[cooler] one forward carries 4 rows", coolerRowsLineOf(&buf, true, .{ .rows = 4 }).?);
    try std.testing.expectEqualStrings("[cooler] one forward carries 4 rows; a cost", coolerRowsLineOf(&buf, true, .{ .rows = 4, .note = "a cost" }).?);
    try std.testing.expectEqualStrings("[cooler] one forward carries one row: its arch declares no Cooler rows", coolerRowsLineOf(&buf, true, .{ .rows = 1, .why = "its arch declares no Cooler rows" }).?);
    try std.testing.expect(coolerRowsLineOf(&buf, false, .{ .rows = 4 }) == null);
}

test "Cooler rows: a streamed DeepSeek load serves one row and names why; a fully resident one joins rows and names their cost" {
    var config = ModelConfig{ .model_type = "deepseek_v41" };
    if (registry.ofModelType(config.model_type) == null) return error.SkipZigTest;
    config.expert_streaming = true;
    const streamed = coolerRows(&config);
    try std.testing.expectEqual(@as(usize, 1), streamed.rows);
    try std.testing.expect(std.mem.indexOf(u8, streamed.why.?, "stream") != null);
    // A resident plan still runs through the stream engine: the plan's answer is `expert_resident`.
    config.expert_resident = true;
    // Rows are Sushi Cooler's: a resident load on MLX keeps one row.
    try std.testing.expectEqual(@as(usize, 1), coolerRows(&config).rows);
    config.cooler_for_test = true;
    const resident = coolerRows(&config);
    try std.testing.expect(resident.rows >= 8);
    try std.testing.expect(resident.why == null);
    try std.testing.expect(std.mem.indexOf(u8, resident.note.?, "exact arm") != null);
}

test "only Flash-Next gates its long-context rules and caps its checkpoints; only the thinking-point archs arm one" {
    const Row = struct { model_type: []const u8, long_ctx: bool, caps: bool, thought: ThoughtPoint, release: bool, always: bool };
    const rows = [_]Row{
        .{ .model_type = "qwen4_exp", .long_ctx = true, .caps = true, .thought = .with_mtp, .release = false, .always = false },
        .{ .model_type = "glm5_next", .long_ctx = false, .caps = false, .thought = .always, .release = true, .always = false },
        .{ .model_type = "deepseek_v41", .long_ctx = false, .caps = false, .thought = .off, .release = false, .always = true },
        .{ .model_type = "mimo_v2", .long_ctx = false, .caps = false, .thought = .off, .release = false, .always = false },
        .{ .model_type = "hy_v4_flash", .long_ctx = false, .caps = false, .thought = .off, .release = false, .always = true },
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
pub fn testSlot(xfm: *Trunk, lm: *model_registry.LoadedModel, config: ModelConfig) Slot {
    xfm.config = config;
    lm.handle = arch_handle.fixtureHandle(xfm);
    lm.config = null;
    var slot = Slot.initForTest(std.testing.io, std.testing.allocator, lm);
    slot.model = lm;
    slot.request = null;
    slot.legacy_gen = null;
    return slot;
}

test "an arch with no request state restores into the SSM entries and commits the generic way" {
    inline for (.{ "qwen4_exp", "mimo_v2" }) |model_type| {
        if (registry.ofModelType(model_type) != null) {
            var xfm: Trunk = undefined;
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
    lm.handle = null;
    lm.config = null;
    var slot = Slot.initForTest(std.testing.io, std.testing.allocator, &lm);
    slot.bindCtxForTest();
    slot.model = &lm;
    slot.request = null;
    try std.testing.expectEqual(@as(u64, 0), requestBytes(&slot));
    try std.testing.expect(!commitsAtCull(&slot));
    slot.request = NativeRequest.of(.glm5, try arch_state.RequestType(.glm5).init(std.testing.allocator, 2));
    defer slot.request.?.deinit();
    slot.request.?.as(.glm5).?.offset = 4;
    try std.testing.expectEqual(slot.request.?.residentBytes(), requestBytes(&slot));
    try std.testing.expect(commitsAtCull(&slot));
    releaseState(&slot);
    try std.testing.expectEqual(@as(usize, 0), slot.request.?.as(.glm5).?.offset);
}
