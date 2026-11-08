//! GLM as the scheduler sees it: slots share a model and each owns its native request; a drafting slot rides a plain
//! grouped row when its model has company; commits and restores go through the MLA rows and KDA checkpoints.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const model_registry = @import("../model_registry.zig");
const kv_quant = @import("../kv_quant.zig");
const trunk_mod = @import("../trunk.zig");
const glm5_state = @import("glm5_state.zig");
const prefix_cache = @import("../prefix_cache.zig");
const prefix_arch = @import("../arch/prefix.zig");
const scheduler = @import("../scheduler.zig");
const generate = @import("../generate.zig");
const sched = @import("../arch/sched.zig");
const forward = @import("glm5_forward.zig");
const glm5_prefix = @import("glm5_prefix.zig");
const flags = @import("glm5_flags.zig");
const rows_tick = @import("glm5_rows_tick.zig");
const glm5_load = @import("glm5_load.zig");

const Slot = scheduler.Slot;
const Scheduler = scheduler.Scheduler;
const Trunk = trunk_mod.Trunk;
const ForwardCtx = trunk_mod.ForwardCtx;
const Generator = generate.Generator;
const ModelConfig = model.ModelConfig;

pub fn facts(config: *const ModelConfig) sched.Facts {
    _ = config;
    return .{ .thought_point = .always, .bills_checkpoints = true, .releases_checkpoints = true };
}

pub fn batches(config: *const ModelConfig) bool {
    return config.supportsBatchedGlmRows();
}

pub fn rowCap(config: *const ModelConfig) ?usize {
    return if (config.supportsBatchedGlmRows()) 4 else null;
}

pub fn modelBatches(xfm: *const Trunk) bool {
    return glm5_state.supportsBatchedRows(xfm);
}

/// Per-row attention reads each slot's own cache: nothing pads.
pub fn padsBatch(config: *const ModelConfig) bool {
    return !config.supportsBatchedGlmRows();
}

pub fn batchArm(xfm: *Trunk, ctxs: []const *ForwardCtx) ?sched.BatchArm {
    _ = ctxs;
    if (!glm5_state.supportsBatchedRows(xfm)) return null;
    return .{ .moe_positions = false, .advances_offset = true };
}

pub fn batchedForward(
    sch: *Scheduler,
    allocator: std.mem.Allocator,
    xfm: *Trunk,
    batch: *[]*Slot,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
    arm: sched.BatchArm,
    hidden_rows: ?*?[]mlx.mlx_array,
) ![]mlx.mlx_array {
    _ = .{ ctxs, rope_offsets, arm, hidden_rows };
    return rows_tick.glmRowsForward(sch, allocator, xfm, batch, next_tokens);
}

pub fn tickClass(slot: *const Slot) sched.TickClass {
    return if (rows_tick.slotGlmRowable(slot)) .rows_company else .none;
}

pub fn initRequest(allocator: std.mem.Allocator, config: *const ModelConfig) !?sched.NativeRequest {
    return try forward.Request.initServing(allocator, config.num_hidden_layers);
}

/// The same fact as the request-side refusal (`glm5_flags`): the latent's schemes are one list.
pub fn checkKvQuant(kv: kv_quant.KVQuantConfig) !void {
    if (flags.kvQuantRequestRefusal(kv) != null) return error.GlmKvQuantUnsupported;
}

pub fn coldLoadBill(load: sched.ColdLoad) !?u64 {
    return try glm5_load.coldLoadBillBytes(load.io, load.allocator, load.config, load.path, load.load_vision, load.no_drafter, load.drafter_dir);
}

pub fn commitOwn(
    hc: *prefix_cache.HotPrefixCache,
    slot: *Slot,
    gen: ?*Generator,
    tokens: []const u32,
    cps: ?[]trunk_mod.SSMCheckpoint,
    prompt_len: usize,
) bool {
    const request = if (slot.request) |*r| r else return false;
    const window: ?trunk_mod.KVCache = if (gen) |g| if (g.glm_prefill_window) |*w|
        .{ .entries = w.snapshot.entries, .step = w.snapshot.step, .allocator = w.snapshot.allocator, .config = w.snapshot.config }
    else
        null else null;
    const dflash: ?prefix_cache.DflashCommit = if (window) |*w| .{ .cache = w, .base_pos = gen.?.glm_prefill_window.?.base_pos } else null;
    rows_tick.commitGlmSlot(hc, slot, request, tokens, cps, dflash, prompt_len);
    return true;
}

pub fn restoreTarget(slot: *Slot, xfm: *Trunk) ?sched.RestoreTarget {
    _ = xfm;
    const request = if (slot.request) |*r| r else return null;
    return .{ .rec = prefix_arch.recurrent("glm5", request), .takes_dflash = true };
}

pub fn writeThrough(
    hc: *prefix_cache.HotPrefixCache,
    slot: *Slot,
    entry_id: *?u64,
    prompt: []const u32,
    media: prefix_cache.MediaStamp,
    cps: []trunk_mod.SSMCheckpoint,
    s: mlx.mlx_stream,
) bool {
    const request = if (slot.request) |*r| r else return false;
    glm5_prefix.writeThroughGlm(hc, entry_id, request, prompt, slot.has_tools, media, cps, s);
    return true;
}

pub fn nativeDflashArmed(slot: *const Slot) bool {
    return rows_tick.glmNativeDflashArmed(true, slot.enable_drafter, slot.dflash != null, slot.sampling, slot.logprobs_n);
}

pub fn thoughtGrid(gen: *const Generator) ?u32 {
    return if (gen.native_dflash) @intCast(glm5_prefix.pool_size) else null;
}

test "an arch with a native request restores into it with the assistant context and releases it when the thread lets go" {
    var xfm: Trunk = undefined;
    var lm: model_registry.LoadedModel = undefined;
    var slot = sched.testSlot(&xfm, &lm, .{ .model_type = "glm5_next", .num_hidden_layers = 2 });
    slot.request = (try initRequest(std.testing.allocator, &xfm.config)).?;
    defer slot.request.?.deinit();
    const target = sched.restoreTarget(&slot, &xfm).?;
    try std.testing.expectEqualStrings("[prefix]", target.log_tag);
    try std.testing.expect(target.takes_dflash and !target.takes_mtp);
    try std.testing.expect(sched.commitsAtCull(&slot));
    slot.request.?.offset = 4;
    sched.releaseState(&slot);
    try std.testing.expectEqual(@as(usize, 0), slot.request.?.offset);
    try std.testing.expectError(error.GlmKvQuantUnsupported, sched.checkKvQuant(&xfm.config, .{ .scheme = .affine, .bits = 4, .group_size = 64 }));
}

test "a KV scheme is refused at submit exactly when the request-side check refuses it" {
    const odd_group: kv_quant.KVQuantConfig = .{ .scheme = .affine, .bits = 8, .group_size = 32 };
    const schemes = [_]kv_quant.KVQuantConfig{ .dense, .affine(8), .affine(6), .affine(4), odd_group };
    for (schemes) |kv| {
        const refused = flags.kvQuantRequestRefusal(kv) != null;
        if (refused) {
            try std.testing.expectError(error.GlmKvQuantUnsupported, checkKvQuant(kv));
        } else {
            try checkKvQuant(kv);
        }
    }
}
