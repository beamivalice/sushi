//! Hy4-Flash as the scheduler sees it: one request at a time, as a streamed DeepSeek load. Restore points go to the SSD
//! tier as the prefill takes them, and the live record's reach it before its state is lost (`hy4_prefix`).
const model = @import("../model.zig");
const hy4_prefix = @import("hy4_prefix.zig");

/// A stored point's rows are views of the state's, so every one goes to SSD before the next forward, however short the
/// new span (`hy4_prefix.writeThrough`).
pub fn facts(config: *const model.ModelConfig) @import("../arch/sched.zig").Facts {
    _ = config;
    return .{ .always_write_through = true };
}

pub fn writeThrough(
    hc: *@import("../prefix_cache.zig").HotPrefixCache,
    slot: *@import("../scheduler.zig").Slot,
    entry_id: *?u64,
    prompt: []const u32,
    media: @import("../prefix_cache.zig").MediaStamp,
    cps: []@import("../trunk.zig").SSMCheckpoint,
    s: @import("../mlx.zig").mlx_stream,
) bool {
    _ = entry_id;
    const m = @import("../arch/handle.zig").trunk(slot.model.handle orelse return false).arch_state.hy4flash.model orelse return false;
    hy4_prefix.writeThrough(hc, m, &slot.cache, prompt, slot.has_tools, media, slot.restored_disk, cps, s);
    return true;
}

/// The live restore points of a resident model reach SSD (unload, idle, shutdown).
pub fn flushResident(entry: *@import("../model_registry.zig").LoadedModel, why: []const u8) void {
    const hc = if (entry.prefix_cache) |*p| p else return;
    const xfm = @import("../arch/handle.zig").trunk(entry.handle orelse return);
    hy4_prefix.flushLive(hc, xfm.arch_state.hy4flash.model orelse return, why);
}

pub fn batches(config: *const model.ModelConfig) bool {
    _ = config;
    return false;
}

/// One request a forward: the composed forward keeps one request's state, and a streamed load's expert cache is one
/// request's working set.
pub fn coolerRows(config: *const model.ModelConfig) @import("../arch/sched.zig").CoolerRows {
    _ = config;
    return .{ .rows = 1, .why = "its composed forward keeps one request's state, and its routed experts stream from SSD" };
}

/// A lookup restores into the model's own request state (`hy4_prefix`), reserved first at the request's end
/// (`hy4_prefix.reserveRequest`), so a restore into it asks for no buffer.
pub fn restoreTarget(slot: *@import("../scheduler.zig").Slot, xfm: *@import("../trunk.zig").Trunk) ?@import("../arch/sched.zig").RestoreTarget {
    const m = xfm.arch_state.hy4flash.model orelse return null;
    const past_end = @import("../bills.zig").speculativeRowsPastEnd(&xfm.config, .{ .mtp_on = slot.enable_mtp and slot.mtp != null });
    hy4_prefix.reserveRequest(m, slot.prompt_ids.len, slot.max_tokens, past_end) catch |e|
        @import("../log.zig").warn("[hy4-prefix] the request's state was not reserved before its lookup ({t}): its restore reserves it\n", .{e});
    return .{ .rec = @import("../arch/prefix.zig").recurrent("hy4flash", m), .log_tag = "[hy4-prefix]" };
}

/// The request over the model's state finished with `tokens`: the live record's tokens and the tools and media they
/// were computed under, which the next prompt is matched against (`hy4_prefix.rollbackLive`).
pub fn publishLive(sch: *@import("../scheduler.zig").Scheduler, slot: *@import("../scheduler.zig").Slot, hc: *@import("../prefix_cache.zig").HotPrefixCache, tokens: []const u32) void {
    const xfm = @import("../arch/handle.zig").trunk(slot.model.handle orelse return);
    const m = xfm.arch_state.hy4flash.model orelse return;
    const media = @import("../prefix_cache.zig").MediaStamp.of(slot.vision_key, slot.media_start);
    hy4_prefix.publishLive(hc, m, tokens, slot.has_tools, media);
    // A point only in RAM reaches SSD on the idle flush, if no other request takes the state first.
    if (hy4_prefix.liveOf(hc).unflushed(if (hc.disk) |*d| d else null)) sch.live_unflushed.store(true, .release);
}

/// The state the model holds outside its hot cache's entries, which a running request's slot bills (the request's own
/// is the model's): the state, its running points and the live record's (`hy4_prefix.heldBytes`).
pub fn modelStateBytes(slot: *const @import("../scheduler.zig").Slot) u64 {
    return parkedStateBytes(slot.model);
}

/// Admission needs the room of the state the model parks: the live record's points reach SSD and go, and the state
/// is released; the next request starts over or restores from SSD.
pub fn releaseParked(entry: *@import("../model_registry.zig").LoadedModel) void {
    const xfm = @import("../arch/handle.zig").trunk(entry.handle orelse return);
    const m = xfm.arch_state.hy4flash.model orelse return;
    if (entry.prefix_cache) |*hc| hy4_prefix.releaseLive(hc, m, "admission");
    m.reset();
    @import("../mlx.zig").engine.trim();
}

/// The same with no request on it: what the model keeps for the next one.
pub fn parkedStateBytes(entry: *@import("../model_registry.zig").LoadedModel) u64 {
    const xfm = @import("../arch/handle.zig").trunk(entry.handle orelse return 0);
    const m = xfm.arch_state.hy4flash.model orelse return 0;
    return @import("hy4_prefix.zig").heldBytes(m, if (entry.prefix_cache) |*hc| hc else null);
}
