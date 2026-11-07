//! Flash-Next prefix-cache parts: the QSA-history section of an SSD record, the GDN/QSA accelerators on an SSM
//! entry, and the adoption of a committed MTP head. Reached through `arch/prefix.zig`.
const trunk_mod = @import("../trunk.zig");
const qwen4_forward = @import("qwen4_forward.zig");
const pc = @import("../prefix_cache.zig");
const log = @import("../log.zig");
const qwen4_exp = @import("qwen4_exp.zig");
const gdn_decode = @import("gdn_decode.zig");
const qsa_state = @import("qwen4_qsa_state.zig");

/// The `qsa.safetensors` section of a hybrid record.
pub const disk = @import("qwen4_disk_qsa.zig");

/// What an SSM entry carries beside the GDN pair: the PLE token window, the QSA key history and its accelerators,
/// and the replay record of a verify that kept only its final state.
pub const aux = struct {
    pub const ple_prev_len = qwen4_exp.MAX_NGRAM_SIZE;
    pub const Replay = gdn_decode.Replay;
    pub const replayState = gdn_decode.replayState;
    pub const qsaHistoryRows = qsa_state.qsaHistoryRows;
    pub const qsaRawKeyBytes = qsa_state.qsaRawKeyBytes;
    pub const ssmAuxIsQsaHistory = qsa_state.ssmAuxIsQsaHistory;
    pub const ssmFreeQsaState = qsa_state.ssmFreeQsaState;
    pub const qsaKeyAppendPlan = qsa_state.qsaKeyAppendPlan;
    pub const qsaLeftoverAt = qsa_state.qsaLeftoverAt;
    pub const qsaResliceToKeep = qsa_state.qsaResliceToKeep;

    pub fn noteHistorySeed() void {
        qsa_state.qsa_history_seeds += 1;
    }

    pub fn noteCapBufAlloc() void {
        qsa_state.qsa_cap_buf_allocs += 1;
    }
};

/// The in-checkpoint MTP head's snapshot (KV and QSA aux together) adopted into the model's own head. Null leaves
/// the head blind, which costs acceptance and never a token.
pub fn adoptSpecHead(xfm: *trunk_mod.Transformer, plan: pc.SpecAdopt, snap: *const pc.DflashSnap, base_pos: *usize) ?usize {
    const want = switch (plan) {
        .head => |w| w,
        .decline_head_no_history => {
            log.info("  [qwen4] MTP head restore declined (snapshot carries no QSA history) — head starts blind\n", .{});
            return null;
        },
        .skip, .kv_only => return null,
    };
    qwen4_forward.qwen4MtpAdopt(xfm, &snap.snapshot, &snap.head_aux.?, snap.head_marks.slice(), snap.head_pos_base, want) catch |err| {
        log.warn("  [qwen4] MTP head restore declined ({s}) — head starts blind\n", .{@errorName(err)});
        return null;
    };
    base_pos.* = snap.base_pos;
    log.info("  [qwen4] MTP head restored ({d} tokens from base {d})\n", .{ want, snap.base_pos });
    return snap.base_pos;
}
