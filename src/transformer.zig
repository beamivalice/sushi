//! The qwen4 names that shared files still reach: the prefix cache, the SSD tier, the scheduler and its admission,
//! `generate.zig`, `model.zig`, `main.zig` and `arch/sched.zig` have no arch hook for the QSA history, the member
//! generation or the gather floor yet. Each leaves with its hook in S4-qwen4, and this file with the last of them.
pub const QsaHeadMark = @import("qwen4/qwen4_qsa.zig").QsaHeadMark;
pub const QsaHeadMarkSet = @import("qwen4/qwen4_qsa.zig").QsaHeadMarkSet;
pub const applyQsaHistoryAt = @import("qwen4/qwen4_qsa_state.zig").applyQsaHistoryAt;
pub const attachQsaHistoryToLatest = @import("qwen4/qwen4_qsa_state.zig").attachQsaHistoryToLatest;
pub const checkpointHasQsaPooled = @import("qwen4/qwen4_qsa_state.zig").checkpointHasQsaPooled;
pub const checkpointListHasQsaPooled = @import("qwen4/qwen4_qsa_state.zig").checkpointListHasQsaPooled;
pub const checkpointQsaCoversPos = @import("qwen4/qwen4_qsa_state.zig").checkpointQsaCoversPos;
pub const entriesHaveQsaHistory = @import("qwen4/qwen4_qsa_state.zig").entriesHaveQsaHistory;
pub const handoffQsaHistoryToLatest = @import("qwen4/qwen4_qsa_state.zig").handoffQsaHistoryToLatest;
pub const keepOnlyLatestQsaHistory = @import("qwen4/qwen4_qsa_state.zig").keepOnlyLatestQsaHistory;
pub const nextSsmMemberGen = @import("qwen4/qwen4_ssm_group.zig").nextSsmMemberGen;
pub const qsaBatchedGatherFloor = @import("qwen4/qwen4_qsa_gather.zig").qsaBatchedGatherFloor;
pub const qsaBatchedGatherOn = @import("qwen4/qwen4_qsa_gather.zig").qsaBatchedGatherOn;
pub const qsaRestoreSatisfiesForward = @import("qwen4/qwen4_qsa_state.zig").qsaRestoreSatisfiesForward;
pub const qsaScoreFusedActiveFor = @import("qwen4/qwen4_qsa_select.zig").qsaScoreFusedActiveFor;
pub const qsaTestAppendPool = @import("qwen4/qwen4_qsa_state.zig").qsaTestAppendPool;
pub const reserveQsaHistoryWithHead = @import("qwen4/qwen4_qsa_state.zig").reserveQsaHistoryWithHead;
pub const sliceQsaHistoryOntoCheckpoint = @import("qwen4/qwen4_qsa_state.zig").sliceQsaHistoryOntoCheckpoint;
pub const snapshotHasQsaHistory = @import("qwen4/qwen4_qsa_state.zig").snapshotHasQsaHistory;
pub const ssmDetachFromGroup = @import("qwen4/qwen4_ssm_group.zig").ssmDetachFromGroup;
pub const warmQsaEnvCaches = @import("qwen4/qwen4_qsa_attn.zig").warmQsaEnvCaches;
