const std = @import("std");
const glm5_mod = @import("glm5/glm5_forward.zig");
const dsv41_model = @import("dsv41/dsv41_model.zig");
const dsv41_stream = @import("dsv41/dsv41_stream.zig");
const dsv41_borrow = @import("dsv41/dsv41_borrow.zig");
const GlmRequest = glm5_mod.Request;
const qwen4_mod = @import("qwen4/qwen4_exp.zig");
const expert_stream_mod = @import("expert_stream.zig");
const expert_usage = @import("dsv41/expert_usage.zig");
const expert_bf16 = @import("expert_bf16_kernels.zig");
const imatrix_capture = @import("imatrix.zig");
const sushi_exl3 = @import("sushi_exl3");
const expert_exl3_kernels = sushi_exl3.kernels;
const expert_exl3 = sushi_exl3.format;
const expert_quant_mod = @import("expert_quant.zig");
const fp8_block = @import("fp8_block.zig");
const gdn_decode = @import("qwen4/gdn_decode.zig");
// The qwen4_exp MTP head shares the sidecar head's draft-rerank scheme
// (`mtp.rerankSelect` + `QLinear`), which reads only the TARGET's lm_head and
// so has no head-shaped state of its own. mtp.zig imports this file back for
// `Transformer`; neither type closes a comptime loop.
pub const mtp_mod = @import("mtp.zig");
const mlx = @import("mlx.zig");
const mrope = @import("mrope.zig");
const kv_quant = @import("kv_quant.zig");

const mimo_forward = @import("mimo2/mimo_forward.zig");
const qwen4_forward = @import("qwen4/qwen4_forward.zig");
const qwen4_qsa = @import("qwen4/qwen4_qsa.zig");
const qwen4_hc = @import("qwen4/qwen4_hc.zig");
const qwen4_qsa_attn = @import("qwen4/qwen4_qsa_attn.zig");
const qwen4_qsa_gather = @import("qwen4/qwen4_qsa_gather.zig");
const qwen4_qsa_select = @import("qwen4/qwen4_qsa_select.zig");
const qwen4_qsa_state = @import("qwen4/qwen4_qsa_state.zig");
const qwen4_ssm_group = @import("qwen4/qwen4_ssm_group.zig");
const qwen4_fixture_test = @import("qwen4/qwen4_fixture_test.zig");
const qwen4_gdn = @import("qwen4/qwen4_gdn.zig");
const kv_cache = @import("kv_cache.zig");
const ssm_cache = @import("ssm_cache.zig");
const GDN_FUSED_MAX_ROWS = qwen4_gdn.GDN_FUSED_MAX_ROWS;
const GdnPrework = qwen4_gdn.GdnPrework;
const gdnDecodeFusedEnabled = qwen4_gdn.gdnDecodeFusedEnabled;
const gdnDecodeRecurEnabled = qwen4_gdn.gdnDecodeRecurEnabled;
pub const gdnGateChain = qwen4_gdn.gdnGateChain;
const gdnNormGateFused = qwen4_gdn.gdnNormGateFused;
const gdnPrefillFusedFor = qwen4_gdn.gdnPrefillFusedFor;
const gdnPrefillRoute = qwen4_gdn.gdnPrefillRoute;
const gdnPreworkFused = qwen4_gdn.gdnPreworkFused;
const gdnRouteKernel = qwen4_gdn.gdnRouteKernel;
const gdnSetGrid = qwen4_gdn.gdnSetGrid;
const gdnVerifyFoldFor = qwen4_gdn.gdnVerifyFoldFor;
const getGdnKernelSeq = qwen4_gdn.getGdnKernelSeq;
const test_support = @import("test_support.zig");
const Exl3MoeHarness = test_support.Exl3MoeHarness;
const attn256MaxAbs = test_support.attn256MaxAbs;
const attn256MaxDiff = test_support.attn256MaxDiff;
const attn256RandBf16 = test_support.attn256RandBf16;
const attn256RandBf16Scaled = test_support.attn256RandBf16Scaled;
const attn256Reference = test_support.attn256Reference;
const bf16Trunc = test_support.bf16Trunc;
const evalToF32 = test_support.evalToF32;
const mimoExpectClose = test_support.mimoExpectClose;
const mlxSettledActiveBytes = test_support.mlxSettledActiveBytes;
const qkvExpectBitsEqual = test_support.qkvExpectBitsEqual;
const qkvIdentityArray = test_support.qkvIdentityArray;
const qsaArraysAllEqual = test_support.qsaArraysAllEqual;
const qwen4ReadF32 = test_support.qwen4ReadF32;
const routerRefRow = test_support.routerRefRow;
const snapshotsBitEqual = test_support.snapshotsBitEqual;
const testKVWide = test_support.testKVWide;
const testRandUniformBf16 = test_support.testRandUniformBf16;
const testRandWeightBf16 = test_support.testRandWeightBf16;
const testReadF32 = test_support.testReadF32;
const qwen4_test_support = @import("qwen4/qwen4_test_support.zig");
const expectRowsByteEqual = qwen4_test_support.expectRowsByteEqual;
const expectSlotStateEqual = qwen4_test_support.expectSlotStateEqual;
const logicalSsmCapture = qwen4_test_support.logicalSsmCapture;
const qwen4BatchedPleForward = qwen4_test_support.qwen4BatchedPleForward;
const qwen4ExpectMoeBitsEqual = qwen4_test_support.qwen4ExpectMoeBitsEqual;
const glm5_serving = @import("glm5/glm5_serving.zig");
const forwardGlm5WithImpl = glm5_serving.forwardGlm5WithImpl;
const initGlm5 = glm5_serving.initGlm5;
const dsv41_serving = @import("dsv41/dsv41_serving.zig");
const dsv41DraftImpl = dsv41_serving.dsv41DraftImpl;
const dsv41FinishImpl = dsv41_serving.dsv41FinishImpl;
const dsv41VerifyImpl = dsv41_serving.dsv41VerifyImpl;
const forwardDsv41With = dsv41_serving.forwardDsv41With;
const initDsv41 = dsv41_serving.initDsv41;
pub const DenseKVView = kv_cache.DenseKVView;
pub const KVCache = kv_cache.KVCache;
pub const KVCacheEntry = kv_cache.KVCacheEntry;
pub const KVCacheSnapshot = kv_cache.KVCacheSnapshot;
pub const KvPrefixView = kv_cache.KvPrefixView;
pub const MAX_BATCH_ROWS = kv_cache.MAX_BATCH_ROWS;
pub const MIMO_VERIFY_ROWS_MAX = mimo_forward.MIMO_VERIFY_ROWS_MAX;
pub const OneShotBits = qwen4_qsa_attn.OneShotBits;
pub const PersistentSsmGroup = ssm_cache.PersistentSsmGroup;
pub const QSA_ATTN_PACKED_MAX_S = qwen4_qsa_attn.QSA_ATTN_PACKED_MAX_S;
pub const QSA_HEAD_MARKS_MAX = qwen4_qsa.QSA_HEAD_MARKS_MAX;
pub const QSA_MAX_COMPRESS_RATIO = ssm_cache.QSA_MAX_COMPRESS_RATIO;
pub const QSA_PACKED_GATHER_MAX_REUSE = qwen4_qsa_attn.QSA_PACKED_GATHER_MAX_REUSE;
pub const QSA_RING_ROWS = ssm_cache.QSA_RING_ROWS;
pub const QsaArm = qwen4_qsa_attn.QsaArm;
const QsaArmCounts = qwen4_qsa_attn.QsaArmCounts;
const QsaBlockConsts = qwen4_qsa_select.QsaBlockConsts;
pub const QsaBlockFixture = qwen4_qsa_gather.QsaBlockFixture;
pub const QsaHeadMark = qwen4_qsa.QsaHeadMark;
pub const QsaHeadMarkSet = qwen4_qsa.QsaHeadMarkSet;
const QsaPooledRope = qwen4_qsa_select.QsaPooledRope;
pub const QsaSelectCfgKey = qwen4_qsa_select.QsaSelectCfgKey;
pub const QsaSelectSplitCfgKey = qwen4_qsa_select.QsaSelectSplitCfgKey;
pub const Qwen4Mtp = qwen4_forward.Qwen4Mtp;
const PlePending = qwen4_hc.PlePending;
pub const SLIDING_TRIM_UNBOUNDED = kv_cache.SLIDING_TRIM_UNBOUNDED;
pub const SSMCacheEntry = ssm_cache.SSMCacheEntry;
pub const SSMCacheEntrySnapshot = ssm_cache.SSMCacheEntrySnapshot;
pub const SSMCheckpoint = ssm_cache.SSMCheckpoint;
pub const SSM_RESIDENT_MAX = ssm_cache.SSM_RESIDENT_MAX;
pub const SsmMergeKind = ssm_cache.SsmMergeKind;
pub const ThinPolicy = ssm_cache.ThinPolicy;
pub const applyQsaHistoryAt = qwen4_qsa_state.applyQsaHistoryAt;
pub const arrayBytes = ssm_cache.arrayBytes;
pub const attachQsaHistoryToLatest = qwen4_qsa_state.attachQsaHistoryToLatest;
pub const capBufAppend = ssm_cache.capBufAppend;
pub const capturePrefillHidden = ssm_cache.capturePrefillHidden;
pub const capturePrefillHiddenLast = ssm_cache.capturePrefillHiddenLast;
pub const captureSsmCheckpoint = ssm_cache.captureSsmCheckpoint;
const checkQsaCompressRatio = qwen4_qsa_state.checkQsaCompressRatio;
pub const checkpointHasQsaHistory = qwen4_qsa_state.checkpointHasQsaHistory;
pub const checkpointHasQsaPooled = qwen4_qsa_state.checkpointHasQsaPooled;
pub const checkpointListHasQsaPooled = qwen4_qsa_state.checkpointListHasQsaPooled;
pub const checkpointQsaCoversPos = qwen4_qsa_state.checkpointQsaCoversPos;
pub const compactConvStateEnabled = ssm_cache.compactConvStateEnabled;
const computeYarnFreqs = qwen4_forward.computeYarnFreqs;
pub const entriesHaveQsaHistory = qwen4_qsa_state.entriesHaveQsaHistory;
const evalQwen4MtpResident = qwen4_forward.evalQwen4MtpResident;
const fillPleSpecTokens = qwen4_hc.fillPleSpecTokens;
const freeKVEntry = kv_cache.freeKVEntry;
pub const fusedSdpa256Masked = qwen4_qsa_gather.fusedSdpa256Masked;
pub const gatherQsa256 = qwen4_qsa_gather.gatherQsa256;
pub const gatherQsa256Packed = qwen4_qsa_gather.gatherQsa256Packed;
pub const handoffQsaHistoryToLatest = qwen4_qsa_state.handoffQsaHistoryToLatest;
pub const keepOnlyLatestQsaHistory = qwen4_qsa_state.keepOnlyLatestQsaHistory;
pub const kvEntriesBytes = ssm_cache.kvEntriesBytes;
pub const materializedOwnedCopy = ssm_cache.materializedOwnedCopy;
pub const mimoGlobalDecodeRebuildMaxKeys = mimo_forward.mimoGlobalDecodeRebuildMaxKeys;
pub const nextSsmMemberGen = qwen4_ssm_group.nextSsmMemberGen;
pub const newEmptyKVEntry = kv_cache.newEmptyKVEntry;
pub const positionDropIndex = ssm_cache.positionDropIndex;
pub const positionDropIndexUsize = ssm_cache.positionDropIndexUsize;
pub const qsaAttnKernelEnabled = qwen4_qsa_attn.qsaAttnKernelEnabled;
pub const qsaBatchedGatherEnabled = qwen4_qsa_gather.qsaBatchedGatherEnabled;
pub const qsaBatchedGatherFloor = qwen4_qsa_gather.qsaBatchedGatherFloor;
pub const qsaBatchedGatherOn = qwen4_qsa_gather.qsaBatchedGatherOn;
pub const qsaDecodeGatherAttn = qwen4_qsa_attn.qsaDecodeGatherAttn;
pub const qsaDecodeGatherEnabled = qwen4_qsa_gather.qsaDecodeGatherEnabled;
pub const qsaDecodeGatherIdsHost = qwen4_qsa_gather.qsaDecodeGatherIdsHost;
pub const qsaDenseRebuildRows = qwen4_qsa_attn.qsaDenseRebuildRows;
pub const qsaEntrySatisfiesForward = qwen4_qsa_state.qsaEntrySatisfiesForward;
pub const qsaGatherEnabled = qwen4_qsa_gather.qsaGatherEnabled;
pub const qsaGatherMinKv = qwen4_qsa_gather.qsaGatherMinKv;
pub const qsaHistoryRows = qwen4_qsa_state.qsaHistoryRows;
pub const qsaLeftoverAt = qwen4_qsa_state.qsaLeftoverAt;
pub const qsaMaskFromBlockSel = qwen4_qsa_attn.qsaMaskFromBlockSel;
pub const qsaMaskFromBlocks = qwen4_qsa_attn.qsaMaskFromBlocks;
const qsaNaxArm = qwen4_qsa_gather.qsaNaxArm;
pub const qsaNaxEnabled = qwen4_qsa_gather.qsaNaxEnabled;
pub const qsaPoolNormRopeFused = qwen4_qsa_select.qsaPoolNormRopeFused;
pub const qsaPrefillGatherMinKvFrom = qwen4_qsa_gather.qsaPrefillGatherMinKvFrom;
pub const qsaPrefillTransientBytes = qwen4_qsa_attn.qsaPrefillTransientBytes;
pub const qsaResliceToKeep = qwen4_qsa_state.qsaResliceToKeep;
pub const qsaRestoreSatisfiesForward = qwen4_qsa_state.qsaRestoreSatisfiesForward;
pub const qsaScoreFusedActive = qwen4_qsa_select.qsaScoreFusedActive;
pub const qsaScoreFusedActiveFor = qwen4_qsa_select.qsaScoreFusedActiveFor;
pub const qsaScoreFusedArm = qwen4_qsa_select.qsaScoreFusedArm;
pub const qsaScoreFusedDispatch = qwen4_qsa_select.qsaScoreFusedDispatch;
pub const qsaScoreRowsPerChunk = qwen4_qsa_select.qsaScoreRowsPerChunk;
pub const qsaScoreRowsPerChunkFused = qwen4_qsa_attn.qsaScoreRowsPerChunkFused;
pub const qsaScoreSheetComposed = qwen4_qsa_select.qsaScoreSheetComposed;
pub const qsaSelectKernelEnabled = qwen4_qsa_select.qsaSelectKernelEnabled;
pub const qsaSelectSplitEnabled = qwen4_qsa_select.qsaSelectSplitEnabled;
pub const qsaSparseAttn = qwen4_qsa_attn.qsaSparseAttn;
pub const qsaSparseAttnServes = qwen4_qsa_attn.qsaSparseAttnServes;
pub const qsaTestAppendPool = qwen4_qsa_state.qsaTestAppendPool;
pub const qsaVerifyGatherAttn = qwen4_qsa_attn.qsaVerifyGatherAttn;
pub const qsaVerifyGatherEnabled = qwen4_qsa_attn.qsaVerifyGatherEnabled;
pub const qsaVerifyGatherMinKvFor = qwen4_qsa_attn.qsaVerifyGatherMinKvFor;
pub const qsaWidthBucket = qwen4_qsa_attn.qsaWidthBucket;
pub const reserveQsaHistoryWithHead = qwen4_qsa_state.reserveQsaHistoryWithHead;
pub const resetKVEntry = kv_cache.resetKVEntry;
pub const restoreSsmCheckpoint = ssm_cache.restoreSsmCheckpoint;
pub const seedCapBuf = ssm_cache.seedCapBuf;
pub const shareSsmCheckpoint = ssm_cache.shareSsmCheckpoint;
pub const sliceQsaHistoryOntoCheckpoint = qwen4_qsa_state.sliceQsaHistoryOntoCheckpoint;
pub const slidingTailSpan = kv_cache.slidingTailSpan;
pub const slidingViewFor = kv_cache.slidingViewFor;
pub const snapshotHasQsaHistory = qwen4_qsa_state.snapshotHasQsaHistory;
pub const spanPreservingDropIndex = ssm_cache.spanPreservingDropIndex;
pub const splitSpecCaptures = ssm_cache.splitSpecCaptures;
pub const ssmBindSlotRefs = qwen4_ssm_group.ssmBindSlotRefs;
pub const ssmBindTickState = qwen4_ssm_group.ssmBindTickState;
const ssmConcatAxis0 = qwen4_ssm_group.ssmConcatAxis0;
pub const ssmDetachFromGroup = qwen4_ssm_group.ssmDetachFromGroup;
pub const ssmCheckpointBytes = ssm_cache.ssmCheckpointBytes;
pub const ssmCheckpointDropIndex = ssm_cache.ssmCheckpointDropIndex;
pub const ssmEntryBytes = ssm_cache.ssmEntryBytes;
pub const ssmEvalHandles = ssm_cache.ssmEvalHandles;
pub const ssmFreeMergedLayer = ssm_cache.ssmFreeMergedLayer;
pub const ssmFreeQsaState = qwen4_qsa_state.ssmFreeQsaState;
const ssmGroupDropMember = qwen4_ssm_group.ssmGroupDropMember;
const ssmGroupReleaseCtx = qwen4_ssm_group.ssmGroupReleaseCtx;
const ssmMergeLayer = qwen4_ssm_group.ssmMergeLayer;
pub const ssmSplitAll = qwen4_ssm_group.ssmSplitAll;
const ssmSplitLayer = qwen4_ssm_group.ssmSplitLayer;
pub const ssmFreeSpecCapture = ssm_cache.ssmFreeSpecCapture;
pub const ssmHasSpecCapture = ssm_cache.ssmHasSpecCapture;
pub const ssmLiveAux = ssm_cache.ssmLiveAux;
pub const ssmMaterializeLive = ssm_cache.ssmMaterializeLive;
pub const ssmRestore = ssm_cache.ssmRestore;
pub const ssmRollbackFromCapture = ssm_cache.ssmRollbackFromCapture;
pub const ssmRowView = ssm_cache.ssmRowView;
pub const ssmSnapshot = ssm_cache.ssmSnapshot;
pub const ssmSnapshotDeinit = ssm_cache.ssmSnapshotDeinit;
pub const ssmUnbindAll = ssm_cache.ssmUnbindAll;
pub const ssmUnbindEntries = ssm_cache.ssmUnbindEntries;
pub const stackMimoFixtureExperts = mimo_forward.stackMimoFixtureExperts;
pub const qwen4Standin = qwen4_forward.qwen4Standin;
const sharedGateRowsEligible = qwen4_forward.sharedGateRowsEligible;
pub const splitMaskedSdpa256 = qwen4_forward.splitMaskedSdpa256;
pub const truncatePooled = ssm_cache.truncatePooled;
pub const warmQsaEnvCaches = qwen4_qsa_attn.warmQsaEnvCaches;
const yarnMscale = qwen4_forward.yarnMscale;
pub const yarnSpec = qwen4_forward.yarnSpec;

pub const KVQuantConfig = kv_quant.KVQuantConfig;
pub const KvCacheChoice = kv_quant.KvCacheChoice;

/// `std.meta.fields` was replaced by parallel `fieldNames`/`fieldTypes`
/// arrays — zip them into the `.name`/`.type` shape every call site below
/// already expects.
const WeightField = struct { name: [:0]const u8, type: type };

fn structFields(comptime T: type) []const WeightField {
    return comptime blk: {
        const names = std.meta.fieldNames(T);
        const types = std.meta.fieldTypes(T);
        var result: [names.len]WeightField = undefined;
        for (names, types, 0..) |name, ty, i| result[i] = .{ .name = name, .type = ty };
        const final = result;
        break :blk &final;
    };
}

// ── GatedDeltaNet fused Metal kernel ──
// Ported from mlx-lm/models/gated_delta.py: `_make_gated_delta_kernel(has_mask=False, vectorized=False)`.
// Processes the entire T-step delta recurrence in a single kernel dispatch, eliminating
// the per-token kernel-launch overhead that otherwise caps prefill at ~330 tok/s on
// Qwen 3.5/3.6 MoE. Template args (Dk, Dv, Hk, Hv) specialize the kernel; inputs carry
// the runtime shapes. State math runs in float32 for numerical stability regardless
// of the input/state storage dtype.
// The recurrence exists ONCE, specialized on two independent axes:
//
//   `vectorized` — where the forget gate lives. Flash-Next's GatedDeltaNet has one
//     scalar per (token, value head), so `g` is [B,T,Hv]. KDA (Kimi Delta
//     Attention, GLM) gates per KEY CHANNEL, so `g` is [B,T,Hv,Dk]
//     and each state element decays by its own factor. Mirrors mlx-lm's
//     `_make_gated_delta_kernel(vectorized=…)`, which parameterizes the same
//     three lines.
//
//   `capture_seq` — spec-decode support, used ONLY on the verify forward (small
//     T). Besides the FINAL state (`state_out`, written from registers like the
//     stock kernel) it records the state after every INTERMEDIATE timestep into
//     `state_seq` ([T, B, Hv, Dv, Dk]), so partial-accept rollback restores the
//     accepted position by SLICING — no re-forward of the accepted prefix (the
//     costly part on a 48-layer trunk). Decode and prefill never pay for it.
//     `seq_stride` = (B*Hv)*Dv*Dk = the per-timestep element stride into
//     `state_seq` (passed as a scalar, mirroring `T`).
//     Capture-tail trim: state_seq[T-1] is deliberately NEVER written — a
//     partial accept reads index `accepted` <= T-2 (index T-1 would be a full
//     accept, which keeps `state_out` via the normal flow), so the last row was
//     a redundant global write + forced the engine's final state to be a slice
//     VIEW pinning the whole [T,...] capture buffer across rounds.
fn gdnKernelSource(comptime vectorized: bool, comptime capture_seq: bool) [:0]const u8 {
    const g_setup = if (vectorized)
        \\auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;
    else
        \\auto g_ = g + b_idx * T * Hv;
    ;
    const g_access = if (vectorized) "g_[s_idx]" else "g_[hv_idx]";
    const g_advance = if (vectorized) "  g_ += Hv * Dk;" else "  g_ += Hv;";
    const seq_base = if (capture_seq)
        \\auto seq_base = (n * Dv + dv_idx) * Dk;
        \\
    else
        "";
    const seq_write = if (capture_seq)
        \\  if (t + 1 < T) {
        \\    auto t_state = state_seq + t * seq_stride + seq_base;
        \\    for (int i = 0; i < n_per_t; ++i) {
        \\      auto s_idx = n_per_t * dk_idx + i;
        \\      t_state[s_idx] = static_cast<StT>(state[i]);
        \\      state[i] = static_cast<float>(t_state[s_idx]);
        \\    }
        \\  }
        \\
    else
        "";
    // comptimePrint re-terminates the concatenation: the `if` expressions above
    // are slices, so `++` alone would drop the NUL the mlx-c entry point needs.
    const body = "auto n = thread_position_in_grid.z;\n" ++
        "auto b_idx = n / Hv;\n" ++
        "auto hv_idx = n % Hv;\n" ++
        "auto hk_idx = hv_idx / (Hv / Hk);\n" ++
        "constexpr int n_per_t = Dk / 32;\n" ++
        "\n" ++
        "auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;\n" ++
        "auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;\n" ++
        "\n" ++
        "auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;\n" ++
        "y += b_idx * T * Hv * Dv + hv_idx * Dv;\n" ++
        "\n" ++
        "auto dk_idx = thread_position_in_threadgroup.x;\n" ++
        "auto dv_idx = thread_position_in_grid.y;\n" ++
        "\n" ++
        "auto i_state = state_in + (n * Dv + dv_idx) * Dk;\n" ++
        "auto o_state = state_out + (n * Dv + dv_idx) * Dk;\n" ++
        seq_base ++
        "\n" ++
        "float state[n_per_t];\n" ++
        "for (int i = 0; i < n_per_t; ++i) {\n" ++
        "  auto s_idx = n_per_t * dk_idx + i;\n" ++
        "  state[i] = static_cast<float>(i_state[s_idx]);\n" ++
        "}\n" ++
        "\n" ++
        g_setup ++ "\n" ++
        "auto beta_ = beta + b_idx * T * Hv;\n" ++
        "\n" ++
        "for (int t = 0; t < T; ++t) {\n" ++
        "  float kv_mem = 0.0f;\n" ++
        "  for (int i = 0; i < n_per_t; ++i) {\n" ++
        "    auto s_idx = n_per_t * dk_idx + i;\n" ++
        "    state[i] = state[i] * " ++ g_access ++ ";\n" ++
        "    kv_mem += state[i] * k_[s_idx];\n" ++
        "  }\n" ++
        "  kv_mem = simd_sum(kv_mem);\n" ++
        "\n" ++
        "  auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];\n" ++
        "\n" ++
        "  float out = 0.0f;\n" ++
        "  for (int i = 0; i < n_per_t; ++i) {\n" ++
        "    auto s_idx = n_per_t * dk_idx + i;\n" ++
        "    state[i] = state[i] + k_[s_idx] * delta;\n" ++
        "    out += state[i] * q_[s_idx];\n" ++
        "  }\n" ++
        "  out = simd_sum(out);\n" ++
        "  if (thread_index_in_simdgroup == 0) {\n" ++
        "    y[dv_idx] = static_cast<OutT>(out);\n" ++
        "  }\n" ++
        seq_write ++
        "  q_ += Hk * Dk;\n" ++
        "  k_ += Hk * Dk;\n" ++
        "  v_ += Hv * Dv;\n" ++
        "  y += Hv * Dv;\n" ++
        g_advance ++ "\n" ++
        "  beta_ += Hv;\n" ++
        "}\n" ++
        "for (int i = 0; i < n_per_t; ++i) {\n" ++
        "  auto s_idx = n_per_t * dk_idx + i;\n" ++
        "  o_state[s_idx] = static_cast<StT>(state[i]);\n" ++
        "}\n";
    return std.fmt.comptimePrint("{s}", .{body});
}

/// One compiled kernel per (vectorized, capture_seq) pair. Each comptime
/// instantiation gets its own cache slot; the Metal name must differ per
/// variant or MLX hands back the wrong compiled function.
pub fn getGdnKernelFor(comptime vectorized: bool, comptime capture_seq: bool) !mlx.mlx_fast_metal_kernel {
    // The two `const`s are load-bearing: a struct declared inside a function
    // that captures NOTHING from the comptime parameters is memoized to ONE
    // type across every instantiation, so all four variants would share a
    // single cache slot and the first kernel compiled would be handed back for
    // the rest (a seq kernel answering a non-seq call = wrong output count).
    const Cache = struct {
        const is_vectorized = vectorized;
        const is_capture_seq = capture_seq;
        var kernel: ?mlx.mlx_fast_metal_kernel = null;
    };
    if (Cache.kernel) |k| return k;
    const input_names = if (capture_seq)
        [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T", "seq_stride" }
    else
        [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
    const output_names = if (capture_seq)
        [_][*:0]const u8{ "y", "state_seq", "state_out" }
    else
        [_][*:0]const u8{ "y", "state_out" };
    const name = "gated_delta_step" ++
        (if (capture_seq) "_seq" else "") ++
        (if (vectorized) "_vec" else "");
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        name,
        in_vec,
        out_vec,
        gdnKernelSource(vectorized, capture_seq),
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    Cache.kernel = kernel;
    return kernel;
}

pub fn getGdnKernel(vector_gate: bool) !mlx.mlx_fast_metal_kernel {
    return if (vector_gate) getGdnKernelFor(true, false) else getGdnKernelFor(false, false);
}

// ── Verify-width split-K quantized matmul (spec-decode fast path) ──
//
// Stock MLX qmm is tuned for M=1 decode (qmv) and large-M prefill (steel);
// the M=2..8-row shapes of speculative VERIFY forwards (and small decode
// batches) fall in a dead zone that underuses memory bandwidth, and at huge
// N (lm_head) the tiny-tile grid thrashes the scheduler. This kernel is a
// port of MTPLX's split-K verify family (mtplx/verify_kernels.py,
// Apache-2.0; design + measurement ledger in that file's header — their
// in-context winner over the wide multi-simdgroup tile): each threadgroup
// owns FOUR output columns, the K reduction splits across K_PARTS
// simdgroups (2 for big N, 4 for small N — deep occupancy queues + latency
// hiding under mixed scheduling with attention/GDN kernels), per-column
// sequential dequant+FMA chains (the interleaved construction measurably
// loses), fp32 accumulation, one threadgroup-barrier partial reduction.
// MROWS/GS/K_PARTS ride as template ints so MLX caches one specialization
// per shape class. 4-bit affine only (the shipping trunk class); everything
// else falls through to stock qmm. Numerics: fp32 accumulate in a different
// order than stock → bf16 tail-ULP class differences (same accepted class
// as every fused kernel here); parity pinned by the verifyQmm test.
//
// NAX lane (M5-class machines): on "applegpu_g17" GPUs under macOS >= 26.2
// a fixed 16x32x16 tensor-ops tile (MetalPerformancePrimitives matmul2d on
// the per-core matrix units) extends the family to M 8..16 — past the
// plain-SIMD register cliff at M=8 (T(7) rounds measured 636 ms vs 115
// stock on the M4) — covering deep MTP verify widths (seq 8..9), the merged
// MTP-head forward at high accepted counts, lm_head at M 8..16, and small
// decode batches. Third-generation port; keep the attribution chain:
// DFlash (arXiv:2602.06036) → bstnxbt/dflash-mlx verify_qmm.py (Apache-2.0)
// → MTPLX nax_verify.py (Apache-2.0: the M-padding dispatch, env gating,
// availability probes) → us (plan: todo-m5-nax.md). The kernel object is
// NEVER built — not just never dispatched — where the probe is false:
// matmul2d<.., execution_simdgroup> pipeline creation can fail on non-G17
// hardware. Switch: SUSHI_FORCE_GPU_FAMILY_FALLBACK=1 pretends the units are absent (QA
// rehearsal of the exact M1-M4 path on an M5).
/// Runtime codegen of the per-M split-K kernel body. The row loads, weight
/// loads, and dequant+FMA chains are emitted as NAMED SCALARS with LITERAL
/// accumulator indices — the ledger's load-bearing constraint: array-indexed
/// rows behind a runtime-ish loop stack-spill (measured 10x at M=6 with the
/// `Vec8 v[MROWS]` form of this same kernel). GS and K_PARTS stay template
/// ints; M is baked into the source (one cached kernel per M, distinct Metal
/// host_names — two specializations sharing a name bind the wrong binary).
/// The activation-FMA lines one output column contributes for a single K
/// value: `wv` is the dequantized weight, `idx` the element inside the Vec8
/// activation group. Shared by both plain-SIMD verify tiles so the two cannot
/// drift.
fn vqmmAccLines(a: std.mem.Allocator, m: usize, j: usize, idx: []const u8) std.mem.Allocator.Error![]const u8 {
    var out: []const u8 = "";
    for (0..m) |r| {
        out = try vqmmCat(a, out, try std.fmt.allocPrint(a, "        acc[{d}] += float(v{d}[{s}]) * wv;\n", .{ j * m + r, r, idx }));
    }
    return out;
}

/// Hoisted weight WORDS for one column, decoded per width.
///
/// One `pack` iteration always covers 8 K values (that is the Vec8 activation
/// load), so its weights are exactly BITS bytes at
/// `w_q + n * K_bytes + pack * BITS`. The loads stay HOISTED above the FMA
/// chains — the tile's whole shape is "issue every column's weight word, then
/// run one sequential chain per column", and leaving the load inside the chain
/// serializes it behind the previous column's arithmetic. Hoisting only the
/// POINTER is not enough: the widened arms must hoist the BYTES.
///
/// Widest load each width's alignment allows. 4-bit keeps its original aligned
/// uint32 verbatim — that width must stay bit-identical, and it is this
/// refactor's tripwire. 8-bit sits on `pack * 8` and takes two uints (not one
/// ulong: 4-byte alignment is what the shipped 4-bit path already assumes of
/// this buffer, and widening that assumption for one fused load buys nothing).
/// 6-bit lands on `pack * 6`, only 2-aligned, so three ushorts re-assembled
/// into two 3-byte words. 5-bit on `pack * 5` is 1-aligned and takes bytes.
fn vqmmColumnLoad(a: std.mem.Allocator, j: usize) std.mem.Allocator.Error![]const u8 {
    return std.fmt.allocPrint(a,
        \\  uint32_t p{d}; ulong r{d}; uint32_t a{d}0; uint32_t a{d}1;
        \\  if constexpr (BITS == 4) {{ p{d} = w_q[(n0 + {d}) * K_by_p + pack]; }}
        \\  else {{
        \\    const device uchar* wp = ((const device uchar*)w_q) + (n0 + {d}) * K_bytes + pack * BITS;
        \\    if constexpr (BITS == 8) {{
        \\      const device uint* wu = (const device uint*)wp;
        \\      r{d} = ulong(wu[0]) | (ulong(wu[1]) << 32);
        \\    }} else if constexpr (BITS == 5) {{
        \\      r{d} = ulong(wp[0]) | (ulong(wp[1]) << 8) | (ulong(wp[2]) << 16) |
        \\             (ulong(wp[3]) << 24) | (ulong(wp[4]) << 32);
        \\    }} else {{
        \\      const device ushort* wh = (const device ushort*)wp;
        \\      uint32_t h0 = uint32_t(wh[0]); uint32_t h1 = uint32_t(wh[1]); uint32_t h2 = uint32_t(wh[2]);
        \\      a{d}0 = h0 | ((h1 & 0xFFu) << 16);
        \\      a{d}1 = (h1 >> 8) | (h2 << 8);
        \\    }}
        \\  }}
        \\
    , .{ j, j, j, j, j, j, j, j, j, j, j });
}

/// Per-column dequant + FMA chain, one arm per supported affine width. The
/// byte-addressed arms are lifted from the NAX m16 tile's validated unpack
/// (4 values per 3 bytes at 6 bits, 8 per 5 bytes at 5, one byte per value at
/// 8) — this is not a fresh reading of MLX's packing.
fn vqmmColumnChain(a: std.mem.Allocator, m: usize, j: usize) std.mem.Allocator.Error![]const u8 {
    const acc_ki = try vqmmAccLines(a, m, j, "ki");
    const acc_hi = try vqmmAccLines(a, m, j, "4 + ki");
    return std.fmt.allocPrint(a,
        \\  {{
        \\    float sj = s{d}; float bj = b{d};
        \\    if constexpr (BITS == 4) {{
        \\      for (int ki = 0; ki < 8; ++ki) {{
        \\        float wv = float((p{d} >> (ki * 4)) & 0xFu) * sj + bj;
        \\{s}      }}
        \\    }} else if constexpr (BITS == 8) {{
        \\      for (int ki = 0; ki < 8; ++ki) {{
        \\        float wv = float(uint32_t((r{d} >> (ki * 8)) & 0xFFul)) * sj + bj;
        \\{s}      }}
        \\    }} else if constexpr (BITS == 5) {{
        \\      for (int ki = 0; ki < 8; ++ki) {{
        \\        float wv = float(uint32_t((r{d} >> (ki * 5)) & 0x1Ful)) * sj + bj;
        \\{s}      }}
        \\    }} else {{
        \\      for (int ki = 0; ki < 4; ++ki) {{
        \\        float wv = float((a{d}0 >> (ki * 6)) & 0x3Fu) * sj + bj;
        \\{s}      }}
        \\      for (int ki = 0; ki < 4; ++ki) {{
        \\        float wv = float((a{d}1 >> (ki * 6)) & 0x3Fu) * sj + bj;
        \\{s}      }}
        \\    }}
        \\  }}
        \\
    , .{ j, j, j, acc_ki, j, acc_ki, j, acc_ki, j, acc_ki, j, acc_hi });
}

fn verifyQmmSource(a: std.mem.Allocator, m: usize, bn: usize) std.mem.Allocator.Error![:0]const u8 {
    const nacc = bn * m;
    var body: []const u8 = "";
    // Row activation loads, all up front.
    for (0..m) |r| {
        body = try vqmmCat(a, body, try std.fmt.allocPrint(a, "  Vec8 v{d} = xv[({d} * K + k_base) / 8];\n", .{ r, r }));
    }
    // Weight handles for the owned columns, then scales/biases.
    for (0..bn) |j| {
        body = try vqmmCat(a, body, try vqmmColumnLoad(a, j));
    }
    for (0..bn) |j| {
        body = try vqmmCat(a, body, try std.fmt.allocPrint(a, "  float s{d} = float(scales[(n0 + {d}) * K_by_gs + gi]); float b{d} = float(biases[(n0 + {d}) * K_by_gs + gi]);\n", .{ j, j, j, j }));
    }
    // One sequential dequant+FMA chain per output column (column-major
    // construction; the interleaved form measurably loses).
    for (0..bn) |j| {
        body = try vqmmCat(a, body, try vqmmColumnChain(a, m, j));
    }
    var acc_init: []const u8 = "";
    for (0..nacc) |i| {
        acc_init = try vqmmCat(a, acc_init, try std.fmt.allocPrint(a, "acc[{d}] = 0.0f; ", .{i}));
    }
    var acc_sum: []const u8 = "";
    for (0..nacc) |i| {
        acc_sum = try vqmmCat(a, acc_sum, try std.fmt.allocPrint(a, "acc[{d}] = simd_sum(acc[{d}]); ", .{ i, i }));
    }
    return std.fmt.allocPrintSentinel(a,
        \\auto part = simdgroup_index_in_threadgroup;
        \\auto lane = thread_index_in_simdgroup;
        \\auto tg_n = threadgroup_position_in_grid.y;
        \\
        \\int K = int(K_size);
        \\int N = int(N_size);
        \\int K_by_p = K / 8;
        \\int K_bytes = (K * BITS) / 8;
        \\int K_by_gs = K / GS;
        \\int per_part = K_by_p / K_PARTS;
        \\int n0 = int(tg_n) * {d};
        \\int p_start = int(part) * per_part;
        \\int p_end = (int(part) == K_PARTS - 1) ? K_by_p : p_start + per_part;
        \\
        \\float acc[{d}];
        \\{s}
        \\
        \\using Vec8 = vec<T, 8>;
        \\const device Vec8 *xv = (const device Vec8*)x;
        \\
        \\for (int pack = p_start + int(lane); pack < p_end; pack += 32) {{
        \\  int k_base = pack * 8;
        \\  int gi = k_base / GS;
        \\{s}
        \\}}
        \\
        \\{s}
        \\
        \\threadgroup float partials[K_PARTS * {d}];
        \\if (lane == 0) {{
        \\  _Pragma("unroll")
        \\  for (int i = 0; i < {d}; ++i) {{
        \\    partials[int(part) * {d} + i] = acc[i];
        \\  }}
        \\}}
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\
        \\if (part == 0) {{
        \\  for (int i = int(lane); i < {d}; i += 32) {{
        \\    float total = 0.0f;
        \\    _Pragma("unroll")
        \\    for (int p = 0; p < K_PARTS; ++p) {{
        \\      total += partials[p * {d} + i];
        \\    }}
        \\    int j = i / {d};
        \\    int row = i - j * {d};
        \\    y[row * N + n0 + j] = T(total);
        \\  }}
        \\}}
    , .{ bn, nacc, acc_init, body, acc_sum, nacc, nacc, nacc, nacc, nacc, m, m }, 0);
}

/// Wide multi-simdgroup tile for HUGE N (the lm_head class): NSG independent
/// simdgroups ride one threadgroup purely to give the scheduler fewer,
/// heavier units — the split-K tiny-tile grid measurably thrashes there
/// (2.1x stock at M=4), while this tile holds 1.18x off the weight-stream
/// floor per the source ledger. Full-K per simdgroup, no barrier.
fn verifyQmmMsgSource(a: std.mem.Allocator, m: usize, bn: usize) std.mem.Allocator.Error![:0]const u8 {
    const nacc = bn * m;
    var body: []const u8 = "";
    for (0..m) |r| {
        body = try vqmmCat(a, body, try std.fmt.allocPrint(a, "  Vec8 v{d} = xv[({d} * K + k_base) / 8];\n", .{ r, r }));
    }
    for (0..bn) |j| {
        body = try vqmmCat(a, body, try vqmmColumnLoad(a, j));
    }
    for (0..bn) |j| {
        body = try vqmmCat(a, body, try std.fmt.allocPrint(a, "  float s{d} = float(scales[(n0 + {d}) * K_by_gs + gi]); float b{d} = float(biases[(n0 + {d}) * K_by_gs + gi]);\n", .{ j, j, j, j }));
    }
    for (0..bn) |j| {
        body = try vqmmCat(a, body, try vqmmColumnChain(a, m, j));
    }
    var acc_init: []const u8 = "";
    for (0..nacc) |i| {
        acc_init = try vqmmCat(a, acc_init, try std.fmt.allocPrint(a, "acc[{d}] = 0.0f; ", .{i}));
    }
    var acc_sum: []const u8 = "";
    for (0..nacc) |i| {
        acc_sum = try vqmmCat(a, acc_sum, try std.fmt.allocPrint(a, "acc[{d}] = simd_sum(acc[{d}]); ", .{ i, i }));
    }
    return std.fmt.allocPrintSentinel(a,
        \\auto sg = simdgroup_index_in_threadgroup;
        \\auto lane = thread_index_in_simdgroup;
        \\auto tg_n = threadgroup_position_in_grid.y;
        \\
        \\int K = int(K_size);
        \\int N = int(N_size);
        \\int K_by_p = K / 8;
        \\int K_bytes = (K * BITS) / 8;
        \\int K_by_gs = K / GS;
        \\int n0 = (int(tg_n) * NSG + int(sg)) * {d};
        \\if (n0 + {d} >= N) {{ return; }}
        \\
        \\float acc[{d}];
        \\{s}
        \\
        \\using Vec8 = vec<T, 8>;
        \\const device Vec8 *xv = (const device Vec8*)x;
        \\
        \\for (int pack = int(lane); pack < K_by_p; pack += 32) {{
        \\  int k_base = pack * 8;
        \\  int gi = k_base / GS;
        \\{s}
        \\}}
        \\
        \\{s}
        \\
        \\if (lane < {d}) {{
        \\  int j = int(lane) / {d};
        \\  int row = int(lane) - j * {d};
        \\  y[row * N + n0 + j] = T(acc[int(lane)]);
        \\}}
    , .{ bn, bn - 1, nacc, acc_init, body, acc_sum, nacc, m, m }, 0);
}

/// msg column-tile width per M: 4 columns through M=6, 2 for M=7 (14
/// accumulators — under the 24 ceiling; covers the depth-6 verify lm_head).
fn vqmmMsgBn(m: c_int) c_int {
    return if (m <= 6) 4 else 2;
}

/// Kernel text is built at runtime into a scratch arena: comptime string work for 24 tiles costs seconds of Sema.
fn vqmmCat(a: std.mem.Allocator, x: []const u8, y: []const u8) std.mem.Allocator.Error![]const u8 {
    return std.mem.concat(a, u8, &.{ x, y });
}

const VQMM_MSG_NAMES = [6][*:0]const u8{
    "sushi_vqmm_msg_m2", "sushi_vqmm_msg_m3", "sushi_vqmm_msg_m4",
    "sushi_vqmm_msg_m5", "sushi_vqmm_msg_m6", "sushi_vqmm_msg_m7",
};

var vqmm_msg_kernels: [6]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getVerifyQmmMsgKernel(m: c_int) !mlx.mlx_fast_metal_kernel {
    if (m < 2 or m > 7) return error.UnsupportedShape;
    const idx: usize = @intCast(m - 2);
    if (vqmm_msg_kernels[idx]) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const source = try verifyQmmMsgSource(arena.allocator(), @intCast(m), @intCast(vqmmMsgBn(m)));
    const kernel = mlx.mlx_fast_metal_kernel_new(
        VQMM_MSG_NAMES[idx],
        in_vec,
        out_vec,
        source,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    vqmm_msg_kernels[idx] = kernel;
    return kernel;
}

/// Column-tile width per M. A threadgroup owns `bn` output columns and
/// streams ALL M activation rows over its K partition, so the x traffic is
/// `N/bn * K * M * 2` bytes against `N * K * bits/8` of weight — at M=5,
/// K=6656, N=19968, bn=4 that is 332 MB of activation reads for 66 MB of
/// weights. Widening the tile is the direct lever on that ratio; the ceiling
/// is the accumulator budget (`bn * m` live floats per thread) and N's
/// divisibility by bn. Measured per (M, bn) by the width-sweep µbench.
pub const VQMM_BN_CHOICES = [_]c_int{ 2, 4, 8 };

/// Test seam: forces a column tile (0 = per-M default) so the parity test drives every tile.
pub var vqmm_bn_test_override: ?c_int = null;
fn vqmmBnOverride() c_int {
    return vqmm_bn_test_override orelse 0;
}

/// Column tile per M: 4 columns through M=6 (24 accumulators, the measured
/// register ceiling), 2 for M=7 (14) — one deep-verify width past MTPLX's
/// family. M=8 measured a CLIFF at bn=4 (T(7) round 636 ms vs 115 stock —
/// spill/occupancy past 7 live Vec8 row vectors), so 8+ falls back to stock.
///
/// WIDENING THE TILE IS DISPROVEN. The activation-traffic arithmetic above
/// predicts bn=8 should nearly halve the dominant read; measured on 4-bit
/// trunk shapes it is 5-8% SLOWER at every width (M=5: 45.9 ms vs 43.0
/// whole-forward, same direction at M 2/3/4). The x rows are 66 KB and stay
/// cache-resident, so the "traffic" was never leaving the cache — the extra
/// accumulators just cost occupancy. The ceiling below admits the 8-column
/// tile only through the test override.
pub const VQMM_MAX_ACC: c_int = 40;

/// Simdgroups per threadgroup, each owning a K partition. Wide-N shapes
/// already emit N/bn threadgroups, so the grid does not need more
/// parallelism there — 2 keeps the reduction cheap, and the sweep agrees:
/// on 4-bit trunk shapes at M=5 the whole-forward estimate is 42.4 ms at
/// 2, 42.5 at 1, 44.5 at 4, 49.0 at 8.
fn vqmmKParts(N: c_int) c_int {
    return if (N >= 4096) 2 else 4;
}

pub fn vqmmBnFor(m: c_int, N: c_int) c_int {
    const forced = vqmmBnOverride();
    if (forced != 0 and @mod(N, forced) == 0 and forced * m <= VQMM_MAX_ACC) return forced;
    const want: c_int = if (m <= 6) 4 else 2;
    return if (@mod(N, want) == 0) want else 0;
}

fn vqmmSourceIdx(m: c_int, bn: c_int) usize {
    const mi: usize = @intCast(m - 2);
    const bi: usize = switch (bn) {
        2 => 0,
        4 => 1,
        else => 2,
    };
    return mi * VQMM_BN_CHOICES.len + bi;
}

const VQMM_N_KERNELS = 6 * VQMM_BN_CHOICES.len;

const VQMM_NAMES = blk: {
    var out: [VQMM_N_KERNELS][*:0]const u8 = undefined;
    for (2..8) |m| {
        for (VQMM_BN_CHOICES, 0..) |bn, bi| {
            out[(m - 2) * VQMM_BN_CHOICES.len + bi] =
                std.fmt.comptimePrint("sushi_vqmm_ks_m{d}_n{d}", .{ m, bn });
        }
    }
    break :blk out;
};

var vqmm_kernels: [VQMM_N_KERNELS]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getVerifyQmmKernel(m: c_int, bn: c_int) !mlx.mlx_fast_metal_kernel {
    if (m < 2 or m > 7) return error.UnsupportedShape;
    const idx = vqmmSourceIdx(m, bn);
    if (vqmm_kernels[idx]) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const source = try verifyQmmSource(arena.allocator(), @intCast(m), @intCast(bn));
    const name: [*:0]const u8 = VQMM_NAMES[idx];
    const kernel = mlx.mlx_fast_metal_kernel_new(
        name,
        in_vec,
        out_vec,
        source,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    vqmm_kernels[idx] = kernel;
    return kernel;
}

/// Process-lifetime cache of 0-d int scalars for the kernel's K/N inputs —
/// only a handful of distinct layer geometries exist per model, and the
/// verify path builds ~300 kernel nodes per round (inference thread only,
/// same single-caller discipline as the kernel caches above).
const VqmmScalar = struct { v: c_int, arr: mlx.mlx_array, tick: u64 };
var vqmm_scalar_cache: [16]VqmmScalar = undefined;
var vqmm_scalar_count: usize = 0;
var vqmm_scalar_tick: u64 = 0;

/// Which slot may be evicted: the LEAST RECENTLY USED one, never a fixed slot.
///
/// The old policy always evicted slot 0, and that is a use-after-free. Every
/// caller takes TWO scalars back to back (`K_arr` then `N_arr`) and holds BOTH
/// in one `inputs_arr`. With the cache full, the K call evicts slot 0 and lands
/// there; the N call then frees slot 0 again — which is now K's array — so the
/// caller carried a dangling handle into `mlx_vector_array_new_data`. Observed
/// on a 40-layer GatedDeltaNet + MoE trunk with a 248k
/// vocab (enough distinct (K, N) pairs to overflow 16 slots) as SIGBUS building
/// the vector and SIGSEGV freeing it — one request killed the server, and the
/// surfaces probed after it all read as 0%.
///
/// The old comment claimed eviction was safe because "in-flight lazy graph
/// nodes hold their own refs". True, and irrelevant: the array has not reached
/// a graph node yet, it is still sitting in the caller's local array.
///
/// LRU is correct by construction here — the scalars a call site is still
/// holding are by definition the most recently ticked, so they can never be
/// the minimum while the cache (16) is far larger than the most any one site
/// takes (2).
fn vqmmScalarEvictIndex(entries: []const VqmmScalar) usize {
    var best: usize = 0;
    for (entries, 0..) |e, i| {
        if (e.tick < entries[best].tick) best = i;
    }
    return best;
}

fn cachedScalarInt(v: c_int) mlx.mlx_array {
    vqmm_scalar_tick += 1;
    for (vqmm_scalar_cache[0..vqmm_scalar_count]) |*e| {
        if (e.v == v) {
            e.tick = vqmm_scalar_tick;
            return e.arr;
        }
    }
    const arr = mlx.mlx_array_new_int(v);
    if (vqmm_scalar_count < vqmm_scalar_cache.len) {
        vqmm_scalar_cache[vqmm_scalar_count] = .{ .v = v, .arr = arr, .tick = vqmm_scalar_tick };
        vqmm_scalar_count += 1;
    } else {
        const victim = vqmmScalarEvictIndex(vqmm_scalar_cache[0..vqmm_scalar_count]);
        _ = mlx.mlx_array_free(vqmm_scalar_cache[victim].arr);
        vqmm_scalar_cache[victim] = .{ .v = v, .arr = arr, .tick = vqmm_scalar_tick };
    }
    return arr;
}

test "vqmm scalar cache never evicts a scalar the caller is still holding" {
    // The live crash, as a pure decision: every verifyQmm-family call site
    // takes K then N back to back and holds BOTH handles in one `inputs_arr`.
    // The old policy evicted slot 0 unconditionally, so the N call freed the
    // array the K call had just been handed — a dangling entry that killed the
    // server inside mlx_vector_array_new_data / _free.
    var cache: [16]VqmmScalar = undefined;
    for (&cache, 0..) |*e, i| {
        e.* = .{ .v = @intCast(i), .arr = .{ .ctx = null }, .tick = @intCast(i + 1) };
    }

    // Call 1 (K): the cache is full, so something is evicted and K takes its
    // slot with the newest tick.
    const k_slot = vqmmScalarEvictIndex(&cache);
    try std.testing.expectEqual(@as(usize, 0), k_slot); // oldest tick == slot 0 here
    cache[k_slot] = .{ .v = 1000, .arr = .{ .ctx = null }, .tick = 17 };

    // Call 2 (N): the victim must NOT be the slot K just took — the caller is
    // still holding that handle. Under the old always-slot-0 policy this
    // returned 0 and freed K.
    const n_slot = vqmmScalarEvictIndex(&cache);
    try std.testing.expect(n_slot != k_slot);
    cache[n_slot] = .{ .v = 2000, .arr = .{ .ctx = null }, .tick = 18 };

    // And it generalizes: after any run of insertions, the two most recently
    // ticked entries are never the next victim.
    var tick: u64 = 19;
    var round: usize = 0;
    while (round < 40) : (round += 1) {
        const victim = vqmmScalarEvictIndex(&cache);
        var newest: u64 = 0;
        var second: u64 = 0;
        for (cache) |e| {
            if (e.tick > newest) {
                second = newest;
                newest = e.tick;
            } else if (e.tick > second) second = e.tick;
        }
        try std.testing.expect(cache[victim].tick != newest);
        try std.testing.expect(cache[victim].tick != second);
        cache[victim] = .{ .v = @intCast(3000 + round), .arr = .{ .ctx = null }, .tick = tick };
        tick += 1;
    }
}

pub var verify_qmm_override: ?bool = null;
pub fn verifyQmmEnabled() bool {
    return verify_qmm_override orelse true;
}

pub const VqmmLane = enum { none, splitk, msg, nax };

var vqmm_nax_engaged = false;

/// The byte-unpack NAX tile is profitable for wide q5/q6/q8 projections, but
/// loses to stock MLX on the 1024-wide K/V projections. This threshold is the
/// measured split on M5 Max at M=8; q4 keeps its existing full geometry.
/// Qwen3.8-27B 8-bit/g64, forced depth 8, six paired settled boots: 89.81
/// tok/s NAX vs 79.51 stock (+12.96%); every arm asserted its own engagement.
fn mixedNaxShapeEnabled(bits: u32, group_size: u32, n: c_int) bool {
    if (bits == 4) return true;
    return (bits == 5 or bits == 6 or bits == 8) and group_size == 64 and n >= 5120;
}

/// Whether the PLAIN-SIMD tiles (split-K / msg) adopt a non-4-bit affine
/// weight. Profitability is a SEPARATE question from "the kernel is correct":
/// the byte-addressed unpack does strictly more ALU per value than the 4-bit
/// nibble path, so the shapes where it beats stock differ from the NAX tile's
/// (`mixedNaxShapeEnabled` exists because that tile LOSES on the 1024-wide K/V
/// projections).
///
/// MEASURED, Qwen3.8-27B on an M4 Max (no NAX lane, so the plain-SIMD tiles are
/// all this machine has), MTP on, decode tok/s from the server's own timings,
/// per-prompt medians over two passes with the CELL order reversed on the
/// second (the first cell of a pass reads ~2% fast, which a control pack that
/// ignores the lever entirely made visible):
///
/// | pack | code | prose | agentish |
/// |---|---|---|---|
/// | 6-bit gs64 | **1.10x** | **1.22x** | **1.17x** |
/// | 8-bit gs64 | 1.03x | 1.04x | 1.01x |
///
/// For scale: killing the 4-bit lane outright on the same box costs
/// 1.09-1.12x, so the 6-bit lane is worth MORE than the 4-bit one it was
/// modelled on — stock's 6-bit qmm has more to give back at these widths.
///
/// So 6-bit is default-on and 8-bit is NOT: 0.7-3.7% is inside the noise this
/// harness can resolve, and a lane that only adds dispatch surface for that is
/// not worth defaulting. 5-bit has the kernel arm and the parity test but no
/// pack on this box to measure, so it stays off for the same reason rather
/// than riding 6-bit's number.
///
/// The N floor is where the whole result lives: forcing every shape on
/// (which adopts the 1024-wide K/V projections too) measured
/// only 1.01-1.05x on the same pack. Excluding them is worth ~10 points.
pub fn mixedPlainShapeEnabled(bits: u32, group_size: u32, n: c_int) bool {
    if (bits == 4) return true;
    if (bits != 6) return false;
    return group_size == 64 and n >= MIXED_PLAIN_MIN_N;
}

/// The N below which the byte-addressed unpack loses to stock — the same split
/// the NAX tile measured, and the reason force-all-shapes gives back most of
/// the win.
pub const MIXED_PLAIN_MIN_N: c_int = 5120;

var vqmm_mixed_plain_engaged: bool = false;

/// Test seam: the parity test drives the plain-SIMD mixed-width arms without
/// touching the process environment (correctness is not a perf question).
pub var vqmm_mixed_plain_test_force: ?bool = null;

fn mixedPlainEnabled(bits: u32, group_size: u32, n: c_int) bool {
    if (vqmm_mixed_plain_test_force) |v| return v;
    return mixedPlainShapeEnabled(bits, group_size, n);
}

/// Pure lane selection for the verify-qmm family (hermetically pinned by
/// the NAX dispatch-table test). Shared floors: M 2..16, N >= 512 (tiny-N
/// projections — GDN ba proj, MoE routers — gain nothing over stock qmv
/// while the per-call kernel-node build ~10us is real; ~50 such calls ride
/// every verify round). Lanes:
/// - nax: M >= nax_min_m (default 8, see `naxMinMForBits`) when
///   the M5-class probe is live AND the m16 tile's stricter geometry holds
///   (K % 256 == 0, N % 32 == 0) — the lm_head N=151936 qualifies natively,
///   so no msg variant is needed past M=7.
/// - splitk/msg: the plain-SIMD M 2..7 lanes (K % 64 == 0, N % 4 == 0),
///   byte-identical to the pre-NAX dispatch; msg takes N >= 100000.
/// - none: stock mlx_quantized_matmul.
pub fn vqmmLaneFor(m: c_int, K: c_int, N: c_int, nax_on: bool, nax_min_m: c_int) VqmmLane {
    if (m < 2 or m > 16) return .none;
    if (N < 512) return .none;
    if (nax_on and m >= nax_min_m and @mod(K, 256) == 0 and @mod(N, 32) == 0) return .nax;
    if (m > 7) return .none;
    if (@mod(K, 64) != 0 or @mod(N, 4) != 0) return .none;
    return if (N >= 100000) .msg else .splitk;
}

/// Verify-width qmm dispatch (three lanes; see vqmmLaneFor). Returns the
/// [.., M, N] product when the shape is eligible — split-K (M 2..7),
/// msg wide tile (M 2..7 at huge N), or the NAX m16 tile (M 8..16,
/// M5-class machines only) — null to fall through to stock. 4-bit affine,
/// gs in {32,64,128}, bf16/fp16 activations.
pub fn verifyQmm(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
) !?mlx.mlx_array {
    if (!verifyQmmEnabled()) return null;
    // The plain-SIMD split-K/msg kernels carry byte-addressed 5/6/8-bit unpack,
    // but mixed-width adoption remains restricted to measured width/shape
    // combinations. The M5 NAX tile additionally handles eligible mixed-bit
    // projections at and above its takeover row.
    if (bits != 4 and bits != 5 and bits != 6 and bits != 8) return null;
    if (group_size != 32 and group_size != 64 and group_size != 128) return null;
    if (sc.ctx == null or bi.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len < 2) return null;
    const K: c_int = xsh[xsh.len - 1];
    var m: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| m *= d;
    const wsh = mlx.getShape(w);
    if (wsh.len != 2) return null;
    const N: c_int = wsh[0];
    // Affine packed geometry sanity: one uint32 column carries 32/bits source
    // values (5/6-bit rows are byte-packed but still exposed as uint32 arrays).
    if (@as(i64, wsh[1]) * 32 != @as(i64, K) * @as(i64, bits)) return null;
    const nax_on = verifyQmmNaxAvailable();
    const lane = vqmmLaneFor(m, K, N, nax_on, naxMinMForBits(bits));
    if (bits != 4) switch (lane) {
        .nax => if (!mixedNaxShapeEnabled(bits, group_size, N)) return null,
        // The plain-SIMD tiles now carry the same byte-addressed unpack the
        // NAX tile validated, so 5/6/8-bit no longer fall through to stock on
        // every M1-M4 machine — but adoption is per-shape and measured.
        .splitk, .msg => {
            if (!mixedPlainEnabled(bits, group_size, N)) return null;
            // A lane that silently declines is indistinguishable from one that
            // is merely slow, so every A/B arm has to be readable from its own
            // log rather than from the env it was launched with.
            if (!vqmm_mixed_plain_engaged) {
                vqmm_mixed_plain_engaged = true;
                log.info("[vqmm] plain-SIMD verify lane engaged at {d}-bit: lane={s} M={d} K={d} N={d} gs={d}\n", .{ bits, @tagName(lane), m, K, N, group_size });
            }
        },
        .none => return null,
    };
    switch (lane) {
        .none => return null,
        // NAX m16 tile (M5-class): M 8..16 by default — past the plain-SIMD
        // register cliff. Construction stays strictly behind the probe.
        .nax => return try runVerifyQmmNax(s, x, w, sc, bi, bits, group_size, m, K, N, xd, xsh),
        // Huge-N (lm_head class): the tiny-tile split-K grid thrashes the
        // scheduler there (measured 2.1x stock at M=4) — route through the
        // wide multi-simdgroup tile instead (2-column tile at M=7).
        .msg => return try runVerifyQmmMsg(s, x, w, sc, bi, bits, group_size, m, K, N, xd, xsh),
        .splitk => {},
    }

    // x rides in with its original shape — the kernel indexes the buffer
    // linearly, and a contiguous [.., M, K] has the same layout as [M, K]
    // (ensure_row_contiguous copies the rare non-contiguous case).
    const k_parts: c_int = vqmmKParts(N);
    const bn = vqmmBnFor(m, N);
    if (bn == 0) return null;
    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const y_shape = [_]c_int{ m, N };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32 * k_parts, @divExact(N, bn), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32 * k_parts, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K_PARTS", k_parts));

    const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getVerifyQmmKernel(m, bn);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outputs_vec, 0));

    // Restore the caller's leading shape with N as the last dim.
    var out_shape_buf: [8]c_int = undefined;
    const ndim = xsh.len;
    if (ndim > out_shape_buf.len) return error.UnsupportedShape;
    for (xsh[0 .. ndim - 1], 0..) |d, i| out_shape_buf[i] = d;
    out_shape_buf[ndim - 1] = N;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, out_shape_buf[0..ndim].ptr, @intCast(ndim), s));
    return y;
}

const VQMM_MSG_NSG: c_int = 8; // simdgroups per threadgroup (their sweep winner)

fn runVerifyQmmMsg(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    m: c_int,
    K: c_int,
    N: c_int,
    xd: mlx.mlx_dtype,
    xsh: []const c_int,
) !?mlx.mlx_array {
    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const y_shape = [_]c_int{ m, N };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
    const cols: c_int = vqmmMsgBn(m) * VQMM_MSG_NSG;
    const tg_count: c_int = @divTrunc(N + cols - 1, cols); // in-kernel n0 guard covers the tail
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32 * VQMM_MSG_NSG, tg_count, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32 * VQMM_MSG_NSG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSG", VQMM_MSG_NSG));

    const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getVerifyQmmMsgKernel(m);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outputs_vec, 0));

    var out_shape_buf: [8]c_int = undefined;
    const ndim = xsh.len;
    if (ndim > out_shape_buf.len) return error.UnsupportedShape;
    for (xsh[0 .. ndim - 1], 0..) |d, i| out_shape_buf[i] = d;
    out_shape_buf[ndim - 1] = N;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, out_shape_buf[0..ndim].ptr, @intCast(ndim), s));
    return y;
}

// ── NAX m16 verify tile (M5-class matrix units; see the section comment
// above for provenance, gating, and the never-build-off-probe rule) ──

/// Case-insensitive prefix match on the M5-class GPU family identifier.
/// Prefix (not equality) is MTPLX's shipping behavior — device variants
/// report suffixed forms ("applegpu_g17s", "applegpu_g17d").
pub fn naxArchGeneration(arch: []const u8) struct { gen: u32, phone: bool } {
    if (arch.len < 3) return .{ .gen = 0, .phone = false };
    var buf: [128]u8 = undefined;
    const n = @min(arch.len, buf.len);
    for (arch[0..n], 0..) |c, i| buf[i] = std.ascii.toLower(c);
    const a = buf[0..n];
    // Generic compiler targets such as air64_v27 do not name a GPU generation.
    const suffix = if (std.mem.startsWith(u8, a, "applegpu_g")) a[10..] else if (a[0] == 'g') a[1..] else return .{ .gen = 0, .phone = false };
    var digits: usize = 0;
    while (digits < suffix.len and std.ascii.isDigit(suffix[digits])) digits += 1;
    if (digits == 0 or suffix.len - digits > 1) return .{ .gen = 0, .phone = false };
    if (digits < suffix.len and !std.ascii.isAlphabetic(suffix[digits])) return .{ .gen = 0, .phone = false };
    return .{
        .gen = std.fmt.parseInt(u32, suffix[0..digits], 10) catch 0,
        .phone = digits < suffix.len and suffix[digits] == 'p',
    };
}

pub fn naxArchSupportedFrom(arch: []const u8) bool {
    const parsed = naxArchGeneration(arch);
    const floor: u32 = if (parsed.phone) 18 else 17;
    return parsed.gen >= floor;
}

pub fn naxArchIsG17(arch: []const u8) bool {
    return naxArchSupportedFrom(arch);
}

/// "26.4"/"26.4.1"-style product version at least req_major.req_minor.
/// Unparseable components read as 0 (mirrors MTPLX: int() failures fall
/// back to 0, so garbage can never satisfy the floor).
pub fn macosVersionAtLeast(ver: []const u8, req_major: u32, req_minor: u32) bool {
    var it = std.mem.splitScalar(u8, ver, '.');
    const major = std.fmt.parseInt(u32, it.first(), 10) catch 0;
    const minor: u32 = if (it.next()) |mn| (std.fmt.parseInt(u32, mn, 10) catch 0) else 0;
    return major > req_major or (major == req_major and minor >= req_minor);
}

/// The whole availability gate, pure over its inputs (mirror of MTPLX's
/// nax_available()): not force-fallback, G17-class GPU, macOS >= 26.2 (the
/// MetalPerformancePrimitives floor).
pub fn naxAvailableFrom(force_fallback: bool, arch: []const u8, os_ver: []const u8) bool {
    if (force_fallback) return false;
    if (!naxArchIsG17(arch)) return false;
    return macosVersionAtLeast(os_ver, 26, 2);
}

/// Human-readable NAX status for `--version` / the app's Settings ("nax"
/// line, first token on/off, remainder the reason). Hardware + OS is the
/// whole story for OUR binaries: the bundled MLX always ships the NAX
/// kernels (build-mlx.sh + tests/test_mlx_staged_nax.sh assert them), so
/// this mirrors MLX's stock-op is_nax_available() gate (GPU gen >= 17,
/// macOS >= 26.2 — that symbol is not exported, hence the mirror).
/// Deliberately ignores the verifyQmm-lane QA env switches: those scope
/// our custom kernel lane, not MLX's stock dispatch.
pub fn naxStatusFrom(arch: []const u8, os_ver: []const u8) []const u8 {
    if (!naxArchIsG17(arch)) return "off (requires M5-class GPU)";
    if (!macosVersionAtLeast(os_ver, 26, 2)) return "off (requires macOS 26.2+)";
    return "on (M5 neural accelerators)";
}

/// naxStatusFrom over the real device: mlx device-info arch + sysctl OS
/// version (same sources as the verifyQmm probe).
pub fn naxStatus() []const u8 {
    var arch_buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&arch_buf) orelse "";
    var ver_buf: [64]u8 = undefined;
    const ver = macosProductVersion(&ver_buf) orelse "";
    return naxStatusFrom(arch, ver);
}

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// "kern.osproductversion" → "26.4"-style string (the sysctl mirror of
/// Python's platform.mac_ver()[0]).
pub fn macosProductVersion(buf: []u8) ?[]const u8 {
    var len: usize = buf.len;
    if (sysctlbyname("kern.osproductversion", buf.ptr, &len, null, 0) != 0) return null;
    var n = @min(len, buf.len);
    while (n > 0 and buf[n - 1] == 0) n -= 1;
    if (n == 0) return null;
    return buf[0..n];
}

/// GPU architecture identifier off mlx device info ("applegpu_g16" on the
/// M4 Max). The returned pointer from mlx is borrowed from the info object,
/// so the string is copied into the caller's buffer before the info frees.
fn gpuArchitecture(buf: []u8) ?[]const u8 {
    var dev = mlx.mlx_device{ .ctx = null };
    if (mlx.mlx_get_default_device(&dev) != 0) return null;
    defer _ = mlx.mlx_device_free(dev);
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) != 0) return null;
    var cstr: [*:0]const u8 = undefined;
    if (mlx.mlx_device_info_get_string(&cstr, info, "architecture") != 0) return null;
    const arch = std.mem.span(cstr);
    if (arch.len == 0 or arch.len > buf.len) return null;
    @memcpy(buf[0..arch.len], arch);
    return buf[0..arch.len];
}

/// Test seam: force the NAX availability probe (null = real probe). Only
/// ever force FALSE on non-G17 machines — forcing true would let dispatch
/// build a kernel whose matmul2d pipeline cannot be created off M5-class
/// hardware.
pub var vqmm_nax_probe_override: ?bool = null;

var vqmm_nax_avail_cache: ?bool = null;

/// M5-class NAX units present: "applegpu_g17" GPU + macOS >= 26.2.
/// SUSHI_FORCE_GPU_FAMILY_FALLBACK=1 pretends the units are absent so
/// an M5 can rehearse the exact M1-M4 plain-SIMD path (QA switch, mirrored
/// from MTPLX). Cached for the process lifetime (inference-thread caller
/// discipline, same as the kernel caches).
pub fn verifyQmmNaxAvailable() bool {
    if (vqmm_nax_probe_override) |v| return v;
    if (vqmm_nax_avail_cache) |v| return v;
    const ok = computeNaxAvailable();
    vqmm_nax_avail_cache = ok;
    return ok;
}

fn computeNaxAvailable() bool {
    var force = false;
    if (std.c.getenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK")) |p| {
        const v = std.mem.span(p);
        force = v.len > 0 and v[0] == '1';
    }
    var arch_buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&arch_buf) orelse "";
    var ver_buf: [64]u8 = undefined;
    const ver = macosProductVersion(&ver_buf) orelse "";
    return naxAvailableFrom(force, arch, ver);
}

var nax_sdpa_avail_cache: ?bool = null;
pub var nax_sdpa_override: ?bool = null; // test seam

/// mlx >= 0.32.2 serves hd-256 attention on NAX with a fused full-attention
/// kernel (`sdpa_full_self_attention_nax`), auto only at >= 1024 causal rows
/// and reachable below that through `force_fused`. Preferred wherever NAX is
/// available; elsewhere sushi_attn_p256 and the split/vector path serve.
pub fn naxSdpaPreferred() bool {
    if (nax_sdpa_override) |v| return v;
    if (nax_sdpa_avail_cache) |v| return v;
    const on = computeNaxAvailable();
    nax_sdpa_avail_cache = on;
    if (on) log.info("[nax-sdpa] engaged: stock fused sdpa (force_fused) serves hd-256 causal prefill and verify blocks > 8 rows; sushi_attn_p256 causal arm declined\n", .{});
    return on;
}

/// Family-wide NAX readiness: the hardware probe passes. This intentionally says nothing about a specific
/// M/K/N dispatch.
pub fn verifyQmmNaxEnabled() bool {
    return verifyQmmEnabled() and verifyQmmNaxAvailable();
}

/// M width where the NAX tile takes over from the plain-SIMD lanes.
pub fn naxMinMForBits(bits: u32) c_int {
    // The byte-per-value q8 NAX tile wins already at M=7 (+6.09% over the
    // plain lane, five thermally-settled counterbalanced pairs). Packed q4/q6
    // still lose there, so their measured takeover remains M=8.
    return if (bits == 8) 7 else 8;
}

const VQMM_NAX_HEADER: [:0]const u8 =
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\
;

/// MTPLX's _build_kernel_m16_nax_ktmpl body VERBATIM (nax_verify.py,
/// Apache-2.0; single-brace form of their f-string). The tensor
/// extents/strides encode dflash's working matmul2d convention — treat
/// them as opaque and correct, do not re-derive. Geometry: threadgroup =
/// 256 threads = 8 simdgroups; each simdgroup owns a K/8 chunk and stages
/// a dequantized 16x32 B tile in threadgroup memory (each of its 32 lanes
/// dequants ONE output column's 16 K-values from MLX's packed byte stream per
/// iteration); matmul2d (16x32x16 multiply_accumulate, execution_simdgroup)
/// multiplies the row-padded activation tile against it into a cooperative
/// fp32 accumulator; after the K loop the 8 partial C tiles reduce across
/// the threadgroup. T/GS/KCONST arrive as template args (MLX name-mangles
/// template values into the host_name, so one kernel object serves every
/// specialization — the GDN Dk/Dv precedent).
const VQMM_NAX_SOURCE: [:0]const u8 =
    \\using namespace metal;
    \\using namespace mpp::tensor_ops;
    \\
    \\constexpr int BM = 16;
    \\constexpr int BN = 32;
    \\constexpr int BK = 16;
    \\constexpr int NSG = 8;
    \\constexpr int K = KCONST;
    \\constexpr int K_bytes = K * BITS / 8;
    \\constexpr int K_by_gs = K / GS;
    \\constexpr int K_chunk = K / NSG;
    \\
    \\uint tid = thread_position_in_threadgroup.x;
    \\uint sg_id = simdgroup_index_in_threadgroup;
    \\uint lane = thread_index_in_simdgroup;
    \\uint tg_n = threadgroup_position_in_grid.y;
    \\int N = int(N_size);
    \\int n0 = int(tg_n) * BN;
    \\int k_begin = int(sg_id) * K_chunk;
    \\int k_end = k_begin + K_chunk;
    \\
    \\threadgroup T B_tile[NSG][BK * BN];
    \\threadgroup float partial[NSG][BM * BN];
    \\
    \\constexpr auto desc = matmul2d_descriptor(
    \\    16,
    \\    32,
    \\    16,
    \\    false,
    \\    false,
    \\    false,
    \\    matmul2d_descriptor::mode::multiply_accumulate);
    \\matmul2d<desc, metal::execution_simdgroup> op;
    \\
    \\tensor<device T, dextents<int, 2>, tensor_inline> A(
    \\    (device T*)x,
    \\    dextents<int, 2>{K, BM},
    \\    array<int, 2>{1, K});
    \\tensor<threadgroup T, dextents<int, 2>, tensor_inline> B(
    \\    B_tile[sg_id],
    \\    dextents<int, 2>{BN, BK},
    \\    array<int, 2>{1, BN});
    \\tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(
    \\    partial[sg_id],
    \\    dextents<int, 2>{BN, BM},
    \\    array<int, 2>{1, BN});
    \\
    \\auto ct_c = op.template get_destination_cooperative_tensor<
    \\    tensor<device T, extents<int, 16, 16>, tensor_inline>,
    \\    tensor<threadgroup T, extents<int, 32, 16>, tensor_inline>,
    \\    float>();
    \\_Pragma("unroll")
    \\for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {
    \\    ct_c[i] = 0.0f;
    \\}
    \\
    \\int n_global = n0 + int(lane);
    \\for (int k0 = k_begin; k0 < k_end; k0 += BK) {
    \\    const device uchar* wp =
    \\        ((const device uchar*)w_q) + n_global * K_bytes + (k0 * BITS) / 8;
    \\    float scale = float(scales[n_global * K_by_gs + (k0 / GS)]);
    \\    float bias = float(biases[n_global * K_by_gs + (k0 / GS)]);
    \\
    \\    if constexpr (BITS == 4) {
    \\        _Pragma("unroll")
    \\        for (int pack = 0; pack < 2; ++pack) {
    \\            uint32_t p =
    \\                uint32_t(wp[pack * 4 + 0]) |
    \\                (uint32_t(wp[pack * 4 + 1]) << 8) |
    \\                (uint32_t(wp[pack * 4 + 2]) << 16) |
    \\                (uint32_t(wp[pack * 4 + 3]) << 24);
    \\            _Pragma("unroll")
    \\            for (int ki = 0; ki < 8; ++ki) {
    \\                uint32_t q = (p >> (ki * 4)) & 0xFu;
    \\                B_tile[sg_id][(pack * 8 + ki) * BN + int(lane)] =
    \\                    T(float(q) * scale + bias);
    \\            }
    \\        }
    \\    } else if constexpr (BITS == 5) {
    \\        _Pragma("unroll")
    \\        for (int pack = 0; pack < 2; ++pack) {
    \\            ulong p =
    \\                ulong(wp[pack * 5 + 0]) |
    \\                (ulong(wp[pack * 5 + 1]) << 8) |
    \\                (ulong(wp[pack * 5 + 2]) << 16) |
    \\                (ulong(wp[pack * 5 + 3]) << 24) |
    \\                (ulong(wp[pack * 5 + 4]) << 32);
    \\            _Pragma("unroll")
    \\            for (int ki = 0; ki < 8; ++ki) {
    \\                uint32_t q = uint32_t((p >> (ki * 5)) & 0x1Ful);
    \\                B_tile[sg_id][(pack * 8 + ki) * BN + int(lane)] =
    \\                    T(float(q) * scale + bias);
    \\            }
    \\        }
    \\    } else if constexpr (BITS == 6) {
    \\        _Pragma("unroll")
    \\        for (int pack = 0; pack < 4; ++pack) {
    \\            uint32_t p =
    \\                uint32_t(wp[pack * 3 + 0]) |
    \\                (uint32_t(wp[pack * 3 + 1]) << 8) |
    \\                (uint32_t(wp[pack * 3 + 2]) << 16);
    \\            _Pragma("unroll")
    \\            for (int ki = 0; ki < 4; ++ki) {
    \\                uint32_t q = (p >> (ki * 6)) & 0x3Fu;
    \\                B_tile[sg_id][(pack * 4 + ki) * BN + int(lane)] =
    \\                    T(float(q) * scale + bias);
    \\            }
    \\        }
    \\    } else {
    \\        static_assert(BITS == 8, "unsupported NAX affine width");
    \\        _Pragma("unroll")
    \\        for (int ki = 0; ki < BK; ++ki) {
    \\            uint32_t q = uint32_t(wp[ki]);
    \\            B_tile[sg_id][ki * BN + int(lane)] =
    \\                T(float(q) * scale + bias);
    \\        }
    \\    }
    \\    simdgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    auto tA = A.template slice<16, 16>(k0, 0);
    \\    auto tB = B.template slice<32, 16>(0, 0);
    \\    op.run(tA, tB, ct_c);
    \\    simdgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\
    \\auto tC = C.template slice<32, 16>(0, 0);
    \\ct_c.store(tC);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\for (int off = int(tid); off < BM * BN; off += NSG * 32) {
    \\    float acc01 = partial[0][off] + partial[1][off];
    \\    float acc23 = partial[2][off] + partial[3][off];
    \\    float acc45 = partial[4][off] + partial[5][off];
    \\    float acc67 = partial[6][off] + partial[7][off];
    \\    float acc = (acc01 + acc23) + (acc45 + acc67);
    \\    int row = off / BN;
    \\    int col = off - row * BN;
    \\    y[row * N + n0 + col] = T(acc);
    \\}
;

var vqmm_nax_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// ONLY call where verifyQmmNaxAvailable() — see the never-build rule.
fn getVerifyQmmNaxKernel() !mlx.mlx_fast_metal_kernel {
    if (vqmm_nax_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_vqmm_nax_m16",
        in_vec,
        out_vec,
        VQMM_NAX_SOURCE,
        VQMM_NAX_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    vqmm_nax_kernel = kernel;
    return kernel;
}

/// The M-padding half of the NAX host contract, split out so it can be
/// exercised hermetically on non-M5 machines (the scaffolding test wraps
/// it around STOCK qmm — zero pad rows must produce zero output rows):
/// collapse the caller's leading shape to [m, K] (mirrors MTPLX's
/// x.reshape(m, k) so row padding is well-defined for every batch shape),
/// then zero-pad the row axis to the tile's fixed 16. Owned handle.
fn naxPadTo16(s: mlx.mlx_stream, x: mlx.mlx_array, m: c_int, K: c_int, xd: mlx.mlx_dtype) !mlx.mlx_array {
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    const x2_shape = [_]c_int{ m, K };
    try mlx.check(mlx.mlx_reshape(&x2, x, &x2_shape, 2, s));
    var x16 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(x16);
    if (m < 16) {
        var zero = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zero);
        const zdims = [_]c_int{};
        try mlx.check(mlx.mlx_zeros(&zero, &zdims, 0, xd, s));
        const pad_axes = [_]c_int{0};
        const pad_low = [_]c_int{0};
        const pad_high = [_]c_int{16 - m};
        try mlx.check(mlx.mlx_pad(&x16, x2, &pad_axes, 1, &pad_low, 1, &pad_high, 1, zero, "constant", s));
    } else {
        try mlx.check(mlx.mlx_array_set(&x16, x2));
    }
    return x16;
}

/// The slice-back half: first m rows of the [16, N] tile output (pad rows
/// dropped). Owned handle.
fn naxSliceRows(s: mlx.mlx_stream, y16: mlx.mlx_array, m: c_int, N: c_int) !mlx.mlx_array {
    var ym = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ym);
    if (m < 16) {
        const start = [_]c_int{ 0, 0 };
        const stop = [_]c_int{ m, N };
        const strides = [_]c_int{ 1, 1 };
        try mlx.check(mlx.mlx_slice(&ym, y16, &start, 2, &stop, 2, &strides, 2, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&ym, y16));
    }
    return ym;
}

/// Run the NAX m16 tile: pad the activations to the fixed [16, K] tile
/// ("weight streaming dominates so padded rows are nearly free" — the
/// source ledger), grid (256, N/32), slice y back to the caller's M rows.
fn runVerifyQmmNax(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    m: c_int,
    K: c_int,
    N: c_int,
    xd: mlx.mlx_dtype,
    xsh: []const c_int,
) !?mlx.mlx_array {
    if (!vqmm_nax_engaged) {
        vqmm_nax_engaged = true;
        log.info("[vqmm] NAX verify lane engaged at {d}-bit: M={d} K={d} N={d} gs={d} min_m={d} (SUSHI_VERIFY_QMM=0 restores stock)\n", .{ bits, m, K, N, group_size, naxMinMForBits(bits) });
    }
    const x16 = try naxPadTo16(s, x, m, K, xd);
    defer _ = mlx.mlx_array_free(x16);

    const N_arr = cachedScalarInt(N);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const y_shape = [_]c_int{ 16, N };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 256, @divExact(N, 32), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KCONST", K));

    const inputs_arr = [_]mlx.mlx_array{ x16, w, sc, bi, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getVerifyQmmNaxKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y16);
    try mlx.check(mlx.mlx_vector_array_get(&y16, outputs_vec, 0));

    // Drop the pad rows, restore the caller's leading shape with N last.
    const ym = try naxSliceRows(s, y16, m, N);
    defer _ = mlx.mlx_array_free(ym);

    var out_shape_buf: [8]c_int = undefined;
    const ndim = xsh.len;
    if (ndim > out_shape_buf.len) return error.UnsupportedShape;
    for (xsh[0 .. ndim - 1], 0..) |d, i| out_shape_buf[i] = d;
    out_shape_buf[ndim - 1] = N;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, ym, out_shape_buf[0..ndim].ptr, @intCast(ndim), s));
    return y;
}

// ── Fused prefill attention at the head dims MLX has no kernel for ──
//
// MLX's fused SDPA (steel_attention) ships bd 64/96/128 and a 256 dsplit and
// nothing else, so two widths fall back to the composed path that
// MATERIALIZES a [heads, chunk, total_kv] bf16 score tensor per layer: 256
// (a head_dim 256 trunk — 26.8 GB/layer at a 102K prompt,
// the long-context OOM class commit 7550895 budgeted around), and the
// asymmetric 192/128 that mimo_v2's global layers and MLA score at (32 GiB at
// a 512k prompt). `BDK`/`BDV` are the two widths; see `fusedPdDims`. This kernel started as a
// faithful self-contained port of MLX's steel attention (FA-2 online
// softmax, float32 accumulation, exp2 softmax) specialized to BD=256; the
// v2 tiles (BQ=64, BK=32, 8 simdgroups, register-resident Q, uint4 staging
// — see the geometry comment above ATTN_PD_KERNEL_SOURCE) are our own,
// picked by micro-bench sweep. An optional sliding-window band covers
// sliding-window layers ("array" mask during prefill) too, and an optional
// per-head sink column mimo_v2's sliding layers. The score tensor
// never exists — O(tile) working memory — so the three prefill OOM guards
// (generate.boundedPrefillChunk / prefillEvalCadence / server.prefillMemoryNeeded)
// drop their score term via `prefillHeadDimFused` when this kernel is active.
//
// Scope: seq_len > 1 (prefill), any batch, GQA (Hq % Hk == 0), bf16, causal
// bottom-right alignment (query row r sits at absolute KV position
// kL - qL + r, matching MLX "causal" and createSlidingWindowMask semantics).
// K/V may be non-contiguous cache views (strides used; innermost dim must be
// contiguous — guaranteed for slices along T). The OOM guards and the
// dispatch share one predicate, so they cannot drift.
const ATTN_PD_KERNEL_HEADER =
    \\#include <metal_simdgroup_matrix>
    \\
    \\// Fragment layout mirrors MLX steel BaseMMAFrag<float,8,8>: each thread
    \\// of a simdgroup holds 2 adjacent elements of an 8x8 tile; the hardware
    \\// mma runs on simdgroup_float8x8 built from those elements. The 4 threads
    \\// holding one row differ in lane bits 0 and 3 (see sushi_coord).
    \\inline short2 sushi_coord(ushort lane) {
    \\  const short qid = lane / 4;
    \\  const short fm = (qid & 4) + ((lane / 2) % 4);
    \\  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    \\  return short2(fn, fm);
    \\}
    \\
    \\inline void sushi_mma(thread float2 &d, float2 a, float2 b) {
    \\  metal::simdgroup_float8x8 D, A, B, C;
    \\  A.thread_elements()[0] = a.x;
    \\  A.thread_elements()[1] = a.y;
    \\  B.thread_elements()[0] = b.x;
    \\  B.thread_elements()[1] = b.y;
    \\  C.thread_elements()[0] = d.x;
    \\  C.thread_elements()[1] = d.y;
    \\  simdgroup_multiply_accumulate(D, A, B, C);
    \\  d.x = D.thread_elements()[0];
    \\  d.y = D.thread_elements()[1];
    \\}
    \\
    \\inline float sushi_row_max(float2 v) {
    \\  float t = metal::max(v.x, v.y);
    \\  t = metal::max(t, metal::simd_shuffle_xor(t, ushort(1)));
    \\  t = metal::max(t, metal::simd_shuffle_xor(t, ushort(8)));
    \\  return t;
    \\}
    \\
    \\inline float sushi_row_sum(float2 v) {
    \\  float t = v.x + v.y;
    \\  t += metal::simd_shuffle_xor(t, ushort(1));
    \\  t += metal::simd_shuffle_xor(t, ushort(8));
    \\  return t;
    \\}
    \\
;

// v2 tile geometry (2026-07-12 micro-bench sweep on the M4 Max, 24q/4kv
// geometry): BQ=64 with 8 simdgroups halves the K/V staging walks
// per query row vs the v1 32-row tile; BK=32 halves the per-column softmax
// rescale + barrier overhead; Q lives in REGISTERS loaded straight from
// global (no Q smem at all — that memory goes to the wider K tile); staging
// runs on uint4 (8 bf16 per instruction, was scalar). Measured vs v1 at
// (qL x kL): (2048x16384) 250 -> 73 ms, (2048x65536) 999 -> 325 ms,
// (8192x8192) 279 -> 77 ms. vs the composed path the v2 kernel wins where
// kL <= ~4*qL (+63% at ratio 1, +15% at 2, +1.8% at 4) and loses beyond
// (-2.3% at 6, -6% at 8) — the causal arm's ratio gate (fusedSdpaPrefill)
// encodes exactly that envelope. Rejected variants (measured worse): BK=8
// ping-pong (per-block rescale doubles), V-from-global fragments (device
// loads too slow), K-only double-buffer (barrier structure unchanged),
// Q-smem BQ=32 with 2 threadgroups/core (SLC thrash at long kL).
const ATTN_PD_KERNEL_SOURCE =
    \\constexpr int BQ = 64;
    \\constexpr int BK = 32;
    \\constexpr int LDK = BK + 8;
    \\constexpr int LDV = BDV + 8;
    \\constexpr int NT = 256;
    \\// uint4 staging strides: 8 halves per instruction along each width.
    \\constexpr int NWK = BDK / 8;
    \\constexpr int NWV = BDV / 8;
    \\
    \\const int qL = q_shape[2];
    \\const int kL = k_shape[2];
    \\const int Hq = q_shape[1];
    \\const int Hk = k_shape[1];
    \\const int gqa = Hq / Hk;
    \\
    \\const int tqx = int(threadgroup_position_in_grid.x);
    \\const int hq = int(threadgroup_position_in_grid.y);
    \\const int bb = int(threadgroup_position_in_grid.z);
    \\const ushort lane = ushort(thread_index_in_simdgroup);
    \\const ushort warp = ushort(simdgroup_index_in_threadgroup);
    \\const int tix = int(thread_index_in_threadgroup);
    \\
    \\// exp2-based softmax (steel convention): fold log2(e) into the scale.
    \\const float scale_log2e = scl[0] * 1.44269504088896340736f;
    \\const int SW = win[0];
    \\
    \\// Budgeted kv-chunk dispatch (causal arm): this dispatch covers keys
    \\// [k_begin, k_end); online-softmax state (m/l/unnormalized O) rides fp32
    \\// carry buffers between dispatches — exact register precision, so the
    \\// chunked chain is bit-identical to one long dispatch. phase bit 0 =
    \\// carry-in present, bit 1 = final chunk (normalize + bf16 store).
    \\const int k_begin = kr[0];
    \\const int k_end = metal::min(kr[1], kL);
    \\// The buffer is not always the cache: a quantized cache hands each
    \\// dispatch a DEQUANTIZED SLICE, so `koff` is where this buffer's row 0
    \\// sits in the cache and `kL_abs` is the cache's own length. Every causal
    \\// and band comparison is in cache coordinates; staging is in buffer
    \\// coordinates. A dense caller passes {0, kL} and nothing below moves.
    \\const int koff = kr[2];
    \\const int kL_abs = kr[3];
    \\const bool has_carry = (phase[0] & 1) != 0;
    \\const bool is_final = (phase[0] & 2) != 0;
    \\
    \\// Bottom-right aligned causal: query row r of this chunk sits at
    \\// absolute KV position q_off + r.
    \\const int q_off = kL_abs - qL;
    \\
    \\const device T* Qp = q + bb * q_strides[0] + hq * q_strides[1]
    \\    + (long)(tqx * BQ) * q_strides[2];
    \\const device T* Kp = k + bb * k_strides[0] + (hq / gqa) * k_strides[1];
    \\const device T* Vp = v + bb * v_strides[0] + (hq / gqa) * v_strides[1];
    \\device T* Op = out + (((long)bb * Hq + hq) * (long)qL + (long)(tqx * BQ)) * BDV;
    \\
    \\// K^T tile [BDK][LDK] (20.5 KB at qk 256) is strictly larger than the
    \\// V tile [BK][LDV] (16.9 KB at v 256) — share one buffer, steel-style
    \\// (barriers separate the K reads from the V staging).
    \\constexpr int SMEM = (LDK * BDK > BK * LDV) ? (LDK * BDK) : (BK * LDV);
    \\threadgroup T KVs[SMEM];
    \\threadgroup T* Ks = KVs;
    \\threadgroup T* Vs = KVs;
    \\
    \\const int q_rows = metal::min(BQ, qL - tqx * BQ);
    \\
    \\const short2 sc = sushi_coord(lane);
    \\const short sn = sc.x;
    \\const short sm = sc.y;
    \\const short tm = 8 * short(warp);
    \\const int Ks_off = sm * LDK + sn;
    \\const int Vs_off = sm * LDV + sn;
    \\
    \\// Q fragment straight from global into registers, once per threadgroup:
    \\// row (tm+sm), element pairs (dd*8+sn, +1). sn is always even, so the
    \\// vec2 load is 4B-aligned for any row whose stride is even (structural:
    \\// a q row strides BDK). Rows past the ragged tail read zero — their
    \\// outputs are discarded by the store guard.
    \\float2 Qfrag[NWK];
    \\{
    \\  const int qr = tm + sm;
    \\  if (qr < q_rows) {
    \\    const device T* Qrow = Qp + (long)qr * q_strides[2];
    \\    for (int dd = 0; dd < NWK; ++dd) {
    \\      const vec<T, 2> p = *((const device vec<T, 2>*)(Qrow + dd * 8 + sn));
    \\      Qfrag[dd] = float2(float(p.x), float(p.y));
    \\    }
    \\  } else {
    \\    for (int dd = 0; dd < NWK; ++dd) Qfrag[dd] = float2(0.0f);
    \\  }
    \\}
    \\
    \\float2 Ofrag[NWV];
    \\for (int i = 0; i < NWV; ++i) Ofrag[i] = float2(0.0f);
    \\// Init max FINITE (not -inf) so the rescale factor exp2(old-new) can
    \\// never be exp2(nan); masked scores use true -INFINITY so a row whose
    \\// first blocks are fully banded out contributes exp2(-inf)=0, not 1.
    \\float max_score = -3.0e38f;
    \\float sum_score = 0.0f;
    \\
    \\// Resume carried softmax state from the previous kv chunk. Rows past
    \\// the ragged q tail never carry (their outputs are discarded anyway).
    \\// (m_in/l_in/o_in are indexed directly, never through a typed pointer:
    \\// MLX binds sub-4KB inputs — the no-carry dummies — in the `constant`
    \\// address space and real carries in `device`, and the source must
    \\// compile under both.)
    \\if (has_carry) {
    \\  const int qr = tm + sm;
    \\  if (qr < q_rows) {
    \\    const long crow = (((long)bb * Hq + hq) * (long)qL + (long)(tqx * BQ + qr));
    \\    max_score = m_in[crow];
    \\    sum_score = l_in[crow];
    \\    const long obase = crow * (long)BDV;
    \\    for (int dd = 0; dd < NWV; ++dd) {
    \\      Ofrag[dd] = float2(o_in[obase + dd * 8 + sn], o_in[obase + dd * 8 + sn + 1]);
    \\    }
    \\  }
    \\}
    \\// Learned per-head sink (template SINK=1): one extra softmax column that
    \\// joins the max and the sum and carries no value row, so it is exactly a
    \\// running state of (max = sink, sum = 1, O = 0) before the first key.
    \\if (SINK && !has_carry) {
    \\  max_score = sinks[hq] * 1.44269504088896340736f;
    \\  sum_score = 1.0f;
    \\}
    \\
    \\const int NK = (k_end + BK - 1) / BK;
    \\const int q_lo = tqx * BQ + q_off;
    \\const int q_hi = q_lo + BQ - 1;
    \\const int kb_lim = metal::min(NK, (q_hi - koff + BK) / BK);
    \\int kb = k_begin / BK;
    \\if (SW > 0) kb = metal::max(kb, metal::max(0, q_lo - SW + 1 - koff) / BK);
    \\const int kb_min_causal = metal::max(0, q_lo - koff) / BK;
    \\const int row_pos = q_lo + tm + sm;
    \\
    \\// QSA arm (template QSA=1): `mask` is the [B,1,qL,kL] bool visibility
    \\// (causal folded in), `skip` the [B, nq_tiles, NK] per-tile any-visible
    \\// table — a key block no row of this tile sees is never staged.
    \\const long skip_base = ((long)bb * ((qL + BQ - 1) / BQ) + tqx) * (long)((kL + BK - 1) / BK);
    \\const long mask_row0 = QSA ? ((long)bb * mask_strides[0] + (long)(tqx * BQ + tm + sm) * mask_strides[2]) : 0;
    \\const bool mask_row_ok = (tqx * BQ + tm + sm) < qL;
    \\
    \\for (; kb < kb_lim; kb++) {
    \\  if (QSA && !skip[skip_base + kb]) continue;
    \\  const int c0 = kb * BK;
    \\  const int rows_k = metal::min(BK, kL - c0);
    \\
    \\  // Stage K transposed (Ks[d][kk]): uint4 global loads (8 bf16 each),
    \\  // scalar transposed scatter into smem. Consecutive lanes take consecutive
    \\  // KEY rows, so a store lands on adjacent addresses of one Ks row; lanes
    \\  // spread over head-dim chunks would stride 8*LDK halves, one bank.
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * NWK; i += NT) {
    \\    const int r = i % BK;
    \\    const int c8 = i / BK;
    \\    uint4 w = uint4(0);
    \\    if (r < rows_k) {
    \\      w = *((const device uint4*)(Kp + (long)(c0 + r) * k_strides[2]) + c8);
    \\    }
    \\    thread T* e = (thread T*)&w;
    \\    const int cb = c8 * 8;
    \\    Ks[(cb + 0) * LDK + r] = e[0];
    \\    Ks[(cb + 1) * LDK + r] = e[1];
    \\    Ks[(cb + 2) * LDK + r] = e[2];
    \\    Ks[(cb + 3) * LDK + r] = e[3];
    \\    Ks[(cb + 4) * LDK + r] = e[4];
    \\    Ks[(cb + 5) * LDK + r] = e[5];
    \\    Ks[(cb + 6) * LDK + r] = e[6];
    \\    Ks[(cb + 7) * LDK + r] = e[7];
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
    \\  // S = Q @ K^T for this simdgroup's 8 query rows.
    \\  float2 Sfrag[BK / 8];
    \\  for (int i = 0; i < BK / 8; ++i) Sfrag[i] = float2(0.0f);
    \\  for (int dd = 0; dd < NWK; ++dd) {
    \\    const float2 qf = Qfrag[dd];
    \\    const int kbase = Ks_off + dd * 8 * LDK;
    \\    const float2 kf0 = float2(float(Ks[kbase]), float(Ks[kbase + 1]));
    \\    const float2 kf1 = float2(float(Ks[kbase + 8]), float(Ks[kbase + 9]));
    \\    const float2 kf2 = float2(float(Ks[kbase + 16]), float(Ks[kbase + 17]));
    \\    const float2 kf3 = float2(float(Ks[kbase + 24]), float(Ks[kbase + 25]));
    \\    sushi_mma(Sfrag[0], qf, kf0);
    \\    sushi_mma(Sfrag[1], qf, kf1);
    \\    sushi_mma(Sfrag[2], qf, kf2);
    \\    sushi_mma(Sfrag[3], qf, kf3);
    \\  }
    \\  Sfrag[0] *= scale_log2e;
    \\  Sfrag[1] *= scale_log2e;
    \\  Sfrag[2] *= scale_log2e;
    \\  Sfrag[3] *= scale_log2e;
    \\
    \\  // Masking: kL remainder + causal + sliding band, all element-wise.
    \\  const bool tail_k = (rows_k < BK);
    \\  const bool need_causal = (kb >= kb_min_causal);
    \\  const bool need_band = (SW > 0) && (koff + c0 <= q_hi - SW);
    \\  if (QSA || tail_k || need_causal || need_band) {
    \\    for (int kt = 0; kt < BK / 8; ++kt) {
    \\      for (int jj = 0; jj < 2; ++jj) {
    \\        const int col = c0 + kt * 8 + sn + jj;
    \\        const int col_abs = koff + col;
    \\        bool masked = false;
    \\        if (tail_k && col >= kL) masked = true;
    \\        if (need_causal && row_pos < col_abs) masked = true;
    \\        if (need_band && (row_pos - col_abs) >= SW) masked = true;
    \\        if (QSA && !masked && (!mask_row_ok || !mask[mask_row0 + (long)col_abs * mask_strides[3]])) masked = true;
    \\        if (masked) Sfrag[kt][jj] = -INFINITY;
    \\      }
    \\    }
    \\  }
    \\
    \\  // Stage V (same smem as K — K reads are done): uint4 on both sides.
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * NWV; i += NT) {
    \\    const int r = i / NWV;
    \\    const int c8 = i % NWV;
    \\    uint4 w = uint4(0);
    \\    if (r < rows_k) {
    \\      w = *((const device uint4*)(Vp + (long)(c0 + r) * v_strides[2]) + c8);
    \\    }
    \\    *((threadgroup uint4*)(Vs + r * LDV) + c8) = w;
    \\  }
    \\
    \\  // Online softmax (registers only, overlaps the V staging above).
    \\  float new_max = max_score;
    \\  new_max = metal::max(new_max, sushi_row_max(Sfrag[0]));
    \\  new_max = metal::max(new_max, sushi_row_max(Sfrag[1]));
    \\  new_max = metal::max(new_max, sushi_row_max(Sfrag[2]));
    \\  new_max = metal::max(new_max, sushi_row_max(Sfrag[3]));
    \\  Sfrag[0] = metal::exp2(Sfrag[0] - new_max);
    \\  Sfrag[1] = metal::exp2(Sfrag[1] - new_max);
    \\  Sfrag[2] = metal::exp2(Sfrag[2] - new_max);
    \\  Sfrag[3] = metal::exp2(Sfrag[3] - new_max);
    \\  const float factor = metal::exp2(max_score - new_max);
    \\  max_score = new_max;
    \\  const float rowsum = sushi_row_sum(Sfrag[0]) + sushi_row_sum(Sfrag[1])
    \\      + sushi_row_sum(Sfrag[2]) + sushi_row_sum(Sfrag[3]);
    \\  sum_score = sum_score * factor + rowsum;
    \\  for (int i = 0; i < NWV; ++i) Ofrag[i] *= factor;
    \\
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
    \\  // O += P @ V.
    \\  for (int id = 0; id < NWV; ++id) {
    \\    const int vbase = Vs_off + id * 8;
    \\    const float2 vf0 = float2(float(Vs[vbase]), float(Vs[vbase + 1]));
    \\    const float2 vf1 = float2(float(Vs[vbase + 8 * LDV]),
    \\                              float(Vs[vbase + 8 * LDV + 1]));
    \\    const float2 vf2 = float2(float(Vs[vbase + 16 * LDV]),
    \\                              float(Vs[vbase + 16 * LDV + 1]));
    \\    const float2 vf3 = float2(float(Vs[vbase + 24 * LDV]),
    \\                              float(Vs[vbase + 24 * LDV + 1]));
    \\    sushi_mma(Ofrag[id], Sfrag[0], vf0);
    \\    sushi_mma(Ofrag[id], Sfrag[1], vf1);
    \\    sushi_mma(Ofrag[id], Sfrag[2], vf2);
    \\    sushi_mma(Ofrag[id], Sfrag[3], vf3);
    \\  }
    \\}
    \\
    \\// Final chunk: normalize + store bf16 (output freshly allocated,
    \\// contiguous). Mid chunk: store the raw fp32 state for the next
    \\// dispatch (m/l written by all 4 lane-threads of a row — same value).
    \\const int local_row = tm + sm;
    \\if (local_row < q_rows) {
    \\  if (is_final) {
    \\    const float inv = 1.0f / sum_score;
    \\    device T* Optr = Op + (long)local_row * BDV + sn;
    \\    for (int id = 0; id < NWV; ++id) {
    \\      Optr[id * 8] = T(Ofrag[id].x * inv);
    \\      Optr[id * 8 + 1] = T(Ofrag[id].y * inv);
    \\    }
    \\  } else {
    \\    const long crow = (((long)bb * Hq + hq) * (long)qL + (long)(tqx * BQ + local_row));
    \\    m_out[crow] = max_score;
    \\    l_out[crow] = sum_score;
    \\    device float* Orow = o_out + crow * (long)BDV;
    \\    for (int id = 0; id < NWV; ++id) {
    \\      Orow[id * 8 + sn] = Ofrag[id].x;
    \\      Orow[id * 8 + sn + 1] = Ofrag[id].y;
    \\    }
    \\  }
    \\}
;

/// Which kernel a `sushi_attn_pd` dispatch runs: the SIMD-matrix kernel above, or
/// `sushi_attn_pd_nax` (src/kernels/attn_pd_nax.metal) on the M5 matrix units.
pub const AttnPdArm = enum { simd, nax };

/// [simd, nax]
var attn_pd_kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };

fn getAttnPdKernel(arm: AttnPdArm) !mlx.mlx_fast_metal_kernel {
    const slot = &attn_pd_kernels[@intFromEnum(arm)];
    if (slot.*) |kk| return kk;
    const kernel = try newAttnPdKernel(
        if (arm == .nax) "sushi_attn_pd_nax" else "sushi_attn_pd",
        if (arm == .nax) @embedFile("kernels/attn_pd_nax.metal") else ATTN_PD_KERNEL_SOURCE,
        if (arm == .nax) @embedFile("kernels/attn_pd_nax_header.metal") else ATTN_PD_KERNEL_HEADER,
    );
    slot.* = kernel;
    return kernel;
}

fn newAttnPdKernel(name: [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const input_names = [_][*:0]const u8{ "q", "k", "v", "scl", "win", "kr", "phase", "m_in", "l_in", "o_in", "mask", "skip", "sinks" };
    const output_names = [_][*:0]const u8{ "out", "m_out", "l_out", "o_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        name,
        in_vec,
        out_vec,
        source,
        header,
        false, // ensure_row_contiguous=false — K/V are cache VIEWS; a forced
        // contiguous copy of the full cache per layer would erase the win.
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    return kernel;
}

/// Test seam: the arm the LAST fusedSdpaPrefill call ran.
pub var attn_pd_last_arm: AttnPdArm = .simd;

/// null = not probed yet; the verdict holds for the process.
var attn_pd_nax_probe_ok: ?bool = null;
/// Test seam: perturb the probe's NAX output so the numeric check must fail.
pub var attn_pd_nax_probe_corrupt_override: bool = false;

pub fn attnPdNaxProbeReset() void {
    attn_pd_nax_probe_ok = null;
}

/// The NAX arm serves the MiMo global and sliding widths (192/128) without a QSA mask, on
/// matrix-unit silicon with cooperative input tensors (macOS 26.3, `qsaNaxOsOk`) whose
/// one-tile probe passed.
fn attnPdNaxServes(qk_dim: c_int, v_dim: c_int, masked: bool) bool {
    if (masked or qk_dim != 192 or v_dim != 128) return false;
    if (!(attn_pd_nax_override orelse true) or !verifyQmmNaxAvailable() or !qsaNaxOsOk()) return false;
    if (attn_pd_nax_probe_ok == null) attn_pd_nax_probe_ok = attnPdNaxRunProbe();
    return attn_pd_nax_probe_ok.?;
}

const ATTN_PD_NAX_PROBE_FLOOR: f32 = 4.9e-4;

/// Metal JIT-compiles at first EVAL, so both instantiations the engine dispatches (causal, and
/// band + sinks) run one ragged tile against an f32 composed reference before any prefill
/// trusts them: no worse than 1.5x the SIMD kernel's error. The latch a failure raised is dropped.
pub fn attnPdNaxRunProbe() bool {
    if (mlx.noGpuBackend()) return false;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return false;
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);
    for ([_]bool{ false, true }) |banded| {
        if (!attnPdNaxProbeCase(s, banded, had_error)) {
            log.info("[attn-pd] sushi_attn_pd_nax declined: probe failed ({s}); sushi_attn_pd serves\n", .{if (banded) "band + sinks" else "causal"});
            return false;
        }
    }
    return true;
}

fn attnPdNaxProbeCase(s: mlx.mlx_stream, banded: bool, had_error: bool) bool {
    // A ragged simdgroup (40 rows) and a partial key block (72 keys), gqa 2.
    const ql: c_int = 40;
    const kl: c_int = 72;
    const window: c_int = if (banded) 24 else 0;
    const scale: f32 = 1.0 / @sqrt(192.0);
    const q = qsaProbeLcgBf16(s, &[_]c_int{ 1, 2, ql, 192 }, 0xA77D01) orelse return false;
    defer _ = mlx.mlx_array_free(q);
    const k = qsaProbeLcgBf16(s, &[_]c_int{ 1, 1, kl, 192 }, 0xA77D02) orelse return false;
    defer _ = mlx.mlx_array_free(k);
    const v = qsaProbeLcgBf16(s, &[_]c_int{ 1, 1, kl, 128 }, 0xA77D03) orelse return false;
    defer _ = mlx.mlx_array_free(v);
    const sink_data = [_]f32{ 0.25, -0.5 };
    const sinks: mlx.mlx_array = if (banded) mlx.mlx_array_new_data(&sink_data, &[_]c_int{2}, 1, .float32) else .{ .ctx = null };
    defer if (sinks.ctx != null) {
        _ = mlx.mlx_array_free(sinks);
    };

    var dispatches: u32 = 0;
    const simd = (attnPdDispatch(s, .simd, q, k, v, scale, window, null, null, sinks, &dispatches) catch return false) orelse return false;
    defer _ = mlx.mlx_array_free(simd);
    var nax = (attnPdDispatch(s, .nax, q, k, v, scale, window, null, null, sinks, &dispatches) catch return false) orelse return false;
    defer _ = mlx.mlx_array_free(nax);
    if (mlx.mlx_array_eval(simd) != 0 or mlx.mlx_array_eval(nax) != 0) return false;
    if (!had_error and mlx.errorPending()) return false;
    if (attn_pd_nax_probe_corrupt_override) {
        const bad = qsaProbeCorruptFirst(s, nax) orelse return false;
        _ = mlx.mlx_array_free(nax);
        nax = bad;
    }

    var mask_buf: [ql * kl]bool = undefined;
    for (0..ql) |r| for (0..kl) |c| {
        const row: c_int = kl - ql + @as(c_int, @intCast(r));
        const col: c_int = @intCast(c);
        mask_buf[r * kl + c] = col <= row and (window == 0 or row - col < window);
    };
    const mask = mlx.mlx_array_new_data(&mask_buf, &[_]c_int{ 1, 1, ql, kl }, 4, .bool_);
    defer _ = mlx.mlx_array_free(mask);
    var wide: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (wide) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for ([_]mlx.mlx_array{ q, k, v }, &wide) |src, *dst| {
        if (mlx.mlx_astype(dst, src, .float32, s) != 0) return false;
    }
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    if (mlx.mlx_fast_scaled_dot_product_attention(&ref, wide[0], wide[1], wide[2], scale, "array", mask, sinks, false, s) != 0) return false;
    if (!qsaProbeAllFinite(s, nax) or !qsaProbeAllFinite(s, simd) or !qsaProbeAllFinite(s, ref)) return false;
    const err_simd = qsaProbeMaxAbsDiff(s, simd, ref) orelse return false;
    const err_nax = qsaProbeMaxAbsDiff(s, nax, ref) orelse return false;
    return err_nax <= @max(1.5 * err_simd, ATTN_PD_NAX_PROBE_FLOOR);
}

/// Test seam: `fused256_override = false` forces the composed path on both arms (the parity oracle).
pub var fused256_override: ?bool = null;
/// Test seam: true = the NAX arm wherever the hardware and its probe allow it, false = SIMD.
pub var attn_pd_nax_override: ?bool = null;

pub fn fused256Enabled() bool {
    return fused256_override orelse true;
}

/// Plain-CAUSAL dispatch mode. Default `.all` since the kv-chunk dispatch
/// budget landed (2026-07-22): the historical net-LOSS of every pre-budget
/// ratio-gated variant (8K: fused-to-ratio-4 231.5, fused-to-ratio-2 232.9,
/// composed 234.7 tok/s — monotonic: more fused = slower, despite the v2
/// kernel winning the µbench at kL <= 2*qL) was the macOS IOGPU
/// interactivity-preemption class: one long dispatch scanning the whole KV
/// gets preempted, and the penalty scales with dispatch length. With the
/// key axis split into budget-sized dispatches (see
/// FUSED256_DEFAULT_DISPATCH_BUDGET) the causal arm WINS the same
/// same-session A/B on the 27B: +2.9%/+2.3%/+4.6% at 8K/16K/32K.
/// The SLIDING-BAND arm (window > 0) has no such competition — composed
/// computes full-width scores + a GB-scale mask while the kernel
/// block-skips outside the band — so it
/// rides the master switch alone and never chunks.
pub const Fused256CausalMode = enum { all, off };

// ── Dispatch work budget (kv-axis chunked dispatch) ──
// One long dispatch scanning the whole KV monopolizes the GPU long enough at
// high kv_len to trip macOS IOGPU interactivity preemption, collapsing
// long-context prefill (oMLX issue #2225: the M3 Max cliff sits near ~1.5e9
// work units ≈ 50 ms/dispatch; design ported from oMLX's
// qwen35_fa256_attention.py, Apache-2.0, oMLX by jundot). Above the budget
// (work = batch·Hq·qL·kL) the CAUSAL arm splits the key axis into
// separately-dispatched BK-aligned chunks with exact online-softmax carry
// (fp32 m/l/O buffers chained between dispatches — bit-identical to the
// single-dispatch result). The band arm never chunks: its in-kernel block
// skip already bounds per-dispatch work by the window.
// The default mirrors oMLX's 250M fallback (~23 ms/dispatch on the
// M4 Max at the kernel's measured ~1.1e10 work-units/s).
pub const FUSED256_DEFAULT_DISPATCH_BUDGET: i64 = 250_000_000;
/// `sushi_attn_pd_nax` does the same work in about a third of the time, and its lockstep simdgroups
/// keep a longer key chunk in cache, so fewer fp32 carries pay.
pub const ATTN_PD_NAX_DISPATCH_BUDGET: i64 = 1_000_000_000;

/// Rows of a quantized cache one dispatch may dequantize. The work budget
/// already chunks the key axis, but it scales with heads x q_len, so a narrow
/// forward would otherwise rebuild the whole prompt in one slice. 16384 rows
/// is ~21 MB at the widest cache geometry we serve.
pub const PACKED_KV_SLICE_MAX: c_int = 16384;

/// Test seam: forces the budget (0 = single dispatch).
pub var fused256_budget_override: ?i64 = null;

pub fn fused256DispatchBudget(arm: AttnPdArm) i64 {
    if (fused256_budget_override) |v| return v;
    return if (arm == .nax) ATTN_PD_NAX_DISPATCH_BUDGET else FUSED256_DEFAULT_DISPATCH_BUDGET;
}

/// kv-axis chunk length for the budgeted causal dispatch: the largest
/// BK(32)-multiple keeping work-per-dispatch = batch·hq·ql·chunk within
/// budget, floored at one BK block, capped at kl. budget <= 0 = one dispatch.
pub fn fused256KvChunkLen(batch: i64, hq: i64, ql: i64, kl: i64, budget: i64) c_int {
    if (budget <= 0) return @intCast(kl);
    const per_key = batch * hq * ql;
    var chunk = @divTrunc(budget, @max(per_key, 1));
    chunk = @divTrunc(chunk, 32) * 32;
    if (chunk < 32) chunk = 32;
    if (chunk > kl) chunk = kl;
    return @intCast(chunk);
}

/// Test seam: dispatches issued by the LAST fusedSdpaPrefill call
/// (engagement is counted, never inferred from output equality).
pub var fused256_last_dispatch_count: u32 = 0;

pub fn fused256CausalMode() Fused256CausalMode {
    return if (fused256Enabled()) .all else .off;
}

/// (query/key width, value width) pairs `sushi_attn_pd` is instantiated for.
/// MLX's own steel prefill kernel ships bd 64/96/128 plus a 256 dsplit, so
/// 192/128 — MiMo-V2's global layers, and the MLA shape that scores over
/// nope+rope while storing narrower values — has no fused arm anywhere else.
pub fn fusedPdDims(qk_dim: c_int, v_dim: c_int) bool {
    return (qk_dim == 256 and v_dim == 256) or (qk_dim == 192 and v_dim == 128);
}

/// ONE predicate consumed by the three prefill OOM guards AND the dispatch
/// sites: true when prefill attention at this head_dim will NOT materialize
/// the composed [heads, chunk, total_kv] score tensor — either MLX's own
/// fused kernel covers it (<= 128) or `sushi_attn_pd` does for EVERY arm.
/// Keyed on causal mode: at `.all` (the default since the budgeted-dispatch
/// flip) no score tensor materializes. Guards and dispatch must never drift
/// (the effectivePrefillChunk rule).
///
/// A width listed here owes a dispatch at EVERY prefill site that scores at
/// it: 192 covers mimo_v2's global layers and MLA. mimo_v2's SLIDING layers
/// (band + sinks) ride their own predicate, `slidingPrefillFused`, which
/// `server.slidingBandScoreBytes` reads too.
pub fn prefillHeadDimFused(head_dim: u32) bool {
    if (head_dim <= 128) return true;
    if (head_dim != 256 and head_dim != 192) return false;
    return fused256CausalMode() == .all;
}

/// Shortest query block the fused hd-256 kernel accepts. Below it MLX's
/// sdpa_vector wins (see the decline in fusedSdpaPrefill).
pub const FUSED256_MIN_Q_LEN: c_int = 16;

/// Try the fused flash prefill kernel (sushi_attn_pd, see `fusedPdDims`).
/// Returns null when a precondition doesn't hold — the caller falls back to
/// the composed path. `window` > 0 adds the sliding-band mask (sliding-window layers,
/// createSlidingWindowMask semantics: key masked when row_abs - col >=
/// window); 0 = plain bottom-right causal. Free function (stream param) so
/// the MTP head can use it too.
///
/// STRUCTURAL precondition (not checkable — lazy arrays carry no strides at
/// graph-build time): q/k/v innermost dim contiguous. True for fresh
/// rope/transpose outputs, cache views (slices along T of a contiguous
/// buffer), and dequantized dense views — i.e. every attention call site.
pub fn fusedSdpaPrefill(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    window: c_int,
) !?mlx.mlx_array {
    if (!fusedPrefillArmOn(q, window)) return null;
    return fusedSdpaPrefillImpl(s, q, k, v, scale, window, null, null, .{ .ctx = null });
}

fn fusedPrefillArmOn(q: mlx.mlx_array, window: c_int) bool {
    // Band arm rides the master switch; plain causal is opt-in (see
    // fused256CausalMode — composed wins live on qwen chunked prefill).
    if (window > 0) return fused256Enabled();
    if (fused256CausalMode() == .off) return false;
    // The band arm stays ours (mlx's NAX kernel takes no band); the
    // causal arm yields to the stock fused kernel on NAX — which exists
    // at hd 256 only, so a narrower-value shape keeps our kernel.
    return !(naxSdpaPreferred() and mlx.mlx_array_ndim(q) == 4 and mlx.getShape(q)[3] == 256);
}

/// The causal arm over a KV cache view. When the cache is quantized the kernel
/// reads it one dispatch at a time (`PackedKv`) and `view.k`/`view.v` — the
/// lazy dense rebuild `denseView` set up — are never evaluated. Falls through
/// to the dense arm on an `off`-scheme cache or a shape the slicing cannot
/// serve, so the caller has one entry either way. The band arm reads the view
/// whole: a ringed sliding view is already one dispatch's rows. `sinks` ([Hq],
/// null = none) is a learned logit per head that joins the softmax as one
/// extra column with no value row.
pub fn fusedSdpaPrefillKv(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    view: *const DenseKVView,
    scale: f32,
    window: c_int,
    sinks: mlx.mlx_array,
) !?mlx.mlx_array {
    if (window > 0 or !view.has_quant_triple or view.bits == 0 or view.group_size == 0) {
        if (!fusedPrefillArmOn(q, window)) return null;
        return fusedSdpaPrefillImpl(s, q, view.k, view.v, scale, window, null, null, sinks);
    }
    if (fused256CausalMode() == .off) return null;
    return fusedSdpaPrefillImpl(s, q, view.k, view.v, scale, 0, null, .{
        .k_q = view.k_triple_q,
        .k_scales = view.k_triple_scales,
        .k_biases = view.k_triple_biases,
        .v_q = view.v_triple_q,
        .v_scales = view.v_triple_scales,
        .v_biases = view.v_triple_biases,
        .bits = view.bits,
        .group_size = view.group_size,
    }, sinks);
}

/// Test seam for the switch half of `slidingPrefillFused` (never the shape half).
pub var sliding_prefill_fused_override: ?bool = null;

/// ONE predicate for a sliding layer's prefill dispatch (`slidingPrefillAttn`)
/// and `server.slidingBandScoreBytes`: true when the band + sink attention runs
/// in `sushi_attn_pd` and no [heads, q, window + q - 1] score sheet exists.
pub fn slidingPrefillFused(cfg: *const ModelConfig, q_len: u64) bool {
    const on = sliding_prefill_fused_override orelse fused256Enabled();
    const v_dim = if (cfg.v_head_dim > 0) cfg.v_head_dim else cfg.head_dim;
    return on and q_len >= FUSED256_MIN_Q_LEN and fusedPdDims(@intCast(cfg.head_dim), @intCast(v_dim));
}

var sliding_prefill_engaged_logged = false;

/// A sliding layer's prefill attention over its (ringed) view, sinks included.
/// Null = the composed path serves.
pub fn slidingPrefillAttn(
    s: mlx.mlx_stream,
    cfg: *const ModelConfig,
    q: mlx.mlx_array,
    view: *const DenseKVView,
    scale: f32,
    sinks: mlx.mlx_array,
) !?mlx.mlx_array {
    const q_len = mlx.getShape(q)[2];
    if (!slidingPrefillFused(cfg, @intCast(q_len))) return null;
    const out = try fusedSdpaPrefillKv(s, q, view, scale, @intCast(cfg.sliding_window), sinks);
    if (out != null and !sliding_prefill_engaged_logged) {
        sliding_prefill_engaged_logged = true;
        log.info("[attn-pd] sliding band engaged: {s} window={d} sinks={} qL={d} kL={d}\n", .{ attnPdKernelName(attn_pd_last_arm), cfg.sliding_window, sinks.ctx != null, q_len, mlx.getShape(view.k)[2] });
    }
    return out;
}

/// One line per (qk, value) width the kernel actually serves in this process:
/// an A/B arm is proven by an engagement line in its own log, not by the flag
/// it was launched with.
var attn_pd_engaged_logged: [2][2]bool = .{ .{ false, false }, .{ false, false } };

fn attnPdKernelName(arm: AttnPdArm) []const u8 {
    return if (arm == .nax) "sushi_attn_pd_nax" else "sushi_attn_pd";
}

fn logAttnPdEngaged(arm: AttnPdArm, qk_dim: c_int, v_dim: c_int, q_len: c_int, kv_len: c_int, dispatches: u32) void {
    const slot = &attn_pd_engaged_logged[@intFromEnum(arm)][if (qk_dim == 256) 0 else 1];
    if (slot.*) return;
    slot.* = true;
    log.info("[attn-pd] engaged: {s} qk={d} v={d} qL={d} kL={d} dispatches={d}\n", .{ attnPdKernelName(arm), qk_dim, v_dim, q_len, kv_len, dispatches });
}

/// True when a bf16 view's bound address is 16-byte aligned, which every `uint4` K/V staging load needs. A lazy
/// array has no address yet and passes: the stride gates are all that can be checked for it.
pub fn attn256Base16Aligned(a: mlx.mlx_array) bool {
    var avail = false;
    if (mlx._mlx_array_is_available(&avail, a) != 0 or !avail) return true;
    const p = mlx.mlx_array_data_uint32(a) orelse return true;
    return @intFromPtr(p) % 16 == 0;
}

/// Per-(batch, q tile, key block) any-visible table for the QSA arm:
/// [B, ceil(qL/64), ceil(kL/32)] bool.
fn qsaSkipTable(s: mlx.mlx_stream, mask: mlx.mlx_array) !mlx.mlx_array {
    const ms = mlx.getShape(mask);
    const nq: c_int = @divTrunc(ms[2] + 63, 64);
    const nk: c_int = @divTrunc(ms[3] + 31, 32);
    const axes = [_]c_int{ 2, 3 };
    const low = [_]c_int{ 0, 0 };
    const high = [_]c_int{ nq * 64 - ms[2], nk * 32 - ms[3] };
    const falsev = mlx.mlx_array_new_bool(false);
    defer _ = mlx.mlx_array_free(falsev);
    var padded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(padded);
    try mlx.check(mlx.mlx_pad(&padded, mask, &axes, 2, &low, 2, &high, 2, falsev, "constant", s));
    var r5 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r5);
    try mlx.check(mlx.mlx_reshape(&r5, padded, &[_]c_int{ ms[0], nq, 64, nk, 32 }, 5, s));
    const red = [_]c_int{ 2, 4 };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_any_axes(&out, r5, &red, 2, false, s));
    return out;
}

/// Virtual key index -> cache position for one query's block selection:
/// the `sel_len` tokens of its sorted blocks, then its own incomplete tail.
pub const ATTN_QSA256_KERNEL_HEADER = ATTN_PD_KERNEL_HEADER ++
    \\inline int sushi_qsa_pos(const device int* blk, int vi, int sel_len, int tail_start, int ratio) {
    \\  const int b = vi / ratio;
    \\  return (vi < sel_len) ? (blk[b] * ratio + (vi - b * ratio)) : (tail_start + (vi - sel_len));
    \\}
    \\
;

// The Metal JIT of a cooperative-tensor kernel happens at first eval, not at
// construction, so `qsaNaxArm` runs it once (tiny apply + eval, inference
// thread) and a failure latches the stock gather for the process.
var qsa_nax_os_cached: ?bool = null;

pub fn qsaNaxOsSupported(version: []const u8) bool {
    return macosVersionAtLeast(version, 26, 3);
}

pub fn qsaNaxOsOk() bool {
    if (qsa_nax_os_cached) |v| return v;
    var version_buf: [64]u8 = undefined;
    const supported = qsaNaxOsSupported(macosProductVersion(&version_buf) orelse "");
    qsa_nax_os_cached = supported;
    return supported;
}

pub fn qsaProbeLcgBf16(s: mlx.mlx_stream, shape: []const c_int, seed: u32) ?mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = std.heap.page_allocator.alloc(f32, n) catch return null;
    defer std.heap.page_allocator.free(buf);
    var state = seed;
    for (buf) |*x| {
        state = state *% 1664525 +% 1013904223;
        x.* = @as(f32, @floatFromInt(state >> 8)) * (1.0 / 16777216.0) - 0.5;
    }
    const host = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(host);
    var out = mlx.mlx_array_new();
    if (mlx.mlx_astype(&out, host, .bfloat16, s) != 0) {
        _ = mlx.mlx_array_free(out);
        return null;
    }
    if (mlx.mlx_array_eval(out) != 0) {
        _ = mlx.mlx_array_free(out);
        return null;
    }
    return out;
}

pub fn qsaProbeCorruptFirst(s: mlx.mlx_stream, arr: mlx.mlx_array) ?mlx.mlx_array {
    var contig = mlx.mlx_array_new();
    if (mlx.mlx_contiguous(&contig, arr, false, s) != 0) {
        _ = mlx.mlx_array_free(contig);
        return null;
    }
    defer _ = mlx.mlx_array_free(contig);
    var f32a = mlx.mlx_array_new();
    if (mlx.mlx_astype(&f32a, contig, .float32, s) != 0) {
        _ = mlx.mlx_array_free(f32a);
        return null;
    }
    defer _ = mlx.mlx_array_free(f32a);
    if (mlx.mlx_array_eval(f32a) != 0) return null;
    const n = mlx.mlx_array_size(f32a);
    if (n == 0) return null;
    const src = mlx.mlx_array_data_float32(f32a) orelse return null;
    const buf = std.heap.page_allocator.alloc(f32, n) catch return null;
    defer std.heap.page_allocator.free(buf);
    @memcpy(buf, src[0..n]);
    buf[0] += 1.0;
    const shape = mlx.getShape(arr);
    const host = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(host);
    var out = mlx.mlx_array_new();
    if (mlx.mlx_astype(&out, host, mlx.mlx_array_dtype(arr), s) != 0) {
        _ = mlx.mlx_array_free(out);
        return null;
    }
    return out;
}

pub fn qsaProbeAllFinite(s: mlx.mlx_stream, a: mlx.mlx_array) bool {
    var fin = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fin);
    if (mlx.mlx_isfinite(&fin, a, s) != 0) return false;
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    if (mlx.mlx_all(&all, fin, false, s) != 0) return false;
    if (mlx.mlx_array_eval(all) != 0) return false;
    var ok: bool = false;
    if (mlx.mlx_array_item_bool(&ok, all) != 0) return false;
    return ok;
}

pub fn qsaProbeMaxAbsDiff(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) ?f32 {
    var sub = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sub);
    if (mlx.mlx_subtract(&sub, a, b, s) != 0) return null;
    var ab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ab);
    if (mlx.mlx_abs(&ab, sub, s) != 0) return null;
    var mx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx);
    if (mlx.mlx_max(&mx, ab, false, s) != 0) return null;
    if (mlx.mlx_array_eval(mx) != 0) return null;
    var v: f32 = 0;
    if (mlx.mlx_array_item_float32(&v, mx) != 0) return null;
    return v;
}

/// Dequantizes chunk `c8` (8 values) of one packed row into `ev`. 4 and 8 bits fill whole uint32
/// words; 6 bits packs four values into three bytes (little-endian bitstream, `mlx_quantize`'s
/// layout), so a chunk is six bytes and the row is read bytewise.
pub const KV_UNPACK_HEADER =
    \\template <typename T, int BITS>
    \\inline void sushi_kv_unpack8(const device uint32_t* wq, int c8, float sj, float bj, thread T* ev) {
    \\  constexpr uint MASKB = (1u << BITS) - 1u;
    \\  if constexpr (BITS == 6) {
    \\    const device uchar* p = ((const device uchar*)wq) + c8 * 6;
    \\    const uint lo = uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16);
    \\    const uint hi = uint(p[3]) | (uint(p[4]) << 8) | (uint(p[5]) << 16);
    \\    for (int u = 0; u < 4; ++u) {
    \\      ev[u] = T(float((lo >> (u * 6)) & MASKB) * sj + bj);
    \\      ev[4 + u] = T(float((hi >> (u * 6)) & MASKB) * sj + bj);
    \\    }
    \\  } else {
    \\    constexpr int VPW = 32 / BITS;
    \\    constexpr int NW = 8 / VPW;
    \\    for (int wj = 0; wj < NW; ++wj) {
    \\      const uint w = wq[c8 * NW + wj];
    \\      for (int u = 0; u < VPW; ++u) ev[wj * VPW + u] = T(float((w >> (u * BITS)) & MASKB) * sj + bj);
    \\    }
    \\  }
    \\}
    \\
    \\// Sixteen 6-bit values from three words (96 bits), as the matmul2d kernels load them.
    \\template <typename T>
    \\inline void sushi_kv_store16_b6(uint w0, uint w1, uint w2, float sj, float bj, threadgroup T* dst) {
    \\  const uint ws[4] = {w0, w1, w2, 0u};
    \\  #pragma clang loop unroll(full)
    \\  for (int u = 0; u < 16; ++u) {
    \\    const int bit = u * 6;
    \\    const int wi = bit >> 5;
    \\    const int sh = bit & 31;
    \\    uint v = ws[wi] >> sh;
    \\    if (sh + 6 > 32) v |= ws[wi + 1] << (32 - sh);
    \\    dst[u] = T(float(v & 63u) * sj + bj);
    \\  }
    \\}
    \\
;












pub const SingleVerifyFeature = enum { routing, down_reduce, hc_graph };
pub var single_verify_test_features: [3]bool = @splat(true);



pub var mtp_coarse_pair_dispatches: u64 = 0;

pub var mtp_verify_indexed_input_calls: u64 = 0;

pub var mtp_verify_expert_reduce_calls: u64 = 0;

pub var mtp_verify_expert_pairs_calls: u64 = 0;
pub var mtp_verify_expert_rows_calls: u64 = 0;



const SharedProjKind = enum { qmatmul, mtp_rows, verify_rows };



pub var mtp_verify_kernel_calls: [3]u64 = @splat(0);


pub const QSA_SCORE_KERNEL_HEADER =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup>
    \\#include <metal_simdgroup_matrix>
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace metal;
    \\
;

// The lane -> (row, col) map (`fm`/`fn`) is the matmul2d cooperative-fragment layout,
// hand-derived from the SDK; every A, B and C fragment must use it. K stays 8 ascending
// 16-deep `op.run` calls: that is the stock tf32 GEMM's order and the bit-identity contract.








const QSA_ATTN_Q_SOURCE =
    \\constexpr int KC = DK / 8;
    \\constexpr int VC = DV / 8;
    \\constexpr int LDK = BK + 8;
    \\constexpr int LDV = DV + 8;
    \\// Simdgroups past NSG only stage tiles: who loads an element never changes its value.
    \\constexpr int NT = 32 * NSGT;
    \\constexpr int KT = BK / 8;
    \\constexpr int KV_ELEMS = (DK * LDK > BK * LDV) ? DK * LDK : BK * LDV;
    \\// Per-length split counts ride the grid, never a template.
    \\const int NSPLIT = int(threadgroups_per_grid.z);
    \\
    \\const int qL = q_shape[2];
    \\const int kL = kq_shape[2];
    \\const int Hq = q_shape[1];
    \\const int Hk = kq_shape[1];
    \\const int gqa = Hq / Hk;
    \\const int KB = blocks_shape[2];
    \\
    \\const int s = int(threadgroup_position_in_grid.x);
    \\const int hk = int(threadgroup_position_in_grid.y);
    \\// The caller pins batch to 1 (qs[0] != 1 declines), so the grid's z is
    \\// free to carry the split index instead of the batch index.
    \\const int bb = 0;
    \\const int jsp = int(threadgroup_position_in_grid.z);
    \\const ushort lane = ushort(thread_index_in_simdgroup);
    \\const ushort warp = ushort(simdgroup_index_in_threadgroup);
    \\const int tix = int(thread_index_in_threadgroup);
    \\
    \\const float scale_log2e = scl[0] * 1.44269504088896340736f;
    \\// Query s sits at absolute key position p (bottom-right aligned). Its
    \\// visible keys: `count` complete blocks, then the tail [tail_start, p].
    \\// SPARSE=0 has no block list: its whole causal range [0, p] is the tail.
    \\const int p = (kL - qL) + s;
    \\const int complete = (p + 1) / RATIO;
    \\const int count = SPARSE ? metal::min(complete, KB) : 0;
    \\const int sel_len = count * RATIO;
    \\const int tail_start = SPARSE ? complete * RATIO : 0;
    \\const int L = sel_len + (p + 1 - tail_start);
    \\// Split-K. This threadgroup owns virtual keys [t_lo, t_hi) of the row's
    \\// own range [0, L). The virtual index already concatenates the row's
    \\// selected blocks and THEN its ragged tail, so a contiguous virtual
    \\// range is exactly a contiguous block range — plus, for the last
    \\// non-empty split, the tail. Boundaries are BK-aligned so every tile
    \\// but a split's last is full. A split starting past L is EMPTY and
    \\// writes the merge's identity.
    \\const int tiles = (L + BK - 1) / BK;
    \\// ceil(tiles/NSPLIT) per split, so the last splits can be short or empty; the
    \\// merge is exact over whatever the boundaries are.
    \\const int tiles_per = (tiles + NSPLIT - 1) / NSPLIT;
    \\const int t_lo = jsp * tiles_per * BK;
    \\const int t_hi = metal::min(L, t_lo + tiles_per * BK);
    \\
    \\const long block_base = (long)bb * blocks_strides[0] + (long)s * blocks_strides[1];
    \\const long kq_base  = (long)bb * kq_strides[0]  + (long)hk * kq_strides[1];
    \\const long ksc_base = (long)bb * ksc_strides[0] + (long)hk * ksc_strides[1];
    \\const long kbi_base = (long)bb * kbi_strides[0] + (long)hk * kbi_strides[1];
    \\const long vq_base  = (long)bb * vq_strides[0]  + (long)hk * vq_strides[1];
    \\const long vsc_base = (long)bb * vsc_strides[0] + (long)hk * vsc_strides[1];
    \\const long vbi_base = (long)bb * vbi_strides[0] + (long)hk * vbi_strides[1];
    \\
    \\threadgroup T KVs[KV_ELEMS];
    \\threadgroup T* Ks = KVs;
    \\threadgroup T* Vs = KVs;
    \\
    \\const short2 sc = sushi_coord(lane);
    \\const short sn = sc.x;
    \\const short sm = sc.y;
    \\const short tm = 8 * short(warp);
    \\const int Ks_off = sm * LDK + sn;
    \\const int Vs_off = sm * LDV + sn;
    \\const int row = tm + sm;
    \\const bool row_ok = row < gqa;
    \\const bool mma_sg = int(warp) < NSG;
    \\
    \\float2 Qfrag[KC];
    \\if (row_ok) {
    \\  const device T* Qrow = q + bb * q_strides[0] + (long)(hk * gqa + row) * q_strides[1] + (long)s * q_strides[2];
    \\  for (int dd = 0; dd < KC; ++dd) {
    \\    const vec<T, 2> pr = *((const device vec<T, 2>*)(Qrow + dd * 8 + sn));
    \\    Qfrag[dd] = float2(float(pr.x), float(pr.y));
    \\  }
    \\} else {
    \\  for (int dd = 0; dd < KC; ++dd) Qfrag[dd] = float2(0.0f);
    \\}
    \\float2 Ofrag[VC];
    \\for (int i = 0; i < VC; ++i) Ofrag[i] = float2(0.0f);
    \\float max_score = -3.0e38f;
    \\float sum_score = 0.0f;
    \\
    \\for (int t0 = t_lo; t0 < t_hi; t0 += BK) {
    \\  const int rows_k = metal::min(BK, t_hi - t0);
    \\
    \\  // K tile: dequantize into the SAME transposed staging the dense
    \\  // kernel fills, so the fragment math below is untouched. The 8 dims of
    \\  // one chunk share a quant group (GS is a multiple of 8), so the
    \\  // scale/bias pair is read once per chunk, not per element.
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * KC; i += NT) {
    \\    const int r = i / KC;
    \\    const int c8 = i % KC;
    \\    const int cb = c8 * 8;
    \\    T ev[8];
    \\    if (r < rows_k) {
    \\      const int vi = t0 + r;
    \\      const int bi = vi / RATIO;
    \\      const int pos = vi < sel_len ? blocks[block_base + bi] * RATIO + (vi - bi * RATIO) : tail_start + vi - sel_len;
    \\      if (QUANT) {
    \\      const long wrow = kq_base + (long)pos * kq_strides[2];
    \\      const int g = cb / GS;
    \\      const float sj = float(ksc[ksc_base + (long)pos * ksc_strides[2] + (long)g]);
    \\      const float bj = float(kbi[kbi_base + (long)pos * kbi_strides[2] + (long)g]);
    \\      sushi_kv_unpack8<T, BITS>((const device uint32_t*)(kq + wrow), c8, sj, bj, ev);
    \\      } else {
    \\        for (int j = 0; j < 8; ++j) ev[j] = T(kq[kq_base + (long)pos * kq_strides[2] + cb + j]);
    \\      }
    \\    } else {
    \\      for (int j = 0; j < 8; ++j) ev[j] = T(0.0f);
    \\    }
    \\    for (int j = 0; j < 8; ++j) Ks[(cb + j) * LDK + r] = ev[j];
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
    \\  float2 Sfrag[KT];
    \\  for (int i = 0; i < KT; ++i) Sfrag[i] = float2(0.0f);
    \\  if (mma_sg) {
    \\  for (int dd = 0; dd < KC; ++dd) {
    \\    const float2 qf = Qfrag[dd];
    \\    const int kbase = Ks_off + dd * 8 * LDK;
    \\    for (int kt = 0; kt < KT; ++kt) {
    \\      const float2 kf = float2(float(Ks[kbase + kt * 8]), float(Ks[kbase + kt * 8 + 1]));
    \\      sushi_mma(Sfrag[kt], qf, kf);
    \\    }
    \\  }
    \\  for (int kt = 0; kt < KT; ++kt) Sfrag[kt] *= scale_log2e;
    \\  if (rows_k < BK) {
    \\    for (int kt = 0; kt < KT; ++kt) {
    \\      if (kt * 8 + sn >= rows_k) Sfrag[kt].x = -INFINITY;
    \\      if (kt * 8 + sn + 1 >= rows_k) Sfrag[kt].y = -INFINITY;
    \\    }
    \\  }
    \\  }
    \\
    \\  // V tile: dequantize straight into the row-major staging.
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * VC; i += NT) {
    \\    const int r = i / VC;
    \\    const int c8 = i % VC;
    \\    const int cb = c8 * 8;
    \\    T ev[8];
    \\    if (r < rows_k) {
    \\      const int vi = t0 + r;
    \\      const int bi = vi / RATIO;
    \\      const int pos = vi < sel_len ? blocks[block_base + bi] * RATIO + (vi - bi * RATIO) : tail_start + vi - sel_len;
    \\      if (QUANT) {
    \\      const long wrow = vq_base + (long)pos * vq_strides[2];
    \\      const int g = cb / GS;
    \\      const float sj = float(vsc[vsc_base + (long)pos * vsc_strides[2] + (long)g]);
    \\      const float bj = float(vbi[vbi_base + (long)pos * vbi_strides[2] + (long)g]);
    \\      sushi_kv_unpack8<T, BITS>((const device uint32_t*)(vq + wrow), c8, sj, bj, ev);
    \\      } else {
    \\        for (int j = 0; j < 8; ++j) ev[j] = T(vq[vq_base + (long)pos * vq_strides[2] + cb + j]);
    \\      }
    \\    } else {
    \\      for (int j = 0; j < 8; ++j) ev[j] = T(0.0f);
    \\    }
    \\    for (int j = 0; j < 8; ++j) Vs[r * LDV + cb + j] = ev[j];
    \\  }
    \\
    \\  if (mma_sg) {
    \\  float new_max = max_score;
    \\  for (int kt = 0; kt < KT; ++kt) new_max = metal::max(new_max, sushi_row_max(Sfrag[kt]));
    \\  float rowsum = 0.0f;
    \\  for (int kt = 0; kt < KT; ++kt) {
    \\    Sfrag[kt] = metal::exp2(Sfrag[kt] - new_max);
    \\    rowsum += sushi_row_sum(Sfrag[kt]);
    \\  }
    \\  const float factor = metal::exp2(max_score - new_max);
    \\  max_score = new_max;
    \\  sum_score = sum_score * factor + rowsum;
    \\  for (int i = 0; i < VC; ++i) Ofrag[i] *= factor;
    \\  }
    \\
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (mma_sg) {
    \\  for (int id = 0; id < VC; ++id) {
    \\    const int vbase = Vs_off + id * 8;
    \\    for (int kt = 0; kt < KT; ++kt) {
    \\      const float2 vf = float2(float(Vs[vbase + kt * 8 * LDV]), float(Vs[vbase + kt * 8 * LDV + 1]));
    \\      sushi_mma(Ofrag[id], Sfrag[kt], vf);
    \\    }
    \\  }
    \\  }
    \\}
    \\
    \\// Partials for the merge pass, UNNORMALIZED: the running max and the
    \\// running sum belong to this split alone, and only the merge knows the
    \\// row's global max. `sushi_row_max`/`sushi_row_sum` reduce across the four
    \\// lanes holding a row, so every lane of the row carries the same m/l
    \\// and exactly one of them (sn == 0) writes the pair.
    \\//
    \\// An empty split writes l = 0, which is the merge's skip flag: a
    \\// NON-empty split always has l >= 1, because its own maximum element
    \\// contributes exp2(max - max) = 1. So l == 0 identifies emptiness
    \\// exactly, and the merge never evaluates (-inf) - (-inf).
    \\if (row_ok) {
    \\  const int hq = hk * gqa + row;
    \\  const long pidx = ((long)hq * (long)qL + (long)s) * NSPLIT + jsp;
    \\  device float* Aptr = pacc + pidx * DV + sn;
    \\  for (int id = 0; id < VC; ++id) {
    \\    Aptr[id * 8] = Ofrag[id].x;
    \\    Aptr[id * 8 + 1] = Ofrag[id].y;
    \\  }
    \\  if (sn == 0) {
    \\    const bool empty = (t_lo >= t_hi);
    \\    pml[pidx * 2] = empty ? -INFINITY : max_score;
    \\    pml[pidx * 2 + 1] = empty ? 0.0f : sum_score;
    \\  }
    \\}
;

/// sushi_attn_qsa256_qmerge: the split-K reduction, one threadgroup per (q head, query row).
/// Standard flash-decoding merge in the log2 domain; `l_j == 0` marks an empty split and is
/// skipped. NSPLIT, qL and DV ride `pacc_shape`, so the config depends on nothing per token.
const QSA_ATTN_MERGE_SOURCE =
    \\const int DV = pacc_shape[3];
    \\const int d = int(thread_index_in_threadgroup);
    \\const int s = int(threadgroup_position_in_grid.y);
    \\const int hq = int(threadgroup_position_in_grid.z);
    \\const int qL = pacc_shape[1];
    \\const int NS = pacc_shape[2];
    \\const long pbase = ((long)hq * (long)qL + (long)s) * NS;
    \\
    \\// Every thread reads all NS (m, l) pairs. The pairs land in cache after
    \\// the first thread of the threadgroup touches them, which is
    \\// cheaper than a barrier plus threadgroup storage for two floats.
    \\float M = -INFINITY;
    \\for (int j = 0; j < NS; ++j) {
    \\  if (pml[(pbase + j) * 2 + 1] > 0.0f) M = metal::max(M, pml[(pbase + j) * 2]);
    \\}
    \\float Z = 0.0f;
    \\float acc = 0.0f;
    \\for (int j = 0; j < NS; ++j) {
    \\  const float lj = pml[(pbase + j) * 2 + 1];
    \\  if (lj <= 0.0f) continue;
    \\  const float w = metal::exp2(pml[(pbase + j) * 2] - M);
    \\  Z += w * lj;
    \\  acc += w * pacc[(pbase + j) * DV + d];
    \\}
    \\out[((long)hq * (long)qL + (long)s) * DV + d] = T(acc / Z);
;

var qsa_attn_merge_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQsaAttnMergeKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_attn_merge_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "pacc", "pml" };
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_attn_qsa256_qmerge",
        in_vec,
        out_vec,
        QSA_ATTN_MERGE_SOURCE,
        ATTN_QSA256_KERNEL_HEADER,
        false,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_attn_merge_kernel_cached = kernel;
    return kernel;
}

/// Threadgroup tile depth: 16 keys stages 12 KiB, half the dense prefill kernel's footprint.
pub const QSA_ATTN_BK: c_int = 16;

var qsa_attn_q_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQsaAttnQKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_attn_q_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "q", "scl", "blocks", "kq", "ksc", "kbi", "vq", "vsc", "vbi" };
    const output_names = [_][*:0]const u8{ "pacc", "pml" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_attn_qsa256_qsplit",
        in_vec,
        out_vec,
        QSA_ATTN_Q_SOURCE,
        KV_UNPACK_HEADER ++ ATTN_QSA256_KERNEL_HEADER,
        false, // packed triples are cache VIEWS — a contiguous copy erases the win
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_attn_q_kernel_cached = kernel;
    return kernel;
}

/// Keyed on everything the config depends on and nothing else: `kv`/`kb` are runtime shapes,
/// `NSPLIT` changes the grid and both output shapes.
/// A small LRU of metal_kernel configs: adaptive MTP alternates verify widths within one
/// request, and a single slot rebuilt on every alternation. Handles are borrowed.
pub fn QsaCfgCache(comptime Key: type, comptime NCFG: usize) type {
    return struct {
        const Self = @This();
        pub const SLOTS: usize = 4;
        /// Named so the default can be written as an array literal.
        const MaybeCfg = ?mlx.mlx_fast_metal_kernel_config;
        const Entry = struct {
            key: Key = std.mem.zeroes(Key),
            cfg: [NCFG]MaybeCfg = @splat(null),
            stamp: u64 = 0,
        };
        entries: [SLOTS]Entry = @splat(.{}),
        clock: u64 = 0,
        /// Build counter.
        builds: usize = 0,

        pub fn get(self: *Self, key: Key) ?[NCFG]mlx.mlx_fast_metal_kernel_config {
            for (&self.entries) |*e| {
                if (e.cfg[0] == null) continue;
                if (!std.meta.eql(e.key, key)) continue;
                self.clock += 1;
                e.stamp = self.clock;
                var out: [NCFG]mlx.mlx_fast_metal_kernel_config = undefined;
                for (e.cfg, 0..) |c, i| out[i] = c.?;
                return out;
            }
            return null;
        }

        pub fn put(self: *Self, key: Key, cfg: [NCFG]mlx.mlx_fast_metal_kernel_config) void {
            var victim: usize = 0;
            var oldest: u64 = std.math.maxInt(u64);
            for (&self.entries, 0..) |*e, i| {
                if (e.cfg[0] == null) {
                    victim = i;
                    oldest = 0;
                    break;
                }
                if (e.stamp < oldest) {
                    oldest = e.stamp;
                    victim = i;
                }
            }
            const e = &self.entries[victim];
            for (&e.cfg) |*c| {
                if (c.*) |cc| _ = mlx.mlx_fast_metal_kernel_config_free(cc);
                c.* = null;
            }
            e.key = key;
            for (cfg, 0..) |c, i| e.cfg[i] = c;
            self.clock += 1;
            e.stamp = self.clock;
            self.builds += 1;
        }
    };
}

pub const QsaAttnCfgKey = struct {
    q_shape: ShapeKey,
    /// Shapes the grid and is not recoverable from the q shape.
    h_kv: c_int,
    dv: c_int,
    bits: u8,
    gs: u32,
    quant: bool,
    /// False = no block list: every row attends its whole causal range.
    sparse: bool,
    nsg: c_int,
    /// Simdgroups per threadgroup, `nsg` of them computing.
    nsgt: c_int,
    nsplit: c_int,
    bk: c_int,
    ratio: c_int,
    dtype: mlx.mlx_dtype,
};
pub var qsa_attn_cfgs = QsaCfgCache(QsaAttnCfgKey, 2){};

/// One split-K pass plus its merge. `arrays` = the packed K and V triples, or dense K and V
/// three times each; `blocks` is read only when `key.sparse`. Null = a kernel failed to build.
pub fn splitKAttnDispatch(s: mlx.mlx_stream, q: mlx.mlx_array, blocks: mlx.mlx_array, arrays: [6]mlx.mlx_array, attn_scale: f32, key: QsaAttnCfgKey) !?mlx.mlx_array {
    const kernel = getQsaAttnQKernel() catch return null;
    const merge_kernel = getQsaAttnMergeKernel() catch return null;
    const qs = mlx.getShape(q);
    const seq_len = qs[2];
    const scl_data = [_]f32{attn_scale};
    const one = [_]c_int{1};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);

    const cfgs: [2]mlx.mlx_fast_metal_kernel_config = qsa_attn_cfgs.get(key) orelse blk: {
        // Split pass: one threadgroup per (row, kv head, split).
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        const acc_shape = [_]c_int{ qs[1], seq_len, key.nsplit, key.dv };
        const ml_shape = [_]c_int{ qs[1], seq_len, key.nsplit, 2 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &acc_shape, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, seq_len * 32, key.h_kv * key.nsgt, key.nsplit));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, key.nsgt, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", key.dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSG", key.nsg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSGT", key.nsgt));        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BK", key.bk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RATIO", key.ratio));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(key.bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(key.gs)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "QUANT", @intFromBool(key.quant)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DK", qs[3]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DV", key.dv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "SPARSE", @intFromBool(key.sparse)));

        // Merge pass: one threadgroup per (q head, row), one thread per value element.
        const mconfig = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(mconfig);
        const o_shape = [_]c_int{ qs[0], qs[1], seq_len, key.dv };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mconfig, &o_shape, 4, key.dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mconfig, key.dv, seq_len, qs[1]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mconfig, key.dv, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(mconfig, "T", key.dtype));

        qsa_attn_cfgs.put(key, .{ config, mconfig });
        break :blk .{ config, mconfig };
    };

    const inputs_arr = [_]mlx.mlx_array{ q, scl, blocks } ++ arrays;
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var parts_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&parts_vec, kernel, inputs_vec, cfgs[0], s));
    if (mlx.mlx_vector_array_size(parts_vec) != 2) return error.MetalKernelBadOutputCount;
    var pacc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pacc);
    var pml = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pml);
    try mlx.check(mlx.mlx_vector_array_get(&pacc, parts_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&pml, parts_vec, 1));

    const merge_in = [_]mlx.mlx_array{ pacc, pml };
    const merge_vec = mlx.mlx_vector_array_new_data(&merge_in, merge_in.len);
    defer _ = mlx.mlx_vector_array_free(merge_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, merge_kernel, merge_vec, cfgs[1], s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

// ── QSA arm meters ──
//
// A single process-global `engaged` bool is not a meter: it fires on the
// first request of a session and then says nothing about the width or the
// context length that is actually slow, and a DECLINE says nothing at all —
// so a gather that silently hands every 62k decode step back to the dense
// mask (a full kv8 dequant per layer per step) reads exactly like a healthy
// run. These three pieces make the arm that ran observable: one-shot bits
// per (arm, width bucket) for engagement, one-shot bits per reason for
// declines, and a per-request tally.



/// A quantized KV cache as the kernel reads it: the packed triples plus the
/// shape of the dense view they stand for. `KVCache.denseView` already borrows
/// these; taking them here is what keeps its lazy dense arrays unevaluated.
pub const PackedKv = struct {
    k_q: mlx.mlx_array,
    k_scales: mlx.mlx_array,
    k_biases: mlx.mlx_array,
    v_q: mlx.mlx_array,
    v_scales: mlx.mlx_array,
    v_biases: mlx.mlx_array,
    bits: u8,
    group_size: u32,
};

/// One dispatch's keys or values: slice the packed triple along the token axis
/// and dequantize the slice. The slice is a VIEW and mlx's quantized ops want
/// their own rows, so it is materialized first (a few MB of packed bytes).
fn dequantKvSlice(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    k0: c_int,
    k1: c_int,
    group_size: u32,
    bits: u8,
) !mlx.mlx_array {
    var parts: [3]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null }, .{ .ctx = null } };
    defer for (&parts) |*a| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a.*);
    };
    for ([_]mlx.mlx_array{ q, scales, biases }, 0..) |src, i| {
        const sh = mlx.getShape(src);
        if (sh.len != 4) return error.InvalidKvSliceShape;
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(
            &view,
            src,
            &[_]c_int{ 0, 0, k0, 0 },
            4,
            &[_]c_int{ sh[0], sh[1], k1, sh[3] },
            4,
            &[_]c_int{ 1, 1, 1, 1 },
            4,
            s,
        ));
        parts[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_contiguous(&parts[i], view, false, s));
    }
    return kv_quant.dequantizeAffine(s, parts[0], parts[1], parts[2], group_size, bits);
}

pub fn fusedSdpaPrefillImpl(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    window: c_int,
    mask: ?mlx.mlx_array,
    packed_kv: ?PackedKv,
    sinks: mlx.mlx_array,
) !?mlx.mlx_array {
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    if (ks[3] != qs[3] or !fusedPdDims(qs[3], vs[3])) return null;
    const has_sinks = sinks.ctx != null;
    if (has_sinks and (mlx.mlx_array_ndim(sinks) != 1 or mlx.getShape(sinks)[0] != qs[1])) return null;
    // Short sequences (< 16: decode AND spec-decode VERIFY forwards) belong
    // to MLX's sdpa_vector, which covers hd 256 natively and beats a 64-row
    // prefill tile walking the whole KV for a few-row query — dispatching
    // those here measured decode 48 -> 18 tok/s at 4K ctx on the 27B (MTP
    // verify is seq 1+depth, up to 9 at the NAX depth-8 cap — with causal
    // fused default-on the floor must clear it). 16 matches oMLX's
    // _MIN_ROUTE_Q_LEN; genuine prefill chunks are always far above it.
    if (qs[2] < FUSED256_MIN_Q_LEN) return null;
    if (ks[1] <= 0 or @rem(qs[1], ks[1]) != 0) return null;
    if (ks[2] < qs[2] or ks[2] != vs[2] or ks[1] != vs[1] or ks[0] != qs[0] or vs[0] != qs[0]) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16 or mlx.mlx_array_dtype(k) != .bfloat16 or mlx.mlx_array_dtype(v) != .bfloat16) return null;
    if (packed_kv == null and (!attn256Base16Aligned(k) or !attn256Base16Aligned(v))) return null;

    const arm: AttnPdArm = if (attnPdNaxServes(qs[3], vs[3], mask != null)) .nax else .simd;
    var dispatches: u32 = 0;
    const out = (try attnPdDispatch(s, arm, q, k, v, scale, window, mask, packed_kv, sinks, &dispatches)) orelse return null;
    fused256_last_dispatch_count = dispatches;
    attn_pd_last_arm = arm;
    logAttnPdEngaged(arm, qs[3], vs[3], qs[2], ks[2], dispatches);
    return out;
}

/// The dispatch chain of one validated `fusedSdpaPrefillImpl` call on the given arm. Null = the
/// kernel could not be built.
fn attnPdDispatch(
    s: mlx.mlx_stream,
    arm: AttnPdArm,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    window: c_int,
    mask: ?mlx.mlx_array,
    packed_kv: ?PackedKv,
    sinks: mlx.mlx_array,
    dispatches: *u32,
) !?mlx.mlx_array {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    const has_sinks = sinks.ctx != null;
    const kernel = getAttnPdKernel(arm) catch return null;

    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const win_data = [_]i32{@intCast(window)};
    const win = mlx.mlx_array_new_data(&win_data, &one, 1, .int32);
    defer _ = mlx.mlx_array_free(win);

    // Budgeted kv-axis chunking (CAUSAL arm only — the band arm's in-kernel
    // block skip already bounds per-dispatch work by the window, so chunking
    // it would only add carry traffic). Chunk boundaries are BK(32)-aligned;
    // the fp32 m/l/O carry chained between dispatches keeps the result
    // bit-identical to a single dispatch (see the budget comment above).
    const kL: c_int = ks[2];
    const chunk_len: c_int = if (window > 0)
        kL
    else if (packed_kv != null)
        @min(PACKED_KV_SLICE_MAX, fused256KvChunkLen(@intCast(qs[0]), @intCast(qs[1]), @intCast(qs[2]), @intCast(kL), fused256DispatchBudget(arm)))
    else
        fused256KvChunkLen(@intCast(qs[0]), @intCast(qs[1]), @intCast(qs[2]), @intCast(kL), fused256DispatchBudget(arm));

    // Dummy 1-elem f32 stands in for absent carry inputs / unwritten outputs.
    const dummy_data = [_]f32{0};
    const dummy = mlx.mlx_array_new_data(&dummy_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(dummy);
    // 4-D so `mask_strides[..3]` exists for the QSA=0 template too.
    const dummy_b_data = [_]bool{false};
    const dummy_b_shape = [_]c_int{ 1, 1, 1, 1 };
    const dummy_b = mlx.mlx_array_new_data(&dummy_b_data, &dummy_b_shape, 4, .bool_);
    defer _ = mlx.mlx_array_free(dummy_b);
    var skip = mlx.mlx_array{ .ctx = null };
    defer if (skip.ctx != null) {
        _ = mlx.mlx_array_free(skip);
    };
    if (mask) |m| skip = try qsaSkipTable(s, m);
    var sinks_f32 = mlx.mlx_array{ .ctx = null };
    defer if (sinks_f32.ctx != null) {
        _ = mlx.mlx_array_free(sinks_f32);
    };
    if (has_sinks) {
        sinks_f32 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&sinks_f32, sinks, .float32, s));
    }

    var m_prev = mlx.mlx_array{ .ctx = null };
    var l_prev = mlx.mlx_array{ .ctx = null };
    var o_prev = mlx.mlx_array{ .ctx = null };
    defer {
        if (m_prev.ctx != null) _ = mlx.mlx_array_free(m_prev);
        if (l_prev.ctx != null) _ = mlx.mlx_array_free(l_prev);
        if (o_prev.ctx != null) _ = mlx.mlx_array_free(o_prev);
    }

    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    dispatches.* = 0;
    var k0: c_int = 0;
    while (k0 < kL) : (k0 += chunk_len) {
        const k1: c_int = @min(k0 + chunk_len, kL);
        const final = k1 == kL;
        const has_carry = k0 > 0;

        // {begin, end} in the BUFFER, then where the buffer starts in the
        // cache and how long the cache is. Dense: the buffer is the cache.
        const four = [_]c_int{4};
        const kr_data = [_]i32{
            if (packed_kv != null) 0 else @as(i32, @intCast(k0)),
            if (packed_kv != null) @as(i32, @intCast(k1 - k0)) else @as(i32, @intCast(k1)),
            if (packed_kv != null) @as(i32, @intCast(k0)) else 0,
            @intCast(kL),
        };
        const kr = mlx.mlx_array_new_data(&kr_data, &four, 1, .int32);
        defer _ = mlx.mlx_array_free(kr);

        // One dispatch's keys, dequantized here and freed as the graph runs
        // past them — the whole-prompt dense rebuild `denseView` would make
        // never exists (1.25 GiB per global layer at a 512k prompt).
        var k_slice: mlx.mlx_array = .{ .ctx = null };
        var v_slice: mlx.mlx_array = .{ .ctx = null };
        defer {
            if (k_slice.ctx != null) _ = mlx.mlx_array_free(k_slice);
            if (v_slice.ctx != null) _ = mlx.mlx_array_free(v_slice);
        }
        if (packed_kv) |pk| {
            k_slice = try dequantKvSlice(s, pk.k_q, pk.k_scales, pk.k_biases, k0, k1, pk.group_size, pk.bits);
            v_slice = try dequantKvSlice(s, pk.v_q, pk.v_scales, pk.v_biases, k0, k1, pk.group_size, pk.bits);
        }
        const phase_data = [_]i32{(if (has_carry) @as(i32, 1) else 0) | (if (final) @as(i32, 2) else 0)};
        const phase = mlx.mlx_array_new_data(&phase_data, &one, 1, .int32);
        defer _ = mlx.mlx_array_free(phase);

        const config = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        if (final) {
            const o_shape = [_]c_int{ qs[0], qs[1], qs[2], vs[3] };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 4, .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
        } else {
            const ml_shape = [_]c_int{ qs[0], qs[1], qs[2] };
            const oc_shape = [_]c_int{ qs[0], qs[1], qs[2], vs[3] };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &oc_shape, 4, .float32));
        }
        // One threadgroup per 64-row q tile per head per batch; grid is in
        // THREADS (dispatch_threads), padded to whole tiles so every
        // threadgroup is full (the SIMD kernel's staging loops assume NT=256).
        // SIMD: (32,8,1), 8 rows per simdgroup; NAX: (128,1,1), 16 rows per simdgroup.
        const nq_tiles: c_int = @divTrunc(qs[2] + 63, 64);
        switch (arm) {
            .simd => {
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, nq_tiles * 32, qs[1] * 8, qs[0]));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 8, 1));
            },
            .nax => {
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, nq_tiles * 128, qs[1], qs[0]));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1));
            },
        }
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BDK", qs[3]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BDV", vs[3]));
        if (arm == .simd) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "QSA", if (mask != null) 1 else 0));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "SINK", if (has_sinks) 1 else 0));

        const inputs_arr = [_]mlx.mlx_array{
            q,                                if (packed_kv != null) k_slice else k,
            if (packed_kv != null) v_slice else v, scl,
            win,                              kr,
            phase,                            if (has_carry) m_prev else dummy,
            if (has_carry) l_prev else dummy, if (has_carry) o_prev else dummy,
            if (mask) |m| m else dummy_b,     if (mask != null) skip else dummy_b,
            if (has_sinks) sinks_f32 else dummy,
        };
        const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
        defer _ = mlx.mlx_vector_array_free(inputs_vec);

        var outputs_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs_vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
        if (mlx.mlx_vector_array_size(outputs_vec) != 4) return error.MetalKernelBadOutputCount;
        dispatches.* += 1;

        if (final) {
            try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
        } else {
            var m_new = mlx.mlx_array_new();
            var l_new = mlx.mlx_array_new();
            var o_new = mlx.mlx_array_new();
            errdefer {
                if (m_new.ctx != null) _ = mlx.mlx_array_free(m_new);
                if (l_new.ctx != null) _ = mlx.mlx_array_free(l_new);
                if (o_new.ctx != null) _ = mlx.mlx_array_free(o_new);
            }
            try mlx.check(mlx.mlx_vector_array_get(&m_new, outputs_vec, 1));
            try mlx.check(mlx.mlx_vector_array_get(&l_new, outputs_vec, 2));
            try mlx.check(mlx.mlx_vector_array_get(&o_new, outputs_vec, 3));
            if (m_prev.ctx != null) _ = mlx.mlx_array_free(m_prev);
            if (l_prev.ctx != null) _ = mlx.mlx_array_free(l_prev);
            if (o_prev.ctx != null) _ = mlx.mlx_array_free(o_prev);
            m_prev = m_new;
            l_prev = l_new;
            o_prev = o_new;
            m_new = .{ .ctx = null };
            l_new = .{ .ctx = null };
            o_new = .{ .ctx = null };
        }
    }
    return out;
}

pub fn sliceAttentionSeq(s: mlx.mlx_stream, a: mlx.mlx_array, start: c_int, end: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(a);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, a, &.{ 0, 0, start, 0 }, 4, &.{ sh[0], sh[1], end, sh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    return out;
}

const model_mod = @import("model.zig");
const qwen4_config = @import("qwen4/qwen4_config.zig");
pub const log = @import("log.zig");
const io_util_mod = @import("io_util.zig");
const round_cost_mod = @import("round_cost.zig");
const group_cost_mod = @import("mtp_group_cost.zig");

pub const ModelConfig = model_mod.ModelConfig;
const QuantMode = model_mod.QuantMode;
pub const Weights = model_mod.Weights;

// ── KV Cache (standard attention) ──

// ── SSM Cache (for GatedDeltaNet linear attention layers) ──






extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "a streamed expert engine whose imatrix collector fails is deinit'd, not only freed" {
    try mlx.requireGpu();
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const raw = try expert_stream_mod.writeTinyExpertCheckpoint(t.allocator, tmp.dir, 4, 64, 64);
    defer t.allocator.free(raw);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const config = ModelConfig{ .model_type = "qwen4_exp", .num_hidden_layers = 1, .num_experts = 4, .hidden_size = 64, .moe_intermediate_size = 64, .expert_cache_bytes = 2 * 24576, .expert_source_dir = path_buf[0..path_len] };
    // Armed capture: the collector is the allocation that can fail after the engine exists.
    _ = setenv(imatrix_capture.ENV_VAR, "/nonexistent/imatrix.safetensors", 1);
    defer _ = unsetenv(imatrix_capture.ENV_VAR);
    var fail_index: usize = 0;
    while (true) : (fail_index += 1) {
        var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = fail_index });
        const parts = Transformer.initExpertStream(failing.allocator(), &config, mlx.gpuStream()) catch continue;
        try t.expect(parts.imatrix != null);
        parts.imatrix.?.deinit();
        parts.engine.deinit();
        failing.allocator().destroy(parts.engine);
        break;
    }
}

test "batched decode mask at verify width: row j of slot n sees its own causal prefix only" {
    try mlx.requireGpu();
    const t = std.testing;
    var xfm: Transformer = undefined;
    xfm.s = mlx.gpuStream();
    xfm.allocator = t.allocator;
    // Two slots, three verify rows each; kv_len is the length AFTER the rows were appended.
    const kv_lens = [_]i32{ 5, 8 };
    const mask = try xfm.buildBatchedDecodeMask(&kv_lens, 8, 3);
    defer _ = mlx.mlx_array_free(mask);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, mask, .float32, xfm.s));
    try mlx.check(mlx.mlx_array_eval(f));
    const d = mlx.mlx_array_data_float32(f) orelse return error.MlxArrayDataNull;
    const sh = mlx.getShape(f);
    try t.expectEqual(@as(c_int, 2), sh[0]);
    try t.expectEqual(@as(c_int, 3), sh[2]);
    try t.expectEqual(@as(c_int, 8), sh[3]);
    for (0..2) |n| for (0..3) |j| for (0..8) |pos| {
        const visible = @as(i32, @intCast(pos)) < kv_lens[n] - 3 + 1 + @as(i32, @intCast(j));
        const v = d[(n * 3 + j) * 8 + pos];
        if (visible) try t.expectEqual(@as(f32, 0.0), v) else try t.expect(v == -std.math.inf(f32));
    };
}

// ── Prompt Cache (snapshot of KV + SSM state for prefix reuse) ──

// ── Standard model per-layer weights ──

// ── Standard decoder-only layer weights ──

// ── MoE model per-layer weights ──

pub const FullAttnWeights = struct {
    q_w: mlx.mlx_array,
    q_s: mlx.mlx_array,
    q_b: mlx.mlx_array,
    k_w: mlx.mlx_array,
    k_s: mlx.mlx_array,
    k_b: mlx.mlx_array,
    v_w: mlx.mlx_array,
    v_s: mlx.mlx_array,
    v_b: mlx.mlx_array,
    o_w: mlx.mlx_array,
    o_s: mlx.mlx_array,
    o_b: mlx.mlx_array,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    // qwen4_exp QSA indexer: one projection for the index queries + the
    // single shared key head, and their per-head RMS norms. Null-ctx on
    // every other arch.
    idx_qk_w: mlx.mlx_array = .{ .ctx = null },
    idx_qk_s: mlx.mlx_array = .{ .ctx = null },
    idx_qk_b: mlx.mlx_array = .{ .ctx = null },
    idx_q_norm: mlx.mlx_array = .{ .ctx = null },
    idx_k_norm: mlx.mlx_array = .{ .ctx = null },
    // MiMo: learned per-head attention sinks (`self_attn.attention_sink_bias`,
    // [n_heads]). One extra column in the softmax DENOMINATOR, letting a head
    // attend to "nothing". Null-ctx on every other arch.
    sinks: mlx.mlx_array = .{ .ctx = null },
};

/// qwen4_exp gated residual ("hyper connection"). `norm_w` is the grouped
/// hc_norm weight viewed as [hc, hidden]; `inject_*` is absent on the final
/// mixer (`use_combine=False`).
pub const HcWeights = struct {
    norm_w: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    inject_w: mlx.mlx_array = .{ .ctx = null },
    inject_s: mlx.mlx_array = .{ .ctx = null },
    inject_b: mlx.mlx_array = .{ .ctx = null },
    /// Dense inject weight materialized row-major `[hc*hidden, hc]` for the
    /// fused decode kernel (the qmatmul path reads the lazy transpose view).
    inject_flat: mlx.mlx_array = .{ .ctx = null },
};

/// qwen4_exp PLE: n-gram embedding → per-stream keys + one value, gated by
/// the normalized streams, plus a dilated depthwise short conv.
pub const PleWeights = struct {
    key_w: mlx.mlx_array,
    key_s: mlx.mlx_array,
    key_b: mlx.mlx_array,
    value_w: mlx.mlx_array,
    value_s: mlx.mlx_array,
    value_b: mlx.mlx_array,
    norm_key: mlx.mlx_array, // [hc, hidden]
    norm_query: mlx.mlx_array,
    norm_conv: mlx.mlx_array,
    conv_w: mlx.mlx_array, // [hc*hidden, K, 1]
};


pub const LinearAttnWeights = struct {
    // Separate projections: qkv=QKV, z=Z, a=A, b=B
    qkv_w: mlx.mlx_array,
    qkv_s: mlx.mlx_array,
    qkv_b: mlx.mlx_array,
    z_w: mlx.mlx_array,
    z_s: mlx.mlx_array,
    z_b: mlx.mlx_array,
    a_w: mlx.mlx_array,
    a_s: mlx.mlx_array,
    a_b: mlx.mlx_array,
    b_w: mlx.mlx_array,
    b_s: mlx.mlx_array,
    b_b: mlx.mlx_array,
    conv1d_w: mlx.mlx_array,
    A_log: mlx.mlx_array,
    dt_bias: mlx.mlx_array,
    norm_w: mlx.mlx_array,
    out_w: mlx.mlx_array,
    out_s: mlx.mlx_array,
    out_b: mlx.mlx_array,
};

pub const DenseMlpWeights = struct {
    gate_w: mlx.mlx_array,
    gate_s: mlx.mlx_array,
    gate_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
};

pub const MoeMlpWeights = struct {
    /// A rate-grouped EXL3 layer's banks; its handles are the transformer's to free.
    exl3_ragged: ?sushi_exl3.Ragged = null,
    router_w: mlx.mlx_array,
    router_s: mlx.mlx_array,
    router_b: mlx.mlx_array,
    switch_gate_w: mlx.mlx_array,
    switch_gate_s: mlx.mlx_array,
    switch_gate_b: mlx.mlx_array,
    switch_up_w: mlx.mlx_array,
    switch_up_s: mlx.mlx_array,
    switch_up_b: mlx.mlx_array,
    switch_down_w: mlx.mlx_array,
    switch_down_s: mlx.mlx_array,
    switch_down_b: mlx.mlx_array,
    shared_gate_w: mlx.mlx_array,
    shared_gate_s: mlx.mlx_array,
    shared_gate_b: mlx.mlx_array,
    shared_up_w: mlx.mlx_array,
    shared_up_s: mlx.mlx_array,
    shared_up_b: mlx.mlx_array,
    shared_down_w: mlx.mlx_array,
    shared_down_s: mlx.mlx_array,
    shared_down_b: mlx.mlx_array,
    // The dense shared-expert gate as one contiguous [hidden] vector (null ctx otherwise), for the router kernel's dot.
    shared_gate_vec: mlx.mlx_array = .{ .ctx = null },
    // Shared expert gating (qwen4_exp; null for MiMo)
    shared_expert_gate_w: ?mlx.mlx_array = null,
    shared_expert_gate_s: ?mlx.mlx_array = null,
    shared_expert_gate_b: ?mlx.mlx_array = null,
    // MiMo sigmoid routing: f32 [num_experts] bias added to the sigmoid scores for
    // top-k SELECTION only (weights come from the unbiased scores). Non-null
    // expert_bias routes moeMLP2 through the sigmoid chain.
    expert_bias: ?mlx.mlx_array = null,
    route_norm: bool = true,
    route_scale: f32 = 1.0,
};

const HybridMlpWeights = union(enum) {
    dense: DenseMlpWeights,
    moe: MoeMlpWeights,
};

pub const MoeLayerWeights = struct {
    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,
    is_linear: bool,
    attn: union(enum) { full: FullAttnWeights, linear: LinearAttnWeights },
    mlp: HybridMlpWeights,
    // qwen4_exp: the two gated residual reads/writes around the block and
    // the n-gram PLE on its one layer. Null for every other arch.
    hc_attn: ?HcWeights = null,
    hc_mlp: ?HcWeights = null,
    ple: ?PleWeights = null,
};


// ── Quantization params cache ──
// A tiny lock-free pointer → (bits, group_size) cache. Quantized weights are loaded
// once at init and reused for every forward pass; we detect (or pre-bind) params on
// first touch and serve hits thereafter. Uses open addressing with linear probing on
// a fixed-size array — this fits in L1 and keeps the cost of a lookup to ~5ns,
// matching the perf commit's intent of "eliminate per-call detect overhead" while
// supporting mixed-precision quantization (per-layer bits).
const BITS_CACHE_CAP: usize = 1024; // plenty for 60 layers × ~10 quant weights × factor
pub const QuantParams = struct { bits: u32, group_size: u32, mode: QuantMode = .affine };

const QuantParamsCache = struct {
    keys: [BITS_CACHE_CAP]?*anyopaque = @splat(null),
    vals_bits: [BITS_CACHE_CAP]u8 = @splat(0),
    // group_size is always a small power of two in MLX (16, 32, 64, 128).
    // Store as group_size/8 to fit u8 with headroom up to 2040.
    vals_gs_div8: [BITS_CACHE_CAP]u8 = @splat(0),
    vals_mode: [BITS_CACHE_CAP]u8 = @splat(0),

    inline fn slot(key: *anyopaque) usize {
        const h: usize = @intFromPtr(key);
        // Golden-ratio multiplier for quick hash on pointer values (high bits).
        return (h *% 0x9E3779B97F4A7C15) >> @as(u6, @intCast(@bitSizeOf(usize) - 10));
    }

    /// Insert params for `key`. Returns false if the probe window saturated
    /// (caller should treat as "fall through to detection" — but realistically
    /// never fires given BITS_CACHE_CAP is 1024 vs ~2k weights max).
    fn put(self: *QuantParamsCache, key: *anyopaque, qp: QuantParams) bool {
        const start = slot(key);
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            const idx = (start + i) & (BITS_CACHE_CAP - 1);
            if (self.keys[idx] == null or self.keys[idx] == key) {
                self.keys[idx] = key;
                self.vals_bits[idx] = @intCast(qp.bits);
                self.vals_gs_div8[idx] = @intCast(qp.group_size / 8);
                self.vals_mode[idx] = @backingInt(qp.mode);
                return true;
            }
        }
        return false;
    }
};

// Backwards-compatibility alias — keeps existing field names readable.
const BitsCache = QuantParamsCache;

// ── Forward context ──
//
// Routes per-request mutable state (KV cache, MoE seq offset, SSM entries,
// hidden-state capture target, vision embeddings) into the forward pass via
// a single struct. The legacy single-slot path uses `Transformer.defaultCtx()`
// which points at the Transformer's own fields — semantically identical to the
// pre-refactor code. Concurrent batching (Phase 1+) constructs one ctx per
// in-flight request so multiple slots can share a Transformer's weights while
// owning their own caches.
/// DFlash capture request: intermediate layer OUTPUTS, `hidden_states[i+1]`
/// semantics (the residual stream AFTER layer `ids[k]` completes, before the
/// next layer's input norm). Standard forward path only (v1).
pub const CaptureLayers = struct {
    /// Trunk layer indices to capture, ascending.
    ids: []const u32,
    /// Filled per forward: `out[k]` = h after layer `ids[k]`, ALL positions
    /// `[B, L, H]`, refcount-shared with the live graph. Caller owns the
    /// handles (init with `mlx_array_new`, free after use).
    out: []mlx.mlx_array,
    /// The residual entering the first layer (`hidden_states[0]`), same
    /// ownership as `out`. Honored by `forwardMoeWith` only.
    input: ?*mlx.mlx_array = null,
    /// qwen4_exp: average the layer output over this many hyper-connection
    /// streams, giving `[B, L, H]` taps (0 = the raw `[B, L, hc*H]` stream).
    hc_mean: u32 = 0,
};

pub const ForwardCtx = struct {
    cache: *KVCache,
    moe_seq_offset: *usize,
    ssm_entries: ?[]SSMCacheEntry,
    ssm_member_gen: u64 = 0,
    /// glm5_next: the request's native KDA/MLA state, owned by the slot (the default
    /// context names the shell's own). A GLM forward without one is refused by name.
    glm5_request: ?*GlmRequest = null,
    capture_hidden: ?*mlx.mlx_array,
    /// Like `capture_hidden` but receives the FULL post-final-norm hidden
    /// `[B, L, H]` (all positions, refcount-shared) instead of the last
    /// position only. Used by the Qwen MTP head, whose committed-history
    /// cache needs the trunk hidden at every verify/prefill position.
    capture_hidden_all: ?*mlx.mlx_array = null,
    /// qwen4_exp: receives the PRE-mixer hyper-connection stream `[B, L,
    /// hc*hidden]` (all positions) — what the MTP head consumes.
    capture_stream_all: ?*mlx.mlx_array = null,
    /// DFlash: capture the residual stream at these layer OUTPUTS (all
    /// positions). Honored by `forwardStandardWith` only — DflashModel.bind
    /// refuses non-standard-path targets so this can never be silently
    /// ignored on an engaged path.
    capture_layers: ?*CaptureLayers = null,
    /// PLD spec-decode: when true, the GatedDeltaNet trunk records per-position
    /// SSM/conv state during the verify forward (see `SSMCacheEntry.spec_*`),
    /// so partial-accept rollback needs no re-forward. Set only by `nextPld`
    /// for the verify pass on a GDN model; the flag reaches the layers via
    /// `Transformer.spec_capture_ssm`.
    capture_ssm_seq: bool = false,
    /// MiMo spec verify: every row of this multi-row forward computes with its
    /// own decode tick's arithmetic (`mimoVerifyRowsAttn`), so greedy MTP is serial.
    verify_rows: bool = false,
    vision_embeddings: ?mlx.mlx_array,
    /// Chunked vision prefill (issue #197): number of image/audio placeholder
    /// tokens already spliced by earlier chunks of this prefill. The splice's
    /// row lookup is a cumsum over the CURRENT forward's placeholder mask, so
    /// without this every chunk would restart at vision row 0. Set per chunk
    /// by the generate.zig prefill loop; 0 for unchunked/decode forwards.
    vision_splice_offset: usize = 0,
    /// DeepSeek-V4.1 chunked prefill: the prompt tokens after this chunk, whose Engram rows the model reads while the
    /// chunk runs. Empty for the last chunk and for decode.
    ahead_ids: []const i32 = &.{},
    /// Qwen3-VL interleaved M-RoPE. `mrope_pos` is the server-computed flat
    /// [3 × mrope_total] i32 position-id table (axis-major: t,h,w rows), borrowed
    /// from the slot. `mrope_delta` shifts the scalar decode RoPE (decode tokens
    /// are text → t=h=w). `mrope_cos/sin_cur` are the per-prefill-chunk cos/sin
    /// tables, built once in forwardMoeWith and consumed by every full-attn layer
    /// that chunk, then freed. All null/0 for non-Qwen / text-only requests.
    mrope_pos: ?[]const i32 = null,
    mrope_total: usize = 0,
    mrope_delta: i32 = 0,
    mrope_cos_cur: ?mlx.mlx_array = null,
    mrope_sin_cur: ?mlx.mlx_array = null,
    /// qwen4_exp QSA: the bool `[B, 1, S, kv]` mask (indexer-selected blocks
    /// ∧ causal) `gatedFullAttnWith` must apply this layer. Null-ctx = dense.
    qsa_mask: mlx.mlx_array = .{ .ctx = null },
    /// qwen4_exp QSA: the sorted `[B, S, kb]` int32 block selection (INT_MAX
    /// past each row's count) for the three gather arms — `gatherQsa256`
    /// (prefill), `qsaVerifyGatherAttn` (S = 2..15) and `qsaDecodeGatherAttn`
    /// (S == 1); set INSTEAD of `qsa_mask`. Null-ctx = not selected.
    qsa_blocks: mlx.mlx_array = .{ .ctx = null },
    /// Which QSA arm served each full-attention call of THIS request. The
    /// `[qsa-arms]` line is a per-request meter, so the tally is owned by the
    /// seam that logs it (`Generator.logQsaArms` over the Generator's own
    /// ctx) — a Transformer-level counter merged every concurrent slot's
    /// calls (batched decode + interleaved prefill) into whichever request
    /// finished first. A batched forward attributes each call to every
    /// participating slot's own ctx (`noteQsaArmForSlots`).
    ///
    /// Scoped to the TRUNK: the qwen4 MTP head's forward builds its own ctx
    /// and its single QSA layer is not part of this tally.
    qsa_arms: QsaArmCounts = .{},
    /// Return the post-final-norm hidden instead of logits (caller frees
    /// either way), skipping the lm_head projection.
    skip_lm_head: bool = false,
    stream_in: ?mlx.mlx_array = null,
    skip_ple: bool = false,
    /// Set by the Generator when this request consumes ONLY the argmax of
    /// the logits (greedy or top-1 sampling, no logit-modifying penalties,
    /// no per-token logprobs, no grammar mask). Gates the certified lm_head
    /// prune, whose output row is bit-identical at every candidate slot and
    /// certified-below-the-winner elsewhere — exactly enough for an argmax,
    /// not for a tail distribution. Default false = dense path everywhere.
    argmax_only: bool = false,
    /// Batched decode over a GatedDeltaNet trunk (`forwardMoeBatchedDecode`).
    /// Non-null means: this forward runs at `batch == batch_slots.len`,
    /// `seq_len == 1`, and every per-slot KV cache lives on the slot ctxs
    /// here — NOT on `ctx.cache`, which is only a shape donor in this mode.
    /// The GDN layers need no branch (their state is merged into this ctx's
    /// `ssm_entries` at B=N and split back out by the driver); only the
    /// full-attention layers do, because K/V lengths differ per slot and
    /// each slot ropes at its own offset.
    ///
    /// `batch_rope_offsets` is the matching int32 `[N]` offset array for
    /// `mlx_fast_rope_dynamic` — already carrying each slot's M-RoPE delta,
    /// so the attention layer never re-derives a per-slot offset.
    batch_slots: ?[]const *ForwardCtx = null,
    batch_rope_offsets: ?mlx.mlx_array = null,
    /// MiMo batched decode (`forwardMimoBatchedDecode`): row i of this `[1, N]` forward is
    /// slot i's decode tick, its attention and KV append on `batch_rows[i]`'s cache at
    /// `batch_row_offsets[i]`; every other op already computes rows independently.
    batch_rows: ?[]const *ForwardCtx = null,
    batch_row_offsets: []const u32 = &.{},
    /// MTP head: one QSA entry, not `ssm_entries[layer]`. Null on the trunk.
    qsa_entry: ?*SSMCacheEntry = null,
    /// Absolute position of this slot's QSA key row 0 (0 on the trunk).
    qsa_pos_base: c_int = 0,
    /// qwen4_exp pipelined decode: the n-gram PLE rows are a HOST gather keyed
    /// on the token ids, and the pipelined step builds its graph on a token
    /// that is still a lazy sample. With `ple_defer` set, `pleEmbedding` hands
    /// the graph an unfilled leaf and records it here; the caller fills it via
    /// `Transformer.flushDeferredPle` (syncing on the token THEN) right before
    /// the graph is evaluated. Set by `generate.lazyForward` (decode) and by
    /// the MTP verify build in `Generator.nextMtp` (whose `[1, 1+m]` input is
    /// the still-lazy draft chain); a forward evaluated with `ple_pending`
    /// still set reads zero rows.
    ple_defer: bool = false,
    ple_pending: ?PlePending = null,
};





/// The additive `[1, 1, 1, kv_len]` bf16 mask a sliding decode row applies.
pub fn slidingDecodeMask(s: mlx.mlx_stream, kv_len: c_int, window: c_int) !mlx.mlx_array {
    var positions = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(positions);
    try mlx.check(mlx.mlx_arange(&positions, 0, @floatFromInt(kv_len), 1, .int32, s));
    const window_start = mlx.mlx_array_new_int(kv_len - window);
    defer _ = mlx.mlx_array_free(window_start);
    var too_old = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(too_old);
    try mlx.check(mlx.mlx_less(&too_old, positions, window_start, s));
    const zero = bf16Scalar(0.0, s);
    defer _ = mlx.mlx_array_free(zero);
    const neg_inf = bf16Scalar(-std.math.inf(f32), s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var sw_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sw_mask);
    try mlx.check(mlx.mlx_where(&sw_mask, too_old, neg_inf, zero, s));
    const mask_shape = [_]c_int{ 1, 1, 1, kv_len };
    var mask_4d = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&mask_4d, sw_mask, &mask_shape, 4, s));
    return mask_4d;
}














// ── Transformer ──

pub const Transformer = struct {
    config: ModelConfig,
    cache: KVCache,
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,

    emb_w: mlx.mlx_array,
    emb_s: mlx.mlx_array,
    emb_b: mlx.mlx_array,
    emb_scale: ?mlx.mlx_array,
    final_norm: mlx.mlx_array,
    lm_head_w: mlx.mlx_array,
    lm_head_s: mlx.mlx_array,
    lm_head_b: mlx.mlx_array,


    owns_lm_head: bool,
    owns_norms: bool,

    /// Layer stride for the decode async-eval ladder (0 = off). Resolved once
    /// at construction from `SUSHI_DECODE_ASYNC_LADDER` — a per-layer
    /// getenv would cost more than the overlap it buys.
    decode_async_ladder: u32 = 0,
    /// Batched Qwen4 stride; serial decode uses decode_async_ladder.
    decode_async_ladder_qwen4: u32 = 0,

    gelu_coeff: ?mlx.mlx_array,
    gelu_inner: ?mlx.mlx_array,
    half: mlx.mlx_array,
    one: mlx.mlx_array,
    /// `rms_norm_eps` as a 0-dim f32 array, for the fused residual+RMSNorm
    /// kernel (built once — an eps array per layer per token is exactly the
    /// per-call allocation this fusion exists to remove).
    rms_eps_arr: mlx.mlx_array = .{ .ctx = null },
    /// Reserved-token suppression mask (`[vocab]` bool, true = never
    /// sample), built once at load by `server`'s install step from the
    /// tokenizer's `special: true` set minus template-emitted markers and
    /// EOS ids (`tokenizer.reservedOutputIds`). Owned here, freed in deinit;
    /// samplers reference it non-owning via `SamplingParams.suppress_mask`.
    /// Null = suppression off (kill switch, no chat template).
    suppress_mask: ?mlx.mlx_array = null,
    /// Overthinking-marker mask (`[vocab]` bool, `think_penalty.markerIds`), built once at load;
    /// owned here, read by the Generator's think penalty. Null = no penalty can apply.
    think_marker_mask: ?mlx.mlx_array = null,
    logit_bias: []const @import("logit_bias.zig").Bias = &.{},
    /// Measured spec round-cost table (`round_cost.Table`): per draft
    /// width, per KV bucket, fed by every MTP/DFlash round on this model and
    /// read by the EV plan in place of the fitted surface once a bucket has
    /// two measured widths. Lives on the model, not the request — a request
    /// spans only its own max_tokens. Inference thread only.
    round_cost: round_cost_mod.Table = .{},
    mtp_group_cost: group_cost_mod.Table = .{},
    cost_trace_active: bool = false,
    cost_attention: u8 = 0,
    cost_moe: u8 = 0,
    verify_laps: VerifyLaps = .{},
    /// The joined `[1, Σ(1+m_i), V]` logits of the last row-axis verify whose
    /// lm_head ran once for the whole group, so the accept side can filter the
    /// block once instead of one slice per row. Null whenever the projection
    /// ran per row; the next verify releases it.
    verify_joined_logits: mlx.mlx_array = .{ .ctx = null },
    /// Persistence key of `round_cost` (empty = not persisted).
    round_cost_key_buf: [64]u8 = undefined,
    round_cost_key_len: u8 = 0,
    /// MTP depth cap WITHOUT the per-silicon row (explicit --mtp-max-depth, else
    /// the adaptive default), set at load. The row-resolved cap travels as
    /// `mtp_depth` through server -> slot -> Generator and reads there like
    /// an explicit flag, so the row-less value needs its own carrier.
    mtp_depth_free: u32 = 0,
    /// A coarse copy of this lm_head (MiMo, resident): an argmax-only forward shortlists its top-32
    /// on it and re-scores them through the full head. The MTP heads draft on the same copy, so a
    /// load with MTP and one without read the same greedy argmax.
    lm_head_coarse: ?mtp_mod.RerankCoarse = null,
    lm_head_coarse_logged: bool = false,
    /// Per-token cos|sin rows for the fused QK-norm+RoPE kernel, one slot per
    /// rope family (0 = default/sliding base, 1 = YaRN freqs). Keyed by
    /// offset: every layer of one decode step shares the row, the next token
    /// replaces it.
    qk_angle_cache: [2]QkAngleSlot = .{ .{}, .{} },
    three: ?mlx.mlx_array,
    neg_one: ?mlx.mlx_array,

    ones_hidden: ?mlx.mlx_array = null,



    // YaRN RoPE scaling. `rope_freqs_yarn` is the per-dim RoPE DENOMINATOR
    // array handed to mlx_fast_rope (angle = pos/freqs), `yarn_mscale` the
    // [head_dim] vector carrying attention_factor on the rotated dims (1.0 on
    // the pass-through tail), and `yarn_inv_freq` the same spectrum as the
    // HOST inverse frequencies the M-RoPE cos/sin tables and the QSA pooled-key
    // ladder are filled from (angle = pos×inv_freq). All three null on
    // unscaled checkpoints.
    rope_freqs_yarn: ?mlx.mlx_array = null,
    yarn_mscale: ?mlx.mlx_array = null,
    yarn_inv_freq: ?[]const f64 = null,
    // `yarn_mscale` narrowed to the activation dtype, built on first use. See
    // `constTableAs` — the f32 table is what a bf16 forward must NOT multiply
    // by directly.
    yarn_mscale_cast: DtypeCastCache = .{},

    glm5: ?*glm5_mod.Model = null,
    glm5_request: ?glm5_mod.Request = null,

    // deepseek_v41: the model owns its one request's state (single flight), its routed experts
    // stream through `dsv41_experts` over `expert_stream`.
    dsv41: ?*dsv41_model.Model = null,
    dsv41_experts: ?*dsv41_stream.Stream = null,
    dsv41_stq_experts: ?*dsv41_stream.StqStream = null,

    // Qwen3.8-Flash-Next (qwen4_exp): the n-gram hash + mmapped table are
    // module-owned (serial, spec-off); the trunk itself rides moe_layers
    // through forwardQwen4With. `qwen4_mixer` is the final hyper-connection
    // read that replaces model.norm.
    qwen4: ?*qwen4_mod.Qwen4State = null,
    /// Run the qwen4 residual streams in f32 (QWEN4_STREAM_F32=1, or set by
    /// the oracle test): the shipped checkpoint is bf16 end to end, the f32
    /// fixture is the tighter bar for the MATH.
    qwen4_stream_f32: bool = false,
    qwen4_mixer: ?HcWeights = null,
    /// nb-keyed QSA block constants (`QsaBlockConsts`).
    qsa_consts: QsaBlockConsts = .{},
    /// QSA pooled-block cos/sin, built once per forward (`QsaPooledRope`).
    qsa_pooled_rope: QsaPooledRope = .{},
    /// Bumped by every forward entry that can reach QSA; keys the per-forward QSA scratch.
    fwd_gen: u64 = 0,
    /// Test meter: how many times `qsaScoreBank` built or extended the transposed f32 operand.
    qsa_score_bank_builds: usize = 0,
    /// qwen4_exp MTP head (one hyper-connected QSA+MoE layer over the trunk's
    /// pre-mixer stream + next-token embedding). Loaded when the pack carries
    /// `mtp.*`; served by `qwen4MtpForward`. Not yet wired into spec decode.
    qwen4_mtp: ?Qwen4Mtp = null,
    /// Whose per-request state the head currently holds (`qwen4MtpActivate`).
    qwen4_mtp_owner: ?*Qwen4MtpState = null,
    expert_stream: ?*expert_stream_mod.Engine = null,
    expert_bf16_kernels: bool = true,
    imatrix: ?*imatrix_capture.Collector = null,


    // MoE-specific (null/empty for standard models)
    moe_layers: ?[]MoeLayerWeights,
    ssm_entries: ?[]SSMCacheEntry,
    moe_seq_offset: usize,
    // Pre-transposed plain-bf16 linear_attn weights owned by the Transformer
    // (Unsloth Dynamic checkpoints — null for the common all-quantized case).
    moe_owned_bf16: ?[]mlx.mlx_array = null,


    // Vision: set before prefill when images are present, cleared after.
    // Shape: [B, num_image_tokens, hidden_size]. Spliced at image_token_id positions.
    vision_embeddings: ?mlx.mlx_array = null,

    // Compiled closures (fuse ops into single kernels, matching mlx-lm's @mx.compile)
    compiled_gelu: ?mlx.mlx_closure = null,
    compiled_geglu: ?mlx.mlx_closure = null, // gelu(gate) * up → 1 kernel
    compiled_moe_routing: ?mlx.mlx_closure = null, // negate→argpartition→slice→softmax→take→sum→expand→divide → 1 kernel
    compiled_gdn_gate: ?mlx.mlx_closure = null, // exp(-exp(A_log)·softplus(a+dt_bias)) → 1 kernel (mirrors mlx-lm compute_g)
    // qwen4_exp hyper-connection elementwise tails, one kernel each (the
    // `/hc` scalars are folded into the weights at load): silu(down),
    // mean(σ(up)·normed), 2σ(inject), stream + out·inj.
    compiled_hc_silu: ?mlx.mlx_closure = null,
    compiled_hc_mix: ?mlx.mlx_closure = null,
    compiled_hc_inj: ?mlx.mlx_closure = null,
    compiled_hc_write: ?mlx.mlx_closure = null,

    // GatedDeltaNet per-token decode used to rebuild these on EVERY layer/step
    // (measured dispatch overhead on Qwen 3.6 hybrid). Built lazily on first
    // use, freed in deinit.
    gdn_ones_w: ?mlx.mlx_array = null, // ones([dk]) for parameter-free rms_norm
    gdn_q_scale: ?mlx.mlx_array = null, // bf16 scalar 1/dk
    gdn_k_scale: ?mlx.mlx_array = null, // bf16 scalar 1/sqrt(dk)
    gdn_eps: ?mlx.mlx_array = null, // f32 scalar rms_norm_eps for the fused norm-gate kernel
    /// PLD spec-decode: mirrors `ForwardCtx.capture_ssm_seq` for the current
    /// forward so `gatedDeltaNet`/`conv1dWithCache` (which don't take the ctx)
    /// can capture per-position state. Set+reset only inside `forwardMoeWith`.
    spec_capture_ssm: bool = false,
    /// Test seam only (no env): `false` runs the old per-tick merge/split path so the
    /// transition tests can compare bytes. The persistent group is an exact change.
    persistent_group_state_override: ?bool = null,
    /// Test seam only (no env): `false` runs the old per-slot KV update for byte comparisons.
    batched_kv_append_override: ?bool = null,
    ssm_group: PersistentSsmGroup = .{},

    // Per-weight quantization bit cache (see bitsFor). Populated lazily on first use.
    // Keyed by the scales array's ctx pointer (stable for the lifetime of a weight).
    // Used instead of config.quant_bits so mixed-precision models
    // with per-layer overrides work correctly while keeping zero per-call FFI overhead
    // after the first touch.
    bits_cache: BitsCache = .{},

    pub fn init(io: std.Io, allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !Transformer {
        // Use the current thread's default GPU stream rather than a dedicated stream.
        // mlx 0.31.2 made streams thread-local — a stream created on one thread isn't
        // visible to other threads, so a long-lived dedicated stream stored on Transformer
        // would break as soon as a different thread (e.g. an HTTP connection handler)
        // tried to use it. We re-bind `self.s` to the connection thread's default stream
        // via `useCurrentThreadStream` before each request.
        const s = mlx.gpuStream();
        if (config.indexer_budget != 0) {
            qsaScoreFusedArm();
            qsaNaxArm();
        }
        fp8_block.tile_nax = verifyQmmNaxAvailable();
        const prefix = config.weight_prefix;

        var name_buf: [256]u8 = undefined;

        if (config.isGlm5()) return initGlm5(allocator, config, weights, s);
        if (config.isDsv41()) return initDsv41(allocator, config, weights, s);

        const emb_base: []const u8 = "embed_tokens";
        const emb_w = getNamedWeight(weights, &name_buf, prefix, emb_base, "weight") orelse {
            log.err("MISSING WEIGHT: {s}.{s}.weight\n", .{ prefix, emb_base });
            return error.MissingWeight;
        };
        // Dense bf16 (quant_bits==0): no scales/biases exist → null-ctx arrays
        // signal "plain bf16" to embedding()/dequantTake().
        // Bias-less quant modes (mxfp4/mxfp8): scales exist, biases are
        // OPTIONAL — absent on fp8 tensors, present on affine-override tensors
        // in mixed QAT checkpoints. A mandatory fetch would be a spurious
        // MISSING WEIGHT crash (issue #24).
        // Mixed checkpoints can also leave the WHOLE embedding table dense
        // beside quantized layers (a 2-bit pack can ship a bf16 embed_tokens) —
        // scales-presence is decided by the TABLE's dtype (float ⇒ dense),
        // never by the config's global bits. A packed (uint32) table missing
        // its scales still crashes honestly.
        const bias_mandatory = config.quant_bits > 0 and config.quant_mode.hasBiases();
        const emb_dense = floatDtypeTable(mlx.mlx_array_dtype(emb_w));
        const emb_s_arr = if (emb_dense)
            mlx.mlx_array_new()
        else
            getNamedWeight(weights, &name_buf, prefix, emb_base, "scales") orelse {
                log.err("MISSING WEIGHT: {s}.{s}.scales\n", .{ prefix, emb_base });
                return error.MissingWeight;
            };
        const emb_b_arr = if (emb_dense)
            mlx.mlx_array_new()
        else
            getNamedWeight(weights, &name_buf, prefix, emb_base, "biases") orelse blk: {
                if (bias_mandatory) {
                    log.err("MISSING WEIGHT: {s}.{s}.biases\n", .{ prefix, emb_base });
                    return error.MissingWeight;
                }
                break :blk mlx.mlx_array_new();
            };

        const emb_scale: ?mlx.mlx_array = if (config.scale_embeddings)
            bf16Scalar(@sqrt(@as(f32, @floatFromInt(config.hidden_size))), s)
        else
            null;

        var final_norm: mlx.mlx_array = undefined;
        if (!config.has_final_norm) {
            final_norm = mlx.mlx_array_new(); // placeholder, unused
        } else {
            const final_norm_raw = try getWeightFmt(weights, &name_buf, "{s}.norm.weight", prefix);
            if (config.norm_has_offset) {
                final_norm = try addOne(final_norm_raw, s);
                try mlx.check(mlx.mlx_array_eval(final_norm));
            } else {
                final_norm = final_norm_raw;
            }
        }

        var lm_head_w: mlx.mlx_array = undefined;
        var lm_head_s: mlx.mlx_array = undefined;
        var lm_head_b: mlx.mlx_array = undefined;
        var owns_lm_head = false;

        {
            // lm_head prefix: "language_model.model" -> "language_model", "model" -> try root, else -> prefix
            const lm_prefix = if (std.mem.eql(u8, prefix, "language_model.model")) "language_model" else prefix;
            const maybe_lm_w = getWeightFmtOpt(weights, &name_buf, "{s}.lm_head.weight", lm_prefix);
            if (maybe_lm_w) |w| {
                lm_head_w = w;
                // Dense bf16: no scales/biases → null-ctx; lmHeadProject() then
                // projects via a transposed view of the [vocab, hidden] weight.
                // Per-TENSOR dense detection (float dtype), same as embed_tokens
                // above — mixed checkpoints may quantize layers but not the head.
                const head_dense = config.quant_bits == 0 or floatDtypeTable(mlx.mlx_array_dtype(w));
                lm_head_s = if (head_dense)
                    mlx.mlx_array_new()
                else
                    try getWeightFmt(weights, &name_buf, "{s}.lm_head.scales", lm_prefix);
                lm_head_b = if (head_dense)
                    mlx.mlx_array_new()
                else if (bias_mandatory)
                    try getWeightFmt(weights, &name_buf, "{s}.lm_head.biases", lm_prefix)
                else
                    getWeightFmtOpt(weights, &name_buf, "{s}.lm_head.biases", lm_prefix) orelse mlx.mlx_array_new();
                owns_lm_head = !config.tie_word_embeddings;
            } else if (weights.get("lm_head.weight")) |w| {
                lm_head_w = w;
                // A float head is dense even beside a packed embedding.
                const head_dense = floatDtypeTable(mlx.mlx_array_dtype(w));
                lm_head_s = if (head_dense) mlx.mlx_array_new() else weights.get("lm_head.scales") orelse emb_s_arr;
                lm_head_b = if (head_dense) mlx.mlx_array_new() else weights.get("lm_head.biases") orelse emb_b_arr;
                owns_lm_head = !config.tie_word_embeddings;
            } else if (config.tie_word_embeddings) {
                lm_head_w = emb_w;
                lm_head_s = emb_s_arr;
                lm_head_b = emb_b_arr;
            } else {
                log.err("MISSING WEIGHT: lm_head.weight\n", .{});
                return error.MissingWeight;
            }
        }

        // Cache for KV (standard models use all entries, MoE only uses full-attn layers)
        var cache = try KVCache.init(allocator, config.num_hidden_layers);
        errdefer cache.deinit();
        if (config.swaRingTokens() > 0) cache.setSwaRing(config.sliding_window);

        const need_gelu = config.hidden_act == .gelu_approx;
        const need_silu = config.hidden_act == .silu;

        var ssm_entries: ?[]SSMCacheEntry = null;
        const loaded = try initMoeLayers(allocator, config, weights, &name_buf, s);
        const moe_layers: ?[]MoeLayerWeights = loaded.moe_layers;
        var moe_owned_bf16: ?[]mlx.mlx_array = loaded.owned_bf16;
        // Only a family with linear-attention (GDN) layers carries ssm_entries: a
        // pure-attention model carrying them makes the hot prefix cache classify
        // it as hybrid and force a cold prefill on every request.
        if (config.needsSsmEntries()) {
            ssm_entries = loaded.ssm_entries;
        } else {
            for (loaded.ssm_entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                ssmFreeQsaState(e);
            }
            allocator.free(loaded.ssm_entries);
        }

        const bits_cache: BitsCache = .{};

        // qwen4_exp: ones(hidden_size) for its weight-less RMS norm.
        var ones_hidden: ?mlx.mlx_array = null;
        if (config.isQwen4()) {
            const one_val = bf16Scalar(1.0, s);
            defer _ = mlx.mlx_array_free(one_val);
            const h_shape = [_]c_int{@intCast(config.hidden_size)};
            ones_hidden = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_full(&ones_hidden.?, &h_shape, 1, one_val, .bfloat16, s));
        }

        // YaRN (qwen4_exp: the trunk's one rope). Precompute the mlx_fast_rope denominator array, the mscale
        // vector (attention_factor on the rotated dims, 1.0 on the pass-through
        // tail) and the HOST inverse frequencies the M-RoPE tables and the QSA
        // pooled-key ladder are filled from. Sliding layers use default RoPE.
        var rope_freqs_yarn: ?mlx.mlx_array = null;
        var yarn_mscale: ?mlx.mlx_array = null;
        var yarn_inv_freq: ?[]const f64 = null;
        if (config.rope_yarn) {
            const spec = yarnSpec(&config);
            const rotary_dim: u32 = config.yarnRotaryDim();
            const half: usize = rotary_dim / 2;
            const freqs_f64 = try allocator.alloc(f64, half);
            defer allocator.free(freqs_f64);
            computeYarnFreqs(
                freqs_f64,
                config.head_dim,
                config.yarnPartial(),
                config.rope_theta,
                config.yarn_factor,
                config.yarn_beta_fast,
                config.yarn_beta_slow,
                config.yarn_orig_max_pos,
                config.yarn_truncate,
            );
            // The same table without the reciprocal: `fillCosSin` and
            // `ropeAtPositions` multiply positions by inv_freq (host f64), while
            // mlx_fast_rope divides by freqs — two spellings, one spectrum.
            const inv_f64 = try allocator.alloc(f64, half);
            errdefer allocator.free(inv_f64);
            spec.invFreq(inv_f64);
            yarn_inv_freq = inv_f64;
            const freqs_f32 = try allocator.alloc(f32, half);
            defer allocator.free(freqs_f32);
            for (freqs_f64, 0..) |v, i| freqs_f32[i] = @floatCast(v);
            const fshape = [_]c_int{@intCast(half)};
            rope_freqs_yarn = mlx.mlx_array_new_data(freqs_f32.ptr, &fshape, 1, .float32);
            // mscale vector [head_dim]: af on rotary dims, 1.0 on the tail.
            // Multiplying the post-rope q/k by this is exactly the reference's
            // cos/sin *= attention_factor (rotated dims only, pass dims untouched).
            const mscale_f32 = try allocator.alloc(f32, config.head_dim);
            defer allocator.free(mscale_f32);
            for (mscale_f32, 0..) |*m, i| m.* = if (i < rotary_dim) config.yarn_attention_factor else 1.0;
            const mshape = [_]c_int{@intCast(config.head_dim)};
            yarn_mscale = mlx.mlx_array_new_data(mscale_f32.ptr, &mshape, 1, .float32);
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            _ = mlx.mlx_vector_array_append_value(eval_vec, rope_freqs_yarn.?);
            _ = mlx.mlx_vector_array_append_value(eval_vec, yarn_mscale.?);
            try mlx.check(mlx.mlx_eval(eval_vec));
            const cr = spec.correctionRange();
            log.info("[rope] YaRN: factor {d:.4} (window {d} -> {d}), rotary {d} dims, ramp [{d:.0}, {d:.0}], mscale {d:.6}\n", .{
                config.yarn_factor,
                config.yarn_orig_max_pos,
                config.contextCap(),
                rotary_dim,
                cr.low,
                cr.high,
                config.yarn_attention_factor,
            });
        }

        // Batch eval all weights
        {
            const eval_start = std.Io.Timestamp.now(io, .awake);
            const all_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(all_vec);

            _ = mlx.mlx_vector_array_append_value(all_vec, emb_w);
            // Dense bf16 models have null-ctx embedding/lm_head scales/biases —
            // appending a null array aborts in mlx (vector.cpp "non-empty" guard).
            if (emb_s_arr.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, emb_s_arr);
            if (emb_b_arr.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, emb_b_arr);
            _ = mlx.mlx_vector_array_append_value(all_vec, lm_head_w);
            if (lm_head_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lm_head_s);
            if (lm_head_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lm_head_b);
            if (moe_layers) |ml| {
                for (ml) |lw| {
                    // qwen4_exp has no per-layer norms (null-ctx); its
                    // hyper-connection + PLE arrays are appended below.
                    if (lw.input_norm.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.input_norm);
                    if (lw.post_attn_norm.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.post_attn_norm);
                    if (lw.hc_attn) |*hw| appendStructArrays(all_vec, hw);
                    if (lw.hc_mlp) |*hw| appendStructArrays(all_vec, hw);
                    if (lw.ple) |*pw| appendStructArrays(all_vec, pw);
                    appendHybridMlpWeights(all_vec, &lw.mlp);
                    switch (lw.attn) {
                        .full => |fa| appendFullAttnWeights(all_vec, &fa),
                        .linear => |la| appendLinearAttnWeights(all_vec, &la),
                    }
                }
            }

            try mlx.check(mlx.mlx_eval(all_vec));
            const eval_ms: i64 = @intCast(@divTrunc(eval_start.untilNow(io, .awake).nanoseconds, std.time.ns_per_ms));
            log.info("Batch eval all weights: {d}ms\n", .{eval_ms});
        }

        // qwen4_exp: the final hyper-connection read (replaces model.norm)
        // and the module-owned n-gram hash + mmapped table.
        var qwen4_state: ?*qwen4_mod.Qwen4State = null;
        // Once published the state may run its table's warm thread, which `deinit` joins.
        errdefer if (qwen4_state) |st| {
            st.deinit();
            allocator.destroy(st);
        };
        var qwen4_mixer: ?HcWeights = null;
        var qwen4_mtp: ?Qwen4Mtp = null;
        errdefer if (qwen4_mtp) |*m| m.deinit(allocator);
        var expert_stream: ?*expert_stream_mod.Engine = null;
        errdefer if (expert_stream) |e| {
            e.deinit();
            allocator.destroy(e);
        };
        var imatrix: ?*imatrix_capture.Collector = null;
        errdefer if (imatrix) |c| c.deinit();
        if (config.isQwen4()) {
            var extra: std.ArrayList(mlx.mlx_array) = .empty;
            defer extra.deinit(allocator);
            var mb: [160]u8 = undefined;
            qwen4_mixer = try loadHcWeights(weights, std.fmt.bufPrint(&mb, "{s}.hyper_connection_mixer", .{prefix}) catch unreachable, false, config.hc_count, config.hidden_size, &extra, allocator, s);
            if (extra.items.len > 0) {
                const old = moe_owned_bf16 orelse &[_]mlx.mlx_array{};
                const merged = try std.mem.concat(allocator, mlx.mlx_array, &.{ old, extra.items });
                if (moe_owned_bf16) |o| allocator.free(o);
                moe_owned_bf16 = merged;
            }
            const eos: u32 = config.ngram_eos;
            // The `0` is the PLE's ordinal among the config's injection points, not
            // `config.ple_layer_idx` (pinned by the oracle fixture with `ple_layer_ids=[2]`).
            try checkQsaCompressRatio(config.indexer_compress_ratio);
            const st = try allocator.create(qwen4_mod.Qwen4State);
            {
                errdefer allocator.destroy(st);
                st.* = .{
                    .hash = try qwen4_mod.NgramHash.init(config.vocab_size, config.ngram_size, config.heads_per_ngram, config.ngram_vocab_base, config.ngram_vocab_divisor, config.ngram_seed, 0, eos),
                    .table = switch (config.ngramTableSource()) {
                        .bf16_override => blk: {
                            const dir = config.ngram_bf16_dir.?;
                            log.info("[qwen4] ngram table from the bf16 shards in {s} (SUSHI_NGRAM_BF16_DIR)\n", .{dir});
                            break :blk try qwen4_mod.NgramTable.openBf16(allocator, dir);
                        },
                        .bf16_streamed => try qwen4_mod.NgramTable.openBf16(allocator, config.expert_source_dir orelse return error.MissingExpertSourceDir),
                        .quantized => try qwen4_mod.NgramTable.open(config.ngram_table_path orelse return error.MissingNgramTable),
                    },
                };
            }
            qwen4_state = st;
            if (st.table.rows != st.hash.total_rows or st.table.dim * st.hash.n_heads != config.ple_embed_dim) {
                log.err("[qwen4] ngram_table.bin geometry {d}x{d} does not match the config ({d} rows, {d} heads x dim)\n", .{ st.table.rows, st.table.dim, st.hash.total_rows, st.hash.n_heads });
                return error.NgramTableMismatch;
            }
            var gpu_bytes: usize = 0;
            _ = mlx.mlx_get_active_memory(&gpu_bytes);
            st.table.calibrateArm();
            st.table.startWarm(); // the weights load just evicted the table from page cache
            qwen4_mtp = try loadQwen4Mtp(allocator, config, weights, &name_buf, s);
            if (config.expert_streaming) {
                if (qwen4_mtp) |*mtp_head| try evalQwen4MtpResident(mtp_head);
            }
            if (config.expert_streaming) {
                const parts = try initExpertStream(allocator, &config, s);
                expert_stream = parts.engine;
                imatrix = parts.imatrix;
            }
            log.info("[qwen4] n-gram table {d} rows x {d} ({d}-bit, {s}), PLE at layer {d}, QSA budget {d}/{d}\n", .{ st.table.rows, st.table.dim, st.table.bits, if (config.expert_streaming and st.table.bits == 16) "sharded pread" else "mmapped", config.ple_layer_idx, config.indexer_budget, config.indexer_compress_ratio });
        } else if (std.mem.eql(u8, config.model_type, "mimo_v2") and config.expert_streaming) {
            const parts = try initExpertStream(allocator, &config, s);
            expert_stream = parts.engine;
            imatrix = parts.imatrix;
        }

        return .{
            .config = config,
            .cache = cache,
            .s = s,
            .allocator = allocator,
            .emb_w = emb_w,
            .emb_s = emb_s_arr,
            .emb_b = emb_b_arr,
            .emb_scale = emb_scale,
            .final_norm = final_norm,
            .qwen4 = qwen4_state,
            .qwen4_mixer = qwen4_mixer,
            .qwen4_mtp = qwen4_mtp,
            .expert_stream = expert_stream,
            .imatrix = imatrix,
            .lm_head_w = lm_head_w,
            .lm_head_s = lm_head_s,
            .lm_head_b = lm_head_b,
            .owns_lm_head = owns_lm_head,
            .owns_norms = config.norm_has_offset,
            .decode_async_ladder = decodeAsyncLadderStride(
                if (std.c.getenv("SUSHI_DECODE_ASYNC_LADDER")) |v| std.mem.span(v) else null,
            ),
            .decode_async_ladder_qwen4 = qwen4DecodeLadderStride(
                if (std.c.getenv("SUSHI_DECODE_ASYNC_LADDER")) |v| std.mem.span(v) else null,
            ),
            .gelu_coeff = if (need_gelu) bf16Scalar(0.7978845608028654, s) else null,
            .gelu_inner = if (need_gelu) bf16Scalar(0.044715, s) else null,
            .half = bf16Scalar(0.5, s),
            .one = bf16Scalar(1.0, s),
            .rms_eps_arr = mlx.mlx_array_new_float(config.rms_norm_eps),
            .three = if (need_gelu) bf16Scalar(3.0, s) else null,
            .neg_one = if (need_silu) bf16Scalar(-1.0, s) else null,
            .ones_hidden = ones_hidden,
            .rope_freqs_yarn = rope_freqs_yarn,
            .yarn_mscale = yarn_mscale,
            .yarn_inv_freq = yarn_inv_freq,
            .moe_layers = moe_layers,
            .ssm_entries = ssm_entries,
            .moe_seq_offset = 0,
            .moe_owned_bf16 = moe_owned_bf16,
            .bits_cache = bits_cache,
        };
    }

    /// Reset all caches for a new request (KV cache + SSM state for MoE).
    pub fn resetCache(self: *Transformer) !void {
        if (self.glm5_request) |*request| request.reset();
        if (self.dsv41) |m| dsv41_model.resetState(m);
        const prev_config = self.cache.config;
        try self.cache.reinit(self.config.num_hidden_layers, prev_config);
        if (self.ssm_entries) |entries| {
            for (entries) |*e| {
                ssmFreeSpecCapture(e);
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                ssmFreeQsaState(e);
                e.conv_state = mlx.mlx_array_new();
                e.ssm_state = mlx.mlx_array_new();
                e.ple_prev_valid = false;
                e.initialized = false;
            }
        }
        self.moe_seq_offset = 0;
    }

    /// Compile GELU activation for kernel fusion.
    /// Fuses 8 separate ops into 1 GPU kernel, matching mlx-lm's @mx.compile behavior.
    pub fn compileGelu(self: *Transformer) void {
        const raw_closure = mlx.mlx_closure_new_func_payload(
            &geluClosureCallback,
            @ptrCast(self),
            null,
        );
        var compiled = mlx.mlx_closure{ .ctx = null };
        const rc = mlx.mlx_compile(&compiled, raw_closure, true); // shapeless=true
        _ = mlx.mlx_closure_free(raw_closure);
        if (rc == 0 and compiled.ctx != null) {
            self.compiled_gelu = compiled;
            log.info("GELU compiled (kernel fusion enabled)\n", .{});
        } else {
            log.warn("GELU compilation failed, using uncompiled path\n", .{});
        }
    }

    /// Compile GeGLU: gelu(gate) * up → single fused kernel. The silu arm is
    /// NOT routed here — mlx_compile's elementwise fusion does not reproduce
    /// the op chain's intermediate rounding (a same-binary greedy A/B diverged
    /// at token ~55), so silu gets the bit-exact `fusedSwiGLU` kernel instead.
    pub fn compileGeglu(self: *Transformer) void {
        const raw_closure = mlx.mlx_closure_new_func_payload(
            &gegluClosureCallback,
            @ptrCast(self),
            null,
        );
        var compiled = mlx.mlx_closure{ .ctx = null };
        const rc = mlx.mlx_compile(&compiled, raw_closure, true);
        _ = mlx.mlx_closure_free(raw_closure);
        if (rc == 0 and compiled.ctx != null) {
            self.compiled_geglu = compiled;
            log.info("GeGLU compiled (kernel fusion enabled)\n", .{});
        }
    }

    fn gegluClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const self: *Transformer = @ptrCast(@alignCast(payload.?));
        var gate = mlx.mlx_array_new();
        var up = mlx.mlx_array_new();
        if (mlx.mlx_vector_array_get(&gate, input, 0) != 0) return -1;
        if (mlx.mlx_vector_array_get(&up, input, 1) != 0) {
            _ = mlx.mlx_array_free(gate);
            return -1;
        }
        defer _ = mlx.mlx_array_free(gate);
        defer _ = mlx.mlx_array_free(up);

        // geglu(gate, up) = gelu_approx(gate) * up
        const activated = self.geluUncompiled(gate) catch return -1;
        defer _ = mlx.mlx_array_free(activated);
        var result = mlx.mlx_array_new();
        mlx.check(mlx.mlx_multiply(&result, activated, up, self.s)) catch return -1;

        const out_arr = [_]mlx.mlx_array{result};
        res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
        _ = mlx.mlx_array_free(result);
        return 0;
    }

    fn geluClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const self: *Transformer = @ptrCast(@alignCast(payload.?));
        var x = mlx.mlx_array_new();
        const get_rc = mlx.mlx_vector_array_get(&x, input, 0);
        if (get_rc != 0) return get_rc;
        defer _ = mlx.mlx_array_free(x);

        // gelu_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
        const result = self.geluUncompiled(x) catch return -1;

        const out_arr = [_]mlx.mlx_array{result};
        res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
        _ = mlx.mlx_array_free(result);
        return 0;
    }












    /// Result type for the MoE routing helpers. Both fields are owned arrays —
    /// caller is responsible for freeing them.
    pub const MoeRouting = struct { inds: mlx.mlx_array, norm_scores: mlx.mlx_array, gate_logit: mlx.mlx_array = .{ .ctx = null } };


    /// Reference implementation of the MoE routing chain (used both as fallback
    /// and as the body the compiled closure traces). Returns owned `inds` +
    /// `norm_scores` arrays — caller must free both.
    pub fn moeRoutingUncompiled(self: *const Transformer, router_logits: mlx.mlx_array, k: c_int) !MoeRouting {
        return moeRoutingChain(router_logits, k, self.s);
    }


    /// Apply the compiled MoE routing closure if available, else fall back.
    /// Returns owned `inds` + `norm_scores` — caller must free both.
    fn computeMoeRouting(self: *const Transformer, router_logits: mlx.mlx_array, gate: ?RouterGate) !MoeRouting {
        const k: c_int = @intCast(self.config.num_experts_per_tok);
        if (try moeRouterTopKGate(
            self.s,
            router_logits,
            .{ .ctx = null },
            k,
            .softmax,
            true,
            1.0,
            mlx.mlx_array_dtype(router_logits),
            0,
            0,
            gate,
        )) |fused| return fused;
        if (self.compiled_moe_routing) |compiled| {
            const in_arr = [_]mlx.mlx_array{router_logits};
            const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 1);
            defer _ = mlx.mlx_vector_array_free(in_vec);
            var out_vec = mlx.mlx_vector_array{ .ctx = null };
            try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
            defer _ = mlx.mlx_vector_array_free(out_vec);

            var inds = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(inds);
            try mlx.check(mlx.mlx_vector_array_get(&inds, out_vec, 0));
            var norm_scores = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(norm_scores);
            try mlx.check(mlx.mlx_vector_array_get(&norm_scores, out_vec, 1));
            return .{ .inds = inds, .norm_scores = norm_scores };
        }
        return self.moeRoutingUncompiled(router_logits, k);
    }

    /// Write the activation capture out and drop it. Idempotent, and MUST run on
    /// the inference thread: the accumulators' stream is bound to it, while
    /// `Scheduler.deinit` tears models down on the caller's thread.
    pub fn flushImatrix(self: *Transformer) void {
        const c = self.imatrix orelse return;
        self.imatrix = null;
        _ = c.flush() catch |e| log.warn("[imatrix] flush failed: {s}\n", .{@errorName(e)});
        c.deinit();
    }

    pub fn deinit(self: *Transformer) void {
        self.flushImatrix(); // `/v1/unload-model` reaches here on the inference thread
        if (self.lm_head_coarse) |*rc| rc.deinit();
        self.lm_head_coarse = null;
        self.releaseJoinedVerifyLogits();
        if (self.glm5_request) |*request| request.deinit();
        self.glm5_request = null;
        if (self.glm5) |mdl| {
            mdl.deinit();
            self.allocator.destroy(mdl);
            self.glm5 = null;
        }
        if (self.compiled_gelu) |cg| _ = mlx.mlx_closure_free(cg);
        if (self.compiled_geglu) |cg| _ = mlx.mlx_closure_free(cg);
        if (self.compiled_moe_routing) |cmr| _ = mlx.mlx_closure_free(cmr);
        if (self.compiled_gdn_gate) |cgg| _ = mlx.mlx_closure_free(cgg);
        if (self.gdn_ones_w) |w| _ = mlx.mlx_array_free(w);
        if (self.gdn_q_scale) |q| _ = mlx.mlx_array_free(q);
        if (self.gdn_eps) |e| _ = mlx.mlx_array_free(e);
        if (self.gdn_k_scale) |k| _ = mlx.mlx_array_free(k);
        if (self.ones_hidden) |o| _ = mlx.mlx_array_free(o);
        if (self.rope_freqs_yarn) |f| _ = mlx.mlx_array_free(f);
        if (self.yarn_mscale) |m| _ = mlx.mlx_array_free(m);
        if (self.yarn_inv_freq) |f| self.allocator.free(f);
        if (self.suppress_mask) |m| _ = mlx.mlx_array_free(m);
        if (self.think_marker_mask) |m| _ = mlx.mlx_array_free(m);
        self.allocator.free(self.logit_bias);
        self.qsa_consts.deinit();
        self.qsa_pooled_rope.deinit();
        self.yarn_mscale_cast.deinit();
        self.cache.deinit();
        if (self.emb_scale) |es| _ = mlx.mlx_array_free(es);
        if (self.owns_norms) _ = mlx.mlx_array_free(self.final_norm);
        if (self.gelu_coeff) |g| _ = mlx.mlx_array_free(g);
        if (self.gelu_inner) |g| _ = mlx.mlx_array_free(g);
        _ = mlx.mlx_array_free(self.half);
        _ = mlx.mlx_array_free(self.one);
        if (self.rms_eps_arr.ctx != null) _ = mlx.mlx_array_free(self.rms_eps_arr);
        for (&self.qk_angle_cache) |*slot| {
            if (slot.arr.ctx != null) _ = mlx.mlx_array_free(slot.arr);
        }
        if (self.three) |t| _ = mlx.mlx_array_free(t);
        if (self.neg_one) |n| _ = mlx.mlx_array_free(n);
        if (self.ssm_entries) |entries| {
            for (entries) |*e| {
                ssmFreeSpecCapture(e);
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                ssmFreeQsaState(e);
            }
            self.allocator.free(entries);
        }
        self.ssm_group.deinit();
        if (self.moe_layers) |ml| self.allocator.free(ml);
        if (self.qwen4_mtp) |*m| {
            m.deinit(self.allocator);
            self.qwen4_mtp = null;
        }
        if (self.dsv41) |m| {
            m.deinit();
            self.dsv41 = null;
        }
        if (self.dsv41_experts) |st| {
            self.allocator.destroy(st);
            self.dsv41_experts = null;
        }
        if (self.dsv41_stq_experts) |st| {
            self.allocator.destroy(st);
            self.dsv41_stq_experts = null;
        }
        if (self.expert_stream) |engine| {
            engine.deinit();
            self.allocator.destroy(engine);
            self.expert_stream = null;
        }
        if (self.qwen4) |st| {
            st.deinit();
            self.allocator.destroy(st);
            self.qwen4 = null;
        }
        if (self.moe_owned_bf16) |arrs| {
            for (arrs) |a| _ = mlx.mlx_array_free(a);
            self.allocator.free(arrs);
        }
        // self.s is the thread's default GPU stream (not owned by us) — don't free it.
        _ = mlx.mlx_stream_free(self.s);
        // The free above is a no-op on the default stream's wrapper but we keep it for symmetry
        // with the Zig copy of the mlx_stream struct that init handed us.
    }

    /// Re-bind `self.s` to the *current* thread's default GPU stream. Must be called from any
    /// thread that's about to use this Transformer for inference, since mlx 0.31.2 made streams
    /// thread-local (a stream created on thread A is invisible to thread B).
    pub fn useCurrentThreadStream(self: *Transformer) void {
        self.s = mlx.gpuStream();
        if (self.glm5) |mdl| mdl.s = self.s;
        if (self.dsv41) |m| m.s = self.s;
    }

    // ── Core ops ──

    pub inline fn qmatmul(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array) !mlx.mlx_array {
        // u8 codes + f32 tile scales = an FP8 source linear (`mimo_source`).
        if (mlx.mlx_array_dtype(w) == .uint8) return fp8_block.linearServing(self.s, x, w, sc);
        // Resolve (bits, group_size, mode) per weight. Most weights inherit the
        // global config; per-weight overrides (mixed-precision checkpoints, e.g.
        // affine 8-bit shared MLP inside an mxfp4 model) are detected on
        // first touch — x's inner dim pins (bits, group_size) exactly.
        const qp = self.quantParamsHinted(w, sc, lastDim(x));
        return qmatmulBits(x, w, sc, bi, qp.bits, qp.group_size, qp.mode, self.s);
    }

    /// Final logits projection. For dense bf16 the lm_head weight is [vocab, hidden]
    /// and is NOT pre-transposed at load (when tied it aliases emb_w, which must stay
    /// [vocab, hidden] for the embedding lookup), so we project via a lazy transposed
    /// view. Quantized models fall through to the standard gather/qmm path unchanged.
    /// `argmax_only` (the Generator's request-level gate, via ForwardCtx) permits
    /// MiMo's coarse shortlist (quantized); false = the full head.
    pub inline fn lmHeadProject(self: *const Transformer, x: mlx.mlx_array, argmax_only: bool) !mlx.mlx_array {
        if (self.lm_head_s.ctx == null) {
            const wt = try transposeBf16Weight(self.lm_head_w, self.s); // [hidden, vocab] view
            defer _ = mlx.mlx_array_free(wt);
            return qmatmulBits(x, wt, .{}, .{}, 0, 64, .affine, self.s);
        }
        if (argmax_only) {
            if (try self.lmHeadShortlistFor(x)) |sparse| return sparse;
        }
        return self.qmatmul(x, self.lm_head_w, self.lm_head_s, self.lm_head_b);
    }

    /// Argmax-only readout on MiMo: the coarse head's top-32 per row, re-scored through the full head
    /// (a 32-row slice of the same qmv, so each value is the full readout's), every other id -inf.
    /// Serial ticks and verify rows take it alike; null keeps the full head.
    fn lmHeadShortlistFor(self: *const Transformer, x: mlx.mlx_array) !?mlx.mlx_array {
        const coarse = self.lm_head_coarse orelse return null;
        if (!self.config.isMimo() or !lmHeadShortlistOn()) return null;
        const xsh = mlx.getShape(x);
        if (xsh.len != 3 or mlx.mlx_array_dtype(x) != .bfloat16) return null;
        const n: usize = @intCast(xsh[0] * xsh[1]);
        if (n < 1 or n > LMHEAD_SHORTLIST_MAX_ROWS) return null;
        const vocab = coarse.rows;
        const mut = @constCast(self);
        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        try mlx.check(mlx.mlx_reshape(&x3, x, &[_]c_int{ @intCast(n), 1, xsh[2] }, 3, self.s));
        var logged = true;
        var lists: [LMHEAD_SHORTLIST_MAX_ROWS]mtp_mod.Shortlist = undefined;
        if (!try mtp_mod.rerankShortlistsBatched(self.s, mut, &mut.lm_head_coarse, &logged, x3, self.suppress_mask, lists[0..n])) return null;
        defer for (lists[0..n]) |*sl| sl.deinit();
        const sparse = try shortlistSparseLogits(self.s, lists[0..n], vocab);
        defer _ = mlx.mlx_array_free(sparse.idx);
        defer _ = mlx.mlx_array_free(sparse.logits);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_reshape(&out, sparse.logits, &[_]c_int{ xsh[0], xsh[1], vocab }, 3, self.s));
        if (diagEnvOnCached(&lmhead_shortlist_audit_env, "SUSHI_LMHEAD_SHORTLIST_AUDIT")) {
            const full = try self.qmatmul(x3, self.lm_head_w, self.lm_head_s, self.lm_head_b);
            defer _ = mlx.mlx_array_free(full);
            try shortlistAudit(self.s, full, sparse.logits, sparse.idx, n, vocab);
        }
        if (!mut.lm_head_coarse_logged) {
            mut.lm_head_coarse_logged = true;
            log.info("[lm_head] greedy shortlist engaged: {d}-bit coarse top-32 + full-head re-score (SUSHI_LMHEAD_FULL=1 reads the full head)\n", .{coarse.bits});
        }
        return out;
    }

    /// The trunk's full vocabulary projection of `x` (an MTP head's draft logits).
    pub fn lmHeadLogits(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        return self.lmHeadProject(x, false);
    }

    /// Resolve per-weight quant params without an activation hint — for callers
    /// like the embedding gather whose tensors follow the model-wide scheme.
    pub inline fn quantParamsFor(self: *const Transformer, w: mlx.mlx_array, sc: mlx.mlx_array) QuantParams {
        return self.quantParamsHinted(w, sc, null);
    }

    /// Resolve per-weight (bits, group_size, mode) with a lazy cache keyed by
    /// the scales array pointer. First touch computes params from the scales
    /// dtype + shapes (~4 FFI calls); subsequent calls are a single pointer
    /// compare. `in_dim` is the weight's input dimension when the caller knows
    /// it (activation inner dim) — required to disambiguate (bits, group_size)
    /// for affine-override tensors inside a non-affine model.
    /// Thread-safety: generation is single-threaded.
    pub inline fn quantParamsHinted(self: *const Transformer, w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: ?u32) QuantParams {
        const key_raw = sc.ctx orelse return .{
            .bits = self.config.quant_bits,
            .group_size = self.config.quant_group_size,
            .mode = self.config.quant_mode,
        };
        const cache = @constCast(&self.bits_cache);
        const start = BitsCache.slot(key_raw);
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            const idx = (start + i) & (BITS_CACHE_CAP - 1);
            const entry = cache.keys[idx];
            if (entry == key_raw) {
                return .{
                    .bits = cache.vals_bits[idx],
                    .group_size = @as(u32, cache.vals_gs_div8[idx]) * 8,
                    .mode = @fromBackingInt(@intCast(cache.vals_mode[idx])),
                };
            }
            if (entry == null) {
                const detected = computeQuantParams(&self.config, w, sc, in_dim);
                _ = cache.put(key_raw, detected);
                return detected;
            }
        }
        // Probe window saturated — fall through to direct detect, no cache write.
        return computeQuantParams(&self.config, w, sc, in_dim);
    }

    pub inline fn rmsNorm(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rms_norm(&result, x, w, self.config.rms_norm_eps, self.s));
        return result;
    }

    /// Raw embedding-table lookup: gather + dequant — no
    /// normed-embeddings RMS norm, no emb_scale. The DFlash noise embeds are
    /// contractually the raw table rows (the reference calls `F.embedding`
    /// on the weight directly — the un-foldable embed norm is exactly what
    /// makes this hook possible); the trunk's own forward goes through
    /// `embedding`, which layers the norm/scale on top of this. Returns
    /// `[B, L, hidden]` bf16 (MiMo dense tables retain their dtype), caller frees.
    pub fn rawEmbedding(self: *const Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
        if (self.emb_w.ctx == null) return error.MissingWeight;
        const id_shape = mlx.getShape(token_ids);
        const batch = id_shape[0];
        const seq_len = id_shape[1];

        const flat_shape = [_]c_int{batch * seq_len};
        var flat_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat_ids);
        try mlx.check(mlx.mlx_reshape(&flat_ids, token_ids, &flat_shape, 1, self.s));

        var taken_w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken_w);
        try mlx.check(mlx.mlx_take_axis(&taken_w, self.emb_w, flat_ids, 0, self.s));

        var emb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(emb);
        if (self.emb_s.ctx == null) {
            // MiMo's residual dtype follows its dense embedding table.
            const dtype = if (std.mem.eql(u8, self.config.model_type, "mimo_v2"))
                mlx.mlx_array_dtype(taken_w)
            else
                mlx.mlx_dtype.bfloat16;
            try mlx.check(mlx.mlx_astype(&emb, taken_w, dtype, self.s));
        } else {
            var taken_s = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(taken_s);
            try mlx.check(mlx.mlx_take_axis(&taken_s, self.emb_s, flat_ids, 0, self.s));
            // Bias-less modes (mxfp4/mxfp8) have a null-ctx emb_b —
            // mlx_take_axis on a null handle aborts, so gather only when present.
            var taken_b = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(taken_b);
            if (self.emb_b.ctx != null) {
                try mlx.check(mlx.mlx_take_axis(&taken_b, self.emb_b, flat_ids, 0, self.s));
            }
            const emb_qp = self.quantParamsHinted(self.emb_w, self.emb_s, self.config.hidden_size);
            try mlx.check(mlx.mlx_dequantize(
                &emb,
                taken_w,
                taken_s,
                taken_b,
                mlx.mlx_optional_int.some(@intCast(emb_qp.group_size)),
                mlx.mlx_optional_int.some(@intCast(emb_qp.bits)),
                emb_qp.mode.cstr(),
                .{}, // global_scale (null)
                .{ .value = .bfloat16, .has_value = true },
                self.s,
            ));
        }

        const out_shape = [_]c_int{ batch, seq_len, @intCast(self.config.hidden_size) };
        var reshaped = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&reshaped, emb, &out_shape, 3, self.s));
        return reshaped;
    }

    pub fn embedding(self: *const Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
        if (self.glm5) |mdl| return mdl.rawEmbedding(token_ids);
        const raw = try self.rawEmbedding(token_ids);

        if (self.emb_scale) |scale| {
            defer _ = mlx.mlx_array_free(raw);
            var scaled = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&scaled, raw, scale, self.s));
            return scaled;
        }
        return raw;
    }

    /// Splice vision embeddings into text embeddings at image_token_id positions.
    /// h: [B, seq_len, hidden] text embeddings
    /// token_ids: [B, seq_len] original token IDs
    /// vision_emb: [B, N_img, hidden] vision embeddings
    /// Returns new h with vision embeddings replacing image token positions.
    /// masked_scatter: replaces image token positions with vision features.
    /// Matches Python reference: cumsum-based indexing into flattened source.
    pub fn spliceVisionEmbeddings(self: *Transformer, h: mlx.mlx_array, token_ids: mlx.mlx_array, vision_emb: mlx.mlx_array, image_token_id: u32, video_token_id: u32, row_offset: usize) !mlx.mlx_array {
        return spliceVisionRows(h, token_ids, vision_emb, image_token_id, video_token_id, row_offset, self.s);
    }

    /// Scatter vision/audio embedding rows into `h` at placeholder-token
    /// positions. `row_offset` is the number of placeholder tokens earlier
    /// chunks of this prefill already consumed (chunked vision prefill): the
    /// row lookup is a cumsum over THIS forward's mask, so the offset is what
    /// keeps a chunk boundary inside an image from restarting the scatter at
    /// row 0. Frees `h` and returns the spliced replacement.
    pub fn spliceVisionRows(h: mlx.mlx_array, token_ids: mlx.mlx_array, vision_emb: mlx.mlx_array, image_token_id: u32, video_token_id: u32, row_offset: usize, s: mlx.mlx_stream) !mlx.mlx_array {
        const h_shape = mlx.getShape(h);

        // mask = (token_ids == image_token_id) [| (token_ids == video_token_id)].
        const img_id_arr = mlx.mlx_array_new_int(@intCast(image_token_id));
        defer _ = mlx.mlx_array_free(img_id_arr);
        var mask_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask_2d);
        try mlx.check(mlx.mlx_equal(&mask_2d, token_ids, img_id_arr, s));
        // Qwen video pads (`video_token_id`) ride the same row stream in prompt
        // order — the encoder output is concatenated in the order the
        // placeholder runs sit in the prompt.
        for ([_]u32{video_token_id}) |extra_id| {
            if (extra_id == 0) continue;
            const extra_arr = mlx.mlx_array_new_int(@intCast(extra_id));
            defer _ = mlx.mlx_array_free(extra_arr);
            var extra_mask = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(extra_mask);
            try mlx.check(mlx.mlx_equal(&extra_mask, token_ids, extra_arr, s));
            var combined = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_logical_or(&combined, mask_2d, extra_mask, s));
            _ = mlx.mlx_array_free(mask_2d);
            mask_2d = combined;
        }

        // Expand mask to [B, seq_len, hidden] via broadcast
        const expand_shape = [_]c_int{ h_shape[0], h_shape[1], 1 };
        var mask_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask_3d);
        try mlx.check(mlx.mlx_reshape(&mask_3d, mask_2d, &expand_shape, 3, s));

        // Broadcast to full shape via logical and with ones
        var mask_expanded = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask_expanded);
        {
            var ones_h = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ones_h);
            try mlx.check(mlx.mlx_ones(&ones_h, h_shape.ptr, 3, .bool_, s));
            try mlx.check(mlx.mlx_multiply(&mask_expanded, mask_3d, ones_h, s));
        }

        // Flatten everything
        const total = h_shape[0] * h_shape[1] * h_shape[2];
        const flat_shape = [_]c_int{total};

        var mask_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask_flat);
        try mlx.check(mlx.mlx_reshape(&mask_flat, mask_expanded, &flat_shape, 1, s));

        // mask_int = mask_flat.astype(int32)
        var mask_int = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask_int);
        try mlx.check(mlx.mlx_astype(&mask_int, mask_flat, .int32, s));

        // indices = cumsum(mask_int, axis=0) - 1 + row_offset*hidden. The
        // cumsum counts masked ELEMENTS of this forward (hidden consecutive
        // per placeholder token); earlier chunks consumed row_offset rows of
        // the flat source, i.e. row_offset*hidden elements.
        var cumsum_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cumsum_arr);
        try mlx.check(mlx.mlx_cumsum(&cumsum_arr, mask_int, 0, false, true, s));
        const elem_offset: i64 = @as(i64, @intCast(row_offset)) * @as(i64, @intCast(h_shape[2]));
        const off_i = mlx.mlx_array_new_int(@intCast(elem_offset - 1));
        defer _ = mlx.mlx_array_free(off_i);
        var indices = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(indices);
        try mlx.check(mlx.mlx_add(&indices, cumsum_arr, off_i, s));

        // source = vision_emb.flatten()
        const ve_shape = mlx.getShape(vision_emb);
        const source_size = ve_shape[0] * ve_shape[1] * ve_shape[2];
        const source_shape = [_]c_int{source_size};
        var source_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(source_flat);
        try mlx.check(mlx.mlx_reshape(&source_flat, vision_emb, &source_shape, 1, s));

        // indices_mod = indices % source_size
        const source_size_arr = mlx.mlx_array_new_int(source_size);
        defer _ = mlx.mlx_array_free(source_size_arr);
        var indices_mod = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(indices_mod);
        try mlx.check(mlx.mlx_remainder(&indices_mod, indices, source_size_arr, s));

        // aligned = source[indices_mod]
        var aligned = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(aligned);
        try mlx.check(mlx.mlx_take(&aligned, source_flat, indices_mod, s));

        // Cast aligned to bf16 to match h
        var aligned_bf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(aligned_bf);
        try mlx.check(mlx.mlx_astype(&aligned_bf, aligned, .bfloat16, s));

        // input_flat = h.flatten()
        var input_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(input_flat);
        try mlx.check(mlx.mlx_reshape(&input_flat, h, &flat_shape, 1, s));

        // result_flat = where(mask_flat, aligned_bf, input_flat)
        var result_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(result_flat);
        try mlx.check(mlx.mlx_where(&result_flat, mask_flat, aligned_bf, input_flat, s));

        // Reshape back to [B, seq_len, hidden]
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&result, result_flat, h_shape.ptr, 3, s));
        _ = mlx.mlx_array_free(h);
        return result;
    }

    /// Apply vision embeddings to text embeddings during prefill.
    /// Handles scaling and splicing at image_token_id positions.
    /// Returns the (potentially modified) h; caller should replace their h with the result.
    pub fn applyVisionEmbeddingsWith(self: *Transformer, ctx: *ForwardCtx, h: mlx.mlx_array, token_ids: mlx.mlx_array) !mlx.mlx_array {
        const cfg = &self.config;
        const h_shape = mlx.getShape(h);
        // Only during prefill (seq_len > 1)
        if (h_shape[1] <= 1) return h;
        const ve = ctx.vision_embeddings orelse return h;
        if (cfg.image_token_id == 0) return h;

        // Vision embeddings come out of the MultimodalEmbedder already in text-hidden space;
        // mlx-vlm does NOT re-scale them by sqrt(hidden) the way text embeddings are scaled
        // at LM embedding time. Splice directly — scaling here corrupts the MoE router's
        // magnitude assumptions (visible as "please provide an image" responses on 26B MoE).
        return self.spliceVisionEmbeddings(h, token_ids, ve, cfg.image_token_id, cfg.video_token_id, ctx.vision_splice_offset);
    }

    // ── Activation functions ──

    /// GELU approximate: dispatches to compiled (fused kernel) when available.
    fn gelu(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        if (self.compiled_gelu) |compiled| {
            const in_arr = [_]mlx.mlx_array{x};
            const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 1);
            defer _ = mlx.mlx_vector_array_free(in_vec);
            var out_vec = mlx.mlx_vector_array{ .ctx = null };
            try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
            defer _ = mlx.mlx_vector_array_free(out_vec);
            var result = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&result, out_vec, 0));
            return result;
        }
        return self.geluUncompiled(x);
    }

    /// Raw GELU implementation (8 ops, used as fallback and as compilation source).
    fn geluUncompiled(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        try mlx.check(mlx.mlx_power(&x3, x, self.three.?, self.s));
        var inner = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inner);
        try mlx.check(mlx.mlx_multiply(&inner, self.gelu_inner.?, x3, self.s));
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, x, inner, self.s));
        var scaled_val = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled_val);
        try mlx.check(mlx.mlx_multiply(&scaled_val, self.gelu_coeff.?, sum, self.s));
        var tanh_val = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tanh_val);
        try mlx.check(mlx.mlx_tanh(&tanh_val, scaled_val, self.s));
        var one_plus = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(one_plus);
        try mlx.check(mlx.mlx_add(&one_plus, self.one, tanh_val, self.s));
        var x_times = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_times);
        try mlx.check(mlx.mlx_multiply(&x_times, x, one_plus, self.s));
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&result, x_times, self.half, self.s));
        return result;
    }

    pub fn silu(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, x, self.s));
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&result, x, sig, self.s));
        return result;
    }

    inline fn mlpActivation(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        return switch (self.config.hidden_act) {
            .gelu_approx => self.gelu(x),
            .gelu => self.geluErf(x),
            .silu => self.silu(x),
            .relu_sq => self.reluSquared(x),
        };
    }

    /// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))), in the activation dtype.
    fn geluErf(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        const inv_sqrt2 = try scalarOf(0.70710678118654752, mlx.mlx_array_dtype(x), self.s);
        defer _ = mlx.mlx_array_free(inv_sqrt2);
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_multiply(&scaled, x, inv_sqrt2, self.s));
        var erf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(erf);
        try mlx.check(mlx.mlx_erf(&erf, scaled, self.s));
        var one_plus = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(one_plus);
        try mlx.check(mlx.mlx_add(&one_plus, self.one, erf, self.s));
        var x_times = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_times);
        try mlx.check(mlx.mlx_multiply(&x_times, x, one_plus, self.s));
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&result, x_times, self.half, self.s));
        return result;
    }

    fn reluSquared(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        const zero = bf16Scalar(0.0, self.s);
        defer _ = mlx.mlx_array_free(zero);
        var relu = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(relu);
        try mlx.check(mlx.mlx_maximum(&relu, x, zero, self.s));
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_square(&result, relu, self.s));
        return result;
    }

    // ── Conv1d with cache (GatedDeltaNet) ──

    /// Fused GeGLU: gelu(gate) * up in a single compiled kernel.
    /// Falls back to separate ops if not compiled.
    fn computeGeglu(self: *const Transformer, gate: mlx.mlx_array, up: mlx.mlx_array) !mlx.mlx_array {
        if (self.compiled_geglu) |compiled| {
            const in_arr = [_]mlx.mlx_array{ gate, up };
            const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 2);
            defer _ = mlx.mlx_vector_array_free(in_vec);
            var out_vec = mlx.mlx_vector_array{ .ctx = null };
            try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
            defer _ = mlx.mlx_vector_array_free(out_vec);
            var result = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&result, out_vec, 0));
            return result;
        }
        // silu: one bit-exact fused kernel (see fusedSwiGLU).
        if (self.config.hidden_act == .silu) {
            if (try fusedSwiGLU(self.s, gate, up)) |fused| return fused;
        }
        // Fallback: separate activation + multiply
        const activated = try self.mlpActivation(gate);
        defer _ = mlx.mlx_array_free(activated);
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&result, activated, up, self.s));
        return result;
    }

    /// SwiGLU: silu(gate) * x
    pub fn swiglu(self: *const Transformer, gate: mlx.mlx_array, x: mlx.mlx_array) !mlx.mlx_array {
        const activated = try self.silu(gate);
        defer _ = mlx.mlx_array_free(activated);
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&result, activated, x, self.s));
        return result;
    }

    // ── Forward dispatch ──

    const EVAL_EVERY_N_LAYERS: u32 = 48;
    /// One value for the prefill loops AND the admission bill: at most `cadence + 1`
    /// linear layers hold a chunk-wide stream between cadence points.
    pub const PREFILL_EVAL_CADENCE_DEFAULT: u32 = 4;
    pub const MOE_EVAL_EVERY_N_LAYERS: u32 = PREFILL_EVAL_CADENCE_DEFAULT;
    const RECURRENCE_EVAL_INTERVAL: usize = 32;

    /// Per-layer prefill transient (bytes) above which the layer loop eval()s
    /// after EVERY layer instead of the coarse default cadence. Between eval
    /// points ~3 layers' transients coexist under MLX lazy eval, so a
    /// multi-GB transient triples; eval-per-layer bounds peak to ~1 layer.
    /// Two transients count (both scale with total_kv, so short prompts keep
    /// the coarse cadence and pay nothing):
    ///   - the composed-SDPA score tensor [heads, chunk, total_kv] — only for
    ///     head_dims no fused kernel covers (prefillHeadDimFused: <= 128 via
    ///     MLX, 256 via sushi_attn_p256; unfused only via the kill switch or an
    ///     exotic dim);
    ///   - the dense-fp16 rebuild of a quantized KV cache (denseView runs
    ///     per layer under --kv-quant, over the FULL cache).
    /// Budget calibrated live (M4 Max):
    /// eval-per-layer costs ~4.5% prefill at a 5K prompt (845 MB scores —
    /// below this budget, coarse cadence kept) and ~0% at 102K (3.4 GB
    /// scores — flips to per-layer, peak 51.6 -> 27.0 GB, prefill +14%).
    /// 2 GiB puts the flip at ~8K tokens on 16-head/256-hd geometry, where
    /// the ~3% sync cost buys a 13.7 -> 4.3 GB transient bound — the margin
    /// that keeps a 32 GB Mac serving a 26B alive.
    const PREFILL_EVAL_TRANSIENT_BUDGET: u64 = 2 << 30;

    /// Forwards narrower than this skip the mid-loop eval cadence entirely.
    /// Spec-decode VERIFY forwards (PLD/drafter/MTP: seq 2..~9) ride the
    /// prefill layer loops because they're multi-token, but their lazy-graph
    /// transients are decode-scale (KB-MB) — the periodic eval() that bounds
    /// GB-scale prefill-chunk transients only costs them synchronous
    /// pipeline drains. Measured (qwen3.6-27B, 64 layers, cadence 4 = 16
    /// drains per MTP round): depth-1 rounds 48 ms where the AR forward is
    /// ~34 ms, and the drain tax grows superlinearly with draft depth.
    /// seq-1 decode never ran cadence evals, so exempting verify-width
    /// forwards bounds nothing that wasn't already unbounded at decode.
    pub const PREFILL_EVAL_MIN_SEQ: c_int = 32;

    /// Pure gate: does the mid-loop eval cadence apply to a forward of this
    /// width? (True = real prefill chunk; false = decode/spec-verify shape.)
    pub fn prefillEvalCadenceApplies(seq_len: c_int) bool {
        return seq_len >= PREFILL_EVAL_MIN_SEQ;
    }

    /// The cadence eval must NAME every conv_state beside `h`: the owned tail copy is
    /// not in the residual's graph, so `mlx_array_eval(h)` alone would leave it lazy
    /// and still holding its chunk parent. Re-appending an evaluated array is free.
    pub fn evalCadencePoint(h: mlx.mlx_array, ssm_entries: ?[]SSMCacheEntry) !void {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, h);
        var extra: u64 = 0;
        if (ssm_entries) |entries| {
            for (entries) |*e| {
                if (e.initialized and e.conv_state.ctx != null) {
                    _ = mlx.mlx_vector_array_append_value(vec, e.conv_state);
                    extra += 1;
                }
                if (e.initialized and e.ssm_state.ctx != null) {
                    _ = mlx.mlx_vector_array_append_value(vec, e.ssm_state);
                    extra += 1;
                }
            }
        }
        if (extra == 0) {
            try mlx.check(mlx.mlx_array_eval(h));
            return;
        }
        ssm_cache.cadence_conv_eval_count += extra;
        try mlx.check(mlx.mlx_eval(vec));
    }

    /// Decode-side intra-step async-eval ladder.
    ///
    /// A decode step builds ~40 layers of graph on the CPU before anything is
    /// evaluated, so the GPU sits idle through the build and the CPU sits idle
    /// through the execution. `mlx_async_eval` at a layer boundary hands the
    /// prefix of the graph to the GPU WITHOUT blocking, so the two overlap
    /// inside one token. It adds no op and changes no value — the same graph is
    /// evaluated in the same order, just started earlier — so output is
    /// bit-identical with the ladder on or off.
    ///
    /// This is distinct from the BLOCKING prefill cadence above, which exists
    /// to bound transient memory on wide forwards; the ladder is additive and
    /// never replaces it.
    ///
    /// DEFAULT OFF — measured, not assumed. The mlxfast-challenge tree gets
    /// +9.7% from exactly this (their `DARKBLOOM_DECODE_ASYNC_STAGE`), but
    /// their worker cannot pipeline BETWEEN steps: its protocol is one token
    /// per request, so the ladder is the only overlap it can buy. sushi
    /// already overlaps at the step boundary (`generate.zig`'s build-next →
    /// async_eval → resolve-pending), so the overlap is already collected and
    /// the ladder only adds submissions. Swept on a 40-layer MoE / M4 Max,
    /// 512-token prompt / 128 decode steps, medians of 2:
    ///   off 39.88 | stride 8 40.22 | 4 40.37 | 2 41.08 | 1 41.49 ms/token
    /// Monotonically worse as the stride tightens. Kept as a knob because the
    /// balance is per-arch (a model with a much cheaper CPU-side graph could
    /// still be build-bound), but it ships off and any future default-on needs
    /// its own A/B on the arch in question. Batched Qwen4 has a separate stride.
    ///
    /// `SUSHI_DECODE_ASYNC_LADDER` picks the boundary set:
    ///   unset / "0" / "off"  disabled (the shipped default)
    ///   "auto"               every 8th layer
    ///   "<n>"                every n-th layer
    /// Only decode-width forwards ladder: a spec-verify or prefill forward
    /// already has enough GPU work per layer to keep the device busy, and an
    /// async eval mid-verify would fight the cadence's memory bound.
    pub fn decodeAsyncLadderStride(raw: ?[]const u8) u32 {
        const v = raw orelse return 0;
        if (v.len == 0 or std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return 0;
        if (std.mem.eql(u8, v, "auto")) return DECODE_ASYNC_LADDER_AUTO;
        return std.fmt.parseInt(u32, v, 10) catch 0;
    }

    const DECODE_ASYNC_LADDER_AUTO: u32 = 8;


    /// Fire the ladder at this layer boundary, if it is one. `h` is the live
    /// residual; handing it to `mlx_async_eval` starts every op it depends on.
    pub fn ladderStep(self: *Transformer, h: mlx.mlx_array, layer_idx: usize, seq_len: c_int) void {
        ladderStepMulti(&.{h}, self.decode_async_ladder, layer_idx, seq_len);
    }

    pub fn ladderStepMulti(arrs: []const mlx.mlx_array, stride: u32, layer_idx: usize, seq_len: c_int) void {
        if (seq_len != 1 or stride == 0 or (layer_idx + 1) % stride != 0) return;
        const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        // Best-effort: a failed async eval is a missed overlap, never a wrong
        // answer — the value is computed by the step's own terminal eval.
        _ = mlx.mlx_async_eval(vec);
        if (!decode_ladder_engaged) {
            decode_ladder_engaged = true;
            log.info("[decode] async-eval ladder engaged: stride={d}\n", .{stride});
        }
    }

    /// Pure cadence pick for the prefill layer loops (standard/MoE/hybrid).
    pub fn prefillEvalCadence(
        default_cadence: u32,
        head_dim: u32,
        n_heads: u32,
        kv_heads: u32,
        chunk_len: u64,
        total_kv: u64,
        kv_dequant: bool,
    ) u32 {
        const scores: u64 = if (!prefillHeadDimFused(head_dim)) @as(u64, n_heads) * chunk_len * total_kv * 2 else 0;
        const dequant: u64 = if (kv_dequant) 2 * total_kv * @as(u64, kv_heads) * @as(u64, head_dim) * 2 else 0;
        return if (scores + dequant > PREFILL_EVAL_TRANSIENT_BUDGET) 1 else default_cadence;
    }

    /// Default forward context, routing through the Transformer's own state.
    /// Used by the single-slot legacy path and by Phase-2 prefill on a slot
    /// that has had its KVCache temporarily swapped onto the Transformer.
    pub fn defaultCtx(self: *Transformer) ForwardCtx {
        return .{
            .cache = &self.cache,
            .moe_seq_offset = &self.moe_seq_offset,
            .ssm_entries = self.ssm_entries,
            .capture_hidden = null,
            .vision_embeddings = self.vision_embeddings,
            .glm5_request = if (self.glm5_request) |*request| request else null,
        };
    }

    /// Module pointer fields that hold READ-ONLY per-model state (qwen4: the
    /// n-gram hash + mmapped table). Every per-request thing lives on the
    /// slot's `SSMCacheEntry` (GDN pair, PLE window + tokens, QSA keys +
    /// pooled blocks), so slots interleave AND batch freely. The one
    /// module-owned piece on this arch is the MTP head's cache, and the
    /// scheduler makes the SLOT that uses it exclusive (`slotExclusiveDecode`).
    /// glm5: weights only; each slot's `glm5_forward.Request` rides `ForwardCtx.glm5_request`.
    pub const module_shared_readonly_fields = [_][]const u8{ "qwen4", "expert_stream", "glm5" };

    /// Whether this model is single-flight at admission (`scheduler.modelExclusiveDecode`).
    pub fn ownsModuleDecodeState(self: *const Transformer) bool {
        // A streamed GLM load (the KLD teacher) admits one claimant per expert stream.
        if (self.glm5) |m| if (m.expert_stream != null) return true;
        return self.dsv41 != null;
    }

    pub fn sharesModuleReadonlyState(self: *const Transformer) bool {
        inline for (module_shared_readonly_fields) |f| {
            if (@field(self, f) != null) return true;
        }
        return false;
    }

    /// Spec wiring class: module archs run only their OWN MTP head
    /// (`moduleStateSpecRollback`), PLD/drafters stay off — unmeasured there.
    /// Covers both the owned and the shared-readonly lists so a shared arch
    /// does not fall onto the generic wiring and pick up PLD by default.
    pub fn moduleSpecWiring(self: *const Transformer) bool {
        return self.ownsModuleDecodeState() or self.sharesModuleReadonlyState();
    }

    /// Can this arch's MODULE-owned decode state be rolled back across a
    /// speculative verify?
    ///
    /// This is what `ownsModuleDecodeState` actually stood in for: the reason
    /// module-owned archs had every spec mode forced off is that a rejected
    /// draft tail cannot be rewound, not the ownership itself. An arch earns
    /// a true here by shipping per-position state capture plus a rewind,
    /// pinned against fresh forwards at every accepted count. dsv4 cannot:
    /// its rings and compressed caches have no per-position capture, and it
    /// has DSpark for drafting anyway.
    pub fn moduleStateSpecRollback(self: *const Transformer) bool {
        // qwen4_exp: GDN per-position capture + PLE window/token capture +
        // positional QSA key histories (`ssmRollbackFromCapture`), so the MTP
        // head's verify can rewind. dsv4 cannot. deepseek_v41 cuts its own rows
        // back to the accepted prefix (`dsv41_model.finishVerify`).
        if (self.dsv41) |m| return m.dspark != null;
        return self.qwen4_mtp != null;
    }

    /// Is this arch's MTP head the checkpoint's OWN trained layer AND measured
    /// no-worse-than-serial across the whole context ladder on this family?
    ///
    /// `defaultEnableMtp`'s `is_moe` caution is about a BOLTED-ON sidecar
    /// paying expert routing in a verify it was never designed around. It does
    /// not describe a head that ships inside the checkpoint — dsv4's DSpark
    /// stages already get the same exemption through `dsv4_stages`.
    ///
    /// The bar for adding an arch here is a MEASUREMENT, not the head's
    /// provenance: no losing cell on the context ladder against the same
    /// model's serial decode, on prompt text that is not trivially draftable.
    pub fn nativeMoeMtpHeadMeasured(self: *const Transformer) bool {
        _ = self;
        return false;
    }

    pub fn forward(self: *Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
        var ctx = self.defaultCtx();
        return self.forwardWith(&ctx, token_ids);
    }

    // Per-model forward code lives in the model files; these keep the method API.
    pub const AttnVerifyInputs = qwen4_forward.AttnVerifyInputs;
    pub const DeferredQwenMoe = qwen4_forward.DeferredQwenMoe;
    pub const HcRead = qwen4_hc.HcRead;
    pub const MimoWarmRows = mimo_forward.MimoWarmRows;
    pub const Qwen4MtpBatchRow = qwen4_forward.Qwen4MtpBatchRow;
    pub const Qwen4MtpLive = qwen4_forward.Qwen4MtpLive;
    pub const Qwen4MtpOut = qwen4_forward.Qwen4MtpOut;
    pub const Qwen4MtpOutBuf = qwen4_forward.Qwen4MtpOutBuf;
    pub const Qwen4MtpProject = qwen4_forward.Qwen4MtpProject;
    pub const Qwen4MtpState = qwen4_forward.Qwen4MtpState;
    pub const Qwen4Trace = qwen4_forward.Qwen4Trace;
    pub const VerifyRowsKind = qwen4_forward.VerifyRowsKind;
    pub const addVerifyPle = qwen4_hc.addVerifyPle;
    pub const advancePlePrev = qwen4_hc.advancePlePrev;
    pub const applyClosure = qwen4_forward.applyClosure;
    pub const attnDecodeRows = qwen4_forward.attnDecodeRows;
    pub const attnVerifyRows = qwen4_forward.attnVerifyRows;
    pub const batchedGdnReady = qwen4_forward.batchedGdnReady;
    pub const closureIn = qwen4_forward.closureIn;
    pub const closureOut = qwen4_forward.closureOut;
    pub const compileGdnGate = qwen4_forward.compileGdnGate;
    pub const compileMoeRouting = qwen4_forward.compileMoeRouting;
    pub const compileQwen4Hc = qwen4_forward.compileQwen4Hc;
    pub const computeGdnGate = qwen4_gdn.computeGdnGate;
    pub const computeMimoRouting = mimo_forward.computeMimoRouting;
    pub const conv1dWithCache = qwen4_gdn.conv1dWithCache;
    pub const discardDeferredPle = qwen4_hc.discardDeferredPle;
    pub const flushDeferredPle = qwen4_hc.flushDeferredPle;
    pub const flushDeferredPleGroup = qwen4_hc.flushDeferredPleGroup;
    pub const forwardMimoBatchedDecode = mimo_forward.forwardMimoBatchedDecode;
    pub const forwardMoeBatchedDecode = qwen4_forward.forwardMoeBatchedDecode;
    pub const forwardMoeBatchedRows = qwen4_forward.forwardMoeBatchedRows;
    pub const forwardMoeBatchedVerify = qwen4_forward.forwardMoeBatchedVerify;
    pub const forwardMoeWith = mimo_forward.forwardMoeWith;
    pub const forwardQwen4DecodeRows = qwen4_forward.forwardQwen4DecodeRows;
    pub const forwardQwen4VerifyRows = qwen4_forward.forwardQwen4VerifyRows;
    pub const forwardQwen4With = qwen4_forward.forwardQwen4With;
    pub const gatedAttnTail = qwen4_forward.gatedAttnTail;
    pub const gatedDeltaNet = qwen4_forward.gatedDeltaNet;
    pub const gatedDeltaNetProjected = qwen4_gdn.gatedDeltaNetProjected;
    pub const gatedFullAttnProjected = qwen4_forward.gatedFullAttnProjected;
    pub const gdnCheckFused = qwen4_gdn.gdnCheckFused;
    pub const gdnDecodeRows = qwen4_forward.gdnDecodeRows;
    pub const gdnGateClosureCallback = qwen4_forward.gdnGateClosureCallback;
    pub const gdnProjGpu = qwen4_gdn.gdnProjGpu;
    pub const gdnVerifyRows = qwen4_forward.gdnVerifyRows;
    pub const hcFlush = qwen4_hc.hcFlush;
    pub const hcGroupNorm = qwen4_hc.hcGroupNorm;
    pub const hcInjCallback = qwen4_hc.hcInjCallback;
    pub const hcMixCallback = qwen4_hc.hcMixCallback;
    pub const hcRead = qwen4_hc.hcRead;
    pub const hcReadFusedFor = qwen4_hc.hcReadFusedFor;
    pub const hcReadNormed = qwen4_hc.hcReadNormed;
    pub const hcReadPending = qwen4_hc.hcReadPending;
    pub const hcReadRows = qwen4_hc.hcReadRows;
    pub const hcReadStandin = qwen4_hc.hcReadStandin;
    pub const hcReadVerifyRows = qwen4_hc.hcReadVerifyRows;
    pub const hcSiluCallback = qwen4_hc.hcSiluCallback;
    pub const hcWrite = qwen4_hc.hcWrite;
    pub const hcWriteCallback = qwen4_hc.hcWriteCallback;
    pub const hcWriteOrDefer = qwen4_hc.hcWriteOrDefer;
    pub const loadQwen4Mtp = qwen4_forward.loadQwen4Mtp;
    pub const markQsaPooledRopeStale = qwen4_qsa.markQsaPooledRopeStale;
    pub const mergeSsmAcrossSlots = qwen4_ssm_group.mergeSsmAcrossSlots;
    pub const mimoAttnCore = mimo_forward.mimoAttnCore;
    pub const mimoAttnWith = mimo_forward.mimoAttnWith;
    pub const mimoBatchRowsAttn = mimo_forward.mimoBatchRowsAttn;
    pub const mimoDecodePrep = mimo_forward.mimoDecodePrep;
    pub const mimoRouterLogits = mimo_forward.mimoRouterLogits;
    pub const moeRoutingClosureCallback = qwen4_forward.moeRoutingClosureCallback;
    pub const mtpHeadForceBatched = qwen4_forward.mtpHeadForceBatched;
    pub const mtpHeadKvQuantEnabled = qwen4_forward.mtpHeadKvQuantEnabled;
    pub const mtpMoeRows = qwen4_forward.mtpMoeRows;
    pub const mtpPadSeq = qwen4_forward.mtpPadSeq;
    pub const pleClaimSpecCapture = qwen4_hc.pleClaimSpecCapture;
    pub const pleDeferrable = qwen4_hc.pleDeferrable;
    pub const pleEmbedding = qwen4_hc.pleEmbedding;
    pub const pleFillFromIds = qwen4_hc.pleFillFromIds;
    pub const pleForward = qwen4_hc.pleForward;
    pub const pleGatherBf16 = qwen4_hc.pleGatherBf16;
    pub const plePreEval = qwen4_hc.plePreEval;
    pub const plePrepareIds = qwen4_hc.plePrepareIds;
    pub const projectRows = qwen4_forward.projectRows;
    pub const qkAngleFor = qwen4_forward.qkAngleFor;
    pub const qsaAppendKeys = qwen4_qsa.qsaAppendKeys;
    pub const qsaAppendPooled = qwen4_qsa.qsaAppendPooled;
    pub const qsaBatchedAttn = qwen4_qsa.qsaBatchedAttn;
    pub const qsaBlockVisibility = qwen4_qsa.qsaBlockVisibility;
    pub const qsaChunkSelect = qwen4_qsa.qsaChunkSelect;
    pub const qsaInstallRing = qwen4_qsa.qsaInstallRing;
    pub const qsaMask = qwen4_qsa.qsaMask;
    pub const qsaMaskBatched = qwen4_qsa.qsaMaskBatched;
    pub const qsaMaskFromQk = qwen4_qsa.qsaMaskFromQk;
    pub const qsaPoolNorm = qwen4_qsa.qsaPoolNorm;
    pub const qsaPooledCosSin = qwen4_qsa.qsaPooledCosSin;
    pub const qsaPooledKeys = qwen4_qsa.qsaPooledKeys;
    pub const qsaPooledRopeComposed = qwen4_qsa.qsaPooledRopeComposed;
    pub const qsaScoreBank = qwen4_qsa.qsaScoreBank;
    pub const qsaScoreCols = qwen4_qsa.qsaScoreCols;
    pub const qsaScoreOperand = qwen4_qsa.qsaScoreOperand;
    pub const qsaSelectBlocks = qwen4_qsa.qsaSelectBlocks;
    pub const qsaSelectMaskOps = qwen4_qsa.qsaSelectMaskOps;
    pub const qsaSlotEntry = qwen4_qsa.qsaSlotEntry;
    pub const qwen4AttnProjected = qwen4_forward.qwen4AttnProjected;
    pub const qwen4AttnWith = qwen4_forward.qwen4AttnWith;
    pub const qwen4BuildDraftRerank = qwen4_forward.qwen4BuildDraftRerank;
    pub const qwen4DecodeLadderStride = qwen4_forward.qwen4DecodeLadderStride;
    pub const qwen4DraftRerankReady = qwen4_forward.qwen4DraftRerankReady;
    pub const qwen4DraftSelect = qwen4_forward.qwen4DraftSelect;
    pub const qwen4DraftSelectBatched = qwen4_forward.qwen4DraftSelectBatched;
    pub const qwen4DraftShortlist = qwen4_forward.qwen4DraftShortlist;
    pub const qwen4DraftShortlistsBatched = qwen4_forward.qwen4DraftShortlistsBatched;
    pub const qwen4MtpActivate = qwen4_forward.qwen4MtpActivate;
    pub const qwen4MtpAdopt = qwen4_forward.qwen4MtpAdopt;
    pub const qwen4MtpAdvance = qwen4_forward.qwen4MtpAdvance;
    pub const qwen4MtpApplyKvQuant = qwen4_forward.qwen4MtpApplyKvQuant;
    pub const qwen4MtpForward = qwen4_forward.qwen4MtpForward;
    pub const qwen4MtpForwardBatched = qwen4_forward.qwen4MtpForwardBatched;
    pub const qwen4MtpForwardLive = qwen4_forward.qwen4MtpForwardLive;
    pub const qwen4MtpForwardOn = qwen4_forward.qwen4MtpForwardOn;
    pub const qwen4MtpHeadKvConfig = qwen4_forward.qwen4MtpHeadKvConfig;
    pub const qwen4MtpLiveModule = qwen4_forward.qwen4MtpLiveModule;
    pub const qwen4MtpLiveState = qwen4_forward.qwen4MtpLiveState;
    pub const qwen4MtpMarkQsaLeftover = qwen4_forward.qwen4MtpMarkQsaLeftover;
    pub const qwen4MtpMarks = qwen4_forward.qwen4MtpMarks;
    pub const qwen4MtpRelease = qwen4_forward.qwen4MtpRelease;
    pub const qwen4MtpReset = qwen4_forward.qwen4MtpReset;
    pub const qwen4MtpResetOwned = qwen4_forward.qwen4MtpResetOwned;
    pub const qwen4MtpStateNew = qwen4_forward.qwen4MtpStateNew;
    pub const qwen4MtpSwapState = qwen4_forward.qwen4MtpSwapState;
    pub const qwen4MtpTruncate = qwen4_forward.qwen4MtpTruncate;
    pub const qwen4MtpTruncateLive = qwen4_forward.qwen4MtpTruncateLive;
    pub const qwen4MtpTruncateOn = qwen4_forward.qwen4MtpTruncateOn;
    pub const qwen4MtpTruncateToMarkLive = qwen4_forward.qwen4MtpTruncateToMarkLive;
    pub const qwen4VerifyRowsEligible = qwen4_forward.qwen4VerifyRowsEligible;
    pub const resetQsaPooledRope = qwen4_qsa.resetQsaPooledRope;
    pub const restoreStreamH = qwen4_forward.restoreStreamH;
    pub const splitSsmToSlots = qwen4_ssm_group.splitSsmToSlots;
    pub const ssmBindBatched = qwen4_ssm_group.ssmBindBatched;
    pub const ssmGroupDrop = qwen4_ssm_group.ssmGroupDrop;
    pub const ssmGroupRelease = qwen4_ssm_group.ssmGroupRelease;
    pub const supportsBatchedGdnDecode = qwen4_forward.supportsBatchedGdnDecode;
    pub const supportsBatchedMimoDecode = mimo_forward.supportsBatchedMimoDecode;
    pub const supportsBatchedQwen4Rows = qwen4_forward.supportsBatchedQwen4Rows;
    pub const verifyQwenMoe = qwen4_forward.verifyQwenMoe;
    pub const verifyRowsJoined = qwen4_forward.verifyRowsJoined;
    pub const warmupMimoVerify = mimo_forward.warmupMimoVerify;
    pub const yarnScaleQK = qwen4_forward.yarnScaleQK;
    pub const yarnScaleRotated = qwen4_forward.yarnScaleRotated;

    pub fn dsv41Draft(self: *Transformer, t1: u32) !dsv41_model.Draft {
        return dsv41DraftImpl(self, t1);
    }

    pub fn dsv41Verify(self: *Transformer, ctx: *ForwardCtx, ids: []const u32) !mlx.mlx_array {
        return dsv41VerifyImpl(self, ctx, ids);
    }

    pub fn dsv41Finish(self: *Transformer, ctx: *ForwardCtx, accepted: usize) !void {
        return dsv41FinishImpl(self, ctx, accepted);
    }

    pub fn forwardWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
        if (ctx.batch_slots == null) try self.ssmGroupRelease(ctx);
        if (self.glm5) |mdl| return forwardGlm5WithImpl(self, ctx, token_ids, mdl);
        if (self.dsv41) |m| return forwardDsv41With(self, ctx, token_ids, m);
        if (self.qwen4 != null) return self.forwardQwen4With(ctx, token_ids);
        return self.forwardMoeWith(ctx, token_ids);
    }

    /// Project a hidden state through the trunk's lm_head, dense (no argmax
    /// prune). DFlash draft logits come from the assistant's hidden through
    /// THIS head — the reference applies the bare Linear (no softcap /
    /// output_multiplier; both are monotone, so draft argmax is unaffected).
    pub fn lmHeadForDraft(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        if (self.glm5) |mdl| return mdl.projectHead(x);
        return self.lmHeadProject(x, false);
    }

    /// Free compiled JIT closures (compiled_gelu /
    /// compiled_geglu / compiled_softcap / compiled_moe_routing). They get
    /// bound to the calling thread's mlx GPU stream at compile time; once
    /// inference moves to a different thread (Phase 2 scheduler) calls
    /// against them fail with "no Stream(gpu, N) in current thread". Clear
    /// them here so subsequent forward calls take the unfused fallback path,
    /// then optionally re-warm on the new thread to recompile against its
    /// own stream.
    pub fn clearCompiledClosures(self: *Transformer) void {
        if (self.compiled_gelu) |c| {
            _ = mlx.mlx_closure_free(c);
            self.compiled_gelu = null;
        }
        if (self.compiled_geglu) |c| {
            _ = mlx.mlx_closure_free(c);
            self.compiled_geglu = null;
        }
        if (self.compiled_moe_routing) |c| {
            _ = mlx.mlx_closure_free(c);
            self.compiled_moe_routing = null;
        }
        if (self.compiled_gdn_gate) |c| {
            _ = mlx.mlx_closure_free(c);
            self.compiled_gdn_gate = null;
        }
        inline for (.{ &self.compiled_hc_silu, &self.compiled_hc_mix, &self.compiled_hc_inj, &self.compiled_hc_write }) |slot| {
            if (slot.*) |c| {
                _ = mlx.mlx_closure_free(c);
                slot.* = null;
            }
        }
    }

    /// Pre-fault weight pages and trigger first-touch kernel compiles before
    /// the first real request so cold prefill doesn't pay 800+ms of GPU page
    /// faulting. Runs three forward passes:
    ///   1. [1, 1] decode-shape: faults embed matrix + compiles decode kernel
    ///   2. [1, 8] prefill-shape: compiles short-prefill kernel
    /// then resets the cache so the first real request starts from clean state.
    /// Idempotent — calling twice is wasted work but not incorrect.
    pub fn warmup(self: *Transformer) !void {
        moe_dump_warmup_depth += 1;
        defer moe_dump_warmup_depth -= 1;
        const dummy_id: i32 = 0; // BOS-ish placeholder; the actual id doesn't matter for warmup
        const decode_shape = [_]c_int{ 1, 1 };
        const decode_input = mlx.mlx_array_new_data(&dummy_id, &decode_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(decode_input);
        const decode_logits = try self.forward(decode_input);
        _ = mlx.mlx_array_free(decode_logits);
        // Materialize the cache update so subsequent forwards see initialized entries.
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            for (self.cache.entries) |*entry| {
                if (!entry.initialized) continue;
                _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
                _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
            }
            _ = mlx.mlx_eval(eval_vec);
        }
        _ = mlx.mlx_clear_cache();

        // Reset before the prefill-shape pass so we exercise the cold-init path,
        // not the partial-cache path.
        try self.resetCache();

        const ids_8 = [_]i32{ 0, 0, 0, 0, 0, 0, 0, 0 };
        const prefill_shape = [_]c_int{ 1, 8 };
        const prefill_input = mlx.mlx_array_new_data(&ids_8, &prefill_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(prefill_input);
        const prefill_logits = try self.forward(prefill_input);
        _ = mlx.mlx_array_free(prefill_logits);
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            for (self.cache.entries) |*entry| {
                if (!entry.initialized) continue;
                _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
                _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
            }
            _ = mlx.mlx_eval(eval_vec);
        }
        _ = mlx.mlx_clear_cache();
        try self.resetCache();
    }

    // A throw-away slot for the spec warm-up: its own KV, SSM entries and context, so nothing
    // the warm-up forwards reaches the model's own state or the round-cost table.
    pub const SpecWarmSlot = struct {
        cache: KVCache,
        off: usize = 0,
        entries: []SSMCacheEntry,
        ctx: ForwardCtx = undefined,

        pub fn init(xfm: *Transformer, kv_config: KVQuantConfig) !*SpecWarmSlot {
            const alloc = xfm.allocator;
            const n_layers = xfm.config.num_hidden_layers;
            const sl = try alloc.create(SpecWarmSlot);
            errdefer alloc.destroy(sl);
            var cache = try KVCache.initWithConfig(alloc, n_layers, kv_config);
            errdefer cache.deinit();
            sl.* = .{ .cache = cache, .entries = try alloc.alloc(SSMCacheEntry, n_layers) };
            for (sl.entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
            sl.ctx = .{
                .cache = &sl.cache,
                .moe_seq_offset = &sl.off,
                .ssm_entries = sl.entries,
                .ssm_member_gen = nextSsmMemberGen(),
                .capture_hidden = null,
                .vision_embeddings = null,
            };
            return sl;
        }

        pub fn deinit(sl: *SpecWarmSlot, alloc: std.mem.Allocator) void {
            for (sl.entries) |*e| {
                ssmFreeSpecCapture(e);
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                ssmFreeQsaState(e);
            }
            alloc.free(sl.entries);
            sl.cache.deinit();
            alloc.destroy(sl);
        }
    };

    pub fn specWarmIds(buf: []i32) mlx.mlx_array {
        for (buf, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 40000);
        const shape = [_]c_int{ 1, @intCast(buf.len) };
        return mlx.mlx_array_new_data(buf.ptr, &shape, 2, .int32);
    }

    pub fn specWarmEvalFree(arrays: []const mlx.mlx_array) void {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (arrays) |a| {
            if (a.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, a);
        }
        _ = mlx.mlx_eval(vec);
        for (arrays) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        }
    }

    fn specWarmVerify(self: *Transformer, slots: []const *SpecWarmSlot, seq: usize) !void {
        var ids: [MAX_BATCH_ROWS][MAX_WARM_SEQ]i32 = undefined;
        var rows: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
        var ctxs: [MAX_BATCH_ROWS]*ForwardCtx = undefined;
        defer for (rows[0..slots.len]) |r| {
            _ = mlx.mlx_array_free(r);
        };
        for (slots, 0..) |sl, i| {
            rows[i] = specWarmIds(ids[i][0..seq]);
            ctxs[i] = &sl.ctx;
        }
        var logits: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
        var last: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
        var all: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
        try self.forwardRowAxisVerify(rows[0..slots.len], ctxs[0..slots.len], logits[0..slots.len], last[0..slots.len], all[0..slots.len], null, null);
        specWarmEvalFree(logits[0..slots.len]);
        specWarmEvalFree(last[0..slots.len]);
        specWarmEvalFree(all[0..slots.len]);
    }

    const MAX_WARM_SEQ: usize = MAX_WIDTH_WARM + 1;
    const MAX_WIDTH_WARM: usize = 8;

    // The verify at each draft width, the grouped verify at each row total and the head's own step
    // each JIT their Metal pipelines on first dispatch; left to the first live request that compile
    // lands inside a measured round and the table folds it as that width's price.
    pub fn warmupSpecVerify(self: *Transformer, max_width: u32, kv_config: KVQuantConfig) !void {
        moe_dump_warmup_depth += 1;
        defer moe_dump_warmup_depth -= 1;
        if (self.qwen4 == null or self.qwen4_mtp == null) return;
        var lap = io_util_mod.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
        const width_cap: usize = @min(@max(max_width, 1), MAX_WIDTH_WARM);
        const group_cap: usize = if (verifyQmmNaxAvailable()) 16 else 7;
        const max_rows: usize = @min(specWarmGroupRows(group_cap), MAX_BATCH_ROWS);

        var slots: [MAX_BATCH_ROWS]*SpecWarmSlot = undefined;
        var n_slots: usize = 0;
        defer {
            for (slots[0..n_slots]) |sl| {
                self.ssmGroupRelease(&sl.ctx) catch {};
                sl.deinit(self.allocator);
            }
            _ = mlx.mlx_clear_cache();
        }
        var head_in = mlx.mlx_array_new();
        defer if (head_in.ctx != null) {
            _ = mlx.mlx_array_free(head_in);
        };
        while (n_slots < max_rows) : (n_slots += 1) {
            slots[n_slots] = try SpecWarmSlot.init(self, kv_config);
            var prompt_ids: [8]i32 = undefined;
            const prompt = specWarmIds(&prompt_ids);
            defer _ = mlx.mlx_array_free(prompt);
            var stream = mlx.mlx_array_new();
            slots[n_slots].ctx.capture_stream_all = if (n_slots == 0) &stream else null;
            const logits = try self.forwardWith(&slots[n_slots].ctx, prompt);
            slots[n_slots].ctx.capture_stream_all = null;
            if (n_slots == 0 and stream.ctx != null) {
                const sh = mlx.getShape(stream);
                try mlx.check(mlx.mlx_slice(&head_in, stream, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ sh[0], 1, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
            }
            specWarmEvalFree(&[_]mlx.mlx_array{ logits, stream });
        }

        var w: usize = 1;
        while (w <= width_cap) : (w += 1) try self.specWarmVerify(slots[0..1], w + 1);

        var n: usize = 2;
        var group_seq_max: usize = 0;
        while (n <= n_slots) : (n += 1) {
            const per_row_cap = specWarmGroupSeq(group_cap, n, width_cap) - 1;
            group_seq_max = @max(group_seq_max, per_row_cap + 1);
            var seq: usize = 2;
            while (seq <= per_row_cap + 1) : (seq += 1) try self.specWarmVerify(slots[0..n], seq);
        }
        if (head_in.ctx != null) try self.specWarmHead(head_in);
        log.info("[spec-warmup] verify widths 1..{d} solo, group rows 2..{d} at up to {d} tokens each, head width 1 ({d} ms).\n", .{ width_cap, n_slots, group_seq_max, lap.read() / std.time.ns_per_ms });
    }



    // Mirror of `scheduler.mtpSubGroupSize`: the slots one row-axis tick can carry at this row
    // budget (NAX wide lane 16, stock lane 7) and the drafts each slot gets.
    pub fn specWarmGroupRows(group_cap: usize) usize {
        return if (group_cap >= 16) 4 else 3;
    }

    pub fn specWarmGroupSeq(group_cap: usize, rows: usize, width_cap: usize) usize {
        return @min(group_cap / rows - 1, width_cap) + 1;
    }

    fn specWarmHead(self: *Transformer, stream_row: mlx.mlx_array) !void {
        var st = try self.qwen4MtpStateNew();
        defer st.deinit();
        var id_buf: [1]i32 = undefined;
        const id_arr = specWarmIds(&id_buf);
        defer _ = mlx.mlx_array_free(id_arr);
        for ([_]Qwen4MtpProject{ .mixed_last_row, .none }, 0..) |project, i| {
            const out = try self.qwen4MtpForwardOn(&st, stream_row, id_arr, @intCast(1 + i), null, project);
            specWarmEvalFree(&[_]mlx.mlx_array{ out.logits, out.stream, out.mixed });
        }
    }

    test "spec warm-up covers exactly the group shapes the scheduler can build" {
        try testing.expectEqual(@as(usize, 4), specWarmGroupRows(16));
        try testing.expectEqual(@as(usize, 3), specWarmGroupRows(7));
        try testing.expectEqual(@as(usize, 7), specWarmGroupSeq(16, 2, 6));
        try testing.expectEqual(@as(usize, 5), specWarmGroupSeq(16, 3, 6));
        try testing.expectEqual(@as(usize, 4), specWarmGroupSeq(16, 4, 6));
        try testing.expectEqual(@as(usize, 3), specWarmGroupSeq(7, 2, 6));
        try testing.expectEqual(@as(usize, 2), specWarmGroupSeq(7, 3, 6));
        try testing.expectEqual(@as(usize, 3), specWarmGroupSeq(16, 2, 2));
    }

    /// Variant of `forwardWith` that overrides `ctx.capture_hidden` for this
    /// call only (saved and restored on exit). Used by per-slot generators
    /// (Phase 2) so the capture target is request-local without mutating
    /// shared state on the ctx.
    pub fn forwardWithCapture(
        self: *Transformer,
        ctx: *ForwardCtx,
        token_ids: mlx.mlx_array,
        out_hidden: *mlx.mlx_array,
    ) !mlx.mlx_array {
        const saved = ctx.capture_hidden;
        ctx.capture_hidden = out_hidden;
        defer ctx.capture_hidden = saved;
        return self.forwardWith(ctx, token_ids);
    }

    /// Like `forwardWithCapture` but ALSO captures the full `[B, L, H]`
    /// post-final-norm hidden (all positions) into `out_hidden_all`. The MTP
    /// verify forward needs the last-position hidden (next round's h_prev)
    /// and every position's hidden (committed-history re-append) in one pass.
    pub fn forwardWithCaptureAll(
        self: *Transformer,
        ctx: *ForwardCtx,
        token_ids: mlx.mlx_array,
        out_hidden: *mlx.mlx_array,
        out_hidden_all: *mlx.mlx_array,
    ) !mlx.mlx_array {
        const saved = ctx.capture_hidden;
        const saved_all = ctx.capture_hidden_all;
        ctx.capture_hidden = out_hidden;
        ctx.capture_hidden_all = out_hidden_all;
        defer {
            ctx.capture_hidden = saved;
            ctx.capture_hidden_all = saved_all;
        }
        return self.forwardWith(ctx, token_ids);
    }


    pub const VerifyLaps = struct {
        build_ns: u64 = 0,
        ple_ns: u64 = 0,
        ple_sync_ns: u64 = 0,
    };

    pub const VerifyRollback = struct {
        const KvMark = struct { offset: usize, initialized: bool };

        allocator: std.mem.Allocator,
        xfm: *Transformer,
        ctx: *ForwardCtx,
        kv_marks: []KvMark,
        cache_step: usize,
        ssm: []SSMCacheEntrySnapshot,
        offset: usize,

        pub fn capture(xfm: *Transformer, ctx: *ForwardCtx) !@This() {
            const marks = try xfm.allocator.alloc(KvMark, ctx.cache.entries.len);
            errdefer xfm.allocator.free(marks);
            for (ctx.cache.entries, marks) |entry, *mark| mark.* = .{ .offset = entry.offset, .initialized = entry.initialized };
            const entries = ctx.ssm_entries orelse &.{};
            const states = try xfm.allocator.alloc(SSMCacheEntrySnapshot, entries.len);
            for (entries, states) |*entry, *state| state.* = ssmSnapshot(entry);
            return .{ .allocator = xfm.allocator, .xfm = xfm, .ctx = ctx, .kv_marks = marks, .cache_step = ctx.cache.step, .ssm = states, .offset = ctx.moe_seq_offset.* };
        }

        pub fn restore(self: *@This()) void {
            for (self.ctx.cache.entries, self.kv_marks) |*current, saved| {
                if (!saved.initialized) {
                    freeKVEntry(current);
                    current.* = newEmptyKVEntry();
                } else {
                    // As with KVCache.restore, the next update rebuilds views from the retained prefix.
                    inline for (.{ "key_view", "value_view", "key_scales_view", "key_biases_view", "value_scales_view", "value_biases_view" }) |field| {
                        _ = mlx.mlx_array_free(@field(current, field));
                        @field(current, field) = mlx.mlx_array_new();
                    }
                }
                current.offset = saved.offset;
            }
            self.ctx.cache.step = self.cache_step;
            self.ctx.moe_seq_offset.* = self.offset;
            if (self.ctx.ssm_entries) |entries| {
                for (entries, self.ssm) |*entry, *saved| {
                    _ = mlx.mlx_array_free(entry.conv_state);
                    _ = mlx.mlx_array_free(entry.ssm_state);
                    ssmFreeSpecCapture(entry);
                    ssmFreeQsaState(entry);
                    entry.conv_state = saved.conv_state;
                    entry.ssm_state = saved.ssm_state;
                    entry.aux_state = saved.aux_state;
                    entry.qsa_pooled = saved.qsa_pooled;
                    saved.conv_state = .{ .ctx = null };
                    saved.ssm_state = .{ .ctx = null };
                    saved.aux_state = .{ .ctx = null };
                    saved.qsa_pooled = .{ .ctx = null };
                    entry.initialized = saved.initialized;
                    entry.qsa_ratio = saved.qsa_ratio;
                    entry.qsa_hist_rows = saved.qsa_rows;
                    entry.ple_prev = saved.ple_prev;
                    entry.ple_prev_valid = saved.ple_prev_valid;
                    if (entry.aux_state.ctx != null) entry.qsa_key_rows = mlx.getShape(entry.aux_state)[1];
                    if (entry.qsa_pooled.ctx != null) entry.qsa_pooled_blocks = mlx.getShape(entry.qsa_pooled)[1];
                }
            }
            self.ctx.ssm_member_gen = nextSsmMemberGen();
            self.xfm.fwd_gen +%= 1;
        }

        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.kv_marks);
            for (self.ssm) |*state| ssmSnapshotDeinit(state);
            self.allocator.free(self.ssm);
        }
    };

    pub fn forwardRowAxisVerify(
        self: *Transformer,
        token_rows: []const mlx.mlx_array,
        ctxs: []const *ForwardCtx,
        out_logits: []mlx.mlx_array,
        out_last: []mlx.mlx_array,
        out_all: []mlx.mlx_array,
        row_ns: ?[]u64,
        rollback_out: ?[]VerifyRollback,
    ) !void {
        std.debug.assert(token_rows.len == ctxs.len);
        std.debug.assert(out_logits.len >= token_rows.len and out_last.len >= token_rows.len and out_all.len >= token_rows.len);
        if (token_rows.len > 1 and !kv_cache.row_axis_verify_logged) {
            kv_cache.row_axis_verify_logged = true;
            log.info("[batched] row-axis mtp verify engaged (slots={d}, width={d})\n", .{ token_rows.len, mlx.getShape(token_rows[0])[1] });
        }
        if (row_ns) |ns| @memset(ns[0..token_rows.len], 0);
        self.verify_laps = .{};
        self.releaseJoinedVerifyLogits();
        var rollback: [MAX_BATCH_ROWS]VerifyRollback = undefined;
        var n_rollback: usize = 0;
        defer for (rollback[0..n_rollback]) |*saved| saved.deinit();
        errdefer for (rollback[0..n_rollback]) |*saved| saved.restore();
        if (self.qwen4 != null and token_rows.len > 1) {
            if (token_rows.len > MAX_BATCH_ROWS) return error.MtpBatchTooWide;
            for (ctxs) |ctx| try self.ssmGroupRelease(ctx);
            for (ctxs, 0..) |ctx, i| {
                var snapshot_lap = io_util_mod.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
                rollback[i] = try VerifyRollback.capture(self, ctx);
                n_rollback += 1;
                if (row_ns) |ns| ns[i] = snapshot_lap.read();
            }
        }
        var lap = io_util_mod.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
        var built: usize = 0;
        var touched: usize = 0;
        errdefer {
            for (0..built) |i| {
                _ = mlx.mlx_array_free(out_logits[i]);
                _ = mlx.mlx_array_free(out_last[i]);
                _ = mlx.mlx_array_free(out_all[i]);
            }
            for (0..touched) |i| {
                self.discardDeferredPle(ctxs[i]);
                if (ctxs[i].ssm_entries) |entries| for (entries) |*e| ssmFreeSpecCapture(e);
            }
        }
        if (self.qwen4VerifyRowsEligible(token_rows, ctxs)) {
            const Once = struct {
                var logged = false;
            };
            if (!Once.logged) {
                Once.logged = true;
                log.info("[batched] row-axis mtp verify grouped MoE engaged (slots={d})\n", .{token_rows.len});
            }
            for (ctxs) |ctx| {
                ctx.capture_ssm_seq = true;
                ctx.ple_defer = true;
            }
            touched = token_rows.len;
            {
                defer for (ctxs) |ctx| {
                    ctx.ple_defer = false;
                    ctx.capture_ssm_seq = false;
                };
                try self.forwardQwen4VerifyRows(token_rows, ctxs, out_logits, out_last, out_all);
            }
            built = token_rows.len;
            const group_ns = lap.read();
            self.verify_laps.build_ns += group_ns;
            if (row_ns) |ns| {
                for (ns[0..token_rows.len]) |*value| value.* += group_ns;
            }
            lap.reset();
        } else {
            for (token_rows, ctxs, 0..) |row, ctx, i| {
                touched = i + 1;
                var last = mlx.mlx_array_new();
                var all = mlx.mlx_array_new();
                // Per-position SSM capture lets a partial accept roll back without a re-forward;
                // the deferred PLE leaf keeps qwen4_exp's host gather out of the graph build.
                ctx.capture_ssm_seq = ctx.ssm_entries != null;
                ctx.ple_defer = true;
                const logits = self.forwardWithCaptureAll(ctx, row, &last, &all) catch |e| {
                    ctx.ple_defer = false;
                    ctx.capture_ssm_seq = false;
                    _ = mlx.mlx_array_free(last);
                    _ = mlx.mlx_array_free(all);
                    return e;
                };
                ctx.ple_defer = false;
                ctx.capture_ssm_seq = false;
                out_logits[i] = logits;
                out_last[i] = last;
                out_all[i] = all;
                built = i + 1;
                const row_build_ns = lap.read();
                self.verify_laps.build_ns += row_build_ns;
                lap.reset();
                if (row_ns) |ns| ns[i] += row_build_ns;
            }
        }
        // Fills the deferred PLE leaves and advances every row's n-gram history: after EVERY
        // row's build, before the first eval downstream of a leaf and before any rollback.
        try self.flushDeferredPleGroup(ctxs[0..token_rows.len]);
        const ple_ns = lap.read();
        self.verify_laps.ple_ns = ple_ns;
        if (row_ns) |ns| ns[0] += ple_ns;
        if (rollback_out) |out| {
            @memcpy(out[0..n_rollback], rollback[0..n_rollback]);
            n_rollback = 0;
        }
    }

    /// Drop the published joined verify block. Called before every row-axis
    /// verify and at teardown; the accept side takes ownership when it wants it.
    pub fn releaseJoinedVerifyLogits(self: *Transformer) void {
        if (self.verify_joined_logits.ctx == null) return;
        _ = mlx.mlx_array_free(self.verify_joined_logits);
        self.verify_joined_logits = .{ .ctx = null };
    }

    /// Hand the joined verify block to the caller (null when the projection ran
    /// per row).
    pub fn takeJoinedVerifyLogits(self: *Transformer) mlx.mlx_array {
        const out = self.verify_joined_logits;
        self.verify_joined_logits = .{ .ctx = null };
        return out;
    }

    /// Row `i` of `[N, S, V]` logits as an owned `[1, S, V]` array, for every row.
    pub fn sliceBatchRows(allocator: std.mem.Allocator, s: mlx.mlx_stream, arr: mlx.mlx_array, n: usize) ![]mlx.mlx_array {
        const shape = mlx.getShape(arr);
        const out = try allocator.alloc(mlx.mlx_array, n);
        var built: usize = 0;
        errdefer {
            for (out[0..built]) |a| _ = mlx.mlx_array_free(a);
            allocator.free(out);
        }
        for (out, 0..) |*slot, i| {
            const i_c: c_int = @intCast(i);
            const start = [_]c_int{ i_c, 0, 0 };
            const stop = [_]c_int{ i_c + 1, shape[1], shape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            slot.* = mlx.mlx_array_new();
            built = i + 1;
            try mlx.check(mlx.mlx_slice(slot, arr, &start, 3, &stop, 3, &strides, 3, s));
        }
        return out;
    }

    pub fn concatAxis0(s: mlx.mlx_stream, parts: []const mlx.mlx_array) !mlx.mlx_array {
        return ssmConcatAxis0(s, parts);
    }

    pub fn supportsBatchedGlmRows(self: *const Transformer) bool {
        const glm = self.glm5 orelse return false;
        return self.config.supportsBatchedGlmRows() and glm.expert_stream == null;
    }



    // Pads each slot's dense KV view (shape [1, kv_h, kv_len_i, head_dim]) to a
    // common kv_max along axis 2 with bf16 zeros and concatenates axis 0 →
    // [N, kv_h, kv_max, head_dim]. Views come from KVCache.denseView so quant
    // schemes are already dequantized; `key_not_value` selects k or v.
    pub fn padAndStackBatchedKV(
        self: *const Transformer,
        views: []const DenseKVView,
        key_not_value: bool,
        kv_max: c_int,
    ) !mlx.mlx_array {
        const pad_axes = [_]c_int{2};
        const pad_value = bf16Scalar(0.0, self.s);
        defer _ = mlx.mlx_array_free(pad_value);

        const padded_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(padded_vec);
        // Track padded arrays so we can free them after the concat.
        var padded_arrs = try self.allocator.alloc(mlx.mlx_array, views.len);
        defer {
            for (padded_arrs) |a| _ = mlx.mlx_array_free(a);
            self.allocator.free(padded_arrs);
        }

        for (views, 0..) |dv, i| {
            const view = if (key_not_value) dv.k else dv.v;
            const vshape = mlx.getShape(view);
            const klen: c_int = vshape[2];
            const high_pad: c_int = kv_max - klen;
            const low_pad_arr = [_]c_int{0};
            const high_pad_arr = [_]c_int{high_pad};
            padded_arrs[i] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_pad(
                &padded_arrs[i],
                view,
                &pad_axes,
                1,
                &low_pad_arr,
                1,
                &high_pad_arr,
                1,
                pad_value,
                "constant",
                self.s,
            ));
            _ = mlx.mlx_vector_array_append_value(padded_vec, padded_arrs[i]);
        }

        var stacked = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&stacked, padded_vec, 0, self.s));
        return stacked;
    }

    // Builds the additive per-slot decode mask [N,1,1,kv_max] in bf16 where
    // valid columns are 0 and out-of-range columns are -inf. Computed via
    // broadcasting: positions[1,1,1,kv_max] < kv_lens[N,1,1,1].
    /// `[N, 1, S, kv_max]` additive mask: row j of slot n sees positions below
    /// `kv_len_n - S + 1 + j` (its own causal prefix; the pad past `kv_len_n` never).
    pub fn buildBatchedDecodeMask(
        self: *const Transformer,
        kv_lens: []const i32,
        kv_max: c_int,
        seq_len: c_int,
    ) !mlx.mlx_array {
        const N: c_int = @intCast(kv_lens.len);
        var positions = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(positions);
        try mlx.check(mlx.mlx_arange(&positions, 0, @floatFromInt(kv_max), 1, .int32, self.s));
        const pos_shape = [_]c_int{ 1, 1, 1, kv_max };
        var pos_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos_4d);
        try mlx.check(mlx.mlx_reshape(&pos_4d, positions, &pos_shape, 4, self.s));

        // limit[n, 0, j, 0] = kv_len_n - S + 1 + j
        var lim_buf: [MAX_BATCH_ROWS * 16]i32 = undefined;
        if (kv_lens.len * @as(usize, @intCast(seq_len)) > lim_buf.len) return error.BatchedMaskTooWide;
        for (kv_lens, 0..) |kl, n| {
            for (0..@intCast(seq_len)) |j| lim_buf[n * @as(usize, @intCast(seq_len)) + j] = kl - seq_len + 1 + @as(i32, @intCast(j));
        }
        const lim_shape = [_]c_int{ N, 1, seq_len, 1 };
        const lens_4d = mlx.mlx_array_new_data(lim_buf[0 .. kv_lens.len * @as(usize, @intCast(seq_len))].ptr, &lim_shape, 4, .int32);
        defer _ = mlx.mlx_array_free(lens_4d);

        var valid = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(valid);
        try mlx.check(mlx.mlx_less(&valid, pos_4d, lens_4d, self.s));

        const zero = bf16Scalar(0.0, self.s);
        defer _ = mlx.mlx_array_free(zero);
        const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
        defer _ = mlx.mlx_array_free(neg_inf);
        var mask = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&mask, valid, zero, neg_inf, self.s));
        return mask;
    }

    // ── qwen4_exp (Qwen3.8-Flash-Next) ──



    pub fn verifyFeatureEnabled(self: *const Transformer, comptime feature: SingleVerifyFeature, grouped: bool, batch: c_int, width: c_int) bool {
        if (self.qwen4 == null) return false;
        if (grouped) return true;
        if (@import("builtin").is_test and !single_verify_test_features[@backingInt(feature)]) return false;
        // Capture marks speculative verification. A short prefill tail has the
        // same shape but must keep the ordinary forward path and shared expert.
        return self.spec_capture_ssm and batch == 1 and width >= 2 and width <= 8;
    }
















    pub var qsa_engaged_logged: bool = false;

    /// The `[half]` f32 inverse-frequency spectrum this trunk ropes with. Caller owns the result.
    pub fn ropeInvFreq(self: *Transformer, rope_dims: c_int, theta: f32) !mlx.mlx_array {
        const half = @divExact(rope_dims, 2);
        // YaRN: the load-time host spectrum (f64 → the tensor dtype), the SAME
        // table attention and the M-RoPE tables rotate with. Deriving inv_freq
        // from `theta` in-graph below is the unscaled rope only.
        if (self.yarnActive()) {
            const yf = self.yarn_inv_freq.?;
            std.debug.assert(half == @as(c_int, @intCast(yf.len)));
            const f32s = try self.allocator.alloc(f32, yf.len);
            defer self.allocator.free(f32s);
            for (yf, 0..) |v, i| f32s[i] = @floatCast(v);
            const shape = [_]c_int{@intCast(yf.len)};
            return mlx.mlx_array_new_data(f32s.ptr, &shape, 1, .float32);
        }
        var j = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(j);
        try mlx.check(mlx.mlx_arange(&j, 0, @floatFromInt(half), 1.0, .float32, self.s));
        // inv_freq_j = theta^(-2j/rope_dims) = exp(-j * 2 ln(theta) / rope_dims)
        const neg = mlx.mlx_array_new_float(-2.0 * @log(theta) / @as(f32, @floatFromInt(rope_dims)));
        defer _ = mlx.mlx_array_free(neg);
        var jl = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(jl);
        try mlx.check(mlx.mlx_multiply(&jl, j, neg, self.s));
        var inv_freq = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_exp(&inv_freq, jl, self.s)); // [half]
        return inv_freq;
    }

    /// Partial neox RoPE of `x [B,1,N,hd]` at positions `base + step*i` with explicit cos/sin.
    pub fn ropeAtPositions(self: *Transformer, x: mlx.mlx_array, rope_dims: c_int, theta: f32, base: f32, step: f32, n: c_int) !mlx.mlx_array {
        return self.ropeAtFreqs(x, rope_dims, try self.ropeInvFreq(rope_dims, theta), base, step, n); // takes ownership
    }

    /// The shared tail of `ropeAtPositions`: rotate `x` at positions
    /// `base + step*i` using the given inv_freq array (`[half]`, f32).
    fn ropeAtFreqs(self: *Transformer, x: mlx.mlx_array, rope_dims: c_int, inv_freq: mlx.mlx_array, base: f32, step: f32, n: c_int) !mlx.mlx_array {
        const cs = try self.ropeCosSinFromFreqs(rope_dims, inv_freq, base, step, n, mlx.mlx_array_dtype(x));
        defer _ = mlx.mlx_array_free(cs.cos);
        defer _ = mlx.mlx_array_free(cs.sin);
        return self.ropeApplyCosSin(x, rope_dims, cs.cos, cs.sin);
    }

    /// The build half of `ropeAtFreqs`: cos/sin `[n, rope_dims]` for positions `base + step*i`. `inv_freq` is taken.
    pub fn ropeCosSinFromFreqs(self: *Transformer, rope_dims: c_int, inv_freq: mlx.mlx_array, base: f32, step: f32, n: c_int, dt: mlx.mlx_dtype) !MropeCosSin {
        defer _ = mlx.mlx_array_free(inv_freq);
        std.debug.assert(mlx.getShape(inv_freq)[0] == @divExact(rope_dims, 2));
        var pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos);
        try mlx.check(mlx.mlx_arange(&pos, base, base + step * @as(f32, @floatFromInt(n)) - step * 0.5, step, .float32, self.s)); // [n]
        var pos2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos2);
        try mlx.check(mlx.mlx_reshape(&pos2, pos, &[_]c_int{ n, 1 }, 2, self.s));
        var ang = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ang);
        try mlx.check(mlx.mlx_multiply(&ang, pos2, inv_freq, self.s)); // [n, half]
        const parts = [_]mlx.mlx_array{ ang, ang };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var emb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(emb);
        try mlx.check(mlx.mlx_concatenate_axis(&emb, vec, 1, self.s)); // [n, rd]
        var cos_f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cos_f);
        try mlx.check(mlx.mlx_cos(&cos_f, emb, self.s));
        var sin_f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sin_f);
        try mlx.check(mlx.mlx_sin(&sin_f, emb, self.s));
        var cosv = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(cosv);
        try mlx.check(mlx.mlx_astype(&cosv, cos_f, dt, self.s));
        var sinv = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(sinv);
        try mlx.check(mlx.mlx_astype(&sinv, sin_f, dt, self.s));
        if (self.yarnActive()) {
            const ms = try scalarOf(self.config.yarn_attention_factor, dt, self.s);
            defer _ = mlx.mlx_array_free(ms);
            var cos_s = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&cos_s, cosv, ms, self.s));
            _ = mlx.mlx_array_free(cosv);
            cosv = cos_s;
            var sin_s = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&sin_s, sinv, ms, self.s));
            _ = mlx.mlx_array_free(sinv);
            sinv = sin_s;
        }
        return .{ .cos = cosv, .sin = sinv };
    }

    /// The apply half of `ropeAtFreqs`: rotate the first `rope_dims` dims with borrowed cos/sin.
    pub fn ropeApplyCosSin(self: *Transformer, x: mlx.mlx_array, rope_dims: c_int, cosv: mlx.mlx_array, sinv: mlx.mlx_array) !mlx.mlx_array {
        const half = @divExact(rope_dims, 2);
        const sh = mlx.getShape(x);
        const strides = [_]c_int{ 1, 1, 1, 1 };
        var xr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xr);
        try mlx.check(mlx.mlx_slice(&xr, x, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ sh[0], sh[1], sh[2], rope_dims }, 4, &strides, 4, self.s));
        var xp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xp);
        try mlx.check(mlx.mlx_slice(&xp, x, &[_]c_int{ 0, 0, 0, rope_dims }, 4, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, 4, &strides, 4, self.s));
        var x1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x1);
        try mlx.check(mlx.mlx_slice(&x1, xr, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ sh[0], sh[1], sh[2], half }, 4, &strides, 4, self.s));
        var x2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x2);
        try mlx.check(mlx.mlx_slice(&x2, xr, &[_]c_int{ 0, 0, 0, half }, 4, &[_]c_int{ sh[0], sh[1], sh[2], rope_dims }, 4, &strides, 4, self.s));
        var nx2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(nx2);
        try mlx.check(mlx.mlx_negative(&nx2, x2, self.s));
        const rparts = [_]mlx.mlx_array{ nx2, x1 };
        const rvec = mlx.mlx_vector_array_new_data(&rparts, 2);
        defer _ = mlx.mlx_vector_array_free(rvec);
        var rot = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rot);
        try mlx.check(mlx.mlx_concatenate_axis(&rot, rvec, -1, self.s));
        var a = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a);
        try mlx.check(mlx.mlx_multiply(&a, xr, cosv, self.s));
        var b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(b);
        try mlx.check(mlx.mlx_multiply(&b, rot, sinv, self.s));
        var roped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(roped);
        try mlx.check(mlx.mlx_add(&roped, a, b, self.s));
        const oparts = [_]mlx.mlx_array{ roped, xp };
        const ovec = mlx.mlx_vector_array_new_data(&oparts, 2);
        defer _ = mlx.mlx_vector_array_free(ovec);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&out, ovec, -1, self.s));
        return out;
    }

















    const ExpertStreamParts = struct { engine: *expert_stream_mod.Engine, imatrix: ?*imatrix_capture.Collector };

    /// The ONE streamed expert engine of a load, keyed by the config's layout.
    pub fn newExpertEngine(allocator: std.mem.Allocator, config: *const ModelConfig, s: mlx.mlx_stream) !*expert_stream_mod.Engine {
        const engine = try allocator.create(expert_stream_mod.Engine);
        errdefer allocator.destroy(engine);
        const dir = config.expert_source_dir orelse return error.MissingExpertSourceDir;
        // The engine owns the drop for its life; nothing else lets go of it.
        const drop: ?*expert_usage.Drop = if (config.drop_expert == 0) null else blk: {
            if (!config.isDsv41()) return error.LaunchFlagUnsupported;
            const d = try allocator.create(expert_usage.Drop);
            errdefer allocator.destroy(d);
            d.* = try expert_usage.load(allocator, dir, config.expertDropRequest());
            break :blk d;
        };
        errdefer if (drop) |d| {
            d.deinit();
            allocator.destroy(d);
        };
        var geometry = config.expertGeometry();
        geometry.drop = drop;
        engine.* = try expert_stream_mod.Engine.initWithOptions(allocator, dir, geometry, config.expert_cache_bytes, s, .{ .layout = config.expert_layout, .drop_owned = drop, .tile_major = config.isDsv41() and config.expert_layout == .exl3_k4 });
        return engine;
    }

    /// The streamed expert engine and the imatrix collector that rides it. Once built, the engine
    /// is deinit'd on failure: its I/O workers run until `deinit` joins them.
    fn initExpertStream(allocator: std.mem.Allocator, config: *const ModelConfig, s: mlx.mlx_stream) !ExpertStreamParts {
        const engine = try newExpertEngine(allocator, config, s);
        errdefer {
            engine.deinit();
            allocator.destroy(engine);
        }
        const im = try imatrix_capture.Collector.forModel(allocator, s, config.model_type, config.num_hidden_layers, @intCast(config.num_experts));
        return .{ .engine = engine, .imatrix = im };
    }











    /// `--mtp-head-kv-quant` opts the qwen4 MTP head's KV into `--kv-quant`; the default
    /// stays dense bf16 (the head is a lossy draft and short prompts dequantize under the
    /// fused-kernel floor). Set once at parse; the override is a test seam.
    pub var mtp_head_kv_quant_override: ?bool = null;
    pub var mtp_head_kv_quant_flag: bool = false;




    test "qwen4MtpResetOwned: a non-MTP request leaves the head untouched" {
        const t = std.testing;
        var xfm: Transformer = undefined;
        xfm.s = mlx.gpuStream();
        xfm.allocator = t.allocator;
        const head_cache = try KVCache.init(t.allocator, 1);
        var head: Qwen4Mtp = undefined;
        head.cache = head_cache;
        head.entry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
        head.qsa_marks = .{};
        head.seq_offset = 7;
        head.pos_base = 3;
        xfm.qwen4_mtp = head;
        const m = &xfm.qwen4_mtp.?;
        defer {
            ssmFreeQsaState(&m.entry);
            m.qsa_marks.deinit();
            m.cache.deinit();
        }
        xfm.qwen4MtpResetOwned(false);
        try t.expectEqual(@as(usize, 7), m.seq_offset);
        try t.expectEqual(@as(i32, 3), m.pos_base);
    }

















    pub fn mtpProjectionRows(self: *Transformer, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, lm_head: bool) !mlx.mlx_array {
        const sh = mlx.getShape(x);
        if (sh.len < 2 or sh.len > 8 or sh[0] < 1 or sh[0] > MAX_BATCH_ROWS) return error.MtpProjectionShape;
        const n: usize = @intCast(sh[0]);
        var virtual_rows: c_int = 1;
        for (sh[1 .. sh.len - 1]) |d| virtual_rows *= d;
        if (virtual_rows == 1) {
            const qp = self.quantParamsHinted(w, sc, lastDim(x));
            if (qp.mode == .affine) {
                if (try msvQmvRows(self.s, x, w, sc, bi, qp.bits, qp.group_size)) |y| return y;
            }
        }
        var parts: [MAX_BATCH_ROWS]mlx.mlx_array = @splat(.{ .ctx = null });
        defer for (parts[0..n]) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        };
        for (0..n) |row| {
            var starts: [8]c_int = @splat(0);
            var stops: [8]c_int = @splat(0);
            const strides: [8]c_int = @splat(1);
            @memcpy(stops[0..sh.len], sh);
            starts[0] = @intCast(row);
            stops[0] = @intCast(row + 1);
            var xrow = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xrow);
            try mlx.check(mlx.mlx_slice(&xrow, x, &starts, @intCast(sh.len), &stops, @intCast(sh.len), &strides, @intCast(sh.len), self.s));
            parts[row] = if (lm_head) try self.lmHeadProject(xrow, false) else try self.qmatmul(xrow, w, sc, bi);
        }
        return concatAxis0(self.s, parts[0..n]);
    }










    pub var qwen4_trace: ?*Qwen4Trace = null;







    // ── Full Attention for MoE models (with optional output gate) ──

    /// Build per-token interleaved-M-RoPE cos/sin [1,1,seq_len,rope_dims] (bf16).
    /// Positions inside the prompt come from the explicit 3-D table; positions
    /// beyond it are generated text and collapse to scalar `absolute + delta`.
    /// `positions.base` lets a suffix-only speculative KV cache map its relative
    /// offsets back to the full prompt table.
    pub fn buildMropeCosSin(self: *Transformer, positions: mrope.PositionContext, offset: usize, seq_len: usize) !MropeCosSin {
        return self.mropeCosSinAt(positions, offset, 1, seq_len, .bfloat16);
    }

    pub const MropeCosSin = struct { cos: mlx.mlx_array, sin: mlx.mlx_array };

    /// cos/sin [1,1,n,rope_dims] at the `n` absolute positions `start +
    /// stride*i`, in `dtype`. `stride 1` = a prefill chunk; the qwen4 QSA
    /// indexer ropes pooled block keys at block-START positions (`stride =
    /// ratio`) — the reference reads `full_cos[group_starts]`, which is 3-D
    /// inside an image and `abs + delta` past the prompt, exactly like the
    /// per-token table.
    pub fn mropeCosSinAt(self: *Transformer, positions: mrope.PositionContext, start: usize, stride: usize, n: usize, dtype: mlx.mlx_dtype) !MropeCosSin {
        const cfg = &self.config;
        const rope_dims: usize = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);
        const half = rope_dims / 2;

        const inv_freq = try self.allocator.alloc(f64, half);
        defer self.allocator.free(inv_freq);
        // One spectrum for every rope consumer: the YaRN table built at load
        // (when the config scales) or the plain theta^(-2j/d) one.
        if (self.yarnActive()) {
            const yf = self.yarn_inv_freq.?;
            std.debug.assert(yf.len == half);
            @memcpy(inv_freq, yf);
        } else mrope.computeInvFreq(inv_freq, rope_dims, cfg.rope_theta);
        const sel = try self.allocator.alloc(u8, half);
        defer self.allocator.free(sel);
        mrope.interleavedSelector(sel, cfg.mrope_section);

        const cos_buf = try self.allocator.alloc(f32, n * rope_dims);
        defer self.allocator.free(cos_buf);
        const sin_buf = try self.allocator.alloc(f32, n * rope_dims);
        defer self.allocator.free(sin_buf);
        mrope.fillCosSin(cos_buf, sin_buf, positions, start, stride, n, inv_freq, sel, rope_dims, yarnMscale(cfg));
        const shape = [_]c_int{ 1, 1, @intCast(n), @intCast(rope_dims) };
        const cf = mlx.mlx_array_new_data(cos_buf.ptr, &shape, 4, .float32);
        defer _ = mlx.mlx_array_free(cf);
        const sf = mlx.mlx_array_new_data(sin_buf.ptr, &shape, 4, .float32);
        defer _ = mlx.mlx_array_free(sf);
        var cos = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&cos, cf, dtype, self.s));
        errdefer _ = mlx.mlx_array_free(cos);
        var sin = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(sin);
        try mlx.check(mlx.mlx_astype(&sin, sf, dtype, self.s));
        return .{ .cos = cos, .sin = sin };
    }

    pub fn mropeContext(ctx: *const ForwardCtx) mrope.PositionContext {
        return .{ .pos = ctx.mrope_pos.?, .total = ctx.mrope_total, .delta = ctx.mrope_delta };
    }

    /// Per-prefill-chunk M-RoPE cos/sin on `ctx.mrope_cos/sin_cur`, built once
    /// per forward (freed by `endMropeChunk`). Forwards past the explicit
    /// position table (decode, multi-token spec verify) leave them null and
    /// take the scalar `offset + delta` path.
    pub fn beginMropeChunk(self: *Transformer, ctx: *ForwardCtx, offset: usize, seq_len: usize, dtype: mlx.mlx_dtype) !void {
        if (ctx.mrope_pos == null or seq_len <= 1 or offset + seq_len > ctx.mrope_total) return;
        const cs = try self.mropeCosSinAt(mropeContext(ctx), offset, 1, seq_len, dtype);
        ctx.mrope_cos_cur = cs.cos;
        ctx.mrope_sin_cur = cs.sin;
    }

    pub fn endMropeChunk(ctx: *ForwardCtx) void {
        if (ctx.mrope_cos_cur) |c| _ = mlx.mlx_array_free(c);
        if (ctx.mrope_sin_cur) |sn| _ = mlx.mlx_array_free(sn);
        ctx.mrope_cos_cur = null;
        ctx.mrope_sin_cur = null;
    }

    /// Does this trunk rotate with a YaRN-scaled rope? The DECISION comes from
    /// the config — every Transformer has one, including the hand-built fixtures
    /// in the tests, which leave the optionals below undefined. The short-circuit
    /// means an unscaled model never reads them.
    pub fn yarnActive(self: *const Transformer) bool {
        return self.config.rope_yarn and self.rope_freqs_yarn != null;
    }

    /// Apply NeoX RoPE to the first `rope_dims` dims of `arr` [B,H,S,hd] using
    /// precomputed cos/sin [1,1,S,rope_dims] (broadcast over B,H); pass remaining
    /// dims through. Equivalent to `mlx_fast_rope(traditional=false)` but with
    /// per-token (M-RoPE) angles instead of a scalar offset.
    pub fn applyMrope(self: *Transformer, arr: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, rope_dims: c_int) !mlx.mlx_array {
        const sh = mlx.getShape(arr);
        const b = sh[0];
        const h = sh[1];
        const s = sh[2];
        const hd = sh[3];
        const half = @divExact(rope_dims, 2);
        const st = [_]c_int{ 1, 1, 1, 1 };

        var rot = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rot);
        try mlx.check(mlx.mlx_slice(&rot, arr, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ b, h, s, rope_dims }, 4, &st, 4, self.s));

        // rotate_half(rot) = concat(-rot[..,half:], rot[..,:half])
        var r2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r2);
        try mlx.check(mlx.mlx_slice(&r2, rot, &[_]c_int{ 0, 0, 0, half }, 4, &[_]c_int{ b, h, s, rope_dims }, 4, &st, 4, self.s));
        var r1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r1);
        try mlx.check(mlx.mlx_slice(&r1, rot, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ b, h, s, half }, 4, &st, 4, self.s));
        var neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg);
        try mlx.check(mlx.mlx_negative(&neg, r2, self.s));
        const rh_arrs = [_]mlx.mlx_array{ neg, r1 };
        const rh_vec = mlx.mlx_vector_array_new_data(&rh_arrs, 2);
        defer _ = mlx.mlx_vector_array_free(rh_vec);
        var rh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rh);
        try mlx.check(mlx.mlx_concatenate_axis(&rh, rh_vec, -1, self.s));

        var xcos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xcos);
        try mlx.check(mlx.mlx_multiply(&xcos, rot, cos, self.s));
        var rsin = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rsin);
        try mlx.check(mlx.mlx_multiply(&rsin, rh, sin, self.s));
        var out_rot = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out_rot);
        try mlx.check(mlx.mlx_add(&out_rot, xcos, rsin, self.s));

        if (hd == rope_dims) {
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&out, out_rot, .bfloat16, self.s));
            return out;
        }
        var pass = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pass);
        try mlx.check(mlx.mlx_slice(&pass, arr, &[_]c_int{ 0, 0, 0, rope_dims }, 4, &[_]c_int{ b, h, s, hd }, 4, &st, 4, self.s));
        const cat_arrs = [_]mlx.mlx_array{ out_rot, pass };
        const cat_vec = mlx.mlx_vector_array_new_data(&cat_arrs, 2);
        defer _ = mlx.mlx_vector_array_free(cat_vec);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&out, cat_vec, -1, self.s));
        return out;
    }




    /// Multi-row variant for the hd-256 fused path at verify widths: one
    /// [rows, rd] table for positions offset..offset+rows-1, same per-forward
    /// cache (every layer of one verify round shares the table).
    ///
    /// `mscale` is the YaRN attention factor folded into the rows (1.0 = no
    /// scaling). The kernels take cos|sin rows and nothing else, and YaRN's
    /// mscale is one uniform scalar over the rotated slice, so folding it here
    /// reaches both halves of every pair — the fused path needs no mscale slot
    /// of its own. It is baked into the cached rows, which stay keyed by
    /// family: a scaled model ropes in family 1, so a scaled row and an
    /// unscaled one can never be served to each other.
    pub fn qkAngleRowsFor(self: *Transformer, family: usize, rd: c_int, base: mlx.mlx_optional_float, freqs: mlx.mlx_array, offset: c_int, rows: c_int, mscale: f32) !mlx.mlx_array {
        const slot = &self.qk_angle_cache[family];
        if (slot.arr.ctx != null and slot.offset == offset and slot.rd == rd and slot.rows == rows) return slot.arr;
        if (slot.arr.ctx != null) _ = mlx.mlx_array_free(slot.arr);
        slot.arr = .{ .ctx = null };
        slot.arr = try ropeAngleRows(self.s, rd, base, freqs, offset, rows, self.allocator, mscale);
        slot.offset = offset;
        slot.rd = rd;
        slot.rows = rows;
        return slot.arr;
    }

    pub const GdnVerifyInputs = struct {
        qkv: mlx.mlx_array = .{},
        z: mlx.mlx_array = .{},
        a: mlx.mlx_array = .{},
        b: mlx.mlx_array = .{},

        pub fn deinit(self: *GdnVerifyInputs) void {
            inline for (.{ "qkv", "z", "a", "b" }) |name| {
                if (@field(self, name).ctx != null) _ = mlx.mlx_array_free(@field(self, name));
            }
        }

        pub fn slice(self: *const GdnVerifyInputs, s: mlx.mlx_stream, start: c_int, width: c_int) !GdnVerifyInputs {
            var result: GdnVerifyInputs = .{};
            errdefer result.deinit();
            inline for (.{ "qkv", "z", "a", "b" }) |name| {
                const value = @field(self, name);
                try mlx.check(mlx.mlx_slice(&@field(result, name), value, &.{ 0, start, 0 }, 3, &.{ 1, start + width, mlx.getShape(value)[2] }, 3, &.{ 1, 1, 1 }, 3, s));
            }
            return result;
        }
    };




    // ── Dense MLP (SwiGLU: SiLU(gate(x)) * up(x) -> down) ──

    pub fn denseMLP(self: *Transformer, x: mlx.mlx_array, dw: *const DenseMlpWeights) !mlx.mlx_array {
        const gate = try self.qmatmul(x, dw.gate_w, dw.gate_s, dw.gate_b);
        defer _ = mlx.mlx_array_free(gate);
        const up = try self.qmatmul(x, dw.up_w, dw.up_s, dw.up_b);
        defer _ = mlx.mlx_array_free(up);
        const activated = try self.computeGeglu(gate, up);
        defer _ = mlx.mlx_array_free(activated);
        return self.qmatmul(activated, dw.down_w, dw.down_s, dw.down_b);
    }

    pub const GdnProj = struct { qkv: mlx.mlx_array, z: mlx.mlx_array };

    // ── Sparse MoE MLP ──


    fn moeRouterLogits(self: *Transformer, router_x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
        const cfg = &self.config;
        if (std.mem.eql(u8, cfg.model_type, "mimo_v2")) return self.mimoRouterLogits(router_x, mw);
        const qp = self.quantParamsHinted(mw.router_w, mw.router_s, lastDim(router_x));
        return qmatmulBits(router_x, mw.router_w, mw.router_s, mw.router_b, qp.bits, qp.group_size, qp.mode, self.s);
    }

    /// Gate/up/geglu/down of the shared expert. Only `.verify_rows` declines (null);
    /// the other primitives always project.
    pub fn moeSharedDown(self: *Transformer, comptime proj: SharedProjKind, x: mlx.mlx_array, mw: *const MoeMlpWeights, widths: []const c_int) !?mlx.mlx_array {
        const gate = (try self.moeSharedProject(proj, x, mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b, widths)) orelse return null;
        defer _ = mlx.mlx_array_free(gate);
        const up = (try self.moeSharedProject(proj, x, mw.shared_up_w, mw.shared_up_s, mw.shared_up_b, widths)) orelse return null;
        defer _ = mlx.mlx_array_free(up);
        const act = try self.computeGeglu(gate, up);
        defer _ = mlx.mlx_array_free(act);
        return self.moeSharedProject(proj, act, mw.shared_down_w, mw.shared_down_s, mw.shared_down_b, widths);
    }

    fn moeSharedProject(self: *Transformer, comptime proj: SharedProjKind, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
        return switch (proj) {
            .qmatmul => try self.qmatmul(x, w, sc, bi),
            .mtp_rows => try self.mtpProjectionRows(x, w, sc, bi, false),
            .verify_rows => (try decodeRowsProjection(self.s, x, w, sc, bi, widths)) orelse try verifySharedProjection(self.s, x, w, sc, bi, widths),
        };
    }

    pub fn moeSharedGateTail(self: *Transformer, expert_sum: mlx.mlx_array, gate_logit: mlx.mlx_array, down: mlx.mlx_array) !mlx.mlx_array {
        var gate_sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_sig);
        try mlx.check(mlx.mlx_sigmoid(&gate_sig, gate_logit, self.s));
        var shared_gated = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(shared_gated);
        try mlx.check(mlx.mlx_multiply(&shared_gated, gate_sig, down, self.s));
        var result = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(result);
        try mlx.check(mlx.mlx_add(&result, expert_sum, shared_gated, self.s));
        return result;
    }

    pub fn moeSharedGateLogit(self: *Transformer, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
        const seg_w = mw.shared_expert_gate_w.?;
        const seg_qp = self.quantParamsHinted(seg_w, mw.shared_expert_gate_s.?, lastDim(expert_x));
        return qmatmulBits(expert_x, seg_w, mw.shared_expert_gate_s.?, mw.shared_expert_gate_b.?, seg_qp.bits, seg_qp.group_size, seg_qp.mode, self.s);
    }

    fn moeSharedGated(self: *Transformer, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
        const down = (try self.moeSharedDown(.qmatmul, expert_x, mw, &.{})).?;
        defer _ = mlx.mlx_array_free(down);
        const gate_logit = try self.moeSharedGateLogit(expert_x, mw);
        defer _ = mlx.mlx_array_free(gate_logit);
        var gate_sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_sig);
        try mlx.check(mlx.mlx_sigmoid(&gate_sig, gate_logit, self.s));
        var shared_gated = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(shared_gated);
        try mlx.check(mlx.mlx_multiply(&shared_gated, gate_sig, down, self.s));
        return shared_gated;
    }

    /// `gate_logit`: the shared-expert gate's logit when the router kernel already computed it.
    fn moeAddGatedShared(self: *Transformer, expert_sum: mlx.mlx_array, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights, gate_logit: ?mlx.mlx_array) !mlx.mlx_array {
        const down = (try self.moeSharedDown(.qmatmul, expert_x, mw, &.{})).?;
        defer _ = mlx.mlx_array_free(down);
        return self.moeAddSharedDown(expert_sum, expert_x, down, mw, gate_logit);
    }

    fn moeAddSharedDown(self: *Transformer, expert_sum: mlx.mlx_array, expert_x: mlx.mlx_array, down: mlx.mlx_array, mw: *const MoeMlpWeights, gate_logit_in: ?mlx.mlx_array) !mlx.mlx_array {
        if (gate_logit_in) |logit| return self.moeSharedGateTail(expert_sum, logit, down);
        const gate_logit = try self.moeSharedGateLogit(expert_x, mw);
        defer _ = mlx.mlx_array_free(gate_logit);
        return self.moeSharedGateTail(expert_sum, gate_logit, down);
    }

    fn moeAddSharedGateRows(self: *Transformer, expert_sum: mlx.mlx_array, joined_input: mlx.mlx_array, inputs: []const mlx.mlx_array, down: mlx.mlx_array, widths: []const c_int, mw: *const MoeMlpWeights) !?mlx.mlx_array {
        if (!sharedGateRowsEligible(widths) or inputs.len != widths.len) return null;
        const gate_logit = try self.qmatmul(joined_input, mw.shared_expert_gate_w.?, mw.shared_expert_gate_s.?, mw.shared_expert_gate_b.?);
        defer _ = mlx.mlx_array_free(gate_logit);
        const result = try self.moeSharedGateTail(expert_sum, gate_logit, down);
        qwen4_forward.mtp_verify_shared_gate_rows_calls +%= 1;
        const Once = struct {
            var logged = false;
        };
        if (!Once.logged) {
            Once.logged = true;
            log.info("[batched] shared-expert gate tail engaged (slots={d})\n", .{inputs.len});
        }
        return result;
    }

    fn moeSharedDownRows(self: *Transformer, joined_input: mlx.mlx_array, widths: []const c_int, mw: *const MoeMlpWeights) !?mlx.mlx_array {
        const down = (try self.moeSharedDown(.verify_rows, joined_input, mw, widths)) orelse return null;
        qwen4_forward.mtp_verify_shared_rows_calls +%= 1;
        const Once = struct {
            var logged = false;
        };
        if (!Once.logged) {
            Once.logged = true;
            log.info("[batched] shared-expert verify projections engaged (slots={d})\n", .{widths.len});
        }
        return down;
    }

    fn moeVerifyRowsSolo(self: *Transformer, inputs: []const mlx.mlx_array, mw: *const MoeMlpWeights) ![]mlx.mlx_array {
        const out = try self.allocator.alloc(mlx.mlx_array, inputs.len);
        var built: usize = 0;
        errdefer {
            for (out[0..built]) |value| _ = mlx.mlx_array_free(value);
            self.allocator.free(out);
        }
        for (inputs, 0..) |input, row| {
            out[row] = try self.moeMLP(input, mw);
            built += 1;
        }
        return out;
    }

    pub fn moeVerifyRows(self: *Transformer, inputs: []const mlx.mlx_array, mw: *const MoeMlpWeights) ![]mlx.mlx_array {
        if (self.qwen4 == null or !mlx.streamIsGpu(self.s) or !moeRouterFusedEnabled() or
            inputs.len < 2 or inputs.len > 8 or self.config.hidden_size < 32 or self.config.num_experts_per_tok >= 32 or
            mw.shared_expert_gate_w == null or qwen4Standin().moe_shared or
            self.config.num_experts == 0 or self.config.num_experts_per_tok == 0 or
            mw.expert_bias != null)
        {
            return self.moeVerifyRowsSolo(inputs, mw);
        }
        var total_tokens: u64 = 0;
        var hidden: c_int = 0;
        for (inputs) |input| {
            const shape = mlx.getShape(input);
            if (shape.len != 3 or shape[0] != 1 or shape[1] < 1) return self.moeVerifyRowsSolo(inputs, mw);
            if (hidden == 0) hidden = shape[2] else if (shape[2] != hidden) return self.moeVerifyRowsSolo(inputs, mw);
            total_tokens += @intCast(shape[1]);
        }
        const total_inds = total_tokens * self.config.num_experts_per_tok;
        if (total_inds / self.config.num_experts >= 4) return self.moeVerifyRowsSolo(inputs, mw);

        var router_parts: [MAX_BATCH_ROWS]mlx.mlx_array = @splat(.{ .ctx = null });
        defer {
            for (router_parts[0..inputs.len]) |part| {
                if (part.ctx != null) _ = mlx.mlx_array_free(part);
            }
        }
        const router_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(router_vec);
        const input_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(input_vec);
        for (inputs, 0..) |input, row| {
            router_parts[row] = try self.moeRouterLogits(input, mw);
            _ = mlx.mlx_vector_array_append_value(router_vec, router_parts[row]);
            _ = mlx.mlx_vector_array_append_value(input_vec, input);
        }
        var joined_router = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(joined_router);
        try mlx.check(mlx.mlx_concatenate_axis(&joined_router, router_vec, 1, self.s));
        var joined_input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(joined_input);
        try mlx.check(mlx.mlx_concatenate_axis(&joined_input, input_vec, 1, self.s));
        var routed = mw.*;
        routed.shared_expert_gate_w = null;
        const joined_sum = try self.moeMLP2WithRouter(joined_input, joined_input, &routed, joined_router, true, null, null);
        defer _ = mlx.mlx_array_free(joined_sum);
        var widths: [MAX_BATCH_ROWS]c_int = undefined;
        for (inputs, 0..) |input, i| widths[i] = mlx.getShape(input)[1];
        const shared_down = try self.moeSharedDownRows(joined_input, widths[0..inputs.len], mw);
        defer if (shared_down) |value| {
            _ = mlx.mlx_array_free(value);
        };
        const shared_joined = if (shared_down) |down| try self.moeAddSharedGateRows(joined_sum, joined_input, inputs, down, widths[0..inputs.len], mw) else null;
        defer if (shared_joined) |value| {
            _ = mlx.mlx_array_free(value);
        };

        const out = try self.allocator.alloc(mlx.mlx_array, inputs.len);
        var built: usize = 0;
        errdefer {
            for (out[0..built]) |value| _ = mlx.mlx_array_free(value);
            self.allocator.free(out);
        }
        var start: c_int = 0;
        for (inputs, 0..) |input, row| {
            const seq = mlx.getShape(input)[1];
            if (shared_joined) |joined| {
                out[row] = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(out[row]);
                try mlx.check(mlx.mlx_slice(&out[row], joined, &.{ 0, start, 0 }, 3, &.{ 1, start + seq, hidden }, 3, &.{ 1, 1, 1 }, 3, self.s));
                built += 1;
                start += seq;
                continue;
            }
            var expert_row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(expert_row);
            try mlx.check(mlx.mlx_slice(&expert_row, joined_sum, &.{ 0, start, 0 }, 3, &.{ 1, start + seq, hidden }, 3, &.{ 1, 1, 1 }, 3, self.s));
            out[row] = if (shared_down) |down| blk: {
                var down_row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(down_row);
                try mlx.check(mlx.mlx_slice(&down_row, down, &.{ 0, start, 0 }, 3, &.{ 1, start + seq, hidden }, 3, &.{ 1, 1, 1 }, 3, self.s));
                break :blk try self.moeAddSharedDown(expert_row, input, down_row, mw, null);
            } else try self.moeAddGatedShared(expert_row, input, mw, null);
            built += 1;
            start += seq;
        }
        return out;
    }

    pub fn moeMLP(self: *Transformer, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
        return self.moeMLP2(x, x, mw);
    }

    const MoeStreamCtx = struct {
        ctx: *ForwardCtx,
        layer: u16,
        deferred: ?*?StreamMoe = null,
    };

    const MoeRoutingOverride = struct {
        inds: mlx.mlx_array,
        norm_scores: mlx.mlx_array,
        imatrix: ?ImatrixTap = null,
        shared_gated: ?mlx.mlx_array = null,
    };

    pub fn moeMLPStreamed(self: *Transformer, ctx: *ForwardCtx, x: mlx.mlx_array, mw: *const MoeMlpWeights, layer: u16) !mlx.mlx_array {
        return self.moeMLP2WithRouter(x, x, mw, null, false, .{ .ctx = ctx, .layer = layer }, null);
    }

    pub const StreamMoe = struct {
        stream_ctx: MoeStreamCtx,
        expert_x: mlx.mlx_array,
        inds: mlx.mlx_array,
        norm_scores: mlx.mlx_array,
        ids: mlx.mlx_array,
        shared: mlx.mlx_array,
        swap_logits: mlx.mlx_array,
        result: mlx.mlx_array,
        picked: mlx.mlx_array = .{ .ctx = null },
        map: ?[]const i32,
        detail: expert_stream_mod.RouteDetail,

        fn retain(self: *const StreamMoe) !StreamMoe {
            var out = self.*;
            inline for (.{ "expert_x", "inds", "norm_scores", "ids", "shared", "swap_logits", "result", "picked" }) |name|
                @field(out, name) = .{ .ctx = null };
            errdefer out.deinit();
            inline for (.{ "expert_x", "inds", "norm_scores", "ids", "shared", "swap_logits", "result", "picked" }) |name| {
                const arr = @field(self, name);
                if (arr.ctx != null) @field(out, name) = try standinRef(arr);
            }
            out.stream_ctx.deferred = null;
            return out;
        }

        pub fn deinit(self: *StreamMoe) void {
            inline for (.{ "expert_x", "inds", "norm_scores", "ids", "shared", "swap_logits", "result", "picked" }) |name| {
                const arr = @field(self, name);
                if (arr.ctx != null) _ = mlx.mlx_array_free(arr);
            }
        }
    };




    fn swapMissedExperts(self: *Transformer, engine: *expert_stream_mod.Engine, layer: u16, occurrences: []u16, logits: mlx.mlx_array, tolerance: f32) !void {
        const experts: usize = @intCast(self.config.num_experts);
        const k: usize = @intCast(self.config.num_experts_per_tok);
        if (k == 0) return;
        const rows = occurrences.len / k;
        if (rows * k != occurrences.len or @as(usize, @intCast(mlx.mlx_array_size(logits))) != rows * experts) return;
        try mlx.check(mlx.mlx_array_eval(logits));
        const logits_ptr = mlx.mlx_array_data_float32(logits) orelse return error.ExpertLogitsUnreadable;
        const flags = try self.allocator.alloc(bool, experts * 3);
        defer self.allocator.free(flags);
        const starved = engine.pickState(layer, flags[0..experts]) orelse return;
        const swapped = expert_stream_mod.substituteMisses(occurrences, logits_ptr[0 .. rows * experts], flags[0..experts], tolerance, k, flags[experts .. experts * 2], flags[experts * 2 ..], starved);
        const first_swap = swapped > 0 and expert_swap_total_swapped == 0;
        expert_swap_total_swapped +%= swapped;
        expert_swap_total_ids +%= occurrences.len;
        if (!expert_swap_logged) {
            expert_swap_logged = true;
            log.info("[expert-swap] engaged: tolerance={d:.2} (lossy: a missed expert is replaced by a cached one within that relative probability)\n", .{tolerance});
        }
        if (swapReportDue(first_swap, expert_swap_total_ids, expert_swap_last_report)) {
            expert_swap_last_report = expert_swap_total_ids;
            log.info("[expert-swap] swapped {d} of {d} routed ids so far\n", .{ expert_swap_total_swapped, expert_swap_total_ids });
        }
    }

    fn streamedMoeResult(self: *Transformer, stream_ctx: MoeStreamCtx, expert_x: mlx.mlx_array, inds: mlx.mlx_array, norm_scores: mlx.mlx_array, router_logits: mlx.mlx_array, mw: *const MoeMlpWeights) anyerror!mlx.mlx_array {
        try self.flushDeferredPle(stream_ctx.ctx);
        const engine = self.expert_stream orelse return error.MissingExpertStream;
        var detail = expert_stream_mod.RouteDetail{};
        var sync_clock = ProfClock.init();
        var ids_i32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids_i32);
        try mlx.check(mlx.mlx_astype(&ids_i32, inds, .int32, self.s));
        var ids_contiguous = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids_contiguous);
        try mlx.check(mlx.mlx_contiguous(&ids_contiguous, ids_i32, false, self.s));
        var shared_early: ?mlx.mlx_array = null;
        defer if (shared_early) |value| {
            _ = mlx.mlx_array_free(value);
        };
        const shared_eligible = streamSharedEarlyApplies(self.config.expert_layout == .exl3_k4 and mw.shared_expert_gate_w != null and !qwen4Standin().moe_shared, mlx.getShape(inds));
        const swap_tolerance = expert_stream_mod.pick_tolerance;
        var swap_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(swap_logits);
        if (expertSwapKind(swap_tolerance, mw.expert_bias != null, self.config.moe_sigmoid_router, self.imatrix != null, mlx.getShape(inds))) |kind| if (router_logits.ctx != null) {
            try routerSwapLogits(&swap_logits, router_logits, kind, self.s);
        };
        detail.build_ns = sync_clock.lap();
        if (engine.routeProbeActive()) {
            try mlx.check(mlx.mlx_array_eval(expert_x));
            detail.x_wait_ns = sync_clock.lap();
        }
        const spec_route = if (expertSpecApplies(self.imatrix != null, mlx.getShape(inds))) try engine.specRoute(stream_ctx.layer) else null;
        var spec_picked = mlx.mlx_array{ .ctx = null };
        defer if (spec_picked.ctx != null) {
            _ = mlx.mlx_array_free(spec_picked);
        };
        var picked_slots = mlx.mlx_array{ .ctx = null };
        defer if (picked_slots.ctx != null) {
            _ = mlx.mlx_array_free(picked_slots);
        };
        if (spec_route) |route| if (swap_logits.ctx != null and streamRows(mlx.getShape(inds)) == 1) {
            const starved = mlx.mlx_array_new_data(route.starved.ptr, &[_]c_int{@intCast(route.starved.len)}, 1, .uint8);
            defer _ = mlx.mlx_array_free(starved);
            // A shape the kernel does not take just skips speculation for the lossy pick.
            if (expertPickGpu(ids_contiguous, swap_logits, route.raw, starved, swap_tolerance, self.s)) |pick| {
                spec_picked = pick.picked;
                picked_slots = pick.slots;
            } else |err| if (err != error.UnsupportedShape) return err;
        };
        if (shared_eligible or swap_logits.ctx != null) {
            // The picked ids ride the ids' command buffer, so reading them never
            // waits on the experts queued behind them.
            var early: [3]mlx.mlx_array = .{ ids_contiguous, undefined, undefined };
            var early_n: usize = 1;
            for ([_]mlx.mlx_array{ swap_logits, spec_picked }) |extra| {
                if (extra.ctx == null) continue;
                early[early_n] = extra;
                early_n += 1;
            }
            const ids_vec = mlx.mlx_vector_array_new_data(&early, early_n);
            defer _ = mlx.mlx_vector_array_free(ids_vec);
            try mlx.check(mlx.mlx_async_eval(ids_vec));
        }
        if (shared_eligible) {
            shared_early = try self.moeSharedGated(expert_x, mw);
            const early_vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{shared_early.?}, 1);
            defer _ = mlx.mlx_vector_array_free(early_vec);
            try mlx.check(mlx.mlx_async_eval(early_vec));
        }
        var spec_host: ?[]const i32 = null;
        var spec_result = mlx.mlx_array{ .ctx = null };
        defer if (spec_result.ctx != null) {
            _ = mlx.mlx_array_free(spec_result);
        };
        if (spec_route) |route| {
            var gathered = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gathered);
            if (picked_slots.ctx != null) {
                const shape = mlx.getShape(inds);
                try mlx.check(mlx.mlx_reshape(&gathered, picked_slots, shape.ptr, shape.len, self.s));
            } else try mlx.check(mlx.mlx_take(&gathered, route.slots, ids_contiguous, self.s));
            const mw_spec = slabMoeWeights(mw, &route.operands);
            spec_result = try self.moeMLP2WithRouter(expert_x, expert_x, &mw_spec, null, false, null, .{
                .inds = gathered,
                .norm_scores = norm_scores,
                .shared_gated = shared_early,
            });
            try handOffStreamedResult(spec_result, streamRows(mlx.getShape(inds)));
            spec_host = route.host;
        }
        const work = StreamMoe{
            .stream_ctx = stream_ctx,
            .expert_x = expert_x,
            .inds = inds,
            .norm_scores = norm_scores,
            .ids = ids_contiguous,
            .shared = shared_early orelse .{ .ctx = null },
            .swap_logits = swap_logits,
            .result = spec_result,
            .picked = spec_picked,
            .map = spec_host,
            .detail = detail,
        };
        // A lossy pick defers only when the GPU made it: the host pick then runs at verify.
        if (stream_ctx.deferred) |target| if (spec_host != null and (swap_logits.ctx == null or spec_picked.ctx != null)) {
            std.debug.assert(target.* == null);
            target.* = try work.retain();
            return standinRef(spec_result);
        };
        return self.finishStreamedMoe(&work, mw, null);
    }

    pub fn finishStreamedMoe(self: *Transformer, work: *const StreamMoe, mw: *const MoeMlpWeights, kept_out: ?*bool) anyerror!mlx.mlx_array {
        const stream_ctx = work.stream_ctx;
        const engine = self.expert_stream orelse return error.MissingExpertStream;
        const expert_x = work.expert_x;
        const inds = work.inds;
        const norm_scores = work.norm_scores;
        const ids_contiguous = work.ids;
        const shared_early: ?mlx.mlx_array = if (work.shared.ctx != null) work.shared else null;
        const swap_logits = work.swap_logits;
        const spec_host = work.map;
        var detail = work.detail;
        var sync_clock = ProfClock.init();
        try mlx.check(mlx.mlx_array_eval(ids_contiguous));
        detail.ids_wait_ns = sync_clock.lap();
        const ids_ptr = mlx.mlx_array_data_int32(ids_contiguous) orelse return error.ExpertIdsUnreadable;
        const id_count = mlx.mlx_array_size(ids_contiguous);
        const occurrences = try self.allocator.alloc(u16, id_count);
        defer self.allocator.free(occurrences);
        for (occurrences, 0..) |*expert, i| {
            if (ids_ptr[i] < 0 or ids_ptr[i] >= self.config.num_experts) return error.ExpertOutOfRange;
            expert.* = @intCast(ids_ptr[i]);
        }
        const routed_ids = if (spec_host != null) try self.allocator.dupe(u16, occurrences) else &[_]u16{};
        defer if (spec_host != null) self.allocator.free(routed_ids);
        if (swap_logits.ctx != null) try self.swapMissedExperts(engine, stream_ctx.layer, occurrences, swap_logits, expert_stream_mod.pick_tolerance);
        detail.read_ns = sync_clock.lap();
        engine.noteRoute(stream_ctx.layer, self.moe_layers.?[stream_ctx.layer].attn == .full, detail);
        var prepared = try engine.prepareHost(stream_ctx.layer, occurrences);
        defer prepared.deinit();
        const remapped_host = try self.allocator.alloc(i32, prepared.remapped.len);
        defer self.allocator.free(remapped_host);
        for (prepared.remapped, 0..) |slot, i| remapped_host[i] = slot;
        const inds_shape = mlx.getShape(inds);
        const topk = inds_shape[inds_shape.len - 1];
        var rows: c_int = 1;
        for (inds_shape[0 .. inds_shape.len - 1]) |d| rows *= d;
        var compute_clock = ProfClock.init();
        // Both arms index the slab by REMAPPED slot; the capture keys on the
        // routing's own global ids.
        const tap = try ImatrixTap.forLayer(self.imatrix, stream_ctx.layer, ids_contiguous, rows, topk, self.s);
        defer if (tap) |t| t.deinit();
        if (spec_host) |map| {
            var kept = false;
            if (work.picked.ctx != null) {
                // The GPU made its own lossy pick: keep it only when it is the host's pick.
                try mlx.check(mlx.mlx_array_eval(work.picked));
                const picked_ptr = mlx.mlx_array_data_int32(work.picked) orelse return error.ExpertIdsUnreadable;
                var picked: [32]u16 = undefined;
                const n = occurrences.len;
                var agree = n <= picked.len and mlx.mlx_array_size(work.picked) == n;
                if (agree) for (picked[0..n], picked_ptr[0..n], occurrences) |*dst, gpu, host| {
                    dst.* = @intCast(@max(gpu, 0));
                    if (gpu != host) agree = false;
                };
                kept = agree and prepared.quantized and expert_stream_mod.specMatches(map, picked[0..n], prepared.remapped, prepared.workspace);
            } else kept = prepared.quantized and expert_stream_mod.specMatches(map, routed_ids, prepared.remapped, prepared.workspace);
            noteExpertSpec(kept);
            if (kept) {
                if (kept_out) |flag| flag.* = true;
                engine.noteExpertCompute(stream_ctx.layer, compute_clock.lap());
                return standinRef(work.result);
            }
        }
        if (prepared.quantized) {
            const slab_inds = mlx.mlx_array_new_data(remapped_host.ptr, inds_shape.ptr, @intCast(inds_shape.len), .int32);
            defer _ = mlx.mlx_array_free(slab_inds);
            var mw_slab = slabMoeWeights(mw, &prepared.quant_raw);
            if (prepared.ragged) |bank| {
                const first = bank.views[0];
                mw_slab.exl3_ragged = bank.*;
                mw_slab.switch_gate_w = first.gate.trellis;
                mw_slab.switch_gate_s = first.gate.suh;
                mw_slab.switch_gate_b = first.gate.svh;
                mw_slab.switch_up_w = first.up.trellis;
                mw_slab.switch_up_s = first.up.suh;
                mw_slab.switch_up_b = first.up.svh;
                mw_slab.switch_down_w = first.down.trellis;
                mw_slab.switch_down_s = first.down.suh;
                mw_slab.switch_down_b = first.down.svh;
            }
            const slab_result = try self.moeMLP2WithRouter(expert_x, expert_x, &mw_slab, null, false, null, .{
                .inds = slab_inds,
                .norm_scores = norm_scores,
                .imatrix = tap,
                .shared_gated = shared_early,
            });
            errdefer _ = mlx.mlx_array_free(slab_result);
            engine.noteExpertCompute(stream_ctx.layer, compute_clock.lap());
            try handOffStreamedResult(slab_result, rows);
            return slab_result;
        }
        const remapped_inds = mlx.mlx_array_new_data(remapped_host.ptr, &[_]c_int{ rows, topk }, 2, .int32);
        defer _ = mlx.mlx_array_free(remapped_inds);
        const x_shape = mlx.getShape(expert_x);
        var x_rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_rows);
        try mlx.check(mlx.mlx_reshape(&x_rows, expert_x, &[_]c_int{ rows, x_shape[x_shape.len - 1] }, 2, self.s));
        var w_rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w_rows);
        try mlx.check(mlx.mlx_reshape(&w_rows, norm_scores, &[_]c_int{ rows, topk }, 2, self.s));
        const row_out = try streamedExpertCompute(self.s, .{
            .x_rows = x_rows,
            .gate = prepared.gate,
            .up = prepared.up,
            .down = prepared.down,
            .slab_gate_up = prepared.raw_gate_up,
            .slab_down = prepared.raw_down,
            .slots = remapped_inds,
            .weights = w_rows,
            .imatrix = tap,
        }, rows, self.expert_bf16_kernels);
        defer _ = mlx.mlx_array_free(row_out);
        var expert_sum = mlx.mlx_array_new();
        var sum_owned = true;
        errdefer if (sum_owned) {
            _ = mlx.mlx_array_free(expert_sum);
        };
        try mlx.check(mlx.mlx_reshape(&expert_sum, row_out, &[_]c_int{ x_shape[0], x_shape[1], mlx.getShape(row_out)[1] }, 3, self.s));
        if (std.mem.eql(u8, self.config.model_type, "mimo_v2") and
            mlx.mlx_array_dtype(expert_sum) != mlx.mlx_array_dtype(expert_x))
        {
            var narrowed = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&narrowed, expert_sum, mlx.mlx_array_dtype(expert_x), self.s));
            _ = mlx.mlx_array_free(expert_sum);
            expert_sum = narrowed;
        }
        if (mw.shared_expert_gate_w == null or qwen4Standin().moe_shared) {
            engine.noteExpertCompute(stream_ctx.layer, compute_clock.lap());
            try handOffStreamedResult(expert_sum, rows);
            return expert_sum;
        }
        const result = try self.moeAddGatedShared(expert_sum, expert_x, mw, null);
        _ = mlx.mlx_array_free(expert_sum);
        sum_owned = false;
        errdefer _ = mlx.mlx_array_free(result);
        engine.noteExpertCompute(stream_ctx.layer, compute_clock.lap());
        try handOffStreamedResult(result, rows);
        return result;
    }

    /// Decode-width MoE expert compute through the in-place gather-qmv kernel
    /// (`gatherQmv`), which indexes the expert bank directly instead of
    /// materializing the top-K experts. On success writes [B,S,K,hidden] into
    /// `out` and returns true; returns false when any projection is outside the
    /// kernel's supported set (mxfp4/mxfp8 bank, 3/5/6-bit affine width, odd
    /// geometry), leaving `out` untouched so the caller runs the batched take path.
    fn moeDecodeGatherQmv(
        self: *Transformer,
        out: *mlx.mlx_array,
        expert_x: mlx.mlx_array,
        inds: mlx.mlx_array,
        norm_scores: mlx.mlx_array,
        reduced: *bool,
        mw: *const MoeMlpWeights,
        gate_qp: anytype,
        up_qp: anytype,
        down_qp: anytype,
        D: c_int,
        K: c_int,
        B: c_int,
        S: c_int,
    ) !bool {
        // Routing indices arrive as int32/uint32 [.., K]; the kernel reads a
        // flat uint32 [K].
        var inds_u32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inds_u32);
        {
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const kshape = [_]c_int{K};
            try mlx.check(mlx.mlx_reshape(&flat, inds, &kshape, 1, self.s));
            try mlx.check(mlx.mlx_astype(&inds_u32, flat, .uint32, self.s));
        }
        var x_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_flat);
        const dshape = [_]c_int{D};
        try mlx.check(mlx.mlx_reshape(&x_flat, expert_x, &dshape, 1, self.s));

        // gate / up share the single token's hidden state, so one kernel can
        // do both dot products and the activation — three dispatches off the
        // chain into down_proj. Falls back to the split form on any geometry,
        // quant mode or dtype the fused kernel declines.
        var act_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act_2d);
        var inter: c_int = 0;
        if (qwen4Standin().moe_gateup and self.config.moe_intermediate_size > 0) {
            // Diagnostic: unit activations, the gate+up kernel drops out.
            const sh = [_]c_int{ K, @intCast(self.config.moe_intermediate_size) };
            _ = mlx.mlx_array_free(act_2d);
            act_2d = try standinOnes(&sh, self.s);
            inter = sh[1];
        } else if (self.config.hidden_act == .silu and
            gate_qp.bits == up_qp.bits and gate_qp.group_size == up_qp.group_size and gate_qp.mode == up_qp.mode)
        {
            if (try gatherQmvGateUp(
                self.s,
                x_flat,
                mw.switch_gate_w,
                mw.switch_gate_s,
                mw.switch_gate_b,
                mw.switch_up_w,
                mw.switch_up_s,
                mw.switch_up_b,
                inds_u32,
                gate_qp.bits,
                gate_qp.group_size,
                gate_qp.mode,
            )) |fused| {
                _ = mlx.mlx_array_free(act_2d);
                act_2d = fused; // [K, inter], activation already applied
                inter = mlx.getShape(act_2d)[1];
            }
        }
        if (inter == 0) {
            const gate_2d = try gatherQmv(self.s, x_flat, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, inds_u32, gate_qp.bits, gate_qp.group_size, gate_qp.mode, false) orelse return false;
            defer _ = mlx.mlx_array_free(gate_2d);
            const up_2d = try gatherQmv(self.s, x_flat, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, inds_u32, up_qp.bits, up_qp.group_size, up_qp.mode, false) orelse return false;
            defer _ = mlx.mlx_array_free(up_2d);

            inter = mlx.getShape(gate_2d)[1];
            const k1i = [_]c_int{ K, 1, inter };
            var gate_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_out);
            try mlx.check(mlx.mlx_reshape(&gate_out, gate_2d, &k1i, 3, self.s));
            var up_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up_out);
            try mlx.check(mlx.mlx_reshape(&up_out, up_2d, &k1i, 3, self.s));

            const expert_act = try self.computeGeglu(gate_out, up_out); // [K,1,inter]
            defer _ = mlx.mlx_array_free(expert_act);

            // down: each expert consumes ITS OWN activation → per-expert x layout.
            const ki = [_]c_int{ K, inter };
            _ = mlx.mlx_array_free(act_2d);
            act_2d = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&act_2d, expert_act, &ki, 2, self.s));
        }
        // Fused down + router weighting + K-reduction: one dispatch replaces
        // the down gather, the score multiply and the sum over K — and the
        // [K, hidden] intermediate never materializes. Declines (dtype
        // mismatch, geometry, kill switch) fall through to the split form
        // with `reduced` untouched.
        if (qwen4Standin().moe_down and @rem(D, inter) == 0) {
            // Diagnostic: the down kernel drops out; the activations are still
            // consumed (sum over K, tiled to hidden) so gate+up stays live.
            var summed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(summed);
            try mlx.check(mlx.mlx_sum_axis(&summed, act_2d, 0, false, self.s));
            const reps = [_]c_int{@divExact(D, inter)};
            var tiled = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(tiled);
            try mlx.check(mlx.mlx_tile(&tiled, summed, &reps, 1, self.s));
            const bsh_shape = [_]c_int{ B, S, D };
            try mlx.check(mlx.mlx_reshape(out, tiled, &bsh_shape, 3, self.s));
            reduced.* = true;
            return true;
        }
        {
            var scores_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(scores_flat);
            const kshape2 = [_]c_int{K};
            try mlx.check(mlx.mlx_reshape(&scores_flat, norm_scores, &kshape2, 1, self.s));
            if (try gatherQmvDownReduce(self.s, act_2d, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, inds_u32, scores_flat, down_qp.bits, down_qp.group_size, down_qp.mode)) |sum_flat| {
                defer _ = mlx.mlx_array_free(sum_flat);
                const hidden = mlx.getShape(sum_flat)[0];
                const bsh_shape = [_]c_int{ B, S, hidden };
                try mlx.check(mlx.mlx_reshape(out, sum_flat, &bsh_shape, 3, self.s));
                reduced.* = true;
                if (!downred_engaged) {
                    downred_engaged = true;
                    log.info("[moe] fused down+reduce kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, down_qp.bits, down_qp.group_size });
                }
                if (!gqmv_engaged) {
                    gqmv_engaged = true;
                    log.info("[moe] gather-qmv kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, gate_qp.bits, gate_qp.group_size });
                }
                return true;
            }
        }
        const down_2d = try gatherQmv(self.s, act_2d, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, inds_u32, down_qp.bits, down_qp.group_size, down_qp.mode, true) orelse return false;
        defer _ = mlx.mlx_array_free(down_2d);

        const hidden = mlx.getShape(down_2d)[1];
        const bskh_shape = [_]c_int{ B, S, K, hidden };
        try mlx.check(mlx.mlx_reshape(out, down_2d, &bskh_shape, 4, self.s));
        // Engagement is COUNTED, never inferred: a silent fallback to the
        // batched take path is output-identical and would look like a null win.
        if (!gqmv_engaged) {
            gqmv_engaged = true;
            log.info("[moe] gather-qmv kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, gate_qp.bits, gate_qp.group_size });
        }
        return true;
    }

    /// Each token row uses the serial expert arithmetic; output retains [B, S, hidden].
    fn moeDecodeGatherQmvRows(
        self: *Transformer,
        out: *mlx.mlx_array,
        expert_x: mlx.mlx_array,
        inds: mlx.mlx_array,
        norm_scores: mlx.mlx_array,
        reduced: *bool,
        mw: *const MoeMlpWeights,
        gate_qp: anytype,
        up_qp: anytype,
        down_qp: anytype,
        D: c_int,
        K: c_int,
        B: c_int,
        S: c_int,
    ) !bool {
        if (!gatherQmvDownReduceRowsEligible(self.config.hidden_act, gate_qp, up_qp, down_qp, D)) return false;
        var inds_u32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inds_u32);
        {
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const kshape = [_]c_int{ B * S, K };
            try mlx.check(mlx.mlx_reshape(&flat, inds, &kshape, 2, self.s));
            try mlx.check(mlx.mlx_astype(&inds_u32, flat, .uint32, self.s));
        }
        var x_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_2d);
        const dshape = [_]c_int{ B * S, D };
        try mlx.check(mlx.mlx_reshape(&x_2d, expert_x, &dshape, 2, self.s));

        const act_3d = (try gatherQmvGateUpRows(
            self.s,
            x_2d,
            mw.switch_gate_w,
            mw.switch_gate_s,
            mw.switch_gate_b,
            mw.switch_up_w,
            mw.switch_up_s,
            mw.switch_up_b,
            inds_u32,
            gate_qp.bits,
            gate_qp.group_size,
            gate_qp.mode,
        )) orelse return false;
        defer _ = mlx.mlx_array_free(act_3d);
        var scores_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scores_2d);
        const kshape2 = [_]c_int{ B * S, K };
        try mlx.check(mlx.mlx_reshape(&scores_2d, norm_scores, &kshape2, 2, self.s));
        const sum_2d = (try gatherQmvDownReduceRows(
            self.s,
            act_3d,
            mw.switch_down_w,
            mw.switch_down_s,
            mw.switch_down_b,
            inds_u32,
            scores_2d,
            down_qp.bits,
            down_qp.group_size,
            down_qp.mode,
        )) orelse return false;
        defer _ = mlx.mlx_array_free(sum_2d);
        const hidden = mlx.getShape(sum_2d)[1];
        const bsh_shape = [_]c_int{ B, S, hidden };
        try mlx.check(mlx.mlx_reshape(out, sum_2d, &bsh_shape, 3, self.s));
        reduced.* = true;
        return true;
    }

    /// MoE MLP with separate router and expert inputs.
    /// router_x: input for routing (raw hidden states).
    /// expert_x: input for expert computation (possibly normalized).
    fn moeExl3(self: *Transformer, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights, inds: mlx.mlx_array, scores: mlx.mlx_array, verify_rows: bool, shared: ?mlx.mlx_array, shared_gate: ?mlx.mlx_array) !mlx.mlx_array {
        if (verify_rows) mtp_verify_expert_rows_calls +%= 1;
        if (mw.exl3_ragged) |*rb| return sushi_exl3.moeRagged(self.s, expert_x, rb, inds, scores, self.config.expertDecode(), verify_rows, shared, shared_gate);
        const bank = sushi_exl3.Bank{
            .gate = .{ .trellis = mw.switch_gate_w, .suh = mw.switch_gate_s, .svh = mw.switch_gate_b },
            .up = .{ .trellis = mw.switch_up_w, .suh = mw.switch_up_s, .svh = mw.switch_up_b },
            .down = .{ .trellis = mw.switch_down_w, .suh = mw.switch_down_s, .svh = mw.switch_down_b },
        };
        const dec = self.config.expertDecode();
        if (shared) |value| {
            if (shared_gate) |logit| return sushi_exl3.moeWithGatedShared(self.s, expert_x, bank, inds, scores, dec, verify_rows, value, logit);
            return sushi_exl3.moeWithShared(self.s, expert_x, bank, inds, scores, dec, verify_rows, value);
        }
        return sushi_exl3.moe(self.s, expert_x, bank, inds, scores, dec, verify_rows);
    }

    pub fn moeMLP2(self: *Transformer, router_x: mlx.mlx_array, expert_x_in: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
        return self.moeMLP2WithRouter(router_x, expert_x_in, mw, null, false, null, null);
    }

    pub fn moeMLP2WithRouter(self: *Transformer, router_x: mlx.mlx_array, expert_x_in: mlx.mlx_array, mw: *const MoeMlpWeights, router_override: ?mlx.mlx_array, skip_shared: bool, stream_ctx: ?MoeStreamCtx, routing_override: ?MoeRoutingOverride) !mlx.mlx_array {
        const cfg = &self.config;
        // DIAGNOSTIC (SUSHI_DISPATCH_PROBE=N): inject N extra small
        // elementwise kernels per MoE layer and read the slope. Multiplying by
        // an exact 1.0 is output-identical for every finite value, and the
        // result FEEDS the expert path so MLX cannot elide it — the two
        // properties the "duplicate a component" anti-pattern lacks. The slope
        // is what says whether a fusion that removes k dispatches per layer is
        // worth writing: it prices one dispatch in the LIVE graph, which no
        // micro-benchmark of an isolated kernel can.
        var probe_chain = mlx.mlx_array{ .ctx = null };
        defer if (probe_chain.ctx != null) {
            _ = mlx.mlx_array_free(probe_chain);
        };
        const expert_x = blk: {
            const n = dispatchProbeCount();
            if (n == 0) break :blk expert_x_in;
            var cur = expert_x_in;
            for (0..n) |_| {
                var next = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_multiply(&next, cur, self.one, self.s));
                if (probe_chain.ctx != null) _ = mlx.mlx_array_free(probe_chain);
                probe_chain = next;
                cur = next;
            }
            break :blk cur;
        };
        // Decode MoE-internals profiler (SUSHI_DECODE_PROFILE=<rows>).
        const moe_prof = decodeProfileEnabled() and mlx.getShape(expert_x)[1] <= decodeProfileRows();
        var mclk: ProfClock = if (moe_prof) ProfClock.init() else undefined;
        // Per-expert-weight params: mixed-precision MoE checkpoints vary bits
        // (and, with non-affine modes, group size + mode) per weight — resolve
        // each individually. gate/up consume the hidden dim; down consumes the
        // expert intermediate dim.
        const gate_qp = self.quantParamsHinted(mw.switch_gate_w, mw.switch_gate_s, lastDim(expert_x));
        const up_qp = self.quantParamsHinted(mw.switch_up_w, mw.switch_up_s, lastDim(expert_x));
        const down_qp = self.quantParamsHinted(mw.switch_down_w, mw.switch_down_s, if (cfg.moe_intermediate_size > 0) cfg.moe_intermediate_size else null);

        const router_logits = if (routing_override != null)
            mlx.mlx_array{ .ctx = null }
        else if (router_override) |given| blk: {
            var owned = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&owned, given));
            break :blk owned;
        } else try self.moeRouterLogits(router_x, mw);
        defer if (router_logits.ctx != null) {
            _ = mlx.mlx_array_free(router_logits);
        };

        // Top-K + softmax/renormalize as a single fused kernel (when compiled).
        // expert_bias bound: sigmoid+bias selection instead of softmax.
        // At one decode row the shared-expert gate's logit rides the softmax router kernel.
        const gate_fold: ?RouterGate = blk: {
            if (routing_override != null or router_override != null or stream_ctx != null or skip_shared) break :blk null;
            if (mw.expert_bias != null or mw.shared_gate_vec.ctx == null or qwen4Standin().moe_shared or qwen4Standin().moe_router) break :blk null;
            if (mlx.mlx_array_size(expert_x) != mlx.mlx_array_size(mw.shared_gate_vec)) break :blk null;
            break :blk .{ .x = expert_x, .w = mw.shared_gate_vec };
        };
        const routed: MoeRouting = if (routing_override) |given| blk: {
            var ids = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(ids);
            try mlx.check(mlx.mlx_array_set(&ids, given.inds));
            var weights = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&weights, given.norm_scores));
            break :blk .{ .inds = ids, .norm_scores = weights };
        } else if (mw.expert_bias) |bias|
            try self.computeMimoRouting(router_logits, bias)
        else
            try self.computeMoeRouting(router_logits, gate_fold);
        defer if (routed.gate_logit.ctx != null) {
            _ = mlx.mlx_array_free(routed.gate_logit);
        };
        var inds = routed.inds;
        defer _ = mlx.mlx_array_free(inds);
        var norm_scores = routed.norm_scores;
        defer _ = mlx.mlx_array_free(norm_scores);
        if (routing_override == null and qwen4Standin().moe_router) {
            // Diagnostic: fixed experts 0..K-1, uniform scores (routing off the critical path).
            const ish = mlx.getShape(inds);
            const k: c_int = ish[ish.len - 1];
            var ar = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ar);
            try mlx.check(mlx.mlx_arange(&ar, 0, @floatFromInt(k), 1, mlx.mlx_array_dtype(inds), self.s));
            var fixed = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_broadcast_to(&fixed, ar, ish.ptr, @intCast(ish.len), self.s));
            var fixed_c = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_contiguous(&fixed_c, fixed, false, self.s));
            _ = mlx.mlx_array_free(fixed);
            try mlx.check(mlx.mlx_array_eval(fixed_c));
            _ = mlx.mlx_array_free(inds);
            inds = fixed_c;
            const ssh = mlx.getShape(norm_scores);
            var uni = mlx.mlx_array_new();
            const one_k = try scalarOf(1.0 / @as(f32, @floatFromInt(k)), mlx.mlx_array_dtype(norm_scores), self.s);
            defer _ = mlx.mlx_array_free(one_k);
            try mlx.check(mlx.mlx_full(&uni, ssh.ptr, @intCast(ssh.len), one_k, mlx.mlx_array_dtype(norm_scores), self.s));
            try mlx.check(mlx.mlx_array_eval(uni));
            _ = mlx.mlx_array_free(norm_scores);
            norm_scores = uni;
        }

        // The group-limited chain deliberately hands its weights back in f32
        // (a bf16 round trip after routed_scaling_factor costs real accuracy),
        // and it is the CALLER's job to land them in the activation dtype.
        // Skipping that multiplies the residual stream up to f32 at the first
        // MoE layer, which then promotes every weight on read for the rest of
        // the forward — a whole-model slowdown that shows up only as a
        // `[dtype-trace] residual widened` line.
        if (!std.mem.eql(u8, cfg.model_type, "mimo_v2") and mlx.mlx_array_dtype(norm_scores) != mlx.mlx_array_dtype(expert_x)) {
            var cast_scores = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&cast_scores, norm_scores, mlx.mlx_array_dtype(expert_x), self.s));
            _ = mlx.mlx_array_free(norm_scores);
            norm_scores = cast_scores;
        }

        if (stream_ctx) |info| return self.streamedMoeResult(info, expert_x, inds, norm_scores, router_logits, mw);
        if (moeDumpDir() != null) {
            moeDumpTensor(self.s, "x", moe_dump_layer, expert_x);
            if (router_logits.ctx != null) moeDumpTensor(self.s, "rlogits", moe_dump_layer, router_logits);
            moeDumpTensor(self.s, "ids", moe_dump_layer, inds);
            moeDumpTensor(self.s, "w", moe_dump_layer, norm_scores);
        }
        if (cfg.expert_layout == .exl3_k4) {
            if (routing_override) |given| if (given.imatrix) |tap| {
                const xs = mlx.getShape(expert_x);
                try tap.observeInput(self.s, expert_x, xs[0] * xs[1], xs[xs.len - 1]);
            };
            const shared = if (routing_override) |given| given.shared_gated else null;
            const fold_shared = shared != null and moeDumpDir() == null;
            const tail_shared = routing_override == null and !skip_shared and mw.shared_expert_gate_w != null and !qwen4Standin().moe_shared;
            if (tail_shared and moeDumpDir() == null) {
                if (try @import("qwen4/qwen4_shared.zig").folded(self.s, expert_x, mw, inds, norm_scores, self.config.expertDecode(), routed.gate_logit)) |folded| return folded;
                const down = (try self.moeSharedDown(.qmatmul, expert_x, mw, &.{})).?;
                defer _ = mlx.mlx_array_free(down);
                const own_logit = routed.gate_logit.ctx == null;
                const logit = if (own_logit) try self.moeSharedGateLogit(expert_x, mw) else routed.gate_logit;
                defer if (own_logit) {
                    _ = mlx.mlx_array_free(logit);
                };
                return self.moeExl3(expert_x, mw, inds, norm_scores, false, down, logit);
            }
            const y = try self.moeExl3(expert_x, mw, inds, norm_scores, skip_shared and router_override != null, if (fold_shared) shared else null, null);
            if (moeDumpDir() != null) {
                moeDumpTensor(self.s, "y", moe_dump_layer, y);
            }
            if (fold_shared) return y;
            if (routing_override) |given| if (given.shared_gated) |shared_gated| {
                defer _ = mlx.mlx_array_free(y);
                var summed = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(summed);
                try mlx.check(mlx.mlx_add(&summed, y, shared_gated, self.s));
                return summed;
            };
            if (skip_shared or mw.shared_expert_gate_w == null or qwen4Standin().moe_shared) return y;
            defer _ = mlx.mlx_array_free(y);
            return self.moeAddGatedShared(y, expert_x, mw, if (routed.gate_logit.ctx != null) routed.gate_logit else null);
        }

        // Expert computation. Two paths:
        //
        //   Decode (S=1): per-expert gather_qmm with `rhs_indices=inds` shape
        //   [B,S,K]. Output is [B,S,K,1,inter]; each token reads K expert
        //   blocks from random offsets, but at S=1 there are at most K unique
        //   experts so HBM scatter is bounded.
        //
        //   Multi-position (S>1): mlx-lm's `_gather_sort` flow. Flatten inds →
        //   argsort globally → `lhs_indices = order // K` selects which token
        //   row to feed each sorted slot; `rhs_indices = inds[order]` selects
        //   the expert (now sorted, so consecutive slots hit the same expert
        //   block → one HBM stream). After down_proj, an inverse permutation
        //   restores the original [B,S,K] layout. Critical for drafter verify
        //   on MoE: at block_size=4 + top_k=8 the old `total_inds >= 64`
        //   threshold left verify (32 inds) on the slow scatter path while the
        //   sorted path's argsort overhead is negligible at that size.
        if (moe_prof) {
            try mlx.check(mlx.mlx_array_eval(inds));
            try mlx.check(mlx.mlx_array_eval(norm_scores));
            decode_prof.moe_router_ns += mclk.lap();
        }

        const x_shape = mlx.getShape(expert_x);
        const B = x_shape[0];
        const S = x_shape[1];
        const D = x_shape[x_shape.len - 1];
        const inds_shape = mlx.getShape(inds);
        const K = inds_shape[inds_shape.len - 1];
        const total_inds: c_int = B * S * K;
        // A streamed QUANTIZED layer's capture arrives here (the override's ids
        // are slab slots). The fused arms never materialize the activation rows
        // the down statistic needs, so an armed capture takes the sorted arm at
        // every width.
        const imx: ?ImatrixTap = if (routing_override) |given| given.imatrix else null;
        if (imx) |tap| try tap.observeInput(self.s, expert_x, B * S, D);
        const do_sort = imx != null or B * S > 1 or total_inds >= 64;
        const no_idx = mlx.mlx_array{ .ctx = null };

        var down_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_out);
        // True when the decode fast path already produced the WEIGHTED expert
        // sum [B,S,hidden] (fused down+reduce kernel) — the multiply+sum
        // below must then be skipped, not repeated.
        var moe_reduced = false;
        var cost_arm: u8 = 4;

        const grouped_rows = skip_shared and router_override != null and B == 1 and S >= 2 and S <= 128 and moeRowsFusedEnabled() and !qwen4Standin().moe_gateup and !qwen4Standin().moe_down;
        if (imx == null and (grouped_rows or moeDecodeDispatchArm(B, S, K) == .rows) and
            useGatherQmvDecode(self, gate_qp, up_qp) and mw.switch_gate_s.ctx != null and mw.switch_up_s.ctx != null and mw.switch_down_s.ctx != null and
            try self.moeDecodeGatherQmvRows(&down_out, expert_x, inds, norm_scores, &moe_reduced, mw, gate_qp, up_qp, down_qp, D, K, B, S))
        {
            moe_rows_fused_layers +%= 1;
            if (grouped_rows) mtp_verify_expert_rows_calls +%= 1;
            cost_arm = 1;
        } else if (do_sort) {
            moe_sorted_layers +%= 1;
            cost_arm = 2;
            // ── Global-sort prefill path ──

            // Flatten inds → [N] where N = B*S*K
            const flat_shape = [_]c_int{total_inds};
            var flat_inds = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat_inds);
            try mlx.check(mlx.mlx_reshape(&flat_inds, inds, &flat_shape, 1, self.s));

            const route_pack = if (self.verifyFeatureEnabled(.routing, skip_shared, B, S)) try verifyRoutePack(self.s, flat_inds, K) else null;
            var inv_order = if (route_pack) |value| value.inverse else mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(inv_order);
            var sorted_inds = if (route_pack) |value| value.sorted else mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sorted_inds);
            var lhs_idx = if (route_pack) |value| value.lhs else mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(lhs_idx);
            if (route_pack == null) {
                var order = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(order);
                try mlx.check(mlx.mlx_argsort_axis(&order, flat_inds, 0, self.s));
                try mlx.check(mlx.mlx_argsort_axis(&inv_order, order, 0, self.s));
                try mlx.check(mlx.mlx_take_axis(&sorted_inds, flat_inds, order, 0, self.s));
                const k_arr = mlx.mlx_array_new_int(K);
                defer _ = mlx.mlx_array_free(k_arr);
                try mlx.check(mlx.mlx_floor_divide(&lhs_idx, order, k_arr, self.s));
            } else {
                mtp_verify_kernel_calls[2] +%= 1;
                const Once = struct {
                    var logged = false;
                };
                if (!Once.logged) {
                    Once.logged = true;
                    log.info("[mtp-verify] fused verifier route packing engaged\n", .{});
                }
            }

            // x_flat: [B,S,D] → [B*S, D]
            const bs_d_shape = [_]c_int{ B * S, D };
            var x_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_flat);
            try mlx.check(mlx.mlx_reshape(&x_flat, expert_x, &bs_d_shape, 2, self.s));

            var indexed_lhs: mlx.mlx_array = .{};
            defer if (indexed_lhs.ctx != null) {
                _ = mlx.mlx_array_free(indexed_lhs);
            };
            const input_view = if (self.verifyFeatureEnabled(.routing, skip_shared, B, S) and
                gate_qp.bits == 4 and up_qp.bits == 4 and gate_qp.group_size == 64 and up_qp.group_size == 64 and
                gate_qp.mode == .affine and up_qp.mode == .affine and
                mlx.getShape(mw.switch_gate_w)[0] == mlx.getShape(mw.switch_up_w)[0])
            blk: {
                try mlx.check(mlx.mlx_astype(&indexed_lhs, lhs_idx, .uint32, self.s));
                break :blk try verifyIndexedExpertInput(self.s, x_flat, indexed_lhs, mlx.getShape(mw.switch_gate_w)[0]);
            } else null;
            var x_gathered = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_gathered);
            if (input_view == null) try mlx.check(mlx.mlx_take_axis(&x_gathered, x_flat, lhs_idx, 0, self.s));
            var x_rep = input_view orelse mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_rep);
            if (input_view == null) try mlx.check(mlx.mlx_reshape(&x_rep, x_gathered, &.{ total_inds, 1, D }, 3, self.s));
            const projection_lhs = if (input_view != null) indexed_lhs else no_idx;
            if (input_view != null) {
                mtp_verify_indexed_input_calls +%= 1;
                const Once = struct {
                    var logged = false;
                };
                if (!Once.logged) {
                    Once.logged = true;
                    log.info("[mtp-verify] indexed expert verify inputs engaged (tokens={d})\n", .{B * S});
                }
            }

            const expert_act = blk: {
                var gate_out_3d = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(gate_out_3d);
                try gatherExpertMm(&gate_out_3d, x_rep, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, projection_lhs, sorted_inds, gate_qp.bits, gate_qp.group_size, gate_qp.mode, true, self.s);
                var gate_out = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(gate_out);
                try mlx.check(mlx.mlx_squeeze(&gate_out, gate_out_3d, self.s));

                var up_out_3d = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(up_out_3d);
                try gatherExpertMm(&up_out_3d, x_rep, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, projection_lhs, sorted_inds, up_qp.bits, up_qp.group_size, up_qp.mode, true, self.s);
                var up_out = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(up_out);
                try mlx.check(mlx.mlx_squeeze(&up_out, up_out_3d, self.s));

                break :blk try self.computeGeglu(gate_out, up_out);
            };
            defer _ = mlx.mlx_array_free(expert_act);
            if (imx) |tap| try tap.observeDownSorted(self.s, expert_act, inv_order, total_inds);

            // down: expand inner singleton → [N,1,intermediate] → gather_qmm → [N,1,hidden]
            var act_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(act_exp);
            try mlx.check(mlx.mlx_expand_dims(&act_exp, expert_act, -2, self.s));
            var down_3d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(down_3d);
            const paired = if (self.verifyFeatureEnabled(.down_reduce, skip_shared, B, S) and down_qp.bits == 4 and down_qp.group_size == 64 and down_qp.mode == .affine)
                try verifyExpertReuse(self.s, act_exp, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, sorted_inds)
            else
                null;
            if (paired) |value| {
                _ = mlx.mlx_array_free(down_3d);
                down_3d = value;
                mtp_verify_expert_pairs_calls +%= 1;
                const Once = struct {
                    var logged = false;
                };
                if (!Once.logged) {
                    Once.logged = true;
                    log.info("[mtp-verify] paired expert down verify engaged (assignments={d})\n", .{total_inds});
                }
            } else try gatherExpertMm(&down_3d, act_exp, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, no_idx, sorted_inds, down_qp.bits, down_qp.group_size, down_qp.mode, true, self.s);
            var down_squeezed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(down_squeezed);
            try mlx.check(mlx.mlx_squeeze(&down_squeezed, down_3d, self.s)); // [N, hidden]

            const fused_sum = if (self.verifyFeatureEnabled(.down_reduce, skip_shared, B, S))
                try verifyExpertReduce(self.s, down_squeezed, inv_order, norm_scores)
            else
                null;
            if (fused_sum) |value| {
                _ = mlx.mlx_array_free(down_out);
                down_out = value;
                moe_reduced = true;
                mtp_verify_expert_reduce_calls +%= 1;
                const Once = struct {
                    var logged = false;
                };
                if (!Once.logged) {
                    Once.logged = true;
                    log.info("[mtp-verify] fused expert reduction verify engaged (tokens={d})\n", .{B * S});
                }
            } else {
                // Inverse permute → original order, then reshape back to [B,S,K,hidden].
                var down_unsorted = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(down_unsorted);
                try mlx.check(mlx.mlx_take_axis(&down_unsorted, down_squeezed, inv_order, 0, self.s));
                const hidden = mlx.getShape(down_unsorted)[1];
                const bskh_shape = [_]c_int{ B, S, K, hidden };
                try mlx.check(mlx.mlx_reshape(&down_out, down_unsorted, &bskh_shape, 4, self.s));
            }
        } else if (B * S == 1 and useGatherQmvDecode(self, gate_qp, up_qp) and mw.switch_gate_s.ctx != null and mw.switch_up_s.ctx != null and mw.switch_down_s.ctx != null and
            try self.moeDecodeGatherQmv(&down_out, expert_x, inds, norm_scores, &moe_reduced, mw, gate_qp, up_qp, down_qp, D, K, B, S))
        {
            // ── Decode fast path: in-place gather-qmv kernel ──
            // Reads the expert bank directly with GPU-resident indices, so it
            // moves the ideal 9.8 MB/projection instead of the take path's 3x
            // (µbench at a 201 MB bank: 37 us vs batched 72 vs stock
            // gather_qmm 349). Everything is done inside the predicate; a null
            // return (unsupported quant mode or width) falls through to the
            // stock gather_qmm path below with no behaviour change.
        } else {
            // ── Decode / small-prefill path (gather_qmm) ──
            const exp_shape = [_]c_int{ B, S, 1, 1, D };
            var x_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_exp);
            try mlx.check(mlx.mlx_reshape(&x_exp, expert_x, &exp_shape, 5, self.s));

            var gate_out_5d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_out_5d);
            try gatherExpertMm(&gate_out_5d, x_exp, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, no_idx, inds, gate_qp.bits, gate_qp.group_size, gate_qp.mode, false, self.s);
            var gate_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_out);
            try mlx.check(mlx.mlx_squeeze(&gate_out, gate_out_5d, self.s));

            var up_out_5d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up_out_5d);
            try gatherExpertMm(&up_out_5d, x_exp, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, no_idx, inds, up_qp.bits, up_qp.group_size, up_qp.mode, false, self.s);
            var up_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up_out);
            try mlx.check(mlx.mlx_squeeze(&up_out, up_out_5d, self.s));

            const expert_act = try self.computeGeglu(gate_out, up_out);
            defer _ = mlx.mlx_array_free(expert_act);

            var act_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(act_exp);
            try mlx.check(mlx.mlx_expand_dims(&act_exp, expert_act, -2, self.s));
            var down_5d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(down_5d);
            try gatherExpertMm(&down_5d, act_exp, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, no_idx, inds, down_qp.bits, down_qp.group_size, down_qp.mode, false, self.s);
            try mlx.check(mlx.mlx_squeeze(&down_out, down_5d, self.s)); // [B, S, K, hidden]
        }

        if (self.cost_trace_active) self.cost_moe |= cost_arm;

        if (moe_prof) {
            try mlx.check(mlx.mlx_array_eval(down_out));
            decode_prof.moe_experts_ns += mclk.lap();
        }

        // Weight by scores: down_out * norm_scores[..., None] → sum over K.
        // The fused down+reduce path already did both inside its one dispatch.
        var expert_sum = mlx.mlx_array_new();
        if (moe_reduced) {
            _ = mlx.mlx_array_set(&expert_sum, down_out); // [B, S, hidden]
        } else {
            var scores_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(scores_exp);
            try mlx.check(mlx.mlx_expand_dims(&scores_exp, norm_scores, -1, self.s)); // [B, S, K, 1]
            var weighted = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(weighted);
            try mlx.check(mlx.mlx_multiply(&weighted, down_out, scores_exp, self.s));
            try mlx.check(mlx.mlx_sum_axis(&expert_sum, weighted, -2, false, self.s)); // [B, S, hidden]
        }

        if (std.mem.eql(u8, cfg.model_type, "mimo_v2") and
            mlx.mlx_array_dtype(expert_sum) != mlx.mlx_array_dtype(expert_x))
        {
            var narrowed = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&narrowed, expert_sum, mlx.mlx_array_dtype(expert_x), self.s));
            _ = mlx.mlx_array_free(expert_sum);
            expert_sum = narrowed;
        }
        if (moeDumpDir() != null) {
            moeDumpTensor(self.s, "y", moe_dump_layer, expert_sum);
        }

        if (skip_shared) return expert_sum;


        // No shared-expert gate: the routed sum is the whole output.
        if (mw.shared_expert_gate_w == null or qwen4Standin().moe_shared) return expert_sum;
        defer _ = mlx.mlx_array_free(expert_sum);
        return self.moeAddGatedShared(expert_sum, expert_x, mw, if (routed.gate_logit.ctx != null) routed.gate_logit else null);
    }

    // ── Mask helpers ──

    fn createCausalMask(self: *const Transformer, q_len: c_int, kv_len: c_int) !mlx.mlx_array {
        const offset_val = kv_len - q_len;
        const shape = [_]c_int{ q_len, kv_len };
        var ones = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ones);
        try mlx.check(mlx.mlx_full(&ones, &shape, 2, self.one, .bfloat16, self.s));
        var upper = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(upper);
        try mlx.check(mlx.mlx_triu(&upper, ones, offset_val + 1, self.s));
        var bool_upper = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bool_upper);
        try mlx.check(mlx.mlx_astype(&bool_upper, upper, .bool_, self.s));
        const zero = bf16Scalar(0.0, self.s);
        defer _ = mlx.mlx_array_free(zero);
        const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
        defer _ = mlx.mlx_array_free(neg_inf);
        var mask = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&mask, bool_upper, neg_inf, zero, self.s));
        const mask_shape = [_]c_int{ 1, 1, q_len, kv_len };
        var mask_4d = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&mask_4d, mask, &mask_shape, 4, self.s));
        _ = mlx.mlx_array_free(mask);
        return mask_4d;
    }

    pub fn createSlidingWindowDecodeMask(self: *const Transformer, kv_len: c_int, window: c_int) !mlx.mlx_array {
        return slidingDecodeMask(self.s, kv_len, window);
    }

    pub fn createSlidingWindowMask(self: *const Transformer, q_len: c_int, kv_len: c_int, window: c_int) !mlx.mlx_array {
        const causal = try self.createCausalMask(q_len, kv_len);
        defer _ = mlx.mlx_array_free(causal);
        const offset_val = kv_len - q_len;
        var row_idx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row_idx);
        try mlx.check(mlx.mlx_arange(&row_idx, @floatFromInt(offset_val), @floatFromInt(offset_val + q_len), 1, .int32, self.s));
        var col_idx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(col_idx);
        try mlx.check(mlx.mlx_arange(&col_idx, 0, @floatFromInt(kv_len), 1, .int32, self.s));
        const row_shape = [_]c_int{ q_len, 1 };
        const col_shape = [_]c_int{ 1, kv_len };
        var row_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row_r);
        var col_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(col_r);
        try mlx.check(mlx.mlx_reshape(&row_r, row_idx, &row_shape, 2, self.s));
        try mlx.check(mlx.mlx_reshape(&col_r, col_idx, &col_shape, 2, self.s));
        var dist = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dist);
        try mlx.check(mlx.mlx_subtract(&dist, row_r, col_r, self.s));
        const window_arr = mlx.mlx_array_new_int(window);
        defer _ = mlx.mlx_array_free(window_arr);
        var too_far = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(too_far);
        try mlx.check(mlx.mlx_greater_equal(&too_far, dist, window_arr, self.s));
        const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
        defer _ = mlx.mlx_array_free(neg_inf);
        const zero = bf16Scalar(0.0, self.s);
        defer _ = mlx.mlx_array_free(zero);
        var sw_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sw_mask);
        try mlx.check(mlx.mlx_where(&sw_mask, too_far, neg_inf, zero, self.s));
        const mask_shape = [_]c_int{ 1, 1, q_len, kv_len };
        var sw_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sw_4d);
        try mlx.check(mlx.mlx_reshape(&sw_4d, sw_mask, &mask_shape, 4, self.s));
        var combined = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&combined, causal, sw_4d, self.s));
        return combined;
    }
};

// ── Init helpers ──

/// Pre-transpose a plain-BF16 weight stored as `[out, in]` to `[in, out]` so
/// `mlx_matmul(x, w_t)` lands the contraction over the input axis. Used by
/// Unsloth Dynamic checkpoints that leave linear-attention projections
/// unquantized while quantizing the rest. Caller owns the returned array.
pub fn transposeBf16Weight(w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    // Swap the last two axes for any rank: 2D weights [out, in] → [in, out];
    // stacked MoE expert tensors [experts, out, in] → [experts, in, out].
    const ndim = mlx.getShape(w).len;
    var perm: [8]c_int = undefined; // mlx arrays never exceed 8 dims here
    for (0..ndim) |i| perm[i] = @intCast(i);
    perm[ndim - 2] = @intCast(ndim - 1);
    perm[ndim - 1] = @intCast(ndim - 2);
    var w_t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&w_t, w, &perm, @intCast(ndim), s));
    return w_t;
}

/// In-place: if `*sc` is null-ctx, we treat the matching `*w` as plain bf16.
/// Replace `*w` with its pre-transposed `[in, out]` form and record the new
/// array in `owned` so we can free it on Transformer.deinit.
fn maybeTransposeForBf16(
    w: *mlx.mlx_array,
    sc: mlx.mlx_array,
    owned: *std.ArrayList(mlx.mlx_array),
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
) !void {
    if (sc.ctx != null) return; // quantized weight — leave as-is
    if (w.ctx == null) return; // absent weight (e.g. KV-shared layer) — nothing to transpose
    const transposed = try transposeBf16Weight(w.*, s);
    w.* = transposed;
    try owned.append(allocator, transposed);
}

/// One quantized-or-dense projection triple by full tensor base name.
pub fn weightTriple(weights: *const Weights, base: []const u8, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array } {
    var nb: [160]u8 = undefined;
    const w = weights.get(std.fmt.bufPrint(&nb, "{s}.weight", .{base}) catch unreachable) orelse {
        log.err("MISSING WEIGHT: {s}.weight\n", .{base});
        return error.MissingWeight;
    };
    const sc = weights.get(std.fmt.bufPrint(&nb, "{s}.scales", .{base}) catch unreachable) orelse mlx.mlx_array_new();
    const bi = weights.get(std.fmt.bufPrint(&nb, "{s}.biases", .{base}) catch unreachable) orelse mlx.mlx_array_new();
    var wm = w;
    try maybeTransposeForBf16(&wm, sc, owned_bf16, allocator, s);
    return .{ .w = wm, .sc = sc, .bi = bi };
}

/// A grouped-norm weight `[hc*hidden]` viewed as `[hc, hidden]` (owned).
fn groupedNormView(weights: *const Weights, name: []const u8, hc: u32, hidden: u32, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
    const shape = [_]c_int{ @intCast(hc), @intCast(hidden) };
    var view = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&view, raw, &shape, 2, s));
    try owned_bf16.append(allocator, view);
    return view;
}

pub fn loadHcWeights(weights: *const Weights, base: []const u8, with_inject: bool, hc: u32, hidden: u32, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !HcWeights {
    var nb: [160]u8 = undefined;
    const down = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.input_mix_weight_down", .{base}) catch unreachable, owned_bf16, allocator, s);
    const up = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.input_mix_weight_up", .{base}) catch unreachable, owned_bf16, allocator, s);
    var out: HcWeights = .{
        .norm_w = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.hc_norm.weight", .{base}) catch unreachable, hc, hidden, owned_bf16, allocator, s),
        .down_w = down.w,
        .down_s = down.sc,
        .down_b = down.bi,
        .up_w = up.w,
        .up_s = up.sc,
        .up_b = up.bi,
    };
    // The reference divides the down and inject projections by hc_count;
    // fold that into the (de)quantization affine or the dense weight.
    const inv_hc = mlx.mlx_array_new_float(1.0 / @as(f32, @floatFromInt(hc)));
    defer _ = mlx.mlx_array_free(inv_hc);
    try scaleTriple(&out.down_w, &out.down_s, &out.down_b, inv_hc, owned_bf16, allocator, s);
    if (with_inject) {
        const inj = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.block_inject_weight", .{base}) catch unreachable, owned_bf16, allocator, s);
        out.inject_w = inj.w;
        out.inject_s = inj.sc;
        out.inject_b = inj.bi;
        try scaleTriple(&out.inject_w, &out.inject_s, &out.inject_b, inv_hc, owned_bf16, allocator, s);
        if (out.inject_s.ctx == null) {
            var flat = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_contiguous(&flat, out.inject_w, false, s));
            try mlx.check(mlx.mlx_array_eval(flat));
            try owned_bf16.append(allocator, flat);
            out.inject_flat = flat;
        }
    }
    return out;
}

/// Multiply a projection by a scalar: scales+biases of an affine weight, or
/// the dense weight itself. Replacement arrays are owned (and evaluated).
fn scaleTriple(w: *mlx.mlx_array, sc: *mlx.mlx_array, bi: *mlx.mlx_array, k: mlx.mlx_array, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !void {
    // The scalar must carry the TARGET's dtype: an f32 scalar promotes the
    // bf16 weight/scales, and a widened inject gate then widened the whole
    // residual stream to f32 (1/hc is exact in bf16 for hc <= 8).
    const Scaled = struct {
        fn f(a: mlx.mlx_array, k_raw: mlx.mlx_array, st: mlx.mlx_stream) !mlx.mlx_array {
            var kt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(kt);
            try mlx.check(mlx.mlx_astype(&kt, k_raw, mlx.mlx_array_dtype(a), st));
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&out, a, kt, st));
            try mlx.check(mlx.mlx_array_eval(out));
            return out;
        }
    }.f;
    if (sc.ctx != null) {
        const s2 = try Scaled(sc.*, k, s);
        try owned_bf16.append(allocator, s2);
        sc.* = s2;
        if (bi.ctx != null) {
            const b2 = try Scaled(bi.*, k, s);
            try owned_bf16.append(allocator, b2);
            bi.* = b2;
        }
    } else {
        const w2 = try Scaled(w.*, k, s);
        try owned_bf16.append(allocator, w2);
        w.* = w2;
    }
}

fn loadPleWeights(weights: *const Weights, name_buf: *[256]u8, prefix: []const u8, li: u32, hc: u32, hidden: u32, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !PleWeights {
    var nb: [160]u8 = undefined;
    const key = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.key_proj", .{ prefix, li }) catch unreachable, owned_bf16, allocator, s);
    const value = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.value_proj", .{ prefix, li }) catch unreachable, owned_bf16, allocator, s);
    return .{
        .key_w = key.w,
        .key_s = key.sc,
        .key_b = key.bi,
        .value_w = value.w,
        .value_s = value.sc,
        .value_b = value.bi,
        .norm_key = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.norm_key.weight", .{ prefix, li }) catch unreachable, hc, hidden, owned_bf16, allocator, s),
        .norm_query = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.norm_query.weight", .{ prefix, li }) catch unreachable, hc, hidden, owned_bf16, allocator, s),
        .norm_conv = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.norm_conv.weight", .{ prefix, li }) catch unreachable, hc, hidden, owned_bf16, allocator, s),
        .conv_w = try getLayerWeight(weights, name_buf, prefix, li, "ple.conv1d.weight"),
    };
}

fn shouldStreamExpertBank(config: *const ModelConfig, prefix: []const u8) bool {
    return config.expert_streaming and model_mod.isExpertStreamingArch(config.model_type) and
        std.mem.eql(u8, prefix, config.weight_prefix);
}

pub fn initMoeLayers(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights, name_buf: *[256]u8, s: mlx.mlx_stream) !struct { moe_layers: []MoeLayerWeights, ssm_entries: []SSMCacheEntry, owned_bf16: []mlx.mlx_array } {
    log.info("Precomputing MoE layer weights...\n", .{});
    const prefix = config.weight_prefix;
    const moe_layers = try allocator.alloc(MoeLayerWeights, config.num_hidden_layers);
    errdefer allocator.free(moe_layers);
    const ssm_entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
    errdefer allocator.free(ssm_entries);
    var owned_bf16: std.ArrayList(mlx.mlx_array) = .empty;
    errdefer {
        for (owned_bf16.items) |a| _ = mlx.mlx_array_free(a);
        owned_bf16.deinit(allocator);
    }
    const is_qwen4 = config.isQwen4();
    const is_mimo = std.mem.eql(u8, config.model_type, "mimo_v2");

    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const lw = &moe_layers[i];
        const is_linear = config.isLinearLayer(li);

        // qwen4_exp has no per-layer norms: the hyper-connection read normalizes.
        lw.input_norm = if (is_qwen4)
            mlx.mlx_array_new()
        else
            try getLayerWeight(weights, name_buf, prefix, li, "input_layernorm.weight");
        lw.post_attn_norm = if (is_qwen4)
            mlx.mlx_array_new()
        else
            try getLayerWeight(weights, name_buf, prefix, li, "post_attention_layernorm.weight");
        lw.is_linear = is_linear;

        // `moe_layers` comes from `allocator.alloc` which skips struct defaults, so every
        // optional must be initialized: the eval loop would read uninitialized memory as
        // valid handles.
        lw.hc_attn = null;
        lw.hc_mlp = null;
        lw.ple = null;
        if (is_qwen4) {
            var hb: [96]u8 = undefined;
            lw.hc_attn = try loadHcWeights(weights, std.fmt.bufPrint(&hb, "{s}.layers.{d}.attn_hyper_connection", .{ prefix, li }) catch unreachable, true, config.hc_count, config.hidden_size, &owned_bf16, allocator, s);
            lw.hc_mlp = try loadHcWeights(weights, std.fmt.bufPrint(&hb, "{s}.layers.{d}.mlp_hyper_connection", .{ prefix, li }) catch unreachable, true, config.hc_count, config.hidden_size, &owned_bf16, allocator, s);
            if (config.ple_layer_idx == @as(i32, @intCast(li))) {
                lw.ple = try loadPleWeights(weights, name_buf, prefix, li, config.hc_count, config.hidden_size, &owned_bf16, allocator, s);
            }
        }

        if (is_linear) {
            // Each projection's `*_s`/`*_b` are loaded optionally: a plain-bf16 projection has
            // null-ctx scales and is transposed on load so `qmatmulBits` can dispatch to
            // `mlx_matmul`.
            lw.attn = .{ .linear = .{
                .qkv_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.weight"),
                .qkv_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.scales") orelse mlx.mlx_array_new(),
                .qkv_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.biases") orelse mlx.mlx_array_new(),
                .z_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_z.weight"),
                .z_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_z.scales") orelse mlx.mlx_array_new(),
                .z_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_z.biases") orelse mlx.mlx_array_new(),
                .a_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_a.weight"),
                .a_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_a.scales") orelse mlx.mlx_array_new(),
                .a_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_a.biases") orelse mlx.mlx_array_new(),
                .b_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_b.weight"),
                .b_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_b.scales") orelse mlx.mlx_array_new(),
                .b_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_b.biases") orelse mlx.mlx_array_new(),
                .conv1d_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.conv1d.weight"),
                .A_log = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.A_log"),
                .dt_bias = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.dt_bias"),
                .norm_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.norm.weight"),
                .out_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.out_proj.weight"),
                .out_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.scales") orelse mlx.mlx_array_new(),
                .out_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.biases") orelse mlx.mlx_array_new(),
            } };
            const la = &lw.attn.linear;
            try maybeTransposeForBf16(&la.qkv_w, la.qkv_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&la.z_w, la.z_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&la.a_w, la.a_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&la.b_w, la.b_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&la.out_w, la.out_s, &owned_bf16, allocator, s);
        } else if (is_mimo) {
            // MiMo V2 supports both converter-split projections and the HF
            // native fused QKV tensor. The latter is split by explicit row
            // counts because K and V use different per-head widths.
            const q_rows: u32 = config.layerNumHeads(li) * config.layerHeadDim(li);
            const k_rows: u32 = config.layerKVHeads(li) * config.layerHeadDim(li);
            const v_rows: u32 = config.layerKVHeads(li) * config.layerVHeadDim(li);
            var q_w: mlx.mlx_array = undefined;
            var q_s: mlx.mlx_array = undefined;
            var q_b: mlx.mlx_array = undefined;
            var k_w: mlx.mlx_array = undefined;
            var k_s: mlx.mlx_array = undefined;
            var k_b: mlx.mlx_array = undefined;
            var v_w: mlx.mlx_array = undefined;
            var v_s: mlx.mlx_array = undefined;
            var v_b: mlx.mlx_array = undefined;
            const fused_opt = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.qkv_proj.weight");
            if (fused_opt != null and mlx.mlx_array_dtype(fused_opt.?) == .uint8) {
                // FP8 source QKV: rank-local rows the forward splits (`mimoAttnWith`).
                q_w = fused_opt.?;
                q_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.qkv_proj.scales") orelse return error.MissingWeight;
                q_b = mlx.mlx_array_new();
                k_w = mlx.mlx_array_new();
                k_s = mlx.mlx_array_new();
                k_b = mlx.mlx_array_new();
                v_w = mlx.mlx_array_new();
                v_s = mlx.mlx_array_new();
                v_b = mlx.mlx_array_new();
            } else if (fused_opt) |fused_w| {
                const fused_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.qkv_proj.scales") orelse mlx.mlx_array_new();
                const fused_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.qkv_proj.biases") orelse mlx.mlx_array_new();
                const own_fused_s = fused_s.ctx == null;
                const own_fused_b = fused_b.ctx == null;
                defer {
                    if (own_fused_s) _ = mlx.mlx_array_free(fused_s);
                    if (own_fused_b) _ = mlx.mlx_array_free(fused_b);
                }
                const fused_transpose = fused_s.ctx == null;
                const parts_w = try splitFusedQkvAsymRows(fused_w, q_rows, k_rows, v_rows, fused_transpose, &owned_bf16, allocator, s);
                const parts_s = try splitFusedQkvAsymRows(fused_s, q_rows, k_rows, v_rows, false, &owned_bf16, allocator, s);
                const parts_b = try splitFusedQkvAsymRows(fused_b, q_rows, k_rows, v_rows, false, &owned_bf16, allocator, s);
                q_w = parts_w[0];
                k_w = parts_w[1];
                v_w = parts_w[2];
                q_s = parts_s[0];
                k_s = parts_s[1];
                v_s = parts_s[2];
                q_b = parts_b[0];
                k_b = parts_b[1];
                v_b = parts_b[2];
            } else {
                q_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.weight");
                q_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_proj.scales") orelse mlx.mlx_array_new();
                q_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_proj.biases") orelse mlx.mlx_array_new();
                k_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.weight");
                k_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_proj.scales") orelse mlx.mlx_array_new();
                k_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_proj.biases") orelse mlx.mlx_array_new();
                v_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.weight");
                v_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.scales") orelse mlx.mlx_array_new();
                v_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.biases") orelse mlx.mlx_array_new();
                try maybeTransposeForBf16(&q_w, q_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&k_w, k_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&v_w, v_s, &owned_bf16, allocator, s);
            }
            lw.attn = .{ .full = .{
                .q_w = q_w,
                .q_s = q_s,
                .q_b = q_b,
                .k_w = k_w,
                .k_s = k_s,
                .k_b = k_b,
                .v_w = v_w,
                .v_s = v_s,
                .v_b = v_b,
                .o_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.weight"),
                .o_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.o_proj.scales") orelse mlx.mlx_array_new(),
                .o_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.o_proj.biases") orelse mlx.mlx_array_new(),
                .q_norm = mlx.mlx_array_new(),
                .k_norm = mlx.mlx_array_new(),
                .sinks = if (config.layerHasAttnSinks(li))
                    try getLayerWeight(weights, name_buf, prefix, li, "self_attn.attention_sink_bias")
                else
                    mlx.mlx_array_new(),
            } };
            try maybeTransposeForBf16(&lw.attn.full.o_w, lw.attn.full.o_s, &owned_bf16, allocator, s);
        } else {
            const k_s = try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.k_proj.scales", config.quant_bits);
            const k_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.weight");
            const k_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.k_proj.biases", &config);
            lw.attn = .{
                .full = .{
                    .q_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.weight"),
                    .q_s = try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.q_proj.scales", config.quant_bits),
                    .q_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.q_proj.biases", &config),
                    .k_w = k_w,
                    .k_s = k_s,
                    .k_b = k_b,
                    .v_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.weight"),
                    .v_s = try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.v_proj.scales", config.quant_bits),
                    .v_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.v_proj.biases", &config),
                    .o_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.weight"),
                    .o_s = try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.o_proj.scales", config.quant_bits),
                    .o_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.o_proj.biases", &config),
                    .q_norm = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_norm.weight"),
                    .k_norm = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_norm.weight"),
                },
            };
            {
                // Dense bf16 (null-ctx scales): pre-transpose [out,in]→[in,out] so
                // qmatmulBits dispatches to a plain matmul. No-op on quantized weights.
                const fa = &lw.attn.full;
                try maybeTransposeForBf16(&fa.q_w, fa.q_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.k_w, fa.k_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.o_w, fa.o_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.v_w, fa.v_s, &owned_bf16, allocator, s);
                if (is_qwen4 and config.indexer_n_heads > 0) {
                    fa.idx_qk_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.indexer.index_qk_proj.weight");
                    fa.idx_qk_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.indexer.index_qk_proj.scales") orelse mlx.mlx_array_new();
                    fa.idx_qk_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.indexer.index_qk_proj.biases") orelse mlx.mlx_array_new();
                    fa.idx_q_norm = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.indexer.q_layernorm.weight");
                    fa.idx_k_norm = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.indexer.k_layernorm.weight");
                    try maybeTransposeForBf16(&fa.idx_qk_w, fa.idx_qk_s, &owned_bf16, allocator, s);
                }
            }
        }

        const layer_is_moe = config.isMoe() and li >= config.first_k_dense_replace;

        if (layer_is_moe and is_mimo) {
            // MiMo V2: sigmoid/noaux_tc routing with a selection-only
            // correction bias, no shared expert, and biasless mxfp4 banks.
            const stream_bank = shouldStreamExpertBank(&config, prefix);
            const exl3 = config.expert_layout == .exl3_k4;
            const switch_bank: SwitchMlpBank = if (stream_bank) .{
                .gate_w = mlx.mlx_array_new(),
                .gate_s = mlx.mlx_array_new(),
                .gate_b = mlx.mlx_array_new(),
                .up_w = mlx.mlx_array_new(),
                .up_s = mlx.mlx_array_new(),
                .up_b = mlx.mlx_array_new(),
                .down_w = mlx.mlx_array_new(),
                .down_s = mlx.mlx_array_new(),
                .down_b = mlx.mlx_array_new(),
            } else try loadSwitchMlpBank(weights, name_buf, prefix, li, exl3, &config);
            if (switch_bank.ragged) |*rb| try rb.handOff(allocator, &owned_bf16);
            lw.mlp = .{ .moe = .{
                .router_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.gate.weight"),
                .router_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.scales") orelse mlx.mlx_array_new(),
                .router_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.biases") orelse mlx.mlx_array_new(),
                .exl3_ragged = switch_bank.ragged,
                .switch_gate_w = switch_bank.gate_w,
                .switch_gate_s = switch_bank.gate_s,
                .switch_gate_b = switch_bank.gate_b,
                .switch_up_w = switch_bank.up_w,
                .switch_up_s = switch_bank.up_s,
                .switch_up_b = switch_bank.up_b,
                .switch_down_w = switch_bank.down_w,
                .switch_down_s = switch_bank.down_s,
                .switch_down_b = switch_bank.down_b,
                .shared_gate_w = mlx.mlx_array_new(),
                .shared_gate_s = mlx.mlx_array_new(),
                .shared_gate_b = mlx.mlx_array_new(),
                .shared_up_w = mlx.mlx_array_new(),
                .shared_up_s = mlx.mlx_array_new(),
                .shared_up_b = mlx.mlx_array_new(),
                .shared_down_w = mlx.mlx_array_new(),
                .shared_down_s = mlx.mlx_array_new(),
                .shared_down_b = mlx.mlx_array_new(),
                .expert_bias = try getLayerWeight(weights, name_buf, prefix, li, "mlp.gate.e_score_correction_bias"),
                .route_norm = config.moe_route_norm,
                .route_scale = config.router_scaling_factor,
            } };
            {
                const mw = &lw.mlp.moe;
                try maybeTransposeForBf16(&mw.router_w, mw.router_s, &owned_bf16, allocator, s);
                // Routing is f32 (`mimoRouterLogits`): widen once here rather than per
                // forward. Billed by `mimo_source.countResidentBytes`.
                if (config.usesMimoSourceTrunk() and mw.router_s.ctx == null and mlx.mlx_array_dtype(mw.router_w) != .float32) {
                    var w32 = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_astype(&w32, mw.router_w, .float32, s));
                    try owned_bf16.append(allocator, w32);
                    mw.router_w = w32;
                }
                if (!exl3) {
                    try maybeTransposeForBf16(&mw.switch_gate_w, mw.switch_gate_s, &owned_bf16, allocator, s);
                    try maybeTransposeForBf16(&mw.switch_up_w, mw.switch_up_s, &owned_bf16, allocator, s);
                    try maybeTransposeForBf16(&mw.switch_down_w, mw.switch_down_s, &owned_bf16, allocator, s);
                }
            }
        } else if (layer_is_moe) {
            // Each `*_s`/`*_b` is loaded optionally — a plain-bf16 router
            // (`mlp.gate`) and shared-expert gate (`mlp.shared_expert_gate`) carry
            // no scales/biases. The shared-expert WEIGHTS themselves are also
            // optional: when absent they bind to empty handles and
            // `shared_expert_gate_w` stays null, which makes moeMLP early-return
            // the routed-expert sum without ever reading them — so no MISSING
            // WEIGHT crash. The
            // `maybeTransposeForBf16` calls below pre-transpose bf16 weights from
            // `[out, in]` → `[in, out]` so `qmatmulBits` can dispatch to plain
            // `mlx_matmul`; they no-op on already-quantized AND on empty handles.
            const stream_bank = shouldStreamExpertBank(&config, prefix);
            const exl3 = config.expert_layout == .exl3_k4;
            const switch_bank: SwitchMlpBank = if (stream_bank) .{
                .gate_w = mlx.mlx_array_new(),
                .gate_s = mlx.mlx_array_new(),
                .gate_b = mlx.mlx_array_new(),
                .up_w = mlx.mlx_array_new(),
                .up_s = mlx.mlx_array_new(),
                .up_b = mlx.mlx_array_new(),
                .down_w = mlx.mlx_array_new(),
                .down_s = mlx.mlx_array_new(),
                .down_b = mlx.mlx_array_new(),
            } else try loadSwitchMlpBank(weights, name_buf, prefix, li, exl3, &config);
            if (switch_bank.ragged) |*rb| try rb.handOff(allocator, &owned_bf16);
            lw.mlp = .{ .moe = .{
                .router_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.gate.weight"),
                .router_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.scales") orelse mlx.mlx_array_new(),
                .router_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.biases") orelse mlx.mlx_array_new(),
                .exl3_ragged = switch_bank.ragged,
                .switch_gate_w = switch_bank.gate_w,
                .switch_gate_s = switch_bank.gate_s,
                .switch_gate_b = switch_bank.gate_b,
                .switch_up_w = switch_bank.up_w,
                .switch_up_s = switch_bank.up_s,
                .switch_up_b = switch_bank.up_b,
                .switch_down_w = switch_bank.down_w,
                .switch_down_s = switch_bank.down_s,
                .switch_down_b = switch_bank.down_b,
                .shared_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.weight") orelse mlx.mlx_array_new(),
                .shared_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.scales") orelse mlx.mlx_array_new(),
                .shared_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.biases") orelse mlx.mlx_array_new(),
                .shared_up_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.weight") orelse mlx.mlx_array_new(),
                .shared_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.scales") orelse mlx.mlx_array_new(),
                .shared_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.biases") orelse mlx.mlx_array_new(),
                .shared_down_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.weight") orelse mlx.mlx_array_new(),
                .shared_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.scales") orelse mlx.mlx_array_new(),
                .shared_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.biases") orelse mlx.mlx_array_new(),
                .shared_expert_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert_gate.weight"),
                .shared_expert_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert_gate.scales") orelse mlx.mlx_array_new(),
                .shared_expert_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert_gate.biases") orelse mlx.mlx_array_new(),
            } };
            {
                const mw = &lw.mlp.moe;
                try maybeTransposeForBf16(&mw.router_w, mw.router_s, &owned_bf16, allocator, s);
                if (!exl3) {
                    try maybeTransposeForBf16(&mw.switch_gate_w, mw.switch_gate_s, &owned_bf16, allocator, s);
                    try maybeTransposeForBf16(&mw.switch_up_w, mw.switch_up_s, &owned_bf16, allocator, s);
                    try maybeTransposeForBf16(&mw.switch_down_w, mw.switch_down_s, &owned_bf16, allocator, s);
                }
                try maybeTransposeForBf16(&mw.shared_gate_w, mw.shared_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_up_w, mw.shared_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_down_w, mw.shared_down_s, &owned_bf16, allocator, s);
                if (mw.shared_expert_gate_w) |*seg_w_ptr| {
                    try maybeTransposeForBf16(seg_w_ptr, mw.shared_expert_gate_s.?, &owned_bf16, allocator, s);
                    if (mw.shared_expert_gate_s.?.ctx == null and mlx.mlx_array_size(seg_w_ptr.*) == config.hidden_size) {
                        var flat = mlx.mlx_array_new();
                        defer _ = mlx.mlx_array_free(flat);
                        try mlx.check(mlx.mlx_reshape(&flat, seg_w_ptr.*, &[_]c_int{@intCast(config.hidden_size)}, 1, s));
                        var vec = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_contiguous(&vec, flat, false, s));
                        try owned_bf16.append(allocator, vec);
                        mw.shared_gate_vec = vec;
                    }
                }
            }
        } else {
            lw.mlp = .{ .dense = .{
                .gate_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.weight"),
                .gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
                .gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
                .up_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.weight"),
                .up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.scales") orelse mlx.mlx_array_new(),
                .up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.biases") orelse mlx.mlx_array_new(),
                .down_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.weight"),
                .down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.scales") orelse mlx.mlx_array_new(),
                .down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.biases") orelse mlx.mlx_array_new(),
            } };
            {
                const dw = &lw.mlp.dense;
                try maybeTransposeForBf16(&dw.gate_w, dw.gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&dw.up_w, dw.up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&dw.down_w, dw.down_s, &owned_bf16, allocator, s);
            }
        }

        ssm_entries[i] = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = false,
        };
    }

    // The PLE is placed by exact equality inside the loop above and nothing downstream notices
    // its absence; a silently PLE-less qwen4 emits plausible text.
    if (config.isQwen4()) {
        const has_ple = try allocator.alloc(bool, moe_layers.len);
        defer allocator.free(has_ple);
        for (moe_layers, 0..) |*lw, i| has_ple[i] = lw.ple != null;
        if (!qwen4_config.qwen4PleInstalledAt(has_ple, config.ple_layer_idx)) {
            log.err("[qwen4] the PLE was not installed at layer {d} ({d} layers carry PLE weights)\n", .{ config.ple_layer_idx, std.mem.count(bool, has_ple, &.{true}) });
            return error.Qwen4PleNotInstalled;
        }
    }

    return .{
        .moe_layers = moe_layers,
        .ssm_entries = ssm_entries,
        .owned_bf16 = try owned_bf16.toOwnedSlice(allocator),
    };
}

/// Append every non-null mlx_array field of a plain weight struct.
pub fn appendStructArrays(vec: mlx.mlx_vector_array, w: anytype) void {
    inline for (comptime structFields(@TypeOf(w.*))) |field| {
        if (comptime field.type != mlx.mlx_array) continue;
        const arr = @field(w, field.name);
        if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
    }
}

pub fn appendFullAttnWeights(vec: mlx.mlx_vector_array, fa: *const FullAttnWeights) void {
    inline for (comptime structFields(FullAttnWeights)) |field| {
        // Dense bf16 full-attn layers carry null-ctx scales/biases — skip those
        // so null arrays don't poison the eval batch. Mirrors the linear/mlp paths.
        const arr = @field(fa, field.name);
        if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
    }
}

pub fn appendLinearAttnWeights(vec: mlx.mlx_vector_array, la: *const LinearAttnWeights) void {
    inline for (comptime structFields(LinearAttnWeights)) |field| {
        if (comptime field.type != mlx.mlx_array) continue;
        {
            const arr = @field(la, field.name);
            // Plain-bf16 layers (Unsloth Dynamic) carry null scales/biases — skip those.
            if (arr.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(vec, arr);
            }
        }
    }
}

pub fn appendHybridMlpWeights(vec: mlx.mlx_vector_array, hw: *const HybridMlpWeights) void {
    // Plain-bf16 layers (Unsloth Dynamic) carry null-ctx scales/biases — skip those
    // so they don't pollute the eval batch. Mirrors `appendLinearAttnWeights`.
    switch (hw.*) {
        .moe => |*mw| {
            if (mw.exl3_ragged) |*rb| {
                for (rb.storage()) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
                for (rb.views[0..rb.count]) |bank| {
                    for ([_]sushi_exl3.Proj{ bank.gate, bank.up, bank.down }) |proj| {
                        for ([_]mlx.mlx_array{ proj.trellis, proj.suh, proj.svh }) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
                    }
                }
            }
            inline for (comptime structFields(MoeMlpWeights)) |field| {
                if (field.type == ?mlx.mlx_array) {
                    if (@field(mw, field.name)) |arr| {
                        if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
                    }
                } else if (field.type == mlx.mlx_array) {
                    const arr = @field(mw, field.name);
                    if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
                }
            }
        },
        .dense => |*dw| {
            inline for (comptime structFields(DenseMlpWeights)) |field| {
                const arr = @field(dw, field.name);
                if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
            }
        },
    }
}


/// Detect quantization bits from weight and scales shapes: bits = w_cols * 32 / (s_cols * group_size)
fn detectQuantBits(w: mlx.mlx_array, sc: mlx.mlx_array, group_size: u32) u32 {
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    if (w_shape.len < 2 or s_shape.len < 2) return 4;
    const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
    const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
    if (s_cols == 0) return 4;
    return (w_cols * 32) / (s_cols * group_size);
}

/// Last-axis size of an array, as the input-dimension hint for
/// `quantParamsHinted`. Null for null-ctx handles or degenerate shapes.
pub inline fn lastDim(x: mlx.mlx_array) ?u32 {
    if (x.ctx == null) return null;
    const shape = mlx.getShape(x);
    if (shape.len == 0) return null;
    const d = shape[shape.len - 1];
    return if (d > 0) @as(u32, @intCast(d)) else null;
}

/// Detect a quantized weight's (bits, group_size, mode) from its scales tensor
/// plus the model config:
/// - uint8 scales → fp8-family. In an mxfp4/mxfp8 model the config's
///   mode/bits/group_size apply (the fp8 scheme is uniform within a model).
///   Under an affine config this is the legacy "mxfp8 tensor inside an affine
///   checkpoint" case: bits is always 8, group size from the shape ratio.
/// - float scales → affine. In an affine model, bits via detectQuantBits
///   against the config group_size (mixed-precision models share one gs). For
///   affine OVERRIDES inside a non-affine model (a QAT checkpoint keeping its
///   shared MLP at affine 8-bit/gs64) the config group_size doesn't
///   apply: w_cols*32/s_cols only pins bits×gs, so solve exactly with the
///   caller's `in_dim` (activation inner dim); without a hint, assume mlx-lm's
///   override default group size 64.
/// Solve an affine weight's exact (bits, group_size) from packed-column
/// geometry when the input dimension is known:
///   bits = w_cols * 32 / in_dim,  group_size = in_dim / s_cols
/// Returns null unless both divide EXACTLY and land on values MLX's affine
/// packing supports — callers then fall back to config-based detection.
/// This is what lets a sidecar (e.g. an MTP head quantized 5-bit/gs-128 over
/// a 4-bit/gs-64 trunk) resolve per-weight instead of inheriting the trunk's
/// group size.
/// True when an embedding/lm_head TABLE is stored dense (float dtype) even
/// though the config declares global quantization. Mixed checkpoints quantize
/// per-tensor — a 2-bit pack can ship a bf16 embed_tokens beside 2-bit experts —
/// so scales-presence must be decided by the WEIGHT's dtype, never the
/// config's bits (a packed uint32 table missing scales still crashes
/// honestly at the mandatory fetch).
pub fn floatDtypeTable(dtype: mlx.mlx_dtype) bool {
    return dtype == .bfloat16 or dtype == .float16 or dtype == .float32;
}

test "floatDtypeTable: float tables are dense, packed/quantized ones are not" {
    try std.testing.expect(floatDtypeTable(.bfloat16));
    try std.testing.expect(floatDtypeTable(.float16));
    try std.testing.expect(floatDtypeTable(.float32));
    try std.testing.expect(!floatDtypeTable(.uint32)); // packed affine quant
    try std.testing.expect(!floatDtypeTable(.uint8)); // fp8-encoded scales/modes
    try std.testing.expect(!floatDtypeTable(.int32));
}

pub fn affineParamsFromGeometry(w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: u32) ?QuantParams {
    if (sc.ctx == null) return null;
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    if (w_shape.len < 2 or s_shape.len < 2) return null;
    const geom = expert_quant_mod.affineGeomFromShapes(
        @intCast(w_shape[w_shape.len - 1]),
        @intCast(s_shape[s_shape.len - 1]),
        in_dim,
    ) orelse return null;
    return .{ .bits = geom.bits, .group_size = geom.group_size, .mode = .affine };
}

/// The fp mode a solved (bits, group_size) pair belongs to. Each format
/// DEFINES its block size — MXFP8/MXFP4 are 32 — so the pair
/// names exactly one of them. Anything else is a layout we don't serve;
/// returning null keeps a checkpoint's own declaration rather than inventing a
/// mode for it.
fn fpQuantModeFor(bits: u32, group_size: u32) ?QuantMode {
    if (bits == 8 and group_size == 32) return .mxfp8;
    if (bits == 4 and group_size == 32) return .mxfp4;
    return null;
}

/// Per-weight (bits, group_size, mode) for an fp-quantized tensor, solved from
/// the packed geometry against the activation's inner dim: a row holds
/// `w_cols * 32` packed bits over `in_dim` values, and `s_cols` scales cover
/// them in groups.
///
/// Returns null when the geometry AGREES with the model-wide declaration —
/// the caller then returns the config verbatim, so a consistent checkpoint
/// resolves exactly as it did before this existed.
fn fpParamsFromGeometry(config: *const ModelConfig, w_cols: u32, s_cols: u32, in_dim: u32) ?QuantParams {
    if (w_cols == 0 or s_cols == 0 or in_dim == 0) return null;
    const packed_bits = w_cols * 32;
    if (packed_bits % in_dim != 0 or in_dim % s_cols != 0) return null;
    const bits = packed_bits / in_dim;
    const group_size = in_dim / s_cols;
    if (bits == config.quant_bits and group_size == config.quant_group_size) return null;
    const mode = fpQuantModeFor(bits, group_size) orelse return null;
    return .{ .bits = bits, .group_size = group_size, .mode = mode };
}

pub fn computeQuantParams(config: *const ModelConfig, w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: ?u32) QuantParams {
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    const w_cols: u32 = if (w_shape.len >= 2) @intCast(w_shape[w_shape.len - 1]) else 0;
    const s_cols: u32 = if (s_shape.len >= 2) @intCast(s_shape[s_shape.len - 1]) else 0;

    if (mlx.mlx_array_dtype(sc) == .uint8) {
        if (config.quant_mode != .affine) {
            // A per-tensor override can live INSIDE the fp family: an
            // mxfp4 build can store `model.embed_tokens` as mxfp8. Every fp mode
            // carries uint8 (E8M0/E4M3) scales, so the scales-dtype rule above
            // cannot separate them and the embedding resolved at the model-wide
            // 4 bits — an `mlx_dequantize` shape rejection, i.e. an MLX error,
            // i.e. a dead server on the first request. The packed geometry says
            // it exactly; solve, exactly as the affine arm below does.
            if (in_dim) |in| {
                if (fpParamsFromGeometry(config, w_cols, s_cols, in)) |qp| return qp;
            }
            return .{ .bits = config.quant_bits, .group_size = config.quant_group_size, .mode = config.quant_mode };
        }
        const gs: u32 = if (w_cols > 0 and s_cols > 0) (w_cols * 32) / (s_cols * 8) else 32;
        return .{ .bits = 8, .group_size = gs, .mode = .mxfp8 };
    }

    if (config.quant_mode == .affine) {
        // Exact per-weight solve when the caller knows the input dim — a
        // hinted tensor whose geometry contradicts the config's group size
        // (MTP sidecars mix gs 128/64/32 over a gs-64 trunk) resolves to its
        // TRUE params. Consistent tensors solve to the config values anyway.
        if (in_dim) |in| {
            if (affineParamsFromGeometry(w, sc, in)) |qp| return qp;
        }
        return .{
            .bits = detectQuantBits(w, sc, config.quant_group_size),
            .group_size = config.quant_group_size,
            .mode = .affine,
        };
    }

    if (in_dim) |in| {
        if (in > 0 and s_cols > 0 and w_cols > 0) {
            return .{ .bits = (w_cols * 32) / in, .group_size = in / s_cols, .mode = .affine };
        }
    }
    if (w_cols > 0 and s_cols > 0) {
        const ratio = (w_cols * 32) / s_cols; // = bits × group_size
        return .{ .bits = @max(ratio / 64, 1), .group_size = 64, .mode = .affine };
    }
    return .{ .bits = 8, .group_size = 64, .mode = .affine };
}

fn moeRoutingChain(router_logits: mlx.mlx_array, k: c_int, s: mlx.mlx_stream) !Transformer.MoeRouting {
    var neg_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_logits);
    try mlx.check(mlx.mlx_negative(&neg_logits, router_logits, s));

    var partitioned = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(partitioned);
    try mlx.check(mlx.mlx_argpartition_axis(&partitioned, neg_logits, k - 1, -1, s));

    const p_shape = mlx.getShape(partitioned);
    var inds = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inds);
    {
        var start_arr: [4]c_int = undefined;
        var stop_arr: [4]c_int = undefined;
        var strides_arr: [4]c_int = undefined;
        for (0..p_shape.len) |d| {
            start_arr[d] = 0;
            stop_arr[d] = if (d == p_shape.len - 1) k else p_shape[d];
            strides_arr[d] = 1;
        }
        try mlx.check(mlx.mlx_slice(&inds, partitioned, &start_arr, p_shape.len, &stop_arr, p_shape.len, &strides_arr, p_shape.len, s));
    }

    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, router_logits, -1, true, s));

    var top_weights = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top_weights);
    try mlx.check(mlx.mlx_take_along_axis(&top_weights, probs, inds, -1, s));

    var weight_sum_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weight_sum_raw);
    try mlx.check(mlx.mlx_sum_axis(&weight_sum_raw, top_weights, -1, false, s));
    var weight_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weight_sum);
    try mlx.check(mlx.mlx_expand_dims(&weight_sum, weight_sum_raw, -1, s));
    var norm_scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(norm_scores);
    try mlx.check(mlx.mlx_divide(&norm_scores, top_weights, weight_sum, s));

    return .{ .inds = inds, .norm_scores = norm_scores };
}


// ── Fused MoE router (top-K selection + weight normalization in ONE kernel) ──
//
// The op chains above are ~11 dispatches each — sigmoid/softmax, bias add,
// negate, argpartition (a full 256-wide sort), slice, take_along_axis, sum,
// eps add, divide, scale multiply, cast — over a [rows, E] tensor that is 512
// bytes wide at decode. Every one of them is launch-latency bound, and a MoE
// model pays the whole chain once per LAYER: 40 layers x 11 = 440 dispatches
// per token, none of which move meaningful data.
//
// One threadgroup per row does all of it: phase 1 computes the selection key
// for every expert into threadgroup memory, phase 2 runs K rounds of
// simdgroup-wide argmax with a LOWEST-INDEX tie-break, phase 3 normalizes the
// selected weights. Selection is exact integer/compare logic, so the picked
// expert set is the same set the chain picks whenever the keys are distinct.
//
// The ARITHMETIC deliberately replicates the chain's rather than improving on
// it — same `fast::exp`, same reciprocal-multiply softmax normalization, same
// MLX `Sigmoid` two-sided form, same intermediate rounding to the output dtype
// at the same points. A "more accurate" fused router is still a DIFFERENT
// model: the first live A/B of this kernel computed the softmax weights in one
// f32 pass and diverged from the chain at token ~80, which
// is the known INT4 near-tie argmax class amplified by an avoidable numerical
// change. The tests below therefore assert BIT equality with the chain, not
// just no-worse-than-reference; only the reduction ORDER over the expert axis
// differs (a simd tree vs MLX's N_READS blocking), which cannot move a bf16
// result except at an exact rounding boundary.
//
// Engagement is LOGGED, never inferred — a declined kernel is output-identical to the chain.
/// `sigmoid_bias_grouped` is `sigmoid_bias` with a GROUP LIMIT spliced
/// between the bias and the top-k: the experts are split into NG equal groups,
/// each group is scored by the SUM OF ITS TOP TWO, and every expert outside the
/// best TOPKG groups is pushed to the composed chain's own finite -1e4 penalty
/// before selection. The composed chain is ~28 dispatches per MoE layer x 41 layers,
/// and a dispatch prices at 1.76 us on this arch (SUSHI_DISPATCH_PROBE
/// slope over 1000 injected ops), so that chain alone is ~2 ms of a 19 ms token.
/// `softmax_gate` is `softmax` plus the shared-expert gate's logit, a dot of one x row with a flat
/// weight vector that the simdgroups sitting out the selection compute meanwhile.
const RouterMode = enum(u8) { softmax = 0, sigmoid_bias = 1, sigmoid_bias_grouped = 2, softmax_gate = 3 };

/// Largest expert count the kernel will stage in threadgroup memory. The
/// sigmoid arm keeps two f32 planes (key + raw weight), so 2048 experts is
/// 16 KiB of the 32 KiB budget.
const ROUTER_MAX_EXPERTS: c_int = 2048;

fn moeRouterSource(comptime mode: RouterMode) [:0]const u8 {
    const sigmoid = mode == .sigmoid_bias or mode == .sigmoid_bias_grouped;
    const grouped = mode == .sigmoid_bias_grouped;
    const gate = mode == .softmax_gate;
    return std.fmt.comptimePrint(
        \\uint row = thread_position_in_grid.y;
        \\uint tid = thread_position_in_threadgroup.x;
        \\uint lane = thread_index_in_simdgroup;
        \\
        \\threadgroup float rk[NE];
        \\{s}
        \\threadgroup uint sel[NK];
        \\{s}
        \\size_t rbase = (size_t)row * (size_t)NE;
        \\for (uint e = tid; e < NE; e += TG) {{
        \\{s}
        \\}}
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\{s}
        \\// Selection runs on simdgroup 0 only: K rounds of "max, then mask".
        \\// simd_min over the candidate indices makes the tie-break the LOWEST
        \\// expert id, so the pick is a pure function of the keys.
        \\if (tid < 32) {{
        \\{s}
        \\{s}
        \\  for (uint r = 0; r < NK; ++r) {{
        \\    float best = -INFINITY;
        \\    uint bidx = 0xFFFFFFFFu;
        \\    for (uint e = lane; e < NE; e += 32) {{
        \\      float v = rk[e];
        \\      if (v > best) {{ best = v; bidx = e; }}
        \\    }}
        \\    float gmax = simd_max(best);
        \\    uint cand = (best == gmax) ? bidx : 0xFFFFFFFFu;
        \\    uint gidx = metal::min(simd_min(cand), uint(NE - 1));
        \\    if (lane == 0) {{ sel[r] = gidx; rk[gidx] = -INFINITY; }}
        \\    simdgroup_barrier(mem_flags::mem_threadgroup);
        \\  }}
        \\
        \\  float w = 0.0f;
        \\  uint myidx = 0;
        \\  if (lane < NK) {{
        \\    myidx = sel[lane];
        \\{s}
        \\  }}
        \\  // MLX's `row_reduce_small` owns an 8-wide row in ONE thread and
        \\  // folds it in ASCENDING order in the reduce's accumulator type —
        \\  // which for a bf16 input is bf16, rounding after every add. A simd
        \\  // butterfly over the same values is a different number.
        \\  float tot = 0.0f;
        \\{s}
        \\{s}
        \\  if (lane < NK) {{
        \\    inds[(size_t)row * (size_t)NK + lane] = myidx;
        \\    scores[(size_t)row * (size_t)NK + lane] = TOUT(w);
        \\  }}
        \\}}
        \\{s}
    , .{
        if (grouped)
            "threadgroup float rw[NE];\nthreadgroup float gs[NG];"
        else if (sigmoid)
            "threadgroup float rw[NE];"
        else
            "",
        if (gate) "threadgroup float gp[16];" else "",
        if (sigmoid)
            // MLX's `Sigmoid` unary op, verbatim (unary_ops.h): the two-sided
            // form, not 1/(1+exp(-x)) — they differ in the last ulp. `exp` is
            // spelled PRECISE because copying upstream's TEXT is not copying
            // its ARITHMETIC: MLX's metallib is built `-fno-fast-math`
            // (kernels/CMakeLists.txt), so its bare `metal::exp` is the precise
            // one, while this source is JIT-compiled at MathMode::Safe where
            // the same spelling lowers to the fast variant. Measured 13.8% of
            // 1M random f32 sigmoids 1 ulp apart, and 6.1% of the routing
            // weights that come out of them; precise is 0/1M. The softmax arm
            // below spells `fast::exp` for the mirror-image reason (softmax.h
            // asks for fast explicitly). Costs ~1% of the kernel at an 8192-row
            // prefill width and nothing measurable at decode.
            \\  float l = float(logits[rbase + e]);
            \\  float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(l)));
            \\  float sg = (l < 0.0f) ? sy : (1.0f - sy);
            \\  rk[e] = sg + float(bias[e]);
            \\  rw[e] = sg;
        else
            \\  rk[e] = float(logits[rbase + e]);
        ,
        if (gate)
            // MLX's `dot_product` for a bf16/f16 [1,K]x[K,1] matmul, reproduced lane for lane: one
            // virtual simdgroup per 1024 elements, a float4 accumulator over four strided 8-element
            // chunks, then c0+c1+c2+c3 and a simd_sum. Any other order changes the rounded logit.
            \\const uint GSG = (GK + 1023) / 1024;
            \\if (tid >= 32 && tid < 32 * (1 + GSG)) {
            \\  const uint vs = tid / 32 - 1;
            \\  float4 c = 0.0f;
            \\  for (uint i = 0; i < 32; i += 8) {
            \\    const uint idx = vs * 1024 + lane * 8 + i * 32;
            \\    if (idx + 8 <= uint(GK)) {
            \\      for (uint j = 0; j < 8; j += 4) {
            \\        c += float4(*reinterpret_cast<const device metal::vec<TX, 4>*>(gx + (size_t)row * (size_t)GK + idx + j)) *
            \\             float4(*reinterpret_cast<const device metal::vec<TX, 4>*>(gw + idx + j));
            \\      }
            \\    }
            \\  }
            \\  float ga = c[0] + c[1] + c[2] + c[3];
            \\  ga = simd_sum(ga);
            \\  if (lane == 0) gp[vs] = ga;
            \\}
        else
            "",
        if (grouped)
            // The group limit (`group_limited_topk` shape). One lane
            // per group scores it by the SUM OF ITS TOP TWO biased experts, then
            // TOPKG rounds of the same max-then-mask/lowest-index selection the
            // expert top-k uses, then every expert in a losing group is pushed to
            // -1e4 — the composed chain's own FINITE penalty, not -inf: biased
            // scores are sigmoid+bias so they live well inside +/-2, and a
            // non-finite sentinel in a tensor that feeds selection is how the
            // composed causal-mask arm once shipped an all-NaN path that scored a
            // perfect parity max_err of 0.
            \\  for (uint g = lane; g < NG; g += 32) {
            \\    float m1 = -INFINITY, m2 = -INFINITY;
            \\    for (uint j = 0; j < GSZ; ++j) {
            \\      float v = rk[g * GSZ + j];
            \\      if (v > m1) { m2 = m1; m1 = v; } else if (v > m2) { m2 = v; }
            \\    }
            \\    gs[g] = m1 + m2;
            \\  }
            \\  simdgroup_barrier(mem_flags::mem_threadgroup);
            \\  uint keep_mask = 0u;
            \\  for (uint r = 0; r < TOPKG; ++r) {
            \\    float best = -INFINITY;
            \\    uint bidx = 0xFFFFFFFFu;
            \\    for (uint g = lane; g < NG; g += 32) {
            \\      float v = gs[g];
            \\      if (v > best) { best = v; bidx = g; }
            \\    }
            \\    float gmax = simd_max(best);
            \\    uint cand = (best == gmax) ? bidx : 0xFFFFFFFFu;
            \\    uint gidx = metal::min(simd_min(cand), uint(NG - 1));
            \\    keep_mask |= (1u << gidx);
            \\    if (lane == 0) gs[gidx] = -INFINITY;
            \\    simdgroup_barrier(mem_flags::mem_threadgroup);
            \\  }
            \\  for (uint e = lane; e < NE; e += 32) {
            \\    if ((keep_mask & (1u << (e / GSZ))) == 0u) rk[e] = -1.0e4f;
            \\  }
            \\  simdgroup_barrier(mem_flags::mem_threadgroup);
        else
            "",
        if (sigmoid)
            ""
        else
            // MLX's `softmax_single_row`, reduction tree included. It launches
            // ceil(NE/4) threads, each summing FOUR consecutive elements in
            // order, then one `simd_sum` per simdgroup, then a final `simd_sum`
            // over the per-simdgroup partials sitting in lanes 0..SIMDS-1 with
            // zeros above. Emulating those virtual threads 32 at a time
            // reproduces the identical butterfly on identical values, which is
            // what makes the rounded bf16 probability bit-equal; a plain
            // strided sum over the same data was 1 ulp off on real inputs.
            \\  constexpr int VT = (NE + 3) / 4;
            \\  constexpr int SIMDS = (VT + 31) / 32;
            \\  float lane_max = -INFINITY;
            \\  for (int g = 0; g < SIMDS; ++g) {
            \\    int vt = g * 32 + int(lane);
            \\    float m = -INFINITY;
            \\    for (int i = 0; i < 4; ++i) {
            \\      int e = vt * 4 + i;
            \\      if (e < NE) m = metal::max(m, rk[e]);
            \\    }
            \\    float mg = simd_max(m);
            \\    if (int(lane) == g) lane_max = mg;
            \\  }
            \\  float smax = simd_max(lane_max);
            \\  float lane_sum = 0.0f;
            \\  for (int g = 0; g < SIMDS; ++g) {
            \\    int vt = g * 32 + int(lane);
            \\    float acc = 0.0f;
            \\    for (int i = 0; i < 4; ++i) {
            \\      int e = vt * 4 + i;
            \\      acc += (e < NE) ? fast::exp(rk[e] - smax) : 0.0f;
            \\    }
            \\    float pg = simd_sum(acc);
            \\    if (int(lane) == g) lane_sum = pg;
            \\  }
            \\  float snorm = 1.0f / simd_sum(lane_sum);
        ,
        if (sigmoid)
            \\    w = rw[myidx];
        else
            // softmax(...) writes probabilities in the OUTPUT dtype, so the
            // per-expert rounding happens here, before the top-K sum.
            \\    w = float(TOUT(fast::exp(float(logits[rbase + myidx]) - smax) * snorm));
        ,
        if (sigmoid)
            // The chain sums FLOAT32 sigmoid scores, so the accumulator is
            // float — but still ascending, still sequential.
            \\  for (int j = 0; j < NK; ++j) tot += simd_shuffle(w, ushort(j));
        else
            \\  for (int j = 0; j < NK; ++j) tot = float(TOUT(simd_shuffle(w, ushort(j)) + tot));
        ,
        if (sigmoid)
            \\  if (NORM != 0 && NK > 1) {
            \\    w = w / (tot + 1e-20f);
            \\  }
            \\  // `scale` is a 0-dim array, which MLX binds as `const constant
            \\  // float&` — a reference, not a subscriptable pointer.
            \\  w = w * scale;
        else
            // `sum` over the K probabilities lands back in the output dtype
            // before the divide, exactly as mlx_sum_axis + mlx_divide do.
            \\  w = w / float(TOUT(tot));
        ,
        if (gate)
            // MLX folds the per-simdgroup sums with a second simd_sum over 16 lanes, zeros past the last.
            \\threadgroup_barrier(mem_flags::mem_threadgroup);
            \\if (tid < 16) {
            \\  float gt = (tid < GSG) ? gp[tid] : 0.0f;
            \\  gt = simd_sum(gt);
            \\  if (tid == 0) glogit[row] = TOUT(gt);
            \\}
        else
            "",
    });
}

const ROUTER_SOURCES = [4][:0]const u8{ moeRouterSource(.softmax), moeRouterSource(.sigmoid_bias), moeRouterSource(.sigmoid_bias_grouped), moeRouterSource(.softmax_gate) };
const ROUTER_NAMES = [4][*:0]const u8{ "sushi_moe_router_softmax", "sushi_moe_router_sigmoid", "sushi_moe_router_sigmoid_grouped", "sushi_moe_router_softmax_gate" };
var router_kernels: [4]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getMoeRouterKernel(mode: RouterMode) !mlx.mlx_fast_metal_kernel {
    const mi: usize = @backingInt(mode);
    if (router_kernels[mi]) |k| return k;
    const softmax_inputs = [_][*:0]const u8{"logits"};
    const gate_inputs = [_][*:0]const u8{ "logits", "gx", "gw" };
    const sigmoid_inputs = [_][*:0]const u8{ "logits", "bias", "scale" };
    const input_names: []const [*:0]const u8 = switch (mode) {
        .softmax => &softmax_inputs,
        .softmax_gate => &gate_inputs,
        .sigmoid_bias, .sigmoid_bias_grouped => &sigmoid_inputs,
    };
    const plain_outputs = [_][*:0]const u8{ "inds", "scores" };
    const gate_outputs = [_][*:0]const u8{ "inds", "scores", "glogit" };
    const output_names: []const [*:0]const u8 = if (mode == .softmax_gate) &gate_outputs else &plain_outputs;
    const in_vec = mlx.mlx_vector_string_new_data(input_names.ptr, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(output_names.ptr, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        ROUTER_NAMES[mi],
        in_vec,
        out_vec,
        ROUTER_SOURCES[mi],
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    router_kernels[mi] = kernel;
    return kernel;
}

/// Threads per row. Selection needs a full simdgroup; phase 1 wants enough
/// threads to cover the expert row without a long strided loop.
fn moeRouterThreadGroup(num_experts: c_int) c_int {
    const rounded = @divTrunc(num_experts + 31, 32) * 32;
    return @max(@as(c_int, 32), @min(@as(c_int, 256), rounded));
}

/// Everything the cached kernel config bakes in. A mismatch re-arms it.
/// Full-shape component for cached metal-kernel config keys. A cached config
/// bakes the OUTPUT SHAPES (and grid) of the call that built it, so the key
/// must carry the caller's actual dims: keying on a PRODUCT (rows, n) collides
/// across layouts with equal element counts and hands the colliding caller an
/// output labeled with the FIRST caller's shape. Live 2026-07-29:
/// a MoE-layer SwiGLU at a 16-token hot-cache-restore prefill is (1,16,512) =
/// 8192 elements — the same count/rank/dtype as a dense mlp_only layer at
/// decode (1,1,8192) — and the mislabeled activation killed the server in the
/// dense down-proj matmul. Every cached config key whose config bakes an
/// input-derived shape uses this instead of derived products.
pub const ShapeKey = struct {
    dims: [8]c_int = @splat(0),
    len: u8 = 0,
    pub fn from(sh: []const c_int) ShapeKey {
        var key = ShapeKey{};
        for (sh, 0..) |d, i| key.dims[i] = d;
        key.len = @intCast(sh.len);
        return key;
    }
};

const RouterCfgKey = struct {
    shape: ShapeKey,
    k: c_int,
    out_dtype: mlx.mlx_dtype,
    norm: bool,
    scale: f32,
    n_group: c_int = 0,
    topk_group: c_int = 0,
    gate_k: c_int = 0,
    gate_dtype: mlx.mlx_dtype = .bfloat16,
};
const RouterCfgSlot = struct {
    cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null },
    key: RouterCfgKey = std.mem.zeroes(RouterCfgKey),
    scale_arr: mlx.mlx_array = .{ .ctx = null },
};
var router_cfg_cache: [4]RouterCfgSlot = .{ .{}, .{}, .{}, .{} };

pub var moe_router_fused_override: ?bool = null;
var moe_router_engaged: bool = false;

pub fn moeRouterFusedEnabled() bool {
    return moe_router_fused_override orelse true;
}

/// The shared-expert gate's logit folded into a softmax router: one `x` row and the flat weight vector `w`.
pub const RouterGate = struct { x: mlx.mlx_array, w: mlx.mlx_array };

/// One-kernel replacement for `moeRoutingChain` / `sigmoidBiasRoutingChain`.
/// Returns null when the geometry or dtype is outside the supported set, so
/// every caller keeps its chain fallback.
pub fn moeRouterTopK(
    s: mlx.mlx_stream,
    logits: mlx.mlx_array,
    bias: mlx.mlx_array,
    k: c_int,
    mode: RouterMode,
    route_norm: bool,
    route_scale: f32,
    out_dtype: mlx.mlx_dtype,
    n_group: c_int,
    topk_group: c_int,
) !?Transformer.MoeRouting {
    return moeRouterTopKGate(s, logits, bias, k, mode, route_norm, route_scale, out_dtype, n_group, topk_group, null);
}

fn moeRouterTopKGate(
    s: mlx.mlx_stream,
    logits: mlx.mlx_array,
    bias: mlx.mlx_array,
    k: c_int,
    mode_in: RouterMode,
    route_norm: bool,
    route_scale: f32,
    out_dtype: mlx.mlx_dtype,
    n_group: c_int,
    topk_group: c_int,
    gate: ?RouterGate,
) !?Transformer.MoeRouting {
    if (!moeRouterFusedEnabled()) return null;
    const mode: RouterMode = if (gate != null and mode_in == .softmax) .softmax_gate else mode_in;
    const sigmoid_mode = mode == .sigmoid_bias or mode == .sigmoid_bias_grouped;
    const lsh = mlx.getShape(logits);
    if (lsh.len < 1 or lsh.len > 4) return null;
    const num_experts = lsh[lsh.len - 1];
    if (num_experts < k or num_experts > ROUTER_MAX_EXPERTS) return null;
    if (k < 1 or k > 32) return null;
    const ldt = mlx.mlx_array_dtype(logits);
    if (ldt != .bfloat16 and ldt != .float16 and ldt != .float32) return null;
    if (sigmoid_mode and bias.ctx == null) return null;
    if (mode == .sigmoid_bias_grouped) {
        // One lane per group, one bit per group in the keep mask, and a group
        // score that is a top-2 sum. Anything outside that declines to the
        // composed chain rather than computing a different model.
        if (n_group <= 1 or n_group > 32) return null;
        if (topk_group <= 0 or topk_group > n_group) return null;
        if (@rem(num_experts, n_group) != 0) return null;
        if (@divExact(num_experts, n_group) < 2) return null;
    }

    var rows: c_int = 1;
    for (lsh[0 .. lsh.len - 1]) |d| rows *= d;
    if (rows < 1) return null;

    const tg = moeRouterThreadGroup(num_experts);
    // Bit-equality holds only for the one-row dot MLX runs as a single 512-thread block, and each
    // 1024 elements of it needs a simdgroup besides the selecting one.
    var gate_k: c_int = 0;
    var gate_dtype: mlx.mlx_dtype = .bfloat16;
    if (gate) |gt| {
        const wn = mlx.mlx_array_size(gt.w);
        gate_dtype = mlx.mlx_array_dtype(gt.x);
        if (rows != 1 or wn == 0 or wn > 16384 or wn % 8 != 0 or mlx.mlx_array_size(gt.x) != wn) return null;
        if ((gate_dtype != .bfloat16 and gate_dtype != .float16) or mlx.mlx_array_dtype(gt.w) != gate_dtype or out_dtype != gate_dtype) return null;
        if (@divTrunc(tg, 32) - 1 < @as(c_int, @intCast((wn + 1023) / 1024))) return null;
        gate_k = @intCast(wn);
    }

    // The whole point of the kernel is to spend ONE dispatch where the chain
    // spent a dozen; rebuilding the config every call would just move that cost
    // to the CPU side (measured +0.16 ms of graph-build per token on a 40-layer MoE).
    // A model's routing geometry is fixed, so one cached config per mode covers
    // decode, and prefill re-arms it once per width.
    const key = RouterCfgKey{
        .shape = ShapeKey.from(lsh),
        .k = k,
        .out_dtype = out_dtype,
        .norm = route_norm,
        .scale = route_scale,
        .n_group = n_group,
        .topk_group = topk_group,
        .gate_k = gate_k,
        .gate_dtype = gate_dtype,
    };
    const slot = &router_cfg_cache[@backingInt(mode)];
    if (slot.cfg.ctx == null or !std.meta.eql(slot.key, key)) {
        if (slot.cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(slot.cfg);
        if (slot.scale_arr.ctx != null) _ = mlx.mlx_array_free(slot.scale_arr);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        // Outputs keep the caller's leading dims with the expert axis replaced by K.
        var out_shape: [4]c_int = undefined;
        for (lsh[0 .. lsh.len - 1], 0..) |d, i| out_shape[i] = d;
        out_shape[lsh.len - 1] = k;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, lsh.len, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, lsh.len, out_dtype));
        if (mode == .softmax_gate) {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{1}, 1, out_dtype));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GK", gate_k));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "TX", gate_dtype));
        }
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "TOUT", out_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NE", num_experts));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NK", k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TG", tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NORM", @intFromBool(route_norm)));
        if (mode == .sigmoid_bias_grouped) {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NG", n_group));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GSZ", @divExact(num_experts, n_group)));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TOPKG", topk_group));
        }
        slot.* = .{ .cfg = config, .key = key, .scale_arr = mlx.mlx_array_new_float(route_scale) };
    }
    const config = slot.cfg;

    const softmax_inputs = [_]mlx.mlx_array{logits};
    const gate_inputs = [_]mlx.mlx_array{ logits, if (gate) |gt| gt.x else logits, if (gate) |gt| gt.w else logits };
    const sigmoid_inputs = [_]mlx.mlx_array{ logits, bias, slot.scale_arr };
    const inputs_arr: []const mlx.mlx_array = switch (mode) {
        .softmax => &softmax_inputs,
        .softmax_gate => &gate_inputs,
        .sigmoid_bias, .sigmoid_bias_grouped => &sigmoid_inputs,
    };
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs_arr.ptr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getMoeRouterKernel(mode);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != (if (mode == .softmax_gate) @as(usize, 3) else 2)) return error.MetalKernelBadOutputCount;

    var inds = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inds);
    try mlx.check(mlx.mlx_vector_array_get(&inds, outputs_vec, 0));
    var scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_vector_array_get(&scores, outputs_vec, 1));
    var gate_logit = mlx.mlx_array{ .ctx = null };
    if (mode == .softmax_gate) {
        gate_logit = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&gate_logit, outputs_vec, 2));
    }

    if (!moe_router_engaged) {
        moe_router_engaged = true;
        log.info("[moe] fused router kernel engaged: mode={s} E={d} K={d} tg={d}\n", .{ @tagName(mode), num_experts, k, tg });
    }
    return .{ .inds = inds, .norm_scores = scores, .gate_logit = gate_logit };
}

// ── Decode sub-block profiler (SUSHI_DECODE_PROFILE=1) ──
// Serializes the S=1 decode forward with per-sub-block evals to attribute the
// per-token GPU cost across embed / attn / mlp / lm_head. DIAGNOSTIC ONLY: the
// evals defeat the async pipeline (absolute tok/s drops while active), so the
// signal is the RELATIVE split plus the serialized total vs the pipelined
// per-token time — a serialized total near the pipelined per-token means we are
// compute-bound; far below it means the gap is dispatch/pipeline overhead.
const DecodeProf = struct {
    calls: u64 = 0,
    embed_ns: u64 = 0,
    attn_ns: u64 = 0,
    mlp_ns: u64 = 0,
    lmhead_ns: u64 = 0,
    // MoE internals (summed across all MoE layers per token)
    moe_router_ns: u64 = 0,
    moe_experts_ns: u64 = 0,
    moe_shared_ns: u64 = 0,
};
/// One narrowed copy of a load-time constant table, keyed by dtype.
const DtypeCastCache = struct {
    arr: mlx.mlx_array = .{ .ctx = null },
    dtype: mlx.mlx_dtype = .float32,

    pub fn deinit(self: *DtypeCastCache) void {
        if (self.arr.ctx != null) _ = mlx.mlx_array_free(self.arr);
        self.arr = .{ .ctx = null };
    }
};

/// Hand back `table` in the dtype the activation it scales is in, building and
/// caching one narrowed copy on first use. Returns a BORROWED array — `cache`
/// owns it.
///
/// A constant table built f32 at load and multiplied straight into a bf16
/// activation promotes the PRODUCT to f32 (mlx's promotion rule). Where that
/// product feeds the residual stream, every projection and expert weight for
/// the rest of the forward is upcast on read — a ~3x token cost that shows up
/// in no op count, no kernel choice, and no pre-loop dtype log. Same class as
/// the MageFlow `scalarLike` rule.
pub fn constTableAs(table: mlx.mlx_array, want: mlx.mlx_dtype, cache: *DtypeCastCache, s: mlx.mlx_stream) !mlx.mlx_array {
    if (mlx.mlx_array_dtype(table) == want) return table;
    if (cache.arr.ctx != null and cache.dtype == want) return cache.arr;
    cache.deinit();
    var cast = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cast);
    try mlx.check(mlx.mlx_astype(&cast, table, want, s));
    try mlx.check(mlx.mlx_array_eval(cast));
    cache.arr = cast;
    cache.dtype = want;
    return cast;
}

/// DIAGNOSTIC (SUSHI_LAYER_CAP): clamp the layer loop so a ms-vs-layers
/// sweep separates per-layer slope from fixed cost. Unset — the only shipped
/// state — returns the model's real depth.
var layer_cap_env: ?usize = null;
pub fn layerCap(n: usize) usize {
    if (layer_cap_env == null) {
        layer_cap_env = blk: {
            const raw = std.c.getenv("SUSHI_LAYER_CAP") orelse break :blk 0;
            break :blk std.fmt.parseInt(usize, std.mem.sliceTo(raw, 0), 10) catch 0;
        };
    }
    const cap = layer_cap_env.?;
    return if (cap == 0) n else @min(cap, n);
}

pub var decode_prof: DecodeProf = .{};
var decode_prof_enabled: ?bool = null;

var standin_ones: ?mlx.mlx_array = null;
pub fn standinOnes(shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    if (standin_ones) |a| {
        if (std.mem.eql(c_int, mlx.getShape(a), shape)) {
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&r, a));
            return r;
        }
        _ = mlx.mlx_array_free(a);
    }
    var a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_ones(&a, shape.ptr, @intCast(shape.len), .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(a));
    standin_ones = a;
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&r, a));
    return r;
}
pub fn standinRef(x: mlx.mlx_array) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&r, x));
    return r;
}



pub const ProfClock = struct {
    io: std.Io,
    start: std.Io.Timestamp,
    mark_ns: u64 = 0,
    pub fn init() ProfClock {
        const io = std.Io.Threaded.global_single_threaded.io();
        return .{ .io = io, .start = std.Io.Timestamp.now(io, .boot) };
    }
    pub fn lap(self: *ProfClock) u64 {
        const cum: u64 = @intCast(self.start.untilNow(self.io, .boot).nanoseconds);
        const d = cum - self.mark_ns;
        self.mark_ns = cum;
        return d;
    }
};


pub fn decodeProfileEnabled() bool {
    if (decode_prof_enabled) |v| return v;
    const v = diagEnvOn("SUSHI_DECODE_PROFILE");
    decode_prof_enabled = v;
    return v;
}

/// Widest forward the profiler times: `SUSHI_DECODE_PROFILE=<rows>` also times verify widths.
pub fn decodeProfileRows() c_int {
    if (decode_prof_forced_rows > 0) return decode_prof_forced_rows;
    return decodeProfileRowsFrom(std.c.getenv("SUSHI_DECODE_PROFILE"));
}

fn decodeProfileRowsFrom(raw: ?[*:0]const u8) c_int {
    if (!diagEnvValueOn(raw)) return 0;
    return @max(1, std.fmt.parseInt(c_int, std.mem.sliceTo(raw.?, 0), 10) catch 1);
}
var decode_prof_forced_rows: c_int = 0;

/// The forward microbench's profiled pass: `rows > 0` times forwards up to that
/// width, `rows == 0` reports what was timed and restores the env's setting.
pub fn decodeProfileSession(rows: c_int) void {
    if (rows > 0) {
        decode_prof = .{};
        decode_prof_forced_rows = rows;
        decode_prof_enabled = true;
        return;
    }
    decodeProfReport();
    decode_prof_forced_rows = 0;
    decode_prof_enabled = null;
}

pub const diagEnvValueOn = io_util_mod.diagEnvValueOn;
/// SUSHI_MOE_DUMP=<dir> writes one forward's MoE tensors as raw f32, with shapes in filenames.
var moe_dump_dir: ?[]const u8 = null;
var moe_dump_asked: bool = false;
pub var moe_dump_layer: u32 = 0;
var moe_dump_done: bool = false;
var moe_dump_active: bool = false;
pub var moe_dump_warmup_depth: u32 = 0;

pub fn moeDumpBeginForward() bool {
    if (moe_dump_warmup_depth != 0 or moe_dump_done or moe_dump_active or moeDumpDir() == null) return false;
    moe_dump_active = true;
    return true;
}

fn moeDumpDir() ?[]const u8 {
    if (!moe_dump_asked) {
        moe_dump_asked = true;
        if (std.c.getenv("SUSHI_MOE_DUMP")) |p| {
            const v = std.mem.span(p);
            if (v.len > 0 and v[0] != '0') moe_dump_dir = v;
        }
    }
    return moe_dump_dir;
}

/// One forward only: the layer counter walks the MoE layers in order and the
/// dump latches off when it has seen a whole model's worth.
pub fn moeDumpForwardDone() void {
    moe_dump_active = false;
    moe_dump_done = true;
}

/// One tensor, evaluated and written as f32. Never on a serving path: the dump
/// is a host read per layer, which is a GPU barrier by construction.
pub fn moeDumpTensor(s: mlx.mlx_stream, tag: []const u8, layer: u32, a: mlx.mlx_array) void {
    const dir = moeDumpDir() orelse return;
    if (!moe_dump_active or moe_dump_done) return;
    var f32a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32a);
    if (mlx.mlx_astype(&f32a, a, .float32, s) != 0) return;
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    if (mlx.mlx_contiguous(&c, f32a, false, s) != 0) return;
    if (mlx.mlx_array_eval(c) != 0) return;
    const p = mlx.mlx_array_data_float32(c) orelse return;
    const shape = mlx.getShape(c);
    var name: [512]u8 = undefined;
    var used: usize = 0;
    used += (std.fmt.bufPrint(name[used..], "{s}/L{d:0>2}_{s}", .{ dir, layer, tag }) catch return).len;
    var n: usize = 1;
    for (shape) |d| {
        n *= @intCast(d);
        used += (std.fmt.bufPrint(name[used..], "_{d}", .{d}) catch return).len;
    }
    used += (std.fmt.bufPrint(name[used..], ".f32\x00", .{}) catch return).len;
    const path: [*:0]const u8 = @ptrCast(name[0 .. used - 1].ptr);
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return;
    defer _ = std.c.close(fd);
    const bytes = std.mem.sliceAsBytes(p[0..n]);
    var put: usize = 0;
    while (put < bytes.len) {
        const w = std.c.write(fd, bytes.ptr + put, bytes.len - put);
        if (w <= 0) return;
        put += @intCast(w);
    }
}

test "moe dump waits for a real forward before writing tensors" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const saved_dir = moe_dump_dir;
    const saved_asked = moe_dump_asked;
    const saved_done = moe_dump_done;
    const saved_active = moe_dump_active;
    const saved_warmup = moe_dump_warmup_depth;
    const saved_layer = moe_dump_layer;
    defer {
        moe_dump_dir = saved_dir;
        moe_dump_asked = saved_asked;
        moe_dump_done = saved_done;
        moe_dump_active = saved_active;
        moe_dump_warmup_depth = saved_warmup;
        moe_dump_layer = saved_layer;
    }
    moe_dump_dir = path_buf[0..path_len];
    moe_dump_asked = true;
    moe_dump_done = false;
    moe_dump_active = false;
    moe_dump_warmup_depth = 0;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const values = [_]f32{ 1, 2, 3, 4 };
    const a = mlx.mlx_array_new_data(&values, &.{ 1, 2, 2 }, 3, .float32);
    defer _ = mlx.mlx_array_free(a);
    moeDumpTensor(s, "x", 1, a);
    try t.expectError(error.FileNotFound, tmp.dir.openFile(t.io, "L01_x_1_2_2.f32", .{}));

    const config = ModelConfig{ .model_type = "mimo_v2", .num_hidden_layers = 3, .first_k_dense_replace = 1 };
    const forward = struct {
        fn run(cfg: ModelConfig, stream: mlx.mlx_stream, x: mlx.mlx_array) void {
            const dumping = moeDumpBeginForward();
            defer if (dumping) moeDumpForwardDone();
            for (cfg.first_k_dense_replace..cfg.num_hidden_layers) |li| {
                if (dumping) moe_dump_layer = @intCast(li);
                moeDumpTensor(stream, "x", moe_dump_layer, x);
            }
        }
    }.run;
    moe_dump_warmup_depth += 1;
    forward(config, s, a);
    moe_dump_warmup_depth -= 1;
    try t.expectError(error.FileNotFound, tmp.dir.openFile(t.io, "L01_x_1_2_2.f32", .{}));
    try t.expect(!moe_dump_done);
    forward(config, s, a);
    for ([_][]const u8{ "L01_x_1_2_2.f32", "L02_x_1_2_2.f32" }) |name| {
        const bytes = try tmp.dir.readFileAlloc(t.io, name, t.allocator, .limited(128));
        defer t.allocator.free(bytes);
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&values), bytes);
    }
    try t.expect(moe_dump_done and !moe_dump_active);
    const later_values = [_]f32{ 5, 6, 7, 8 };
    const later = mlx.mlx_array_new_data(&later_values, &.{ 1, 2, 2 }, 3, .float32);
    defer _ = mlx.mlx_array_free(later);
    forward(config, s, later);
    const unchanged = try tmp.dir.readFileAlloc(t.io, "L01_x_1_2_2.f32", t.allocator, .limited(128));
    defer t.allocator.free(unchanged);
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&values), unchanged);
    var files = tmp.dir.iterate();
    var count: usize = 0;
    while (try files.next(t.io)) |_| count += 1;
    try t.expectEqual(@as(usize, 2), count);
}

var expert_swap_logged: bool = false;
var expert_swap_total_swapped: u64 = 0;
var expert_swap_total_ids: u64 = 0;
var expert_swap_last_report: u64 = 0;

/// The running swap count is logged at the first swap, so a short run shows the pick swapping, then every 100k ids.
fn swapReportDue(first_swap: bool, total_ids: u64, last_report: u64) bool {
    return first_swap or total_ids -% last_report >= 100_000;
}

test "the swap count is reported at the first swap and then every 100k routed ids" {
    try testing.expect(swapReportDue(true, 376, 0));
    try testing.expect(!swapReportDue(false, 376, 0));
    try testing.expect(swapReportDue(false, 100_000, 0));
    try testing.expect(!swapReportDue(false, 199_999, 100_000));
}

const STREAM_DECODE_MAX_ROWS: c_int = 16;

fn streamRows(shape: []const c_int) c_int {
    var rows: c_int = 1;
    for (shape[0 .. shape.len - 1]) |d| rows *= d;
    return rows;
}

fn streamHandoffApplies(rows: c_int) bool {
    return rows <= STREAM_DECODE_MAX_ROWS;
}

fn streamSharedEarlyApplies(eligible: bool, inds_shape: []const c_int) bool {
    return eligible and streamRows(inds_shape) <= STREAM_DECODE_MAX_ROWS;
}

/// What the lossy pick compares: softmax logits differ by the log of a probability ratio as they are; a
/// sigmoid router's do once mapped to log sigmoid, unbiased (the correction bias only selects).
const SwapLogits = enum { softmax, log_sigmoid };

fn expertSwapKind(tolerance: f32, biased_router: bool, sigmoid_router: bool, imatrix_armed: bool, inds_shape: []const c_int) ?SwapLogits {
    if (!(tolerance > 0) or imatrix_armed or streamRows(inds_shape) > STREAM_DECODE_MAX_ROWS) return null;
    if (sigmoid_router) return .log_sigmoid;
    return if (biased_router) null else .softmax;
}

/// The pick's f32 logits; log sigmoid as -logaddexp(0, -x), finite where sigmoid underflows.
fn routerSwapLogits(out: *mlx.mlx_array, router_logits: mlx.mlx_array, kind: SwapLogits, s: mlx.mlx_stream) !void {
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_astype(&wide, router_logits, .float32, s));
    if (kind == .softmax) return mlx.check(mlx.mlx_contiguous(out, wide, false, s));
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, wide, s));
    const zero = mlx.mlx_array_new_float(0);
    defer _ = mlx.mlx_array_free(zero);
    var softplus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(softplus);
    try mlx.check(mlx.mlx_logaddexp(&softplus, neg, zero, s));
    var log_sigmoid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_sigmoid);
    try mlx.check(mlx.mlx_negative(&log_sigmoid, softplus, s));
    try mlx.check(mlx.mlx_contiguous(out, log_sigmoid, false, s));
}

/// One-row twin of `expert_stream.substituteMisses` on the GPU, so a lossy pick
/// needs no host round trip before the layer's experts are queued: misses in
/// descending logit order, each takes the highest-logit ready expert not yet
/// selected within `gap` (lowest index on ties) unless it is starved. Writes the
/// picked ids and their slots. The host still runs its own pick and keeps the
/// GPU result only when both agree.
const EXPERT_PICK_SOURCE =
    \\uint t = thread_position_in_threadgroup.x;
    \\uint T = threads_per_threadgroup.x;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\threadgroup bool taken[E];
    \\threadgroup int sel[K];
    \\threadgroup int order[K];
    \\threadgroup int nmiss;
    \\threadgroup float red_v[32];
    \\threadgroup int red_i[32];
    \\for (uint e = t; e < uint(E); e += T) taken[e] = false;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (t == 0) {
    \\  int n = 0;
    \\  for (int p = 0; p < K; ++p) {
    \\    int e = ids[p];
    \\    sel[p] = e;
    \\    taken[e] = true;
    \\    if (map[e] >= 0) continue;
    \\    int at = n;
    \\    while (at > 0 && logits[ids[order[at - 1]]] < logits[e]) { order[at] = order[at - 1]; at -= 1; }
    \\    order[at] = p;
    \\    n += 1;
    \\  }
    \\  nmiss = n;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float g = gap[0];
    \\for (int i = 0; i < nmiss; ++i) {
    \\  int pos = order[i];
    \\  int missed = sel[pos];
    \\  if (int(starved[missed]) >= LIMIT) continue;
    \\  float lm = logits[missed];
    \\  float best_v = -INFINITY;
    \\  int best_i = -1;
    \\  for (uint e = t; e < uint(E); e += T) {
    \\    if (taken[e] || map[e] < 0) continue;
    \\    float v = logits[e];
    \\    if (!(v - lm >= g) || !(v > best_v)) continue;
    \\    best_v = v;
    \\    best_i = int(e);
    \\  }
    \\  for (uint off = 16; off > 0; off >>= 1) {
    \\    float ov = simd_shuffle_down(best_v, off);
    \\    int oi = simd_shuffle_down(best_i, off);
    \\    if (lane + off < 32 && oi >= 0 && (best_i < 0 || ov > best_v || (ov == best_v && oi < best_i))) { best_v = ov; best_i = oi; }
    \\  }
    \\  if (lane == 0) { red_v[sg] = best_v; red_i[sg] = best_i; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (t == 0) {
    \\    float bv = -INFINITY;
    \\    int bi = -1;
    \\    for (uint s2 = 0; s2 < (T + 31) / 32; ++s2) {
    \\      int oi = red_i[s2];
    \\      float ov = red_v[s2];
    \\      if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
    \\    }
    \\    if (bi >= 0) { sel[pos] = bi; taken[bi] = true; }
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\if (t < uint(K)) {
    \\  int e = sel[t];
    \\  picked[t] = e;
    \\  slots[t] = max(map[e], 0);
    \\}
;

var expert_pick_kernel: ?mlx.mlx_fast_metal_kernel = null;

const ExpertPick = struct { slots: mlx.mlx_array, picked: mlx.mlx_array };

fn expertPickGpu(ids: mlx.mlx_array, logits: mlx.mlx_array, map: mlx.mlx_array, starved: mlx.mlx_array, tolerance: f32, s: mlx.mlx_stream) !ExpertPick {
    const k: c_int = @intCast(mlx.mlx_array_size(ids));
    const experts: c_int = @intCast(mlx.mlx_array_size(map));
    if (k < 1 or k > 32 or experts < 1 or experts > 1024 or mlx.mlx_array_size(logits) != @as(usize, @intCast(experts))) return error.UnsupportedShape;
    const kernel = expert_pick_kernel orelse blk: {
        const input_names = [_][*:0]const u8{ "ids", "logits", "map", "starved", "gap" };
        const output_names = [_][*:0]const u8{ "slots", "picked" };
        const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const made = mlx.mlx_fast_metal_kernel_new("sushi_expert_pick", in_vec, out_vec, EXPERT_PICK_SOURCE, "", true, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        expert_pick_kernel = made;
        break :blk made;
    };
    const gap_value: f32 = @log(1.0 - @min(tolerance, expert_stream_mod.MAX_PICK_TOLERANCE));
    const gap = mlx.mlx_array_new_data(&gap_value, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(gap);
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const out_shape = [_]c_int{k};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 1, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 1, .int32));
    const threads: c_int = @min(1024, @divTrunc(experts + 31, 32) * 32);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K", k));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "E", experts));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "LIMIT", expert_stream_mod.PICK_STARVE_LIMIT));
    const inputs = [_]mlx.mlx_array{ ids, logits, map, starved, gap };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, inputs_vec, config, s));
    var slots = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(slots);
    try mlx.check(mlx.mlx_vector_array_get(&slots, outputs, 0));
    var picked = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&picked, outputs, 1));
    return .{ .slots = slots, .picked = picked };
}

fn slabMoeWeights(mw: *const MoeMlpWeights, raw: *const [expert_stream_mod.quant.component_count]mlx.mlx_array) MoeMlpWeights {
    const C = expert_stream_mod.quant.Component;
    var out = mw.*;
    out.switch_gate_w = raw[@backingInt(C.gate_w)];
    out.switch_gate_s = raw[@backingInt(C.gate_s)];
    out.switch_gate_b = raw[@backingInt(C.gate_b)];
    out.switch_up_w = raw[@backingInt(C.up_w)];
    out.switch_up_s = raw[@backingInt(C.up_s)];
    out.switch_up_b = raw[@backingInt(C.up_b)];
    out.switch_down_w = raw[@backingInt(C.down_w)];
    out.switch_down_s = raw[@backingInt(C.down_s)];
    out.switch_down_b = raw[@backingInt(C.down_b)];
    return out;
}

/// A decode-width streamed layer queues its expert compute from the GPU copy
/// of the cache map before the host reads the router ids; the host keeps it
/// when every id resolved to the gathered slot, else rebuilds it.
fn expertSpecApplies(imatrix_armed: bool, inds_shape: []const c_int) bool {
    return !imatrix_armed and streamRows(inds_shape) <= STREAM_DECODE_MAX_ROWS;
}

var expert_spec_kept: u64 = 0;
var expert_spec_total: u64 = 0;
fn noteExpertSpec(kept: bool) void {
    if (kept) expert_spec_kept += 1;
    expert_spec_total += 1;
    if (expert_spec_total == 1 or expert_spec_total % 20_000 == 0)
        log.info("[expert-spec] kept {d} of {d} speculated layers\n", .{ expert_spec_kept, expert_spec_total });
}

fn handOffStreamedResult(result: mlx.mlx_array, rows: c_int) !void {
    if (!streamHandoffApplies(rows)) return;
    const vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{result}, 1);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_async_eval(vec));
}

test "streamed expert work is handed to the GPU early only at decode widths" {
    try testing.expect(streamHandoffApplies(1));
    try testing.expect(streamHandoffApplies(16));
    try testing.expect(!streamHandoffApplies(17));
    try testing.expect(streamSharedEarlyApplies(true, &.{ 1, 1, 10 }));
    try testing.expect(!streamSharedEarlyApplies(true, &.{ 1, 512, 10 }));
    try testing.expect(!streamSharedEarlyApplies(false, &.{ 1, 1, 10 }));
}

test "expert swap reads softmax logits raw and a biased sigmoid router through log sigmoid, at decode widths without a capture" {
    try testing.expectEqual(@as(?SwapLogits, .softmax), expertSwapKind(0.3, false, false, false, &.{ 1, 4, 10 }));
    try testing.expectEqual(@as(?SwapLogits, .log_sigmoid), expertSwapKind(0.3, true, true, false, &.{ 1, 1, 8 }));
    try testing.expectEqual(@as(?SwapLogits, null), expertSwapKind(0.3, true, false, false, &.{ 1, 1, 10 }));
    try testing.expectEqual(@as(?SwapLogits, null), expertSwapKind(0, true, true, false, &.{ 1, 1, 8 }));
    try testing.expectEqual(@as(?SwapLogits, null), expertSwapKind(0.3, false, false, true, &.{ 1, 1, 10 }));
    try testing.expectEqual(@as(?SwapLogits, null), expertSwapKind(0.3, true, true, false, &.{ 1, 512, 8 }));
}

test "a sigmoid router's swap logits make the pick test its sigmoid probability ratio" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // (missed logit, cached candidate logit, tolerance): near saturation a raw logit gap would refuse a
    // substitute the sigmoid ratio accepts, and -120 must stay finite.
    const cases = [_][3]f32{ .{ 8, 3, 0.2 }, .{ 8, 3, 0.04 }, .{ -2, -2.3, 0.2 }, .{ -2, -2.1, 0.2 }, .{ 0, -0.2, 0.2 }, .{ 20, -5, 0.6 }, .{ -120, -120.1, 0.2 } };
    for (cases) |c| {
        const raw = [_]f32{ c[0], c[1], -50, -50 };
        const arr = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 1, 4 }, 3, .float32);
        defer _ = mlx.mlx_array_free(arr);
        var swap = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(swap);
        try routerSwapLogits(&swap, arr, .log_sigmoid, s);
        try mlx.check(mlx.mlx_array_eval(swap));
        const got = mlx.mlx_array_data_float32(swap).?;
        var ids = [_]u16{0};
        const cached = [_]bool{ false, true, false, false };
        var taken: [4]bool = undefined;
        var loading: [4]bool = undefined;
        var starved = [_]u8{ 0, 0, 0, 0 };
        _ = expert_stream_mod.substituteMisses(&ids, got[0..4], &cached, c[2], 1, &taken, &loading, &starved);
        const sig = struct {
            fn f(x: f64) f64 {
                return 1.0 / (1.0 + @exp(-x));
            }
        }.f;
        const ratio = if (c[0] < -100) @exp(@as(f64, c[1] - c[0])) else sig(c[1]) / sig(c[0]);
        try testing.expect(std.math.isFinite(got[0]) and std.math.isFinite(got[1]));
        try testing.expectEqual(@as(u16, if (ratio >= 1.0 - c[2]) 1 else 0), ids[0]);
    }
}

test "speculative expert compute runs at decode widths without an imatrix capture" {
    try testing.expect(expertSpecApplies(false, &.{ 1, 1, 10 }));
    try testing.expect(expertSpecApplies(false, &.{ 1, 16, 10 }));
    try testing.expect(!expertSpecApplies(false, &.{ 1, 17, 10 }));
    try testing.expect(!expertSpecApplies(true, &.{ 1, 1, 10 }));
}

pub const diagEnvOn = io_util_mod.diagEnvOn;
pub const diagEnvOnCached = io_util_mod.diagEnvOnCached;

/// Whether decode should try the in-place gather-qmv kernels at all.
///
/// The gate is "would the FUSED gate+up+SwiGLU kernel engage", because that is
/// what changed the answer. SPLIT gatherQmv loses to stock `gather_qmm` (a
/// 40-layer MoE, GPU eval 12.73 vs 12.47 ms) and lost on a 26B-A4B too, which is why
/// no arch opted in. Fusing the two projections and the activation into one
/// simdgroup flips it: 13.007/13.568 -> 12.768/13.114 ms/forward, both pairs.
///
/// So the eligibility predicate is deliberately NARROW — silu activation
/// (baked into the kernel), matching quant geometry on gate and up — rather
/// than a model_type list. An arch that does not meet it, a gelu MoE
/// included, keeps stock gather and cannot regress.
fn useGatherQmvDecode(self: *const Transformer, gate_qp: QuantParams, up_qp: QuantParams) bool {
    if (self.config.hidden_act != .silu) return false;
    if (!gatherQmvGateUpEnabled()) return false;
    return gate_qp.bits == up_qp.bits and
        gate_qp.group_size == up_qp.group_size and
        gate_qp.mode == up_qp.mode;
}

const QkAngleSlot = struct {
    offset: c_int = -1,
    rd: c_int = 0,
    rows: c_int = 0,
    arr: mlx.mlx_array = .{ .ctx = null },
};

// ── Fused decode QK-norm+RoPE (mlxfast 9e06de6 class) ──
//
// At decode the chain q/k reshape → per-head RMSNorm → transpose → RoPE
// (→ YaRN mscale on scaled layers) is strictly serial — SDPA needs its
// output and nothing overlaps it — and costs 4-6 dispatches per layer for a
// few KB of work. One 32-thread-per-head kernel does all of it. cos/sin are
// never re-derived: a per-token PROBE row (ones|zeros through the stock
// `mlx_fast_rope` at the token's offset) extracts the exact floats the stock
// kernel would use, cached per rope family for the forward (2 probe
// dispatches replace ~160 removed ones). Bit-identity with the composed
// chain is the acceptance bar (`qk norm rope fused` parity tests).

/// Multi-position variant: [rows * rd] f32, row p = the cos|sin row at
/// position `offset + p` — the probe is a [1,1,rows,rd] stack of ones|zeros
/// rows, so `mlx_fast_rope` stamps each row at its own position exactly as it
/// would the real tensor.
///
/// `mscale` multiplies the rows (YaRN's attention factor; 1.0 leaves the probe
/// exactly as the stock kernel computed it). Both halves of a rotary pair tile
/// the same frequency, so scaling the rows scales the rotation of the pair —
/// the same product the reference gets from `cos*af, sin*af`.
pub fn ropeAngleRows(
    s: mlx.mlx_stream,
    rd: c_int,
    base: mlx.mlx_optional_float,
    freqs: mlx.mlx_array,
    offset: c_int,
    rows: c_int,
    allocator: std.mem.Allocator,
    mscale: f32,
) !mlx.mlx_array {
    const n: usize = @intCast(rd);
    const nrows: usize = @intCast(rows);
    const buf = try allocator.alloc(f32, n * nrows);
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| v.* = if ((i % n) < n / 2) 1.0 else 0.0;
    const shape = [_]c_int{ 1, 1, rows, rd };
    const probe = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(probe);
    var rotated = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rotated);
    try mlx.check(mlx.mlx_fast_rope(&rotated, probe, rd, false, base, 1.0, offset, freqs, s));
    if (mscale == 1.0) {
        var flat = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(flat);
        const flat_shape = [_]c_int{rows * rd};
        try mlx.check(mlx.mlx_reshape(&flat, rotated, &flat_shape, 1, s));
        return flat;
    }
    // `defer`, not `errdefer`: the reshape below is a view, so the handle is ours to drop on the
    // success path too (an errdefer leaked one array per decode tick to the MTLSharedEvent cap).
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    const ms = mlx.mlx_array_new_float(mscale);
    defer _ = mlx.mlx_array_free(ms);
    try mlx.check(mlx.mlx_multiply(&scaled, rotated, ms, s));
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    const flat_shape = [_]c_int{rows * rd};
    try mlx.check(mlx.mlx_reshape(&flat, scaled, &flat_shape, 1, s));
    return flat;
}

// ── Fused QK-norm+RoPE, hd-256 (mlxfast-challenge qwen35_attention_qk_rms_rope port) ──
//
// The hd-128 kernel above never fires on qwen4_exp full-attention layers
// (hd 256), so their decode AND verify
// widths paid the composed per-head RMSNorm → transpose → partial-RoPE chain.
// This is the 256-wide sibling: 64 threads per (head, position) row, RMS via
// the same simdgroup-partials tree `rms_single_row` uses, normalized row
// staged bf16 in threadgroup memory (one rounding — the value the separate
// rms kernel would have written), then the RD-dim rotation reads the staged
// row against a PER-POSITION angle table (`ropeAngleRows` probe — the stock
// kernel's own floats, never re-derived; the challenge computes cos/sin
// in-kernel, which trusts JIT transcendentals our house rule forbids).
// Output is head-major [B,H,S,D], so the transposes vanish. Covers S 1..32
// (verify widths + one-chunk tiny-prompt prefills; challenge raised the same
// gate 16 -> 32 in 033f622).
// Bit-identity with the composed chain is the acceptance bar.

// ── Packed GDN prework mixer (mlxfast-challenge qwen35_packed_gdn_prework port) ──
//
// ONE launch replacing the GDN prework chain at decode and verify widths
// (S 1..9) — conv-state concat + depthwise conv1d + SiLU + q/k/v split +
// reshapes + Q/K rmsNorm-and-scale + conv-state slice + the forget gate
// (exp(-exp(A_log)·softplus(a+dt_bias))) + beta (sigmoid(b)) — ~14
// dispatches per GDN layer per step. Six outputs: normed/scaled Q and K,
// activated V, the next 3-row conv state, g and beta.
// The gate rides the SAME formulas MLX's compiled gate closure emits
// (precise exp, utils.h log1p, GT add rounding), so it is bit-identical
// to `computeGdnGate`; the in-kernel sigmoid (SiLU half, beta) is MLX's own
// unary formula (exp-of-abs form) — the challenge swept all 65,280 finite
// bf16 inputs against it: bit-exact everywhere.
// q/k/v, b and a are read through (offset, row stride) pairs (a combined
// projection could hand them over without slice copies).
// S < 3 rows of the next conv state come from the OLD state (row r reads
// concat(conv_state, qkv)[r + S]); before this generalization the kernel
// was gated to S >= 3 and decode ran the composed chain. Bit-identity with
// the composed chain is the acceptance bar. Test seams `gdn_prework_override` (whole kernel)
// and `gdn_decode_fused_override` (S < 3, in-kernel gate, norm-gate: the pre-fusion decode
// path) select the composed chain.
pub const GDN_KERNEL_HEADER =
    \\inline float sushi_log1p(float x) {
    \\    float xp1 = 1.0f + x;
    \\    if (xp1 == metal::numeric_limits<float>::max()) { return metal::numeric_limits<float>::max(); }
    \\    if (xp1 == 1.0f) { return x; }
    \\    return x * (metal::log(xp1) / (xp1 - 1.0f));
    \\}
;

// ── Fused QSA pooled-key upkeep ──

/// `[n, 1, vocab]` bf16 rows holding each shortlist's exact logits at its ids and -inf elsewhere, so
/// an argmax over a row picks the lowest id among the shortlist's maxima. That is the full row's
/// argmax whenever the full row's lowest-id maximum is in the shortlist; one outside is the miss the
/// audit counts. Also returns the `[n, 1, 32]` int32 ids.
fn shortlistSparseLogits(s: mlx.mlx_stream, lists: []const mtp_mod.Shortlist, vocab: c_int) !struct { logits: mlx.mlx_array, idx: mlx.mlx_array } {
    const n = lists.len;
    if (n == 0 or n > LMHEAD_SHORTLIST_MAX_ROWS) return error.ShortlistRowsOutOfRange;
    var idx_parts: [LMHEAD_SHORTLIST_MAX_ROWS]mlx.mlx_array = @splat(.{ .ctx = null });
    var val_parts: [LMHEAD_SHORTLIST_MAX_ROWS]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (idx_parts[0..n], val_parts[0..n]) |a, b| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a);
        if (b.ctx != null) _ = mlx.mlx_array_free(b);
    };
    for (lists, 0..) |sl, i| {
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        try mlx.check(mlx.mlx_astype(&ids, sl.cands, .int32, s));
        idx_parts[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&idx_parts[i], ids, &[_]c_int{ 1, 1, 32 }, 3, s));
        var v = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v);
        try mlx.check(mlx.mlx_astype(&v, sl.exact, .bfloat16, s));
        val_parts[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&val_parts[i], v, &[_]c_int{ 1, 1, 32 }, 3, s));
    }
    const iv = mlx.mlx_vector_array_new_data(&idx_parts, n);
    defer _ = mlx.mlx_vector_array_free(iv);
    const vv = mlx.mlx_vector_array_new_data(&val_parts, n);
    defer _ = mlx.mlx_vector_array_free(vv);
    var idx = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_concatenate_axis(&idx, iv, 0, s));
    var vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals);
    try mlx.check(mlx.mlx_concatenate_axis(&vals, vv, 0, s));
    const neg_inf = try scalarOf(-std.math.inf(f32), .bfloat16, s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var base = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(base);
    try mlx.check(mlx.mlx_full(&base, &[_]c_int{ @intCast(n), 1, vocab }, 3, neg_inf, .bfloat16, s));
    var filled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(filled);
    try mlx.check(mlx.mlx_put_along_axis(&filled, base, idx, vals, -1, s));
    return .{ .logits = filled, .idx = idx };
}

/// Rows an argmax-only shortlist readout serves: a decode tick and every verify width.
const LMHEAD_SHORTLIST_MAX_ROWS: usize = 16;
var lmhead_full_env: ?bool = null;
var lmhead_shortlist_audit_env: ?bool = null;

fn lmHeadShortlistOn() bool {
    return !diagEnvOnCached(&lmhead_full_env, "SUSHI_LMHEAD_FULL");
}

/// DIAGNOSTIC (SUSHI_LMHEAD_SHORTLIST_AUDIT=1): every shortlist row also reads the full head and
/// counts rows whose full argmax is outside the shortlist or differs from the served one.
const ShortlistAudit = struct {
    rows: u64 = 0,
    outside: u64 = 0,
    differ: u64 = 0,
    /// Rows where a shortlist logit is not bit-equal to the full head's at that id.
    inexact: u64 = 0,
};
var lmhead_shortlist_audit: ShortlistAudit = .{};

fn shortlistAudit(s: mlx.mlx_stream, full: mlx.mlx_array, sparse: mlx.mlx_array, idx: mlx.mlx_array, n: usize, vocab: c_int) !void {
    var fa = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fa);
    try mlx.check(mlx.mlx_argmax_axis(&fa, full, -1, false, s));
    var sa = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sa);
    try mlx.check(mlx.mlx_argmax_axis(&sa, sparse, -1, false, s));
    var fa32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fa32);
    try mlx.check(mlx.mlx_astype(&fa32, fa, .int32, s));
    var sa32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sa32);
    try mlx.check(mlx.mlx_astype(&sa32, sa, .int32, s));
    var f32full = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32full);
    try mlx.check(mlx.mlx_astype(&f32full, full, .float32, s));
    var f32sparse = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32sparse);
    try mlx.check(mlx.mlx_astype(&f32sparse, sparse, .float32, s));
    try mlx.check(mlx.mlx_array_eval(fa32));
    try mlx.check(mlx.mlx_array_eval(sa32));
    try mlx.check(mlx.mlx_array_eval(idx));
    try mlx.check(mlx.mlx_array_eval(f32full));
    try mlx.check(mlx.mlx_array_eval(f32sparse));
    const fp = mlx.mlx_array_data_int32(fa32) orelse return;
    const sp = mlx.mlx_array_data_int32(sa32) orelse return;
    const ip = mlx.mlx_array_data_int32(idx) orelse return;
    const lp = mlx.mlx_array_data_float32(f32full) orelse return;
    const vp = mlx.mlx_array_data_float32(f32sparse) orelse return;
    const v: usize = @intCast(vocab);
    for (0..n) |r| {
        lmhead_shortlist_audit.rows += 1;
        var inside = false;
        var exact = true;
        for (ip[r * 32 ..][0..32]) |c| {
            inside = inside or c == fp[r];
            const at = r * v + @as(usize, @intCast(c));
            exact = exact and vp[at] == lp[at];
        }
        if (!inside) lmhead_shortlist_audit.outside += 1;
        if (!exact) lmhead_shortlist_audit.inexact += 1;
        if (fp[r] != sp[r]) {
            lmhead_shortlist_audit.differ += 1;
            const row = lp[r * v ..][0..v];
            var second: f32 = -std.math.inf(f32);
            for (row, 0..) |val, j| if (j != @as(usize, @intCast(fp[r])) and val > second) {
                second = val;
            };
            log.info("[lm_head-audit] argmax differs: full {d} ({d:.4}) served {d}, full top-2 gap {d:.5}, in shortlist {}\n", .{ fp[r], row[@intCast(fp[r])], sp[r], row[@intCast(fp[r])] - second, inside });
        }
    }
    if (lmhead_shortlist_audit.rows % 256 < n) log.info("[lm_head-audit] rows {d}, full argmax outside the shortlist {d}, served argmax differs {d}, shortlist logits not bit-equal {d}\n", .{ lmhead_shortlist_audit.rows, lmhead_shortlist_audit.outside, lmhead_shortlist_audit.differ, lmhead_shortlist_audit.inexact });
}
// ── Fused residual add + RMSNorm ──
//
// `h = h + branch; normed = rms_norm(h, w)` is two dispatches, and unlike the
// attention gate they are STRICTLY serial with everything around them: the norm
// waits on the add, the projections wait on the norm. Every transformer layer
// in the shared MoE loop pays it.
//
// The kernel is `rms_single_row` (rms_norm.metal) with the add folded into its
// load, and it keeps MLX's launch geometry — ceil(AXIS/4) threads, four
// CONSECUTIVE elements each, `simd_sum` per simdgroup, a zero-initialised
// `local_sums` plane, then one `simd_sum` over the partials — so the reduction
// tree is the same tree on the same values, not merely the same total. The sum
// is rounded to T before it is squared, exactly where `mlx_add` writes it, and
// it is emitted as a second output because the residual is still needed.
// `metal::precise::rsqrt` is deliberately the precise variant, as upstream.
//
// MEASURED NEUTRAL-TO-NEGATIVE on the shared MoE loop, so only MiMo's
// fused-norm path wires it. Six pairs, both orders, on a
// machine under other GPU load so only the RATIO is meaningful: ON median
// 14.890 vs OFF 14.747 ms/forward, 4 of 6 pairs favouring OFF.
//
// The reason is instructive and is NOT the critical-path argument that sank the
// attention gate — this pair really is serial. It is that the fusion barely
// removes any WORK: add is 2 reads + 1 write of a 4 KB row, rms_norm is 1 read
// + 1 write, and the fused kernel still does 2 reads + 2 writes because the
// residual sum is needed downstream and has to be emitted. One 4 KB read saved,
// against giving up MLX's metallib-compiled `rms_single_row` for a JIT copy.
// So the round's rule has a second half: a fusion pays when it removes real
// work or a real serialization (the router's 256-wide sort, the SwiGLU's 3-op
// chain, two expert GEMV passes), not when it merely removes a launch around
// an op MLX already does well.

// ── Fused SwiGLU: silu(gate) * up in one dispatch ──
//
// `computeGeglu` only had a fused path for `gelu_approx`, so every SILU model ran sigmoid +
// multiply + multiply as three dispatches, and a MoE layer pays it TWICE
// (routed experts + shared expert). It sits on the expert chain's critical
// path, so the launches are not overlapped: paired
// A/B on a 40-layer MoE measured 13.12 -> 12.81 ms/forward, 3/3 pairs.
//
// Extending mlx_compile to the silu arm gets the same 2.4% but is NOT
// output-preserving — a same-binary greedy A/B diverged at token ~55, so the
// compiled elementwise fusion does not reproduce the chain's intermediate
// rounding. This kernel does: MLX's `Sigmoid` (the two-sided form) evaluated
// and ROUNDED TO T, then two `Multiply`s each promoting to float and rounding
// back, exactly as the three op kernels do.
// The sigmoid is a TABLE LOOKUP, not a recomputation, and that is the whole
// trick. Writing MLX's `Sigmoid` source verbatim is not enough to reproduce it:
// MLX's own kernels live in a metallib built with a different Metal math mode
// than the one `metal_kernel` JITs custom sources with (CompileOptions
// defaults to MathMode::Safe and mlx-c exposes no way to ask for another), so
// `1 / (1 + exp(|x|))` lands a rounding apart. An exhaustive sweep found it on
// exactly ONE of the 65536 bf16 patterns (-6.84375) — which random sampling
// missed and a 260-token greedy run hit within ~55 tokens, because a
// 8192-wide MLP draws 8192 values from that domain every layer.
//
// A 16-bit activation has only 65536 possible inputs, so `mlx_sigmoid` is
// evaluated over ALL of them once at first use and the kernel indexes the
// result by the input's bit pattern. Exact by construction, no transcendental
// in the hot loop, 128 KB resident. The two multiplies need no such care: a
// bf16/f16 product is exact in float, so one rounding is one rounding on any
// compiler.
const SWIGLU_SOURCE =
    \\uint i = thread_position_in_grid.x;
    \\if (i >= uint(N_size)) return;
    \\T g = gate[i];
    \\T sig = sigtab[as_type<ushort>(g)];
    \\// mlx Multiply (binary_ops.h), twice: silu(g) = g * sigmoid(g), then * up.
    \\T act = g * sig;
    \\y[i] = act * up[i];
;

/// `mlx_sigmoid` tabulated over every 16-bit float pattern, indexed [bf16, f16].
var swiglu_sig_tables: [2]?mlx.mlx_array = @splat(null);

pub fn swigluSigTable(s: mlx.mlx_stream, dt: mlx.mlx_dtype, allocator: std.mem.Allocator) !mlx.mlx_array {
    const slot: usize = switch (dt) {
        .bfloat16 => 0,
        .float16 => 1,
        else => return error.UnsupportedSwigluDtype,
    };
    if (swiglu_sig_tables[slot]) |t| return t;

    const N: usize = 1 << 16;
    const buf = try allocator.alloc(f32, N);
    defer allocator.free(buf);
    // Every 16-bit pattern, widened to the f32 that converts back to it
    // EXACTLY, so the astype below is a relabel and not a rounding.
    for (buf, 0..) |*v, i| v.* = switch (dt) {
        .bfloat16 => @bitCast(@as(u32, @intCast(i)) << 16),
        else => @floatCast(@as(f16, @bitCast(@as(u16, @intCast(i))))),
    };
    const sh = [_]c_int{@intCast(N)};
    const v32 = mlx.mlx_array_new_data(buf.ptr, &sh, 1, .float32);
    defer _ = mlx.mlx_array_free(v32);
    var vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals);
    try mlx.check(mlx.mlx_astype(&vals, v32, dt, s));
    var table = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(table);
    try mlx.check(mlx.mlx_sigmoid(&table, vals, s));
    // Materialize now: the table outlives every graph that reads it.
    try mlx.check(mlx.mlx_array_eval(table));
    swiglu_sig_tables[slot] = table;
    return table;
}

var swiglu_kernel: ?mlx.mlx_fast_metal_kernel = null;
var swiglu_engaged: bool = false;
pub var swiglu_fused_override: ?bool = null;

fn swigluFusedEnabled() bool {
    return swiglu_fused_override orelse true;
}

fn getSwigluKernel() !mlx.mlx_fast_metal_kernel {
    if (swiglu_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "gate", "up", "sigtab", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_swiglu",
        in_vec,
        out_vec,
        SWIGLU_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    swiglu_kernel = kernel;
    return kernel;
}

const SwigluCfgKey = struct { shape: ShapeKey, dtype: mlx.mlx_dtype };
var swiglu_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var swiglu_cfg_key: SwigluCfgKey = std.mem.zeroes(SwigluCfgKey);

/// `silu(gate) * up`, elementwise, one dispatch. Null → caller keeps the chain.
/// Exact by construction (bit-level sigmoid table), so callers outside this
/// file (tts.zig's talker/code-predictor MLPs) reuse it rather than growing a
/// second kernel.
pub fn fusedSwiGLU(s: mlx.mlx_stream, gate: mlx.mlx_array, up: mlx.mlx_array) !?mlx.mlx_array {
    if (!swigluFusedEnabled()) return null;
    const dt = mlx.mlx_array_dtype(gate);
    if (dt != mlx.mlx_array_dtype(up)) return null;
    // FLOAT32 IS DECLINED ON PURPOSE. MLX ships its own kernels in a metallib
    // built with a different Metal math mode than the one `metal_kernel` JITs
    // custom sources with (CompileOptions defaults to MathMode::Safe, and
    // mlx-c exposes no way to ask for another), so an f32 `1 / (1 + exp(..))`
    // lands 1-2 ulps away from the stock sigmoid's. That is invisible in bf16
    // and f16 — both round the result far below the difference, and the test
    // pins them — but it is real in f32, and a fused path that is exact for
    // some dtypes and not others is worse than one that declines.
    if (dt != .bfloat16 and dt != .float16) return null;
    const gsh = mlx.getShape(gate);
    const ush = mlx.getShape(up);
    // No broadcasting: the gated MLP always pairs equal shapes, and a mismatch
    // here would silently read out of bounds rather than fail.
    if (gsh.len != ush.len or gsh.len == 0 or gsh.len > 5) return null;
    var n: i64 = 1;
    for (gsh, ush) |a, b| {
        if (a != b) return null;
        n *= a;
    }
    if (n <= 0 or n > std.math.maxInt(c_int)) return null;
    const ni: c_int = @intCast(n);

    const tg: c_int = @min(@as(c_int, 256), @max(@as(c_int, 32), @divTrunc(ni + 31, 32) * 32));
    const key = SwigluCfgKey{ .shape = ShapeKey.from(gsh), .dtype = dt };
    if (swiglu_cfg == null or !std.meta.eql(swiglu_cfg_key, key)) {
        if (swiglu_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, gsh.ptr, gsh.len, dt));
        // Grid is rounded UP to the threadgroup so the launch is uniform; the
        // kernel's own bound check drops the tail threads.
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(ni + tg - 1, tg) * tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        swiglu_cfg = config;
        swiglu_cfg_key = key;
    }

    const sigtab = try swigluSigTable(s, dt, std.heap.c_allocator);
    const inputs_arr = [_]mlx.mlx_array{ gate, up, sigtab, cachedScalarInt(ni) };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getSwigluKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, swiglu_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!swiglu_engaged) {
        swiglu_engaged = true;
        log.info("[mlp] fused SwiGLU kernel engaged: n={d} tg={d}\n", .{ ni, tg });
    }
    return y;
}

/// Extra per-MoE-layer elementwise dispatches injected by the sizing probe
/// (`SUSHI_DISPATCH_PROBE`). 0 = off, which is every non-diagnostic run.
var dispatch_probe_cached: ?usize = null;
pub fn dispatchProbeCount() usize {
    if (dispatch_probe_cached) |v| return v;
    const raw = std.c.getenv("SUSHI_DISPATCH_PROBE");
    const n: usize = if (raw) |r| std.fmt.parseInt(usize, std.mem.sliceTo(r, 0), 10) catch 0 else 0;
    dispatch_probe_cached = n;
    return n;
}

pub fn decodeProfReport() void {
    const n = decode_prof.calls;
    if (n == 0) return;
    const per = struct {
        fn ms(ns: u64, calls: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(calls)) / 1.0e6;
        }
    };
    const e = per.ms(decode_prof.embed_ns, n);
    const a = per.ms(decode_prof.attn_ns, n);
    const m = per.ms(decode_prof.mlp_ns, n);
    const l = per.ms(decode_prof.lmhead_ns, n);
    const mr = per.ms(decode_prof.moe_router_ns, n);
    const mx_ = per.ms(decode_prof.moe_experts_ns, n);
    const ms_ = per.ms(decode_prof.moe_shared_ns, n);
    log.info("[decode-prof] n={d} serial/tok={d:.2}ms  embed={d:.2} attn={d:.2} mlp={d:.2} lmhead={d:.2} ms | moe: router={d:.2} experts={d:.2} shared={d:.2} ms\n", .{ n, e + a + m + l, e, a, m, l, mr, mx_, ms_ });
    decode_prof = .{};
}

// ── MoE decode gather-qmv kernel ──
// MLX's `gather_qmm` at decode width costs O(EXPERT BANK SIZE), not O(work), in
// any C/C++ process (it is flat under pip-Python with the identical dylib —
// measured E-sweep in the `MoE decode gather` µbench: 30/50/77/300 us at banks
// of 8/25/50/201 MB, all reading the SAME 10 experts). 141 gathers/token x 300 us
// is the whole 42 ms/token decode. `take` from the same 201 MB bank does
// NOT pay it, so indexed reads are fine; the penalty is specific to gather_qmm's
// addressing.
//
// The shipped workaround (`batchedExpertMm`: take + broadcast + batched qmm)
// dodges it but pays a materialization round-trip — read 9.8 MB of experts,
// write 9.8 MB, read 9.8 MB back = 3x the ideal traffic, which is exactly its
// measured 70 us against a 67 us bandwidth prediction. It cannot go faster.
//
// This kernel reads the bank IN PLACE with GPU-resident indices: one simdgroup
// per (expert slot, output row), each lane striding the packed row and reducing
// with simd_sum. Ideal traffic, no copy, no host sync for the top-K indices.
//
// `X_PER_EXPERT` picks the input layout: 0 = one shared x [K] (gate/up
// projections, every expert sees the same token), 1 = x [TOPK, K] (the down
// projection, where each expert consumes its own activation).
fn gatherQmvSource(comptime x_per_expert: bool) [:0]const u8 {
    // Everything is indexed off the raw buffer names rather than through
    // explicitly address-spaced pointers: MLX generates the signature from each
    // input's ACTUAL dtype and puts arrays with fewer than 8 elements in
    // `constant` instead of `device` (backend/common/metal_kernel.cpp), so a
    // hand-declared `const device T*` would mismatch on both counts. `x` in
    // particular can arrive as float32 while scales/biases are bf16.
    //
    // A uint4 (128-bit) load variant was measured and is NOT kept: 38.9 us vs
    // 37.3 for the scalar form. At ~5 ALU ops per dequantized value this loop
    // is ALU-bound, not load-width bound, so widening the loads only adds code.
    return std.fmt.comptimePrint(
        \\auto lane = thread_index_in_simdgroup;
        \\uint n = thread_position_in_grid.y;      // output row within the expert
        \\uint e = thread_position_in_grid.z;      // top-K slot
        \\
        \\int K = int(K_size);
        \\int N = int(N_size);
        \\constexpr int VPW = (BITS == 3) ? 8 : 32 / BITS;  // values per pack: a word, or 3 bytes at 3-bit
        \\int K_by_p = K / VPW;                    // packs per row
        \\int K_by_w = K * BITS / 32;              // words per row
        \\int K_by_gs = K / GS;                    // quant groups per row
        \\uint mask = (1u << BITS) - 1u;
        \\
        \\uint eid = inds[e];                      // bank row for this slot
        \\size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_w + (size_t)n * (size_t)K_by_w;
        \\size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
        \\size_t xoff = {s};
        \\
        \\// Four independent accumulators so the per-lane FMA chain is not one
        \\// serial dependency; VPW is 16/8/4 for bits 2/4/8, always a multiple of 4.
        \\float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        \\for (int pack = int(lane); pack < K_by_p; pack += 32) {{
        \\  uint32_t packed = sushi_qpack<BITS>(w_q, wbase, pack);
        \\  int k_base = pack * VPW;
        \\  int gi = k_base / GS;                  // GS >= VPW, so one group per word
        \\{s}
        \\}}
        \\float acc = simd_sum((a0 + a1) + (a2 + a3));
        \\if (lane == 0) {{
        \\  y[(size_t)e * (size_t)N + (size_t)n] = T(acc);
        \\}}
    , .{
        if (x_per_expert) "(size_t)e * (size_t)K" else "0",
        GQMV_BODY_AFFINE,
    });
}

const GQMV_BODY_AFFINE =
    \\  float sj = float(scales[gbase + (size_t)gi]);
    \\  float bj = float(biases[gbase + (size_t)gi]);
    \\  for (int ki = 0; ki < VPW; ki += 4) {
    \\    size_t xi = xoff + (size_t)(k_base + ki);
    \\    uint32_t q = packed >> (ki * BITS);
    \\    a0 += float(x[xi + 0]) * (float((q >> (0 * BITS)) & mask) * sj + bj);
    \\    a1 += float(x[xi + 1]) * (float((q >> (1 * BITS)) & mask) * sj + bj);
    \\    a2 += float(x[xi + 2]) * (float((q >> (2 * BITS)) & mask) * sj + bj);
    \\    a3 += float(x[xi + 3]) * (float((q >> (3 * BITS)) & mask) * sj + bj);
    \\  }
;

// One packed unit of quantized values as a uint32: a word for 2/4/8-bit; at
// 3-bit, mx.quantize packs 8 values into 3 bytes (value i at bit 3i), so the
// unit is a byte triple and the body's `>> (i * BITS)` walk is unchanged.
const GQMV_AFFINE_HEADER =
    \\template <int B>
    \\inline uint32_t sushi_qpack(const device uint32_t* w, size_t wbase_words, int pack) {
    \\  if (B == 3) {
    \\    const device uchar* wb = (const device uchar*)w + wbase_words * 4 + (size_t)pack * 3;
    \\    return uint32_t(wb[0]) | (uint32_t(wb[1]) << 8) | (uint32_t(wb[2]) << 16);
    \\  }
    \\  return w[wbase_words + (size_t)pack];
    \\}
;

// ── Fused routed gate+up+SwiGLU (one kernel for what was three) ──
//
// The decode expert path ran gate gather -> up gather -> activation as three
// dispatches, all on the critical chain into down_proj. This computes BOTH dot
// products in the SAME simdgroup and applies the activation before it writes,
// so the x element loaded for the gate FMA is reused for the up FMA and the two
// dequant chains interleave instead of running as separate launches.
//
// It halves the simdgroup count rather than the work (each does two rows'
// worth), which is the right trade here: 4096 simdgroups over 40 cores was ~100
// per core, so there is occupancy to spend on instruction-level parallelism.
//
// Bit-exactness comes from rounding each accumulator to T FIRST — exactly where
// `gatherQmv` writes `T(acc)` — and then running the same table-lookup SwiGLU
// `fusedSwiGLU` does. No f32 shortcut through the activation.
fn gatherQmvGateUpSource() [:0]const u8 {
    return std.fmt.comptimePrint(
        \\auto lane = thread_index_in_simdgroup;
        \\uint n = thread_position_in_grid.y;      // output row within the expert
        \\uint e = thread_position_in_grid.z;      // top-K slot
        \\
        \\int K = int(K_size);
        \\int N = int(N_size);
        \\constexpr int VPW = (BITS == 3) ? 8 : 32 / BITS;
        \\int K_by_p = K / VPW;
        \\int K_by_w = K * BITS / 32;
        \\int K_by_gs = K / GS;
        \\uint mask = (1u << BITS) - 1u;
        \\
        \\uint eid = inds[e];
        \\size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_w + (size_t)n * (size_t)K_by_w;
        \\size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
        \\size_t xrow = 0;
        \\
        \\// Every lane's packed words are issued up front (KP = K_by_p is a
        \\// template constant): −0.7% per forward in idle same-boot pairs vs
        \\// the runtime-count loop. Same per-lane accumulation order.
        \\constexpr int IT = (KP + 31) / 32;
        \\uint32_t pg[IT];
        \\uint32_t pu[IT];
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(lane) + 32 * i;
        \\  pg[i] = (pack < KP) ? sushi_qpack<BITS>(wg_q, wbase, pack) : 0u;
        \\  pu[i] = (pack < KP) ? sushi_qpack<BITS>(wu_q, wbase, pack) : 0u;
        \\}}
        \\// Two independent 4-way accumulator sets: the gate chain and the up
        \\// chain never wait on each other.
        \\float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
        \\float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(lane) + 32 * i;
        \\  if (pack >= KP) break;
        \\  uint32_t packed_g = pg[i];
        \\  uint32_t packed_u = pu[i];
        \\  int k_base = pack * VPW;
        \\  int gi = k_base / GS;
        \\{s}
        \\}}
        \\float acc_g = simd_sum((g0 + g1) + (g2 + g3));
        \\float acc_u = simd_sum((u0 + u1) + (u2 + u3));
        \\if (lane == 0) {{
        \\  // Round exactly where the unfused path's two kernels wrote T(acc),
        \\  // then the same table-lookup SwiGLU.
        \\  T gt = T(acc_g);
        \\  T ut = T(acc_u);
        \\  T sig = sigtab[as_type<ushort>(gt)];
        \\  y[(size_t)e * (size_t)N + (size_t)n] = (gt * sig) * ut;
        \\}}
    , .{
        GQMV_GATEUP_BODY_AFFINE,
    });
}

const GQMV_GATEUP_BODY_AFFINE =
    \\  float sjg = float(g_scales[gbase + (size_t)gi]);
    \\  float bjg = float(g_biases[gbase + (size_t)gi]);
    \\  float sju = float(u_scales[gbase + (size_t)gi]);
    \\  float bju = float(u_biases[gbase + (size_t)gi]);
    \\  for (int ki = 0; ki < VPW; ki += 4) {
    \\    size_t xi = xrow + (size_t)(k_base + ki);
    \\    uint32_t qg = packed_g >> (ki * BITS);
    \\    uint32_t qu = packed_u >> (ki * BITS);
    \\    float x0 = float(x[xi + 0]);
    \\    float x1 = float(x[xi + 1]);
    \\    float x2 = float(x[xi + 2]);
    \\    float x3 = float(x[xi + 3]);
    \\    g0 += x0 * (float((qg >> (0 * BITS)) & mask) * sjg + bjg);
    \\    g1 += x1 * (float((qg >> (1 * BITS)) & mask) * sjg + bjg);
    \\    g2 += x2 * (float((qg >> (2 * BITS)) & mask) * sjg + bjg);
    \\    g3 += x3 * (float((qg >> (3 * BITS)) & mask) * sjg + bjg);
    \\    u0 += x0 * (float((qu >> (0 * BITS)) & mask) * sju + bju);
    \\    u1 += x1 * (float((qu >> (1 * BITS)) & mask) * sju + bju);
    \\    u2 += x2 * (float((qu >> (2 * BITS)) & mask) * sju + bju);
    \\    u3 += x3 * (float((qu >> (3 * BITS)) & mask) * sju + bju);
    \\  }
;

const GQMV_GATEUP_SOURCE = gatherQmvGateUpSource();
var gqmv_gateup_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getGatherQmvGateUpKernel() !mlx.mlx_fast_metal_kernel {
    if (gqmv_gateup_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases", "inds", "sigtab", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_moe_gather_qmv_gateup",
        in_vec,
        out_vec,
        GQMV_GATEUP_SOURCE,
        GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_gateup_kernel = kernel;
    return kernel;
}














/// One quant group per pack: a word for 2/4/8-bit, a byte triple (8 values) for 3-bit.
fn affineQmvUnitOk(bits: u32, group_size: u32) bool {
    return switch (bits) {
        2, 4, 8 => group_size % (32 / bits) == 0,
        3 => group_size % 8 == 0,
        else => false,
    };
}

fn affineQmvPackUnits(K: c_int, bits: u32) c_int {
    return if (bits == 3) @divExact(K, 8) else @divExact(K * @as(c_int, @intCast(bits)), 32);
}

const GateUpCfgKey = struct { topk: c_int, n: c_int, k: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype };
var gateup_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gateup_cfg_key: GateUpCfgKey = std.mem.zeroes(GateUpCfgKey);
var gqmv_gateup_engaged: bool = false;
pub var gqmv_gateup_override: ?bool = null;

fn gatherQmvGateUpEnabled() bool {
    return gqmv_gateup_override orelse true;
}

/// One kernel for `silu(gather(gate) @ x) * (gather(up) @ x)` at decode width.
/// Returns [TOPK, N] in x's dtype, or null when anything is outside the
/// supported set (caller falls back to the two separate gathers + activation).
pub fn gatherQmvGateUp(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gw: mlx.mlx_array,
    gs_arr: mlx.mlx_array,
    gb: mlx.mlx_array,
    uw: mlx.mlx_array,
    us_arr: mlx.mlx_array,
    ub: mlx.mlx_array,
    inds: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
) !?mlx.mlx_array {
    if (!gatherQmvGateUpEnabled()) return null;
    if (mode != .affine) return null;
    if (!affineQmvUnitOk(bits, group_size)) return null;
    if (gb.ctx == null or ub.ctx == null) return null;
    if (gs_arr.ctx == null or us_arr.ctx == null) return null;
    // The activation is a 16-bit table lookup, exactly as `fusedSwiGLU` gates
    // it; f32 activations keep the unfused path so the two can never disagree.
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    // gate and up must be the SAME geometry — the kernel walks one index set
    // through both banks.
    const wsh = mlx.getShape(gw);
    const ush = mlx.getShape(uw);
    if (wsh.len != 3 or ush.len != 3) return null;
    for (wsh, ush) |a, b| if (a != b) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 1) return null;
    const topk: c_int = ish[0];
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |d| xelems *= d;
    if (xelems != K) return null;
    const sgs_per_tg: c_int = 8;
    if (@rem(N, sgs_per_tg) != 0) return null;

    const key = GateUpCfgKey{ .topk = topk, .n = N, .k = K, .bits = bits, .gs = group_size, .dtype = xd };
    if (gateup_cfg == null or !std.meta.eql(gateup_cfg_key, key)) {
        if (gateup_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ topk, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, N, topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, sgs_per_tg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KP", affineQmvPackUnits(K, bits)));
        gateup_cfg = config;
        gateup_cfg_key = key;
    }

    const sigtab = try swigluSigTable(s, xd, std.heap.c_allocator);
    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const inputs_arr = [_]mlx.mlx_array{ x, gw, gs_arr, gb, uw, us_arr, ub, inds, sigtab, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvGateUpKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, gateup_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!gqmv_gateup_engaged) {
        gqmv_gateup_engaged = true;
        log.info("[moe] fused gate+up+SwiGLU kernel engaged: E_topk={d} inter={d} bits={d} gs={d}\n", .{ topk, N, bits, group_size });
    }
    return y;
}

fn gatherQmvGateUpRowsSource() [:0]const u8 {
    return std.fmt.comptimePrint(
        \\auto lane = thread_index_in_simdgroup;
        \\uint n = thread_position_in_grid.y;
        \\uint combo = thread_position_in_grid.z;
        \\uint e = combo % uint(TOPK);
        \\uint row = combo / uint(TOPK);
        \\
        \\int K = int(K_size);
        \\int N = int(N_size);
        \\constexpr int VPW = (BITS == 3) ? 8 : 32 / BITS;
        \\int K_by_p = K / VPW;
        \\int K_by_w = K * BITS / 32;
        \\int K_by_gs = K / GS;
        \\uint mask = (1u << BITS) - 1u;
        \\
        \\uint eid = inds[(size_t)row * (size_t)TOPK + e];
        \\size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_w + (size_t)n * (size_t)K_by_w;
        \\size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
        \\size_t xrow = (size_t)row * (size_t)K;
        \\
        \\constexpr int IT = (KP + 31) / 32;
        \\uint32_t pg[IT];
        \\uint32_t pu[IT];
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(lane) + 32 * i;
        \\  pg[i] = (pack < KP) ? sushi_qpack<BITS>(wg_q, wbase, pack) : 0u;
        \\  pu[i] = (pack < KP) ? sushi_qpack<BITS>(wu_q, wbase, pack) : 0u;
        \\}}
        \\float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
        \\float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(lane) + 32 * i;
        \\  if (pack >= KP) break;
        \\  uint32_t packed_g = pg[i];
        \\  uint32_t packed_u = pu[i];
        \\  int k_base = pack * VPW;
        \\  int gi = k_base / GS;
        \\{s}
        \\}}
        \\float acc_g = simd_sum((g0 + g1) + (g2 + g3));
        \\float acc_u = simd_sum((u0 + u1) + (u2 + u3));
        \\if (lane == 0) {{
        \\  T gt = T(acc_g);
        \\  T ut = T(acc_u);
        \\  T sig = sigtab[as_type<ushort>(gt)];
        \\  y[(((size_t)row * (size_t)TOPK) + e) * (size_t)N + (size_t)n] = (gt * sig) * ut;
        \\}}
    , .{
        GQMV_GATEUP_BODY_AFFINE,
    });
}

const GQMV_GATEUP_ROWS_SOURCE = gatherQmvGateUpRowsSource();
var gqmv_gateup_rows_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getGatherQmvGateUpRowsKernel() !mlx.mlx_fast_metal_kernel {
    if (gqmv_gateup_rows_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases", "inds", "sigtab", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_moe_gather_qmv_gateup_rows",
        in_vec,
        out_vec,
        GQMV_GATEUP_ROWS_SOURCE,
        GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_gateup_rows_kernel = kernel;
    return kernel;
}

const GateUpRowsCfgKey = struct { nrows: c_int, topk: c_int, n: c_int, k: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype };
var gateup_rows_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gateup_rows_cfg_key: GateUpRowsCfgKey = std.mem.zeroes(GateUpRowsCfgKey);
var gqmv_gateup_rows_engaged: bool = false;
pub var moe_rows_fused_override: ?bool = null;
pub var moe_rows_fused_layers: u32 = 0;
pub var moe_sorted_layers: u32 = 0;

fn moeRowsFusedEnabled() bool {
    return moe_rows_fused_override orelse true;
}

pub const MoeDecodeDispatchArm = enum { rows, sorted, gather_qmv };

pub fn moeDecodeDispatchArm(B: c_int, S: c_int, K: c_int) MoeDecodeDispatchArm {
    const total_inds: c_int = B * S * K;
    const do_sort = B * S > 1 or total_inds >= 64;
    if (B >= 1 and B <= 8 and S >= 1 and B * S >= 2 and B * S <= 16 and moeRowsFusedEnabled() and
        !qwen4Standin().moe_gateup and !qwen4Standin().moe_down)
        return .rows;
    if (do_sort) return .sorted;
    return .gather_qmv;
}

/// Row-axis twin of `gatherQmvGateUp` for the batched decode tick (S = 1,
/// 2 <= B <= 8): the grid runs over output tiles x (row, expert choice) and
/// each row is the single-row arithmetic unchanged, so the result is
/// bit-identical per row to B solo calls. Shares the single-row Metal body.
pub fn gatherQmvGateUpRows(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gw: mlx.mlx_array,
    gs_arr: mlx.mlx_array,
    gb: mlx.mlx_array,
    uw: mlx.mlx_array,
    us_arr: mlx.mlx_array,
    ub: mlx.mlx_array,
    inds: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
) !?mlx.mlx_array {
    if (!gatherQmvGateUpEnabled()) return null;
    if (!moeRowsFusedEnabled()) return null;
    if (mode != .affine) return null;
    if (!affineQmvUnitOk(bits, group_size)) return null;
    if (gb.ctx == null or ub.ctx == null) return null;
    if (gs_arr.ctx == null or us_arr.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    const wsh = mlx.getShape(gw);
    const ush = mlx.getShape(uw);
    if (wsh.len != 3 or ush.len != 3) return null;
    for (wsh, ush) |a, b| if (a != b) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 2) return null;
    const nrows: c_int = ish[0];
    const topk: c_int = ish[1];
    if (nrows < 1 or nrows > 128 or topk < 1 or topk > 16) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len != 2 or xsh[0] != nrows or xsh[1] != K) return null;
    const sgs_per_tg: c_int = 8;
    if (@rem(N, sgs_per_tg) != 0) return null;

    const key = GateUpRowsCfgKey{ .nrows = nrows, .topk = topk, .n = N, .k = K, .bits = bits, .gs = group_size, .dtype = xd };
    if (gateup_rows_cfg == null or !std.meta.eql(gateup_rows_cfg_key, key)) {
        if (gateup_rows_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ nrows, topk, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 3, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, N, nrows * topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, sgs_per_tg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KP", affineQmvPackUnits(K, bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TOPK", topk));
        gateup_rows_cfg = config;
        gateup_rows_cfg_key = key;
    }

    const sigtab = try swigluSigTable(s, xd, std.heap.c_allocator);
    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const inputs_arr = [_]mlx.mlx_array{ x, gw, gs_arr, gb, uw, us_arr, ub, inds, sigtab, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvGateUpRowsKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, gateup_rows_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!gqmv_gateup_rows_engaged) {
        gqmv_gateup_rows_engaged = true;
        log.info("[moe] fused rows gate+up+SwiGLU kernel engaged: B={d} E_topk={d} inter={d} bits={d} gs={d}\n", .{ nrows, topk, N, bits, group_size });
    }
    return y;
}

const GQMV_SOURCES = [2][:0]const u8{ gatherQmvSource(false), gatherQmvSource(true) };
const GQMV_NAMES = [2][*:0]const u8{ "sushi_moe_gather_qmv", "sushi_moe_gather_qmv_px" };
var gqmv_kernels: [2]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getGatherQmvKernel(x_per_expert: bool) !mlx.mlx_fast_metal_kernel {
    const xi: usize = @intFromBool(x_per_expert);
    if (gqmv_kernels[xi]) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "inds", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        GQMV_NAMES[xi],
        in_vec,
        out_vec,
        GQMV_SOURCES[xi],
        GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_kernels[xi] = kernel;
    return kernel;
}

/// Everything a `gatherQmv` config bakes in; one slot per input layout, since
/// the gate/up (shared-x) and down (per-expert-x) calls alternate every layer.
const GqmvCfgKey = struct {
    topk: c_int,
    n: c_int,
    bits: u32,
    gs: u32,
    dtype: mlx.mlx_dtype,
    x_per_expert: bool,
};
const GqmvCfgSlot = struct {
    cfg: ?mlx.mlx_fast_metal_kernel_config = null,
    key: GqmvCfgKey = std.mem.zeroes(GqmvCfgKey),
};
var gqmv_cfg_cache: [2]GqmvCfgSlot = .{ .{}, .{} };

var gqmv_engaged: bool = false;
pub var decode_ladder_engaged: bool = false;

/// Decode-width gathered qmv over an expert bank, read in place.
/// `x`: [K] shared (x_per_expert=false) or [TOPK, K] (true), bf16/fp16.
/// `w`/`sc`/`bi`: the affine bank, [E, N, K*BITS/32] / [E, N, K/GS].
/// `inds`: [TOPK] uint32 bank rows. Returns [TOPK, N], or null when the shape
/// or quant geometry is outside the supported set (caller falls back).
pub fn gatherQmv(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    inds: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
    x_per_expert: bool,
) !?mlx.mlx_array {
    // Any other mode (mxfp4/mxfp8, whose scales are e8m0 over different group
    // sizes) falls back — running a bank through the wrong dequant is silently
    // wrong, not a crash.
    if (mode != .affine) return null;
    // Only widths where one uint32 holds a whole number of values (5/6-bit
    // affine is byte-packed differently — those fall back).
    if (!affineQmvUnitOk(bits, group_size)) return null;
    if (bi.ctx == null) return null;
    if (sc.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16 and xd != .float32) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    const wsh = mlx.getShape(w);
    if (wsh.len != 3) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 1) return null;
    const topk: c_int = ish[0];
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |d| xelems *= d;
    const want: i64 = if (x_per_expert) @as(i64, topk) * K else K;
    if (xelems != want) return null;

    // One simdgroup per (slot, row); 8 simdgroups to a threadgroup.
    const sgs_per_tg: c_int = 8;
    if (@rem(N, sgs_per_tg) != 0) return null;

    // CACHED, not rebuilt per call. This runs three times per MoE layer, and a
    // fresh config is ~10 FFI calls: measured `ops/forward` 3226 vs 2329 and
    // +0.4 ms of CPU graph build per token on a 40-layer MoE, which is roughly 60%
    // of the margin this path was being charged against stock `gather_qmm`.
    const key = GqmvCfgKey{ .topk = topk, .n = N, .bits = bits, .gs = group_size, .dtype = xd, .x_per_expert = x_per_expert };
    const slot = &gqmv_cfg_cache[@intFromBool(x_per_expert)];
    if (slot.cfg == null or !std.meta.eql(slot.key, key)) {
        if (slot.cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ topk, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, N, topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, sgs_per_tg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        slot.* = .{ .cfg = cfg, .key = key };
    }
    const config = slot.cfg.?;

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, inds, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvKernel(x_per_expert);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    return y;
}

// ── Fused MoE down + router weighting + K-reduction (mlxfast down-path mega-fusion class) ──
//
// The decode expert tail ran down-gather -> score multiply -> sum over K as
// three dispatches, materializing a [K, hidden] intermediate between them.
// This kernel keeps the down projection's per-row arithmetic TEXTUALLY
// identical to `gatherQmv` (same GQMV_BODY_*, same lane-strided pack loop,
// same simd_sum tree, same T(acc) rounding — so each (slot, row) value is
// bit-identical to the unfused kernel's), parks the K per-row values in
// threadgroup memory, and finishes with the composed pair's exact semantics:
// per-slot product rounded to T (the elementwise `mlx_multiply`), then an
// ascending T accumulate over slots 0..K-1 (MLX's bf16 sum reduction
// accumulates in the INPUT dtype — the moeRouterTopK lesson). One dispatch
// replaces three, and the [K, hidden] round-trip never happens.
//
// Each simdgroup owns one top-K slot and computes ROWS consecutive output
// rows for it (mlxfast tuned 1-vs-4 rows per simd; 4 won). Bit-identity is
// pinned by the "down+reduce" parity tests.
fn gatherQmvDownReduceSource() [:0]const u8 {
    // LPR lanes share one output row and ROWS = 32 / LPR rows ride one simdgroup
    // in parallel: K is short (the MoE intermediate), so one 32-lane reduction
    // per row cost more than the dot product it reduced, and K/8 packs over 32
    // lanes left a third of them idle. Every row's packs split exactly over LPR
    // lanes, and one log2(LPR)-step shuffle reduces all ROWS rows at once.
    return std.fmt.comptimePrint(
        \\auto lane = thread_index_in_simdgroup;
        \\uint slot = simdgroup_index_in_threadgroup;   // top-K slot
        \\uint tile = threadgroup_position_in_grid.x;   // block of ROWS output rows
        \\
        \\int K = int(K_size);
        \\int N = int(N_size);
        \\constexpr int VPW = (BITS == 3) ? 8 : 32 / BITS;
        \\int K_by_w = K * BITS / 32;
        \\int K_by_gs = K / GS;
        \\uint mask = (1u << BITS) - 1u;
        \\
        \\uint eid = inds[slot];
        \\size_t xoff = (size_t)slot * (size_t)K;
        \\threadgroup T slot_vals[uint(TOPK) * uint(ROWS)];
        \\uint r = lane / uint(LPR);
        \\uint sub = lane % uint(LPR);
        \\uint n = tile * uint(ROWS) + r;
        \\size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_w + (size_t)n * (size_t)K_by_w;
        \\size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
        \\float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        \\constexpr int IT = (KP + LPR - 1) / LPR;
        \\uint32_t pw[IT];
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(sub) + LPR * i;
        \\  pw[i] = (pack < KP) ? sushi_qpack<BITS>(w_q, wbase, pack) : 0u;
        \\}}
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(sub) + LPR * i;
        \\  if (pack >= KP) break;
        \\  uint32_t packed = pw[i];
        \\  int k_base = pack * VPW;
        \\  int gi = k_base / GS;
        \\{s}
        \\}}
        \\float acc = (a0 + a1) + (a2 + a3);
        \\for (uint d = uint(LPR) / 2u; d >= 1u; d >>= 1u) acc += simd_shuffle_xor(acc, ushort(d));
        \\if (sub == 0u) {{
        \\  slot_vals[slot * uint(ROWS) + r] = T(acc);
        \\}}
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\// Composed-pair replica: T-rounded per-slot product, ascending T sum.
        \\if (slot == 0 && lane < uint(ROWS)) {{
        \\  T total = T(0.0f);
        \\  for (uint s2 = 0; s2 < uint(TOPK); ++s2) {{
        \\    T p = slot_vals[s2 * uint(ROWS) + lane] * scores[s2];
        \\    total = total + p;
        \\  }}
        \\  y[(size_t)tile * (size_t)ROWS + (size_t)lane] = total;
        \\}}
    , .{
        GQMV_BODY_AFFINE,
    });
}

const GQMV_DOWNRED_SOURCE = gatherQmvDownReduceSource();
var gqmv_downred_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getGatherQmvDownReduceKernel() !mlx.mlx_fast_metal_kernel {
    if (gqmv_downred_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "inds", "scores", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_moe_gather_qmv_downred",
        in_vec,
        out_vec,
        GQMV_DOWNRED_SOURCE,
        GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_downred_kernel = kernel;
    return kernel;
}

/// Lanes per output row in the down+reduce kernels; 32/LPR rows per simdgroup.
/// 4 spills its hoisted packs, 16 measured the same as 8 on the shipped shapes.
const DOWNRED_LPR: c_int = 8;
const DownRedCfgKey = struct { topk: c_int, n: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype };
var downred_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var downred_cfg_key: DownRedCfgKey = std.mem.zeroes(DownRedCfgKey);
var downred_engaged: bool = false;
pub var downred_override: ?bool = null;

fn downReduceFusedEnabled() bool {
    return downred_override orelse true;
}

/// Fused down-projection + router weighting + K-reduction at decode width.
/// `x`: [TOPK, K] per-expert activations; `scores`: [TOPK] router weights in
/// the SAME dtype as x (the composed multiply is elementwise in that dtype —
/// a differing dtype would promote, so it declines). Returns the weighted
/// expert sum [N], or null when outside the kernel's envelope.
pub fn gatherQmvDownReduce(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    inds: mlx.mlx_array,
    scores: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
) !?mlx.mlx_array {
    if (!downReduceFusedEnabled()) return null;
    if (mode != .affine) return null;
    if (!affineQmvUnitOk(bits, group_size)) return null;
    if (bi.ctx == null) return null;
    if (sc.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(scores) != xd) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    const wsh = mlx.getShape(w);
    if (wsh.len != 3) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 1) return null;
    const topk: c_int = ish[0];
    if (topk <= 0 or topk > 32) return null; // one simdgroup per slot, <= 1024 threads
    const ssh = mlx.getShape(scores);
    if (ssh.len != 1 or ssh[0] != topk) return null;
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |d| xelems *= d;
    if (xelems != @as(i64, topk) * K) return null;
    const rows: c_int = @divExact(32, DOWNRED_LPR);
    if (@rem(N, rows) != 0) return null;

    const key = DownRedCfgKey{ .topk = topk, .n = N, .bits = bits, .gs = group_size, .dtype = xd };
    if (downred_cfg == null or !std.meta.eql(downred_cfg_key, key)) {
        if (downred_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{N};
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 1, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(N, rows) * topk * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, topk * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TOPK", topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LPR", DOWNRED_LPR));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KP", affineQmvPackUnits(K, bits)));
        downred_cfg = cfg;
        downred_cfg_key = key;
    }

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, inds, scores, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvDownReduceKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, downred_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    return y;
}

fn gatherQmvDownReduceRowsSource() [:0]const u8 {
    // The solo kernel's layout with a batch row on grid.y; per-row output must
    // stay bit-identical to the solo call (the rows-vs-solo test pins it).
    return std.fmt.comptimePrint(
        \\auto lane = thread_index_in_simdgroup;
        \\uint slot = simdgroup_index_in_threadgroup;
        \\uint tile = threadgroup_position_in_grid.x;
        \\uint row = threadgroup_position_in_grid.y;
        \\
        \\int K = int(K_size);
        \\int N = int(N_size);
        \\constexpr int VPW = (BITS == 3) ? 8 : 32 / BITS;
        \\int K_by_w = K * BITS / 32;
        \\int K_by_gs = K / GS;
        \\uint mask = (1u << BITS) - 1u;
        \\
        \\uint eid = inds[(size_t)row * (size_t)TOPK + slot];
        \\size_t xoff = ((size_t)row * (size_t)TOPK + (size_t)slot) * (size_t)K;
        \\threadgroup T slot_vals[uint(TOPK) * uint(ROWS)];
        \\uint r = lane / uint(LPR);
        \\uint sub = lane % uint(LPR);
        \\uint n = tile * uint(ROWS) + r;
        \\size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_w + (size_t)n * (size_t)K_by_w;
        \\size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
        \\float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        \\constexpr int IT = (KP + LPR - 1) / LPR;
        \\uint32_t pw[IT];
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(sub) + LPR * i;
        \\  pw[i] = (pack < KP) ? sushi_qpack<BITS>(w_q, wbase, pack) : 0u;
        \\}}
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(sub) + LPR * i;
        \\  if (pack >= KP) break;
        \\  uint32_t packed = pw[i];
        \\  int k_base = pack * VPW;
        \\  int gi = k_base / GS;
        \\{s}
        \\}}
        \\float acc = (a0 + a1) + (a2 + a3);
        \\for (uint d = uint(LPR) / 2u; d >= 1u; d >>= 1u) acc += simd_shuffle_xor(acc, ushort(d));
        \\if (sub == 0u) {{
        \\  slot_vals[slot * uint(ROWS) + r] = T(acc);
        \\}}
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\if (slot == 0 && lane < uint(ROWS)) {{
        \\  T total = T(0.0f);
        \\  for (uint s2 = 0; s2 < uint(TOPK); ++s2) {{
        \\    T p = slot_vals[s2 * uint(ROWS) + lane] * scores[(size_t)row * (size_t)TOPK + s2];
        \\    total = total + p;
        \\  }}
        \\  y[(size_t)row * (size_t)N + (size_t)tile * (size_t)ROWS + (size_t)lane] = total;
        \\}}
    , .{
        GQMV_BODY_AFFINE,
    });
}

const GQMV_DOWNRED_ROWS_SOURCE = gatherQmvDownReduceRowsSource();
var gqmv_downred_rows_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getGatherQmvDownReduceRowsKernel() !mlx.mlx_fast_metal_kernel {
    if (gqmv_downred_rows_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "inds", "scores", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_moe_gather_qmv_downred_rows",
        in_vec,
        out_vec,
        GQMV_DOWNRED_ROWS_SOURCE,
        GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_downred_rows_kernel = kernel;
    return kernel;
}

const DownRedRowsCfgKey = struct { nrows: c_int, topk: c_int, n: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype };
var downred_rows_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var downred_rows_cfg_key: DownRedRowsCfgKey = std.mem.zeroes(DownRedRowsCfgKey);
var downred_rows_engaged: bool = false;

/// Asked BEFORE the gate/up dispatch: a decline is static per geometry and
/// would otherwise cost a wasted gate/up launch on every MoE layer of every tick.
fn gatherQmvDownReduceRowsEligible(hidden_act: anytype, gate_qp: anytype, up_qp: anytype, down_qp: anytype, N: c_int) bool {
    if (!downReduceFusedEnabled()) return false;
    if (!moeRowsFusedEnabled()) return false;
    if (hidden_act != .silu) return false;
    if (gate_qp.bits != up_qp.bits or gate_qp.group_size != up_qp.group_size or gate_qp.mode != up_qp.mode) return false;
    if (down_qp.mode != .affine) return false;
    if (down_qp.bits != 2 and down_qp.bits != 4 and down_qp.bits != 8) return false;
    if (down_qp.group_size % (32 / down_qp.bits) != 0) return false;
    if (@rem(N, 4) != 0) return false;
    return true;
}

/// Row-axis twin of `gatherQmvDownReduce`: the K-expert threadgroup reduction
/// runs per row (grid.y = B) in the single-row order, never redistributed
/// across rows, so each row is bit-identical to its solo call.
pub fn gatherQmvDownReduceRows(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    inds: mlx.mlx_array,
    scores: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
) !?mlx.mlx_array {
    if (!downReduceFusedEnabled()) return null;
    if (!moeRowsFusedEnabled()) return null;
    if (mode != .affine) return null;
    if (!affineQmvUnitOk(bits, group_size)) return null;
    if (bi.ctx == null) return null;
    if (sc.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(scores) != xd) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    const wsh = mlx.getShape(w);
    if (wsh.len != 3) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 2) return null;
    const nrows: c_int = ish[0];
    const topk: c_int = ish[1];
    if (nrows < 1 or nrows > 128 or topk <= 0 or topk > 32) return null;
    const ssh = mlx.getShape(scores);
    if (ssh.len != 2 or ssh[0] != nrows or ssh[1] != topk) return null;
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |d| xelems *= d;
    if (xelems != @as(i64, nrows) * topk * K) return null;
    const rows: c_int = @divExact(32, DOWNRED_LPR);
    if (@rem(N, rows) != 0) return null;

    const key = DownRedRowsCfgKey{ .nrows = nrows, .topk = topk, .n = N, .bits = bits, .gs = group_size, .dtype = xd };
    if (downred_rows_cfg == null or !std.meta.eql(downred_rows_cfg_key, key)) {
        if (downred_rows_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ nrows, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(N, rows) * topk * 32, nrows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, topk * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TOPK", topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LPR", DOWNRED_LPR));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KP", affineQmvPackUnits(K, bits)));
        downred_rows_cfg = cfg;
        downred_rows_cfg_key = key;
    }

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, inds, scores, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvDownReduceRowsKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, downred_rows_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!downred_rows_engaged) {
        downred_rows_engaged = true;
        log.info("[moe] fused rows down+reduce kernel engaged: B={d} E_topk={d} hidden={d} bits={d} gs={d}\n", .{ nrows, topk, N, bits, group_size });
    }
    return y;
}

const SUSHI_QMV_ROWS_SOURCE =
    \\using U = float;
    \\auto simd_lid = thread_position_in_threadgroup.x;
    \\auto simd_gid = thread_position_in_threadgroup.y;
    \\auto tg = threadgroup_position_in_grid;
    \\const int in_vec_size = int(K_size);
    \\const int out_vec_size = int(N_size);
    \\constexpr int packs_per_thread = 2;
    \\constexpr int pack_factor = BITS == 3 ? 8 : 32 / BITS;
    \\constexpr int bytes_per_pack = BITS == 3 ? 3 : 4;
    \\constexpr int values_per_thread = pack_factor * packs_per_thread;
    \\constexpr int block_size = values_per_thread * 32;
    \\constexpr int scale_step_per_thread = GS / values_per_thread;
    \\const device uint8_t* ws = (const device uint8_t*)w_q;
    \\const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    \\const int in_vec_size_g = in_vec_size / GS;
    \\const int out_row = int(tg.y) * 8 + int(simd_gid) * 4;
    \\ws += out_row * in_vec_size_w + int(simd_lid) * packs_per_thread * bytes_per_pack;
    \\const device T* scales_p = scales + out_row * in_vec_size_g + int(simd_lid) / scale_step_per_thread;
    \\const device T* biases_p = biases + out_row * in_vec_size_g + int(simd_lid) / scale_step_per_thread;
    \\const device T* xp = x + int(tg.z) * in_vec_size + int(simd_lid) * values_per_thread;
    \\device T* yp = y + int(tg.z) * out_vec_size + out_row;
    \\thread U x_thread[values_per_thread];
    \\thread U result[4] = {0, 0, 0, 0};
    \\for (int k = 0; k < in_vec_size; k += block_size) {
    \\  U sum = 0;
    \\  if (BITS == 3) {
    \\    for (int i = 0; i < values_per_thread; i += 8) {
    \\      sum += xp[i] + xp[i + 1] + xp[i + 2] + xp[i + 3] + xp[i + 4] + xp[i + 5] + xp[i + 6] + xp[i + 7];
    \\      x_thread[i] = xp[i];
    \\      x_thread[i + 1] = xp[i + 1] / 8.0f;
    \\      x_thread[i + 2] = xp[i + 2] / 64.0f;
    \\      x_thread[i + 3] = xp[i + 3] / 2.0f;
    \\      x_thread[i + 4] = xp[i + 4] / 16.0f;
    \\      x_thread[i + 5] = xp[i + 5] / 128.0f;
    \\      x_thread[i + 6] = xp[i + 6] / 4.0f;
    \\      x_thread[i + 7] = xp[i + 7] / 32.0f;
    \\    }
    \\  } else if (BITS == 4) {
    \\    for (int i = 0; i < values_per_thread; i += 4) {
    \\      sum += xp[i] + xp[i + 1] + xp[i + 2] + xp[i + 3];
    \\      x_thread[i] = xp[i];
    \\      x_thread[i + 1] = xp[i + 1] / 16.0f;
    \\      x_thread[i + 2] = xp[i + 2] / 256.0f;
    \\      x_thread[i + 3] = xp[i + 3] / 4096.0f;
    \\    }
    \\  } else {
    \\    for (int i = 0; i < values_per_thread; i++) {
    \\      sum += xp[i];
    \\      x_thread[i] = xp[i];
    \\    }
    \\  }
    \\  for (int rr = 0; rr < 4; rr++) {
    \\    const device uint8_t* wl = ws + rr * in_vec_size_w;
    \\    U scv = scales_p[rr * in_vec_size_g];
    \\    U biv = biases_p[rr * in_vec_size_g];
    \\    U accum = 0;
    \\    if (BITS == 3) {
    \\      for (int i = 0; i < values_per_thread / 8; i++) {
    \\        const device uint8_t* w3 = wl + 3 * i;
    \\        thread U* x3 = x_thread + 8 * i;
    \\        accum += (w3[0] & 0x07) * x3[0];
    \\        accum += (w3[0] & 0x38) * x3[1];
    \\        accum += (w3[0] & 0xc0) * x3[2];
    \\        accum += (w3[1] & 0x01) * (x3[2] * 256.0f);
    \\        accum += (w3[1] & 0x0e) * x3[3];
    \\        accum += (w3[1] & 0x70) * x3[4];
    \\        accum += (w3[1] & 0x80) * x3[5];
    \\        accum += (w3[2] & 0x03) * (x3[5] * 256.0f);
    \\        accum += (w3[2] & 0x1c) * x3[6];
    \\        accum += (w3[2] & 0xe0) * x3[7];
    \\      }
    \\    } else if (BITS == 4) {
    \\      const device uint16_t* w16 = (const device uint16_t*)wl;
    \\      for (int i = 0; i < (values_per_thread / 4); i++) {
    \\        accum += (x_thread[4 * i] * (w16[i] & 0x000f) + x_thread[4 * i + 1] * (w16[i] & 0x00f0) + x_thread[4 * i + 2] * (w16[i] & 0x0f00) + x_thread[4 * i + 3] * (w16[i] & 0xf000));
    \\      }
    \\    } else {
    \\      for (int i = 0; i < values_per_thread; i++) {
    \\        accum += x_thread[i] * wl[i];
    \\      }
    \\    }
    \\    result[rr] += scv * accum + sum * biv;
    \\  }
    \\  ws += block_size * bytes_per_pack / pack_factor;
    \\  scales_p += block_size / GS;
    \\  biases_p += block_size / GS;
    \\  xp += block_size;
    \\}
    \\for (int rr = 0; rr < 4; rr++) {
    \\  result[rr] = simd_sum(result[rr]);
    \\  if (simd_lid == 0) {
    \\    yp[rr] = static_cast<T>(result[rr]);
    \\  }
    \\}
;

var sushi_qmv_rows_kernel: ?mlx.mlx_fast_metal_kernel = null;
const MsvQmvRowsCfgKey = struct { nrows: c_int, n: c_int, k: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype };
var sushi_qmv_rows_cfgs = QsaCfgCache(MsvQmvRowsCfgKey, 1){};
var sushi_qmv_rows_logged = false;
pub var mtp_head_row_dispatches: usize = 0;

fn getMsvQmvRowsKernel() !mlx.mlx_fast_metal_kernel {
    if (sushi_qmv_rows_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_qmv_rows",
        in_vec,
        out_vec,
        SUSHI_QMV_ROWS_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    sushi_qmv_rows_kernel = kernel;
    return kernel;
}

pub fn msvQmvRows(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    if (bits != 3 and bits != 4 and bits != 8) return null;
    if (group_size != 64) return null;
    if (sc.ctx == null or bi.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(sc) != xd or mlx.mlx_array_dtype(bi) != xd) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh.len > 8) return null;
    const K = xsh[xsh.len - 1];
    var nrows: c_int = 1;
    var i: usize = 0;
    while (i + 1 < xsh.len) : (i += 1) nrows *= xsh[i];
    if (nrows < 2 or nrows > 16) return null;
    const vpt: c_int = if (bits == 3) 16 else @intCast((32 / bits) * 2);
    const block: c_int = vpt * 32;
    if (@rem(K, block) != 0) return null;
    const wsh = mlx.getShape(w);
    if (wsh.len != 2) return null;
    const Nout = wsh[0];
    if (@rem(Nout, 8) != 0) return null;
    if (@as(i64, wsh[1]) * 32 != @as(i64, K) * bits) return null;
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;

    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    if (xsh.len == 2 and xsh[0] == nrows) {
        try mlx.check(mlx.mlx_array_set(&x2, x));
    } else {
        try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ nrows, K }, 2, s));
    }

    const key = MsvQmvRowsCfgKey{ .nrows = nrows, .n = Nout, .k = K, .bits = bits, .gs = group_size, .dtype = xd };
    const config = blk: {
        if (sushi_qmv_rows_cfgs.get(key)) |hit| break :blk hit[0];
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const y_shape = [_]c_int{ nrows, Nout };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, @divExact(Nout, 4), nrows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        sushi_qmv_rows_cfgs.put(key, .{cfg});
        break :blk cfg;
    };

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(Nout);
    const inputs = [_]mlx.mlx_array{ x2, w, sc, bi, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getMsvQmvRowsKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    mtp_head_row_dispatches += 1;
    if (!sushi_qmv_rows_logged) {
        sushi_qmv_rows_logged = true;
        log.info("[mtp] row-axis projection engaged: rows={d} K={d} N={d} bits={d} gs={d}\n", .{ nrows, K, Nout, bits, group_size });
    }
    var y2 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outputs_vec, 0));
    if (xsh.len == 2) return y2;
    var ysh_buf: [8]c_int = undefined;
    for (xsh[0 .. xsh.len - 1], 0..) |d, di| ysh_buf[di] = d;
    ysh_buf[xsh.len - 1] = Nout;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &ysh_buf, @intCast(xsh.len), s));
    _ = mlx.mlx_array_free(y2);
    return y;
}

/// Gathered matmul for MoE expert dispatch — handles both quantized and dense bf16.
/// Quantized (sc.ctx != null): mlx_gather_qmm with transpose_w=true, w stored [E,out,in].
/// Dense bf16 (sc.ctx == null): mlx_gather_mm; w was pre-transposed to [E,in,out] at load
/// (maybeTransposeForBf16 + generalized transposeBf16Weight), so x @ w is correct with no
/// transpose flag — mirrors mlx-lm's `gather_mm(x, weight.swapaxes(-1,-2))`.
const EXPERT_SLAB_ALIGN_BYTES: usize = 16;

const StreamedExpertOperands = struct {
    x_rows: mlx.mlx_array,
    gate: mlx.mlx_array,
    up: mlx.mlx_array,
    down: mlx.mlx_array,
    slab_gate_up: mlx.mlx_array,
    slab_down: mlx.mlx_array,
    slots: mlx.mlx_array,
    weights: mlx.mlx_array,
    imatrix: ?ImatrixTap = null,
};

/// imatrix capture through a streamed MoE layer. Every compute arm indexes the
/// slab by REMAPPED slot, so the tap carries the routing's own [rows, top_k]
/// GLOBAL expert ids: the statistics are keyed by those.
const ImatrixTap = struct {
    collector: *imatrix_capture.Collector,
    layer: u16,
    global_ids: mlx.mlx_array,

    /// The tap for one streamed layer, owning its [rows, top_k] view of the
    /// routing's ids; null when no capture is armed. The arms only read it, so
    /// the layer's own `deinit` frees it once.
    fn forLayer(col: ?*imatrix_capture.Collector, layer: u16, ids: mlx.mlx_array, rows: c_int, topk: c_int, s: mlx.mlx_stream) !?ImatrixTap {
        const collector = col orelse return null;
        var ids_2d = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(ids_2d);
        try mlx.check(mlx.mlx_reshape(&ids_2d, ids, &[_]c_int{ rows, topk }, 2, s));
        return .{ .collector = collector, .layer = layer, .global_ids = ids_2d };
    }

    pub fn deinit(self: ImatrixTap) void {
        _ = mlx.mlx_array_free(self.global_ids);
    }

    /// The layer's MLP input rows, [.., hidden] over `rows` tokens.
    fn observeInput(self: ImatrixTap, s: mlx.mlx_stream, x: mlx.mlx_array, rows: c_int, hidden: c_int) !void {
        var x_rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_rows);
        try mlx.check(mlx.mlx_reshape(&x_rows, x, &[_]c_int{ rows, hidden }, 2, s));
        try self.collector.observeGateUp(self.layer, x_rows, self.global_ids);
    }

    /// One MoE layer's SwiGLU activation rows. `act` is [.., inter] in whatever
    /// row order the arm produced; `ids_flat` names each of those `n` rows'
    /// global expert in the SAME order.
    fn observeDown(self: ImatrixTap, s: mlx.mlx_stream, act: mlx.mlx_array, ids_flat: mlx.mlx_array, n: c_int) !void {
        const ash = mlx.getShape(act);
        var act_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act_2d);
        try mlx.check(mlx.mlx_reshape(&act_2d, act, &[_]c_int{ n, ash[ash.len - 1] }, 2, s));
        var ids_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids_2d);
        try mlx.check(mlx.mlx_reshape(&ids_2d, ids_flat, &[_]c_int{ n, 1 }, 2, s));
        try self.collector.observeDown(self.layer, act_2d, ids_2d);
    }

    /// `act` in the sorted arm's row order; `inv_order` is that arm's own
    /// permutation back to the routing's order, where the global ids live.
    fn observeDownSorted(self: ImatrixTap, s: mlx.mlx_stream, act: mlx.mlx_array, inv_order: mlx.mlx_array, n: c_int) !void {
        const ash = mlx.getShape(act);
        var act_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act_2d);
        try mlx.check(mlx.mlx_reshape(&act_2d, act, &[_]c_int{ n, ash[ash.len - 1] }, 2, s));
        var routed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(routed);
        try mlx.check(mlx.mlx_take_axis(&routed, act_2d, inv_order, 0, s));
        try self.observeDown(s, routed, self.global_ids, n);
    }
};

var streamed_expert_kernels_engaged: bool = false;

fn expertSlabAligned(slab: mlx.mlx_array) !void {
    try mlx.check(mlx.mlx_array_eval(slab));
    const p = mlx.mlx_array_data_uint8(slab) orelse return error.ExpertSlabUnaligned;
    if (@intFromPtr(p) % EXPERT_SLAB_ALIGN_BYTES != 0) return error.ExpertSlabUnaligned;
}

fn streamedExpertSwiglu(s: mlx.mlx_stream, gate: mlx.mlx_array, up: mlx.mlx_array) !mlx.mlx_array {
    if (try fusedSwiGLU(s, gate, up)) |fused| return fused;
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, gate, s));
    var activated = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(activated);
    try mlx.check(mlx.mlx_multiply(&activated, gate, sig, s));
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&result, activated, up, s));
    return result;
}

fn streamedExpertUnsortedDown(s: mlx.mlx_stream, op: StreamedExpertOperands, activation: mlx.mlx_array, rows: c_int, topk: c_int) !mlx.mlx_array {
    const no_idx = mlx.mlx_array{ .ctx = null };
    const no_side = mlx.mlx_array{ .ctx = null };
    var inds3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inds3);
    try mlx.check(mlx.mlx_reshape(&inds3, op.slots, &[_]c_int{ 1, rows, topk }, 3, s));
    var act_exp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act_exp);
    try mlx.check(mlx.mlx_expand_dims(&act_exp, activation, -2, s));
    var down_5d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(down_5d);
    try gatherExpertMm(&down_5d, act_exp, op.down, no_side, no_side, no_idx, inds3, 0, 64, .affine, false, s);
    var squeezed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(squeezed);
    try mlx.check(mlx.mlx_squeeze(&squeezed, down_5d, s));
    const sh = mlx.getShape(squeezed);
    const hidden = sh[sh.len - 1];
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, squeezed, &[_]c_int{ rows, topk, hidden }, 3, s));
    return out;
}

fn streamedExpertCompositeDown(s: mlx.mlx_stream, op: StreamedExpertOperands, rows: c_int, topk: c_int) !mlx.mlx_array {
    const D = mlx.getShape(op.x_rows)[1];
    const total_inds: c_int = rows * topk;
    const no_idx = mlx.mlx_array{ .ctx = null };
    const no_side = mlx.mlx_array{ .ctx = null };
    if (rows > 1 or total_inds >= 64) {
        var flat_inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat_inds);
        try mlx.check(mlx.mlx_reshape(&flat_inds, op.slots, &[_]c_int{total_inds}, 1, s));
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try mlx.check(mlx.mlx_argsort_axis(&order, flat_inds, 0, s));
        var inv_order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inv_order);
        try mlx.check(mlx.mlx_argsort_axis(&inv_order, order, 0, s));
        var sorted_inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted_inds);
        try mlx.check(mlx.mlx_take_axis(&sorted_inds, flat_inds, order, 0, s));
        const k_arr = mlx.mlx_array_new_int(topk);
        defer _ = mlx.mlx_array_free(k_arr);
        var lhs_idx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lhs_idx);
        try mlx.check(mlx.mlx_floor_divide(&lhs_idx, order, k_arr, s));
        var x_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_flat);
        try mlx.check(mlx.mlx_reshape(&x_flat, op.x_rows, &[_]c_int{ rows, D }, 2, s));
        var x_gathered = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_gathered);
        try mlx.check(mlx.mlx_take_axis(&x_gathered, x_flat, lhs_idx, 0, s));
        var x_rep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_rep);
        try mlx.check(mlx.mlx_reshape(&x_rep, x_gathered, &[_]c_int{ total_inds, 1, D }, 3, s));
        var gate_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_3d);
        try gatherExpertMm(&gate_3d, x_rep, op.gate, no_side, no_side, no_idx, sorted_inds, 0, 64, .affine, true, s);
        var gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate);
        try mlx.check(mlx.mlx_squeeze(&gate, gate_3d, s));
        var up_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up_3d);
        try gatherExpertMm(&up_3d, x_rep, op.up, no_side, no_side, no_idx, sorted_inds, 0, 64, .affine, true, s);
        var up = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up);
        try mlx.check(mlx.mlx_squeeze(&up, up_3d, s));
        const activation = try streamedExpertSwiglu(s, gate, up);
        defer _ = mlx.mlx_array_free(activation);
        if (op.imatrix) |tap| {
            var flat_global = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat_global);
            try mlx.check(mlx.mlx_reshape(&flat_global, tap.global_ids, &[_]c_int{total_inds}, 1, s));
            var sorted_global = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sorted_global);
            try mlx.check(mlx.mlx_take_axis(&sorted_global, flat_global, order, 0, s));
            try tap.observeDown(s, activation, sorted_global, total_inds);
        }
        var act_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act_exp);
        try mlx.check(mlx.mlx_expand_dims(&act_exp, activation, -2, s));
        var down_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_3d);
        try gatherExpertMm(&down_3d, act_exp, op.down, no_side, no_side, no_idx, sorted_inds, 0, 64, .affine, true, s);
        var down_squeezed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_squeezed);
        try mlx.check(mlx.mlx_squeeze(&down_squeezed, down_3d, s));
        var down_unsorted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_unsorted);
        try mlx.check(mlx.mlx_take_axis(&down_unsorted, down_squeezed, inv_order, 0, s));
        const hidden = mlx.getShape(down_unsorted)[1];
        var down_out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&down_out, down_unsorted, &[_]c_int{ rows, topk, hidden }, 3, s));
        return down_out;
    }
    var x_exp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_exp);
    try mlx.check(mlx.mlx_reshape(&x_exp, op.x_rows, &[_]c_int{ 1, rows, 1, 1, D }, 5, s));
    var inds3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inds3);
    try mlx.check(mlx.mlx_reshape(&inds3, op.slots, &[_]c_int{ 1, rows, topk }, 3, s));
    var gate_5d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_5d);
    try gatherExpertMm(&gate_5d, x_exp, op.gate, no_side, no_side, no_idx, inds3, 0, 64, .affine, false, s);
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_squeeze(&gate, gate_5d, s));
    var up_5d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up_5d);
    try gatherExpertMm(&up_5d, x_exp, op.up, no_side, no_side, no_idx, inds3, 0, 64, .affine, false, s);
    var up = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up);
    try mlx.check(mlx.mlx_squeeze(&up, up_5d, s));
    const activation = try streamedExpertSwiglu(s, gate, up);
    defer _ = mlx.mlx_array_free(activation);
    if (op.imatrix) |tap| try tap.observeDown(s, activation, tap.global_ids, rows * topk);
    return streamedExpertUnsortedDown(s, op, activation, rows, topk);
}

fn streamedExpertWeightedReduce(s: mlx.mlx_stream, down_out: mlx.mlx_array, weights: mlx.mlx_array) !mlx.mlx_array {
    var w_exp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_exp);
    try mlx.check(mlx.mlx_expand_dims(&w_exp, weights, -1, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, down_out, w_exp, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sum_axis(&out, weighted, -2, false, s));
    return out;
}

fn streamedExpertCompute(s: mlx.mlx_stream, op: StreamedExpertOperands, rows: c_int, use_kernels: bool) !mlx.mlx_array {
    const x_shape = mlx.getShape(op.x_rows);
    if (x_shape.len != 2 or x_shape[0] != rows) return error.StreamedExpertShapeMismatch;
    const slot_shape = mlx.getShape(op.slots);
    if (slot_shape.len != 2 or slot_shape[0] != rows) return error.StreamedExpertShapeMismatch;
    const topk = slot_shape[1];
    if (op.imatrix) |tap| try tap.observeInput(s, op.x_rows, rows, x_shape[1]);
    if (use_kernels and rows >= 1 and rows <= expert_bf16.MAX_ROWS and topk >= 1 and topk <= expert_bf16.MAX_TOPK) {
        try expertSlabAligned(op.slab_gate_up);
        try expertSlabAligned(op.slab_down);
        const h = try expert_bf16.gateUpSwiglu(s, op.x_rows, op.slab_gate_up, op.slots);
        defer _ = mlx.mlx_array_free(h);
        if (op.imatrix) |tap| try tap.observeDown(s, h, tap.global_ids, rows * topk);
        const down_kernel = expert_bf16.downKernelPreferred(@intCast(rows));
        if (!streamed_expert_kernels_engaged) {
            streamed_expert_kernels_engaged = true;
            log.info("[expert-bf16] streamed MoE kernel arm engaged: rows={d} topk={d} down={s}\n", .{ rows, topk, if (down_kernel) "kernel" else "gather_mm" });
        }
        if (down_kernel) {
            var w32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(w32);
            try mlx.check(mlx.mlx_astype(&w32, op.weights, .float32, s));
            return expert_bf16.downReduce(s, h, op.slab_down, op.slots, w32);
        }
        var act = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act);
        if (rows == 1) {
            try mlx.check(mlx.mlx_reshape(&act, h, &[_]c_int{ topk, mlx.getShape(h)[2] }, 2, s));
        } else {
            try mlx.check(mlx.mlx_array_set(&act, h));
        }
        const down_out = try streamedExpertUnsortedDown(s, op, act, rows, topk);
        defer _ = mlx.mlx_array_free(down_out);
        return streamedExpertWeightedReduce(s, down_out, op.weights);
    }
    const down_out = try streamedExpertCompositeDown(s, op, rows, topk);
    defer _ = mlx.mlx_array_free(down_out);
    return streamedExpertWeightedReduce(s, down_out, op.weights);
}

fn gatherExpertMm(res: *mlx.mlx_array, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, lhs_idx: mlx.mlx_array, rhs_idx: mlx.mlx_array, bits: u32, group_size: u32, mode: QuantMode, sorted: bool, s: mlx.mlx_stream) !void {
    if (sc.ctx == null) {
        // mlx 0.31.2's `mlx_gather_mm` returns WRONG results with sorted_indices=true
        // for the dense (non-quantized) path — the quantized `mlx_gather_qmm` honors
        // the flag correctly, but the dense kernel does not. The sorted-indices flag is
        // only a performance hint (rhs_indices ARE sorted here), so forcing false is
        // always numerically correct; it just forgoes the sorted-stream optimization.
        // Without this, dense bf16 MoE prefill (the sorted path, S>1) produced fluent
        // but semantically-wrong output (e.g. a bf16 MoE calling clean prompts
        // "jumbled"). Verified byte-identical to mlx-lm once forced false. `sorted` is
        // still honored by the quantized branch below, where gather_qmm handles it
        // correctly.
        try mlx.check(mlx.mlx_gather_mm(res, x, w, lhs_idx, rhs_idx, false, s));
    } else {
        try mlx.check(mlx.mlx_gather_qmm(res, x, w, sc, bi, lhs_idx, rhs_idx, true, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), sorted, s));
    }
}

// ── Prefill-width dequant+GEMM qmm route ──
// Stock MLX qmm_t runs a 32x32x32 steel tile; at prefill widths (M >= 2048)
// on the qwen 27B shapes that underuses the compute units by ~10% (µbench,
// M4 Max: gate/up q4 M=2048 stock 28.05 ms vs dequant+bf16-GEMM 25.36 ms
// INCLUDING the per-call dequant; pre-dequantized floor 24.64 ms). oMLX
// closes the same gap by re-instantiating qmm_t at bigger tiles (bm 64-128,
// their qwen35_q4_mlp patch); we get within ~3% of that with zero custom
// kernel: dequantize the weight to a bf16 [N,K] transient and run MLX's
// steel GEMM. The transient repeats identical sizes per layer, so the MLX
// allocator cache recycles it; numerics differ from in-kernel dequant only
// by the bf16 rounding of w = s*q + b (pinned no-worse-than-stock by test).
// Decode (M=1) and spec-verify widths never route.
pub const PREFILL_DQ_GEMM_MIN_M: usize = 2048;

/// Test seam: forces the route on/off without the environment.
pub var prefill_dq_gemm_override: ?bool = null;

pub fn prefillDqGemmEnabled() bool {
    return prefill_dq_gemm_override orelse true;
}

/// Test seam: engagement is counted, never inferred from output equality.
pub var prefill_dq_gemm_engaged: u64 = 0;

var verify_shared_arch: ?bool = null;

// Preserve MLX's affine qmv K reduction while sharing each packed word across a pair.
const VERIFY_EXPERT_REUSE_SOURCE =
    \\auto lane = thread_index_in_simdgroup;
    \\auto sg = simdgroup_index_in_threadgroup;
    \\auto tg = threadgroup_position_in_grid;
    \\uint row = tg.z;
    \\uint eid = ids[row];
    \\uint first = row;
    \\while (first > 0 && ids[first - 1] == eid) --first;
    \\if ((row - first) % 2 != 0) return;
    \\bool pair = row + 1 < uint(ids_shape[0]) && ids[row + 1] == eid;
    \\uint out_row = tg.y * 8 + sg * 4;
    \\float2 result[4] = {float2(0), float2(0), float2(0), float2(0)};
    \\constexpr uint VPT = 8;
    \\for (uint k = 0; k < K; k += VPT * 32) {
    \\  uint offset = k + lane * VPT;
    \\  if (offset >= K) continue;
    \\  float2 xt[VPT];
    \\  float2 sum = float2(0);
    \\  for (uint i = 0; i < VPT; i += 4) {
    \\    size_t p = size_t(row) * K + offset + i;
    \\    T a0 = x[p], a1 = x[p+1], a2 = x[p+2], a3 = x[p+3];
    \\    T b0 = T(0), b1 = T(0), b2 = T(0), b3 = T(0);
    \\    if (pair) { b0 = x[p+K]; b1 = x[p+K+1]; b2 = x[p+K+2]; b3 = x[p+K+3]; }
    \\    sum.x += a0 + a1 + a2 + a3;
    \\    sum.y += b0 + b1 + b2 + b3;
    \\    xt[i] = float2(float(a0), float(b0));
    \\    xt[i+1] = float2(float(a1), float(b1)) / 16.0f;
    \\    xt[i+2] = float2(float(a2), float(b2)) / 256.0f;
    \\    xt[i+3] = float2(float(a3), float(b3)) / 4096.0f;
    \\  }
    \\  for (uint r = 0; r < 4; ++r) {
    \\    size_t wr = size_t(eid) * N + out_row + r;
    \\    uint packed[VPT/8];
    \\    for (uint i = 0; i < VPT/8; ++i) packed[i] = w[wr * (K/8) + offset/8 + i];
    \\    float scale = float(sc[wr * (K/64) + offset/64]);
    \\    float bias = float(bi[wr * (K/64) + offset/64]);
    \\    float2 accum = float2(0);
    \\    for (uint i = 0; i < VPT/4; ++i) {
    \\      uint ws = (packed[i/2] >> (16*(i%2))) & 65535;
    \\      accum += (xt[4*i] * float(ws & 0x000f) + xt[4*i+1] * float(ws & 0x00f0) + xt[4*i+2] * float(ws & 0x0f00) + xt[4*i+3] * float(ws & 0xf000));
    \\    }
    \\    result[r] += scale * accum + sum * bias;
    \\  }
    \\}
    \\for (uint r = 0; r < 4; ++r) {
    \\  float a = simd_sum(result[r].x);
    \\  float b = simd_sum(result[r].y);
    \\  if (lane == 0) {
    \\    y[size_t(row) * N + out_row + r] = T(a);
    \\    if (pair) y[size_t(row+1) * N + out_row + r] = T(b);
    \\  }
    \\}
;

var verify_expert_reuse_kernel: ?mlx.mlx_fast_metal_kernel = null;

const ExpertReuseKey = struct { rows: c_int, n: c_int, k: c_int };
var verify_expert_reuse_cfgs = QsaCfgCache(ExpertReuseKey, 1){};

fn verifyExpertReuse(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, ids: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware()) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null or ids.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(ids) != .uint32) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    const bs = mlx.getShape(bi);
    const is = mlx.getShape(ids);
    if (xs.len != 3 or xs[1] != 1 or ws.len != 3 or ss.len != 3 or !std.mem.eql(c_int, ss, bs) or is.len != 1 or is[0] != xs[0]) return null;
    const count = xs[0];
    const k = xs[2];
    const n = ws[1];
    if (count < 2 or count > 320 or ws[0] < 1 or @divTrunc(count, ws[0]) >= 4) return null;
    if (k != 640 or n != 2560) return null;
    if (ws[2] != @divExact(k, 8) or ss[0] != ws[0] or ss[1] != n or ss[2] != @divExact(k, 64)) return null;
    const kernel = blk: {
        if (verify_expert_reuse_kernel) |value| break :blk value;
        const ins = [_][*:0]const u8{ "x", "w", "sc", "bi", "ids" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("verify_expert_reuse", iv, ov, VERIFY_EXPERT_REUSE_SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        verify_expert_reuse_kernel = value;
        break :blk value;
    };
    const key = ExpertReuseKey{ .rows = count, .n = n, .k = k };
    const cfg = blk: {
        if (verify_expert_reuse_cfgs.get(key)) |hit| break :blk hit[0];
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ count, 1, n }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, 32, @divExact(n, 4), count));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, "K", k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, "N", n));
        verify_expert_reuse_cfgs.put(key, .{value});
        break :blk value;
    };
    const values = [_]mlx.mlx_array{ x, w, sc, bi, ids };
    const iv = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

// Match the eight bf16 partial sums used by MLX col_reduce_small for top-10.
const VERIFY_EXPERT_REDUCE_SOURCE =
    \\uint p = thread_position_in_grid.x;
    \\uint row = p / 2560;
    \\uint col = p % 2560;
    \\T partial[8];
    \\#pragma unroll
    \\for (uint i = 0; i < 8; ++i) partial[i] = T(0);
    \\#pragma unroll
    \\for (uint e = 0; e < 10; ++e) {
    \\  uint at = row * 10 + e;
    \\  T product = down[size_t(inverse[at]) * 2560 + col] * scores[at];
    \\  partial[e % 8] = product + partial[e % 8];
    \\}
    \\T total = partial[0];
    \\#pragma unroll
    \\for (uint i = 1; i < 8; ++i) total = partial[i] + total;
    \\y[p] = total;
;
var verify_expert_reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;
var verify_expert_reduce_cfgs = QsaCfgCache(c_int, 1){};

fn verifyExpertReduce(s: mlx.mlx_stream, down: mlx.mlx_array, inverse: mlx.mlx_array, scores: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware()) return null;
    if (down.ctx == null or inverse.ctx == null or scores.ctx == null) return null;
    if (mlx.mlx_array_dtype(down) != .bfloat16 or mlx.mlx_array_dtype(scores) != .bfloat16 or mlx.mlx_array_dtype(inverse) != .uint32) return null;
    const ds = mlx.getShape(down);
    const ss = mlx.getShape(scores);
    const ix = mlx.getShape(inverse);
    if (ds.len != 2 or ds[1] != 2560 or ss.len != 3 or ss[0] != 1 or ss[2] != 10 or ix.len != 1) return null;
    const tokens = ss[1];
    if (tokens < 2 or tokens > 32 or ds[0] != tokens * 10 or ix[0] != ds[0]) return null;
    const kernel = blk: {
        if (verify_expert_reduce_kernel) |value| break :blk value;
        const ins = [_][*:0]const u8{ "down", "inverse", "scores" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("verify_expert_reduce", iv, ov, VERIFY_EXPERT_REDUCE_SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        verify_expert_reduce_kernel = value;
        break :blk value;
    };
    const cfg = blk: {
        if (verify_expert_reduce_cfgs.get(tokens)) |hit| break :blk hit[0];
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ 1, tokens, 2560 }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, tokens * 2560, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        verify_expert_reduce_cfgs.put(tokens, .{value});
        break :blk value;
    };
    const values = [_]mlx.mlx_array{ down, inverse, scores };
    const iv = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    return result;
}

// MLX affine qmv-fast reduction, with one 3-bit weight read serving two rows.
const MTP_COARSE_PAIRS_SOURCE =
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\auto tg = threadgroup_position_in_grid;
    \\int K = int(K_size), N = int(N_size);
    \\uint row = tg.z * 2;
    \\bool pair = row + 1 < uint(x_shape[0]);
    \\uint out_row = tg.y * 8 + sg * 4;
    \\const device uint8_t* w8 = (const device uint8_t*)w;
    \\float2 result[4] = {float2(0), float2(0), float2(0), float2(0)};
    \\for (int k = 0; k < K; k += 512) {
    \\  float2 xt[16];
    \\  float2 sum = float2(0);
    \\  for (uint i = 0; i < 16; i += 8) {
    \\    size_t at = size_t(row) * K + k + lane * 16 + i;
    \\    const device T* a = x + at;
    \\    sum.x += a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
    \\    float b[8] = {0,0,0,0,0,0,0,0};
    \\    if (pair) {
    \\      const device T* v = a + K;
    \\      sum.y += v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7];
    \\      for (uint j = 0; j < 8; ++j) b[j] = float(v[j]);
    \\    }
    \\    xt[i] = float2(float(a[0]), b[0]);
    \\    xt[i+1] = float2(float(a[1]), b[1]) / 8.0f;
    \\    xt[i+2] = float2(float(a[2]), b[2]) / 64.0f;
    \\    xt[i+3] = float2(float(a[3]), b[3]) / 2.0f;
    \\    xt[i+4] = float2(float(a[4]), b[4]) / 16.0f;
    \\    xt[i+5] = float2(float(a[5]), b[5]) / 128.0f;
    \\    xt[i+6] = float2(float(a[6]), b[6]) / 4.0f;
    \\    xt[i+7] = float2(float(a[7]), b[7]) / 32.0f;
    \\  }
    \\  for (uint r = 0; r < 4; ++r) {
    \\    const device uint8_t* packed = w8 + size_t(out_row+r) * (K*3/8) + k*3/8 + lane*6;
    \\    float scale = float(sc[size_t(out_row+r) * (K/64) + k/64 + lane/4]);
    \\    float bias = float(bi[size_t(out_row+r) * (K/64) + k/64 + lane/4]);
    \\    float2 accum = float2(0);
    \\    for (uint i = 0; i < 2; ++i) {
    \\      uint a = packed[3*i], b = packed[3*i+1], c = packed[3*i+2];
    \\      accum += float(a & 0x07) * xt[8*i];
    \\      accum += float(a & 0x38) * xt[8*i+1];
    \\      accum += float(a & 0xc0) * xt[8*i+2];
    \\      accum += float(b & 0x01) * (xt[8*i+2] * 256.0f);
    \\      accum += float(b & 0x0e) * xt[8*i+3];
    \\      accum += float(b & 0x70) * xt[8*i+4];
    \\      accum += float(b & 0x80) * xt[8*i+5];
    \\      accum += float(c & 0x03) * (xt[8*i+5] * 256.0f);
    \\      accum += float(c & 0x1c) * xt[8*i+6];
    \\      accum += float(c & 0xe0) * xt[8*i+7];
    \\    }
    \\    result[r] += scale * accum + sum * bias;
    \\  }
    \\}
    \\for (uint r = 0; r < 4; ++r) {
    \\  float a = simd_sum(result[r].x), b = simd_sum(result[r].y);
    \\  if (lane == 0) {
    \\    y[size_t(row)*N + out_row+r] = T(a);
    \\    if (pair) y[size_t(row+1)*N + out_row+r] = T(b);
    \\  }
    \\}
;
var mtp_coarse_pairs_kernel: ?mlx.mlx_fast_metal_kernel = null;
var mtp_coarse_pairs_cfgs = QsaCfgCache(c_int, 1){};

pub fn mtpCoarsePairs(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, gs: u32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or bits != 3 or gs != 64) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    if (xs.len != 3 or xs[0] < 2 or xs[0] > 8 or xs[1] != 1 or xs[2] != 2560) return null;
    if (!std.mem.eql(c_int, ws, &.{ 248320, 240 }) or !std.mem.eql(c_int, ss, &.{ 248320, 40 }) or !std.mem.eql(c_int, ss, mlx.getShape(bi))) return null;
    const kernel = blk: {
        if (mtp_coarse_pairs_kernel) |value| break :blk value;
        const ins = [_][*:0]const u8{ "x", "w", "sc", "bi", "K_size", "N_size" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("mtp_coarse_pairs", iv, ov, MTP_COARSE_PAIRS_SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        mtp_coarse_pairs_kernel = value;
        break :blk value;
    };
    const cfg = blk: {
        if (mtp_coarse_pairs_cfgs.get(xs[0])) |hit| break :blk hit[0];
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ xs[0], 1, 248320 }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, 32, 248320 / 4, @divTrunc(xs[0] + 1, 2)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        mtp_coarse_pairs_cfgs.put(xs[0], .{value});
        break :blk value;
    };
    const values = [_]mlx.mlx_array{ x, w, sc, bi, cachedScalarInt(2560), cachedScalarInt(248320) };
    const iv = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    mtp_coarse_pair_dispatches +%= 1;
    const Once = struct {
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[mtp] paired coarse head engaged (rows={d})\n", .{xs[0]});
    }
    return result;
}

// Preserve input order on ties so routing and the inverse permutation stay stable.
const VERIFY_ROUTE_PACK_SOURCE =
    \\threadgroup uint values[320];
    \\const uint n = uint(ids_shape[0]);
    \\const uint lane = thread_index_in_threadgroup;
    \\for (uint i = lane; i < n; i += 128) values[i] = ids[i];
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for (uint i = lane; i < n; i += 128) {
    \\  const uint key = values[i];
    \\  uint rank = 0;
    \\  for (uint j = 0; j < n; ++j) {
    \\    const uint other = values[j];
    \\    rank += uint(other < key || (other == key && j < i));
    \\  }
    \\  inverse[i] = rank;
    \\  sorted[rank] = key;
    \\  lhs[rank] = i / 10;
    \\}
;
var verify_route_pack_kernel: ?mlx.mlx_fast_metal_kernel = null;
var verify_route_pack_configs = QsaCfgCache(c_int, 1){};
const VerifyRoutePack = struct {
    inverse: mlx.mlx_array = .{},
    sorted: mlx.mlx_array = .{},
    lhs: mlx.mlx_array = .{},
    pub fn deinit(self: *VerifyRoutePack) void {
        inline for (.{ "inverse", "sorted", "lhs" }) |name| {
            if (@field(self, name).ctx != null) _ = mlx.mlx_array_free(@field(self, name));
        }
    }
};

fn verifyRoutePack(s: mlx.mlx_stream, ids: mlx.mlx_array, top_k: c_int) !?VerifyRoutePack {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or top_k != 10 or ids.ctx == null or mlx.mlx_array_dtype(ids) != .uint32) return null;
    const shape = mlx.getShape(ids);
    if (shape.len != 1 or shape[0] < 20 or shape[0] > 320 or @mod(shape[0], 10) != 0) return null;
    const kernel = blk: {
        if (verify_route_pack_kernel) |kernel| break :blk kernel;
        const ins = [_][*:0]const u8{"ids"};
        const outs = [_][*:0]const u8{ "inverse", "sorted", "lhs" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const kernel = mlx.mlx_fast_metal_kernel_new("verify_route_pack", iv, ov, VERIFY_ROUTE_PACK_SOURCE, "", true, false);
        if (kernel.ctx == null) return error.MetalKernelCompileFailed;
        verify_route_pack_kernel = kernel;
        break :blk kernel;
    };
    const config = blk: {
        if (verify_route_pack_configs.get(shape[0])) |hit| break :blk hit[0];
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, shape.ptr, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1));
        verify_route_pack_configs.put(shape[0], .{config});
        break :blk config;
    };
    const iv = mlx.mlx_vector_array_new_data(&.{ids}, 1);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, config, s));
    var result: VerifyRoutePack = .{};
    errdefer result.deinit();
    inline for (.{ "inverse", "sorted", "lhs" }, 0..) |name, i| {
        @field(result, name) = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&@field(result, name), ov, i));
    }
    return result;
}

fn verifyIndexedExpertInput(s: mlx.mlx_stream, x: mlx.mlx_array, lhs: mlx.mlx_array, experts: c_int) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or experts <= 0) return null;
    if (x.ctx == null or lhs.ctx == null or mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(lhs) != .uint32) return null;
    const xs = mlx.getShape(x);
    const ix = mlx.getShape(lhs);
    if (xs.len != 2 or xs[1] != 2560 or xs[0] < 2 or xs[0] > 32 or ix.len != 1 or ix[0] != xs[0] * 10) return null;
    // Explicit LHS indices disable MLX's sorted matrix path; keep both arms below it.
    if (@divTrunc(ix[0], experts) >= 4) return null;
    var view = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_reshape(&view, x, &.{ xs[0], 1, xs[1] }, 3, s));
    return view;
}

pub fn verifySharedHardware() bool {
    if (!verifyQmmNaxAvailable()) return false;
    if (verify_shared_arch) |value| return value;
    var arch_buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&arch_buf) orelse return false;
    const value = std.mem.startsWith(u8, arch, "applegpu_g17") and !std.mem.endsWith(u8, arch, "d");
    verify_shared_arch = value;
    return value;
}

pub const VerifyProjectionKind = enum { shared, gdn, lm_head, attention };

fn verifySharedProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    return verifyJoinedProjection(s, x, w, sc, bi, widths, .shared);
}

pub fn verifyJoinedProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int, comptime kind: VerifyProjectionKind) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or widths.len < 2 or widths.len > 8) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or
        mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    const bs = mlx.getShape(bi);
    if (xs.len != 3 or xs[0] != 1 or ws.len != 2 or ss.len != 2 or bs.len != 2) return null;
    const k = xs[2];
    const n = ws[0];
    const wide = kind != .shared;
    const shape_ok = switch (kind) {
        .shared => (k == 2560 and n == 640) or (k == 640 and n == 2560),
        .gdn => (k == 2560 and (n == 10240 or n == 6144 or n == 48)) or (k == 6144 and n == 2560),
        .lm_head => k == 2560 and n == 248320,
        .attention => (k == 2560 and (n == 12288 or n == 512)) or (k == 6144 and n == 2560),
    };
    if (!shape_ok) return null;
    if (ws[1] != @divExact(k, 4) or ss[0] != n or ss[1] != @divExact(k, 64) or !std.mem.eql(c_int, ss, bs)) return null;
    // The joined path requires the reference qmv_wide K reduction.
    if (mixedPlainEnabled(8, 64, n)) return null;
    var total: c_int = 0;
    for (widths) |width| {
        if (width < 2 or width > (if (wide) @as(c_int, 6) else 7)) return null;
        if (wide and verifyQmmEnabled() and
            mixedNaxShapeEnabled(8, 64, n) and width >= naxMinMForBits(8) and @mod(k, 256) == 0) return null;
        total += width;
    }
    if (total != xs[1] or total > (if (wide) @as(c_int, 12) else 24)) return null;
    const value = (try @import("mtp_qmv.zig").matmul(s, x, w, sc, bi)) orelse return error.Affine8SerialProjectionUnsupported;
    mtp_verify_kernel_calls[0] +%= 1;
    return value;
}

/// One row per slot's decode tick, any affine8 group size: each row is computed as the single-row qmv computes it.
pub fn decodeRowsProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or widths.len < 2 or widths.len > 8) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    if (xs.len != 3 or xs[0] != 1 or xs[1] != @as(c_int, @intCast(widths.len)) or ws.len != 2 or ss.len != 2 or ss[0] != ws[0]) return null;
    for (widths) |width| if (width != 1) return null;
    const geom = @import("expert_quant.zig").affineGeomFromShapes(@intCast(ws[1]), @intCast(ss[1]), @intCast(xs[2])) orelse return null;
    if (geom.bits != 8 or mixedPlainEnabled(8, geom.group_size, ws[0])) return null;
    return @import("mtp_qmv.zig").matmul(s, x, w, sc, bi);
}

pub fn qmatmulBits(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, mode: QuantMode, s: mlx.mlx_stream) !mlx.mlx_array {
    // Unquantized weights are pre-transposed to [in, out] at load.
    // Short forwards retain each serial row's matmul arithmetic.
    if (sc.ctx == null) {
        if (try @import("mtp_qmv.zig").denseMatmul(s, x, w)) |out| return out;
        var fp_result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&fp_result, x, w, s));
        return fp_result;
    }

    // Non-affine model-wide mode (mxfp4 / mxfp8 from config.json):
    // bias-less by construction; bits/group_size come from the config like
    // affine. Checked BEFORE the legacy null-bias heuristic below, which
    // would misroute e.g. an mxfp4 weight to mxfp8.
    if (mode != .affine) {
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(
            &result,
            x,
            w,
            sc,
            mlx.mlx_array{ .ctx = null },
            true,
            mlx.mlx_optional_int.some(@intCast(group_size)),
            mlx.mlx_optional_int.some(@intCast(bits)),
            mode.cstr(),
            s,
        ));
        return result;
    }

    // Legacy heuristic for mixed checkpoints whose config declares affine but
    // whose individual tensors lack biases (NVIDIA mxfp8 layers): biases array
    // has null ctx (created with mlx_array_new()).
    const is_mxfp = bi.ctx == null;

    if (is_mxfp) {
        // mxfp8: bits is always 8; infer group_size from scales/weight shape ratio
        const mxfp_bits: u32 = 8;
        const s_shape = mlx.getShape(sc);
        const w_shape = mlx.getShape(w);
        const mxfp_gs: u32 = if (s_shape.len >= 2 and w_shape.len >= 2) blk: {
            const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
            const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
            if (s_cols > 0) break :blk (w_cols * 32) / (s_cols * mxfp_bits);
            break :blk 32;
        } else 32;

        const null_bi = mlx.mlx_array{ .ctx = null };
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(
            &result,
            x,
            w,
            sc,
            null_bi,
            true,
            mlx.mlx_optional_int.some(@intCast(mxfp_gs)),
            mlx.mlx_optional_int.some(@intCast(mxfp_bits)),
            "mxfp8",
            s,
        ));
        return result;
    }

    if (bits == 8 and mode == .affine) {
        const shape = mlx.getShape(x);
        if (shape.len >= 2 and shape[shape.len - 1] > 0) {
            const rows = mlx.mlx_array_size(x) / @as(usize, @intCast(shape[shape.len - 1]));
            if (rows >= 2 and rows <= 32)
                return (try @import("mtp_qmv.zig").matmul(s, x, w, sc, bi)) orelse error.Affine8SerialProjectionUnsupported;
        }
    }

    // Spec-verify fast path: M=2..6-row activations (verify forwards, small
    // decode batches) route through the split-K verify kernel when eligible
    // (see VERIFY_QMM_SOURCE) — stock qmm's qmv/steel dead zone.
    if (try verifyQmm(s, x, w, sc, bi, bits, group_size)) |vy| return vy;

    // Prefill-width fast path: M >= 2048 dequantizes + runs the steel bf16
    // GEMM (see prefillDqGemm) — stock qmm_t's 32x32x32 tile dead zone.
    if (try prefillDqGemm(x, w, sc, bi, bits, group_size, s)) |py| return py;

    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_quantized_matmul(
        &result,
        x,
        w,
        sc,
        bi,
        true,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        s,
    ));
    return result;
}

/// The dequant+GEMM prefill route (see the block comment above
/// prefillDqGemmEnabled). Returns null when the call must stay on stock qmm.
fn prefillDqGemm(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!prefillDqGemmEnabled()) return null;
    switch (bits) {
        2, 3, 4, 5, 6, 8 => {},
        else => return null,
    }
    const last = lastDim(x) orelse return null;
    if (last == 0) return null;
    const rows = mlx.mlx_array_size(x) / @as(usize, @intCast(last));
    if (rows < PREFILL_DQ_GEMM_MIN_M) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;

    var dq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dq);
    try mlx.check(mlx.mlx_dequantize(&dq, w, sc, bi, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = xd, .has_value = true }, s));
    var dq_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dq_t);
    try mlx.check(mlx.mlx_transpose(&dq_t, dq, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_matmul(&out, x, dq_t, s));
    prefill_dq_gemm_engaged += 1;
    return out;
}

fn getWeightFmt(weights: *const Weights, buf: *[256]u8, comptime fmt: []const u8, prefix: []const u8) error{MissingWeight}!mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, .{prefix}) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
}

test "a missing weight is a load ERROR, not a process exit (issue #217)" {
    var w = model_mod.Weights.init(std.testing.allocator);
    defer w.deinit();
    var buf: [256]u8 = undefined;
    try std.testing.expectError(error.MissingWeight, getLayerWeight(&w, &buf, "model", 0, "mlp.w1.weight"));
    try std.testing.expectError(error.MissingWeight, getWeightFmt(&w, &buf, "{s}.norm.weight", "model"));
}

test "exl3 MTP switch_mlp binds restacked trellis/suh/svh under the head prefix" {
    const t = std.testing;
    const s = mlx.gpuStream();
    const put = struct {
        fn add(w: *Weights, name: []const u8, dtype: mlx.mlx_dtype, shape: []const c_int, st: mlx.mlx_stream) !void {
            var arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&arr, shape.ptr, @intCast(shape.len), dtype, st));
            try w.map.put(try w.allocator.dupe(u8, name), arr);
        }
    }.add;
    var w = Weights.init(t.allocator);
    defer w.deinit();
    const trellis = [_]c_int{ 4, 8, 8, 64 };
    const suh = [_]c_int{ 4, 128 };
    const svh = [_]c_int{ 4, 128 };
    for ([_][]const u8{ "gate", "up", "down" }) |proj| {
        var tbuf: [96]u8 = undefined;
        const tk = try std.fmt.bufPrint(&tbuf, "language_model.mtp.layers.0.mlp.switch_mlp.{s}_proj.trellis", .{proj});
        try put(&w, tk, .uint16, &trellis, s);
        const sk = try std.fmt.bufPrint(&tbuf, "language_model.mtp.layers.0.mlp.switch_mlp.{s}_proj.suh", .{proj});
        try put(&w, sk, .float16, &suh, s);
        const vk = try std.fmt.bufPrint(&tbuf, "language_model.mtp.layers.0.mlp.switch_mlp.{s}_proj.svh", .{proj});
        try put(&w, vk, .float16, &svh, s);
    }
    var buf: [256]u8 = undefined;
    const config = exl3SwitchBankConfig(.{ .n = 64 });
    const bank = try loadSwitchMlpBank(&w, &buf, "language_model.mtp", 0, true, &config);
    try t.expectEqual(@as(usize, 4), mlx.getShape(bank.gate_w).len);
    try t.expectEqual(@as(c_int, 4), mlx.getShape(bank.gate_w)[0]);
    try t.expectEqual(mlx.mlx_dtype.uint16, mlx.mlx_array_dtype(bank.gate_w));
    try t.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(bank.gate_s));
    try t.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(bank.gate_b));
    try t.expectEqualSlices(c_int, mlx.getShape(bank.gate_w), mlx.getShape(bank.up_w));
    try t.expectEqualSlices(c_int, mlx.getShape(bank.down_s), mlx.getShape(bank.gate_s));
}

test "exl3 MTP switch_mlp refuses a missing restacked tensor by MissingWeight" {
    const t = std.testing;
    const s = mlx.gpuStream();
    const put = struct {
        fn add(w: *Weights, name: []const u8, dtype: mlx.mlx_dtype, shape: []const c_int, st: mlx.mlx_stream) !void {
            var arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&arr, shape.ptr, @intCast(shape.len), dtype, st));
            try w.map.put(try w.allocator.dupe(u8, name), arr);
        }
    }.add;
    var w = Weights.init(t.allocator);
    defer w.deinit();
    const trellis = [_]c_int{ 4, 8, 8, 64 };
    const svh = [_]c_int{ 4, 128 };
    for ([_][]const u8{ "gate", "up", "down" }) |proj| {
        var tbuf: [96]u8 = undefined;
        const tk = try std.fmt.bufPrint(&tbuf, "language_model.mtp.layers.0.mlp.switch_mlp.{s}_proj.trellis", .{proj});
        try put(&w, tk, .uint16, &trellis, s);
        const vk = try std.fmt.bufPrint(&tbuf, "language_model.mtp.layers.0.mlp.switch_mlp.{s}_proj.svh", .{proj});
        try put(&w, vk, .float16, &svh, s);
    }
    var buf: [256]u8 = undefined;
    const config = exl3SwitchBankConfig(.{ .n = 64 });
    try t.expectError(error.MissingWeight, loadSwitchMlpBank(&w, &buf, "language_model.mtp", 0, true, &config));
}

/// The 4-expert 128x128 pack the synthetic switch_mlp binds answer to.
fn exl3SwitchBankConfig(rate: expert_exl3.Rate) ModelConfig {
    return .{
        .expert_layout = .exl3_k4,
        .expert_quant_rate = rate,
        .num_experts = 4,
        .hidden_size = 128,
        .moe_intermediate_size = 128,
    };
}

/// Bind an EXL3 switch_mlp bank whose three trellises carry `trellis`, against
/// a config that names `rate`.
fn bindExl3SwitchBank(st: mlx.mlx_stream, trellis: []const c_int, rate: expert_exl3.Rate) !SwitchMlpBank {
    var w = Weights.init(std.testing.allocator);
    defer w.deinit();
    const scales = [_]c_int{ 4, 128 };
    for ([_][]const u8{ "gate", "up", "down" }) |proj| {
        for ([_]struct { leaf: []const u8, dtype: mlx.mlx_dtype, shape: []const c_int }{
            .{ .leaf = "trellis", .dtype = .uint16, .shape = trellis },
            .{ .leaf = "suh", .dtype = .float16, .shape = &scales },
            .{ .leaf = "svh", .dtype = .float16, .shape = &scales },
        }) |part| {
            var tbuf: [96]u8 = undefined;
            const key = try std.fmt.bufPrint(&tbuf, "language_model.mtp.layers.0.mlp.switch_mlp.{s}_proj.{s}", .{ proj, part.leaf });
            var arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&arr, part.shape.ptr, @intCast(part.shape.len), part.dtype, st));
            try w.map.put(try w.allocator.dupe(u8, key), arr);
        }
    }
    var buf: [256]u8 = undefined;
    const config = exl3SwitchBankConfig(rate);
    return loadSwitchMlpBank(&w, &buf, "language_model.mtp", 0, true, &config);
}

test "exl3 a trellis the kernels cannot decode refuses at load" {
    const t = std.testing;
    const s = mlx.gpuStream();
    _ = try bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 48 }, .{ .n = 48 });
    _ = try bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 40 }, .{ .n = 40 });
    _ = try bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 44 }, .{ .n = 44 });
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 130 }, .{ .n = 130 }));
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 14 }, .{ .n = 14 }));
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 41 }, .{ .n = 41 }));
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 64 }, .{ .n = 64 }));
}

test "exl3 a trellis the config's rate or expert geometry does not name refuses at load" {
    const t = std.testing;
    const s = mlx.gpuStream();
    // The streaming bill prices n from the config, so a shard WIDER than the
    // config names must never load; a narrower one is over-billed, not wrong.
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 48 }, .{ .n = 40 }));
    _ = try bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 40 }, .{ .n = 48 });
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 8, 8, 8, 64 }, .{ .n = 64 }));
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 4, 4, 8, 64 }, .{ .n = 64 }));
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 16, 64 }, .{ .n = 64 }));
    _ = try bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 64 }, .{ .n = 64 });
}

/// A MiMo-shaped EXL3 MoE layer: hidden != inter, sigmoid routing with a
/// selection bias, no shared expert. Every expert holds the SAME bank and the
/// router is flat, so the normalized routed sum is one expert's own output
/// whichever top-k the GPU picks — that is the host oracle.
pub const MimoExl3MoeHarness = struct {
    xfm: Transformer,
    x: mlx.mlx_array,
    mw: MoeMlpWeights,
    owned: [10]mlx.mlx_array,
    gate_t: []u16,
    down_t: []u16,
    suh_h: []u16,
    svh_i: []u16,
    suh_i: []u16,
    svh_h: []u16,
    xf: []f32,
    dec: expert_exl3.Decode,
    rows: usize,

    const E: usize = 4;
    const hidden: usize = 128;
    const inter: usize = 256;

    pub fn init(alloc: std.mem.Allocator, s: mlx.mlx_stream, rate: expert_exl3.Rate, rows: usize, seed: u64, dec: expert_exl3.Decode) !MimoExl3MoeHarness {
        const packed_n = rate.halfwords();
        const gu_tile = (hidden / 16) * (inter / 16) * packed_n;
        const d_tile = (inter / 16) * (hidden / 16) * packed_n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const gate_t = try alloc.alloc(u16, E * gu_tile);
        const down_t = try alloc.alloc(u16, E * d_tile);
        for (gate_t[0..gu_tile]) |*v| v.* = @truncate(rnd.int(u32));
        for (down_t[0..d_tile]) |*v| v.* = @truncate(rnd.int(u32));
        for (1..E) |e| {
            @memcpy(gate_t[e * gu_tile ..][0..gu_tile], gate_t[0..gu_tile]);
            @memcpy(down_t[e * d_tile ..][0..d_tile], down_t[0..d_tile]);
        }
        const scale = struct {
            fn fill(bank: []u16, n: usize, r: std.Random) void {
                for (bank[0..n]) |*v| v.* = expert_exl3.f32ToF16Bits(if (r.boolean()) @as(f32, 0.9) else -0.9);
                var e: usize = 1;
                while (e * n < bank.len) : (e += 1) @memcpy(bank[e * n ..][0..n], bank[0..n]);
            }
        }.fill;
        const suh_h = try alloc.alloc(u16, E * hidden);
        const svh_i = try alloc.alloc(u16, E * inter);
        const suh_i = try alloc.alloc(u16, E * inter);
        const svh_h = try alloc.alloc(u16, E * hidden);
        scale(suh_h, hidden, rnd);
        scale(svh_i, inter, rnd);
        scale(suh_i, inter, rnd);
        scale(svh_h, hidden, rnd);
        const xh = try alloc.alloc(u16, rows * hidden);
        const xf = try alloc.alloc(f32, rows * hidden);
        for (xh, xf) |*b, *v| {
            b.* = expert_exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
            v.* = expert_exl3.f16BitsToF32(b.*);
        }
        // A flat router: every expert scores the same, so the normalized
        // weights sum to one whichever top-k the selection returns.
        const rw = try alloc.alloc(u16, hidden * E);
        @memset(rw, 0);
        const bias = try alloc.alloc(f32, E);
        @memset(bias, 0);
        const ci = struct {
            fn i(v: usize) c_int {
                return @intCast(v);
            }
        }.i;
        const gt = mlx.mlx_array_new_data(gate_t.ptr, &[_]c_int{ ci(E), ci(hidden / 16), ci(inter / 16), ci(packed_n) }, 4, .uint16);
        const dt = mlx.mlx_array_new_data(down_t.ptr, &[_]c_int{ ci(E), ci(inter / 16), ci(hidden / 16), ci(packed_n) }, 4, .uint16);
        const a_suh_h = mlx.mlx_array_new_data(suh_h.ptr, &[_]c_int{ ci(E), ci(hidden) }, 2, .float16);
        const a_svh_i = mlx.mlx_array_new_data(svh_i.ptr, &[_]c_int{ ci(E), ci(inter) }, 2, .float16);
        const a_suh_i = mlx.mlx_array_new_data(suh_i.ptr, &[_]c_int{ ci(E), ci(inter) }, 2, .float16);
        const a_svh_h = mlx.mlx_array_new_data(svh_h.ptr, &[_]c_int{ ci(E), ci(hidden) }, 2, .float16);
        const router = mlx.mlx_array_new_data(rw.ptr, &[_]c_int{ ci(hidden), ci(E) }, 2, .float16);
        const eb = mlx.mlx_array_new_data(bias.ptr, &[_]c_int{ci(E)}, 1, .float32);
        const x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ 1, ci(rows), ci(hidden) }, 3, .float16);
        const one = mlx.mlx_array_new_float(1.0);
        const none = mlx.mlx_array{ .ctx = null };
        var h: MimoExl3MoeHarness = .{
            .xfm = undefined,
            .x = x,
            .owned = .{ gt, dt, a_suh_h, a_svh_i, a_suh_i, a_svh_h, router, eb, x, one },
            .gate_t = gate_t,
            .down_t = down_t,
            .suh_h = suh_h,
            .svh_i = svh_i,
            .suh_i = suh_i,
            .svh_h = svh_h,
            .xf = xf,
            .dec = dec,
            .rows = rows,
            .mw = .{
                .router_w = router,
                .router_s = none,
                .router_b = none,
                .switch_gate_w = gt,
                .switch_gate_s = a_suh_h,
                .switch_gate_b = a_svh_i,
                .switch_up_w = gt,
                .switch_up_s = a_suh_h,
                .switch_up_b = a_svh_i,
                .switch_down_w = dt,
                .switch_down_s = a_suh_i,
                .switch_down_b = a_svh_h,
                .shared_gate_w = none,
                .shared_gate_s = none,
                .shared_gate_b = none,
                .shared_up_w = none,
                .shared_up_s = none,
                .shared_up_b = none,
                .shared_down_w = none,
                .shared_down_s = none,
                .shared_down_b = none,
                .expert_bias = eb,
                .route_norm = true,
                .route_scale = 1.0,
            },
        };
        h.xfm.s = s;
        h.xfm.config = .{
            .model_type = "mimo_v2",
            .expert_layout = .exl3_k4,
            .expert_quant_rate = rate,
            .expert_quant_codebook = dec.codebook,
            .expert_quant_window = dec.window,
            .expert_quant_table = dec.table,
            .num_experts = @intCast(E),
            .num_experts_per_tok = 2,
            .hidden_size = @intCast(hidden),
            .moe_intermediate_size = @intCast(inter),
            .quant_bits = 0,
            .quant_group_size = 64,
            .quant_mode = .affine,
            .hidden_act = .silu,
            .moe_sigmoid_router = true,
            .moe_n_group = 1,
            .moe_topk_group = 1,
            .moe_route_norm = true,
        };
        h.xfm.one = one;
        h.xfm.bits_cache = .{};
        h.xfm.compiled_moe_routing = null;
        h.xfm.compiled_gelu = null;
        h.xfm.compiled_geglu = null;
        h.xfm.cost_trace_active = false;
        h.xfm.expert_stream = null;
        for (h.owned) |a| try mlx.check(mlx.mlx_array_eval(a));
        return h;
    }

    /// One expert's SwiGLU on the host, at the rate the trellis names.
    fn hostRow(self: *MimoExl3MoeHarness, alloc: std.mem.Allocator, rate: expert_exl3.Rate, row: usize, out: []f32) !void {
        const x = self.xf[row * hidden ..][0..hidden];
        const t_hidden = try alloc.alloc(f32, hidden);
        defer alloc.free(t_hidden);
        const t_inter = try alloc.alloc(f32, inter);
        defer alloc.free(t_inter);
        const inner = try alloc.alloc(f32, inter);
        defer alloc.free(inner);
        const gate_y = try alloc.alloc(f32, inter);
        defer alloc.free(gate_y);
        const up_y = try alloc.alloc(f32, inter);
        defer alloc.free(up_y);
        const gu_tile = (hidden / 16) * (inter / 16) * rate.halfwords();
        const d_tile = (inter / 16) * (hidden / 16) * rate.halfwords();
        expert_exl3.project(x, self.gate_t[0..gu_tile], self.suh_h[0..hidden], self.svh_i[0..inter], hidden, inter, rate, self.dec, t_hidden, inner, gate_y);
        expert_exl3.project(x, self.gate_t[0..gu_tile], self.suh_h[0..hidden], self.svh_i[0..inter], hidden, inter, rate, self.dec, t_hidden, inner, up_y);
        for (gate_y, up_y) |*g, u| g.* = (g.* / (1.0 + @exp(-g.*))) * u;
        expert_exl3.project(gate_y, self.down_t[0..d_tile], self.suh_i[0..inter], self.svh_h[0..hidden], inter, hidden, rate, self.dec, t_inter, inner[0..hidden], out);
    }

    /// One forward through the engine, scored against this harness's own
    /// codebook on the host.
    fn expectMatchesHost(self: *MimoExl3MoeHarness, alloc: std.mem.Allocator, rate: expert_exl3.Rate) !void {
        const y = try self.xfm.moeMLP(self.x, &self.mw);
        defer _ = mlx.mlx_array_free(y);
        var c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_contiguous(&c, y, false, self.xfm.s));
        try mlx.check(mlx.mlx_array_eval(c));
        const got = mlx.mlx_array_data_float16(c) orelse return error.F16Unreadable;
        const want = try alloc.alloc(f32, hidden);
        defer alloc.free(want);
        var ss: f64 = 0;
        var ref: f64 = 0;
        for (0..self.rows) |r| {
            try self.hostRow(alloc, rate, r, want);
            for (want, 0..) |w, i| {
                const a: f64 = @floatCast(got[r * hidden + i]);
                ss += (a - w) * (a - w);
                ref += @as(f64, w) * @as(f64, w);
            }
        }
        const rel = @sqrt(ss / @max(ref, 1e-20));
        if (!(rel < 0.01)) {
            std.debug.print("mimo exl3 forward codebook={s} window={d} rel_rms={d:.6}\n", .{ @tagName(self.dec.codebook), self.dec.window.bits(), rel });
            return error.TestExpectedEqual;
        }
    }

    pub fn deinit(self: *MimoExl3MoeHarness) void {
        for (self.owned) |a| _ = mlx.mlx_array_free(a);
    }
};

fn mimoExl3ForwardMatchesHost(rate: expert_exl3.Rate, rows: usize, seed: u64) !void {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var h = try MimoExl3MoeHarness.init(alloc, s, rate, rows, seed, .mcg);
    defer h.deinit();
    try h.expectMatchesHost(alloc, rate);
}

/// Two EXL3 packs on different codebooks, resident together and dispatching
/// alternately: every forward must decode with ITS OWN model's codebook.
fn exl3CodebookFollowsModel(rows: usize) !void {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const rate: expert_exl3.Rate = .{ .n = 40 };
    var mul1 = try MimoExl3MoeHarness.init(alloc, s, rate, rows, 101, .mul1);
    defer mul1.deinit();
    var mcg = try MimoExl3MoeHarness.init(alloc, s, rate, rows, 103, .mcg);
    defer mcg.deinit();
    for (0..2) |_| {
        try mul1.expectMatchesHost(alloc, rate);
        try mcg.expectMatchesHost(alloc, rate);
    }
}

test "exl3 decode rows follow their own model's codebook with two packs resident" {
    try exl3CodebookFollowsModel(4);
}

test "exl3 prefill rows follow their own model's codebook with two packs resident" {
    try exl3CodebookFollowsModel(expert_exl3_kernels.DECODE_ROWS_MAX * 2);
}

test "mimo_v2 EXL3 resident MoE decode rows match the host SwiGLU oracle at K2.5 MCG" {
    try mimoExl3ForwardMatchesHost(.{ .n = 40 }, 4, 71);
}

test "mimo_v2 EXL3 resident MoE prefill rows match the host SwiGLU oracle at K2.5 MCG" {
    try mimoExl3ForwardMatchesHost(.{ .n = 40 }, expert_exl3_kernels.DECODE_ROWS_MAX * 2, 73);
}

fn mimoExl3DistinctRouting(rows: usize) !void {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const H = MimoExl3MoeHarness.hidden;
    const I = MimoExl3MoeHarness.inter;
    const E = MimoExl3MoeHarness.E;
    const rate: expert_exl3.Rate = .{ .n = 40 };
    var h = try MimoExl3MoeHarness.init(alloc, s, rate, rows, 191, .{ .codebook = .mcg, .window = .w12 });
    defer h.deinit();
    var prng = std.Random.DefaultPrng.init(193);
    const rnd = prng.random();
    for (h.gate_t) |*v| v.* = rnd.int(u16);
    for (h.down_t) |*v| v.* = rnd.int(u16);
    const router = try alloc.alloc(f32, H * E);
    for (router) |*v| {
        const value = (rnd.float(f32) * 2 - 1) * 0.2;
        const bits: u32 = @bitCast(value);
        v.* = @bitCast((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
    }
    const bias = [_]f32{ 0.03, -0.08, 0.11, -0.02 };
    const replace = struct {
        fn array(dst: *mlx.mlx_array, src: mlx.mlx_array) void {
            _ = mlx.mlx_array_free(dst.*);
            dst.* = src;
        }
    }.array;
    replace(&h.owned[0], mlx.mlx_array_new_data(h.gate_t.ptr, &.{ E, H / 16, I / 16, 40 }, 4, .uint16));
    replace(&h.owned[1], mlx.mlx_array_new_data(h.down_t.ptr, &.{ E, I / 16, H / 16, 40 }, 4, .uint16));
    replace(&h.owned[6], mlx.mlx_array_new_data(router.ptr, &.{ H, E }, 2, .float32));
    replace(&h.owned[7], mlx.mlx_array_new_data(&bias, &.{E}, 1, .float32));
    var xb = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&xb, h.x, .bfloat16, s));
    replace(&h.owned[8], xb);
    h.x = xb;
    h.mw.switch_gate_w = h.owned[0];
    h.mw.switch_up_w = h.owned[0];
    h.mw.switch_down_w = h.owned[1];
    h.mw.router_w = h.owned[6];
    h.mw.expert_bias = h.owned[7];
    var x32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x32);
    try mlx.check(mlx.mlx_astype(&x32, h.x, .float32, s));
    try mlx.check(mlx.mlx_array_eval(x32));
    @memcpy(h.xf, (mlx.mlx_array_data_float32(x32) orelse return error.F32Unreadable)[0..h.xf.len]);
    const logits = try h.xfm.mimoRouterLogits(h.x, &h.mw);
    defer _ = mlx.mlx_array_free(logits);
    const routed = try h.xfm.computeMimoRouting(logits, h.mw.expert_bias.?);
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.norm_scores);
    try t.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(routed.norm_scores));
    var ids32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids32);
    try mlx.check(mlx.mlx_astype(&ids32, routed.inds, .uint32, s));
    try mlx.check(mlx.mlx_array_eval(ids32));
    try mlx.check(mlx.mlx_array_eval(routed.norm_scores));
    const ids = mlx.mlx_array_data_uint32(ids32) orelse return error.ExpertIdsUnreadable;
    const scores = mlx.mlx_array_data_float32(routed.norm_scores) orelse return error.F32Unreadable;
    const y = try h.xfm.moeMLP(h.x, &h.mw);
    defer _ = mlx.mlx_array_free(y);
    try t.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(y));
    var y32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y32);
    try mlx.check(mlx.mlx_astype(&y32, y, .float32, s));
    try mlx.check(mlx.mlx_array_eval(y32));
    const got = mlx.mlx_array_data_float32(y32) orelse return error.F32Unreadable;
    var err2: f64 = 0;
    var ref2: f64 = 0;
    var seen: u32 = 0;
    for (0..rows) |r| {
        var probabilities: [E]f32 = undefined;
        for (0..E) |e| {
            var dot: f32 = 0;
            for (0..H) |d| dot += h.xf[r * H + d] * router[d * E + e];
            probabilities[e] = 1 / (1 + @exp(-dot));
        }
        const den = probabilities[ids[r * 2]] + probabilities[ids[r * 2 + 1]];
        var want: [H]f32 = @splat(0);
        for (0..2) |k| {
            const e = ids[r * 2 + k];
            try t.expect(e < E);
            seen |= @as(u32, 1) << @intCast(e);
            const score = probabilities[e] / den;
            try t.expectApproxEqAbs(score, scores[r * 2 + k], 1e-6);
            for (0..E) |other| {
                if (other == ids[r * 2] or other == ids[r * 2 + 1]) continue;
                try t.expect(probabilities[e] + bias[e] >= probabilities[other] + bias[other]);
            }
            var oracle = h;
            oracle.gate_t = h.gate_t[e * (h.gate_t.len / E) ..];
            oracle.down_t = h.down_t[e * (h.down_t.len / E) ..];
            oracle.suh_h = h.suh_h[e * H ..];
            oracle.svh_i = h.svh_i[e * I ..];
            oracle.suh_i = h.suh_i[e * I ..];
            oracle.svh_h = h.svh_h[e * H ..];
            var out: [H]f32 = undefined;
            try oracle.hostRow(alloc, rate, r, &out);
            for (&want, out) |*w, value| w.* += value * score;
        }
        for (want, 0..) |w, d| {
            const actual = got[r * H + d];
            try t.expect(std.math.isFinite(actual) and std.math.isFinite(w));
            const delta: f64 = actual - w;
            err2 += delta * delta;
            ref2 += @as(f64, w) * w;
        }
    }
    if (rows > 1) try t.expect(@popCount(seen) > 2);
    const relative = @sqrt(err2 / @max(ref2, 1e-20));
    if (diagEnvOn("SUSHI_EXL3_PARITY_TRACE"))
        std.debug.print("mimo distinct routed MoE rows={d} relative RMS={d:.8}\n", .{ rows, relative });
    if (!(relative < 0.01)) {
        std.debug.print("mimo distinct routed MoE rows={d} relative RMS={d:.8}\n", .{ rows, relative });
        return error.TestExpectedEqual;
    }
}

test "mimo_v2 EXL3 distinct routed experts match host with BF16 inputs" {
    for ([_]usize{ 1, 4, 14, 17, 33 }) |rows| try mimoExl3DistinctRouting(rows);
}

/// `{prefix}.{base}.{suffix}` with a RUNTIME base — the embedding table's name
/// is the checkpoint's (embed_tokens / word_embeddings / embed / embeddings)
/// and its three lookups must agree, which a comptime format can't express
/// without repeating the arch chain per suffix.
fn getNamedWeight(weights: *const Weights, buf: *[256]u8, prefix: []const u8, base: []const u8, suffix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.{s}.{s}", .{ prefix, base, suffix }) catch unreachable;
    return weights.get(name);
}

fn getWeightFmtOpt(weights: *const Weights, buf: *[256]u8, comptime fmt: []const u8, prefix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, .{prefix}) catch unreachable;
    return weights.get(name);
}

fn getLayerWeightOpt(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, suffix }) catch unreachable;
    return weights.get(name);
}

fn getLayerWeight(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8) error{MissingWeight}!mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, suffix }) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
}

const SwitchMlpBank = struct {
    ragged: ?sushi_exl3.Ragged = null,
    gate_w: mlx.mlx_array,
    gate_s: mlx.mlx_array,
    gate_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
};

/// A trellis the EXL3 kernels can decode AND the memory plan can price:
/// `[E, in/16, out/16, n]` at a rate the CONFIG's bill covers. Decode reads n
/// off the tensor and a layer may carry fewer bits than the pack's widest — a
/// wider one under-bills, and an under-bill here is an uncatchable Metal OOM.
pub fn loadSwitchMlpBank(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, exl3: bool, config: *const ModelConfig) !SwitchMlpBank {
    if (exl3) {
        var plan = sushi_exl3.GroupLayout.init(config.hidden_size, config.moe_intermediate_size, config.expert_quant_rate);
        plan.rates = config.expert_quant_rates;
        var groups: [sushi_exl3.group_layout.max_groups]sushi_exl3.Bank = undefined;
        var start_buf: [256]u8 = undefined;
        const start = try std.fmt.bufPrint(&start_buf, "{s}.layers.{d}.mlp.switch_mlp.", .{ prefix, layer });
        var it = weights.map.iterator();
        while (it.next()) |entry| {
            if (!std.mem.startsWith(u8, entry.key_ptr.*, start)) continue;
            const arr = entry.value_ptr.*;
            const shape = mlx.getShape(arr);
            if (shape.len > 4) return error.Exl3TrellisGeometry;
            var dims: [4]u64 = undefined;
            for (shape, 0..) |d, i| {
                if (d <= 0) return error.Exl3TrellisGeometry;
                dims[i] = @intCast(d);
            }
            const name = plan.add(entry.key_ptr.*[start.len..], dims[0..shape.len], switch (mlx.mlx_array_dtype(arr)) {
                .uint16 => .u16,
                .float16 => .f16,
                else => .other,
            }) catch |err| {
                if (err == error.Exl3GroupGeometry) return error.Exl3TrellisGeometry;
                return err;
            };
            const bank = &groups[name.group];
            const proj = switch (name.projection) {
                0 => &bank.gate,
                1 => &bank.up,
                else => &bank.down,
            };
            switch (name.part) {
                0 => proj.trellis = arr,
                1 => proj.suh = arr,
                else => proj.svh = arr,
            }
        }
        const router = getLayerWeightOpt(weights, buf, prefix, layer, "mlp.gate.weight");
        var experts = config.num_experts;
        if (router) |r| {
            const sh = mlx.getShape(r);
            if (sh.len != 2 or sh[0] <= 0) return error.Exl3RouterWidthMismatch;
            experts = @intCast(sh[0]);
        }
        try plan.finish(experts, config.num_experts_per_tok, config.expert_streaming, config.num_experts);
        if (config.moe_n_group > 1 and (plan.grouped.? or experts != config.num_experts)) return error.Exl3RouterGroupsUnsupported;
        var ragged: ?sushi_exl3.Ragged = null;
        if (plan.grouped.?) {
            ragged = try sushi_exl3.Ragged.init(mlx.gpuStream(), groups[0..plan.count]);
            errdefer ragged.?.deinit();
            // The pack's group tensors become views of the ragged storage, so the loaded originals
            // are freed once it is evaluated and the layer's bytes stay resident once.
            var views = weights.map.iterator();
            while (views.next()) |entry| {
                if (!std.mem.startsWith(u8, entry.key_ptr.*, start)) continue;
                const name = try sushi_exl3.group_layout.Name.parse(entry.key_ptr.*[start.len..]);
                const bank = ragged.?.views[name.group];
                const proj = switch (name.projection) {
                    0 => bank.gate,
                    1 => bank.up,
                    else => bank.down,
                };
                var view = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&view, switch (name.part) {
                    0 => proj.trellis,
                    1 => proj.suh,
                    else => proj.svh,
                }));
                _ = mlx.mlx_array_free(entry.value_ptr.*);
                entry.value_ptr.* = view;
            }
        }
        const g = if (ragged) |rb| rb.views[0] else groups[0];
        return .{
            .gate_w = g.gate.trellis,
            .gate_s = g.gate.suh,
            .gate_b = g.gate.svh,
            .up_w = g.up.trellis,
            .up_s = g.up.suh,
            .up_b = g.up.svh,
            .down_w = g.down.trellis,
            .down_s = g.down.suh,
            .down_b = g.down.svh,
            .ragged = ragged,
        };
    }
    return .{
        .gate_w = try getLayerWeight(weights, buf, prefix, layer, "mlp.switch_mlp.gate_proj.weight"),
        .gate_s = getLayerWeightOpt(weights, buf, prefix, layer, "mlp.switch_mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
        .gate_b = getLayerWeightOpt(weights, buf, prefix, layer, "mlp.switch_mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
        .up_w = try getLayerWeight(weights, buf, prefix, layer, "mlp.switch_mlp.up_proj.weight"),
        .up_s = getLayerWeightOpt(weights, buf, prefix, layer, "mlp.switch_mlp.up_proj.scales") orelse mlx.mlx_array_new(),
        .up_b = getLayerWeightOpt(weights, buf, prefix, layer, "mlp.switch_mlp.up_proj.biases") orelse mlx.mlx_array_new(),
        .down_w = try getLayerWeight(weights, buf, prefix, layer, "mlp.switch_mlp.down_proj.weight"),
        .down_s = getLayerWeightOpt(weights, buf, prefix, layer, "mlp.switch_mlp.down_proj.scales") orelse mlx.mlx_array_new(),
        .down_b = getLayerWeightOpt(weights, buf, prefix, layer, "mlp.switch_mlp.down_proj.biases") orelse mlx.mlx_array_new(),
    };
}


/// Fetch a quantization scale/bias tensor, tolerant of dense bf16 models.
/// quant_bits == 0 ⇒ dense bf16 (no .scales/.biases exist anywhere) ⇒ return a
/// null-ctx array, which downstream code (qmatmulBits, gatherExpertMm, append
/// guards) reads as "this weight is plain bf16". quant_bits > 0 ⇒ a genuinely
/// quantized model, so fetch mandatorily — a missing scale stays a clear
/// MISSING WEIGHT error rather than silently degrading to a dense path.
fn getLayerScaleOrEmpty(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8, quant_bits: u32) error{MissingWeight}!mlx.mlx_array {
    if (quant_bits == 0) return mlx.mlx_array_new();
    return try getLayerWeight(weights, buf, prefix, layer, suffix);
}

/// Split a native fused QKV projection whose value width is independent of its
/// key width. MiMo's `[q | k | v]` rows are `[nq*hd | nkv*hd | nkv*vhd]`;
/// `splitFusedQkvRows` remains the symmetric compatibility wrapper.
fn splitFusedQkvAsymRows(
    arr: mlx.mlx_array,
    q_rows: u32,
    k_rows: u32,
    v_rows: u32,
    transpose: bool,
    owned: *std.ArrayList(mlx.mlx_array),
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
) ![3]mlx.mlx_array {
    if (arr.ctx == null) return .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    const shape = mlx.getShape(arr);
    const total_rows = std.math.add(u32, q_rows, std.math.add(u32, k_rows, v_rows) catch return error.BadFusedQkvShape) catch return error.BadFusedQkvShape;
    if (shape.len != 2 or shape[0] != @as(c_int, @intCast(total_rows))) return error.BadFusedQkvShape;
    const bounds = [_]c_int{ 0, @intCast(q_rows), @intCast(q_rows + k_rows), shape[0] };
    var out: [3]mlx.mlx_array = undefined;
    for (0..3) |i| {
        const start = [_]c_int{ bounds[i], 0 };
        const stop = [_]c_int{ bounds[i + 1], shape[1] };
        const strides = [_]c_int{ 1, 1 };
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, arr, &start, 2, &stop, 2, &strides, 2, s));
        var part = mlx.mlx_array_new();
        if (transpose) {
            part = try transposeBf16Weight(view, s);
        } else {
            try mlx.check(mlx.mlx_contiguous(&part, view, false, s));
        }
        try owned.append(allocator, part);
        out[i] = part;
    }
    return out;
}

fn getLayerBias(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8, config: *const ModelConfig) error{MissingWeight}!mlx.mlx_array {
    if (config.quant_bits == 0) return mlx.mlx_array_new();
    if (config.quant_mode.hasBiases()) return try getLayerWeight(weights, buf, prefix, layer, suffix);
    return getLayerWeightOpt(weights, buf, prefix, layer, suffix) orelse mlx.mlx_array_new();
}

/// A 0-d scalar in `dt` (a bare f32 scalar array promotes bf16 operands).
pub fn scalarOf(val: f32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = mlx.mlx_array_new_float(val);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, raw, dt, s));
    return out;
}

pub fn bf16Scalar(val: f32, s: mlx.mlx_stream) mlx.mlx_array {
    const f32_arr = mlx.mlx_array_new_float(val);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s);
    return bf16_arr;
}

pub fn initModuleShell(allocator: std.mem.Allocator, config: ModelConfig, s: mlx.mlx_stream) !Transformer {
    const cache = try KVCache.init(allocator, 0);
    return .{
        .config = config,
        .cache = cache,
        .s = s,
        .allocator = allocator,
        .emb_w = mlx.mlx_array_new(),
        .emb_s = mlx.mlx_array_new(),
        .emb_b = mlx.mlx_array_new(),
        .emb_scale = null,
        .final_norm = mlx.mlx_array_new(),
        .lm_head_w = mlx.mlx_array_new(),
        .lm_head_s = mlx.mlx_array_new(),
        .lm_head_b = mlx.mlx_array_new(),
        .owns_lm_head = false,
        .owns_norms = false,
        .gelu_coeff = bf16Scalar(0.7978845608028654, s),
        .gelu_inner = bf16Scalar(0.044715, s),
        .half = bf16Scalar(0.5, s),
        .one = bf16Scalar(1.0, s),
        .rms_eps_arr = mlx.mlx_array_new_float(config.rms_norm_eps),
        .three = bf16Scalar(3.0, s),
        .neg_one = null,
        .moe_layers = null,
        .ssm_entries = null,
        .moe_seq_offset = 0,
    };
}

fn addOne(arr: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const one = bf16Scalar(1.0, s);
    defer _ = mlx.mlx_array_free(one);
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&result, one, arr, s));
    return result;
}

// ── Tests ──

const testing = std.testing;
const expectError = @import("test_expect.zig").expectError;

const kv_append_kv_h: c_int = 2;
const kv_append_hd: c_int = 256;

pub fn kvAppendPacked(n: c_int, seq: c_int, seed: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const nn: usize = @intCast(n * kv_append_kv_h * seq * kv_append_hd);
    const buf = try testing.allocator.alloc(f32, nn);
    defer testing.allocator.free(buf);
    var st: u32 = seed;
    for (buf) |*x| {
        st = st *% 1664525 +% 1013904223;
        x.* = @as(f32, @floatFromInt(st >> 16)) / 65536.0 - 0.5;
    }
    const shape = [_]c_int{ n, kv_append_kv_h, seq, kv_append_hd };
    const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f32_arr, .bfloat16, s));
    return out;
}

pub fn kvAppendPrefill(cache: *KVCache, tokens: usize, seed: u32, s: mlx.mlx_stream) !void {
    const k = try kvAppendPacked(1, @intCast(tokens), seed, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try kvAppendPacked(1, @intCast(tokens), seed ^ 0x9E3779B9, s);
    defer _ = mlx.mlx_array_free(v);
    var dv = try cache.update(0, k, v, s, 0);
    dv.deinit();
}

pub fn kvAppendOperandsEqual(a: *const KVCacheEntry, b: *const KVCacheEntry, s: mlx.mlx_stream) !void {
    try testing.expectEqual(a.offset, b.offset);
    try testing.expect(try qsaArraysAllEqual(a.keys, b.keys, s));
    try testing.expect(try qsaArraysAllEqual(a.values, b.values, s));
    try testing.expect(try qsaArraysAllEqual(a.keys_scales, b.keys_scales, s));
    try testing.expect(try qsaArraysAllEqual(a.keys_biases, b.keys_biases, s));
    try testing.expect(try qsaArraysAllEqual(a.values_scales, b.values_scales, s));
    try testing.expect(try qsaArraysAllEqual(a.values_biases, b.values_biases, s));
}

pub fn kvAppendDummyAttn(cache: *KVCache, s: mlx.mlx_stream) !mlx.mlx_array {
    var dv = try cache.denseView(0, s);
    defer dv.deinit();
    const ksh = mlx.getShape(dv.k);
    const q_shape = [_]c_int{ ksh[0], ksh[1], 1, ksh[3] };
    var q = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_ones(&q, &q_shape, 4, .bfloat16, s));
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kv_append_hd)));
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &out,
        q,
        dv.k,
        dv.v,
        scale,
        "",
        none,
        .{ .ctx = null },
        false,
        s,
    ));
    _ = mlx.mlx_array_free(q);
    return out;
}

// Bar: after the conv, freeing the chunk input leaves only the tail resident.

pub fn conv1dExpectBitEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !void {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var ok: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&ok, eq));
    try std.testing.expect(ok);
}

// Bar: output and stored state byte-equal with the copy on and off.

// Bar: every layer's tail is copied and named in a cadence eval inside the loop (the peak itself is measured live, not on a tiny fixture).

test "affineParamsFromGeometry: exact per-weight solve for off-config sidecar quants" {
    // Real geometries from the stamsam 35B-A3B MTP sidecar over a 4-bit/gs-64
    // affine trunk: q_proj 5-bit/gs-128, v_proj 6-bit/gs-128, shared expert
    // 8-bit/gs-128, switch experts 4-bit/gs-64. The old config-gs detection
    // (detectQuantBits with gs 64) returns garbage for the gs-128 tensors.
    const s = mlx.gpuStream();
    const H = 2048; // sidecar hidden

    const mk = struct {
        fn arr(shape: []const c_int, dt: mlx.mlx_dtype, st: mlx.mlx_stream) !mlx.mlx_array {
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&a, shape.ptr, shape.len, dt, st));
            return a;
        }
    };

    // q_proj: w [8192, 320] u32, scales [8192, 16] bf16 → 5-bit / gs 128.
    const wq = try mk.arr(&.{ 8192, 320 }, .uint32, s);
    defer _ = mlx.mlx_array_free(wq);
    const sq = try mk.arr(&.{ 8192, 16 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(sq);
    const qp_q = affineParamsFromGeometry(wq, sq, H) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u32, 5), qp_q.bits);
    try testing.expectEqual(@as(u32, 128), qp_q.group_size);

    // v_proj: w [512, 384] u32, scales [512, 16] bf16 → 6-bit / gs 128.
    const wv = try mk.arr(&.{ 512, 384 }, .uint32, s);
    defer _ = mlx.mlx_array_free(wv);
    const sv = try mk.arr(&.{ 512, 16 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(sv);
    const qp_v = affineParamsFromGeometry(wv, sv, H) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u32, 6), qp_v.bits);
    try testing.expectEqual(@as(u32, 128), qp_v.group_size);

    // Switch experts (3D): w [256, 512, 256] u32, scales [256, 512, 32] bf16
    // → 4-bit / gs 64 (trunk-standard — must still solve).
    const we = try mk.arr(&.{ 256, 512, 256 }, .uint32, s);
    defer _ = mlx.mlx_array_free(we);
    const se = try mk.arr(&.{ 256, 512, 32 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(se);
    const qp_e = affineParamsFromGeometry(we, se, H) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u32, 4), qp_e.bits);
    try testing.expectEqual(@as(u32, 64), qp_e.group_size);

    // Degenerate: bits wouldn't divide exactly → null (caller falls back).
    const qp_bad = affineParamsFromGeometry(wq, sq, 3000);
    try testing.expect(qp_bad == null);

    // And through computeQuantParams on an AFFINE gs-64 config: the hinted
    // exact solve must WIN over the config group size (pre-fix this returned
    // bits 10 / gs 64 for the q_proj geometry).
    var cfg = ModelConfig{};
    cfg.quant_bits = 4;
    cfg.quant_group_size = 64;
    cfg.quant_mode = .affine;
    const qp_via = computeQuantParams(&cfg, wq, sq, H);
    try testing.expectEqual(@as(u32, 5), qp_via.bits);
    try testing.expectEqual(@as(u32, 128), qp_via.group_size);
}

test "computeQuantParams resolves an fp override INSIDE the fp family (mxfp8 embed under mxfp4)" {
    // A checkpoint can declare a model-wide mxfp4/gs32 AND a
    // per-tensor override: `"model.embed_tokens": {bits 8, group_size 32,
    // mode "mxfp8"}`. Both modes carry uint8 (E8M0) scales, so the
    // scales-dtype rule cannot tell them apart and the embedding resolved at
    // the model-wide 4 bits — `mlx_dequantize` then rejected the shapes
    // ("matrix of shape (1,512) and scales of shape (1,64)"), and an MLX error
    // is not a Zig error: the first request killed the server.
    //
    // The packed geometry states the answer exactly, the same way the affine
    // path already solves off-config sidecar quants.
    const s = mlx.gpuStream();
    const H = 2048; // hidden

    const mk = struct {
        fn arr(shape: []const c_int, dt: mlx.mlx_dtype, st: mlx.mlx_stream) !mlx.mlx_array {
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&a, shape.ptr, shape.len, dt, st));
            return a;
        }
    };

    var cfg = ModelConfig{};
    cfg.quant_bits = 4;
    cfg.quant_group_size = 32;
    cfg.quant_mode = .mxfp4;

    // embed_tokens: w [V, 512] u32 + scales [V, 64] u8 → 2048 values/row at
    // 8 bits, gs 32. (Real shapes; V shrunk — only the last axis is read.)
    const we = try mk.arr(&.{ 1024, 512 }, .uint32, s);
    defer _ = mlx.mlx_array_free(we);
    const se = try mk.arr(&.{ 1024, 64 }, .uint8, s);
    defer _ = mlx.mlx_array_free(se);
    const qp_e = computeQuantParams(&cfg, we, se, H);
    try testing.expectEqual(QuantMode.mxfp8, qp_e.mode);
    try testing.expectEqual(@as(u32, 8), qp_e.bits);
    try testing.expectEqual(@as(u32, 32), qp_e.group_size);

    // An ordinary mxfp4 linear in the SAME model: w [2048, 256] + [2048, 64].
    // Unchanged — the config already describes it.
    const wl = try mk.arr(&.{ 2048, 256 }, .uint32, s);
    defer _ = mlx.mlx_array_free(wl);
    const sl = try mk.arr(&.{ 2048, 64 }, .uint8, s);
    defer _ = mlx.mlx_array_free(sl);
    const qp_l = computeQuantParams(&cfg, wl, sl, H);
    try testing.expectEqual(QuantMode.mxfp4, qp_l.mode);
    try testing.expectEqual(@as(u32, 4), qp_l.bits);
    try testing.expectEqual(@as(u32, 32), qp_l.group_size);

    // A geometry that maps to no fp format we serve is NOT invented into one:
    // the config stands, and the load fails honestly if it really disagrees.
    const s_odd = try mk.arr(&.{ 2048, 21 }, .uint8, s);
    defer _ = mlx.mlx_array_free(s_odd);
    const qp_odd = computeQuantParams(&cfg, wl, s_odd, H);
    try testing.expectEqual(QuantMode.mxfp4, qp_odd.mode);
    try testing.expectEqual(@as(u32, 4), qp_odd.bits);
    try testing.expectEqual(@as(u32, 32), qp_odd.group_size);

    // No inner-dim hint ⇒ nothing to solve against; config stands (every
    // non-hinted fp call site keeps its pre-fix behavior).
    const qp_nohint = computeQuantParams(&cfg, we, se, null);
    try testing.expectEqual(QuantMode.mxfp4, qp_nohint.mode);
    try testing.expectEqual(@as(u32, 4), qp_nohint.bits);
}

test "computeQuantParams resolves mixed mxfp4 + affine-override weights" {
    // Mixed shape: most tensors mxfp4 (uint8 fp8 scales, gs 32) with an
    // affine override on one linear
    // 8-bit/gs64 (bf16 scales + biases). Detection key: scales dtype, then
    // the activation inner dim to pin (bits, group_size) for the override.
    const s = mlx.gpuStream();
    const IN = 2560;
    const OUT = 8;

    var cfg = ModelConfig{};
    cfg.quant_bits = 4;
    cfg.quant_group_size = 32;
    cfg.quant_mode = .mxfp4;

    // mxfp4 tensor: w [OUT, IN*4/32] u32, scales [OUT, IN/32] u8.
    var w_nv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_nv);
    var sc_nv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc_nv);
    const w_nv_sh = [_]c_int{ OUT, IN * 4 / 32 };
    const sc_nv_sh = [_]c_int{ OUT, IN / 32 };
    try mlx.check(mlx.mlx_zeros(&w_nv, &w_nv_sh, 2, .uint32, s));
    try mlx.check(mlx.mlx_zeros(&sc_nv, &sc_nv_sh, 2, .uint8, s));
    const qp_nv = computeQuantParams(&cfg, w_nv, sc_nv, null);
    try testing.expectEqual(QuantMode.mxfp4, qp_nv.mode);
    try testing.expectEqual(@as(u32, 4), qp_nv.bits);
    try testing.expectEqual(@as(u32, 32), qp_nv.group_size);

    // Affine 8-bit/gs64 override: w [OUT, IN*8/32] u32, scales [OUT, IN/64] bf16.
    var w_af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_af);
    var sc_af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc_af);
    const w_af_sh = [_]c_int{ OUT, IN * 8 / 32 };
    const sc_af_sh = [_]c_int{ OUT, IN / 64 };
    try mlx.check(mlx.mlx_zeros(&w_af, &w_af_sh, 2, .uint32, s));
    try mlx.check(mlx.mlx_zeros(&sc_af, &sc_af_sh, 2, .bfloat16, s));

    // With the activation inner-dim hint: exact.
    const qp_af = computeQuantParams(&cfg, w_af, sc_af, IN);
    try testing.expectEqual(QuantMode.affine, qp_af.mode);
    try testing.expectEqual(@as(u32, 8), qp_af.bits);
    try testing.expectEqual(@as(u32, 64), qp_af.group_size);

    // Without a hint: falls back to mlx-lm's override default gs=64.
    const qp_af_nohint = computeQuantParams(&cfg, w_af, sc_af, null);
    try testing.expectEqual(QuantMode.affine, qp_af_nohint.mode);
    try testing.expectEqual(@as(u32, 8), qp_af_nohint.bits);
    try testing.expectEqual(@as(u32, 64), qp_af_nohint.group_size);

    // Plain affine model (config mode affine, bf16 scales): unchanged
    // detect-against-config-gs behavior.
    var cfg_affine = ModelConfig{};
    cfg_affine.quant_bits = 4;
    cfg_affine.quant_group_size = 64;
    cfg_affine.quant_mode = .affine;
    const qp_plain = computeQuantParams(&cfg_affine, w_af, sc_af, null);
    try testing.expectEqual(QuantMode.affine, qp_plain.mode);
    try testing.expectEqual(@as(u32, 8), qp_plain.bits);
    try testing.expectEqual(@as(u32, 64), qp_plain.group_size);
}

test "QuantParamsCache put/lookup round-trip" {
    var cache: BitsCache = .{};
    const s = mlx.gpuStream();

    // Three real arrays with distinct ctx pointers — the cache keys.
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    const shape = [_]c_int{8};
    try mlx.check(mlx.mlx_zeros(&a, &shape, 1, .bfloat16, s));
    try mlx.check(mlx.mlx_zeros(&b, &shape, 1, .bfloat16, s));
    try mlx.check(mlx.mlx_zeros(&c, &shape, 1, .bfloat16, s));

    try testing.expect(cache.put(a.ctx.?, .{ .bits = 4, .group_size = 32, .mode = .affine }));
    try testing.expect(cache.put(b.ctx.?, .{ .bits = 4, .group_size = 64, .mode = .mxfp4 }));
    try testing.expect(cache.put(c.ctx.?, .{ .bits = 8, .group_size = 128, .mode = .affine }));

    // Build a Transformer-like view over the cache via quantParamsFor — we
    // need a real Transformer to call the method, so verify by direct slot
    // lookup instead. (The forward path goes through quantParamsFor, which is
    // exercised by integration tests; this test pins the data structure.)
    {
        const idx = BitsCache.slot(a.ctx.?);
        var found = false;
        for (0..4) |i| {
            const j = (idx + i) & (BITS_CACHE_CAP - 1);
            if (cache.keys[j] == a.ctx) {
                try testing.expectEqual(@as(u8, 4), cache.vals_bits[j]);
                try testing.expectEqual(@as(u8, 32 / 8), cache.vals_gs_div8[j]);
                try testing.expectEqual(@backingInt(QuantMode.affine), cache.vals_mode[j]);
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }
    {
        const idx = BitsCache.slot(c.ctx.?);
        var found = false;
        for (0..4) |i| {
            const j = (idx + i) & (BITS_CACHE_CAP - 1);
            if (cache.keys[j] == c.ctx) {
                try testing.expectEqual(@as(u8, 8), cache.vals_bits[j]);
                try testing.expectEqual(@as(u8, 128 / 8), cache.vals_gs_div8[j]);
                try testing.expectEqual(@backingInt(QuantMode.affine), cache.vals_mode[j]);
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }
}

test "qmatmulBits dispatches to plain matmul when scales has null ctx (Unsloth Dynamic bf16)" {
    const s = mlx.gpuStream();

    // x: [1, 1, in=4], w: [out=2, in=4] (PyTorch convention).
    // Expected: x @ w.T = [1, 1, 2]
    var x_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_flat);
    try mlx.check(mlx.mlx_arange(&x_flat, 0.0, 4.0, 1.0, .float32, s));
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    {
        const sh = [_]c_int{ 1, 1, 4 };
        try mlx.check(mlx.mlx_reshape(&x, x_flat, &sh, 3, s));
    }

    var w_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_flat);
    try mlx.check(mlx.mlx_arange(&w_flat, 0.0, 8.0, 1.0, .float32, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    {
        const sh = [_]c_int{ 2, 4 };
        try mlx.check(mlx.mlx_reshape(&w, w_flat, &sh, 2, s));
    }

    // Pre-transpose like initMoeLayers does for null-scales weights: [out, in] → [in, out]
    const w_t = try transposeBf16Weight(w, s);
    defer _ = mlx.mlx_array_free(w_t);

    // qmatmulBits with null sc/bi must plain-matmul x @ w_t.
    const null_sc = mlx.mlx_array{ .ctx = null };
    const null_bi = mlx.mlx_array{ .ctx = null };
    const got = try qmatmulBits(x, w_t, null_sc, null_bi, 4, 64, .affine, s);
    defer _ = mlx.mlx_array_free(got);

    // Reduce to host floats for comparison.
    var got_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(got_f32);
    try mlx.check(mlx.mlx_astype(&got_f32, got, .float32, s));
    var got_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(got_flat);
    {
        const sh = [_]c_int{2};
        try mlx.check(mlx.mlx_reshape(&got_flat, got_f32, &sh, 1, s));
    }
    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, got_flat);
        try mlx.check(mlx.mlx_eval(ev));
    }
    const data = mlx.mlx_array_data_float32(got_flat) orelse return error.TestUnexpectedNullData;
    // [0,1,2,3] @ [[0,4],[1,5],[2,6],[3,7]] = [0+1*1+2*2+3*3, 0+1*5+2*6+3*7] = [14, 38]
    // (w transposed from [[0,1,2,3],[4,5,6,7]] gives w_t = [[0,4],[1,5],[2,6],[3,7]])
    try testing.expectApproxEqAbs(@as(f32, 14.0), data[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 38.0), data[1], 1e-3);
}

test "fused MoE router reproduces the softmax routing chain's top-K and beats its weights" {
    try mlx.requireGpu();
    // Selection is discrete: a different expert set is a different model, so
    // the ids are asserted EXACTLY against an fp64 host reference. The weights
    // follow the kernel rule — no worse than the fp64 reference, never
    // kernel-vs-kernel equality (the chain rounds a full bf16 softmax; the
    // kernel rounds once).
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE11);
    const rnd = prng.random();
    moe_router_fused_override = true;
    defer moe_router_fused_override = null;

    const ROWS: c_int = 5;
    const E: c_int = 256;
    const K: c_int = 8;
    const n: usize = @intCast(ROWS * E);

    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    // Spread wide enough that a bf16 round-trip leaves the top-K boundary
    // unambiguous; ties there would make "the chain's set" undefined, not wrong.
    for (buf, 0..) |*v, i| v.* = (rnd.float(f32) - 0.5) * 8.0 + @as(f32, @floatFromInt(i % 7)) * 0.011;
    const lsh = [_]c_int{ ROWS, E };
    const l32 = mlx.mlx_array_new_data(buf.ptr, &lsh, 2, .float32);
    defer _ = mlx.mlx_array_free(l32);
    var lbf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lbf);
    try mlx.check(mlx.mlx_astype(&lbf, l32, .bfloat16, s));

    // The kernel and the reference must see the SAME bf16 values.
    const host = try allocator.alloc(f32, n);
    defer allocator.free(host);
    try testReadF32(lbf, host, s);

    const fused = (try moeRouterTopK(s, lbf, .{ .ctx = null }, K, .softmax, true, 1.0, .bfloat16, 0, 0)) orelse
        return error.FusedRouterDeclined;
    defer _ = mlx.mlx_array_free(fused.inds);
    defer _ = mlx.mlx_array_free(fused.norm_scores);
    const chain = try moeRoutingChain(lbf, K, s);
    defer _ = mlx.mlx_array_free(chain.inds);
    defer _ = mlx.mlx_array_free(chain.norm_scores);

    const kn: usize = @intCast(ROWS * K);
    const f_ids = try allocator.alloc(f32, kn);
    defer allocator.free(f_ids);
    const c_ids = try allocator.alloc(f32, kn);
    defer allocator.free(c_ids);
    const f_w = try allocator.alloc(f32, kn);
    defer allocator.free(f_w);
    const c_w = try allocator.alloc(f32, kn);
    defer allocator.free(c_w);
    try testReadF32(fused.inds, f_ids, s);
    try testReadF32(chain.inds, c_ids, s);
    try testReadF32(fused.norm_scores, f_w, s);
    try testReadF32(chain.norm_scores, c_w, s);

    const keys = try allocator.alloc(f64, @intCast(E));
    defer allocator.free(keys);
    var ref_ids: [8]usize = undefined;
    var fused_err: f64 = 0;
    var chain_err: f64 = 0;
    for (0..@intCast(ROWS)) |r| {
        const row = host[r * @as(usize, @intCast(E)) ..][0..@intCast(E)];
        for (row, 0..) |v, i| keys[i] = @floatCast(v);
        routerRefRow(keys, @intCast(K), &ref_ids);

        // Exact ids, in the kernel's descending-key order.
        for (0..@intCast(K)) |j| {
            try testing.expectEqual(ref_ids[j], @as(usize, @intFromFloat(f_ids[r * @as(usize, @intCast(K)) + j])));
        }
        // The chain's order is argpartition's, so compare it as a SET.
        var chain_set: [8]usize = undefined;
        for (0..@intCast(K)) |j| chain_set[j] = @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + j]);
        std.mem.sort(usize, &chain_set, {}, std.sort.asc(usize));
        var ref_sorted = ref_ids;
        std.mem.sort(usize, &ref_sorted, {}, std.sort.asc(usize));
        try testing.expectEqualSlices(usize, &ref_sorted, &chain_set);

        // fp64 reference weights: softmax over the selected logits.
        var denom: f64 = 0;
        var num: [8]f64 = undefined;
        for (0..@intCast(K)) |j| {
            num[j] = @exp(keys[ref_ids[j]] - keys[ref_ids[0]]);
            denom += num[j];
        }
        for (0..@intCast(K)) |j| {
            const want = num[j] / denom;
            const got = f_w[r * @as(usize, @intCast(K)) + j];
            fused_err = @max(fused_err, @abs(want - @as(f64, got)));
            // Chain rows are in argpartition order — find this id's slot.
            var slot: usize = 0;
            while (slot < @as(usize, @intCast(K))) : (slot += 1) {
                if (@as(usize, @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + slot])) == ref_ids[j]) break;
            }
            const chain_got = c_w[r * @as(usize, @intCast(K)) + slot];
            chain_err = @max(chain_err, @abs(want - @as(f64, chain_got)));
            // BIT equality with the chain: the kernel replicates the chain's
            // arithmetic, and anything else is a different model (a live A/B on
            // a live A/B diverged at token ~80 when it did not).
            try testing.expectEqual(chain_got, got);
        }
    }
    try testing.expect(fused_err <= chain_err);
}

test "fused MoE router breaks ties on the lowest expert id at every width and mode" {
    try mlx.requireGpu();
    // Duplicated keys put ties inside the top-K, across lanes and within one
    // lane's strided slice; the pick must stay lowest-index-first.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x71E5_0001);
    const rnd = prng.random();
    moe_router_fused_override = true;
    defer moe_router_fused_override = null;

    const widths = [_]c_int{ 100, 256, 512, 2048 };
    const modes = [_]RouterMode{ .softmax, .sigmoid_bias };
    const ROWS: c_int = 48;
    const K: c_int = 8;
    for (modes) |mode| for (widths) |E| {
        const ne: usize = @intCast(E);
        const n: usize = @as(usize, @intCast(ROWS)) * ne;
        const buf = try allocator.alloc(f32, n);
        defer allocator.free(buf);
        for (buf) |*v| v.* = @floatFromInt(rnd.uintLessThan(u32, 5));
        const lsh = [_]c_int{ ROWS, E };
        const l32 = mlx.mlx_array_new_data(buf.ptr, &lsh, 2, .float32);
        defer _ = mlx.mlx_array_free(l32);
        const zeros = try allocator.alloc(f32, ne);
        defer allocator.free(zeros);
        @memset(zeros, 0);
        const bsh = [_]c_int{E};
        const bias = mlx.mlx_array_new_data(zeros.ptr, &bsh, 1, .float32);
        defer _ = mlx.mlx_array_free(bias);

        const fused = (try moeRouterTopK(s, l32, bias, K, mode, true, 1.0, .float32, 0, 0)) orelse
            return error.FusedRouterDeclined;
        defer _ = mlx.mlx_array_free(fused.inds);
        defer _ = mlx.mlx_array_free(fused.norm_scores);
        const ids = try allocator.alloc(f32, @as(usize, @intCast(ROWS)) * 8);
        defer allocator.free(ids);
        try testReadF32(fused.inds, ids, s);

        const keys = try allocator.alloc(f64, ne);
        defer allocator.free(keys);
        var ref_ids: [8]usize = undefined;
        for (0..@intCast(ROWS)) |r| {
            for (buf[r * ne ..][0..ne], 0..) |v, i| keys[i] = v;
            routerRefRow(keys, 8, &ref_ids);
            for (0..8) |j| try testing.expectEqual(ref_ids[j], @as(usize, @intFromFloat(ids[r * 8 + j])));
        }
    };
}

test "fused softmax router folds the shared-expert gate dot bit-identically to the matmul it replaces" {
    try mlx.requireGpu();
    // The dot rides the router's idle simdgroups in MLX's own `dot_product` order; the cancelling
    // +-2^16 terms make any other summation order change the bf16 result.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6A7E_0002);
    const rnd = prng.random();
    moe_router_fused_override = true;
    defer moe_router_fused_override = null;

    const E: c_int = 512;
    const K: c_int = 10;
    const GK: usize = 2560;
    const bf = struct {
        fn f(buf: []const f32, shape: []const c_int, st: mlx.mlx_stream) !mlx.mlx_array {
            const a32 = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(a32);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&out, a32, .bfloat16, st));
            return out;
        }
    }.f;
    const lbuf = try allocator.alloc(f32, @intCast(E));
    defer allocator.free(lbuf);
    const xbuf = try allocator.alloc(f32, GK);
    defer allocator.free(xbuf);
    const wbuf = try allocator.alloc(f32, GK);
    defer allocator.free(wbuf);

    for (0..300) |_| {
        for (lbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 8.0;
        for (xbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;
        for (wbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 0.4;
        for (0..3) |_| {
            const i = rnd.uintLessThan(usize, GK);
            const j = rnd.uintLessThan(usize, GK);
            if (i == j) continue;
            xbuf[i] = 65536.0;
            wbuf[i] = 1.0;
            xbuf[j] = -65536.0;
            wbuf[j] = 1.0;
        }
        const logits = try bf(lbuf, &.{ 1, E }, s);
        defer _ = mlx.mlx_array_free(logits);
        const x = try bf(xbuf, &.{ 1, 1, @intCast(GK) }, s);
        defer _ = mlx.mlx_array_free(x);
        const w_row = try bf(wbuf, &.{ 1, @intCast(GK) }, s);
        defer _ = mlx.mlx_array_free(w_row);
        const w_t = try transposeBf16Weight(w_row, s);
        defer _ = mlx.mlx_array_free(w_t);
        var w_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w_flat);
        try mlx.check(mlx.mlx_reshape(&w_flat, w_t, &[_]c_int{@intCast(GK)}, 1, s));
        var w_vec = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w_vec);
        try mlx.check(mlx.mlx_contiguous(&w_vec, w_flat, false, s));

        const want = try qmatmulBits(x, w_t, .{ .ctx = null }, .{ .ctx = null }, 0, 0, .affine, s);
        defer _ = mlx.mlx_array_free(want);
        const plain = (try moeRouterTopK(s, logits, .{ .ctx = null }, K, .softmax, true, 1.0, .bfloat16, 0, 0)) orelse
            return error.FusedRouterDeclined;
        defer _ = mlx.mlx_array_free(plain.inds);
        defer _ = mlx.mlx_array_free(plain.norm_scores);
        const gated = (try moeRouterTopKGate(s, logits, .{ .ctx = null }, K, .softmax, true, 1.0, .bfloat16, 0, 0, .{ .x = x, .w = w_vec })) orelse
            return error.FusedRouterDeclined;
        defer _ = mlx.mlx_array_free(gated.inds);
        defer _ = mlx.mlx_array_free(gated.norm_scores);
        defer _ = mlx.mlx_array_free(gated.gate_logit);
        try testing.expect(gated.gate_logit.ctx != null);

        var a: [1]f32 = undefined;
        var b: [1]f32 = undefined;
        try testReadF32(want, &a, s);
        try testReadF32(gated.gate_logit, &b, s);
        try testing.expectEqual(@as(u32, @bitCast(a[0])), @as(u32, @bitCast(b[0])));
        var pi: [10]f32 = undefined;
        var gi: [10]f32 = undefined;
        var ps: [10]f32 = undefined;
        var gs: [10]f32 = undefined;
        try testReadF32(plain.inds, &pi, s);
        try testReadF32(gated.inds, &gi, s);
        try testReadF32(plain.norm_scores, &ps, s);
        try testReadF32(gated.norm_scores, &gs, s);
        try testing.expectEqualSlices(f32, &pi, &gi);
        try testing.expectEqualSlices(f32, &ps, &gs);
    }
}

test "the router declines the gate fold when it has no idle simdgroups or the dot leaves MLX's single-block shape" {
    const s = mlx.gpuStream();
    moe_router_fused_override = true;
    defer moe_router_fused_override = null;
    const ones = struct {
        fn f(n: c_int, st: mlx.mlx_stream) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_ones(&o, &[_]c_int{n}, 1, .bfloat16, st));
            return o;
        }
    }.f;
    const small = try ones(32, s);
    defer _ = mlx.mlx_array_free(small);
    const wide = try ones(512, s);
    defer _ = mlx.mlx_array_free(wide);
    const x2560 = try ones(2560, s);
    defer _ = mlx.mlx_array_free(x2560);
    const x2564 = try ones(2564, s);
    defer _ = mlx.mlx_array_free(x2564);
    const x20000 = try ones(20000, s);
    defer _ = mlx.mlx_array_free(x20000);
    // 32 experts: one simdgroup, none left for the dot.
    try testing.expect((try moeRouterTopKGate(s, small, .{ .ctx = null }, 4, .softmax, true, 1.0, .bfloat16, 0, 0, .{ .x = x2560, .w = x2560 })) == null);
    // K not a multiple of 8, and K past one 512-thread block of MLX's kernel.
    try testing.expect((try moeRouterTopKGate(s, wide, .{ .ctx = null }, 4, .softmax, true, 1.0, .bfloat16, 0, 0, .{ .x = x2564, .w = x2564 })) == null);
    try testing.expect((try moeRouterTopKGate(s, wide, .{ .ctx = null }, 4, .softmax, true, 1.0, .bfloat16, 0, 0, .{ .x = x20000, .w = x20000 })) == null);
}

test "fused SwiGLU is bit-identical to the sigmoid+multiply+multiply chain" {
    try mlx.requireGpu();
    // The reason this kernel exists instead of extending mlx_compile to the
    // silu arm: the compiled fusion is NOT bit-identical (greedy diverged at
    // token ~55 on a 40-layer MoE). Assert what the compiled path could not.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x51117001);
    const rnd = prng.random();
    swiglu_fused_override = true;
    defer swiglu_fused_override = null;

    // Sizes that are and are not multiples of the threadgroup, so the grid
    // round-up and the kernel's bound check are both exercised.
    for ([_]c_int{ 8, 512, 4096, 4097 }) |n| {
        for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
            const cnt: usize = @intCast(n);
            const gbuf = try allocator.alloc(f32, cnt);
            defer allocator.free(gbuf);
            const ubuf = try allocator.alloc(f32, cnt);
            defer allocator.free(ubuf);
            for (gbuf, 0..) |*v, i| v.* = switch (i % 5) {
                0 => -30.0,
                1 => 25.0,
                2 => 0.0,
                else => (rnd.float(f32) - 0.5) * 10.0,
            };
            for (ubuf) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;

            const sh = [_]c_int{n};
            const g32 = mlx.mlx_array_new_data(gbuf.ptr, &sh, 1, .float32);
            defer _ = mlx.mlx_array_free(g32);
            const u32a = mlx.mlx_array_new_data(ubuf.ptr, &sh, 1, .float32);
            defer _ = mlx.mlx_array_free(u32a);
            var gate = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate);
            try mlx.check(mlx.mlx_astype(&gate, g32, dt, s));
            var up = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up);
            try mlx.check(mlx.mlx_astype(&up, u32a, dt, s));

            // Reference: exactly the ops computeGeglu's fallback issues.
            var sig = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sig);
            try mlx.check(mlx.mlx_sigmoid(&sig, gate, s));
            var act = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(act);
            try mlx.check(mlx.mlx_multiply(&act, gate, sig, s));
            var ref = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref);
            try mlx.check(mlx.mlx_multiply(&ref, act, up, s));

            const got = (try fusedSwiGLU(s, gate, up)) orelse return error.FusedSwigluDeclined;
            defer _ = mlx.mlx_array_free(got);

            const ref_host = try allocator.alloc(f32, cnt);
            defer allocator.free(ref_host);
            const got_host = try allocator.alloc(f32, cnt);
            defer allocator.free(got_host);
            try testReadF32(ref, ref_host, s);
            try testReadF32(got, got_host, s);
            try testing.expectEqualSlices(f32, ref_host, got_host);
        }
    }
}

test "fused SwiGLU config cache keys on the SHAPE, not the element count (restore crash)" {
    try mlx.requireGpu();
    // Live 2026-07-29: a 16-token hot-cache-restore suffix prefill on a MoE
    // killed the server with "[matmul] (1,16,512) vs (8192,2048)". A MoE
    // layer's SwiGLU at 16 tokens is (1,16,512) = 8192 elements — the SAME
    // count, rank and dtype as a dense mlp_only layer at decode (1,1,8192).
    // The config cache was keyed on {n, tg, ndim, dtype}, so the second call
    // reused a config whose baked OUTPUT SHAPE was the first caller's, and
    // the dense down-proj matmul died on the mislabeled activation. The key
    // must be the full shape; equal products are not equal shapes.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    swiglu_fused_override = true;
    defer swiglu_fused_override = null;

    const shapes = [2][3]c_int{ .{ 1, 16, 512 }, .{ 1, 1, 8192 } };
    for (shapes) |sh| {
        const cnt: usize = 8192;
        const buf = try allocator.alloc(f32, cnt);
        defer allocator.free(buf);
        for (buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) - 3.0;
        const a32 = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
        defer _ = mlx.mlx_array_free(a32);
        var gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate);
        try mlx.check(mlx.mlx_astype(&gate, a32, .bfloat16, s));

        const got = (try fusedSwiGLU(s, gate, gate)) orelse return error.FusedSwigluDeclined;
        defer _ = mlx.mlx_array_free(got);
        try mlx.check(mlx.mlx_array_eval(got));
        const osh = mlx.getShape(got);
        try testing.expectEqualSlices(c_int, &sh, osh);
    }
}

test "fused gate+up+SwiGLU expert kernel is bit-identical to the split gatherQmv path" {
    try mlx.requireGpu();
    // The split path is `gatherQmv(gate)` + `gatherQmv(up)` + `fusedSwiGLU`.
    // Fusing them may not change a single rounding: each accumulator is still
    // rounded to T before the activation, and the activation is still the
    // table. Covers the live affine widths.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6A7E0FF5);
    const rnd = prng.random();
    gqmv_gateup_override = true;
    defer gqmv_gateup_override = null;
    swiglu_fused_override = true;
    defer swiglu_fused_override = null;

    const E: c_int = 8;
    const N: c_int = 64;
    const K: c_int = 256;
    const TOPK: c_int = 3;

    for ([_]struct { bits: u32, gs: u32 }{
        .{ .bits = 4, .gs = 64 },
        .{ .bits = 3, .gs = 128 },
    }) |cfg| {
        const bits = cfg.bits;
        const gs = cfg.gs;
        const mode: QuantMode = .affine;

        // Two independent banks, so a kernel that read the gate bank twice
        // could not pass.
        var banks: [2]struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array } = undefined;
        var made: usize = 0;
        defer for (banks[0..made]) |b| {
            _ = mlx.mlx_array_free(b.w);
            _ = mlx.mlx_array_free(b.sc);
            _ = mlx.mlx_array_free(b.bi);
        };
        for (0..2) |bi_idx| {
            const wb = try testRandWeightBf16(rnd, &[_]c_int{ E, N, K }, s);
            defer _ = mlx.mlx_array_free(wb);
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
            var wq = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
            var sc = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
            var bias = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&bias, triple, 2));
            banks[bi_idx] = .{ .w = wq, .sc = sc, .bi = bias };
            made += 1;
        }

        const xbuf = try allocator.alloc(f32, @intCast(K));
        defer allocator.free(xbuf);
        for (xbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
        const xsh = [_]c_int{K};
        const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xsh, 1, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

        var idx: [3]u32 = .{ 5, 0, 3 };
        const ish = [_]c_int{TOPK};
        const inds = mlx.mlx_array_new_data(&idx, &ish, 1, .uint32);
        defer _ = mlx.mlx_array_free(inds);

        // Reference: the two separate gathers, then the fused activation.
        const g2 = (try gatherQmv(s, x, banks[0].w, banks[0].sc, banks[0].bi, inds, bits, gs, mode, false)) orelse
            return error.GatherQmvDeclined;
        defer _ = mlx.mlx_array_free(g2);
        const up2 = (try gatherQmv(s, x, banks[1].w, banks[1].sc, banks[1].bi, inds, bits, gs, mode, false)) orelse
            return error.GatherQmvDeclined;
        defer _ = mlx.mlx_array_free(up2);
        const ref = (try fusedSwiGLU(s, g2, up2)) orelse return error.FusedSwigluDeclined;
        defer _ = mlx.mlx_array_free(ref);

        const got = (try gatherQmvGateUp(s, x, banks[0].w, banks[0].sc, banks[0].bi, banks[1].w, banks[1].sc, banks[1].bi, inds, bits, gs, mode)) orelse
            return error.FusedGateUpDeclined;
        defer _ = mlx.mlx_array_free(got);

        const cnt: usize = @intCast(TOPK * N);
        const ref_host = try allocator.alloc(f32, cnt);
        defer allocator.free(ref_host);
        const got_host = try allocator.alloc(f32, cnt);
        defer allocator.free(got_host);
        try testReadF32(ref, ref_host, s);
        try testReadF32(got, got_host, s);
        try testing.expectEqualSlices(f32, ref_host, got_host);
    }
}

fn moeRowsAffineBank(s: mlx.mlx_stream, rnd: std.Random, E: c_int, N: c_int, K: c_int, bits: u32, gs: u32) !struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array } {
    const words: c_int = @intCast(@divExact(@as(u32, @intCast(K)) * bits, 32));
    const groups: c_int = @divExact(K, @as(c_int, @intCast(gs)));
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, rnd.int(u64)));
    const w_shape = [_]c_int{ E, N, words };
    var w = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_bits(&w, &w_shape, 3, 4, key, s));
    const s_shape = [_]c_int{ E, N, groups };
    const sc = try testRandUniformBf16(rnd, &s_shape, 0.01, 0.06, s);
    const bi = try testRandUniformBf16(rnd, &s_shape, -0.5, 0.5, s);
    return .{ .w = w, .sc = sc, .bi = bi };
}

fn moeRowsSliceRow(s: mlx.mlx_stream, x: mlx.mlx_array, i: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var start: [3]c_int = .{ i, 0, 0 };
    var stop: [3]c_int = .{ i + 1, 1, 1 };
    const strides = [_]c_int{ 1, 1, 1 };
    for (1..sh.len) |d| stop[d] = sh[d];
    var sl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sl);
    try mlx.check(mlx.mlx_slice(&sl, x, &start, @intCast(sh.len), &stop, @intCast(sh.len), &strides, @intCast(sh.len), s));
    var sq = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sq);
    try mlx.check(mlx.mlx_squeeze(&sq, sl, s));
    return sq;
}

test "gatherQmvGateUpRows accepts N=8 K=10" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    moe_rows_fused_override = true;
    defer moe_rows_fused_override = null;
    gqmv_gateup_override = true;
    defer gqmv_gateup_override = null;
    const nrows: c_int = 8;
    const topk: c_int = 10;
    const hid: c_int = 2560;
    const inter: c_int = 640;
    const E: c_int = 8;
    const bits: u32 = 4;
    const gs: u32 = 64;
    const xsh = [_]c_int{ nrows, hid };
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_zeros(&x, &xsh, 2, .bfloat16, s));
    const ish = [_]c_int{ nrows, topk };
    var idata: [80]u32 = @splat(0);
    const inds = mlx.mlx_array_new_data(&idata, &ish, 2, .uint32);
    defer _ = mlx.mlx_array_free(inds);
    const wsh = [_]c_int{ E, inter, @intCast(@divExact(hid * @as(c_int, @intCast(bits)), 32)) };
    var gw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gw);
    try mlx.check(mlx.mlx_zeros(&gw, &wsh, 3, .uint32, s));
    var uw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(uw);
    try mlx.check(mlx.mlx_zeros(&uw, &wsh, 3, .uint32, s));
    const ssh = [_]c_int{ E, inter, @divExact(hid, @as(c_int, @intCast(gs))) };
    var gscl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gscl);
    try mlx.check(mlx.mlx_zeros(&gscl, &ssh, 3, .bfloat16, s));
    var uscl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(uscl);
    try mlx.check(mlx.mlx_zeros(&uscl, &ssh, 3, .bfloat16, s));
    var gb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gb);
    try mlx.check(mlx.mlx_zeros(&gb, &ssh, 3, .bfloat16, s));
    var ub = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ub);
    try mlx.check(mlx.mlx_zeros(&ub, &ssh, 3, .bfloat16, s));
    const got = (try gatherQmvGateUpRows(s, x, gw, gscl, gb, uw, uscl, ub, inds, bits, gs, .affine)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(got);
}

test "moe rows fused: N=1 is byte-identical to the single-row kernel" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x51F0);
    const rnd = prng.random();
    moe_rows_fused_override = true;
    defer moe_rows_fused_override = null;
    gqmv_gateup_override = true;
    defer gqmv_gateup_override = null;
    downred_override = true;
    defer downred_override = null;
    const E: c_int = 32;
    const hid: c_int = 2560;
    const inter: c_int = 640;
    const topk: c_int = 10;
    const bits: u32 = 4;
    const gs: u32 = 64;
    const gate = try moeRowsAffineBank(s, rnd, E, inter, hid, bits, gs);
    defer _ = mlx.mlx_array_free(gate.w);
    defer _ = mlx.mlx_array_free(gate.sc);
    defer _ = mlx.mlx_array_free(gate.bi);
    const up = try moeRowsAffineBank(s, rnd, E, inter, hid, bits, gs);
    defer _ = mlx.mlx_array_free(up.w);
    defer _ = mlx.mlx_array_free(up.sc);
    defer _ = mlx.mlx_array_free(up.bi);
    const down = try moeRowsAffineBank(s, rnd, E, hid, inter, 8, 32);
    defer _ = mlx.mlx_array_free(down.w);
    defer _ = mlx.mlx_array_free(down.sc);
    defer _ = mlx.mlx_array_free(down.bi);
    const x1 = try attn256RandBf16(rnd, &[_]c_int{hid}, s);
    defer _ = mlx.mlx_array_free(x1);
    var xrow = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xrow);
    const xrsh = [_]c_int{ 1, hid };
    try mlx.check(mlx.mlx_reshape(&xrow, x1, &xrsh, 2, s));
    var idx: [10]u32 = .{ 0, 3, 7, 2, 11, 5, 9, 1, 4, 8 };
    const inds1 = mlx.mlx_array_new_data(&idx, &[_]c_int{topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(inds1);
    const inds2 = mlx.mlx_array_new_data(&idx, &[_]c_int{ 1, topk }, 2, .uint32);
    defer _ = mlx.mlx_array_free(inds2);
    const solo_gu = (try gatherQmvGateUp(s, x1, gate.w, gate.sc, gate.bi, up.w, up.sc, up.bi, inds1, bits, gs, .affine)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(solo_gu);
    const rows_gu = (try gatherQmvGateUpRows(s, xrow, gate.w, gate.sc, gate.bi, up.w, up.sc, up.bi, inds2, bits, gs, .affine)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(rows_gu);
    var rows_gu_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows_gu_2d);
    try mlx.check(mlx.mlx_squeeze(&rows_gu_2d, rows_gu, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(solo_gu, rows_gu_2d, s));
    const scores1 = try attn256RandBf16(rnd, &[_]c_int{topk}, s);
    defer _ = mlx.mlx_array_free(scores1);
    var scores2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores2);
    const scsh = [_]c_int{ 1, topk };
    try mlx.check(mlx.mlx_reshape(&scores2, scores1, &scsh, 2, s));
    const solo_dn = (try gatherQmvDownReduce(s, solo_gu, down.w, down.sc, down.bi, inds1, scores1, 8, 32, .affine)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(solo_dn);
    const rows_dn = (try gatherQmvDownReduceRows(s, rows_gu, down.w, down.sc, down.bi, inds2, scores2, 8, 32, .affine)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(rows_dn);
    var rows_dn_1d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows_dn_1d);
    try mlx.check(mlx.mlx_squeeze(&rows_dn_1d, rows_dn, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(solo_dn, rows_dn_1d, s));
}

test "moe rows fused: N=8 K=10 gate/up and down/reduce match N solo calls" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xC0DE);
    const rnd = prng.random();
    moe_rows_fused_override = true;
    defer moe_rows_fused_override = null;
    gqmv_gateup_override = true;
    defer gqmv_gateup_override = null;
    downred_override = true;
    defer downred_override = null;
    const E: c_int = 512;
    const hid: c_int = 2560;
    const inter: c_int = 640;
    const topk: c_int = 10;
    const nrows: c_int = 8;
    for ([_]struct { bits: u32, gs: u32, dbits: u32, dgs: u32 }{
        .{ .bits = 4, .gs = 64, .dbits = 8, .dgs = 32 },
        .{ .bits = 3, .gs = 128, .dbits = 3, .dgs = 64 },
    }) |cfg| try moeRowsFusedCase(s, rnd, E, hid, inter, topk, nrows, cfg.bits, cfg.gs, cfg.dbits, cfg.dgs);
}

test "moe verify rows retain serial expert arithmetic through grouped widths" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x165e71a1);
    const saved_rows = moe_rows_fused_override;
    const saved_gu = gqmv_gateup_override;
    const saved_down = downred_override;
    moe_rows_fused_override = true;
    gqmv_gateup_override = true;
    downred_override = true;
    defer moe_rows_fused_override = saved_rows;
    defer gqmv_gateup_override = saved_gu;
    defer downred_override = saved_down;
    for ([_]c_int{ 2, 3, 8, 9, 16, 18, 32 }) |rows| {
        try moeRowsFusedCase(s, prng.random(), 16, 2560, 640, 10, rows, 4, 64, 4, 64);
    }
}

fn moeRowsFusedCase(s: mlx.mlx_stream, rnd: std.Random, E: c_int, hid: c_int, inter: c_int, topk: c_int, nrows: c_int, bits: u32, gs: u32, dbits: u32, dgs: u32) !void {
    const gate = try moeRowsAffineBank(s, rnd, E, inter, hid, bits, gs);
    defer _ = mlx.mlx_array_free(gate.w);
    defer _ = mlx.mlx_array_free(gate.sc);
    defer _ = mlx.mlx_array_free(gate.bi);
    const up = try moeRowsAffineBank(s, rnd, E, inter, hid, bits, gs);
    defer _ = mlx.mlx_array_free(up.w);
    defer _ = mlx.mlx_array_free(up.sc);
    defer _ = mlx.mlx_array_free(up.bi);
    const down = try moeRowsAffineBank(s, rnd, E, hid, inter, dbits, dgs);
    defer _ = mlx.mlx_array_free(down.w);
    defer _ = mlx.mlx_array_free(down.sc);
    defer _ = mlx.mlx_array_free(down.bi);
    const x = try attn256RandBf16(rnd, &[_]c_int{ nrows, hid }, s);
    defer _ = mlx.mlx_array_free(x);
    const idata = try testing.allocator.alloc(u32, @intCast(nrows * topk));
    defer testing.allocator.free(idata);
    const perm = [_]u32{ 0, 3, 0, 7, 2, 3, 5, 1 };
    for (0..@intCast(nrows)) |r| {
        const base = perm[r % perm.len] * 17;
        for (0..10) |k| idata[r * 10 + k] = (base + @as(u32, @intCast(k)) * 13) % @as(u32, @intCast(E));
    }
    const inds = mlx.mlx_array_new_data(idata.ptr, &[_]c_int{ nrows, topk }, 2, .uint32);
    defer _ = mlx.mlx_array_free(inds);
    const scores = try attn256RandBf16(rnd, &[_]c_int{ nrows, topk }, s);
    defer _ = mlx.mlx_array_free(scores);
    const fused_gu = (try gatherQmvGateUpRows(s, x, gate.w, gate.sc, gate.bi, up.w, up.sc, up.bi, inds, bits, gs, .affine)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused_gu);
    const fused_dn = (try gatherQmvDownReduceRows(s, fused_gu, down.w, down.sc, down.bi, inds, scores, dbits, dgs, .affine)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused_dn);
    var r: c_int = 0;
    while (r < nrows) : (r += 1) {
        const xr = try moeRowsSliceRow(s, x, r);
        defer _ = mlx.mlx_array_free(xr);
        const ir = try moeRowsSliceRow(s, inds, r);
        defer _ = mlx.mlx_array_free(ir);
        const sr = try moeRowsSliceRow(s, scores, r);
        defer _ = mlx.mlx_array_free(sr);
        const solo_gu = (try gatherQmvGateUp(s, xr, gate.w, gate.sc, gate.bi, up.w, up.sc, up.bi, ir, bits, gs, .affine)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(solo_gu);
        const fused_r = try moeRowsSliceRow(s, fused_gu, r);
        defer _ = mlx.mlx_array_free(fused_r);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(fused_r, solo_gu, s));
        const solo_dn = (try gatherQmvDownReduce(s, solo_gu, down.w, down.sc, down.bi, ir, sr, dbits, dgs, .affine)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(solo_dn);
        const fused_dr = try moeRowsSliceRow(s, fused_dn, r);
        defer _ = mlx.mlx_array_free(fused_dr);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(fused_dr, solo_dn, s));
    }
}

test "fused SwiGLU matches the chain on EVERY finite bf16 gate value" {
    try mlx.requireGpu();
    // Random sampling said bit-identical while the live model diverged at
    // token ~55, so the sigmoid is swept exhaustively instead: all 65536 bf16
    // bit patterns, which is the whole domain of the activation.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    swiglu_fused_override = true;
    defer swiglu_fused_override = null;

    const N: usize = 1 << 16;
    const sh = [_]c_int{@intCast(N)};
    // An f32 whose low 16 mantissa bits are zero converts to bf16 EXACTLY, so
    // walking the top half of the f32 word walks every bf16 pattern there is —
    // subnormals, both zeros, infinities and NaNs included.
    const gbuf = try allocator.alloc(f32, N);
    defer allocator.free(gbuf);
    for (gbuf, 0..) |*v, i| v.* = @bitCast(@as(u32, @intCast(i)) << 16);
    const g32 = mlx.mlx_array_new_data(gbuf.ptr, &sh, 1, .float32);
    defer _ = mlx.mlx_array_free(g32);
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_astype(&gate, g32, .bfloat16, s));

    const ones = try allocator.alloc(f32, N);
    defer allocator.free(ones);
    for (ones, 0..) |*v, i| v.* = if (i % 3 == 0) 1.0 else if (i % 3 == 1) -0.375 else 2.5;
    const up32 = mlx.mlx_array_new_data(ones.ptr, &sh, 1, .float32);
    defer _ = mlx.mlx_array_free(up32);
    var up = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up);
    try mlx.check(mlx.mlx_astype(&up, up32, .bfloat16, s));

    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, gate, s));
    var act = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_multiply(&act, gate, sig, s));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_multiply(&ref, act, up, s));

    const got = (try fusedSwiGLU(s, gate, up)) orelse return error.FusedSwigluDeclined;
    defer _ = mlx.mlx_array_free(got);

    const ref_host = try allocator.alloc(f32, N);
    defer allocator.free(ref_host);
    const got_host = try allocator.alloc(f32, N);
    defer allocator.free(got_host);
    try testReadF32(ref, ref_host, s);
    try testReadF32(got, got_host, s);
    var bad: usize = 0;
    var first: usize = 0;
    for (ref_host, got_host, 0..) |r, g, i| {
        // NaN inputs produce NaN on both sides; compare them as bit patterns.
        const same = (std.math.isNan(r) and std.math.isNan(g)) or r == g;
        if (!same) {
            if (bad == 0) first = i;
            bad += 1;
        }
    }
    if (bad != 0) {
        std.debug.print("swiglu mismatch on {d}/{d} bf16 values; first at 0x{X:0>4}: ref={d} got={d}\n", .{ bad, N, first, ref_host[first], got_host[first] });
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "fused SwiGLU declines mismatched shapes rather than reading out of bounds" {
    const s = mlx.gpuStream();
    swiglu_fused_override = true;
    defer swiglu_fused_override = null;
    const g_sh = [_]c_int{ 4, 8 };
    const u_sh = [_]c_int{ 4, 4 };
    var gbuf: [32]f32 = @splat(0.5);
    var ubuf: [16]f32 = @splat(0.25);
    const g32 = mlx.mlx_array_new_data(&gbuf, &g_sh, 2, .float32);
    defer _ = mlx.mlx_array_free(g32);
    const u32a = mlx.mlx_array_new_data(&ubuf, &u_sh, 2, .float32);
    defer _ = mlx.mlx_array_free(u32a);
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_astype(&g, g32, .bfloat16, s));
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_astype(&u, u32a, .bfloat16, s));
    try testing.expect((try fusedSwiGLU(s, g, u)) == null);
    // f32 is declined outright — see the dtype gate's comment.
    try testing.expect((try fusedSwiGLU(s, g32, g32)) == null);
}

test "verify shared projection preserves solo qmv values across request widths" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E);
    for ([_][2]c_int{ .{ 2560, 640 }, .{ 640, 2560 } }) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weight = try attn256RandBf16(prng.random(), &.{ n, k }, s);
        defer _ = mlx.mlx_array_free(weight);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]usize{ 2, 3, 4, 8 }) |count| for ([_][8]c_int{ @splat(2), @splat(3), @splat(4), @splat(5), @splat(6), @splat(7), .{ 5, 3, 2, 4, 2, 2, 3, 3 } }) |lengths| {
            var total: c_int = 0;
            for (lengths[0..count]) |width| total += width;
            const x = try attn256RandBf16(prng.random(), &.{ 1, total, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (total > 24) {
                try testing.expect((try verifySharedProjection(s, x, w, sc, bi, lengths[0..count])) == null);
                continue;
            }
            const grouped = (try verifySharedProjection(s, x, w, sc, bi, lengths[0..count])) orelse return error.SharedProjectionDeclined;
            defer _ = mlx.mlx_array_free(grouped);
            var offset: c_int = 0;
            for (lengths[0..count]) |width| {
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, k }, 3, &.{ 1, 1, 1 }, 3, s));
                const expected = try qmatmulBits(row, w, sc, bi, 8, 64, .affine, s);
                defer _ = mlx.mlx_array_free(expected);
                var actual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_slice(&actual, grouped, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, n }, 3, &.{ 1, 1, 1 }, 3, s));
                try testing.expect(try qsaArraysAllEqual(expected, actual, s));
                offset += width;
            }
        };
        const x = try attn256RandBf16(prng.random(), &.{ 1, 8, k }, s);
        defer _ = mlx.mlx_array_free(x);
        for ([_][]const c_int{ &.{ 1, 7 }, &.{8}, &.{ 2, 2 }, &.{ 0, 8 } }) |widths| {
            try testing.expect((try verifySharedProjection(s, x, w, sc, bi, widths)) == null);
        }
        try testing.expect((try verifySharedProjection(s, x, w, sc, .{}, &.{ 4, 4 })) == null);
        const wide = try attn256RandBf16(prng.random(), &.{ 1, 25, k }, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expect((try verifySharedProjection(s, wide, w, sc, bi, &.{ 5, 5, 5, 5, 5 })) == null);
        const old = vqmm_mixed_plain_test_force;
        vqmm_mixed_plain_test_force = true;
        defer vqmm_mixed_plain_test_force = old;
        try testing.expect((try verifySharedProjection(s, x, w, sc, bi, &.{ 4, 4 })) == null);
    }
}

test "verify GDN projections preserve solo qmv values across request widths" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E);
    for ([_][2]c_int{ .{ 2560, 10240 }, .{ 2560, 6144 }, .{ 2560, 48 }, .{ 6144, 2560 } }) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weight = try attn256RandBf16(prng.random(), &.{ n, k }, s);
        defer _ = mlx.mlx_array_free(weight);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]usize{ 2, 3, 4, 8 }) |count| for ([_][8]c_int{ @splat(2), @splat(3), @splat(4), @splat(5), @splat(6), @splat(7), .{ 5, 3, 2, 4, 2, 2, 3, 3 } }) |lengths| {
            var total: c_int = 0;
            for (lengths[0..count]) |width| total += width;
            const x = try attn256RandBf16(prng.random(), &.{ 1, total, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (total > 12 or lengths[0] > 6) {
                try testing.expect((try verifyJoinedProjection(s, x, w, sc, bi, lengths[0..count], .gdn)) == null);
                continue;
            }
            const grouped = (try verifyJoinedProjection(s, x, w, sc, bi, lengths[0..count], .gdn)) orelse return error.GdnProjectionDeclined;
            defer _ = mlx.mlx_array_free(grouped);
            var offset: c_int = 0;
            for (lengths[0..count]) |width| {
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, k }, 3, &.{ 1, 1, 1 }, 3, s));
                const expected = try qmatmulBits(row, w, sc, bi, 8, 64, .affine, s);
                defer _ = mlx.mlx_array_free(expected);
                var actual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_slice(&actual, grouped, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, n }, 3, &.{ 1, 1, 1 }, 3, s));
                try testing.expect(try qsaArraysAllEqual(expected, actual, s));
                offset += width;
            }
        };
        const x = try attn256RandBf16(prng.random(), &.{ 1, 8, k }, s);
        defer _ = mlx.mlx_array_free(x);
        for ([_][]const c_int{ &.{ 1, 7 }, &.{8}, &.{ 2, 2 }, &.{ 0, 8 } }) |widths| {
            try testing.expect((try verifyJoinedProjection(s, x, w, sc, bi, widths, .gdn)) == null);
        }
        try testing.expect((try verifyJoinedProjection(s, x, w, sc, .{}, &.{ 4, 4 }, .gdn)) == null);
        const wide = try attn256RandBf16(prng.random(), &.{ 1, 25, k }, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expect((try verifyJoinedProjection(s, wide, w, sc, bi, &.{ 5, 5, 5, 5, 5 }, .gdn)) == null);
        const old = vqmm_mixed_plain_test_force;
        vqmm_mixed_plain_test_force = true;
        defer vqmm_mixed_plain_test_force = old;
        try testing.expect((try verifyJoinedProjection(s, x, w, sc, bi, &.{ 4, 4 }, .gdn)) == null);
    }
}

test "verify attention projections preserve solo qmv values across request widths" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E);
    for ([_][2]c_int{ .{ 2560, 12288 }, .{ 2560, 512 }, .{ 6144, 2560 } }) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weight = try attn256RandBf16(prng.random(), &.{ n, k }, s);
        defer _ = mlx.mlx_array_free(weight);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]usize{ 2, 3, 4, 8 }) |count| for ([_][8]c_int{ @splat(2), @splat(3), @splat(4), @splat(5), @splat(6), @splat(7), .{ 5, 3, 2, 4, 2, 2, 3, 3 } }) |lengths| {
            var total: c_int = 0;
            for (lengths[0..count]) |width| total += width;
            const x = try attn256RandBf16(prng.random(), &.{ 1, total, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (total > 12 or lengths[0] > 6) {
                try testing.expect((try verifyJoinedProjection(s, x, w, sc, bi, lengths[0..count], .attention)) == null);
                continue;
            }
            const grouped = (try verifyJoinedProjection(s, x, w, sc, bi, lengths[0..count], .attention)) orelse return error.AttnProjectionDeclined;
            defer _ = mlx.mlx_array_free(grouped);
            var offset: c_int = 0;
            for (lengths[0..count]) |width| {
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, k }, 3, &.{ 1, 1, 1 }, 3, s));
                const expected = try qmatmulBits(row, w, sc, bi, 8, 64, .affine, s);
                defer _ = mlx.mlx_array_free(expected);
                var actual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_slice(&actual, grouped, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, n }, 3, &.{ 1, 1, 1 }, 3, s));
                try testing.expect(try qsaArraysAllEqual(expected, actual, s));
                offset += width;
            }
        };
        const x = try attn256RandBf16(prng.random(), &.{ 1, 8, k }, s);
        defer _ = mlx.mlx_array_free(x);
        for ([_][]const c_int{ &.{ 1, 7 }, &.{8}, &.{ 2, 2 }, &.{ 0, 8 } }) |widths| {
            try testing.expect((try verifyJoinedProjection(s, x, w, sc, bi, widths, .attention)) == null);
        }
        try testing.expect((try verifyJoinedProjection(s, x, w, sc, .{}, &.{ 4, 4 }, .attention)) == null);
        const wide = try attn256RandBf16(prng.random(), &.{ 1, 25, k }, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expect((try verifyJoinedProjection(s, wide, w, sc, bi, &.{ 5, 5, 5, 5, 5 }, .attention)) == null);
        const old = vqmm_mixed_plain_test_force;
        vqmm_mixed_plain_test_force = true;
        defer vqmm_mixed_plain_test_force = old;
        try testing.expect((try verifyJoinedProjection(s, x, w, sc, bi, &.{ 4, 4 }, .attention)) == null);
    }
}

test "qmatmulBits uses canonical arithmetic for plain batched projections" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const saved_kernel = sushi_qmv_rows_kernel;
    sushi_qmv_rows_kernel = null;
    defer {
        if (sushi_qmv_rows_kernel) |kernel| _ = mlx.mlx_fast_metal_kernel_free(kernel);
        sushi_qmv_rows_kernel = saved_kernel;
    }
    var prng = std.Random.DefaultPrng.init(0xB47C4);
    const w_bf = try attn256RandBf16(prng.random(), &.{ 512, 2560 }, s);
    defer _ = mlx.mlx_array_free(w_bf);
    for ([_]u32{ 4, 8 }) |bits| {
        var qvec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(qvec);
        try mlx.check(mlx.mlx_quantize(&qvec, w_bf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, qvec, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, qvec, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, qvec, 2));
        for ([_]c_int{ 2, 4, 5, 9 }) |rows| {
            const x = try attn256RandBf16(prng.random(), &.{ rows, 2560 }, s);
            defer _ = mlx.mlx_array_free(x);
            const y = try qmatmulBits(x, w, sc, bi, bits, 64, .affine, s);
            defer _ = mlx.mlx_array_free(y);
            try mlx.check(mlx.mlx_array_eval(y));
            if (sushi_qmv_rows_kernel != null) std.debug.print("generic projection invoked MTP row-axis kernel: rows={d} bits={d}\n", .{ rows, bits });
            try testing.expect(sushi_qmv_rows_kernel == null);
            const expected = if (bits == 8) blk: {
                var parts: [9]mlx.mlx_array = @splat(.{});
                var count: usize = 0;
                defer for (parts[0..count]) |array| {
                    _ = mlx.mlx_array_free(array);
                };
                var row: c_int = 0;
                while (row < rows) : (row += 1) {
                    var xr = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(xr);
                    try mlx.check(mlx.mlx_slice(&xr, x, &[_]c_int{ row, 0 }, 2, &[_]c_int{ row + 1, 2560 }, 2, &[_]c_int{ 1, 1 }, 2, s));
                    parts[count] = mlx.mlx_array_new();
                    count += 1;
                    try mlx.check(mlx.mlx_quantized_matmul(&parts[count - 1], xr, w, sc, bi, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", s));
                }
                const vec = mlx.mlx_vector_array_new_data(&parts, count);
                defer _ = mlx.mlx_vector_array_free(vec);
                var result = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(result);
                try mlx.check(mlx.mlx_concatenate_axis(&result, vec, 0, s));
                break :blk result;
            } else (try verifyQmm(s, x, w, sc, bi, bits, 64)) orelse blk: {
                var stock = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_quantized_matmul(&stock, x, w, sc, bi, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", s));
                break :blk stock;
            };
            defer _ = mlx.mlx_array_free(expected);
            try testing.expect(try qsaArraysAllEqual(y, expected, s));
        }
    }
}

test "sushi_qmv_rows is mlx_equal per row to stock M=1 qmv at every head shape" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x51500A01);
    const rnd = prng.random();
    const shapes = [_][2]c_int{
        .{ 2560, 2560 },
        .{ 2560, 12288 },
        .{ 2560, 512 },
        .{ 12288, 2560 },
        .{ 2560, 640 },
        .{ 2560, 248320 },
    };
    const bitss = [_]u32{ 3, 4, 8 };
    const ns = [_]c_int{ 2, 4, 8, 16 };
    const gs: u32 = 64;
    for (bitss) |bits| {
        for (shapes) |kn| {
            const K = kn[0];
            const Nout = kn[1];
            const w_bf = try testRandWeightBf16(rnd, &[_]c_int{ Nout, K }, s);
            defer _ = mlx.mlx_array_free(w_bf);
            var qvec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(qvec);
            try mlx.check(mlx.mlx_quantize(&qvec, w_bf, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, s));
            var w = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(w);
            var sc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sc);
            var bi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bi);
            try mlx.check(mlx.mlx_vector_array_get(&w, qvec, 0));
            try mlx.check(mlx.mlx_vector_array_get(&sc, qvec, 1));
            try mlx.check(mlx.mlx_vector_array_get(&bi, qvec, 2));
            for (ns) |nrows| {
                const x = try attn256RandBf16(rnd, &[_]c_int{ nrows, K }, s);
                defer _ = mlx.mlx_array_free(x);
                const got = (try msvQmvRows(s, x, w, sc, bi, bits, gs)) orelse {
                    std.debug.print("[sushi_qmv_rows] declined bits={d} K={d} N={d} rows={d}\n", .{ bits, K, Nout, nrows });
                    return error.QmvRowsDeclined;
                };
                defer _ = mlx.mlx_array_free(got);
                var r: c_int = 0;
                while (r < nrows) : (r += 1) {
                    const xr = try moeRowsSliceRow(s, x, r);
                    defer _ = mlx.mlx_array_free(xr);
                    var xr2 = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(xr2);
                    try mlx.check(mlx.mlx_reshape(&xr2, xr, &[_]c_int{ 1, K }, 2, s));
                    var want = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(want);
                    try mlx.check(mlx.mlx_quantized_matmul(
                        &want,
                        xr2,
                        w,
                        sc,
                        bi,
                        true,
                        mlx.mlx_optional_int.some(@intCast(gs)),
                        mlx.mlx_optional_int.some(@intCast(bits)),
                        "affine",
                        s,
                    ));
                    const gr = try moeRowsSliceRow(s, got, r);
                    defer _ = mlx.mlx_array_free(gr);
                    const diff = try attn256MaxDiff(gr, want, s);
                    if (diff != 0) {
                        std.debug.print("[sushi_qmv_rows] NE bits={d} K={d} N={d} rows={d} row={d} maxdiff={d}\n", .{ bits, K, Nout, nrows, r, diff });
                    }
                    try testing.expectEqual(@as(f32, 0.0), diff);
                }
            }
        }
    }
}

test "moeRouterThreadGroup keeps a full simdgroup and caps at 256" {
    try testing.expectEqual(@as(c_int, 32), moeRouterThreadGroup(8));
    try testing.expectEqual(@as(c_int, 32), moeRouterThreadGroup(32));
    try testing.expectEqual(@as(c_int, 64), moeRouterThreadGroup(33));
    try testing.expectEqual(@as(c_int, 128), moeRouterThreadGroup(128));
    try testing.expectEqual(@as(c_int, 256), moeRouterThreadGroup(256));
    try testing.expectEqual(@as(c_int, 256), moeRouterThreadGroup(2048));
}

test "gatherExpertMm dense bf16 matches per-expert ground truth (decode + prefill shapes)" {
    try mlx.requireGpu();
    // Centerpiece TDD gate for fully-dense bf16 MoE. Proves the dense gather_mm
    // path (gatherExpertMm with null scales + generalized transposeBf16Weight)
    // computes the same per-expert matmul as (a) an independent fp32 ground truth
    // and (b) the established quantized gather_qmm path — for BOTH the decode
    // (S=1, unsorted, 5D x) and prefill ([N,1,in], sorted) call shapes used by
    // moeMLP2. If the historical Qwen3.6-A3B-bf16 generation bug lived in the
    // expert gather, this test fails for the right reason.
    const s = mlx.gpuStream();
    const alloc = testing.allocator;

    const E = 4;
    const IN = 32; // must be a multiple of gs
    const OUT = 8;
    const gs = 32; // mlx_quantize supports group sizes 32, 64, 128
    const bits = 8; // near-lossless quant for a tight cross-check

    // Build w_orig [E, OUT, IN] bf16 from a deterministic small-valued buffer.
    var w_host: [E * OUT * IN]f32 = undefined;
    for (&w_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6))) * 0.05;
    var w_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_f32);
    {
        const sh = [_]c_int{ E, OUT, IN };
        w_f32 = mlx.mlx_array_new_data(&w_host, &sh, 3, .float32);
    }
    var w_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_bf16);
    try mlx.check(mlx.mlx_astype(&w_bf16, w_f32, .bfloat16, s));

    // Quantize → [q, sc, bi]; dequantize back so the dense path consumes the
    // exact numbers gather_qmm sees internally (cancels quant error in the
    // dense-vs-quant comparison).
    var qvec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_quantize(&qvec, w_bf16, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{}, s));
    var q_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_w);
    var q_sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_sc);
    var q_bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_bi);
    try mlx.check(mlx.mlx_vector_array_get(&q_w, qvec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q_sc, qvec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&q_bi, qvec, 2));

    var w_deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_deq);
    try mlx.check(mlx.mlx_dequantize(&w_deq, q_w, q_sc, q_bi, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{}, .{ .value = .bfloat16, .has_value = true }, s));
    // w_deq_t [E, IN, OUT] — the dense weight layout the loader produces.
    const w_deq_t = try transposeBf16Weight(w_deq, s);
    defer _ = mlx.mlx_array_free(w_deq_t);

    // Read dequantized weights to host for ground truth: w_deq_host[e*OUT*IN + o*IN + k].
    var w_deq_host: [E * OUT * IN]f32 = undefined;
    try testReadF32(w_deq, &w_deq_host, s);

    const null_sc = mlx.mlx_array{ .ctx = null };
    const null_bi = mlx.mlx_array{ .ctx = null };
    const no_idx = mlx.mlx_array{ .ctx = null };

    // ── Decode shape: x_exp [1,1,1,1,IN], inds [1,1,K], sorted=false ──
    {
        const K = 2;
        var x_host: [IN]f32 = undefined;
        for (&x_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2))) * 0.1;
        var x_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_f32);
        {
            const sh = [_]c_int{IN};
            x_f32 = mlx.mlx_array_new_data(&x_host, &sh, 1, .float32);
        }
        var x_bf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_bf);
        try mlx.check(mlx.mlx_astype(&x_bf, x_f32, .bfloat16, s));
        var x_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_exp);
        {
            const sh = [_]c_int{ 1, 1, 1, 1, IN };
            try mlx.check(mlx.mlx_reshape(&x_exp, x_bf, &sh, 5, s));
        }
        var x_bf_host: [IN]f32 = undefined;
        try testReadF32(x_bf, &x_bf_host, s);

        const inds_host = [_]u32{ 1, 3 };
        var inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inds);
        {
            const sh = [_]c_int{ 1, 1, K };
            inds = mlx.mlx_array_new_data(&inds_host, &sh, 3, .uint32);
        }

        // Ground truth: gt[k*OUT + o] = sum_in x[in] * w_deq[inds[k]][o][in].
        var gt: [K * OUT]f32 = undefined;
        for (0..K) |k| {
            const e = inds_host[k];
            for (0..OUT) |o| {
                var acc: f32 = 0;
                for (0..IN) |in| acc += x_bf_host[in] * w_deq_host[e * OUT * IN + o * IN + in];
                gt[k * OUT + o] = acc;
            }
        }

        // Dense path.
        var dense5 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense5);
        try gatherExpertMm(&dense5, x_exp, w_deq_t, null_sc, null_bi, no_idx, inds, bits, gs, .affine, false, s);
        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_squeeze(&dense, dense5, s)); // [K, OUT]
        const dense_host = try alloc.alloc(f32, K * OUT);
        defer alloc.free(dense_host);
        try testReadF32(dense, dense_host, s);

        // Quantized path (cross-check).
        var quant5 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant5);
        try gatherExpertMm(&quant5, x_exp, q_w, q_sc, q_bi, no_idx, inds, bits, gs, .affine, false, s);
        var quant = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant);
        try mlx.check(mlx.mlx_squeeze(&quant, quant5, s));
        const quant_host = try alloc.alloc(f32, K * OUT);
        defer alloc.free(quant_host);
        try testReadF32(quant, quant_host, s);

        for (0..K * OUT) |i| {
            try testing.expectApproxEqAbs(gt[i], dense_host[i], 2e-2); // dense == ground truth
            try testing.expectApproxEqAbs(gt[i], quant_host[i], 2e-2); // quant agrees too
        }
    }

    // ── Prefill/sorted shape: x_rep [N,1,IN], sorted_inds [N], sorted=true ──
    {
        const N = 5;
        var x_host: [N * IN]f32 = undefined;
        for (&x_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3))) * 0.07;
        var x_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_f32);
        {
            const sh = [_]c_int{ N, 1, IN };
            x_f32 = mlx.mlx_array_new_data(&x_host, &sh, 3, .float32);
        }
        var x_rep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_rep);
        try mlx.check(mlx.mlx_astype(&x_rep, x_f32, .bfloat16, s));
        var x_bf_host: [N * IN]f32 = undefined;
        try testReadF32(x_rep, &x_bf_host, s);

        const sorted_host = [_]u32{ 0, 0, 1, 2, 3 }; // sorted experts
        var sorted_inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted_inds);
        {
            const sh = [_]c_int{N};
            sorted_inds = mlx.mlx_array_new_data(&sorted_host, &sh, 1, .uint32);
        }

        // Ground truth: gt[i*OUT + o] = sum_in x[i][in] * w_deq[sorted[i]][o][in].
        var gt: [N * OUT]f32 = undefined;
        for (0..N) |i| {
            const e = sorted_host[i];
            for (0..OUT) |o| {
                var acc: f32 = 0;
                for (0..IN) |in| acc += x_bf_host[i * IN + in] * w_deq_host[e * OUT * IN + o * IN + in];
                gt[i * OUT + o] = acc;
            }
        }

        var dense3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense3);
        try gatherExpertMm(&dense3, x_rep, w_deq_t, null_sc, null_bi, no_idx, sorted_inds, bits, gs, .affine, true, s);
        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_squeeze(&dense, dense3, s)); // [N, OUT]
        const dense_host = try alloc.alloc(f32, N * OUT);
        defer alloc.free(dense_host);
        try testReadF32(dense, dense_host, s);

        var quant3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant3);
        try gatherExpertMm(&quant3, x_rep, q_w, q_sc, q_bi, no_idx, sorted_inds, bits, gs, .affine, true, s);
        var quant = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant);
        try mlx.check(mlx.mlx_squeeze(&quant, quant3, s));
        const quant_host = try alloc.alloc(f32, N * OUT);
        defer alloc.free(quant_host);
        try testReadF32(quant, quant_host, s);

        for (0..N * OUT) |i| {
            try testing.expectApproxEqAbs(gt[i], dense_host[i], 2e-2);
            try testing.expectApproxEqAbs(gt[i], quant_host[i], 2e-2);
        }
    }
}

test "qwen4 expert streaming targets only the trunk bank" {
    var config = ModelConfig{ .model_type = "qwen4_exp", .weight_prefix = "language_model.model", .expert_streaming = true };
    try std.testing.expect(shouldStreamExpertBank(&config, "language_model.model"));
    try std.testing.expect(!shouldStreamExpertBank(&config, "language_model.mtp"));
    config.expert_streaming = false;
    try std.testing.expect(!shouldStreamExpertBank(&config, "language_model.model"));
    var moe = ModelConfig{ .model_type = "not_a_served_arch", .weight_prefix = "language_model.model", .expert_streaming = true };
    try std.testing.expect(!shouldStreamExpertBank(&moe, "language_model.model"));
}

test "qmatmulBits: prefill-width affine calls take the dequant+GEMM route (engaged + no worse than stock)" {
    const s = mlx.gpuStream();
    const al = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xD0DE);
    const rnd = prng.random();
    const K: c_int = 256;
    const N: c_int = 384;
    const M: c_int = @intCast(PREFILL_DQ_GEMM_MIN_M);

    // Random bf16 weight quantized to affine q4 gs64 (the 27B trunk class).
    const wn: usize = @intCast(N * K);
    const wbuf = try al.alloc(f32, wn);
    for (wbuf) |*v| v.* = bf16Trunc(rnd.float(f32) - 0.5);
    const wshape = [_]c_int{ N, K };
    const w32 = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 2, .float32);
    al.free(wbuf);
    defer _ = mlx.mlx_array_free(w32);
    var wb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wb);
    try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var wsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsc);
    var wbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

    const xn: usize = @intCast(M * K);
    const xbuf = try al.alloc(f32, xn);
    for (xbuf) |*v| v.* = bf16Trunc(rnd.float(f32) - 0.5);
    const xshape = [_]c_int{ 1, M, K };
    const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
    al.free(xbuf);
    defer _ = mlx.mlx_array_free(x32);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

    // f32 ground truth: fp32 dequant + fp32 GEMM.
    var wdq32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wdq32);
    try mlx.check(mlx.mlx_dequantize(&wdq32, wq, wsc, wbi, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = .float32, .has_value = true }, s));
    var wdq32_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wdq32_t);
    try mlx.check(mlx.mlx_transpose(&wdq32_t, wdq32, s));
    var gt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gt);
    try mlx.check(mlx.mlx_matmul(&gt, x32, wdq32_t, s));
    try mlx.check(mlx.mlx_array_eval(gt));
    const gtd = mlx.mlx_array_data_float32(gt) orelse return error.InvalidDtype;
    const on: usize = @intCast(M * N);

    // Stock qmm (route forced off).
    prefill_dq_gemm_override = false;
    const stock = try qmatmulBits(x, wq, wsc, wbi, 4, 64, .affine, s);
    defer _ = mlx.mlx_array_free(stock);
    const stock_f = try evalToF32(al, stock, on, s);
    defer al.free(stock_f);

    // Routed (default on): engagement must be COUNTED.
    prefill_dq_gemm_override = true;
    defer prefill_dq_gemm_override = null;
    const before = prefill_dq_gemm_engaged;
    const routed = try qmatmulBits(x, wq, wsc, wbi, 4, 64, .affine, s);
    defer _ = mlx.mlx_array_free(routed);
    try testing.expectEqual(before + 1, prefill_dq_gemm_engaged);
    const routed_f = try evalToF32(al, routed, on, s);
    defer al.free(routed_f);

    var stock_err: f32 = 0;
    var routed_err: f32 = 0;
    for (0..on) |i| {
        stock_err = @max(stock_err, @abs(stock_f[i] - gtd[i]));
        routed_err = @max(routed_err, @abs(routed_f[i] - gtd[i]));
    }
    // bf16-weight-rounding is the only extra error source — same magnitude
    // class as the stock kernel's own bf16 output rounding.
    try testing.expect(routed_err <= 1.5 * stock_err + 5e-3);

    // Decode/verify widths never route (M below the floor).
    var x8 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x8);
    {
        var x8_32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x8_32);
        const s8 = [_]c_int{ 0, 0, 0 };
        const e8 = [_]c_int{ 1, 8, K };
        const st8 = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&x8_32, x32, &s8, 3, &e8, 3, &st8, 3, s));
        try mlx.check(mlx.mlx_astype(&x8, x8_32, .bfloat16, s));
    }
    const small_before = prefill_dq_gemm_engaged;
    const small = try qmatmulBits(x8, wq, wsc, wbi, 4, 64, .affine, s);
    defer _ = mlx.mlx_array_free(small);
    try mlx.check(mlx.mlx_array_eval(small));
    try testing.expectEqual(small_before, prefill_dq_gemm_engaged);
}

/// Max |a - ref| as f32, for the gatherQmv no-worse-than-reference parity tests.
fn gqmvMaxAbsErr(a: mlx.mlx_array, ref: mlx.mlx_array, str: mlx.mlx_stream) !f32 {
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    try mlx.check(mlx.mlx_astype(&a32, a, .float32, str));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, a32, ref, str));
    var ad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ad);
    try mlx.check(mlx.mlx_abs(&ad, d, str));
    var mx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx);
    try mlx.check(mlx.mlx_max(&mx, ad, false, str));
    try mlx.check(mlx.mlx_array_eval(mx));
    var out: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&out, mx));
    return out;
}

test "gatherQmv is no worse than stock gather_qmm vs fp32 dequant ground truth" {
    try mlx.requireGpu();
    // Per the kernel-testing rule: assert NO-WORSE-THAN-REFERENCE against fp32
    // dequant ground truth, never kernel-vs-kernel agreement (both round to
    // bf16 in different orders, so exact agreement is not a real invariant).
    // Covers BOTH input layouts: the shared-x gate/up projection and the
    // per-expert-x down projection.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x9E3779B9);
    const rnd = prng.random();

    const E: c_int = 8;
    const N: c_int = 64;
    const K: c_int = 256;
    const TOPK: c_int = 3;

    for ([_]struct { bits: u32, gs: u32 }{
        .{ .bits = 2, .gs = 64 },
        .{ .bits = 3, .gs = 64 },
        .{ .bits = 3, .gs = 128 },
        .{ .bits = 4, .gs = 64 },
        .{ .bits = 8, .gs = 32 },
    }) |cfg| {
        // fp32 is the LIVE case: expert_x arrives as float32 while the
        // bank's scales/biases are bf16. A bf16-only test missed it entirely.
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |xdt| {
            for ([_]bool{ false, true }) |x_per_expert| {
                // ── quantized bank [E,N,K] ──
                const wb = try testRandWeightBf16(rnd, &[_]c_int{ E, N, K }, s);
                defer _ = mlx.mlx_array_free(wb);
                var triple = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(triple);
                try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(cfg.gs)), mlx.mlx_optional_int.some(@intCast(cfg.bits)), "affine", .{}, s));
                var wq = mlx.mlx_array_new();
                var wsc = mlx.mlx_array_new();
                var wbi = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(wq);
                defer _ = mlx.mlx_array_free(wsc);
                defer _ = mlx.mlx_array_free(wbi);
                try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
                try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
                try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

                // ── x: [K] shared, or [TOPK,K] per expert ──
                const xrows: c_int = if (x_per_expert) TOPK else 1;
                const xcnt: usize = @intCast(xrows * K);
                const xbuf = try allocator.alloc(f32, xcnt);
                defer allocator.free(xbuf);
                for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
                const xsh = [_]c_int{ xrows, K };
                const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xsh, 2, .float32);
                defer _ = mlx.mlx_array_free(x32);
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_astype(&x, x32, xdt, s));

                const idxvals = [_]u32{ 5, 0, 3 };
                const idxsh = [_]c_int{TOPK};
                const inds = mlx.mlx_array_new_data(&idxvals, &idxsh, 1, .uint32);
                defer _ = mlx.mlx_array_free(inds);

                // ── fp32 ground truth: dequantize the bank, take the 3 experts,
                // and do the matmul entirely in float32.
                var deq = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(deq);
                try mlx.check(mlx.mlx_dequantize(&deq, wq, wsc, wbi, mlx.mlx_optional_int.some(@intCast(cfg.gs)), mlx.mlx_optional_int.some(@intCast(cfg.bits)), "affine", .{}, .{}, s));
                var deq32 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(deq32);
                try mlx.check(mlx.mlx_astype(&deq32, deq, .float32, s));
                var sel = mlx.mlx_array_new(); // [TOPK,N,K]
                defer _ = mlx.mlx_array_free(sel);
                try mlx.check(mlx.mlx_take_axis(&sel, deq32, inds, 0, s));
                var selT = mlx.mlx_array_new(); // [TOPK,K,N]
                defer _ = mlx.mlx_array_free(selT);
                const tax = [_]c_int{ 0, 2, 1 };
                try mlx.check(mlx.mlx_transpose_axes(&selT, sel, &tax, 3, s));
                // x as [TOPK,1,K] (broadcast the shared row when needed)
                var x32r = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x32r);
                const x3 = [_]c_int{ xrows, 1, K };
                try mlx.check(mlx.mlx_reshape(&x32r, x32, &x3, 3, s));
                var x32b = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x32b);
                const xb3 = [_]c_int{ TOPK, 1, K };
                try mlx.check(mlx.mlx_broadcast_to(&x32b, x32r, &xb3, 3, s));
                var truth = mlx.mlx_array_new(); // [TOPK,1,N]
                defer _ = mlx.mlx_array_free(truth);
                try mlx.check(mlx.mlx_matmul(&truth, x32b, selT, s));
                const tflat = [_]c_int{ TOPK, N };
                var truth2 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(truth2);
                try mlx.check(mlx.mlx_reshape(&truth2, truth, &tflat, 2, s));

                // ── stock gather_qmm, same layouts ──
                var xg = mlx.mlx_array_new(); // [TOPK,1,K]
                defer _ = mlx.mlx_array_free(xg);
                {
                    var xr = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(xr);
                    try mlx.check(mlx.mlx_reshape(&xr, x, &x3, 3, s));
                    try mlx.check(mlx.mlx_broadcast_to(&xg, xr, &xb3, 3, s));
                }
                var stock3 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(stock3);
                const lhs3 = [_]u32{ 0, 1, 2 };
                const lhs_used = if (x_per_expert) blk: {
                    break :blk mlx.mlx_array_new_data(&lhs3, &idxsh, 1, .uint32);
                } else blk: {
                    const zeros = [_]u32{ 0, 0, 0 };
                    break :blk mlx.mlx_array_new_data(&zeros, &idxsh, 1, .uint32);
                };
                defer _ = mlx.mlx_array_free(lhs_used);
                try mlx.check(mlx.mlx_gather_qmm(&stock3, xg, wq, wsc, wbi, lhs_used, inds, true, mlx.mlx_optional_int.some(@intCast(cfg.gs)), mlx.mlx_optional_int.some(@intCast(cfg.bits)), "affine", false, s));
                var stock2 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(stock2);
                try mlx.check(mlx.mlx_reshape(&stock2, stock3, &tflat, 2, s));

                // ── ours ──
                const maybe = try gatherQmv(s, x, wq, wsc, wbi, inds, cfg.bits, cfg.gs, .affine, x_per_expert);
                try testing.expect(maybe != null);
                const ours = maybe.?;
                defer _ = mlx.mlx_array_free(ours);

                const err_ours = try gqmvMaxAbsErr(ours, truth2, s);
                const err_stock = try gqmvMaxAbsErr(stock2, truth2, s);
                // No worse than stock against the same fp32 ground truth (small
                // slack for bf16 accumulation-order differences).
                testing.expect(err_ours <= err_stock * 1.05 + 1e-3) catch |e| {
                    std.debug.print("\n[gather-qmv] bits={d} gs={d} xdt={any} per_expert_x={} ours_err={d:.6} stock_err={d:.6}\n", .{ cfg.bits, cfg.gs, xdt, x_per_expert, err_ours, err_stock });
                    return e;
                };
            }
        }
    }
}

test "appendLinearAttnWeights skips fields with null ctx (plain bf16 layers)" {
    const s = mlx.gpuStream();
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);

    // Real arrays for the 9 non-scale/bias fields so they have non-null ctx.
    const sh = [_]c_int{1};
    var arrs: [9]mlx.mlx_array = undefined;
    for (&arrs) |*a| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(a, &sh, 1, .bfloat16, s));
    }
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };

    // Simulate the UD layout: weights set, scales/biases null.
    const la: LinearAttnWeights = .{
        .qkv_w = arrs[0],
        .qkv_s = mlx.mlx_array{ .ctx = null },
        .qkv_b = mlx.mlx_array{ .ctx = null },
        .z_w = arrs[1],
        .z_s = mlx.mlx_array{ .ctx = null },
        .z_b = mlx.mlx_array{ .ctx = null },
        .a_w = arrs[2],
        .a_s = mlx.mlx_array{ .ctx = null },
        .a_b = mlx.mlx_array{ .ctx = null },
        .b_w = arrs[3],
        .b_s = mlx.mlx_array{ .ctx = null },
        .b_b = mlx.mlx_array{ .ctx = null },
        .conv1d_w = arrs[4],
        .A_log = arrs[5],
        .dt_bias = arrs[6],
        .norm_w = arrs[7],
        .out_w = arrs[8],
        .out_s = mlx.mlx_array{ .ctx = null },
        .out_b = mlx.mlx_array{ .ctx = null },
    };

    appendLinearAttnWeights(vec, &la);

    // Of 19 mlx_array fields: 5 weights (qkv/z/a/b/out) + 4 SSM bits
    // (conv1d/A_log/dt_bias/norm_w) = 9 expected. The 10 null-ctx scales/biases
    // are skipped — confirms the optional-bf16 path doesn't poison the eval batch.
    try testing.expectEqual(@as(usize, 9), mlx.mlx_vector_array_size(vec));
}

test "appendHybridMlpWeights skips MoE fields with null ctx (UD MoE bf16 router/SEG)" {
    const s = mlx.gpuStream();
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);

    // Real arrays for every weight (`*_w`) — non-null ctx. Quantized projections
    // also have real scales/biases. UD bf16 layers (router, shared_expert_gate)
    // get null-ctx scales/biases.
    const sh = [_]c_int{1};
    var arrs: [16]mlx.mlx_array = undefined;
    for (&arrs) |*a| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(a, &sh, 1, .bfloat16, s));
    }
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };

    // Unsloth-dynamic MoE layout: router + shared_expert_gate are bf16 (null s/b);
    // routed experts (switch_*) and shared_expert (shared_*) stay quantized.
    const mw: MoeMlpWeights = .{
        .router_w = arrs[0],
        .router_s = mlx.mlx_array{ .ctx = null }, // UD bf16
        .router_b = mlx.mlx_array{ .ctx = null }, // UD bf16
        .switch_gate_w = arrs[1],
        .switch_gate_s = arrs[2],
        .switch_gate_b = arrs[3],
        .switch_up_w = arrs[4],
        .switch_up_s = arrs[5],
        .switch_up_b = arrs[6],
        .switch_down_w = arrs[7],
        .switch_down_s = arrs[8],
        .switch_down_b = arrs[9],
        .shared_gate_w = arrs[10],
        .shared_gate_s = arrs[11],
        .shared_gate_b = arrs[12],
        .shared_up_w = arrs[13],
        .shared_up_s = arrs[14],
        .shared_up_b = arrs[15],
        .shared_down_w = arrs[0], // reuse — only ctx-null check matters here
        .shared_down_s = arrs[1],
        .shared_down_b = arrs[2],
        .shared_expert_gate_w = arrs[3],
        .shared_expert_gate_s = mlx.mlx_array{ .ctx = null }, // UD bf16
        .shared_expert_gate_b = mlx.mlx_array{ .ctx = null }, // UD bf16
    };
    const hw: HybridMlpWeights = .{ .moe = mw };

    appendHybridMlpWeights(vec, &hw);

    // Counted by hand: the non-optional `mlx.mlx_array` fields minus the 2 null-ctx
    // ones (router_s, router_b) → 19 appended. Plus the optional `?mlx.mlx_array`
    // fields: shared_expert_gate_w is Some(real) → +1; SEG scales/biases are
    // Some(null-ctx) → +0 each. Total: 19 + 1 = 20.
    try testing.expectEqual(@as(usize, 20), mlx.mlx_vector_array_size(vec));
}

test "appendHybridMlpWeights skips dense fields with null ctx (UD dense bf16)" {
    const s = mlx.gpuStream();
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);

    const sh = [_]c_int{1};
    var arrs: [3]mlx.mlx_array = undefined;
    for (&arrs) |*a| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(a, &sh, 1, .bfloat16, s));
    }
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };

    // All-bf16 dense MLP: weights set, scales/biases null.
    const dw: DenseMlpWeights = .{
        .gate_w = arrs[0],
        .gate_s = mlx.mlx_array{ .ctx = null },
        .gate_b = mlx.mlx_array{ .ctx = null },
        .up_w = arrs[1],
        .up_s = mlx.mlx_array{ .ctx = null },
        .up_b = mlx.mlx_array{ .ctx = null },
        .down_w = arrs[2],
        .down_s = mlx.mlx_array{ .ctx = null },
        .down_b = mlx.mlx_array{ .ctx = null },
    };
    const hw: HybridMlpWeights = .{ .dense = dw };

    appendHybridMlpWeights(vec, &hw);

    // 9 fields, 3 weights non-null + 6 null-ctx scales/biases skipped → 3.
    try testing.expectEqual(@as(usize, 3), mlx.mlx_vector_array_size(vec));
}

test "moeRoutingChain produces top-K indices and renormalized softmax weights" {
    const s = mlx.gpuStream();

    // Two rows of router logits over 6 experts. Top-2 of each row is unambiguous:
    //   row 0: experts {0, 3} (logits 10 and 5)
    //   row 1: experts {1, 4} (logits 10 and 5)
    const n_rows: c_int = 2;
    const n_exp: c_int = 6;
    const k: c_int = 2;
    const data = [_]f32{
        10.0, 0.0,  0.0, 5.0, 0.0, 0.0,
        0.0,  10.0, 0.0, 0.0, 5.0, 0.0,
    };
    const shape = [_]c_int{ n_rows, n_exp };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const routed = try moeRoutingChain(logits, k, s);
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.norm_scores);

    // Shape check (cheap, no data read needed).
    {
        const inds_shape = mlx.getShape(routed.inds);
        try testing.expectEqual(@as(usize, 2), inds_shape.len);
        try testing.expectEqual(n_rows, inds_shape[0]);
        try testing.expectEqual(k, inds_shape[1]);
        const sc_shape = mlx.getShape(routed.norm_scores);
        try testing.expectEqual(@as(usize, 2), sc_shape.len);
        try testing.expectEqual(n_rows, sc_shape[0]);
        try testing.expectEqual(k, sc_shape[1]);
    }

    // To verify top-K correctness without reading non-contiguous slice memory
    // directly, gather the original logits at the selected indices: gathered[i,j]
    // == logits[i, inds[i,j]]. Then sum across K — for our fixture, the top-2
    // logits in each row are {10, 5}, so the per-row sum must be 15.
    var gathered = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered);
    try mlx.check(mlx.mlx_take_along_axis(&gathered, logits, routed.inds, -1, s));
    var gathered_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered_sum);
    try mlx.check(mlx.mlx_sum_axis(&gathered_sum, gathered, -1, false, s));

    // norm_scores must sum to 1 along K (verifies the renormalize step).
    var scores_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_sum);
    try mlx.check(mlx.mlx_sum_axis(&scores_sum, routed.norm_scores, -1, false, s));

    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, gathered_sum);
        _ = mlx.mlx_vector_array_append_value(ev, scores_sum);
        try mlx.check(mlx.mlx_eval(ev));
    }

    // gathered_sum and scores_sum are 1D outputs of sum_axis (contiguous).
    const gs = mlx.mlx_array_data_float32(gathered_sum) orelse return error.InvalidDtype;
    const ss = mlx.mlx_array_data_float32(scores_sum) orelse return error.InvalidDtype;
    const tol: f32 = 1e-3;
    try testing.expect(@abs(gs[0] - 15.0) < tol);
    try testing.expect(@abs(gs[1] - 15.0) < tol);
    try testing.expect(@abs(ss[0] - 1.0) < tol);
    try testing.expect(@abs(ss[1] - 1.0) < tol);
}

/// `[1, heads, len, dim]` of distinct, position-dependent values — two caches
/// fed the same chunk must return byte-identical views, so only distinctness
/// matters, not the numbers.
pub fn swaRingChunk(s: mlx.mlx_stream, start: c_int, len: c_int, heads: c_int, dim: c_int) !mlx.mlx_array {
    const n: f64 = @floatFromInt(heads * len * dim);
    const base: f64 = @floatFromInt(start * heads * dim);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_arange(&flat, base, base + n, 1.0, .float32, s));
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    const inv = mlx.mlx_array_new_float(1.0 / 4096.0);
    defer _ = mlx.mlx_array_free(inv);
    try mlx.check(mlx.mlx_multiply(&scaled, flat, inv, s));
    var bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bf);
    try mlx.check(mlx.mlx_astype(&bf, scaled, .bfloat16, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const shape = [_]c_int{ 1, heads, len, dim };
    try mlx.check(mlx.mlx_reshape(&out, bf, &shape, 4, s));
    return out;
}

/// Layer 0 global, layer 1 sliding: rows `[from, to)` in `chunk`-wide forwards.
pub fn ringCheckpointFeed(a: *KVCache, b: *KVCache, window: u32, from: usize, to: usize, chunk: usize) !void {
    const s = mlx.gpuStream();
    var pos = from;
    while (pos < to) {
        const q: c_int = @intCast(@min(chunk, to - pos));
        const k = try swaRingChunk(s, @intCast(pos), q, 1, 64);
        defer _ = mlx.mlx_array_free(k);
        const v = try swaRingChunk(s, @as(c_int, @intCast(pos)) + 7919, q, 1, 64);
        defer _ = mlx.mlx_array_free(v);
        for ([_]u32{ 0, slidingTailSpan(window, @intCast(q), SLIDING_TRIM_UNBOUNDED) }, 0..) |span, li| {
            var av = try a.update(@intCast(li), k, v, s, span);
            defer av.deinit();
            var bv = try b.update(@intCast(li), k, v, s, span);
            defer bv.deinit();
            try std.testing.expectEqual(@as(f32, 0), try maxAbsDiffF32(av.k, bv.k, s));
            try std.testing.expectEqual(@as(f32, 0), try maxAbsDiffF32(av.v, bv.v, s));
        }
        pos += @intCast(q);
    }
}

/// Max |a-b| between two same-shaped float arrays, with a NaN guard (a NaN
/// candidate must FAIL, never score 0 — the parity-loop finiteness rule).
pub fn maxAbsDiffF32(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    // materializedOwnedCopy: a same-dtype astype is a NO-OP view and a slice's
    // raw data pointer walks the PARENT's physical layout — force a contiguous
    // owned buffer before touching bytes.
    var a_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_t);
    try mlx.check(mlx.mlx_astype(&a_t, a, .float32, s));
    const a32 = try materializedOwnedCopy(s, a_t);
    defer _ = mlx.mlx_array_free(a32);
    var b_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b_t);
    try mlx.check(mlx.mlx_astype(&b_t, b, .float32, s));
    const b32 = try materializedOwnedCopy(s, b_t);
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    const n = mlx.mlx_array_size(a32);
    if (n != mlx.mlx_array_size(b32)) return error.ShapeMismatch;
    const pa = mlx.mlx_array_data_float32(a32) orelse return error.NoData;
    const pb = mlx.mlx_array_data_float32(b32) orelse return error.NoData;
    var max_err: f32 = 0;
    for (0..n) |i| {
        const d = @abs(pa[i] - pb[i]);
        if (!std.math.isFinite(d)) return error.NonFiniteValue;
        if (d > max_err) max_err = d;
    }
    return max_err;
}

// Helper for the GDN seq-kernel parity test below: run the recurrence via the
// requested kernel variant and return the FINAL state [B,Hv,Dv,Dk] as f32.

// Like gdnTestRun(seq=true) but returns the state at an arbitrary position `pos`
// (not just the last) — mirrors what `ssmRollbackFromCapture` slices out.

// ── GDN blocked-seq kernel test harness ──

/// Assert a verify-kernel output is NO LESS ACCURATE than the stock qmm it
/// replaces, both measured against `wdq_t` — the SAME 4-bit weights dequantized
/// to fp32 (so quantization error cancels and only the two kernels' own
/// arithmetic is under test).
///
/// This is the machine-independent invariant. A direct kernel-vs-stock bound is
/// NOT: both paths accumulate fp32 in a different order and round to bf16, so on
/// heavily-cancelling data each already sits ~0.03 from truth on its worst
/// element and they may legitimately differ from each other by more than that.
/// Tuning a pair-agreement threshold on one GPU produces a test that is green on
/// the dev machine and red on the CI runner while the kernel is bit-for-bit as
/// accurate (measured: kernel-vs-truth == stock-vs-truth to four decimals).
/// A genuine defect — a partial-sum race, a register spill, bad indexing —
/// pushes kernel-vs-truth far past stock-vs-truth and is still caught here.
/// Parity bars for a verify lane, stated so that no bar is fit per GPU.
///
/// A per-chip max-error slack was tried first (M1 Pro measured kern_max 0.0313
/// against stock 0.0039 while cosine stayed at 0.999998, so the bar was
/// widened to 0.04 there). That is the wrong shape of guard twice over:
///
///   - Widening the MAX blinds the test to what it exists to catch. Cosine is
///     a GLOBAL measure, so a partial-sum race, a register spill or a bad
///     index — one element far off, everything else intact — passes a widened
///     max with cosine unmoved.
///   - `stock_max` is not a stable reference. The same shapes measure
///     stock_max 0.02-0.035 on M4 Max, so the M1 Pro row was not a lane
///     landing 8x worse than normal, it was STOCK landing unusually well
///     there. Anything keyed to stock's worst element inherits that.
///
/// So neither bar references stock's max any more:
///
///   - `GROSS_CEILING` is an ABSOLUTE ceiling on the worst element. On the
///     clamped-relative metric below, a bf16 output's own rounding noise sits
///     at 0.02-0.035 on every machine measured; a corrupted accumulator lands
///     orders of magnitude past 0.5. This is the single-element bar.
///   - `RMS_FACTOR` bounds the whole error distribution against stock's,
///     which is scale-free: when stock is unusually accurate the ratio is
///     what stays meaningful. This is the systematic bar. (M1 Pro's cosines
///     imply a ratio near 1.4; M4 Max measures 1.00 across every lane.)
///   - The cosine bars are unchanged and catch a broken reference.
const VerifyQmmParity = struct {
    /// Absolute ceiling on the worst element's clamped relative error.
    const GROSS_CEILING: f64 = 0.5;
    /// How much of stock's RMS error the lane may carry.
    const RMS_FACTOR: f64 = 3.0;
    /// Cosine may not fall below stock's by more than this.
    const COS_SLACK: f64 = 1e-5;
    /// …and both must actually be tracking the truth.
    const COS_FLOOR: f64 = 0.999;
};

/// Why a verify-lane parity check failed, or null when it passed. Pure so the
/// decision itself is unit-testable without a GPU — each arm is a distinct
/// failure shape the bars are meant to separate.
const VerifyQmmParityFail = enum { gross_element, systematic, cosine, not_tracking };

fn verifyQmmParityVerdict(
    kern_max: f64,
    rms_stock: f64,
    rms_kern: f64,
    cos_stock: f64,
    cos_kern: f64,
) ?VerifyQmmParityFail {
    if (cos_kern < VerifyQmmParity.COS_FLOOR) return .not_tracking;
    if (cos_kern < cos_stock - VerifyQmmParity.COS_SLACK) return .cosine;
    if (kern_max > VerifyQmmParity.GROSS_CEILING) return .gross_element;
    // A zero stock RMS means the reference is exact on this data; the ratio
    // then demands the lane be exact too, which is the strictest reading and
    // the one we want — an exactly-representable case a lane cannot reproduce
    // is a defect, not reduction-order noise.
    if (rms_kern > VerifyQmmParity.RMS_FACTOR * rms_stock) return .systematic;
    return null;
}

fn expectVerifyQmmNoWorseThanStock(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    wq: mlx.mlx_array,
    wsc: mlx.mlx_array,
    wbi: mlx.mlx_array,
    bits: u32,
    gs: u32,
    wdq_t: mlx.mlx_array,
    got: mlx.mlx_array,
    label: []const u8,
) !void {
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_quantized_matmul(&ref, x, wq, wsc, wbi, true, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", s));

    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    var truth = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(truth);
    try mlx.check(mlx.mlx_matmul(&truth, xf, wdq_t, s));

    var tf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tf);
    var rf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rf);
    var gf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gf);
    try mlx.check(mlx.mlx_astype(&tf, truth, .float32, s));
    try mlx.check(mlx.mlx_astype(&rf, ref, .float32, s));
    try mlx.check(mlx.mlx_astype(&gf, got, .float32, s));
    try mlx.check(mlx.mlx_array_eval(tf));
    try mlx.check(mlx.mlx_array_eval(rf));
    try mlx.check(mlx.mlx_array_eval(gf));
    const td = mlx.mlx_array_data_float32(tf).?;
    const rd = mlx.mlx_array_data_float32(rf).?;
    const gd = mlx.mlx_array_data_float32(gf).?;

    const gsh = mlx.getShape(got);
    const count: usize = @intCast(gsh[1] * gsh[2]);
    var stock_max: f64 = 0;
    var kern_max: f64 = 0;
    var dot_s: f64 = 0;
    var dot_k: f64 = 0;
    var nt: f64 = 0;
    var ns: f64 = 0;
    var nk: f64 = 0;
    // First pass: cosines against the fp32 truth, plus each side's worst
    // element. Neither max is a BAR any more (see VerifyQmmParity) — stock's
    // is reported alongside the kernel's so a failure line shows both.
    for (0..count) |i| {
        const t: f64 = td[i];
        const r: f64 = rd[i];
        const g: f64 = gd[i];
        // Clamped denominator: a near-zero output of a heavily-cancelling dot
        // product must not manufacture a huge ratio from a tiny absolute error.
        const denom = @max(1.0, @abs(t));
        stock_max = @max(stock_max, @abs(r - t) / denom);
        kern_max = @max(kern_max, @abs(g - t) / denom);
        dot_s += r * t;
        dot_k += g * t;
        nt += t * t;
        ns += r * r;
        nk += g * g;
    }
    var se_s: f64 = 0;
    var se_k: f64 = 0;
    for (0..count) |i| {
        const t: f64 = td[i];
        const denom = @max(1.0, @abs(t));
        const es = (@as(f64, rd[i]) - t) / denom;
        const ek = (@as(f64, gd[i]) - t) / denom;
        se_s += es * es;
        se_k += ek * ek;
    }
    const n_f: f64 = @floatFromInt(count);
    const rms_stock = @sqrt(se_s / n_f);
    const rms_kern = @sqrt(se_k / n_f);
    const cos_stock = dot_s / (@sqrt(ns) * @sqrt(nt));
    const cos_kern = dot_k / (@sqrt(nk) * @sqrt(nt));

    if (diagEnvOn("SUSHI_VQMM_PARITY_DEBUG")) {
        std.debug.print(
            "[vqmm-parity] {s:>16}: n={d} max {d:.5}/{d:.5} rms {d:.6}/{d:.6} (ratio {d:.3}) cos {d:.7}/{d:.7}\n",
            .{ label, count, stock_max, kern_max, rms_stock, rms_kern, if (rms_stock > 0) rms_kern / rms_stock else 0, cos_stock, cos_kern },
        );
    }

    if (verifyQmmParityVerdict(kern_max, rms_stock, rms_kern, cos_stock, cos_kern)) |why| {
        std.debug.print(
            "verifyQmm parity FAIL [{s}] ({s}): n={d} kernel max={d:.4} rms={d:.6} cos={d:.6}; " ++
                "stock max={d:.4} rms={d:.6} cos={d:.6}; bars: max<={d:.2} rms<={d:.1}x cos>=stock-{d}\n",
            .{
                label,                      @tagName(why),
                count,                      kern_max,
                rms_kern,                   cos_kern,
                stock_max,                  rms_stock,
                cos_stock,                  VerifyQmmParity.GROSS_CEILING,
                VerifyQmmParity.RMS_FACTOR, VerifyQmmParity.COS_SLACK,
            },
        );
        return error.TestExpectedApproxEq;
    }
}

test "vqmm Metal sources are byte-identical to the shipped text" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var sha = std.crypto.hash.sha2.Sha256.init(.{});
    var bytes: usize = 0;
    for (2..8) |m| {
        for (VQMM_BN_CHOICES) |bn| {
            const src = try verifyQmmSource(arena.allocator(), m, @intCast(bn));
            sha.update(src);
            bytes += src.len;
        }
    }
    for (2..8) |m| {
        const src = try verifyQmmMsgSource(arena.allocator(), m, @intCast(vqmmMsgBn(@intCast(m))));
        sha.update(src);
        bytes += src.len;
    }
    try testing.expectEqual(@as(usize, 312_254), bytes);
    const digest = std.fmt.bytesToHex(sha.finalResult(), .lower);
    try testing.expectEqualStrings("484c91a3c6460e022df1f2a5f1f2db9d8630c50c760722b90cc4821bf66f9aea", &digest);
}

test "verifyQmmParityVerdict: reduction-order noise passes, a gross or systematic error does not" {
    // Reduction-order noise: a worse worst-element than stock, cosine intact.
    // This is the M1 Pro measurement (kern_max 0.0313 while stock landed an
    // unusually good 0.0039 at cos 0.999998) and it must PASS with no
    // per-chip bar — the max is no longer judged against stock's at all.
    try testing.expect(verifyQmmParityVerdict(0.0313, 0.0009, 0.0013, 0.999999, 0.999998) == null);
    // M4 Max: every shipped lane measures ratio 1.00.
    try testing.expect(verifyQmmParityVerdict(0.0308, 0.0012, 0.0012, 0.999998, 0.999998) == null);
    // One accumulator corrupted: a single element, rms and cosine still fine
    // — the exact shape a max-slack widened to 0.04 would have waved through.
    try testing.expectEqual(
        VerifyQmmParityFail.gross_element,
        verifyQmmParityVerdict(12.0, 0.0012, 0.0013, 0.999999, 0.999997).?,
    );
    // Systematically noisier without a single gross element: rms is the bar
    // that sees it, and it is scale-free so an accurate stock cannot hide it.
    try testing.expectEqual(
        VerifyQmmParityFail.systematic,
        verifyQmmParityVerdict(0.4, 0.0009, 0.0040, 0.999999, 0.999995).?,
    );
    try testing.expectEqual(
        VerifyQmmParityFail.cosine,
        verifyQmmParityVerdict(0.005, 0.0012, 0.0012, 0.999999, 0.99990).?,
    );
    try testing.expectEqual(
        VerifyQmmParityFail.not_tracking,
        verifyQmmParityVerdict(0.005, 0.0012, 0.0012, 0.5, 0.4).?,
    );
    // A lane BETTER than stock is never a failure.
    try testing.expect(verifyQmmParityVerdict(0.0001, 0.0012, 0.0002, 0.999998, 0.999999) == null);
}

test "verifyQmm: the plain-SIMD tiles match stock qmm at 5/6/8-bit affine too" {
    try mlx.requireGpu();
    // The split-K and msg tiles were 4-bit specializations by construction, so
    // every non-4-bit pack fell through to stock on the whole M1-M4 line — the
    // Qwen3.8-27B 6-bit pack (95.0% top-1 against bf16 vs the 4-bit pack's
    // 83.6%) paid the full lane-off penalty on every MTP round. The byte-
    // addressed unpack lifted from the NAX m16 tile fixes that; this pins it.
    //
    // Same bar as the 4-bit test: no worse than stock against an fp32 dequant
    // reference, never kernel-vs-kernel. Parity is asserted independently of
    // the PROFITABILITY predicate — `vqmm_mixed_plain_test_force` opens the
    // gate so a shape the measurement later declines is still proven correct.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xB175);
    const rnd = prng.random();

    vqmm_mixed_plain_test_force = true;
    defer vqmm_mixed_plain_test_force = null;
    // The NAX probe must not steal these rows on an M5 box: this test is about
    // the plain-SIMD lanes, and M 2..7 is below the NAX takeover anyway.
    vqmm_nax_probe_override = false;
    defer vqmm_nax_probe_override = null;

    const cases = [_]struct { k: c_int, n: c_int, gs: u32, label: []const u8 }{
        .{ .k = 5120, .n = 17408, .gs = 64, .label = "mlp" }, // MLP gate/up, split-K
        .{ .k = 2048, .n = 151936, .gs = 64, .label = "lm_head" }, // huge N, msg tile
        // The unpack is independent of the group size (gs only indexes the
        // scale/bias pair), but the forced-on A/B arm adopts every gs, so the
        // other two are proven rather than argued.
        .{ .k = 2048, .n = 5120, .gs = 32, .label = "gs32" },
        .{ .k = 2048, .n = 5120, .gs = 128, .label = "gs128" },
    };
    for ([_]u32{ 5, 6, 8 }) |bits| {
        for (cases) |cs| {
            const wb = try testRandWeightBf16(rnd, &[_]c_int{ cs.n, cs.k }, s);
            defer _ = mlx.mlx_array_free(wb);

            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(cs.gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
            var wq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wq);
            var wsc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wsc);
            var wbi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wbi);
            try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
            try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
            try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

            var wdq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wdq);
            try mlx.check(mlx.mlx_dequantize(&wdq, wq, wsc, wbi, mlx.mlx_optional_int.some(@intCast(cs.gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .has_value = true, .value = .float32 }, s));
            var wdq_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wdq_t);
            const t_axes = [_]c_int{ 1, 0 };
            try mlx.check(mlx.mlx_transpose_axes(&wdq_t, wdq, &t_axes, 2, s));

            for ([_]c_int{ 2, 4, 7 }) |m| {
                const xn: usize = @intCast(m * cs.k);
                const xbuf = try allocator.alloc(f32, xn);
                defer allocator.free(xbuf);
                for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
                const xshape = [_]c_int{ 1, m, cs.k };
                const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
                defer _ = mlx.mlx_array_free(x32);
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

                const got_opt = try verifyQmm(s, x, wq, wsc, wbi, bits, cs.gs);
                try testing.expect(got_opt != null);
                const got = got_opt.?;
                defer _ = mlx.mlx_array_free(got);
                const gsh = mlx.getShape(got);
                try testing.expectEqual(m, gsh[1]);
                try testing.expectEqual(cs.n, gsh[2]);
                try expectVerifyQmmNoWorseThanStock(s, x, wq, wsc, wbi, bits, cs.gs, wdq_t, got, cs.label);
            }

            // With the gate closed the same shape falls through to stock — the
            // profitability predicate is what decides adoption, and a lane that
            // cannot be declined is a lane that cannot be A/B'd.
            {
                vqmm_mixed_plain_test_force = false;
                defer vqmm_mixed_plain_test_force = true;
                const xbuf = try allocator.alloc(f32, @intCast(4 * cs.k));
                defer allocator.free(xbuf);
                for (xbuf) |*v| v.* = 0.1;
                const xshape = [_]c_int{ 1, 4, cs.k };
                const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
                defer _ = mlx.mlx_array_free(x32);
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
                try testing.expectEqual(@as(?mlx.mlx_array, null), try verifyQmm(s, x, wq, wsc, wbi, bits, cs.gs));
            }
        }
    }
}

test "mixedPlainShapeEnabled: adoption is what was MEASURED, per width and per shape" {
    const t = testing;
    // 4-bit is the original specialization and is always in.
    try t.expect(mixedPlainShapeEnabled(4, 64, 1024));

    // 6-bit is the measured win (1.10-1.22x on the Qwen3.8-27B 6-bit pack).
    try t.expect(mixedPlainShapeEnabled(6, 64, MIXED_PLAIN_MIN_N));
    try t.expect(mixedPlainShapeEnabled(6, 64, 17408));
    // Narrow projections (GDN ba, K/V) are where the byte-addressed unpack
    // loses to stock — the same shape-dependence mixedNaxShapeEnabled encodes,
    // and forcing them on gave back ~10 of the ~20 points.
    try t.expect(!mixedPlainShapeEnabled(6, 64, MIXED_PLAIN_MIN_N - 1));
    try t.expect(!mixedPlainShapeEnabled(6, 64, 1024));
    // Only gs 64 was measured.
    try t.expect(!mixedPlainShapeEnabled(6, 32, 17408));
    try t.expect(!mixedPlainShapeEnabled(6, 128, 17408));

    // 8-bit HAS a correct kernel arm and a parity test, and is still default
    // OFF: measured 1.01-1.04x, inside what this harness resolves. 5-bit has
    // the arm too but no pack on the measuring box, so it does not get to ride
    // 6-bit's number. Both are one env var from on — an unmeasured default is
    // the thing this predicate exists to prevent.
    for ([_]u32{ 5, 8 }) |b| {
        try t.expect(!mixedPlainShapeEnabled(b, 64, 17408));
    }
    // A width with no unpack arm is never adopted, whatever the shape.
    for ([_]u32{ 2, 3, 7 }) |b| try t.expect(!mixedPlainShapeEnabled(b, 64, 17408));
}

test "verifyQmm: split-K + msg + NAX verify-width kernels match stock qmm (4-bit affine)" {
    try mlx.requireGpu();
    // The spec-verify fast path: stock MLX qmm is tuned for M=1 (qmv) and
    // large M (steel); the 2..8-row verify shapes fall in a dead zone.
    // Port of MTPLX's split-K verify kernel family (verify_kernels.py,
    // Apache-2.0) — parity vs mlx_quantized_matmul at real 27B shapes.
    //
    // THE INVARIANT IS "NO LESS ACCURATE THAN STOCK", NOT "AGREES WITH STOCK".
    // Both paths accumulate in fp32 in a DIFFERENT ORDER and then round the
    // result to bf16, so the two outputs legitimately disagree by a few ULPs —
    // and on this data (K=5120 dot products of U(-0.5,0.5) → heavy cancellation)
    // each path already sits up to ~0.036 from the true value on its worst
    // element, purely from that bf16 output rounding. A direct kernel-vs-stock
    // bound therefore pins AGREEMENT tighter than bf16 permits: the original
    // `max_rel <= 0.02` / `cos > 0.99999` thresholds were tuned on one GPU and
    // the CI runner's GPU (different rounding, same correctness) failed them at
    // max_rel 0.0977 / cos 0.999988 — a green-here/red-there test that says
    // nothing about correctness. Measured: kernel-vs-truth equals stock-vs-truth
    // to four decimals on every shape.
    //
    // So both paths are measured against an fp32 DEQUANT REFERENCE (the same
    // 4-bit weights, dequantized, matmul'd in fp32 — the ground truth both are
    // approximating) and the kernel must be no worse than the stock kernel it
    // replaces. That is machine-independent, and a real defect (a partial-sum
    // race, a register spill, bad indexing) blows kernel-vs-truth far past
    // stock-vs-truth while a rounding-order difference cannot.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rnd = prng.random();

    const cases = [_]struct { k: c_int, n: c_int, gs: u32 }{
        .{ .k = 5120, .n = 17408, .gs = 64 }, // MLP gate/up
        .{ .k = 5120, .n = 1024, .gs = 32 }, // small-N above the gate, MTP gs-32 class
        .{ .k = 2048, .n = 5120, .gs = 128 }, // gs-128 arm
    };
    for (cases) |cs| {
        // Random dense weight → 4-bit affine triple.
        const wb = try testRandWeightBf16(rnd, &[_]c_int{ cs.n, cs.k }, s);
        defer _ = mlx.mlx_array_free(wb);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(cs.gs)), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        var wq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wq);
        var wsc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wsc);
        var wbi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wbi);
        try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

        // fp32 ground truth operand: the same 4-bit weights, dequantized.
        var wdq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq);
        try mlx.check(mlx.mlx_dequantize(&wdq, wq, wsc, wbi, mlx.mlx_optional_int.some(@intCast(cs.gs)), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .has_value = true, .value = .float32 }, s));
        var wdq_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq_t);
        const t_axes = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&wdq_t, wdq, &t_axes, 2, s));

        // Kernel rows: split-K M 2..7 on every machine; the NAX m16 tile rows
        // (M 8..16, the padding path at 8/9/12 and the exact-16 tile) run only
        // where the M5-class probe is live (todo-m5-nax.md §9.3) — the probe
        // self-gates, so plain-SIMD machines skip them.
        const simd_ms = [_]c_int{ 2, 3, 4, 5, 6, 7 };
        const nax_ms = [_]c_int{ 8, 9, 12, 16 };
        const total_ms: usize = if (verifyQmmNaxEnabled()) simd_ms.len + nax_ms.len else simd_ms.len;
        var mi: usize = 0;
        while (mi < total_ms) : (mi += 1) {
            const m = if (mi < simd_ms.len) simd_ms[mi] else nax_ms[mi - simd_ms.len];
            const label: []const u8 = if (mi < simd_ms.len) "split-K" else "nax m16";
            const xn: usize = @intCast(m * cs.k);
            const xbuf = try allocator.alloc(f32, xn);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ 1, m, cs.k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

            // Kernel path — must ENGAGE for every listed M, at EVERY column
            // tile the test override can force; null takes the shipped per-M default.
            for ([_]?c_int{null} ++ [_]?c_int{ 2, 4, 8 }) |bn_force| {
                if (bn_force) |bn| {
                    // The NAX tile ignores bn; only the split-K rows sweep.
                    if (mi >= simd_ms.len) continue;
                    if (@mod(cs.n, bn) != 0 or bn * m > VQMM_MAX_ACC) continue;
                }
                vqmm_bn_test_override = bn_force;
                defer vqmm_bn_test_override = null;
                const got_opt = try verifyQmm(s, x, wq, wsc, wbi, 4, cs.gs);
                try testing.expect(got_opt != null);
                const got = got_opt.?;
                defer _ = mlx.mlx_array_free(got);
                const gsh = mlx.getShape(got);
                try testing.expectEqual(@as(c_int, 1), gsh[0]);
                try testing.expectEqual(m, gsh[1]);
                try testing.expectEqual(cs.n, gsh[2]);

                try expectVerifyQmmNoWorseThanStock(s, x, wq, wsc, wbi, 4, cs.gs, wdq_t, got, label);
            }
        }

        // Ineligible widths fall through to stock (null). The NAX probe is
        // FORCED false so this pins the plain-SIMD (M1-M4) dispatch on every
        // machine, an M5 running the suite included: M=1 stays qmv, M 8..16
        // must NOT reach the split-K family (the measured spill cliff past 7
        // live row vectors), and M=17 is past every lane.
        {
            vqmm_nax_probe_override = false;
            defer vqmm_nax_probe_override = null;
            for ([_]c_int{ 1, 8, 16, 17 }) |bad_m| {
                const xn: usize = @intCast(bad_m * cs.k);
                const xbuf = try allocator.alloc(f32, xn);
                defer allocator.free(xbuf);
                for (xbuf) |*v| v.* = 0.1;
                const xshape = [_]c_int{ 1, bad_m, cs.k };
                const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
                defer _ = mlx.mlx_array_free(x32);
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
                try testing.expectEqual(@as(?mlx.mlx_array, null), try verifyQmm(s, x, wq, wsc, wbi, 4, cs.gs));
            }
        }

        // M=1 stays stock under the REAL probe too (MTPLX's tile nominally
        // covers M 1..16, but M=1 belongs to stock qmv on every machine).
        {
            const xbuf = try allocator.alloc(f32, @intCast(cs.k));
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = 0.1;
            const xshape = [_]c_int{ 1, 1, cs.k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
            try testing.expectEqual(@as(?mlx.mlx_array, null), try verifyQmm(s, x, wq, wsc, wbi, 4, cs.gs));
        }
    }

    // Mixed-precision trunk projections: the M5 NAX lane adopts affine
    // 5/6/8-bit weights on the measured wide shape class. Keep this self-gated
    // like the NAX rows above so non-G17 CI never attempts to compile an
    // unavailable MPP kernel.
    if (verifyQmmNaxEnabled()) {
        const k: c_int = 512;
        const n: c_int = 5120;
        const gs: u32 = 64;
        for ([_]u32{ 5, 6, 8 }) |bits| {
            const wb = try testRandWeightBf16(rnd, &[_]c_int{ n, k }, s);
            defer _ = mlx.mlx_array_free(wb);

            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(
                &triple,
                wb,
                mlx.mlx_optional_int.some(@intCast(gs)),
                mlx.mlx_optional_int.some(@intCast(bits)),
                "affine",
                .{},
                s,
            ));
            var wq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wq);
            var wsc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wsc);
            var wbi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wbi);
            try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
            try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
            try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

            var wdq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wdq);
            try mlx.check(mlx.mlx_dequantize(
                &wdq,
                wq,
                wsc,
                wbi,
                mlx.mlx_optional_int.some(@intCast(gs)),
                mlx.mlx_optional_int.some(@intCast(bits)),
                "affine",
                .{ .ctx = null },
                mlx.mlx_optional_dtype{ .has_value = true, .value = .float32 },
                s,
            ));
            var wdq_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wdq_t);
            const t_axes = [_]c_int{ 1, 0 };
            try mlx.check(mlx.mlx_transpose_axes(&wdq_t, wdq, &t_axes, 2, s));

            const m: c_int = 8;
            const xn: usize = @intCast(m * k);
            const xbuf = try allocator.alloc(f32, xn);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ 1, m, k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

            const got_opt = try verifyQmm(s, x, wq, wsc, wbi, bits, gs);
            try testing.expect(got_opt != null);
            const got = got_opt.?;
            defer _ = mlx.mlx_array_free(got);
            try expectVerifyQmmNoWorseThanStock(
                s,
                x,
                wq,
                wsc,
                wbi,
                bits,
                gs,
                wdq_t,
                got,
                switch (bits) {
                    5 => "nax oQe q5",
                    6 => "nax oQe q6",
                    else => "nax q8 candidate",
                },
            );

            // Below the NAX takeover row, default adoption remains the measured
            // width/shape surface: this q6/gs64/N=5120 projection uses the
            // plain-SIMD lane, while q5 still falls through to stock.
            const x6buf = try allocator.alloc(f32, 6 * @as(usize, @intCast(k)));
            defer allocator.free(x6buf);
            for (x6buf) |*v| v.* = 0.1;
            const x6shape = [_]c_int{ 1, 6, k };
            const x6f = mlx.mlx_array_new_data(x6buf.ptr, &x6shape, 3, .float32);
            defer _ = mlx.mlx_array_free(x6f);
            var x6 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x6);
            try mlx.check(mlx.mlx_astype(&x6, x6f, .bfloat16, s));
            const got6_opt = try verifyQmm(s, x6, wq, wsc, wbi, bits, gs);
            if (bits == 6) {
                try testing.expect(got6_opt != null);
                const got6 = got6_opt.?;
                defer _ = mlx.mlx_array_free(got6);
                try expectVerifyQmmNoWorseThanStock(s, x6, wq, wsc, wbi, bits, gs, wdq_t, got6, "plain-SIMD q6");
            } else {
                try testing.expectEqual(@as(?mlx.mlx_array, null), got6_opt);
            }
        }
    }

    // ── msg wide-tile parity (the huge-N/lm_head arm), called directly with
    // a RAGGED N (1000 % 32 != 0) so the in-kernel n0 guard is exercised. ──
    {
        const mk: c_int = 2048;
        const mn: c_int = 1000;
        const wb = try testRandWeightBf16(rnd, &[_]c_int{ mn, mk }, s);
        defer _ = mlx.mlx_array_free(wb);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        var wq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wq);
        var wsc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wsc);
        var wbi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wbi);
        try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

        var wdq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq);
        try mlx.check(mlx.mlx_dequantize(&wdq, wq, wsc, wbi, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .has_value = true, .value = .float32 }, s));
        var wdq_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq_t);
        const t_axes = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&wdq_t, wdq, &t_axes, 2, s));

        var m: c_int = 2;
        while (m <= 6) : (m += 1) {
            const xn: usize = @intCast(m * mk);
            const xbuf = try allocator.alloc(f32, xn);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ 1, m, mk };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

            const xsh = mlx.getShape(x);
            const got = (try runVerifyQmmMsg(s, x, wq, wsc, wbi, 4, 64, m, mk, mn, .bfloat16, xsh)).?;
            defer _ = mlx.mlx_array_free(got);

            try expectVerifyQmmNoWorseThanStock(s, x, wq, wsc, wbi, 4, 64, wdq_t, got, "msg ragged-N");
        }
    }
}

test "vqmmLaneFor: NAX dispatch table (M 8..16 route to the m16 tile only when the probe is live)" {
    // Pure lane selection — todo-m5-nax.md §7. Hermetic on every machine:
    // no kernels are built, only the routing decision is pinned.
    const nax_off = false;
    const nax_on = true;

    // Plain-SIMD machine (probe false): the table is byte-identical to the
    // pre-NAX dispatch — M 8..16 fall through to stock.
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(1, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(2, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(7, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.msg, vqmmLaneFor(4, 5120, 151936, nax_off, 8));
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(16, 5120, 17408, nax_off, 8));
    // Geometry floors shared by every lane.
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(4, 5120, 256, nax_off, 8)); // N < 512
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(4, 100, 17408, nax_off, 8)); // K % 64 != 0
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(4, 5120, 17409, nax_off, 8)); // N % 4 != 0

    // M5 (probe true), default takeover width 8: M 2..7 KEEP the plain-SIMD
    // lanes (MTPLX's own dispatcher keeps SIMD through M=6 with NAX lit —
    // 16-row padding waste makes SIMD competitive at small M).
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(7, 5120, 17408, nax_on, 8));
    try testing.expectEqual(VqmmLane.msg, vqmmLaneFor(7, 5120, 151936, nax_on, 8));
    // M 8..16 route to the NAX tile when its stricter geometry holds…
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(8, 5120, 17408, nax_on, 8));
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(16, 5120, 17408, nax_on, 8));
    // …including the lm_head class natively (N=151936 % 32 == 0 — no msg
    // variant needed past M=7).
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(9, 5120, 151936, nax_on, 8));
    // Tile-ineligible geometry at M 8..16 stays stock: K % 256, N % 32.
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5184, 17408, nax_on, 8)); // 5184 % 256 == 64
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5120, 17412, nax_on, 8)); // 17412 % 32 == 4
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(17, 5120, 17408, nax_on, 8)); // past every lane
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(1, 5120, 17408, nax_on, 8)); // M=1 stays qmv
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5120, 480, nax_on, 8)); // tiny-N floor holds for NAX too

    // A lowered takeover width (5): M 5..7 route to NAX, M 4 keeps split-K,
    // and the tile geometry is still required, else the SIMD lanes serve.
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(5, 5120, 17408, nax_on, 5));
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(7, 5120, 151936, nax_on, 5));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(4, 5120, 17408, nax_on, 5));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(5, 5184, 17408, nax_on, 5)); // K % 256 fails, % 64 holds
}

test "mixedNaxShapeEnabled keeps measured q5/q6/q8 wins and rejects narrow regressions" {
    try testing.expect(mixedNaxShapeEnabled(4, 32, 1024));
    try testing.expect(mixedNaxShapeEnabled(5, 64, 5120));
    try testing.expect(mixedNaxShapeEnabled(6, 64, 5120));
    try testing.expect(mixedNaxShapeEnabled(8, 64, 5120));
    try testing.expect(!mixedNaxShapeEnabled(5, 64, 1024));
    try testing.expect(!mixedNaxShapeEnabled(6, 64, 1024));
    try testing.expect(!mixedNaxShapeEnabled(8, 64, 1024));
    try testing.expect(!mixedNaxShapeEnabled(5, 32, 5120));
    try testing.expect(!mixedNaxShapeEnabled(6, 128, 5120));
    try testing.expect(!mixedNaxShapeEnabled(8, 32, 5120));
}

test "NAX availability rejects compiler versions and malformed GPU identifiers" {
    for ([_][]const u8{ "air64_v27", "AIR64_V27", "metal_27", "foo17", "gfoobar27", "g17dd", "applegpu_g", "applegpu_g17_extra" }) |arch| {
        try testing.expect(!naxAvailableFrom(false, arch, "26.6.2"));
    }
    try testing.expect(naxAvailableFrom(false, "AppleGPU_G17S", "26.6.2"));
    try testing.expect(naxAvailableFrom(false, "g18p", "26.6.2"));
    try testing.expect(!naxAvailableFrom(false, "g17p", "26.6.2"));
}

test "NAX availability probe: G17 prefix + macOS 26.2 floor + fallback rehearsal (pure parts)" {
    // Mirrors MTPLX's shipping nax_available() gate exactly (nax_verify.py).
    // Arch: case-insensitive prefix match on "applegpu_g17".
    try testing.expect(naxArchIsG17("applegpu_g17s"));
    try testing.expect(naxArchIsG17("AppleGPU_G17"));
    try testing.expect(naxArchIsG17("applegpu_g17"));
    try testing.expect(!naxArchIsG17("applegpu_g16")); // M4-class
    try testing.expect(!naxArchIsG17("applegpu_g13"));
    try testing.expect(!naxArchIsG17(""));
    try testing.expect(naxArchIsG17("g17"));

    // macOS floor: >= 26.2 (where MetalPerformancePrimitives ships).
    try testing.expect(macosVersionAtLeast("26.2", 26, 2));
    try testing.expect(macosVersionAtLeast("26.4.1", 26, 2));
    try testing.expect(macosVersionAtLeast("27.0", 26, 2));
    try testing.expect(macosVersionAtLeast("27", 26, 2));
    try testing.expect(!macosVersionAtLeast("26.1", 26, 2));
    try testing.expect(!macosVersionAtLeast("26", 26, 2));
    try testing.expect(!macosVersionAtLeast("25.5", 26, 2));
    // Unparseable components read as 0 and never satisfy the floor (MTPLX
    // semantics: int() failures fall back to 0).
    try testing.expect(!macosVersionAtLeast("", 26, 2));
    try testing.expect(!macosVersionAtLeast("garbage", 26, 2));
    try testing.expect(!macosVersionAtLeast("26.x", 26, 2));

    // The combined gate.
    try testing.expect(naxAvailableFrom(false, "applegpu_g17d", "26.2"));
    try testing.expect(!naxAvailableFrom(false, "applegpu_g16", "26.2"));
    try testing.expect(!naxAvailableFrom(false, "applegpu_g17d", "26.1"));
    // SUSHI_FORCE_GPU_FAMILY_FALLBACK=1 QA rehearsal: an M5 pretends the
    // units are absent so the exact M1-M4 plain-SIMD path runs there.
    try testing.expect(!naxAvailableFrom(true, "applegpu_g17d", "26.4"));
}

test "naxStatusFrom: on for G17 + 26.2, off names the missing leg (--version / Settings display)" {
    // Hardware + OS both satisfied → active. Kernels-present is a build-time
    // invariant for our binaries (build-mlx.sh + tests/test_mlx_staged_nax.sh),
    // so hardware capability == MLX's stock-op is_nax_available() gate.
    try testing.expectEqualStrings("on (M5 neural accelerators)", naxStatusFrom("applegpu_g17s", "26.2"));
    try testing.expectEqualStrings("on (M5 neural accelerators)", naxStatusFrom("applegpu_g17d", "27.0"));
    // Pre-M5 GPU: the GPU is the blocker regardless of OS.
    try testing.expectEqualStrings("off (requires M5-class GPU)", naxStatusFrom("applegpu_g16s", "26.4"));
    try testing.expectEqualStrings("off (requires M5-class GPU)", naxStatusFrom("", "26.4"));
    // M5 on an old OS: the OS is the blocker (bundle min is 26.2 anyway, but
    // a dev build can run anywhere).
    try testing.expectEqualStrings("off (requires macOS 26.2+)", naxStatusFrom("applegpu_g17s", "26.1"));
}

test "verifyQmmNaxAvailable: false on every non-G17 device (kernel is never built here)" {
    // The real-device leg of the probe: on this M4 (and every CI runner) the
    // architecture is not applegpu_g17, so the probe must read false and the
    // NAX kernel object can never be constructed. On a real M5 the arch IS
    // G17 and this test only checks the plumbing returned something.
    var buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&buf) orelse {
        // No Metal device info at all: the probe must be false.
        try testing.expect(!verifyQmmNaxAvailable());
        return;
    };
    try testing.expect(arch.len > 0);
    var vbuf: [64]u8 = undefined;
    const ver = macosProductVersion(&vbuf) orelse "<none>";
    errdefer std.debug.print("[nax-probe] arch={s} macos={s} available={}\n", .{ arch, ver, verifyQmmNaxAvailable() });
    if (!naxArchIsG17(arch)) {
        try testing.expect(!verifyQmmNaxAvailable());
    }
}

test "naxMinMForBits: q8 takes over at the measured M7, packed widths at M8" {
    try testing.expectEqual(@as(c_int, 8), naxMinMForBits(4));
    try testing.expectEqual(@as(c_int, 8), naxMinMForBits(6));
    try testing.expectEqual(@as(c_int, 7), naxMinMForBits(8));
}

test "NAX host scaffolding: zero-pad to 16 rows + slice-back are exact (runs off-M5)" {
    // The half of runVerifyQmmNax that CAN execute without G17 hardware —
    // the exact production helpers (naxPadTo16 / naxSliceRows) wrapped
    // around STOCK qmm standing in for the NAX kernel. Pins the mlx_pad
    // axis/value plumbing (pad rows are EXACT zeros in the activation
    // dtype, real rows byte-preserved), the slice bounds, and that zero
    // activation rows produce exactly-zero output rows (so on the M5 the
    // padded tile positions can never bleed into the sliced result).
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA110C);
    const rnd = prng.random();
    const K: c_int = 512;
    const N: c_int = 640; // % 32 == 0, NAX-shaped
    const gs: u32 = 64;

    const wb = try testRandWeightBf16(rnd, &[_]c_int{ N, K }, s);
    defer _ = mlx.mlx_array_free(wb);
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var wsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsc);
    var wbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

    for ([_]c_int{ 9, 16 }) |m| {
        const xn: usize = @intCast(m * K);
        const xbuf = try allocator.alloc(f32, xn);
        defer allocator.free(xbuf);
        for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
        const xshape = [_]c_int{ 1, m, K };
        const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

        // Pad half: [1, m, K] → [16, K]; real rows byte-preserved, pad rows 0.
        const x16 = try naxPadTo16(s, x, m, K, .bfloat16);
        defer _ = mlx.mlx_array_free(x16);
        const psh = mlx.getShape(x16);
        try testing.expectEqual(@as(usize, 2), psh.len);
        try testing.expectEqual(@as(c_int, 16), psh[0]);
        try testing.expectEqual(K, psh[1]);
        {
            var xf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xf);
            var pf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pf);
            try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
            try mlx.check(mlx.mlx_astype(&pf, x16, .float32, s));
            try mlx.check(mlx.mlx_array_eval(xf));
            try mlx.check(mlx.mlx_array_eval(pf));
            const xd_ = mlx.mlx_array_data_float32(xf).?;
            const pd_ = mlx.mlx_array_data_float32(pf).?;
            for (0..xn) |i| try testing.expectEqual(xd_[i], pd_[i]);
            for (xn..@intCast(16 * K)) |i| try testing.expectEqual(@as(f32, 0.0), pd_[i]);
        }

        // Slice half around stock qmm: zero rows in → exactly-zero rows out,
        // and the slice returns the first m rows byte-identically.
        var y16 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y16);
        try mlx.check(mlx.mlx_quantized_matmul(&y16, x16, wq, wsc, wbi, true, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(4), "affine", s));
        const ym = try naxSliceRows(s, y16, m, N);
        defer _ = mlx.mlx_array_free(ym);
        const ysh = mlx.getShape(ym);
        try testing.expectEqual(@as(usize, 2), ysh.len);
        try testing.expectEqual(m, ysh[0]);
        try testing.expectEqual(N, ysh[1]);
        {
            var ff = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ff);
            var mf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(mf);
            try mlx.check(mlx.mlx_astype(&ff, y16, .float32, s));
            try mlx.check(mlx.mlx_astype(&mf, ym, .float32, s));
            try mlx.check(mlx.mlx_array_eval(ff));
            try mlx.check(mlx.mlx_array_eval(mf));
            const fd = mlx.mlx_array_data_float32(ff).?;
            const md = mlx.mlx_array_data_float32(mf).?;
            const mn: usize = @intCast(m * N);
            for (0..mn) |i| try testing.expectEqual(fd[i], md[i]);
            for (mn..@intCast(16 * N)) |i| try testing.expectEqual(@as(f32, 0.0), @abs(fd[i]));
        }
    }
}

test "prefillEvalCadence: small transients keep the coarse cadence" {
    const t = std.testing;
    fused256_override = false;
    defer fused256_override = null;
    // 26B geometry, short prompt: 16 heads x 2048 chunk x 2048 kv x 2B = 128 MB
    // of unfused scores — well inside the budget, keep today's cadence.
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 16, 8, 2048, 2048, false));
    // The LIVE-measured no-regression point: a 5,140-token agent prompt
    // (845 MB scores) must keep cadence 4 — flipping it cost a measured 4.5%
    // prefill for zero memory benefit (peak 18.1 GB both ways).
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 16, 8, 5140, 5140, false));
    // Fused head dim + dense KV: no transient at all, keep even the 48 default.
    try t.expectEqual(@as(u32, 48), Transformer.prefillEvalCadence(48, 128, 8, 8, 8192, 500_000, false));
}

test "prefillEvalCadence: fused hd-256 kernel drops the score term" {
    const t = std.testing;
    // With sushi_attn_p256 active (the default) the score transient never
    // exists — hd 256 keeps the coarse cadence at any context, exactly like
    // hd 128. The dequant term must still fire under --kv-quant.
    fused256_override = true;
    defer fused256_override = null;
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 16, 8, 8192, 102_448, false));
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(4, 256, 16, 8, 8192, 400_000, true));
}

test "prefillEvalCadence: big unfused score tensor forces eval-per-layer" {
    const t = std.testing;
    fused256_override = false;
    defer fused256_override = null;
    // 16 heads x 8192 chunk x 16384 kv x 2B = 4 GiB > 2 GiB budget: bounds a
    // ~13 GB windowed transient to ~4 GB. (8192 chunk x 8192 kv sits exactly
    // ON the 2 GiB budget and deliberately keeps the coarse cadence.)
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(4, 256, 16, 8, 8192, 16384, false));
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 16, 8, 8192, 8192, false));
    // The capped 1024 chunk at 102K ctx still materializes 3.4 GB — the chunk
    // cap (generate.zig) does NOT restore the coarse cadence at long context.
    // (Live: this cadence measured peak 27.0 GB and +14% prefill vs baseline.)
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(4, 256, 16, 8, 1024, 102_448, false));
}

test "decodeAsyncLadderStride: ships OFF (measured negative); auto/<n> opt in" {
    const t = testing;
    // Unset and "auto" both take the measured default; a caller must be able
    // to turn the ladder OFF for an A/B, and a silent fallback to ON would
    // make the off-cell output-identical to the on-cell and read as a null win.
    try t.expectEqual(@as(u32, 0), Transformer.decodeAsyncLadderStride(null));
    try t.expectEqual(@as(u32, 0), Transformer.decodeAsyncLadderStride(""));
    try t.expectEqual(@as(u32, 0), Transformer.decodeAsyncLadderStride("0"));
    try t.expectEqual(@as(u32, 0), Transformer.decodeAsyncLadderStride("off"));
    try t.expectEqual(@as(u32, 8), Transformer.decodeAsyncLadderStride("auto"));
    try t.expectEqual(@as(u32, 1), Transformer.decodeAsyncLadderStride("1"));
    try t.expectEqual(@as(u32, 4), Transformer.decodeAsyncLadderStride("4"));
    // Garbage stays OFF rather than silently enabling a measured-negative lever.
    try t.expectEqual(@as(u32, 0), Transformer.decodeAsyncLadderStride("banana"));
}

test "prefillEvalCadenceApplies: spec-verify-width forwards skip the cadence entirely" {
    const t = std.testing;
    // Spec-decode verify forwards (PLD/drafter/MTP: seq 2..9) are
    // decode-shaped — KB-scale transients — but ride the prefill layer loop
    // because they're multi-token. The mid-loop eval() cadence only costs
    // them synchronous pipeline drains (measured: cadence 4 on the 64-layer
    // qwen3.6-27B = 16 drains per MTP round, ~13 ms of a 48 ms depth-1
    // round, superlinear in draft depth). seq-1 decode never ran cadence
    // evals, so exempting verify widths bounds nothing new.
    try t.expect(!Transformer.prefillEvalCadenceApplies(2));
    try t.expect(!Transformer.prefillEvalCadenceApplies(9));
    try t.expect(!Transformer.prefillEvalCadenceApplies(31));
    // Real prefill chunks keep the cadence (and the budget-driven flip).
    try t.expect(Transformer.prefillEvalCadenceApplies(32));
    try t.expect(Transformer.prefillEvalCadenceApplies(512));
    try t.expect(Transformer.prefillEvalCadenceApplies(8192));
}

test "prefillEvalCadence: quantized-KV dequant forces eval-per-layer even at fused head dims" {
    const t = std.testing;
    // hd 128 is fused (zero scores) but denseView rebuilds the FULL cache per
    // layer under --kv-quant: 2 x 600K x 8 x 128 x 2B = 2.46 GB > 2 GiB.
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(48, 128, 8, 8, 8192, 600_000, true));
    // Same shape with dense fp16 KV: nothing to dequantize, keep coarse.
    try t.expectEqual(@as(u32, 48), Transformer.prefillEvalCadence(48, 128, 8, 8, 8192, 600_000, false));
}

// ── fused hd-256 prefill attention parity ──

test "fusedSdpaPrefill: causal parity vs composed SDPA (GQA, ragged shapes, chunk offset)" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    // The causal arm yields to the stock fused kernel on NAX silicon, so on an
    // M5-class Mac this parity check would DECLINE instead of running — the
    // kernel under test would never be exercised. Pin the arm, not the host.
    nax_sdpa_override = false;
    defer nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x256256);
    const rnd = prng.random();

    // qL=70 (partial 32-tile), kL=193 (partial 16-block), offset 123 (chunked
    // prefill bottom-right alignment), Hq=6/Hk=2 (gqa 3).
    const q_shape = [_]c_int{ 1, 6, 70, 256 };
    const kv_shape = [_]c_int{ 1, 2, 193, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0; // 1/sqrt(256)

    const fused = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);

    // Not byte-parity: the composed path rounds softmax probs to bf16 before
    // the AV matmul, the fused kernel keeps float32 — bf16-rounding-scale
    // differences only.
    // Measured 0.00049 (one bf16 ULP at 0.5 magnitude) on this seed.
    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.005);
}

/// fp32 ground truth for an attention shape: the same composed chain MLX runs
/// for an unfused head dim, with every operand widened first. Both the bf16
/// composed path and the kernel are measured against THIS, never each other.
fn attnPdTruthF32(
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    var q32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q32);
    var k32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k32);
    var v32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v32);
    try mlx.check(mlx.mlx_astype(&q32, q, .float32, s));
    try mlx.check(mlx.mlx_astype(&k32, k, .float32, s));
    try mlx.check(mlx.mlx_astype(&v32, v, .float32, s));
    return attn256Reference(q32, k32, v32, scale, "causal", .{ .ctx = null }, s);
}

/// Parity seeds: a kernel bar is a sweep, never one chosen seed.
const ATTN_PD_PARITY_SEEDS: usize = 4;

/// One causal case at (qk, v) = (192, 128): the kernel must land no further
/// from fp32 truth than the composed bf16 path it replaces.
fn attnPd192CausalCase(seed: u64, hq: c_int, hkv: c_int, q_len: c_int, kv_len: c_int) !void {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const q_shape = [_]c_int{ 1, hq, q_len, 192 };
    const k_shape = [_]c_int{ 1, hkv, kv_len, 192 };
    const v_shape = [_]c_int{ 1, hkv, kv_len, 128 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &k_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &v_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / @sqrt(192.0);

    const fused = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const composed = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(composed);
    const truth = try attnPdTruthF32(q, k, v, scale, s);
    defer _ = mlx.mlx_array_free(truth);

    // Finiteness before the diff: an all-NaN pair diffs to NaN, which compares
    // false against every bar and would read as a pass under `!(d > bar)`.
    const scale_of = try attn256MaxAbs(truth, s);
    try std.testing.expect(std.math.isFinite(scale_of) and scale_of > 0);
    const fused_err = try attn256MaxDiff(fused, truth, s);
    const composed_err = try attn256MaxDiff(composed, truth, s);
    try std.testing.expect(std.math.isFinite(fused_err));
    // No worse than the path it replaces, plus one bf16 ulp of slack at the
    // reference's own scale (the two round in different places).
    try std.testing.expect(fused_err <= composed_err + scale_of * 0.00390625);
}

test "fusedSdpaPrefill: qk 192 / v 128 causal parity vs fp32 truth (MiMo global geometry)" {
    try mlx.requireGpu();
    fused256_override = true;
    defer fused256_override = null;
    for (0..ATTN_PD_PARITY_SEEDS) |i| {
        const seed: u64 = 0x192a + i;
        // Ragged 64-row q tile, ragged 32-key block, chunk offset, gqa 16 —
        // the mimo_v2 global layer's own head ratio.
        try attnPd192CausalCase(seed, 32, 2, 70, 193);
        try attnPd192CausalCase(seed + 1, 16, 1, 512, 512);
        try attnPd192CausalCase(seed + 2, 16, 1, 512, 4096);
    }
}

test "fusedSdpaPrefill: qk 192 / v 128 kv chunking carries the narrower value width" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0x19208);
    const rnd = prng.random();
    const q_shape = [_]c_int{ 1, 8, 64, 192 };
    const k_shape = [_]c_int{ 1, 2, 2048, 192 };
    const v_shape = [_]c_int{ 1, 2, 2048, 128 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &k_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &v_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / @sqrt(192.0);

    fused256_budget_override = 0;
    const single = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse {
        fused256_budget_override = null;
        return error.FusedDeclined;
    };
    defer _ = mlx.mlx_array_free(single);
    try std.testing.expectEqual(@as(u32, 1), fused256_last_dispatch_count);

    // Small budget -> several kv chunks; the fp32 m/l/O carry makes the chain
    // bit-identical to the single dispatch.
    fused256_budget_override = 8 * 64 * 256;
    defer fused256_budget_override = null;
    const chunked = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(chunked);
    try std.testing.expect(fused256_last_dispatch_count > 1);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(single, chunked, s));
}

test "fusedSdpaPrefill: only the instantiated (qk, value) width pairs are served" {
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    nax_sdpa_override = false;
    defer nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x19288);
    const rnd = prng.random();
    try std.testing.expect(fusedPdDims(256, 256));
    try std.testing.expect(fusedPdDims(192, 128));
    try std.testing.expect(!fusedPdDims(192, 192));
    try std.testing.expect(!fusedPdDims(256, 128));
    try std.testing.expect(!fusedPdDims(128, 128));

    // A 192-wide value (no instantiation) declines rather than reading past
    // the tile it staged.
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 32, 192 }, s);
    defer _ = mlx.mlx_array_free(q);
    const kv = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 32, 192 }, s);
    defer _ = mlx.mlx_array_free(kv);
    try std.testing.expect((try fusedSdpaPrefill(s, q, kv, kv, 1.0, 0)) == null);
}

/// A `DenseKVView` over a freshly quantized K/V pair, as `KVCache.denseView`
/// hands one out: lazy dense arrays plus the packed triples they stand for.
const TestQuantView = struct {
    view: DenseKVView,
    kq: kv_quant.QuantizedKV,
    vq: kv_quant.QuantizedKV,

    pub fn init(s: mlx.mlx_stream, k: mlx.mlx_array, v: mlx.mlx_array, bits: u8, group_size: u32) !TestQuantView {
        var kq = try kv_quant.quantizeAffine(s, k, group_size, bits);
        errdefer kq.deinit();
        var vq = try kv_quant.quantizeAffine(s, v, group_size, bits);
        errdefer vq.deinit();
        const dk = try kv_quant.dequantizeAffine(s, kq.q, kq.scales, kq.biases, group_size, bits);
        errdefer _ = mlx.mlx_array_free(dk);
        const dv = try kv_quant.dequantizeAffine(s, vq.q, vq.scales, vq.biases, group_size, bits);
        return .{
            .kq = kq,
            .vq = vq,
            .view = .{
                .k = dk,
                .v = dv,
                .owned = true,
                .k_triple_q = kq.q,
                .k_triple_scales = kq.scales,
                .k_triple_biases = kq.biases,
                .v_triple_q = vq.q,
                .v_triple_scales = vq.scales,
                .v_triple_biases = vq.biases,
                .has_quant_triple = true,
                .bits = bits,
                .group_size = group_size,
            },
        };
    }

    pub fn deinit(self: *TestQuantView) void {
        self.view.deinit();
        self.kq.deinit();
        self.vq.deinit();
    }
};

test "fusedSdpaPrefillKv: a quantized cache is read per dispatch, never rebuilt whole" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    nax_sdpa_override = false;
    defer nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x19282);
    const rnd = prng.random();

    // mimo_v2's global-layer geometry, ragged on both axes.
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 16, 70, 192 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 1993, 192 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 1993, 128 }, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / @sqrt(192.0);

    var qv = try TestQuantView.init(s, k, v, 8, 64);
    defer qv.deinit();

    // Force several slices so the absolute-offset arithmetic is exercised.
    fused256_budget_override = 16 * 70 * 256;
    defer fused256_budget_override = null;
    const sliced = (try fusedSdpaPrefillKv(s, q, &qv.view, scale, 0, .{ .ctx = null })) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(sliced);
    try std.testing.expect(fused256_last_dispatch_count > 1);

    // The dense arm over the SAME dequantized values, in one dispatch: the
    // fp32 carry makes the sliced chain exact, so this is byte equality.
    fused256_budget_override = 0;
    const whole = (try fusedSdpaPrefill(s, q, qv.view.k, qv.view.v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(whole);
    try std.testing.expectEqual(@as(u32, 1), fused256_last_dispatch_count);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(sliced, whole, s));

    // And no worse than the composed path against fp32 truth.
    const truth = try attnPdTruthF32(q, qv.view.k, qv.view.v, scale, s);
    defer _ = mlx.mlx_array_free(truth);
    const ref = try attn256Reference(q, qv.view.k, qv.view.v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);
    const scale_of = try attn256MaxAbs(truth, s);
    try std.testing.expect(std.math.isFinite(scale_of) and scale_of > 0);
    const sliced_err = try attn256MaxDiff(sliced, truth, s);
    try std.testing.expect(std.math.isFinite(sliced_err));
    try std.testing.expect(sliced_err <= (try attn256MaxDiff(ref, truth, s)) + scale_of * 0.00390625);
}

test "fusedSdpaPrefillKv: a kv6 cache is read per dispatch and matches the dense arm over its rebuild" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    nax_sdpa_override = false;
    defer nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x19286);
    const rnd = prng.random();
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 16, 70, 192 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 1993, 192 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 1993, 128 }, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / @sqrt(192.0);
    var qv = try TestQuantView.init(s, k, v, 6, 64);
    defer qv.deinit();

    fused256_budget_override = 16 * 70 * 256;
    defer fused256_budget_override = null;
    const sliced = (try fusedSdpaPrefillKv(s, q, &qv.view, scale, 0, .{ .ctx = null })) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(sliced);
    try std.testing.expect(fused256_last_dispatch_count > 1);
    fused256_budget_override = 0;
    const whole = (try fusedSdpaPrefill(s, q, qv.view.k, qv.view.v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(whole);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(sliced, whole, s));
}

test "fusedSdpaPrefillKv: an unquantized cache view takes the dense arm unchanged" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    nax_sdpa_override = false;
    defer nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x19283);
    const rnd = prng.random();
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 8, 64, 192 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 512, 192 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 512, 128 }, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / @sqrt(192.0);
    const off = DenseKVView{ .k = k, .v = v, .owned = false };
    const via_view = (try fusedSdpaPrefillKv(s, q, &off, scale, 0, .{ .ctx = null })) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(via_view);
    const direct = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(direct);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(via_view, direct, s));
}

const SWA_TEST_WINDOW: u32 = 128;

/// mimo_v2's sliding-layer attention geometry, enough for the ring and the arm.
fn swaSinkTestConfig() ModelConfig {
    var c = ModelConfig{};
    c.model_type = "mimo_v2";
    c.has_sliding_window = true;
    c.sliding_window = SWA_TEST_WINDOW;
    c.head_dim = 192;
    c.v_head_dim = 128;
    return c;
}

fn swaHostF32(alloc: std.mem.Allocator, a: mlx.mlx_array, s: mlx.mlx_stream) ![]f32 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, a, false, s));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, c, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const n = mlx.mlx_array_size(f);
    const src = mlx.mlx_array_data_float32(f) orelse return error.Unreadable;
    const out = try alloc.alloc(f32, n);
    @memcpy(out, src[0..n]);
    return out;
}

const SwaGeom = struct { hq: usize, hk: usize, qk: usize, vd: usize, t: usize };

/// fp64 truth for one chunk of a sliding layer with sinks, read off the FULL
/// history at absolute positions: a ring or band offset bug cannot hide in it.
/// `q` is the chunk `[hq, ql, qk]`; `k`/`v` the whole history `[hk, t, *]`.
fn swaSinkTruth(
    alloc: std.mem.Allocator,
    g: SwaGeom,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    sinks: []const f32,
    pos: usize,
    ql: usize,
    scale: f64,
) ![]f64 {
    const out = try alloc.alloc(f64, g.hq * ql * g.vd);
    @memset(out, 0);
    var logits: [SWA_TEST_WINDOW]f64 = undefined;
    for (0..g.hq) |h| {
        const kh = h / (g.hq / g.hk);
        for (0..ql) |r| {
            const p = pos + r;
            const lo = (p + 1) -| SWA_TEST_WINDOW;
            const qrow = q[(h * ql + r) * g.qk ..][0..g.qk];
            var m: f64 = sinks[h];
            for (lo..p + 1, 0..) |j, i| {
                const krow = k[(kh * g.t + j) * g.qk ..][0..g.qk];
                var dot: f64 = 0;
                for (qrow, krow) |a, b| dot += @as(f64, a) * b;
                logits[i] = dot * scale;
                m = @max(m, logits[i]);
            }
            var denom: f64 = @exp(@as(f64, sinks[h]) - m);
            for (logits[0 .. p + 1 - lo]) |l| denom += @exp(l - m);
            const orow = out[(h * ql + r) * g.vd ..][0..g.vd];
            for (lo..p + 1, 0..) |j, i| {
                const w = @exp(logits[i] - m) / denom;
                const vrow = v[(kh * g.t + j) * g.vd ..][0..g.vd];
                for (orow, vrow) |*o, x| o.* += w * x;
            }
        }
    }
    return out;
}

/// `[1, 1, ql, kl]` bool band: the mask the composed arm reads, bottom-right aligned.
fn swaBandMask(ql: c_int, kl: c_int) !mlx.mlx_array {
    const alloc = std.testing.allocator;
    const n: usize = @intCast(ql * kl);
    const buf = try alloc.alloc(bool, n);
    defer alloc.free(buf);
    for (0..@intCast(ql)) |r| {
        const row: c_int = kl - ql + @as(c_int, @intCast(r));
        for (0..@intCast(kl)) |c| {
            const col: c_int = @intCast(c);
            buf[r * @as(usize, @intCast(kl)) + c] = col <= row and row - col < @as(c_int, SWA_TEST_WINDOW);
        }
    }
    const shape = [_]c_int{ 1, 1, ql, kl };
    return mlx.mlx_array_new_data(buf.ptr, &shape, 4, .bool_);
}

fn absMax(xs: []const f32) f64 {
    var m: f64 = 0;
    for (xs) |x| m = @max(m, @abs(@as(f64, x)));
    return m;
}

fn swaMaxErr(got: []const f32, truth: []const f64) f64 {
    var e: f64 = 0;
    for (got, truth) |a, b| e = @max(e, @abs(@as(f64, a) - b));
    return e;
}

fn sliceSeq(s: mlx.mlx_stream, a: mlx.mlx_array, start: c_int, end: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(a);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, a, &.{ 0, 0, start, 0 }, 4, &.{ sh[0], sh[1], end, sh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    return out;
}

/// Sum-of-squares slack (RMS within 0.5%): bf16 store rounding flips land either way between the arms.
const ATTN_PD_NAX_RMS_SLACK: f64 = 1.01;

/// The kernel arms this machine can run, SIMD first: every parity case covers each of them.
fn attnPdTestArms() []const AttnPdArm {
    const all = [_]AttnPdArm{ .simd, .nax };
    return if (verifyQmmNaxAvailable() and qsaNaxOsOk()) &all else all[0..1];
}

/// The NAX arm against fp64 truth, element by element, may exceed the SIMD kernel's error only by
/// the bf16 store's own rounding flip plus its f16 P's rounding (2^-11 of the largest |V|, the
/// weights summing to one), and its RMS error by rounding noise.
fn expectAttnPdNoWorse(nax: []const f32, simd: []const f32, truth: []const f64, v_absmax: f64) !void {
    const p16 = v_absmax * 0.00048828125;
    var scale_of: f64 = 0;
    for (truth) |t| scale_of = @max(scale_of, @abs(t));
    var sq_nax: f64 = 0;
    var sq_simd: f64 = 0;
    for (nax, simd, truth) |a, b, t| {
        const ea = @abs(@as(f64, a) - t);
        const eb = @abs(@as(f64, b) - t);
        try std.testing.expect(std.math.isFinite(ea) and std.math.isFinite(eb));
        // One bf16 ulp of the element, floored at one ulp of a value 2^-8 below the tensor's scale
        // (near zero the fp32 accumulation, not the store, sets the error).
        try std.testing.expect(ea <= eb + (@abs(t) + scale_of * 0.00390625) * 0.00390625 + p16);
        sq_nax += ea * ea;
        sq_simd += eb * eb;
    }
    const n: f64 = @floatFromInt(truth.len);
    try std.testing.expect(sq_nax <= sq_simd * ATTN_PD_NAX_RMS_SLACK + n * (p16 * 0.5) * (p16 * 0.5));
}

/// Walk a ringed sliding cache through `schedule` and hold every chunk's fused
/// band + sink attention to fp64 truth, no worse than the composed arm it replaces.
fn swaSinkParityWalk(kv_cfg: KVQuantConfig, seed: u64, schedule: []const c_int) !void {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    const cfg = swaSinkTestConfig();
    const g = SwaGeom{ .hq = 16, .hk = 2, .qk = 192, .vd = 128, .t = blk: {
        var t: usize = 0;
        for (schedule) |w| t += @intCast(w);
        break :blk t;
    } };
    const t: c_int = @intCast(g.t);
    const scale: f32 = 1.0 / @sqrt(192.0);
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();

    // Peaked scores, so both the band edge and the sink move the output.
    const q_all = try attn256RandBf16Scaled(rnd, &[_]c_int{ 1, 16, t, 192 }, 8.0, s);
    defer _ = mlx.mlx_array_free(q_all);
    const k_all = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, t, 192 }, s);
    defer _ = mlx.mlx_array_free(k_all);
    const v_all = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, t, 128 }, s);
    defer _ = mlx.mlx_array_free(v_all);
    // bf16 like the checkpoint's `attention_sink_bias`, spread so some heads' sink dominates.
    var sink_f: [16]f32 = undefined;
    for (&sink_f) |*x| x.* = rnd.float(f32) * 6.0 - 2.0;
    const sink_shape = [_]c_int{16};
    const sinks_f32 = mlx.mlx_array_new_data(&sink_f, &sink_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(sinks_f32);
    var sinks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sinks);
    try mlx.check(mlx.mlx_astype(&sinks, sinks_f32, .bfloat16, s));

    // Truth reads the values the kernel reads: a quantized cache's are dequantized.
    var k_read = k_all;
    var v_read = v_all;
    var kq: ?kv_quant.QuantizedKV = null;
    var vq: ?kv_quant.QuantizedKV = null;
    defer if (kq) |*x| x.deinit();
    defer if (vq) |*x| x.deinit();
    defer if (kv_cfg.isQuant()) {
        _ = mlx.mlx_array_free(k_read);
        _ = mlx.mlx_array_free(v_read);
    };
    if (kv_cfg.isQuant()) {
        kq = try kv_quant.quantizeAffine(s, k_all, kv_cfg.group_size, kv_cfg.bits);
        vq = try kv_quant.quantizeAffine(s, v_all, kv_cfg.group_size, kv_cfg.bits);
        k_read = try kv_quant.dequantizeAffine(s, kq.?.q, kq.?.scales, kq.?.biases, kv_cfg.group_size, kv_cfg.bits);
        v_read = try kv_quant.dequantizeAffine(s, vq.?.q, vq.?.scales, vq.?.biases, kv_cfg.group_size, kv_cfg.bits);
    }
    const q_h = try swaHostF32(alloc, q_all, s);
    defer alloc.free(q_h);
    const k_h = try swaHostF32(alloc, k_read, s);
    defer alloc.free(k_h);
    const v_h = try swaHostF32(alloc, v_read, s);
    defer alloc.free(v_h);
    const sink_h = try swaHostF32(alloc, sinks, s);
    defer alloc.free(sink_h);

    var cache = try KVCache.initWithConfig(alloc, 1, kv_cfg);
    defer cache.deinit();
    cache.setSwaRing(SWA_TEST_WINDOW);

    var pos: c_int = 0;
    for (schedule) |ql| {
        const q_c = try sliceSeq(s, q_all, pos, pos + ql);
        defer _ = mlx.mlx_array_free(q_c);
        const k_c = try sliceSeq(s, k_all, pos, pos + ql);
        defer _ = mlx.mlx_array_free(k_c);
        const v_c = try sliceSeq(s, v_all, pos, pos + ql);
        defer _ = mlx.mlx_array_free(v_c);
        var view = try cache.update(0, k_c, v_c, s, slidingViewFor(&cfg, pos + ql, ql).span);
        defer view.deinit();

        const mask = try swaBandMask(ql, mlx.getShape(view.k)[2]);
        defer _ = mlx.mlx_array_free(mask);
        var composed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(composed);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&composed, q_c, view.k, view.v, scale, "array", mask, sinks, false, s));

        const q_ch = try swaHostF32(alloc, q_c, s);
        defer alloc.free(q_ch);
        const truth = try swaSinkTruth(alloc, g, q_ch, k_h, v_h, sink_h, @intCast(pos), @intCast(ql), scale);
        defer alloc.free(truth);
        const composed_h = try swaHostF32(alloc, composed, s);
        defer alloc.free(composed_h);

        var scale_of: f64 = 0;
        for (truth) |x| scale_of = @max(scale_of, @abs(x));
        try std.testing.expect(std.math.isFinite(scale_of) and scale_of > 0);
        const composed_err = swaMaxErr(composed_h, truth);
        try std.testing.expect(std.math.isFinite(composed_err));
        var simd_h: ?[]f32 = null;
        defer if (simd_h) |h| alloc.free(h);
        for (attnPdTestArms()) |arm| {
            attn_pd_nax_override = arm == .nax;
            defer attn_pd_nax_override = null;
            const fused = (try slidingPrefillAttn(s, &cfg, q_c, &view, scale, sinks)) orelse return error.FusedDeclined;
            defer _ = mlx.mlx_array_free(fused);
            try std.testing.expectEqual(arm, attn_pd_last_arm);
            const fused_h = try swaHostF32(alloc, fused, s);
            var kept = false;
            defer if (!kept) alloc.free(fused_h);
            const fused_err = swaMaxErr(fused_h, truth);
            try std.testing.expect(std.math.isFinite(fused_err));
            // One bf16 ulp of slack at the truth's own scale: the arms round in different places.
            try std.testing.expect(fused_err <= composed_err + scale_of * 0.00390625);
            if (simd_h) |simd| try expectAttnPdNoWorse(fused_h, simd, truth, absMax(v_h)) else {
                simd_h = fused_h;
                kept = true;
            }
        }
        pos += ql;
    }
    // The walk really wrapped: the ring holds a window, not the history.
    try std.testing.expect(cache.seqLen(0) < g.t);
}

// A first chunk inside the window, two chunks that push the ring past its cap
// (the next reads window rows an earlier chunk wrote), the arm's shortest q, a
// 2048 chunk, and a ragged tail — every key length a partial 32-key block.
const SWA_SINK_SCHEDULE = [_]c_int{ 100, 512, 512, 16, 2048, 37 };

test "sliding band + sink prefill matches fp64 truth over a ringed dense cache" {
    try mlx.requireGpu();
    fused256_override = true;
    defer fused256_override = null;
    for (0..ATTN_PD_PARITY_SEEDS) |i| try swaSinkParityWalk(KVQuantConfig.dense, 0x5a1c + i, &SWA_SINK_SCHEDULE);
}

test "sliding band + sink prefill matches fp64 truth over a ringed kv8 cache" {
    try mlx.requireGpu();
    fused256_override = true;
    defer fused256_override = null;
    for (0..ATTN_PD_PARITY_SEEDS) |i| try swaSinkParityWalk(KVQuantConfig.affine(8), 0x5a2c + i, &SWA_SINK_SCHEDULE);
}

/// fp64 truth for a global layer's causal attention over `g.t` keys, bottom-right aligned.
/// `q` is `[hq, ql, qk]`, `k`/`v` are `[hk, t, *]`.
fn attnPdCausalTruth(alloc: std.mem.Allocator, g: SwaGeom, q: []const f32, k: []const f32, v: []const f32, ql: usize, scale: f64) ![]f64 {
    const out = try alloc.alloc(f64, g.hq * ql * g.vd);
    @memset(out, 0);
    const logits = try alloc.alloc(f64, g.t);
    defer alloc.free(logits);
    for (0..g.hq) |h| {
        const kh = h / (g.hq / g.hk);
        for (0..ql) |r| {
            const p = g.t - ql + r;
            const qrow = q[(h * ql + r) * g.qk ..][0..g.qk];
            var m: f64 = -std.math.inf(f64);
            for (0..p + 1) |j| {
                const krow = k[(kh * g.t + j) * g.qk ..][0..g.qk];
                var dot: f64 = 0;
                for (qrow, krow) |a, b| dot += @as(f64, a) * b;
                logits[j] = dot * scale;
                m = @max(m, logits[j]);
            }
            var denom: f64 = 0;
            for (logits[0 .. p + 1]) |l| denom += @exp(l - m);
            const orow = out[(h * ql + r) * g.vd ..][0..g.vd];
            for (0..p + 1) |j| {
                const w = @exp(logits[j] - m) / denom;
                const vrow = v[(kh * g.t + j) * g.vd ..][0..g.vd];
                for (orow, vrow) |*o, x| o.* += w * x;
            }
        }
    }
    return out;
}

/// One global-layer case (qk 192 / v 128) on every arm: `budget` > 0 forces fp32 carries across
/// dispatches, `kv8` reads a packed cache one dequantized slice per dispatch. Each arm lands no
/// further from fp64 truth than the composed bf16 path, and NAX no worse than the SIMD kernel.
fn attnPdGlobalCase(seed: u64, hq: c_int, hkv: c_int, q_len: c_int, kv_len: c_int, q_scale: f32, budget: i64, kv8: bool) !void {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const q = try attn256RandBf16Scaled(rnd, &[_]c_int{ 1, hq, q_len, 192 }, q_scale, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kv_len, 192 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kv_len, 128 }, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / @sqrt(192.0);
    var qv: ?TestQuantView = if (kv8) try TestQuantView.init(s, k, v, 8, 64) else null;
    defer if (qv) |*x| x.deinit();
    const dense = DenseKVView{ .k = k, .v = v, .owned = false };
    const view: *const DenseKVView = if (qv) |*x| &x.view else &dense;

    const g = SwaGeom{ .hq = @intCast(hq), .hk = @intCast(hkv), .qk = 192, .vd = 128, .t = @intCast(kv_len) };
    const q_h = try swaHostF32(alloc, q, s);
    defer alloc.free(q_h);
    const k_h = try swaHostF32(alloc, view.k, s);
    defer alloc.free(k_h);
    const v_h = try swaHostF32(alloc, view.v, s);
    defer alloc.free(v_h);
    const truth = try attnPdCausalTruth(alloc, g, q_h, k_h, v_h, @intCast(q_len), scale);
    defer alloc.free(truth);
    var scale_of: f64 = 0;
    for (truth) |x| scale_of = @max(scale_of, @abs(x));
    try std.testing.expect(std.math.isFinite(scale_of) and scale_of > 0);

    const composed = try attn256Reference(q, view.k, view.v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(composed);
    const composed_h = try swaHostF32(alloc, composed, s);
    defer alloc.free(composed_h);
    const composed_err = swaMaxErr(composed_h, truth);
    try std.testing.expect(std.math.isFinite(composed_err));

    fused256_budget_override = budget;
    defer fused256_budget_override = null;
    var simd_h: ?[]f32 = null;
    defer if (simd_h) |h| alloc.free(h);
    for (attnPdTestArms()) |arm| {
        attn_pd_nax_override = arm == .nax;
        defer attn_pd_nax_override = null;
        const out = (try fusedSdpaPrefillKv(s, q, view, scale, 0, .{ .ctx = null })) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(out);
        try std.testing.expectEqual(arm, attn_pd_last_arm);
        if (budget > 0) try std.testing.expect(fused256_last_dispatch_count > 1);
        const out_h = try swaHostF32(alloc, out, s);
        var kept = false;
        defer if (!kept) alloc.free(out_h);
        const err = swaMaxErr(out_h, truth);
        try std.testing.expect(std.math.isFinite(err));
        try std.testing.expect(err <= composed_err + scale_of * 0.00390625);
        if (simd_h) |simd| try expectAttnPdNoWorse(out_h, simd, truth, absMax(v_h)) else {
            simd_h = out_h;
            kept = true;
        }
    }
}

test "sushi_attn_pd every arm: qk 192 / v 128 causal vs fp64 truth (ragged, carries, peaked, kv8)" {
    try mlx.requireGpu();
    fused256_override = true;
    defer fused256_override = null;
    for (0..ATTN_PD_PARITY_SEEDS) |i| {
        const seed: u64 = 0x192f + 8 * i;
        // Ragged 16-row simdgroup, 64-row tile and 32-key block at a chunk offset.
        try attnPdGlobalCase(seed, 32, 2, 70, 193, 1.0, 0, false);
        // The shortest q, a partial last key block.
        try attnPdGlobalCase(seed + 1, 8, 1, 16, 48, 1.0, 0, false);
        // Peaked scores across fp32 carries: 96-key dispatches.
        try attnPdGlobalCase(seed + 2, 16, 1, 130, 1111, 8.0, 16 * 130 * 96, false);
        try attnPdGlobalCase(seed + 3, 4, 1, 300, 2300, 20.0, 4 * 300 * 256, false);
        // A packed kv8 cache, dequantized one slice per dispatch.
        try attnPdGlobalCase(seed + 4, 16, 1, 70, 1993, 8.0, 16 * 70 * 256, true);
    }
}

test "sushi_attn_pd every arm: a chunked dispatch chain is bit-identical to one dispatch" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0x19299);
    const rnd = prng.random();
    const q = try attn256RandBf16Scaled(rnd, &[_]c_int{ 1, 16, 70, 192 }, 8.0, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 1993, 192 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 1993, 128 }, s);
    defer _ = mlx.mlx_array_free(v);
    var qv = try TestQuantView.init(s, k, v, 8, 64);
    defer qv.deinit();
    const scale: f32 = 1.0 / @sqrt(192.0);
    defer fused256_budget_override = null;
    for (attnPdTestArms()) |arm| {
        attn_pd_nax_override = arm == .nax;
        defer attn_pd_nax_override = null;
        fused256_budget_override = 0;
        const whole = (try fusedSdpaPrefill(s, q, qv.view.k, qv.view.v, scale, 0)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(whole);
        try std.testing.expectEqual(@as(u32, 1), fused256_last_dispatch_count);
        fused256_budget_override = 16 * 70 * 256;
        const chunked = (try fusedSdpaPrefill(s, q, qv.view.k, qv.view.v, scale, 0)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(chunked);
        try std.testing.expect(fused256_last_dispatch_count > 1);
        const sliced = (try fusedSdpaPrefillKv(s, q, &qv.view, scale, 0, .{ .ctx = null })) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(sliced);
        try std.testing.expectEqual(arm, attn_pd_last_arm);
        try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(whole, chunked, s));
        try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(whole, sliced, s));
    }
}

test "sushi_attn_pd_nax takes a 4x dispatch budget: fewer key chunks than the SIMD kernel" {
    if (mlx.noGpuBackend() or !verifyQmmNaxAvailable() or !qsaNaxOsOk()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0x1b4d);
    const rnd = prng.random();
    // 16 heads x 4096 rows: the SIMD budget covers 3808 keys per dispatch, the NAX one 15232.
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 16, 4096, 192 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, 8192, 192 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, 8192, 128 }, s);
    defer _ = mlx.mlx_array_free(v);
    for ([_]AttnPdArm{ .simd, .nax }, [_]u32{ 3, 1 }) |arm, want| {
        attn_pd_nax_override = arm == .nax;
        defer attn_pd_nax_override = null;
        const out = (try fusedSdpaPrefill(s, q, k, v, 1.0 / @sqrt(192.0), 0)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(out);
        try std.testing.expectEqual(arm, attn_pd_last_arm);
        try std.testing.expectEqual(want, fused256_last_dispatch_count);
    }
}

test "sushi_attn_pd_nax: rows that start off an 8-byte boundary give the bytes of a contiguous copy" {
    if (mlx.noGpuBackend() or !verifyQmmNaxAvailable() or !qsaNaxOsOk()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    attn_pd_nax_override = true;
    defer attn_pd_nax_override = null;
    var prng = std.Random.DefaultPrng.init(0x1b4e);
    const rnd = prng.random();
    // Row strides of 193 and 129 elements: rows start on every 2-byte offset of an 8-byte word.
    var strided: [3]mlx.mlx_array = undefined;
    var dense: [3]mlx.mlx_array = undefined;
    var built: usize = 0;
    defer for (0..built) |i| {
        _ = mlx.mlx_array_free(strided[i]);
        _ = mlx.mlx_array_free(dense[i]);
    };
    for ([_][4]c_int{ .{ 1, 4, 40, 192 }, .{ 1, 1, 96, 192 }, .{ 1, 1, 96, 128 } }) |sh| {
        const wide = try attn256RandBf16(rnd, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] + 1 }, s);
        defer _ = mlx.mlx_array_free(wide);
        strided[built] = mlx.mlx_array_new();
        dense[built] = mlx.mlx_array_new();
        built += 1;
        try mlx.check(mlx.mlx_slice(&strided[built - 1], wide, &[_]c_int{ 0, 0, 0, 0 }, 4, &sh, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        try mlx.check(mlx.mlx_contiguous(&dense[built - 1], strided[built - 1], false, s));
    }
    const scale: f32 = 1.0 / @sqrt(192.0);
    const want = (try fusedSdpaPrefill(s, dense[0], dense[1], dense[2], scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(want);
    try std.testing.expectEqual(AttnPdArm.nax, attn_pd_last_arm);
    const got = (try fusedSdpaPrefill(s, strided[0], strided[1], strided[2], scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(got);
    try std.testing.expectEqual(AttnPdArm.nax, attn_pd_last_arm);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(got, want, s));
}

test "sushi_attn_pd_nax: a failed probe declines to the SIMD kernel by name" {
    if (mlx.noGpuBackend() or !verifyQmmNaxAvailable() or !qsaNaxOsOk()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    attn_pd_nax_override = true;
    defer attn_pd_nax_override = null;
    var prng = std.Random.DefaultPrng.init(0x19277);
    const rnd = prng.random();
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, 32, 192 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, 80, 192 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, 80, 128 }, s);
    defer _ = mlx.mlx_array_free(v);
    defer attnPdNaxProbeReset();

    // The real probe passes on this machine, a corrupted NAX tile fails it.
    attnPdNaxProbeReset();
    try std.testing.expect(attnPdNaxRunProbe());
    attn_pd_nax_probe_corrupt_override = true;
    defer attn_pd_nax_probe_corrupt_override = false;
    try std.testing.expect(!attnPdNaxRunProbe());

    // A failed probe latches the SIMD arm for the process.
    attnPdNaxProbeReset();
    const out = (try fusedSdpaPrefill(s, q, k, v, 1.0 / @sqrt(192.0), 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(out);
    try std.testing.expectEqual(AttnPdArm.simd, attn_pd_last_arm);
    try std.testing.expect(!mlx.errorPending());
}

test "the sliding arm engages exactly where slidingPrefillFused says it does" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const cfg = swaSinkTestConfig();
    fused256_override = true;
    defer fused256_override = null;
    defer sliding_prefill_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x5a3c);
    const rnd = prng.random();
    const sinks = try attn256RandBf16(rnd, &[_]c_int{4}, s);
    defer _ = mlx.mlx_array_free(sinks);
    var engaged_any = false;
    for ([_]?bool{ null, false }) |ov| {
        sliding_prefill_fused_override = ov;
        for ([_]c_int{ 8, 15, 16, 64 }) |ql| {
            const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, ql, 192 }, s);
            defer _ = mlx.mlx_array_free(q);
            const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, ql + 127, 192 }, s);
            defer _ = mlx.mlx_array_free(k);
            const v = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, ql + 127, 128 }, s);
            defer _ = mlx.mlx_array_free(v);
            const view = DenseKVView{ .k = k, .v = v, .owned = false };
            const out = try slidingPrefillAttn(s, &cfg, q, &view, 1.0 / @sqrt(192.0), sinks);
            defer if (out) |o| {
                _ = mlx.mlx_array_free(o);
            };
            try std.testing.expectEqual(slidingPrefillFused(&cfg, @intCast(ql)), out != null);
            engaged_any = engaged_any or out != null;
        }
    }
    try std.testing.expect(engaged_any);
}

test "qsaNaxOsSupported: cooperative input tensors require macOS 26.3" {
    try std.testing.expect(!qsaNaxOsSupported("26.2"));
    try std.testing.expect(qsaNaxOsSupported("26.3"));
    try std.testing.expect(qsaNaxOsSupported("26.5"));
    try std.testing.expect(!qsaNaxOsSupported(""));
}

test "naxArchSupportedFrom: desktop gen>=17, phone gen>=18" {
    try std.testing.expect(naxArchSupportedFrom("applegpu_g17s"));
    try std.testing.expect(naxArchSupportedFrom("applegpu_g17"));
    try std.testing.expect(naxArchSupportedFrom("g17"));
    try std.testing.expect(naxArchSupportedFrom("applegpu_g18s"));
    try std.testing.expect(naxArchSupportedFrom("g18"));
    try std.testing.expect(!naxArchSupportedFrom("applegpu_g16s"));
    try std.testing.expect(!naxArchSupportedFrom("g16"));
    try std.testing.expect(!naxArchSupportedFrom("applegpu_g17p"));
    try std.testing.expect(!naxArchSupportedFrom("g17p"));
    try std.testing.expect(naxArchSupportedFrom("applegpu_g18p"));
    try std.testing.expect(naxArchSupportedFrom("g18p"));
    try std.testing.expect(!naxArchSupportedFrom("garbage"));
    try std.testing.expect(!naxArchSupportedFrom(""));
}

// ── Verify-width QSA gather (S = 2 .. FUSED256_MIN_Q_LEN-1) ──

// ── Fused sparse attention at verify widths (sushi_attn_qsa256_q) ──

test "fusedSdpaPrefill: sliding-band parity vs composed 'array' mask (sliding layers)" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0xBA9D);
    const rnd = prng.random();

    const qL: c_int = 70;
    const kL: c_int = 193;
    const sw: c_int = 40;
    const q_shape = [_]c_int{ 1, 4, qL, 256 };
    const kv_shape = [_]c_int{ 1, 4, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0;

    // Reference mask: exactly createSlidingWindowMask semantics — masked when
    // col > row_abs (causal) or row_abs - col >= sw, row_abs = kL - qL + r.
    const mask_n: usize = @intCast(qL * kL);
    const mask_data = try std.testing.allocator.alloc(f32, mask_n);
    defer std.testing.allocator.free(mask_data);
    const off: c_int = kL - qL;
    var r: c_int = 0;
    while (r < qL) : (r += 1) {
        var c: c_int = 0;
        while (c < kL) : (c += 1) {
            const row_abs = off + r;
            const masked = (c > row_abs) or (row_abs - c >= sw);
            mask_data[@intCast(r * kL + c)] = if (masked) -std.math.inf(f32) else 0.0;
        }
    }
    const mask_shape = [_]c_int{ 1, 1, qL, kL };
    const mask_f32 = mlx.mlx_array_new_data(mask_data.ptr, &mask_shape, 4, .float32);
    defer _ = mlx.mlx_array_free(mask_f32);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_astype(&mask, mask_f32, .bfloat16, s));

    const fused = (try fusedSdpaPrefill(s, q, k, v, scale, sw)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);

    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.02);
}

test "fusedSdpaPrefill: declines cleanly outside its envelope" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xDEC1);
    const rnd = prng.random();

    // Wrong head_dim -> null.
    const q128_shape = [_]c_int{ 1, 4, 64, 128 };
    const q128 = try attn256RandBf16(rnd, &q128_shape, s);
    defer _ = mlx.mlx_array_free(q128);
    fused256_override = true;
    defer fused256_override = null;
    try std.testing.expect((try fusedSdpaPrefill(s, q128, q128, q128, 1.0, 0)) == null);

    // Decode/verify shapes (q_len < 16) -> null (sdpa_vector owns hd 256
    // there; stealing MTP verify forwards measured decode 48 -> 18 tok/s;
    // the 16 floor matches oMLX's _MIN_ROUTE_Q_LEN — with causal fused
    // default-on, a depth-8 MTP verify is q_len 9 and must NOT route here).
    const q1_shape = [_]c_int{ 1, 4, 1, 256 };
    const q8_shape = [_]c_int{ 1, 4, 8, 256 };
    const q15_shape = [_]c_int{ 1, 4, 15, 256 };
    const kv_shape = [_]c_int{ 1, 4, 64, 256 };
    const q1 = try attn256RandBf16(rnd, &q1_shape, s);
    defer _ = mlx.mlx_array_free(q1);
    const q8 = try attn256RandBf16(rnd, &q8_shape, s);
    defer _ = mlx.mlx_array_free(q8);
    const q15 = try attn256RandBf16(rnd, &q15_shape, s);
    defer _ = mlx.mlx_array_free(q15);
    const k1 = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k1);
    try std.testing.expect((try fusedSdpaPrefill(s, q1, k1, k1, 1.0, 0)) == null);
    try std.testing.expect((try fusedSdpaPrefill(s, q8, k1, k1, 1.0, 0)) == null);
    try std.testing.expect((try fusedSdpaPrefill(s, q15, k1, k1, 1.0, 0)) == null);

    // Kill switch -> null even for a conforming call.
    fused256_override = false;
    const q_shape = [_]c_int{ 1, 4, 64, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    try std.testing.expect((try fusedSdpaPrefill(s, q, q, q, 1.0, 0)) == null);
}

test "fusedSdpaPrefill: both arms FUSED off NAX, causal yields to stock ON NAX" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x4A7E);
    const rnd = prng.random();

    // No master-switch override: the causal arm's historical net-loss (every
    // pre-budget ratio-gated variant lost same-boot on the 27B) was the IOGPU
    // preemption class — with the kv-chunk dispatch budget it wins live
    // (2026-07-22 same-session A/B: +2.9%/+2.3%/+4.6% at 8K/16K/32K on the
    // 27B), so causal is default-on now.
    std.debug.assert(fused256_override == null);
    const q_shape = [_]c_int{ 1, 6, 64, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const kv_shape = [_]c_int{ 1, 2, 64, 256 };
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);

    // Off NAX both arms engage — the budgeted-dispatch flip this test was
    // written for. The preference is PINNED rather than read off the host:
    // `naxSdpaPreferred()` defaults to whether the machine has NAX, so leaving
    // it to the silicon makes this assertion true on an M4 and false on an M5.
    nax_sdpa_override = false;
    defer nax_sdpa_override = null;
    {
        const causal = try fusedSdpaPrefill(s, q, k, k, 1.0, 0);
        try std.testing.expect(causal != null);
        if (causal) |f| _ = mlx.mlx_array_free(f);
    }

    // On NAX the causal arm yields to the stock fused sdpa (mlx's own hd-256
    // kernel) while the BAND arm stays ours — mlx's NAX kernel takes no band.
    // Nothing pinned that asymmetry before; a decline is invisible from the
    // outside, since the composed fallback is output-equivalent.
    nax_sdpa_override = true;
    {
        const causal = try fusedSdpaPrefill(s, q, k, k, 1.0, 0);
        try std.testing.expect(causal == null);
    }

    const banded = try fusedSdpaPrefill(s, q, k, k, 1.0, 40);
    try std.testing.expect(banded != null);
    if (banded) |f| _ = mlx.mlx_array_free(f);
}

test "fused256KvChunkLen: BK alignment, one-block floor, kl cap, budget-off" {
    const t = std.testing;
    // budget <= 0: single dispatch covering the full key axis.
    try t.expectEqual(@as(c_int, 193), fused256KvChunkLen(1, 6, 70, 193, 0));
    try t.expectEqual(@as(c_int, 65536), fused256KvChunkLen(1, 24, 2048, 65536, -1));
    // 27B live shape (24 heads, 2048-chunk q) at the default budget:
    // 250M / (24*2048) = 5086 keys -> floored to the BK(32) multiple 5056.
    try t.expectEqual(@as(c_int, 5056), fused256KvChunkLen(1, 24, 2048, 65536, FUSED256_DEFAULT_DISPATCH_BUDGET));
    // Never below one BK block even at absurdly small budgets…
    try t.expectEqual(@as(c_int, 32), fused256KvChunkLen(1, 24, 2048, 65536, 1));
    // …and never above the actual key length.
    try t.expectEqual(@as(c_int, 193), fused256KvChunkLen(1, 6, 70, 193, 1 << 40));
}

test "fusedSdpaPrefill: budgeted kv chunking engages and is exact vs single dispatch" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    // Causal arm: declines on NAX silicon unless the preference is pinned.
    nax_sdpa_override = false;
    defer nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0xC4A2);
    const rnd = prng.random();

    // Hq=6/Hk=2, qL=70 (ragged q tile), kL=193 (ragged kv block). Per-key
    // work = 6*70 = 420; budget 40000 -> chunk 64 -> dispatches over
    // [0,64) [64,128) [128,192) [192,193) — 4, incl. a 1-key ragged tail.
    const q_shape = [_]c_int{ 1, 6, 70, 256 };
    const kv_shape = [_]c_int{ 1, 2, 193, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0;

    fused256_budget_override = 0;
    const single = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse {
        fused256_budget_override = null;
        return error.FusedDeclined;
    };
    defer _ = mlx.mlx_array_free(single);
    try std.testing.expectEqual(@as(u32, 1), fused256_last_dispatch_count);

    fused256_budget_override = 40_000;
    defer fused256_budget_override = null;
    const chunked = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(chunked);
    // Engagement counted, not inferred: the budget must actually split.
    try std.testing.expectEqual(@as(u32, 4), fused256_last_dispatch_count);

    // The carry rides fp32 buffers — the exact register precision — so the
    // chunked result is BIT-IDENTICAL to the single dispatch, not just close.
    const max_diff = try attn256MaxDiff(single, chunked, s);
    try std.testing.expect(max_diff == 0.0);

    // And still correct in absolute terms vs the composed reference.
    const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);
    const ref_diff = try attn256MaxDiff(chunked, ref, s);
    try std.testing.expect(ref_diff < 0.005);
}

test "fusedSdpaPrefill: band arm never chunks (single dispatch under a tiny budget)" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    fused256_budget_override = 1; // would force max chunking on the causal arm
    defer fused256_budget_override = null;
    var prng = std.Random.DefaultPrng.init(0xBA9D2);
    const rnd = prng.random();

    const qL: c_int = 70;
    const kL: c_int = 193;
    const sw: c_int = 40;
    const q_shape = [_]c_int{ 1, 4, qL, 256 };
    const kv_shape = [_]c_int{ 1, 4, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);

    const fused = (try fusedSdpaPrefill(s, q, k, v, 1.0, sw)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    // The band arm's block skip already bounds per-dispatch work; chunking it
    // would only add carry traffic. It must stay a single dispatch.
    try std.testing.expectEqual(@as(u32, 1), fused256_last_dispatch_count);

    // Unchanged band semantics (same reference as the band parity test).
    const mask_n: usize = @intCast(qL * kL);
    const mask_data = try std.testing.allocator.alloc(f32, mask_n);
    defer std.testing.allocator.free(mask_data);
    const off: c_int = kL - qL;
    var r: c_int = 0;
    while (r < qL) : (r += 1) {
        var c: c_int = 0;
        while (c < kL) : (c += 1) {
            const row_abs = off + r;
            const masked = (c > row_abs) or (row_abs - c >= sw);
            mask_data[@intCast(r * kL + c)] = if (masked) -std.math.inf(f32) else 0.0;
        }
    }
    const mask_shape = [_]c_int{ 1, 1, qL, kL };
    const mask_f32 = mlx.mlx_array_new_data(mask_data.ptr, &mask_shape, 4, .float32);
    defer _ = mlx.mlx_array_free(mask_f32);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_astype(&mask, mask_f32, .bfloat16, s));
    const ref = try attn256Reference(q, k, v, 1.0, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.02);
}

test "fusedSdpaPrefill: causal parity at Qwen 24q/4kv geometry (gqa 6, ragged 64-row tile)" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    // Causal arm: declines on NAX silicon unless the preference is pinned.
    nax_sdpa_override = false;
    defer nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x27B27B);
    const rnd = prng.random();

    // The production Qwen3.6-27B full-attention geometry: 24 q heads over 4
    // kv heads (gqa 6), hd 256. qL=97 exercises a partial 64-row q tile,
    // kL=250 a partial kv block; offset 153 = chunked-prefill alignment.
    const q_shape = [_]c_int{ 1, 24, 97, 256 };
    const kv_shape = [_]c_int{ 1, 4, 250, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0;

    const fused = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);

    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.005);
}

// ── KV cache growth policy (issue #110) ──────────────────────────────────────

// ── Constant-table dtype class guard ─────────────────────────────────────────
// A constant lookup table built at load in f32 and multiplied into a bf16
// activation PROMOTES the product to f32 (mlx's promotion rule). Once that
// product was post-RoPE q/k, so the f32 travelled through SDPA, o_proj and the
// residual — every projection and expert weight for the REST of the forward was
// upcast on read, at 3x the token cost, with nothing in the op count, the
// kernel choice or a pre-loop dtype log to show for it.

test "constTableAs hands a load-time f32 table back in the activation dtype" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const vals = [_]f32{ 1.3465735, 1.3465735, 1.0, 1.0 };
    const shape = [_]c_int{4};
    const table = mlx.mlx_array_new_data(&vals, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(table);

    var cache: DtypeCastCache = .{};
    defer cache.deinit();

    // Same dtype in, same ARRAY out — no copy, no cache entry.
    const same = try constTableAs(table, .float32, &cache, s);
    try std.testing.expect(same.ctx == table.ctx);
    try std.testing.expect(cache.arr.ctx == null);

    // Narrower activation: the table comes back narrowed, and is cached.
    const bf = try constTableAs(table, .bfloat16, &cache, s);
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(bf));
    const again = try constTableAs(table, .bfloat16, &cache, s);
    try std.testing.expect(again.ctx == bf.ctx); // cached, not rebuilt
}

test "a constant table must not widen the activation it scales" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const mvals = [_]f32{ 1.3465735, 1.3465735, 1.0, 1.0 };
    const mshape = [_]c_int{4};
    const table = mlx.mlx_array_new_data(&mvals, &mshape, 1, .float32);
    defer _ = mlx.mlx_array_free(table);

    // A bf16 activation, exactly as post-RoPE q/k arrives.
    const avals = [_]f32{ 0.5, 0.25, 0.125, 0.0625 };
    const ashape = [_]c_int{4};
    const a32 = mlx.mlx_array_new_data(&avals, &ashape, 1, .float32);
    defer _ = mlx.mlx_array_free(a32);
    var act = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_astype(&act, a32, .bfloat16, s));

    // The raw f32 table is what the bug shipped: the product widens.
    var widened = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(widened);
    try mlx.check(mlx.mlx_multiply(&widened, act, table, s));
    try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(widened));

    // Routed through constTableAs it stays in the activation's dtype.
    var cache: DtypeCastCache = .{};
    defer cache.deinit();
    const matched = try constTableAs(table, mlx.mlx_array_dtype(act), &cache, s);
    var kept = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kept);
    try mlx.check(mlx.mlx_multiply(&kept, act, matched, s));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(kept));
}

test "dtypeStep reports a residual change once, in the direction it moved" {
    // Pure decision layer of [dtype-trace]. The mscale bug presented as a
    // UNIFORM 3x with an identical op count, so the only cheap way to find the
    // next one is to watch the residual itself — and a watcher that reports
    // every layer is unreadable while one that reports only the endpoints
    // can't say WHERE it happened.
    try std.testing.expectEqual(mlx.DtypeStep.quiet, mlx.dtypeStep(.bfloat16, .bfloat16));
    try std.testing.expectEqual(mlx.DtypeStep.widened, mlx.dtypeStep(.bfloat16, .float32));
    try std.testing.expectEqual(mlx.DtypeStep.narrowed, mlx.dtypeStep(.float32, .bfloat16));
    // Equal width but a different dtype still changes kernel selection.
    try std.testing.expectEqual(mlx.DtypeStep.changed, mlx.dtypeStep(.bfloat16, .float16));

    // Walk a stack that widens at layer 7 and stays wide: exactly one report,
    // naming layer 7. A per-layer dump would emit 40 lines; endpoints alone
    // would say "it widened" without saying where.
    const stack = [_]mlx.mlx_dtype{
        .bfloat16, .bfloat16, .bfloat16, .bfloat16, .bfloat16, .bfloat16, .bfloat16,
        .float32,  .float32,  .float32,  .float32,
    };
    var prev = stack[0];
    var reports: usize = 0;
    var first_at: usize = 0;
    for (stack[1..], 1..) |cur, idx| {
        if (mlx.dtypeStep(prev, cur) != .quiet) {
            if (reports == 0) first_at = idx;
            reports += 1;
        }
        prev = cur;
    }
    try std.testing.expectEqual(@as(usize, 1), reports);
    try std.testing.expectEqual(@as(usize, 7), first_at);
}

test "dtypeTraceArm latches one trace per forward path" {
    // The trace must fire on the FIRST forward of each path and then cost
    // nothing — a per-layer FFI dtype read on every token would be a real
    // (small) tax, and repeated identical lines would bury the one that
    // matters. Distinct paths latch independently so a vision tower and its
    // text trunk both get traced in the same boot.
    mlx.resetDtypeTraceForTest();
    try std.testing.expect(mlx.dtypeTraceArm("moe"));
    try std.testing.expect(!mlx.dtypeTraceArm("moe"));
    try std.testing.expect(mlx.dtypeTraceArm("vision"));
    try std.testing.expect(!mlx.dtypeTraceArm("vision"));
    try std.testing.expect(!mlx.dtypeTraceArm("moe"));
}

test "moe down+reduce fused: no worse than gatherQmv + multiply + sum vs fp32 truth" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xD09E);
    const rnd = prng.random();
    const t = std.testing;
    downred_override = true;
    defer downred_override = null;

    // Two geometries: a small one, and a real decode shape. The fused
    // kernel reduces each row over 8 lanes, the composed chain over 32, so the
    // bar is error against the f32 truth, never bit identity.
    const geoms = [_][4]c_int{
        .{ 16, 8, 64, 32 }, // E, TOPK, KIN, N
        .{ 64, 8, 512, 2048 },
    };
    for (geoms) |geom| {
        const E: usize = @intCast(geom[0]);
        const TOPK: c_int = geom[1];
        const KIN: c_int = geom[2];
        const N: c_int = geom[3];
        try moeDownReduceParityCase(s, rnd, t, E, TOPK, KIN, N);
    }
}

fn moeDownReduceParityCase(s: mlx.mlx_stream, rnd: std.Random, t: anytype, E: usize, TOPK: c_int, KIN: c_int, N: c_int) !void {
    for ([_]struct { bits: u32, gs: u32 }{
        .{ .bits = 4, .gs = 32 },
        .{ .bits = 3, .gs = 64 },
    }) |cfg| {
        const bits = cfg.bits;
        const gs = cfg.gs;
        const words_per_row: usize = @intCast(@divExact(@as(u32, @intCast(KIN)) * bits, 32));
        const groups_per_row: usize = @intCast(@divExact(@as(u32, @intCast(KIN)), gs));

        // Packed weight bank: any bit pattern is a valid quantized row.
        const wn = E * @as(usize, @intCast(N)) * words_per_row;
        const wdata = try t.allocator.alloc(u32, wn);
        defer t.allocator.free(wdata);
        for (wdata) |*v| v.* = rnd.int(u32);
        const w_shape = [_]c_int{ @intCast(E), N, @intCast(words_per_row) };
        const w = mlx.mlx_array_new_data(wdata.ptr, &w_shape, 3, .uint32);
        defer _ = mlx.mlx_array_free(w);

        const sn = E * @as(usize, @intCast(N)) * groups_per_row;
        const s_shape = [_]c_int{ @intCast(E), N, @intCast(groups_per_row) };
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        const sf = try t.allocator.alloc(f32, sn);
        defer t.allocator.free(sf);
        for (sf) |*v| v.* = 0.01 + 0.05 * rnd.float(f32);
        const bf = try t.allocator.alloc(f32, sn);
        defer t.allocator.free(bf);
        for (bf) |*v| v.* = rnd.float(f32) - 0.5;
        const sc_src = mlx.mlx_array_new_data(sf.ptr, &s_shape, 3, .float32);
        defer _ = mlx.mlx_array_free(sc_src);
        const bi_src = mlx.mlx_array_new_data(bf.ptr, &s_shape, 3, .float32);
        defer _ = mlx.mlx_array_free(bi_src);
        try mlx.check(mlx.mlx_astype(&sc, sc_src, .bfloat16, s));
        try mlx.check(mlx.mlx_astype(&bi, bi_src, .bfloat16, s));

        const x_shape = [_]c_int{ TOPK, KIN };
        const x = try attn256RandBf16(rnd, &x_shape, s);
        defer _ = mlx.mlx_array_free(x);
        const sc_shape = [_]c_int{TOPK};
        const scores = try attn256RandBf16(rnd, &sc_shape, s);
        defer _ = mlx.mlx_array_free(scores);
        var idata: [8]u32 = undefined;
        for (&idata) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
        const inds = mlx.mlx_array_new_data(&idata, &sc_shape, 1, .uint32);
        defer _ = mlx.mlx_array_free(inds);

        const mode: QuantMode = .affine;

        // Composed chain: gatherQmv down → [TOPK,N] → ·scores → sum over K.
        const down_2d = (try gatherQmv(s, x, w, sc, bi, inds, bits, gs, mode, true)) orelse return error.GatherDeclined;
        defer _ = mlx.mlx_array_free(down_2d);
        var down_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_4d);
        const d4 = [_]c_int{ 1, 1, TOPK, N };
        try mlx.check(mlx.mlx_reshape(&down_4d, down_2d, &d4, 4, s));
        var scores_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scores_exp);
        const se = [_]c_int{ 1, 1, TOPK, 1 };
        try mlx.check(mlx.mlx_reshape(&scores_exp, scores, &se, 4, s));
        var weighted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(weighted);
        try mlx.check(mlx.mlx_multiply(&weighted, down_4d, scores_exp, s));
        var composed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(composed);
        try mlx.check(mlx.mlx_sum_axis(&composed, weighted, -2, false, s));

        // Fused: one dispatch.
        const fused_flat = (try gatherQmvDownReduce(s, x, w, sc, bi, inds, scores, bits, gs, mode)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(fused_flat);
        var fused = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(fused);
        const f3 = [_]c_int{ 1, 1, N };
        try mlx.check(mlx.mlx_reshape(&fused, fused_flat, &f3, 3, s));

        // f32 truth: dequantized bank, the TOPK experts, f32 matmul, f32 weighted sum.
        var deq32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(deq32);
        try mlx.check(mlx.mlx_dequantize(&deq32, w, sc, bi, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
        var sel = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sel);
        try mlx.check(mlx.mlx_take_axis(&sel, deq32, inds, 0, s));
        var x32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x32);
        try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        const x3s = [_]c_int{ TOPK, KIN, 1 };
        try mlx.check(mlx.mlx_reshape(&x3, x32, &x3s, 3, s));
        var prod = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(prod);
        try mlx.check(mlx.mlx_matmul(&prod, sel, x3, s));
        var sc32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc32);
        try mlx.check(mlx.mlx_astype(&sc32, scores, .float32, s));
        var sc3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc3);
        const sc3s = [_]c_int{ TOPK, 1, 1 };
        try mlx.check(mlx.mlx_reshape(&sc3, sc32, &sc3s, 3, s));
        var wprod = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wprod);
        try mlx.check(mlx.mlx_multiply(&wprod, prod, sc3, s));
        var truth_n = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(truth_n);
        try mlx.check(mlx.mlx_sum_axis(&truth_n, wprod, 0, false, s));
        var truth = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(truth);
        try mlx.check(mlx.mlx_reshape(&truth, truth_n, &f3, 3, s));

        const cnt: usize = @intCast(N);
        const hf = try t.allocator.alloc(f32, cnt);
        defer t.allocator.free(hf);
        const hc = try t.allocator.alloc(f32, cnt);
        defer t.allocator.free(hc);
        const ht = try t.allocator.alloc(f32, cnt);
        defer t.allocator.free(ht);
        try testReadF32(fused, hf, s);
        try testReadF32(composed, hc, s);
        try testReadF32(truth, ht, s);
        var se_f: f64 = 0;
        var se_c: f64 = 0;
        for (hf, hc, ht) |a, b, c| {
            try t.expect(std.math.isFinite(a));
            se_f += (a - c) * (a - c);
            se_c += (b - c) * (b - c);
        }
        const rms_f = @sqrt(se_f / @as(f64, @floatFromInt(cnt)));
        const rms_c = @sqrt(se_c / @as(f64, @floatFromInt(cnt)));
        t.expect(rms_f <= rms_c * 1.15 + 1e-6) catch |e| {
            std.debug.print("\n[down-reduce] bits={d} gs={d} K={d} N={d} fused_rms={e:.4} composed_rms={e:.4}\n", .{ bits, gs, KIN, N, rms_f, rms_c });
            return e;
        };
    }
}

test "moe down+reduce fused: declines outside its envelope" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xD09F);
    const rnd = prng.random();
    downred_override = true;
    defer downred_override = null;

    const TOPK: c_int = 8;
    const KIN: c_int = 64;
    const x_shape = [_]c_int{ TOPK, KIN };
    const x = try attn256RandBf16(rnd, &x_shape, s);
    defer _ = mlx.mlx_array_free(x);
    const sc_shape = [_]c_int{TOPK};
    const scores = try attn256RandBf16(rnd, &sc_shape, s);
    defer _ = mlx.mlx_array_free(scores);
    var idata: [8]u32 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const inds = mlx.mlx_array_new_data(&idata, &sc_shape, 1, .uint32);
    defer _ = mlx.mlx_array_free(inds);
    const no_arr = mlx.mlx_array{ .ctx = null };

    // mxfp8 mode: wrong dequant arithmetic — must decline, not compute.
    try std.testing.expect((try gatherQmvDownReduce(s, x, no_arr, no_arr, no_arr, inds, scores, 8, 32, .mxfp8)) == null);
    // f32 scores against bf16 x: the composed multiply would promote — decline.
    var scores_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_f32);
    try mlx.check(mlx.mlx_astype(&scores_f32, scores, .float32, s));
    const wn: usize = 8 * 32 * 8;
    const wdata = try std.testing.allocator.alloc(u32, wn);
    defer std.testing.allocator.free(wdata);
    for (wdata) |*v| v.* = rnd.int(u32);
    const w_shape = [_]c_int{ 8, 32, 8 };
    const w = mlx.mlx_array_new_data(wdata.ptr, &w_shape, 3, .uint32);
    defer _ = mlx.mlx_array_free(w);
    const s_shape = [_]c_int{ 8, 32, 2 };
    const scq = try attn256RandBf16(rnd, &s_shape, s);
    defer _ = mlx.mlx_array_free(scq);
    try std.testing.expect((try gatherQmvDownReduce(s, x, w, scq, scq, inds, scores_f32, 4, 32, .affine)) == null);
}

// The YaRN `mscale != 1.0` arm registered only an `errdefer` for its `scaled` intermediate, so
// every successful call leaked one array per decode tick, each pinning its `MTLSharedEvent`,
// until the process wedged at the 65,535 cap. Settled residency is the observable.
test "ropeAngleRows hands back every array it builds, success path included" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    var freq_buf: [32]f32 = undefined;
    for (&freq_buf, 0..) |*f, i| f.* = std.math.pow(f32, 10000000.0, @as(f32, @floatFromInt(i)) / 32.0);
    const f_shape = [_]c_int{32};
    const freqs = mlx.mlx_array_new_data(&freq_buf, &f_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(freqs);
    try mlx.check(mlx.mlx_array_eval(freqs));
    const no_base = mlx.mlx_optional_float{ .value = 0, .has_value = false };
    // qwen4_exp geometry: head_dim 256 x partial_rotary_factor 0.25.
    const rd: c_int = 64;

    // 1.0 is the unscaled arm; the second value is the YaRN attention factor at `factor: 4.0`.
    for ([_]f32{ 1.0, 1.1386294364929199 }) |mscale| {
        const c0 = mlx.op_count.load(.monotonic);
        const warm = try ropeAngleRows(s, rd, no_base, freqs, 0, 1, allocator, mscale);
        const n_ops = mlx.op_count.load(.monotonic) - c0;
        try mlx.check(mlx.mlx_array_eval(warm));
        _ = mlx.mlx_array_free(warm);
        try testing.expect(n_ops >= 2);

        var k: u64 = 0;
        while (k <= n_ops) : (k += 1) {
            const base = mlxSettledActiveBytes(s);
            mlx.fault.arm(k); // arm(0) is a no-op arm: the SUCCESS path
            const r = ropeAngleRows(s, rd, no_base, freqs, @intCast(k + 1), 1, allocator, mscale);
            mlx.fault.disarm();
            if (r) |a| {
                try mlx.check(mlx.mlx_array_eval(a));
                _ = mlx.mlx_array_free(a);
            } else |err| try testing.expectEqual(error.MlxError, err);
            const after = mlxSettledActiveBytes(s);
            if (after != base) std.debug.print(
                "[angle-rows] mscale={d} k={d}: base {d} after {d}\n",
                .{ mscale, k, base, after },
            );
            try testing.expectEqual(base, after);
        }
    }
}

// ── Chunked vision splice (issue #197) ───────────────────────────────────────
// A vision prompt prefills in chunks; the splice's row lookup is a cumsum over
// the CURRENT forward's placeholder mask, so each chunk passes the rows earlier
// chunks consumed as `row_offset` — a restarted scatter silently reuses the
// first image rows in every chunk (coherent output, wrong image content).

test "spliceVisionRows: two chunks with row offsets equal the whole-prompt splice" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    const hidden: c_int = 4;
    const seq: c_int = 6;
    const img_id: u32 = 99;
    // token ids: [7, 99, 99, 8, 99, 9] — placeholders at 1, 2 (chunk A) and 4
    // (chunk B), so the boundary lands INSIDE the image row sequence.
    const ids = [_]i32{ 7, 99, 99, 8, 99, 9 };
    // h rows: row i filled with i+1 (bf16-exact values).
    var h_data: [6 * 4]f32 = undefined;
    for (0..6) |r| for (0..4) |c| {
        h_data[r * 4 + c] = @floatFromInt(r + 1);
    };
    // vision rows: row k filled with 100+k — distinct per row, so a restarted
    // scatter (row 0 reused in chunk B) produces different bytes.
    var ve_data: [3 * 4]f32 = undefined;
    for (0..3) |r| for (0..4) |c| {
        ve_data[r * 4 + c] = @floatFromInt(100 + r);
    };
    const ve_shape = [_]c_int{ 1, 3, hidden };
    const ve = mlx.mlx_array_new_data(&ve_data, &ve_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(ve);

    const readRow = struct {
        fn f(arr: mlx.mlx_array, row: usize, out: *[4]f32, st: mlx.mlx_stream) !void {
            var f32arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32arr);
            try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, st));
            try mlx.check(mlx.mlx_array_eval(f32arr));
            const ptr = mlx.mlx_array_data_float32(f32arr).?;
            for (0..4) |c| out[c] = ptr[row * 4 + c];
        }
    }.f;

    // Whole-prompt splice (the reference: what the unchunked path computes).
    var full_rows: [6][4]f32 = undefined;
    {
        const h_shape = [_]c_int{ 1, seq, hidden };
        const h = mlx.mlx_array_new_data(&h_data, &h_shape, 3, .float32);
        const ids_shape = [_]c_int{ 1, seq };
        const tid = mlx.mlx_array_new_data(&ids, &ids_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tid);
        const spliced = try Transformer.spliceVisionRows(h, tid, ve, img_id, 0, 0, s);
        defer _ = mlx.mlx_array_free(spliced);
        for (0..6) |r| try readRow(spliced, r, &full_rows[r], s);
    }

    // Chunk A = tokens [0..3), offset 0; chunk B = tokens [3..6), offset 2
    // (chunk A consumed 2 placeholder rows).
    const chunks = [_]struct { start: usize, len: c_int, off: usize }{
        .{ .start = 0, .len = 3, .off = 0 },
        .{ .start = 3, .len = 3, .off = 2 },
    };
    for (chunks) |ck| {
        const h_shape = [_]c_int{ 1, ck.len, hidden };
        const h = mlx.mlx_array_new_data(h_data[ck.start * 4 ..].ptr, &h_shape, 3, .float32);
        const ids_shape = [_]c_int{ 1, ck.len };
        const tid = mlx.mlx_array_new_data(ids[ck.start..].ptr, &ids_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tid);
        const spliced = try Transformer.spliceVisionRows(h, tid, ve, img_id, 0, ck.off, s);
        defer _ = mlx.mlx_array_free(spliced);
        for (0..@intCast(ck.len)) |r| {
            var got: [4]f32 = undefined;
            try readRow(spliced, r, &got, s);
            for (0..4) |c| try std.testing.expectEqual(full_rows[ck.start + r][c], got[c]);
        }
    }
}

test "spliceVisionRows: the rows of a skipped leading block are not needed once the offset drops by their count" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const img_id: u32 = 99;
    // [7, 99, 99, 8, 99, 9]: the first block holds one row; a prefix of three tokens restores it
    // and the second block's first row, so the tail splices from row 2 of the whole array.
    const ids = [_]i32{ 8, 99, 9 };
    var h_data: [3 * 4]f32 = undefined;
    for (0..3) |r| for (0..4) |c| {
        h_data[r * 4 + c] = @floatFromInt(r + 1);
    };
    var ve_data: [3 * 4]f32 = undefined;
    for (0..3) |r| for (0..4) |c| {
        ve_data[r * 4 + c] = @floatFromInt(100 + r);
    };
    const h_shape = [_]c_int{ 1, 3, 4 };
    const ids_shape = [_]c_int{ 1, 3 };
    const tid = mlx.mlx_array_new_data(&ids, &ids_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(tid);

    const whole = mlx.mlx_array_new_data(&ve_data, &[_]c_int{ 1, 3, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(whole);
    const tail_only = mlx.mlx_array_new_data(ve_data[4..].ptr, &[_]c_int{ 1, 2, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(tail_only);

    const h_a = mlx.mlx_array_new_data(&h_data, &h_shape, 3, .float32);
    const via_whole = try Transformer.spliceVisionRows(h_a, tid, whole, img_id, 0, 2, s);
    defer _ = mlx.mlx_array_free(via_whole);
    const h_b = mlx.mlx_array_new_data(&h_data, &h_shape, 3, .float32);
    const via_tail = try Transformer.spliceVisionRows(h_b, tid, tail_only, img_id, 0, 1, s);
    defer _ = mlx.mlx_array_free(via_tail);

    for ([_]mlx.mlx_array{ via_whole, via_tail }) |arr| try mlx.check(mlx.mlx_array_eval(arr));
    var f_whole = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f_whole);
    var f_tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f_tail);
    try mlx.check(mlx.mlx_astype(&f_whole, via_whole, .float32, s));
    try mlx.check(mlx.mlx_astype(&f_tail, via_tail, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f_whole));
    try mlx.check(mlx.mlx_array_eval(f_tail));
    const a = mlx.mlx_array_data_float32(f_whole).?[0..12];
    const b = mlx.mlx_array_data_float32(f_tail).?[0..12];
    try std.testing.expectEqualSlices(f32, a, b);
    try std.testing.expectEqual(@as(f32, 102), b[4]);
}

test "spliceVisionRows: video placeholders are scattered like image rows (Qwen video_token_id)" {
    // Live 2026-08-26 (qwen4_exp, 4-frame clip): 690 <|video_pad|> tokens were
    // inserted and the tower encoded 690 rows, but the mask keyed on image |
    // audio ids only, so no video row ever reached the prompt and the model
    // described a clip it never saw.
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const hidden: c_int = 4;
    const img_id: u32 = 99;
    const vid_id: u32 = 77;
    // [99, 7, 77, 77] — one image row, text, two video rows from ONE source.
    const ids = [_]i32{ 99, 7, 77, 77 };
    var h_data: [16]f32 = @splat(0);
    var ve_data: [3 * 4]f32 = undefined;
    for (0..3) |r| for (0..4) |c| {
        ve_data[r * 4 + c] = @floatFromInt(50 + r);
    };
    const ve_shape = [_]c_int{ 1, 3, hidden };
    const ve = mlx.mlx_array_new_data(&ve_data, &ve_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(ve);
    const h_shape = [_]c_int{ 1, 4, hidden };
    const h = mlx.mlx_array_new_data(&h_data, &h_shape, 3, .float32);
    const ids_shape = [_]c_int{ 1, 4 };
    const tid = mlx.mlx_array_new_data(&ids, &ids_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(tid);
    const spliced = try Transformer.spliceVisionRows(h, tid, ve, img_id, vid_id, 0, s);
    defer _ = mlx.mlx_array_free(spliced);
    var f32arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32arr);
    try mlx.check(mlx.mlx_astype(&f32arr, spliced, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32arr));
    const ptr = mlx.mlx_array_data_float32(f32arr).?;
    try std.testing.expectEqual(@as(f32, 50), ptr[0]); // image row 0
    try std.testing.expectEqual(@as(f32, 0), ptr[4]); // text untouched
    try std.testing.expectEqual(@as(f32, 51), ptr[8]); // video row 1
    try std.testing.expectEqual(@as(f32, 52), ptr[12]); // video row 2
}

// ── qwen4_exp fixture parity (env-gated: QWEN4_TEST_MODEL + QWEN4_FIXTURE) ──

// Row i of a batched decode is slot i's solo tick, byte for byte: ragged KV past the QSA budget, slots joining
// and leaving, one M-RoPE slot. `QWEN4_ROWS_OLD=1` drives the padded batch instead (the red probe).

test "mlx check fault injection: the ownership shape releases its array on every faulted op" {
    // The machinery behind the fixture-gated sweep below, exercised on a shape
    // that holds nothing but the ownership pattern of `qwen4MtpForward`'s `h`:
    // an array born from an op, guarded by ONE function-scope errdefer,
    // rebound through a BLOCK-scoped slice that frees its predecessor itself,
    // then handed to a further fallible op. Every array here is eval'd, so a
    // leaked handle holds real Metal bytes and the active-memory oracle can
    // see it. The `leaky` arm is the NEGATIVE CONTROL: the same shape minus
    // the one errdefer must trip the oracle, or a green sweep means nothing.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    const H = struct {
        fn guarded(st: mlx.mlx_stream, seed: mlx.mlx_array) !mlx.mlx_array {
            var h = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&h, seed, seed, st));
            errdefer _ = mlx.mlx_array_free(h);
            try mlx.check(mlx.mlx_array_eval(h));
            {
                var h_last = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(h_last);
                try mlx.check(mlx.mlx_add(&h_last, h, seed, st));
                try mlx.check(mlx.mlx_array_eval(h_last));
                _ = mlx.mlx_array_free(h);
                h = h_last;
            }
            var out = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_multiply(&out, h, seed, st));
            try mlx.check(mlx.mlx_array_eval(out));
            _ = mlx.mlx_array_free(h);
            return out;
        }
        /// Same shape, ONE errdefer short. `h` lives in the CALLER's frame, so
        /// the test can free whatever this function drops after measuring it —
        /// an mlx op OVERWRITES its destination handle, so a copy taken before
        /// the op is a stale pointer, not a rescue line.
        fn leaky(st: mlx.mlx_stream, seed: mlx.mlx_array, h: *mlx.mlx_array) !mlx.mlx_array {
            try mlx.check(mlx.mlx_multiply(h, seed, seed, st));
            try mlx.check(mlx.mlx_array_eval(h.*));
            {
                var h_last = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(h_last);
                try mlx.check(mlx.mlx_add(&h_last, h.*, seed, st));
                try mlx.check(mlx.mlx_array_eval(h_last));
                _ = mlx.mlx_array_free(h.*);
                h.* = h_last;
            }
            var out = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_multiply(&out, h.*, seed, st));
            try mlx.check(mlx.mlx_array_eval(out));
            _ = mlx.mlx_array_free(h.*);
            h.* = .{ .ctx = null };
            return out;
        }
    };

    const n_elem: usize = 1 << 20; // 4 MB per array: far above measurement noise
    const host = try allocator.alloc(f32, n_elem);
    defer allocator.free(host);
    @memset(host, 1.5);
    const shape = [_]c_int{@intCast(n_elem)};
    const seed = mlx.mlx_array_new_data(host.ptr, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(seed);
    try mlx.check(mlx.mlx_array_eval(seed));

    // One un-faulted run: warms the allocator and counts the checked ops the
    // shape issues, which is the sweep's upper bound. Its output is freed here.
    const c0 = mlx.op_count.load(.monotonic);
    const warm = try H.guarded(s, seed);
    const n_ops = mlx.op_count.load(.monotonic) - c0;
    _ = mlx.mlx_array_free(warm);
    try testing.expect(n_ops >= 6);

    var fired: usize = 0;
    var k: u64 = 1;
    while (k <= n_ops) : (k += 1) {
        const base = mlxSettledActiveBytes(s);
        mlx.fault.arm(k);
        const r = H.guarded(s, seed);
        const did = mlx.fault.didFire();
        mlx.fault.disarm();
        if (r) |ok| {
            _ = mlx.mlx_array_free(ok);
        } else |err| {
            try testing.expectEqual(error.MlxError, err);
            try testing.expect(did);
            fired += 1;
        }
        // A DOUBLE free would abort this binary here rather than fail an
        // assertion — that abort is the other half of the proof.
        const after = mlxSettledActiveBytes(s);
        if (after != base) std.debug.print("[fault] guarded k={d}: base {d} after {d}\n", .{ k, base, after });
        try testing.expectEqual(base, after);
    }
    try testing.expectEqual(n_ops, fired);

    // Negative control: the same sweep without the errdefer must be CAUGHT.
    var caught: usize = 0;
    k = 1;
    while (k <= n_ops) : (k += 1) {
        var probe = mlx.mlx_array_new();
        const base = mlxSettledActiveBytes(s);
        mlx.fault.arm(k);
        const r = H.leaky(s, seed, &probe);
        mlx.fault.disarm();
        if (r) |ok| {
            _ = mlx.mlx_array_free(ok);
        } else |_| {}
        if (mlxSettledActiveBytes(s) > base + (n_elem * @sizeOf(f32)) / 2) caught += 1;
        if (probe.ctx != null) _ = mlx.mlx_array_free(probe);
        const after_ctl = mlxSettledActiveBytes(s);
        if (after_ctl != base) std.debug.print("[fault] leaky k={d}: base {d} after {d}\n", .{ k, base, after_ctl });
        try testing.expectEqual(base, after_ctl);
    }
    errdefer std.debug.print("[fault] ownership shape: {d} checked ops, {d} faulted cleanly, leak oracle caught {d}\n", .{ n_ops, fired, caught });
    try testing.expect(caught >= 1);
}

// ── The fused QSA pooled-key kernel against the composed chain ──

// Bar: bit-identical at every block count, rope position, table and key range the indexer hands it.

// ── The QSA all-visible identity skip at the engagement boundary ──

// ── Fused QSA block select (sushi_qsa_select) ──

test "decode profile rows: 0 and absent are off, a bare switch is one row" {
    try testing.expectEqual(@as(c_int, 0), decodeProfileRowsFrom(null));
    try testing.expectEqual(@as(c_int, 0), decodeProfileRowsFrom("0"));
    try testing.expectEqual(@as(c_int, 1), decodeProfileRowsFrom("on"));
    try testing.expectEqual(@as(c_int, 8), decodeProfileRowsFrom("8"));
}

test "diagEnvValueOn: absent, empty or 0 is off" {
    try testing.expectEqual(false, diagEnvValueOn(null));
    try testing.expectEqual(false, diagEnvValueOn("0"));
    try testing.expectEqual(false, diagEnvValueOn(""));
    try testing.expectEqual(true, diagEnvValueOn("1"));
    try testing.expectEqual(true, diagEnvValueOn("on"));
}

test "diagEnvOnCached answers once and latches" {
    var cache: ?bool = null;
    try testing.expectEqual(false, diagEnvOnCached(&cache, "SUSHI_NO_SUCH_DIAG_SWITCH_PROBE"));
    try testing.expectEqual(@as(?bool, false), cache);
    cache = true;
    try testing.expectEqual(true, diagEnvOnCached(&cache, "SUSHI_NO_SUCH_DIAG_SWITCH_PROBE"));
}

test "row kernel declines a CPU stream" {
    const s = mlx.gpuStream();
    var fx = try @import("mtp.zig").RerankFixture.init(s, 512, 2560, 4, 64, 11);
    defer fx.deinit();
    var prng = std.Random.DefaultPrng.init(12);
    const x = try attn256RandBf16(prng.random(), &.{ 2, 2560 }, s);
    defer _ = mlx.mlx_array_free(x);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const got = try msvQmvRows(cpu, x, fx.xfm.lm_head_w, fx.xfm.lm_head_s, fx.xfm.lm_head_b, 4, 64);
    defer if (got) |y| {
        _ = mlx.mlx_array_free(y);
    };
    try testing.expect(got == null);
}

test "row kernel declines scales and biases with another dtype" {
    const s = mlx.gpuStream();
    var fx = try @import("mtp.zig").RerankFixture.init(s, 512, 2560, 4, 64, 11);
    defer fx.deinit();
    var prng = std.Random.DefaultPrng.init(12);
    const x = try attn256RandBf16(prng.random(), &.{ 2, 2560 }, s);
    defer _ = mlx.mlx_array_free(x);
    var sc32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc32);
    var bi32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi32);
    try mlx.check(mlx.mlx_astype(&sc32, fx.xfm.lm_head_s, .float32, s));
    try mlx.check(mlx.mlx_astype(&bi32, fx.xfm.lm_head_b, .float32, s));
    for ([_][2]mlx.mlx_array{ .{ sc32, fx.xfm.lm_head_b }, .{ fx.xfm.lm_head_s, bi32 } }) |pair| {
        const got = try msvQmvRows(s, x, fx.xfm.lm_head_w, pair[0], pair[1], 4, 64);
        defer if (got) |y| {
            _ = mlx.mlx_array_free(y);
        };
        try testing.expect(got == null);
    }
}

test "row kernel cache reuses alternating full-shape configurations" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var fx = try @import("mtp.zig").RerankFixture.init(s, 512, 2560, 8, 64, 11);
    defer fx.deinit();
    var prng = std.Random.DefaultPrng.init(12);
    const before = sushi_qmv_rows_cfgs.builds;
    for ([_]c_int{ 3, 5, 3 }) |rows| {
        const x = try attn256RandBf16(prng.random(), &.{ rows, 2560 }, s);
        defer _ = mlx.mlx_array_free(x);
        const y = (try msvQmvRows(s, x, fx.xfm.lm_head_w, fx.xfm.lm_head_s, fx.xfm.lm_head_b, 8, 64)) orelse return error.HeadRowsDeclined;
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_array_eval(y));
    }
    try testing.expectEqual(@as(usize, 2), sushi_qmv_rows_cfgs.builds - before);
}

pub fn verifyRollbackEvalStorage(entry: *const KVCacheEntry) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    inline for (.{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" }) |field| {
        const value = @field(entry, field);
        if (value.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, value);
    }
    try mlx.check(mlx.mlx_eval(vec));
}

pub fn verifyRollbackValidPrefixEqual(a: *const KVCacheEntry, b: *const KVCacheEntry, s: mlx.mlx_stream) !void {
    try testing.expectEqual(a.initialized, b.initialized);
    try testing.expectEqual(a.offset, b.offset);
    if (!a.initialized) return;
    inline for (.{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" }) |field| {
        const av = @field(a, field);
        const bv = @field(b, field);
        try testing.expectEqual(av.ctx == null, bv.ctx == null);
        if (av.ctx != null) {
            const ash = mlx.getShape(av);
            const bsh = mlx.getShape(bv);
            var aslice = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(aslice);
            var bslice = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bslice);
            try mlx.check(mlx.mlx_slice(&aslice, av, &.{ 0, 0, 0, 0 }, 4, &.{ ash[0], ash[1], @intCast(a.offset), ash[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
            try mlx.check(mlx.mlx_slice(&bslice, bv, &.{ 0, 0, 0, 0 }, 4, &.{ bsh[0], bsh[1], @intCast(b.offset), bsh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
            try testing.expect(try qsaArraysAllEqual(aslice, bslice, s));
        }
    }
}

test "verify expert reuse preserves stock gather reductions" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xE4E4);
    for ([_][2]c_int{.{ 640, 2560 }}) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weight = try attn256RandBf16(prng.random(), &.{ 64, n, k }, s);
        defer _ = mlx.mlx_array_free(weight);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]c_int{ 20, 30, 40, 50, 60, 80, 200 }) |count| for (0..3) |pattern| {
            var host_ids: [200]u32 = undefined;
            var expert: u32 = 0;
            var remain: usize = 0;
            for (host_ids[0..@intCast(count)], 0..) |*id, i| {
                if (remain == 0) {
                    if (i != 0) expert += 1;
                    remain = switch (pattern) {
                        0 => @intCast(@divTrunc(count + 63, 64)),
                        1 => 4,
                        else => ([_]usize{ 1, 3, 6 })[expert % 3],
                    };
                }
                id.* = expert;
                remain -= 1;
            }
            const ids = mlx.mlx_array_new_data(&host_ids, &.{count}, 1, .uint32);
            defer _ = mlx.mlx_array_free(ids);
            const x = try attn256RandBf16(prng.random(), &.{ count, 1, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (count == 20 and pattern == 0) {
                try testing.expect((try verifyExpertReuse(s, x, w, sc, .{}, ids)) == null);
                const cpu = mlx.mlx_default_cpu_stream_new();
                defer _ = mlx.mlx_stream_free(cpu);
                try testing.expect((try verifyExpertReuse(cpu, x, w, sc, bi, ids)) == null);
                var x32 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x32);
                try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
                try testing.expect((try verifyExpertReuse(s, x32, w, sc, bi, ids)) == null);
                var signed_ids = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(signed_ids);
                try mlx.check(mlx.mlx_astype(&signed_ids, ids, .int32, s));
                try testing.expect((try verifyExpertReuse(s, x, w, sc, bi, signed_ids)) == null);
                var large_x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(large_x);
                var large_ids = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(large_ids);
                try mlx.check(mlx.mlx_zeros(&large_x, &.{ 256, 1, k }, 3, .bfloat16, s));
                try mlx.check(mlx.mlx_zeros(&large_ids, &.{256}, 1, .uint32, s));
                try testing.expect((try verifyExpertReuse(s, large_x, w, sc, bi, large_ids)) == null);
            }
            var expected = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(expected);
            try gatherExpertMm(&expected, x, w, sc, bi, .{}, ids, 4, 64, .affine, true, s);
            const actual = (try verifyExpertReuse(s, x, w, sc, bi, ids)) orelse return error.ExpertReuseDeclined;
            defer _ = mlx.mlx_array_free(actual);
            const same = try qsaArraysAllEqual(expected, actual, s);
            if (!same) std.debug.print("[expert-reuse-mismatch] K={d} N={d} count={d} pattern={d}\n", .{ k, n, count, pattern });
            try testing.expect(same);
        };
    }
}

fn expertReduceReference(s: mlx.mlx_stream, down: mlx.mlx_array, inverse: mlx.mlx_array, scores: mlx.mlx_array) !mlx.mlx_array {
    const tokens = mlx.getShape(scores)[1];
    var unsorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(unsorted);
    try mlx.check(mlx.mlx_take_axis(&unsorted, down, inverse, 0, s));
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_reshape(&rows, unsorted, &.{ 1, tokens, 10, 2560 }, 4, s));
    var expanded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(expanded);
    try mlx.check(mlx.mlx_expand_dims(&expanded, scores, -1, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, rows, expanded, s));
    var expected = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(expected);
    try mlx.check(mlx.mlx_sum_axis(&expected, weighted, -2, false, s));
    return expected;
}

test "verify expert reduction preserves unsort multiply and sum" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xE55E);
    for ([_]c_int{ 2, 3, 4, 5, 6, 8, 10, 12, 20, 32 }) |tokens| {
        const count = tokens * 10;
        const down = try attn256RandBf16(prng.random(), &.{ count, 2560 }, s);
        defer _ = mlx.mlx_array_free(down);
        const scores = try attn256RandBf16(prng.random(), &.{ 1, tokens, 10 }, s);
        defer _ = mlx.mlx_array_free(scores);
        var perm: [320]u32 = undefined;
        for (perm[0..@intCast(count)], 0..) |*id, i| id.* = @intCast(i);
        prng.random().shuffle(u32, perm[0..@intCast(count)]);
        const inverse = mlx.mlx_array_new_data(&perm, &.{count}, 1, .uint32);
        defer _ = mlx.mlx_array_free(inverse);
        if (tokens == 4) {
            try testing.expect((try verifyExpertReduce(s, down, inverse, .{})) == null);
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try testing.expect((try verifyExpertReduce(cpu, down, inverse, scores)) == null);
            var down32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(down32);
            try mlx.check(mlx.mlx_astype(&down32, down, .float32, s));
            try testing.expect((try verifyExpertReduce(s, down32, inverse, scores)) == null);
            var signed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(signed);
            try mlx.check(mlx.mlx_astype(&signed, inverse, .int32, s));
            try testing.expect((try verifyExpertReduce(s, down, signed, scores)) == null);
            var batched = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(batched);
            try mlx.check(mlx.mlx_reshape(&batched, scores, &.{ 2, 2, 10 }, 3, s));
            try testing.expect((try verifyExpertReduce(s, down, inverse, batched)) == null);
        }
        const expected = try expertReduceReference(s, down, inverse, scores);
        defer _ = mlx.mlx_array_free(expected);
        const actual = (try verifyExpertReduce(s, down, inverse, scores)) orelse return error.ExpertReduceDeclined;
        defer _ = mlx.mlx_array_free(actual);
        try testing.expect(try qsaArraysAllEqual(expected, actual, s));
    }
}

test "verify route packing preserves stable order and row indices" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var ids: [320]u32 = undefined;
    for ([_]c_int{ 20, 30, 40, 50, 60, 80, 120, 200, 320 }) |count| {
        for (0..3) |pattern| {
            for (ids[0..@intCast(count)], 0..) |*id, i| id.* = switch (pattern) {
                0 => @intCast((i * 37 + 11) % 17),
                1 => 7,
                else => @intCast(@as(usize, @intCast(count)) - i),
            };
            const input = mlx.mlx_array_new_data(&ids, &.{count}, 1, .uint32);
            defer _ = mlx.mlx_array_free(input);
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try testing.expect((try verifyRoutePack(cpu, input, 10)) == null);
            try testing.expect((try verifyRoutePack(s, input, 8)) == null);
            try testing.expect((try verifyRoutePack(s, .{}, 10)) == null);
            var got = (try verifyRoutePack(s, input, 10)) orelse return error.RoutePackDeclined;
            defer got.deinit();
            var order = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(order);
            var inverse = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(inverse);
            var sorted = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sorted);
            var lhs = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(lhs);
            try mlx.check(mlx.mlx_argsort_axis(&order, input, 0, s));
            try mlx.check(mlx.mlx_argsort_axis(&inverse, order, 0, s));
            try mlx.check(mlx.mlx_take_axis(&sorted, input, order, 0, s));
            try mlx.check(mlx.mlx_floor_divide(&lhs, order, cachedScalarInt(10), s));
            try testing.expect(try qsaArraysAllEqual(got.inverse, inverse, s));
            try testing.expect(try qsaArraysAllEqual(got.sorted, sorted, s));
            try testing.expect(try qsaArraysAllEqual(got.lhs, lhs, s));
        }
    }
}

test "verify indexed expert inputs preserve projections without activation copies" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x1D1E);
    const weight = try attn256RandBf16(prng.random(), &.{ 64, 640, 2560 }, s);
    defer _ = mlx.mlx_array_free(weight);
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
    for ([_]c_int{ 2, 3, 4, 5, 6, 8, 20 }) |tokens| {
        const count = tokens * 10;
        const x = try attn256RandBf16(prng.random(), &.{ tokens, 2560 }, s);
        defer _ = mlx.mlx_array_free(x);
        var left: [200]u32 = undefined;
        var right: [200]u32 = undefined;
        for (left[0..@intCast(count)], right[0..@intCast(count)], 0..) |*l, *r, i| {
            l.* = @intCast(i / 10);
            r.* = @intCast(i / @as(usize, @intCast(@divTrunc(count + 63, 64))));
        }
        prng.random().shuffle(u32, left[0..@intCast(count)]);
        const lhs = mlx.mlx_array_new_data(&left, &.{count}, 1, .uint32);
        defer _ = mlx.mlx_array_free(lhs);
        const rhs = mlx.mlx_array_new_data(&right, &.{count}, 1, .uint32);
        defer _ = mlx.mlx_array_free(rhs);
        const start_copy = mlx.op_count.load(.monotonic);
        var copied = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(copied);
        try mlx.check(mlx.mlx_take_axis(&copied, x, lhs, 0, s));
        var expanded = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(expanded);
        try mlx.check(mlx.mlx_reshape(&expanded, copied, &.{ count, 1, 2560 }, 3, s));
        const copy_ops = mlx.op_count.load(.monotonic) - start_copy;
        const start_view = mlx.op_count.load(.monotonic);
        const view = (try verifyIndexedExpertInput(s, x, lhs, 64)) orelse return error.IndexedExpertInputDeclined;
        defer _ = mlx.mlx_array_free(view);
        const view_ops = mlx.op_count.load(.monotonic) - start_view;
        try testing.expectEqualSlices(c_int, &.{ tokens, 1, 2560 }, mlx.getShape(view));
        try testing.expect(view_ops < copy_ops);
        var expected = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(expected);
        try gatherExpertMm(&expected, expanded, w, sc, bi, .{}, rhs, 4, 64, .affine, true, s);
        var actual = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(actual);
        try gatherExpertMm(&actual, view, w, sc, bi, lhs, rhs, 4, 64, .affine, true, s);
        try testing.expect(try qsaArraysAllEqual(expected, actual, s));
        // The sorted-matrix boundary is four assignments per expert. Ten
        // experts crosses it at width 4, but remains eligible at widths 2–3.
        const boundary_experts = @divTrunc(count, 4);
        try testing.expect((try verifyIndexedExpertInput(s, x, lhs, boundary_experts)) == null);
        const sparse_view = (try verifyIndexedExpertInput(s, x, lhs, boundary_experts + 1)) orelse return error.IndexedExpertInputDeclined;
        defer _ = mlx.mlx_array_free(sparse_view);
        try testing.expect((try verifyIndexedExpertInput(s, x, lhs, 0)) == null);
        if (tokens == 4) {
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try testing.expect((try verifyIndexedExpertInput(cpu, x, lhs, 64)) == null);
            var x32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x32);
            try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
            try testing.expect((try verifyIndexedExpertInput(s, x32, lhs, 64)) == null);
            var signed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(signed);
            try mlx.check(mlx.mlx_astype(&signed, lhs, .int32, s));
            try testing.expect((try verifyIndexedExpertInput(s, x, signed, 64)) == null);
            try testing.expect((try verifyIndexedExpertInput(s, view, lhs, 64)) == null);
        }
    }
}

test "MTP coarse pairs preserve row-kernel logits (MTP_LMHEAD_TEST_FILE)" {
    const path = std.c.getenv("MTP_LMHEAD_TEST_FILE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var weights = try model_mod.loadWeightsSingleFile(testing.allocator, std.mem.span(path));
    defer weights.deinit();
    const w = weights.get("language_model.lm_head.weight") orelse return error.MissingLmHead;
    const sc = weights.get("language_model.lm_head.scales") orelse return error.MissingLmHead;
    const bi = weights.get("language_model.lm_head.biases") orelse return error.MissingLmHead;
    var coarse = try mtp_mod.requantizeRows(s, w, sc, bi, 64, 8, "affine", 64, 3, 32768);
    defer coarse.deinit();
    var prng = std.Random.DefaultPrng.init(0xC0A45E);
    for ([_]c_int{ 2, 3, 4, 8 }) |rows| {
        const x = try attn256RandBf16(prng.random(), &.{ rows, 1, 2560 }, s);
        defer _ = mlx.mlx_array_free(x);
        const reference = (try msvQmvRows(s, x, coarse.w, coarse.s, coarse.b, 3, 64)) orelse return error.HeadRowsDeclined;
        defer _ = mlx.mlx_array_free(reference);
        const actual = (try mtpCoarsePairs(s, x, coarse.w, coarse.s, coarse.b, 3, 64)) orelse return error.CoarsePairsDeclined;
        defer _ = mlx.mlx_array_free(actual);
        try testing.expect(try qsaArraysAllEqual(reference, actual, s));
        if (rows == 4) {
            try testing.expect((try mtpCoarsePairs(s, x, coarse.w, coarse.s, .{}, 3, 64)) == null);
            try testing.expect((try mtpCoarsePairs(s, x, coarse.w, coarse.s, coarse.b, 2, 64)) == null);
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try testing.expect((try mtpCoarsePairs(cpu, x, coarse.w, coarse.s, coarse.b, 3, 64)) == null);
            var wide = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wide);
            try mlx.check(mlx.mlx_reshape(&wide, x, &.{ 2, 2, 2560 }, 3, s));
            try testing.expect((try mtpCoarsePairs(s, wide, coarse.w, coarse.s, coarse.b, 3, 64)) == null);
            var x32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x32);
            try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
            try testing.expect((try mtpCoarsePairs(s, x32, coarse.w, coarse.s, coarse.b, 3, 64)) == null);
        }
    }
}

fn seBf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn seFillBf16(rnd: std.Random, buf: []u16) void {
    for (buf) |*v| {
        const f = rnd.float(f32) * 2.0 - 1.0;
        v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
    }
}

fn seHash(arr: mlx.mlx_array) !u64 {
    try mlx.check(mlx.mlx_array_eval(arr));
    const p = mlx.mlx_array_data_bfloat16(arr) orelse return error.Bf16Unreadable;
    var h: u64 = 0xcbf29ce484222325;
    for (p[0..mlx.mlx_array_size(arr)]) |v| {
        h ^= v;
        h *%= 0x100000001b3;
    }
    return h;
}

fn seSplitViews(s: mlx.mlx_stream, gate_up: mlx.mlx_array, down_raw: mlx.mlx_array, u: c_int, inter: c_int, hidden: c_int) !struct { gate: mlx.mlx_array, up: mlx.mlx_array, down: mlx.mlx_array } {
    const strides = [_]c_int{ 1, 1, 1 };
    const axes = [_]c_int{ 0, 2, 1 };
    var gate_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_raw);
    try mlx.check(mlx.mlx_slice(&gate_raw, gate_up, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ u, inter, hidden }, 3, &strides, 3, s));
    var up_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up_raw);
    try mlx.check(mlx.mlx_slice(&up_raw, gate_up, &[_]c_int{ 0, inter, 0 }, 3, &[_]c_int{ u, 2 * inter, hidden }, 3, &strides, 3, s));
    var gate = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_transpose_axes(&gate, gate_raw, &axes, 3, s));
    var up = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(up);
    try mlx.check(mlx.mlx_transpose_axes(&up, up_raw, &axes, 3, s));
    var down = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(down);
    try mlx.check(mlx.mlx_transpose_axes(&down, down_raw, &axes, 3, s));
    return .{ .gate = gate, .up = up, .down = down };
}

fn seTruth(
    alloc: std.mem.Allocator,
    x: [*]const u16,
    gu: [*]const u16,
    dn: [*]const u16,
    slots: []const i32,
    w: [*]const u16,
    rows: usize,
    topk: usize,
    hidden: usize,
    inter: usize,
) ![]f32 {
    const out = try alloc.alloc(f32, rows * hidden);
    @memset(out, 0);
    const hvals = try alloc.alloc(f32, inter);
    defer alloc.free(hvals);
    for (0..rows) |r| {
        for (0..topk) |k| {
            const slot: usize = @intCast(slots[r * topk + k]);
            for (0..inter) |j| {
                var g: f32 = 0;
                var u: f32 = 0;
                const grow = (slot * 2 * inter + j) * hidden;
                const urow = (slot * 2 * inter + inter + j) * hidden;
                for (0..hidden) |c| {
                    const xv = seBf16(x[r * hidden + c]);
                    g += xv * seBf16(gu[grow + c]);
                    u += xv * seBf16(gu[urow + c]);
                }
                hvals[j] = (g / (1.0 + @exp(-g))) * u;
            }
            const wv = seBf16(w[r * topk + k]);
            for (0..hidden) |o| {
                var acc: f32 = 0;
                const drow = (slot * hidden + o) * inter;
                for (0..inter) |j| acc += hvals[j] * seBf16(dn[drow + j]);
                out[r * hidden + o] += wv * acc;
            }
        }
    }
    return out;
}

const SeErr = struct { max: f32, rms: f32 };

fn seUlpBf16(x: f32) f32 {
    const a = @abs(x);
    if (a == 0 or !std.math.isFinite(a)) return 0;
    return @exp2(@floor(@log2(a)) - 7.0);
}

fn seErrStats(alloc: std.mem.Allocator, got: mlx.mlx_array, truth: []const f32) !SeErr {
    try mlx.check(mlx.mlx_array_eval(got));
    const p = mlx.mlx_array_data_bfloat16(got) orelse return error.Bf16Unreadable;
    _ = alloc;
    var max: f32 = 0;
    var sum: f64 = 0;
    for (truth, 0..) |t, i| {
        const v = seBf16(p[i]);
        if (!std.math.isFinite(v)) return error.NonFiniteOutput;
        const d = @abs(v - t);
        if (d > max) max = d;
        sum += @as(f64, d) * @as(f64, d);
    }
    return .{ .max = max, .rms = @floatCast(@sqrt(sum / @as(f64, @floatFromInt(truth.len)))) };
}

fn seSlots(alloc: std.mem.Allocator, rows: usize, topk: usize, u: usize) ![]i32 {
    const out = try alloc.alloc(i32, rows * topk);
    for (0..rows) |r| {
        for (0..topk) |k| out[r * topk + k] = @intCast((r * 7 + k * 5 + 1) % u);
        out[r * topk + 2] = out[r * topk + 0];
        out[r * topk + 5] = out[r * topk + 1];
        if (r > 0) out[r * topk + 7] = out[0 * topk + 3];
    }
    return out;
}

test "streamed expert kernel arm is no worse than the composite arm against fp32 truth" {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const hidden: c_int = 2560;
    const inter: c_int = 640;
    const topk: c_int = 10;
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 0x5eed51ab));
    var worse: usize = 0;
    const SE_DRAWS: usize = 4;

    for ([_]c_int{ 16, 128 }) |u| {
        var gu = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gu);
        try mlx.check(mlx.mlx_random_normal(&gu, &[_]c_int{ u, 2 * inter, hidden }, 3, .bfloat16, 0.0, 0.02, key, s));
        try mlx.check(mlx.mlx_array_eval(gu));
        var dn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dn);
        try mlx.check(mlx.mlx_random_normal(&dn, &[_]c_int{ u, hidden, inter }, 3, .bfloat16, 0.0, 0.02, key, s));
        try mlx.check(mlx.mlx_array_eval(dn));
        const views = try seSplitViews(s, gu, dn, u, inter, hidden);
        defer _ = mlx.mlx_array_free(views.gate);
        defer _ = mlx.mlx_array_free(views.up);
        defer _ = mlx.mlx_array_free(views.down);

        for ([_]c_int{ 1, 2, 4 }) |rows| {
            for (0..SE_DRAWS) |draw| {
                var dkey = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(dkey);
                try mlx.check(mlx.mlx_random_key(&dkey, 0xa5a50000 + @as(u64, draw) * 7919 + @as(u64, @intCast(u))));
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ rows, hidden }, 2, .bfloat16, 0.0, 1.0, dkey, s));
                try mlx.check(mlx.mlx_array_eval(x));
                var w = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(w);
                try mlx.check(mlx.mlx_random_normal(&w, &[_]c_int{ rows, topk }, 2, .bfloat16, 0.0, 1.0, dkey, s));
                try mlx.check(mlx.mlx_array_eval(w));
                const slots_host = try seSlots(alloc, @intCast(rows), @intCast(topk), @intCast(u));
                defer alloc.free(slots_host);
                const slots = mlx.mlx_array_new_data(slots_host.ptr, &[_]c_int{ rows, topk }, 2, .int32);
                defer _ = mlx.mlx_array_free(slots);

                const op = StreamedExpertOperands{
                    .x_rows = x,
                    .gate = views.gate,
                    .up = views.up,
                    .down = views.down,
                    .slab_gate_up = gu,
                    .slab_down = dn,
                    .slots = slots,
                    .weights = w,
                };
                const truth = try seTruth(
                    alloc,
                    mlx.mlx_array_data_bfloat16(x).?,
                    mlx.mlx_array_data_bfloat16(gu).?,
                    mlx.mlx_array_data_bfloat16(dn).?,
                    slots_host,
                    mlx.mlx_array_data_bfloat16(w).?,
                    @intCast(rows),
                    @intCast(topk),
                    @intCast(hidden),
                    @intCast(inter),
                );
                defer alloc.free(truth);

                const composite = try streamedExpertCompute(s, op, rows, false);
                defer _ = mlx.mlx_array_free(composite);
                const kernel = try streamedExpertCompute(s, op, rows, true);
                defer _ = mlx.mlx_array_free(kernel);
                try std.testing.expectEqualSlices(c_int, &[_]c_int{ rows, hidden }, mlx.getShape(composite));
                try std.testing.expectEqualSlices(c_int, &[_]c_int{ rows, hidden }, mlx.getShape(kernel));
                const ce = try seErrStats(alloc, composite, truth);
                const ke = try seErrStats(alloc, kernel, truth);
                var tmax: f32 = 0;
                for (truth) |v| {
                    if (@abs(v) > tmax) tmax = @abs(v);
                }
                const allow: f32 = if (expert_bf16.downKernelPreferred(@intCast(rows))) 0 else seUlpBf16(tmax);
                const bad = ke.max > ce.max + allow or ke.rms > ce.rms;
                if (bad) {
                    std.debug.print("streamed-expert U={d:>4} R={d} draw={d}: kernel max={e:.3} rms={e:.3} | composite max={e:.3} rms={e:.3} | tmax={e:.3} allow={e:.3} [WORSE]\n", .{ u, rows, draw, ke.max, ke.rms, ce.max, ce.rms, tmax, allow });
                    worse += 1;
                }
            }
        }
        _ = mlx.mlx_clear_cache();
    }
    try std.testing.expectEqual(@as(usize, 0), worse);
}

test "streamed expert composite arm keeps the pre-change bytes" {
    try mlx.requireGpu();
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    const hidden: c_int = 64;
    const inter: c_int = 32;
    const u: c_int = 8;
    const topk: c_int = 10;
    var prng = std.Random.DefaultPrng.init(0xbf16c0de);
    const rnd = prng.random();

    const gu_host = try alloc.alloc(u16, @intCast(u * 2 * inter * hidden));
    defer alloc.free(gu_host);
    seFillBf16(rnd, gu_host);
    const dn_host = try alloc.alloc(u16, @intCast(u * hidden * inter));
    defer alloc.free(dn_host);
    seFillBf16(rnd, dn_host);
    const gu = mlx.mlx_array_new_data(gu_host.ptr, &[_]c_int{ u, 2 * inter, hidden }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(gu);
    const dn = mlx.mlx_array_new_data(dn_host.ptr, &[_]c_int{ u, hidden, inter }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(dn);
    const views = try seSplitViews(s, gu, dn, u, inter, hidden);
    defer _ = mlx.mlx_array_free(views.gate);
    defer _ = mlx.mlx_array_free(views.up);
    defer _ = mlx.mlx_array_free(views.down);

    const pinned = [_]u64{ 0xc39bfc6d04f4e917, 0x1e43226d66b1fd96 };
    for ([_]c_int{ 1, 3 }, pinned) |rows, want| {
        const x_host = try alloc.alloc(u16, @intCast(rows * hidden));
        defer alloc.free(x_host);
        seFillBf16(rnd, x_host);
        const w_host = try alloc.alloc(u16, @intCast(rows * topk));
        defer alloc.free(w_host);
        seFillBf16(rnd, w_host);
        const slots_host = try alloc.alloc(i32, @intCast(rows * topk));
        defer alloc.free(slots_host);
        for (slots_host, 0..) |*v, i| v.* = @intCast((i * 3 + 1) % @as(usize, @intCast(u)));

        const x = mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{ rows, hidden }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        const slots = mlx.mlx_array_new_data(slots_host.ptr, &[_]c_int{ rows, topk }, 2, .int32);
        defer _ = mlx.mlx_array_free(slots);
        const w = mlx.mlx_array_new_data(w_host.ptr, &[_]c_int{ rows, topk }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(w);

        const out = try streamedExpertCompute(s, .{
            .x_rows = x,
            .gate = views.gate,
            .up = views.up,
            .down = views.down,
            .slab_gate_up = gu,
            .slab_down = dn,
            .slots = slots,
            .weights = w,
        }, rows, false);
        defer _ = mlx.mlx_array_free(out);
        try std.testing.expectEqualSlices(c_int, &[_]c_int{ rows, hidden }, mlx.getShape(out));
        try std.testing.expectEqual(want, try seHash(out));
    }
}

test "imatrix capture keys the streamed MoE statistics on GLOBAL expert ids, on every arm" {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const hidden: c_int = 64;
    const inter: c_int = 32;
    const u: c_int = 8;
    const experts: c_int = 16; // the slab holds 8 of them; ids 8..15 are the resident half
    const topk: c_int = 10;
    var prng = std.Random.DefaultPrng.init(0x1A7A1);
    const rnd = prng.random();

    const gu_host = try alloc.alloc(u16, @intCast(u * 2 * inter * hidden));
    defer alloc.free(gu_host);
    seFillBf16(rnd, gu_host);
    const dn_host = try alloc.alloc(u16, @intCast(u * hidden * inter));
    defer alloc.free(dn_host);
    seFillBf16(rnd, dn_host);
    const gu = mlx.mlx_array_new_data(gu_host.ptr, &[_]c_int{ u, 2 * inter, hidden }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(gu);
    const dn = mlx.mlx_array_new_data(dn_host.ptr, &[_]c_int{ u, hidden, inter }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(dn);
    const views = try seSplitViews(s, gu, dn, u, inter, hidden);
    defer _ = mlx.mlx_array_free(views.gate);
    defer _ = mlx.mlx_array_free(views.up);
    defer _ = mlx.mlx_array_free(views.down);

    // rows 1 (unsorted composite) and 2 (sorted composite), each on the kernel arm too.
    for ([_]c_int{ 1, 2 }) |rows| {
        for ([_]bool{ false, true }) |kernels| {
            const col = try imatrix_capture.Collector.init(alloc, s, "/dev/null", 3, experts, .qwen4_exp);
            defer col.deinit();
            const x_host = try alloc.alloc(u16, @intCast(rows * hidden));
            defer alloc.free(x_host);
            seFillBf16(rnd, x_host);
            const w_host = try alloc.alloc(u16, @intCast(rows * topk));
            defer alloc.free(w_host);
            seFillBf16(rnd, w_host);
            const slots_host = try seSlots(alloc, @intCast(rows), @intCast(topk), @intCast(u));
            defer alloc.free(slots_host);
            const global_host = try alloc.alloc(i32, slots_host.len);
            defer alloc.free(global_host);
            for (slots_host, 0..) |slot, i| global_host[i] = slot + u;

            const x = mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{ rows, hidden }, 2, .bfloat16);
            defer _ = mlx.mlx_array_free(x);
            const w = mlx.mlx_array_new_data(w_host.ptr, &[_]c_int{ rows, topk }, 2, .bfloat16);
            defer _ = mlx.mlx_array_free(w);
            const slots = mlx.mlx_array_new_data(slots_host.ptr, &[_]c_int{ rows, topk }, 2, .int32);
            defer _ = mlx.mlx_array_free(slots);
            const globals = mlx.mlx_array_new_data(global_host.ptr, &[_]c_int{ rows, topk }, 2, .int32);
            defer _ = mlx.mlx_array_free(globals);

            const out = try streamedExpertCompute(s, .{
                .x_rows = x,
                .gate = views.gate,
                .up = views.up,
                .down = views.down,
                .slab_gate_up = gu,
                .slab_down = dn,
                .slots = slots,
                .weights = w,
                .imatrix = .{ .collector = col, .layer = 2, .global_ids = globals },
            }, rows, kernels);
            defer _ = mlx.mlx_array_free(out);

            const slot = &col.layers[2];
            try std.testing.expectEqual(@as(u64, @intCast(rows)), slot.tokens);
            try std.testing.expectEqualSlices(c_int, &[_]c_int{ experts, hidden }, mlx.getShape(slot.gu));
            try std.testing.expectEqualSlices(c_int, &[_]c_int{ experts, inter }, mlx.getShape(slot.down));
            try mlx.check(mlx.mlx_array_eval(slot.rows));
            const routed = mlx.mlx_array_data_float32(slot.rows).?;
            var low: f32 = 0;
            var high: f32 = 0;
            for (0..@intCast(u)) |e| low += routed[e];
            for (@intCast(u)..@intCast(experts)) |e| high += routed[e];
            try std.testing.expectEqual(@as(f32, 0), low);
            try std.testing.expectEqual(@as(f32, @floatFromInt(rows * topk)), high);
            // Every expert the slab served owns some mass on both statistics.
            try mlx.check(mlx.mlx_array_eval(slot.gu));
            try mlx.check(mlx.mlx_array_eval(slot.down));
            const gu_acc = mlx.mlx_array_data_float32(slot.gu).?;
            const dn_acc = mlx.mlx_array_data_float32(slot.down).?;
            for (0..@intCast(u * hidden)) |i| try std.testing.expectEqual(@as(f32, 0), gu_acc[i]);
            for (0..@intCast(u * inter)) |i| try std.testing.expectEqual(@as(f32, 0), dn_acc[i]);
            var seen: f32 = 0;
            for (@intCast(u * inter)..@intCast(experts * inter)) |i| seen += dn_acc[i];
            try std.testing.expect(seen > 0);
        }
    }
}

/// [E, out, in] affine-4/g64 expert bank — the stand-in for a streamed MXFP4
/// slab, which reaches the same quantized arms through the routing override.
fn imxQuantBank(s: mlx.mlx_stream, alloc: std.mem.Allocator, rnd: std.Random, e: c_int, out_dim: c_int, in_dim: c_int, owned: *std.ArrayList(mlx.mlx_array)) ![3]mlx.mlx_array {
    const buf = try alloc.alloc(f32, @intCast(e * out_dim * in_dim));
    defer alloc.free(buf);
    for (buf) |*v| v.* = (rnd.float(f32) - 0.5) * 0.4;
    const a32 = mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ e, out_dim, in_dim }, 3, .float32);
    defer _ = mlx.mlx_array_free(a32);
    var bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bf);
    try mlx.check(mlx.mlx_astype(&bf, a32, .bfloat16, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, bf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var out: [3]mlx.mlx_array = undefined;
    for (&out, 0..) |*a, i| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(a, triple, i));
        try owned.append(alloc, a.*);
    }
    return out;
}

test "imatrix capture on the streamed QUANTIZED MoE path keys the routing's global ids" {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const hidden: c_int = 128;
    const inter: c_int = 64;
    const u: c_int = 4; // experts the slab holds; global id = slot + u
    const experts: c_int = 8;
    const topk: c_int = 2;
    var prng = std.Random.DefaultPrng.init(0x515A);
    const rnd = prng.random();

    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (owned.items) |a| _ = mlx.mlx_array_free(a);
        owned.deinit(alloc);
    }
    const gate = try imxQuantBank(s, alloc, rnd, u, inter, hidden, &owned);
    const up = try imxQuantBank(s, alloc, rnd, u, inter, hidden, &owned);
    const down = try imxQuantBank(s, alloc, rnd, u, hidden, inter, &owned);
    const none = mlx.mlx_array{ .ctx = null };
    const one = mlx.mlx_array_new_float(1.0);
    try owned.append(alloc, one);
    const mw = MoeMlpWeights{
        .router_w = none,
        .router_s = none,
        .router_b = none,
        .switch_gate_w = gate[0],
        .switch_gate_s = gate[1],
        .switch_gate_b = gate[2],
        .switch_up_w = up[0],
        .switch_up_s = up[1],
        .switch_up_b = up[2],
        .switch_down_w = down[0],
        .switch_down_s = down[1],
        .switch_down_b = down[2],
        .shared_gate_w = none,
        .shared_gate_s = none,
        .shared_gate_b = none,
        .shared_up_w = none,
        .shared_up_s = none,
        .shared_up_b = none,
        .shared_down_w = none,
        .shared_down_s = none,
        .shared_down_b = none,
    };

    const col = try imatrix_capture.Collector.init(alloc, s, "/dev/null", 2, experts, .mimo_v2);
    defer col.deinit();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.config = .{
        .model_type = "mimo_v2",
        .num_experts = @intCast(experts),
        .num_experts_per_tok = @intCast(topk),
        .hidden_size = hidden,
        .moe_intermediate_size = inter,
        .quant_bits = 4,
        .quant_group_size = 64,
        .quant_mode = .affine,
        .hidden_act = .silu,
    };
    xfm.one = one;
    xfm.bits_cache = .{};
    xfm.qwen4 = null;
    xfm.compiled_moe_routing = null;
    xfm.compiled_gelu = null;
    xfm.compiled_geglu = null;
    xfm.cost_trace_active = false;
    xfm.imatrix = col;

    // Slots DESCEND, so the sorted arm's row order is not the routing's: a
    // statistic folded in sorted order lands on the wrong expert. Token 0 is
    // all zeros, so the experts it routes to (and only those) owe zero mass.
    for ([_]c_int{ 2, 1 }) |rows| {
        const layer: u16 = if (rows == 2) 1 else 0;
        const x_host = try alloc.alloc(u16, @intCast(rows * hidden));
        defer alloc.free(x_host);
        seFillBf16(rnd, x_host);
        if (rows == 2) @memset(x_host[0..@intCast(hidden)], 0);
        const x = mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{ 1, rows, hidden }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        const slots_host = try alloc.alloc(i32, @intCast(rows * topk));
        defer alloc.free(slots_host);
        const globals_host = try alloc.alloc(i32, slots_host.len);
        defer alloc.free(globals_host);
        for (slots_host, 0..) |*slot, i| {
            slot.* = @intCast(u - 1 - @as(i32, @intCast(i)));
            globals_host[i] = slot.* + u;
        }
        const slab_inds = mlx.mlx_array_new_data(slots_host.ptr, &[_]c_int{ 1, rows, topk }, 3, .int32);
        defer _ = mlx.mlx_array_free(slab_inds);
        const globals = mlx.mlx_array_new_data(globals_host.ptr, &[_]c_int{ rows, topk }, 2, .int32);
        defer _ = mlx.mlx_array_free(globals);
        const w_host = try alloc.alloc(u16, @intCast(rows * topk));
        defer alloc.free(w_host);
        seFillBf16(rnd, w_host);
        const scores = mlx.mlx_array_new_data(w_host.ptr, &[_]c_int{ 1, rows, topk }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(scores);

        const out = try xfm.moeMLP2WithRouter(x, x, &mw, null, false, null, .{
            .inds = slab_inds,
            .norm_scores = scores,
            .imatrix = .{ .collector = col, .layer = layer, .global_ids = globals },
        });
        defer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));

        const slot = &col.layers[layer];
        try std.testing.expectEqual(@as(u64, @intCast(rows)), slot.tokens);
        try std.testing.expect(slot.gu.ctx != null and slot.down.ctx != null and slot.rows.ctx != null);
        try std.testing.expectEqualSlices(c_int, &[_]c_int{ experts, hidden }, mlx.getShape(slot.gu));
        try std.testing.expectEqualSlices(c_int, &[_]c_int{ experts, inter }, mlx.getShape(slot.down));
        try mlx.check(mlx.mlx_array_eval(slot.rows));
        try mlx.check(mlx.mlx_array_eval(slot.gu));
        try mlx.check(mlx.mlx_array_eval(slot.down));
        const routed = mlx.mlx_array_data_float32(slot.rows).?;
        const gu_acc = mlx.mlx_array_data_float32(slot.gu).?;
        const dn_acc = mlx.mlx_array_data_float32(slot.down).?;
        // Nothing lands on a slab SLOT id, everything on the global id.
        for (0..@intCast(u)) |e| try std.testing.expectEqual(@as(f32, 0), routed[e]);
        for (0..@intCast(u * hidden)) |i| try std.testing.expectEqual(@as(f32, 0), gu_acc[i]);
        for (0..@intCast(u * inter)) |i| try std.testing.expectEqual(@as(f32, 0), dn_acc[i]);
        var high: f32 = 0;
        for (@intCast(u)..@intCast(experts)) |e| high += routed[e];
        try std.testing.expectEqual(@as(f32, @floatFromInt(rows * topk)), high);
        // Row order: token 0's experts (the two HIGHEST global ids, slots
        // descend) carry the zero row, every other routed expert real mass.
        for (@intCast(u)..@intCast(experts)) |e| {
            if (routed[e] == 0) continue;
            var mass: f32 = 0;
            for (0..@intCast(inter)) |c| mass += dn_acc[e * @as(usize, @intCast(inter)) + c];
            const zero_row = rows == 2 and e >= @as(usize, @intCast(experts - topk));
            if (zero_row) try std.testing.expectEqual(@as(f32, 0), mass) else try std.testing.expect(mass > 0);
        }
    }
}

test "a streamed layer's imatrix tap views the routing ids and is absent with no capture" {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const ids_host = [_]i32{ 7, 3, 1, 5 };
    const ids = mlx.mlx_array_new_data(&ids_host, &[_]c_int{ 1, 2, 2 }, 3, .int32);
    defer _ = mlx.mlx_array_free(ids);
    try std.testing.expect(try ImatrixTap.forLayer(null, 0, ids, 2, 2, s) == null);

    const col = try imatrix_capture.Collector.init(alloc, s, "/dev/null", 2, 8, .mimo_v2);
    defer col.deinit();
    const tap = (try ImatrixTap.forLayer(col, 1, ids, 2, 2, s)).?;
    defer tap.deinit();
    try std.testing.expectEqual(@as(u16, 1), tap.layer);
    try std.testing.expectEqualSlices(c_int, &[_]c_int{ 2, 2 }, mlx.getShape(tap.global_ids));
    try mlx.check(mlx.mlx_array_eval(tap.global_ids));
    const got = mlx.mlx_array_data_int32(tap.global_ids).?;
    for (ids_host, 0..) |want, i| try std.testing.expectEqual(want, got[i]);
}

test "streamed expert compute refuses a misaligned slab by name" {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const hidden: c_int = 64;
    const inter: c_int = 32;
    const u: c_int = 8;
    const topk: c_int = 10;
    const rows: c_int = 1;

    const n: c_int = u * 2 * inter * hidden + 1;
    const gu_host = try alloc.alloc(u16, @intCast(n));
    defer alloc.free(gu_host);
    var prng = std.Random.DefaultPrng.init(7);
    seFillBf16(prng.random(), gu_host);
    const gu_base = mlx.mlx_array_new_data(gu_host.ptr, &[_]c_int{n}, 1, .bfloat16);
    defer _ = mlx.mlx_array_free(gu_base);
    var gu_sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gu_sliced);
    try mlx.check(mlx.mlx_slice(&gu_sliced, gu_base, &[_]c_int{1}, 1, &[_]c_int{n}, 1, &[_]c_int{1}, 1, s));
    var gu_bad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gu_bad);
    try mlx.check(mlx.mlx_reshape(&gu_bad, gu_sliced, &[_]c_int{ u, 2 * inter, hidden }, 3, s));
    const gu_good = mlx.mlx_array_new_data(gu_host.ptr + 1, &[_]c_int{ u, 2 * inter, hidden }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(gu_good);
    try mlx.check(mlx.mlx_array_eval(gu_good));

    const dn_host = try alloc.alloc(u16, @intCast(u * hidden * inter));
    defer alloc.free(dn_host);
    seFillBf16(prng.random(), dn_host);
    const dn = mlx.mlx_array_new_data(dn_host.ptr, &[_]c_int{ u, hidden, inter }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(dn);
    const views = try seSplitViews(s, gu_good, dn, u, inter, hidden);
    defer _ = mlx.mlx_array_free(views.gate);
    defer _ = mlx.mlx_array_free(views.up);
    defer _ = mlx.mlx_array_free(views.down);

    const x_host = try alloc.alloc(u16, @intCast(rows * hidden));
    defer alloc.free(x_host);
    seFillBf16(prng.random(), x_host);
    const x = mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{ rows, hidden }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const slots_host = try seSlots(alloc, @intCast(rows), @intCast(topk), @intCast(u));
    defer alloc.free(slots_host);
    const slots = mlx.mlx_array_new_data(slots_host.ptr, &[_]c_int{ rows, topk }, 2, .int32);
    defer _ = mlx.mlx_array_free(slots);
    const w_host = try alloc.alloc(u16, @intCast(rows * topk));
    defer alloc.free(w_host);
    seFillBf16(prng.random(), w_host);
    const w = mlx.mlx_array_new_data(w_host.ptr, &[_]c_int{ rows, topk }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(w);

    var op = StreamedExpertOperands{
        .x_rows = x,
        .gate = views.gate,
        .up = views.up,
        .down = views.down,
        .slab_gate_up = gu_bad,
        .slab_down = dn,
        .slots = slots,
        .weights = w,
    };
    try mlx.check(mlx.mlx_array_eval(gu_bad));
    try std.testing.expect(@intFromPtr(mlx.mlx_array_data_uint8(gu_bad).?) % EXPERT_SLAB_ALIGN_BYTES != 0);
    try std.testing.expectError(error.ExpertSlabUnaligned, streamedExpertCompute(s, op, rows, true));

    op.slab_gate_up = gu_good;
    const ok = try streamedExpertCompute(s, op, rows, true);
    defer _ = mlx.mlx_array_free(ok);
    try std.testing.expectEqualSlices(c_int, &[_]c_int{ rows, hidden }, mlx.getShape(ok));
}

test "streamed expert compute falls back to the composite arm past the kernel row cap" {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const hidden: c_int = 64;
    const inter: c_int = 32;
    const u: c_int = 8;
    const topk: c_int = 10;
    const rows: c_int = expert_bf16.MAX_ROWS + 1;
    var prng = std.Random.DefaultPrng.init(0xcafe17);
    const rnd = prng.random();

    const gu_host = try alloc.alloc(u16, @intCast(u * 2 * inter * hidden));
    defer alloc.free(gu_host);
    seFillBf16(rnd, gu_host);
    const dn_host = try alloc.alloc(u16, @intCast(u * hidden * inter));
    defer alloc.free(dn_host);
    seFillBf16(rnd, dn_host);
    const x_host = try alloc.alloc(u16, @intCast(rows * hidden));
    defer alloc.free(x_host);
    seFillBf16(rnd, x_host);
    const w_host = try alloc.alloc(u16, @intCast(rows * topk));
    defer alloc.free(w_host);
    seFillBf16(rnd, w_host);
    const slots_host = try seSlots(alloc, @intCast(rows), @intCast(topk), @intCast(u));
    defer alloc.free(slots_host);

    const gu = mlx.mlx_array_new_data(gu_host.ptr, &[_]c_int{ u, 2 * inter, hidden }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(gu);
    const dn = mlx.mlx_array_new_data(dn_host.ptr, &[_]c_int{ u, hidden, inter }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(dn);
    const x = mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{ rows, hidden }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const w = mlx.mlx_array_new_data(w_host.ptr, &[_]c_int{ rows, topk }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(w);
    const slots = mlx.mlx_array_new_data(slots_host.ptr, &[_]c_int{ rows, topk }, 2, .int32);
    defer _ = mlx.mlx_array_free(slots);
    const views = try seSplitViews(s, gu, dn, u, inter, hidden);
    defer _ = mlx.mlx_array_free(views.gate);
    defer _ = mlx.mlx_array_free(views.up);
    defer _ = mlx.mlx_array_free(views.down);
    const op = StreamedExpertOperands{
        .x_rows = x,
        .gate = views.gate,
        .up = views.up,
        .down = views.down,
        .slab_gate_up = gu,
        .slab_down = dn,
        .slots = slots,
        .weights = w,
    };
    const asked = try streamedExpertCompute(s, op, rows, true);
    defer _ = mlx.mlx_array_free(asked);
    const composite = try streamedExpertCompute(s, op, rows, false);
    defer _ = mlx.mlx_array_free(composite);
    try std.testing.expectEqualSlices(c_int, &[_]c_int{ rows, hidden }, mlx.getShape(asked));
    try std.testing.expectEqual(try seHash(composite), try seHash(asked));
}

fn sqIds(host: []const u32, len: c_int) mlx.mlx_array {
    return mlx.mlx_array_new_data(host.ptr, &[_]c_int{len}, 1, .uint32);
}

fn sqRandomBf16(alloc: std.mem.Allocator, rnd: std.Random, n: usize) ![]u16 {
    const host = try alloc.alloc(u16, n);
    for (host) |*v| {
        const value: f32 = -1.0 + rnd.float(f32) * 2.0;
        v.* = @truncate(@as(u32, @bitCast(value)) >> 16);
    }
    return host;
}

const SqArm = struct {
    gate_w: mlx.mlx_array,
    gate_s: mlx.mlx_array,
    gate_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
};

test "a streamed quantized slab matches the resident bank through the same kernels" {
    try mlx.requireGpu();
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    const experts: u16 = 6;
    const hidden: u32 = 128;
    const inter: u32 = 64;
    const gs: u32 = 64;

    for ([_]u32{ 4, 8 }) |bits| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var pack = try expert_stream_mod.writeTinyQuantCheckpoint(alloc, tmp.dir, 0x9e37 + bits, experts, hidden, inter, gs, .{ bits, bits, bits });
        defer pack.deinit();
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try tmp.dir.realPath(std.testing.io, &path_buf);
        const path = path_buf[0..path_len];
        const geom = expert_stream_mod.Geometry{ .layers = 1, .experts = experts, .hidden = hidden, .intermediate = inter };
        const per_expert = try expert_stream_mod.expertBytesFor(alloc, path, geom, .quantized_split);

        var bank: [expert_stream_mod.quant.component_count]mlx.mlx_array = undefined;
        for (&bank, 0..) |*value, ci| value.* = pack.residentBank(experts, ci);
        defer for (bank) |value| {
            _ = mlx.mlx_array_free(value);
        };

        var engine = try expert_stream_mod.Engine.initWithOptions(alloc, path, geom, 3 * per_expert, s, .{
            .bounce_size = 1 << 20,
            .layout = .quantized_split,
        });
        defer engine.deinit();
        var prepared = try engine.prepareHost(0, &.{ 5, 1, 4 });
        defer prepared.deinit();
        try std.testing.expect(prepared.quantized);
        var slab: [expert_stream_mod.quant.component_count]mlx.mlx_array = undefined;
        for (&slab, 0..) |*value, ci| value.* = prepared.quantOperand(@fromBackingInt(@intCast(ci)));

        const arms = [_]SqArm{
            .{ .gate_w = bank[0], .gate_s = bank[1], .gate_b = bank[2], .up_w = bank[3], .up_s = bank[4], .up_b = bank[5], .down_w = bank[6], .down_s = bank[7], .down_b = bank[8] },
            .{ .gate_w = slab[0], .gate_s = slab[1], .gate_b = slab[2], .up_w = slab[3], .up_s = slab[4], .up_b = slab[5], .down_w = slab[6], .down_s = slab[7], .down_b = slab[8] },
        };
        const bank_decode = [_]u32{ 5, 4 };
        const slab_decode = [_]u32{ prepared.remapped[0], prepared.remapped[2] };
        const topk: c_int = 2;

        var prng = std.Random.DefaultPrng.init(0xc0ffee + bits);
        const rnd = prng.random();

        {
            const x_host = try sqRandomBf16(alloc, rnd, hidden);
            defer alloc.free(x_host);
            const x = mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{@intCast(hidden)}, 1, .bfloat16);
            defer _ = mlx.mlx_array_free(x);
            const score_host = [_]u16{ @truncate(@as(u32, @bitCast(@as(f32, 0.7))) >> 16), @truncate(@as(u32, @bitCast(@as(f32, 0.3))) >> 16) };
            const scores = mlx.mlx_array_new_data(&score_host, &[_]c_int{topk}, 1, .bfloat16);
            defer _ = mlx.mlx_array_free(scores);
            var hashes: [2]u64 = undefined;
            for (arms, [_][]const u32{ &bank_decode, &slab_decode }, 0..) |arm, ids_host, i| {
                const ids = sqIds(ids_host, topk);
                defer _ = mlx.mlx_array_free(ids);
                const act = (try gatherQmvGateUp(s, x, arm.gate_w, arm.gate_s, arm.gate_b, arm.up_w, arm.up_s, arm.up_b, ids, bits, gs, .affine)) orelse return error.SkipZigTest;
                defer _ = mlx.mlx_array_free(act);
                const out = (try gatherQmvDownReduce(s, act, arm.down_w, arm.down_s, arm.down_b, ids, scores, bits, gs, .affine)) orelse return error.SkipZigTest;
                defer _ = mlx.mlx_array_free(out);
                try std.testing.expectEqualSlices(c_int, &[_]c_int{@intCast(hidden)}, mlx.getShape(out));
                hashes[i] = try seHash(out);
            }
            try std.testing.expectEqual(hashes[0], hashes[1]);
        }

        for ([_]c_int{ 4, 64 }) |rows| {
            const n: c_int = rows * topk;
            const order = try alloc.alloc(usize, @intCast(n));
            defer alloc.free(order);
            const bank_ids = try alloc.alloc(u32, @intCast(n));
            defer alloc.free(bank_ids);
            const slab_ids = try alloc.alloc(u32, @intCast(n));
            defer alloc.free(slab_ids);
            const members = [_]u16{ 5, 1, 4 };
            for (order, 0..) |*slot, i| slot.* = (i * members.len) / @as(usize, @intCast(n));
            std.mem.sort(usize, order, prepared.remapped, struct {
                fn less(remap: []const u16, a: usize, b: usize) bool {
                    return remap[a] < remap[b];
                }
            }.less);
            for (bank_ids, slab_ids, order) |*b, *l, which| {
                b.* = members[which];
                l.* = prepared.remapped[which];
            }
            const x_host = try sqRandomBf16(alloc, rnd, @intCast(n * @as(c_int, @intCast(hidden))));
            defer alloc.free(x_host);
            const x = mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{ n, 1, @intCast(hidden) }, 3, .bfloat16);
            defer _ = mlx.mlx_array_free(x);
            const act_host = try sqRandomBf16(alloc, rnd, @intCast(n * @as(c_int, @intCast(inter))));
            defer alloc.free(act_host);
            const act = mlx.mlx_array_new_data(act_host.ptr, &[_]c_int{ n, 1, @intCast(inter) }, 3, .bfloat16);
            defer _ = mlx.mlx_array_free(act);

            var gate_hashes: [2]u64 = undefined;
            var down_hashes: [2]u64 = undefined;
            for (arms, [_][]const u32{ bank_ids, slab_ids }, 0..) |arm, ids_host, i| {
                // Repacking changes expert IDs; each sorted arm needs its own row order.
                const permutation = try alloc.alloc(u32, @intCast(n));
                defer alloc.free(permutation);
                const inverse = try alloc.alloc(u32, @intCast(n));
                defer alloc.free(inverse);
                for (permutation, 0..) |*slot, j| slot.* = @intCast(j);
                std.mem.sort(u32, permutation, ids_host, struct {
                    fn less(ids_: []const u32, a: u32, b: u32) bool {
                        return ids_[a] < ids_[b];
                    }
                }.less);
                const sorted_ids = try alloc.alloc(u32, @intCast(n));
                defer alloc.free(sorted_ids);
                for (permutation, 0..) |slot, j| {
                    sorted_ids[j] = ids_host[slot];
                    inverse[slot] = @intCast(j);
                }
                const perm = sqIds(permutation, n);
                defer _ = mlx.mlx_array_free(perm);
                const inv = sqIds(inverse, n);
                defer _ = mlx.mlx_array_free(inv);
                const ids = sqIds(sorted_ids, n);
                defer _ = mlx.mlx_array_free(ids);
                var sorted_x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(sorted_x);
                var sorted_act = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(sorted_act);
                try mlx.check(mlx.mlx_take_axis(&sorted_x, x, perm, 0, s));
                try mlx.check(mlx.mlx_take_axis(&sorted_act, act, perm, 0, s));
                var gate_out = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(gate_out);
                var gate_restored = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(gate_restored);
                try gatherExpertMm(&gate_out, sorted_x, arm.gate_w, arm.gate_s, arm.gate_b, .{ .ctx = null }, ids, bits, gs, .affine, true, s);
                try mlx.check(mlx.mlx_take_axis(&gate_restored, gate_out, inv, 0, s));
                gate_hashes[i] = try seHash(gate_restored);
                var down_out = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(down_out);
                var down_restored = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(down_restored);
                try gatherExpertMm(&down_out, sorted_act, arm.down_w, arm.down_s, arm.down_b, .{ .ctx = null }, ids, bits, gs, .affine, true, s);
                try mlx.check(mlx.mlx_take_axis(&down_restored, down_out, inv, 0, s));
                down_hashes[i] = try seHash(down_restored);
            }
            try std.testing.expectEqual(gate_hashes[0], gate_hashes[1]);
            try std.testing.expectEqual(down_hashes[0], down_hashes[1]);
        }
    }
}

test "exl3 shared add releases routed output study3" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var xfm: Transformer = undefined;
    xfm.config = .{};
    xfm.bits_cache = .{};
    xfm.compiled_geglu = null;
    xfm.allocator = testing.allocator;
    xfm.s = s;
    xfm.config.expert_layout = .exl3_k4;
    xfm.config.hidden_act = .silu;
    xfm.config.quant_bits = 4;
    xfm.config.quant_group_size = 64;
    xfm.config.moe_intermediate_size = 128;
    var rng = std.Random.DefaultPrng.init(1103);
    const x = try attn256RandBf16(rng.random(), &.{ 1, 64, 128 }, s);
    defer _ = mlx.mlx_array_free(x);
    const shared = try attn256RandBf16(rng.random(), &.{ 128, 128 }, s);
    defer _ = mlx.mlx_array_free(shared);
    const gate = try attn256RandBf16(rng.random(), &.{ 128, 1 }, s);
    defer _ = mlx.mlx_array_free(gate);
    const trh: [8 * 8 * 64]u16 = @splat(1234);
    const tr = mlx.mlx_array_new_data(&trh, &.{ 1, 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const sh: [128]u16 = @splat(0x3c00);
    const scale = mlx.mlx_array_new_data(&sh, &.{ 1, 128 }, 2, .float16);
    defer _ = mlx.mlx_array_free(scale);
    const ih: [64]u32 = @splat(0);
    const inds = mlx.mlx_array_new_data(&ih, &.{ 1, 64, 1 }, 3, .uint32);
    defer _ = mlx.mlx_array_free(inds);
    const sc: [64]u16 = @splat(0x3f80);
    const scores = mlx.mlx_array_new_data(&sc, &.{ 1, 64, 1 }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(scores);
    var mw = std.mem.zeroes(MoeMlpWeights);
    mw.switch_gate_w = tr;
    mw.switch_up_w = tr;
    mw.switch_down_w = tr;
    mw.switch_gate_s = scale;
    mw.switch_up_s = scale;
    mw.switch_down_s = scale;
    mw.switch_gate_b = scale;
    mw.switch_up_b = scale;
    mw.switch_down_b = scale;
    mw.shared_gate_w = shared;
    mw.shared_up_w = shared;
    mw.shared_down_w = shared;
    mw.shared_expert_gate_w = gate;
    mw.shared_expert_gate_s = .{ .ctx = null };
    mw.shared_expert_gate_b = .{ .ctx = null };
    var before: usize = 0;
    var after: usize = 0;
    for (0..5) |rep| {
        const out = try xfm.moeMLP2WithRouter(x, x, &mw, null, false, null, .{ .inds = inds, .norm_scores = scores });
        try mlx.check(mlx.mlx_array_eval(out));
        _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_synchronize(s));
        _ = mlx.mlx_clear_cache();
        if (rep == 0) try mlx.check(mlx.mlx_get_active_memory(&before));
    }
    try mlx.check(mlx.mlx_get_active_memory(&after));
    if (after > before + 4096) {
        std.debug.print("[study3 ownership] before={d} after={d} retained={d}\n", .{ before, after, after -| before });
        return error.TestUnexpectedResult;
    }
}

test "affine8 projections are row-identical gs32" {
    try testAffine8ProjectionRows(32);
}

test "affine8 projections are row-identical gs64" {
    try testAffine8ProjectionRows(64);
}

test "affine8 projections are row-identical gs128" {
    try testAffine8ProjectionRows(128);
}

fn testAffine8ProjectionRows(gs: u32) !void {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x8dec0de);
    for ([_]mlx.mlx_dtype{ .bfloat16, .float16, .float32 }) |dtype| {
        for ([_][2]c_int{ .{ 12288, 2560 }, .{ 512, 2560 }, .{ 2560, 6144 }, .{ 2560, 640 }, .{ 10240, 384 }, .{ 7, 256 }, .{ 8, 128 }, .{ 1, 128 }, .{ 13, 512 }, .{ 9, @intCast(gs) }, .{ 8, @intCast(gs * 3) }, .{ 8, @intCast(gs * 9) } }) |shape| {
            const n = shape[0];
            const k = shape[1];
            const bf = try testRandWeightBf16(prng.random(), &shape, s);
            defer _ = mlx.mlx_array_free(bf);
            var dense = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dense);
            try mlx.check(mlx.mlx_astype(&dense, bf, dtype, s));
            var weights = try kv_quant.quantizeAffine(s, dense, gs, 8);
            defer weights.deinit();
            for ([_]usize{ 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 32 }) |rows| {
                const x = try qkvIdentityArray(s, 1, @intCast(rows), k, 0.039 * @as(f32, @floatFromInt(rows)), dtype);
                defer _ = mlx.mlx_array_free(x);
                const together = try qmatmulBits(x, weights.q, weights.scales, weights.biases, 8, gs, .affine, s);
                defer _ = mlx.mlx_array_free(together);
                for (0..rows) |row| {
                    const begin: c_int = @intCast(row);
                    const xr = try sliceAttentionSeq(s, x, begin, begin + 1);
                    defer _ = mlx.mlx_array_free(xr);
                    const single = try qmatmulBits(xr, weights.q, weights.scales, weights.biases, 8, gs, .affine, s);
                    defer _ = mlx.mlx_array_free(single);
                    const candidate = try sliceAttentionSeq(s, together, begin, begin + 1);
                    defer _ = mlx.mlx_array_free(candidate);
                    qkvExpectBitsEqual(s, single, candidate) catch |err| {
                        std.debug.print("affine8 gs={d} N={d} K={d} rows={d} row={d} dtype={s}\n", .{ gs, n, k, rows, row, @tagName(dtype) });
                        return err;
                    };
                }
            }
        }
    }
}

test "dense projections are row-identical to serial matmul" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xded8);
    for ([_]mlx.mlx_dtype{ .bfloat16, .float16, .float32 }) |dtype| {
        for ([_][2]c_int{ .{ 512, 2560 }, .{ 1, 2560 }, .{ 4, 10240 }, .{ 2560, 640 } }) |shape| {
            const bf = try testRandWeightBf16(prng.random(), &shape, s);
            defer _ = mlx.mlx_array_free(bf);
            var dense = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dense);
            try mlx.check(mlx.mlx_astype(&dense, bf, dtype, s));
            const w = try transposeBf16Weight(dense, s);
            defer _ = mlx.mlx_array_free(w);
            const k = shape[1];
            for ([_]usize{ 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 32 }) |rows| {
                const x = try qkvIdentityArray(s, 1, @intCast(rows), k, 0.039 * @as(f32, @floatFromInt(rows)), dtype);
                defer _ = mlx.mlx_array_free(x);
                const together = try qmatmulBits(x, w, .{}, .{}, 0, 64, .affine, s);
                defer _ = mlx.mlx_array_free(together);
                for (0..rows) |row| {
                    const begin: c_int = @intCast(row);
                    const xr = try sliceAttentionSeq(s, x, begin, begin + 1);
                    defer _ = mlx.mlx_array_free(xr);
                    const single = try qmatmulBits(xr, w, .{}, .{}, 0, 64, .affine, s);
                    defer _ = mlx.mlx_array_free(single);
                    const candidate = try sliceAttentionSeq(s, together, begin, begin + 1);
                    defer _ = mlx.mlx_array_free(candidate);
                    qkvExpectBitsEqual(s, single, candidate) catch |err| {
                        std.debug.print("dense N={d} K={d} rows={d} row={d} dtype={s}\n", .{ shape[0], k, rows, row, @tagName(dtype) });
                        return err;
                    };
                }
            }
        }
    }
}

test "mimo v2 fused qkv split keeps asymmetric value rows" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const q_rows: usize = 4 * 3;
    const k_rows: usize = 2 * 3;
    const v_rows: usize = 2 * 2;
    const in_dim: usize = 5;
    var raw_data: [((q_rows + k_rows + v_rows) * in_dim)]f32 = undefined;
    for (&raw_data, 0..) |*v, i| v.* = @floatFromInt(i);
    const raw = mlx.mlx_array_new_data(
        raw_data[0..].ptr,
        &[_]c_int{ @intCast(q_rows + k_rows + v_rows), @intCast(in_dim) },
        2,
        .float32,
    );
    defer _ = mlx.mlx_array_free(raw);
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (owned.items) |arr| _ = mlx.mlx_array_free(arr);
        owned.deinit(allocator);
    }

    const parts = try splitFusedQkvAsymRows(
        raw,
        @intCast(q_rows),
        @intCast(k_rows),
        @intCast(v_rows),
        false,
        &owned,
        allocator,
        s,
    );
    try testing.expectEqualSlices(c_int, &[_]c_int{ @intCast(q_rows), @intCast(in_dim) }, mlx.getShape(parts[0]));
    try testing.expectEqualSlices(c_int, &[_]c_int{ @intCast(k_rows), @intCast(in_dim) }, mlx.getShape(parts[1]));
    try testing.expectEqualSlices(c_int, &[_]c_int{ @intCast(v_rows), @intCast(in_dim) }, mlx.getShape(parts[2]));
}



test "sushi coder uniform qwen4 and mimo banks preserve stored handles" {
    const t = std.testing;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    for ([_][]const u8{ "language_model.model", "model", "language_model.mtp" }) |prefix| {
        var w = Weights.init(t.allocator);
        defer w.deinit();
        var buf: [256]u8 = undefined;
        for ([_][]const u8{ "gate", "up", "down" }) |proj| {
            for ([_][]const u8{ "trellis", "suh", "svh" }) |part| {
                const shape: []const c_int = if (std.mem.eql(u8, part, "trellis")) &.{ 4, 8, 8, 48 } else &.{ 4, 128 };
                var arr = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_zeros(&arr, shape.ptr, @intCast(shape.len), if (shape.len == 4) .uint16 else .float16, s));
                const key = try std.fmt.allocPrint(t.allocator, "{s}.layers.0.mlp.switch_mlp.{s}_proj.{s}", .{ prefix, proj, part });
                try w.map.put(key, arr);
            }
        }
        const cfg = exl3SwitchBankConfig(.{ .n = 48 });
        const bank = try loadSwitchMlpBank(&w, &buf, prefix, 0, true, &cfg);
        const key = try std.fmt.bufPrint(&buf, "{s}.layers.0.mlp.switch_mlp.gate_proj.trellis", .{prefix});
        try t.expectEqual(w.get(key).?.ctx, bank.gate_w.ctx);
        try t.expectEqualSlices(c_int, &.{ 4, 8, 8, 48 }, mlx.getShape(bank.gate_w));
    }
}

test "sushi coder GPU grouped binders cover qwen mimo and MTP" {
    const t = std.testing;
    const s = mlx.gpuStream();
    for ([_][]const u8{ "language_model.model", "model", "language_model.mtp" }) |prefix| {
        var w = Weights.init(t.allocator);
        defer w.deinit();
        var buf: [256]u8 = undefined;
        var router = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&router, &[_]c_int{ 3, 128 }, 2, .bfloat16, s));
        try w.map.put(try std.fmt.allocPrint(t.allocator, "{s}.layers.0.mlp.gate.weight", .{prefix}), router);
        for (0..2) |group| {
            const e: c_int = @intCast(group + 1);
            for ([_][]const u8{ "gate", "up", "down" }, 0..) |proj, p| {
                const n: c_int = @intCast(32 + 16 * ((group + p) % 3));
                for ([_][]const u8{ "trellis", "suh", "svh" }, 0..) |part, i| {
                    const shape: []const c_int = if (i == 0) &.{ e, 8, 8, n } else &.{ e, 128 };
                    var arr = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_zeros(&arr, shape.ptr, @intCast(shape.len), if (i == 0) .uint16 else .float16, s));
                    try w.map.put(try std.fmt.allocPrint(t.allocator, "{s}.layers.0.mlp.switch_mlp.{s}_proj.g{d}.{s}", .{ prefix, proj, group, part }), arr);
                }
            }
        }
        var cfg = exl3SwitchBankConfig(.{ .n = 64 });
        cfg.num_experts = 512;
        cfg.num_experts_per_tok = 2;
        var bank = try loadSwitchMlpBank(&w, &buf, prefix, 0, true, &cfg);
        defer bank.ragged.?.deinit();
        try t.expectEqual(@as(usize, 2), bank.ragged.?.count);
        try t.expectEqual(@as(c_int, 2), mlx.getShape(bank.ragged.?.views[1].up.trellis)[0]);
        try t.expectEqual(@as(c_int, 64), mlx.getShape(bank.ragged.?.views[1].up.trellis)[3]);
        // The pack's tensors now share the ragged storage.
        const key = try std.fmt.bufPrint(&buf, "{s}.layers.0.mlp.switch_mlp.down_proj.g1.trellis", .{prefix});
        try t.expectEqualSlices(c_int, &.{ 2, 8, 8, 32 }, mlx.getShape(w.get(key).?));
        cfg.num_experts_per_tok = 4;
        try t.expectError(error.Exl3TopKExceedsExperts, loadSwitchMlpBank(&w, &buf, prefix, 0, true, &cfg));
    }
}

test "sushi coder GPU a loaded grouped layer keeps its stored bytes resident once" {
    const t = std.testing;
    const s = mlx.gpuStream();
    const prefix = "language_model.model";
    var w = Weights.init(t.allocator);
    defer w.deinit();
    var buf: [256]u8 = undefined;
    // A cached buffer up to two pages larger can serve an allocation, which would count its slack as resident;
    // with no cache every buffer is fresh and the count is exact.
    var cache_limit: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&cache_limit, 0));
    defer {
        var unused: usize = 0;
        _ = mlx.mlx_set_cache_limit(&unused, cache_limit);
    }
    try mlx.check(mlx.mlx_clear_cache());
    var before: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&before));
    var stored: usize = 0;
    for (0..2) |group| {
        for ([_][]const u8{ "gate", "up", "down" }, 0..) |proj, p| {
            const n: c_int = if ((group + p) % 2 == 0) 24 else 48;
            for ([_][]const u8{ "trellis", "suh", "svh" }, 0..) |part, i| {
                const shape: []const c_int = if (i == 0) &.{ 64, 16, 16, n } else &.{ 64, 256 };
                var arr = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_ones(&arr, shape.ptr, shape.len, if (i == 0) .uint16 else .float16, s));
                try mlx.check(mlx.mlx_array_eval(arr));
                stored += mlx.mlx_array_size(arr) * mlx.mlx_array_itemsize(arr);
                try w.map.put(try std.fmt.allocPrint(t.allocator, "{s}.layers.0.mlp.switch_mlp.{s}_proj.g{d}.{s}", .{ prefix, proj, group, part }), arr);
            }
        }
    }
    var cfg = exl3SwitchBankConfig(.{ .n = 64 });
    cfg.hidden_size = 256;
    cfg.moe_intermediate_size = 256;
    cfg.num_experts = 128;
    cfg.num_experts_per_tok = 8;
    var bank = try loadSwitchMlpBank(&w, &buf, prefix, 0, true, &cfg);
    defer bank.ragged.?.deinit();
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var hw: HybridMlpWeights = .{ .moe = std.mem.zeroInit(MoeMlpWeights, .{ .exl3_ragged = bank.ragged }) };
    appendHybridMlpWeights(vec, &hw);
    try mlx.check(mlx.mlx_eval(vec));
    try mlx.check(mlx.mlx_synchronize(s));
    var after: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&after));
    // The stored bytes once, plus the per-expert table; the loaded originals are gone.
    try t.expectEqual(stored + 128 * sushi_exl3.rate_groups.META_WORDS * 4, after - before);
}

test "sushi coder GPU grouped layer matches the original uniform bank" {
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var h = try Exl3MoeHarness.init(arena.allocator(), s, 1, 181);
    defer h.deinit();
    h.xfm.config.num_experts = 512;
    h.xfm.config.num_experts_per_tok = 4;
    const original = try h.xfm.moeMLP(h.x, &h.mw);
    defer _ = mlx.mlx_array_free(original);
    var slices: [18]mlx.mlx_array = undefined;
    var built: usize = 0;
    defer for (slices[0..built]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var banks: [2]sushi_exl3.Bank = undefined;
    for (0..2) |group| {
        const bank = &banks[group];
        for ([_]*sushi_exl3.Proj{ &bank.gate, &bank.up, &bank.down }) |proj| {
            for ([_]mlx.mlx_array{ h.mw.switch_gate_w, h.mw.switch_gate_s, h.mw.switch_gate_b }, 0..) |a, part| {
                const sh = mlx.getShape(a);
                var starts: [4]c_int = @splat(0);
                var ends: [4]c_int = @splat(0);
                const strides: [4]c_int = @splat(1);
                @memcpy(ends[0..sh.len], sh);
                starts[0] = @intCast(group * 2);
                ends[0] = starts[0] + 2;
                var slice = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(&slice, a, &starts, sh.len, &ends, sh.len, &strides, sh.len, s));
                slices[built] = slice;
                built += 1;
                switch (part) {
                    0 => proj.trellis = slice,
                    1 => proj.suh = slice,
                    else => proj.svh = slice,
                }
            }
        }
    }
    h.mw.exl3_ragged = try sushi_exl3.Ragged.init(s, &banks);
    defer h.mw.exl3_ragged.?.deinit();
    const grouped = try h.xfm.moeMLP(h.x, &h.mw);
    defer _ = mlx.mlx_array_free(grouped);
    // Both groups carry the uniform bank's rate, so the grouped layer is the uniform one to the byte.
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_astype(&a, original, .float32, s));
    try mlx.check(mlx.mlx_astype(&b, grouped, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    try t.expectEqualSlices(f32, mlx.mlx_array_data_float32(a).?[0..128], mlx.mlx_array_data_float32(b).?[0..128]);
}

test "argmax-only shortlist rows hold the exact logits at their ids and argmax to the lowest tied id" {
    const s = mlx.gpuStream();
    const vocab: c_int = 256;
    var cand_ids: [2][32]u32 = undefined;
    var vals: [2][32]u16 = undefined;
    for (0..32) |j| {
        cand_ids[0][j] = @intCast(200 - 5 * j);
        cand_ids[1][j] = @intCast(3 + 7 * j);
        // Integers and quarters are exact in bf16: the top half of the f32 bits.
        vals[0][j] = @truncate(@as(u32, @bitCast(-@as(f32, @floatFromInt(j)))) >> 16);
        vals[1][j] = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(j)) * 0.25)) >> 16);
    }
    // Row 0: ids 200 (slot 0) and 45 (slot 31) tie at the max; the lower id wins.
    vals[0][31] = vals[0][0];
    var lists: [2]mtp_mod.Shortlist = undefined;
    for (0..2) |r| {
        lists[r] = .{
            .cands = mlx.mlx_array_new_data(&cand_ids[r], &[_]c_int{32}, 1, .uint32),
            .exact = mlx.mlx_array_new_data(&vals[r], &[_]c_int{ 1, 1, 32 }, 3, .bfloat16),
            .rows = vocab,
        };
    }
    defer for (&lists) |*sl| sl.deinit();
    const sparse = try shortlistSparseLogits(s, &lists, vocab);
    defer _ = mlx.mlx_array_free(sparse.idx);
    defer _ = mlx.mlx_array_free(sparse.logits);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, sparse.logits, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const got = (mlx.mlx_array_data_float32(f) orelse return error.Unreadable)[0 .. 2 * 256];
    for (0..2) |r| {
        var expect: [256]f32 = @splat(-std.math.inf(f32));
        for (0..32) |j| expect[cand_ids[r][j]] = @bitCast(@as(u32, vals[r][j]) << 16);
        try std.testing.expectEqualSlices(f32, &expect, got[r * 256 ..][0..256]);
    }
    var am = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, sparse.logits, -1, false, s));
    var am32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am32);
    try mlx.check(mlx.mlx_astype(&am32, am, .int32, s));
    try mlx.check(mlx.mlx_array_eval(am32));
    const ids = mlx.mlx_array_data_int32(am32) orelse return error.Unreadable;
    try std.testing.expectEqual(@as(i32, 45), ids[0]);
    try std.testing.expectEqual(@as(i32, 3 + 7 * 31), ids[1]);
}

test "a MiMo argmax-only readout holds the full head's bit-exact logits at its top-32, one row or four" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    // K 512 and a vocab that is a multiple of 8 keep both reads on the served head's qmv kernel.
    var fx = try mtp_mod.RerankFixture.init(s, mtp_mod.TOP32_MIN_ROWS + 96, 512, 8, 64, 0x1A4D);
    defer fx.deinit();
    fx.xfm.config.model_type = "mimo_v2";
    fx.xfm.suppress_mask = null;
    fx.xfm.bits_cache = .{};
    fx.xfm.lm_head_coarse = mtp_mod.buildRerankCoarse(s, &fx.xfm, 2) orelse return error.CoarseBuildDeclined;
    fx.xfm.lm_head_coarse_logged = true;
    defer if (fx.xfm.lm_head_coarse) |*c| c.deinit();
    const vocab: usize = @intCast(fx.vocab);
    var prng = std.Random.DefaultPrng.init(0x7E57);
    for ([_]c_int{ 1, 4 }) |rows| {
        const n: usize = @intCast(rows);
        var buf: [4 * 512]f32 = undefined;
        for (buf[0 .. n * 512]) |*v| v.* = @as(f32, @floatFromInt(prng.random().intRangeAtMost(i32, -64, 63))) * 0.015625;
        const xf = mlx.mlx_array_new_data(&buf, &[_]c_int{ 1, rows, 512 }, 3, .float32);
        defer _ = mlx.mlx_array_free(xf);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, xf, .bfloat16, s));
        const sparse = try fx.xfm.lmHeadProject(x, true);
        defer _ = mlx.mlx_array_free(sparse);
        const full = try fx.xfm.lmHeadProject(x, false);
        defer _ = mlx.mlx_array_free(full);
        var sf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sf);
        try mlx.check(mlx.mlx_astype(&sf, sparse, .float32, s));
        var ff = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ff);
        try mlx.check(mlx.mlx_astype(&ff, full, .float32, s));
        try mlx.check(mlx.mlx_array_eval(sf));
        try mlx.check(mlx.mlx_array_eval(ff));
        const sp = mlx.mlx_array_data_float32(sf) orelse return error.Unreadable;
        const fp = mlx.mlx_array_data_float32(ff) orelse return error.Unreadable;
        for (0..n) |r| {
            var kept: usize = 0;
            for (sp[r * vocab ..][0..vocab], fp[r * vocab ..][0..vocab]) |a, b| {
                if (a == -std.math.inf(f32)) continue;
                kept += 1;
                try std.testing.expectEqual(b, a);
            }
            try std.testing.expectEqual(@as(usize, 32), kept);
        }
    }
}

test "the GPU expert pick matches the host substitution on one row" {
    try gpuPickMatchesHost(512, 10, 0x5eed, false);
}

test "the GPU expert pick matches the host on a sigmoid router's log-sigmoid logits, MiMo's E and K" {
    try gpuPickMatchesHost(256, 8, 0x51a7, true);
}

fn gpuPickMatchesHost(comptime E: usize, comptime K: usize, seed: u64, log_sigmoid: bool) !void {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    for (0..200) |case| {
        var logits: [E]f32 = undefined;
        for (&logits) |*v| v.* = rand.floatNorm(f32) * 2;
        if (case % 7 == 0) logits[rand.uintLessThan(usize, E)] = logits[3];
        if (log_sigmoid) {
            // Every value <= 0, some where sigmoid underflows: the pick reads what the served path feeds it.
            for (&logits) |*v| if (rand.float(f32) < 0.05) {
                v.* = -120;
            };
            const raw = mlx.mlx_array_new_data(&logits, &[_]c_int{ 1, 1, E }, 3, .float32);
            defer _ = mlx.mlx_array_free(raw);
            var swap = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(swap);
            try routerSwapLogits(&swap, raw, .log_sigmoid, s);
            try mlx.check(mlx.mlx_array_eval(swap));
            @memcpy(&logits, mlx.mlx_array_data_float32(swap).?[0..E]);
        }
        var map: [E]i32 = undefined;
        var cached: [E]bool = undefined;
        for (&map, &cached, 0..) |*m, *c, e| {
            c.* = rand.float(f32) < 0.5;
            m.* = if (c.*) @intCast(e % 251) else -1;
        }
        var starved: [E]u8 = undefined;
        for (&starved) |*v| v.* = if (rand.float(f32) < 0.1) expert_stream_mod.PICK_STARVE_LIMIT else 0;
        var ids: [K]u16 = undefined;
        var picked_n: usize = 0;
        while (picked_n < K) {
            const e: u16 = @intCast(rand.uintLessThan(usize, E));
            if (std.mem.indexOfScalar(u16, ids[0..picked_n], e) != null) continue;
            ids[picked_n] = e;
            picked_n += 1;
        }
        const tolerance: f32 = if (case % 2 == 0) 0.2 else 0.6;
        var host_ids = ids;
        var taken: [E]bool = undefined;
        var loading: [E]bool = undefined;
        var host_starved = starved;
        _ = expert_stream_mod.substituteMisses(&host_ids, &logits, &cached, tolerance, K, &taken, &loading, &host_starved);

        var ids_i32: [K]i32 = undefined;
        for (&ids_i32, ids) |*d, v| d.* = v;
        const a_ids = mlx.mlx_array_new_data(&ids_i32, &[_]c_int{ 1, 1, K }, 3, .int32);
        defer _ = mlx.mlx_array_free(a_ids);
        const a_logits = mlx.mlx_array_new_data(&logits, &[_]c_int{ 1, 1, E }, 3, .float32);
        defer _ = mlx.mlx_array_free(a_logits);
        const a_map = mlx.mlx_array_new_data(&map, &[_]c_int{E}, 1, .int32);
        defer _ = mlx.mlx_array_free(a_map);
        const a_starved = mlx.mlx_array_new_data(&starved, &[_]c_int{E}, 1, .uint8);
        defer _ = mlx.mlx_array_free(a_starved);
        const pick = try expertPickGpu(a_ids, a_logits, a_map, a_starved, tolerance, s);
        defer _ = mlx.mlx_array_free(pick.slots);
        defer _ = mlx.mlx_array_free(pick.picked);
        try mlx.check(mlx.mlx_array_eval(pick.picked));
        try mlx.check(mlx.mlx_array_eval(pick.slots));
        const got = mlx.mlx_array_data_int32(pick.picked).?;
        const slots = mlx.mlx_array_data_int32(pick.slots).?;
        for (host_ids, 0..) |want, i| {
            try testing.expectEqual(@as(i32, want), got[i]);
            try testing.expectEqual(@max(map[want], 0), slots[i]);
        }
    }
}

test "the GPU expert pick declines a shape its threadgroup cannot hold" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var ids = [_]i32{ 0, 1 };
    var logits: [2048]f32 = @splat(0);
    var map: [2048]i32 = @splat(-1);
    var starved: [2048]u8 = @splat(0);
    const a_ids = mlx.mlx_array_new_data(&ids, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(a_ids);
    const a_logits = mlx.mlx_array_new_data(&logits, &[_]c_int{2048}, 1, .float32);
    defer _ = mlx.mlx_array_free(a_logits);
    const a_map = mlx.mlx_array_new_data(&map, &[_]c_int{2048}, 1, .int32);
    defer _ = mlx.mlx_array_free(a_map);
    const a_starved = mlx.mlx_array_new_data(&starved, &[_]c_int{2048}, 1, .uint8);
    defer _ = mlx.mlx_array_free(a_starved);
    try testing.expectError(error.UnsupportedShape, expertPickGpu(a_ids, a_logits, a_map, a_starved, 0.2, s));
}
