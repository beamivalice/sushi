const std = @import("std");
const glm5_mod = @import("glm5/glm5_forward.zig");
const dsv41_model = @import("dsv41/dsv41_model.zig");
const dsv41_stream = @import("dsv41/dsv41_stream.zig");
const dsv41_borrow = @import("dsv41/dsv41_borrow.zig");
const GlmRequest = glm5_mod.Request;
const qwen4_mod = @import("qwen4/qwen4_exp.zig");
const expert_stream_mod = @import("expert_stream.zig");
const expert_usage = @import("dsv41/expert_usage.zig");
const imatrix_capture = @import("imatrix.zig");
const sushi_exl3 = @import("sushi_exl3");
const expert_exl3 = sushi_exl3.format;
const fp8_block = @import("fp8_block.zig");
const trunk_stq = @import("trunk_stq.zig");
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
const nax_gate = @import("nax_gate.zig");
const attn_kernels = @import("attn_kernels.zig");
const verify_qmm = @import("verify_qmm.zig");
const verify_moe = @import("verify_moe.zig");
const moe_router = @import("moe_router.zig");
const moe_gqmv = @import("moe_gqmv.zig");
const moe_stream_compute = @import("moe_stream_compute.zig");
const layer_weights = @import("layer_weights.zig");
const forward_diag = @import("forward_diag.zig");
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
const attn256Reference = test_support.attn256Reference;
const mlxSettledActiveBytes = test_support.mlxSettledActiveBytes;
const qsaArraysAllEqual = test_support.qsaArraysAllEqual;
const testRandUniformBf16 = test_support.testRandUniformBf16;
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
pub const ATTN_QSA256_KERNEL_HEADER = attn_kernels.ATTN_QSA256_KERNEL_HEADER;
pub const AttnPdArm = attn_kernels.AttnPdArm;
pub const FUSED256_MIN_Q_LEN = attn_kernels.FUSED256_MIN_Q_LEN;
pub const KV_UNPACK_HEADER = attn_kernels.KV_UNPACK_HEADER;
const ImatrixTap = moe_stream_compute.ImatrixTap;
const BITS_CACHE_CAP = layer_weights.BITS_CACHE_CAP;
const BitsCache = layer_weights.BitsCache;
pub const DenseMlpWeights = layer_weights.DenseMlpWeights;
pub const FullAttnWeights = layer_weights.FullAttnWeights;
pub const HcWeights = layer_weights.HcWeights;
pub const LinearAttnWeights = layer_weights.LinearAttnWeights;
const DtypeCastCache = forward_diag.DtypeCastCache;
pub const MIMO_VERIFY_ROWS_MAX = mimo_forward.MIMO_VERIFY_ROWS_MAX;
pub const MoeDecodeDispatchArm = moe_gqmv.MoeDecodeDispatchArm;
pub const MoeLayerWeights = layer_weights.MoeLayerWeights;
pub const MoeMlpWeights = layer_weights.MoeMlpWeights;
pub const MimoExl3MoeHarness = test_support.MimoExl3MoeHarness;
pub const OneShotBits = qwen4_qsa_attn.OneShotBits;
pub const PersistentSsmGroup = ssm_cache.PersistentSsmGroup;
pub const PACKED_KV_SLICE_MAX = attn_kernels.PACKED_KV_SLICE_MAX;
pub const QSA_ATTN_BK = attn_kernels.QSA_ATTN_BK;
pub const PREFILL_DQ_GEMM_MIN_M = verify_moe.PREFILL_DQ_GEMM_MIN_M;
pub const PleWeights = layer_weights.PleWeights;
pub const ProfClock = forward_diag.ProfClock;
pub const QSA_ATTN_PACKED_MAX_S = qwen4_qsa_attn.QSA_ATTN_PACKED_MAX_S;
pub const QSA_HEAD_MARKS_MAX = qwen4_qsa.QSA_HEAD_MARKS_MAX;
pub const QSA_MAX_COMPRESS_RATIO = ssm_cache.QSA_MAX_COMPRESS_RATIO;
pub const QSA_PACKED_GATHER_MAX_REUSE = qwen4_qsa_attn.QSA_PACKED_GATHER_MAX_REUSE;
pub const QSA_RING_ROWS = ssm_cache.QSA_RING_ROWS;
pub const QSA_SCORE_KERNEL_HEADER = attn_kernels.QSA_SCORE_KERNEL_HEADER;
pub const QsaArm = qwen4_qsa_attn.QsaArm;
const QsaArmCounts = qwen4_qsa_attn.QsaArmCounts;
pub const QsaAttnCfgKey = attn_kernels.QsaAttnCfgKey;
const QsaBlockConsts = qwen4_qsa_select.QsaBlockConsts;
pub const QsaBlockFixture = qwen4_qsa_gather.QsaBlockFixture;
pub const QsaCfgCache = attn_kernels.QsaCfgCache;
pub const QsaHeadMark = qwen4_qsa.QsaHeadMark;
pub const QsaHeadMarkSet = qwen4_qsa.QsaHeadMarkSet;
const QsaPooledRope = qwen4_qsa_select.QsaPooledRope;
pub const QsaSelectCfgKey = qwen4_qsa_select.QsaSelectCfgKey;
pub const QsaSelectSplitCfgKey = qwen4_qsa_select.QsaSelectSplitCfgKey;
pub const QuantParams = layer_weights.QuantParams;
pub const Qwen4Mtp = qwen4_forward.Qwen4Mtp;
const PlePending = qwen4_hc.PlePending;
pub const SLIDING_TRIM_UNBOUNDED = kv_cache.SLIDING_TRIM_UNBOUNDED;
pub const SSMCacheEntry = ssm_cache.SSMCacheEntry;
pub const SSMCacheEntrySnapshot = ssm_cache.SSMCacheEntrySnapshot;
pub const SSMCheckpoint = ssm_cache.SSMCheckpoint;
pub const SSM_RESIDENT_MAX = ssm_cache.SSM_RESIDENT_MAX;
pub const SsmMergeKind = ssm_cache.SsmMergeKind;
pub const ThinPolicy = ssm_cache.ThinPolicy;
pub const VqmmLane = verify_qmm.VqmmLane;
pub const VerifyProjectionKind = verify_moe.VerifyProjectionKind;
const RouterGate = moe_router.RouterGate;
pub const ShapeKey = moe_router.ShapeKey;
const SwitchMlpBank = layer_weights.SwitchMlpBank;
const addOne = layer_weights.addOne;
pub const affineParamsFromGeometry = layer_weights.affineParamsFromGeometry;
pub const appendFullAttnWeights = layer_weights.appendFullAttnWeights;
pub const appendHybridMlpWeights = layer_weights.appendHybridMlpWeights;
pub const appendLinearAttnWeights = layer_weights.appendLinearAttnWeights;
pub const appendStructArrays = layer_weights.appendStructArrays;
pub const applyQsaHistoryAt = qwen4_qsa_state.applyQsaHistoryAt;
pub const arrayBytes = ssm_cache.arrayBytes;
pub const attachQsaHistoryToLatest = qwen4_qsa_state.attachQsaHistoryToLatest;
pub const capBufAppend = ssm_cache.capBufAppend;
pub const capturePrefillHidden = ssm_cache.capturePrefillHidden;
pub const capturePrefillHiddenLast = ssm_cache.capturePrefillHiddenLast;
pub const captureSsmCheckpoint = ssm_cache.captureSsmCheckpoint;
pub const attn256Base16Aligned = attn_kernels.attn256Base16Aligned;
const cachedScalarInt = verify_qmm.cachedScalarInt;
pub const bf16Scalar = layer_weights.bf16Scalar;
const checkQsaCompressRatio = qwen4_qsa_state.checkQsaCompressRatio;
pub const checkpointHasQsaHistory = qwen4_qsa_state.checkpointHasQsaHistory;
pub const checkpointHasQsaPooled = qwen4_qsa_state.checkpointHasQsaPooled;
pub const checkpointListHasQsaPooled = qwen4_qsa_state.checkpointListHasQsaPooled;
pub const checkpointQsaCoversPos = qwen4_qsa_state.checkpointQsaCoversPos;
pub const compactConvStateEnabled = ssm_cache.compactConvStateEnabled;
pub const computeQuantParams = layer_weights.computeQuantParams;
const computeYarnFreqs = qwen4_forward.computeYarnFreqs;
pub const decodeRowsProjection = verify_moe.decodeRowsProjection;
pub const constTableAs = forward_diag.constTableAs;
pub const decodeProfileEnabled = forward_diag.decodeProfileEnabled;
pub const decodeProfileRows = forward_diag.decodeProfileRows;
pub const decodeProfileSession = forward_diag.decodeProfileSession;
pub const diagEnvValueOn = forward_diag.diagEnvValueOn;
pub const entriesHaveQsaHistory = qwen4_qsa_state.entriesHaveQsaHistory;
const evalQwen4MtpResident = qwen4_forward.evalQwen4MtpResident;
const expertPickGpu = moe_stream_compute.expertPickGpu;
const expertSpecApplies = moe_stream_compute.expertSpecApplies;
const expertSwapKind = moe_stream_compute.expertSwapKind;
const expert_exl3_kernels = layer_weights.expert_exl3_kernels;
const exl3CodebookFollowsModel = test_support.exl3CodebookFollowsModel;
const fillPleSpecTokens = qwen4_hc.fillPleSpecTokens;
const freeKVEntry = kv_cache.freeKVEntry;
pub const fused256CausalMode = attn_kernels.fused256CausalMode;
pub const fused256Enabled = attn_kernels.fused256Enabled;
const floatDtypeTable = layer_weights.floatDtypeTable;
pub const fusedSdpa256Masked = qwen4_qsa_gather.fusedSdpa256Masked;
pub const fusedSdpaPrefill = attn_kernels.fusedSdpaPrefill;
pub const fusedSdpaPrefillImpl = attn_kernels.fusedSdpaPrefillImpl;
pub const fusedSdpaPrefillKv = attn_kernels.fusedSdpaPrefillKv;
const gatherQmv = moe_gqmv.gatherQmv;
const gatherQmvDownReduce = moe_gqmv.gatherQmvDownReduce;
const gatherQmvDownReduceRows = moe_gqmv.gatherQmvDownReduceRows;
const gatherQmvDownReduceRowsEligible = moe_gqmv.gatherQmvDownReduceRowsEligible;
pub const gatherQmvGateUp = moe_gqmv.gatherQmvGateUp;
const gatherQmvGateUpEnabled = moe_gqmv.gatherQmvGateUpEnabled;
const gatherQmvGateUpRows = moe_gqmv.gatherQmvGateUpRows;
pub const gatherExpertMm = moe_stream_compute.gatherExpertMm;
pub const gatherQsa256 = qwen4_qsa_gather.gatherQsa256;
pub const gatherQsa256Packed = qwen4_qsa_gather.gatherQsa256Packed;
const gpuArchitecture = nax_gate.gpuArchitecture;
const handOffStreamedResult = moe_stream_compute.handOffStreamedResult;
const getLayerWeight = layer_weights.getLayerWeight;
const layerProj = layer_weights.layerProj;
const stackProjs = layer_weights.stackProjs;
const getLayerWeightOpt = layer_weights.getLayerWeightOpt;
const getNamedWeight = layer_weights.getNamedWeight;
const getWeightFmt = layer_weights.getWeightFmt;
const getWeightFmtOpt = layer_weights.getWeightFmtOpt;
pub const handoffQsaHistoryToLatest = qwen4_qsa_state.handoffQsaHistoryToLatest;
pub const initModuleShell = layer_weights.initModuleShell;
pub const keepOnlyLatestQsaHistory = qwen4_qsa_state.keepOnlyLatestQsaHistory;
pub const kvEntriesBytes = ssm_cache.kvEntriesBytes;
pub const materializedOwnedCopy = ssm_cache.materializedOwnedCopy;
pub const macosProductVersion = nax_gate.macosProductVersion;
pub const lastDim = layer_weights.lastDim;
pub const loadHcWeights = layer_weights.loadHcWeights;
const loadPleWeights = layer_weights.loadPleWeights;
pub const loadSwitchMlpBank = layer_weights.loadSwitchMlpBank;
const maybeTransposeForBf16 = layer_weights.maybeTransposeForBf16;
pub const layerCap = forward_diag.layerCap;
const mimoExl3DistinctRouting = test_support.mimoExl3DistinctRouting;
const mimoExl3ForwardMatchesHost = test_support.mimoExl3ForwardMatchesHost;
pub const mimoGlobalDecodeRebuildMaxKeys = mimo_forward.mimoGlobalDecodeRebuildMaxKeys;
pub const nextSsmMemberGen = qwen4_ssm_group.nextSsmMemberGen;
pub const newEmptyKVEntry = kv_cache.newEmptyKVEntry;
pub const positionDropIndex = ssm_cache.positionDropIndex;
pub const positionDropIndexUsize = ssm_cache.positionDropIndexUsize;
const naxMinMForBits = nax_gate.naxMinMForBits;
pub const naxSdpaPreferred = nax_gate.naxSdpaPreferred;
pub const naxStatus = nax_gate.naxStatus;
pub const prefillHeadDimFused = attn_kernels.prefillHeadDimFused;
const mixedNaxShapeEnabled = verify_qmm.mixedNaxShapeEnabled;
const mixedPlainEnabled = verify_qmm.mixedPlainEnabled;
pub const mtpCoarsePairs = verify_moe.mtpCoarsePairs;
pub const prefillDqGemmEnabled = verify_moe.prefillDqGemmEnabled;
pub const qmatmulBits = verify_moe.qmatmulBits;
pub const moeRouterFusedEnabled = moe_router.moeRouterFusedEnabled;
pub const moeRouterTopK = moe_router.moeRouterTopK;
const moeRouterTopKGate = moe_router.moeRouterTopKGate;
const moeRoutingChain = moe_router.moeRoutingChain;
pub const moeDecodeDispatchArm = moe_gqmv.moeDecodeDispatchArm;
const moeRowsFusedEnabled = moe_gqmv.moeRowsFusedEnabled;
pub const msvQmvRows = moe_gqmv.msvQmvRows;
const noteExpertSpec = moe_stream_compute.noteExpertSpec;
pub const moeDumpBeginForward = forward_diag.moeDumpBeginForward;
const moeDumpDir = forward_diag.moeDumpDir;
pub const moeDumpForwardDone = forward_diag.moeDumpForwardDone;
pub const moeDumpTensor = forward_diag.moeDumpTensor;
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
pub const qsaNaxOsOk = nax_gate.qsaNaxOsOk;
pub const qsaNaxOsSupported = nax_gate.qsaNaxOsSupported;
pub const qsaPoolNormRopeFused = qwen4_qsa_select.qsaPoolNormRopeFused;
pub const qsaPrefillGatherMinKvFrom = qwen4_qsa_gather.qsaPrefillGatherMinKvFrom;
pub const qsaPrefillTransientBytes = qwen4_qsa_attn.qsaPrefillTransientBytes;
pub const qsaProbeAllFinite = nax_gate.qsaProbeAllFinite;
pub const qsaProbeCorruptFirst = nax_gate.qsaProbeCorruptFirst;
pub const qsaProbeLcgBf16 = nax_gate.qsaProbeLcgBf16;
pub const qsaProbeMaxAbsDiff = nax_gate.qsaProbeMaxAbsDiff;
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
pub const sliceAttentionSeq = attn_kernels.sliceAttentionSeq;
const routerSwapLogits = moe_stream_compute.routerSwapLogits;
const slabMoeWeights = moe_stream_compute.slabMoeWeights;
pub const scalarOf = layer_weights.scalarOf;
const shouldStreamExpertBank = layer_weights.shouldStreamExpertBank;
pub const sliceQsaHistoryOntoCheckpoint = qwen4_qsa_state.sliceQsaHistoryOntoCheckpoint;
pub const slidingTailSpan = kv_cache.slidingTailSpan;
pub const slidingViewFor = kv_cache.slidingViewFor;
pub const slidingPrefillAttn = attn_kernels.slidingPrefillAttn;
pub const slidingPrefillFused = attn_kernels.slidingPrefillFused;
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
pub const splitKAttnDispatch = attn_kernels.splitKAttnDispatch;
const splitFusedQkvAsymRows = layer_weights.splitFusedQkvAsymRows;
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
pub const verifyQmmNaxAvailable = nax_gate.verifyQmmNaxAvailable;
const verifyQmmNaxEnabled = nax_gate.verifyQmmNaxEnabled;
const verifyQmm = verify_qmm.verifyQmm;
pub const verifyQmmEnabled = verify_qmm.verifyQmmEnabled;
pub const vqmmLaneFor = verify_qmm.vqmmLaneFor;
const verifyExpertReduce = verify_moe.verifyExpertReduce;
const verifyExpertReuse = verify_moe.verifyExpertReuse;
const verifyIndexedExpertInput = verify_moe.verifyIndexedExpertInput;
pub const verifyJoinedProjection = verify_moe.verifyJoinedProjection;
const verifyRoutePack = verify_moe.verifyRoutePack;
pub const verifySharedHardware = verify_moe.verifySharedHardware;
const verifySharedProjection = verify_moe.verifySharedProjection;
const streamRows = moe_stream_compute.streamRows;
const streamSharedEarlyApplies = moe_stream_compute.streamSharedEarlyApplies;
const streamedExpertCompute = moe_stream_compute.streamedExpertCompute;
const swapReportDue = moe_stream_compute.swapReportDue;
pub const transposeBf16Weight = layer_weights.transposeBf16Weight;
pub const standinOnes = forward_diag.standinOnes;
pub const standinRef = forward_diag.standinRef;
pub const warmQsaEnvCaches = qwen4_qsa_attn.warmQsaEnvCaches;
pub const weightTriple = layer_weights.weightTriple;
const yarnMscale = qwen4_forward.yarnMscale;
pub const yarnSpec = qwen4_forward.yarnSpec;

pub const KVQuantConfig = kv_quant.KVQuantConfig;
pub const KvCacheChoice = kv_quant.KvCacheChoice;

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

// ── NAX m16 verify tile (M5-class matrix units; see the section comment
// above for provenance, gating, and the never-build-off-probe rule) ──

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
// (prefill_plan.boundedPrefillChunk / prefillEvalCadence / server.prefillMemoryNeeded)
// drop their score term via `prefillHeadDimFused` when this kernel is active.
//
// Scope: seq_len > 1 (prefill), any batch, GQA (Hq % Hk == 0), bf16, causal
// bottom-right alignment (query row r sits at absolute KV position
// kL - qL + r, matching MLX "causal" and createSlidingWindowMask semantics).
// K/V may be non-contiguous cache views (strides used; innermost dim must be
// contiguous — guaranteed for slices along T). The OOM guards and the
// dispatch share one predicate, so they cannot drift.

// The Metal JIT of a cooperative-tensor kernel happens at first eval, not at
// construction, so `qsaNaxArm` runs it once (tiny apply + eval, inference
// thread) and a failure latches the stock gather for the process.












pub const SingleVerifyFeature = enum { routing, down_reduce, hc_graph };
pub var single_verify_test_features: [3]bool = @splat(true);



pub var mtp_coarse_pair_dispatches: u64 = 0;

pub var mtp_verify_indexed_input_calls: u64 = 0;

pub var mtp_verify_expert_reduce_calls: u64 = 0;

pub var mtp_verify_expert_pairs_calls: u64 = 0;
pub var mtp_verify_expert_rows_calls: u64 = 0;



const SharedProjKind = enum { qmatmul, mtp_rows, verify_rows };



pub var mtp_verify_kernel_calls: [3]u64 = @splat(0);


// The lane -> (row, col) map (`fm`/`fn`) is the matmul2d cooperative-fragment layout,
// hand-derived from the SDK; every A, B and C fragment must use it. K stays 8 ascending
// 16-deep `op.run` calls: that is the stock tf32 GEMM's order and the bit-identity contract.








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
    /// DeepSeek-V4.1 chunked prefill: the fewest rows the chunk after this one can have, 0 when none follows in the
    /// prefill loop. A model whose decoder tail nothing reads may leave state only (`dsv41_model.extendState`).
    ahead_min_rows: usize = 0,
    /// A restore point is taken at this chunk's end.
    holds_point: bool = false,
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
    /// The caller's next act is to evaluate a value computed from the returned logits, so the forward may hand them
    /// back unevaluated (DeepSeek's one-row decode). Nothing else may read the logits' bytes first.
    lazy_logits: bool = false,
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

/// The encoder's view of the projections a forward runs: `record(ctx, w, x)` before each `x @ w` that goes
/// through `Transformer.qmatmul`, with `w` the loaded weight handle. Null outside the encoder.
pub var linear_tap: ?LinearTap = null;
pub const LinearTap = struct {
    ctx: *anyopaque,
    record: *const fn (ctx: *anyopaque, w: mlx.mlx_array, x: mlx.mlx_array) void,
};

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

    pub fn init(io: std.Io, allocator: std.mem.Allocator, config: ModelConfig, weights: *Weights) !Transformer {
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
        // The views of a trellis head; they join `moe_owned_bf16` once the layers have loaded.
        var head_views: std.ArrayList(mlx.mlx_array) = .empty;
        defer head_views.deinit(allocator);
        errdefer for (head_views.items) |a| {
            _ = mlx.mlx_array_free(a);
        };

        {
            // lm_head prefix: "language_model.model" -> "language_model", "model" -> try root, else -> prefix
            const lm_prefix = if (std.mem.eql(u8, prefix, "language_model.model")) "language_model" else prefix;
            var head_name: [96]u8 = undefined;
            const head_base = std.fmt.bufPrint(&head_name, "{s}.lm_head", .{lm_prefix}) catch unreachable;
            const maybe_lm_w = getWeightFmtOpt(weights, &name_buf, "{s}.lm_head.weight", lm_prefix);
            if (try trunk_stq.load(weights, head_base, config.trunk_quant, &head_views, allocator, s)) |head| {
                lm_head_w = head.w;
                lm_head_s = head.s;
                lm_head_b = head.b;
            } else if (maybe_lm_w) |w| {
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
        if (config.trunk_quant) |dec| {
            const counts = layer_weights.TrunkCounts.of(loaded.moe_layers);
            log.info("[trunk-stq] {d} trellis projections ({d} stacked), lm_head {s}; {s} window {d}\n", .{
                counts.banks, counts.stacks, if (trunk_stq.isStq(lm_head_w)) "trellis" else "affine", @tagName(dec.codebook), dec.window.bits(),
            });
        }
        if (head_views.items.len > 0) {
            moe_owned_bf16 = try std.mem.concat(allocator, mlx.mlx_array, &.{ head_views.items, loaded.owned_bf16 });
            allocator.free(loaded.owned_bf16);
            head_views.clearRetainingCapacity();
        }
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
        if (linear_tap) |tap| tap.record(tap.ctx, w, x);
        // u8 codes + f32 tile scales = an FP8 source linear (`mimo_source`).
        if (mlx.mlx_array_dtype(w) == .uint8) return fp8_block.linearServing(self.s, x, w, sc);
        if (mlx.mlx_array_dtype(w) == .uint16) return trunk_stq.matmul(self.s, x, w, sc, bi, self.config.trunk_quant orelse return error.TrunkQuantMissing);
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

    /// The readout's width, whichever way the pack stores the head.
    pub fn lmHeadRows(self: *const Transformer) c_int {
        return if (trunk_stq.isStq(self.lm_head_w)) @intCast(trunk_stq.outDim(self.lm_head_w)) else mlx.getShape(self.lm_head_w)[0];
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
        if (!moe_gqmv.decode_ladder_engaged) {
            moe_gqmv.decode_ladder_engaged = true;
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

    /// Whether this model is single-flight at admission (`slot_admission.modelExclusiveDecode`).
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
        forward_diag.moe_dump_warmup_depth += 1;
        defer forward_diag.moe_dump_warmup_depth -= 1;
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
        forward_diag.moe_dump_warmup_depth += 1;
        defer forward_diag.moe_dump_warmup_depth -= 1;
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



    // Mirror of `mtp_tick.mtpSubGroupSize`: the slots one row-axis tick can carry at this row
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
        if (virtual_rows == 1 and !trunk_stq.isStq(w)) {
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
        const stacked = try self.stackedPair(x, dw.gate_up_stack);
        const gate = if (stacked) |pair| pair[0] else try self.qmatmul(x, dw.gate_w, dw.gate_s, dw.gate_b);
        defer _ = mlx.mlx_array_free(gate);
        const up = if (stacked) |pair| pair[1] else try self.qmatmul(x, dw.up_w, dw.up_s, dw.up_b);
        defer _ = mlx.mlx_array_free(up);
        const activated = try self.computeGeglu(gate, up);
        defer _ = mlx.mlx_array_free(activated);
        return self.qmatmul(activated, dw.down_w, dw.down_s, dw.down_b);
    }

    /// The two linears of a stacked trellis bank on `x`, or null when there is no stack or `x` is wider than a stacked launch.
    pub fn stackedPair(self: *Transformer, x: mlx.mlx_array, stack: ?trunk_stq.Shared) !?[2]mlx.mlx_array {
        const sh = stack orelse return null;
        var outs: [2]mlx.mlx_array = undefined;
        if (!try trunk_stq.sharedProject(self.s, x, sh, self.config.trunk_quant orelse return error.TrunkQuantMissing, &outs)) return null;
        return outs;
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
        // The joined rows of `.verify_rows` decline a trellis bank before any projection runs.
        const stacked = if (proj == .verify_rows) null else try self.stackedPair(x, mw.shared_gate_up_stack);
        const gate = if (stacked) |pair| pair[0] else (try self.moeSharedProject(proj, x, mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b, widths)) orelse return null;
        defer _ = mlx.mlx_array_free(gate);
        const up = if (stacked) |pair| pair[1] else (try self.moeSharedProject(proj, x, mw.shared_up_w, mw.shared_up_s, mw.shared_up_b, widths)) orelse return null;
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
        const first_swap = swapped > 0 and moe_stream_compute.expert_swap_total_swapped == 0;
        moe_stream_compute.expert_swap_total_swapped +%= swapped;
        moe_stream_compute.expert_swap_total_ids +%= occurrences.len;
        if (!moe_stream_compute.expert_swap_logged) {
            moe_stream_compute.expert_swap_logged = true;
            log.info("[expert-swap] engaged: tolerance={d:.2} (lossy: a missed expert is replaced by a cached one within that relative probability)\n", .{tolerance});
        }
        if (swapReportDue(first_swap, moe_stream_compute.expert_swap_total_ids, moe_stream_compute.expert_swap_last_report)) {
            moe_stream_compute.expert_swap_last_report = moe_stream_compute.expert_swap_total_ids;
            log.info("[expert-swap] swapped {d} of {d} routed ids so far\n", .{ moe_stream_compute.expert_swap_total_swapped, moe_stream_compute.expert_swap_total_ids });
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
                if (!moe_gqmv.downred_engaged) {
                    moe_gqmv.downred_engaged = true;
                    log.info("[moe] fused down+reduce kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, down_qp.bits, down_qp.group_size });
                }
                if (!moe_gqmv.gqmv_engaged) {
                    moe_gqmv.gqmv_engaged = true;
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
        if (!moe_gqmv.gqmv_engaged) {
            moe_gqmv.gqmv_engaged = true;
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
            moeDumpTensor(self.s, "x", forward_diag.moe_dump_layer, expert_x);
            if (router_logits.ctx != null) moeDumpTensor(self.s, "rlogits", forward_diag.moe_dump_layer, router_logits);
            moeDumpTensor(self.s, "ids", forward_diag.moe_dump_layer, inds);
            moeDumpTensor(self.s, "w", forward_diag.moe_dump_layer, norm_scores);
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
                moeDumpTensor(self.s, "y", forward_diag.moe_dump_layer, y);
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
            forward_diag.decode_prof.moe_router_ns += mclk.lap();
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
            moe_gqmv.moe_rows_fused_layers +%= 1;
            if (grouped_rows) mtp_verify_expert_rows_calls +%= 1;
            cost_arm = 1;
        } else if (do_sort) {
            moe_gqmv.moe_sorted_layers +%= 1;
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
            forward_diag.decode_prof.moe_experts_ns += mclk.lap();
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
            moeDumpTensor(self.s, "y", forward_diag.moe_dump_layer, expert_sum);
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

pub fn initMoeLayers(allocator: std.mem.Allocator, config: ModelConfig, weights: *Weights, name_buf: *[256]u8, s: mlx.mlx_stream) !struct { moe_layers: []MoeLayerWeights, ssm_entries: []SSMCacheEntry, owned_bf16: []mlx.mlx_array } {
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
            var qkv = try layerProj(weights, name_buf, &config, li, "linear_attn.in_proj_qkv", .optional, &owned_bf16, allocator, s);
            var z = try layerProj(weights, name_buf, &config, li, "linear_attn.in_proj_z", .optional, &owned_bf16, allocator, s);
            const out = try layerProj(weights, name_buf, &config, li, "linear_attn.out_proj", .optional, &owned_bf16, allocator, s);
            const qkvz_stack = try stackProjs(weights, &config, li, &.{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z" }, &.{ &qkv, &z }, &owned_bf16, allocator, s);
            lw.attn = .{ .linear = .{
                .qkvz_stack = qkvz_stack,
                .qkv_w = qkv.w,
                .qkv_s = qkv.s,
                .qkv_b = qkv.b,
                .z_w = z.w,
                .z_s = z.s,
                .z_b = z.b,
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
                .out_w = out.w,
                .out_s = out.s,
                .out_b = out.b,
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
            var q = try layerProj(weights, name_buf, &config, li, "self_attn.q_proj", .by_config, &owned_bf16, allocator, s);
            var k = try layerProj(weights, name_buf, &config, li, "self_attn.k_proj", .by_config, &owned_bf16, allocator, s);
            var v = try layerProj(weights, name_buf, &config, li, "self_attn.v_proj", .by_config, &owned_bf16, allocator, s);
            const o = try layerProj(weights, name_buf, &config, li, "self_attn.o_proj", .by_config, &owned_bf16, allocator, s);
            const qkv_stack = try stackProjs(weights, &config, li, &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" }, &.{ &q, &k, &v }, &owned_bf16, allocator, s);
            lw.attn = .{
                .full = .{
                    .qkv_stack = qkv_stack,
                    .q_w = q.w,
                    .q_s = q.s,
                    .q_b = q.b,
                    .k_w = k.w,
                    .k_s = k.s,
                    .k_b = k.b,
                    .v_w = v.w,
                    .v_s = v.s,
                    .v_b = v.b,
                    .o_w = o.w,
                    .o_s = o.s,
                    .o_b = o.b,
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
                    const idx = try layerProj(weights, name_buf, &config, li, "self_attn.indexer.index_qk_proj", .optional, &owned_bf16, allocator, s);
                    fa.idx_qk_w = idx.w;
                    fa.idx_qk_s = idx.s;
                    fa.idx_qk_b = idx.b;
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
            var sh_gate = try layerProj(weights, name_buf, &config, li, "mlp.shared_expert.gate_proj", .absent_ok, &owned_bf16, allocator, s);
            var sh_up = try layerProj(weights, name_buf, &config, li, "mlp.shared_expert.up_proj", .absent_ok, &owned_bf16, allocator, s);
            const sh_down = try layerProj(weights, name_buf, &config, li, "mlp.shared_expert.down_proj", .absent_ok, &owned_bf16, allocator, s);
            const sh_stack = try stackProjs(weights, &config, li, &.{ "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj" }, &.{ &sh_gate, &sh_up }, &owned_bf16, allocator, s);
            lw.mlp = .{ .moe = .{
                .shared_gate_up_stack = sh_stack,
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
                .shared_gate_w = sh_gate.w,
                .shared_gate_s = sh_gate.s,
                .shared_gate_b = sh_gate.b,
                .shared_up_w = sh_up.w,
                .shared_up_s = sh_up.s,
                .shared_up_b = sh_up.b,
                .shared_down_w = sh_down.w,
                .shared_down_s = sh_down.s,
                .shared_down_b = sh_down.b,
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
            var gate = try layerProj(weights, name_buf, &config, li, "mlp.gate_proj", .optional, &owned_bf16, allocator, s);
            var up = try layerProj(weights, name_buf, &config, li, "mlp.up_proj", .optional, &owned_bf16, allocator, s);
            const down = try layerProj(weights, name_buf, &config, li, "mlp.down_proj", .optional, &owned_bf16, allocator, s);
            const gate_up_stack = try stackProjs(weights, &config, li, &.{ "mlp.gate_proj", "mlp.up_proj" }, &.{ &gate, &up }, &owned_bf16, allocator, s);
            lw.mlp = .{ .dense = .{
                .gate_up_stack = gate_up_stack,
                .gate_w = gate.w,
                .gate_s = gate.s,
                .gate_b = gate.b,
                .up_w = up.w,
                .up_s = up.s,
                .up_b = up.b,
                .down_w = down.w,
                .down_s = down.s,
                .down_b = down.b,
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

// ── Decode sub-block profiler (SUSHI_DECODE_PROFILE=1) ──
// Serializes the S=1 decode forward with per-sub-block evals to attribute the
// per-token GPU cost across embed / attn / mlp / lm_head. DIAGNOSTIC ONLY: the
// evals defeat the async pipeline (absolute tok/s drops while active), so the
// signal is the RELATIVE split plus the serialized total vs the pipelined
// per-token time — a serialized total near the pipelined per-token means we are
// compute-bound; far below it means the gap is dispatch/pipeline overhead.

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
    const n = forward_diag.decode_prof.calls;
    if (n == 0) return;
    const per = struct {
        fn ms(ns: u64, calls: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(calls)) / 1.0e6;
        }
    };
    const e = per.ms(forward_diag.decode_prof.embed_ns, n);
    const a = per.ms(forward_diag.decode_prof.attn_ns, n);
    const m = per.ms(forward_diag.decode_prof.mlp_ns, n);
    const l = per.ms(forward_diag.decode_prof.lmhead_ns, n);
    const mr = per.ms(forward_diag.decode_prof.moe_router_ns, n);
    const mx_ = per.ms(forward_diag.decode_prof.moe_experts_ns, n);
    const ms_ = per.ms(forward_diag.decode_prof.moe_shared_ns, n);
    log.info("[decode-prof] n={d} serial/tok={d:.2}ms  embed={d:.2} attn={d:.2} mlp={d:.2} lmhead={d:.2} ms | moe: router={d:.2} experts={d:.2} shared={d:.2} ms\n", .{ n, e + a + m + l, e, a, m, l, mr, mx_, ms_ });
    forward_diag.decode_prof = .{};
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

test "mimo_v2 EXL3 distinct routed experts match host with BF16 inputs" {
    for ([_]usize{ 1, 4, 14, 17, 33 }) |rows| try mimoExl3DistinctRouting(rows);
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

pub fn moeRowsAffineBank(s: mlx.mlx_stream, rnd: std.Random, E: c_int, N: c_int, K: c_int, bits: u32, gs: u32) !struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array } {
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

pub fn moeRowsSliceRow(s: mlx.mlx_stream, x: mlx.mlx_array, i: c_int) !mlx.mlx_array {
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
pub const VerifyQmmParity = struct {
    /// Absolute ceiling on the worst element's clamped relative error.
    pub const GROSS_CEILING: f64 = 0.5;
    /// How much of stock's RMS error the lane may carry.
    pub const RMS_FACTOR: f64 = 3.0;
    /// Cosine may not fall below stock's by more than this.
    pub const COS_SLACK: f64 = 1e-5;
    /// …and both must actually be tracking the truth.
    pub const COS_FLOOR: f64 = 0.999;
};

/// Why a verify-lane parity check failed, or null when it passed. Pure so the
/// decision itself is unit-testable without a GPU — each arm is a distinct
/// failure shape the bars are meant to separate.
const VerifyQmmParityFail = enum { gross_element, systematic, cosine, not_tracking };

pub fn verifyQmmParityVerdict(
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

/// fp32 ground truth for an attention shape: the same composed chain MLX runs
/// for an unfused head dim, with every operand widened first. Both the bf16
/// composed path and the kernel are measured against THIS, never each other.
pub fn attnPdTruthF32(
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

/// A `DenseKVView` over a freshly quantized K/V pair, as `KVCache.denseView`
/// hands one out: lazy dense arrays plus the packed triples they stand for.
pub const TestQuantView = struct {
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

pub const SWA_TEST_WINDOW: u32 = 128;

/// mimo_v2's sliding-layer attention geometry, enough for the ring and the arm.
pub fn swaSinkTestConfig() ModelConfig {
    var c = ModelConfig{};
    c.model_type = "mimo_v2";
    c.has_sliding_window = true;
    c.sliding_window = SWA_TEST_WINDOW;
    c.head_dim = 192;
    c.v_head_dim = 128;
    return c;
}

pub fn swaHostF32(alloc: std.mem.Allocator, a: mlx.mlx_array, s: mlx.mlx_stream) ![]f32 {
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

pub const SwaGeom = struct { hq: usize, hk: usize, qk: usize, vd: usize, t: usize };

/// fp64 truth for one chunk of a sliding layer with sinks, read off the FULL
/// history at absolute positions: a ring or band offset bug cannot hide in it.
/// `q` is the chunk `[hq, ql, qk]`; `k`/`v` the whole history `[hk, t, *]`.
pub fn swaSinkTruth(
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
pub fn swaBandMask(ql: c_int, kl: c_int) !mlx.mlx_array {
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

pub fn absMax(xs: []const f32) f64 {
    var m: f64 = 0;
    for (xs) |x| m = @max(m, @abs(@as(f64, x)));
    return m;
}

pub fn swaMaxErr(got: []const f32, truth: []const f64) f64 {
    var e: f64 = 0;
    for (got, truth) |a, b| e = @max(e, @abs(@as(f64, a) - b));
    return e;
}

pub fn sliceSeq(s: mlx.mlx_stream, a: mlx.mlx_array, start: c_int, end: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(a);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, a, &.{ 0, 0, start, 0 }, 4, &.{ sh[0], sh[1], end, sh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    return out;
}

/// Sum-of-squares slack (RMS within 0.5%): bf16 store rounding flips land either way between the arms.
const ATTN_PD_NAX_RMS_SLACK: f64 = 1.01;

/// The kernel arms this machine can run, SIMD first: every parity case covers each of them.
pub fn attnPdTestArms() []const AttnPdArm {
    const all = [_]AttnPdArm{ .simd, .nax };
    return if (verifyQmmNaxAvailable() and qsaNaxOsOk()) &all else all[0..1];
}

/// The NAX arm against fp64 truth, element by element, may exceed the SIMD kernel's error only by
/// the bf16 store's own rounding flip plus its f16 P's rounding (2^-11 of the largest |V|, the
/// weights summing to one), and its RMS error by rounding noise.
pub fn expectAttnPdNoWorse(nax: []const f32, simd: []const f32, truth: []const f64, v_absmax: f64) !void {
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

/// fp64 truth for a global layer's causal attention over `g.t` keys, bottom-right aligned.
/// `q` is `[hq, ql, qk]`, `k`/`v` are `[hk, t, *]`.
pub fn attnPdCausalTruth(alloc: std.mem.Allocator, g: SwaGeom, q: []const f32, k: []const f32, v: []const f32, ql: usize, scale: f64) ![]f64 {
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

// ── Verify-width QSA gather (S = 2 .. FUSED256_MIN_Q_LEN-1) ──

// ── Fused sparse attention at verify widths (sushi_attn_qsa256_q) ──

// ── KV cache growth policy (issue #110) ──────────────────────────────────────

// ── Constant-table dtype class guard ─────────────────────────────────────────
// A constant lookup table built at load in f32 and multiplied into a bf16
// activation PROMOTES the product to f32 (mlx's promotion rule). Once that
// product was post-RoPE q/k, so the f32 travelled through SDPA, o_proj and the
// residual — every projection and expert weight for the REST of the forward was
// upcast on read, at 3x the token cost, with nothing in the op count, the
// kernel choice or a pre-loop dtype log to show for it.

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

test "diagEnvOnCached answers once and latches" {
    var cache: ?bool = null;
    try testing.expectEqual(false, diagEnvOnCached(&cache, "SUSHI_NO_SUCH_DIAG_SWITCH_PROBE"));
    try testing.expectEqual(@as(?bool, false), cache);
    cache = true;
    try testing.expectEqual(true, diagEnvOnCached(&cache, "SUSHI_NO_SUCH_DIAG_SWITCH_PROBE"));
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

pub fn seBf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn seFillBf16(rnd: std.Random, buf: []u16) void {
    for (buf) |*v| {
        const f = rnd.float(f32) * 2.0 - 1.0;
        v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
    }
}

pub fn seHash(arr: mlx.mlx_array) !u64 {
    try mlx.check(mlx.mlx_array_eval(arr));
    const p = mlx.mlx_array_data_bfloat16(arr) orelse return error.Bf16Unreadable;
    var h: u64 = 0xcbf29ce484222325;
    for (p[0..mlx.mlx_array_size(arr)]) |v| {
        h ^= v;
        h *%= 0x100000001b3;
    }
    return h;
}

pub const SeErr = struct { max: f32, rms: f32 };



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
