//! Plan 01 Phase 2 — continuous-batching scheduler.
//!
//! Owns the single inference thread (the only thread that calls into mlx
//! ops; mlx 0.31.2 made GPU streams thread-local so the model weights are
//! bound to whichever thread first calls `useCurrentThreadStream` after
//! load). Connection threads parse HTTP, build prompt token ids, call
//! `submit()` and then loop on `Slot.waitNext()` to read generated tokens
//! one at a time. The state machines for tool-call detection, thinking
//! blocks, SSE streaming, etc. live on the connection thread, unchanged.
//!
//! Per-slot state (KVCache, moe_seq_offset, ssm_entries, vision_embeddings)
//! lives on `Slot` itself; each slot's `ForwardCtx` points at those fields,
//! and the `Generator` constructed for the slot stores the ctx so
//! `xfm.forwardWith(&self.ctx, ...)` routes through slot-local state. The
//! shared Transformer holds the weights only — single-writer-per-tick on
//! the inference thread guards correctness while many slots can be in
//! flight at once.
//!
//! Decode tick logic (the Phase 3 gate):
//!   * `active.len == 1` (the common single-stream case) → legacy path:
//!     `Generator.next` / `nextPld` / `nextDrafter` with the same lazy
//!     pipeline that today's serial path uses. Bit-identical to pre-Phase-2.
//!   * `active.len >= 2` → batched: `forwardBatchedDecode` produces N logits
//!     in one kernel pass, sampled per-slot. PLD / drafter are forced off in
//!     this path (the speculative paths assume a single in-flight slot's
//!     KV cache; expensive to interleave).
//!   * `cancelled` slots are skipped and culled from `decoding`.
//!
//! Output channel: each slot owns a bounded ring (`std.ArrayList(u32)` +
//! cursor + cv) the inference thread pushes into and the connection thread
//! drains. Generation end is signaled by `state == .finished` (or `.errored`)
//! plus a cv broadcast.

const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const qwen4_qsa_gather = @import("qwen4/qwen4_qsa_gather.zig");
const qwen4_qsa_attn = @import("qwen4/qwen4_qsa_attn.zig");
const tokenizer_mod = @import("tokenizer.zig");
const generate_mod = @import("generate.zig");
const rp_mod = @import("reasoning_protocol.zig");
const mtp_mod = @import("mtp.zig");
const depth_bounds = @import("mtp_depth_bounds.zig");
const dspark_rows = @import("dsv41/dspark_rows.zig");
const mimo_mtp = @import("mimo2/mimo_mtp.zig");
const chip_mod = @import("chip.zig");
const model_mod = @import("model.zig");
const vision_mod = @import("vision.zig");
const chat_mod = @import("chat.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const dsv41_prefix = @import("dsv41/dsv41_prefix.zig");
const glm5_prefix = @import("glm5/glm5_prefix.zig");
const restore_dump = @import("restore_dump.zig");
const metrics_mod = @import("metrics.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const disk_placement = @import("disk_placement.zig");
const tokenize_cache_mod = @import("tokenize_cache.zig");
const model_registry_mod = @import("model_registry.zig");
const model_settings = @import("model_settings.zig");
const mtp_acceptance_mod = @import("mtp_acceptance.zig");
const model_discovery = @import("model_discovery.zig");
const log = @import("log.zig");
const io_util = @import("io_util.zig");
const status = @import("status.zig");
const sleep_inhibit = @import("sleep_inhibit.zig");

const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;
const SSMCacheEntry = transformer_mod.SSMCacheEntry;
const ForwardCtx = transformer_mod.ForwardCtx;
const ModelConfig = model_mod.ModelConfig;
const Tokenizer = tokenizer_mod.Tokenizer;
const Generator = generate_mod.Generator;
const SamplingParams = sampler.SamplingParams;
const dflash_mod = @import("dflash.zig");
const glm5_forward_mod = @import("glm5/glm5_forward.zig");
const round_cost_mod = @import("round_cost.zig");
const group_cost_mod = @import("mtp_group_cost.zig");
const expert_stream_mod = @import("expert_stream.zig");
const expert_usage = @import("dsv41/expert_usage.zig");
const DflashModel = dflash_mod.DflashModel;
const VisionEncoder = vision_mod.VisionEncoder;
const Weights = model_mod.Weights;
const ChatConfig = chat_mod.ChatConfig;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;
const slot_outcome = @import("slot_outcome.zig");
pub const loopStopReason = slot_outcome.loopStopReason;
const loopStopDecision = slot_outcome.loopStopDecision;
const loopTrimEnabled = slot_outcome.loopTrimEnabled;
pub const recordSlotCleanup = slot_outcome.recordSlotCleanup;
pub const finishSlot = slot_outcome.finishSlot;
const mtp_tick = @import("mtp_tick.zig");
const plannerShape = mtp_tick.plannerShape;
const tryPlannerTick = mtp_tick.tryPlannerTick;
pub const anyPenalised = mtp_tick.anyPenalised;
const mtpCrowdThresholdFor = mtp_tick.mtpCrowdThresholdFor;
const runMtpGroups = mtp_tick.runMtpGroups;
pub const mtpGroupRowCap = mtp_tick.mtpGroupRowCap;
pub const mtpSubGroupSize = mtp_tick.mtpSubGroupSize;
const slot_admission = @import("slot_admission.zig");
pub const postEvictionPrefillChunk = slot_admission.postEvictionPrefillChunk;
pub const fewerCheckpointsToAdmit = slot_admission.fewerCheckpointsToAdmit;
pub const admissionPassArmed = slot_admission.admissionPassArmed;
pub const adaptiveChunkWidthFor = slot_admission.adaptiveChunkWidthFor;
pub const MAX_BATCH_GROUP = slot_admission.MAX_BATCH_GROUP;
pub const BatchVerdict = slot_admission.BatchVerdict;
pub const configBatchesDecode = slot_admission.configBatchesDecode;
pub const batchGroupCap = slot_admission.batchGroupCap;
const noteSerial = slot_admission.noteSerial;
const batchedKvKeepCount = slot_admission.batchedKvKeepCount;
const batchedPadWaste = slot_admission.batchedPadWaste;
const groupKeepCount = slot_admission.groupKeepCount;
const padWasteKeepCount = slot_admission.padWasteKeepCount;
const groupCtxMax = slot_admission.groupCtxMax;
const batchKvLenOf = slot_admission.batchKvLenOf;
const batchKvLenOfWith = slot_admission.batchKvLenOfWith;
const fillGroupPadWasteKvLens = slot_admission.fillGroupPadWasteKvLens;
pub const modelExclusiveDecode = slot_admission.modelExclusiveDecode;
const slotExclusiveDecode = slot_admission.slotExclusiveDecode;
const heldExclusive = slot_admission.heldExclusive;
const AdmitCand = slot_admission.AdmitCand;
const admitPendingTick = slot_admission.admitPendingTick;
pub const QueueWait = slot_admission.QueueWait;
pub const queueWait = slot_admission.queueWait;
pub const queueWaitText = slot_admission.queueWaitText;
const queueLogLine = slot_admission.queueLogLine;
const queueStartLine = slot_admission.queueStartLine;
pub const MemoryBill = slot_admission.MemoryBill;
pub const admitsWithinMemory = slot_admission.admitsWithinMemory;
const memoryAdmitCount = slot_admission.memoryAdmitCount;
const vision_queue = @import("vision_queue.zig");
pub const MediaBlock = vision_queue.MediaBlock;
pub const DeferredMedia = vision_queue.DeferredMedia;
pub const MediaRefusal = vision_queue.MediaRefusal;
const encodeMediaAfterRestore = vision_queue.encodeMediaAfterRestore;
const orphan_vision_cap = vision_queue.orphan_vision_cap;
const freeVisionArray = vision_queue.freeVisionArray;
pub const VisionImagePixels = vision_queue.VisionImagePixels;
const VisionEncodeRequest = vision_queue.VisionEncodeRequest;
pub const decode_workers = vision_queue.decode_workers;
pub const ImageDecodeWindow = vision_queue.ImageDecodeWindow;
pub const encodeVideoBlock = vision_queue.encodeVideoBlock;
const runVisionEncode = vision_queue.runVisionEncode;
const model_load = @import("model_load.zig");
pub const LoadParams = model_load.LoadParams;
const signalStarted = model_load.signalStarted;
const recordLoadError = model_load.recordLoadError;
pub const applyModelSettings = model_load.applyModelSettings;
pub const mimoVerifyWarmWanted = model_load.mimoVerifyWarmWanted;
pub const mtpChoiceFor = model_load.mtpChoiceFor;
const mtpStreamingVerdict = model_load.mtpStreamingVerdict;
const streamedHeadSupportedAt = model_load.streamedHeadSupportedAt;
pub const resolveSsdBudget = model_load.resolveSsdBudget;
const preloadCpuState = model_load.preloadCpuState;
pub const planExpertStreaming = model_load.planExpertStreaming;
pub const planExpertStreamingWith = model_load.planExpertStreamingWith;
pub const resolveStreamedVision = model_load.resolveStreamedVision;
pub const billedFillPeakBytes = model_load.billedFillPeakBytes;
pub const loadStagingBytes = model_load.loadStagingBytes;
pub const applyStreamingPlan = model_load.applyStreamingPlan;
pub const resolveExpertCache = model_load.resolveExpertCache;
pub const mimoCoarseHeadBytes = model_load.mimoCoarseHeadBytes;
pub const freeCpuState = model_load.freeCpuState;
pub const headlessStubCpuState = model_load.headlessStubCpuState;
pub const gateEstimateBytes = model_load.gateEstimateBytes;
pub const LoadServingBill = model_load.LoadServingBill;
pub const glmStreamedDrafterOff = model_load.glmStreamedDrafterOff;
const LoadDrafterDir = model_load.LoadDrafterDir;
const streamedAssistantBytes = model_load.streamedAssistantBytes;
const residentColdLoadBillBytes = model_load.residentColdLoadBillBytes;
const coldLoadDrafterDir = model_load.coldLoadDrafterDir;
pub const visionChoiceFor = model_load.visionChoiceFor;
pub const coldLoadVision = model_load.coldLoadVision;
const streamedLoadRequirementBytes = model_load.streamedLoadRequirementBytes;
const loadRequirementBytes = model_load.loadRequirementBytes;
const glmColdLoadBillBytes = model_load.glmColdLoadBillBytes;
const LoadBillFixtures = model_load.LoadBillFixtures;
pub const doLoadOnInferenceThread = model_load.doLoadOnInferenceThread;
const runLoadRequest = model_load.runLoadRequest;
const runUnloadRequest = model_load.runUnloadRequest;
const dsv41_thought_point = @import("dsv41/dsv41_thought_point.zig");
const dsv41StateBytes = dsv41_thought_point.dsv41StateBytes;
const armDsv41ThoughtPoint = dsv41_thought_point.armDsv41ThoughtPoint;
const writeDsv41ThoughtPoint = dsv41_thought_point.writeDsv41ThoughtPoint;
const glm5_rows_tick = @import("glm5/glm5_rows_tick.zig");
const commitGlmSlot = glm5_rows_tick.commitGlmSlot;
const glmRowsUbench = glm5_rows_tick.glmRowsUbench;
const glmRoundUbench = glm5_rows_tick.glmRoundUbench;
const glmNativeDflashArmed = glm5_rows_tick.glmNativeDflashArmed;
const glmRowsInCompany = glm5_rows_tick.glmRowsInCompany;
const slotGlmRowable = glm5_rows_tick.slotGlmRowable;
const slotMtpGroupable = glm5_rows_tick.slotMtpGroupable;
const glmRowsForward = glm5_rows_tick.glmRowsForward;

/// Submit-time parameters. `prompt_ids` and `eos_token_ids` are duped into the
/// slot so callers can free their copies immediately. `vision_embeddings`
/// ownership transfers into the slot when non-null (the slot will free on
/// deinit); a `submit` that fails hands it to the inference thread to free.
pub const SubmitParams = struct {
    prompt_ids: []const u32,
    /// Full original prompt for the MTP lookup. When null, defaults to
    /// `prompt_ids` (the lookup table = full_prompt + generated).
    full_prompt: ?[]const u32 = null,
    cached_tokens: u32 = 0,
    has_tools: bool = false,
    /// The server's final resolved thinking mode after request overrides and
    /// model defaults. DFlash economics key off this, not tool presence.
    enable_thinking: bool = false,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    max_tokens: u32,
    timeout_ns: u64 = 0,
    enable_drafter: bool = false,
    /// DFlash assistant for this model. Rides the SAME `enable_drafter`
    /// request switch (a model loads at most ONE of drafter/dflash);
    /// `drafter_block_size` carries the dflash-resolved block size too.
    dflash: ?*DflashModel = null,
    drafter_block_size: u32 = 4,
    enable_mtp: bool = false,
    allow_batch_mtp: bool = true,
    mtp: ?generate_mod.MtpHeadRef = null,
    /// 0 = auto (see generate_mod.resolveMtpDepthCap).
    mtp_depth: u32 = 0,
    /// Vision embeddings spliced at image-token positions during prefill.
    /// Ownership transferred to the slot; freed on slot.deinit.
    vision_embeddings: ?mlx.mlx_array = null,
    /// The request's media, encoded after the prefix restore. Ownership transfers to the slot on a
    /// successful submit; a failed one frees it.
    deferred_media: ?*DeferredMedia = null,
    /// Prefix-cache key for the media under the placeholder tokens (0 = none).
    vision_key: u64 = 0,
    /// Per media block of `full_prompt`, its start and chained key
    /// (`prefix_cache.MediaSpan`). Borrowed; the slot keeps a copy.
    media_chain: []const prefix_cache_mod.MediaSpan = &.{},
    /// Workload key for hot-cache eviction (`server.requestCacheKey`, 0 = anonymous).
    cache_key: u64 = 0,
    /// Qwen3-VL interleaved M-RoPE: server-computed flat [3 × mrope_total] i32
    /// position-id table + decode delta. Ownership of `mrope_pos` transfers to
    /// the slot; freed on slot.deinit. Null for non-image / non-Qwen requests.
    mrope_pos: ?[]const i32 = null,
    mrope_total: usize = 0,
    mrope_delta: i32 = 0,
    logprobs_n: u32 = 0,
    /// Wave 1.A: per-request override of the process-default KV-cache quant
    /// scheme. When non-null, this slot's KVCache is constructed with this
    /// config instead of `Scheduler.kv_quant_config`. Lets one server host a
    /// single model and let clients trade accuracy for context length on a
    /// per-call basis (`{"kv_quant": "off"|4|8}` body field).
    kv_quant_config: ?transformer_mod.KVQuantConfig = null,
    /// Plan 05 Phase D: the target model for this request. The conn thread
    /// resolves this via `scheduler.ensureLoaded(id)` BEFORE submitting and
    /// keeps a refcount on it for the slot's lifetime, so the model can't
    /// be evicted mid-flight. The scheduler routes prefill/decode through
    /// `slot.model.transformer.?` (and friends) instead of `sch.xfm`, so
    /// per-tick model switching is just a pointer hop. Required field
    /// post-Phase-D; tests using the legacy path pass the default model.
    model: *model_registry_mod.LoadedModel,
};

/// The KDA checkpoints a GLM request keeps for the prefix cache, as admission bills them: how many it
/// keeps, and whether its prefill moves each to SSD and drops the RAM copy (`glmCheckpointsRelease`).
pub const CheckpointPlan = struct {
    keep: u32 = 0,
    released: bool = false,
    /// The request will take a thought point mid-decode (`thoughtPointWanted`): one more state held to its commit.
    thought: bool = false,
};

/// Set by `server.installPrefillAdmission`: does a request of this shape fit in GPU memory
/// right now? Null (unit tests, no HTTP server) disables evict-to-admit. The trailing warm
/// arguments are the restored rows, their capacity, and whether the restore checked the
/// entry out (the only restore whose rows the request will not allocate); then MTP and the GLM
/// checkpoints the request keeps (`server.WarmPrefix.checkpoints`).
pub var prefill_admission_fits: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, u64, u64, bool, bool, CheckpointPlan) bool = null;

/// {needed, available, commit} of the same cold bill, live memory re-read (`server.prefillBillNumbersNow`).
/// Null (unit tests) skips the inference thread's hold for an ungated arch.
pub var prefill_admission_numbers: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, bool) MemoryBill = null;

/// The prefill width this request should run at, chosen against live post-eviction memory
/// (`server.requestPrefillChunkNow`). Null keeps the model's load-time pin.
pub var prefill_request_chunk: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, u64, u64, bool, bool) u32 = null;
/// The same chooser for the load-time prefill meter, installed before the load (the one above
/// arrives after it); only the meter reads it.
pub var prefill_ubench_chunk: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, u64, u64, bool, bool) u32 = null;

/// Asked by the prefill loop: the width of the next chunk, re-priced at every chunk boundary
/// (`server.adaptivePrefillWidthNow`). Null keeps the admitted width.
pub var prefill_chunk_adapt: ?*const fn (
    *const model_mod.ModelConfig,
    u64,
    usize,
    u32,
    u32,
    *mtp_plan.AdaptiveWidthState,
    u64,
) u32 = null;

/// Whether the per-chunk adaptive width is enabled for this model. The hook above is
/// installed process-wide, so its presence is not the arch gate.
pub var prefill_chunk_adaptive_enabled: ?*const fn (*const model_mod.ModelConfig) bool = null;

/// Re-price a widen after the interleave tick (`server.adaptivePrefillWidenStillFits`). Null declines every widen.
pub var prefill_chunk_widen_ok: ?*const fn (
    *const model_mod.ModelConfig,
    u64,
    usize,
    u32,
    u64,
) bool = null;

/// Logs the numbers the estimator compared on a refusal.
pub var prefill_admission_refused_log: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, u64, u64, bool, bool, CheckpointPlan) ?model_registry_mod.MemoryContextRefusal = null;

/// GPU bytes a vision encode may still take, read live (`request_media.visionEncodeAvailable`). Null skips the recheck.
pub var vision_encode_available: ?*const fn (*const model_mod.ModelConfig) u64 = null;

/// `request_media.decodeEncodedImage`: an `undecoded` image's file to the pixels its tower reads. Null cannot decode.
pub var vision_image_decode: ?*const fn (std.mem.Allocator, []const u8, *const model_mod.ModelConfig) ?chat_mod.ImageData = null;

/// `request_media.visionEncodeBill`: the GPU bytes an encode of these blocks peaks at.
pub var vision_encode_bill: ?*const fn (*const model_mod.ModelConfig, []const chat_mod.ImageData, []const chat_mod.VideoData, usize) u64 = null;

/// `server.towerFitFault`: the refusal text when `available` bytes do not hold the encode, written into `out`.
pub var vision_fit_fault: ?*const fn ([]u8, *const model_mod.ModelConfig, []const chat_mod.ImageData, []const chat_mod.VideoData, usize, u64) ?[]const u8 = null;

/// Invalidate the published hot-cache budget on unload/switch (`server.clearResolvedPrefixCacheMem`).
pub var hot_cache_budget_invalidate: ?*const fn () void = null;

pub const SlotState = enum { pending_prefill, decoding, finished, errored };

/// The error name latched when the real one cannot be copied; `Slot.deinit` must not free it.
const static_error_name: []const u8 = "OutOfMemory";

/// Result of `Slot.waitNext`. Driven by the inference thread; consumed by
/// the connection thread.
pub const NextResult = union(enum) {
    /// Next decoded token id.
    token: u32,
    /// Generation completed (EOS, max_tokens, timeout, etc.). `finish_reason`
    /// is set on the slot at this point.
    done: void,
    /// Generation errored. `error_code` is set on the slot; the caller
    /// should surface it to the client and call `complete(slot)`.
    err: void,
};

fn firstMediaPlaceholder(
    has_media: bool,
    tokens: []const u32,
    image_token_id: u32,
    video_token_id: u32,
) ?usize {
    // The placeholder ids are ordinary vocabulary entries, so a text-only
    // prompt can contain one; a boundary exists only where media rows do.
    if (!has_media) return null;
    for (tokens, 0..) |token, i| {
        if ((image_token_id > 0 and token == image_token_id) or
            (video_token_id > 0 and token == video_token_id)) return i;
    }
    return null;
}

/// Per-request state. Owned by the Scheduler from `submit` until `complete`.
pub const Slot = struct {
    allocator: std.mem.Allocator,
    /// io reference for Stopwatch / async-eval (captured from Scheduler).
    io: std.Io,

    /// Plan 05 Phase D: target model for this request. Captured from
    /// `SubmitParams.model` and borrowed for the slot's lifetime; the conn
    /// thread holds a refcount (via `scheduler.ensureLoaded`) so the
    /// pointer stays valid until `complete()`. Prefill/decode route forward
    /// passes through `model.transformer.?` (and `model.vision_encoder`,
    /// `model.drafter`, `model.prefix_cache`). Batched decode groups slots
    /// by this field so kernels never cross model boundaries.
    model: *model_registry_mod.LoadedModel,

    // ── Per-slot model state. Owned by the slot. ──
    cache: KVCache,
    moe_seq_offset: usize,
    ssm_entries: ?[]SSMCacheEntry,
    /// glm5_next: this request's native KDA/MLA state (`ctx.glm5_request`).
    glm5_request: ?glm5_forward_mod.Request = null,
    /// SSM stride checkpoints salvaged from a prefill the client cancelled:
    /// `Generator.initWithOptions` moves its captured checkpoints into this
    /// sink before returning `error.Cancelled` (they die with the failed
    /// construction otherwise). Consumed by `commitCancelledPrefillSlot`
    /// (ownership transfers into the hot-cache entry); freed by `deinit`
    /// when never consumed.
    cancelled_prefill: Generator.CancelledCheckpointSink = .{},
    /// A ringed cache's restore points (`KVCache.ringCheckpoint`): `.fork` right after a
    /// restore, `.prompt_end` right after prefill while the ring still holds it. The commit takes
    /// ownership, `deinit` frees them otherwise.
    ring_cps: prefix_cache_mod.SlotRingCps = .{},
    vision_embeddings: ?mlx.mlx_array,
    /// Kept to the slot's end: a retried cold prefill encodes it again from the start.
    deferred_media: ?*DeferredMedia = null,
    /// Set with `error.MediaRefused`, which the connection thread answers as a 4xx/5xx.
    media_refusal: ?MediaRefusal = null,
    vision_key: u64,
    cache_key: u64 = 0,
    /// Hot-cache entry this request restored from (`LookupResult.entry_id`).
    restored_entry: u64 = 0,
    skip_prefix_cache: bool = false,
    /// First dynamic image/audio/video placeholder in `full_prompt`. Cache
    /// state before this position is safe to share across media hashes.
    media_start: ?usize,
    /// Owned copy of `SubmitParams.media_chain`.
    media_chain: []const prefix_cache_mod.MediaSpan,
    /// Qwen3-VL M-RoPE position-id table (flat [3 × mrope_total]) + decode delta.
    /// Owned by the slot; `mrope_pos` freed on deinit.
    mrope_pos: ?[]const i32,
    mrope_total: usize,
    mrope_delta: i32,

    /// Forward context backed by the fields above. Initialized in `init`
    /// and aliased by the Generator's own `ctx` field at prefill time.
    ctx: ForwardCtx,

    /// Generator (constructed on inference thread post-prefill).
    legacy_gen: ?Generator,

    // ── Submission data. Owned by the slot, freed in deinit. ──
    prompt_ids: []u32,
    full_prompt: []u32,
    sampling: SamplingParams,
    eos_token_ids: []u32,
    max_tokens: u32,
    timeout_ns: u64,
    has_tools: bool,
    enable_thinking: bool,
    enable_drafter: bool,
    dflash: ?*DflashModel,
    drafter_block_size: u32,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    mtp: ?generate_mod.MtpHeadRef,
    mtp_depth: u32,
    cached_tokens: u32,
    logprobs_n: u32,
    serial_reason_logged: bool = false,
    memory_hold_logged: bool = false,
    /// The `[queue]` line was logged for this pending slot (inference thread, `queue_mu`).
    queue_logged: bool = false,
    /// Bytes of cache growth admission priced for this request that it had not allocated yet;
    /// `slotOutstandingGrowth` is the part still unallocated.
    growth_commit: u64 = 0,
    /// This tick decodes plain (batched) although the slot's MTP head is armed: the
    /// batched forward captures its hidden so the next solo tick can resume speculating.
    mtp_plain_tick: bool = false,
    planner_plain_transition: bool = false,
    planner_price_transition: bool = false,
    planner_last_width: u8 = 255,
    planner_force_plain: bool = false,
    mtp_publish_ns: u64 = 0,
    mtp_publish_gap_ms: f32 = 0,

    // ── State + output channel. ──
    state: SlotState,

    out_mu: std.Io.Mutex,
    out_cond: std.Io.Condition,
    /// Wake signal for `waitNextTimeout` — set alongside every `out_cond`
    /// broadcast. Events support timed waits (Io.Condition does not), which
    /// is what lets the conn thread poll the peer socket during long
    /// prefills instead of blocking until the first token.
    out_event: std.Io.Event,
    out_buf: std.ArrayList(u32),
    out_idx: usize,
    finished: bool,
    error_code: ?[]const u8,
    memory_context_refusal: ?model_registry_mod.MemoryContextRefusal = null,
    finish_reason: []const u8,
    /// Set ONLY by the degenerate-tail guard. The wire reason is "stop" so a
    /// client does not mistake the guard for output/context exhaustion; this
    /// sibling signal preserves the specific cause. Static string, never freed.
    finish_details: ?[]const u8,
    /// Index into the emitted tokens where the degenerate span begins;
    /// everything from here on is the loop. Non-streaming responses are cut
    /// here so the client cannot round-trip the loop into the next prompt.
    loop_trim_start: ?usize,
    /// t1 streamed at the MTP prefill handover; the first block still commits it, and its
    /// echo is not published twice (`takeHandoverEcho`).
    handover_token: ?u32 = null,
    /// The EOS (or pad) that ended the turn, never pushed or counted; names the `[stop]` line.
    stop_token: ?u32 = null,
    /// `"spec"` or `"batch"` where the finishing tick knew; empty = derive from the slot's mode.
    stop_path: []const u8 = "",
    cancelled: std.atomic.Value(bool),
    /// The server, not the client, ended this slot (shutdown drain): readers see an error, never a finish reason.
    shutdown_cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    /// Inference thread only: `cullDecodingWith` committed this cancelled slot, so its cleanup drain must not.
    committed_at_cull: bool = false,
    /// Inference thread only: the request's outcome has been counted in `Metrics`.
    metrics_recorded: bool = false,
    /// Set by the connection thread when the request's own stop sequence completed, before it cancels
    /// the slot: that end is a normal stop, not a disconnect.
    stop_hit: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    /// Inference-thread passes (a prefill, a decode tick) holding this slot, taken
    /// under `queue_mu`. `complete` waits it out: the handler owns sampling state
    /// the pass reads (`think_bound`, `constraint`) and frees it once `complete` returns.
    in_pass: std.atomic.Value(u32) = .init(0),
    /// Reasoning-protocol payload boundary, published by the inference thread
    /// BEFORE the token that carries it is pushed (single-writer atomics, so
    /// the conn thread reading token i already sees its span). `token_index`
    /// is the index into the generated stream; `maxInt(u32)` = not started.
    constraint_payload_at: std.atomic.Value(u32),
    constraint_payload_offset: std.atomic.Value(u32),

    // ── Stats (filled by inference thread, safe to read after finish). ──
    prompt_tokens: u32,
    completion_tokens: u32,
    prefill_tps: f64,
    decode_tps: f64,
    /// Monotonic submit sequence stamped by `submit` under `queue_mu`. Immutable
    /// after assignment; it is what keeps one request's `/metrics.json` session
    /// row the same row across polls (the wire `chatcmpl` id is minted later,
    /// at response time, and does not map to this).
    request_id: u64 = 0,
    abandoned_next: ?*Slot = null,
    /// Monotonic timestamp captured in `Slot.init`, BEFORE the queue wait.
    /// Anchors the exact time-to-first-token measurement.
    request_start_ts: std.Io.Timestamp,
    /// Wall-clock nanoseconds from `request_start_ts` (request arrival) to
    /// prefill completion = queue_wait + prefill = real time-to-first-token.
    /// Captured directly when prefill finishes (never derived by subtracting
    /// decode time), so it stays exact even if a slot finishes mid-tick. Used
    /// as `real_ttft_ns` for the metrics histogram; e2e = first_token_ns + decode_ns.
    first_token_ns: u64,
    /// Wall-clock nanoseconds spent in `runPrefill` for this slot. Includes
    /// hot-prefix-cache lookup/restore and the model forward over the
    /// uncached tail. Populated by the scheduler main loop.
    prefill_ns: u64,
    /// Wall-clock nanoseconds of interleaved decode ticks hosted INSIDE this
    /// slot's prefill (chunk-boundary yields). Charged to the decoding slots
    /// that received the tokens; subtracted from this slot's `prefill_ns` so
    /// prefill_tps stays a statement about the prefill forward.
    prefill_interleaved_ns: u64,
    /// Wall-clock nanoseconds the slot spent in decode ticks. For batched
    /// decode the full tick wall-clock is added to every participating slot,
    /// so this matches the per-slot throughput a user actually observes
    /// (`completion_tokens / decode_ns`).
    decode_ns: u64,
    /// Actual generated ids (from legacy_gen.generated_ids). Shallow copy at
    /// completion so the connection thread can read them without locking.
    generated_ids: ?[]u32,
    /// pad-only flag: set by inference thread when the entire generation was
    /// token id 0. Server-level cache invalidation reads this after complete.
    was_pad_only: bool,
    /// Phase A5: per-token logprobs accumulated by the inference thread when
    /// `logprobs_n > 0`. The conn thread takes ownership via
    /// `nonStreamingViaScheduler` (or equivalent) at completion; if the
    /// caller doesn't consume, `Slot.deinit` frees the contents.
    logprobs_buf: std.ArrayList(sampler.LogprobResult),

    /// Initialize but do NOT take ownership of caches — those are allocated
    /// inside `init` from the slot's allocator.
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        config: *const ModelConfig,
        params: SubmitParams,
        kv_quant_config: transformer_mod.KVQuantConfig,
    ) !*Slot {
        const slot = try allocator.create(Slot);
        errdefer allocator.destroy(slot);

        // Per-slot KVCache, honoring the process-level kv-quant setting.
        var cache = try KVCache.initWithConfig(allocator, config.num_hidden_layers, kv_quant_config);
        errdefer cache.deinit();
        if (config.swaRingTokens() > 0) cache.setSwaRing(config.sliding_window);

        // Per-slot SSM cache. Mirror the same predicate `Transformer.init`
        // uses to allocate `xfm.ssm_entries` (`needsSsmEntries`). Without this
        // branch the slot's `ctx.ssm_entries` is null and the
        // linear-attention layers crash on `ctx.ssm_entries.?`. Pure-attention
        // MoE (isMoe() but interval == 0) must NOT get entries: a non-null
        // slice makes the hot prefix cache treat the model as hybrid and
        // cold-prefill every request.
        var ssm_entries: ?[]SSMCacheEntry = null;
        if (config.needsSsmEntries()) {
            const entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
            for (entries) |*e| {
                e.* = .{
                    .conv_state = mlx.mlx_array_new(),
                    .ssm_state = mlx.mlx_array_new(),
                    .initialized = false,
                };
            }
            ssm_entries = entries;
        }
        errdefer if (ssm_entries) |entries| {
            for (entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                transformer_mod.ssmFreeQsaState(e);
            }
            allocator.free(entries);
        };
        var glm5_request: ?glm5_forward_mod.Request = if (config.isGlm5()) try glm5_forward_mod.Request.initServing(allocator, config.num_hidden_layers) else null;
        errdefer if (glm5_request) |*request| request.deinit();

        // Dup owned slices.
        const prompt_owned = try allocator.dupe(u32, params.prompt_ids);
        errdefer allocator.free(prompt_owned);
        const full_prompt_src = params.full_prompt orelse params.prompt_ids;
        const full_prompt_owned = try allocator.dupe(u32, full_prompt_src);
        errdefer allocator.free(full_prompt_owned);
        const media_chain_owned = try allocator.dupe(prefix_cache_mod.MediaSpan, params.media_chain);
        errdefer allocator.free(media_chain_owned);
        const media_start = firstMediaPlaceholder(
            params.vision_embeddings != null or params.deferred_media != null,
            full_prompt_owned,
            config.image_token_id,
            config.video_token_id,
        );
        const eos_owned = try allocator.dupe(u32, params.eos_token_ids);
        errdefer allocator.free(eos_owned);

        slot.* = .{
            .allocator = allocator,
            .io = io,
            .model = params.model,
            .cache = cache,
            .moe_seq_offset = 0,
            .ssm_entries = ssm_entries,
            .glm5_request = glm5_request,
            .vision_embeddings = params.vision_embeddings,
            .deferred_media = params.deferred_media,
            .vision_key = params.vision_key,
            .cache_key = params.cache_key,
            .media_start = media_start,
            .media_chain = media_chain_owned,
            .mrope_pos = params.mrope_pos,
            .mrope_total = params.mrope_total,
            .mrope_delta = params.mrope_delta,
            .ctx = undefined, // set after slot is in stable storage so pointers are valid
            .legacy_gen = null,
            .prompt_ids = prompt_owned,
            .full_prompt = full_prompt_owned,
            .sampling = params.sampling,
            .eos_token_ids = eos_owned,
            .max_tokens = params.max_tokens,
            .timeout_ns = params.timeout_ns,
            .has_tools = params.has_tools,
            .enable_thinking = params.enable_thinking,
            .enable_drafter = params.enable_drafter,
            .dflash = params.dflash,
            .drafter_block_size = params.drafter_block_size,
            .enable_mtp = params.enable_mtp,
            .allow_batch_mtp = params.allow_batch_mtp,
            .mtp = params.mtp,
            .mtp_depth = params.mtp_depth,
            .cached_tokens = params.cached_tokens,
            .logprobs_n = params.logprobs_n,
            .state = .pending_prefill,
            .out_mu = .init,
            .out_cond = .init,
            .out_event = .unset,
            .out_buf = std.ArrayList(u32).empty,
            .out_idx = 0,
            .finished = false,
            .error_code = null,
            .finish_reason = "length",
            .finish_details = null,
            .loop_trim_start = null,
            .cancelled = std.atomic.Value(bool).init(false),
            .shutdown_cancelled = std.atomic.Value(bool).init(false),
            .constraint_payload_at = std.atomic.Value(u32).init(std.math.maxInt(u32)),
            .constraint_payload_offset = std.atomic.Value(u32).init(0),
            .prompt_tokens = 0,
            .completion_tokens = 0,
            .prefill_tps = 0.0,
            .decode_tps = 0.0,
            .request_start_ts = std.Io.Timestamp.now(io, .boot),
            .first_token_ns = 0,
            .prefill_ns = 0,
            .prefill_interleaved_ns = 0,
            .decode_ns = 0,
            .generated_ids = null,
            .was_pad_only = true,
            .logprobs_buf = .empty,
        };

        // ForwardCtx points at fields owned by `slot` — must outlive the
        // Generator. The slot is heap-allocated so addresses are stable
        // until `complete` frees it.
        slot.ctx = .{
            .cache = &slot.cache,
            .moe_seq_offset = &slot.moe_seq_offset,
            .ssm_entries = slot.ssm_entries,
            .ssm_member_gen = transformer_mod.nextSsmMemberGen(),
            .glm5_request = if (slot.glm5_request) |*request| request else null,
            .vision_embeddings = slot.vision_embeddings,
            .mrope_pos = slot.mrope_pos,
            .mrope_total = slot.mrope_total,
            .mrope_delta = slot.mrope_delta,
            .capture_hidden = null,
        };

        return slot;
    }

    /// Free everything the slot owns. Only safe to call when no thread can
    /// observe the slot anymore (i.e. after the inference thread has
    /// finished/errored it AND the connection thread has consumed the final
    /// `done`/`err` from `waitNext`).
    pub fn deinit(self: *Slot) void {
        if (self.model.transformer) |xfm| xfm.markQsaPooledRopeStale();
        if (self.legacy_gen) |*gen| {
            gen.deinit(self.allocator);
        }
        // Salvaged-but-never-consumed cancelled-prefill checkpoints.
        self.cancelled_prefill.deinit();
        self.ring_cps.deinit();
        self.cache.deinit();
        if (self.glm5_request) |*request| request.deinit();
        if (self.ssm_entries) |entries| {
            if (self.model.transformer) |xfm| xfm.ssmGroupDrop(entries);
            for (entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                transformer_mod.ssmFreeQsaState(e);
            }
            self.allocator.free(entries);
        }
        if (self.vision_embeddings) |ve| freeVisionArray(ve);
        if (self.deferred_media) |d| d.deinit();
        if (self.mrope_pos) |mp| self.allocator.free(mp);
        self.allocator.free(self.prompt_ids);
        self.allocator.free(self.full_prompt);
        self.allocator.free(self.media_chain);
        self.allocator.free(self.eos_token_ids);
        if (self.error_code) |code| {
            if (code.ptr != static_error_name.ptr) self.allocator.free(code);
        }
        if (self.generated_ids) |g| self.allocator.free(g);
        // Free any logprobs the conn thread didn't claim. After
        // `nonStreamingViaScheduler` calls `toOwnedSlice`, items.len becomes
        // 0 so this is a no-op on the success path.
        for (self.logprobs_buf.items) |*lp| self.allocator.free(lp.top_logprobs);
        self.logprobs_buf.deinit(self.allocator);
        self.out_buf.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Inference thread: record the reasoning→payload boundary so it is
    /// visible to the conn thread no later than the token that carries it.
    fn publishConstraintSpan(self: *Slot, token_index: u32, byte_offset: u32) void {
        self.constraint_payload_offset.store(byte_offset, .release);
        self.constraint_payload_at.store(token_index, .release);
    }

    /// Connection thread: where the constrained payload begins relative to
    /// the generated stream, or null while it has not begun. `token_index`
    /// matches the reader's own token count (token i carries the payload
    /// start when the returned index equals i).
    pub fn constraintPayloadStart(self: *const Slot) ?rp_mod.ConstraintSpan {
        const at = self.constraint_payload_at.load(.acquire);
        if (at == std.math.maxInt(u32)) return null;
        return .{ .token_index = at, .byte_offset = self.constraint_payload_offset.load(.acquire) };
    }

    /// Inference thread: enqueue a generated token for the consumer.
    fn pushToken(self: *Slot, t: u32) void {
        self.pushTokenWithLogprob(t, null);
    }

    /// True once, for the first token published after the handover when it is t1 again.
    fn takeHandoverEcho(self: *Slot, t: u32) bool {
        const h = self.handover_token orelse return false;
        self.handover_token = null;
        if (h == t) return true;
        log.err("[mtp] handover streamed {d} but the first committed token is {d}\n", .{ h, t });
        return false;
    }

    /// Publish one token and, when the request asked for logprobs, the entry
    /// describing it — in ONE critical section.
    ///
    /// Both must move under `out_mu` together. The streaming path reads entry
    /// i as soon as it is handed token i, so an append outside the lock races
    /// the reader two ways: the entry may not be visible yet (a silently
    /// missing trailing logprob), and a concurrent grow reallocates the
    /// backing array under a reader mid-copy. Non-streaming consumes the whole
    /// buffer at completion and is blind to both, which is why the streaming
    /// gap survived: it is invisible to output-equality tests AND to llmprobe,
    /// which probes logprobs non-streaming only.
    pub fn pushTokenWithLogprob(self: *Slot, t: u32, lp: ?sampler.LogprobResult) void {
        if (self.takeHandoverEcho(t)) return;
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        if (lp) |entry| {
            // Degrade to error like the token append below — and do NOT return:
            // the broadcast at the bottom is what wakes a reader blocked on the
            // condvar, so an early exit here trades an OOM for a hang.
            self.logprobs_buf.append(self.allocator, entry) catch |err| self.latchErrorLocked(@errorName(err));
        }
        self.out_buf.append(self.allocator, t) catch |err| self.latchErrorLocked(@errorName(err));
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }

    /// Connection thread: copy logprob entries produced since `cursor` into
    /// `out`, returning the new cursor. Under `out_mu`, so it cannot observe a
    /// half-written entry or a reallocating buffer.
    ///
    /// The COPY is shallow and that is deliberate: each entry's `top_logprobs`
    /// is its own allocation, stable for the life of the slot, and owned by
    /// the slot (`Slot.deinit` frees it). The caller borrows and must not free
    /// — only the ArrayList's backing array is at risk from a concurrent grow,
    /// and that is exactly what the lock covers.
    pub fn copyLogprobsFrom(
        self: *Slot,
        allocator: std.mem.Allocator,
        cursor: usize,
        out: *std.ArrayList(sampler.LogprobResult),
    ) !usize {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        const items = self.logprobs_buf.items;
        if (cursor >= items.len) return cursor;
        try out.appendSlice(allocator, items[cursor..]);
        return items.len;
    }

    /// Connection thread: take ownership of the first `limit` logprob entries. A slot cancelled
    /// mid-decode may hold entries for tokens the caller never read; those are freed here.
    pub fn takeLogprobs(self: *Slot, limit: usize) ?[]sampler.LogprobResult {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        while (self.logprobs_buf.items.len > limit) self.allocator.free(self.logprobs_buf.pop().?.top_logprobs);
        if (self.logprobs_buf.items.len == 0) return null;
        return self.logprobs_buf.toOwnedSlice(self.allocator) catch null;
    }

    /// Inference thread: signal normal completion. Safe to call multiple
    /// times (idempotent on `finished`).
    pub fn markFinished(self: *Slot, reason: []const u8) void {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        if (self.finished) return;
        self.finished = true;
        self.state = .finished;
        self.finish_reason = reason;
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }

    /// Inference thread: signal error. `name` is borrowed; we dupe so the
    /// connection thread can read it after the inference loop drops the slot.
    /// Whether a slot error name names a memory failure (`OutOfMemory` from the MLX latch or Zig's allocator).
    pub fn errorNameIsMemory(name: []const u8) bool {
        return std.mem.eql(u8, name, "OutOfMemory") or
            std.mem.eql(u8, name, "InsufficientMemory");
    }

    /// Whether the slot's latched error name is exactly `name`.
    pub fn errorNameIs(self: *Slot, name: []const u8) bool {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        const have = self.error_code orelse return false;
        return std.mem.eql(u8, have, name);
    }

    pub fn errorIsMemory(self: *Slot) bool {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        const name = self.error_code orelse return false;
        return errorNameIsMemory(name);
    }

    /// Caller holds `out_mu`. `error_code != null` is what the consumers and the cull read as
    /// terminal, so an uncopyable name falls back to a static one instead of leaving it null.
    fn latchErrorLocked(self: *Slot, name: []const u8) void {
        if (self.error_code == null) self.error_code = self.allocator.dupe(u8, name) catch static_error_name;
        self.state = .errored;
    }

    pub fn markError(self: *Slot, name: []const u8) void {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        if (self.error_code != null or self.finished) return;
        self.latchErrorLocked(name);
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }

    /// Connection thread: block until the next token, completion, or error.
    /// Returns `.token` for each generated id, then exactly one terminator
    /// (`.done` or `.err`).
    pub fn waitNext(self: *Slot) NextResult {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        while (true) {
            if (self.out_idx < self.out_buf.items.len) {
                const t = self.out_buf.items[self.out_idx];
                self.out_idx += 1;
                return .{ .token = t };
            }
            if (self.error_code != null) return .{ .err = {} };
            if (self.finished) return .{ .done = {} };
            // Cancellation (client disconnect or server shutdown) must unblock a
            // blocked reader — `cancel()` only broadcasts; without this the
            // reader would sleep until the inference thread happened to finish
            // the slot, so shutdown could never drain in-flight requests and
            // raced `Scheduler.deinit` into a use-after-free (SIGSEGV in
            // `complete`). Buffered tokens above still drain first.
            if (self.cancelled.load(.acquire)) return self.cancelledResult();
            self.out_cond.waitUncancelable(self.io, &self.out_mu);
        }
    }

    /// Connection thread: like `waitNext`, but wakes with `null` (idle)
    /// after `timeout_ms` with no token or terminator. Lets the caller poll
    /// the peer socket and emit SSE keepalives during long prefills —
    /// Claude Code disconnects after ~60s of stream silence, and pre-2026-06
    /// the handler sat blocked in `waitNext` for the whole multi-minute
    /// prefill, never noticed the disconnect, and abandoned giant prefills
    /// piled up serially behind every client retry (the server looked dead
    /// while the GPU ground ghosts).
    pub fn waitNextTimeout(self: *Slot, timeout_ms: i64) ?NextResult {
        while (true) {
            self.out_mu.lockUncancelable(self.io);
            if (self.out_idx < self.out_buf.items.len) {
                const t = self.out_buf.items[self.out_idx];
                self.out_idx += 1;
                self.out_mu.unlock(self.io);
                return .{ .token = t };
            }
            if (self.error_code != null) {
                self.out_mu.unlock(self.io);
                return .{ .err = {} };
            }
            if (self.finished) {
                self.out_mu.unlock(self.io);
                return .{ .done = {} };
            }
            // Cancellation unblocks the reader promptly (see waitNext) so a
            // disconnected/shutdown request stops instead of waiting out the
            // generation — the precondition for draining conn threads before
            // teardown.
            if (self.cancelled.load(.acquire)) {
                self.out_mu.unlock(self.io);
                return self.cancelledResult();
            }
            // Arm the event under the lock: producers mutate under this lock
            // and set() before releasing it, so a set racing our reset leaves
            // the event set and the wait below returns immediately — no lost
            // wakeups. A spurious wake just reads as an early idle (benign).
            self.out_event.reset();
            self.out_mu.unlock(self.io);
            self.out_event.waitTimeout(self.io, .{ .duration = .{
                .raw = .fromMilliseconds(timeout_ms),
                .clock = .awake,
            } }) catch return null;
        }
    }

    fn cancelledResult(self: *Slot) NextResult {
        return if (self.shutdown_cancelled.load(.acquire)) .{ .err = {} } else .{ .done = {} };
    }

    /// Connection thread: signal cancellation. The inference thread will
    /// drop this slot at the next tick boundary.
    /// Connection thread: the request's own stop sequence completed; end generation as a normal stop.
    pub fn cancelOnStop(self: *Slot) void {
        self.stop_hit.store(true, .release);
        self.cancel();
    }

    pub fn cancelForShutdown(self: *Slot) void {
        self.shutdown_cancelled.store(true, .release);
        self.cancel();
    }

    pub fn cancel(self: *Slot) void {
        self.cancelled.store(true, .release);
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }
};

/// Plan 05 Phase D: cold-load work item. Posted by `Scheduler.ensureLoaded`
/// when a request targets an `.unloaded` (or freshly-evicted) entry; the
/// inference thread drains the queue between ticks. The conn thread parses
/// `config.json` / tokenizer / chat_config on its own thread (CPU only,
/// no mlx ops) and hands the pre-parsed CPU state to the inference thread,
/// which does the mlx-allocating work (weights + Transformer + vision +
/// drafter + JIT + warmup) and installs everything on `entry`.
///
/// Eviction: when set, `evict_entry` is unloaded on the inference thread
/// BEFORE the new load starts. The conn thread has already marked the
/// victim `.evicting` and waited for refcount == 0, so freeing GPU memory
/// is safe.
pub const LoadRequest = struct {
    /// Target entry to populate. Already transitioned to `.loading` by
    /// the conn thread before posting; the inference thread completes the
    /// load and calls `registry.markReadyLocked(entry, bytes)` or
    /// `markErrorLocked` on failure.
    entry: *LoadedModel,
    /// Pre-parsed CPU state. Ownership transfers to `entry` on success;
    /// on failure the conn thread takes them back via the `done` cond-var
    /// and frees them.
    config: *ModelConfig,
    tok: *Tokenizer,
    chat_config: *ChatConfig,

    /// Borrowed paths. Conn thread keeps the buffers alive until `done`.
    model_dir: []const u8,
    drafter_dir: []const u8 = "",
    /// `--no-drafter`: never load a drafter, including one MERGED into the
    /// checkpoint. `drafter_dir == ""` stopped meaning "off" the moment a
    /// checkpoint could carry its own, so the opt-out needs its own bit.
    no_drafter: bool = false,
    /// Auto-load the Qwen native MTP sidecar when the model dir ships one.
    mtp_enabled: bool = true,
    mtp_explicit: bool = false,
    mtp_head_kv_quant: bool = false,
    /// Max MTP draft depth (CLI --mtp-max-depth; 0 = auto, resolved by
    /// generate_mod.resolveMtpDepthCap at load/Generator init).
    mtp_depth: u32 = 0,

    load_vision: bool = false,
    warmup_eager: bool = true,
    kv_quant_config: transformer_mod.KVQuantConfig = transformer_mod.KVQuantConfig.engine_default,
    kv_quant_explicit: bool = false,
    prefix_cache_capacity: u32 = 1,
    prefix_cache_ram_enabled: bool = true,
    prefix_cache_mem_bytes: u64 = 0,
    prefix_cache_mem_resolver: ?*const fn (*model_mod.ModelConfig, u64, BudgetRevise, *u64) u64 = null,
    /// SSD tier byte budget (mirrors `LoadParams.prefix_cache_disk_bytes`).
    prefix_cache_disk_bytes: u64 = 0,
    prefix_cache_disk_resolver: ?*const fn (*model_mod.ModelConfig, []const u8, u64) ?disk_placement.DefaultDiskBudget = null,
    ssd_budget_bytes: u64 = 0,
    expert_cache_fit_resolver: ?*const fn (*const model_mod.ModelConfig, u64) anyerror!void = null,
    /// Phase 1 (perf-plan): SSM/conv state snapshot stride during prefill.
    /// Zero disables (hybrid models bypass the hot prefix cache, as before).
    /// Non-zero enables multi-turn warm reuse on hybrid SSM archs. Plumbed
    /// to `HotPrefixCache.shouldUse(enable_ssm_checkpoints = stride > 0)`
    /// and to every `Generator.initWithOptions` call so the prefill loop
    /// captures snapshots.
    ssm_checkpoint_stride: u32 = 0,
    /// Phase 1: maximum checkpoints retained per request. Older ones are
    /// dropped front-first when the buffer would grow past this. 0 = no cap
    /// beyond the prefix-cache byte budget.
    ssm_checkpoint_max: u32 = 16,
    /// Iteration 2: tokenize cache LRU capacity. Mirrored on
    /// `LoadParams.tokenize_cache_entries`; both paths feed
    /// `doLoadOnInferenceThread`.
    tokenize_cache_entries: u32 = 4,

    /// Victims to evict before the load (LRU-selected by the planner). Each is
    /// already marked `.evicting` with refcount == 0 by the conn thread under
    /// registry.mutex. The inference thread calls `unloadResident()` on each to
    /// free GPU memory, drops its resident-bytes accounting via
    /// `registry.accountEvictedLocked`, then `registry.finalizeEvictionLocked`.
    /// Borrows the conn thread's stack buffer; valid until `done`.
    evict_entries: []*LoadedModel = &.{},

    /// Output: error name on failure (owned by `allocator`; conn thread
    /// frees). Null on success.
    error_name: ?[]const u8 = null,
    memory_context_refusal: ?model_registry_mod.MemoryContextRefusal = null,

    /// Conn-thread synchronization. Inference thread broadcasts when done.
    done: bool = false,
    allocator: std.mem.Allocator,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
};

/// Model-unload work item. Posted by `unloadModel` after the conn thread
/// marked the entry `.evicting` and drained its refcount. The inference thread
/// frees the entry's resident mlx state (stream-bound) and finalizes the
/// eviction accounting, then the entry returns to `.unloaded` (the stub stays
/// in the registry so it can reload later).
pub const UnloadRequest = struct {
    entry: *model_registry_mod.LoadedModel,
    done: bool = false,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
};

// Connection-local transport preserves a load diagnostic across a concurrent retry.
threadlocal var last_load_context_refusal: ?model_registry_mod.MemoryContextRefusal = null;

/// Continuous-batching scheduler. One per server. Owns the inference
/// thread, the queue of in-flight slots, AND (post-A1) the loaded model
/// state — Transformer + weights + vision encoder + drafter all live here,
/// allocated on the inference thread so mlx's thread-local GPU stream is
/// bound on the right thread from the start.
pub const Scheduler = struct {
    allocator: std.mem.Allocator,
    io: std.Io,

    // ── Plan 05 — multi-model state. Source of truth for model fields is
    //    `current_model` (a borrowed *LoadedModel owned by the registry).
    //    The fields below (`xfm`, `weights`, …) are *borrowed views* into
    //    `current_model` set at load time so existing scheduler-internal
    //    code can keep reading them as fields. Phase D will refresh these
    //    views on every model swap inside `pickNextTickWork`.
    registry: *ModelRegistry,
    current_model: ?*LoadedModel,

    // ── Borrowed views (non-owning). Cleared on shutdown via
    //    `clearCurrentModelViews`. Null only before load completes or after
    //    an eviction in Phase D.
    xfm: ?*Transformer,
    weights: ?*Weights,
    vision_encoder: ?*VisionEncoder,
    dflash: ?*DflashModel = null,
    drafter_block_size: u32,
    kv_quant_config: transformer_mod.KVQuantConfig,
    kv_quant_explicit: bool,
    /// `LoadParams.ctx_size`, retained so every load resolves its context
    /// against the launch flag (`model_settings.contextPick`).
    ctx_size_flag: u32,
    /// Launch-flag prefix-cache settings, retained so COLD-LOADED models
    /// (`ensureLoaded` → /v1/load-model, model switches) get the same
    /// prefix-cache behavior as the `--model` primary. Pre-plumbing these
    /// were hardcoded to (1, 0, stride 0) on the cold path, which silently
    /// crippled warm reuse — and disabled it entirely on hybrids — after
    /// every model switch.
    prefix_cache_capacity: u32,
    prefix_cache_ram_enabled: bool,
    prefix_cache_mem_bytes: u64,
    prefix_cache_mem_resolver: ?*const fn (*model_mod.ModelConfig, u64, BudgetRevise, *u64) u64,
    prefix_cache_disk_bytes: u64,
    prefix_cache_disk_resolver: ?*const fn (*model_mod.ModelConfig, []const u8, u64) ?disk_placement.DefaultDiskBudget,
    ssd_budget_bytes: u64,
    expert_cache_fit_resolver: ?*const fn (*const model_mod.ModelConfig, u64) anyerror!void,
    ssm_checkpoint_stride: u32,
    ssm_checkpoint_max: u32,
    /// Launch-flag MTP settings, retained (same rationale as the prefix-cache
    /// fields above) so COLD-LOADED models — on-demand `/v1/load-model`, model
    /// switches — honor `--no-mtp` / `--mtp-max-depth` like the `--model` primary.
    /// Pre-plumbing, the cold-load `LoadRequest` used its struct defaults
    /// (mtp on, default depth), silently ignoring these flags on every
    /// on-demand load and model switch.
    mtp_enabled: bool,
    mtp_explicit: bool,
    mtp_head_kv_quant: bool,
    mtp_depth: u32,
    /// Launch-flag drafter settings, retained for cold loads. `--no-drafter`
    /// became load-bearing on this path the moment `dflash.resolveInDirDrafter`
    /// started probing `<model_dir>/drafter` at load: without it here, a server
    /// launched with speculation off re-enabled it on every model switched to.
    /// `drafter_dir` is the launch `--drafter` path and `primary_model_dir` the
    /// `--model` it belongs to — see `coldLoadDrafterDir` for why it is not
    /// simply copied across.
    no_drafter: bool,
    drafter_dir: []const u8,
    primary_model_dir: []const u8,

    // ── Borrowed refs (CPU-only state owned by the LoadedModel). ──
    drafter_path: []const u8,

    /// Phase A6 → Plan 05: per-model hot prefix cache. Pre-Plan-05 this was
    /// a server-owned global; Plan 05 moves it onto `LoadedModel` so each
    /// model gets isolation by construction. Borrowed view here is the
    /// current model's cache (or null when the cache isn't applicable for
    /// the model, e.g. hybrid SSM archs).
    hot_prefix_cache: ?*prefix_cache_mod.HotPrefixCache,
    /// Resident hot-cache bytes, published for the connection thread: `hot_prefix_cache` is
    /// inference-thread state, freed on every model switch, so the guard reads this number
    /// and never the pointer.
    resident_hot_cache_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// KV + recurrent state the live slots own beyond what the hot caches bill, once per tick (`/props`).
    resident_live_kv_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    /// The part of the above an eviction can prove it will return (residency minus the largest entry).
    reclaimable_hot_cache_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// Per-slot context snapshot for `/metrics.json`, published by the inference
    /// thread inside the `queue_mu` holds that already bracket the cull, the
    /// prefill entry, and the interleave chunk boundaries; readers copy under the
    /// same lock (the `hot_cache_digests` discipline, on the queue instead).
    live_sessions: [metrics_mod.MAX_SESSIONS]metrics_mod.Session = undefined,
    live_session_count: usize = 0,
    /// Submit sequence the session rows carry as `request_id`: taken under
    /// `queue_mu` in `submit`, immutable on the slot, never reused.
    req_seq: u64 = 0,
    /// Every ready model's hot-cache entries for `/metrics.json`, under `digest_mu`.
    cached_sessions: [metrics_mod.MAX_SESSIONS]metrics_mod.Session = undefined,
    cached_session_count: usize = 0,

    /// Set on unload: the OS hands freed pages back lazily, so the budget revise repeats
    /// before each prefill batch while this is armed and settles as the ceiling recovers.
    budget_revise_sw: ?io_util.Stopwatch = null,

    /// Per-entry digest snapshot the connection-thread guard reads instead of the cache.
    /// Replaced under `digest_mu` by the inference thread; readers copy under the lock.
    hot_cache_digests: []prefix_cache_mod.HotPrefixCache.EntryDigest = &.{},
    /// The residency the digests describe, published in the same critical section.
    digest_residency: u64 = 0,
    digest_mu: std.Io.Mutex = .init,

    max_concurrent: u32,
    /// Phase A7 test hook: when true, `runDecodeTick` forces the batched
    /// kernel even at `active.len == 1`. Set via the `SUSHI_FORCE_BATCHED`
    /// environment variable (`=1` to enable). Test-only — production uses
    /// the auto-gate that drops to `runSingleDecodeTick` for single-slot
    /// requests because that path is bit-identical to legacy and supports
    /// speculative decoding (which the batched kernel doesn't).
    force_batched: bool,

    queue_mu: std.Io.Mutex,
    queue_cond: std.Io.Condition,
    pending: std.ArrayList(*Slot),
    decoding: std.ArrayList(*Slot),
    /// Phase A4: pending vision-encode requests. The inference thread drains
    /// these in the gap before/after each prefill+decode tick. Posting
    /// broadcasts on `queue_cond` to wake an idle inference thread.
    vision_queue: std.ArrayList(*VisionEncodeRequest),
    /// Phase D: pending cold-load requests. Conn threads post here via
    /// `scheduler.ensureLoaded`; the inference thread drains between ticks
    /// (load runs after cleanup + vision/embed, before prefill). Multiple
    /// concurrent requesters for the same id share one load via the
    /// `.loading` state on the entry — `ensureLoaded` only posts when it
    /// successfully flips the entry from `.unloaded` → `.loading`.
    load_queue: std.ArrayList(*LoadRequest),
    /// Pending model-unload jobs. Conn threads post here via `unloadModel`
    /// after marking the entry `.evicting` + draining its refcount; the
    /// inference thread frees the mlx state (stream-bound, like cleanup).
    unload_queue: std.ArrayList(*UnloadRequest),
    /// Slots awaiting cleanup. The conn thread queues a slot here in
    /// `complete()` instead of calling `slot.deinit()` directly — `deinit`
    /// frees mlx_arrays via refcount-decrement, and the underlying GPU
    /// memory release races against the inference thread's stream. The
    /// inference thread drains this queue between ticks where it owns the
    /// stream binding, so all mlx ops stay on one thread.
    cleanup_queue: std.ArrayList(*Slot),
    /// Slots whose `submit` failed after they were built, linked through `Slot.abandoned_next`
    /// (queue_mu). An intrusive list, so handing one over cannot fail for want of memory.
    abandoned: ?*Slot,
    /// Vision arrays of requests whose `submit` failed before a slot took them (queue_mu).
    /// Fixed-size, so handing one over never allocates; the inference thread frees them.
    orphan_vision: [orphan_vision_cap]mlx.mlx_array,
    orphan_vision_n: usize,
    /// Slots out of `pending` whose prefill pass is running: neither pending nor decoding, so
    /// a shutdown reaches them only here (queue_mu).
    prefilling: std.ArrayList(*Slot),
    /// Metrics sink. Null while metrics are off. Populated from LoadParams, and adopted by the inference thread
    /// (its only writer) once `POST /metrics/enable` publishes one. Read once per REQUEST in `finishSlot`.
    metrics: ?*metrics_mod.Metrics,
    /// In-flight generated-token aggregate: the sum of `completion_tokens`
    /// over the slots still decoding, republished by the inference thread once
    /// per decode tick (O(1) at the tick boundary, NOT per token). The gauge
    /// sampler reads this race-free to derive a live tok/s — it never touches
    /// per-slot fields off-thread. Zero when nothing is decoding.
    inflight_generated_tokens: std.atomic.Value(u64),
    /// Tokens forwarded so far by the prefill currently running on the
    /// inference thread; 0 when no prefill is in flight. Mirror of
    /// `inflight_generated_tokens` for the OTHER phase — without it the metrics
    /// panel shows nothing while a multi-minute prefill pins the GPU, because
    /// prompt-token counters and prefill-time histograms only advance when the
    /// request finishes. Written per prefill CHUNK, read by the gauge sampler.
    inflight_prefill_tokens: std.atomic.Value(u64),
    /// Post-cache tail the in-flight prefill will forward, same scale as
    /// `inflight_prefill_tokens`; 0 when none is running.
    inflight_prefill_expected: std.atomic.Value(u64),
    /// Number of slots currently inside `runPrefill`. Set on entry, cleared on
    /// every exit — so the panel can say "prefilling" IMMEDIATELY, rather than
    /// waiting for the first 8192-token chunk to land (~40 s on a 27B).
    requests_prefilling: std.atomic.Value(u64),
    /// Counts `pending.len + decoding.len` for back-pressure.
    in_flight: u32,
    /// Capacity for back-pressure. `submit` waits when in_flight >= cap.
    /// `cap = max_concurrent + queue_depth`. queue_depth = 32 hardcoded
    /// (matches the legacy `max_queue_size`).
    queue_cap: u32,
    submit_cond: std.Io.Condition,

    inference_thread: ?std.Thread,
    shutdown: std.atomic.Value(bool),
    started: std.atomic.Value(bool),
    started_mu: std.Io.Mutex,
    started_cond: std.Io.Condition,

    /// Set true by the inference thread if model load fails. Read by `init`
    /// after `started` is signaled to decide whether to surface a load error.
    load_failed: std.atomic.Value(bool),
    /// Owned, dupe'd error name (e.g. "MissingVisionWeights"). null on
    /// success. Freed in `deinit`.
    load_error_name: ?[]const u8,

    /// Construct a Scheduler whose inference thread loads the model. Returns
    /// only after load + (optional) warmup completes. On load failure, returns
    /// `error.LoadFailed`; the inference thread has already cleaned up any
    /// partially-allocated mlx state by then.
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        params: LoadParams,
        max_concurrent: u32,
    ) !*Scheduler {
        const self = try allocator.create(Scheduler);
        errdefer allocator.destroy(self);

        const cap = if (max_concurrent == 0) 1 else max_concurrent;
        // Phase A7: force-batched test hook. The byte-equivalence test sets
        // `SUSHI_FORCE_BATCHED=1` to verify that the batched-kernel
        // output matches the single-slot path token-for-token at temp=0,
        // single client. Uses libc getenv to stay allocator-free.
        const force_batched = blk: {
            const raw = std.c.getenv("SUSHI_FORCE_BATCHED");
            if (raw == null) break :blk false;
            const slice = std.mem.sliceTo(raw.?, 0);
            break :blk std.mem.eql(u8, slice, "1");
        };
        if (force_batched) {
            log.info("[scheduler] force_batched=on (SUSHI_FORCE_BATCHED=1) — single-slot ticks will route through batched kernel\n", .{});
        }
        self.* = .{
            .allocator = allocator,
            .io = io,
            .registry = params.registry,
            .current_model = null,
            .xfm = null,
            .weights = null,
            .vision_encoder = null,
            .drafter_block_size = 0,
            .kv_quant_config = params.kv_quant_config,
            .kv_quant_explicit = params.kv_quant_explicit,
            .ctx_size_flag = params.ctx_size,
            .prefix_cache_capacity = params.prefix_cache_capacity,
            .prefix_cache_ram_enabled = params.prefix_cache_ram_enabled,
            .prefix_cache_mem_bytes = params.prefix_cache_mem_bytes,
            .prefix_cache_mem_resolver = params.prefix_cache_mem_resolver,
            .prefix_cache_disk_bytes = params.prefix_cache_disk_bytes,
            .prefix_cache_disk_resolver = params.prefix_cache_disk_resolver,
            .ssd_budget_bytes = params.ssd_budget_bytes,
            .expert_cache_fit_resolver = params.expert_cache_fit_resolver,
            .ssm_checkpoint_stride = params.ssm_checkpoint_stride,
            .ssm_checkpoint_max = params.ssm_checkpoint_max,
            .mtp_enabled = params.mtp_enabled,
            .mtp_explicit = params.mtp_explicit,
            .mtp_head_kv_quant = params.mtp_head_kv_quant,
            .mtp_depth = params.mtp_depth,
            .no_drafter = params.no_drafter,
            .drafter_dir = params.drafter_dir,
            .primary_model_dir = params.model_dir,
            .drafter_path = params.drafter_dir,
            .hot_prefix_cache = null,
            .max_concurrent = cap,
            .force_batched = force_batched,
            .queue_mu = .init,
            .queue_cond = .init,
            .pending = std.ArrayList(*Slot).empty,
            .decoding = std.ArrayList(*Slot).empty,
            .vision_queue = std.ArrayList(*VisionEncodeRequest).empty,
            .load_queue = std.ArrayList(*LoadRequest).empty,
            .unload_queue = std.ArrayList(*UnloadRequest).empty,
            .cleanup_queue = std.ArrayList(*Slot).empty,
            .abandoned = null,
            .orphan_vision = undefined,
            .orphan_vision_n = 0,
            .prefilling = std.ArrayList(*Slot).empty,
            .metrics = params.metrics,
            .inflight_generated_tokens = std.atomic.Value(u64).init(0),
            .inflight_prefill_tokens = std.atomic.Value(u64).init(0),
            .inflight_prefill_expected = std.atomic.Value(u64).init(0),
            .requests_prefilling = std.atomic.Value(u64).init(0),
            .in_flight = 0,
            .queue_cap = cap + 32,
            .submit_cond = .init,
            .inference_thread = null,
            .shutdown = std.atomic.Value(bool).init(false),
            .started = std.atomic.Value(bool).init(false),
            .started_mu = .init,
            .started_cond = .init,
            .load_failed = std.atomic.Value(bool).init(false),
            .load_error_name = null,
        };

        const ctx = ThreadCtx{ .scheduler = self, .params = params };
        self.inference_thread = try std.Thread.spawn(.{}, inferenceLoop, .{ctx});

        // Wait until inference thread has loaded the model and (optionally)
        // warmed up. submit() relies on xfm/weights being live.
        self.started_mu.lockUncancelable(io);
        defer self.started_mu.unlock(io);
        while (!self.started.load(.acquire)) {
            self.started_cond.waitUncancelable(io, &self.started_mu);
        }

        if (self.load_failed.load(.acquire)) {
            // Inference thread already exited cleanly. Join + free + bubble up.
            if (self.inference_thread) |t| t.join();
            self.inference_thread = null;
            const name = self.load_error_name orelse "unknown";
            log.err("[scheduler] model load failed: {s}\n", .{name});
            return error.LoadFailed;
        }

        return self;
    }

    pub fn deinit(self: *Scheduler) void {
        self.shutdown.store(true, .release);
        // Wake inference thread if it's waiting on queue_cond.
        self.queue_mu.lockUncancelable(self.io);
        self.queue_cond.broadcast(self.io);
        self.submit_cond.broadcast(self.io);
        self.queue_mu.unlock(self.io);

        if (self.inference_thread) |t| t.join();

        // Freed below the join: the publisher is gone, so this is the last writer.
        if (self.hot_cache_digests.len > 0) {
            self.allocator.free(self.hot_cache_digests);
            self.hot_cache_digests = &.{};
        }

        // Drain any leftover slots — should be empty if all conn threads
        // called `complete` properly, but defensive. Inference thread has
        // already exited by now (joined above), so freeing here is safe.
        for (self.pending.items) |slot| slot.deinit();
        self.pending.deinit(self.allocator);
        for (self.decoding.items) |slot| slot.deinit();
        self.decoding.deinit(self.allocator);
        for (self.cleanup_queue.items) |slot| slot.deinit();
        self.cleanup_queue.deinit(self.allocator);
        destroyAbandoned(self.abandoned);
        for (self.orphan_vision[0..self.orphan_vision_n]) |ve| freeVisionArray(ve);
        self.prefilling.deinit(self.allocator);
        // The vision queue should be empty (encodeVision blocks until done) but
        // guard against shutdown-mid-encode by signaling done with an error.
        for (self.vision_queue.items) |req| {
            req.done_mu.lockUncancelable(self.io);
            req.error_name = self.allocator.dupe(u8, "Shutdown") catch null;
            req.done = true;
            req.done_cond.broadcast(self.io);
            req.done_mu.unlock(self.io);
        }
        self.vision_queue.deinit(self.allocator);
        // Phase D: signal any pending cold-load requesters that the
        // server is shutting down. They roll back their entry state and
        // free pre-loaded CPU resources.
        for (self.load_queue.items) |req| {
            req.done_mu.lockUncancelable(self.io);
            req.error_name = self.allocator.dupe(u8, "Shutdown") catch null;
            req.done = true;
            req.done_cond.broadcast(self.io);
            req.done_mu.unlock(self.io);
        }
        self.load_queue.deinit(self.allocator);
        // Wake any conn threads blocked in unloadModel so they don't hang.
        for (self.unload_queue.items) |req| {
            req.done_mu.lockUncancelable(self.io);
            req.done = true;
            req.done_cond.broadcast(self.io);
            req.done_mu.unlock(self.io);
        }
        self.unload_queue.deinit(self.allocator);

        // Plan 05: mlx-allocating state lives on the `LoadedModel` owned by
        // the registry. We can't free the entries here (registry teardown
        // happens later, after serve() returns), but we DO need to release
        // their mlx pieces while we still have a thread bound to the mlx
        // GPU stream — that's actually the calling thread, since the
        // existing pattern frees mlx_array refcount-zeros from
        // `Scheduler.deinit` directly. Walk EVERY .ready entry in the
        // registry (multi-model: more than just `current_model`).
        {
            self.registry.mutex.lockUncancelable(self.io);
            defer self.registry.mutex.unlock(self.io);
            var it = self.registry.entries.valueIterator();
            while (it.next()) |entry_ptr| {
                const entry = entry_ptr.*;
                if (entry.state == .ready or entry.state == .evicting) {
                    entry.unloadResident();
                }
            }
        }
        self.current_model = null;
        // Clear the borrowed views so post-shutdown reads (defensive) see
        // null rather than dangling pointers.
        self.xfm = null;
        self.weights = null;
        self.vision_encoder = null;
        self.dflash = null;
        self.hot_prefix_cache = null;
        self.resident_hot_cache_bytes.store(0, .monotonic);
        if (hot_cache_budget_invalidate) |f| f();
        self.reclaimable_hot_cache_bytes.store(0, .monotonic);
        if (self.load_error_name) |n| self.allocator.free(n);

        self.allocator.destroy(self);
    }

    /// Submit a new request. Builds a Slot, queues it, returns the handle.
    /// Blocks if the queue is full (i.e. `in_flight >= queue_cap`). When
    /// the scheduler is shutting down this returns `error.Shutdown`.
    pub fn submit(self: *Scheduler, params: SubmitParams) !*Slot {
        // Construct the slot up front so we don't hold the queue mutex
        // through any allocation. Per-request `kv_quant_config` override (Wave
        // 1.A) wins over the process-level default carried on the scheduler.
        // Phase D fix: use the slot's target-model config (not the
        // scheduler's startup-model config) so per-slot state allocation
        // (KVCache shape, SSM entries) matches the model that will
        // actually run the request. Critical when two models with
        // different architectures (e.g. pure-attention + hybrid SSM)
        // share one scheduler.
        const slot = self.initSlot(params) catch |err| {
            if (params.vision_embeddings) |ve| self.orphanVision(ve);
            if (params.deferred_media) |d| d.deinit();
            return err;
        };
        // The slot may own GPU arrays that only the inference thread frees.
        errdefer self.abandon(slot);

        self.queue_mu.lockUncancelable(self.io);
        defer self.queue_mu.unlock(self.io);

        while (self.in_flight >= self.queue_cap and !self.shutdown.load(.acquire)) {
            self.submit_cond.waitUncancelable(self.io, &self.queue_mu);
        }
        if (self.shutdown.load(.acquire)) return error.Shutdown;

        try self.cleanup_queue.ensureUnusedCapacity(self.allocator, self.in_flight + 1);
        try self.prefilling.ensureUnusedCapacity(self.allocator, self.in_flight + 1);
        // Stamp the stable id the `/metrics.json` session rows carry before the
        // slot becomes visible to the inference thread; immutable from here.
        slot.request_id = self.req_seq;
        self.req_seq += 1;

        try self.pending.append(self.allocator, slot);
        self.in_flight += 1;
        self.queue_cond.broadcast(self.io);
        return slot;
    }

    fn abandon(self: *Scheduler, slot: *Slot) void {
        self.queue_mu.lockUncancelable(self.io);
        defer self.queue_mu.unlock(self.io);
        slot.abandoned_next = self.abandoned;
        self.abandoned = slot;
        self.queue_cond.broadcast(self.io);
    }

    fn initSlot(self: *Scheduler, params: SubmitParams) !*Slot {
        if ((params.full_prompt orelse params.prompt_ids).len == 0) return error.EmptyPrompt;
        const slot_config: *const ModelConfig = params.model.config orelse return error.ModelNotReady;
        const eff_kv_quant = params.kv_quant_config orelse
            transformer_mod.KvCacheChoice.resolve(slot_config.kv_quant_override, self.kv_quant_config, self.kv_quant_explicit).config;
        if (slot_config.isGlm5() and eff_kv_quant.glmLatentBits() == null) return error.GlmKvQuantUnsupported;
        return Slot.init(self.allocator, self.io, slot_config, params, eff_kv_quant);
    }

    /// `submit` failed before any slot owned `ve`; only the inference thread may free it. Waits
    /// for room rather than allocating; a shutting-down scheduler drops it with the process.
    pub fn orphanVision(self: *Scheduler, ve: mlx.mlx_array) void {
        self.queue_mu.lockUncancelable(self.io);
        defer self.queue_mu.unlock(self.io);
        while (self.orphan_vision_n == orphan_vision_cap and !self.shutdown.load(.acquire)) {
            self.submit_cond.waitUncancelable(self.io, &self.queue_mu);
        }
        if (self.orphan_vision_n == orphan_vision_cap) return;
        self.orphan_vision[self.orphan_vision_n] = ve;
        self.orphan_vision_n += 1;
        self.queue_cond.broadcast(self.io);
    }

    /// Hand the slot off to the inference thread for cleanup, and notify any
    /// submitter waiting for queue space. Must be called once per slot
    /// returned by `submit`. Safe whether or not the slot finished normally.
    ///
    /// Why not free here: `slot.deinit()` walks the per-slot KVCache and
    /// calls `mlx_array_free` on each entry. Refcount-shared GPU memory
    /// release queues work against the array's owning stream, which is the
    /// inference thread's. Freeing from a conn thread without that stream
    /// binding crashes mlx 0.31.2 ("no Stream(gpu, N) in current thread").
    /// So we remove the slot from any active list (so the inference thread
    /// stops touching it) and queue it for cleanup on the inference thread.
    pub fn complete(self: *Scheduler, slot: *Slot) void {
        // Mark cancelled so any in-flight tick filters this slot out of its
        // active list before we remove it from `decoding`. Idempotent.
        slot.cancelled.store(true, .release);

        self.queue_mu.lockUncancelable(self.io);

        // Remove from pending (rare — only if conn thread cancels before
        // prefill) and from decoding (the common case). After this, only the
        // cleanup queue references the slot, so the next inference-thread
        // cleanup drain can safely deinit it.
        var i: usize = 0;
        while (i < self.pending.items.len) : (i += 1) {
            if (self.pending.items[i] == slot) {
                _ = self.pending.orderedRemove(i);
                break;
            }
        }
        // No new pass can take it (cancelled); wait out the one that has it. It stays in `decoding`
        // meanwhile, so exclusive admission sees it until it reaches the cleanup queue, unless a cull
        // already dropped it, which commits first (`commitsAtCull`).
        waitPassesOut(self.io, &self.queue_mu, &slot.in_pass);
        i = 0;
        while (i < self.decoding.items.len) : (i += 1) {
            if (self.decoding.items[i] == slot) {
                _ = self.decoding.orderedRemove(i);
                break;
            }
        }

        self.cleanup_queue.append(self.allocator, slot) catch {
            // OOM on the cleanup list — fall back to inline deinit. This
            // races on mlx but is strictly better than the leak; the slot
            // is no longer referenced from pending/decoding above.
            self.queue_mu.unlock(self.io);
            slot.deinit();
            self.queue_mu.lockUncancelable(self.io);
        };
        if (self.in_flight > 0) self.in_flight -= 1;
        self.queue_cond.broadcast(self.io); // wake inference thread to drain
        self.submit_cond.broadcast(self.io); // wake any blocked submitter
        self.queue_mu.unlock(self.io);
    }

    /// Wait out the inference pass that holds a cancelled slot, so its statistics stop changing.
    /// The tick filters cancelled slots under `queue_mu`, so no new pass can take it afterwards.
    pub fn quiesce(self: *Scheduler, slot: *Slot) void {
        std.debug.assert(slot.cancelled.load(.acquire));
        self.queue_mu.lockUncancelable(self.io);
        waitPassesOut(self.io, &self.queue_mu, &slot.in_pass);
        self.queue_mu.unlock(self.io);
    }

    /// Shutdown helper: signal every in-flight slot (pending + decoding) to
    /// cancel so their owning connection threads unblock from `waitNext` and
    /// run their `defer complete(...)` promptly. `server.serve` calls this when
    /// the accept loop exits, THEN waits for the connection threads to drain
    /// before returning (which triggers `deinit`) — otherwise a thread still in
    /// `complete()` races `deinit`'s teardown of `pending`/`decoding`/
    /// `cleanup_queue` into a use-after-free. Does NOT remove or free slots;
    /// the conn threads own that via `complete()`.
    pub fn cancelAllInFlight(self: *Scheduler) void {
        self.queue_mu.lockUncancelable(self.io);
        defer self.queue_mu.unlock(self.io);
        for (self.pending.items) |slot| slot.cancelForShutdown();
        for (self.prefilling.items) |slot| slot.cancelForShutdown();
        for (self.decoding.items) |slot| slot.cancelForShutdown();
    }

    /// Plan 05 Phase D: resolve `id_or_empty` ("" / "sushi" → default)
    /// to a refcounted, ready `*LoadedModel`. Cold-loads on demand: if the
    /// entry is `.unloaded`, parses CPU state, picks an LRU victim if
    /// over caps, and posts a `LoadRequest` to the inference thread,
    /// blocking until the load completes.
    ///
    /// Caller MUST call `release(lm)` once done.
    ///
    /// Errors:
    ///   error.UnknownModelId    — id isn't in the registry.
    ///   error.NoDefaultModel    — id empty AND no default set.
    ///   error.NotEnoughMemory   — would exceed caps and no LRU victim.
    ///   error.InsufficientMemory — memory preflight refused the load (free
    ///                             RAM can't hold weights + headroom).
    ///   error.LoadFailed        — inference thread reported a load failure.
    ///   error.Shutdown          — scheduler is shutting down.
    pub fn ensureLoaded(self: *Scheduler, id_or_empty: []const u8) !*LoadedModel {
        last_load_context_refusal = null;
        // Fast path: ready entries. registry.ensureLoaded handles waiting
        // out .loading / .evicting transitions by other callers.
        const fast_result = self.registry.ensureLoaded(id_or_empty);
        if (fast_result) |lm| return lm else |err| switch (err) {
            error.NotLoaded => {}, // fall through to slow path
            else => return err,
        }

        // Slow path: cold load. Resolve the entry. Re-acquire mutex and
        // re-check state — between the fast-path call and now another
        // caller could have completed the load.
        const entry = try self.registry.resolveEntry(id_or_empty);

        // CPU-only pre-load: parse config / load tokenizer / load chat
        // config. Cheap (~tens of ms); kept outside the mutex so other
        // requests on other models stay unblocked. On failure, mark the
        // entry `.error_state` so /v1/models surfaces the failure (and
        // future ensureLoaded calls fail fast instead of re-tripping the
        // same parse error). FileNotFound / parse errors land here.
        const settings = model_settings.overrideFor(self.allocator, self.io, entry.path);
        const cpu_state = preloadCpuState(self.allocator, self.io, entry.path) catch |err| {
            self.registry.mutex.lockUncancelable(self.io);
            self.registry.markErrorLocked(entry, @errorName(err));
            self.registry.mutex.unlock(self.io);
            return error.LoadFailed;
        };
        // Ownership: on success transfers to the entry inside the
        // inference thread's `doLoadOnInferenceThread`; on any error from
        // here on, free them ourselves before returning.
        var owned = cpu_state;
        var owned_active: bool = true;
        defer if (owned_active) freeCpuState(self.allocator, &owned);
        applyModelSettings(owned.config, settings);

        // Victims selected by the eviction planner (multi-victim: one load may
        // need to free several models to fit). Lives on this stack frame; the
        // slice handed to the LoadRequest stays valid while we block on `done`.
        var victims_buf: [16]*LoadedModel = undefined;
        var n_victims: usize = 0;
        var pinned = false;
        errdefer if (pinned) self.registry.releaseStatus(entry);

        const settings_budget = resolveSsdBudget(self.ssd_budget_bytes, owned.config.ssd_budget_gb_override, owned.config.supportsExpertStreaming()).bytes;
        const streaming_gate_bytes: ?u64 = if (expert_stream_mod.expertStreamingEngaged(
            owned.config.supportsExpertStreaming(),
            owned.config.expertStreamingRequired(),
            settings_budget,
        )) blk: {
            if (settings_budget == 0) return error.ExpertStreamingRequired;
            const mtp = mtpChoiceFor(self.mtp_enabled, self.mtp_explicit, owned.config);
            const verdict = mtpStreamingVerdict(mtp, streamedHeadSupportedAt(self.io, self.allocator, owned.config, entry.path));
            switch (verdict) {
                .keep => owned.config.stream_mtp_head = true,
                .refuse => return error.ExpertStreamingMtpUnsupported,
                .drop_settings => owned.config.mtp_override = false,
                .drop_default, .off => {},
            }
            const assistant_bytes = try streamedAssistantBytes(self.io, self.allocator, owned.config, self.no_drafter, coldLoadDrafterDir(self.no_drafter, self.primary_model_dir, self.drafter_dir, entry.path), entry.path);
            const plan = try planExpertStreamingWith(self.io, self.allocator, owned.config, entry.path, settings_budget, coldLoadVision(owned.config, true), assistant_bytes);
            break :blk streamedLoadRequirementBytes(plan.split.trunk +| plan.split.mtp +| plan.split.vision, plan.cache.cache_bytes, loadStagingBytes(owned.config, billedFillPeakBytes(owned.config, plan.cache.prefill_peak_bytes), plan.cache.workspace_bytes), plan.cache.bounce_bytes);
        } else if (owned.config.isGlm5())
            try glmColdLoadBillBytes(self.io, self.allocator, owned.config, entry.path, coldLoadVision(owned.config, false), self.no_drafter, coldLoadDrafterDir(self.no_drafter, self.primary_model_dir, self.drafter_dir, entry.path))
        else if (model_mod.usesSushiQuantMemoryBill(owned.config) or owned.config.usesMimoSourceTrunk())
            try residentColdLoadBillBytes(
                self.io,
                self.allocator,
                owned.config,
                entry.path,
                coldLoadVision(owned.config, false),
                mtpChoiceFor(self.mtp_enabled, self.mtp_explicit, owned.config).on,
                self.no_drafter,
                coldLoadDrafterDir(self.no_drafter, self.primary_model_dir, self.drafter_dir, entry.path),
            )
        else
            null;

        // ── Stage 1 (registry mutex): claim .loading, plan eviction.
        {
            self.registry.mutex.lockUncancelable(self.io);
            errdefer self.registry.mutex.unlock(self.io); // the one unlock of every error return

            // Wait out any concurrent .loading / .evicting state.
            wait_loop: while (true) {
                switch (entry.state) {
                    .ready => {
                        _ = entry.refcount.fetchAdd(1, .acq_rel);
                        self.registry.mutex.unlock(self.io);
                        return entry;
                    },
                    .loading, .evicting => {
                        self.registry.state_cond.waitUncancelable(self.io, &self.registry.mutex);
                        continue :wait_loop;
                    },
                    .error_state => {
                        const load_err = ModelRegistry.loadErrorFromName(entry.error_name);
                        return load_err;
                    },
                    .unloaded => break :wait_loop,
                }
            }

            // Claim the slot.
            std.debug.assert(self.registry.tryBeginLoadLocked(entry));
            pinned = true;

            // Estimate post-load bytes (`gateEstimateBytes`).
            const estimated: u64 = streaming_gate_bytes orelse gateEstimateBytes(entry.bytes_on_disk, owned.config.num_hidden_layers, owned.config.hidden_size);

            // Reserve this load's estimate BEFORE planning eviction, so a
            // concurrent loader sees the pending allocation in its own gate.
            // Without this, two loads can both read a stale resident total
            // (one's bytes not yet committed at markReady), both skip eviction,
            // and oversubscribe GPU memory → Metal OOM → process crash.
            self.registry.reserveLoadLocked(entry, estimated);

            // Evict LRU victims until both caps hold for this reservation
            // (multi-victim). On failure — every other resident model is pinned
            // by an in-flight request — roll back and surface a 503 instead of
            // loading anyway and crashing.
            const n = self.registry.planEvictionsLocked(entry.id, &victims_buf) orelse {
                // Name the numbers. A refusal that logs NOTHING sends the user
                // hunting for a concurrent request that does not exist: on an
                // idle server the cause is always the static cap (#126), and
                // the flag that moves it is not otherwise discoverable.
                const gb = 1024.0 * 1024.0 * 1024.0;
                log.err("Refusing to load {s}: needs ~{d:.2} GB but --max-resident-mem is {d:.2} GB ({d:.2} GB already resident across {d} model(s), none evictable). Raise or disable the cap with --max-resident-mem <size>|0.\n", .{
                    entry.id,
                    @as(f64, @floatFromInt(estimated)) / gb,
                    @as(f64, @floatFromInt(self.registry.max_resident_mem)) / gb,
                    @as(f64, @floatFromInt(self.registry.current_resident_bytes)) / gb,
                    self.registry.countLoadedLocked(),
                });
                self.registry.markUnloadedLocked(entry); // releases the reservation
                return error.NotEnoughMemory;
            };
            n_victims = n;
            // Drain readers on each victim before the inference thread frees it.
            for (victims_buf[0..n_victims]) |v| self.registry.waitForRefcountZeroLocked(v);
            self.registry.mutex.unlock(self.io);
        }

        // ── Stage 2: build + post LoadRequest, wait for completion.
        var req = LoadRequest{
            .entry = entry,
            .config = owned.config,
            .tok = owned.tok,
            .chat_config = owned.chat_config,
            .model_dir = entry.path,
            // Every retained launch setting is re-applied from self.
            // `--no-drafter` / `--drafter` reach cold loads too; the path
            // itself is scoped by `coldLoadDrafterDir`.
            .drafter_dir = coldLoadDrafterDir(self.no_drafter, self.primary_model_dir, self.drafter_dir, entry.path),
            .no_drafter = self.no_drafter,
            .load_vision = coldLoadVision(owned.config, false),
            .warmup_eager = true,
            .kv_quant_config = self.kv_quant_config,
            .kv_quant_explicit = self.kv_quant_explicit,
            // Cold loads get the SAME prefix-cache configuration as the
            // startup model — pre-plumbing these were (1, 0, stride 0),
            // which silently degraded warm reuse after every model switch.
            .prefix_cache_capacity = self.prefix_cache_capacity,
            .prefix_cache_ram_enabled = self.prefix_cache_ram_enabled,
            .prefix_cache_mem_bytes = self.prefix_cache_mem_bytes,
            .prefix_cache_mem_resolver = self.prefix_cache_mem_resolver,
            .prefix_cache_disk_bytes = self.prefix_cache_disk_bytes,
            .prefix_cache_disk_resolver = self.prefix_cache_disk_resolver,
            .ssd_budget_bytes = self.ssd_budget_bytes,
            .expert_cache_fit_resolver = self.expert_cache_fit_resolver,
            .ssm_checkpoint_stride = self.ssm_checkpoint_stride,
            .ssm_checkpoint_max = self.ssm_checkpoint_max,
            // Cold loads honor the launch-flag MTP settings too (same reason
            // as prefix-cache above) — pre-plumbing these were LoadRequest
            // defaults, so --no-mtp / --mtp-max-depth were silently dropped on
            // every on-demand load and model switch.
            .mtp_enabled = self.mtp_enabled,
            .mtp_explicit = self.mtp_explicit,
            .mtp_head_kv_quant = self.mtp_head_kv_quant,
            .mtp_depth = self.mtp_depth,
            .evict_entries = victims_buf[0..n_victims],
            .allocator = self.allocator,
        };

        {
            self.queue_mu.lockUncancelable(self.io);
            defer self.queue_mu.unlock(self.io);
            if (self.shutdown.load(.acquire)) return error.Shutdown;
            try self.load_queue.append(self.allocator, &req);
            self.queue_cond.broadcast(self.io);
        }

        // Block until the inference thread signals done.
        req.done_mu.lockUncancelable(self.io);
        while (!req.done) req.done_cond.waitUncancelable(self.io, &req.done_mu);
        req.done_mu.unlock(self.io);

        if (req.error_name) |name| {
            last_load_context_refusal = req.memory_context_refusal;
            // The failure crosses the thread boundary by NAME — map it back
            // to a typed error so a memory-preflight refusal surfaces as a
            // named 503, not the generic "Model load failed" 500 (#144).
            defer self.allocator.free(name);
            // On success the inference thread took ownership of cpu_state;
            // on failure it didn't, so we still hold it.
            return ModelRegistry.loadErrorFromName(name);
        }

        // Success — inference thread installed cpu_state onto the entry.
        owned_active = false;

        // Already referenced since the claim, so no other load evicted it; only an explicit
        // unload can have claimed it since `markReady`, and that is a 503, not a failed load.
        self.registry.mutex.lockUncancelable(self.io);
        const ready = entry.state == .ready;
        self.registry.mutex.unlock(self.io);
        if (!ready) return error.NotLoaded;
        return entry;
    }

    /// Release a borrowed pointer obtained from `ensureLoaded`. Forwards
    /// to the registry so the refcount decrement + LRU clock bump happen
    /// under registry.mutex.
    pub fn release(self: *Scheduler, lm: *LoadedModel) void {
        self.registry.release(lm);
    }

    /// `release` for a status read — same refcount protocol, no recency
    /// stamp. See `ModelRegistry.releaseStatus`.
    pub fn releaseStatus(self: *Scheduler, lm: *LoadedModel) void {
        self.registry.releaseStatus(lm);
    }

    /// Duped stored failure name for the id `ensureLoaded` just refused with
    /// `error.LoadFailed` — feeds the "Model load failed: <name>" HTTP
    /// message (#144). Caller frees.
    pub fn loadErrorName(self: *Scheduler, alloc: std.mem.Allocator, id_or_empty: []const u8) ?[]u8 {
        return self.registry.loadErrorNameDupe(alloc, id_or_empty);
    }

    pub fn takeLoadContextRefusal(self: *Scheduler, id: []const u8) ?model_registry_mod.MemoryContextRefusal {
        const context = last_load_context_refusal;
        last_load_context_refusal = null;
        return context orelse self.registry.loadContextRefusal(id);
    }

    /// Free a model's resident GPU state, returning its registry stub to
    /// `.unloaded` so it can reload later. Idempotent — a non-resident model
    /// returns immediately. Marks the entry `.evicting`, drains in-flight
    /// requests (refcount → 0), then hands the mlx free to the inference
    /// thread (stream-bound). Blocks until the free completes.
    pub fn unloadModel(self: *Scheduler, id_or_empty: []const u8) !void {
        return self.unloadModelIfIdle(id_or_empty, null);
    }

    /// `unloadModel` with an optional idle precondition, re-checked under the
    /// registry mutex at the moment we commit.
    ///
    /// The sweep picks a victim, drops the mutex, then calls in, so a request
    /// can arrive in that gap: refcount and age are re-checked under the mutex
    /// before committing. (`planEvictionsLocked` picks and marks under one
    /// hold instead; the sweep cannot, because the unload itself is slow.)
    pub fn unloadModelIfIdle(self: *Scheduler, id_or_empty: []const u8, idle_window_ms: ?i64) !void {
        const entry = try self.registry.resolveEntry(id_or_empty);
        {
            self.registry.mutex.lockUncancelable(self.io);
            wait: while (true) {
                switch (entry.state) {
                    // Already free (or failed-load stub) → nothing to do.
                    .unloaded, .error_state => {
                        self.registry.mutex.unlock(self.io);
                        return;
                    },
                    // Another caller is mid load/evict — wait it out, then
                    // re-check (it may end up resident or unloaded).
                    .loading, .evicting => {
                        self.registry.state_cond.waitUncancelable(self.io, &self.registry.mutex);
                        continue :wait;
                    },
                    .ready => break :wait,
                }
            }
            if (idle_window_ms) |window_ms| {
                const now_ms = io_util.nowMsMonotonic(self.io);
                if (!model_registry_mod.ModelRegistry.idleEvictable(entry, now_ms, window_ms)) {
                    self.registry.mutex.unlock(self.io);
                    return;
                }
            }
            self.registry.markEvictingLocked(entry);
            self.registry.waitForRefcountZeroLocked(entry);
            self.registry.mutex.unlock(self.io);
        }

        var req = UnloadRequest{ .entry = entry };
        {
            self.queue_mu.lockUncancelable(self.io);
            defer self.queue_mu.unlock(self.io);
            // On shutdown the entry stays `.evicting`; `Scheduler.deinit`
            // unloads every `.ready`/`.evicting` entry, so it's still freed.
            if (self.shutdown.load(.acquire)) return error.Shutdown;
            try self.unload_queue.append(self.allocator, &req);
            self.queue_cond.broadcast(self.io);
        }
        req.done_mu.lockUncancelable(self.io);
        while (!req.done) req.done_cond.waitUncancelable(self.io, &req.done_mu);
        req.done_mu.unlock(self.io);
    }

    pub const UnloadedIdle = struct { id: []const u8, bytes: u64 };

    /// Unloads the least recently used resident model nobody holds, other than `keep`, on the
    /// inference thread. Null when none was idle or a request took it before the commit.
    pub fn unloadLruIdleModel(self: *Scheduler, keep: []const u8) ?UnloadedIdle {
        self.registry.mutex.lockUncancelable(self.io);
        const entry = self.registry.pickLruEvictable(keep) orelse {
            self.registry.mutex.unlock(self.io);
            return null;
        };
        const bytes = entry.bytes_resident;
        self.registry.mutex.unlock(self.io);

        self.unloadModelIfIdle(entry.id, 0) catch return null;

        self.registry.mutex.lockUncancelable(self.io);
        defer self.registry.mutex.unlock(self.io);
        if (entry.state == .ready) return null;
        return .{ .id = entry.id, .bytes = bytes };
    }

    /// Phase A4: synchronously encode one or more images and return the
    /// embedding tensor. Conn thread fills `req.images` (CHW float32 pixel
    /// buffers, decoded by stb_image / libwebp on the conn thread); this
    /// method posts the request to the inference thread, blocks until done,
    /// and returns the resulting `mlx_array` on success. Ownership of the
    /// returned array transfers to the caller — typically passed straight
    /// into `submit(.{ .vision_embeddings = arr, ... })` so the slot owns
    /// it and frees on `deinit`.
    ///
    /// Returns `error.VisionEncodeFailed` if the inference thread fails.
    /// The request struct must outlive this call, but since the call blocks,
    /// a stack allocation in the caller works.
    pub fn encodeVision(self: *Scheduler, req: *VisionEncodeRequest) !mlx.mlx_array {
        // Post + wake.
        self.queue_mu.lockUncancelable(self.io);
        self.vision_queue.append(self.allocator, req) catch |err| {
            self.queue_mu.unlock(self.io);
            return err;
        };
        self.queue_cond.broadcast(self.io);
        self.queue_mu.unlock(self.io);

        // Wait for completion.
        req.done_mu.lockUncancelable(self.io);
        defer req.done_mu.unlock(self.io);
        while (!req.done) {
            req.done_cond.waitUncancelable(self.io, &req.done_mu);
        }
        if (req.error_name) |_| return error.VisionEncodeFailed;
        return req.result orelse error.VisionEncodeFailed;
    }

    /// Does this slot's next decode tick actually run the regular (non-speculative)
    /// path? The slot's `enable_*` flags carry the REQUEST's wish; `specTickMode` is
    /// the authoritative dispatch answer, and a generator can also have turned spec
    /// off at runtime. Same class as "a guard that shapes INIT options does not bind DISPATCH".
    fn slotTicksRegular(slot: *const Slot) bool {
        if (Planner.enabled() and slot.planner_force_plain) return true;
        const gen = if (slot.legacy_gen) |*g| g else return !(slot.enable_drafter or slot.enable_mtp);
        // A runtime-disabled generator is already ticking regular, so batching it
        // dispatches what it was going to dispatch anyway.
        return gen.spec_disabled_runtime or specTickMode(
            slot.enable_mtp,
            gen.mtp != null or gen.dspark,
            slot.enable_drafter,
            gen.dflash != null,
        ) == .regular;
    }

    /// Does this slot owe a module-head release? One single-slot tick lands it.
    fn slotReleasePending(slot: *const Slot) bool {
        const gen = if (slot.legacy_gen) |*g| g else return false;
        return gen.mtpReleasePending();
    }

    /// Active-tick gate. Decides whether a slot is eligible for the batched
    /// decode kernel. Hybrid SSM / MoE / encoder models can't ride
    /// the batched kernel (it doesn't model their state), so any slot
    /// targeting such a model falls through to the single-slot path. Phase
    /// D: the gate reads off the slot's own model config — multi-model
    /// means the scheduler's startup config is no longer authoritative.
    fn batchable(self: *const Scheduler, slot: *const Slot) bool {
        return self.batchVerdict(slot) == .ok;
    }

    /// Why a slot does or does not ride the batched kernel. `.ok` batches; every
    /// other arm decodes serial this tick and is what the `[batched] serial` line
    /// and `decode_serial_total{reason}` name.
    pub fn batchVerdict(self: *const Scheduler, slot: *const Slot) BatchVerdict {
        _ = self;
        // The gate reads dispatch via slotTicksRegular, never the armed enable_drafter or enable_mtp flags.
        if (!slotTicksRegular(slot)) return .spec_active;
        // A slot whose module-head release is armed but not landed still holds the head.
        if (slotReleasePending(slot)) return .head_release_pending;
        if (slot.sampling.constraint != null) return .grammar;
        // The batched tick samples every slot's successor; a forced call decides its own.
        if (slot.sampling.call_force) |cf| {
            if (cf.pending()) return .forced_call;
        }
        if (slot.logprobs_n > 0) return .logprobs;
        if (sampler.penaltyActive(slot.sampling)) return .penalty;
        // Ask the transformer, never name the arch here — same rule as `modelExclusiveDecode`.
        const t = slot.model.transformer orelse return .arch;
        return if (t.supportsBatchedGdnDecode() or t.supportsBatchedMimoDecode() or t.supportsBatchedGlmRows()) .ok else .arch;
    }
};

var kv_skew_split_logged: bool = false; // one-shot log guard

const ThreadCtx = struct {
    scheduler: *Scheduler,
    params: LoadParams,
};

/// Set by `--skip-mem-preflight` (main.zig) to bypass the model-load memory
/// pre-flight below. A module global, not a `LoadParams` field, so it applies
/// uniformly to startup loads AND later hot-loads — matching the env var
/// (`SUSHI_SKIP_MEM_PREFLIGHT`) it replaced.
pub var skip_mem_preflight: bool = false;

pub var load_serving_bill: ?*const fn (*const ModelConfig, u64) ?LoadServingBill = null;

test "a loaded model's hot-cache budget is the resolver's clamp, never the raw launch flag" {
    const Stub = struct {
        fn clamp(_: *model_mod.ModelConfig, requested: u64, _: BudgetRevise, idle_out: *u64) u64 {
            idle_out.* = 7;
            return requested / 4;
        }
    };
    var cfg = model_mod.ModelConfig{ .model_type = "qwen4_exp" };
    const forty_gb: u64 = 40 << 30;
    var idle: u64 = 0;
    try testing.expectEqual(forty_gb / 4, hotCacheMemForLoad(true, Stub.clamp, &cfg, forty_gb, &idle));
    try testing.expectEqual(@as(u64, 7), idle);
    try testing.expectEqual(@as(u64, 0), hotCacheMemForLoad(false, Stub.clamp, &cfg, forty_gb, &idle));
    try testing.expectEqual(forty_gb, hotCacheMemForLoad(true, null, &cfg, forty_gb, &idle));
}

test "a load bills weights, then the larger of the measured warmup plus a 1 GiB net and the context bill" {
    const gib: u64 = 1 << 30;
    var f = try LoadBillFixtures.init();
    defer f.deinit();
    // Flash-Next's MTP head load peaks above its resident bytes, so only an MTP-on load pays for it.
    try testing.expectEqual(44 * gib + gib / 8 + gib, loadRequirementBytes(&f.qwen, 44 * gib, false, 0));
    try testing.expectEqual(44 * gib + gib * 5 / 4 + gib, loadRequirementBytes(&f.qwen, 44 * gib, true, 0));
    try testing.expectEqual(90 * gib + gib / 4 + gib, loadRequirementBytes(&f.mimo, 90 * gib, true, 0));
    const glm_state = f.glm.ssmCheckpointBytes() + f.glm.qsaRingBytes() + (f.glm.kvBytesPerToken() + f.glm.qsaHistoryBytesPerToken()) * 1024;
    try testing.expect(glm_state > 0);
    try testing.expectEqual(90 * gib + glm_state + gib * 3 / 8 + gib, loadRequirementBytes(&f.glm, 90 * gib, false, 0));
    // A requested context's admission bill replaces warmup and net once it is the larger: no request has allocated
    // during the warmup, and request admission bills that context again.
    try testing.expectEqual(44 * gib + 6 * gib, loadRequirementBytes(&f.qwen, 44 * gib, true, 6 * gib));
    try testing.expectEqual(44 * gib + gib * 5 / 4 + gib, loadRequirementBytes(&f.qwen, 44 * gib, true, gib));
    try testing.expectEqual(44 * gib + gib * 9 / 4, loadRequirementBytes(&f.qwen, 44 * gib, true, gib * 9 / 4));
}

test "a streamed Flash-Next load bills no expert fill peak; MiMo and GLM keep theirs" {
    const fill: u64 = 1_848_000_000;
    const qwen = ModelConfig{ .model_type = "qwen4_exp", .expert_streaming = true };
    const mimo = ModelConfig{ .model_type = "mimo_v2", .expert_streaming = true };
    const glm = ModelConfig{ .model_type = "glm5_next", .expert_streaming = true };
    try testing.expectEqual(@as(u64, 0), billedFillPeakBytes(&qwen, fill));
    try testing.expectEqual(fill, billedFillPeakBytes(&mimo, fill));
    try testing.expectEqual(fill, billedFillPeakBytes(&glm, fill));
}

/// Republish the hot cache's residency for the connection thread's admission guard.
/// Called from the inference thread after a commit, eviction, invalidation or model switch.
const BUDGET_REVISE_WINDOW_NS: u64 = 10 * std.time.ns_per_s;

/// How a budget resolve differs from the load-time one: the cache's own resident bytes
/// are excluded from the machine read, and the resolver keeps quiet (`setBudget` logs).
pub const BudgetRevise = struct { exclude_bytes: u64 = 0, quiet: bool = false };

fn takeOwned(comptime T: type, held: *?T) ?T {
    defer held.* = null;
    return held.*;
}

pub fn hotCacheMemForLoad(
    ram_prefix_cache: bool,
    resolver: ?*const fn (*model_mod.ModelConfig, u64, BudgetRevise, *u64) u64,
    config: *model_mod.ModelConfig,
    requested: u64,
    ssd_idle_out: *u64,
) u64 {
    if (!ram_prefix_cache) return 0;
    const resolve = resolver orelse return requested;
    return resolve(config, requested, .{}, ssd_idle_out);
}

/// Re-clamp every resident model's hot-cache budget after residency changed (#364): the
/// load-time clamp read the machine with the other models on it and was never revisited, so
/// a model loaded beside a large one kept a ~0 budget for life. Each cache's own resident
/// entries are excluded from the read so a full cache cannot ratchet itself down.
fn reviseHotCacheBudgets(sch: *Scheduler) void {
    const resolve = sch.prefix_cache_mem_resolver orelse return;
    sch.registry.mutex.lockUncancelable(sch.io);
    // The resolver publishes the process-global budget the admission guard reads, so the
    // current model goes last.
    for ([_]bool{ false, true }) |current_pass| {
        var it = sch.registry.entries.valueIterator();
        while (it.next()) |entry_ptr| {
            const entry = entry_ptr.*;
            if ((entry == sch.current_model) != current_pass) continue;
            if (entry.state != .ready) continue;
            const hc = if (entry.prefix_cache) |*h| h else continue;
            if (!hc.ram_enabled) continue;
            const config = entry.config orelse continue;
            var idle: u64 = 0;
            hc.setBudget(resolve(config, sch.prefix_cache_mem_bytes, .{ .exclude_bytes = hc.residentBytes(), .quiet = true }, &idle));
            hc.ssd_idle_mem = idle;
        }
    }
    sch.registry.mutex.unlock(sch.io);
    publishHotCacheResidency(sch);
}

pub fn publishHotCacheResidency(sch: *Scheduler) void {
    const bytes: u64 = if (sch.hot_prefix_cache) |hc| hc.residentBytes() else 0;
    sch.resident_hot_cache_bytes.store(bytes, .monotonic);
    const reclaimable: u64 = if (sch.hot_prefix_cache) |hc| hc.reclaimableBytes() else 0;
    sch.reclaimable_hot_cache_bytes.store(reclaimable, .monotonic);
    publishDiskStats(sch);
    publishHotCacheDigests(sch);
    if (sch.metrics != null) publishCachedSessions(sch);
}

/// Caller must not hold `registry.mutex`.
fn publishCachedSessions(sch: *Scheduler) void {
    var rows: [metrics_mod.MAX_SESSIONS]metrics_mod.Session = undefined;
    var n: usize = 0;
    {
        sch.registry.mutex.lockUncancelable(sch.io);
        defer sch.registry.mutex.unlock(sch.io);
        var it = sch.registry.entries.valueIterator();
        outer: while (it.next()) |entry_ptr| {
            const entry = entry_ptr.*;
            if (entry.state != .ready) continue;
            const hc = if (entry.prefix_cache) |*h| h else continue;
            for (hc.entries.items) |*e| {
                if (n == rows.len) break :outer;
                const len: u32 = @intCast(@min(e.tokens.len, std.math.maxInt(u32)));
                rows[n] = .init(entry.id, .cached, len, len, 0, e.kv_bytes);
                rows[n].entry_id = e.id;
                n += 1;
            }
        }
    }
    sch.digest_mu.lockUncancelable(sch.io);
    defer sch.digest_mu.unlock(sch.io);
    @memcpy(sch.cached_sessions[0..n], rows[0..n]);
    sch.cached_session_count = n;
}

/// Swap in a fresh digest snapshot and free the one it supersedes (inference thread only).
/// An allocation failure keeps the previous snapshot: a stale digest is a hint, an empty one
/// credits bytes that exist.
fn publishHotCacheDigests(sch: *Scheduler) void {
    const fresh: []prefix_cache_mod.HotPrefixCache.EntryDigest = if (sch.hot_prefix_cache) |hc|
        (hc.digestsAlloc(sch.allocator) catch return)
    else
        &.{};
    const residency: u64 = if (sch.hot_prefix_cache) |hc| hc.residentBytes() else 0;
    sch.digest_mu.lockUncancelable(sch.io);
    const old = sch.hot_cache_digests;
    sch.hot_cache_digests = fresh;
    // Published under the same lock as the digests it describes.
    sch.digest_residency = residency;
    sch.digest_mu.unlock(sch.io);
    if (old.len > 0) sch.allocator.free(old);
}

/// Connection-thread entry point: what an eviction pass can prove it will get back for this prompt.
pub fn reclaimableHotCacheBytesFor(sch: *Scheduler, prompt_tokens: []const u32) u64 {
    const fp = prefix_cache_mod.HotPrefixCache.prefixFingerprint(prompt_tokens);
    sch.digest_mu.lockUncancelable(sch.io);
    defer sch.digest_mu.unlock(sch.io);
    return prefix_cache_mod.HotPrefixCache.reclaimableFromDigests(
        sch.hot_cache_digests,
        sch.digest_residency,
        fp,
    );
}

/// What one inference pass takes under `queue_mu` for teardown and model jobs.
const TeardownTake = struct {
    cleanup_n: usize,
    abandoned: ?*Slot,
    load: ?*LoadRequest,
    unload: ?*UnloadRequest,
};

/// Caller holds `queue_mu`.
pub fn takeTeardownLocked(sch: *Scheduler, batch: []*Slot) TeardownTake {
    var n: usize = 0;
    while (n < batch.len and sch.cleanup_queue.items.len > 0) {
        batch[n] = sch.cleanup_queue.orderedRemove(0);
        n += 1;
    }
    const abandoned = sch.abandoned;
    sch.abandoned = null;
    // A queued slot's generator points into its model's transformer, and its connection released
    // the model before the job was queued: the job waits until the whole queue is destroyed.
    const settled = sch.cleanup_queue.items.len == 0;
    return .{
        .cleanup_n = n,
        .abandoned = abandoned,
        .load = if (settled and sch.load_queue.items.len > 0) sch.load_queue.orderedRemove(0) else null,
        .unload = if (settled and sch.unload_queue.items.len > 0) sch.unload_queue.orderedRemove(0) else null,
    };
}

pub fn destroyAbandoned(head: ?*Slot) void {
    var next = head;
    while (next) |slot| {
        next = slot.abandoned_next;
        slot.deinit();
    }
}

/// Caller holds `queue_mu`; wakes a submit waiting for room in `orphan_vision`.
pub fn takeOrphanedVisionLocked(sch: *Scheduler, out: *[orphan_vision_cap]mlx.mlx_array) usize {
    const n = sch.orphan_vision_n;
    if (n == 0) return 0;
    @memcpy(out[0..n], sch.orphan_vision[0..n]);
    sch.orphan_vision_n = 0;
    sch.submit_cond.broadcast(sch.io);
    return n;
}

/// Caller holds `queue_mu`. Shared with the wait condition below.
fn hasWorkPendingLocked(sch: *const Scheduler) bool {
    return sch.pending.items.len > 0 or
        sch.decoding.items.len > 0 or
        sch.vision_queue.items.len > 0 or
        sch.cleanup_queue.items.len > 0 or
        sch.abandoned != null or
        sch.orphan_vision_n > 0 or
        sch.load_queue.items.len > 0 or
        sch.unload_queue.items.len > 0;
}

/// `--gpu-warm-secs` (0 = off): how long after its last prefill or decode the idle inference
/// thread keeps the GPU awake. Left idle, the driver drops the resident set's state, and the next
/// submission then waits before any work runs (docs/server-lifecycle.md#threads).
pub var gpu_warm_secs: u32 = 60;
/// Shorter than the idle time after which the state is gone.
pub const GPU_WARM_TICK_NS: u64 = 500 * std.time.ns_per_ms;

/// PURE: whether an idle thread ticks now. `idle_ns` is null until the thread has run a
/// prefill or decode; a thread with work never ticks.
pub fn gpuWarmTickDue(has_work: bool, idle_ns: ?u64, window_ns: u64) bool {
    const idle = idle_ns orelse return false;
    return !has_work and idle < window_ns;
}

/// PURE: how long an idle thread parks before its next tick, or null to park until work arrives.
pub fn gpuWarmParkNs(idle_ns: ?u64, window_ns: u64) ?u64 {
    const idle = idle_ns orelse return null;
    if (idle >= window_ns) return null;
    return @min(GPU_WARM_TICK_NS, window_ns - idle);
}

pub const GpuWarmWindow = enum { keep, restart, close };

/// PURE: the window as the thread is about to park. The last pass's prefill or decode restarts it;
/// an unload closes it, since no model may be left to keep warm.
pub fn gpuWarmBeforePark(worked: bool, unloaded: bool) GpuWarmWindow {
    if (unloaded) return .close;
    return if (worked) .restart else .keep;
}

/// One synced element-op on the GPU stream. Inference thread only; a failure stays unlatched.
fn gpuWarmTick() void {
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const one = mlx.mlx_array_new_float(1);
    defer _ = mlx.mlx_array_free(one);
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    if (mlx.mlx_multiply(&out, one, one, s) != 0) return;
    _ = mlx.mlx_array_eval(out);
}

/// Poll interval for the idle-eviction sweep, given the configured window.
///
/// A quarter of the window, so a model is evicted within ~1.25x the configured
/// idle time rather than up to 2x it. Floored at a second so `--idle-evict-secs 1`
/// does not spin, capped at 30s so a long window still costs ~nothing.
pub fn idleEvictTickMs(window_ms: i64) i64 {
    return @max(1000, @min(@divTrunc(window_ms, 4), 30_000));
}

fn inferenceLoop(ctx: ThreadCtx) void {
    const sch = ctx.scheduler;
    const params = ctx.params;
    // Covers shutdown and startup-load failure.
    defer sleep_inhibit.release();

    // ── Phase A1 → Plan 05: load runs on this thread (mlx GPU stream
    //    binding). On failure, mark the entry `.error_state` in the
    //    registry AND set load_failed so `Scheduler.init`'s parent sees
    //    the same shape it always has.
    //
    // Headless boot (`no_initial_load`): start with NO primary model. The
    // server runs idle until a chat or media model is loaded on demand via
    // `/v1/load-model` (or a request targeting a discovered id). Used by the
    // app's "start headless, load gen on demand" flow.
    if (params.no_initial_load) {
        log.info("Headless: no primary model loaded; models load on demand.\n", .{});
        signalStarted(sch);
    } else {
        // Startup load runs before the wait loop can acquire.
        sleep_inhibit.setActive(true);
        if (doLoadOnInferenceThread(sch, params)) |_| {
            log.info("Model ready (loaded on inference thread).\n", .{});
            signalStarted(sch);
        } else |err| {
            recordLoadError(sch, @errorName(err));
            sch.registry.mutex.lockUncancelable(sch.io);
            sch.registry.markErrorLocked(params.entry, @errorName(err));
            sch.registry.mutex.unlock(sch.io);
            signalStarted(sch);
            return;
        }
    }

    // The GPU warm window (`--gpu-warm-secs`): open since the last pass that ran a prefill or a
    // decode tick (`worked`).
    var warm_since: ?io_util.Stopwatch = null;
    var warm_ticks: u32 = 0;
    var worked = false;
    while (!sch.shutdown.load(.acquire)) {
        // 0a. Drain slots queued for cleanup. Conn threads hand finished
        //     slots here in `complete()` — we own the mlx stream binding,
        //     so freeing per-slot KVCache + vision_embeddings + ssm_entries
        //     is safe here even though those slots' arrays might trigger
        //     real GPU memory release on refcount-zero.
        var cleanup_batch: [16]*Slot = undefined;
        var cleanup_n: usize = 0;
        var abandoned: ?*Slot = null;
        var orphan_vision: [orphan_vision_cap]mlx.mlx_array = undefined;
        var orphan_vision_n: usize = 0;
        // 0b. Drain any pending vision work. It runs synchronously on behalf of
        //     conn threads waiting in `encodeVision`. Processed here (not
        //     concurrently with decode ticks) so it shares the inference
        //     thread's mlx stream cleanly.
        var vision_batch: [4]*VisionEncodeRequest = undefined;
        var vision_n: usize = 0;
        // Phase D: cold-load drain. Process ONE load per tick — loading a
        // model is heavy (~seconds; weight read + JIT compile + warmup)
        // and we want the rest of the inference loop to stay responsive.
        // Other pending loads wait in queue and get picked up next tick.
        var load_req: ?*LoadRequest = null;
        // Unload work item (one per tick, like load — heavy, and we re-check
        // the loop between them).
        var unload_req: ?*UnloadRequest = null;
        {
            sch.queue_mu.lockUncancelable(sch.io);
            defer sch.queue_mu.unlock(sch.io);
            const teardown = takeTeardownLocked(sch, &cleanup_batch);
            cleanup_n = teardown.cleanup_n;
            abandoned = teardown.abandoned;
            orphan_vision_n = takeOrphanedVisionLocked(sch, &orphan_vision);
            while (vision_n < vision_batch.len and sch.vision_queue.items.len > 0) {
                vision_batch[vision_n] = sch.vision_queue.orderedRemove(0);
                vision_n += 1;
            }
            load_req = teardown.load;
            unload_req = teardown.unload;
        }
        for (cleanup_batch[0..cleanup_n]) |s| {
            // Decode-phase cancel: `complete()` pulled this slot straight
            // into the cleanup queue, so it never went through finishSlot
            // and its committed KV (prompt + every emitted token) would die
            // right here with the slot. Commit it first — the same guards
            // as a normal finish apply inside (pad-only / error / vision /
            // empty all decline) — then flush what was committed to the
            // SSD tier, since no finishSlot will. Normally-finished slots
            // arrive here with `finished` already set (finishSlot committed
            // them) and skip; errored slots decline via the error guard.
            // Runs on the inference thread — the sole mlx caller — which is
            // what makes the refcount-sharing snapshot legal here.
            if (s.cancelled.load(.acquire) and !s.finished and s.error_code == null) {
                // Streams end this way on a stop sequence or a client that leaves: finishSlot's line is theirs too.
                if (s.legacy_gen) |*g| {
                    g.logSpecStats();
                    g.logQsaArms();
                }
                commitCancelledAtCleanup(sch, s, commitSlotIfApplicable);
                if (s.model.prefix_cache) |*hc| {
                    if (s.model.transformer) |xf| hc.flushPendingDisk(xf.s);
                }
            }
            recordSlotCleanup(sch.metrics, s);
            // Second slot-end path (a decode-phase cancel never reaches finishSlot); the
            // record must not outlive the bytes `s.deinit()` frees.
            if (s.model.prefix_cache) |*hc| hc.releaseCheckout(@intFromPtr(s), "slot cleanup");
            if (s.model.transformer) |xfm| xfm.resetQsaPooledRope();
            if (s.model.config) |c| @import("dsv41/dsv41_borrow.zig").endRequestMlx(c);
        }
        deinitSlotsReturningPool(cleanup_batch[0..cleanup_n]);
        destroyAbandoned(abandoned);
        for (orphan_vision[0..orphan_vision_n]) |ve| freeVisionArray(ve);
        for (vision_batch[0..vision_n]) |req| runVisionEncode(sch, req);
        if (load_req) |req| runLoadRequest(sch, req);
        if (unload_req) |req| runUnloadRequest(sch, req);
        if (load_req != null or unload_req != null) reviseHotCacheBudgets(sch);
        switch (gpuWarmBeforePark(worked, unload_req != null)) {
            .keep => {},
            .restart => warm_since = io_util.Stopwatch.init(sch.io),
            .close => warm_since = null,
        }
        if (worked or unload_req != null) warm_ticks = 0;
        worked = false;
        if (unload_req != null) sch.budget_revise_sw = io_util.Stopwatch.init(sch.io);

        // 1. Wait for work. Drain pending slots into a local list under lock,
        //    run prefills outside the lock.
        var to_prefill: [16]*Slot = undefined;
        var n_prefill: usize = 0;
        {
            sch.queue_mu.lockUncancelable(sch.io);
            defer sch.queue_mu.unlock(sch.io);
            const warm_ns: u64 = @as(u64, gpu_warm_secs) * std.time.ns_per_s;
            while (!hasWorkPendingLocked(sch) and !sch.shutdown.load(.acquire)) {
                // No later tick runs while parked, so release here.
                sleep_inhibit.setActive(false);
                if (gpuWarmParkNs(if (warm_since) |sw| sw.read() else null, warm_ns)) |park_ns| {
                    const timed_out = if (sch.queue_cond.waitTimeout(sch.io, &sch.queue_mu, .{ .duration = .{
                        .raw = .fromNanoseconds(@intCast(park_ns)),
                        .clock = .awake,
                    } })) |_| false else |err| err == error.Timeout;
                    if (timed_out and gpuWarmTickDue(hasWorkPendingLocked(sch), if (warm_since) |sw| sw.read() else null, warm_ns)) {
                        if (warm_ticks == 0) log.debug("[gpu-warm] keeping the GPU awake for {d} s\n", .{gpu_warm_secs});
                        warm_ticks += 1;
                        sch.queue_mu.unlock(sch.io);
                        gpuWarmTick();
                        sch.queue_mu.lockUncancelable(sch.io);
                    }
                } else {
                    if (warm_ticks > 0) log.debug("[gpu-warm] idle window over after {d} ticks\n", .{warm_ticks});
                    warm_since = null;
                    warm_ticks = 0;
                    sch.queue_cond.waitUncancelable(sch.io, &sch.queue_mu);
                }
            }
            if (sch.shutdown.load(.acquire)) break;
            // POST /metrics/enable publishes the sink from another thread; only this thread ever stores `sch.metrics`.
            sch.metrics = metrics_mod.adoptPublished(sch.metrics, metrics_mod.current());
            // Hold until the loop parks again.
            sleep_inhibit.setActive(true);

            // If only vision/embed/cleanup/load work is pending, loop back to drain it.
            if (sch.pending.items.len == 0 and sch.decoding.items.len == 0) continue;

            // Single-flight admission: a model with
            // MODULE-OWNED decode state admits at most one live slot.
            // Snapshot live exclusive-model slots (same liveness predicate
            // as the step-3 active list), let `admitPendingTick` decide,
            // and leave held slots in `pending` — their conn threads keep
            // flowing SSE keepalives while they wait, and the wait
            // condition above never blocks while `pending` is non-empty,
            // so a held slot admits on the first tick after the active one
            // is culled (step 5, same mutex).
            var live_buf: [32]AdmitCand = undefined;
            const n_live = heldExclusive(&live_buf, &.{ sch.decoding.items, sch.cleanup_queue.items }, slotExclusiveDecode);
            var cand_buf: [32]AdmitCand = undefined;
            const n_cands = @min(sch.pending.items.len, cand_buf.len);
            for (sch.pending.items[0..n_cands], 0..) |s, i| {
                cand_buf[i] = .{ .model = @intFromPtr(s.model), .exclusive = slotExclusiveDecode(s) };
            }
            var admit_idx: [to_prefill.len]usize = undefined;
            // Pending admits only through admitPendingTick.
            const n_admit = memoryAdmitCount(sch, admit_idx[0..admitPendingTick(cand_buf[0..n_cands], live_buf[0..n_live], &admit_idx)]);
            for (admit_idx[0..n_admit]) |idx| {
                to_prefill[n_prefill] = admitForPrefillLocked(sch, idx);
                n_prefill += 1;
            }
            // Remove admitted entries in DESCENDING index order so the
            // earlier (ascending) indices stay valid during removal.
            var r = n_admit;
            while (r > 0) {
                r -= 1;
                _ = sch.pending.orderedRemove(admit_idx[r]);
            }
            var queue_line: [192]u8 = undefined;
            for (sch.pending.items) |held| {
                if (queueLogLine(sch, &queue_line, held)) |line| log.info("{s}\n", .{line});
            }
        }

        if (sch.budget_revise_sw) |sw| {
            if (sw.read() > BUDGET_REVISE_WINDOW_NS) sch.budget_revise_sw = null;
            if (n_prefill > 0) reviseHotCacheBudgets(sch);
        }

        // 2. Prefill each pending slot (heavy; mlx ops on this thread).
        //    The inference thread is the sole mlx caller post-cleanup, so
        //    no per-tick stream rebind / mutex coexistence is needed.
        if (n_prefill > 0) {
            for (to_prefill[0..n_prefill], 0..) |slot, pi| {
                defer endPrefillPass(sch, slot);
                // Between the slots of one admitted batch, tick the streams
                // that just started decoding — a single-chunk prefill exposes
                // no chunk-boundary yield, so without this every slot's first
                // token waits for the LAST slot's prefill (the TTFT
                // staircase collapse).
                if (pi > 0 and prefillInterleaveEnabled()) _ = interleaveDecodeTick(sch);
                if (slot.cancelled.load(.acquire)) {
                    // finishSlot (not raw markFinished) so the metrics sink
                    // counts the cancellation; safe pre-prefill — commit
                    // no-ops with legacy_gen==null.
                    finishSlot(sch, slot, "cancelled");
                    continue;
                }
                var prefill_sw = io_util.Stopwatch.init(sch.io);
                var qsa_gap_retried = false;
                prefill: while (true) {
                    runPrefill(sch, slot) catch |err| {
                        if (err == error.Cancelled) {
                            log.info("[scheduler] prefill aborted: client disconnected\n", .{});
                            finishSlot(sch, slot, "cancelled");
                            break :prefill;
                        }
                        if (err == error.QsaHistoryGap and !qsa_gap_retried) {
                            if (slot.model.prefix_cache) |*hc| _ = prefix_cache_mod.HotPrefixCache.dropQsaGapEntry(hc);
                            if (slot.ssm_entries) |ents| prefix_cache_mod.HotPrefixCache.resetSsmEntries(ents);
                            if (slot.model.transformer) |xf| {
                                slot.cache.truncate(0, xf.s) catch {};
                                xf.resetQsaPooledRope();
                                xf.qwen4MtpResetOwned(slot.enable_mtp);
                            }
                            if (slot.legacy_gen) |*g| {
                                g.deinit(slot.allocator);
                                slot.legacy_gen = null;
                            }
                            slot.moe_seq_offset = 0;
                            slot.cached_tokens = 0;
                            slot.skip_prefix_cache = true;
                            log.warn("[hot-cache] restored entry failed the QSA history check — dropped, cold prefill\n", .{});
                            qsa_gap_retried = true;
                            continue :prefill;
                        }
                        const name = mlx.settledErrorName(err);
                        log.err("[scheduler] prefill failed for slot: {s}\n", .{name});
                        slot.markError(name);
                        releaseNativeState(slot);
                        break :prefill;
                    };
                    break :prefill;
                }
                if (postPrefillTerminal(slot)) continue;
                // Interleaved decode time is billed to the decoding slots, so prefill_ns excludes it.
                slot.prefill_ns = prefill_sw.read() -| slot.prefill_interleaved_ns;
                if (slot.prefill_interleaved_ns > 0) log.debug("[interleave] prefill {d} ms, hosted decode {d} ms\n", .{
                    slot.prefill_ns / std.time.ns_per_ms, slot.prefill_interleaved_ns / std.time.ns_per_ms,
                });
                // Exact time-to-first-token: elapsed from request arrival
                // (Slot.init, pre-queue-wait) to prefill completion. Captured
                // here rather than derived by subtraction in finishSlot, so a
                // slot that finishes mid-tick can't skew it (metrics TTFT fix).
                slot.first_token_ns = @intCast(slot.request_start_ts.untilNow(sch.io, .boot).nanoseconds);
                sch.queue_mu.lockUncancelable(sch.io);
                sch.decoding.append(sch.allocator, slot) catch |err| {
                    sch.queue_mu.unlock(sch.io);
                    slot.markError(@errorName(err));
                    releaseNativeState(slot);
                    continue;
                };
                sch.queue_mu.unlock(sch.io);
            }
        }

        // 3. Build active-list snapshot (skip cancelled / finished / errored).
        var active: std.ArrayList(*Slot) = .empty;
        defer active.deinit(sch.allocator);
        {
            sch.queue_mu.lockUncancelable(sch.io);
            defer sch.queue_mu.unlock(sch.io);
            for (sch.decoding.items) |s| {
                if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
                active.append(sch.allocator, s) catch break;
                _ = s.in_pass.fetchAdd(1, .acq_rel);
            }
        }
        defer for (active.items) |s| {
            _ = s.in_pass.fetchSub(1, .acq_rel);
        };

        // 4. Decode tick. Charge the full wall-clock tick time to each
        //    participating slot — for batched ticks this matches the per-slot
        //    throughput a user actually observes (their stream advances at
        //    the tick cadence regardless of how many peers share it).
        if (active.items.len > 0) {
            var decode_sw = io_util.Stopwatch.init(sch.io);
            runDecodeTick(sch, active.items) catch |err| {
                const name = mlx.settledErrorName(err);
                log.err("[scheduler] decode tick failed: {s}\n", .{name});
                for (active.items) |s| s.markError(name);
            };
            const tick_ns = decode_sw.read();
            for (active.items) |s| s.decode_ns +|= tick_ns;
        }
        worked = n_prefill > 0 or active.items.len > 0;

        // 5. Cull finished / errored / cancelled from `decoding`.
        cullDecoding(sch);
    }
    flushImatrixCaptures(sch);
}

pub fn postPrefillTerminal(slot: *Slot) bool {
    if (slot.state == .errored) return true;
    if (!slot.cancelled.load(.acquire)) return false;
    releaseNativeState(slot);
    return true;
}

/// The slot still belongs to its connection thread until that thread calls `complete`; the
/// inference thread just stops touching it.
pub fn cullDecoding(sch: *Scheduler) void {
    cullDecodingWith(sch, commitSlotIfApplicable);
}

pub fn cullDecodingWith(sch: *Scheduler, commit: *const fn (*Scheduler, *Slot) void) void {
    // One drop decision per slot, under the lock `complete()` removes under. A slot dropped here is
    // out of `decoding`, and its memory lives until this thread's cleanup drain.
    var dropped: std.ArrayList(*Slot) = .empty;
    defer dropped.deinit(sch.allocator);
    sch.queue_mu.lockUncancelable(sch.io);
    var i: usize = 0;
    while (i < sch.decoding.items.len) {
        const s = sch.decoding.items[i];
        const drop = s.cancelled.load(.acquire) or s.finished or s.error_code != null;
        // An allocation failure keeps the slot for the next tick; every pass skips it meanwhile.
        if (drop and if (dropped.append(sch.allocator, s)) true else |_| false) {
            _ = sch.decoding.orderedRemove(i);
        } else i += 1;
    }
    publishLiveKvResidency(sch, null);
    sch.queue_mu.unlock(sch.io);
    // A GLM decode-phase cancel commits before its native state is released; the cleanup drain's
    // commit would read a reset request and add nothing.
    for (dropped.items) |s| {
        if (commitsAtCull(s)) {
            commit(sch, s);
            s.committed_at_cull = true;
        }
        releaseNativeState(s);
    }
}

/// A cancelled decode-phase slot commits at the cull when its state is GLM's native request, which resets right
/// after, or DeepSeek's live record: the next exclusive request is admitted as soon as the slot leaves `decoding`
/// and takes that record over, so a commit at the cleanup drain would publish onto its state.
fn commitsAtCull(s: *const Slot) bool {
    if (!s.cancelled.load(.acquire) or s.finished or s.error_code != null) return false;
    if (s.glm5_request != null) return true;
    const xfm = s.model.transformer orelse return false;
    return xfm.dsv41 != null;
}

pub fn commitCancelledAtCleanup(sch: *Scheduler, s: *Slot, commit: *const fn (*Scheduler, *Slot) void) void {
    if (!s.committed_at_cull) commit(sch, s);
}

/// Native GLM has no reusable prefix snapshot: a slot the inference thread stops touching
/// (finished, errored or cancelled) releases its reserved target state before the next
/// admission read, not when its connection thread completes it.
pub fn releaseNativeState(slot: *Slot) void {
    if (slot.glm5_request) |*request| request.reset();
}

/// Caller holds `queue_mu`; inference thread only (it owns the slots' arrays).
/// `prefilling` is the slot mid-prefill, which is not in `decoding` yet.
/// The byte total feeds `/props` and is always published; the rows only with a `--metrics` sink.
fn publishLiveKvResidency(sch: *Scheduler, prefilling: ?*Slot) void {
    const observe = sch.metrics != null;
    sch.live_session_count = 0;
    var bytes: u64 = 0;
    if (prefilling) |p| {
        const b = slotStateBytes(p);
        bytes += b -| slotDonatedBytes(sch, p);
        if (observe) recordLiveSession(sch, p, .prefill, b);
    }
    for (sch.decoding.items) |s| {
        const b = slotStateBytes(s);
        bytes += b -| slotDonatedBytes(sch, s);
        if (observe) recordLiveSession(sch, s, .decode, b);
    }
    sch.resident_live_kv_bytes.store(bytes, .monotonic);
}

fn recordLiveSession(sch: *Scheduler, s: *const Slot, phase: metrics_mod.Session.Phase, state_bytes: u64) void {
    if (sch.live_session_count == sch.live_sessions.len) return;
    const prompt: u32 = if (phase == .prefill) @intCast(s.full_prompt.len) else s.prompt_tokens;
    var session = metrics_mod.Session.init(s.model.id, phase, prompt + s.completion_tokens, s.cached_tokens, s.completion_tokens, state_bytes);
    // sushi extensions over the upstream row: the stable submit sequence, the
    // request's own output cap, and its age — measured from the immutable
    // arrival anchor `Slot.init` stamped, re-derived at every publish so a
    // long tick never leaves a stale age behind.
    session.request_id = s.request_id;
    session.max_tokens = s.max_tokens;
    const age_ns = s.request_start_ts.untilNow(sch.io, .boot).nanoseconds;
    session.elapsed_seconds = @as(f64, @floatFromInt(age_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s));
    session.entry_id = s.restored_entry;
    sch.live_sessions[sch.live_session_count] = session;
    sch.live_session_count += 1;
}

/// GPU bytes the slot's own state holds: its KV layers, the hybrid's recurrent entries and the
/// ring restore points it holds. A restored share whose buffer the hot cache still owns is billed
/// to that entry (`shared_view`), not here.
pub fn slotStateBytes(s: *const Slot) u64 {
    var bytes = s.cache.residentBytes();
    if (s.glm5_request) |*request| bytes += request.residentBytes();
    if (s.legacy_gen) |*gen| {
        if (gen.dflash_ctx) |*ctx| bytes += ctx.cache.residentBytes();
    }
    if (s.ssm_entries) |ents| for (ents) |*e| {
        bytes += transformer_mod.ssmEntryBytes(e);
    };
    return bytes + s.ring_cps.bytes() + dsv41StateBytes(s);
}

/// The part of `slotStateBytes` that `resident_hot_cache_bytes` already bills: a donated checkout's
/// buffers, when the slot's cache is the one published.
fn slotDonatedBytes(sch: *const Scheduler, s: *const Slot) u64 {
    const hc = if (s.model.prefix_cache) |*h| h else return 0;
    if (sch.hot_prefix_cache != hc) return 0;
    return hc.donatedBytes(@intFromPtr(s));
}

/// Shutdown exit: an armed activation capture is written HERE, the last point
/// this thread is alive. `Scheduler.deinit` unloads on the caller's thread,
/// where the capture's mlx stream does not exist.
fn flushImatrixCaptures(sch: *Scheduler) void {
    sch.registry.mutex.lockUncancelable(sch.io);
    defer sch.registry.mutex.unlock(sch.io);
    var it = sch.registry.entries.valueIterator();
    while (it.next()) |entry_ptr| {
        if (entry_ptr.*.transformer) |x| x.flushImatrix();
    }
}

fn dflashContextCoversPrefix(context_len: usize, prefix_len: usize) bool {
    return context_len == prefix_len;
}

/// Only an all-pad generation poisons the prefix; a zero-token one still holds a real prefill.
fn commitDeclinesPadOnly(n_gen: usize, all_pad: bool) bool {
    return n_gen > 0 and all_pad;
}

/// Phase A6: commit a successfully completed slot's KV cache to the hot
/// prefix cache. Called from the inference thread BEFORE `markFinished`
/// broadcasts, so the slot is still alive (the conn thread is blocked in
/// `waitNext`). Skipped for pad-only generations, vision-bearing slots
/// (stale embeddings would be reused), and slots with no generated tokens.
pub fn commitSlotIfApplicable(sch: *Scheduler, slot: *Slot) void {
    // Phase D: per-model prefix cache — read off the slot's LoadedModel.
    const hc: *prefix_cache_mod.HotPrefixCache = if (slot.model.prefix_cache) |*p| p else return;
    if (slot.error_code != null) return;
    // `finishSlot` can be reached on an EOS the failing forward itself produced, before the
    // tick wrapper reads the latch: an unread latch means the KV under this commit may be garbage.
    if (mlx.errorPending()) return;
    const gen_ptr = if (slot.legacy_gen) |*g| g else {
        // A Generator-less slot whose cache is non-empty is a prefill the
        // client disconnected from mid-chunk-loop: initWithOptions threw
        // error.Cancelled before `slot.legacy_gen = gen` ever ran, but every
        // chunk that DID forward still lives in slot.cache. Commit that
        // forwarded prefix instead of dropping it. This arm sits ABOVE the
        // pad-only guard: `was_pad_only` starts true and only flips on the
        // first pushed token, so a slot that never pushed one is not
        // pad-POISONED, it is merely empty.
        commitCancelledPrefillSlot(slot, hc);
        publishHotCacheResidency(sch);
        return;
    };
    const n_gen = gen_ptr.generated_ids.items.len;
    if (commitDeclinesPadOnly(n_gen, slot.was_pad_only)) return;

    // Construct the full token sequence: the original prompt + everything
    // generated this turn. The cache reflects exactly this state — Generator
    // forwarded each emitted token into slot.cache as it was sampled.
    const total_len = slot.full_prompt.len + gen_ptr.generated_ids.items.len;
    if (slot.model.transformer) |t| if (t.dsv41 != null) @import("dsv41/dsv41_prefix.zig").markRows(&slot.cache) catch |err| {
        log.warn("[hot-cache] DeepSeek row marker failed: {s} — not committed\n", .{@errorName(err)});
        return;
    };
    const total_tokens = sch.allocator.alloc(u32, total_len) catch return;
    defer sch.allocator.free(total_tokens);
    @memcpy(total_tokens[0..slot.full_prompt.len], slot.full_prompt);
    @memcpy(total_tokens[slot.full_prompt.len..], gen_ptr.generated_ids.items);
    if (slot.model.transformer) |t| if (t.dsv41) |dsv| hc.publishLiveDsv41(dsv, total_tokens, slot.has_tools, prefix_cache_mod.MediaStamp.of(slot.vision_key, slot.media_start));

    // Phase 1: drain any SSM checkpoints captured by the Generator's prefill
    // loop and hand them to the cache alongside the KV snapshot. For plain-
    // attn models this returns an empty slice (no allocator hit). Ownership
    // transfers to the cache via `commitWithSsm`; freeing happens on
    // eviction.
    settleThoughtPoint(slot, gen_ptr);
    const ssm_cps_slice = gen_ptr.takeSsmCheckpoints();
    const ssm_cps_opt: ?[]transformer_mod.SSMCheckpoint = if (ssm_cps_slice.len > 0) ssm_cps_slice else null;
    if (ssm_cps_slice.len == 0 and gen_ptr.ssm_checkpoint_alloc != null) {
        // Empty list — free the (zero-length) slice we got back so the
        // allocator's bookkeeping stays clean.
        gen_ptr.ssm_checkpoint_alloc.?.free(ssm_cps_slice);
    }
    if (slot.glm5_request) |*request| {
        const window: ?transformer_mod.KVCache = if (gen_ptr.glm_prefill_window) |*w| .{ .entries = w.snapshot.entries, .step = w.snapshot.step, .allocator = w.snapshot.allocator, .config = w.snapshot.config } else null;
        const dflash: ?prefix_cache_mod.DflashCommit = if (window) |*w| .{ .cache = w, .base_pos = gen_ptr.glm_prefill_window.?.base_pos } else null;
        commitGlmSlot(hc, slot, request, total_tokens, ssm_cps_opt, dflash, slot.full_prompt.len);
        publishHotCacheResidency(sch);
        return;
    }
    // qwen4_exp: the newest checkpoint takes the slot's live QSA indexer history as a view
    // of the capacity buffer (the slot is torn down right after). A failure commits the
    // entry history-less, which a QSA arch treats as a miss.
    if (ssm_cps_opt) |cps| {
        if (!transformer_mod.checkpointHasQsaPooled(&cps[cps.len - 1])) {
            if (slot.ssm_entries) |ents| {
                if (slot.model.transformer) |xf| {
                    transformer_mod.handoffQsaHistoryToLatest(cps, ents, xf.s) catch |err| {
                        log.warn("[hot-cache] QSA history handoff failed: {s} — committing without it\n", .{@errorName(err)});
                    };
                }
            }
        }
    }
    // A runtime fallback leaves the dormant assistant context at its last
    // speculative boundary while serial decode continues growing the trunk.
    // Only pair the assistant payload with this prefix when both end at the
    // exact same absolute position; otherwise the next turn must rebuild it.
    const dflash_commit: ?prefix_cache_mod.DflashCommit = if (gen_ptr.dflash_ctx) |*dc|
        if (dflashContextCoversPrefix(dc.absLen(), total_len))
            .{ .cache = &dc.cache, .base_pos = dc.base_pos }
        else
            null
    else
        null;
    // MTP committed history: unlike dflash there is no exact-coverage
    // requirement — the restore clamps to the matched length and declines a
    // history that ends short, so a partial history (the deferred-stash lag,
    // a runtime disable) is still worth committing. What MUST hold is that
    // only COMMITTED entries are snapshotted: truncate off the speculative
    // draft tail first (offset-only, cheap).
    const mtp_commit: ?prefix_cache_mod.DflashCommit = blk: {
        const mc = if (gen_ptr.mtp_cache) |*m| m else break :blk null;
        // A released module head is another slot's to read now; the history stopped
        // growing at the switch anyway.
        if (gen_ptr.mtpModuleHeadReleased()) break :blk null;
        const committed = gen_ptr.mtpCommittedHistoryLen();
        if (committed == 0) break :blk null;
        mc.truncate(committed, slot.model.transformer.?.s) catch |err| {
            log.warn("[hot-cache] mtp history trim failed: {s} — not committed\n", .{@errorName(err)});
            break :blk null;
        };
        // The qwen4_exp in-checkpoint head also commits its QSA half.
        mc.activate();
        const head = mc.head();
        // Its row count IS its cache's step (`KVCache.update` never advances a non-zero
        // layer index); a snapshot whose step disagrees is refused, and says why.
        if (head) |t| {
            const hm = &t.qwen4_mtp.?;
            if (hm.cache.step != hm.seq_offset) {
                log.warn("[hot-cache] mtp head step gap (cache.step={d}, history={d} rows) — head history not committed\n", .{ hm.cache.step, hm.seq_offset });
                break :blk null;
            }
        }
        break :blk .{
            .cache = mc.kv() orelse break :blk null,
            .base_pos = gen_ptr.mtp_position_base,
            .head = if (head) |t| &t.qwen4_mtp.?.entry else null,
            .head_pos_base = if (head) |t| t.qwen4_mtp.?.pos_base else 0,
            .head_marks = if (head) |t| t.qwen4MtpMarks() else &.{},
        };
    };
    // Ownership transfers to the cache on every outcome, like the checkpoints.
    const ring_cps = slot.ring_cps;
    slot.ring_cps = .{};
    const finish_st = hc.commitWithRing(&slot.cache, total_tokens, slot.has_tools, slot.vision_key, slot.cache_key, slot.media_start, ssm_cps_opt, dflash_commit, mtp_commit, slot.full_prompt.len, ring_cps) catch |err| {
        // Ownership of the checkpoints transferred to the cache regardless of
        // the outcome — its error paths free them (#330 adjacent: freeing
        // here too was a double free, with a different allocator).
        log.warn("[hot-cache] commit failed: {s}\n", .{@errorName(err)});
        return;
    };
    publishHotCacheResidency(sch);
    // The decline paths already log their reason inside the cache; a silent
    // `_ =` here is fine — this caller has no commit-shaped log to lie about.
    _ = finish_st;
}

/// Logical committed length for a cancelled-prefill commit: the tokens
/// actually forwarded into the KV when the chunk loop aborted, clamped to
/// the prompt and gated on the floor below which an entry is LRU pollution
/// rather than saved work. Pure so the policy is unit-testable without a slot.
fn cancelledPrefillCommitLen(step: usize, prompt_len: usize) ?usize {
    const len = @min(step, prompt_len);
    if (len < prefix_cache_mod.MIN_CANCELLED_COMMIT_TOKENS) return null;
    return len;
}

/// Commit the forwarded prefix of a prefill the client disconnected from
/// (error.Cancelled out of `Generator.initWithOptions`; `legacy_gen` was
/// never assigned). The partial KV in `slot.cache` is valid — every chunk
/// that ran was a real forward — so the next request sharing this prefix
/// skips exactly those chunks. The entry key is the forwarded prefix ONLY:
/// `full_prompt` beyond `cache.step` was never forwarded and must not ride
/// the key (a key longer than its KV is the alignment-error class).
///
/// Hybrids commit iff checkpoints were salvaged (`slot.cancelled_prefill`):
/// `initWithOptions` hands its captured stride checkpoints to the slot on
/// the abort, and they restore exactly like a normal finish's. KV-only
/// hybrid entries still restore as a cold miss ("hybrid miss") while
/// occupying an LRU slot, so a salvage-less hybrid prefill is declined —
/// the cancel landed before the first stride boundary. decode-phase cancels
/// keep their checkpoints in the live Generator and commit through the
/// normal arm above.
fn commitCancelledPrefillSlot(slot: *Slot, hc: *prefix_cache_mod.HotPrefixCache) void {
    const salvage = &slot.cancelled_prefill;
    // Hybrid restore requires SSM checkpoints; a checkpoint-less hybrid
    // entry restores as a cold miss ("hybrid miss") while occupying an LRU
    // slot. Non-hybrids commit KV-only.
    if (slot.ssm_entries != null and salvage.checkpoints.len == 0) {
        log.debug("[hot-cache] cancelled prefill carried no stride checkpoints — a hybrid entry would restore as a miss; not committed\n", .{});
        return;
    }
    const is_dsv41 = if (slot.model.transformer) |t| t.dsv41 != null else false;
    if (is_dsv41 and salvage.checkpoints.len == 0) {
        log.debug("[hot-cache] cancelled prefill carried no checkpoints — a DeepSeek entry would restore as a miss; not committed\n", .{});
        return;
    }
    // The sink's `forwarded` is the authoritative length — `cache.step`
    // only advances when Generator init completes, so it reads 0 on every
    // aborted prefill.
    const len = cancelledPrefillCommitLen(salvage.forwarded, slot.full_prompt.len) orelse {
        log.debug("[hot-cache] cancelled prefill forwarded < {d} tokens — below the commit floor; not committed\n", .{prefix_cache_mod.MIN_CANCELLED_COMMIT_TOKENS});
        return;
    };
    const cps: ?[]transformer_mod.SSMCheckpoint = if (salvage.checkpoints.len > 0) salvage.checkpoints else null;
    if (is_dsv41) @import("dsv41/dsv41_prefix.zig").markRows(&slot.cache) catch return;
    // Pass the media boundary RAW: when the cancelled prefill forwarded less
    // than the media position, the cache re-keys the pure-text entry to the
    // null vision key (a kept pixel key with no boundary is the
    // conservative-rejection poison shape; live 2026-09-07).
    // A cancelled prefill stopped mid-prompt, so the entry claims only the media blocks it
    // forwarded (`truncatedMediaKey`), not the whole request's chain.
    const media = prefix_cache_mod.truncatedMediaKey(slot.media_chain, slot.vision_key, slot.media_start, len);
    const media_start: ?usize = if (media.start) |v| @as(usize, v) else null;
    // Ownership of the checkpoints transfers to the cache unconditionally —
    // its error paths free them (#330 adjacent) — so detach from the slot
    // BEFORE the call or Slot.deinit frees them a second time.
    slot.cancelled_prefill = .{};
    if (slot.glm5_request) |*request| return commitGlmSlot(hc, slot, request, slot.full_prompt[0..len], cps, null, len);
    const st = hc.commitWithMediaState(&slot.cache, slot.full_prompt[0..len], slot.has_tools, media.key, slot.cache_key, media_start, cps, null, null, len) catch |err| {
        log.warn("[hot-cache] cancelled-prefill commit failed: {s}\n", .{@errorName(err)});
        return;
    };
    // Truthful outcome only: a budget decline must never print as a commit
    // (live 2026-09-07: "skipped oversized" + "committed N/M" were the SAME
    // event and the cache looked healthy while a 122k session re-prefilled
    // ~95k tokens per retry).
    switch (st) {
        .ok => |n| log.info("[hot-cache] committed {d}/{d} prompt tokens from a cancelled prefill\n", .{ n, slot.full_prompt.len }),
        .disk_only => |n| log.info("[disk-cache] captured {d}/{d} prompt tokens from a cancelled prefill\n", .{ n, slot.full_prompt.len }),
        .kept_resident => |n| log.info("[hot-cache] kept resident {d}-token entry; oversized candidate declined\n", .{n}),
        .declined => {},
    }
}

/// Publish every resident model's SSD tier (budget, bytes in use, entries) onto its own `LoadedModel`.
/// Inference thread.
pub fn publishDiskStats(sch: *Scheduler) void {
    sch.registry.mutex.lockUncancelable(sch.io);
    defer sch.registry.mutex.unlock(sch.io);
    var it = sch.registry.entries.valueIterator();
    while (it.next()) |entry_ptr| {
        const entry = entry_ptr.*;
        var snap: model_registry_mod.DiskStats.Snapshot = .{};
        if (entry.state == .ready) {
            if (entry.prefix_cache) |*hc| if (hc.disk) |*d| {
                snap = .{ .budget = d.max_bytes, .used = d.total_bytes, .entries = d.entryCount() };
            };
        }
        entry.disk_stats.publish(snap);
    }
}

/// Free finished slots on the inference thread. The request-end clear ran before this, so a
/// slot that reserved its whole KV up front parked it in MLX's pool, sized to its own prompt
/// (reused by no later request, and the admission bill never counts it back): return it.
fn deinitSlotsReturningPool(slots: []const *Slot) void {
    var returns_pool = false;
    for (slots) |s| {
        if (s.model.config) |c| returns_pool = returns_pool or c.reservesKvCapacity();
        s.deinit();
    }
    if (returns_pool) _ = mlx.mlx_clear_cache();
}

/// Allocate the slot's KVCache state (already done in Slot.init), construct
/// the per-slot Generator via `Generator.initWithOptions(.{ .ctx = slot.ctx,
/// .skip_lazy_preforward = true_for_regular, ... })`, and store it on the
/// slot. After return, the slot is ready for decode ticks.
/// Kill switch for prefill-side interleaving (SUSHI_PREFILL_INTERLEAVE=0
/// restores whole-prefill-then-decode scheduling). Default ON: the hook
/// no-ops when nothing is decoding, so an idle or single-stream server never
/// pays for it.
var prefill_interleave_cached: ?bool = null;
pub fn prefillInterleaveEnabled() bool {
    if (prefill_interleave_cached) |v| return v;
    const raw = std.c.getenv("SUSHI_PREFILL_INTERLEAVE");
    const on = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    prefill_interleave_cached = on;
    return on;
}

pub var prefill_decode_share: f32 = 0.25;

pub fn prefillDecodeShare() f32 {
    return if (prefillInterleaveEnabled()) prefill_decode_share else 0;
}

pub const PREFILL_DECODE_SHARE_MAX: f32 = 0.9;
pub fn parseDecodeShare(text: []const u8) error{InvalidDecodeShare}!f32 {
    const v = std.fmt.parseFloat(f32, text) catch return error.InvalidDecodeShare;
    if (std.math.isNan(v) or v < 0) return error.InvalidDecodeShare;
    return @min(v, PREFILL_DECODE_SHARE_MAX);
}
pub fn decodeShareBudgetNs(chunk_ns: u64, share: f32) u64 {
    if (std.math.isNan(share) or share <= 0) return 0;
    const fraction: f64 = @min(share, PREFILL_DECODE_SHARE_MAX);
    const budget = @as(f64, @floatFromInt(chunk_ns)) * fraction / (1 - fraction);
    if (budget >= @as(f64, @floatFromInt(std.math.maxInt(u64)))) return std.math.maxInt(u64);
    return @intFromFloat(budget);
}
pub const OwedTicks = struct { ticks: u32, spent_ns: u64 };
pub fn runOwedDecodeTicks(share: f32, chunk_ns: u64, first_ns: u64, ctx: *anyopaque, tick: *const fn (*anyopaque) u64) OwedTicks {
    var result = OwedTicks{ .ticks = 1, .spent_ns = first_ns };
    if (first_ns == 0) return result;
    const budget = decodeShareBudgetNs(chunk_ns, share);
    while (result.spent_ns < budget) {
        const ns = tick(ctx);
        if (ns == 0) break;
        result.ticks +|= 1;
        result.spent_ns +|= ns;
    }
    return result;
}

const InterleaveCtx = struct {
    sch: *Scheduler,
    /// The slot being prefilled: not in `decoding` yet, its KV still counts.
    slot: *Slot,
    cancelled: *const std.atomic.Value(bool),
    chunk_sw: io_util.Stopwatch,
    decode_ns: u64 = 0,
    ticks: u32 = 0,
};

/// SSD-first write-through: persist each completed prefill chunk as the prefill produces it,
/// so a cancelled or killed prefill leaves a restorable chunk-aligned prefix. Armed only with
/// a live background writer, so the inference thread never pays the file write.
fn prefillWriteThroughCb(opaque_ctx: *anyopaque, abs_kv_pos: usize, cps: []transformer_mod.SSMCheckpoint) void {
    const wc: *WriteThroughCtx = @ptrCast(@alignCast(opaque_ctx));
    const slot = wc.slot;
    const hc: *prefix_cache_mod.HotPrefixCache = if (slot.model.prefix_cache) |*p| p else return;
    if (!hc.ssd_first) return;
    const d = if (hc.disk) |*dd| dd else return;
    if (d.writer == null) return;
    if (abs_kv_pos == 0 or abs_kv_pos > slot.full_prompt.len) return;
    const s = if (slot.model.transformer) |x| x.s else return;
    wc.chunks += 1;
    // A chunk record covers a prefix, so it may claim only the media blocks inside it.
    const media = prefix_cache_mod.truncatedMediaKey(slot.media_chain, slot.vision_key, slot.media_start, abs_kv_pos);
    if (slot.glm5_request) |*request| return hc.writeThroughGlm(&wc.entry_id, request, slot.full_prompt[0..abs_kv_pos], slot.has_tools, media, cps, s);
    if (slot.model.transformer.?.dsv41) |dsv| return hc.writeThroughDsv41(dsv, &slot.cache, slot.full_prompt[0..abs_kv_pos], slot.has_tools, media, cps, s, "prefill");
    // RAM-backed mode banks one chunk as crash salvage. SSD-only mode must keep pace with
    // prefill or the first completed turn can leave most of its prefix unavailable after restart.
    hc.writeThroughHybrid(
        slot.cache.entries,
        slot.cache.config,
        slot.full_prompt[0..abs_kv_pos],
        slot.has_tools,
        media,
        cps,
        if (hc.ram_enabled) WRITE_THROUGH_FLUSH_BOUND_BYTES else std.math.maxInt(u64),
        s,
    );
}

/// The write-through hook's per-call flush bound: one byte, so the chunk loop stops after the first chunk it writes.
pub const WRITE_THROUGH_FLUSH_BOUND_BYTES: u64 = 1;

const WriteThroughCtx = struct {
    slot: *Slot,
    chunks: u32 = 0,
    /// GLM: the SSD entry this prefill is writing.
    entry_id: ?u64 = null,
};

/// The write-through only pays when this turn's prefill produces at least one whole disk
/// chunk: below that it persisted a warm turn's whole restored prefix inside TTFT
/// (+183/+369/+737 ms at 16k/32k/64k) for a prefix the end-of-request commit persists anyway.
fn writeThroughSpanReached(new_span: usize, chunk_tokens: u32) bool {
    return new_span >= chunk_tokens;
}

var write_through_span_declined_logged = std.atomic.Value(bool).init(false);
/// The one gate for the write-through: SSD-first + a live writer + a disk tier + a new span
/// worth at least one chunk.
fn writeThroughArmed(slot: *Slot, new_span: usize) bool {
    const hc: *prefix_cache_mod.HotPrefixCache = if (slot.model.prefix_cache) |*p| p else return false;
    if (!hc.ssd_first) return false;
    const d = if (hc.disk) |*dd| dd else return false;
    if (d.writer == null) return false;
    // A DeepSeek restore point is a full copy of the state: it goes to SSD however short the new span.
    if (slot.model.transformer) |t| if (t.dsv41 != null) return true;
    if (!writeThroughSpanReached(new_span, d.chunk_tokens)) {
        if (!write_through_span_declined_logged.swap(true, .monotonic)) {
            log.info("  [disk-cache] prefill write-through declined: {d} new tokens is under one chunk ({d}) — the end-of-request commit persists this turn\n", .{ new_span, d.chunk_tokens });
        }
        return false;
    }
    return true;
}

/// The checkpoints a Qwen request may list when no write-through frees them as they land: with an SSD tier
/// behind the cache it holds at most `SSM_RESIDENT_MAX` states, which is what `server.ssmResidentCap` bills.
/// A write-through request holds two whatever it lists (`HotPrefixCache.writeThroughHybrid`).
fn qwenCheckpointCap(slot: *Slot, cp_max: u32, write_through: bool) u32 {
    if (!qwenCheckpointCapped(slot, write_through)) return cp_max;
    const cap = transformer_mod.SSM_RESIDENT_MAX - 1;
    return if (cp_max == 0) cap else @min(cp_max, cap);
}

fn qwenCheckpointCapped(slot: *Slot, write_through: bool) bool {
    const cfg = slot.model.config orelse return false;
    if (write_through or !cfg.isQwen4()) return false;
    const hc: *prefix_cache_mod.HotPrefixCache = if (slot.model.prefix_cache) |*p| p else return false;
    return hc.disk != null;
}

/// Does this GLM request's prefill move each KDA checkpoint to SSD and free its RAM copy? Admission bills
/// from this answer and the prefill arms its hook from the same one.
fn glmCheckpointsRelease(slot: *Slot, new_span: usize) bool {
    return slot.glm5_request != null and writeThroughArmed(slot, new_span);
}

/// Context for `Generator.InitOptions.chunk_width_hook`. `cfg` is optional (embedded engines).
const ChunkWidthCtx = struct {
    sch: *Scheduler,
    cfg: ?*const model_mod.ModelConfig,
    kv_bits: u64,
    /// This slot's own per-model cache, never `sch.hot_prefix_cache`; only `stagedHostBytes`
    /// is read, on the inference thread.
    hc: ?*prefix_cache_mod.HotPrefixCache,
};

/// Host bytes the SSD writer is holding for this slot right now.
fn chunkWidthStagedBytes(wc: *ChunkWidthCtx) u64 {
    const hc = wc.hc orelse return 0;
    return hc.stagedHostBytes();
}

fn chunkWidenConfirmCb(opaque_ctx: *anyopaque, pos: usize, want: u32) bool {
    const wc: *ChunkWidthCtx = @ptrCast(@alignCast(opaque_ctx));
    const cfg = wc.cfg orelse return false;
    const ok = prefill_chunk_widen_ok orelse return false;
    return ok(cfg, wc.kv_bits, pos, want, chunkWidthStagedBytes(wc));
}

fn chunkWidthCb(
    opaque_ctx: *anyopaque,
    pos: usize,
    cur: u32,
    cap: u32,
    st: *mtp_plan.AdaptiveWidthState,
) u32 {
    const wc: *ChunkWidthCtx = @ptrCast(@alignCast(opaque_ctx));
    const cfg = wc.cfg orelse return cur;
    const pick = prefill_chunk_adapt orelse return cur;
    const next = pick(cfg, wc.kv_bits, pos, cur, cap, st, chunkWidthStagedBytes(wc));
    return next;
}

/// Called with `mu` held; returns with it held. Drops `mu` while waiting so the
/// pass holding the slot can finish its tick (which takes `mu` itself).
fn waitPassesOut(io: std.Io, mu: *std.Io.Mutex, in_pass: *std.atomic.Value(u32)) void {
    if (in_pass.load(.acquire) == 0) return;
    mu.unlock(io);
    while (in_pass.load(.acquire) != 0) std.Io.sleep(io, .fromMilliseconds(1), .real) catch {};
    mu.lockUncancelable(io);
}

fn interleaveDecodeTickCb(opaque_ctx: *anyopaque) void {
    const ic: *InterleaveCtx = @ptrCast(@alignCast(opaque_ctx));
    if (ic.ticks == 0) {
        log.debug("[interleave] engaged: decode ticks between prefill chunks\n", .{});
    }
    if (ic.cancelled.load(.acquire)) return;
    const chunk_ns = ic.chunk_sw.read();
    const first_ns = interleaveDecodeTick(ic.sch);
    const owed = runOwedDecodeTicks(prefillDecodeShare(), chunk_ns, first_ns, ic, interleaveDecodeTickOpaque);
    ic.ticks +|= owed.ticks;
    ic.decode_ns +|= owed.spent_ns;
    {
        // Refresh the snapshot at the chunk boundary: the prefill row's token
        // counts, its growing KV and every decode row's age move between chunks,
        // and inside a long prefill this is the only point that republishes them.
        ic.sch.queue_mu.lockUncancelable(ic.sch.io);
        defer ic.sch.queue_mu.unlock(ic.sch.io);
        publishLiveKvResidency(ic.sch, ic.slot);
    }
    ic.chunk_sw.reset();
}

/// One decode tick for the streams currently decoding, run from INSIDE a
/// prefill (between chunks, and between the slots of one admitted batch).
/// Returns the tick's wall-clock ns (0 when no stream is active). The
/// prefilling slot is not in `decoding` yet, so the tick only advances OTHER
/// requests' Generators — same-thread MLX, no reentrancy into this prefill.
fn interleaveDecodeTickOpaque(ctx: *anyopaque) u64 {
    const ic: *InterleaveCtx = @ptrCast(@alignCast(ctx));
    if (ic.cancelled.load(.acquire)) return 0;
    return interleaveDecodeTick(ic.sch);
}

fn interleaveDecodeTick(sch: *Scheduler) u64 {
    var buf: [32]*Slot = undefined;
    var n: usize = 0;
    sch.queue_mu.lockUncancelable(sch.io);
    for (sch.decoding.items) |s| {
        if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
        if (n >= buf.len) break;
        buf[n] = s;
        _ = s.in_pass.fetchAdd(1, .acq_rel);
        n += 1;
    }
    sch.queue_mu.unlock(sch.io);
    defer for (buf[0..n]) |s| {
        _ = s.in_pass.fetchSub(1, .acq_rel);
    };
    if (n == 0) return 0;
    // The interval since these slots' previous tick contains a prefill chunk; the serial
    // cell must not fold it as a token's wall time. Drop it; the next tick seeds afresh.
    for (buf[0..n]) |s| {
        if (s.legacy_gen) |*g| {
            g.invalidateSerialClock();
            g.invalidateRoundClock();
        }
    }
    var sw = io_util.Stopwatch.init(sch.io);
    runDecodeTick(sch, buf[0..n]) catch |err| {
        const name = mlx.settledErrorName(err);
        log.err("[interleave] decode tick failed: {s}\n", .{name});
        for (buf[0..n]) |s| s.markError(name);
    };
    const tick_ns = sw.read();
    for (buf[0..n]) |s| s.decode_ns +|= tick_ns;
    return tick_ns;
}

/// Take `pending[idx]` into this tick's prefill pass (queue_mu held; the caller removes it
/// from `pending`).
fn admitForPrefillLocked(sch: *Scheduler, idx: usize) *Slot {
    const slot = sch.pending.items[idx];
    var start_line: [64]u8 = undefined;
    const waited_ns: u64 = @intCast(slot.request_start_ts.untilNow(sch.io, .boot).nanoseconds);
    if (queueStartLine(&start_line, slot, waited_ns)) |line| log.info("{s}\n", .{line});
    _ = slot.in_pass.fetchAdd(1, .acq_rel);
    // `submit` reserved room for every in-flight slot.
    sch.prefilling.appendAssumeCapacity(slot);
    return slot;
}

/// The prefill pass over `slot` is over, however it ended.
fn endPrefillPass(sch: *Scheduler, slot: *Slot) void {
    sch.queue_mu.lockUncancelable(sch.io);
    for (sch.prefilling.items, 0..) |s, i| {
        if (s != slot) continue;
        _ = sch.prefilling.swapRemove(i);
        break;
    }
    sch.queue_mu.unlock(sch.io);
    _ = slot.in_pass.fetchSub(1, .acq_rel);
}

/// t1 is on the host once an MTP prefill ends; the first round would only stream it after its
/// verify. An EOS t1 stays with the round, which ends the answer the usual way.
fn handoverToken(mtp: bool, completion_tokens: u32, done: bool, logprobs_n: u32, t1: u32, eos: []const u32) ?u32 {
    // The swallowed round echo would also drop t1's logprob entry.
    if (!mtp or completion_tokens != 0 or done or logprobs_n > 0 or generate_mod.isEosId(t1, eos)) return null;
    return t1;
}

fn publishHandoverToken(slot: *Slot, gen: *const Generator) void {
    const t1 = handoverToken(gen.mtp != null, gen.completion_tokens, gen.done, slot.logprobs_n, gen.next_token_id, slot.eos_token_ids) orelse return;
    slot.pushToken(t1);
    slot.handover_token = t1;
}

fn runPrefill(sch: *Scheduler, slot: *Slot) !void {
    // Mark the phase for the whole of prefill, and clear both signals on EVERY
    // exit path (success, cancel, error). `requests_prefilling` flips at entry
    // so the panel isn't blind until the first chunk lands; the chunk loop in
    // generate.zig stores absolute progress into `inflight_prefill_tokens`.
    //
    // Gated on `--metrics`, per the observability contract: when it's off the
    // prefill path executes NO extra instruction at all (the chunk loop's hook
    // is null too — see the `prefill_progress` option below).
    const observe = sch.metrics != null;
    if (observe) {
        _ = sch.requests_prefilling.fetchAdd(1, .monotonic);
        // Advertise the prefilling slot before its first chunk lands: the cull
        // publish only lists `decoding`, so without this the row is invisible
        // for the whole (possibly multi-minute) prefill.
        sch.queue_mu.lockUncancelable(sch.io);
        defer sch.queue_mu.unlock(sch.io);
        publishLiveKvResidency(sch, slot);
    }
    defer if (observe) {
        _ = sch.requests_prefilling.fetchSub(1, .monotonic);
        sch.inflight_prefill_tokens.store(0, .monotonic);
        sch.inflight_prefill_expected.store(0, .monotonic);
    };

    const sampling = slot.sampling;
    // Refresh ctx in case slot was relocated (paranoia — slot is heap so no,
    // but cheap).
    slot.ctx.cache = &slot.cache;
    slot.ctx.moe_seq_offset = &slot.moe_seq_offset;
    slot.ctx.ssm_entries = slot.ssm_entries;
    slot.ctx.vision_embeddings = slot.vision_embeddings;
    slot.ctx.mrope_pos = slot.mrope_pos;
    slot.ctx.mrope_total = slot.mrope_total;
    slot.ctx.mrope_delta = slot.mrope_delta;
    slot.ctx.capture_hidden = null;
    if (slot.model.transformer) |xfm| try xfm.ssmGroupRelease(&slot.ctx);

    // A module-owned arch that CAN rewind (`moduleStateSpecRollback`) may run
    // its own MTP head; PLD/drafter stay off there for the reasons in
    // `specInitWiring`.
    // Module CLASS (owned or shared-readonly), not ownership alone: qwen4
    // batches its plain slots but its spec wiring stays the module one.
    const owns_module_state = slot.model.transformer != null and
        slot.model.transformer.?.moduleSpecWiring();
    const module_spec_rollback = slot.model.transformer != null and
        slot.model.transformer.?.moduleStateSpecRollback();
    const wiring = specInitWiring(
        owns_module_state,
        module_spec_rollback,
        slot.enable_mtp,
        slot.mtp != null,
        slot.enable_drafter,
        slot.dflash != null,
    );
    // deepseek_v41's head is DSpark: no head cache, no hidden capture; the model holds its own state.
    const use_mtp = wiring.use_mtp;
    const use_dspark = use_mtp and if (slot.mtp) |h| h == .dsv41 else false;
    const use_head_mtp = use_mtp and !use_dspark;
    const use_dflash = wiring.use_dflash or glmNativeDflashArmed(slot.model.config.?.isGlm5(), slot.enable_drafter, slot.dflash != null, slot.sampling, slot.logprobs_n);
    log.debug("[spec-wiring] mtp={} dflash={} (slot: drafter_flag={} dflash_handle={})\n", .{
        use_mtp,             use_dflash,
        slot.enable_drafter, slot.dflash != null,
    });

    // Phase A6: prefill source-of-truth is `slot.full_prompt` — the conn
    // thread's `reuseKVCache` may have trimmed `slot.prompt_ids` based on
    // `xfm.cache` (the legacy global cache), but the slot has its own cache
    // which started empty. Using `slot.prompt_ids` would cause the model to
    // attend to only the trailing portion with empty cache, producing
    // garbage. Always start from the full prompt and let the hot prefix
    // cache (if configured) trim it back via the slot's own cache state.
    //
    // Vision-bearing slots: skip the hot cache altogether. Image tokens
    // have identical IDs but the underlying vision embeddings differ
    // per-request, so prefix matching would reuse stale features.
    var prefill_tokens: []const u32 = slot.full_prompt;
    var hot_matched: u32 = 0;
    // Are the restored rows the slot's own (a checkout, or an SSD restore)? Only then are they credited.
    var hot_checked_out: bool = false;
    // The DFlash assistant's context rides the prefix cache: a restore
    // forwards no trunk layers, so without it the assistant starts every
    // reused turn blind and drafts against nothing. Adopted by the Generator below.
    var dflash_restored: ?dflash_mod.DflashCtx = null;
    errdefer if (dflash_restored) |*dc| dc.deinit();
    // The MTP head's committed history rides the prefix cache the same way:
    // it is built from trunk hiddens, a restore forwards nothing, and a
    // blind start collapses acceptance (measured ~70 -> ~38 tok/s on warm
    // echo). Adopted by the Generator below.
    var mtp_restored: ?generate_mod.MtpRestored = null;
    errdefer if (mtp_restored) |*mr| mr.cache.deinit();
    // Phase D: per-slot model — pull transformer + prefix cache off the
    // slot's LoadedModel. Both stay resident for the slot's lifetime
    // because the conn thread holds a refcount on slot.model.
    const xfm_ptr: *Transformer = slot.model.transformer.?;
    if (slot.model.prefix_cache) |*hc| {
        {
            // Only build a restore target when this request will actually
            // draft — a non-dflash turn leaves the payload in the entry for
            // the next one that does.
            var dfl_target: ?dflash_mod.DflashCtx = if (use_dflash and slot.dflash != null)
                dflash_mod.DflashCtx.init(slot.allocator, slot.dflash.?, 0) catch null
            else
                null;
            errdefer if (dfl_target) |*dc| dc.deinit();
            var dfl_base: usize = 0;
            var mtp_target: ?generate_mod.MtpCacheRef = if (use_head_mtp and slot.mtp != null)
                slot.mtp.?.makeCache(slot.allocator) catch null
            else
                null;
            errdefer if (mtp_target) |*mc| mc.deinit();
            var mtp_base: usize = 0;
            const mtp_kv: ?*KVCache = if (mtp_target) |*mc| mc.kv() else null;
            // qwen4_exp: the head's QSA half travels with its KV; adoption is all-or-nothing.
            const mtp_head: ?*Transformer = if (mtp_target) |*mc| mc.head() else null;
            // Restore by move: the slot names itself, opting into the checkout; `finishSlot`
            // releases it on every path that ends the slot.
            const dfl_restore: ?prefix_cache_mod.DflashTarget = if (dfl_target) |*dc| .{ .cache = &dc.cache, .base_pos = &dfl_base } else null;
            const lookup_sw = io_util.Stopwatch.init(sch.io);
            const lookup = (if (xfm_ptr.dsv41) |dsv| hc.lookupAndRestoreDsv41(
                &slot.cache,
                &slot.moe_seq_offset,
                dsv,
                xfm_ptr.s,
                slot.full_prompt,
                slot.has_tools,
                slot.vision_key,
                slot.media_start,
                slot.media_chain,
                @intFromPtr(slot),
                slot.skip_prefix_cache,
            ) else if (slot.glm5_request) |*request| hc.lookupAndRestoreGlm(
                &slot.cache,
                &slot.moe_seq_offset,
                request,
                xfm_ptr.s,
                slot.full_prompt,
                slot.has_tools,
                slot.vision_key,
                slot.media_start,
                slot.media_chain,
                dfl_restore,
                @intFromPtr(slot),
                slot.skip_prefix_cache,
            ) else hc.lookupAndRestoreWithMedia(
                &slot.cache,
                &slot.moe_seq_offset,
                slot.ssm_entries,
                xfm_ptr.s,
                slot.full_prompt,
                slot.has_tools,
                slot.vision_key,
                slot.media_start,
                slot.media_chain,
                dfl_restore,
                if (mtp_kv) |k| .{ .cache = k, .base_pos = &mtp_base, .head = mtp_head } else null,
                @intFromPtr(slot),
                slot.skip_prefix_cache,
            )) catch |err| blk: {
                log.warn("[hot-cache] lookup failed: {s} — proceeding with cold prefill\n", .{@errorName(err)});
                break :blk prefix_cache_mod.LookupResult{ .matched = 0, .full_match = false };
            };
            var restore_line_buf: [192]u8 = undefined;
            if (prefix_cache_mod.restoreLogLine(
                &restore_line_buf,
                if (xfm_ptr.dsv41 != null) "[dsv41-prefix]" else "[prefix]",
                lookup.matched,
                slot.full_prompt.len,
                if (lookup.matched == 0) "nothing" else if (hc.last_restored_live) "the live state" else if (hc.last_restored_disk_id != null) "SSD" else "RAM",
                lookup_sw.read() / std.time.ns_per_ms,
            )) |line| log.info("{s}\n", .{line}) else |_| {}
            if (lookup.matched > 0 and lookup.matched <= slot.full_prompt.len) {
                hot_matched = @intCast(lookup.matched);
                prefill_tokens = slot.full_prompt[hot_matched..];
                hot_checked_out = lookup.ownsRestoredRows();
                slot.restored_entry = lookup.entry_id;
            }
            // The entry this request restored off may be evicted before it commits, so the fork
            // keeps its own restore point; a full reuse is covered by the prompt end.
            if (hot_matched >= prefix_cache_mod.RING_RESTORE_MIN_TOKENS and !lookup.full_match and slot.ring_cps.fork == null) {
                slot.ring_cps.fork = slot.cache.ringCheckpoint(hot_matched, xfm_ptr.s) catch |err| blk: {
                    log.warn("[hot-cache] ring checkpoint at the restore failed: {s}\n", .{@errorName(err)});
                    break :blk null;
                };
            }
            if (dfl_target) |*dc| {
                // Adopt only a context that lines up EXACTLY with the trunk
                // cursor — `nextDflash` asserts `absLen() == cache.step`, and
                // a blind start is always a valid fallback.
                if (lookup.dflash_base != null and dfl_base + dc.cache.step == hot_matched) {
                    dc.base_pos = dfl_base;
                    dflash_restored = dc.*;
                } else {
                    dc.deinit();
                }
                dfl_target = null;
            }
            if (mtp_target) |*mc| {
                // Same exact-alignment rule (the Generator asserts
                // `base + step == ssm_cp_offset` on adoption).
                if (lookup.mtp_base != null and mtp_base + mc.step() == hot_matched) {
                    mtp_restored = .{ .cache = mc.*, .base = mtp_base };
                } else {
                    mc.deinit();
                }
                mtp_target = null;
            }
        }
    }

    // Placeholder rows the restored prefix already holds: their embeddings are never read, so the
    // media is encoded only now that the restore is known.
    const vision_consumed: usize = if (hot_matched > 0)
        prefill_plan.countSpliceRows(@ptrCast(slot.full_prompt[0..hot_matched]), xfm_ptr.config.image_token_id, xfm_ptr.config.video_token_id)
    else
        0;
    var vision_skipped: usize = 0;
    if (slot.deferred_media) |d| {
        if (slot.vision_embeddings) |old| {
            freeVisionArray(old);
            slot.vision_embeddings = null;
        }
        const encoded = encodeMediaAfterRestore(sch, slot.model, d, vision_consumed, &slot.media_refusal) catch |err| {
            if (err == error.MediaRefused) log.warn("[scheduler] media refused: {s}\n", .{if (slot.media_refusal) |*r| r.text() else "no text"});
            return err;
        };
        slot.vision_embeddings = encoded.embeddings;
        vision_skipped = encoded.skipped_rows;
        slot.ctx.vision_embeddings = slot.vision_embeddings;
    }

    // Phase 1 (perf-plan): forward the SSM-checkpoint stride from the
    // LoadedModel so the prefill loop snapshots SSM state at stride-aligned
    // positions for hybrid archs. Plain-attn models have empty ssm_entries
    // and ignore the stride entirely (no-op even at stride > 0). When the
    // hot prefix cache is disabled or off, set stride to 0 to skip
    // snapshot work that would just be discarded.
    var cp_stride: u32 = if (slot.model.prefix_cache != null) slot.model.ssm_checkpoint_stride else 0;
    var cp_max: u32 = slot.model.ssm_checkpoint_max;

    // Evict the hot cache to admit (#353). Here, not on the connection thread: the inference
    // thread is the sole mlx caller and the restore has already happened, so the entry this
    // request uses is the MRU one. The estimator is re-asked after every eviction.
    // `admitted_prefill_chunk` is the pre-eviction width (0 = no pass ran); the width is
    // re-asked after the pass against the live delta.
    var admitted_prefill_chunk: u32 = 0;
    var evicted_live_bytes: u64 = 0;
    // `publishHotCacheResidency` is not gated.
    if (admissionPassArmed(slot.model.config)) if (prefill_admission_fits) |fits_fn| {
        if (slot.model.config) |cfg| {
            // Bills the request's own kv-quant scheme and vision chunking, not the process defaults.
            const Probe = struct {
                cfg: *const model_mod.ModelConfig,
                seq: usize,
                max_tokens: u32,
                kv_cfg: transformer_mod.KVQuantConfig,
                unchunked: bool,
                /// The prefix the hot cache restored and the capacity its buffers hold: already
                /// inside active memory, so billing them again invents a copy.
                warm_matched: u64,
                warm_capacity: u64,
                /// Only a checked-out restore is credited: a refcount share is copied by the first append.
                warm_will_donate: bool,
                enable_mtp: bool,
                /// The KDA checkpoints a GLM request keeps: billed here only, and fewer where they do not fit.
                checkpoints: CheckpointPlan,
                fits: *const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, u64, u64, bool, bool, CheckpointPlan) bool,
                fn call(ctx: ?*anyopaque) bool {
                    const self: *@This() = @ptrCast(@alignCast(ctx.?));
                    return self.fits(self.cfg, self.seq, self.max_tokens, self.kv_cfg, self.unchunked, self.warm_matched, self.warm_capacity, self.warm_will_donate, self.enable_mtp, self.checkpoints);
                }
                fn keeping(ctx: ?*anyopaque, n: u32) bool {
                    const self: *@This() = @ptrCast(@alignCast(ctx.?));
                    self.checkpoints.keep = n;
                    return call(ctx);
                }
            };
            var probe = Probe{
                .cfg = cfg,
                .seq = slot.full_prompt.len,
                .max_tokens = slot.max_tokens,
                .kv_cfg = slot.cache.config,
                .unchunked = prefill_plan.visionPrefillUnchunked(slot.vision_embeddings != null),
                .warm_matched = hot_matched,
                .warm_capacity = slot.cache.residentCapacityTokens(),
                .warm_will_donate = hot_checked_out,
                .enable_mtp = slot.enable_mtp,
                .checkpoints = .{
                    .keep = if (cfg.isGlm5() and cp_stride > 0) cp_max else 0,
                    .released = glmCheckpointsRelease(slot, slot.full_prompt.len - hot_matched),
                    .thought = thoughtPointWanted(slot, cp_stride > 0),
                },
                .fits = fits_fn,
            };
            var fits = Probe.call(&probe);
            // A shared restore is billed a whole second copy; taking the entry over moves it instead.
            if (!fits and !hot_checked_out and hot_matched > 0) if (slot.model.prefix_cache) |*hc| {
                if (hc.checkoutRestored(@intFromPtr(slot), slot.full_prompt.len)) {
                    hot_checked_out = true;
                    probe.warm_will_donate = true;
                    fits = Probe.call(&probe);
                }
            };
            if (!fits) {
                // The width admission was billed at, read before anything is evicted.
                if (prefill_request_chunk) |pick_pre| {
                    admitted_prefill_chunk = pick_pre(cfg, slot.full_prompt.len, slot.max_tokens, slot.cache.config, probe.unchunked, probe.warm_matched, probe.warm_capacity, probe.warm_will_donate, probe.enable_mtp);
                }
                // Per-model, off the slot: `sch.hot_prefix_cache` is whichever model loaded last.
                // Captured by pointer: a `|hc|` capture would evict from a stack copy.
                const report = if (slot.model.prefix_cache) |*hc|
                    // Never evict the entry this request restored from: its buffers are shared.
                    hc.evictLruToAdmit(slot.full_prompt.len, &probe, Probe.call, true)
                else
                    prefix_cache_mod.EvictionReport{ .admitted = false };
                publishHotCacheResidency(sch);
                // What the allocator returned, not what the cache was billed for.
                evicted_live_bytes = report.bytes;
                const kept: ?u32 = if (!report.admitted and cfg.isGlm5() and cp_stride > 0)
                    fewerCheckpointsToAdmit(cp_max, &probe, Probe.keeping)
                else
                    null;
                if (kept) |n| {
                    log.info("[hot-cache] GLM request keeps {d} of {d} KDA checkpoints: the rest do not fit beside its {d}-token prefill\n", .{ n, cp_max, slot.full_prompt.len });
                    cp_max = n;
                    if (n == 0) cp_stride = 0;
                }
                if (!report.admitted and kept == null) {
                    log.warn("[scheduler] prefill refused: {d} tokens do not fit even with an empty hot cache\n", .{slot.full_prompt.len});
                    if (prefill_admission_refused_log) |report_fn| {
                        slot.memory_context_refusal = report_fn(cfg, slot.full_prompt.len, slot.max_tokens, slot.cache.config, probe.unchunked, probe.warm_matched, probe.warm_capacity, probe.warm_will_donate, probe.enable_mtp, probe.checkpoints);
                    }
                    // Not `error.OutOfMemory` (the MLX latch's name, a 503): this is a request the
                    // machine cannot hold, a named 400.
                    return error.PrefillDoesNotFit;
                }
            }
        }
    };

    // The prefill width for this request, chosen after the admission pass evicted. Falls
    // back to the load-time pin without the hook or the arch opt-in.
    const req_prefill_chunk: u32 = if (slot.model.config) |cfg| blk: {
        const pin = cfg.pinned_prefill_chunk;
        const pick = prefill_request_chunk orelse break :blk pin;
        const reasked = pick(
            cfg,
            slot.full_prompt.len,
            slot.max_tokens,
            slot.cache.config,
            prefill_plan.visionPrefillUnchunked(slot.vision_embeddings != null),
            hot_matched,
            slot.cache.residentCapacityTokens(),
            hot_checked_out,
            slot.enable_mtp,
        );
        const decision = postEvictionPrefillChunk(admitted_prefill_chunk, reasked);
        if (decision.widened) {
            log.info("[prefill] re-ask: width {d} -> {d} after the eviction pass returned {d} MB\n", .{ admitted_prefill_chunk, decision.width, evicted_live_bytes >> 20 });
        } else if (decision.moved) {
            log.warn("[prefill] re-ask: width {d} is NARROWER than the admitted {d} after the eviction pass returned {d} MB — memory moved between the two reads; the re-ask is the live reading, so it runs\n", .{ decision.width, admitted_prefill_chunk, evicted_live_bytes >> 20 });
        }
        break :blk decision.width;
    } else 0;

    // Chunk-boundary decode yields: the hook advances already-decoding
    // streams between this prefill's chunks. Ticks hosted here are billed
    // out of prefill_ns below (the decoding slots got the time).
    var interleave_ctx = InterleaveCtx{ .sch = sch, .slot = slot, .cancelled = &slot.cancelled, .chunk_sw = io_util.Stopwatch.init(sch.io) };
    var write_through_ctx = WriteThroughCtx{ .slot = slot };
    const write_through_on = writeThroughArmed(slot, prefill_tokens.len);
    cp_max = qwenCheckpointCap(slot, cp_max, write_through_on);
    // Per-chunk prefill width context. Stack-scoped like `interleave_ctx`.
    var width_ctx = ChunkWidthCtx{
        .sch = sch,
        .cfg = slot.model.config,
        .kv_bits = if (slot.cache.config.scheme == .off) 16 else slot.cache.config.bits,
        .hc = if (slot.model.prefix_cache) |*p| p else null,
    };

    // A ringed cache keeps restore points only where its prefill passes: mark the message starts,
    // where a new session sharing this one's preamble diverges.
    if (slot.model.prefix_cache != null and slot.cache.swa_ring_window > 0) {
        var at: [prefix_cache_mod.RING_MARKS_MAX]usize = undefined;
        const positions = if (slot.model.tokenizer.?.specialTokenId(prefix_cache_mod.RING_MARK_TOKEN)) |im_start|
            prefix_cache_mod.ringMarkPositions(slot.full_prompt, hot_matched, im_start, &at)
        else
            at[0..0];
        slot.ring_cps.armMarks(&slot.cache, positions);
    }
    slot.cache.ring_marks = slot.ring_cps.markSlice();
    errdefer slot.cache.ring_marks = &.{};

    // Ownership of the restored spec caches transfers AT THE CALL:
    // initWithOptions adopts them and frees them via its own errdefers on
    // any failure past adoption (a mid-prefill disconnect throws
    // error.Cancelled from its chunk loop). MtpCacheRef/DflashCtx hold the
    // KVCache BY VALUE, so a second deinit from our errdefers walked freed
    // mlx handles — SIGSEGV in freeKVEntry (issue #266). Clear FIRST.
    const dflash_pass = takeOwned(dflash_mod.DflashCtx, &dflash_restored);
    const mtp_pass = takeOwned(generate_mod.MtpRestored, &mtp_restored);
    // Restore by move, the transfer: the last point at which nothing has written to `slot.cache`.
    // Everything above can still refuse and `releaseCheckout` then hands the entry back whole;
    // nothing between here and `initWithOptions` may fail.
    if (slot.model.prefix_cache) |*hc| hc.donateCheckout(@intFromPtr(slot));
    if (hot_checked_out) slot.cache.adoptRestored();
    var gen = try Generator.initWithOptions(
        sch.io,
        slot.allocator,
        xfm_ptr,
        slot.model.tokenizer.?,
        prefill_tokens,
        slot.max_tokens,
        sampling,
        slot.eos_token_ids,
        .{
            .drafter_block_size = slot.drafter_block_size,
            .dflash_enabled = use_dflash,
            .dflash = if (use_dflash) slot.dflash else null,
            // The dflash-resolved block rides the shared drafter_block_size.
            .dflash_block_size = slot.drafter_block_size,
            // Use the resolved thinking mode and normalize the M5/block-16
            // calibration to this machine's effective assistant width.
            .mtp_enabled = use_head_mtp,
            .mtp_acceptance = mtp_plan.mtpAcceptanceFor(slot.model.config.?.mtp_acceptance_override).value,
            .mtp_greedy_tail = mtp_plan.mtpGreedyTailFor(slot.model.config.?.mtp_greedy_tail_override).value,
            .mtp = if (use_head_mtp) slot.mtp else null,
            // The model's head before this request's opt-out (`entry.mtp` already ANDs `--no-mtp`).
            .model_has_mtp = slot.mtp != null,
            .mtp_depth = slot.mtp_depth,
            .lookup_prompt = slot.full_prompt,
            .ctx = slot.ctx,
            // Regular path: skip the lazy preforward so cache.step lands at
            // exactly prompt_len with t1 NOT in cache. Generator.next's
            // transition shim sync-forwards [t1] on the first decode call.
            // DFlash/MTP init paths already skip preforward unconditionally.
            .skip_lazy_preforward = !use_head_mtp and !use_dflash,
            .dspark_enabled = use_dspark,
            .ssm_checkpoint_stride = cp_stride,
            .ssm_checkpoint_max = cp_max,
            .ssm_checkpoint_pos_offset = hot_matched,
            // The encoded rows start at the first block the restore does not cover: the splice
            // resumes at the placeholder rows the restored prefix holds of them.
            .vision_rows_before = if (slot.vision_embeddings != null) vision_consumed - vision_skipped else 0,
            // The width the admission guard billed for this request; the forward can never run wider.
            .pinned_prefill_chunk = req_prefill_chunk,
            .dflash_ctx_restored = dflash_pass,
            .mtp_cache_restored = mtp_pass,
            // Abandoned-prefill abort: the conn thread sets slot.cancelled
            // when the client disconnects; the chunk loop checks it between
            // chunks so a ghost 40K prefill stops within one chunk.
            .cancel_flag = &slot.cancelled,
            // Salvage sink: on abort the Generator moves its captured SSM
            // stride checkpoints here (they die with the failed
            // construction otherwise) so the cancelled-prefill commit can
            // restore hybrids too.
            .cancelled_checkpoint_sink = &slot.cancelled_prefill,
            .prefill_progress = if (observe) &sch.inflight_prefill_tokens else null,
            .prefill_expected = if (observe) &sch.inflight_prefill_expected else null,
            .interleave_hook = if (prefillInterleaveEnabled())
                .{ .ctx = &interleave_ctx, .call = interleaveDecodeTickCb }
            else
                null,
            .write_through_hook = if (write_through_on)
                .{ .ctx = &write_through_ctx, .call = prefillWriteThroughCb }
            else
                null,
            .chunk_width_hook = if (prefill_chunk_adapt != null)
                .{ .ctx = &width_ctx, .call = chunkWidthCb, .confirm = chunkWidenConfirmCb }
            else
                null,
            // The install above is process-wide, so it is not the arch gate.
            .adaptive_chunk_width = adaptiveChunkWidthFor(width_ctx.cfg),
            // Init's argmax-only gate must see logprobs BEFORE the split-
            // prefill final-token forward runs — a post-init field write is
            // too late for the certified lm_head prune.
            .logprobs_n = slot.logprobs_n,
        },
    );
    slot.cache.ring_marks = &.{};
    slot.ring_cps.keepCompleteMarks(&slot.cache);
    slot.prefill_interleaved_ns = interleave_ctx.decode_ns;
    gen.timeout_ns = slot.timeout_ns;
    gen.logprobs_n = slot.logprobs_n;

    slot.legacy_gen = gen;
    armDsv41ThoughtPoint(slot, &slot.legacy_gen.?, write_through_on and cp_stride > 0);
    armThoughtPoint(slot, &slot.legacy_gen.?, cp_stride > 0);
    publishHandoverToken(slot, &slot.legacy_gen.?);
    // A long reply compacts the ring past the prompt end, where the next turn diverges.
    if (slot.model.prefix_cache != null and slot.ring_cps.prompt_end == null) {
        slot.ring_cps.prompt_end = slot.cache.ringCheckpoint(slot.full_prompt.len, xfm_ptr.s) catch |err| blk: {
            log.warn("[hot-cache] ring checkpoint failed: {s}; the entry restores at its end only\n", .{@errorName(err)});
            break :blk null;
        };
    }
    // The last chunks' transient is freed AFTER the loop's own per-chunk clear, so it
    // parks in MLX's pool up to the cap and the first decode tick allocates on top of
    // it. Returned once here, at the handover — long-context gate only.
    if (slot.legacy_gen) |*g| g.clearPoolBeforeDecode();
    // The conn thread's `cached_tokens` counted against `xfm.cache` (legacy
    // global cache) which the slot doesn't use. The slot's `cached_tokens`
    // is the hot-cache match (or 0 if the hot cache missed / isn't
    // configured) — those are the only tokens actually present in slot.cache
    // before this turn's prefill. With this override, slot.prompt_tokens
    // reports the full prompt length: gen.prompt_tokens (= prefill_tokens.len)
    // covers the un-cached tail, and `hot_matched` covers the restored prefix.
    slot.cached_tokens = hot_matched;
    slot.prompt_tokens = gen.prompt_tokens + slot.cached_tokens;
    slot.state = .decoding;
    if (hot_matched == 0) {
        const cold_pos = if (slot.moe_seq_offset > 0) slot.moe_seq_offset else slot.cache.step;
        _ = restore_dump.dumpRestoreIfEnabled(&slot.cache, slot.ssm_entries, xfm_ptr.s, .{
            .kind = "cold",
            .pos = cold_pos,
            .source = "cold",
        });
    }
}

/// Sum the in-flight generated tokens over the active slots for the live-tok/s
/// gauge, EXCLUDING any slot that already finished/cancelled/errored this tick
/// (its tokens were counted into `generation_tokens_total` by `finishSlot`, so
/// including them here would double-count — and would leave a stale non-zero
/// aggregate once the last slot finishes, breaking the "at rest ⇒ live == total"
/// invariant). Generic over the slot type so the exact filter is unit-testable
/// with a lightweight stub; the real caller passes `[]*Slot`.
fn sumInflightGeneratedTokens(active: anytype) u64 {
    var inflight: u64 = 0;
    for (active) |s| {
        if (s.finished or s.error_code != null or s.cancelled.load(.acquire)) continue;
        inflight += @as(u64, s.completion_tokens);
    }
    return inflight;
}

pub fn runDecodeTick(sch: *Scheduler, active: []*Slot) !void {
    if (active.len == 0) return;

    // Publish the in-flight generated-token aggregate for the gauge sampler.
    // Runs after the whole tick (all decode work + any finishSlot calls) on
    // every return path — race-free (inference thread owns these slots' fields)
    // and O(active), never per token. Finished slots are excluded, so the last
    // slot's completion drives this to 0 (live == total at rest).
    defer sch.inflight_generated_tokens.store(sumInflightGeneratedTokens(active), .monotonic);

    // Contention discipline for the spec cost model's kv term: it learns
    // from realized round times, and contention only ever ADDS time. Rather
    // than try to correct for it, a busy server simply stops sampling.
    for (active) |s| {
        if (s.legacy_gen) |*g| g.spec_cost_solo = active.len == 1;
    }

    if (Planner.enabled()) {
        for (active) |slot| if (slot.legacy_gen) |*gen| {
            gen.mtp_planner_width = null;
            gen.mtp_planner_probe = false;
            gen.mtp_planner_recovering = false;
            slot.planner_plain_transition = false;
            slot.planner_force_plain = false;
        };
        if (try tryPlannerTick(sch, active)) return;
        for (active) |slot| if (slot.legacy_gen) |*gen| {
            if (gen.mtp_planner_owned) slot.planner_force_plain = true;
        };
    }
    defer if (Planner.enabled()) for (active) |slot| {
        slot.planner_force_plain = false;
        if (slot.legacy_gen) |*gen| gen.mtp_planner_pending = false;
    };

    // Phase 3 gate: at len==1, route to legacy single-slot path. Bit-identical
    // to pre-Phase-2 behavior including PLD/drafter speculative decoding.
    // Phase A7 test hook: `SUSHI_FORCE_BATCHED=1` bypasses the gate so the
    // byte-equivalence test can run the batched kernel at active.len==1 and
    // assert it matches the single-slot path token-for-token.
    if (active.len == 1 and !sch.force_batched) {
        try runSingleDecodeTick(sch, active[0]);
        return;
    }

    // active.len >= 2 (or force_batched at len==1): split into batchable +
    // non-batchable. Batchable slots share one `forwardBatchedDecode` call.
    // Non-batchable (slots running PLD/drafter or grammar-constrained) fall
    // back to legacy single-slot decode this tick.
    //
    // Plan 05 Phase D: batched decode requires all participating slots to
    // share a transformer. We partition `batchable` by `slot.model` and
    // emit one batched call per model. The non-batchable bucket doesn't
    // care — each slot runs against its own `slot.model.transformer.?`.
    var batchable_buf: [MAX_BATCH_GROUP]*Slot = undefined;
    var batchable_n: usize = 0;
    var mtp_buf: [MAX_BATCH_GROUP]*Slot = undefined;
    var mtp_n: usize = 0;
    var glm_buf: [MAX_BATCH_GROUP]*Slot = undefined;
    var glm_n: usize = 0;
    for (active) |s| {
        const why = sch.batchVerdict(s);
        if (why == .ok and batchable_n < batchable_buf.len) {
            batchable_buf[batchable_n] = s;
            batchable_n += 1;
        } else if (why == .spec_active and (slotMtpGroupable(s) or slotMimoMtpCrowdable(s)) and mtp_n < mtp_buf.len) {
            mtp_buf[mtp_n] = s;
            mtp_n += 1;
        } else if (why == .spec_active and slotGlmRowable(s) and glm_n < glm_buf.len) {
            glm_buf[glm_n] = s;
            glm_n += 1;
        } else {
            // legacy single-slot for spec / grammar / overflow
            noteSerial(sch, s, why);
            try runSingleDecodeTick(sch, s);
        }
    }
    // A crowded MTP group (measured past 3 slots on the 27B) beats its sub-grouped verify
    // rounds with ONE plain batched tick; those slots keep their head resumable by
    // capturing the hidden the tick forwards (`mtp_plain_tick`).
    var mtp_group_n: usize = 0;
    {
        std.sort.pdq(*Slot, mtp_buf[0..mtp_n], {}, struct {
            fn lt(_: void, a: *Slot, b: *Slot) bool {
                return @intFromPtr(a.model) < @intFromPtr(b.model);
            }
        }.lt);
        var i: usize = 0;
        while (i < mtp_n) {
            var j = i + 1;
            while (j < mtp_n and mtp_buf[j].model == mtp_buf[i].model) j += 1;
            if (j - i >= mtpCrowdThresholdFor(mtp_buf[i]) and batchable_n + (j - i) <= batchable_buf.len and !anyPenalised(mtp_buf[i..j])) {
                for (mtp_buf[i..j]) |slot| {
                    const gen = &slot.legacy_gen.?;
                    gen.mtpDetachHead(slot.allocator, true) catch |e| {
                        slot.markError(@errorName(e));
                        continue;
                    };
                    gen.mtp_group_cap = 0;
                    slot.mtp_plain_tick = true;
                    batchable_buf[batchable_n] = slot;
                    batchable_n += 1;
                }
            } else {
                for (mtp_buf[i..j]) |slot| {
                    mtp_buf[mtp_group_n] = slot;
                    mtp_group_n += 1;
                }
            }
            i = j;
        }
    }
    try runMtpGroups(sch, mtp_buf[0..mtp_group_n]);
    // A DFlash2 GLM slot in company decodes as a plain row of the grouped tick (a row for another
    // request beats a draft row, glm5-arch#concurrency); alone it drafts.
    for (glm_buf[0..glm_n]) |slot| {
        if (glmRowsInCompany(slot, glm_buf[0..glm_n], batchable_buf[0..batchable_n]) and batchable_n < batchable_buf.len) {
            batchable_buf[batchable_n] = slot;
            batchable_n += 1;
        } else try runSingleDecodeTick(sch, slot);
    }
    if (batchable_n == 0) {
        if (sch.metrics) |m| m.batched_group_size.set(0);
        return;
    }

    // Group batchable slots by model pointer (in-place partition by sort).
    // The order of slots within a model doesn't matter for batched decode;
    // we only need contiguous runs per model.
    std.sort.pdq(*Slot, batchable_buf[0..batchable_n], {}, struct {
        fn lt(_: void, a: *Slot, b: *Slot) bool {
            return @intFromPtr(a.model) < @intFromPtr(b.model);
        }
    }.lt);

    var start: usize = 0;
    while (start < batchable_n) {
        var end = start + 1;
        while (end < batchable_n and batchable_buf[end].model == batchable_buf[start].model) end += 1;
        var group = batchable_buf[start..end];
        const row_cap = if (group[0].model.config) |c| batchGroupCap(c) else MAX_BATCH_GROUP;
        if (group.len > row_cap) {
            for (group[row_cap..]) |s| {
                noteSerial(sch, s, .row_cap);
                try runSingleDecodeTick(sch, s);
            }
            group = group[0..row_cap];
        }
        // One predicate for both halves of the pad-waste change: the kv-length rule and the sort.
        const gate_batch_kv_len = if (group[0].model.config) |c| c.longCtxGated() else false;
        // Per-row attention reads each slot's own cache: nothing pads.
        const pads = if (group[0].model.transformer) |t| !t.supportsBatchedMimoDecode() and !t.supportsBatchedGlmRows() and !t.supportsBatchedQwen4Rows() else true;
        // Cap the group by padding waste: the batched kernel pads every slot's
        // KV to the longest in the group, so one long-context stream would make
        // its short neighbours build a tensor orders of magnitude bigger than
        // they need. Sort ascending by kv_len and let `batchedKvKeepCount` say
        // how many still fit; the tail decodes serially this tick.
        if (pads and group.len >= 2) {
            var kv_lens: [32]u32 = undefined;
            const keep = padWasteKeepCount(group, gate_batch_kv_len, &kv_lens);
            if (keep < group.len) {
                if (!kv_skew_split_logged) {
                    kv_skew_split_logged = true;
                    log.info("[batched] pad-waste cap: kept {d} of {d} slots (waste {d:.2}x, kv_len {d}..{d}), rest serial\n", .{
                        keep,
                        group.len,
                        batchedPadWaste(kv_lens[0..group.len]),
                        kv_lens[0],
                        kv_lens[group.len - 1],
                    });
                }
                for (group[keep..]) |s| {
                    noteSerial(sch, s, .pad_waste);
                    try runSingleDecodeTick(sch, s);
                }
                group = group[0..keep];
            }
        }
        // Honor force_batched even when only one slot is batchable so the
        // test hook actually exercises forwardBatchedDecode at N=1.
        if (group.len >= 2 or (sch.force_batched and group.len == 1)) {
            if (sch.metrics) |m| m.batched_group_size.set(group.len);
            try runBatchedDecodeTick(sch, group);
        } else if (group.len == 1) {
            try runSingleDecodeTick(sch, group[0]);
        }
        start = end;
    }
}

/// What `runPrefill` arms in a slot's Generator init options.
pub const SpecInitWiring = struct {
    use_mtp: bool,
    use_dflash: bool,
};

/// Per-site spec gating for one slot, as a pure function.
///
/// Speculative decode must be able to ROLL BACK a rejected tail, and the shell
/// rolls back what it owns: the slot's KVCache and ssm_entries. A MODULE-OWNED
/// arch uses neither — by the time a verify forward returns, its own state has
/// already absorbed every draft token, and the shell's snapshot/truncate run
/// over an empty entries array. So every shell-driven spec mode is off there.
pub fn specInitWiring(
    owns_module_state: bool,
    module_spec_rollback: bool,
    enable_mtp: bool,
    has_mtp: bool,
    enable_drafter: bool,
    has_dflash: bool,
) SpecInitWiring {
    if (owns_module_state) {
        // A module-owned arch that CAN rewind its state across a verify runs
        // a DFlash2 drafter or its own MTP head like any other target (qwen4_exp:
        // `ssmRollbackFromCapture` rewinds GDN, PLE and QSA state). Without
        // rollback every spec mode stays off.
        const use_dflash = module_spec_rollback and enable_drafter and has_dflash;
        return .{
            .use_mtp = !use_dflash and module_spec_rollback and enable_mtp and has_mtp,
            .use_dflash = use_dflash,
        };
    }
    // enable_drafter is the request-level "assistant sidecar" switch for BOTH
    // sidecar kinds; the loader guarantees at most one of drafter/dflash is
    // loaded per model. Priority: dflash > MTP — a
    // loaded DFlash sidecar is an explicit choice (`--drafter` or the pack's
    // own `drafter/`) while the MTP head ships with the checkpoint;
    // `--no-drafter` (or `enable_drafter:false`) hands the round back to MTP.
    const use_dflash = enable_drafter and has_dflash;
    const use_mtp = !use_dflash and enable_mtp and has_mtp;
    return .{
        .use_mtp = use_mtp,
        .use_dflash = use_dflash,
    };
}

/// Which spec mode a decode tick drives for a slot.
pub const SpecTickMode = enum { mtp, dflash, regular };

/// Pure decode-tick dispatch decision. The slot flags carry the REQUEST's
/// wish; the generator-side values carry what `Generator.initWithOptions`
/// actually armed. Every arm must require BOTH.
pub fn specTickMode(
    slot_enable_mtp: bool,
    gen_has_mtp: bool,
    slot_enable_drafter: bool,
    gen_has_dflash: bool,
) SpecTickMode {
    if (slot_enable_drafter and gen_has_dflash) return .dflash;
    if (slot_enable_mtp and gen_has_mtp) return .mtp;
    return .regular;
}

/// Drive one Generator step (regular / DFlash / MTP) and push emitted
/// tokens into the slot's output ring. Mirrors the existing
/// `StreamingTokenStream` adapter contract: 0..N tokens per call, with EOS
/// stopping the slot but NOT being emitted.
fn plannerOutputClock(slot: *Slot, gen: *Generator) void {
    const now = gen.timer.read();
    slot.mtp_publish_gap_ms = if (slot.mtp_publish_ns > 0 and now >= slot.mtp_publish_ns) @as(f32, @floatFromInt(now - slot.mtp_publish_ns)) / std.time.ns_per_ms else 0;
    slot.mtp_publish_ns = now;
    if (Planner.enabled() and gen.mtp_planner_owned) gen.mtp_planner_max_gap_ms = @max(gen.mtp_planner_max_gap_ms, slot.mtp_publish_gap_ms);
}

pub fn publishSpeculativeBlock(sch: *Scheduler, slot: *Slot, gen: *Generator, tokens: []const u32) void {
    if (publishBlockTokens(slot, gen, tokens)) |reason| finishSlot(sch, slot, reason);
}

/// Publishes `tokens` one by one; the result is the reason the block ended the slot, null when it did not.
fn publishBlockTokens(slot: *Slot, gen: *Generator, tokens: []const u32) ?[]const u8 {
    // Ahead of the block: a reader handed any of its tokens already sees where the payload starts.
    if (gen.takeConstraintSpan()) |span| slot.publishConstraintSpan(span.token_index, span.byte_offset);
    if (gen.mtp != null and tokens.len > 0) {
        plannerOutputClock(slot, gen);
    }
    // The Generator has already committed the whole block internally. Publish
    // it incrementally so streaming, cancellation, EOS and usage all observe
    // the same per-token boundary. The generator-side cap guarantees max_tokens
    // cannot land before the returned block ends.
    for (tokens, 0..) |t, i| {
        if (slot.cancelled.load(.acquire)) {
            if (gen.mtp != null) log.debug("[mtp-publish] cancelled after round: discarded={d}\n", .{tokens.len - i});
            return null;
        }
        if (generate_mod.isEosId(t, slot.eos_token_ids)) {
            slot.stop_token = t;
            slot.stop_path = "spec";
            return "stop";
        }
        slot.pushToken(t);
        slot.completion_tokens += 1;
        if (t != 0) slot.was_pad_only = false;
        if (slot.completion_tokens >= slot.max_tokens) return "length";
    }
    std.debug.assert(slot.completion_tokens == gen.completion_tokens);
    return null;
}

/// Every single-slot decode tick funnels through here, so a decode-time MLX failure is
/// attributed to the slot whose forward raised it instead of the next request's prefill.
/// The repetition-loop guard every decode tick runs first. True = the slot was finished here.
pub fn loopGuardTick(sch: *Scheduler, slot: *Slot, gen: *Generator) !bool {
    // Stop a runaway repetition loop before generating more. Some models
    // collapse after a large/confusing tool result into spamming
    // one short cycle — e.g. the thinking opener — forever;
    // with no repeat penalty by default and a generous max_tokens, nothing else
    // halts it until the cap. Checked here, before this tick's step, so it
    // covers the regular, PLD, and drafter paths uniformly.
    const loop_guard_start = gen.loopGuardStart();
    if (loopStopDecision(gen.generated_ids.items[loop_guard_start..])) |relative_stop| {
        if (gen.hasPrePayloadConstraint()) {
            switch (try gen.forceConstraintTransition(slot.allocator)) {
                .committed => |boundary| {
                    slot.pushToken(boundary);
                    slot.completion_tokens += 1;
                    std.debug.assert(slot.completion_tokens == gen.completion_tokens);
                    if (boundary != 0) slot.was_pad_only = false;
                    log.warn("[grammar] reasoning boundary forced after repetition loop at {d} generated tokens\n", .{gen.generated_ids.items.len - 1});
                    return true;
                },
                .activated => return true,
                // The completion cap won the race with recovery: fall through
                // to the ordinary loop-stop cut.
                .refused => {},
            }
        }
        if (try closeThoughtOnLoop(sch, slot, gen)) return true;
        var stop = relative_stop;
        stop.trim_start += loop_guard_start;
        // Never cut silently: the 2026-07-14 php.html post-mortem took log
        // archaeology because this guard left no trace of having fired. The
        // tier and the trim point are logged too — five cuts in a row is a
        // different diagnosis from one, and the trim is what breaks the chain.
        log.warn("[loop-stop] degenerate tail loop cut after {d} generated tokens (finish_reason={s} details={s} tier={s} trim_start={d})\n", .{
            gen.generated_ids.items.len, stop.finish_reason, stop.finish_details,
            @tagName(stop.tier),         stop.trim_start,
        });
        slot.finish_details = stop.finish_details;
        if (loopTrimEnabled()) slot.loop_trim_start = stop.trim_start;
        finishSlot(sch, slot, stop.finish_reason);
        return true;
    }
    return thinkBoundTick(sch, slot, gen);
}

pub const LoopAction = enum { cut, close_thought };

/// A loop convicted INSIDE an open think block is recoverable once per request: the
/// bound's early-stop line and closer are committed and the model answers. Anything else
/// (answer-side loop, a second loop, no room before `max_tokens`) keeps the ordinary cut.
pub fn loopRecoveryAction(tb: *mtp_plan.ThinkBound, ids: []const u32, completion_tokens: u32, max_tokens: u32, recovered: bool) LoopAction {
    tb.observe(ids);
    if (recovered or tb.fired or !tb.in_think) return .cut;
    if (!generate_mod.forcedBoundaryCanContinue(completion_tokens, max_tokens, tb.forced.len + 1)) return .cut;
    return .close_thought;
}

/// The thought a loop was convicted in is closed instead of the turn ending. The closing line
/// is built here, not at request start: a request carrying a `think_bound` is refused DFlash2.
/// True = the slot's tick is spent.
fn closeThoughtOnLoop(sch: *Scheduler, slot: *Slot, gen: *Generator) !bool {
    var lazy: ?mtp_plan.ThinkBound = null;
    defer if (lazy) |b| slot.allocator.free(b.forced);
    const tb: *mtp_plan.ThinkBound = gen.sampling.think_bound orelse blk: {
        const tok = slot.model.tokenizer orelse return false;
        lazy = @import("server.zig").loopCloseBound(slot.allocator, slot.model, tok, slot.full_prompt, slot.enable_thinking) orelse return false;
        break :blk &lazy.?;
    };
    if (loopRecoveryAction(tb, gen.generated_ids.items, gen.completion_tokens, gen.max_tokens, gen.think_closed_by_loop) == .cut) return false;
    log.warn("[loop-recover] repetition loop inside the thought at {d} generated tokens; closing it\n", .{gen.generated_ids.items.len});
    tb.fired = true;
    gen.think_closed_by_loop = true;
    _ = try commitThoughtClose(sch, slot, gen, tb.forced);
    return true;
}

/// A thinking budget at its limit: commit the early-stop line and the closer
/// through the model this tick; an MTP or DSpark slot goes back to speculating, any other spec mode
/// decodes the answer regular (`spec_disable_reason = .think_bound`). True = the slot's tick is spent.
fn thinkBoundTick(sch: *Scheduler, slot: *Slot, gen: *Generator) !bool {
    const tb = gen.sampling.think_bound orelse return false;
    tb.observe(gen.generated_ids.items);
    if (!tb.due()) return false;
    tb.fired = true;
    if (!generate_mod.forcedBoundaryCanContinue(gen.completion_tokens, gen.max_tokens, tb.forced.len + 1)) {
        log.warn("[think-bound] budget {d} reached with no room to close the thought (max_tokens {d})\n", .{ tb.budget, gen.max_tokens });
        return false;
    }
    if (!try commitThoughtClose(sch, slot, gen, tb.forced)) {
        log.info("[think-bound] reasoning budget {d} reached at {d} generated tokens; thought closed\n", .{ tb.budget, gen.generated_ids.items.len });
    }
    return true;
}

/// Commits `forced` (early-stop line, closer, separator) through the model and decodes the rest
/// regular. True = the slot finished on the way.
fn commitThoughtClose(sch: *Scheduler, slot: *Slot, gen: *Generator, forced: []const u32) !bool {
    const r = try gen.commitForcedTokens(slot.allocator, forced);
    defer slot.allocator.free(r.emitted);
    for (r.emitted) |t| pushForcedToken(slot, t);
    std.debug.assert(slot.completion_tokens == gen.completion_tokens);
    if (r.stopped) {
        finishSlot(sch, slot, gen.finish_reason);
        return true;
    }
    if (gen.dspark) switch (try gen.dsparkResumeAfterForced(slot.allocator)) {
        .resumed => |closer| {
            pushForcedToken(slot, closer);
            std.debug.assert(slot.completion_tokens == gen.completion_tokens);
            return false;
        },
        .stopped => {
            finishSlot(sch, slot, gen.finish_reason);
            return true;
        },
        .refused => {},
    } else if (try gen.mtpResumeAfterForced(slot.allocator)) return false;
    gen.spec_disabled_runtime = true;
    gen.spec_disable_reason = .think_bound;
    return false;
}

fn pushForcedToken(slot: *Slot, t: u32) void {
    slot.pushToken(t);
    slot.completion_tokens += 1;
    if (t != 0) slot.was_pad_only = false;
}

/// The request shape that takes a thought point: admission bills the state from this answer, the arming acts on it.
fn thoughtPointWanted(slot: *const Slot, checkpoints_on: bool) bool {
    const cfg = slot.model.config orelse return false;
    if (!checkpoints_on or !slot.has_tools or !slot.enable_thinking) return false;
    if (slot.full_prompt.len < kv_disk_cache.MIN_PERSIST_TOKENS) return false;
    return cfg.isGlm5() or (cfg.isQwen4() and slot.enable_mtp);
}

/// Qwen over MTP and GLM over DFlash2 keep a recurrent state that restores only at a saved point, and a thinking request
/// with tools is re-rendered with its thought: the state right behind the generated `</think>` (on GLM the pool position
/// at or below it) is a restore point the next prompt shares (docs/engine-prefix-cache.md#thought-points).
fn armThoughtPoint(slot: *Slot, gen: *Generator, checkpoints_on: bool) void {
    if (!thoughtPointWanted(slot, checkpoints_on)) return;
    const grid: u32 = if (gen.glm_dflash_native)
        @intCast(glm5_prefix.pool_size)
    else if (gen.mtp != null and !gen.dspark and slot.ssm_entries != null)
        1
    else
        return;
    const tok = slot.model.tokenizer orelse return;
    const open = tok.specialTokenId("<think>") orelse return;
    const close = tok.specialTokenId("</think>") orelse return;
    if (std.mem.indexOfScalar(u32, slot.full_prompt[slot.full_prompt.len - 2 ..], open) == null) return;
    gen.thought_point = .{ .closer = close, .grid = grid, .min_tokens = mtp_plan.thought_point_min_tokens, .base = slot.full_prompt.len };
}

/// A thought the next prompt would re-render differently is no restore point: its state is dropped before the commit.
fn settleThoughtPoint(slot: *Slot, gen: *Generator) void {
    const tp = gen.thought_point orelse return;
    if (tp.state != .taken) return;
    const tok = slot.model.tokenizer orelse return gen.dropThoughtPoint();
    const generated = gen.generated_ids.items;
    const end = std.mem.indexOfScalar(u32, generated, tp.closer) orelse return gen.dropThoughtPoint();
    if (end < 2) return gen.dropThoughtPoint();
    const first = tok.decode(slot.allocator, generated[0..1]) catch return gen.dropThoughtPoint();
    defer slot.allocator.free(first);
    const tail = tok.decode(slot.allocator, generated[end - 2 .. end]) catch return gen.dropThoughtPoint();
    defer slot.allocator.free(tail);
    const style: mtp_plan.CloserStyle = if (gen.glm_dflash_native) .bare_closer else .newline_before_closer;
    if (!mtp_plan.thoughtEdgesSurvive(style, first, tail)) {
        log.info("[prefix] no thought restore point: the template re-renders the thought's edges\n", .{});
        gen.dropThoughtPoint();
    }
}

fn thinkBoundFired(gen: *const Generator) bool {
    if (gen.think_closed_by_loop) return true;
    const tb = gen.sampling.think_bound orelse return false;
    return tb.fired;
}

/// A fired bound decodes plain unless an MTP head or DSpark resumed behind it (DFlash2 still decodes plain).
pub fn thinkBoundForcesPlain(gen: *const Generator) bool {
    return thinkBoundFired(gen) and (gen.spec_disable_reason == .think_bound or (gen.mtp == null and !gen.dspark));
}

/// Drafts a thinking bound lets the slot's next MTP round carry, so it accepts no further than serial decode
/// closes the thought: null = unbounded, 0 = the room's last token runs plain. `observe` first.
pub fn thinkBoundDraftLimit(gen: *const Generator) ?u32 {
    const room = (gen.sampling.think_bound orelse return null).room() orelse return null;
    return mtp_plan.mtpDraftCapForRoom(room) orelse 0;
}

/// A solo or grouped MTP round keeps the tighter of its own cap and the thought room, or runs the
/// last room token serial.
pub fn capMtpRoundToThinkBound(gen: *Generator) void {
    const limit = thinkBoundDraftLimit(gen) orelse return;
    if (limit == 0) {
        gen.mtp_serial_left = 1;
    } else gen.mtp_group_cap = if (gen.mtp_group_cap > 0) @min(gen.mtp_group_cap, limit) else limit;
}

pub fn runSingleDecodeTick(sch: *Scheduler, slot: *Slot) !void {
    var inner_err: ?anyerror = null;
    runSingleDecodeTickInner(sch, slot) catch |e| switch (e) {
        // A reasoning-protocol dead end is this request's failure; the tick
        // error path would fail every active slot.
        error.NoValidProtocolToken, error.InvalidProtocolTransition, error.InvalidProtocolPayload => {
            log.err("[grammar] reasoning protocol failed ({s}); failing this request only\n", .{@errorName(e)});
            slot.markError(@errorName(e));
        },
        else => inner_err = e,
    };
    mlx.checkErrorDecode() catch |mlx_err| {
        log.err("[scheduler] decode aborted: MLX failure mid-generation ({s}) — failing this request, the server keeps serving\n", .{@errorName(mlx_err)});
        slot.markError(@errorName(mlx_err));
        return;
    };
    if (inner_err) |e| return e;
}

fn runSingleDecodeTickInner(sch: *Scheduler, slot: *Slot) !void {
    const gen = if (slot.legacy_gen) |*g| g else {
        slot.markError("no_generator");
        return;
    };

    if (try loopGuardTick(sch, slot, gen)) return;
    if (Planner.enabled() and slot.planner_force_plain) {
        gen.mtp_hidden_stale = true;
    }
    if (slot.model.transformer) |xfm| try xfm.ssmGroupRelease(&slot.ctx);

    // NOTE: no `!gen.spec_disabled_runtime` short-circuit here — the
    // generators handle the disabled fallback internally.
    const tick_mode: SpecTickMode = if ((Planner.enabled() and slot.planner_force_plain) or thinkBoundForcesPlain(gen)) .regular else specTickMode(
        slot.enable_mtp,
        gen.mtp != null or gen.dspark,
        slot.enable_drafter,
        gen.dflash != null,
    );
    if (tick_mode == .mtp and gen.dspark) {
        const result = try gen.nextDspark(slot.allocator);
        if (result == null) {
            finishSlot(sch, slot, gen.finish_reason);
            return;
        }
        defer slot.allocator.free(result.?.tokens);
        publishSpeculativeBlock(sch, slot, gen, result.?.tokens);
        writeDsv41ThoughtPoint(slot, gen, result.?.tokens);
        return;
    }
    if (tick_mode == .mtp) {
        gen.mtp_group_cap = 0;
        capMtpRoundToThinkBound(gen);
        const result = try gen.nextMtp(slot.allocator);
        if (result == null) {
            finishSlot(sch, slot, gen.finish_reason);
            return;
        }
        defer slot.allocator.free(result.?.tokens);
        publishSpeculativeBlock(sch, slot, gen, result.?.tokens);
        return;
    }

    if (tick_mode == .dflash) {
        const result = try gen.nextDflash(slot.allocator);
        if (result == null) {
            finishSlot(sch, slot, gen.finish_reason);
            return;
        }
        defer slot.allocator.free(result.?.tokens);
        publishSpeculativeBlock(sch, slot, gen, result.?.tokens);
        return;
    }

    // Regular path. A request that never armed MTP teaches the round-cost table what a plain
    // serial token costs; `observeSerialTick` owns the drop rules and `serialCellWanted`.
    const tick_on = generate_mod.tickUbenchArmed();
    var tick_sw = if (tick_on) io_util.Stopwatch.init(sch.io) else undefined;
    const tok_opt = try gen.next(slot.allocator);
    const next_ns: u64 = if (tick_on) tick_sw.read() else 0;
    if (tok_opt == null) {
        finishSlot(sch, slot, gen.finish_reason);
        return;
    }
    gen.observeSerialTick();
    if (Planner.enabled()) {
        if (slot.planner_force_plain) gen.mtp_planner_plain_ticks += 1;
        plannerOutputClock(slot, gen);
    }
    const t = tok_opt.?;
    // Phase A5: capture per-token logprob. `gen.last_logprob` ownership
    // transfers into slot.logprobs_buf (gen sets the field, we null it here).
    // Published in the SAME critical section as the token so a streaming
    // reader that has been handed token i can read entry i — see
    // `pushTokenWithLogprob`.
    var lp_take: ?sampler.LogprobResult = null;
    if (slot.logprobs_n > 0) {
        if (gen.last_logprob) |lp| {
            lp_take = lp;
            gen.last_logprob = null;
        }
    }
    // Publish the reasoning→payload boundary (if this is the token that
    // carries it) BEFORE the token itself, so a streaming reader handed token
    // i already sees its span.
    if (gen.takeConstraintSpan()) |span| {
        slot.publishConstraintSpan(span.token_index, span.byte_offset);
    }
    slot.pushTokenWithLogprob(t, lp_take);
    if (t != 0) slot.was_pad_only = false;
    slot.completion_tokens = gen.completion_tokens;
    if (tick_on) {
        const rest_ns = tick_sw.read() - next_ns;
        log.info("[tick-ubench] sched next={d:.3} ms rest={d:.3} ms\n", .{
            @as(f64, @floatFromInt(next_ns)) / 1e6,
            @as(f64, @floatFromInt(rest_ns)) / 1e6,
        });
    }
}

test "DFlash cache payload is committed only when it spans the trunk prefix" {
    try testing.expect(dflashContextCoversPrefix(128, 128));
    try testing.expect(!dflashContextCoversPrefix(96, 128));
}

test "firstMediaPlaceholder finds every dynamic media kind and ignores disabled ids" {
    const tokens = [_]u32{ 0, 11, 22, 33, 44 };
    try testing.expectEqual(@as(?usize, 2), firstMediaPlaceholder(true, &tokens, 22, 0));
    try testing.expectEqual(@as(?usize, 4), firstMediaPlaceholder(true, &tokens, 0, 44));
    try testing.expectEqual(@as(?usize, 2), firstMediaPlaceholder(true, &tokens, 22, 44));
    try testing.expect(firstMediaPlaceholder(true, &tokens, 0, 0) == null);
}

test "cancelled-prefill commit length: floor, clamp, and zero" {
    // A cancelled prefill only pays its way into the LRU once the forwarded
    // prefix is past the chat-template-prologue class (~dozens of tokens);
    // below the floor the entry is pollution, not saved work.
    try testing.expect(cancelledPrefillCommitLen(0, 1000) == null);
    try testing.expect(cancelledPrefillCommitLen(1, 1000) == null);
    try testing.expect(cancelledPrefillCommitLen(255, 1000) == null);
    try testing.expectEqual(@as(?usize, 256), cancelledPrefillCommitLen(256, 1000));
    try testing.expectEqual(@as(?usize, 300), cancelledPrefillCommitLen(300, 1000));
    // cache.step can never exceed the prompt (restore clamps to matched and
    // chunks stop at the tail); clamp defensively anyway so a bad step can
    // never key tokens the KV does not hold.
    try testing.expectEqual(@as(?usize, 1000), cancelledPrefillCommitLen(5000, 1000));
    // A short prompt is not worth an entry at any step.
    try testing.expect(cancelledPrefillCommitLen(50, 50) == null);
}

test "a decode-phase cancel that never passed the cull commits at the cleanup drain, once" {
    const Hook = struct {
        var commits: usize = 0;
        fn commit(_: *Scheduler, _: *Slot) void {
            commits += 1;
        }
    };
    var sch: Scheduler = undefined;
    var slot: Slot = undefined;
    slot.committed_at_cull = false;
    Hook.commits = 0;
    commitCancelledAtCleanup(&sch, &slot, Hook.commit);
    try testing.expectEqual(@as(usize, 1), Hook.commits);
    slot.committed_at_cull = true;
    commitCancelledAtCleanup(&sch, &slot, Hook.commit);
    try testing.expectEqual(@as(usize, 1), Hook.commits);
}

test "idleEvictTickMs: sweeps well inside the window without spinning" {
    // Bar: a quarter of the window, clamped to [1s, 30s] — eviction lands near
    // the configured time without the sweep becoming a busy loop.
    try testing.expectEqual(@as(i64, 5_000), idleEvictTickMs(20_000));
    try testing.expectEqual(@as(i64, 15_000), idleEvictTickMs(60_000));
    try testing.expectEqual(@as(i64, 30_000), idleEvictTickMs(900_000));
    try testing.expectEqual(@as(i64, 1000), idleEvictTickMs(1000));
    try testing.expectEqual(@as(i64, 1000), idleEvictTickMs(2000));
    try testing.expectEqual(@as(i64, 30_000), idleEvictTickMs(28_800_000));
    try testing.expect(idleEvictTickMs(0) >= 1000);
}

test "gpu warm: ticks every half second inside the window after work, never past it or with work" {
    const w: u64 = 60 * std.time.ns_per_s;
    try testing.expectEqual(@as(?u64, null), gpuWarmParkNs(null, w));
    try testing.expectEqual(@as(?u64, GPU_WARM_TICK_NS), gpuWarmParkNs(0, w));
    try testing.expectEqual(@as(?u64, GPU_WARM_TICK_NS), gpuWarmParkNs(30 * std.time.ns_per_s, w));
    try testing.expectEqual(@as(?u64, 100), gpuWarmParkNs(w - 100, w));
    try testing.expectEqual(@as(?u64, null), gpuWarmParkNs(w, w));
    try testing.expectEqual(@as(?u64, null), gpuWarmParkNs(0, 0));
    try testing.expect(gpuWarmTickDue(false, 0, w));
    try testing.expect(!gpuWarmTickDue(true, 0, w));
    try testing.expect(!gpuWarmTickDue(false, null, w));
    try testing.expect(!gpuWarmTickDue(false, w, w));
    try testing.expect(!gpuWarmTickDue(false, 0, 0));
    try testing.expectEqual(GpuWarmWindow.restart, gpuWarmBeforePark(true, false));
    try testing.expectEqual(GpuWarmWindow.keep, gpuWarmBeforePark(false, false));
    try testing.expectEqual(GpuWarmWindow.close, gpuWarmBeforePark(false, true));
    try testing.expectEqual(GpuWarmWindow.close, gpuWarmBeforePark(true, true));
}

test "a cancelled prefill with no Generator commits its forwarded prefix, pad-only flag or not" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var reg = try ModelRegistry.init(testing.allocator, testing.io, null, 3, 0, null);
    defer reg.deinit();
    var sch: Scheduler = undefined;
    sch.registry = reg;
    sch.io = testing.io;
    sch.allocator = testing.allocator;
    sch.metrics = null;
    sch.hot_prefix_cache = null;
    sch.hot_cache_digests = &.{};
    sch.digest_residency = 0;
    sch.digest_mu = .init;
    var model: model_registry_mod.LoadedModel = undefined;
    model.prefix_cache = prefix_cache_mod.HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer model.prefix_cache.?.deinit();
    model.transformer = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var no_ssm: [0]SSMCacheEntry = .{};
    const Case = struct { forwarded: usize, hybrid: bool, committed: bool };
    for ([_]Case{
        .{ .forwarded = 400, .hybrid = false, .committed = true },
        .{ .forwarded = 400, .hybrid = true, .committed = false },
        .{ .forwarded = prefix_cache_mod.MIN_CANCELLED_COMMIT_TOKENS - 1, .hybrid = false, .committed = false },
    }) |c| {
        const before = model.prefix_cache.?.entries.items.len;
        var slot: Slot = undefined;
        slot.model = &model;
        slot.error_code = null;
        slot.legacy_gen = null;
        slot.glm5_request = null;
        slot.cancelled_prefill = .{ .forwarded = c.forwarded };
        slot.ssm_entries = if (c.hybrid) &no_ssm else null;
        slot.full_prompt = &tokens;
        slot.media_chain = &.{};
        slot.vision_key = 0;
        slot.media_start = null;
        slot.has_tools = false;
        slot.cache_key = 0;
        slot.was_pad_only = true;
        slot.cache = try KVCache.init(testing.allocator, 2);
        defer slot.cache.deinit();
        try prefix_cache_mod.testFillCache(&slot.cache, s, 2, 600);
        commitSlotIfApplicable(&sch, &slot);
        try testing.expectEqual(before + @intFromBool(c.committed), model.prefix_cache.?.entries.items.len);
        if (c.committed) try testing.expectEqual(c.forwarded, model.prefix_cache.?.entries.items[before].tokens.len);
    }
}

test "a spec cache handed to Generator init is out of the caller's errdefer reach, so a cancelled init frees it once" {
    const Owned = struct {
        frees: *u32,
        fn deinit(self: *@This()) void {
            self.frees.* += 1;
        }
    };
    const Run = struct {
        fn run(held: *?Owned) !void {
            errdefer if (held.*) |*o| o.deinit();
            var adopted = takeOwned(Owned, held).?;
            errdefer adopted.deinit();
            return error.Cancelled;
        }
    };
    var frees: u32 = 0;
    var held: ?Owned = .{ .frees = &frees };
    try testing.expectError(error.Cancelled, Run.run(&held));
    try testing.expectEqual(@as(u32, 1), frees);
}

/// Per-slot follow-up to a batched forward: `record` moves the id ledger, `publish` samples and streams.
const BatchedTickAction = struct { record: bool, publish: bool };

fn batchedTickAction(cancelled: bool) BatchedTickAction {
    // The KV row is already written, so the ledger records it even when cancelled.
    return .{ .record = true, .publish = !cancelled };
}

/// Batched decode kernel for >=2 active slots. All slots must have already
/// done a non-spec prefill (`skip_lazy_preforward = true`) so cache.step is
/// at prompt_len with `next_token_id` carrying t1. We forward those N tokens
/// in one kernel pass, sample per-slot, push the OLD next_token_id (= the
/// token we just committed to cache via the forward), and load the new
/// sampled id back into next_token_id.
/// Batched sibling: a batched group shares one forward, so a failure belongs to every slot in it.
/// Can this MTP slot's verify ride one batched trunk forward with its neighbours?
/// Per-request head (never qwen4's module-owned one), a GDN trunk with per-slot state
/// the batched path merges, and a round that is actually speculating this tick.
/// A MiMo MTP slot that may drop to plain batched ticks when crowded (`mtpCrowdThresholdFor`);
/// its rounds otherwise stay solo, and no grouped verify ever takes it.
pub fn slotMimoMtpCrowdable(slot: *const Slot) bool {
    const gen = if (slot.legacy_gen) |*g| g else return false;
    if (!slot.enable_mtp or gen.mtp == null or gen.mtp_cache == null or !gen.has_last_hidden) return false;
    if (gen.spec_disabled_runtime or gen.mtp_serial_left > 0 or gen.mtp_serial_exit != .none) return false;
    if (slot.sampling.constraint != null or slot.logprobs_n > 0 or sampler.penaltyActive(slot.sampling)) return false;
    const t = slot.model.transformer orelse return false;
    if (!t.supportsBatchedMimoDecode()) return false;
    return specTickMode(slot.enable_mtp, true, slot.enable_drafter, gen.dflash != null) == .mtp;
}

const Planner = @import("mtp_group_planner.zig");
const degenerate = @import("degenerate.zig");
const mtp_plan = @import("mtp_plan.zig");
const prefill_plan = @import("prefill_plan.zig");
const sampler = @import("sampler.zig");

pub fn runBatchedDecodeTick(sch: *Scheduler, active: []*Slot) !void {
    var inner_err: ?anyerror = null;
    runBatchedDecodeTickInner(sch, active) catch |e| {
        inner_err = e;
    };
    mlx.checkErrorDecode() catch |mlx_err| {
        log.err("[scheduler] batched decode aborted: MLX failure ({s}) — failing all {d} slots in the group\n", .{ @errorName(mlx_err), active.len });
        for (active) |s| s.markError(@errorName(mlx_err));
        return;
    };
    if (inner_err) |e| return e;
}

/// The stop check a slot with no pipeline state owes before its pending token is forwarded
/// into a batch. Pad runs were counted when a batch sampled the token, not again here.
fn batchEntryStops(gen: *Generator) !bool {
    const pad_run = gen.consecutive_pad;
    if (try gen.checkStop()) return true;
    gen.consecutive_pad = pad_run;
    return false;
}

/// The slots a plain batched tick forwards, in `live`: each past the per-tick guards every decode
/// path runs (loop stop, thinking budget), drained of its pipeline state, the ones that finish
/// or spend their tick left out.
pub fn batchedTickRows(sch: *Scheduler, active: []*Slot, live: []*Slot) !usize {
    var live_n: usize = 0;
    for (active) |slot| {
        const gen = if (slot.legacy_gen) |*g| g else {
            slot.markError("no_generator");
            continue;
        };
        if (try loopGuardTick(sch, slot, gen)) continue;
        if (gen.has_pending_logits or gen.has_pending_token) {
            const emitted = gen.drainPipelineForBatch(slot.allocator) catch |err| {
                slot.markError(@errorName(err));
                continue;
            };
            if (emitted) |tok| {
                slot.pushToken(tok);
                if (Planner.enabled() and gen.mtp_planner_owned) {
                    gen.mtp_planner_plain_ticks += 1;
                    plannerOutputClock(slot, gen);
                }
                if (tok != 0) slot.was_pad_only = false;
                slot.completion_tokens = gen.completion_tokens;
                if (generate_mod.isEosId(gen.next_token_id, slot.eos_token_ids)) {
                    slot.stop_token = gen.next_token_id;
                    slot.stop_path = "batch";
                    finishSlot(sch, slot, "stop");
                    continue;
                }
                if (slot.completion_tokens >= slot.max_tokens) {
                    finishSlot(sch, slot, "length");
                    continue;
                }
            } else {
                // checkStop fired on the pipelined lookahead (EOS / pad-run
                // / max_tokens / timeout); nothing to emit.
                slot.completion_tokens = gen.completion_tokens;
                slot.stop_path = "batch";
                finishSlot(sch, slot, gen.finish_reason);
                continue;
            }
        } else if (try batchEntryStops(gen)) {
            slot.completion_tokens = gen.completion_tokens;
            slot.stop_path = "batch";
            finishSlot(sch, slot, gen.finish_reason);
            continue;
        }
        live[live_n] = slot;
        live_n += 1;
    }
    return live_n;
}

fn runBatchedDecodeTickInner(sch: *Scheduler, active: []*Slot) !void {
    const N = active.len;
    if (N == 0) return;
    const allocator = sch.allocator;

    // Phase D: all batched slots must share the same model — the caller
    // (`runDecodeTick`) partitions by `slot.model` before dispatching here.
    // The first slot's transformer is authoritative; debug-assert the rest
    // match to surface partitioning bugs early.
    const xfm_ptr: *Transformer = active[0].model.transformer.?;
    if (std.debug.runtime_safety) {
        for (active) |s| std.debug.assert(s.model.transformer.? == xfm_ptr);
    }

    const observe_plain = Planner.enabled() and xfm_ptr.qwen4 != null and N <= Planner.MAX_ROWS;
    var price_watch = io_util.Stopwatch.init(active[0].io);
    var price_key: group_cost_mod.GroupShape = .{};
    var price_clean = observe_plain;
    if (observe_plain) {
        const zeros: [Planner.MAX_ROWS]u8 = @splat(0);
        price_key = plannerShape(active, zeros[0..N]);
        xfm_ptr.cost_trace_active = true;
        xfm_ptr.cost_attention = 0;
        xfm_ptr.cost_moe = 0;
        for (active) |slot| {
            const gen = &slot.legacy_gen.?;
            if (gen.has_pending_logits or gen.has_pending_token or slot.planner_plain_transition or slot.mtp_plain_tick) price_clean = false;
        }
    }
    defer if (observe_plain) {
        xfm_ptr.cost_trace_active = false;
    };

    // Legacy→batched transition: a slot arriving from a legacy single-slot
    // tick (or fresh from prefill) carries lazy pipeline state — a lookahead
    // token ALREADY FORWARDED into its KV cache plus `pending_logits` for
    // the position after it. Consume that state via `drainPipelineForBatch`
    // (emit the lookahead, sample the new next_token_id from the pending
    // logits). Dropping it and re-forwarding `next_token_id` — the pre-fix
    // behavior — appended a duplicate cache position and re-emitted an
    // already-emitted token, corrupting any stream whose slot joined a
    // batch mid-generation (tests/test_batched_transition.sh). Slots that
    // finish during the drain are excluded from the batch.
    const live = try allocator.alloc(*Slot, N);
    defer allocator.free(live);
    const live_n = try batchedTickRows(sch, active, live);
    if (live_n == 0) return;
    var batch = live[0..live_n];

    // Build inputs.
    const next_tokens = try allocator.alloc(u32, live_n);
    defer allocator.free(next_tokens);
    const ctxs = try allocator.alloc(*ForwardCtx, live_n);
    defer allocator.free(ctxs);
    const rope_offsets = try allocator.alloc(u32, live_n);
    defer allocator.free(rope_offsets);

    for (batch, 0..) |slot, i| {
        const gen = &slot.legacy_gen.?;
        next_tokens[i] = gen.next_token_id;
        ctxs[i] = &gen.ctx;
    }

    // Two batched kernels: the standard one, and the GatedDeltaNet twin for
    // hybrid trunks. `batchedGdnReady` is the runtime half of
    // the gate — a slot that has not prefilled yet carries no recurrent state
    // to merge, so that tick stays serial rather than merging a wrong width.
    const use_gdn = xfm_ptr.supportsBatchedGdnDecode() and xfm_ptr.batchedGdnReady(ctxs);
    const use_mimo = xfm_ptr.supportsBatchedMimoDecode();
    const use_glm = xfm_ptr.supportsBatchedGlmRows();
    const use_qwen4_rows = use_gdn and xfm_ptr.supportsBatchedQwen4Rows();
    // Position source is per PATH: a GDN trunk positions from the slot's
    // `moe_seq_offset` — `KVCache.step` only advances on layer 0, which is a
    // linear layer there, so it reads 0 forever and every batched token was
    // roped at position 0.
    for (batch, 0..) |slot, i| rope_offsets[i] = @intCast(if (use_gdn or use_mimo) slot.moe_seq_offset else slot.cache.step);
    if (xfm_ptr.supportsBatchedGdnDecode() and !use_gdn) {
        // A slot with no recurrent state yet cannot join the merge. Decode the
        // group serially this tick instead of skipping it — skipping advances
        // nothing, so a group that never becomes ready would spin forever.
        for (batch) |s| try runSingleDecodeTick(sch, s);
        return;
    }
    var want_hidden = false;
    for (batch) |slot| {
        want_hidden = want_hidden or slot.mtp_plain_tick;
        if (Planner.enabled() and slot.legacy_gen.?.mtp_planner_owned) {
            const gen = &slot.legacy_gen.?;
            if (slot.mtp_plain_tick) gen.mtp_planner_prime_ticks += 1 else {
                gen.mtp_hidden_stale = true;
                gen.mtp_planner_plain_ticks += 1;
            }
        }
    }
    var hidden_rows: ?[]mlx.mlx_array = null;
    defer if (hidden_rows) |rows| {
        for (rows) |a| _ = mlx.mlx_array_free(a);
        allocator.free(rows);
    };
    const logits_arr = if (use_glm)
        try glmRowsForward(sch, allocator, xfm_ptr, &batch, next_tokens)
    else if (use_qwen4_rows)
        try xfm_ptr.forwardQwen4DecodeRows(next_tokens, ctxs, if (want_hidden) &hidden_rows else null)
    else if (use_gdn)
        try xfm_ptr.forwardMoeBatchedDecode(next_tokens, ctxs, rope_offsets, if (want_hidden) &hidden_rows else null)
    else if (use_mimo)
        try xfm_ptr.forwardMimoBatchedDecode(next_tokens, ctxs, rope_offsets, if (want_hidden) &hidden_rows else null)
    else
        return error.NoBatchedKernel;
    defer {
        for (logits_arr) |a| _ = mlx.mlx_array_free(a);
        allocator.free(logits_arr);
    }
    if (hidden_rows) |rows| {
        for (batch, 0..) |slot, i| {
            if (!slot.mtp_plain_tick) continue;
            slot.mtp_plain_tick = false;
            const gen = &slot.legacy_gen.?;
            if (gen.has_last_hidden) _ = mlx.mlx_array_free(gen.last_hidden);
            gen.last_hidden = rows[i];
            rows[i] = mlx.mlx_array_new();
            gen.has_last_hidden = true;
            if (Planner.enabled()) gen.mtp_hidden_stale = false;
        }
    }
    for (batch) |slot| slot.mtp_plain_tick = false;
    // The batched forward advances only its scratch offset; each slot's own
    // position moves here so a slot leaving the batch resumes serial from
    // the right place (qwen4's QSA reads it for kv length + tail rule).
    if (!use_qwen4_rows) {
        for (batch) |slot| slot.moe_seq_offset += 1;
    }

    // `gen.sampling`, not `slot.sampling`: the Generator's copy passed
    // the initWithOptions chokepoint and carries the model's
    // reserved-token suppression mask; the slot's copy is the raw
    // request params.
    var sample_params: [MAX_BATCH_GROUP]sampler.SamplingParams = undefined;
    var sample_rows: [MAX_BATCH_GROUP]mlx.mlx_array = undefined;
    var sample_ids: [MAX_BATCH_GROUP]i32 = undefined;
    std.debug.assert(batch.len == logits_arr.len);
    std.debug.assert(batch.len <= sample_params.len);
    var shifted_n: usize = 0;
    defer for (sample_rows[0..shifted_n], logits_arr[0..shifted_n]) |row, raw| {
        if (row.ctx != raw.ctx) _ = mlx.mlx_array_free(row);
    };
    for (batch, 0..) |slot, i| {
        const gen = &slot.legacy_gen.?;
        sample_rows[i] = try gen.thinkShifted(logits_arr[i], .{ .decided = &.{gen.next_token_id} });
        shifted_n = i + 1;
        sample_params[i] = gen.sampling;
    }
    if (batch.len > 0) {
        try sampler.sampleRows(sample_ids[0..batch.len], sample_rows[0..batch.len], sample_params[0..batch.len], xfm_ptr.s);
        for (batch, 0..) |slot, i| slot.legacy_gen.?.sampling.draw = sample_params[i].draw;
    }

    // Sample per slot, emit prev id, set new next_token_id.
    for (batch, 0..) |slot, i| {
        const gen = &slot.legacy_gen.?;
        const act = batchedTickAction(slot.cancelled.load(.acquire));
        const sampled: ?i32 = if (act.publish) sample_ids[i] else null;

        const emit = gen.next_token_id;
        gen.generated_ids.append(slot.allocator, emit) catch |err| {
            slot.markError(@errorName(err));
            continue;
        };
        gen.advanceStep(1);
        if (emit != 0) slot.was_pad_only = false;
        const val = sampled orelse continue;
        gen.next_token_id = @intCast(val);
        if (Planner.enabled()) plannerOutputClock(slot, gen);

        // Stop checks (mirrors Generator.checkStop).
        if (generate_mod.isEosId(emit, slot.eos_token_ids)) {
            // Per existing contract, emit IS NOT yielded when it's EOS — the
            // STOP token comes BEFORE the yield. But here the cache has
            // already moved past it. The legacy path's checkStop runs on the
            // NEXT token (it's checked before emit). To preserve that
            // behavior we emit and then mark finished if `next_token_id` is
            // EOS (i.e. STOP is the next sampled token, ignored).
            slot.pushToken(emit);
            slot.completion_tokens = gen.completion_tokens;
            // not finished yet; next tick's checkStop on next_token_id ends it
        } else {
            slot.pushToken(emit);
            slot.completion_tokens = gen.completion_tokens;
        }

        if (generate_mod.isEosId(gen.next_token_id, slot.eos_token_ids)) {
            slot.stop_token = gen.next_token_id;
            slot.stop_path = "batch";
            finishSlot(sch, slot, "stop");
            continue;
        }
        if (gen.next_token_id == 0) {
            gen.consecutive_pad += 1;
            if (gen.consecutive_pad >= 3) {
                slot.stop_token = 0;
                slot.stop_path = "batch";
                finishSlot(sch, slot, "stop");
                continue;
            }
        } else {
            gen.consecutive_pad = 0;
        }
        if (slot.completion_tokens >= slot.max_tokens) {
            finishSlot(sch, slot, "length");
            continue;
        }
    }
    if (price_clean and live_n == N) {
        var gaps: [Planner.MAX_ROWS]f32 = undefined;
        for (batch, 0..) |slot, i| {
            if (slot.state != .decoding or slot.cancelled.load(.acquire)) return;
            gaps[i] = slot.mtp_publish_gap_ms;
        }
        price_key.attention = xfm_ptr.cost_attention;
        price_key.moe = xfm_ptr.cost_moe;
        const emitted: [Planner.MAX_ROWS]u32 = @splat(1);
        const ms = @as(f32, @floatFromInt(price_watch.read())) / std.time.ns_per_ms;
        _ = xfm_ptr.mtp_group_cost.observeShape(price_key, ms, emitted[0..N], gaps[0..N], .round);
    }
}

const testing = std.testing;
pub const expectError = @import("test_expect.zig").expectError;

test "a live MTP slot never ticks regular, an unarmed one always does" {
    try testing.expectEqual(SpecTickMode.mtp, specTickMode(true, true, false, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(true, false, false, false));
}

test "complete keeps a slot in decoding until its pass is out, so admission never misses it" {
    const allocator = testing.allocator;
    var sch: Scheduler = undefined;
    sch.allocator = allocator;
    sch.io = testing.io;
    sch.kv_quant_config = .dense;
    sch.kv_quant_explicit = false;
    sch.queue_mu = .init;
    sch.queue_cond = .init;
    sch.submit_cond = .init;
    sch.shutdown = .init(false);
    sch.queue_cap = 2;
    sch.in_flight = 0;
    sch.pending = .empty;
    sch.decoding = .empty;
    sch.cleanup_queue = .empty;
    sch.prefilling = .empty;
    defer sch.pending.deinit(allocator);
    defer sch.decoding.deinit(allocator);
    defer sch.cleanup_queue.deinit(allocator);
    defer sch.prefilling.deinit(allocator);
    var cfg = try model_mod.parseConfigFromJson(allocator, @embedFile("fixtures/glm5_config.json"));
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.transformer = null;
    model.prefix_cache = null;
    const slot = try sch.submit(.{ .model = &model, .prompt_ids = &.{1}, .sampling = .{}, .eos_token_ids = &.{}, .max_tokens = 1 });
    _ = sch.pending.orderedRemove(0);
    try sch.decoding.append(allocator, slot);
    _ = slot.in_pass.fetchAdd(1, .acq_rel);
    const t = try std.Thread.spawn(.{}, Scheduler.complete, .{ &sch, slot });
    std.Io.sleep(testing.io, .fromMilliseconds(30), .real) catch {};
    sch.queue_mu.lockUncancelable(sch.io);
    const visible = sch.decoding.items.len + sch.cleanup_queue.items.len;
    sch.queue_mu.unlock(sch.io);
    _ = slot.in_pass.fetchSub(1, .acq_rel);
    t.join();
    try testing.expectEqual(@as(usize, 1), visible);
    try testing.expectEqual(@as(usize, 0), sch.decoding.items.len);
    try testing.expectEqual(@as(usize, 1), sch.cleanup_queue.items.len);
    while (sch.cleanup_queue.items.len > 0) sch.cleanup_queue.orderedRemove(0).deinit();
}

test "sumInflightGeneratedTokens sums active slots, excludes finished/cancelled/errored" {
    // Lightweight stub carrying exactly the fields the aggregate reads — proves
    // the live-gauge filter without constructing a real (mlx-backed) Slot.
    const StubSlot = struct {
        completion_tokens: u32,
        finished: bool,
        error_code: ?[]const u8,
        cancelled: std.atomic.Value(bool),
        fn make(tok: u32) @This() {
            return .{ .completion_tokens = tok, .finished = false, .error_code = null, .cancelled = std.atomic.Value(bool).init(false) };
        }
    };
    var a = StubSlot.make(10);
    var b = StubSlot.make(20);
    var done = StubSlot.make(99);
    done.finished = true; // already counted in generation_tokens_total
    var errd = StubSlot.make(50);
    errd.error_code = "OutOfMemory";
    var cxl = StubSlot.make(7);
    cxl.cancelled.store(true, .release);

    // Two still-decoding slots (10 + 20); the finished/errored/cancelled ones
    // are excluded, so the aggregate is 30 — never double-counting the tail.
    const active = [_]*StubSlot{ &a, &b, &done, &errd, &cxl };
    try testing.expectEqual(@as(u64, 30), sumInflightGeneratedTokens(active[0..]));

    // At rest: the last decoding slot finishes ⇒ aggregate collapses to 0,
    // pinning the "live == total at rest" invariant test_metrics.sh checks.
    a.finished = true;
    b.finished = true;
    try testing.expectEqual(@as(u64, 0), sumInflightGeneratedTokens(active[0..]));
}

pub fn expectOutcomes(m: *const metrics_mod.Metrics, success: u64, cancelled: u64, failed: u64, rejected: u64) !void {
    try testing.expectEqual(success, m.requests_success_total.load());
    try testing.expectEqual(cancelled, m.requests_cancelled_total.load());
    try testing.expectEqual(failed, m.requests_failed_total.load());
    try testing.expectEqual(rejected, m.requests_rejected_total.load());
}

test "specInitWiring: a module-owned arch only gets the spec modes it can roll back" {
    // Spec decode must be able to ROLL BACK a rejected tail. The shell rolls
    // back its KVCache/ssm_entries — which a module-owned arch does not use, so
    // by the time the verify returns, the module has already absorbed every
    // draft and the shell's snapshot/truncate run over an EMPTY entries array.
    // The predicate is per-ARCH CAPABILITY (`moduleStateSpecRollback`), not ownership.
    // Args: (owns_module_state, module_spec_rollback,
    //        enable_mtp, has_mtp, enable_drafter, has_dflash)

    // Plain arch: MTP wins over a missing sidecar.
    {
        const w = specInitWiring(false, false, true, true, true, false);
        try testing.expect(w.use_mtp and !w.use_dflash);
    }
    // A flag with no loaded handle never arms.
    {
        const w = specInitWiring(false, false, true, false, false, false);
        try testing.expect(!w.use_mtp and !w.use_dflash);
    }

    // DFlash rides the enable_drafter switch: dflash > MTP.
    {
        const w = specInitWiring(false, false, false, false, true, true);
        try testing.expect(!w.use_mtp and w.use_dflash);
    }
    // A loaded drafter outranks the checkpoint's own MTP head: the sidecar
    // is an explicit choice (`--drafter` or the in-dir `drafter/`), the head
    // ships with every pack; `--no-drafter` restores MTP by not loading it.
    {
        const w = specInitWiring(false, false, true, true, true, true);
        try testing.expect(w.use_dflash and !w.use_mtp);
    }
    // enable_drafter:false on the request hands the round back to MTP.
    {
        const w = specInitWiring(false, false, true, true, false, true);
        try testing.expect(w.use_mtp and !w.use_dflash);
    }
    // enable_drafter:false opts the sidecar out.
    {
        const w = specInitWiring(false, false, false, false, false, true);
        try testing.expect(!w.use_dflash);
    }

    // Module-owned with NO rollback: everything off.
    {
        const w = specInitWiring(true, false, true, true, true, true);
        try testing.expect(!w.use_mtp and !w.use_dflash);
    }

    // Module-owned WITH rollback: a loaded, requested drafter outranks the head;
    // without one the head arms.
    {
        const w = specInitWiring(true, true, true, true, true, true);
        try testing.expect(w.use_dflash and !w.use_mtp);
    }
    {
        const w = specInitWiring(true, true, true, true, false, true);
        try testing.expect(w.use_mtp and !w.use_dflash);
    }
    {
        const w = specInitWiring(true, false, false, false, true, true);
        try testing.expect(!w.use_mtp and !w.use_dflash);
    }
    // Rollback capability alone never arms a head or a drafter that is not loaded.
    {
        const w = specInitWiring(true, true, true, false, true, false);
        try testing.expect(!w.use_mtp and !w.use_dflash);
    }
    // ...nor one the request opted out of.
    {
        const w = specInitWiring(true, true, false, true, true, false);
        try testing.expect(!w.use_mtp);
    }
    // An image request keeps the head when the drafter is off for it.
    {
        const w = specInitWiring(true, true, true, true, false, true);
        try testing.expect(w.use_mtp and !w.use_dflash);
    }
}

test "specTickMode: every spec arm requires the GENERATOR's armed state, not the slot flag alone" {
    // mtp and dflash each need the slot's wish AND the generator's handle.
    try testing.expectEqual(SpecTickMode.mtp, specTickMode(true, true, false, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(true, false, false, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, true, false));
    // Generator armed but the request opted enable_mtp off → serial.
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, true, false, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, false));

    // DFlash: slot's enable_drafter + generator's dflash handle; outranks the MTP
    // head (a loaded sidecar is the explicit choice). A generator handle alone never
    // resurrects it, and a dflash generator with the slot flag off stays regular, or
    // MTP when that is armed.
    try testing.expectEqual(SpecTickMode.dflash, specTickMode(false, false, true, true));
    try testing.expectEqual(SpecTickMode.dflash, specTickMode(true, true, true, true));
    try testing.expectEqual(SpecTickMode.mtp, specTickMode(true, true, false, true));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, true));
}

test "writeThroughSpanReached: a sub-chunk warm turn never persists inside the prefill" {
    const chunk: u32 = 1024;
    // Warm turn: a restored 32k prefix plus a 31-token instruction tail.
    try testing.expect(!writeThroughSpanReached(31, chunk));
    try testing.expect(!writeThroughSpanReached(0, chunk));
    try testing.expect(!writeThroughSpanReached(1023, chunk));
    try testing.expect(writeThroughSpanReached(1024, chunk));
    try testing.expect(writeThroughSpanReached(4096, chunk));
    try testing.expect(writeThroughSpanReached(65_665, chunk));
    try testing.expect(writeThroughSpanReached(4096, 4096));
    try testing.expect(!writeThroughSpanReached(4095, 4096));
}

test "prefix-cache commit declines a pad-only generation, never a zero-token one" {
    // Bar: a zero-token answer still commits; only an all-pad generation is declined.
    try testing.expect(!commitDeclinesPadOnly(0, true));
    try testing.expect(!commitDeclinesPadOnly(0, false));
    try testing.expect(commitDeclinesPadOnly(3, true));
    try testing.expect(!commitDeclinesPadOnly(3, false));
}

test "a speculative block ending on EOS records the stop token and leaves it out of the count" {
    var gen: Generator = undefined;
    gen.mtp = null;
    gen.sampling = .{};
    gen.completion_tokens = 0;
    var slot: Slot = undefined;
    slot.io = testing.io;
    slot.allocator = testing.allocator;
    slot.out_mu = .init;
    slot.out_cond = .init;
    slot.out_event = .unset;
    slot.out_buf = .empty;
    defer slot.out_buf.deinit(testing.allocator);
    slot.handover_token = null;
    slot.cancelled = .init(false);
    var eos = [_]u32{ 248046, 248044 };
    slot.eos_token_ids = &eos;
    slot.completion_tokens = 0;
    slot.max_tokens = 100;
    slot.was_pad_only = true;
    slot.stop_token = null;
    slot.stop_path = "";

    const reason = publishBlockTokens(&slot, &gen, &.{ 1510, 248046 }) orelse return error.TestExpectedFinish;
    try testing.expectEqualStrings("stop", reason);
    try testing.expectEqual(@as(?u32, 248046), slot.stop_token);
    try testing.expectEqualStrings("spec", slot.stop_path);
    try testing.expectEqual(@as(u32, 1), slot.completion_tokens);
    try testing.expectEqualSlices(u32, &.{1510}, slot.out_buf.items);

    // A block that runs out of tokens without an EOS or the cap names no stop token.
    slot.stop_token = null;
    slot.completion_tokens = 1;
    gen.completion_tokens = 3;
    try testing.expect(publishBlockTokens(&slot, &gen, &.{ 5, 6 }) == null);
    try testing.expectEqual(@as(?u32, null), slot.stop_token);
}

test "complete waits out an inference pass that still holds the slot" {
    // The pass reads handler-owned sampling state (`think_bound`) until it drops
    // `in_pass`; returning earlier lets the handler free what the pass still reads.
    const io = std.testing.io;
    const Pass = struct {
        in_pass: std.atomic.Value(u32) = .init(1),
        read_done: std.atomic.Value(bool) = .init(false),
        fn run(p: *@This(), pio: std.Io) void {
            std.Io.sleep(pio, .fromMilliseconds(30), .real) catch {};
            p.read_done.store(true, .release);
            _ = p.in_pass.fetchSub(1, .acq_rel);
        }
    };
    var pass: Pass = .{};
    var mu: std.Io.Mutex = .init;
    const t = try std.Thread.spawn(.{}, Pass.run, .{ &pass, io });
    mu.lockUncancelable(io);
    waitPassesOut(io, &mu, &pass.in_pass);
    const done = pass.read_done.load(.acquire);
    mu.unlock(io);
    t.join();
    try testing.expect(done);
}

test "a cancel mid batched tick keeps generated_ids level with the KV rows" {
    // Bar: kv rows == full_prompt + generated_ids even when the slot cancels mid tick.
    const prompt_len: usize = 7;
    var kv_rows = prompt_len;
    var ids: usize = 0;
    for ([_]bool{ false, false, false, true }) |cancelled| {
        kv_rows += 1;
        const act = batchedTickAction(cancelled);
        if (act.record) ids += 1;
        if (cancelled) try testing.expect(!act.publish);
    }
    try testing.expectEqual(kv_rows, prompt_len + ids);
}

test "retained position: a padded row can never take the full-accept arm" {
    // Unpadded full accept commits 1+m and keeps the trunk state the forward left.
    try testing.expect(generate_mod.Generator.mtpFullAccept(2, 2, 3));
    try testing.expectEqual(@as(u32, 3), generate_mod.Generator.mtpRetained(2));
    // Same accept count, row padded to a neighbour's width: rollback arm.
    try testing.expect(!generate_mod.Generator.mtpFullAccept(2, 2, 5));
    try testing.expectEqual(@as(u32, 3), generate_mod.Generator.mtpRetained(2));
    // Partial and zero accepts are rollback at any width.
    try testing.expect(!generate_mod.Generator.mtpFullAccept(1, 2, 3));
    try testing.expect(!generate_mod.Generator.mtpFullAccept(0, 2, 3));
    try testing.expectEqual(@as(u32, 1), generate_mod.Generator.mtpRetained(0));
}

test "single MTP slot reaches the round entry through runDecodeTick" {
    try mlx.requireGpu();
    var xfm: Transformer = undefined;
    var model: model_registry_mod.LoadedModel = undefined;
    model.transformer = null;
    var gen: Generator = undefined;
    gen.mtp_planner_owned = false;
    gen.generated_ids = .empty;
    gen.loop_guard_start = 0;
    gen.think_closed_by_loop = false;
    gen.mtp = .{ .qwen4 = &xfm };
    gen.mtp_cache = .{ .mimo = undefined };
    gen.has_last_hidden = true;
    gen.dflash = null;
    gen.done = false;
    gen.sampling = .{};
    gen.logprobs_n = 1;
    var slot: Slot = undefined;
    slot.allocator = testing.allocator;
    slot.model = &model;
    slot.legacy_gen = gen;
    slot.enable_mtp = true;
    slot.enable_drafter = false;
    slot.finished = false;
    slot.error_code = null;
    defer if (slot.error_code) |code| testing.allocator.free(code);
    slot.cancelled = std.atomic.Value(bool).init(false);
    slot.completion_tokens = 0;
    var sch: Scheduler = undefined;
    sch.force_batched = false;
    sch.inflight_generated_tokens = std.atomic.Value(u64).init(0);
    var active = [_]*Slot{&slot};
    // The tick's decode checkpoint consumes any MLX error latched before it and errors the slot instead.
    try testing.expect(!mlx.errorPending());
    try testing.expectError(error.SpecDecodeUnsupported, runDecodeTick(&sch, &active));
    try testing.expect(slot.legacy_gen.?.spec_cost_solo);
    try testing.expectEqual(@as(u32, 0), slot.legacy_gen.?.mtp_group_cap);
    try testing.expectEqual(@as(u64, 0), sch.inflight_generated_tokens.load(.monotonic));
}

test "a slot with no pipeline state is stop-checked before it joins a batch, EOS is never forwarded" {
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(testing.allocator);
    var gen: Generator = undefined;
    gen.generated_ids = ids;
    gen.eos_token_ids = &.{2};
    gen.next_token_id = 2;
    gen.step = 0;
    gen.max_tokens = 8;
    gen.consecutive_pad = 0;
    gen.timeout_ns = 0;
    gen.stall = .{};
    gen.timer = io_util.Stopwatch.init(testing.io);
    gen.done = false;
    gen.finish_reason = "length";
    // A fresh DFlash2 slot: prefill sampled EOS synchronously and left no pending pipeline.
    try testing.expect(try batchEntryStops(&gen));
    try testing.expect(gen.done);
    try testing.expectEqualStrings("stop", gen.finish_reason);
    try testing.expectEqual(@as(u32, 0), gen.step);
    try testing.expectEqual(@as(usize, 0), gen.generated_ids.items.len);

    // A real token joins the batch, and a pad the batch sampled is not counted a second time.
    gen.done = false;
    gen.next_token_id = 0;
    gen.consecutive_pad = 1;
    try testing.expect(!(try batchEntryStops(&gen)));
    try testing.expectEqual(@as(u32, 1), gen.consecutive_pad);
    try testing.expect(!gen.done);

    // A spent budget or stall ends it as the solo path does.
    gen.next_token_id = 5;
    gen.step = 8;
    try testing.expect(try batchEntryStops(&gen));
    try testing.expectEqualStrings("length", gen.finish_reason);
}

test "firstMediaPlaceholder: a placeholder id in ORDINARY TEXT is not a media boundary" {
    // The ids are ordinary vocabulary entries, so a text-only prompt can carry
    // one (live: a pasted source file held 248056 at index 18338 of a 73k
    // prompt). A media boundary exists only where media rows do.
    const image_id: u32 = 248056;
    const text_only = [_]u32{ 7, 8, image_id, 9 };
    try testing.expectEqual(@as(?usize, null), firstMediaPlaceholder(false, &text_only, image_id, 0));
    try testing.expectEqual(@as(?usize, 2), firstMediaPlaceholder(true, &text_only, image_id, 0));
}

pub fn submitTestScheduler(allocator: std.mem.Allocator, queue_cap: u32) Scheduler {
    var sch: Scheduler = undefined;
    sch.allocator = allocator;
    sch.io = testing.io;
    sch.kv_quant_config = .dense;
    sch.kv_quant_explicit = false;
    sch.queue_mu = .init;
    sch.queue_cond = .init;
    sch.submit_cond = .init;
    sch.shutdown = .init(false);
    sch.queue_cap = queue_cap;
    sch.in_flight = 0;
    sch.req_seq = 0;
    sch.pending = .empty;
    sch.decoding = .empty;
    sch.cleanup_queue = .empty;
    sch.prefilling = .empty;
    sch.load_queue = .empty;
    sch.unload_queue = .empty;
    sch.abandoned = null;
    sch.orphan_vision_n = 0;
    return sch;
}

test "a model load or unload waits for every queued slot to be destroyed" {
    var cfg = ModelConfig{ .num_hidden_layers = 0 };
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.transformer = null;
    const allocator = testing.allocator;
    for ([_]bool{ false, true }) |evicting| {
        var sch = submitTestScheduler(allocator, 32);
        defer sch.pending.deinit(allocator);
        defer sch.cleanup_queue.deinit(allocator);
        defer sch.prefilling.deinit(allocator);
        defer sch.load_queue.deinit(allocator);
        defer sch.unload_queue.deinit(allocator);
        var load: LoadRequest = undefined;
        var unload = UnloadRequest{ .entry = &model };
        for (0..17) |_| {
            const slot = try sch.submit(.{ .model = &model, .prompt_ids = &.{1}, .sampling = .{}, .eos_token_ids = &.{}, .max_tokens = 1 });
            sch.complete(slot);
        }
        if (evicting) try sch.load_queue.append(allocator, &load) else try sch.unload_queue.append(allocator, &unload);
        var destroyed: usize = 0;
        var model_job_after: ?usize = null;
        var batch: [16]*Slot = undefined;
        for (0..4) |_| {
            const take = takeTeardownLocked(&sch, &batch);
            for (batch[0..take.cleanup_n]) |s| s.deinit();
            destroyed += take.cleanup_n;
            if (take.load != null or take.unload != null) {
                model_job_after = destroyed;
                break;
            }
        }
        try testing.expectEqual(@as(?usize, 17), model_job_after);
    }
}

test "a per-request kv_quant override (16, \"16\", 8, 6, 4) reaches the submitted slot's cache as that scheme" {
    const allocator = testing.allocator;
    var sch: Scheduler = undefined;
    sch.allocator = allocator;
    sch.io = testing.io;
    sch.kv_quant_config = .dense;
    sch.kv_quant_explicit = false;
    sch.queue_mu = .init;
    sch.queue_cond = .init;
    sch.submit_cond = .init;
    sch.shutdown = .init(false);
    sch.queue_cap = 2;
    sch.in_flight = 0;
    sch.pending = .empty;
    sch.decoding = .empty;
    sch.cleanup_queue = .empty;
    sch.prefilling = .empty;
    defer sch.pending.deinit(allocator);
    defer sch.decoding.deinit(allocator);
    defer sch.cleanup_queue.deinit(allocator);
    defer sch.prefilling.deinit(allocator);
    var plain = ModelConfig{ .model_type = "not_a_served_arch", .num_hidden_layers = 1 };
    var model: LoadedModel = undefined;
    model.config = &plain;
    model.transformer = null;
    model.prefix_cache = null;

    const Case = struct { body: std.json.Value, want: transformer_mod.KVQuantConfig };
    for ([_]Case{
        .{ .body = .{ .integer = 16 }, .want = transformer_mod.KVQuantConfig.dense },
        .{ .body = .{ .string = "16" }, .want = transformer_mod.KVQuantConfig.dense },
        .{ .body = .{ .integer = 8 }, .want = transformer_mod.KVQuantConfig.affine(8) },
        .{ .body = .{ .integer = 6 }, .want = transformer_mod.KVQuantConfig.affine(6) },
        .{ .body = .{ .string = "6" }, .want = transformer_mod.KVQuantConfig.affine(6) },
        .{ .body = .{ .integer = 4 }, .want = transformer_mod.KVQuantConfig.affine(4) },
    }) |c| {
        const override = transformer_mod.KVQuantConfig.fromJsonValue(c.body).?;
        const slot = try sch.submit(.{ .model = &model, .prompt_ids = &.{1}, .sampling = .{}, .eos_token_ids = &.{}, .max_tokens = 1, .kv_quant_config = override });
        try testing.expectEqual(c.want, slot.ctx.cache.config);
        sch.complete(slot);
        while (sch.cleanup_queue.items.len > 0) sch.cleanup_queue.orderedRemove(0).deinit();
    }
}

test "an empty prompt is refused at submit, before any slot reaches the inference thread" {
    const allocator = testing.allocator;
    var sch: Scheduler = undefined;
    sch.allocator = allocator;
    sch.io = testing.io;
    sch.kv_quant_config = .dense;
    sch.kv_quant_explicit = false;
    sch.queue_mu = .init;
    sch.queue_cond = .init;
    sch.submit_cond = .init;
    sch.shutdown = .init(false);
    sch.queue_cap = 2;
    sch.in_flight = 0;
    sch.pending = .empty;
    sch.decoding = .empty;
    sch.cleanup_queue = .empty;
    sch.prefilling = .empty;
    defer sch.pending.deinit(allocator);
    defer sch.decoding.deinit(allocator);
    defer sch.cleanup_queue.deinit(allocator);
    defer sch.prefilling.deinit(allocator);
    var cfg = ModelConfig{ .num_hidden_layers = 0 };
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.transformer = null;
    model.prefix_cache = null;
    const result = sch.submit(.{ .model = &model, .prompt_ids = &.{}, .sampling = .{}, .eos_token_ids = &.{}, .max_tokens = 4 });
    try expectError(error.EmptyPrompt, result);
    try testing.expectEqual(@as(usize, 0), sch.pending.items.len);
}

test "a freed MiMo slot returns its KV to the OS, not to MLX's pool" {
    try mlx.requireGpu();
    const allocator = testing.allocator;
    var sch: Scheduler = undefined;
    sch.allocator = allocator;
    sch.io = testing.io;
    sch.kv_quant_config = .dense;
    sch.kv_quant_explicit = false;
    sch.queue_mu = .init;
    sch.queue_cond = .init;
    sch.submit_cond = .init;
    sch.shutdown = .init(false);
    sch.queue_cap = 2;
    sch.in_flight = 0;
    sch.pending = .empty;
    sch.decoding = .empty;
    sch.cleanup_queue = .empty;
    sch.prefilling = .empty;
    defer sch.pending.deinit(allocator);
    defer sch.decoding.deinit(allocator);
    defer sch.cleanup_queue.deinit(allocator);
    defer sch.prefilling.deinit(allocator);
    const s = mlx.gpuStream();
    const kv_bytes: usize = 64 << 20;
    var mimo = ModelConfig{ .model_type = "mimo_v2", .num_hidden_layers = 1, .has_sliding_window = true, .sliding_window = 128, .head_dim = 192 };
    var plain = ModelConfig{ .model_type = "not_a_served_arch", .num_hidden_layers = 1 };
    try testing.expect(mimo.reservesKvCapacity() and !plain.reservesKvCapacity());
    for ([_]*ModelConfig{ &mimo, &plain }) |cfg| {
        var model: LoadedModel = undefined;
        model.config = cfg;
        model.transformer = null;
        model.prefix_cache = null;
        _ = mlx.mlx_clear_cache();
        const slot = try sch.submit(.{ .model = &model, .prompt_ids = &.{1}, .sampling = .{}, .eos_token_ids = &.{}, .max_tokens = 1 });
        sch.complete(slot);
        const done = sch.cleanup_queue.orderedRemove(0);
        // The reservation a long request made, as the slot's own KV buffer.
        var kv = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_arange(&kv, 0, @floatFromInt(kv_bytes / 4), 1, .float32, s));
        try mlx.check(mlx.mlx_array_eval(kv));
        _ = mlx.mlx_synchronize(s);
        _ = mlx.mlx_array_free(done.cache.entries[0].keys);
        done.cache.entries[0].keys = kv;
        deinitSlotsReturningPool(&.{done});
        var pooled: usize = 0;
        _ = mlx.mlx_get_cache_memory(&pooled);
        if (cfg == &mimo) {
            try testing.expectEqual(@as(usize, 0), pooled);
        } else {
            try testing.expect(pooled >= kv_bytes);
        }
    }
    _ = mlx.mlx_clear_cache();
}

test "shutdown cancels a slot whose prefill is running, so the prefill stops at its next chunk" {
    const allocator = testing.allocator;
    var cfg = ModelConfig{ .num_hidden_layers = 0 };
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.transformer = null;
    model.prefix_cache = null;
    var sch: Scheduler = undefined;
    sch.allocator = allocator;
    sch.io = testing.io;
    sch.kv_quant_config = .dense;
    sch.kv_quant_explicit = false;
    sch.queue_mu = .init;
    sch.queue_cond = .init;
    sch.submit_cond = .init;
    sch.shutdown = .init(false);
    sch.queue_cap = 2;
    sch.in_flight = 0;
    sch.pending = .empty;
    sch.decoding = .empty;
    sch.cleanup_queue = .empty;
    sch.prefilling = .empty;
    defer sch.pending.deinit(allocator);
    defer sch.decoding.deinit(allocator);
    defer sch.cleanup_queue.deinit(allocator);
    defer sch.prefilling.deinit(allocator);
    const slot = try sch.submit(.{ .model = &model, .prompt_ids = &.{1}, .sampling = .{}, .eos_token_ids = &.{}, .max_tokens = 1 });
    // The inference thread's admission step, as `inferenceLoop` runs it.
    sch.queue_mu.lockUncancelable(sch.io);
    const taken = admitForPrefillLocked(&sch, 0);
    _ = sch.pending.orderedRemove(0);
    sch.queue_mu.unlock(sch.io);
    try testing.expectEqual(slot, taken);
    sch.cancelAllInFlight();
    const cancelled = slot.cancelled.load(.acquire);
    endPrefillPass(&sch, slot);
    sch.complete(slot);
    while (sch.cleanup_queue.items.len > 0) sch.cleanup_queue.orderedRemove(0).deinit();
    try testing.expect(cancelled);
}

test "a shutdown drain ends a queued request with an error, a client disconnect still ends it as done" {
    var cfg = ModelConfig{ .num_hidden_layers = 0 };
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.transformer = null;
    model.prefix_cache = null;
    var sch = submitTestScheduler(testing.allocator, 2);
    defer sch.pending.deinit(testing.allocator);
    defer sch.cleanup_queue.deinit(testing.allocator);
    defer sch.prefilling.deinit(testing.allocator);
    const params: SubmitParams = .{ .model = &model, .prompt_ids = &.{1}, .sampling = .{}, .eos_token_ids = &.{}, .max_tokens = 1 };
    const gone = try sch.submit(params);
    gone.cancel();
    try testing.expect(gone.waitNext() == .done);
    const drained = try sch.submit(params);
    sch.cancelAllInFlight();
    // A drained slot never ran: `done` would surface as the default finish reason, `length`.
    try testing.expect(drained.waitNextTimeout(0).? == .err);
    try testing.expect(drained.waitNext() == .err);
    sch.complete(drained);
    sch.complete(gone);
    var batch: [16]*Slot = undefined;
    const take = takeTeardownLocked(&sch, &batch);
    for (batch[0..take.cleanup_n]) |s| s.deinit();
    destroyAbandoned(take.abandoned);
}


test "decode share: the default is 0.25 and reaches /props" {
    const saved = prefill_interleave_cached;
    defer prefill_interleave_cached = saved;
    prefill_interleave_cached = true;
    try testing.expectEqual(@as(f32, 0.25), prefill_decode_share);
    try testing.expectEqual(@as(f32, 0.25), prefillDecodeShare());
}

test "decode share: parse clamps above 0.9 and rejects anything that is not a share" {
    try testing.expectEqual(@as(f32, 0), try parseDecodeShare("0"));
    try testing.expectEqual(@as(f32, 0.5), try parseDecodeShare("0.5"));
    try testing.expectEqual(PREFILL_DECODE_SHARE_MAX, try parseDecodeShare("0.95"));
    try testing.expectEqual(PREFILL_DECODE_SHARE_MAX, try parseDecodeShare("inf"));
    for ([_][]const u8{ "", "abc", "nan", "-0.3", "0.5x" }) |bad| {
        try testing.expectError(error.InvalidDecodeShare, parseDecodeShare(bad));
    }
}

test "decode share: the budget is chunk * S / (1 - S), zero when off" {
    const ms = std.time.ns_per_ms;
    try testing.expectEqual(@as(u64, 0), decodeShareBudgetNs(400 * ms, 0));
    try testing.expectApproxEqAbs(@as(f64, 400 * ms), @as(f64, @floatFromInt(decodeShareBudgetNs(400 * ms, 0.5))), 1e3);
    try testing.expectApproxEqAbs(@as(f64, 3600 * ms), @as(f64, @floatFromInt(decodeShareBudgetNs(400 * ms, 0.9))), 1e4);
    try testing.expectApproxEqAbs(@as(f64, 400 * ms * 3 / 7), @as(f64, @floatFromInt(decodeShareBudgetNs(400 * ms, 0.3))), 1e4);
}

const FakeDecodeTicks = struct {
    left: u32,
    ns: u64,
    calls: u32 = 0,
    fn tick(ctx: *anyopaque) u64 {
        const f: *FakeDecodeTicks = @ptrCast(@alignCast(ctx));
        f.calls += 1;
        if (f.left == 0) return 0;
        f.left -= 1;
        return f.ns;
    }
};

test "decode share: ticks to the budget, keeps one tick at zero, stops when decoders run out" {
    const ms = std.time.ns_per_ms;
    // S=0.5 on a 400 ms chunk with 8 ms ticks: 50 ticks, 400 ms of decode.
    var full = FakeDecodeTicks{ .left = 1000, .ns = 8 * ms };
    const r = runOwedDecodeTicks(0.5, 400 * ms, 8 * ms, &full, FakeDecodeTicks.tick);
    try testing.expectEqual(@as(u32, 50), r.ticks);
    try testing.expectEqual(@as(u64, 400 * ms), r.spent_ns);
    // Share zero retains sushi's single boundary tick.
    for ([_]u64{ 100, 300, 2300, 8000 }) |c| {
        var legacy = FakeDecodeTicks{ .left = 1000, .ns = 8 * ms };
        const l = runOwedDecodeTicks(0, c * ms, 8 * ms, &legacy, FakeDecodeTicks.tick);
        try testing.expectEqual(@as(u32, 1), l.ticks);
    }
    // The decoders finish after 3 more ticks: the 4th call returns 0 and the loop stops.
    var dry = FakeDecodeTicks{ .left = 3, .ns = 8 * ms };
    const d = runOwedDecodeTicks(0.5, 400 * ms, 8 * ms, &dry, FakeDecodeTicks.tick);
    try testing.expectEqual(@as(u32, 4), d.ticks);
    try testing.expectEqual(@as(u32, 4), dry.calls);
    try testing.expectEqual(@as(u64, 32 * ms), d.spent_ns);
    // Nobody decoding at the boundary: the first tick measured 0, no more are tried.
    var none = FakeDecodeTicks{ .left = 1000, .ns = 8 * ms };
    const n = runOwedDecodeTicks(0.5, 2300 * ms, 0, &none, FakeDecodeTicks.tick);
    try testing.expectEqual(@as(u32, 1), n.ticks);
    try testing.expectEqual(@as(u32, 0), none.calls);
}



test "decode share: 0.25 owes a third of the chunk time in ticks" {
    const ms = std.time.ns_per_ms;
    var f = FakeDecodeTicks{ .left = 1000, .ns = 10 * ms };
    const r = runOwedDecodeTicks(0.25, 3000 * ms, 10 * ms, &f, FakeDecodeTicks.tick);
    try testing.expectEqual(@as(u32, 100), r.ticks);
    try testing.expectEqual(@as(u64, 1000 * ms), r.spent_ns);
}

test "decode share: the kill switch disables the effective share" {
    const saved_share = prefill_decode_share;
    const saved_interleave = prefill_interleave_cached;
    defer { prefill_decode_share = saved_share; prefill_interleave_cached = saved_interleave; }
    prefill_decode_share = 0.5;
    prefill_interleave_cached = false;
    try testing.expectEqual(@as(f32, 0), prefillDecodeShare());
    prefill_interleave_cached = true;
    try testing.expectEqual(@as(f32, 0.5), prefillDecodeShare());
    try testing.expectEqual(std.math.maxInt(u64), decodeShareBudgetNs(std.math.maxInt(u64), 0.9));
}


test "decode share: cancelled prefill stops its hosted ticks before touching the scheduler" {
    var cancelled = std.atomic.Value(bool).init(true);
    var ctx = InterleaveCtx{ .sch = undefined, .slot = undefined, .cancelled = &cancelled, .chunk_sw = undefined };
    const result = runOwedDecodeTicks(0.9, 1000, 1, &ctx, interleaveDecodeTickOpaque);
    try testing.expectEqual(@as(u32, 1), result.ticks);
    try testing.expectEqual(@as(u64, 1), result.spent_ns);
}

test "MTP handover: t1 streams at the prefill handover unless it ends the answer or nothing speculates" {
    const eos = [_]u32{ 2, 7 };
    try testing.expectEqual(@as(?u32, 42), handoverToken(true, 0, false, 0, 42, &eos));
    try testing.expectEqual(@as(?u32, null), handoverToken(true, 0, false, 0, 7, &eos));
    try testing.expectEqual(@as(?u32, null), handoverToken(false, 0, false, 0, 42, &eos));
    try testing.expectEqual(@as(?u32, null), handoverToken(true, 1, false, 0, 42, &eos));
    try testing.expectEqual(@as(?u32, null), handoverToken(true, 0, true, 0, 42, &eos));
    try testing.expectEqual(@as(?u32, null), handoverToken(true, 0, false, 5, 42, &eos));
}

test "MTP handover: the first block's echo of the streamed t1 is swallowed once, anything else is published" {
    var slot: Slot = undefined;
    slot.handover_token = 42;
    try testing.expect(slot.takeHandoverEcho(42));
    try testing.expectEqual(@as(?u32, null), slot.handover_token);
    try testing.expect(!slot.takeHandoverEcho(42));
    slot.handover_token = 42;
    try testing.expect(!slot.takeHandoverEcho(43));
    try testing.expectEqual(@as(?u32, null), slot.handover_token);
}

test "an error whose name cannot be copied still ends the consumer and leaves scheduling" {
    var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    var mm = metrics_mod.Metrics.init();
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    sch.allocator = testing.allocator;
    sch.queue_mu = .init;
    sch.metrics = &mm;
    sch.live_sessions = undefined;
    sch.live_session_count = 0;
    sch.decoding = .empty;
    defer sch.decoding.deinit(testing.allocator);
    var model: model_registry_mod.LoadedModel = undefined;
    model.id = "org/err-alloc";
    model.prefix_cache = null;
    var slot: Slot = undefined;
    slot.io = testing.io;
    slot.allocator = failing.allocator();
    slot.model = &model;
    slot.cache = .{ .entries = &.{}, .step = 0, .allocator = testing.allocator, .config = .dense };
    slot.ssm_entries = null;
    slot.ring_cps = .{};
    slot.restored_entry = 0;
    slot.full_prompt = &.{};
    slot.prompt_tokens = 4;
    slot.completion_tokens = 0;
    slot.cached_tokens = 0;
    slot.request_id = 0;
    slot.max_tokens = 8;
    slot.request_start_ts = std.Io.Timestamp.now(testing.io, .boot);
    slot.legacy_gen = null;
    slot.glm5_request = null;
    slot.state = .decoding;
    slot.finished = false;
    slot.error_code = null;
    slot.cancelled = .init(false);
    slot.out_mu = .init;
    slot.out_cond = .init;
    slot.out_event = .unset;
    slot.out_buf = .empty;
    slot.out_idx = 0;
    try sch.decoding.append(testing.allocator, &slot);
    slot.markError("MlxFailure");
    try testing.expect(slot.waitNextTimeout(0).? == .err);
    try testing.expect(slot.waitNext() == .err);
    cullDecoding(&sch);
    try testing.expectEqual(@as(usize, 0), sch.decoding.items.len);
}

test "publishLiveKvResidency snapshots decode and prefill rows with stable ids" {
    var mm = metrics_mod.Metrics.init();
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    sch.metrics = &mm;
    sch.live_sessions = undefined;
    sch.live_session_count = 999; // a stale count must not survive a publish
    sch.decoding = .empty;
    var model: model_registry_mod.LoadedModel = undefined;
    model.id = "org/live-test";
    model.prefix_cache = null;
    model.transformer = null;
    var slot: Slot = undefined;
    // Only the fields the publish path reads: rows are value copies.
    slot.model = &model;
    slot.legacy_gen = null;
    slot.cache = .{ .entries = &.{}, .step = 0, .allocator = testing.allocator, .config = .dense };
    slot.ssm_entries = null;
    slot.glm5_request = null;
    slot.ring_cps = .{};
    slot.restored_entry = 0;
    slot.full_prompt = &.{};
    slot.prompt_tokens = 1000;
    slot.completion_tokens = 7;
    slot.cached_tokens = 900;
    slot.request_id = 42;
    slot.max_tokens = 32000;
    slot.request_start_ts = std.Io.Timestamp.now(testing.io, .boot);
    var ptrs = [_]*Slot{&slot};
    sch.decoding.items = &ptrs;
    sch.decoding.capacity = ptrs.len;

    publishLiveKvResidency(&sch, null);
    try testing.expectEqual(@as(usize, 1), sch.live_session_count);
    const d = &sch.live_sessions[0];
    try testing.expectEqualStrings("org/live-test", d.model());
    try testing.expectEqual(metrics_mod.Session.Phase.decode, d.phase);
    // Decode row: prompt_tokens + completion_tokens, the generated tail split
    // out for the dashboard's context-fill bar.
    try testing.expectEqual(@as(u32, 1007), d.context_tokens);
    try testing.expectEqual(@as(u32, 900), d.cached_tokens);
    try testing.expectEqual(@as(u32, 7), d.generated_tokens);
    try testing.expectEqual(@as(u64, 42), d.request_id);
    try testing.expectEqual(@as(u32, 32000), d.max_tokens);
    try testing.expectApproxEqAbs(@as(f64, 0), d.elapsed_seconds, 10.0);

    // An empty KV cache and no SSM entries bill EXACTLY zero — the state
    // bytes come from the slot's own arrays, never a modeled estimate.
    try testing.expectEqual(@as(u64, 0), d.state_bytes);

    // A republish (the next poll) carries the SAME request_id — the row
    // identity a per-request downstream key needs — and a non-decreasing age.
    const first_elapsed = d.elapsed_seconds;
    publishLiveKvResidency(&sch, null);
    try testing.expectEqual(@as(u64, 42), sch.live_sessions[0].request_id);
    try testing.expect(sch.live_sessions[0].elapsed_seconds >= first_elapsed);

    // Prefilling slot (not in `decoding` yet) gets its own row: two live rows.
    var prompt_ids: [64]u32 = undefined;
    slot.full_prompt = &prompt_ids;
    publishLiveKvResidency(&sch, &slot);
    try testing.expectEqual(@as(usize, 2), sch.live_session_count);
    try testing.expectEqual(metrics_mod.Session.Phase.prefill, sch.live_sessions[0].phase);
    // The prefill row bills the FULL prompt (the decode row's `prompt_tokens`
    // only lands at prefill completion): 64 + 7.
    try testing.expectEqual(@as(u32, 71), sch.live_sessions[0].context_tokens);
    try testing.expectEqual(metrics_mod.Session.Phase.decode, sch.live_sessions[1].phase);

    // The step-5 cull republishes without the prefilling row; the slot still
    // decoding keeps only its decode row.
    publishLiveKvResidency(&sch, null);
    try testing.expectEqual(@as(usize, 1), sch.live_session_count);
    // Culling the slot empties the snapshot: a poll must never resurrect a
    // finished row.
    _ = sch.decoding.orderedRemove(0);
    publishLiveKvResidency(&sch, null);
    try testing.expectEqual(@as(usize, 0), sch.live_session_count);

    // Zero-when-off: with no sink the publish is a single reset — no rows.
    var ptrs2 = [_]*Slot{&slot};
    sch.decoding.items = &ptrs2;
    sch.metrics = null;
    publishLiveKvResidency(&sch, &slot);
    try testing.expectEqual(@as(usize, 0), sch.live_session_count);
}

test "the live KV bill counts ring restore points and nets out a donated checkout the hot cache bills" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    sch.metrics = null;
    sch.live_session_count = 0;
    sch.resident_live_kv_bytes = .init(999);
    var model: model_registry_mod.LoadedModel = undefined;
    model.id = "org/live-test";
    model.prefix_cache = prefix_cache_mod.HotPrefixCache.init(testing.allocator, 4);
    model.transformer = null;
    defer model.prefix_cache.?.entries.deinit(testing.allocator);
    sch.hot_prefix_cache = &model.prefix_cache.?;

    var cp_entries = [_]transformer_mod.KVCacheEntry{transformer_mod.newEmptyKVEntry()};
    _ = mlx.mlx_zeros(&cp_entries[0].keys, &[_]c_int{ 1, 1, 8, 4 }, 4, .bfloat16, s);
    defer _ = mlx.mlx_array_free(cp_entries[0].keys);
    _ = mlx.mlx_zeros(&cp_entries[0].values, &[_]c_int{ 1, 1, 8, 4 }, 4, .bfloat16, s);
    defer _ = mlx.mlx_array_free(cp_entries[0].values);
    cp_entries[0].initialized = true;
    const cp_bytes: u64 = 2 * (8 * 4) * 2;

    var slot: Slot = undefined;
    slot.model = &model;
    slot.legacy_gen = null;
    slot.cache = .{ .entries = &.{}, .step = 0, .allocator = testing.allocator, .config = .dense };
    slot.ssm_entries = null;
    slot.glm5_request = null;
    slot.ring_cps = .{ .fork = .{ .entries = &cp_entries, .step = 8, .allocator = testing.allocator, .config = .dense } };
    slot.full_prompt = &.{};
    slot.prompt_tokens = 8;
    slot.completion_tokens = 0;
    slot.cached_tokens = 8;
    slot.request_id = 1;
    slot.max_tokens = 16;
    slot.request_start_ts = std.Io.Timestamp.now(testing.io, .boot);
    slot.restored_entry = 7;
    var ptrs = [_]*Slot{&slot};
    sch.decoding = .empty;
    sch.decoding.items = &ptrs;
    sch.decoding.capacity = ptrs.len;

    // `/props` reads the bill without `--metrics`: no rows, but the bytes are published.
    publishLiveKvResidency(&sch, null);
    try testing.expectEqual(cp_bytes, sch.resident_live_kv_bytes.load(.monotonic));
    try testing.expectEqual(@as(usize, 0), sch.live_session_count);

    try model.prefix_cache.?.entries.append(testing.allocator, .{
        .tokens = &.{},
        .has_tools = false,
        .snapshot = .{ .entries = &.{}, .step = 0, .allocator = testing.allocator, .config = .dense },
        .last_used = 1,
        .quant_config = .dense,
        .kv_bytes = 48,
        .checked_out_by = @intFromPtr(&slot),
        .checkout_donated = true,
        .donated_bytes = 48,
    });
    var mm = metrics_mod.Metrics.init();
    sch.metrics = &mm;
    publishLiveKvResidency(&sch, null);
    try testing.expectEqual(cp_bytes - 48, sch.resident_live_kv_bytes.load(.monotonic));
    const row = &sch.live_sessions[0];
    try testing.expectEqual(cp_bytes, row.state_bytes);
    try testing.expectEqual(@as(u64, 7), row.entry_id);
}

test "ensureLoaded: a load refused because every other resident model is pinned leaves the registry unlocked and unpinned" {
    const io = testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"not_a_served_arch\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"chat_template\":\"x\"}" });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];

    var reg = try ModelRegistry.init(testing.allocator, io, null, 1, 0, null);
    defer reg.deinit();
    const resident = try reg.registerStub("a", "a", 10);
    const target = try reg.registerStub("b", dir, 10);
    reg.mutex.lockUncancelable(io);
    reg.markReadyLocked(resident, 10);
    reg.mutex.unlock(io);
    _ = resident.refcount.fetchAdd(1, .acq_rel); // an in-flight request

    var sch: Scheduler = undefined;
    sch.registry = reg;
    sch.io = io;
    sch.allocator = testing.allocator;
    sch.ssd_budget_bytes = 0;
    sch.mtp_enabled = true;
    sch.mtp_explicit = false;

    try testing.expectError(error.NotEnoughMemory, sch.ensureLoaded("b"));
    try testing.expectEqual(model_registry_mod.LoadState.unloaded, target.state);
    try testing.expectEqual(@as(u32, 0), target.refcount.load(.acquire));
    // The failed claim unlocked once: a second unlock would leave this free or release another thread's hold.
    try testing.expect(reg.mutex.tryLock());
    reg.mutex.unlock(io);
    resident.refcount.store(0, .release);
}

pub var queue_test_prompt: [4096]u32 = undefined;
