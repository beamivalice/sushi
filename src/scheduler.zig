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
pub var slot_vision_free_test_hook: ?*const fn (mlx.mlx_array) void = null;
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const mimo_forward = @import("mimo_forward.zig");
const qwen4_qsa = @import("qwen4_qsa.zig");
const tokenizer_mod = @import("tokenizer.zig");
const generate_mod = @import("generate.zig");
const rp_mod = @import("reasoning_protocol.zig");
const mtp_mod = @import("mtp.zig");
const depth_bounds = @import("mtp_depth_bounds.zig");
const mimo_mtp = @import("mimo_mtp.zig");
const chip_mod = @import("chip.zig");
const model_mod = @import("model.zig");
const vision_mod = @import("vision.zig");
const chat_mod = @import("chat.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const restore_dump = @import("restore_dump.zig");
const metrics_mod = @import("metrics.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const tokenize_cache_mod = @import("tokenize_cache.zig");
const model_registry_mod = @import("model_registry.zig");
const model_settings = @import("model_settings.zig");
const mtp_acceptance_mod = @import("mtp_acceptance.zig");
const model_discovery = @import("model_discovery.zig");
const log = @import("log.zig");
const io_util = @import("io_util.zig");
const status = @import("status.zig");
const sleep_inhibit = @import("sleep_inhibit.zig");
const exl3_kernels = @import("sushi_exl3").kernels;

const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;
const SSMCacheEntry = transformer_mod.SSMCacheEntry;
const ForwardCtx = transformer_mod.ForwardCtx;
const ModelConfig = model_mod.ModelConfig;
const Tokenizer = tokenizer_mod.Tokenizer;
const Generator = generate_mod.Generator;
const SamplingParams = generate_mod.SamplingParams;
const dflash_mod = @import("dflash.zig");
const glm5_forward_mod = @import("glm5_forward.zig");
const round_cost_mod = @import("round_cost.zig");
const group_cost_mod = @import("mtp_group_cost.zig");
const expert_stream_mod = @import("expert_stream.zig");
const DflashModel = dflash_mod.DflashModel;
const VisionEncoder = vision_mod.VisionEncoder;
const Weights = model_mod.Weights;
const ChatConfig = chat_mod.ChatConfig;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;

/// Phase A1: model-load plan executed on the scheduler's inference thread.
///
/// mlx 0.31.2 uses thread-local GPU streams: every `mlx_*` op binds to the
/// stream of the calling thread, and JIT-compiled closures are tied to that
/// stream too. If main loads the model and the scheduler later calls forward,
/// mlx aborts with "no Stream(gpu, N) in current thread". Solution: the
/// scheduler's inference thread does the load itself so the stream is bound
/// on the right thread from t0.
///
/// CPU-only state (parsed config, tokenizer, chat config) is loaded by main
/// and passed in by reference. The mlx-allocating pieces (weights tensors,
/// Transformer, vision encoder, drafter, JIT compile, warmup) all run on the
/// inference thread before `init()` returns.
pub const LoadParams = struct {
    /// Registry that owns the entry to populate + provides snapshot/eviction
    /// bookkeeping. Outlives the scheduler.
    registry: *ModelRegistry,
    /// The (pre-registered) LoadedModel stub that will be promoted to
    /// `.ready` by the inference-thread load. Holds id/path on entry;
    /// `loadModelOnInferenceThread` installs weights/transformer/vision/
    /// drafter/tokenizer/chat_config/config into this slot. Lifetime
    /// matches the registry.
    entry: *LoadedModel,
    /// Heap-allocated parsed config. Ownership transfers to `entry` on
    /// successful load; caller must NOT deinit/free externally after
    /// `Scheduler.init` returns.
    config: *ModelConfig,
    /// Heap-allocated tokenizer. Ownership transfers to `entry`.
    tok: *Tokenizer,
    /// Heap-allocated chat config. Ownership transfers to `entry`.
    chat_config: *ChatConfig,
    /// Path to the model directory. Borrowed; outlive scheduler.
    model_dir: []const u8,
    /// Path to the assistant drafter checkpoint. Empty disables the drafter.
    /// Borrowed; outlive scheduler.
    drafter_dir: []const u8 = "",
    /// `--no-drafter`: never load a drafter, including one MERGED into the
    /// checkpoint. `drafter_dir == ""` stopped meaning "off" the moment a
    /// checkpoint could carry its own, so the opt-out needs its own bit.
    no_drafter: bool = false,
    /// Auto-load the Qwen native MTP sidecar when the model dir ships one.
    mtp_enabled: bool = true,
    /// `--mtp` / `--no-mtp` was given: `mtp_enabled` then outranks the per-model `mtp`.
    mtp_explicit: bool = false,
    mtp_head_kv_quant: bool = false,
    /// Max MTP draft depth (CLI --mtp-max-depth; 0 = auto, resolved by
    /// generate_mod.resolveMtpDepthCap at load/Generator init).
    mtp_depth: u32 = 0,
    /// Whether to also load vision-tower weights. Combined with
    /// `config.has_vision` — false here disables vision regardless of config.
    load_vision: bool = false,
    /// Eager warmup: fault weight pages + run a tiny forward to JIT-compile
    /// the decode path on the inference thread. Adds ~600-900 ms at boot but
    /// keeps the first user request fast.
    warmup_eager: bool = true,
    /// Drafter block size (caller computed via `drafter.recommendedBlockSize`).
    /// Ignored when `drafter_dir` is empty.
    draft_block_size: u32 = 4,
    /// Whether the user passed --draft-block-size explicitly (used for
    /// human-readable startup logging). Ignored when `drafter_dir` is empty.
    draft_block_size_explicit: bool = false,
    /// KV-cache storage backend: `--kv-quant`, else the kv8 engine default. Stored on every
    /// per-slot KVCache and consulted at every read/write boundary.
    kv_quant_config: transformer_mod.KVQuantConfig = transformer_mod.KVQuantConfig.engine_default,
    /// `--kv-quant` was given: the flag then outranks the per-model `kv_quant`.
    kv_quant_explicit: bool = false,
    /// Per-model hot prefix cache capacity (count). 0 disables.
    prefix_cache_capacity: u32 = 1,
    prefix_cache_ram_enabled: bool = true,
    /// Per-model hot prefix cache KV-bytes budget. 0 disables the byte cap.
    prefix_cache_mem_bytes: u64 = 0,
    /// Clamp the hot-cache byte budget against live post-load headroom
    /// (`server.prefixCacheMemForLoad`) — a pointer because the scheduler
    /// deliberately has no server.zig import. Null = no clamp (tests).
    prefix_cache_mem_resolver: ?*const fn (*model_mod.ModelConfig, u64, BudgetRevise, *u64) u64 = null,
    /// SSD tier byte budget for the hot prefix cache (`--prefix-cache-disk`).
    /// 0 disables persistence. Attached per model at load for pure-attention
    /// archs; entries live under `~/.sushi/kv-cache/<fingerprint>`.
    prefix_cache_disk_bytes: u64 = 0,
    /// Sizes the SSD budget for this model and logs the `Allocating` line (`server.prefixCacheDiskForLoad`);
    /// takes the cache directory. Null = `prefix_cache_disk_bytes` as given (tests).
    prefix_cache_disk_resolver: ?*const fn (*model_mod.ModelConfig, []const u8, u64) ?kv_disk_cache.DefaultDiskBudget = null,
    expert_cache_bytes: u64 = 0,
    ssd_budget_bytes: u64 = 0,
    expert_cache_fit_resolver: ?*const fn (*const model_mod.ModelConfig, u64) anyerror!void = null,
    /// Phase 1 (perf-plan): SSM/conv state snapshot stride during prefill.
    /// 0 = disabled (hybrid models bypass the hot prefix cache). Non-zero
    /// enables hybrid in `HotPrefixCache.shouldUse` and triggers per-stride
    /// snapshots in the Generator's prefill loop. Default 0 here so callers
    /// that don't set it (legacy paths) preserve pre-Phase-1 behavior;
    /// `main.zig` overrides via `--ssm-checkpoint-stride` for the serve path.
    ssm_checkpoint_stride: u32 = 0,
    /// Phase 1: cap on snapshots retained per request.
    ssm_checkpoint_max: u32 = 16,
    /// Iteration 2 (perf-plan Phase 4 #3): per-LoadedModel LRU cache
    /// of chat-template render+tokenize results. 0 disables the cache
    /// (useful for ablation benches / debugging). Default 4 matches
    /// `prefix_cache_capacity` — most warm-reuse benches exercise a
    /// handful of repeated prompts, and full chat conversations bump
    /// this counter anyway via LRU as new turns arrive.
    tokenize_cache_entries: u32 = 4,
    /// Headless boot: skip the startup load entirely and run the inference
    /// loop idle. The registry holds discovery stubs (or nothing); the first
    /// model — chat or media — loads on demand via `/v1/load-model`. `entry`/
    /// `config`/`tok`/`chat_config` are still required (they seed the
    /// scheduler's borrowed-view fields) but are never installed on an entry.
    no_initial_load: bool = false,
    /// Optional metrics sink. Null when --metrics is off (the default).
    /// Stored on the Scheduler and read by the `finishSlot` per-request funnel.
    metrics: ?*metrics_mod.Metrics = null,
    /// The --ctx-size launch flag (0 = not given).
    ctx_size: u32 = 0,
};

/// Submit-time parameters. `prompt_ids` and `eos_token_ids` are duped into the
/// slot so callers can free their copies immediately. `vision_embeddings`
/// ownership transfers into the slot when non-null (the slot will free on
/// deinit); a `submit` that fails hands it to the inference thread to free.
pub const SubmitParams = struct {
    prompt_ids: []const u32,
    /// Full original prompt for PLD lookup. When null, defaults to
    /// `prompt_ids` (PLD's lookup table = full_prompt + generated).
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
    enable_pld: bool = false,
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
    pld_draft_len: u32 = 5,
    pld_key_len: u32 = 3,
    /// Phase 2 (Plan ricky): route SDPA through `kv_quant.quantAttention`
    /// instead of dequant + dense SDPA. No effect when the cache scheme
    /// isn't `.affine`. Default false → unchanged behavior.
    kv_attn_fused: bool = false,
    /// Vision embeddings spliced at image-token positions during prefill.
    /// Ownership transferred to the slot; freed on slot.deinit.
    vision_embeddings: ?mlx.mlx_array = null,
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

/// Set by `server.installPrefillAdmission`: does a request of this shape fit in GPU memory
/// right now? Null (unit tests, no HTTP server) disables evict-to-admit. The trailing warm
/// arguments are the restored rows, their capacity, and whether the restore checked the
/// entry out (the only restore whose rows the request will not allocate); then MTP and the GLM
/// checkpoints the request keeps (`server.WarmPrefix.checkpoints`).
pub var prefill_admission_fits: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, u64, u64, bool, bool, u32) bool = null;

/// {needed, available, commit} of the same cold bill, live memory re-read (`server.prefillBillNumbersNow`).
/// Null (unit tests) skips the inference thread's hold for an ungated arch.
pub var prefill_admission_numbers: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, bool) MemoryBill = null;

/// The prefill width this request should run at, chosen against live post-eviction memory
/// (`server.requestPrefillChunkNow`). Null keeps the model's load-time pin.
pub var prefill_request_chunk: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, u64, u64, bool, bool) u32 = null;
/// The same chooser for the load-time prefill meter, installed before the load (the one above
/// arrives after it); only the meter reads it.
pub var prefill_ubench_chunk: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, u64, u64, bool, bool) u32 = null;

pub const PostEvictionWidth = struct {
    /// The width the prefill runs at. Always the re-ask.
    width: u32,
    widened: bool,
    /// The re-ask came back narrower than the admitted width: memory moved between the reads.
    moved: bool,
};

/// The prefill width to run, given the width admission was billed at before the eviction
/// pass and the width re-asked against live memory after it. The re-ask wins in both
/// directions: `@max` would widen on memory that is gone. `admitted == 0` = no pass ran.
pub fn postEvictionPrefillChunk(admitted: u32, reasked: u32) PostEvictionWidth {
    return .{
        .width = reasked,
        .widened = admitted != 0 and reasked > admitted,
        .moved = admitted != 0 and reasked < admitted,
    };
}

/// A GLM request whose prefix-cache checkpoints do not fit even with the hot cache evicted keeps
/// fewer rather than be refused: the most below `cap` that `fits` admits, 0 = none; null = not
/// even that.
pub fn fewerCheckpointsToAdmit(cap: u32, ctx: ?*anyopaque, fits: *const fn (?*anyopaque, u32) bool) ?u32 {
    var n = cap;
    while (n > 0) {
        n -= 1;
        if (fits(ctx, n)) return n;
    }
    return null;
}

/// Does the inference thread run the evict-to-admit pass for this model? The connection
/// thread's credits (`server.creditedAdmissionBill`) read the same predicate.
pub fn admissionPassArmed(cfg: ?*const ModelConfig) bool {
    const c = cfg orelse return false;
    return c.admissionEvictsHotCache();
}

/// Asked by the prefill loop: the width of the next chunk, re-priced at every chunk boundary
/// (`server.adaptivePrefillWidthNow`). Null keeps the admitted width.
pub var prefill_chunk_adapt: ?*const fn (
    *const model_mod.ModelConfig,
    u64,
    usize,
    u32,
    u32,
    *generate_mod.AdaptiveWidthState,
    u64,
) u32 = null;

/// Whether the per-chunk adaptive width is enabled for this model. The hook above is
/// installed process-wide, so its presence is not the arch gate.
pub var prefill_chunk_adaptive_enabled: ?*const fn (*const model_mod.ModelConfig) bool = null;

/// What `Generator.InitOptions.adaptive_chunk_width` gets for a slot.
pub fn adaptiveChunkWidthFor(cfg: ?*const model_mod.ModelConfig) bool {
    const c = cfg orelse return false;
    const enabled = prefill_chunk_adaptive_enabled orelse return false;
    return enabled(c);
}

/// Re-price a widen after the interleave tick (`server.adaptivePrefillWidenStillFits`). Null declines every widen.
pub var prefill_chunk_widen_ok: ?*const fn (
    *const model_mod.ModelConfig,
    u64,
    usize,
    u32,
    u64,
) bool = null;

/// Logs the numbers the estimator compared on a refusal.
pub var prefill_admission_refused_log: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, u64, u64, bool, bool, u32) ?model_registry_mod.MemoryContextRefusal = null;

/// GPU bytes a vision encode may still take, read live (`server.visionEncodeAvailable`). Null skips the recheck.
pub var vision_encode_available: ?*const fn (*const model_mod.ModelConfig) u64 = null;

/// The `VisionEncodeRequest.error_name` of an encode whose bill no longer fits when its turn comes.
pub const vision_headroom_error = "VisionHeadroomExceeded";

/// Tests only: stands in for the tower on one image block.
var vision_block_test_hook: ?*const fn (VisionImagePixels) anyerror!mlx.mlx_array = null;

/// Invalidate the published hot-cache budget on unload/switch (`server.clearResolvedPrefixCacheMem`).
pub var hot_cache_budget_invalidate: ?*const fn () void = null;

pub const orphan_vision_cap = 16;

/// Inference thread only (the sole mlx caller), like every other free of a slot's arrays.
pub fn freeVisionArray(ve: mlx.mlx_array) void {
    if (@import("builtin").is_test and slot_vision_free_test_hook != null) {
        slot_vision_free_test_hook.?(ve);
    } else {
        _ = mlx.mlx_array_free(ve);
    }
}

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
    enable_pld: bool,
    enable_drafter: bool,
    dflash: ?*DflashModel,
    drafter_block_size: u32,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    mtp: ?generate_mod.MtpHeadRef,
    mtp_depth: u32,
    pld_draft_len: u32,
    pld_key_len: u32,
    /// Phase 2 (Plan ricky): see SubmitParams.kv_attn_fused.
    kv_attn_fused: bool,
    cached_tokens: u32,
    logprobs_n: u32,
    serial_reason_logged: bool = false,
    memory_hold_logged: bool = false,
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
    cancelled: std.atomic.Value(bool),
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
    logprobs_buf: std.ArrayList(generate_mod.LogprobResult),

    /// Initialize but do NOT take ownership of caches — those are allocated
    /// inside `init` from the slot's allocator.
    fn init(
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
        // uses to allocate `xfm.ssm_entries` (transformer.zig: `has_hybrid_layers`
        // OR `full_attention_interval > 0`). Without this branch the
        // slot's `ctx.ssm_entries` is null and `forwardMoeWith`'s
        // linear-attention layers crash on `ctx.ssm_entries.?` for Qwen 3.5/3.6
        // MoE (which carries GatedDeltaNet inside its MoE structure but does
        // NOT set `has_hybrid_layers`). Pure-attention MoE (qwen3_moe, Gemma 4
        // MoE — isMoe() but interval == 0) must NOT get entries: a non-null
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
            params.vision_embeddings != null,
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
            .enable_pld = params.enable_pld,
            // An external drafter does not carry M-RoPE positions for image tokens.
            .enable_drafter = assistantSidecarEnabledForRequest(
                params.enable_drafter,
                params.vision_embeddings != null,
            ),
            .dflash = params.dflash,
            .drafter_block_size = params.drafter_block_size,
            .enable_mtp = params.enable_mtp,
            .allow_batch_mtp = params.allow_batch_mtp,
            .mtp = params.mtp,
            .mtp_depth = params.mtp_depth,
            .pld_draft_len = params.pld_draft_len,
            .pld_key_len = params.pld_key_len,
            .kv_attn_fused = params.kv_attn_fused,
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
            .kv_attn_fused = params.kv_attn_fused,
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
    pub fn pushTokenWithLogprob(self: *Slot, t: u32, lp: ?generate_mod.LogprobResult) void {
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
        out: *std.ArrayList(generate_mod.LogprobResult),
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
    pub fn takeLogprobs(self: *Slot, limit: usize) ?[]generate_mod.LogprobResult {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        while (self.logprobs_buf.items.len > limit) self.allocator.free(self.logprobs_buf.pop().?.top_logprobs);
        if (self.logprobs_buf.items.len == 0) return null;
        return self.logprobs_buf.toOwnedSlice(self.allocator) catch null;
    }

    /// Inference thread: signal normal completion. Safe to call multiple
    /// times (idempotent on `finished`).
    fn markFinished(self: *Slot, reason: []const u8) void {
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

    fn markError(self: *Slot, name: []const u8) void {
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
            if (self.cancelled.load(.acquire)) return .{ .done = {} };
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
    /// while the GPU ground ghosts; observed live with Claude Code + a
    /// 40K-token MCP prompt on gemma-4-12b).
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
                return .{ .done = {} };
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

    /// Connection thread: signal cancellation. The inference thread will
    /// drop this slot at the next tick boundary.
    /// Connection thread: the request's own stop sequence completed; end generation as a normal stop.
    pub fn cancelOnStop(self: *Slot) void {
        self.stop_hit.store(true, .release);
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

/// Phase A4: pixel data for a single image, decoded by the connection
/// thread (CPU only — stb_image / libwebp). The inference thread wraps this
/// in an `mlx_array` via `mlx_array_new_data` and runs the vision encoder.
pub const VisionImagePixels = struct {
    /// Raw bytes holding float32 pixel data. Gemma: CHW (3 × H × W × 4). Qwen3-VL:
    /// merge-order pixel_values (N × C·tps·ps·ps × 4). Borrowed; must outlive the
    /// encodeVision call (which blocks until completion).
    pixels: []const u8,
    width: u32,
    height: u32,
    /// Qwen3-VL only: full patch grid (0 ⇒ Gemma CHW). Selects QwenVision.
    grid_h: u32 = 0,
    grid_w: u32 = 0,
};

/// Qwen3-VL video: pre-patchified pixel_values for ALL `grid_t` temporal-patch
/// groups, concatenated (see `vision_common.buildPixelValuesVideo`). Borrowed;
/// must outlive the encodeVision call.
pub const VisionVideoPixels = struct {
    pixels: []const u8,
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
};

/// Phase A4: vision-encode work item. Conn thread fills `images` (raw pixel
/// data, CPU-only) and calls `Scheduler.encodeVision`, which posts the
/// request and blocks until the inference thread fills `result` and signals
/// `done`. Ownership of `result` transfers to the caller on success — pass
/// to `scheduler.submit(.{ .vision_embeddings = arr, ... })` and the slot's
/// `deinit` will free it.
pub const VisionEncodeRequest = struct {
    /// Plan 05 Phase D: target model whose `vision_encoder` services this
    /// request. The conn thread holds a refcount (via `ensureLoaded`) for
    /// the duration of the call.
    model: *model_registry_mod.LoadedModel,
    /// Per-image float32 CHW pixel buffers. Borrowed; must outlive the call.
    images: []const VisionImagePixels,
    /// Per-video pre-patchified pixel buffers. Borrowed; must outlive the call.
    /// Qwen-only (video_token_id != 0) — empty on every other arch.
    videos: []const VisionVideoPixels = &.{},
    /// Gemma 4 12B unified audio: per-clip raw float32-LE 16 kHz mono sample
    /// buffers. Borrowed; must outlive the call. The inference thread frames
    /// each into 640-sample tokens and projects them through the audio embedder.
    audio: []const []const u8 = &.{},
    /// The prompt order of the image and video blocks (each kind in its own
    /// list order); empty means every image, then every video.
    order: []const chat_mod.MediaPart.Kind = &.{},
    /// Output: encoded embedding tensor on success — image and video soft
    /// tokens in `order`, then audio soft tokens, concatenated along the token
    /// axis so the splice scatters each row into its placeholder.
    /// Ownership transfers to the caller.
    result: ?mlx.mlx_array = null,
    /// Output: number of vision / video / audio soft tokens in `result` (in
    /// that order). The caller inserts exactly this many image / video / audio
    /// placeholders.
    n_vision_tokens: usize = 0,
    n_video_tokens: usize = 0,
    /// `server.visionEncodeBill` of this request; the inference thread refuses the encode when
    /// the GPU has less left when its turn comes. 0 = unbilled.
    bill_bytes: u64 = 0,
    /// Output: what the GPU had left when `vision_headroom_error` refused the encode.
    available_bytes: u64 = 0,
    /// Output: error name on failure. Owned by `allocator`; caller frees.
    error_name: ?[]const u8 = null,
    /// Done flag (under done_mu). Caller's wait-loop drains the cond when
    /// this flips true.
    done: bool = false,
    allocator: std.mem.Allocator,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
};

/// Phase: embedding work item for encoder-only models. Conn thread fills
/// `token_seqs` and calls `Scheduler.computeEmbeddings`; the inference
/// thread services the request via `generate.computeEmbeddingsBatch` — one
/// padded, key-masked GPU forward per EMBED_MAX_BATCH chunk — and writes
/// the float vectors into `results` (caller frees). Mirrors the
/// VisionEncodeRequest pattern.
pub const EmbedRequest = struct {
    /// Plan 05 Phase D: target model whose `transformer` services this
    /// request. The conn thread holds a refcount for the duration.
    model: *model_registry_mod.LoadedModel,
    /// Tokenized inputs, one slice per text. Borrowed; must outlive the call.
    token_seqs: []const []const u32,
    /// Output: one pooled L2-normalized embedding per input on success.
    /// Rows + outer slice owned by `allocator`; caller frees.
    results: ?[][]f32 = null,
    /// Output: error name on failure. Owned by `allocator`; caller frees.
    error_name: ?[]const u8 = null,
    done: bool = false,
    allocator: std.mem.Allocator,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
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
    draft_block_size: u32 = 4,
    draft_block_size_explicit: bool = false,
    kv_quant_config: transformer_mod.KVQuantConfig = transformer_mod.KVQuantConfig.engine_default,
    kv_quant_explicit: bool = false,
    prefix_cache_capacity: u32 = 1,
    prefix_cache_ram_enabled: bool = true,
    prefix_cache_mem_bytes: u64 = 0,
    prefix_cache_mem_resolver: ?*const fn (*model_mod.ModelConfig, u64, BudgetRevise, *u64) u64 = null,
    /// SSD tier byte budget (mirrors `LoadParams.prefix_cache_disk_bytes`).
    prefix_cache_disk_bytes: u64 = 0,
    prefix_cache_disk_resolver: ?*const fn (*model_mod.ModelConfig, []const u8, u64) ?kv_disk_cache.DefaultDiskBudget = null,
    expert_cache_bytes: u64 = 0,
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
    prefix_cache_disk_resolver: ?*const fn (*model_mod.ModelConfig, []const u8, u64) ?kv_disk_cache.DefaultDiskBudget,
    expert_cache_bytes: u64,
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
    draft_block_size: u32,
    draft_block_size_explicit: bool,

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
    /// Host bytes every resident model's SSD writer holds or may hold (`LoadedModel.disk_stats`
    /// summed), outside MLX's accounting; read through `diskWriterHostBytes`.
    disk_writer_host_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
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
    /// Pending embedding requests (encoder-only models). Same shape as
    /// vision_queue; serviced inline between decode ticks.
    embed_queue: std.ArrayList(*EmbedRequest),
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
    /// Metrics sink. Null when --metrics is off. Populated from LoadParams.
    /// Read once per REQUEST in `finishSlot` — never on the per-token path.
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
            .drafter_block_size = params.draft_block_size,
            .kv_quant_config = params.kv_quant_config,
            .kv_quant_explicit = params.kv_quant_explicit,
            .ctx_size_flag = params.ctx_size,
            .prefix_cache_capacity = params.prefix_cache_capacity,
            .prefix_cache_ram_enabled = params.prefix_cache_ram_enabled,
            .prefix_cache_mem_bytes = params.prefix_cache_mem_bytes,
            .prefix_cache_mem_resolver = params.prefix_cache_mem_resolver,
            .prefix_cache_disk_bytes = params.prefix_cache_disk_bytes,
            .prefix_cache_disk_resolver = params.prefix_cache_disk_resolver,
            .expert_cache_bytes = params.expert_cache_bytes,
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
            .draft_block_size = params.draft_block_size,
            .draft_block_size_explicit = params.draft_block_size_explicit,
            .drafter_path = params.drafter_dir,
            .hot_prefix_cache = null,
            .max_concurrent = cap,
            .force_batched = force_batched,
            .queue_mu = .init,
            .queue_cond = .init,
            .pending = std.ArrayList(*Slot).empty,
            .decoding = std.ArrayList(*Slot).empty,
            .vision_queue = std.ArrayList(*VisionEncodeRequest).empty,
            .embed_queue = std.ArrayList(*EmbedRequest).empty,
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
        // Vision/embed queues should be empty (encodeVision/computeEmbedding
        // block until done) but guard against shutdown-mid-encode by signaling
        // done with an error.
        for (self.vision_queue.items) |req| {
            req.done_mu.lockUncancelable(self.io);
            req.error_name = self.allocator.dupe(u8, "Shutdown") catch null;
            req.done = true;
            req.done_cond.broadcast(self.io);
            req.done_mu.unlock(self.io);
        }
        self.vision_queue.deinit(self.allocator);
        for (self.embed_queue.items) |req| {
            req.done_mu.lockUncancelable(self.io);
            req.error_name = self.allocator.dupe(u8, "Shutdown") catch null;
            req.done = true;
            req.done_cond.broadcast(self.io);
            req.done_mu.unlock(self.io);
        }
        self.embed_queue.deinit(self.allocator);
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
        // meanwhile, so exclusive admission sees it until it reaches the cleanup queue.
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
        for (self.pending.items) |slot| slot.cancel();
        for (self.prefilling.items) |slot| slot.cancel();
        for (self.decoding.items) |slot| slot.cancel();
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

        const settings_budget = resolveSsdBudget(self.ssd_budget_bytes, owned.config.ssd_budget_gb_override, owned.config.supportsExpertStreaming()).bytes;
        const streaming_gate_bytes: ?u64 = if (expert_stream_mod.expertStreamingEngaged(
            owned.config.supportsExpertStreaming(),
            owned.config.expertStreamingRequired(),
            self.expert_cache_bytes,
            settings_budget,
        )) blk: {
            if (self.expert_cache_bytes == 0 and settings_budget == 0) return error.ExpertStreamingRequired;
            const mtp = mtpChoiceFor(self.mtp_enabled, self.mtp_explicit, owned.config);
            const verdict = mtpStreamingVerdict(mtp, streamedHeadSupportedAt(self.io, self.allocator, owned.config, entry.path));
            switch (verdict) {
                .keep => owned.config.stream_mtp_head = true,
                .refuse => return error.ExpertStreamingMtpUnsupported,
                .drop_settings => owned.config.mtp_override = false,
                .drop_default, .off => {},
            }
            const plan = try planExpertStreaming(self.io, self.allocator, owned.config, entry.path, self.expert_cache_bytes, settings_budget, coldLoadVision(owned.config, true));
            break :blk streamedLoadRequirementBytes(plan.split.trunk +| plan.split.mtp +| plan.split.vision, plan.cache.cache_bytes, plan.cache.prefill_peak_bytes, plan.cache.bounce_bytes);
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
            errdefer self.registry.mutex.unlock(self.io); // bail on early returns

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
                        self.registry.mutex.unlock(self.io);
                        return load_err;
                    },
                    .unloaded => break :wait_loop,
                }
            }

            // Claim the slot.
            std.debug.assert(self.registry.tryBeginLoadLocked(entry));

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
                self.registry.mutex.unlock(self.io);
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
            // `--no-drafter` / `--drafter` / `--draft-block-size` reach cold
            // loads too; the path itself is scoped by `coldLoadDrafterDir`.
            .drafter_dir = coldLoadDrafterDir(self.no_drafter, self.primary_model_dir, self.drafter_dir, entry.path),
            .no_drafter = self.no_drafter,
            .load_vision = coldLoadVision(owned.config, false),
            .warmup_eager = true,
            .draft_block_size = self.draft_block_size,
            .draft_block_size_explicit = self.draft_block_size_explicit,
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
            .expert_cache_bytes = self.expert_cache_bytes,
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

        // Re-acquire under mutex and refcount the ready entry.
        self.registry.mutex.lockUncancelable(self.io);
        defer self.registry.mutex.unlock(self.io);
        if (entry.state != .ready) return error.LoadFailed;
        _ = entry.refcount.fetchAdd(1, .acq_rel);
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

    /// Synchronously compute embeddings for `req.token_seqs` using the
    /// batched encoder forward pass on the inference thread. Same lifecycle
    /// as `encodeVision`: post + block + return results. Caller frees the
    /// returned rows + outer slice (allocated with `req.allocator`).
    pub fn computeEmbeddings(self: *Scheduler, req: *EmbedRequest) ![][]f32 {
        self.queue_mu.lockUncancelable(self.io);
        self.embed_queue.append(self.allocator, req) catch |err| {
            self.queue_mu.unlock(self.io);
            return err;
        };
        self.queue_cond.broadcast(self.io);
        self.queue_mu.unlock(self.io);

        req.done_mu.lockUncancelable(self.io);
        defer req.done_mu.unlock(self.io);
        while (!req.done) {
            req.done_cond.waitUncancelable(self.io, &req.done_mu);
        }
        if (req.error_name) |_| return error.EmbedFailed;
        return req.results orelse error.EmbedFailed;
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
    /// off at runtime. Reading the armed flags here vetoed batched decode for any
    /// slot whose prompt merely n-gram-scored high enough to ARM PLD, even after
    /// PLD's own yield gate had disabled itself — neither speculation nor batching
    /// (measured 2.4x on concurrent GDN decode). Same class as "a guard that shapes
    /// INIT options does not bind DISPATCH".
    fn slotTicksRegular(slot: *const Slot) bool {
        if (Planner.enabled() and slot.planner_force_plain) return true;
        const gen = if (slot.legacy_gen) |*g| g else return !(slot.enable_pld or slot.enable_drafter or slot.enable_mtp);
        // A runtime-disabled generator is already ticking regular, so batching it
        // dispatches what it was going to dispatch anyway.
        //
        // The POLICY this encodes, which is deliberate and not free: `nextPld`'s
        // periodic re-enable check (`SPEC_REENABLE_INTERVAL`) lives on the serial
        // path, and the batched tick never calls it. So a slot whose PLD yield
        // gate disabled itself stays disabled for as long as it keeps company —
        // even if its tail later turns into the file/tool echo PLD is best at.
        // That is the right trade at N>1 (the batched kernel reads the weight set
        // once for the whole group, which beats one slot's lookup wins) and it
        // self-corrects: a SOLO slot never reaches the batched path, so the
        // re-enable check resumes the moment concurrency drops back to one.
        // Pinned by `a spec_disabled_runtime slot is batchable, and that is the
        // documented trade` below — flip either half deliberately, not by accident.
        return gen.spec_disabled_runtime or specTickMode(
            slot.enable_mtp,
            gen.mtp != null,
            slot.enable_drafter,
            gen.dflash != null,
            slot.enable_pld,
            gen.pld_enabled,
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
    fn batchVerdict(self: *const Scheduler, slot: *const Slot) BatchVerdict {
        _ = self;
        if (!slotTicksRegular(slot)) return .spec_active;
        // A slot whose module-head release is armed but not landed still holds the head.
        if (slotReleasePending(slot)) return .head_release_pending;
        if (slot.sampling.constraint != null) return .grammar;
        // The batched tick samples every slot's successor; a forced call decides its own.
        if (slot.sampling.call_force) |cf| {
            if (cf.pending()) return .forced_call;
        }
        if (slot.logprobs_n > 0) return .logprobs;
        if (generate_mod.penaltyActive(slot.sampling)) return .penalty;
        // Ask the transformer, never name the arch here — same rule as `modelExclusiveDecode`.
        const t = slot.model.transformer orelse return .arch;
        return if (t.supportsBatchedGdnDecode() or t.supportsBatchedMimoDecode() or t.supportsBatchedGlmRows()) .ok else .arch;
    }
};

/// Slots one batched forward can carry; the tail decodes serial this tick.
pub const MAX_BATCH_GROUP = 32;

pub const BatchVerdict = enum {
    ok,
    spec_active,
    head_release_pending,
    grammar,
    forced_call,
    logprobs,
    penalty,
    arch,
    pad_waste,
    row_cap,
    /// The model owns its decode state (`slotExclusiveDecode`): requests queue, never co-decode.
    exclusive,
};

/// Does the loaded model's config batch at all? The arch half of `batchVerdict`,
/// shared with `/props`, `/v1/models` and the serve-mode startup line.
pub fn configBatchesDecode(cfg: *const model_mod.ModelConfig) bool {
    return cfg.supportsBatchedGdnDecode() or cfg.supportsBatchedMimoDecode() or cfg.supportsBatchedGlmRows();
}

/// MiMo and GLM batching are certified (and GLM measured) for up to four independent slots; qwen4 rows
/// share kernels up to the joined hyper-connection and expert width of eight.
pub fn batchGroupCap(cfg: *const model_mod.ModelConfig) usize {
    if (cfg.supportsBatchedQwen4Rows()) return 8;
    return if (cfg.supportsBatchedMimoDecode() or cfg.supportsBatchedGlmRows()) 4 else MAX_BATCH_GROUP;
}

/// One line per slot the first time it decodes serial beside live company;
/// the counter moves every tick so the rate is visible under `--metrics`.
fn noteSerial(sch: *Scheduler, slot: *Slot, why: BatchVerdict) void {
    if (sch.metrics) |m| m.decode_serial_total[@backingInt(why)].inc();
    if (slot.serial_reason_logged) return;
    slot.serial_reason_logged = true;
    log.info("[batched] slot serial: {s} (model={s})\n", .{ @tagName(why), slot.model.id });
}

/// Batched decode pads every slot's KV to the group's LONGEST (`padAndStackBatchedKV`),
/// so the tensor it builds is `N x kv_max`, not `sum(kv_len)`. A group mixing one
/// 100k-token stream with three 1k ones therefore materializes ~100x the bytes the
/// short slots need — on a qwen3_5 trunk (hd 256, `--ctx-size` up to 262144) that is
/// ~410 MB per full-attention layer, ~6.5 GB across the 16 of them, and NOTHING bills
/// it: it is a per-tick transient, invisible to `prefillTransientReserve` and to the
/// load-time gate. An uncatchable Metal OOM is the failure mode.
///
/// So the group is capped by PADDING WASTE, which is the quantity that actually hurts,
/// rather than by a length ratio: keep the largest prefix of the ascending-sorted
/// lengths whose padded tensor stays within `MAX_PAD_WASTE` of its useful bytes. The
/// slots that fall out are the LONGEST ones — the ones dominating kv_max — and they
/// decode serially this tick, so every slot still advances.
///
/// Returns how many of `kv_lens_asc` may batch together (0 or 1 = nobody batches).
/// The lengths are the arch's true attention KV length (`batchKvLenOf`), pre-tick counts,
/// so this is a heuristic bar on the padding the forward will build, not an accounting identity.
/// The padded tensor may be at most this multiple of the bytes the group
/// actually needs. It must stay BELOW 2.0 or a two-slot group can never be
/// vetoed: one 1-token slot beside one 200k slot pads to exactly 2x, which is
/// the worst case for N=2 and the pathological pair this cap exists for.
pub const MAX_PAD_WASTE: f64 = 1.5;

var kv_skew_split_logged: bool = false; // one-shot log guard

pub fn batchedKvKeepCount(kv_lens_asc: []const u32) usize {
    if (kv_lens_asc.len < 2) return 0;
    var k: usize = kv_lens_asc.len;
    while (k >= 2) : (k -= 1) {
        var sum: u64 = 0;
        for (kv_lens_asc[0..k]) |l| sum += l;
        if (sum == 0) return k; // nothing prefilled yet: no padding to waste
        const padded: f64 = @floatFromInt(@as(u64, k) * kv_lens_asc[k - 1]);
        if (padded <= MAX_PAD_WASTE * @as(f64, @floatFromInt(sum))) return k;
    }
    return 0;
}

/// The padding waste the whole group would pay; reported by the cap's log.
pub fn batchedPadWaste(kv_lens_asc: []const u32) f64 {
    var sum: u64 = 0;
    for (kv_lens_asc) |l| sum += l;
    if (sum == 0 or kv_lens_asc.len == 0) return 1.0;
    const padded: f64 = @floatFromInt(@as(u64, kv_lens_asc.len) * kv_lens_asc[kv_lens_asc.len - 1]);
    return padded / @as(f64, @floatFromInt(sum));
}

/// A group whose longest billed slot is at most this long pads a small tensor, and splitting it sends a slot
/// through a whole serial forward instead: under the qwen4 gather arm every long slot bills at the indexer
/// budget, so a short sub-agent beside long streams reads as a 2x pad.
pub const PAD_FREE_KV: u32 = 4096;

/// The floor only engages once a slot holds this much context: below it the serial forward it saves is cheap.
pub const PAD_FREE_MIN_CTX: u32 = 131072;

/// `batchedKvKeepCount` for a group whose longest TRUE context is `ctx_max`.
pub fn groupKeepCount(kv_lens_asc: []const u32, ctx_max: u32) usize {
    if (kv_lens_asc.len >= 2 and ctx_max >= PAD_FREE_MIN_CTX and kv_lens_asc[kv_lens_asc.len - 1] <= PAD_FREE_KV) return kv_lens_asc.len;
    return batchedKvKeepCount(kv_lens_asc);
}

fn groupCtxMax(caches: []const *const KVCache) u32 {
    var m: usize = 0;
    for (caches) |c| m = @max(m, c.kvLenForBatching());
    return @intCast(@min(m, std.math.maxInt(u32)));
}

/// One source for the length the batched group is sorted and capped by. `cache.step`
/// advances only on global layer 0, so on a linear-layer-0 trunk (GDN, gated-conv, Mamba2,
/// KDA) it is 0 forever and the pad-waste cap never fired. `KVCache.kvLenForBatching` reads
/// the first attention layer's own offset there.
pub fn batchKvLenOf(cache: *const KVCache, cfg: ?*const model_mod.ModelConfig) u32 {
    return batchKvLenOfWith(cache, cfg, 1);
}

pub fn batchKvLenOfWith(cache: *const KVCache, cfg: ?*const model_mod.ModelConfig, seq_len: c_int) u32 {
    // Arch gate: the multi-stream batched wins on the 27B were measured with the cap dead,
    // so every other arch keeps `cache.step` (and the dead cap) pending a measurement.
    const c = cfg orelse return @intCast(cache.step);
    if (!c.longCtxGated()) return @intCast(cache.step);
    const raw: u32 = @intCast(cache.kvLenForBatching());
    const gather_on = transformer_mod.qsaBatchedGatherOn(seq_len);
    const min_kv: u32 = @intCast(transformer_mod.qsaBatchedGatherFloor(seq_len, cache.config.scheme == .affine));
    return c.batchedEffectiveKvLen(raw, gather_on, min_kv);
}

pub fn fillGroupPadWasteKvLens(
    caches: []const *const KVCache,
    cfg: ?*const model_mod.ModelConfig,
    seq_len: c_int,
    out: []u32,
) void {
    for (caches, 0..) |c, i| out[i] = batchKvLenOfWith(c, cfg, seq_len);
}

/// A model whose per-request decode state is MODULE-OWNED (one per model,
/// not per slot) — at most ONE in-flight request may touch it. Ask the
/// transformer which archs own their decode state; never key on
/// a batching verdict and never name one arch here.
fn modelExclusiveDecode(model: *const model_registry_mod.LoadedModel) bool {
    const t = model.transformer orelse return false;
    return t.ownsModuleDecodeState();
}

/// Pure core of `slotExclusiveDecode`. `model_owns_state` wins outright and is never
/// released; the head clause is qwen4_exp's per-model head, released for good once the
/// adaptive switch moved the slot to serial (`Generator.mtpModuleHeadReleased`).
pub fn headExclusiveFor(
    model_owns_state: bool,
    head_module_owned: bool,
    slot_enable_mtp: bool,
    head_released: bool,
) bool {
    if (model_owns_state) return true;
    if (!head_module_owned or !slot_enable_mtp) return false;
    return !head_released;
}

/// Per-SLOT exclusivity: the model's own bit, OR a slot that will drive a
/// module-owned MTP head (qwen4: `Qwen4Mtp.cache` is one per model). Plain
/// slots on the same model keep interleaving/batching beside it; two MTP
/// slots serialize, until one of them releases the head.
fn slotExclusiveDecode(slot: *const Slot) bool {
    const head = slot.model.mtp;
    return headExclusiveFor(
        modelExclusiveDecode(slot.model),
        if (head) |h| h.moduleOwned() else false,
        slot.enable_mtp,
        if (slot.legacy_gen) |*g| g.mtpModuleHeadReleased() else false,
    );
}

/// A cancel can land after the tick's cull, and `complete()` moves a slot from `decoding` to
/// `cleanup_queue` without a cull in between: until the inference thread has released it, the slot
/// still owns its native state and keeps blocking exclusive admission.
fn slotHoldsAdmission(slot: *const Slot) bool {
    return !slot.finished and slot.error_code == null;
}

/// Exclusive slots in `lists` (decoding, then queued for cleanup) that still hold admission.
fn heldExclusive(out: []AdmitCand, lists: []const []const *Slot, exclusive: *const fn (*const Slot) bool) usize {
    var n: usize = 0;
    for (lists) |list| for (list) |s| {
        if (n >= out.len) return n;
        if (!slotHoldsAdmission(s) or !exclusive(s)) continue;
        out[n] = .{ .model = @intFromPtr(s.model), .exclusive = true };
        n += 1;
    };
    return n;
}

/// One pending-drain candidate (or live decoding slot), reduced to what
/// admission needs: an opaque model identity + the exclusive-decode bit.
pub const AdmitCand = struct { model: usize, exclusive: bool };

/// FIFO admission for one drain tick. An EXCLUSIVE candidate admits only if
/// no EXCLUSIVE `active` slot (live decoding) holds its model and no earlier
/// admitted exclusive candidate claimed it this tick
/// (a same-tick sibling is not in `decoding` yet — the claim covers the
/// window). Non-exclusive candidates always admit — no head-of-line
/// blocking behind a held exclusive request. Writes admitted candidate
/// indices to `out` (queue order preserved), returns the count. Held
/// candidates stay where they are and retry next tick; no release
/// bookkeeping exists — the busy signal IS presence in `decoding`.
pub fn admitPendingTick(cands: []const AdmitCand, active: []const AdmitCand, out: []usize) usize {
    var n: usize = 0;
    outer: for (cands, 0..) |c, i| {
        if (n >= out.len) break;
        if (c.exclusive) {
            for (active) |a| if (a.exclusive and a.model == c.model) continue :outer;
            for (out[0..n]) |j| {
                if (cands[j].exclusive and cands[j].model == c.model) continue :outer;
            }
        }
        out[n] = i;
        n += 1;
    }
    return n;
}

/// `commit` is the part of `needed` the request allocates after admission (cache growth the bill
/// prices up front); until it is resident, later admissions must leave it free.
pub const MemoryBill = struct { needed: u64, available: u64, commit: u64 = 0 };

/// How many of one tick's admits (queue order) go on to prefill. Each connection-thread bill
/// ran before any sibling allocated, so an ungated arch is re-billed here: an admit that does
/// not fit beside live requests plus this tick's earlier admits stays pending, and so does
/// every admit after it. Alone it proceeds: nobody would ever free memory for it. The gated arch
/// is billed here too, so a Qwen and a MiMo admit see each other, and again inside `runPrefill`.
/// `outstanding` is what live requests were admitted to allocate and have not yet.
pub fn admitsWithinMemory(bills: []const ?MemoryBill, live_company: bool, outstanding: u64) usize {
    var promised: u64 = outstanding;
    for (bills, 0..) |bill, i| {
        const b = bill orelse continue;
        if ((live_company or i > 0) and b.needed +| promised > b.available) return i;
        promised +|= b.needed;
    }
    return bills.len;
}

/// What a live slot was admitted to allocate and has not: its commitment less its resident
/// cache. A DFlash2 slot reserves its capacity after prefill, so its share drains at once.
fn outstandingGrowth(commit: u64, resident: u64) u64 {
    return commit -| resident;
}

fn slotOutstandingGrowth(s: *const Slot) u64 {
    var resident = s.cache.residentBytes();
    if (s.glm5_request) |*request| resident +|= request.residentBytes();
    return outstandingGrowth(s.growth_commit, resident);
}

/// `admitsWithinMemory` over this tick's admits; called under `queue_mu`.
fn memoryAdmitCount(sch: *Scheduler, admit_idx: []const usize) usize {
    const numbers_fn = prefill_admission_numbers orelse return admit_idx.len;
    var live = false;
    var outstanding: u64 = 0;
    for (sch.decoding.items) |s| {
        if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
        live = true;
        outstanding +|= slotOutstandingGrowth(s);
    }
    var bills: [16]?MemoryBill = undefined;
    const n = @min(admit_idx.len, bills.len);
    for (admit_idx[0..n], 0..) |idx, i| {
        bills[i] = null;
        const s = sch.pending.items[idx];
        const cfg = s.model.config orelse continue;
        if (s.model.transformer == null) continue;
        bills[i] = numbers_fn(cfg, s.full_prompt.len, s.max_tokens, s.cache.config, generate_mod.visionPrefillUnchunked(s.vision_embeddings != null), s.enable_mtp);
    }
    const admitted = admitsWithinMemory(bills[0..n], live, outstanding);
    for (admit_idx[0..admitted], bills[0..admitted]) |idx, bill| sch.pending.items[idx].growth_commit = if (bill) |b| b.commit else 0;
    if (admitted < n) {
        const held = sch.pending.items[admit_idx[admitted]];
        if (!held.memory_hold_logged) {
            held.memory_hold_logged = true;
            log.info("[admission] held: {d} tokens need ~{d}MB, ~{d}MB available beside live requests (~{d}MB of their growth outstanding); waiting for one to finish\n", .{
                held.full_prompt.len, bills[admitted].?.needed >> 20, bills[admitted].?.available >> 20, outstanding >> 20,
            });
        }
    }
    return admitted;
}

test "admitsWithinMemory: siblings are billed together, a lone request always proceeds" {
    const gb: u64 = 1 << 30;
    const b: ?MemoryBill = .{ .needed = 8 * gb, .available = 20 * gb };
    // Four arrivals each fit alone against the same free memory; only two fit together.
    try testing.expectEqual(@as(usize, 2), admitsWithinMemory(&.{ b, b, b, b }, false, 0));
    // Beside a live request the first must fit by itself.
    const big: ?MemoryBill = .{ .needed = 30 * gb, .available = 20 * gb };
    try testing.expectEqual(@as(usize, 0), admitsWithinMemory(&.{ big, b }, true, 0));
    // Alone it proceeds whatever the bill: nothing would ever free memory for it.
    try testing.expectEqual(@as(usize, 1), admitsWithinMemory(&.{big}, false, 0));
    // A gated admit carries no bill here and never blocks the queue.
    try testing.expectEqual(@as(usize, 3), admitsWithinMemory(&.{ null, b, b }, false, 0));
}

test "admissionPassArmed: the evict-to-admit pass runs for every served arch, never an unserved one" {
    const mimo = ModelConfig{ .model_type = "mimo_v2", .num_hidden_layers = 1, .has_sliding_window = true, .sliding_window = 128, .head_dim = 192 };
    const qwen4 = ModelConfig{ .model_type = "qwen4_exp" };
    const glm = ModelConfig{ .model_type = "glm5_next" };
    const llama = ModelConfig{ .model_type = "llama", .has_sliding_window = true, .sliding_window = 128, .head_dim = 192 };
    try testing.expect(admissionPassArmed(&mimo));
    try testing.expect(admissionPassArmed(&qwen4));
    try testing.expect(admissionPassArmed(&glm));
    try testing.expect(!admissionPassArmed(&llama));
    try testing.expect(!admissionPassArmed(null));
}

const ThreadCtx = struct {
    scheduler: *Scheduler,
    params: LoadParams,
};

/// Signal to the parent waiting in `init()` that the inference thread is done
/// with its load (or has failed). After `started` flips, the parent reads
/// `load_failed` to decide whether to surface a startup error.
fn signalStarted(sch: *Scheduler) void {
    sch.started_mu.lockUncancelable(sch.io);
    defer sch.started_mu.unlock(sch.io);
    sch.started.store(true, .release);
    sch.started_cond.broadcast(sch.io);
}
/// Set the load_error_name + flip load_failed. Best-effort dupe; on OOM the
/// parent still sees `load_failed=true` and surfaces "unknown".
fn recordLoadError(sch: *Scheduler, err_name: []const u8) void {
    if (sch.load_error_name) |old| sch.allocator.free(old);
    sch.load_error_name = sch.allocator.dupe(u8, err_name) catch null;
    sch.load_failed.store(true, .release);
}

/// Both load construction sites (here and main.zig's startup load) stamp the
/// per-model settings onto the config the bills and defaults read.
pub fn applyModelSettings(config: *ModelConfig, o: model_settings.Override) void {
    config.ctx_override = o.ctx_size orelse 0;
    config.kv_quant_override = o.kv_quant;
    config.mtp_override = o.mtp;
    config.mtp_acceptance_override = o.mtp_acceptance;
    config.mtp_greedy_tail_override = o.mtp_greedy_tail;
    config.ssd_budget_gb_override = o.ssd_budget_gb orelse 0;
    config.preserve_thinking_override = o.preserve_thinking;
    config.think_penalty_override = o.think_penalty;
    config.logit_bias_file_override = o.logit_bias_file;
    config.vision_override = o.vision;
    if (resolveSsdBudget(0, config.ssd_budget_gb_override, config.supportsExpertStreaming()).setting_ignored)
        log.warn("[model-settings] ssd_budget_gb ignored: this checkpoint does not stream experts from SSD\n", .{});
}

/// The verify-rows arms a forward microbench width runs.
fn ubenchRowArms(rows: usize, is_mimo: bool, both: bool) []const bool {
    const eligible = is_mimo and rows > 1 and rows <= transformer_mod.MIMO_VERIFY_ROWS_MAX;
    if (!eligible) return &.{false};
    return if (both) &.{ false, true } else &.{true};
}

/// One forward-microbench block: its verify-rows arm and its QSA pooled-key arm.
const UbenchArm = struct {
    verify_rows: bool,
    /// null keeps the shipped default; false runs the composed chain, true the fused kernel.
    qsa_pool: ?bool = null,

    /// Arms the block and zeroes the fused-launch count it reports.
    fn apply(self: UbenchArm, ctx: *ForwardCtx) void {
        ctx.verify_rows = self.verify_rows;
        qwen4_qsa.qsa_pool_rope_fused_override = self.qsa_pool;
        transformer_mod.qsa_pool_rope_dispatches = 0;
    }

    fn poolName(self: UbenchArm) []const u8 {
        const fused = self.qsa_pool orelse return "default";
        return if (fused) "fused" else "composed";
    }
};

/// A width's blocks: each verify-rows arm, run A B B A over the QSA pooled-key chain (A) and kernel (B)
/// when `pool_arms` asks.
fn ubenchArms(buf: *[8]UbenchArm, rows: usize, is_mimo: bool, row_arms: bool, pool_arms: bool) []const UbenchArm {
    const pools: []const ?bool = if (pool_arms) &.{ false, true, true, false } else &.{null};
    var n: usize = 0;
    for (ubenchRowArms(rows, is_mimo, row_arms)) |verify_rows| for (pools) |pool| {
        buf[n] = .{ .verify_rows = verify_rows, .qsa_pool = pool };
        n += 1;
    };
    return buf[0..n];
}

/// A resident MiMo warms its verify rows with or without its heads: a PLD request verifies on them too.
fn mimoVerifyWarmWanted(config: *const ModelConfig) bool {
    return config.isMimo() and !config.expert_streaming;
}

test "a resident MiMo warms its verify rows without its heads; a streamed one and another arch do not" {
    var mimo = ModelConfig{};
    mimo.model_type = "mimo_v2";
    try testing.expect(mimoVerifyWarmWanted(&mimo));
    mimo.expert_streaming = true;
    try testing.expect(!mimoVerifyWarmWanted(&mimo));
    var qwen = ModelConfig{};
    qwen.model_type = "qwen4_exp";
    try testing.expect(!mimoVerifyWarmWanted(&qwen));
}

fn mimoSpecWarmup(io: std.Io, xfm: *Transformer, head: ?*mimo_mtp.Head, kv_config: transformer_mod.KVQuantConfig) void {
    const start = std.Io.Timestamp.now(io, .awake);
    const rows = xfm.warmupMimoVerify(kv_config) catch |err| {
        log.warn("[spec-warmup] MiMo verify failed ({s}); the first round at each width pays its kernel compile inside the round.\n", .{@errorName(err)});
        return;
    };
    const steps = if (head) |h| h.warmup(xfm) catch |err| {
        log.warn("[spec-warmup] MiMo heads failed ({s}); the first round pays their kernel compile inside the round.\n", .{@errorName(err)});
        return;
    } else 0;
    const ms: u64 = @as(u64, @intCast(start.untilNow(io, .awake).nanoseconds)) / std.time.ns_per_ms;
    log.info("[spec-warmup] MiMo verify rows 0x{x}, {d} head steps ({d} ms).\n", .{ rows, steps, ms });
}

/// MiMo's MTP heads from the checkpoint's own shard, bound to the trunk they share.
fn loadMimoHeads(sch: *Scheduler, model_dir: []const u8, config: *const ModelConfig, xfm: *Transformer) !?*mimo_mtp.Head {
    var weights = try @import("mimo_source.zig").loadMtpWeights(sch.io, sch.allocator, model_dir);
    defer weights.deinit();
    var head = (try mimo_mtp.Head.load(sch.allocator, mlx.gpuStream(), config, &weights)) orelse return null;
    errdefer head.deinit();
    head.target = xfm;
    const ptr = try sch.allocator.create(mimo_mtp.Head);
    ptr.* = head;
    // The coarse lm_head copy is a load cost, never the first draft's.
    const rerank = ptr.canRerankDrafts();
    log.info("[mimo-mtp] {d} heads loaded ({d:.2} GB resident); draft rerank {s}\n", .{ head.heads, @as(f64, @floatFromInt(head.residentBytes())) / 1e9, if (rerank) "on" else "off" });
    return ptr;
}

/// A load's MTP decision: `--mtp`/`--no-mtp` > `--fast` > the per-model `mtp` > on.
pub fn mtpChoiceFor(mtp_enabled: bool, mtp_explicit: bool, config: *const ModelConfig) model_settings.MtpChoice {
    const choice = model_settings.MtpChoice.resolve(model_settings.launchFlag(bool, mtp_enabled, mtp_explicit), config.mtp_override, true);
    return if (config.expert_streaming) choice.streamed() else choice;
}

/// The streaming gate's verdict on a load's MTP choice (`expert_stream.mtpUnderStreaming`). The gate
/// runs before the load marks its config streamed.
fn mtpStreamingVerdict(choice: model_settings.MtpChoice, head_supported: bool) expert_stream_mod.MtpUnderStreaming {
    const c = choice.streamed();
    return expert_stream_mod.mtpUnderStreaming(c.on, c.source == .model_settings, c.source == .default, head_supported);
}

fn streamedHeadSupportedAt(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8) bool {
    const geometry = config.expertGeometry();
    const layout = expert_stream_mod.quant.streamingLayoutOfDir(allocator, io, config.model_type, model_dir, geometry.layers, geometry.first_moe_layer) catch return false;
    return expert_stream_mod.streamedMtpHeadSupported(config.isQwen4(), layout);
}

/// An engine-default MTP under expert streaming resolves off (`expert_stream.mtpUnderStreaming`):
/// the head is not loaded and the load log reads `off (streaming; default)`.
pub fn mtpDefaultOffUnderStreaming(choice: model_settings.MtpChoice, expert_streaming: bool, head_kept: bool) bool {
    return expert_streaming and !head_kept and choice.on and choice.source == .default;
}

test "a streamed load drops only the engine-default MTP, never an asked-for one" {
    try testing.expect(mtpDefaultOffUnderStreaming(.{ .on = true, .source = .default }, true, false));
    try testing.expect(!mtpDefaultOffUnderStreaming(.{ .on = true, .source = .default }, true, true));
    try testing.expect(!mtpDefaultOffUnderStreaming(.{ .on = true, .source = .default }, false, false));
    try testing.expect(!mtpDefaultOffUnderStreaming(.{ .on = true, .source = .flag }, true, false));
    try testing.expect(!mtpDefaultOffUnderStreaming(.{ .on = false, .source = .default }, true, false));
}

/// What a load line and `/props` read for the four keys `--fast` sets, under the launch globals.
const LaunchPicks = struct {
    mtp: model_settings.MtpChoice,
    acceptance: model_settings.Pick(mtp_acceptance_mod.Mode),
    greedy_tail: model_settings.Pick(bool),
    kv: transformer_mod.KvCacheChoice,

    fn of(mtp_flag: ?bool, kv_flag: ?transformer_mod.KVQuantConfig, config: *const ModelConfig) LaunchPicks {
        return .{
            .mtp = mtpChoiceFor(mtp_flag orelse true, mtp_flag != null, config),
            .acceptance = generate_mod.mtpAcceptanceFor(config.mtp_acceptance_override),
            .greedy_tail = generate_mod.mtpGreedyTailFor(config.mtp_greedy_tail_override),
            .kv = transformer_mod.KvCacheChoice.resolve(config.kv_quant_override, kv_flag orelse transformer_mod.KVQuantConfig.engine_default, kv_flag != null),
        };
    }
};

const LaunchGlobals = struct {
    fast: bool,
    acceptance: mtp_acceptance_mod.Mode,
    acceptance_explicit: bool,
    greedy_tail_explicit: bool,

    fn save() LaunchGlobals {
        return .{
            .fast = model_settings.fast,
            .acceptance = generate_mod.mtp_acceptance_default,
            .acceptance_explicit = generate_mod.mtp_acceptance_explicit,
            .greedy_tail_explicit = generate_mod.mtp_greedy_tail_explicit,
        };
    }

    fn restore(self: LaunchGlobals) void {
        model_settings.fast = self.fast;
        generate_mod.mtp_acceptance_default = self.acceptance;
        generate_mod.mtp_acceptance_explicit = self.acceptance_explicit;
        generate_mod.mtp_greedy_tail_explicit = self.greedy_tail_explicit;
    }
};

test "--fast: its preset, named --fast, outranks model-settings.json; an explicit flag outranks --fast; no --fast, no change" {
    const saved = LaunchGlobals.save();
    defer saved.restore();
    (LaunchGlobals{ .fast = false, .acceptance = .exact, .acceptance_explicit = false, .greedy_tail_explicit = false }).restore();
    const p = model_settings.fast_preset;
    var file = ModelConfig{};
    file.mtp_override = !p.mtp.?;
    file.mtp_acceptance_override = .{ .tokenv3 = 0.95 };
    file.mtp_greedy_tail_override = !p.mtp_greedy_tail.?;
    file.kv_quant_override = .dense;

    const unfast = LaunchPicks.of(null, null, &file);
    try testing.expectEqual(!p.mtp.?, unfast.mtp.on);
    try testing.expectEqual(model_settings.Source.model_settings, unfast.mtp.source);
    try testing.expectEqual(model_settings.Source.model_settings, unfast.acceptance.source);
    try testing.expectEqual(model_settings.Pick(bool){ .value = !p.mtp_greedy_tail.?, .source = .model_settings }, unfast.greedy_tail);
    try testing.expectEqualStrings("model-settings.json", unfast.kv.sourceName());

    model_settings.fast = true;
    for ([_]ModelConfig{ .{}, file }) |cfg| {
        const got = LaunchPicks.of(null, null, &cfg);
        try testing.expectEqual(p.mtp.?, got.mtp.on);
        try testing.expectEqualStrings("--fast", got.mtp.sourceName());
        try testing.expect(std.meta.eql(p.mtp_acceptance.?, got.acceptance.value));
        try testing.expectEqualStrings("--fast", model_settings.sourceLabel(got.acceptance.source, model_settings.acceptanceFlagName(got.acceptance.value)));
        try testing.expectEqual(model_settings.Pick(bool){ .value = p.mtp_greedy_tail.?, .source = .fast }, got.greedy_tail);
        try testing.expectEqual(p.kv_quant.?, got.kv.config);
        try testing.expectEqualStrings("--fast", got.kv.sourceName());
    }

    generate_mod.mtp_acceptance_default = .{ .typical = .{ .delta = 0.1 } };
    generate_mod.mtp_acceptance_explicit = true;
    generate_mod.mtp_greedy_tail_explicit = true;
    const flagged = LaunchPicks.of(false, transformer_mod.KVQuantConfig.affine(4), &file);
    try testing.expect(!flagged.mtp.on);
    try testing.expectEqualStrings("--no-mtp", flagged.mtp.sourceName());
    try testing.expectEqual(@as(f32, 0.1), flagged.acceptance.value.typical.delta);
    try testing.expectEqual(model_settings.Source.flag, flagged.acceptance.source);
    try testing.expectEqual(model_settings.Source.flag, flagged.greedy_tail.source);
    try testing.expectEqualStrings("kv4", flagged.kv.label());
    try testing.expectEqualStrings("--kv-quant", flagged.kv.sourceName());
}

test "--fast drops its MTP on a streamed load, before and after the load marks it streamed; an explicit --mtp keeps the head" {
    const saved = LaunchGlobals.save();
    defer saved.restore();
    model_settings.fast = true;
    const gate = mtpChoiceFor(true, false, &ModelConfig{});
    try testing.expect(gate.on);
    try testing.expectEqual(expert_stream_mod.MtpUnderStreaming.off, mtpStreamingVerdict(gate, true));
    var streamed = ModelConfig{};
    streamed.expert_streaming = true;
    const loaded = mtpChoiceFor(true, false, &streamed);
    try testing.expect(!loaded.on);
    try testing.expectEqualStrings("--fast", loaded.sourceName());
    try testing.expect(!loaded.forced());
    const asked = mtpChoiceFor(true, true, &ModelConfig{});
    try testing.expectEqual(expert_stream_mod.MtpUnderStreaming.keep, mtpStreamingVerdict(asked, true));
    try testing.expectEqual(expert_stream_mod.MtpUnderStreaming.refuse, mtpStreamingVerdict(asked, false));
    try testing.expect(mtpChoiceFor(true, true, &streamed).on);
}

pub const SsdBudgetChoice = struct {
    bytes: u64,
    from_setting: bool,
    setting_ignored: bool,
};

/// The launch flag wins over the per-model setting; a non-streaming model ignores the
/// setting (warned once). `--expert-cache-gb` is decided earlier, in `resolveExpertCache`.
pub fn resolveSsdBudget(flag_bytes: u64, setting_gb: u32, streaming: bool) SsdBudgetChoice {
    if (!streaming) return .{ .bytes = 0, .from_setting = false, .setting_ignored = setting_gb > 0 };
    if (flag_bytes > 0) return .{ .bytes = flag_bytes, .from_setting = false, .setting_ignored = false };
    return .{ .bytes = @as(u64, setting_gb) << 30, .from_setting = setting_gb > 0, .setting_ignored = false };
}

/// Plan 05 Phase D: pre-loaded CPU state bundle. Built by the conn thread
/// (CPU only — file I/O + parse, no mlx) ahead of posting a LoadRequest.
/// Ownership transfers to the entry on successful load; on failure the
/// conn thread frees via `freeCpuState`.
pub const CpuState = struct {
    config: *ModelConfig,
    tok: *Tokenizer,
    chat_config: *ChatConfig,
};

/// Phase D: parse config.json, tokenizer, and chat config from `model_dir`
/// into heap pointers ready to hand off to `LoadRequest`. Mirrors the
/// pre-load that main.zig does for the startup model in serve mode.
///
/// Errdefer pattern: for each `try ... else error`, the `deinit` errdefer
/// is registered AFTER the successful init so a downstream failure doesn't
/// call deinit on uninitialized memory. `ModelConfig.deinit` frees the config's
/// one owned field; `allocator.destroy` alone would leak it.
fn preloadCpuState(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !CpuState {
    // An unsupported file format is refused by name before any config.json
    // read (such a dir has none).
    if (model_discovery.isGgufModelPath(io, model_dir)) return error.ModelFormatUnsupported;

    const config = try allocator.create(ModelConfig);
    errdefer allocator.destroy(config);
    config.* = try model_mod.parseConfig(io, allocator, model_dir);

    const tok = try allocator.create(Tokenizer);
    errdefer allocator.destroy(tok);
    tok.* = try tokenizer_mod.loadTokenizer(io, allocator, model_dir);
    errdefer tok.deinit();

    const cc = try allocator.create(ChatConfig);
    errdefer allocator.destroy(cc);
    cc.* = try chat_mod.loadChatConfig(io, allocator, model_dir);
    errdefer cc.deinit();

    // Same EOS-resolution as main.zig — merge the tokenizer's chat-terminator
    // EOS into the stop set ALWAYS, even when config.json already specified an
    // eos_token_id. Some checkpoints (e.g. Qwen2.5-Coder-7B) set config.json
    // eos_token_id to <|endoftext|> but end chat turns with <|im_end|>; gating
    // on `num_eos_tokens == 0` left <|im_end|> out of the stop set and it leaked
    // into output. Additive + dedup-guarded: only ever ADDS a declared stop.
    if (cc.eos_token) |eos_str| {
        if (tok.special_tokens.get(eos_str)) |eos_id| {
            if (!config.isEosToken(eos_id)) config.addEosToken(eos_id);
        }
    }
    if (tok.special_tokens.get("<|endoftext|>")) |eot_id| {
        if (!config.isEosToken(eot_id)) config.addEosToken(eot_id);
    }
    if (tok.special_tokens.get("<pad>")) |pad_id| {
        if (pad_id > 0 and !config.isEosToken(pad_id)) {
            config.addEosToken(pad_id);
        }
    }

    return .{ .config = config, .tok = tok, .chat_config = cc };
}

pub const StreamingPlan = struct {
    layout: expert_stream_mod.quant.Layout,
    /// `vision` is the tower only when the load keeps it.
    split: model_mod.ResidentSplit,
    resolved: ExpertCacheResolution,
    cache: expert_stream_mod.CachePlan,
    vision: VisionCost,
};

/// What the pack's tower costs a streamed load: its weights in the budget, its largest image
/// encode beside it. Slot counts come from the ssd-budget ledger: `slots_without` 0 = no
/// ledger (`--expert-cache-gb`), `slots_with` null = the budget cannot hold the tower.
pub const VisionCost = struct {
    on: bool = false,
    tower_bytes: u64 = 0,
    encode_bytes: u64 = 0,
    slots_without: u16 = 0,
    slots_with: ?u16 = null,
};

/// The ONE streamed-load plan, read by the serving load, its registry gate and `kld`: the
/// on-disk layout, what stays resident, and the expert cache the budget leaves.
pub fn planExpertStreaming(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8, explicit_cache_bytes: u64, budget_bytes: u64, vision: bool) !StreamingPlan {
    const geometry = config.expertGeometry();
    var streamed = config.*;
    streamed.expert_layout = try expert_stream_mod.quant.streamingLayoutOfDir(allocator, io, config.model_type, model_dir, geometry.layers, geometry.first_moe_layer);
    if (config.stream_mtp_head and !expert_stream_mod.streamedMtpHeadSupported(config.isQwen4(), streamed.expert_layout)) {
        log.err("[expert-stream] {s}; drop --mtp\n", .{expert_stream_mod.MTP_UNSUPPORTED});
        return error.ExpertStreamingMtpUnsupported;
    }
    var split = try model_mod.streamingResidentSplit(io, allocator, model_dir, &streamed);
    split.trunk +|= mimoCoarseHeadBytes(&streamed) +| fp8ExpertScratchBytes(streamed.expert_layout, geometry);
    if (!config.has_vision) split.vision = 0;
    if (!config.stream_mtp_head) split.mtp = 0;
    const per_expert = try expert_stream_mod.expertBytesFor(allocator, model_dir, geometry, streamed.expert_layout);
    const picked = resolveStreamedVision(explicit_cache_bytes, budget_bytes, config, split, per_expert, vision) catch |err| {
        if (err == error.SsdBudgetBelowVision) log.err("[vision] --ssd-budget-gb {d} cannot hold the vision tower ({d:.2} GB) beside two expert slots per layer; raise the budget or drop --vision\n", .{
            budget_bytes >> 30,
            @as(f64, @floatFromInt(split.vision)) / 1e9,
        });
        return err;
    };
    if (!picked.cost.on) split.vision = 0;
    const cache = try expert_stream_mod.cachePlanBytesForGeometry(picked.resolved.cache_bytes, geometry, per_expert);
    return .{ .layout = streamed.expert_layout, .split = split, .resolved = picked.resolved, .cache = cache, .vision = picked.cost };
}

/// PURE: the expert cache with the pack's tower (`split.vision`) kept or left off, and what
/// keeping it costs. Asked for on a budget that cannot hold it, it is refused by name.
pub fn resolveStreamedVision(explicit_bytes: u64, budget_bytes: u64, config: *const ModelConfig, split: model_mod.ResidentSplit, per_expert: u64, want: bool) !struct { resolved: ExpertCacheResolution, cost: VisionCost } {
    var text = split;
    text.vision = 0;
    const without = try resolveExpertCache(explicit_bytes, budget_bytes, config, text, config.stream_mtp_head, per_expert);
    if (split.vision == 0) return .{ .resolved = without, .cost = .{} };
    const with: ?ExpertCacheResolution = resolveExpertCache(explicit_bytes, budget_bytes, config, split, config.stream_mtp_head, per_expert) catch |err| switch (err) {
        error.SsdBudgetBelowResident => null,
        else => return err,
    };
    if (want and with == null) return error.SsdBudgetBelowVision;
    const cost = VisionCost{
        .on = want,
        .tower_bytes = split.vision,
        .encode_bytes = @import("server.zig").largestImageEncodeBytes(config),
        .slots_without = if (without.ledger) |l| l.slots_per_layer else 0,
        .slots_with = if (with) |w| (if (w.ledger) |l| l.slots_per_layer else null) else null,
    };
    return .{ .resolved = if (want) with.? else without, .cost = cost };
}

/// One `[vision]` line per load: its choice and source, and on a streamed load what the tower costs.
fn visionLoadLine(buf: []u8, choice: model_settings.VisionChoice, streamed: bool, cost: VisionCost) []const u8 {
    const tower = @as(f64, @floatFromInt(cost.tower_bytes)) / 1e9;
    const encode = @as(f64, @floatFromInt(cost.encode_bytes)) / 1e9;
    if (!streamed or (!choice.on and choice.source != .default))
        return std.fmt.bufPrint(buf, "[vision] {s} ({s})", .{ if (choice.on) "on" else "off", choice.sourceName() }) catch "[vision]";
    if (cost.tower_bytes == 0) return "[vision] off: this checkpoint ships no vision tower weights";
    const lost = cost.slots_without -| (cost.slots_with orelse 0);
    return (if (cost.on and cost.slots_without == 0)
        std.fmt.bufPrint(buf, "[vision] on ({s}): tower {d:.2} GB resident beside the expert cache; an image encode up to {d:.2} GB is billed per request", .{ choice.sourceName(), tower, encode })
    else if (cost.on)
        std.fmt.bufPrint(buf, "[vision] on ({s}): tower {d:.2} GB in the ssd budget (-{d} slots/layer); an image encode up to {d:.2} GB is billed per request", .{ choice.sourceName(), tower, lost, encode })
    else if (cost.slots_without == 0)
        std.fmt.bufPrint(buf, "[vision] off under expert streaming; pass --vision to load the tower (+{d:.2} GB resident)", .{tower})
    else if (cost.slots_with == null)
        std.fmt.bufPrint(buf, "[vision] off under expert streaming; pass --vision to load the tower (+{d:.2} GB: more than this ssd budget holds)", .{tower})
    else
        std.fmt.bufPrint(buf, "[vision] off under expert streaming; pass --vision to load the tower (+{d:.2} GB, -{d} slots/layer)", .{ tower, lost })) catch "[vision]";
}

/// FP8 experts compute from bf16 copies of a layer's routed slots (`glm5_stream.fp8Routed`): at most
/// every expert, plus the gathered codes when the routed slots of a cache slab are not contiguous.
pub fn fp8ExpertScratchBytes(layout: expert_stream_mod.quant.Layout, geometry: expert_stream_mod.Geometry) u64 {
    if (layout != .fp8_individual) return 0;
    const codes = 3 *| @as(u64, geometry.hidden) *| geometry.intermediate;
    return @as(u64, geometry.experts) *| codes *| (2 + 1);
}

/// Marks the config streamed: the fields every streamed forward and bill reads.
pub fn applyStreamingPlan(config: *ModelConfig, plan: StreamingPlan, budget_bytes: u64) void {
    config.expert_layout = plan.layout;
    config.expert_streaming = true;
    config.expert_cache_bytes = plan.cache.cache_bytes;
    config.expert_ssd_budget_bytes = if (plan.resolved.ledger != null) budget_bytes else 0;
    config.expert_workspace_bytes = plan.cache.workspace_bytes;
    config.expert_bounce_bytes = plan.cache.bounce_bytes;
    config.expert_fill_peak_bytes = plan.cache.prefill_peak_bytes;
    config.expert_vision_tower_bytes = plan.split.vision;
    config.expert_vision_encode_bytes = if (plan.vision.on) plan.vision.encode_bytes else 0;
}

pub const ExpertCacheResolution = struct {
    cache_bytes: u64,
    ledger: ?expert_stream_mod.BudgetLedger,
    overridden: bool,
};

/// PURE: the expert-cache byte budget for this load. `--expert-cache-gb` wins outright;
/// otherwise `--ssd-budget-gb` is a TOTAL resident target and the cache is what is left of it
/// after the resident trunk, the MTP head (only when it stays resident), the vision tower the
/// split carries, the whole-layer prefill union, the selected-expert slab and the fill bounce buffers.
pub fn resolveExpertCache(
    explicit_bytes: u64,
    budget_bytes: u64,
    config: *const model_mod.ModelConfig,
    split: model_mod.ResidentSplit,
    mtp_resident: bool,
    per_expert: u64,
) !ExpertCacheResolution {
    const overridden = expert_stream_mod.budgetOverriddenByExplicitCache(explicit_bytes, budget_bytes);
    if (explicit_bytes > 0 or budget_bytes == 0)
        return .{ .cache_bytes = explicit_bytes, .ledger = null, .overridden = overridden };
    const ledger = try expert_stream_mod.budgetLedger(
        budget_bytes,
        split.trunk,
        if (mtp_resident) split.mtp else 0,
        split.vision,
        @intCast(config.expertLayerCount()),
        @intCast(config.num_experts),
        @intCast(config.num_experts_per_tok),
        per_expert,
        expert_stream_mod.BOUNCE_BYTES,
    );
    return .{ .cache_bytes = ledger.cache_bytes, .ledger = ledger, .overridden = false };
}

/// The 2-bit lm_head copy a MiMo source trunk keeps, resident or streamed: the heads draft on it
/// and a greedy readout shortlists on it.
fn mimoCoarseHeadBytes(config: *const ModelConfig) u64 {
    if (!config.usesMimoSourceTrunk() or mimo_mtp.rerankBits() == 0) return 0;
    return mtp_mod.rerankCoarseBytes(@intCast(config.vocab_size), @intCast(config.hidden_size), mimo_mtp.rerankBits());
}

test "a MiMo source trunk bills one coarse lm_head copy, streamed or resident; another arch none" {
    var mimo = ModelConfig{ .model_type = "mimo_v2", .vocab_size = 152576, .hidden_size = 4096 };
    const want = mtp_mod.rerankCoarseBytes(152576, 4096, mimo_mtp.rerankBits());
    for ([_]@import("expert_quant.zig").Layout{ .exl3_k4, .mxfp4_individual }) |layout| {
        mimo.expert_layout = layout;
        mimo.expert_streaming = false;
        try std.testing.expectEqual(want, mimoCoarseHeadBytes(&mimo));
        mimo.expert_streaming = true;
        try std.testing.expectEqual(want, mimoCoarseHeadBytes(&mimo));
    }
    const qwen = ModelConfig{ .model_type = "qwen4_exp", .vocab_size = 248320, .hidden_size = 2560, .expert_layout = .exl3_k4 };
    try std.testing.expectEqual(@as(u64, 0), mimoCoarseHeadBytes(&qwen));
}

test "mimo_v2 expert cache budget excludes the dense prefix" {
    const per_expert: u64 = 1024 * 1024;
    const config = model_mod.ModelConfig{
        .model_type = "mimo_v2",
        .num_hidden_layers = 4,
        .first_k_dense_replace = 1,
        .num_experts = 8,
        .num_experts_per_tok = 2,
    };
    const fixed = per_expert + 8 * per_expert + 2 * per_expert + expert_stream_mod.BOUNCE_BYTES;
    const result = try resolveExpertCache(0, fixed + 9 * per_expert, &config, .{ .trunk = per_expert, .mtp = 0 }, false, per_expert);
    try std.testing.expectEqual(@as(u16, 3), result.ledger.?.slots_per_layer);
    try std.testing.expectEqual(9 * per_expert, result.cache_bytes);
}

/// Frees the three CPU-state pointers.
pub fn freeCpuState(allocator: std.mem.Allocator, s: *CpuState) void {
    s.config.deinit(allocator);
    allocator.destroy(s.config);
    s.tok.deinit();
    allocator.destroy(s.tok);
    s.chat_config.deinit();
    allocator.destroy(s.chat_config);
}

/// The scheduler's borrowed-view seed for a headless boot (`no_initial_load`):
/// never installed on an entry and never loaded.
pub fn headlessStubCpuState(allocator: std.mem.Allocator) !CpuState {
    return stubCpuState(allocator, .{
        .model_type = "headless",
        .weight_prefix = "model",
        .num_hidden_layers = 1,
        .hidden_size = 1,
        .head_dim = 1,
        .num_attention_heads = 1,
        .num_key_value_heads = 1,
        .max_position_embeddings = 4096,
        .is_encoder_only = false,
    });
}

/// Heap CPU state around `config` with an empty byte-level tokenizer and an
/// empty chat template: a caller that never tokenizes only needs the shapes.
fn stubCpuState(allocator: std.mem.Allocator, config_value: ModelConfig) !CpuState {
    const config = try allocator.create(ModelConfig);
    errdefer allocator.destroy(config);
    config.* = config_value;

    const tok = try allocator.create(Tokenizer);
    errdefer allocator.destroy(tok);
    var byte_map: [256]u21 = undefined;
    for (0..256) |b| byte_map[b] = @intCast(b);
    tok.* = .{
        .vocab = std.StringHashMap(u32).init(allocator),
        .id_to_token = std.AutoHashMap(u32, []const u8).init(allocator),
        .merge_ranks = @TypeOf(tok.merge_ranks).init(allocator),
        .allocator = allocator,
        .special_tokens = std.StringHashMap(u32).init(allocator),
        .tok_type = .byte_level_bpe,
        .byte_to_unicode = byte_map,
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = null,
        .eos_id = null,
        .parsed_json = null,
    };
    errdefer tok.deinit();

    const cc = try allocator.create(ChatConfig);
    errdefer allocator.destroy(cc);
    cc.* = .{
        .chat_template = try allocator.dupe(u8, ""),
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    return .{ .config = config, .tok = tok, .chat_config = cc };
}

/// The post-load residency bill the eviction gate reserves, in bytes: the
/// weights plus 10% headroom for KV / vision / drafter overhead.
pub fn gateEstimateBytes(bytes_on_disk: ?u64, num_hidden_layers: u32, hidden_size: u32) u64 {
    const base: u64 = if (bytes_on_disk) |b|
        b
    else
        @as(u64, num_hidden_layers) * @as(u64, hidden_size) * 4 * 4;
    return base + base / 10;
}

/// A boot `--model` entry carries no discovery `bytes_on_disk`, so the measured shard sum is
/// the answer before the layers x hidden guess (which read 1966080 for a ~64 GiB pack).
pub fn residentWeightBytes(billed: ?u64, bytes_on_disk: ?u64, measured_disk: u64, num_hidden_layers: u32, hidden_size: u32) u64 {
    if (billed) |b| return b;
    if (bytes_on_disk) |b| return b;
    if (measured_disk > 0) return measured_disk;
    return @as(u64, num_hidden_layers) * @as(u64, hidden_size) * 4 * 4;
}

test "residentWeightBytes: a boot entry without a disk hint reports its measured shards" {
    const gib: u64 = 1 << 30;
    try testing.expectEqual(64 * gib, residentWeightBytes(null, null, 64 * gib, 48, 2560));
    try testing.expectEqual(@as(u64, 97) * gib, residentWeightBytes(97 * gib, 64 * gib, 64 * gib, 48, 2560));
    try testing.expectEqual(@as(u64, 5), residentWeightBytes(null, 5, 64 * gib, 48, 2560));
    try testing.expectEqual(@as(u64, 48 * 2560 * 16), residentWeightBytes(null, null, 0, 48, 2560));
}

pub fn expertStreamingGateBytes(resident_bytes: u64, cache_bytes: u64, fill_peak_bytes: u64, bounce_bytes: u64) u64 {
    return resident_bytes +| cache_bytes +| fill_peak_bytes +| bounce_bytes;
}

/// Sum of `*.safetensors` bytes in `model_dir` — the MLX weight footprint used
/// by the load pre-flight. Returns 0 if the dir can't be read (treated as
/// "unknown" by the caller, which then skips the check). Symlinked weights
/// count (statFile follows links) — an HF hub-cache snapshot is ALL symlinks.
fn modelDiskBytes(io: std.Io, model_dir: []const u8) u64 {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    // A pack's index names the shards the loader reads; a stray shard beside
    // them (issue #274) is dead weight and must not be billed.
    var referenced: ?std.StringHashMapUnmanaged(void) = model_discovery.indexShardSet(io, dir);
    defer if (referenced) |*r| model_discovery.freeShardSet(r);
    var it = dir.iterate();
    var total: u64 = 0;
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        if (referenced) |r| if (!r.contains(entry.name)) continue;
        const st = dir.statFile(io, entry.name, .{}) catch continue;
        if (st.kind != .file) continue;
        total += @intCast(st.size);
    }
    return total;
}

/// Bytes of the MTP head sidecar `mtp.loadMtp` reads beside the trunk shards: 0 when MTP is off, the
/// head rides the checkpoint (its shards are billed already), none ships, or the arch loads its head
/// elsewhere (MiMo and GLM never call `loadMtp`).
fn mtpSidecarBytes(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8, mtp_on: bool) u64 {
    if (!mtp_on or config.isMimo() or config.isGlm5()) return 0;
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return 0;
    defer dir.close(io);
    const rel = switch (mtp_mod.resolveMtpSource(io, allocator, dir) orelse return 0) {
        .sidecar_file => |r| r,
        .in_checkpoint => return 0,
    };
    const st = dir.statFile(io, rel, .{}) catch return 0;
    return @intCast(st.size);
}

/// The resident weight bill of a load no component-exact bill covers: the shards the index names plus the sidecar head.
fn plainWeightBytes(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8, mtp_on: bool) u64 {
    return modelDiskBytes(io, model_dir) +| mtpSidecarBytes(io, allocator, config, model_dir, mtp_on);
}

/// A one-tensor safetensors file: 8-byte LE header length, JSON header, `payload` zero bytes.
fn writeTestSafetensors(io: std.Io, dir: std.Io.Dir, rel: []const u8, key: []const u8, payload: usize) !u64 {
    var header_buf: [256]u8 = undefined;
    const header = try std.fmt.bufPrint(&header_buf, "{{\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}}}}", .{ key, payload / 2, payload });
    var file: [512]u8 = undefined;
    std.mem.writeInt(u64, file[0..8], header.len, .little);
    @memcpy(file[8 .. 8 + header.len], header);
    const total = 8 + header.len + payload;
    @memset(file[8 + header.len .. total], 0);
    if (std.fs.path.dirname(rel)) |d| try dir.createDirPath(io, d);
    try dir.writeFile(io, .{ .sub_path = rel, .data = file[0..total] });
    return total;
}

fn testTmpDirPath(tmp: *std.testing.TmpDir, buf: []u8) ![]const u8 {
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    return std.fmt.bufPrint(buf, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
}

test "the preflight bills a separately loaded MTP sidecar once, never an in-checkpoint head twice" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try testTmpDirPath(&tmp, &path_buf);

    const shard = try writeTestSafetensors(io, tmp.dir, "model.safetensors", "model.embed_tokens.weight", 8);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"model.embed_tokens.weight\":\"model.safetensors\"}}" });
    const plain = ModelConfig{ .model_type = "qwen4_exp" };
    try testing.expectEqual(shard, plainWeightBytes(io, a, &plain, dir, true));

    const sidecar = try writeTestSafetensors(io, tmp.dir, "mtp/weights.safetensors", "mtp.fc.weight", 16);
    try testing.expectEqual(shard + sidecar, plainWeightBytes(io, a, &plain, dir, true));
    // The head is billed only when it will load, and only where `loadMtp` loads it.
    try testing.expectEqual(shard, plainWeightBytes(io, a, &plain, dir, false));
    const mimo = ModelConfig{ .model_type = "mimo_v2" };
    const glm = ModelConfig{ .model_type = "glm5_next" };
    try testing.expectEqual(@as(u64, 0), mtpSidecarBytes(io, a, &mimo, dir, true));
    try testing.expectEqual(@as(u64, 0), mtpSidecarBytes(io, a, &glm, dir, true));

    // A head inside the shards is already in the shard sum.
    try tmp.dir.deleteFile(io, "mtp/weights.safetensors");
    const with_head = try writeTestSafetensors(io, tmp.dir, "model.safetensors", "mtp.fc.weight", 8);
    try testing.expectEqual(with_head, plainWeightBytes(io, a, &plain, dir, true));
}

test "the Sushi resident bill carries a separately loaded MTP sidecar and its coarse rerank copy" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try testTmpDirPath(&tmp, &path_buf);

    _ = try writeTestSafetensors(io, tmp.dir, "model.safetensors", "model.embed_tokens.weight", 8);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"model.embed_tokens.weight\":\"model.safetensors\"}}" });
    var config = ModelConfig{ .model_type = "qwen4_exp" };
    config.vocab_size = 128;
    config.hidden_size = 64;
    const without = try sushiResidentLoadBytes(io, a, dir, &config, false, true);
    const sidecar = try writeTestSafetensors(io, tmp.dir, "mtp/weights.safetensors", "mtp.fc.weight", 16);
    const with = try sushiResidentLoadBytes(io, a, dir, &config, false, true);
    try testing.expect(with >= without + sidecar);
    try testing.expectEqual(without, try sushiResidentLoadBytes(io, a, dir, &config, false, false));
}

test "modelDiskBytes follows HF-cache symlinks (a snapshot dir measured ZERO)" {
    // A model served straight out of the HuggingFace hub cache is a snapshot
    // dir of SYMLINKS into ../../blobs. Skipping .sym_link entries measured a
    // 121 GB checkpoint at 0 bytes, and memInsufficientForLoad treats 0 as
    // "unknown" → the preflight waved the load through into 34 GB of swap.
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.createDirPath(io, "blobs");
    try tmp.dir.writeFile(io, .{ .sub_path = "blobs/abc123", .data = "0123456789abcdef" });
    try tmp.dir.createDirPath(io, "snapshots/rev");
    try tmp.dir.symLink(io, "../../blobs/abc123", "snapshots/rev/model.safetensors", .{});
    // A dangling link (blob pruned) is skipped, never an error…
    try tmp.dir.symLink(io, "../../blobs/gone", "snapshots/rev/model-00002.safetensors", .{});
    // …and a symlink to a DIRECTORY must not be summed (statFile follows it).
    try tmp.dir.symLink(io, "../../blobs", "snapshots/rev/dir.safetensors", .{});

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const snap = try std.fmt.allocPrint(std.testing.allocator, "{s}/.zig-cache/tmp/{s}/snapshots/rev", .{ cwd, tmp.sub_path });
    defer std.testing.allocator.free(snap);

    try std.testing.expectEqual(@as(u64, 16), modelDiskBytes(io, snap));
}

test "modelDiskBytes bills only the shards the index names (issue #274)" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "m");
    try tmp.dir.writeFile(io, .{ .sub_path = "m/model-00001-of-00002.safetensors", .data = "0123456789" });
    try tmp.dir.writeFile(io, .{ .sub_path = "m/model-00002-of-00002.safetensors", .data = "01234" });
    // Dead weight: present on disk, referenced by nothing.
    try tmp.dir.writeFile(io, .{ .sub_path = "m/stray.safetensors", .data = "0123456789abcdef0123456789abcdef" });
    try tmp.dir.writeFile(io, .{ .sub_path = "m/model.safetensors.index.json", .data =
        \\{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors","c":"model-00001-of-00002.safetensors"}}
    });

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const dir = try std.fmt.allocPrint(std.testing.allocator, "{s}/.zig-cache/tmp/{s}/m", .{ cwd, tmp.sub_path });
    defer std.testing.allocator.free(dir);
    try std.testing.expectEqual(@as(u64, 15), modelDiskBytes(io, dir));
}

/// Set by `--skip-mem-preflight` (main.zig) to bypass the model-load memory
/// pre-flight below. A module global, not a `LoadParams` field, so it applies
/// uniformly to startup loads AND later hot-loads — matching the env var
/// (`SUSHI_SKIP_MEM_PREFLIGHT`) it replaced.
pub var skip_mem_preflight: bool = false;

/// The full requested-context bill and the largest context fitting the same budget.
pub const LoadServingBill = struct {
    requested_context: u32,
    needed: u64,
    max_context: u32,
    kv_bits: u64,
    chunk: u64,
};
pub var load_serving_bill: ?*const fn (*const ModelConfig, u64) ?LoadServingBill = null;

fn sushiAssistantLoadBytes(io: std.Io, allocator: std.mem.Allocator, directory: []const u8) !u64 {
    if (directory.len == 0) return 0;
    if (std.c.getenv("SUSHI_DFLASH")) |v| if (v[0] == '0' and dflash_mod.probeIsDflash(io, allocator, directory)) return 0;
    return @import("glm5_diagnostic.zig").assistantResidentBytes(io, allocator, directory);
}

fn sushiResidentLoadBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, load_vision: bool, mtp_on: bool) !u64 {
    if (config.isMimo()) return mimoResidentLoadBytes(io, allocator, model_dir, config, load_vision, mtp_on);
    const split = try model_mod.qwenResidentWeightBytes(io, allocator, model_dir, load_vision, mtp_on);
    const head_bytes = split.mtp +| mtpSidecarBytes(io, allocator, config, model_dir, mtp_on);
    const coarse_bits = mtp_mod.rerankCoarseBits();
    const coarse = if (mtp_on and head_bytes > 0 and coarse_bits > 0 and mtp_mod.MtpModel.draftRerankMode() != .off)
        mtp_mod.rerankCoarseBytes(@intCast(config.vocab_size), @intCast(config.hidden_size), coarse_bits)
    else
        0;
    return split.trunk +| head_bytes +| coarse;
}

/// The resident bytes a MiMo source-trunk load holds: trunk and vision tower as stored, the heads
/// when MTP is on, and the coarse lm_head copy the heads and the greedy readout share.
fn mimoResidentLoadBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, load_vision: bool, mtp_on: bool) !u64 {
    var bytes = try model_mod.mimoSourceResidentBytes(io, allocator, model_dir, load_vision and config.mimo_vision);
    if (mtp_on) bytes += try model_mod.mimoMtpResidentBytes(io, allocator, model_dir);
    return bytes + mimoCoarseHeadBytes(config);
}

/// A streamed GLM forward keeps no speculative state: a found assistant stays unloaded,
/// an asked-for one is refused by name.
fn glmStreamedDrafterOff(config: *const ModelConfig, drafter_dir: []const u8) !bool {
    if (!config.isGlm5() or !config.expert_streaming) return false;
    if (drafter_dir.len > 0) return error.GlmStreamingSpecUnsupported;
    log.info("[glm-dflash] off: unsupported under expert streaming\n", .{});
    return true;
}

test "a streamed GLM load leaves a found assistant unloaded and refuses an asked-for one" {
    var glm = ModelConfig{ .model_type = "glm5_next" };
    try testing.expect(!try glmStreamedDrafterOff(&glm, "/assistant"));
    glm.expert_streaming = true;
    try testing.expect(try glmStreamedDrafterOff(&glm, ""));
    try testing.expectError(error.GlmStreamingSpecUnsupported, glmStreamedDrafterOff(&glm, "/assistant"));
    const qwen = ModelConfig{ .model_type = "qwen4_exp", .expert_streaming = true };
    try testing.expect(!try glmStreamedDrafterOff(&qwen, ""));
}

/// The drafter a load binds: `--no-drafter` wins, then an explicit dir, then one shipped in the model dir.
const LoadDrafterDir = struct {
    dir: []const u8,
    owned: ?[]u8 = null,

    fn resolve(io: std.Io, allocator: std.mem.Allocator, no_drafter: bool, drafter_dir: []const u8, model_dir: []const u8) LoadDrafterDir {
        if (no_drafter) return .{ .dir = "" };
        if (drafter_dir.len > 0) return .{ .dir = drafter_dir };
        const in_dir = dflash_mod.resolveInDirDrafter(io, allocator, model_dir) orelse return .{ .dir = "" };
        return .{ .dir = in_dir, .owned = in_dir };
    }

    fn deinit(self: LoadDrafterDir, allocator: std.mem.Allocator) void {
        if (self.owned) |p| allocator.free(p);
    }
};

/// What a resident cold load reserves in the registry: the requirement its load preflight
/// compares with free memory, so the gate and the preflight read one bill.
fn residentColdLoadBillBytes(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8, load_vision: bool, mtp_on: bool, no_drafter: bool, drafter_dir: []const u8) !u64 {
    var weights = if (model_mod.usesSushiQuantMemoryBill(config))
        try sushiResidentLoadBytes(io, allocator, model_dir, config, load_vision, mtp_on)
    else
        try mimoResidentLoadBytes(io, allocator, model_dir, config, load_vision, mtp_on);
    const drafter = LoadDrafterDir.resolve(io, allocator, no_drafter, drafter_dir, model_dir);
    defer drafter.deinit(allocator);
    var context: u64 = 0;
    if (model_mod.usesSushiQuantMemoryBill(config)) {
        weights +|= try sushiAssistantLoadBytes(io, allocator, drafter.dir);
        if (load_serving_bill) |bill| if (bill(config, std.math.maxInt(u64))) |b| {
            context = b.needed;
        };
    }
    return loadRequirementBytes(config, weights, mtp_on, context);
}

test "a resident MiMo cold load reserves its load preflight's requirement" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var fixture = try @import("mimo_source.zig").makeTinySourceFixture(io, a, &tmp);
    defer fixture.deinit();
    fixture.config.expert_layout = .mxfp4_individual;
    const weights = try mimoResidentLoadBytes(io, a, fixture.path, &fixture.config, false, false);
    const preflight = loadRequirementBytes(&fixture.config, weights, false, 0);
    try testing.expectEqual(preflight, try residentColdLoadBillBytes(io, a, &fixture.config, fixture.path, false, false, false, ""));
    try testing.expect(weights > 0);
}

/// Which drafter directory a COLD load should use.
///
/// `--no-drafter` is a policy — it silences every model, including one whose
/// own dir ships a sidecar `dflash.resolveInDirDrafter` would otherwise find.
/// `--drafter <path>`, by contrast, names a sidecar for the checkpoint it was
/// passed beside: handing it to whatever model is swapped in next would load a
/// mismatched assistant, so it applies only when the entry being loaded IS the
/// launch model (which happens on a reload after eviction). Every other model
/// is served by the in-dir probe.
pub fn coldLoadDrafterDir(
    no_drafter: bool,
    primary_model_dir: []const u8,
    drafter_dir: []const u8,
    entry_path: []const u8,
) []const u8 {
    if (no_drafter) return "";
    if (drafter_dir.len == 0) return "";
    if (!std.mem.eql(u8, primary_model_dir, entry_path)) return "";
    return drafter_dir;
}

/// The load's vision choice: `--vision`/`--no-vision` (process-wide, so cold loads honour it
/// too) > the per-model `vision` > on resident, off streamed.
pub fn visionChoiceFor(config: *const ModelConfig, streamed: bool) model_settings.VisionChoice {
    return model_settings.VisionChoice.resolve(model_settings.vision_flag, config.vision_override, streamed);
}

/// Should a cold load bring up the checkpoint's vision tower?
pub fn coldLoadVision(config: *const ModelConfig, streamed: bool) bool {
    return config.has_vision and visionChoiceFor(config, streamed).on;
}

test "coldLoadDrafterDir: --no-drafter wins, an explicit --drafter belongs to its OWN model" {
    // `--drafter <path>` names a sidecar for the checkpoint it was passed
    // with; handing it to whatever model gets swapped in next would load a
    // mismatched assistant. Other models are served by the in-dir probe.
    // Reloading the launch model AFTER an eviction must still get it back.
    try testing.expectEqualStrings("/d", coldLoadDrafterDir(false, "/m", "/d", "/m"));
    try testing.expectEqualStrings("", coldLoadDrafterDir(false, "/m", "/d", "/other"));
    // --no-drafter is a policy, not a path: it silences every model, including
    // one whose own dir ships a sidecar the in-dir probe would find.
    try testing.expectEqualStrings("", coldLoadDrafterDir(true, "/m", "/d", "/m"));
    try testing.expectEqualStrings("", coldLoadDrafterDir(true, "/m", "/d", "/other"));
    // No --drafter at launch: nothing to carry, the in-dir probe decides.
    try testing.expectEqualStrings("", coldLoadDrafterDir(false, "/m", "", "/m"));
}

test "the expert cache resolves from --ssd-budget-gb unless --expert-cache-gb is explicit" {
    const t = testing;
    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 48;
    cfg.num_experts = 512;
    cfg.num_experts_per_tok = 10;
    cfg.hidden_size = 2560;
    cfg.moe_intermediate_size = 640;
    const split = model_mod.ResidentSplit{ .trunk = 9_900_000_000, .mtp = 5_200_000_000 };
    const GiB: u64 = 1 << 30;
    const fused: u64 = 9_830_400;

    const from_budget = try resolveExpertCache(0, 60 * GiB, &cfg, split, false, fused);
    try t.expect(from_budget.ledger != null);
    try t.expectEqual(@as(u16, 103), from_budget.ledger.?.slots_per_layer);
    try t.expectEqual(from_budget.ledger.?.cache_bytes, from_budget.cache_bytes);
    try t.expect(!from_budget.overridden);

    const with_mtp = try resolveExpertCache(0, 60 * GiB, &cfg, split, true, fused);
    try t.expectEqual(@as(u16, 92), with_mtp.ledger.?.slots_per_layer);

    const explicit = try resolveExpertCache(60_000_000_000, 60 * GiB, &cfg, split, false, fused);
    try t.expectEqual(@as(u64, 60_000_000_000), explicit.cache_bytes);
    try t.expect(explicit.ledger == null);
    try t.expect(explicit.overridden);

    const no_budget = try resolveExpertCache(60_000_000_000, 0, &cfg, split, false, fused);
    try t.expectEqual(@as(u64, 60_000_000_000), no_budget.cache_bytes);
    try t.expect(!no_budget.overridden);

    try t.expectError(error.SsdBudgetBelowResident, resolveExpertCache(0, 14 * GiB, &cfg, split, false, fused));
}

test "streamed vision: the tower comes out of the ssd budget only when asked, refused by name when it does not fit" {
    const t = testing;
    const cfg = model_mod.ModelConfig{
        .num_hidden_layers = 48,
        .num_experts = 512,
        .num_experts_per_tok = 10,
        .hidden_size = 2560,
        .moe_intermediate_size = 640,
        .has_vision = true,
        .qwen_vision = true,
        .qv_heads = 16,
        .qv_hidden = 1152,
        .qv_intermediate = 4304,
        .qv_out_hidden = 2560,
        .qv_patch = 16,
        .qv_temporal_patch = 2,
        .qv_merge = 2,
    };
    const split = model_mod.ResidentSplit{ .trunk = 9_900_000_000, .mtp = 0, .vision = 897_862_112 };
    const GiB: u64 = 1 << 30;
    const fused: u64 = 9_830_400;
    const encode = @import("server.zig").largestImageEncodeBytes(&cfg);
    try t.expect(encode > 0);

    const off = try resolveStreamedVision(0, 60 * GiB, &cfg, split, fused, false);
    try t.expect(!off.cost.on);
    try t.expectEqual(@as(u64, 0), off.resolved.ledger.?.vision_bytes);
    const on = try resolveStreamedVision(0, 60 * GiB, &cfg, split, fused, true);
    try t.expect(on.cost.on);
    try t.expectEqual(split.vision, on.resolved.ledger.?.vision_bytes);
    try t.expectEqual(encode, on.cost.encode_bytes);
    try t.expectEqual(off.resolved.ledger.?.slots_per_layer, on.cost.slots_without);
    try t.expectEqual(on.resolved.ledger.?.slots_per_layer, on.cost.slots_with.?);
    try t.expect(on.cost.slots_with.? < on.cost.slots_without);
    // Off, the plan still prices what --vision would take.
    try t.expectEqual(on.cost.slots_with, off.cost.slots_with);

    const fixed = split.trunk + 522 * fused + expert_stream_mod.BOUNCE_BYTES;
    const tight = fixed + 2 * 48 * fused;
    try t.expectEqual(@as(?u16, null), (try resolveStreamedVision(0, tight, &cfg, split, fused, false)).cost.slots_with);
    try t.expectError(error.SsdBudgetBelowVision, resolveStreamedVision(0, tight, &cfg, split, fused, true));
    try t.expectError(error.SsdBudgetBelowResident, resolveStreamedVision(0, 14 * GiB, &cfg, split, fused, true));

    // --expert-cache-gb sizes the cache itself; the tower is resident beside it.
    const explicit = try resolveStreamedVision(60_000_000_000, 0, &cfg, split, fused, true);
    try t.expect(explicit.cost.on);
    try t.expectEqual(@as(u64, 60_000_000_000), explicit.resolved.cache_bytes);
    var bare = split;
    bare.vision = 0;
    try t.expect(!(try resolveStreamedVision(0, 60 * GiB, &cfg, bare, fused, true)).cost.on);
}

test "the vision load line names its source, and what --vision would cost a streamed load" {
    const t = testing;
    var buf: [256]u8 = undefined;
    const cost = VisionCost{ .tower_bytes = 900_000_000, .encode_bytes = 1_700_000_000, .slots_without = 103, .slots_with = 101 };
    const off = visionLoadLine(&buf, .{ .on = false, .source = .default }, true, cost);
    try t.expectEqualStrings("[vision] off under expert streaming; pass --vision to load the tower (+0.90 GB, -2 slots/layer)", off);
    var on_cost = cost;
    on_cost.on = true;
    try t.expectEqualStrings("[vision] on (--vision): tower 0.90 GB in the ssd budget (-2 slots/layer); an image encode up to 1.70 GB is billed per request", visionLoadLine(&buf, .{ .on = true, .source = .flag }, true, on_cost));
    var cache_cost = cost;
    cache_cost.slots_with = null;
    cache_cost.slots_without = 0;
    try t.expectEqualStrings("[vision] off under expert streaming; pass --vision to load the tower (+0.90 GB resident)", visionLoadLine(&buf, .{ .on = false, .source = .default }, true, cache_cost));
    cache_cost.on = true;
    try t.expectEqualStrings("[vision] on (model-settings.json): tower 0.90 GB resident beside the expert cache; an image encode up to 1.70 GB is billed per request", visionLoadLine(&buf, .{ .on = true, .source = .model_settings }, true, cache_cost));
    cache_cost.on = false;
    cache_cost.slots_without = 103;
    try t.expectEqualStrings("[vision] off under expert streaming; pass --vision to load the tower (+0.90 GB: more than this ssd budget holds)", visionLoadLine(&buf, .{ .on = false, .source = .default }, true, cache_cost));
    try t.expectEqualStrings("[vision] off (--no-vision)", visionLoadLine(&buf, .{ .on = false, .source = .flag }, true, cost));
    try t.expectEqualStrings("[vision] on (default)", visionLoadLine(&buf, .{ .on = true, .source = .default }, false, .{}));
    try t.expectEqualStrings("[vision] off (model-settings.json)", visionLoadLine(&buf, .{ .on = false, .source = .model_settings }, false, .{}));
}

test "the ssd budget falls back to the per-model setting, and a non-streaming model ignores it" {
    const t = testing;
    const GiB: u64 = 1 << 30;

    const flag_wins = resolveSsdBudget(60 * GiB, 40, true);
    try t.expectEqual(60 * GiB, flag_wins.bytes);
    try t.expect(!flag_wins.from_setting);
    try t.expect(!flag_wins.setting_ignored);

    const from_setting = resolveSsdBudget(0, 60, true);
    try t.expectEqual(60 * GiB, from_setting.bytes);
    try t.expect(from_setting.from_setting);
    try t.expect(!from_setting.setting_ignored);

    const neither = resolveSsdBudget(0, 0, true);
    try t.expectEqual(@as(u64, 0), neither.bytes);
    try t.expect(!neither.from_setting);

    const not_streaming = resolveSsdBudget(0, 60, false);
    try t.expectEqual(@as(u64, 0), not_streaming.bytes);
    try t.expect(not_streaming.setting_ignored);
    try t.expect(!resolveSsdBudget(0, 0, false).setting_ignored);

    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 48;
    cfg.num_experts = 512;
    cfg.num_experts_per_tok = 10;
    cfg.hidden_size = 2560;
    cfg.moe_intermediate_size = 640;
    const split = model_mod.ResidentSplit{ .trunk = 9_900_000_000, .mtp = 5_200_000_000 };
    const via_setting = try resolveExpertCache(0, resolveSsdBudget(0, 60, true).bytes, &cfg, split, false, 9_830_400);
    try t.expectEqual(@as(u16, 103), via_setting.ledger.?.slots_per_layer);
    const cache_flag = try resolveExpertCache(60_000_000_000, resolveSsdBudget(0, 60, true).bytes, &cfg, split, false, 9_830_400);
    try t.expectEqual(@as(u64, 60_000_000_000), cache_flag.cache_bytes);
    try t.expect(cache_flag.overridden);
}

test "a non-qwen4 checkpoint never engages streaming: the flag is dropped, the per-model setting warns" {
    const t = testing;
    const GiB: u64 = 1 << 30;
    var moe = model_mod.ModelConfig{
        .model_type = "qwen3_5_moe",
        .num_hidden_layers = 48,
        .num_experts = 512,
        .num_experts_per_tok = 10,
        .hidden_size = 2560,
        .moe_intermediate_size = 640,
    };
    applyModelSettings(&moe, .{ .ssd_budget_gb = 60 });
    try t.expectEqual(@as(u32, 60), moe.ssd_budget_gb_override);
    try t.expect(!moe.supportsExpertStreaming());

    const from_setting = resolveSsdBudget(0, moe.ssd_budget_gb_override, moe.supportsExpertStreaming());
    try t.expectEqual(@as(u64, 0), from_setting.bytes);
    try t.expect(from_setting.setting_ignored);
    const from_flag = resolveSsdBudget(60 * GiB, 0, moe.supportsExpertStreaming());
    try t.expectEqual(@as(u64, 0), from_flag.bytes);
    try t.expect(!from_flag.setting_ignored);

    try t.expect(!expert_stream_mod.expertStreamingEngaged(
        moe.supportsExpertStreaming(),
        moe.expertStreamingRequired(),
        60_000_000_000,
        from_flag.bytes,
    ));
    try t.expect(!moe.expert_streaming);

    var q4 = moe;
    q4.model_type = "qwen4_exp";
    try t.expectEqual(60 * GiB, resolveSsdBudget(60 * GiB, 0, q4.supportsExpertStreaming()).bytes);
    try t.expect(expert_stream_mod.expertStreamingEngaged(
        q4.supportsExpertStreaming(),
        q4.expertStreamingRequired(),
        0,
        resolveSsdBudget(0, q4.ssd_budget_gb_override, q4.supportsExpertStreaming()).bytes,
    ));
}

test "EXL3 streaming CPU settings budget engages the EXL3 loader" {
    // The streamed load refuses an EXL3 layout by name; the budget used to engage it anyway, so a
    // settings-file budget made the pack unloadable with a "re-convert the pack" 503.
    const t = std.testing;
    var q4 = ModelConfig{
        .model_type = "qwen4_exp",
        .num_hidden_layers = 4,
        .num_experts = 4,
        .num_experts_per_tok = 2,
        .hidden_size = 128,
        .moe_intermediate_size = 128,
        .expert_layout = .exl3_k4,
    };
    q4.ssd_budget_gb_override = 60;
    try t.expect(q4.supportsExpertStreaming());
    const budget = resolveSsdBudget(0, q4.ssd_budget_gb_override, q4.supportsExpertStreaming());
    try t.expect(!budget.setting_ignored);
    try t.expectEqual(@as(u64, 60 << 30), budget.bytes);
    try t.expect(expert_stream_mod.expertStreamingEngaged(q4.supportsExpertStreaming(), q4.expertStreamingRequired(), 0, budget.bytes));
}

test "the cold-load LoadRequest re-applies EVERY retained launch setting" {
    // Three separate rounds of this bug shipped: prefix-cache, then MTP,
    // then the drafter/ssd group — each time a launch flag reached
    // only `--model` and the cold path (hot switch, /v1/load-model, first
    // request naming an unloaded model) quietly used a struct default. The
    // scan is the class guard: a field retained on the Scheduler for this
    // purpose that no cold-load assignment mentions is the next round.
    // Needles are ++-split so this test's own source can't satisfy the scan.
    const src = @embedFile("scheduler.zig");
    inline for (.{
        "kv_quant_config",           "prefix_cache_capacity", "prefix_cache_mem_bytes",
        "prefix_cache_disk_bytes",   "ssm_checkpoint_stride", "ssm_checkpoint_max",
        "mtp_enabled",               "mtp_head_kv_quant",     "mtp_depth",
        "no_drafter",                "draft_block_size",      "draft_block_size_explicit",
        "prefix_cache_mem_resolver", "expert_cache_bytes",    "expert_cache_fit_resolver",
        "ssd_budget_bytes",          "kv_quant_explicit",     "mtp_explicit",
        "prefix_cache_disk_resolver",
    }) |field| {
        const needle = "." ++ field ++ " = self" ++ "." ++ field ++ ",";
        try testing.expect(std.mem.indexOf(u8, src, needle) != null);
    }
    // The drafter path is the one retained setting that must NOT be copied
    // straight across — it goes through the ownership rule above.
    const via_rule = "coldLoadDrafterDir(" ++ "self.no_drafter, self.primary_model_dir, self.drafter_dir, entry.path)";
    try testing.expect(std.mem.indexOf(u8, src, via_rule) != null);
    // The stale TODO that stood in for the wiring must be gone.
    const old = "Phase E will wire the load-model API" ++ " to set this.";
    try testing.expect(std.mem.indexOf(u8, src, old) == null);
}

test "bf16 streaming registry gate uses resident plan not checkpoint disk size" {
    try testing.expectEqual(@as(u64, 81), expertStreamingGateBytes(10, 60, 5, 6));
    try testing.expect(expertStreamingGateBytes(15_000_000_000, 60_000_000_000, 5_033_164_800, 536_870_912) < 360_000_000_000);
}

test "every HotPrefixCache.initWithMem load site reads the CLAMPED budget, never the raw launch flag" {
    // 2026-08-30 Metal OOM: a 40 GB `--prefix-cache-mem` beside a ~70 GB pack
    // was never validated against what the weights left under the GPU ceiling;
    // the cache filled to its cap and a 143k prefill died uncatchably. The
    // budget must pass through the resolver (server.prefixCacheMemForLoad)
    // before it reaches initWithMem — and a SECOND construction site must not
    // be able to skip the clamp (the cold-load launch flags class).
    const src = @embedFile("scheduler.zig");
    const needle = "HotPrefixCache.initWith" ++ "Mem(";
    var found: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, src, pos, needle)) |at| : (pos = at + needle.len) {
        found += 1;
        const window = src[at..@min(at + 240, src.len)];
        try testing.expect(std.mem.indexOf(u8, window, "clamped_prefix_mem") != null);
        try testing.expect(std.mem.indexOf(u8, window, "params.prefix_cache_mem_bytes") == null);
    }
    try testing.expect(found >= 1);
}

test "coldLoadVision honors --no-vision, --vision and the per-model vision, resident and streamed" {
    const saved = model_settings.vision_flag;
    defer model_settings.vision_flag = saved;
    var cfg = ModelConfig{ .has_vision = true };
    model_settings.vision_flag = null;
    try std.testing.expect(coldLoadVision(&cfg, false));
    try std.testing.expect(!coldLoadVision(&cfg, true));
    try std.testing.expect(!coldLoadVision(&ModelConfig{}, false));
    model_settings.vision_flag = false;
    try std.testing.expect(!coldLoadVision(&cfg, false));
    model_settings.vision_flag = true;
    try std.testing.expect(coldLoadVision(&cfg, true));
    try std.testing.expect(!coldLoadVision(&ModelConfig{}, true));
    model_settings.vision_flag = null;
    cfg.vision_override = true;
    try std.testing.expect(coldLoadVision(&cfg, true));
    cfg.vision_override = false;
    try std.testing.expect(!coldLoadVision(&cfg, false));
}

/// Which "available memory" figure the preflight should trust. On iOS the
/// host-wide number is meaningless — the OS keeps RAM full (file cache,
/// jetsam-evictable background apps) and will evict on our behalf, so the
/// per-PROCESS jetsam headroom (`os_proc_available_memory`, nonzero only on
/// iOS) is the figure that decides whether the load survives. Live bug: an
/// 8 GB iPhone reported ~4 GB host-free and the preflight refused a 3.6 GB
/// model that fit comfortably inside the ~6.4 GB process limit.
/// `gpu_limit` = Metal's working-set limit (0 = unknown): a lowered `iogpu.wired_limit_mb` binds
/// below free RAM, and weights past it OOM in warmup instead of refusing by name.
fn effectiveAvailableBytes(host_avail: u64, proc_avail: u64, gpu_limit: u64) u64 {
    const avail = if (proc_avail > 0) proc_avail else host_avail;
    return if (gpu_limit > 0) @min(avail, gpu_limit) else avail;
}

/// A just-exited process returns its Metal memory over seconds: while a short reading keeps
/// rising, poll every 500 ms; stop at enough, after 2 s without growth, or after 15 s.
fn settleAvailableBytes(first: u64, needed: u64, reader: anytype) u64 {
    var avail = first;
    var stalled: u32 = 0;
    var polls: u32 = 0;
    while (avail > 0 and avail < needed and stalled < 4 and polls < 30) : (polls += 1) {
        reader.sleep();
        const next = reader.read();
        stalled = if (next > avail) 0 else stalled + 1;
        avail = next;
    }
    return avail;
}

/// A resident Sushi load shares the working-set cap with already active model buffers.
fn effectiveSushiAvailableBytes(host_avail: u64, proc_avail: u64, gpu_limit: u64, active: u64) u64 {
    const available = effectiveAvailableBytes(host_avail, proc_avail, gpu_limit);
    return if (gpu_limit > 0) @min(available, gpu_limit -| active) else available;
}

test "load preflight waits for a prior process's memory to come back, and only while it does" {
    const gib: u64 = 1 << 30;
    const Fake = struct {
        seq: []const u64,
        i: usize = 0,
        fn read(f: *@This()) u64 {
            const v = f.seq[@min(f.i, f.seq.len - 1)];
            f.i += 1;
            return v;
        }
        fn sleep(_: *@This()) void {}
    };
    var rising = Fake{ .seq = &.{ 80 * gib, 95 * gib, 110 * gib } };
    try testing.expectEqual(@as(u64, 110 * gib), settleAvailableBytes(69 * gib, 100 * gib, &rising));
    var short = Fake{ .seq = &.{ 70 * gib, 70 * gib, 70 * gib, 70 * gib, 70 * gib, 200 * gib } };
    try testing.expectEqual(@as(u64, 70 * gib), settleAvailableBytes(69 * gib, 100 * gib, &short));
    try testing.expectEqual(@as(usize, 5), short.i);
    var never = Fake{ .seq = &.{200 * gib} };
    try testing.expectEqual(@as(u64, 120 * gib), settleAvailableBytes(120 * gib, 100 * gib, &never));
    try testing.expectEqual(@as(usize, 0), never.i);
}

test "Sushi quant memory subtracts existing resident buffers from the GPU limit" {
    const gib: u64 = 1 << 30;
    try testing.expectEqual(@as(u64, 54 * gib), effectiveSushiAvailableBytes(88 * gib, 0, 94 * gib, 40 * gib));
    try testing.expectEqual(@as(u64, 30 * gib), effectiveSushiAvailableBytes(30 * gib, 0, 94 * gib, 40 * gib));
    try testing.expectEqual(@as(u64, 0), effectiveSushiAvailableBytes(88 * gib, 0, 40 * gib, 40 * gib));
    try testing.expectEqual(@as(u64, 88 * gib), effectiveSushiAvailableBytes(88 * gib, 0, 0, 40 * gib));
}

test "effectiveAvailableBytes is capped by the GPU working-set limit" {
    const GB: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(36 * GB, effectiveAvailableBytes(98 * GB, 0, 36 * GB));
    try std.testing.expect(memInsufficientForLoad(70 * GB, effectiveAvailableBytes(98 * GB, 0, 36 * GB), 71 * GB));
}

test "effectiveAvailableBytes prefers the per-process jetsam headroom when present" {
    const GB: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(6 * GB, effectiveAvailableBytes(4 * GB, 6 * GB, 0)); // iOS: proc wins
    try std.testing.expectEqual(4 * GB, effectiveAvailableBytes(4 * GB, 0, 0)); // macOS: proc query = 0 → host
    try std.testing.expectEqual(@as(u64, 0), effectiveAvailableBytes(0, 0, 0)); // both unknown → 0 (never blocks)
}

/// Free memory a load keeps beyond everything it bills: allocator slack and pages macOS has not yet handed back.
const LOAD_SAFETY_NET_BYTES: u64 = 1 << 30;

/// What a load's warmup allocates above its billed weights: the measured peak over the billed bytes at
/// each architecture's worst setting, rounded up (engine-memory-admission.md#load-warmup).
fn loadWarmupBytes(config: *const ModelConfig, mtp_on: bool) u64 {
    const gib: u64 = 1 << 30;
    // GLM warms T=1 and T=8, then resets; its KDA state, rings and first latent rows exist from then on.
    if (config.isGlm5()) return config.ssmCheckpointBytes() +| config.qsaRingBytes() +|
        (config.kvBytesPerToken() +| config.qsaHistoryBytesPerToken()) *| 1024 +| gib * 3 / 8;
    if (config.isMimo()) return gib / 4;
    // Flash-Next: reading the MTP head peaks about a GiB above its resident bytes.
    return gib / 8 +| (if (mtp_on) gib * 9 / 8 else 0);
}

/// A streamed load holds the cache, the whole-layer fill union and the bounce buffers beside the resident trunk.
fn streamedLoadRequirementBytes(resident_bytes: u64, cache_bytes: u64, fill_peak_bytes: u64, bounce_bytes: u64) u64 {
    return expertStreamingGateBytes(resident_bytes, cache_bytes, fill_peak_bytes, bounce_bytes) +| LOAD_SAFETY_NET_BYTES;
}

/// Free memory a load demands: its billed weights, then the larger of the warmup transient and the requested
/// context's admission bill (the warmup ends before any request allocates), then the safety net.
pub fn loadRequirementBytes(config: *const ModelConfig, weights_bytes: u64, mtp_on: bool, context_bill: u64) u64 {
    if (config.expert_streaming) return streamedLoadRequirementBytes(weights_bytes, config.expert_cache_bytes, config.expert_fill_peak_bytes, config.expert_bounce_bytes);
    return weights_bytes +| @max(loadWarmupBytes(config, mtp_on), context_bill) +| LOAD_SAFETY_NET_BYTES;
}

/// Unknown figures (a failed memory query, an unmeasurable size) never block a load.
fn memInsufficientForLoad(weights_bytes: u64, avail_bytes: u64, required_bytes: u64) bool {
    if (weights_bytes == 0 or avail_bytes == 0) return false;
    return avail_bytes < required_bytes;
}

fn glmDflashLoadBytes(io: std.Io, allocator: std.mem.Allocator, config: *ModelConfig, model_dir: []const u8, directory: []const u8) !u64 {
    if (directory.len == 0) return 0;
    if (std.c.getenv("SUSHI_DFLASH")) |v| if (v[0] == '0') return 0;
    if (!dflash_mod.probeIsDflash(io, allocator, directory)) return error.GlmDflashAssistantRequired;
    var cfg = try dflash_mod.parseConfig(io, allocator, directory);
    defer cfg.deinit(allocator);
    if (!cfg.isDflash2() or cfg.hidden_size != config.hidden_size or cfg.sliding_window < 2) return error.GlmDraftTargetMismatch;
    for (cfg.layer_types) |kind| if (kind != .sliding_attention) return error.GlmDflashWindowUnsupported;
    config.glm_dflash_loaded = true;
    const per_token = @as(u64, cfg.num_hidden_layers) * cfg.num_key_value_heads * cfg.head_dim * 4;
    // A draft block grows a 2047-row window to the next 256-row allocation.
    config.glm_dflash_window_bytes = per_token * ((@as(u64, cfg.sliding_window) + cfg.block_size + 255) / 256 * 256);
    config.glm_dflash_capture_bytes_per_token = (@as(u64, cfg.target_layer_ids.len) * cfg.hidden_size + @as(u64, cfg.hidden_size) * 2) * 2 + per_token;
    const runtime_cache = @import("glm5_dflash_cache.zig");
    return if (runtime_cache.isShippedSource(directory, model_dir))
        try runtime_cache.plannedResidentBytes(io, allocator, directory)
    else
        try @import("glm5_diagnostic.zig").assistantResidentBytes(io, allocator, directory);
}

fn glmColdLoadBillBytes(io: std.Io, allocator: std.mem.Allocator, config: *ModelConfig, model_dir: []const u8, load_vision: bool, no_drafter: bool, drafter_dir: []const u8) !u64 {
    const drafter = LoadDrafterDir.resolve(io, allocator, no_drafter, drafter_dir, model_dir);
    defer drafter.deinit(allocator);
    const weights = try @import("glm5_diagnostic.zig").residentBytesWithVision(io, allocator, model_dir, config.num_hidden_layers, load_vision and config.glm5_vision);
    return loadRequirementBytes(config, weights +| try glmDflashLoadBytes(io, allocator, config, model_dir, drafter.dir), false, 0);
}

test "Sushi quant memory production pack CPU header audit" {
    const raw = std.c.getenv("SUSHI_MEMORY_AUDIT_MODEL") orelse return error.SkipZigTest;
    const path = std.mem.span(raw);
    const io = std.testing.io;
    const a = std.testing.allocator;
    var cfg = try model_mod.parseConfig(io, a, path);
    defer cfg.deinit(a);
    try testing.expect(model_mod.usesSushiQuantMemoryBill(&cfg));
    const text = try sushiResidentLoadBytes(io, a, path, &cfg, false, false);
    const vision = try sushiResidentLoadBytes(io, a, path, &cfg, true, false);
    const all = try sushiResidentLoadBytes(io, a, path, &cfg, true, true);
    try testing.expect(all > vision);
    try testing.expect(vision > text);
    try testing.expectEqual(loadRequirementBytes(&cfg, all, true, 0), try residentColdLoadBillBytes(io, a, &cfg, path, true, true, true, ""));
    std.debug.print("Sushi memory audit {s}: text={d}, vision={d}, mtp={d}, all={d} bytes\n", .{ path, text, vision - text, all - vision, all });
}

const LoadBillFixtures = struct {
    qwen: ModelConfig,
    mimo: ModelConfig,
    glm: ModelConfig,

    fn init() !LoadBillFixtures {
        const a = testing.allocator;
        var qwen = try model_mod.parseConfigFromJson(a, @embedFile("fixtures/model-configs/qwen4_exp.json"));
        errdefer qwen.deinit(a);
        var mimo = try model_mod.parseConfigFromJson(a, @embedFile("fixtures/model-configs/mimo_v2.json"));
        errdefer mimo.deinit(a);
        return .{ .qwen = qwen, .mimo = mimo, .glm = try model_mod.parseConfigFromJson(a, @embedFile("fixtures/glm5_config.json")) };
    }

    fn deinit(self: *LoadBillFixtures) void {
        self.qwen.deinit(testing.allocator);
        self.mimo.deinit(testing.allocator);
        self.glm.deinit(testing.allocator);
    }
};

test "a load bills weights, the measured warmup or the context bill, and a 1 GiB net" {
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
    // A requested context's admission bill replaces the warmup once it is the larger; the net rides on either.
    try testing.expectEqual(44 * gib + 6 * gib + gib, loadRequirementBytes(&f.qwen, 44 * gib, true, 6 * gib));
    try testing.expectEqual(44 * gib + gib * 5 / 4 + gib, loadRequirementBytes(&f.qwen, 44 * gib, true, gib));
}

test "a streamed load bills its trunk, expert cache, fill union and bounce buffers plus the net" {
    const gib: u64 = 1 << 30;
    const cfg = ModelConfig{ .model_type = "qwen4_exp", .expert_streaming = true, .expert_cache_bytes = 12 * gib, .expert_fill_peak_bytes = 2 * gib, .expert_bounce_bytes = gib / 2 };
    try testing.expectEqual(6 * gib + 12 * gib + 2 * gib + gib / 2 + gib, loadRequirementBytes(&cfg, 6 * gib, true, 0));
    try testing.expectEqual(loadRequirementBytes(&cfg, 6 * gib, true, 0), streamedLoadRequirementBytes(6 * gib, 12 * gib, 2 * gib, gib / 2));
}

test "the exact load bill admits a MiMo load the flat 7 GiB headroom refused" {
    const gib: u64 = 1 << 30;
    var f = try LoadBillFixtures.init();
    defer f.deinit();
    const weights = 90 * gib;
    const available = 92 * gib;
    // The retired headroom demanded weights + min(weights / 8, 6 GiB) + 1 GiB.
    try testing.expect(memInsufficientForLoad(weights, available, weights + 7 * gib));
    const exact = loadRequirementBytes(&f.mimo, weights, true, 0);
    try testing.expect(!memInsufficientForLoad(weights, available, exact));
    try testing.expect(!memInsufficientForLoad(weights, exact, exact));
    try testing.expect(memInsufficientForLoad(weights, exact - 1, exact));
    // Unknown figures (a failed memory query, an unmeasurable size) never block a load.
    try testing.expect(!memInsufficientForLoad(0, available, exact));
    try testing.expect(!memInsufficientForLoad(weights, 0, exact));
}

test "the eviction gate bills weights plus 10% headroom" {
    try testing.expectEqual(@as(u64, 1100), gateEstimateBytes(1000, 0, 0));
    try testing.expectEqual(@as(u64, 0), gateEstimateBytes(0, 0, 0));
    // No bytes_on_disk → the layers × hidden × 16 fallback.
    const fallback: u64 = 32 * 4096 * 16;
    try testing.expectEqual(fallback + fallback / 10, gateEstimateBytes(null, 32, 4096));
}

/// Phase A1 → Plan 05: do the full model load on the inference thread.
/// mlx ops here bind to this thread's GPU stream from t0; subsequent
/// forwards stay on the same thread.
///
/// `params` is duck-typed (`anytype`): both `LoadParams` (startup) and
/// `*LoadRequest` (on-demand) supply the same field set — `entry`,
/// `config`/`tok`/`chat_config` (heap pointers), `model_dir`,
/// `drafter_dir`, `load_vision`, `warmup_eager`, `draft_block_size`,
/// `draft_block_size_explicit`, `kv_quant_config`, `prefix_cache_capacity`,
/// `prefix_cache_mem_bytes`. The function reads them by name.
///
/// On any error the partial state has already been freed via errdefer; the
/// caller decides how to surface (startup → recordLoadError + signal
/// started; on-demand → req.error_name + done broadcast).
fn doLoadOnInferenceThread(sch: *Scheduler, params: anytype) !void {
    var streaming_resident_bytes: ?u64 = null;
    var expert_source_assigned = false;
    errdefer if (expert_source_assigned) {
        sch.allocator.free(params.config.expert_source_dir.?);
        params.config.expert_source_dir = null;
    };
    if (params.config.isGlm5()) {
        const kv = transformer_mod.KvCacheChoice.resolve(params.config.kv_quant_override, params.kv_quant_config, params.kv_quant_explicit);
        if (kv.config.glmLatentBits() == null) {
            log.err("[glm] native MLA cache supports --kv-quant 8 or off\n", .{});
            return error.GlmKvQuantUnsupported;
        }
    }
    const streaming_budget = resolveSsdBudget(params.ssd_budget_bytes, params.config.ssd_budget_gb_override, params.config.supportsExpertStreaming());
    const streamed = expert_stream_mod.expertStreamingEngaged(
        params.config.supportsExpertStreaming(),
        params.config.expertStreamingRequired(),
        params.expert_cache_bytes,
        streaming_budget.bytes,
    );
    const vision_choice = visionChoiceFor(params.config, streamed);
    var load_vision = params.load_vision and vision_choice.on;
    var vision_cost: VisionCost = .{};
    if (streamed) {
        const budget = streaming_budget;
        if (params.expert_cache_bytes == 0 and budget.bytes == 0) return error.ExpertStreamingRequired;
        const mtp = mtpChoiceFor(params.mtp_enabled, params.mtp_explicit, params.config);
        if (mtp.source == .fast) log.info("[mtp] off: unsupported under streaming (--fast)\n", .{});
        const verdict = mtpStreamingVerdict(mtp, streamedHeadSupportedAt(sch.io, sch.allocator, params.config, params.model_dir));
        switch (verdict) {
            .refuse => {
                log.err("[expert-stream] {s}; MTP is on ({s}), pass --no-mtp\n", .{ expert_stream_mod.MTP_UNSUPPORTED, mtp.sourceName() });
                return error.ExpertStreamingMtpUnsupported;
            },
            .keep => {
                log.info("[expert-stream] MTP head resident ({s}); routed experts stream\n", .{mtp.sourceName()});
                params.config.stream_mtp_head = true;
            },
            .drop_settings => {
                log.info("[expert-stream] model-settings mtp=true ignored: {s}\n", .{expert_stream_mod.MTP_UNSUPPORTED});
                params.config.mtp_override = false;
            },
            .drop_default, .off => {},
        }
        if (budget.from_setting)
            log.info("[expert-stream] ssd budget {d} GiB from model-settings.json\n", .{budget.bytes >> 30});
        const plan = try planExpertStreaming(sch.io, sch.allocator, params.config, params.model_dir, params.expert_cache_bytes, budget.bytes, load_vision);
        vision_cost = plan.vision;
        load_vision = plan.vision.on;
        if (plan.resolved.overridden)
            log.info("[expert-stream] --expert-cache-gb overrides --ssd-budget-gb: cache {d:.2} GB\n", .{
                @as(f64, @floatFromInt(plan.resolved.cache_bytes)) / 1e9,
            });
        if (plan.resolved.ledger) |led| log.info("[expert-stream] ssd budget {d} GiB: trunk {d:.2} GB, mtp {d:.2} GB, vision {d:.2} GB, workspace {d:.2} GB, selected {d:.2} GB, bounce {d:.2} GB -> expert cache {d:.2} GB = {d} slots/layer\n", .{
            led.budget_bytes >> 30,
            @as(f64, @floatFromInt(led.trunk_bytes)) / 1e9,
            @as(f64, @floatFromInt(led.mtp_bytes)) / 1e9,
            @as(f64, @floatFromInt(led.vision_bytes)) / 1e9,
            @as(f64, @floatFromInt(led.workspace_bytes)) / 1e9,
            @as(f64, @floatFromInt(led.selected_bytes)) / 1e9,
            @as(f64, @floatFromInt(led.bounce_bytes)) / 1e9,
            @as(f64, @floatFromInt(led.cache_bytes)) / 1e9,
            plan.cache.slots_per_layer,
        });
        applyStreamingPlan(params.config, plan, budget.bytes);
        if (params.config.expert_source_dir == null) {
            params.config.expert_source_dir = try sch.allocator.dupe(u8, params.model_dir);
            expert_source_assigned = true;
        }
        streaming_resident_bytes = plan.split.trunk +| plan.split.mtp +| plan.split.vision;
        if (params.expert_cache_fit_resolver) |fit| try fit(params.config, streaming_resident_bytes.?);
    } else if (params.config.isGlm5()) {
        streaming_resident_bytes = try @import("glm5_diagnostic.zig").residentBytesWithVision(sch.io, sch.allocator, params.model_dir, params.config.num_hidden_layers, load_vision and params.config.glm5_vision);
    } else if (model_mod.usesSushiQuantMemoryBill(params.config)) {
        streaming_resident_bytes = try sushiResidentLoadBytes(sch.io, sch.allocator, params.model_dir, params.config, load_vision, mtpChoiceFor(params.mtp_enabled, params.mtp_explicit, params.config).on);
    } else if (params.config.usesMimoSourceTrunk()) {
        streaming_resident_bytes = try mimoResidentLoadBytes(sch.io, sch.allocator, params.model_dir, params.config, load_vision, mtpChoiceFor(params.mtp_enabled, params.mtp_explicit, params.config).on);
    }
    if (params.config.has_vision) {
        var line_buf: [256]u8 = undefined;
        log.info("{s}\n", .{visionLoadLine(&line_buf, vision_choice, streamed, vision_cost)});
    } else if (model_settings.vision_flag == true) log.warn("[vision] --vision ignored: this checkpoint has no vision tower\n", .{});

    // Resolve the sidecar before preflight so billing and loading see the same dependency.
    const drafter = LoadDrafterDir.resolve(sch.io, sch.allocator, params.no_drafter or try glmStreamedDrafterOff(params.config, params.drafter_dir), params.drafter_dir, params.model_dir);
    defer drafter.deinit(sch.allocator);
    var drafter_dir = drafter.dir;
    var prepared_drafter_path: ?[]u8 = null;
    defer if (prepared_drafter_path) |path| sch.allocator.free(path);
    if (drafter.owned != null) log.info("[dflash] auto-detected assistant: {s}\n", .{drafter_dir});
    if (model_mod.usesSushiQuantMemoryBill(params.config) and drafter_dir.len > 0)
        streaming_resident_bytes = streaming_resident_bytes.? +| try sushiAssistantLoadBytes(sch.io, sch.allocator, drafter_dir);
    if (params.config.isGlm5()) {
        const assistant_bytes = try glmDflashLoadBytes(sch.io, sch.allocator, params.config, params.model_dir, drafter_dir);
        streaming_resident_bytes = streaming_resident_bytes.? +| assistant_bytes;
        if (assistant_bytes > 0) log.info("[glm-dflash] resident assistant {d:.3} GiB; window {d} MiB; BF16 context; verification scratch billed per request\n", .{ @as(f64, @floatFromInt(assistant_bytes)) / (1024 * 1024 * 1024), params.config.glm_dflash_window_bytes >> 20 });
        if (drafter_dir.len == 0 and !params.no_drafter and !params.config.expert_streaming)
            log.warn("[glm-dflash] off: no assistant found (expected {s}/{s}, or an intact {s}/{s} cache); re-run `sushi pull` to fetch it\n", .{ params.model_dir, dflash_mod.SHIPPED_GLM_SUBDIR, params.model_dir, dflash_mod.DFLASH2_IN_DIR_SUBDIR });
    }

    // GPU-memory pre-flight (MLX path). A Metal OOM during weight load / warmup
    // is thrown by MLX as a C++ exception that can't be caught across the C ABI,
    // so it terminates the whole process. Refuse the load up front instead, with
    // an actionable error, when free RAM clearly can't hold the weights + warmup
    // headroom — catches the common "restarted before the prior server released
    // its memory" case. Bypass with --skip-mem-preflight.
    if (!skip_mem_preflight) {
        const weights_bytes = streaming_resident_bytes orelse plainWeightBytes(sch.io, sch.allocator, params.config, params.model_dir, mtpChoiceFor(params.mtp_enabled, params.mtp_explicit, params.config).on);
        const gpu_limit = mlx.maxRecommendedWorkingSet();
        const Reader = struct {
            io: std.Io,
            gpu_limit: usize,
            sushi: bool,
            fn read(r: *@This()) u64 {
                if (!r.sushi) return effectiveAvailableBytes(status.getAvailableMemBytes(), status.getProcAvailableMemBytes(), r.gpu_limit);
                var active: usize = 0;
                _ = mlx.mlx_get_active_memory(&active);
                return effectiveSushiAvailableBytes(status.getAvailableMemBytes(), status.getProcAvailableMemBytes(), r.gpu_limit, active);
            }
            fn sleep(r: *@This()) void {
                std.Io.sleep(r.io, .fromMilliseconds(500), .real) catch {};
            }
        };
        var reader = Reader{ .io = sch.io, .gpu_limit = gpu_limit, .sushi = model_mod.usesSushiQuantMemoryBill(params.config) };
        var avail_bytes = reader.read();
        const mtp_on = mtpChoiceFor(params.mtp_enabled, params.mtp_explicit, params.config).on;
        var serving = if (load_serving_bill) |bill| bill(params.config, avail_bytes -| weights_bytes -| LOAD_SAFETY_NET_BYTES) else null;
        var needed = loadRequirementBytes(params.config, weights_bytes, mtp_on, if (serving) |b| b.needed else 0);
        if (memInsufficientForLoad(weights_bytes, avail_bytes, needed)) {
            log.info("[preflight] {d:.2} GB available of {d:.2} GB needed; waiting while memory is still being released\n", .{
                @as(f64, @floatFromInt(avail_bytes)) / (1024.0 * 1024.0 * 1024.0),
                @as(f64, @floatFromInt(needed)) / (1024.0 * 1024.0 * 1024.0),
            });
            avail_bytes = settleAvailableBytes(avail_bytes, needed, &reader);
            serving = if (load_serving_bill) |bill| bill(params.config, avail_bytes -| weights_bytes -| LOAD_SAFETY_NET_BYTES) else null;
            needed = loadRequirementBytes(params.config, weights_bytes, mtp_on, if (serving) |b| b.needed else 0);
        }
        var active: usize = 0;
        if (reader.sushi) _ = mlx.mlx_get_active_memory(&active);
        log.info("[preflight] weights ~{d:.2} GB, needs ~{d:.2} GB, available {d:.2} GB (warmup/context ~{d:.2} GB + {d:.2} GB net)\n", .{
            @as(f64, @floatFromInt(weights_bytes)) / (1024.0 * 1024.0 * 1024.0),
            @as(f64, @floatFromInt(needed)) / (1024.0 * 1024.0 * 1024.0),
            @as(f64, @floatFromInt(avail_bytes)) / (1024.0 * 1024.0 * 1024.0),
            @as(f64, @floatFromInt(needed -| weights_bytes -| LOAD_SAFETY_NET_BYTES)) / (1024.0 * 1024.0 * 1024.0),
            @as(f64, @floatFromInt(LOAD_SAFETY_NET_BYTES)) / (1024.0 * 1024.0 * 1024.0),
        });
        const gpu_exhausted = model_mod.usesSushiQuantMemoryBill(params.config) and gpu_limit > 0 and active >= gpu_limit;
        if (weights_bytes > 0 and (memInsufficientForLoad(weights_bytes, avail_bytes, needed) or gpu_exhausted)) {
            const gb = 1024.0 * 1024.0 * 1024.0;
            if (serving) |b| {
                const maximum: u32 = if (avail_bytes < loadRequirementBytes(params.config, weights_bytes, mtp_on, 0)) 0 else b.max_context;
                sch.registry.mutex.lockUncancelable(sch.io);
                const refusal = model_registry_mod.MemoryContextRefusal{ .requested = b.requested_context, .maximum = maximum, .needed = needed, .available = avail_bytes, .kv_bits = b.kv_bits, .chunk = b.chunk };
                params.entry.memory_context_refusal = refusal;
                if (@TypeOf(params) == *LoadRequest) params.memory_context_refusal = refusal;
                sch.registry.mutex.unlock(sch.io);
                log.err("Insufficient memory for requested context: {d} tokens at KV{d} and prefill chunk {d} need ~{d:.2} GiB including {d:.2} GiB of enabled model/vision/MTP weights, but only {d:.2} GiB is available. Maximum context with this memory budget: {d} tokens. Lower --ctx-size or free memory; --skip-mem-preflight overrides this check.\n", .{
                    b.requested_context,                         b.kv_bits,                                 b.chunk,                                                                              @as(f64, @floatFromInt(needed)) / gb,
                    @as(f64, @floatFromInt(weights_bytes)) / gb, @as(f64, @floatFromInt(avail_bytes)) / gb, maximum,
                });
                return error.InsufficientMemory;
            }
            log.err("Insufficient memory to load model: needs ~{d:.1} GB free ({d:.1} GB of weights plus headroom for warmup buffers and a baseline KV cache) but only {d:.1} GB is available (free RAM, capped at the GPU working-set limit that iogpu.wired_limit_mb sets). Close other models/apps (or wait for a prior sushi to fully exit) and retry; pass --skip-mem-preflight to override.\n", .{
                @as(f64, @floatFromInt(needed)) / gb,
                @as(f64, @floatFromInt(weights_bytes)) / gb,
                @as(f64, @floatFromInt(avail_bytes)) / gb,
            });
            return error.InsufficientMemory;
        }
    }

    if (params.config.isGlm5() and @import("glm5_dflash_cache.zig").isShippedSource(drafter_dir, params.model_dir)) {
        const prepared = try @import("glm5_dflash_cache.zig").prepare(sch.io, sch.allocator, drafter_dir, mlx.gpuStream());
        prepared_drafter_path = prepared.path;
        if (prepared.fallback) {
            const cached_bytes = try @import("glm5_dflash_cache.zig").plannedResidentBytes(sch.io, sch.allocator, drafter_dir);
            const bf16_bytes = try @import("glm5_diagnostic.zig").assistantResidentBytes(sch.io, sch.allocator, prepared.path);
            streaming_resident_bytes = streaming_resident_bytes.? -| cached_bytes +| bf16_bytes;
            if (!skip_mem_preflight) {
                const available = effectiveAvailableBytes(status.getAvailableMemBytes(), status.getProcAvailableMemBytes(), mlx.maxRecommendedWorkingSet());
                const needed = loadRequirementBytes(params.config, streaming_resident_bytes.?, false, 0);
                log.info("[preflight] BF16 DFlash2 fallback: weights ~{d:.2} GiB, needs ~{d:.2} GiB, available {d:.2} GiB\n", .{ @as(f64, @floatFromInt(streaming_resident_bytes.?)) / (1 << 30), @as(f64, @floatFromInt(needed)) / (1 << 30), @as(f64, @floatFromInt(available)) / (1 << 30) });
                if (available > 0 and needed > available) return error.InsufficientMemory;
            }
        }
        drafter_dir = prepared.path;
        _ = mlx.mlx_clear_cache();
    }

    // Allocate the drafter_path dupe up front so the post-publish step
    // (lower down) has no fallible operations — once we start assigning
    // pointers onto `params.entry`, an OOM during a dupe would leave the
    // entry holding pointers that the per-ptr errdefers would double-free.
    var drafter_path_owned: []u8 = &[_]u8{};
    errdefer if (drafter_path_owned.len > 0) sch.allocator.free(drafter_path_owned);
    if (params.drafter_dir.len > 0) {
        drafter_path_owned = try sch.allocator.dupe(u8, drafter_dir);
    }

    // Weights — first mlx call. Binds the stream on this thread.
    const weights_ptr = try sch.allocator.create(Weights);
    errdefer sch.allocator.destroy(weights_ptr);
    var load_config = params.config.*;
    // The resolved launch choice gates lazy native-head tensors before Transformer.init binds them.
    load_config.mtp_override = mtpChoiceFor(params.mtp_enabled, params.mtp_explicit, params.config).on;
    weights_ptr.* = try model_mod.loadWeightsForConfig(sch.io, sch.allocator, params.model_dir, &load_config, load_vision);
    errdefer weights_ptr.deinit();
    model_mod.resolveWeightPrefix(params.config, weights_ptr);

    // Transformer — owns the bulk of the GPU memory.
    const xfm_ptr = try sch.allocator.create(Transformer);
    errdefer sch.allocator.destroy(xfm_ptr);
    xfm_ptr.* = try Transformer.init(sch.io, sch.allocator, params.config.*, weights_ptr);
    errdefer xfm_ptr.deinit();

    // Reserved-token suppression mask (never sample `<|fim_hole|>`-class
    // specials): derived per model from tokenizer + template + eos.
    generate_mod.installSuppressMask(xfm_ptr, params.tok, params.chat_config.chat_template, params.config.eosTokenSlice());
    generate_mod.installThinkMarkers(xfm_ptr, params.tok);
    try generate_mod.installLogitBias(sch.io, xfm_ptr, params.tok);

    // Propagate the kv-quant config to the Transformer's own cache. Slot
    // caches in serve mode honor this independently in `Slot.init`; this
    // call covers any path that still touches `xfm.cache` directly (legacy
    // single-slot fallbacks, prompt-cache reuse).
    // An explicit launch flag outranks the per-model settings stamped on the config at BOTH construction sites.
    const kv_cache = transformer_mod.KvCacheChoice.resolve(params.config.kv_quant_override, params.kv_quant_config, params.kv_quant_explicit);
    const load_ctx = model_settings.contextPick(sch.ctx_size_flag, params.config.ctx_override);
    var ctx_buf: [16]u8 = undefined;
    log.info("[kv-cache] {s} ({s}); ctx {s} ({s})\n", .{
        kv_cache.label(),                                      kv_cache.sourceName(),
        model_settings.contextLabel(&ctx_buf, load_ctx.value), model_settings.sourceLabel(load_ctx.source, "--ctx-size"),
    });
    const kv_quant_config = kv_cache.config;
    const mtp = mtpChoiceFor(params.mtp_enabled, params.mtp_explicit, params.config);
    const mtp_streaming_off = mtpDefaultOffUnderStreaming(mtp, params.config.expert_streaming, params.config.stream_mtp_head);
    const acceptance = generate_mod.mtpAcceptanceFor(params.config.mtp_acceptance_override);
    const greedy_tail = generate_mod.mtpGreedyTailFor(params.config.mtp_greedy_tail_override);
    // GLM has no MTP head; the server's `MTP: off` line already says so.
    if (!params.config.isGlm5()) log.info("[mtp] {s} ({s}{s}); acceptance {s} ({s}); greedy tail {s} ({s})\n", .{
        if (mtp_streaming_off) "off" else mtp.label(), if (mtp_streaming_off) "streaming; " else "", mtp.sourceName(),
        mtp_acceptance_mod.name(acceptance.value),     model_settings.sourceLabel(acceptance.source, model_settings.acceptanceFlagName(acceptance.value)),
        if (greedy_tail.value) "on" else "off",        model_settings.sourceLabel(greedy_tail.source, "--mtp-greedy-tail"),
    });
    const mtp_enabled = mtp.on and !mtp_streaming_off and !params.config.isGlm5();
    if (params.config.isGlm5()) {
        const latent_bits: u64 = if (kv_quant_config.isQuant()) kv_quant_config.bits else 16;
        log.info("[glm] native {s} MLA: {d} latent + {d} pooled-index bytes/token; serial decode\n", .{ if (kv_quant_config.isQuant()) "kv8" else "BF16", @import("server.zig").kvBytesPerTokenAtBits(params.config.kvBytesPerToken(), latent_bits), params.config.qsaHistoryBytesPerToken() });
        log.info("[glm] NAX arms {s}\n", .{if (@import("glm5_model.zig").naxArms()) "on: packed sparse and B1/B3 attention, NAX index scores, KDA cluster, A6 dense-once, MLA head batches" else "off: FP32 composite sparse and B1/B3 attention, scalar index scores, staged KDA cluster, affine QMM trunk"});
        if (mtp.on) log.warn("[glm] MTP head is not integrated with serving; MTP off\n", .{});
        if (params.prefix_cache_capacity > 0 and params.ssm_checkpoint_stride > 0) log.info("[glm] prefix cache: KDA checkpoints at every prefill chunk and the prompt end (<= {d} per entry, {d} MiB each)\n", .{ @import("glm5_prefix.zig").checkpointMax(params.ssm_checkpoint_max), params.config.ssmCheckpointBytes() >> 20 });
    }
    if (std.mem.indexOf(u8, params.chat_config.chat_template, "preserve_thinking") != null) {
        const keep = model_settings.pick(bool, model_settings.preserve_thinking_flag, params.config.preserve_thinking_override, true);
        log.info("[chat] preserve_thinking {s} ({s})\n", .{ if (keep.value) "on" else "off", model_settings.sourceLabel(keep.source, "--preserve-thinking") });
    }
    const think = model_settings.pick(f32, model_settings.think_penalty_flag, params.config.think_penalty_override, 0);
    if (think.value > 0) log.info("[think-penalty] lambda {d} ({s}); a request's think_penalty outranks it\n", .{ think.value, model_settings.sourceLabel(think.source, "--think-penalty") });
    if (kv_quant_config.scheme != .off) {
        try xfm_ptr.cache.reinit(params.config.num_hidden_layers, kv_quant_config);
    }
    Transformer.mtp_head_kv_quant_flag = params.mtp_head_kv_quant;
    try xfm_ptr.qwen4MtpApplyKvQuant(kv_quant_config);

    // Wire model weights into GPU memory (prevents paging, matches mlx-lm).
    // Policy in mlx.applyWiredPolicy; re-applied in runLoadRequest /
    // runUnloadRequest so `fit` capacity tracks the live set across
    // load/unload churn (this early call covers warmup's forwards).
    logWiredPolicy(mlx.applyWiredPolicy());

    // JIT-compile activation kernels. These are bound to THIS thread's mlx
    // stream — that's exactly the point of doing them here. Skipped entirely
    // when there's no GPU backend (iOS Simulator's CPU-only MLX): mlx_compile
    // is Metal kernel fusion and requests a GPU stream that doesn't exist; the
    // runtime falls back to the uncompiled paths when compiled_* stays null.
    if (!mlx.noGpuBackend()) {
        if (params.config.hidden_act == .gelu_approx) {
            xfm_ptr.compileGelu();
            xfm_ptr.compileGeglu();
        }
        if (xfm_ptr.moe_layers != null) {
            xfm_ptr.compileMoeRouting();
        }
        if (params.config.linear_num_key_heads > 0) {
            xfm_ptr.compileGdnGate();
        }
        if (params.config.isQwen4()) {
            xfm_ptr.compileQwen4Hc();
        }
    }

    // Phase 2 experiment: opt-in full-forward Metal fusion via
    // SUSHI_COMPILE_FORWARD=1. This wraps the entire forward pass in
    // mlx_compile so the chunked-prefill loop dispatches a fused graph
    // instead of ~hundreds of separate ops per chunk. Gated because
    // (a) the compiled closure captures `xfm.cache` / `xfm.ssm_entries`
    // as state, and any path that swaps those (multi-slot scheduler,
    // future re-entrant callers) must verify they're not racing the
    // compiled call; (b) mlx_compile with shapeless=false recompiles
    // per unique input shape, which thrashes if the prefill loop sees
    // many different chunk sizes.
    if (std.c.getenv("SUSHI_COMPILE_FORWARD") != null) {
        const raw = std.c.getenv("SUSHI_COMPILE_FORWARD").?;
        const slice = std.mem.sliceTo(raw, 0);
        if (std.mem.eql(u8, slice, "1")) {
            xfm_ptr.compileForward();
        }
    }

    if (transformer_mod.diagEnvOn("SUSHI_PREFILL_UBENCH")) prefillUbench(sch.allocator, xfm_ptr, params.config, params.tok);
    if (transformer_mod.diagEnvOn("SUSHI_GLM_ROWS_UBENCH")) if (xfm_ptr.glm5) |target| glmRowsUbench(sch.allocator, target, kv_quant_config.glmLatentBits() orelse 0, params.tok);

    // DIAGNOSTIC (SUSHI_DECODE_FWD_UBENCH=N): time N decode-width forward
    // passes back to back, with NO sampling, detokenization, stop-checking or
    // cache bookkeeping around them. The server reports `predicted_ms` around
    // the whole decode LOOP, so this is the only way to say how much of a
    // token is the model and how much is everything else. Resets the KV cache
    // afterwards so the probe cannot pollute real requests.
    if (std.c.getenv("SUSHI_DECODE_FWD_UBENCH")) |raw| {
        const n = std.fmt.parseInt(usize, std.mem.sliceTo(raw, 0), 10) catch 0;
        if (n > 0) {
            const io_u = @import("io_util.zig");
            const tio = std.Io.Threaded.global_single_threaded.io();
            var ctx = xfm_ptr.defaultCtx();
            // SUSHI_DECODE_FWD_UBENCH_S=<rows>[,<rows>...]: verify-width forwards
            // (per-position SSM capture on, as spec verify runs them), one pass per width.
            // SUSHI_DECODE_FWD_UBENCH_KV=<tokens>: prefill that many
            // tokens first so the meter runs at a real context length.
            // SUSHI_DECODE_FWD_UBENCH_PROFILE=1: one more pass per width under the
            // sub-block profiler (`[decode-prof]`).
            var widths: [16]usize = @splat(1);
            var n_widths: usize = 1;
            if (std.c.getenv("SUSHI_DECODE_FWD_UBENCH_S")) |r| {
                n_widths = 0;
                var it = std.mem.tokenizeScalar(u8, std.mem.sliceTo(r, 0), ',');
                while (it.next()) |w| {
                    if (n_widths == widths.len) break;
                    widths[n_widths] = @max(1, std.fmt.parseInt(usize, w, 10) catch 1);
                    n_widths += 1;
                }
                n_widths = @max(n_widths, 1);
            }
            const profile_pass = std.c.getenv("SUSHI_DECODE_FWD_UBENCH_PROFILE") != null;
            const kv_pre: usize = blk: {
                const r = std.c.getenv("SUSHI_DECODE_FWD_UBENCH_KV") orelse break :blk 0;
                break :blk std.fmt.parseInt(usize, std.mem.sliceTo(r, 0), 10) catch 0;
            };
            if (kv_pre > 0) {
                var done_pre: usize = 0;
                const pre_buf = try sch.allocator.alloc(i32, 2048);
                defer sch.allocator.free(pre_buf);
                for (pre_buf, 0..) |*v, i| v.* = @intCast(1 + (i % 1000));
                while (done_pre < kv_pre) {
                    const n_chunk = @min(2048, kv_pre - done_pre);
                    const psh = [_]c_int{ 1, @intCast(n_chunk) };
                    const ti = mlx.mlx_array_new_data(pre_buf.ptr, &psh, 2, .int32);
                    defer _ = mlx.mlx_array_free(ti);
                    const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                    _ = mlx.mlx_array_eval(lg);
                    _ = mlx.mlx_array_free(lg);
                    done_pre += n_chunk;
                }
                log.info("[fwd-ubench] prefilled {d} tokens\n", .{done_pre});
            }
            // SUSHI_DECODE_FWD_UBENCH_ROW_ARMS=1: a MiMo verify width runs twice, the
            // prefill-shaped forward first, then the verify rows (decode arithmetic per row).
            const row_arms = std.c.getenv("SUSHI_DECODE_FWD_UBENCH_ROW_ARMS") != null;
            // SUSHI_DECODE_FWD_UBENCH_GDN_ARMS=1: every width runs the GDN chain and the fused
            // decode step (`gdn_decode.step`) as off, on, on, off passes in this one process.
            const gdn_arms = std.c.getenv("SUSHI_DECODE_FWD_UBENCH_GDN_ARMS") != null;
            defer transformer_mod.gdn_decode_recur_override = null;
            const fold_arms = transformer_mod.diagEnvOn("SUSHI_DECODE_FWD_UBENCH_GDN_FOLD_ARMS");
            defer transformer_mod.gdn_verify_fold_override = null;
            // SUSHI_DECODE_FWD_UBENCH_QSA_POOL_ARMS=1: each width runs A B B A, A the composed QSA
            // pooled-key chain and B the fused kernel; each arm logs its fused launches.
            const pool_arms = transformer_mod.diagEnvOn("SUSHI_DECODE_FWD_UBENCH_QSA_POOL_ARMS");
            defer qwen4_qsa.qsa_pool_rope_fused_override = null;
            // SUSHI_DECODE_FWD_UBENCH_QKV_PREP_ARMS=1: MiMo's composed rope + kv8 quantize (off) and
            // the one-dispatch prep (on) as off, on, on, off passes.
            const qkv_arms = transformer_mod.diagEnvOn("SUSHI_DECODE_FWD_UBENCH_QKV_PREP_ARMS");
            defer mimo_forward.mimo_qkv_prep_override = null;
            // SUSHI_DECODE_FWD_UBENCH_LMHEAD_ARMS=1: argmax-only forwards on the full lm_head (off) and
            // the coarse shortlist (on) as off, on, on, off passes; the heads load later, so it builds
            // its own coarse copy for the meter.
            const lm_arms = transformer_mod.diagEnvOn("SUSHI_DECODE_FWD_UBENCH_LMHEAD_ARMS");
            // SUSHI_DECODE_FWD_UBENCH_GLOBAL_ROWS_ARMS=1: MiMo global verify rows one dispatch each (off)
            // and in row groups sharing one page walk each (on) as off, on, on, off passes.
            const rows_arms = transformer_mod.diagEnvOn("SUSHI_DECODE_FWD_UBENCH_GLOBAL_ROWS_ARMS");
            defer mimo_forward.mimo_global_rows_override = null;
            const lm_built = lm_arms and xfm_ptr.lm_head_coarse == null;
            if (lm_built) xfm_ptr.lm_head_coarse = mtp_mod.buildRerankCoarse(mlx.gpuStream(), xfm_ptr, mimo_mtp.rerankBits());
            defer if (lm_arms) {
                if (lm_built) if (xfm_ptr.lm_head_coarse) |*c| {
                    c.deinit();
                    xfm_ptr.lm_head_coarse = null;
                };
                transformer_mod.lmhead_shortlist_override = null;
            };
            var arm_buf: [8]UbenchArm = undefined;
            for (widths[0..n_widths]) |rows| {
            for (ubenchArms(&arm_buf, rows, xfm_ptr.config.isMimo(), row_arms, pool_arms)) |arm| {
            const abba: []const ?bool = &.{ false, true, true, false };
            const abba2: []const ?bool = &.{ false, true, true, false, false, true, true, false };
            const no_arms: []const ?bool = &.{null};
            for (if (qkv_arms) abba2 else if (gdn_arms or fold_arms or lm_arms or rows_arms) abba else no_arms) |gdn_arm| {
            transformer_mod.gdn_decode_recur_override = if (fold_arms) true else if (qkv_arms or lm_arms or rows_arms) null else gdn_arm;
            mimo_forward.mimo_global_rows_override = if (rows_arms) gdn_arm else null;
            if (lm_arms) transformer_mod.lmhead_shortlist_override = gdn_arm;
            transformer_mod.gdn_verify_fold_override = if (fold_arms) gdn_arm else null;
            mimo_forward.mimo_qkv_prep_override = if (qkv_arms) gdn_arm else null;
            transformer_mod.gdn_verify_fold_calls = 0;
            if (gdn_arm) |on| log.info("[fwd-ubench] {s} arm: {s}\n", .{ if (rows_arms) "global rows" else if (lm_arms) "lm_head shortlist" else if (qkv_arms) "qkv prep" else if (fold_arms) "gdn fold" else "gdn recur", if (on) "on" else "off" });
            const tok_slice = try sch.allocator.alloc(i32, @min(rows, 4096));
            defer sch.allocator.free(tok_slice);
            for (tok_slice, 0..) |*v, i| v.* = @intCast(1 + (i % 997));
            const tok = tok_slice.ptr;
            const tsh = [_]c_int{ 1, @intCast(tok_slice.len) };
            ctx.capture_ssm_seq = rows > 1 and rows <= 16 and ctx.ssm_entries != null; // verify widths capture, prefill chunks do not
            arm.apply(&ctx);
            if (lm_arms) ctx.argmax_only = true;
            log.info("[fwd-ubench] rows={d} capture={} verify_rows={} qsa_pool={s}\n", .{ tok_slice.len, ctx.capture_ssm_seq, arm.verify_rows, arm.poolName() });
            // Warm: first forward pays kernel JIT + lazy weight materialization.
            for (0..3) |_| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                _ = mlx.mlx_array_eval(lg);
                _ = mlx.mlx_array_free(lg);
            }
            // Split CPU graph CONSTRUCTION from GPU execution. MLX is lazy, so
            // `forwardWith` only issues ops — if that half dominates, the token
            // is bounded by op count / FFI overhead, not by memory bandwidth,
            // and no kernel-level optimization can reach it.
            var sw = io_u.Stopwatch.init(tio);
            var build_ns: u64 = 0;
            var eval_ns: u64 = 0;
            var ops_total: u64 = 0;
            var done: usize = 0;
            for (0..n) |_| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const ops_before = mlx.op_count.load(.monotonic);
                var swb = io_u.Stopwatch.init(tio);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                build_ns += swb.read();
                ops_total += mlx.op_count.load(.monotonic) - ops_before;
                var swe = io_u.Stopwatch.init(tio);
                _ = mlx.mlx_array_eval(lg);
                eval_ns += swe.read();
                _ = mlx.mlx_array_free(lg);
                done += 1;
            }
            if (fold_arms) log.info("[fwd-ubench] gdn fold launches: {d}\n", .{transformer_mod.gdn_verify_fold_calls});
            const dn: f64 = @floatFromInt(@max(done, 1));
            const ms = @as(f64, @floatFromInt(sw.read())) / 1.0e6 / dn;
            log.info("[fwd-ubench] {d} decode forwards, eval-per-step: {d:.3} ms/forward (build {d:.3} ms CPU + eval {d:.3} ms GPU, {d:.0} ops/forward)\n", .{
                done,
                ms,
                @as(f64, @floatFromInt(build_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(eval_ns)) / 1.0e6 / dn,
                @as(f64, @floatFromInt(ops_total)) / dn,
            });
            // SUSHI_DECODE_FWD_GRAPH=<abs path>: one forward's unevaluated tape, one primitive per
            // line; a primitive is at most one dispatch, so a histogram of it is the dispatch plan.
            if (std.c.getenv("SUSHI_DECODE_FWD_GRAPH")) |path| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const lg = try xfm_ptr.forwardWith(&ctx, ti);
                defer _ = mlx.mlx_array_free(lg);
                if (std.c.fopen(path, "w")) |f| {
                    const outs = mlx.mlx_vector_array_new_value(lg);
                    defer _ = mlx.mlx_vector_array_free(outs);
                    const namer = mlx.mlx_node_namer_new();
                    defer _ = mlx.mlx_node_namer_free(namer);
                    _ = mlx.mlx_print_graph(f, namer, outs);
                    _ = std.c.fclose(f);
                    log.info("[fwd-ubench] graph written to {s}\n", .{std.mem.sliceTo(path, 0)});
                }
                _ = mlx.mlx_array_eval(lg);
            }

            // Same forward with the vocab projection suppressed. lm_head is
            // terminal — nothing downstream depends on it — so dropping it
            // cannot change the work the rest of the graph does, which makes
            // this the one sound ablation in the probe.
            ctx.skip_lm_head = true;
            for (0..3) |_| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                _ = mlx.mlx_array_eval(lg);
                _ = mlx.mlx_array_free(lg);
            }
            var sw_nolm = io_u.Stopwatch.init(tio);
            var done_nolm: usize = 0;
            for (0..n) |_| {
                const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                defer _ = mlx.mlx_array_free(ti);
                const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                _ = mlx.mlx_array_eval(lg);
                _ = mlx.mlx_array_free(lg);
                done_nolm += 1;
            }
            const ms_nolm = @as(f64, @floatFromInt(sw_nolm.read())) / 1.0e6 / @as(f64, @floatFromInt(@max(done_nolm, 1)));
            ctx.skip_lm_head = false;
            log.info("[fwd-ubench] without lm_head: {d:.3} ms/forward  => lm_head = {d:.3} ms; qsa_pool={s} fused launches={d}\n", .{ ms_nolm, ms - ms_nolm, arm.poolName(), transformer_mod.qsa_pool_rope_dispatches });
            if (profile_pass) {
                transformer_mod.decodeProfileSession(@intCast(rows));
                for (0..n) |_| {
                    const ti = mlx.mlx_array_new_data(tok, &tsh, 2, .int32);
                    defer _ = mlx.mlx_array_free(ti);
                    const lg = xfm_ptr.forwardWith(&ctx, ti) catch break;
                    _ = mlx.mlx_array_eval(lg);
                    _ = mlx.mlx_array_free(lg);
                }
                transformer_mod.decodeProfileSession(0);
            }
            }
            }
            }
            log.info("[fwd-ubench] done\n", .{});
            xfm_ptr.resetCache() catch {};
        }
    }

    // Vision encoder if requested. `MissingVisionWeights` is a benign opt-out
    // (model declares vision in config but the safetensors didn't ship the
    // tower); other errors fail the whole load.
    var vision_ptr: ?*VisionEncoder = null;
    if (load_vision) {
        const v = try sch.allocator.create(VisionEncoder);
        if (VisionEncoder.init(sch.allocator, params.config.*, weights_ptr)) |encoder| {
            v.* = encoder;
            vision_ptr = v;
        } else |err| {
            sch.allocator.destroy(v);
            if (err == error.MissingVisionWeights) {
                log.warn("Vision weights missing — vision disabled (model may have been quantized without vision tower)\n", .{});
            } else {
                return err;
            }
        }
    }
    errdefer if (vision_ptr) |v| {
        v.deinit();
        sch.allocator.destroy(v);
    };

    // The measured round-cost table (`round_cost.zig`) is keyed per (chip,
    // model, quant, OS build); restored here, written at request end.
    {
        var quant_buf: [32]u8 = undefined;
        const quant = std.fmt.bufPrint(&quant_buf, "q{d}g{d}", .{
            params.config.quant_bits,
            params.config.quant_group_size,
        }) catch "q?";
        var os_buf: [64]u8 = undefined;
        const os_build = transformer_mod.macosProductVersion(&os_buf) orelse "";
        // The measured round-cost table rides the same identity: restored
        // here, written at the end of any request that folded new samples.
        // The bucket grid and store version are the arch's (only qwen4_exp gets the long
        // grid); every other arch keeps the `rc1` table 26.9.1 wrote and boots warm.
        const rc_layout: round_cost_mod.Layout = round_cost_mod.layoutFor(params.config);
        xfm_ptr.round_cost.layout = rc_layout;
        const rc_key = round_cost_mod.cacheKey(&xfm_ptr.round_cost_key_buf, chip_mod.chipBrand(), params.model_dir, quant, os_build, rc_layout, round_cost_mod.engineBuildId());
        xfm_ptr.round_cost_key_len = @intCast(rc_key.len);
        if (round_cost_mod.loadCached(sch.allocator, sch.io, rc_key, rc_layout)) |t| {
            xfm_ptr.round_cost = t;
            log.info("[spec-cost] round-cost table restored ({d} width cells, {d} serial cells)\n", .{ t.restored, t.restored_serial });
            if (t.restored_dropped > 0) log.info("[spec-cost] dropped {d} implausible persisted cell(s)\n", .{t.restored_dropped});
        }
    }

    // Assistant sidecar (optional). Loaded only when `drafter_dir` is
    // non-empty. The sidecar KIND is decided by its config CONTRACT: a
    // config declaring block_size + mask_token_id + target_layer_ids is a
    // DFlash block-drafter (any `*_assistant` family); anything else is refused.
    var dflash_ptr: ?*DflashModel = null;
    if (drafter_dir.len > 0 and dflash_mod.probeIsDflash(sch.io, sch.allocator, drafter_dir)) {
        const env_off = if (std.c.getenv("SUSHI_DFLASH")) |v| v[0] == '0' else false;
        if (env_off) {
            log.info("[dflash] sidecar at {s} skipped (SUSHI_DFLASH=0)\n", .{drafter_dir});
        } else {
            const d = try sch.allocator.create(DflashModel);
            d.* = (if (params.config.isGlm5()) @import("glm5_dflash.zig").loadAssistantStored(sch.io, sch.allocator, drafter_dir, xfm_ptr.glm5.?) else dflash_mod.loadDflash(sch.io, sch.allocator, mlx.gpuStream(), drafter_dir)) catch |err| {
                sch.allocator.destroy(d);
                log.err("Failed to load DFlash assistant at {s}: {s}\n", .{ drafter_dir, @errorName(err) });
                return err;
            };
            if (params.config.isGlm5()) {
                d.native_glm_serving = true;
                log.info("[glm-dflash] native DFlash2 loaded; greedy and sampled target decisions use layerwise tree verification\n", .{});
                if (transformer_mod.diagEnvOn("SUSHI_GLM_ROUND_UBENCH")) glmRoundUbench(sch.allocator, xfm_ptr.glm5.?, d, kv_quant_config.glmLatentBits() orelse 0, params.tok);
            } else d.bind(xfm_ptr) catch |err| {
                d.deinit();
                sch.allocator.destroy(d);
                log.err(
                    "DFlash assistant at {s} is incompatible with target: {s}\n" ++
                        "  (assistant+target must share hidden_size, the mask token and\n" ++
                        "  target_layer_ids must exist in the target, and the target must\n" ++
                        "  run the standard dense-attention forward path)\n",
                    .{ drafter_dir, @errorName(err) },
                );
                return err;
            };
            dflash_ptr = d;
            const wide_lane = dflash_mod.wideVerifyLaneAvailable();
            const block_cap = dflash_mod.blockCapForMachine(chip_mod.chipBrand());
            sch.drafter_block_size = dflash_mod.resolveBlockSize(
                d.config.block_size,
                params.draft_block_size,
                params.draft_block_size_explicit,
                wide_lane,
                block_cap.cap,
            );
            var cap_note_buf: [96]u8 = undefined;
            const cap_note: []const u8 = if (params.draft_block_size_explicit)
                ", user-clamped"
            else if (!wide_lane and d.config.block_size > sch.drafter_block_size)
                std.fmt.bufPrint(&cap_note_buf, ", capped ({s} cap {d})", .{
                    block_cap.label,
                    block_cap.cap,
                }) catch ", capped"
            else
                "";
            log.info("DFlash drafter ready (block_size={d}{s}, wide_verify_lane={}, targets={any}).\n", .{
                sch.drafter_block_size,
                cap_note,
                wide_lane,
                d.config.target_layer_ids,
            });
        }
    } else if (drafter_dir.len > 0) {
        log.err("Drafter at {s} is not a DFlash assistant (its config lacks block_size, mask_token_id and target_layer_ids)\n", .{drafter_dir});
        return error.UnsupportedDrafter;
    }
    errdefer if (dflash_ptr) |d| {
        d.deinit();
        sch.allocator.destroy(d);
    };

    // Qwen native MTP head (optional). Auto-loaded when the model dir ships
    // one — an `mtp/weights.safetensors`-class sidecar file OR in-checkpoint
    // `[language_model.]mtp.*` tensors in the trunk shards; a failed load or
    // bind only disables the head — the model still serves.
    var mtp_ptr: ?*mtp_mod.MtpModel = null;
    var mtp_cost_profile: mtp_mod.MtpCostProfile = .generic;
    // MiMo's trained heads, resident beside a resident trunk (streaming refuses MTP above).
    var mimo_head: ?*mimo_mtp.Head = null;
    if (mtp_enabled and params.config.isMimo() and !params.config.expert_streaming) {
        mimo_head = loadMimoHeads(sch, params.model_dir, params.config, xfm_ptr) catch |err| blk: {
            log.warn("[mimo-mtp] heads not loaded ({s}) — MTP off\n", .{@errorName(err)});
            break :blk null;
        };
    }
    errdefer if (mimo_head) |h| {
        h.deinit();
        sch.allocator.destroy(h);
    };
    // The heads built the coarse copy for their drafts; without them the trunk's greedy readout still takes one.
    // Same predicate as the resident bill that prices this copy.
    if (mimoCoarseHeadBytes(params.config) > 0 and xfm_ptr.lm_head_coarse == null)
        xfm_ptr.lm_head_coarse = mtp_mod.buildRerankCoarse(mlx.gpuStream(), xfm_ptr, mimo_mtp.rerankBits());
    if (mtp_enabled and !params.config.isMimo() and !params.config.isGlm5() and mtp_mod.hasMtpHead(sch.io, sch.allocator, params.model_dir)) {
        if (sch.allocator.create(mtp_mod.MtpModel)) |h| {
            if (mtp_mod.loadMtp(sch.io, sch.allocator, mlx.gpuStream(), params.model_dir)) |loaded| {
                h.* = loaded;
                if (h.bind(xfm_ptr)) {
                    mtp_ptr = h;
                    mtp_cost_profile = h.m5NaxCostProfile(xfm_ptr);
                    // Price the DRAFT side. The verify ladder above measures
                    // the trunk forward and nothing else, but an m-deep round
                    // is that forward PLUS m sequential head steps — which on
                    // a 27B dominate the per-position marginal. Fitting the
                    // EV surface without this under-prices depth ~9x (live
                    // 2026-08-21: forward marginal 0.8 ms/position against a
                    // hand-measured composite of 7.6). A cached curve already
                    // carries it, so this is paid once per (chip, model,
                    // quant, OS build) like the ladder itself.
                    log.info("MTP head ready (depth={d}, profile={s}).\n", .{
                        generate_mod.Generator.resolveMtpDepthCapForProfile(params.mtp_depth, mtp_cost_profile),
                        @tagName(mtp_cost_profile),
                    });
                } else |bind_err| {
                    log.warn("MTP sidecar incompatible with target ({s}) — disabled.\n", .{@errorName(bind_err)});
                    h.deinit();
                    sch.allocator.destroy(h);
                }
            } else |load_err| {
                log.warn("Failed to load MTP sidecar: {s} — disabled.\n", .{@errorName(load_err)});
                sch.allocator.destroy(h);
            }
        } else |_| {}
    } else if (mtp_enabled) {
        // A quiet fallback to mode=pld cost a tester a day: nothing logged
        // when the probe finds no head. Debug-level — most checkpoints have
        // no MTP head and an info line per load would be noise.
        log.debug(
            "[mtp] no head found: no mtp/ sidecar and no [language_model.]mtp.* " ++
                "keys resolvable from the index at {s} — MTP off\n",
            .{params.model_dir},
        );
    }
    errdefer if (mtp_ptr) |h| {
        h.deinit();
        sch.allocator.destroy(h);
    };

    // The qwen4_exp in-checkpoint head's twin of the sidecar's own coarse
    // rerank build (inside `bind`, above): a `requantizeRows` of the whole
    // trunk lm_head plus a synchronous eval of ~240 MB. Lazily it ran inside
    // the FIRST request's draft chain — on this thread, mid-round, with the
    // stream drained — i.e. first-token latency for whoever loaded the model.
    // Gated exactly like `entry.mtp`'s `.qwen4` arm below: `--no-mtp` never
    // drafts, so it never pays. One-shot, so the draft path's ask stays a
    // pure read; if this ever does not run, that ask still builds.
    if (mtp_ptr == null and mtp_enabled) _ = xfm_ptr.qwen4BuildDraftRerank();

    if (xfm_ptr.expert_stream) |engine| {
        const slots = engine.warmSlotsPerLayer();
        const layers = expert_stream_mod.moeLayerCount(engine.geometry);
        const bytes = @as(u64, slots) * layers * engine.plan.expert_bytes;
        log.info("[expert-stream] cache warm: preloading {d:.3} GB (80% of cache slots, lowest expert IDs)\n", .{@as(f64, @floatFromInt(bytes)) / 1e9});
        const start = std.Io.Timestamp.now(sch.io, .awake);
        try engine.warmCache();
        const elapsed_ns: u64 = @intCast(start.untilNow(sch.io, .awake).nanoseconds);
        log.info("[expert-stream] cache warm complete: slots={d}/{d} per layer, layers={d}, bytes={d}, elapsed_ms={d}\n", .{
            slots, engine.plan.slots_per_layer, layers, bytes, elapsed_ns / std.time.ns_per_ms,
        });
    }

    // Eager warmup: faults weight pages + compiles the decode-path kernels
    // on this thread's stream. ~600-900 ms at boot but the first user request
    // skips a cold path — observed savings on Gemma 4 E4B 4-bit.
    if (params.warmup_eager) {
        const warmup_start = std.Io.Timestamp.now(sch.io, .awake);
        xfm_ptr.warmup() catch |err| {
            log.warn("Warmup failed ({s}); continuing without it — first request may be slow.\n", .{@errorName(err)});
        };
        const warmup_ns: u64 = @intCast(warmup_start.untilNow(sch.io, .awake).nanoseconds);
        log.info("Warmup complete ({d} ms).\n", .{warmup_ns / std.time.ns_per_ms});
        if (mtp_enabled and xfm_ptr.qwen4_mtp != null) {
            const cap = generate_mod.Generator.resolveMtpDepthCapForProfile(params.mtp_depth, mtp_cost_profile);
            xfm_ptr.warmupSpecVerify(cap, params.kv_quant_config) catch |err| {
                log.warn("[spec-warmup] failed ({s}); the first round at each width pays its kernel compile inside the round.\n", .{@errorName(err)});
            };
        }
        if (mimoVerifyWarmWanted(params.config)) mimoSpecWarmup(sch.io, xfm_ptr, mimo_head, params.kv_quant_config);
    }

    // ── Phase 05: install everything onto the LoadedModel entry, mark
    //    ready, and update the scheduler's borrowed views. The registry
    //    mutex guards the state transition + `current_resident_bytes`
    //    accounting; `state_cond.broadcast` (inside markReadyLocked) wakes
    //    any waiter blocked in ensureLoaded.
    //
    //    Everything below this comment must be infallible — once we begin
    //    assigning to `params.entry`, the per-ptr errdefers above would
    //    double-free if we error-return. `drafter_path_owned` was alloc'd
    //    up front for exactly this reason; the per-model prefix cache
    //    init is a struct literal (no fallible alloc).
    const entry = params.entry;
    entry.weights = weights_ptr;
    entry.transformer = xfm_ptr;
    entry.vision_encoder = vision_ptr;
    entry.dflash = dflash_ptr;
    entry.drafter_block_size = sch.drafter_block_size;
    entry.mtp = if (mtp_ptr) |h|
        generate_mod.MtpHeadRef{ .qwen = h }
    else if (mtp_enabled and xfm_ptr.qwen4_mtp != null)
        generate_mod.MtpHeadRef{ .qwen4 = xfm_ptr }
    else if (mimo_head) |h|
        generate_mod.MtpHeadRef{ .mimo = h }
    else
        null;
    // Resolve the auto (0) cap here so every downstream reader of
    // `lm.mtp_depth` (server log lines, slot params) sees the real value.
    entry.mtp_depth = generate_mod.Generator.resolveMtpDepthCapForProfile(params.mtp_depth, mtp_cost_profile);
    xfm_ptr.mtp_depth_free = generate_mod.Generator.mtpDepthCapFree(params.mtp_depth);
    if (mimo_head) |h| {
        // The head count AND the widest verify whose rows stay byte-identical
        // to serial decode: a deeper round would hand the forward more rows
        // than the decode-shaped MiMo verify serves.
        const rows_max: u32 = generate_mod.Generator.mtpVerifyDraftsMax(true);
        entry.mtp_depth = @min(entry.mtp_depth, @min(@as(u32, @intCast(h.heads)), rows_max));
        xfm_ptr.mtp_depth_free = @min(xfm_ptr.mtp_depth_free, @min(@as(u32, @intCast(h.heads)), rows_max));
    }
    if (entry.mtp != null) log.info("[mtp] depth range {d}..{d} ({s})\n", .{
        depth_bounds.floorFor(depth_bounds.active, entry.mtp_depth),
        entry.mtp_depth,
        if (depth_bounds.active.explicit()) "--mtp-min-depth/--mtp-max-depth" else "default",
    });
    // A MERGED drafter has no `--drafter` to echo, so the reported path comes
    // from what was actually resolved — `drafter_loaded` and `drafter_path`
    // must not disagree about the same sidecar.
    if (drafter_path_owned.len == 0 and drafter_dir.len > 0 and
        dflash_ptr != null)
    {
        drafter_path_owned = try sch.allocator.dupe(u8, drafter_dir);
    }
    entry.drafter_path = drafter_path_owned;
    drafter_path_owned = &[_]u8{}; // disarm the errdefer
    // Transfer ownership of the heap-allocated CPU state from `params` to
    // the entry. The caller (main.zig) MUST NOT free these — `LoadedModel.deinit`
    // walks them in the same `*X` pointer form they came in.
    entry.releaseRetainedCpuState();
    entry.config = params.config;
    entry.tokenizer = params.tok;
    entry.chat_config = params.chat_config;
    // Per-model hot prefix cache (Plan 03 → Plan 05 move). Hybrid recurrent
    // archs are accepted iff `ssm_checkpoint_stride > 0` (Phase 1 of the
    // performance plan): with per-stride SSM checkpoints we can rewind both
    // KV and SSM to a snapshotted prefix; without them, divergence forces a
    // full reset, so we keep the legacy single-slot path for hybrid.
    const enable_ssm_cps = params.ssm_checkpoint_stride > 0;
    const ram_prefix_cache = params.prefix_cache_ram_enabled;
    // Phase 3 persists hybrid recurrent state too: the disk tier is allowed whenever the RAM tier
    // accepts the arch — pure-attention always, hybrid iff SSM checkpoints are enabled.
    const has_ssm_layers = params.config.has_hybrid_layers or
        params.config.full_attention_interval > 0;
    const disk_ok = !has_ssm_layers or enable_ssm_cps;
    const cache_applies = params.prefix_cache_capacity > 0 and
        prefix_cache_mod.HotPrefixCache.shouldUse(params.config, enable_ssm_cps);
    // The SSD tier comes up BEFORE anything is sized for it: whether this model has one decides its
    // RAM semantics (SSD-first or not), its bills and its checkpoint capture. A tier that was wanted
    // and did not come up (no room, an unreadable fingerprint, a failed init) is recorded on the config,
    // so every bill reads the same answer as `--prefix-cache-disk off`.
    var disk_tier: ?kv_disk_cache.DiskTier = null;
    if (cache_applies and params.prefix_cache_disk_bytes > 0 and disk_ok) attach: {
        const fp = kv_disk_cache.modelFingerprint(sch.allocator, sch.io, entry.path) catch |err| {
            log.warn("[disk-cache] fingerprint failed: {s} — persistence off for this model\n", .{@errorName(err)});
            break :attach;
        };
        defer sch.allocator.free(fp);
        const base = kv_disk_cache.defaultBaseDir(sch.allocator) catch break :attach;
        defer sch.allocator.free(base);
        const plan: kv_disk_cache.DefaultDiskBudget = if (params.prefix_cache_disk_resolver) |resolve|
            resolve(params.config, base, kv_disk_cache.tierBytes(sch.io, base, fp)) orelse break :attach
        else
            .{ .bytes = params.prefix_cache_disk_bytes, .bound = .flag, .cap = params.prefix_cache_disk_bytes };
        disk_tier = kv_disk_cache.DiskTier.init(
            sch.allocator,
            sch.io,
            base,
            fp,
            plan.bytes,
            kv_disk_cache.DEFAULT_CHUNK_TOKENS,
        ) catch |err| {
            log.warn("[disk-cache] init failed: {s} — persistence off for this model\n", .{@errorName(err)});
            break :attach;
        };
        disk_tier.?.operator_cap = plan.cap;
    }
    params.config.prefix_cache_disk_declined = params.prefix_cache_disk_bytes > 0 and disk_tier == null;
    const disk_prefix_cache = disk_tier != null;
    if (cache_applies and (ram_prefix_cache or disk_prefix_cache)) {
        // The weights are resident here, so the resolver's active-memory read
        // is honest; the raw launch budget never reaches initWithMem (a 40 GB
        // cap beside a ~70 GB pack was the 2026-08-30 uncatchable Metal OOM).
        // Disk-only mode retains no reusable KV in RAM, so it asks for no RAM budget.
        var ssd_idle_mem: u64 = 0;
        const clamped_prefix_mem: u64 = if (!ram_prefix_cache)
            0
        else if (params.prefix_cache_mem_resolver) |resolve|
            resolve(params.config, params.prefix_cache_mem_bytes, .{}, &ssd_idle_mem)
        else
            params.prefix_cache_mem_bytes;
        if (ram_prefix_cache) {
            // On an SSD-first arch `--prefix-cache-mem` is the idle allowance beside the live session.
            const ram_bytes = if (prefix_cache_mod.ssdFirstActive(params.config, disk_prefix_cache, ram_prefix_cache)) ssd_idle_mem else clamped_prefix_mem;
            log.info("Allocating {d:.1} GB RAM for the prefix cache (--prefix-cache-mem)\n", .{@as(f64, @floatFromInt(ram_bytes)) / (1024.0 * 1024.0 * 1024.0)});
        }
        entry.prefix_cache = prefix_cache_mod.HotPrefixCache.initWithMem(
            sch.allocator,
            if (ram_prefix_cache) params.prefix_cache_capacity else 0,
            clamped_prefix_mem,
        );
        entry.prefix_cache.?.qsa_history_required = params.config.indexer_budget != 0 and !params.config.isGlm5();
        // Checkpoint-retention arch gate, mirrored once: `HotPrefixCache`/`DiskTier` never
        // see a ModelConfig. The ungated value names the previous behaviour at each site.
        entry.prefix_cache.?.cp_thin = if (params.config.longCtxGated() or params.config.isGlm5()) .min_span_recency else .min_span;
        entry.prefix_cache.?.ssd_idle_mem = ssd_idle_mem;
        entry.prefix_cache.?.hybrid = has_ssm_layers or params.config.isGlm5();
        if (disk_tier) |tier| {
            entry.prefix_cache.?.disk = tier;
            entry.prefix_cache.?.disk.?.cp_thin =
                if (params.config.longCtxGated() or !ram_prefix_cache or params.config.isGlm5()) .min_span_recency else .oldest;
            entry.prefix_cache.?.disk.?.ssm_max_per_entry = if (params.config.isGlm5())
                @import("glm5_prefix.zig").checkpoint_cap
            else if (params.config.longCtxGated() or !ram_prefix_cache)
                kv_disk_cache.SSM_DISK_MAX_PER_ENTRY
            else
                kv_disk_cache.SSM_DISK_MAX_PER_ENTRY_LEGACY;
        }
        // SSD-first: arch + env switch + a live disk tier (attached above); without one qwen4_exp takes the RAM arm.
        entry.prefix_cache.?.ssd_first = prefix_cache_mod.ssdFirstActive(
            params.config,
            entry.prefix_cache.?.disk != null,
            ram_prefix_cache,
        );
        // Every tier writes off the inference thread, so a commit lands whole after the response.
        if (entry.prefix_cache.?.disk) |*d| d.enableBackgroundWriter();
        if (entry.prefix_cache.?.ssd_first) {
            entry.prefix_cache.?.disk.?.ssd_first = true;
            // Startup sweep of strays + root-wide LRU across sibling fingerprints.
            entry.prefix_cache.?.disk.?.sweepSiblings();
        }
        entry.ssm_checkpoint_stride = params.ssm_checkpoint_stride;
        const cp_max = if (params.config.isGlm5()) @import("glm5_prefix.zig").checkpointMax(params.ssm_checkpoint_max) else params.ssm_checkpoint_max;
        entry.ssm_checkpoint_max = cp_max;
        // The cache re-applies the cap after a replace-path merge; without this
        // it defaults to 0 (unlimited) and multi-turn entries grow unbounded.
        entry.prefix_cache.?.ssm_checkpoint_max = cp_max;
    }
    // Iteration 2 (perf-plan Phase 4 #3): tokenize cache for warm-path
    // chat-template renders (`chat_mod.formatChat` at the handler boundary).
    // Default capacity is small (4 entries) because chat conversations
    // mutate the messages list every turn; the goal is to catch warm
    // reuse benches and repeated agent-loop probes, not to memoize a
    // full session.
    if (params.tokenize_cache_entries > 0) {
        entry.tokenize_cache = tokenize_cache_mod.TokenizeCache.init(
            sch.allocator,
            params.tokenize_cache_entries,
        );
    }

    // The weights the load preflight billed. Drives the registry's resident-memory gate and
    // `/v1/models` `bytes_resident`.
    const bytes_resident: u64 = if (params.config.expert_streaming)
        expertStreamingGateBytes(
            streaming_resident_bytes.?,
            params.config.expert_cache_bytes,
            params.config.expert_fill_peak_bytes,
            params.config.expert_bounce_bytes,
        )
    else
        residentWeightBytes(streaming_resident_bytes, entry.bytes_on_disk, plainWeightBytes(sch.io, sch.allocator, params.config, params.model_dir, mtpChoiceFor(params.mtp_enabled, params.mtp_explicit, params.config).on), params.config.num_hidden_layers, params.config.hidden_size);

    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.markReadyLocked(entry, bytes_resident);
    sch.registry.mutex.unlock(sch.io);

    // Set borrowed views for scheduler-internal code (and `current_model`
    // for ensureLoaded → request handlers in Phase C).
    sch.current_model = entry;
    sch.weights = weights_ptr;
    sch.xfm = xfm_ptr;
    sch.vision_encoder = vision_ptr;
    sch.dflash = dflash_ptr;
    if (entry.prefix_cache) |*hc| sch.hot_prefix_cache = hc;
    publishHotCacheResidency(sch);
}

/// Republish the hot cache's residency for the connection thread's admission guard.
/// Called from the inference thread after a commit, eviction, invalidation or model switch.
const BUDGET_REVISE_WINDOW_NS: u64 = 10 * std.time.ns_per_s;

/// How a budget resolve differs from the load-time one: the cache's own resident bytes
/// are excluded from the machine read, and the resolver keeps quiet (`setBudget` logs).
pub const BudgetRevise = struct { exclude_bytes: u64 = 0, quiet: bool = false };

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
fn takeTeardownLocked(sch: *Scheduler, batch: []*Slot) TeardownTake {
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

fn destroyAbandoned(head: ?*Slot) void {
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
        sch.embed_queue.items.len > 0 or
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
        // 0b. Drain any pending vision/embed work. These run synchronously on
        //     behalf of conn threads waiting in `encodeVision` /
        //     `computeEmbedding`. Processed here (not concurrently with decode
        //     ticks) so they share the inference thread's mlx stream cleanly.
        var vision_batch: [4]*VisionEncodeRequest = undefined;
        var vision_n: usize = 0;
        var embed_batch: [4]*EmbedRequest = undefined;
        var embed_n: usize = 0;
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
            while (embed_n < embed_batch.len and sch.embed_queue.items.len > 0) {
                embed_batch[embed_n] = sch.embed_queue.orderedRemove(0);
                embed_n += 1;
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
                commitSlotIfApplicable(sch, s);
                if (s.model.prefix_cache) |*hc| {
                    if (s.model.transformer) |xf| hc.flushPendingDisk(xf.s);
                }
            }
            recordSlotCleanup(sch.metrics, s);
            // Second slot-end path (a decode-phase cancel never reaches finishSlot); the
            // record must not outlive the bytes `s.deinit()` frees.
            if (s.model.prefix_cache) |*hc| hc.releaseCheckout(@intFromPtr(s), "slot cleanup");
            if (s.model.transformer) |xfm| xfm.resetQsaPooledRope();
        }
        deinitSlotsReturningPool(cleanup_batch[0..cleanup_n]);
        destroyAbandoned(abandoned);
        for (orphan_vision[0..orphan_vision_n]) |ve| freeVisionArray(ve);
        if (vision_n > 0 or embed_n > 0) {
            for (vision_batch[0..vision_n]) |req| runVisionEncode(sch, req);
            for (embed_batch[0..embed_n]) |req| runEmbedRequest(sch, req);
        }
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
                        log.err("[scheduler] prefill failed for slot: {s}\n", .{@errorName(err)});
                        slot.markError(@errorName(err));
                        releaseNativeState(slot);
                        break :prefill;
                    };
                    break :prefill;
                }
                if (postPrefillTerminal(slot)) continue;
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
                log.err("[scheduler] decode tick failed: {s}\n", .{@errorName(err)});
                for (active.items) |s| s.markError(@errorName(err));
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

fn postPrefillTerminal(slot: *Slot) bool {
    if (slot.state == .errored) return true;
    if (!slot.cancelled.load(.acquire)) return false;
    releaseNativeState(slot);
    return true;
}

/// The slot still belongs to its connection thread until that thread calls `complete`; the
/// inference thread just stops touching it.
fn cullDecoding(sch: *Scheduler) void {
    cullDecodingWith(sch, commitSlotIfApplicable);
}

fn cullDecodingWith(sch: *Scheduler, commit: *const fn (*Scheduler, *Slot) void) void {
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
        if (s.glm5_request != null and s.cancelled.load(.acquire) and !s.finished and s.error_code == null) commit(sch, s);
        releaseNativeState(s);
    }
}

/// Native GLM has no reusable prefix snapshot: a slot the inference thread stops touching
/// (finished, errored or cancelled) releases its reserved target state before the next
/// admission read, not when its connection thread completes it.
fn releaseNativeState(slot: *Slot) void {
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
fn slotStateBytes(s: *const Slot) u64 {
    var bytes = s.cache.residentBytes();
    if (s.glm5_request) |*request| bytes += request.residentBytes();
    if (s.legacy_gen) |*gen| {
        if (gen.dflash_ctx) |*ctx| bytes += ctx.cache.residentBytes();
    }
    if (s.ssm_entries) |ents| for (ents) |*e| {
        bytes += transformer_mod.ssmEntryBytes(e);
    };
    return bytes + s.ring_cps.bytes();
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

/// One image through the tower: a patch-grid ViT takes pixel_values [N, feat]
/// and yields [1, N/merge², hidden].
fn encodeImageBlock(vision_enc: *VisionEncoder, img: VisionImagePixels) !mlx.mlx_array {
    if (@import("builtin").is_test) if (vision_block_test_hook) |hook| return hook(img);
    if (img.grid_h > 0) {
        const n: usize = @as(usize, img.grid_h) * img.grid_w;
        const feat: usize = (img.pixels.len / 4) / n;
        const shape = [_]c_int{ @intCast(n), @intCast(feat) };
        const pixel_arr = mlx.mlx_array_new_data(img.pixels.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(pixel_arr);
        return vision_enc.forwardPatches(pixel_arr, img.grid_h, img.grid_w);
    }
    return error.NoPatchGridEncoder;
}

fn encodeVideoBlock(vision_enc: *VisionEncoder, vid: VisionVideoPixels) !mlx.mlx_array {
    const n: usize = @as(usize, vid.grid_t) * vid.grid_h * vid.grid_w;
    const feat: usize = (vid.pixels.len / 4) / n;
    const shape = [_]c_int{ @intCast(n), @intCast(feat) };
    const pixel_arr = mlx.mlx_array_new_data(vid.pixels.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(pixel_arr);
    return vision_enc.forwardVideoPatches(pixel_arr, vid.grid_t, vid.grid_h, vid.grid_w);
}

/// Phase A4: encode one or more images on the inference thread. Writes the
/// result into a request struct + signals done, so the conn thread (blocked
/// in `encodeVision`) gets the output. On error, sets `req.error_name` and
/// still signals done.
/// Plan 05 Phase D: routes the encode through `req.model.vision_encoder`,
/// not the scheduler's borrowed-view singleton — each LoadedModel has its
/// own vision encoder when applicable.
fn runVisionEncode(sch: *Scheduler, req: *VisionEncodeRequest) void {
    const vision_enc = req.model.vision_encoder orelse {
        finishVisionRequest(sch, req, "VisionEncoderNotLoaded");
        return;
    };
    if (req.images.len == 0 and req.videos.len == 0 and req.audio.len == 0) {
        finishVisionRequest(sch, req, "EmptyImages");
        return;
    }
    // The connection thread's fit check ran before the queue: a prefill's cache growth, or the
    // output of an encode ahead of this one, may have taken the headroom since.
    if (req.bill_bytes > 0) if (vision_encode_available) |available_fn| if (req.model.config) |cfg| {
        const available = available_fn(cfg);
        if (req.bill_bytes > available) {
            req.available_bytes = available;
            finishVisionRequest(sch, req, vision_headroom_error);
            return;
        }
    };

    // Encode all soft tokens into `emb_parts`: images and videos in prompt
    // `order`, then audio, so the single splice channel scatters them in the
    // order of the placeholder rows.
    var emb_parts = std.ArrayList(mlx.mlx_array).empty;
    defer emb_parts.deinit(req.allocator);
    const failParts = struct {
        fn f(s: *Scheduler, r: *VisionEncodeRequest, parts: []mlx.mlx_array, name: []const u8) void {
            for (parts) |e| _ = mlx.mlx_array_free(e);
            finishVisionRequest(s, r, name);
        }
    }.f;

    if (req.order.len > 0 and req.order.len != req.images.len + req.videos.len) {
        finishVisionRequest(sch, req, "MediaOrderMismatch");
        return;
    }
    var n_vision: usize = 0;
    var n_video: usize = 0;
    var next_image: usize = 0;
    var next_video: usize = 0;
    for (0..req.images.len + req.videos.len) |block| {
        const is_image = if (req.order.len > 0) req.order[block] == .image else block < req.images.len;
        if ((is_image and next_image == req.images.len) or (!is_image and next_video == req.videos.len)) {
            failParts(sch, req, emb_parts.items, "MediaOrderMismatch");
            return;
        }
        const emb = (if (is_image) encodeImageBlock(vision_enc, req.images[next_image]) else encodeVideoBlock(vision_enc, req.videos[next_video])) catch |err| {
            failParts(sch, req, emb_parts.items, @errorName(err));
            return;
        };
        const rows: usize = @intCast(mlx.getShape(emb)[1]);
        if (is_image) {
            next_image += 1;
            n_vision += rows;
        } else {
            next_video += 1;
            n_video += rows;
        }
        emb_parts.append(req.allocator, emb) catch |err| {
            _ = mlx.mlx_array_free(emb);
            failParts(sch, req, emb_parts.items, @errorName(err));
            return;
        };
    }

    if (emb_parts.items.len == 0) {
        finishVisionRequest(sch, req, "EmptyImages");
        return;
    }

    // Single modality/clip: pass through. Multiple: concatenate along token dim.
    var combined: mlx.mlx_array = undefined;
    if (emb_parts.items.len == 1) {
        combined = emb_parts.items[0];
        emb_parts.items[0] = mlx.mlx_array_new(); // sentinel so the deferred-free path is a no-op
    } else {
        const cat_vec = mlx.mlx_vector_array_new_data(emb_parts.items.ptr, emb_parts.items.len);
        defer _ = mlx.mlx_vector_array_free(cat_vec);
        combined = mlx.mlx_array_new();
        if (mlx.mlx_concatenate_axis(&combined, cat_vec, 1, vision_enc.s) != 0) {
            _ = mlx.mlx_array_free(combined);
            failParts(sch, req, emb_parts.items, "ConcatenateFailed");
            return;
        }
    }
    for (emb_parts.items) |e| _ = mlx.mlx_array_free(e);

    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    req.result = combined;
    req.n_vision_tokens = n_vision;
    req.n_video_tokens = n_video;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

fn finishVisionRequest(sch: *Scheduler, req: *VisionEncodeRequest, err_name: []const u8) void {
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    if (req.error_name) |old| req.allocator.free(old);
    req.error_name = req.allocator.dupe(u8, err_name) catch null;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

/// Service one embedding request on the inference thread. Runs the batched
/// forward pass via `generate.computeEmbeddingsBatch(xfm, ...)`, which resets
/// the global xfm.cache before every sub-batch, and wakes the conn thread.
fn runEmbedRequest(sch: *Scheduler, req: *EmbedRequest) void {
    const xfm_ptr = req.model.transformer.?;
    const results = generate_mod.computeEmbeddingsBatch(req.allocator, xfm_ptr, req.token_seqs) catch |err| {
        finishEmbedRequest(sch, req, @errorName(err));
        return;
    };
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    req.results = results;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

fn finishEmbedRequest(sch: *Scheduler, req: *EmbedRequest, err_name: []const u8) void {
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    if (req.error_name) |old| req.allocator.free(old);
    req.error_name = req.allocator.dupe(u8, err_name) catch null;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

/// Plan 05 Phase D: service a cold-load work item on the inference thread.
/// Conn thread has already:
///   * Marked `req.entry.state = .loading` (transitioned from .unloaded).
///   * (Optionally) marked `req.evict_entry.state = .evicting` and drained
///     its refcount to 0.
///   * Parsed CPU-only state (config/tok/chat_config) and handed pointers
///     in via the request.
/// We unload the victim (if any), run the load body, install everything on
/// `req.entry`, mark ready, and broadcast `req.done_cond` so the conn
/// thread wakes. On failure, mark `.error_state` so future ensureLoaded
/// calls fail fast; the conn thread surfaces a 500.
fn logWiredPolicy(r: mlx.WiredPolicyResult) void {
    if (r.target) |t| {
        log.info("[wired] mode={s} limit={d} MB\n", .{ @tagName(r.mode), t / (1024 * 1024) });
    } else {
        log.debug("[wired] mode={s} declined (no gpu / empty live set)\n", .{@tagName(r.mode)});
    }
}

fn runLoadRequest(sch: *Scheduler, req: *LoadRequest) void {
    // Step 1: evict victims (if any) BEFORE the load, so peak GPU residency
    // never holds the old + new model at once. unloadResident() drops
    // mlx_arrays — same thread-stream invariant as cleanup_queue drain.
    const wired_before = if (req.evict_entries.len > 0) status.getWiredMemBytes() else 0;
    var evicted_bytes: u64 = 0;
    for (req.evict_entries) |victim| {
        const victim_bytes = victim.bytes_resident; // unloadResident zeroes it
        log.info("[registry] evicting model id={s} ({d:.2} GB resident)\n", .{
            victim.id,
            @as(f64, @floatFromInt(victim_bytes)) / 1_073_741_824.0,
        });
        victim.unloadResident();
        evicted_bytes +|= victim_bytes;
        sch.registry.mutex.lockUncancelable(sch.io);
        sch.registry.accountEvictedLocked(victim_bytes);
        sch.registry.finalizeEvictionLocked(victim);
        sch.registry.mutex.unlock(sch.io);
    }
    // The preflight below reads free memory: the victims' must be back first.
    if (evicted_bytes > 0) {
        _ = mlx.mlx_clear_cache();
        waitForUnwire(sch.io, wired_before, evicted_bytes);
    }

    // Step 2: the actual load. On error, mark .error_state and signal done
    // (conn thread frees pre-parsed CPU state — ownership stays on req on
    // the failure path).
    doLoadOnInferenceThread(sch, req) catch |err| {
        log.err("[registry] load failed for model id={s}: {s}\n", .{ req.entry.id, @errorName(err) });
        sch.registry.mutex.lockUncancelable(sch.io);
        if (err == error.InsufficientMemory) {
            // A memory-preflight refusal is transient, not a property of the
            // checkpoint: the 503 tells the user to free memory and retry, so
            // the entry must go back to .unloaded — stuck in .error_state the
            // retry would fail fast until a server restart (#144).
            sch.registry.markUnloadedLocked(req.entry);
        } else {
            // markErrorLocked dupes the error name onto the entry; the
            // conn thread reads it back from the entry, not from req.
            sch.registry.markErrorLocked(req.entry, @errorName(err));
        }
        sch.registry.mutex.unlock(sch.io);
        finishLoadRequest(sch, req, @errorName(err));
        return;
    };

    log.info("[registry] model id={s} ready ({d:.2} GB resident)\n", .{
        req.entry.id,
        @as(f64, @floatFromInt(req.entry.bytes_resident)) / 1_073_741_824.0,
    });
    // Re-apply so `fit` capacity covers everything this load brought in
    // (MTP head / drafter / vision land after the mid-load apply).
    logWiredPolicy(mlx.applyWiredPolicy());
    finishLoadRequest(sch, req, null);
}

fn finishLoadRequest(sch: *Scheduler, req: *LoadRequest, err_name: ?[]const u8) void {
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    if (err_name) |name| {
        if (req.error_name) |old| req.allocator.free(old);
        req.error_name = req.allocator.dupe(u8, name) catch null;
    }
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

/// Free a model's resident mlx state on the inference thread (stream-bound,
/// same invariant as the cleanup-queue drain) and finalize the eviction
/// accounting. The conn thread already marked the entry `.evicting` and
/// drained its refcount in `unloadModel`.
fn runUnloadRequest(sch: *Scheduler, req: *UnloadRequest) void {
    const entry = req.entry;
    const bytes = entry.bytes_resident; // unloadResident zeroes it
    log.info("[registry] unloading model id={s} ({d:.2} GB resident)\n", .{
        entry.id,
        @as(f64, @floatFromInt(bytes)) / 1_073_741_824.0,
    });
    const wired_before = status.getWiredMemBytes();
    entry.unloadResident();
    // unloadResident freed the arrays into MLX's allocator cache — clear it
    // so the unload actually returns the memory to the OS (the whole point
    // of the load→generate→unload flow).
    _ = mlx.mlx_clear_cache();
    waitForUnwire(sch.io, wired_before, bytes);
    // Drop any borrowed views that pointed at this entry so post-unload reads
    // don't dangle (gen entries leave xfm null already, but an LLM unload
    // must clear them).
    if (sch.current_model == entry) {
        sch.current_model = null;
        sch.xfm = null;
        sch.weights = null;
        sch.vision_encoder = null;
        sch.dflash = null;
        sch.hot_prefix_cache = null;
        publishHotCacheResidency(sch);
        if (hot_cache_budget_invalidate) |f| f();
    }
    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.accountEvictedLocked(bytes);
    sch.registry.finalizeEvictionLocked(entry);
    sch.registry.mutex.unlock(sch.io);
    publishDiskStats(sch);

    // Shrink `fit` capacity back to the surviving live set — leaving the
    // freed model's headroom in place is exactly the per-transient-commit
    // configuration the policy exists to avoid.
    logWiredPolicy(mlx.applyWiredPolicy());

    req.done_mu.lockUncancelable(sch.io);
    req.done = true;
    req.done_cond.broadcast(sch.io);
    req.done_mu.unlock(sch.io);
}

/// The kernel unwires a freed Metal buffer asynchronously (measured ~0.5 s for 50 GB), and a
/// load right behind the unload read that memory as taken (preflight 45 GB free, 95 GB half a
/// second later). The unload answers once most of what it freed is back, or after a bound.
fn waitForUnwire(io: std.Io, wired_before: u64, freed: u64) void {
    if (wired_before == 0 or freed == 0) return;
    var waited_ms: u32 = 0;
    while (!unwireSettled(wired_before, status.getWiredMemBytes(), freed) and waited_ms < UNWIRE_WAIT_MAX_MS) : (waited_ms += 50) {
        std.Io.sleep(io, .fromMilliseconds(50), .real) catch return;
    }
    if (waited_ms > 0) log.info("[registry] waited {d} ms for the unloaded model's memory to unwire\n", .{waited_ms});
}

const UNWIRE_WAIT_MAX_MS: u32 = 3000;

/// Nine tenths of the freed bytes back from the wired set; other processes wire and unwire too.
pub fn unwireSettled(wired_before: u64, wired_now: u64, freed: u64) bool {
    return wired_before -| wired_now >= freed / 10 * 9;
}

test "unwireSettled: the unload waits until most of the freed bytes left the wired set" {
    const gib: u64 = 1 << 30;
    try testing.expect(!unwireSettled(53 * gib, 37 * gib, 50 * gib)); // t+0.1 s in the measured unload
    try testing.expect(unwireSettled(53 * gib, 7 * gib, 50 * gib)); // t+0.7 s
    try testing.expect(unwireSettled(53 * gib, 60 * gib, 0)); // nothing freed, nothing to wait for
    try testing.expect(!unwireSettled(10 * gib, 12 * gib, 5 * gib)); // wired grew: never underflows
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
fn commitSlotIfApplicable(sch: *Scheduler, slot: *Slot) void {
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
    const total_tokens = sch.allocator.alloc(u32, total_len) catch return;
    defer sch.allocator.free(total_tokens);
    @memcpy(total_tokens[0..slot.full_prompt.len], slot.full_prompt);
    @memcpy(total_tokens[slot.full_prompt.len..], gen_ptr.generated_ids.items);

    // Phase 1: drain any SSM checkpoints captured by the Generator's prefill
    // loop and hand them to the cache alongside the KV snapshot. For plain-
    // attn models this returns an empty slice (no allocator hit). Ownership
    // transfers to the cache via `commitWithSsm`; freeing happens on
    // eviction.
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
        if (transformer_mod.qsaHistoryShareEnabled() and !transformer_mod.checkpointHasQsaPooled(&cps[cps.len - 1])) {
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

/// GLM: the request's MLA rows through the newest checkpoint a tier keeps serve every KDA checkpoint
/// at or below it; without one nothing restores. The RAM tier copies the rows its budget keeps; the
/// SSD tier shares every row through the newest checkpoint, read by the flush after the response.
/// Ownership of `cps` passes to the cache.
fn commitGlmSlot(hc: *prefix_cache_mod.HotPrefixCache, slot: *Slot, request: *const glm5_forward_mod.Request, tokens: []const u32, cps_opt: ?[]transformer_mod.SSMCheckpoint, dflash: ?prefix_cache_mod.DflashCommit, prompt_len: usize) void {
    const cps_all = cps_opt orelse return;
    const glm5_prefix = @import("glm5_prefix.zig");
    const s = slot.model.transformer.?.s;
    // The rows are chosen before the copy: only what the RAM tier keeps, never a full copy followed by a trim.
    const window: u64 = if (dflash) |d| prefix_cache_mod.HotPrefixCache.liveCacheBytes(d.cache) else 0;
    const row_bytes = glm5_prefix.rowBytesOf(request);
    const ram_len = hc.glmCommitLen(cps_all, request.offset, row_bytes, window);
    const disk_len = hc.glmDiskLen(cps_all, request.offset, row_bytes, window);
    const len = @max(ram_len orelse 0, disk_len orelse 0);
    if (len == 0) {
        log.debug("[hot-cache] GLM commit keeps no rows within either tier's budget; not committed\n", .{});
        for (cps_all) |*cp| cp.deinit(hc.allocator);
        hc.allocator.free(cps_all);
        return;
    }
    const cps = glm5_prefix.keepThrough(hc.allocator, cps_all, len);
    const ram_rows: ?glm5_prefix.MlaRows = if (ram_len) |n| glm5_prefix.MlaRows.capture(hc.allocator, request, n, s) catch |err| blk: {
        log.warn("[hot-cache] GLM MLA rows not captured: {s}; the RAM tier keeps nothing\n", .{@errorName(err)});
        break :blk null;
    } else null;
    const disk_rows: ?glm5_prefix.MlaRows = if (disk_len) |n| glm5_prefix.MlaRows.shareLive(hc.allocator, request, n, s) catch |err| blk: {
        log.warn("[disk-cache] GLM MLA rows not shared: {s}; the SSD tier gets the RAM tier's rows\n", .{@errorName(err)});
        break :blk null;
    } else null;
    if (ram_rows == null and disk_rows == null) {
        for (cps) |*cp| cp.deinit(hc.allocator);
        hc.allocator.free(cps);
        return;
    }
    const n_cps = cps.len;
    const st = hc.commitGlmTiers(&slot.cache, tokens, slot.has_tools, slot.vision_key, slot.cache_key, slot.media_start, cps, ram_rows, disk_rows, dflash, prompt_len) catch |err| {
        log.warn("[hot-cache] commit failed: {s}\n", .{@errorName(err)});
        return;
    };
    switch (st) {
        .ok => |n| log.info("[hot-cache] committed {d}/{d} GLM tokens ({d} KDA checkpoints)\n", .{ n, tokens.len, n_cps }),
        .disk_only => log.info("[disk-cache] captured {d}/{d} GLM tokens for the SSD tier ({d} KDA checkpoints)\n", .{ disk_len orelse 0, tokens.len, n_cps }),
        else => {},
    }
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
    // The sink's `forwarded` is the authoritative length — `cache.step`
    // only advances when Generator init completes, so it reads 0 on every
    // aborted prefill.
    const len = cancelledPrefillCommitLen(salvage.forwarded, slot.full_prompt.len) orelse {
        log.debug("[hot-cache] cancelled prefill forwarded < {d} tokens — below the commit floor; not committed\n", .{prefix_cache_mod.MIN_CANCELLED_COMMIT_TOKENS});
        return;
    };
    const cps: ?[]transformer_mod.SSMCheckpoint = if (salvage.checkpoints.len > 0) salvage.checkpoints else null;
    // Pass the media boundary RAW: when the cancelled prefill forwarded less
    // than the media position, the cache re-keys the pure-text entry to the
    // null vision key (a kept pixel key with no boundary is the
    // conservative-rejection poison shape; live 2026-09-07).
    const media_start = slot.media_start;
    // Ownership of the checkpoints transfers to the cache unconditionally —
    // its error paths free them (#330 adjacent) — so detach from the slot
    // BEFORE the call or Slot.deinit frees them a second time.
    slot.cancelled_prefill = .{};
    if (slot.glm5_request) |*request| return commitGlmSlot(hc, slot, request, slot.full_prompt[0..len], cps, null, len);
    const st = hc.commitWithMediaState(&slot.cache, slot.full_prompt[0..len], slot.has_tools, slot.vision_key, slot.cache_key, media_start, cps, null, null, len) catch |err| {
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
    tier: generate_mod.DegenerateTail.Tier,
    trim_start: usize,
};

/// Pure decision + trim point. The three tiers live in generate.zig
/// (`degenerateTail`); this is where their verdict becomes server behaviour.
pub fn loopStopDecision(generated_ids: []const u32) ?LoopStop {
    const d = generate_mod.degenerateTail(generated_ids) orelse return null;
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
    n += (std.fmt.bufPrint(out[n..], "] bytes=\"", .{}) catch return out[0..n]).len;
    var cap = @min(text.len, SHORT_GEN_TEXT_CAP);
    // A token is a BPE fragment: never cut inside a multi-byte sequence.
    while (cap > 0 and cap < text.len and (text[cap] & 0xC0) == 0x80) cap -= 1;
    for (text[0..cap]) |c| {
        var one = [_]u8{if (c < 0x20) ' ' else c};
        const esc: []const u8 = switch (c) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            else => one[0..1],
        };
        if (n + esc.len > out.len) break;
        @memcpy(out[n..][0..esc.len], esc);
        n += esc.len;
    }
    n += (std.fmt.bufPrint(out[n..], "\" path={s}", .{path}) catch return out[0..n]).len;
    return out[0..n];
}

/// A one-word answer and a block-path handling bug look identical until the ids are named.
fn logShortGen(slot: *Slot, reason: []const u8) void {
    const gen = if (slot.legacy_gen) |*g| g else return;
    const mode = specTickMode(
        slot.enable_mtp,
        gen.mtp != null,
        slot.enable_drafter,
        gen.dflash != null,
        slot.enable_pld,
        gen.pld_enabled,
    );
    const ids = gen.generated_ids.items;
    const decoded: ?[]u8 = if (slot.model.tokenizer) |tok|
        (tok.decode(slot.allocator, ids, false) catch null)
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
fn recordSlotCleanup(metrics: ?*metrics_mod.Metrics, slot: anytype) void {
    const m = metrics orelse return;
    if (slot.metrics_recorded) return;
    if (slot.error_code) |name| if (std.mem.eql(u8, name, "PrefillDoesNotFit")) {
        slot.metrics_recorded = true;
        m.recordRejected();
        return;
    };
    recordSlotEnd(m, slot, cleanupOutcome(slot));
}

fn finishSlot(sch: *Scheduler, slot: *Slot, reason: []const u8) void {
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

/// Publish every resident model's SSD tier (budget, bytes in use, entries, writer bytes) onto its own
/// `LoadedModel`, and the writers' total for admission. Inference thread.
fn publishDiskStats(sch: *Scheduler) void {
    var writer_total: u64 = 0;
    sch.registry.mutex.lockUncancelable(sch.io);
    defer sch.registry.mutex.unlock(sch.io);
    var it = sch.registry.entries.valueIterator();
    while (it.next()) |entry_ptr| {
        const entry = entry_ptr.*;
        var snap: model_registry_mod.DiskStats.Snapshot = .{};
        if (entry.state == .ready) {
            if (entry.prefix_cache) |*hc| if (hc.disk) |*d| {
                snap = .{ .budget = d.max_bytes, .used = d.total_bytes, .entries = d.entryCount(), .writer_bytes = d.writerHostBytes() };
            };
        }
        entry.disk_stats.publish(snap);
        writer_total +|= snap.writer_bytes;
    }
    sch.disk_writer_host_bytes.store(writer_total, .monotonic);
}

/// The one reading admission bills for the SSD writers: host bytes every resident model's writer
/// holds or may hold, which `mlx_get_active_memory` never counts. Safe from any thread.
pub fn diskWriterHostBytes(sch: *const Scheduler) u64 {
    return sch.disk_writer_host_bytes.load(.monotonic);
}

/// One `[disk-cache] usage` line per finished turn, from the finishing model's own tier.
fn logDiskUsage(model: *const LoadedModel) void {
    const snap = model.disk_stats.snapshot();
    if (snap.budget == 0) return;
    var buf: [128]u8 = undefined;
    log.info("{s}\n", .{kv_disk_cache.diskUsageLine(&buf, snap.used, snap.budget, snap.entries)});
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

/// DIAGNOSTIC (SUSHI_GLM_ROWS_UBENCH=N): `glm5_rows_ubench.run` per `_CTX=<tokens>[,<tokens>...]`
/// (default 1024) over `_TEXT=<abs path>`.
fn glmRowsUbench(alloc: std.mem.Allocator, target: *@import("glm5_forward.zig").Model, latent_bits: u8, tok: *Tokenizer) void {
    const tio = std.Io.Threaded.global_single_threaded.io();
    const rounds = @max(1, std.fmt.parseInt(usize, std.mem.sliceTo(std.c.getenv("SUSHI_GLM_ROWS_UBENCH").?, 0), 10) catch 8);
    const ctxs: []const u8 = if (std.c.getenv("SUSHI_GLM_ROWS_UBENCH_CTX")) |r| std.mem.sliceTo(r, 0) else "1024";
    var ids: []u32 = &.{};
    defer alloc.free(ids);
    if (std.c.getenv("SUSHI_GLM_ROWS_UBENCH_TEXT")) |path| blk: {
        const text = std.Io.Dir.cwd().readFileAlloc(tio, std.mem.sliceTo(path, 0), alloc, .limited(64 << 20)) catch break :blk;
        defer alloc.free(text);
        ids = tok.encode(alloc, text) catch break :blk;
    }
    var it = std.mem.tokenizeScalar(u8, ctxs, ',');
    while (it.next()) |field| {
        const ctx = std.fmt.parseInt(usize, field, 10) catch continue;
        @import("glm5_rows_ubench.zig").run(alloc, target, latent_bits, ids, ctx, rounds) catch |err| log.warn("[glm-rows-ubench] failed: {s}\n", .{@errorName(err)});
    }
}

/// DIAGNOSTIC (SUSHI_GLM_ROUND_UBENCH=N): `glm5_round_ubench.run` per `_CTX=<tokens>[,<tokens>...]`
/// (default 8192) over `_TEXT=<abs path>`.
fn glmRoundUbench(alloc: std.mem.Allocator, target: *@import("glm5_forward.zig").Model, assistant: *DflashModel, latent_bits: u8, tok: *Tokenizer) void {
    const tio = std.Io.Threaded.global_single_threaded.io();
    const rounds = @max(1, std.fmt.parseInt(usize, std.mem.sliceTo(std.c.getenv("SUSHI_GLM_ROUND_UBENCH").?, 0), 10) catch 64);
    const ctxs: []const u8 = if (std.c.getenv("SUSHI_GLM_ROUND_UBENCH_CTX")) |r| std.mem.sliceTo(r, 0) else "8192";
    var ids: []u32 = &.{};
    defer alloc.free(ids);
    if (std.c.getenv("SUSHI_GLM_ROUND_UBENCH_TEXT")) |path| blk: {
        const text = std.Io.Dir.cwd().readFileAlloc(tio, std.mem.sliceTo(path, 0), alloc, .limited(64 << 20)) catch break :blk;
        defer alloc.free(text);
        ids = tok.encode(alloc, text) catch break :blk;
    }
    var it = std.mem.tokenizeScalar(u8, ctxs, ',');
    while (it.next()) |field| {
        const ctx = std.fmt.parseInt(usize, field, 10) catch continue;
        @import("glm5_round_ubench.zig").run(alloc, target, assistant, latent_bits, ids, ctx, rounds) catch |err| log.warn("[glm-round-ubench] failed: {s}\n", .{@errorName(err)});
    }
}

/// DIAGNOSTIC (SUSHI_PREFILL_UBENCH=N): N cold one-chunk prefills per arm at load, each from an
/// empty cache, logits never projected (the chunk loop never reads them). `_ROWS` (default 2025)
/// is capped at the chunk admission would pick for such a prompt; `_TEXT=<abs path>` tokenized
/// for real routing; `_ARMS=0,1,1,0` runs a pass per arm, 0 on the reference EXL3 NAX GEMM body.
fn prefillUbench(alloc: std.mem.Allocator, xfm: *Transformer, cfg: *const ModelConfig, tok: *Tokenizer) void {
    const tio = std.Io.Threaded.global_single_threaded.io();
    const n = @max(1, std.fmt.parseInt(usize, std.mem.sliceTo(std.c.getenv("SUSHI_PREFILL_UBENCH").?, 0), 10) catch 3);
    const asked: usize = if (std.c.getenv("SUSHI_PREFILL_UBENCH_ROWS")) |r| std.fmt.parseInt(usize, std.mem.sliceTo(r, 0), 10) catch 2025 else 2025;
    const pick = prefill_ubench_chunk orelse {
        log.warn("[prefill-ubench] skipped: no admission hook to size the chunk\n", .{});
        return;
    };
    // A wider forward than admission would run allocates outside the load's memory bill.
    const rows = @max(1, @min(asked, pick(cfg, asked + 1, 1, xfm.cache.config, false, 0, 0, false, false)));
    const ids = alloc.alloc(i32, rows) catch return;
    defer alloc.free(ids);
    for (ids, 0..) |*v, i| v.* = @intCast(1 + (i % 997));
    if (std.c.getenv("SUSHI_PREFILL_UBENCH_TEXT")) |path| blk: {
        const text = std.Io.Dir.cwd().readFileAlloc(tio, std.mem.sliceTo(path, 0), alloc, .limited(64 << 20)) catch break :blk;
        defer alloc.free(text);
        const enc = tok.encode(alloc, text) catch break :blk;
        defer alloc.free(enc);
        if (enc.len == 0) break :blk;
        for (ids, 0..) |*v, i| v.* = @intCast(enc[i % enc.len]);
    }
    var arms: [16]?bool = @splat(null);
    const n_arms = if (std.c.getenv("SUSHI_PREFILL_UBENCH_ARMS")) |r| parseUbenchArms(std.mem.sliceTo(r, 0), &arms) else 1;
    defer exl3_kernels.nax_reference_override = false;
    const had_error = mlx.errorPending();
    defer xfm.resetCache() catch {};
    const shape = [_]c_int{ 1, @intCast(rows) };
    const times = alloc.alloc(f64, n) catch return;
    defer alloc.free(times);
    for (arms[0..n_arms]) |arm| {
        exl3_kernels.nax_reference_override = if (arm) |a| !a else false;
        for (0..n + 1) |iter| {
            xfm.resetCache() catch return;
            var ctx = xfm.defaultCtx();
            ctx.skip_lm_head = true;
            const ti = mlx.mlx_array_new_data(ids.ptr, &shape, 2, .int32);
            defer _ = mlx.mlx_array_free(ti);
            var sw = io_util.Stopwatch.init(tio);
            const out = xfm.forwardWith(&ctx, ti) catch |err| {
                mlx.dropLatchedErrorUnless(had_error);
                log.warn("[prefill-ubench] forward failed: {s}\n", .{@errorName(err)});
                return;
            };
            defer _ = mlx.mlx_array_free(out);
            mlx.check(mlx.mlx_array_eval(out)) catch |err| {
                mlx.dropLatchedErrorUnless(had_error);
                log.warn("[prefill-ubench] eval failed: {s}\n", .{@errorName(err)});
                return;
            };
            if (iter > 0) times[iter - 1] = @as(f64, @floatFromInt(sw.read())) / 1.0e6;
        }
        std.mem.sort(f64, times, {}, std.sort.asc(f64));
        log.info("[prefill-ubench] arm={s} rows={d} n={d} median {d:.2} ms min {d:.2} max {d:.2} ({d:.1} tok/s)\n", .{
            if (arm) |a| (if (a) "1" else "0") else "shipped", rows, n, times[n / 2], times[0], times[n - 1],
            @as(f64, @floatFromInt(rows)) / times[n / 2] * 1000.0,
        });
    }
}

/// `_ARMS` list: each comma-separated entry is an arm, `0` the reference, anything else the shipped one.
fn parseUbenchArms(raw: []const u8, out: *[16]?bool) usize {
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, raw, ',');
    while (it.next()) |a| {
        if (n == out.len) break;
        out[n] = a[0] != '0';
        n += 1;
    }
    return @max(n, 1);
}

test "prefill ubench arms parse one entry per comma, 0 the reference and an empty list the shipped arm" {
    var arms: [16]?bool = @splat(null);
    try std.testing.expectEqual(@as(usize, 4), parseUbenchArms("0,1,1,0", &arms));
    try std.testing.expectEqualSlices(?bool, &.{ false, true, true, false }, arms[0..4]);
    arms = @splat(null);
    try std.testing.expectEqual(@as(usize, 1), parseUbenchArms("", &arms));
    try std.testing.expectEqual(@as(?bool, null), arms[0]);
    try std.testing.expectEqual(@as(usize, 16), parseUbenchArms("1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0", &arms));
}

/// Scale the measured M5/block-16 break-even to the effective number of
/// draft positions. `block_size` includes the always-emitted anchor. Thinking
/// is the actual resolved request mode; tools are neither necessary nor
/// sufficient for a reasoning preamble.
fn dflashGateMinimum(block_size: u32, enable_thinking: bool, moe_target: bool) f32 {
    const drafts = block_size -| 1;
    if (drafts == 0) return 0;
    const calibrated_min = if (enable_thinking)
        generate_mod.Generator.DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND
    else
        generate_mod.Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND;
    const scaled = calibrated_min * @as(f32, @floatFromInt(drafts)) / 15.0;
    // The bar is really "round cost / serial step cost", and the whole
    // calibration above was measured on DENSE trunks where one verify forward
    // costs about one serial step. A sparse trunk breaks that: an A1B MoE
    // decodes ~1B of weights per token but its verify reads every expert the
    // block's positions route to, so a round costs far more than a step while
    // the bar stayed at 0.53 and nothing ever disabled.
    //
    // Measured LFM2.5-8B-A1B + its DSpark sidecar, M4 Max, block 5, greedy:
    // novel prose accepts 1.40/round and runs 171 tok/s against 199 serial
    // (round cost = 1.40 x 199/171 = 1.63 steps), while an echo prompt accepts
    // 4.00 and runs 273. A floor of 1.8 disables the losing class after its
    // three warmup rounds and leaves the winning one untouched.
    if (!moe_target) return scaled;
    return @max(scaled, generate_mod.Generator.DFLASH_MOE_GATE_MIN_ACCEPTED_PER_ROUND);
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

pub var prefill_decode_share: f32 = 0;

pub fn prefillDecodeShare() f32 {
    return if (prefillInterleaveEnabled()) prefill_decode_share else 0;
}

fn liveDecodingCount(sch: *Scheduler) usize {
    sch.queue_mu.lockUncancelable(sch.io);
    defer sch.queue_mu.unlock(sch.io);
    var count: usize = 0;
    for (sch.decoding.items) |slot| {
        if (!slot.cancelled.load(.acquire) and !slot.finished and slot.error_code == null) count += 1;
    }
    return count;
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
fn prefillWriteThroughCb(opaque_ctx: *anyopaque, abs_kv_pos: usize, cps: []const transformer_mod.SSMCheckpoint) void {
    const wc: *WriteThroughCtx = @ptrCast(@alignCast(opaque_ctx));
    const slot = wc.slot;
    const hc: *prefix_cache_mod.HotPrefixCache = if (slot.model.prefix_cache) |*p| p else return;
    if (!hc.ssd_first) return;
    const d = if (hc.disk) |*dd| dd else return;
    if (d.writer == null) return;
    if (slot.vision_key != 0) return;
    if (abs_kv_pos == 0 or abs_kv_pos > slot.full_prompt.len) return;
    const s = if (slot.model.transformer) |x| x.s else return;
    wc.chunks += 1;
    // RAM-backed mode banks one chunk as crash salvage. SSD-only mode must keep pace with
    // prefill or the first completed turn can leave most of its prefix unavailable after restart.
    _ = d.appendCommitBounded(
        slot.cache.entries,
        abs_kv_pos,
        slot.cache.config,
        slot.full_prompt[0..abs_kv_pos],
        slot.has_tools,
        if (cps.len > 0) cps else null,
        s,
        if (hc.ram_enabled) WRITE_THROUGH_FLUSH_BOUND_BYTES else std.math.maxInt(u64),
    ) catch |err| {
        log.warn("  [disk-cache] prefill write-through failed: {s}\n", .{@errorName(err)});
    };
}

/// The write-through hook's per-call flush bound: one byte, so the chunk loop stops after the first chunk it writes.
pub const WRITE_THROUGH_FLUSH_BOUND_BYTES: u64 = 1;

const WriteThroughCtx = struct {
    slot: *Slot,
    chunks: u32 = 0,
};

/// The write-through only pays when this turn's prefill produces at least one whole disk
/// chunk: below that it persisted a warm turn's whole restored prefix inside TTFT
/// (+183/+369/+737 ms at 16k/32k/64k) for a prefix the end-of-request commit persists anyway.
fn writeThroughSpanReached(new_span: usize, chunk_tokens: u32) bool {
    return new_span >= chunk_tokens;
}

var write_through_span_declined_logged = std.atomic.Value(bool).init(false);
var write_through_off_logged = std.atomic.Value(bool).init(false);
var write_through_env_cached: ?bool = null;

/// `SUSHI_SSD_WRITE_THROUGH=0` takes the write-through out of the prefill loop; the
/// end-of-request commit then persists the whole turn. A real two-arm tradeoff (crash-safe
/// prefix vs TTFT) and the only way to A/B its cost.
pub fn writeThroughEnabledFromEnv(raw: ?[]const u8) bool {
    const v = raw orelse return true;
    return !std.mem.eql(u8, v, "0");
}

/// Read on the inference thread only.
fn writeThroughEnabled() bool {
    if (write_through_env_cached) |v| return v;
    const v = blk: {
        const raw = std.c.getenv("SUSHI_SSD_WRITE_THROUGH") orelse break :blk writeThroughEnabledFromEnv(null);
        break :blk writeThroughEnabledFromEnv(std.mem.sliceTo(raw, 0));
    };
    write_through_env_cached = v;
    return v;
}

/// The one gate for the write-through: SSD-first + a live writer + a disk tier + a new span
/// worth at least one chunk.
fn writeThroughArmed(slot: *Slot, new_span: usize) bool {
    const hc: *prefix_cache_mod.HotPrefixCache = if (slot.model.prefix_cache) |*p| p else return false;
    if (!writeThroughEnabled()) {
        if (hc.ssd_first and !write_through_off_logged.swap(true, .monotonic)) {
            log.info("  [disk-cache] prefill write-through disabled by SUSHI_SSD_WRITE_THROUGH=0 — the end-of-request commit persists every turn\n", .{});
        }
        return false;
    }
    if (!hc.ssd_first) return false;
    const d = if (hc.disk) |*dd| dd else return false;
    const writer_up = d.writer != null;
    if (!writer_up or slot.vision_key != 0) return false;
    if (!writeThroughSpanReached(new_span, d.chunk_tokens)) {
        if (!write_through_span_declined_logged.swap(true, .monotonic)) {
            log.info("  [disk-cache] prefill write-through declined: {d} new tokens is under one chunk ({d}) — the end-of-request commit persists this turn\n", .{ new_span, d.chunk_tokens });
        }
        return false;
    }
    return true;
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
    st: *generate_mod.AdaptiveWidthState,
) u32 {
    const wc: *ChunkWidthCtx = @ptrCast(@alignCast(opaque_ctx));
    const cfg = wc.cfg orelse return cur;
    const pick = prefill_chunk_adapt orelse return cur;
    const next = pick(cfg, wc.kv_bits, pos, cur, cap, st, chunkWidthStagedBytes(wc));
    if (!adaptiveChunkWidthFor(cfg)) return next;
    const share_cap = prefillShareCapFor(cfg, liveDecodingCount(wc.sch), prefillDecodeShare());
    return if (share_cap == 0) next else @min(next, share_cap);
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
        log.err("[interleave] decode tick failed: {s}\n", .{@errorName(err)});
        for (buf[0..n]) |s| s.markError(@errorName(err));
    };
    const tick_ns = sw.read();
    for (buf[0..n]) |s| s.decode_ns +|= tick_ns;
    return tick_ns;
}

/// Take `pending[idx]` into this tick's prefill pass (queue_mu held; the caller removes it
/// from `pending`).
fn admitForPrefillLocked(sch: *Scheduler, idx: usize) *Slot {
    const slot = sch.pending.items[idx];
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
    slot.ctx.kv_attn_fused = slot.kv_attn_fused;
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
        slot.enable_pld,
    );
    const use_mtp = wiring.use_mtp;
    const use_dflash = wiring.use_dflash or glmNativeDflashArmed(slot.model.config.?.isGlm5(), slot.enable_drafter, slot.dflash != null, slot.sampling, slot.logprobs_n);
    const use_pld = wiring.use_pld;
    log.debug("[spec-wiring] mtp={} dflash={} pld={} (slot: drafter_flag={} dflash_handle={})\n", .{
        use_mtp,             use_dflash,          use_pld,
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
    // reused turn blind and drafts against nothing. Measured on Muse 4-bit,
    // 160-token generations: a full-prefix hit cost 92.6% -> 66.5% per-draft
    // acceptance and 80.2 -> 60.9 tok/s. Adopted by the Generator below.
    var dflash_restored: ?dflash_mod.DflashCtx = null;
    errdefer if (dflash_restored) |*dc| dc.deinit();
    // The MTP head's committed history rides the prefix cache the same way:
    // it is built from trunk hiddens, a restore forwards nothing, and a
    // blind start collapses acceptance (measured ~70 -> ~38 tok/s on warm
    // Qwen3.6-27B echo). Adopted by the Generator below.
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
            var mtp_target: ?generate_mod.MtpCacheRef = if (use_mtp and slot.mtp != null)
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
            const lookup = (if (slot.glm5_request) |*request| hc.lookupAndRestoreGlm(
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
                checkpoints: u32,
                fits: *const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool, u64, u64, bool, bool, u32) bool,
                fn call(ctx: ?*anyopaque) bool {
                    const self: *@This() = @ptrCast(@alignCast(ctx.?));
                    return self.fits(self.cfg, self.seq, self.max_tokens, self.kv_cfg, self.unchunked, self.warm_matched, self.warm_capacity, self.warm_will_donate, self.enable_mtp, self.checkpoints);
                }
                fn keeping(ctx: ?*anyopaque, n: u32) bool {
                    const self: *@This() = @ptrCast(@alignCast(ctx.?));
                    self.checkpoints = n;
                    return call(ctx);
                }
            };
            var probe = Probe{
                .cfg = cfg,
                .seq = slot.full_prompt.len,
                .max_tokens = slot.max_tokens,
                .kv_cfg = slot.cache.config,
                .unchunked = generate_mod.visionPrefillUnchunked(slot.vision_embeddings != null),
                .warm_matched = hot_matched,
                .warm_capacity = slot.cache.residentCapacityTokens(),
                .warm_will_donate = hot_checked_out,
                .enable_mtp = slot.enable_mtp,
                .checkpoints = if (cfg.isGlm5() and cp_stride > 0) cp_max else 0,
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
            generate_mod.visionPrefillUnchunked(slot.vision_embeddings != null),
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
    const dflash_pass = dflash_restored;
    dflash_restored = null;
    const mtp_pass = mtp_restored;
    mtp_restored = null;
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
            .pld_enabled = use_pld,
            .drafter_block_size = slot.drafter_block_size,
            .dflash_enabled = use_dflash,
            .dflash = if (use_dflash) slot.dflash else null,
            // The dflash-resolved block rides the shared drafter_block_size.
            .dflash_block_size = slot.drafter_block_size,
            // Use the resolved thinking mode and normalize the M5/block-16
            // calibration to this machine's effective assistant width.
            .dflash_min_accepted_per_round = dflashGateMinimum(
                slot.drafter_block_size,
                slot.enable_thinking,
                slot.model.config.?.isMoe(),
            ),
            .mtp_enabled = use_mtp,
            .mtp_acceptance = generate_mod.mtpAcceptanceFor(slot.model.config.?.mtp_acceptance_override).value,
            .mtp_greedy_tail = generate_mod.mtpGreedyTailFor(slot.model.config.?.mtp_greedy_tail_override).value,
            .mtp = if (use_mtp) slot.mtp else null,
            // The model's head before this request's opt-out (`entry.mtp` already ANDs `--no-mtp`).
            .model_has_mtp = slot.mtp != null,
            .mtp_depth = slot.mtp_depth,
            .lookup_prompt = slot.full_prompt,
            .ctx = slot.ctx,
            // Regular path: skip the lazy preforward so cache.step lands at
            // exactly prompt_len with t1 NOT in cache. Generator.next's
            // transition shim sync-forwards [t1] on the first decode call.
            // PLD/drafter/MTP init paths already skip preforward unconditionally.
            .skip_lazy_preforward = !use_pld and !use_mtp and !use_dflash,
            .ssm_checkpoint_stride = cp_stride,
            .ssm_checkpoint_max = cp_max,
            .ssm_checkpoint_pos_offset = hot_matched,
            // A restored prefix already holds its image rows: the splice
            // resumes at the placeholder count inside the matched prefix.
            .vision_rows_before = if (slot.vision_embeddings != null and hot_matched > 0)
                generate_mod.countSpliceRows(@ptrCast(slot.full_prompt[0..hot_matched]), xfm_ptr.config.image_token_id, xfm_ptr.config.video_token_id)
            else
                0,
            // The width the admission guard billed for this request; the forward can never run wider.
            .pinned_prefill_chunk = req_prefill_chunk,
            .decode_share_width_cap = prefillShareCapFor(&xfm_ptr.config, liveDecodingCount(sch), prefillDecodeShare()),
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
            .write_through_hook = if (writeThroughArmed(slot, prefill_tokens.len))
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

fn runDecodeTick(sch: *Scheduler, active: []*Slot) !void {
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
            if (j - i >= mtpCrowdThresholdFor(mtp_buf[i]) and batchable_n + (j - i) <= batchable_buf.len) {
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
            var ctx_max: u32 = 0;
            // The stable insertion sort is part of the change: `std.sort.pdq` is unstable and
            // off qwen4_exp every key is `cache.step` == 0, so the sort decides the ordering.
            if (gate_batch_kv_len) {
                var caches_buf: [MAX_BATCH_GROUP]*const KVCache = undefined;
                for (group, 0..) |g, i| {
                    caches_buf[i] = &g.cache;
                }
                fillGroupPadWasteKvLens(caches_buf[0..group.len], group[0].model.config, 1, kv_lens[0..group.len]);
                ctx_max = groupCtxMax(caches_buf[0..group.len]);
                // Stable insertion sort, ascending, slots and lengths moving together.
                var i: usize = 1;
                while (i < group.len) : (i += 1) {
                    const slot_i = group[i];
                    const len_i = kv_lens[i];
                    var j = i;
                    while (j > 0 and kv_lens[j - 1] > len_i) : (j -= 1) {
                        group[j] = group[j - 1];
                        kv_lens[j] = kv_lens[j - 1];
                    }
                    group[j] = slot_i;
                    kv_lens[j] = len_i;
                }
            } else {
                std.sort.pdq(*Slot, group, {}, struct {
                    fn lt(_: void, a: *Slot, b: *Slot) bool {
                        return a.cache.step < b.cache.step;
                    }
                }.lt);
                for (group, 0..) |g, i| kv_lens[i] = @intCast(g.cache.step);
            }
            const keep = groupKeepCount(kv_lens[0..group.len], ctx_max);
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
    use_pld: bool,
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
    enable_pld: bool,
) SpecInitWiring {
    if (owns_module_state) return .{
        // A module-owned arch that CAN rewind its state across a verify runs
        // its own MTP head like any other target. PLD and the drafters
        // stay off regardless: the drafters need a bound sidecar (DflashModel
        // .bind refuses these archs anyway), and PLD's win case is
        // echo-shaped traffic the n-gram gate rarely opens on a
        // head this cheap — neither has been measured on this family, and an
        // unmeasured spec mode is worse than none. Image requests ride the
        // head too: `qwen4MtpForward` takes the slot's M-RoPE table.
        .use_mtp = module_spec_rollback and enable_mtp and has_mtp,
        .use_dflash = false,
        .use_pld = false,
    };
    // enable_drafter is the request-level "assistant sidecar" switch for BOTH
    // sidecar kinds; the loader guarantees at most one of drafter/dflash is
    // loaded per model. Priority: dflash > MTP > PLD — a
    // loaded DFlash sidecar is an explicit choice (`--drafter` or the pack's
    // own `drafter/`) while the MTP head ships with the checkpoint;
    // `--no-drafter` (or `enable_drafter:false`) hands the round back to MTP.
    const use_dflash = enable_drafter and has_dflash;
    const use_mtp = !use_dflash and enable_mtp and has_mtp;
    return .{
        .use_mtp = use_mtp,
        .use_dflash = use_dflash,
        .use_pld = !use_mtp and !use_dflash and enable_pld,
    };
}

pub fn glmNativeDflashArmed(
    is_glm5: bool,
    enable_drafter: bool,
    has_dflash: bool,
    sampling: SamplingParams,
    logprobs_n: u32,
) bool {
    return is_glm5 and enable_drafter and has_dflash and generate_mod.glmDflashEligible(sampling, logprobs_n);
}

/// Resolve the request-level switch shared by the classic external drafter
/// and DFlash. Vision stays guarded: placeholder ids alone do not tell an
/// external sidecar how to position image tokens.
pub fn assistantSidecarEnabledForRequest(requested: bool, has_vision: bool) bool {
    return requested and !has_vision;
}

test "assistant sidecar is refused for image requests" {
    try testing.expect(assistantSidecarEnabledForRequest(true, false));
    try testing.expect(!assistantSidecarEnabledForRequest(false, false));
    try testing.expect(!assistantSidecarEnabledForRequest(true, true));
}

/// Which spec mode a decode tick drives for a slot.
pub const SpecTickMode = enum { mtp, dflash, pld, regular };

/// Pure decode-tick dispatch decision. The slot flags carry the REQUEST's
/// wish; the generator-side values carry what `Generator.initWithOptions`
/// actually armed. Every arm must require BOTH: dispatching on the slot flag
/// alone put PLD verify forwards through a trunk init had disarmed — mtp/drafter
/// had their generator-state conjunct (`gen.mtp != null`), model-less PLD did
/// not, so runPrefill's `use_pld=false` shaped init options while every tick
/// still called `gen.nextPld`.
pub fn specTickMode(
    slot_enable_mtp: bool,
    gen_has_mtp: bool,
    slot_enable_drafter: bool,
    gen_has_dflash: bool,
    slot_enable_pld: bool,
    gen_pld_enabled: bool,
) SpecTickMode {
    if (slot_enable_drafter and gen_has_dflash) return .dflash;
    if (slot_enable_mtp and gen_has_mtp) return .mtp;
    if (slot_enable_pld and gen_pld_enabled) return .pld;
    return .regular;
}

/// Drive one Generator step (regular / PLD / drafter) and push emitted
/// tokens into the slot's output ring. Mirrors the existing
/// `StreamingTokenStream` adapter contract: 0..N tokens per call, with EOS
/// stopping the slot but NOT being emitted.
fn plannerOutputClock(slot: *Slot, gen: *Generator) void {
    const now = gen.timer.read();
    slot.mtp_publish_gap_ms = if (slot.mtp_publish_ns > 0 and now >= slot.mtp_publish_ns) @as(f32, @floatFromInt(now - slot.mtp_publish_ns)) / std.time.ns_per_ms else 0;
    slot.mtp_publish_ns = now;
    if (Planner.enabled() and gen.mtp_planner_owned) gen.mtp_planner_max_gap_ms = @max(gen.mtp_planner_max_gap_ms, slot.mtp_publish_gap_ms);
}

fn publishSpeculativeBlock(sch: *Scheduler, slot: *Slot, gen: *Generator, tokens: []const u32) void {
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
            return;
        }
        if (generate_mod.isEosId(t, slot.eos_token_ids)) {
            finishSlot(sch, slot, "stop");
            return;
        }
        slot.pushToken(t);
        slot.completion_tokens += 1;
        if (t != 0) slot.was_pad_only = false;
        if (slot.completion_tokens >= slot.max_tokens) {
            finishSlot(sch, slot, "length");
            return;
        }
    }
    std.debug.assert(slot.completion_tokens == gen.completion_tokens);
}

/// Every single-slot decode tick funnels through here, so a decode-time MLX failure is
/// attributed to the slot whose forward raised it instead of the next request's prefill.
/// The repetition-loop guard every decode tick runs first. True = the slot was finished here.
fn loopGuardTick(sch: *Scheduler, slot: *Slot, gen: *Generator) !bool {
    // Stop a runaway repetition loop before generating more. Some models (seen
    // on Gemma 4 12B after a large/confusing tool result) collapse into spamming
    // one short cycle — e.g. the thinking opener `<|channel>thought` — forever;
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

/// A thinking budget at its limit: commit the early-stop line and the closer
/// through the model this tick, and decode the answer regular from here on
/// (`spec_disable_reason = .think_bound`). True = the slot's tick is spent.
fn thinkBoundTick(sch: *Scheduler, slot: *Slot, gen: *Generator) !bool {
    const tb = gen.sampling.think_bound orelse return false;
    tb.observe(gen.generated_ids.items);
    if (!tb.due()) return false;
    tb.fired = true;
    if (!generate_mod.forcedBoundaryCanContinue(gen.completion_tokens, gen.max_tokens, tb.forced.len + 1)) {
        log.warn("[think-bound] budget {d} reached with no room to close the thought (max_tokens {d})\n", .{ tb.budget, gen.max_tokens });
        return false;
    }
    const r = try gen.commitForcedTokens(slot.allocator, tb.forced);
    defer slot.allocator.free(r.emitted);
    for (r.emitted) |t| {
        slot.pushToken(t);
        slot.completion_tokens += 1;
        if (t != 0) slot.was_pad_only = false;
    }
    std.debug.assert(slot.completion_tokens == gen.completion_tokens);
    if (r.stopped) {
        finishSlot(sch, slot, gen.finish_reason);
        return true;
    }
    gen.spec_disabled_runtime = true;
    gen.spec_disable_reason = .think_bound;
    log.info("[think-bound] reasoning budget {d} reached at {d} generated tokens; thought closed\n", .{ tb.budget, gen.generated_ids.items.len });
    return true;
}

fn thinkBoundFired(gen: *const Generator) bool {
    const tb = gen.sampling.think_bound orelse return false;
    return tb.fired;
}

fn runSingleDecodeTick(sch: *Scheduler, slot: *Slot) !void {
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
    // generators handle the disabled fallback internally, and `nextPld`'s
    // disabled branch is also where the mid-request RE-ENABLE check lives
    // (bypassing it pinned PLD off for the rest of the request even when the
    // generated tail turned echo-heavy).
    const tick_mode: SpecTickMode = if ((Planner.enabled() and slot.planner_force_plain) or thinkBoundFired(gen)) .regular else specTickMode(
        slot.enable_mtp,
        gen.mtp != null,
        slot.enable_drafter,
        gen.dflash != null,
        slot.enable_pld,
        gen.pld_enabled,
    );
    if (tick_mode == .mtp) {
        gen.mtp_group_cap = 0;
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

    if (tick_mode == .pld) {
        const result = try gen.nextPld(slot.allocator, slot.pld_draft_len, slot.pld_key_len);
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
    var lp_take: ?generate_mod.LogprobResult = null;
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
    const source = @embedFile("scheduler.zig");
    const start = std.mem.indexOf(u8, source, "fn commitSlotIfApplicable(") orelse return error.MissingCommitSlot;
    const end = std.mem.indexOfPos(u8, source, start + 1, "\nfn finishSlot(") orelse return error.MissingFinishSlot;
    const body = source[start..end];
    try testing.expect(std.mem.indexOf(u8, body, "dflashContextCoversPrefix(dc.absLen(), total_len)") != null);
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

test "the cleanup drain commits a cancelled slot before deinit" {
    // Class guard: a decode-phase cancel is pulled straight into the cleanup
    // queue by `complete()` and NEVER passes through finishSlot — without a
    // commit in the drain, its prompt+generated KV dies with the slot. The
    // commit must run BEFORE deinit (the snapshot refcount-shares live
    // buffers; after deinit they are freed).
    const source = @embedFile("scheduler.zig");
    const drain_start = std.mem.indexOf(u8, source, "for (cleanup_batch[0..cleanup_n])") orelse return error.MissingCleanupDrain;
    const region = source[drain_start..@min(drain_start + 1800, source.len)];
    const commit_pos = std.mem.indexOf(u8, region, "commitSlotIfApplicable") orelse return error.DrainDoesNotCommit;
    const deinit_pos = std.mem.indexOf(u8, region, ".deinit()") orelse return error.MissingDeinit;
    try testing.expect(commit_pos < deinit_pos);
    // The SSD tier has no finishSlot flush on this path — the drain must
    // flush what it just committed itself.
    try testing.expect(std.mem.indexOf(u8, region, "flushPendingDisk") != null);
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

test "the inference loop parks without holding the sleep-inhibition assertion" {
    // Pin release < park < acquire and startup acquire < load.
    const source = @embedFile("scheduler.zig");
    const start = std.mem.indexOf(u8, source, "fn inferenceLoop(") orelse return error.MissingInferenceLoop;
    const end = std.mem.indexOfPos(u8, source, start + 1, "\nfn ") orelse return error.MissingInferenceLoopEnd;
    const body = source[start..end];
    const drop = std.mem.indexOf(u8, body, "sleep_inhibit.setActive(false);") orelse return error.MissingSleepRelease;
    const park = std.mem.indexOf(u8, body, "sch.queue_cond.waitUncancelable(sch.io, &sch.queue_mu);") orelse return error.MissingPark;
    const hold = std.mem.indexOfPos(u8, body, park, "sleep_inhibit.setActive(true);") orelse return error.MissingSleepAcquire;
    try testing.expect(drop < park);
    try testing.expect(park < hold);
    try testing.expect(std.mem.indexOf(u8, body, "while (!hasWorkPendingLocked(sch)") != null);
    const boot_load = std.mem.indexOf(u8, body, "doLoadOnInferenceThread(sch, params)") orelse return error.MissingStartupLoad;
    const boot_arm = std.mem.indexOf(u8, body, "sleep_inhibit.setActive(true);") orelse return error.MissingSleepAcquire;
    try testing.expect(boot_arm < boot_load);
    try testing.expect(std.mem.indexOf(u8, body, "defer sleep_inhibit.release();") != null);
}

test "commitSlotIfApplicable routes a Generator-less slot to the cancelled-prefill commit" {
    // Prefill abort: `Generator.initWithOptions` throws error.Cancelled from
    // its chunk loop, so `slot.legacy_gen` is never assigned — the legacy
    // `else return` silently dropped every chunk that DID forward. The
    // cancelled-prefill arm must be reachable from commitSlotIfApplicable,
    // and hybrids are excluded inside it (their stride checkpoints die with
    // the failed Generator init; a checkpoint-less hybrid entry restores as
    // a cold miss while still occupying an LRU slot).
    const source = @embedFile("scheduler.zig");
    const start = std.mem.indexOf(u8, source, "fn commitSlotIfApplicable(") orelse return error.MissingCommitSlot;
    const end = std.mem.indexOfPos(u8, source, start + 1, "\nfn finishSlot(") orelse return error.MissingFinishSlot;
    const body = source[start..end];
    const route_pos = std.mem.indexOf(u8, body, "commitCancelledPrefillSlot") orelse return error.MissingRoute;
    // `was_pad_only` initializes TRUE and only flips on the first pushed
    // token, so a guard on it ABOVE the Generator-less arm makes that arm
    // unreachable (live 2026-08-22: "prefill aborted" logged, nothing
    // committed, full re-prefill on retry).
    const pad_pos = std.mem.indexOf(u8, body, "slot.was_pad_only") orelse return error.MissingPadGuard;
    try testing.expect(route_pos < pad_pos);

    const cp_start = std.mem.indexOf(u8, source, "fn commitCancelledPrefillSlot(") orelse return error.MissingCancelledPrefillFn;
    const cp_end = std.mem.indexOfPos(u8, source, cp_start + 1, "\nfn ") orelse return error.MissingCancelledPrefillEnd;
    const cp_body = source[cp_start..cp_end];
    // Hybrid gate: KV-only hybrid entries restore as cold misses and are
    // declined; with salvaged checkpoints (handed off by initWithOptions on
    // error.Cancelled) a cancelled hybrid prefill commits like a normal one.
    try testing.expect(std.mem.indexOf(u8, cp_body, "slot.ssm_entries != null and salvage.checkpoints.len == 0") != null);
    try testing.expect(std.mem.indexOf(u8, cp_body, "cancelledPrefillCommitLen") != null);
    // The authoritative commit length is the sink's `forwarded` counter —
    // `cache.step` only advances when Generator init COMPLETES, so it reads
    // 0 on every aborted prefill (found live: step=0 while pos=1536).
    try testing.expect(std.mem.indexOf(u8, cp_body, "salvage.forwarded") != null);
    try testing.expect(std.mem.indexOf(u8, cp_body, "commitWithMediaState") != null);
}

test "Generator.initWithOptions hands off checkpoints on cancel" {
    // The chunk-loop cancel path must MOVE the captured stride checkpoints
    // into the sink before returning error.Cancelled — they die with the
    // failed construction otherwise, and hybrid cancelled-prefill commits
    // are impossible. Anchored on the prefill loop's abandoned-request
    // abort comment so a decode-loop cancel check can't satisfy it.
    const source = @embedFile("generate.zig");
    const anchor = std.mem.indexOf(u8, source, "Abandoned-request abort") orelse return error.MissingAbortComment;
    const region = source[anchor..@min(anchor + 2400, source.len)];
    try testing.expect(std.mem.indexOf(u8, region, "cancelled_checkpoint_sink") != null);
    try testing.expect(std.mem.indexOf(u8, region, "error.Cancelled") != null);
}

test "Slot.deinit frees unconsumed cancelled-prefill salvage" {
    // Ownership discipline: the sink holds checkpoint arrays allocated on
    // the inference thread; anything commitCancelledPrefillSlot did not
    // consume must die with the slot, not leak.
    const source = @embedFile("scheduler.zig");
    const start = std.mem.indexOf(u8, source, "pub fn deinit(self: *Slot)") orelse return error.MissingSlotDeinit;
    const end = std.mem.indexOfPos(u8, source, start + 1, "pub fn deinit(self: *Scheduler)") orelse return error.MissingSchedulerDeinit;
    const body = source[start..end];
    try testing.expect(std.mem.indexOf(u8, body, "cancelled_prefill.deinit()") != null);
}

test "runPrefill clears restored spec-cache ownership BEFORE Generator.initWithOptions (issue #266)" {
    // Generator.initWithOptions ADOPTS the hot-cache-restored DFlash/MTP
    // caches and frees them via its own errdefers on any failure past the
    // adoption point — a mid-prefill client disconnect throws
    // error.Cancelled from its chunk loop. MtpCacheRef/DflashCtx hold their
    // KVCache BY VALUE, so runPrefill's own errdefers then walked the same
    // entries slice + mlx handles a second time: SIGSEGV in
    // KVCache.deinit -> freeKVEntry (issue #266, disconnect storms on long
    // agent prompts). Ownership transfers AT THE CALL, so the locals must
    // be cleared before the try, never after it.
    const source = @embedFile("scheduler.zig");
    const start = std.mem.indexOf(u8, source, "fn runPrefill(") orelse return error.MissingRunPrefill;
    const end = std.mem.indexOfPos(u8, source, start + 1, "\nfn ") orelse return error.MissingRunPrefillEnd;
    const body = source[start..end];
    const call = std.mem.indexOf(u8, body, "try Generator.initWithOptions(") orelse return error.MissingInitCall;
    const dfl = std.mem.indexOf(u8, body, "dflash_restored = null") orelse return error.MissingDflashClear;
    const mtp = std.mem.indexOf(u8, body, "mtp_restored = null") orelse return error.MissingMtpClear;
    try testing.expect(dfl < call);
    try testing.expect(mtp < call);
}

test "DFlash gate policy follows effective block width and resolved thinking" {
    // block_size includes the always-emitted anchor, so block 16 has 15 draft
    // positions and block 7 has 6. The M5 calibration is normalized to that
    // actual draft width instead of being imposed as an absolute threshold on
    // machines without the wide verification lane.
    try testing.expectApproxEqAbs(@as(f32, 2.0), dflashGateMinimum(16, false, false), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), dflashGateMinimum(16, true, false), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.8), dflashGateMinimum(7, false, false), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.4), dflashGateMinimum(7, true, false), 0.0001);
    try testing.expectEqual(@as(f32, 0), dflashGateMinimum(1, false, false));

    // A SPARSE target's verify reads every expert its block routes to, so the
    // dense width scaling under-bars it: at block 5 the scaled value is 0.53
    // and LFM2.5-8B-A1B measured break-even at 1.63 accepted/round. The floor
    // binds there and in the thinking arm, and never lowers a bar the width
    // scaling already set higher.
    try testing.expectApproxEqAbs(@as(f32, 1.8), dflashGateMinimum(5, false, true), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 1.8), dflashGateMinimum(5, true, true), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 2.0), dflashGateMinimum(16, false, true), 0.0001);
    try testing.expectEqual(@as(f32, 0), dflashGateMinimum(1, false, true));
}

test "every server scheduler path forwards resolved thinking to the DFlash gate" {
    const source = @embedFile("server.zig");

    // All direct streaming submissions, plus the shared non-streaming submit,
    // must populate the field. Text completions resolve it explicitly false.
    var submit_pos: usize = 0;
    var submits: usize = 0;
    while (std.mem.indexOfPos(u8, source, submit_pos, "sch.submit(.{")) |start| {
        const end = std.mem.indexOfPos(u8, source, start, "});") orelse return error.UnclosedSchedulerSubmit;
        try testing.expect(std.mem.indexOf(u8, source[start..end], ".enable_thinking =") != null);
        submits += 1;
        submit_pos = end + 3;
    }
    try testing.expectEqual(@as(usize, 5), submits);

    // Positional calls into the non-streaming wrapper must pass the resolved
    // value; the raw text-completion path is the sole explicit false arm.
    var lines = std.mem.splitScalar(u8, source, '\n');
    var calls: usize = 0;
    while (lines.next()) |line| {
        if (std.mem.indexOf(u8, line, "nonStreamingViaScheduler(") == null) continue;
        if (std.mem.indexOf(u8, line, "fn nonStreamingViaScheduler(") != null) continue;
        calls += 1;
        try testing.expect(std.mem.indexOf(u8, line, "enable_thinking") != null or
            std.mem.indexOf(u8, line, ", false, false, use_pld") != null);
    }
    try testing.expectEqual(@as(usize, 4), calls);
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
fn slotMimoMtpCrowdable(slot: *const Slot) bool {
    if (!mtpGroupEnabled()) return false;
    const gen = if (slot.legacy_gen) |*g| g else return false;
    if (!slot.enable_mtp or gen.mtp == null or gen.mtp_cache == null or !gen.has_last_hidden) return false;
    if (gen.spec_disabled_runtime or gen.mtp_serial_left > 0 or gen.mtp_serial_exit != .none) return false;
    if (slot.sampling.constraint != null or slot.logprobs_n > 0) return false;
    const t = slot.model.transformer orelse return false;
    if (!t.supportsBatchedMimoDecode()) return false;
    return specTickMode(slot.enable_mtp, true, slot.enable_drafter, gen.dflash != null, slot.enable_pld, gen.pld_enabled) == .mtp;
}

/// Step costs of a grouped GLM forward in ms (rows ubench, glm5-arch#concurrency): fixed, per
/// row, and one assistant draft.
pub const GlmRowCost = struct { fixed: f32 = 25, row: f32 = 14.5, draft: f32 = 5 };

/// Rows one grouped GLM tick carries: the verify kernels' four-row tile.
const glm_tick_rows = 4;

/// Draft nodes (0, 1 or 2) for the best drafter among `slots` single rows under the four-row cap:
/// a tree pays while its expected accepted tokens, priced at the group's ms per token, beat its
/// rows and draft. `rate` is that request's accepted share of drafts (`Generator.glm_draft_rate`).
pub fn glmTreeNodes(slots: usize, rate: f32, cost: GlmRowCost) usize {
    if (slots == 0) return 0;
    const n: f32 = @floatFromInt(slots);
    const per_token = (cost.fixed + n * cost.row) / n;
    var best: usize = 0;
    var best_gain: f32 = 0;
    var accepted: f32 = 0;
    var power: f32 = 1;
    for (1..@min(2, glm_tick_rows -| slots) + 1) |k| {
        power *= rate;
        accepted += power;
        const gain = accepted * per_token - (@as(f32, @floatFromInt(k)) * cost.row + cost.draft);
        if (gain > best_gain) {
            best = k;
            best_gain = gain;
        }
    }
    return best;
}

/// Does another slot of this one's model decode this tick (a drafting GLM slot or a plain batchable one)?
fn glmRowsInCompany(slot: *const Slot, drafting: []const *Slot, plain: []const *Slot) bool {
    var company: usize = 0;
    for (drafting) |other| company += @intFromBool(other.model == slot.model);
    for (plain) |other| company += @intFromBool(other.model == slot.model);
    return company >= 2;
}

/// A DFlash2 GLM slot that can ride a plain grouped tick: its rounds only need the row committed
/// with the assistant's taps (`Generator.glmRowCommit`).
fn slotGlmRowable(slot: *const Slot) bool {
    const gen = if (slot.legacy_gen) |*g| g else return false;
    if (!gen.glm_dflash_native or gen.dflash_ctx == null or gen.done) return false;
    if (slot.sampling.constraint != null or slot.logprobs_n > 0 or generate_mod.penaltyActive(slot.sampling)) return false;
    const t = slot.model.transformer orelse return false;
    return t.supportsBatchedGlmRows();
}

fn slotMtpGroupable(slot: *const Slot) bool {
    if (!mtpGroupEnabled()) return false;
    const gen = if (slot.legacy_gen) |*g| g else return false;
    if (!slot.enable_mtp or gen.mtp == null or gen.mtp_cache == null or !gen.has_last_hidden) return false;
    if (gen.mtp.?.moduleOwned()) return false;
    if (gen.spec_disabled_runtime or gen.mtp_serial_left > 0 or gen.mtp_serial_exit != .none) return false;
    if (gen.ctx.ssm_entries == null) return false;
    if (slot.sampling.constraint != null or slot.logprobs_n > 0) return false;
    const t = slot.model.transformer orelse return false;
    if (!t.supportsBatchedGdnDecode()) return false;
    return specTickMode(slot.enable_mtp, true, slot.enable_drafter, gen.dflash != null, slot.enable_pld, gen.pld_enabled) == .mtp;
}

const Planner = @import("mtp_group_planner.zig");

fn plannerShape(slots: []const *Slot, widths: []const u8) group_cost_mod.GroupShape {
    const xfm = slots[0].model.transformer.?;
    var shape = group_cost_mod.GroupShape{ .n = @intCast(slots.len), .kv_format = cacheCostFormat(slots[0].cache.config) };
    for (slots, widths, 0..) |slot, width, i| {
        const gen = &slot.legacy_gen.?;
        const kv: u8 = @intCast(xfm.round_cost.bucketOf(@intCast(slot.moe_seq_offset)));
        if (width == 0) {
            shape.rows[i] = group_cost_mod.GroupShape.row(0, 0, 0, kv, 0, 0, group_cost_mod.GroupShape.samplingMode(false, false, false, !generate_mod.isGreedyTemperature(gen.sampling.temperature)));
        } else {
            const off = Generator.mtpRoundOff0(gen.mtp_hist_stash, gen.mtp_cache.?.step());
            const history = 1 + if (gen.mtp_hist_stash) |stash| stash.n else @as(usize, 0);
            shape.head_format = cacheCostFormat(xfm.qwen4_mtp.?.cache.config);
            shape.rows[i] = group_cost_mod.GroupShape.row(width, width, width, kv, @intCast(xfm.round_cost.bucketOf(@intCast(off))), @intCast(@min(history, round_cost_mod.MAX_WIDTH + 2)), gen.mtpPlannerMode());
        }
    }
    return shape;
}

fn plannerInputSame(a: group_cost_mod.GroupShape, b: group_cost_mod.GroupShape) bool {
    var aa = a;
    var bb = b;
    aa.attention = 0;
    aa.moe = 0;
    aa.head_calls = 0;
    bb.attention = 0;
    bb.moe = 0;
    bb.head_calls = 0;
    return std.meta.eql(aa, bb);
}

fn plannerPriceTransition(previous: u8, width: u8, pending: bool) bool {
    return width > 0 and (pending or previous != width);
}

fn plannerRecurringInputSame(a: group_cost_mod.GroupShape, b: group_cost_mod.GroupShape) bool {
    if (a.n != b.n or a.width() == 0 or b.width() == 0) return false;
    inline for (.{ @as(u6, 24), @as(u6, 32) }) |shift| {
        var a_max: u8 = 0;
        var b_max: u8 = 0;
        for (a.rows[0..a.n], b.rows[0..b.n]) |arow, brow| {
            a_max = @max(a_max, group_cost_mod.GroupShape.part(arow, shift));
            b_max = @max(b_max, group_cost_mod.GroupShape.part(brow, shift));
        }
        if (a_max != b_max) return false;
    }
    var aa = a;
    var bb = b;
    const recurring_mask = ~((@as(u64, 0xff) << 24) | (@as(u64, 0xff) << 32) | (@as(u64, 0xff) << 40));
    for (aa.rows[0..aa.n], bb.rows[0..bb.n]) |*arow, *brow| {
        if (group_cost_mod.GroupShape.part(arow.*, 40) < 2 or group_cost_mod.GroupShape.part(brow.*, 40) < 2) return false;
        arow.* &= recurring_mask;
        brow.* &= recurring_mask;
    }
    return plannerInputSame(aa.canonical(), bb.canonical());
}

const PlannerPrices = struct {
    slots: []const *Slot,
    geometry: group_cost_mod.GroupShape,

    fn init(slots: []const *Slot) PlannerPrices {
        var widths: [Planner.MAX_ROWS]u8 = @splat(0);
        for (slots, 0..) |slot, i| widths[i] = @intFromBool(slot.legacy_gen.?.mtp != null and slot.legacy_gen.?.mtp_cache != null);
        return .{ .slots = slots, .geometry = plannerShape(slots, widths[0..slots.len]) };
    }

    fn keyFor(self: *const PlannerPrices, slots: []const *Slot, widths: []const u8) group_cost_mod.GroupShape {
        var key = self.geometry;
        key.n = @intCast(slots.len);
        key.rows = @splat(0);
        key.head_format = 0;
        for (slots, widths, 0..) |slot, width, i| {
            for (self.slots, 0..) |original, j| {
                if (original != slot) continue;
                const code = self.geometry.rows[j];
                key.rows[i] = if (width == 0) group_cost_mod.GroupShape.row(0, 0, 0, group_cost_mod.GroupShape.part(code, 24), 0, 0, group_cost_mod.GroupShape.part(code, 48) & 8) else (code & ~@as(u64, 0xffffff)) | @as(u64, width) * 0x10101;
                if (width > 0) key.head_format = self.geometry.head_format;
                break;
            }
        }
        return key.canonical();
    }

    fn sample(self: *const PlannerPrices, slots: []const *Slot, widths: []const u8, trusted: bool) ?Planner.Price {
        if (slots.len == 0) return .{ .ms = 0, .samples = std.math.maxInt(u32) };
        const key = self.keyFor(slots, widths);
        const table = &slots[0].model.transformer.?.mtp_group_cost;
        var selected: ?Planner.Price = null;
        var score: f32 = 0;
        var max_gap: f32 = 0;
        var mean_upper: f32 = 0;
        var mean_lower: f32 = std.math.inf(f32);
        const plain = key.width() == 0;
        for (table.shapes) |entry| {
            if (entry.cell.n == 0 or !plannerInputSame(key, entry.shape)) continue;
            if (trusted and entry.cell.n < Planner.MIN_SAMPLES) continue;
            const deviation = 2 * @sqrt(entry.variance);
            const mean_deviation = 2 * @sqrt(entry.variance * entry.mean_weight_sq);
            mean_upper = @max(mean_upper, entry.cell.ms + mean_deviation);
            mean_lower = @min(mean_lower, entry.cell.ms - mean_deviation);
            const value = if (plain) entry.cell.ms - deviation else entry.cell.ms + deviation;
            for (entry.gap_ms[0..key.n]) |gap| max_gap = @max(max_gap, gap);
            if (selected == null or (if (plain) value < score else value > score)) {
                score = value;
                selected = .{ .ms = entry.cell.ms, .variance = entry.variance, .samples = entry.cell.n };
            }
        }
        if (selected) |*value| {
            value.max_gap_ms = max_gap;
            if (!plain) value.mean_upper_ms = mean_upper;
            if (plain) value.mean_lower_ms = mean_lower;
            return selected;
        }
        if (plain) return null;

        var pooled_n: u32 = 0;
        var sum: f64 = 0;
        var sum_sq: f64 = 0;
        var mean_sum_sq: f64 = 0;
        max_gap = 0;
        for (table.shapes) |entry| {
            if (entry.cell.n == 0 or !plannerRecurringInputSame(key, entry.shape)) continue;
            const n: f64 = @floatFromInt(entry.cell.n);
            const mean: f64 = entry.cell.ms;
            pooled_n +|= entry.cell.n;
            sum += n * mean;
            sum_sq += n * (@as(f64, entry.variance) + mean * mean);
            // History-to-history differences remain uncertainty in a pooled prediction.
            mean_sum_sq += n * (@as(f64, entry.variance) * entry.mean_weight_sq + mean * mean);
            for (entry.gap_ms[0..key.n]) |gap| max_gap = @max(max_gap, gap);
        }
        const required: u32 = if (trusted) Planner.MIN_SAMPLES else 1;
        if (pooled_n < required) return null;
        const n: f64 = @floatFromInt(pooled_n);
        const mean = sum / n;
        const variance = @max(0, sum_sq / n - mean * mean);
        const mean_variance = @max(0, mean_sum_sq / n - mean * mean);
        return .{ .ms = @floatCast(mean), .variance = @floatCast(variance), .samples = pooled_n, .max_gap_ms = max_gap, .mean_upper_ms = @floatCast(mean + 2 * @sqrt(mean_variance)) };
    }

    pub fn price(self: *PlannerPrices, widths: []const u8) ?Planner.Price {
        var positive: [Planner.MAX_ROWS]*Slot = undefined;
        var drafts: [Planner.MAX_ROWS]u8 = undefined;
        var zeros: [Planner.MAX_ROWS]*Slot = undefined;
        const zero_widths: [Planner.MAX_ROWS]u8 = @splat(0);
        var pn: usize = 0;
        var zn: usize = 0;
        for (self.slots, widths) |slot, width| {
            if (width == 0) {
                zeros[zn] = slot;
                zn += 1;
            } else {
                positive[pn] = slot;
                drafts[pn] = width;
                pn += 1;
            }
        }
        const plain = self.sample(zeros[0..zn], zero_widths[0..zn], true) orelse return null;
        const spec = self.sample(positive[0..pn], drafts[0..pn], true) orelse return null;
        const deviation = @sqrt(plain.variance) + @sqrt(spec.variance);
        return .{ .ms = plain.ms + spec.ms, .variance = deviation * deviation, .samples = @min(plain.samples, spec.samples), .max_gap_ms = @max(plain.max_gap_ms, spec.max_gap_ms), .mean_upper_ms = if (spec.mean_upper_ms) |bound| plain.ms + 2 * @sqrt(plain.variance) + bound else null, .mean_lower_ms = if (pn == 0) plain.mean_lower_ms else null };
    }

    fn probeEstimate(self: *PlannerPrices, widths: []const u8, plain: Planner.Price) f32 {
        if (self.sample(self.slots, widths, false)) |value| return value.ms + 2 * @sqrt(value.variance);
        var widest: u8 = 0;
        for (widths) |width| widest = @max(widest, width);
        var lower: [Planner.MAX_ROWS]u8 = undefined;
        for ([_]u8{ 2, 1 }) |cap| {
            if (cap >= widest) continue;
            for (widths, 0..) |width, i| lower[i] = @min(width, cap);
            if (self.sample(self.slots, lower[0..widths.len], false)) |value| return scaledProbeEstimate(value, widest, cap);
        }
        if (self.slots.len >= 4 and self.slots.len % 2 == 0) {
            const half = self.slots.len / 2;
            if (self.sample(self.slots[0..half], widths[0..half], false)) |a| {
                if (self.sample(self.slots[half..], widths[half..], false)) |b| return a.ms + b.ms + 2 * (@sqrt(a.variance) + @sqrt(b.variance));
            }
        }
        return self.probeFallbackEstimate(widths, plain);
    }

    fn probeFallbackEstimate(_: *PlannerPrices, widths: []const u8, plain: Planner.Price) f32 {
        var widest: u8 = 0;
        for (widths) |width| widest = @max(widest, width);
        const upper = plain.ms + 2 * @sqrt(plain.variance);
        if (widest <= 2) return 2 * upper;
        return 1.5 * upper * @as(f32, @floatFromInt(widest + 1));
    }

    fn scaledProbeEstimate(value: Planner.Price, widest: u8, measured: u8) f32 {
        const ratio = @as(f32, @floatFromInt(widest + 1)) / @as(f32, @floatFromInt(measured + 1));
        return value.ms * ratio;
    }
};

var planner_engaged_logged = false;
var planner_calibration_block_logged = false;
var planner_depth_two_reject_logged = false;

const ProbeSource = enum { calibrate, recovery };

/// Widths for a calibration or recovery probe round, priced against the plain
/// tick. False leaves `decision` untouched: no candidate, no price, or over budget.
fn planProbe(
    prices: *PlannerPrices,
    active: []*Slot,
    rows: []const Planner.Row,
    decision: *Planner.Decision,
    comptime source: ProbeSource,
) bool {
    const zero_widths: [Planner.MAX_ROWS]u8 = @splat(0);
    var candidate: [Planner.MAX_ROWS]u8 = @splat(0);
    for (active, rows, 0..) |slot, row, i| {
        const gen = &slot.legacy_gen.?;
        candidate[i] = switch (source) {
            .calibrate => Planner.probeWidth(gen.mtp_planner_probes, gen.max_tokens -| gen.completion_tokens, row.cap),
            .recovery => gen.mtp_planner_recovery.width(gen.completion_tokens, gen.max_tokens -| gen.completion_tokens, row.cap, gen.mtp_planner_probes),
        } orelse return false;
    }
    const plain = prices.price(zero_widths[0..active.len]) orelse return false;
    const estimate = prices.probeEstimate(candidate[0..active.len], plain);
    var latency_limit: f32 = std.math.floatMax(f32);
    var available = true;
    for (rows) |row| {
        latency_limit = @min(latency_limit, row.latency_ms);
        if (estimate > row.latency_ms) available = false;
    }
    if (!available) {
        if (source == .calibrate and !planner_calibration_block_logged) {
            planner_calibration_block_logged = true;
            log.info("[mtp-planner] calibration blocked widths={any} predicted_ms={d:.2} latency_ms={d:.2}\n", .{ candidate[0..active.len], estimate, latency_limit });
        }
        return false;
    }
    decision.widths = candidate;
    decision.upper_ms = estimate;
    return true;
}

fn tryPlannerTick(sch: *Scheduler, active: []*Slot) anyerror!bool {
    if (!Planner.enabled() or active.len == 0 or active.len > Planner.MAX_ROWS) return false;
    const xfm = active[0].model.transformer orelse return false;
    if (xfm.qwen4 == null or xfm.qwen4_mtp == null) return false;
    if (active.len == 1) {
        const gen = if (active[0].legacy_gen) |*g| g else return false;
        if (!gen.mtp_planner_owned) return false;
    }
    var rows: [Planner.MAX_ROWS]Planner.Row = undefined;
    var lenses: [Planner.MAX_ROWS]u32 = undefined;
    var has_mtp = false;
    var pending_pipeline = false;
    var pending_draft = false;
    for (active, 0..) |slot, i| {
        if (slot.model != active[0].model or !slot.allow_batch_mtp) return false;
        const gen = if (slot.legacy_gen) |*g| g else return false;
        if (gen.ctx.mrope_pos != null) return false;
        pending_pipeline = pending_pipeline or gen.has_pending_logits or gen.has_pending_token;
        pending_draft = pending_draft or gen.mtp_pre_draft != null;
        const why = sch.batchVerdict(slot);
        const mtp_row = why == .spec_active and slotMtpGroupable(slot);
        if (why != .ok and !mtp_row) return false;
        has_mtp = has_mtp or mtp_row;
        const cap = if (mtp_row) @min(@min(if (gen.mtp_depth > 0) gen.mtp_depth else @max(1, gen.mtp_depth_current), Planner.MAX_DEPTH), gen.max_tokens -| gen.completion_tokens -| 1) else 0;
        rows[i] = .{ .history = &gen.mtp_planner_history, .cap = @intCast(cap), .latency_ms = Planner.LATENCY_MS };
        lenses[i] = @intCast(slot.moe_seq_offset);
    }
    if (!has_mtp) return false;
    std.sort.insertion(u32, lenses[0..active.len], {}, std.sort.asc(u32));
    if (active.len > 1 and batchedKvKeepCount(lenses[0..active.len]) != active.len) return false;
    const entry = Planner.entry(pending_pipeline, pending_draft);
    if (entry == .wait_predraft) {
        // Consume the pending rounds without perpetually replacing their successors.
        for (active) |slot| slot.legacy_gen.?.mtp_planner_pending = true;
        return false;
    }
    if (entry == .drain) {
        for (active, rows[0..active.len]) |slot, row| {
            const gen = &slot.legacy_gen.?;
            if (gen.mtp != null) try gen.mtpDetachHead(slot.allocator, true);
            if (row.cap > 0) gen.mtp_planner_owned = true;
            slot.planner_plain_transition = true;
            slot.mtp_plain_tick = row.cap > 0;
        }
        log.info("[mtp-planner] rows={d} action=drain\n", .{active.len});
        try runBatchedDecodeTick(sch, active);
        return true;
    }

    var live: [Planner.MAX_ROWS]*Slot = undefined;
    var count: usize = 0;
    for (active) |slot| {
        const gen = &slot.legacy_gen.?;
        if (try loopGuardTick(sch, slot, gen)) continue;
        if (try gen.checkStop()) {
            finishSlot(sch, slot, gen.finish_reason);
            continue;
        }
        live[count] = slot;
        count += 1;
    }
    if (count != active.len) {
        if (count > 0) try runDecodeTick(sch, live[0..count]);
        return true;
    }
    for (active, rows[0..active.len]) |slot, row| if (row.cap > 0) {
        slot.legacy_gen.?.mtp_planner_owned = true;
    };
    var prices = PlannerPrices.init(active);
    var decision = Planner.choose(rows[0..active.len], &prices);
    var probe = false;
    var recovering = false;
    var probe_rounds: u8 = std.math.maxInt(u8);
    for (active, rows[0..active.len]) |slot, row| if (row.cap > 0) {
        probe_rounds = @min(probe_rounds, slot.legacy_gen.?.mtp_planner_probes);
    };
    if (!planner_depth_two_reject_logged and probe_rounds >= 12 and Planner.shouldProbe(decision, probe_rounds)) {
        var depth_two: [Planner.MAX_ROWS]u8 = @splat(0);
        var expected: f32 = 0;
        for (rows[0..active.len], 0..) |row, i| {
            depth_two[i] = @min(2, row.cap);
            expected += row.history.expected(depth_two[i]);
        }
        planner_depth_two_reject_logged = true;
        log.info("[mtp-planner] depth-two rejected price={any} expected_tokens={d:.2} plain_rate={d:.1}\n", .{ prices.price(depth_two[0..active.len]), expected, decision.plain_rate });
    }
    if (Planner.shouldProbe(decision, probe_rounds)) {
        probe = planProbe(&prices, active, rows[0..active.len], &decision, .calibrate);
    }
    if (!probe and Planner.shouldProbe(decision, probe_rounds) and planProbe(&prices, active, rows[0..active.len], &decision, .recovery)) {
        probe = true;
        recovering = true;
    }
    // A width the range moves no longer matches what a probe was meant to price.
    for (rows[0..active.len], 0..) |row, i| {
        const width: u8 = @intCast(depth_bounds.plannerWidth(decision.widths[i], row.cap, depth_bounds.active));
        if (width != decision.widths[i] or depth_bounds.active.pinned() != null) {
            decision.widths[i] = width;
            probe = false;
            recovering = false;
        }
    }
    var stale: [Planner.MAX_ROWS]bool = undefined;
    for (active, 0..) |slot, row| stale[row] = slot.legacy_gen.?.mtp_hidden_stale;
    const execution = Planner.execution(decision.widths[0..active.len], stale[0..active.len]);
    const prime = execution.prime;
    var positives: [Planner.MAX_ROWS]*Slot = undefined;
    var plains: [Planner.MAX_ROWS]*Slot = undefined;
    const pn = execution.speculative_n;
    const zn = execution.plain_n;
    var changed = false;
    for (active, decision.widths[0..active.len]) |slot, width| {
        const row_changed = slot.planner_last_width != width;
        changed = changed or row_changed;
        slot.planner_price_transition = plannerPriceTransition(slot.planner_last_width, width, slot.planner_price_transition);
        slot.planner_last_width = width;
    }
    for (execution.plain[0..zn], 0..) |row, i| {
        const slot = active[row];
        const gen = &slot.legacy_gen.?;
        slot.planner_plain_transition = gen.mtp_hist_stash != null or gen.mtp_pre_draft != null or prime;
        if (gen.mtp != null) try gen.mtpDetachHead(slot.allocator, true);
        slot.mtp_plain_tick = execution.capture[row];
        plains[i] = slot;
    }
    for (execution.speculative[0..pn], 0..) |row, i| {
        const slot = active[row];
        const gen = &slot.legacy_gen.?;
        gen.mtp_planner_width = decision.widths[row];
        gen.mtp_planner_probe = probe;
        gen.mtp_planner_recovering = recovering;
        positives[i] = slot;
    }
    if (!planner_engaged_logged) {
        planner_engaged_logged = true;
        log.info("[mtp-planner] engaged: margin=5% noise=2sigma latency_ms={d:.1} max_rows=8\n", .{Planner.LATENCY_MS});
    }
    if (changed or prime or probe) log.info("[mtp-planner] rows={d} widths={any} action={s} predicted_ms={d:.2} rate={d:.1} plain={d:.1}\n", .{ active.len, decision.widths[0..active.len], if (prime) "prime" else if (recovering) "recover" else if (probe) "calibrate" else "choose", decision.upper_ms, decision.rate, decision.plain_rate });
    if (pn > 0) {
        try runBatchedMtpHeadTick(sch, positives[0..pn]);
    }
    if (zn > 0) try runBatchedDecodeTick(sch, plains[0..zn]);
    return true;
}

var mtp_group_env: ?bool = null;
fn mtpGroupEnabled() bool {
    if (mtp_group_env) |v| return v;
    const on = if (std.c.getenv("SUSHI_MTP_BATCHED")) |p| !std.mem.eql(u8, std.mem.span(p), "0") else true;
    mtp_group_env = on;
    return on;
}

/// qwen4_exp verify rows are expert bytes, so a batched verify measured no better than
/// solo rounds: its MTP rounds stay solo unless opted in; two interleave, three go plain.
var mtp_batched_qwen4_env: ?bool = null;
fn mtpBatchedQwen4Enabled() bool {
    if (mtp_batched_qwen4_env) |v| return v;
    const on = if (std.c.getenv("SUSHI_MTP_BATCHED_QWEN4")) |p| !std.mem.eql(u8, std.mem.span(p), "0") else false;
    mtp_batched_qwen4_env = on;
    return on;
}

fn mtpQwen4StaySolo(has_qwen4: bool, env_on: bool) bool {
    return has_qwen4 and !env_on;
}

/// Why this group cannot take the merged `[N, S]` verify forward, or null for a shape that
/// may: a qwen4 trunk's batched sub-paths are decode-shaped and uncertified past width 1.
pub fn mergedVerifyDeclineReason(has_qwen4: bool, width: u32) ?[]const u8 {
    if (has_qwen4 and width > 1) return "qwen4 trunk past verify width 1";
    return null;
}

pub const VerifyShape = enum { solo, row_axis, merged };

pub fn verifyShapeFor(rows: usize, has_qwen4: bool, width: u32) VerifyShape {
    if (rows <= 1) return .solo;
    if (mergedVerifyDeclineReason(has_qwen4, width) != null) return .row_axis;
    return .merged;
}

var merged_verify_decline_logged: bool = false;

fn mtpRoundsStaySolo(slot: *const Slot) bool {
    const t = slot.model.transformer orelse return true;
    // A MiMo verify keeps each row's decode arithmetic only in its own solo forward.
    if (t.config.isMimo()) return true;
    return mtpQwen4StaySolo(t.qwen4 != null, mtpBatchedQwen4Enabled());
}

fn mtpCrowdThresholdFor(slot: *const Slot) usize {
    return if (mtpRoundsStaySolo(slot)) 3 else mtpCrowdThreshold();
}

fn runBatchedMtpHeadTick(sch: *Scheduler, group: []*Slot) !void {
    const cost_xfm = group[0].model.transformer.?;
    cost_xfm.cost_trace_active = cost_xfm.qwen4 != null;
    cost_xfm.cost_attention = 0;
    cost_xfm.cost_moe = 0;
    const head_calls_before = transformer_mod.mtp_head_row_dispatches;
    defer cost_xfm.cost_trace_active = false;
    defer for (group) |slot| {
        if (slot.legacy_gen) |*gen| gen.mtp_batch_head = false;
    };
    var tick_sw = io_util.Stopwatch.init(group[0].io);
    var gens: [MAX_BATCH_GROUP]*generate_mod.Generator = undefined;
    var chains: [MAX_BATCH_GROUP]generate_mod.Generator.MtpPreDraft = undefined;
    var opens: [MAX_BATCH_GROUP]generate_mod.Generator.MtpRoundOpen = undefined;
    var live: [MAX_BATCH_GROUP]*Slot = undefined;
    var n: usize = 0;
    var n_chain: usize = 0;
    defer for (opens[0..n]) |*o| {
        if (o.chain.drafts.len != 0) o.chain.deinit(live[0].allocator);
    };
    defer for (chains[0..n_chain]) |*c| c.deinit(live[0].allocator);

    for (group) |slot| {
        const gen = &slot.legacy_gen.?;
        if (try loopGuardTick(sch, slot, gen)) continue;
        gen.mtp_batch_head = true;
        const begun = gen.mtpRoundBegin(slot.allocator) catch |e| {
            gen.mtp_batch_head = false;
            slot.markError(@errorName(e));
            continue;
        };
        switch (begun) {
            .done => |r| publishMtpResult(sch, slot, gen, r),
            .verify => |st_in| {
                var state = st_in;
                defer state.deinit(slot.allocator);
                gen.mtpRoundVerify(&state) catch |e| {
                    slot.markError(@errorName(e));
                    continue;
                };
                mtpFinishPublish(sch, slot, gen, &state);
            },
            .open => |o| {
                opens[n] = o;
                gens[n] = gen;
                live[n] = slot;
                n += 1;
            },
        }
    }
    if (n == 0) return;

    var depth: u32 = 1;
    for (opens[0..n], 0..) |*o, i| {
        depth = @max(depth, o.chain.m);
        chains[i] = o.chain;
        o.chain = .{
            .plan = chains[i].plan,
            .off0 = 0,
            .t1 = 0,
            .t1_arr = .{ .ctx = null },
            .drafts = &.{},
            .draft_arrs = &.{},
            .n_drafted = 0,
            .conf_arrs = null,
            .n_conf = 0,
            .q_probs = null,
            .n_qp = 0,
            .h_chain = null,
            .m = 0,
        };
        n_chain = i + 1;
    }

    const chain_lap = generate_mod.Generator.SubLap.start(generate_mod.Generator.mtpTraceOn(), live[0].io);
    generate_mod.Generator.mtpChainBuildBatched(gens[0..n], chains[0..n], 0, depth) catch |e| {
        for (live[0..n]) |slot| slot.markError(@errorName(e));
        return;
    };
    // Dispatched before the verify build so the chain's GPU time covers the build.
    generate_mod.Generator.mtpChainDispatchBatched(chains[0..n]) catch |e| {
        for (live[0..n]) |slot| slot.markError(@errorName(e));
        return;
    };
    if (chain_lap.read()) |chain_ns| {
        for (gens[0..n]) |gen| gen.mtpTraceSub(.chain, chain_ns);
    }

    const taken = n;
    n = 0;
    n_chain = 0;
    var states: [MAX_BATCH_GROUP]generate_mod.Generator.MtpRoundState = undefined;
    var vgens: [MAX_BATCH_GROUP]*generate_mod.Generator = undefined;
    var vlive: [MAX_BATCH_GROUP]*Slot = undefined;
    var vn: usize = 0;
    defer for (states[0..vn], vlive[0..vn]) |*st, slot| {
        st.deinit(slot.allocator);
    };
    for (0..taken) |i| {
        opens[i].chain = chains[i];
        states[vn] = gens[i].mtpRoundContinue(live[i].allocator, opens[i]) catch |e| {
            live[i].markError(@errorName(e));
            continue;
        };
        vgens[vn] = gens[i];
        vlive[vn] = live[i];
        vn += 1;
    }
    if (vn == 0) return;

    var sts: [MAX_BATCH_GROUP]*generate_mod.Generator.MtpRoundState = undefined;
    for (states[0..vn], 0..) |*st, i| sts[i] = st;
    if (generate_mod.Generator.mtpGroupVerify(vgens[0..vn], sts[0..vn])) |_| {
        generate_mod.Generator.mtpGroupAccept(vgens[0..vn], sts[0..vn]) catch |e| {
            for (vlive[0..vn]) |slot| slot.markError(@errorName(e));
            return;
        };
    } else |e| {
        for (vlive[0..vn]) |slot| slot.markError(@errorName(e));
        return;
    }
    for (states[0..vn], vgens[0..vn], vlive[0..vn]) |*st, gen, slot| {
        if (slot.state == .errored) continue;
        mtpFinishPublish(sch, slot, gen, st);
    }
    observeGroupRound(vlive[0..vn], states[0..vn], group.len, &tick_sw, head_calls_before);
}

/// Group MTP slots by model (pad-waste capped like the plain group) and run one
/// batched verify per group; a group of one takes the solo round.
fn runMtpGroups(sch: *Scheduler, slots: []*Slot) !void {
    if (slots.len == 0) return;
    std.sort.pdq(*Slot, slots, {}, struct {
        fn lt(_: void, a: *Slot, b: *Slot) bool {
            return @intFromPtr(a.model) < @intFromPtr(b.model);
        }
    }.lt);
    var start: usize = 0;
    while (start < slots.len) {
        var end = start + 1;
        while (end < slots.len and slots[end].model == slots[start].model) end += 1;
        var group = slots[start..end];
        if (mtpRoundsStaySolo(group[0])) {
            for (group) |slot| try runSingleDecodeTick(sch, slot);
            start = end;
            continue;
        }
        if (group.len >= 2) {
            var kv_lens: [MAX_BATCH_GROUP]u32 = undefined;
            var caches_buf: [MAX_BATCH_GROUP]*const KVCache = undefined;
            for (group, 0..) |g, i| {
                caches_buf[i] = &g.cache;
            }
            fillGroupPadWasteKvLens(caches_buf[0..group.len], group[0].model.config, 2, kv_lens[0..group.len]);
            const ctx_max = groupCtxMax(caches_buf[0..group.len]);
            var i: usize = 1;
            while (i < group.len) : (i += 1) {
                const slot_i = group[i];
                const len_i = kv_lens[i];
                var j = i;
                while (j > 0 and kv_lens[j - 1] > len_i) : (j -= 1) {
                    group[j] = group[j - 1];
                    kv_lens[j] = kv_lens[j - 1];
                }
                group[j] = slot_i;
                kv_lens[j] = len_i;
            }
            const keep = groupKeepCount(kv_lens[0..group.len], ctx_max);
            for (group[keep..]) |s| {
                noteSerial(sch, s, .pad_waste);
                try runSingleDecodeTick(sch, s);
            }
            group = group[0..keep];
        }
        // Sub-groups sized to the verify lane's row budget; the leftover slot rounds solo.
        var g0: usize = 0;
        while (g0 < group.len) {
            const rem = group.len - g0;
            const size = mtpSubGroupSize(rem, mtpGroupRowCap());
            const sub = group[g0 .. g0 + size];
            if (size >= 2) {
                const cap = mtpGroupRowCap() / @as(u32, @intCast(size)) - 1;
                for (sub) |slot| slot.legacy_gen.?.mtp_group_cap = cap;
                try runBatchedMtpTick(sch, sub);
            } else {
                sub[0].legacy_gen.?.mtp_group_cap = 0;
                try runSingleDecodeTick(sch, sub[0]);
            }
            g0 += size;
        }
        start = end;
    }
}

/// Rows one batched verify may carry: past 7 the projections leave the split-K lane for
/// stock kernels; the NAX m16 tile carries 16.
fn mtpGroupRowCap() u32 {
    return if (dflash_mod.wideVerifyLaneAvailable()) 16 else 7;
}

/// MTP slots on one model at or past this count decode on the plain batched tick instead:
/// sub-grouped verify rounds lose to one plain tick there.
fn mtpCrowdThreshold() usize {
    return mtpSubGroupSize(std.math.maxInt(usize), mtpGroupRowCap()) + 1;
}

/// Slots in the next sub-group: two at depth 2 or three at depth 1 fill 7 rows; four is
/// better as 2+2 than 3+1.
pub fn mtpSubGroupSize(remaining: usize, row_cap: u32) usize {
    if (remaining < 2) return remaining;
    if (row_cap >= 16) return @min(remaining, 4);
    if (remaining == 3 or remaining >= 5) return 3;
    return 2;
}

fn runBatchedMtpTick(sch: *Scheduler, group: []*Slot) !void {
    var inner_err: ?anyerror = null;
    runBatchedMtpTickInner(sch, group) catch |e| {
        inner_err = e;
    };
    mlx.checkErrorDecode() catch |mlx_err| {
        log.err("[scheduler] batched verify aborted: MLX failure mid-generation ({s}) — failing these requests, the server keeps serving\n", .{@errorName(mlx_err)});
        for (group) |slot| slot.markError(@errorName(mlx_err));
        return;
    };
    if (inner_err) |e| return e;
}

/// One speculative round for a group: every slot drafts on its own head, the trunk
/// verifies all of them in ONE `[N, S]` forward (rows padded to the widest draft),
/// every slot accepts and rolls back on its own row.
fn runBatchedMtpTickInner(sch: *Scheduler, group: []*Slot) !void {
    const allocator = sch.allocator;
    var states: [MAX_BATCH_GROUP]generate_mod.Generator.MtpRoundState = undefined;
    var live: [MAX_BATCH_GROUP]*Slot = undefined;
    var n: usize = 0;
    defer for (states[0..n], live[0..n]) |*st, slot| {
        st.deinit(slot.allocator);
    };

    for (group) |slot| {
        const gen = &slot.legacy_gen.?;
        if (try loopGuardTick(sch, slot, gen)) continue;
        const begun = gen.mtpRoundBegin(slot.allocator) catch |e| {
            slot.markError(@errorName(e));
            continue;
        };
        switch (begun) {
            .open => unreachable,
            .done => |r| publishMtpResult(sch, slot, gen, r),
            .verify => |st| {
                states[n] = st;
                live[n] = slot;
                n += 1;
            },
        }
    }
    if (n == 0) return;
    if (n == 1) {
        const gen = &live[0].legacy_gen.?;
        try gen.mtpRoundVerify(&states[0]);
        const r = try gen.mtpRoundFinish(live[0].allocator, &states[0]);
        publishMtpResult(sch, live[0], gen, r);
        return;
    }

    const xfm = live[0].model.transformer.?;
    const s_stream = xfm.s;
    var width: u32 = 0;
    for (states[0..n]) |*st| width = @max(width, st.verify_len);
    const ctxs = try allocator.alloc(*ForwardCtx, n);
    defer allocator.free(ctxs);
    const rope_offsets = try allocator.alloc(u32, n);
    defer allocator.free(rope_offsets);
    for (live[0..n], 0..) |slot, i| {
        ctxs[i] = &slot.legacy_gen.?.ctx;
        rope_offsets[i] = @intCast(slot.moe_seq_offset);
    }
    if (!xfm.batchedGdnReady(ctxs)) {
        // No recurrent state to merge yet: every round solo this tick.
        for (states[0..n], live[0..n]) |*st, slot| {
            const gen = &slot.legacy_gen.?;
            try gen.mtpRoundVerify(st);
            const r = try gen.mtpRoundFinish(slot.allocator, st);
            publishMtpResult(sch, slot, gen, r);
        }
        return;
    }

    // [N, width] rows: each verify input right-padded with token 0 (never read past 1+m).
    const rows_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(rows_vec);
    var padded: [MAX_BATCH_GROUP]mlx.mlx_array = undefined;
    var padded_n: usize = 0;
    defer for (padded[0..padded_n]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (states[0..n]) |*st| {
        const pad: c_int = @intCast(width - st.verify_len);
        var row = mlx.mlx_array_new();
        if (pad > 0) {
            const axes = [_]c_int{1};
            const low = [_]c_int{0};
            const high = [_]c_int{pad};
            const zero = mlx.mlx_array_new_int(0);
            defer _ = mlx.mlx_array_free(zero);
            try mlx.check(mlx.mlx_pad(&row, st.verify_input, &axes, 1, &low, 1, &high, 1, zero, "constant", s_stream));
        } else {
            try mlx.check(mlx.mlx_array_set(&row, st.verify_input));
        }
        padded[padded_n] = row;
        padded_n += 1;
        _ = mlx.mlx_vector_array_append_value(rows_vec, row);
    }
    var token_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(token_arr);
    try mlx.check(mlx.mlx_concatenate_axis(&token_arr, rows_vec, 0, s_stream));

    var hidden_last = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden_last);
    var hidden_all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden_all);
    const logits = try xfm.forwardMoeBatchedVerify(token_arr, ctxs, rope_offsets, &hidden_last, &hidden_all);
    defer _ = mlx.mlx_array_free(logits);
    // The batched forward moved only its scratch offset; the solo forward would have
    // advanced every slot by the rows it ran, and the finish rolls back from there.
    for (live[0..n]) |slot| slot.moe_seq_offset += width;

    const logit_rows = try Transformer.sliceBatchRows(allocator, s_stream, logits, n);
    defer allocator.free(logit_rows);
    const last_rows = try Transformer.sliceBatchRows(allocator, s_stream, hidden_last, n);
    defer allocator.free(last_rows);
    const all_rows = try Transformer.sliceBatchRows(allocator, s_stream, hidden_all, n);
    defer allocator.free(all_rows);
    for (states[0..n], live[0..n], 0..) |*st, slot, i| {
        st.verify_logits = logit_rows[i];
        st.new_hidden = last_rows[i];
        st.verify_hidden_all = all_rows[i];
        st.verify_len = width;
        _ = try slot.legacy_gen.?.thinkShiftRows(&st.verify_logits, padded[i]);
    }
    if (sch.metrics) |m| m.batched_group_size.set(n);
    for (states[0..n], live[0..n]) |*st, slot| {
        const gen = &slot.legacy_gen.?;
        mtpFinishPublish(sch, slot, gen, st);
    }
}

fn groupCostSampleComplete(expected: usize, retained: []const u32, published: []const bool) bool {
    if (expected < 1 or expected > group_cost_mod.MAX_GROUP_ROWS or retained.len != expected or published.len != expected) return false;
    for (retained, published) |count, complete| if (count == 0 or !complete) return false;
    return true;
}

fn cacheCostFormat(config: transformer_mod.KVQuantConfig) u64 {
    return (@as(u64, config.group_size) << 16) | (@as(u64, config.bits) << 8) | @backingInt(config.scheme);
}

/// One wall for the group grid: from the tick's first round-begin to its last publish.
/// Both group sites hand their tick stopwatch here rather than starting one of their own.
fn observeGroupRound(live: []*Slot, states: []const Generator.MtpRoundState, expected: usize, tick_sw: *io_util.Stopwatch, head_calls_before: u64) void {
    const xfm = live[0].model.transformer orelse return;
    if (live.len > group_cost_mod.MAX_GROUP_ROWS) return;
    var price_transition = false;
    for (live) |slot| price_transition = price_transition or slot.planner_price_transition;
    defer {
        for (live) |slot| slot.planner_price_transition = false;
    }
    var emitted: [group_cost_mod.MAX_GROUP_ROWS]u32 = undefined;
    var published: [group_cost_mod.MAX_GROUP_ROWS]bool = undefined;
    var gaps: [group_cost_mod.MAX_GROUP_ROWS]f32 = undefined;
    for (live, states, 0..) |slot, st, i| {
        emitted[i] = st.retained;
        published[i] = slot.state == .decoding and !slot.cancelled.load(.acquire);
        gaps[i] = slot.mtp_publish_gap_ms;
    }
    if (!groupCostSampleComplete(expected, emitted[0..live.len], published[0..live.len])) return;
    if (price_transition) return;
    var key = group_cost_mod.GroupShape{
        .n = @intCast(live.len),
        .kv_format = cacheCostFormat(live[0].cache.config),
        .head_format = if (xfm.qwen4_mtp) |head| cacheCostFormat(head.cache.config) else 0,
        .attention = xfm.cost_attention,
        .moe = xfm.cost_moe,
        .head_calls = @intCast(@min(std.math.maxInt(u16), transformer_mod.mtp_head_row_dispatches -% head_calls_before)),
    };
    for (live, states, 0..) |slot, st, i| {
        const chain = st.chain;
        if (chain.m > round_cost_mod.MAX_WIDTH or chain.n_drafted > round_cost_mod.MAX_WIDTH or chain.head_input_rows > round_cost_mod.MAX_WIDTH + 2) return;
        const mode = group_cost_mod.GroupShape.samplingMode(slot.legacy_gen.?.mtp.?.canRerankDrafts(), chain.conf_arrs != null, chain.q_probs != null, !generate_mod.isGreedyTemperature(slot.legacy_gen.?.sampling.temperature));
        key.rows[i] = group_cost_mod.GroupShape.row(@intCast(chain.m), @intCast(chain.n_drafted), @intCast(@min(chain.plan.m_lo, chain.n_drafted)), @intCast(xfm.round_cost.bucketOf(@intCast(st.moe_seq_offset_snap))), @intCast(xfm.round_cost.bucketOf(@intCast(chain.off0))), @intCast(chain.head_input_rows), mode);
        key.padding += @intCast(st.verify_len - (1 + chain.m));
    }
    const ms = @as(f32, @floatFromInt(tick_sw.read())) / @as(f32, std.time.ns_per_ms);
    _ = xfm.mtp_group_cost.observeShape(key, ms, emitted[0..live.len], gaps[0..live.len], .round);
}

/// One verify forward for a whole group: ragged rows ride the row-axis form, equal-lane
/// trunks the merged padded forward, one row a solo round.
/// Finish one row's round and publish it; a failed finish errors the slot instead.
fn mtpFinishPublish(sch: *Scheduler, slot: *Slot, gen: *Generator, st: *Generator.MtpRoundState) void {
    const r = gen.mtpRoundFinish(slot.allocator, st) catch |e| {
        slot.markError(@errorName(e));
        return;
    };
    publishMtpResult(sch, slot, gen, r);
}

fn publishMtpResult(sch: *Scheduler, slot: *Slot, gen: *Generator, result: ?Generator.DrafterStepResult) void {
    const r = result orelse {
        finishSlot(sch, slot, gen.finish_reason);
        return;
    };
    defer slot.allocator.free(r.tokens);
    publishSpeculativeBlock(sch, slot, gen, r.tokens);
}

fn runBatchedDecodeTick(sch: *Scheduler, active: []*Slot) !void {
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
fn batchedTickRows(sch: *Scheduler, active: []*Slot, live: []*Slot) !usize {
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
                finishSlot(sch, slot, gen.finish_reason);
                continue;
            }
        } else if (try batchEntryStops(gen)) {
            slot.completion_tokens = gen.completion_tokens;
            finishSlot(sch, slot, gen.finish_reason);
            continue;
        }
        live[live_n] = slot;
        live_n += 1;
    }
    return live_n;
}

var glm_rows_logged = false;

var glm_tree_logged = false;

/// A batched GLM tick: one group per slot through `verifyGroups`, at most four rows. Every slot
/// decodes one plain row, committed by its own generator, unless `glmTreeNodes` hands the spare
/// rows to the best drafter's DFlash2 tree: that slot's round is published here and it leaves
/// `batch`. Returns the plain rows' logits in the order of the shortened `batch`.
fn glmRowsForward(sch: *Scheduler, allocator: std.mem.Allocator, xfm: *Transformer, batch: *[]*Slot, tokens: []const u32) ![]mlx.mlx_array {
    const verifier = @import("glm5_dflash_model.zig");
    const target = xfm.glm5.?;
    target.s = xfm.s;
    target.suppress_mask = xfm.suppress_mask;
    const schedule = verifier.bindDefaultSchedule();
    defer schedule.restore();
    var timer = io_util.Stopwatch.init(sch.io);
    // The drafter with the best landing rate; its tree takes the rows the plain slots leave.
    var tree_at: ?usize = null;
    var proposal: @import("glm5_dflash.zig").Proposal = undefined;
    for (batch.*, 0..) |slot, i| {
        const gen = &slot.legacy_gen.?;
        if (!gen.glm_dflash_native or gen.dflash_ctx == null) continue;
        if (tree_at) |best| if (batch.*[best].legacy_gen.?.glm_draft_rate >= gen.glm_draft_rate) continue;
        tree_at = i;
    }
    if (tree_at) |at| tree: {
        const gen = &batch.*[at].legacy_gen.?;
        const nodes = glmTreeNodes(batch.len, gen.glm_draft_rate, .{});
        if (nodes == 0 or gen.done or gen.completion_tokens + 2 > gen.max_tokens) {
            tree_at = null;
            break :tree;
        }
        gen.dflash.?.s = xfm.s;
        proposal = try gen.glmPropose(nodes);
    }
    const draft_ns = timer.read();
    timer.reset();
    var groups: [verifier.max_rows]verifier.Group = undefined;
    // A DFlash2 row keeps its assistant context; every row then carries the assistant's taps.
    var taps: []const u32 = &.{};
    for (batch.*, 0..) |slot, i| {
        const gen = &slot.legacy_gen.?;
        if (gen.glm_dflash_native) taps = gen.dflash.?.config.target_layer_ids;
        const request = gen.ctx.glm5_request orelse return error.GlmRequestMissing;
        groups[i] = if (tree_at == i)
            .{ .request = request, .tokens = proposal.tokens[0..proposal.count], .parents = proposal.parents[0..proposal.count] }
        else
            .{ .request = request, .tokens = tokens[i .. i + 1], .parents = &.{-1} };
    }
    if (!glm_rows_logged) {
        glm_rows_logged = true;
        log.info("[glm-rows] grouped plain tick engaged: {d} rows\n", .{batch.len});
    }
    if (tree_at != null and !glm_tree_logged) {
        glm_tree_logged = true;
        log.info("[glm-rows] a {d}-row draft tree joined {d} plain rows\n", .{ proposal.count, batch.len - 1 });
    }
    var rows = GlmRows{ .sch = sch, .target = target, .taps = taps, .proposal = &proposal, .draft_ns = draft_ns, .timer = timer };
    return glmRowsSettle(&rows, allocator, batch, groups[0..batch.len], tree_at);
}

/// The grouped tick's production side of `glmRowsSettle`.
const GlmRows = struct {
    const verifier = @import("glm5_dflash_model.zig");
    sch: *Scheduler,
    target: *const @import("glm5_forward.zig").Model,
    taps: []const u32,
    proposal: *const @import("glm5_dflash.zig").Proposal,
    draft_ns: u64,
    timer: io_util.Stopwatch,
    verify_ns: u64 = 0,

    fn verify(self: *GlmRows, groups: []const verifier.Group, out: []verifier.Verified) !void {
        try verifier.verifyGroups(self.target, groups, self.taps, .affine_rows_ffn, out);
        self.verify_ns = self.timer.read();
    }

    fn commitRow(_: *GlmRows, slot: *Slot, v: *const verifier.Verified) !mlx.mlx_array {
        return slot.legacy_gen.?.glmRowCommit(v);
    }

    fn endTree(self: *GlmRows, slot: *Slot, v: *verifier.Verified) void {
        const gen = &slot.legacy_gen.?;
        const result = gen.glmRoundEnd(slot.allocator, self.proposal, v, self.draft_ns, self.verify_ns, false) catch |err| {
            slot.markError(@errorName(err));
            return;
        };
        defer slot.allocator.free(result.tokens);
        publishSpeculativeBlock(self.sch, slot, gen, result.tokens);
    }
};

/// Verifies `groups` (one per `batch` slot), commits the plain rows and ends the tree slot's round
/// through `ops`; the plain slots stay in `batch`, in order, and their logits are returned.
fn glmRowsSettle(ops: anytype, allocator: std.mem.Allocator, batch: *[]*Slot, groups: []const GlmRows.verifier.Group, tree_at: ?usize) ![]mlx.mlx_array {
    const verifier = GlmRows.verifier;
    var out: [verifier.max_rows]verifier.Verified = undefined;
    // `batch` shrinks below; the verified rows are destroyed by the count that built them.
    const verified = out[0..batch.len];
    try ops.verify(groups, verified);
    defer for (verified) |*v| v.deinit();
    const logits = try allocator.alloc(mlx.mlx_array, batch.len - @intFromBool(tree_at != null));
    var made: usize = 0;
    errdefer {
        for (logits[0..made]) |a| _ = mlx.mlx_array_free(a);
        allocator.free(logits);
    }
    var kept: usize = 0;
    for (batch.*, verified, 0..) |slot, *v, i| {
        if (tree_at == i) {
            ops.endTree(slot, v);
            continue;
        }
        logits[made] = try ops.commitRow(slot, v);
        made += 1;
        batch.*[kept] = slot;
        kept += 1;
    }
    batch.* = batch.*[0..kept];
    return logits;
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
    // hybrid trunks (qwen3_5 family). `batchedGdnReady` is the runtime half of
    // the gate — a slot that has not prefilled yet carries no recurrent state
    // to merge, so that tick stays serial rather than merging a wrong width.
    const use_gdn = xfm_ptr.supportsBatchedGdnDecode() and xfm_ptr.batchedGdnReady(ctxs);
    const use_mimo = xfm_ptr.supportsBatchedMimoDecode();
    const use_glm = xfm_ptr.supportsBatchedGlmRows();
    const use_qwen4_rows = use_gdn and xfm_ptr.supportsBatchedQwen4Rows();
    // Position source is per PATH: a GDN trunk positions from the slot's
    // `moe_seq_offset` — `KVCache.step` only advances on layer 0, which is a
    // linear layer there, so it reads 0 forever and every batched token was
    // roped at position 0 (qwen3_5 batched diverged from serial at token 14).
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
    var sample_params: [MAX_BATCH_GROUP]generate_mod.SamplingParams = undefined;
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
        try generate_mod.sampleRows(sample_ids[0..batch.len], sample_rows[0..batch.len], sample_params[0..batch.len], xfm_ptr.s);
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
            finishSlot(sch, slot, "stop");
            continue;
        }
        if (gen.next_token_id == 0) {
            gen.consecutive_pad += 1;
            if (gen.consecutive_pad >= 3) {
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

test "runPrefill wires the interleave hook and bills its decode ticks out of prefill_ns" {
    const src = @embedFile("scheduler.zig");
    // The hook is wired at the ONE Generator construction site, env-gated.
    const wire = ".interleave" ++ "_hook = if (prefillInterleaveEnabled())";
    try testing.expect(std.mem.indexOf(u8, src, wire) != null);
    // Interleaved decode time is charged to the DECODING slots (they got the
    // tokens), so the prefilling slot's prefill_ns must exclude it or
    // prefill_tps under-reports on every interleaved prefill.
    const bill = "slot.prefill_ns = prefill_sw.read() -| slot.prefill_" ++ "interleaved_ns;";
    try testing.expect(std.mem.indexOf(u8, src, bill) != null);
}

test "a resident MiMo batches decode in groups of the FP8 GEMV's row-identical width; a streamed one stays serial" {
    var cfg = std.mem.zeroes(model_mod.ModelConfig);
    cfg.model_type = "mimo_v2";
    cfg.num_experts = 256;
    try testing.expect(configBatchesDecode(&cfg));
    try testing.expectEqual(@as(usize, 4), batchGroupCap(&cfg));
    cfg.expert_streaming = true;
    try testing.expect(!configBatchesDecode(&cfg));
    try testing.expectEqual(MAX_BATCH_GROUP, batchGroupCap(&cfg));
}

test "a resident qwen4_exp decodes rows of one forward without padding; a streamed one keeps the padded batch" {
    var cfg = std.mem.zeroes(model_mod.ModelConfig);
    cfg.model_type = "qwen4_exp";
    try testing.expect(cfg.supportsBatchedQwen4Rows());
    try testing.expectEqual(@as(usize, 8), batchGroupCap(&cfg));
    cfg.expert_streaming = true;
    try testing.expect(!cfg.supportsBatchedQwen4Rows());
    try testing.expectEqual(MAX_BATCH_GROUP, batchGroupCap(&cfg));
}

test "batchedKvKeepCount: padding waste caps the group, and the long slots are the ones dropped" {
    // Even lengths: no padding waste, everybody batches.
    try testing.expectEqual(@as(usize, 4), batchedKvKeepCount(&[_]u32{ 1000, 1000, 1000, 1000 }));
    try testing.expectEqual(@as(usize, 4), batchedKvKeepCount(&[_]u32{ 900, 1000, 1100, 1200 }));

    // The case this exists for: three short streams and one long one. Batching
    // all four pads to 4 x 100000 = 400k against 103k useful (3.9x); dropping
    // the long one leaves 3 x 1000 vs 3000 (1.0x).
    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&[_]u32{ 1000, 1000, 1000, 100_000 }));

    // Two long ones: the pair still batches together, since 2 x 100000 against
    // 200000 useful wastes nothing — the veto is about the padding, not length.
    try testing.expectEqual(@as(usize, 2), batchedKvKeepCount(&[_]u32{ 100_000, 100_000 }));
    // One short slot among two long ones still batches: 3 x 100000 padded
    // against 200010 useful is 1.5x, inside the bar. The veto is about the
    // WASTE the padding creates, not about any slot being an outlier.
    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&[_]u32{ 10, 100_000, 100_000 }));

    // One slot never "batches", and neither does an empty group.
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&[_]u32{1000}));
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&[_]u32{}));

    // Nothing prefilled yet: no padding to waste, so nothing is vetoed (the
    // ready gate, not this one, is what keeps unprefilled slots out).
    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&[_]u32{ 0, 0, 0 }));

    // A pathological pair degrades to no batch at all rather than making the
    // 1-token slot build a 200k-wide tensor nobody billed. This is why the bar
    // has to sit below 2.0 — at 2.0 a pair is unvetoable by construction.
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&[_]u32{ 1, 200_000 }));
}

test "groupKeepCount: the small-pad floor keeps a group whole only at long context" {
    // A long stream billed at the gather arm's 2052-row cap beside two short sub-agents: 3x pad.
    const billed = [_]u32{ 10, 10, 2052 };
    try testing.expectEqual(@as(usize, 2), groupKeepCount(&billed, PAD_FREE_MIN_CTX - 1));
    try testing.expectEqual(@as(usize, 3), groupKeepCount(&billed, PAD_FREE_MIN_CTX));
    // Past the floor's billed bound the cap holds at any context.
    try testing.expectEqual(@as(usize, 0), groupKeepCount(&[_]u32{ 1, PAD_FREE_KV + 1 }, 300_000));
    try testing.expectEqual(@as(usize, 2), groupKeepCount(&[_]u32{ 1, PAD_FREE_KV }, 300_000));
    // A group of one never batches.
    try testing.expectEqual(@as(usize, 0), groupKeepCount(&[_]u32{2052}, 300_000));
}

test "a 137k stream beside two fresh sub-agents stays one group on the gather bill" {
    const prev_b = transformer_mod.qsa_batched_gather_override;
    const prev_g = transformer_mod.qsa_gather_override;
    const prev_d = transformer_mod.qsa_decode_gather_override;
    defer {
        transformer_mod.qsa_batched_gather_override = prev_b;
        transformer_mod.qsa_gather_override = prev_g;
        transformer_mod.qsa_decode_gather_override = prev_d;
    }
    transformer_mod.qsa_batched_gather_override = true;
    transformer_mod.qsa_gather_override = true;
    transformer_mod.qsa_decode_gather_override = true;
    var q4 = model_mod.ModelConfig{ .model_type = "qwen4_exp", .indexer_budget = 2048, .indexer_compress_ratio = 4 };
    const lens = [_]usize{ 10, 12, 137_000 };
    var caches: [3]KVCache = undefined;
    var built: usize = 0;
    defer for (caches[0..built]) |*c| c.deinit();
    var ptrs: [3]*const KVCache = undefined;
    for (lens, 0..) |len, i| {
        caches[i] = try KVCache.init(testing.allocator, 32);
        built += 1;
        caches[i].entries[3].initialized = true;
        caches[i].entries[3].offset = len;
        ptrs[i] = &caches[i];
    }
    var billed: [3]u32 = undefined;
    fillGroupPadWasteKvLens(&ptrs, &q4, 1, &billed);
    try testing.expectEqual(@as(u32, 2052), billed[2]);
    try testing.expectEqual(@as(usize, 2), batchedKvKeepCount(&billed));
    try testing.expectEqual(@as(u32, 137_000), groupCtxMax(&ptrs));
    try testing.expectEqual(@as(usize, 3), groupKeepCount(&billed, groupCtxMax(&ptrs)));
    // Below the context floor the serial forward is cheap and the plain cap splits the stream off.
    caches[2].entries[3].offset = 100_000;
    try testing.expectEqual(@as(usize, 2), groupKeepCount(&billed, groupCtxMax(&ptrs)));
}

test "the batched group is capped by padding waste before it is dispatched" {
    // Source-scan class guard: the grouping loop must consult the cap. Without
    // it a single long-context stream makes every short neighbour materialize
    // a padded KV tensor sized by ITS length — a per-tick transient no gate
    // bills, whose failure mode is an uncatchable Metal OOM.
    const src = @embedFile("scheduler.zig");
    const start = std.mem.indexOf(u8, src, "// Group batchable slots by model pointer") orelse return error.MissingGrouping;
    const end = std.mem.indexOfPos(u8, src, start, "\n}\n") orelse return error.MissingGroupingEnd;
    const body = src[start..end];
    try testing.expect(std.mem.indexOf(u8, body, "groupKeepCount(") != null);
    // ...and the dropped slots must still be ticked, or they never advance.
    try testing.expect(std.mem.indexOf(u8, body, "noteSerial(sch, s, .pad_waste)") != null);
}

test "the pad-waste cap reads the arch's TRUE attention KV length, not cache.step" {
    // `cache.step` is 0 forever on a linear-layer-0 trunk, so the cap never fired there.
    const lens = [_]usize{ 1_000, 1_000, 60_000 };
    const first_attn = [_]u32{ 3, 2, 7 }; // GDN / gated-conv / KDA spacings
    var caches: [3]KVCache = undefined;
    var built: usize = 0;
    defer for (caches[0..built]) |*c| c.deinit();
    for (lens, first_attn, 0..) |len, fa, i| {
        caches[i] = try KVCache.init(testing.allocator, 32);
        built += 1;
        caches[i].entries[fa].initialized = true;
        caches[i].entries[fa].offset = len;
        try testing.expectEqual(@as(usize, 0), caches[i].step); // the trap
    }

    var q4 = model_mod.ModelConfig{ .model_type = "qwen4_exp" };
    var kv_lens: [3]u32 = undefined;
    for (caches[0..], 0..) |*c, i| kv_lens[i] = batchKvLenOf(c, &q4);
    try testing.expectEqualSlices(u32, &[_]u32{ 1_000, 1_000, 60_000 }, &kv_lens);

    // Every other hybrid keeps `cache.step` (the cap stays dead there, pending a multi-stream measurement).
    for ([_][]const u8{ "qwen3_5", "qwen3_5_moe", "qwen3_next", "lfm2", "nemotron_h", "bailing_hybrid" }) |mt| {
        var cfg = model_mod.ModelConfig{ .model_type = mt };
        for (caches[0..]) |*c| try testing.expectEqual(@as(u32, 0), batchKvLenOf(c, &cfg));
    }
    for (caches[0..]) |*c| try testing.expectEqual(@as(u32, 0), batchKvLenOf(c, null));

    try testing.expectEqual(@as(usize, 2), batchedKvKeepCount(&kv_lens));
    try testing.expect(batchedPadWaste(&kv_lens) > MAX_PAD_WASTE);

    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&[_]u32{ 0, 0, 0 }));
    try testing.expectEqual(@as(f64, 1.0), batchedPadWaste(&[_]u32{ 0, 0, 0 }));
}

test "an attention-first trunk's batching lengths are unchanged by the fix" {
    // Dense / attention-first archs are byte-identical across this change.
    const lens = [_]usize{ 1_000, 1_000, 1_000, 100_000 };
    var llama = model_mod.ModelConfig{ .model_type = "llama" };
    var caches: [4]KVCache = undefined;
    var built: usize = 0;
    defer for (caches[0..built]) |*c| c.deinit();
    var kv_lens: [4]u32 = undefined;
    for (lens, 0..) |len, i| {
        caches[i] = try KVCache.init(testing.allocator, 8);
        built += 1;
        caches[i].step = len;
        caches[i].entries[0].initialized = true;
        caches[i].entries[0].offset = len;
        kv_lens[i] = batchKvLenOf(&caches[i], &llama);
    }
    try testing.expectEqualSlices(u32, &[_]u32{ 1_000, 1_000, 1_000, 100_000 }, &kv_lens);
    var q4b = model_mod.ModelConfig{ .model_type = "qwen4_exp" };
    for (caches[0..], 0..) |*c, i| try testing.expectEqual(kv_lens[i], batchKvLenOf(c, &q4b));
    try testing.expectEqual(@as(usize, 3), batchedKvKeepCount(&kv_lens));
    try testing.expectEqual(
        batchedKvKeepCount(&[_]u32{ 1_000, 1_000, 1_000, 100_000 }),
        batchedKvKeepCount(&kv_lens),
    );
}

test "batchKvLenOf bills qwen4 selected length when the gather arm is on" {
    const prev = transformer_mod.qsa_batched_gather_override;
    defer transformer_mod.qsa_batched_gather_override = prev;
    transformer_mod.qsa_batched_gather_override = true;
    var q4 = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
    };
    var cache = try KVCache.init(testing.allocator, 32);
    defer cache.deinit();
    cache.entries[3].initialized = true;
    cache.entries[3].offset = 162_000;
    try testing.expectEqual(@as(u32, 2052), batchKvLenOf(&cache, &q4));
    transformer_mod.qsa_batched_gather_override = false;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOf(&cache, &q4));
    var llama = model_mod.ModelConfig{ .model_type = "llama" };
    cache.step = 162_000;
    cache.entries[0].initialized = true;
    cache.entries[0].offset = 162_000;
    transformer_mod.qsa_batched_gather_override = true;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOf(&cache, &llama));
}

test "batchKvLenOf bills raw when the gather switch for that width is off" {
    const prev_b = transformer_mod.qsa_batched_gather_override;
    const prev_g = transformer_mod.qsa_gather_override;
    const prev_d = transformer_mod.qsa_decode_gather_override;
    const prev_v = transformer_mod.qsa_verify_gather_override;
    defer {
        transformer_mod.qsa_batched_gather_override = prev_b;
        transformer_mod.qsa_gather_override = prev_g;
        transformer_mod.qsa_decode_gather_override = prev_d;
        transformer_mod.qsa_verify_gather_override = prev_v;
    }
    transformer_mod.qsa_batched_gather_override = true;
    transformer_mod.qsa_gather_override = true;
    transformer_mod.qsa_decode_gather_override = true;
    transformer_mod.qsa_verify_gather_override = true;
    var q4 = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
    };
    var cache = try KVCache.init(testing.allocator, 32);
    defer cache.deinit();
    cache.entries[3].initialized = true;
    cache.entries[3].offset = 162_000;
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 1));
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 4));
    transformer_mod.qsa_batched_gather_override = false;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOfWith(&cache, &q4, 1));
    transformer_mod.qsa_batched_gather_override = true;
    transformer_mod.qsa_gather_override = false;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOfWith(&cache, &q4, 1));
    transformer_mod.qsa_gather_override = true;
    transformer_mod.qsa_decode_gather_override = false;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOfWith(&cache, &q4, 1));
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 4));
    transformer_mod.qsa_decode_gather_override = true;
    transformer_mod.qsa_verify_gather_override = false;
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 1));
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOfWith(&cache, &q4, 4));
}

test "a 300k slot beside a 1k slot groups on the sparse bill" {
    const prev_b = transformer_mod.qsa_batched_gather_override;
    const prev_g = transformer_mod.qsa_gather_override;
    const prev_d = transformer_mod.qsa_decode_gather_override;
    defer {
        transformer_mod.qsa_batched_gather_override = prev_b;
        transformer_mod.qsa_gather_override = prev_g;
        transformer_mod.qsa_decode_gather_override = prev_d;
    }
    transformer_mod.qsa_batched_gather_override = true;
    transformer_mod.qsa_gather_override = true;
    transformer_mod.qsa_decode_gather_override = true;
    var q4 = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
    };
    var caches: [2]KVCache = undefined;
    var built: usize = 0;
    defer for (caches[0..built]) |*c| c.deinit();
    caches[0] = try KVCache.init(testing.allocator, 32);
    built = 1;
    caches[0].entries[3].initialized = true;
    caches[0].entries[3].offset = 300_000;
    caches[1] = try KVCache.init(testing.allocator, 32);
    built = 2;
    caches[1].entries[3].initialized = true;
    caches[1].entries[3].offset = 1_000;
    const ptrs = [_]*const KVCache{ &caches[0], &caches[1] };
    var billed: [2]u32 = undefined;
    fillGroupPadWasteKvLens(&ptrs, &q4, 1, &billed);
    try testing.expectEqual(@as(u32, 2052), billed[0]);
    try testing.expectEqual(@as(u32, 1_000), billed[1]);
    var billed_asc = billed;
    std.mem.sort(u32, &billed_asc, {}, std.sort.asc(u32));
    try testing.expectEqual(@as(usize, 2), batchedKvKeepCount(&billed_asc));
}

test "S>=2 pad-waste floor is max of gather and verify mins" {
    const prev_b = transformer_mod.qsa_batched_gather_override;
    const prev_g = transformer_mod.qsa_gather_override;
    const prev_v = transformer_mod.qsa_verify_gather_override;
    const prev_gm = transformer_mod.qsa_gather_min_kv_override;
    const prev_vm = transformer_mod.qsa_verify_gather_min_kv_override;
    defer {
        transformer_mod.qsa_batched_gather_override = prev_b;
        transformer_mod.qsa_gather_override = prev_g;
        transformer_mod.qsa_verify_gather_override = prev_v;
        transformer_mod.qsa_gather_min_kv_override = prev_gm;
        transformer_mod.qsa_verify_gather_min_kv_override = prev_vm;
    }
    transformer_mod.qsa_batched_gather_override = true;
    transformer_mod.qsa_gather_override = true;
    transformer_mod.qsa_verify_gather_override = true;
    transformer_mod.qsa_gather_min_kv_override = 20_000;
    transformer_mod.qsa_verify_gather_min_kv_override = 16_384;
    var q4 = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
    };
    var cache = try KVCache.init(testing.allocator, 32);
    defer cache.deinit();
    cache.entries[3].initialized = true;
    cache.entries[3].offset = 18_000;
    try testing.expectEqual(@as(u32, 18_000), batchKvLenOfWith(&cache, &q4, 4));
    cache.entries[3].offset = 162_000;
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 4));
}

test "batchedEffectiveKvLen: qwen4 bills selected length, other archs keep raw kv" {
    var q4 = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
    };
    const min_kv: u32 = 8192;
    try testing.expectEqual(@as(u32, 2052), q4.batchedEffectiveKvLen(162_000, true, min_kv));
    try testing.expectEqual(@as(u32, 2052), q4.batchedEffectiveKvLen(64_000, true, min_kv));
    try testing.expectEqual(@as(u32, 8000), q4.batchedEffectiveKvLen(8000, true, min_kv));
    try testing.expectEqual(@as(u32, 162_000), q4.batchedEffectiveKvLen(162_000, false, min_kv));
    var llama = model_mod.ModelConfig{ .model_type = "llama" };
    try testing.expectEqual(@as(u32, 162_000), llama.batchedEffectiveKvLen(162_000, true, min_kv));
    var q35 = model_mod.ModelConfig{ .model_type = "qwen3_5" };
    try testing.expectEqual(@as(u32, 162_000), q35.batchedEffectiveKvLen(162_000, true, min_kv));

    const billed = [_]u32{
        q4.batchedEffectiveKvLen(16_384, true, min_kv),
        q4.batchedEffectiveKvLen(162_000, true, min_kv),
    };
    try testing.expectEqual(@as(usize, 2), batchedKvKeepCount(&billed));
    const raw_pair = [_]u32{ 16_384, 162_000 };
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&raw_pair));
    const other = [_]u32{ 1_000, 162_000 };
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&other));
}

test "mtpQwen4StaySolo is opt-in" {
    try testing.expect(mtpQwen4StaySolo(true, false));
    try testing.expect(!mtpQwen4StaySolo(true, true));
    try testing.expect(!mtpQwen4StaySolo(false, false));
    try testing.expect(!mtpQwen4StaySolo(false, true));
}

test "a GDN trunk is batchable AND is not clamped by the server's concurrency gate" {
    // The two sites that decide "does this model batch?" must agree. They
    // disagreed once: the scheduler batched qwen4_exp while server.zig still
    // clamped --max-concurrent to 1 for anything with
    // full_attention_interval > 0, so asking for concurrency turned the
    // batched path OFF. Both now read ModelConfig.supportsBatchedGdnDecode.
    var cfg = std.mem.zeroes(model_mod.ModelConfig);
    cfg.model_type = "qwen4_exp";
    cfg.full_attention_interval = 4;
    try testing.expect(cfg.supportsBatchedGdnDecode());

    // The server's startup line, /props and /v1/models read this one predicate.
    try testing.expect(configBatchesDecode(&cfg));
}

test "a spec_disabled_runtime slot is batchable, and that is the documented trade" {
    // `slotTicksRegular` admits a generator whose speculation turned itself off
    // at runtime. That is what recovered the throughput (9.6 -> 12.4 tok/s per
    // stream on 4 concurrent 5.5k prompts), and it costs something real: the
    // batched tick does not call `nextPld`, so its periodic re-enable check
    // cannot run while the slot is batched. Both halves are load-bearing, so
    // both are stated here — if the re-enable check ever moves onto the batched
    // path, or the clause is dropped, this is the note to revisit.
    const src = @embedFile("scheduler.zig");
    const hs = std.mem.indexOf(u8, src, "fn slotTicksRegular(") orelse return error.MissingHelper;
    const he = std.mem.indexOfPos(u8, src, hs + 1, "\n    }\n") orelse return error.MissingHelperEnd;
    const hbody = src[hs..he];
    // The clause that admits it...
    try testing.expect(std.mem.indexOf(u8, hbody, "gen.spec_disabled_runtime or") != null);
    // ...and the trade it makes, written down where it is made.
    try testing.expect(std.mem.indexOf(u8, hbody, "re-enable check") != null);
    try testing.expect(std.mem.indexOf(u8, hbody, "SOLO slot never reaches the batched path") != null);
}

test "the batched gate reads DISPATCH, not the armed spec flags" {
    // A prompt that merely n-gram-scores high enough to ARM PLD used to veto
    // batched decode forever, even after PLD's own yield gate disabled itself
    // at runtime: neither speculation nor batching (9.7 vs 14.3 tok/s per
    // stream, 4 concurrent 4232-token prompts on Qwen3.5-4B). The armed flags
    // are the REQUEST's wish; specTickMode is what the tick actually
    // dispatches, and spec_disabled_runtime is what recovered the throughput.
    const src = @embedFile("scheduler.zig");
    const start = std.mem.indexOf(u8, src, "fn batchVerdict(self: *const Scheduler") orelse return error.MissingBatchable;
    const end = std.mem.indexOfPos(u8, src, start + 1, "\n    }\n") orelse return error.MissingBatchableEnd;
    const body = src[start..end];
    try testing.expect(std.mem.indexOf(u8, body, "slotTicksRegular(slot)") != null);
    // No armed-flag read may come back into the gate.
    try testing.expect(std.mem.indexOf(u8, body, "slot.enable_pld") == null);
    try testing.expect(std.mem.indexOf(u8, body, "slot.enable_drafter") == null);
    try testing.expect(std.mem.indexOf(u8, body, "slot.enable_mtp") == null);

    // ...and the helper answers from BOTH: the generator's runtime kill and
    // the authoritative dispatch decision.
    const hs = std.mem.indexOf(u8, src, "fn slotTicksRegular(") orelse return error.MissingHelper;
    const he = std.mem.indexOfPos(u8, src, hs + 1, "\n    }\n") orelse return error.MissingHelperEnd;
    const hbody = src[hs..he];
    try testing.expect(std.mem.indexOf(u8, hbody, "gen.spec_disabled_runtime") != null);
    try testing.expect(std.mem.indexOf(u8, hbody, "specTickMode(") != null);

    // The dispatch answer itself: a live MTP slot never ticks regular, an
    // unarmed one always does.
    try testing.expectEqual(SpecTickMode.mtp, specTickMode(true, true, false, false, true, true));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(true, false, true, false, true, false));
}

test "supportsBatchedGdnDecode refuses every arch the batched GDN path does not model" {
    // A new arch on the shared moe forward must default to SERIAL, not ride
    // a kernel that never modelled its state.
    {
        var moe = std.mem.zeroes(model_mod.ModelConfig);
        moe.model_type = "hy_v3";
        moe.full_attention_interval = 4;
        moe.num_experts = 128;
        moe.num_experts_per_tok = 8;
        try testing.expect(!moe.supportsBatchedGdnDecode());
    }
    {
        var lfm2 = std.mem.zeroes(model_mod.ModelConfig);
        lfm2.model_type = "lfm2";
        lfm2.has_hybrid_layers = true;
        try testing.expect(!lfm2.supportsBatchedGdnDecode());
    }
    {
        var kda = std.mem.zeroes(model_mod.ModelConfig);
        kda.model_type = "bailing_hybrid";
        kda.full_attention_interval = 4;
        kda.kda_vector_gate = true;
        try testing.expect(!kda.supportsBatchedGdnDecode());
    }
    {
        var ink = std.mem.zeroes(model_mod.ModelConfig);
        ink.model_type = "inkling_mm_model";
        ink.full_attention_interval = 4;
        try testing.expect(!ink.supportsBatchedGdnDecode());
    }
    {
        // Pure attention: not a GDN trunk at all, rides the standard kernel.
        var dense = std.mem.zeroes(model_mod.ModelConfig);
        dense.model_type = "qwen3";
        try testing.expect(!dense.supportsBatchedGdnDecode());
    }
    {
        // qwen4_exp: MoE, but every per-slot piece is on the SSMCacheEntry.
        var q4 = std.mem.zeroes(model_mod.ModelConfig);
        q4.model_type = "qwen4_exp";
        q4.full_attention_interval = 4;
        q4.num_experts = 256;
        q4.num_experts_per_tok = 8;
        try testing.expect(q4.supportsBatchedGdnDecode());
    }
}

test "admitPendingTick: per-slot exclusivity (qwen4 MTP slot) blocks only its own class" {
    const A: usize = 0xA0;
    var out: [16]usize = undefined;
    // An MTP candidate beside a live PLAIN slot on the same model admits.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = false }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &active, &out));
    }
    // An MTP candidate beside a live MTP slot holds.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 0), admitPendingTick(&cands, &active, &out));
    }
    // A plain candidate beside a live MTP slot admits.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = false }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &active, &out));
    }
}

test "admitPendingTick: exclusive single-flight FIFO contract" {
    const A: usize = 0xA0;
    const B: usize = 0xB0;
    var out: [16]usize = undefined;

    // Held while a live slot on the same exclusive model is decoding.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 0), admitPendingTick(&cands, &active, &out));
    }
    // Admits once no live slot holds the model.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &.{}, &out));
        try testing.expectEqual(@as(usize, 0), out[0]);
    }
    // Two exclusive candidates on the SAME model in one tick: only the
    // first admits — the tick-claim covers slots not yet in `decoding`.
    {
        const cands = [_]AdmitCand{
            .{ .model = A, .exclusive = true },
            .{ .model = A, .exclusive = true },
        };
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &.{}, &out));
        try testing.expectEqual(@as(usize, 0), out[0]);
    }
    // Distinct exclusive models admit independently.
    {
        const cands = [_]AdmitCand{
            .{ .model = A, .exclusive = true },
            .{ .model = B, .exclusive = true },
        };
        try testing.expectEqual(@as(usize, 2), admitPendingTick(&cands, &.{}, &out));
        try testing.expectEqual(@as(usize, 0), out[0]);
        try testing.expectEqual(@as(usize, 1), out[1]);
    }
}

test "admitPendingTick: non-exclusive concurrency and queue order preserved" {
    const A: usize = 0xA0;
    const L: usize = 0x10;
    var out: [16]usize = undefined;

    // Non-exclusive candidates (laguna/hy3: per-slot state) admit freely
    // even beside live slots — their serial-tick interleave is safe.
    {
        const cands = [_]AdmitCand{
            .{ .model = L, .exclusive = false },
            .{ .model = L, .exclusive = false },
        };
        const active = [_]AdmitCand{.{ .model = L, .exclusive = false }};
        try testing.expectEqual(@as(usize, 2), admitPendingTick(&cands, &active, &out));
    }
    // A held exclusive candidate must not head-of-line-block a later
    // candidate on another model (requests for OTHER models keep flowing).
    {
        const cands = [_]AdmitCand{
            .{ .model = A, .exclusive = true },
            .{ .model = L, .exclusive = false },
        };
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &active, &out));
        try testing.expectEqual(@as(usize, 1), out[0]);
    }
    // `out` caps the admitted count (mirrors to_prefill's 16), in order.
    {
        const cands = [_]AdmitCand{
            .{ .model = L, .exclusive = false },
            .{ .model = L, .exclusive = false },
            .{ .model = L, .exclusive = false },
        };
        var small: [2]usize = undefined;
        try testing.expectEqual(@as(usize, 2), admitPendingTick(&cands, &.{}, &small));
        try testing.expectEqual(@as(usize, 0), small[0]);
        try testing.expectEqual(@as(usize, 1), small[1]);
    }
}

test "a cancelled or queued-for-cleanup slot still blocks exclusive admission until released" {
    const Fake = struct {
        fn exclusive(_: *const Slot) bool {
            return true;
        }
    };
    var model: LoadedModel = undefined;
    var live: Slot = undefined;
    var cancelled: Slot = undefined;
    var queued: Slot = undefined;
    var finished: Slot = undefined;
    for ([_]*Slot{ &live, &cancelled, &queued, &finished }) |s| {
        s.model = &model;
        s.finished = false;
        s.error_code = null;
        s.cancelled = .init(false);
    }
    cancelled.cancelled.store(true, .release);
    finished.finished = true;
    var out: [8]AdmitCand = undefined;
    const decoding = [_]*Slot{ &live, &cancelled, &finished };
    const cleanup = [_]*Slot{&queued};
    try testing.expectEqual(@as(usize, 3), heldExclusive(&out, &.{ &decoding, &cleanup }, Fake.exclusive));
    try testing.expectEqual(@as(usize, 0), heldExclusive(&out, &.{ &.{}, &.{} }, Fake.exclusive));
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

test "inferenceLoop pending drain routes through admitPendingTick" {
    // A pure admission fn nobody calls is a silent no-op (the specTickMode /
    // hardcoded use_drafter=false class). Needles are ++-split so this
    // test's own source can't satisfy the scan.
    const src = @embedFile("scheduler.zig");
    const call = "admitPendingTick(" ++ "cand_buf[0..n_cands], live_buf[0..n_live], &admit_idx)";
    try testing.expect(std.mem.indexOf(u8, src, call) != null);
    // The candidates' exclusive bit must come from the per-slot predicate
    // (model bit OR a module-owned MTP head on this slot).
    const pred = ".exclusive = slotExclusiveDecode(" ++ "s)";
    try testing.expect(std.mem.indexOf(u8, src, pred) != null);
    // The pre-gate unconditional drain shape must be GONE — its survival
    // would mean a path still admits without the gate.
    const old = "to_prefill[n_prefill] = sch.pending." ++ "orderedRemove(0)";
    try testing.expect(std.mem.indexOf(u8, src, old) == null);
}

test "S21: a released module head drops the slot's exclusivity, and the MODEL bit is untouched" {
    try testing.expect(headExclusiveFor(false, true, true, false));
    try testing.expect(!headExclusiveFor(false, true, true, true));
    try testing.expect(!headExclusiveFor(false, true, false, false));
    try testing.expect(!headExclusiveFor(false, true, false, true));
    try testing.expect(!headExclusiveFor(false, false, true, false));
    try testing.expect(!headExclusiveFor(false, false, true, true));
    try testing.expect(headExclusiveFor(true, false, false, true));
    try testing.expect(headExclusiveFor(true, true, true, true));
}

test "preloadCpuState refuses an unsupported checkpoint format by name" {
    // The verdict is a NAMED load error the client sees as a 503, never a
    // stub config.
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "g");
    try tmp.dir.writeFile(io, .{ .sub_path = "g/model-Q4_K_M.gguf", .data = "GGUF" });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/g", .{buf[0..root_len]});
    defer testing.allocator.free(path);
    try testing.expectError(error.ModelFormatUnsupported, preloadCpuState(testing.allocator, io, path));
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

fn expectOutcomes(m: *const metrics_mod.Metrics, success: u64, cancelled: u64, failed: u64, rejected: u64) !void {
    try testing.expectEqual(success, m.requests_success_total.load());
    try testing.expectEqual(cancelled, m.requests_cancelled_total.load());
    try testing.expectEqual(failed, m.requests_failed_total.load());
    try testing.expectEqual(rejected, m.requests_rejected_total.load());
}

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

test "outcome: a real Slot's fields satisfy the recorder, and a cancelled GLM slot counts once" {
    var m = metrics_mod.Metrics.init();
    var slot: Slot = undefined;
    slot.metrics_recorded = false;
    slot.finished = false;
    slot.error_code = null;
    slot.stop_hit = .init(false);
    slot.first_token_ns = 1;
    slot.prefill_ns = 1;
    slot.decode_ns = 1;
    slot.prompt_tokens = 4;
    slot.completion_tokens = 2;
    slot.cached_tokens = 0;
    slot.finish_reason = "";
    recordSlotCleanup(&m, &slot);
    recordSlotCleanup(&m, &slot);
    try expectOutcomes(&m, 0, 1, 0, 0);
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
    for (0..generate_mod.degenerate_loop_min_span / 6 + 1) |_| {
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
    for (0..generate_mod.degenerate_loop_long_min_span / cycle.len) |_| try ids.appendSlice(testing.allocator, &cycle);
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
    for (0..generate_mod.degenerate_loop_min_span / 3 + 1) |_| {
        try ids.appendSlice(testing.allocator, &[_]u32{ 101, 102, 103 });
    }

    const stop = loopStopDecision(ids.items) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", stop.finish_reason);
    try testing.expectEqualStrings("repetition_loop", stop.finish_details);
    try testing.expectEqual(generate_mod.DegenerateTail.Tier.exact_cycle, stop.tier);
    // Trimmed to the honest prefix plus one copy of the cycle.
    try testing.expectEqual(@as(usize, 6), stop.trim_start);

    // Once a deferred boundary has been committed, the old reasoning loop is
    // outside the guard window and cannot stop the very next answer tick.
    try ids.append(testing.allocator, 99);
    const answer_start = ids.items.len;
    try testing.expect(loopStopDecision(ids.items[answer_start..]) == null);

    // A new loop wholly inside the constrained answer retains the existing
    // stop/repetition result, with an absolute trim point for response code.
    for (0..generate_mod.degenerate_loop_min_span / 3 + 1) |_| {
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

test "loopStopReason: a VARIED-phrasing restatement loop is cut at the near-repeat tier" {
    // The agent-traffic shape (2026-08-04): the same intent restated forever in
    // slightly different words. No exact cycle exists, so tiers 1 and 2 are
    // blind and this ran to max_tokens.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    const phrasings = [_][]const u32{
        &[_]u32{ 40, 41, 42, 43, 44, 45, 46 },
        &[_]u32{ 40, 41, 42, 43, 44, 46 },
        &[_]u32{ 40, 41, 42, 43, 44, 45, 47, 48, 46 },
        &[_]u32{ 49, 40, 41, 42, 43, 44, 45, 46 },
        &[_]u32{ 40, 41, 42, 43, 44, 45, 50, 46 },
    };
    var i: usize = 0;
    while (ids.items.len < generate_mod.near_repeat_min_span + 32) : (i += 1) {
        try ids.appendSlice(testing.allocator, phrasings[i % phrasings.len]);
    }
    const reason = loopStopReason(ids.items) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", reason);

    // A long answer that keeps introducing new material is untouched, however
    // repetitive its scaffolding.
    var healthy = std.ArrayList(u32).empty;
    defer healthy.deinit(testing.allocator);
    var line: u32 = 0;
    while (healthy.items.len < generate_mod.near_repeat_window + 32) : (line += 1) {
        try healthy.appendSlice(testing.allocator, &[_]u32{ 10, 11, 12 });
        try healthy.append(testing.allocator, 1000 + line);
        try healthy.appendSlice(testing.allocator, &[_]u32{ 13, 14 });
    }
    try testing.expect(loopStopReason(healthy.items) == null);
}

test "specInitWiring: a module-owned arch only gets the spec modes it can roll back" {
    // Spec decode must be able to ROLL BACK a rejected tail. The shell rolls
    // back its KVCache/ssm_entries — which a module-owned arch does not use, so
    // by the time the verify returns, the module has already absorbed every
    // draft and the shell's snapshot/truncate run over an EMPTY entries array.
    // The predicate is per-ARCH CAPABILITY (`moduleStateSpecRollback`), not ownership.
    // Args: (owns_module_state, module_spec_rollback,
    //        enable_mtp, has_mtp, enable_drafter, has_drafter, has_dflash,
    //        enable_pld)

    // Plain arch: today's precedence, unchanged.
    {
        const w = specInitWiring(false, false, true, true, true, false, true);
        try testing.expect(w.use_mtp and !w.use_dflash and !w.use_pld);
    }
    {
        const w = specInitWiring(false, false, false, false, false, false, true);
        try testing.expect(!w.use_mtp and w.use_pld);
    }
    // A flag with no loaded handle never arms.
    {
        const w = specInitWiring(false, false, true, false, false, false, false);
        try testing.expect(!w.use_mtp and !w.use_dflash and !w.use_pld);
    }

    // DFlash rides the enable_drafter switch: dflash > MTP > drafter > PLD.
    {
        const w = specInitWiring(false, false, false, false, true, true, true);
        try testing.expect(!w.use_mtp and w.use_dflash and !w.use_pld);
    }
    // A loaded drafter outranks the checkpoint's own MTP head: the sidecar
    // is an explicit choice (`--drafter` or the in-dir `drafter/`), the head
    // ships with every pack; `--no-drafter` restores MTP by not loading it.
    {
        const w = specInitWiring(false, false, true, true, true, true, true);
        try testing.expect(w.use_dflash and !w.use_mtp and !w.use_pld);
    }
    // enable_drafter:false on the request hands the round back to MTP.
    {
        const w = specInitWiring(false, false, true, true, false, true, true);
        try testing.expect(w.use_mtp and !w.use_dflash);
    }
    // enable_drafter:false opts BOTH sidecar kinds out.
    {
        const w = specInitWiring(false, false, false, false, false, true, true);
        try testing.expect(!w.use_dflash and w.use_pld);
    }

    // Module-owned with NO rollback: everything off.
    {
        const w = specInitWiring(true, false, true, true, true, true, true);
        try testing.expect(!w.use_mtp and !w.use_dflash and !w.use_pld);
    }

    // Module-owned WITH rollback: its own MTP head arms; the shell
    // spec modes stay off because none has been measured on this family.
    {
        const w = specInitWiring(true, true, true, true, true, true, true);
        try testing.expect(w.use_mtp and !w.use_dflash and !w.use_pld);
    }
    // Rollback capability alone never arms a head that is not loaded.
    {
        const w = specInitWiring(true, true, true, false, true, true, true);
        try testing.expect(!w.use_mtp and !w.use_dflash and !w.use_pld);
    }
    // ...nor one the request opted out of.
    {
        const w = specInitWiring(true, true, false, true, true, false, true);
        try testing.expect(!w.use_mtp);
    }
    // An image request keeps the head (the qwen4 head takes the slot's
    // M-RoPE table); the drafters stay off.
    {
        const w = specInitWiring(true, true, true, true, true, true, true);
        try testing.expect(w.use_mtp and !w.use_dflash and !w.use_pld);
    }

}

test "runPrefill gates spec through specInitWiring, not per-arch conjuncts" {
    // A pure predicate nobody calls is a silent no-op. Needles are ++-split so
    // this test's own source cannot satisfy the scan.
    const src = @embedFile("scheduler.zig");
    // Keyed on the call site's own bindings, not on a `specInitWiring(` prefix
    // this test's own arms would satisfy.
    inline for (.{ "const use_mtp = wiring" ++ ".use_mtp;", "const use_pld = wiring" ++ ".use_pld;" }) |needle| {
        try testing.expect(std.mem.indexOf(u8, src, needle) != null);
    }
    // The exclusion must come from the shared predicate, not a new arch list.
    const from_predicate = "transformer.?.moduleSpec" ++ "Wiring()";
    try testing.expect(std.mem.indexOf(u8, src, from_predicate) != null);
}

test "glmNativeDflashArmed: eligible greedy/sampled requests arm it, constrained/penalised/logprobs/budgeted-thinking/disabled ones do not" {
    const t = testing;
    try t.expect(glmNativeDflashArmed(true, true, true, .{ .temperature = 0 }, 0));
    try t.expect(glmNativeDflashArmed(true, true, true, .{ .temperature = 1 }, 0));

    try t.expect(!glmNativeDflashArmed(false, true, true, .{ .temperature = 0 }, 0));
    try t.expect(!glmNativeDflashArmed(true, false, true, .{ .temperature = 0 }, 0));
    try t.expect(!glmNativeDflashArmed(true, true, false, .{ .temperature = 0 }, 0));

    var constraint: generate_mod.Constraint = undefined;
    try t.expect(!glmNativeDflashArmed(true, true, true, .{ .temperature = 0, .constraint = &constraint }, 0));
    try t.expect(!glmNativeDflashArmed(true, true, true, .{ .temperature = 0, .presence_penalty = 1 }, 0));
    try t.expect(!glmNativeDflashArmed(true, true, true, .{ .temperature = 0 }, 1));
    var think_bound: generate_mod.ThinkBound = undefined;
    try t.expect(!glmNativeDflashArmed(true, true, true, .{ .temperature = 0, .think_bound = &think_bound }, 0));
}

test "specTickMode: every spec arm requires the GENERATOR's armed state, not the slot flag alone" {
    // runPrefill's per-site guard can compute use_pld=false and wire it into
    // Generator init options; a decode tick dispatching on `slot.enable_pld`
    // alone would still call `gen.nextPld`. mtp/drafter are saved by their
    // generator-state conjunct (`gen.mtp != null`); PLD has no model handle,
    // so its conjunct must be the generator's post-init `pld_enabled`.

    // Slot wants PLD, generator was NOT armed (chokepoint or per-site guard
    // flipped it off) → the tick must run the regular path.
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, false, true, false));
    // Slot wants PLD and init armed it → PLD runs.
    try testing.expectEqual(SpecTickMode.pld, specTickMode(false, false, false, false, true, true));
    // Generator armed but the slot never asked (stale generator state must
    // not resurrect spec either) → regular.
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, false, false, true));

    // mtp/drafter keep their existing both-sides contract.
    try testing.expectEqual(SpecTickMode.mtp, specTickMode(true, true, false, false, false, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(true, false, false, false, false, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, true, false, false, false));

    // Priority: MTP > drafter > PLD (the spec-dispatch rule).
    try testing.expectEqual(SpecTickMode.mtp, specTickMode(true, true, true, false, true, true));

    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, false, true, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, true, false, false, false));
    // Generator armed but the request opted enable_mtp off → serial.
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, false, false, false));

    // DFlash: slot's enable_drafter + generator's dflash handle; outranks the
    // gemma drafter AND the MTP head (a loaded sidecar is the explicit
    // choice). Generator handle alone never resurrects it,
    // and a dflash generator with the slot flag off stays regular (the
    // specTickMode both-sides contract) — or MTP when that is armed.
    try testing.expectEqual(SpecTickMode.dflash, specTickMode(false, false, true, true, false, false));
    try testing.expectEqual(SpecTickMode.dflash, specTickMode(false, false, true, true, false, false));
    try testing.expectEqual(SpecTickMode.dflash, specTickMode(true, true, true, true, false, false));
    try testing.expectEqual(SpecTickMode.mtp, specTickMode(true, true, false, true, false, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(false, false, false, true, false, false));
}

test "the qwen4 coarse rerank head is built at LOAD, on both load paths" {
    // Built lazily it landed inside the first request's draft chain: a
    // 248320-row requantize of the trunk lm_head plus a synchronous eval of
    // ~240 MB, on the inference thread, mid-round. That is first-token
    // latency on the first request after a load — the sidecar arm has always
    // paid it at bind time instead.
    //
    // `doLoadOnInferenceThread` is the ONE Transformer construction site, and
    // the boot load and the `/v1/load` cold load both route through it (the
    // "a launch flag that shapes a LOAD" class: a hook on only one of them
    // ships a server where the first cold-loaded model still pays lazily).
    // This pins all three.
    const source = @embedFile("scheduler.zig");
    const start = std.mem.indexOf(u8, source, "fn doLoadOnInferenceThread(") orelse return error.MissingLoadFn;
    const end = std.mem.indexOfPos(u8, source, start + 1, "\nfn ") orelse return error.MissingLoadFnEnd;
    const body = source[start..end];
    const build = std.mem.indexOf(u8, body, "qwen4BuildDraftRerank()") orelse return error.MissingEagerRerankBuild;
    // Gated exactly like the head it drafts for: `--no-mtp` never drafts, so
    // it must never pay for the coarse head.
    const gate = std.mem.lastIndexOf(u8, body[0..build], "mtp_enabled") orelse return error.EagerRerankBuildUngated;
    try testing.expect(build - gate < 200);

    // Boot load (inferenceLoop) and cold load (runLoadRequest) both land there.
    const boot = std.mem.indexOf(u8, source, "doLoadOnInferenceThread(sch, params)") orelse return error.MissingBootLoad;
    const cold = std.mem.indexOf(u8, source, "doLoadOnInferenceThread(sch, req)") orelse return error.MissingColdLoad;
    try testing.expect(boot != cold);
}

test "SUSHI_SSD_WRITE_THROUGH=0 takes mechanism 3 out of the prefill, and only that" {
    try testing.expect(writeThroughEnabledFromEnv(null));
    try testing.expect(writeThroughEnabledFromEnv(""));
    try testing.expect(writeThroughEnabledFromEnv("1"));
    try testing.expect(writeThroughEnabledFromEnv("true"));
    try testing.expect(!writeThroughEnabledFromEnv("0"));
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

test "postEvictionPrefillChunk: the re-asked width is the one that runs, in BOTH directions" {
    // 1. Admitted at 1024, affords 4096 after the pass: runs at 4096, logs the widen.
    {
        const d = postEvictionPrefillChunk(1024, 4096);
        try testing.expectEqual(@as(u32, 4096), d.width);
        try testing.expect(d.widened);
        try testing.expect(!d.moved);
    }
    try testing.expectEqual(@as(u32, 8192), postEvictionPrefillChunk(512, 8192).width);

    // 2. No eviction pass ran: the re-ask is the only ask, nothing reportable.
    {
        const d = postEvictionPrefillChunk(0, 1024);
        try testing.expectEqual(@as(u32, 1024), d.width);
        try testing.expect(!d.widened);
        try testing.expect(!d.moved);
    }
    try testing.expectEqual(@as(u32, 512), postEvictionPrefillChunk(0, 512).width);
    try testing.expectEqual(@as(u32, 0), postEvictionPrefillChunk(0, 0).width);

    // 3. A narrower re-ask still wins (`@max` would widen on memory that is gone).
    {
        const d = postEvictionPrefillChunk(4096, 1024);
        try testing.expectEqual(@as(u32, 1024), d.width);
        try testing.expect(d.moved);
        try testing.expect(!d.widened);
    }
    try testing.expectEqual(@as(u32, 512), postEvictionPrefillChunk(2048, 512).width);

    // 4. Unchanged memory is a no-op in both directions.
    inline for (.{ 512, 1024, 2048, 4096, 8192 }) |w| {
        const d = postEvictionPrefillChunk(w, w);
        try testing.expectEqual(@as(u32, w), d.width);
        try testing.expect(!d.widened and !d.moved);
    }

    // 5. Never below the ladder floor: this function computes no width of its own.
    inline for (.{ 512, 1024, 2048, 4096, 8192 }) |floor| {
        try testing.expectEqual(@as(u32, floor), postEvictionPrefillChunk(8192, floor).width);
    }
}

test "prefix-cache commit declines a pad-only generation, never a zero-token one" {
    // Bar: a zero-token answer still commits; only an all-pad generation is declined.
    try testing.expect(!commitDeclinesPadOnly(0, true));
    try testing.expect(!commitDeclinesPadOnly(0, false));
    try testing.expect(commitDeclinesPadOnly(3, true));
    try testing.expect(!commitDeclinesPadOnly(3, false));
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

test "mtpSubGroupSize fills the verify lane's row budget" {
    // 7 rows off-NAX: pairs at depth 2, triples at depth 1, four as 2+2, one rounds solo.
    try testing.expectEqual(@as(usize, 1), mtpSubGroupSize(1, 7));
    try testing.expectEqual(@as(usize, 2), mtpSubGroupSize(2, 7));
    try testing.expectEqual(@as(usize, 3), mtpSubGroupSize(3, 7));
    try testing.expectEqual(@as(usize, 2), mtpSubGroupSize(4, 7));
    try testing.expectEqual(@as(usize, 3), mtpSubGroupSize(5, 7));
    try testing.expectEqual(@as(usize, 4), mtpSubGroupSize(9, 16));
    // The depth each sub-group gets: floor(rows / size) - 1.
    try testing.expectEqual(@as(u32, 2), 7 / @as(u32, 2) - 1);
    try testing.expectEqual(@as(u32, 1), 7 / @as(u32, 3) - 1);
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

test "MiMo crowded MTP respects the batching kill switch" {
    const saved = mtp_group_env;
    defer mtp_group_env = saved;
    var xfm: Transformer = undefined;
    xfm.config = .{ .model_type = "mimo_v2" };
    xfm.moe_layers = &.{};
    xfm.expert_stream = null;
    var model: model_registry_mod.LoadedModel = undefined;
    model.transformer = &xfm;
    var gen: Generator = undefined;
    gen.mtp = .{ .mimo = undefined };
    gen.mtp_cache = .{ .mimo = undefined };
    gen.has_last_hidden = true;
    gen.spec_disabled_runtime = false;
    gen.mtp_serial_left = 0;
    gen.mtp_serial_exit = .none;
    gen.dflash = null;
    gen.pld_enabled = false;
    var slot: Slot = undefined;
    slot.model = &model;
    slot.legacy_gen = gen;
    slot.enable_mtp = true;
    slot.enable_drafter = false;
    slot.enable_pld = false;
    slot.sampling = .{};
    slot.logprobs_n = 0;
    mtp_group_env = true;
    try testing.expect(slotMimoMtpCrowdable(&slot));
    mtp_group_env = false;
    try testing.expect(!slotMimoMtpCrowdable(&slot));
}

test "single MTP slot reaches the round entry through runDecodeTick" {
    var xfm: Transformer = undefined;
    var model: model_registry_mod.LoadedModel = undefined;
    model.transformer = null;
    var gen: Generator = undefined;
    gen.mtp_planner_owned = false;
    gen.generated_ids = .empty;
    gen.loop_guard_start = 0;
    gen.mtp = .{ .qwen4 = &xfm };
    gen.mtp_cache = .{ .qwen = undefined };
    gen.has_last_hidden = true;
    gen.dflash = null;
    gen.pld_enabled = false;
    gen.done = false;
    gen.sampling = .{};
    gen.logprobs_n = 1;
    var slot: Slot = undefined;
    slot.allocator = testing.allocator;
    slot.model = &model;
    slot.legacy_gen = gen;
    slot.enable_mtp = true;
    slot.enable_drafter = false;
    slot.enable_pld = false;
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

test "a plain batched tick checks the thinking budget before its rows forward" {
    // A thought past its budget with no room left to close it: the bound fires, the slot decodes on.
    const forced = [_]u32{ 9, 7 };
    var tb = generate_mod.ThinkBound{ .budget = 2, .opener_id = 5, .closer_id = 7, .forced = &forced, .in_think = false };
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(testing.allocator);
    try ids.appendSlice(testing.allocator, &.{ 5, 11, 12, 13 });
    var gen: Generator = undefined;
    gen.generated_ids = ids;
    gen.loop_guard_start = 0;
    gen.sampling = .{ .think_bound = &tb };
    gen.completion_tokens = 4;
    gen.max_tokens = 4;
    gen.has_pending_logits = false;
    gen.has_pending_token = false;
    gen.step = 0;
    gen.next_token_id = 9;
    gen.eos_token_ids = &.{2};
    gen.consecutive_pad = 0;
    gen.timeout_ns = 0;
    gen.stall = .{};
    gen.timer = io_util.Stopwatch.init(testing.io);
    var slot: Slot = undefined;
    slot.legacy_gen = gen;
    var sch: Scheduler = undefined;
    var active = [_]*Slot{&slot};
    var live: [1]*Slot = undefined;
    try testing.expectEqual(@as(usize, 1), try batchedTickRows(&sch, &active, &live));
    try testing.expect(tb.fired);
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

test "merged verify: an uncertified [N,S] shape is declined by name, never dispatched" {
    try testing.expect(mergedVerifyDeclineReason(true, 5) != null);
    try testing.expect(mergedVerifyDeclineReason(true, 2) != null);
    // Width 1 is a plain batched decode row, not a verify.
    try testing.expect(mergedVerifyDeclineReason(true, 1) == null);
    // Other GDN trunks keep the padded merged forward they had.
    try testing.expect(mergedVerifyDeclineReason(false, 5) == null);
}

test "group verify: ragged widths keep ONE forward, never a solo cohort per width" {
    // The row-axis form runs each row at its own width, so a mixed group is never split.
    try testing.expectEqual(VerifyShape.row_axis, verifyShapeFor(4, true, 9));
    try testing.expectEqual(VerifyShape.row_axis, verifyShapeFor(2, true, 2));
    try testing.expectEqual(VerifyShape.merged, verifyShapeFor(4, false, 9));
    try testing.expectEqual(VerifyShape.merged, verifyShapeFor(2, false, 3));
    // One row is a solo round on every trunk.
    try testing.expectEqual(VerifyShape.solo, verifyShapeFor(1, true, 5));
    try testing.expectEqual(VerifyShape.solo, verifyShapeFor(1, false, 5));
}

test "group cost geometry rejects partial rounds and keeps complete cache formats" {
    try testing.expect(groupCostSampleComplete(2, &.{ 1, 3 }, &.{ true, true }));
    try testing.expect(!groupCostSampleComplete(3, &.{ 1, 3 }, &.{ true, true }));
    try testing.expect(!groupCostSampleComplete(2, &.{ 1, 0 }, &.{ true, true }));
    try testing.expect(!groupCostSampleComplete(2, &.{ 1, 3 }, &.{ true, false }));
    const a = transformer_mod.KVQuantConfig.affine(8);
    var b = a;
    b.group_size = 128;
    try testing.expect(cacheCostFormat(a) != cacheCostFormat(b));
    try testing.expect(cacheCostFormat(a) != cacheCostFormat(transformer_mod.KVQuantConfig.affine(4)));
}

test "scheduler prices each shared execution once and preserves row sampling geometry" {
    var xfm: Transformer = undefined;
    xfm.round_cost = .{ .layout = .long };
    xfm.mtp_group_cost = .{};
    var model: LoadedModel = undefined;
    model.transformer = &xfm;
    var a: Slot = undefined;
    var b: Slot = undefined;
    a.model = &model;
    b.model = &model;
    const slots = [_]*Slot{ &a, &b };
    var geometry = group_cost_mod.GroupShape{ .n = 2, .kv_format = 8, .head_format = 8 };
    geometry.rows[0] = group_cost_mod.GroupShape.row(1, 1, 1, 2, 1, 3, 1);
    geometry.rows[1] = group_cost_mod.GroupShape.row(1, 1, 1, 5, 2, 2, 9);
    var prices = PlannerPrices{ .slots = &slots, .geometry = geometry };
    const speculative = prices.keyFor(&slots, &.{ 2, 4 });
    const swapped = prices.keyFor(&.{ &b, &a }, &.{ 4, 2 });
    try testing.expectEqual(speculative, swapped);
    for (0..3) |_| _ = xfm.mtp_group_cost.observeShape(speculative, 30, &.{ 2, 4 }, &.{ 30, 30 }, .round);
    try testing.expectEqual(@as(f32, 30), prices.price(&.{ 2, 4 }).?.ms);
    const plain_a = prices.keyFor(&.{&a}, &.{0});
    const spec_b = prices.keyFor(&.{&b}, &.{4});
    for (0..3) |_| {
        _ = xfm.mtp_group_cost.observeShape(plain_a, 10, &.{1}, &.{10}, .round);
        _ = xfm.mtp_group_cost.observeShape(spec_b, 25, &.{4}, &.{25}, .round);
    }
    try testing.expectEqual(@as(f32, 35), prices.price(&.{ 0, 4 }).?.ms);
    try testing.expect(prices.price(&.{ 4, 0 }) == null);
    const plain_b = prices.keyFor(&.{&b}, &.{0});
    try testing.expectEqual(@as(u8, 8), group_cost_mod.GroupShape.part(plain_b.rows[0], 48));
    try testing.expectEqual(@as(u8, 5), group_cost_mod.GroupShape.part(plain_b.rows[0], 24));
    try testing.expectEqual(@as(u8, 0), group_cost_mod.GroupShape.part(plain_b.rows[0], 40));
    try testing.expectEqual(@as(u64, 0), plain_b.head_format);
}

test "planner pools recurring head histories without mixing the cold first history" {
    var xfm: Transformer = undefined;
    xfm.round_cost = .{ .layout = .long };
    xfm.mtp_group_cost = .{};
    var model: LoadedModel = undefined;
    model.transformer = &xfm;
    var slots_storage: [4]Slot = undefined;
    var slots: [4]*Slot = undefined;
    for (&slots_storage, 0..) |*slot, i| {
        slot.model = &model;
        slots[i] = slot;
    }
    var geometry = group_cost_mod.GroupShape{ .n = 4, .kv_format = 8, .head_format = 8 };
    @memset(geometry.rows[0..4], group_cost_mod.GroupShape.row(2, 2, 2, 2, 1, 4, 1));
    var prices = PlannerPrices{ .slots = &slots, .geometry = geometry };

    var cold = geometry;
    cold.rows[0] = group_cost_mod.GroupShape.row(2, 2, 2, 2, 1, 1, 1);
    _ = xfm.mtp_group_cost.observeShape(cold, 180, &.{ 2, 2, 2, 2 }, &.{ 180, 180, 180, 180 }, .round);
    var h2 = geometry;
    h2.rows[0] = group_cost_mod.GroupShape.row(2, 2, 2, 2, 1, 2, 1);
    _ = xfm.mtp_group_cost.observeShape(h2, 74, &.{ 2, 3, 3, 2 }, &.{ 74, 74, 74, 74 }, .round);
    const probe_seed = prices.sample(&slots, &.{ 2, 2, 2, 2 }, false).?;
    try testing.expectEqual(@as(u32, 1), probe_seed.samples);
    try testing.expectEqual(@as(f32, 74), probe_seed.ms);
    var h3 = geometry;
    h3.rows[1] = group_cost_mod.GroupShape.row(2, 2, 2, 1, 1, 3, 1);
    _ = xfm.mtp_group_cost.observeShape(h3, 76, &.{ 3, 2, 3, 2 }, &.{ 76, 76, 76, 76 }, .round);
    var other_head_kv = geometry;
    other_head_kv.rows[1] = group_cost_mod.GroupShape.row(2, 2, 2, 2, 2, 3, 1);
    _ = xfm.mtp_group_cost.observeShape(other_head_kv, 20, &.{ 3, 3, 3, 3 }, &.{ 20, 20, 20, 20 }, .round);
    var other_mode = geometry;
    other_mode.rows[1] = group_cost_mod.GroupShape.row(2, 2, 2, 2, 1, 3, 9);
    _ = xfm.mtp_group_cost.observeShape(other_mode, 20, &.{ 3, 3, 3, 3 }, &.{ 20, 20, 20, 20 }, .round);
    try testing.expect(prices.price(&.{ 2, 2, 2, 2 }) == null);

    var h4 = geometry;
    h4.rows[2] = group_cost_mod.GroupShape.row(2, 2, 2, 2, 0, 2, 1);
    _ = xfm.mtp_group_cost.observeShape(h4, 78, &.{ 3, 3, 2, 2 }, &.{ 78, 78, 78, 78 }, .round);
    const pooled = prices.price(&.{ 2, 2, 2, 2 }).?;
    try testing.expectEqual(@as(u32, 3), pooled.samples);
    try testing.expect(pooled.ms >= 74 and pooled.ms <= 78);
    try testing.expect(pooled.variance > 0);
    try testing.expect(pooled.max_gap_ms >= 78);

    const plain = prices.keyFor(&slots, &.{ 0, 0, 0, 0 });
    for (0..3) |_| _ = xfm.mtp_group_cost.observeShape(plain, 37, &.{ 1, 1, 1, 1 }, &.{ 37, 37, 37, 37 }, .round);
    var histories: [4]Planner.History = @splat(.{});
    var rows: [4]Planner.Row = undefined;
    for (&histories, 0..) |*history, i| {
        for (0..3) |_| history.observe(2, 2);
        rows[i] = .{ .history = history };
    }
    const decision = Planner.choose(&rows, &prices);
    try testing.expectEqualSlices(u8, &.{ 2, 2, 2, 2 }, decision.widths[0..4]);
}

test "mean cost pooling retains differences between histories and both worst-case bounds" {
    var xfm: Transformer = undefined;
    xfm.round_cost = .{ .layout = .long };
    xfm.mtp_group_cost = .{};
    var model: LoadedModel = undefined;
    model.transformer = &xfm;
    var storage: [2]Slot = undefined;
    var slots: [2]*Slot = undefined;
    for (&storage, 0..) |*slot, i| {
        slot.model = &model;
        slots[i] = slot;
    }
    var geometry = group_cost_mod.GroupShape{ .n = 2 };
    @memset(geometry.rows[0..2], group_cost_mod.GroupShape.row(2, 2, 2, 4, 4, 4, 1));
    var prices = PlannerPrices{ .slots = &slots, .geometry = geometry };
    var a = geometry;
    var b = geometry;
    a.rows[0] = group_cost_mod.GroupShape.row(2, 2, 2, 4, 4, 2, 1);
    b.rows[0] = group_cost_mod.GroupShape.row(2, 2, 2, 4, 4, 3, 1);
    for (0..20) |_| {
        _ = xfm.mtp_group_cost.observeShape(a, 64, &.{ 3, 3 }, &.{ 64, 64 }, .round);
        _ = xfm.mtp_group_cost.observeShape(b, 80, &.{ 3, 3 }, &.{ 80, 80 }, .round);
    }
    const pooled = prices.price(&.{ 2, 2 }).?;
    try testing.expectApproxEqAbs(@as(f32, 88), pooled.mean_upper_ms.?, 0.001);
    xfm.round_cost = .{ .layout = .long };
    xfm.mtp_group_cost = .{};
    a = geometry;
    b = geometry;
    b.head_calls = 2;
    xfm.mtp_group_cost.shapes[0] = .{ .shape = a, .cell = .{ .ms = 60, .n = 100 }, .variance = 225, .mean_weight_sq = 1.0 / 19.0 };
    xfm.mtp_group_cost.shapes[1] = .{ .shape = b, .cell = .{ .ms = 75, .n = 3 }, .variance = 9, .mean_weight_sq = 1.0 / 3.0 };
    const exact = prices.price(&.{ 2, 2 }).?;
    try testing.expectApproxEqAbs(@as(f32, 90), exact.ms + 2 * @sqrt(exact.variance), 0.001);
    try testing.expectApproxEqAbs(@as(f32, 75) + 2 * @sqrt(@as(f32, 3)), exact.mean_upper_ms.?, 0.001);
    a = prices.keyFor(&slots, &.{ 0, 0 });
    b = a;
    b.head_calls = 2;
    xfm.mtp_group_cost.shapes[0] = .{ .shape = a, .cell = .{ .ms = 40, .n = 100 }, .variance = 25, .mean_weight_sq = 1.0 / 19.0 };
    xfm.mtp_group_cost.shapes[1] = .{ .shape = b, .cell = .{ .ms = 35, .n = 3 }, .variance = 1, .mean_weight_sq = 1.0 / 3.0 };
    const plain = prices.price(&.{ 0, 0 }).?;
    try testing.expectApproxEqAbs(@as(f32, 35) - 2 * @sqrt(@as(f32, 1.0 / 3.0)), plain.mean_lower_ms.?, 0.001);
}

test "cold grouped depths one and two use a bounded two-tick bootstrap" {
    var prices: PlannerPrices = undefined;
    prices.slots = &.{};
    const plain = Planner.Price{ .ms = 40, .variance = 4, .samples = 10 };
    try testing.expectEqual(@as(f32, 88), prices.probeFallbackEstimate(&.{ 1, 1, 1, 1 }, plain));
    try testing.expectEqual(@as(f32, 88), prices.probeFallbackEstimate(&.{ 2, 2, 2, 2 }, plain));
    try testing.expectEqual(@as(f32, 330), prices.probeFallbackEstimate(&.{ 4, 4, 4, 4 }, plain));
    const lower = Planner.Price{ .ms = 60, .variance = 100, .samples = 2, .max_gap_ms = 170 };
    try testing.expectEqual(@as(f32, 90), PlannerPrices.scaledProbeEstimate(lower, 2, 1));
}

test "transition pricing skips the first realized round after prime or width change" {
    try testing.expect(plannerPriceTransition(255, 1, false));
    try testing.expect(plannerPriceTransition(1, 1, true));
    try testing.expect(!plannerPriceTransition(1, 1, false));
    try testing.expect(plannerPriceTransition(1, 2, false));
    try testing.expect(!plannerPriceTransition(2, 0, true));
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

test "every real streamed pack and source on this box plans a streamed load" {
    const t = std.testing;
    const Case = struct { name: []const u8, layout: expert_stream_mod.quant.Layout, tower: u64 };
    for ([_]Case{
        .{ .name = "Qwen3.8-Flash-Next-Sushi-2.6bpw", .layout = .exl3_k4, .tower = 897_862_112 },
        .{ .name = "Qwen3.8-Flash-Next-Sushi-4bpw", .layout = .exl3_k4, .tower = 897_862_112 },
        .{ .name = "Qwen/Qwen3.8-Flash-Next", .layout = .bf16_fused, .tower = 897_862_112 },
        .{ .name = "MiMo-V2.6-Flash-Sushi-2.3bpw", .layout = .exl3_k4, .tower = 1_457_188_864 },
        .{ .name = "GLM-5.3-Flash-Sushi-2.3bpw", .layout = .exl3_k4, .tower = 493_389_824 },
        .{ .name = "GLM-5.3-Flash-Sushi-2.5bpw", .layout = .exl3_k4, .tower = 493_389_824 },
        .{ .name = "GLM-5.3-Flash-BF16", .layout = .bf16_individual, .tower = 1_127_254_016 },
        .{ .name = "GLM-5.3-Flash-FP8", .layout = .fp8_individual, .tower = 1_127_254_016 },
    }) |case| {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try @import("test_models.zig").packPath(&path_buf, case.name);
        var config = model_mod.parseConfig(t.io, t.allocator, path) catch continue;
        defer config.deinit(t.allocator);
        const plan = try planExpertStreaming(t.io, t.allocator, &config, path, 0, 60 << 30, false);
        try t.expectEqual(case.layout, plan.layout);
        try t.expect(plan.cache.slots_per_layer > 1);
        try t.expect(plan.split.trunk > 0 and plan.split.trunk < 40 << 30);
        try t.expectEqual(@as(u64, 0), plan.split.vision);
        try t.expectEqual(case.tower, plan.vision.tower_bytes);
        const seen = try planExpertStreaming(t.io, t.allocator, &config, path, 0, 60 << 30, true);
        try t.expectEqual(case.tower, seen.split.vision);
        try t.expectEqual(plan.split.trunk, seen.split.trunk);
        try t.expectEqual(seen.vision.slots_with.?, seen.cache.slots_per_layer);
        // Slots are whole per layer: a tower smaller than one slot across the layers, or a pack that fits in full, costs none.
        try t.expect(seen.cache.slots_per_layer <= plan.cache.slots_per_layer);
    }
}

test "a cleanup allocation failure never frees MLX on the connection thread" {
    const Probe = struct {
        var inference_id: std.Thread.Id = undefined;
        var frees: usize = 0;
        var off_thread_frees: usize = 0;
        fn free(_: mlx.mlx_array) void {
            frees += 1;
            if (std.Thread.getCurrentId() != inference_id) off_thread_frees += 1;
        }
        fn complete(sch: *Scheduler, slots: []const *Slot) void {
            for (slots) |slot| sch.complete(slot);
        }
    };
    Probe.inference_id = std.Thread.getCurrentId();
    Probe.frees = 0;
    Probe.off_thread_frees = 0;
    slot_vision_free_test_hook = Probe.free;
    defer slot_vision_free_test_hook = null;
    var failing = testing.FailingAllocator.init(testing.allocator, .{});
    const allocator = failing.allocator();
    var cfg = ModelConfig{ .num_hidden_layers = 0 };
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.transformer = null;
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
    defer for (sch.cleanup_queue.items) |slot| slot.deinit();
    for (0..12) |_| {
        failing.fail_index = std.math.maxInt(usize);
        failing.resize_fail_index = std.math.maxInt(usize);
        var slots: [2]*Slot = undefined;
        for (&slots) |*slot| slot.* = try sch.submit(.{
            .model = &model,
            .prompt_ids = &.{1},
            .sampling = .{},
            .eos_token_ids = &.{},
            .max_tokens = 1,
            .vision_embeddings = .{ .ctx = @ptrFromInt(1) },
        });
        failing.fail_index = failing.alloc_index;
        failing.resize_fail_index = failing.resize_index;
        const conn = try std.Thread.spawn(.{}, Probe.complete, .{ &sch, &slots });
        conn.join();
        try testing.expectEqual(@as(usize, 0), Probe.off_thread_frees);
        try testing.expectEqual(@as(usize, 0), Probe.frees);
        try testing.expectEqual(@as(usize, 0), sch.in_flight);
    }
    try testing.expectEqual(@as(usize, 24), sch.cleanup_queue.items.len);
    while (sch.cleanup_queue.items.len > 0) sch.cleanup_queue.orderedRemove(0).deinit();
    try testing.expectEqual(@as(usize, 24), Probe.frees);
    try testing.expectEqual(@as(usize, 0), Probe.off_thread_frees);
}

fn submitTestScheduler(allocator: std.mem.Allocator, queue_cap: u32) Scheduler {
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

test "a submit that fails once its slot is built hands the slot to the inference thread" {
    const Probe = struct {
        var inference_id: std.Thread.Id = undefined;
        var frees: usize = 0;
        var off_thread_frees: usize = 0;
        fn free(_: mlx.mlx_array) void {
            frees += 1;
            if (std.Thread.getCurrentId() != inference_id) off_thread_frees += 1;
        }
        fn submit(sch: *Scheduler, model: *LoadedModel, slot: *?*Slot, err: *?anyerror) void {
            slot.* = sch.submit(.{
                .model = model,
                .prompt_ids = &.{1},
                .sampling = .{},
                .eos_token_ids = &.{},
                .max_tokens = 1,
                .vision_embeddings = .{ .ctx = @ptrFromInt(1) },
            }) catch |e| {
                err.* = e;
                return;
            };
        }
    };
    Probe.inference_id = std.Thread.getCurrentId();
    slot_vision_free_test_hook = Probe.free;
    defer slot_vision_free_test_hook = null;
    var cfg = ModelConfig{ .num_hidden_layers = 0 };
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.transformer = null;
    var handed_over: usize = 0;
    var fail_at: usize = 0;
    while (fail_at < 32) : (fail_at += 1) {
        var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = fail_at });
        const allocator = failing.allocator();
        var sch = submitTestScheduler(allocator, 2);
        defer sch.pending.deinit(allocator);
        defer sch.cleanup_queue.deinit(allocator);
        defer sch.prefilling.deinit(allocator);
        Probe.frees = 0;
        Probe.off_thread_frees = 0;
        var slot: ?*Slot = null;
        var err: ?anyerror = null;
        const conn = try std.Thread.spawn(.{}, Probe.submit, .{ &sch, &model, &slot, &err });
        conn.join();
        try testing.expectEqual(@as(usize, 0), Probe.off_thread_frees);
        if (slot) |s| sch.complete(s);
        // The inference thread's pass over what the connection left behind.
        var batch: [16]*Slot = undefined;
        const take = takeTeardownLocked(&sch, &batch);
        for (batch[0..take.cleanup_n]) |s| s.deinit();
        const built = slot != null or take.abandoned != null;
        if (take.abandoned != null) handed_over += 1;
        destroyAbandoned(take.abandoned);
        try testing.expectEqual(@as(usize, if (built) 1 else 0), Probe.frees);
        try testing.expectEqual(@as(usize, 0), Probe.off_thread_frees);
    }
    // cleanup_queue, prefilling and pending each grow once after the slot is built.
    try testing.expect(handed_over >= 3);

    var shut = submitTestScheduler(testing.allocator, 2);
    shut.shutdown.store(true, .release);
    Probe.frees = 0;
    var shut_slot: ?*Slot = null;
    var shut_err: ?anyerror = null;
    const shut_conn = try std.Thread.spawn(.{}, Probe.submit, .{ &shut, &model, &shut_slot, &shut_err });
    shut_conn.join();
    try testing.expect(shut_err.? == error.Shutdown);
    try testing.expectEqual(@as(usize, 0), Probe.off_thread_frees);
    destroyAbandoned(shut.abandoned);
    try testing.expectEqual(@as(usize, 1), Probe.frees);
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

test "the four-row planner gives spare rows to drafts only while they beat the group's rows" {
    const cost = GlmRowCost{};
    try testing.expectEqual(@as(usize, 2), glmTreeNodes(2, 0.98, cost));
    try testing.expectEqual(@as(usize, 0), glmTreeNodes(2, 0.5, cost));
    try testing.expectEqual(@as(usize, 1), glmTreeNodes(3, 0.95, cost));
    try testing.expectEqual(@as(usize, 0), glmTreeNodes(3, 0.7, cost));
    try testing.expectEqual(@as(usize, 0), glmTreeNodes(4, 1.0, cost));
}

test "a grouped GLM tick destroys every verified row whichever slot holds the tree" {
    const verifier = GlmRows.verifier;
    const Fake = struct {
        allocator: std.mem.Allocator,
        built: usize = 0,
        ended: usize = 0,
        committed: [4]*Slot = undefined,
        committed_n: usize = 0,

        fn verify(self: *@This(), _: []const verifier.Group, out: []verifier.Verified) !void {
            const Tape = @typeInfo(@FieldType(verifier.Verified, "layers")).pointer.child;
            for (out) |*v| {
                const layers = try self.allocator.alloc(Tape, 1);
                layers[0] = null;
                const arrays = try self.allocator.alloc(mlx.mlx_array, 0);
                v.* = .{ .allocator = self.allocator, .layers = layers, .captures = .{ .allocator = self.allocator, .hook = .{ .ids = &.{}, .out = arrays } }, .count = 1, .offset = 0 };
                self.built += 1;
            }
        }
        fn commitRow(self: *@This(), slot: *Slot, _: *const verifier.Verified) !mlx.mlx_array {
            self.committed[self.committed_n] = slot;
            self.committed_n += 1;
            return mlx.mlx_array_new();
        }
        fn endTree(self: *@This(), _: *Slot, _: *verifier.Verified) void {
            self.ended += 1;
        }
    };
    var slots: [4]*Slot = undefined;
    for (&slots, 1..) |*s, i| s.* = @ptrFromInt(@alignOf(Slot) * i);
    var groups: [4]verifier.Group = undefined;
    for (2..5) |width| {
        for (0..width) |tree_at| {
            var fake = Fake{ .allocator = testing.allocator };
            var live = slots;
            var batch: []*Slot = live[0..width];
            const logits = try glmRowsSettle(&fake, testing.allocator, &batch, groups[0..width], tree_at);
            defer testing.allocator.free(logits);
            for (logits) |a| _ = mlx.mlx_array_free(a);
            try testing.expectEqual(width, fake.built);
            try testing.expectEqual(@as(usize, 1), fake.ended);
            try testing.expectEqual(width - 1, batch.len);
            try testing.expectEqual(width - 1, logits.len);
            for (batch, fake.committed[0..fake.committed_n]) |kept, done| try testing.expectEqual(kept, done);
            for (batch) |kept| try testing.expect(kept != slots[tree_at]);
        }
    }
}

test "a submit that fails before its slot exists hands the vision array to the inference thread" {
    const Probe = struct {
        var inference_id: std.Thread.Id = undefined;
        var frees: usize = 0;
        var off_thread_frees: usize = 0;
        fn free(_: mlx.mlx_array) void {
            frees += 1;
            if (std.Thread.getCurrentId() != inference_id) off_thread_frees += 1;
        }
        fn submit(sch: *Scheduler, model: *LoadedModel, err: *?anyerror) void {
            _ = sch.submit(.{
                .model = model,
                .prompt_ids = &.{1},
                .sampling = .{},
                .eos_token_ids = &.{},
                .max_tokens = 1,
                .vision_embeddings = .{ .ctx = @ptrFromInt(1) },
            }) catch |e| {
                err.* = e;
            };
        }
        /// One connection-thread submit that must fail, then the inference thread's drain.
        fn expectHandedOver(allocator: std.mem.Allocator, model: *LoadedModel, kv: transformer_mod.KVQuantConfig, want: ?anyerror) !void {
            var sch: Scheduler = undefined;
            sch.allocator = allocator;
            sch.io = testing.io;
            sch.kv_quant_config = kv;
            sch.kv_quant_explicit = true;
            sch.queue_mu = .init;
            sch.queue_cond = .init;
            sch.submit_cond = .init;
            sch.shutdown = .init(false);
            sch.queue_cap = 2;
            sch.in_flight = 0;
            sch.req_seq = 0;
            sch.pending = .empty;
            sch.cleanup_queue = .empty;
            sch.prefilling = .empty;
            sch.orphan_vision_n = 0;
            defer sch.pending.deinit(allocator);
            defer sch.cleanup_queue.deinit(allocator);
            defer sch.prefilling.deinit(allocator);
            frees = 0;
            off_thread_frees = 0;
            var err: ?anyerror = null;
            const conn = try std.Thread.spawn(.{}, submit, .{ &sch, model, &err });
            conn.join();
            if (want) |w| try testing.expectEqual(w, err.?) else try testing.expect(err != null);
            try testing.expectEqual(@as(usize, 0), off_thread_frees);
            try testing.expectEqual(@as(usize, 0), frees);
            var buf: [orphan_vision_cap]mlx.mlx_array = undefined;
            const n = takeOrphanedVisionLocked(&sch, &buf);
            for (buf[0..n]) |ve| freeVisionArray(ve);
            try testing.expectEqual(@as(usize, 1), n);
            try testing.expectEqual(@as(usize, 1), frees);
            try testing.expectEqual(@as(usize, 0), off_thread_frees);
            try testing.expectEqual(@as(usize, 0), takeOrphanedVisionLocked(&sch, &buf));
        }
    };
    Probe.inference_id = std.Thread.getCurrentId();
    slot_vision_free_test_hook = Probe.free;
    defer slot_vision_free_test_hook = null;

    var cfg = ModelConfig{ .num_hidden_layers = 0 };
    var model: LoadedModel = undefined;
    model.transformer = null;

    model.config = null;
    try Probe.expectHandedOver(testing.allocator, &model, .dense, error.ModelNotReady);

    var glm_cfg = ModelConfig{ .num_hidden_layers = 0, .model_type = "glm5_next" };
    model.config = &glm_cfg;
    try Probe.expectHandedOver(testing.allocator, &model, .affine(4), error.GlmKvQuantUnsupported);

    // Every allocation Slot.init makes is a point where it can fail.
    model.config = &cfg;
    var counting = testing.FailingAllocator.init(testing.allocator, .{});
    const probe_slot = try Slot.init(counting.allocator(), testing.io, &cfg, .{
        .prompt_ids = &.{1},
        .sampling = .{},
        .eos_token_ids = &.{},
        .max_tokens = 1,
        .vision_embeddings = null,
        .model = &model,
    }, .dense);
    probe_slot.deinit();
    try testing.expect(counting.alloc_index > 0);
    for (0..counting.alloc_index) |fail_index| {
        var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = fail_index });
        try Probe.expectHandedOver(failing.allocator(), &model, .dense, error.OutOfMemory);
    }
}

test "a drafting GLM slot rides plain rows only in company of its own model" {
    var glm_a: LoadedModel = undefined;
    var glm_b: LoadedModel = undefined;
    var slots: [4]Slot = undefined;
    for (&slots, 0..) |*s, i| s.model = if (i == 3) &glm_b else &glm_a;
    try testing.expect(!glmRowsInCompany(&slots[0], &.{&slots[0]}, &.{}));
    try testing.expect(!glmRowsInCompany(&slots[0], &.{&slots[0]}, &.{&slots[3]}));
    try testing.expect(glmRowsInCompany(&slots[0], &.{&slots[0]}, &.{&slots[1]}));
    try testing.expect(glmRowsInCompany(&slots[0], &.{ &slots[0], &slots[2] }, &.{}));
}

test "a resident GLM batches plain rows four at a time, a streamed one stays serial" {
    var cfg = try model_mod.parseConfigFromJson(testing.allocator, @embedFile("fixtures/glm5_config.json"));
    try testing.expect(configBatchesDecode(&cfg));
    try testing.expectEqual(@as(usize, 4), batchGroupCap(&cfg));
    cfg.expert_streaming = true;
    try testing.expect(!configBatchesDecode(&cfg));
}

test "every GLM slot owns its native request and hands it to its forward context" {
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
    var slots: [2]*Slot = undefined;
    for (&slots) |*slot| slot.* = try sch.submit(.{ .model = &model, .prompt_ids = &.{1}, .sampling = .{}, .eos_token_ids = &.{}, .max_tokens = 1 });
    for (slots) |slot| {
        const request = &slot.glm5_request.?;
        try testing.expectEqual(request, slot.ctx.glm5_request.?);
        try testing.expectEqual(@as(usize, cfg.num_hidden_layers), request.layers.len);
        try testing.expect(request.dense_prefill and request.prefill_async);
    }
    try testing.expect(slots[0].ctx.glm5_request.? != slots[1].ctx.glm5_request.?);
    for (slots) |slot| sch.complete(slot);
    while (sch.cleanup_queue.items.len > 0) sch.cleanup_queue.orderedRemove(0).deinit();
}

test "a per-request kv_quant override (16, \"16\", 8, 4) reaches the submitted slot's cache as that scheme" {
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
    var plain = ModelConfig{ .model_type = "qwen3", .num_hidden_layers = 1 };
    var model: LoadedModel = undefined;
    model.config = &plain;
    model.transformer = null;
    model.prefix_cache = null;

    const Case = struct { body: std.json.Value, want: transformer_mod.KVQuantConfig };
    for ([_]Case{
        .{ .body = .{ .integer = 16 }, .want = transformer_mod.KVQuantConfig.dense },
        .{ .body = .{ .string = "16" }, .want = transformer_mod.KVQuantConfig.dense },
        .{ .body = .{ .integer = 8 }, .want = transformer_mod.KVQuantConfig.affine(8) },
        .{ .body = .{ .integer = 4 }, .want = transformer_mod.KVQuantConfig.affine(4) },
    }) |c| {
        const override = transformer_mod.KVQuantConfig.fromJsonValue(c.body).?;
        const slot = try sch.submit(.{ .model = &model, .prompt_ids = &.{1}, .sampling = .{}, .eos_token_ids = &.{}, .max_tokens = 1, .kv_quant_config = override });
        try testing.expectEqual(c.want, slot.ctx.cache.config);
        sch.complete(slot);
        while (sch.cleanup_queue.items.len > 0) sch.cleanup_queue.orderedRemove(0).deinit();
    }
}

test "a GLM slot submitted with a kv4 override is refused by the same name admission uses elsewhere" {
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
    const result = sch.submit(.{ .model = &model, .prompt_ids = &.{1}, .sampling = .{}, .eos_token_ids = &.{}, .max_tokens = 1, .kv_quant_config = transformer_mod.KVQuantConfig.affine(4) });
    try testing.expectError(error.GlmKvQuantUnsupported, result);
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
    try testing.expectError(error.EmptyPrompt, result);
    try testing.expectEqual(@as(usize, 0), sch.pending.items.len);
}

test "a freed MiMo slot returns its KV to the OS, not to MLX's pool" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
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
    var plain = ModelConfig{ .model_type = "qwen3", .num_hidden_layers = 1 };
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

test "admission combines Qwen and MiMo reservations in either order" {
    const Probe = struct {
        fn numbers(cfg: *const ModelConfig, _: usize, _: u32, _: transformer_mod.KVQuantConfig, _: bool, _: bool) MemoryBill {
            return .{ .needed = if (cfg.longCtxGated()) 8 else 7, .available = 10 };
        }
    };
    const saved = prefill_admission_numbers;
    prefill_admission_numbers = Probe.numbers;
    defer prefill_admission_numbers = saved;
    var qwen_cfg = ModelConfig{ .model_type = "qwen4_exp" };
    var mimo_cfg = ModelConfig{ .model_type = "mimo_v2" };
    var xfm: Transformer = undefined;
    var qwen: LoadedModel = undefined;
    qwen.config = &qwen_cfg;
    qwen.transformer = &xfm;
    var mimo: LoadedModel = undefined;
    mimo.config = &mimo_cfg;
    mimo.transformer = &xfm;
    var a: Slot = undefined;
    a.model = &qwen;
    a.full_prompt = &.{};
    a.max_tokens = 1;
    a.cache.config = .dense;
    a.vision_embeddings = null;
    a.enable_mtp = false;
    a.memory_hold_logged = true;
    var b = a;
    b.model = &mimo;
    var sch: Scheduler = undefined;
    sch.pending = .empty;
    sch.decoding = .empty;
    defer sch.pending.deinit(testing.allocator);
    try sch.pending.appendSlice(testing.allocator, &.{ &a, &b });
    try testing.expectEqual(@as(usize, 1), memoryAdmitCount(&sch, &.{ 0, 1 }));
    try testing.expectEqual(@as(usize, 1), memoryAdmitCount(&sch, &.{ 1, 0 }));
    try testing.expectEqual(@as(usize, 1), memoryAdmitCount(&sch, &.{0}));
    try testing.expectEqual(@as(usize, 1), memoryAdmitCount(&sch, &.{1}));
}

test "admitsWithinMemory: a live request's outstanding growth is not free headroom" {
    const gb: u64 = 1 << 30;
    const b: ?MemoryBill = .{ .needed = 9 * gb, .available = 10 * gb };
    try testing.expectEqual(@as(usize, 1), admitsWithinMemory(&.{b}, true, 0));
    try testing.expectEqual(@as(usize, 0), admitsWithinMemory(&.{b}, true, 2 * gb));
    // Alone nothing is outstanding, and the lone request proceeds whatever the bill.
    try testing.expectEqual(@as(usize, 1), admitsWithinMemory(&.{b}, false, 0));
}

test "a staggered admission waits for the growth an earlier request was admitted to allocate" {
    const gb: u64 = 1 << 30;
    const Probe = struct {
        fn numbers(_: *const ModelConfig, _: usize, _: u32, _: transformer_mod.KVQuantConfig, _: bool, _: bool) MemoryBill {
            return .{ .needed = 9 * gb, .available = 10 * gb, .commit = 6 * gb };
        }
    };
    const saved = prefill_admission_numbers;
    prefill_admission_numbers = Probe.numbers;
    defer prefill_admission_numbers = saved;
    var cfg = ModelConfig{ .model_type = "glm5_next" };
    var xfm: Transformer = undefined;
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.transformer = &xfm;
    var a: Slot = undefined;
    a.model = &model;
    a.full_prompt = &.{};
    a.max_tokens = 1;
    a.cache.config = .dense;
    a.cache.entries = &.{};
    a.glm5_request = null;
    a.vision_embeddings = null;
    a.enable_mtp = false;
    a.memory_hold_logged = true;
    a.cancelled = std.atomic.Value(bool).init(false);
    a.finished = false;
    a.error_code = null;
    a.growth_commit = 0;
    var b = a;
    var sch: Scheduler = undefined;
    sch.pending = .empty;
    sch.decoding = .empty;
    defer sch.pending.deinit(testing.allocator);
    defer sch.decoding.deinit(testing.allocator);

    try sch.pending.append(testing.allocator, &a);
    try testing.expectEqual(@as(usize, 1), memoryAdmitCount(&sch, &.{0}));
    try testing.expectEqual(6 * gb, a.growth_commit);

    // A decodes with nothing of its growth resident yet; B arrives a tick later.
    _ = sch.pending.orderedRemove(0);
    try sch.decoding.append(testing.allocator, &a);
    try sch.pending.append(testing.allocator, &b);
    try testing.expectEqual(@as(usize, 0), memoryAdmitCount(&sch, &.{0}));
    try testing.expectEqual(@as(u64, 0), b.growth_commit);

    // Once A finishes its commitment is gone and B proceeds.
    a.finished = true;
    try testing.expectEqual(@as(usize, 1), memoryAdmitCount(&sch, &.{0}));
}

test "a commitment is released as the cache allocates it" {
    try testing.expectEqual(@as(u64, 6), outstandingGrowth(6, 0));
    try testing.expectEqual(@as(u64, 2), outstandingGrowth(6, 4));
    // A DFlash2 request reserves its whole capacity after prefill.
    try testing.expectEqual(@as(u64, 0), outstandingGrowth(6, 6));
    try testing.expectEqual(@as(u64, 0), outstandingGrowth(6, 7));
}

test "the fwd-ubench QSA pooled-key arms interleave composed and fused A B B A inside each verify-rows arm" {
    var buf: [8]UbenchArm = undefined;
    const off = [_]UbenchArm{.{ .verify_rows = false }};
    try testing.expectEqualSlices(UbenchArm, &off, ubenchArms(&buf, 3, false, false, false));
    const qwen = [_]UbenchArm{
        .{ .verify_rows = false, .qsa_pool = false }, .{ .verify_rows = false, .qsa_pool = true },
        .{ .verify_rows = false, .qsa_pool = true },  .{ .verify_rows = false, .qsa_pool = false },
    };
    try testing.expectEqualSlices(UbenchArm, &qwen, ubenchArms(&buf, 3, false, false, true));
    const mimo = ubenchArms(&buf, 3, true, true, true);
    try testing.expectEqual(@as(usize, 8), mimo.len);
    try testing.expectEqualSlices(UbenchArm, &qwen, mimo[0..4]);
    for (mimo[4..], qwen) |got, want| try testing.expectEqual(UbenchArm{ .verify_rows = true, .qsa_pool = want.qsa_pool }, got);
}


pub const PREFILL_DECODE_SHARE_MAX: f32 = 0.9;
pub const DECODE_SHARE_PREFILL_CHUNK: u32 = 1024;
pub fn parseDecodeShare(text: []const u8) error{InvalidDecodeShare}!f32 {
    const v = std.fmt.parseFloat(f32, text) catch return error.InvalidDecodeShare;
    if (std.math.isNan(v) or v < 0) return error.InvalidDecodeShare;
    return @min(v, PREFILL_DECODE_SHARE_MAX);
}
pub fn resolveDecodeShare(flag: ?[]const u8, env: ?[]const u8) error{InvalidDecodeShare}!f32 {
    return parseDecodeShare(flag orelse env orelse return 0);
}
/// The decode share's prefill width cap for this model. GLM prefill numerics depend on the chunk
/// boundaries, so its chunks stay whole whatever else decodes: it yields only between chunks.
pub fn prefillShareCapFor(cfg: *const ModelConfig, decoding: usize, share: f32) u32 {
    return if (cfg.isGlm5()) 0 else decodeShareAdmissionCap(decoding, share);
}
pub fn decodeShareAdmissionCap(decoding: usize, share: f32) u32 {
    return if (decoding == 0 or share <= 0) 0 else DECODE_SHARE_PREFILL_CHUNK;
}
pub fn decodeShareWidthCap(width: u32, decoding: usize, share: f32) u32 {
    const cap = decodeShareAdmissionCap(decoding, share);
    return if (cap == 0) width else @min(width, cap);
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

test "decode share never narrows a GLM prefill: its chunk boundaries are numerics" {
    const glm = try model_mod.parseConfigFromJson(testing.allocator, @embedFile("fixtures/glm5_config.json"));
    const qwen4 = ModelConfig{ .model_type = "qwen4_exp" };
    try testing.expectEqual(@as(u32, 0), prefillShareCapFor(&glm, 3, 0.5));
    try testing.expectEqual(DECODE_SHARE_PREFILL_CHUNK, prefillShareCapFor(&qwen4, 3, 0.5));
    try testing.expectEqual(@as(u32, 0), prefillShareCapFor(&qwen4, 0, 0.5));
}

test "decode share: a live share caps the width only while someone decodes" {
    const w = DECODE_SHARE_PREFILL_CHUNK;
    try testing.expectEqual(@as(u32, 0), decodeShareAdmissionCap(0, 0.5));
    try testing.expectEqual(@as(u32, 0), decodeShareAdmissionCap(2, 0));
    try testing.expectEqual(w, decodeShareAdmissionCap(1, 0.5));
    try testing.expectEqual(@as(u32, 8192), decodeShareWidthCap(8192, 1, 0));
    try testing.expectEqual(@as(u32, 8192), decodeShareWidthCap(8192, 0, 0.5));
    try testing.expectEqual(w, decodeShareWidthCap(8192, 1, 0.5));
    try testing.expectEqual(@as(u32, 512), decodeShareWidthCap(512, 1, 0.5));
}

test "decode share: parse clamps above 0.9 and rejects anything that is not a share" {
    try testing.expectEqual(@as(f32, 0), try parseDecodeShare("0"));
    try testing.expectEqual(@as(f32, 0.5), try parseDecodeShare("0.5"));
    try testing.expectEqual(PREFILL_DECODE_SHARE_MAX, try parseDecodeShare("0.95"));
    try testing.expectEqual(PREFILL_DECODE_SHARE_MAX, try parseDecodeShare("inf"));
    for ([_][]const u8{ "", "abc", "nan", "-0.3", "0.5x" }) |bad| {
        try testing.expectError(error.InvalidDecodeShare, parseDecodeShare(bad));
    }
    // The flag outranks the env; neither set is 0; a bad env is an error, not 0.
    try testing.expectEqual(@as(f32, 0.3), try resolveDecodeShare("0.3", "0.5"));
    try testing.expectEqual(@as(f32, 0.5), try resolveDecodeShare(null, "0.5"));
    try testing.expectEqual(@as(f32, 0), try resolveDecodeShare(null, null));
    try testing.expectError(error.InvalidDecodeShare, resolveDecodeShare(null, "abc"));
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



test "decode share: the kill switch disables the effective share and only live slots count" {
    const saved_share = prefill_decode_share;
    const saved_interleave = prefill_interleave_cached;
    defer { prefill_decode_share = saved_share; prefill_interleave_cached = saved_interleave; }
    prefill_decode_share = 0.5;
    prefill_interleave_cached = false;
    try testing.expectEqual(@as(f32, 0), prefillDecodeShare());
    prefill_interleave_cached = true;
    try testing.expectEqual(@as(f32, 0.5), prefillDecodeShare());
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    sch.queue_mu = .init;
    var slots: [4]Slot = undefined;
    var ptrs: [4]*Slot = undefined;
    for (&slots, &ptrs) |*slot, *ptr| {
        slot.cancelled = std.atomic.Value(bool).init(false);
        slot.finished = false;
        slot.error_code = null;
        ptr.* = slot;
    }
    slots[1].cancelled.store(true, .release);
    slots[2].finished = true;
    slots[3].error_code = "test error";
    sch.decoding = .empty;
    sch.decoding.items = &ptrs;
    sch.decoding.capacity = ptrs.len;
    try testing.expectEqual(@as(usize, 1), liveDecodingCount(&sch));
    slots[0].finished = true;
    try testing.expectEqual(@as(usize, 0), liveDecodingCount(&sch));
    try testing.expectEqual(std.math.maxInt(u64), decodeShareBudgetNs(std.math.maxInt(u64), 0.9));
    try testing.expectEqual(@as(f32, 0), try resolveDecodeShare("0", "invalid"));
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

test "culling an errored or cancelled GLM slot releases its native state, a live one keeps it" {
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
    model.id = "org/glm-cull";
    model.prefix_cache = null;
    var slots: [3]Slot = undefined;
    for (&slots, 0..) |*slot, i| {
        slot.model = &model;
        slot.cache = .{ .entries = &.{}, .step = 0, .allocator = testing.allocator, .config = .dense };
        slot.ssm_entries = null;
        slot.ring_cps = .{};
        slot.restored_entry = 0;
        slot.full_prompt = &.{};
        slot.prompt_tokens = 4;
        slot.completion_tokens = 0;
        slot.cached_tokens = 0;
        slot.request_id = i;
        slot.max_tokens = 8;
        slot.request_start_ts = std.Io.Timestamp.now(testing.io, .boot);
        slot.legacy_gen = null;
        slot.finished = false;
        slot.error_code = null;
        slot.cancelled = .init(false);
        slot.glm5_request = try glm5_forward_mod.Request.init(testing.allocator, 2);
        slot.glm5_request.?.offset = 4;
        try sch.decoding.append(testing.allocator, slot);
    }
    defer for (&slots) |*slot| slot.glm5_request.?.deinit();
    slots[0].error_code = "GlmReserveMemoryLimit";
    slots[1].cancelled.store(true, .release);
    cullDecoding(&sch);
    try testing.expectEqual(@as(usize, 1), sch.decoding.items.len);
    try testing.expectEqual(&slots[2], sch.decoding.items[0]);
    try testing.expectEqual(@as(usize, 0), slots[0].glm5_request.?.offset);
    try testing.expectEqual(@as(usize, 0), slots[1].glm5_request.?.offset);
    try testing.expectEqual(@as(usize, 4), slots[2].glm5_request.?.offset);
}

test "a GLM cancel landing between the cull passes is committed once before its native reset" {
    const Hook = struct {
        var commits: [2]usize = .{ 0, 0 };
        var offset_at_commit: [2]usize = .{ 0, 0 };
        var slots: [2]*Slot = undefined;
        fn commit(_: *Scheduler, s: *Slot) void {
            const i: usize = if (s == slots[0]) 0 else 1;
            commits[i] += 1;
            offset_at_commit[i] = s.glm5_request.?.offset;
            slots[1].cancelled.store(true, .release);
        }
    };
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    sch.allocator = testing.allocator;
    sch.queue_mu = .init;
    sch.metrics = null;
    sch.live_session_count = 0;
    sch.decoding = .empty;
    defer sch.decoding.deinit(testing.allocator);
    var model: model_registry_mod.LoadedModel = undefined;
    model.id = "org/glm-cull-race";
    model.prefix_cache = null;
    var slots: [2]Slot = undefined;
    for (&slots, 0..) |*slot, i| {
        slot.model = &model;
        slot.cache = .{ .entries = &.{}, .step = 0, .allocator = testing.allocator, .config = .dense };
        slot.ssm_entries = null;
        slot.ring_cps = .{};
        slot.restored_entry = 0;
        slot.full_prompt = &.{};
        slot.prompt_tokens = 4;
        slot.completion_tokens = 0;
        slot.cached_tokens = 0;
        slot.request_id = i;
        slot.max_tokens = 8;
        slot.legacy_gen = null;
        slot.finished = false;
        slot.error_code = null;
        slot.cancelled = .init(false);
        slot.glm5_request = try glm5_forward_mod.Request.init(testing.allocator, 2);
        slot.glm5_request.?.offset = 4;
        Hook.slots[i] = slot;
        try sch.decoding.append(testing.allocator, slot);
    }
    defer for (&slots) |*slot| slot.glm5_request.?.deinit();
    Hook.commits = .{ 0, 0 };
    slots[0].cancelled.store(true, .release);

    cullDecodingWith(&sch, Hook.commit);
    // The late cancel stays in `decoding` with its state, and the next cull commits it.
    try testing.expectEqual(@as(usize, 1), sch.decoding.items.len);
    try testing.expectEqual(@as(usize, 4), slots[1].glm5_request.?.offset);
    cullDecodingWith(&sch, Hook.commit);
    try testing.expectEqual(@as(usize, 0), sch.decoding.items.len);
    try testing.expectEqual([2]usize{ 1, 1 }, Hook.commits);
    try testing.expectEqual([2]usize{ 4, 4 }, Hook.offset_at_commit);
    try testing.expectEqual(@as(usize, 0), slots[1].glm5_request.?.offset);
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
    try testing.expectEqual(@as(usize, 0), liveDecodingCount(&sch));
    cullDecoding(&sch);
    try testing.expectEqual(@as(usize, 0), sch.decoding.items.len);
}

test "postPrefillTerminal releases native GLM state for a cancelled slot but not a live one" {
    var cancelled_slot: Slot = undefined;
    cancelled_slot.state = .decoding;
    cancelled_slot.cancelled = .init(true);
    cancelled_slot.glm5_request = try glm5_forward_mod.Request.init(testing.allocator, 2);
    cancelled_slot.glm5_request.?.offset = 4;
    defer cancelled_slot.glm5_request.?.deinit();

    var live_slot: Slot = undefined;
    live_slot.state = .decoding;
    live_slot.cancelled = .init(false);
    live_slot.glm5_request = try glm5_forward_mod.Request.init(testing.allocator, 2);
    live_slot.glm5_request.?.offset = 4;
    defer live_slot.glm5_request.?.deinit();

    var errored_slot: Slot = undefined;
    errored_slot.state = .errored;
    errored_slot.cancelled = .init(false);
    errored_slot.glm5_request = null;

    try testing.expect(postPrefillTerminal(&cancelled_slot));
    try testing.expectEqual(@as(usize, 0), cancelled_slot.glm5_request.?.offset);

    try testing.expect(!postPrefillTerminal(&live_slot));
    try testing.expectEqual(@as(usize, 4), live_slot.glm5_request.?.offset);

    try testing.expect(postPrefillTerminal(&errored_slot));
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
    if (mlx.noGpuBackend()) return error.SkipZigTest;
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

test "a queued vision encode is refused when its bill no longer fits, and an earlier output counts as spent" {
    const Probe = struct {
        var capacity: u64 = 0;
        var resident: u64 = 0;
        var tower_runs: usize = 0;
        fn available(_: *const ModelConfig) u64 {
            return capacity -| resident;
        }
        fn tower(_: VisionImagePixels) anyerror!mlx.mlx_array {
            tower_runs += 1;
            resident += 100;
            const rows = [_]f32{ 0, 0, 0, 0 };
            return mlx.mlx_array_new_data(&rows, &[_]c_int{ 1, 1, 4 }, 3, .float32);
        }
        fn run(sch: *Scheduler, model: *LoadedModel, bill: u64) !VisionEncodeRequest {
            const pixels = [_]VisionImagePixels{.{ .pixels = &.{}, .width = 1, .height = 1 }};
            var req = VisionEncodeRequest{ .model = model, .images = &pixels, .bill_bytes = bill, .allocator = testing.allocator };
            runVisionEncode(sch, &req);
            return req;
        }
    };
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    var cfg = ModelConfig{ .num_hidden_layers = 0 };
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.vision_encoder = @ptrFromInt(@alignOf(VisionEncoder));
    vision_block_test_hook = Probe.tower;
    vision_encode_available = Probe.available;
    defer {
        vision_block_test_hook = null;
        vision_encode_available = null;
    }

    // Headroom shrank after the connection thread's check: the tower never runs.
    Probe.capacity = 50;
    Probe.resident = 0;
    Probe.tower_runs = 0;
    const refused = try Probe.run(&sch, &model, 100);
    defer if (refused.error_name) |e| testing.allocator.free(e);
    try testing.expectEqualStrings(vision_headroom_error, refused.error_name orelse return error.TestExpectedRefusal);
    try testing.expectEqual(@as(u64, 50), refused.available_bytes);
    try testing.expect(refused.done);
    try testing.expect(refused.result == null);
    try testing.expectEqual(@as(usize, 0), Probe.tower_runs);

    // Two encodes queued together: the first fits, and its output (held by its connection thread)
    // leaves the second without room.
    Probe.capacity = 150;
    Probe.resident = 0;
    const first = try Probe.run(&sch, &model, 100);
    defer if (first.result) |r| {
        _ = mlx.mlx_array_free(r);
    };
    try testing.expect(first.error_name == null);
    try testing.expectEqual(@as(usize, 1), Probe.tower_runs);
    const second = try Probe.run(&sch, &model, 100);
    defer if (second.error_name) |e| testing.allocator.free(e);
    try testing.expectEqualStrings(vision_headroom_error, second.error_name orelse return error.TestExpectedRefusal);
    try testing.expectEqual(@as(u64, 50), second.available_bytes);
    try testing.expectEqual(@as(usize, 1), Probe.tower_runs);

    // An unbilled request (a tower with no bill) is never refused here.
    const unbilled = try Probe.run(&sch, &model, 0);
    defer if (unbilled.result) |r| {
        _ = mlx.mlx_array_free(r);
    };
    try testing.expect(unbilled.error_name == null);
}
