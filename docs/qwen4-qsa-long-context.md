# Engine: QSA sparse attention and long context (qwen4_exp)

How Qwen3.8-Flash-Next attends past 2048 tokens: the QSA indexer, block selection, the gather/split-K arms per query
width, the indexer history, and the long-context admission and load-time bills. Every mechanism here is gated by ONE
predicate, `ModelConfig.longCtxGated()`. Read this before touching any `qsa*` function in `src/transformer.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [qwen4-arch](qwen4-arch.md),
[engine-kv-cache](engine-kv-cache.md), [engine-memory-admission](engine-memory-admission.md),
[engine-mtp](engine-mtp.md), [perf-baselines](qwen4-perf.md#qsa).

## Attention arms by query width

- **QSA GATHERS selected blocks, never a dense `[S, kv]` mask** on a packed cache.
- Selection = the exact radix-select `sushi_qsa_select` on both arms, SPLIT across 16 threadgroups at decode widths
  (`SUSHI_QSA_SELECT_SPLIT=0`).
- Rows 1..15 (decode and verify) = the fused split-K `qsaSparseAttn` on dense and packed caches alike
  (`qsaAlignedSparseAttn`; `SUSHI_QSA_ATTN_MIN_S` raises its floor). When it declines, decode falls to
  `qsaDecodeGatherAttn` and rows 2..15 to the union gather `qsaVerifyGatherAttn`
  (`SUSHI_QSA_ATTN_KERNEL=0` forces the union gather).
- **A PACKED (kv8/kv4) cache never takes the dense-mask arm and never rebuilds whole per layer**
  (`qsaSparseAttnServes`): split-K serves up to `QSA_ATTN_PACKED_MAX_S` = 40 rows; past that
  `gatherQsa256Packed` gathers the packed rows in place while each cache row is staged at most
  `QSA_PACKED_GATHER_MAX_REUSE` = 4 times (`qsaPackedGatherServes`), else a gather over ONE rebuild.
  `kvDequantScratchBytes` bills the rebuild per forward width (`qsaDenseRebuildRows`); `SUSHI_QSA_ATTN_KERNEL=0`
  restores the old route and its bill.
- Prefill = `gatherQsa256`; batched slots keep the dense mask up to the gather floor: (top-512 + 1) x 4 = 2052 keys
  where the NAX gather serves (`qsaPrefillGatherMinKv`), 8192 elsewhere; decode and verify floors are unchanged.
- **Verify gather kv floor is per KV SCHEME** (`qsaVerifyGatherMinKvFor`: dense 32768, quantized 16384).

## Small sparse groups at long context

A batched group is never split by the pad-waste cap when its longest BILLED length is at most `PAD_FREE_KV` (4096) and
its longest TRUE context is at least `PAD_FREE_MIN_CTX` (131072) (`groupKeepCount`, decode tick and `runMtpGroups`; the MTP
planner gate keeps the plain cap). Under the gather arm every long slot bills at the indexer budget (2052 rows), so a
short sub-agent beside a long session read as a 2x pad and sent a slot through a whole serial forward every tick. Below
131072 tokens that serial forward is cheap and the plain cap still splits. The group runs the batched path that
already ships, so its output differs from the serial one only by the documented batched-versus-solo Qwen tolerance.
This floor now serves the grouped MTP verify and a streamed load only: a resident plain decode batch pads nothing.

- Measured (this change on d3c65fc6, Sushi-2.6bpw, kv8, one ~134k-token session beside two ~200-token requests, 256
  tokens each, decode-only tok/s of the aggregate, `taskpolicy -a`, GPU lock held, boot-interleaved old/new arms): `--no-mtp
  --no-pld` 59 -> 97-99 (96.8-99.4 floor on, 59.1-62.1 off); default MTP on 65.1-66.0 -> 95.6-96.2.
  Engagement: the old arm logs `[batched] pad-waste cap: kept 0 of 2 slots (waste 1.97x, kv_len 32..2052)`, the new arm does not.

## Batched image and text streams

M-RoPE slots use the same batched QSA gather as text slots. Queries are rotated before attention, cached keys
already carry their positions, and batched RoPE offsets include each slot's M-RoPE delta. Keeping an image slot's
selected blocks avoids forcing the whole group onto a dense mask over the full KV cache. The scheduler bills
pad waste using the selected length whenever the gather for that query width is enabled; its existing switches
and KV floors still apply.

Ported from [mlx-serve #668](https://github.com/ddalcu/mlx-serve/pull/668). The regression test compares the same
batched attention inputs with and without M-RoPE metadata and requires identical output. The scheduler test
checks that a 300k/1k pair remains grouped under sparse billing. This does not change MTP-head batching's
separate M-RoPE guard.

## Selection semantics

- **The always-visible tail is PER QUERY** (tokens at/after `ratio·floor((p+1)/ratio)`), scores in f32 like the
  reference, `torch.topk` keeps the LOWER block index on exact-zero ties.
- QSA caps attended keys at top-512 blocks x 4 = 2048 per query: decode attention reads ~25 MB/token at kv8, under
  1% of the weights, which is why generic kv8 attention kernels never engage on this arch.

## Indexer

- The score sheet is ONE NAX kernel (`sushi_qsa_score`, bit-identical to the stock tf32 chain;
  `SUSHI_QSA_SCORE_FUSED=0`).
- The prefill gather rides NAX cooperative tensors on its own predicate (`qsaNaxEligible`: G17 + macOS 26.3 + bf16 +
  hd 256 + gqa 12 + q_len ≥ 16; bar = per-element error vs float64 no worse than stock,
  `tests/qsa_nax_precision.py`, never bytes). The packed NAX variant joins the NAX probe.
- The dense NAX gather is oMLX's occupancy-tuned kernel (mlx-serve #636): one (query, KV head) per threadgroup, the
  12 grouped heads as rows of a 16-row tile, no K/V staging, P as fp16 hi+lo; packed kv4/kv8 gathers keep the old
  kernel. `SUSHI_PROFILE_ATTN=1` logs synced indexer / projection / QSA / tail laps once per forward (`[qwen4-attn]`).
- One effective YaRN mscale on every indexer arm; the indexer ropes with the SAME M-RoPE table as attention.
- The pooled block keys are ONE kernel (`sushi_qsa_pool_rope`, from mlx-serve #556): block mean, key norm and
  partial RoPE, bit-identical to the MLX chain. It serves text turns, bf16, 128-wide keys and ratios up to 8; M-RoPE
  turns and other shapes keep the chain. No env lever; `SUSHI_DECODE_FWD_UBENCH_QSA_POOL_ARMS=1` is its A/B
  ([perf-baselines](qwen4-perf.md#qsa-pool-rope)).

## Indexer history

- ONE copy per (slot ∪ entry); the newest snap VIEWS the live buffer at commit (`handoffQsaHistoryToLatest`); none
  after restore = MISS; the f32 score bank is billed; raw keys are a 32-row ring billed once per slot
  (`qsaRingBytes`).
- Per-request state outside conv/ssm rides `SSMCacheEntry.aux_state` + `ple_prev`, freed only through
  `ssmFreeQsaState`. Never hand a null `mlx_array` to the spec tensor-map insert (a pooled-only head arrives with
  `aux_state.ctx == null`).
- A cache keyed on a POINTER is invalidated by an ATOMIC MARK (`markQsaPooledRopeStale`), never an off-thread free.

## Admission and load-time bills

- Past 32k a request RESERVES its KV capacity up front (`KVCache.reservedTokens`; `SUSHI_KV_RESERVE=0`).
- A long prefill EVICTS the hot cache to be admitted on the INFERENCE thread (`evictLruToAdmit`, credits only
  PROVABLY reclaimable bytes, refuses by NAME `PrefillDoesNotFit` → 400, defers a warm prompt to the `WarmPrefix`
  bill); ONE `[admission]` line.
- The prefill width is per-REQUEST and re-chosen per CHUNK (`chooseRequestPrefillChunk`, `adaptivePrefillWidth`).
- **Load-time bills** run INSIDE `Scheduler.init`: KV width from `configuredKvQuantFor(config)`, context from
  `resolvedContextForLoad` with the CONSTANT `CTX_SIZING_CACHE_RESERVE`, the hot-cache clamp reserves the ladder
  FLOOR. A KV bill is per CACHING LAYER (`kvBytesPerToken` via `attnCacheLayerCount`); `prefillStreamBytesPerToken`
  adds the arch's own streams (GDN chunk-wide q/k/v, MoE `top_k` replication); the chunk-independent part is a
  RUNTIME floor.
- The general admission and memory rules: [engine-memory-admission](engine-memory-admission.md).
