# Engine: KV cache and kv-quant

What the KV cache stores, how a quantized cache is read, how it grows, and which settings give byte-stable
output. Read this before touching `KVCache` in `src/transformer.zig`, `src/kv_quant.zig`, or any attention path that
reads cached keys.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-prefix-cache](engine-prefix-cache.md),
[qwen4-qsa-long-context](qwen4-qsa-long-context.md), [mimo2-arch](mimo2-arch.md) (sliding-window ring),
[engine-memory-admission](engine-memory-admission.md), [engine-kernels](engine-kernels.md).

## Defaults and flags

- **kv8 is the engine default.** Every load logs `[kv-cache] <scheme> (<source>)`, i.e.
  `[kv-cache] <kv8|kv4|off> (<default|--kv-quant|model-settings.json>)`
  (scheduler adds `; ctx N (source)`); `/props` reports `settings.kv_cache`, `/v1/models` `meta.kv_cache`.
  `kld capture|compare` teachers stay dense; the qwen4 MTP head KV is dense unless `--mtp-head-kv-quant`.
- `--kv-quant 8|4|16|tbq4|tbq3` (16 = dense BF16, also spelled `off`/`0`; `tbq4`/`tbq3` = rotated fixed-codebook
  storage, `src/kv_quant.zig`, `configuredKvQuantFor(config)`); an explicit flag beats
  `model-settings.json` `kv_quant`, which beats the default. A per-model setting of `kv_quant: off` still wins over
  the default.
- GLM (`glm5_next`) takes the kv8 default too, through its own MLA latent backend (`glm5_latent.zig`,
  [glm5-arch](glm5-arch.md#scope)); kv4 is refused.
- The packed-read arm is `auto` (fused from an 8K PROMPT, fixed at admission); a request's `kv_attn_mode` field
  (`dense` or `fused`) overrides it. MiMo's global-layer decode ignores both and picks per step from the cache's
   length.

## The kv-quant contract

- Attention reads every cache through `KVCache.denseView` on EVERY path (batched included). The returned
  `DenseKVView` carries the packed triple when the cache is quantized (`has_quant_triple` affine, `tbq_triple`
  TBQ), so a kernel either reads it packed or gets the dense rebuild; nothing else touches the storage.
- Schemes extend via enum + two switch arms; a prefix-cache entry records its scheme (`Entry.quant_config`), so a
  slot running `kv_quant=4` never restores from an entry committed at another width
  (`tests/test_kv_quant_per_request.sh`). Every generation endpoint, `/v1/completions` included, reads the per-request
  `kv_quant` through `parseKvQuantOverride` and logs the override it applied.
- Packed reads are kernel-or-DENSE per WIDTH for both schemes (`kvAttnFusedEligible` t_q==1, `kvAttnVerifyEligible`
  t_q 2..8; verify kernel OFF on G17, `SUSHI_KV_ATTN_VERIFY=1|0`; floor 2048). The TBQ fused kernels
  (`qkv_attn_dec_tbq`) dequantize packed rows in-kernel in the rotated domain — Q pre-rotated by the folded M1,
  the merged output inverse-rotated by M2 = M1ᵀ — so decode/verify never rebuild the whole cache; the dense
  fallback stays a whole-cache dequant and the affine path is bit-identical.
  Guard: `tests/test_kv_quant_fused_equivalence.sh`.
- TBQ writes quantize in ONE dispatch: the fused Metal encoder `sushi_tbq_encode` (`kv_quant.quantizeTBQ`; one
  threadgroup per 128-block runs norm → s1 → FWHT (register shuffles for stages 1..16, threadgroup scratch for
  32/64) → [1/√128] → s2 → midpoint search → 64 B (tbq4) or 48 B (tbq3) fixed-point pack + the fp16 norms and
  their biases copy). The ~25-op composed chain remains the fallback and any `SUSHI_TBQ_ENCODE_FUSED=0` restore.
  The butterfly accumulates in a different order than the composed 128×128 matmul, so a value within ~1 ULP of a
  midpoint can bucket one code away: RMSE is preserved, bit-exactness with the composed path is NOT (pinned by
  the fused-vs-composed tolerance tests in `src/kv_quant.zig`).
- TBQ subset reads (QSA decode/verify gathers) stage rows in ONE dispatch:
  `sushi_tbq_take` gathers them through the arm's own index array and emits
  **rotated-domain** values — the query is rotated once around the SDPA
  (per-128-block orthogonal; the row-indexed additive mask rides unchanged)
  and the raw output folds back once. Measured on a grown cache at 67k rows:
  0.52 ms/layer/step vs 1.34 composed (−61%) and 0.72 kv8 affine — TBQ
  subset reads now cost *less* than 8-bit. Geometry/dtype mismatch declines
  silently to the composed chain; force it with `SUSHI_TBQ_TAKE_FUSED=0`.
  Engage line: `[qsa-take] fused engaged: bits=…`.
  Routing note: the verify floor (`qsaVerifyGatherServes`) bills `isQuant`
  schemes — TBQ takes the 16384-row QUANT floor, not the 32768 dense one;
  billing it as dense pinned interleaved MTP ticks at 16k-32k rows to the
  whole-cache rebuild and refused a 67k prompt on a live tbq4 server.
- TBQ split-K arm (Chunk 6): `qsaSparseAttn` serves tbq views at every width
  `qsaSparseAttnServes` admits through `sushi_attn_qsa256_qsplit_tbq` — the split
  pass stages packed rows through `sushi_qsa_tbq8` (no per-tick dense subset at
  all), M1 folds into the queries once per tick, M2 folds the merged output once
  (softmax linearity makes one post-merge fold equal NSPLIT per-split folds), the
  affine merge kernel is reused byte-identical. It replaced the gather arms as the
  FIRST choice after a tbq4 server run decoded 8-12 t/s vs kv8's 42: the split-K
  decline (`if (kv_view.tbq_triple) return null`) had left only per-tick
  materialization — ~+4.5 GB active_bytes and 53 ms/tok serial. Measured per
  layer-tick at kv=36864: split-K 0.253 (S=8) / 0.350 (S=1) vs fused-take
  0.876/1.029 and affine8 0.955/1.214. `SUSHI_QSA_ATTN_KERNEL=0` restores the
  gather arms; `qsaAlignedSparseAttn` still passes tbq only when the fused prefix
  budget is free (cut>0 declines, so the per-row prefix SDPA never evaluates the
  lazy whole-cache dequant). Engage line: `[qsa-attn] engaged (S=… quant=tbq4/gs128 …)`.
- Arch-specific packed readers: QSA on qwen4_exp ([qwen4-qsa-long-context](qwen4-qsa-long-context.md)); the
  matmul2d decode kernel and the fused prefill on MiMo's global layers ([mimo2-arch](mimo2-arch.md#attention-kernels)).
- `server.kvDequantScratchBytes` bills a dense rebuild as ONE layer at the rows that layer stores (per forward width
  on QSA).

## Growth and lifetime

- Affine scales and biases retain the quantizer's activation dtype at allocation and growth; dense reconstruction returns that same dtype, so f16 attention does not widen through bf16 K/V.

- KV growth is PROPORTIONAL (`nextCapacity` +25%, capped 8192). Past 32k a request reserves its capacity up front
  (`KVCache.reservedTokens`); a ringed arch reserves always (`ModelConfig.reservesKvCapacity`).
- **A shared restore whose buffer is longer than the request reserves copies its rows into a request-sized buffer**
  on the first append (`KVCache.copiesSharedRestore`), because the admission bills the copy at the reservation: a 4k
  chat restored from an 80k entry held the 80k buffer (mlx-serve #492), and a donor that decoded past its own
  reservation was copied past the bill. A donated checkout owns its buffers (`adoptRestored`) and appends in place.
- **A fallible re-init BEHIND a `deinit` leaves a freed object on the error path** — build first, then swap
  (`KVCache.reinit`). A handle freed before a fallible op is reset AT the free (`updateDense`).
- **A leaked owned reshape VIEW handle keeps its source primitive's DEVICE buffer pinned** even after the source handle is freed — an `errdefer`-only `mlx_array_free` of the view leaks it on the SUCCESS path. Three tbq sites did exactly this (fused-encoder input view `xr` in `kv_quant.quantizeTBQFused`, gather-output view `orows` in `transformer.gatherQsa256Packed`, split-K merged view `flat` in the `transformer.qsaSparseAttn` tbq branch) and ratcheted a live tbq4 server from 44.47 to 58.5 GB `active_bytes` (96 MB / ~2.3 MB / ~588 MB per request at server geometry; a 32k admission refusal and a 64k-row METAL OOM downstream). Fix `b841bda9`: `defer`, not `errdefer`, wherever a lazy view of a buffer is created on the read/write path; the retention floor is pinned by transformer.zig test "tbq write and arm paths release every pinned device buffer (floor)" (per-iteration arm-engagement bits + <256 KB `mlx_get_active_memory` retention after eval+free+clear_cache). When measuring retention, `mlx_synchronize(stream)` before sampling `mlx_get_active_memory` — async Metal releases otherwise read as retained bytes (an affine8 control stage flapped 0/511 MB until the probe drained them).
- A lazily copied side-channel state is not in the residual's graph: name the owned copy in the cadence eval vector
  (`evalCadencePoint`, the `conv1dWithCache` tail).

<a id="gdn"></a>
## GDN trunks

A GDN trunk's `KVCache.step` is 0 forever (it advances on layer 0, a linear layer): batched rope offsets read the
slot's `moe_seq_offset`; the pad-waste cap reads `KVCache.kvLenForBatching`. Batched N=2 acquits near-ties
(≤ 0.15 nats).

- The decay gate `g` is f32 at every producer (`ModelConfig.gdnGateDtype`, the `GDN gate compiled ... g f32` line): a
  bf16 gate gave a head whose exact g is within 2^-9 of 1 g == 1.0, so it never forgot (206 of 1728 heads in our
  Flash-Next packs). The gate is never stored. The recurrent state is stored bf16 between tokens (the math runs in f32),
  which still rounds a per-token decay under 2^-9 away: `--qwen-gdn fp32` stores it f32 (`ModelConfig.gdnStateDtype`,
  one `[gdn] recurrent state f32` line) and a slow head then forgets at its exact rate. A Qwen checkpoint grows
  58.8 → 115.5 MB, billed by `ssmCheckpointBytes`, and a restored bf16 state is widened before its next step. GLM's
  state is KDA, not GDN: f32 by construction, so the flag does not apply and the bill follows the arch, not the flag.
- Measured against the f32-gate default: the f32 state's KLD gain is inside the noise ([qwen4-kld](qwen4-kld.md#gdn-f32-teacher)),
  its cost is about 2% of 4-stream aggregate and +54 MB per checkpoint ([qwen4-perf](qwen4-perf.md#gdn-f32-cost)).

## Byte stability

- INT4 long-greedy divergence is legit (the AR/verify INT4 kernel float-noise tail).
- Byte-stable greedy ⇒ no spec + `--kv-quant off/8` + `--prefix-cache-entries 0`.
- A prompt longer than one chunk is byte-stable only at a fixed prefill width: MiMo at 64k gives a first-token
  logprob of -1.0685 at 4096 and -1.0807 at 2048 (each width repeats exactly), and the per-request ladder narrows
  the width when memory is short. `--prefill-chunk N` caps it.
- A restore is not bit-identical on a hybrid ([engine-prefix-cache](engine-prefix-cache.md)).
