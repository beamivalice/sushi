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
- `--kv-quant 8|4|16` (16 = dense BF16, also spelled `off`/`0`; `src/kv_quant.zig`, `configuredKvQuantFor(config)`); an explicit flag beats
  `model-settings.json` `kv_quant`, which beats the default. A per-model setting of `kv_quant: off` still wins over
  the default.
- GLM (`glm5_next`) takes the kv8 default too, through its own MLA latent backend (`glm5_latent.zig`,
  [glm5-arch](glm5-arch.md#scope)); kv4 is refused.
- `--kv-attn-mode auto|dense|fused` picks the packed-read arm (`auto` from an 8K PROMPT, fixed at admission); MiMo's
  global-layer decode ignores it and picks per step from the cache's length. `--decode-attn-quant` (default ON,
  LOSSY) requants dense attention at decode AND verify.

## The kv-quant contract

- Attention reads every cache through `KVCache.denseView` on EVERY path (batched included). The returned
  `DenseKVView` carries the packed triple when the cache is quantized (`has_quant_triple`), so a kernel either reads
  it packed or gets the dense rebuild; nothing else touches the storage.
- Schemes extend via enum + two switch arms; a prefix-cache entry records its scheme (`Entry.quant_config`), so a
  slot running `kv_quant=4` never restores from an entry committed at another width
  (`tests/test_kv_quant_per_request.sh`). Every generation endpoint, `/v1/completions` included, reads the per-request
  `kv_quant` through `parseKvQuantOverride` and logs the override it applied.
- Packed reads are kernel-or-DENSE per WIDTH (`kvAttnFusedEligible` t_q==1, `kvAttnVerifyEligible` t_q 2..8; verify
  kernel OFF on G17, `SUSHI_KV_ATTN_VERIFY=1|0`; floor 2048). Guard: `tests/test_kv_quant_fused_equivalence.sh`.
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
- A lazily copied side-channel state is not in the residual's graph: name the owned copy in the cadence eval vector
  (`evalCadencePoint`, the `conv1dWithCache` tail).

<a id="gdn"></a>
## GDN trunks

A GDN trunk's `KVCache.step` is 0 forever (it advances on layer 0, a linear layer): batched rope offsets read the
slot's `moe_seq_offset`; the pad-waste cap reads `KVCache.kvLenForBatching`. Batched N=2 acquits near-ties
(≤ 0.15 nats).

- The recurrent state is stored bf16 between tokens (the math runs in f32). `SUSHI_GDN_STATE_F32=1` stores it f32
  (`ModelConfig.gdnStateDtype`, one `[gdn] recurrent state f32` line): a Qwen checkpoint grows 58.8 → 115.5 MB, billed
  by `ssmCheckpointBytes`, and a restored bf16 state is widened before its next step.

## Byte stability

- INT4 long-greedy divergence is legit (the AR/verify INT4 kernel float-noise tail).
- Byte-stable greedy ⇒ no spec + `--kv-quant off/8` + `--prefix-cache-entries 0`.
- A prompt longer than one chunk is byte-stable only at a fixed prefill width: MiMo at 64k gives a first-token
  logprob of -1.0685 at 4096 and -1.0807 at 2048 (each width repeats exactly), and the per-request ladder narrows
  the width when memory is short. `--prefill-chunk N` caps it.
- A restore is not bit-identical on a hybrid ([engine-prefix-cache](engine-prefix-cache.md)).
