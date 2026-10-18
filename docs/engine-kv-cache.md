# Engine: KV cache and kv-quant

What the KV cache stores, how a quantized cache is read, how it grows, and which settings give byte-stable
output. Read this before touching `KVCache` in `src/transformer.zig`, `src/kv_quant.zig`, or any attention path that
reads cached keys.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-prefix-cache](engine-prefix-cache.md),
[qwen4-qsa-long-context](qwen4-qsa-long-context.md), [mimo2-arch](mimo2-arch.md) (sliding-window ring),
[engine-memory-admission](engine-memory-admission.md), [engine-kernels](engine-kernels.md).

## Defaults and flags

- **kv8 is the engine default.** Every load logs `[kv-cache] <scheme> (<source>)`, i.e.
  `[kv-cache] <kv8|kv6|kv4|off> (<default|--kv-quant|model-settings.json>)`
  (scheduler adds `; ctx N (source)`); `/props` reports `settings.kv_cache`, `/v1/models` `meta.kv_cache`.
  `kld capture|compare` teachers stay dense; the qwen4 MTP head KV is dense unless `--mtp-head-kv-quant`.
- `--kv-quant 8|6|4|16` (16 = dense BF16, also spelled `off`/`0`; `src/kv_quant.zig`, `configuredKvQuantFor(config)`); an explicit flag beats
  `model-settings.json` `kv_quant`, which beats the default. A per-model setting of `kv_quant: off` still wins over
  the default.
- GLM (`glm5_next`) takes the kv8 default too, through its own MLA latent backend (`glm5_latent.zig`,
  [glm5-arch](glm5-arch.md#scope)); kv6 is served the same way (416 bytes per 512-wide latent row against kv8's 544), kv4 is refused.
- Nothing picks a packed read per request: MiMo's global-layer decode chooses its packed arm per step from the
  cache's length (`mimoGlobalDecodeArm`), and Flash-Next's QSA arms follow the indexer.

## The kv-quant contract

- Attention reads every cache through `KVCache.denseView` on EVERY path (batched included). The returned
  `DenseKVView` carries the packed triple when the cache is quantized (`has_quant_triple`), so a kernel either reads
  it packed or gets the dense rebuild; nothing else touches the storage.
- Schemes extend via enum + two switch arms; a prefix-cache entry records its scheme (`Entry.quant_config`), so a
  slot running `kv_quant=4` never restores from an entry committed at another width
  (`tests/test_kv_quant_per_request.sh`). Every generation endpoint, `/v1/completions` included, reads the per-request
  `kv_quant` through `parseKvQuantOverride` and logs the override it applied.
- **kv6 sits between kv4 and kv8**: 6-bit affine, group 64, so a row bills 6.5 bits per element against kv8's 8.5.
  `mlx_quantize` packs four values into three bytes (a little-endian bitstream), so a row is `D * 6 / 32` words and
  a value can straddle two of them: every custom reader unpacks it bytewise (`sushi_kv_unpack8`) or in three-word
  units (matmul2d), and each is pinned against its dense oracle at 6 bits. The memory bill is the generic
  `(2 * bits + 1) / 32` of the dense row, so no bill changed. The batched affine append serves kv6 and kv8 (kv4 stays
  per slot). The fused MiMo QKV prep serves kv6 as it does kv8 (the six-bit pack writes four values as three bytes, two
  lanes per three bytes; bit-identical to the composed rope + `quantizeAffine`). The MiMo EXL3 round-cost surface
  (`MTP_EV_MIMO_EXL3_COSTS`) serves every width: its forward meter reads the same at kv6 and kv8 (below), so kv6 needs no
  row of its own. Qwen EXL3 packs plan under the chip's generic row at every width; the affine mixed 4/8 Qwen surface
  stays kv8-only (no such pack was available to measure at kv6).
- **GLM kv6** (`glm5_latent.zig`): `Latent.bits()` reads the stored width from the code columns against the scale columns;
  `sushi_latent6` unpacks a value from at most two bytes of its row and the kernels that read the latent (scalar and
  packed sparse attention, the B1/B3/B4 gather, the serial row gather, the verify overlay) each take a kv8 or kv6
  header by `Latent.bits()`, every arm tested against MLX's dequantizer at 6 bits. The prefill scratch bill follows the
  stored width (`scratchBytes`); a kv8 latent keeps its bytes (a hash of its codes, scales and biases is pinned).
- Packed reads are kernel-or-DENSE: MiMo's global layers read packed at t_q==1 (`packedDecodeServesFrom`, from a
  4096-key floor) and Flash-Next's QSA arms read packed (next bullet); every other path reads the dense view.
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

- The decay gate `g` is f32 at every producer (`ModelConfig.gdnGateDtype`, the `GDN gate compiled ... g f32` line): a
  bf16 gate gave a head whose exact g is within 2^-9 of 1 g == 1.0, so it never forgot (206 of 1728 heads in our
  Flash-Next packs). The gate is never stored. The recurrent state is stored f32 between tokens
  by default (`ModelConfig.gdnStateDtype`, one `[gdn] recurrent state f32` line): a bf16 state rounds a per-token decay
  under 2^-9 away, and a slow head forgets at its exact rate in f32. `--qwen-gdn 16` stores it bf16 (the gate stays f32)
  to save memory: a Qwen checkpoint is 115.5 MB at f32 and 58.8 MB at bf16, billed by `ssmCheckpointBytes`, and a
  restored state of the other width is cast before its next step. The 32 GB, 48 GB and 64 GB recipes run it. GLM's state is
  KDA, not GDN: f32 by construction, so the flag does not apply and the bill follows the arch, not the flag.
- Measured against the f32-gate bf16 state (arm b): the f32 state's KLD gain is inside the noise
  ([qwen4-kld](qwen4-kld.md#gdn-f32-teacher)), its cost is about 2% of 4-stream aggregate and +54 MB per checkpoint
  ([qwen4-perf](qwen4-perf.md#gdn-f32-cost)).

<a id="kv6-measured"></a>
## kv6 measured (binary `84d3ed8f`, M5 Max 128 GB, `taskpolicy -a`, GPU lock, fans max, AC power)

- KLD to first EOS (`sushi kld compare`, same binary, kv width the only change):

| model, fixture | kv8 | kv6 | kv4 |
|---|---|---|---|
| Sushi-2.6bpw, `qwen38-flash-next-bf16-gdnf32-v1` raw 16x512, 7539 positions | 0.114252 / 90.79% | 0.115611 / 91.07% (+1.19%) | 0.123894 / 90.53% (+8.44%) |
| MiMo Sushi-2.3bpw, MOPD 16x512, 8067 positions | 0.086035 / 91.95% | 0.088832 / 91.62% (+3.25%) | 0.113654 / 90.42% (+32.1%) |
| GLM Sushi-2.4bpw, 4x512 native teacher (BF16 latent 0.073302 / 89.94%) | 0.074699 / 89.89% | 0.073922 / 89.79% | |

  The Qwen kv6 delta sits at the ~1% rounding-flip floor; GLM's kv8 and kv6 deltas against BF16 (+1.9%, +0.8%) are
  inside it. kv4 costs 8-32%.
- MiMo decode forward meter (`SUSHI_DECODE_FWD_UBENCH=40`, `_S=1,2,3,4`, `_KV=4096`, kv8 / kv6 / kv6 / kv8 boots): 20.33 /
  26.59 / 31.99 / 38.48 ms at 1-4 rows for kv8 against 20.30 / 26.38 / 31.83 / 39.15 for kv6.
- kv4 and kv8 kernels unchanged by the shared unpack helper (interleaved main / branch / branch / main, main `d8b967a5`):
  the MiMo matmul2d microbench (`SUSHI_KVQ_MPP_UBENCH=1`, `_KV4=1`) reads 0.97-1.03 x main at every context from 2k to 512k
  (one cold first main run at 512k aside), and a Qwen QSA decode boot (`SUSHI_DECODE_FWD_UBENCH=40`, `_S=1,3,5,7`,
  `_KV=16384`) reads 17.20 / 24.37 / 31.37 / 38.73 ms (main) against 17.58 / 24.75 / 31.42 / 38.89 (branch) at kv8 and
  17.15 / 24.31 / 31.41 / 38.83 against 17.15 / 24.15 / 31.05 / 38.65 at kv4 (one kv4 branch boot read 1.5-1.9x slower while
  another job held the box, and its rerun is the pair quoted).
- llmprobe (`--bench-only --full`, rungs 512 to 64k, median of 3, `--mtp` as `tests/bench.sh` forces it, Sushi-2.6bpw and MiMo
  Sushi-2.3bpw, boots kv8 / kv6 / kv6 / kv8 / kv4, QUIET, 25 s settle): Qwen decode 100.2 / 95.9 / 94.8 tok/s and prefill
  1970 / 2076 / 2034 for kv8 / kv6 / kv4 (means over the arms); MiMo decode 59.5 / 67.5 / 72.7 and prefill 1297 / 1273 / 1306.
  Speculative decode swings 50-68 tok/s inside one MiMo rung and prefill moves up to 10% with arm order, so these cells
  name no winner.
- The same ladder serial (`--no-mtp`, rungs 2k / 8k / 32k / 64k, kv8 kv6 kv4 kv6 kv8), decode tok/s at 2k / 8k / 32k / 64k: Qwen kv8
  63.7-64.2 / 63.1-63.9 / 63.0-63.5 / 62.0-62.4, kv6 62.2-63.1 / 62.2-63.2 / 63.0-63.3 / 61.9-62.3, kv4 (one arm) 59.9 / 60.6 /
  62.8 / 62.1; MiMo kv8 53.1 / 52.6 / 51.0 / 47.3 (the session's first boot read 47.1 / 50.0 / 48.6 / 46.2), kv6 52.6-53.6 /
  53.0-53.2 / 51.7-51.8 / 48.0, kv4 51.9 / 52.7 / 51.7 / 48.5. kv6 decodes within about 2% of kv8 at every rung.
- Engagement at kv6 (server logs): `[qsa-attn] engaged (S=1 kv=2052 quant=6/gs64 ...)`, `S=31`/`S=7` verify widths at `quant=6/gs64`,
  `[qsa-gather] engaged: sushi_qsa_nax_precise` (the dense-view gather; the packed gather did not engage on the llmprobe
  prompts at kv8 either). MiMo: `[kv-attn] matmul2d packed attention engaged: bits=6 gs=64 DK=192 DV=128 Hq=64 Hkv=4 Tq=1 Tk=4105` and
  `[mimo-verify] global rows in matmul2d row groups engaged: rows=3 keys=4108`, as at kv8; the split-K arm stays off on a NAX GPU.
- `tests/test_glm_prefix_reuse.sh` passes at kv8 and at kv6 (every restore, the cold-vs-warm logprob bounds, and the BF16 request
  that never restores the kv6 entry); `tests/test_kv_quant_per_request.sh` and `tests/test_kv_quant_equivalence.sh` pass on
  Sushi-2.6bpw with kv6 (first 30 tokens byte-identical to dense at 4, 6 and 8 bits).
- GLM kv6 smoke on Sushi-2.4bpw: loads, answers, and DFlash2 engages (`[spec-stats] mode=dflash ... avg_per_round=2.38`);
  `[glm] native kv6 MLA: 4576 latent + 704 pooled-index bytes/token` (11 MLA layers at 416 bytes).

## Byte stability

- INT4 long-greedy divergence is legit (the AR/verify INT4 kernel float-noise tail).
- Byte-stable greedy ⇒ no spec + `--kv-quant off/8` + `--prefix-cache-entries 0`.
- A prompt longer than one chunk is byte-stable only at a fixed prefill width: MiMo at 64k gives a first-token
  logprob of -1.0685 at 4096 and -1.0807 at 2048 (each width repeats exactly), and the per-request ladder narrows
  the width when memory is short. `--prefill-chunk N` caps it.
- A restore is not bit-identical on a hybrid ([engine-prefix-cache](engine-prefix-cache.md)).
