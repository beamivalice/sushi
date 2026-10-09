# GLM-5.3-Flash (`glm5_next`)

The native GLM path: checkpoint geometry, what `sushi serve`/`run` load and bill, DFlash2 speculation, recorded speed
and quality, and the lessons the bring-up paid for. Per-kernel contracts and the alternatives that lost are in
[glm5-kernels](glm5-kernels.md); the clamped EXL3 expert chain is in
[engine-exl3-experts](glm5-clamped-experts.md#glm).

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-memory-admission](engine-memory-admission.md),
[quality-kld](quality-kld.md), [engine-expert-streaming](engine-expert-streaming.md).

## Scope

- Concurrent requests batch as plain rows, up to four ([concurrency](#concurrency)). MTP is off: the checkpoint's MTP
  layer is not integrated.
- Prefix reuse, RAM, SSD and SSD-only: KDA checkpoints on the prefill chunk grid and at the prompt end, MLA rows below them, the
  assistant window beside them ([engine-prefix-cache](glm5-prefix-cache.md#glm)). The RAM tier defaults to 1 GiB,
  which admission evicts for a long prefill; long reuse belongs on `--prefix-cache-disk` or SSD-only.
- Cache: kv8 compressed MLA latent plus FP32 KDA state, the engine default; it passed its KLD gate
  ([quality-kld](quality-kld.md)). `--kv-quant 16` (or `kv_quant: 16` per request or in model-settings) keeps the
  latent BF16. The pooled index and KDA state stay lossless. kv4 is refused (`GlmKvQuantUnsupported`), and the KLD teacher capture refuses any kv-quant.
- Thinking: `low`, `high`, `max` (the template's `effective_reasoning_effort`); Sushi defaults to `high` (the HF
  template defaults to `max`); thinking off is refused. The effort words impose no token cap.
- Image and video input through the native tower, on when present (a streamed load only with `--vision`);
  `--no-vision` drops its weights and buffers.
- DFlash2 speculation when an assistant is found ([below](#dflash2)), for greedy and sampled requests.
- M1–M4 GPUs (no NAX) run the same model with the arms in
  [glm5-kernels](glm5-kernels.md#without-nax); `SUSHI_FORCE_GPU_FAMILY_FALLBACK=1` rehearses them.

## Checkpoint geometry

| Part | Shape |
|---|---|
| Trunk | 45 layers, hidden 4096, vocab 154880, `max_position_embeddings` 1,048,576 |
| Attention mix | 34 KDA layers; 11 MLA layers at 3, 7, …, 43 (every fourth) |
| FFN | layers 0–2 dense (width 12288); 3–44 MoE: 288 routed experts, top-8, width 2048, one shared expert |
| Routing | sigmoid scores; the FP32 correction bias picks the top-8 only; unbiased scores normalized, times 2.5 |
| Expert activation | gate upper clamp and symmetric up clamp at 10 before SwiGLU |
| mHC | 4 residual streams; per layer two collapses (24 FP32 mix values from `[4×4096]`, Sinkhorn 20 iterations) and two expansions: 90 collapses per token |
| KDA | 64 heads × 128, short conv 4, per-key-channel FP32 decay `exp(−5·sigmoid(exp(A_log)·(a + dt_bias)))`, beta, FP32 state, gated output norm |
| MLA | NoPE; q LoRA 1536, one 512-wide compressed latent per token, 64 heads with 256-wide q/v; queries absorbed into latent space, scale 1/16 (from the 256-wide query, not the 512 latent) |
| IndexPool | 32 index heads × 128; keys pooled every 4 tokens (positional bias, softmax over the pool); each query keeps at most 512 completed causal pools (2048 tokens) plus the 0–3 token tail |
| Extras | MTP layer 45 (54 tensors under `layers.45.*`, unused); vision tower (24 layers, width 1024, patch 14, spatial merge 2, temporal patch 2) |

Selection is dense through position 2050: the first selective row is 2051, when the 513th pool completes. The
checkpoint stores the HC mixing matrices, router `[288,4096]` and the small KDA projections (FA/GA `[128,4096]`, FB/GB
`[8192,128]`, beta `[64,4096]`) in BF16 and the HC scale/base, decay parameters and correction bias in FP32;
`moe_router_dtype=float32` names compute precision, not storage.

## Packs and loading

The shipped pack, GLM-5.3-Flash-Sushi-2.4bpw (97.8 GB), carries MCG EXL3 routed experts at window 14 (K2.25 in MoE
layers 3–36 and the MTP layer, K2.5 in layers 37–44) and an affine 6-bit group-128 trunk. Earlier packs, none shipped:
Sushi-2.3bpw (K2.25 everywhere, A6 g128, window 12), Sushi-2.5bpw (K2.5 everywhere, A6 g128, window 12), an A8 g128
trunk experiment (earlier "Sushi-2.4bpw") and a raw FP8 E4M3FN block-128 trunk (Sushi-2.45bpw). Small BF16/FP32 tensors
keep their source precision. The consumer contract is [pack-format](pack-format.md); how packs are made lives in the
private converter repo.

- `model.loadWeightsForConfig` reads only indexed text tensors, plus the tower when vision is on. It uploads one shard
  at a time (a 566-shard pack once exhausted the 256-descriptor limit) and preserves every stored dtype.
- Affine 6/8-bit is inferred from the packed row width and the declared input width, never from a directory name.
- Raw FP8 projections run MiMo's `fp8_block` kernels: direct FP32-accumulating GEMV up to 16 rows, a temporary BF16
  expansion beyond it (billed for the widest pending layer: 576 MiB at two pending layers).
- The BF16 source checkpoint is the KLD teacher, run with SSD-streamed experts
  ([engine-expert-streaming](engine-expert-streaming.md)).
- `--ssd-budget-gb`/`--expert-cache-gb` stream any pack's EXL3 experts, or the BF16 source's, through the same engine
  for `serve`, `run` and `kld compare`: DFlash2 off, image and video only with `--vision`, output identical to the
  resident load.
- The FP8 release (E4M3FN block-128 trunk and experts) streams its experts as stored and runs its trunk on `fp8_block`
  like Sushi-2.45bpw; it is never a teacher. Hermetic proof only: the checkpoint is no longer on the box.

## Serving loop

- Prefill runs 2048-token chunks with two layers in flight, stepping a chunk down to 1024 or 512 only where it no
  longer fits beside the KV (at release defaults 2048 fits past 1M); chunks ending at or before token 2051 use dense
  expanded-K/V attention, later chunks the absorbed sparse path. Decode submits every four layers and evaluates logits
  and every cache array once per token.
- Vision: padded CLIP preprocessing, temporal placement of video frames, visual embeddings spliced into the HC input.
  The tower weights join the load bill and the encoder scratch is checked before each image/video encode.
- Sampling, stop and output budgets run in the shared generator; a request that cannot speculate decodes serially.

<a id="dflash2"></a>
## DFlash2

The assistant is a separate 5-layer draft model, not the checkpoint's MTP layer: hidden 4096, block 8, mask token
154856, noncausal block attention, sliding window 2048, two-tap dynamic convolutions, selector rank 256 with top-16
lattice edges. It has no embedding or head and uses the target's. Its input is the mean of the four HC streams after
target layers 5, 14, 24, 33 and 42, before the final norm.

- **Discovery**: `--drafter <dir>` wins, `--no-drafter` disables; otherwise a valid `dflash2/` inside the pack, then
  legacy `drafter/`, then `GLM-5.3-Flash-DFlash2/` (the user's own download; the pack does not ship it). One resolved path feeds
  both the bill and the loader.
- **First-load cache**: when only the released BF16 `GLM-5.3-Flash-DFlash2/` exists, `serve` and `run` quantize its
  matrices once to A4 group-64 with MLX's affine quantizer into `dflash2/` (selector codebooks, selector hidden
  projection and non-matrix tensors stay BF16), under a per-pack lock, staged and synced before publication, and
  invalidated by source/config identity, but an intact cache (size and mtime match its manifest) stays valid once
  the source folder is deleted. 2.18 GiB → 0.721 GiB. No space or no write permission
  falls back to the BF16 assistant and reruns preflight with its full size. The cache is local only: the assistant's
  CC BY-NC-ND 4.0 license is unchanged and the cache is no redistribution artifact.
- **Stored formats**: BF16, or one uniform affine format per assistant: A4 g64, A6 g128 or A8 g128 (anything else is
  `UnsupportedGlmDraftStorage`). A4 g64 is the first-load default (774.5 MB vs A6's 1013.1 MB; drafts ~11% faster, decode within drift, target
  output exact); an A6 cache from the earlier policy is regenerated.
- **Tree**: two draft nodes plus the root, up to four children per node; the verifier runs all rows layerwise.
  KDA replays only the accepted path from a prework tape; IndexPool builds branch-local pools from the committed prefix
  plus each node's ancestry (pooling flattened tree rows would pool siblings together), held beside the reserved
  pooled buffer and never written into it ([kernels](glm5-kernels.md#dflash2-verification)); MLA reads the
  committed prefix plus the ancestry tail. Commit publishes target state and assistant context together; a commit that fails after
  taking over the request's MLA buffers leaves the request failed.
- **Draft execution**: the fixed two-node tree keeps all eight noise rows as attention keys and values, but the
  final layer computes only the anchor and two required output rows. Two-tap BF16 convolutions preserve the original
  multiply/add rounding in one kernel, and sliding layers share a block mask. On M5 Max, the A4 g64 assistant's
  eight-row FFN projections reuse weights across the full block with MLX's original reduction order.
  [Paired measurements](glm5-perf.md#glm-draft-ten-percent) show about 12% lower draft time; depth and acceptance
  are unchanged.
- **Decisions**: greedy follows the target argmax; sampled requests draw only the visited target path with the
  request's sampling parameters, advancing the RNG exactly as serial decoding does (budgets and EOS included). Both
  verify through the same batched rows, whose logits equal per-row serial projections bit for bit.
  Constrained, forced-tool-call, penalized, logprobs or explicitly budgeted-thinking requests decode serially.
- **A round's tree is cut to `min(output budget, positions left)` rows deep** (`proposeRound`): the verifier refuses an
  ancestry past the context, so a two-token tail never carries a three-row chain.
- **Bills**: assistant weights at load; per request the sliding window ×4, captures per prefill row, three recurrent
  checkpoints, the 256 MiB verification-scratch cap and 64 MiB. The MLA reservation is input + max_tokens + 3 rows:
  a request without max_tokens reserves its whole context window (946K rows: 11.3 GB BF16, 6.3 GB kv8).
  The reserve gate's budget sizes the cache from the request's own clamped rows, never from the generator's config
  copy (unpinned, it resolves a smaller auto context and under-budgets the reservation).
- **No yield gate**: `[spec-stats] gate_min` is the generic DFlash bar (1.80 here) and is never evaluated on the
  native path. N2 beats a serial step above ~1.1 accepted drafts per round at 1K–30K; measured requests ran 1.31–1.90.

Wider trees lost: N3 with every four-row kernel optimized measured 40.44 vs N2 40.22 tok/s at 8192 IDs (`e1597cc2`,
inside 1.46% drift) because verification per round grew 20.6%.

- **N2 saturates on copies, breaks even on prose** (no runtime yield gate): a ~2K-token verbatim copy and a rename
  edit accepted 2.00 of 2 drafts every round (41.4/40.8 vs 25.2 tok/s serial), low-effort prose 0.88 (26.1 vs 25.6);
  greedy bytes equal serial (`144f63db`, BF16 latent, Sushi-2.3bpw + A4 g64, `taskpolicy -a`, busy box, 2026-10-04).
- **Verbatim lookup chains instead of PLD**: PLD never runs on GLM (`specInitWiring`'s module branch), so `--no-drafter`
  decodes plain serial and the 0.010 n-gram gate is inert. A request drafting alone whose output agrees with its context
  for `mtp_lookup.STRONG_SUFFIX` tokens verifies the context's next three tokens as a four-row chain in place of the
  assistant's tree (`glmLookupProposal`, `[spec-stats] … lookup=rounds/landed`). `--no-mtp-lookup` turns the chains off.

<a id="concurrency"></a>
## Concurrency

- Each slot owns its target state ([server-lifecycle](server-lifecycle.md#scheduler-and-batching)); a prefill yields to
  the others' decode ticks. A streamed GLM load (the teacher) still queues.
- **Concurrent requests decode as rows of one forward**: `verifyGroups` takes one group per request (rows in order);
  projections, router, experts, HC and head read each weight once, the KDA recurrence and MLA attention run per
  request on its own state, and every group's logits, targets, captures and commit equal its solo `verify` bit for bit.
- **Speculate alone, plain rows in company** (`glmRowsInCompany`, up to `batchGroupCap` 4 slots): a DFlash2 request
  with company commits its row with the assistant's taps (`Generator.glmRowCommit`), so its rounds resume alone.
- **Rows are the currency** (rows ubench `02d2ee4d`, Sushi-2.3bpw kv8, 1K/6K context): a grouped forward costs ~25 ms
  plus ~14.5 ms per row; two requests 54 ms (1.39× serial), four 82 ms (1.84×); one row is within 2% of serial decode.
  A draft row pays only while its ms per accepted token (~19 copy, ~35 prose) beats the batch's (27 at two, 21 at four).
- **Four-row planner** (`glmTreeNodes`, `GlmRowCost` from those costs): a grouped tick carries at most four rows; the
  drafter with the best landing rate (`Generator.glm_draft_rate`, accepted share of a tree's depth) takes the spare
  rows as its DFlash2 tree (two requests: N2, three: one draft, four: none) while its expected tokens pay for them.

## Memory

- Load bill: text weights, the enabled tower, the selected assistant and warmup. Request bill: BF16 latent 11,264
  plus pooled-index 704 bytes per token (11,968); under kv8 the latent is 5,984 (11 × 512 codes + 8 BF16 scale/bias
  pairs, 6,688 per token). Then capacity growth (256-row rounding, at the stored row width), the raw key/gate ring
  and FP32 KDA state (147,619,840 bytes), plus native kernel transients at two pending layers: A6 expansion 512 MiB,
  B1/B3 decode attention 128 MiB, KDA cluster 1.25 MiB per pending layer. The MLA-only terms are held by the one MLA
  layer a two-layer pending window can contain (`glmMlaLayersPending`): head-batched MLA copies 384 MiB, packed
  attention with its second tile 256 MiB, index scores 8 MiB, and under kv8 the dense-prefill dequantization
  (≤ 2051 rows) plus one chunk's quantizer output, 3.1 MiB. Without NAX the packed tiles (the FP32 composite) keep
  their 256 MiB and the A6, MLA, index and cluster terms drop. With the prefix cache on, a
  prefill also holds up to 9 KDA checkpoints (141 MiB each) and one assistant window, fewer where they do not fit
  ([prefix cache](glm5-prefix-cache.md#glm)).
- Advertised context: billed at the widest rung up to 2048 that advertises as much as 512 (`glmPrefillChunk`), with
  the engine's 93% margin: GLM's admission bills each request exactly and refuses past it. Sushi-2.5bpw +
  vision + A4 at its measured 104.35 GB active advertises 1,048,576 at a 2048 bill (1,144,691 tokens by the bill).
  The prefix cache reserves nothing here: admission evicts its RAM tier and bills a request's checkpoints, keeping
  fewer where they do not fit. The quarter-share rule had pinned the prefill to 512 rows; at 85% the auto context
  read 972,800.
- `max_safe_context` = (ceiling − active − transients) × 0.8 × 0.8 / per-token bill. The kv8 default drops the bill
  44%: Sushi-2.5bpw + vision + A4 assistant boots at 104.32 GB active with `max_safe_context` 1,048,576 (the position
  cap; about 1.36M by the bill), against 758,793 at `--kv-quant 16` (976a0dbb, auto context, margin 4 GiB).
- Measured: Sushi-2.3bpw plus the A6 assistant settles at 94.55 GB active (88.06 GiB) under a 115.45 GB limit on the
  128 GB box; a 16K prefill peaks at 96.43 GB without an assistant. Sushi-2.45bpw (raw FP8) is 101.75 GB resident,
  102.51 GB peak while scoring KLD. Sushi-2.5bpw with the A6 assistant and vision is 104.56 GB active, leaving
  `max_safe_context` 746,036 tokens (A4 assistant: 104.32 GB, 758,793; BF16 cache with no assistant, no vision and
  `--wired-margin-gib 2`: 955,781) under `iogpu.wired_limit_mb=120000` (margin 4 GiB): 1M context fits only at kv8.
- Shipped Sushi-2.4bpw (91.1 GiB on disk) with the A4 assistant and vision: 96.62 GB active, `max_safe_context`
  1,048,576 under `iogpu.wired_limit_mb=120000` (`e26fd471`). The figures above are other packs'.
- An explicit `--ctx-size` is not checked against that bill at load (GLM is outside the load-time serving bill), so
  `n_ctx` can advertise more than a request may use; request admission refuses past the affordable context.

## Recorded performance

llmprobe 0.6.13 `--bench-only --runs 1`, reasoning default, one request per cell, server timers. Runtime `4fcb541e`
(2026-10-04): Sushi-2.3bpw + A6 g128 assistant, N2/children 4, prefill chunk 2048, BF16 MLA, FP32 KDA, greedy,
prefix reuse off; decode = (outputs − 1) / decode time, 192 outputs (ordinary 16K stopped at 177 on EOS). The cells
ran through the loopback bench bridge that `sushi serve` has since replaced; the `sushi serve` quiet-box ladder
(`b8267038`) is in [perf-baselines](perf-baselines.md).

| Context | Ordinary prefill | Ordinary decode | Predictable prefill | Predictable decode |
|---|---:|---:|---:|---:|
| 2K | 869.79 | 46.18 | 918.33 | 50.61 |
| 4K | 844.65 | 39.39 | 844.02 | 49.33 |
| 8K | 769.25 | 46.05 | 779.09 | 49.64 |
| 16K | 730.25 | 42.47 | 721.95 | 48.08 |
| 32K | 660.77 | 42.99 | 663.04 | 47.56 |

Input IDs: ordinary 2072/4095/8261/16314/32783, predictable 2036/4059/8225/16278/32747. Without an assistant, serial
decode measured 31.4 tok/s at 512 context (2.3bpw, `ba106e5e`). The 1500 tok/s prefill / 60 tok/s decode goals are
open; verification dominates a speculative round (about 60 of 70 ms at 16K–32K).

## Quality

A pack is scored with `sushi kld compare --model <pack> --fixture <teacher>` against the native BF16 teacher
(`sushi kld capture --prompts standard4`, streamed BF16 experts). Shipped Sushi-2.4bpw, four prompts × 512 against the
NAX-path teacher: KLD 0.0742, top-1 90.3%, code 0.0429 / prose 0.1055. Earlier packs against the
first teacher, four prompts × 512 (2026-10-04, TF32 off): Sushi-2.3bpw 0.0930, the A8 experiment 0.0915, Sushi-2.45bpw
0.0913 mean KLD (code ~0.045, prose ~0.139); Sushi-2.5bpw (K2.5 experts, A6 trunk) 0.0721. Not yet the 16x512 release
reading; tables and settings in [quality-kld](glm5-kld.md#glm-53-flash-native-bf16-teacher-4x512-2026-10-04).
The M1–M4 path rehearsed on the M5 scores the first prompt at 0.0457 against the stock path's 0.0446, top-1 equal
([perf-baselines](glm5-perf.md#glm-nonnax)).

The native teacher also captures block boundaries (`SUSHI_HIDDEN_OUT`, all four HC streams, 16,384 BF16 values per
token per boundary, 46 boundaries) and, for many short windows, runs layer-major: a batch of windows reads each
layer's experts once, with byte-identical output ([quality-kld](quality-kld.md#layer-major)).

## Lessons

- A fast path gates on what its kernel needs, never on the served pack's value: W12-only expert gates sent the W14
  Sushi-2.4bpw through the generic chain at +30% verify per round with no error. Diff engagement lines when a pack
  changes ([measurement](glm5-perf.md#glm-w14-lanes)).
- mHC expansion contracts the residual streams first, then adds the separately rounded FP32 branch product; the
  reverse order changes BF16 results.
- SiLU rounds its sigmoid to BF16 before the multiply; an FP32 HC mix through generic matmul may pick TF32, so the
  mix uses an explicit FP32 dot product.
- Storage dtype is read from the headers, never from config compute precision (router and HC matrices are BF16).
- A contiguous batch-one slice still aliases its prompt buffer: the KDA convolution tail takes a materialized copy.
- Async scheduling may hold two cache generations at once; it is not a bill reduction.
- Absorbed and expanded MLA round at different BF16 boundaries; chunk width changes GEMM row shapes, and the chunk
  schedule moves IndexPool scoring between scalar and NAX. Prefill chunking is a numerics decision.
- The first generated token comes from prefill (64 outputs = 63 decode forwards); speculative harnesses that count
  committed input tokens are not comparable with server delivery rates.
- Count tokens through the tokenizer: the "2K" predictable prompt is 2036–2037 IDs, so exact-2048 guards never
  engage on it; 2048-token content repeated 4× is 8156 tokens, not 8192.
- Every fused path needs an engagement counter: a router fusion keyed on FP32 storage made zero calls behind a
  correct fallback.
- MLX masked SDPA does not protect against NaN/Inf in masked K/V rows (0·NaN poisoned every earlier query): zero
  invalid rows before load.
- Retaining a full KDA state per tree node would cost ~2.1 GiB for 16 nodes; replay the accepted path instead.
- A synchronizing per-stage profile changes overlap and allocation; it ranks stages, never times them.
- The pinned Metal backend cannot load safetensors on the GPU stream: load on the CPU stream, compute on the GPU.
