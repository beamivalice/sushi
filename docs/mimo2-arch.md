# Architecture: MiMo-V2.6-Flash (`mimo_v2`)

How the engine serves MiMo-V2.6-Flash (the served pack is quantized from the MOPD release; the RL release shares its
layout): the source checkpoint's layout, the resident trunk, the MXFP4 and EXL3
expert paths, the hybrid global/sliding attention with its ring, the vision tower, and the bills that follow the
storage. MiMo serves text and image input; the supported product is the MCG EXL3 pack. Read this before touching
`src/mimo_source.zig`, `src/mimo_vision.zig`, `src/mimo_forward.zig` (forward, attention arms, MoE dispatch), the shared `src/transformer.zig`, or anything that bills MiMo's KV.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-exl3-experts](engine-exl3-experts.md),
[engine-expert-streaming](engine-expert-streaming.md), [engine-kv-cache](engine-kv-cache.md),
[engine-memory-admission](engine-memory-admission.md), [quality-kld](quality-kld.md),
[perf-baselines](perf-baselines.md).

## Product policy

- **MCG EXL3 only.** The served MiMo target is the MCG EXL3 pack `MiMo-V2.6-Flash-Sushi-2.3bpw` (K2.25 experts,
  the last layer K4).
- **Thinking defaults ON** (the vendor template's default; `generation_config.json` declares none); effort words
  only set the thinking budget (see [server-http-apis](server-http-apis.md)).

## The checkpoint

- 302.8 B routed-expert weights (256 experts, top-8, 47 MoE layers, hidden 4096, expert intermediate 2048) stored
  MXFP4; ~8 B dense (FP8 e4m3 with 128x128 f32 block scales, plus bf16); an MTP head. No bf16 release exists.
- 48 layers: layer 0 is a dense MLP (`first_moe_layer` = 1); 9 global layers (4 KV heads, key 192, value 128) and
  39 sliding layers at a 128-token window (8 KV heads).
- The FP8 part is `qkv_proj` + the layer-0 MLP (3.07 GB e4m3); the bf16 3.32 GB is mostly `o_proj` (in
  `ignored_layers`). lm_head and embed_tokens are bf16, 1.25 GB each, untied.
- **Loaded by the engine: text + the vision tower.** The checkpoint also ships the vision tower (MiMo-ViT, 364
  tensors, 1.457 GB bf16; see [Vision](#vision)), an audio encoder + 20 speech-embedding tables + a separate 24 kHz
  audio tokenizer (input only: the LLM has no speech-output head), 3 MTP layers (`model.mtp.*`, 48 tensors; ~396 MB
  each, 1.19 GB) and a 5-layer DFlash drafter (block 8, target layers 0/11/23/35/47). The trunk loader skips
  `visual.`, `audio_encoder.`, `speech_embeddings.` and `model.mtp.`; the tower loads beside it
  (`mimo_source.loadVisionWeightsInto`) unless `--no-vision`; audio and video are not wired (their pad ids are
  zeroed so they never join the splice). The three MTP heads load separately under `--mtp`
  (`mimo_source.loadMtpWeights`, `mimo_mtp.zig`; [engine-mtp](mimo2-mtp.md#mimo)); the DFlash drafter is not loaded.

## Source checkpoint and packs

- **Original checkpoint**: `.mxfp4_individual` streams per-expert U8 payloads directly into U32 slabs without
  changing bytes. The FP8 trunk (`qkv_proj`, layer-0 MLP) stays resident AS STORED: e4m3 codes + f32 128x128
  tile scales, served by `fp8_block.zig` (f32 decode GEMV for 1-8 rows, staged x for 9-16, one linear dequantized to
  billed bf16 scratch + MLX matmul for wider forwards). That is the checkpoint's exact math, so the KLD teacher
  carries no quantization of its own ([quality-kld](quality-kld.md#teacher-path)); sources are read-only, MTP/media
  excluded, residency billed as stored plus `server.fp8DequantScratchBytes` at prefill. Those kernels are plain
  SIMD Metal (no NAX/matmul2d), so M4 runs the same code. On NAX a serving forward of 9-128 rows takes the
  matrix-unit tile instead ([engine-kernels](engine-kernels.md#prefill-kernels)); `kld capture` never does.
- **Converted MXFP4 pack**: the private MiMo pack converter optionally restacks the same
  MXFP4 bytes into `model.layers.N.mlp.switch_mlp` U32 weights + U8 e8m0/32 scales, without biases, and prepares
  the trunk ahead of time. Both source layouts use the same streaming kernels.
- **Packed QKV is rank-local**: each rank's FP8 rows and their scale blocks are written straight into global
  Q/K/V (`[Q_rank | K_rank | V_rank]` regrouped without dequantizing). Extra scale rows belong to partial
  rank-local tiles, not trailing padding on the full tensor. Tensor-parallel 4 is solved from geometry.
- **A MiMo EXL3 pack serves RESIDENT**: see [engine-exl3-experts](engine-exl3-experts.md#mimo). The trunk takes
  the source loader (`usesMimoSourceTrunk`), billed as stored by `mimoSourceResidentBytes`.
- **A load holds ONE 2-bit coarse copy of the lm_head** (`Transformer.lm_head_coarse`, ~0.19 GB, billed with or
  without MTP, `scheduler.mimoCoarseHeadBytes`): the heads draft on it and the trunk's greedy readout shortlists on it,
  so a greedy argmax is the same with MTP or without ([engine-mtp](engine-mtp.md#greedy-shortlist)). A streamed load
  keeps it too, billed as trunk in the SSD budget ledger.
- **The weight loader is ONE decision** (`model.loadWeightsForConfig`): a MiMo pack read without its source trunk
  binds the raw FP8 fused QKV and its logits stop following the routed experts (two packs sharing a hard-linked
  trunk produced bit-identical logits until `kld` took the served loader).
- **Stored-affine trunk**: a SERVED pack stores o_proj, lm_head and embed_tokens as affine triples (8-bit g64,
  written by the private converter); the loader serves and bills them as stored (lm_head via quantized matmul, embed
  via the quantized row gather), with no load-time step. The source checkpoint stores them bf16, so the teacher keeps
  bf16. Contract: [pack-format](pack-format.md); how the triples are packed lives in the private converter repo.
- The FP8 linears stay FP8: affine-8 for them measured no faster and is lossy.

## Geometry and math

- `hybrid_layer_pattern` 0 = global, 1 = sliding; read heads, KV heads and K/V widths per layer. Rotate only the
  first `int(head_dim * partial_rotary_factor)` channels (64 dims); multiply V by `attention_value_scale` BEFORE
  caching. Attention scale is on the 192-wide key; two rope bases.
- **Routing/sinks**: sigmoid routing uses f32 inputs/weights, selection-only correction bias and unbiased
  normalized scores (1e-20 normalization). A sink is an extra softmax denominator column, not a real key; its
  presence follows the layer type (sliding layers only).
- mlx-lm's MiMo support (upstream PR 1219) agrees with this engine on every mechanism except that it computes the
  router matmul in bf16 (changes the top-8 set for 2.8% of tokens per layer; patch it to f32 before using it as a
  cross-check). `attention_chunk_size` is read by nothing, in the engine or the reference: `modeling_mimo_v2.py`
  masks a sliding layer with transformers' `create_sliding_window_causal_mask`, a query and the 127 keys before it,
  which the `mimo v2 fixture` parity test pins across the window edge.

## Expert streaming and imatrix

- `first_moe_layer` preserves absolute layer indices while excluding dense prefix layers from expert slabs and cache
  budgets. MXFP4 has six operands in nine stable component slots; absent biases acquire no slab or lease. MTP
  remains refused while streaming; `--expert-pick-tolerance` compares the router's sigmoid probabilities.
  Streaming engine: [engine-expert-streaming](engine-expert-streaming.md).
- **Imatrix** keys by ARCH (`imatrix.Arch.mimo_v2` → `model.layers.{L}.mlp.experts.*`, one flat entry per layer) and
  reaches the streamed QUANTIZED layer through the routing override's tap; armed, it forces the SORTED expert arm —
  the fused decode kernels never materialize the activation rows the down statistic needs. The driver lives in the
  private converter repo.
- The same capture records the trunk's dense inputs (`Collector.observeLinear`): every layer's o_proj input
  (`model.layers.{L}.self_attn.o_proj.weight`, [heads x v_head_dim]) in `mimoAttnWith`, and the final normed hidden
  as `lm_head.weight` [hidden] in `forwardMoeWith` (every row, also where a chunk skips the projection); each as
  sum(x²)/rows beside `<name>.rows`. embed_tokens has no input activation and gets no entry.

## Sliding layers: the ring

- **Sliding layers RING** (`ModelConfig.swaRingTokens`, `KVCache.setSwaRing`): they store
  `sliding_window + SWA_RING_SLACK` rows (128 + 512; a compaction copies the retained window when the slack fills),
  never the context.
  A non-zero `max_seq` into `KVCache.update` IS the ring predicate, so `slidingViewFor` may never decline the trim
  on a ringed arch.
- **A compaction lets go of the pre-compaction buffer when the forward that read it is evaluated**
  (`KVCache.handOffRingViews`, `ringCompact`): the forward takes the entry's views, the entry re-points at its ring,
  and the ring is evaluated at once. Left lazy, the ring and the views pinned every sliding layer's staged chunk until
  the next forward, 39 layers where `swaStreamBytesPerToken` bills the eval cadence's 5.
- **A ringed entry's `offset` is LOCAL**; absolute = `base + offset` (`absSeqLen`). A clamp below the retained
  window declines by NAME (`SlidingRingRewindPastWindow`) and the hot-cache restore cold-prefills; a byte-budget trim
  lands only at the entry's end or a ring checkpoint ([engine-prefix-cache](engine-prefix-cache.md#candidate-ranking-and-trimming)).
- **The SSD tier persists a ringed entry as chunks of the global layers plus one ring file per restore point**
  (`r{pos}.safetensors`: the prompt end, message marks, inherited forks, the entry's end) and restores only at one of them
  (`restoreIntoRinged`, manifest v9); a ringed slot never takes an entry without them (its sliding layers are billed
  as the ring, so a full prefix there would be unbilled).
- **A hot entry keeps a prompt-end ring checkpoint** (window + 30 rows per sliding layer): a reply past ~256 tokens
  compacts the ring past the prompt end, and a client that sends back content only diverges at prompt + 1 (history
  renders `<think></think>` where the model wrote its thought; one that echoes `reasoning_content` matches the whole
  entry). The content-only reply re-prefills at its new positions; the restore keeps the conversation before it
  ([engine-prefix-cache](engine-prefix-cache.md#basics)). `swaRingCheckpointBytes` bills each of the slot's
  `SLOT_RING_CHECKPOINTS` = 6 copies beside the ring (its restore, up to four message marks, its prompt end; 158
  rows: 30 MiB bf16, 16 MiB kv8, `server.slotRingBytes`); each entry bills its own in `kv_bytes`, up to eight with
  those it inherits from the entry it forked off (`bestRingDonor`). `slotRingBytes` also bills the global-layer
  decode rebuilds the other rows of a batched decode hold below the packed arms' floor (`batchedDecodeRowsBytes`).
- **A hot entry holds a ringed layer's RETAINED ROWS, never the ring's capacity** (`KVCache.snapshotRetained`): the
  buffer is allocated at `ringCap` from token one, so a plain share billed and pinned rows no restore can read.
- Per token: bf16 288 KiB → 22.5 KiB, kv8 153 KiB → 12.0 KiB; ring per slot 122 MiB bf16, 65 MiB kv8.

## Attention kernels

- **Every layer PREFILLS FUSED** (`sushi_attn_pd`, qk 192 / v 128). A sliding layer's learned sink joins the online
  max and sum with no value row (template flag `SINK`; `SINK=0` compiles the global layers' code unchanged, proven
  byte-identical); `slidingPrefillFused` gates the dispatch AND `server.slidingBandScoreBytes`, so the band sheet is
  billed only where it still composes (chunks under 16 rows, `SUSHI_FUSED_256=0`). A quantized cache is read one
  DISPATCH at a time (`fusedSdpaPrefillKv`; `kr` = {begin, end, koff, kL_abs} puts every causal comparison in CACHE
  coordinates), never rebuilt whole; the sliding ring view is window + chunk rows, dequantized per view.
  Landed 2026-09-23 (668278c): 39-layer band attention 39.5 -> 20.3 ms at chunk 512, 603 -> 95.5 at 2048, 2439 -> 181
  at 4096; 16x512 KLD -0.2%, inside the rounding-flip floor.
- **Global-layer attention is what grows a long prompt's TTFT**: ~26% of it at 64k, ~43% at 128k and ~62% at 256k at
  chunk 2048; the band calls are ~1% ([perf-baselines](mimo2-perf.md#mimo-longctx-prefill-attn)).
- **A global-layer forward under 16 rows runs row by row** (`MimoAttnArm.prefill_rows`, the verify rows' arm): the
  fused kernel declines there, and the composed arm would rebuild the whole packed cache dense beside a
  [heads, rows, keys] score sheet, unbilled. A warm restore's short tail (a follow-up of a few tokens) is the case;
  each row is its serial decode tick bit for bit.
- **On M5 both layer kinds prefill on the matrix units** (`sushi_attn_pd_nax`, same carries, bill and slices;
  `SUSHI_ATTN_PD_NAX=0` = the SIMD kernel): global attention ~3x faster per layer
  ([engine-kernels](engine-kernels.md#prefill-kernels), [perf-baselines](mimo2-perf.md#mimo-attn-kernels)).
- **A packed-cache global-layer DECODE reads in place** (`mimoGlobalDecodeArm`): with matrix units (M5) the matmul2d
  `sushi_qkv_mpp` (`qkvMppDecodeServes`, from `QKV_MPP_DECODE_MIN_TK` = 4096 keys); without them (M4) the QSA split-K
  body over the whole causal range (`qkvAttnSplitKKernel`, from `QKV_SPLITK_DECODE_MIN_TK` = 4096 keys; 512 keys per
  split, 64-128 splits, split count a runtime value); else the dense rebuild. `SUSHI_KVQ_FORCE_SPLITK=1` takes
  the split-K arm on an M5 for A/B.
- **The arm is chosen per decode STEP from the cache's current key count**, never from the request's admission-time
  `kv_attn_fused` (`auto` resolves that from the PROMPT, so a short prompt that grew long stayed on the rebuild,
  unbilled, and at 256k drove wired memory to the limit). `--kv-attn-mode` and the per-request field no longer reach
  these layers; `SUSHI_KV_ATTN_FUSED=0` still does, and `kvDequantScratchBytes` then bills the whole-cache rebuild
  (`mimoGlobalDecodeRebuildMaxKeys`). The older SIMD `qkvAttnDecodeKernel` cannot stage gqa 16 x qk 192.
  Attention-only microbench (9 layers, kv8) vs the per-call dequant+SDPA rebuild: split-K 0.62x at 4k, 0.50x at 16k,
  0.42x at 64k, 0.35-0.36x at 512k (M5 proxy for M4; split-K is compute-bound, the rebuild bandwidth-bound);
  matmul2d is ~1.2-1.4x faster than split-K at 16k-512k (cross-run). Live 64k split-K decode not yet measured.
- **`sushi_qkv_mpp` is latency-bound, not bandwidth-bound**: 4 simdgroups with register-prefetched words
  (bit-identical to the 8-simdgroup kernel) cut global-layer attention ~30% at 256k keys. Split-K never beats it on M5
  at 8k keys or more ([perf-baselines](mimo2-perf.md#mimo-attn-kernels)).
- Before the sliding fusion landed, the composed band+sink sheet was the biggest chunk-dependent bill term (0.17 GB
  at chunk 512, 2.28 GB at 2048). That term is now zero wherever the fused arm serves. Global-layer decode no longer
  rebuilds on M4-class GPUs (split-K, above).

## Decode dispatches

- QKV is ONE FP8 GEMV per layer with three outputs (`fp8_block` `gemv3`); V leaves it already multiplied by
  `attention_value_scale` (`RowSplit.v_scale`, rounded to the output dtype first, as the composed multiply did).
- The GEMV's three width arms REASSOCIATE the f32 sum, so a row is byte-identical to a decode tick only at or below
  `MIMO_VERIFY_ROWS_MAX`: direct (<=8 rows) strides each row in 16-byte chunks per lane, staged x (9-16) gives each
  lane one 4-column group per 128-column tile, and the wide arm (>=17) dequantizes the weights to bf16; the serving
  tile (9-128 on NAX) sums the wide arm's bf16 weights in its own order. Past four
  rows the direct arm runs two stored rows per simdgroup (8 per group), which moves no row's sum and beat the staged
  arm by 23-36% at 8 rows on every FP8 trunk shape ([perf-baselines](mimo2-perf.md#mimo-verify-8)).
- Every residual add runs in one kernel with the norm that reads its sum (`fusedAddRmsNormUngated`): the
  post-attention norm (`fusedAddRmsNormRouted` also emits the f32 router input), the next layer's input norm and the
  final norm. The router is widened to f32 once at load (source-trunk packs), not per forward.
- All bit-identical to the ops they replaced: greedy text and top-3 logprobs match over 2x160 tokens.
- Count one decode forward's primitives with `SUSHI_DECODE_FWD_GRAPH=<path>` beside `SUSHI_DECODE_FWD_UBENCH`.
  What is left, per token: the kv8 append (2 quantize + 6 slice updates per layer, 384), the sliding ring's
  dequant (78) and the partial rotary's input copy (96) are ~650 dispatches, but they overlap the heavy kernels:
  removing all three families outright saved under 2% of a decode forward (79a4cb4, 4096 keys, one boot, arms
  interleaved), so fusing them is worth under 1%. The decode idle time is the dependent chain of heavy kernels.
- An MLX custom kernel writes fresh outputs, never the cache in place; writing through an input buffer would bypass
  MLX's hazard tracking and the copy-on-write the prefix-cache snapshots rely on.
- **Any forward of up to `MIMO_VERIFY_ROWS_MAX` rows ropes Q/K and quantizes K/V in one dispatch on a kv8 cache**
  (`mimoDecodeQkvPrep`, FP8 QKV only; decode, verify rows, short tails, the final prompt token): the kernel writes
  fresh quantized rows and `KVCache.appendQuantized` appends them through the usual slice updates, so the ring, its
  compaction and the packed-arm switch are untouched. Bit-identical to the composed rope + `quantizeAffine` (unit test
  at MiMo shapes, both rope bases, 1-8 rows). It removes ~380 primitives from a decode forward;
  `SUSHI_DECODE_FWD_UBENCH_QKV_PREP_ARMS=1` is its A/B.
- A joined `[Q | K]` GEMV output with one rope over both passed its unit tests but moved live logits by ~0.05
  nats at the first token, cause unfound; it is not in the tree.

## Batched decode

- **Concurrent plain slots decode as rows of ONE forward** (`forwardMimoBatchedDecode`): the slots' next tokens
  go through the verify-row path, whose every op but attention already computes a row as its decode tick does;
  row i attends and appends on slot i's own cache at its own position through the solo core (`mimoAttnCore`,
  `ForwardCtx.batch_rows`), so ring compaction and marks happen per slot as in a solo tick, and each row is read out
  as its tick reads it (the shortlist under `argmax_only`). Byte-identical to the solo ticks (`mimo batched decode
  rows` on the real pack; `tests/test_mimo_batched_equivalence.sh`).
- A group holds at most four slots (`batchGroupCap`, independently of the MTP verify width); the rest decode serial by name
  (`row_cap`). Nothing pads: rows never share a key tensor. Groups of three or four eligible MTP slots take
  plain batched ticks and retain each row's hidden state so solo rounds can resume; smaller groups keep solo MTP.
  `SUSHI_MTP_BATCHED=0` disables this MTP crowd policy.
- Measured against interleaved MTP streams: [perf-baselines](mimo2-perf.md#mimo-batched-decode).

## Prompt lookup decoding

- **A PLD verify is MTP's verify** (`ctx.verify_rows`, drafts capped at `MIMO_VERIFY_ROWS_MAX` - 1 = 7): every
  row reads the packed cache as its own decode tick would, a partial accept truncates, and no `KVCache.snapshot` is
  taken, so greedy PLD is serial byte for byte. The prefill-shaped verify it replaced declined the fused kernel below
  16 rows, rebuilt every global layer's whole cache dense, copied the cache on every write under the snapshot and
  re-forwarded partial accepts.
- MTP outranks PLD (`server.requestSpecModes`), so PLD runs only on requests without MTP; an MTP request copies its
  context through prompt lookup inside the round instead ([engine-mtp](engine-mtp.md#lookup)).
- PLD stays on by default: it pays on echo workloads (a code edit that echoes the context). The prompt n-gram gate
  passes ordinary prompts too (score 0.16-0.32 against 0.01), so the runtime yield and per-draft gates are what bound
  the loss on text that does not echo.

<a id="vision"></a>
## Vision (MiMo-ViT)

- **Tower** (`src/mimo_vision.zig`, port of the checkpoint's `MiMoVisionTransformer`): Conv3d patch embed (2x16x16,
  served as its [1280, 1536] Linear), 28 blocks of RMSNorm + GQA attention (32 q / 8 kv heads, head 64, qkv and proj
  biases) + SwiGLU (4608, biases), 2-D rotary positions as Qwen2-VL, then LayerNorm + 5120 -> 5120 -> GELU -> 4096
  over each 2x2 merge unit. The checkpoint ships the merger WITHOUT biases (the modeling code declares them): zero.
- **Attention per block**: 0/9/18/27 full; the rest a band |i - j| <= 64 over the image's patches, in row-major
  (types 0) or column-major (type 1) order of whole merge units, with a per-head sink. The sink is a bias on KEY 0's
  logit (the checkpoint's code and vLLM); SGLang reads it as an extra softmax column. The two readings differ (tiny
  fixture cos 0.98) and `mimo vision tiny` pins ours. A band block runs as query blocks of 64 rows against 192 keys
  through fused SDPA; only the first two blocks can see key 0, so only they carry the per-head sink mask.
- **Positions are 1-D**: an image is `<|vision_start|>` + N `<|image_pad|>` + `<|vision_end|>` at plain text positions
  (both vendor processors assert rope_type "rope"); no M-RoPE.
- **Preprocessing is the vendor processors'** (SGLang and vLLM agree; `preprocessor_config.json` is NOT what they
  read): pixel bounds from config.json `processor_config` (8192 .. 8,388,608, capped at the engine's 1536²),
  `smart_resize` with tiny sides scaled up first, torch bilinear (align_corners=False, no antialias) on 0..255 floats,
  ImageNet mean/std x255 (123.675 / 58.395 ...), Qwen2-VL merge-block patch order. A 1920x1080 screenshot is
  1088x1920: 2040 tokens.
- **Precision: f32 stream, bf16 matmuls.** The residual stream, norms, rope and attention run in f32 around bf16
  Linear matmuls. An all-bf16 tower drifts to cos 0.997 against the f32 reference by the last block, and the
  checkpoint's own tower run in torch bf16 does the same (0.99697; per block ~1.000 through 13, 0.995 at 27).
  Real weights, 448x640 image (280 tokens), vs the reference on the CPU in f32: preprocessing max |diff| 3.6e-7;
  features cos 0.99718 all-bf16 -> 0.99945 f32 stream (RMS ratio 1.0032).
- **Encode time**, one image, best of 5 (`mimo vision ubench`; `taskpolicy -a`, fans max, die <= 65 C at start;
  all-bf16 = eee7d24, f32 stream = the served code, which also evaluates per block):

  | patches (tokens) | all-bf16 | f32 stream | cost |
  |---|---|---|---|
  | 14x14 (49) | 14.1 ms | 21.9 ms | +55% |
  | 68x120, a 1920x1080 screenshot (2040) | 383 ms | 498 ms | +30% |
  | 96x96, the 1536² cap (2304) | 502 ms | 570 ms | +14% |

- **Bills**: `mimoSourceResidentBytes(dir, vision)` adds the tower as stored (1,457,188,864 bytes) when it loads.
  The encode's scratch is the media path's fit check (`server.towerFitFault` -> `visionScratchBytes`, which takes
  `mimo_vision.encodeScratchBytes` for this tower instead of the N² score-sheet formula): a request whose largest
  image does not fit is a named 400. The stream is evaluated per block so one block's buffers are the peak:
  measured 57 / 1645 / 1686 MB at 196 / 8160 / 9216 patches against bills of 111 / 1905 / 2143 MB. Without the
  per-block eval the lazy graph held 508 MB even at 196 patches.
- A streamed load carries the tower only under `--vision`, billed in the ssd budget
  ([engine-expert-streaming](engine-expert-streaming.md#vision)).
- Video (MM:SS timestamp text between 2-frame groups) and audio are not wired.

## Bills (the bill follows the storage in the SAME commit)

- `kvBytesPerToken` counts the 9 global layers per token (spread over `kvPerTokenLayerCount`, never every caching
  layer), `swaRingBytes` the ring once per slot (`server.slotRingBytes`, at `kv_bits`), `swaStreamBytesPerToken` the
  chunk a prefill stages before compaction, for the layers one eval-cadence window lets coexist.
  `server.kvDequantScratchBytes` bills the kv-quant dense rebuild as ONE layer at the rows that layer stores.
- `mimo_source.countResidentBytes` bills each MoE router twice: as stored (bf16) and as the f32 copy the
  transformer loader keeps (~0.2 GB on the Flash pack).
- **With a disk tier, a request bills a restored global layer beside its first grow** (`server.growCoexistBytes`): an
  SSD restore installs each global layer at exactly the restored rows, and the first append grows it while the
  restored buffer is alive. The admission never sees the restore, so it bills one eval window of that at the prompt's
  length (one global layer: 0.54 GB at 400k, kv8).
- **The prefill chunk is chosen per request** (`perRequestPrefillChunk` covers a ringed arch): the widest rung up
  to 2048 whose admission bill fits live memory, which is 2048 at every context to 256k on a 128 GB Mac. An
  explicit `--prefill-chunk` caps the ladder and may raise the default as far as the 4096 ceiling
  (`boundedPrefillChunk`). At 64k the two widths prefill within ~2% of each other
  ([perf-baselines](mimo2-perf.md#mimo-longctx-prefill-attn)), but their bytes differ, so the output of a
  prompt longer than 2048 tokens depends on the width memory allowed. The ungated load-time pin subtracts the
  hot-cache ask first and lands on 512, 1024 or 2048 with the memory active at load. It is only the fallback
  (`SUSHI_PREFILL_CHUNK_PER_REQUEST=0`), and the load line says so.
- **A MiMo prefill evicts the hot cache to be admitted** (`admissionEvictsHotCache`): the warm credit is the restored
  global rows only (`kvBytesPerToken` and `residentCapacityTokens` skip the ring), the ring and the slot's checkpoint
  copies are billed whole every turn, and a shared or SSD restore credits nothing. Adaptive width and mid-prefill
  stepping stay qwen4_exp-only.
- **A ringed arch RESERVES its cache capacity up front** (`ModelConfig.reservesKvCapacity`, narrower than
  `longCtxGated`) and bills the reservation headroom and the ring: growing +25% at a time duplicated a global layer
  mid-prefill.
- Admission bills at kv8: 64k 2.75 GiB, 128k 3.69, 512k 9.29 at chunk 512 (10.34 at 1024), 1M 16.76; 1M at kv8
  needs 12.83 GB of KV alone. An explicit `--ctx-size` outranks auto-context.
- **Resident MiMo takes the wired-limit floor** (`iogpu.wired_limit_mb` minus `--wired-margin-gib`), as Flash-Next and
  a streamed load do. Under the old footprint + free RAM ceiling, ~17 GiB of other apps' memory left 8.7 GiB and a
  768k server refused a 140k prompt; at 120,000 MB the floor leaves ~16 GiB beside a 93 GiB resident MiMo.

## Evidence

- `tests/dump_mimo_v2_fixtures.py` supplies the independent HF oracle; `MIMO_V2_SOURCE` tests the downloaded Flash
  config/template. Native-byte preservation, forward parity, and live serving are separate gates; a header audit
  proves neither numerical parity nor generation. Test commands: [tests/CLAUDE.md](../tests/CLAUDE.md).
- Quality: KLD through EOS on the 16x512 teacher, recorded in [quality-kld](mimo2-kld.md#mimo).
