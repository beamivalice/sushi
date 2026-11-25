# Hy4-Flash (`hy_v4_flash`): the design for Sushi Cooler

Tencent's Hy4-Flash is not served yet. This doc maps its forward pass onto Sushi Cooler, component by component:
the formula, the public source that pins it, and the engine piece it reuses or needs. It also lists the seams the
engine must generalize first, the memory bill, what a pack needs, the parity plan, the open questions, and the parts
the work lands in.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-arch-seam](engine-arch-seam.md),
[engine-own-forward](engine-own-forward.md), [engine-memory-admission](engine-memory-admission.md),
[dsv41-arch](dsv41-arch.md) (the closest served arch), [qwen4-arch](qwen4-arch.md) (vision tower),
[glm5-arch](glm5-arch.md) (IndexPool, MLA latent), [engine-prefix-cache](engine-prefix-cache.md),
[pack-format](pack-format.md).

## Release plan

- Hy4-Flash ships in **sushi 1.8.0**, on Sushi Cooler only. No MLX route is built for it, so it has no MLX opt-out.
  The engine therefore needs a way to declare an arch that is on Sushi Cooler with no opt-out ([seam R1](#r1)).
- The existing archs' MLX opt-outs (`--ds41-mlx-engine`, `--qwen-mlx-engine`) stay through 1.7, 1.8 and 1.9.
- From 2.0, sushi is Sushi Cooler only, with no MLX route for any arch.
- Text first, then images, then the DFly2 drafter: all three ship in 1.8.0. Video is outside 1.8.0.

## Sources

Every formula below is pinned to one of these. The two pull requests are open; line references are to the heads
named.

| source | what it pins |
|---|---|
| [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash): `config.json`, `chat_template.jinja`, `tokenizer_config.json`, `processor_config.json`, `generation_config.json`, `model.safetensors.index.json`, README | geometry, tensor names, template, special tokens, image processor bounds |
| [Tencent-Hunyuan/Hy4-flash](https://github.com/Tencent-Hunyuan/Hy4-flash) | the release's GitHub home (finetune scripts, links) |
| [huggingface/transformers#50903](https://github.com/huggingface/transformers/pull/50903), head `c2ec8505`: `models/hy_v4_flash/modeling_hy_v4_flash.py`, `configuration_hy_v4_flash.py`, `processing_hy_v4_flash.py`, the `hy_v4_flash` block of `conversion_mapping.py` | **the math reference**: Tencent's modeling code, which reads the released checkpoint |
| [ggml-org/llama.cpp#32517](https://github.com/ggml-org/llama.cpp/pull/32517), head `0a054d0b` (`Little0o0:hy4-flash`): `src/models/hy-v4-flash.cpp`, `src/llama-memory-idx-iswa.{h,cpp}`, `conversion/hy_v4_flash.py`, `conversion/hy_v4.py`, `tools/mtmd/models/hy-v4-flash.cpp` | **the runnable reference**: the graph AngelSlim's GGUF runs on |
| [AngelSlim/Hy4-flash-GGUF](https://huggingface.co/AngelSlim/Hy4-flash-GGUF): card, `patch/hy4-flash-mac-minimal.patch` | how the GGUF is built and run on an M5 Max; the Mac patch changes speed only (a fused row-gather indexer, dense attention for short prefill), not the math |
| [Tencent/AngelSlim](https://github.com/Tencent/AngelSlim) | the compression toolkit behind the GGUF |
| [Tencent/AngelSpec](https://github.com/Tencent/AngelSpec), [arXiv 2607.25852](https://arxiv.org/abs/2607.25852) | DFly, the drafter family DFly2 belongs to; at 0505773a, the released drafter's training code, the reference for its forward ([DFly2](#drafter-dfly2)) |

Both references run the released checkpoint correctly. Where they disagree, Cooler follows llama.cpp#32517 (owner,
2026-11-18); the disagreements are listed in [where the references differ](#where-the-references-differ). Elsewhere
transformers is the math reference, and the tiny oracle takes llama.cpp's side at each listed point.

## Checkpoint geometry

| Part | Shape |
|---|---|
| Trunk | 42 layers, hidden 3072, vocab 120,832 (the tokenizer has 120,818), `max_position_embeddings` 1,048,576, `rms_norm_eps` 1e-5 |
| Layer pattern | `layer_types`: sparse at 0, 1, 5, 9, …, 41 (12 layers); sliding window at the other 30 (2 sparse, then 10 × (3 sliding, 1 sparse)) |
| Attention | MLA on both kinds: q LoRA 1024, KV latent 512, qk head 192 nope + 64 rope, v head 256, gated output, a learned sink per head. Sparse layers 32 heads, rope θ 1e7; sliding layers 64 heads, rope θ 1e4, window 512 (511 earlier keys and the token itself) |
| Compressed indexer | sparse layers only: 16 heads × 128, groups of 4 tokens, gated pooling with an intra-group position bias, 511 best groups plus the token's own group (2048 keys), no fine stage (`skip_fine_select`) |
| FFN | layer 0 dense SwiGLU (13,824). Layers 1-41 MoE: 512 routed experts, top-16, one shared expert; routed experts run in a 1536-wide latent (`routed_expert_hidden_size`), width 1536; the shared expert runs at 3072 → 1536 → 3072 |
| Routing | sigmoid scores; an f32 bias (`e_score_correction_bias`) picks only; picked scores normalized, × 3.998; SwiGLU clamp 10 on routed experts only |
| Hyper-connections | iHC, 4 streams, no comb matrix (no Sinkhorn), `hc_eps` 1e-6, post magnitude 2.0 |
| Engram | 4 modules, at 0-based layers 1, 11, 21, 31 (config ids 2, 12, 22, 32 are 1-based); bigrams and trigrams, 8 hash heads per order, 96 dims a head, 16 prime buckets of about 10.31M rows each per module: 660,006,416 rows in all |
| Vision | SigLIP2-derived, 27 layers × 1152, MLP 4304, 16 heads, patch 16, learned 128 × 128 position table, merge 2 × 2 by two convolutions, a newline embedding per merged row |
| Drafter | DFly2: 5 layers under `model.dfly_layers.*`, reading the target's KV and indexer picks at layers 25, 29, 33, 37, 41; `num_nextn_predict_layers` is 0 |

The tokenizer is a byte-level BPE (120,000 merged tokens, 832 added) whose `pre_tokenizer` is DeepSeek-V4.x's to the
byte (three isolating Splits, then a ByteLevel that does not split; no normalizer), so Hy4 runs
`src/tokenizer_deepseek.zig`, shared with DeepSeek's chat face.

Storage: 148 shards, 1,964 tensors, 440,520,833,910 bytes, all BF16 except the f32 tensors the reference keeps in f32
(`_keep_in_fp32_modules_strict`: the hyper-connection mixes, scales and bases, sinks, the router bias, the indexer's
two pooling projections and its position bias). The routed experts are fused per layer: `mlp.experts.gate_up_proj`
`[512, 3072, 1536]` (gate rows first, then up) and `mlp.experts.down_proj` `[512, 1536, 1536]`, 7.25 GB a layer over
three shards (4.83 + 2.42 GB). `lm_head.weight` is stored BF16: the index's total holds only with a BF16 head, leaving
exactly the 0.85B DFly2 parameters the README states.

| part | GB (BF16 on disk) |
|---|---:|
| routed experts (41 × 512) | 297.16 |
| Engram tables | 126.72 |
| attention, sliding (30 × 64 heads) | 8.22 |
| attention, sparse (12 × 32 heads) | 1.70 |
| DFly2 | 1.69 |
| shared experts | 1.16 |
| vision tower | 0.93 |
| latent down/up projections | 0.77 |
| embedding, lm head | 0.74 each |
| dense FFN (layer 0) | 0.25 |
| Engram projections, router, indexer, hyper-connections, norms | 0.42 |

`hy4_census.zig` maps every name to its role and these parts, and its test sums the release's tensor list
(`src/hy4flash/fixtures/hy4_tensors.txt`, read from the shard headers) to this table to the byte.

Checkpoint names are not transformers' names. The renames are in the `hy_v4_flash` block of `conversion_mapping.py`.
- `self_attn.learnable_sink_param` → `sinks`
- `self_attn.linear_gate` → `gate_proj`
- `hc_attn_layer.hc_pre.hc_{fn,base,scale}` → `attn_hc.{fn,base,scale}`, and `hc_mlp_layer` the same → `ffn_hc`
- `hc_head.hc_head_*` → `hc_head.hc_*`
- the vision tower: `vit.*` → `model.vision_tower.*`, `perceive.*` → `patch_merger.*`, Megatron MLP names → `fc1`/`fc2`

Sushi reads the checkpoint's own names, as DeepSeek's loader does.

## The component map in short

| component | Cooler piece | kind |
|---|---|---|
| token embedding, image splice | trunk embedding gather; `spliceVisionRows` | reused as is |
| Engram hash, row ids | DeepSeek's `Hash` (multipliers, prime buckets, rolling XOR) | reused with a parameter ([R2](#r2)) |
| Engram table read, row cache, read-ahead | DeepSeek's `Table`, `RowCache`, `Prefetch` | reused with a parameter ([R2](#r2)); a BF16 / plain e4m3 row format is new |
| Engram gate and delta | — | new kernel (the `dsv41_engram` gate chain's shape) |
| iHC mix, collapse, expand, head | DeepSeek's hyper-connection stage kernels | reused with a parameter: no comb, post magnitude and eps ([R3](#r3)) |
| norms (RMS, sandwich) | the engine's RMS norm | reused as is |
| MLA projections (q LoRA, KV latent, gate, output) | trunk linears, stacked input launch | reused as is |
| absorbed MLA attention, window ring and picked rows, sink | DeepSeek's attention kernels | reused with a parameter: K 576 / V 512 on one row, no rotation fold ([R5](#r5)) |
| RoPE (consecutive pairs, θ per layer kind; image rows at consecutive positions) | DeepSeek's pair rotation | reused with a parameter |
| indexer pooling (gated, position bias, pooled-key norm and rope) | — | new kernel (GLM's IndexPool is the algorithm's relative, on MLX) |
| indexer scores and group top-k | DeepSeek's index score and top-k kernels | reused with a parameter: BF16 pooled keys, 16 × 128, k 511 ([R4](#r4)) |
| router (sigmoid, bias picks, normalize, × 3.998) | `moe.biasRouting` family | reused with a parameter |
| routed experts (clamped SwiGLU, rate groups, SSD streaming) | the STQ rate-group chain, the expert cache | reused as is (width 1536 × 1536) |
| latent down/up around the experts | two trunk linears | new arch code (plain linears) |
| shared expert, dense layer 0 | trunk SwiGLU | reused as is |
| lm head in f32 | DeepSeek's f32 head | reused as is |
| Sushi Cooler side, no opt-out | `cooler_side.sideOf` | seam generalized ([R1](#r1)) |
| vision tower blocks | Qwen's tower (same widths) | reused with a parameter (Conv2d patch, position table rule, LayerNorm with bias, no 2-D rope) |
| vision merger (two convs, newline) | — | new arch code |
| chat template, tool calls | the Zig Jinja engine; GLM's `arg_key` tool-call family | reused as is; tool calls with a parameter (the `:opensource` tag spelling) |
| prefix cache restore point | DeepSeek's whole-state checkpoint and row segments | reused with a parameter |
| 8-bit KV latent (default) | the attention core and the row writes | the engine's affine kv8 (group 64), as Qwen's and GLM's latent rows; rope dims BF16 ([KV width](#kv-width)) |
| DFly2 drafter | DSpark's block rounds (`dspark_rows.zig`, the drafter slot's `dspark_rounds`) | round machinery reused with a parameter (block 6, a mask token for the noise rows); the drafter's forward is new arch code |

## The forward, component by component

Notation: `x` is a row of the 3072-wide residual input to a sublayer, `h_c` (c = 0..3) the four hyper-connection
streams, `RMS(v) = v · rsqrt(mean(v²) + ε)`, `RMSNorm_w(v) = w ⊙ RMS(v)`. The reference casts `RMS(v)` to the
activation dtype before the weight multiplies.

### Token embedding

- **Formula**: `e_t = E[id_t]` (BF16). Image rows replace `e_t` at the image placeholder positions. All four streams
  start as `e_t`.
- **Pinned by**: transformers `HYV4FlashTextModel.forward` (`inputs_embeds.unsqueeze(2).expand(..., hc_mult, ...)`);
  llama.cpp `graph::graph` (`hc_init`).
- **Cooler**: reused as is. That is the trunk embedding gather of the BF16 table and the vision splice. An STQ pack's
  K6 embedding bank ([pack-format](pack-format.md#hy4-stq)) is gathered on the GPU on every Mac (`hy4_embed`): the chunk's 128-id blocks decoded (`trunk_gemm.decodeDense`), then two kernels take the Hadamard down
  each 128-row chunk and along the token's column and scale by `suh·svh`, in the host decode's f32 order, so a row is
  `EmbedGather.rows`' bytes. The decode needs only NAX's fragment layout, not its matrix units, so an M1 or M2 runs it
  too (a host gather made the DSpark draft read each slot's token back to the CPU: five waits a draft, not one). The host
  gather serves only off the GPU; it keeps the blocks it decoded last, to `EmbedGather.block_cache_bytes` (64 MiB, 42
  blocks at dim 3072), billed with its copy (`hy4_load.embedCacheBytes`). A group launches at a power-of-two block count (`launchBlocks`):
  the decode is templated on its width, so its six widths are made at load (`warmKernels`), never inside a request, with
  a second group's join and its take back into the ids' order (a prompt's ids are in no block order). Its
  buffers are a forward's. On the engine the last group is not evaluated: host data is
  copied as it is made, and that eval drained the GPU once a decode token (2 waits a token to 1 on the tiny STQ pack).

### Engram (layers 1, 11, 21, 31)

Engram runs before the layer's attention pre-mix and adds into the four streams.
- **History**: for each position `t`, the token ids `t, t-1, t-2`. A slot before the sequence start reads the pad id
  (120002). Each id maps to a compressed id through the checkpoint's `engram_layer.lookup_table` (int32, 120,818
  entries); an id at or past 120,818 reads the pad id's entry.
- **Multipliers**: `m_k = 2·d_k + 1` for k = 0..2. `d` is numpy `default_rng(engram_seed + 10007·L).integers(0, B)`,
  where `L` is the **1-based** config id (2, 12, 22, 32) and `B = (2^63 - 1) // C // 2`. `C = max(lookup_table) + 1`.
- **Row ids**: for order `n` in {2, 3}, `mix_n = (c_0·m_0) xor … xor (c_{n-1}·m_{n-1})` in wrapping 64-bit. Head `j`'s
  row is `mix_n mod p_{n,j}` plus that head's offset in the module's table (offsets are cumulative over the 16 primes,
  order-major). The config lists the primes. They are exactly the sequence DeepSeek's loader derives (the next primes
  after `engram_vocab_size - 1`, continuing across heads, orders and modules); this was checked over all 64.
- **Rows**: `r_j = T[row_j]`, 96 values as stored, widened to f32 (llama.cpp's `get_rows`). The table is BF16 in the
  checkpoint; transformers' per-row FP8 round trip (`engram_fp8_embedding`) is not applied.
- **Gate and delta**:
  - `g_j = RMSNorm_{w_j}(r_j)` per head, ε the fp32 machine ε (1.19e-7), and `u = concat_j g_j` (1536 wide).
  - Keys `k_c = RMSNorm_{K_c}(W_c u + b_c)` for c = 0..3; value `v = W_v u + b_v`.
  - `d_c = <k_c, RMSNorm_{Q_c}(h_c)> / sqrt(3072)`, and the gate is `a_c = sigmoid(sign(d_c)·sqrt(max(|d_c|, 1e-6)))`.
  - `y = γ ⊙ RMS(v)`, and `h_c ← h_c + (a_c + σ_c) ⊙ y`. `γ` is `unified_pre_gamma`; `σ_c` is the per-channel
    `unified_per_hc_scales[c]`.
  - The delta is zero where the position's own token is an image, video or pad id. In the history an image position is
    dead and reads the pad id; an id at or past 120,818 maps by identity.
  - One mapping makes every history id (`Model.historyId`: image to pad, then compressed): the forward, a restored point and
    the layer-major run all go through it, so a restored point's history is the cold run's.
- **Pinned by**: transformers `HYV4FlashEngramLayer` (`hash_ngrams`, `draw_hash_multipliers`, `forward`),
  `_fp8_round_trip`, `HYV4FlashTextModel.build_ngram_tokens`, `HYV4FlashModel.build_engram_inputs`; llama.cpp
  `engram_hash`, `graph::build_engram` (which folds `norm_key·norm_query` into one `engram_gate` vector); converter
  `_engram_multipliers`.
- **Cooler**:
  - The hash is DeepSeek's `Hash` with two parameters, the seed and the id the multipliers are drawn with
    ([R2](#r2)).
  - The table is never resident. That is 126.7 GB BF16, or 63.4 GB as plain e4m3: read by row from SSD through DeepSeek's
    `Table`, `RowCache` and `Prefetch`. The read covers the prompt's next chunk ahead, and a decode token's rows hide
    behind the layers before layer 1.
  - The gate chain is a new kernel of the `dsv41_engram` shape. Hy4's has biases, per-head lookup norms, `γ` and an
    additive per-stream scale where DeepSeek's has none of these.
  - The token map is the checkpoint's `lookup_table` tensor, not an embedded file.
- Per decode token: 4 modules × 16 rows = 64 rows, 12 KB of BF16.
- On a GPU stream `fwd.engramAdd` runs the gate chain in 21 launches a module instead of about 115: the 16 head
  norms in one (`hy4_engram_gate.headNorms`, each head's weight read as stored), the value's bias and square in one,
  each key's bias add and norm and each query norm times its key in `hy4_norm`'s launch, the five linears and five
  row sums as they were, and the gates' write into the streams in one (the value's RMS and `γ`, sign, √, σ, `σ_c`,
  the live mask, the add), all in the ops' f32 order. Seam `fwd.chain_kernels_on`.

### Hyper-connections (iHC)

- **Pre and post**:
  - `f = concat_c h_c` (12,288 wide), `μ = rsqrt(mean(f²) + 1e-5)`, `m = (F·f)·μ` with `F` `[8, 12288]` f32.
  - `pre_c = sigmoid(m_c·s_0 + b_c) + hc_eps`, and `post_c = 2·sigmoid(m_{4+c}·s_1 + b_{4+c}) + hc_eps`.
  - The sublayer input is `x = Σ_c pre_c·h_c`. After the sublayer output `y`: `h_c ← h_c + post_c·y`.
  - There is no comb matrix and no Sinkhorn: each stream keeps its own residual.
  - Mixes and collapse run in f32, and the streams stay f32 between sublayers, as llama.cpp keeps them (transformers
    rounds them to BF16).
- **Head**: `m = (F_h·f)·μ` with `F_h` `[4, 12288]`, `pre_c = sigmoid(m_c·s + b_c) + hc_eps`, `x = Σ_c pre_c·h_c`, then
  `RMSNorm_{model.norm}`.
- **Layer order**: Engram (where present) → attention pre-mix → `input_layernorm` → attention → post. Then FFN
  pre-mix → `post_attention_layernorm` → FFN → post.
- **Pinned by**: transformers `HYV4FlashHyperConnection`, `HYV4FlashHyperHead`, `HYV4FlashDecoderLayer.forward`;
  llama.cpp `build_hc_pre`, `build_hc_post` ("comb is the identity here"), `build_hc_head`.
- **Cooler**: DeepSeek's stage kernels compute the same mix product, collapse and expand, plus a Sinkhorn comb. With
  the comb dropped, and the post given a magnitude and an eps, they serve Hy4 ([R3](#r3)). The mixes are 8 a layer
  (4 pre, 4 post), against DeepSeek's 24.

### Attention, both kinds (MLA)

- **Query**: `q_r = RMSNorm(W_qa x)` (1024 wide), `q = W_qb q_r` → H heads × 256 = 192 nope + 64 rope.
- **Latent**: `[c_kv | k_pe] = W_kva x` (512 + 64), `c = RMSNorm(c_kv)`, and `k_pe` is rotated.
- **Absorbed form** (as llama.cpp runs it): the key row is `K = [c | rope(k_pe)]` (576) and the value is `V = c`
  (512), one row per token shared by every head. Head h's query is `[W_kb,h^T q_nope,h | rope(q_pe,h)]`.
  - Score: `q·K / 16`. The scale is 1/sqrt(256), from the 256-wide qk head, not the 576 row.
  - Softmax over the visible keys plus the head's sink logit (f32), the sink's own share dropped. That gives `o_h`
    (512).
  - `u_h = W_vb,h o_h` (256), then `u ⊙ sigmoid(W_g x)` elementwise (32 or 64 heads × 256).
  - The output is `RMSNorm(W_o u)` (`post_proj_norm`, the attention half of the sandwich).
  - On a GPU stream the gate `u ⊙ σ(W_g x)` is one launch (`hy4_chain.gated`: the gate rows widened, MLX's Sigmoid,
    the product, written as `W_o` reads it), not the cast, the Sigmoid, the Multiply and the transpose's copy.
- **Pinned by**: transformers `HYV4FlashAttention.forward` (the expanded form, the same math) and
  `eager_attention_forward` (sinks); llama.cpp `graph::build_attention`.
- **Cooler**:
  - The projections are trunk linears; `W_qa`, `W_kva` and `W_g` read the same `x` and stack into one launch, as
    DeepSeek's `Layer.in_shared`.
  - A decode row's BF16 linear (the STQ pack's `W_kva`, stored BF16) is one launch (`hy4_gemv.rowsThru`): the row
    rounded to BF16 in the register, MLX's `gemv` tile over the BF16 operands, the sum rounded to BF16 and widened,
    not the casts around MLX's BF16 `gemv`. A verify block of 2 to 8 rows is one launch too, the weight read once
    and each row its one-row sums (3-row verify 21.68 to 21.41 ms a forward, 92d6397b, M5 Ultra, STQ 4bpw, A B B A).
  - A decode row's attention past the window is a score, a softmax and a value launch (`hy4_chain.steelScores`,
    `steelValues`): each sums in MLX's split-k or fused order (`steelPlan` mirrors the pinned mlx's choice), so the
    bytes are the op chain's; the output gate rides the launch that reads `W_v` (`hy4_absorb.valueOutGated`) for a
    span that returns its rows (a chunk's span, even of one row, writes into the chunk's block through `gatedInto`), and
    the absorbed query reads the queries before their rotation (`Projected.q_raw`), so the rope runs beside it. A row:
    15.45 to 14.84 ms, 1,618 to 1,378 dispatches (dbcca609, M5 Ultra, STQ 4bpw, kv8, 4,096 tokens, A B B A).
  - A decode row's absorbed products (`hy4_absorb`) read the query's nope part in place (template `QS`, the row's
    width) and keep their launch configs; their grid's y takes a block's rows, each its one-row launch's bytes.
  - The attention core of a wide span (more than 8 rows, on an M5) is `hy4_attn`, one launch a span on the matrix units
    ([R5](#r5)): the scores over the 576-wide key row, the sink softmax in f32, the value sum over its first 512
    channels. A decode row and a verify block (2..8 rows) never reach it: they keep the op chain's bytes.
  - GLM's absorbed MLA (`glm5_attention.zig`) is the same math, but on MLX only.
- **The composed forward's spans** (`hy4_model.layerFront`):
  - A chunk's attention runs in spans of `attn_span` (256) rows, `gathered_span` (32) on a sparse layer past its
    selection, each in a scope of its own over only the keys it reads. A span's bytes follow its width: the softmax
    and the value sum run over its key range, whose length and alignment move with the span's first row. The spans
    stay; widening them moves bytes, and a sliding span of `S` rows reads `S + 511` keys for 512 useful ones.
  - What every span reads is made once a chunk: the gate over the chunk's rows (held as its product stores it, BF16
    on the served trunk), `W_kvb` widened to f32, and `W_o` with `post_proj_norm` over the spans' gated rows joined
    (each span hands them on as `W_o` reads them, BF16). Before, each span ran its own gate and `W_o`, re-reading
    both weights 16 times a 4,096-row chunk on a sliding layer and 128 times on a sparse one.
  - `fwd.linearSpans` gives each row its own span's launch's bytes: the whole spans in one launch and a short last
    span in one of its own (a launch of a few rows takes the vector kernels, other sums). A BF16 product takes the
    kernel, tile and k split chosen for one span (`mlx.engine.matmulRowsAs`): MLX's choice follows the row count,
    and without it the release's gate at 512 rows and its sliding `W_o` at 8,192 (past the split-K bound) move bytes.
    A trellis bank's sorted GEMM, and its decoded path past 1,024 rows, give a row its bytes at any count past the
    rows arm, so they launch as they stand.
    The rows arm takes a width as a template value, so a short last span's one to eight rows through a bank's gate and
    `W_o` are made at load (`warmSpanTails`); a width past the arm shares the GEMM's pipelines (`hy4 late` cells).
  - A wide span's core reads its keys where they are held, not gathered per span: a sliding layer's window rows, and a
    sparse layer's rows to the chunk's end, are read from kv8 to BF16 once a layer and chunk (`hy4_latent.rangeBf16`;
    the copy grows with depth, so a deep-zone chunk runs padded to its class's deepest recording: [D110](bill_redesign.md)),
    and a row's picks are read from that copy by index in the kernel, never copied (the chain gathered 2,048 rows a
    row, widened to f32, 134 MB a 32-row span).
  - The region and the classes a load measures are kept beside its manifest and read back by the next load of the
    build where one probe still asks what it asked ([D111](bill_redesign.md)).
  - **A wide forward's structure turns with depth only through `Model.wideClass`**: an edge, an arm or a span width
    that depth decides goes into it, or a chunk of one class asks unlike its recording and is refused by name (the
    `hy4 plan: a load-width prefill chunk` cells sweep every residue and span-width edge).
  - A prefill chunk's glue runs in the launches beside it, each byte for byte: the iHC mix, expand and write-back and
    the router's pick take wide prefill rows in their fused kernels; `post_proj_norm` rides the expand's mix
    (`hy4_hc.expandNormMixNorm`); a span's gated rows land in the chunk's `W_o` block (`hy4_chain.gatedInto`); a
    sparse layer's indexer projections, rope and group keys run once a chunk (`fwd.indexPrepare`); sparse spans widen
    to 256 rows, their products at 32-row launches' bytes (`matmulRowsAs`); BF16 rows stay BF16 into the norms, rope,
    SwiGLU and the routed latent up. A 4,096-row chunk after 4,096: 2,240 to 1,952 ms, 24,291 to about 13,000
    dispatches (bc7f5124 on 36090072, M5 Ultra, STQ 4bpw, prefill meter, A B B A, GPU lock).
  - A wide chunk's trunk banks past the shared decoded cap (the sliding layers' gate and `W_o`, layer 0's dense FFN)
    are decoded to f16 for the forward and multiplied on NAX (`hy4_stq.applyWideDecoded`, cap
    `decoded_max_weights`), forced to the fused arm where MLX would split K; on the Ultra a wide K takes the 128 x 64
    tile (`matmul.fusedNaxTileFor`); and a chunk's spans write `W_o`'s prepared rows (`hy4_chain.gatedPreparedInto`),
    so its sorted GEMM runs no prepare. Each keeps the sorted GEMM's bytes. A 4,096-row chunk after 4,096: 2,075 to
    2,012 ms (024543f7 on dff0cc49, M5 Ultra, STQ 4bpw, prefill meter, A B B A, GPU lock).
  - A wide chunk's trellis SwiGLU finishes its gate and up, runs the SwiGLU and prepares the down bank in one launch
    (`hy4_chain.swigluFinishPrepare`), and the shared expert's down and `lat_up` finish inside the iHC back-mix and the
    expand's norm (`hy4_hc.backMixNormInner`, `hy4_norm.rowsExpandInner`); the routed windows' inverse and scatter
    spread over threadgroups (`rate_groups.groupWindowGrid`). Byte for byte; a 4,096-row chunk after 4,096: 1,986 to
    1,971 ms, 208 fewer dispatches (018502b3 on 5f6fb12e, M5 Ultra, STQ 4bpw, prefill meter, A B B A, GPU lock).
  - Pinned by `hy4 attention spans`: at the release's gate and `W_o` shapes (trellis K4 and BF16, spans of 256 and
    32, short last spans) against each span's own launch, byte for byte. The tiny's logits also matched the parent
    commit's by file (366 rows, both trunks).
  - `sushi-ubench run hy4.span_linears` (9eee9eb3, M5 Ultra, 7 rounds, GPU lock, AC), a layer's gate and `W_o` through
    trellis banks, per span against whole, ms at 4,096 rows: sliding K4 12.02 / 11.47, K6 14.32 / 13.89; gathered
    K4 8.11 / 5.86, K6 8.62 / 5.26. At K4 that is about 43 ms of a 4,096-row chunk's 30 sliding and 12 sparse layers;
    the launches run 0.46 s a chunk on this box whole or by span, so a synced stage profile's per-span waits, not
    the weight re-reads, made most of their share.

### Sliding-window layers (30)

- **Formula**: the keys are the last 512 slots (the token and the 511 before it). The mask counts cache slots, not
  positions. 64 heads, θ 1e4.
- **Pinned by**: transformers `create_sliding_window_causal_mask` with `sliding_window` 512 and `window_size`
  `[511, 0]`; llama.cpp `LLAMA_SWA_TYPE_STANDARD` with `n_swa` 512.
- **Cooler**: DeepSeek's window ring reused with a parameter. The ring holds 512 rows of 576, written at
  `pos % 512` by the rope-and-write launch (`dsv41_rope_kv_ring`'s pattern).
- **The composed forward today** (`hy4_model`): each sliding layer's latent and rope rows are a ring of 512 rows
  (`writeRing`), position `p` at row `p % 512`, written in place: the ring's handle is its only reference when it
  writes, so the engine updates it where it stands. A decode row writes its row, then reads the ring as stored (the keys
  carry their positions, and the row reads all of them). A chunk reads the ring's last 511 rows in position order before
  its own rows (`windowKeys`), and writes its last 512 rows after its spans, in at most two runs at the wrap.

### Sparse layers (12) and the compressed indexer

- **Token key**: `k_t = RMSNorm(W_k x_t)` (128 wide).
- **Groups**: 4 consecutive cache slots; only complete groups are pooled. In f32:
  - `v = W_v k` and `a = W_g k + ape[slot]`;
  - `K_g = RMSNorm( Σ_slot softmax_slot(a) ⊙ v )`, per channel, rounded to BF16;
  - the trailing 64 dims are rotated at the position of the group's first slot (θ 1e7).
- **Query and weights**: `Q = W_q q_r` → 16 heads × 128, the trailing 64 dims rotated at the token's own position.
  Head weights are `w = (W_w x) / sqrt(16·128)`.
- **Score**: `S(t, g) = Σ_h w_h · relu(Q_h · K_g)` in f32, over groups `g < ⌊slot_t / 4⌋`.
- **Selection**: the 511 highest-scoring groups, plus the token's own group, forced. That is at most 2048 keys; the
  own group's slots past the token are causally masked. Attention then runs over those keys only.
  - A query whose own group index is below 512 (slot < 2048) keeps every earlier group, so it attends densely. The
    reference also skips the indexer outright while the whole cache has 2048 keys or fewer, which gives the same
    result.
  - Whether a query is dense is a function of its own slot only, so a prefill chunk's rows do not depend on the
    chunking.
  - `skip_fine_select` is true: there is no token-level second stage.
- **Pinned by**: transformers `HYV4FlashCompressedIndexer` (`pool_groups`, `forward`) and
  `HYV4FlashAttention._mask_unrecalled_tokens`; llama.cpp `build_kpool_select` and
  `llama_memory_idx_iswa_context::set_input_kpool` (the forced current group is the "tail").
- **Cooler**:
  - Scores and top-k are DeepSeek's index score kernel (relu-weighted, unseen groups at -inf) and its top-k kernels
    (one threadgroup, split over 8 chunks, the below-mask), over BF16 pooled keys with k 511 ([R4](#r4)).
  - The pooling is a new kernel, run once per completed group. GLM's IndexPool is the algorithm's relative: it pools
    the key itself, gated off the hidden state, has no rope, and keeps 512 pools plus a 0-3 token tail.
  - The picked rows feed the attention core's picks path.
  - On a GPU stream the groups (the picks widened beside each row's own group) and the recalled rows and their keep
    mask are a launch each (`hy4_chain.recallGroups`, `recalledRows`), in the ops' f32 order: they were 4 and 10
    small dispatches a sparse layer. Seam `fwd.chain_kernels_on`.
  - A decode block's scores are one launch (`hy4_index.scores`, up to 8 rows, f32 pooled keys): the query's and each
    key's tail rotated as they stage in threadgroup memory (`hy4_rope.tail`'s arithmetic), then the shared score
    kernel's sum, a key on 16 lanes (one head each, its dot in channel order) and the heads summed in order by the
    key's first lane. It replaces the query rope, the rotated f32 copy of every pooled key each token made, and the
    shared kernel's one thread a key (4 threadgroups at 1,024 groups). Seam `fwd.index_kernel_on`.
  - A prefill span's scores are one tiled launch (`hy4_index.wideScores`, f32): 8 rows by 32 keys a threadgroup, the
    queries and keys staged 16 channels at a time and each head's dot summed in channel order, so a score is
    `sparse.indexScores`' bytes; a tile past every row's seen groups writes −∞ and reads nothing. The shared kernel
    re-read a row's queries for every key, and its share of a chunk grows with the context.
  - The top-k writes each pick's group, key rows and keep flags where `recallGroups` and `recalledRows` write them
    (`hy4_index.pick`, `sparse.TOPK_BODY`), while it stays one threadgroup a row: one launch, not three. The picks
    do not depend on its thread count (`hy4_index.pick_tg`, a seam for timing it).

### RoPE

- **Formula**: consecutive pairs `(x_{2i}, x_{2i+1})` (GPT-J order, ggml `NORM`) on the trailing 64 dims of the query
  and key (the indexer's too). `inv_freq_i = θ^(-2i/64)`, with θ 1e7 on sparse layers and the indexer and θ 1e4 on
  sliding layers. `rope_type` is default: no YaRN.
- **Image positions**: one plane, as text. llama.cpp gives a `NORM`-rope decoder `MTMD_POS_TYPE_NORMAL`, so each image
  row (newline rows included) takes the next position, and text resumes after the last. transformers' interleaved
  3-D M-RoPE (section [12, 10, 10]) is not used. cos and sin are computed and applied in f32.
- **Pinned by**: transformers `HYV4FlashRotaryEmbedding` (`recomposition_frequencies`), `apply_rotary_pos_emb`
  (`repeat_interleave`, pairwise `rotate_half`); llama.cpp `LLAMA_ROPE_TYPE_NORM` for `hy_v4_flash`, and mtmd's
  `pos_type` from the decoder's rope type (`mtmd.cpp`).
- **Cooler**:
  - The pair rotation is DeepSeek's tail rotation without YaRN, θ by layer kind.
  - No position index: image rows are ordinary consecutive positions.
  - On a GPU stream one launch (`hy4_rope.tail`) copies a row's leading dims and rotates its tail in MLX's rope
    arithmetic, reading a view through its strides: the slice, MLX's rope over a copy and the concat were four
    dispatches for the query, two for `k_pe`, which it writes at the cache's BF16. Seam `fwd.rope_kernel_on`.
  - An attention's latent norm and its `k_pe`'s rotation are one launch (`hy4_front`, seam `fwd.front_kernel_on`),
    a threadgroup a row: the latent normed as `hy4_norm.rows` does, then `k_pe` rotated as `tail` does. Both read
    `kv_a` alone; folding the query's rotation in too made them wait for `q_b`, a dependency level (a barrier) more a
    layer, so it stays a launch of its own.

### MoE (layers 1-41) and the dense layer 0

- **Router**: `ℓ = W_r x` in f32 (the BF16 weight widened, which is exact), `s = sigmoid(ℓ)`. Pick the top 16 of
  `s + b` (`b` is the f32 bias). Weights are `s_e / (Σ_picked s + 1e-20) × 3.998`. With `n_group = topk_group = 1`
  there is no group stage.
- **Routed experts, in the latent**:
  - `z = W_down x` (3072 → 1536). For each pick e, `[g | u] = W_gu,e z` (gate rows first), `g ← min(g, 10)`,
    `u ← clamp(u, -10, 10)`, `y_e = W_d,e (silu(g)·u)` (1536).
  - `R = W_up Σ_e w_e y_e` (1536 → 3072).
- **Shared expert**: SwiGLU at full width (3072 → 1536 → 3072), no clamp. On a GPU stream its `silu(g) · u` (and
  the dense layer's) is one f32 launch (`hy4_chain.swiglu`, MLX's Sigmoid then the two products, no contraction),
  not the Sigmoid and two Multiply dispatches. Seam `fwd.chain_kernels_on`.
- **Out**: `RMSNorm(R + shared)` (`combine_output_norm`, the FFN half of the sandwich). Layer 0 is a dense SwiGLU
  (13,824) followed by `post_fc2_norm`.
- **Pinned by**: transformers `HYV4FlashTopkRouter`, `HYV4FlashExperts._apply_gate`, `HYV4FlashMoE`,
  `HYV4FlashMLP`; llama.cpp `build_latent_moe` with `hy_v4.py`'s sigmoid gating and clamp keys.
- **Cooler**:
  - Routing is the arch's `moe.biasRouting` hook (sigmoid, bias picks only, normalize, × 3.998, 512 experts, top-16).
  - A decode row's route past its logits is one launch (`hy4_route.pick`): the Sigmoid, the bias, the top 16 in the
    order the engine's stable argsort of the negated scores gives (larger first, ties by lower index, NaN last), the
    sum in `row_reduce_small`'s order, the division and the scale, and a resident stream's slots for the picks.
    They were ten dispatches and the slot map's take and casts. Seam `fwd.chain_kernels_on`.
  - **The top 16 is one bitonic sort of 64-bit keys** (the negated score as an order-keeping uint, -0 as +0, NaN past
    every number, then the index), strides under 32 by simd shuffles. Each of 512 threads ranking its expert against
    all 512 took 33 us a launch on one core, 1.36 ms a token, at 259c0a5a on the M5 Ultra under the kernel profile.
    That rank loop also left NaNs unordered: a NaN counted only lower-index NaNs, so it took a number's rank and two
    threads wrote one pick.
  - At 257 to 512 experts and top 16 or fewer the route selects instead of sorting whole (the keys are a total order, so
    the picks, their order and the sums are the sort's bytes): each simdgroup sorts its 32 keys by shuffles and keeps 16, one
    simdgroup merges four such lists by shuffles, and one merges those four. Two barriers, against the sort's 22
    (`hy4_route.fast_barriers`, `sortBarriers`). The tiny models' 8 experts keep the sort.
  - `--expert-cache-bias n` on a streamed STQ load: a decode-width route adds the layer's penalty (-n on each expert
    its cache lacks) to the bias row the launch reads (`fwd.cacheBiased`), in sigmoid units; on the tiny at 4 slots a
    layer it raises the decode's hit rate, and at n = 8 no decode route misses (`hy4_stq_test`). Lossy; KLD on the
    release pack not yet measured ([engine-expert-streaming](engine-expert-streaming.md#cache-bias)).
  - The routed chain is the STQ rate-group chain with the clamped SwiGLU at 1536 × 1536, streamed through the expert
    cache from `--ssd-budget-gb` as DeepSeek's are ([engine-exl3-experts](engine-exl3-experts.md#rate-groups)). Both
    run tile-major: the resident bank is `raggedOwned(.pair_major)` and the stream's pools take the pack's tiles
    unchanged (`ExpertStreamTraits.tile_major`, no CPU reshuffle; a page-aligned pack reads straight into them, its
    pools' page padding billed with the layer by the stream), the funnel decode arms and the `_tm` prefill GEMMs read
    them, and a load's pipeline manifest is keyed by the tile order. The bytes are the pack order's (`hy4 pair-major`
    cells: a bank in each order, decode to a prefill). A load skips every `__pad__.` key a repack lays between
    trellises (never loaded or billed). With
    no flag and no setting Hy4 takes the RAM-tier default as DeepSeek does (`serveDefaults.ram_budget`): the 4bpw
    pack's 512 slots a layer (74.68 GB, a need of about 81 GB) resident on a 192 or 256 GB Mac; Engram's tables are
    never in it, they are read by row from SSD. The vision tower stays on such a load (`vision_on_streamed`): 512 slots
    beside it on 128 GB and up, 390 on 96 GB; `--no-vision` drops it.
  - A request that omits `max_tokens` gets 65,536 (`serveDefaults.max_tokens`), the completion cap Tencent serves the
    release at (its OpenRouter listing); thinking counts toward it. `--max-tokens` or the model's `max_tokens` setting
    outranks it.
  - The latent projections are two plain trunk linears around the chain.
  - The shared expert reads `x`, not `z`, so the shared fold into the routed launches does not apply; it runs as the
    trunk SwiGLU.
  - The BF16 checkpoint (the internal path: teacher and capture) streams its experts through the engine's `gather_mm`,
    as Qwen's BF16 checkpoint does on Sushi Cooler. The fused `gate_up` bank is split by views (gate rows 0..1535, up
    rows 1536..3071).
- **The composed forward today** (`hy4_forward.routedExperts`, `hy4_model.moeLayerImpl`):
  - Each pick runs two gathered products over the banks as stored (BF16, f32 sums), its SwiGLU and clamp in f32, in
    blocks of at most 8,192 picks (`pick_block`), each block placed at its picks' positions before a row sums its 16.
  - A streamed layer's cache holds no union workspace (`Geometry.union_within_cache`; a BF16 layer's 512 would take
    7.25 GB): the picks go in groups of at most the cache's slots of distinct experts, each group landing its own
    picks at their positions in one [T·k, L] sum (`placeExperts`), so the rows sum their k as one group's would: the
    grouped path gives one group's bytes (summing each group's rows and adding the sums moves 126 of 384 values at
    k = 4).
  - A streamed route's groups hold each expert's picks whole (`expertGroups`), so a chunk's groups are its experts over the
    slots: grouped by the picks' order, a one-slot cache made about one group a pick, each a whole chain over every pick.
  - Calibration hooks (`hy4_model`): `forwardReadout` is `embedStreams`, `engramRowsAfter`, `passLayer` per layer and
    `readoutStreams`, so a layer-major capture runs windows a layer at a time (`releaseLayer` drops a window's layer
    state) and gets window-major's bytes. `pin(layer)` holds a layer's experts in the stream's slots, its picks one
    group. `Model.tap` (`fwd.Tap`) reads each linear's input, the routes and the experts' down inputs, called only
    when set. The layer-major capture's opt-in `calib/head-in.f32` (`SUSHI_HY4_HEAD_ROWS=1`) is `[tokens, 3072]` f32
    little-endian, one row per captured token in window order: the head's own input, rounded never ([quality-kld](quality-kld.md#hy4-capture)).
  - The trunk's BF16 linears multiply as stored (rows rounded to BF16, f32 sums; transformers' BF16 forward and
    llama.cpp's BF16 kernels). The router, the indexer, the head and Engram widen their weights to f32 (`linearF32`):
    the router's and the indexer's picks are not moved by a rounding.
  - A decode row's f32 linear over a BF16 weight is one launch (`hy4_row`): two output rows a simdgroup, two
    simdgroups a threadgroup, each row summed as MLX's `gemv` sums it, so the bytes are the product over the widened
    weight. MLX's tile (four rows a simdgroup, four a threadgroup) ran a router's 512 rows on 128 simdgroups: 9.2 us for
    3.1 MB a launch at 259c0a5a (M5 Ultra, kernel profile).
  - Where MLX takes its bm1 bn8 tile (`K ≥ 16·N`: the indexer's token key [128, 3072] and head weights [16, 3072]),
    `hy4_row` is that tile, the BF16 weight read as stored: the f32 product had widened the whole weight in a cast
    launch first, every token.
  - `fwd.products` names the precision: `.bf16` served, `.f32` for the tiny's oracle tests (a BF16 rounding moves its
    random router's picks).
  - On the GPU a BF16 bank's picks run `hy4_expert_mv` (the BF16 bank read as stored, widened in the register,
    `gather_mm`'s bytes) at either product: an f32 capture never widens the stream's slab (196 slots at 120 GiB: ~340 GB
    of casts a token), and each expert's matrix is read once for up to four of its picks. At `.bf16` MLX's CPU gather takes f32 only, so there the BF16 operands multiply widened
    and the outputs round to BF16 as the GPU's do: the same products on both streams, held to an f64 recompute
    rounded where BF16 rounds (`hy4 routed experts at BF16`).
  - Every f32 product of a forward (the head, router, mixes, indexer, Engram, the attention's f32 products) keeps
    f32 tiles on a NAX GPU: `passLayer` and `readoutStreams` run inside the engine's `exactF32`, which takes MLX's
    f32 tiles whatever `MLX_ENABLE_TF32` reads. A Hy4 model is refused on MLX's own GPU stream.

### Norms

- **Formula**: RMSNorm with ε 1e-5 everywhere in the trunk, in f32, cast to BF16 before the weight. The sandwich
  norms sit after the attention output and after the FFN. The Engram norms take the fp32 machine ε (1.19e-7), as
  llama.cpp's do.
- **Cooler**: the engine's RMS norm, reused as is. On a GPU stream `fwd.normAs` is one launch (`hy4_norm.rows`, MLX's
  single-row norm): the BF16 weight widened in the register, the rows read through their strides, the combine norm's
  `R + shared` added in it, and the latent written at the cache's BF16 (a calibration tap keeps it f32). The weight's
  cast, the add and the cast after were a dispatch each. Seam `fwd.norm_kernel_on`.

### LM head in f32

- **Formula**: `logits = W_lm · RMSNorm(x_head)`, with both operands in f32 (`enable_lm_head_fp32`). The weight is
  stored BF16 and widened.
- **Pinned by**: transformers `HYV4FlashForConditionalGeneration` (`lm_head` in `_keep_in_fp32_modules_strict`);
  llama.cpp `hy_v4.py` keeps `output.weight` F32.
- **Cooler**: DeepSeek's f32 head. The composed forward keeps the head BF16 (0.74 GB) and widens it a 64 MB block of
  rows at a time per forward (`linearF32`), for the readout's row only.

### Drafter: DFly2

- **On** whenever the load's MTP decision (`mtpChoiceFor`) is on and the pack ships DFly2's layers (`hy4_load.shipsHead`),
  resident or streamed; the default on a pack without them resolves off (`[mtp] off (the pack ships no drafting head;
  default)`), an asked-for one is refused (`MtpHeadMissing`). One decision (`stream_mtp_head`) is then read by
  the weights filter, the drafter, the bill and the pipeline manifest's scope (`coolerPipelineScope`), so a `--no-mtp`
  manifest never lets a drafting load skip DFly2's warm.
- **What the checkpoint holds** (shard 148's header):
  - 5 layers, each with MLA attention in the sparse layers' shape: 32 heads, `q_a` 1024, its own `kv_a` (576), `kv_b`
    `[14336, 512]`, a sink per head, the gate (8192) and `post_proj_norm`. No indexer.
  - Each layer's dense SwiGLU MLP is 3072 → 9216 → 3072.
  - Around them:
    - `mtp_fusion`: `enorm`, `hnorm`, `eh_proj` `[3072, 6144]` over the two normed inputs, and a per-channel `gate`
      `[3072]`;
    - `hidden_correction`: a gated MLP, 6144 → 3072 → 3072, over `embed_norm` and `hidden_norm`;
    - `confidence_head`: `[1, 3072]` plus a bias, one scalar a position;
    - `final_norm`.
  - There is no head of its own: the drafter reads the target's f32 head.
  - 0.85B parameters, 1.69 GB BF16.
- **What the README and config say**:
  - Draft layer i shares the target's KV cache and the indexer's selection (up to 2048 positions,
    `dfly_target_index_topk`) at target layer 25, 29, 33, 37 or 41 (`dfly_target_kv_layers`). It keeps no KV of its
    own.
  - 5 draft tokens a round (block 6) from one pass over the block, mask token 120009, probabilistic draft sampling.
  - The fusion runs at slot 0 (`dfly_mtp_fusion: slot0`), with the hidden correction on and no local window
    (`dfly_context_local_window: null`).
  - The README says the drafter is not involved in prefill: the prefill-to-decode handover is the last layer's hidden
    state.
- **Pinned by**: README, `config.json` (`speculative_config`, `dfly_*`), the index, and the drafter's training code,
  [Tencent/AngelSpec](https://github.com/Tencent/AngelSpec) at 0505773a (`DFly2DraftModel`: DSpark with no Markov
  head; `angelspec/config/dfly2_hy_v4_flash_draft_config.json`). transformers ignores the keys
  (`_keys_to_ignore_on_load_unexpected`). llama.cpp#32517 drives a one-layer `model.mtp_layers.*` block instead,
  which the released checkpoint does not carry.
- **The forward** (`hy4_dfly.Drafter.forwardBlock`), as AngelSpec runs it:
  - The block is `[x_a, mask × 5]` at positions `pos .. pos + 5`. `x_a` is the pending token, and slot k predicts
    `x_{a+k}`.
  - Slot 0's input is `E(x_a) + gate ⊙ eh_proj([enorm(E(x_a)) ; hnorm(final_norm(h))])` (`MTPFusion`, `slot0`).
    `h` is the target's last hidden state, the head's collapse of the streams, put through the target's own final
    norm. The mask slots enter as `E(mask)` alone.
  - Each layer is MLA over the target's shared rows (`kv_a_layernorm` latent and roped `k_pe`, as cached) joined with
    the block's own `kv_a` rows, unrounded. The block is causal within itself (trained with `dflash_causal_attn`).
  - Then the drafter's `final_norm`. Then the hidden correction, a chain over the slots (`_served_draft_chain`): `h'_j =
    h_j + down(silu(gate·z) ⊙ up·z)` with `z = [hidden_norm(h_j) ; embed_norm(E(prev_j))]`. `prev` is `x_a` for
    slot 1 and slot j-1's draft (its argmax) after that.
  - Each slot's logits are the target's head over `h'_j`, with no other norm. Its confidence is `σ(proj(h'_j))`.
  - The chain is 5 head passes and 5 host reads a draft, one after another.
- **Parity**: proven against AngelSpec's own modules in f32 on CPU, on the tiny's drafter weights over the target
  state the engine hands over: each slot's logits 6.6e-7 of their peak, the drafts exact, the confidence 5.4e-7 off.
  The test is retired: a drafter changes speed, never text, so `hy4 dfly: greedy rounds` holds the output and the
  bench's decode holds acceptance. Before 1z the wiring was guessed from the README and tensors: the fusion
  replaced `E(x_a)` and skipped the target's final norm, the correction ran on every input row from the target's `h`
  with the embedding first, the block was fully visible and the confidence read the uncorrected hidden. On the tiny
  that left the drafts wrong, the logits 1.28 of peak off and the confidence 0.46 off.
- **Cooler**: in 1.8.0.
  - On by default, as every arch's drafter is (`--no-mtp` or the model's `mtp` setting turns it off): served on the
    release it decodes about 5.5% faster than plain over greedy code and prose replies, winning on each (M5 Ultra, STQ
    4bpw, 512-token replies, 06fddc60 on 3083cdea). With the guessed wiring before 1z a round accepted 2.3-2.7% of its
    drafts, so it stayed off until then.
  - Acceptance on a real pack: `SUSHI_HY4_DFLY_PROBE` ([parity](#dfly2-parity)).
  - The forward is new arch code.
  - The rounds are DeepSeek's DSpark rounds, which already draft a block `[t, noise…]` in one pass on Sushi Cooler,
    with verify, rollback and the adaptive row policy. Here the noise rows become mask rows and the block is 6.
  - The drafter adds no KV bytes: its attention reads the target's rows and picks of the shared layers. Its block
    rows live for one round. Past the indexer's threshold a draft layer reads its target layer's selection for the
    last row the target ran (`Model.last_picks`, taken by every forward when DFly2 is loaded), never every row.
  - How it is checked: [parity](#dfly2-parity).

### Vision tower and merger

- **Served load**: the model's state takes the pack's tensors from the loader's map but the tower's (`vit.*`, census
  role `.vision`), which stay there for the shared loader's vision encoder (`model_load` builds it after `arch.init`).

- **Patches**: a Conv2d 16 × 16 (3 → 1152, with bias). Pixels are rescaled by 1/255, then normalized with mean 0.5 and
  std 0.5.
- **Positions**: a learned table of 128 × 128 rows (row 0, a class token, is unused), resampled bilinearly to the
  grid with scale `(n + 0.1)/128` and `align_corners` false.
- **Blocks**: 27 pre-norm blocks: LayerNorm with bias, MHA 16 × 72 with biases and no rotary (the config's
  `rope_2d_*` keys are read by neither reference), and an MLP 1152 → 4304 → 1152 with tanh GELU.
- **Merger**:
  - `RMSNorm` (1152) → Conv2d 1152 → 4608, kernel and stride 2 → GELU (tanh) → Conv2d 1 × 1, 4608 → 3072.
  - Then the `image_newline` vector is appended at the end of every merged row. An image of H × W patches gives
    `(H/2)·(W/2 + 1)` rows.
- **Prompt**: `<｜hy_image▁start:opensource｜>`, then that many `<｜hy_image▁pad:opensource｜>` (120120), then
  `<｜hy_image▁end:opensource｜>`.
- **Processor**: the release's `processor_config.json`: smart resize to whole 32-pixel cells within 4,096 to 4,194,304
  pixels, bicubic (transformers forces it over the config's LANCZOS), rescale by 1/255, mean and std 0.5. Patches run
  in raster order, each one's values (C, P, P). The engine caps the area at `ENGINE_MAX_PIXELS` (1536²), as on Qwen.
- **Pinned by**: transformers `HYV4FlashVisionPatchEmbed`, `HYV4FlashVisionBlock`, `HYV4FlashVisionPatchMerger`,
  `get_vision_position_ids`, `HYV4FlashProcessor._num_frame_tokens`; llama.cpp `mtmd_graph_hy_v4_flash::build` and
  the HunyuanVL position resampling it shares.
- **Cooler**:
  - Qwen's tower has the same widths (27 × 1152, 4304, 16 heads, patch 16) and runs on Sushi Cooler
    ([qwen4-arch](qwen4-arch.md#vision-tower)). It is reused with parameters: a Conv2d patch instead of Conv3d, the
    position-table rule, LayerNorm with bias, no 2-D rotary.
  - Hy4's tower is served as stored: MLX affine 8 (0.63 GB) in the shipped STQ pack (owner's ruling), written by
    `sushi convert vision-affine8` and stamped `vision_quant` ([pack-format](pack-format.md#hy4-vision-affine8)); the
    original checkpoint's BF16 tower (0.93 GB) runs on the internal path. Nothing is quantized at load, unlike Qwen's tower.
  - Affine parity on the tiny (f32 products): each stage within twice the unit-gain quadrature `sqrt(Σ ε_l²)` of its
    upstream matrices' weight errors (ε about 6.2e-3 a matrix at group 32): the merged rows 1.67e-2 against a model of
    2.31e-2. The tiny's random trunk is far from unit gain, so the image prompt's logits are held to what Gaussian
    noise at the rows' bar moves them (0.225 against 0.385). The codes' product equals the tower over their f32 values
    within the 2e-5 tower bar.
  - The merger is new arch code. Both GELUs are the tanh form, as llama.cpp's encoder runs them.
  - Video is refused by name in 1.8.0.
- **Served** (`hy4_media.zig`, 2a): the slot claims a config with a tower; a load asked for vision keeps its 441
  tensors (0.93 GB BF16, 0.63 GB at affine 8 with its grids; billed in the streamed split as stored). Attention scores
  512 query rows at a time and each stage runs in a scope of its own, so an encode holds the stream and one stage's
  buffers. `encodeHeldBytes` bills it by replaying the encode's buffers per pool size class, an affine product with its
  split-K partials (`quantized.partialsBytesHere`; a pass frees them only when handed over); a test holds the measured
  held peak under it and within 5% of it, BF16 and affine. The rows
  splice inside the model's pass, `offset` rows on from earlier chunks; an image id with no row left is a plain token.
  On the static bill an encode is an execution on the arena writing the request's `Media`
  ([bill_redesign](bill_redesign.md), stage 4c).
  The tower warms at load, before the seal (`Tower.warmEngine`, as Qwen's and DeepSeek's do): one block at seven patch
  counts and the attention over 8,192 keys by shape alone (the looped softmax past 4,096 keys), so an image of any grid
  compiles nothing in its request. The splice takes the embedding's and the tower's rows through one f32 `take` by
  uint32 picks, which no text prompt makes: `warmKernels` splices one row at load, so an STQ pack (its embedding a GPU
  gather) compiles nothing at the first image (`hy4 late` image cells, `hy4 vision` tower cells at the release's
  widths, BF16 and affine 8).
  Processor parity: `--processor` dumps one 70 × 45 image through transformers' PIL processor, held to one ulp.

### Chat template and special tokens

- **Turns**: `<｜hy_start:opensource｜>role<｜hy_middle:opensource｜>…<｜hy_end:opensource｜>`. BOS 120000 is
  `hy_start`; EOS 120025 is `hy_end`; pad is 120002.
- **System turn**: a leading system turn always carries `<｜reasoning_mode:opensource｜>reasoning_effort:<e>` and,
  with tools, the tools prompt. The template synthesizes one when there is none.
- **Effort words**: the template takes `no_think`, `high` (its default) and `max`. The generation prompt ends in
  `<think:opensource>`, or `<think:opensource></think:opensource>` for `no_think`.
- **Sushi's effort words** (owner's ruling):

  | sushi | template | thinking budget |
  |---|---|---|
  | `off` | `no_think` | none |
  | `low` | `high` | 2,048 tokens |
  | `medium` | `high` | 16,384 tokens |
  | `high` | `high` | uncapped |
  | `xhigh` | `max` | uncapped |

  `minimal` takes `low`'s row, `on` takes `high`'s and `max` takes `xhigh`'s. The budgets are the effort table's caps
  (`effort:hy_v4_flash`), which an operator may change as on every arch.
- **Tool calls**:
  `<tool_calls:opensource><tool_call:opensource>NAME<arg_key:opensource>K</arg_key:opensource><arg_value:opensource>V</arg_value:opensource>…</tool_call:opensource></tool_calls:opensource>`.
  Non-string values are written as JSON.
- **Tool results**: one `tool` turn of `<tool_response:opensource>…</tool_response:opensource>` items, ordered by the
  assistant's call ids. Earlier reasoning is kept by default when tools are present (`preserved_thinking`).
- **How Sushi reads the output**: the think and call markers are normal added tokens. The tokenizer reads
  `<think:opensource>`, `</think:opensource>`, `<tool_call:opensource>`, `<arg_key:opensource>`,
  `<arg_value:opensource>` and their closers as GLM's plain tags where the pack has no plain one: they decode as
  `<think>`, `<tool_call>`, …, and `</think>` encodes to Hy4's closer. The reasoning split, the stream's think gate and
  a budget's closer then work as on every `<think>` family, and the calls parse with GLM's arg_key parser
  (`tool_calls` drops the `<tool_calls:opensource>` wrapper; the reader, the stream's progress beats and the declines of a cut call read the wrapper and the `:opensource` spelling too, [tool-calling](server-tool-calling.md)). Sampling defaults to the release's 0.9 / 1.0 when a pack
  ships no `generation_config.json`.
- **Sampling defaults**: temperature 0.9, top_p 1.0.
- **Cooler**: `hy4_chat.zig` holds the face. The template renders in the Zig Jinja engine (it uses
  `str.format`, `namespace`, macros and `tojson(ensure_ascii=False)`). Tool calls parse through GLM's `arg_key`
  family, given the `:opensource` spelling and the `<tool_calls:opensource>` wrapper.

<a id="where-the-references-differ"></a>
### Where the references differ

| item | transformers | llama.cpp#32517 (followed) |
|---|---|---|
| Engram FP8 round trip | applied per row | not in the graph: a row is read as stored. AngelSlim's GGUF stores the table as `GGML_TYPE_F8_E4M3` (type 43, `per_layer_token_embd.weight` `[96, 660006416]`), one byte an element after a 192-byte header, no scale: a lossy storage choice of that GGUF, not the forward |
| Engram norm ε | `nn.RMSNorm` with ε unset, which torch resolves from the input dtype | fp32 machine ε (1.19e-7) |
| Engram history across an image | the image placeholder id is hashed like any token | an image position is dead and reads as the pad id |
| Engram ids at or past 120,818 | read the pad id's entry | identity |
| Rope for image positions | interleaved 3-D M-RoPE | `NORM` with one position plane |
| Vision encoder GELU | `ACT2FN["gelu"]` (erf) | tanh (`bias_activation_fusion`) |
| Drafter | DFly2 keys ignored | a one-layer MTP block the release does not carry; not followed: DFly2 is the checkpoint's own ([DFly2](#dfly2-parity)) |
| Config spelling | `deepseek_sparse_attention` remapped to `indexed_attention` | the converter accepts only `sparse_attention`, registers `HYV4NextForCausalLM` and needs `engram_enable` and `engram_vocab_map.npz`: it does not convert the released checkpoint as published; not math, so nothing to follow |

<a id="decode-on-cooler"></a>
## Decode on Sushi Cooler: the trunk's launches

- **The K6 trunk GEMVs read near the box's rate; their dispatches were the gap.** `sushi-ubench run shared.exl3_trunk_hy4`
  times each trunk launch at the release's shapes: the bare exact launch, `hy4_stq.apply` as it was (`hy4`: the f32 rows
  cast to bf16, the launch, its rows cast back), as it is (`thru`), and each with the f32 norm after it (`hy4_norm`,
  `thru_norm`).
- **A trunk linear rounds its rows and outputs inside its launch** (`trunk_stq.matmulThruBf16`,
  [engine-exl3-experts](engine-exl3-experts.md#dense-linears)): no cast dispatch either side, the same bytes. Past the
  multi-row arm (a prefill chunk) the call is the casts around the GEMM, as before. `fwd.linear` takes that launch for
  a bank; only a caller that holds the rows narrow (the gate, `linearStored`) takes the BF16 launch and its cast in.
- **q_a, o and the dense layer's down leave rms-normed by their finish** (`fwd.linearNormed`): the finish, the norm
  weight's widening and the norm were three dispatches; they are one, with the bytes of the f32 norm after the launch.
- **The attention's write-back rides the FFN mix's launch** (`fwd.expandMixNorm`, `hck.mixExpandIdentity`, kernel
  `hc_ident_mix_expand`): the mix kernel computes `post·y + h` as it reads the stream, the expand's two roundings, and
  its x-0 tiles store it, so a layer's iHC is five launches, not six. A resident STQ MoE's
  FFN write-back rides its combine norm's launch (`fwd.moeFinishBack`, `hy4_norm.rowsExpand`: the norm's value, then
  `post_k · y + h_k` as `hc_ident_expand` computes it), so `passLayer` gets the streams; the dense layer's, the BF16
  paths' and a layer-major run's still run alone.
- **A decode block's iHC mix and its collapse and norm are one launch** (`hy4_hc`, seam `fwd.mix_norm_on`), one
  threadgroup a row: the tiled mix's arithmetic over all `2·HC` mixes at once (the same `hc_tg` lanes, terms and
  sums), then the identity split, collapse and rms norm from the mixes in threadgroup memory; with the attention's
  write-back, the stream it expands as it reads, stored as `m`. A layer's iHC is two launches; a prefill keeps the
  tiled mix.
- **A layer's combine norm and write-back run the next layer's attention mix and norm** (`hy4_hc.backMixNorm`, seam
  `fwd.back_mix_on`): the norm of `R + shared` as `hy4_norm.rowsExpand` reduces it (MLX's single-row tree, its 768
  threads run as virtual lanes of the 1,024), held in threadgroup memory, then the expanding mix reads `post·y + h`
  from it. `passLayer` hands the next layer's mix across (`Passed.next`) where no Engram delta comes between: 37 of
  the release's 41 MoE layers, one launch and one barrier each a token.
- **A decode block's iHC mix finishes and norms W_o** (`fwd.expandMixNormParts`, `hy4_hc.mixExpandNormParts`, seam
  `fwd.o_parts_on`, no tap): W_o's launch leaves its k-split partials (16 at 8,192 wide, 32 at 16,384) and the mix's
  threadgroup finishes them into `t_y` and norms them by `post_proj_norm` as `rows_finish_rms_thru` does (a block a
  simdgroup, fused squares, `simd_sum`s of the blocks), so W_o's finish_rms launch is none. Its one threadgroup then
  reads 196 to 393 KB of partials ahead of its 393 KB of `hc_fn`.
- **On the engine's stream a row's iHC mix spreads over `2·HC` threadgroups in that one launch** (`hy4_hc` wide forms,
  `kernel.launchCounted`, seam `hy4_hc.wide_on`): each simdgroup sums one (virtual simdgroup, mix) pair of the
  one-threadgroup launch, the same lanes, terms and `simd_sum`, into the engine's counter page; the row's last threadgroup
  to count itself in reads the sums in the same order and runs the split, collapse and norm. Every threadgroup runs the
  form's first part itself (the combine norm, W_o's finish), and loads the collapse's stream values before it counts. The
  bytes and the launch count are the one-threadgroup launch's, at one row and at a verify block's 2..8 (a grid of rows
  takes an even count). Each wide pipeline (form, template values, input dtypes) holds its own grant: one a GPU grants
  fewer than `hc_tg` threads keeps one threadgroup a row for the process, as MLX's stream does, and the others stay
  wide. A one-threadgroup pipeline holds its own threads a group: `hc_tg`, halved until the GPU grants it (an M1 grants
  the fused forms 640 or 384), latched for the process, and the load lowers `hc_tg` to the widest halving every fused
  form is granted (`hy4_hc.probeForms`, after `probeIdentity`), so the tiled and fused kernels run at one width and agree. A served launch latches its pipeline's grant; before a dry recording, `Model.probeWide` latches
  every grant the forward needs (a dry census, then a launch of each). Chained in one process (M5 Ultra, 4 streams of 3,072, GPU busy a launch, not quiet,
  no lock), base 01871ac7 against this change at one row: the plain mix 15.5 to 10.2 us, `back` 14.6 to 10.2,
  `expand_parts` 20.4 to 17.1 (W_o split 32) and 17.2 to 12.9 (split 16); every threadgroup reading all of W_o's
  partials is what keeps `expand_parts` highest. A micro indicator only: the forward's A/B decides.
- **On the engine's stream W_o of up to three rows finishes its own partials** (`hy4_stq.applyFinished`,
  `trunk.matmulThruBf16Counted`, `fwd.expandMixNormRows`, form `expand_rows`, seam `fwd.o_rows_on`, `fwd.o_rows_max`):
  its split threadgroups store their partials by relaxed device atomics and count in at their 128-wide block's counter;
  the block's last finishes it as `rows_finish_thru` does (a thread a column's parts in order, a simdgroup a row's H128),
  so each of the mix's threadgroups reads one finished row and norms it as `rows_finish_rms_thru` does. Two launches a
  site as before, the same bytes at 1..8 rows. MLX's stream and 4..8 rows keep the partials: the finisher's tail grows
  with its rows and past three outweighs what the mix no longer re-reads.
  - Chained in one process (M5 Ultra, W_o and the mix a step, GPU busy a step, the fastest of 8 rounds A B B A, not
    quiet, no lock), base f9b57c55 against this change, us at 8,192 / 16,384 wide: one row 40.6 to 39.2 / 55.1 to 48.7;
    three 37.7 to 38.1 / 54.1 to 50.7; eight 40.8 to 45.7 / 56.7 to 62.7. Of it, W_o alone at one row 22.3 to 24.3 /
    33.3 to 36.2 (its count and fences 0.3-0.5) and the mix 16.4 to 13.1 / 23.1 to 14.0. A micro indicator only: the
    forward's A/B decides.
  - Ruled out: the finisher also leaving each block's sum of squares, so the mix skips its squares (no gain in the
    mix); plain partial stores (no faster than the atomics).
- **On the engine's stream every trunk launch of up to three rows finishes its own partials** (`trunk.matmulThruBf16`,
  `matmulThruBf16Normed`, `matmulThruBf16Stored`, `linear.linearCounted`, `project_rows.launchRowsCounted`): q_b,
  the gate (its f32 rows read through bf16, its rows stored bf16, so its cast goes too), the shared expert's and the latents' linears, the dense layer's FFN, and at one row q_a, the dense down and W_o
  (where a tap reads it) through their norm, each as W_o's counted launch: the block's last threadgroup finishes it as
  `rows_finish_thru` stores it (`rows_finish` for the gate), and a normed launch's last block to finish norms every row
  as `rows_finish_rms_thru` does. One launch a site where there were two (three at the gate): 376 fewer a release decode
  token (42 layers' q_a, q_b and the gate's two, 41 MoE layers' five, the dense layer's three), 333 a 2- or 3-row block.
  The head's 944 blocks are past the page's 256 counters and keep its finish; the tiny's trunk folds at its 128 width but
  for W_o and the dense down. Counted launches run in the order made, so the MoE makes its shared expert before
  W_lat_up, beside the routed experts.
- **A SwiGLU's gate and up of up to three rows are one launch** (`fwd.swigluAct`, `hy4_stq.applySwiglu`,
  `trunk.matmulThruBf16Swiglu`, `project_rows.launchRowsPair`): over the one input, grid z picks the bank, each bank's
  threadgroups run its own launch's sums at one bank's split, and a 128-wide block's last of both banks' threadgroups
  to count in finishes the gate's block then the up's as `rows_finish_thru` does and stores `hy4_chain.swiglu`'s
  `(g · σ(g)) · u`. The shared expert is two launches a layer where it was four with counted finishes (seven before),
  the dense layer's SwiGLU likewise: 84 fewer a release decode or 2..3-row token. The bill holds both banks' partials in
  one buffer (`pairScratch`).
- **An STQ MoE's latent leaves W_lat_down as stored, BF16** (`fwd.expertLatent`, no tap): up to three rows its launch
  reads the f32 rows through bf16 and finishes as `rows_finish` does, so the trellis experts' cast before the routed
  GEMV goes (41 fewer a release token); a tap reads the f32 latent as before. Chained in one process (as above),
  f3ef43fd's self-finished f32 latent and its cast against the stored latent, us at one / two / three rows: 12.47 to
  10.38 / 13.22 to 11.28 / 14.43 to 12.47.
  - Chained in one process (as above), f9f367b4's gate and up each finishing itself and `hy4_chain.swiglu` against the
    pair, us at one / two / three rows: the shared expert's 3,072 to 1,536 20.59 to 14.45 / 48.71 to 35.52 / 54.00 to
    40.68; the dense layer's 3,072 to 13,824 60.63 to 50.52 / 62.65 to 52.18 / 64.68 to 53.83. A micro indicator only.
  - Chained in one process (M5 Ultra, a site's launches and a one-launch link a step, GPU busy a step, the fastest of 8
    rounds A B B A, not quiet, no lock), base 712418c7 against this change, us at one / two / three rows: q_b 1,024 to
    8,192 13.49 to 11.85 / 13.43 to 12.51 / 14.34 to 14.49; the gate 3,072 to 8,192 24.94 to 21.98 / 25.19 to 21.88 /
    25.49 to 23.57, to 16,384 36.52 to 32.59 / 37.92 to 35.07 / 38.43 to 36.63; 3,072 to 1,536 19.24 to 16.04 / 27.68
    to 25.23 / 27.62 to 28.03; 1,536 to 3,072 26.80 to 22.77 / 26.16 to 24.74 / 26.26 to 27.33; q_a through its norm at
    one row 12.78 to 12.60, the dense down 36.88 to 35.96. A normed launch past one row lost (its last block norms
    every row alone with 128 threads): q_a 12.74 to 14.63 at two rows, W_o 42.51 to 57.98 and the dense down 37.93 to
    46.39 at three. A micro indicator only: the forward's A/B decides.
- **A verify block's router is one launch** (`hy4_route.pick` at 1..8 rows, a threadgroup a row): at 2d8a204b
  (M5 Ultra, fwd-ubench @4K) a 3-row verify 26.54 to 25.26 ms, 6 rows 37.66 to 36.01; one row unchanged.
- **A verify block's attention keys are built once** (`hy4_chain.ringKeys`: every row's rotated sliding window in one
  launch; rows' products written into one result by `matmulInto`): at 02ec20eb (M5 Ultra, fwd-ubench @4K) a 3-row
  verify 21.70 to 21.24 ms, 6 rows 31.18 to 29.86; one row unchanged.
- **Every 2..8-row linear is per-row exact** (BF16 `linearStoredAs` and the indexer pool through `isVerifyBlock`):
  a BF16 `W_kva` at 6..8 rows takes MLX's wide GEMV otherwise. Cost at 461af929 (M5 Ultra, fwd-ubench @4K): a 3-row
  verify 21.26 to 21.50 ms, 6 rows 29.90 to 30.80. A load's warm makes the STQ head's 2..8-row pipelines.
- **The 2..8-row rule is one predicate, by the span's own width** (`fwd.isVerifyBlock`): every site that must give a
  verify row its one-row bytes (row linears, absorb and value-out with or without `hy4_absorb`, scores, the ring)
  asks it, and a chunk's short last span decides by its own rows, so the shared launch keeps each span's bytes.
- **A layer's new kv8 latent rows are quantized into the store in one launch** (`hy4_latent.append`: MLX's affine
  quantize arithmetic, written in place through `kernel.launchInPlace`, rings wrapped): at 01871ac7 (M5 Ultra,
  fwd-ubench @4K) 16.019 to 15.982 ms a token, 168 dispatches fewer; a 3-row verify unchanged within 0.05 ms.
- **A layer's stored kv8 latent is read to f32 in one launch** (`hy4_latent.range` for a span, `gather` for the sparse
  layers' picks; kernel `sushi_hy4_latent_read`): MLX's affine dequantize arithmetic at bits 8 rounded to bf16, then
  widened, the bytes of the take, dequantize and cast it replaces. At 2cacb6fb (M5 Ultra, fwd-ubench @4K) 16.09 to
  15.88 ms a token, 78 dispatches fewer. A kv16 store keeps its cast.
- **A verify block of 2 to 8 rows takes the one-row f32 kernels** (`hy4_row.rowF32`, `hy4_absorb.queryLatent` and
  `valueOut` at `max_rows` 8): one launch reads each BF16 weight once and gives every row the bytes it has alone, so
  no layer casts `kv_b` or a router/Engram/indexer weight to f32 for a verify. At f555da15 (M5 Ultra, fwd-ubench
  @4K) a 2-row forward 28.9 to 21.6 ms, 3 rows 31.2 to 24.0, 6 rows 38.2 to 31.8; one row unchanged in bytes.
- **A decode or verify block's attention scale, mask, sink and softmax are one launch** (`hy4_chain.sinkSoftmaxCausal`,
  `sinkSoftmaxMask`; kernels `sushi_hy4_sink_softmax_causal` and `_mask`): the two score products' sum, the scale, the
  causal/window or selection mask, the sink column and MLX's `softmax_single_row` order (`fp contract(off)`), the bytes
  of the eight launches it replaces, for key counts under 4,096. At 144af704 (M5 Ultra, fwd-ubench @4K) 16.68 to
  16.25 ms a token, 240 dispatches fewer.
- **Hy4's own kernels keep their launch configs** (`hy4_kcfg.Cache`, keyed by the launch's shape): a fresh config is
  about ten FFI calls a launch, and the engine finds the pipeline a kept config last named. A constant fixed by the
  geometry (ε, log2 θ, the rope step, the router's scale) rides as a template int of its f32 bits, not a host array
  a launch; only a position is made per launch.

## The state and the prefix cache

- **Per token**: each sparse layer keeps its latent row and, per closed group, its pooled index key (128 f32, unrotated:
  the indexer rotates it at its group's position when it scores). Token keys of closed groups are not kept: a group's
  pooled key is final once its fourth token arrives (`hy4_model.poolClosed`). The pooled keys stay f32, as the oracle
  pools them: they decide the selection, which a rounding can flip.
  - With the default 8-bit affine latent (672 B a row, [below](#kv-width)): 12 × 672 + 12 × 128 = **9,600 B a token**,
    10.1 GB at 1,048,576 tokens.
  - With the BF16 latent (576 × 2 = 1,152 B a row): 12 × 1,152 + 12 × 128 = **15,360 B a token**, 16.1 GB at
    1,048,576 tokens.
- **The rows grow in place** (`row_store.Store`, shared with DeepSeek's compressed rows): a sparse layer's latent
  (codes, scales and biases at 8 bits, the BF16 rows at 16) and rope rows are the parts of one store, its pooled keys
  another. An append writes only its new rows (`slice_update`, in place while the buffer has one holder); a store that
  outgrows its buffers copies its rows once into buffers a sixteenth larger (`hy4_model.rows_grow_div`, 64 rows at
  least). A read still pending when an append writes holds the buffer, so that write goes to a copy and the read sees
  the rows it was taken over (`hy4 state:` cells).
- **Per request, fixed**:
  - 30 sliding layers' rings of 512 latent rows (above): 10,321,920 B at 8 bits, 17,694,720 B at BF16;
  - each sparse layer's open group: up to 3 token keys of 128 f32, 18,432 B in all;
  - the Engram history: the last 2 token ids.
- **A point within the running state is its length and its small state** (`Model.Point`), as DeepSeek's live points
  are ([dsv41-arch](dsv41-arch.md#prefix-cache)): the position and a `Points` slot (`Model.point_fixed`,
  `point_hidden`) into which each sliding layer's ring, each sparse layer's open block (whole, its rows a count) and a
  drafter's hidden state are copied in place, as a mark's are. The rows below the position are the stores' own: they
  only grow past it, so a rollback (`rollbackPoint`) cuts the stores back to it, writes the slot back into the state's
  blocks and cuts the Engram history. A rollback lets go of the points past it (the next append writes over their
  rows), and a reset (`Model.epoch`) of every slot.
  - The slots are `Model.point_slots` (12): the request's `live_point_cap` and the record's `record_point_cap`, made
    with the request's state (`reserveFor`; at a slot's first take where no request reserved one). A point names the
    take that holds its slot, so one let go by a reset or a newer take is not let go twice; a take with every slot
    held is `Hy4PointsFull`. Taking a point and rolling back to one ask nothing of the pool.
  - A verify block's mark (`mark`, `rollbackTo`, `dropMark`) keeps the same small state in the model's mark blocks,
    reserved with a drafting request's state and copied whole in place, so it asks for nothing; one stands at a time.
    `rollbackTo` runs the kept rows again through `hy4_plan.run`, on the arena where the load planned that replay.
  - A DSpark round's finish keeps its accepted rows without running them again where the verify journaled them
    (`Model.keepRows`, taken by `Hooks.finish` when `keepRowsApplies`): a 2..8-row verify of a drafting load writes
    each row's hc streams and each sparse layer's token keys into fixed blocks made with the request's state
    (`rows_streams`, `key_journal`, billed in `pickBytes`); the finish cuts the sparse rows and pooled keys, rebuilds
    the open group from the journal, restores the rejected ring slots from the mark, clamps the kept row's picks as
    the replay does and runs row keep-1's one-row head for the handover, all in a wide phase. Its state is the
    replay's byte for byte (`hy4 verify: a 6-row verify block's first rows kept in place ...`). A round after a
    rollback: about 17 to 4.3 ms; served DFly2 rounds 65.6 to 84.3 tok/s code, 64.4 to 79.2 prose, same text
    (b6ad2b2f on 8dc89a86, M5 Ultra, STQ 4bpw, kv8, round meter, one arm each).
  - The prefill keeps one live point at each checkpoint and at the prompt's end, the newest `Model.live_point_cap`
    (4); a finished request hands them to the cache's live record (`hy4_prefix.Live`), which keeps the conversation's
    newest `record_point_cap` (8). A next turn continues the state where it stands, or rolls it back to the deepest
    point within the shared prefix, unless a RAM entry or the SSD tier holds a deeper one; the record's points at or
    below where it lands stay for the turn after.
  - At release geometry a point is 10.34 MB at kv8 (the rings) and 17.71 MB at BF16: the 12 slots stay
    under the RAM tier's 250 MB cap whatever the context (`hy4_ledger.pointsBytes`, `hy4_prefix.ram_tier_max_bytes`).
- **A stored restore point is the whole state at one position** (`hy4_prefix.capture`): the small state and views of
  every row (`Model.capturePoint`), four snapshot entries a layer (latent and rope rows, scales and biases, pooled keys,
  open group), and the latent width it holds; the Engram history is rebuilt from the prompt ids. Its views hold the
  stores, so it never outlives the next forward: only a write-through takes it (`captureCheckpoint`'s `to_ssd`), and
  any other is its position alone, its live point serving. One taken where a live point is (`captureKept`) shares that
  point's slot; its absent arrays share the model's placeholder, made at load (`Model.placeholder`).
  - A restore of a stored point (a RAM entry's, or the SSD tier's as read back) installs into the request's own
    state (`restoreArrays`): the state emptied in place, its stores reserved for the point's rows or the state's reach,
    the larger, the rows written into them and the rings and open blocks written in place; it asks nothing of the pool
    where the state is reserved. The tier's read of the point still makes arrays of its own (`kv_disk_cache`).
- **The SSD tier stores Hy4 as DeepSeek** (`setDiskGrownSpec`, namespace `hy4-state-v2`, `DiskTier.points_hold_state`
  since the trunk cache holds no KV rows): a point's file holds its small state and the rows past its last whole span,
  and an entry's rows go once into segments of `row_span` (8,192) tokens, which an heir links by hard link below the
  position it restored or rolled back to (`DiskTier.rows_base`, `Live.base`).
  - The write-through (`writeThrough`, `always_write_through`) stores the prefill's points on that grid and releases
    every point at once, the writer having copied what it stages; a point off the grid is its live point only.
  - The live record's points reach SSD where its state is about to be lost (`flushLive`): a lookup the record cannot
    serve (another conversation, a deeper stored point), the handover, and the scheduler's flush (unload, 300 s idle,
    shutdown, `hy4_sched.flushResident`). A finished request keeps one point where its state stands, behind its reply,
    so a turn restored after a handover or a restart prefills only what is new.
  - A turn so writes its new rows once and each point's small state (10.34 MB at kv8 on the release), plus the rows
    past the last grid line in each off-grid point it flushes. The RAM and SSD matches compare an entry's KV width with
    the request's, so a point never crosses widths.
- **The live record is keyed by what its rows depend on**, as DeepSeek's is: the model's `epoch` (its points stand on
  the model's rows), its `version` (it moves on every layer append, so only an untouched state continues in place),
  the tools, and the media key and start (the next turn is held to the text before the first media row, placeholder
  ids being ordinary vocab). A forward that fails mid-layer resets the model; a restore of a stored point keeps that
  point alone as the request's own.

<a id="kv-width"></a>
**KV width: an 8-bit affine latent by default, BF16 also served** (owner's ruling, 2026-11-18: 8-bit affine KV on
Sushi Cooler for both Qwen and Hy4).
- **The format**: the engine's KV default, `KVQuantConfig.engine_default` (affine, 8 bits, group 64), the one Qwen's
  Cooler loads take: per latent row, 512 codes with a BF16 scale and bias per 64 (544 B, as GLM's kv8 latent), and the
  64 rope values kept BF16 (128 B). That is **672 B a row**, against 1,152 at BF16.
- **What it covers**: the latent rows of the sparse layers and the sliding rings. The pooled index keys stay f32, as
  the oracle pools them: they decide the selection, as GLM keeps its pooled index lossless.
- **The widths are served** as `--kv-quant 8` (the default) and `--kv-quant 16`, as on GLM's latent.
- **Error budget**: a value is held to half its group's step, `(max − min) / 255 / 2`. The error reaches K (the 512
  latent values) and V (the same values) of every layer; the rope values, the indexer and the pooling are exact. It is
  read as the KLD of the 8-bit latent against BF16 KV (the 8-category 16x512 fixture, the BF16 checkpoint as teacher),
  which goes to the owner beside the decode speed it buys (part 1y).

## The memory bill

Hy4-Flash is on the static bill ([bill_redesign](bill_redesign.md), stage 5): its `bill` slot is `hy4_ledger.zig`, the
state rates below and `ledgerTerms`, and shared code holds no other Hy4 term or hook. The terms:

| term | bytes |
|---|---|
| resident trunk (BF16 checkpoint) | 14,016,920,660 B streamed, every tensor `hy4_load.kept` keeps at its stored bytes (`residentLoadBytes`, pinned against the release's tensor list): sliding attention 8.22 GB, sparse attention 1.70, shared experts 1.16, latent projections 0.77, embedding 0.74, head 0.74 (BF16; its f32 widening is a forward's transient), dense FFN 0.25, the small tensors 0.42; an STQ pack bills its banks' stored bytes |
| expert cache | `--ssd-budget-gb` less the trunk, the selected slab and the bounce. A BF16 expert is 14,155,776 B; a layer's 512 are 7.25 GB, so Hy4 keeps no union workspace: its picks route in groups the cache holds (above). On the M1 (32 GB) at `--ssd-budget-gb 18` that is 4.06 GB, 7 slots a layer |
| Engram row caches | one `RowCache` per module, its rows times the row bytes (192 B BF16, 96 B as plain e4m3), billed with the trunk |
| state per token, per request | 9,600 B a token at the 8-bit affine default, 15,360 B at BF16, and 10,340,352 B / 17,713,152 B of rings and open groups (above). A request holds the state at its stores' most rows (`hy4_ledger.stateBytes`, `capRows`: the rows and a sixteenth, 64 at least), each array a pool buffer at its size class; the ledger's rate counts both slacks (`heldRate`), because a forward submits the state arrays it leaves as graphs (a ring's write, the pooled keys and open group no span read) in one batch with its logits (`Model.submitState`: a submit, no wait), so no graph holds a chunk's inputs |
| restore points | a stored point holds views of the rows and leaves RAM before the next forward (the write-through takes it, or it is its position alone): `hy4_ledger.checkpointCap` is the cap of the writer's copy of the request's state SSD-first, its permit (`disk_writer_permit_bytes`), else 0 (`RequestTerms.copy_cap`, D108). Live points and a mark are small states (`hy4_ledger.pointBytes`): the `Points` slots, the record's 8 and the request's 4, are the request's state (`pointsBytes`), under 250 MB at any context; a drafting load's mark blocks are its state too (`markBytes`). The model's own holding is `hy4_prefix.heldBytes` (each buffer once, past what the entries hold), what the ledger parks at a request's end and `/props` reads (`hy4_sched.modelStateBytes`, `parkedStateBytes`) |
| a load that drafts | with DFly2 on (`arch.dsparkOn`), every request holds, drafting or not, the blocks `reserveFor` makes beside its state (D56): the mark (`markBytes`), the model's selections and hidden state (`pickBytes`, `hiddenBytes`) and each point's hidden state (`pointsBytes`); a speculative round writes up to a verify block's rows past the reply (`sampler_draw.max_rows`). All are in `ledgerTerms`' fixed term |
| forwards | every forward runs on the load's plans (decode, verify, draft, sample) or in the region its load measured (a prefill chunk, an image's encode, a round wider than a verify block; D97, D98), refused by name past it (`WideFull`, `PlanMiss`): the model bill's `arena`, measured, never a request's term. A forward runs under no pool cap (`arch/cooler.zig` `plansForwards`): the pool is its turn's gate (D64). The arena is made when the chunk is fixed, before the warm, so the warm runs what a request runs (D107) |
| the plans' reach | `bills.planReach`: a manual context (`--ctx-size`, `ctx_size` in model-settings.json) caps every request, so the plans record to it, its region measured at no wider than it; one narrower than the load's own served warm records to the warm's reach (`warmReach` at the widest candidate chunk, a verify block past it). Auto context is solved on the ledger after the plans, so it records to `contextCap` |
| on the ledger | every Hy4 load that warms is on its ledger ([engine-memory-admission](engine-memory-admission.md#the-ledger-a-load-on-the-static-bill)); one that does not warm is refused by name (`StaticBillLoadDoesNotWarm`). Its budget is read once at load, its model bill committed at the seal, its context in closed form over `hy4_ledger.ledgerTerms` (the state at a token's held rate, the fixed blocks, the restore points), admission on the ledger's lines at the load's KV width (D103), every reader of a request's bill on one `static_bill.Reservation`. Before the load, the preflight and the streamed fit price a request by the same terms (D105): the arena is the ledger's, chosen with the chunk |
| the prefill chunk | fixed once at load (stage 4b, D4: `bills.fixStaticPrefillChunk`, before the warm and the plans, `ModelConfig.static_prefill_chunk`): the context is priced at `--prefill-chunk-min` (default 512), then the load takes the widest rung (512, 1024, 2048, 4096 or 8192 rows; a bound off them is refused by name, D106) from `--prefill-chunk-max` (default 8192, `hy4_config.widest_chunk`, the forward's 256 gathered spans) down to the min whose arena (measured, D98: the plans recorded before the warm and the region a prefill of that chunk asks, D97) fits beside it; none fits refuses the load with the numbers, and a fixed context (`--ctx-size`, the model's `ctx_size`, a pinned one) that the seal's lines no longer hold (the warm's static and pipelines were unknown to the choice) refuses it there, naming the widest narrower chunk that holds it. Every prefill then runs chunks of exactly that width and one shorter tail (`prefill_plan.chunkEndFor` merges none): the per-request chooser and the adaptive width are off for the load, and the bills read the chunk (`requestPrefillPick`, `requestForwardRows`, `sushiContextWidth`). `--prefill-chunk N` is deprecated here and means `--prefill-chunk-max N` |
| vision encode | a request's `Media` (its rows × `out_hidden` f32, from its grids); the encode itself runs as a wide execution in the load's region (`hy4_encode_plan.run`), which the load measures at the largest grid of each class it accepts (`hy4_encode_plan.need`); a load that plans nothing bills the encode per request as before |

## What a quantized pack needs

Hy4-Flash is served from **STQ packs** only (owner's ruling, 2026-11-21). The **original checkpoint** (its own BF16)
still loads and is never refused, but as an internal path, not a served form: the lossless KLD teacher and the
encoder's capture source, held by its tiny oracle cells. No affine (MLX 4/8-bit) or non-STQ EXL3 path is designed for
it: from 2.0 the engine keeps only STQ packs and original weights. The llama.cpp GGUFs are a parity reference, never a
served format.

The consumer contract is [pack-format](pack-format.md). How packs are made, and every rate and codebook choice,
lives in sashimi, the private converter repo. A Hy4-Flash STQ pack has these parts:
- **Routed experts**: STQ rate groups under the routed nesting, 41 layers × 512 experts, split from the fused
  `gate_up` into gate and up banks. The router rows and the bias follow group order. `swiglu_limit` 10 is a whole
  number, as the clamped chain requires. **Pair-major only** ([pack-format](pack-format.md#tile-order)): the trellises
  are `…g<N>.trellis_pm` (`[E, OT/2, IT, 2, n]`, `expert_quant.tile_order` `pair_major`), and a pack-order pack is
  refused at parse by name (`Exl3TileOrderUnsupported`, the log line naming `sushi repack <pack>`). The DFly2
  drafter's banks are dense trellis banks, not routed experts, so its shard stays in the pack order.
- **Trunk linears**: trellis banks where both sides sit on the 128 grid. Every projection does: 3072, 1024, 512,
  1536, 2048 and 16,384 are all multiples of 128.
- **Kept in their stored precision**: every norm weight; sinks; the hyper-connection `fn`, `base` and `scale` (f32);
  the router bias (f32); the indexer's `key_value_projection`, `key_gate_projection` and `intra_group_ape` (f32,
  pooled in f32 by the reference) and `head_weight_projection`; the Engram biases, `γ`, `σ_c` and lookup norms; the
  `lookup_table` (int32); the vision patch embedding, position table, norms and biases. The tower's matrices are MLX
  affine 8 in the shipped pack ([pack-format](pack-format.md#hy4-vision-affine8)), its 4304-wide MLP down projections
  BF16 by name.
- **The Engram tables**: kept BF16 (lossless, the teacher's), or stored as plain e4m3 codes with no scale, AngelSlim's
  GGUF form (63.4 GB). The second rounds every row, so it is a lossy pack choice judged by KLD against the BF16
  teacher like any other.
- **The head and the embedding**: the head may be a trellis bank like the trunk's, and is computed in f32 either
  way. The embedding stays BF16 (0.74 GB, a row gather).

<a id="seams"></a>
## Seams Sushi Cooler generalizes first

The rule: where Hy4 would need a special case in shared code, a copied DeepSeek file or a flag threaded through, the
seam is generalized first, as its own part, with every existing arch's bytes, bills and query counts unchanged.

<a id="r1"></a>
- **R1. An arch on Sushi Cooler with no opt-out.**
  - Today "the arch has Sushi Cooler" is the same fact as "it declares `cooler_opt_out`" (`arch_config.coolerOptOut`,
    `cooler_side.sideOf`). Hy4 runs on Sushi Cooler only, with no MLX route, so it could only be declared by
    inventing a flag, or by a special case in `sideOf`.
  - The general seam: an arch declares `cooler: arch_config.Cooler = .{ .opt_out = … }`, where `opt_out` is the flag
    and setting, or null for an arch whose every load runs on Sushi Cooler.
  - `sideOf` answers `.engine` for such an arch, tests included: there is no MLX route to keep. `coolerGap` is
    required only beside an opt-out.
  - Part 1a: done (`arch_config.Cooler`, `cooler_side.sideFor`).
<a id="r2"></a>
- **R2. Engram's hash and table reader, shared.**
  - DeepSeek's `dsv41_engram.zig` held the reference Engram hash (numpy-seeded multipliers, prime buckets, rolling
    XOR) and the SSD row reader (`Table`, `RowCache`, `Prefetch`). Hy4 uses the same hash and the same read pattern.
  - An arch folder cannot import another's, and a copy would fork the hash.
  - The general seam: `src/engram.zig`, named by capability, with an `engram.Geometry` that carries the layer each
    module sits at and the id its multipliers are drawn with (Hy4 hashes with the 1-based id, DeepSeek with the
    index), plus the seed. DeepSeek fills it from its config.
  - The table's tensor names become a parameter. DeepSeek's mxfp8 row format stays the reader's only format until
    Hy4's loader adds the BF16 and plain e4m3 arms (part 1k3).
  - Part 1b: `src/engram.zig` holds them now; `dsv41_engram.zig` keeps DeepSeek's table names and token map. A test
    pins Hy4's multipliers (ids 2, 12, 22, 32) and its listed primes.
<a id="r3"></a>
- **R3. Hyper-connection stage kernels with a comb kind.**
  - DeepSeek's stage kernels (`dsv41_hc_stage*`, `dsv41_hc_mix(_nax)`, `dsv41_hc_split`, `dsv41_hc_expand`, and the
    split's rides on the router and the attention launches) compute the mix, collapse and expand that iHC needs, with
    a Sinkhorn comb Hy4 lacks.
  - The general seam: a shared hyper-connection stage with `comb: .sinkhorn | .identity`, the post's magnitude and
    eps, and the mix count. The expand without a comb is `post·y + h`.
  - Part 1c: done. `src/hc_kernels.zig` holds the stage kernels with their source and pipeline names unchanged, over
    `src/metal_def.zig`'s shared launch; `dsv41_kernels` wraps them.
  - Part 1d: done, as identity-comb kernels beside them rather than a comb flag on DeepSeek's stage: Hy4 collapses a
    stage by its own pre, DeepSeek's stage by the previous one. Hy4 on a GPU stream runs the shared mix, then one
    split, collapse and norm kernel (`normIdentity`, the head's `headIdentity`), and `expandIdentity` (the op chain's
    bytes); the CPU stream keeps the op chain. `probeIdentity` halves `hc_tg` at load where the GPU caps it.
<a id="r4"></a>
- **R4. Sparse-select kernels shared.**
  - DeepSeek's relu-weighted index score kernels (scalar, tiled, NAX) and its top-k kernels (one threadgroup, split,
    below-mask, ties to the lower index) are the score and selection Hy4's indexer runs, over BF16 pooled keys
    instead of packed FP4 keys.
  - The general seam: shared kernels with a key format. Each kernel keeps its pipeline name so the manifest's
    recipes do not move.
  - Part 1e: done. `src/sparse_select.zig` holds the plain-key score kernels (one thread a key, and the matrix tile) and
    the top-k kernels, source and pipeline names unchanged; DeepSeek's FP4-packed and NAX score kernels stay in its
    folder on the shared fragments.
  - Part 1f: Hy4's recall on a GPU stream scores with `indexScores` (from `p0 - 1`, so a row sees the groups before its
    own) and selects with `topkIndices`, in place of the `[H, T, G]` product and argpartition; the kernel reads keys in
    their stored dtype, BF16 included. Pooled keys stay f32 until the streamed STQ prefill sums its groups in f32: with
    BF16 keys, that prefill's rounded partials flip near-tied picks against the resident path.
<a id="r5"></a>
- **R5. Latent attention kernels with a row geometry.**
  - DeepSeek's window-and-picks attention (decode, split and merge; the wide fused and NAX spans) reads one shared
    KV row whose 512 values serve as both K and V, and folds an inverse rotation into the output. Hy4's row is a
    576-wide K whose first 512 are V, with no fold.
  - The general seam: the same kernels with `(dk, dv, fold)`.
  - Same 1.7.0 reason. Parts 1g and 1h.
  - What it bought, as an arch kernel and not the shared seam (DeepSeek's kernels are untouched, so its bytes are): the
    key row, the f32 query split in three bf16 parts, the row ranges and picks, and the 9 simdgroups (8 over the latent's
    64-channel slices, one over the rope channels) are Hy4's own, on DeepSeek's fragment layout (`hy4_attn.zig`).
    The query is split into three bf16 parts the matrix units take (f32's range and 24 bits), P (in [0, 1]) into two;
    the keys are bf16, so exact. Dropping a low pass moves the heads 5e-4 to 3e-3 of the peak from the f32 chain's. Two bf16
    parts (2^-18) leave the kernel 4 to 10 times less exact than the op chain against the f64 truth (1.6e-5 against
    7e-6 at a query of 4 sigma, 6e-5 against 3e-5 at 16), and a model amplifies that: a tiny's logits drift from the
    chain's by KL 1e-2 to 1e-1 at the last prefill row; three parts on the query bring every release-shaped cell under the chain's own error (`hy4 attn
    range`, `hy4 attn picks`, bar 5e-6 of the truth's peak at a query of 4 sigma) and hold the absorbed query's
    outliers (`hy4 attn outliers`). An f16 low part of the query (times 512) overflowed past |q| of 32,752 and
    collapsed a head onto its sink: real-pack 8-category KLD 0.025859 against the chain's 0.022150, and 0.021761 with
    three bf16 parts on both (hy4-next 92bacb22 plus the core; two bf16 parts on both read 0.022844), 0.020555 with P
    at two parts, the form served. Each pass is a matrix-unit product: the kernel is bound by them (P at two parts
    takes a 4,096-row chunk after 4,096 in 2,208 ms against 2,357 at three, prefill meter, 408fcc02, M5 Ultra, STQ
    4bpw, A B B A, QUIET, GPU lock). The tile is 32 keys, two
    barriers a tile; the kernel runs near MLX's own bf16 GEMM rate (about 65 TFLOP/s counting its passes, MLX's bf16
    GEMM 70 on this box), so its floor is the passes.
  - In-process, M5 Ultra, `taskpolicy -a`, not quiet, no lock, best of 5 rounds in rotated order (`hy4_attn.zig`'s
    timing, not committed), one layer's core over a 4,096-row chunk, ms: sliding (64 heads, 512-key windows) 23.7 for
    the op chain, 9.5 in 256-row spans, 8.0 in one launch; gathered (32 heads, 2,048 picks a row) 55.4 for the chain
    with its gather in 32-row spans, 18.7 in 32-row spans and 16.8 in 256-row spans, reading the store at BF16 (read
    in the kernel from kv8 it took 34.8 to 41.3: a key row's dequantize costs more than its dot).
- **Not a seam**:
  - The router (`moe.biasRouting` already takes an arch's numbers).
  - The rate-group chain (width and clamp are data).
  - The prefix cache (DeepSeek's whole-state checkpoint is reached through the arch's `prefix` module).

## Speed

M5 Ultra, STQ 4bpw, no launch flags (every routed expert resident), `tests/bench.sh --rungs 4k,8k`, arms A B B A in one
session, ReleaseFast, GPU lock and QUIET:

| tree | prefill 4K | prefill 8K | decode |
|---|---|---|---|
| f2c57141 (`--ssd-budget-gb 220`) | 498 | 517 | 17.9-18.3 |
| 8014b557 | 758-763 | 749-752 | 37.9-38.0 |
| 13c8fa96 (through-bf16 trunk launches, norm folds, iHC expand, gate and W_o once a chunk) | 853-861 | 854-856 | 39.1-39.2 |

The release ladder at 9f0f1bea (`sushi bench --model`, no launch flags: DFly2 drafting, kv8, every routed expert
resident; M5 Ultra, ReleaseFast, `taskpolicy -a`, GPU lock and QUIET, one run; tok/s):

| rung | code prefill | code decode | prose prefill | prose decode |
|---|---|---|---|---|
| 4k | 1,263 | 139.6 | 1,009 | 59.5 |
| 8k | 1,086 | 110.6 | 913 | 67.0 |
| 16k | 1,073 | 91.4 | 941 | 59.6 |
| 32k | 1,048 | 80.1 | 937 | 62.0 |
| 64k | 942 | 85.1 | 938 | 58.7 |
| 128k | 848 | 90.1 | 861 | 55.9 |

Code decodes faster than prose because DFly2's drafts are accepted more often on it. The 8-category KLD of the same
tree (kv8, `--no-mtp`) is 0.022150 to the first EOS.

A decode token at 8014b557 is about 31 ms of GPU work and the rest host time the GPU waits behind: the engine log's
`GPU busy` against `idle between passes` over 256 tokens.

## Parity plan

**The oracle.**
- A dumper, `tests/dump_hy4_fixtures.py`, shaped like `tests/dsv41/dump_dsv41_fixtures.py`, runs transformers' `modeling_hy_v4_flash`
  on the CPU in f32 over a TINY random model written in the release's own layout and names.
- It dumps the logits, every layer's four streams, and sub-module captures: the Engram row ids, rows and delta; the
  iHC mixes; the indexer's pooled keys, scores and picks; the router picks and weights; the sink-softmax attention
  output.
- Tiny geometry:
  - 6 layers (sparse, sparse, sliding × 3, sparse), hidden 128, 2 sparse heads and 4 sliding heads, latent 64,
    rope 16;
  - an indexer of 2 × 32 with groups of 4 and a group top-k of 4, so selection engages at slot 16;
  - window 8; 8 experts, top-2, latent 32;
  - one Engram module with small primes; a two-block tower for images.
- **Bars** (DeepSeek's): on the CPU stream in f32, every layer's trace within 1e-4 of its peak and the logits within
  2e-3. Row ids and picks identical. Ties held to the reference's own margins (relu leaves exact-zero scores, as
  Qwen's `Qwen4Ties` found), never to our output.

**The real checkpoint against llama.cpp.**
- Build llama.cpp#32517 and run a lossless GGUF of the release. The GGUF is a parity reference only. The PR's
  converter does not take the released config as published, so the conversion maps the three spellings first.
  AngelSlim's GGUFs are lossy and serve only as a smoke test.
- Run text prompts on both sides with greedy decode. Compare the tokens up to the first near-tie (margin recorded),
  the top-k logit overlap, and the logits' cosine.
- A divergence is bisected with llama.cpp's named graph nodes (`llama-eval-callback`: `hc_mixes`, `attn_norm`,
  `q_resid`, `indexer_score`, `indexer_top_k`, `ffn_moe_logits`, `engram_delta`, `l_out`) against Cooler's per-layer
  captures.
- Cooler already follows llama.cpp at every listed [difference](#where-the-references-differ), so the run expects
  agreement there too; the tiny oracle's dumper patches transformers to llama.cpp's choice at the same points.
- Never in a test: the real checkpoint is the KLD teacher and the parity run, as on every arch, loaded by hand under
  the GPU lock.

**The teacher.**
- The BF16 checkpoint, experts streamed, is the lossless KLD teacher for every pack ([quality-kld](quality-kld.md)).
- `sushi kld capture|compare` takes it through the generic path; `hy4_kld_capture.zig` refuses what Hy4 cannot honor, and
  the latent width a run names (`--kv-quant 16` or `8`) is the one the model stores, with f32 products held off TF32
  ([quality-kld](quality-kld.md#hy4-kld)). On an M5 (NAX) the tiny's GPU tests need `MLX_ENABLE_TF32=0`: TF32 products
  put the composed forward's logits up to 33% of the peak off the oracle; the CPU stream and an M1 Max are unaffected.
- The calibration capture for the STQ pack is `sushi kld capture --layer-major` over 6pCorpus-1m's text windows under
  Hy4's tokenizer and template ([quality-kld](quality-kld.md#hy4-capture)). Hy4's tokenizer is denser than Flash-Next's:
  the design part holds 179,538 tokens and chat 195,018, so the recipe's balanced cut (2048/1024/1024/512 x 4, five parts
  round robin) reaches 1,015 windows and 890,880 tokens, not Flash-Next's 1,195 and 1,054,720.

<a id="dfly2-parity"></a>
**DFly2 parity.** A drafter cannot change what the target emits, only how fast. Its checks are three kinds.
- **The forward**: the AngelSpec oracle above, on the tiny's random drafter weights.
- **Exactness (the tiny)**:
  - Greedy rounds give the serial greedy tokens byte for byte, at every accepted count, text and image.
  - Seeded sampled rounds give the seeded serial request's tokens under the exact accept rule.
  - A partial accept rolls the target's rows, open groups and Engram history back as DSpark's verify does.
  - The drafter's reads of the shared layers' rows and picks equal the target's own, row for row.
  - These hold whatever the drafter's weights are, so a random tiny proves them.
- **Acceptance (the release)**: `SUSHI_HY4_DFLY_PROBE=<tokens>[,<every>]` (read once at load, with `--mtp`) is
  teacher-forced. Each prompt decodes `<tokens>` (default 256) greedily on the target alone. At every `<every>`-th
  position (default 1) a block is drafted from the state there and scored against the greedy tokens after it. It logs
  `[dfly-probe]` lines: per slot, the acceptance given the slots before accepted, the match alone and the mean
  confidence; per prompt and in all, the mean accepted a block. The prompts are a built-in code and prose pair (no
  thinking), or one file's text, chat-formatted, with `SUSHI_HY4_DFLY_PROBE_TEXT=<abs path>`. The model is reset after.

<a id="open-questions"></a>
## Open questions

Settled by the owner on 2026-11-18: DFly2 in 1.8.0, the effort words, the 8-bit latent default, and llama.cpp at every
reference difference (the Engram norm ε and rows, the streams' dtype, image positions, the vision GELU).

1. **Indexer ties**: torch's `topk` order among equal scores. Exact ties are common, since relu zeroes whole groups.
   DeepSeek's kernels tie to the lower index; the tiny oracle's ties are held to the reference's own margins.
2. **Batched decode**: a streamed load serves one request at a time, as DeepSeek's does. Joining requests as rows on a
   fully resident pack follows DeepSeek's `coolerRows`, after the core.

## Implementation plan

Row 1 of the 1.8.x board; each part is one reviewable unit of about half a day that lands green on its own: its test
first (TDD), then the subset of the suite its diff reaches.
- **dev1: BF16 streaming on a 32 GB Mac.** The forward is first composed from the engine's existing ops, as DeepSeek's
  composed boot runs (`fused` off): no new kernel, so nothing waits for 1.7.0. It is held to the tiny oracle, then
  served from the release checkpoint with its experts and Engram tables streamed.
- **dev2: the fused forward.** Parts 1c to 1h move DeepSeek's release kernels, so they start once 1.7.0 ships; each
  replaces a composed piece and is held to the composed output on the tiny. Warm, the prefix cache, the 8-bit latent
  and vision follow.
- **dev3: the STQ pack and DFly2.**

| key | dev | part | depends | done when |
|---|---|---|---|---|
| 1a | dev1 | R1: the `cooler` declaration, an opt-out or none |  | a Cooler-only declaration resolves to the engine in a unit test; DeepSeek's and Qwen's sides, `--help` (`src/fixtures/help.txt`), flags and settings unchanged; lint green |
| 1b | dev1 | R2: `src/engram.zig`, hash and table reader shared, `Geometry` with hash ids and seed |  | DeepSeek's Engram tests pass from the shared file unchanged; a Hy4-geometry test reproduces the config's 64 primes and the reference's multipliers for ids 2, 12, 22, 32; DeepSeek tinies byte-identical |
| 1j0 | dev1 | Land the seams: exp/hy4-cooler-seams (R1 3c87319a, R2 4ab495bf) rebased onto the doc branch | 1a, 1b | rebased clean; their tests and lint green on the M1; arch drop test (--since b015eace) run on the M5 |
| 1i1 | dev1 | hy4_config.zig: parse the released config.json | 1j0 | the release config parses (layer_types with its three spellings, MLA, indexer, Engram, iHC, DFly2 keys); an unknown or out-of-range key refused by name; host test |
| 1i2 | dev1 | src/hy4flash/ arch skeleton: arch.zig, registry line, DROPPABLE, the test folder in build.zig | 1i1 | hy_v4_flash is an arch whose load refuses by name (`Hy4ForwardNotBuilt`) and ships no pack (`packs.unreleased`); the arch slice and drop test pass |
| 1i3 | dev1 | Tensor census: every one of the 1,964 checkpoint names mapped to a role | 1i2 | a host test over the release's tensor list (`fixtures/hy4_tensors.txt`, from its shard headers) maps each name (trunk, experts, Engram, vision, DFly2) and refuses an unknown one by name; byte totals equal the doc's table |
| 1i4 | dev1 | Tokenizer and chat template | 1i2 | the template renders byte-exact against apply_chat_template on its cases (system synthesis, effort words, tools, tool results, preserved thinking); special tokens (BOS 120000, EOS 120025, pad 120002) read from the release files |
| 1j1 | dev1 | Oracle: tiny random model in the release's own layout and names | 1i1 | the dumper writes a 6-layer tiny (doc's tiny geometry) as safetensors + config; transformers (PR 50903 head c2ec8505) loads it in its own round trip |
| 1j2 | dev1 | Oracle: llama.cpp's choices patched into the dumper | 1j1 | each listed reference difference switched to llama.cpp's side (Engram rows as stored, fp32-eps Engram norms, pad-id image history, identity ids, f32 streams, consecutive image positions, tanh GELU); a check that each patch changes the output it should |
| 1j3 | dev1 | Oracle: dumps | 1j2 | logits, every layer's four streams, and captures (Engram ids/rows/delta, iHC mixes, indexer pooled keys/scores/picks, router picks/weights, sink-attention output) for prefill and decode, text first |
| 1k1 | dev1 | Loader: resident BF16 trunk | 1i3 | attention, shared experts, dense layer 0, embedding, latent projections, norms and the f32 keep list loaded by the release's names on Sushi Cooler with no MLX call (census); the head widened to f32; `arch.zig` declares `cooler` with no opt-out and its `cooler_contract` on the tiny fixture |
| 1k2 | dev1 | Loader: streamed BF16 experts | 1k1 | the fused gate_up split by views (gate rows 0..1535, up 1536..3071), down banks, 41 layers x 512 through the expert stream under --ssd-budget-gb |
| 1k3 | dev1 | Loader: Engram tables by row (BF16 arm) and lookup_table as the token map | 1k1, 1b | rows read from the safetensors by offset through the shared engram reader; the token map from the checkpoint tensor |
| 1k4 | dev1 | Loader bill: resident bytes | 1k2, 1k3 | resident bytes equal the bill on the tiny and in a host count of the release |
| 1l1 | dev1 | Composed forward: embedding, iHC pre/post/head, f32 streams | 1j3, 1k1 | first layer's streams and the head within the bars on the tiny (engine ops, no new kernel) |
| 1l2 | dev1 | Composed forward: dense layer 0 and the f32 head | 1l1 | layer 0's trace and the logits of a 0-layer-MoE slice within the bars |
| 1m1 | dev1 | Composed forward: Engram | 1l1, 1k3 | row ids identical to the oracle's; delta within the bar; zero at image and pad ids; history across a chunk boundary |
| 1n1 | dev1 | Composed forward: sliding MLA prefill | 1l2 | q LoRA, latent, absorbed attention over the 512 window with the sink, gate, sandwich norm: sliding layers within the bars on a prefill |
| 1n2 | dev1 | Composed forward: sliding ring decode | 1n1 | the 512-row ring written at pos % 512; decode steps within the bars past the window |
| 1o1 | dev1 | Composed forward: router and latent MoE | 1l2, 1k2 | sigmoid router, bias picks, x3.998, latent down/up around the clamped SwiGLU experts, shared expert, combine norm: router picks identical, layer traces within the bars |
| 1o2 | dev1 | Composed forward: sparse layers, dense phase | 1n2, 1o1 | sparse layers attending densely below 2,048 keys within the bars, prefill and decode |
| 1p1 | dev1 | Composed forward: indexer pooling | 1o2 | pooled keys (ape, gated pool, norm, rope at the group's first slot) equal the oracle's within the bar |
| 1p2 | dev1 | Composed forward: index scores, top-511 and picked-row attention | 1p1 | scores within the bar, picks identical past slot 16 (ties to the reference's margins), logits within 2e-3 |
| 1p3 | dev1 | Chunked prefill state | 1p2 | open groups, pooled keys, rings and Engram history the same at every chunking |
| 1q1 | dev1 | Request loop: chunked prefill and serial greedy decode | 1p3 | greedy tokens after a prefill equal the oracle's on the tiny; a served request on the tiny end to end |
| 1q2 | dev1 | Chat face: effort words and thinking budgets | 1q1, 1i4 | off/minimal/low/medium/high/on/xhigh/max render and cap as ruled, pinned by a test; sampling defaults 0.9 / 1.0 |
| 1q3 | dev1 | Tool calls: the :opensource arg_key spelling | 1q2 | GLM's arg_key parse with the <tool_calls:opensource> wrapper; format-corpus entries; streamed and non-streamed bytes equal |
| 1s1 | dev1 | Bills: BF16 streamed, exact plus the margin | 1q1 | per token (9,600 B at kv8, 15,360 B at BF16), per request, per chunk (streams, attention rows, routed union, selection, score tile, Engram rows); a live bill test holds the engine's pool peak under each |
| 1s2 | dev1 | Fit on a 32 GB Mac | 1s1 | the release's BF16 load admitted on the M1 at a stated --ssd-budget-gb: trunk ~14.7 GB resident, a prefill union that fits (a 7.25 GB BF16 layer union does not: the chunk and union sized by the bill), or refused by name with the bytes |
| 1r1 | dev1 | First real BF16 streamed load on the M1 | 1s2 | the release loads, warms and answers greedy prompts coherently (text, English and one other language); decode and prefill tok/s and RSS recorded with commit and budget |
| 1r2 | dev1 | Real-checkpoint smoke against llama.cpp | 1r1 | the PR built; greedy agreement against AngelSlim's Q4_K_M recorded as a smoke (lossy); the lossless BF16 GGUF parity (~440 GB) on the M5 or a disk that holds it |
| 1r3 | dev1 | KLD teacher fixture from the BF16 streamed load | 1r1 | sushi kld capture on the 16x512 fixture from the BF16 checkpoint: the teacher every later pack and the 8-bit KV are judged against |
| 1c | dev2 | R3, step 1: the hyper-connection stage kernels move to a shared module, unchanged | 1.7.0 shipped | DeepSeek tinies and the hc kernel tests byte-identical; pipeline names unchanged |
| 1d | dev2 | R3, step 2: `comb` identity, post magnitude and eps, mix count | 1c, 1l1 | an identity-comb case equals an f64 chain; DeepSeek bytes unchanged |
| 1e | dev2 | R4, step 1: index score and top-k kernels shared, unchanged | 1.7.0 shipped | the saved scans (`DSV41_INDEX_SCANS`) bit-identical; top-k tests unchanged |
| 1f | dev2 | R4, step 2: BF16 pooled-key arm | 1e, 1p2 | BF16 keys score as the f64 truth within the scalar kernel's bar; top-k ties to the lower index |
| 1g | dev2 | R5, step 1: window-and-picks attention shared, unchanged | 1.7.0 shipped | DeepSeek attention tests bit-identical |
| 1h | dev2 | R5, step 2: `dk` 576 / `dv` 512, no fold | 1g, 1n2 | the new geometry within the fused kernel's bar against the CPU f32 core |
| 1t | dev2 | warm and the pipeline manifest | 1s1 | a request compiles no kernel (`compiled inside a request` 0); `[pipelines]` cases as DeepSeek's |
| 1u | dev2 | prefix cache: restore points, row segments, namespace, live rollback | 1s1 | a restored turn's state and logits equal a cold prefill's, byte for byte, both tiers |
| 1y | dev2 | KV width: the 8-bit affine latent (the engine's kv8, group 64, rope dims BF16) as the default, BF16 also served | 1u, 1r3 | both widths serve; a row's 8-bit round trip equals the engine's affine quantize/dequantize; KLD of 8-bit against BF16 KV recorded (16x512 fixture, BF16 teacher); bills pinned for both widths (9,600 and 15,360 B a token); the prefix-cache namespace names the width and a restore never crosses it |
| 1v | dev2 | vision: tower parameters (tanh GELU), merger, consecutive image positions | 1q1 | merger output and the first image row's logits within the bars against the oracle's image case |
| 1w | dev3 | pack format, STQ only: the `pack-format` section, STQ experts and trunk, the Engram table forms | 1r3 | a tiny STQ pack loads and serves; served against its dense twin within DeepSeek's STQ bars; KLD measured against the BF16 teacher |
| 1x | dev3 | DFly2 on the tiny: the loader (`model.dfly_layers.*`), the forward over the shared layers' rows and picks, DSpark's rounds with a mask block of 6 | 1u | greedy rounds equal serial greedy byte for byte at every accepted count, text and image; seeded sampled rounds equal the seeded serial request; rollback restores rows, open groups and Engram history; the drafter's reads equal the target's rows and picks; its bill adds weights and one round's block rows, no KV |
| 1z | dev3 | DFly2 on the release checkpoint: the open wiring settled, the adaptive round policy | 1x, 1r1 | each wiring choice's accepted tokens a round recorded with its margin; every spec-named component lowers acceptance when switched off; confidence ranks the slots as they verify; the round policy never loses to serial decode on the prompt set |