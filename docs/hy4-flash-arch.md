# Hy4-Flash (`hy_v4_flash`): the design for Sushi Cooler

Tencent's Hy4-Flash is not served yet. This doc maps its forward pass onto Sushi Cooler, component by component:
the formula, the public source that pins it, and the engine piece it reuses or needs. It also lists the seams the
engine must generalize first, the memory bill, what a pack needs, the parity plan, the open questions, and the parts
the work lands in.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-arch-seam](engine-arch-seam.md),
[engine-own-forward](engine-own-forward.md), [engine-memory-admission](engine-memory-admission.md),
[dsv41-arch](dsv41-arch.md) (the closest served arch), [qwen4-arch](qwen4-arch.md) (vision tower, M-RoPE),
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
| [Tencent/AngelSpec](https://github.com/Tencent/AngelSpec), [arXiv 2607.25852](https://arxiv.org/abs/2607.25852) | DFly, the drafter family DFly2 belongs to |

Where the two references disagree, Cooler follows transformers, because it is Tencent's code and it runs on the
released checkpoint. The disagreements are listed in [where the references differ](#where-the-references-differ).

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
| Engram projections, router, indexer, hyper-connections, norms | 0.43 |

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
| Engram table read, row cache, read-ahead | DeepSeek's `Table`, `RowCache`, `Prefetch` | reused with a parameter ([R2](#r2)); a BF16 / row-scaled FP8 row format is new |
| Engram gate and delta | — | new kernel (the `dsv41_engram` gate chain's shape) |
| iHC mix, collapse, expand, head | DeepSeek's hyper-connection stage kernels | reused with a parameter: no comb, post magnitude and eps ([R3](#r3)) |
| norms (RMS, sandwich) | the engine's RMS norm | reused as is |
| MLA projections (q LoRA, KV latent, gate, output) | trunk linears, stacked input launch | reused as is |
| absorbed MLA attention, window ring and picked rows, sink | DeepSeek's attention kernels | reused with a parameter: K 576 / V 512 on one row, no rotation fold ([R5](#r5)) |
| RoPE (consecutive pairs, θ per layer kind, interleaved M-RoPE) | `mrope.interleavedSelector`, `fillCosSin`; DeepSeek's pair rotation | reused with a parameter |
| vision M-RoPE position index | — | new arch code (rows of W + 1 with the newline) |
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
| 8-bit KV latent (default) | the attention core and the row writes | new kernel arms: e4m3 tiles with f32 scales, rope dims BF16 ([KV width](#kv-width)) |
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
- **Cooler**: reused as is. That is the trunk embedding gather of the BF16 table (an STQ pack keeps the embedding as
  stored) and the vision splice.

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
- **Rows**: `r_j = T[row_j]`, 96 BF16 values. With `engram_fp8_embedding` the reference rounds each row through FP8:
  `s = amax|r|/448`, `r' = e4m3(r/s)·s`, in the activation dtype.
- **Gate and delta**:
  - `g_j = RMSNorm_{w_j}(r'_j)` per head, and `u = concat_j g_j` (1536 wide).
  - Keys `k_c = RMSNorm_{K_c}(W_c u + b_c)` for c = 0..3; value `v = W_v u + b_v`.
  - `d_c = <k_c, RMSNorm_{Q_c}(h_c)> / sqrt(3072)`, and the gate is `a_c = sigmoid(sign(d_c)·sqrt(max(|d_c|, 1e-6)))`.
  - `y = γ ⊙ RMS(v)`, and `h_c ← h_c + (a_c + σ_c) ⊙ y`. `γ` is `unified_pre_gamma`; `σ_c` is the per-channel
    `unified_per_hc_scales[c]`.
  - The delta is zero where the position's own token is an image, video or pad id.
- **Pinned by**: transformers `HYV4FlashEngramLayer` (`hash_ngrams`, `draw_hash_multipliers`, `forward`),
  `_fp8_round_trip`, `HYV4FlashTextModel.build_ngram_tokens`, `HYV4FlashModel.build_engram_inputs`; llama.cpp
  `engram_hash`, `graph::build_engram` (which folds `norm_key·norm_query` into one `engram_gate` vector); converter
  `_engram_multipliers`.
- **Cooler**:
  - The hash is DeepSeek's `Hash` with two parameters, the seed and the id the multipliers are drawn with
    ([R2](#r2)).
  - The table is never resident. That is 126.7 GB BF16, or 63.4 GB as FP8: read by row from SSD through DeepSeek's
    `Table`, `RowCache` and `Prefetch`. The read covers the prompt's next chunk ahead, and a decode token's rows hide
    behind the layers before layer 1.
  - The gate chain is a new kernel of the `dsv41_engram` shape. Hy4's has biases, per-head lookup norms, `γ` and an
    additive per-stream scale where DeepSeek's has none of these.
  - The token map is the checkpoint's `lookup_table` tensor, not an embedded file.
- Per decode token: 4 modules × 16 rows = 64 rows, 12 KB of BF16.

### Hyper-connections (iHC)

- **Pre and post**:
  - `f = concat_c h_c` (12,288 wide), `μ = rsqrt(mean(f²) + 1e-5)`, `m = (F·f)·μ` with `F` `[8, 12288]` f32.
  - `pre_c = sigmoid(m_c·s_0 + b_c) + hc_eps`, and `post_c = 2·sigmoid(m_{4+c}·s_1 + b_{4+c}) + hc_eps`.
  - The sublayer input is `x = Σ_c pre_c·h_c`. After the sublayer output `y`: `h_c ← h_c + post_c·y`.
  - There is no comb matrix and no Sinkhorn: each stream keeps its own residual.
  - Mixes and collapse run in f32; the streams are rounded to BF16 after each sublayer (`.to(dtype)` in the
    reference).
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
- **Pinned by**: transformers `HYV4FlashAttention.forward` (the expanded form, the same math) and
  `eager_attention_forward` (sinks); llama.cpp `graph::build_attention`.
- **Cooler**:
  - The projections are trunk linears; `W_qa`, `W_kva` and `W_g` read the same `x` and stack into one launch, as
    DeepSeek's `Layer.in_shared`.
  - The attention core is DeepSeek's window-and-picks kernel family (online f32 softmax with a sink, picked rows read
    by index), given a K row wider than its V prefix and no inverse-rotation fold ([R5](#r5)).
  - GLM's absorbed MLA (`glm5_attention.zig`) is the same math, but on MLX only.

### Sliding-window layers (30)

- **Formula**: the keys are the last 512 slots (the token and the 511 before it). The mask counts cache slots, not
  M-RoPE positions. 64 heads, θ 1e4.
- **Pinned by**: transformers `create_sliding_window_causal_mask` with `sliding_window` 512 and `window_size`
  `[511, 0]`; llama.cpp `LLAMA_SWA_TYPE_STANDARD` with `n_swa` 512.
- **Cooler**: DeepSeek's window ring reused with a parameter. The ring holds 512 rows of 576, written at
  `pos % 512` by the rope-and-write launch (`dsv41_rope_kv_ring`'s pattern).

### Sparse layers (12) and the compressed indexer

- **Token key**: `k_t = RMSNorm(W_k x_t)` (128 wide).
- **Groups**: 4 consecutive cache slots; only complete groups are pooled. In f32:
  - `v = W_v k` and `a = W_g k + ape[slot]`;
  - `K_g = RMSNorm( Σ_slot softmax_slot(a) ⊙ v )`, per channel, rounded to BF16;
  - the trailing 64 dims are rotated at the position of the group's first slot (θ 1e7, M-RoPE planes for images).
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

### RoPE

- **Formula**: consecutive pairs `(x_{2i}, x_{2i+1})` (GPT-J order, ggml `NORM`) on the trailing 64 dims of the query
  and key (the indexer's too). `inv_freq_i = θ^(-2i/64)`, with θ 1e7 on sparse layers and the indexer and θ 1e4 on
  sliding layers. `rope_type` is default: no YaRN.
- **M-RoPE**: interleaved with section [12, 10, 10]. Frequency `i` takes the height position when `i mod 3 = 1` and
  `i < 30`, the width position when `i mod 3 = 2` and `i < 30`, and the temporal position otherwise. A text token has
  all three equal, so it rotates as plain RoPE. cos and sin are computed in f32 and applied in f32, then the result
  is rounded to BF16.
- **Pinned by**: transformers `HYV4FlashRotaryEmbedding` (`recomposition_frequencies`), `apply_rotary_pos_emb`
  (`repeat_interleave`, pairwise `rotate_half`); llama.cpp `LLAMA_ROPE_TYPE_NORM` for `hy_v4_flash`.
- **Cooler**:
  - `mrope.interleavedSelector` and `fillCosSin` are reused with a parameter (section [12, 10, 10], θ by layer kind).
  - The pair rotation is DeepSeek's tail rotation without YaRN.
  - The vision position index is new arch code. A merged row has W/2 + 1 tokens (the newline included), all three
    axes of a frame start at the frame's offset, frames step by `max(H/2, W/2 + 1)`, and text resumes at the largest
    position + 1. Qwen's `getRopeIndex` has no newline column.

### MoE (layers 1-41) and the dense layer 0

- **Router**: `ℓ = W_r x` in f32 (the BF16 weight widened, which is exact), `s = sigmoid(ℓ)`. Pick the top 16 of
  `s + b` (`b` is the f32 bias). Weights are `s_e / (Σ_picked s + 1e-20) × 3.998`. With `n_group = topk_group = 1`
  there is no group stage.
- **Routed experts, in the latent**:
  - `z = W_down x` (3072 → 1536). For each pick e, `[g | u] = W_gu,e z` (gate rows first), `g ← min(g, 10)`,
    `u ← clamp(u, -10, 10)`, `y_e = W_d,e (silu(g)·u)` (1536).
  - `R = W_up Σ_e w_e y_e` (1536 → 3072).
- **Shared expert**: SwiGLU at full width (3072 → 1536 → 3072), no clamp.
- **Out**: `RMSNorm(R + shared)` (`combine_output_norm`, the FFN half of the sandwich). Layer 0 is a dense SwiGLU
  (13,824) followed by `post_fc2_norm`.
- **Pinned by**: transformers `HYV4FlashTopkRouter`, `HYV4FlashExperts._apply_gate`, `HYV4FlashMoE`,
  `HYV4FlashMLP`; llama.cpp `build_latent_moe` with `hy_v4.py`'s sigmoid gating and clamp keys.
- **Cooler**:
  - Routing is the arch's `moe.biasRouting` hook (sigmoid, bias picks only, normalize, × 3.998, 512 experts, top-16).
  - The routed chain is the STQ rate-group chain with the clamped SwiGLU at 1536 × 1536, streamed through the expert
    cache from `--ssd-budget-gb` as DeepSeek's are ([engine-exl3-experts](engine-exl3-experts.md#rate-groups)).
  - The latent projections are two plain trunk linears around the chain.
  - The shared expert reads `x`, not `z`, so the shared fold into the routed launches does not apply; it runs as the
    trunk SwiGLU.
  - The BF16 checkpoint streams its experts through the engine's `gather_mm`, as Qwen's BF16 checkpoint does on
    Sushi Cooler. The fused `gate_up` bank is split by views (gate rows 0..1535, up rows 1536..3071).

### Norms

- **Formula**: RMSNorm with ε 1e-5 everywhere in the trunk, in f32, cast to BF16 before the weight. The sandwich
  norms sit after the attention output and after the FFN. The Engram norms are built with ε unset
  ([open question](#open-questions)).
- **Cooler**: the engine's RMS norm, reused as is.

### LM head in f32

- **Formula**: `logits = W_lm · RMSNorm(x_head)`, with both operands in f32 (`enable_lm_head_fp32`). The weight is
  stored BF16 and widened.
- **Pinned by**: transformers `HYV4FlashForConditionalGeneration` (`lm_head` in `_keep_in_fp32_modules_strict`);
  llama.cpp `hy_v4.py` keeps `output.weight` F32.
- **Cooler**: DeepSeek's f32 head, reused as is: widened at load, 1.48 GB.

### Drafter: DFly2

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
- **Pinned by**: README, `config.json` (`speculative_config`, `dfly_*`), the index. transformers ignores the keys
  (`_keys_to_ignore_on_load_unexpected`). llama.cpp#32517 drives a one-layer `model.mtp_layers.*` block instead,
  which the released checkpoint does not carry. No public modeling code for DFly2 was found.
- **The spec**: there is no public modeling code, so the README, the config and these tensors are the spec. Where
  they leave the wiring open, part 1z settles it by measurement:
  - which token's embedding and which position's hidden state reach `enorm`, `hnorm` and the correction;
  - how the `gate` vector mixes the fusion;
  - the block rows' rope positions;
  - how the block's own `kv_a` rows join the target's selected rows;
  - the confidence head's activation.
- **Cooler**: in 1.8.0.
  - The forward is new arch code.
  - The rounds are DeepSeek's DSpark rounds, which already draft a block `[t, noise…]` in one pass on Sushi Cooler,
    with verify, rollback and the adaptive row policy. Here the noise rows become mask rows and the block is 6.
  - The drafter adds no KV bytes: its attention reads the target's rows and picks of the shared layers. Its block
    rows live for one round.
  - How parity is checked without a reference implementation: [parity plan](#dfly2-parity).

### Vision tower and merger

- **Patches**: a Conv2d 16 × 16 (3 → 1152, with bias). Pixels are rescaled by 1/255, then normalized with mean 0.5 and
  std 0.5.
- **Positions**: a learned table of 128 × 128 rows (row 0, a class token, is unused), resampled bilinearly to the
  grid with scale `(n + 0.1)/128` and `align_corners` false.
- **Blocks**: 27 pre-norm blocks: LayerNorm with bias, MHA 16 × 72 with biases and no rotary (the config's
  `rope_2d_*` keys are not read by the reference), and an MLP 1152 → 4304 → 1152 with GELU.
- **Merger**:
  - `RMSNorm` (1152) → Conv2d 1152 → 4608, kernel and stride 2 → GELU (erf) → Conv2d 1 × 1, 4608 → 3072.
  - Then the `image_newline` vector is appended at the end of every merged row. An image of H × W patches gives
    `(H/2)·(W/2 + 1)` rows.
- **Prompt**: `<｜hy_image▁start:opensource｜>`, then that many `<｜hy_image▁pad:opensource｜>` (120120), then
  `<｜hy_image▁end:opensource｜>`.
- **Processor bounds**: 4,096 to 4,194,304 pixels.
- **Pinned by**: transformers `HYV4FlashVisionPatchEmbed`, `HYV4FlashVisionBlock`, `HYV4FlashVisionPatchMerger`,
  `get_vision_position_ids`, `HYV4FlashProcessor._num_frame_tokens`; llama.cpp `mtmd_graph_hy_v4_flash::build` and
  the HunyuanVL position resampling it shares.
- **Cooler**:
  - Qwen's tower has the same widths (27 × 1152, 4304, 16 heads, patch 16) and runs on Sushi Cooler
    ([qwen4-arch](qwen4-arch.md#vision-tower)). It is reused with parameters: a Conv2d patch instead of Conv3d, the
    position-table rule, LayerNorm with bias, no 2-D rotary.
  - Hy4's tower stays as stored (BF16, 0.93 GB). Qwen's load-time 8-bit requantization is an affine format, and
    affine formats leave with MLX in 2.0.
  - The merger is new arch code.
  - Video is refused by name in 1.8.0.

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

  The budgets are the effort table's caps (`effort:hy_v4_flash`), which an operator may change as on every arch.
  Sushi's `max` word is not in the ruling ([open questions](#open-questions)).
- **Tool calls**:
  `<tool_calls:opensource><tool_call:opensource>NAME<arg_key:opensource>K</arg_key:opensource><arg_value:opensource>V</arg_value:opensource>…</tool_call:opensource></tool_calls:opensource>`.
  Non-string values are written as JSON.
- **Tool results**: one `tool` turn of `<tool_response:opensource>…</tool_response:opensource>` items, ordered by the
  assistant's call ids. Earlier reasoning is kept by default when tools are present (`preserved_thinking`).
- **Sampling defaults**: temperature 0.9, top_p 1.0.
- **Cooler**: `hy4_chat.zig` holds the face. The template renders in the Zig Jinja engine (it uses
  `str.format`, `namespace`, macros and `tojson(ensure_ascii=False)`). Tool calls parse through GLM's `arg_key`
  family, given the `:opensource` spelling and the `<tool_calls:opensource>` wrapper.

<a id="where-the-references-differ"></a>
### Where the references differ

| item | transformers (followed) | llama.cpp#32517 |
|---|---|---|
| Engram FP8 round trip | applied per row | not in the graph; the GGUF's table shard is exactly one byte an element plus a 192-byte header (63,360,616,128 bytes = 660,006,416 × 96 + 192), so no per-row scale is stored |
| Engram norm ε | `nn.RMSNorm` with ε unset, which torch resolves from the input dtype | fp32 machine ε (1.19e-7) |
| Engram history across an image | the image placeholder id is hashed like any token | an image position is dead and reads as the pad id |
| Engram ids at or past 120,818 | read the pad id's entry | identity |
| Rope for image positions | interleaved 3-D M-RoPE | `NORM` with one position plane |
| Vision encoder GELU | `ACT2FN["gelu"]` (erf) | tanh (`bias_activation_fusion`) |
| Drafter | DFly2 keys ignored | a one-layer MTP block the release does not carry |
| Config spelling | `deepseek_sparse_attention` remapped to `indexed_attention` | the converter accepts only `sparse_attention`, registers `HYV4NextForCausalLM` and needs `engram_enable` and `engram_vocab_map.npz`: it does not convert the released checkpoint as published |

## The state and the prefix cache

- **Per token**: each sparse layer keeps its latent row and, per completed group, its pooled index key (128 BF16,
  rotated). Token keys of closed groups are not kept: a group's pooled key is final once its fourth token arrives.
  - With the default 8-bit latent (656 B a row, [below](#kv-width)): 12 × 656 + 12 × 64 = **8,640 B a token**, 9.06 GB
    at 1,048,576 tokens.
  - With the BF16 latent (576 × 2 = 1,152 B a row): 12 × 1,152 + 12 × 64 = **14,592 B a token**, 15.3 GB at
    1,048,576 tokens.
- **Per request, fixed**:
  - 30 sliding rings of 512 latent rows: 10,076,160 B at 8 bits, 17,694,720 B at BF16;
  - each sparse layer's open group: up to 3 token keys of 128 BF16, 9,216 B in all;
  - the Engram history: the last 2 token ids.
- **A restore point is the whole state at one position**, as DeepSeek's is ([dsv41-arch](dsv41-arch.md#prefix-cache)):
  - the 30 rings and the open groups, as arrays;
  - the latent rows and pooled keys through the position, as row segments that an SSD entry stores once and links
    across turns;
  - the Engram history, rebuilt from the prompt ids.
  
  A follow-up turn rolls the live state back to the deepest point at or below its shared prefix: rows are
  append-only, so a point is the fixed arrays plus each row buffer's used length. The SSD namespace tag names the
  arch, the kernel arms and the KV width (8 or 16), so a restore point never crosses widths.

<a id="kv-width"></a>
**KV width: an 8-bit latent by default, BF16 also served** (owner's ruling).
- **The format**: per latent row, e4m3 codes for the 512 latent values in 4 tiles of 128, each tile with one f32
  scale (its largest magnitude over 448), and the 64 rope values kept BF16. That is 512 + 16 + 128 = **656 B a row**,
  against 1,152 at BF16.
  - It is the layout of DeepSeek's public FP8 sparse cache in [FlashMLA](https://github.com/deepseek-ai/FlashMLA).
  - It has no zero point, so it is not an affine format and stays on Sushi Cooler past 2.0. GLM's 8-bit latent is
    affine and leaves with MLX.
- **What it covers**: the latent rows of the sparse layers and the sliding rings. The pooled index keys stay BF16:
  they decide the selection, as GLM keeps its pooled index lossless.
- **The widths are served** as `--kv-quant 8` (the default) and `--kv-quant 16`, as on GLM's latent.
- **Error budget** (the design point to measure):
  - An e4m3 code holds a value to within 2^-4 of it, relative, in the normal range (3 mantissa bits). The tile's
    scale keeps its largest value at the top of that range.
  - The error reaches K (the 512 latent values) and V (the same values) of every layer. The rope values, the
    indexer and the pooling are exact.
  - The budget is read as the KLD of the 8-bit latent against BF16 KV: the 8-category 16x512 fixture, with the BF16
    checkpoint as the teacher. The reading goes to the owner beside the decode speed it buys (part 1y).
  - A value more than about 2^14.8 below its tile's largest (448 over e4m3's smallest normal, 2^-6) falls into the
    subnormal range and keeps fewer bits. That tail is what the KLD reading prices.

## The memory bill

On Sushi Cooler a bill is the exact bytes held plus `BILL_MARGIN_BYTES`, once
([engine-memory-admission](engine-memory-admission.md)). The terms, each billed in the same landing as its storage:

| term | bytes |
|---|---|
| resident trunk (BF16 checkpoint) | ~14.7 GB: sliding attention 8.22, sparse attention 1.70, head widened to f32 1.48, shared experts 1.16, latent projections 0.77, embedding 0.74, dense FFN 0.25, Engram projections 0.19, router 0.13, indexer 0.06, hyper-connections 0.03; an STQ pack bills its banks' stored bytes |
| expert cache | `--ssd-budget-gb` less the trunk, the union workspace, the selected slab and the bounce, as on every streamed load. A BF16 expert is 14,155,776 B; a layer's 512 are 7.25 GB, which is the prefill union workspace at BF16 |
| Engram row caches | one `RowCache` per module, its rows times the row bytes (192 B BF16, 98 B as FP8 with a row scale), billed with the trunk |
| state per token | 8,640 B at the 8-bit default, 14,592 B at BF16 (above), times the request's concrete context |
| state per request | 10,085,376 B at 8 bits, 17,703,936 B at BF16 (rings and open groups) |
| per chunk of `C` rows at context `T` | the four streams (`4 × 3072 × C`, BF16, plus the f32 mix rows), the attention rows of the widest layer (64 heads), the routed union and its planes, the selection (`C × 2048` int32), the index scores (one tile of `C × tile` f32, never `C × T/4` whole, as DeepSeek's `indexStream`), the Engram rows of the chunk and the next (`C × 64 × 192` B host); each term is its arrays' exact bytes, pinned by a bill test against the engine's pool peak |
| vision encode | the tower's per-sheet buffers, as Qwen's on Sushi Cooler (`EngineHeld`) |

## What a quantized pack needs

Hy4-Flash is served from two forms only: **STQ packs** and the **original checkpoint** (its own BF16). No affine (MLX
4/8-bit) or non-STQ EXL3 path is designed for it: from 2.0 the engine keeps only STQ packs and original weights. The
llama.cpp GGUFs are a parity reference, never a served format.

The consumer contract is [pack-format](pack-format.md). How packs are made, and every rate and codebook choice,
lives in sashimi, the private converter repo. A Hy4-Flash STQ pack has these parts:
- **Routed experts**: STQ rate groups under the routed nesting, 41 layers × 512 experts, split from the fused
  `gate_up` into gate and up banks. The router rows and the bias follow group order. `swiglu_limit` 10 is a whole
  number, as the clamped chain requires.
- **Trunk linears**: trellis banks where both sides sit on the 128 grid. Every projection does: 3072, 1024, 512,
  1536, 2048 and 16,384 are all multiples of 128.
- **Kept in their stored precision**: every norm weight; sinks; the hyper-connection `fn`, `base` and `scale` (f32);
  the router bias (f32); the indexer's `key_value_projection`, `key_gate_projection` and `intra_group_ape` (f32,
  pooled in f32 by the reference) and `head_weight_projection`; the Engram biases, `γ`, `σ_c` and lookup norms; the
  `lookup_table` (int32); the vision position table and convolutions.
- **The Engram tables**: kept BF16, or stored as e4m3 codes with one BF16 scale per row. That second form is the
  reference's own FP8 round trip, so it loses nothing against the forward the reference runs: 63.4 GB of codes plus
  1.3 GB of scales.
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
  - Part 1a.
<a id="r2"></a>
- **R2. Engram's hash and table reader, shared.**
  - DeepSeek's `dsv41_engram.zig` holds the reference Engram hash (numpy-seeded multipliers, prime buckets, rolling
    XOR) and the SSD row reader (`Table`, `RowCache`, `Prefetch`). Hy4 uses the same hash and the same read pattern.
  - An arch folder cannot import another's, and a copy would fork the hash.
  - The general seam: `src/engram.zig`, named by capability, with an `engram.Geometry` that carries the layer each
    module sits at and the id its multipliers are drawn with (Hy4 hashes with the 1-based id, DeepSeek with the
    index), plus the seed. DeepSeek fills it from its config.
  - The table's tensor names become a parameter. DeepSeek's mxfp8 row format stays the reader's only format until
    Hy4's loader adds the BF16 and row-scaled FP8 arms (part 1k).
  - Part 1b.
<a id="r3"></a>
- **R3. Hyper-connection stage kernels with a comb kind.**
  - DeepSeek's stage kernels (`dsv41_hc_stage*`, `dsv41_hc_mix(_nax)`, `dsv41_hc_split`, `dsv41_hc_expand`, and the
    split's rides on the router and the attention launches) compute the mix, collapse and expand that iHC needs, with
    a Sinkhorn comb Hy4 lacks.
  - The general seam: a shared hyper-connection stage with `comb: .sinkhorn | .identity`, the post's magnitude and
    eps, and the mix count. The expand without a comb is `post·y + h`.
  - It moves kernels on DeepSeek's 1.7.0 decode path, so it waits for 1.7.0 to ship. Parts 1c and 1d; designed here,
    not started.
<a id="r4"></a>
- **R4. Sparse-select kernels shared.**
  - DeepSeek's relu-weighted index score kernels (scalar, tiled, NAX) and its top-k kernels (one threadgroup, split,
    below-mask, ties to the lower index) are the score and selection Hy4's indexer runs, over BF16 pooled keys
    instead of packed FP4 keys.
  - The general seam: shared kernels with a key format. Each kernel keeps its pipeline name so the manifest's
    recipes do not move.
  - Same 1.7.0 reason. Parts 1e and 1f.
<a id="r5"></a>
- **R5. Latent attention kernels with a row geometry.**
  - DeepSeek's window-and-picks attention (decode, split and merge; the wide fused and NAX spans) reads one shared
    KV row whose 512 values serve as both K and V, and folds an inverse rotation into the output. Hy4's row is a
    576-wide K whose first 512 are V, with no fold.
  - The general seam: the same kernels with `(dk, dv, fold)`.
  - Same 1.7.0 reason. Parts 1g and 1h.
- **Not a seam**:
  - The router (`moe.biasRouting` already takes an arch's numbers).
  - The rate-group chain (width and clamp are data).
  - M-RoPE tables (`mrope.zig` already takes a section).
  - The prefix cache (DeepSeek's whole-state checkpoint is reached through the arch's `prefix` module).

## Parity plan

**The oracle.**
- A dumper in `tests/hy4/`, shaped like `tests/dsv41/dump_dsv41_fixtures.py`, runs transformers' `modeling_hy_v4_flash`
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
- The [differences](#where-the-references-differ) are switched to llama.cpp's side by a diagnostic env read once at
  load, so the comparison isolates them. Each one is then decided against the transformers reference.
- Never in a test: the real checkpoint is the KLD teacher and the parity run, as on every arch, loaded by hand under
  the GPU lock.

**The teacher.**
- The BF16 checkpoint, experts streamed, is the lossless KLD teacher for every pack ([quality-kld](quality-kld.md)).

<a id="dfly2-parity"></a>
**DFly2 without a reference implementation.** A drafter cannot change what the target emits, only how fast. Its
checks are therefore two kinds: exactness, which needs no reference, and acceptance, which reads the spec's own
claims off the release checkpoint.
- **Exactness (the tiny, random drafter weights).**
  - Greedy rounds give the serial greedy tokens byte for byte, at every accepted count, text and image.
  - Seeded sampled rounds give the seeded serial request's tokens under the exact accept rule.
  - A partial accept rolls the target's rows, open groups and Engram history back as DSpark's verify does.
  - The drafter's reads of the shared layers' rows and picks equal the target's own, row for row.
  - These hold whatever the drafter's weights are, so a random tiny proves them.
- **Acceptance (the release checkpoint).** The trained wiring is the one the weights fit, and a wrong wiring
  drafts near-misses that verify rejects.
  - For each open wiring choice, the candidates run behind a test-only switch on a held prompt set (code, prose,
    reasoning, agent traffic). The one with the clearly highest accepted tokens a round is kept, and its margin over
    the next is recorded.
  - Each component the spec names (the fusion gate, the hidden correction) is switched off once. It must lower
    acceptance; a component whose removal raises it is wired wrong.
  - The confidence head's output is held against the measured acceptance per slot. A head that ranks the slots as
    they verify confirms its inputs are the trained ones.
  - Where the margin between candidates is too small to call, the vLLM and SGLang images the README names are the
    only other public trace of the wiring.

<a id="open-questions"></a>
## Open questions

1. **Engram norm ε**: torch's unset ε for a BF16 row is 2^-7, against llama.cpp's fp32 machine ε. Which one the
   checkpoint was trained with decides the parity bar of every Engram layer. Still open: the real-checkpoint parity
   run (part 1r) decides it.
2. **Engram FP8 in the GGUF**: which 8-bit type the table shard uses, and how its codes were scaled (one byte an
   element, no row scale stored).
3. **Indexer ties**: torch's `topk` order among equal scores. Exact ties are common, since relu zeroes whole groups.
   DeepSeek's kernels tie to the lower index.
4. **Vision**: the encoder's GELU (erf in transformers, tanh in llama.cpp), and the unread `rope_2d_*` config keys.
5. **The streams' dtype between sublayers**: BF16 as the reference rounds them, or f32 as llama.cpp keeps them;
   Cooler follows the reference.
6. **Batched decode**: a streamed load serves one request at a time, as DeepSeek's does. Joining requests as rows on a
   fully resident pack follows DeepSeek's `coolerRows`, after the core.
7. **Sushi's `max` effort word**: the ruling maps `off` through `xhigh`. Whether `max` also takes the template's
   `max`, or is refused, is the owner's call.

## Implementation plan

Row 1 of the 1.8.x board. Each part is one reviewable unit, about a day or less, that lands green on its own: its
test first (TDD), then the subset of the suite its diff reaches. The seam parts come first. Parts 1c to 1h wait until
1.7.0 ships, because they move DeepSeek's release kernels.

| key | part | depends | done when |
|---|---|---|---|
| 1a | R1: the `cooler` declaration, an opt-out or none | | a Cooler-only declaration resolves to the engine in a unit test; DeepSeek's and Qwen's sides, `--help` (`src/fixtures/help.txt`), flags and settings unchanged; lint green |
| 1b | R2: `src/engram.zig`, hash and table reader shared, `Geometry` with hash ids and seed | | DeepSeek's Engram tests pass from the shared file unchanged; a Hy4-geometry test reproduces the config's 64 primes and the reference's multipliers for ids 2, 12, 22, 32; DeepSeek tinies byte-identical |
| 1c | R3, step 1: the hyper-connection stage kernels move to a shared module, unchanged | 1.7.0 shipped | DeepSeek tinies and the hc kernel tests byte-identical; pipeline names unchanged |
| 1d | R3, step 2: `comb` identity, post magnitude and eps, mix count | 1c | an identity-comb case equals an f64 chain; DeepSeek bytes unchanged |
| 1e | R4, step 1: index score and top-k kernels shared, unchanged | 1.7.0 shipped | the saved scans (`DSV41_INDEX_SCANS`) bit-identical; top-k tests unchanged |
| 1f | R4, step 2: BF16 pooled-key arm | 1e | BF16 keys score as the f64 truth within the scalar kernel's bar; top-k ties to the lower index |
| 1g | R5, step 1: window-and-picks attention shared, unchanged | 1.7.0 shipped | DeepSeek attention tests bit-identical |
| 1h | R5, step 2: `dk` 576 / `dv` 512, no fold | 1g | the new geometry within the fused kernel's bar against the CPU f32 core |
| 1i | arch skeleton: `src/hy4/` (`arch.zig`, `hy4_config.zig`, registry line, `cooler` with no opt-out, `DROPPABLE`) | 1a | the released config parses (the three layer-type spellings, refusals by name); the drop test and arch slice pass |
| 1j | the oracle: the dumper in `tests/hy4/` and the tiny fixture | 1i | the fixture loads in the dumper's own round trip; streams, captures and logits dumped |
| 1k | loader: resident trunk by the release's names, f32 widenings, fused `gate_up` views, streamed experts, Engram tables by row (the BF16 and row-scaled FP8 arms), `lookup_table` as the token map | 1b, 1j | the tiny loads on Sushi Cooler with no MLX call (census); resident bytes equal the bill |
| 1l | forward on the tiny: embedding, iHC, dense layer 0, head | 1d, 1k | the first layer's streams and the head within the bars |
| 1m | Engram on the tiny | 1l | row ids identical; delta within the bar; zero at image and pad ids |
| 1n | sliding MLA layers on the tiny (ring, sink, gate, sandwich) | 1h, 1l | sliding layers' traces within the bars, prefill and decode |
| 1o | sparse layers, dense phase, and the latent MoE on the tiny | 1n | every layer within the bars at contexts under the budget; router picks identical |
| 1p | indexer pooling and selection on the tiny | 1f, 1o | pooled keys, scores and picks identical past slot 16; logits within 2e-3; chunked prefill gives the same state |
| 1q | decode, chunked prefill, the request loop, chat face and tool-call parse | 1p | greedy tokens after a prefill equal the oracle's; the template byte-exact against `apply_chat_template` on its cases; format-corpus entries for the tool-call spelling; the effort words render and cap as ruled (`off` → `no_think`, `low` → `high` + 2,048, `medium` → `high` + 16,384, `high` → `high` uncapped, `xhigh` → `max`), pinned by a test |
| 1r | the real checkpoint against llama.cpp | 1q | greedy tokens agree up to a recorded near-tie on the prompt set; logits cosine and top-k overlap recorded; each listed difference decided, the Engram norm ε among them |
| 1s | bills: per token, per request, per chunk, exact bytes plus the margin | 1q | bill pins on the committed config; a live bill test holds the engine's pool peak under each bill |
| 1t | warm and the pipeline manifest | 1s | a request compiles no kernel (`compiled inside a request` 0); `[pipelines]` cases as DeepSeek's |
| 1u | prefix cache: restore points, row segments, namespace, live rollback | 1s | a restored turn's state and logits equal a cold prefill's, byte for byte, both tiers |
| 1v | vision: tower parameters, merger, M-RoPE positions | 1q | merger output and the first image row's logits within the bars against the oracle's image case |
| 1w | pack format, STQ only: the `pack-format` section, STQ experts and trunk, the Engram table forms | 1r | a tiny STQ pack loads and serves; served against its dense twin within DeepSeek's STQ bars; KLD measured against the BF16 teacher |
| 1x | DFly2 on the tiny: the loader (`model.dfly_layers.*`), the forward over the shared layers' rows and picks, DSpark's rounds with a mask block of 6 | 1u | greedy rounds equal serial greedy byte for byte at every accepted count, text and image; seeded sampled rounds equal the seeded serial request; rollback restores rows, open groups and Engram history; the drafter's reads equal the target's rows and picks; its bill adds weights and one round's block rows, no KV |
| 1y | KV width: the 8-bit latent (e4m3 tiles of 128 with f32 scales, rope dims BF16) as the default, BF16 also served | 1u | both widths serve; a row's 8-bit round trip matches the format's reference within one e4m3 step; KLD of 8-bit against BF16 KV recorded (16x512 fixture, BF16 teacher); bills pinned for both widths (8,640 and 14,592 B a token); the prefix-cache namespace names the width and a restore never crosses it |
| 1z | DFly2 on the release checkpoint: the open wiring settled, the adaptive round policy | 1x, 1r | each wiring choice's accepted tokens a round recorded with its margin; every spec-named component lowers acceptance when switched off; confidence ranks the slots as they verify; the round policy never loses to serial decode on the prompt set |
