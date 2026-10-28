# DeepSeek-V4.1-Flash (`deepseek_v41`)

The fourth served architecture: the release's own checkpoint (FP8 trunk, FP4 routed experts, Engram tables), or an STQ
pack of it, served with the experts streamed from SSD. What it loads, how it is billed, what is proven against the reference and
what is not.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-expert-streaming](engine-expert-streaming.md),
[engine-memory-admission](engine-memory-admission.md), [quality-kld](quality-kld.md).

## Scope

- It serves `deepseek_v41` from the ORIGINAL checkpoint (`config.json`, `model-*.safetensors`, the index). Text in,
  text out, one request at a time (the model holds the request's state: `ownsModuleDecodeState`).
- Off, by name: `--kv-quant` (the state is the activation dtype;
  `kvCacheFor` answers dense; a launch `--kv-quant`, `--qwen-gdn 16`, `--mtp-head-kv-quant`, `--mtp-typical`, `--mtp-tokenv3` or `--mtp-greedy-tail` refuses the load (DSpark accepts by the exact rule only), a settings `kv_quant`, `mtp_acceptance` or `mtp_greedy_tail` is ignored with one line). The prefix cache reuses a prompt's prefix ([below](#prefix-cache)). Speculative decoding is DSpark
  ([below](#dspark)), the checkpoint's own drafter, on by default like every served MTP head; `--no-mtp` turns it off.
- Routed experts always stream: 289 GB of FP4 against a 128 GB box. `--ssd-budget-gb` (or `ssd_budget_gb` in
  model-settings) is required, as on every streamed load; without one the load fails with `ExpertStreamingRequired`.
- Thinking: `low`, `high`, `max` (the encoder's ladder, rendered as a 50/75/100 budget), default `high`; `medium` is
  served as `high`; a request that names no thinking or effort thinks, at `high` (`ModelConfig.defaultEnableThinking`; an explicit field wins, `enable_thinking: false` or effort `off` is a `chat` render). The template is [src/fixtures/dsv41_chat_template.jinja](../src/fixtures/dsv41_chat_template.jinja),
  pinned byte for byte to the release's `encoding/encoding.py` ([below](#oracle)).
- Tool calls: DSML with V4.1's leading-space tag names (`<｜DSML｜ invoke name=…>`); the parser takes both spellings.

## Checkpoint geometry

| Part | Shape |
|---|---|
| Trunk | 40 layers, hidden 5120, vocab 129280, `max_position_embeddings` 1,048,576 |
| Attention | 64 heads, ONE KV head, `head_dim` 512, rope 64 (YaRN ×16 over 65536 for the compressed layers), q LoRA 1280, grouped output LoRA 1024 × 8 groups, per-head sink, window 128 |
| Compression | `compress_ratios` per layer: 0 (window only), 2 (layers 2–19), 1 (layers 20–39). KV sources 2, 8, 14, 20 write the compressed rows every compressed layer below the next source reads; index sources 2, 8, 14, 20, 24, 28, 32, 36 pick the top-512 rows; owners (a layer that is both) keep index keys. Layer 20 is the candidate source (2048 blocks of 8) |
| FFN | every layer MoE: 384 routed experts, top-6, width 2304, one shared expert; `sqrtsoftplus` scores, the f32 bias picks only, normalized, ×1.5; gate clamp 10 |
| Hyper-connections | 4 streams, 20 Sinkhorn iterations, `hc_eps` 1e-6; two collapses and expansions per layer |
| Engram | layers 1 and 14: 384M rows × 256 FP8 each (~100 GB per table), 8 heads, 4-grams over a 99092-entry compressed vocabulary, hashed with numpy-seeded multipliers and primes |
| Loaded by default, skippable | the vision tower (`vision.*`, `aligner.*`, `image_*`, bf16 on disk, 0.97 GB; held as affine 8-bit, 0.50 GB) and each layer's `bias_vl`, unless `--no-vision` ([image input](#image-input)); the 3 DSpark stages (`mtp.*`) only without `--no-mtp` |

Storage in the checkpoint: trunk linears `F8_E4M3` with `F8_E8M0` scales on 32×32 blocks; routed experts `I8` holding
two e2m1 nibbles (low first) with `F8_E8M0` scales per 32 columns; `embed`, `head`, gate bf16; hyper-connection
mixes, sinks and the gate bias f32. Each backbone layer sits in its own shard (~7.4 GB, of which ~170 MB is trunk).

## What loads

- **Resident trunk** (`src/dsv41_source.zig`, ONE decision in `model.loadWeightsForConfig`): every tensor but the experts,
  the Engram rows, the MTP/DSpark tensors and (under `--no-vision`) the vision/aligner/image tensors and `bias_vl`, read by name from the shard headers. FP8
  linears become MLX `mxfp8`: the u8 codes viewed as u32 and each 32-row scale row repeated down its 32 rows. FP4 experts
  are MLX `mxfp4` exactly (the int8 viewed as u32, low nibble first; proven bit for bit against a numpy decode of a real
  expert). The head is widened to f32; a dense embedding stays bf16. An affine embedding (an STQ pack's 4-bit g64 triple, `embed.weight/scales/biases`) is [read from the shard by row](#embed-ssd), not loaded.
<a id="embed-ssd"></a>
- **Token embedding on SSD** (`src/dsv41_embed.zig`): a pack's affine triple (`embed.weight` u32, `embed.scales`, `embed.biases`)
  is left out of the resident load and of the trunk term (`dsv41_source.embedOnSsd`, the one test for both), 0.37 GB on the
  STQ pack. A forward `pread`s the rows of its token ids (`Ssd.read`: each distinct id once, 16 rows a thread, `F_RDADVISE`
  first, page cache on, read-ahead off) and `Embed.rows` dequantizes them with the resident call, over the same stored bytes
  (scales and biases cast to the activation dtype as the loader cast them), so the rows are bit-identical. The page cache
  keeps the common tokens' rows hot. A prefill chunk's pass is told the prompt tokens after it (`ForwardCtx.ahead_ids`)
  and reads the next chunk's rows on a thread while it runs (`aheadEmbed`, ~2.3 KB a token of host memory, unbilled);
  the next pass takes them when its ids are the same, else reads its own. Decode and a DSpark draft read their few rows
  inline. The DSpark draft embeds `[t1, noise_id...]` through the same table; MTP stages, the head and the vision tower
  never touch it. A dense embedding (the FP8 checkpoint's bf16 table) stays resident. Load logs
  `[dsv41] token embedding on SSD (0.37 GB freed to the expert cache)`. Live (873b0fd4 vs 09977e23, budget 108, AC, fans
  max, one A then one B): +1 slot a layer (270 -> 271 serial, 260 -> 261 DSpark), 512-token decode byte-identical at
  46.8 / 46.9 tok/s serial and 52.0 / 52.2 DSpark, cold 32,888-token prefill 792.7 / 790.4 tok/s.
- **Routed experts** (`src/dsv41_stream.zig`): the shared expert cache reads each layer's picked experts from the shards
  into slabs (layout `mxfp4_w123`, `layers.L.ffn.experts.E.w1|w3|w2.{weight,scale}`; w1 gate, w3 up, w2 down) and the
  layer gathers them as `mxfp4`. The slab path gives the resident path's logits bit for bit on a tiny pack, with a cache
  that holds every expert and one that does not.
- **Engram tables** (`src/dsv41_engram.zig`): never resident. Each lookup `pread`s its rows from the shard that holds the
  table (page cache on, read-ahead off, `F_RDADVISE` first so a batch costs about one SSD latency) and dequantizes them
  as `mxfp8`. A row cache per table (`RowCache`, 8-way set-associative, least recent out) keeps 1M rows; on 1M
  tokens of documents in turn it serves 31% of reads at 64k rows, 42% at 1M and 47% at 4M
  (`SUSHI_DSV41_ENGRAM_REUSE=<tokens>` replays a u32 token file). Every pass, decode included, reads its rows on a thread from
  `beginPass` (`prefetchEngram`): a decode token's layer-1 rows hide behind layer 0, its layer-14 rows behind layers 0-13. A batch reads each
  distinct row id once and copies it to every place it sits (`Table.read`). A prefill chunk's pass is told the prompt tokens after it
  (`ForwardCtx.ahead_ids`) and reads the next chunk's rows while it runs (`aheadEngram`, one chunk of host rows, ~13 KB a token, unbilled);
  the next pass takes them when its row ids are the same, else reads its own. The n-gram hash and the compressed-vocabulary map
  are the reference's (`engram.py`); the map ships embedded (`src/fixtures/dsv41_engram_token_map.bin`) and a pack's own
  `engram_token_map.json` outranks it. `tests/dump_dsv41_token_map.py --check` re-derives it from `tokenizer.json`.
- **Forward** (`src/dsv41_model.zig`): bf16 activations, f32 where the reference is f32 (head, compressor, hyper-connection
  mixes, attention scores and sink, router). Prefill runs in spans of 256 rows; the window is a ring, the compressed and
  index rows grow by a sixteenth (`rows_grow_div`); decode is the one-row case of the same code.
- **Indexer top-k, streamed** (`indexStream`): a wide span (more than 16 rows) with more index keys than one tile
  (`index_stream_tile`, 32768) scores one tile of keys at a time with the matrix kernel (unpacking only that tile) and folds
  each tile's top-k into a running top-k, so no `[rows, keys]` scores exist. The candidate source folds its block maxima
  the same way into its mask, and later layers mask each tile with theirs. The picks are the whole row's, order and ties
  included: the running ids precede the tile's, so the merged row's ties go to the lower id as `dsv41_topk` does. A
  tile scores the bits of the whole list's columns (`indexScoresTile` takes the tile's key offset, a multiple of the
  kernel's key tile). A decode-width block (up to 16 rows) scores the packed index keys directly
  (`indexScoresPacked`), with no unpacked copy. A trace run keeps the whole scores.
- **Decode indexer cost, by context** (`SUSHI_DSV41_INDEX_UBENCH=1 -Dtest-filter="dsv41 decode index stage timing"`): the
  score pass is one thread a key and linear in keys AND rows (the FMAs and threadgroup loads of q, not the key reads, bound
  it); the top-k is one threadgroup a row. A plain step at 209k tokens cost ~4 ms more than at 5k (serial `--no-mtp`, 46079949:
  40.0 against 47.8 tok/s), nearly all of it the eight index sources. A decode block's later index sources (layers 24-36)
  score only the candidate source's picked blocks (`indexScoresGather` over the block ids `candidateBlockIds` hands down,
  then `candidateKeys` maps the picks back): the picks are the masked whole scan's, ties included, while the blocks hold
  more reachable keys than a query keeps (05779ced: 42.0 tok/s at 209k, 47.6 at 4.7k, the same 600 greedy tokens).
  The packed and gathered score kernels read q four channels a load (each head adds the channels in the same order):
  262144 keys, 1 row 0.74 to 0.44 ms, 4 rows 2.35 to 1.27 ms per eval (about 0.2 ms of each is the eval floor); live serial b349f67e 42.8 tok/s at 209k, 47.8 at 4.7k, the same 600
  greedy tokens (one boot, GPU lock, QUIET, fans max, die 49 C, taskpolicy -a). Ruled out
  on the ubench: a 512-thread score threadgroup and a replicated top-k histogram (both unchanged).
- **Top-k threadgroup width** (`topk_tg`, `probeTopk` at load): 1024 threads, halved where the GPU caps the kernel lower (M1/M2: 832); the picks are width-independent (`dsv41 topk: the picks do not depend` test).
- **Decode top-k, split** (`topkIndicesSplit`, `topkSplits`): a decode block (up to 16 rows) of at least 131072 scores (262144 past
  one row) with k <= 512 splits each row over 8 chunks; each chunk's threadgroup selects its own top-k and one merge
  threadgroup selects over the chunks' picks, which are ascending in row index, so the indices, their order and the ties
  (lower index) are the one-threadgroup kernel's (`dsv41 topk split` test: random, 8-level tied and -inf rows, n to
  262144). `SUSHI_DSV41_TOPK_UBENCH=1 -Dtest-filter="dsv41 topk split timing"`, per row of work with eight scans in one eval,
  best of 15, splits 1 to 8: n 262144 1 row 0.065 to 0.041 ms, 16 rows 0.186 to 0.149; n 524288 1 row 0.127 to 0.051
  (16 rows 0.413 to 0.279); n 131072 1 row 0.055 to 0.042, 4 and 16 rows a tie. n up to 65536 and the k = 2048 block maxima
  tie or lose a little at any split (the merge costs what the split saves); 32 splits lose everywhere. Output is
  bit-identical, so the gate is the unit test (47f5d3c1's timing run, 487ecb62's kernel; GPU lock, QUIET, fans max, die
  under 50 C, `taskpolicy -a`, server down). Eight chunks is the best count at every size that splits; 16 is 5-10% slower.
  The matrix-unit decode index score kernel that the same run timed (20-30% slower than the scalar one) is ruled out.

- **Tile-major expert pools**: the streamed STQ pools hold each trellis projection per output tile pair
  (`layout.tileMajor`, [engine-exl3-experts](engine-exl3-experts.md#rate-groups)), filled through the bounce buffers and
  read by the decode GEMVs and the three prefill GEMM bodies with a new first tile and k stride; the pack on disk and every
  plane's bytes are unchanged. The chain is 1.6-3.3% faster than pack order, not the ~25% the no-trellis-loads ablation
  allowed: the scatter of the 96-192 B chunks was not the limit (`SUSHI_DSV41_REPACK_UBENCH=1`, exl3 test binary, pack order
  against tile-major interleaved per round: 1 row 130.2 to 128.1 us at L2, 134.0 to 130.6 at L23, 4 rows 464 to 449 us at
  L23; GPU lock, QUIET, fans max, die 49 C, `taskpolicy -a`, e4c87be5). Live, ba4abafc, serial `--no-mtp` at 4.7k, the same
  600 greedy tokens: 48.3 tok/s against 47.9 (d57f8ea8, one boot each); the first request's first prefill chunk and
  last-token step ran 0.5 s and 0.7 s longer (595.6 against 700.9 tok/s; chunks 2 and 3 equal at 2508 and 1019 ms; not
  isolated, likely the new kernels' first compile); a cold prefill fill reads 6.9-8.8 GB/s against 7.5-9.3. Adaptive DSpark 61.0 against 62.5 with a
  different row mix (1.46 against 1.71 tokens a round), inside the boot-to-boot spread.
- **Hyper-connection folds**: the FFN stage (`dsv41_hc_stage_z`) splits the attention's mixes itself, one simdgroup in each
  threadgroup, so the attention side has no `dsv41_hc_split` dispatch (a layer less; on e4c87be5 serial `--no-mtp` 47.9
  tok/s at 4.7k against 47.8, the same 600 greedy tokens: a tie that removes 40 dispatches; one boot, GPU lock, QUIET,
  fans max, die 48 C, taskpolicy -a). The FFN-side split and expand stay: the FFN output's `hcExpand` is the MLX op chain,
  whose f32 matmul is TF32-class (rms error 7.6e-4 against the f64 truth, the `_x` kernel's arithmetic 4.6e-8; 9% of bf16
  and every f32 element differ), so folding it into the next stage changes output bytes and needs the KLD gate.

### QAT round trips

The reference fake-quantizes at five places and the forward applies each (`src/dsv41_qat.zig`, Metal, round-half-even,
the reference's f32 reciprocal constants): the window KV after rope (FP8, ue8m0, 32), the compressed latent (FP4, e4m3
scale, 16), the indexer key and query (FP4, ue8m0, 32) and the attention output (FP8, ue8m0, 32). They are bit-exact
against the reference's own kernels on a midpoint-rich fixture.

## Tokenizer

The release's pre-tokenizer is a `Sequence` of three isolating Splits (`\p{N}{1,3}`, a CJK class, a punctuation grammar)
and a ByteLevel that does not split. `src/tokenizer_deepseek.zig` runs it exactly, each Split on the pieces the one before
left, so a whitespace run at a piece's edge stays whole; it is selected only when the tokenizer.json spells those three
regexes verbatim. It needs `\p{P}` and `\p{S}`, which `scripts/gen-tokenizer-unicode.py` now emits beside letters, marks and
numbers. Pinned to HF `tokenizers` by `deepseek pre-tokenizer` (21 inputs, hermetic) and by the real-pack parity in
`tests/test_tokenizer_reference.sh`. The embedded Engram token map matches the real tokenizer (129280 tokens to 99092 ids).

## Memory and bills

A bill follows the storage in the same commit; every term below is an ESTIMATE until the first real load measures it.

| Term | Bill |
|---|---|
| Resident trunk | `dsv41_source.residentBytes`: stored bytes of every kept tensor, ×2 for the head (f32), ×32 for each scale grid; ~11–12 GB |
| Expert cache | `--ssd-budget-gb` less trunk, workspace, selected slab and bounce: the ledger of every streamed arch. One expert is 18.8 MB (a layer's 384: 7.2 GB, the prefill union workspace) |
| State per token | `Config.stateBytesPerToken`: 890 B, the release's own: each KV source's compressed rows as FP4 with an E4M3 scale per 16 (288 B a row) and each owner's index keys as MXFP4 (68 B), at their ratio. Stored as the QAT round trips left them (`qat_mod.packFp4`) and unpacked on read bit for bit: a span's picks only, the index keys a tile at a time for a wide span and not at all for a decode block. With the round trips off (tests) the rows stay in the activation dtype. ~0.93 GB at 1M tokens |
| State per request | `Config.stateFixedBytes`: the 40 window rings in the activation dtype (128 x 512 x 2 B each) and the open pooling groups (up to ratio - 1 rows of kv and gate, f32); with DSpark the stage windows and the rings' verify slack |
| Forward transient | `Config.forwardCap(rows, seq)` / `chunkCap(chunk, seq)`: the transient cap a forward runs under and admission bills ([dsv41-memory](dsv41-memory.md#transient-caps)); 2.3 MiB a prefill row, the decode block's scores and the rows' growth per context token, a quarter over, 256 MiB slack |
| Fill peak | none: slabs are filled in place, the union workspace (a layer's 384 experts) is billed once at load beside the cache and bounce (`scheduler.loadStagingBytes`) and sits in active memory after |
| Indexer peak by context | `SUSHI_DSV41_INDEX_UBENCH=1 zig build test -Dtest-filter="dsv41 index stream peak"`: MLX `peak - active` of one span at the release shapes (32 heads x 128, top-512, blocks of 8). 256 rows at ratio 2: whole scores 12 / 58 / 211 / 415 / 823 MB at 16k / 64k / 256k / 512k / 1M context, streamed 13 / 58 / 81 / 81 / 81 MB. The 128-row CED tail at ratio 1: whole 13 / 62 / 239 / 476 / 948 MB, streamed 13 / 44 / 44 / 44 / 44 MB. A 16-row decode block over the packed keys: 4 / 16 / 32 / 64 MB, 9 / 36 / 72 / 144 MB with a candidate mask. A row store's growth holds 1.06 x its bytes beyond them (306 MB for 1M compressed rows, 72 MB for 1M index keys; 1.25 x at the old quarter). Synthetic keys, no model; the live per-chunk peak on the pack is the check |
| Measured transient | Per-chunk MLX `peak - active` (`SUSHI_PREFILL_CHUNK_PEAK`) of the CED prefill on an STQ pack: 2.63 GB at 512 rows, 3.56 GB at 1024, 4.05 GB at a 1507-row last chunk, nearly flat in the context (512 rows: 2.64 GB at 18k, 2.60 GB at 108k, about 0.6 KB a context token). The bill uses `chunk + 511` rows, 400 B a row per hidden channel (2.0 MB a row) and 3 score copies per context token; the bracket test pins bill >= 1.1x each measurement and <= 4x. Before the per-span scopes and CED: ~6.2-6.9 MB a row and 33 KB a context token. These per-chunk numbers predate the streamed top-k; the bill is refitted from the indexer table above, so re-measure the live peak before trusting a margin |
| Engram | each table's row cache (`dsv41_engram.cacheBytes`: 1M rows at 272 B, 0.57 GB for both), billed with the trunk, so the expert cache gives it up; the page cache carries the rest |

`kvBytesPerToken`, `sessionBytesPerToken`, `slotRingBytes`, `prefillTransientReserveAtKv` and `prefillNeededAtChunk` answer
these for `deepseek_v41`, so auto context, the prefill chunk ladder and admission read one set of numbers.

<a id="ced"></a>
## CED prefill (the decoder runs the tail only)

DeepSeek-V4.1 is a causal encoder-decoder. Layers 0 to 19 encode every prompt row. The split layer (20, the last KV
source) compresses every row. Layers 20 to 39 decode only the last `window` (128) rows of each prefill chunk, and
their attention band is floored at the tail's start. This is the architecture as defined, not an approximation;
there is no full-forward arm. Port of mlx-serve's `cedSplit`/`cedTail`/`cedReplayIn`.

When it engages:
- `Config.cedSplit` picks the split from the config, or returns null. The split must be an index source with ratio-1
  layers after it. Engram layers must sit below it. DSpark targets and the candidate source must sit at or above it.
- A prefill pass engages CED when it wants only the last row's logits, has no verify, and is wider than the window
  (`Model.planCed`).
- Load line: `[dsv41] CED: layers 20..39 run the last 128 rows of a prefill chunk`. Each chunk logs
  `[dsv41] CED prefill: N rows encoder-only, replay 128 rows` at debug.

Proofs (`DSV41_TINY`, CPU stream, f32, QAT off):
- The encoder's state slots are bit-identical to the full forward's.
- The decoder's compressed rows equal the full forward's.
- The same prompt in other chunkings gives the same state, logits within 1e-4, and the same argmax (measured 6e-6).
- DSpark and decode continue after a CED prefill.
- The reference-oracle tests (decode after prefill, DSpark QAT off) compare against the reference's full forward,
  so they set `ced_split = null`.

Prefill width:
- DeepSeek picks its chunk per request: 2048 (`dsv41_config.prefill_width`) while that request's bill fits, then 1024,
  then 512. Load line: `Prefill chunk: per request, up to 2048 at a short prompt (...); load-time fallback N`. On an STQ
  pack the lendable decoder pools cover the working set, so every request takes 2048 and a chunk that does not fit
  releases pools instead of narrowing ([borrow](dsv41-memory.md#borrow)); the rungs below are the FP4 release's.
  - The advertised context is priced at 512, so the wider start leaves it unchanged (701440 at budget 96).
  - At 8k and longer a 2048 bill is the 1024 bill plus 2500 MiB (1024 more rows at 400 B a channel, x 5/4). At
    budget 96 (13020 MiB to serve) prompts up to ~413k take 2048, up to ~642k take 1024, and the rest take 512.
  - Measured transient at 2048 rows: 4.35 GB, and 5.15 GB on a 2524-row last chunk (8.7k prompt), inside the bill.
- Its attention scores only each row's picked keys, so the dense score-sheet narrowing (`boundedPrefillChunk`) never
  applies (`prefillScoreHeadDim` is 0). Before this change, every prompt past ~32k ran at 512.
- The advertised context is the longest prompt its own request bill admits at the fallback width
  (`memoryContextAtChunk`). Past the indexer's scores (now streamed) the bill grows with 0.9 KB of state, 0.6 KB of
  growth and decode scores, and (SSD-first) the restore points in flight to the tier, so `--ssd-budget-gb 96` (13020
  MiB to serve) advertises 974848 (1M raw, less the auto margin) where the whole-scores bill advertised ~740k. With the
  borrow, budget 117 still advertises it on wired 124000 ([borrow](dsv41-memory.md#borrow)).
- An explicit `--prefill-chunk` outranks both.

Other pieces:
- A wide chunk's Engram rows are read ahead on a host thread over the whole chunk (`prefetchEngram`). They are
  dequantized at their layer, bit-identical to rows read there.
- The next chunk's Engram rows are read during this one, once this chunk's own are in RAM, so layer 1 of the next chunk
  finds its rows already read; a repeated row id in a batch is read once. Both are bit-identical to reading at the layer.
  Measured at bf21aa5e against 646ec552 (sushi bench, STQ 1.75bpw, `--ssd-budget-gb 108`, debug log, one A then one B): the prose
  arm's Engram wait per 2048-row chunk fell from 225 ms to 1.2 ms (median), its 0-128k cold prefill from 770 to 815 tok/s
  (8k windows +5-10%); code, whose n-grams hit the row cache, waited 1 ms before and is unchanged (911 vs 909 tok/s).
- Attention runs each span and each attention or FFN block in its own scope, so MLX frees the f32 gathers and logits
  copies between spans.

Measured on an STQ pack, M5 Max, `--ssd-budget-gb 94 --prefill-chunk 1024`, one A then one B after 180 s idle:

| prompt | full forward | CED |
|---|---|---|
| 8.7k tokens, cold | 179 tok/s | 286 tok/s |
| 8.7k tokens, warm | 258 tok/s | 482 tok/s |
| 26k new tokens after an 8.7k cached prefix (admission ran 512-row chunks) | 208 tok/s | 355 tok/s |

- A steady warm 1024-row chunk took 3.7 s before and 2.0 s after.
- The prefill transient (`peak - active`) fell from 7.08 to 3.56 GB at 1024 rows, and from 10.05 to 4.05 GB at a
  1500-row last chunk: about 6.2 MB a row before, 1.0 MB after.

Where a chunk's time went before CED, from the synced `SUSHI_DSV41_PREFILL_PROFILE` stage profile at 512 rows:
routed experts (fill and grouped GEMM) 52-55%, attention core 16%, Engram host reads 8%, `attn_out` 5%, indexer
3.5-9%, the rest under 3% each. The trellis trunk GEMM runs at the speed of MLX's mxfp8 and affine4 GEMMs at 512 rows
([engine-exl3-experts](engine-exl3-experts.md#dsv41-stq)).

After CED and the fused gather-attention (budget 96, chunk 1024, synced stage profile, ms per chunk):

| stage | warm encoder (rows 0-19, 1024 rows) | cold encoder | decoder (rows 20-39, 128 rows) |
|---|---|---|---|
| expert fill (SSD reads ~190, union host copies ~175, rest waits) | ~600 | 850-1090 | ~30 |
| routed grouped GEMM | ~560 (~15.5 TFLOP/s, ~77 GB/s of expert bytes) | 700-900 | ~140 (~7.8 TFLOP/s) |
| attention core (fused) | ~205 | | ~28 |
| `attn_out` / Engram / indexer / HC FFN | ~90 / ~86 / ~64 / ~64 | | ~16 / - / ~12 / - |
| expand / q / shared expert / HC attn / `attn_in` / router | ~47 / ~46 / ~41 / ~19 / ~15 / ~8 | | |

- A warm 1024-token chunk is ~1.85 s encoder plus ~0.29 s decoder. The routed experts (fill and GEMM) are ~55% of it.
- A 2048-token chunk ran warm 8.7k at 553 tok/s against 419 at 1024 (same profile, `--prefill-chunk 2048`): each
  expert is read and run once per 2048 rows.
- A layer-ahead read of the next layer's experts (its router on this layer's rows) guessed ~28% of what it read and
  ~24% of the picks; a cold novel 10k prompt ran 294 tok/s without it and 296 with it, so it was cut.

<a id="oracle"></a>
## Oracle and what is proven

`tests/dump_dsv41_fixtures.py` runs the release's `inference/model.py` on the CPU in f32 over a TINY random model (8
layers, hidden 128, ratios 0,0,2,2,2,1,1,1, 8 experts top-3, window 8, Engram at layers 1 and 2) written in the
ORIGINAL checkpoint layout, and dumps logits, per-layer streams and sub-module captures (`fixture.safetensors`, QAT on;
`fixture_noqat.safetensors`, off). The reference's GPU kernels are stubbed on the CPU. `DSV41_TINY=<dir>` points the Zig
tests at the output; `zig build test` caches a passing env-gated run, so re-run the test binary to re-execute.

`DSV41_TINY` takes the DEFAULT-geometry dump (no `--geometry`); a `--geometry stq` dump there fails `dsv41 fused: prefill ...` at `attn_2` (diff 0.39, bar 0.010). `DSV41_STQ_TINY` takes the STQ PACK built from the `--geometry stq` dump, not the dump itself.

| Test | Bar |
|---|---|
| prefill, whole and in 13-row chunks over 5-row spans, QAT off, CPU stream | every layer's trace within 1e-4 of its peak, logits 2e-3 |
| decode one token at a time after a prefill | logits 2e-3 |
| QAT on | layers 0–1 within 1e-3 of the reference, logits cosine ≥ 0.99 (a rounding flip is a jump, so only the first layers are tight) |
| QAT kernels, hyper-connection split kernel, rotary | bit-exact against the reference / the op chain |
| Engram hash, table, prefetch, token map | the reference's multipliers, primes, row ids; the table dequantizes as `mxfp8` |
| slab experts vs resident experts | identical logits |
| DSpark draft: ids, Markov-biased logits, confidence, main rows | QAT off 1e-4 / 2e-3 of the peak; QAT on cosine > 0.99 from the reference's rows |
| DSpark verify rollback at every accepted count, a second round, the next draft | serial logits 2e-3, serial draft |
| `nextDspark` through the Generator on the GPU in bf16 | the budget, the stats and the first token; the later tokens may break a near-tie against serial decode (DSpark rows are the same at every width, and the serial step sums as the SIMD one-row arm) (the exact-stream rows above pin the identity) |
| DSpark bills | resident bytes equal the held arrays; window and slack bytes equal the bill |
| `Transformer.forward` through the shell | identical to the model's own, across prefill and decode chunks |
| chat template | 14 cases byte for byte with `encode_messages` |

The tight bar runs on the CPU stream (`dsv41_model.exact_matmul`, dequantize plus a plain matmul): MLX's GPU f32 `matmul`
keeps about 11 bits (3e-3 absolute on values of 4) and its `quantized_matmul` with an f32 input is only ~5e-4 relative.
Serving is bf16, where neither matters.

<a id="fused"></a>
## Fused kernels

`src/dsv41_kernels.zig` holds Metal kernels, each replacing a chain of MLX ops of `dsv41_model.zig`; the chain stays as
the reference of its test and as the arm for a scope off the GPU stream or `Model.fused = false`. They are ported from
mlx-serve's `deepseek_v41.zig` (49bb6be0) and adapted to the original layout: experts are separate `mxfp4` gate, up and
down banks (upstream fuses affine ones), the shared expert and trunk are `mxfp8`, rotary takes sushi's `freqs` and the
exp/cos/sin are MLX's (`precise`), so an elementwise kernel is bit-identical to its chain.

| Kernel | Replaces | Engages | Proof |
|---|---|---|---|
| `dsv41_moe_gu`, `dsv41_moe_dn` | routed gate/up/SwiGLU/down, sum, shared expert, hyper-connection expand | one token (decode) | no further from an f64 truth than the chain; bf16 within 2 ulps of its peak |
| `dsv41_moe_act`, `dsv41_moe_combine` | the SwiGLU x weight and the unsort/sum/shared add of the sorted-gather MoE | many rows (prefill) | bit for bit |
| `dsv41_gate`, `dsv41_route`, `dsv41_topk`, `dsv41_topk_chunk`, `dsv41_topk_merge` | router logits (the router read as stored, bf16, and widened in the kernel), softplus scores/top-k/normalize, the indexer's top-k (a decode block over 131072+ scores splits each row over 8 chunks and merges their picks) | route: always; gate: <= 16 rows; topk: GPU | gate vs f64 truth; route experts equal, weights within 2e-6; topk the same set, ties to the lower index, the split the one-threadgroup picks in the same order |
| `dsv41_dense_f32x` | the f32 matmul of the compressor projections | decode rows | vs f64 truth |
| `dsv41_rope_tail`, `dsv41_kv_row` | `rotate` plus the QAT round trip (q, indexer q/k, compressed latent, inverse rotation of the heads); the window row's rms norm, rotation and fp8 round trip | every width | bit for bit (rms in MLX's accumulation order) |
| `dsv41_attn_decode`, `_split`, `_merge` | one token's window and selected-row attention with the sink and the folded inverse rotation (1..8 splits of 160 rows) | one token | every split no further from the f64 truth than the chain; the folded rotation bit for bit; the attention output's fp8 round trip (`qat_fp8`, QAT on) folded into the kernel is the separate round trip's bytes |
| `dsv41_attn_fused` | a wide span's whole `attnCore`: window band and picked rows read by index, online f32 softmax with the sink, no gathered rows or logits | wide chunks (more than 16 rows) on the GPU stream, bf16 activations, 16 divides the heads | 4e-6 to 1.3e-5 of peak against the CPU f32 `attnCore`, where the GPU `attnCore` is 3.5e-4 to 7e-4; the tiny oracle (4 heads) never engages it |
| `dsv41_attn_softmax` | the f16 arm of a wide span (`Model.attn_f16`, off) | wide spans, window + picks <= 1024 columns | within 1/64 of peak of the f32 chain (bf16 outputs); f16 scores change output, so it waits for a KLD run |
| `dsv41_index_scores`, `dsv41_index_scores_fp4`, `dsv41_index_tile` | the indexer's relu-weighted scores over key tiles, with the unseen groups at -inf | scalar up to 16 rows (`_fp4` reads the packed keys), matrix tiles when 8 divides the heads and the width | scalar and tile no further from the f64 truth than the chain; `_fp4` the unpacked keys' scores bit for bit; a tile scored at its key offset the whole list's columns bit for bit |
| `dsv41_hc_stage`, `dsv41_hc_stage_x`, `dsv41_hc_stage_z`, `dsv41_hc_split` | `hcMixes`, the previous-mix collapse and rms norm, and (`_x`) the `hcExpand` that feeds the FFN stage; the Sinkhorn split on one simdgroup; `_z` takes the attention's mixes and splits them in every threadgroup (no `dsv41_hc_split` dispatch after the attention stage) | up to 16 rows | mixes and normed collapse no further from the f64 truth than the chain; the expanded stream likewise; the split to 2e-6 of the thread-per-row split; `_z` bit for bit the split kernel plus `_x` (1, 3, 16 rows, bf16 and f32) |
| `dsv41_hc_mix`, `dsv41_hc_norm` | the same for a wide block (a tile of rows and mixes a threadgroup; split + collapse + norm) | more than 16 rows | bit for bit the stage's z, split and normed collapse |
| `dsv41_engram` | the Engram gate chain after `wkv` | up to 16 rows | bit for bit (MLX's row-reduction order) |

`SUSHI_DSV41_COMPOSED=1` (a diagnostic, read at load) runs the op chains instead and the load logs
`[dsv41] fused kernels on|off`: the teacher arm of the KLD A/B. The hyper-connection kernels run at 1024 threads a
threadgroup and halve it at load where the GPU caps them lower (`probeHc`).

The f32 matmul and `quantized_matmul` of MLX carry noise the kernels do not (about 3e-5 of the peak on a decode FFN), so
"closer to the f64 truth than the chain" is the bar wherever the sum order differs; a kernel that changes f32 sums
changes output at rounding level and rides the KLD gate with the first real checkpoint run. On the tiny model the
fused and composed GPU arms agree to the GPU matmul noise in the first layers; past them a routing near-tie flips an
expert, so only the logits' direction is held (`dsv41 fused`).

## Gaps against the reference

- The reference fake-quantizes the INPUT of every linear (`act_quant`); the forward does not. Upstream mlx-serve skips it
  too. Whether it moves KLD is unmeasured.

<a id="dspark"></a>
## DSpark

The release's drafter: three sliding-window stages (`mtp.0..2`, window 128, rope theta 10000 and no YaRN, 128 routed
experts top-3 and a shared expert each) that draft a block `[t, noise x 4]` in one pass. A request drafts unless it carries logprobs or a forced call
(`generate.dsparkEligible`, MTP's rule); those decode serially (`gen.dspark` is false and its ticks are plain). A grammar or JSON schema drafts too
([below](#dspark-grammar)). A repeat or
presence penalty and the think penalty shift the verify rows by the ids before them (`penaltyShiftRows`, `thinkShiftRows`). A thinking bound
(Anthropic `budget_tokens`, `reasoning_budget_tokens`) is checked between rounds, so a round may pass the budget by up to the
block; when it fires the closer is committed and the rest decodes serially.
A sampled request verifies the same block: the drafts are argmax tokens, a point mass, so each is kept with probability
p of the temperature/top-k/top-p filtered row and the first reject draws from the residual (the bonus draw on a full
accept), the serial sampler's distribution (`Generator.dsparkSampledVerdict`, MTP's `mtpBatchedAcceptGraph` and
`exactPrefix`).

- **Inputs.** The mean over the 4 hyper-connection copies entering layers 37, 38 and 39 (after Engram), concatenated,
  goes through `main_proj` (fp8) and `main_norm`; each stage projects that stream through its own `wkv`, key norm, rotary
  and fp8 round trip into a 128-row ring of committed positions. A stage attends over its ring and the 5 block rows,
  with no causal mask inside the block (`dsparkAttention`: `attnCore` under an all-true band).
- **Heads.** The last stage's collapse is normed and read through the TRUNK head; each position adds the Markov bias
  `embed[prev] @ head^T` (rank 256, f32) and its argmax is the next position's `prev`; the confidence head reads
  `[collapsed stream, markov embedding]`. `draft(m, t1, with_heads)` returns the picks, and the biased logits and
  confidence when asked; both are pinned to the reference (`ds_*` in the fixture, QAT off to 1e-4/2e-3 of the peak, QAT on
  to a cosine over the reference's own main rows).
- **A round** (`Generator.nextDspark`): draft from the pending token, verify `[t1, drafts]` as one chunk through the
  normal forward and the streamed experts (`dsv41_model.verify`), keep the longest prefix the trunk's own masked argmax
  agrees with plus the argmax after it (`finishVerify`). The chunk is cut to the tokens the budget still allows.
- **Rollback** (the model's own state, no capture): the backbone rings hold 16 rows of slack (`ring_slack`), so a rejected
  row never overwrites a row a later query reads; the compressed rows and index keys roll back by their row count; an open
  pooling group is re-cut from the concatenation the verify saved; Engram history is truncated; the stage rings take the main rows of the
  kept positions only. The tests check every accepted count 0..block against serial decode, then a second round, then the next draft.
- **Memory.** Resident, always (`dsv41_source.dsparkBytes`): 3 x 128 mxfp4 experts are 7.2 GB, the stages' trunk tensors
  0.58 GB and the Markov tables 0.13 GB (f32 head), about 7.9 GB out of the SSD budget's expert cache. The state bill adds the
  stage windows and the backbone slack (`Config.stateFixedBytes(true)`). The load is exact against the arrays the model holds
  (`dsparkHeldBytes` test), so the earlier "3.7 GB" figure of the audit was low by the experts' scales and half the experts.
- **Wiring.** The checkpoint's DSpark is the model's MTP head: `MtpHeadRef.dsv41`, kept under streaming
  (`expert_stream.streamedMtpHeadSupported`) when the pack ships the stages (`dsv41_source.shipsDspark`), so the engine-default MTP
  is on and `--no-mtp` / `mtp: false` drop it. `--mtp-min-depth` / `--mtp-max-depth` bound the verify width (below); the block is fixed at 5 drafts.
  The boot line reads `[spec] DeepSeek-V4.1-Flash decodes with DSpark (...)` or `... plain ticks (...)`, and a request that
  drafted logs `[spec-stats] mode=dspark` with acceptance, draft and verify ms.

- <a id="dspark-grammar"></a>**Grammar** (`row_grammar.zig`, `Generator.dsparkConstrainedEntry`). Each verify row takes the mask serial decode builds
  after the drafts before it (`row_grammar.plan` walks a scratch copy of the JSON grammar and reasoning state over the drafts), so the
  masked argmax of a row is the serial token and a greedy request emits serial's tokens; a sampled one draws from the masked, filtered
  row. A draft its row's mask forbids ends the chunk there, and so does one serial decode answers by forcing the boundary (EOS or pad
  inside the reasoning). While a schema waits behind the reasoning close the rows carry only the reasoning mask (it excludes the
  tokens that would close with a payload the schema rejects); past the close, the JSON mask. The pending token is judged at the next
  round's start as serial decode judges a sampled one (`dsparkJudgePending`), and a serial tick (`next`, the loop guard's forced
  boundary) first emits it (`dsparkEmitPending`). `[spec-stats]` adds the host `mask_ms` per round and the serial ticks. A channel-header
  protocol or an opener choice keeps the request serial (`Constraint.rowsDraftable`).

- **Verify width** (`dspark_rows.zig`, `Generator.nextDspark`). A round verifies the first R rows of its 6-row block (t1 and R-1
  drafts; the draft is greedy, so a shorter verify keeps exactly the tokens a longer one would have kept up to R) and R=1 is a plain
  step with no draft. R maximizes expected tokens per ms, `E[tokens(R)] = 1 + sum_i prod_{j<=i} p_j`, over the cost of the round
  (`dflash_policy.chooseRows`). `p_j` is the confidence head's logit through a sigmoid and the per-bin calibration of
  `dflash_policy.Calibration`, seeded with the identity (the head is a calibrated probability: bins of it measured 0.26 for
  0.1-0.2, 0.61 for 0.5-0.6, 1.00 for the 287 positions above 0.9, and the conditional rate is flat across positions, so one
  calibration serves all five). The cost is `plain + step * (R-1) + draft` for every R, the one-row round included: `rows` runs on
  a block already drafted, so the draft is spent whatever R is and a dismissed block is not a free plain step (a break-even second
  row then needs `p > (draft + step) / plain` only against drafting at all, which the back-off below judges). Plain and step come from
  the request's own rounds, each moving a bounded share per sample (a cold expert cache cost 100-250 ms for a one-row step and 2 s
  for a first wide verify, and the unbounded fit it replaced never chose a wide verify again), the step drifting back to its prior
  (9 ms a row, plain 28 ms) while unmeasured. Drafts that lose back the next ones off: `Policy.edge`, a decayed mean of
  `(1 + accepted) * plain - (draft + verify)` ms per drafted round, turns negative and the next drafts owe plain steps 1, 2, 4 .. 16,
  cleared by a round that pays. One dismissed block does not back off: blocks are bimodal (on the sweep prompt 26% accept
  nothing and 24% all five), the next block pays whatever happened between, and the earlier per-block rule (R=1 owes the next
  `period` plain steps) turned 25% dismissals into 55% one-row rounds once the verify got cheap (replay of the sweep's 68 blocks
  through the policy: 44.8 tok/s with it, 50.4 without; the gate ended 12-18 ticks of 3000 either way). Live (the shipped STQ pack, budget 110, `--ctx-size 65536`, one A then one B, die under 50 °C, commit c42b60a5): the 5.4k harbour prompt 45.1 -> 47.9 tok/s, a novel story 44.0 -> 47.6 (the old rule made 590 of 595 rounds plain there, the new one drafts a third). Greedy text is the same at every round-width mix (exact greedy, below): a row has the same bytes at every verify width. The stop switch is GLM's
  (`spec_cost_gate.zig`: 8-round windows against a probed plain step, 5% margin); the plain step is the 1-row round, so the state
  is the engine's and no other drafter runs. Sampled requests use the same choice and the acceptance rule above (a 1-row round draws
  its token from the filtered row). `--mtp-max-depth n` caps R at n+1 and `--mtp-min-depth n` floors it at n+1 with no plain step or back-off below it (equal values pin R; a depth past the 5-draft block clamps to it, logged once at load; `--mtp-min-depth 1` is the default). `model-settings.json` has no depth keys. `SUSHI_DSV41_DSPARK_ROWS` forces or sweeps R. The adaptive R follows measured ms, so a seeded sampled
  request repeats only at a forced R: under GPU load the round widths differ and the draws with them (the Generator test pins R=3
  for its repeat check).
- **Verify cost** (an STQ pack, M5 Max, `taskpolicy -a`, budget 88, one boot, attention arms interleaved, trimmed means, ms per
  verify): `v(R) = v(1) + step * (R-1)`, v(1) 37 / 42 / 53 and step 12.5 / 13 / 13.5 at 1k / 4k / 16k of context; the 6-row verify
  of today's fixed block is 106 / 106 / 110 ms against a 37-53 ms serial step. The routed experts are most of the step: a pick is its
  own GEMV (26 ms of a profiled one-row layer stack to 69 ms at six rows), the shared expert and the engram add ~1.5, and
  the trunk linears and the router do not grow. Rows 2..16 take the decode attention kernel (it already read a row per
  position and picks); against the composed ops that saves 0-6 ms at 1k-4k and 10-13 ms at 16k for 3-6 rows (a boot-to-boot A/B said
  10-60 ms and was the boots; interleave).
- **Speed** (same prompts and boot class, warm second pass, decode tok/s; `--no-mtp` serial / fixed 6 rows / adaptive): copy 1k
  27.5 / 41.0 / 48.9, copy 4k 27.4 / 45.5 / 47.8 (verbatim copy of the first 60 lines of a document), thinking 4k 27.6 / 35.0 / 45.6,
  novel story 1k 27.6 / 17.3 / 28.4, story 4k 28.1 / 19.2 / 27.3. The fixed block loses a third on novel text because it pays six rows
  for about 1.3 accepted tokens; the adaptive round matches serial there (the drafts of a novel text are worth about what their rows
  cost) and takes 70-90% of the rounds as 1-row steps. A one-row step through the verify path decodes at 32 tok/s warm at 1k, 4k and
  16k, above the 27.5 of `--no-mtp` (different boots: unconfirmed).
- **Exact greedy costs nothing measurable** (base `1aececf3` + the exact route, the knob on against off in one binary; the shipped
  STQ 1.75bpw, `--ssd-budget-gb 108 --ctx-size 65536`, `taskpolicy -a`, GPU lock, QUIET, fans max, die under 50 C at each boot, one
  boot per arm and kind, A then B; a ~4.9k-token code prompt (a Zig source review) and a ~4.0k-token prose prompt, 600 greedy
  tokens, second pass of each; forced `SUSHI_DSV41_DSPARK_ROWS=sweep`, verify ms trimmed means of `[dspark-round]`, ~30-40 rounds a
  width). A = arms that sum by width (`SUSHI_DSV41_DSPARK_EXACT=0`), B = exact (the default):

| R | code A | code B | delta | prose A | prose B | delta |
|---|---|---|---|---|---|---|
| 1 (plain step) | 22.4 | 22.7 | +0.3 | 22.1 | 22.2 | +0.2 |
| 2 | 32.0 | 31.6 | -0.4 | 31.3 | 31.3 | 0.0 |
| 3 | 38.0 | 38.2 | +0.2 | 36.6 | 36.4 | -0.2 |
| 4 | 46.3 | 45.7 | -0.5 | 43.5 | 42.9 | -0.6 |
| 5 | 53.1 | 53.4 | +0.2 | 49.3 | 49.2 | -0.2 |
| 6 | 60.9 | 59.8 | -1.1 | 56.1 | 56.0 | -0.2 |
| 8 | 77.6 | 74.3 | -3.3 | 68.6 | 68.6 | 0.0 |

  The plain step costs +0.3 ms (1.3%) with NAX at one row; the wider rows are unchanged to within noise (the head's SIMD arm
  gains a little at 2 to 5 rows). Adaptive decode, warm second pass: code 60.7 / 64.5 tok/s (A / B), prose 46.9 / 46.5; the texts
  differ, so each arm's acceptance differs (code 84% / 88% of drafts), and the numbers are within what the text moves. Text: B is
  byte-identical over three repeats of a request and equal between the forced-width sweep boot and the adaptive boot; A differs
  between its own repeats and between its sweep and adaptive boots. `--no-mtp` serial: 49.4 tok/s code, 48.7 prose (20.2 and
  20.5 ms a token), text identical over repeats and not equal to DSpark's. The plain DSpark step is 22.4 ms, 2 ms above serial.
- **A one-row round is the serial step.** It opens no verify (nothing to give back), and the stage windows' commit
  (`commitMain`, per row, never batched across rows: the 1-row and multi-row `main_proj` arms may differ in bits) is left
  lazy and submitted when the next pass begins, instead of a second host wait after the verdict. Greedy text of a forced
  one-row run is byte-identical to `--no-mtp` over 600 tokens with `SUSHI_DSV41_DSPARK_EXACT=0`; by default its trunk takes the NAX arm
  (exact greedy, below) and can part from `--no-mtp` at a near-tie. Measured on `353f8f79` (shipped STQ 1.75bpw, `--ssd-budget-gb 108
  --ctx-size 1000000`, 4754-token prompt, 600 greedy tokens, first request of a cold boot, lock, QUIET, fans max, die 49.5 C,
  `taskpolicy -a`): forced one-row 21.8 -> 21.2 ms a token (verify median 21.4 -> 21.0) against 20.9 serial (`d57f8ea8`, other
  boots, so unpaired); adaptive 62.5 -> 63.8 tok/s. What is left (~0.3 ms) is the commit's own GPU time plus the 7.9 GB
  DSpark residency taken out of the expert cache, which `--no-mtp` does not pay (not separated).
- **Queued routed chains and paired verify rows** (the shipped STQ 1.75bpw pack, M5 Max, budget 110, `--ctx-size 65536`,
  `taskpolicy -a`, one A then one B per boot, die under 50 °C, a 5.4k-token prompt, 600 greedy tokens; base main `0e165e29`):
  `--no-mtp` serial 31.7 -> 39.8 tok/s (28-32 -> 22-24 ms a token), greedy text byte-identical. A Metal trace of the
  serial step before: GPU busy 61%, one idle gap of ~280 us per layer (~110 us host turnaround, ~170 us from commit to
  GPU start on an empty queue); with the queued chains 72% and ~16 gaps a token (38.4 tok/s), and with the head, Engram
  layers, DSpark commit and rebuild cap no longer settling first, 39.8. Verify ms by forced R (DSpark
  sweep, same prompt, text byte-identical across arms) before / after: R1 36.8 / 24.9, R2 47.5 / 36.4, R3 53.8 / 44.0,
  R4 63.2 / 53.5, R5 72.0 / 61.9, R6 80.5 / 68.2, R8 97.6 / 84.9. Pairing rows that share an expert gives 3-7% of that at
  2-8 rows. 200 MLX ops per command buffer instead of 50 changed nothing (28.3 vs 27.8 tok/s).
- **Serial decode 39.8 -> 46.5 tok/s** (same pack, settings and prompt as above, base `4132a1db`, lock, QUIET, fans max, die
  under 50 °C, `taskpolicy -a`; the 600-token request twice after the warm request, greedy text byte-identical in every
  arm). A Metal trace of a plain layer puts it at ~508 us of GPU: routed chain 160 (48 MB, ~300 GB/s, ALU-bound by the PAIR1
  decode), shared expert 54 (17.7 MB), wq_b 48 (21 MB, ~450 GB/s), wo_a + wo_b 89 (38 MB, ~410 GB/s), attention core 41, gate
  18, the hyper-connection stages ~40, the rest under 20 each. Three changes: the one-row routed chain prepares its pair input
  and SwiGLU mid once (-23 us a layer); the attention output's fp8 round trip rides the attention kernel (-10 us a layer,
  one dispatch fewer); the load-time warm fills each pool with its most used experts (above). First request 40.8 -> 43.3
  from the first two, 43.4 -> 46.5 from the warm; the second request, with no cache misses, runs 46.8 -> 50.2. What is left
  is the misses of a first request (each rebuilds the layer and the successor: +1.0 to 1.7 ms) and kernels at their
  floor: the attention core is bound by its per-row softmax (unrolling the row loads or reading the packed fp4 rows in place
  ran 2x slower), trunk weight prefetch under the small stages gained nothing in a process bench, and the rows geometry
  policy is within 2-5% of the best of a sweep.

- **Expert prefetch** (forwards of 2 to 16 rows, i.e. DSpark verify; a one-row step does nothing extra): a missed expert
  rebuilds its layer (~1.9 ms), and 6-9 of 40 layers did per R4 round. A layer's router input (V1) times the gate matrix of
  the layer `d` = 3 ahead ranks its experts, and the first uncached of each row's top 12 is read ahead of the layer. On the
  diag build (perf/dspark-picker `c42b60a5`, forced R4, reqC, 600 tokens) the first uncached of each row's top 8 at d=2
  covered 40-49% of the experts that missed at 1.0-1.3 reads a layer, but the experts it reads also serve the next passes:
  layers with a miss fell 43-46% with the reads completed ahead (0.7 reads a layer). The layer's output stream as the
  predictor is only 3 points better for ~8 more ops; recency and frequency of the uncached experts cover 1-4%. Timed on that
  base, 4 fill workers: forced R4 round 59.0 -> 53.2 ms (46.4 -> 51.5 tok/s) at d=3, top 12 (d=2..4 and top 8 within 1 ms;
  2 workers 55.8 ms, more than 4 nothing). The predictor costs ~0.9 ms a round at R4 (a 7.9 MB f32 gate read a layer). At one
  row it measured 41.0 -> 42.4 tok/s on a cold cache but lost on a warmed one (46.6/49.8 -> 46.0/46.1 on main `d810c50c`), so
  a one-row forward is left alone.

Differences from mlx-serve: the attention-output fp8 round trip stays (upstream drops it), the verify block takes the same
masked argmax as serial decode, and a verify is not split from the draft with a host sync per Markov step (the chain is
one lazy graph). Gaps: a verify chunk is not drafted from a copied context (prompt lookup; a copy turn accepts 3.3-3.9 of 5 drafts, so
a wider block would pay there), and a verify row is its own routed-expert GEMV where a third of the picks of five neighbouring rows
(30-40% by the `[expert-stream]` distinct-pick counts) repeat an expert (decoding each distinct expert once for all its rows is the
lever on the 13 ms step).

<a id="stq"></a>
## STQ packs

A pack with `expert_quant` and `trunk_quant` ([pack-format](pack-format.md#dsv41-stq)) serves its routed experts as
PAIR1 rate groups streamed through the expert cache (`dsv41_stream.StqStream`: each group a slot pool, the picks on the
grouped trellis chain with the clamped SwiGLU, `exl3.rate_groups`), its DSpark stage experts as resident ragged banks,
its trunk linears and head as SPEC1 banks (`dsv41_model.Stq`, [dense linears](engine-exl3-experts.md#dense-linears)),
and its embedding through the affine row gather. The FP4 fused kernels do not engage on it (the shared expert is a
trellis bank); the FFN is the routed sum plus the shared expert, which at decode width (<= 16 rows) is one expert of the fused
trellis chain (`sharedFused`: pair GEMV, fused mid and down, reduce; the clamped SwiGLU stays in f32) and over more rows the
three linears and the SwiGLU ops; the routed decode reduce adds it in f32 (`stqRouted`), the byte-equal of a separate add.
On the route-spec queue and in the DSpark stage layers the shared expert rides the routed chain's own launches (5 fewer
launches a layer, byte-identical; [engine-exl3-experts](engine-exl3-experts.md#dsv41-shared-fused)); the waiting schedule keeps
it a chain of its own.
Trunk linears take the multi-row kernel up to 8 rows. Up to 8 rows, `wq_a` and `wkv` (and a ratio-1 KV source's compressor
`wkv`) run as one stacked launch, as do a pooling compressor's `wkv` and `wgate` (`Layer.in_shared`, `CompW.shared`,
stacked at load; each linear keeps a view of the stack for wider chunks). `dsv41 real` holds both stacks of layer 2's own
bytes to the 2e-3 bar.

Decode schedule (<= 16 rows, any pack): a layer's output, state rows and DSpark targets are submitted, not waited on,
so the host builds the next layer while the GPU runs this one. On an STQ pack the router's picks are submitted first and
the shared expert is queued behind them (the GPU runs it while the host reads the picks and fills the misses). The next
layer's picks depend on this layer, so their read is the one host wait per layer and no union slab is refilled under a
reader. `Model.sync_layers` (tests) waits at every layer instead; the two schedules give the same bytes through hits,
misses and the union.

STQ decode also queues the routed chain before the picks are read (`Model.route_spec`, `Experts.Spec`, the Qwen schedule
of [engine-expert-streaming](engine-expert-streaming.md#decode-schedule)): `StqStream.queue` takes the slot of each pick
from the layer's GPU map (expert to global slot of its rate group's pool, -1 as slot 0 where not cached) and builds the
chain at once, so the host is a layer ahead of its reads. `passLayer` settles the previous layer after submitting the
current one (`settleRecord`): `StqStream.settle` reads the picks, resolves them in the pools and returns null when every
pick sat in the slot the GPU read, else the exact sum. A miss rebuilds the layer's output (`hcExpand` over the saved
stream, `Unresolved`), restores what the successor's pass changed (`Undo`: the open pooling group, the verify copies, the
compressed/index row counts and, over several rows, the window ring, because a later row's write can land on a slot an
earlier row still reads; a one-row pass's ring slot and appended rows are written again at the same places) and runs the
successor again. Only the settle touches the cache, so a discarded route moves no LRU or stat. The picks of a layer
are verified against the pool before any later layer's, so a DSpark target, a verify round's rows and the state arrays
of a kept layer never come from a wrong sum. An Engram layer is rebuilt from the rows its pass read once
(`Pass.engram_rows`). A layer the stream cannot queue (resident, released, lossy pick) settles its predecessor first; a
pass that rebuilt 3 layers plus a third of those built (`Model.route_spec_rebuilds`) stops queueing. A decode pass does
not wait for its embedding, and queues the head from the last layer's queued sum before settling it (a rebuild builds the
head again), so the GPU's only idle points are the logits read and the host's sampling. The bytes equal
the waiting schedule's (`dsv41 stq: queueing the routed chains ...`, `dsv41 stq: queued routed chains leave a verify
round's kept rows ...`). Kernel shapes, arms and timings:
[engine-exl3-experts](engine-exl3-experts.md#dsv41-stq).

| Proof (`DSV41_STQ_TINY`, a pack made from the `--geometry stq` fixture) | Bar |
|---|---|
| served vs its dense twin (every bank decoded on the host, the MLX op chains, QAT off, f32) | layers 0–1 cosine > 0.9999, logits > 0.99, prefill and decode chunks of 1, 3 and 5 rows, a cache that holds every expert and one that streams |
| vs the FP4 model it was made from | first attention cosine > 0.95 (random weights at K1.5–K4 move the experts by tens of percent) |
| greedy DSpark rounds | the serial greedy tokens (f32, QAT off) |

The GPU forward is not batch-invariant: a verify chunk and a serial row differ in the last bits (MLX's matmuls at 1 and
at 5 rows, the one-row SIMD and the multi-row NAX trunk kernels), which QAT round trips and the indexer's top-k turn into jumps on the random tiny
model, FP4 and STQ alike; the exact greedy bar therefore runs in f32 with QAT off, and a kept row that left its serial row
is accepted only where a selection of compressed rows (the candidate blocks or the index top-k, traced as `cand_*` and
`index_*`) at or before the first diverging layer differs, each side picked its own top-k, and the last pick ties to 2e-2
of the score scale on one side (the GPU twin's own f32 matmul noise is ~1e-2 of a stream by layer 4; the served path stays no
farther from the f32 CPU truth than the twin, `the served path is no farther from the f32 CPU truth`); a router flip is held to its own margin the same way. Either ends the comparison, so on
the tiny pack the test covers the rounds before the first tie.

<a id="drop-expert"></a>
## Expert drop (`--drop-expert n`)

The pack's optional `expert_usage.json` ([pack-format](pack-format.md#expert-usage)) counts how often each routed expert was
picked per layer. `--drop-expert n` (a multiple of 8, 0 = off, DeepSeek-V4.1 only; other archs refuse the flag at load)
drops the `n` least-used experts of every backbone layer (ties by expert id): they are never read, billed, cached or picked.

- **Router.** The dropped experts' `gate_bias` entries are -inf, so the pick (score plus bias, before top-k) goes to the
  next-best kept expert, and the weights are the unbiased scores of the kept picks, normalized and scaled as the normal
  path: only the candidate set shrinks. The fused router kernel and the generic path share the mask. DSpark's stage experts
  are not dropped.
- **Bills.** Every expert count in the plan is the kept count: the cache slot cap, the union workspace, the ledger's
  workspace term, the profile split, the per-group pools and the union bank (`LayerGroups.dropped`, `GroupedLayer.union_base`).
  The expert ids and the shard spans keep the pack's numbering; a route that names a dropped id is `ExpertDropped`.
  A drop that would empty a rate group of some layer is refused by name (`DropEmptiesRateGroup`).
- **Resident mode.** `StoreShape.all_bytes` counts the kept experts and their meta rows only, so a budget that holds every
  kept expert engages the resident mode (pools of the kept count, the GPU expert-to-slot map carries -1 for a dropped id).
  On an STQ pack at `--ssd-budget-gb 116` the whole pack is not resident; with `--drop-expert 8` or `16` it is (plan only,
  headers read, no weights).
- **Why a multiple of 8.** No kernel needs a multiple: the route kernel pads to 32 lanes with a bound check, the ragged pools
  and meta table take any slot count, and the window kernel holds up to 512 experts. The one expert-count tile left is the
  router GEMV, 8 experts a threadgroup, so a multiple of 8 keeps the kept width on whole threadgroups; 16 is one. The pack's
  rate groups are not aligned (12, 30, 80 and 262 experts in one layer), so alignment cannot come from them.
- **Log line.** `[expert-drop] n=16 per layer, 640 experts, 4.27 GB not loaded, list from <source>`.
- **Proof** (`DSV41_TINY`, `DSV41_STQ_TINY`; `-Dtest-filter="drop"`): the masked router picks the top-k of the kept set and
  weights as the normal path, bit for bit on the CPU stream, and within the kernel's rounding on the GPU; the STQ pack with
  experts dropped gives the logits of the full-pool engine under the same mask bit for bit, caches no dropped expert and
  refuses one by name; greedy DSpark rounds emit the serial tokens of the same mask; the transformer loads the list, the
  plan bills the smaller set. KLD capture refuses the flag; `sushi kld compare --drop-expert n` scores a pruned student.

<a id="prefix-cache"></a>
## Prefix cache

A turn that extends an earlier prompt restores the request's state and prefills only the suffix. It follows the
engine default ([engine-prefix-cache](engine-prefix-cache.md#defaults)): the SSD tier on at its formula size, RAM
retention off unless `--prefix-cache-mem` names a size. With no flags every turn restores from SSD (SSD-first); with
`--prefix-cache-mem` the RAM tier serves the active session and the SSD tier backs it.

- **A restore point is the whole state at one position** (`dsv41_model.captureState`): each layer's window ring and open
  pooling group, the compressed and index rows through the position, the DSpark stage windows. The Engram history is
  rebuilt from the prompt's token ids. It is stored as an `SSMCheckpoint` (`src/dsv41_prefix.zig`): each slot's four
  arrays ride two snapshot layers' conv and ssm fields (the aux and pooled fields mean a QSA history to the cache, so
  they stay empty; an absent array is a one-byte placeholder), so the cache's checkpoint list, thinning, merge, byte bill
  and the SSD tier's `s<pos>.safetensors` files serve it unchanged. A restore shares the checkpoint's arrays and never
  writes them.
- **The SSD tier indexes by KV rows and this arch has none**: before a commit `dsv41_prefix.markRows` gives layer 0 of
  the slot's cache one zero byte per token, so the tier's chunk files, sizes, scan and eviction cover the entry
  (about 2 B a token); the forward never reads it. A restore (`DiskTier.restoreCheckpointInto`) loads the checkpoint
  file and its row segments and proves the chunks cover its position. SSD-first, a prefill's points go to the tier as it takes them
  (`HotPrefixCache.writeThroughDsv41`, armed whatever the new span): a point at or past the tier's 512-token floor shares
  the request's rows (`dsv41_prefix.captureShared`), is staged into the writer's bounded host buffers and released before
  the next forward appends, the newest and the prompt-end snapshot included. Each logs `[dsv41-prefix] restore points to
  SSD at P: X GB staged, prefill held T ms` (X: the point files and row segments the call staged). With a RAM tier they wait for the commit's flush after the response.
- **Namespace**: `ModelConfig.cacheLayoutNamespace` returns `dsv41_prefix.namespaceTag` (`dsv41-state-v2`, plus
  `-dspark` when the stages are resident), hashed into the tier's fingerprint beside the pack's shard names, sizes and
  mtimes and `config.json`: another pack, another slot set or a bumped tag never reads these entries. Any change to what
  a stored state means bumps the tag in the same landing.
- **Checkpoints** come from the Qwen schedule: absolute multiples of the stride (at least the prefill chunk, 8192 by
  default) and the prompt-end snapshot 30 tokens before the end, where the next turn's re-rendered generation prefix
  still matches. In RAM (a RAM entry, a prefill's list) at most 4 (`dsv41_prefix.checkpoint_cap`): each is a full copy,
  about 890 B a token plus the fixed rings. An SSD entry keeps up to 160 (`disk_checkpoint_cap`): the 8192-token grid
  to 1M tokens with room for earlier turns' prompt ends and thought points. The thinning drops the interior point whose
  neighbours are closest and always keeps the oldest and the newest, so the newest point written (below) survives until
  a later turn adds one.
- **An SSD entry stores the rows once** (`dsv41_prefix.row_span` = 8192, `rowDivisors`, `DiskTier.setGrownSpec`): the
  compressed rows and index keys of every 8192 tokens are one `g<n>.safetensors` segment, appended as the entry grows;
  a point's `s<pos>.safetensors` keeps the window rings, open pooling groups, DSpark windows and the rows past its last
  whole span (a grid point none, a prompt end under 8192 tokens). A restore reads the point and its first `pos / 8192`
  segments and joins them into the arrays a full capture holds, byte for byte. A grid point costs about 6 MB; the
  store about 890 B a token (`diskBytesPerToken` = rows plus the rings' share per span). Thinning a point never touches
  a segment, and the store is not trimmed when the point that reached furthest goes. An entry at 393k tokens holds the
  48 spans (350 MB) and 49 points (about 300 MB), against four full copies of 28 to 357 MB.
- **A follow-up turn rolls the live state back instead of reading SSD** (`dsv41_prefix.Live`, `dsv41_model.rollbackState`):
  rows are append-only, so a point at `p` is the fixed arrays at `p` (window rings, open pooling groups, DSpark
  windows, about 6 MB) plus each row buffer's `used = p / ratio`. A request keeps those arrays for its deepest 8 points
  (`live_point_cap`; those it writes through and the point it restored from) and publishes them at its commit with its
  tokens. The next prompt rolls back to the deepest point at or below its shared prefix (under the prompt's end, the
  media rule's ceiling, equal `has_tools`) unless a stored point reaches further; the result is byte-identical to an
  SSD restore of that point. It is skipped when the state was replaced (`state_epoch`), the request never finished,
  or a row buffer exceeds what the prompt's own state would be billed with its quarter (`liveRowsFit`): the live rows
  past the point stay allocated, so they must fit that bill. The admission reserve adds 8 sets of fixed rings
  (`dsv41CheckpointReserve`). Log: `[dsv41-prefix] reused N of M prompt tokens from the live state`.
  A cancelled request (stop sequence, disconnect) publishes at the scheduler's cull (`commitsAtCull`), not at the cleanup
  drain: the cull drops it from `decoding`, so the next request is admitted over the model's state before `complete()`,
  and a later publish would pair this request's tokens with that request's points and epoch.
- **A thinking request with tools writes one more point mid-decode, right behind its `</think>`**
  (`scheduler.writeDsv41ThoughtPoint`; DSpark rounds and an SSD-first tier only). With tools the template re-renders an
  earlier assistant turn as `reasoning_content</think>content` plus the call block, so the next prompt repeats the
  generated tokens through the closer and the 5-20k tokens of a long thought need no re-prefill. A round that would
  commit past the closer is cut to end on it (`Generator.dspark_round_ends_at`: fewer verify rows, the pending token
  still the trunk's own), so the state stands exactly behind it; the point is staged like an SSD-first prefill point
  (`HotPrefixCache.writeDecodePointDsv41`: shared rows, released before the next forward) and logs
  `[dsv41-prefix] restore points to SSD at P: X GB staged, decode held T ms`. Its bill is the prefill's
  (`dsv41CheckpointReserve`: the writer permit and one set of fixed rings, held by the request's admission bill), so it
  needs a prompt of the SSD floor (512 tokens) and checkpointing on. It is skipped for a thought under 512 tokens (the
  prompt-end snapshot's re-prefill is cheaper) and for one whose first or last token is `\n` or a space, which the
  server trims off the delivered `reasoning_content`. A client must send `reasoning_content` back verbatim, as the
  server delivered it. Serial decode (no DSpark) leaves the state one
  token ahead of the emitted ones and a RAM tier keeps points on the commit's list, so neither writes it.
  What a later prompt repeats of a generated turn (`dsv41 replay` test, the pack's `tokenizer.json`, the real template,
  `splitThinkBlock` and `parseToolCalls`):

  | turn | generated tokens the next prompt repeats |
  |---|---|
  | thinking, tools: content, a call with string parameters (code, quotes, backslashes included), a plain answer | all |
  | thinking, tools: a call with a JSON array parameter | through the call's first parameter; `tojson` spaces the items |
  | thinking, tools: the thought ends in `\n` | up to the `\n`, one token short of the closer |
  | thinking, no tools | none: the earlier turn is opened `</think>`, not `<think>` |
  | chat mode | all |

  So the closer is the one position every tools turn shares. The end of the generation is no second point: a tail
  that differs voids the whole point (a restore cannot pass the first differing token), whether it differs depends on
  the call's parameter types, the server's argument coercion and the client, and under the cap of 4 the end point
  would thin the thought point out (the smallest span), turning a miss into the loss of the whole thought.
  A round that finished the request (`length` on the closer) writes none: its commit published the live record, and
  `Live.keep` would clear it.
- **Equal to cold** means bit for bit when both run the same chunks from the restore point on: the tiny pack's logits
  (bf16, QAT on and off) over odd-width prefill chunks, decode, a verify round and its rollback, and the state they end
  in, are identical bytes, from RAM and from an SSD file written by one tier and read by a fresh one. A restore off the
  cold run's chunk grid differs from cold at rounding level, as on every arch (the GPU forward is not
  batch-invariant); the Generator test restores at the prompt-end snapshot.
- **Bills.** The RAM tier's residency is the checkpoints' bytes (`kv_bytes`), released on eviction and unload. A prefill
  holds its list while it runs: the connection-thread and admission bill add `server.dsv41CheckpointReserve`, with a
  RAM tier the capped list plus the copy taken before a thin (each at most the request's state), SSD-first the writer's
  permit (or one point, when smaller) and one point's fixed rings. An SSD restore holds in RAM only what the model then owns: the checkpoint's arrays become the request's state
  (the state bill), and its file read is transient. The disk budget formula prices an entry at the cap of full states
  (`diskBytesPerToken`).
- Every request logs `[dsv41-prefix] reused N of M prompt tokens from RAM|SSD|nothing, prefilling K; lookup and restore X ms`; every other arch logs the same line tagged `[prefix]`.

**Real pack.** `DeepSeek-V4.1-Flash-1017`, M5 Max, `--ssd-budget-gb 76` (100, 92 and 84 are refused with `ExpertCacheDoesNotFit`: the
stages' 8 GB sit beside the serving bytes the load check adds), bf16 KV, 160 greedy tokens, no GPU lock held by
another job:

| prompt | per-draft acceptance | accepted per round | DSpark tok/s | serial tok/s | bytes vs serial |
|---|---|---|---|---|---|
| prose | 36% | 1.76 | 4.6-5.4 | 4.4 | differ from char 524 (a near-tie) |
| code | 69% | 3.44 | 5.8-6.2 | 4.2 | differ from char 449 (a near-tie) |
| copy | 100% | 4.93 | 12.2-13.0 | 5.5 | identical |

Greedy DSpark text is the same at every round-width mix on the same hardware (exact greedy): the same request gives the same
bytes whatever plain and drafted rounds the policy (`dspark_rows.Policy`) picks from measured timings, and at any forced width
(`SUSHI_DSV41_DSPARK_ROWS`). A row's bytes depend on the arm of each linear it passes through, so each linear has one arm at
every width. The trunk linears (`projectRows`) were the only arms that changed with the width: the one-row step took the SIMD
rows arm and 2 or more rows the NAX arm, whose k sums differ. A DSpark round now runs the NAX arm at one row too
(`kernels.rows_exact`, set around `verify`), except the lm_head, whose SIMD rows arm equals its one-row launch bit for bit and
is the faster one up to 5 rows, so it takes the SIMD arm at every width. The rest is already width-invariant, by test: the shared
expert's one-row coop chain against the lane chain (`STQ shared expert at DeepSeek's shape`), the expert-grouped routed chain
(`verify rows that decode each distinct expert once`), the per-row commit and attention, router and indexer rows (the indexer's
top-k split changes no pick). Where there is no NAX the arms are the same and nothing changes. `--no-mtp` serial keeps its SIMD
one-row arm, so its text can part from DSpark's at a near-tie (it is itself the same at every repeat). `SUSHI_DSV41_DSPARK_EXACT=0`
restores the arms that sum by width (SIMD one-row step, NAX head at 2..8 rows); text then depends on the round mix. What holds
either way is that a generation's stream and non-stream responses carry the same bytes.
`--max-mtp-ctx` ends drafting for the rest of a request once prompt + generated passes it (plain 1-row rounds), as it does
for MTP. A drafted image id ends the verify chunk before it: only a prompt carries one, and the model refuses it without rows.
A round (draft 12-17 ms, verify 450-750 ms) costs about 2.6 serial
decode steps, since five rows route to more streamed experts than one, so only acceptance above that pays.

**Batch invariance.** The GPU forward is not row-count invariant: a 5-row verify and the serial rows differ in the last bits,
so greedy DSpark equals serial only until a near-tie, and the exact proofs run on the CPU stream. On the tiny model in f32
with the composed ops, the first op to differ is the hyper-connection mix projection (an f32 matmul whose reduction order
depends on the row count); with the kernels on, the decode attention kernel differs from the span core as well, and the
attention-output fp8 round trip turns a last-bit difference into a whole fp8 step. A bitwise fix is row-count-invariant
kernels for the trunk matmuls, the hyper-connection mixes, the indexer and the MoE (one reduction order for any number of rows,
as the GLM and Qwen verify kernels have); it is not in this port.

<a id="image-input"></a>
## Image input

Images are served by default: the tower loads on every DeepSeek-V4.1 load, streamed or resident, unlike MiMo and GLM
whose streamed loads leave it off. `--no-vision` (or `vision: false` in model-settings) skips it and saves its 0.50 GB,
and an image without it is a 400; an explicit flag outranks the setting, which outranks the default. One image is `[start] + ([n_w aligner rows] +
[newline]) * n_h + [end]`, every position of which is `image_token_id` (`<｜deepseek_image｜>`) in the prompt.

- **Processor** (`src/dsv41_vision.zig`, the release's `inference/image_processor.py`): `planGrid` picks the canvas
  (14 px patches, 3x3 aligner cells, at most `max_image_tokens` 1024 span tokens, scaled up to `min_pixels`), `letterbox`
  is Pillow's `ImageOps.pad` (bicubic `contain`, gray 127, half-to-even margin; `vision_common.resizeRgbU8` is
  Pillow's two-pass uint8 resize), pixels are `(x/255 - 0.5)/0.5` in f32 in `(c, py, px)` order. Equal to the
  reference to the bit after the bf16 round, and `plan.json` pins the geometry of twelve sizes. The prompt needs only
  the grid, so a request holds each image as its file and `planGrid` runs on the size read from the header; the
  patches are built for the media key and again when the tower reaches the image, a window at a time ([server-lifecycle](server-lifecycle.md#request-ownership-and-media)).
- **Tower** (`Dsv41Vision`, `vision.py`): 32 blocks of pre-norm attention (16 heads, 2-D rope: row angles in the first
  half of each rotate-half pair, column angles in the second, f32) and a SwiGLU MLP, then `Aligner`
  (zero-pad to multiples of 3, unfold channel-major, `w2(gelu(w1(x)))`). It returns the whole span including the three
  learned delimiter rows. Queries run in chunks of 512; `encodeScratchBytes` bills the worst block.
- **Requantized at load** (`Dsv41Vision.init`, `quantizesAtLoad`): the pack keeps the tower bf16, and the load turns every
  block `wqkv`, `wo`, `w1`, `w2` and the aligner's `w1`, `w2` (322 matrices, 483M of the 485M parameters) into MLX
  affine 8-bit, group 128, run through `mlx_quantized_matmul`, and frees each bf16 copy as its codes exist. The patch
  embedding (588 wide), norms, biases and delimiters stay bf16. One `[vision] tower requantized at load: ...` line
  reports it. Against the bf16 tower on house.jpeg at 642 tokens: cosine 0.99906 after the final norm, 0.99957 on the
  span; the tower runs ~3% slower (a quantized matmul over 5,394 rows is compute-bound, [measured](dsv41-memory.md#vision-requant)).
  The reference-parity tests build the tower with `initWith(.., false)`.
- **Trunk** (`dsv41_model.beginPassWith`): the chunk's image positions take the span rows in order
  (`VisionRows.offset` = rows earlier chunks consumed, so a prompt chunks anywhere); they pick experts by the layer's
  `gate_bias_vl` instead of `gate_bias` (the route kernel reads a `[L, E]` bias row by row; `--drop-expert` masks it
  too); `engramWrite` leaves their stream untouched; their Engram id is `Hash.DEAD`, so no n-gram spans an image
  (`compressedId`, also on a restored history). A one-row chunk holding an image position skips the fused decode FFN.
  The CED replay slices the image mask with its rows. DSpark's draft is text and reads the target layers' streams, which
  already hold the image rows, so an image request keeps DSpark.
- **Prefix cache**: the lookup takes the request's media key and boundary (`lookupAndRestoreDsv41`), so ids that
  repeat under another image restore only the text before it; the splice resumes at the image rows inside the restored
  prefix; the tower encodes only the images the restored prefix does not hold whole, so a 450k-token history of
  screenshots costs no tower pass.
- **Chat**: the template reads `content` as a string, so `ChatConfig.image_placeholder` renders each image as the atom
  where it sat, blank-line joined with its text (the release's `process_image_messages`); `placeMedia` expands the atom
  to `spanRows` tokens.
- **Bill**: the tower as requantized and the image biases (0.50 GB) are the budget line's `vision` term (`dsv41_source.visionBytes`,
  `streamingResidentSplit`), a request's encode is billed by `visionEncodeBill`, and a load with the tower keeps
  `largestImageEncodeBytes` beside the budget ([engine-memory-admission](engine-memory-admission.md)).

Proof: `python3 tests/dump_dsv41_vision_fixtures.py --ref-dir <release>/inference --pack <pack>` runs the release's
`image_processor.py` and `vision.py` on CPU in bf16 (its default dtype) on two fixed images, with the pack's tensors,
and writes the patches, tower, aligner and span (and an fp32 run of the same tower) under
`$SUSHI_MODELS_DIR/kld-teacher/dsv41-vision`. `zig build test -Dtest-filter="dsv41 vision"` holds the patches to the
reference bit for bit, the embedding to cosine 0.999998 and block 0 to 0.99999, and the 32-block tower and span to cosine
0.9957..0.9986 and 0.9984..0.9995 against the reference's bf16 run. That drift is bf16, not the port: the reference's own bf16
tower sits at cosine 0.9942..0.9982 from its fp32 run and ours at 0.9939..0.9975, and the test asserts ours is within 0.002 of
the reference's distance. The trunk's splice, routing, Engram and chunking proofs run on the tiny pack
(`DSV41_TINY`, `-Dtest-filter="dsv41 vision"`); `tests/test_dsv41_media.sh` is the live check.

## Not done

The f16 attention arm's KLD run (the kernel is in, `Model.attn_f16` off); the linear-input fake-quant arm.

## Lessons

- A resident-weight scan by `*.safetensors` swept the fixture files in the same directory: a duplicate tensor name
  leaked its key and replaced its value. The scan is `model*.safetensors` and a duplicate is refused.
- `I8` and `F8_E8M0` are byte payloads to the expert reader (`expert_io.isByteDtype`); before, they parsed as `other` and
  every DeepSeek expert tensor was refused.
- The "slow state" (half-speed decode that came and went, blocked waits at layers 2 and 20) was Metal compiles on
  the inference thread: the index top-k width, the packed-KV and QAT row counts, the rope tail's element count and
  the attention split count were template args, so a context under 512 tokens (the ratio-1 layers' top-k width grows
  every token) or a new prompt length built new libraries. They are read at run time now. Row counts of 1..16 stay
  template args of the row kernels (register tiles), so the load runs every decode width and DSpark verify width once,
  then a CED chunk of 256 rows and a 42-row chunk, on a throwaway state (`dsv41_model.warmKernels`, logged `[dsv41]
  kernels warmed ... in N ms`); after it nothing compiles but the streamed index top-k past 32k keys
  (`dsv41 kernels:` tests, `SUSHI_KERNEL_BUILD_LOG=1`).
