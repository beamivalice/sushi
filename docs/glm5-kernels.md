# Engine: GLM-5.3 native kernels

What the native `glm5_next` forward runs in prefill, decode and DFlash2 verification, the arithmetic contract of each
path, and the alternatives that lost. Every path here is always on for its eligible shape; anything else falls back to
the staged MLX chain. Read this before touching `src/glm5_*.zig`. Architecture, bills and numbers:
[glm5-arch](glm5-arch.md); routed experts: [engine-exl3-experts](glm5-clamped-experts.md#glm).

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-kernels](engine-kernels.md),
[engine-mlx-gotchas](engine-mlx-gotchas.md).

## Code map

| File | Role |
|---|---|
| `glm5_forward.zig` | `Model`/`Request`, layer loop, routing, async2 prefill and async4 decode schedules, captures |
| `glm5_model.zig` / `glm5_next.zig` | KDA layer, dense MLP, affine linears; KDA recurrence, HC collapse/expand primitives |
| `glm5_attention.zig` | IndexPool state, absorbed latent attention, packed prefill orchestration |
| `glm5_latent.zig` | latent storage view (BF16 or kv8), the `SUSHI_LATENT` kernel helper, kv8 row bytes |
| `glm5_attention_nax_packed.zig` / `glm5_indexpool_nax.zig` | head-packed native sparse attention (B16/B32); NAX prefill index scores |
| `glm5_attention_decode_batch.zig` / `glm5_attention_overlay.zig` | native B1/B3/B4 decode and verify attention; verify latent overlays |
| `glm5_mla_prefill_batch.zig` / `glm5_mla_verify_batch.zig` | head-batched MLA prefill projections (affine or BF16 banks); three-row query and three/four-row value verification |
| `glm5_kda_prework.zig` / `glm5_kda_value_rows.zig` / `glm5_kda_fused.zig` / `glm5_kda_prefill_cluster.zig` | KDA prework, R4 recurrence, one-token body and output epilogue, FA/GA/beta cluster |
| `glm5_a6_dense_once.zig` / `glm5_decode.zig` / `glm5_router.zig` / `glm5_activation.zig` | T2048 A6 expansion; copy-free QKV; router; dense/shared activation |
| `glm5_hc_prefill.zig` / `glm5_hc_collapse_simd32.zig` | RMS-fused HC prefill projection; SIMD32 verify collapse |
| `glm5_dflash*.zig` | assistant adapter, tree, layerwise verifier, KDA tape, FFN, row projections, A6 hoist, reserve, scratch, local cache |
| `glm5_stream.zig` / `glm5_vision.zig` | BF16 expert streaming (teacher); vision tower and processor |

## Contracts

- **Exact** means every output and state bit equals the staged chain (for verify: the serial row), proven on real
  checkpoint captures as well as synthetic fixtures. A dispatch counter proves engagement; a flag never does.
- Four paths change numerics (BF16 operands, FP32 accumulators, NAX reduction order): head-batched MLA prefill
  projections, NAX prefill index scores, packed prefill attention and B1/B3 decode attention. The owner accepted
  plain NAX rounding on same-pack forced-logit drift checks that predate the frozen screen below (MLA projections at
  4K/64: mean KL 0.0116, max 0.396; the prefill trio at 16K/32: mean 0.0040, max 0.068). B1 decode attention runs
  inside the 4x512 teacher KLD; the three prefill paths engage only past the KLD prompts' lengths, so none of them has
  a long-context teacher KLD. The BF16 teacher takes all four on a NAX GPU ([below](#teacher-nax)).
- **Drift screen** for a numerics change: two frozen nonrepeated 16,384-ID prompts (code, prose), 24 late-prefix plus
  192 forced rows each; per prompt mean KL ≤ 0.01, max KL ≤ 0.15, top-1 ≥ 95%, mean NLL increase ≤ 0.02, no new
  nonfinite. Bounds are fixed before results; late-prefix rows are far more sensitive than forced continuation.
- One numerical target across serial (M1), verify (M3) and partial rounds; a native mode that cannot run raises a mode
  error instead of falling back to scalar math.
- An append swaps request state handles only after its graph builds; a lazy evaluation failure leaves the request
  unusable until reset, never half-advanced.
- Scalar fallback attention runs FP32 online softmax over the latent cache with split partials merged into the query
  dtype; no query×head×history score tensor or full mask exists, and per query chunk the index-score plane is capped at
  2 MiB and the partials at 8 MiB.
- **kv8 latent** (the default; `--kv-quant 16` keeps BF16): append quantizes each new row once (MLX affine, group 64). Every kernel that reads
  the latent (packed and B1/B3 gathers, scalar and overlay partials) dequantizes through `SUSHI_LATENT`, whose
  expression matches MLX's dequantizer bit for bit; dense prefill and the B1 tail row use MLX's dequantizer. Verify
  tails are the rows' quantize-dequantize round trip and commits quantize the raw rows, so verify equals serial decode.
- **Model gate**: a component win counts only if a loaded-model ABBA gain exceeds control drift,
  |A_last − A_first| / mean(A), on each workload. Several 8–26% component wins below failed that gate.

## Prefill (chunk 2048, two layers pending)

- **Chunk**: GLM output depends on the prefill chunk width: the auto chunk is the widest rung up to 2048
  (`glm5_forward.prefill_chunk`, shared with KLD) that costs no admissible context, and arms compared for identity
  must prefill at the same chunk.

- **Cold MLA** (BF16, more than 8 rows, ending at or before token 2051): latent expanded through the per-head K/V banks
  (64 MiB each at T2048) into native causal SDPA D256, `force_fused`.
- **Absorbed MLA projections**: query absorption (256→512) and value unembed (512→256) run head-batched (`[64,T,D]`)
  as native affine NAX QMM on A6 g128 banks, T128–2048. T2048 query 13.69 → 1.69 ms, value 19.45 → 1.64 ms; rel L2
  0.26%/0.32%; 4K/64 drift mean KL 0.0116. 768 MiB of copies at async2. A BF16 `kv_b_proj` (the teacher) takes the same
  `[64,T,D]` layout as a stored-dtype batched GEMM, within one BF16 ulp of an FP64 oracle like the per-row staged chain.
- **Index scores, tree scorer** (decode, verify, and prefill outside the NAX window): one lane per head forms the
  32 lane partials of the old 32-lane scorer (each sequential over d = l, l+32, l+64, l+96) and joins them in
  `simd_sum`'s order, an xor butterfly over 1, 2, 4, 8, 16 lanes, i.e. balanced pairs in lane order (200000/200000
  probe sums). Exact against that scorer at real magnitudes, every NAX and non-NAX GPU; 16 rows × 32768 pools 4.4 →
  0.93 ms, a decode row at 32768 pools 0.42 → 0.20 ms.
  The shared NAX gate bounds row subgroups per threadgroup: 8 (256 threads) without NAX, 16 (512) with it.
- **Index scores, NAX window** (9–16 query rows, 3584–8192 completed pools): Q `[T·32,128]` × pooled-key tiles of at
  most 2048 pools, scalar epilogue kept (BF16 dot, BF16 ReLU·weight, sequential FP32 32-head sum, BF16 total, −inf
  for future pools). 0/7/6 of 16K/65K/131K scores differ, every 512-pool set kept. The tiles stay lazy until the packed
  tile pair settles (36 MiB per pending layer): a host settle per tile cost 95 vs 21 ms per MLA layer at 32K.
- **Selection**: a prefill tile ranks its top 512 pools in one dispatch per row (radix-16 search for the 512th key,
  every larger key plus the lowest-pool ties at it, each candidate's rank), equal to ArgPartition's stable order
  (higher score first, ties by pool, NaN last). Decode and verify rows keep ArgPartition.
- **Meter**: `SUSHI_GLM_LONGCTX_UBENCH=1` (`_CTX`, `_REPS`) runs one MLA layer on a synthetic kv8 state: a decode row
  and a 2048-row chunk split into scores, selection and whole attention, plus decode and chunk append costs.
- **Sparse attention**: per real query, gather its 512 pools plus raw tail (2051 latent rows) into one BF16 bank used
  as K and V, native SDPA at scale 1/16 for at most 16 queries per graph (16K: 0.69 vs 5.22 ms scalar). Two such graphs
  stay in flight (−18.6% at 8K, −6.9% at 16K), and exactly 32 rows join two unchanged T16 selections into one B32 call
  (whole attention −15.4%, 16K model prefill −4.9% vs 1.95% drift, exact against B16) at any history: the 2 MiB
  score plane no longer narrows tiles past 64K, and the selection planes are billed per history token (64 B).
  Invalid and future slots are zeroed before load, all-invalid outputs after; MLX's automatic input copy is off for
  gathers.
- **KDA**: FA/GA/beta read the same normalized input, so at exactly 2048 rows they run as one prepared BF16
  `[320,4096]` NAX GEMM with compact outputs (exact, −6.9%; 85 MiB resident, built at load on a NAX GPU). A6 QKV and
  output banks are dequantized to temporary BF16 and multiplied by dense NAX at exactly 2048 rows (exact, −7.4%/−5.3%;
  512 MiB at async2). Prework is one dispatch: conv4, BF16 SiLU, Q/K L2, FP32 decay, BF16 beta (−74%). The recurrence
  carries four value rows per SIMD group sharing K/decay/Q loads, one `simd_sum` per member, FP32 state (−22% at 512,
  −34% at 2048 tokens). The output epilogue fuses FP32 RMS, norm weight and sigmoid gate. The conv history keeps raw
  BF16 bits (integer copies of the last three QKV rows: a float round trip flushes subnormals and signed zeros).
- **HC**: at 128+ BF16 rows one kernel fuses widening, RMS (MLX's 1024-logical-thread reduction on 128 threads) and
  the 24-output projection with shared accumulators (exact, −51% at 512 rows).

## Decode (one evaluation per token, submitted every four layers)

- Async4 is bit-identical to synchronous layers but may hold two cache generations.
- KDA QKV in one dispatch over the three resident A6/A8 banks (restricted port of oMLX `multi_qmv`; a concatenated
  bank copy would cost 3.29 GiB), then the fused one-token body: conv, BF16 SiLU, L2, per-channel forget gate, delta
  recurrence, gated output norm. Its unary math variants are chosen by probe at first use.
- Router: FP32 GEMV with sigmoid and correction, then stable top-8 and unbiased normalization (two dispatches; BF16
  weights widened locally). Dense/shared activation: one dispatch over the exhaustive BF16 sigmoid table (128 KiB).
- Attention: B1 per decode row and B3/B4 for verification branches, from one gather source (`[B,2051,512]` bank,
  Q `[B,1,64,512]`), through FP32 GEMMs and a precise softmax on every GPU. A NAX GPU multiplies with MLX's
  block-masked GEMM, which never takes the TF32 path, so the bits do not depend on `MLX_ENABLE_TF32`. Its fixed
  32×32×16 tiles let B3/B4 run together with every B1 bit preserved. Ordinary GEMM still runs each branch separately.
  Multi-row gathers write FP32 directly after the BF16/kv8 read, retaining its rounding boundary and the 32 MiB
  scratch cap. Every cell sits within one BF16 ulp of an FP64 oracle;
  the fused D512 NAX SDPA it replaced missed that on ~70% of cells (rel L2 1.2e-2 against 1.6e-3). Gate (`cc56be2b`
  diag arms, Sushi-2.5bpw, kv8, 4x512 teacher): KLD 0.071569 against the fused arm's 0.071762 (−0.27%), top-1 90.33%
  against 89.70%. Serial decode 34.07 → 32.54 ms/token at 8K and 33.84 → 32.66 at 32K, arms interleaved in one
  process on a contended box; DFlash2 verification per round is unchanged (~60 ms). 32 MiB per pending layer.
- HC collapse on one to four rows (T3 first, then T1/T2/T4: serial decode, tails, lookup and company): one SIMD32 subgroup runs the coefficients and 20 Sinkhorn iterations that thread 0
  ran alone while 255 threads waited (exact; −27.6% component, 8K model −2.65% vs 1.99% drift). T1/T2/T4 in one process on Sushi-2.5bpw kv8 at 1K (`SUSHI_GLM_ROWS_UBENCH`, 24 rounds,
  arms interleaved, `taskpolicy -a`, lock, AC): serial token 30.87 → 28.98 ms (−6.1%), grouped T2 40.92 → 38.98 (−4.7%),
  T4 66.41 → 64.19 (−3.3%); T3 unchanged (51.50 vs 51.41, the control). Bit-exact against the reference kernel at every width.

## DFlash2 verification

- **Prepared three-row inputs**: large A6 projections reuse the input's original BF16 group sums and exact
  power-of-two coefficients across output tiles. HC expansion emits both the rounded residual and its FP32
  normalized view for the next collapse. The default NAX three-row path uses two-layer asynchronous groups;
  explicit schedules and other widths are unchanged. Together these save
  [3.5–3.9% versus the dev2 baseline](glm5-perf.md#glm-three-prepared-input), below the additional 5% target.

- **Three-row MLA value and normalization**: on the NAX path, the A6 value bank reuses its weights across all
  three rows, and HC collapse emits the normalized branch input directly. The latter preserves the intermediate
  BF16 value and native RMS reduction order. Together with expert reuse/tiling, measured verification is
  [5.25–5.98% lower verification time](glm5-perf.md#glm-three-value-norm) than `f40fa548`; draft depth is unchanged.
- **Shared-expert rows serve every admitted n** (`glm_group2.servesRate`) and every pack window, never one pack's
  rate or window. The three-row, 24-slot lane path reuses each expert's decoded weights across up to three matching
  routes and computes two output tiles per threadgroup. Other widths retain two-member reuse and one output tile. Mixed gate/up and down
  rates are exact, and the engagement test runs `apply` at each rate. Together these changes save
  [4.0–4.7% of full verification time](glm5-perf.md#glm-three-output-tiles) at the measured prefixes.
- **Groups**: `verifyGroups` runs several requests' trees (≤ 16 rows) in one layer loop. KDA `project`/`finish` and
  MLA `mlaProject`/`mlaFinish` take every row; `recur` and `mlaAttend` take one request's rows and state. Each group
  equals its solo `verify` bit for bit; `SUSHI_GLM_ROWS_UBENCH=N` (`_CTX`, `_TEXT`) times grouped against serial rows.
- **KDA**: parent-indexed prework over 1–16 nodes, then a tree recurrence holding FP32 parent states locally; the tape
  keeps projected prework and replays only the accepted path. The first-child leaf (chain row 2, fork row 1) is kept and
  aliased on a hit (4 MiB per layer; −25%; 59% hits at 8K, break-even 21%).
- **MLA**: trees of at most four nodes read the committed buffer plus a ≤4-row ancestry tail instead of a replaced
  latent buffer (1.97× at 32K); the native gather and the overlay both read four-row tails (`max_tail`), exact against
  those rows committed. Query projections broadcast the one-row geometry over three rows; value projections do so
  over three or four rows.
  Accepted rows append at commit. An overlay branch writes no shared buffer: the pools its ancestry completes stay in
  `State.pool_tail`, and the tree scorer reads pools from `tail_base` on from it (same per-pool arithmetic, exact).
  Writing them into the reserved pooled buffer copied the whole reservation per branch and MLA layer: at 32K with a
  512K-token budget verify cost 56.27 vs 54.90 ms per round at a 287-token budget; branch-local, 54.97 vs 55.17 (kv8,
  Sushi-2.5bpw + A4, `da9882fb` without and with it, same greedy bytes). Live branch scratch is capped at 256 MiB;
  overlay branches bill no buffer copy, so every branch fits beside the B1/B3/B4 scratch at any reservation, while wider trees
  still bill a latent and pooled copy per branch. Branch groups that do not fit settle in turn and B3/B4 fall back to
  per-node B1.
- **Commit**: the commit hands the request's latent (kv8: codes, scales, biases) and pooled buffers to the accepted
  state before evaluating, so MLX appends in place; a buffer the committed request still shares is copied whole,
  reservation included (BF16, 200K-row reservation: replay 10.3 → 1.3–2.1 ms per round, decode 26.7 → 30.15 tok/s,
  `194351a3`). A failure after the hand-over leaves the request failed.
- **Projections**: affine row tiles reuse each weight group across up to four rows in serial qmv order. The A6
  three/four-row specialization uses complete row tiles and two outputs per SIMD subgroup, reducing per-thread
  accumulators. Combined with batched FP32 attention, verification takes about 11% less time on M5 Max with
  Sushi-2.5bpw/A4 g64 ([measurement](glm5-perf.md#glm-verify-final)). Retained BF16 KDA projections run as column
  GEMVs with rows in the batch grid (exact; stock multi-row `Linear` is not); the router batches up to 16 rows.
  Sampled rounds use the same batched rows: their logits equal per-row serial projections bit for bit (215 real 8K
  rounds, every tape and capture too); per-row projections cost 64.2 vs 54.2 ms of verify per round and sampled 8K
  decode rose 31.15 → 37.25 tok/s (BF16, A4, `11566d93`, ABBA, 2.2% drift).
- **Assistant**: only draft positions 1–2 reach the vocab head, since N2 visits depths 0–1 (exact, −58% readout);
  the temporary 8-row block attends a read-only slice of the last 2047 context rows (assistant forward 11.4 → 4.6 ms at
  32K; assistant rounding changes, target exact); the next context is cropped to 2047 rows before accepted captures
  append (50 MiB bound at any length; commit 4.27 → 0.73 ms at 32K; exact). Once cropped, the 2047-row context is
  below the block-tail gate, so serving drafts append the block to the cache and attend its view.
- **Assistant pipeline** (all exact; [measurement](glm5-perf.md#glm-round-levers)): every layer but the last is
  submitted as soon as it is built; gate, up and the BF16 SiLU product run as one A4 g64 kernel at 8 and 3 rows (each
  projection keeps qmv_wide's per-vector order); the conv finish and its residual add are one kernel; the sliding mask
  is built once per (rows, context rows, anchor offset); the lattice's top-16 is two dispatches (per-stretch top-16,
  then a merge) equal to ArgPartition's last entries, ties to the higher index, NaN above every number.
- **Verify chain trims** (exact): on the three/four-row lane path the shared expert's BF16 output is added inside the
  routed reduce, after the routed rows' own BF16 rounding; the MLA attention rows stay joined into the value and output
  projections; the 32-wide MLA index weights take the column GEMV batch; layer 0 is submitted before layer 1 is built.
- At the end of prefill, latent and pooled capacity for input + max output + 3 is reserved once, so verification never
  grows a buffer. Every array a replay needs is an async dispatch output.
- The reserve's ledger is the sequential peak (grown buffers keep their growth; the one in flight holds old rows, new
  buffer and padding), checked against the request's admission bill less what it holds, never live headroom.

<a id="without-nax"></a>
## Without NAX (M1–M4)

- One decision: `glm5_model.naxArms()` is `verifyQmmNaxAvailable()`, the Qwen/MiMo gate; `SUSHI_FORCE_GPU_FAMILY_FALLBACK=1`
  turns it off on an M5. The load line `[glm] NAX arms on|off` names the result.
- MLX has no fused D512 SDPA without NAX and `force_fused` throws there, so the fused arms may never run on a wrong
  gate. Off NAX the packed sparse tiles send the same gathered bank through FP32 GEMMs and a precise softmax, as
  native decode does on every GPU (`[glm-attn] FP32 composite sparse|native ... engaged`), held per element to an FP64 oracle no worse than
  the scalar arm plus a store flip and 2^-11 of max|V|. At 16K: 0.628 vs 1.845 ms per 8-row tile, 0.505 vs 1.045 ms
  per decode row against the scalar latent attention, the same error ([perf-baselines](glm5-perf.md#glm-nonnax)).
- Packed tiles take eight rows (67 MB, inside the 128 MiB tile bill). B3 and B4 run each branch through a B1 GEMM:
  MLX picks GEMM tiles and split-K by batch size, so a batched B3 differed from B1 in the last bit. B1/B3/B4 bill
  32 MiB per pending layer.
- MLX runs FP32 GEMMs as TF32 on a NAX GPU (`MLX_ENABLE_TF32` defaults on), so the sparse composite rehearsed on the
  stock libmlx is looser than on an M1–M4; its tests widen the bar only when a probe GEMM shows TF32. Another GLM
  decode op still follows that default: with the fused decode attention, `MLX_ENABLE_TF32=0` moved the 4x512 KLD
  0.071762 → 0.071636 (op not yet identified).
- The scalar latent attention is the non-NAX teacher's arm and serves every shape the native arms decline
  (`[glm-attn] scalar dense|sparse latent attention engaged`).
- Also off: NAX index scores (the tree scorer serves them), the KDA cluster (three GEMMs), A6 dense-once and MLA head/verify batches
  (affine QMM). Their transient bills drop with the gate.
- The KDA body, prework, post and FP32 router are plain SIMD kernels whose unary variants are probed against MLX on the
  device; they run on every GPU. The 1024-thread KDA body compiles to 24 GPRs for G13/G14 (`metal-tt`), inside M1/M2's
  1024-thread cap; every other GLM kernel dispatches at most 256 threads.
- The T2048 grid takes NHW from each projection's trellis and keys its config cache on the rate.
- Routed experts: the T2048 grid declines off NAX and the sorted chain takes the simdgroup-matrix body
  (`[exl3-gemm] simdgroup-matrix body engaged`); decode lanes and group2 rows are SIMD already.
- The cold MLA D256 `force_fused` SDPA has a non-NAX steel kernel (256 threads); vision uses stock D64 attention.
- MLX picks its own NAX kernels from the device, so the env lever covers only Sushi's arms. A full M1–M4 rehearsal on
  an M5 also loads a NAX-less libmlx (the same pins built at deployment target 26.0, so `MLX_METAL_NO_NAX`) through
  `DYLD_LIBRARY_PATH`. Every GLM unit test passes on both.

<a id="teacher-nax"></a>
## BF16 teacher on NAX

- One decision, `glm5_model.enterTeacher()`: `reference_numerics = !naxArms()`, so a NAX GPU runs the arms a served pack
  runs and any other GPU the reference arms (`SUSHI_FORCE_GPU_FAMILY_FALLBACK=1` reproduces the old teacher). The
  `teacher` flag, not `reference_numerics`, keeps the latent BF16 (`GlmTeacherLatentMustBeBf16`).
- Weights stay as stored: the BF16 trunk needs no A6 dense-once (its linears are dense GEMMs already) and the KDA cluster
  banks are BF16 for every pack; only the MLA head batches had an affine-only guard, now `run` takes BF16 `[64,256,512]` banks.
- At standard4's lengths (chunk 512 dense prefill, ctx 2048) the capture engages HC prefill, KDA value rows, SIMD32 HC
  collapse (all bit-exact against the staged chain) and B1 decode attention (the only changed numerics: FP32 composite
  with block-masked GEMM instead of the scalar latent kernel).
- Unreachable by shape, covered by arm tests only: KDA cluster and A6 dense-once need a 2048-row chunk (the capture's
  chunk is at most 512), packed attention and NAX index scores need history past 2051, MLA head batches need a
  non-dense chunk (also past 2051).
- The capture's reserve bills the arms it takes (`glm5_stream.armsReserve`: cluster banks resident, each arm's scratch for
  one pending layer); it is zero under the reference arms.
- Routed BF16 experts are MLX `gather_mm` on every GPU; MLX picks its own kernel from the device.

## Ruled out

Prefill attention and index:
- Masked full-history NAX D512 attention: 2.2× scalar at 32K but visits every history tile; exact-membership variant
  37% slower at T2048/16K (7.9× the K tiles).
- Indexed K/V fragment loads instead of the gather: exact, 85% slower (130.4 → 241.4 ms).
- Ranked top-512 for decode rows: one threadgroup per row is equal to ArgPartition at 32768 pools and slower at
  262144. At short context the first radix selector measured +1.6%, 2/11.
- Two-tile cadence inside the NAX index scorer (−16.3% component, −0.45% model at 16K): superseded by keeping every
  tile lazy until the tile pair settles.
- Tree scorer variants: FMA partials (matched on random data, no faster), keys or queries held as FP32 register
  arrays (spills, 5–7× slower), the partial terms as a loop over the head width with per-lane head ternaries (6×
  slower than four statements and a clamped head).
- Index-score NAX cap 8192 → 8448 pools for the 32K tail: −9.7% component, code max KL 0.345 on the 32K screen.
- Four-query shared-bank retrieval: −53% attention, code mean KL 0.513 (70% recall still fails).
- Cold absorbed D512 packed MLA instead of expanded K/V: +523%.
- Direct single-split prefill finalization: exact, no controlled win ever measured; removed.
- Steady 4096-row chunks after the first 2048: moved IndexPool NAX engagement, code mean KL 0.034.
- Head-batch hi/lo precision restoration: less drift, several times slower; plain NAX rounding accepted.

Prefill KDA and trunk:
- Staged (threadgroup) recurrence schedules: +7.8% to +65%; threadgroup height 8/16: −3–4%, superseded by R4.
- R8 value rows: −1.6% (8/11), about 2.4 ms per T2048 prompt.
- Temporal 128-token KDA tiles: +11.8%, 0/11.
- A6 dense expansion at T1536–2047: −7.9% component, +3.1% on the equal-memory model gate.
- Request-lifetime BF16 copies of all 136 A6 banks (8.5 GiB): +0.36%, 4/11.
- Affine QMM tile swizzle (≤1%), tile aspect (wide +16–21%; tall WM2/WN2 wrote zeros from row 16), A6 word unpack and
  BM128 tiles (−3–5% primitives, never model-gated).
- Four-output HC prefill expansion: −0.43% component, +0.34% at model level vs 1.9% drift.
- HC projection with 6 or 12 columns per group: +26–28% / +9%. Factored RMS after a BF16 NAX projection: failed the
  drift bound on cancellation inputs.

Decode and verify:
- One-token HC norm/mix fusion: −8% queued component, model 26.49 vs 26.54 tok/s; removed. Short-row HC collapse +
  RMS fusion: 0.4 µs per call, never integrated.
- Whole HC prep in one kernel (RMS, 24 mixes, gates, collapse, sublayer norm; bit-exact, BF16 matrices read in place):
  -5% serial before the 1-4 row SIMD32 collapse; on top of it 0 to +2% at B=1-4, 8K (`de867e94`).
- Joined QKV dispatch (3–4 rows: +0.7–3.0%; later over the three hoisted A6 banks: −0.57%, 6/11): noise.
- Raw A6 four-product unpack: +1.3% (5/11). A6 hoist on the output projection: −2.8%, 7/11, drifting.
- Per-node packed NAX decode attention: +23–60%, 0/6 in all nine cases. Shared-factor split-8 merge: +2–5%.
- Short-row NAX index scorer (M128/K128/C2048): +14%, 0/11. Four-subgroup scalar scorer: 0.83% slower paired.
  Shared-prefix three-query scoring: +18%, 1/22.
- Fused T3 KDA core (prework + leaf recurrence + post): −0.7%, 6/11; removed. Canonical chain/fork recurrence:
  chain 6/11, fork slower. Keeping all three T3 endpoints: −1.5% (6/11) for +272 MiB.
- Retained KDA projections: native NAX batch +5.7%; FA+GA N256 column join +3.7% (N320 changes the GEMV kernel).
- Head-rebatched three-row MLA value QMM: switches to `qmv_wide`, not exact.
- Draft head shortlists (3-bit top-32 over 7 rows; A3 top-32 over 2 rows): −26% readout, decode gain inside drift,
  +265–278 MiB resident; removed. The A4 assistant leaves the premise: the 2-row readout is 1.3 of a 5.3 ms draft,
  at most 2.2% of an 8K round (BF16, `11566d93`).
- A measured-cost round planner (serial / N2 tree / lookup per request, hysteresis, probes): byte-identical, within
  noise of the fixed plan at 8K and 128K (2.5bpw kv8 A4). Prose lands 1.7-2.3 tokens per round against a ~1.9
  break-even, so the always-on tree leaves at most a few percent.
- Narrower trees: N1 (one draft node) loses to N2 at 1K and 30K (37.5–38.4 vs 32.2–33.6 ms per token; BF16, A4,
  arms rotated every 16 rounds in one request, contended box); N2 beats a serial step above ~1.1 accepted per round.
- Wider trees: N3 with every T4 kernel optimized 40.44 vs N2 40.22 tok/s at 8192 IDs (`e1597cc2`, 1.46% drift);
  N4 31.4 vs N2 42.4 tok/s at 512/64. Verify per round grows faster than acceptance.
- Assistant split-buffer block attention (port of MLX's two-pass vector SDPA reading context and block K/V in place):
  exact, +27% draft on the cropped cache; MLX's own kernel is far faster than the JIT port.
- Draft readout through the multi-row affine tiles: no faster at 2 rows and not exact (MLX runs `qmv_wide`).
- KDA tree prework reading q/k/v separately, async readout before the lattice, a second assistant submit per layer,
  the MCG multiply split into 16-bit halves (`mul24` equal, `ushort` +22%), larger MLX command buffers: 0.
- Per-kernel attribution tools: synchronizing verifier markers (halve throughput), xctrace Metal System Trace (no
  shader names or intervals; 95.4% GPU busy overall) and private MLX timestamp hooks (only GPUTimestamp; overlapping
  command-buffer intervals). None attributes decode time by kernel.
