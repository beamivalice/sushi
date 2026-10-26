# Engine: EXL3 trellis experts (`expert_layout == .exl3_k4`)

How routed experts in turboderp's EXL3 trellis format are decoded and multiplied: the rate, codebook and window a
pack names, the prefill GEMM and the four-dispatch decode chain, and the parity bars their tests hold. Read this
before touching `src/exl3/`, `src/expert_quant.zig` or `moeExl3`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [pack-format](pack-format.md) (the on-disk contract),
[engine-kernels](engine-kernels.md), [engine-expert-streaming](engine-expert-streaming.md),
[perf-baselines](qwen4-perf.md#exl3), [quality-kld](quality-kld.md).

## Code map

| File | Role |
|---|---|
| `src/expert_quant.zig` | Expert layout detection from PACKED shapes: `.quantized_split` (affine banks) vs `.exl3_k4` (trellis); affine (bits, group_size) solved from geometry |
| `src/exl3/root.zig` | The `sushi_exl3` module's host API: `expert_quant` parse (`parseExpertQuant`), `admitTopK`, `trellisAdmitted`, `moe` (the one dispatch), `moeRagged` (a rate-grouped layer) |
| `src/exl3/codebook_tables.zig` | STQ's named codebook tables: rows, pinned digests, per-table rate range, `resolve` and its refusals |
| `src/exl3/rate_groups.zig` | Rate-grouped layers: the ragged bank (`Ragged`), the one-dispatch mixed-rate decode kernels, the per-group prefill GEMMs |
| `src/exl3/expert_exl3.zig` | Host reference decoders (MUL1, MCG, STD1, SPEC1, PAIR1), `Rate`, `Window`, `Decode`, fixtures |
| `src/exl3/expert_exl3_kernels.zig` | Prefill run-aligned 32-row window GEMM (NAX body, K4 fast branch; simdgroup-matrix body off NAX; scalar body), decode chain (`moeSwigluFused`), `DECODE_ROWS_MAX` (16), `usesPrefillArm` |
| `src/expert_bf16_kernels.zig` | bf16 selected-expert kernels over a slab (`gateUpSwiglu`; `downReduce`) for the unquantized HF checkpoint |

**`src/exl3` is a module** (`sushi_exl3`), so another MLX host (mlx-serve) can serve EXL3 packs through the same code.
It reaches `mlx`, `log` and `io_util` through an `mlx_host` import whose root file exposes them as `pub const`; here that
root is `src/main.zig` (and `src/tests.zig` for tests), and it can never import a Sushi file by path. Its tests run as
their own artifacts (`exl3-test-1` to `-6`, balanced by `exl3_test_weights` in `build.zig`) on `zig build test`.

The NAX compile-probe regression test checks hardware capability independently
of the runtime failure latch. It deliberately latches dispatch off before
probing the real kernel, so an earlier SIMD fallback cannot hide a broken NAX
kernel on supported hardware.

## Format as the engine sees it

- Routed experts are stacked per layer as `[E, ...]` so gather kernels index expert e on axis 0; 16x16 tiles,
  `suh`/`svh` with the H128 Hadamard; `config.json` carries `expert_quant = {format: exl3, k, codebook: mul1|mcg}`
  (plus `window`), or `format: stq` with `codebook: std1|spec1|pair1` and a named table; per-tensor rate read from the
  trellis shape. Every other module stays the affine pack's.
- **A rate is K = n/16**, n the packed halfwords per 256-weight tile (36 = K2.25, 48 = K3, 64 = K4): weight t's
  codeword is the 16-bit window ending at `((t+1)*n)>>4`, so its fresh bits follow from n and the pattern is never
  stored. Even n in [16, 128] admits (K1 to K8); `expert_quant.k` may be fractional JSON.
- **Every reader keys on n, never on an integer K** (`exl3.Rate`, kernel template `NHW`, cache keys,
  `exl3ExpertBytes`); a K printed anywhere reads 2.25, not 36.
- **Every fast path serves every admitted n; a guard test enumerates them** (`every admitted rate takes the fast
  arms`). The funnel readers take their word indices and shifts from n at compile time: eight weights span n/2 whole
  bits, and 16 weights span n. The decode GEMVs read every n through them; the prefill readers every n below 64,
  where a packed K4 branch does the same reads at a word-aligned rate:
  - decode GEMV lane: 64 bits ending at the lane's last bit (at n64, two whole words). A third word is read only
    when the first codeword can start before the two words (every n from 42 to 62 but 48), and then on every lane:
    a branch around that load on the lane's bit alignment kept every byte and doubled the n42 pair GEMV
    ([perf-baselines](perf-baselines.md#exl3-lane-third-word)). The decode GEMVs are ALU-bound, so the third word
    is merged as two 32-bit funnels of adjacent words, never OR-ed into a 64-bit shift (10% of the one-row chain at
    n42); a register shuffle, selects or a branch to save the load cost more than the load
    ([perf-baselines](perf-baselines.md#exl3-lane-funnel)).
  - simdgroup-matrix group: one or two 32-bit funnels, split at the widest weight whose first codeword still fits.
  - NAX fragment: one funnel per quad of weights; above n50 a quad's codewords pass 32 bits, and the funnel reads 64
    bits from three words.
- **The window is a pack field** (`expert_quant.window`, absent = 16, 8..16 admitted): the codeword is masked to the
  window in the one helper every weight kernel inlines, and kernel slots are keyed by codebook AND window. A w16
  bitstream decodes to different weights at every other window, so a window can never come from a flag.
- **The codebook follows the MODEL at every dispatch**: `exl3.moe` calls `kernels.setDecodeParams`
  (codebook + window) before each dispatch because several EXL3 packs can be resident at once; every weight kernel
  inlines `exl3_pairh` from `codebookHelpers`, built per (codebook, window). A codebook name this build does not decode is refused at load
  ([pack-format](pack-format.md#configjson)). A/B lever:
  `SUSHI_EXL3_CODEBOOK_AB=1` on the `codebook A/B` test.
- **`std1` and `spec1` decode through a (multiplier, mask) row of the pack's table** (`Decode.table`;
  [pack-format](pack-format.md#stq)). The kernels take the row as `exl3_hash_row_at(N)`, a constexpr lookup on the
  kernel's rate `N` placed ahead of the helpers, so each row is an immediate of its own pipeline and a new table needs
  no kernel change. A kernel's name and slot row spell the table, since MLX keys a library by name. No pair is built
  in: a decode without a table row panics on the host and a pack naming no table is refused at load.
  - The 2^12 scale is folded out of the GEMV decode: `exl3_decode2` is the weight over `foldScale`, and every kernel
    that reads it passes each stored f32 sum through `exl3_out`. A power of two, so the bytes are the unfolded ones
    (`unfolded_for_test` is the reference). Never fold after an f16 store (into `svh` or `suh`): the unscaled inner
    sum underflows f16. `exl3_pairh`, the MMA operand, keeps the 2^12, so the matrix units never see a subnormal weight.
  - The halves are often subnormal and the half sum rounds on some codewords, the same way on the host and in Metal
    (held over every codeword).
  - The test-only `spec1-test-per-rate` (a different pair at every rate) holds every reader to its own row.
- **`pair1` decodes a pair from one window.** Weights 2j and 2j+1 both come from the window of weight 2j+1 (`cw.y`;
  `cw.x` is never read), so the bitstream and every reader's codeword reads are the scalar ones. The constants are one
  row per rate (`exl3.pair1Pair`); the kernel header is generated from those rows and the kernel's own `N` picks one at
  compile time (the `exl3_pairh`, `exl3_decode2` and `exl3_slot` macros hand `N` to templates), so a rate has no
  kernel copy of its own and a rate with no row fails to compile. A reader must hand `exl3_pairh` a trellis-adjacent
  (even, odd) pair, which every reader does, and the generic one-tile reader decodes a weight through `exl3_slot`,
  which reads its partner's window. The 2^12 folds like STD1's. Window 16 only.
- **GLM's lane, one-row, group2 and prefill-grid kernels serve the codebooks `Codebook.servesGlmFastPaths` names**
  (MCG, std1, spec1, pair1), each held to its own bytes.
- **A shard's `__metadata__` stamp is CHECKED against `expert_quant` before upload**
  (`mimo_source.validateShardStamps`): see [pack-format](pack-format.md#the-shard-stamp).
- `num_experts_per_tok` above 32 refuses by name (`Exl3TopKExceedsReduceBank`).

<a id="mimo"></a>
## MiMo EXL3 packs

A MiMo EXL3 pack serves RESIDENT: its banks nest under `model.layers.` (qwen4's under
`language_model.model.layers.`), `expertStreamingRequired` excepts `.exl3_k4`, and the trunk still takes the
source FP8→bf16 loader (`usesMimoSourceTrunk`), billed dense by `mimoSourceResidentBytes`. See
[mimo2-arch](mimo2-arch.md).

## GLM clamped experts

Moved to [glm5-clamped-experts](glm5-clamped-experts.md).

## Kernels

- **Prefill**: run-aligned 32-row windows, K-generic cooperative readers, the NAX 16x32x16 GEMM body with a K4 fast
  branch; ONE GEMM config reused across window counts (a per-row-count JIT compiled per novel prompt length).
- **At the two served geometries (`mimoPrefillOn`: MiMo, Flash-Next) the routing never leaves the GPU**: one
  threadgroup (a thread per expert, at most 512) builds the window table and the inverse sort order, and the finish
  reduce reads the sorted down plane through that inverse, bytes equal to un-sorting it first. Any other geometry
  builds the table on the host (a sync) and un-sorts with a copy ([perf-baselines](qwen4-perf.md#exl3-gpu-routing-meta)).
- **The NAX body's x loads carry no bounds branch**: a lane's row pointers are clamped into the input once per run
  (a padded row reads a live neighbour whose product is never stored), the k loop is unswitched on the window's
  second 16-row block and unrolled by two. Every row's products are the branch-guarded body's, so its bytes are
  `GEMM_NAX_REFERENCE_SOURCE`'s at every admitted rate (the test's reference); -25% per GEMM at MiMo geometry
  ([perf-baselines](mimo2-perf.md#mimo-prefill-nax-body)).
- **Prefill off NAX** (M1–M4, or NAX declined): the 8x8 `simdgroup_matrix` body computes D = W^T X^T so each lane's
  eight-weight slot group lands straight in its A fragments (the tile layout is the MMA fragment layout). f16 x and
  128-multiple widths only; anything else takes the scalar body. Not byte-identical to the scalar body (sum order).
- The funnel readers' codewords and decoded weights equal the host tile decode's at every n, so the GEMM
  accumulation order and output bytes are unchanged (n32..36 read a group through one funnel, wider rates two).
- **Its block count is compile-time** (`(WIN+7)/8`), never the run's: a data-dependent bound over the
  `simdgroup_matrix` arrays spilled them, 2.6x slower. Short runs pay the padding and still win
  ([perf-baselines](qwen4-perf.md#m2max-64gb)).
- **Decode**: four dispatches per MoE layer — pair prepare, split-K pair GEMV with f32 inner planes, fused mid+down
  GEMV, f32 finish reduce (`moeSwigluFused`; top-k ≤ 32, named refusal above). On MiMo geometry the SwiGLU mid is
  prepared once per (row, expert) (`preparedMidOn`, disabled for Qwen; like `mimoPrefillOn` it keys on geometry,
  never on the rate), and the pair input too, in its own dispatch (`pairPrepare`): five dispatches. A pair
  threadgroup would otherwise re-derive its K span of x (64 times per slot). Rows ≤ `DECODE_ROWS_MAX` or verify rows take this
  chain; wider takes `moePrefill`. The MTP head's MoE rows ride the decode chain and refuse wider
  (`Exl3MtpRowsExceedDecode`).
- **The prepared pair input is stored in GEMV lane order** (`LANE_ORDER_SLOT`: each 16-row tile keeps rows 2q,
  2q+1, 2q+8, 2q+9 at 4q..4q+3), so a lane reads its four as one half4 (`laneQuadReads`). Same values in the same
  order: the pair planes keep the self-preparing kernel's bytes. The same layout for the prepared middle measured no
  gain on the down (mid + down 52.1 vs 51.7 us at 1 row, 145.1 vs 145.9 at 4) and is not taken.
- **Streamed serial decode** can add its already gated shared expert in the finish reduce, removing a dependent
  elementwise dispatch. The routed sum is rounded to its output dtype before the shared addition, matching the
  separate store and add bit for bit. Other widths, mixed gate/up rates and dtype mismatches retain the separate add.
- **Resident bf16 serial decode folds the whole shared-expert tail** (`moeWithGatedShared`): the finish reduce reads
  the shared down projection and the gate logit and adds `sigmoid(logit) * down`, removing Sigmoid, Multiply and Add
  (three dependent dispatches per MoE layer). The gate follows MLX's bf16 Sigmoid step by step (precise exp, then
  reciprocal, then `1 - r` for a non-negative logit, each rounded to bf16) and the product rounds to bf16 before the
  add; the unit test sweeps every bf16 logit against the MLX ops. Other widths and dtypes run the three ops.
- **The decode GEMVs are bound by fixed per-tile work, not DRAM** (64-bit index math, two word loads and a 64-bit
  shift, four input reads, loop control). The lane-funnel arms (`gemvLayout`: every n) carry two output
  tiles per threadgroup, load both k-tiles of an iteration before decoding, and bump pointers; the per-tile
  accumulation order is unchanged, so the bytes equal the one-tile generic reader's (`FUNNEL=0`, the test's
  reference). A layout that changes which simdgroup sums which k-tile (8 simdgroups) is NOT bit-identical.
- A rate on the generic reader decodes ~40% slower per GEMV than on the funnel, with no other symptom. The engagement
  line `[exl3] n<n> funnel engaged arm=<arm>` names the rate and arm in a live log.
- A decode rate kept off the lane layout keeps every byte and loses its speed: n64's own packed branch (one tile and
  one k-tile per step) ran the one-row chain 1.27x n62's; through the funnel it runs 0.93x.
- **A reader change is timed on every served pack's decode before it lands** (forward meter, against its parent): the
  byte-identity and engagement tests pass on a reader that runs at half speed. Speed is owed to K1.5 to K4, the
  range the converters write: the test `every K1.5 to K4 rate decodes within a margin of n48` holds each even n from
  24 to 64 within 1.12x of n48's GEMV steps. Rates below K1.5 and above K4 stay admitted and tested for correctness
  (SPEC1 at every reader, GEMM, chain and f32-truth bar on `spec1-w15`'s rows); `SUSHI_EXL3_WIDE_UBENCH=1` times
  K4.5/5/6/7/8 against K4 (decode, verify rows 3 and 4, prefill GEMM). `SUSHI_EXL3_LOWK_UBENCH=1` on the exl3 test binary prints n24..n48's decode chain and sorted prefill
  GEMMs over n32's at Flash-Next geometry (w8, w12, w15), interleaved in one process; `SUSHI_EXL3_CHAIN_UBENCH=1`
  times the one-row chain at every even n from 24 to 64, the lane funnel against the 64-bit merge.
- **Rates above K4 on the M5 Max** (pre-port tree, GPU lock, QUIET, fans max, AC, `taskpolicy -a`). The one-row chain,
  SPEC1 over SPEC1's K2, median of 7 repeats: K4.5 1.10, K5 1.13, K6 1.22, K6.5 1.27, K7 1.52, K7.5 1.73, K8 1.52.
  - There are two reader cliffs, the same on MCG: n110-112 (about 70 us against 61 at n108) and n114-126 (about 80 us,
    above n128's 70).
  - `SUSHI_EXL3_WIDE_UBENCH`, at 512 experts, ratios over K4 in one process:

    | rate | decode, 1 row | verify, 3-4 rows | prefill GEMM |
    |---|---|---|---|
    | K4.5 | 1.11 | 1.14-1.15 | 1.19-1.29 |
    | K5 | 1.09 | 1.20-1.24 | 1.19-1.23 |
    | K6 | 1.21 | 1.15-1.23 | 1.27-1.29 |
    | K7 | 1.56 | 1.55-1.61 | 1.26-1.27 |
    | K8 | 1.56 | 1.29-1.60 | 1.06-1.17 |
- **Verify rows share an expert's weight reads** (`pairGroupedSource`, `downGroupedSource`; prepared-mid geometry,
  2+ rows): among an expert's slots, each slot whose rank is a multiple of the bucket leads itself and the next
  bucket - 1, decodes each weight once and feeds every member in the single-slot order, so every row's bytes are its
  one-row decode tick's. Served: buckets of 2 on MiMo only. Built and byte-tested at 2, 4, 8 and 16 members for any
  slot count up to 512 (one ballot per 32 slots), on every codec: a bucket above 2 runs one output tile per
  threadgroup, a block of `m` members takes the smallest rung (1, 2, 3, 4, 6, 8, 12, 16) that holds it, and rungs
  above 8 decode a weight once and sweep the members over it. `SUSHI_EXL3_GROUP_UBENCH=1` on the exl3 test binary times
  ungrouped, pairs and the member arms in one process at 1-8 rows on Flash-Next and DeepSeek geometry.
- **A decode GEMV slot is bound by its own FMA and input path**, not the weight decode or DRAM, so deduplicating
  shared experts recovers only 4-7% of the expert kernels at 3-4 rows.
- **GLM's one-row decode GEMVs are latency-bound, not ALU-bound**: two tiles per group and two k-tiles per iteration
  took the chain from 179 to 141 µs per layer ([glm5-clamped-experts](glm5-clamped-experts.md#glm)); judge a decode
  GEMV change on the whole chain meter (`SUSHI_GLM_T1_UBENCH=1`, test binary), not on an instruction count.
- Dead for the decode GEMVs (microbenched): 4 or 8 tiles per threadgroup, software prefetch, 2 simdgroups, a
  threadgroup LUT decode, a 24-bit multiply split, half2 input reads, bitfield extracts.
- Dead for the prefill GEMM on the NAX body (MiMo and Flash-Next, outputs bit-identical, all slower): 64-row windows
  with one decode feeding 4 MMAs (+7-18%), a threadgroup-shared double-buffered decode (+30%), decoding tile k+1
  before tile k's MMA (+27%), 256- or 64-thread groups (+12% at 2048 rows); on the branch-free body: a threadgroup
  LUT decode of the w12 codebook (+32%), a per-k-step threadgroup barrier (+9%), unroll 4 (+20% over unroll 2),
  64-row windows again (+9-13% on gate/up). The kernel is register/occupancy bound: added live state loses.
- Dead for prefill routing: a counting sort (histogram, scan, stable scatter) in place of the argsort, exact with
  ties: -0.03 to -0.16 ms per MoE layer as a component, yet Qwen 2.6bpw prefill 5-10% slower in an A B B A (de867e94 era).
- **The SwiGLU chain is f32**: gate, up, sigmoid, SiLU and their product stay in f32 registers through the multiply
  by the down suh. In f16, MiMo's activations put gate and up near 400 each and the product past 65504, so a whole
  routed row became inf. The next ceiling is the f16 down inner plane (about 2x above the measured peak).
- **The shared-expert add must free the routed output it consumed**: it once retained 1920 MiB per 8192-token chunk
  (the 48k prefill cliff). Owned-copy hidden captures at the chunk boundary; kernel configs dropped on their error
  paths.
- **Levers**: `SUSHI_EXL3_GEMM_WIN`, `SUSHI_EXL3_WIN_ALIGN` (window geometry A/B); diagnostics
  `SUSHI_EXL3_LAYER_UBENCH`, `SUSHI_EXL3_UNION_HIST`, `SUSHI_EXL3_SWIGLU_MAXABS`, `SUSHI_EXL3_GEMM_ARMS` (served vs
  reference NAX body, interleaved, at MiMo geometry; `SUSHI_EXL3_GEMM_COUNTS` replays the per-layer `[exl3-counts]`
  lines `SUSHI_EXL3_UNION_HIST` logs on a prefill).

<a id="rate-groups"></a>
## Rate-grouped layers (`.gN`)

- **The load builds a ragged bank from the pack's `.gN` tensors** (`Ragged`, in `loadSwitchMlpBank`; the pack format
  is unchanged). Each projection's group trellises are concatenated into one flat U16 bank and the six scale planes
  over all experts. A per-expert u32 table holds the gate, up and down n, the group, the three word offsets and the
  index in the group. The `.gN` tensors become views of that storage, so the stored bytes stay resident once, plus 32
  bytes per expert, which the Qwen and MiMo bills count. A layer without `.gN` names never builds one: every uniform
  pack is served as before. GLM's loader binds one bank per layer and does not serve groups
  (`Exl3RateGroupsUnsupported`, at preflight).
- **Decode and verify rows are ONE pair GEMV and ONE fused-mid down** over the layer's rate classes. Each is the
  served body with N constant per `case`, the rate read per threadgroup from the table. The served finish reduce
  follows, folding the shared tail where a uniform layer does.
  - Every slot's planes are the served kernels' bytes at its own rate.
  - So a layer whose groups share one rate is the uniform layer to the byte.
  - The served body is taken verbatim apart from N and the trellis offset, and the layout comes from `gemvLayout`,
    so a change to the uniform decode GEMVs (e.g. K4 onto the lane funnel) reaches every class with no edit here.
- **The pair kernel switches ONCE, on the expert's (gate n, up n) class, with both projections unrolled in a case.**
  A switch inside its projection loop kept the loop rolled: +40% at 21 rates. The class switch costs 1.5-3% at 12-21
  classes ([perf-baselines](perf-baselines.md#exl3-rate-groups)).
- **A pass per group lost at every group count**: a dispatch costs a fixed ~10 us in the chain however few of its
  threadgroups work, and it lost even with host-compacted slot lists (1.10x the one dispatch at 2 groups, 4.4x at 20).
- **Prefill runs each GEMM once per group** over the group's bank view at local expert ids and its own segment of a
  GPU-built window table (`C + E_g` windows from `g * C + base_g`, C = ceil(rows * topk / WIN)). A combine dispatch
  (24 planes at most) joins the planes. N stays compile-time in the NAX body. Prepare, mid and the inverse-indexed
  finish run once over the concatenated scales. Bytes equal the uniform prefill when the groups share one rate. Up to
  512 experts (one window threadgroup).
- **`moeGroups` is the test reference, not a served path**: it decodes every out-of-group slot at local expert 0
  with weight 0, sums each group's slots in f32 and adds the group totals in f32. Its output differs from the
  served path by at most one rounding of the output dtype; a KLD gate judges that.
- **SSD streaming of a grouped layer** (Flash-Next) keeps one slot pool per rate group and builds this `Ragged` bank
  over the slabs (`Ragged.initPools`); MiMo and GLM refuse it ([engine-expert-streaming](engine-expert-streaming.md#rate-groups)).
  A wide route through an overflowing group gets a prefill-only bank (`Ragged.initSplit`): the group's cache pool
  and its spill in the union workspace as two pools, scales concatenated per route; the decode kernels refuse it.
- **Tile-major pools** (`Ragged.tile_major`, DeepSeek-V4.1 STQ streaming only): the fill lands each trellis projection
  of an expert as `[OT/2][IT][2][n]` (`layout.tileMajor`) instead of the pack's `[IT][OT][n]`, so an output tile pair's
  whole K range is one run. Only the first tile and the k stride of a reader change (`retile` over the decode funnel
  arms, `GEMM_NAX_TM_SOURCE`, `GEMM_SIMDMAT_TM_SOURCE`, `GEMM_SORTED_TM_SOURCE`, `TileOrder`); the words, the decode and
  the FMA order are the pack order's, so every plane is the same bytes (held at 1-16 and 256 rows, every body, PAIR1
  and SPEC1). The pack on disk is unchanged; the DSpark stage's resident bank and every other pack keep the pack order.
  A kernel keyed on a tile-major bank carries a `_tm` name. The non-funnel decode arms and a bank made by `Ragged.init`
  stay in pack order; `moeDecodeClamped` refuses a tile-major bank when the funnel is off (`Exl3TileMajorNeedsFunnel`).
- Timing: `SUSHI_EXL3_RG_UBENCH=1` (decode) and `SUSHI_EXL3_RG_PREFILL_UBENCH=1` (prefill) on the exl3 test binary;
  `SUSHI_EXL3_RG_ROWS_UBENCH=1` times a DeepSeek layer with every expert routed the same 1 to 48 rows.

<a id="dense-rows"></a>
## Dense multi-row projection

- `projectRows` computes `projectIndexed`'s dense `y = svh * H128(W . H128(suh * x))` for 1 to 8 rows of one matrix and
  decodes each weight once for all of them. Each threadgroup prepares its own k range for every row in threadgroup
  memory, in chunks of at most 8 KiB.
- **At 4 rows the lane GEMV is bound by its per-row FMAs, not the decode.** Ablations at 10240x2560 K7: dropping
  rows 2-4's FMAs took 67.9 to 51.6 us; stubbing the decode took 67.9 to 66.4 us. At 1 row it is memory-bound.
- Two or more rows therefore take `ROWS_NAX_SOURCE` where NAX exists: the NAX GEMM's weight fragments and one
  16x32x16 `matmul2d` per k-tile, rows padded to 16. Its 4 rows cost about what its 1 row does. At 1 row the lane GEMV
  (`ROWS_SOURCE`) is about 5% faster.
- **Launch geometry is chosen on dependent chains, never on independent calls.** Independent one-dispatch calls
  overlap, which hides a threadgroup that walks the whole k range. Example, 10240x2560 at 4 rows: 64 us independent,
  162 us chained.
  - Policy: k is split into ranges of at most 512 over at least 320 threadgroups, then one finish dispatch.
  - An output of at least 1024 128-blocks (lm_head) folds the finish into a single dispatch.
  - Sweep: `SUSHI_EXL3_ROWS_SWEEP=qwen4|glm5|dsv41` (dsv41: SPEC1 n64 at rows 1, 2, 3, 5, 8).
- The parity bar scores each 128-wide output block in inner space: its error L2 (divided by svh) against the L2 of the
  block's summand magnitudes. The output transform is an orthonormal rotation, so this keeps the inner GEMV bar's
  scale. The composite is `projectIndexed` per row, which shares the bitstream, so it cannot convict a codeword flip;
  the host decode tests own that.

- **Clamped SwiGLU**: `moeDecodeClamped` / `moePrefillClamped` cap the gate at the limit and clip the up branch to
  ±limit after their output scales (DeepSeek-V4.1); the clamp is spliced into the down kernel's fused mid, and the
  prefill mid takes `CLAMP_LIMIT`. The plain entry points pass 0 and keep their kernels.

<a id="dense-linears"></a>
## Dense trellis linears (`linear.zig`)

A trunk weight stored as a one-bank (or, for a grouped projection, G-bank) trellis projection: prepare (suh, H128),
inner product, finish (H128, svh), out in x's dtype, rounded once from the f32 result (`projectRows`,
`finishIndexedAs`). Up to `PROJECT_ROWS_MAX` (8) rows take the multi-row kernel
(`projectRows`: two dispatches, each weight decoded once for every row; the NAX kernel from two rows, the SIMD one at one
row, which measured 1.2-4x faster than NAX there); wider chunks the sorted GEMM over a window table built on the host.
`decodeDense` is the host twin a test multiplies beside it.

- Linears that read the same rows can share one launch (`linearShared`, `projectRowsSegmented`): their trellises stacked
  along the output, `suh` one row per linear, `svh` concatenated, up to four, every boundary a multiple of 128 so a
  threadgroup never straddles two. A threadgroup shifts `suh` to its linear's row (`SB1..SB3`), and the output is
  segment-major: each linear's `[rows, width]` contiguous. Its k split follows the stacked width, so a row's sum order
  may differ from the linear's own launch; each holds its own f32-truth bar (`linears sharing their input`).
- The rows kernel takes a bank `[G, in/16, out/16, n]` with `x` `[G * rows, in]` (the grouped `wo_a`): groups ride
  grid z (`GROUPS`), each shifting its own `x`, trellis, `suh`, `svh` and output. `x` is f16, bf16 or f32 (the load is
  `float(x[..])`, the output f16).
- A kernel that reads `exl3_decode2` must pass its stored f32 sum through `exl3_out` (the 2^12 fold of the tabled
  codebooks). MCG folds nothing, so no MCG test can see a missing one: a new reader gets a std1/spec1/pair1 case at its
  first test.
- A row's bytes are the same at widths 2 to 8 (NAX); the one-row SIMD kernel and the GEMM differ from them in sum order
  only. All three hold the f32-truth bar 1.5e-3 past the output dtype's own rounding of the truth, added in quadrature
  (worst kernel error measured 4.1e-4: spec1 n64/n96, pair1 n32, mcg, G 1/2/8, rows 1 to 70, f16/bf16/f32).
- Both arms store bf16 and f32 rows as such: an f16 store capped an output at 65504 and cost a cast dispatch per
  linear. Two f16 stores remain: the prepared input (`H128(suh * x) / sqrt(128)`; real trunk suh is 0.005 to 0.4) and,
  on the GEMM arm only (prefill), the inner products before the output H128 + svh (shared with the expert prefill
  kernels; real trunk svh is 0.1 to 5, so an output saturates there only past about 6500 to 3e5). The multi-row
  kernel sums in f32 to the end. No real-magnitude capture of the trunk activations exists yet.
- The SIMD arm (no NAX) decodes each k-tile into registers once and loops the rows over it in groups of at most four
  (a longer row loop compiles 2-3x slower); from two rows a threadgroup spans two output tile pairs. Its bytes are the
  pre-split kernel's at every width, and a row equals the one-row launch bit for bit; the tests keep the old loop as
  their oracle. Timings: [perf-baselines](perf-baselines.md#simd-rows).
- DeepSeek shapes at 2 to 8 rows (`SUSHI_EXL3_ROWS_SWEEP=dsv41`, chained, on a loaded box so +-50% noise): the NAX policy is within noise of the best geometry at every shape and width; the SIMD kernel wins only some
  two-row cells, by less than the noise. The policy stands.

- **DeepSeek trunk by width** (`SUSHI_DSV41_TRUNK_UBENCH=1 MLX_MAX_OPS_PER_BUFFER=16`, exl3 test binary at `d810c50c`, M5 Max, GPU
  lock, QUIET, fans max, die under 50 °C, `taskpolicy -a`, best of 7 rounds of 96 calls chained over distinct banks): us a call,
  finish included, the served arm (SIMD at 1 row, NAX from 2) and the NAX arm at 1 row, against the same bytes streamed by
  16-byte loads. Without the buffer cap the host's encode time reads as kernel time.

  | shape (MB) | floor | R1 SIMD | R1 NAX | R2 | R4 | R8 |
  |---|---|---|---|---|---|---|
  | `wq_b` (21.0) | 36.6 | 43.8 | 45.4 | 47.9 | 53.1 | 61.3 |
  | `wo_a` (16.8) | 29.8 | 38.2 | 45.5 | 46.8 | 49.9 | 55.3 |
  | `wo_b` (21.0) | 36.5 | 46.1 | 49.1 | 50.2 | 53.0 | 58.2 |
  | `wq_a`+`wkv` (4.6) | 9.2 | 16.7 | 19.4 | 20.0 | 22 | 23.3 |
  | `idx_wq_b` (2.6) | 5.9 | 10.9 | 11.3 | 11.9 | 12.7 | 14.3 |
  | head n96 (496) | 838 | | | 1001 | | 1137 |

  A layer's four trunk calls sum to 145 / 165 / 178 / 198 us at R1 (SIMD) / R2 / R4 / R8, so the trunk over 40 layers is 5.8 /
  6.6 / 7.1 / 7.9 ms: the SIMD-to-NAX switch costs 0.8 ms of the R1-to-R2 step, not the whole step (the routed experts grow
  faster). `wq_a`+`wkv` is host-bound in a chain and reads ±3 us.
- **What bounds the NAX rows kernel.** At 2 rows `wq_b` with decode, MMA and prepare ablated (loads only) takes 46.9 us against 47.9
  for the whole kernel: decode and MMA hide behind the loads, whatever the load width (8+4 bytes, 8, 16) or batch depth. The
  gap to the floor is launch, the partial stores and the finish dispatch. From 2 to 8 rows `wq_b` grows 13 us: finish 7, partial
  stores 5.5, prepare 1.5. The k split is fixed by the output bytes (the finish sums the partials in split order), so the
  levers below keep it.
  - Ruled out, bit-identical, no gain: weight loads issued 2/4/8 tiles ahead (0 / +15-45% / +30-100% worse), 8- and 16-simdgroup
    threadgroups (±1%), float4 partial stores (0), one coalesced word load plus `simd_shuffle` for the fragment words
    (+8 us), the finish folded into the launch (the last threadgroup of an output block sums its partials in split order,
    arrival counters in a persistent buffer: 0 of 9e7 values differ, `wq_b` -1.1 / -2.6 us at 2 / 8 rows, `wo_b` ±1,
    `wq_a`+`wkv` +2.4 us at 8 rows with 40 splits).
- **k sliced across the simdgroups of a threadgroup (sum order changes).** The NAX rows kernel takes `kph` slices of
  four simdgroups each: slice j walks the k-tiles with `(tk - tk0) % kph == j`, the slices sum in threadgroup memory in
  slice order, and `nsplit` 1 folds the finish into the launch (one dispatch, no partials). A row's bytes still do not
  depend on the row count: the geometry reads the shape alone (`NAX_TUNED`: wq_a+wkv k4 s5, wq_b k4 s1, wo_a k4 s8,
  wo_b k2 s4, head k1 s2; other shapes keep one slice and the split policy). Test: a row has the same bytes at widths 2
  to 8 under kph 1/2/3/4/8 (kph 3 convicted a slice assignment read from the row-dependent chunk).
  Sweep (`SUSHI_DSV41_TRUNK_UBENCH`, 642d8ba6, `MLX_MAX_OPS_PER_BUFFER=16`, GPU lock, QUIET, fans max, `kph` 1/2/4/8 x
  every divisor `nsplit`, 7-12 rounds, same session as the "before"): the spread inside a shape is
  2 to 3 us at the top, so only the shapes below move.

  | us a call, R2 / R4 / R8 | before | after | floor |
  |---|---|---|---|
  | `wq_a`+`wkv` | 20.1 / 21.2 / 23.5 | 17.3 / 19.0 / 20.3 | 9.2 |
  | `wq_b` | 48.5 / 53.6 / 61.8 | 48.1 / 49.9 / 54.0 | 35-37 |
  | `wo_a` | 47.5 / 50.8 / 55.8 | 41.5 / 44.6 / 49.8 | 29.8 |
  | `wo_b` | 50.8 / 54.2 / 59.5 | 48.1 / 51.2 / 57.0 | 36.5 |
  | `idx_wq_b` (not tuned: 8 layers, best arm -0.7 us) | 11.9 / 12.6 / 14.2 | 11.9 / 12.7 / 14.1 | 5.9 |
  | head n96 | 1040 / 1112 / 1204 | 1011 / 1030 / 1089 | 853 |

  A layer's four calls: 167 / 180 / 201 us before, 155 / 165 / 181 after (40 layers: -0.5 / -0.6 / -0.8 ms). The rest of the
  gap to the floor is the load pattern (1 KB runs at a 256 KB stride), not the sum. `idx_wq_b` stacked with `wq_b` would
  save a launch in 8 of 40 layers, about 0.06 ms a token: not taken. The NAX arm at one row is no faster than the SIMD
  arm (wq_b 47.7 vs 43.8 us), so one row stays SIMD.

<a id="dsv41-stq"></a>
## DeepSeek-V4.1 STQ pack: shapes, arms, status

An STQ pack of the release: routed experts PAIR1 (`pair1-w16-v1`, window 16) at K1.5/1.625/2/2.5 = n24/26/32/40,
four groups per layer (five where one expert's down sits below its gate/up: layer 23 has gate/up n32, down n26), streamed;
trunk SPEC1 (`spec1-w15`) n64, head n96, DSpark stage experts PAIR1 n32 resident, embedding affine 4/64.

| tensor (per layer) | codec, n | in -> out | 1 row | verify 2-8 | prefill |
|---|---|---|---|---|---|
| routed gate, up / down | PAIR1 n24, n26, n32, n40 (+ mixed class) | 5120 -> 2304 / 2304 -> 5120, 384 experts, top-6 | grouped pair GEMV, fused-mid down, reduce (`moeDecodeClamped`) | same chain, up to 16 rows | per-group NAX GEMM (`moePrefillClamped`, <= 512 experts) |
| shared expert w1, w3 / w2 | SPEC1 n64 | 5120 -> 2304 / 2304 -> 5120 | one-expert fused chain (`moeClamped`, 4 dispatches) | same | w1, w3, SwiGLU ops, w2 (rows kernel to 8, GEMM past) |
| `wq_a`, `wkv`, compressor `wkv`/`wgate` | SPEC1 n64 | 5120 -> 1280 / 512 / 512 / 512 | rows kernel | rows kernel | sorted GEMM |
| `wq_b` | SPEC1 n64 | 1280 -> 32768 | rows kernel | rows kernel | sorted GEMM |
| `wo_a` (8 groups) | SPEC1 n64 | 4096 -> 1024 each | rows kernel, grouped | same | sorted GEMM, grouped |
| `wo_b` | SPEC1 n64 | 8192 -> 5120 | rows kernel | rows kernel | sorted GEMM |
| indexer `wq_b`, `wk` | SPEC1 n64 | 1280 -> 4096, 512 -> 128 | rows kernel | rows kernel | sorted GEMM |
| head | SPEC1 n96 | 5120 -> 129280 | rows kernel (one dispatch: wide output) | rows kernel | sorted GEMM |
| DSpark stage experts | PAIR1 n32, resident, 128 experts | as routed, top-3 | one group, same chain | rows 5 (the block) | n/a |

**Rows prepare once per slot** (`rate_groups.preparedOn`: 1 to 16 rows at 5120/2304, bytes equal to the self-preparing
chain's). A pair threadgroup re-derived its K span of x (the Hadamard) and a down threadgroup the whole SwiGLU mid and
its Hadamard, 72 and 160 times a slot; the chain now runs `pairPrepare` and a mid kernel (`midMixed`, the clamp in it)
once per slot and the pair and down GEMVs read them through the same rate-class switch. Chain per layer, interleaved
in one process (`SUSHI_DSV41_EXPERT_UBENCH=1`, QUIET, lock): rows 2/3/4/5/6/16 =
285/413/544/672/798/2286 us self-preparing, 241/348/456/561/665/1878 prepared (0.82-0.85x) on layers 2, 5 and 23 and
the DSpark stage bank; about 5 ms off a 6-row verify. Live, interleaved per sweep cycle in one boot (idle box, 180 s fans-max idle, ~1250 rounds), the verify ms are 2 to 7 ms lower at rows 2..8 (1k context: 2-row 53 -> 50, 4-row 75 -> 72, 6-row 103 -> 97; 16k inside noise). A repeated expert costs the same as a new one (the `shared` mode of
the ubench, every row picks the same six: 273/395/515/734/1065 us at rows 2..6, level with distinct experts), so the slot
is bound by its own prepare and FMA path and a grouped-by-expert kernel can only save the weight-decode share (about 1.6 us
of ~26 us a pick), as MiMo's two-member grouping found (4-7%). The one-row chain prepares once too (serial decode, layer 23 bank, interleaved in one process: 153 us self-preparing, 130 us prepared; `pairPrepare` 4, pair 76, mid 4, down 44, reduce 2), which took 23 us off each of 40 layers of a serial token.

**Verify rows decode each distinct expert once** (`groupFor` in `rate_groups.zig`; prepared chain, 2 to 16 rows, up to 512
slots): the pair and down kernels of the chain are the expert-grouped ones (see "Verify rows share an expert's weight reads" above),
rewritten per rate class like the one-slot kernels. The slots of one expert share a slot id and so a rate class; the
threadgroup that leads them reads and decodes each weight once and feeds every member in the one-slot order, as a pair
(bucket 2, two output tiles per threadgroup): an expert picked by three rows is led twice. Buckets 4 and 8 halve the
tiles per threadgroup and measured live (real pack, DSpark sweep, median verify ms, grouped / per-slot) 5 / 7 / 14%
slower at 5 / 6 / 8 rows, level at 2 to 4, so they are built and byte-tested but not served.
Tested bit-exact against the one-slot chain (`group_force = 0`) at 2 to 16 rows, random, all-rows-equal, disjoint and
core-plus-private routes, PAIR1 and SPEC1, four rate groups with a mixed gate/up/down class, every bucket forced.
The saving is the weight
decode and trellis read of the repeats, bounded by the unit measurement above (decode is about 1.6 us of a ~26 us pick).

Every row has a kernel; nothing falls to a generic reader (`every admitted rate takes the fast arms` covers n26). The
indexer's `weights_proj` (32 x 5120) stays FP8 off the 128 grid.

Parity at the pack's geometry: `dsv41 geometry` (5120/2304 PAIR1 n24/26/32/40 and the mixed class, rows 1, 5, 16, 33;
the shared expert chain at 5120/2304 and the tiny pack's 256/128, bytes equal across widths 1 to 16) and, with
`DSV41_RT_PACK=<pack>`, `dsv41 real` (every trunk linear of layers 0 and 2 of an STQ pack on their own bytes, two experts of
every rate group of layer 2): all against a dense f64 product of the host tile decode, bar 2e-3 / 5e-3.

Timing, on a loaded box (best of 40 four-step dependent chains per cell, link floor removed, no GPU lock;
ratios and orderings are usable, absolute microseconds are not; commit and settings in
[perf-baselines](perf-baselines.md#dsv41-stq)): `SUSHI_DSV41_STQ_UBENCH=<shape filter|1>` (trunk linears against the
GEMM, the indexed GEMV, mxfp8 and affine-4) and `SUSHI_DSV41_EXPERT_UBENCH=1` (the grouped chain at the plan's layers 2,
5, 23 and the stage), both on the exl3 test binary; `SUSHI_UBENCH_STEPS` / `SUSHI_UBENCH_ROUNDS` size the chains.

**The trunk GEMM is not a prefill lever.** At 512 rows the sorted trellis GEMM matches MLX's mxfp8 GEMM and comes
within ~15% of affine-4. Times in µs, trellis GEMM / mxfp8 / affine-4, interleaved:
- `wq_b`: 837 / 828 / 730.
- `wo_b`: 870 / 968 / 869.
- `wo_a`: 785 / 766 / 681.
- shared `w1`: 297 / 346 / 300.

A decode-once dense arm (the layer's weights decoded once into f16, then an MLX matmul) is slower at 512 rows
(`wq_b` 1005, `wo_b` 1182) and only draws level at 2048. It was not landed. The trunk is ~10% of a chunk; the routed
experts are about half ([dsv41-arch](dsv41-arch.md#ced)).

## Parity bars

- **Quality bar**: KLD vs the bf16 teacher (`sushi kld capture|compare`), never bytes against the affine pack.
  The EXL3 kernel arms are not byte-identical to any composite (they round once). MTP: EXL3 packs take the chip's
  generic depth row ([engine-mtp](engine-mtp.md#round-cost-table)).
- **A GEMM/GEMV parity bar is relative to the SUMMANDS, never the result** (`Exl3GemmParity`): a trellis dot product
  cancels orders below sum|w·x|, so a result-magnitude floor is seed-locked. Element ceiling = one f16 store + an
  f32 accumulation `in_dim` deep; whole-tensor RMS no worse than 3x mlx's own f16 matmul over the decoded weights
  (`measureInnerGemmParity`). A parity case sweeps `PARITY_SEEDS`, never one chosen seed.
- **Test every arm at REAL magnitudes against a TRUE f32 oracle** with a finiteness assert (outlier residual
  channels, products in the 1e4..1e5 range). A host oracle that mirrors the kernel's own f16 stores cannot see a
  saturation. When a pack "mostly works", capture per-layer max|x|, max|gate*up|, max|down inner| on a real prompt
  first.
- **Score the Metal arms on a real pack's own bytes** (real suh vectors span four decades), not only synthetic
  trellises. The w12 fixture (`exl3_k2p5_mcg_w12_linear.safetensors`) certifies the window convention against the
  converter's own decode.
- A "systematically wrong but not garbage" pack whose reference decoders agree points at live-path numerics OR at
  the pack's own weights along real activations (a converter-side defect; converter details live in the private
  repo), not the bit layout.
