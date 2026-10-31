# Process: microbenches (`sushi-ubench`)

A microbench times one kernel or one forward across its rungs. It is a measurement, never a test: it lives in its own
binary, `zig build test` never builds or runs it, and its exit code says whether it measured, never how fast. A test
never asserts on time ([testsuite](testsuite.md)). The release bench gate ([process-measurement](process-measurement.md))
covers the shipped models end to end; a microbench answers the per-kernel question on any model, as ms across rungs.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [process-measurement](process-measurement.md),
[perf-baselines](perf-baselines.md).

## Build and run

```sh
./.zig-toolchain/zig build ubench          # zig-out/bin/sushi-ubench, always ReleaseFast
taskpolicy -a zig-out/bin/sushi-ubench list
taskpolicy -a zig-out/bin/sushi-ubench run dsv41.experts --rungs rows=1,2,4 --rounds 9
zig-out/bin/sushi-ubench compare a.json b.json
```

| flag | meaning |
|---|---|
| `--rungs <axis>=<v,..>` | rungs to time on one axis (repeat per axis); default every rung the bench declares |
| `--rounds N` | same-process rounds per rung (default 7) |
| `--steps N` | dependent calls per round |
| `--arms a,b`, `--ref arm` | arms to time, and the arm ratios are read against |
| `--pack <dir>` | the pack a pack bench loads |
| `--opt <k>=<v>` | a knob the bench declares |
| `--out <dir>` | where the JSON goes (default `~/.sushi/runs/ubench`) |
| `--contended` | skip the quiet-box steps except the lock; the result is stamped CONTENDED |

Exit codes: 0 measured; 1 bench error, invalid run, or `compare` refused; 2 bad flag or value (refused before anything
runs); 3 the box refused the run.

## The quiet box, taken by the binary

1. Refuse a binary older than HEAD or built from a dirty tree (untracked files do not count).
2. Refuse below `taskpolicy -a` priority, or while a `sushi` server or another GPU tool runs.
3. Take the GPU lock as `ubench:<bench>`, then touch `~/.sushi/runs/QUIET` and wait until no `zig`, `clang` or `ld` runs.
4. Refuse on battery.
5. Fans to max (renewed every 5 min), idle until the hottest die sensor is under 50 °C (at most 180 s), run, fans back
   to auto, release the lock, remove QUIET.

A `--contended` number is a smoke, never a recorded baseline.

## Reading a result

- One table per bench: rows are rungs, columns are arms. A cell is the median (min-max) of N rounds in one process,
  the arm order rotated each round, plus the ratio to the reference arm and the bench's metric columns (MB read, GB/s,
  share of roofline).
- The header stamps commit, binary sha and mtime, QoS, power source, die temperature and the lock holder.
- `compare` intersects rungs and arms of two results of one bench and marks overlapping ranges `~` (noise, not a
  verdict). It warns when chip, QoS, dirty state or rounds x steps differ, and refuses a contended-vs-quiet pair.
- Record a number the [process-measurement](process-measurement.md) way: commit and settings beside it, the JSON path
  in the private raw-file index.

## Adding a bench

- A bench is a `report.Bench` entry in the registry (`src/ubench_main.zig`): name `<scope>.<what>` (scope `qwen4`,
  `mimo2`, `glm5`, `dsv41`, `enc` or `shared`), kind (kernel on synthetic tensors, or pack), unit, steps, axes, arms,
  reference arm, opts, and a body.
- The body calls `run.interleave(label, ctx, f)` per rung, then `run.metric` / `run.invalid`. Chains of dependent calls
  come from `src/exl3/bench_util.zig`, so launch latency counts as it does in a forward.
- Bench files (`src/ubench_*.zig`, `src/exl3/ubench.zig`, `src/encoder/ubench.zig`) hold no `test` block: the build
  refuses one. A bench's correctness half (bytes against an oracle) stays a test beside the code.
- Knobs are flags (`--opt`), never env vars.

| bench | scope | what it times |
|---|---|---|
| `dsv41.experts` | dsv41 | the 384-expert PAIR1 grouped chain at the plan's layers 2, 5, 23 and the DSpark stage bank, by rows and routing mode |
| `dsv41.tail_routed` | dsv41 | the CED decoder tail's routed experts at the plan's layers 2, 5, 23: the prefill sorted GEMM against the multi-row decode chain, 8 to 256 rows |
| `dsv41.shared_prefill` | dsv41 | the shared expert beside the routed prefill chain at the plan's layers 2, 5, 23: its own chain, riding the routed launches, and absent, 17 to 2048 rows |
| `dsv41.picks_span` | dsv41 | one attention span over a packed compressed store: picked rows unpacked and gathered then the fused kernel, against the fused kernel reading them packed; peak MB |
| `shared.exl3_decode_chain` | shared | the EXL3 decode chain at Flash-Next geometry for any rate (n24 to n128, MCG and SPEC1, 1 to 4 rows) against n32, n48 and n64 |
| `shared.exl3_grouped_decode` | shared | the decode chain at 1 to 8 rows with verify rows sharing expert reads: ungrouped, prepared, pairs, sized member capacity, on Flash-Next and DeepSeek geometry |
| `shared.exl3_moe_layer` | shared | a Flash-Next MoE layer (E512, K4, top-10) on MUL1, MCG, STD1, SPEC1 and PAIR1 against MLX's affine-4 `gather_qmm` x3, from 1 to 2048 rows |
| `shared.exl3_prefill_gemm` | shared | the sorted prefill GEMM of one projection at Flash-Next and MiMo shapes: served body against the reference body, a rate against n32 and n64, by slot count |
| `shared.exl3_trunk_rows` | shared | DeepSeek-V4.1 trunk linears by width (1 to 256 rows) over distinct banks: the served arm, NAX at every width, the indexed GEMV, mxfp8, affine-4 and a streaming floor |
| `glm5.routed_t1` | glm5 | GLM's one-row routed chain (E288, n36 and n40, MCG w14) up to each phase, per layer: lane GEMVs against the one-row cooperative ones |
| `dsv41.index_decode` | dsv41 | a decode block's indexer stages over packed keys (scores on the scalar and the integer matrix kernel, the candidate-block gather on both, mask applied, top-512, candidate-source blocks, scores + top-k), by keys and rows; the setup refuses a matrix arm that differs by a bit |
| `dsv41.index_peak` | dsv41 | one wide span's indexer, whole scores against the streamed top-k, by context; MLX peak beyond the inputs in MB, reported as the lowest and the highest over the rounds (when freed buffers are released varies from run to run, so the floor and the worst case differ; a bill covers the worst) |
| `dsv41.index_block_peak` | dsv41 | a 16-row decode block's indexer with and without the candidate mask, by context; peak MB (min and max) |
| `dsv41.rows_growth` | dsv41 | a full row store growing by one chunk: time, held MB and the peak beyond them (min and max) |
| `dsv41.attn_span` | dsv41 | one attention span at the release geometry: composed core, f16 wide arm, fused kernel; peak MB (min and max) |
| `dsv41.attn_decode` | dsv41 | one decode token's attention core (64 heads x 512, window 128, 512 or 64 picks, 4k or 250k compressed rows, the release's split count); us a call, 40 dependent calls a sample |
| `dsv41.attn_epilogue` | dsv41 | one prefill span's heads as `wo_a` takes them (64 heads x 512, 8 groups, QAT on): the fused kernel then rotation, fp8 round trip and transpose as ops, against the kernel's store doing all three; peak MB (min and max) |
| `dsv41.hc_expand` | dsv41 | the hyper-connection expand of a wide chunk (hc 4 x 5120, bf16): the op chain against the fused kernel, alone (the layer tail) and with the FFN stage's mix and norm kernels, at 512 and 2048 rows |
| `dsv41.hc_expand_decode` | dsv41 | the hyper-connection expand of one decode row (hc 4 x 5120, bf16), 40 chained calls: a thread a column against a thread an output element |
| `dsv41.launch_fusions` | dsv41 | decode kernel pairs that became one launch, as 40 dependent layers: the router and the FFN hc split, the query rotation and the window row, and the chain's link alone; bytes checked equal, at 1 and 4 rows |
| `dsv41.hc_stages` | dsv41 | the wide chunk's hyper-connection dispatches one by one (hc 4 x 5120, bf16): mix, norm, split, expand, the attention stage and the FFN stage against a copy of the stream; arms are the mix kernel's row x mix tile (4 x 6, 2 x 12), bytes checked equal, at 512 and 2048 rows |
| `dsv41.picks_glue` | dsv41 | a decode layer's picks glue (512 picks of 512-wide FP4 rows, some -1) as 40 dependent layers: the op chain against the one kernel, and the link between layers alone, by store rows and query rows |
| `dsv41.topk` | dsv41 | the decode top-k over index scores, one threadgroup a row against the served split, per row of work in one eval of one or eight |
| `glm5.verify_qmm` | glm5 | verify-row projections per stored width (A6, A5) at the trunk shapes: serial qmv rows, the row-streaming kernel, row tiles |
| `glm5.a6_trunk` | glm5 | A6 g128 trunk projections at 2048 rows: qmm against dequantize + dense NAX GEMM, with the count of differing bf16 values |
| `glm5.nax_arms` | glm5 | NAX-shaped arms on this GPU's MLX (A6 dense-once, MLA head batches, three-row verify batch, KDA cluster) against the staged arms, with how far outputs differ |
| `glm5.kda_prefill` | glm5 | one synthetic T2048 KDA layer, stage by stage, plus the whole reference layer |
| `glm5.mla_decode_attn` | glm5 | one decode row's MLA attention as a 16-layer dependent chain, cumulative stages, by context |
| `glm5.mla_chunk_attn` | glm5 | whole sparse attention of a short chunk on a kv8 state: scalar, packed FP32 composite, packed NAX, served |
| `glm5.mla_chunk_stages` | glm5 | a 2048-row chunk's index scores, top-512 selection and whole attention, by context |
| `shared.kernel_launch` | shared | the host side of one custom-kernel launch on tiny f16 inputs with a 1k or a 32k header, us a launch over 64 launches: the apply alone, and the apply plus its share of one eval of the batch |
| `shared.exl3_rate_groups` | shared | one Flash-Next MoE layer as uniform n32, the masked `moeGroups` and the one-dispatch grouped chain: seven rate layouts at decode rows 1 and 4 and three at prefill chunks 512 and 2048, SPEC1 and PAIR1 |
| `shared.exl3_rate_groups_rows` | shared | a DeepSeek-V4.1 layer (384 experts, four PAIR1 groups) through the grouped prefill chain with every expert routed 1 to 48 rows, as ms per layer, us per expert and TFLOP/s |
| `shared.shared_expert_fold` | shared | one MoE layer with the shared expert separate, folded into the routed launches, and absent, over 8 distinct banks: Flash-Next (one bank, rate groups), GLM (one bank), DeepSeek-V4.1 (layer 2's four PAIR1 groups, with the fused chain's launch count), at 1 to 8 rows |
| `enc.factor` | enc | the block-LDL factor of a Hessian bank at n512 x16 and n2560 x32: float32 on the GPU against the float64 CPU oracle, ms per matrix |
| `enc.search` | enc | trellis search at K2.5 (n40, MCG, window 15) over 4096 random tiles, sustained back to back, us per tile |
| `mimo2.global_decode_attn` | mimo2 | MiMo global-layer decode attention (64/4 heads, qk 192 / v 128) as a 9-layer dependent chain, each layer over its own packed cache, chain overhead subtracted, per layer from 2k to 512k keys: dense SDPA, dequant rebuild, matmul2d (NAX only) and split-K, with GB/s; `--opt kv-quant=4` for the 4-bit cache |
| `mimo2.prefill_attn` | mimo2 | MiMo prefill attention of one chunk (`ql` rows at the end of `kl` keys; `kl` = `ql` is the chunk alone, a smaller `kl` is skipped), us per layer: a global layer's served kv8 chain against pre-dequantized K/V (TFLOP/s, kernel dispatches), and a sliding layer's ring on the fused band kernel against composed SDPA; read the sliding row with `--ref band` |
| `mimo2.fp8_gemv` | mimo2 | the FP8 trunk linear at MiMo's per-rank shapes (qkv global and sliding, gate/up, down) by rows: GEMV, the 9-128-row tile (NAX only) and the dequant route against bf16 and affine-8, with GB/s |
| `mimo2.vision_encode` | mimo2 | the MiMo-ViT encode of one image at 14x14, 68x120 and 96x96 patches on the real tower (`--pack` the MiMo checkpoint), with the scratch peak beside `encodeScratchBytes` |
