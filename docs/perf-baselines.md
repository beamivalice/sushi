# Performance baselines

The recorded speed numbers for the served packs on this box, the roofline they are
judged against, and the levers already ruled out. Before any A/B, find the matching baseline here and INHERIT it
(Team process in CLAUDE.md); every new number lands here, with its commit, binary stamp, QoS and lock, in the same
landing.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [quality-kld](quality-kld.md),
[engine-kernels](engine-kernels.md), [engine-exl3-experts](engine-exl3-experts.md),
`benchmarks.md` (release columns).

Per architecture: [qwen4-perf](qwen4-perf.md), [mimo2-perf](mimo2-perf.md), [glm5-perf](glm5-perf.md). This file holds
the method, the roofline, the EXL3 kernel timings shared by all three and the release ladders.

## How to read these tables

- Box: M5 Max 128 GB (`Mac17,6`), macOS 27, AC power; the desktop takes a small share of the GPU under every cell.
- Only same-session, same-methodology cells compare. MTP cells are variance (sample across boots). Absolute numbers
  drift up to 15% between sessions with no code change (one K3 pack's no-MTP decode read 58.6 and 54.8 in two
  sessions).
- Tools: `./tests/bench.sh` / llmprobe `--bench-only --full` (median of 3 per rung); `/bench` skill for methodology.
- Every row names its binary commit. A row without one is history, not a baseline.

## Roofline

Measured peaks (mlx 0.32.2, binary a73713d):

| probe | median | best |
|---|---|---|
| streaming read kernel, 1 GiB | 538 GB/s | 558 GB/s |
| streaming read kernel, 64-256 MiB | 519-528 GB/s | 536-556 GB/s |
| copy (read+write) | 547 GB/s | |
| bf16 GEMV 16x(16384x4096) | 560 GB/s | |
| bf16 GEMM 4096 / 8192 | 45 / 51 TFLOPS | 59 / 62 TFLOPS |

The chip is a 600 GB/s class part that delivers ~530-557 GB/s to a read kernel. Decode ceilings below use 600 GB/s,
so real ceilings are ~10% lower.

| pack | bytes per decoded token | ceiling at 600 GB/s | measured | share |
|---|---|---|---|---|
| Flash-Next MCG K3 | 5.58 GB (trunk 4.01 affine-8, lm_head 0.66, routed 0.91) | 107 tok/s | 61.8 | 58% |

## Upstream comparison (decided: no rebase)

- Rebased onto upstream vs main 6755ff2 on the MCG K3 pack, interleaved: MTP
  decode 88.2/81.9 vs 82.0/83.1, MTP off 62.1 vs 60.4, prefill 1643 vs 1598: neutral. Four streams with MTP: 85
  aggregate both ways (our merged-verify decline past width one holds).
- kv8 A/B on the affine 4/8 pack: upstream's `qkvAttnMppKernel` engaged
  zero times (QSA caps keys at 2048); all differences were run-to-run and spec variance.
- Decision: main stays; upstream's kv8 attention kernel and grouped MTP are cherry-pick candidates later, each with
  its own certification.

<a id="exl3-decode-layout"></a>
## EXL3 decode GEMV layout (two tiles per threadgroup)

The lane-funnel decode GEMVs (n40 MiMo, n48 Flash-Next) take two output tiles per threadgroup, two k-tiles per
iteration with both loads issued first, and pointer bumps; outputs bit-identical (see
[engine-exl3-experts](engine-exl3-experts.md#kernels)). Kernel microbench: 47 chained dispatches per round, arms
interleaved, median net of a null chain, `taskpolicy -a`, lock `exl3-decode-layout`; base = the served kernels,
recorded in the research run, new arm with sources read verbatim from the commit.

| geometry, kernel | rows 1 | rows 2 | rows 4 | rows 8 |
|---|---|---|---|---|
| Flash-Next pair (E=512, in-process old arm) | 43.9 → 36.6 | | 133.0 → 98.7 | |
| Flash-Next fused-mid down (in-process old arm) | 59.2 → 44.2 | | 95.4 → 65.9 | |

One tile per threadgroup with the unroll and pointer bumps reads the same as two on the Flash-Next pair at one row
(35.7 us) but loses at four rows (107.2) and on the fused-mid down (54.1 / 86.0), whose SwiGLU prepare two tiles
share: one policy, two tiles.

Live, llmprobe `--bench-only`, no MTP, one boot per arm, `taskpolicy -a`, lock `exl3-decode-layout`, greedy
200-token chat completion byte-identical between the arms of each pair:

| pack, flags | base | new | decode | prefill 2k |
|---|---|---|---|---|
| Flash-Next MCG K3, kv off, ctx 65536 | 61.8 recorded (28d7fab, `--full`) | this change on c4f3f7a (aff4f85) | 61.8 → 66.2 | 1763 → 1845 |

<a id="exl3-rate-generic"></a>
## EXL3 readers for every rate (Sushi-2.6bpw n42, MiMo Sushi-2.25bpw n36)

Before this change, only n40 and n48 read through the lane funnel. Every other rate decoded through the generic window
reader at one tile per threadgroup, including Sushi-2.6bpw (n42) and the shipped MiMo pack (n36). MiMo's prepared mid,
grouped verify rows and GPU window metadata were also gated on n40. Every rate below K4 now takes the funnel
([engine-exl3-experts](engine-exl3-experts.md#format-as-the-engine-sees-it)), and outputs are bit-identical.

Setup: M5 Max 128 GB, 2026-09-27. Base 7ad2f407; new = this change (branch commit 49aad597); both ReleaseFast.
`taskpolicy -a`, GPU lock `exl3-n42`, fans at max, box otherwise idle.

Kernel microbench, us per step:
- 47 chained steps; a copy kernel makes each step wait on the last, and every step draws fresh routing.
- Arms interleaved in one process, median of 11, net of the copy-only chain.

| geometry, kernel | rows 1 | rows 2 | rows 4 | rows 8 |
|---|---|---|---|---|
| Flash-Next n42 pair GEMV (E=512) | 55.8 → 34.6 | 103.5 → 58.3 | 198.9 → 110.7 | 389.1 → 213.1 |
| Flash-Next n42 fused-mid down | 32.0 → 20.3 | 59.2 → 36.9 | 112.1 → 68.9 | 218.7 → 133.1 |
| Flash-Next n42 MoE layer | 94.1 → 59.5 | 172.3 → 100.4 | 329.2 → 188.3 | 628.0 → 354.8 |
| Flash-Next n48 pair GEMV, generic → funnel (reference) | 56.4 → 30.1 | 104.6 → 49.0 | 200.5 → 92.6 | 390.8 → 175.8 |
| MiMo n36 MoE layer (E=256): base → funnel → + prepared mid, grouped | 342 → 157 → 146 | 662 → 295 → 276 | 1271 → 562 → 498 | 2437 → 1104 → 912 |

- n42's lane reads a third word, so its funnel pair GEMV costs ~15% more per step than n48's, and its down ~12% more.
- Prefill GEMM on NAX, per projection, one 2048-token chunk:
  - Flash-Next n42 (20480 slots): 3336 → 2323 us.
  - MiMo n36 (16384 slots): 10350 → 7611 us.
  - The simdgroup-matrix body (NAX forced off) at n42: 7302 → 6858 us.

Live runs:
- Sushi-2.6bpw forward meter: `tests/fwd_ubench.sh`, A B B A, no MTP.
- Sushi-2.6bpw llmprobe: one boot of the new binary against today's recorded 7ad2f407 cells.
- MiMo: A B B A, forward meter at load, then a greedy 1024-token chat twice per boot.
- All runs kv8.

| pack, run | meter | 7ad2f407 | this change |
|---|---|---|---|
| Sushi-2.6bpw, no MTP | ms/forward, 1 row | 19.74 / 19.87 | 17.87 / 17.88 |
| Sushi-2.6bpw, no MTP | ms/forward, verify 4 rows | 33.87 / 33.89 | 27.41 / 27.59 |
| Sushi-2.6bpw, `--mtp`, ctx 131072, llmprobe `--bench-only --rungs 4k,16k,64k` | decode / prefill 2k, tok/s | 79.5 / 1492 | 93.9 / 1680 |
| same | decode at 4k / 16k / 64k | 76.7 / 77.9 / 66.5 | 102.1 / 88.2 / 72.3 |
| same, the 192-token decodes in the server log | round ms at tokens per round | 35.9 at 2.83 | 28.8 at 2.74 |
| MiMo Sushi-2.25bpw, `--mtp`, ctx 131072 | ms/forward, 1 row / verify 4 rows | 28.93 / 74.33, 29.08 / 74.36 | 19.56 / 39.06, 19.59 / 39.13 |
| same, greedy 1024-token chat | decode tok/s | 36.6 / 34.9, 35.1 / 34.4 | 62.8 / 60.7, 62.4 / 60.6 |
| same | round ms at tokens per round | 49-83 at 2.05-2.55 | 39.2-39.8 at 2.30-2.48 |

- Sushi-3bpw at the same llmprobe settings read 94.1 / 100.3 decode and 1660 / 1674 prefill today, so Sushi-2.6bpw
  now matches it.
- Residual n42 cost, same session, new binary, forward meter: 2.6bpw vs 3bpw read 17.93 vs 17.80 ms at 1 row, and
  28.72 vs 27.04 ms at 4 verify rows.
- Greedy 1024-token outputs are byte-identical across arms: MiMo 8/8, Sushi-2.6bpw 4/4.
- `4ca5ece4` (rates K1 to K8, in v1.1.0) lost the n42 gain again; [exl3-lane-third-word](#exl3-lane-third-word)
  restores it.

<a id="exl3-lane-third-word"></a>
## EXL3 decode lane: the third word loads on every lane (Sushi-2.6bpw n42)

`4ca5ece4` put the decode lane's third-word load behind `s != 0` (a lane whose window ends on a word boundary needs
no third word). Every byte stayed the same, but at each rate whose lane reads a third word (n42 to n62 but n48) the
pair GEMV took twice as long and the fused-mid down 1.5 times as long. Sushi-2.6bpw forwards ran 13-15% slower
(bisected against its parent `632d5b5b`). This change loads the word on every lane again. Rates whose lane reads
no third word (MiMo and GLM-2.3bpw n36, GLM-2.5bpw n40, Sushi-3bpw n48, Sushi-4bpw n64) compile to the same AIR in
both arms.

Setup: M5 Max 128 GB, 2026-10-05, fans at max, `taskpolicy -a`, GPU lock `qwen-fix`, interleaved on the FIFO lock
with another worker's boots. Base = main `f21637dc` (binary SHA-256 `fd0e9e21`); fix = base plus this change's reader
line (`fc613aa3`); the llmprobe boot and the last two MiMo boots ran the whole change (`f42f57b5`). All ReleaseFast.

Kernel microbench (scratch harness, not landed): 47 dependent steps per chain; the tree's kernel and the same
source over the base reader, interleaved in one process; median of 11, net of a copy-only chain. Flash-Next
geometry (E=512, 2560 -> 640, top-10), MCG w15, us per step, base -> fix:

| rate | rows | pair GEMV | fused-mid down |
|---|---|---|---|
| n42 | 1 | 74.4 -> 36.9 | 31.2 -> 20.6 |
| n42 | 4 | 208.4 -> 114.8 | 105.0 -> 71.2 |
| n44 | 1 | 70.8 -> 37.6 | 33.3 -> 21.0 |
| n56 | 1 | 73.9 -> 37.9 | 34.3 -> 21.7 |
| n48, no third word | 1 | 32.4 / 31.1 | 18.8 / 18.8 |

Forward meter (`SUSHI_DECODE_FWD_UBENCH=50`, `SUSHI_DECODE_FWD_UBENCH_S=1,2,3,4,6`, `--no-mtp --kv-quant 8
--ctx-size 131072`), ms per forward. Sushi-2.6bpw is A B B A; the parent row is the bisect worker's same-day pair.
Sushi-4bpw took one boot per arm.

| pack, arm | 1 row | 2 rows | 3 rows | 4 rows | 6 rows |
|---|---|---|---|---|---|
| Sushi-2.6bpw, base | 21.45 / 21.46 | 25.50 / 25.91 | 30.61 / 30.47 | 35.49 / 36.84 | 44.94 / 48.58 |
| Sushi-2.6bpw, fix | 18.89 / 18.84 | 22.59 / 23.45 | 25.93 / 26.51 | 30.81 / 30.90 | 39.79 / 40.35 |
| Sushi-2.6bpw, parent `632d5b5b` | 19.16 / 19.20 | 24.73 / 24.91 | 27.50 / 27.87 | 32.16 / 32.63 | 42.98 / 43.73 |
| Sushi-4bpw, base / fix | 16.82 / 16.78 | 21.19 / 21.33 | 24.84 / 24.92 | 29.07 / 29.08 | 38.13 / 38.19 |

- MiMo Sushi-2.3bpw (n36, last layer n64), same meter, 1 row / 6 rows over seven boots: base 20.76, 20.64, 20.06 /
  66.39, 54.09, 60.18; fix 21.87, 21.99, 20.28, 20.34 / 54.49, 55.51, 59.53, 61.49. The spread is boot to boot.
- GLM-5.3 Sushi-2.3bpw (n36) and 2.5bpw (n40), `SUSHI_GLM_ROWS_UBENCH=24` at ctx 1024, one boot per arm, grouped
  rows B=1 / B=4 in ms: 2.3bpw base 30.99 / 63.75, fix 30.35 / 62.01; 2.5bpw base 29.99 / 62.45, fix 31.10 / 64.88.
  The second boot of each pair read faster.
- MiMo's, GLM's and Sushi-4bpw's kernels are the same code in both arms. Sushi-3bpw is not on this box; its n48 reader
  compiles to the same AIR in both arms.
- At every even n from 32 to 64 the fix's pair GEMV and fused-mid down compile to the parent's LLVM IR (value names
  stripped); the base differs at exactly the third-word rates.
- Sushi-2.6bpw, `--mtp --ctx-size 131072 --kv-quant 8`, llmprobe 0.6.12 `--bench-only --rungs 2k`, one boot of the
  whole change: decode 86.8 tok/s (80.8-94.4), prefill 1920 tok/s on the 2041-token prompt, first token 221 ms,
  2.74 tokens per step at the 2.1k rung. The bisect worker's same-day decode cells (`--rungs 4k`, median of 7): parent
  `632d5b5b` 83.2 / 83.4, `4ca5ece4` 70.6 / 71.3 tok/s.
- The guard test `every K2 to K4 rate decodes within a margin of n48` reads n42 at 1.73 over n48 on the base reader.
  On the fix, n32 to n62 read 0.92-1.19 quiet and up to 1.25 with three copies contending; n64 reads 1.34-1.41.
  Its limits are 1.4, and 1.8 for n64.

<a id="exl3-lane-funnel"></a>
## EXL3 decode lane: the third word as two 32-bit funnels (n42 to n62)

The decode GEMVs at Flash-Next geometry are ALU-bound, not load-bound: at n42 the third word cost 10% of the one-row
chain through the 64-bit OR, not through its load. A scratch harness (not landed) swapped the lane read inside the
chain ubench below, n42, us per step, arms interleaved in one process. Run 1: main 60.8; no third load but the OR
kept (wrong bytes) 57.9; no third word at all (wrong bytes) 50.7; the third word by `simd_shuffle` from the lane
that loads it 62.6; one load per lane plus three shuffles 79.7. Run 2: main 59.7; the two-funnel merge 52.1; a
packed three-word load with selects for the wrap lanes 57.8; a branch to that packed load 68.8. Merging the word as
two 32-bit funnels of adjacent words keeps the three loads and every byte.

Setup: M5 Max 128 GB, 2026-10-15, fans at max, `taskpolicy -a`, GPU lock `lane3-opt`, quiet window (no build or test
running), on battery at 100%. Base = main `2df81de2`; new = `2eedbef9`; the live binary was built from
`3da27606`, the same served code before a test limit moved. All ReleaseFast.

`SUSHI_EXL3_CHAIN_UBENCH=1` (exl3 test binary; now `sushi-ubench run shared.exl3_decode_chain`): 48 dependent pair GEMV + fused-mid down steps, E=16, 2560 -> 640,
top-10, MCG, one row; the 64-bit merge and the funnel interleaved in one process; us per step. Re-timed on main
`5feaacf1` (n64 on the lane funnel): 2026-10-16, fans at max, `taskpolicy -a`, GPU lock `lowk-rungs`, quiet window,
on battery at 95%, median of 15. The arms are `5feaacf1`'s reader, built into a later exl3 test binary (e4951a82)
whose lane split was off; `=` marks a two-word rate where both arms are the same code.

| n | 24 | 26 | 28 | 30 | 32 | 34 | 36 | 38 | 40 | 42 | 44 | 46 | 48 | 50 | 52 | 54 | 56 | 58 | 60 | 62 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w15 merge | 49.5 | 48.8 | 48.9 | 49.2 | 49.0 | 49.5 | 49.0 | 49.0 | = | 58.2 | 59.2 | 59.2 | = | 59.1 | 59.6 | 59.6 | 60.4 | 60.3 | 60.4 | 60.7 | = |
| w15 funnel | 49.7 | 49.8 | 49.0 | 49.3 | 49.0 | 49.9 | 49.7 | 49.0 | 50.6 | 51.8 | 53.0 | 53.5 | 52.3 | 53.3 | 54.0 | 53.6 | 53.9 | 54.8 | 55.1 | 54.3 | 49.6 |
| w16 merge | 47.7 | 48.2 | 48.3 | 48.0 | 48.4 | 48.7 | 47.9 | 48.8 | = | 56.6 | 57.9 | 58.6 | = | 57.3 | 57.9 | 57.8 | 59.0 | 57.6 | 58.1 | 58.1 | = |
| w16 funnel | 47.3 | 48.8 | 48.2 | 48.2 | 48.6 | 48.6 | 47.9 | 48.9 | 48.8 | 49.8 | 50.7 | 50.9 | 49.3 | 50.2 | 52.1 | 51.5 | 51.7 | 51.6 | 51.8 | 51.0 | 47.5 |

- Rates that read two words (n24 to n40, n48, and n64 since `5feaacf1`) run the same reader in both arms, so MiMo,
  GLM, Sushi-2bpw, Sushi-3bpw and Sushi-4bpw are untouched; their merge cells differ by run-to-run noise only.
- Forward meter on `5feaacf1` (binary d64a79f2, same settings and meter as below, two boots): Sushi-4bpw 16.07 / 16.10,
  20.01 / 19.99, 23.13 / 23.14, 26.59 / 26.78, 33.97 / 33.78 ms at 1 / 2 / 3 / 4 / 6 rows; MiMo Sushi-2.3bpw 19.65 /
  19.50, 26.92 / 25.97, 31.39 / 31.16, 36.79 / 37.64, 50.48 / 50.90.
- n66 to n108 take the funnel too: 2.4-12% faster per step; n110 to n128 (two-window lanes) are unchanged.
- The margin test reads n42 at 1.006 over n48 and n44 to n62 at 1.02-1.06 (main: n42 1.13); its limit is 1.12.

Forward meter on Sushi-2.6bpw (`SUSHI_DECODE_FWD_UBENCH=50`, `SUSHI_DECODE_FWD_UBENCH_S=1,2,3,4,6`, `--no-mtp
--kv-quant 8 --ctx-size 131072`), boots base, new, new, base; ms per forward, both boots per cell:

| arm | 1 row | 2 rows | 3 rows | 4 rows | 6 rows |
|---|---|---|---|---|---|
| main `2df81de2` | 17.81 / 17.52 | 20.47 / 20.53 | 23.91 / 23.93 | 27.88 / 27.79 | 35.69 / 35.78 |
| funnel `3da27606` | 17.34 / 17.52 | 20.06 / 19.94 | 23.50 / 23.22 | 27.04 / 26.91 | 34.64 / 33.88 |

The forward gains less than the chain (-1.4% at one row, within boot noise; -2.3% to -4.1% at the verify widths):
likely because the served 512-expert bank streams from DRAM, where the cache-resident ubench bank does not.

llmprobe (`--mtp`, six boots) did not resolve the change on battery: prefill, whose kernels are unchanged, drifted 26%.

<a id="exl3-rate-groups"></a>
## EXL3 rate-grouped layers: one dispatch over the layer's rate classes

`sushi-ubench run shared.exl3_rate_groups --rungs rows=1,4`: one Flash-Next MoE layer per step (2560 -> 640, E=512, top-10, bf16
rows, the gated shared expert folded at one row and separate at four), 48 dependent steps, arms interleaved in one
process, median of 9, ratio to uniform n32 as served. Groups split the 512 experts evenly; each slot draws its group
from the layout's mix. M5 Max 128 GB, AC power, fans at max, `taskpolicy -a`, GPU lock, quiet window, ReleaseFast,
measured on the tree this was ported from.

| layout | SPEC1 w15, 1 row | PAIR1, 1 row | SPEC1, 4 rows | PAIR1, 4 rows |
|---|---|---|---|---|
| uniform n32, us per step | 55.3 | 54.5 | 154.4 | 166.1 |
| 2 groups {n24, n48}, 5:5 | 1.047 | 1.041 | 1.033 | 1.018 |
| 3 pair classes (24,24)x3 (24,48)x2 (48,48)x5 | 1.056 | 1.048 | 1.040 | 1.026 |
| 20 groups, per-expert mix (an n56-64 slot per row) | 1.114 | 1.078 | 1.174 | 1.088 |
| the same, that slot at n24-26 | 1.095 | 1.067 | 1.129 | 1.052 |
| 20 groups (12 classes), every slot in the n32 group | 0.972 | 0.972 | 0.965 | 0.995 |
| 21 groups n24..n64, every slot in the n32 group | 0.975 | 0.977 | 0.968 | 1.003 |
| 20 groups all n32, every slot in group 4 (no switch) | 0.959 | 0.949 | 0.952 | 0.972 |

- The masked `moeGroups` this replaces read 1.91-1.94 at 2 groups, 2.47-2.80 at 3 and 16.5-18.7 at 20 in the same run.
- The switch costs 1.3-3.1% at 12-21 classes, measured against the switch-free twin; that twin routes into 25 experts and
  is cache-warm, hence under 1.0. The rest of the per-expert cost is the rate mix.
- A switch inside the pair kernel's projection loop read 1.26-1.29 for the per-expert mix; the class switch is the fix.
- One pass per group (each pass skipping other groups' slots, then a G-plane reduce) read 1.36 / 1.55 / 9.0 at 2 / 3 /
  20 groups, one row. Its early exit saved nothing (passes over every slot cost the same): a dispatch costs a fixed
  ~10 us in this chain. Host-compacted slot lists still read 1.10 / 1.35 / 4.4.

<a id="dsv41-stq"></a>
## DeepSeek-V4.1 STQ pack kernels (trunk linears, routed chain)

ReleaseFast, `taskpolicy -a`, M5 Max, NO GPU lock: another job held the GPU. Dependent-chain timings queue behind its
long kernels (a one-row chain read 2-3 ms instead of ~250 us), so each cell is the best of 40 chains of 4 calls with the
chain link removed: orderings and ratios hold, absolute microseconds are upper bounds. Re-take the cells on a quiet box before recording any as a baseline.
Meters: `SUSHI_DSV41_STQ_UBENCH=1` with `SUSHI_UBENCH_STEPS=4 SUSHI_UBENCH_ROUNDS=40` (exl3 test binary,
`-Dtest-filter=UBENCH`); the expert chain is now `sushi-ubench run dsv41.experts` and the trunk table
`sushi-ubench run shared.exl3_trunk_rows` ([process-ubench](process-ubench.md)).

Trunk linears, us per call (and GB/s of trellis read), SPEC1 n64; `gemm` is what the pack used before (the sorted GEMM at
every width), `rows` the multi-row kernel (SIMD at 1 row, NAX from 2), mxfp8 the release's own trunk:

| linear | MB | 1 row: gemm | 1 row: rows | 5 rows: gemm | 5 rows: rows | 1 row: mxfp8 |
|---|---|---|---|---|---|---|
| `wq_a` 5120->1280 | 3.3 | 120 | 24 (134) | 168 | 56 | 31 |
| `wq_b` 1280->32768 | 21.0 | 242 | 104 (201) | 375 | 191 | 265 |
| `wkv`, compressor `wkv`/`wgate` 5120->512 | 1.3 | 97-114 | 20-23 | 101-113 | 23-30 | 24-26 |
| `wo_a` 8 x 4096->1024 | 16.8 | 103 | 51 (328) | 129 | 75 | 74 |
| `wo_b` 8192->5120 | 21.0 | 778 | 85 (247) | 630 | 322 | 660 |
| shared `w1`/`w3` 5120->2304 | 5.9 | 107 | 25 (239) | 99 | 36 | 38 |
| shared `w2` 2304->5120 | 5.9 | 69 | 34 (173) | 73 | 41 | 28 |
| indexer `wq_b` 1280->4096 | 2.6 | 57 | 33 | 68 | 37 | 25 |
| head 5120->129280, n96 | 496 | 1440 | 940 (528) | 1430 | 1410 (352) | 1276 |

- Rows kernel over GEMM: 1.5-9x at one row, 1-4x at five (the head ties the GEMM there); at 256 rows (prefill) the GEMM stays: 8-40 GB/s, level with
  mxfp8 and affine-4 on the same rows.
- One row: SIMD against NAX, same kernel, interleaved: head 940 vs 1342 us, `wq_b` 89 vs 353, `wo_a` 61 vs 113,
  `wo_b` 82 vs 106, `wkv` 19 vs 30, `wq_a` 52 vs 27 (the one shape NAX won). The head at one row reads at the roofline.
- Small linears (<= 6 MB) sit at 20-35 us: a fixed per-call floor (two dispatches), not bytes. Twelve to thirteen of them
  per layer are about half a millisecond of the layer's decode.
- `wq_a` and `wkv` in one stacked launch against two (`SUSHI_DSV41_STQ_UBENCH=shared`, 8-step chains, best of 40,
  same conditions, three runs): one row 29.4-38.1 us against 34.1-47.7 (0.80 to 0.88), five rows 35.7-47.7 against
  43.3-51.3 (0.81 to 1.05).
- The cells time `linear` on f16 rows. Served on bf16 rows a call stores bf16 directly (no cast dispatch); not re-timed.

Routed chain (`moeDecodeClamped` / `moePrefillClamped`, 5120->2304->5120, top-6 clamped SwiGLU, bf16 rows, 16 route sets
so the experts are cold), us per call over the picked experts' bytes:

| layout (plan layer) | 1 row | 5 rows | 16 rows | 256 rows (prefill) |
|---|---|---|---|---|
| L2 n40:5 n32:17 n26:37 n24:325 | 241-259 (40-59 MB, 165-228 GB/s) | 1027 (193 MB, 188 GB/s) | 3666 (587 MB, 160 GB/s) | 35988 (2.6 GB, 72 GB/s) |
| L23 five groups with the mixed class | 217-249 (40-48 MB, 183-192 GB/s) | 999 (229 MB, 229 GB/s) | 3663 (681 MB, 186 GB/s) | 40037 (3.0 GB, 74 GB/s) |
| DSpark stage, 128 x n32, top-3 | 125 (27 MB, 213 GB/s) | 531 (128 MB, 241 GB/s) | 1799 (362 MB, 201 GB/s) | n/a |

- The decode and verify chain runs at 0.3-0.45 of the roofline at every rate mix; cost follows bytes read, not the group
  count (4 or 5 groups, one mixed class: within noise).
- The 256-row prefill decodes every expert of the layer once per call (~13.6 G weights), so its time is the decode rate,
  not the row count: wider spans cost the same per layer until 32 rows per expert, and a streamed layer also reads 2.6 GB
  from SSD per call.

<a id="simd-rows"></a>
## SIMD multi-row trunk linears (machines without NAX)

The multi-row kernel's SIMD arm is what an M1 to M4 runs for every verify width, and what `gemmNaxOn() == false` runs on
any machine. Measured on the M5 Max with the SIMD arm forced (`SUSHI_FORCE_GPU_FAMILY_FALLBACK=1
SUSHI_EXL3_ROWS_SIMD_UBENCH=1`, a since-removed exl3 test meter, `-Dtest-filter="exl3 SIMD multi-row timing"`), main d810c50c plus the
split row loops and `tgrp 2`, GPU lock, QUIET, `taskpolicy -a`. Each call is fed by the previous call's output, before and after
are interleaved in one process, best of 7 chains of up to 48 cold banks; SPEC1 n64 (head and Qwen head n96), f16 rows.
Cells are us per projection, fans on auto; no M1 to M4 timing exists yet. Output bytes are identical before and after.

| linear | | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| `wq_a`+`wkv` | before | 20 | 21 | 26 | 29 | 34 | 37 | 40 | 85 |
| | after | 20 | 20 | 22 | 23 | 24 | 26 | 29 | 30 |
| | after GB/s | 227 | 223 | 213 | 198 | 188 | 175 | 160 | 151 |
| `wq_b` | before | 47 | 50 | 79 | 93 | 115 | 131 | 145 | 243 |
| | after | 47 | 51 | 56 | 64 | 73 | 83 | 93 | 104 |
| | after GB/s | 446 | 413 | 377 | 329 | 286 | 252 | 225 | 203 |
| `wo_a` | before | 40 | 43 | 63 | 74 | 90 | 101 | 113 | 209 |
| | after | 40 | 42 | 45 | 50 | 57 | 64 | 70 | 77 |
| | after GB/s | 418 | 398 | 369 | 333 | 296 | 263 | 240 | 218 |
| `wo_b` | before | 49 | 51 | 75 | 88 | 109 | 122 | 136 | 251 |
| | after | 49 | 51 | 54 | 60 | 67 | 74 | 84 | 92 |
| | after GB/s | 429 | 412 | 388 | 349 | 314 | 283 | 250 | 227 |
| shared `w1`+`w3` | before | 32 | 34 | 48 | 55 | 67 | 75 | 82 | 157 |
| | after | 32 | 33 | 36 | 39 | 43 | 48 | 53 | 58 |
| | after GB/s | 371 | 353 | 329 | 304 | 272 | 243 | 222 | 205 |
| shared `w2` | before | 20 | 22 | 30 | 34 | 40 | 44 | 48 | 92 |
| | after | 20 | 21 | 23 | 24 | 26 | 29 | 32 | 35 |
| | after GB/s | 294 | 286 | 259 | 241 | 223 | 203 | 185 | 170 |
| `head` | before | 894 | 922 | 1077 | 1284 | 1630 | 1841 | 2050 | 3490 |
| | after | 903 | 930 | 955 | 976 | 1029 | 1132 | 1247 | 1380 |
| | after GB/s | 550 | 534 | 520 | 509 | 482 | 438 | 398 | 360 |
| Qwen head | before | 880 | 867 | 912 | 1035 | 1330 | 1483 | 1686 | 3075 |
| | after | 872 | 872 | 873 | 876 | 891 | 915 | 999 | 1091 |
| | after GB/s | 547 | 547 | 546 | 544 | 535 | 521 | 477 | 437 |

- Weights are read once for all rows, so the ideal is the one-row time at every width. One and two rows sit there
  (memory bound, 410-450 GB/s on `wq_b`, 550 on the head). From three rows the kernel is ALU bound: the weight decode costs
  about 23 us on `wq_b` (two to three rows' worth of FMAs) and each row adds about 10 us, an FMA stream at roughly half the
  M5 Max's peak.
- Before, a row loop of five or more rows compiled 2-3x slower (the eight-row cell: 243 us on `wq_b`, +100 over seven).
  After, a k-tile decodes into registers once and the rows run in loops of at most four (`ROWS_LOOP_MAX`).
- Ablations at 8 rows (`wq_b`, 103 us): skipping the activation prep saves 14 us, so a separate prep dispatch cannot pay;
  keeping the prepared rows as f32 in threadgroup memory (no half-to-float converts) is 3-16% slower; threadgroups of 4
  tile pairs match 2 (the policy stays at 256 threads for M1/M2's register-bound thread cap); 1 pair is 10-12% slower.
- Every row of a SIMD multi-row launch equals the one-row launch bit for bit (k split and sum order do not depend on
  the row count): a verify row on a machine without NAX equals its serial step.

## DeepSeek-V4.1 STQ decode: energy per token (2026-11-11)

The baseline for "the same tok/s for less power". Method: [process-measurement](process-measurement.md#energy-per-token).
M5 Max 128 GB, AC, fans at max, STQ 1.75bpw, `--ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000`,
`taskpolicy -a`, GPU lock, quiet box, every load booted at 49 °C and ended at 90 to 91 °C. One load runs four identical
seeded sampled chat requests (256 tokens, temperature 0.8, top-k 40, top-p 0.95, seed 7, fixed three-row rounds) and three
identical greedy completions (384 tokens), 53 and 22 prompt tokens: a short-context reading only. Rows compare **by
position**; the cells are the mean of sampled positions 2 to 4 and of greedy positions 2 to 3, one value per load.

| binary | decode side | sampled tok/s | sampled GPU mJ/token | greedy tok/s | greedy GPU mJ/token |
|---|---|---:|---:|---:|---:|
| v1.6.0 release (`ac3da7aa`) | MLX 0.32.3, no host-overhead patch | 57.1, 57.0 | 1262, 1261 | 70.8, 70.9 | 888, 888 |
| `84f901ad` | MLX with the host-overhead patch | 57.3, 57.5 | 1266, 1257 | 72.1, 70.9 | 879, 894 |
| `2146f69b` | the engine of our own | 56.8, 56.9 | 1271, 1271 | 70.4, 71.6 | 904, 889 |

| binary | server CPU ms/token, sampled / greedy | server CPU mJ/token, sampled / greedy | system W, sampled 3 to 4 / greedy 2 to 3 |
|---|---:|---:|---:|
| v1.6.0 release | 6.9 / 5.1 | 32.6 / 21.5 | 147 / 134 |
| `84f901ad` | 5.0 / 3.9 | 20.5 / 13.6 | 146 / 135 |
| `2146f69b` | 3.5 / 2.6 | 12.6 / 8.4 | 144 / 133 |

- **The three decode sides tie on speed and on GPU energy** (within 1%): the engine's sampled GPU energy is 0.7% above
  1.6.0's. Sampled rounds are fixed work, so the engine's cells repeated to 0.03% between loads; a greedy cell moves about
  1% with the round mix its policy takes.
- **The engine's own CPU cost is 61% below 1.6.0's and 39% below patched MLX's** (CPU joules per token), but the server's
  CPU is 1 to 3% of the GPU's energy here (0.7 to 1.9 W against 63 to 76 W), so the wall reads 1 to 3 W less. Those
  watts are still the ones that count most on a laptop: CPU heat throttles before GPU heat does.
- **Decode is power-limited within one request**: over the three warm sampled positions of a load the GPU falls from 76 W
  to 69 W and tok/s from 58.5 to 55.5 as the die heats, on every side. tok/s is GPU watts over joules per token, so
  under that cap energy per token is speed.
- Idle with the model loaded: 10 to 15 W at the wall, 0 W GPU. The first request after a load: 31, 32 to 33 and 36 tok/s
  for the whole request (1.6.0, patched MLX, engine), 1448 to 1467 GPU mJ/token.
- The first greedy request of a prompt (position 1, the prompt prefilled, its streamed experts cold): 1.6.0 and patched
  MLX 58 to 59 tok/s at 1121 to 1127 GPU mJ/token with 3 plain steps; the engine either the same (59.6, 1089) or, in three
  loads of four, 71 plain steps at 55.5 tok/s and 1158 to 1173 mJ/token (the DSpark stop switch judging against a stale
  one-row price).

**At 122k of context** (the same three decode sides, the engine at `356fc5b5`; one load each, boot 49 °C, end 94 to 96 °C).
One prose prompt of 122,199 tokens (the first 527,954 bytes of the release bench's prose source), completions API,
temperature 0, 256 tokens, four times: position 1 prefills, positions 2 to 4 restore the prompt and decode. The greedy text
is the same on the three sides. About 160 of a request's 210 rounds are plain one-row steps on every side.

| side | prefill tok/s | prefill GPU mJ / prompt token | prefill CPU s | prefill CPU mJ / prompt token |
|---|---:|---:|---:|---:|
| v1.6.0 release | 1253 | 44.6 | 302 | 5.6 |
| patched MLX | 1266 | 44.3 | 296 | 5.8 |
| engine | 1341 | 45.1 | 108 | 3.4 |

| side | decode tok/s, positions 2 / 3 / 4 | GPU W | GPU mJ/token | server CPU mJ/token | system W |
|---|---:|---:|---:|---:|---:|
| v1.6.0 release | 48.1 / 48.9 / 48.3 | 57.7 / 50.7 / 51.5 | 1199 / 1036 / 1065 | 55.5 / 50.0 / 50.3 | 138 / 117 / 117 |
| patched MLX | 48.9 / 48.1 / 48.8 | 57.4 / 55.1 / 55.0 | 1174 / 1146 / 1126 | 30.0 / 30.0 / 28.2 | 136 / 120 / 119 |
| engine | 49.3 / 49.2 / 48.6 | 60.7 / 59.9 / 59.0 | 1232 / 1217 / 1215 | 18.7 / 18.2 / 17.5 | 127 / 128 / 124 |

- **Decode at long context: the engine ties on tok/s and spends 5 to 8% more GPU energy per token than patched MLX**
  (3 to 5 W more GPU power for the same tokens; against 1.6.0 more still at positions 3 and 4, where 1.6.0 fell to 51 W:
  one load). Its GPU is busy 94 to 95% of the decode wall. A second load of each side an hour later read 1149 / 1094 /
  1087 (1.6.0), 1179 / 1114 / 1103 (patched MLX) and 1184 / 1167 / 1156 (engine) GPU mJ/token at 47.8 to 49.0 tok/s: the
  engine level at position 2 and 5% above patched MLX at 3 and 4. Recorded, not chased: watts read this way move with the
  GPU clock and macOS's throttling, and 256 tokens a request is too few to separate watts from tok/s.
- **Its CPU energy per token is 40% below patched MLX's and 65% below 1.6.0's.**
- **Prefill is 6 to 7% faster on main at level GPU energy per prompt token, for 2.7 times less CPU** (108 CPU seconds
  against 296 to 302): a wide chunk's Engram rows are read without `F_RDADVISE` since `0d43d77d`.
- The decode right after the prefill (position 1): 39.3, 39.7 and 47.2 tok/s at 1399, 1404 and 1298 GPU mJ/token.

## DeepSeek-V4.1 STQ on Sushi Cooler: release ladder at `2aa5ddaf` (2026-11-12)

The working baseline of the DeepSeek optimization rounds: `tests/bench.sh --tag cooler-p2c
DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw -- --ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000`, commit
`2aa5ddaf`, binary SHA-256 `d9b45dce6682cb55`, M5 Max 128 GB, AC, fans at max, `taskpolicy -a`, GPU lock, no Warnings
(no kernel compiled inside a request). The box had run no pack load for the 15 minutes before it: boot at 49 °C,
thermal pressure Nominal at every cell. That wait is not a rule ([tiers](process-measurement.md#optimization-tiers)): a
cell is read with its pressure and clock, never against another run's thermal state. Beside it the v1.6.0 release report's cells (MLX decode and
prefill, `ac3da7aa`), the same protocol. Medians of three decode samples; tok/s.

| rung | code decode | v1.6.0 | prose decode | v1.6.0 | code prefill | v1.6.0 | prose prefill | v1.6.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k | 93.3 | 95.5 | 53.1 | 50.3 | 1,300 | 1,242 | 893 | 858 |
| 8k | 99.6 | 100.8 | 58.4 | 56.5 | 1,476 | 1,493 | 1,311 | 1,093 |
| 16k | 76.2 | 76.5 | 54.4 | 46.9 | 1,524 | 1,497 | 1,427 | 1,247 |
| 32k | 88.7 | 91.1 | 54.1 | 51.3 | 1,480 | 1,468 | 1,374 | 1,191 |
| 64k | 83.4 | 84.8 | 54.0 | 51.8 | 1,417 | 1,383 | 1,313 | 1,264 |
| 128k | 75.2 | 75.1 | 52.3 | 52.2 | 1,272 | 1,266 | 1,195 | 1,172 |

- **Prose decode and prose prefill are ahead of v1.6.0 at every rung** (decode 3 to 16% to 64k, level at 128k; prefill
  2 to 20%). Code prefill is ahead or level but for 8k (1%).
- **Code decode is 0 to 3% under v1.6.0 to 64k and level at 128k.** At an equal round mix a verify pass costs what
  the MLX route's does (one binary, both routes, cooled loads back to back at 16k, 32k and 64k: within 3% either way
  at every pass width), so what is left is the round mix a run draws (the 4k cell ran 52 rounds in one run and 55 in
  another) and under 3% of clock.
- **The run three hours of loads earlier (`7eb2da99`, tag `cooler-p2b`) read Moderate and Heavy pressure at the long
  rungs** and 1 to 5% lower there (code 128k 74.2 at Heavy, prose 64k 50.1 at Moderate): the box moved those
  cells, not code. Since it: the draft's Markov kernel (`e27483db`).
- The run before that (`f9cacade`, tag `cooler-p2a`) is not a measurement: one window compiled a kernel inside a
  request (fixed in `771cbaca`), and its copy rounds went to 16 rows after one whole round (`7eb2da99` changed the
  rule), which cost its code cells 88.7 at 8k, 78.3 at 64k and 67.6 at 128k.
- **Where a decode request's GPU time goes** (per-kernel profile of one greedy request, 75 rounds, `7f726d0d`): the
  routed experts' pair and down kernels 37%, the trunk's row kernels 23%, attention 7%, the prefill GEMMs of the
  request's own prompt tail 6%, the draft's stage experts 2%,
  hyper-connections 5%; every small glue dispatch together (finishes, copies, casts, norms) 5%. The GPU is busy 98%
  of the request. Prefill (25.8k tokens): the two GEMM families 62%, attention 16%.

### Prefill after the kept scales (`2c2b36cc`, fast bench)

`tests/bench.sh --tag p2-scales --rungs 8k,32k`, the same pack and flags, the source tree of `2c2b36cc` (on `404e8b38`: a ranged
route hands its pass over before each read, and an expert's scales are read from the SSD once), binary SHA-256
`1ac58f779286f1b8`, M5 Max, AC, fans at max, `taskpolicy -a`, GPU lock, boot at 49 °C, Nominal. Baseline: the table above,
recorded, not rerun.

| rung | code prefill | was | prose prefill | was |
|---|---:|---:|---:|---:|
| 8k | 1,576 | 1,476 | 1,339 | 1,311 |
| 32k | 1,500 | 1,480 | 1,434 | 1,374 |

- The micro indicator that decided it: on one 25.8k-token prompt the GPU's idle time fell from 2,864 to 2,076 ms of
  18.6 s (`[engine] since load:`, uncooled loads back to back, same reply bytes).
- Decode at an equal round mix costs the same a round (code 8k: 56.2 ms a round in both runs on the 52-round sample,
  55.4 against 54.3 to 54.7 on the 47-round one); the code 8k cell reads 93.8 against 99.6 because one of its three
  samples drew a 49-round mix. The ledger gives the scales 0.68 GB, 262 slots a layer for 264.
- What is left of the idle is the trellis reads of the layers that miss the most (sixty experts a chunk on some
  encoder layers, 47 ms of reads against 16 ms of GEMMs to hide them behind).

### Engram batch reads at 64 threads (fast bench)

`tests/bench.sh --tag p2-engram --rungs 4k,8k`, the same pack and flags, the source tree of this change on `2877a95e`,
M5 Max, AC, fans at max, `taskpolicy -a`, GPU lock, boot at 49 °C, Nominal. Baseline: the table above, recorded, not rerun.

| rung | code prefill | was | prose prefill | was | code decode | was | prose decode | was |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k | 1,324 | 1,300 | 1,169 | 893 | 95.7 | 93.3 | 53.1 | 53.1 |
| 8k | 1,528 | 1,476 | 1,390 | 1,311 | 100.8 | 99.6 | 56.1 | 58.4 |

- The micro indicator that decided it: the read pattern replayed on the layer-1 table (two `pread`s a row, random
  rows, arms interleaved, third pass): 3,072 rows 17.2 / 13.0 / 12.3 ms at 16 / 32 / 64 threads, 12,288 rows
  66 / 50 / 47 ms, 49,152 rows 254 / 195 / 180 ms. Past 64 the SSD answers no sooner.
- The bench reads the same corpus every run, so its Engram rows may sit in the page cache: these cells show no loss,
  not the gain. A prompt of fresh text waited 309 ms on the rows of its first 2,048-token chunk (one load, uncooled).

### Attention scores with the keys as the left operand (fast bench)

`tests/bench.sh --tag p2-attn --rungs 4k,8k`, the same pack and flags, the source tree of this change on `a7fd31db`,
M5 Max, AC, fans at max, `taskpolicy -a`, GPU lock, boot at 49 °C, Nominal. Baseline: the Engram table above, recorded,
not rerun.

| rung | code prefill | was | prose prefill | was | code decode | was | prose decode | was |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k | 1,372 | 1,324 | 1,192 | 1,169 | 94.6 | 95.7 | 53.3 | 53.1 |
| 8k | 1,560 | 1,528 | 1,431 | 1,390 | 101.8 | 100.8 | 59.0 | 56.1 |

- The micro indicator that decided it: `sushi-ubench run dsv41.picks_span`, arm `packed_nax`, 7 rounds: a 256-row span
  1.6-1.7 to 1.5-1.6 ms, a 128-row span 0.9-1.0 to 0.9, the same heads byte for byte
  ([engine-kernels](engine-kernels.md)). The kernel is 16% of a long prompt's prefill GPU time, so the cells above
  show no loss; the gain is inside their run-to-run spread.

### Hyper-connection mix on the matrix units (fast bench)

`tests/bench.sh --tag p2-hcmix --rungs 4k,8k`, the same pack and flags, the source tree of this change on `42c364c6`,
M5 Max, AC, fans at max, `taskpolicy -a`, GPU lock, boot at 49 °C, Nominal. Baseline: the attention table above,
recorded, not rerun.

| rung | code prefill | was | prose prefill | was | code decode | was | prose decode | was |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k | 1,400 | 1,372 | 1,228 | 1,192 | 91.0 | 94.6 | 55.8 | 53.3 |
| 8k | 1,546 | 1,560 | 1,435 | 1,431 | 92.8 | 101.8 | 57.6 | 59.0 |

- The micro indicator that decided it: `sushi-ubench run dsv41.hc_stages`, 9 rounds, 2048 rows: mix 1.0 to 0.2 ms, the
  attention stage 1.2 to 0.4, the FFN stage 2.0 to 1.0 ([dsv41-arch](dsv41-arch.md)). On a profiled 25.8k-token
  request (`SUSHI_ENGINE_KERNEL_PROF=all`) the mix's 773 dispatches went from 588 ms to 204 ms, 2.5% of the request's
  GPU time; the prefill cells above carry that inside their run-to-run spread.
- The decode cells are other texts, not another decode: the prefill's bytes moved at rounding level, the greedy
  continuation diverged, and a code round at 8k accepted 3.92 to 4.12 tokens against 4.45. The decode forward is
  unchanged, and the same request sent three times, greedy and seeded, gives one reply each.

### Release ladder at `96220dc6` (2026-11-13, the series' record)

`tests/bench.sh --tag cooler-p2d`, the same pack and flags, commit `96220dc6`, binary SHA-256 `c3930da767fa3faf`, M5 Max
128 GB, AC, fans at max, `taskpolicy -a`, GPU lock, boot at 49 °C, no Warnings (no kernel compiled inside a request).
The box had run the full test suite until a minute before the boot: pressure read Nominal through the code arm and the
prose arm to 32k, Moderate from prose 40k, Heavy from prose 106k. Beside it the `2aa5ddaf` table above, recorded, not
rerun. Since that table: the kept scales, the Engram batch reads, the attention scores and the hyper-connection mix
(the four fast benches above), and the snapshot before the last assistant opener.

| rung | code decode | was | prose decode | was | code prefill | was | prose prefill | was |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k | 94.5 | 93.3 | 56.0 | 53.1 | 1,395 | 1,300 | 874 | 893 |
| 8k | 91.5 | 99.6 | 58.0 | 58.4 | 1,578 | 1,476 | 1,437 | 1,311 |
| 16k | 80.2 | 76.2 | 54.1 | 54.4 | 1,557 | 1,524 | 1,496 | 1,427 |
| 32k | 91.7 | 88.7 | 58.2 | 54.1 | 1,534 | 1,480 | 1,440 | 1,374 |
| 64k | 85.4 | 83.4 | 50.5 (Moderate) | 54.0 | 1,462 | 1,417 | 1,370 (Moderate) | 1,313 |
| 128k | 74.1 | 75.2 | 51.0 (Heavy) | 52.3 | 1,297 | 1,272 | 1,254 (Heavy) | 1,195 |

- **Prefill is 2 to 10% above the `2aa5ddaf` table at every rung but the first prose window**, and above v1.6.0 at
  every rung (code 2 to 12%, prose 2 to 31%). The first prose window (874) is the cold one: its Engram rows came off
  the SSD after the suite had turned the page cache over.
- **Code decode moves with the text, not with the round**: the 8k cell ran 50 to 54 rounds at 54.3 to 56.9 ms where
  the `2aa5ddaf` run ran 47 to 51 at 54.3 to 56.2 ms. The hyper-connection mix moved the prefill's bytes at rounding
  level, so the greedy continuations are other texts: 16k ran 67 rounds for 73 and 64k 63 for 65, 32k the same 57
  (48.3 to 52.3 ms a round against 49.1 to 51.9).
- **Prose decode is level or ahead at Nominal** (4k to 32k) and 2 to 6% under at 64k and 128k, the two cells read at
  Moderate and Heavy pressure against a rested box's.

## v1.2.0-dev release context ladder

[Version summary and full table](bench/v1.2.0-dev/summary.md), commit `73a9659c38f4818f399bdd2cc32309e348a3077c`, ReleaseFast on Apple M5 Max 128 GB, 2026-10-05. llmprobe 0.6.15, `--bench-only --rungs 2k,4k,8k,16k,32k,64k,128k`, GLM 3 runs and subsequent Qwen/MiMo 2 runs. `taskpolicy -a`, per-model GPU lock, fans max, server stop followed by 60 s idle. Prior release reference inherited without rerun or speedup claim; per-model reports retain scenario samples and probe notes.

## v1.2.0-dev2 GLM context ladders

[Version summary and both GLM reports](bench/v1.2.0-dev2/summary.md), runtime commit `6731e2db5cace6c2c090dac0305d54a90ec10fb9`, ReleaseFast on Apple M5 Max 128 GB, 2026-10-06. llmprobe 0.6.15, three measured runs, 2K–128K, default A4 g64 DFlash2 and kv8. Interactive QoS, one model per GPU-lock run, maximum fans, 60 s between servers. Full reports retain scenario samples, acceptance, speculative ceilings and probe notes.

The two reports above are Sushi-2.3bpw and Sushi-2.5bpw (W12), both earlier packs. Shipped Sushi-2.4bpw (K2.25/K2.5 W14,
A6 trunk), same ladder: [v1.2.0 release bench](bench/v1.2.0/summary.md), decode 47.0 / 47.9 / 49.5 / 49.0 / 43.4 / 44.9 /
42.1 and prefill 863 / 831 / 818 / 810 / 790 / 712 / 613 tok/s at 2K–128K (`e26fd471`).

## v1.2.0 release context ladder

[Version summary and per-model reports](bench/v1.2.0/summary.md), 2026-10-07, Apple M5 Max 128 GB. llmprobe 0.6.15, `--bench-only` 2K–128K, 2 runs. `taskpolicy -a`, per-model GPU lock, fans max, 3 minutes idle before each server. Qwen 2.6bpw/4bpw and MiMo 2.3bpw on `177c526f`, GLM 2.4bpw on `e26fd471`.

## v1.2.1 release context ladder

[Version summary and per-model reports](bench/v1.2.1/summary.md), 2026-10-08, Apple M5 Max 128 GB. llmprobe 0.6.15, `--bench-only` 2K–128K, 2 runs. `taskpolicy -a`, per-model GPU lock, fans max, 3 minutes idle before each server. All four on `975a7694`.

## v1.2.2 release context ladder

[Version summary and per-model reports](bench/v1.2.2/summary.md), 2026-10-11, Apple M5 Max 128 GB. llmprobe 0.6.15,
`--bench-only` 2K–128K, 2 runs, all on AC power. `taskpolicy -a`, per-model GPU lock, fans max, at least 60 s idle
before each server. One binary built from `bb9adb6a`.
- Qwen 2.6bpw and 4bpw come from the first pass.
- MiMo 2.3bpw comes from a fresh AC rerun: its first-pass run crossed onto battery, as did GLM's, and battery slowed GLM 8–18%.
- GLM 2.4bpw comes from the release binary `29107ce9` (SHA `98d1760b`), after the DFlash2 in-place commit fix.

## v1.6.0 release context ladder

[Version summary and per-model reports](bench/v1.6.0/summary.md), 2026-11-09/10, Apple M5 Max 128 GB. The v1.4.0 `sushi bench`
protocol and box settings; one binary from `ac3da7aa` (SHA `fcdba905`). Code arm 512 tokens (DeepSeek 256), prose 128.
- Qwen STQ 2.75bpw ran pack revision 2.0; against the 1.5.1 binary on that pack (same session, A B A B) prefill is 2-6%
  lower from 4k to 40k and code decode about 2% lower: a regression to fix after 1.6.0.
- Qwen 4bpw level on prefill, code decode +0.5% to +5%; GLM (Heavy pressure, median 1,498 MHz) level or ahead.
- DeepSeek prefill 1-3% below 1.5.1 on code to 64k, 4-9% on prose to 32k, ahead at 128k.

## v1.5.1 release context ladder

[Version summary and per-model reports](bench/v1.5.1/summary.md), 2026-11-06, Apple M5 Max 128 GB. The v1.4.0 `sushi bench`
protocol and box settings; one binary from `399895f6` (SHA `4391a90f`). Code arm 512 tokens (DeepSeek 256), prose 128.
- Qwen STQ 2.75bpw: prefill +4% to +83% over v1.5.0 (4k 1,878 against 1,026), code decode +1% to +9%.
- GLM throttled under 350 MHz on both runs: released on the first run's code rungs 4k-64k, no prose; 64k decode per MHz
  matches v1.5.0 (0.0488 against 0.0486).
- DeepSeek ran pack revision 2.0 (v1.5.0 ran 1.0), so its deltas mix binary and pack.

## v1.5.0 release context ladder

[Version summary and per-model reports](bench/v1.5.0/summary.md), 2026-11-04, Apple M5 Max 128 GB. The v1.4.0 `sushi bench`
protocol and box settings; one binary from `75996eda` (SHA `62184462`). Code arm 512 tokens (DeepSeek 256), prose 128.
- DeepSeek, `--ssd-budget-gb 108`: prose decode +3% to +10% over v1.4.1 at every rung, prefill level or ahead.
- Qwen STQ 2.75bpw prefill reads 5-15% below v1.4.1's report; a same-session A/B (1.4.1 vs 1.5.0 binary, one pack) put
  1.5.0 level or ahead, so the gap is session drift.

## v1.4.1 release context ladder

[Version summary and per-model reports](bench/v1.4.1/summary.md), 2026-10-30, Apple M5 Max 128 GB. The v1.4.0 `sushi bench`
protocol and box settings; one binary from `b358cd38` (SHA `6944232d`). The prose arm decodes 128 tokens on every model.
- DeepSeek, `--ssd-budget-gb 108`: prefill up to +47% over v1.4.0 (code 128K 1,217 vs 829), decode up to +13%.
- GLM's code arm ran at heavy thermal pressure from 64K (GPU near 1,270 MHz vs 1,490 in v1.4.0), which accounts for its
  64K and 128K decode; its 128K code samples are loop-stopped at 390 tokens.

## v1.4.0 release context ladder

[Version summary and per-model reports](bench/v1.4.0/summary.md), 2026-10-27, Apple M5 Max 128 GB. `sushi bench`
(`./tests/bench.sh`, code and prose arms, one cold prefill in 8k restore windows to 128K, 3 decode samples a rung), AC
power, `iogpu.wired_limit_mb=125000`, `taskpolicy -a`, per-model GPU lock, fans max and idle under 50 °C before each server.
Not comparable with the llmprobe ladders below.
- Qwen STQ 2.75bpw and 4bpw: binary SHA `fcb8ee2f` (source of `846863bd`). GLM: `d361c70b` (SHA `4d25c2d7`); its 128K code
  samples are loop-stopped at 390 tokens. DeepSeek: `bf21aa5e` (SHA `5027a704`), `--ssd-budget-gb 108`; 80 and 96 in usage.md.

## v1.3.0 release context ladder

[Version summary and per-model reports](bench/v1.3.0/summary.md), 2026-10-21, Apple M5 Max 128 GB. llmprobe 0.6.15,
`--bench-only`, AC power, `iogpu.wired_limit_mb=120000`, `taskpolicy -a`, per-model GPU lock, 3 minutes idle before each server.
- MiMo 2K–128K, 2 runs, release candidate `dc93b43f` (SHA `607c6eb0`). It started hot after the full suite and prefilled
  18–24% under 1.2.4; a 2K rerun after fans-max idle read 1,239–1,248 tok/s per request (1.2.4: 1,285–1,353).
- Qwen 2.6bpw and 4bpw are the MTP arms of the DFlash2 ladders (4K/16K/64K/128K, 3 runs, `--prefix-cache-entries 0`),
  branch builds `f9393ba6` and `e2912d04`. GLM was not rerun.

## v1.2.4 release context ladder

[Version summary and per-model reports](bench/v1.2.4/summary.md), 2026-10-15, Apple M5 Max 128 GB. llmprobe 0.6.15,
`--bench-only` 2K–64K, 2 runs, all on AC power, `iogpu.wired_limit_mb=120000`. `taskpolicy -a`, per-model GPU lock,
fans max, 3 minutes idle before each server. One binary built from `a65539e1` (SHA `85c6de9c`).
- MiMo is the empty-prefix-cache run: on the shared cache, over its budget, MiMo prefill read 8–17% low. An
  empty-cache A/B in one session put the released v1.2.3 binary and `a65539e1` within 1% (2K 1,214 vs 1,223, 8K 1,224
  vs 1,218 tok/s prefill).

## v1.2.3 release context ladder

[Version summary and per-model reports](bench/v1.2.3/summary.md), 2026-10-13, Apple M5 Max 128 GB. llmprobe 0.6.15,
`--bench-only` 2K–128K, 2 runs, all on AC power, `iogpu.wired_limit_mb=120000`. `taskpolicy -a`, per-model GPU lock,
fans max, 3 minutes idle before each server. One binary built from `4a1f1f1b` (SHA `b5cd630a`).
- Qwen 2.6bpw, MiMo and GLM ran 5–20% under their v1.2.2 rows that day; the released v1.2.2 binary, rerun the same day
  on 2K–32K, measured the same, so compare v1.2.3 against those same-day reruns, not against the v1.2.2 report.
