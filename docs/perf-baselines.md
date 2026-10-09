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
