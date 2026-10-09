# Performance baselines: Qwen3.8-Flash-Next (`qwen4_exp`)

The recorded speed numbers for Qwen3.8-Flash-Next (`qwen4_exp`) on this box. Method, roofline, shared EXL3 kernel timings and
release ladders: [perf-baselines](perf-baselines.md); the rules there (inherit the recorded baseline, name
commit, binary stamp, QoS and lock beside every number) apply to every section here. Architecture: [qwen4-arch](qwen4-arch.md).

<a id="exl3"></a>
## EXL3 decode and prefill attribution (Flash-Next)

- EXL3 K4 resident, kv8 (2026-09-22): decode forward 18.35 ms (~56 tok/s live), prefill ~1600 tok/s at 9.6k.
  Decode is ~860 kernels per token with only ~9.8 ms of kernel time; the rest is dispatch gaps (~7 us per boundary).
  Delivered: pair GEMV ~313 GB/s, down fused ~275 GB/s, trunk 8-bit qmv ~600 GB/s.
- Prefill per 2048-token chunk: the three trellis GEMMs ~520 ms (~0.4 TFLOPS, latency-bound serial k-loop, whole
  expert re-decoded per 32-row window); rest of layer ~650-980 ms.
- Landed since: pair prepare fused into pair GEMV + simd tile reduce (decode 18.35 → 17.45 ms); half4 activation
  loads + one window table per layer (prefill 1600 → 1850 tok/s at 9.6k); the n=48 lane funnel for every non-MUL1
  codebook (197395d).
- Ruled out (do not retry without a new reason): 64-row GEMM windows (+10%, registers), k-loop unroll/prefetch, ALU
  trims in the MUL1 decode, more split-K, `MLX_MAX_OPS_PER_BUFFER`, `MLX_METAL_FAST_SYNCH`. Decode loads (630 GB/s
  alone) and decode ALU do not overlap.

## M1 Max: Sushi-2bpw streamed decode GPU attribution

2026-09-30, base `677722d`, ReleaseFast, EXL3 MCG n32/window15, 512 experts/top-10, SSD budget 20 GB,
wired margin 5 GiB, ctx 8192, kv8. M1/G13 uses the SIMD decode kernels, without NAX.

A temporary fixed-expert replay removed host routing reads and SSD fills. Block stand-ins estimated MoE at
~13 ms, GDN at ~9 ms, HC at ~5 ms and full attention at ~3 ms per token. These are graph ablations, not additive
kernel timestamps. The per-block forward profiler synchronizes between blocks and distorts the decode chain.

The shared-add finish reduction preserved bf16, f16 and f32 output bits. A same-process replay meter reset KV/SSM
before every arm and alternated A/B four times, 80 forwards per arm:

| GPU evaluation, ms/forward | A: separate add | B: folded shared add |
|---|---:|---:|
| pair 1 | 31.668 | 31.377 |
| pair 2 | 31.696 | 31.405 |
| pair 3 | 31.711 | 31.371 |
| pair 4 | 31.658 | 31.444 |
| mean | 31.683 | 31.399 |

The cut is 0.284 ms (0.90% of GPU evaluation); graph construction was 2.753 vs 2.776 ms. The meter used the real
model with replayed expert IDs; it is attribution, not a generation-quality test. A compiler started during the
last arm, but all four pairs improved. Subsequent measurements require builds to take the box lock too.

Normal greedy boot arms (warm-up plus three 300-token requests per arm, same hash-map/C prompt):

| arm | tok/s, three requests |
|---|---|
| A1 | 17.450, 17.417, 17.501 |
| B1 | 18.019, 18.034, 18.015 |
| A2 | 17.990, 18.022, 17.393 |
| B2 | 15.802, 15.805, 15.683 |

All 12 responses were byte-identical. These boots show substantial box drift; they do not establish a live tok/s
win. The controlled GPU ablation establishes the small kernel gain. One-row GDN verify-fold reuse and narrower
EXL3 lane funnels were also tried and discarded without a measured win.

## Flash-Next K3, serial (no MTP)

llmprobe `--bench-only --full`, ctx 65536, KV unquantized, MTP verified off (1.01 tok/step), one server at a time.

| pack | binary | decode 192 tok | prefill 2k | 4k | 8k | 16k | 32k | 64k | first token at 64k |
|---|---|---|---|---|---|---|---|---|---|
| affine 4/8 control | a05d15f | 65.7 | 1741 | 57.5 | | 59.4 | | 56.6 | 34.9 s |
| affine 4/8 control | 28d7fab | 63.6 | 1618 | 57.3 | | 59.9 | | 56.4 | 33.9 s |
| MCG K3 (clean paired run) | 28d7fab | 60.0 → 62.4 | 1779 | 57.1 | 56.5 | 54.8 | 56.0 | 55.7 | 34.5 s |
| affine 4/8, MTP on (depth 6) | a05d15f | 93.4 (3.4 tok/step) | 1707 | 88.6 | | 82.3 | | 79.0 | 32.9 s |

<a id="mtp"></a>
## Flash-Next MCG K3 with MTP

Binary eb458ad, one session, live cost table, llmprobe short bench (192-token decode, 2k prefill):

| cell | MTP off | MTP |
|---|---|---|
| decode tok/s | 47.2 | 70.2 |
| predictable text | 47.3 | 90.6 |
| novel text | 48.7 | 60.3 |
| tokens per step | 1.02 | 3.69 |
| prefill 2k | 1733 | 1577 |

Verify is 89-92% of a round's wall; forced-depth round ms on code: depth 1 31.2, 2 36.0, 3 42.4. The absolute
off-MTP figures sit below the serial table's: a different session, so only the within-session pairs compare. MCG
verify ms 28.2 / ~33.5 / ~37.5 at 2/3/4 rows: ~4.6 ms per extra row.

## Flash-Next Sushi-3bpw, 1M context ladder (725b76ca)

Pack `Qwen3.8-Flash-Next-Sushi-3bpw`, `--ctx-size 1048576 --kv-quant 8 --mtp`, llmprobe `--bench-only --rungs
4k,8k,16k,32k,64k,128k,256k,512k,980k`, `taskpolicy -a`, lock `bench-qwen-1m-clean`, quiet box (load average < 1), fans at
max with 4 min idle before boot (the thermal protocol in [process-measurement](process-measurement.md)). Headline cells:
decode 93.8 tok/s, prefill 1906 tok/s at 2k, MTP 3.62 tokens per step (predictable 119.9, novel 82.9), and llmprobe
saw an 11.9% sustained-load slide over the 38 min run.

| context | decode tok/s | prefill tok/s | first token | tokens per step |
|---|---|---|---|---|
| 4k | 96.1 | 1702 | 2.5 s | 3.37 |
| 8k | 92.5 | 1940 | 4.2 s | 3.31 |
| 16k | 93.2 | 1949 | 8.4 s | 3.20 |
| 33k | 96.2 | 1961 | 16.8 s | 3.00 |
| 66k | 80.7 | 1945 | 33.7 s | 2.56 |
| 131k | 80.9 | 1900 | 69.0 s | 2.63 |
| 262k | 73.4 | 1826 | 143.7 s | 3.20 |
| 524k | 55.1 | 1686 | 311.1 s | 3.37 |
| 1004k | 56.6 | 1465 | 685.2 s | 3.31 |

The same ladder on a338ca2, run hot after 2 h of other GPU work with workers computing alongside, read 20-33% lower
prefill at every rung (1358 at 2k, 1127 at 1004k) and 71.8 decode at 4k. Prefill code did not change between the two
binaries, so that gap is the box. Decode also gained from d72178a (the MTP regime gate). Never compare a ladder cell
across a thermal state.

## Release 1.0.4, Sushi-3bpw

Owner-scoped Sushi-3bpw run (the default release gate uses 4bpw): `ad4a3ce0` plus the context-bill change,
ReleaseFast binary SHA-256 `2aeee2e678521727e66994d75260c25cd4cffd0d05ecb797c73210a2b0ea9704`,
mtime 2026-09-26 15:26:43 +0700. M5 Max 128 GB, `--ctx-size 1048576 --kv-quant 8 --mtp`,
`tests/bench.sh --url` with llmprobe 0.6.12 `--bench-only` (default ladder, median-of-3 headline cells).
QoS `taskpolicy -a`, GPU lock `release-104-quiet-sushi3bpw`, conversion stopped, no other test running;
fans at max and 10 s idle from 51.9 °C, fans restored to auto afterward.

| decode tok/s | prefill tok/s (2042 tokens) | first token | tokens per step | sustained decode drift |
|---:|---:|---:|---:|---:|
| 98.4 (97.5–100.0) MTP | 1671 (1665.6–1707.4) | 280 ms | 2.87 | -4.5% |

MTP engaged in 37 logged requests; the measured n-gram pool arm engaged and the table warmed fully. Relative to
the inherited `725b76ca` headline cells above, decode is +4.9% and prefill -12.3%; no old binary was rerun.
This is not a paired speedup/regression claim: sessions differ, the old prompt was 2041 tokens, and the old loader
billed a duplicate vision shard (50.17 versus 49.33 GiB). There is no prior measured 3bpw release-column cell.

<a id="hc-row-group"></a>
## Flash-Next Sushi-3bpw: row-grouped HC read on the M5 Max (0fbb74ca vs #6)

A = 0fbb74ca, B = 0fbb74ca + #6 (PR head 3f97e4f), ReleaseFast, M5 Max 128 GB, a fresh boot for every arm run, A B B A
then B A A B, `taskpolicy -a`, GPU lock `pr6-ab` per boot, fans at max and 10 s idle from 55 °C, restored to auto after;
NOT a quiet box (a system daemon at ~100% of one core). Greedy, 4 prompts x 256 tokens (code / list / prose / story),
`--prefix-cache-entries 0`, `SUSHI_ROUND_COST_PERSIST=0`, kv8 and MTP at their defaults. Decode tok/s, mean of 4 boots:

| cell | code | list | prose | story | sum |
|---|---|---|---|---|---|
| forced depth 3 | 103.05 -> 105.49 | 85.14 -> 87.94 | 80.73 -> 83.89 | 67.49 -> 69.82 | +3.2% |
| default adaptive MTP | 98.45 -> 102.25 | 78.65 -> 83.48 | 79.96 -> 80.62 | 77.33 -> 78.90 | +3.2% |
| same boots, `enable_mtp: false` | 65.62 -> 64.76 | 65.22 -> 64.32 | 65.00 -> 64.54 | 64.82 -> 64.74 | -0.9% |

- Forced depth 3 is faster in 16/16 adjacent A/B pairs, adaptive in 11/16. Outputs are identical between the arms in
  84/84 comparisons, MTP equals serial in 32/32, and `test_mtp_equivalence.sh` passes 11/11 on B.
- Decode meter (`SUSHI_DECODE_FWD_UBENCH=100`, 4096 keys), ms per forward at 1 / 2 / 4 / 7 rows, 4 boots per arm:
  18.63 / 22.64 / 30.32 / 46.30 -> 18.68 / 22.37 / 28.68 / 43.90 (-5.4% and -5.2% at 4 and 7 rows). One row alone,
  300 forwards, 4 more boots per arm: 18.81 -> 18.72. Over those 8 boots per arm one row reads 18.72 -> 18.70 ms:
  unchanged, so the MTP-off cell above is boot noise.
- `[spec-warmup]` over 16 boots per arm: 1.20-1.34 s on B and 1.23-1.73 s on A, apart from one ~2.2 s boot in each
  (B's first, 2290 ms; A 2171 ms). The per-width D/U kernel variants add no measurable load time.
- Follow-ups on the M2 Max 64 GB (2f1e4bf vs + change, decode meter at 4096 keys, `taskpolicy -a`, lock per boot, one
  binary per arm, A B C C B A; ms per forward at 1 / 2 / 4 / 7 rows). Configs keyed on (rows, inject, pending write):
  33.67 / 48.35 / 76.69 / 126.84 -> 34.08 / 48.79 / 78.49 / 128.27, and in a second A B B A 33.78 / 48.43 / 76.36 /
  126.15 -> 33.73 / 49.10 / 77.76 / 128.17 with live decode +0.8% (MTP) / -1.5% (MTP off): no change beyond boot noise;
  outputs identical. The row count as a scalar D/U input on top: 33.81 / 50.69 / 79.12 / 130.29, 1-4% slower at 2-7
  rows (one more buffer per dispatch, a runtime clamp); dropped.

<a id="mtp-lookup"></a>
## Flash-Next: prompt lookup inside the MTP round (2f1e4bf2 + the port)

Arms of one binary per step: A = `SUSHI_MTP_LOOKUP=0` (now `--no-mtp-lookup`), B = lookup (c68b4cf7, ReleaseFast, sha256 848bd456…),
C = lookup with the line rule (6ea00e3f, sha256 ea4f6fd0…). M5 Max 128 GB, 2026-09-27, `tests/bench_mtp_lookup.sh`
(a file of ~600 tokens in the prompt; thinking off; greedy, and sampled 0.6 / 0.95 / 20 seed 7; 2 reps per boot),
`--ctx-size 131072 --kv-quant 8 --prefix-cache-entries 0`, `SUSHI_ROUND_COST_PERSIST=0`, MTP and exact acceptance at
their defaults, `taskpolicy -a`, GPU lock `lookup-ab` per boot, fans at max with 3 min idle from 98 °C and restored to
auto after. CONTENDED box: other workers' builds and tests ran throughout, so compare the interleaved arms only.
Sushi-3bpw, decode tok/s, median of 8 A runs (A B B A and A C C A) against 4 B and 4 C runs:

| task | greedy A / B / C | sampled A / B / C | C lookup rounds / drafted / landed |
|---|---|---|---|
| copy the file | 125.6 / 152.7 / 152.1 (+21%) | 125.2 / 140.7 / 149.1 (+19%) | 52 / 409 / 389 |
| rename in the file | 124.3 / 142.9 / 150.1 (+21%) | 123.8 / 141.2 / 145.1 (+17%) | 58 / 449 / 430 |
| fix a bug in the file | 129.4 / 138.0 / 149.5 (+16%) | 125.7 / 141.8 / 146.8 (+17%) | 39 / 301 / 279 |
| unified diff | 122.1 / 116.2 / 120.1 (-1.6%) | 122.4 / 108.4 / 119.3 (-2.5%) | 2 / 14 / 7 |
| write_file tool call | 124.0 / 137.4 / 135.4 (+9%) | 122.2 / 134.9 / 135.3 (+11%) | 44 / 337 / 303 |
| new code | 108.4 / 106.9 / 107.5 (-0.9%) | 103.9 / 103.2 / 102.4 (-1.4%) | 1 / 7 / 6 |
| prose about the file | 86.3 / 87.1 / 84.8 (-1.7%) | 82.2 / 82.2 / 82.3 (0%) | 0 / 0 / 0 |

- MTP alone already takes ~5.5 tokens per round on a copy; a lookup round lands ~7.5 of 8 drafts.
- B lost 4.8% (greedy) and 11.5% (sampled) on the diff: its lines echo the file's behind a `-`/`+`/space prefix, so a
  match agrees to the end of one line and its draft fails at the next (2-7 rounds per request landing 0-60%). The line
  rule (C) keeps 1-3 such rounds; its -1.6% / -2.5% is inside this box's noise: C's prose, with no lookup round at all,
  read -8.0% against the adjacent A boots.
- Long context (B, one pair A B then B A, median of 2, greedy): the file renamed behind ~32k tokens of repo text
  110.0 -> 136.1 (+24%), behind ~64k 108.7 -> 127.9 (+18%).
- Sushi-4bpw (B, one A B pair, median of 2): copy 112.8 -> 125.8, rename 110.7 -> 116.8, fix 102.7 -> 120.4, write_file
  104.6 -> 114.9 greedy; diff 97.0 -> 94.6 (-2.5%) and 96.9 -> 91.4 (-5.7%) before the line rule.
- llmprobe 0.6.12 `--bench-only` (`tests/bench.sh --only sushi-4bpw`, B): decode 77.7 -> 77.9, predictable 78.7 ->
  113.7 tok/s at 2.04 -> 5.33 tokens per step, novel 65.5 -> 65.6, prefill 1524 -> 1479 at 2039 tokens (lookup never
  runs in prefill; one boot per arm), 4 streams 83.2 -> 81.2 aggregate.
- Greedy output was identical to `--no-mtp` on every task, and `test_mtp_equivalence.sh` passed 18/18.

<a id="gdn-decode-recur"></a>
## Flash-Next: GDN prework and recurrence in one dispatch (caa02a4f vs the port)

A = caa02a4f (sha256 0bec2879…), B = caa02a4f + the port with a kill-switch env the landed commit drops, same kernels
(e73439e9, sha256 c45acaf6…), ReleaseFast. M5 Max 128 GB, 2026-09-27, Sushi-3bpw, `--ctx-size 65536 --kv-quant 8`,
`SUSHI_ROUND_COST_PERSIST=0`, MTP prompt lookup on in both arms, `taskpolicy -a`, GPU lock `gdn-ab` per boot, fans at
max with 10 s idle from 64 °C and restored to auto after. CONTENDED box: a system daemon at ~100% of one core
throughout, other workers' builds and tests beside the first meter and live boots.

Decode meter, the chain and the step alternated off / on / on / off per width in one process
(`SUSHI_DECODE_FWD_UBENCH=40 _S=1,3,5,7,9 _GDN_ARMS=1`, `--prefix-cache-entries 0`), three boots per length, ms per
forward, mean over boots:

| rows | 4096 keys | 32768 keys |
|---|---|---|
| 1 | 18.50 -> 18.05 (-2.4%) | 18.70 -> 18.33 (-1.9%) |
| 3 | 24.77 -> 24.35 (-1.7%) | 25.10 -> 24.68 (-1.7%) |
| 5 | 32.99 -> 32.94 (-0.2%) | 34.02 -> 33.37 (-1.9%) |
| 7 | 42.56 -> 42.23 (-0.8%) | 43.50 -> 43.08 (-1.0%) |
| 9 (control: the step declines) | 51.02 -> 51.21 (+0.4%) | 52.25 -> 52.20 (-0.1%) |

- 0.33-0.45 ms per forward at 1, 3 and 7 rows, faster in all 18 boot-width pairs: the 36 prework dispatches and their
  gaps (4668 -> 4055 graph ops per one-row forward, 6903 -> 6255 at 7 rows; 7287 both at 9).
- 5 rows at 4096 keys climbs 1.5-2.7 ms across its four passes in every boot whatever the arm (the meter's context
  grows ~430 keys per pass there), a step the pass order cannot cancel; the same width reads -1.9% in 3/3 boots at
  32768 keys.
- llmprobe 0.6.12 `--bench-only`, A B B A, `--no-mtp`: decode 64.6 / 67.0 / 67.3 / 66.1 tok/s (+2.8%, both B boots above
  both A boots); prefill, first token and the 0.5-16k rungs within noise.
- Default MTP, A B B A: decode 93.5 / 90.4 / 91.0 / 90.4 at 4.92 / 4.92 / 6.40 / 5.82 tokens per step; each boot's
  planner learns its own costs, so this cell is variance. Forced depth 3, one boot per arm, 6 prompts x 256 tokens:
  588.4 -> 588.3 tok/s summed (serial 393.0 -> 396.3 in the same boots); a ~0.4 ms saving is ~1% of a round.
- Identity: those 6 prompts serial and at forced depth 3, A == B 12/12 and MTP == serial 6/6 per arm;
  `test_mtp_equivalence.sh` 18/18 on B, lookup rounds engaged on its copy task.

<a id="qsa-pool-rope"></a>
## Flash-Next: QSA pooled-key upkeep in one kernel (ad5e6be8 vs the port)

A = ad5e6be8 (sha256 bb649a00…), B = ad5e6be8 + the port (8785d33b before the squash, sha256 15ffa4be…), ReleaseFast.
M5 Max 128 GB, 2026-09-27, Sushi-3bpw, kv8, `--mtp`, `SUSHI_ROUND_COST_PERSIST=0`, MTP prompt lookup on in both arms,
`taskpolicy -a`, GPU lock `qsa-pool-ab` per boot, fans at max per boot with 10 s idle (3 min before greedy B, which
followed a KLD run at 90.6 °C) and restored to auto after; load 1.9-3.5 with other workers' builds beside.

Decode meter on B, the composed chain and the kernel alternated A B B A per width in one process
(`SUSHI_DECODE_FWD_UBENCH=48 _S=1,3,5,1,3,5 _QSA_POOL_ARMS=1`, ctx 131072), one boot per length, two rounds per
width; the context grows ~7.3k keys over a boot. ms per forward:

| keys prefilled | rows | chain | kernel | delta | per round | graph ops |
|---|---|---|---|---|---|---|
| 16384 | 1 | 18.21 | 18.03 | -1.0% | -0.2, -1.8 | 4055 -> 4013 |
| 16384 | 3 | 26.65 | 26.72 | +0.3% | +2.1, -1.5 | 4490 -> 4364 |
| 16384 | 5 | 37.04 | 36.88 | -0.4% | +0.2, -1.0 | 4938 -> 4774 |
| 65536 | 1 | 18.88 | 18.55 | -1.7% | -1.4, -2.1 | 4055 -> 4013 |
| 65536 | 3 | 27.27 | 27.17 | -0.4% | +0.4, -1.1 | 4490 -> 4364 |
| 65536 | 5 | 37.90 | 37.25 | -1.7% | -1.0, -2.4 | 4939 -> 4774 |

- 9 of 12 rounds favour the kernel. The 65536-key 1- and 5-row cells agree in both rounds (-0.33 and -0.65 ms, all
  GPU eval); the 16384-key cells sit inside the chain arm's own block-to-block spread (0.6-1.9 ms). CPU graph build
  moves under 0.04 ms. The kernel launched 300-1224 times per fused block, 0 per chain block.
- Identity: 6 prompts of 3.1-21k tokens x 256 tokens, serial and at forced depth 3, A == B 12/12, MTP == serial 6/6
  per arm, identical per-request acceptance (prompt lookup stays off under a forced depth).
- Not attributed: forced depth 3 summed 525.6 -> 593.4 tok/s (serial 353.6 -> 355.4) and llmprobe 0.6.12
  `--bench-only` (ctx 1048576) decode 92.3 -> 95.2, prefill 1475 -> 1674, one boot each. Both A boots started hotter
  (72.8 / 73.6 °C die vs 90.6-then-3-min / 53.2 °C); the kernel is ~1% of a round and cannot reach prefill by 13%
  (the recorded v1.0.4 cell reads 1671).

## Sushi-2bpw streamed decode on an M1 Max 32 GB (`--ssd-budget-gb 20`, kv8)

Greedy 300-token decode of one prompt after a warm-up (tok/s; same box, arms run back to back):

| change | exact | `--expert-pick-tolerance 0.2` |
|---|---|---|
| v1.1.1 (per-layer id sync) | 12 | - |
| async expert hand-off + early shared expert | 14.1 | 15.3 |
| experts queued from the GPU cache map before the id read | 17.9 | 15.3 |
| deferred verification across one GDN successor | 19.1 | - |
| lossy pick on the GPU, deferral in tolerance mode | 19.3 | 22.3 |

llmprobe `--bench-only --rungs 512,2k` (ctx 32768): v1.1.1 10.3 / 9.4 tok/s decode, this branch exact 19.6 / 17.1,
tolerance 0.2 at 2k 18.9. A 3869-token prompt (QSA engaged), 200 greedy tokens: v1.1.1 8.7 / 9.1 (cold / warm), this
branch 15.2 / 16.9. Exact output is byte-identical to v1.1.1 on every single-stream check, and `kld compare` against the
exact teacher reads KLD 0 (NLL 0.4004 unchanged). Two concurrent streams can differ from a single stream by arrival
order alone (batched and solo decode are not bit-identical); v1.1.1 does the same.

Anatomy of an exact token at the end (~52 ms): GPU work ~31 ms (MoE ~13, GDN ~9, HC ~5, attention ~3); SSD fills
~0.63 ms per missed expert, ~18 misses a token (fill tuning: splitting reads or more than 4 workers is slower);
the rest is host round trips on full-attention/PLE layers and rollbacks. Tolerance 0.2 cuts misses to ~7 a token.

<a id="m2max-64gb"></a>
## Flash-Next Sushi-3bpw on an M2 Max 64 GB

A second box: M2 Max, 38-core GPU (no NAX), 64 GB, `iogpu.wired_limit_mb=58000`, `--mtp --skip-mem-preflight` (the
load check refuses with apps open), `taskpolicy -a`, the lock held per run, NOT a quiet box (desktop apps open, 6-13% memory free, swap in use).

- Before the simdgroup-matrix body (f42cd8e): prefill 28-38 tok/s at 3.1-3.9k; omp's first turn, 13,194 tokens,
  prefilled at 19.1 tok/s (11.5 min) and decoded 751 tokens at 19.7 tok/s.
- Metal System Trace over a 2.5k prefill on f42cd8e (`--instrument 'Metal GPU Counters'`): the scalar sorted EXL3
  GEMM took 90.7% of sampled GPU time (gate/up 59.5%, down 31.2%), MLX's bf16 GEMM 5.1%, everything else 4%.
- The simdgroup-matrix body against the scalar body, one process (f22a383 + the body), A B B A, paired per-block ratio,
  window table built outside the timed calls (`SUSHI_EXL3_LAYER_UBENCH=1`, MCG w15, 512 experts, top-10):

| prompt chunk | n48 2560->640 | n48 640->2560 | n64 2560->640 | n64 640->2560 |
|---|---|---|---|---|
| 17 tokens | 3.2x | 5.4x | 3.1x | 4.3x |
| 64 tokens | 3.5x | 5.0x | 3.5x | 4.8x |
| 205 tokens | 4.6x | 6.2x | 4.7x | 5.9x |
| 2048 tokens | 22.2x (241.0 -> 10.9 ms) | 19.7x (323.9 -> 16.9 ms) | 19.2x (343.8 -> 18.1 ms) | 17.8x (354.0 -> 20.3 ms) |

- End to end, f22a383 against f22a383 + the body, B A A B with a distinct 3.4-4.0k prompt per run: 123.8 / 30.9 /
  31.7 / 140.8 tok/s, 4.0x and 4.4x per pair. Disk reads rose from ~1,200 to ~5,000/s: the n-gram walk's share grew.
- llmprobe 0.6.12 `--bench-only` (`tests/bench.sh --url`), both arms `--mtp --skip-mem-preflight` with
  `QWEN4_PLE_PREFETCH_PREFILL=1`, B A A B, one boot per cell, AC power. Prefill at 2k 344.8 / 37.6 / 38.6 / 340.5 tok/s
  (9.2x, 8.8x per pair); first token at 16.3k 45.8 / 418.5 / 418.8 / 47.7 s; decode 32.7 / 34.1 / 33.8 / 32.5 tok/s
  (the change cannot reach decode: rows <= 16 take the decode chain; MTP 2.9-3.9 tokens per step across cells).
- Quality, teacher-forced against f42cd8e's own capture (16 doc paragraphs x 256 tokens, kv8): the body scores KLD
  0.0406 / top-1 91.0% / NLL 0.7830; a rounding-only control (f42cd8e with `SUSHI_FUSED_256=0`) 0.0347 / 90.8% /
  0.7831; f42cd8e itself 0.0000. On this pack any bit-different kernel reads about 0.04 against another.
- `QWEN4_PLE_PREFETCH_PREFILL=1` (pool) vs the default serial walk on f22a383 + the body, A B B A, a distinct 3.5-4.3k
  prompt per run: 165.2 / 378.7 / 353.1 / 224.9 tok/s, the pool 2.29x and 1.57x per pair. (On f42cd8e, with the
  scalar GEMM dominating, the same screen read 14-24%.)
- The n-gram residency gate pools by default on that box: its 29.8 GB table cannot stay beside 47.6 GB of weights in
  64 GB. Default flags, one boot per cell, a distinct 3.4-4.4k prompt per run, `max_tokens` 1. 8c16b2b against 8c16b2b +
  the gate, A B B A: 189.8 / 393.6 / 395.7 / 219.6 tok/s, 2.07x and 1.80x per pair. Before the body, f22a383 against
  f22a383 + the gate, A B B A twice: 33.9 / 36.6 / 31.4 / 34.8 and 30.7 / 37.0 / 37.4 / 35.1 tok/s, paired 1.08,
  0.90, 1.21, 1.07 (mean 1.06, inside the prompt-to-prompt spread).

### v1.1.0 release gate

`tests/bench.sh --tag v1.1.0 --only sushi-4bpw` (Sushi-4bpw, llmprobe 0.6.12 `--bench-only`, MTP) on the release tree
(67b794f3 plus the version bump), M5 Max, `taskpolicy -a`, lock `release-110`, fans max, no build or test running:
decode 89.4 tok/s (87.5-90.2), prefill 2139 tok/s, 5.33 tokens per step, 37/37 requests `mode=mtp`. Against v1.0.5:
decode +8%, prefill +19%.

### v1.0.5 release gate

`tests/bench.sh --tag v1.0.5` (Sushi-4bpw, llmprobe 0.6.12 `--bench-only`, MTP) on the release tree (1ea9492a plus the
version bump), M5 Max, `taskpolicy -a`, lock `release-105`, fans max, no build or test running (load1 1.74):
decode 82.6 tok/s (82.1-89.9), prefill 1796 tok/s, 5.33 tokens per step, 37/37 requests `mode=mtp`. The v1.0.4 column
has no Sushi-4bpw cell.

### v1.0.4 (27ca1c86), user-reported

A user ran the release on their own M2 Max 64 GB: llmprobe 0.6.12 against port 1234, default ladder,
warmup + median of 3, greedy, thinking on (reasoning medium), MTP engaged. Launch flags, QoS, lock and box state were not
reported, so this is a reference point, not a controlled cell.

| prompt | decode tok/s | first token | prefill tok/s | tokens per step |
|---|---:|---:|---:|---:|
| 2041 (headline) | 38.6 (38.5–38.8) | 866 ms | 399 (398.9–401.2) | 2.91 |
| ~512 | 31.4 | 2.0 s | 258 | 2.67 |
| ~4.2k | 34.2 | 10.4 s | 410 | 3.15 |
| ~8.2k | 36.9 | 20.0 s | 413 | 2.95 |
| ~16.3k | 37.8 | 39.8 s | 409 | 2.74 |

- Speculation 1.38x (predictable 46.4, novel 33.5 tok/s); prefix cache 6.8x (4.1 s cold, 607 ms warm, 1509 of 1540
  tokens cached); 4 streams 39.9 tok/s aggregate vs 26.6 alone (0.38 efficiency); sustained 38.6 -> 36 tok/s over 4 m 5 s
  (-6.7%).

<a id="m2max-decode"></a>
### M2 Max decode attribution (8c16b2b)

Decode meter (`SUSHI_DECODE_FWD_UBENCH`, 4096 keys of context, no sampling around the forward), `--mtp
--skip-mem-preflight`, `taskpolicy -a`, lock per boot. A verify row costs ~16 ms, ~47% of a 1-row forward (the M5 Max
reads ~26%), so MTP nets ~1.1-1.35x here. ms per forward at 1 / 2 / 4 / 7 rows, f22a383: 36.1 / 51.3 / 80.6 / 131.4.
"HC grouping" below is the row-grouped HC read that landed as #6 (PR head 3f97e4f), applied on the named commit.

- `QWEN4_STANDIN` sweep on 8c16b2b + HC grouping (baseline 33.4 / 127.3 then 34.0 / 129.3 ms at 1 / 7 rows),
  what each stand-in removes: whole MoE 9.4 ms at 1 row and ~8.6 ms per extra row (experts ~6.4 of it by ablating
  `moeExl3`, shared expert ~1.4); GDN 8.7 / ~3.0 (the projections, not the recurrence); attention 6.7 / ~2.0; HC 4.2 /
  ~2.5. The `gdn_proj` and `moe_router` stand-ins cost more than what they replace: unusable as ablations.
- The expert decode chain alone, 12 chained layers per eval: 149 us per layer at 1 row, 730 at 7 (+97 us per row per
  layer). 12 or 40 distinct layers' weights (11 / 38 GB) read the same as one layer reused: not TLB or working set.
- MLX affine-8 `qmv` chained in one graph: 244-259 GB/s of weights at 12288x2560, 2560x6144, 2560x2560 (~65% of peak).
  The row-identical `mtp_qmv` kernel reads within ~10% of serial `qmv` and batched `quantized_matmul` at 2-7 rows.
- HC reads grouped by row (weights read once per group, configs cached per width), bit-identical to f22a383 in 24/24
  cross-boot greedy comparisons, MTP on and off, A B B A: 1 / 2 / 4 / 7 rows 36.1 / 51.3 / 80.6 / 131.4 and 35.4 /
  49.9 / 79.1 / 129.5 -> 35.1 / 49.0 / 77.7 / 127.4 and 36.3 / 49.1 / 78.6 / 128.4 ms, 2-3% at verify widths and
  nothing at 1 row. On 8c16b2b it passes `test_mtp_equivalence.sh` 11/11 and its MTP output equals 8c16b2b's.
  End to end on 68b6f9f, greedy, 4 prompts x 256 tokens (code / list / prose / story), decode tok/s: default adaptive
  MTP, two A B B A blocks, 131.8 -> 136.0 and 132.1 -> 135.5 summed (+3.2% / +2.5%, 28/28 outputs identical);
  forced depth 3, 43.75 / 38.15 / 28.3 / 25.3 -> 44.75 / 39.15 / 28.35 / 25.6 (+1.7%, faster on every prompt).
- `SUSHI_MTP_DENSE_ROWS=1` read another 2-3% (35.6 / 49.3 / 78.0 / 129.1 and 35.6 / 49.4 / 78.3 / 127.0 -> 35.7 / 47.5
  / 76.1 / 125.0 and 35.2 / 48.2 / 75.8 / 125.5 ms). One `test_mtp_equivalence.sh` run of 8c16b2b + HC grouping +
  dense rows failed 3/11 (the story prompt left `--no-mtp` at output token 16, top-2 gap 1.125 nats; that boot decoded
  at 12 tok/s, under load). Seven reruns passed 11/11: the same commit, dense alone on 8c16b2b and 68b6f9f, router or
  gate alone, and with HC grouping on 68b6f9f. The one wrong value is unexplained, so dense rows stay off by default.
- Ruled out: a vectorized affine-8 reader (one uint2 of codes and two vec4 activations per lane, 8 rows per simdgroup),
  bit-identical to per-row `qmv`. In a chained in-graph ubench it read 10-57% faster on 6k-12k x 2560 and 2560 x 6144
  at 1-3 rows. On the decode meter (8c16b2b + HC grouping + dense rows), A B B A off / on / on / off: 33.96 / 48.59 /
  74.97 / 125.81, 34.65 / 49.87 / 78.21 / 127.24, 34.94 / 47.42 / 76.28 / 124.08, 33.85 / 46.26 / 74.55 / 124.52 ms,
  2-4% slower at 1-4 rows.
- Expert overlap between verify rows on `test_mtp_equivalence.sh` traffic (`SUSHI_EXL3_UNION_HIST`): 20 / 28 / 31% of
  routed slots repeat an expert at 2 / 3 / 4 rows, 41-47% at 6-8. Ruled out all the same: MiMo's grouped gate/up GEMV,
  byte-identical at the qwen geometry, on 68b6f9f at forced depth 3 / 5, greedy, 4 prompts x 256 tokens, A B B A:
  decode -0.8% / -1.3% (code / list / prose / story 43.4 / 37.9 / 27.1 / 24.5 -> 43.0 / 36.9 / 27.5 / 24.5 and
  46.3 / 35.3 / 22.8 / 18.7 -> 45.6 / 34.9 / 22.4 / 18.5 tok/s).
- The sampled shader profiler misattributes decode (HC read 13% of sampled time at 7 rows, ~2.5 ms of ~130 by ablation);
  attribute by stand-in or ablation, never by samples.

<a id="ngram-arm"></a>
## The n-gram gather arm is measured per load (feb9ed7d)

M5 Max 128 GB, macOS 27, Sushi-3bpw (29.8 GB table) and Sushi-4bpw (95.4 GB), `--mtp --kv-quant 8
--mtp-head-kv-quant --ctx-size 128000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --prefix-cache-mem 1GB`,
llmprobe 0.6.12 `--bench-only --rungs 4k,16k --runs 1` (one sample per cell, no median), `taskpolicy -a`, GPU lock held
per arm, NOT a quiet box (no §4b fan protocol available). `SUSHI_FORCE_GPU_FAMILY_FALLBACK=1` for the non-NAX rows.

| arm | calibration read | picked | 2041 tok | ~4.2k | ~16.5k |
|---|---|---|---|---|---|
| 3bpw NAX, table warm | serial 0.67 ms, pool 4.14 ms (0.2x) | SERIAL | 1811.7 | 1890 | 1918 |
| 3bpw NAX, `SUSHI_NGRAM_WARM=0` | serial 20.94 ms, pool 1.88 ms (11.2x) | POOLED | 1535 | 1616 | 1709 |
| 3bpw non-NAX, table warm | serial 0.63 ms, pool 3.75 ms (0.2x) | SERIAL | 1130.9 | 1134 | 1131 |
| 3bpw non-NAX, pool forced | warm at the time of the probe | POOLED | 1015.9 | 990 | 1048 |
| 4bpw NAX, warm declined by the cap | serial 11.45 ms, pool 1.02 ms (11.3x) | POOLED | 1757.1 | 1891 | 1954 |
| 4bpw NAX, warm FORCED past the cap | serial 11.01 ms, pool 1.24 ms (8.9x) | POOLED | 1621.6 | 1620 | 1622 |

- **The arm flips with the state, so it cannot be a constant.** A cold table reads 7.7-11.3x for the pool; a warm one
  5-10x for serial. Forcing the pool on a warm table costs 6-13% of prefill; a serial walk on a cold one costs
  7.7-11.3x in gather time, so the 20% margin leans to the pool.
- **Forcing the 4bpw warm is 8-17% WORSE and buys nothing**: the pread finished all 95.4 GB in 7.9 s, the calibration
  read cold either way (11.45 vs 11.01 ms), because 95.4 GB of table plus ~57 GB of weights does not fit in 128 GB. The
  residency cap already declines it; that is the rule earning its keep.
- `SUSHI_NGRAM_WARM=0` on the 3bpw cost 11-15% of prefill here, but that arm ran against a partly warm page cache:
  `WARM=0` skips the pread, it does not evict. A genuinely cold 3bpw end-to-end cell is still unmeasured, and this box
  cannot produce one (49 GB of weights + 30 GB of table fits inside 128 GB, so nothing is evicted) — a 64 GB box
  would.

<a id="qsa"></a>
## QSA wide verify (b89991a)

Per 12-layer forward on a packed kv8 cache, S=16: 42.2 → 3.8 ms at 8k keys, 13.4 → 4.1 at 64k, 24.0 → 4.1 at 256k.
Prefill gather over the rebuild beats the mask arm at every width (8k keys, S=1024: 227 → 81 ms). Live MCG
Flash-Next kv8: 68k prompt byte-identical, 33k diverges at a 0.125-nat near tie; prefill 1740 → 1763 tok/s (33k).
Declined: packed reads for wide long-context chunks (15-20% slower). Flash-Next attention + indexer is 6-13% of decode
and 4-8% of a depth-3 verify; a matmul2d QSA prototype (branch `qsa-mpp-proto`) cuts the attention part 30-45%
(~2% of a token) and is parked.

<a id="gdn-verify-fold"></a>
## Flash-Next: GDN verify epilogues in the recurrence

`92fd9b71`, ReleaseFast binary built 2026-09-28 10:43:30 (SHA-256
`e79b345002454001b49cf3b1bde86fcc407fb7b3cd7087385de4eaad86926638`), M5 Max 128 GB,
Sushi-3bpw, `--ctx-size 65536 --kv-quant 8 --prefix-cache-entries 0 --no-mtp`.
The forward meter prefills 32768 tokens, captures verify state, and runs 40 forwards
per arm with `SUSHI_DECODE_FWD_UBENCH_GDN_FOLD_ARMS=1`, widths `1,2,3,5,7,9` and
`SUSHI_ROUND_COST_PERSIST=0`. A B B A in one process: A is the existing fused
recurrence plus norm-gate and convolution concat; B folds both epilogues into it.
There was no recorded fold-only comparison on this base, so both arms were measured.

`taskpolicy -a`; GPU lock `codex-gdn-fold-bench`; fans max, 10 seconds idle from
67.4 °C, restored to auto afterward. No other model, build or unit-test job ran
beside the measurement; GUI/background processes remained active. These are
forward timings, not end-to-end generation throughput.

| Rows | A passes, ms/forward | B passes, ms/forward | Change in mean |
|---|---|---|---|
| 1 (control, no fold) | 18.289 / 18.389 | 18.411 / 18.180 | -0.24% |
| 2 | 22.641 / 22.430 | 22.206 / 21.850 | -2.25% |
| 3 | 24.942 / 25.473 | 24.942 / 25.598 | +0.25% |
| 5 | 32.265 / 32.988 | 33.919 / 33.333 | +3.06% |
| 7 | 41.969 / 42.903 | 41.894 / 42.160 | -0.96% |
| 9 (control, no fold) | 51.353 / 51.388 | 51.040 / 51.127 | -0.56% |

At widths 2–7 every B pass records 1548 folded launches (36 layers × 43 warm/timed
forwards), every A pass zero. The folded path removes 72 graph ops per forward.
Two-row B passes beat both A passes; five-row B passes lose to both A passes.
The default therefore serves two rows only. Widths 3–8 remain available to the
parity tests and timing override, not the shipping dispatch. The sub-1% cells are
not evidence of a speedup. Widths 4, 6 and 8 have parity coverage but no timing here.

Parity covers sigmoid and Swish, widths 2–8, both small and real head geometry,
all outputs and rollback states, the pipeline thread-limit fallback, and keeping
real lazy inputs unevaluated during the probe. The production-path test also
checks rollback at every acceptance position against serial decode.


<a id="qwen4-decode-ladder"></a>
## Flash-Next: PLE-safe batched decode ladder

`4f329a4a`, ReleaseFast, M5 Max 128 GB, Sushi-3bpw, llmprobe 0.6.12 on
2026-09-28. Flags: `--ctx-size 65536 --kv-quant 8 --no-mtp --no-pld --no-drafter
--max-concurrent 2 --prefix-cache-entries 0`. Off sets `SUSHI_DECODE_ASYNC_LADDER=0`;
on leaves it unset (batched stride 4, serial off). Probe flags: `--bench-only
--rungs 4k,32k --runs 3 --concurrency 2 --reasoning off --no-save`.

Boot order was off/on/on/off/off/on. Each boot takes its own GPU lock, restores
QoS with `taskpolicy -a`, sets fans to max and cools before starting; cleanup
restores automatic fan control. No builds or unit tests ran alongside timing.
The harness runs three serial samples per rung but only ONE concurrent burst per
rung per boot. These are independent boot samples, not nine bursts per arm.

The final boot was stopped at the user's request to shorten the run, after its
4K burst completed and before its 32K burst. Its completed 4K values come from
llmprobe's progress log; the other five boots have complete JSON reports. No
missing 32K value was imputed.

| Context | Off per-stream decode, tok/s (three boots) | On per-stream decode, tok/s | Median change |
|---|---|---|---|
| 4K | 35.3 / 35.7 / 35.4 | 39.9 / 39.8 / 39.1 (three boots) | 35.4 → 39.8, +12.4% |
| 32K | 25.7 / 25.9 / 25.6 | 28.7 / 28.5 (two boots) | 25.7 → 28.6, +11.3% |

At 4K, aggregate burst throughput including prefill moves from a median 42.6 to
46.1 tok/s (+8.2%). At 32K it is effectively flat (9.5 vs 9.55 tok/s): cold prefill
dominates the request wall time. No claim is made for MTP or single-stream speed.
All on samples exceed all off samples for per-stream decode at both contexts.
The five complete reports mark sustained-load drift steady; the third off boot's
short serial samples varied from 62.4 to 66.6 tok/s and were retained.

Every on boot records the N=2 stride-4 ladder engagement; no off boot does.
Correctness: synthetic eager/lazy PLE and N=2 logits/history parity, full suite
2735 passed / 93 skipped, plus the live batched-equivalence suite (short and long
serial/batched checks, concurrent streams, logprob isolation and kv8 crash guard).

<a id="exl3-gpu-routing-meta"></a>
## Flash-Next: prefill routing metadata on the GPU and the inverse-indexed finish

Main's path (arm 0: host-built window table, then an f16 copy that un-sorts the down plane) against the served path
(arm 1: the window table and inverse built in one GPU threadgroup, the finish reduce reading through the inverse),
alternated inside one boot per row by a local switch in the prefill meter (`SUSHI_PREFILL_UBENCH`): cold prefills from
an empty cache on real text. This change on 90fdedae (4096 rows, 32k) and on 27e81cfc (8192 rows), ReleaseFast, M5 Max
128 GB, Sushi-3bpw, `--ctx-size 65536 --prefix-cache-entries 0`, kv8, `taskpolicy -a`, GPU lock per boot, fans max and
10 s (die 72-83 °C), 2026-10-01; other workers' builds were not frozen. Output bytes are equal (unit test at
Flash-Next geometry).

| chunk | arm order | arm 0, ms per chunk | arm 1, ms per chunk | change of means |
|---|---|---|---|---|
| 4096 rows, median of 3 | 0 1 1 0 0 1 1 0 | 2190.0 / 2216.5 / 2233.9 / 2267.1 | 2141.2 / 2068.9 / 2128.2 / 2145.5 | -4.8% (A B B A sets -4.5%, -5.0%) |
| 8192 rows, median of 5 | 0 1 1 0 0 1 | 3738.3 / 3965.7 / 3999.5 | 3841.2 / 3690.0 / 3844.8 | -2.8% (sets -2.2%, -3.9%) |
| 4 x 8192 rows (a 32k prompt), mean of 2 | 0 1 1 0 0 1 | 18590 / 18325 / 19596 | 18964 / 19047 / 18816 | +0.6%, inside the spread |

Each chunk saves ~0.1 s at both widths (106 ms at 4096 rows, 109 ms at 8192). The saving does not grow with the
chunk, which points at the per-layer host round trip of the host-built table rather than the copy. A 32k prompt would
save ~0.4 s of ~18.8 s (~2%), below the spread of the two-sample 32k run (one arm's samples ranged 17.9 to 20.1 s). The served path also drops the un-sort buffer
(`[rows x 10, 2560]` f16: 210 MB at 4096 rows, 420 MB at 8192).

<a id="mtp-depth-policy"></a>
## MTP depth policy: one chunk from acceptance EMAs below 8k KV (reverted)

c3f29b8e planned a Qwen round below 8192 KV as ONE chunk priced from the acceptance EMAs, with no chunk B; it landed on
a +2% average from solo 256-token prompts on Sushi-2.6bpw (ctx 32768), within the ~5% drift of its own boots. It never
shipped and is reverted.

Its parent 1193f72f against it on Sushi-4bpw (settings below), decode cell median of 7, A B B A: 87.4 / 82.3 against
77.4 / 73.3. The EMAs drove depth ~5 rounds (m_avg 5.06, no extension) at 48.3 ms per round, against m_avg 2.42 with
chunk B at 31.8 ms. With the depth pinned at 3 (`SUSHI_MTP_FORCE_DEPTH=3`), 06a3187b matched v1.1.1 711572e9: decode
93.8 / 94.3 against 92.4 / 90.6, round 33 ms on both.

The revert against main, 26ef54dc against this change, both ReleaseFast, `--mtp --ctx-size 131072` kv8, llmprobe 0.6.12
`--bench-only --rungs 2k,16k --runs 3`, `SUSHI_MTP_TRACE=1` on every arm, `taskpolicy -a`, lock per boot, fans max,
quiet box, AC, 2026-10-05, boots main / revert / revert / main, tok/s:

| pack, cell | main | revert | revert | main |
|---|---|---|---|---|
| Sushi-4bpw, decode | 87.9 | 87.1 | 91.2 | 77.3 |
| Sushi-4bpw, 2k rung | 85.2 | 80.0 | 75.0 | 79.0 |
| Sushi-4bpw, 16k rung | 82.3 | 80.4 | 79.7 | 79.6 |
| Sushi-2.6bpw, decode | 90.8 | 85.2 | 76.4 | 83.4 |
| Sushi-2.6bpw, 2k rung | 81.7 | 84.4 | 82.4 | 82.1 |
| Sushi-2.6bpw, 16k rung | 82.3 | 77.9 | 81.0 | 82.9 |

- On 4bpw's decode prompt main's rounds run m_avg 5.11 / 4.43 at 43.0 / 45.0 ms with no extension, the revert's
  m_avg 2.64 / 3.13 at 32.3 / 36.6 ms with extensions on 13-15% of rounds.
- On 4bpw's 2k rung the one-chunk plan accepts more per round at a similar round time (2.70 / 2.31 against 1.75 /
  2.48), so it wins there.
- On 2.6bpw it does not go deep (m_avg 3.2) and tokens per ms are equal.
- Above 8192 KV both arms run the same planner. Against v1.1.1 on 4bpw in other boots that day (decode 80.8-89.4,
  2k rung 76.1-82.5; llmprobe 0.6.12 and 0.6.13) the revert is at parity.
