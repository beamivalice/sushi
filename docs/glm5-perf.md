# Performance baselines: GLM-5.3-Flash (`glm5_next`)

The recorded speed numbers for GLM-5.3-Flash (`glm5_next`) on this box. Method, roofline, shared EXL3 kernel timings and
release ladders: [perf-baselines](perf-baselines.md); the rules there (inherit the recorded baseline, name
commit, binary stamp, QoS and lock beside every number) apply to every section here. Architecture: [glm5-arch](glm5-arch.md).

<a id="glm-nonnax"></a>
## GLM-5.3 Sushi-2.3bpw without NAX, rehearsed on the M5 Max

The M1–M4 path on this box: the NAX-less libmlx (the pinned mlx/mlx-c built at deployment target 26.0, zero `_nax`
kernels) loaded through `taskpolicy -a env DYLD_LIBRARY_PATH=…`, plus `SUSHI_FORCE_GPU_FAMILY_FALLBACK=1`. It runs the
M1–M4 kernels on M5 clocks and bandwidth, so the numbers order arms; they are not an M1–M4's speed. 2026-10-04,
`taskpolicy -a`, lock per run, contended box (other workers queued), not quiet.

Sparse latent attention at 16K history (`SUSHI_GLM_ATTN_UBENCH=1`, one process, arms interleaved, median of 12,
`4f174096`); error is row 0 against an FP64 oracle:

| rows | stage | scalar | FP32 composite | fused NAX | max abs error (scalar / composite / NAX) |
|---|---|---:|---:|---:|---|
| 1 | NAX-less | 1.045 ms | 0.505 ms | – | 7.58e-3 / 7.58e-3 |
| 8 | NAX-less | 1.845 ms | 0.628 ms | – | 7.75e-3 / 7.75e-3 |
| 1 | stock (TF32) | 1.054 ms | 0.549 ms | 1.366 ms | 7.58e-3 / 8.05e-3 / 7.58e-3 |
| 8 | stock (TF32) | 1.699 ms | 0.487 ms | 0.464 ms | 7.75e-3 / 7.75e-3 / 7.75e-3 |

NAX-shaped arms on MLX's non-NAX kernels (`SUSHI_GLM_ARMS_UBENCH=1`, NAX-less stage, median of 10, `4f174096`): A6
dense-once T2048 4096→8192 10.38 → 9.52 ms and KDA cluster 0.667 → 0.623 ms, both bit-exact; MLA three-row verify
batch 0.179 = 0.179 ms, exact; MLA head batch query 14.15 → 3.64 ms and value 20.14 → 3.85 ms, rel L2 2.6e-3 / 3.2e-3
(a numerics change). All stay NAX-only: none has passed the model gate or the drift screen off NAX.

End to end, one request per cell (not a bench), `sushi serve` defaults (kv8, A4 DFlash2 in the pack), greedy,
reasoning `low`, server timers; DFlash2 and `--no-drafter` boots of `6d0d2a0f` gave byte-identical answers on all five
requests (Paris, 391, the image's red quadrant, a 10,211-token needle, a 336-token story):

| cell | DFlash2 | serial | scalar decode attention (`4f174096`) DFlash2 / serial |
|---|---:|---:|---|
| prefill, 10,211-token needle | 272.1 tok/s | 252.3 tok/s | 212.9 / 225.9 |
| decode, 336-token story | 33.5 tok/s | 31.8 tok/s | 30.0 / 29.6 (317 tokens) |

`sushi kld compare --limit 1` against the 4x512 BF16 teacher (code prompt, 512 positions): stock path 0.044647 KLD,
top-1 489/512, NLL 0.18532 (`4f174096`); NAX-less path 0.045709, 489/512, 0.18865 (`6d0d2a0f`, +2.4%); the non-NAX
arms on the stock libmlx 0.042636, 490/512, 0.18473 (`6d0d2a0f`); with the scalar decode attention 0.047833, 485/512,
0.19311 (`4f174096`). One prompt spreads ±5% across kernel sets: rounding flips, not a quality step.

<a id="glm-round-levers"></a>
## GLM DFlash2: draft pipelining and fusions, verify chain trims

2026-10-08, M5 Max 128 GB, GLM-5.3-Flash-Sushi-2.4bpw (MCG W14), A4 g64 DFlash2, kv8 latent, default flags, greedy.
Baseline is `3f3ff7bc` (v1.2.1). Interactive QoS (`taskpolicy -a`), maximum fans, GPU lock `opt-dflash`, AC power.
Every change is exact: proposals, targets, logits, captures and replay tapes matched bit for bit in every interleaved
round (0 mismatched of 192 per context).

In-process rounds (`SUSHI_GLM_ROUND_UBENCH`, one load, 96 rounds per arm, arms rotated per round on one state; the
build carried lever switches, SHA256 prefix `f07816a786d346e6`; medians per round, A/A control 0.1-0.2%):

| Context | Arm | Draft ms | Verify ms |
|---|---|---:|---:|
| 8192 | `3f3ff7bc` path | 4.403 | 39.765 |
| 8192 | draft changes | 4.114 (-6.6%) | 39.819 |
| 8192 | + verify trims | 4.144 | 39.356 (-1.0%) |
| 32768 | `3f3ff7bc` path | 4.420 | 41.082 |
| 32768 | draft changes | 4.159 (-5.9%) | 41.138 |
| 32768 | + verify trims | 4.155 | 40.693 (-0.9%) |

Single-lever arms on the same harness (paired means): layer pipelining -2.5%, fused gate/up/SiLU -1.6% to -2.5%,
conv finish + residual -0.9%, two-pass top-16 -1.2% to -1.6%, cached sliding mask -0.6% to -0.8% of the draft;
shared expert in the reduce -0.6% to -0.9%, joined MLA rows -0.2%, batched index weights -0.1% to -0.2%, early layer-0
submit -0.1% to -0.2% of the verify.

Live, one boot of the lever build with both arms alternating per request, 5 runs of 384 greedy tokens per prompt, no
lookup rounds: draft 4.85 → 4.56 ms (-6.2%, code edit at 8035 tokens), 4.19 → 3.91 (-6.4%, prose), 4.85 → 4.55 (-6.2%,
long code at 32043 tokens), medians of paired requests; every answer byte-identical, 2.12-2.22 tokens per step in both
arms. That run also carried a branch-batched top-512 for MLA verify that did not land, so its verify (-0.8% to -0.9%)
and decode (+1.2% to +1.5%) deltas are not this change's. The final binary's live answers match `3f3ff7bc` byte for
byte at the same draft depth.

The draft met the revised -6% goal live (the original target was -8%); the -4% verify target was not reached. The
draft is a chain of about 85 dependent dispatches over 5 layers (3.0 ms forward, 1.1 ms readout at the head's
bandwidth, 0.3 ms lattice); the verify is GPU-bound with routed experts at 18-19 ms (ALU-bound trellis decode), KDA
8-9 ms, MLA attention 5-5.5 ms, shared experts 2 ms and the head 1.1 ms, so trimming dispatches tops out near 2%.

<a id="glm-draft-ten-percent"></a>
## GLM DFlash2: faster fixed-depth drafting

2026-10-06, M5 Max 128 GB, GLM-5.3-Flash-Sushi-2.5bpw (W12), A4 g64 assistant, kv8, async-four verification.
Compare the unchanged drafter at `2b70132a` against fused BF16 two-tap convolutions, a shared sliding mask, final-layer
output trimming and eight-row A4 FFN weight reuse. The two draft nodes plus root are unchanged. Every layer still
sees the entire eight-row noise block; only the final layer's queries, output projection and MLP stop after row two.
The FFN specialization admits only eight rows, BF16 activations, A4 g64 weights and the 4096→12288 / 12288→4096
geometries on M5 Max. Other shapes and devices use MLX.

One fresh load; 60 rotated samples per arm/context after two warmups, with an unchanged control repeated in each
rotation. Times cover a full proposal, including readout and host tree selection, at immutable serving contexts.
Interactive QoS (`taskpolicy -a`), maximum fans, GPU lock `codex-glm-draft-qmv`; diagnostic binary SHA256 prefix
`fdaa9e3ef7a7b030`. Prompt lookup is disabled so every measured round runs the assistant.

| Actual context | Original ms | Convolution/mask/prefix ms | FFN reuse only ms | Combined ms | Original control ms | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| 1048 | 4.5653 | 4.1851 | 4.3955 | 4.0322 | 4.5812 | 11.68% |
| 8215 | 5.0677 | 4.5990 | 4.8353 | 4.4737 | 5.0393 | 11.72% |

Paired savings are 0.5331 ± 0.0146 ms and 0.5940 ± 0.0439 ms (approximate 95% CI). All 12,288 retained hidden values
and all proposed tokens/parents match exactly at each context. Engagement counters show 20 fused convolutions and
12 eight-row FFN projections per three-row round; the final layer uses three-row MLX projections. At 8K the five
sliding masks become one. Convolution/mask changes alone saved 3–4%; adding final-layer trimming reached about
8.5%, and FFN reuse crossed the 10% target. Smaller attention/kernel projections did not show a useful microbenchmark
win and retain MLX.

Live greedy comparisons used a 32-token warmup per arm and four 256-token runs per prompt, alternating A/B order.
Every response message and every `(verification rows, accepted drafts)` trace matched exactly. The table reports
medians of per-request three-row means; end-to-end decode rates include verification and commit.

| Prompt | Original draft ms | Tuned draft ms | Draft reduction | Decode tok/s before → after |
|---|---:|---:|---:|---:|
| code-short | 4.6974 | 4.1752 | 11.12% | 50.85 → 51.25 |
| code-8k | 5.5645 | 4.8943 | 12.04% | 42.07 → 42.49 |
| copy-8k | 5.5972 | 4.9528 | 11.51% | 54.14 → 54.81 |
| novel-short | 4.8651 | 4.2492 | 12.66% | 40.17 → 40.73 |

Verification time stayed within 0.2% across these paired live cases. The isolated draft saving translates to about
0.8–1.4% higher end-to-end decode throughput in this run, since verification dominates the round.
The clean ReleaseFast suite passed 3,425 tests, with 108 skipped and zero failures.

<a id="glm-w14-lanes"></a>
## GLM DFlash2: W14 packs take the lane and batched-row expert paths

2026-10-07, M5 Max 128 GB, GLM-5.3-Flash-Sushi-2.4bpw (MCG W14, K2.25 on L03–L36, K2.5 on L37–L44), A4 g64
DFlash2, kv8, default flags. llmprobe 0.6.15 `--bench-only --runs 2`, one boot per arm, 180 s fans-max idle before
each boot, `taskpolicy -a`, GPU lock per boot (`glm-bisect-*`), AC power. `177c526f` (release binary) against the
same tree with the W12-only gates on the decode lane, lane down and three/four-row expert paths removed (binary
SHA256 prefix `5e7c14f922a0`). Decode tok/s (llmprobe tokens per step) and the server's median verify ms per round:

| Context | `177c526f` decode | fix decode | `177c526f` verify ms | fix verify ms |
|---|---:|---:|---:|---:|
| 2K | 37.2 (2.29) | 48.4 (2.36) | 55.04 | 42.13 |
| 8K | 37.6 (2.36) | 49.1 (2.43) | 56.00 | 42.98 |
| 16K | 37.7 (2.38) | 43.5 (2.19) | 57.06 | 43.81 |
| 64K | 34.4 (2.29) | 45.5 (2.43) | 60.00 | 46.78 |

Verify per round drops 22–24%; draft, replay and commit are unchanged, and a 384-token greedy answer is
byte-identical. The W14 verify times match the dev2 W12 packs' (2.3bpw: 43.6–47.6 ms at 2K–64K), so the mixed-rate
layout costs nothing measurable. One arm each (owner call: the effect is many times the run-to-run spread).

<a id="glm-three-prepared-input"></a>
## GLM DFlash2: prepared A6 inputs and HC expansion normalization

2026-10-06, M5 Max 128 GB, GLM-5.3-Flash-Sushi-2.5bpw (W12), A4 g64 assistant and kv8.
Baseline is `35feabc1` (1.2.0-dev2). Three-row NAX verification prepares the A6 input's power-of-two
coefficients and BF16 group sums once, then reuses them across output tiles. HC expansion also emits the
next collapse's FP32 normalized input, preserving the intervening BF16 rounding and native RMS reduction.
The default three-row schedule uses two-layer asynchronous groups; explicit schedules and other widths
retain their existing behavior. Draft depth stays fixed at two draft tokens plus the root.

A fresh model load after 180 seconds of cooling at maximum fans, interactive QoS and an exclusive GPU lock;
160 rotated measurements per arm and context, with two warmups and a repeated baseline. Fixed token chains
and prefixes. Timing includes the vocabulary head, captures and replay arrays, excluding drafting and commit.
Diagnostic binary SHA256 prefix `d3cc2364000bec64`.

| Context | Baseline ms | Candidate ms | Repeated baseline ms | Reduction | Paired savings, approximate 95% CI |
|---|---:|---:|---:|---:|---:|
| 1024 | 40.5360 | 38.9679 | 40.6239 | 3.87% | 1.5681 ± 0.0878 ms |
| 8192 | 42.9172 | 41.3030 | 42.9044 | 3.76% | 1.6142 ± 0.1295 ms |
| 32768 | 44.7481 | 43.1758 | 44.7137 | 3.51% | 1.5723 ± 0.0782 ms |

**The additional 5% target was not reached.** Earlier hotter multi-arm runs sometimes approached 5%;
the cooler repeat above is the retained result. Complete logits, captures and replay arrays matched exactly
for one through four rows at all three prefixes. The three-row path engaged 89 expansion/normalization
fusions and 157 prepared affine projections per round. Vectorized prepared inputs did not establish a
repeatable additional gain and were removed, as were configuration caches, fused QKV, fused preparation
inside HC collapse, alternate layer boundaries and the other unproductive experiments.
The clean full native suite passed 3431 tests, with 108 skipped and zero failures.

<a id="glm-three-value-norm"></a>
## GLM DFlash2: three-row value reuse and normalized residual mixing

2026-10-06, M5 Max 128 GB, GLM-5.3-Flash-Sushi-2.5bpw (W12), A4 g64 assistant, kv8, async-four schedule.
Extend the `0fb194b9` verifier with exact three-row reuse in the MLA value projection and a fused HC collapse/RMS
kernel. The value projection retains the serial A6 dot order. The normalization retains the intermediate BF16
rounding and MLX's four-values-per-thread RMS reduction. Both new paths are restricted to three-row NAX verification;
other shapes and device paths retain their existing operations. Drafting and draft depth are unchanged.

One fresh model load, 60 rotated samples per arm/width/context after two warmups; fixed prefixes and token chains.
Compare original `f40fa548`, current `0fb194b9`, each new component alone, and both. Interactive QoS, maximum fans,
GPU lock `codex-glm-five-verify`; diagnostic binary SHA256 prefix `89f620e8db05d383`. Timings include the vocabulary head
and replay/capture arrays, excluding drafting and commit.

| Context | Original ms | `0fb194b9` ms | Value only | Norm only | Both | Total reduction | Increment |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 43.148 | 41.229 | 41.181 | 40.851 | 40.566 | 5.98% | 1.61% |
| 8192 | 45.107 | 43.217 | 42.979 | 42.785 | 42.739 | 5.25% | 1.10% |

Approximate paired 95% CI half-widths for total savings are 0.244 and 0.219 ms (savings 2.582 and 2.368 ms);
for the increment, 0.282 and 0.265 ms (savings 0.664 and 0.477 ms). Thus the measured means meet the additional 5%
verification target at both prefixes; individual runs remain noisy. One-, two- and four-row controls stayed within
0.3%. Complete logits, captures and replay arrays matched byte for byte at every width and context. Counters proved
11 value projections and 90 fused normalizations per three-row round. Clean GLM tests include weighted RMS,
NaN/Inf and captured-coefficient cases, serial value-projection parity, and explicit projection engagement.
The clean GLM suites passed 313 tests, with 3 skipped and zero failures.

<a id="glm-three-output-tiles"></a>
## GLM DFlash2: two output tiles in the three-member expert path

2026-10-06, M5 Max 128 GB, GLM-5.3-Flash-Sushi-2.5bpw (W12), A4 g64 assistant, kv8, async-four schedule.
The three-row, 24-route lane kernel computes two 16-column output tiles per threadgroup, reusing each member's
input loads. Dot accumulation and F16 reduction order are unchanged; other verification widths retain their
existing kernels. This extends the three-member reuse in `83585178`.

One fresh load, fixed source-text prefixes and token chains, 40 rotated samples per arm/width/context after two
warmups. Compare original `f40fa548`, three-member reuse alone, and the final tiled kernel, plus a repeated original
control. Timings include head and replay/capture arrays, excluding drafting and commit. Interactive QoS, maximum
fans, GPU lock `codex-glm-three-final-model`, diagnostic binary SHA256 prefix `d31f5b24ac2e4c91`.

| Context tokens | Original ms/round | Three-member reuse | Final ms/round | Total reduction | Tile increment |
|---|---:|---:|---:|---:|---:|
| 1024 | 41.394 | 40.297 | 39.463 | 4.67% | 2.07% |
| 8192 | 43.307 | 42.796 | 41.570 | 4.01% | 2.86% |

Approximate paired 95% confidence intervals for total savings: 1.931 ± 0.141 ms and 1.737 ± 0.281 ms; for the tile
increment: 0.834 ± 0.137 ms and 1.226 ± 0.270 ms. The repeated original control differed by −0.126 ms at 1K and
+0.300 ms at 8K. All arms matched complete logits, captures and replay arrays by byte hash at widths 1–4 and both
contexts; engagement counters confirmed the selected expert path. Clean GLM suites: 313 passed, 3 skipped, zero failures.

### Actual three-row draft rounds

Greedy generation, lookup disabled to isolate three-row verification, one 32-token warmup and four 256-token runs
per workload and arm. The statistic is the median of each request's mean verification time over its full three-row
rounds. Every arm produced identical messages and acceptance paths. Both code workloads ran in the first boot;
copying and novel writing ran together in a later boot after an interruption. Comparisons stay within each boot;
partial interrupted copying runs are excluded. All completed runs used the original vision setting.

| Workload | Original verify ms | Final verify ms | Reduction | Decode tok/s, original → final |
|---|---:|---:|---:|---:|
| Short code | 45.431 | 43.535 | 4.17% | 48.32 → 50.16 |
| Code, 8035 prompt tokens | 45.930 | 44.160 | 3.85% | 41.18 → 42.61 |
| Copy, 8042 prompt tokens | 45.573 | 43.263 | 5.07% | 54.28 → 56.89 |
| Novel writing | 44.626 | 43.235 | 3.12% | 39.92 → 40.90 |

The extra 5% target is reached on copying, with 4.0–4.7% at the fixed prefixes and 3.1–5.1% across these live cases.
No depth policy changed. Separate shared-FFN streams, four-output tiles, and hybrid tile layouts were rejected.

<a id="glm-three-expert-reuse"></a>
## GLM DFlash2: reuse an expert across three verification rows

2026-10-06, M5 Max 128 GB, GLM-5.3-Flash-Sushi-2.5bpw (W12), A4 g64 assistant, kv8, async-four schedule.
Against `f40fa548` (including the earlier verification optimizations), reuse the cooperative expert dot's decoded
weights across up to three matching routes in the three-row, 24-slot lane path. Other widths keep two-member reuse.
Each member retains the original F16 dot accumulation and reduction order. The benefit depends on routing overlap.

One fresh model load, fixed source-text prefixes and token chains, 40 rotated samples per arm/width/context after
two warmups, with the unchanged baseline repeated as a control. Timings include the head and replay/capture arrays;
drafting and commit are excluded. Interactive QoS, maximum fans, GPU lock `codex-glm-three-group-model`.
Diagnostic binary SHA256 prefix `6ec70806ece6814c`.

| Context tokens | Verify rows | Baseline ms/round | Three-member reuse ms/round | Reduction | Paired saving, approximate 95% CI |
|---|---:|---:|---:|---:|---:|
| 1024 | 3 | 41.297 | 40.331 | 2.34% | 0.966 ± 0.115 ms |
| 8192 | 3 | 43.884 | 43.104 | 1.78% | 0.780 ± 0.247 ms |

All arms matched complete logits, captures and replay arrays by byte hash at widths 1–4 and both contexts.
The repeated baseline differed by 0.033 ms at 1K and −0.026 ms at 8K for three rows. Draft depth and quantization
are unchanged. Fused gate/up activation and larger asynchronous evaluation groups were measured and rejected:
neither reduced full verification time. The clean GLM suites passed 313 tests, with 3 skipped and zero failures.

<a id="glm-verify-final"></a>
## GLM DFlash2: three/four-row verification, final combined result

2026-10-06, M5 Max 128 GB, GLM-5.3-Flash-Sushi-2.5bpw (W12), A4 g64 assistant, kv8 latent, async-four schedule.
One ReleaseFast binary and model load, 40 alternating samples per width and arm after two warmups; fixed source-text
prefixes and token chains. Timings include the vocabulary head and replay/capture arrays, exclude drafting and commit.
GPU lock `codex-glm-final-model`, interactive QoS, maximum fans. Instrumented binary SHA256 prefix `c252977fe9625d03`.

The original arm restores the pre-`83e1d799` QKV-only A6 hoist and its dynamic row offset. The final arm includes `83e1d799`, `d08b6ed1`,
three/four-row FP32 attention batching, direct FP32 gathers, four-row value projections, and two-output A6 SIMD tiles.

| Context tokens | Verify rows | Original ms/round | Final ms/round | Reduction |
|---|---:|---:|---:|---:|
| 1024 | 3 | 46.861 | 41.476 | 11.49% |
| 8192 | 3 | 49.005 | 43.578 | 11.07% |
| 1024 | 4 | 57.764 | 51.277 | 11.23% |
| 8192 | 4 | 62.199 | 54.916 | 11.71% |

One- and two-row controls stayed within 0.4%. Every arm matched complete logits, captures and replay arrays by byte
hash at all four widths and both contexts. No draft-depth policy or quantization changed. Scratch remains capped at
32 MiB per native MLA layer; the FP32 gather removes the intervening BF16 bank while preserving its rounding.
The clean ReleaseFast build passed the complete suite: 3420 tests passed, 108 skipped, zero failures.

### Wider drafting remains workload-dependent

A separate live pilot used the preceding four-row candidate (before the final FP32-gather/tile combination), greedy
sampling, lookup disabled, one warmup and four 256-token runs per cell. N2 means two drafts plus the root; N3 adds
one draft. The bounded N3 readout projected only its three usable draft positions. All generated messages matched.

| Workload | N2 tok/s | N3, full readout | N3, bounded readout |
|---|---:|---:|---:|
| Short code | 48.01 | 47.29 | 48.35 |
| Code, 8035 prompt tokens | 40.32 | 35.81 | 36.45 |
| Copy, 8042 prompt tokens | 52.18 | 52.11 | 53.03 |
| Novel writing | 38.84 | 36.11 | 36.85 |

Bounding readout saved about 1.1 ms of drafting, but N3 still lost on the longer code and novel-writing cells. The
three-row default remains; the faster four-row verifier also serves existing lookup chains and mixed-request ticks.
The N3 policy/readout experiment and benchmark switches were removed from production code.

<a id="glm-three-row-a6-offset"></a>
## GLM DFlash2: specialize the complete three-row A6 tile

2026-10-06, M5 Max 128 GB, GLM-5.3-Flash-Sushi-2.5bpw (W12), A4 g64 assistant, kv8 latent.
Parent `83e1d799` against `d08b6ed1`'s constant row offset. The helper admits exactly three rows and dispatches one
complete row tile; fixing its offset at zero lets Metal remove unreachable masked loads without changing arithmetic.

One ReleaseFast instrumented binary, one model load, fixed source-text prefixes and token chains, async-four schedule.
Forty alternating samples per depth after two warmups; includes the vocabulary head and all replay/capture arrays,
excludes drafting and commit. GPU lock `codex-glm-bounds-model`, `taskpolicy -a`, maximum fans, no concurrent build
or GPU job. Timing run began 02:13 Asia/Bangkok at 45°C. Instrumented binary SHA256 prefix: `f12465f57a687268`.

| Context tokens | Original verify ms/round | Constant offset ms/round | Reduction | Paired saving, 95% CI (ms) |
|---|---:|---:|---:|---:|
| 1024 | 44.505 | 42.077 | 5.46% | 2.428 ± 0.033 |
| 8192 | 46.166 | 43.661 | 5.43% | 2.505 ± 0.152 |

Unaffected one-, two- and four-row controls stayed within 0.6%. Full-model logits, captures and replay tapes matched
by byte hash at every measured depth in both contexts; the 128-token greedy response also matched the earlier baseline.
The clean build passed 310 GLM/EXL3 tests with 3 skips. This is an additional verification-time gain over the preceding
A6 coefficient-reuse change below; no end-to-end throughput gain is inferred from this table.

<a id="glm-three-row-a6"></a>
## GLM DFlash2: three-row A6 coefficient reuse beyond QKV

2026-10-05, M5 Max 128 GB, GLM-5.3-Flash-Sushi-2.5bpw, A4 g64 assistant, kv8 latent.
Parent `24da5f95`; instrumented ReleaseFast binary built 22:33:36 Asia/Bangkok, SHA256 prefix `216b3800ee543d2f`.
GPU lock `codex-glm-verify`, `taskpolicy -a`, maximum fans and ten seconds idle before boot; no concurrent build or GPU job.

One model load, 20 alternating old/new samples per depth after two warmups, fixed source-text prefix and token
chain, async-four schedule. Timings include the vocabulary head and all replay/capture arrays, exclude drafting
and commit. The isolated comparison is the original QKV-only hoist against the same kernel serving every
supported three-row A6/group128 projection; attention is identical in these two arms.

| Context tokens | Original verify ms/round | Expanded A6 verify ms/round | Reduction |
|---|---:|---:|---:|
| 1024 | 46.311 | 45.192 | 2.4% |
| 8192 | 48.188 | 47.197 | 2.1% |

Full-model logits, captures and replay tapes matched by byte hash at depths 1–4 in both contexts. The affine-row
tests also compare every BF16 output against serial qmv and assert hoist engagement at the additional shapes.
One-, two- and four-row dispatch remains unchanged. Fused QKV, alternate threadgroups, async-two scheduling
and MLA batching experiments are not part of this change. This is a verification-time result, not a measured
end-to-end throughput gain.

<a id="glm-prefill-chunk"></a>
## GLM-5.3-Flash: prefill chunk 2048 against 4096 (observation, not a recorded number)

Sushi-2.3bpw, BF16 and kv8 latent, 30,065-ID prompt, separate boots on a box contended by other workers' builds and
tests, `taskpolicy -a`, GPU lock per boot, 2026-10-04: 2048-row chunks prefilled at 520-676 tok/s (eight boots),
4096-row chunks (the old `--no-drafter` auto pin) at 451-592 tok/s (three boots). GLM's auto chunk is now capped at
2048 for numerics
([glm5-kernels](glm5-kernels.md#prefill-chunk-2048-two-layers-pending)); a quiet-box A/B is owed.

<a id="glm-longctx"></a>
## GLM-5.3-Flash: long-context profile and the exact index-scoring wins (contended)

Sushi-2.5bpw at release defaults (kv8 latent, A4 g64 DFlash2, vision, auto context), one request per cell: a
three-sentence summary of repository source, 256 outputs, T=0, server timers. `taskpolicy -a`, GPU lock, fans at
max (die 84-86 °C), box contended by other workers' builds, 2026-10-04. Absolute tok/s waits for the release bench.

Baseline, main `a28b15de`, which prefilled at 512 rows (the quarter-share rule; fixed in `7324d263`):

| context | prefill tok/s | DFlash2 decode tok/s | serial decode ms/token | verify ms/round | accepted/round |
|---|---:|---:|---:|---:|---:|
| 2K | 487 | 33.3 | 34.8 | 59.5 | 1.28 |
| 8K | 588 | 33.6 | 36.3 | 57.9 | 1.19 |
| 32K | 476 | 30.4 | 35.5 | 60.3 | 1.08 |
| 128K | 292 | 25.0 | 38.0 | 70.4 | 0.97 |

Inside the 128K prefill: 556 tok/s at 0-8K, 454 at 24-32K, 317 at 56-64K, 193 at 120-128K, the scalar index scorer
(one MLA layer at 128K: scores 553 of 634 ms per 2048-row chunk). At 128K DFlash2 (25.0) lost to serial (26.3).

Component meter, quiet box (load 0.94), lock, `SUSHI_GLM_LONGCTX_UBENCH`: one MLA layer on a synthetic kv8 state, whole
sparse attention per 2048-row chunk, arms interleaved in one process (cumulative; `641a9300` carries all four):

| context | main | + tree scorer | + B32 tiles | + ranked top-512 | + lazy NAX tiles |
|---|---:|---:|---:|---:|---:|
| 8K | 80.8 ms | 57.5 | 57.5 | 58.1 | 57.8 |
| 16K | 107.3 | 107.0 | 107.3 | 101.8 | 56.0 |
| 32K | 155.3 | 154.8 | 155.9 | 148.8 | 61.0 |
| 64K | 332.5 | 115.0 | 114.9 | 104.7 | 104.5 |
| 128K | 633.7 | 196.9 | 175.9 | 158.8 | 158.7 |

A decode row's attention per MLA layer: 0.82 → 0.55 ms at 128K, 0.55 → 0.50 at 32K. Appending a chunk to the latent and
pooled buffers costs 0.4 ms per layer at 8K and 1.8 ms at 128K (no reservation lever there).

Model level, one boot of main `223f5f48` + `b4a2b6c4` with per-request arm switches, 2048-row prefill, arm 0 = main's
index path, arm 4 = `641a9300`'s; greedy text equal in every cell, logits hashes equal at every forward (273/273 at
32K serial, 321/321 at 128K serial):

| cell | prefill tok/s arm 0 → 4 | decode tok/s arm 0 → 4 |
|---|---|---|
| 32K DFlash2 (0,4,4,0) | 658 / 646 → 794 / 789 | 30.6 / 31.7 → 32.7 / 32.6 |
| 32K serial | 650 → 799 | 30.0 → 30.5 |
| 128K serial | 363 → 637 | 28.0 → 28.6 |
| 128K DFlash2 (4,0) | 329 → 604 | 27.8 → 31.0 |

Verify per round at 128K: 69.1 → 61.5 ms, so DFlash2 again beats serial there (31.0 vs 28.6). The 2048-row prefill
peaked 2.14 GB above active at 32K and 2.80 GB at 128K (bill 4,984 MiB). A 2K prompt prefilled at 669 tok/s at 512
rows and 853 at 2048 (one request each).

Quiet bench, binary `b8267038` (carries the four wins), llmprobe 0.6.13 `--bench-only --rungs 2k,8k,32k,128k --runs 1`,
`--prefill-chunk 2048` on every arm, fans at max (die 54 °C), load ~1.2, no compiles, GPU lock per arm, 2026-10-04:

| arm | decode tok/s | TTFT | prefill, 9.4K prompt | ladder decode 2K / 8K / 32K / 128K | speculation |
|---|---:|---:|---:|---|---|
| Sushi-2.3bpw, `--kv-quant 16` | 43.0 | 346 ms | 865.7 | 42.1 / 39.8 / 40.0 / 37.8 | ×1.52 (predictable 52.9, novel 34.9) |
| Sushi-2.5bpw, kv8 (default) | 46.3 | 354 ms | 851.7 | 41.2 / 42.7 / 43.1 / 38.8 | ×1.56 (50.0, 32.1) |
| Sushi-2.5bpw, `--kv-quant 16` | 45.7 | 351 ms | 849.7 | 39.4 / 41.9 / 33.3 / 36.3 | ×1.51 (50.2, 33.2) |

kv8 and BF16 latents decode within noise on one pack; the 32K ladder gap is acceptance (2.74 against 2.06 tokens
per step). Four concurrent requests decode 44 tok/s in aggregate against ~34 alone. The 2.3bpw row matches or beats
the `4fcb541e` table in [glm5-arch](glm5-arch.md#recorded-performance).
