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

<a id="glm-a5-verify"></a>
## GLM DFlash2: A5 trunk takes the row tiles, hoist and decode QKV

2026-10-11, M5 Max 128 GB, GLM-5.3-Flash-Sushi-2bpw (A5 g128 trunk, A6 `lm_head`, K2 MCG W14 experts) against
Sushi-2.4bpw (A6 g128 trunk). Before this change every 5-bit verify projection ran one stock qmv per row. Old binary
= `57b7ec5e` (SHA256 prefix `915ff17920c65f41`), new = this change on `57b7ec5e` (`a2a6b8f914e01593`), both ReleaseFast.
`sushi --serve` defaults (kv8 latent, A4 g64 DFlash2 from the pack), llmprobe 0.6.16 `--bench-only` default rungs,
plus two greedy requests per boot. `taskpolicy -a`, GPU lock per boot, QUIET window (no `zig`/`clang` running),
maximum fans, 3 min between boots, one boot per arm in the order shown.
Verify is the median `glm_round_ms verify` over the 36–37 requests of each boot.

| Arm | Power | Decode tok/s | Spec predictable / novel | Ladder ~4.2k / 8.3k / 16.3k | Verify ms/round |
|---|---|---:|---:|---:|---:|
| Sushi-2bpw, old | battery 94% | 45.4 (41.6–45.6) | 58.0 / 37.5 | 41.0 / 41.9 / 40.8 | 49.58 |
| Sushi-2bpw, new | battery 85%, then AC | 53.3 (51.4–54.6) | 64.7 / 36.9 | 41.1 / 42.2 / 45.0 | 45.10 |
| Sushi-2.4bpw, new (A6 path unchanged) | AC | 50.2 (46.9–50.2) | 60.2 / 37.7 | 46.7 / 47.8 / 44.4 | 44.60 |

- Decode +17% on the same pack. At the 8.3k and 16.3k rungs, verify per round is 47.06 / 46.33 ms against 2.4bpw's
  45.82 / 45.64 (old 2bpw: about 51.6). The ~512 rung is left out: new-2bpw's request there slowed every phase about
  2.5x during the battery-to-AC switch, the untouched draft included (10.83 vs 4.12 ms).
- The owner's recorded 2.4bpw run (`8896da54` era, the same llmprobe, rungs 2k–32k) read decode 54 (50.7–54.2), spec
  65.2 / 41.9. Its ladder rungs differ from these, so only the headline cells compare.
- Engagement in the new 2bpw log: `[glm] copy-free QKV engaged (A5g128)`, `[glm-dflash] coefficient hoist rows engaged
  (A5g128)`, `[glm-dflash] affine row tiles engaged (A5g128)`. The A6 lines come from `lm_head`. Both greedy answers
  matched the old binary byte for byte in every 2bpw boot.
- Component, `SUSHI_GLM_QMM_UBENCH` (3 rows, 12 distinct banks per arm, lock, not a quiet window): 8192x4096 serial
  stock 0.104 ms, A5 hoist 0.0645, A5 row tiles 0.078 (A6 hoist 0.089); 4096x16384 0.210 / 0.130 / 0.146. An A5
  hoist holding all 24 coefficients at once ran 0.191 ms, slower than the serial rows.

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

<a id="glm-a6-trunk-dense"></a>
## Prefill: A6 dense-once beyond the KDA banks

Per call at 2048 rows, qmm against dequantize + dense NAX GEMM, six calls per graph, medians of 9, arms interleaved in
one process (`SUSHI_GLM_A6_SHAPES_UBENCH=1`, `a6296520` ancestry, `taskpolicy -a`, GPU lock, QUIET, AC). Bytes are
compared with qmm on random banks.

| K→N | projection | qmm | dense-once | bytes | served |
|---|---|---|---|---|---|
| 4096→1536 | MLA q_a | 0.747 ms | 0.662 (−11.3%) | equal | yes |
| 1536→16384 | MLA q_b | 2.015 | 1.924 (−4.5%) | equal | yes |
| 4096→512 | MLA kv_a | 0.341 | 0.324 (−5.2%) | equal | yes |
| 1536→4096 | indexer wq_b | 0.670 | 0.637 (−4.9%) | equal | yes |
| 4096→2048 | shared gate / up | 0.852 | 0.781 (−8.4%) | equal | no |
| 2048→4096 | shared down | 0.818 | 0.765 (−6.6%) | equal | no |
| gate + up stacked | 4096→4096 | 1.70 (two qmm) | 1.408 (−17%) | equal | no |
| 4096→12288 | dense gate / up | 3.943 | 3.641 (−7.7%) | equal | no |
| 12288→4096 | dense down | 4.203 | 3.958 (−5.8%) | 4421 of 8.4M differ | no |

The served MLA shapes save about 2.7 ms of a roughly 2,470 ms chunk (0.1%), under run-to-run noise. They cost no bill: an
MLA layer expands 140 MiB, a KDA layer's four banks 256 MiB. The shared expert would save about 15 ms a chunk (0.6%; the
stacked gate/up GEMM with a packed-row activation kernel, 12.5 ms of it) but its transient (45 MiB for the stacked pair
plus 16 MiB for down, per pending layer) moves the Sushi-2.5bpw 1M-context default from a 2048-row to a 1024-row prefill
chunk (`GLM release defaults advertise the full context`: 16 MiB per layer still keeps 2048, 32 MiB does not), which also
turns dense-once off. Kept off; the code is on branch `a6-shared-expert-held`. Model level for that arm, one boot,
6,756-token prompt, `--no-drafter --prefix-cache-entries 0`, fans max, AC, old/new/new/old six times: paired `prompt_ms`
−0.4% median (−1.1 to +2.3% per block against a 3% thermal drift), greedy text identical in all 24 requests.

<a id="glm-prefill-pool-clear"></a>
## Prefill: the per-chunk pool clear stays (A/B, not landed)

The prefill chunk loop returns MLX's buffer pool after every chunk (`mlx_clear_cache` in `generate.zig`). Handing the
pool to the next chunk instead lost on speed and on memory. Seam binary `d0d3216a` (main `e8c9c47e` plus PR #41 and a
throwaway env arm), Sushi-2bpw at release defaults (kv8 latent, DFlash2), `--prefix-cache-entries 0`,
`--prefill-trace`, llmprobe's prefill prompt builder at 8,210 and 16,505 tokens, a warmup then three timed requests per
cell, `taskpolicy -a`, GPU lock per boot, fans at max, quiet window, boots A B B A, 2026-10-11 (another worker's benches
ran between boots). Greedy output was equal in all 28 requests.

| arm | 8K tok/s (boot medians) | 16K tok/s | peak footprint 8K / 16K | pool between chunks |
|---|---|---|---|---|
| clear every chunk (shipped) | 768.6, 796.5 | 721.7, 748.6 | 82.1 / 82.3 GB | 0 (2.2-3.0 GB before the clear) |
| keep the pool | 721.7, 737.2 | 693.2, 708.3 | 85.1 / 85.4 GB | 2.9-3.2 GB; 5.5-6.0 after the last chunk |

Keeping the pool: -6.8% at 8K and -4.7% at 16K, with every chunk slower (chunk 0 included) and the fastest keep boot
under the slowest clear boot. The process footprint was 3 GB higher and `/props` `available_bytes` 4.4 GB lower after
the run; `mlx_get_peak_memory` did not move (80.39 GiB). Qwen 2.6bpw at 9,134 tokens, one boot each: 2,221 → 2,121
tok/s (-4.5%), footprint 47.5 → 48.5 GB. A chunk boundary spends 10-17 ms outside the forwards, about 5 ms of it in
the clear, so the clear is not what idles the GPU between chunks.

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

Settle-one-behind tiles and 32-row selection (`12d2e5f6` + `ebb7224c`), Sushi-2.4bpw kv8, GPU lock + QUIET, `taskpolicy -a`.
Whole attention per 2048-row chunk (`SUSHI_GLM_LONGCTX_UBENCH`, one MLA layer, contended box, 6K-32K): about 9 ms lower
than the paired-tile schedule (57 → 48 ms at 6K-16K); the 32-row selection adds −1 to −1.4 ms (locked, quiet:
48.7 vs 50.1 at 6K, 56.3 vs 57.5 at 16K). Model level, one boot with a per-request arm switch, ABBA x2, `--no-drafter`,
prefix cache off, greedy text equal: 7.5K prompt 661.5 → 672.9 tok/s (+1.7%), 15.9K prompt 621.2 → 632.6 (+1.8%).

Serial decode row (`runRow`, this branch on main `075978d7`), Sushi-2.4bpw kv8, `--no-drafter`, `SUSHI_GLM_ROWS_UBENCH=24` B=1
serial ms/token, arms alternated every round in one boot (12 rounds each), GPU lock + QUIET + AC, `taskpolicy -a`:
1K 27.93 → 27.71 (−0.8%), 8K 29.35 → 28.76 (−2.0%), 16K 30.38 → 29.79 (−1.9%); one earlier boot: 27.84 → 27.56, 29.82 → 29.49.
`SUSHI_GLM_DECODE_UBENCH` per MLA layer (µs, dependent 16-layer chain, ~20 µs of it chain plumbing), before → after: 1K 129 → 116,
8K 172 → 157, 32K 187 → 177, 128K 230 → 218.

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

<a id="glm-decode-fusions"></a>
## GLM serial decode: shared-expert and KDA projection fusions (small, exact)

Sushi-2.4bpw, kv8, `--no-drafter`, serial one-row `forwardLast` (`SUSHI_GLM_ROWS_UBENCH`-style loop, arms interleaved in
one process with alternating order each round, 250 rounds, paired differences with standard error), `taskpolicy -a`, GPU
lock, QUIET marker, AC. Arms: all off; shared gate/up/activation kernel; KDA projection kernel; both; both plus the shared
add folded into the routed finish reduce with `shared` as its first input (the landed form).

| arm vs all off | ctx 1K, ms/token | ctx 8K, ms/token |
|---|---:|---:|
| shared gate/up/activation | -0.044 +- 0.020 | -0.059 +- 0.022 |
| KDA projections | -0.105 +- 0.021 | -0.109 +- 0.021 |
| both | -0.131 +- 0.022 | -0.127 +- 0.022 |
| fold only (shared first) | -0.181 +- 0.022 | -0.235 +- 0.022 |
| all three (landed) | -0.231 +- 0.023 | -0.300 +- 0.022 |

Absolute medians of that boot were 33.8 (1K) and 35.6 ms (8K), the box being contended; a quiet boot ran 28.6 ms. The
whole set is 0.7-0.85%. A dispatch costs about 1 us here (consistent with the f_b/g_b in-kernel result), so fusions only
pay through order: the fold with the shared chain LAST was +0.55 to +0.7 ms slower than the unfused add.

<a id="glm-one-row-experts"></a>
## GLM serial decode: one-row routed experts, two tiles and two k-tiles per iteration (exact)

On `ad562c40` plus this landing, Sushi-2.4bpw, kv8, `--no-drafter`, `taskpolicy -a`, GPU lock, QUIET, AC power, fans max.

Component (`SUSHI_GLM_T1_UBENCH=1`, synthetic E288 banks, 4096/2048, top-8 over 24 rotating routes, MCG w14, 24 dependent
layers per eval, medians of 9, arms interleaved in one process; each phase is the difference of prefix chains):

| µs per layer | pair prepare | gate/up | mid prepare | down | finish | chain |
|---|---:|---:|---:|---:|---:|---:|
| n36 lane | 4.1 | 108.8 | 2.8 | 59.5 | 3.4 | 178.6 |
| n36 one-row | 4.5 | 82.7 | 2.9 | 47.0 | 3.5 | 140.5 |
| n40 lane | 4.3 | 109.5 | 2.8 | 59.8 | 3.6 | 179.9 |
| n40 one-row | 5.0 | 84.4 | 3.1 | 48.5 | 3.8 | 144.8 |

Five dispatches per layer before and after; the prepare, mid and finish steps are ~3-4 µs each, dispatch gap included, so
fusing them away is worth at most ~6 µs per layer (not attempted: it needs a cross-group arrival counter).

Model (`SUSHI_GLM_ROWS_UBENCH=24`, B=1 serial, one boot, lane and one-row alternating per round, 12 rounds each):

| ctx | lane | one-row | delta |
|---|---:|---:|---:|
| 1K | 27.04 ms | 25.55 ms | -1.49 ms (-5.5%) |
| 8K | 29.58 ms | 28.02 ms | -1.56 ms (-5.3%) |

Bytes equal the lane chain's at n32..64 and 72/80/96/112/128 (small shape) and at 4096/2048 n36/n40 w14, clamp live.

## Prefill: KDA prework reads q/k/v in place (T2048 layer, 8K prompt)

Settings: Sushi-2.4bpw trunk shapes, `taskpolicy -a`, GPU lock, QUIET, AC power, base `ad562c40`. Synthetic T2048 layer
(`SUSHI_GLM_KDA_PREFILL_UBENCH`, stages queued one eval each, medians of 14, minima in brackets): dequant q,k,v 1.1,
q,k,v GEMMs 7.3, concat 0.57, dequant o 0.33, o GEMM 2.6, FA/GA/beta cluster 0.6, f_b+g_b 0.55, prework 0.89, recurrence
3.6, post 0.39; whole layer 14.7 (stages overlap). A single 8192^3 bf16 GEMM runs 60 TFLOPs, the QKV GEMMs 56, the
output GEMM 53. Dropping the concat (prework addresses the three projections): prework 1.29 -> 0.89 ms with the concat,
whole layer interleaved in one process 14.91 -> 14.46 ms (min), 20.98 -> 20.45 (median, contended).

Live 8338-token prompt, `--no-drafter`, prefix cache off, arms switched per request by a temporary file, W A B B A A B
B A: totals 11805/11907/11901/11913/11971/12058/12102/13066 ms (thermal drift of the box); paired ABBA A 11859 vs B
11904 ms (-0.4%, inside noise). The generated message is byte-identical in all arms.

<a id="glm-opt-combined"></a>
## GLM serial decode and prefill: combined exact levers against `ad562c40`

Sushi-2.4bpw, kv8, `--no-drafter`, prefix cache off, one boot per arm, base `ad562c40` vs branch head, `taskpolicy -a`,
GPU lock per boot, QUIET marker, AC power, fans max. Every arm generates byte-identical text.

Serial decode (`SUSHI_GLM_ROWS_UBENCH=24`, B=1, boots A B B A A B, this landing):

| ctx | base ms/token | branch ms/token | delta |
|---|---:|---:|---:|
| 1K | 27.83 (27.79/27.83/27.87) | 25.37 (25.36/25.37/25.38) | -8.8% (+9.7% tok/s) |
| 8K | 29.61 (29.41/29.53/29.88) | 27.14 (26.99/27.36/27.08) | -8.3% (+9.1% tok/s) |

Prefill (five chat requests per boot, 7.5K-15K prompt tokens, boots A B B A, before the one-row expert GEMVs,
which run on one row only): base 835.3 tok/s (837.4/833.2), branch 876.5 (868.5/884.4), +4.9%.

Levers: prefill queues every layer and waits one interval back; two-row HC RMS mix; settle-one-behind packed tiles and
32-row pool selection; A6 dense-once for the MLA projections; KDA prework in place; serial-decode shared/KDA fusions; HC
normalized carry; one-dispatch MLA decode row; one-row routed expert GEMVs. Prefill chunk widths and bills are unchanged.
