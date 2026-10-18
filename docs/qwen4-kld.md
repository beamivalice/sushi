# Flash-Next: KLD of the served packs

Flash-Next KLD tables, split out of [quality-kld](quality-kld.md); everything shared by the served architectures stays there.


## Flash-Next (16x512, first EOS, 7186 positions, kv8)

| pack | KLD | top-1 | cosine loss | all positions |
|---|---|---|---|---|
| mlx-serve mixed-4-8bit (affine 4-bit gs64 / 8-bit; the control) | 0.0818 | 91.39% | 2.63% | 0.0752 |
| Sushi-3bpw, first release (MCG K3 w15, bf16 table; binary 7ed9795) | 0.1012 | 90.26% | 3.14% | 0.0931 |
| Sushi-4bpw, first release (MCG K4 w15, bf16 table; binary 30a27ba) | 0.0632 | 92.99% | 2.25% | 0.0588 |
| Sushi-3bpw, published 2026-09-29 (MCG K3 w14, 4-bit g32 table; binary 942134d) | 0.1036 | 90.31% | 3.11% | 0.0941 |
| Sushi-4bpw, published 2026-09-29 (MCG K4 w15, bf16 table; binary 942134d) | 0.0592 | 92.89% | 2.18% | 0.0554 |

The control row ran on binaries a05d15f / 28d7fab (the KLD tool is unchanged between them). Sushi-4bpw reads below it.

Comparison packs, all on binary b64c5a0e (weights in GPU memory, n-gram table excluded; Sushi-3bpw 0.10123 and
Sushi-4bpw 0.06319 reproduce on it): affine q3 = routed experts 3-bit g64, dense 8-bit, bf16 n-gram table; oQe =
oMLX packs as published, restacked for sushi with their 4/5-bit n-gram table unchanged (oQ4e ships the table divided
by a `weight_scale` tensor, folded into its scales by the restack); mlx-serve packs as published, both sharing one 4-bit
n-gram table. Sizes are GiB of the weight files the engine loads (the Sushi packs once shipped the vision tower twice,
0.84 GiB, and no longer do).

| pack | GiB | KLD | top-1 |
|---|---|---|---|
| oMLX oQ5e (GBP-DE) | 83.97 | 0.0625 | 92.40% |
| mlx-serve iQ-MLX 4.7bpw (2026-10-02 build; see below) | 70.13 | 0.0676 | 92.26% |
| mlx-serve mixed-4-8bit (ddalcu; the control above) | 70.13 | 0.0818 | 91.39% |
| oMLX oQ4e (Jundot) | 69.21 | 0.1370 | 88.87% |
| affine q3 | 54.94 | 0.1444 | 88.05% |
| Vontra 4-bit g32 (TensorFold), as published | 75.60 | 0.2074 | 85.01% |
| mlx-serve iQ-MLX 3.3bpw (ddalcu; imatrix-weighted affine) | 50.60 | 0.1987 | 86.28% |
| Sushi-3bpw first release with mixed-4-8bit's 4-bit g32 n-gram table (as first published) | 49.33 | 0.1047 | 90.34% |
| Sushi-4bpw first release with the same 4-bit g32 table | 63.68 | 0.0666 | 92.35% |
| Sushi-3bpw published 2026-09-29 with the bf16 table (binary 942134d) | 49.33 | 0.1006 | 90.80% |
| Sushi-4bpw published 2026-09-29 with the 4-bit g32 table (binary 942134d) | 63.68 | 0.0654 | 92.72% |
| Sushi-2.6bpw (binary ad5e6be8, 2026-09-27; Sushi-3bpw's 0.10123 and 0.1047 reproduce on it bit for bit) | 43.95 | 0.1303 | 89.33% |
| Sushi-2.6bpw with the 4-bit g32 table (the published Sushi-2.6bpw) | 43.95 | 0.1355 | 89.08% |
| Sushi-2bpw with the 4-bit g32 table (the published Sushi-2bpw; bf16 KV, see below) | 34.97 | 0.2080 | 85.94% |

iQ-MLX 4.7bpw, measured 2026-10-02: ReleaseFast at `3e700850` plus existing working-tree edits,
binary SHA-256 `f5f5a56ef11396d066cd6577121147fdb93fe3fa0eb6a4854ef3c8ec67ac6a07`
(mtime 2026-10-02 01:28:37 +0700). Flash-Next `mlx-serve-bf16-16x512-raw`, kv8,
`--tokens 512 --top-k 10 --ctx-size 8192 --no-mtp`, shipped 4-bit g32 n-gram table.
First-EOS KLD **0.067597376**, top-1 **92.2627%**, NLL **0.396296626**, 7186 positions;
all-position KLD **0.062826856**, top-1 **93.0298%**, 8192 positions. Weight shards total 70.13 GiB,
excluding the n-gram table. `taskpolicy -a`, GPU lock `kld-iq47-codex`. This row uses a newer binary than
the historical comparison rows; differences under ~1% are within the measured rounding-flip floor.

Release 1.0.4 check: `ad4a3ce0` plus the context-bill change, ReleaseFast binary SHA-256
`2aeee2e678521727e66994d75260c25cd4cffd0d05ecb797c73210a2b0ea9704` (mtime 2026-09-26 15:26:43 +0700),
Sushi-3bpw, the Flash-Next 16x512 raw teacher, kv8, `--tokens 512 --top-k 10 --ctx-size 8192`, no MTP:
first-EOS KLD **0.10469852**, top-1 **90.3423%** (7186 positions); all-position KLD 0.09614411, top-1 91.1743%.
This reproduces the published 0.1047 baseline (-0.0014% relative, inside the 1% floor), without an old-binary rerun.
M5 Max 128 GB, `taskpolicy -a`, GPU lock `release-v1.0.4-kld-sushi3bpw`; conversion suspended, no timing claim.

Sushi-2.6bpw rows: `ad5e6be8`, ReleaseFast binary SHA-256
`e85c49f28330474a581954e9eb439294097e83838d42f71c65ab5338113bda4a` (mtime 2026-09-27 14:19:09 +0700),
`mlx-serve-bf16-16x512-raw`, kv8, `--tokens 512 --top-k 10 --ctx-size 8192`, no MTP, 7186 positions to first EOS;
all-position KLD 0.1183 (bf16 table) and 0.1229 (4-bit table). Same-binary controls: Sushi-3bpw scores 0.101234497
with the bf16 table and 0.104698517 with the 4-bit table, the b64c5a0e figures to nine digits. M5 Max 128 GB,
`taskpolicy -a`, GPU lock `k26-kld` per run, 2026-09-27.

Release 1.2.2 check, 2026-10-11, published Sushi-2.6bpw (4-bit g32 table), kv8, no MTP, 7186 positions to first EOS:

| | KLD | top-1 | NLL |
|---|---|---|---|
| 1.2.2 check (`bb9adb6a`) | 0.137095 | 89.23% | 0.455440 |
| `ad5e6be8` row above | 0.135508 | 89.08% | 0.454090 |
| change | +1.17% | +0.15 pp | +0.30% |

The pipelined GDN prefill landed in between, and its output is not bit-identical. The owner accepted the drift as
noise. Run: ReleaseFast, binary SHA-256 `d8c741577e52dfc2301554fc96642a3df4f289405e302fe4444e605c91fc11e9`,
`taskpolicy -a`, GPU lock `v122-validate`.

Sushi-2bpw row: sushi v1.0.4 ReleaseFast, binary SHA-256
`1c2c952090f2642c5119061fc94a552a131b30ca698779bd9593d1c60f9db934` (mtime 2026-09-26 19:51:49 +0700),
`mlx-serve-bf16-16x512-raw`, `--kv-quant off` (bf16 KV, unlike every other row), no MTP, 7186 positions to first EOS:
KLD 0.208021, top-1 85.94%, NLL 0.539001; all positions 0.189374 / 87.30% / 0.484980. M5 Max 128 GB, 2026-09-28.

Rows published 2026-09-29: binary built from `942134d`, ReleaseFast SHA-256
`baffa6de2624f403be75821caefc924cf4ca3fc087a0c11ce232acf60cdfd8cb` (mtime 2026-09-28 21:01:15 +0700),
`mlx-serve-bf16-16x512-raw`, kv8, `--tokens 512 --top-k 10 --ctx-size 8192`, no MTP, 7186 positions to first EOS.
M5 Max 128 GB, `taskpolicy -a`, GPU lock per run, 2026-09-29.


<a id="kv-width"></a>
### KV cache width (the one setting that is not the pack)

The tables above rank packs at kv8. The cache width is a separate dial, and the first measurement of it:

| Sushi-3bpw, 16x512 raw | mean KLD | top-1 | NLL |
|---|---|---|---|
| `--kv-quant 8` (the default) | 0.104699 | 90.34% | 0.431483 |
| `--kv-quant 4` | 0.114179 | 89.65% | 0.437941 |
| delta | **+0.009480 (+9.05%)** | **-0.70 pp** | +1.50% |

kv4 is 9% of KLD, nine times the ROUNDING-FLIP floor, so it is a real cost and not an accumulation artefact. It
spends 49% of the gap between this pack and the affine 4/8 control. It nearly halves the cache's bytes per token,
which is the only reason to take it.

Binary `db249826` (Zig sources identical to `e8e2a3cb`), M5 Max 128 GB, the Flash-Next 16x512 raw teacher,
`--tokens 512 --top-k 10 --ctx-size 8192`, no `--mtp`, 7186 positions to first EOS. Both arms ran on the same binary,
so the delta stands on that; the absolute kv8 figure reads 0.1047 where the table above records 0.1012, a +3.5% gap
against a different binary and flag set, which is why the delta is quoted rather than either absolute.

Sushi-2.6bpw (4-bit n-gram table, binary `ad5e6be8`, same settings): `--kv-quant 4` scores 0.145841 / top-1 88.84% /
NLL 0.465802 against kv8's 0.135508 / 89.08% / 0.454090, +7.63% KLD and -0.24 pp.

kv6 (Sushi-2.6bpw, the f32-GDN raw teacher `qwen38-flash-next-bf16-gdnf32-v1`, binary `84d3ed8f`, `--qwen-gdn 32 --no-mtp`, 7539
positions to first EOS, `taskpolicy -a`, GPU lock): kv8 0.114252 / 90.79% / NLL 0.376931, kv6 0.115611 / 91.07% / 0.377234
(+1.19% KLD, at the rounding-flip floor), kv4 0.123894 / 90.53% / 0.387879 (+8.44%). kv6 keeps 77% of the cache's bytes
against kv8 (6.5 against 8.5 bits per element) for a cost the floor can hide.

<a id="gdn-f32-teacher"></a>
## The f32 GDN teacher (gate and state)

The 16x512 teachers above round the GatedDeltaNet gate and state to bf16 like every student, so their KLD cannot price that
rounding. This teacher stores both f32. Numbers on it are read only against each other.

- Teacher `qwen38-flash-next-bf16-gdnf32-v1` (under `kld-teacher/`): the BF16 checkpoint streamed (`--ssd-budget-gb 94`),
  `--kv-quant off`, the f32 state and f32 gate (the default; `--qwen-gdn fp32` then, `--qwen-gdn 32` now), `--no-template`, `--ctx-size 20480`, `--top-k 10`, no MTP,
  `MLX_ENABLE_TF32=0`, binary built at `0a50c13c` (the f32-gate change, mtime 2026-10-12 10:43:43, sha256 `c19e5a35...`).
- Three views of it, each a normal fixture: `raw` (the 16 wikitext prompts of the Flash-Next 16x512 raw fixture, 512 greedy
  rows each), `long` (teacher-forced `continuation_ids`, [quality-kld](quality-kld.md#the-tool)) and `code2`.
- `long` holds chat-A (64-id prefix, 16,384 forced rows of one 16.7K-token chat history), chat and `de.rs` at prefill depths
  2048 / 8192 / 16384 with 256 forced rows each (one-forward prefill, no chunk boundary). `code2` holds the same three
  depths of held-out Zig (`gdn_decode.zig` + `qwen4_forward.zig` at `314573ed`) in a second capture with the same settings.
- Held-out text: 0 13-gram hits against `tune-corpus-v2` (what Sushi-2.6bpw was tuned on); chat and `de.rs` are in other calib
  corpora, `code2` in none.
- The raw prompts are the old fixture's text; ids are identical except `wikitext2-test-07-hed-pe` (sushi's tokenizer, 2 of
  322 ids differ). Every arm scores the sushi ids; the old-teacher compare keeps the old ones.
- Arm a on the OLD raw teacher reproduces 0.137095 / 89.23% / 0.455440 of [Release 1.2.2 check](#flash-next-16x512-first-eos-7186-positions-kv8) exactly.
- The new teacher's strict NLL on the raw prompts is 0.2584 against 0.2972 for the old one, and its greedy continuations
  diverge within 40 tokens on every prompt (the old teacher came from mlx-serve): raw numbers are not comparable across teachers.

Arms (student Sushi-2.6bpw, kv8, no MTP, `--ctx-size` 8192 raw / 20480 long, M5 Max 128 GB, `taskpolicy -a`, GPU lock per compare,
2026-10-12): **a** = main `60272184` (bf16 gate, bf16 state; mtime 10:19:17, sha256 `3a03a4f5...`), **b** = the f32-gate change, default flags
(f32 gate, bf16 state), **c** = the same with `--qwen-gdn fp32` (f32 gate and state); c is now the default (`--qwen-gdn 32`) and b is
`--qwen-gdn 16`. Its kernels did not change after the binary above.

### Raw 16x512, first-EOS (7539 positions)

| arm | KLD | top-1 | NLL | paired difference of KLD (95% CI) |
|---|---|---|---|---|
| a | 0.117092 | 91.06% | 0.380222 | |
| b | 0.114813 | 91.01% | 0.378234 | b - a -0.002279 [-0.004711, -0.000048] (-1.95%) |
| c | 0.114252 | 90.79% | 0.376931 | c - b -0.000561 [-0.002512, +0.001482] (-0.49%); c - a -0.002840 [-0.005625, -0.000259] (-2.43%) |

- The f32 gate is a real KLD gain (CI excludes 0, above the 1% rounding floor); the f32 state on top of it is inside the noise.
- The interval is a block bootstrap over 128-row blocks that never cross a prompt (adjacent rows share state, so rows are not independent).

### Long context (256-row cells, one text each)

KLD per arm and the paired difference against b (95% block-bootstrap CI). Held-out Zig, teacher NLL 0.154 / 0.875 / 0.525:

| `code2` depth | a | b | c | a - b | c - b |
|---|---|---|---|---|---|
| 2048 | 0.026049 | 0.025315 | 0.026826 | +0.000734 [-0.005074, +0.007366] | +0.001511 [-0.000955, +0.004361] |
| 8192 | 0.079187 | 0.064072 | 0.063399 | +0.015115 [+0.000823, +0.033306] | -0.000674 [-0.008429, +0.007366] |
| 16384 | 0.071620 | 0.057347 | 0.054681 | +0.014273 [-0.005847, +0.042148] | -0.002666 [-0.009996, +0.004110] |
| all 768 rows | 0.058952 | 0.048912 | 0.048302 | | |

Chat history, B cells (teacher NLL 1.019 / 2.690 / 2.003):

| chat depth | a | b | c | a - b | c - b |
|---|---|---|---|---|---|
| 2048 | 0.017923 | 0.018857 | 0.017857 | -0.000934 [-0.003690, +0.001340] | -0.001000 [-0.002891, +0.000310] |
| 8192 | 0.069275 | 0.068088 | 0.080276 | +0.001187 [-0.012044, +0.016054] | +0.012188 [+0.004023, +0.020366] |
| 16384 | 0.253566 | 0.214445 | 0.199107 | +0.039121 [+0.018751, +0.062827] | -0.015338 [-0.031926, -0.001425] |

Chat-A (16,384 decode-path rows, teacher NLL 1.619), binned by context position:

| context | rows | a | b | c | a - b | c - b |
|---|---|---|---|---|---|---|
| 0-512 | 448 | 0.056277 | 0.054888 | 0.043858 | +0.001389 [-0.013712, +0.017939] | -0.011029 [-0.025070, +0.004600] |
| 512-2K | 1536 | 0.049720 | 0.047626 | 0.039204 | +0.002094 [-0.004878, +0.008797] | -0.008422 [-0.018136, +0.000034] |
| 2K-8K | 6144 | 0.056366 | 0.065772 | 0.061676 | -0.009406 [-0.018055, -0.001747] | -0.004096 [-0.014804, +0.005432] |
| 8K-end | 8256 | 0.095773 | 0.086385 | 0.098596 | +0.009388 [-0.006150, +0.026451] | +0.012211 [-0.000600, +0.027200] |

Memorized control (`serde_json/src/de.rs`, teacher NLL 0.058 / 0.0038 / 0.0014 at depth 2048 / 8192 / 16384): KLD a / b / c is
0.007142 / 0.008345 / 0.009191, 0.004812 / 0.004522 / 0.005065 and 0.001035 / 0.001249 / 0.001082; every paired CI contains 0.

- On unseen code the f32 gate cuts KLD about 17% at depth (code2 all rows, a to b); the f32 state on top of it adds about 1%, inside the noise.
- Chat-A and the chat B cells have mixed signs (a is better than b at 2K-8K, c is worse than b at 8192 and 8K-end): one text of
  highly autocorrelated rows, and the block CI is optimistic. Read no position trend from them.
- Fixture B prefills in one forward, so it carries no chunk-boundary state rounding and under-measures the state benefit.
- A memorized text (NLL near 0.001) has no KLD to resolve: keep any long-context text only if the teacher's NLL on it is above ~0.1.
