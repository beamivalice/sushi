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
