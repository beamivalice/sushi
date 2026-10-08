# Quality: KLD against the teacher

How a pack's quality is measured: the `kld` subcommand, the teacher fixtures, the one reading the owner uses, the
rule that the teacher path is lossless, and the recorded KLD of every served pack and of the comparison packs on the
README chart. Every new KLD of a served pack lands here with its binary commit and fixture; readings of research
packs live in the private repo.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [perf-baselines](perf-baselines.md),
[pack-format](pack-format.md#quality-bar), [engine-exl3-experts](engine-exl3-experts.md#parity-bars),
[arch-mimo-v2](arch-mimo-v2.md).

## The tool

- `src/kld.zig`: `sushi kld capture|compare`. `capture` writes a teacher fixture from the bf16 path; `compare`
  scores a pack against it. KLD is scored to the teacher's first end-of-turn token.
- `compare` prints two lines: "to-first-EOS" and all-positions. **Quote the first-EOS number** (the owner's tables);
  give the all-positions number beside it when comparing with older records.
- `kld` takes the SERVED weight loader (`model.loadWeightsForConfig`); a second loader once bound a MiMo pack's raw FP8
  QKV and made every pack score the same.
- Teacher captures run the KV cache dense (`--kv-quant off`); students are scored at kv8 unless the row says so.
- `SUSHI_HIDDEN_OUT` stores bf16 block boundaries at residual-stream width: `hidden_size` for MiMo, `hc_count * hidden_size` for Qwen4 and GLM (`[tokens, 4, hidden]`), including boundary zero. GLM teacher and served-pack capture retain every native HC stream. The served path appends each chunk's native boundaries and commits token IDs only after every boundary has the complete prompt.
- Spool mode (`SUSHI_HIDDEN_SPOOL_BYTES=<budget>`, `SUSHI_HIDDEN_OUT` = spool root, `SUSHI_HIDDEN_SPOOL_FIRST=<n>`) writes each prompt as an atomic window `w-<n>/` and waits while complete windows hold the budget: a consumer deletes the windows it has used. The live Qwen tune reads it; the chunked GLM path is refused.
- A `--prompts` jsonl line may carry `prompt_ids` (token ids, used as given, no template) instead of `prompt`.

Native GLM capture runs the native forward through `sushi kld capture` with individual BF16 expert streaming,
`--no-template --kv-quant off --ssd-budget-gb <total GiB>`. It audits indexed text headers before loading:
the trunk must be BF16/F32 and routed experts BF16. The capture selects the reference arm of every GLM fast path
(`glm5_model.reference_numerics`) and runs with `MLX_ENABLE_TF32=0`.
Other architecture capture behavior, serving defaults and compare profiles remain unchanged. The total ledger includes
the trunk, full expert union, I/O bounce slabs, per-layer LRU and at least8 GiB request reserve; allocator cache is0.
Dense prefill uses at most512 tokens per chunk and synchronous layers, with BF16 compressed MLA and FP32 KDA state.
Capture refuses any kv-quant; `kld compare --kv-quant 8` scores a GLM student with its served kv8 latent.
A GLM pack (not `bf16_individual`) captures through the generic path instead: a student self-reference of its served
forward at a BF16 latent (kv4/kv8 refused), `reference_numerics` off. Generic GLM capture and compare prefill
prompts in 2048-token chunks, so 128K-token prompts fit and both sides chunk alike.
All requested greedy full-vocabulary rows are captured through EOS, with native logits dtype recorded and exact
F32 export. Output first stays in `<out>.partial`; only a full capture publishes a completed baseline and native
identity atomically, without replacing existing output. A one-prompt study is not the standard release verdict.

<a id="layer-major"></a>
### Layer-major teacher capture (many short windows)

`--layer-major [--batch-windows W] [--pause-file P]` on the native BF16 GLM capture, `--tokens 1` only, runs W
windows through layer 0, then layer 1, and so on. Each MoE layer's 288 experts are read once per batch into the
union workspace (`glm5_stream.Stream.pin`), not once per window. Each window still runs its own `[1, t]` forward per
chunk with the same chunk width, state and kernels, so the order of work is the only difference: its fixture
(logits, tokens, baseline) and `SUSHI_HIDDEN_OUT` boundaries are byte-identical to the window-major capture. A tiny
seeded GLM checks this at several W, across chunks and across a resume (`glm5_layer_major.zig`,
`glm5_kld_capture.zig`).
- The tiny capture tests use a 1 GiB ledger and 128 MiB reserve floor; the CLI keeps its 8 GiB floor and Metal working-set check.
- Measured (binary `65a5904b`, 16 real 500-token tune windows, W=8, `--ssd-budget-gb 100`, `taskpolicy -a`, GPU lock):
  33.1 tok/s against 4.2 window-major (608.8 GB read per batch in ~48 s, ~9 s compute per window); byte-identical
  to window-major on 14 windows (131 files).
- A window carries only its HC residual between layers (32 KiB per token at 4096 hidden): a layer's KDA state,
  MLA latent and pooled index are dropped once the window has run that layer. Decode would need every layer's state per
  window, so `--tokens` above 1 is refused (`GlmLayerMajorNeedsOneRow`).
- Bill (`glm5_stream.layerMajorBudget`): the window-major capture's bill plus W x max tokens x 32 KiB, one cache
  slot per MoE layer; over budget is `GlmLayerMajorBudgetExceeded`. Without `--batch-windows`, W is the largest that
  fits, at most 32.
- Resumable: `<out>.partial/layer-major.json` binds the run (source hashes, prompt ids, chunk, label, hidden
  directory, engine binary hash); `windows.jsonl` gains a line per window only after its logits and boundary rows
  (and its `prompt_tokens.txt` / `generated_tokens.txt`) are synced. Rerunning the command re-hashes every
  committed window (inputs, logits, boundary rows, `tokens.bin`) and compares both token files with the committed
  record, truncates the hidden files to the committed tokens (zero too: an interruption inside the first batch
  resumes) and continues; a changed, truncated or missing window file or another run is refused
  (`GlmLayerMajorWindowChanged`, `GlmLayerMajorResumeMismatch`). A hidden directory that already holds rows is
  refused only by a fresh run (`GlmLayerMajorHiddenNotEmpty`), before its staging exists. While P exists the capture idles between batches
  and writes `P.ack`. A volume without room for every remaining boundary, id and logits row plus 16 GiB is refused
  before the first batch (`GlmLayerMajorDiskTooSmall`).

## Bundled four-prompt screen

`--prompts standard4` uses four prompts compiled into the CLI: deterministic Python
topological sorting, a Zig byte reader, the water cycle and navigation. The exact
two code/two prose texts are independent of the working directory; no checkout or
prompt download is needed. Existing JSONL, text-directory and captured-fixture
sources still work. `--limit` selects the first prompts in the fixed order.

Choose a lossless teacher and a student with the same tokenizer and vocabulary.
For models supported by the relevant capture/compare command (capture MTP is off by default):

```sh
MLX_ENABLE_TF32=0 sushi kld capture --model "$TEACHER_MODEL" --prompts standard4 \
  --out teacher-standard4 --tokens 512 --no-template --kv-quant off
sushi kld compare --model "$STUDENT_MODEL" --fixture teacher-standard4 \
  --json comparison.json --kv-quant off
```

If the teacher needs expert streaming, add `--ssd-budget-gb <total RAM GiB>`;
choose a budget that fits the machine. Native BF16 GLM capture requires streaming;
a GLM pack is scored with `sushi kld compare --model <pack> --fixture <teacher>`, resident or with
`--ssd-budget-gb` (streamed experts score the same logits).

The saved fixture contains token IDs and all 512 full-vocabulary rows per prompt,
including rows after EOS. Compare reports all positions and the first-EOS-inclusive
subset. This four-prompt screen is not the sixteen-prompt release reading.

## The standard reading

- **16 prompts x 512 tokens, scored to the first EOS** (Flash-Next's 16 wikitext prompts, raw text, no template);
  GLM reads its 4x512 native teacher with the code 2x512 apart (below). 60x64 is a short-context screen only, never a verdict.
- Differences under ~1% on ONE pack are inside the ROUNDING-FLIP floor (measured 2026-09-24 on the MiMo MCG pack:
  flipping 0.07-0.13% of attention outputs by one bf16 ulp, no precision loss, moved 16x512 KLD -0.5% .. +0.55%). A
  kernel or storage change that flips bits reads as a KLD change of that size with no quality meaning; each flip
  pattern is deterministic. Compare arms on the same binary; say the floor beside the number.
- **NAX vs SIMD is accepted hardware noise (owner policy).** A NAX (tensor-op) arm and its SIMD twin accumulate in a
  different order, so they score a slightly different KLD. The engine takes the faster arm knowingly: such a delta is
  recorded, never treated as a regression or a reason to hold a NAX kernel back, and never "fixed" toward SIMD.
  A NAX arm still has to pass its fp32 parity test; this policy covers only the model-level KLD difference.

## Teacher fixtures

| fixture | model | notes |
|---|---|---|
| Flash-Next 16x512 raw | Flash-Next | the bf16 checkpoint, bf16 stream, raw text; the standard |
| Flash-Next 16x512 raw, f32 stream | Flash-Next | the same with an f32 residual stream |
| Flash-Next 60x64 | Flash-Next | the 60x64 screen |
| MiMo 16x512 raw | MiMo | the MOPD checkpoint as stored (FP8 trunk: bf16 weights in prefill, FP8 code x f32 block scale in decode), dense KV; captured 2026-09-30 by 83dc9b6c; mean strict NLL 0.2664; 652 s capture |

Commands (MiMo; Flash-Next drops `--ssd-budget-gb` when the source fits):

```sh
sushi kld capture --model <original checkpoint> \
  --prompts <the Flash-Next 16x512 raw fixture> --out <teacher dir> \
  --tokens 512 --top-k 10 --label <label> --no-template --kv-quant off --ctx-size 8192 --ssd-budget-gb 94
sushi kld compare --model <pack> --fixture <teacher dir> --label <label> \
  --kv-quant 8 --tokens 512 --top-k 10 --ctx-size 8192 --json <out>.json
```

Both are heavy GPU jobs: take the lock per run (CLAUDE.md, Team process).

<a id="teacher-path"></a>
## The teacher path is lossless

- The reference forward may add NO quantization of its own. MiMo has no bf16 release (FP8 e4m3 trunk, MXFP4
  experts): use the FP8 as FP8 or dequantize it to bf16/f16, never requantize to affine-8; capture with `--kv-quant`
  off. Before any capture, read the loader path the original checkpoint takes and list every dtype change; each must
  be exact.
- Say "original checkpoint through path X", never "bf16 teacher", unless the checkpoint is bf16.
- A biased reference is refused regardless of size: a MiMo teacher captured through an affine-8 trunk and a kv8 cache
  differed from the lossless one by 0.0076 nats (the whole engine-to-engine gap mlx-lm had measured). A pack's
  stored-affine trunk (served packs only) leaves the teacher untouched.
- **GLM teacher arms**: a NAX GPU captures through the arms served packs take, BF16 weights, BF16 latent and FP32 KDA
  state as stored; any other GPU keeps the reference arms. The capture prints `[glm] NAX arms on|off` and
  `identity.json` records `nax_arms`, `reference_numerics` and each arm's dispatch count.
- A teacher's continuations depend on its arms: compare packs only against the fixture they were scored on, and recapture
  the teacher before comparing rows scored on another route.
- `SUSHI_NGRAM_BF16_DIR=<hf checkpoint>` serves a Flash-Next pack with the original bf16 n-gram table to isolate
  the PLE table's cost.

## Lossy expert pick

`--expert-pick-tolerance` on Sushi-2bpw, `--ssd-budget-gb 20` (251 slots/layer), kv8, 16 prompts x 128 tokens,
teacher = the same pack with exact routing (`kld compare --expert-pick-tolerance n`).

| tolerance | KLD | top-1 | NLL | KLD to first EOS | cache hit |
|---|---|---|---|---|---|
| 0 | 0 | 100% | 0.4004 | 0 | 81.6% |
| 0.2 | 0.0216 | 94.5% | 0.4174 | 0.0298 | 83.2% |
| 0.3 | 0.0264 | 93.7% | 0.4272 | 0.0355 | 83.8% |

The pack's own KLD against the bf16 teacher is 0.208, so 0.2 adds about a tenth of it. The GPU-side pick reads the
same KLD to the ninth digit (0.021600648): it picks exactly what the host picks.

Repetition: 8 prompts x 600 tokens at temperature 0 and 1, tolerance 0 / 0.2 / 0.3: mean distinct 4-grams 0.997-0.999
in every arm (worst run 0.983), no loop-stop cut in any of the 48 runs.

<a id="lossy-expert-pick-mimo"></a>
MiMo (sigmoid router: the pick compares sigmoid probabilities), the MOPD checkpoint (MXFP4 experts) streamed at
`--ssd-budget-gb 60` (81 slots/layer), kv8, 16x512 raw, teacher = the same load with exact routing captured on this
binary (built at cf23043d, the landed change's pick code; strict NLL 0.2609, cache hit 84.6%):

| tolerance | KLD (to first EOS) | top-1 | NLL | cache hit | ids swapped |
|---|---|---|---|---|---|
| 0.2 | 0.00958 | 97.2% | 0.2725 | 92.9% | 7.7% |

Decode at a 4k prompt (llmprobe 0.6.12, `--no-mtp`, one boot each): 5.8 tok/s exact, 10.5 at 0.2
([perf-baselines](perf-baselines.md#mimo-stream-pick)).

## Cross-engine check

mlx-lm's MiMo support (upstream PR 1219, router patched to f32), streamed one layer at a time, against our MiMo
teacher: the original checkpoint scores 0.0077 nats / 95.8% top-1 (the engine-to-engine floor; every flip sits at a
teacher top-2 gap ≤ 0.5 nats, flat across the context). The teacher and the tool are validated by an implementation
that shares no code with ours.

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

<a id="mimo"></a>
## MiMo (16x512, first EOS, student kv8)

| pack | KLD | top-1 | positions | binary |
|---|---|---|---|---|
| MiMo-V2.6-Flash-Sushi-2.3bpw | 0.0860 | 91.95% | 8067 | 83dc9b6c (v1.1.0 gate) |

Against the 2026-09-30 MOPD teacher. Readings against an earlier teacher capture generate different continuations and
are not comparable.

The FP8-native teacher against the bf16-rounded teacher: 0.0034 nats.

The FP8 trunk's matrix-unit tile ([engine-kernels](engine-kernels.md#prefill-kernels)) serves 9-128 rows, so it never
touches these 295-363-token prompt forwards. Forced onto every prompt forward as a stress arm (binary `01319267`, same
settings, same binary as the off arm): KLD 0.086674 / top-1 91.78% (7404) / NLL 0.360444 against 0.086035 / 91.95%
(7418) / 0.357721 off, +0.74% KLD, inside the rounding-flip floor (the same bf16 operands summed in another order).


## GLM-5.3-Flash: native BF16 teacher, 4x512 (2026-10-04)

GLM's release reading is this 4x512 screen, with the two code prompts (2x512) reported apart: a 16x512 BF16 teacher
would stream the BF16 experts and is too slow to capture. Shipped Sushi-2.4bpw (K2.25/K2.5 W14 experts, A6 g128 trunk)
against the NAX-path teacher: KLD 0.0742 / top-1 90.3%; code 0.0429, prose 0.1055. The same
screen on Sushi-2.5bpw (kv8, FP32 decode attention, first teacher): KLD 0.0716 / top-1 90.3%; code 0.0324 / 95.8%,
prose 0.1107 / 84.9%.

Teacher (every row of the table below was scored against the reference-arm capture, whose `identity.json` has no `nax_arms`): the BF16 source checkpoint through the native forward, `MLX_ENABLE_TF32=0 sushi kld capture --prompts
standard4 --tokens 512 --no-template --kv-quant off --ssd-budget-gb 100` (streamed BF16 experts, dense prefill in
chunks of at most 512, synchronous layers, BF16 MLA cache, FP32 KDA state). Native prompt lengths 242/261/190/183; no
EOS in the 2,048 rows, so first-EOS and all-positions readings are the same. Students carry MCG EXL3
experts at W12 (K2.25 unless the row says K2.5; the shipped row is W14), BF16 MLA and FP32 KDA state, one resident target, MTP and DFlash2 off.

| Pack | Trunk | KLD | Top-1 | NLL | Code / prose KLD | Peak active | Commit, settings |
|---|---|---:|---:|---:|---:|---:|---|
| Sushi-2.3bpw | A6 g128 | 0.092950 | 88.96% | 0.4323 | 0.0468 / 0.1391 | 94.08 GB | `3ed533d7`+WIP, TF32 on, fast target kernels |
| A8 experiment (not shipped) | A8 g128 | 0.091519 | 88.87% | 0.4318 | 0.0441 / 0.1389 | 96.30 GB | same binary |
| Sushi-2.45bpw | raw FP8 block-128 | 0.091266 | 88.62% | 0.4304 | 0.0439 / 0.1386 | 102.51 GB | `56017748`, TF32 off, `sushi kld compare` |
| Sushi-2.5bpw (K2.5) | A6 g128 | 0.072071 | 89.94% | 0.4075 | 0.0314 / 0.1127 | 103.59 GB | `00668fcb`, `sushi kld compare` defaults; 2.3bpw reproduces its row bit for bit there |

Against the NAX-path teacher (`glm5_model.enterTeacher()`, binary `177c526f`, TF32 off; decode attention is the only
arm that changes this capture), students at BF16 latent, same 4x512 prompts:

| Pack | Experts | Trunk | KLD | Top-1 | NLL | Code / prose KLD | Peak active | Commit, settings |
|---|---|---|---:|---:|---:|---:|---:|---|
| Sushi-2.3bpw | K2.25 L3-45, W14 | A6 g128 | 0.078609 | 89.94% | 0.4147 | 0.0460 / 0.1112 | 94.07 GB | `177c526f`, `sushi kld compare --kv-quant 16` |
| **Sushi-2.4bpw (shipped)** | K2.25 L3-36 + MTP, K2.5 L37-44, W14 | A6 g128 | 0.074213 | 90.33% | 0.4160 | 0.0429 / 0.1055 | 95.88 GB | same |
| Sushi-2.5bpw | K2.5 L3-45, W14 | A6 g128 | 0.057151 | 91.06% | 0.3920 | 0.0316 / 0.0827 | 103.58 GB | same |

The rows above were scored against the first teacher, so they do not rank against this one.

K2.5 experts cut KLD 22.5% from K2.25 on the same A6 trunk (byte-identical trunk tensors). The three trunks sit within 2% of each other under two numerical profiles; this four-prompt screen does not rank them
and is not the 16x512 release reading. Code scores about 3x lower than prose on every pack.

GLM kv8 latent (`--kv-quant 8`, Sushi-2.5bpw, binary `694e36a3`, `taskpolicy -a`, GPU lock):
- Same 4x512 teacher: KLD 0.071762 / top-1 89.70% against the BF16 latent's 0.072071 / 89.94% (-0.43%, inside the
  noise floor); peak active unchanged at 103.59 GB.
- Long context, against the pack's own BF16-latent reference (2 prompts of 59k and 66k tokens x 2048 teacher-forced
  tokens): KLD to first EOS 0.00715, top-1 96.8% over 496 positions; per-256-position means stay at 0.0006-0.0094 on
  the prose prompt. Past EOS the code prompt's continuation degenerates and its KLD climbs to 0.18, outside the
  scored window.
- Greedy free-run at those prompts diverges early (token 16 and 118) at near ties; the answers reword the same facts.
- FP32 composite B1/B3 decode attention against the fused NAX SDPA, one binary (`cc56be2b` diag arms): KLD 0.071569 /
  top-1 90.33% against 0.071762 / 89.70% (−0.27%, inside the noise floor); with MLX's TF32 GEMMs 0.071602 / 89.84%.

