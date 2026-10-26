# Quality: KLD against the teacher

How a pack's quality is measured: the `kld` subcommand, the teacher fixtures, the one reading the owner uses, the
rule that the teacher path is lossless, and the recorded KLD of every served pack and of the comparison packs on the
README chart. Every new KLD of a served pack lands here with its binary commit and fixture; readings of research
packs live in the private repo.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [perf-baselines](perf-baselines.md),
[pack-format](pack-format.md#quality-bar), [engine-exl3-experts](engine-exl3-experts.md#parity-bars),
[mimo2-arch](mimo2-arch.md).

## The tool

- `src/kld.zig`: `sushi kld capture|compare`. `capture` writes a teacher fixture from the bf16 path; `compare`
  scores a pack against it. KLD is scored to the teacher's first end-of-turn token.
- `compare` prints two lines: "to-first-EOS" and all-positions. **Quote the first-EOS number** (the owner's tables);
  give the all-positions number beside it when comparing with older records.
- `kld` takes the SERVED weight loader (`model.loadWeightsForConfig`); a second loader once bound a MiMo pack's raw FP8
  QKV and made every pack score the same.
- Teacher captures run the KV cache dense (`--kv-quant off`); students are scored at kv8 unless the row says so.
- A Qwen teacher's GDN state and decay gate are f32, like a default student's. Teachers captured before the f32 change hold a
  bf16 gate and state; `qwen38-flash-next-bf16-gdnf32-v1` is the one that matches the served numerics ([qwen4-kld](qwen4-kld.md#gdn-f32-teacher)). A student scored at `--qwen-gdn 16` (`kld compare`/`capture`) pays the bf16 state's rounding against it. GLM is not that case: its recurrent state is KDA, f32 by construction, and a native GLM capture refuses the flag.
- `SUSHI_HIDDEN_OUT` stores bf16 block boundaries at residual-stream width: `hidden_size` for MiMo, `hc_count * hidden_size` for Qwen4 and GLM (`[tokens, 4, hidden]`), including boundary zero. GLM teacher and served-pack capture retain every native HC stream. The served path appends each chunk's native boundaries and commits token IDs only after every boundary has the complete prompt.
- Spool mode (`SUSHI_HIDDEN_SPOOL_BYTES=<budget>`, `SUSHI_HIDDEN_OUT` = spool root, `SUSHI_HIDDEN_SPOOL_FIRST=<n>`) writes each prompt as an atomic window `w-<n>/` and waits while complete windows hold the budget: a consumer deletes the windows it has used. The live Qwen tune reads it; the chunked GLM path is refused.
- A `--prompts` jsonl line may carry `prompt_ids` (token ids, used as given, no template) instead of `prompt`.
- With `prompt_ids` a line may also carry `continuation_ids`: capture feeds and stores exactly those tokens (one full-vocab
  row each, `--tokens` ignored) instead of the greedy continuation, so `compare` scores real text at any depth and
  `per_position_kld` of its `--json` bins by context position. The to-first-EOS line of such a fixture is meaningless
  (the text holds EOS-like tokens): read `mean_kld` or the per-position rows.

Native GLM capture runs the native forward through `sushi kld capture` with individual BF16 expert streaming,
`--no-template --kv-quant off --ssd-budget-gb <total GiB>`. It audits indexed text headers before loading:
the trunk must be BF16/F32 and routed experts BF16. The capture selects the reference arm of every GLM fast path
(`glm5_model.reference_numerics`) and runs with `MLX_ENABLE_TF32=0`.
Other architecture capture behavior, serving defaults and compare profiles remain unchanged. The total ledger includes
the trunk, full expert union, I/O bounce slabs, per-layer LRU and at least8 GiB request reserve; allocator cache is0.
Dense prefill uses at most512 tokens per chunk and synchronous layers, with BF16 compressed MLA and FP32 KDA state.
Capture refuses any kv-quant; `kld compare --kv-quant 8` or `6` scores a GLM student with its served kv8 or kv6 latent.
A GLM pack (not `bf16_individual`) captures through the generic path instead: a student self-reference of its served
forward at a BF16 latent (kv4/kv6/kv8 refused), `reference_numerics` off. Generic GLM capture and compare prefill
prompts in 2048-token chunks, so 128K-token prompts fit and both sides chunk alike.
Every architecture prefills a prompt in its serving start width (`prefillStartWidth`: Flash-Next 4096, MiMo and GLM 2048),
in capture and compare alike and on any box; a prompt within one chunk is one forward, so short fixtures keep their numbers.
A whole-prompt forward peaked at 57 GB at 32K and 77 GB at 64K on Flash-Next 2bpw, and a 128K teacher died on Metal OOM.
A hidden capture (`SUSHI_HIDDEN_OUT`) on MiMo or Flash-Next still takes its prompt whole.
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
- `SUSHI_IMATRIX_OUT=<abs>.safetensors` rides the same pass and writes the experts' imatrix
  ([engine-expert-streaming](engine-expert-streaming.md#imatrix)); with `SUSHI_HIDDEN_OUT` one pass yields the tune's
  boundaries and the converter's calibration.
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

<a id="dsv41-kld"></a>
### DeepSeek-V4.1

`kld capture` and `compare` run `deepseek_v41` through the generic path, streamed (`--ssd-budget-gb` is required; the
load names it before touching the GPU). The state is the activation dtype and the model keeps its own, so a flag that
would silently do nothing is refused by name at load (`refuseForDsv41`): `--kv-quant` (`Dsv41KvQuantUnsupported`),
`--qwen-gdn 16` (`Dsv41HasNoQwenGdnState`), `--hidden-out` (`Dsv41HiddenCaptureUnsupported`) and `SUSHI_IMATRIX_OUT`
(`Dsv41ImatrixCaptureUnsupported`). The teacher is exact routing: `capture` refuses `--drop-expert` and
`--expert-pick-tolerance`; `compare --drop-expert n` scores a pruned student against it
([dsv41-arch](dsv41-arch.md#drop-expert)). `compare --expert-pick-tolerance n` acts on the one-row steps that score the
generated tokens (the prompt prefill is exact) and only where an expert can miss: a budget that holds every expert loads
resident, logs that the flag has no effect, and scores the exact KLD; a real streamed load logs
`[expert-swap] engaged` and counts the swaps ([engine-expert-streaming](engine-expert-streaming.md)).
`SUSHI_DSV41_COMPOSED=1` runs the op chains in place of the fused kernels, the
teacher arm of a fused-kernel A/B.
A tiny-model test (`DSV41_TINY`) captures a teacher and compares it in both arms.
`compare` scores prefill chunks on the GEMM and the generated tokens on one-row steps, so it never runs the NAX 2 to 8 row
arm: a change to that arm's sum order leaves the KLD identical to the last digit (0.186614243 on the STQ 1.75bpw pack for
main and for the reordered trunk rows) and is gated by the f32-truth bars and the width-invariance byte test instead.

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

<a id="eight-category"></a>
## Eight-category reading (Flash-Next)

The standard reading is prose only. The 8-category fixture keeps its size, 16 prompts x 512 tokens, and spends it on
two prompts per category: code, math, reasoning, design, wikitext, multilingual, story, stem.

- A category's mean comes from 2 prompts, at most 1024 positions: a coarse reading. Rank packs on the `all` row, and
  read a category for where the loss sits, not to rank packs whose difference is small.
- Every prompt is `prompt_ids`, so one capture holds three formats and compare needs no flag:
  - math, reasoning and stem: chat template with thinking on, at the effort a request that names none gets (low).
    The chain of thought is the content.
  - code, design, multilingual and story: chat template with thinking off, the served default. The 512 rows are the
    code, the page or brief, the story, or the non-English text itself.
  - multilingual covers 8 languages in 8 scripts: one original passage translated into Thai, Chinese, Japanese and
    Arabic, and an original birthday song (not the copyrighted lyrics) in Hindi, Russian, Spanish and Korean.
  - wikitext: two WikiText-2 test articles outside the standard 16, raw, in the standard layout.
- The non-wikitext prompts are original text written for the fixture. None shares a word 13-gram, a char 32-gram or a
  token 13-gram with any calibration or tune corpus, or with the source pools they were drawn from, and none shares a
  word 8-gram or a non-Latin char 16-gram either.
- The prompt id's prefix before the first `-` is its category. `tests/kld_by_category.py [--csv-dir DIR]
  <compare.json>...` prints one table per pack and a model x category KLD matrix (`--csv-dir`: `matrix.csv` and
  `matrix-top1.csv`), position-weighted to the first EOS, so its `all` column is the compare's own number.
- The teacher is the bf16 checkpoint with an f32 GDN state and gate ([qwen4-kld](qwen4-kld.md#gdn-f32-teacher)). Score
  students with `--qwen-gdn 32` and bf16 KV. These numbers are not comparable with any other fixture.

## Teacher fixtures

| fixture | model | notes |
|---|---|---|
| Flash-Next 16x512 raw | Flash-Next | the bf16 checkpoint, bf16 stream, raw text; the standard |
| Flash-Next 8-category 16x512 | Flash-Next | the bf16 checkpoint, f32 GDN state and gate, 2 prompts per category ([above](#eight-category)) |
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
([perf-baselines](mimo2-perf.md#mimo-stream-pick)).

## Cross-engine check

mlx-lm's MiMo support (upstream PR 1219, router patched to f32), streamed one layer at a time, against our MiMo
teacher: the original checkpoint scores 0.0077 nats / 95.8% top-1 (the engine-to-engine floor; every flip sits at a
teacher top-2 gap ≤ 0.5 nats, flat across the context). The teacher and the tool are validated by an implementation
that shares no code with ours.

## Flash-Next (16x512, first EOS, 7186 positions, kv8)

Moved to [qwen4-kld](qwen4-kld.md).

## MiMo (16x512, first EOS, student kv8)

Moved to [mimo2-kld](mimo2-kld.md).

## GLM-5.3-Flash: native BF16 teacher, 4x512 (2026-10-04)

Moved to [glm5-kld](glm5-kld.md).

