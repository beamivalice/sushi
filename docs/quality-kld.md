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
- A Qwen teacher's GDN state is bf16 like every student's, so KLD cannot see that rounding; a teacher captured under
  `kld capture --qwen-gdn fp32` measures it. GLM is not that case: its recurrent state is KDA, f32 by construction, and
  a native GLM capture refuses the flag.
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

