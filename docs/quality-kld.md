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
- `src/kld_fixture.zig` holds what a fixture is and links no MLX (host tier): the prompt sources, `baseline.json`, the
  per-prompt files, the teacher's logits rows, `scoreRow` and the first-EOS cut. A plugin's parity harness reads and
  scores a fixture through it. What an arch adds to the tools lives in its folder (`<arch>/*_kld*.zig`, the `kld`
  declaration of its `arch.zig`) and is reached through `src/arch/offline.zig`: its hidden-capture width, launch
  refusals, own capture path, post-load setup and the imatrix names of its source checkpoint.
- A capture stamps the RoPE it ran under (`baseline.json` `rope`: method `native|yarn`, factor, original, window, and
  `override`: `none`, `yarn_factor`, `config-overrides` or both) and prints `[kld] RoPE capture: ...`; `compare` prints
  the teacher's and the student's and writes `teacher_rope` and `student_rope` to `--json`. A YaRN teacher is allowed;
  it is never unseen. A fixture captured before the stamp reads "not recorded".
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
`--no-template --kv-quant off`, plus `--ssd-budget-gb` only because GLM has no RAM default to stream its BF16 experts under. It audits indexed text headers before loading:
the trunk must be BF16/F32 and routed experts BF16. The capture selects the reference arm of every GLM fast path
(`glm5_model.reference_numerics`) and runs with `MLX_ENABLE_TF32=0`.
Other architecture capture behavior, serving defaults and compare profiles remain unchanged. The total ledger includes
the trunk, full expert union, I/O bounce slabs, per-layer LRU and at least8 GiB request reserve; allocator cache is0.
Dense prefill uses at most512 tokens per chunk and synchronous layers, with BF16 compressed MLA and FP32 KDA state.
Capture refuses any kv-quant; `kld compare --kv-quant 8` or `6` scores a GLM student with its served kv8 or kv6 latent.
`sushi kld` loads through `model_load.launchRefusal`, the serving load's one question: an unserved arch, a launch flag or `--kv-quant` the arch cannot act on, and a context above the window or `max_kv_tokens` are refused by name before a weight is read.
A GLM pack (not `bf16_individual`) captures through the generic path instead: a student self-reference of its served
forward at a BF16 latent (kv4/kv6/kv8 refused), `reference_numerics` off. Generic GLM capture and compare prefill
prompts in 2048-token chunks, so 128K-token prompts fit and both sides chunk alike.
Every architecture prefills a prompt in its serving start width (`prefillStartWidth`: Flash-Next 4096, MiMo and GLM 2048, DeepSeek 8192),
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
- Measured (binary `65a5904b`, 16 real 500-token tune windows, W=8, `taskpolicy -a`, GPU lock):
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

`kld capture` and `compare` run `deepseek_v41` through the generic path on serve's RAM default (`model_load.withRamDefault`): fully resident on a 192 or 256 GB Mac, and a box outside the tiers is refused by name before touching the GPU. The state is the activation dtype and the model keeps its own, so a flag that
would silently do nothing is refused by name at load (`kld_capture.refuse`): `--kv-quant` (`Dsv41KvQuantUnsupported`),
`--qwen-gdn 16` (`Dsv41HasNoQwenGdnState`), `--hidden-out` (`Dsv41HiddenCaptureUnsupported`) and `SUSHI_IMATRIX_OUT`
(`Dsv41ImatrixCaptureUnsupported`; the layer-major capture below takes it). The teacher is exact routing: `capture` refuses `--drop-expert`,
`--expert-pick-tolerance` and `--expert-cache-bias` (a model setting `expert_cache_bias` is ignored by a capture, with a `[kld]` line); `compare --drop-expert n` scores a pruned student against it
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
`--layer-major --tokens 1` instead runs the calibration capture below.

<a id="hy4-kld"></a>
### Hy4-Flash

`kld capture` and `compare` run `hy_v4_flash` through the generic path on serve's RAM default, as DeepSeek's, with
exact routing. The model takes the latent width from the cache config at a request's step 0, so `--kv-quant off|16`
(BF16, the teacher's) and `--kv-quant 8` (the engine's affine kv8, group 64) both capture and compare: the 1y reading
is a kv8 compare against a BF16-KV capture. Refused by name at load (`hy4_kld_capture.refuse`): another KV width
(`Hy4KvQuantUnsupported`), `--qwen-gdn 16`, `--hidden-out` / `SUSHI_HIDDEN_OUT`, `SUSHI_IMATRIX_OUT`,
`--expert-pick-tolerance` and `--drop-expert` (the routing is the host's exact top-k). `capture` refuses
`--expert-cache-bias` (the teacher is exact routing); `compare --expert-cache-bias n` takes it on the one-row steps of a
streamed load ([engine-expert-streaming](engine-expert-streaming.md#cache-bias)). Capture
and compare run f32 products in f32 (`hy4_kld_capture.beforeLoad`, the offline seam's pre-load op): on a NAX GPU MLX and
the engine take them on TF32 tiles unless `MLX_ENABLE_TF32=0`, which they read once per process, so an unset variable
is set to 0 before the load, another value is refused (`Hy4KldNeedsTf32Off`), and a process that already read it on is
refused (`Hy4KldTf32AlreadyRead`). The f32 products end with the load (`afterUnload`, the seam's post-load op), so a served load
after it in a process measures its region as in a fresh one. On the tiny (`HY4_BF16_TINY`) a captured row is the oracle's within 2e-3, a compare
of the same load reads KLD 0 and top-1 100%, and a kv8 compare reads 0.049 (top-1 71%): the tiny's random routing
magnifies any rounding (TF32 alone puts its last prompt row 33% of the peak off the oracle), so its number says nothing
about the release's.

<a id="hy4-capture"></a>
### Hy4-Flash calibration capture (layer-major)

`--layer-major --tokens 1 --no-template` on the BF16 release (`hy4_layer_major.zig`) runs W windows
through layer 0, then layer 1, and so on, through `hy4_model`'s layer entry (`embedStreams`, `engramRowsAfter`,
`passLayer`, `releaseLayer`, `readoutStreams`), and writes DeepSeek's `calib/` contract with Hy4's inputs:
- `mlp-in-LL.bf16`: what the routed experts read, the 1536-wide latent row `lat_down(ffn_in)`, MoE layers only. At the
  BF16 routed products its values are BF16's, so the file holds them whole (a value that is not is refused,
  `Hy4CaptureLatentNotBf16`); at f32 products it is `mlp-in-LL.f32`.
- `route-LL.u16`, `route-w-LL.f32`: the picks and applied weights, tokens x top-k a layer.
- `trunk-LL` / `trunk-head`: the `x x^T` sums of `attn_in`, `q_lora`, `kv_latent`, `o_in`, `ffn_in`, `lat_up_in`,
  `shared_act` (layer 0: the dense activation), `engram_in`, and `head_in` on every position's final row (the head's
  input when it serves), not only the row read out: a window's earlier chunks give theirs through `readoutStreams(.none)`
  with the tap on. The read row's logits are computed on that row alone, as without a tap, so the bytes do not move.
- `head-in.f32` (opt-in, `SUSHI_HY4_HEAD_ROWS=1`): the head's input for every token, `[tokens, 3072]` f32 little-endian in
  window order (12 KB a token). The forward makes it in f32, so the file holds it whole where DeepSeek's `head-in.bf16`
  holds a bf16 activation (a bf16 file would round the teacher); the run's manifest carries `mode: head-rows`, so a resume
  without the switch, or with it on a run begun without, is refused. `layout.json` names it under `head_rows`.
  With `SUSHI_HY4_CAPTURE_LEAN=1` as well (lean without head rows is refused, `Hy4CaptureLeanNeedsHeadRows`) the run
  writes only `tokens.bin`, `windows.jsonl`, the fixture, `layout.json` and `head-in.f32` (12 KB a token), and none of the
  rows, routes or sums or their GPU work; the manifest says `mode: head-rows-lean`, a rerun in another mode is refused, and
  `layout.json` leaves out what it did not write. The head rows and logits equal the full run's byte for byte.
- No imatrix: the encoder's Hessians come from these rows, and `SUSHI_IMATRIX_OUT` is refused
  (`Hy4ImatrixCaptureUnsupported`: its writer cannot take an engine load's arrays).

The run starts from a reset model at the BF16 latent. On the tiny (`HY4_BF16_TINY`, `MLX_ENABLE_TF32=0`) each window's
logits equal its window-major `forwardReadout` byte for byte at W of 1 and 3 and chunks of 4 and 4096, and a run
interrupted inside its second batch resumes at another W to the same logits, ids, rows and routes; the sums' bytes follow
the batch partition, their values and row counts do not. The load keeps a whole layer's union workspace beside two cache
slots a layer (`hy4_kld_capture.captureFloor`); the bill adds the widest layer's sums in flight and the one running
window's transient, and each window carries its four f32 streams and its Engram rows (49,412 B a token on the release):
W is the largest that fits, at most 32, or a `--batch-windows` past the budget is refused (`Hy4CaptureBudgetExceeded`).
The ledger, resume and disk check are the shared `kld_calib.zig`. Each MoE layer's union is pinned once a batch
(`Model.pin`), so its 512 experts are read once for all W windows.
The routed experts run over up to 16,384 rows of pieces at once (`Model.ffnBatch`, `hy4_expert_mv.zig`): `gather_mm` over
one-row operands is MLX's `gemv_gather`, a gemv per pick that re-reads its expert's matrix (at f32 products over the
bank widened to f32), so batching rows through it does not help. The kernel sorts the picks by expert and applies four
rows to every matrix tile with `gemv`'s thread layout and accumulation order, reading the BF16 bank as stored, so each
product is the byte `gather_mm` writes; attention, router, latent and tail still run per piece at their own shapes. The
per-piece run stays reachable in the tests (`per_piece_for_test`): every file of a capture, f32 and BF16 products, equals
the per-piece run's byte for byte, across a resume from one into the other. The gram sums are submitted without a wait.
Refused by name: a quantized latent,
`--hidden-out`, `--imatrix-windows`, `--tokens` other than 1, a template, a missing budget.

<a id="dsv41-capture"></a>
### DeepSeek-V4.1 calibration capture (layer-major)

`--layer-major --tokens 1 --no-template` on a `deepseek_v41` checkpoint (`kld_capture.zig`) runs W windows through
layer 0, then layer 1, and so on, with one pinned union of the layer's 384 FP4 experts per batch
(`dsv41_stream.Stream.pin`, the budget sizes W: the cache itself stays at its two-slot floor). The budget is the load's own (the setting, else serve's RAM default through `model_load.withRamDefault`) (a resident default sizes W as an explicit budget of its size would); a default that buys no window, or a box outside the tiers, is refused before the load, naming the budget one window needs. A window is one
forward of at most 2048 rows (longer ones run in 2048-row chunks, each layer over all its chunks in order); the layer step
is the model's own (`dsv41_model.passLayer`, the same code `extendSpan` loops), so the standard fixture (`prompts/NN_id`,
`baseline.json`) is the window-major capture's byte for byte. It also writes, in `<out>/calib/`, what a quantizer needs:

| file | holds |
|---|---|
| `mlp-in-LL.bf16` | the FFN input every routed and shared expert of layer LL reads, `[tokens, dim]` bf16, windows in order |
| `route-LL.u16`, `route-w-LL.f32` | the router's top-k expert ids and the weights it applied (normalized, times the routed scale), `[tokens, top_k]` |
| `trunk-LL.safetensors`, `trunk-head.safetensors` | `layers.LL.<input>`: X^T X of the input of each trunk linear, summed over every token in f32, with `<name>.rows` (I64) |
| `imatrix.safetensors` | the experts' gate/up and down second moments (`SUSHI_IMATRIX_OUT` names another file) |
| `dspark-in-LL.bf16` | the mean of the hc copies entering each DSpark target layer (config `dspark_target_layer_ids`), `[tokens, dim]` bf16: what the draft stages' main projection reads |
| `head-in.bf16` | opt-in (`SUSHI_DSV41_HEAD_ROWS=1`): the array the head multiplies for every row (final RMSNorm output after the last hyper-connection collapse), `[tokens, dim]` bf16, windows in order; `head @ row` in f32 is the teacher's logits |
| `tokens.bin`, `layout.json` | the window ids (u32; `../windows.jsonl` has the lengths) and this table |

The trunk inputs: `attn_in` (wq_a, wkv, the compressor projections, the indexer's weights projection), `q_lora` (wq_b and
the indexer's wq_b), `index_latent` (the indexer's key projection, index-key owners), `wo_a_in` (`[groups, in, in]`: one
matrix per output group of wo_a), `wo_b_in`, `ffn_in` (the shared expert's w1 and w3), `shared_act` (its w2),
`engram_in` (the Engram wkv, layers 1 and 14) and `head_in` (the head). They are what the forward feeds, with
no `act_quant` on the way ([dsv41-arch](dsv41-arch.md#gaps-against-the-reference)).
- The sums add on the GPU per layer, in f32 (MLX's f32 matmul keeps full f32 here: 1.3e-6 of peak against f64 on a 2048x5120 X^T X,
  21.6 TFLOPs), and a worker thread folds each layer's batch total into `trunk-LL.<windows>.safetensors` (read, add, write,
  rename); the finished files lose the window count. Bytes depend on the batch partition (f32 addition order); rows,
  routes, logits and the imatrix do not.
- The tiny seeded pack (`DSV41_TINY`) proves it: logits, rows, routes and tokens equal the window-major forward at W of 1
  and 3 and chunks of 4 and 64 rows, the generic `kld capture` fixture equals the layer-major one, the sums match an f64
  X^T X of the recorded inputs, the imatrix matches the dumped rows and routes, and an interruption inside a batch resumes to
  the uninterrupted bytes, also at another batch size.
- The engine writes and reads `imatrix.safetensors` itself (`src/imatrix.zig`): MLX's file I/O refuses an engine stream or map, so
  a capture on a Sushi Cooler load never goes through it.
- Measured (32 real 2048-token windows, `taskpolicy -a`, GPU lock, fans auto):
  309.5 s a batch = 212 tok/s; the forward alone 306 tok/s (213.8 s), the one-per-batch read of all 288.8 GB 27.5 s
  (0.69 s a layer), the capture's extras 68.2 s (22%: the X^T X sums, row and route writes, imatrix). 496 windows in 16
  batches take about 1.3 h. Disk at 1M tokens: rows 416 GB, DSpark rows 31 GB, sums 41 GB.
- Resume as above: `layer-major.json` binds the run, `windows.jsonl` commits a window after every file of its batch is synced,
  a rerun truncates the row, route and token files to the committed tokens and re-adds the in-flight batch; the previous
  sums version stays until the batch commits. The disk check counts rows, routes, two versions of the sums and the logits plus 16 GiB.
- Teacher rows: `SUSHI_DSV41_HEAD_ROWS=1` adds `head-in.bf16` (the forward's activation dtype is bf16 and the head casts it to f32,
  so the file loses nothing); with `SUSHI_DSV41_CAPTURE_LEAN=1` as well the run writes only `tokens.bin`, `windows.jsonl`, the
  fixture, `layout.json` and `head-in.bf16`: no rows, routes, sums, imatrix or DSpark rows, and none of their GPU work. Lean
  without head rows is refused (`Dsv41CaptureLeanNeedsHeadRows`); `layer-major.json` binds the mode, so a rerun in another mode
  is refused. The tiny pack proves the rows equal the forward's head input, the last row of each window through the head
  reproduces the fixture logits, a lean run's rows and logits equal the full run's, and both modes resume.
- `SUSHI_DSV41_COMPOSED=1` (the op chains) captures too: the composed routed sum feeds the imatrix's down moments through the same tap as the fused arm.
- Refused by name: `--hidden-out` / `SUSHI_HIDDEN_OUT`, `--imatrix-windows`, a quantized KV (`Dsv41KvQuantUnsupported`),
  `--qwen-gdn 16`, `--tokens` other than 1, a missing budget.


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

A capture or compare needs no expert budget on DeepSeek or Hy4: the load takes serve's RAM default (streamed experts
score the same logits). `--ssd-budget-gb` is for a box outside the RAM tiers, or a source with no RAM default that does
not fit resident (GLM, MiMo BF16). A GLM pack is scored with `sushi kld compare --model <pack> --fixture <teacher>`.

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

<a id="release-gate"></a>
## Release gate

Each release scores two packs with `sushi kld compare --tokens 512 --ctx-size 8192`, read to the
first EOS, against the previous release's reading (within ~1%). The fixture and flags are part of the reading.

Public numbers (pack cards, the README) are the eight-category reading ([below](#eight-category)). A 16-prompt prose
fixture is a regression gate only: without the other categories its mean leans on raw prose, so it is never reported.

| pack | fixture (`kld-teacher/`) | extra flags | 1.4.0 | 1.5.0 (`f8ec7254`) | 1.5.1 (`399895f6`) | 1.6.0 (`ac3da7aa`) | 1.7.0 | 1.7.3 (`fc2fd843`) | 1.7.4 (`cb048670`) | 1.8.0 (`7c8d11d1`) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw | `qwen38-flash-next-bf16-code8-prose8-16x512` | `--kv-quant 8` | | 0.065924, 91.94%, 7061 positions | 0.065924, 91.94%, 7061 positions | 0.061834, 92.30%, 7061 positions, pack revision 2.0 | 0.061793, 92.48%, 7061 positions (`c59df290`) | 0.061793, 92.48%, 7061 positions | · (the gate moved to STQ 4bpw) | · |
| Qwen3.8-Flash-Next-Sushi-STQ-4bpw | `qwen38-flash-next-bf16-code8-prose8-16x512` | `--kv-quant 8` | | | | | 0.031214, 94.42%, 7061 positions (`c59df290`) | · | 0.031214, 94.42%, 7061 positions | 0.031214, 94.42%, 7061 positions |
| DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw | `deepseek-v41-flash-1017-native-v1` | | 0.221423, 85.53% | 0.223124, 85.49%, 7533 positions | 0.209123, 86.84%, 7533 positions, pack revision 2.0 | 0.210305, 86.46%, 7533 positions | 0.210682, 86.38%, 7533 positions (`b039659c`) | 0.210682, 86.38%, 7533 positions | 0.209273, 86.78%, 7533 positions | 0.209273, 86.78%, 7533 positions |

DeepSeek moved to pack revision 2.0 after 1.5.0; the 1.5.0 binary read it at 0.210305, 86.46%, so
1.5.1 is within 0.6%. Qwen 1.5.1 equals 1.5.0 to the last digit.
The published Qwen pack moved to revision 2.0 (6-bit STQ trunk) after 1.5.1: on the 1.5.1 code (`88c0173a`, the same
flags) it reads 0.061834, 92.30%, 7061 positions, and 1.6.0 reads the same to the last digit. DeepSeek on 1.6.0 reads 0.6% above 1.5.1, equal to the 1.5.0 binary's reading of revision 2.0.
1.7.0 reads Qwen STQ 2.75bpw 0.07% under 1.6.0 and DeepSeek 0.18% above it, both on Sushi Cooler. Its Qwen readings
ran on `c59df290` and DeepSeek's on `b039659c`, the 1.7.0 code before its last merges, which touched no KLD path.
On Sushi Cooler with no call into MLX from load to unload (`df84c004`, the row's flags with `--tokens 512 --top-k 10
--ctx-size 8192 --no-mtp`, AC power) DeepSeek reads 0.210305, 86.46%, 7533 positions: 1.6.0's reading on MLX.
With the router's product in fixed tiles (`f71dfd5c`, the same flags) it reads 0.211004, 86.51%, 7533 positions: +0.33%
in KLD, +0.05 points of top-1. The fixture's prompts are 282 to 338 rows, which took the product's split-K arm before
and take the fused arm now ([dsv41-arch](dsv41-arch.md#prefill-width)); the tree with the one function reverted reads
0.210305 to the last digit.
The reading is the same to the last digit (0.211004283, 86.5127%) at `bee7b113`, after the router's unpadded short
tile, the copy-round width rule and the draft's Markov kernel, and that run's unload reports no call into MLX while it
was at rest (the `f71dfd5c` run reported five, the device reads `a74fcf43` moved above the load).
With the wide hyper-connection mix on the matrix units (the source tree of that change on `42c364c6`, the same flags)
it reads 0.210682, 86.38%, 7533 positions: -0.15% in KLD, -0.13 points of top-1. `42c364c6` itself, run in the same
session, reads 0.211004283. The kernel is within 2e-6 of the f64 truth ([dsv41-arch](dsv41-arch.md)); the move is the
size of the router tile's above and of mixed sign (NLL +0.10%).
1.7.3 reads both packs equal to 1.7.0 to the last digit shown, on Sushi Cooler.
1.7.4 (M5 Max, `MLX_ENABLE_TF32=0`) scores Qwen STQ 4bpw in place of 2.75bpw (owner): equal to its 1.7.0 reading to the
last digit shown; DeepSeek reads 0.67% under 1.7.3 (SSD budget 108 GiB from the settings file, as 1.7.3).

The Qwen 1.5.0 reading equals the 1.4.0 Homebrew binary's on the same settings to the last digit; the 0.0692 / 92.76%
quoted for this pack in [qwen4-kld](qwen4-kld.md) was taken on settings it does not name.

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

<a id="hy4-eight-category"></a>
## Eight-category reading (Hy4-Flash)

The same 16 prompts and categories against Hy4-Flash's BF16 checkpoint (`hy4-flash-bf16-8cat-v1`, captured on a NAX
GPU), compared at `--kv-quant 8 --no-mtp --tokens 512 --ctx-size 8192`, `MLX_ENABLE_TF32=0`, every routed expert
resident, to the first EOS (7859 positions). M5 Max, `7c8d11d1`, 2026-11-26. KLD (top-1):

| pack | all | code | math | reasoning | design | wikitext | multilingual | story | stem |
|---|---|---|---|---|---|---|---|---|---|
| Hy4-Flash-Sushi-STQ-4bpw | 0.019387 (96.30%) | 0.006021 | 0.005801 | 0.013732 | 0.016615 | 0.036889 | 0.016785 | 0.050044 | 0.007921 |
| Hy4-Flash-Sushi-STQ-2.25bpw | 0.102246 (91.72%) | 0.028985 | 0.039940 | 0.087497 | 0.107893 | 0.080213 | 0.109945 | 0.325550 | 0.036802 |

## Teacher fixtures

| fixture | model | notes |
|---|---|---|
| Flash-Next 16x512 raw | Flash-Next | the bf16 checkpoint, bf16 stream, raw text; the standard |
| Flash-Next 8-category 16x512 | Flash-Next | the bf16 checkpoint, f32 GDN state and gate, 2 prompts per category ([above](#eight-category)) |
| Hy4-Flash 8-category 16x512 | Hy4-Flash | the BF16 checkpoint through f32 products, captured on a NAX GPU, 2 prompts per category ([above](#hy4-eight-category)) |
| Flash-Next 16x512 raw, f32 stream | Flash-Next | the same with an f32 residual stream |
| Flash-Next 60x64 | Flash-Next | the 60x64 screen |
| MiMo 16x512 raw | MiMo | the MOPD checkpoint as stored (FP8 trunk: bf16 weights in prefill, FP8 code x f32 block scale in decode), dense KV; captured 2026-09-30 by 83dc9b6c; mean strict NLL 0.2664; 652 s capture |

Commands (MiMo; `--ssd-budget-gb` only because MiMo has no RAM default to stream its source under):

```sh
sushi kld capture --model <original checkpoint> \
  --prompts <the Flash-Next 16x512 raw fixture> --out <teacher dir> \
  --tokens 512 --top-k 10 --label <label> --no-template --kv-quant off --ctx-size 8192 --ssd-budget-gb <n>
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

`--expert-pick-tolerance` on Sushi-2bpw, a 20 GiB expert cache (251 slots/layer), kv8, 16 prompts x 128 tokens,
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
a 60 GiB expert cache (81 slots/layer), kv8, 16x512 raw, teacher = the same load with exact routing captured on this
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

