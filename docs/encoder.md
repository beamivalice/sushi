# Encoder: `sushi convert exl3`

Turns a source checkpoint of a model Sushi serves into a Sushi pack: the routed experts as EXL3, everything else in the
format that model's packs serve. Anyone can reproduce a pack with a fixed rate and judge it with `sushi kld`. The format
the engine reads is [pack-format](pack-format.md); the numerical core and the shard writer are
[engine-exl3-encoder](engine-exl3-encoder.md).

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [quality-kld](quality-kld.md),
[engine-expert-streaming](engine-expert-streaming.md), [flags](flags.md).

## Command

```
sushi convert exl3 --model <source dir> --out <pack dir> [options]
```

| flag | default | meaning |
|---|---|---|
| `--model <dir>` | required | source checkpoint (BF16, FP8 block or MXFP4); Qwen3.8-Flash-Next, MiMo-V2.6-Flash and GLM-5.3-Flash only, any other architecture or format is refused by name |
| `--out <dir>` | required | pack directory; refused when it already holds files, except with `--resume` |
| `--experts-k <K>` | 2.5 | routed-expert rate in bits per weight, 1.5 to 8 in steps of 0.125; with `--rate-plan`, the rate of every layer the plan does not name (the MTP head) |
| `--rate-plan <file>` | none | per-expert gate/up and down rates per layer; each planned layer is written as [rate groups](#rate-groups) |
| `--expert-codebook exl3-mcg\|exl3-mul1\|stq-std1\|stq-spec1\|stq-pair1` | exl3-mcg | the routed experts' codebook; the pack stamp keeps the bare `mcg` / `mul1` / `std1` / `spec1` / `pair1`. `stq-std1` and `stq-spec1` need `--expert-codebook-table`; `stq-pair1` runs the [pair search](#pair1) at window 16 and K1.5 to K4 |
| `--expert-codebook-table <name>` | none | the named table a `stq-std1` or `stq-spec1` pack decodes with (`src/exl3/codebook_tables.zig`); the search uses its row at each bank's rate. `stq-pair1` takes `pair1-w16-v1` without being told |
| `--trunk-codebook stq-spec1` | stq-spec1 | with `--trunk stq`, the codec of the trunk banks and `lm_head`; `stq-std1`, `stq-pair1` and the EXL3 codecs are refused ([STQ trunk](#stq-trunk)) |
| `--trunk-codebook-table <name>` | spec1-w15 | with `--trunk stq`, the trunk's table, named in `config.json`'s `trunk_quant`; only `spec1-w15` is accepted |
| `--trunk-k 4\|5\|6` | 4 | with `--trunk stq`, the rate of the trunk banks (n 64, 80 or 96) in bits per weight; `lm_head` stays K6. Hy4-Flash and DeepSeek-V4.1 read it too ([DeepSeek-V4.1](#deepseek-v41)). Without `--trunk stq` it is refused (`TrunkKNeedsStq`). Experiment-branch flag |
| `--drafter-k 4\|5\|6` | the trunk's | Hy4-Flash and DeepSeek-V4.1 only (refused by name elsewhere, `DrafterKUnsupported`): the rate of the drafter's banks, the DFly2 drafter's or the DSpark stages' trunk; see [Hy4-Flash](#hy4-flash) |
| `--drafter-experts-k <K>` | 2 | DeepSeek-V4.1 only (refused by name elsewhere, `DrafterExpertsKDsv41Only`): the rate of the DSpark stages' routed experts on the PAIR1 grid, 1.5 to 4 in steps of 0.125 (`Pair1DrafterRateUnsupported` beyond, whatever `--expert-codebook` is) |
| `--window <12..16>` | 15 | trellis search window; `stq-pair1` is window 16, and another window beside it is refused (`Pair1WindowNot16`) |
| `--calib <file>` | built-in mix | calibration text |
| `--calib-tokens <n>` | 65536 | calibration tokens, at least 2048 |
| `--capture <dir>` | none | DeepSeek-V4.1 and Hy4-Flash, required for the first and one of three for the second (with `--identity-hessian`, `--synthetic-codes`); another arch refuses it: the `calib/` directory of a layer-major calibration capture (`mlp-in-LL.bf16`, `route-LL.u16`, `trunk-LL.safetensors`, `trunk-head.safetensors`, `dspark-in-37..39.bf16`, `../windows.jsonl`); see [DeepSeek-V4.1](#deepseek-v41) |
| `--synthetic-codes` | off | Hy4-Flash only (refused by name elsewhere), a dev pack: random trellis codes and scales matched to each source tensor's RMS, no calibration or search; see [Hy4-Flash](#hy4-flash) |
| `--calib-ids <file>`, `--calib-windows <file>` | none | calibration as token ids (little-endian u32) cut into the windows a JSON `lengths` list gives, end to end; replaces `--calib` |
| `--fisher-dir <dir>` | none | per-layer output Fishers (`fisher-LL.npz`): routed experts take [Fisher-weighted output scales](#fisher-and-target) |
| `--target none\|compensated` | none | `compensated`: routed experts round the [compensated target](#fisher-and-target), its BF16 path carried in the run |
| `--target-boundaries <dir>` | none | the compensated target with the BF16 model's stored layer inputs over the same calibration (`boundary-LL.u16`) instead of the run's own; it implies the target, and `--target none` beside it is refused |
| `--trunk gptq\|stored\|rtn\|stq` | gptq | how the trunk's linears are rounded: [re-rounded by GPTQ](#gptq-trunk) along the quantized path (the default), taken as stored from `--trunk-from`, rounded to nearest at `--trunk-bits`/`--trunk-group`, or (`stq`) the `--trunk-from` pack's trunk [rounded onto the trellis](#stq-trunk) with its experts kept |
| `--trunk-from <pack>` | none | Qwen3.8-Flash-Next only (beside a Hy4-Flash or DeepSeek-V4.1 source it is refused, `TrunkFromUnsupported`): take every non-expert shard and the n-gram table from this pack of the same model (copy-on-write clones), with its config's affine settings; see [rate groups](#rate-groups). Its width is the pack's, so `--trunk-bits` and `--trunk-group` beside it are refused, and so is a pack written with a rate plan |
| `--trunk-bits <n>`, `--trunk-group <n>` | the model's served pack (Qwen 8, 64) | trunk affine width (2, 3, 4, 5, 6, 8) and group (32, 64, 128); `--trunk gptq` rounds 2, 4 and 8 only, and refuses the others up front; pending the owner's approval ([flags](flags.md#owner-calls)) |
| `--factor-precision 64\|32` | 64 | Hessian factor: 64 on the CPU in float64 (the default and the oracle), 32 on the GPU in float32; 32 changes the pack bytes ([float32 factor](#float32-factor)) |
| `--pin-pass <A>` | 64 | positions on each side of the wrap that the search's first pass covers, a multiple of 16 from 16 to 128; 128 is the full pass, a smaller A is faster and changes the pack bytes ([pin pass](#pin-pass)); `stq-pair1` counts A/2 pairs |
| `--layer-cooldown <s>` | 0 | seconds the encoder rests (Qwen, GLM and MiMo only; Hy4-Flash and DeepSeek-V4.1 refuse it, as every flag only the generic driver reads, by name), CPU and GPU idle, after each written layer; 0 is no rest, and no rest length paid on an M5 Max (see Calibration capture); the pack bytes and the resume key do not depend on it, so a resumed run may change it |
| `--cpu-threads <n>` | all cores | cap on the CPU count every pool is sized from (prep, target, feed, the trunk rounding and the search workers); at least 1, a value above the core count is the core count; the pack bytes do not depend on it |
| `--resume` | off | continue an interrupted conversion in `--out` |

A bad value or an unknown argument exits with an error before anything runs. So do a rate plan or `--experts-k` rate the
search cannot run (odd, or below 1.5), a missing `--rate-plan` file (`RatePlanFileMissing`), and, for a run that reads them
per layer, a missing `fisher-LL.npz` (`FisherFileMissing`) or `boundary-LL.u16` (`TargetBoundaryMissing`); the log names the
file. A resumed run does not look for the files of the layers it keeps. `sushi convert --help` and
`sushi convert exl3 --help` list every flag.

## What is written

Only the routed experts (and the MTP head's experts) are EXL3; the engine serves nothing else in that format. The trunk is
written the way each served pack stores it.

| tensors | format |
|---|---|
| routed experts, one shard per layer (`layer-{L}.safetensors`) | EXL3 at `--experts-k`, MCG or MUL1, `--window` |
| MTP head experts (`mtp-0.safetensors`) | the same rate and codebook as the routed experts |
| Qwen trunk (`model-trunk-{N}.safetensors`) | MLX affine, 8 bits, group 64, under the served names (`language_model.model.*`, `language_model.mtp.*`, `language_model.lm_head`, `vision_tower.*`) |
| Qwen n-gram table (`ngram_table.bin`) | the source's n-gram shards merged into one 4-bit group-32 table |
| tokenizer, chat template, generation config, license, preprocessor configs | copied |
| `model.safetensors.index.json`, `config.json` | written last; `config.json` is the source's plus `quantization`, `ngram_table` and `expert_quant` |

Every Qwen trunk linear uses one group, `--trunk-group` (64 by default). The served packs mix group 64 and 128 across tensors with identical shapes; no shape rule decides it. Their picks come from a tuned, error-based allocation the public encoder does not have, so the encoder stays uniform.

Every expert shard carries the engine's stamp (`format`, `k`, `codebook`, `window`) and the encoder's provenance:
`encoder` (the encoder version), `quantizer`, `global_scale_search`, `calib_sha256` (a SHA-256 of the calibration token ids),
`calib_tokens` and `hessian_min_rows`, and `pin_pass` unless `--pin-pass` is 128. The same keys, except the engine's, sit in
`config.json`'s `expert_quant`, which adds the ones that describe the whole pack when the run used them: `rate_plan_sha256`,
`out_scales_mode`, `target` and `trunk`.

<a id="rate-groups"></a>
## Rate groups, table codebooks and a served trunk

`--rate-plan` reads a JSON plan (`"format": "exls-rate-plan-1"`, `layers: {"L": {"gate_up": [K per expert], "down":
[K per expert]}}`). A planned layer is written as `.gN` rate groups ([pack-format](pack-format.md#rate-groups)): experts
sharing a (gate/up, down) pair form one group, the widest pair first (gate/up, then down), each group's experts in
ascending id order. The router rows follow that order, so the router is rewritten in the pack (the trunk written from the source permutes it as it writes, the `--trunk-from` copy in place); the capture routes with the
rewritten router and maps the ids back to source ids, so every expert's Hessian, signs and bytes are the ones a uniform
layer at its rates would give it (a test proves this expert by expert). `expert_quant.k` and each shard's `k` stamp are the
widest rate written; `config.json` adds `rate_plan_sha256`.

A table codec (`stq-std1`, `stq-spec1`) writes `format: stq`, `codebook_table` and `codebook_digest`. The search decodes each candidate
with the table's (multiplier, mask) row at the bank's rate, and the GPU search matches the CPU oracle code for code at
every admitted rate and window for each table.

<a id="pair1"></a>
`stq-pair1` (table `pair1-w16-v1`, digest `c089b4eb`) also writes `format: stq`. It decodes both weights of a pair from the
window ending at the odd weight, so the search is a trellis of 128 pair steps over that window
([engine-exl3-encoder](engine-exl3-encoder.md#pair1-search)). The admitted rates are those the table has a row for:
even n from 24 to 64 (K1.5 to K4). Any other rate is refused before anything runs (`Pair1RateUnsupported`). The pin pass
is A/2 pairs, so the default 64 runs 32 pairs, and `--pin-pass 128` runs the whole tile, which is the search behind every
PAIR1 pack made before this encoder.

`--trunk-from` takes a served pack's trunk as stored instead of quantizing the source's: every shard of that pack with
no routed-expert bank and its `ngram_table.bin` are cloned (under `--trunk gptq`, which rounds the trunk itself, its
re-rounded `trunk-*` shards stay behind), a planned layer's router is reordered in place in the clone, and
`config.json` starts from that pack's config with its `quantization` and `ngram_table` settings. The expert shards of the
served pack are left behind. A pack this encoder wrote with a rate plan already stores its routers in group order, so it
is refused as a trunk source (`TrunkFromRateGrouped`).

`--calib-ids`/`--calib-windows` take a calibration other tools stored. `calib_sha256` is then the SHA-256 of the ids as
little-endian i32 followed by the window lengths as little-endian i64, the digest those tools record for the same windows.
At about 1M tokens the capture holds one float32 MLP input row per token (about 10.7 GB at hidden 2560) and one hidden
boundary, updated in place layer by layer (about 21 GB).

<a id="fisher-and-target"></a>
## Fisher-weighted output scales and the compensated target

Both are off by default and leave the default bytes unchanged. They take inputs another tool computed over the same
calibration.

- `--fisher-dir`: `fisher-LL.npz` per layer holds float32 `fisher_gate`, `fisher_up` [experts, intermediate] and
  `fisher_down` [experts, hidden], the output Fisher F of each routed expert by source id. Each tensor's output scales
  become t_j = c_j^(1/4) (F_j + 0.01 mean F)^(-1/4), normalized to geometric mean 1 (c_j is the channel's mean square
  weight; an expert no row reached keeps t_j ~ c_j^(1/2)). The quantizer rounds W / t and `svh` carries t. On
  MiniCPM5-1B, K2.5 MCG w15 over the same sequential harness, this moved the geomean KLD from 0.2706 to 0.2465.
- `--target compensated` carries the BF16 path in the run: `.capture/boundary-LL.u16`, in the stored layout, one at a
  time. Layer 0's is each token's source embedding row once per stream. At each layer boundary the layer is staged once
  more as the BF16 model's (`stage-bf16`: the source trunk linears and the source's BF16 routed experts as dense
  `switch_mlp` weights, in the pack's router order) and boundary L is forwarded through it to write L+1. Pass A's
  layer cannot serve: its experts are placeholders, so only its MLP inputs, the target's rows, are the BF16 path's.
  Disk: 21 GB at 1M tokens, 42 GB while a layer advances. The PLE layer reads the source's BF16 n-gram shards, not the
  pack's table, which `--trunk-from` may carry at 4 bits, both for the BF16 boundary and for the target's rows X. At the layer boundary BF16 L+1 is published before the quantized
  `hidden-(L+1)` and BF16 L is dropped after it, so `--resume` finds both boundaries of its first unfinished layer.
- `--target-boundaries`: `boundary-LL.u16` holds the BF16 model's input to layer L for every calibration token (all
  hyper-connection streams, bf16), stored by another tool. The capture forwards it through the staged layer next to the quantized path's state.
  That gives X, each routed row's MLP input on the BF16 path, routed as the quantized path routes. Each expert then
  rounds W* = W + a (M W - W), M = (X~'X~ + 0.01 mean diag)^-1 X~'X (`src/encoder/target.zig`, float64 solves). a is
  0.2 from four times the input width in routed rows and 0 at 2560 rows or fewer, linear in between. Down's X~ runs
  through the expert's rounded gate/up, so every gate/up bank of a layer rounds before its down banks. The staged
  layer's attention half uses the pack's trunk, so X carries that trunk's rounding.
- The target costs about 0.85 s of host time per 2560-wide gate/up expert (one core), spread over the prep threads.
- While the main thread waits on a gate/up bank's host work, it computes the down banks' Hessians and target terms
  ahead (`EarlyDown`), each as soon as its expert's gate/up is decoded. They are held until their bank runs, which
  takes 4.2 GB for 512 experts at 2560 x 640.
- Layer 0 of Qwen3.8-Flash-Next with both on: 1,049,088 calibration tokens in 1194 windows, a 4-rate plan
  (K1.5/2/2.5/3, 7 groups), STD1 w15, served 2 bpw trunk. Commit 0d23f0b3, GPU lock, quiet box, `taskpolicy -a`, one
  run.
  - Capture 28.6 s, convert 392.1 s, advance 17.9 s; peak footprint 68.8 GB.
  - At a million rows the Grams and cross terms share the GPU with the search, and the target solves fill the
    host.

<a id="stq-trunk"></a>
## STQ trunk (`--trunk stq`)

`--trunk stq --trunk-from <pack>` rounds the dense trunk of a Qwen3.8-Flash-Next pack made with `--trunk gptq` onto the
trellis and keeps everything else of it: the routed-expert shards, routers, embedding, n-gram table, MTP head and vision
tower are cloned (copy on write). `--model` is the BF16 checkpoint, `--calib-ids`/`--calib-windows` the calibration.
`src/encoder/trunk_stq.zig` drives it; the rounding is `trunk.zig`'s `Rounding` with a third mode beside GPTQ.

- **Recipe**: `--trunk-codebook stq-spec1 --trunk-codebook-table spec1-w15` (the defaults), the codec of the DeepSeek-V4.1 trunk and of the experts of the pack it starts from. The experts' own codec is whatever `--trunk-from` holds, and `--expert-codebook` does not touch them. Either trunk flag without `--trunk stq` is refused (`TrunkCodebookNeedsStq`). `stq-std1` has no table row at the head's K6 (its rows stop at n 64) and the dense quantizer is SPEC1's, so it is refused, with `stq-pair1` and the EXL3 codecs (`TrunkCodebookUnsupported`); another table is refused too (`TrunkCodebookTableUnsupported`), and a table of another codec is `CodebookTableMismatch`. `config.json`'s `trunk_quant` names the trunk's table apart from `expert_quant`'s.
- **Banks**: every linear of the
  attention, GDN and shared-expert groups that sits on the 128 grid on both sides is a K4 bank (K6 with `--trunk-k 6`; all share one rate, as the stacked decode launches need)
  (`<base>.trellis` U16 `[in/16, out/16, 64]`, `.suh` F16 `[in]`, `.svh` F16 `[out]`); `lm_head` is K6 (`n` 96). The
  bytes of one tensor are those of `dsv41.quantizeDense` on the same Gram (a test pins it): same signs
  (`trunk.sign_seed`, the served name), regularizer, damping, global scale, LDLQ and `--pin-pass`.
- **Kept as the pack stores them** (affine 8-bit g128): the GDN `a` and `b` gates (48 rows), the four hyper-connection
  mixes per layer and the final mixer (320 on one side), the PLE key and value projections (the engine reads them
  affine), the router, norms, the embedding, the MTP head (its forward is not captured by the encoder, though the
  engine would take its projections as banks) and the vision tower.
- **Hessians**: layer by layer, as `--trunk gptq` takes them: the engine's own forward over the calibration windows, a
  one-layer stage whose routed experts are placeholders (the Grams of the trunk do not read them), `linear_tap` handing
  each projection the input it really receives (after the hyper-connection read and norm), summed over every fourth
  batch of 8192 tokens. Pass A is the attention half, pass B the shared expert's gate and up, pass C its
  down; each pass stages the linears rounded so far as their dense decode (`linear.decodeDense`, bf16) and the source
  weights of the rest. After the passes the layer runs over all windows with the new trunk and the pack's own expert
  shard to give the next layer's input. The head's Gram is taken on the last boundary through the source mixer.
- **Written**: per layer `trunk-stq-LNN.safetensors` (the banks, stamped `format stq`, `k`, `codebook`, `window`,
  `codebook_table` of the trunk codec) and `trunk-LNN.safetensors` rewritten without the converted linears' affine triples; the head's
  `trunk-stq-head.safetensors` and `trunk-head.safetensors` alike. `config.json` is the base's with the converted bases
  removed from `quantization` and `quantization_config` and a `trunk_quant` block (`k` the widest, 6).
- **Source pack**: the experts and routers are cloned as stored, so a pack written with a rate plan is a valid `--trunk-from` here (the `TrunkFromRateGrouped` refusal belongs to `--trunk gptq`, which reorders routers).
- **Resume**: `--resume` goes on from the newest stored boundary (`.capture/hidden-N.bin`); a layer is redone whole.
  `convert-progress.json` holds the policy (base config, rates, pin pass, calibration).
- `SUSHI_ENCODER_STQ_LAYERS=n` rounds only the first n layers and writes no config (timing).
- `tests/verify_qwen_stq_trunk.py <base> <new>` checks header, shapes, names, that experts and n-gram are the base's bytes.

<a id="gptq-trunk"></a>
## Re-rounded trunk (`--trunk gptq`)

The trunk starts as `--trunk-from`'s (or, without it, round-to-nearest at `--trunk-bits`/`--trunk-group`), and every
trunk linear a layer runs is then rounded again by GPTQ in that tensor's own affine format (`src/encoder/gptq.zig`): per
output row and input group a bf16 scale and bias, columns rounded left to right with the error fed forward through the
upper Cholesky factor of the damped inverse input Gram (damp 0.01 mean diag, blocks of 128). A test pins the codes, scales,
biases and packed words against a reference implementation.

- The Grams are the inputs the quantized model computes (`src/encoder/trunk.zig`), summed over every fourth forward
  batch. A layer rounds in the order it runs, each pass staging the layer through the linears rounded so far and the
  source weights of the rest:
  - pass A: the attention half (PLE key/value, the attention hyper-connection's down/up, the attention projections and
    the indexer, the output projection);
  - pass B: the MLP hyper-connection's down/up and the shared expert's gate/up; the routed experts' gate/up rows and
    routing come from this pass;
  - pass C: the shared expert's down projection; the routed experts' down rows and routing come from this pass, through
    the rounded MLP input side.
  - The final mixer and `lm_head` round last, on the last boundary, through the source mixer.
- `Transformer.linear_tap` hands the encoder each projection's input with its weight handle; it is null outside the
  encoder. A float weight without scales loads dense, which is how a stage carries source weights.
- Each layer's rounded linears go to `trunk-LNN.safetensors` (the head's to `trunk-head.safetensors`), and they replace
  the base trunk's copies when the pack completes. The log reports each one's tr(E H E') / tr(W H W').
- The BF16 path's rows (`--target-boundaries`) come from the all-source layer, and an expert keeps a measured Hessian
  down to one routed row.
- Forwards run the calibration windows in batches: consecutive windows of one length, at most 8192 tokens together.
  `stored` and `rtn` keep one window per forward, so their bytes are those of earlier encoder versions.

## Hessian rotation and the reference's gate/up error

The encoder rotates each expert's Gram into the inner basis in float64 (`quantize.rotatedHessian`). The reference
converter rotates it with MLX float32 matmuls on the GPU, whose error (see [Float32 factor](#float32-factor)) moves the
rotated Gram's eigenvalues by about ±2.5% of its mean diagonal. That is the size of the 0.025 damping. On a gate/up
Gram that is rank-deficient or close to it, the reference's Cholesky then fails, and its retry adds 0.05 mean diag
(0.075 in all). The extra damping lowers the weight-space error and raises the error on the calibration inputs.

The proof is layer 0 of Qwen3.8-Flash-Next, 1M calibration tokens, pin 128, on the same (W, H, output scales, signs):

- The reference and the encoder agree on `suh`/`svh` bit for bit. Their L factors differ by 32% on two thin experts
  (rank about 1440 of 2560, min rotated eigenvalue -0.028 mean diag) and by 4-6% on two full-rank ones.
- With only its rotation made float64, the reference gives the encoder's weight error: 0.24885 vs 0.24885,
  0.27909 vs 0.28023, 0.2899 vs 0.2899 and 0.20821 vs 0.20845.
- Weight-space rel err over the layer is about 2% above the reference on gate/up (0.2079 / 0.1993 against
  0.2036 / 0.1951), in a quarter of the experts in every row-count bin. Down's 640-wide Grams stay clear of the
  damping, and down matches (0.131631 against 0.131489, per-expert correlation 1.0000).
- On the thin experts, tr(E'HE) / tr(W'HW) is 6-9% lower than the reference's (0.01331 / 0.01100 against
  0.01413 / 0.01212). The encoder keeps the exact rotation, and the pack KLD decides.

## Resume

The pack is built in `--out` directly. Every file is written under a temp name and renamed, and `config.json` and the index
come last, so an interrupted pack has no `config.json` and does not load. `convert-progress.json` records the policy (encoder
version, rate, codebook, window, trunk, calibration, minimum rows, a float32 factor, the pin pass unless 128) and which phases are done; `--resume` refuses a directory
whose policy differs, and skips every layer whose shard is complete (a shard is complete when its header and its size agree,
at any size; under `--trunk gptq` the layer's `trunk-LNN.safetensors` must be there too, and for the last layer the head,
`trunk-head.safetensors`, which only that layer's advance writes). The file is deleted when the pack is complete, before the
scratch files, and `--resume` into a directory that already holds `config.json` stops with `ResumePackComplete` and touches
nothing.
A resumed capture reloads the next layer's saved boundary (`.capture/hidden-N.bin`, 21 GB at 1M tokens), which is read and
written in 1 GiB pieces: macOS refuses one read or write of 2 GiB or more.

## Calibration

The default is 64k tokens as 2048-token windows with no chat template: half WikiText-103 raw, half a frozen slice of
sushi's own `src/` (`src/encoder/calib/code.txt`). The code slice is 128 KiB; when `--calib-tokens` asks for more code than it
holds, the rest comes from WikiText. WikiText-103 is not vendored (CC BY-SA 3.0): the first run downloads
`wikitext-103-raw-v1.zip` from the pinned URL in `src/encoder/calib.zig` into `~/.sushi/cache`, checks its SHA-256 and
reads only the first part of `wiki.train.raw` out of the archive. A wrong checksum deletes the download and stops. `--calib`
tokenizes your own text file instead and takes its first `--calib-tokens` tokens.

## Calibration capture

The capture is the standard GPTQ loop, one layer at a time, and the whole model is never loaded:

1. The calibration windows' hidden state (the hyper-connection streams) is kept on the host and, after every layer, in a
   scratch file under `<out>/.capture/`, which `--resume` reads back.
2. For layer L the encoder stages a one-layer model from the pack it has written so far: layer L's affine trunk renamed to
   layer 0, the globals (embedding, `lm_head`, final mixer), the PLE tensors and the n-gram table. Its experts are a
   placeholder bank, so no BF16 expert is ever streamed. The engine's own Qwen forward runs each window over the stored
   hidden state from a fresh recurrent state, and the MoE tap in `imatrix.Collector` keeps layer L's MLP input rows and
   routing ids (`ForwardCtx.stream_in` starts a forward from a given stream, `ForwardCtx.skip_ple` leaves PLE out of every
   layer but its own).
3. `src/encoder/hessians.zig` turns those rows into each expert's gate/up Hessian (`X^T X`) and down Hessian (over the SwiGLU
   activation, using the source weights). An expert that saw fewer than `hessian_min_rows` (64) rows falls back to the
   identity matrix, and the summary names how many experts did, per layer. The matrix is rotated into the Hadamard basis,
   damped by `0.025 * mean(diag)` and factored by a blocked Cholesky in Zig (Accelerate's `dpotrf_` corrupted heap and Metal driver memory at random during long conversions, so the encoder does not call it).
4. Layer L's experts are quantized and written. A second one-layer model, with the real EXL3 shard and the affine trunk,
   forwards the stored hidden state to give layer L+1's input.

A layer runs bank by bank (`--batch` experts, default 16): the gate and up projections of a group, then the group's down
projections. While the GPU searches one bank, the main thread computes the Hessians of the bank two ahead, one expert per
LDLQ step on a second GPU stream, and a host thread reads, regularizes, rotates and factors the banks ahead. The LDLQ runs
each bank as two lanes, so one lane's 16-row step searches while the host feeds back the other's, and the rows a finished
chunk updates beyond the next chunk are fed on worker threads while the main thread waits for the GPU. A tensor's bytes
do not depend on the batch size. `sushi convert` logs the stage times of every layer: `hessian`, `scale`, `feedback`,
`search` and `pack` are the main thread's wall time (`hessian` mostly overlaps the search), `wait` is the time it waited
for host work, and `read` and `prep` are that host work, overlapped with the rest.

Layer 0 of Qwen3.8-Flash-Next, K2.5 MCG w15, 65536 calibration tokens, commit 6ed88b95, quiet box, fans max, GPU lock,
`taskpolicy -a`, one run each (same bytes at every batch):

| `--batch` | convert | prep | hessian | search | peak footprint |
|---|---|---|---|---|---|
| 8 | 181.5 s | 114.1 s | 17.2 s | 106.3 s | 16.7 GB |
| 16 | 169.1 s | 102.6 s | 26.6 s | 94.5 s | 17.5 GB |
| 32 | 188.1 s | 120.6 s | 51.4 s | 84.4 s | 20.4 GB |

The search kernel's sustained rate (15.2 us per tile, about 179 s of tiles per layer) bounds a layer. The rest is hidden
behind it.

The host threads share the package's power with the GPU. With 6 to 18 cores of float FMA load beside it, the std1 w15 search
ran about 10% slower (11.9 to 13.2 us per tile at n24) and the target Grams 7% slower (ubench, GPU lock, one session). On a
1M-token layer 0 the convert phase is GPU-bound once the host keeps up: raising the worker threads to the main thread's
QoS filled the Super cores and halved the host wait, yet the layer did not get faster. Host-side changes are judged by the
layer's end-to-end time, and a change that cuts host cycles beats one that spreads them over more cores.

At pin 64 with the re-rounded trunk and the compensated target, the gate/up banks turn host-bound. The main thread waited
32.5 s of layer 0 on target solves. Down's Hessians and target terms now fill that wait (see the compensated target), and
the trunk's GPTQ runs on a host thread beside the capture forwards.

Layer 0 of Qwen3.8-Flash-Next, 1M calibration tokens, std1 w15, rate plan beta-B, Fisher scales, compensated target,
`--trunk gptq`, pin 64. M5 Max, GPU lock, quiet box, fans max, `taskpolicy -a`, arms interleaved in one session, one run
each, all with the same bytes:

| arm | layer | capture | convert | main thread waits on host work | GPU busy in convert |
|---|---|---|---|---|---|
| c786de5b, trunk rounded after the forwards | 324.7 s | 60.3 s | 248.7 s | 33.7 s | 74% |
| b76c77e6, trunk rounded beside the forwards | 320.6 s | 48.6 s | 256.6 s | 32.5 s | 76% |
| 3693ab54, plus down's terms early | 266.0 s | 47.0 s | 204.6 s | 11.4 s | 77% |
| 3693ab54, plus every thread at the main thread's QoS | 262.2 s | 47.7 s | 200.1 s | 2.5 s | 81% |

- The QoS raise is inside run-to-run noise, and it did not move the Super-core share of CPU samples (24%), so it is not
  in the code.
- A third LDLQ lane and `--batch 32` did not help.
- The c786de5b row comes from an earlier, back-to-back session.
- Any other process holding a core at 100% adds 15-50 s to a host-bound layer.

A whole-model conversion runs slower per layer than these rows: on the same code and the same layer-0 inputs, layer 0 of
a full run took 312-361 s, and every later layer 350-395 s. Depth adds nothing: the full run's layer 0 and its layer 25
agree stage by stage within 5%. Sustained load puts the box under macOS thermal pressure ("heavy",
`notifyutil -g com.apple.system.thermalpressurelevel` = 2) with the fans already at full speed. How hard that level
clamps follows the slow chassis heat, not the die. After hours the SoC holds about 60 W: the GPU at 1.0-1.1 GHz of 1.62
when busy, even in the advance with the CPU idle, and the Performance cores at 1.6-2.2 GHz of 4.38. Every stage, host and
GPU alike, slows by 25-45%. A single-layer arm escapes this only because it runs a few minutes after a cooldown. So:

- judge a speed change only in interleaved arms at the same soak, and log the thermal level beside the number;
- plan a whole model at about 380-395 s per layer on an M5 Max laptop.

What was tried against the clamp (Qwen3.8-Flash-Next, std1 and spec1 at 2-2.75 bpw, fans max, `taskpolicy -a`):

| lever | result |
|---|---|
| 120 s idle, then resume | the next layers ran about 285-307, 315-354 and 342-377 s; the soak returns in 2-3 layers, about break-even with the pause |
| a 10 / 20 / 30 / 60 s rest before each layer, as `--layer-cooldown` gives | 399 / 400 / 403 / 409 s a layer, rest included, against 390-395 s with no rest; a second of rest buys back 0.5-0.75 s |
| a CPU cap of 9 / 6 (`--cpu-threads`) | slower than all cores at the same position after a cooldown (307 and 312 s against 285-295 s) |
| `taskpolicy -t 4` or `-t 5` | no clock moved |

So `--layer-cooldown` and `--cpu-threads` default to no rest and all cores. A cool start (the first layer after an idle
box) and a cooler room are what help. The clocks were read without sudo from IOReport (GPU and CPU performance-state
residency, GPU energy).

The MTP head's experts use the identity matrix.

Status: the pack-tier test `capture (real Qwen checkpoint, ...)` (`SUSHI_ENCODER_REAL_MODEL=1 SUSHI_MODELS_DIR=<root> zig build test -Doptimize=ReleaseFast -Dmodel=enc`) converts a real four-layer Qwen3.8-Flash-Next slice layer by layer and compares every hidden boundary, MLP input row, routing id and Hessian with the engine's full forward; they match. The MTP head is not captured yet.

## Search

The trellis search is the plain Viterbi with tail-biting pinning. `src/encoder/encode.zig` is the CPU oracle and
`src/encoder/metal_search.zig` is the Metal kernel; a test proves the two choose the same codes at every admitted rate and
window, and another that every written tile decodes to the values the search chose. The oracle follows the kernel's
arithmetic (float32 squared error, float16 path costs), which is the reference the fixtures come from. One global scale per
tensor is a golden-section search, folded into `suh`.

<a id="pin-pass"></a>
## Pin pass

Tail-biting is two passes over the tile. Pass 0 only picks the pin, the state entering position 0, which closes the loop;
pass 1 is the pinned search that writes the codes. `--pin-pass 128` runs it over the whole tile rolled by half (128
positions each side of the wrap), the full search. The default, `--pin-pass 64`, runs it over the 64 positions before
the wrap and the 64 after it: the pin is chosen from a shorter history, so some tiles end on other codes, and the pass costs
A/128 of the full one. A is a multiple of 16 so both ends of pass 0 sit on the rate pattern's period. 64 is the default because
the owner's MiniCPM A/B against the full pass measured KLD -1.2% at K2 (seed 0) and -6.1% at K2.5 (seed 42), both inside seed
noise, with the search about 13% faster.

Any A but 128 changes the pack bytes, is stamped as `pin_pass` on every shard and in `expert_quant`, and is part of the resume
policy: `--resume` refuses a directory begun at another A, in either direction. At 128 nothing is stamped, and the kernel
source and the bytes are those of every pack made before the flag. The global-scale probes and the LDLQ search use the same A. The CPU
oracle takes the same A, and both agree code for code with the codes of another implementation's Metal search at A = 64
and 32 (MCG, STD1 and SPEC1 at K1.5, K2 and K3, window 15; fixture `exl3_encode_pinpass`, dumped by
`tests/dump_encoder_pinpass_fixture.py`).

Search throughput, MCG w15, 4096 tiles per launch, median of three interleaved 3 s rounds of each arm in one process
(`SUSHI_ENCODER_SEARCH_UBENCH`, M5 Max, GPU lock, quiet box, `taskpolicy -a`, base commit 773b7099 plus this change):

| rate | pin 128 | pin 64 |
|---|---|---|
| K1.5 (n24) | 14.09 us/tile | 11.33 us/tile |
| K2 (n32) | 13.89 us/tile | 10.76 us/tile |
| K3 (n48) | 16.59 us/tile | 12.79 us/tile |

Skipping the backpointer stores of the positions pass 0 never reads, as a kernel guard, made K1.5 slower (18.5 us/tile at
pin 64) and is not done.

<a id="float32-factor"></a>
## Float32 factor

`--factor-precision 32` moves the per-expert block-LDL factor (`quantize.ldlFactor`: damping, Cholesky with its retries,
the 16x16 block inverses and the block-lower product) from the host to the GPU, in float32. The host keeps the
regularize and rotate steps and the float64 default stays byte-identical.

- **Kernel** (`gpu_factor.zig`): one threadgroup of 1024 threads per matrix runs the whole left-looking factorization in one
  dispatch, 16 columns per step: update the columns from the earlier Cholesky columns, factor the 16x16 diagonal tile on one
  thread, write W = A21 c^-T and the output block column L21 = W c^-1. Matrices are independent, so a bank is one dispatch of
  `count` threadgroups. MLX's float32 matmul is not an option: on this GPU it carries a 7.8e-4 relative error (a 512x512 by
  16 product against float64), which reconstructs H only to 2e-3, so the products are plain FMA loops over threadgroup-staged
  chunks.
- **Failure policy**: a pivot that is not positive, or a non-finite factor, retries that matrix with the CPU path's diagonal
  bump (up to `max_cholesky_retries`), then falls back to the float64 factor for it. The layer line counts both
  (`factor f32: ... N retries, M float64 fallbacks`).
- **Error bounds** (test spectra: log-normal channel scales plus a rank-one outlier of 0.5 * n * mean(diag), a damping floor
  0.025 * mean(diag), so a condition number near 20 * n): relative Frobenius error of L against the float64 factor 4e-6 at n = 256 and 8e-5 at n = 2560,
  largest entry 6e-5 of the largest |L|, reconstruction residual max|H - (I+L) D (I+L)^T| / max|H| 2e-7. The forward error
  grows with the condition number (float32 eps times cond), the residual does not.
- **Cost**: the factor is on the GPU critical path (one bank at a time, before the scale step), 6.9 ms per 2560x2560 matrix
  in a bank of 32 (a float64 CPU factor is 229 ms on one core); GPU peak 4.4 GB.
- A float32 conversion records the precision in its resume policy, so a pack never mixes the two factors.

Layer 0 of Qwen3.8-Flash-Next, K2.5 MCG w15, 67584 calibration tokens, batch 16, M5 Max, fans max, GPU lock, `taskpolicy -a`,
commit a3bb236b (the float64 numbers are the recorded 9c5b3fec-era run):

| | convert | prep | factor | wait | scale | feedback | search | rel err gate / up / down |
|---|---|---|---|---|---|---|---|---|
| 64 | 212.5 s | 87.8 s | (in prep) | 6.3 s | 33.3 s | 52.4 s | 109.5 s | 0.051027 / 0.051181 / 0.063401 |
| 32 | 173.3 s | 4.6 s | 7.4 s | 0.0 s | 27.8 s | 17.9 s | 113.7 s | 0.051021 / 0.051185 / 0.063400 |

The 32 run had no pivot retries and no float64 fallbacks. Against the 64 shard (sha256 c0f00082...), 66% of the gate and up
trellis words differ (97% of the experts differ somewhere) and 3% of the down words: a change in L reroutes the Viterbi
choices down the feedback chain, and the weight error does not move. The KLD gate has not been run.

Since 6ed88b95 the float64 factor runs on host threads two banks ahead, off the GPU's path. The float32 factor then buys
nothing. Layer 0 at batch 16 in the same session: 64 took 169.1 s, 32 took 186.0 s (factor 7.5 s on the GPU path,
search 108.3 s against 94.5 s).

## DeepSeek-V4.1

`--model` a `deepseek_v41` checkpoint converts from a stored capture instead of a forward: no layer is propagated, every layer's
Hessians come from the source path and any layer can be converted on its own (`model-layer-LL`, `model-trunk-LL` are the resume
unit). Per layer the routed experts become rate groups (`model.layers.L.mlp.switch_mlp.{gate,up,down}_proj.gN`, a uniform run is one
group, written pair-major: every routed trellis is `trellis_pm` on a 16 KiB page, the DSpark stages' too, with the pads, the stamps and the index exactly as `sushi repack` writes them from a pack-order pack ([tile order](pack-format.md#tile-order); the pack order is written for no DeepSeek pack), `--resume` unchanged), PAIR1 by default at `--experts-k` or the `--rate-plan`; the trunk linears are SPEC1 `spec1-w15` banks (K4, or `--trunk-k` 4, 5 or 6) searched against the
capture's `trunk-LL` input sums (`wo_a` as 8 groups), the router rows follow the group order, the head is SPEC1 K6, the embedding
affine 4-bit g64. DSpark's stages are calibrated in the run from `dspark-in-37..39` (trunk at `--drafter-k`, default the trunk's rate;
experts PAIR1 at `--drafter-experts-k`, default K2, one group). `trunk_quant.k` in the config is the widest trunk rate written (the head's K6 at
most `--trunk-k` and `--drafter-k`); every rate is read from the bank's own shape, and each shard stamps `k` at the rate its banks hold
(`model-globals` the head's K6, trunk shards `--trunk-k`, DSpark trunk shards `--drafter-k`). The Engram
tables are cloned copy-on-write as `engram-model-*` (outside the resident loader's `model*` scan). The config gains `expert_quant`
and `trunk_quant`; it is written last. `SUSHI_ENCODER_DSV41_LAYERS=0,5` converts only those layers and writes no config.

<a id="hy4-flash"></a>
## Hy4-Flash

`--model` a `hy_v4_flash` checkpoint writes a pack in the [Hy4-Flash STQ layout](pack-format.md#hy4-stq): per MoE layer the routed
experts as PAIR1 rate groups at `--experts-k` or the `--rate-plan` (`model-layer-LL`, written pair-major: every routed trellis is `trellis_pm` on a
16 KiB page with the pads, the shard stamp, the config's `expert_quant.tile_order` and the index `sushi repack` writes, see
[tile order](pack-format.md#tile-order); the pack order is written for no Hy4-Flash pack), the fused `gate_up_proj` split into gate rows
`0..I` and up rows `I..2I`, the router rows and correction bias in group order; per layer the trunk linears on the 128 grid as SPEC1
`spec1-w15` K6 banks beside every other tensor as stored (`model-trunk-LL`); the DFly2 drafter's linears the same way
(`model-dfly`); the embedding and head as K6 banks (`model-globals`); the Engram tables cloned copy-on-write (`engram-model-*`, BF16
as stored; a clone that fails, as across volumes, is `EngramCloneFailed`, never a copy). Every tensor goes through the engine's
census (`hy4_census.roleOf`), so a name the release does not ship is refused before a write, as is an expert bank off the 128 grid
(`Hy4ExpertsOffGrid`: the oracle tiny's 32-wide experts). It converts with exactly one of three code sources (`Hy4NeedsCodes`
otherwise):
- `--capture <calib dir>`: the Hessians of a stored calibration capture, DeepSeek-V4.1's `calib/` contract. Per MoE layer the
  routed experts go through the expert converter (`convertQwenLayer`, `--batch` experts per search batch): gate and up from the
  routed rows of `mlp-in-LL.bf16` (`[T, L]`, the latent `lat` the experts read; `mlp-in-LL.f32` from a capture at f32 products, widened alike, both forms `CaptureRowsAmbiguous`) picked by `route-LL.u16` (`[T * top_k]`), down
  from the source expert's SwiGLU over the same rows, clamped as the model runs it (gate at most `swiglu_limit`, up within
  +-limit). Each trunk bank takes the X^T X sum of its input from `trunk-LL.safetensors`, key `layers.L.<kind>` with
  `<key>.rows` (`hy4_forward.TapKind`): `attn_in` (q_a, linear_gate), `q_lora` (q_b), `kv_latent` (kv_b), `o_in` (o), `ffn_in`
  (lat_down `routed_expert_down_proj`, shared and dense gate/up), `lat_up_in` (lat_up `routed_expert_up_proj`), `shared_act`
  (shared and dense down); the head takes `head_in` from `trunk-head.safetensors`. The linears kept as stored map to their
  sums too (kv_a, the indexer's token key and head weights to `attn_in`, its query to `q_lora`, the router to `ffn_in`,
  Engram's key and value projections to `engram_in`), and before anything is written the capture is checked whole against
  the source: every MoE layer's routes and rows, every mapped sum at its linear's input width (`CaptureFileMissing`,
  `CaptureRowsMismatch`, `CaptureSumMissing`, `CaptureSumShape`), so a short capture fails in seconds, not after the
  experts. The embedding and the DFly2 drafter have no tap and round against the identity; so does an expert projection or
  sum under 64 rows; the indexer's f32 pooling projections have no tap and are kept as stored. The summary names both
  identity counts: banks with no tap, and banks under 64 rows. `tests/test_hy4_capture_convert.sh` runs the CLI capture and
  convert on the tiny end to end;
- `--synthetic-codes`: random codes, each bank's `suh` the source tensor's RMS over the RMS random codes decode to at unit
  scales (experts: expert 0's rows per projection; trunk: the whole tensor), `svh` 1, no search;
- `--identity-hessian`: each bank rounded by LDLQ and the Metal trellis search against an identity Hessian (no capture), the
  expert converter's signs and pin pass; a correctness fixture.
With a capture or the identity, every trunk bank is rounded before any trunk shard is written: the banks of one shape
(a family across the layers) share one LDLQ and search batch, up to 24 GB of working memory each, since one bank alone
leaves the GPU near idle (each LDLQ step searches only its `out/16` tiles). A bank's bytes do not depend on its batch.
`--trunk-k 4|6` (with `--trunk stq`) sets every trunk, embedding and head bank's rate (default K6), `--drafter-k 4|6` the DFly2
drafter's banks' (default the trunk's; stamped as `drafter_quant` and the `model-dfly` shard's `k`), and
`--expert-codebook stq-spec1 --expert-codebook-table spec1-w15` puts the routed experts on SPEC1 (K1 to K8) instead of
PAIR1 (K1.5 to K4). `SUSHI_ENCODER_HY4_LAYERS=<L,...>` writes only those layers' shards and times each (one-layer timings). `--resume` continues an interrupted Hy4 pack in `--out`: every shard is published whole under its final name, so an
experts, trunk, drafter or globals shard stamped with this run's `pack_revision` is kept and the rest are written as an
uninterrupted run writes them (the resumed pack is that run's byte for byte); the Engram clones are made again; a shard of
another revision is `ResumeRevisionMismatch`, an experts shard in the other tile order than the run writes `ResumeTileOrderMismatch`, a pack with its `config.json` `ResumePackComplete`, and a non-empty `--out`
without `--resume` `OutputNotEmpty`. The trunk banks are rounded only once every expert layer is written, so a stop in
the trunk phase keeps the expert layers and rounds the trunk again.
The config gains `expert_quant` and `trunk_quant` (each with `hessian`: `none`, `identity` or `capture`),
`engram_tables: "bf16"`, `synthetic` and `pack_revision` (`synthetic-`, `identity-` or `capture-<12 hex>` of the encoder, the
codes, the plan, the seed, the source's config and index, and a capture's head sum; the clones unstamped); every other shard
carries `synthetic`, `hessian` and `pack_revision`. Tests: `hy4 convert` (`HY4_BF16_TINY` refused; `HY4_BF16_TINY_STQ`, the dumper's
`--geometry stq` tiny, written each way and read back through its index: every census tensor in its form, the router in group
order, synthetic banks at the source's RMS, identity-Hessian banks within 5% (trunk K6) and 10% (experts K4) of the source's
weights; a capture of identity sums whose experts all fall under 64 rows gives the identity pack's banks byte for byte; a
capture of skewed rows gives a trunk bank and an expert a lower error than the identity pack's under the captured Hessian; the routed
shards of a synthetic and an identity-Hessian pack are the pack order's tiles through `tileMajor`, and every other file the same bytes).

**The vision tower at affine 8** (`sushi convert vision-affine8 --model <pack> --out <dir>`, flags `--model` and `--out`
only): a copy of an encoded Hy4 pack whose tower is MLX affine 8, the layout in
[pack-format](pack-format.md#hy4-vision-affine8). The pack is read, never written: each file but the config, the index and
the shards holding `vit.*` tensors is cloned into `--out` (`CloneFailed` across volumes, never a copy), those shards are
written again without the tower, `model-vision.safetensors` takes it (quantized by `mx.quantize` on the CPU stream), and
the config is written last. Refused by name: another arch (`ArchitectureNotConvertible`), no tower (`Hy4NoVisionTower`), a
tower already affine (`VisionAlreadyAffine`), a pack with a `model-vision.safetensors` (`VisionShardExists`), no index
(`IndexMissing`), a matrix not BF16 of its shape (`VisionMatrixShape`), a non-empty `--out` (`OutputNotEmpty`). The release
tower takes seconds of CPU. Test: `hy4 vision affine8` (`HY4_STQ_TINY`: every other file's sha256, the globals' kept
tensors and stamp, each code within its rounding of the source, the index's names and `total_size`, the config).

## Sources

BF16 is read directly, FP8 e4m3 with 128x128 block scales through the `fp8_block` decode, and MXFP4 (e2m1 values with one
e8m0 scale per 32) through the MiMo source reader plus a dequant; DeepSeek-V4.1's routed experts (`I8` e2m1 pairs, low nibble
first, `F8_E8M0` scales, `layers.L.ffn.experts.E.w1|w3|w2`) are the same decode behind `Arch.deepseek_v41`. A fixture pins each
against the source's own dequant. The dequantized source is the target. A source the reader cannot identify is refused.
