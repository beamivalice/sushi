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
| `--model <dir>` | required | Qwen3.8-Flash-Next BF16 source checkpoint; any other architecture is refused by name |
| `--out <dir>` | required | pack directory; refused when it already holds files, except with `--resume` |
| `--experts-k <K>` | 2.5 | routed-expert rate in bits per weight, 1.5 to 8 in steps of 0.125 |
| `--codebook mcg\|mul1` | mcg | |
| `--window <12..16>` | 15 | trellis search window |
| `--calib <file>` | built-in mix | calibration text |
| `--calib-tokens <n>` | 65536 | calibration tokens, at least 2048 |
| `--trunk-bits <n>`, `--trunk-group <n>` | the model's served pack (Qwen 8, 64) | trunk affine width (2, 3, 4, 5, 6, 8) and group (32, 64, 128) |
| `--factor-precision 64\|32` | 64 | Hessian factor: 64 on the CPU in float64 (the default and the oracle), 32 on the GPU in float32; 32 changes the pack bytes ([float32 factor](#float32-factor)) |
| `--resume` | off | continue an interrupted conversion in `--out` |

A bad value or an unknown argument exits with an error before anything runs. `sushi convert --help` and
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
`calib_tokens` and `hessian_min_rows`. The same keys, except the engine's, sit in `config.json`'s `expert_quant`.

## Resume

The pack is built in `--out` directly. Every file is written under a temp name and renamed, and `config.json` and the index
come last, so an interrupted pack has no `config.json` and does not load. `convert-progress.json` records the policy (encoder
version, rate, codebook, window, trunk, calibration, minimum rows, a float32 factor) and which phases are done; `--resume` refuses a directory
whose policy differs, and skips every layer whose shard is complete (a shard is complete when its header and its size agree, at any size). The file is deleted when the pack is complete, before the scratch files, and `--resume` into a directory that already holds `config.json` stops with `ResumePackComplete` and touches nothing.

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

The MTP head's experts use the identity matrix.

Status: `tests/test_encoder_capture_real.sh` converts a real four-layer Qwen3.8-Flash-Next slice layer by layer and compares every hidden boundary, MLP input row, routing id and Hessian with the engine's full forward; they match. The MTP head is not captured yet.

## Search

The trellis search is the plain Viterbi with tail-biting pinning. `src/encoder/encode.zig` is the CPU oracle and
`src/encoder/metal_search.zig` is the Metal kernel; a test proves the two choose the same codes at every admitted rate and
window, and another that every written tile decodes to the values the search chose. The oracle follows the kernel's
arithmetic (float32 squared error, float16 path costs), which is the reference the fixtures come from. One global scale per
tensor is a golden-section search, folded into `suh`.

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

## Sources

BF16 is read directly, FP8 e4m3 with 128x128 block scales through the `fp8_block` decode, and MXFP4 (e2m1 values with one
e8m0 scale per 32) through the MiMo source reader plus a dequant. A fixture pins each against the source's own dequant. The
dequantized source is the target. A source the reader cannot identify is refused.
