# EXL3 encoder core

The numerical core and the writers in `src/encoder/`. The command, the pack it writes, calibration and resume are
[encoder](encoder.md); the format the engine reads is [pack-format](pack-format.md). The module imports `sushi_exl3` for the
codebooks and decoders and the host root (`mlx_host`) for MLX, the tokenizer, the KLD loader and the MoE tap.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-exl3-experts](engine-exl3-experts.md).

## Search

One 16x16 tile at a time. `encode.zig` is the CPU oracle: a two-pass tail-biting Viterbi. A free pass over the tile rolled by
half picks the closing register, a second pass pinned to it writes the path. It follows the Metal kernel's arithmetic
(float32 squared error, float16 path costs, one-byte backpointers) because that is the reference the fixtures come from.
`metal_search.zig` is the Metal kernel; a test proves it picks the same codes as the oracle at every admitted rate and
window, and another that every packed tile decodes to the values the search scored. When the carried state (window minus
the smallest fresh-bit count) is 13 bits, as at w15 from K2 to K2.875, and every phase's predecessor costs fit in
registers, a tile runs 1024 threads over one
cost row in threadgroup memory (read the predecessors, barrier, write) and reads each codeword's value from a decode table
the hash itself fills, a packed word's edges as half4 loads; other plans keep two cost rows and hash. Every edge scores
the same candidates in the same order either way, so the codes do not change. Rates are the
even halfword counts the loader admits (`n` 16..128, K1 to K8), windows 12..16, codebooks MCG and MUL1. A bad rate, window
or packed length is a named error.

## Quantizer

`quantize.zig` turns a public weight `[in, out]` and its input Hessian into the packed tiles, `suh` and `svh`. `in` and `out`
are multiples of 128. Block-RMS scales, optional output scales, sign vectors and the 128-point Hadamard give the inner matrix.
The Hessian is rotated into that basis, damped by `0.025 * mean(diag)` and factored as a 16-row block LDL (a blocked Cholesky in Zig whose trailing update is
tiled two rows by four columns, each element keeping its own lane order, so tiling does not move a bit; `--factor-precision 32`
runs it on the GPU in float32, see [encoder](encoder.md#float32-factor)).
Reverse LDLQ feeds each 16-row group's error back through that factor. One global scale per tensor is a golden-section
search over the codebook's bracket, folded into `suh` (`inner *= g`, `suh /= g`). `metal_search.zig` runs the same pipeline
with the tile searches on the GPU: `prepareBatch` is the host half (no MLX call, any thread), `finishBatch` the GPU half,
and `Search.submit`/`collect` let the host work while a launch runs. The feedback of a finished 128-row chunk is blocked
over rows and columns with each element summed in row order, so it equals the row-at-a-time loop bit for bit.

Sign vectors are splitmix64 streams seeded by the tensor name, which includes the expert index; `signs(seed, name, len)` is
byte-identical for the same arguments.

## Shard writer

`shard.writeLayer` writes one layer's nine tensors (`trellis`, `suh`, `svh` for gate, up and down) to
`layer-{L}.safetensors`, or `mtp-{L}.safetensors` for the MTP head's bank. `prefix` is `language_model.model` or
`language_model.mtp`. Lengths must match `E`, `in`, `out` and `n`. The file goes to a temp name and is published with `link`,
so a second write is `ShardExists` and a crash leaves no truncated shard. The stamp is `format`, `k`, `codebook`, `window`
plus the caller's provenance pairs.

## Sources and trunk

`source.zig` identifies the architecture and reads BF16, FP8 e4m3 with 128x128 block scales and MXFP4 into f32, streaming one
tensor or one expert at a time, each pinned against the source's own dequant by a fixture. `quantizeAffineBf16` is MLX's
affine quantizer through `mlx_quantize`, pinned bit for bit. `pack.zig` writes the Qwen trunk, the n-gram table, the config
and index.

## Fixtures

`src/encoder/fixtures/` holds three small safetensors files. `exl3_encode_tiles` has eight tiles per config (MCG and MUL1,
windows 12, 15, 16, rates K1.5 to K4); each tile's packed codes and cost are the plain Metal search's output on the
tensor-core-permuted tile, so the CPU oracle must match them. `exl3_encode_tensor` is one 256x128 weight, one SPD Hessian
and the regularized scales, global scale and packed codes at five points. `exl3_encode_sources` holds a BF16, an FP8-block
and an MXFP4 tensor with their reference dequant. The dumpers that produced them stay outside this repo.
