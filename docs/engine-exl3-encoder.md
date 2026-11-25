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

A launch starts at 1024 threads per threadgroup and never assumes a pipeline accepts them: MLX refuses a launch above the
compiled kernel's own limit (384 to 832 on an M2 Pro) and names it, so `gpu_factor` and `metal_search` read the limit from
that refusal and relaunch at the largest count they support below it. The factor's output does not depend on the count; the
search drops the 1024-thread plans (single-cost, and K1.5 w15 paired) for the 256-thread plan, which picks the same codes.

<a id="pair1-search"></a>
### PAIR1 pair search

PAIR1 reads only the window ending at each pair's odd weight. So the search steps once per pair (`pair_search.zig`),
and its state is that 16-bit window.
- A pair step takes the two weights' fresh bits: floor((j+1)n/8) - floor(jn/8), 3 at K1.5 and 8 at K4.
- Each step scores 2^16 (state, candidate) edges with both weights' errors, `fma(d1, d1, fma(d0, d0, prev))`. Path costs
  are float16, the lowest candidate wins a tie, and the pin is chosen as in the scalar search over A/2 pairs.
- The edge count equals a w15 scalar search's (128 x 2^16 = 256 x 2^15), but each edge costs more.
- `expandScalar` turns the 128 pair codewords into the 256 scalar windows (even = the previous pair's window shifted by
  weight 2j's bits), so the scalar packer and the engine's tile decode are unchanged.

What is proven:
- The host search writes another implementation's pair codewords and decoded weights byte for byte at all 21 rates and
  at both 64 and 32 pairs (fixtures `pair1_search_pin{64,32}.bin`, dumped by `tests/dump_pair1_search_fixtures.py`).
- The Metal kernel (`pair_kernel.zig`, one source per rate) packs the host search's bytes at every rate and both pins.
- A tensor quantizes to the CPU quantizer's bytes.
- A tiny PAIR1 pack decodes through the engine's MoE to the host decoder's weights.

The kernel:
- Holds both cost rows in threadgroup memory.
- Stores every step's half costs to device scratch and re-scores the traced state's candidates on traceback, instead of
  keeping backpointers.
- Hashes each candidate's pair (the low rates, where the 32 KiB of costs leaves no room for the 256 KiB table) or
  reads a half2 table built on the host from `format.decodePair`.

The per-rate plan (tiles per threadgroup, threads, table rows) does not change the codes. The plan ubench is
`SUSHI_ENCODER_PAIR1_UBENCH=<n,...>`. It runs every plan against SPEC1 w15 in one process, interleaved, at pin 64:

| n | plan | PAIR1 / SPEC1 w15 |
|---|---|---|
| 24-30 | 1 tile, 1024 threads (n30 512), hash | 1.26-1.52 |
| 32 | 2 tiles, 256 threads, table | 0.96 |
| 34 | 2 tiles, 512 threads, table | 1.07-1.12 |
| 36-38 | 2 tiles, 128 threads, table | 0.75-0.85 |

Hashing every edge bounds K1.5 to K1.875 (n24-30): the exact fma order leaves nothing to factor out of an edge.

## Quantizer

`quantize.zig` turns a public weight `[in, out]` and its input Hessian into the packed tiles, `suh` and `svh`. `in` and `out`
are multiples of 128. Block-RMS scales, optional output scales, sign vectors and the 128-point Hadamard give the inner matrix.
The Hessian is rotated into that basis, damped by `0.025 * mean(diag)` and factored as a 16-row block LDL (a blocked Cholesky in Zig whose trailing update is
tiled two rows by four columns, each element keeping its own lane order, so tiling does not move a bit; `--factor-precision 32`
runs it on the GPU in float32, see [encoder](encoder.md#float32-factor)). The float64 host kernels (the factor, the
target's solves, GPTQ's inverse) join each product to its sum with `@mulAdd`: Zig never fuses `acc += x * y` by itself.
Reverse LDLQ feeds each 16-row group's error back through that factor. One global scale per tensor is a golden-section
search over the codebook's bracket, folded into `suh` (`inner *= g`, `suh /= g`). `metal_search.zig` runs the same pipeline
with the tile searches on the GPU: `prepareBatch` is the host half (no MLX call, any thread), `finishBatch` the GPU half,
and `Search.submit`/`collect` let the host work while a launch runs. The feedback of a finished 128-row chunk is blocked
over rows and columns with each element summed in row order, one `@mulAdd` per term as in the CPU quantizer's
loops, so it equals the row-at-a-time loop bit for bit.

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

`src/encoder/fixtures/` holds four small safetensors files. `exl3_encode_pinpass` holds twelve tiles and, for 21 configs, the codes a Metal search with a shortened pin pass chose (`tests/dump_encoder_pinpass_fixture.py`). `exl3_encode_tiles` has eight tiles per config (MCG and MUL1,
windows 12, 15, 16, rates K1.5 to K4); each tile's packed codes and cost are the plain Metal search's output on the
tensor-core-permuted tile, so the CPU oracle must match them. `exl3_encode_tensor` is one 256x128 weight, one SPD Hessian
and the regularized scales, global scale and packed codes at five points. `exl3_encode_sources` holds a BF16, an FP8-block
and an MXFP4 tensor with their reference dequant. `dsv41_expert` is one DeepSeek-V4.1 expert in the release's own layout with its
numpy dequant (`tests/dump_dsv41_expert_fixture.py`). The other dumpers stay outside this repo.
