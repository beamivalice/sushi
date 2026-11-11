# GPU power and clock on the M5 Max

What the GPU's power ceiling and clock do to tok/s and to energy numbers on this box, measured to learn how to read
them. **Sushi never limits the GPU's power or clock for a user**: nothing here is a setting, a flag or a default.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [energy per token](process-measurement.md#energy-per-token),
[perf-baselines](perf-baselines.md).

## The GPU is less efficient at every step up in clock

The owner's `m5gpu` tool (github.com/beamivalice/m5gpu, 13ebed8) sets a GPU power limit through the `AGXAccelerator`
driver node and times a 4096 x 4096 FP32 matmul at each limit. M5 Max, 40-core GPU, macOS 26.5:

| limit | clock | power | GFLOPS | GFLOPS per watt |
|---|---:|---:|---:|---:|
| 25 W | 989 MHz | 25.0 W | 1664 | 66.6 |
| 40 W | 1238 MHz | 40.0 W | 2087 | 52.2 |
| 60 W | 1428 MHz | 60.0 W | 2421 | 40.3 |
| 70 W | 1495 MHz | 70.0 W | 2556 | 36.5 |
| 80 W | 1546 MHz | 80.1 W | 2642 | 33.0 |
| none | 1620 MHz | 96.6 W | 2761 | 28.6 |

From 70 to 80 W the work rises 3.4% and the energy per unit of work 10.6%; the last 6.5 W buy 0.6%. So GPU joules per
token mostly report where on this curve a run sat. A path that keeps the GPU busier runs a higher clock, is faster and
reads worse per token. That is why GPU joules per token never judge a change (CLAUDE.md, Landing perf).

## macOS moves the ceiling on its own

Read from the driver's `MaxGPUAbsolutePower` and `FilteredGPUPower` properties (no root) and the GPU's P-state
residency, 2026-11-11, macOS 27.0 (26A428), main 47d434f0, DeepSeek-V4.1-Flash STQ 1.75bpw, `--ssd-budget-gb 108`, AC
power, fans at max, boot at 49 °C, GPU lock, quiet box:

- At idle the limit reads 181.8 W. During decode it read 67.9 W while the GPU drew 64 to 67 W: decode runs just under
  a ceiling macOS set, with the die at 80 to 83 °C.
- Decode spent 3 to 8% of its active time in the top clock state (1620 MHz); the mean active clock was 1540 to
  1560 MHz.
- Right after a 4k-token prefill the limit read 30.6 W once, and 181.8 W again at idle.

## One limit below the ceiling: 60 W

Two loads, the same requests at the same position in each; the 60 W limit was written at idle before the second load.
Greedy text and the seeded sampled text were the same bytes in both.

| request | no limit written: tok/s, GPU W, mean clock | 60 W: tok/s, GPU W, mean clock | tok/s | GPU W |
|---|---|---|---:|---:|
| greedy 384 tokens, first | 69.8, 66.5 W, 1561 MHz | 68.7, 60.2 W, 1521 MHz | -1.6% | -9.5% |
| greedy 384 tokens, second | 72.2, 64.4 W, 1539 MHz | 71.3, 60.0 W, 1512 MHz | -1.2% | -6.8% |
| sampled 256 tokens, seed 7 | 53.1, 63.9 W, 1559 MHz | 52.6, 59.7 W, 1531 MHz | -0.9% | -6.6% |
| cold prefill, 3893 tokens | 987, 55.7 W, 1537 MHz | 977, 50.9 W, 1494 MHz | -1.0% | -8.6% |

One load a side: a greedy cell moves about 1% between loads with nothing changed, so the tok/s column is at the edge of
what one load resolves. The watts column is not.

## A limit is a ceiling, not a setting that holds

- A limit written at idle landed and held through three decode requests. After the prefill request the driver read
  30.6 W: macOS had rewritten it.
- An 80 W limit written during decode, while the driver read 67.9 W, never read back. A 40 W limit written after macOS
  had rewritten the limit never read back either (60 s each). The same tool held every limit of its sweep on macOS 26.5.
- So a written limit can lower the ceiling for a while, cannot raise it, cannot pin a clock, and lasts until macOS
  next moves it. Another macOS build can behave differently.

## What to take from it

- Near the ceiling, about 1% of tok/s is worth 7 to 9% of GPU watts. A GPU-energy gap of that size between two builds
  is a clock position, inside tok/s noise: record it, never chase it.
- Top speed is set by a ceiling macOS moves with temperature, so a tok/s number carries its boot temperature, fan
  state and macOS build ([cool the box](process-measurement.md#4b-cool-the-box-before-a-bench)).
- What the engine controls is work: host CPU joules, instructions, dispatches, command buffers and bytes per token.
  Those are what a change is judged by.
- Not measured: limits of 40 W and 25 W (they did not land), long context, other models.
