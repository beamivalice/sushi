# Sushi 🍣 1.7.0 benchmarks

Version: **1.7.0**. Date: 2026-11-18 (Asia/Bangkok). The 1.4.0 gate, `sushi bench` (`./tests/bench.sh`): one cold prefill grown in 8k restore windows to 128k, three greedy decode samples at each rung, on a code arm (a seeded TypeScript corpus) and a prose arm (Pride and Prejudice), with the GPU clock read beside every request. The code arm decodes 256 tokens a sample, the prose arm 128.

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power for every run, `iogpu.wired_limit_mb=125000`. ReleaseFast, `taskpolicy -a`, one model per server, GPU lock per model, fans at max and an idle under 50 °C before each server. Shipping defaults: Sushi Cooler, kv8, the default speculative mode. All rates are tok/s; prefill is the window ending at the rung. `·` = not measured.

| Sushi 🍣 1.7.0 | | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-4bpw**, code | Prefill | 1,663 | 2,321 | 2,279 | 2,132 | 2,019 | 1,734 |
|  | Decode | 160.9 | 157.5 | 129.6 | 149.2 | 164.3 | 123.3 |
| prose | Prefill | 1,756 | 2,113 | 1,760 | 2,079 | 1,932 | 1,735 |
|  | Decode | 73.1 | 86.1 | 86.7 | 100.3 | 82.0 | 95.7 |

| Model run | Speculative mode | Commit | Binary SHA-256 |
|---|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-STQ-4bpw](Qwen3.8-Flash-Next-Sushi-STQ-4bpw.md) | mtp | `5e25d9e0` | `47c1b0c38d313f01` |

The STQ 4bpw pack replaces Sushi-4bpw (EXL3) as sushi's 4-bit Qwen pack; its run compiled no kernel inside a request
from 4k to 128k on a fresh pipeline manifest.

## Other 1.7.0 measurements

Not the full ladder above, so not comparable with its cells except where noted.

- **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw**, `--ssd-budget-gb 108`: `sushi bench --rungs 4k --code-only --tokens 256`,
  the ladder's 4k code arm, four servers in the order Sushi Cooler, MLX (`--ds41-mlx-engine`), MLX, Cooler, on `b039659c`
  (1.7.0's code before its last lint-only commit). Decode median 90.6 and 93.0 tok/s on Cooler, 95.0 and 94.0 on MLX, at
  3.49 accepted tokens a DSpark round on both engines (48.8 against 48.3 ms a round); 4k prefill 1,352 and 1,429 tok/s on
  Cooler, 1,293 and 1,332 on MLX. Peak resident memory 104 to 106 GiB.
- **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw**, 131k prose, the draft depth pinned at 6 so every round verifies 7 rows, four
  servers in the order Cooler, MLX, MLX, Cooler on `621cf78a`: 43.3 ms a round on Cooler against 45.9 on MLX, 41.1
  against 37.7 tok/s, at the same acceptance.
- GLM-5.3-Flash-Sushi-2.4bpw and Sushi-4bpw (EXL3) were not run for 1.7.0.
