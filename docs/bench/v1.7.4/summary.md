# Sushi 🍣 1.7.4 benchmarks

Version: **1.7.4**. Date: 2026-11-24. `sushi bench` (`./tests/bench.sh`): one cold prefill grown in 8k restore windows,
three greedy decode samples at each rung, on a code arm and a prose arm.

## M5 Ultra 256 GB: the release baseline from 1.7.4

Hardware: **Apple M5 Ultra, 256 GB unified memory**, AC power. ReleaseFast, `taskpolicy -a`, one model per server, GPU
lock per model, no cooldown (a Mac Studio does not throttle). One binary from `cb048670` (SHA-256 `064c86caced50c88`),
shipping defaults; ladder 4k to 128k; code arm 512 decoded tokens (DeepSeek 256), prose 128.

| Sushi 🍣 1.7.4, M5 Ultra | | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-4bpw**, code | Prefill | 3,757 | 3,629 | 3,587 | 3,574 | 3,462 | 3,096 |
|  | Decode | 260.3 | 226.1 | 202.1 | 240.7 | 250.6 | 214.0 |
| prose | Prefill | 3,600 | 3,474 | 3,498 | 3,477 | 3,423 | 2,787 |
|  | Decode | 96.5 | 135.6 | 113.6 | 128.2 | 121.1 | 128.5 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw**, code | Prefill | 2,645 | 2,760 | 2,961 | 3,035 | 2,854 | 2,439 |
|  | Decode | 128.2 | 165.9 | 123.8 | 151.8 | 142.2 | 119.1 |
| prose | Prefill | 2,105 | 2,170 | 2,403 | 2,465 | 2,315 | 2,062 |
|  | Decode | 69.9 | 90.2 | 73.4 | 77.8 | 70.0 | 78.0 |

| Model run | Speculative mode | Commit |
|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-STQ-4bpw](Qwen3.8-Flash-Next-Sushi-STQ-4bpw.md), `--tokens 512` | mtp | `cb048670` |
| [DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw](DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw.md) | dspark | `cb048670` |

## M5 Max 128 GB: its last release ladder (4k to 64k)

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power, `iogpu.wired_limit_mb=123000`. ReleaseFast,
`taskpolicy -a`, one model per server, GPU lock, fans at max and an idle under 50 °C before the boot.

| Sushi 🍣 1.7.4, M5 Max | | 4k | 8k | 16k | 32k | 64k |
|---|---|---:|---:|---:|---:|---:|
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw**, code | Prefill | 1,517 | 1,632 | 1,622 | 1,706 | 1,517 |
|  | Decode | 104.9 | 107.3 | 89.0 | 107.8 | 90.0 |
| prose | Prefill | 904 | 1,315 | 1,263 | 1,255 | 1,251 |
|  | Decode | 58.1 | 63.6 | 53.7 | 57.6 | 51.9 |

| Model run | Speculative mode | Commit |
|---|---|---|
| [DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw](DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw.md), default flags | dspark | `76e22371` |

Against [1.7.3](../v1.7.3/summary.md) (`fc2fd843`, which ran `--ctx-size 1000000 --max-tokens 128000`): the cold 4k code
prefill 1,135 to 1,517 tok/s, code decode +10% at 4k and 8k and +6.5% at 16k, level at 32k and 64k; the 8k windows
from 8k to 64k +1.0% on code and +0.7% on prose. One run on each side.
