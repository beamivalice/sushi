# Sushi 🍣 1.8.0 benchmarks

Version: **1.8.0**. Date: 2026-11-26. `sushi bench` (`./tests/bench.sh`): one cold prefill grown in 8k restore windows,
three greedy decode samples at each rung, on a code arm and a prose arm.

## M5 Ultra 256 GB

Hardware: **Apple M5 Ultra, 256 GB unified memory**, AC power. ReleaseFast, `taskpolicy -a`, one model per server, GPU
lock per model, no cooldown (a Mac Studio does not throttle). One binary from `7c8d11d1` (SHA-256 `66e39713d569f967`),
shipping defaults; ladder 4k to 128k; code arm 512 decoded tokens for Qwen (DeepSeek and Hy4-Flash 256), prose 128.

| Sushi 🍣 1.8.0, M5 Ultra | | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-4bpw**, code | Prefill | 3,860 | 3,702 | 3,677 | 3,651 | 3,443 | 2,974 |
|  | Decode | 270.0 | 231.1 | 205.5 | 251.3 | 256.9 | 212.6 |
| prose | Prefill | 3,693 | 3,531 | 3,562 | 3,568 | 3,375 | 2,925 |
|  | Decode | 102.2 | 138.5 | 120.2 | 139.8 | 122.0 | 127.6 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw**, code | Prefill | 2,643 | 2,771 | 2,983 | 3,044 | 2,873 | 2,453 |
|  | Decode | 129.0 | 167.0 | 124.6 | 152.9 | 143.2 | 120.0 |
| prose | Prefill | 2,078 | 2,167 | 2,356 | 2,383 | 2,194 | 1,976 |
|  | Decode | 70.6 | 90.8 | 74.2 | 78.4 | 71.0 | 78.7 |
| **Hy4-Flash-Sushi-STQ-4bpw**, code | Prefill | 1,929 | 1,939 | 1,904 | 1,837 | 1,713 | 1,510 |
|  | Decode | 143.2 | 115.0 | 116.3 | 102.3 | 74.5 | 96.2 |
| prose | Prefill | 1,678 | 1,708 | 1,806 | 1,765 | 1,693 | 1,508 |
|  | Decode | 69.6 | 89.6 | 67.7 | 70.9 | 64.7 | 61.6 |

| Model run | Speculative mode | Commit |
|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-STQ-4bpw](Qwen3.8-Flash-Next-Sushi-STQ-4bpw.md), `--tokens 512` | mtp | `7c8d11d1` |
| [DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw](DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw.md) | dspark | `7c8d11d1` |
| [Hy4-Flash-Sushi-STQ-4bpw](Hy4-Flash-Sushi-STQ-4bpw.md) | dfly2 | `7c8d11d1` |

Against [1.7.4](../v1.7.4/summary.md) (`cb048670`): Qwen STQ 4bpw code decode +1.7% to +4.4% from 4k to 64k and level
at 128k; DeepSeek STQ 2.7bpw code decode +0.6% to +0.8% at every rung. Hy4-Flash STQ 4bpw is new in this release. One
run on each side.
