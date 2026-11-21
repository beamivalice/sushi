# Sushi 🍣 1.7.3 benchmarks

Version: **1.7.3**. Date: 2026-11-21/22 (UTC). `sushi bench` (`./tests/bench.sh --rungs 4k,8k,16k,32k,64k`): one cold
prefill grown in 8k restore windows to 64k, three greedy decode samples at each rung, on a code arm and a prose arm.

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power for every run, `iogpu.wired_limit_mb=123000`. ReleaseFast,
`taskpolicy -a`, one model per server, GPU lock per model, fans at max and an idle under 50 °C before each boot.

| Sushi 🍣 1.7.3 | | 4k | 8k | 16k | 32k | 64k |
|---|---|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw**, code | Prefill | 2,440 | 2,247 | 2,250 | 2,110 | 2,096 |
|  | Decode | 190.6 | 176.7 | 143.3 | 186.6 | 180.7 |
| prose | Prefill | 2,128 | 2,253 | 2,257 | 2,174 | 2,123 |
|  | Decode | 78.7 | 94.2 | 102.8 | 93.3 | 87.7 |
| **Qwen3.8-Flash-Next-Sushi-STQ-4bpw**, code | Prefill | 2,385 | 2,244 | 2,156 | 2,157 | 2,066 |
|  | Decode | 192.5 | 156.7 | 133.7 | 167.5 | 186.2 |
| prose | Prefill | 2,002 | 2,147 | 2,125 | 2,112 | 2,026 |
|  | Decode | 83.7 | 97.8 | 85.5 | 102.7 | 92.9 |
| **GLM-5.3-Flash-Sushi-2.4bpw**, code | Prefill | 702 | 811 | 814 | 803 | 689 |
|  | Decode | 72.6 | 70.3 | 68.9 | 66.3 | 72.6 |
| prose | Prefill | 871 | 821 | 841 | 820 | 698 |
|  | Decode | 41.0 | 53.5 | 36.6 | 58.5 | 38.8 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw**, code | Prefill | 1,135 | 1,591 | 1,578 | 1,643 | 1,550 |
|  | Decode | 95.3 | 96.3 | 83.6 | 106.2 | 91.8 |
| prose | Prefill | 914 | 1,281 | 1,234 | 1,256 | 1,275 |
|  | Decode | 58.0 | 59.6 | 54.4 | 60.0 | 52.5 |

| Model run | Speculative mode | Commit |
|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw](Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw.md) | mtp | `fc2fd843` |
| [Qwen3.8-Flash-Next-Sushi-STQ-4bpw](Qwen3.8-Flash-Next-Sushi-STQ-4bpw.md) | mtp | `fc2fd843` |
| [GLM-5.3-Flash-Sushi-2.4bpw](GLM-5.3-Flash-Sushi-2.4bpw.md) | dflash | `fc2fd843` |
| [DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw](DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw.md), `--ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000` | dspark | `fc2fd843` |

The four runs share one binary (SHA-256 `eb0abb6db57ae23b`).
GLM-5.3-Flash-Sushi-2.4bpw was last run for 1.6.0 (1.7.0 and 1.7.2 skipped it). The STQ-4bpw run here is its second: the
first (same binary, GPU median 1,557 MHz against this run's 1,584) read 2,108 tok/s code prefill at 16k against 2,156 here,
inside the clock difference; every run's GPU median sat 2-2.5% under 1.7.2's, which the prefill rows follow.
