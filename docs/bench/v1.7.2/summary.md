# Sushi 🍣 1.7.2 benchmarks

Version: **1.7.2**. Date: 2026-11-20 (UTC). `sushi bench` (`./tests/bench.sh --rungs 4k,8k,16k,32k,64k`): one cold
prefill grown in 8k restore windows to 64k, three greedy decode samples at each rung, on a code arm and a prose arm.

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power for every run, `iogpu.wired_limit_mb=123000`. ReleaseFast,
`taskpolicy -a`, one model per server, GPU lock per model, fans at max and an idle under 50 °C before each boot.

| Sushi 🍣 1.7.2 | | 4k | 8k | 16k | 32k | 64k |
|---|---|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw**, code | Prefill | 2,410 | 2,311 | 2,300 | 2,119 | 2,122 |
|  | Decode | 193.4 | 178.5 | 141.9 | 176.3 | 182.3 |
| prose | Prefill | 2,154 | 2,253 | 2,275 | 2,200 | 2,151 |
|  | Decode | 81.6 | 93.6 | 102.0 | 77.9 | 93.8 |
| **Qwen3.8-Flash-Next-Sushi-STQ-4bpw**, code | Prefill | 2,357 | 2,309 | 2,228 | 2,150 | 2,079 |
|  | Decode | 191.3 | 159.3 | 127.0 | 167.6 | 177.6 |
| prose | Prefill | 1,914 | 2,132 | 2,119 | 2,115 | 2,046 |
|  | Decode | 81.4 | 102.3 | 88.5 | 103.1 | 92.4 |

| Model run | Speculative mode | Commit |
|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw](Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw.md) | mtp | `b6fa3752` |
| [Qwen3.8-Flash-Next-Sushi-STQ-4bpw](Qwen3.8-Flash-Next-Sushi-STQ-4bpw.md) | mtp | `b6fa3752` |

DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw follows in this folder once its run is done. GLM-5.3-Flash-Sushi-2.4bpw was
not run for 1.7.2.
