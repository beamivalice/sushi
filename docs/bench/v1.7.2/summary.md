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
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw**, code | Prefill | 965 | 1,635 | 1,635 | 1,705 | 1,586 |
|  | Decode | 95.1 | 93.1 | 84.6 | 101.5 | 91.2 |
| prose | Prefill | 812 | 1,321 | 1,246 | 1,303 | 1,288 |
|  | Decode | 58.2 | 58.7 | 54.6 | 60.0 | 51.2 |

| Model run | Speculative mode | Commit |
|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw](Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw.md) | mtp | `b6fa3752` |
| [Qwen3.8-Flash-Next-Sushi-STQ-4bpw](Qwen3.8-Flash-Next-Sushi-STQ-4bpw.md) | mtp | `b6fa3752` |
| [DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw](DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw.md), `--ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000` | dspark | `a0b5022b` (the 1.7.2 code) |

The three runs share one binary (SHA-256 `d00671283f9bdf82`). DeepSeek's 4k prefill is the server's first prefill; the
run before this one, void for one late kernel, read 1,495 tok/s there. GLM-5.3-Flash-Sushi-2.4bpw was not run for
1.7.2.
