# Sushi 🍣 1.4.0 benchmarks

Version: **1.4.0**. Date: 2026-10-27 (Asia/Bangkok). From this release the gate is `sushi bench` (`./tests/bench.sh`), not llmprobe: one cold prefill grown in 8k restore windows to 128k, three greedy decode samples at each rung (512 tokens; DeepSeek 256 on code and 128 on prose), on a code arm (a seeded TypeScript corpus) and a prose arm (Pride and Prejudice), with the GPU clock read beside every request. Its numbers are not comparable with the llmprobe reports of earlier releases.

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power for every run, `iogpu.wired_limit_mb=125000`. ReleaseFast, `taskpolicy -a`, one model per server, GPU lock per model, fans at max and an idle under 50 °C before each server. Shipping defaults (kv8, the default speculative mode); DeepSeek runs `--ssd-budget-gb 108`. All rates are tok/s; prefill is the window ending at the rung.

| Sushi 🍣 1.4.0 | | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw**, code | Prefill | 1,122 | 1,993 | 2,049 | 1,976 | 1,906 | 1,877 |
|  | Decode | 168.6 | 151.1 | 146.1 | 146.8 | 151.0 | 138.2 |
| prose | Prefill | 2,130 | 2,014 | 1,958 | 1,950 | 1,902 | 1,845 |
|  | Decode | 62.8 | 96.9 | 87.0 | 91.8 | 73.0 | 71.5 |
| **Qwen3.8-Flash-Next-Sushi-4bpw**, code | Prefill | 2,079 | 2,379 | 2,326 | 2,231 | 2,147 | 2,120 |
|  | Decode | 146.2 | 150.1 | 136.2 | 149.8 | 153.5 | 122.0 |
| prose | Prefill | 2,216 | 2,301 | 2,144 | 2,115 | 2,077 | 1,957 |
|  | Decode | 65.6 | 86.9 | 79.0 | 72.0 | 66.4 | 74.5 |
| **GLM-5.3-Flash-Sushi-2.4bpw**, code | Prefill | 928 | 775 | 814 | 805 | 688 | 573 |
|  | Decode | 71.5 | 69.4 | 67.3 | 65.8 | 71.8 | 63.8 |
| prose | Prefill | 878 | 759 | 752 | 777 | 646 | 536 |
|  | Decode | 50.9 | 38.4 | 34.7 | 66.4 | 33.5 | 45.7 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw**, code | Prefill | 707 | 1,024 | 998 | 993 | 950 | 829 |
|  | Decode | 76.6 | 84.2 | 76.2 | 85.3 | 78.1 | 63.7 |
| prose | Prefill | 580 | 821 | 913 | 895 | 852 | 744 |
|  | Decode | 49.3 | 53.7 | 46.5 | 48.7 | 46.2 | 43.5 |

| Model run | Speculative mode | Commit | Binary SHA-256 |
|---|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw](Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw.md) | mtp | `92237c07` | `fcb8ee2f7e2e0ede` |
| [Qwen3.8-Flash-Next-Sushi-4bpw](Qwen3.8-Flash-Next-Sushi-4bpw.md) | mtp | `9d4716e8` | `fcb8ee2f7e2e0ede` |
| [GLM-5.3-Flash-Sushi-2.4bpw](GLM-5.3-Flash-Sushi-2.4bpw.md) | dflash | `d361c70b` | `4d25c2d7ba7a5684` |
| [DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw](DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw.md) | dspark | `bf21aa5e` | `5027a704b968b7bf` |

GLM's code arm falls into a greedy repetition loop at 128k, which the server's loop stop ends at 390 tokens; those samples are timed decodes and noted in its report. The Qwen STQ 2.75 code 4k prefill is the server's first request. DeepSeek's budgets of 80 and 96 GB are in [docs/usage.md](../../usage.md#deepseek-launch). Raw reports and server logs are indexed in the private measurement ledger.
