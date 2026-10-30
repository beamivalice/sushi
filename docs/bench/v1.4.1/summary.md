# Sushi 🍣 1.4.1 benchmarks

Version: **1.4.1**. Date: 2026-10-30 (Asia/Bangkok). The 1.4.0 gate, `sushi bench` (`./tests/bench.sh`): one cold prefill grown in 8k restore windows to 128k, three greedy decode samples at each rung, on a code arm (a seeded TypeScript corpus) and a prose arm (Pride and Prejudice), with the GPU clock read beside every request. The code arm decodes 512 tokens a sample (DeepSeek 256), the prose arm 128 on every model. 1.4.0's Qwen and GLM prose rows decoded 512, so compare prose decode across the two releases on DeepSeek only.

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power for every run, `iogpu.wired_limit_mb=125000`. ReleaseFast, `taskpolicy -a`, one model per server, GPU lock per model, fans at max and an idle under 50 °C before each server. Shipping defaults (kv8, the default speculative mode); DeepSeek runs `--ssd-budget-gb 108`. All rates are tok/s; prefill is the window ending at the rung.

| Sushi 🍣 1.4.1 | | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw**, code | Prefill | 2,219 | 2,242 | 2,137 | 2,106 | 2,090 | 1,912 |
|  | Decode | 161.0 | 149.7 | 146.2 | 145.3 | 161.1 | 123.3 |
| prose | Prefill | 2,076 | 2,310 | 2,227 | 2,192 | 2,156 | 2,034 |
|  | Decode | 68.6 | 89.8 | 84.3 | 85.0 | 80.3 | 55.7 |
| **Qwen3.8-Flash-Next-Sushi-4bpw**, code | Prefill | 1,288 | 2,326 | 2,356 | 2,268 | 2,153 | 2,026 |
|  | Decode | 144.8 | 154.4 | 132.7 | 144.9 | 158.1 | 119.6 |
| prose | Prefill | 1,841 | 2,318 | 2,269 | 2,273 | 2,128 | 2,034 |
|  | Decode | 77.4 | 96.5 | 84.5 | 95.4 | 95.4 | 86.8 |
| **GLM-5.3-Flash-Sushi-2.4bpw**, code | Prefill | 710 | 796 | 819 | 804 | 657 | 515 |
|  | Decode | 70.5 | 69.8 | 66.1 | 63.5 | 64.4 | 55.5 |
| prose | Prefill | 892 | 826 | 831 | 817 | 683 | 522 |
|  | Decode | 41.0 | 53.4 | 36.3 | 57.6 | 36.0 | 34.0 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw**, code | Prefill | 807 | 1,390 | 1,428 | 1,408 | 1,352 | 1,217 |
|  | Decode | 79.5 | 91.0 | 78.3 | 90.6 | 84.6 | 72.0 |
| prose | Prefill | 703 | 1,079 | 1,199 | 1,199 | 1,193 | 1,061 |
|  | Decode | 51.4 | 56.9 | 47.8 | 49.0 | 49.6 | 46.6 |

| Model run | Speculative mode | Commit | Binary SHA-256 |
|---|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw](Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw.md) | mtp | `b358cd38` | `6944232d09a932d2` |
| [Qwen3.8-Flash-Next-Sushi-4bpw](Qwen3.8-Flash-Next-Sushi-4bpw.md) | mtp | `b358cd38` | `6944232d09a932d2` |
| [GLM-5.3-Flash-Sushi-2.4bpw](GLM-5.3-Flash-Sushi-2.4bpw.md) | dflash | `b358cd38` | `6944232d09a932d2` |
| [DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw](DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw.md) | dspark | `b358cd38` | `6944232d09a932d2` |

DeepSeek's prefill is up to 47% faster than 1.4.0 (code, 128k: 1,217 against 829) and its decode up to 13%, from the prefill kernels this release lands. GLM's code arm ran at heavy thermal pressure from 64k (GPU near 1,270 MHz against 1,490 in 1.4.0), which accounts for its 64k and 128k decode cells; its 128k code samples end at the loop stop's 390 tokens, as in 1.4.0. The Qwen 4bpw code 4k prefill is the server's first request. DeepSeek's other budgets are in [docs/usage.md](../../usage.md#deepseek-launch). Raw reports and server logs are indexed in the private measurement ledger.
