# Sushi 🍣 1.5.0 benchmarks

Version: **1.5.0**. Date: 2026-11-04 (Asia/Bangkok). The 1.4.0 gate, `sushi bench` (`./tests/bench.sh`): one cold prefill grown in 8k restore windows to 128k, three greedy decode samples at each rung, on a code arm (a seeded TypeScript corpus) and a prose arm (Pride and Prejudice), with the GPU clock read beside every request. The code arm decodes 512 tokens a sample (DeepSeek 256), the prose arm 128 on every model, as in 1.4.1.

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power for every run, `iogpu.wired_limit_mb=125000`. ReleaseFast, `taskpolicy -a`, one model per server, GPU lock per model, fans at max and an idle under 50 °C before each server. Shipping defaults (kv8, the default speculative mode); DeepSeek runs `--ssd-budget-gb 108`. All rates are tok/s; prefill is the window ending at the rung.

| Sushi 🍣 1.5.0 | | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw**, code | Prefill | 1,026 | 2,023 | 2,000 | 1,949 | 1,890 | 1,821 |
|  | Decode | 166.9 | 149.6 | 137.2 | 149.4 | 151.1 | 133.1 |
| prose | Prefill | 1,864 | 2,117 | 2,002 | 1,903 | 1,839 | 1,849 |
|  | Decode | 68.2 | 82.3 | 88.4 | 76.4 | 76.5 | 66.9 |
| **Qwen3.8-Flash-Next-Sushi-4bpw**, code | Prefill | 1,964 | 2,435 | 2,334 | 2,248 | 2,143 | 1,965 |
|  | Decode | 142.7 | 152.3 | 130.1 | 139.6 | 148.5 | 114.4 |
| prose | Prefill | 1,907 | 2,282 | 2,272 | 2,244 | 2,043 | 1,929 |
|  | Decode | 77.1 | 89.6 | 86.9 | 84.1 | 91.0 | 86.9 |
| **GLM-5.3-Flash-Sushi-2.4bpw**, code | Prefill | 751 | 796 | 822 | 813 | 686 | 562 |
|  | Decode | 70.7 | 70.3 | 67.0 | 65.3 | 70.8 | 62.5 |
| prose | Prefill | 876 | 813 | 821 | 800 | 679 | 556 |
|  | Decode | 40.5 | 53.0 | 36.4 | 57.0 | 37.0 | 36.6 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw**, code | Prefill | 832 | 1,420 | 1,439 | 1,387 | 1,358 | 1,216 |
|  | Decode | 80.9 | 97.9 | 77.2 | 88.1 | 86.1 | 76.1 |
| prose | Prefill | 884 | 1,151 | 1,247 | 1,223 | 1,202 | 1,101 |
|  | Decode | 52.8 | 57.9 | 52.4 | 52.4 | 51.4 | 50.3 |

| Model run | Speculative mode | Commit | Binary SHA-256 |
|---|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw](Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw.md) | mtp | `75996eda` | `62184462b5e6c176` |
| [Qwen3.8-Flash-Next-Sushi-4bpw](Qwen3.8-Flash-Next-Sushi-4bpw.md) | mtp | `75996eda` | `62184462b5e6c176` |
| [GLM-5.3-Flash-Sushi-2.4bpw](GLM-5.3-Flash-Sushi-2.4bpw.md) | dflash | `75996eda` | `62184462b5e6c176` |
| [DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw](DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw.md) | dspark | `75996eda` | `62184462b5e6c176` |

DeepSeek decodes up to 10% faster than 1.4.1 on prose (16k: 52.4 against 47.8) and GLM up to 13% on code at 128k, with prefill level or ahead on both. Qwen STQ 2.75bpw prefills 5-15% below the 1.4.1 report; a same-session A/B of the 1.4.1 and 1.5.0 binaries on that pack measured 1.5.0 level or ahead (code 24k-32k window 2,041 against 1,941; prose 2,066 against 2,046), so the gap is between sessions, not releases. Raw reports and server logs are indexed in the private measurement ledger.
