# Sushi 🍣 1.5.1 benchmarks

Version: **1.5.1**. Date: 2026-11-06 (Asia/Bangkok). The 1.4.0 gate, `sushi bench` (`./tests/bench.sh`): one cold prefill grown in 8k restore windows to 128k, three greedy decode samples at each rung, on a code arm (a seeded TypeScript corpus) and a prose arm (Pride and Prejudice), with the GPU clock read beside every request. The code arm decodes 512 tokens a sample (DeepSeek 256), the prose arm 128 on every model.

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power for every run, `iogpu.wired_limit_mb=125000`. ReleaseFast, `taskpolicy -a`, one model per server, GPU lock per model, fans at max and an idle under 50 °C before each server. Shipping defaults (kv8, the default speculative mode); DeepSeek runs `--ssd-budget-gb 108`. All rates are tok/s; prefill is the window ending at the rung. `·` = not measured.

| Sushi 🍣 1.5.1 | | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw**, code | Prefill | 1,878 | 2,242 | 2,282 | 2,162 | 2,078 | 1,897 |
|  | Decode | 175.0 | 163.7 | 147.4 | 151.3 | 156.5 | 135.0 |
| prose | Prefill | 2,092 | 2,328 | 2,326 | 2,146 | 2,163 | 2,045 |
|  | Decode | 73.0 | 99.6 | 94.5 | 93.0 | 87.2 | 73.0 |
| **Qwen3.8-Flash-Next-Sushi-4bpw**, code | Prefill | 2,454 | 2,370 | 2,304 | 2,241 | 2,149 | 1,947 |
|  | Decode | 145.0 | 152.1 | 132.0 | 147.5 | 150.5 | 123.2 |
| prose | Prefill | 1,993 | 2,320 | 2,251 | 2,189 | 2,075 | 2,036 |
|  | Decode | 79.9 | 90.8 | 87.1 | 97.5 | 92.9 | 86.7 |
| **GLM-5.3-Flash-Sushi-2.4bpw**, code | Prefill | 710 | 785 | 819 | 803 | 648 | · |
|  | Decode | 70.4 | 69.4 | 66.0 | 63.6 | 63.1 | · |
| prose | Prefill | · | · | · | · | · | · |
|  | Decode | · | · | · | · | · | · |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw**, code | Prefill | 1,263 | 1,522 | 1,538 | 1,494 | 1,403 | 1,186 |
|  | Decode | 98.2 | 102.2 | 78.7 | 88.0 | 83.6 | 71.8 |
| prose | Prefill | 937 | 1,201 | 1,298 | 1,272 | 1,255 | 1,075 |
|  | Decode | 56.4 | 59.3 | 52.2 | 51.7 | 50.1 | 48.1 |

| Model run | Speculative mode | Commit | Binary SHA-256 |
|---|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw](Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw.md) | mtp | `399895f6` | `4391a90f528f7aaa` |
| [Qwen3.8-Flash-Next-Sushi-4bpw](Qwen3.8-Flash-Next-Sushi-4bpw.md) | mtp | `399895f6` | `4391a90f528f7aaa` |
| [GLM-5.3-Flash-Sushi-2.4bpw](GLM-5.3-Flash-Sushi-2.4bpw.md) | dflash | `399895f6` | `4391a90f528f7aaa` |
| [DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw](DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw.md) | dspark | `399895f6` | `4391a90f528f7aaa` |

Qwen STQ 2.75bpw prefills 4-83% faster than 1.5.0 (4k: 1,878 against 1,026) and decodes 1-9% faster on code. GLM throttled on both runs (Heavy pressure, under 350 MHz), so it is released on the 4k-64k code rungs of the first run; per MHz its 64k decode matches 1.5.0. DeepSeek ran the published pack revision 2.0, which 1.5.0's report predates, so its gains (code 4k decode 98.2 against 80.9, prefill up to 52%) mix the release with the pack. The 1.5.1 tag adds one menu-bar wording change over `399895f6`; the sushi binary's code is the same. Raw reports and server logs are indexed in the private measurement ledger.
