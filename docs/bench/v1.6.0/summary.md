# Sushi 🍣 1.6.0 benchmarks

Version: **1.6.0**. Date: 2026-11-09/10 (Asia/Bangkok). The 1.4.0 gate, `sushi bench` (`./tests/bench.sh`): one cold prefill grown in 8k restore windows to 128k, three greedy decode samples at each rung, on a code arm (a seeded TypeScript corpus) and a prose arm (Pride and Prejudice), with the GPU clock read beside every request. The code arm decodes 512 tokens a sample (DeepSeek 256), the prose arm 128 on every model.

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power for every run, `iogpu.wired_limit_mb=125000`. ReleaseFast, `taskpolicy -a`, one model per server, GPU lock per model, fans at max and an idle under 50 °C before each server. Shipping defaults (kv8, the default speculative mode); DeepSeek runs `--ssd-budget-gb 108`. All rates are tok/s; prefill is the window ending at the rung. `·` = not measured.

| Sushi 🍣 1.6.0 | | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw**, code | Prefill | 1,425 | 2,205 | 2,114 | 2,119 | 2,049 | 1,979 |
|  | Decode | 168.1 | 164.2 | 138.0 | 155.4 | 166.8 | 134.1 |
| prose | Prefill | 1,987 | 2,190 | 2,134 | 2,162 | 2,050 | 2,022 |
|  | Decode | 76.9 | 101.5 | 91.7 | 102.0 | 86.1 | 69.1 |
| **Qwen3.8-Flash-Next-Sushi-4bpw**, code | Prefill | 2,424 | 2,417 | 2,222 | 2,241 | 2,220 | 2,080 |
|  | Decode | 146.2 | 159.8 | 132.6 | 150.2 | 151.9 | 127.0 |
| prose | Prefill | 1,956 | 2,275 | 2,222 | 2,147 | 2,194 | 2,011 |
|  | Decode | 79.5 | 98.5 | 85.8 | 89.7 | 97.5 | 78.8 |
| **GLM-5.3-Flash-Sushi-2.4bpw**, code | Prefill | 716 | 796 | 807 | 803 | 688 | 570 |
|  | Decode | 71.5 | 69.2 | 66.9 | 65.6 | 70.8 | 63.4 |
| prose | Prefill | 906 | 804 | 811 | 817 | 694 | 576 |
|  | Decode | 40.9 | 53.4 | 36.7 | 58.0 | 37.7 | 38.0 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw**, code | Prefill | 1,242 | 1,493 | 1,497 | 1,468 | 1,383 | 1,266 |
|  | Decode | 95.5 | 100.8 | 76.5 | 91.1 | 84.8 | 75.1 |
| prose | Prefill | 858 | 1,093 | 1,247 | 1,191 | 1,264 | 1,172 |
|  | Decode | 50.3 | 56.5 | 46.9 | 51.3 | 51.8 | 52.2 |

| Model run | Speculative mode | Commit | Binary SHA-256 |
|---|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw](Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw.md) | mtp | `ac3da7aa` | `fcdba9054a6ad870` |
| [Qwen3.8-Flash-Next-Sushi-4bpw](Qwen3.8-Flash-Next-Sushi-4bpw.md) | mtp | `ac3da7aa` | `fcdba9054a6ad870` |
| [GLM-5.3-Flash-Sushi-2.4bpw](GLM-5.3-Flash-Sushi-2.4bpw.md) | dflash | `ac3da7aa` | `fcdba9054a6ad870` |
| [DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw](DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw.md) | dspark | `ac3da7aa` | `fcdba9054a6ad870` |

Qwen STQ 2.75bpw runs the published pack revision 2.0 (6-bit STQ trunk), which 1.5.1's report predates. On that pack, a same-session A/B against the 1.5.1 binary (`9f8801bd`, two runs each, interleaved, 256 decode tokens) puts 1.6.0's prefill 2-6% lower from 4k to 40k (16k: 2,105 against 2,235) and level from 48k, with code decode about 2% lower; 1.5.1 repeated itself within 0.4%. The fix follows 1.6.0. Qwen 4bpw, on the same pack as 1.5.1, is level on prefill and 0.5-5% ahead on code decode. GLM ran under Heavy thermal pressure (GPU median 1,498 MHz) and finished every rung on both arms, level with or ahead of 1.5.1's code rungs. DeepSeek prefills 1-3% below 1.5.1 on code up to 64k and 4-9% below on prose up to 32k, ahead at 128k; decode is within ±3% on code. Raw reports and server logs are indexed in the private measurement ledger.
