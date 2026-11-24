# Sushi 🍣 1.7.4 benchmarks

Version: **1.7.4**. Date: 2026-11-24. `sushi bench` (`./tests/bench.sh --rungs 4k,8k,16k,32k,64k`): one cold
prefill grown in 8k restore windows to 64k, three greedy decode samples at each rung, on a code arm and a prose arm.

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power, `iogpu.wired_limit_mb=123000`. ReleaseFast,
`taskpolicy -a`, one model per server, GPU lock, fans at max and an idle under 50 °C before the boot.

| Sushi 🍣 1.7.4 | | 4k | 8k | 16k | 32k | 64k |
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

This is the M5 Max's last release ladder. From 1.7.4 the release baseline is the M5 Ultra on the STQ packs only (Qwen
STQ 4bpw, DeepSeek STQ 2.7bpw; Hy4 STQ 4bpw from 1.8.0): their reports join this directory.
