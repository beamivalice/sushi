# Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw bench (Sushi 1.7.2)

Commit `b6fa3752`, binary SHA-256 `d00671283f9bdf82` (built 2026-11-20 23:09:11). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261120-230631`, 2026-11-20.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1603 MHz, minimum 1134 MHz over 266 busy samples; thermal pressure seen: Nominal.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 72 s between the arms (die 96 C to 49 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 193.4 | 192.6 | 204.5 | 193.4 | mtp | 1620 | Nominal | 73.6 | 83.6 | 81.6 | 81.6 | mtp | 1620 | Nominal | 0.42 |
| 8k | 184.8 | 178.5 | 178.4 | 178.5 | mtp | 1620 | Nominal | 90.1 | 101.5 | 93.6 | 93.6 | mtp | 1620 | Nominal | 0.52 |
| 16k | 163.6 | 141.9 | 136.9 | 141.9 | mtp | 1611 | Nominal | 102.0 | 102.7 | 100.9 | 102.0 | mtp | 1620 | Nominal | 0.72 |
| 32k | 157.9 | 176.3 | 183.2 | 176.3 | mtp | 1614 | Nominal | 77.8 | 85.3 | 77.9 | 77.9 | mtp | 1620 | Nominal | 0.44 |
| 64k | 166.4 | 182.3 | 184.9 | 182.3 | mtp | 1607 | Nominal | 81.4 | 93.8 | 97.3 | 93.8 | mtp | 1620 | Nominal | 0.51 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 2,410 | 0 | 1.78 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,311 | 47 | 1.90 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,300 | 67 | 3.73 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,140 | 67 | 4.02 | 1459 | Nominal | n/a | n/a | n/a | 30.1 | 1458 |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,119 | 89 | 4.04 | 1510 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,186 | 137 | 3.97 | 1530 | Nominal | n/a | n/a | n/a | 34.2 | 1519 |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,121 | 169 | 4.12 | 1477 | Nominal | n/a | n/a | n/a | 31.7 | 1471 |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,132 | 179 | 4.12 | 1504 | Nominal | n/a | n/a | n/a | 33.1 | 1489 |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,122 | 204 | 4.17 | 1565 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 2,154 | 0 | 1.97 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,253 | 45 | 1.95 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,275 | 58 | 3.74 | 1617 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,207 | 106 | 3.94 | 1533 | Nominal | n/a | n/a | n/a | 34.6 | 1537 |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,200 | 79 | 3.89 | 1600 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,231 | 115 | 3.89 | 1614 | Nominal | n/a | n/a | n/a | 37.8 | 1561 |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,157 | 169 | 4.08 | 1532 | Nominal | n/a | n/a | n/a | 34.2 | 1519 |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,166 | 177 | 4.08 | 1556 | Nominal | n/a | n/a | n/a | 35.3 | 1533 |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,151 | 194 | 4.11 | 1583 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Decode energy per token

Energy per token (p = prefill, d = decode): CPU mJ, CPU ms and Minstr (millions of instructions) are this process's own work, cut where the prefill ends; the GPU cells are the whole machine's GPU energy over a request that is only a prefill, or a decode repeat (its restore and short prefill included), beside its mean GPU MHz. GPU energy per token rises with the GPU clock and is a record, not a quality or efficiency score.
CPU energy unavailable: the server is another process; --model loads it in this one.

Decode energy per token, code arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 390.9 | 1597 |
| 8k | n/a | n/a | n/a | 420.4 | 1601 |
| 16k | n/a | n/a | n/a | 498.9 | 1585 |
| 32k | n/a | n/a | n/a | 408.8 | 1577 |
| 64k | n/a | n/a | n/a | 411.5 | 1573 |

Decode energy per token, prose arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 716.5 | 1580 |
| 8k | n/a | n/a | n/a | 696.7 | 1583 |
| 16k | n/a | n/a | n/a | 682.8 | 1560 |
| 32k | n/a | n/a | n/a | 889.9 | 1559 |
| 64k | n/a | n/a | n/a | 672.8 | 1541 |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k |
|---|---|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw** | Prefill | 2,410 | 2,311 | 2,300 | 2,119 | 2,122 |
|  | Decode | 193.4 | 178.5 | 141.9 | 176.3 | 182.3 |
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw (prose)** | Prefill | 2,154 | 2,253 | 2,275 | 2,200 | 2,151 |
|  | Decode | 81.6 | 93.6 | 102.0 | 77.9 | 93.8 |
