# Qwen3.8-Flash-Next-Sushi-STQ-4bpw bench (Sushi 1.7.2)

Commit `b6fa3752`, binary SHA-256 `d00671283f9bdf82` (built 2026-11-20 23:09:11). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261120-231355`, 2026-11-20.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1597 MHz, minimum 1092 MHz over 279 busy samples; thermal pressure seen: Nominal.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 72 s between the arms (die 95 C to 50 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 196.1 | 189.6 | 191.3 | 191.3 | mtp | 1619 | Nominal | 70.3 | 81.4 | 86.8 | 81.4 | mtp | 1620 | Nominal | 0.43 |
| 8k | 158.2 | 167.1 | 159.3 | 159.3 | mtp | 1619 | Nominal | 102.3 | 80.3 | 104.3 | 102.3 | mtp | 1620 | Nominal | 0.64 |
| 16k | 127.8 | 127.0 | 125.2 | 127.0 | mtp | 1599 | Nominal | 88.5 | 90.8 | 87.9 | 88.5 | mtp | 1618 | Nominal | 0.70 |
| 32k | 159.0 | 169.9 | 167.6 | 167.6 | mtp | 1597 | Nominal | 94.0 | 103.1 | 107.2 | 103.1 | mtp | 1618 | Nominal | 0.62 |
| 64k | 165.3 | 177.6 | 182.5 | 177.6 | mtp | 1601 | Nominal | 92.2 | 99.2 | 92.4 | 92.4 | mtp | 1618 | Nominal | 0.52 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 2,357 | 0 | 1.80 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,309 | 35 | 1.90 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,228 | 58 | 3.82 | 1583 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,123 | 108 | 4.10 | 1469 | Nominal | n/a | n/a | n/a | 31.0 | 1461 |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,150 | 148 | 4.06 | 1553 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,091 | 146 | 4.16 | 1448 | Nominal | n/a | n/a | n/a | 30.8 | 1455 |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,105 | 158 | 4.14 | 1492 | Nominal | n/a | n/a | n/a | 32.7 | 1479 |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,100 | 177 | 4.18 | 1513 | Nominal | n/a | n/a | n/a | 33.1 | 1484 |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,079 | 191 | 4.24 | 1547 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 1,914 | 0 | 2.21 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,132 | 41 | 2.05 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,119 | 34 | 3.99 | 1617 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,005 | 101 | 4.31 | 1561 | Nominal | n/a | n/a | n/a | 35.5 | 1534 |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,115 | 131 | 4.11 | 1617 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,078 | 141 | 4.18 | 1564 | Nominal | n/a | n/a | n/a | 36.0 | 1540 |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,103 | 151 | 4.15 | 1602 | Nominal | n/a | n/a | n/a | 38.8 | 1570 |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,092 | 168 | 4.19 | 1583 | Nominal | n/a | n/a | n/a | 38.0 | 1554 |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,046 | 188 | 4.29 | 1585 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Decode energy per token

Energy per token (p = prefill, d = decode): CPU mJ, CPU ms and Minstr (millions of instructions) are this process's own work, cut where the prefill ends; the GPU cells are the whole machine's GPU energy over a request that is only a prefill, or a decode repeat (its restore and short prefill included), beside its mean GPU MHz. GPU energy per token rises with the GPU clock and is a record, not a quality or efficiency score.
CPU energy unavailable: the server is another process; --model loads it in this one.

Decode energy per token, code arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 390.8 | 1596 |
| 8k | n/a | n/a | n/a | 441.5 | 1602 |
| 16k | n/a | n/a | n/a | 524.2 | 1581 |
| 32k | n/a | n/a | n/a | 412.4 | 1569 |
| 64k | n/a | n/a | n/a | 389.8 | 1566 |

Decode energy per token, prose arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 700.5 | 1579 |
| 8k | n/a | n/a | n/a | 739.8 | 1581 |
| 16k | n/a | n/a | n/a | 670.9 | 1565 |
| 32k | n/a | n/a | n/a | 634.4 | 1530 |
| 64k | n/a | n/a | n/a | 655.4 | 1531 |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k |
|---|---|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-4bpw** | Prefill | 2,357 | 2,309 | 2,228 | 2,150 | 2,079 |
|  | Decode | 191.3 | 159.3 | 127.0 | 167.6 | 177.6 |
| **Qwen3.8-Flash-Next-Sushi-STQ-4bpw (prose)** | Prefill | 1,914 | 2,132 | 2,119 | 2,115 | 2,046 |
|  | Decode | 81.4 | 102.3 | 88.5 | 103.1 | 92.4 |
