# Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw bench (Sushi 1.7.3)

Commit `fc2fd843`, binary SHA-256 `eb0abb6db57ae23b` (built 2026-11-22 05:07:29). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261122-050735`, 2026-11-22.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1575 MHz, minimum 1167 MHz over 266 busy samples; thermal pressure seen: Nominal.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 61 s between the arms (die 91 C to 49 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 190.6 | 185.7 | 191.3 | 190.6 | mtp | 1619 | Nominal | 73.3 | 78.9 | 78.7 | 78.7 | mtp | 1615 | Nominal | 0.41 |
| 8k | 183.7 | 176.7 | 175.5 | 176.7 | mtp | 1620 | Nominal | 90.2 | 101.0 | 94.2 | 94.2 | mtp | 1620 | Nominal | 0.53 |
| 16k | 163.4 | 142.1 | 143.3 | 143.3 | mtp | 1584 | Nominal | 100.0 | 102.8 | 104.4 | 102.8 | mtp | 1619 | Nominal | 0.72 |
| 32k | 162.1 | 189.2 | 186.6 | 186.6 | mtp | 1564 | Nominal | 85.2 | 93.3 | 94.8 | 93.3 | mtp | 1620 | Nominal | 0.50 |
| 64k | 179.7 | 180.7 | 185.2 | 180.7 | mtp | 1572 | Nominal | 80.1 | 87.7 | 100.9 | 87.7 | mtp | 1620 | Nominal | 0.49 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 2,440 | 0 | 1.75 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,247 | 36 | 1.95 | 1560 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,250 | 50 | 3.77 | 1576 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,159 | 86 | 4.01 | 1470 | Nominal | n/a | n/a | n/a | 30.2 | 1473 |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,110 | 86 | 4.05 | 1473 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,082 | 99 | 4.12 | 1411 | Nominal | n/a | n/a | n/a | 28.5 | 1418 |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,125 | 145 | 4.09 | 1466 | Nominal | n/a | n/a | n/a | 31.9 | 1470 |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,111 | 161 | 4.14 | 1490 | Nominal | n/a | n/a | n/a | 32.4 | 1479 |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,096 | 181 | 4.19 | 1521 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 2,128 | 0 | 1.99 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,253 | 28 | 1.93 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,257 | 49 | 3.76 | 1618 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,239 | 75 | 3.85 | 1596 | Nominal | n/a | n/a | n/a | 36.6 | 1562 |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,174 | 93 | 3.95 | 1561 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,215 | 93 | 3.88 | 1601 | Nominal | n/a | n/a | n/a | 36.9 | 1556 |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,121 | 141 | 4.11 | 1489 | Nominal | n/a | n/a | n/a | 31.9 | 1476 |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,128 | 156 | 4.11 | 1523 | Nominal | n/a | n/a | n/a | 32.9 | 1494 |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,123 | 178 | 4.14 | 1541 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Decode energy per token

Energy per token (p = prefill, d = decode): CPU mJ, CPU ms and Minstr (millions of instructions) are this process's own work, cut where the prefill ends; the GPU cells are the whole machine's GPU energy over a request that is only a prefill, or a decode repeat (its restore and short prefill included), beside its mean GPU MHz. GPU energy per token rises with the GPU clock and is a record, not a quality or efficiency score.
CPU energy unavailable: the server is another process; --model loads it in this one.

Decode energy per token, code arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 397.7 | 1599 |
| 8k | n/a | n/a | n/a | 420.2 | 1601 |
| 16k | n/a | n/a | n/a | 467.1 | 1569 |
| 32k | n/a | n/a | n/a | 377.0 | 1561 |
| 64k | n/a | n/a | n/a | 388.0 | 1548 |

Decode energy per token, prose arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 801.5 | 1584 |
| 8k | n/a | n/a | n/a | 691.6 | 1586 |
| 16k | n/a | n/a | n/a | 628.9 | 1573 |
| 32k | n/a | n/a | n/a | 722.9 | 1552 |
| 64k | n/a | n/a | n/a | 693.7 | 1535 |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k |
|---|---|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw** | Prefill | 2,440 | 2,247 | 2,250 | 2,110 | 2,096 |
|  | Decode | 190.6 | 176.7 | 143.3 | 186.6 | 180.7 |
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw (prose)** | Prefill | 2,128 | 2,253 | 2,257 | 2,174 | 2,123 |
|  | Decode | 78.7 | 94.2 | 102.8 | 93.3 | 87.7 |
