# Qwen3.8-Flash-Next-Sushi-STQ-4bpw bench (Sushi 1.7.3)

Commit `fc2fd843`, binary SHA-256 `eb0abb6db57ae23b` (built 2026-11-22 05:07:29). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261122-055758`, 2026-11-22.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1584 MHz, minimum 569 MHz over 280 busy samples; thermal pressure seen: Nominal.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 71 s between the arms (die 93 C to 50 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 192.5 | 189.1 | 197.0 | 192.5 | mtp | 1620 | Nominal | 71.8 | 83.7 | 85.8 | 83.7 | mtp | 1619 | Nominal | 0.43 |
| 8k | 161.2 | 156.6 | 156.7 | 156.7 | mtp | 1620 | Nominal | 97.8 | 96.6 | 102.3 | 97.8 | mtp | 1620 | Nominal | 0.62 |
| 16k | 133.7 | 132.4 | 138.3 | 133.7 | mtp | 1614 | Nominal | 80.3 | 85.5 | 92.9 | 85.5 | mtp | 1619 | Nominal | 0.64 |
| 32k | 167.5 | 166.7 | 168.0 | 167.5 | mtp | 1537 | Nominal | 102.7 | 106.1 | 101.0 | 102.7 | mtp | 1619 | Nominal | 0.61 |
| 64k | 156.3 | 186.2 | 188.8 | 186.2 | mtp | 1557 | Nominal | 91.1 | 98.8 | 92.9 | 92.9 | mtp | 1614 | Nominal | 0.50 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 2,385 | 0 | 1.79 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,244 | 30 | 1.95 | 1604 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,156 | 39 | 3.93 | 1526 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,110 | 63 | 4.06 | 1419 | Nominal | n/a | n/a | n/a | 28.9 | 1429 |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,157 | 95 | 3.99 | 1524 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,068 | 111 | 4.17 | 1412 | Nominal | n/a | n/a | n/a | 29.3 | 1417 |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,090 | 132 | 4.15 | 1462 | Nominal | n/a | n/a | n/a | 31.5 | 1457 |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,077 | 155 | 4.19 | 1470 | Nominal | n/a | n/a | n/a | 31.9 | 1452 |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,066 | 168 | 4.22 | 1501 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 2,002 | 0 | 2.10 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,147 | 20 | 2.00 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,125 | 35 | 3.98 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,091 | 63 | 4.09 | 1570 | Nominal | n/a | n/a | n/a | 35.6 | 1547 |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,112 | 95 | 4.06 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,100 | 91 | 4.09 | 1619 | Nominal | n/a | n/a | n/a | 38.6 | 1577 |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,096 | 134 | 4.15 | 1620 | Nominal | n/a | n/a | n/a | 39.1 | 1573 |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,078 | 153 | 4.20 | 1598 | Nominal | n/a | n/a | n/a | 38.7 | 1564 |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,026 | 175 | 4.56 | 1574 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Decode energy per token

Energy per token (p = prefill, d = decode): CPU mJ, CPU ms and Minstr (millions of instructions) are this process's own work, cut where the prefill ends; the GPU cells are the whole machine's GPU energy over a request that is only a prefill, or a decode repeat (its restore and short prefill included), beside its mean GPU MHz. GPU energy per token rises with the GPU clock and is a record, not a quality or efficiency score.
CPU energy unavailable: the server is another process; --model loads it in this one.

Decode energy per token, code arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 370.0 | 1596 |
| 8k | n/a | n/a | n/a | 454.4 | 1605 |
| 16k | n/a | n/a | n/a | 504.1 | 1592 |
| 32k | n/a | n/a | n/a | 382.6 | 1540 |
| 64k | n/a | n/a | n/a | 367.9 | 1548 |

Decode energy per token, prose arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 673.8 | 1584 |
| 8k | n/a | n/a | n/a | 626.4 | 1584 |
| 16k | n/a | n/a | n/a | 708.8 | 1579 |
| 32k | n/a | n/a | n/a | 636.7 | 1548 |
| 64k | n/a | n/a | n/a | 654.9 | 1544 |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k |
|---|---|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-4bpw** | Prefill | 2,385 | 2,244 | 2,156 | 2,157 | 2,066 |
|  | Decode | 192.5 | 156.7 | 133.7 | 167.5 | 186.2 |
| **Qwen3.8-Flash-Next-Sushi-STQ-4bpw (prose)** | Prefill | 2,002 | 2,147 | 2,125 | 2,112 | 2,026 |
|  | Decode | 83.7 | 97.8 | 85.5 | 102.7 | 92.9 |
