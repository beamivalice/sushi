# Qwen3.8-Flash-Next-Sushi-STQ-4bpw bench (Sushi 1.7.4)

Commit `cb048670`, binary SHA-256 `064c86caced50c88` (built 2026-11-24 16:46:32). Hardware: Apple M5 Ultra 256 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261124-164340`, 2026-11-24.

Server flags: defaults. Private SSD prefix cache, one model per server, no cooldown (a Mac Studio does not throttle).

GPU clock, sampled every 500 ms: median 1619 MHz, minimum 817 MHz over 308 busy samples; thermal pressure seen: Nominal.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 0 s between the arms (die not read).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 271.8 | 259.4 | 260.3 | 260.3 | mtp | 1620 | Nominal | 96.5 | 103.7 | 94.2 | 96.5 | mtp | 1620 | Nominal | 0.37 |
| 8k | 226.1 | 227.1 | 220.9 | 226.1 | mtp | 1619 | Nominal | 131.6 | 136.1 | 135.6 | 135.6 | mtp | 1573 | Nominal | 0.60 |
| 16k | 191.2 | 206.0 | 202.1 | 202.1 | mtp | 1620 | Nominal | 102.0 | 120.4 | 113.6 | 113.6 | mtp | 1617 | Nominal | 0.56 |
| 32k | 240.7 | 230.3 | 247.5 | 240.7 | mtp | 1620 | Nominal | 127.5 | 128.2 | 130.5 | 128.2 | mtp | 1599 | Nominal | 0.53 |
| 64k | 216.6 | 250.6 | 251.2 | 250.6 | mtp | 1619 | Nominal | 121.1 | 127.9 | 119.0 | 121.1 | mtp | 1619 | Nominal | 0.48 |
| 128k | 206.4 | 214.0 | 220.4 | 214.0 | mtp | 1620 | Nominal | 128.5 | 129.2 | 128.5 | 128.5 | mtp | 1619 | Nominal | 0.60 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 3,757 | 0 | 1.17 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 3,629 | 41 | 1.25 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 3,587 | 57 | 2.43 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 3,610 | 72 | 2.45 | 1619 | Nominal | n/a | n/a | n/a | 38.3 | 1588 |
| 24,545 to 32,768 | 24,545 | 8,223 | 3,574 | 67 | 2.46 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 3,533 | 133 | 2.56 | 1619 | Nominal | n/a | n/a | n/a | 38.7 | 1582 |
| 40,929 to 49,152 | 40,929 | 8,223 | 3,551 | 136 | 2.56 | 1619 | Nominal | n/a | n/a | n/a | 38.9 | 1574 |
| 49,121 to 57,344 | 49,121 | 8,223 | 3,512 | 161 | 2.61 | 1619 | Nominal | n/a | n/a | n/a | 39.0 | 1566 |
| 57,313 to 65,536 | 57,313 | 8,223 | 3,462 | 177 | 2.67 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 65,505 to 73,728 | 65,505 | 8,223 | 3,414 | 181 | 2.72 | 1619 | Nominal | n/a | n/a | n/a | 39.5 | 1564 |
| 73,697 to 81,920 | 73,697 | 8,223 | 3,425 | 206 | 2.73 | 1620 | Nominal | n/a | n/a | n/a | 39.7 | 1561 |
| 81,889 to 90,112 | 81,889 | 8,223 | 3,253 | 221 | 2.87 | 1616 | Nominal | n/a | n/a | n/a | 39.9 | 1557 |
| 90,081 to 98,304 | 90,081 | 8,223 | 3,245 | 238 | 2.91 | 1620 | Nominal | n/a | n/a | n/a | 40.2 | 1557 |
| 98,273 to 106,496 | 98,273 | 8,223 | 3,089 | 242 | 3.03 | 1615 | Nominal | n/a | n/a | n/a | 40.3 | 1553 |
| 106,465 to 114,688 | 106,465 | 8,223 | 2,990 | 271 | 3.16 | 1613 | Nominal | n/a | n/a | n/a | 40.5 | 1548 |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,967 | 251 | 3.17 | 1613 | Nominal | n/a | n/a | n/a | 40.8 | 1543 |
| 122,849 to 131,072 | 122,849 | 8,223 | 3,096 | 295 | 3.11 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 3,600 | 0 | 1.22 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 3,474 | 39 | 1.31 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 3,498 | 47 | 2.49 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 3,499 | 68 | 2.51 | 1618 | Nominal | n/a | n/a | n/a | 38.4 | 1588 |
| 24,545 to 32,768 | 24,545 | 8,223 | 3,477 | 67 | 2.53 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 3,456 | 114 | 2.60 | 1618 | Nominal | n/a | n/a | n/a | 38.7 | 1576 |
| 40,929 to 49,152 | 40,929 | 8,223 | 3,462 | 135 | 2.63 | 1619 | Nominal | n/a | n/a | n/a | 38.9 | 1576 |
| 49,121 to 57,344 | 49,121 | 8,223 | 3,430 | 151 | 2.66 | 1619 | Nominal | n/a | n/a | n/a | 39.1 | 1572 |
| 57,313 to 65,536 | 57,313 | 8,223 | 3,423 | 170 | 2.69 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 65,505 to 73,728 | 65,505 | 8,223 | 3,340 | 171 | 2.76 | 1618 | Nominal | n/a | n/a | n/a | 39.3 | 1558 |
| 73,697 to 81,920 | 73,697 | 8,223 | 3,329 | 206 | 2.82 | 1619 | Nominal | n/a | n/a | n/a | 39.4 | 1553 |
| 81,889 to 90,112 | 81,889 | 8,223 | 3,122 | 223 | 3.01 | 1615 | Nominal | n/a | n/a | n/a | 39.7 | 1552 |
| 90,081 to 98,304 | 90,081 | 8,223 | 3,013 | 206 | 3.08 | 1616 | Nominal | n/a | n/a | n/a | 40.0 | 1552 |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,841 | 264 | 3.31 | 1611 | Nominal | n/a | n/a | n/a | 40.2 | 1547 |
| 106,465 to 114,688 | 106,465 | 8,223 | 2,652 | 289 | 3.55 | 1604 | Nominal | n/a | n/a | n/a | 40.5 | 1545 |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,647 | 300 | 3.57 | 1606 | Nominal | n/a | n/a | n/a | 40.5 | 1542 |
| 122,849 to 131,072 | 122,849 | 8,223 | 2,787 | 323 | 3.45 | 1613 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Decode energy per token

Energy per token (p = prefill, d = decode): CPU mJ, CPU ms and Minstr (millions of instructions) are this process's own work, cut where the prefill ends; the GPU cells are the whole machine's GPU energy over a request that is only a prefill, or a decode repeat (its restore and short prefill included), beside its mean GPU MHz. GPU energy per token rises with the GPU clock and is a record, not a quality or efficiency score.
CPU energy unavailable: the server is another process; --model loads it in this one.

Decode energy per token, code arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 389.0 | 1603 |
| 8k | n/a | n/a | n/a | 440.2 | 1602 |
| 16k | n/a | n/a | n/a | 469.6 | 1600 |
| 32k | n/a | n/a | n/a | 427.3 | 1584 |
| 64k | n/a | n/a | n/a | 409.6 | 1567 |
| 128k | n/a | n/a | n/a | 472.7 | 1561 |

Decode energy per token, prose arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 910.1 | 1598 |
| 8k | n/a | n/a | n/a | 636.1 | 1578 |
| 16k | n/a | n/a | n/a | 719.3 | 1566 |
| 32k | n/a | n/a | n/a | 747.2 | 1550 |
| 64k | n/a | n/a | n/a | 678.0 | 1538 |
| 128k | n/a | n/a | n/a | 688.3 | 1487 |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-4bpw** | Prefill | 3,757 | 3,629 | 3,587 | 3,574 | 3,462 | 3,096 |
|  | Decode | 260.3 | 226.1 | 202.1 | 240.7 | 250.6 | 214.0 |
| **Qwen3.8-Flash-Next-Sushi-STQ-4bpw (prose)** | Prefill | 3,600 | 3,474 | 3,498 | 3,477 | 3,423 | 2,787 |
|  | Decode | 96.5 | 135.6 | 113.6 | 128.2 | 121.1 | 128.5 |
