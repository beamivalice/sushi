# DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw bench (Sushi 1.8.0)

Commit `7c8d11d1`, binary SHA-256 `66e39713d569f967` (built 2026-11-26 19:12:21). Hardware: Apple M5 Ultra 256 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261126-194246`, 2026-11-26.

Server flags: defaults. Private SSD prefix cache, one model per server, no cooldown (a Mac Studio does not throttle).

GPU clock, sampled every 500 ms: median 1619 MHz, minimum 1040 MHz over 353 busy samples; thermal pressure seen: Nominal, Moderate.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 256 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 0 s between the arms (die not read).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 127.2 | 129.0 | 129.0 | 129.0 | dspark | 1620 | Nominal | 70.4 | 70.6 | 70.7 | 70.6 | dspark | 1620 | Moderate | 0.55 |
| 8k | 166.3 | 167.2 | 167.0 | 167.0 | dspark | 1620 | Nominal | 90.1 | 90.9 | 90.8 | 90.8 | dspark | 1620 | Moderate | 0.54 |
| 16k | 123.3 | 124.6 | 124.6 | 124.6 | dspark | 1620 | Nominal | 74.0 | 74.2 | 74.4 | 74.2 | dspark | 1620 | Moderate | 0.60 |
| 32k | 151.9 | 152.9 | 153.0 | 152.9 | dspark | 1620 | Nominal | 78.1 | 78.4 | 78.4 | 78.4 | dspark | 1620 | Moderate | 0.51 |
| 64k | 142.1 | 143.2 | 143.2 | 143.2 | dspark | 1620 | Moderate | 71.0 | 71.0 | 71.1 | 71.0 | dspark | 1620 | Moderate | 0.50 |
| 128k | 118.2 | 120.0 | 120.0 | 120.0 | dspark | 1620 | Moderate | 78.5 | 78.7 | 78.7 | 78.7 | dspark | 1620 | Moderate | 0.66 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 2,643 | 0 | 1.60 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,771 | 11 | 1.54 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,983 | 13 | 2.82 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,960 | 15 | 2.84 | 1620 | Nominal | n/a | n/a | n/a | 54.0 | 1604 |
| 24,577 to 32,768 | 24,577 | 8,191 | 3,044 | 20 | 2.76 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,876 | 17 | 2.92 | 1620 | Moderate | n/a | n/a | n/a | 55.5 | 1605 |
| 40,961 to 49,152 | 40,961 | 8,191 | 2,960 | 28 | 2.84 | 1620 | Moderate | n/a | n/a | n/a | 54.6 | 1595 |
| 49,153 to 57,344 | 49,153 | 8,191 | 2,901 | 34 | 2.91 | 1620 | Moderate | n/a | n/a | n/a | 55.4 | 1593 |
| 57,345 to 65,536 | 57,345 | 8,191 | 2,873 | 30 | 2.96 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,618 | 42 | 3.23 | 1618 | Moderate | n/a | n/a | n/a | 58.3 | 1599 |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,558 | 44 | 3.31 | 1618 | Moderate | n/a | n/a | n/a | 59.4 | 1603 |
| 81,921 to 90,112 | 81,921 | 8,191 | 2,632 | 49 | 3.22 | 1618 | Moderate | n/a | n/a | n/a | 58.4 | 1593 |
| 90,113 to 98,304 | 90,113 | 8,191 | 2,592 | 32 | 3.24 | 1618 | Moderate | n/a | n/a | n/a | 59.0 | 1595 |
| 98,305 to 106,496 | 98,305 | 8,191 | 2,562 | 57 | 3.31 | 1618 | Moderate | n/a | n/a | n/a | 59.7 | 1591 |
| 106,497 to 114,688 | 106,497 | 8,191 | 2,506 | 49 | 3.37 | 1618 | Moderate | n/a | n/a | n/a | 60.5 | 1592 |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,390 | 61 | 3.56 | 1618 | Moderate | n/a | n/a | n/a | 62.9 | 1597 |
| 122,881 to 131,072 | 122,881 | 8,191 | 2,453 | 65 | 3.48 | 1619 | Moderate | n/a | n/a | n/a | n/a | n/a |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 2,078 | 0 | 2.01 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,167 | 3 | 1.95 | 1619 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 8,320 to 16,384 | 8,320 | 8,064 | 2,356 | 8 | 3.47 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,372 | 13 | 3.53 | 1618 | Moderate | n/a | n/a | n/a | 53.8 | 1599 |
| 24,577 to 32,768 | 24,577 | 8,191 | 2,383 | 15 | 3.49 | 1619 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,348 | 22 | 3.57 | 1618 | Moderate | n/a | n/a | n/a | 55.1 | 1597 |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,308 | 28 | 3.64 | 1617 | Moderate | n/a | n/a | n/a | 56.0 | 1597 |
| 49,153 to 57,344 | 49,153 | 8,191 | 2,221 | 32 | 3.77 | 1617 | Moderate | n/a | n/a | n/a | 55.1 | 1589 |
| 57,345 to 65,536 | 57,345 | 8,191 | 2,194 | 39 | 3.82 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,118 | 27 | 3.96 | 1617 | Moderate | n/a | n/a | n/a | 58.0 | 1601 |
| 73,729 to 81,920 | 73,729 | 8,191 | 2,132 | 39 | 3.94 | 1618 | Moderate | n/a | n/a | n/a | 57.3 | 1590 |
| 81,921 to 90,112 | 81,921 | 8,191 | 2,112 | 46 | 4.00 | 1616 | Moderate | n/a | n/a | n/a | 58.0 | 1589 |
| 90,113 to 98,304 | 90,113 | 8,191 | 2,063 | 45 | 4.07 | 1616 | Moderate | n/a | n/a | n/a | 58.9 | 1592 |
| 98,305 to 106,496 | 98,305 | 8,191 | 2,061 | 57 | 4.09 | 1617 | Moderate | n/a | n/a | n/a | 59.5 | 1588 |
| 106,497 to 114,688 | 106,497 | 8,191 | 2,014 | 61 | 4.19 | 1618 | Moderate | n/a | n/a | n/a | 60.1 | 1587 |
| 114,689 to 122,880 | 114,689 | 8,191 | 2,002 | 66 | 4.22 | 1616 | Moderate | n/a | n/a | n/a | 60.7 | 1587 |
| 122,881 to 131,072 | 122,881 | 8,191 | 1,976 | 67 | 4.28 | 1619 | Moderate | n/a | n/a | n/a | n/a | n/a |

## Decode energy per token

Energy per token (p = prefill, d = decode): CPU mJ, CPU ms and Minstr (millions of instructions) are this process's own work, cut where the prefill ends; the GPU cells are the whole machine's GPU energy over a request that is only a prefill, or a decode repeat (its restore and short prefill included), beside its mean GPU MHz. GPU energy per token rises with the GPU clock and is a record, not a quality or efficiency score.
CPU energy unavailable: the server is another process; --model loads it in this one.

Decode energy per token, code arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 977.3 | 1611 |
| 8k | n/a | n/a | n/a | 841.7 | 1609 |
| 16k | n/a | n/a | n/a | 1025.5 | 1609 |
| 32k | n/a | n/a | n/a | 902.8 | 1606 |
| 64k | n/a | n/a | n/a | 964.6 | 1603 |
| 128k | n/a | n/a | n/a | 1089.8 | 1596 |

Decode energy per token, prose arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 1454.8 | 1610 |
| 8k | n/a | n/a | n/a | 1211.2 | 1605 |
| 16k | n/a | n/a | n/a | 1391.7 | 1606 |
| 32k | n/a | n/a | n/a | 1356.1 | 1603 |
| 64k | n/a | n/a | n/a | 1428.4 | 1603 |
| 128k | n/a | n/a | n/a | 1349.9 | 1587 |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw** | Prefill | 2,643 | 2,771 | 2,983 | 3,044 | 2,873 | 2,453 |
|  | Decode | 129.0 | 167.0 | 124.6 | 152.9 | 143.2 | 120.0 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw (prose)** | Prefill | 2,078 | 2,167 | 2,356 | 2,383 | 2,194 | 1,976 |
|  | Decode | 70.6 | 90.8 | 74.2 | 78.4 | 71.0 | 78.7 |
