# Qwen3.8-Flash-Next-Sushi-STQ-4bpw bench (Sushi 1.8.0)

Commit `7c8d11d1`, binary SHA-256 `66e39713d569f967` (built 2026-11-26 19:12:21). Hardware: Apple M5 Ultra 256 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261126-193923`, 2026-11-26.

Server flags: defaults. Private SSD prefix cache, one model per server, no cooldown (a Mac Studio does not throttle).

GPU clock, sampled every 500 ms: median 1617 MHz, minimum 855 MHz over 306 busy samples; thermal pressure seen: Nominal.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 0 s between the arms (die not read).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 284.0 | 270.0 | 265.4 | 270.0 | mtp | 1620 | Nominal | 93.6 | 102.2 | 111.4 | 102.2 | mtp | 1619 | Nominal | 0.38 |
| 8k | 229.0 | 236.0 | 231.1 | 231.1 | mtp | 1619 | Nominal | 137.2 | 138.5 | 140.3 | 138.5 | mtp | 1574 | Nominal | 0.60 |
| 16k | 196.7 | 205.5 | 206.0 | 205.5 | mtp | 1620 | Nominal | 103.3 | 120.2 | 122.1 | 120.2 | mtp | 1600 | Nominal | 0.58 |
| 32k | 240.5 | 251.4 | 251.3 | 251.3 | mtp | 1619 | Nominal | 141.1 | 139.8 | 130.0 | 139.8 | mtp | 1519 | Nominal | 0.56 |
| 64k | 256.9 | 249.1 | 267.2 | 256.9 | mtp | 1619 | Nominal | 122.0 | 130.0 | 120.7 | 122.0 | mtp | 1618 | Nominal | 0.48 |
| 128k | 212.3 | 212.6 | 216.1 | 212.6 | mtp | 1619 | Nominal | 127.7 | 124.8 | 127.6 | 127.6 | mtp | 1619 | Nominal | 0.60 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 3,860 | 0 | 1.14 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 3,702 | 19 | 1.21 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 3,677 | 44 | 2.36 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 3,665 | 50 | 2.39 | 1619 | Nominal | n/a | n/a | n/a | 38.0 | 1590 |
| 24,545 to 32,768 | 24,545 | 8,223 | 3,651 | 77 | 2.42 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 3,593 | 92 | 2.48 | 1618 | Nominal | n/a | n/a | n/a | 38.4 | 1576 |
| 40,929 to 49,152 | 40,929 | 8,223 | 3,572 | 113 | 2.53 | 1617 | Nominal | n/a | n/a | n/a | 38.7 | 1569 |
| 49,121 to 57,344 | 49,121 | 8,223 | 3,527 | 131 | 2.57 | 1619 | Nominal | n/a | n/a | n/a | 38.8 | 1567 |
| 57,313 to 65,536 | 57,313 | 8,223 | 3,443 | 187 | 2.69 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 65,505 to 73,728 | 65,505 | 8,223 | 3,316 | 194 | 2.80 | 1616 | Nominal | n/a | n/a | n/a | 39.4 | 1561 |
| 73,697 to 81,920 | 73,697 | 8,223 | 3,351 | 212 | 2.79 | 1618 | Nominal | n/a | n/a | n/a | 39.6 | 1556 |
| 81,889 to 90,112 | 81,889 | 8,223 | 3,080 | 234 | 3.04 | 1615 | Nominal | n/a | n/a | n/a | 39.9 | 1550 |
| 90,081 to 98,304 | 90,081 | 8,223 | 3,103 | 231 | 3.03 | 1615 | Nominal | n/a | n/a | n/a | 40.2 | 1550 |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,979 | 262 | 3.17 | 1612 | Nominal | n/a | n/a | n/a | 40.2 | 1547 |
| 106,465 to 114,688 | 106,465 | 8,223 | 2,783 | 295 | 3.40 | 1607 | Nominal | n/a | n/a | n/a | 40.4 | 1541 |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,806 | 301 | 3.39 | 1607 | Nominal | n/a | n/a | n/a | 40.7 | 1543 |
| 122,849 to 131,072 | 122,849 | 8,223 | 2,974 | 337 | 3.26 | 1616 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 3,693 | 0 | 1.19 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 3,531 | 33 | 1.28 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 3,562 | 40 | 2.44 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 3,556 | 73 | 2.48 | 1617 | Nominal | n/a | n/a | n/a | 38.2 | 1586 |
| 24,545 to 32,768 | 24,545 | 8,223 | 3,568 | 74 | 2.47 | 1618 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 3,534 | 102 | 2.53 | 1618 | Nominal | n/a | n/a | n/a | 38.4 | 1575 |
| 40,929 to 49,152 | 40,929 | 8,223 | 3,497 | 140 | 2.60 | 1618 | Nominal | n/a | n/a | n/a | 38.7 | 1576 |
| 49,121 to 57,344 | 49,121 | 8,223 | 3,449 | 155 | 2.65 | 1618 | Nominal | n/a | n/a | n/a | 38.9 | 1566 |
| 57,313 to 65,536 | 57,313 | 8,223 | 3,375 | 171 | 2.72 | 1618 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 65,505 to 73,728 | 65,505 | 8,223 | 3,249 | 184 | 2.85 | 1616 | Nominal | n/a | n/a | n/a | 39.2 | 1557 |
| 73,697 to 81,920 | 73,697 | 8,223 | 3,224 | 216 | 2.91 | 1617 | Nominal | n/a | n/a | n/a | 39.4 | 1569 |
| 81,889 to 90,112 | 81,889 | 8,223 | 3,028 | 222 | 3.08 | 1614 | Nominal | n/a | n/a | n/a | 39.5 | 1562 |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,953 | 243 | 3.17 | 1612 | Nominal | n/a | n/a | n/a | 39.8 | 1556 |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,938 | 289 | 3.24 | 1611 | Nominal | n/a | n/a | n/a | 39.9 | 1547 |
| 106,465 to 114,688 | 106,465 | 8,223 | 2,715 | 303 | 3.48 | 1606 | Nominal | n/a | n/a | n/a | 40.2 | 1546 |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,771 | 293 | 3.44 | 1606 | Nominal | n/a | n/a | n/a | 40.3 | 1540 |
| 122,849 to 131,072 | 122,849 | 8,223 | 2,925 | 314 | 3.28 | 1614 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Decode energy per token

Energy per token (p = prefill, d = decode): CPU mJ, CPU ms and Minstr (millions of instructions) are this process's own work, cut where the prefill ends; the GPU cells are the whole machine's GPU energy over a request that is only a prefill, or a decode repeat (its restore and short prefill included), beside its mean GPU MHz. GPU energy per token rises with the GPU clock and is a record, not a quality or efficiency score.
CPU energy unavailable: the server is another process; --model loads it in this one.

Decode energy per token, code arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 385.0 | 1605 |
| 8k | n/a | n/a | n/a | 426.5 | 1601 |
| 16k | n/a | n/a | n/a | 464.4 | 1596 |
| 32k | n/a | n/a | n/a | 409.5 | 1579 |
| 64k | n/a | n/a | n/a | 404.1 | 1554 |
| 128k | n/a | n/a | n/a | 471.2 | 1538 |

Decode energy per token, prose arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 829.0 | 1600 |
| 8k | n/a | n/a | n/a | 635.0 | 1580 |
| 16k | n/a | n/a | n/a | 737.4 | 1569 |
| 32k | n/a | n/a | n/a | 657.7 | 1541 |
| 64k | n/a | n/a | n/a | 669.8 | 1514 |
| 128k | n/a | n/a | n/a | 650.3 | 1466 |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-4bpw** | Prefill | 3,860 | 3,702 | 3,677 | 3,651 | 3,443 | 2,974 |
|  | Decode | 270.0 | 231.1 | 205.5 | 251.3 | 256.9 | 212.6 |
| **Qwen3.8-Flash-Next-Sushi-STQ-4bpw (prose)** | Prefill | 3,693 | 3,531 | 3,562 | 3,568 | 3,375 | 2,925 |
|  | Decode | 102.2 | 138.5 | 120.2 | 139.8 | 122.0 | 127.6 |
