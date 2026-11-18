# Qwen3.8-Flash-Next-Sushi-STQ-4bpw bench (Sushi 1.7.0)

Commit `5e25d9e0`, binary SHA-256 `47c1b0c38d313f01` (built 2026-11-18 18:05:06). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261118-181658`, 2026-11-18.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1520 MHz, minimum 338 MHz over 420 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 256 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 184 s between the arms (die 90 C to 53 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 174.4 | 160.9 | 150.7 | 160.9 | mtp | 1620 | Nominal | 65.9 | 83.9 | 73.1 | 73.1 | mtp | 1620 | Nominal | 0.45 |
| 8k | 156.9 | 157.5 | 158.7 | 157.5 | mtp | 1619 | Nominal | 85.6 | 86.1 | 97.1 | 86.1 | mtp | 1620 | Nominal | 0.55 |
| 16k | 133.9 | 128.1 | 129.6 | 129.6 | mtp | 1620 | Nominal | 77.8 | 90.2 | 86.7 | 86.7 | mtp | 1618 | Nominal | 0.67 |
| 32k | 145.7 | 149.2 | 155.5 | 149.2 | mtp | 1616 | Nominal | 103.5 | 100.3 | 99.4 | 100.3 | mtp | 1618 | Nominal | 0.67 |
| 64k | 152.3 | 164.3 | 169.5 | 164.3 | mtp | 1575 | Nominal | 82.0 | 76.6 | 96.7 | 82.0 | mtp | 1618 | Nominal | 0.50 |
| 128k | 120.5 | 123.3 | 126.9 | 123.3 | mtp | 1448 | Heavy | 83.2 | 95.7 | 95.7 | 95.7 | mtp | 1613 | Nominal | 0.78 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 1,663 | 0 | 2.53 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,321 | 22 | 1.88 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,279 | 34 | 3.72 | 1618 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,246 | 70 | 3.85 | 1606 | Nominal | n/a | n/a | n/a | 38.5 | 1580 |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,132 | 79 | 4.03 | 1561 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,104 | 113 | 4.11 | 1505 | Nominal | n/a | n/a | n/a | 33.0 | 1487 |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,070 | 128 | 4.18 | 1489 | Nominal | n/a | n/a | n/a | 32.9 | 1477 |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,040 | 157 | 4.27 | 1498 | Nominal | n/a | n/a | n/a | 32.9 | 1476 |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,019 | 170 | 4.33 | 1489 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,974 | 182 | 4.43 | 1420 | Nominal | n/a | n/a | n/a | 30.1 | 1414 |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,939 | 193 | 4.53 | 1409 | Nominal | n/a | n/a | n/a | 29.7 | 1399 |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,874 | 208 | 4.70 | 1350 | Nominal | n/a | n/a | n/a | 28.6 | 1359 |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,900 | 234 | 4.67 | 1423 | Nominal | n/a | n/a | n/a | 29.7 | 1392 |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,860 | 246 | 4.77 | 1373 | Nominal | n/a | n/a | n/a | 28.8 | 1361 |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,810 | 266 | 4.91 | 1317 | Heavy | n/a | n/a | n/a | 27.6 | 1324 |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,737 | 280 | 5.13 | 1245 | Heavy | n/a | n/a | n/a | 25.8 | 1270 |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,734 | 299 | 5.11 | 1308 | Heavy | n/a | n/a | n/a | n/a | n/a |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 1,756 | 0 | 2.39 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,113 | 17 | 2.03 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,760 | 38 | 4.78 | 1601 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,103 | 88 | 4.09 | 1620 | Nominal | n/a | n/a | n/a | 38.5 | 1578 |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,079 | 117 | 4.14 | 1613 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,073 | 158 | 4.20 | 1604 | Nominal | n/a | n/a | n/a | 38.8 | 1578 |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,018 | 136 | 4.29 | 1551 | Nominal | n/a | n/a | n/a | 36.6 | 1533 |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,961 | 173 | 4.45 | 1492 | Nominal | n/a | n/a | n/a | 33.9 | 1491 |
| 57,313 to 65,536 | 57,313 | 8,223 | 1,932 | 181 | 4.52 | 1487 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,032 | 200 | 4.32 | 1588 | Nominal | n/a | n/a | n/a | 39.0 | 1564 |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,985 | 213 | 4.43 | 1547 | Nominal | n/a | n/a | n/a | 36.5 | 1522 |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,893 | 235 | 4.68 | 1478 | Nominal | n/a | n/a | n/a | 33.6 | 1473 |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,884 | 248 | 4.70 | 1494 | Nominal | n/a | n/a | n/a | 33.2 | 1467 |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,852 | 272 | 4.81 | 1506 | Nominal | n/a | n/a | n/a | 33.9 | 1475 |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,828 | 279 | 4.88 | 1500 | Nominal | n/a | n/a | n/a | 33.8 | 1463 |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,798 | 295 | 4.98 | 1439 | Nominal | n/a | n/a | n/a | 31.7 | 1423 |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,735 | 297 | 5.13 | 1448 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Decode energy per token

Energy per token (p = prefill, d = decode): CPU mJ, CPU ms and Minstr (millions of instructions) are this process's own work, cut where the prefill ends; the GPU cells are the whole machine's GPU energy over a request that is only a prefill, or a decode repeat (its restore and short prefill included), beside its mean GPU MHz. GPU energy per token rises with the GPU clock and is a record, not a quality or efficiency score.
CPU energy unavailable: the server is another process; --model loads it in this one.

Decode energy per token, code arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 456.9 | 1586 |
| 8k | n/a | n/a | n/a | 463.3 | 1591 |
| 16k | n/a | n/a | n/a | 561.3 | 1586 |
| 32k | n/a | n/a | n/a | 491.0 | 1557 |
| 64k | n/a | n/a | n/a | 425.7 | 1514 |
| 128k | n/a | n/a | n/a | 405.9 | 1421 |

Decode energy per token, prose arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 750.6 | 1598 |
| 8k | n/a | n/a | n/a | 697.2 | 1585 |
| 16k | n/a | n/a | n/a | 698.9 | 1563 |
| 32k | n/a | n/a | n/a | 636.0 | 1544 |
| 64k | n/a | n/a | n/a | 810.8 | 1559 |
| 128k | n/a | n/a | n/a | 683.4 | 1497 |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-4bpw** | Prefill | 1,663 | 2,321 | 2,279 | 2,132 | 2,019 | 1,734 |
|  | Decode | 160.9 | 157.5 | 129.6 | 149.2 | 164.3 | 123.3 |
| **Qwen3.8-Flash-Next-Sushi-STQ-4bpw (prose)** | Prefill | 1,756 | 2,113 | 1,760 | 2,079 | 1,932 | 1,735 |
|  | Decode | 73.1 | 86.1 | 86.7 | 100.3 | 82.0 | 95.7 |
