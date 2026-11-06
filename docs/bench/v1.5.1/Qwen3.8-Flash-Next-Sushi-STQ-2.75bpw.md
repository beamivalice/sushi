# Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw bench (Sushi 1.5.1)

Commit `399895f6`, binary SHA-256 `4391a90f528f7aaa` (built 2026-11-06 14:56:50). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-v151-qwen-stq275`, 2026-11-06.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1547 MHz, minimum 901 MHz over 439 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 132 s between the arms (die 98 C to 50 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 105.6 | 175.0 | 175.0 | 175.0 | mtp | 1613 | Nominal | 58.7 | 73.7 | 73.0 | 73.0 | mtp | 1618 | Nominal | 0.42 |
| 8k | 158.7 | 163.7 | 169.9 | 163.7 | mtp | 1600 | Nominal | 99.6 | 101.8 | 95.1 | 99.6 | mtp | 1620 | Nominal | 0.61 |
| 16k | 145.5 | 148.2 | 147.4 | 147.4 | mtp | 1571 | Nominal | 95.5 | 89.7 | 94.5 | 94.5 | mtp | 1619 | Nominal | 0.64 |
| 32k | 145.2 | 153.1 | 151.3 | 151.3 | mtp | 1539 | Nominal | 96.2 | 93.0 | 80.2 | 93.0 | mtp | 1607 | Nominal | 0.61 |
| 64k | 156.5 | 164.8 | 149.0 | 156.5 | mtp | 1563 | Nominal | 67.6 | 87.2 | 88.1 | 87.2 | mtp | 1620 | Moderate | 0.56 |
| 128k | 121.5 | 135.0 | 135.4 | 135.0 | mtp | 1568 | Nominal | 75.0 | 73.0 | 72.4 | 73.0 | mtp | 1620 | Moderate | 0.54 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 1,878 | 0 | 2.26 | 1558 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,242 | 16 | 1.94 | 1600 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,282 | 23 | 3.72 | 1568 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,184 | 42 | 3.91 | 1475 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,162 | 56 | 3.95 | 1509 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,162 | 65 | 3.97 | 1481 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,110 | 68 | 4.06 | 1425 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,161 | 84 | 3.99 | 1500 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,078 | 80 | 4.14 | 1492 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,074 | 107 | 4.19 | 1463 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,093 | 117 | 4.17 | 1497 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 2,088 | 121 | 4.19 | 1513 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,032 | 130 | 4.38 | 1499 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,021 | 139 | 4.34 | 1499 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,960 | 147 | 4.49 | 1477 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,929 | 150 | 4.55 | 1475 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,897 | 154 | 4.61 | 1528 | Nominal |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 2,092 | 0 | 2.02 | 1618 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,328 | 16 | 1.85 | 1620 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,326 | 21 | 3.61 | 1619 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,231 | 43 | 3.82 | 1522 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,146 | 57 | 3.96 | 1469 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,222 | 67 | 3.87 | 1551 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,139 | 72 | 4.01 | 1448 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,097 | 73 | 4.08 | 1419 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,163 | 91 | 3.97 | 1571 | Moderate |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,195 | 100 | 3.95 | 1581 | Moderate |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,112 | 112 | 4.10 | 1496 | Moderate |
| 81,889 to 90,112 | 81,889 | 8,223 | 2,130 | 125 | 4.08 | 1524 | Moderate |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,120 | 130 | 4.11 | 1520 | Moderate |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,103 | 132 | 4.16 | 1531 | Moderate |
| 106,465 to 114,688 | 106,465 | 8,223 | 2,088 | 149 | 4.20 | 1508 | Moderate |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,070 | 156 | 4.24 | 1523 | Moderate |
| 122,849 to 131,072 | 122,849 | 8,223 | 2,045 | 170 | 4.30 | 1556 | Moderate |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw** | Prefill | 1,878 | 2,242 | 2,282 | 2,162 | 2,078 | 1,897 |
|  | Decode | 175.0 | 163.7 | 147.4 | 151.3 | 156.5 | 135.0 |
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw (prose)** | Prefill | 2,092 | 2,328 | 2,326 | 2,146 | 2,163 | 2,045 |
|  | Decode | 73.0 | 99.6 | 94.5 | 93.0 | 87.2 | 73.0 |
