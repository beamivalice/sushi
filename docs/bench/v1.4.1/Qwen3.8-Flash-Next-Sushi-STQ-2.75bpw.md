# Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw bench (Sushi 1.4.1)

Commit `b358cd38`, binary SHA-256 `6944232d09a932d2` (built 2026-10-30 16:00:48). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261030-162032`, 2026-10-30.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1540 MHz, minimum 859 MHz over 461 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 141 s between the arms (die 96 C to 49 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 153.4 | 161.0 | 161.7 | 161.0 | mtp | 1612 | Nominal | 57.8 | 68.6 | 74.3 | 68.6 | mtp | 1617 | Nominal | 0.43 |
| 8k | 147.8 | 150.6 | 149.7 | 149.7 | mtp | 1574 | Nominal | 81.0 | 89.8 | 94.8 | 89.8 | mtp | 1620 | Nominal | 0.60 |
| 16k | 135.9 | 147.4 | 146.2 | 146.2 | mtp | 1593 | Nominal | 85.7 | 84.3 | 81.6 | 84.3 | mtp | 1601 | Nominal | 0.58 |
| 32k | 142.0 | 145.3 | 157.0 | 145.3 | mtp | 1569 | Nominal | 87.0 | 84.0 | 85.0 | 85.0 | mtp | 1570 | Nominal | 0.59 |
| 64k | 152.5 | 161.2 | 161.1 | 161.1 | mtp | 1545 | Nominal | 71.0 | 80.3 | 85.5 | 80.3 | mtp | 1592 | Moderate | 0.50 |
| 128k | 123.3 | 120.6 | 125.4 | 123.3 | mtp | 1540 | Nominal | 54.9 | 55.7 | 60.9 | 55.7 | mtp | 1607 | Heavy | 0.45 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 2,219 | 0 | 1.92 | 1613 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,242 | 28 | 1.96 | 1574 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,137 | 38 | 3.96 | 1551 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,133 | 38 | 4.00 | 1452 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,106 | 42 | 4.03 | 1498 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,173 | 80 | 3.98 | 1502 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,104 | 70 | 4.08 | 1428 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,165 | 104 | 4.01 | 1509 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,090 | 88 | 4.13 | 1506 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,032 | 113 | 4.27 | 1416 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,061 | 131 | 4.24 | 1464 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 2,061 | 142 | 4.25 | 1463 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,026 | 139 | 4.32 | 1447 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,001 | 158 | 4.40 | 1446 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,959 | 144 | 4.48 | 1422 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,940 | 149 | 4.52 | 1412 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,912 | 147 | 4.56 | 1503 | Nominal |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 2,076 | 0 | 2.04 | 1617 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,310 | 17 | 1.89 | 1620 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,227 | 41 | 3.81 | 1589 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,238 | 58 | 3.82 | 1583 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,192 | 41 | 3.87 | 1504 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,226 | 85 | 3.88 | 1553 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,126 | 79 | 4.05 | 1457 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,103 | 73 | 4.09 | 1457 | Moderate |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,156 | 121 | 4.02 | 1559 | Moderate |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,170 | 130 | 4.03 | 1577 | Moderate |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,150 | 136 | 4.06 | 1553 | Moderate |
| 81,889 to 90,112 | 81,889 | 8,223 | 2,129 | 144 | 4.11 | 1546 | Moderate |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,110 | 158 | 4.16 | 1521 | Moderate |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,096 | 155 | 4.19 | 1536 | Moderate |
| 106,465 to 114,688 | 106,465 | 8,223 | 2,091 | 169 | 4.21 | 1523 | Moderate |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,079 | 171 | 4.23 | 1508 | Moderate |
| 122,849 to 131,072 | 122,849 | 8,223 | 2,034 | 191 | 4.35 | 1539 | Heavy |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw** | Prefill | 2,219 | 2,242 | 2,137 | 2,106 | 2,090 | 1,912 |
|  | Decode | 161.0 | 149.7 | 146.2 | 145.3 | 161.1 | 123.3 |
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw (prose)** | Prefill | 2,076 | 2,310 | 2,227 | 2,192 | 2,156 | 2,034 |
|  | Decode | 68.6 | 89.8 | 84.3 | 85.0 | 80.3 | 55.7 |
