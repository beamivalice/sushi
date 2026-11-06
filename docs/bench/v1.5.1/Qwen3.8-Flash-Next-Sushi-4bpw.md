# Qwen3.8-Flash-Next-Sushi-4bpw bench (Sushi 1.5.1)

Commit `399895f6`, binary SHA-256 `4391a90f528f7aaa` (built 2026-11-06 14:56:50). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-v151-qwen-4bpw`, 2026-11-06.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1530 MHz, minimum 867 MHz over 432 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 143 s between the arms (die 96 C to 50 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 124.1 | 145.0 | 145.4 | 145.0 | mtp | 1615 | Nominal | 77.1 | 79.9 | 85.0 | 79.9 | mtp | 1620 | Nominal | 0.55 |
| 8k | 152.1 | 155.4 | 148.6 | 152.1 | mtp | 1570 | Nominal | 99.6 | 86.7 | 90.8 | 90.8 | mtp | 1620 | Nominal | 0.60 |
| 16k | 127.4 | 136.1 | 132.0 | 132.0 | mtp | 1549 | Nominal | 77.9 | 87.1 | 87.5 | 87.1 | mtp | 1617 | Nominal | 0.66 |
| 32k | 136.6 | 147.5 | 152.8 | 147.5 | mtp | 1529 | Nominal | 94.2 | 99.7 | 97.5 | 97.5 | mtp | 1618 | Nominal | 0.66 |
| 64k | 144.2 | 152.3 | 150.5 | 150.5 | mtp | 1548 | Nominal | 92.9 | 97.7 | 74.2 | 92.9 | mtp | 1444 | Moderate | 0.62 |
| 128k | 117.3 | 126.0 | 123.2 | 123.2 | mtp | 1508 | Nominal | 81.6 | 91.0 | 86.7 | 86.7 | mtp | 1619 | Heavy | 0.70 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 2,454 | 0 | 1.75 | 1604 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,370 | 20 | 1.84 | 1570 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,304 | 29 | 3.69 | 1549 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,266 | 46 | 3.79 | 1440 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,241 | 54 | 3.81 | 1514 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,245 | 59 | 3.82 | 1427 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,208 | 75 | 3.90 | 1400 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,171 | 85 | 3.98 | 1390 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,149 | 90 | 4.01 | 1408 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,111 | 113 | 4.14 | 1386 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,156 | 118 | 4.06 | 1460 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 2,129 | 121 | 4.11 | 1471 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,085 | 139 | 4.29 | 1473 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,053 | 146 | 4.28 | 1460 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 2,023 | 143 | 4.35 | 1429 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,975 | 168 | 4.48 | 1447 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,947 | 166 | 4.51 | 1508 | Nominal |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 1,993 | 0 | 2.12 | 1620 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,320 | 19 | 1.87 | 1620 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,251 | 28 | 3.75 | 1616 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,234 | 42 | 3.81 | 1617 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,189 | 52 | 3.88 | 1557 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,235 | 66 | 3.84 | 1619 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,188 | 70 | 3.92 | 1542 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,114 | 82 | 4.06 | 1444 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,075 | 85 | 4.12 | 1443 | Moderate |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,083 | 107 | 4.17 | 1484 | Moderate |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,110 | 120 | 4.12 | 1484 | Moderate |
| 81,889 to 90,112 | 81,889 | 8,223 | 2,121 | 133 | 4.12 | 1522 | Moderate |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,115 | 136 | 4.14 | 1534 | Moderate |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,093 | 142 | 4.19 | 1510 | Moderate |
| 106,465 to 114,688 | 106,465 | 8,223 | 2,078 | 149 | 4.23 | 1531 | Moderate |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,060 | 157 | 4.27 | 1497 | Moderate |
| 122,849 to 131,072 | 122,849 | 8,223 | 2,036 | 172 | 4.32 | 1527 | Heavy |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-4bpw** | Prefill | 2,454 | 2,370 | 2,304 | 2,241 | 2,149 | 1,947 |
|  | Decode | 145.0 | 152.1 | 132.0 | 147.5 | 150.5 | 123.2 |
| **Qwen3.8-Flash-Next-Sushi-4bpw (prose)** | Prefill | 1,993 | 2,320 | 2,251 | 2,189 | 2,075 | 2,036 |
|  | Decode | 79.9 | 90.8 | 87.1 | 97.5 | 92.9 | 86.7 |
