# Qwen3.8-Flash-Next-Sushi-4bpw bench (Sushi 1.4.1)

Commit `b358cd38`, binary SHA-256 `6944232d09a932d2` (built 2026-10-30 16:00:48). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261030-163925`, 2026-10-30.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1536 MHz, minimum 845 MHz over 441 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 141 s between the arms (die 95 C to 50 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 110.5 | 144.8 | 145.5 | 144.8 | mtp | 1613 | Nominal | 80.1 | 77.4 | 76.4 | 77.4 | mtp | 1618 | Nominal | 0.53 |
| 8k | 154.4 | 160.6 | 153.8 | 154.4 | mtp | 1568 | Nominal | 99.5 | 96.5 | 90.4 | 96.5 | mtp | 1619 | Nominal | 0.62 |
| 16k | 128.2 | 139.3 | 132.7 | 132.7 | mtp | 1557 | Nominal | 90.4 | 84.5 | 82.1 | 84.5 | mtp | 1619 | Nominal | 0.64 |
| 32k | 139.1 | 144.9 | 149.8 | 144.9 | mtp | 1522 | Nominal | 93.8 | 95.4 | 96.9 | 95.4 | mtp | 1619 | Nominal | 0.66 |
| 64k | 149.8 | 158.8 | 158.1 | 158.1 | mtp | 1574 | Nominal | 91.9 | 95.4 | 95.8 | 95.4 | mtp | 1610 | Moderate | 0.60 |
| 128k | 116.0 | 120.8 | 119.6 | 119.6 | mtp | 1553 | Nominal | 78.0 | 92.5 | 86.8 | 86.8 | mtp | 1607 | Heavy | 0.73 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 1,288 | 0 | 3.25 | 1601 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,326 | 33 | 1.91 | 1539 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,356 | 45 | 3.62 | 1544 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,218 | 35 | 3.85 | 1402 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,268 | 74 | 3.79 | 1522 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,188 | 62 | 3.92 | 1385 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,203 | 91 | 3.93 | 1411 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,189 | 103 | 3.97 | 1407 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,153 | 85 | 4.01 | 1484 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,142 | 124 | 4.09 | 1433 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,173 | 131 | 4.04 | 1455 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 2,164 | 124 | 4.05 | 1429 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,041 | 124 | 4.36 | 1362 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,095 | 148 | 4.20 | 1410 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 2,060 | 149 | 4.29 | 1443 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,043 | 159 | 4.33 | 1400 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 2,026 | 178 | 4.36 | 1485 | Nominal |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 1,841 | 0 | 2.28 | 1620 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,318 | 29 | 1.88 | 1620 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,269 | 42 | 3.74 | 1620 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,248 | 58 | 3.80 | 1620 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,273 | 49 | 3.73 | 1619 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,248 | 75 | 3.83 | 1613 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,240 | 91 | 3.85 | 1606 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,219 | 87 | 3.88 | 1591 | Moderate |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,128 | 103 | 4.05 | 1522 | Moderate |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,100 | 115 | 4.15 | 1484 | Moderate |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,102 | 128 | 4.15 | 1501 | Moderate |
| 81,889 to 90,112 | 81,889 | 8,223 | 2,098 | 125 | 4.15 | 1479 | Moderate |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,101 | 156 | 4.19 | 1502 | Moderate |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,079 | 153 | 4.24 | 1492 | Moderate |
| 106,465 to 114,688 | 106,465 | 8,223 | 2,088 | 158 | 4.21 | 1513 | Moderate |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,083 | 169 | 4.23 | 1498 | Moderate |
| 122,849 to 131,072 | 122,849 | 8,223 | 2,034 | 174 | 4.33 | 1522 | Heavy |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-4bpw** | Prefill | 1,288 | 2,326 | 2,356 | 2,268 | 2,153 | 2,026 |
|  | Decode | 144.8 | 154.4 | 132.7 | 144.9 | 158.1 | 119.6 |
| **Qwen3.8-Flash-Next-Sushi-4bpw (prose)** | Prefill | 1,841 | 2,318 | 2,269 | 2,273 | 2,128 | 2,034 |
|  | Decode | 77.4 | 96.5 | 84.5 | 95.4 | 95.4 | 86.8 |
