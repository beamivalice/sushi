# Qwen3.8-Flash-Next-Sushi-4bpw bench (Sushi 1.4.0)

Commit `9d4716e8`, binary SHA-256 `fcb8ee2f7e2e0ede` (built 2026-10-27 00:50:12). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-gate140-Qwen3.8-Flash-Next-Sushi-4bpw`, 2026-10-27.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1577 MHz, minimum 879 MHz over 638 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample. The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on.

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 145.4 | 147.4 | 146.2 | 146.2 | mtp | 1609 | Nominal | 63.4 | 72.7 | 65.6 | 65.6 | mtp | 1619 | Nominal | 0.45 |
| 8k | 155.1 | 149.5 | 150.1 | 150.1 | mtp | 1595 | Nominal | 85.0 | 89.4 | 86.9 | 86.9 | mtp | 1620 | Nominal | 0.58 |
| 16k | 128.9 | 140.5 | 136.2 | 136.2 | mtp | 1562 | Nominal | 70.1 | 79.0 | 80.8 | 79.0 | mtp | 1619 | Nominal | 0.58 |
| 32k | 137.5 | 149.8 | 154.6 | 149.8 | mtp | 1559 | Nominal | 72.9 | 70.7 | 72.0 | 72.0 | mtp | 1590 | Nominal | 0.48 |
| 64k | 146.7 | 153.5 | 155.2 | 153.5 | mtp | 1555 | Nominal | 62.3 | 66.8 | 66.4 | 66.4 | mtp | 1619 | Nominal | 0.43 |
| 128k | 116.5 | 125.8 | 122.0 | 122.0 | mtp | 1560 | Nominal | 69.4 | 74.9 | 74.5 | 74.5 | mtp | 1619 | Heavy | 0.61 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 2,079 | 0 | 2.05 | 1609 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,379 | 14 | 1.84 | 1562 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,326 | 24 | 3.65 | 1542 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,194 | 37 | 3.90 | 1410 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,231 | 54 | 3.83 | 1519 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,241 | 68 | 3.87 | 1440 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,208 | 76 | 3.91 | 1427 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,177 | 78 | 3.97 | 1409 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,147 | 78 | 4.00 | 1504 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,163 | 105 | 4.04 | 1463 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,197 | 111 | 3.97 | 1492 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 2,126 | 113 | 4.11 | 1407 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,180 | 139 | 4.11 | 1477 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,173 | 142 | 4.06 | 1491 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 2,151 | 154 | 4.16 | 1488 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,149 | 162 | 4.13 | 1485 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 2,120 | 178 | 4.19 | 1535 | Nominal |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 2,216 | 0 | 1.91 | 1599 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,301 | 17 | 1.90 | 1620 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,144 | 19 | 3.94 | 1609 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,153 | 31 | 3.95 | 1554 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,115 | 47 | 4.00 | 1568 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,132 | 62 | 4.02 | 1529 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,080 | 66 | 4.11 | 1476 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,098 | 81 | 4.10 | 1531 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,077 | 92 | 4.11 | 1568 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,118 | 93 | 4.10 | 1590 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,102 | 108 | 4.12 | 1571 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 2,081 | 120 | 4.18 | 1562 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,055 | 133 | 4.25 | 1549 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,001 | 137 | 4.36 | 1522 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,990 | 133 | 4.37 | 1487 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,965 | 148 | 4.45 | 1481 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,957 | 155 | 4.47 | 1580 | Nominal |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-4bpw** | Prefill | 2,079 | 2,379 | 2,326 | 2,231 | 2,147 | 2,120 |
|  | Decode | 146.2 | 150.1 | 136.2 | 149.8 | 153.5 | 122.0 |
| **Qwen3.8-Flash-Next-Sushi-4bpw (prose)** | Prefill | 2,216 | 2,301 | 2,144 | 2,115 | 2,077 | 1,957 |
|  | Decode | 65.6 | 86.9 | 79.0 | 72.0 | 66.4 | 74.5 |
