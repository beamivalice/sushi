# Qwen3.8-Flash-Next-Sushi-4bpw bench (Sushi 1.5.0)

Commit `75996eda`, binary SHA-256 `62184462b5e6c176` (built 2026-11-04 06:26:54). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261104-075133`, 2026-11-04.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1515 MHz, minimum 338 MHz over 459 busy samples; thermal pressure seen: Nominal.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 81 s between the arms (die 91 C to 49 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 126.0 | 142.7 | 143.6 | 142.7 | mtp | 1602 | Nominal | 69.4 | 77.1 | 83.5 | 77.1 | mtp | 1619 | Nominal | 0.54 |
| 8k | 152.3 | 150.4 | 154.2 | 152.3 | mtp | 1593 | Nominal | 98.9 | 89.6 | 86.1 | 89.6 | mtp | 1619 | Nominal | 0.59 |
| 16k | 128.4 | 130.1 | 130.2 | 130.1 | mtp | 1544 | Nominal | 75.6 | 86.9 | 87.4 | 86.9 | mtp | 1619 | Nominal | 0.67 |
| 32k | 137.4 | 139.6 | 145.9 | 139.6 | mtp | 1513 | Nominal | 89.3 | 84.1 | 81.4 | 84.1 | mtp | 1610 | Nominal | 0.60 |
| 64k | 147.1 | 152.6 | 148.5 | 148.5 | mtp | 1534 | Nominal | 91.0 | 96.0 | 71.4 | 91.0 | mtp | 1618 | Nominal | 0.61 |
| 128k | 114.2 | 118.0 | 114.4 | 114.4 | mtp | 1492 | Nominal | 65.0 | 86.9 | 89.9 | 86.9 | mtp | 1591 | Nominal | 0.76 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 1,964 | 0 | 2.17 | 1602 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,435 | 31 | 1.82 | 1568 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,334 | 32 | 3.66 | 1534 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,223 | 61 | 3.87 | 1423 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,248 | 54 | 3.80 | 1513 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,093 | 63 | 4.10 | 1312 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,192 | 84 | 3.95 | 1427 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,179 | 93 | 3.98 | 1398 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,143 | 81 | 4.01 | 1501 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,041 | 130 | 4.28 | 1354 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,036 | 124 | 4.29 | 1349 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 2,066 | 107 | 4.22 | 1363 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,130 | 161 | 4.24 | 1431 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,063 | 153 | 4.27 | 1435 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,968 | 169 | 4.47 | 1363 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,891 | 149 | 4.64 | 1288 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,965 | 186 | 4.50 | 1424 | Nominal |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 1,907 | 0 | 2.21 | 1620 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,282 | 12 | 1.89 | 1620 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,272 | 39 | 3.73 | 1620 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,248 | 58 | 3.80 | 1620 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,244 | 71 | 3.82 | 1617 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,051 | 58 | 4.17 | 1483 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,993 | 95 | 4.32 | 1406 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,004 | 101 | 4.30 | 1393 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,043 | 111 | 4.21 | 1468 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,173 | 119 | 4.02 | 1590 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,153 | 125 | 4.06 | 1571 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 2,096 | 141 | 4.17 | 1535 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,008 | 138 | 4.35 | 1499 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,961 | 152 | 4.47 | 1451 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,860 | 134 | 4.67 | 1259 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,938 | 171 | 4.54 | 1401 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,929 | 179 | 4.55 | 1480 | Nominal |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-4bpw** | Prefill | 1,964 | 2,435 | 2,334 | 2,248 | 2,143 | 1,965 |
|  | Decode | 142.7 | 152.3 | 130.1 | 139.6 | 148.5 | 114.4 |
| **Qwen3.8-Flash-Next-Sushi-4bpw (prose)** | Prefill | 1,907 | 2,282 | 2,272 | 2,244 | 2,043 | 1,929 |
|  | Decode | 77.1 | 89.6 | 86.9 | 84.1 | 91.0 | 86.9 |
