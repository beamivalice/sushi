# Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw bench (Sushi 1.6.0)

Commit `ac3da7aa`, binary SHA-256 `fcdba9054a6ad870` (built 2026-11-09 21:59:29). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261110-004206`, 2026-11-10.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1596 MHz, minimum 522 MHz over 458 busy samples; thermal pressure seen: Nominal.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 71 s between the arms (die 95 C to 49 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 81.7 | 168.1 | 185.5 | 168.1 | mtp | 1613 | Nominal | 68.8 | 77.1 | 76.9 | 76.9 | mtp | 1620 | Nominal | 0.46 |
| 8k | 158.4 | 164.2 | 164.3 | 164.2 | mtp | 1620 | Nominal | 99.8 | 101.5 | 103.2 | 101.5 | mtp | 1620 | Nominal | 0.62 |
| 16k | 151.2 | 138.0 | 132.5 | 138.0 | mtp | 1611 | Nominal | 91.4 | 94.3 | 91.7 | 91.7 | mtp | 1619 | Nominal | 0.66 |
| 32k | 155.4 | 147.3 | 156.7 | 155.4 | mtp | 1608 | Nominal | 96.5 | 102.4 | 102.0 | 102.0 | mtp | 1619 | Nominal | 0.66 |
| 64k | 167.2 | 166.8 | 166.2 | 166.8 | mtp | 1605 | Nominal | 72.0 | 87.0 | 86.1 | 86.1 | mtp | 1619 | Nominal | 0.52 |
| 128k | 122.4 | 135.8 | 134.1 | 134.1 | mtp | 1617 | Nominal | 69.1 | 57.0 | 70.1 | 69.1 | mtp | 1620 | Nominal | 0.52 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 1,425 | 0 | 3.05 | 1542 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,205 | 49 | 2.11 | 1613 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,114 | 61 | 4.15 | 1565 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,081 | 85 | 4.26 | 1554 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,119 | 104 | 4.21 | 1577 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,064 | 113 | 4.33 | 1544 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,074 | 138 | 4.33 | 1530 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,064 | 158 | 4.37 | 1536 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,049 | 177 | 4.41 | 1546 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,957 | 188 | 4.63 | 1485 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,035 | 209 | 4.49 | 1557 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,991 | 224 | 4.59 | 1513 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,033 | 247 | 4.60 | 1553 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,026 | 273 | 4.57 | 1574 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,983 | 289 | 4.69 | 1576 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,006 | 309 | 4.65 | 1549 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,979 | 334 | 4.72 | 1589 | Nominal |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 1,987 | 0 | 2.23 | 1619 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,190 | 54 | 2.13 | 1620 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,134 | 52 | 4.11 | 1616 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,126 | 61 | 4.14 | 1620 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,162 | 101 | 4.11 | 1620 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,064 | 119 | 4.32 | 1562 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,149 | 136 | 4.17 | 1620 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,147 | 151 | 4.19 | 1619 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,050 | 182 | 4.40 | 1556 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,043 | 200 | 4.46 | 1562 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,094 | 215 | 4.36 | 1615 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 2,082 | 238 | 4.41 | 1605 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,051 | 259 | 4.49 | 1579 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,016 | 273 | 4.58 | 1533 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 2,042 | 300 | 4.55 | 1568 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,036 | 315 | 4.58 | 1574 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 2,022 | 343 | 4.63 | 1612 | Nominal |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw** | Prefill | 1,425 | 2,205 | 2,114 | 2,119 | 2,049 | 1,979 |
|  | Decode | 168.1 | 164.2 | 138.0 | 155.4 | 166.8 | 134.1 |
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw (prose)** | Prefill | 1,987 | 2,190 | 2,134 | 2,162 | 2,050 | 2,022 |
|  | Decode | 76.9 | 101.5 | 91.7 | 102.0 | 86.1 | 69.1 |
