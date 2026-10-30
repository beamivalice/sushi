# DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw bench (Sushi 1.4.1)

Commit `b358cd38`, binary SHA-256 `6944232d09a932d2` (built 2026-10-30 16:00:48). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261030-155830`, 2026-10-30.

Server flags: --ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000 --log-level debug. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1486 MHz, minimum 1252 MHz over 650 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 256 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 101 s between the arms (die 95 C to 50 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 53.4 | 80.6 | 79.5 | 79.5 | dspark | 1611 | Nominal | 47.2 | 51.4 | 51.5 | 51.4 | dspark | 1619 | Nominal | 0.65 |
| 8k | 88.8 | 91.0 | 92.7 | 91.0 | dspark | 1570 | Nominal | 53.7 | 58.5 | 56.9 | 56.9 | dspark | 1612 | Nominal | 0.63 |
| 16k | 73.9 | 78.3 | 79.9 | 78.3 | dspark | 1539 | Nominal | 47.5 | 47.8 | 48.9 | 47.8 | dspark | 1567 | Nominal | 0.61 |
| 32k | 88.1 | 90.6 | 90.9 | 90.6 | dspark | 1483 | Nominal | 48.7 | 49.0 | 50.3 | 49.0 | dspark | 1557 | Moderate | 0.54 |
| 64k | 82.7 | 85.8 | 84.6 | 84.6 | dspark | 1537 | Nominal | 46.8 | 49.6 | 50.7 | 49.6 | dspark | 1588 | Moderate | 0.59 |
| 128k | 70.6 | 72.0 | 72.7 | 72.0 | dspark | 1487 | Nominal | 47.0 | 46.6 | 46.5 | 46.6 | dspark | 1484 | Heavy | 0.65 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 807 | 0 | 5.14 | 1611 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,390 | 0 | 3.04 | 1540 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,428 | 0 | 5.81 | 1479 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,373 | 0 | 6.03 | 1429 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,408 | 0 | 5.90 | 1483 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,364 | 0 | 6.07 | 1425 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,366 | 0 | 6.06 | 1440 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,381 | 0 | 6.00 | 1494 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 1,352 | 0 | 6.17 | 1468 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,299 | 0 | 6.39 | 1460 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,301 | 0 | 6.38 | 1487 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,256 | 0 | 6.60 | 1493 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,296 | 0 | 6.41 | 1490 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,266 | 0 | 6.55 | 1468 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,234 | 0 | 6.72 | 1467 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,235 | 0 | 6.72 | 1475 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,217 | 0 | 6.83 | 1478 | Nominal |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 703 | 0 | 5.86 | 1609 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,079 | 0 | 3.89 | 1612 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,199 | 0 | 6.90 | 1450 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,184 | 0 | 6.99 | 1389 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,199 | 0 | 6.91 | 1464 | Moderate |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,168 | 0 | 7.09 | 1418 | Moderate |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,152 | 0 | 7.19 | 1423 | Moderate |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,158 | 0 | 7.15 | 1356 | Moderate |
| 57,313 to 65,536 | 57,313 | 8,223 | 1,193 | 0 | 6.95 | 1486 | Moderate |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,139 | 0 | 7.28 | 1377 | Heavy |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,138 | 0 | 7.29 | 1375 | Heavy |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,140 | 0 | 7.28 | 1364 | Heavy |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,104 | 2 | 7.51 | 1364 | Heavy |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,081 | 0 | 7.66 | 1357 | Heavy |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,091 | 0 | 7.60 | 1368 | Heavy |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,088 | 0 | 7.62 | 1355 | Heavy |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,061 | 0 | 7.81 | 1406 | Heavy |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw** | Prefill | 807 | 1,390 | 1,428 | 1,408 | 1,352 | 1,217 |
|  | Decode | 79.5 | 91.0 | 78.3 | 90.6 | 84.6 | 72.0 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw (prose)** | Prefill | 703 | 1,079 | 1,199 | 1,199 | 1,193 | 1,061 |
|  | Decode | 51.4 | 56.9 | 47.8 | 49.0 | 49.6 | 46.6 |
