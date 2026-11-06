# DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw bench (Sushi 1.5.1)

Commit `399895f6`, binary SHA-256 `4391a90f528f7aaa` (built 2026-11-06 14:56:50). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-v151-dsv41-stq175`, 2026-11-06.

Server flags: --ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000 --log-level debug. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1443 MHz, minimum 830 MHz over 610 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 256 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 163 s between the arms (die 95 C to 50 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 57.5 | 102.0 | 98.2 | 98.2 | dspark | 1611 | Nominal | 48.1 | 56.4 | 57.1 | 56.4 | dspark | 1620 | Nominal | 0.57 |
| 8k | 93.9 | 102.2 | 103.5 | 102.2 | dspark | 1561 | Nominal | 55.2 | 60.4 | 59.3 | 59.3 | dspark | 1609 | Nominal | 0.58 |
| 16k | 74.5 | 80.0 | 78.7 | 78.7 | dspark | 1506 | Nominal | 50.9 | 54.0 | 52.2 | 52.2 | dspark | 1586 | Nominal | 0.66 |
| 32k | 85.7 | 88.0 | 90.4 | 88.0 | dspark | 1491 | Nominal | 50.0 | 51.7 | 54.2 | 51.7 | dspark | 1536 | Nominal | 0.59 |
| 64k | 80.3 | 84.1 | 83.6 | 83.6 | dspark | 1488 | Nominal | 47.3 | 50.6 | 50.1 | 50.1 | dspark | 1548 | Nominal | 0.60 |
| 128k | 69.8 | 73.0 | 71.8 | 71.8 | dspark | 1391 | Heavy | 47.2 | 48.3 | 48.1 | 48.1 | dspark | 1403 | Heavy | 0.67 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 1,263 | 0 | 3.33 | 1611 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,522 | 0 | 2.77 | 1551 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,538 | 0 | 5.41 | 1475 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,453 | 0 | 5.71 | 1397 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,494 | 0 | 5.58 | 1477 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,451 | 0 | 5.72 | 1429 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,452 | 0 | 5.72 | 1408 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,423 | 0 | 5.84 | 1432 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 1,403 | 0 | 5.93 | 1456 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,324 | 0 | 6.27 | 1405 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,305 | 0 | 6.36 | 1406 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,294 | 0 | 6.42 | 1424 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,275 | 0 | 6.51 | 1407 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,237 | 0 | 6.70 | 1390 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,242 | 0 | 6.68 | 1396 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,211 | 0 | 6.85 | 1407 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,186 | 0 | 7.04 | 1391 | Heavy |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 937 | 747 | 5.16 | 1612 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,201 | 0 | 3.48 | 1588 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,298 | 0 | 6.39 | 1485 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,303 | 0 | 6.36 | 1396 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,272 | 0 | 6.52 | 1424 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,265 | 0 | 6.55 | 1384 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,279 | 0 | 6.49 | 1337 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,253 | 0 | 6.62 | 1296 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 1,255 | 0 | 6.60 | 1445 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,211 | 0 | 6.85 | 1291 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,197 | 0 | 6.92 | 1297 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,187 | 0 | 6.98 | 1273 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,153 | 0 | 7.18 | 1283 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,139 | 0 | 7.27 | 1283 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,122 | 0 | 7.38 | 1264 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,106 | 0 | 7.49 | 1236 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,075 | 0 | 7.70 | 1262 | Heavy |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw** | Prefill | 1,263 | 1,522 | 1,538 | 1,494 | 1,403 | 1,186 |
|  | Decode | 98.2 | 102.2 | 78.7 | 88.0 | 83.6 | 71.8 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw (prose)** | Prefill | 937 | 1,201 | 1,298 | 1,272 | 1,255 | 1,075 |
|  | Decode | 56.4 | 59.3 | 52.2 | 51.7 | 50.1 | 48.1 |
