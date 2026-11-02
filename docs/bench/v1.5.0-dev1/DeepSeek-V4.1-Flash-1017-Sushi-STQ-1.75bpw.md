# DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw bench (Sushi 1.5.0-dev1)

Commit `f7da31aa`, binary SHA-256 `49fd4ed02f9d4fb8` (built 2026-11-02 08:25:04). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-v150dev1-dsv41`, 2026-11-02.

Server flags: --ssd-budget-gb 108 --log-level debug. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1477 MHz, minimum 338 MHz over 759 busy samples; thermal pressure seen: Nominal.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 256 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 91 s between the arms (die 98 C to 50 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 45.8 | 81.5 | 80.1 | 80.1 | dspark | 1620 | Nominal | 43.7 | 43.2 | 45.6 | 43.7 | dspark | 1620 | Nominal | 0.55 |
| 8k | 93.5 | 104.5 | 103.7 | 103.7 | dspark | 1620 | Nominal | 48.0 | 54.9 | 56.1 | 54.9 | dspark | 1602 | Nominal | 0.53 |
| 16k | 45.7 | 79.9 | 78.8 | 78.8 | dspark | 1538 | Nominal | 37.5 | 45.5 | 41.8 | 41.8 | dspark | 1488 | Nominal | 0.53 |
| 32k | 82.0 | 90.6 | 90.8 | 90.6 | dspark | 1515 | Nominal | 37.2 | 48.4 | 48.5 | 48.4 | dspark | 1620 | Nominal | 0.53 |
| 64k | 83.3 | 84.7 | 85.3 | 84.7 | dspark | 1485 | Nominal | 40.1 | 44.6 | 48.8 | 44.6 | dspark | 1436 | Nominal | 0.53 |
| 128k | 73.0 | 74.3 | 79.2 | 74.3 | dspark | 1555 | Nominal | 48.5 | 49.4 | 52.2 | 49.4 | dspark | 1565 | Nominal | 0.66 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 782 | 0 | 5.32 | 1610 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,506 | 0 | 3.44 | 1618 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,388 | 0 | 5.96 | 1483 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,354 | 0 | 6.12 | 1396 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,270 | 0 | 6.55 | 1495 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,411 | 0 | 5.87 | 1494 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,373 | 0 | 6.04 | 1466 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,358 | 0 | 6.10 | 1441 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 1,275 | 0 | 6.53 | 1510 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,327 | 0 | 6.25 | 1446 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,316 | 0 | 6.30 | 1482 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,304 | 0 | 6.35 | 1477 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,285 | 0 | 6.45 | 1489 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,205 | 0 | 6.87 | 1463 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,236 | 0 | 6.71 | 1478 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,258 | 0 | 6.59 | 1504 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,270 | 0 | 7.42 | 1555 | Nominal |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 756 | 714 | 6.17 | 1608 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 615 | 0 | 6.78 | 1532 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,102 | 0 | 7.52 | 1386 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 978 | 0 | 8.48 | 1305 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,108 | 0 | 8.34 | 1335 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 511 | 0 | 16.16 | 1442 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,012 | 0 | 8.30 | 1401 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,054 | 0 | 7.86 | 1366 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 978 | 0 | 8.47 | 1429 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,111 | 0 | 7.46 | 1359 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,165 | 0 | 7.12 | 1358 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,199 | 0 | 7.64 | 1474 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,151 | 0 | 7.20 | 1374 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,062 | 0 | 7.81 | 1416 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,002 | 0 | 8.27 | 1333 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,039 | 0 | 7.98 | 1379 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,068 | 0 | 7.76 | 1408 | Nominal |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw** | Prefill | 782 | 1,506 | 1,388 | 1,270 | 1,275 | 1,270 |
|  | Decode | 80.1 | 103.7 | 78.8 | 90.6 | 84.7 | 74.3 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw (prose)** | Prefill | 756 | 615 | 1,102 | 1,108 | 978 | 1,068 |
|  | Decode | 43.7 | 54.9 | 41.8 | 48.4 | 44.6 | 49.4 |
