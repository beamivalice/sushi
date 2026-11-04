# DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw bench (Sushi 1.5.0)

Commit `75996eda`, binary SHA-256 `62184462b5e6c176` (built 2026-11-04 06:26:54). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261104-081056`, 2026-11-04.

Server flags: --ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000 --log-level debug. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1484 MHz, minimum 338 MHz over 628 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 256 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 101 s between the arms (die 98 C to 50 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 51.1 | 80.9 | 81.5 | 80.9 | dspark | 1615 | Nominal | 48.2 | 52.8 | 52.9 | 52.8 | dspark | 1620 | Nominal | 0.65 |
| 8k | 87.3 | 97.9 | 97.9 | 97.9 | dspark | 1527 | Nominal | 54.3 | 57.9 | 58.3 | 57.9 | dspark | 1578 | Nominal | 0.59 |
| 16k | 70.4 | 77.2 | 79.2 | 77.2 | dspark | 1514 | Nominal | 49.7 | 52.4 | 52.6 | 52.4 | dspark | 1616 | Nominal | 0.68 |
| 32k | 85.2 | 88.1 | 90.5 | 88.1 | dspark | 1481 | Nominal | 49.9 | 52.4 | 53.0 | 52.4 | dspark | 1574 | Nominal | 0.59 |
| 64k | 82.3 | 86.8 | 86.1 | 86.1 | dspark | 1514 | Nominal | 49.0 | 51.4 | 52.4 | 51.4 | dspark | 1529 | Nominal | 0.60 |
| 128k | 73.2 | 76.1 | 77.0 | 76.1 | dspark | 1506 | Nominal | 47.7 | 50.3 | 50.3 | 50.3 | dspark | 1533 | Nominal | 0.66 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 832 | 0 | 5.00 | 1613 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,420 | 0 | 2.99 | 1534 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,439 | 0 | 5.78 | 1489 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,368 | 0 | 6.06 | 1412 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,387 | 0 | 6.00 | 1481 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,374 | 0 | 6.03 | 1459 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,348 | 0 | 6.15 | 1456 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,365 | 0 | 6.08 | 1467 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 1,358 | 0 | 6.14 | 1482 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,306 | 0 | 6.35 | 1465 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,291 | 0 | 6.42 | 1467 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,265 | 0 | 6.56 | 1465 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,287 | 0 | 6.45 | 1484 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,245 | 0 | 6.66 | 1459 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,251 | 0 | 6.63 | 1505 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,229 | 0 | 6.75 | 1471 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,216 | 0 | 6.83 | 1506 | Nominal |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 884 | 685 | 5.36 | 1608 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,151 | 0 | 3.65 | 1599 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,247 | 0 | 6.65 | 1440 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,247 | 0 | 6.65 | 1383 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,223 | 0 | 6.77 | 1459 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,203 | 0 | 6.89 | 1368 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,217 | 0 | 6.81 | 1330 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,205 | 0 | 6.88 | 1340 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 1,202 | 0 | 6.89 | 1423 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,186 | 0 | 6.98 | 1356 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,161 | 0 | 7.14 | 1370 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,161 | 0 | 7.14 | 1385 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,145 | 0 | 7.24 | 1414 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,128 | 0 | 7.35 | 1385 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,123 | 0 | 7.38 | 1374 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,116 | 0 | 7.43 | 1359 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,101 | 0 | 7.52 | 1404 | Nominal |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw** | Prefill | 832 | 1,420 | 1,439 | 1,387 | 1,358 | 1,216 |
|  | Decode | 80.9 | 97.9 | 77.2 | 88.1 | 86.1 | 76.1 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw (prose)** | Prefill | 884 | 1,151 | 1,247 | 1,223 | 1,202 | 1,101 |
|  | Decode | 52.8 | 57.9 | 52.4 | 52.4 | 51.4 | 50.3 |
