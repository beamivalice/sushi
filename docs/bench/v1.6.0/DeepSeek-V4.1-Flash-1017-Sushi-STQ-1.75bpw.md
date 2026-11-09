# DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw bench (Sushi 1.6.0)

Commit `ac3da7aa`, binary SHA-256 `fcdba9054a6ad870` (built 2026-11-09 21:59:29). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261109-231554`, 2026-11-09.

Server flags: --ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000 --log-level debug. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1490 MHz, minimum 338 MHz over 645 busy samples; thermal pressure seen: Nominal.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 256 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 102 s between the arms (die 96 C to 49 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 57.9 | 101.2 | 95.5 | 95.5 | dspark | 1606 | Nominal | 45.9 | 50.3 | 52.1 | 50.3 | dspark | 1620 | Nominal | 0.53 |
| 8k | 94.2 | 100.8 | 103.5 | 100.8 | dspark | 1573 | Nominal | 50.0 | 56.5 | 57.1 | 56.5 | dspark | 1593 | Nominal | 0.56 |
| 16k | 72.8 | 76.5 | 80.9 | 76.5 | dspark | 1512 | Nominal | 45.4 | 47.1 | 46.9 | 46.9 | dspark | 1507 | Nominal | 0.61 |
| 32k | 85.4 | 91.1 | 91.8 | 91.1 | dspark | 1566 | Nominal | 49.1 | 51.3 | 54.3 | 51.3 | dspark | 1538 | Nominal | 0.56 |
| 64k | 80.5 | 87.1 | 84.8 | 84.8 | dspark | 1514 | Nominal | 48.1 | 51.8 | 51.9 | 51.8 | dspark | 1598 | Nominal | 0.61 |
| 128k | 73.9 | 75.1 | 76.9 | 75.1 | dspark | 1528 | Nominal | 49.8 | 52.7 | 52.2 | 52.2 | dspark | 1589 | Nominal | 0.70 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 1,242 | 0 | 3.38 | 1606 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,493 | 0 | 2.83 | 1575 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,497 | 0 | 5.56 | 1512 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,424 | 0 | 5.83 | 1414 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,468 | 0 | 5.68 | 1506 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,434 | 0 | 5.79 | 1439 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,418 | 0 | 5.85 | 1449 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,397 | 0 | 5.94 | 1437 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 1,383 | 0 | 6.02 | 1501 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,339 | 0 | 6.19 | 1449 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,346 | 0 | 6.17 | 1484 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,329 | 0 | 6.25 | 1474 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,313 | 0 | 6.32 | 1462 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,290 | 0 | 6.43 | 1490 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,276 | 0 | 6.50 | 1467 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,252 | 0 | 6.63 | 1478 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,266 | 0 | 6.59 | 1528 | Nominal |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 858 | 774 | 5.59 | 1612 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,093 | 0 | 3.83 | 1605 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,247 | 0 | 6.64 | 1411 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,135 | 0 | 7.30 | 1326 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,191 | 0 | 6.96 | 1443 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,276 | 0 | 6.50 | 1406 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,225 | 0 | 6.76 | 1309 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,259 | 0 | 6.59 | 1348 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 1,264 | 0 | 6.56 | 1460 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,247 | 0 | 6.65 | 1364 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,231 | 0 | 6.74 | 1374 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,217 | 0 | 6.81 | 1337 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,187 | 0 | 6.98 | 1353 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,169 | 0 | 7.09 | 1361 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,189 | 0 | 6.98 | 1374 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,163 | 0 | 7.12 | 1346 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,172 | 0 | 7.08 | 1462 | Nominal |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw** | Prefill | 1,242 | 1,493 | 1,497 | 1,468 | 1,383 | 1,266 |
|  | Decode | 95.5 | 100.8 | 76.5 | 91.1 | 84.8 | 75.1 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw (prose)** | Prefill | 858 | 1,093 | 1,247 | 1,191 | 1,264 | 1,172 |
|  | Decode | 50.3 | 56.5 | 46.9 | 51.3 | 51.8 | 52.2 |
