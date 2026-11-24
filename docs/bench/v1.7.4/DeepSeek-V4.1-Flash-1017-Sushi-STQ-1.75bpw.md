# DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw bench (Sushi 1.7.4)

Commit `76e22371` (the binary reports 1.7.4-dev; it was built from `531ef4c5`, three test and doc commits on top that change no `src/` or build file), binary SHA-256 `65488407d8e19395` (built 2026-11-24 12:08:51). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-ds175-main-20261124-108`, 2026-11-24.

Server flags: defaults (the pack's model settings give an SSD budget of 108 GB; context auto). Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1507 MHz, minimum 918 MHz over 333 busy samples; thermal pressure seen: Nominal, Moderate, Heavy (every ladder request read Nominal).

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 256 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 184 s between the arms (die 99 C to 51 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 98.0 | 105.8 | 104.9 | 104.9 | dspark | 1579 | Nominal | 51.4 | 58.1 | 59.0 | 58.1 | dspark | 1619 | Nominal | 0.55 |
| 8k | 99.7 | 107.3 | 107.4 | 107.3 | dspark | 1530 | Nominal | 59.3 | 63.6 | 65.7 | 63.6 | dspark | 1603 | Nominal | 0.59 |
| 16k | 84.0 | 89.0 | 89.8 | 89.0 | dspark | 1506 | Nominal | 51.8 | 53.7 | 56.0 | 53.7 | dspark | 1579 | Nominal | 0.60 |
| 32k | 103.9 | 107.8 | 108.1 | 107.8 | dspark | 1496 | Nominal | 55.6 | 57.6 | 58.6 | 57.6 | dspark | 1541 | Nominal | 0.53 |
| 64k | 87.4 | 90.0 | 90.3 | 90.0 | dspark | 1453 | Nominal | 49.6 | 52.2 | 51.9 | 51.9 | dspark | 1551 | Nominal | 0.58 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 1,517 | 0 | 2.77 | 1612 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,632 | 0 | 2.60 | 1458 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,622 | 0 | 5.13 | 1470 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,624 | 0 | 5.11 | 1404 | Nominal | n/a | n/a | n/a | 41.1 | 1413 |
| 24,577 to 32,768 | 24,577 | 8,191 | 1,706 | 0 | 4.87 | 1446 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,561 | 0 | 5.31 | 1367 | Nominal | n/a | n/a | n/a | 41.1 | 1391 |
| 40,961 to 49,152 | 40,961 | 8,191 | 1,583 | 0 | 5.22 | 1335 | Nominal | n/a | n/a | n/a | 38.6 | 1354 |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,516 | 0 | 5.47 | 1362 | Nominal | n/a | n/a | n/a | 40.9 | 1376 |
| 57,345 to 65,536 | 57,345 | 8,191 | 1,517 | 0 | 5.46 | 1401 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 904 | 24 | 4.60 | 1616 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,315 | 0 | 3.18 | 1569 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,263 | 0 | 6.55 | 1579 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,279 | 0 | 6.46 | 1376 | Nominal | n/a | n/a | n/a | 42.4 | 1432 |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,255 | 0 | 6.59 | 1518 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,252 | 0 | 6.61 | 1369 | Nominal | n/a | n/a | n/a | 42.8 | 1407 |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,299 | 0 | 6.38 | 1484 | Nominal | n/a | n/a | n/a | 48.1 | 1496 |
| 49,153 to 57,344 | 49,153 | 8,191 | 1,275 | 0 | 6.48 | 1466 | Nominal | n/a | n/a | n/a | 46.4 | 1461 |
| 57,345 to 65,536 | 57,345 | 8,191 | 1,251 | 0 | 6.60 | 1429 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Decode energy per token

Energy per token (p = prefill, d = decode): CPU mJ, CPU ms and Minstr (millions of instructions) are this process's own work, cut where the prefill ends; the GPU cells are the whole machine's GPU energy over a request that is only a prefill, or a decode repeat (its restore and short prefill included), beside its mean GPU MHz. GPU energy per token rises with the GPU clock and is a record, not a quality or efficiency score.
CPU energy unavailable: the server is another process; --model loads it in this one.

Decode energy per token, code arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 775.1 | 1551 |
| 8k | n/a | n/a | n/a | 747.5 | 1531 |
| 16k | n/a | n/a | n/a | 845.6 | 1524 |
| 32k | n/a | n/a | n/a | 695.4 | 1497 |
| 64k | n/a | n/a | n/a | 756.1 | 1475 |

Decode energy per token, prose arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 1359.7 | 1619 |
| 8k | n/a | n/a | n/a | 1253.2 | 1598 |
| 16k | n/a | n/a | n/a | 1323.3 | 1572 |
| 32k | n/a | n/a | n/a | 1217.8 | 1559 |
| 64k | n/a | n/a | n/a | 1367.8 | 1568 |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k |
|---|---|---:|---:|---:|---:|---:|
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw** | Prefill | 1,517 | 1,632 | 1,622 | 1,706 | 1,517 |
|  | Decode | 104.9 | 107.3 | 89.0 | 107.8 | 90.0 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw (prose)** | Prefill | 904 | 1,315 | 1,263 | 1,255 | 1,251 |
|  | Decode | 58.1 | 63.6 | 53.7 | 57.6 | 51.9 |
