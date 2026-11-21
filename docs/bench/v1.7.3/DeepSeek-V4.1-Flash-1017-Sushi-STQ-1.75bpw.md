# DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw bench (Sushi 1.7.3)

Commit `fc2fd843`, binary SHA-256 `eb0abb6db57ae23b` (built 2026-11-22 05:07:29). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261122-052517`, 2026-11-22.

Server flags: --ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1506 MHz, minimum 1050 MHz over 346 busy samples; thermal pressure seen: Nominal.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 256 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 71 s between the arms (die 96 C to 49 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 90.3 | 97.9 | 95.3 | 95.3 | dspark | 1612 | Nominal | 51.1 | 58.0 | 58.0 | 58.0 | dspark | 1620 | Nominal | 0.61 |
| 8k | 88.6 | 96.3 | 96.9 | 96.3 | dspark | 1506 | Nominal | 57.2 | 60.1 | 59.6 | 59.6 | dspark | 1559 | Nominal | 0.62 |
| 16k | 79.5 | 83.6 | 85.0 | 83.6 | dspark | 1486 | Nominal | 53.0 | 54.4 | 54.8 | 54.4 | dspark | 1619 | Nominal | 0.65 |
| 32k | 99.0 | 106.2 | 106.2 | 106.2 | dspark | 1548 | Nominal | 57.3 | 60.1 | 60.0 | 60.0 | dspark | 1562 | Nominal | 0.56 |
| 64k | 87.9 | 91.8 | 93.9 | 91.8 | dspark | 1521 | Nominal | 51.4 | 52.5 | 53.3 | 52.5 | dspark | 1541 | Nominal | 0.57 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 1,135 | 0 | 3.68 | 1612 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,591 | 0 | 2.67 | 1506 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,578 | 0 | 5.28 | 1472 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,583 | 0 | 5.25 | 1392 | Nominal | n/a | n/a | n/a | 40.7 | 1404 |
| 24,577 to 32,768 | 24,577 | 8,191 | 1,643 | 0 | 5.06 | 1438 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,555 | 0 | 5.34 | 1424 | Nominal | n/a | n/a | n/a | 43.1 | 1425 |
| 40,961 to 49,152 | 40,961 | 8,191 | 1,599 | 0 | 5.18 | 1379 | Nominal | n/a | n/a | n/a | 41.2 | 1409 |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,534 | 0 | 5.42 | 1429 | Nominal | n/a | n/a | n/a | 44.5 | 1436 |
| 57,345 to 65,536 | 57,345 | 8,191 | 1,550 | 0 | 5.38 | 1455 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 914 | 32 | 4.57 | 1611 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,281 | 0 | 3.28 | 1559 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,234 | 0 | 6.71 | 1530 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,289 | 0 | 6.43 | 1510 | Nominal | n/a | n/a | n/a | 48.8 | 1515 |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,256 | 0 | 6.60 | 1450 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,261 | 0 | 6.57 | 1430 | Nominal | n/a | n/a | n/a | 43.9 | 1434 |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,275 | 0 | 6.50 | 1482 | Nominal | n/a | n/a | n/a | 48.2 | 1484 |
| 49,153 to 57,344 | 49,153 | 8,191 | 1,251 | 0 | 6.60 | 1420 | Nominal | n/a | n/a | n/a | 43.7 | 1429 |
| 57,345 to 65,536 | 57,345 | 8,191 | 1,275 | 0 | 6.48 | 1520 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Decode energy per token

Energy per token (p = prefill, d = decode): CPU mJ, CPU ms and Minstr (millions of instructions) are this process's own work, cut where the prefill ends; the GPU cells are the whole machine's GPU energy over a request that is only a prefill, or a decode repeat (its restore and short prefill included), beside its mean GPU MHz. GPU energy per token rises with the GPU clock and is a record, not a quality or efficiency score.
CPU energy unavailable: the server is another process; --model loads it in this one.

Decode energy per token, code arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 921.9 | 1590 |
| 8k | n/a | n/a | n/a | 831.2 | 1543 |
| 16k | n/a | n/a | n/a | 839.6 | 1498 |
| 32k | n/a | n/a | n/a | 781.1 | 1543 |
| 64k | n/a | n/a | n/a | 851.6 | 1535 |

Decode energy per token, prose arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 1345.3 | 1609 |
| 8k | n/a | n/a | n/a | 1188.4 | 1554 |
| 16k | n/a | n/a | n/a | 1429.8 | 1609 |
| 32k | n/a | n/a | n/a | 1215.0 | 1565 |
| 64k | n/a | n/a | n/a | 1333.3 | 1566 |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k |
|---|---|---:|---:|---:|---:|---:|
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw** | Prefill | 1,135 | 1,591 | 1,578 | 1,643 | 1,550 |
|  | Decode | 95.3 | 96.3 | 83.6 | 106.2 | 91.8 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw (prose)** | Prefill | 914 | 1,281 | 1,234 | 1,256 | 1,275 |
|  | Decode | 58.0 | 59.6 | 54.4 | 60.0 | 52.5 |
