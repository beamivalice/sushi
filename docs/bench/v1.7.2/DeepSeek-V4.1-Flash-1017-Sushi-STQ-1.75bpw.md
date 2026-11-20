# DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw bench (Sushi 1.7.2)

Commit `a0b5022b`, binary SHA-256 `d00671283f9bdf82` (built 2026-11-20 23:09:11). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261121-062710`, 2026-11-21.

Server flags: --ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1536 MHz, minimum 1019 MHz over 342 busy samples; thermal pressure seen: Nominal.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 256 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 71 s between the arms (die 96 C to 49 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 92.6 | 95.1 | 95.2 | 95.1 | dspark | 1581 | Nominal | 51.0 | 58.7 | 58.2 | 58.2 | dspark | 1619 | Nominal | 0.61 |
| 8k | 87.8 | 93.1 | 96.2 | 93.1 | dspark | 1527 | Nominal | 54.5 | 58.7 | 58.7 | 58.7 | dspark | 1548 | Nominal | 0.63 |
| 16k | 79.5 | 84.9 | 84.6 | 84.6 | dspark | 1540 | Nominal | 51.2 | 54.6 | 54.8 | 54.6 | dspark | 1620 | Nominal | 0.64 |
| 32k | 98.1 | 101.5 | 104.1 | 101.5 | dspark | 1504 | Nominal | 56.2 | 61.0 | 60.0 | 60.0 | dspark | 1582 | Nominal | 0.59 |
| 64k | 85.8 | 91.2 | 93.3 | 91.2 | dspark | 1560 | Nominal | 50.5 | 51.2 | 53.1 | 51.2 | dspark | 1586 | Nominal | 0.56 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 965 | 0 | 4.31 | 1615 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,635 | 0 | 2.60 | 1527 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,635 | 0 | 5.09 | 1491 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,618 | 0 | 5.13 | 1386 | Nominal | n/a | n/a | n/a | 40.1 | 1408 |
| 24,577 to 32,768 | 24,577 | 8,191 | 1,705 | 0 | 4.88 | 1489 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,582 | 0 | 5.25 | 1399 | Nominal | n/a | n/a | n/a | 42.1 | 1421 |
| 40,961 to 49,152 | 40,961 | 8,191 | 1,647 | 0 | 5.03 | 1446 | Nominal | n/a | n/a | n/a | 42.2 | 1434 |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,509 | 0 | 5.49 | 1376 | Nominal | n/a | n/a | n/a | 41.8 | 1403 |
| 57,345 to 65,536 | 57,345 | 8,191 | 1,586 | 0 | 5.26 | 1466 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 812 | 31 | 5.12 | 1615 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,321 | 0 | 3.17 | 1543 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,246 | 0 | 6.65 | 1543 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,339 | 0 | 6.19 | 1499 | Nominal | n/a | n/a | n/a | 49.8 | 1539 |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,303 | 0 | 6.36 | 1528 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,215 | 0 | 6.81 | 1400 | Nominal | n/a | n/a | n/a | 43.1 | 1420 |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,298 | 0 | 6.38 | 1436 | Nominal | n/a | n/a | n/a | 44.8 | 1451 |
| 49,153 to 57,344 | 49,153 | 8,191 | 1,304 | 0 | 6.34 | 1474 | Nominal | n/a | n/a | n/a | 45.9 | 1467 |
| 57,345 to 65,536 | 57,345 | 8,191 | 1,288 | 0 | 6.41 | 1514 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Decode energy per token

Energy per token (p = prefill, d = decode): CPU mJ, CPU ms and Minstr (millions of instructions) are this process's own work, cut where the prefill ends; the GPU cells are the whole machine's GPU energy over a request that is only a prefill, or a decode repeat (its restore and short prefill included), beside its mean GPU MHz. GPU energy per token rises with the GPU clock and is a record, not a quality or efficiency score.
CPU energy unavailable: the server is another process; --model loads it in this one.

Decode energy per token, code arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 899.2 | 1587 |
| 8k | n/a | n/a | n/a | 810.7 | 1539 |
| 16k | n/a | n/a | n/a | 929.9 | 1559 |
| 32k | n/a | n/a | n/a | 752.0 | 1533 |
| 64k | n/a | n/a | n/a | 890.6 | 1562 |

Decode energy per token, prose arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 1339.5 | 1613 |
| 8k | n/a | n/a | n/a | 1216.5 | 1564 |
| 16k | n/a | n/a | n/a | 1430.1 | 1614 |
| 32k | n/a | n/a | n/a | 1282.5 | 1593 |
| 64k | n/a | n/a | n/a | 1390.0 | 1593 |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k |
|---|---|---:|---:|---:|---:|---:|
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw** | Prefill | 965 | 1,635 | 1,635 | 1,705 | 1,586 |
|  | Decode | 95.1 | 93.1 | 84.6 | 101.5 | 91.2 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw (prose)** | Prefill | 812 | 1,321 | 1,246 | 1,303 | 1,288 |
|  | Decode | 58.2 | 58.7 | 54.6 | 60.0 | 51.2 |
