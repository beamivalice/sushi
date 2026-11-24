# DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw bench (Sushi 1.7.4)

Commit `cb048670`, binary SHA-256 `064c86caced50c88` (built 2026-11-24 16:46:32). Hardware: Apple M5 Ultra 256 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261124-165000`, 2026-11-24.

Server flags: defaults. Private SSD prefix cache, one model per server, no cooldown (a Mac Studio does not throttle).

GPU clock, sampled every 500 ms: median 1619 MHz, minimum 742 MHz over 352 busy samples; thermal pressure seen: Nominal, Moderate.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 256 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 0 s between the arms (die not read).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 126.4 | 128.2 | 128.2 | 128.2 | dspark | 1620 | Nominal | 69.9 | 70.1 | 68.7 | 69.9 | dspark | 1620 | Moderate | 0.54 |
| 8k | 165.1 | 165.9 | 165.9 | 165.9 | dspark | 1620 | Nominal | 89.5 | 90.2 | 90.2 | 90.2 | dspark | 1620 | Moderate | 0.54 |
| 16k | 122.1 | 123.8 | 123.8 | 123.8 | dspark | 1620 | Nominal | 73.2 | 73.4 | 73.4 | 73.4 | dspark | 1620 | Moderate | 0.59 |
| 32k | 150.9 | 151.8 | 151.8 | 151.8 | dspark | 1620 | Nominal | 77.5 | 77.8 | 77.8 | 77.8 | dspark | 1620 | Moderate | 0.51 |
| 64k | 141.1 | 142.2 | 142.2 | 142.2 | dspark | 1620 | Moderate | 69.4 | 70.0 | 70.1 | 70.0 | dspark | 1620 | Moderate | 0.49 |
| 128k | 117.7 | 119.1 | 119.1 | 119.1 | dspark | 1620 | Moderate | 77.6 | 79.3 | 78.0 | 78.0 | dspark | 1618 | Moderate | 0.65 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 2,645 | 0 | 1.60 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,760 | 14 | 1.55 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,961 | 20 | 2.85 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,932 | 23 | 2.87 | 1620 | Nominal | n/a | n/a | n/a | 54.3 | 1608 |
| 24,577 to 32,768 | 24,577 | 8,191 | 3,035 | 24 | 2.78 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,850 | 43 | 2.98 | 1620 | Nominal | n/a | n/a | n/a | 56.0 | 1615 |
| 40,961 to 49,152 | 40,961 | 8,191 | 2,929 | 34 | 2.88 | 1620 | Nominal | n/a | n/a | n/a | 55.0 | 1594 |
| 49,153 to 57,344 | 49,153 | 8,191 | 2,887 | 48 | 2.94 | 1620 | Nominal | n/a | n/a | n/a | 56.0 | 1616 |
| 57,345 to 65,536 | 57,345 | 8,191 | 2,854 | 43 | 2.99 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,591 | 49 | 3.28 | 1618 | Moderate | n/a | n/a | n/a | 58.6 | 1600 |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,541 | 52 | 3.34 | 1618 | Moderate | n/a | n/a | n/a | 59.6 | 1598 |
| 81,921 to 90,112 | 81,921 | 8,191 | 2,608 | 54 | 3.25 | 1618 | Moderate | n/a | n/a | n/a | 58.6 | 1592 |
| 90,113 to 98,304 | 90,113 | 8,191 | 2,575 | 59 | 3.30 | 1619 | Moderate | n/a | n/a | n/a | 59.2 | 1592 |
| 98,305 to 106,496 | 98,305 | 8,191 | 2,549 | 57 | 3.33 | 1618 | Moderate | n/a | n/a | n/a | 60.0 | 1591 |
| 106,497 to 114,688 | 106,497 | 8,191 | 2,482 | 65 | 3.43 | 1618 | Moderate | n/a | n/a | n/a | 60.8 | 1593 |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,367 | 68 | 3.60 | 1618 | Moderate | n/a | n/a | n/a | 63.2 | 1598 |
| 122,881 to 131,072 | 122,881 | 8,191 | 2,439 | 70 | 3.51 | 1619 | Moderate | n/a | n/a | n/a | n/a | n/a |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 2,105 | 0 | 1.99 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,170 | 8 | 1.96 | 1619 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 8,320 to 16,384 | 8,320 | 8,064 | 2,403 | 11 | 3.41 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,438 | 15 | 3.43 | 1619 | Moderate | n/a | n/a | n/a | 53.9 | 1593 |
| 24,577 to 32,768 | 24,577 | 8,191 | 2,465 | 22 | 3.39 | 1619 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,439 | 25 | 3.45 | 1619 | Moderate | n/a | n/a | n/a | 55.3 | 1596 |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,415 | 31 | 3.48 | 1619 | Moderate | n/a | n/a | n/a | 56.1 | 1594 |
| 49,153 to 57,344 | 49,153 | 8,191 | 2,396 | 24 | 3.49 | 1618 | Moderate | n/a | n/a | n/a | 55.4 | 1590 |
| 57,345 to 65,536 | 57,345 | 8,191 | 2,315 | 40 | 3.63 | 1619 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,160 | 42 | 3.90 | 1618 | Moderate | n/a | n/a | n/a | 58.2 | 1593 |
| 73,729 to 81,920 | 73,729 | 8,191 | 2,126 | 55 | 3.96 | 1618 | Moderate | n/a | n/a | n/a | 57.6 | 1584 |
| 81,921 to 90,112 | 81,921 | 8,191 | 2,119 | 56 | 3.99 | 1617 | Moderate | n/a | n/a | n/a | 58.2 | 1585 |
| 90,113 to 98,304 | 90,113 | 8,191 | 2,123 | 43 | 3.96 | 1618 | Moderate | n/a | n/a | n/a | 59.0 | 1591 |
| 98,305 to 106,496 | 98,305 | 8,191 | 2,125 | 62 | 3.98 | 1617 | Moderate | n/a | n/a | n/a | 59.8 | 1587 |
| 106,497 to 114,688 | 106,497 | 8,191 | 2,099 | 51 | 4.01 | 1617 | Moderate | n/a | n/a | n/a | 60.4 | 1590 |
| 114,689 to 122,880 | 114,689 | 8,191 | 2,084 | 54 | 4.04 | 1617 | Moderate | n/a | n/a | n/a | 61.1 | 1591 |
| 122,881 to 131,072 | 122,881 | 8,191 | 2,062 | 67 | 4.10 | 1618 | Moderate | n/a | n/a | n/a | n/a | n/a |

## Decode energy per token

Energy per token (p = prefill, d = decode): CPU mJ, CPU ms and Minstr (millions of instructions) are this process's own work, cut where the prefill ends; the GPU cells are the whole machine's GPU energy over a request that is only a prefill, or a decode repeat (its restore and short prefill included), beside its mean GPU MHz. GPU energy per token rises with the GPU clock and is a record, not a quality or efficiency score.
CPU energy unavailable: the server is another process; --model loads it in this one.

Decode energy per token, code arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 981.6 | 1607 |
| 8k | n/a | n/a | n/a | 845.5 | 1609 |
| 16k | n/a | n/a | n/a | 1030.1 | 1608 |
| 32k | n/a | n/a | n/a | 907.5 | 1606 |
| 64k | n/a | n/a | n/a | 969.4 | 1601 |
| 128k | n/a | n/a | n/a | 1095.3 | 1601 |

Decode energy per token, prose arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 1464.6 | 1609 |
| 8k | n/a | n/a | n/a | 1217.5 | 1605 |
| 16k | n/a | n/a | n/a | 1405.1 | 1607 |
| 32k | n/a | n/a | n/a | 1362.8 | 1603 |
| 64k | n/a | n/a | n/a | 1444.8 | 1601 |
| 128k | n/a | n/a | n/a | 1351.3 | 1588 |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw** | Prefill | 2,645 | 2,760 | 2,961 | 3,035 | 2,854 | 2,439 |
|  | Decode | 128.2 | 165.9 | 123.8 | 151.8 | 142.2 | 119.1 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw (prose)** | Prefill | 2,105 | 2,170 | 2,403 | 2,465 | 2,315 | 2,062 |
|  | Decode | 69.9 | 90.2 | 73.4 | 77.8 | 70.0 | 78.0 |
