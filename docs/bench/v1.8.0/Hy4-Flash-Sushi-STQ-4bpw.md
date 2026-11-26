# Hy4-Flash-Sushi-STQ-4bpw bench (Sushi 1.8.0)

Commit `7c8d11d1`, binary SHA-256 `66e39713d569f967` (built 2026-11-26 19:12:21). Hardware: Apple M5 Ultra 256 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261126-194652`, 2026-11-26.

Server flags: defaults. Private SSD prefix cache, one model per server, no cooldown (a Mac Studio does not throttle).

GPU clock, sampled every 500 ms: median 1620 MHz, minimum 1280 MHz over 455 busy samples; thermal pressure seen: Moderate.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 256 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 0 s between the arms (die not read).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 140.6 | 143.2 | 143.3 | 143.2 | dfly2 | 1620 | Moderate | 69.5 | 69.6 | 69.7 | 69.6 | dfly2 | 1620 | Moderate | 0.49 |
| 8k | 113.8 | 115.0 | 115.0 | 115.0 | dfly2 | 1620 | Moderate | 89.6 | 88.9 | 90.4 | 89.6 | dfly2 | 1620 | Moderate | 0.78 |
| 16k | 114.8 | 116.3 | 116.3 | 116.3 | dfly2 | 1620 | Moderate | 67.7 | 67.7 | 67.7 | 67.7 | dfly2 | 1620 | Moderate | 0.58 |
| 32k | 101.0 | 102.3 | 102.3 | 102.3 | dfly2 | 1620 | Moderate | 70.2 | 70.9 | 70.9 | 70.9 | dfly2 | 1620 | Moderate | 0.69 |
| 64k | 71.1 | 74.5 | 74.5 | 74.5 | dfly2 | 1620 | Moderate | 63.8 | 64.7 | 65.1 | 64.7 | dfly2 | 1620 | Moderate | 0.87 |
| 128k | 94.9 | 96.3 | 96.2 | 96.2 | dfly2 | 1620 | Moderate | 61.3 | 62.1 | 61.6 | 61.6 | dfly2 | 1620 | Moderate | 0.64 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 1,929 | 17 | 2.19 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,939 | 69 | 2.25 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,904 | 89 | 4.47 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,868 | 104 | 4.55 | 1620 | Moderate | n/a | n/a | n/a | 79.2 | 1616 |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,837 | 102 | 4.64 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,806 | 139 | 4.73 | 1620 | Moderate | n/a | n/a | n/a | 82.0 | 1615 |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,774 | 156 | 4.85 | 1620 | Moderate | n/a | n/a | n/a | 83.6 | 1616 |
| 49,153 to 57,344 | 49,153 | 8,191 | 1,773 | 172 | 4.83 | 1620 | Moderate | n/a | n/a | n/a | 84.1 | 1615 |
| 57,313 to 65,536 | 57,313 | 8,223 | 1,713 | 180 | 5.05 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,688 | 190 | 5.10 | 1620 | Moderate | n/a | n/a | n/a | 88.0 | 1617 |
| 73,729 to 81,920 | 73,729 | 8,191 | 1,689 | 199 | 5.11 | 1620 | Moderate | n/a | n/a | n/a | 88.4 | 1616 |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,632 | 224 | 5.31 | 1620 | Moderate | n/a | n/a | n/a | 90.9 | 1617 |
| 90,113 to 98,304 | 90,113 | 8,191 | 1,634 | 237 | 5.31 | 1620 | Moderate | n/a | n/a | n/a | 91.2 | 1616 |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,584 | 256 | 5.49 | 1620 | Moderate | n/a | n/a | n/a | 93.6 | 1618 |
| 106,497 to 114,688 | 106,497 | 8,191 | 1,578 | 263 | 5.52 | 1620 | Moderate | n/a | n/a | n/a | 94.3 | 1615 |
| 114,689 to 122,880 | 114,689 | 8,191 | 1,554 | 261 | 5.57 | 1620 | Moderate | n/a | n/a | n/a | 95.7 | 1615 |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,510 | 295 | 5.81 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 1,678 | 17 | 2.49 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,708 | 82 | 2.54 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,806 | 65 | 4.66 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,793 | 103 | 4.73 | 1620 | Moderate | n/a | n/a | n/a | 79.3 | 1613 |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,765 | 127 | 4.83 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,739 | 139 | 4.91 | 1620 | Moderate | n/a | n/a | n/a | 82.2 | 1613 |
| 40,961 to 49,152 | 40,961 | 8,191 | 1,750 | 158 | 4.89 | 1620 | Moderate | n/a | n/a | n/a | 83.0 | 1615 |
| 49,153 to 57,344 | 49,153 | 8,191 | 1,721 | 168 | 4.97 | 1620 | Moderate | n/a | n/a | n/a | 84.5 | 1615 |
| 57,345 to 65,536 | 57,345 | 8,191 | 1,693 | 176 | 5.06 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,647 | 191 | 5.23 | 1620 | Moderate | n/a | n/a | n/a | 88.0 | 1615 |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,616 | 203 | 5.35 | 1620 | Moderate | n/a | n/a | n/a | 89.5 | 1613 |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,598 | 226 | 5.42 | 1620 | Moderate | n/a | n/a | n/a | 91.0 | 1615 |
| 90,113 to 98,304 | 90,113 | 8,191 | 1,596 | 228 | 5.42 | 1620 | Moderate | n/a | n/a | n/a | 91.8 | 1615 |
| 98,305 to 106,496 | 98,305 | 8,191 | 1,572 | 248 | 5.51 | 1620 | Moderate | n/a | n/a | n/a | 93.2 | 1615 |
| 106,497 to 114,688 | 106,497 | 8,191 | 1,548 | 253 | 5.60 | 1620 | Moderate | n/a | n/a | n/a | 94.7 | 1615 |
| 114,689 to 122,880 | 114,689 | 8,191 | 1,530 | 267 | 5.66 | 1620 | Moderate | n/a | n/a | n/a | 96.1 | 1615 |
| 122,881 to 131,072 | 122,881 | 8,191 | 1,508 | 301 | 5.79 | 1620 | Moderate | n/a | n/a | n/a | n/a | n/a |

## Decode energy per token

Energy per token (p = prefill, d = decode): CPU mJ, CPU ms and Minstr (millions of instructions) are this process's own work, cut where the prefill ends; the GPU cells are the whole machine's GPU energy over a request that is only a prefill, or a decode repeat (its restore and short prefill included), beside its mean GPU MHz. GPU energy per token rises with the GPU clock and is a record, not a quality or efficiency score.
CPU energy unavailable: the server is another process; --model loads it in this one.

Decode energy per token, code arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 722.6 | 1614 |
| 8k | n/a | n/a | n/a | 873.4 | 1615 |
| 16k | n/a | n/a | n/a | 871.4 | 1615 |
| 32k | n/a | n/a | n/a | 965.8 | 1615 |
| 64k | n/a | n/a | n/a | 1192.7 | 1617 |
| 128k | n/a | n/a | n/a | 1046.1 | 1616 |

Decode energy per token, prose arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 1298.2 | 1614 |
| 8k | n/a | n/a | n/a | 1078.8 | 1612 |
| 16k | n/a | n/a | n/a | 1281.5 | 1614 |
| 32k | n/a | n/a | n/a | 1282.8 | 1614 |
| 64k | n/a | n/a | n/a | 1325.0 | 1614 |
| 128k | n/a | n/a | n/a | 1390.6 | 1615 |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Hy4-Flash-Sushi-STQ-4bpw** | Prefill | 1,929 | 1,939 | 1,904 | 1,837 | 1,713 | 1,510 |
|  | Decode | 143.2 | 115.0 | 116.3 | 102.3 | 74.5 | 96.2 |
| **Hy4-Flash-Sushi-STQ-4bpw (prose)** | Prefill | 1,678 | 1,708 | 1,806 | 1,765 | 1,693 | 1,508 |
|  | Decode | 69.6 | 89.6 | 67.7 | 70.9 | 64.7 | 61.6 |
