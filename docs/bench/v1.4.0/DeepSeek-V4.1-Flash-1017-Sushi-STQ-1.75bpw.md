# DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw bench (Sushi 1.4.0)

Commit `bf21aa5e`, binary SHA-256 `5027a704b968b7bf` (built 2026-10-27 11:20:39). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-dsv41-engram-b`, 2026-10-27.

Server flags: --ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000 --log-level debug. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1489 MHz, minimum 1202 MHz over 827 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 256 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 170 s between the arms (die 92 C to 50 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 52.0 | 76.6 | 77.8 | 76.6 | dspark | 1620 | Nominal | 46.4 | 49.3 | 49.4 | 49.3 | dspark | 1620 | Nominal | 0.64 |
| 8k | 86.0 | 82.4 | 84.2 | 84.2 | dspark | 1561 | Nominal | 52.1 | 53.7 | 55.8 | 53.7 | dspark | 1612 | Nominal | 0.64 |
| 16k | 72.4 | 76.2 | 76.7 | 76.2 | dspark | 1550 | Nominal | 45.8 | 47.6 | 46.5 | 46.5 | dspark | 1602 | Nominal | 0.61 |
| 32k | 84.3 | 85.3 | 86.3 | 85.3 | dspark | 1557 | Nominal | 46.8 | 48.7 | 48.7 | 48.7 | dspark | 1583 | Nominal | 0.57 |
| 64k | 78.1 | 77.9 | 79.8 | 78.1 | dspark | 1485 | Nominal | 45.1 | 46.2 | 47.0 | 46.2 | dspark | 1550 | Nominal | 0.59 |
| 128k | 61.7 | 64.8 | 63.7 | 63.7 | dspark | 1364 | Heavy | 43.5 | 43.0 | 43.8 | 43.5 | dspark | 1411 | Heavy | 0.68 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 707 | 0 | 5.87 | 1615 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,024 | 0 | 4.10 | 1612 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 998 | 0 | 8.31 | 1532 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 982 | 0 | 8.42 | 1515 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 993 | 0 | 8.36 | 1496 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 964 | 0 | 8.59 | 1506 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 972 | 0 | 8.52 | 1519 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 955 | 0 | 8.67 | 1491 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 950 | 0 | 8.74 | 1485 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 903 | 0 | 9.16 | 1463 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 908 | 0 | 9.11 | 1468 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 885 | 0 | 9.35 | 1419 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 877 | 0 | 9.44 | 1408 | Heavy |
| 98,273 to 106,496 | 98,273 | 8,223 | 864 | 0 | 9.59 | 1408 | Heavy |
| 106,465 to 114,688 | 106,465 | 8,223 | 848 | 0 | 9.75 | 1372 | Heavy |
| 114,657 to 122,880 | 114,657 | 8,223 | 840 | 0 | 9.85 | 1385 | Heavy |
| 122,849 to 131,072 | 122,849 | 8,223 | 829 | 0 | 10.03 | 1364 | Heavy |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 580 | 0 | 7.10 | 1615 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 821 | 0 | 5.08 | 1612 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 913 | 0 | 9.06 | 1602 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 887 | 0 | 9.33 | 1498 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 895 | 0 | 9.24 | 1545 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 864 | 0 | 9.57 | 1497 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 875 | 0 | 9.45 | 1506 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 862 | 0 | 9.59 | 1489 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 852 | 0 | 9.71 | 1474 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 819 | 0 | 10.09 | 1438 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 822 | 0 | 10.06 | 1421 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 806 | 0 | 10.25 | 1416 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 793 | 0 | 10.43 | 1436 | Heavy |
| 98,273 to 106,496 | 98,273 | 8,223 | 773 | 0 | 10.69 | 1424 | Heavy |
| 106,465 to 114,688 | 106,465 | 8,223 | 769 | 0 | 10.76 | 1397 | Heavy |
| 114,657 to 122,880 | 114,657 | 8,223 | 765 | 0 | 10.81 | 1376 | Heavy |
| 122,849 to 131,072 | 122,849 | 8,223 | 744 | 0 | 11.11 | 1355 | Heavy |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw** | Prefill | 707 | 1,024 | 998 | 993 | 950 | 829 |
|  | Decode | 76.6 | 84.2 | 76.2 | 85.3 | 78.1 | 63.7 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw (prose)** | Prefill | 580 | 821 | 913 | 895 | 852 | 744 |
|  | Decode | 49.3 | 53.7 | 46.5 | 48.7 | 46.2 | 43.5 |
