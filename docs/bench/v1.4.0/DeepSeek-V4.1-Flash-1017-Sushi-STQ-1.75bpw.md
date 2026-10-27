# DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw bench (Sushi 1.4.0)

Commit `6927e96c`, binary SHA-256 `282b5969c0fb8d23` (built 2026-10-27 09:54:57). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-dsv41-1.4.0-b108`, 2026-10-27.

Server flags: --ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1532 MHz, minimum 1382 MHz over 811 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 256 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 120 s between the arms (die 94 C to 49 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 72.9 | 75.7 | 77.7 | 75.7 | dspark | 1614 | Nominal | 46.3 | 49.2 | 48.6 | 48.6 | dspark | 1619 | Nominal | 0.64 |
| 8k | 87.9 | 85.3 | 85.2 | 85.3 | dspark | 1568 | Nominal | 51.4 | 56.9 | 55.6 | 55.6 | dspark | 1619 | Nominal | 0.65 |
| 16k | 74.3 | 75.6 | 76.0 | 75.6 | dspark | 1570 | Nominal | 46.7 | 48.0 | 47.2 | 47.2 | dspark | 1614 | Nominal | 0.62 |
| 32k | 84.8 | 86.5 | 86.8 | 86.5 | dspark | 1561 | Moderate | 47.0 | 50.3 | 49.9 | 49.9 | dspark | 1614 | Moderate | 0.58 |
| 64k | 77.9 | 79.7 | 81.3 | 79.7 | dspark | 1524 | Heavy | 45.2 | 46.6 | 44.8 | 45.2 | dspark | 1558 | Heavy | 0.57 |
| 128k | 65.6 | 65.6 | 66.3 | 65.6 | dspark | 1431 | Heavy | 44.8 | 45.0 | 46.1 | 45.0 | dspark | 1531 | Heavy | 0.68 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 663 | 0 | 6.25 | 1614 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,018 | 0 | 4.14 | 1615 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 993 | 0 | 8.34 | 1521 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 983 | 0 | 8.41 | 1526 | Moderate |
| 24,545 to 32,768 | 24,545 | 8,223 | 988 | 0 | 8.39 | 1507 | Moderate |
| 32,737 to 40,960 | 32,737 | 8,223 | 967 | 0 | 8.56 | 1524 | Moderate |
| 40,929 to 49,152 | 40,929 | 8,223 | 963 | 0 | 8.59 | 1511 | Moderate |
| 49,121 to 57,344 | 49,121 | 8,223 | 959 | 0 | 8.63 | 1509 | Moderate |
| 57,313 to 65,536 | 57,313 | 8,223 | 960 | 0 | 8.65 | 1510 | Heavy |
| 65,505 to 73,728 | 65,505 | 8,223 | 914 | 0 | 9.06 | 1488 | Heavy |
| 73,697 to 81,920 | 73,697 | 8,223 | 911 | 0 | 9.09 | 1484 | Heavy |
| 81,889 to 90,112 | 81,889 | 8,223 | 903 | 0 | 9.16 | 1479 | Heavy |
| 90,081 to 98,304 | 90,081 | 8,223 | 901 | 0 | 9.19 | 1501 | Heavy |
| 98,273 to 106,496 | 98,273 | 8,223 | 890 | 0 | 9.30 | 1481 | Heavy |
| 106,465 to 114,688 | 106,465 | 8,223 | 884 | 0 | 9.37 | 1478 | Heavy |
| 114,657 to 122,880 | 114,657 | 8,223 | 861 | 0 | 9.60 | 1466 | Heavy |
| 122,849 to 131,072 | 122,849 | 8,223 | 853 | 0 | 9.73 | 1431 | Heavy |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 596 | 0 | 6.90 | 1615 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 786 | 0 | 5.31 | 1614 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 836 | 0 | 9.88 | 1614 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 823 | 0 | 10.04 | 1548 | Moderate |
| 24,545 to 32,768 | 24,545 | 8,223 | 829 | 0 | 9.97 | 1552 | Moderate |
| 32,737 to 40,960 | 32,737 | 8,223 | 820 | 0 | 10.07 | 1575 | Moderate |
| 40,929 to 49,152 | 40,929 | 8,223 | 822 | 0 | 10.05 | 1543 | Moderate |
| 49,121 to 57,344 | 49,121 | 8,223 | 810 | 0 | 10.20 | 1516 | Heavy |
| 57,313 to 65,536 | 57,313 | 8,223 | 814 | 0 | 10.15 | 1558 | Heavy |
| 65,505 to 73,728 | 65,505 | 8,223 | 785 | 0 | 10.52 | 1509 | Heavy |
| 73,697 to 81,920 | 73,697 | 8,223 | 785 | 0 | 10.53 | 1500 | Heavy |
| 81,889 to 90,112 | 81,889 | 8,223 | 778 | 0 | 10.62 | 1497 | Heavy |
| 90,081 to 98,304 | 90,081 | 8,223 | 761 | 0 | 10.84 | 1474 | Heavy |
| 98,273 to 106,496 | 98,273 | 8,223 | 748 | 0 | 11.05 | 1502 | Heavy |
| 106,465 to 114,688 | 106,465 | 8,223 | 743 | 0 | 11.12 | 1469 | Heavy |
| 114,657 to 122,880 | 114,657 | 8,223 | 738 | 0 | 11.20 | 1419 | Heavy |
| 122,849 to 131,072 | 122,849 | 8,223 | 730 | 0 | 11.33 | 1491 | Heavy |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw** | Prefill | 663 | 1,018 | 993 | 988 | 960 | 853 |
|  | Decode | 75.7 | 85.3 | 75.6 | 86.5 | 79.7 | 65.6 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw (prose)** | Prefill | 596 | 786 | 836 | 829 | 814 | 730 |
|  | Decode | 48.6 | 55.6 | 47.2 | 49.9 | 45.2 | 45.0 |
