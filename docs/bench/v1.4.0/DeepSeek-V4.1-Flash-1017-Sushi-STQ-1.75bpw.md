# DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw bench (Sushi 1.4.0)

Commit `a2337236`, binary SHA-256 `f3fc6fbb1812621e` (built 2026-10-27 00:05:00). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-dsv41-1.4.0-r3-114`, 2026-10-27.

Server flags: --ssd-budget-gb 114 --ctx-size 1000000 --max-tokens 128000. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1601 MHz, minimum 1270 MHz over 1093 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample. The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on.

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 70.2 | 84.6 | 83.3 | 83.3 | dspark | 1619 | Nominal | 47.3 | 48.3 | 48.8 | 48.3 | dspark | 1620 | Nominal | 0.58 |
| 8k | 67.8 | 80.4 | 82.4 | 80.4 | dspark | 1588 | Nominal | 47.4 | 50.2 | 50.4 | 50.2 | dspark | 1587 | Nominal | 0.62 |
| 16k | 69.8 | 79.0 | 75.8 | 75.8 | dspark | 1573 | Nominal | 57.6 | 48.0 | 59.3 | 57.6 | dspark | 1585 | Heavy | 0.76 |
| 32k | 68.7 | 79.2 | 79.2 | 79.2 | dspark | 1573 | Nominal | 45.6 | 53.1 | 46.4 | 46.4 | dspark | 1566 | Heavy | 0.59 |
| 64k | 68.8 | 80.4 | 79.0 | 79.0 | dspark | 1606 | Nominal | 44.0 | 76.3 | 78.2 | 76.3 | dspark | 1473 | Heavy | 0.97 |
| 128k | 63.0 | 72.9 | 72.7 | 72.7 | dspark | 1543 | Nominal | 45.4 | 51.3 | 46.0 | 46.0 | dspark | 1560 | Heavy | 0.63 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 641 | 0 | 6.46 | 1619 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 681 | 0 | 6.12 | 1614 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 804 | 0 | 10.28 | 1618 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 801 | 0 | 11.10 | 1578 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 781 | 0 | 10.58 | 1610 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 769 | 0 | 11.53 | 1616 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 819 | 0 | 10.89 | 1614 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 780 | 0 | 11.39 | 1617 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 793 | 0 | 10.43 | 1614 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 764 | 0 | 11.61 | 1611 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 754 | 0 | 11.74 | 1614 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 730 | 0 | 12.10 | 1615 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 740 | 0 | 11.96 | 1617 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 729 | 0 | 12.13 | 1618 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 740 | 0 | 11.96 | 1617 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 730 | 0 | 12.11 | 1617 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 659 | 0 | 12.57 | 1618 | Nominal |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 636 | 0 | 6.49 | 1619 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 544 | 0 | 7.63 | 1618 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 712 | 0 | 11.61 | 1612 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 703 | 0 | 12.56 | 1597 | Heavy |
| 24,545 to 32,768 | 24,545 | 8,223 | 699 | 0 | 11.82 | 1620 | Heavy |
| 32,737 to 40,960 | 32,737 | 8,223 | 697 | 0 | 12.66 | 1540 | Heavy |
| 40,929 to 49,152 | 40,929 | 8,223 | 702 | 0 | 12.57 | 1609 | Heavy |
| 49,121 to 57,344 | 49,121 | 8,223 | 729 | 0 | 12.14 | 1614 | Heavy |
| 57,313 to 65,536 | 57,313 | 8,223 | 617 | 0 | 13.38 | 1591 | Heavy |
| 65,505 to 73,728 | 65,505 | 8,223 | 689 | 0 | 12.80 | 1548 | Heavy |
| 73,697 to 81,920 | 73,697 | 8,223 | 684 | 0 | 12.87 | 1615 | Heavy |
| 81,889 to 90,112 | 81,889 | 8,223 | 664 | 0 | 13.23 | 1517 | Heavy |
| 90,081 to 98,304 | 90,081 | 8,223 | 702 | 0 | 12.57 | 1615 | Heavy |
| 98,273 to 106,496 | 98,273 | 8,223 | 656 | 0 | 13.39 | 1615 | Heavy |
| 106,465 to 114,688 | 106,465 | 8,223 | 648 | 0 | 13.58 | 1530 | Heavy |
| 114,657 to 122,880 | 114,657 | 8,223 | 695 | 0 | 12.70 | 1615 | Heavy |
| 122,849 to 131,072 | 122,849 | 8,223 | 665 | 0 | 12.41 | 1617 | Heavy |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw** | Prefill | 641 | 681 | 804 | 781 | 793 | 659 |
|  | Decode | 83.3 | 80.4 | 75.8 | 79.2 | 79.0 | 72.7 |
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw (prose)** | Prefill | 636 | 544 | 712 | 699 | 617 | 665 |
|  | Decode | 48.3 | 50.2 | 57.6 | 46.4 | 76.3 | 46.0 |
