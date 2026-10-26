# Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw bench (Sushi 1.4.0)

Commit `92237c07`, binary SHA-256 `fcb8ee2f7e2e0ede` (built 2026-10-27 00:50:12). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-gate140-Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw`, 2026-10-27.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1587 MHz, minimum 888 MHz over 608 busy samples; thermal pressure seen: Nominal.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample. The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on.

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 125.8 | 170.8 | 168.6 | 168.6 | mtp | 1612 | Nominal | 57.9 | 62.8 | 63.6 | 62.8 | mtp | 1586 | Nominal | 0.37 |
| 8k | 151.1 | 150.2 | 164.2 | 151.1 | mtp | 1611 | Nominal | 92.8 | 98.1 | 96.9 | 96.9 | mtp | 1618 | Nominal | 0.64 |
| 16k | 146.1 | 143.5 | 150.1 | 146.1 | mtp | 1607 | Nominal | 68.7 | 90.3 | 87.0 | 87.0 | mtp | 1604 | Nominal | 0.60 |
| 32k | 142.9 | 146.8 | 152.3 | 146.8 | mtp | 1574 | Nominal | 91.8 | 94.7 | 91.5 | 91.8 | mtp | 1619 | Nominal | 0.63 |
| 64k | 146.2 | 155.5 | 151.0 | 151.0 | mtp | 1547 | Nominal | 90.7 | 72.7 | 73.0 | 73.0 | mtp | 1620 | Nominal | 0.48 |
| 128k | 128.6 | 138.4 | 138.2 | 138.2 | mtp | 1608 | Nominal | 63.2 | 71.5 | 72.8 | 71.5 | mtp | 1620 | Nominal | 0.52 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 1,122 | 0 | 3.73 | 1612 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 1,993 | 16 | 2.20 | 1610 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,049 | 25 | 4.15 | 1616 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,990 | 40 | 4.29 | 1566 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,976 | 54 | 4.30 | 1574 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,951 | 53 | 4.40 | 1567 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,934 | 65 | 4.42 | 1543 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,941 | 73 | 4.41 | 1508 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 1,906 | 82 | 4.51 | 1547 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,893 | 85 | 4.55 | 1567 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,877 | 97 | 4.60 | 1520 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,898 | 117 | 4.57 | 1559 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,897 | 131 | 4.67 | 1561 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,899 | 138 | 4.59 | 1562 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,882 | 133 | 4.66 | 1568 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,880 | 144 | 4.66 | 1565 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,877 | 160 | 4.66 | 1561 | Nominal |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 2,130 | 0 | 1.98 | 1581 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,014 | 16 | 2.16 | 1602 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 1,958 | 18 | 4.31 | 1553 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,951 | 31 | 4.35 | 1524 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,950 | 44 | 4.34 | 1584 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,949 | 60 | 4.39 | 1518 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,952 | 66 | 4.36 | 1560 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,924 | 75 | 4.44 | 1540 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 1,902 | 77 | 4.51 | 1572 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,964 | 97 | 4.39 | 1599 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,907 | 111 | 4.52 | 1577 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,870 | 98 | 4.60 | 1549 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,869 | 112 | 4.62 | 1533 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,874 | 135 | 4.63 | 1554 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,870 | 138 | 4.65 | 1545 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,864 | 146 | 4.68 | 1556 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,845 | 161 | 4.73 | 1593 | Nominal |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw** | Prefill | 1,122 | 1,993 | 2,049 | 1,976 | 1,906 | 1,877 |
|  | Decode | 168.6 | 151.1 | 146.1 | 146.8 | 151.0 | 138.2 |
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw (prose)** | Prefill | 2,130 | 2,014 | 1,958 | 1,950 | 1,902 | 1,845 |
|  | Decode | 62.8 | 96.9 | 87.0 | 91.8 | 73.0 | 71.5 |
