# Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw bench (Sushi 1.5.0)

Commit `75996eda`, binary SHA-256 `62184462b5e6c176` (built 2026-11-04 06:26:54). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261104-082107`, 2026-11-04.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1547 MHz, minimum 338 MHz over 508 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 183 s between the arms (die 96 C to 51 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 121.6 | 169.2 | 166.9 | 166.9 | mtp | 1603 | Nominal | 52.7 | 68.2 | 70.6 | 68.2 | mtp | 1620 | Nominal | 0.41 |
| 8k | 149.4 | 149.6 | 160.9 | 149.6 | mtp | 1591 | Nominal | 98.0 | 82.3 | 77.7 | 82.3 | mtp | 1618 | Nominal | 0.55 |
| 16k | 136.4 | 142.4 | 137.2 | 137.2 | mtp | 1566 | Nominal | 89.6 | 88.4 | 80.4 | 88.4 | mtp | 1620 | Nominal | 0.64 |
| 32k | 129.8 | 152.0 | 149.4 | 149.4 | mtp | 1560 | Nominal | 76.4 | 73.2 | 90.5 | 76.4 | mtp | 1585 | Nominal | 0.51 |
| 64k | 153.6 | 149.4 | 151.1 | 151.1 | mtp | 1564 | Nominal | 65.5 | 76.5 | 85.0 | 76.5 | mtp | 1566 | Nominal | 0.51 |
| 128k | 127.1 | 133.1 | 133.3 | 133.1 | mtp | 1582 | Nominal | 51.5 | 70.3 | 66.9 | 66.9 | mtp | 1608 | Nominal | 0.50 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 1,026 | 0 | 4.06 | 1602 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,023 | 15 | 2.14 | 1585 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,000 | 28 | 4.24 | 1569 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,931 | 30 | 4.39 | 1432 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,949 | 49 | 4.36 | 1560 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,942 | 60 | 4.40 | 1503 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,928 | 72 | 4.42 | 1476 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,909 | 83 | 4.50 | 1480 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 1,890 | 86 | 4.55 | 1526 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,867 | 103 | 4.63 | 1520 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,900 | 115 | 4.56 | 1547 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,898 | 114 | 4.56 | 1543 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,883 | 126 | 4.69 | 1518 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,864 | 128 | 4.66 | 1532 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,827 | 140 | 4.78 | 1514 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,850 | 160 | 4.73 | 1524 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,821 | 177 | 4.82 | 1535 | Nominal |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 1,864 | 0 | 2.27 | 1620 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,117 | 12 | 2.03 | 1620 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,002 | 21 | 4.20 | 1620 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 1,996 | 40 | 4.24 | 1603 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 1,903 | 66 | 4.47 | 1541 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 1,923 | 61 | 4.42 | 1577 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 1,866 | 70 | 4.56 | 1490 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 1,889 | 90 | 4.54 | 1497 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 1,839 | 79 | 4.64 | 1508 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 1,991 | 119 | 4.35 | 1608 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 1,856 | 103 | 4.66 | 1555 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 1,791 | 132 | 4.83 | 1464 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 1,728 | 141 | 5.01 | 1405 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 1,694 | 126 | 5.10 | 1336 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 1,778 | 152 | 4.88 | 1486 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 1,816 | 161 | 4.80 | 1436 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 1,849 | 186 | 4.74 | 1537 | Nominal |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw** | Prefill | 1,026 | 2,023 | 2,000 | 1,949 | 1,890 | 1,821 |
|  | Decode | 166.9 | 149.6 | 137.2 | 149.4 | 151.1 | 133.1 |
| **Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw (prose)** | Prefill | 1,864 | 2,117 | 2,002 | 1,903 | 1,839 | 1,849 |
|  | Decode | 68.2 | 82.3 | 88.4 | 76.4 | 76.5 | 66.9 |
