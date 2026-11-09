# Qwen3.8-Flash-Next-Sushi-4bpw bench (Sushi 1.6.0)

Commit `ac3da7aa`, binary SHA-256 `fcdba9054a6ad870` (built 2026-11-09 21:59:29). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261110-005008`, 2026-11-10.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1565 MHz, minimum 611 MHz over 446 busy samples; thermal pressure seen: Nominal.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 71 s between the arms (die 95 C to 50 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 139.8 | 146.2 | 153.3 | 146.2 | mtp | 1615 | Nominal | 71.4 | 80.7 | 79.5 | 79.5 | mtp | 1619 | Nominal | 0.54 |
| 8k | 155.7 | 163.0 | 159.8 | 159.8 | mtp | 1580 | Nominal | 100.4 | 90.3 | 98.5 | 98.5 | mtp | 1619 | Nominal | 0.62 |
| 16k | 132.3 | 134.6 | 132.6 | 132.6 | mtp | 1591 | Nominal | 91.7 | 85.8 | 85.5 | 85.8 | mtp | 1619 | Nominal | 0.65 |
| 32k | 137.9 | 156.9 | 150.2 | 150.2 | mtp | 1600 | Nominal | 89.7 | 83.7 | 97.8 | 89.7 | mtp | 1617 | Nominal | 0.60 |
| 64k | 148.9 | 154.6 | 151.9 | 151.9 | mtp | 1579 | Nominal | 94.8 | 97.9 | 97.5 | 97.5 | mtp | 1620 | Nominal | 0.64 |
| 128k | 118.1 | 127.6 | 127.0 | 127.0 | mtp | 1601 | Nominal | 73.2 | 78.8 | 92.1 | 78.8 | mtp | 1608 | Nominal | 0.62 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 2,424 | 0 | 1.88 | 1604 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,417 | 45 | 1.95 | 1580 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,222 | 51 | 3.97 | 1531 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,241 | 72 | 3.98 | 1468 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,241 | 87 | 3.98 | 1517 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,200 | 120 | 4.09 | 1482 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,261 | 138 | 4.00 | 1503 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,194 | 140 | 4.11 | 1447 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,220 | 157 | 4.08 | 1514 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,105 | 206 | 4.36 | 1479 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,173 | 222 | 4.21 | 1498 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 2,176 | 236 | 4.25 | 1509 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,168 | 258 | 4.36 | 1505 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,164 | 280 | 4.32 | 1500 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 2,088 | 297 | 4.50 | 1501 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,101 | 308 | 4.47 | 1489 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 2,080 | 329 | 4.51 | 1549 | Nominal |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 1,956 | 0 | 2.27 | 1620 | Nominal |
| 4,065 to 8,192 | 4,065 | 4,127 | 2,275 | 52 | 2.05 | 1620 | Nominal |
| 8,161 to 16,384 | 8,161 | 8,223 | 2,222 | 45 | 3.95 | 1619 | Nominal |
| 16,353 to 24,576 | 16,353 | 8,223 | 2,178 | 94 | 4.09 | 1620 | Nominal |
| 24,545 to 32,768 | 24,545 | 8,223 | 2,147 | 107 | 4.16 | 1588 | Nominal |
| 32,737 to 40,960 | 32,737 | 8,223 | 2,104 | 114 | 4.25 | 1550 | Nominal |
| 40,929 to 49,152 | 40,929 | 8,223 | 2,201 | 135 | 4.10 | 1615 | Nominal |
| 49,121 to 57,344 | 49,121 | 8,223 | 2,140 | 158 | 4.16 | 1524 | Nominal |
| 57,313 to 65,536 | 57,313 | 8,223 | 2,194 | 180 | 4.09 | 1618 | Nominal |
| 65,505 to 73,728 | 65,505 | 8,223 | 2,087 | 197 | 4.37 | 1548 | Nominal |
| 73,697 to 81,920 | 73,697 | 8,223 | 2,153 | 218 | 4.22 | 1608 | Nominal |
| 81,889 to 90,112 | 81,889 | 8,223 | 2,157 | 240 | 4.28 | 1597 | Nominal |
| 90,081 to 98,304 | 90,081 | 8,223 | 2,128 | 251 | 4.35 | 1561 | Nominal |
| 98,273 to 106,496 | 98,273 | 8,223 | 2,094 | 270 | 4.43 | 1540 | Nominal |
| 106,465 to 114,688 | 106,465 | 8,223 | 2,062 | 295 | 4.52 | 1491 | Nominal |
| 114,657 to 122,880 | 114,657 | 8,223 | 2,035 | 309 | 4.54 | 1503 | Nominal |
| 122,849 to 131,072 | 122,849 | 8,223 | 2,011 | 326 | 4.64 | 1541 | Nominal |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next-Sushi-4bpw** | Prefill | 2,424 | 2,417 | 2,222 | 2,241 | 2,220 | 2,080 |
|  | Decode | 146.2 | 159.8 | 132.6 | 150.2 | 151.9 | 127.0 |
| **Qwen3.8-Flash-Next-Sushi-4bpw (prose)** | Prefill | 1,956 | 2,275 | 2,222 | 2,147 | 2,194 | 2,011 |
|  | Decode | 79.5 | 98.5 | 85.8 | 89.7 | 97.5 | 78.8 |
