# GLM-5.3-Flash-Sushi-2.4bpw bench (Sushi 1.6.0)

Commit `ac3da7aa`, binary SHA-256 `fcdba9054a6ad870` (built 2026-11-09 21:59:29). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261110-005805`, 2026-11-10.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1498 MHz, minimum 760 MHz over 1122 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 81 s between the arms (die 94 C to 49 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 46.2 | 73.5 | 71.5 | 71.5 | dflash | 1585 | Nominal | 40.9 | 40.9 | 40.2 | 40.9 | dflash | 1620 | Nominal | 0.57 |
| 8k | 69.2 | 70.6 | 69.0 | 69.2 | dflash | 1506 | Nominal | 51.5 | 53.5 | 53.4 | 53.4 | dflash | 1620 | Nominal | 0.77 |
| 16k | 66.3 | 67.3 | 66.9 | 66.9 | dflash | 1499 | Nominal | 36.3 | 36.7 | 36.7 | 36.7 | dflash | 1620 | Nominal | 0.55 |
| 32k | 64.8 | 66.6 | 65.6 | 65.6 | dflash | 1480 | Nominal | 56.5 | 58.0 | 60.0 | 58.0 | dflash | 1540 | Nominal | 0.88 |
| 64k | 70.4 | 71.8 | 70.8 | 70.8 | dflash | 1489 | Nominal | 37.0 | 37.7 | 38.2 | 37.7 | dflash | 1578 | Nominal | 0.53 |
| 128k | 62.8 | 64.7 | 63.4 | 63.4 | dflash | 1495 | Heavy | 37.4 | 38.4 | 38.0 | 38.0 | dflash | 1571 | Heavy | 0.60 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 716 | 0 | 5.98 | 1616 | Nominal |
| 4,064 to 8,192 | 4,064 | 4,128 | 796 | 41 | 5.49 | 1500 | Nominal |
| 8,160 to 16,384 | 8,160 | 8,224 | 807 | 48 | 10.47 | 1478 | Nominal |
| 16,352 to 24,576 | 16,352 | 8,224 | 802 | 54 | 10.55 | 1449 | Nominal |
| 24,544 to 32,768 | 24,544 | 8,224 | 803 | 52 | 10.55 | 1453 | Nominal |
| 32,736 to 40,960 | 32,736 | 8,224 | 744 | 74 | 11.37 | 1453 | Nominal |
| 40,928 to 49,152 | 40,928 | 8,224 | 731 | 81 | 11.57 | 1467 | Nominal |
| 49,120 to 57,344 | 49,120 | 8,224 | 705 | 87 | 11.98 | 1466 | Nominal |
| 57,312 to 65,536 | 57,312 | 8,224 | 688 | 91 | 12.29 | 1478 | Nominal |
| 65,504 to 73,728 | 65,504 | 8,224 | 668 | 99 | 12.60 | 1495 | Nominal |
| 73,696 to 81,920 | 73,696 | 8,224 | 656 | 97 | 12.87 | 1472 | Nominal |
| 81,888 to 90,112 | 81,888 | 8,224 | 645 | 119 | 13.10 | 1498 | Nominal |
| 90,080 to 98,304 | 90,080 | 8,224 | 623 | 117 | 13.54 | 1483 | Heavy |
| 98,272 to 106,496 | 98,272 | 8,224 | 609 | 137 | 13.87 | 1498 | Heavy |
| 106,464 to 114,688 | 106,464 | 8,224 | 593 | 135 | 14.24 | 1505 | Heavy |
| 114,656 to 122,880 | 114,656 | 8,224 | 584 | 149 | 14.48 | 1504 | Heavy |
| 122,848 to 131,072 | 122,848 | 8,224 | 570 | 160 | 14.79 | 1504 | Heavy |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 906 | 0 | 4.77 | 1620 | Nominal |
| 4,064 to 8,192 | 4,064 | 4,128 | 804 | 50 | 5.45 | 1528 | Nominal |
| 8,160 to 16,384 | 8,160 | 8,224 | 811 | 47 | 10.45 | 1516 | Nominal |
| 16,352 to 24,576 | 16,352 | 8,224 | 827 | 52 | 10.23 | 1486 | Nominal |
| 24,544 to 32,768 | 24,544 | 8,224 | 817 | 66 | 10.40 | 1474 | Nominal |
| 32,736 to 40,960 | 32,736 | 8,224 | 744 | 64 | 11.34 | 1457 | Nominal |
| 40,928 to 49,152 | 40,928 | 8,224 | 740 | 84 | 11.43 | 1458 | Nominal |
| 49,120 to 57,344 | 49,120 | 8,224 | 713 | 87 | 11.86 | 1463 | Nominal |
| 57,312 to 65,536 | 57,312 | 8,224 | 694 | 77 | 12.17 | 1488 | Nominal |
| 65,504 to 73,728 | 65,504 | 8,224 | 674 | 101 | 12.55 | 1482 | Nominal |
| 73,696 to 81,920 | 73,696 | 8,224 | 658 | 110 | 12.85 | 1493 | Nominal |
| 81,888 to 90,112 | 81,888 | 8,224 | 648 | 100 | 13.02 | 1490 | Nominal |
| 90,080 to 98,304 | 90,080 | 8,224 | 628 | 127 | 13.45 | 1491 | Nominal |
| 98,272 to 106,496 | 98,272 | 8,224 | 615 | 122 | 13.73 | 1507 | Nominal |
| 106,464 to 114,688 | 106,464 | 8,224 | 601 | 143 | 14.07 | 1527 | Nominal |
| 114,656 to 122,880 | 114,656 | 8,224 | 587 | 147 | 14.40 | 1507 | Nominal |
| 122,848 to 131,072 | 122,848 | 8,224 | 576 | 156 | 14.66 | 1519 | Heavy |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **GLM-5.3-Flash-Sushi-2.4bpw** | Prefill | 716 | 796 | 807 | 803 | 688 | 570 |
|  | Decode | 71.5 | 69.2 | 66.9 | 65.6 | 70.8 | 63.4 |
| **GLM-5.3-Flash-Sushi-2.4bpw (prose)** | Prefill | 906 | 804 | 811 | 817 | 694 | 576 |
|  | Decode | 40.9 | 53.4 | 36.7 | 58.0 | 37.7 | 38.0 |

## Notes

The server's loop stop cut a greedy repetition loop. The sample counts: its tok/s is a real decode.

- code, 131072 tokens (ladder): the loop stop ended it at 390 tokens
- code, 131072 tokens (repeat): the loop stop ended it at 390 tokens
- code, 131072 tokens (repeat): the loop stop ended it at 390 tokens
