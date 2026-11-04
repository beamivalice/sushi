# GLM-5.3-Flash-Sushi-2.4bpw bench (Sushi 1.5.0)

Commit `75996eda`, binary SHA-256 `62184462b5e6c176` (built 2026-11-04 06:26:54). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261104-075729`, 2026-11-04.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1471 MHz, minimum 912 MHz over 1142 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 101 s between the arms (die 94 C to 50 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 44.8 | 72.3 | 70.7 | 70.7 | dflash | 1558 | Nominal | 40.8 | 40.4 | 40.5 | 40.5 | dflash | 1620 | Nominal | 0.57 |
| 8k | 69.1 | 70.3 | 70.5 | 70.3 | dflash | 1505 | Nominal | 51.4 | 53.2 | 53.0 | 53.0 | dflash | 1620 | Nominal | 0.75 |
| 16k | 66.5 | 67.5 | 67.0 | 67.0 | dflash | 1481 | Nominal | 35.8 | 36.4 | 36.4 | 36.4 | dflash | 1620 | Nominal | 0.54 |
| 32k | 64.7 | 66.5 | 65.3 | 65.3 | dflash | 1457 | Nominal | 56.3 | 58.9 | 57.0 | 57.0 | dflash | 1523 | Moderate | 0.87 |
| 64k | 70.6 | 71.7 | 70.8 | 70.8 | dflash | 1458 | Nominal | 37.0 | 36.6 | 37.3 | 37.0 | dflash | 1504 | Heavy | 0.52 |
| 128k | 62.1 | 63.2 | 62.5 | 62.5 | dflash | 1446 | Heavy | 36.6 | 37.2 | 36.6 | 36.6 | dflash | 1492 | Heavy | 0.59 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 751 | 0 | 5.54 | 1619 | Nominal |
| 4,064 to 8,192 | 4,064 | 4,128 | 796 | 30 | 5.30 | 1479 | Nominal |
| 8,160 to 16,384 | 8,160 | 8,224 | 822 | 15 | 10.11 | 1463 | Nominal |
| 16,352 to 24,576 | 16,352 | 8,224 | 810 | 33 | 10.26 | 1427 | Nominal |
| 24,544 to 32,768 | 24,544 | 8,224 | 813 | 59 | 10.27 | 1441 | Nominal |
| 32,736 to 40,960 | 32,736 | 8,224 | 749 | 58 | 11.11 | 1454 | Nominal |
| 40,928 to 49,152 | 40,928 | 8,224 | 726 | 51 | 11.45 | 1450 | Nominal |
| 49,120 to 57,344 | 49,120 | 8,224 | 703 | 79 | 11.85 | 1455 | Nominal |
| 57,312 to 65,536 | 57,312 | 8,224 | 686 | 82 | 12.16 | 1458 | Nominal |
| 65,504 to 73,728 | 65,504 | 8,224 | 669 | 92 | 12.46 | 1468 | Heavy |
| 73,696 to 81,920 | 73,696 | 8,224 | 645 | 100 | 12.93 | 1461 | Heavy |
| 81,888 to 90,112 | 81,888 | 8,224 | 634 | 102 | 13.15 | 1456 | Heavy |
| 90,080 to 98,304 | 90,080 | 8,224 | 618 | 73 | 13.46 | 1465 | Heavy |
| 98,272 to 106,496 | 98,272 | 8,224 | 605 | 106 | 13.79 | 1471 | Heavy |
| 106,464 to 114,688 | 106,464 | 8,224 | 591 | 113 | 14.12 | 1477 | Heavy |
| 114,656 to 122,880 | 114,656 | 8,224 | 573 | 118 | 14.56 | 1459 | Heavy |
| 122,848 to 131,072 | 122,848 | 8,224 | 562 | 114 | 14.80 | 1466 | Heavy |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 876 | 0 | 4.75 | 1620 | Nominal |
| 4,064 to 8,192 | 4,064 | 4,128 | 813 | 27 | 5.19 | 1518 | Nominal |
| 8,160 to 16,384 | 8,160 | 8,224 | 821 | 37 | 10.16 | 1522 | Nominal |
| 16,352 to 24,576 | 16,352 | 8,224 | 827 | 51 | 10.07 | 1457 | Nominal |
| 24,544 to 32,768 | 24,544 | 8,224 | 800 | 34 | 10.42 | 1412 | Moderate |
| 32,736 to 40,960 | 32,736 | 8,224 | 748 | 63 | 11.13 | 1447 | Moderate |
| 40,928 to 49,152 | 40,928 | 8,224 | 717 | 54 | 11.59 | 1438 | Moderate |
| 49,120 to 57,344 | 49,120 | 8,224 | 692 | 63 | 12.02 | 1445 | Moderate |
| 57,312 to 65,536 | 57,312 | 8,224 | 679 | 79 | 12.28 | 1467 | Moderate |
| 65,504 to 73,728 | 65,504 | 8,224 | 661 | 67 | 12.59 | 1465 | Heavy |
| 73,696 to 81,920 | 73,696 | 8,224 | 643 | 72 | 12.93 | 1467 | Heavy |
| 81,888 to 90,112 | 81,888 | 8,224 | 629 | 104 | 13.27 | 1472 | Heavy |
| 90,080 to 98,304 | 90,080 | 8,224 | 613 | 104 | 13.59 | 1468 | Heavy |
| 98,272 to 106,496 | 98,272 | 8,224 | 594 | 82 | 14.01 | 1465 | Heavy |
| 106,464 to 114,688 | 106,464 | 8,224 | 581 | 93 | 14.32 | 1466 | Heavy |
| 114,656 to 122,880 | 114,656 | 8,224 | 572 | 115 | 14.57 | 1475 | Heavy |
| 122,848 to 131,072 | 122,848 | 8,224 | 556 | 111 | 14.99 | 1455 | Heavy |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **GLM-5.3-Flash-Sushi-2.4bpw** | Prefill | 751 | 796 | 822 | 813 | 686 | 562 |
|  | Decode | 70.7 | 70.3 | 67.0 | 65.3 | 70.8 | 62.5 |
| **GLM-5.3-Flash-Sushi-2.4bpw (prose)** | Prefill | 876 | 813 | 821 | 800 | 679 | 556 |
|  | Decode | 40.5 | 53.0 | 36.4 | 57.0 | 37.0 | 36.6 |

## Notes

The server's loop stop cut a greedy repetition loop. The sample counts: its tok/s is a real decode.

- code, 131072 tokens (ladder): the loop stop ended it at 390 tokens
- code, 131072 tokens (repeat): the loop stop ended it at 390 tokens
- code, 131072 tokens (repeat): the loop stop ended it at 390 tokens
