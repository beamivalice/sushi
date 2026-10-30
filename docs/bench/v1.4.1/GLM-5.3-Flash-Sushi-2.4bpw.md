# GLM-5.3-Flash-Sushi-2.4bpw bench (Sushi 1.4.1)

Commit `b358cd38`, binary SHA-256 `6944232d09a932d2` (built 2026-10-30 16:00:48). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261030-164837`, 2026-10-30.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1389 MHz, minimum 1214 MHz over 1206 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 161 s between the arms (die 86 C to 50 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 45.8 | 71.9 | 70.5 | 70.5 | dflash | 1567 | Nominal | 41.0 | 41.0 | 41.0 | 41.0 | dflash | 1620 | Nominal | 0.58 |
| 8k | 68.7 | 69.8 | 70.0 | 69.8 | dflash | 1517 | Moderate | 51.2 | 53.5 | 53.4 | 53.4 | dflash | 1620 | Nominal | 0.76 |
| 16k | 65.9 | 66.8 | 66.1 | 66.1 | dflash | 1484 | Heavy | 36.1 | 36.3 | 36.4 | 36.3 | dflash | 1617 | Moderate | 0.55 |
| 32k | 63.5 | 64.0 | 63.5 | 63.5 | dflash | 1430 | Heavy | 56.5 | 57.6 | 59.4 | 57.6 | dflash | 1531 | Moderate | 0.91 |
| 64k | 65.2 | 64.4 | 63.6 | 64.4 | dflash | 1292 | Heavy | 36.3 | 35.8 | 36.0 | 36.0 | dflash | 1478 | Heavy | 0.56 |
| 128k | 55.4 | 55.6 | 55.5 | 55.5 | dflash | 1257 | Heavy | 34.0 | 33.7 | 34.1 | 34.0 | dflash | 1315 | Heavy | 0.61 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 710 | 0 | 5.85 | 1620 | Nominal |
| 4,064 to 8,192 | 4,064 | 4,128 | 796 | 27 | 5.31 | 1479 | Nominal |
| 8,160 to 16,384 | 8,160 | 8,224 | 819 | 36 | 10.17 | 1468 | Moderate |
| 16,352 to 24,576 | 16,352 | 8,224 | 799 | 25 | 10.39 | 1405 | Heavy |
| 24,544 to 32,768 | 24,544 | 8,224 | 804 | 55 | 10.39 | 1426 | Heavy |
| 32,736 to 40,960 | 32,736 | 8,224 | 724 | 67 | 11.51 | 1375 | Heavy |
| 40,928 to 49,152 | 40,928 | 8,224 | 696 | 72 | 11.96 | 1383 | Heavy |
| 49,120 to 57,344 | 49,120 | 8,224 | 675 | 80 | 12.35 | 1366 | Heavy |
| 57,312 to 65,536 | 57,312 | 8,224 | 657 | 86 | 12.69 | 1344 | Heavy |
| 65,504 to 73,728 | 65,504 | 8,224 | 608 | 91 | 13.70 | 1297 | Heavy |
| 73,696 to 81,920 | 73,696 | 8,224 | 602 | 99 | 13.84 | 1308 | Heavy |
| 81,888 to 90,112 | 81,888 | 8,224 | 585 | 91 | 14.23 | 1317 | Heavy |
| 90,080 to 98,304 | 90,080 | 8,224 | 572 | 100 | 14.55 | 1329 | Heavy |
| 98,272 to 106,496 | 98,272 | 8,224 | 552 | 98 | 15.08 | 1320 | Heavy |
| 106,464 to 114,688 | 106,464 | 8,224 | 539 | 115 | 15.48 | 1320 | Heavy |
| 114,656 to 122,880 | 114,656 | 8,224 | 525 | 95 | 15.85 | 1317 | Heavy |
| 122,848 to 131,072 | 122,848 | 8,224 | 515 | 114 | 16.14 | 1304 | Heavy |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 892 | 0 | 4.66 | 1620 | Nominal |
| 4,064 to 8,192 | 4,064 | 4,128 | 826 | 27 | 5.11 | 1556 | Nominal |
| 8,160 to 16,384 | 8,160 | 8,224 | 831 | 35 | 10.03 | 1500 | Nominal |
| 16,352 to 24,576 | 16,352 | 8,224 | 831 | 48 | 10.02 | 1464 | Moderate |
| 24,544 to 32,768 | 24,544 | 8,224 | 817 | 57 | 10.24 | 1449 | Moderate |
| 32,736 to 40,960 | 32,736 | 8,224 | 752 | 39 | 11.05 | 1436 | Heavy |
| 40,928 to 49,152 | 40,928 | 8,224 | 727 | 64 | 11.46 | 1432 | Heavy |
| 49,120 to 57,344 | 49,120 | 8,224 | 704 | 82 | 11.85 | 1434 | Heavy |
| 57,312 to 65,536 | 57,312 | 8,224 | 683 | 71 | 12.19 | 1436 | Heavy |
| 65,504 to 73,728 | 65,504 | 8,224 | 644 | 80 | 12.93 | 1383 | Heavy |
| 73,696 to 81,920 | 73,696 | 8,224 | 627 | 95 | 13.30 | 1370 | Heavy |
| 81,888 to 90,112 | 81,888 | 8,224 | 608 | 92 | 13.71 | 1365 | Heavy |
| 90,080 to 98,304 | 90,080 | 8,224 | 585 | 104 | 14.25 | 1349 | Heavy |
| 98,272 to 106,496 | 98,272 | 8,224 | 558 | 104 | 14.92 | 1332 | Heavy |
| 106,464 to 114,688 | 106,464 | 8,224 | 540 | 110 | 15.44 | 1321 | Heavy |
| 114,656 to 122,880 | 114,656 | 8,224 | 529 | 116 | 15.75 | 1325 | Heavy |
| 122,848 to 131,072 | 122,848 | 8,224 | 522 | 117 | 15.95 | 1329 | Heavy |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **GLM-5.3-Flash-Sushi-2.4bpw** | Prefill | 710 | 796 | 819 | 804 | 657 | 515 |
|  | Decode | 70.5 | 69.8 | 66.1 | 63.5 | 64.4 | 55.5 |
| **GLM-5.3-Flash-Sushi-2.4bpw (prose)** | Prefill | 892 | 826 | 831 | 817 | 683 | 522 |
|  | Decode | 41.0 | 53.4 | 36.3 | 57.6 | 36.0 | 34.0 |

## Notes

The server's loop stop cut a greedy repetition loop. The sample counts: its tok/s is a real decode.

- code, 131072 tokens (ladder): the loop stop ended it at 390 tokens
- code, 131072 tokens (repeat): the loop stop ended it at 390 tokens
- code, 131072 tokens (repeat): the loop stop ended it at 390 tokens
