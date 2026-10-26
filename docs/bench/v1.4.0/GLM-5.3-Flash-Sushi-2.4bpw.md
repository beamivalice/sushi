# GLM-5.3-Flash-Sushi-2.4bpw bench (Sushi 1.4.0)

Commit `d361c70b`, binary SHA-256 `4d25c2d7ba7a5684` (built 2026-10-27 06:33:21). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-gate140b-GLM-5.3-Flash-Sushi-2.4bpw`, 2026-10-27.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1463 MHz, minimum 1156 MHz over 1497 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample. The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on.

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 44.1 | 72.5 | 71.5 | 71.5 | dflash | 1570 | Nominal | 50.6 | 51.0 | 50.9 | 50.9 | dflash | 1476 | Heavy | 0.71 |
| 8k | 69.4 | 70.4 | 69.3 | 69.4 | dflash | 1519 | Nominal | 38.2 | 38.4 | 38.4 | 38.4 | dflash | 1468 | Heavy | 0.55 |
| 16k | 66.7 | 67.9 | 67.3 | 67.3 | dflash | 1489 | Nominal | 34.8 | 34.6 | 34.7 | 34.7 | dflash | 1455 | Heavy | 0.52 |
| 32k | 65.3 | 65.8 | 65.8 | 65.8 | dflash | 1484 | Nominal | 66.4 | 66.8 | 66.0 | 66.4 | dflash | 1377 | Heavy | 1.01 |
| 64k | 71.4 | 73.0 | 71.8 | 71.8 | dflash | 1495 | Nominal | 33.5 | 34.0 | 33.2 | 33.5 | dflash | 1432 | Heavy | 0.47 |
| 128k | 63.4 | 64.1 | 63.8 | 63.8 | dflash | 1486 | Heavy | 45.6 | 45.8 | 45.7 | 45.7 | dflash | 1411 | Heavy | 0.72 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 928 | 0 | 4.49 | 1620 | Nominal |
| 4,064 to 8,192 | 4,064 | 4,128 | 775 | 20 | 5.44 | 1510 | Nominal |
| 8,160 to 16,384 | 8,160 | 8,224 | 814 | 17 | 10.20 | 1488 | Nominal |
| 16,352 to 24,576 | 16,352 | 8,224 | 804 | 23 | 10.32 | 1424 | Nominal |
| 24,544 to 32,768 | 24,544 | 8,224 | 805 | 37 | 10.34 | 1472 | Nominal |
| 32,736 to 40,960 | 32,736 | 8,224 | 738 | 38 | 11.26 | 1446 | Nominal |
| 40,928 to 49,152 | 40,928 | 8,224 | 726 | 50 | 11.45 | 1456 | Nominal |
| 49,120 to 57,344 | 49,120 | 8,224 | 696 | 62 | 11.96 | 1461 | Nominal |
| 57,312 to 65,536 | 57,312 | 8,224 | 688 | 71 | 12.13 | 1475 | Nominal |
| 65,504 to 73,728 | 65,504 | 8,224 | 668 | 65 | 12.45 | 1469 | Nominal |
| 73,696 to 81,920 | 73,696 | 8,224 | 660 | 75 | 12.62 | 1480 | Nominal |
| 81,888 to 90,112 | 81,888 | 8,224 | 642 | 77 | 12.95 | 1485 | Nominal |
| 90,080 to 98,304 | 90,080 | 8,224 | 625 | 76 | 13.30 | 1486 | Nominal |
| 98,272 to 106,496 | 98,272 | 8,224 | 609 | 95 | 13.69 | 1486 | Heavy |
| 106,464 to 114,688 | 106,464 | 8,224 | 598 | 86 | 13.91 | 1500 | Heavy |
| 114,656 to 122,880 | 114,656 | 8,224 | 584 | 94 | 14.26 | 1493 | Heavy |
| 122,848 to 131,072 | 122,848 | 8,224 | 573 | 105 | 14.52 | 1494 | Heavy |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 878 | 0 | 4.79 | 1448 | Heavy |
| 4,064 to 8,192 | 4,064 | 4,128 | 759 | 14 | 5.53 | 1472 | Heavy |
| 8,160 to 16,384 | 8,160 | 8,224 | 752 | 20 | 11.04 | 1446 | Heavy |
| 16,352 to 24,576 | 16,352 | 8,224 | 757 | 40 | 10.96 | 1329 | Heavy |
| 24,544 to 32,768 | 24,544 | 8,224 | 777 | 50 | 10.77 | 1379 | Heavy |
| 32,736 to 40,960 | 32,736 | 8,224 | 708 | 69 | 11.76 | 1375 | Heavy |
| 40,928 to 49,152 | 40,928 | 8,224 | 704 | 68 | 11.84 | 1384 | Heavy |
| 49,120 to 57,344 | 49,120 | 8,224 | 682 | 83 | 12.23 | 1401 | Heavy |
| 57,312 to 65,536 | 57,312 | 8,224 | 646 | 52 | 12.87 | 1432 | Heavy |
| 65,504 to 73,728 | 65,504 | 8,224 | 615 | 85 | 13.54 | 1347 | Heavy |
| 73,696 to 81,920 | 73,696 | 8,224 | 606 | 70 | 13.72 | 1379 | Heavy |
| 81,888 to 90,112 | 81,888 | 8,224 | 601 | 88 | 13.87 | 1392 | Heavy |
| 90,080 to 98,304 | 90,080 | 8,224 | 576 | 90 | 14.45 | 1373 | Heavy |
| 98,272 to 106,496 | 98,272 | 8,224 | 579 | 100 | 14.38 | 1420 | Heavy |
| 106,464 to 114,688 | 106,464 | 8,224 | 567 | 106 | 14.70 | 1419 | Heavy |
| 114,656 to 122,880 | 114,656 | 8,224 | 549 | 116 | 15.18 | 1393 | Heavy |
| 122,848 to 131,072 | 122,848 | 8,224 | 536 | 115 | 15.60 | 1382 | Heavy |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **GLM-5.3-Flash-Sushi-2.4bpw** | Prefill | 928 | 775 | 814 | 805 | 688 | 573 |
|  | Decode | 71.5 | 69.4 | 67.3 | 65.8 | 71.8 | 63.8 |
| **GLM-5.3-Flash-Sushi-2.4bpw (prose)** | Prefill | 878 | 759 | 752 | 777 | 646 | 536 |
|  | Decode | 50.9 | 38.4 | 34.7 | 66.4 | 33.5 | 45.7 |

## Notes

The server's loop stop cut a greedy repetition loop. The sample counts: its tok/s is a real decode.

- code, 131072 tokens (ladder): the loop stop ended it at 390 tokens
- code, 131072 tokens (repeat): the loop stop ended it at 390 tokens
- code, 131072 tokens (repeat): the loop stop ended it at 390 tokens
