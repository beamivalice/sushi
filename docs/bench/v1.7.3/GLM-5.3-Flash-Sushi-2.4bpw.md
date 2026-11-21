# GLM-5.3-Flash-Sushi-2.4bpw bench (Sushi 1.7.3)

Commit `fc2fd843`, binary SHA-256 `eb0abb6db57ae23b` (built 2026-11-22 05:07:29). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-20261122-051706`, 2026-11-22.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

GPU clock, sampled every 500 ms: median 1478 MHz, minimum 961 MHz over 655 busy samples; thermal pressure seen: Nominal.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on. The server idled 72 s between the arms (die 94 C to 50 C).

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure | Prose 1 | Prose 2 | Prose 3 | Prose median | Prose mode | Prose MHz | Prose pressure | Prose / code |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| 4k | 46.5 | 73.6 | 72.6 | 72.6 | dflash | 1561 | Nominal | 41.1 | 41.0 | 41.0 | 41.0 | dflash | 1620 | Nominal | 0.56 |
| 8k | 70.3 | 70.8 | 70.1 | 70.3 | dflash | 1494 | Nominal | 52.7 | 53.6 | 53.5 | 53.5 | dflash | 1620 | Nominal | 0.76 |
| 16k | 67.1 | 68.9 | 68.9 | 68.9 | dflash | 1484 | Nominal | 36.5 | 36.6 | 36.7 | 36.6 | dflash | 1620 | Nominal | 0.53 |
| 32k | 66.2 | 66.3 | 66.9 | 66.3 | dflash | 1469 | Nominal | 58.3 | 59.7 | 58.5 | 58.5 | dflash | 1506 | Nominal | 0.88 |
| 64k | 72.5 | 72.6 | 73.2 | 72.6 | dflash | 1469 | Nominal | 38.2 | 38.8 | 39.0 | 38.8 | dflash | 1569 | Nominal | 0.53 |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 702 | 0 | 5.92 | 1619 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,064 to 8,192 | 4,064 | 4,128 | 811 | 18 | 5.20 | 1490 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,160 to 16,384 | 8,160 | 8,224 | 814 | 22 | 10.22 | 1462 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,352 to 24,576 | 16,352 | 8,224 | 825 | 28 | 10.06 | 1441 | Nominal | n/a | n/a | n/a | 77.5 | 1439 |
| 24,544 to 32,768 | 24,544 | 8,224 | 803 | 33 | 10.37 | 1448 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,736 to 40,960 | 32,736 | 8,224 | 746 | 44 | 11.14 | 1446 | Nominal | n/a | n/a | n/a | 86.5 | 1449 |
| 40,928 to 49,152 | 40,928 | 8,224 | 732 | 54 | 11.36 | 1475 | Nominal | n/a | n/a | n/a | 91.9 | 1466 |
| 49,120 to 57,344 | 49,120 | 8,224 | 708 | 56 | 11.75 | 1459 | Nominal | n/a | n/a | n/a | 92.6 | 1458 |
| 57,312 to 65,536 | 57,312 | 8,224 | 689 | 61 | 12.08 | 1459 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Prefill by window, prose (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 871 | 0 | 4.77 | 1620 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 4,064 to 8,192 | 4,064 | 4,128 | 821 | 19 | 5.13 | 1519 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 8,160 to 16,384 | 8,160 | 8,224 | 841 | 22 | 9.90 | 1505 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 16,352 to 24,576 | 16,352 | 8,224 | 836 | 31 | 9.94 | 1470 | Nominal | n/a | n/a | n/a | 79.3 | 1464 |
| 24,544 to 32,768 | 24,544 | 8,224 | 820 | 36 | 10.17 | 1435 | Nominal | n/a | n/a | n/a | n/a | n/a |
| 32,736 to 40,960 | 32,736 | 8,224 | 748 | 36 | 11.10 | 1445 | Nominal | n/a | n/a | n/a | 83.8 | 1442 |
| 40,928 to 49,152 | 40,928 | 8,224 | 728 | 42 | 11.42 | 1449 | Nominal | n/a | n/a | n/a | 87.2 | 1449 |
| 49,120 to 57,344 | 49,120 | 8,224 | 712 | 63 | 11.70 | 1455 | Nominal | n/a | n/a | n/a | 91.5 | 1461 |
| 57,312 to 65,536 | 57,312 | 8,224 | 698 | 61 | 11.93 | 1482 | Nominal | n/a | n/a | n/a | n/a | n/a |

## Decode energy per token

Energy per token (p = prefill, d = decode): CPU mJ, CPU ms and Minstr (millions of instructions) are this process's own work, cut where the prefill ends; the GPU cells are the whole machine's GPU energy over a request that is only a prefill, or a decode repeat (its restore and short prefill included), beside its mean GPU MHz. GPU energy per token rises with the GPU clock and is a record, not a quality or efficiency score.
CPU energy unavailable: the server is another process; --model loads it in this one.

Decode energy per token, code arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 998.7 | 1555 |
| 8k | n/a | n/a | n/a | 890.6 | 1497 |
| 16k | n/a | n/a | n/a | 939.1 | 1504 |
| 32k | n/a | n/a | n/a | 920.6 | 1478 |
| 64k | n/a | n/a | n/a | 847.3 | 1476 |

Decode energy per token, prose arm (median of the rung's samples):

| Rung | CPU mJ/t | CPU ms/t | Minstr/t | GPU mJ/t | GPU mean MHz |
|---|---:|---:|---:|---:|---:|
| 4k | n/a | n/a | n/a | 1818.2 | 1611 |
| 8k | n/a | n/a | n/a | 1480.8 | 1610 |
| 16k | n/a | n/a | n/a | 1652.2 | 1608 |
| 32k | n/a | n/a | n/a | 1210.3 | 1533 |
| 64k | n/a | n/a | n/a | 1746.7 | 1569 |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k |
|---|---|---:|---:|---:|---:|---:|
| **GLM-5.3-Flash-Sushi-2.4bpw** | Prefill | 702 | 811 | 814 | 803 | 689 |
|  | Decode | 72.6 | 70.3 | 68.9 | 66.3 | 72.6 |
| **GLM-5.3-Flash-Sushi-2.4bpw (prose)** | Prefill | 871 | 821 | 841 | 820 | 698 |
|  | Decode | 41.0 | 53.5 | 36.6 | 58.5 | 38.8 |
