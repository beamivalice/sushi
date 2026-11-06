# GLM-5.3-Flash-Sushi-2.4bpw bench (Sushi 1.5.1)

Commit `399895f6`, binary SHA-256 `4391a90f528f7aaa` (built 2026-11-06 14:56:50). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-v151-glm-24`, 2026-11-06.

Server flags: defaults. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

**Aborted, not a measurement:** the GPU was throttled during the code 73728-token ladder request: median 338 MHz (abort at 1100 MHz or under), thermal pressure Heavy

Released on this run's code rungs 4k to 64k (owner, 2026-11-06): each passed the abort rule; the 73k ladder and the prose arm did not run. A second run on the same binary throttled at the 64k ladder (347 MHz, Heavy) and read 4k to 32k within 1% of this one. The 64k rung ran under Heavy pressure at 1294 MHz: 0.0488 tok/s per MHz against 1.5.0's 0.0486 (70.8 at 1458 MHz).

GPU clock, sampled every 500 ms: median 1423 MHz, minimum 338 MHz over 457 busy samples; thermal pressure seen: Nominal, Moderate, Heavy.

Protocol: token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample (128 on prose). The code arm is one fixed seeded TypeScript corpus; the prose arm is Pride and Prejudice (public domain), where speculative acceptance is lower. Each arm runs in turn on the same server: the 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on.

## Decode (tok/s)

| Rung | Code 1 | Code 2 | Code 3 | Code median | Code mode | Code MHz | Code pressure |
|---|---:|---:|---:|---:|---|---:|---|
| 4k | 46.2 | 72.3 | 70.4 | 70.4 | dflash | 1575 | Nominal |
| 8k | 68.5 | 69.4 | 69.4 | 69.4 | dflash | 1510 | Nominal |
| 16k | 65.7 | 66.6 | 66.0 | 66.0 | dflash | 1490 | Nominal |
| 32k | 63.5 | 64.0 | 63.6 | 63.6 | dflash | 1452 | Nominal |
| 64k | 64.4 | 63.1 | 62.7 | 63.1 | dflash | 1294 | Heavy |

## Prefill by window, code (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s | GPU MHz | Pressure |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 to 4,096 | 0 | 4,096 | 710 | 0 | 5.86 | 1619 | Nominal |
| 4,064 to 8,192 | 4,064 | 4,128 | 785 | 12 | 5.35 | 1483 | Nominal |
| 8,160 to 16,384 | 8,160 | 8,224 | 819 | 26 | 10.16 | 1467 | Nominal |
| 16,352 to 24,576 | 16,352 | 8,224 | 802 | 34 | 10.37 | 1410 | Nominal |
| 24,544 to 32,768 | 24,544 | 8,224 | 803 | 42 | 10.38 | 1433 | Nominal |
| 32,736 to 40,960 | 32,736 | 8,224 | 722 | 51 | 11.52 | 1380 | Nominal |
| 40,928 to 49,152 | 40,928 | 8,224 | 695 | 53 | 11.96 | 1369 | Heavy |
| 49,120 to 57,344 | 49,120 | 8,224 | 673 | 62 | 12.36 | 1379 | Heavy |
| 57,312 to 65,536 | 57,312 | 8,224 | 648 | 65 | 12.85 | 1334 | Heavy |
| 65,504 to 73,728 | 65,504 | 8,224 | 305 | 66 | 27.10 | 338 | Heavy |

## Summary rows

Prefill is the window that ends at the rung; decode is the median of the samples. The code rows are the ones to compare across releases.

| Sushi |  | 4k | 8k | 16k | 32k | 64k |
|---|---|---:|---:|---:|---:|---:|
| **GLM-5.3-Flash-Sushi-2.4bpw** | Prefill | 710 | 785 | 819 | 803 | 648 |
|  | Decode | 70.4 | 69.4 | 66.0 | 63.6 | 63.1 |

## Warnings

A row with a warning is not a valid measurement.

- code, 73728 tokens (ladder): the GPU clock or thermal pressure failed the abort rule (see the GPU columns)
