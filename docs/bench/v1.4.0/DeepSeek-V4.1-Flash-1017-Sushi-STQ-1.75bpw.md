# DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw bench (Sushi 1.4.0)

Commit `edd3c1c2`, binary SHA-256 `12d2338d3abdd8a0` (built 2026-10-26 16:39:06). Hardware: Apple M5 Max 128 GB, AC power, ReleaseFast, `taskpolicy -a`, GPU lock `bench-dsv41-1.4.0-1`, 2026-10-26.

Server flags: --ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot.

Protocol: one fixed seeded TypeScript corpus as token ids on `/v1/completions`, greedy, `ignore_eos`, 512 decoded tokens a sample. The 4k request is cold; each later request restores the previous prompt-end point. A decode rung's samples 2 and 3 run right after its ladder request, and the first request counts. The server's own prefill time, the restore time (logged apart) taken out; TTFT is the request wall time minus its decode time. The server's default speculative mode stays on.

## Decode (tok/s)

| Rung | Sample 1 | Sample 2 | Sample 3 | Median | Speculative mode | Restored tokens |
|---|---:|---:|---:|---:|---|---|
| 4k | 79.4 | 80.4 | 80.0 | 80.0 | dspark | 0 / 4,065 / 4,065 |
| 8k | 76.9 | 75.7 | 76.1 | 76.1 | dspark | 4,065 / 8,161 / 8,161 |
| 16k | 74.0 | 73.4 | 73.4 | 73.4 | dspark | 8,161 / 16,353 / 16,353 |
| 32k | 75.3 | 75.1 | 76.1 | 75.3 | dspark | 24,545 / 32,737 / 32,737 |
| 64k | 75.7 | 75.6 | 75.4 | 75.6 | dspark | 57,313 / 65,505 / 65,505 |
| 128k | 68.5 | 69.5 | 69.5 | 69.5 | dspark | 122,849 / 131,041 / 131,041 |

## Prefill by window (tok/s)

| Window | Restored | New | Prefill tok/s | Restore ms | TTFT s |
|---|---:|---:|---:|---:|---:|
| 0 to 4,096 | 0 | 4,096 | 788 | 0 | 5.28 |
| 4,065 to 8,192 | 4,065 | 4,127 | 990 | 0 | 4.24 |
| 8,161 to 16,384 | 8,161 | 8,223 | 974 | 0 | 8.51 |
| 16,353 to 24,576 | 16,353 | 8,223 | 958 | 0 | 8.63 |
| 24,545 to 32,768 | 24,545 | 8,223 | 971 | 0 | 8.52 |
| 32,737 to 40,960 | 32,737 | 8,223 | 968 | 0 | 8.55 |
| 40,929 to 49,152 | 40,929 | 8,223 | 973 | 0 | 8.50 |
| 49,121 to 57,344 | 49,121 | 8,223 | 966 | 0 | 8.57 |
| 57,313 to 65,536 | 57,313 | 8,223 | 954 | 0 | 8.68 |
| 65,505 to 73,728 | 65,505 | 8,223 | 929 | 0 | 8.91 |
| 73,697 to 81,920 | 73,697 | 8,223 | 927 | 0 | 8.92 |
| 81,889 to 90,112 | 81,889 | 8,223 | 921 | 0 | 8.97 |
| 90,081 to 98,304 | 90,081 | 8,223 | 912 | 0 | 9.07 |
| 98,273 to 106,496 | 98,273 | 8,223 | 906 | 0 | 9.13 |
| 106,465 to 114,688 | 106,465 | 8,223 | 894 | 0 | 9.26 |
| 114,657 to 122,880 | 114,657 | 8,223 | 879 | 0 | 9.41 |
| 122,849 to 131,072 | 122,849 | 8,223 | 879 | 0 | 9.43 |

## Summary row

Prefill is the window that ends at the rung; decode is the median of the samples.

| Sushi |  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|
| **DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw** | Prefill | 788 | 990 | 974 | 971 | 954 | 879 |
|  | Decode | 80.0 | 76.1 | 73.4 | 75.3 | 75.6 | 69.5 |
