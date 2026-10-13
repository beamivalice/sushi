# Qwen3.8-Flash-Next-Sushi-2.6bpw — 1.2.3

Version: **1.2.3**. Commit: `4a1f1f1b`. Binary SHA-256: `b5cd630a85ecb7c913aaaa2761a38d51f0aeefe7f730990d629bfdbb272a8452`. Date: 2026-10-13 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/Qwen3.8-Flash-Next-Sushi-2.6bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m Qwen3.8-Flash-Next-Sushi-2.6bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 92.6 | 101.1 | 96.9 | 92.6 | 101.1 |
| Prefill tok/s | 2188.9 | 2149.6 | 2169.2 | 2149.6 | 2188.9 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2109 | 2 | 1648 | 84.3 | 1.3 | 3.08 |
| 4k | 4082 | 2 | 1839 | 88.3 | 2.2 | 3.31 |
| 8k | 8271 | 2 | 1912 | 83.2 | 4.3 | 3.60 |
| 16k | 16273 | 2 | 1972 | 75.2 | 8.3 | 3.00 |
| 32k | 32895 | 2 | 1984 | 83.9 | 16.6 | 3.76 |
| 64k | 65478 | 2 | 1953 | 79.6 | 33.5 | 3.81 |
| 128k | 131151 | 2 | 1901 | 74.6 | 69.0 | 3.37 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 5.57 |
| Speculative verdict | effective |
| Prefix cache speedup | 6.6× |
| Cached / prompt tokens | 1509 / 1540 |
| Batch streams | 4 |
| Single stream tok/s | 76.7 |
| Aggregate tok/s | 87.6 |
| Batch efficiency | 0.29 |
| Sustained initial / final tok/s | 96.9 / 89.3 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k — not comparable to default runs
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work
