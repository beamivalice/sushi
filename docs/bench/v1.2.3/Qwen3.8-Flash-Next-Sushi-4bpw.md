# Qwen3.8-Flash-Next-Sushi-4bpw — 1.2.3

Version: **1.2.3**. Commit: `4a1f1f1b`. Binary SHA-256: `b5cd630a85ecb7c913aaaa2761a38d51f0aeefe7f730990d629bfdbb272a8452`. Date: 2026-10-13 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/Qwen3.8-Flash-Next-Sushi-4bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m Qwen3.8-Flash-Next-Sushi-4bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 88.8 | 93.9 | 91.3 | 88.8 | 93.9 |
| Prefill tok/s | 2447.3 | 2421.7 | 2434.5 | 2421.7 | 2447.3 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2109 | 2 | 1810 | 83.5 | 1.2 | 3.18 |
| 4k | 4082 | 2 | 2112 | 72.8 | 1.9 | 3.69 |
| 8k | 8271 | 2 | 2153 | 77.3 | 3.8 | 3.84 |
| 16k | 16273 | 2 | 2143 | 82.9 | 7.6 | 3.92 |
| 32k | 32895 | 2 | 2201 | 82.8 | 14.9 | 3.81 |
| 64k | 65478 | 2 | 2166 | 76.2 | 30.2 | 3.15 |
| 128k | 131151 | 2 | 2112 | 77.4 | 62.1 | 3.74 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 5.15 |
| Speculative verdict | effective |
| Prefix cache speedup | 5.9× |
| Cached / prompt tokens | 1509 / 1540 |
| Batch streams | 4 |
| Single stream tok/s | 60.4 |
| Aggregate tok/s | 87.4 |
| Batch efficiency | 0.36 |
| Sustained initial / final tok/s | 91.3 / 79.8 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k — not comparable to default runs
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work
