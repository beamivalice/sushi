# MiMo-V2.6-Flash-Sushi-2.3bpw — 1.2.3

Version: **1.2.3**. Commit: `4a1f1f1b`. Binary SHA-256: `b5cd630a85ecb7c913aaaa2761a38d51f0aeefe7f730990d629bfdbb272a8452`. Date: 2026-10-13 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/MiMo-V2.6-Flash-Sushi-2.3bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m MiMo-V2.6-Flash-Sushi-2.3bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 52.6 | 57.8 | 55.2 | 52.6 | 57.8 |
| Prefill tok/s | 1247.4 | 1068.8 | 1158.1 | 1068.8 | 1247.4 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2090 | 2 | 1051 | 43.1 | 2.0 | 2.92 |
| 4k | 4088 | 2 | 1064 | 58.6 | 3.8 | 2.80 |
| 8k | 8269 | 2 | 1084 | 60.9 | 7.6 | 3.26 |
| 16k | 16334 | 2 | 1051 | 57.6 | 15.5 | 3.56 |
| 32k | 32725 | 2 | 973 | 53 | 33.6 | 3.65 |
| 64k | 65627 | 2 | 842 | 48.1 | 78.0 | 3.28 |
| 128k | 131064 | 2 | 668 | 46.7 | 196.1 | 3.40 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 5.82 |
| Speculative verdict | effective |
| Prefix cache speedup | 12.7× |
| Cached / prompt tokens | 1536 / 1537 |
| Batch streams | 4 |
| Single stream tok/s | 33.4 |
| Aggregate tok/s | 72 |
| Batch efficiency | 0.54 |
| Sustained initial / final tok/s | 55.2 / 59.5 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k — not comparable to default runs
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work
