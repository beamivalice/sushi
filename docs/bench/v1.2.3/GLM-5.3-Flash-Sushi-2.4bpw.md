# GLM-5.3-Flash-Sushi-2.4bpw — 1.2.3

Version: **1.2.3**. Commit: `4a1f1f1b`. Binary SHA-256: `b5cd630a85ecb7c913aaaa2761a38d51f0aeefe7f730990d629bfdbb272a8452`. Date: 2026-10-13 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **dflash**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/GLM-5.3-Flash-Sushi-2.4bpw --port 12345 --kv-quant 8 --no-update-check`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m GLM-5.3-Flash-Sushi-2.4bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 50 | 52.2 | 51.1 | 50 | 52.2 |
| Prefill tok/s | 809.4 | 770.2 | 789.8 | 770.2 | 809.4 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2079 | 2 | 841 | 44.7 | 2.5 | 2.53 |
| 4k | 4090 | 2 | 754 | 42.4 | 5.4 | 2.31 |
| 8k | 8266 | 2 | 747 | 45 | 11.1 | 2.65 |
| 16k | 16308 | 2 | 747 | 46 | 21.8 | 2.71 |
| 32k | 32781 | 2 | 748 | 41.9 | 43.8 | 2.79 |
| 64k | 65652 | 2 | 692 | 42.7 | 94.9 | 2.97 |
| 128k | 131092 | 2 | 615 | 39.9 | 213.0 | 2.78 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 3.56 |
| Speculative verdict | effective |
| Prefix cache speedup | 6.5× |
| Cached / prompt tokens | 1384 / 1418 |
| Batch streams | 4 |
| Single stream tok/s | 35.5 |
| Aggregate tok/s | 44.8 |
| Batch efficiency | 0.32 |
| Sustained initial / final tok/s | 51.1 / 47.2 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k, reasoning default — not comparable to default runs
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work
- engine rejected the reasoning effort param; ran at its default — not comparable to runs that set the effort
