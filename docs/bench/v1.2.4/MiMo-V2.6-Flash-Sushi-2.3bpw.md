# MiMo-V2.6-Flash-Sushi-2.3bpw — 1.2.4

Version: **1.2.4**. Commit: `a65539e1`. Binary SHA-256: `85c6de9cac778da70fd1b56a275896cfbb1738a217c4bc1c7fa8098e7dfaf058`. Date: 2026-10-15 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/MiMo-V2.6-Flash-Sushi-2.3bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

The server ran with its own empty SSD prefix cache (a fresh `HOME`); see the [version summary](summary.md).

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m MiMo-V2.6-Flash-Sushi-2.3bpw --rungs 2k,4k,8k,16k,32k,64k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 49.1 | 62.6 | 55.9 | 49.1 | 62.6 |
| Prefill tok/s | 1288.3 | 1272.3 | 1280.3 | 1272.3 | 1288.3 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2091 | 2 | 1259 | 44.5 | 1.7 | 2.60 |
| 4k | 4090 | 2 | 1218 | 63.6 | 3.4 | 2.77 |
| 8k | 8270 | 2 | 1145 | 76.7 | 7.2 | 3.66 |
| 16k | 16330 | 2 | 1094 | 64.8 | 14.9 | 3.57 |
| 32k | 32728 | 2 | 966 | 54.4 | 33.9 | 3.47 |
| 64k | 65631 | 2 | 784 | 47.8 | 83.7 | 2.83 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 6 |
| Speculative verdict | effective |
| Prefix cache speedup | 11.3× |
| Cached / prompt tokens | 1535 / 1536 |
| Batch streams | 4 |
| Single stream tok/s | 37.8 |
| Aggregate tok/s | 80.6 |
| Batch efficiency | 0.53 |
| Sustained initial / final tok/s | 55.9 / 48.4 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k — not comparable to default runs
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work
