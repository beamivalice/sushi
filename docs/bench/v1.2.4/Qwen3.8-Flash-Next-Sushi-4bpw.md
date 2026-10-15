# Qwen3.8-Flash-Next-Sushi-4bpw — 1.2.4

Version: **1.2.4**. Commit: `a65539e1`. Binary SHA-256: `85c6de9cac778da70fd1b56a275896cfbb1738a217c4bc1c7fa8098e7dfaf058`. Date: 2026-10-15 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/Qwen3.8-Flash-Next-Sushi-4bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m Qwen3.8-Flash-Next-Sushi-4bpw --rungs 2k,4k,8k,16k,32k,64k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 94.1 | 97.3 | 95.7 | 94.1 | 97.3 |
| Prefill tok/s | 2515.1 | 2472.1 | 2493.6 | 2472.1 | 2515.1 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2112 | 2 | 2095 | 79.2 | 1.0 | 3.46 |
| 4k | 4081 | 2 | 2339 | 82.2 | 1.7 | 3.64 |
| 8k | 8270 | 2 | 2378 | 82.6 | 3.5 | 3.60 |
| 16k | 16276 | 2 | 2293 | 86 | 7.1 | 3.57 |
| 32k | 32892 | 2 | 2203 | 86.6 | 14.9 | 4.14 |
| 64k | 65479 | 2 | 2165 | 81.6 | 30.2 | 3.59 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 4.85 |
| Speculative verdict | effective |
| Prefix cache speedup | 5.3× |
| Cached / prompt tokens | 1508 / 1539 |
| Batch streams | 4 |
| Single stream tok/s | 61.3 |
| Aggregate tok/s | 91.3 |
| Batch efficiency | 0.37 |
| Sustained initial / final tok/s | 95.7 / 81.1 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k — not comparable to default runs
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work
