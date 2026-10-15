# Qwen3.8-Flash-Next-Sushi-2.6bpw — 1.2.4

Version: **1.2.4**. Commit: `a65539e1`. Binary SHA-256: `85c6de9cac778da70fd1b56a275896cfbb1738a217c4bc1c7fa8098e7dfaf058`. Date: 2026-10-15 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/Qwen3.8-Flash-Next-Sushi-2.6bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m Qwen3.8-Flash-Next-Sushi-2.6bpw --rungs 2k,4k,8k,16k,32k,64k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 104.4 | 100.5 | 102.4 | 100.5 | 104.4 |
| Prefill tok/s | 2312.4 | 2292.1 | 2302.3 | 2292.1 | 2312.4 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2109 | 2 | 1886 | 95.2 | 1.1 | 3.10 |
| 4k | 4082 | 2 | 2118 | 96 | 1.9 | 3.65 |
| 8k | 8271 | 2 | 2164 | 85.9 | 3.8 | 3.38 |
| 16k | 16273 | 2 | 2141 | 91.8 | 7.6 | 3.66 |
| 32k | 32895 | 2 | 2052 | 89.3 | 16.0 | 3.60 |
| 64k | 65478 | 2 | 1961 | 77.3 | 33.4 | 3.96 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 5.74 |
| Speculative verdict | effective |
| Prefix cache speedup | 5.7× |
| Cached / prompt tokens | 1509 / 1540 |
| Batch streams | 4 |
| Single stream tok/s | 67 |
| Aggregate tok/s | 96.4 |
| Batch efficiency | 0.36 |
| Sustained initial / final tok/s | 102.4 / 93 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k — not comparable to default runs
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work
