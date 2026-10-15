# GLM-5.3-Flash-Sushi-2.4bpw — 1.2.4

Version: **1.2.4**. Commit: `a65539e1`. Binary SHA-256: `85c6de9cac778da70fd1b56a275896cfbb1738a217c4bc1c7fa8098e7dfaf058`. Date: 2026-10-15 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **dflash**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/GLM-5.3-Flash-Sushi-2.4bpw --port 12345 --kv-quant 8 --no-update-check`.

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m GLM-5.3-Flash-Sushi-2.4bpw --rungs 2k,4k,8k,16k,32k,64k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 54.7 | 51.6 | 53.1 | 51.6 | 54.7 |
| Prefill tok/s | 886.6 | 874.7 | 880.7 | 874.7 | 886.6 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2078 | 2 | 879 | 47.6 | 2.4 | 2.50 |
| 4k | 4087 | 2 | 834 | 48 | 4.9 | 2.51 |
| 8k | 8267 | 2 | 812 | 47.9 | 10.2 | 2.45 |
| 16k | 16311 | 2 | 783 | 46.2 | 20.8 | 2.71 |
| 32k | 32776 | 2 | 749 | 44.6 | 43.8 | 2.98 |
| 64k | 65660 | 2 | 673 | 43.8 | 97.6 | 3.20 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 3.53 |
| Speculative verdict | effective |
| Prefix cache speedup | 6.5× |
| Cached / prompt tokens | 1388 / 1419 |
| Batch streams | 4 |
| Single stream tok/s | 31.7 |
| Aggregate tok/s | 51.1 |
| Batch efficiency | 0.4 |
| Sustained initial / final tok/s | 53.1 / 46.8 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, reasoning default — not comparable to default runs
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work
- engine rejected the reasoning effort param; ran at its default — not comparable to runs that set the effort
