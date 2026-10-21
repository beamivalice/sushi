# MiMo-V2.6-Flash-Sushi-2.3bpw — 1.3.0

Version: **1.3.0**. Commit: `dc93b43f`. Binary SHA-256: `607c6eb0d931eafe9e495b35cf276055eaa34aa96a8e1cec1ce969b24510981d`. Date: 2026-10-21 (Asia/Bangkok).

Measured runs: **2**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/MiMo-V2.6-Flash-Sushi-2.3bpw --port 12345 --kv-quant 8 --no-update-check --mtp`.

The server ran with its own empty SSD prefix cache; see the [version summary](summary.md).

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m MiMo-V2.6-Flash-Sushi-2.3bpw --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`.

## Scenario samples

| Metric | Run 1 | Run 2 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 43.8 | 80.4 | 62.1 | 43.8 | 80.4 |
| Prefill tok/s | 1247.4 | 1098.8 | 1173.1 | 1098.8 | 1247.4 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 2k | 2095 | 2 | 958 | 53.4 | 2.2 | 2.95 |
| 4k | 4088 | 2 | 427 | 37.6 | 10.0 | 3.53 |
| 8k | 8268 | 2 | 914 | 50.8 | 9.0 | 3.15 |
| 16k | 16337 | 2 | 886 | 48.9 | 18.5 | 2.67 |
| 32k | 32721 | 2 | 789 | 45.5 | 41.5 | 3.44 |
| 64k | 65631 | 2 | 635 | 37.7 | 103.6 | 3.29 |
| 128k | 131057 | 2 | 447 | 32.5 | 293.4 | 3.56 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 5.65 |
| Speculative verdict | effective |
| Prefix cache speedup | 12.6× |
| Cached / prompt tokens | 1534 / 1535 |
| Batch streams | 4 |
| Single stream tok/s | 35.9 |
| Aggregate tok/s | 62.8 |
| Batch efficiency | 0.44 |
| Sustained initial / final tok/s | 62.1 / 51.1 |

## Probe notes

- custom setup: 2 runs per scenario, rungs 2k, 4k, 8k, 16k, 32k, 64k, 128k — not comparable to default runs
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work
