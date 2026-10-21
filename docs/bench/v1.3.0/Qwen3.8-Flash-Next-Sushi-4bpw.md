# Qwen3.8-Flash-Next-Sushi-4bpw — 1.3.0

Version: **1.3.0**. Commit: `e2912d04`. Binary SHA-256: `3ba404e29633e2e79af375a184700b60b2b21cf7a57483c64411b51a8800a95d`. Date: 2026-10-21 (Asia/Bangkok).

Measured runs: **3**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/Qwen3.8-Flash-Next-Sushi-4bpw --port 12345 --kv-quant 8 --no-update-check --prefix-cache-entries 0 --mtp --no-drafter`.

The server ran with prefix-cache reuse off (`--prefix-cache-entries 0`).

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m Qwen3.8-Flash-Next-Sushi-4bpw --rungs 4k,16k,64k,128k --runs 3`.

## Scenario samples

| Metric | Run 1 | Run 2 | Run 3 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Decode tok/s | 105.7 | 100.8 | 100.8 | 100.8 | 100.8 | 105.7 |
| Prefill tok/s | 2666.6 | 2601.9 | 2569.1 | 2601.9 | 2569.1 | 2666.6 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 4k | 4248 | 3 | 2562 | 95.8 | 1.7 | 4.47 |
| 16k | 16452 | 3 | 2386 | 94.2 | 6.9 | 3.76 |
| 64k | 65580 | 3 | 2106 | 80.6 | 31.1 | 4.36 |
| 128k | 131109 | 3 | 1749 | 73 | 75.0 | 3.43 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 4.68 |
| Speculative verdict | effective |
| Prefix cache speedup | 1× |
| Cached / prompt tokens | 0 / 1539 |
| Batch streams | 4 |
| Single stream tok/s | 82.9 |
| Aggregate tok/s | 109.4 |
| Batch efficiency | 0.33 |
| Sustained initial / final tok/s | 100.8 / 86.1 |

## Probe notes

- custom setup: 3 runs per scenario, rungs 4k, 16k, 64k, 128k — not comparable to default runs
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work
