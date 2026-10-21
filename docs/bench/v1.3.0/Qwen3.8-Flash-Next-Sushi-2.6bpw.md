# Qwen3.8-Flash-Next-Sushi-2.6bpw — 1.3.0

Version: **1.3.0**. Commit: `f9393ba6`. Binary SHA-256: `9f65c878ba7ac689694ca116ae28c4649ec537efe4cc619dad926518899c0741`. Date: 2026-10-21 (Asia/Bangkok).

Measured runs: **3**, one server boot. Speculative mode: **mtp**, verified from engagement logs. Hardware, thermal protocol and common settings: [version summary](summary.md).

Server command: `sushi --serve --model <model-dir>/Qwen3.8-Flash-Next-Sushi-2.6bpw --port 12345 --kv-quant 8 --no-update-check --prefix-cache-entries 0 --mtp --no-drafter`.

The server ran with prefix-cache reuse off (`--prefix-cache-entries 0`).

Probe command: `npx llmprobe@0.6.15 localhost:12345 --bench-only -m Qwen3.8-Flash-Next-Sushi-2.6bpw --rungs 4k,16k,64k,128k --runs 3`.

## Scenario samples

| Metric | Run 1 | Run 2 | Run 3 | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Decode tok/s | 96.2 | 101 | 102.8 | 101 | 96.2 | 102.8 |
| Prefill tok/s | 2359.2 | 2257.3 | 2264.2 | 2264.2 | 2257.3 | 2359.2 |

## Context ladder

| Target context | Actual input tokens | Runs | Prefill tok/s | Decode tok/s | First token s | Tok/step |
|---|---:|---:|---:|---:|---:|---:|
| 4k | 4253 | 3 | 2283 | 92 | 1.9 | 3.84 |
| 16k | 16455 | 3 | 2169 | 88.3 | 7.6 | 3.69 |
| 64k | 65571 | 3 | 1843 | 79.4 | 35.6 | 3.25 |
| 128k | 131112 | 3 | 1597 | 71.3 | 82.1 | 3.31 |

## Additional benchmark results

| Metric | Value |
|---|---|
| Tokens per decode step | 5.82 |
| Speculative verdict | effective |
| Prefix cache speedup | 1× |
| Cached / prompt tokens | 0 / 1538 |
| Batch streams | 4 |
| Single stream tok/s | 79.1 |
| Aggregate tok/s | 110.5 |
| Batch efficiency | 0.35 |
| Sustained initial / final tok/s | 101 / 84.9 |

## Probe notes

- custom setup: 3 runs per scenario, rungs 4k, 16k, 64k, 128k — not comparable to default runs
- engine ignored ignore_eos/min_tokens — decode length follows the model's own stop, so decode figures cover model-dependent amounts of work
