# Sushi 🍣 1.2.4 benchmarks

Version: **1.2.4**. Date: 2026-10-15 (Asia/Bangkok). Every model ran a binary built from `a65539e1`; it predates the version bump, so it prints 1.2.3.

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power for every run, `iogpu.wired_limit_mb=120000`. ReleaseFast, `taskpolicy -a`, one model per server, GPU lock per model, fans at maximum and 3 minutes idle before each server. All rates are tok/s.

`npx llmprobe@0.6.15 localhost:12345 --bench-only -m <model> --rungs 2k,4k,8k,16k,32k,64k --runs 2`. Scenario warmups are discarded; the table holds llmprobe medians. Server flags: `--kv-quant 8`, default context and vision, `--mtp` on Qwen and MiMo, DFlash2 auto-detected on GLM. Qwen serves its default f32 GatedDeltaNet state.

| Sushi 🍣 1.2.4 | | 2k | 4k | 8k | 16k | 32k | 64k |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen 3.8 Flash-Next-2.6bpw** | Prefill | 1,886 | 2,118 | 2,164 | 2,141 | 2,052 | 1,961 |
|  | Decode | 95.2 | 96 | 85.9 | 91.8 | 89.3 | 77.3 |
| **Qwen 3.8 Flash-Next-4bpw** | Prefill | 2,095 | 2,339 | 2,378 | 2,293 | 2,203 | 2,165 |
|  | Decode | 79.2 | 82.2 | 82.6 | 86 | 86.6 | 81.6 |
| **MiMo V2.6 Flash-2.3bpw** | Prefill | 1,259 | 1,218 | 1,145 | 1,094 | 966 | 784 |
|  | Decode | 44.5 | 63.6 | 76.7 | 64.8 | 54.4 | 47.8 |
| **GLM 5.3 Flash-2.4bpw** | Prefill | 879 | 834 | 812 | 783 | 749 | 673 |
|  | Decode | 47.6 | 48 | 47.9 | 46.2 | 44.6 | 43.8 |

| Model run | Measured runs | Speculative mode | Commit | Binary SHA-256 |
|---|---:|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-2.6bpw](Qwen3.8-Flash-Next-Sushi-2.6bpw.md) | 2 | mtp | `a65539e1` | `85c6de9cac778da7` |
| [Qwen3.8-Flash-Next-Sushi-4bpw](Qwen3.8-Flash-Next-Sushi-4bpw.md) | 2 | mtp | `a65539e1` | `85c6de9cac778da7` |
| [MiMo-V2.6-Flash-Sushi-2.3bpw](MiMo-V2.6-Flash-Sushi-2.3bpw.md) | 2 | mtp | `a65539e1` | `85c6de9cac778da7` |
| [GLM-5.3-Flash-Sushi-2.4bpw](GLM-5.3-Flash-Sushi-2.4bpw.md) | 2 | dflash | `a65539e1` | `85c6de9cac778da7` |

MiMo is the run on an empty SSD prefix cache. Its first two runs shared a 29 GB cache over its 20 GB budget with the other models' benches, so each boot evicted 12 GB of their entries and MiMo prefill read 8–17% under 1.2.3. With an empty cache per server, the released 1.2.3 binary and this one prefilled within 1% of each other at 2k and 8k in the same session. The Qwen and GLM runs used the shared cache and read at or above 1.2.3. From 1.2.4 on, each perf-gate server starts on an empty prefix cache.

Qwen and GLM prefill read 5–16% above the 1.2.3 report at short prompts with no prefill change in between, so part of every gain here is the machine on the day. 1.2.4's own decode changes are bit-identical kernel speedups measured within one process. The 128k rung was not run this release. Raw reports and server logs are indexed in the private measurement ledger.
