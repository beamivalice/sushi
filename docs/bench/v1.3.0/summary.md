# Sushi 🍣 1.3.0 benchmarks

Version: **1.3.0**. Date: 2026-10-21 (Asia/Bangkok). MiMo ran the release candidate `dc93b43f`; it predates the version bump, so it prints 1.2.4. The Qwen ladders are the MTP arms of the same day's DFlash2 measurements, on branch builds whose MTP path is the one `dc93b43f` ships. GLM was not rerun: its serving path is unchanged apart from the drafter lookup, which the release's live DFlash test covers.

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power for every run, `iogpu.wired_limit_mb=120000`. ReleaseFast, `taskpolicy -a`, one model per server, GPU lock per model, 3 minutes idle before each server. All rates are tok/s.

`npx llmprobe@0.6.15 localhost:12345 --bench-only -m <model> --rungs <rungs> --runs <n>`. Scenario warmups are discarded; the table holds llmprobe medians. Server flags: `--kv-quant 8`, default vision, `--mtp` on Qwen and MiMo (Qwen also `--no-drafter --prefix-cache-entries 0`). Qwen serves its default f32 GatedDeltaNet state.

| Sushi 🍣 1.3.0 | | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen 3.8 Flash-Next-2.6bpw** | Prefill | · | 2,283 | · | 2,169 | · | 1,843 | 1,597 |
|  | Decode | · | 92 | · | 88.3 | · | 79.4 | 71.3 |
| **Qwen 3.8 Flash-Next-4bpw** | Prefill | · | 2,562 | · | 2,386 | · | 2,106 | 1,749 |
|  | Decode | · | 95.8 | · | 94.2 | · | 80.6 | 73 |
| **MiMo V2.6 Flash-2.3bpw** | Prefill | 958 | 427 | 914 | 886 | 789 | 635 | 447 |
|  | Decode | 53.4 | 37.6 | 50.8 | 48.9 | 45.5 | 37.7 | 32.5 |

| Model run | Measured runs | Speculative mode | Commit | Binary SHA-256 |
|---|---:|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-2.6bpw](Qwen3.8-Flash-Next-Sushi-2.6bpw.md) | 3 | mtp | `f9393ba6` | `9f65c878ba7ac689` |
| [Qwen3.8-Flash-Next-Sushi-4bpw](Qwen3.8-Flash-Next-Sushi-4bpw.md) | 3 | mtp | `e2912d04` | `3ba404e29633e2e7` |
| [MiMo-V2.6-Flash-Sushi-2.3bpw](MiMo-V2.6-Flash-Sushi-2.3bpw.md) | 2 | mtp | `dc93b43f` | `607c6eb0d931eafe` |

MiMo started right after the full test suite, with the fans on automatic until 21 minutes into the run (they were already at full speed under that load), and read 18–24% under the 1.2.4 prefill. A rerun of the 2k rung after 3 minutes of fans-max idle prefilled 1,239–1,248 tok/s per request against 1.2.4's 1,285–1,353, and the 10k headline prompt 1,283–1,386 against 1,281–1,296: the gap is the machine's heat, not the build. Its 4k rung was disturbed (427 tok/s prefill). Raw reports and server logs are indexed in the private measurement ledger.
