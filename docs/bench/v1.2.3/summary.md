# Sushi 🍣 1.2.3 benchmarks

Version: **1.2.3**. Date: 2026-10-13 (Asia/Bangkok). Every model ran a binary built from `4a1f1f1b` with the 1.2.3 version stamp.

Hardware: **Apple M5 Max, 128 GB unified memory**, AC power for every run, `iogpu.wired_limit_mb=120000`. ReleaseFast, `taskpolicy -a`, one model per server, GPU lock per model, fans at maximum and 3 minutes idle before each server. All rates are tok/s.

`npx llmprobe@0.6.15 localhost:12345 --bench-only -m <model> --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`. Scenario warmups are discarded; the table holds llmprobe medians. Server flags: `--kv-quant 8`, default context and vision, `--mtp` on Qwen and MiMo, DFlash2 auto-detected on GLM. Qwen serves its default f32 GatedDeltaNet state.

| Sushi 🍣 1.2.3 | | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen 3.8 Flash-Next-2.6bpw** | Prefill | 1,648 | 1,839 | 1,912 | 1,972 | 1,984 | 1,953 | 1,901 |
|  | Decode | 84.3 | 88.3 | 83.2 | 75.2 | 83.9 | 79.6 | 74.6 |
| **Qwen 3.8 Flash-Next-4bpw** | Prefill | 1,810 | 2,112 | 2,153 | 2,143 | 2,201 | 2,166 | 2,112 |
|  | Decode | 83.5 | 72.8 | 77.3 | 82.9 | 82.8 | 76.2 | 77.4 |
| **MiMo V2.6 Flash-2.3bpw** | Prefill | 1,051 | 1,064 | 1,084 | 1,051 | 973 | 842 | 668 |
|  | Decode | 43.1 | 58.6 | 60.9 | 57.6 | 53.0 | 48.1 | 46.7 |
| **GLM 5.3 Flash-2.4bpw** | Prefill | 841 | 754 | 747 | 747 | 748 | 692 | 615 |
|  | Decode | 44.7 | 42.4 | 45.0 | 46.0 | 41.9 | 42.7 | 39.9 |

| Model run | Measured runs | Speculative mode | Commit | Binary SHA-256 |
|---|---:|---|---|---|
| [Qwen3.8-Flash-Next-Sushi-2.6bpw](Qwen3.8-Flash-Next-Sushi-2.6bpw.md) | 2 | mtp | `4a1f1f1b` | `b5cd630a85ecb7c9` |
| [Qwen3.8-Flash-Next-Sushi-4bpw](Qwen3.8-Flash-Next-Sushi-4bpw.md) | 2 | mtp | `4a1f1f1b` | `b5cd630a85ecb7c9` |
| [MiMo-V2.6-Flash-Sushi-2.3bpw](MiMo-V2.6-Flash-Sushi-2.3bpw.md) | 2 | mtp | `4a1f1f1b` | `b5cd630a85ecb7c9` |
| [GLM-5.3-Flash-Sushi-2.4bpw](GLM-5.3-Flash-Sushi-2.4bpw.md) | 2 | dflash | `4a1f1f1b` | `b5cd630a85ecb7c9` |

Qwen 2.6bpw, MiMo and GLM measured 5–20% under their 1.2.2 report on this day, most at short prompts; Qwen 4bpw matched it. The released 1.2.2 binary, run the same day with the same probe on 2k–32k, measured the same: 2.6bpw prefill 1,559–1,940 against 1.2.3's 1,574–1,945 (that pair at the default wired limit), MiMo prefill 955–1,074 against 973–1,084, and GLM prefill 733–789 against 747–841, with decode within each model's run-to-run spread. The difference is the machine on that day, not 1.2.3.

The release tree adds the prefix-cache checkpoint changes on top of `4a1f1f1b`. The same probe on the release commit
`5cd9f946` (binary SHA-256 `c061c229cc861c6b`), 2k–32k, with an empty prefix cache per server and fans on automatic,
measured within noise of the table above: prefill was +1.6% to +4.9% on Qwen 2.6bpw (–0.8% at 32k), +4.5% to +13% on
Qwen 4bpw, –2.9% to +1.1% on MiMo and –0.7% to –2.1% on GLM. These throughput measurements do not establish long-context answer quality. Raw reports and server logs are indexed in the private measurement ledger.
