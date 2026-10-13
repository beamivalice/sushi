# Benchmarks — sushi decode by release

**Update rules — read before editing:**
- Results go into the tables ONLY. No text, no commentary, no per-release notes — commit messages carry the stories.
- **Apple M5 Max 128 GB ONLY.** Numbers across hardware are not comparable and one mixed column poisons the history.
- A cell is `./tests/bench.sh` decode tok/s (llmprobe `--bench-only`: warmup discarded, median of measured runs (3 unless specified), its own code-completion prompt), sushi ReleaseFast at its FASTEST config, with the speculative mode that engaged named beside the number. `·` = not measured that release.
- The perf gate is one pack, Qwen3.8-Flash-Next-Sushi-4bpw. `speedup` = first measured column vs the latest.

## Decode tok/s by release

| Model | v1.0.0 | v1.0.4 | v1.0.5 | v1.1.0 | v1.2.0-dev | v1.2.0-dev2 | v1.2.0 | v1.2.1 | v1.2.2 | v1.2.3 | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next-Sushi-3bpw (MTP) | · | 98 mtp | · | · | · | · | · | · | · | · | · |
| Qwen3.8-Flash-Next-Sushi-4bpw (MTP) | · | · | 83 mtp | 89 mtp | 92.7 mtp (2 runs) | · | 85.0 mtp (2 runs) | 92.9 mtp (2 runs) | 91.9 mtp (2 runs) | 91.3 mtp (2 runs) | · |
| MiMo-V2.6-Flash-Sushi-2.3bpw (MTP) | · | · | · | · | 42.8 mtp (2 runs) | · | 47.4 mtp (2 runs) | 60.4 mtp (2 runs) | 56.8 mtp (2 runs) | 55.2 mtp (2 runs) | · |
| GLM-5.3-Flash-Sushi-2.5bpw | · | · | · | · | 44.7 dflash (3 runs) | 55.4 dflash (3 runs) | · | · | · | · | · |
| Qwen3.8-Flash-Next-Sushi-2bpw | · | · | · | · | 108.8 mtp (2 runs) | · | · | · | · | · | · |
| Qwen3.8-Flash-Next-Sushi-2.6bpw | · | · | · | · | 95.3 mtp (2 runs) | · | 92.1 mtp (2 runs) | 89.9 mtp (2 runs) | 98.6 mtp (2 runs) | 96.9 mtp (2 runs) | · |
| GLM-5.3-Flash-Sushi-2.3bpw | · | · | · | · | · | 51.1 dflash (3 runs) | · | · | · | · | · |
| GLM-5.3-Flash-Sushi-2.4bpw | · | · | · | · | · | · | 54.6 dflash (2 runs) | 53.7 dflash (2 runs) | 55.1 dflash (2 runs) | 51.1 dflash (2 runs) | · |
| GLM-5.3-Flash-Sushi-2bpw | · | · | · | · | · | · | · | · | 49.4 dflash (2 runs) | · | · |
