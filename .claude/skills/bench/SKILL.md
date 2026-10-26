---
name: bench
description: sushi benchmarking methodology — bench.sh/llmprobe usage, comparison-trap rules (same-methodology cells only, spec-decode variance, thermal lies, engine naming), perf-claim etiquette. Use before running benchmarks or making any performance claim.
---

## Benchmarking

**`sushi bench` is the measurement layer.** `tests/bench.sh` boots one model, drives it with `sushi bench` and stops it. The protocol (a code corpus and a prose corpus, each one cold prefill in 8k restore windows, three decode samples per rung 4k to 128k, restore time taken out of the prefill rate, the order rule) is in `docs/process-measurement.md#release-bench`; read it before changing anything. The wrapper takes the box rules itself, so a perf-gate run is one command:

```
./tests/bench.sh --version 1.4.0 DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw -- --ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000
```

**Each cell is the shipping default**: the server's own speculative mode (DSpark, MTP, DFlash2) stays on, and the mode that actually engaged is read off each request's `[spec-stats] mode=` line and named in the report. Read the prose rows beside the code rows: code decode flatters speculation (prose accepts fewer tokens a round), and only the code rows compare with earlier releases. A rung with `none` or `mixed` there is the regression signal; a request that did not restore from the previous prompt end is listed under Warnings and the run fails.

**Another engine = llmprobe by hand.** Start LM Studio / oMLX / MTPLX / llama-server yourself and run `npx -y llmprobe@latest <host:port> -m <id> --bench-only`. llmprobe's cold-prefill-per-rung numbers are a different methodology from the release bench: never diff the two.

The artifacts: `docs/bench/v<ver>/<pack>.md` (the report, with a summary row for `benchmarks.md`) and the raw `bench.json` + server log under `~/.sushi/runs/bench-<tag>/`.

### Before you run anything

Follow CLAUDE.md's Team process and `docs/process-measurement.md`: take `scripts/gpu-lock.sh` per run, rebuild and stamp the binary, restore QoS (`taskpolicy -a`), and look up the recorded baseline in `docs/perf-baselines.md` first. A recorded old-binary baseline is inherited, not rerun (rerun only in a clean new session); cite its commit and settings beside the new number, and record the new number there in the same landing.

### Comparison traps (these cost real days)

- **Only diff same-methodology cells.** Columns through 26.7.12 are the old in-repo harness; 26.8 to 1.3.0 are llmprobe (cold prefill per rung); from 1.4.0 `sushi bench` (restore windows, 3 decode samples) — different prompts, warmup and rate math. Never diff across those boundaries. Same rule inside one column: a forced-spec cell and a shipping-defaults cell are not the same measurement.
- **"Reproducible ≠ not variance"** for spec-decode cells — sample across runs and boot orders before any regression claim. A cell that reads the same twice can still be variance.
- **Attribute before believing.** Check whether the change could physically reach the cell that moved; reachability is faster to check than another bench and is what makes a repeat a confirmation.
- **Never quote a win without naming the engine it is over** — vs LM-GGUF a row reads +33%; vs oMLX the same row is +1.6%.
- **Thermal soak lies harder than drift** — same-session ratios only. llmprobe's own sustained-load check catches drift WITHIN a cell; run the comparison engine right after ours, not hours later, or say so beside the number.
- **An A/B arm is proven by ENGAGEMENT lines in its own log, never by its launch env.** zsh does not word-split `env $VAR`, so a multi-switch arm's first switch swallowed the rest as its value and the "composed" arm silently ran the fast path — reading a 2x win as "neutral" for half a session (live 2026-07-30, story in docs/qwentts-cache.md).
- **A spec-decode collapse that survives SERVER restarts is machine state, not the build** (M3 Ultra, PR #223: 5.45→2.0 tok/step persisted across restarts; a macOS reboot fully restored it) — reboot and re-baseline before attributing.
- **Power source is a variable on laptops**: measure on AC at high charge — battery→AC roughly doubled one tester's decode MID-RUN (PR #223, M5 Max); Latin-square or A-B-A the arms on laptops so a power/thermal shift reads as drift, not an effect.
- **A bench's port wait-list must equal its kill-list.** LM Studio's server is a persistent daemon you never kill (its MODEL is freed by `lms unload --all`), so waiting on its port burns the full timeout on every stop — measured 11 of 20 min on one run. This is why bench.sh manages no other engine.
