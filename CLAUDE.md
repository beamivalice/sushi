# Sushi — project context for AI

A native Zig inference engine for Apple Silicon serving Qwen3.8-Flash-Next (`qwen4_exp`), GLM-5.3-Flash (`glm5_next`)
and DeepSeek-V4.1-Flash (`deepseek_v41`). MiMo-V2.6-Flash (`mimo_v2`) is unsupported from 1.5.1: still served, never
release-gated (no test, bench or tuning for a release). Qwen and MiMo use native MTP and affine8 KV by
default. GLM batches up to four requests over an affine8 MLA latent and FP32 KDA state, with DFlash2; its MTP stays off.
DeepSeek serves one request at a time with streamed experts and DSpark. OpenAI/Anthropic-compatible HTTP, no Python at
serve time. Fork of ddalcu's mlx-serve.

- **sushi** is public code: every change ships with its license, NOTICE and docs story.
- **sashimi** is the PRIVATE quant-creation stack. This repo keeps only the CONSUMER contract (`docs/pack-format.md`,
  the shard-stamp check, `src/fixtures/`, `sushi kld`) and the oracle fixture dumpers (`tests/dump_*_fixtures.py`).
  Converter knowledge goes to `docs/private/`, never a committed file.
- `src/` holds only the four served architectures and their shared machinery. Any other `model_type` is refused by
  name (`ArchitectureUnsupported` → 503), as is an unsupported file format (`ModelFormatUnsupported` → 503).

<a id="docs-index"></a>
## Docs index

This file holds rules and the map. Knowledge, measurements and lessons live in `docs/<category>-<topic>.md`; read the
doc for the area before changing it, and update it in the same landing. A doc about ONE served architecture takes
that architecture as its prefix (`qwen4-`, `mimo2-`, `glm5-`, `dsv41-`); `engine-`, `server-` and the rest are shared.

| doc | what it holds |
|---|---|
| [docs/source-map.md](docs/source-map.md) | which `src/` file holds what, and the doc for each |
| [docs/qwen4-arch.md](docs/qwen4-arch.md) | Flash-Next trunk, hyper-connections, n-gram PLE, oracle and ties, packs |
| [docs/qwen4-perf.md](docs/qwen4-perf.md) | Flash-Next tok/s: decode/MTP ladders, GDN/QSA/HC kernel A/Bs, M1/M2 Max |
| [docs/qwen4-kld.md](docs/qwen4-kld.md) | Flash-Next KLD of every served pack, the KV-width reading |
| [docs/qwen4-qsa-long-context.md](docs/qwen4-qsa-long-context.md) | QSA arms per query width, indexer, long-context admission and bills |
| [docs/mimo2-arch.md](docs/mimo2-arch.md) | MiMo checkpoint, FP8 trunk, rank-local QKV, routing/sinks, sliding ring, bills |
| [docs/mimo2-perf.md](docs/mimo2-perf.md) | MiMo tok/s: context ladders, verify rows, prefill, attention kernels, TTFT |
| [docs/mimo2-kld.md](docs/mimo2-kld.md) | MiMo KLD of the served packs |
| [docs/mimo2-mtp.md](docs/mimo2-mtp.md) | MiMo's three MTP heads, chain and acceptance |
| [docs/glm5-arch.md](docs/glm5-arch.md) | GLM geometry, serving, DFlash2, bills, speed, KLD, lessons |
| [docs/glm5-kernels.md](docs/glm5-kernels.md) | GLM kernels, exactness rules, drift screen, ruled-out list |
| [docs/glm5-perf.md](docs/glm5-perf.md) | GLM tok/s: DFlash2 round levers, three/four-row verify, prefill chunk, long context |
| [docs/glm5-kld.md](docs/glm5-kld.md) | GLM KLD against the native BF16 teacher |
| [docs/glm5-prefix-cache.md](docs/glm5-prefix-cache.md) | GLM cache keys, latent and KDA state, restore, spec state |
| [docs/glm5-clamped-experts.md](docs/glm5-clamped-experts.md) | GLM clamped EXL3 experts |
| [docs/dsv41-arch.md](docs/dsv41-arch.md) | DeepSeek-V4.1-Flash: geometry, FP8 trunk + streamed experts, Engram, CED prefill, oracle, kernels, DSpark, STQ, `--drop-expert`, prefix cache, images |
| [docs/dsv41-memory.md](docs/dsv41-memory.md) | DeepSeek-V4.1 ledger, per-layer cache split, 1M-token state, prefill borrow, resident load |
| [docs/dsv41-parity.md](docs/dsv41-parity.md) | DeepSeek-V4.1 against Qwen: feature matrix, larger gaps, live checks |
| [docs/engine-exl3-experts.md](docs/engine-exl3-experts.md) | EXL3 rate/codebook/window, prefill GEMM, decode chain, f32 SwiGLU, parity bars |
| [docs/encoder.md](docs/encoder.md) | `sushi convert exl3`: flags, what a pack holds, trunk rules, calibration, Hessians, resume, stamps |
| [docs/engine-exl3-encoder.md](docs/engine-exl3-encoder.md) | EXL3 encoder core: trellis search, LDLQ, signs, shard writer, Qwen BF16 reader |
| [docs/mlx-serve-integration.md](docs/mlx-serve-integration.md) | how mlx-serve vendors `sushi_exl3`, the API it calls, what the host owes |
| [docs/mlx-serve-sync.md](docs/mlx-serve-sync.md) | upstream mlx-serve review ledger: cursor, verdicts, how to run the next pass |
| [docs/engine-expert-streaming.md](docs/engine-expert-streaming.md) | SSD budget ledger, per-layer LRU, slab I/O, imatrix capture, discovery |
| [docs/engine-mtp.md](docs/engine-mtp.md) | MTP head, verify invariant, draft re-scoring, round-cost table |
| [docs/engine-kv-cache.md](docs/engine-kv-cache.md) | kv8 default, kv-quant contract, growth, GDN step, byte-stability settings |
| [docs/engine-prefix-cache.md](docs/engine-prefix-cache.md) | hot cache, hybrid restore, trimming, SSD tier, SSD-first, checkouts, spec state |
| [docs/engine-kernels.md](docs/engine-kernels.md) | decode/MoE/prefill/verify kernels, NAX/MPP pitfalls, proving and timing a kernel |
| [docs/engine-mlx-gotchas.md](docs/engine-mlx-gotchas.md) | MLX errors, dtype promotion, barriers, views vs copies, allocator, Zig/tokenizer traps |
| [docs/engine-memory-admission.md](docs/engine-memory-admission.md) | Metal OOM, preflight, auto-context, prefill chunk, admission |
| [docs/server-http-apis.md](docs/server-http-apis.md) | API contracts, streaming, sampling, reasoning budget, constrained JSON, launcher |
| [docs/server-webui.md](docs/server-webui.md) | web UI design: tokens and contrast, salmon accent, motion, logo, tooltips |
| [docs/server-tool-calling.md](docs/server-tool-calling.md) | templates, tool-call parse chain and invariants, think tags, loop stops |
| [docs/server-lifecycle.md](docs/server-lifecycle.md) | arch gate, weight loader, settings precedence, scheduler/batching, threads, media, sushibar |
| [docs/envvar.md](docs/envvar.md) | every env var the engine reads, by kind: read site and cost, docs, flag twins, reads to fix, removed list |
| [docs/flags.md](docs/flags.md) | every launch flag: default, who it is for, keep/cut recommendation, cut list, owner calls |
| [docs/pack-format.md](docs/pack-format.md) | what a pack owes the engine: tensors, `expert_quant`, metadata stamp, window, loader rules |
| [docs/perf-baselines.md](docs/perf-baselines.md) | method, roofline, shared EXL3 kernel timings, release ladders |
| [docs/bench/v1.4.0/summary.md](docs/bench/v1.4.0/summary.md) | release context ladder and per-model benchmark reports |
| [docs/quality-kld.md](docs/quality-kld.md) | `kld` tool, teacher fixtures, the 16x512 reading, lossless teacher rule |
| [docs/process-ubench.md](docs/process-ubench.md) | `sushi-ubench`: microbench registry, CLI, rungs x arms tables, JSON and compare, the quiet-box steps it takes |
| [docs/process-measurement.md](docs/process-measurement.md) | GPU lock, binary stamp, QoS, cooling, waiting, baseline lookup, recording a number, the release bench, build environment |
| [docs/testsuite.md](docs/testsuite.md) | what a test may be, tiers (`test-host`, gpu, pack, bench), placement, how a test is built, what a change runs |
| [tests/CLAUDE.md](tests/CLAUDE.md) | the integration-test matrix (auto-loads in `tests/`) |

**Private, local-only** (`docs/private/`, gitignored, main checkout only: a worktree reads them from there). Never
link to or quote them from a committed file.

| doc | what it holds |
|---|---|
| `docs/private/sashimi-workflow.md` | converting with sashimi: venv, subcommands, served-pack recipes, imatrix files, stamps, lessons |
| `docs/private/sashimi-codebooks.md` | MCG decision, decoder dead ends, fractional-rate trellis |
| `docs/private/quality-kld-research.md` | KLD of every intermediate and experimental pack, sweeps, how packs were made |
| `docs/private/measurement-raw.md` | the raw-file path behind every committed measurement, keyed by doc and section |
| `docs/private/future.md` | future repo work; read at each release; an entry leaves when it starts or lands |
| `docs/private/fail/` | one file per failed experiment: the numbers that killed it, where its code lives |
| `docs/private/hf-upload.md` | publishing a pack to Hugging Face: staging, card, upload, verification |

Skills: `/release` (release order, gates, CHANGELOG, contributor credit), `/bench` (methodology, claim etiquette).

## Stack and building

Zig 0.17.1 (`scripts/fetch-zig.sh` stages it at `.zig-toolchain/`); mlx + mlx-c are pinned submodules built NAX-enabled
by `scripts/build-mlx.sh` into `lib/mlx/` (FFI `src/mlx.zig`); jinja.cpp as `lib/jinja_cpp/libjinja.a`. Min macOS 26.2.
Box: M5 Max 128 GB, macOS 27. Bumping mlx or rebuilding jinja: [process-measurement](docs/process-measurement.md#build-environment-notes).

- **ALWAYS `zig build -Doptimize=ReleaseFast`, never bare `zig build`** (Debug is 2–4× slower ⇒ fake regressions).
  `zig build test` does NOT refresh `zig-out/bin/sushi`: rebuild before any live A/B.
- A git worktree lacks `.zig-toolchain/` and `lib/mlx/`: symlink both from the main checkout. After a toolchain/SDK
  change `rm -rf .zig-cache` (configure-time output is cached).

## Testing — TDD is mandatory

Order: (1) failing test FIRST, for the right reason; (2) minimum code to green; (3) the subset that
[testsuite](docs/testsuite.md#what-a-change-runs) names; (4) refactor. A live curl is NOT a test. The full suite runs
only for a wide stacked change (shared files, several areas) or a version bump.

Feature = unit test that fails without it (+ integration script if HTTP-observable). Bug fix = regression test
red→fix→green. Refactor = characterization test first. A live failure revealing a CLASS ships the instance test plus a
corpus entry or invariant in `src/format_corpus_test.zig` plus a rule in the matching doc.

`zig build test-host` runs the MLX-free tier beside a live server. Hermetic suites: `-Dtest-filter="format corpus"`,
`-Dtest-filter="tool traffic"`. Live scripts and their matrix: `tests/CLAUDE.md`.

- **A test never writes to stdout**: under `zig build test` fd 1 is the build runner's protocol pipe; one stray line
  hangs the runner.
- **A PASSING test prints NOTHING, on either stream**: Zig renders any test stderr as a failure. Diagnostics ride an env
  switch (`SUSHI_EXL3_PARITY_TRACE`). Guard: `tests/test_test_runner_quiet.sh`.
- **No source-scan tests** (`@embedFile` + "this string appears in that function"): test the behaviour or state the
  rule in a comment.
- **An integration assertion that a MODEL must think/answer/call is a checkpoint expectation**: assert the INVARIANT,
  branch on the model's choice.
- A first `zig build test` run sometimes fails and a re-run is green (Metal contention, benign).

## Releases and packs

`/release` holds the process; the perf gate is `tests/bench.sh` on the FINAL tree (Qwen STQ 2.75bpw, Qwen 4bpw, GLM
2.4bpw, DeepSeek STQ 1.75bpw). Deferred repo work goes to `docs/private/future.md`, never agent memory.
- A pack reaches Hugging Face only as the staged, checked copy that `docs/private/hf-upload.md` describes, with an
  owner-approved card.
- **Adding a pack** (pair-major: [tile order](docs/pack-format.md#tile-order)) updates, in the same landing, its
  `aliases` row in `src/cli.zig` (`Alias.tier_gb`, `recommend_gb`), the tier table and recommended launch in
  `docs/usage.md`, and the README pack line.

## Conventions

- Minimal DRY Zig; tests at the bottom of each source file; shell integration tests in `tests/`.
- No tracked file names a box's own directories: packs are `${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/<pack>`, run
  outputs `~/.sushi/runs/` (guard: `tests/test_no_local_paths.sh`).
- **Env vars** ([envvar](docs/envvar.md)): a temporary A/B knob is fine while developing; once the answer is known it
  goes with its losing arm. Only diagnostics and live dev levers stay, read once via `diagEnvOn`. A user tradeoff or
  machine-fit setting is a flag, never an env twin.
- **Flags** ([flags](docs/flags.md)): every parsed flag is in `--help`, none hidden; a bad value or unknown argument
  exits with an error before anything runs. Adding, cutting or changing a flag is the owner's decision.
- **Diffs are read by a human: keep them small.** A comment says what the code cannot (a non-obvious WHY, a contract,
  a unit) in one to three lines; never bug history, measurements, dates, PR numbers or a restatement of the code.
- **One story per gotcha, one line per rule**, in the matching doc. CHANGELOG: one user-facing sentence per change,
  no provisional numbers.
- One squash commit per PR, rebased onto main. Its author is the human who takes responsibility for it (an outside PR:
  the contributor's GitHub identity); no AI author or Co-Authored-By trailer.

## Engine invariants (never break; the doc has the why)

- The inference thread is the SOLE mlx caller, even for frees ([engine-mlx-gotchas](docs/engine-mlx-gotchas.md)).
- The weight loader is ONE decision, `model.loadWeightsForConfig` ([server-lifecycle](docs/server-lifecycle.md)).
- An explicit launch flag outranks `model-settings.json`, which outranks the default
  ([server-lifecycle](docs/server-lifecycle.md#settings)).
- A bill follows the storage in the SAME commit; under-billing is a Metal OOM
  ([engine-memory-admission](docs/engine-memory-admission.md)).
- A stream and a non-stream answer are the SAME BYTES; emitted tool `arguments` are ALWAYS valid JSON
  ([server-http-apis](docs/server-http-apis.md), [server-tool-calling](docs/server-tool-calling.md)).
- A pack is judged by KLD against a lossless teacher, never by bytes against another pack
  ([quality-kld](docs/quality-kld.md)).
- The engine reads only aggregate machine load and its own I/O timings, states them only in a visible log line, and
  never names another program ([server-lifecycle](docs/server-lifecycle.md#machine-state)).

## Debugging

Server log `~/.sushi/logs/sushi-<port>.log` is THE post-mortem file (`--log-level debug`). Grep its tags: `jinja error:`,
`<- N+M tokens`, `tool_msgs=`, `[cache]`, `[spec-stats]`, `[mtp-planner]`, `[loop-stop]`, `[admission]`, `[kv-cache]`,
`[expert-stream]`, `[disk-cache]`, `[hot-cache]`, `[prefill-trace]`. Capture traffic: `SUSHI_RAW_DUMP_FILE=<abs>` →
`tests/harvest_tool_traffic.py`. Reproduce tool bugs `stream:false` first; stop your own server by PID, never
`pkill`. `/props` reports `active_bytes`,
`memory.cache_bytes`, `batching`; RSS is blind to Metal.

<a id="team-process"></a>
## Team process

Procedures and examples: [docs/process-measurement.md](docs/process-measurement.md).

**GPU sharing.** The lock is for jobs that need the box to themselves: a full model load (server boot, KLD, live test)
and a quiet-box bench or timing whose absolute number is recorded. Conversions, pilots, builds and tests run in
parallel without it if the total fits memory: never beside a MiMo-size (≥ 90 GB) load, and a quiet bench waits.
- Take the lock right before EACH run: `scripts/gpu-lock.sh run <owner> [--max-min N] -- <cmd>` (releases on any
  exit). Declare `--max-min` over ~15 min; model-free tests never take it; `status` shows holder, age, queue.
- Never hold it across a batch, or while analysing, editing, building or waiting. An A B B A re-acquires per arm.
  Every brief that runs on the GPU names the lock.
- A dead `run` holder frees itself. An OVERDUE or idle holder: workers report it; the coordinator verifies, stops it,
  runs `gpu-lock.sh break <holder>`.

**Baselines.** Never rerun an already-recorded old-code baseline: run only the new arm and compare with the recorded
number (within a session, inherit the previous one); run an old baseline only in a clean session or when none exists
for that setting, and say which. Existing pack shards are a converter's byte-identity baseline. Committed docs cite
commit + settings beside every new number; raw-file paths go to `docs/private/measurement-raw.md`.

**Landing perf.** A speedup is real when it is outside run-to-run noise with arms interleaved in one session.
- A real speedup that leaves output bit-identical lands whatever its size.
- A prefill, decode or batching lever worth 5% or more is built and tested, never only noted; one that changes output
  goes through the KLD gate ([quality-kld](docs/quality-kld.md)).
- At the gate, a KLD change under 0.01% passes; a larger one goes to the owner with its speed gain to decide.
- Exactness is judged on one hardware path: a NAX arm whose bytes differ from the SIMD arm's (new vs old hardware) is
  not an output change.
- An output-changing tweak worth about 1% of a token or less is dropped, not queued.
- Judge a tweak on top of the change it builds on, not alone against the old code.

**Measurement hygiene.**
- Rebuild ReleaseFast from the head under test right before any live number; stamp commit + binary mtime beside it.
- Run every CPU-heavy agent job (builds, tests, conversions, timings) under `taskpolicy -a`, and state the QoS, lock
  and baseline beside every number.
- Quiet window: a timed run first touches `~/.sushi/runs/QUIET` and waits until no `zig`/`clang` runs; no worker
  starts a build or test while that file exists. The GPU lock does not stop compiles.
- Measure only what the diff touches: no arm whose code path is unchanged, no metric the change cannot move.
- Bench thermal: [cool the box](docs/process-measurement.md#4b-cool-the-box-before-a-bench) before each server and
  between arms; never slot a GPU job into a bench's cooldown idle.
- Wait on END markers, PIDs or `pgrep -x`, never `pgrep -f <string>` (the waiting shell matches itself).
- Detach a long job with `nohup perl -e 'use POSIX; POSIX::setsid(); exec @ARGV' zsh <script> > <log> 2>&1 < /dev/null &`
  (macOS has no `setsid`; plain `nohup … &` dies with the tool's session); END marker from an EXIT trap.
- An agent never leaves `sushibar` or a server running for the owner's use: agent shells run reniced, children
  inherit it, and the GPU clocks down ([QoS](docs/process-measurement.md#4-restore-qos-for-agent-launched-jobs)).
- Confirm the load lines (`[kv-cache]`, `[mtp]`, engagement lines) show the intended arm.
- Reported KLD: the eight-category 16x512 fixture to first EOS; a prose-only fixture is a gate, never reported. The
  teacher carries no lossy step of its own.

**Coordinator.** Reports partial work to the owner every :00 and :30 while work runs; relays owner decisions and rule
changes to ALL live workers at once; checks each landing updated its matching doc, and this index when a doc appears.

**Workers.** Report progress and blockers to the coordinator; commit per step; don't push or merge unless told; put new
knowledge in the matching doc in the branch.

**Clean up.** Once a branch lands, remove its worktree and branch (local and remote). When a job ends, stop by PID every
process it started (servers, watchers, `tail -F`, samplers) and delete its temp files and scripts; packs need the owner.

**Growth policy (ENFORCED).** This file holds rules and links only and stays under 25 KB; every rule bullet is
≤ 3 lines. Knowledge, measurements and war stories go to `docs/` (converter knowledge to `docs/private/`); commit
messages carry the story of a change.

## Licensing

Ported kernels + vendored code are enumerated in `NOTICE` (the ONE place); `LICENSE`/`LICENSE-APACHE-2.0`/`NOTICE`
ride every packaging path (`tests/test_release_workflow_gates.sh`). To enumerate ports, grep comments for
`mlxfast|oMLX|MTPLX|mlx-lm|port`.
