# Sushi — project context for AI

A native Zig inference engine for Apple Silicon serving Qwen3.8-Flash-Next (`qwen4_exp`),
MiMo-V2.6-Flash (`mimo_v2`) and GLM-5.3-Flash (`glm5_next`). Qwen and MiMo use
native MTP and affine8 KV by default. GLM batches up to four requests over an affine8 MLA latent
(`--kv-quant 16` = BF16) and FP32 KDA state, with DFlash2, image/video input and prefix reuse; its MTP
stays off. OpenAI/Anthropic-compatible HTTP, no Python at serve time. Fork of ddalcu's mlx-serve.

- **sushi** is this engine's new name (runtime, to be opened to the public): a scripted rename changes the binary
  name, the env prefix and the home dir everywhere; until it lands, write today's names. Treat every change as future
  public code (license, NOTICE, docs); sushi is not scoped to EXL3 or to these two models forever.
- **sashimi** is the PRIVATE quant-creation stack (its own private repo): every converter, allocator, imatrix driver
  and repacker. This repo keeps only the CONSUMER contract — `docs/pack-format.md`, the Zig shard-stamp check, the
  committed `src/fixtures/`, and `sushi kld`. The oracle fixture dumpers (`tests/dump_*_fixtures.py`) stay: they
  verify the engine, they do not make packs. Converter knowledge (recipes, calibration data, research notes) never
  goes into a committed file: it goes to `docs/private/`, and committed files say only that it lives in the private
  repo.
- `src/` holds only the three served architectures (and their shared machinery): inherited forwards, loaders and the
  ANE driver for other architectures were removed. The loader refuses any other `model_type` by name (`model.served_model_types`, `ArchitectureUnsupported` → 503); an unsupported file
  format is refused by name (`ModelFormatUnsupported` → 503; `--model` exits).

<a id="docs-index"></a>
## Docs index

This file holds rules and the map. Knowledge, measurements and lessons live in `docs/<category>-<topic>.md`; read the
doc for the area before changing it, and update it in the same landing. A doc about ONE served architecture takes
that architecture as its prefix (`qwen4-`, `mimo2-`, `glm5-`); `engine-`, `server-` and the rest are shared.

| doc | what it holds |
|---|---|
| [docs/qwen4-arch.md](docs/qwen4-arch.md) | Flash-Next trunk, hyper-connections, n-gram PLE table, oracle and ties, packs on this box |
| [docs/qwen4-perf.md](docs/qwen4-perf.md) | Flash-Next recorded tok/s: decode and MTP ladders, GDN/QSA/HC kernel A/Bs, M1/M2 Max boxes |
| [docs/qwen4-kld.md](docs/qwen4-kld.md) | Flash-Next KLD of every served pack, the KV-width reading |
| [docs/qwen4-qsa-long-context.md](docs/qwen4-qsa-long-context.md) | QSA arms per query width, indexer and its history, long-context admission and bills |
| [docs/mimo2-arch.md](docs/mimo2-arch.md) | MiMo checkpoint, FP8 trunk, rank-local QKV, routing/sinks, sliding ring, bills, product policy |
| [docs/mimo2-perf.md](docs/mimo2-perf.md) | MiMo recorded tok/s: context ladders, verify rows, prefill, attention kernels, TTFT |
| [docs/mimo2-kld.md](docs/mimo2-kld.md) | MiMo KLD of the served packs |
| [docs/mimo2-mtp.md](docs/mimo2-mtp.md) | MiMo's three MTP heads, chain and acceptance |
| [docs/glm5-arch.md](docs/glm5-arch.md) | GLM geometry, serving, DFlash2, bills, speed, KLD, lessons |
| [docs/glm5-kernels.md](docs/glm5-kernels.md) | GLM kernels, exactness rules, drift screen, ruled-out list |
| [docs/glm5-perf.md](docs/glm5-perf.md) | GLM recorded tok/s: DFlash2 round levers, three/four-row verify, prefill chunk, long-context profile |
| [docs/glm5-kld.md](docs/glm5-kld.md) | GLM KLD against the native BF16 teacher |
| [docs/glm5-prefix-cache.md](docs/glm5-prefix-cache.md) | GLM cache keys, latent and KDA state, restore, spec state |
| [docs/glm5-clamped-experts.md](docs/glm5-clamped-experts.md) | GLM clamped EXL3 experts |
| [docs/engine-exl3-experts.md](docs/engine-exl3-experts.md) | EXL3 rate/codebook/window, prefill GEMM, decode chain, f32 SwiGLU, parity bars |
| [docs/mlx-serve-integration.md](docs/mlx-serve-integration.md) | mlx-serve's pin of `sushi_exl3`, the API it calls, handoff |
| [docs/mlx-serve-sync.md](docs/mlx-serve-sync.md) | ledger of upstream mlx-serve reviews: cursor, speed and bug-fix verdicts, how to run the next pass |
| [docs/engine-expert-streaming.md](docs/engine-expert-streaming.md) | SSD budget ledger, per-layer LRU, slab I/O, imatrix capture, discovery |
| [docs/engine-mtp.md](docs/engine-mtp.md) | native MTP head, verify invariant, draft re-scoring, round-cost table, head KV/norms |
| [docs/engine-kv-cache.md](docs/engine-kv-cache.md) | kv8 default, kv-quant contract, growth, GDN step, byte-stability settings |
| [docs/engine-prefix-cache.md](docs/engine-prefix-cache.md) | hot cache, hybrid restore, trimming, SSD tier, SSD-first, checkouts, spec state |
| [docs/engine-kernels.md](docs/engine-kernels.md) | decode/MoE/prefill/verify kernels, NAX/MPP pitfalls, how to prove and time a kernel |
| [docs/engine-mlx-gotchas.md](docs/engine-mlx-gotchas.md) | MLX errors, dtype promotion, barriers, views vs copies, allocator pool, Zig and tokenizer traps |
| [docs/engine-memory-admission.md](docs/engine-memory-admission.md) | Metal OOM, preflight, auto-context, prefill chunk, admission |
| [docs/server-http-apis.md](docs/server-http-apis.md) | API contracts, streaming, logprobs/seeds/sampling, reasoning budget, constrained JSON, launcher |
| [docs/server-tool-calling.md](docs/server-tool-calling.md) | templates, tool-call parse chain and invariants, think tags, loop stops |
| [docs/server-lifecycle.md](docs/server-lifecycle.md) | arch gate, weight loader, settings precedence, scheduler/batching, threads, ownership, media |
| [docs/pack-format.md](docs/pack-format.md) | what a pack owes the engine: tensors, `expert_quant`, `__metadata__` stamp, window, g-scale in `suh`, loader rules |
| [docs/perf-baselines.md](docs/perf-baselines.md) | method, roofline, shared EXL3 kernel timings, release ladders; per-arch numbers are in the three `*-perf` docs |
| [docs/bench/v1.2.1/summary.md](docs/bench/v1.2.1/summary.md) | version-pinned release context ladder and individual model benchmark reports |
| [docs/quality-kld.md](docs/quality-kld.md) | `kld` tool, teacher fixtures, the 16x512 reading, lossless teacher rule, lossy expert pick; per-arch results in the `*-kld` docs |
| [docs/process-measurement.md](docs/process-measurement.md) | GPU lock, binary stamp, QoS, waiting, baseline lookup, recording a number |
| [tests/CLAUDE.md](tests/CLAUDE.md) | the integration-test matrix (auto-loads in `tests/`) |

**Private, local-only** (`docs/private/`, gitignored): they exist only in the main checkout, so a git worktree does
not contain them; a worker in a worktree reads them from the main checkout's `docs/private/`. Never link to or
quote them from a committed file.

| doc | what it holds |
|---|---|
| `docs/private/sashimi-workflow.md` | how to convert with sashimi: venv, subcommands, served-pack recipes, imatrix files and hashes, stamps, window speeds, wall times, speed work, lessons |
| `docs/private/sashimi-codebooks.md` | MCG decision, decoder dead ends, fractional-rate trellis |
| `docs/private/quality-kld-research.md` | KLD of every intermediate and experimental pack, sweeps, how packs were made |
| `docs/private/measurement-raw.md` | the raw-file path behind every committed measurement, keyed by doc and section |
| `docs/private/future.md` | strictly work still to continue or explore; an entry leaves when it starts or lands |
| `docs/private/fail/` | one file per failed experiment: the numbers that killed it, where its code lives |
| `docs/private/hf-upload.md` | publishing a pack to Hugging Face: staging script, card, upload, verification, past hiccups |

Skills: `/release` (SemVer, CHANGELOG), `/bench` (llmprobe methodology, comparison traps).

## Stack

Zig 0.17.0 (pinned release via `scripts/fetch-zig.sh`; 0.16 does not build); mlx + mlx-c PINNED SUBMODULES
(`lib/mlx-src` 64ea011c / v0.32.3, `lib/mlxc-src` 56b2d39: mlx-serve's pins) self-built NAX-enabled by `scripts/build-mlx.sh` into `lib/mlx/`
(FFI `src/mlx.zig`); jinja.cpp (wangzhaode, Apache-2.0) as `lib/jinja_cpp/libjinja.a`; safetensors; BPE; `stb_image`
+ libwebp decode image INPUT. Min macOS 26.2; NAX kernels need the 26.2 deployment target (asserted by
`tests/test_mlx_staged_nax.sh`). The served binary's only non-system dylibs are `libmlxc` and Homebrew's `libwebp`
(`tests/test_serving_deps.sh`). Box: M5 Max 128 GB, macOS 27.

## Navigating `src/` (served path only)

| File | Role | Doc |
|---|---|---|
| `main.zig` / `cli.zig` | entry, flags, subcommands (`run/pull/list/serve/launch/kld`); pull, `run` REPL | server-lifecycle |
| `repl_tools.zig` | `run` REPL's client-side tools (web search/fetch, confined files, images; write needs `/edit`) | server-http-apis |
| `server.zig` / `responses.zig` / `ws.zig` / `stop_sequences.zig` | all HTTP: `/v1/*`, `/metrics(.json)`, WS, `--api-key`; Responses store | server-http-apis |
| `chat.zig` | chat templates (Jinja2 + fallback), thinking tags, tool-call parse/repair/coercion | server-tool-calling |
| `reasoning_protocol.zig` / `json_schema.zig` / `json_grammar.zig` / `token_mask.zig` / `regex.zig` | constrained decoding | server-http-apis |
| `launch.zig` | `sushi launch <agent>` configs | server-http-apis |
| `scheduler.zig` / `generate.zig` | slots, inference thread, batching, admission; generation, sampling, MTP orchestration | server-lifecycle |
| `model.zig` / `model_settings.zig` / `model_discovery.zig` / `model_registry.zig` | config + weights, per-model settings, discovery, registry | server-lifecycle |
| `transformer.zig` | shared `Transformer`, `KVCache`, `ForwardCtx`, arch dispatch, quant resolution, custom kernels | arch-*, engine-* |
| `qwen4_forward.zig` / `qwen4_hc.zig` / `qwen4_qsa.zig` | Flash-Next forward, attention, GDN, MTP, verify rows / hyper-connections + PLE / QSA | qwen4-arch, engine-mtp |
| `mimo_forward.zig` | MiMo forward, sliding/global attention arms, MoE dispatch, batched decode | mimo2-arch |
| `qwen4_exp.zig` / `hc_prefill.zig` | Flash-Next n-gram host side; fused HC prefill | qwen4-arch |
| `gdn_decode.zig` | fused GDN decode/verify step (prework + recurrence, one dispatch) | engine-kernels |
| `mimo_source.zig` / `fp8_block.zig` | MiMo source headers, FP8 trunk kept as stored + its GEMV, rank-local QKV, stored-affine trunk, shard-stamp check | mimo2-arch |
| `expert_quant.zig` / `exl3/` (`sushi_exl3` module) | expert layout, EXL3 decoders and kernels | engine-exl3-experts |
| `expert_stream.zig` / `expert_io.zig` / `expert_bf16_kernels.zig` / `imatrix.zig` | SSD expert streaming | engine-expert-streaming |
| `mtp*.zig` / `round_cost.zig` | MTP head, acceptance, planner, round-cost table | engine-mtp |
| `kv_quant.zig` | quantized KV contract (`--kv-quant 4|8`) | engine-kv-cache |
| `prefix_cache.zig` / `kv_disk_cache.zig` / `kv_disk_writer.zig` / `restore_dump.zig` | prefix cache, SSD tier | engine-prefix-cache |
| `tokenizer.zig` / `tokenize_cache.zig` | BPE, special tokens, per-model `digit_group`; prompt LRU | engine-mlx-gotchas |
| `vision.zig` / `vision_common.zig` / `qwen_vision.zig` / `mimo_vision.zig` / `glm5_vision.zig` / `mrope.zig` | media INPUT (shared preprocessing, Qwen3-VL tower + M-RoPE, MiMo-ViT, GLM tower) | server-lifecycle, mimo2-arch |
| `kld.zig` | `sushi kld capture|compare` | quality-kld |
| `metrics.zig` / `status.zig` / `log.zig` | metrics, status bar, logging | server-http-apis |
| `format_corpus_test.zig` / `tool_traffic_replay_test.zig` | hermetic format corpus, real-traffic replay | server-tool-calling |

Flags: `sushi --help` lists them all; the ones the docs lean on are named in each doc.

## Building

- First-time: `./scripts/fetch-zig.sh` stages the pinned Zig (`0.17.0`, sha256-checked) at `.zig-toolchain/`. A git
  worktree lacks `.zig-toolchain/` and `lib/mlx/`: symlink both from the main checkout (fetch-zig replaces a stale
  link with its own copy). After a toolchain/SDK change `rm -rf .zig-cache` (configure-time output is cached).
- **ALWAYS `zig build -Doptimize=ReleaseFast`, never bare `zig build`** (Debug is 2–4× slower ⇒ fake regressions).
  `zig build test` does NOT refresh `zig-out/bin/sushi` — rebuild before any live A/B.
- mlx + mlx-c: `scripts/build-mlx.sh`. Bump = checkout tag → `git add` the submodule → rerun → re-diff `src/mlx.zig` externs against
  `lib/mlxc-src/mlx/c/*.h`.
- Jinja after `lib/jinja_cpp/*.cpp` changes: compile the 7 `.cpp` (`clang++ -std=c++17 -O2 -DNDEBUG -I .`) into
  `obj/`, `ar rcs libjinja.a obj/*.o`.

## Testing — TDD is mandatory

Order: (1) failing test FIRST, for the right reason; (2) minimum code to green; (3) full suite (`zig build test
-Doptimize=ReleaseFast` 0 fail + relevant `tests/*.sh`); (4) refactor. A live curl is a sanity check, NOT a test.

Feature = unit test that fails without it (+ integration script if HTTP-observable). Bug fix = regression test
red→fix→green. Refactor = characterization test first. A live failure revealing a CLASS ships the instance test plus a
corpus entry or invariant in `src/format_corpus_test.zig` plus a rule in the matching doc.

Hermetic suites: `zig build test -Dtest-filter="format corpus"`, `-Dtest-filter="tool traffic"`. Live:
`tests/test_qwen4_exp.sh`, `tests/test_bf16_streaming.sh`, `tests/test_mtp_equivalence.sh`,
`tests/test_prefix_cache_*.sh`, `tests/test_smoke_matrix.sh`. Matrix: `tests/CLAUDE.md`.

- **A test never writes to stdout** (stderr only): under `zig build test` fd 1 is the build runner's protocol pipe;
  one stray line hangs the runner while the standalone binary passes.
- **A PASSING test prints NOTHING, on either stream**: Zig 0.17 renders any test stderr through its failure
  renderer (`failed command: … --listen=-`, exit 0), which reads as a failed suite. Diagnostics ride an env switch
  (`SUSHI_EXL3_LAYER_UBENCH`). Guard: `tests/test_test_runner_quiet.sh`.
- **No source-scan tests** (`@embedFile` + "this string appears in that function"): they pin text, not behaviour.
  Test the behaviour or state the rule in a comment.
- **An integration assertion that a MODEL must think/answer/call is a checkpoint expectation**: assert the INVARIANT,
  branch on the model's choice.
- `zig build test` sometimes reports `failed command` on a FIRST run and is green on a direct re-run (Metal
  contention between parallel test binaries, benign, unpinned).

## Releases & benchmarking

`/release` for process, SemVer, CHANGELOG. Perf gate = the `docs/bench/v<ver>/` run on the FINAL tree (Qwen 2.6bpw,
Qwen 4bpw, MiMo 2.3bpw, GLM 2.4bpw) vs the previous report; ONE new `benchmarks.md` column per release. `/bench` for
methodology: same-methodology cells only, spec cells are variance (sample across boots), an A/B arm is proven by
ENGAGEMENT lines in its log. Interleave A/B kernels in ONE process (separate runs drift 15%); same-boot medians per
cell; sub-2% calls need an IDLE box.

Release notes thank every outside contributor by @handle for each merged PR the release ships (`gh pr list --state
merged` since the last tag); a credit missed in a shipped release goes into the next one, marked as belated.

A pack reaches Hugging Face only as the staged, checked copy that `docs/private/hf-upload.md` describes, with an
owner-approved card; the README pack line follows once the repo is live.

## Conventions

- Minimal DRY Zig; tests at the bottom of each source file; shell integration tests in `tests/`.
- No tracked file names a box's own directories: packs are `${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/<pack>`, run
  outputs `~/.sushi/runs/` (guard: `tests/test_no_local_paths.sh`).
- Env levers only for paths with two arms worth comparing (lossy/tradeoff), never for an obvious win/fix. A
  diagnostic env is read through `diagEnvOn` (absent or `0` = off), never `getenv != null`.
- **Diffs are read by a human. Keep them small.** A comment says what the code cannot (a non-obvious WHY, a contract,
  a unit) in one to three lines. Never bug history, measurements, review items, dates, PR numbers or a restatement of
  the code.
- **One story per gotcha, one line per rule**, in the matching doc. CHANGELOG: one user-facing sentence per change,
  no provisional numbers.
- Squash commits, one per PR. No Co-Authored-By attribution.

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

## Debugging

Server log `~/.sushi/logs/sushi-<port>.log` is THE post-mortem file (`--log-level debug`). Grep:
`jinja error:`, `[cache]`, `<- N+M tokens`, `tool_msgs=`, `[spec-stats]`, `[mtp-planner]`, `[mtp-trace]`,
`[loop-stop]`, `[admission]`, `[kv-cache]`, `[expert-stream]`, `[disk-cache]`, `[hot-cache]`, `[dtype-trace]`,
`[short-gen]`. Capture traffic: `SUSHI_RAW_DUMP_FILE=<abs>` → `tests/harvest_tool_traffic.py`. Reproduce tool bugs
`stream:false` first; between KV-poison tests stop your own server by PID, never `pkill`. `/props` reports
`active_bytes`, `memory.cache_bytes`, `batching`; RSS is blind to Metal.

<a id="team-process"></a>
## Team process

Procedures and examples: [docs/process-measurement.md](docs/process-measurement.md).

**GPU sharing.** The lock is for jobs that need the box to themselves: a full model load (server boot, KLD, live test)
and a quiet-box bench or timing whose absolute number is recorded. Conversions, pilots, builds and tests run in
parallel without it, as long as the total fits memory: never beside a MiMo-size (≥ 90 GB) load, and a quiet bench waits.
- Acquire the lock immediately before EACH run and release right after: `scripts/gpu-lock.sh acquire|release <owner>`;
  waiters are served FIFO by ticket, `status` shows holder + queue (`${GPU_LOCK_DIR:-/tmp/sushi-gpu.lock.d}`).
- Never hold it across a batch or queue, or while analysing, editing, building or waiting. An A B B A re-acquires per
  arm. Every brief that runs on the GPU names the lock.
- A holder that looks dead (its run gone, lock still held): a worker never breaks it; it tells the coordinator, who
  verifies the run is gone, clears it with `gpu-lock.sh break <holder>`, and the queue proceeds.

**Baselines.**
- Never rerun an old-binary/old-code baseline that is already recorded: run only the new arm and compare it with the
  recorded number. Within a session, inherit the previous number.
- Run an old-binary baseline only in a clean new session (or when none exists for that exact setting; say which).
- Existing pack shards are a converter's byte-identity baseline.
- Committed docs cite commit + settings beside every new number; raw-file paths go to `docs/private/measurement-raw.md`.

**Landing perf.** A speedup that is real (outside run-to-run noise, arms interleaved in one session) and leaves output
bit-identical lands whatever its size. A change that alters output lands only through the KLD gate
([quality-kld](docs/quality-kld.md)). Judge a tweak on top of the change it builds on, not alone against the old code.

**Measurement hygiene.**
- Rebuild ReleaseFast from the head under test right before any live number; stamp commit + binary mtime beside it.
- Run every CPU-heavy agent-launched job (builds, tests, conversions, timings) under `taskpolicy -a`: inherited QoS never
  reaches the Super cores. State the QoS, lock and baseline beside every number.
- Bench thermal protocol: under heavy GPU workload AND with a die sensor over 90 °C, fans to max and 3 min idle before
  the bench starts; otherwise fans to max and a 10 s wait. Fans back to auto when the bench ends. Nothing else runs meanwhile.
  Under it one A then one B suffices; A B B A only when the expected difference is within a few percent.
- Never wait on `pgrep -f <string>` (the waiting shell matches itself): wait on END markers, PIDs, or `pgrep -x`.
- Launch flags outrank `model-settings.json`; confirm the load lines (`[kv-cache]`, `[mtp]`) show the intended arm.
- KLD is 16 prompts x 512 tokens scored to the first EOS (GLM: the 4x512 native teacher, code 2x512 reported apart);
  the teacher carries no lossy step of its own.

**Coordinator.**
- Reports partial work to the owner every :00 and :30 while work runs.
- Relays owner decisions and rule changes to ALL live workers at once.
- Keeps docs/ fresh: every landing updates the matching doc (knowledge, numbers, lessons learned) in the same
  landing, and this index when a doc appears. CLAUDE.md stays rules + index, never the knowledge store.

**Workers.**
- Report progress and blockers to the coordinator; commit per step; don't push or merge unless told.
- State the QoS, lock and baseline used beside every number; put new knowledge in the matching doc in the branch.

**Growth policy (ENFORCED).** This file holds rules and links only and stays well under 20 KB; every rule bullet is
≤ 3 lines. Knowledge, measurements and war stories go to `docs/` (converter knowledge to `docs/private/`); commit
messages carry the story of a change.

## Licensing

Ported kernels + vendored code are enumerated in `NOTICE` (the ONE place); `LICENSE`/`LICENSE-APACHE-2.0`/`NOTICE`
ride every packaging path (`tests/test_release_workflow_gates.sh`). To enumerate ports, grep comments for
`mlxfast|oMLX|MTPLX|mlx-lm|port`.
