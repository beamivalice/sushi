# Server: lifecycle, loading, settings and scheduling

How a model gets from a path to a serving slot and back: discovery, the arch gate, the one weight-loader decision,
load and unload on the inference thread, settings precedence, the scheduler's slots and batching, threads, and the
ownership rules that keep request data alive. Read this before touching `src/scheduler.zig` or its split files below, `src/model_registry.zig`,
`src/model_settings.zig`, `src/main.zig` or `src/cli.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [server-http-apis](server-http-apis.md),
[engine-memory-admission](engine-memory-admission.md), [engine-mlx-gotchas](engine-mlx-gotchas.md).

## Entry points

- `src/main.zig`: entry, CLI flags + subcommands (`run/pull/list/serve/launch/kld/update`).
- `--model <path>` and `sushi run <path>` accept a relative path: `cli.resolveRelativeDir` makes an existing directory absolute (cwd at launch) before anything reads it, because the layout, discovery and cache helpers take absolute paths only.
- `src/cli.zig`: alias → HF repo, resumable pull into `~/.sushi/models/<org>/<repo>`, `list`, `run` REPL.
- `pull` takes top-level files, `mtp/` and GLM's `GLM-5.3-Flash-DFlash2/` assistant (the loader auto-detects that folder; a
  pull that skips it serves GLM with DFlash2 off, logged as `[glm-dflash] off: no assistant found`).
- Headless and `--model` boots build their `ServerConfig` defaults from one `LaunchServe` (sampling,
  context, timeout): a flag one mode parses and the other drops is the class. `--no-drafter`/`--drafter` reach both too.
- **The embedded REPL uses in-process HTTP**: never fork `curl` from the resident engine for readiness checks or chat
  turns. Test `run` on a real TTY; a serving-only smoke test does not exercise its client.
- **A served arch must be in `model_discovery.supported_model_types`**: `run` refuses what `classifyModelPath` calls non-chat
  (guard test: every `model.served_model_types` entry classifies chat).
- **An arg loop with no else branch is a silent flag eater** (`cli.classifyUnparsedArg`): every `--flag` any script
  passes must be in main.zig's match list. Removed flags are rejected by name, never eaten.
- **`--parent-pid <pid>`** is for a host that runs sushi as its engine (`src/parent_watch.zig`): a thread polls the pid
  once a second and, once it is gone (or has reparented sushi), sends this process SIGTERM. Mid-load that ends the
  process; in the serve loop it is the ordinary graceful shutdown. Test: `tests/test_parent_pid.sh`.
- **`sushi --guest-manifest`** prints the JSON such a host checks before routing (`version.writeGuestManifest`):
  version, commit, `guest_api`, the mlx/mlx-c pins, `min_macos` from the binary's own target, `model_types`
  (`version.guest_model_types`, a subset of the served types) and the codebooks and window range the decoder
  accepts: `codebooks` under format `exl3`, `stq_codebooks` under format `stq` (the pairs `parseExpertQuant` admits).
  The release tarball ships it as `guest.json`, next to a `.sha256` of the tarball.

<a id="self-update"></a>
## Self-update

- **`sushi update`** (`src/update.zig`) reads `/repos/beamivalice/sushi/releases?per_page=100` and takes the highest
  SemVer carrying both assets (drafts never, prereleases by flag or by version only with `--pre`; `/releases/latest`
  is newest by date, so a 1.0.x hotfix published after 1.1.0 would be offered to 1.0.4). Never an equal or lower one.
- **Refused by name**: a source build (under `zig-out/bin`, or in a checkout holding `build.zig`), an app bundle, a
  folder without `lib/`, an unwritable install or parent, another process running from the install (`proc_pidpath`;
  `--force` skips it; a running `sushibar` does not count, it restarts itself on the new version), and an install folder holding anything the release does not ship (the swap would carry it off).
- **A Homebrew install** (real path under `/Cellar/sushi/`, any prefix; tap `beamivalice/homebrew-tap`) is brew's:
  `sushi update` and `--rollback` print `brew upgrade sushi` and exit 0, `/v1/update` and `/update` refuse naming it.
- **Steps**: curl into `<parent>/.<install>.update` (same volume; the tarball resumes), SHA-256 against the `.sha256`
  asset, `tar -x`, `codesign --verify --strict` (a Developer ID install takes only its own TeamIdentifier; an ad-hoc
  one takes either), `renamex_np(RENAME_SWAP)` (three self-undoing renames where the volume lacks it), then
  `<install>/sushi --version` and `--guest-manifest` must name the tag or the swap is undone; the old install becomes
  the one `<install>.previous`, which `--rollback` swaps back. One line per step to stderr and
  `~/.sushi/logs/update.log`; the `--relaunch` line redacts the `--api-key` value.
- **Every tool runs through `posix_spawn`**, never `std.process.spawn`: Zig 0.17's forks on macOS, and a fork of a
  server copies its whole MLX mapping.
- **Daily check**: a serving process (`serve`, `--serve`, `run`) asks GitHub at most once a day on a detached thread,
  with the in-process HTTPS client and the cached ETag (`~/.sushi/update-check.json`); a failure is one debug line and
  a retry an hour later. A newer release logs `sushi X is available: run \`sushi update\`` and shows in `/props` and the
  `run` banner. Boot logs `[update] daily check on|off (source)`: `--no-update-check` > off for a source
  build > `server.update_check` in `model-settings.json` > on. The loop wakes every minute and reads a published value
  (`setAutoCheck`) before each look, so the Settings toggle takes hold without a restart; `sushi update`, `/update` and
  `POST /v1/update` are never gated. `SUSHI_UPDATE_API` is a test-only hook (tests/test_self_update.sh).
- **Pack revision** (`pack_update.zig`): `--no-update-check` and `server.pack_update_check` gate a look at the loaded
  pack's `config.json` on Hugging Face (`SUSHI_HF_ENDPOINT` is the test-only base URL, `tests/test_pack_update.sh`)
  once after each load (`markReadyLocked` starts a detached thread per pack folder) and then daily, retrying hourly
  while there is no answer. Hugging Face is asked at most once a day per repo even across restarts and `sushi list`
  runs: each answer (repo, `checked_at`, revision) is kept in `~/.sushi/pack-update-check.json`, written beside and
  renamed over, and reused while under a day old. `sushi list` honors the key too; `sushi pull` is never gated. The
  repo comes from
  the pull manifest's `repo` or the folder name's match to an alias. `/v1/models` carries the result as
  `meta.pack_update`; a 401/403/404, a timeout or a bad body claims nothing.
- **Update and restart** (`/v1/update`, the REPL's `/update`): the server shuts down through its SIGTERM path, then
  main's last defer REPLACES the process (`posix_spawn` with `SETEXEC | CLOEXEC_DEFAULT`, fds 0-2 kept) with
  `sushi update --relaunch -- <argv>`, which in turn replaces itself with `<install>/sushi <argv>`, new or restored.
  Same pid, same argv, same terminal and parent: host, launchd and `nohup` supervision carry through, and no socket
  outlives the server. An exec that cannot happen logs `[update] cannot run …` and exits 1; the outcome lands in the
  cache so `/props.update.error` names a failure after the relaunch.

## One-shot prompts

- `sushi --model <path> --prompt "text"` (shorthand `-p`) and `sushi run <model> -p "text"` execute one request
  and exit, even with redirected stdin. The prompt is passed verbatim, including leading/trailing whitespace.
- Explicit prompts use the normal scheduler and chat request policy through a private loopback listener on an OS-assigned
  port. They never connect to an existing server or enter the REPL; the client is joined and the listener closes on exit.
- `--think` enables thinking; `--think <word>` selects an effort the `--model` model's served effort table takes (built-in GLM low|high|max, with
  `medium` served as `high` and `xhigh` as `max` and logged; Qwen off|low|medium|xhigh, with `high` and `max` served as `xhigh` and logged; MiMo off or any level as on; `/think` alike); another word fails with the accepted list and a
  nonzero exit. Omission keeps the model default, one-shot and REPL alike. A model loaded on demand takes the word where it can, else keeps its own
  default ([server-http-apis](server-http-apis.md)).
- MTP/KV settings, context, timeout, reasoning budgets, and sampling flags follow the serving policy, including
  model-settings precedence. One-shot sampling retains its defaults: 100 output tokens, temperature 0, top-p 1, top-k 0.
- `--stream` flushes the same reply incrementally; otherwise it buffers until success. Stdout contains only the reply,
  with reasoning in `<think>` tags; diagnostics go to stderr. HTTP/decode errors and incomplete streams exit nonzero.
- `serve`, `--serve`, `--host`, `--port`, and `--tool on` conflict with explicit prompts and fail before loading.
  Use the REPL for client-side tools. Guard: `tests/test_prompt_flags.py`; `SUSHI_PROMPT_MODEL` enables live Qwen checks.

<a id="startup"></a>
## Startup order

A `--model` boot (`server.serve`) serves before its model is loaded:
1. `ensurePortFree` (main.zig, before the config is read), then the listener binds.
2. `Scheduler.init` claims the launch model's registry entry (`claimStartupLoad`: `loading`, the claim's reference,
   its gate estimate) and spawns the inference thread, which runs the load's refusals: arch gate, launch-flag and KV
   refusals, the streaming plan (`ExpertStreamingRequired`, the budget refusals), the memory preflight (context and
   weights). `init` returns at `model_load.signalPrechecked`, just before the weights, so a refusal still exits
   non-zero before anything is served, with the same message.
3. The gauge sampler starts, `Server listening on …` is logged and the accept thread (`acceptLoop`) starts. Requests
   are answered while the rest of the load runs ([server-http-apis](server-http-apis.md#during-the-load)).
4. `awaitStartupLoad`. A failure here logs `[scheduler] model load failed: <name>`, answers every waiting request
   with that error, stops the accept loop, drains the connection threads and exits non-zero (`error: LoadFailed`).
   The server never stays up without its launch model: scripts and the app read the exit as the load's verdict.
5. After the load, as before: the request hooks, idle eviction, the cache and `Concurrency:` lines, the SIGINT/SIGTERM
   handlers, `on_ready`, the pinned context, the route list and `chat in your browser:`. Then `releaseStartupPin` and
   `boot_gate.process.open`: a request that got the model earlier waits at the gate for this step, so it never runs
   before the context is pinned or the hooks are in.
- The signal handlers stay after the load: until then SIGTERM (and `--parent-pid`) ends the process at once, as a
  load cannot be cancelled.
- `SUSHI_TEST_LOAD_DELAY_MS` holds the startup load after its refusals (test seam, [envvar](envvar.md)). Guard:
  `tests/test_early_listener.sh`.
- Headless (`--model-dir`, no `--model`) and the embedded bench client (no listener) take the same path with nothing to
  wait for.

## What loads

- A rescan makes a failed load retryable only when discovery finds the same ID at the same path; it clears the error and refreshes the on-disk byte count without disturbing live entries.
- A qwen4_exp load that fails after the n-gram state or the MTP head exists frees both (`Qwen4Mtp.deinit`; the
  state's `deinit` joins the table's warm thread first). Guard: the `QWEN4_TEST_MODEL` FailingAllocator sweep over
  `loadQwen4Mtp`.
- A streamed load that fails after its expert engine exists deinits it (`initExpertStream`, both archs): freed
  alone, its I/O workers ran on in freed memory. Guard: a FailingAllocator sweep over `initExpertStream` with the
  imatrix collector armed.

- **Bind**: `server.resolveBind` takes `--port`, else `server.port` of `model-settings.json`, else `127.0.0.1:12345`; `--host` takes an IPv4 literal, `0.0.0.0` or
  `localhost` (= 127.0.0.1; anything else is refused by name, never widened). Before any model loads,
  `ensurePortFree` probes the address with a connect AND a bind, so a listener or a bound-but-silent socket both refuse
  with "port N is already in use"; the listener binds with SO_REUSEADDR only, so a racing second sushi fails its bind
  with the same message instead of co-binding. Guard: `tests/test_port_conflict.sh`.
- The arch gate: the loader refuses any `model_type` outside `model.served_model_types` (`qwen4_exp`, `mimo_v2`, `glm5_next`, `deepseek_v41`) by
  name (`ArchitectureUnsupported` → 503). A checkpoint in an unsupported file format is refused by name
  (`ModelFormatUnsupported` → 503; `--model` exits). A served arch off the supported list (MiMo, from 1.5.1) sets
  `ModelConfig.unsupported_note` at parse, and the server logs that `[load]` warning on every load.
- The pack-revision check is the first thing `loadWeightsForConfig` does after the arch gate, for every arch: a config
  that names `pack_revision` and a shard (MTP shards included) that carries another revision or none is refused
  `PackRevisionMixed` (503 `pack_revision_mixed`; the log names each file) before any weight bytes load
  ([pack-format](pack-format.md#pack-revision)). `sushi pull` ([usage](usage.md)) compares checksums, not sizes
  (`src/hf_pull.zig`); the HF address is built in one place, `hf_pull.hfUrl`, over `hf_pull.endpoint()`.
- **The weight loader is ONE decision** (`model.loadWeightsForConfig`: streaming index > MiMo source trunk > vision >
  plain). A second site builds a model the server never serves — a MiMo pack read without its source trunk binds the
  raw FP8 fused QKV and its logits stop following the routed experts.
- `qwen4_exp` `trunk_quant` rides `config.trunk_quant`; `Transformer.init` binds each trellis projection per tensor and
  consumes the banks it stacks from the map ([qwen4-arch](qwen4-arch.md#trunk-rules)).
- **A missing tensor is a load ERROR, never `unreachable`** (`error.MissingWeight` → named 503 via
  `loadErrorFromName`); a load failure crosses the inference thread by NAME (`req.error_name`).
- Discovery (`src/model_discovery.zig` / `src/model_registry.zig`): two-level org/name, multi-root, streaming stubs,
  multi-model registry. **`--model-dir` is REPEATABLE** (`discoverModelsMany` merges roots FIRST-WINS). One path never
  registers under TWO ids (`registry.peekByPath`).
- **Discovery never lists a dot folder or a folder named `test-fixtures` BELOW a root**, at either depth (`skipsBelowRoot`;
  one debug line per skipped fixtures folder). The repo's tiny test models live in `${SUSHI_MODELS_DIR}/test-fixtures/<name>`
  and are not packs to serve. A root the user NAMES is always scanned (`--model-dir .../test-fixtures` lists them), and
  `--model <path>` never goes through discovery; rescan shares the rule.
- **The auto `--max-resident-mem` (80% of the GPU working-set limit) bounds CO-RESIDENCE only**: a cold load evicts
  every other model first, and one that then loads alone is the load preflight's call (`mem_cap_binds_alone`); an
  explicit `--max-resident-mem` binds a sole model too. Before this the 2.3bpw MiMo pack could not cold-load through
  `/v1/load-model` (the app's path) on a 128 GB Mac at default flags.
- **A reload FREES the CPU state `unloadResident` retains** while the entry is `.loading` (`releaseRetainedCpuState`):
  a reader holding no refcount takes the mutex AND skips them while `.loading`.
- **A cold load pins its model at the claim** (`tryBeginLoadLocked` takes the requester's reference), so the entry cannot be another load's eviction victim between `markReady` and the requester waking; an unload that wins that window is a 503 (`NotLoaded`). A failed load drops the MLX error it latched (`failLoadRequest`), or the next request fails on it.

<a id="settings"></a>
## Settings precedence

- **An explicit launch flag outranks `model-settings.json`**, which outranks the default (`model_settings.pick`;
  `--ctx-size 0` = not given). Applies to `--mtp/--no-mtp`, `--kv-quant`, `--ctx-size`, `--mtp-typical/--mtp-tokenv3`,
  `--mtp-greedy-tail`, `--ssd-budget-gb`, `--ssd-prefetch`, `--preserve-thinking`, `--think-penalty`, `--vision/--no-vision`, `--max-tokens`; a request's own field still applies on top. Design reviews reject "file beats
  flag".
- **`--mtp-min-depth` / `--mtp-max-depth` are launch-only**: the range is a property of the machine, so it has no
  `model-settings.json` key ([engine-mtp](engine-mtp.md#depth-range)).
- **`--no-mtp-lookup` is launch-only too** (no `model-settings.json` key): it turns off the prompt-lookup drafts in MTP
  rounds (Qwen, MiMo) and GLM DFlash2's lookup chains.
- A flag that shapes a LOAD is retained on the Scheduler with its `*_explicit` bit (`ensureLoaded`'s cold-load
  `LoadRequest` is a SECOND site); read via `server.manualContext` / `kvCacheFor` / `mtpChoiceFor`. Each load logs its
  resolved value and source (`[kv-cache] kv8 (source); ctx N (source)`, `[mtp] on|off (source)`; `/props
  settings.mtp.source`). Guard: `tests/test_cold_load.sh`.
- MTP's default is ON for `qwen4_exp` and `mimo_v2` (GLM's MTP stays off; source `default`; `/props settings.mtp.default_on` true); the flag and
  the file can only turn it off, or on for an SSD-streamed pack.
- Per-model settings live in `~/.sushi/model-settings.json` (`src/model_settings.zig`: `ctx_size`, `kv_quant`,
  `mtp`, `mtp_acceptance`, `mtp_greedy_tail`, `ssd_budget_gb`, `ssd_prefetch` (0..100; ignored with a log line on an arch with no expert prefetcher, whereas the flag is refused there), `preserve_thinking`, `think_penalty`, `logit_bias_file`, `vision`, `max_tokens`: the per-model twin of `--max-tokens`, a positive integer at most the model's context, the omitted-`max_tokens` budget of `ModelConfig.defaultMaxTokens`), stamped at BOTH load construction sites and resolved
  ONCE in `doLoadOnInferenceThread` (`preserve_thinking` per render, where a request can override it, and logged as
  `[chat] preserve_thinking on|off (source)` at load; `think_penalty` per request, logged as
  `[think-penalty] lambda L (source)`); read via `server.manualContext(config)` / `configuredKvQuantFor(config)`, never the raw
  server config.
- **Effort tables live under `"effort:<family>"`** (`qwen4_exp`, `deepseek_v41`, `glm5_next`): `default` and one
  object per effort word holding only the changed `offered`, `sends` and `cap` (shape, validation and precedence:
  [server-http-apis](server-http-apis.md#effort-tables)). Read once at start (`model_settings.loadEffortTables`, a bad
  field skipped with an `[effort]` line), replaced by a settings save at once, with no reload. Precedence: a request's
  own budget > the table's cap (a word with no cap has none); `--think` > the file's default word > the built-in one.
  No launch flag: `--reasoning-budget` is gone ([flags](flags.md#removed)).
- **Server-wide keys live under the file's `"server"` object** (a model's key is its absolute path, so they never meet):
  `prefix_cache_disk_gb` (the SSD tier's size, 0 = off), `models_dir` (the discovery folder when no `--model-dir` is
  given, else `~/.sushi/models`) and the booleans `update_check` (the automatic sushi release check) and
  `pack_update_check` (the automatic pack revision check), both on by default; `--no-update-check` pins both off;
  `skip_mem_preflight` (off by default: load without the free-memory check; `--skip-mem-preflight` pins it on, locked;
  the Settings switch reaches the next load at once through an atomic in `scheduler.zig`, read at boot from the file
  when no flag is given); and `port`
  (1-65535, the bind when `--port` is absent, also read by `sushi launch` and sushibar; flag > file > 12345, read at start,
  so a running server keeps its port; `--port` pins it and the API reports it locked).
  The order is the same: `--prefix-cache-disk` / `--model-dir` outrank the file, which
  outranks the default; a file value counts as the operator's own number (no formula sizing). `main.zig` records which
  flags were given (`model_settings.launch_locks`), and `GET /v1/settings` reports those keys locked. The web UI edits the
  file through `POST /v1/settings` (validated, temp file + rename, [server-http-apis](server-http-apis.md#ssd-prefix-cache-and-settings-v1prefix-cache-v1settings));
  `model_settings.model_keys` is the one list of editable per-model keys, and the loader applies exactly the values its `validate` accepts
  (one parse and one range: a hand-edited `ssd_budget_gb` over 512 is unset to both). The SSD ask (`server.diskAsk`) is atomic: the settings POST
  writes it from a connection thread, and every load, cold loads and their checkpoint stride included, reads it at load time rather than a boot snapshot.
  A key that does not apply to an arch (`model_settings.applies`) is logged as ignored at load and listed as not applying by `GET /v1/settings` and `/props`
  (`settings.mtp.acceptance_applies`); `ssd_prefetch` also logs that it is ignored on a DeepSeek pack without STQ experts. `model_settings.Fit` (built by `server.settingsFit` from a loaded config, by the settings API from a stub's files) is the one set of facts behind `applies`, `limit` and `refuseEdits`; add a gate there, never in the page. A `kv_quant` file value GLM's latent cannot store (4) is dropped at stamping with a warning, never a refused load. `SUSHI_MODELS_DIR` is a test variable the engine does not read.
- **The prefix cache's other tiers are launch flags only**: RAM retention is off unless
  `--prefix-cache-mem` is given with a size (`off` or `0` keeps it off), and the SSD tier is on unless `--prefix-cache-disk 0`
  or `--prefix-cache-entries 0`, sized per model at load ([engine-prefix-cache](engine-prefix-cache.md#defaults)); its size alone has the
  `server.prefix_cache_disk_gb` key.
- A new per-model setting or launch flag follows this order, carries an `*_explicit` bit through both load sites and
  cold loads, and logs its resolved value with its source at load.
- Load-time context bills see explicit KV and MTP choices from the load's start (`serve` publishes them before `Scheduler.init`), including `--no-mtp`.

## Scheduler and batching

- `src/scheduler.zig`: slots, inference thread (sole MLX caller), queues, batching, admission, spec wiring, hot-cache
  budget revise.
- `src/model_load.zig`: load plan, streaming plan, load bills; `src/slot_admission.zig`: slot and memory admission, batching
  policy; `src/mtp_tick.zig`: grouped MTP tick; `src/slot_outcome.zig`: slot end, finish outcome, loop-stop reasons;
  `src/vision_queue.zig`: vision encode queue, media deferral.
- Text slots BATCH-decode on `qwen4_exp` (`configBatchesDecode`); `--max-concurrent` sizes the submit queue. A resident
  `qwen4_exp` decodes plain slots as rows of one forward (`forwardQwen4DecodeRows`, up to eight, no padding, no cap):
  each row's recurrence, attention, PLE and KV append run as the slot's solo tick runs them, and the ops that read the
  same weights for every row (hyper-connection reads, projections, routed experts, lm_head) share one pass through
  kernels whose per-row arithmetic is the single-row one, so a slot's output does not depend on who shares its tick.
  A STREAMED load and the other GDN trunks keep the padded batch, capped by PADDING WASTE (`batchedKvKeepCount`,
  `MAX_PAD_WASTE` 1.5 < 2.0); the grouped MTP verify keeps both cap functions (`groupKeepCount` lifts the cap for a group
  billed <= 4096 rows whose longest true context is >= 131072,
  [qwen4-qsa-long-context](qwen4-qsa-long-context.md#small-sparse-groups-at-long-context)).
  Resident MiMo batches plain slots as rows of one forward, capped by `batchGroupCap` (4) with no padding
  ([mimo2-arch](mimo2-arch.md#batched-decode)); resident GLM does the same through `verifyGroups`, and a
  drafting GLM slot joins as a plain row when its model has company ([glm5-arch](glm5-arch.md#concurrency)).
- A cold prefill YIELDS to decode ticks at chunk boundaries (`scheduler.interleaveDecodeTick`;
  `SUSHI_PREFILL_INTERLEAVE=0` restores). Greedy byte-identical.
- `--prefill-decode-share S` (default 0.25) targets the fraction of wall time given
  to existing decoders during another request's prefill. Values above 0.9 clamp; negative values and NaN refuse.
  Zero preserves one decode tick per boundary; a nonzero share adds ticks until `chunk_ns*S/(1-S)` is spent or
  decoders finish. `SUSHI_PREFILL_INTERLEAVE=0` disables the share.
- **The share never narrows a prefill chunk** (any arch): a request's chunk width, bill and numerics do not depend on
  what else decodes; decoders get time only between whole chunks. Hosted decode time is excluded from prefill compute
  time but remains part of request latency.
- **Serial ≠ exclusive**: only a slot driving a module-owned decode state is exclusive (`slotExclusiveDecode`);
  qwen4's state is read-only shared and batches freely. The batched-decode gate reads DISPATCH, not ARMED flags
  (`slotTicksRegular` asks `specTickMode`). A batched decode guard that only runs at N=1 pins nothing:
  `tests/test_batched_equivalence.sh` runs a real two-stream arm.
- **A GLM slot owns its `glm5_forward.Request`** (`Slot.glm5_request`, handed to the forward as
  `ForwardCtx.glm5_request`; a GLM forward without one is `GlmRequestMissing`), so GLM requests share a model; only a
  streamed GLM load stays exclusive. The boot line and `/props batching.reason` (`ok` vs `exclusive`) say which.
- **An exclusive slot blocks admission until the inference thread releases it** (`heldExclusive`: `decoding` plus
  `cleanup_queue`; `complete()` keeps a slot in `decoding` until its pass is out): a disconnect must not let the next
  request prefill while the slot still owns the stream (`GlmStreamRequestBusy`).
- **A cancelled slot commits at the cull when its state is not the slot's own** (`commitsAtCull`: GLM's native request,
  DeepSeek's live record), and its cleanup drain skips the commit (`committed_at_cull`). The cull drops the slot from
  `decoding` before `complete()` queues it, so the next exclusive request is admitted in between; a DeepSeek commit
  run after that publishes onto the next request's state.
- **A plain batched tick stop-checks a slot with no pipeline state** (`batchEntryStops`) before forwarding its pending
  token: a fresh DFlash2 slot whose prefill sampled EOS finishes `stop` with nothing published, as solo does.
- **Count a pending token once, in the decoder that consumes it.** Planner preflight uses `checkDispatchStop`
  (budget/EOS only), never the stateful pad-run check. Batch sampling does not count the next token; batch entry
  does, including a new token sampled while draining a pipeline. `tests/test_pad_run_replay.py` replays a local
  `SUSHI_PAD_REPLAY_REQUEST` fixture in warm concurrent waves; private captures are never committed.
- **A slot the inference thread drops releases its GLM state there** (`releaseNativeState`: finish, error, cancel
  (also one landing after prefill, `postPrefillTerminal`), failed prefill), not when its connection thread completes it: an errored request's reserve once held 11 GB.
- `src/generate.zig`: generation, `commitForcedTokens`; `src/mtp_policy.zig` / `src/mtp_round.zig`: MTP policy and rounds; `src/sampler.zig`: sampling;
  `src/prefill_plan.zig`: prefill chunking; `src/degenerate.zig`: `StallClock`, loop-stop tiers. `src/tokenize_cache.zig`: per-LoadedModel LRU of rendered+encoded prompts.

## sushibar

`sushibar` (`src/sushibar_main.zig`, `src/sushibar/`) is a separate pure-Zig binary: the Objective-C runtime and AppKit, no MLX, jinja or server module (only `default_port.zig` is shared with `server.zig`).

- **Design rule**: a small binary with no new framework, bundle or dependency; refined, light motion (short Core Animation cross-fades, skipped under Reduce motion) instead of more features; convenience first, and everything else belongs in the web UI.

- **Launch**: bad arguments and an ssh login (`SSH_CONNECTION`/`SSH_TTY`) are refused by name before anything runs; a live `~/.sushi/sushibar.pid` (verified as a `sushibar` process) prints "already running" and exits 0. The detached copy holds an `flock` on `~/.sushi/sushibar.lock` for its life (an `exec` relaunch takes it again), so two launches at the same instant leave one sushibar. Otherwise the terminal copy spawns itself with `POSIX_SPAWN_SETSID` (a session leader is the detached copy), waits for it to take the pid file, prints one line and exits 0.
- **Menu**: a header card (a custom `NSView`, `header_view.zig`; strings, rows and sparkline points come from `sushibar_logic.zig`) over system-symbol items. The card shows the icon (colour online, outline offline), `sushi <version>` in the secondary colour with ` · server X` when the running server's version differs, and only when a release waits an accent ` · X available`, a status pill (Running, Loading, Starting, Stopped, No model, Port in use, No sushi, Not answering, Key needed, Metrics off, Updating…, Connecting… until the first poll of a newly chosen port) and two lines that give the cause or the next step instead of repeating the pill: the model with `Port N · X GB in use`; "Ready to start · port N" over the last model's name (the one Load brings back); "Another program holds this port" or "sushi is not beside sushibar" (why Start is off, in an orange pill); "Holds the port, does not answer" (a sushi that does not answer; Stop is offered); "The server requires an API key"; and, with no model and no last model, an accent line "Open Models to load one" that opens the web Models page (with a last model the card reads "Ready to load · port N" over its name). The accent "X available" tail opens the release page (`/props.update.url`, else the project's latest release) and the accent pack line opens the Models page (a click closes the menu first, `header_view.zig`). When the loaded model's `/v1/models` row has `meta.pack_update`, an accent line `Pack 2.0 available (you have 1.0)` under the model name, and, while a model is loaded, GEN TOK/S (LAST GEN TOK/S when idle; GENERATION TOK/S would not fit the tile), PROMPT TOK/S and GPU columns (`gpu_utilization_pct` as the value, `gpu_clock_mhz` in the label) over a five-minute generation-speed sparkline from `/metrics.json?since=` history, drawn in the accent colour over an area that fades downward, with a dotted baseline where the server was down or sent no history, and a caption with its peak and span. A speed reads one way in the title and the card: one decimal below 100 tok/s, whole numbers from 100. It refreshes in place while the menu is open (the timer runs in the common run-loop modes), redraws only when what it shows changed, and cross-fades over 0.2 s when it does. Below it: one row of two buttons (`button_row.zig`, same look as the bottom row), Start Server / Stop Server on the left (Stop Server… when sushibar did not start it, Starting…/Stopping… while it changes, Stopping… until the signalled sushi has exited since it closes its port first, Force Stop when a Stop has not taken after 15 s) and Load / Unload on the right (see Start/Stop below), with an inert grey line under them that says why Load is off, Port N with its submenu (the port rows, a separator, the checkable Listen on LAN, another separator, and the port field; when the LAN item is on, Start binds every interface so devices on the network reach the server, a running server keeps its current bind which the note after a click says, and the card names the bind `Port N · LAN`), Dashboard, Chat, Models, Settings (⌘1-⌘4, disabled while nothing answers), Copy Pull Command while a pack update waits (a pack is hundreds of GB, so the menu never pulls it; the command is its tooltip and the title reads Copied for ten seconds after a click), a note line that carries the last update, start or exit news (cut to 46 bytes with an ellipsis, the full text in its tooltip, so the menu keeps its width; news, not a problem, is dropped when the menu opens a minute or more after it was given). A finished update check replaces only the notes about updates; a server note (exited, did not start, stopped at, did not stop) stays until a server answers on its port again or Start is pressed. A note that reports a problem (a failed update or check, a server that did not start, exited with a signal or exit status, or went away: "stopped at HH:MM" for one sushibar did not start) is orange and enabled, and a click opens the Log window; other notes are grey and inert. Then one row of three rounded buttons (`button_row.zig`, a custom menu-item view): Log, Update and Quit. A button is a symbol and a label, fills lighter under the pointer (a tracking area, cross-faded in 0.12 s and out in 0.2 s) and darker while pressed (at once); a new label cross-fades in 0.18 s; a disabled button (Checking…, Up to date) keeps its fill and dims its label; a click runs its action, closing the menu first for Log, Quit and an update that asks first. Update keeps the menu open and shows the answer live: Check shows a disabled Checking… with a spinning wheel at once, then a disabled Up to date, Retry (enabled, the check failed) or, when a release waits, Update in the brand orange (`#EE5A2F`), whose click shows a disabled Updating… until the bar relaunches into the new version or the update fails (the note says why and the button is Update again). With a model loaded the click first asks (an alert naming the model and the port, with Release Notes), since the update restarts the server mid-request. A Homebrew install gets one orange Upgrade button that copies the command and says Copied for ten seconds. An answer stays for a minute across closing and reopening the menu, then Check returns (`updateState`, `updateButton` and `shownResult` in `sushibar_logic.zig`). The button follows the main-thread timer, which runs in the common run-loop modes and so during menu tracking, and a click re-renders the menu at once instead of waiting for the next tick. Quit's ⌘Q lives on a hidden "Quit sushibar" item. Quit leaves the servers sushibar started running and its tooltip says so; while the menu is open a 0.1 s timer watches the Option key, and with Option held and such a server alive (one this sushibar started, or the watched port's verified `sushibar-server-<port>.pid`) the button reads Quit & Stop, which sends each the Stop of the menu and quits (`quitButton`). The app also installs a main menu (Quit, Edit, Window) that no one sees, since an accessory app has no menu bar: it is what carries ⌘C, ⌘A, ⌘F, ⌘W in the log window and ⌘V in the Port field. The menu bar icon is the colour sushi while a server answers and the outline template otherwise. An orange badge (a layer-backed view on the status button, not tinted) marks the icon while a server is Not answering, wants a key, holds a port another program took, or exited on its own; the icon and the badge cross-fade when they change. The title shows only what runs now: the live generation speed, `generating…` before its first rate, `prompt…` while it reads a prompt; an idle server shows the icon alone and the card keeps the last speed. Light and dark are checked with `SUSHIBAR_RENDER_HEADER=<dir>` ([envvar](envvar.md)).
- **Log**: the Log button opens one reused window (`log_view.zig`) with the watched port's server log: its last 2000 lines, then every complete line the file grows by (polled twice a second while the window is open), in a monospaced read-only text view that follows the end only while it is scrolled to the bottom. A file that shrinks or is replaced (rotation) is read again from its start under a "log restarted" line; a window past 4000 lines drops its oldest lines in place (the scroll position shifts by what was removed, a follower stays at the end, a selection below the cut stays). The window follows the watched port and remembers its frame; Copy copies the selection, or the whole log with none, and says Copied for a moment (cross-faded); Find (and ⌘F) opens the text view's own find bar, which highlights as you type; when lines arrive while the view is scrolled up an accent "New lines" capsule fades in at its lower right, and a click on it (or scrolling to the end) follows the end again; Open in Console and Reveal act on the file shown and are disabled while there is none. Lines are not coloured by level: the server's log file carries no level, and a keyword guess would colour request text. The file is `~/.sushi/logs/sushi-<port>.log` unless the listening server's own argv (read by the poll thread from `KERN_PROCARGS2`, again only when the pid changes) carries `--log-file`: a path is followed, `off`/`none` and a relative path are said in the window, since the former has no file and the latter's folder is the server's working directory, which sushibar cannot read. With no server listening, or an argv that cannot be read, the default file is the guess and a missing file's notice names it and `--log-file`.
- **Title**: polled once a second off the main thread from `/metrics.json?since=<ms>`. Live = tokens added to the `decode` rows of `sessions` (same `request_id`) over the wall time between polls; the card's last speed = `generated_tokens / decode_seconds` of the newest finished row of `requests`; 401/403 reads "key needed", 503 "metrics off", no answer (a poll times out after 4 s) is offline. The title is in monospaced digits, a dot follows it while an update waits, and the status line is its tooltip. The icon is an outline while offline and in colour once any server answers (`icon-offline.svg`, `icon-online.svg`, embedded; swapped only on a state change). AppKit runs only on the main thread, which renders the published snapshot from a run-loop timer.
- **Start/Stop and Load/Unload**: the left button reads Start Server or Stop Server (the header and the Port item name the port). Start and Load are two actions and never run together: Start brings the server up empty and Load is off until the server answers. Start and Stop re-render the menu at once, as Load and Update do. Start runs the `sushi` beside the executable as `--serve --port N` (`--host 0.0.0.0` too while `Listen on LAN` is on; no model flags, so the saved `server.models_dir` applies) in its own session and records `~/.sushi/sushibar-server-<port>.pid` (`pid port`), one file per port; the server comes up with no model. `sushibar.json` holds a `model` (the id the watched server last listed as ready, saved whenever it changes), the last model, and a `lan` key only while `Listen on LAN` is on. The right button is Load while no model is loaded: enabled only when the watched server answers (`.up` or metrics off), no load is running and a last model is known; its tooltip names that model and a click posts `{"model":<id>,"default":true}` to `/v1/load-model` with the page's Origin, as the Models tab does (an id the server does not know, such as a pack registered by path and not under the models folder, answers 404 and the note says so). It shows a spinning Loading… while the request blocks (also while the page loads a model), the request waits up to 30 minutes, and a server stopped under it reports through its exit note, not a load error. Off, it keeps the label Load and the grey line says why: "Start the server to load a model", "Waiting for the server to answer", "The server does not answer", "The server requires an API key", "No model to load yet: pick one in Models", and "Loading <model> on port N…" (or Unloading) while an operation runs on another port than the watched one, since only one model operation runs at a time. With a model loaded the button reads Unload (tooltip names the model, says the server keeps running) and posts `{"model":<id>}` to `/v1/unload-model`, showing Unloading… meanwhile; it never stops the server. Stop while a load runs stops the process. The last model is the only one either button acts on (`modelButton` in `sushibar_logic.zig`); an id is kept up to 512 bytes (`max_model`) and a longer one is not stored cut: a note says Load cannot bring it back. A server that exits before it ever answered shows "Port N: server did not start (exit 1)" and one that answered and then exits on its own "Port N: server exited (signal 11)" in an orange note (a started server counts as having answered once the watched poll or, whatever port is watched, a probe of its port during the process scan got a reply; sushibar reaps only the pids it started and reads each wait status, and after relaunching itself takes the ones it started before back from their pid files; an exit it reaped is not reported again as a stop); a server it did not start that stops answering shows "Port N: server stopped at HH:MM", unless Stop was just pressed; it is disabled while another program holds the port (a connect succeeds and no sushi listens) and when no `sushi` sits beside sushibar. Stop on a server sushibar started sends SIGTERM to its pid at once, while that pid is still a `sushi` whose argv has `--serve --port N`. Stop on any other sushi listening on the port asks first (an NSAlert naming the port and pid), then re-scans and signals that one verified pid; only an executable named `sushi` that listens on the watched port is ever signalled. When the pid a Stop signalled still serves the port 15 s later (`stopStuck`), an orange note says "Port N: the server did not stop" and the item reads Force Stop: SIGKILL to that same pid, re-verified by the same scan, never to another. The listener scan is libproc (`proc_listpids`, `proc_pidinfo` fd list, `proc_pidfdinfo` socket info: a TCP socket with a local port and no foreign port, and the address it is bound to), no sudo and no `lsof`. Every URL and Origin sushibar builds for a port (`originOf`) uses the address of the sushi listening there, so a server on `--host 100.x.y.z` is watched at that address; a wildcard or loopback bind, and a port with no sushi listener, use 127.0.0.1 (the Start path and the port-taken check stay on loopback; a server on a non-loopback bind refuses `POST /v1/update`, so sushibar's Update reports that refusal). Quit ends sushibar alone; Quit & Stop (Option) also stops the servers it started.
- **Port**: the Port submenu lists the default port first (`server.port` of `model-settings.json`, else 12345, re-read when the menu opens; it has no remove button), then the ports the user added, then the watched port if it is none of those (label "N — model", "not running" or "not answering", a check on the watched one). A sushi on any other port is not listed, whoever started it: a port appears when the user types or picks it. Each row is a custom view (`port_row.zig`): hovering a row that can be forgotten shows an x, and a click on it (or on the row) closes the menu and then runs. An added port is one the user typed or picked; it is kept in `sushibar.json` as `{"port":N,"ports":[…]}` (at most 16, the oldest dropped first; an older file with the one key `port` still reads) and can be forgotten unless it is the default; the watched port can always be, even when `--port` or a file from before added ports named it and it was never added. Forgetting a port drops its row even while a sushi still listens there; forgetting the watched port watches the default port. It never stops a server. Rows are custom views, so the arrow keys do not step through them; the row under the pointer is highlighted in the accent colour as a system menu item is, and the default port's row carries a "Default" tag where the others show the x. The watched port is polled on its own deadline; a listed port's sushi is asked for `/v1/models` only when the submenu opens (all in parallel, 1 s timeout, never on the poll thread), so a hung sushi elsewhere cannot slow the watched poll and sushibar does not keep hitting other servers; a sushi on an unlisted port is never contacted. A listener not yet asked reads "…", one that did not answer reads "not answering", never "no model" and, last, a field (`port_field.zig`, a custom menu-item view: label, text field, Switch button). Return or Switch on a valid number (1-65535) switches the watched port and closes the menu; anything else turns the text red with a "1–65535" hint and keeps the menu open. A click inside the field makes the menu window key so it takes typing. The choice is saved in `~/.sushi/sushibar.json`; an explicit `--port` beats it, which beats the default port. A server sushibar started on another port stays tracked by its own pid file.
- **Version and update**: the menu shows `sushi <version>` of the sibling binary (`sushi --version`, at launch and after an update) and the running server's version when it differs. A check is `/props.update` of a running server (once a day, never GitHub from sushibar) or `sushi update --check` (no server; at launch, then once a day, an hour after a failure; sushi's own ETag cache applies). The automatic one follows `server.update_check` (read from `model-settings.json` at each due check) and answers from `update-check.json` while that is under a day old, so a bar launch asks GitHub only when the cache is a day old; the Check button stays a forced check. A failure is the CLI's last `update:` or `error:` line, never a log header. A server update that does not finish is reported after two minutes, and following it stops when the watched port changes; the Check button always runs the CLI. A newer release turns the Update button orange (its tooltip reads "Update to sushi X") and adds a dot after the title. With a server on the watched port the update is `POST /v1/update` (Origin the server's own `http://<host>:<port>`, as the page sends) and sushibar reads the server's version first (an answer can only be called an update against it) and follows the page's flow (health down, back up, `/props.update.current`/`error`); with none it runs `sushi update` and shows the updater's own refusal (another sushi from the install folder blocks it by pid; a running sushibar does not). A Homebrew keg or `/props.update.command` makes the button Upgrade: it copies `brew upgrade sushi` (or the server's command), and nothing is run. Whenever the sibling's `sushi --version` differs from the one sushibar launched with (re-read at each check, after an update and when the server's version changes), whatever replaced it, sushibar re-executes itself (same pid, `--port N`). sushibar keeps the path it was started by and resolves it with `realpath` at each spawn, so a Homebrew `bin/sushibar` symlink finds the current keg's `libexec/sushi` (brew's `opt/sushi/libexec` link when the keg is gone), is recognised as a brew install, and survives `brew upgrade` deleting the keg it started in; the relaunch execs the same unresolved path. Each captured command writes its own file under `~/.sushi/`.
- **Polling**: the poll thread waits `pollWaitMs` between polls: 1 s while a request decodes or prefills, the menu is open, a model loads, an update runs, or for 30 s after launch, Start, Stop, a port switch or Update; 5 s while the server idles; 10 s while nothing answers. Opening the menu wakes it at once and the menu shows that answer as it lands. `/v1/models` is asked each poll until a model is ready and while one loads, else every fifth poll (`modelsDue`); the process scan runs each poll until the server answers in full (offline, silent, or a 401 whose key is in its argv) or while the menu is open, else every fifth (`scanDue`); `/props` every fifth. Each request is a new connection: the server answers every one with `Connection: close`, so a kept-alive client would buy nothing.
- **API key**: the watched server's own argv (read with the log file, again only when its pid changes) gives its key: `--api-key <key>`, or `--api-key-env <VAR>` read from the server's environment, which follows argv in `KERN_PROCARGS2` (`argsApiKey`, the last flag wins as in the server). Every request to the watched port then carries `Authorization: Bearer <key>`; no other port ever gets it, and nothing is stored. With no key found a strict server still reads Key needed.
- Test seams: `SUSHIBAR_TEST_POLLS`, `SUSHIBAR_TEST_ACTIONS` ([envvar](envvar.md)); they print only what the menu shows (a hidden item is not printed), the tile labels and the spoken text included; sushibar uses the web page's words (Generation speed, Prompt speed: [server-webui](server-webui.md) copy rule), so `decode`, `prefill`, `prefix cache` and `guess` never reach the menu, and a poll line carrying one fails `tests/test_sushibar.sh`. No sushibar tooltip names a metric, so none carries a technical name.
- **PROMPT TOK/S** in the card is `(prompt_tokens - cached_tokens) / prefill_seconds` of the newest finished (`ok`) request; a request served wholly from the cache shows none.

## Threads

- Detach every per-connection `std.Thread` immediately; on teardown drain conn threads before `scheduler.deinit`.
- **Downloads run on `downloads.zig`'s own detached thread**, one pack at a time, spawning `curl` per file (on Darwin
  `std.process.spawn` is `posix_spawn`, so the resident engine is never forked); the catalog builds on another detached
  thread. Neither touches MLX. `stopServing` SIGTERMs a running curl and refuses new starts (`downloads.stop`); the partial stays to resume.
- Sleep inhibition follows the inference-thread wait.
- **The first GPU submission after about a second of idle waits 0.6-1.0 s before any work runs** (M5 Max, MiMo
  2.3bpw, ~90 GB resident, measured on e2d5be76; even a one-element op pays it, and a tick every 2 s does not prevent
  it; [perf-baselines](mimo2-perf.md#mimo-ttft-idle)). For `--gpu-warm-secs` (default 60, 0 = off) after its
  last prefill or decode tick, the parked inference thread runs one synced element-op every 500 ms (`gpuWarmTick`),
  never while work is queued; an unload closes the window. Output is unchanged.
- `Slot.deinit` runs on conn threads: it stores marks, the inference thread frees.
- **A `submit` that fails after its slot is built hands the slot to the inference thread** (`Scheduler.abandoned`, an
  intrusive list, so the handoff cannot fail for memory): its vision array is freed there, never on the conn thread.
- **A model load or unload runs only once `cleanup_queue` is empty**: a queued slot's generator points into its
  model's transformer, and the inference thread drains 16 entries per pass.
- **A slot's error is latched with a static name when the name cannot be copied** (`Slot.latchErrorLocked`):
  `error_code != null` is the terminal predicate for the consumers and the cull, so it must never be lost.
- A `submit` that fails before a slot exists (`ModelNotReady`, `GlmKvQuantUnsupported`, `Slot.init`) parks the request's
  `vision_embeddings` in `Scheduler.orphan_vision` (fixed size, no allocation); the inference thread frees it.
- A handler frees no embeddings array itself: a request refused after its media was encoded (`PreparedMedia.deinit`, the
  sub-handlers' early returns) parks it through `Scheduler.orphanVision` in the same `orphan_vision` list.
- A request's sampling state (`think_bound`, `constraint`) lives in its handler's frame: `complete` waits out any
  inference pass holding the slot (`Slot.in_pass`, taken under `queue_mu`) before the handler may free it.
  Guard: `tests/test_cancel_mid_tick.sh`.
- **A request the admission tick holds (single-flight arch, memory) logs `[queue] waiting behind N request(s); running: P+G of
  max M tokens (~S s at R tok/s)` once, and `[queue] started after X s` when it starts**; the ETA is the running request's
  remaining max, an upper bound. Visibility only: no priority, preemption or queue limit.
- **A turn that finishes `stop` on an EOS or a pad run logs `[stop] reason=eos|pad-run token=<id> "<text>" after=<id> "<text>"
  completion=<n> path=serial|spec|batch`; a request's own stop sequence logs `[stop] reason=stop-seq match="..."`** (grep
  `[stop]`). The EOS is never pushed or counted, so `Slot.stop_token` / `Generator.stop_token` (set where the stop is
  decided) are the only record of it; a length finish or a disconnect logs nothing.
- **A slot whose prefill runs is in `Scheduler.prefilling`**, neither `pending` nor `decoding`, so a SIGTERM's
  `cancelAllInFlight` stops it at the next chunk; it ran a MiMo 512k prefill on past 120 s until SIGKILL.
- **A shutdown drain is an ERROR, never a finish reason**: `cancelAllInFlight` marks `Slot.shutdown_cancelled`, so the
  reader gets `.err` (`error.ServerShutdown`, 503 "Server is shutting down") instead of `.done`, whose default
  `finish_reason` is `length` (a queued request logged `0+0 tokens [length]`).

## Request ownership and media

- **`messages.deinit(allocator)` frees the Message array and NOTHING it points at**: request media is owned by ONE
  `request_media.RequestMedia`; `Message` BORROWS. Ownership by PROVENANCE (`{slice, owned}` returns), never
  free-unless-equals-literal.
- **A media placeholder id occurs in ordinary TEXT**, so a media boundary is gated on the request CARRYING media
  (`firstMediaPlaceholder(has_media)`); media on a tower-less load is refused by NAME (`mediaRejectReason`; a streamed load that left the tower off by default names `--vision`).
- **Every message's media is decoded and placed where it was sent**: user parts, OpenAI `tool` messages, Anthropic
  `tool_result` blocks, Responses `input_image` (tool outputs too). The wire walk (`readOpenAiMessages`,
  `readAnthropicMessages`, `responses.parseInput`) records each part's offset in the joined text
  (`Message.media_parts`); the serializer hands the template a typed part list, so the TEMPLATE renders each
  placeholder; `prepareRequestMedia` expands every pad to its block's rows and keys each block for the prefix cache.
  The engine never inserts pads itself: a template that renders fewer placeholders than blocks is a named 400.
  DeepSeek-V4.1's template reads `content` as a string, so `ChatConfig.image_placeholder` renders each image's atom at
  its offset instead, and `placeMedia` expands it to the image's whole span (`dsv41_vision.spanRows`: aligner rows,
  newlines, start and end), not a merged-grid count.
- **The tower runs after the prefix restore, on the inference thread** (`encodeMediaAfterRestore`, `DeferredMedia`):
  a leading block whose every placeholder row sits inside the restored prefix is not encoded (its KV is restored),
  the encode and its bill (`visionEncodeBill`) cover the blocks after it, and a block the restore cuts through is
  encoded whole. The slot's splice offset is the placeholder rows in the prefix less the rows skipped.
- **A patch-grid image is held as its file, never as f32 patches**: the connection thread base64-decodes the payload,
  reads the size from the JPEG/PNG/GIF/BMP/WebP header and plans the grid with the processor's resize rule (`planImage`), so
  the prompt places its rows and the request keeps the compressed bytes (`ImageData.undecoded`); a 1080p screenshot is
  19.8 MB of patches on DeepSeek-V4.1. The media key is unchanged, a hash of the decoded pixels: `keyMedia` decodes
  `decode_workers` (8) images at a time on that many threads (`ImageDecodeWindow`), feeds each to the chain and frees
  it, so keying holds at most 8 decoded images, and a file that will not decode is a 400 naming its place there.
  `encodeMediaAfterRestore` decodes again, only the images it encodes, through the same window, freeing each after
  its tower call; an image its restored prefix holds is decoded once, for its key. Qwen, MiMo and GLM take the same
  path; Gemma's fixed square and `x-mlx-pixels` decode on the connection thread. The `Multimodal: encoded N of M` line gives decode and
  encode ms, and the encode counts in the request's prefill time.
- **Media refusals are named**: an undecodable image is a 400 naming `messages[i]`/`input[i]` and the reason
  (`imageRejectReason`), an `input_audio` part a 400 unless the model encodes audio (`RequestMedia.accepts_audio`),
  more than `chat.MAX_REQUEST_IMAGES` (1024) a 400 with both counts, a prompt the media pushes past the context a 400
  naming the media's tokens, an encode that does not fit a 400 (`towerFitFault`), a failed encode a 500
  (`MediaFault`); both of the last two are raised after the restore as the slot's `MediaRefused`; never a text-only answer.
- Media INPUT code: `src/vision.zig` / `src/vision_common.zig` (shared preprocessing) / `src/qwen4/qwen_vision.zig` / `src/mimo2/mimo_vision.zig` / `src/mrope.zig` (Qwen3-VL
  image/video tower, M-RoPE positions over every block; MiMo-ViT images, [mimo2-arch](mimo2-arch.md#vision));
  `src/dsv41/dsv41_vision.zig` (DeepSeek-V4.1 images, [dsv41-arch](dsv41-arch.md#image-input));
  `stb_image` (JPEG, PNG, GIF first frame, BMP) + libwebp decode image input; `lib/stb_image_impl.c` names the formats compiled in.

<a id="machine-state"></a>
## Machine state

The engine runs on the user's machine and reads no more of it than the job needs:
- aggregate readings only: system memory and CPU percent for `/metrics` and the status bar, wired bytes to wait out a
  Metal free (`status.zig`);
- its own I/O timings (expert fills);
- other processes only to protect its own files, and only by path: `sushi update` refuses while a sushi binary runs from
  the install dir, `sushi repack` while any process holds a shard open.

A reading reaches the user only as a visible log or status line ("machine under heavy load" at most). The engine
never names another program, never reads temperatures or per-process load, and never writes a reading to a file of
its own. `sushi bench` and `sushi-ubench` are the exception: box-measurement tools the user starts, they read fans,
power, temperature and GPU jobs and print them in their report.

## Config reading

- `generation_config.json` `eos_token_id` is part of the stop set (additive). Read `text_config` FIRST, then root,
  PER FIELD. A config field HF allows in two SHAPES must be read as both (`chat_template` string OR list); `.string`
  on unchecked `std.json.Value` panics.
- When an arch's reference IGNORES a config field, that field is not the truth.
- A reference probe with SYNTHETIC dtypes proves the reference's SEMANTICS, not the checkpoint; parity fixtures for
  deep stacks are dumped fp32 on CPU.
