# Server: lifecycle, loading, settings and scheduling

How a model gets from a path to a serving slot and back: discovery, the arch gate, the one weight-loader decision,
load and unload on the inference thread, settings precedence, the scheduler's slots and batching, threads, and the
ownership rules that keep request data alive. Read this before touching `src/scheduler.zig`, `src/model_registry.zig`,
`src/model_settings.zig`, `src/main.zig` or `src/cli.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [server-http-apis](server-http-apis.md),
[engine-memory-admission](engine-memory-admission.md), [engine-mlx-gotchas](engine-mlx-gotchas.md).

## Entry points

- `src/main.zig`: entry, CLI flags + subcommands (`run/pull/list/serve/launch/kld/update`).
- `src/cli.zig`: alias → HF repo, resumable pull into `~/.sushi/models/<org>/<repo>`, `list`, `run` REPL.
- `pull` takes top-level files, `mtp/` and GLM's `GLM-5.3-Flash-DFlash2/` assistant (the loader auto-detects that folder; a
  pull that skips it serves GLM with DFlash2 off, logged as `[glm-dflash] off: no assistant found`).
- Headless and `--model` boots build their `ServerConfig` defaults from one `LaunchServe` (sampling, PLD, `--kv-attn-mode`,
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
  (`version.guest_model_types`, a subset of the served types) and the EXL3 codebooks and window range the decoder
  accepts. The release tarball ships it as `guest.json`, next to a `.sha256` of the tarball.

<a id="self-update"></a>
## Self-update

- **`sushi update`** (`src/update.zig`) reads `/repos/beamivalice/sushi/releases?per_page=100` and takes the highest
  SemVer carrying both assets (drafts never, prereleases by flag or by version only with `--pre`; `/releases/latest`
  is newest by date, so a 1.0.x hotfix published after 1.1.0 would be offered to 1.0.4). Never an equal or lower one.
- **Refused by name**: a source build (under `zig-out/bin`, or in a checkout holding `build.zig`), an app bundle, a
  folder without `lib/`, an unwritable install or parent, another process running from the install (`proc_pidpath`;
  `--force` skips it), and an install folder holding anything the release does not ship (the swap would carry it off).
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
  `run` banner. Boot logs `[update] daily check on|off (source)`: `--no-update-check` > `SUSHI_NO_UPDATE_CHECK`
  (`diagEnvOn`) > off for a source build > on. `SUSHI_UPDATE_API` is a test-only hook (tests/test_self_update.sh).
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
- `--think` enables thinking; `--think <word>` selects an effort the `--model` model takes (GLM low|high|max, Qwen
  off|low|medium|xhigh, MiMo off or any level as on; `/think` alike); another word fails with the accepted list and a
  nonzero exit. Omission keeps the model default, one-shot and REPL alike. A model loaded on demand takes the word where it can, else keeps its own
  default ([server-http-apis](server-http-apis.md)).
- `--fast`, MTP/KV settings, context, timeout, reasoning budgets, and sampling flags follow the serving policy, including
  model-settings precedence. One-shot sampling retains its defaults: 100 output tokens, temperature 0, top-p 1, top-k 0.
- `--stream` flushes the same reply incrementally; otherwise it buffers until success. Stdout contains only the reply,
  with reasoning in `<think>` tags; diagnostics go to stderr. HTTP/decode errors and incomplete streams exit nonzero.
- `serve`, `--serve`, `--host`, `--port`, and `--tool on` conflict with explicit prompts and fail before loading.
  Use the REPL for client-side tools. Guard: `tests/test_prompt_flags.py`; `SUSHI_PROMPT_MODEL` enables live Qwen checks.

## What loads

- A rescan makes a failed load retryable only when discovery finds the same ID at the same path; it clears the error and refreshes the on-disk byte count without disturbing live entries.
- A qwen4_exp load that fails after the n-gram state or the MTP head exists frees both (`Qwen4Mtp.deinit`; the
  state's `deinit` joins the table's warm thread first). Guard: the `QWEN4_TEST_MODEL` FailingAllocator sweep over
  `loadQwen4Mtp`.
- A streamed load that fails after its expert engine exists deinits it (`initExpertStream`, both archs): freed
  alone, its I/O workers ran on in freed memory. Guard: a FailingAllocator sweep over `initExpertStream` with the
  imatrix collector armed.

- **Bind**: `server.resolveBind` defaults to `127.0.0.1:12345`; `--host` takes an IPv4 literal, `0.0.0.0` or
  `localhost` (= 127.0.0.1; anything else is refused by name, never widened). Before any model loads,
  `ensurePortFree` probes the address with a connect AND a bind, so a listener or a bound-but-silent socket both refuse
  with "port N is already in use"; the listener binds with SO_REUSEADDR only, so a racing second sushi fails its bind
  with the same message instead of co-binding. Guard: `tests/test_port_conflict.sh`.
- The arch gate: the loader refuses any `model_type` outside `model.served_model_types` (`qwen4_exp`, `mimo_v2`, `glm5_next`) by
  name (`ArchitectureUnsupported` → 503). A checkpoint in an unsupported file format is refused by name
  (`ModelFormatUnsupported` → 503; `--model` exits).
- **The weight loader is ONE decision** (`model.loadWeightsForConfig`: streaming index > MiMo source trunk > vision >
  plain). A second site builds a model the server never serves — a MiMo pack read without its source trunk binds the
  raw FP8 fused QKV and its logits stop following the routed experts.
- **A missing tensor is a load ERROR, never `unreachable`** (`error.MissingWeight` → named 503 via
  `loadErrorFromName`); a load failure crosses the inference thread by NAME (`req.error_name`).
- Discovery (`src/model_discovery.zig` / `src/model_registry.zig`): two-level org/name, multi-root, streaming stubs,
  multi-model registry. **`--model-dir` is REPEATABLE** (`discoverModelsMany` merges roots FIRST-WINS). One path never
  registers under TWO ids (`registry.peekByPath`).
- **The auto `--max-resident-mem` (80% of the GPU working-set limit) bounds CO-RESIDENCE only**: a cold load evicts
  every other model first, and one that then loads alone is the load preflight's call (`mem_cap_binds_alone`); an
  explicit `--max-resident-mem` binds a sole model too. Before this the 2.3bpw MiMo pack could not cold-load through
  `/v1/load-model` (the app's path) on a 128 GB Mac at default flags.
- **A reload FREES the CPU state `unloadResident` retains** while the entry is `.loading` (`releaseRetainedCpuState`):
  a reader holding no refcount takes the mutex AND skips them while `.loading`.

<a id="settings"></a>
## Settings precedence

- **An explicit launch flag outranks `model-settings.json`**, which outranks the default (`model_settings.pick`;
  `--ctx-size 0` = not given). Applies to `--mtp/--no-mtp`, `--kv-quant`, `--ctx-size`, `--mtp-typical/--mtp-tokenv3`,
  `--mtp-greedy-tail`, `--ssd-budget-gb/--expert-cache-gb`, `--preserve-thinking`, `--think-penalty`, `--logit-bias-file`, `--vision/--no-vision`; a request's own field still applies on top. Design reviews reject "file beats
  flag".
- **`--mtp-min-depth` / `--mtp-max-depth` are launch-only**: the range is a property of the machine, so it has no
  `model-settings.json` key; a removed `--mtp-depth` exits naming the two
  ([engine-mtp](engine-mtp.md#depth-range)).
- **`--no-mtp-lookup` is launch-only too** (no `model-settings.json` key): it turns off the prompt-lookup drafts in MTP
  rounds (Qwen, MiMo) and GLM DFlash2's lookup chains. Standalone PLD keeps `--pld`/`--no-pld`; neither implies the other.
- **`--fast` is a flag profile, ranked between the flags and the file**: an explicit flag > `--fast` >
  `model-settings.json` > the default, per key (`model_settings.pickLaunch`; the one table is
  `model_settings.fast_preset`: MTP, typical acceptance, greedy tail, kv8). Its values report source `--fast` in the
  load lines, `/props` and the boot line `[args] fast: ...`. It asks only for what applies: an SSD-streamed load drops
  its MTP (`[mtp] off: unsupported under streaming (--fast)`, `MtpChoice.streamed`), where an explicit `--mtp` refuses (a Sushi EXL3 Qwen pack keeps its head instead).
- A flag that shapes a LOAD is retained on the Scheduler with its `*_explicit` bit (`ensureLoaded`'s cold-load
  `LoadRequest` is a SECOND site); read via `server.manualContext` / `kvCacheFor` / `mtpChoiceFor`. Each load logs its
  resolved value and source (`[kv-cache] kv8 (source); ctx N (source)`, `[mtp] on|off (source)`; `/props
  settings.mtp.source`). Guard: `tests/test_cold_load_launch_flags.sh`, `tests/test_model_settings.sh`.
- MTP's default is ON for `qwen4_exp` and `mimo_v2` (GLM's MTP stays off; source `default`; `/props settings.mtp.default_on` true); the flag and
  the file can only turn it off, or on for an SSD-streamed pack.
- `[pld] on|off (source)` and `/props settings.pld` report what a slot runs (`server.pldReport`): a module-wired arch
  (qwen4_exp) reads `off (module spec wiring)` whatever `--pld` says, since `scheduler.specInitWiring` never runs it.
- Per-model settings live in `~/.sushi/model-settings.json` (`src/model_settings.zig`: `ctx_size`, `kv_quant`,
  `mtp`, `mtp_acceptance`, `mtp_greedy_tail`, `ssd_budget_gb`, `preserve_thinking`, `think_penalty`, `logit_bias_file`, `vision`), stamped at BOTH load construction sites and resolved
  ONCE in `doLoadOnInferenceThread` (`preserve_thinking` per render, where a request can override it, and logged as
  `[chat] preserve_thinking on|off (source)` at load; `think_penalty` per request, logged as
  `[think-penalty] lambda L (source)`); read via `server.manualContext(config)` / `configuredKvQuantFor(config)`, never the raw
  server config.
- **The prefix cache's tiers are launch flags only** (no `model-settings.json` key): RAM retention is off unless
  `--prefix-cache-mem` is given (`--no-prefix-cache-ram` wins over it), and the SSD tier is on unless `--prefix-cache-disk 0`
  or `--prefix-cache-entries 0`, sized per model at load ([engine-prefix-cache](engine-prefix-cache.md#defaults)).
- A new per-model setting or launch flag follows this order, carries an `*_explicit` bit through both load sites and
  cold loads, and logs its resolved value with its source at load.
- Load-time context bills see explicit KV and MTP choices before `Scheduler.init` returns, including `--no-mtp`.

## Scheduler and batching

- `src/scheduler.zig`: slots, inference thread (sole MLX caller), queues, batching, admission, spec wiring, hot-cache
  budget revise.
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
- `--prefill-decode-share S` (flag > `SUSHI_PREFILL_DECODE_SHARE` > 0) targets the fraction of wall time given
  to existing decoders during another request's prefill. Values above 0.9 clamp; negative values and NaN refuse.
  Zero preserves one decode tick per boundary; a nonzero share adds ticks until `chunk_ns*S/(1-S)` is spent or
  decoders finish. `SUSHI_PREFILL_INTERLEAVE=0` disables both the share and its width cap.
- While decoders are live, the share caps the base prefill width at 1024 after explicit/environment chunk settings;
  normal tail merging still applies. Admission bills the original width, and adaptive widening retains its memory
  confirmation. An adaptive slot can lift the cap after the other decoders finish; a pinned slot keeps its cap.
  Hosted decode time is excluded from prefill compute time but remains part of request latency.
- **GLM prefill chunks never narrow for company** (`prefillShareCapFor`): its numerics depend on chunk boundaries, so a
  request's output must not depend on what else decodes; it yields only between whole chunks.
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
- **A plain batched tick stop-checks a slot with no pipeline state** (`batchEntryStops`) before forwarding its pending
  token: a fresh DFlash2 slot whose prefill sampled EOS finishes `stop` with nothing published, as solo does.
- **A slot the inference thread drops releases its GLM state there** (`releaseNativeState`: finish, error, cancel
  (also one landing after prefill, `postPrefillTerminal`), failed prefill), not when its connection thread completes it: an errored request's reserve once held 11 GB.
- `src/generate.zig`: generation, sampling, MTP orchestration, `StallClock`, prefill chunking, loop-stop tiers,
  `commitForcedTokens`. `src/tokenize_cache.zig`: per-LoadedModel LRU of rendered+encoded prompts.

## Threads

- Detach every per-connection `std.Thread` immediately; on teardown drain conn threads before `scheduler.deinit`.
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
  sub-handlers' early returns) parks it through `server.disposeVision` in the same `orphan_vision` list.
- A request's sampling state (`think_bound`, `constraint`) lives in its handler's frame: `complete` waits out any
  inference pass holding the slot (`Slot.in_pass`, taken under `queue_mu`) before the handler may free it.
  Guard: `tests/test_cancel_mid_tick.sh`.
- **A slot whose prefill runs is in `Scheduler.prefilling`**, neither `pending` nor `decoding`, so a SIGTERM's
  `cancelAllInFlight` stops it at the next chunk; it ran a MiMo 512k prefill on past 120 s until SIGKILL.

## Request ownership and media

- **`messages.deinit(allocator)` frees the Message array and NOTHING it points at**: request media is owned by ONE
  `server.RequestMedia`; `Message` BORROWS. Ownership by PROVENANCE (`{slice, owned}` returns), never
  free-unless-equals-literal.
- **A media placeholder id occurs in ordinary TEXT**, so a media boundary is gated on the request CARRYING media
  (`firstMediaPlaceholder(has_media)`); media on a tower-less load is refused by NAME (`mediaRejectReason`; a streamed load without `--vision` names the flag).
- **Every message's media is decoded and placed where it was sent**: user parts, OpenAI `tool` messages, Anthropic
  `tool_result` blocks, Responses `input_image` (tool outputs too). The wire walk (`readOpenAiMessages`,
  `readAnthropicMessages`, `responses.parseInput`) records each part's offset in the joined text
  (`Message.media_parts`); the serializer hands the template a typed part list, so the TEMPLATE renders each
  placeholder; `prepareRequestMedia` expands every pad to its block's rows and encodes all blocks in prompt order.
  The engine never inserts pads itself: a template that renders fewer placeholders than blocks is a named 400.
- **Media refusals are named**: an undecodable image is a 400 naming `messages[i]`/`input[i]` and the reason
  (`imageRejectReason`), an `input_audio` part a 400 unless the model encodes audio (`RequestMedia.accepts_audio`),
  more than `chat.MAX_REQUEST_IMAGES` (64) a 400 with both counts, a prompt the media pushes past the context a 400
  naming the media's tokens, an encode that does not fit a 400 (`towerFitFault`), a failed encode a 500
  (`MediaFault`); never a text-only answer.
- Media INPUT code: `src/vision.zig` / `src/vision_common.zig` (shared preprocessing) / `src/qwen_vision.zig` / `src/mimo_vision.zig` / `src/mrope.zig` (Qwen3-VL
  image/video tower, M-RoPE positions over every block; MiMo-ViT images, [mimo2-arch](mimo2-arch.md#vision));
  `stb_image` + libwebp decode image input.

## Config reading

- `generation_config.json` `eos_token_id` is part of the stop set (additive). Read `text_config` FIRST, then root,
  PER FIELD. A config field HF allows in two SHAPES must be read as both (`chat_template` string OR list); `.string`
  on unchecked `std.json.Value` panics.
- When an arch's reference IGNORES a config field, that field is not the truth.
- A reference probe with SYNTHETIC dtypes proves the reference's SEMANTICS, not the checkpoint; parity fixtures for
  deep stacks are dumped fp32 on CPU.
