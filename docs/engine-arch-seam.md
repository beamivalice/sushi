# Architectures: the seam, adding one, dropping one

An architecture is a folder, `src/<arch>/`, plus one line in `src/arch/registry.zig`. The engine reaches it only through
the tagged unions that registry builds, so adding an arch is a folder and a line, and dropping one is deleting both.

Every served arch is in `DROPPABLE`: `tests/test_arch_drop.sh` builds the engine without each one and runs the host
tier, so no shared file imports a folder.
The type system enforces the seam, with no source-scan guard: shared code that imports an arch folder does not compile
without it. An arch added later joins the list in the landing that adds it.

## The rules

- `src/arch/registry.zig` is the only file outside `src/<arch>/` that names an architecture: one entry per arch, marked
  `// arch:<name>`.
- `src/<arch>/arch.zig` is the arch's only public face. It declares `model_types`, `Config`, `Model`, the operations
  the dispatch files call (`init`, `deinit`, `resetCache`, `warmup`, `forward`, ...), the slot `bill` (its
  `<arch>_bill.zig`), the slot `sched` (its `<arch>_sched.zig`), and the slots `packs` and `flags`.
  The arch a model runs as is stamped on it once, by `life.init` (`Trunk.arch`); `handleOf` reads that tag and tests
  nothing else. The arch that runs a `model_type` no arch claims declares `default_arch`. The
  `packs` slot holds its `product` name and its pack `aliases` rows; the `flags` slot is its `<arch>_flags.zig`.
- **The launch face.** `src/arch/packs.zig` builds `cli.aliases` from the enabled archs' `packs` (in gate order, the
  order `sushi list` shows) and the load refusals' arch lists. `src/arch/flags.zig` reads each `flags` module, every
  part optional: `options` (the flags only that arch takes, parsed by `main.zig` through `parseOption`), `help` (their
  `--help` lines, placed before the shared flag they name), `think_help` and `think_refusal` (its `--think` words),
  `command` (a subcommand such as DeepSeek's `repack`), `refusal` (launch flags it cannot act on), the settings-ignored
  and boot lines, and `serve` / `warm` (the MLX-side hooks `arch/serve.zig` calls). `src/help.zig` assembles `--help`
  from the shared text and those parts; with every arch built the text is byte for byte the committed
  `src/fixtures/help.txt`. A flag, subcommand or refusal text is the owner's decision: moving one never changes it.
  `ExpertStreamTraits` also says whether an arch has an expert prefetcher (`prefetch`), which `--ssd-prefetch` and the
  settings-ignored line read.
- **An arch's bill terms are hooks.** `bills.zig` keeps the generic bill and asks the arch's `bill` module through
  `archHook(name, ...)`: a hook the arch omits leaves the generic term (`requestBill`, `transientReserve`,
  `neutralChunkWidth`, `unnamedCacheAsk`, `diskTokenBytes`, `memoryContext`, `cacheReserve`, `commitBytes`,
  `loadWarmup`, `slotRing`, `sessionBytes`, `stateBilled`, `retainedCheckpoints`, and the MTP-head, thought-point,
  decode-rows and FP8-scratch terms). Bill code never names an arch.
- **Shared code may name a format, never an arch**: a tool-call grammar, a template dialect, a codebook, a prefix-cache
  kind. The arch picks among them.
- A model is `Trunk` (`src/trunk.zig`: embeddings, lm head, rope tables, compiled closures, the MoE family, the
  warmup scaffold) plus the arch's own state, which lives in `src/<arch>/<arch>_state.zig` (load, `deinit`, cache
  reset). `Trunk.deinit`, `resetCache` and `warmup` are `model_handle.zig` operations that dispatch to the arch, which
  frees or resets its own state and then calls the trunk's `deinitTrunk` / `resetTrunk` / `warmupTrunk`.
  The QSA history the prefix cache and the SSD tier keep on SSM entries and checkpoints is `arch/prefix.zig`'s `aux`;
  the kernel arms shared code sizes by (the batched gather admission caps a batch's KV by, a slot's score bank, the
  env reads warmed at startup, the diagnostic MoE stand-ins) are `arch/kernels.zig`, asked by the config's arch (the
  stand-ins and env reads by every built arch). No `Trunk` field names an arch, and `trunk.zig` re-exports no arch module's names; MiMo's three
  `ForwardCtx` fields (`verify_rows`, `batch_rows`, `batch_row_offsets`) are the last to move into `arch_ctx`.
- **The trunk archs' hooks.** `initTrunk` fills the state last, through `arch_state.initState`: the config's arch's
  `State.init` gets `InitArgs` (allocator, config, weights, name prefix, stream, the trunk's owned arrays) and returns
  a streamed load's expert engine; nothing after it can fail. The lm head's argmax-only readout asks
  `arch_state.headShortlist`. The MoE family asks `arch/moe.zig`, keyed by the config's arch, for an arch's own router
  logits, its routing over a selection bias, and whether the loader widens a dense router to f32 (`moe.routerLogits`,
  `moe.biasRouting`, `moe.widensRouter` in the arch's `arch.zig`).
- **Trunk ops an arch's state answers.** A `State` may declare functions of the trunk that `arch_state.zig` runs on
  every trunk for every built arch, comptime-unrolled, each acting on that trunk's own state (empty state: nothing to
  do): `compile` (`.gates` for the offline tools, `.all` for the server), `applyHeadKvQuant`, `buildDraftRerank`,
  `draftShortlists`, `markHeadCheckpoint`, `markTablesStale` / `resetTables`, `releaseGroup` / `dropGroup`,
  `flushDeferred` / `flushDeferredGroup` / `discardDeferred`, `verifyRowsGrouped` / `forwardVerifyRowsGrouped` (one
  grouped forward over several verify rows). A `Ctx` may declare `mrope` / `setMrope` (the request's M-RoPE table,
  `arch_state.mropeOf`), `setLayerCapture` (a layer-at-a-time capture's input stream) and `logMeters`. `ForwardCtx.lazy_ids` marks ids that are still a lazy
  sample: an arch's host-side lookup on them waits for `flushDeferred`. A config fact in `Facts` replaces an arch
  predicate in shared code (`hyper_connected`: no per-layer norms, hyper-connection and PLE weights per layer).
- **An arch's own state rides two registry-built fields.** `Trunk.arch_state` and `ForwardCtx.arch_ctx`
  (`src/arch/arch_state.zig`) are structs with a field per enabled arch: the `State` and `Ctx` its `arch.zig` declares,
  void where it declares none, each default-initialised so a context built by hand needs no arch field. An arch's S3
  stage moves its `Trunk` and `ForwardCtx` fields into them; a build without the arch has none. A `State` may declare
  `init`, `ctx` (the default context's part), `runsOwnModel`, `sharesReadonly`, `specRollsBack`, `rowVerify` (the
  trunk's row-verify kernels and per-row rollback), `headCache` / `headEntry` (an MTP head riding the trunk),
  `rawEmbedding`, `headShortlist` and `clearCompiled`, which `arch_state.zig` asks generically; shared code that needs
  one arch's state goes through that arch's accessors (`glm5_state.modelOf`, `requestOf`, `dsv41_state.modelOf`,
  `mimo2_state.coarseHead`). No shared file imports an arch's folder, so a build without the arch never analyses it
  and a folder test needs no skip for it. A `prefix` module may declare
  `diskLayoutKey`, the arch's own SSD namespace key (DeepSeek names its drop list and arms there; `arch/prefix.zig` asks
  it, and the config's tag is the fallback). The request a slot owns beside its KV cache is `arch_state.SlotRequest`,
  the type of the one `Ctx.request` a built arch declares (`NoRequest` when none does). A moved field leaves `Trunk` in
  the commit that fills its `State`: two owners of one pointer release it twice.
- **Load, prefill and bill hooks an arch may declare** (every one optional; the default is stated where it is called).
  `arch.zig`: the `prefill_start_width` fact, `ubench`, `tools`. The `load` slot (`arch/load.zig`) beside its
  `weights` and `residentLoadBytes`: `drafterOff`, `stageDrafter`, `prepareDrafter`, `loadDflash`, `bindDflash`,
  `dflashRowBounds(min, max, drafter_block)`, `logLoad`, `logDflashReady`, `warmUbench`, which `model_load.zig` calls by the
  config's arch; `bindDflash` returns whether the arch takes the drafter (an error is a bind that failed), and the refusal that
  names the archs which do is `packs.dflash_targets_and`. The `prefix` module (`arch/prefix.zig`):
  `checkpointMax` (the restore points a request keeps in memory) and, for the arch whose slots keep a request beside the
  KV, `captureRequest`, `restorableAt` and the prefill grid (`requestCheckpointStride`, `requestCaptureStride`,
  `requestCaptureBackoff`, `requestChunkEnd`), which `generate.zig` reaches through `arch_state.requestOf`. The `drafter`
  module: `nativeEligible(xfm, sampling, logprobs_n)` / `reserveNative` (rounds the arch runs on its own model) and
  `inDirCacheValid` (a drafter folder the engine derived from the model directory is intact). A shared helper
  that two archs once borrowed from one is named by capability and lives in `src/` (`mlx_ops.zig`: the eager-graph `Ops`).
- **The seam guard** (`arch_state.zig`, at compile time) covers `Trunk`, `ForwardCtx` and `Slot` only, not `Generator`,
  `LoadedModel` or `ModelConfig`. A field there fails the build when it holds a registered arch's name stem (`glm`,
  `dsv`, `qwen`, `mimo`) or, as whole `_`-separated words, one of the `seam_terms` on that arch's registry line (qwen4:
  `qsa`, `gdn`, `ple`, `compiled_hc`), and is not on the allowlist; a listed field that is gone fails too. The list is
  empty: the seam is done for these three handles. A term names a concept only that arch owns, never a
  shared one (`hc_mean`).
- **The chat face.** `src/<arch>/<arch>_chat.zig` is the arch's MLX-free face for the host tier (chat templates, tool
  calls, tokenizer: no MLX, so `test-host` runs beside a server). It holds `model_types` (`arch.zig` re-exports them),
  `effort` (an editable table or fixed arms), and, where the arch has them, `pin` (a template it serves instead of the
  checkpoint's, plus its image atom) `pretokenizer` (a tokenizer.json spelling it claims and the grammar that
  splits it) and `lateSystemTemplate(allocator, tpl)` (its stock template with a late system turn rendered in place, null for any
  other revision; `chat.zig` asks `arch_chat.lateSystemTemplate`). `src/arch/chat.zig` reads the faces through `registry.Chat`; `effort.zig`, `chat.zig` and `tokenizer.zig`
  call it. A host file never imports `registry.zig`'s `Mod` or an `arch.zig`: naming one pulls its tests, and with them
  MLX, into the host binary. That is also why `registry.zig` holds no test (`registry_test.zig` does) and lists each arch's
  `mod` and `chat` as separate decls on the one marked line. A new editable effort table also adds its `effort:<family>`
  name to `effort.Family`, which settings keys and `/v1/settings` use.
- Dispatch is `inline else` over the union, one area file per concern in `src/arch/`: `life` (construction,
  teardown, cache reset, warmup), `forward`, `request`, `admission` (expert-stream traits from the config), `drafter` (the
  speculative head: [engine-mtp](engine-mtp.md#the-drafter-seam)), `media` (tower, processor, prompt rows, M-RoPE; the
  config picks the arch whose `media.claims` holds), `prefix` (the prefix cache's restore target, per-entry
  rows and disk sections: [engine-prefix-cache](engine-prefix-cache.md#arch-seam)), `sched` (what the scheduler asks of
  an arch: [server-lifecycle](server-lifecycle.md#scheduler-and-batching)), `moe` (the MoE family's routing hooks), `load` (an
  arch's own weights loader, what it keeps resident, a head allocated beside the trunk, the load-time build and spec
  warmup, the verify warm-up, the PLE placement check, the key map and norm folds of a native fused BF16 checkpoint the
  streamed loader reads), `kernels` (the kernel arms shared code sizes by), and `spec` (the config side: `Spec`, the arch of a `model_type`). No vtables, no function pointers, and no layer loop crosses the seam, so the GPU
  graphs do not change.
- **The engine holds a handle, not a trunk.** `LoadedModel.handle` is `arch/handle.zig`'s `Handle` (the registry's
  `Arch` union), and every read the engine makes of a loaded model goes through that file: `trunk(h)` (the engine state
  the generic paths run on: an arch's `host(m)` when its `Model` is not that state, else the `Model`), `destroy`,
  `stream`, `config`, `ownsModuleDecodeState`, `moduleSpecWiring`, `moduleStateSpecRollback`, `logitBias`,
  `flushImatrix`. Each is one switch on the tag; a test fixture stores `handleOf(&trunk)`.
- **Numbers and tables an arch owns are `Facts` or hooks.** The MTP numbers are `Facts` (`mtp_draft_temp`,
  `mtp_adaptive_min_kv`, `mtp_adaptive_calibrated`, `mtp_rollback_by_truncate`, `cost_layout`); the EV cost surface of a
  `MtpCostProfile` is the `evCosts(profile)` of the arch's drafter module (the enum stays in `mtp.zig`, its names are profile
  ids); a drafter folder a pack ships beside the model files is its drafter module's `shipped_subdir`
  (`dflash.isShippedSubdir`); the DFlash2 row bounds, bind and ready line are its `load` hooks. How a pack keeps its
  routed experts for the streamed loader is `Facts.expert_source` (`isExpertStreamingArch` is "not `none`") and
  `streams_rate_groups`; whether a streamed pack keeps its MTP head resident is the arch's `streamed_mtp_head`
  (`supported(layout)`, `phrase`), from which `admission.mtp_unsupported` builds the refusal text. The key that says a
  pack carries an arch's MTP head is `Facts.mtp_head_marker`.
- **Text an arch's branch prints or sends is the arch's.** The bytes never change when a line moves. An arch declares
  the load refusals it words in `load.refusals` (a decl per error name, read by `arch_load.refusalText`; the HTTP mapping in
  `generation_stream.zig` keeps a neutral fallback for a build without the arch), the 400 body of a thinking-off request its
  template cannot honour as `think_off_refusal` in its chat face, the hot-cache lines as `sched` labels
  (`prepare_commit_label`, `cancelled_entry_label`) or hooks (`logCheckpointsKept`), the `[spec]` line of a load as
  `load.drafterChosen`, its `[spec-stats]` round line as `drafter.logRoundStats`, the first-engagement line of the shared-expert fold
  as `kernels.sharedFoldEngaged` and an env switch that disarms round-cost persistence as `kernels.diagArmed`
  (`QWEN4_PROFILE_*`, read in `qwen4_profile_env.zig`). The archs a refusal names are `packs.dflash_targets_and` / `_or`, from
  each `packs.dflash_target`; the pull hint names `packs.aliases[0]`. The DFlash2 round state a `Generator` keeps for an arch's
  native round is `native_*` (`native_draft_rate`, `native_lookup`, `native_prefill_window`, `native_round`, `native_gate`).
- A trunk rule one arch keeps is a `Facts` flag on its config (`residual_from_embedding`, `moe_scores_f32`,
  `qkv_split_rows`, `bias_routed_moe`), read with `arch_config.fact`. Where a pack keeps its routed experts (key
  prefix, stamped shards, the head's layer, rate groups) is `pack_layout` (`arch_config.PackLayout`, read by
  `pack_index`'s preflight), its trailing head layers the `trailingPackLayers` hook. An arch's committed test data (configs, its chat template) live in its chat face's `fixtures` (MLX-free): a shared test
  reads the config with `arch_config.fixtureConfig(name)` (null when the build left it out) and any other decl through
  `arch_chat.Fixtures(name)` (an empty struct when it did, so the test checks `@hasDecl` and skips), and the config
  pins take each arch's `fixtures.config_pins` and `config_variants`. A helper or constant that builds a shared type
  from an arch's data (a `ModelConfig`, a `ChatConfig`) sits in a `*_test.zig` beside the shared file that uses it
  (`arch_fixtures_test.zig`, `bills_fixtures_test.zig`, `request_media_fixtures_test.zig`), never in the shared file. Its microbenches are `ubench` in its `arch.zig`, which `sushi-ubench` lists.
- `-Darchs=<a,b,...>` (default: every registered arch) builds `Tag`, `Spec` and `Arch` from the listed archs. An
  excluded arch has no prong: its `arch.zig` is not analysed. A name the registry does not register fails the build.
  `registry.ofModelType` returns no tag for a `model_type` no enabled arch claims.
- `src/arch_host.zig` lists the engine services an arch's model files may reach; `src/arch_host_stub.zig` declares the
  same names for a host that carries less, and a test keeps the two name lists equal.
- The config side lives in `src/arch/config.zig`. The parse stores the arch tag in `ModelConfig.arch`; shared code asks the config's `Facts` and never an arch tag
  (`arch_config.fact`), and an arch's own folder asks `arch_config.isOf(config, "<arch>")` (a config built by hand falls back to
  `model_type`). The facts that replace a predicate are named by the capability: `state_beside_kv`, `no_mtp_head`, `mtp_unserved`,
  `latent_kv`, `kda_state`, `long_ctx_gated`, `evicts_hot_cache`, `sliding_ring`, `dflash_by_flag`, `vision_on_streamed`,
  `sushi_quant_memory_bill`, `thinking_off_checkpoint`, `skips_mtp_weights_when_off`, `own_kv_state`, `scores_picked_keys`,
  `declared_wired_ceiling`, `own_dflash_bill` and `fills_slabs_in_place`. A predicate only one arch's folder reads (batched decode,
  the source trunk) is a function of that arch's config file. An arch declares
  `config_types` (what its parse reads, text twins included), `discovery_types`, `config_facts` and `parseConfig`,
  and may declare `parseDir`, `adoptLayout`, `samplingDefaults` and `streamedResidentSplit`. `model.served_model_types`
  lists every registered type, enabled or not (`registry.registered_model_types`, from each registry line's own
  `model_types` and `gate_rank`, its place in the gate's refusal text, so a left-out arch's files are never read); a test of one arch skips when `registry.ofModelType` says the build left it out. Shared parse code calls those, never a type name.
- **The guest line.** A registry line's `guest` says whether a host may route the arch's model types to this release
  line (false for an unsupported arch such as MiMo, which is still served). `registry.guest_model_types`, the built archs
  marked `guest` in `gate_rank` order, is the `model_types` of `sushi --guest-manifest`; a new arch decides its `guest`
  on its registry line.
- A shared file only one arch uses moves into that arch's folder. A shared file a shared engine path also reaches stays
  shared and is named by capability.

## Adding an architecture

1. `src/<arch>/` with `arch.zig` and `<arch>_chat.zig` (every declaration above; `registry_test.zig` names any that is missing), the
   forward, config parse and refusals, bills, drafter, prefix/SSD codec parts, media, kernels only it uses, its tests
   and fixtures.
2. One entry in `src/arch/registry.zig`, marked `// arch:<arch>`. Its `arch.zig` tests run with the registry's.
   The test files of its folder are listed in `src/<arch>/arch_tests.zig`, which the registry line reaches as `tests`,
   one `@import("<file>.zig"),` a line: `pub const tests = .{ ... }` for the MLX root `src/tests.zig`, and `pub const
   host_tests = .{ ... }` for the MLX-free files of `src/tests_host.zig` (leave it out when there are none). Both roots
   walk the registry's enabled archs, so no shared file names the folder; `build.zig` reads `host_tests` as text,
   hence the one-import-a-line form, and a host file sits in the arch's folder (no `/` in its import). A host file must
   stay out of `tests`. The lists are not in `arch.zig`: a test file reaches engine helpers by path, which the
   arch-slice build refuses.
3. Its packs: the `packs` rows in its `arch.zig`, the tier table and recommended launch in [usage](usage.md), the
   README pack line (the CLAUDE.md "Adding a pack" rule).
4. Its docs, prefixed `<arch>-`: at least `<arch>-arch.md`; their rows in CLAUDE.md's docs index and in
   [source-map](source-map.md); its section in [pack-format](pack-format.md); its flags and env vars in
   [flags](flags.md) and [envvar](envvar.md).
5. Its live tests in `tests/<arch>/`, their rows in `tests/CLAUDE.md`, its pack-tier env in [testsuite](testsuite.md).
6. `NOTICE` for any kernel it ports.
7. The arch joins `DROPPABLE` in `tests/test_arch_drop.sh` in the same landing.

## What an arch's `arch.zig` and slot modules offer shared code

Each is optional; shared code asks through the dispatch file named, never the arch.
- `OwnConfig` (`arch/config.zig`): the config data only that arch reads, parsed with the config (`ModelConfig.dsv41` is typed by it; an empty
  struct in a build without the arch). `releaseOwn` and `serveDefaults` let a shared test build the arch's release geometry
  (`arch_config.releaseOwn`, `releaseServeDefaults`; its config text is the chat face's `fixtures.config`, `arch_config.fixtureConfig`), and a
  test that needs it skips where the arch is not built.
- `Facts.prefill_start_width`, `.expert_pool_bytes`, `.streams_experts` and `.reads_own_source` (its weight loader checks the pack, so the
  shared EXL3 pack index check does not apply); `kvBytesPerToken`, `cacheLayoutNamespace` and `dsparkOn`, `config`-keyed through
  `arch/config.zig`. The arch's own weights and whether its pack ships its drafting head are the `load` slot's `weights` and `shipsHead`
  (`arch/load.zig`).
- `streamedResidentExtra(config)` (`arch/admission.zig`): bytes a streamed load keeps resident beside the trunk. `ExpertStreamTraits.stq_experts`
  says the routed experts are STQ rate groups.
- The `prefix` slot: `checkpointMax(global)`, `disk_checkpoint_cap`, `setDiskGrownSpec(allocator, config, tier)`, `restorePointModel(xfm)` (the
  whole request state is checkpointed as restore points and the prefill reads the next chunk's rows ahead) and `captureCheckpoint`
  (`arch/prefix.zig`).
- The `bill` slot adds `prefillChunkLoadLine(config, buf, pinned, launch)`.
- The `flags` slot's `serve` part adds `poolCapped(prev)`, `unloadModel`, `enterVision`/`leaveVision`, `plannedBorrowBytes` and
  `loadLines(config, LoadLog)` (`arch/serve.zig`).
- The `drafter` slot's `dspark_rounds` (`nextDspark`, `dsparkResumeAfterForced`, `dsparkEmitPending`, `dsparkDiag`, `stageBlock`):
  `arch/dspark.zig` binds them for the generator and the scheduler, or a stand-in that refuses in a build without such an arch. The
  generator's DSpark types are the shared `dspark_rows.zig`.
- `ubench` in `arch.zig`: the arch's microbench module, whose `benches` `ubench_main.zig` lists. It may also declare
  `sharedFold(run, stream, layouts)`, the arch's arm of `shared.shared_expert_fold` (`glm5` does).
- A `State` may declare `specRollsBack` (`arch_state.zig`): its decode state rewinds across a speculative verify.

## Build roots an arch declares

An arch folder may hold `build_roots.zon`; `build.zig` reads every `src/<folder>/build_roots.zon` at configure time, so
the shared build names no arch for these:
- `.tools`: an offline tool built in place of the server by `-D<option>=true` (`option`, `help`). The tool itself is
  listed in the arch's `arch.zig` as `pub const tools = struct { pub const @"<option>" = @import("<file>.zig"); }`, and
  `src/tool_root.zig` is the root the build compiles: Zig lets a file reach the engine by relative path only from a
  root in `src/`, so a tool in an arch folder cannot be a root itself. `glm5` declares `-Dglm-ffn-replay` and
  `-Dglm-trunk-cov`; the option name and what it builds are unchanged.
- `.benches`: the arch's microbench files, which `build.zig` holds to the no-`test`-block rule (see
  [testsuite](testsuite.md#microbenches)). `sushi-ubench` lists the `ubench` decl of each built arch's `arch.zig`; the
  bench file may also declare `sharedFold(run, stream, layouts)`, the arch's arm of `shared.shared_expert_fold`
  (`glm5` does).

Bench paths are from the repo root. Dropping the folder drops its options.
A key the file does not know, or a bench path that is missing, stops the configure.

## Dropping an architecture

`tests/test_arch_drop.sh` proves a drop compiles before anyone makes it: for each arch in `DROPPABLE`, in a temporary
worktree of HEAD, it deletes `src/<arch>/` and the `// arch:<arch>` registry lines, then runs `zig build`,
`zig build test-build` (every test binary compiles) and `zig build test-host` with `-Darchs=<the rest>` (Debug), and
`tests/test_doc_links.sh`. Deleting the folder is the real
check: with the folder present, `-Darchs` alone still resolves a stray import. No GPU, no model, no lock; it obeys the
QUIET marker like any build. Before any drop it checks that each registry entry is one marked line over an existing
folder. `DROP_TRY=<arch>` runs the same check for an arch not in the list (a stray reference fails it in seconds),
`DROP_KEEP=1` keeps the worktree.

CI runs it on every PR as a step of the build-and-test job, after the test tier, on the runner and build that job
already set up; the loop covers every arch in `DROPPABLE` (mimo2, dsv41, glm5 and qwen4 today).

`zig build arch-slice -Dslice-arch=<name>` is the quick form of the same question, for the arch's own folder: it compiles
`src/<name>/arch.zig` as a module of its own, with `build_options`, `sushi_exl3` and `arch_host` (here
`arch_host_stub.zig`) as its only imports, and walks its public declarations (`src/arch/slice_check.zig`). Zig reads a
module only inside its root folder, so every `../` import of an engine file is an error; what the folder needs of the
engine goes through `arch_host`. Nothing links or runs. Until an arch's S4 stage lands the step fails on the first stray
import (one error is shown at a time); `tests/test_arch_slice.sh` pins both outcomes on probe folders. The test lists
are in `arch_tests.zig`, outside the slice's module graph, so a test file may reach the engine by path.

A passing drop of one arch costs about 2.7 minutes on the M5 Max (cold `.zig-cache`, warm global cache,
`taskpolicy -a`, Debug): 27 s for `zig build`, 41 s for `zig build test-build`, 92 s for `zig build test-host`, 1 s for
the link check. A busy box roughly doubles it. Expect the same order on a CI runner.

A drop is one commit that removes:
- `src/<arch>/` (its fixtures go with it), its registry entry (its packs and flags go with the folder);
- what `build.zig` still names for the arch (`test_dirs`, `TestModel`) until
  those leave the shared files; its test lists leave with its `arch_tests.zig` and registry line, and `src/tests.zig` and `src/tests_host.zig`
  name no arch folder;
- its sections of `src/fixtures/model-configs/parsed.pins`: the pins test skips a case whose arch is not built, so they
  stay behind as dead entries;
- `tests/<arch>/`, its rows in `tests/CLAUDE.md`, its pack-tier env in [testsuite](testsuite.md);
- `docs/<arch>-*.md` and their rows in CLAUDE.md's docs index and [source-map](source-map.md); its rows in
  [usage](usage.md), [flags](flags.md), [envvar](envvar.md) and its [pack-format](pack-format.md) section
  (`docs/bench/` stays as history);
- the README pack lines and tables, and `NOTICE` entries for kernels ported only for it;
- arch-specific strings in the web UI.

What a DeepSeek drop leaves behind, by design: `src/expert_usage.zig` (the pack's `expert_usage.json` and the `--drop-expert` set; no arch
declares the trait, so the flag is refused at load), `src/dspark_rows.zig` (DSpark is a drafting method, like DFlash) and
`src/weights_where.zig`; `ModelConfig.dsv41` (an empty struct without the arch); and, in `src/fixtures/`, the DeepSeek chat template
and its cases, which 30-odd shared template tests embed (a drop skips them through `arch_chat.pinOf`).

It adds a CHANGELOG line ("<Arch> packs are no longer served"). The diff touches nothing else; if it must, the seam
leaked and that leak is fixed first.
