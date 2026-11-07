# Architectures: the seam, adding one, dropping one

An architecture is a folder, `src/<arch>/`, plus one line in `src/arch/registry.zig`. The engine reaches it only through
the tagged unions that registry builds, so adding an arch is a folder and a line, and dropping one is deleting both.

Status: this is the target. Today the dispatch files, the registry and `-Darchs` exist, but shared files still name
archs and still import the folders, and a `-Darchs` build still links the other archs' serving code. The stray
references `tests/test_arch_drop.sh` reports (run it with `DROP_TRY=<arch>`) are removed one stage at a time, and an arch
joins `DROPPABLE` when its drop passes.

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
  `Transformer` is an alias of `Trunk`, and `transformer.zig` re-exports the names that moved, until the arch stages
  move their callers. The arch handle fields still on `Trunk` (`dsv41`, `qwen4`, ...) are reached by name until
  each arch's S3 stage moves its own; GLM's have moved.
- **An arch's own state rides two registry-built fields.** `Trunk.arch_state` and `ForwardCtx.arch_ctx`
  (`src/arch/arch_state.zig`) are structs with a field per enabled arch: the `State` and `Ctx` its `arch.zig`
  declares, void where it declares none, each default-initialised so a context built by hand needs no arch field.
  An arch's S3 stage moves its `Trunk` and `ForwardCtx` fields into them; a build without the arch has none. A seam guard in
  `arch_state.zig` fails the build when a `Trunk`, `ForwardCtx` or `Slot` field names a registered arch beyond its
  allowlist, or an allowlisted field is gone; each S3 stage deletes its lines, and an empty list means the seam is done. A `State` may declare `ctx` (the default context's part), `runsOwnModel`,
  `sharesReadonly` and `rawEmbedding`, which `arch_state.zig` asks generically; shared code that needs one arch's state
  goes through that arch's accessors (`glm5_state.modelOf`, `requestOf`), which are null in a build without it. The
  request a slot owns beside its KV cache is `arch_state.SlotRequest`, the type of the one `Ctx.request` a built arch
  declares (`NoRequest` when none does). A moved field leaves `Trunk` in the commit that fills its `State`: two
  owners of one pointer release it twice.
- **The chat face.** `src/<arch>/<arch>_chat.zig` is the arch's MLX-free face for the host tier (chat templates, tool
  calls, tokenizer: no MLX, so `test-host` runs beside a server). It holds `model_types` (`arch.zig` re-exports them),
  `effort` (an editable table or fixed arms), and, where the arch has them, `pin` (a template it serves instead of the
  checkpoint's, plus its image atom) and `pretokenizer` (a tokenizer.json spelling it claims and the grammar that
  splits it). `src/arch/chat.zig` reads the faces through `registry.Chat`; `effort.zig`, `chat.zig` and `tokenizer.zig`
  call it. A host file never imports `registry.zig`'s `Mod` or an `arch.zig`: naming one pulls its tests, and with them
  MLX, into the host binary. That is also why `registry.zig` holds no test (`registry_test.zig` does) and lists each arch's
  `mod` and `chat` as separate decls on the one marked line. A new editable effort table also adds its `effort:<family>`
  name to `effort.Family`, which settings keys and `/v1/settings` use.
- Dispatch is `inline else` over the union, one area file per concern in `src/arch/`: `life` (construction,
  teardown, cache reset, warmup), `forward`, `request`, `admission` (expert-stream traits from the config), `drafter` (the
  speculative head: [engine-mtp](engine-mtp.md#the-drafter-seam)), `media` (tower, processor, prompt rows, M-RoPE; the
  config picks the arch whose `media.claims` holds), `prefix` (the prefix cache's restore target, per-entry
  rows and disk sections: [engine-prefix-cache](engine-prefix-cache.md#arch-seam)), `sched` (what the scheduler asks of
  an arch: [server-lifecycle](server-lifecycle.md#scheduler-and-batching)), and `spec` (the config side: `Spec`,
  the arch of a `model_type`). No vtables, no function pointers, and no layer loop crosses the seam, so the GPU
  graphs do not change.
- `-Darchs=<a,b,...>` (default: every registered arch) builds `Tag`, `Spec` and `Arch` from the listed archs. An
  excluded arch has no prong: its `arch.zig` is not analysed. A name the registry does not register fails the build.
  `registry.ofModelType` returns no tag for a `model_type` no enabled arch claims.
- `src/arch_host.zig` lists the engine services an arch's model files may reach; `src/arch_host_stub.zig` declares the
  same names for a host that carries less, and a test keeps the two name lists equal.
- The config side lives in `src/arch/config.zig`. The parse stores the arch tag in `ModelConfig.arch`, and `isQwen4`,
  `isGlm5`, `isDsv41` and `isMimo` read it (a config built by hand falls back to `model_type`). An arch declares
  `config_types` (what its parse reads, text twins included), `discovery_types`, `config_facts` and `parseConfig`,
  and may declare `parseDir`, `adoptLayout`, `samplingDefaults` and `streamedResidentSplit`. `model.served_model_types`
  lists every registered type, enabled or not (`registry.registered_model_types`, from each registry line's own
  `model_types` and `gate_rank`, its place in the gate's refusal text, so a left-out arch's files are never read); a test of one arch skips when `registry.ofModelType` says the build left it out. Shared parse code calls those, never a type name.
- A shared file only one arch uses moves into that arch's folder. A shared file a shared engine path also reaches stays
  shared and is named by capability.

## Adding an architecture

1. `src/<arch>/` with `arch.zig` and `<arch>_chat.zig` (every declaration above; `registry_test.zig` names any that is missing), the
   forward, config parse and refusals, bills, drafter, prefix/SSD codec parts, media, kernels only it uses, its tests
   and fixtures.
2. One entry in `src/arch/registry.zig`, marked `// arch:<arch>`. Its `arch.zig` tests run with the registry's.
3. Its packs: the `packs` rows in its `arch.zig`, the tier table and recommended launch in [usage](usage.md), the
   README pack line (the CLAUDE.md "Adding a pack" rule).
4. Its docs, prefixed `<arch>-`: at least `<arch>-arch.md`; their rows in CLAUDE.md's docs index and in
   [source-map](source-map.md); its section in [pack-format](pack-format.md); its flags and env vars in
   [flags](flags.md) and [envvar](envvar.md).
5. Its live tests in `tests/<arch>/`, their rows in `tests/CLAUDE.md`, its pack-tier env in [testsuite](testsuite.md).
6. `NOTICE` for any kernel it ports.
7. The arch joins `DROPPABLE` in `tests/test_arch_drop.sh` in the same landing.

## Build roots an arch declares

An arch folder may hold `build_roots.zon`; `build.zig` reads every `src/<folder>/build_roots.zon` at configure time, so
the shared build names no arch for these:
- `.tools`: an offline tool built in place of the server by `-D<option>=true` (`option`, `help`, `root`). `glm5` declares
  `-Dglm-ffn-replay` and `-Dglm-trunk-cov`; the option name and what it builds are unchanged.
- `.benches`: the arch's microbench files, which `build.zig` holds to the no-`test`-block rule (see
  [testsuite](testsuite.md#microbenches)). The files are still imported by `ubench_main.zig`, until the arch's S4 stage.

Paths are from the repo root. Moving a root file edits its folder's list; dropping the folder drops its options.
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
already set up; the loop covers every arch in `DROPPABLE` and passes at once while the list is empty.

`zig build arch-slice -Dslice-arch=<name>` is the quick form of the same question, for the arch's own folder: it compiles
`src/<name>/arch.zig` as a module of its own, with `build_options`, `sushi_exl3` and `arch_host` (here
`arch_host_stub.zig`) as its only imports, and walks its public declarations (`src/arch/slice_check.zig`). Zig reads a
module only inside its root folder, so every `../` import of an engine file is an error; what the folder needs of the
engine goes through `arch_host`. Nothing links or runs. Until an arch's S4 stage lands the step fails on the first stray
import (one error is shown at a time); `tests/test_arch_slice.sh` pins both outcomes on probe folders.

A passing drop of one arch costs about 2.7 minutes on the M5 Max (cold `.zig-cache`, warm global cache,
`taskpolicy -a`, Debug): 27 s for `zig build`, 41 s for `zig build test-build`, 92 s for `zig build test-host`, 1 s for
the link check. A busy box roughly doubles it. Expect the same order on a CI runner.

A drop is one commit that removes:
- `src/<arch>/` (its fixtures go with it), its registry entry (its packs and flags go with the folder);
- the arch's files from the `src/tests.zig` import list, and what `build.zig` still names for it (`test_dirs`,
  `TestModel`, the bench imports of `ubench_main.zig`), until those leave the shared files;
- `tests/<arch>/`, its rows in `tests/CLAUDE.md`, its pack-tier env in [testsuite](testsuite.md);
- `docs/<arch>-*.md` and their rows in CLAUDE.md's docs index and [source-map](source-map.md); its rows in
  [usage](usage.md), [flags](flags.md), [envvar](envvar.md) and its [pack-format](pack-format.md) section
  (`docs/bench/` stays as history);
- the README pack lines and tables, and `NOTICE` entries for kernels ported only for it;
- arch-specific strings in the web UI.

It adds a CHANGELOG line ("<Arch> packs are no longer served"). The diff touches nothing else; if it must, the seam
leaked and that leak is fixed first.
