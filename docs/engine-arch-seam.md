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
  the dispatch files call (`init`, `deinit`, `resetCache`, `warmup`, `forward`, ...), and the slots `packs` and `flags`.
  The arch a model runs as is stamped on it once, by `life.init` (`Trunk.arch`); `handleOf` reads that tag and tests
  nothing else. The arch that runs a `model_type` no arch claims declares `default_arch`. The
  slots are empty: alias rows still live in `aliases` in `src/cli.zig`, and arch-only flags in `src/cli.zig`.
- **Shared code may name a format, never an arch**: a tool-call grammar, a template dialect, a codebook, a prefix-cache
  kind. The arch picks among them.
- A model is `Trunk` (`src/trunk.zig`: embeddings, lm head, rope tables, compiled closures, the MoE family, the
  warmup scaffold) plus the arch's own state, which lives in `src/<arch>/<arch>_state.zig` (load, `deinit`, cache
  reset). `Trunk.deinit`, `resetCache` and `warmup` are `model_handle.zig` operations that dispatch to the arch, which
  frees or resets its own state and then calls the trunk's `deinitTrunk` / `resetTrunk` / `warmupTrunk`.
  `Transformer` is an alias of `Trunk`, and `transformer.zig` re-exports the names that moved, until the arch stages
  move their callers. The arch handle fields (`glm5`, `dsv41`, `qwen4`, ...) stay on `Trunk` for now: callers reach
  them by name, and each arch's S3 stage moves its own.
- Dispatch is `inline else` over the union, one area file per concern in `src/arch/`: `life` (construction,
  teardown, cache reset, warmup), `forward`, `request`, `admission` (expert-stream traits from the config), `drafter` (the
  speculative head: [engine-mtp](engine-mtp.md#the-drafter-seam)), `prefix` (the prefix cache's restore target, per-entry
  rows and disk sections: [engine-prefix-cache](engine-prefix-cache.md#arch-seam)), and `spec` (the config side: `Spec`,
  the arch of a `model_type`). No vtables, no function pointers, and no layer loop crosses the seam, so the GPU
  graphs do not change.
- `-Darchs=<a,b,...>` (default: every registered arch) builds `Tag`, `Spec` and `Arch` from the listed archs. An
  excluded arch has no prong: its `arch.zig` is not analysed. A name the registry does not register fails the build.
  `registry.ofModelType` returns no tag for a `model_type` no enabled arch claims.
- `src/arch_host.zig` lists the engine services an arch's model files may reach; `src/arch_host_stub.zig` declares the
  same names for a host that carries less, and a test keeps the two name lists equal.
- A shared file only one arch uses moves into that arch's folder. A shared file a shared engine path also reaches stays
  shared and is named by capability.

## Adding an architecture

1. `src/<arch>/` with `arch.zig` (every declaration above; `registry.zig`'s test names any that is missing), the
   forward, config parse and refusals, bills, drafter, prefix/SSD codec parts, media, kernels only it uses, its tests
   and fixtures.
2. One entry in `src/arch/registry.zig`, marked `// arch:<arch>`. Its `arch.zig` tests run with the registry's.
3. Its packs: the `aliases` row in `src/cli.zig`, the tier table and recommended launch in [usage](usage.md), the
   README pack line (the CLAUDE.md "Adding a pack" rule).
4. Its docs, prefixed `<arch>-`: at least `<arch>-arch.md`; their rows in CLAUDE.md's docs index and in
   [source-map](source-map.md); its section in [pack-format](pack-format.md); its flags and env vars in
   [flags](flags.md) and [envvar](envvar.md).
5. Its live tests in `tests/<arch>/`, their rows in `tests/CLAUDE.md`, its pack-tier env in [testsuite](testsuite.md).
6. `NOTICE` for any kernel it ports.
7. The arch joins `DROPPABLE` in `tests/test_arch_drop.sh` in the same landing.

## Dropping an architecture

`tests/test_arch_drop.sh` proves a drop compiles before anyone makes it: for each arch in `DROPPABLE`, in a temporary
worktree of HEAD, it deletes `src/<arch>/` and the `// arch:<arch>` registry lines, then runs `zig build`,
`zig build test-build` (every test binary compiles) and `zig build test-host` with `-Darchs=<the rest>` (Debug), and
`tests/test_doc_links.sh`. Deleting the folder is the real
check: with the folder present, `-Darchs` alone still resolves a stray import. No GPU, no model, no lock; it obeys the
QUIET marker like any build. Before any drop it checks that each registry entry is one marked line over an existing
folder. `DROP_TRY=<arch>` runs the same check for an arch not in the list (a stray reference fails it in seconds),
`DROP_KEEP=1` keeps the worktree.

A passing drop of one arch costs about 2.7 minutes on the M5 Max (cold `.zig-cache`, warm global cache,
`taskpolicy -a`, Debug): 27 s for `zig build`, 41 s for `zig build test-build`, 92 s for `zig build test-host`, 1 s for
the link check. A busy box roughly doubles it. Expect the same order on a CI runner.

A drop is one commit that removes:
- `src/<arch>/` (its fixtures go with it), its registry entry, its `aliases` rows and arch-only flags in `src/cli.zig`;
- the arch's files from the `src/tests.zig` import list, and what `build.zig` still names for it (`test_dirs`,
  `TestModel`, the `ubench_<arch>.zig` files in `guardBenchFiles` and `ubench_main.zig`), until those leave the shared
  files;
- `tests/<arch>/`, its rows in `tests/CLAUDE.md`, its pack-tier env in [testsuite](testsuite.md);
- `docs/<arch>-*.md` and their rows in CLAUDE.md's docs index and [source-map](source-map.md); its rows in
  [usage](usage.md), [flags](flags.md), [envvar](envvar.md) and its [pack-format](pack-format.md) section
  (`docs/bench/` stays as history);
- the README pack lines and tables, and `NOTICE` entries for kernels ported only for it;
- arch-specific strings in the web UI.

It adds a CHANGELOG line ("<Arch> packs are no longer served"). The diff touches nothing else; if it must, the seam
leaked and that leak is fixed first.
