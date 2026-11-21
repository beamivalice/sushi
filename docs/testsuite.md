# Test suite

A test earns its place by catching a failure nothing else catches, on code a served model reaches. Anything else is
cut: there is no "might be useful" tier. The supported models are Flash-Next (`qwen4`), GLM (`glm5`) and DeepSeek
(`dsv41`). MiMo (`mimo2`) still loads and keeps its tests, but is unsupported from 1.5.1: no release gate runs its
pass. MiniCPM5 and Gemma tool-call parsing stay because quant work and agents use them. The TDD order and the change
classes are in [CLAUDE.md](../CLAUDE.md#testing--tdd-is-mandatory); the script matrix is in
[tests/CLAUDE.md](../tests/CLAUDE.md).

## What a test may be

Every test has exactly one of these purposes. A test that fits none is not written, and one found later is deleted.

| purpose | proves | example |
|---|---|---|
| contract | a public behaviour: an HTTP route, a CLI flag, a `pub` API, a refusal | a GGUF path is refused by name |
| exactness | two arms give the same bytes: kernel vs oracle, stream vs non-stream, batched vs solo, warm vs cold | grouped MTP == serial |
| bill | the memory bill matches what the code stores | DeepSeek trunk bill == loaded bytes |
| regression pin | a real past bug, red before the fix | its name says what failed |
| invariant | a property over a corpus of real inputs | format corpus, tool-traffic replay |

## What a test may not be

- A source scan (`@embedFile` + "this text appears"). Test the behaviour, or state the rule in a comment.
  A lint rule ([Lint](#lint)) is a build step over the AST, not a test and not an `@embedFile` scan.
- A struct literal read back, a compile-shape check, a "does not crash" or "is finite" smoke with no oracle.
- A wall-clock assertion. Timing is a microbench in `sushi-ubench` ([process-ubench](process-ubench.md)); a test that
  timed something keeps its exactness half.
- A test of code no served path reaches. Dead code goes with its tests in the same commit.
- An assertion that a model must think, answer or call a tool. Assert the invariant, branch on the model's choice.
- A second copy of a check another test already makes on the same input with as strong an assertion.
- A parity test between two engines (Sushi Cooler against MLX, a kernel against its replaced kernel) once the new
  one is proven: it proves a fact, then repeats it on every run and catches nothing. It goes in the landing that
  proves it, with its numbers in the commit; the oracle tests against a reference model stay.
- A repetition: the same check over fixtures, prompt lengths, arms or sweep points that run the same code path.
  One input per distinct path is the test; more catch no new error. A fault sweep refuses one point per distinct
  error path, not every site at every ask.
- A read of a field the test never set. A `= undefined` struct is garbage in ReleaseFast, and a field write into its
  nested config (`xfm.config.model_type = ...`) leaves `config.arch.tag`, which the arch dispatch reads, garbage too.
  A hand-built Trunk comes from `test_support.testTrunk(model_type)` (config and arch state defined), and the
  `undefined-nested-write` lint rule ([Lint](#lint)) refuses a nested write into an undefined local with no whole
  assignment first.

## Tiers

| tier | means | where it runs |
|---|---|---|
| host | touches no MLX array; its file reaches no `mlx.zig` | `zig build test-host`: anywhere, beside a live server, no GPU lock |
| gpu | Metal kernels on small synthetic tensors; the first line is `try mlx.requireGpu();` | `zig build test` |
| pack | needs a fixture or pack named by an env var; skips when it is unset | `zig build test` with the env set |

- `zig build test-host` builds `src/tests_host.zig`, a root that links no MLX: a file in it that reaches MLX fails the
  link. The build reads that root's imports, and each arch folder's `pub const host_tests` in its `arch_tests.zig`, as the
  host list, and no MLX domain may also select a host file's tests. An arch's other test files are its `pub const
  tests` ([placement](engine-arch-seam.md#adding-an-architecture)): `src/tests.zig` walks the enabled archs' lists, so
  a test file is added to its arch's `arch_tests.zig`, never to a root.
  The host binary is filtered to the tests of those files (`hostFilters` in build.zig); its module is built with
  `build_options.host_tier`, under which a live GPU read (`gpu_mem.zig`) panics and the NAX probe answers false.
  Every `<arch>/<arch>_bill.zig` is a host file too, by its `_bill` stem; `bills.zig` reaches the enabled archs' through the registry.
  A bill test that needs the live GPU lives in `bills_live_test.zig` (or `<arch>_bill_test.zig`) in the MLX tier.
- `zig build test` runs the host tier, the MLX domains, the exl3 binaries and the encoder binary, `test-encoder`.
  **The MLX domains** (`test_domains` in build.zig) are one binary each, named by what its tests prove:

  | binary | holds |
  |---|---|
  | `test-cooler` | the Sushi Cooler contract: a run step per family file (`cooler-load`, `-pool`, `-prefill-bill`, `-requests`, `-restore`), one per family's fault sweeps (`cooler-load-fault`, `-requests-fault`), two for the cells on held loads (`cooler-held-requests`, `-held-restore`) and one per late cell |
  | `test-deepseek`, `test-qwen`, `test-glm`, `test-mimo` | `src/dsv41/`, `src/qwen4/`, `src/glm5/`, `src/mimo2/`, less their prefix-cache files |
  | `test-kernels` | `src/tensor/` and the shared kernels (attention, verify, MoE, FP8, MTP, DFlash, MLX and Metal glue) |
  | `test-generate` | generation, the scheduler, MTP and DFlash rounds, sampling, the KV and SSM caches, the trunk |
  | `test-load` | `src/arch/`, the model and its loader, registry, discovery, owned weights, expert I/O, imatrix, KLD, live bills |
  | `test-server` | the HTTP server and its APIs, the CLI, launch and main |
  | `test-prefix-cache` | the prefix cache and its SSD tier, run twice: `test-prefix-cache` and, on a run with no `-Dtest-filter`, `test-prefix-cache (SUSHI_PREFIX_FS=real)` |

  A domain holds the test files of its folders and the `src/` files it lists (a listed file leaves its folder's
  domain); a test file that no domain holds, or that two list, fails the configure by name, so a new shared file is
  placed in a domain in the landing that adds it. A domain's binary runs as one or more run steps, each a process with
  its own `SUSHI_TEST_SELECT` (the prefix `<path>.` of each of its files), and the configure fails a test that the steps
  would run other than once.
  The exl3 tests run as one binary per area of `exl3_test_bins` (build.zig): `test-exl3-experts` (the expert
  kernels, rate groups, windows, prefill grid), `test-exl3-trunk` (the trellis linears) and `test-exl3-format`
  (root, device, kernel source, codebook tables, side cache, layout, pack). A file with tests that the table does not
  list, lists twice or whose filter selects a test of another binary is refused at configure, by name.
  `test-encoder` builds only when the run's `-Dtest-filter` selects one of its tests (or there is none).
  `test-sushibar` runs `src/sushibar_main.zig`'s tests (AppKit views read back, no MLX) under the same rule.
- **A binary that runs no test fails.** Zig reports a test binary whose filters select nothing as a pass, so each MLX
  domain, each exl3 binary, the encoder binary and the unfiltered host binary is followed by
  `scripts/test_count_guard.zig`, which asks the runner for the binary's test list (nothing runs) and fails, naming the
  binary (`test-cooler`, `test-exl3-trunk`), when no
  named test is in it. The root file's anonymous `test { ... }` does not count: it survives every filter. A run with a
  `-Dtest-filter` skips the host check, since a filter may select nothing there. `tests/test_test_count_guard.sh` pins it.
- build.zig names a domain's files at configure time, so every folder or file it scans is declared with
  `b.dependOnDirectoryContents`/`dependOnFileContents`; without that a new test file never runs until the cache goes.
- A filtered run keeps the parallel binaries: the scan keeps the tests whose full name holds a `-Dtest-filter` substring,
  in their domains' binaries and steps (a step with none is not made), and the host tier runs once, in `test-host`.
  A filter may hold any character; a test name with a quote is selected by its longest quote-free piece that no
  skipped test holds (the response file the compiler reads cannot carry a quote), and a name with none is renamed.
- `SUSHI_TEST_NO_GPU=1` makes a test build see no Metal backend, so `requireGpu` skips (diagnostic, test builds only).
- **Tests run under sushi's runner** (`scripts/test_runner.zig`, std's with one check): after each test it asks
  `sushi_after_each` (`src/test_rest.zig`), which fails that test by name when it left a hold, a turn, a model's budget
  slot, a forward scope, a seal, an armed fault, an opt-out or a latched MLX error, then puts the engine back at rest.
  `zig build test -Dtest-log=<abs path>` gives every test run step `SUSHI_TEST_LOG` and its name as `SUSHI_TEST_STEP`
  (the second tier run: `test-prefix-cache (SUSHI_PREFIX_FS=real)`); each runner appends, under the file's lock,
  `<UTC>\t<step>\t<PASS|SKIP|FAIL>\t<ms>\t<test name>` per test and `<UTC>\t<step>\tDONE\t<ms>\t<p> pass, <s> skip,
  <f> fail` when its process ends, and prints nothing. A leak or a logged error is a FAIL, as the build runner counts
  it; a crashed test has no line (the build runner's error names it) and the restarted process ends on a DONE of its own.
  The slow tests: `sort -t$'\t' -k4,4rn <file> | head`.
  `SUSHI_TEST_SELECT` (one name prefix a line, `!<prefix>` drops) makes the runner run only the tests it selects, so
  several run steps can share one binary; one that selects nothing fails, naming its step.
- Every test compile and run step declares its peak RSS (`max_rss`: a domain's `compile_gib` and `run_gib` in
  build.zig, `exl3_test_bins` alike), so `zig build test --maxrss <bytes>` runs only the steps that fit at once. A
  compile's is its measured cold MaxRSS; a run's is generous, since RSS does not see Metal. A step that can load the
  Qwen cut (`test-cooler`, `test-qwen`, `test-exl3-trunk`) declares 16 GiB under `-Dfixture-cut=true`, which a run that
  sets `QWEN4_TEST_MODEL` passes (`tests/gate.sh` does): without it such a step fails before its first test, naming
  the flag (`SUSHI_TEST_FIXTURE_CUT`). Without `--maxrss` the budget is the machine's RAM (`hw.memsize`); a box under
  the largest bound refuses the build by name.
- **The Sushi Cooler contract** (`src/cooler_contract_test.zig`, its runner and helpers, and its cells in one file per
  family, `cooler_<family>_test.zig`, each a run step of `test-cooler`; tests `cooler contract:`) runs on the tiny fixtures of
  every arch that declares `cooler_contract`: the side, every weight in an owned region, a request and the model's
  teardown asking MLX nothing, an exact speculative request saying its greedy tokens, fault sweeps over the load and a
  request, nothing compiled after it, and a load of more than one row (`coolerRows`) giving each row its solo request's
  tokens (the arch's `greedyRows`; skipped at one row). It holds no cell against the MLX load: Sushi Cooler is proven.
  Requests run in the model's turn and end as serve ends them; unloads run serve's teardown. A contract load makes its
  trunk as serve does (`model_load.initTrunk`), so the config a cell bills from carries what the load recorded past
  the parse (DeepSeek's wide arms). An arch on Sushi
  Cooler is admitted by declaring it. A cell runs on the fixtures whose paths differ for its property: every fixture
  (`eachFixture`: a kernel, a bill or a late compile, which a geometry changes), one per arch and expert layout
  (`eachLayout`, residency and tile order: the loader, its cleanup, the calls a request makes), one per arch and
  residency (`eachResidency`: pool and memory), or one per arch (`eachArch`, the first set: generator and engine
  logic); a late cell long enough to split is two tests, `eachTinyFixture` (every fixture but the cut) and
  `onTheCut`, each a step of its own; with any of its fixtures unset a cell skips before loading anything. `SUSHI_CONTRACT_FIXTURE=QWEN4_STQ_TINY` (comma-separated envs) runs only
  the named fixtures, and a cell that ran on them passes: the way to check one cell on one fixture.
  A fault sweep refuses each pool-buffer or checked-call site once, at its first ask, and not again on another arm or
  fixture of the same arch (`firstAsks`); a sweep of one step (an image splice, an SSD restore, a rows tick) refuses only
  the sites that step adds to a plain request's. A speculative round's sweep tells sites apart over eight frames: at
  the default four, a verify forward asks at a prefill chunk's sites, so the plain request's set takes them all. A load sweep refuses each step of the load at its first and last
  call, and each step of its warm at its first. A checked call is one that makes a result on a stream: a call on the
  stream handle itself is not (a fault there skipped its free), and a test-only reading made on the way to the call
  under test (the TF32 probe) goes past the fault (`mlx.past_fault`), so a test passes alone as in the suite.
  A cell that names a tick unit builds it and checks it (`tickClass`): a request joins another's unit by the scheduler's rule (DeepSeek's drafting request rides the joined rows from three slots), so a request meant to tick alone is made unjoinable, not left to a slot count.
  Its no-late-kernel tests prove their property only on a fresh pipeline cache, which is the process's: they run alone.
- **A bill test pins every probed arm its bill reads** (`qsaScoreFusedOffGuard` and its kin): a kernel trial runs once a
  process, so an unpinned bill reads whatever arm an earlier test's forward tried.
- **A part of a test that did not run is said, never silent**: a cell that runs once per arch leaves out an arch whose
  fixture is missing and calls `test_note.notePartSkipped` naming it; the runner writes it as the test's log line's
  sixth field (`parts skipped: ...`) and tests/gate.sh's summary lists every such test, so a box's run says what it
  did not cover. The test skips only when no part of it ran.
- **A step declares the RSS it reaches** (`run_gib`, `cut_run_gib`, `holds_cut`, `alone_run_gib` in build.zig): zig runs
  as many steps at once as their declared bounds fit in `--maxrss` (the gate's budget), so a bound far past the
  measured peak keeps steps waiting. A step whose cells never load the Qwen cut says so (`.holds_cut = false`); its
  runner then refuses QWEN4_TEST_MODEL. The longest step bounds the run: split a family's fault sweeps out first.
- **A test whose arch this build leaves out skips by name, never runs**: a shared test that reads an arch's config or
  fixture starts with `test_note.needArchs(&.{...})`; under `-Darchs` it skips with `<arch>: not in this build
  (-Darchs)` on its log line, and an arch's own steps are not made. Check: `zig build test -Darchs=<one>` with no
  fixture envs fails nothing.
- **Held loads** (`heldLoad` in `cooler_contract_test.zig`): a request-level contract cell that proves nothing about
  a load asks for the step's served load of its fixture instead of making one; such cells sit in the
  `cooler_held_*_test.zig` files, run steps of their own, so no cell that makes its own load shares their process.
  After every test the runner's at-rest hook puts each held model between requests and fails the test that left it a
  changed config, other weights' mappings or a dispatch open; the step's end unloads them and asks for an empty pool.
  Pool bytes in use are not read: a model keeps caches its longest request grew. A cell that measures the pool or
  makes a speculative request's first-use state keeps its own load.
- **Tests that run alone** (`alone_tests` in build.zig): each is a run step of its own, a fresh process, left out of its
  file's step. Every test that proves nothing compiles after a load's warm or seal is one (the contract's late cells,
  DeepSeek's, Qwen's GDN, QSA and vision warms, the engine route's first uses): an earlier test's compiles in the same
  process would hide a late one. A new such test joins the table in its landing; an entry must name exactly one test.
- **Another GPU's arms** are tested by running the binaries as that GPU: `MLX_METAL_GPU_ARCH=applegpu_g13s` makes MLX, the device
  profile and the engine dispatch as an M1 Max (`tests/dsv41/test_g13_rehearsal.sh`). A test about one arm's bytes names it
  (`device.current().has(.matrix_g17)`), never a chip.
- The pack tier's envs: `DSV41_TINY` (the default-geometry dump WITH DSpark stages, `${SUSHI_MODELS_DIR}/tiny-models/dsv41-dspark-tiny`), `DSV41_STQ_TINY`, `DSV41_STQ_TINY_PM` (its pair-major copy), `DSV41_EPI_TINY` (tiny DeepSeek fixtures), `DSV41_INDEX_SCANS` (a
  folder of `L<layer>-pos<n>.safetensors` one-row index scans saved from a live decode: `q`, `w`, `codes`, `scales`, `meta`
  `{p0, ratio, keys, group}`, `wscale` and the kernel's `scores`; the test skips without it), `QWEN4_TEST_MODEL`,
  `MIMO_V2_MODEL`, `SUSHI_MODELS_DIR`, `SUSHI_DSV41_REAL_PACK` (the DeepSeek width probe: every row of a verify is its one-row round's bits on the real pack;
  its `SUSHI_DSV41_PROBE_*` knobs are in [envvar](envvar.md); a ~40 GB-budget run, GPU lock). The tiny models live in `${SUSHI_MODELS_DIR}/tiny-models/` (the owner's Hugging Face dataset keeps them under
  `test-fixtures/`, so a fetch lands that folder's contents there; `fixture_envs.sh` refuses a box that still has the
  old `test-fixtures/` folder, naming the `mv`). A landing on main sets the tiny-fixture envs (`source tests/lib/fixture_envs.sh`); the real-pack envs run with the release gate. Without them a green
  suite proves nothing there.
- Three real-weight runs have no script. The encoder's real-checkpoint capture:
  `SUSHI_ENCODER_REAL_MODEL=1 SUSHI_MODELS_DIR=<root holding Qwen/Qwen3.8-Flash-Next> zig build test -Doptimize=ReleaseFast -Dmodel=enc`.
  MTP verify exactness on a real pack: `QWEN4_TEST_MODEL=<pack> zig build test -Doptimize=ReleaseFast -Dfixture-cut=true -Dmodel=qwen4` with
  `-Dtest-filter=` `qwen4 checkpoint affine8 and dense projections`, `qwen4 verify history row and cache identity`,
  `qwen4 MTP verify token values` and `qwen4 deferred PLE: the captured hidden`.
  The decoded trunk GEMM on the pack's own K6 banks (GPU lock): `QWEN4_TEST_MODEL=<pack> zig build test
  -Doptimize=ReleaseFast -Dfixture-cut=true -Dtest-filter="trunk gemm: real pack trunk banks"`.
- Model scope is `qwen4`, `mimo2`, `glm5`, `dsv41`, `enc` (the encoder) or `shared` (every model). A test's folder says
  its model; a test of one model's code lives in that model's folder ([map and guards](#folder-map-and-scope-guards)).

## Folder map and scope guards

A test's model is its file's folder. Counts at the time of writing; `python3 tests/suite_guards.py --map` reprints the
table. A test's tier is read from its body: a `tests_host.zig` import or an `arch_tests.zig` `host_tests` entry is host, a body that reads an env var or skips
(`SkipZigTest`, `getenv`, `packPath`) is pack, `requireGpu` is gpu, the rest is cpu (an MLX binary, no Metal gate);
a test that skips only inside a helper counts as gpu or cpu.

| folder | model | test files | host | gpu | pack | cpu | tests |
|---|---|---|---|---|---|---|---|
| `src/` | shared (every model) | 134 | 1068 | 149 | 70 | 1638 | 2925 |
| `src/exl3/` | shared (the sushi_exl3 module) | 15 | 0 | 31 | 91 | 143 | 265 |
| `src/encoder/` | shared (the encoder, `-Dmodel=enc`) | 17 | 0 | 23 | 4 | 51 | 78 |
| `src/qwen4/` | qwen4 | 21 | 0 | 132 | 72 | 206 | 410 |
| `src/mimo2/` | mimo2 | 10 | 0 | 13 | 20 | 67 | 100 |
| `src/glm5/` | glm5 | 50 | 9 | 159 | 27 | 126 | 321 |
| `src/dsv41/` | dsv41 | 23 | 22 | 148 | 40 | 112 | 322 |
| total | | 270 | 1099 | 655 | 324 | 2343 | 4421 |

A hermetic guard keeps the folders honest. It is checked by `python3 tests/test_suite_guards.py` (the hermetic guards run it) and
reads an allowlist that `python3 tests/suite_guards.py --write` regenerates; run it after every landing that moves code,
and commit the allowlist diff with it. `python3 tests/suite_guards.py` alone checks.

- **Arch-test guard.** A test whose name mentions a model (`qwen`, `mimo`, `glm`, `dsv41`, `deepseek`) must sit in a
  folder of a model it names, or its name is in `tests/guards/arch-tests.allow`: shared code tested at that model's
  geometry. The list keys the test name, so moving a test between shared files changes nothing. A stale entry (the
  test moved into its folder, or renamed) only prints a note; `--write` drops it, and a renamed test needs the same
  `--write` to list its new name. To clear a failure, move the test into
  `src/<model>/`, or rename it for the shared code it tests.
- **Shell guard.** `python3 tests/shell_guards.py` (checked by `python3 tests/test_shell_guards.py`) refuses a pipe into
  a reader that can exit before its input ends (`grep -q`/`-m N`, `head`, `sed` with `q`, `awk` with `exit`) in every
  `.sh` under `tests/` and `scripts/` that sets pipefail, and in `tests/lib/`: the writer's SIGPIPE makes the pipeline
  fail with 141 although the reader succeeded. Read it all (`grep -c PAT >/dev/null`, `sed -n 1,3p`) or give it the
  file. A deliberate hit (a finite, small writer) goes in `tests/guards/pipefail-early-exit.allow` as
  `<file>:<line> <reason>`; an entry that matches nothing fails.
- **The seam is enforced by the type system, not a guard.** `tests/test_arch_drop.sh` compiles the engine without each
  arch and with each arch alone, so shared code that imports one fails to build ([engine-arch-seam](engine-arch-seam.md)).

## Lint

`zig build lint` runs `scripts/lint/` over every `.zig` file under `src/`: `main.zig` parses each file once with
`std.zig.Ast` (no Sema) and runs every rule on the tree. It is a build step that `zig build test` and `test-host` depend
on, and it runs every time, so a new or edited file is always read. A report reads `file:line:col [rule] fn: message`.

- A rule is one file in `scripts/lint/` that exports `name`, `reason` and `check(arena, tree, report)`, plus one line in
  `rules` in `main.zig`. It flags a pattern we never want; `const-write` is the first
  ([engine-mlx-gotchas](engine-mlx-gotchas.md#zig)); `unused-import` flags a non-`pub` file-scope `const` that only
  renames (`@import`, `a.b.C`, `x`) and that no identifier token of the file names again.
  `mlx-extern` flags an `extern` (or `@extern`) of an `mlx_*` or `_mlx_*` symbol outside `src/mlx.zig`, where the
  call counter sits ([engine-own-forward](engine-own-forward.md#the-seam)).
  `undefined-nested-write` flags, in a test block, `v.a.b = ...` on a `var v: T = undefined` before `v.a = ...`, `v = ...`
  or a `v.method()` call; its allowlist key is the test name with every non-alphanumeric character as `_`.
  `double-free` flags a handle that an `errdefer` frees and a later `defer` frees again.
  `scope-state` flags a hand write of `engine.active` (anywhere outside `src/tensor/`) or of `engine.enabled`
  (outside tests): the guards of [the seam](engine-own-forward.md#the-seam) are the only writers.
  `engine-hold-end` flags a call of `engine.unhold` outside `arch_handle.engineHoldEnd` and `src/tensor/`: the last
  hold's end also gives back the EXL3 module's engine-side arrays. `engine-lifecycle` flags a load's engine step
  (`enable`, `hold`, setting `own_weights`, `unsealed`, `seal`) outside `arch/cooler.zig` and `arch/handle.zig`;
  `cooler-side-query` flags `cooler_ask` or `cooler_for_test` outside `cooler_side.zig`. `promise-active` flags a fn
  that reads `.activeBytes()` and never asks `bills.promisedActive`: a promise read off the raw active bytes counts the
  decoder pools a lease lent free ([engine-memory-admission](engine-memory-admission.md#gpu-memory-readers)); a report
  or the lent term's own measure is an allowlist entry. `cache-gpu-stream` flags a
  `gpuStream()` call in `prefix_cache.zig` or `ssm_cache.zig` outside `arraysStream` and tests: cache array work takes
  its caller's stream ([engine-prefix-cache](engine-prefix-cache.md)). `response-store-field` flags
  the Responses store's map or LRU (`by_id`, `recency`) touched outside `responses.zig`: a record is held through
  `acquire`/`release`.
  `result-field-free` flags `.free(x.token_ids)` and `.free(x.logprobs)`: a `GenerationResult` is released with `deinit`.
  `template-runtime-safety` flags a fn under `src/template/` whose body does not open with `@setRuntimeSafety(true)`,
  and any `@setRuntimeSafety(false)` there: the engine reads pack text, and in ReleaseFast only that keeps its bounds
  and overflow checks. It covers the engine's own code, not the std functions it calls (those run unchecked); that is
  sound because the engine hands them only UTF-8 it validated (the template text; `std.json` refuses invalid UTF-8 and
  lone surrogates) or built from validated pieces at code-point boundaries, and slices it bounded itself.
  `template-superlinear-std` flags, under `src/template/` outside tests, `std.sort` and a `mem.` substring search or
  sort (`indexOf`, `lastIndexOf`, `count`, `replace`, `splitSequence`, `sort`, ...): their time grows with text times
  needle, or they assume a consistent comparator; the engine uses `value.Finder` (KMP) and `value.stableSort`.
  `device-probe` flags a call that takes the GPU's architecture name, the OS version or the QA switch by name
  outside `src/exl3/device.zig` ([the device profile](engine-kernels.md#the-device-profile)).
  `log-line-end` flags a `log.info|warn|debug|err` call whose format is one string literal that does not end in
  `\n` (`"{s}"` alone passes: a line built whole): the logger adds no line end, so the next line starts on that one.
  `arch-test-unlisted` flags a file that declares a test and sits beside an `arch_tests.zig` that does not import
  it: the domains select tests by listed file, so its tests would never run.
  `test-unrooted` flags any other `src/` file that declares a test and that no test root (`tests.zig`,
  `tests_host.zig`, `test_rest.zig`, the exl3 and encoder roots) imports: Zig runs only the tests a root reaches.
  `template-text-search` flags, outside `src/template/` and tests, a `mem.` search (`indexOf`, `count`, `startsWith`,
  ...) whose arguments name a template's source (`chat_template`, `template`, `tmpl`, `tpl`, `template_source`): a
  word in the text may sit in a comment or a comparison, so the server asks the compiled template's facts
  (`chat.asks`). It matches names, not data flow: a copy under another name (`isIncludeStub`'s trimmed `t`) passes.
  `template-unchecked-utf8` flags `Utf8View.initUnchecked` under `src/template/`: text walks go through the checked
  `value.codepoints`. `utf8-decode-raw` flags `std.unicode.utf8Decode*` outside `src/utf8_seq.zig`: it is `unreachable` on a
  slice shorter than its lead byte announces (a character cut by a segment end), so every decode asks `utf8_seq.decode`.
  `json-unchecked-shape` flags, outside test code (tests, `*_test.zig`, a private fn only tests
  call), `.object`/`.array`/`.string`/`.integer`/`.float`/`.bool` off `x.?`, `parsed*.value` or a map entry's
  `value_ptr` with no tag check of that receiver earlier in the fn, and an `@intCast(<expr>.integer)` with no bound
  on that integer earlier: valid JSON of another shape or range panics (`[]` as a `tokenizer_config.json`). It cannot
  see a bare read off a `std.json.Value` variable (no Sema).
  `owned-insert-arg` flags, outside test code, a `try xs.append*/put*/insert*(…)` whose argument makes an owned value
  (`try a.dupe*(`, `try a.alloc*(`): when the insert fails to grow, that value leaks. Reserve first and insert with
  `*AssumeCapacity`; an arena's insert goes in the allowlist.
  `refusing-eager` flags `core.refusing(..)` bound to a `const` or `var`: it sets the engine's refusal reason as it
  runs, so a value made ahead of a refusal hands its reason to the next, unrelated refusal.
  `refusal-unnamed` flags, under `src/tensor/`, an `error.TensorUnsupported` returned or handed to `fail` without
  `core.refusing`: the bare error names no reason and is not counted in the engine's refusals.
  `caught-error-unsettled` flags, in the files that run a request's work on the inference thread (`scheduler.zig`,
  `mtp_tick.zig`, `slot_outcome.zig`, `vision_queue.zig`), `@errorName(e)` of a `catch |e|` capture: the failure may
  have left the MLX latch, which `mlx.settledErrorName(e)` takes; a `checkError*` call's own error passes.
  `scheduler-hand-built` flags a `Scheduler` declared `undefined` that reaches `.submit(`: submit reads fields such a
  fixture leaves unset (`draining`), so it reads an earlier test's stack and can refuse; start from `submitTestScheduler`.
  `entry-checkout` flags a write of a hot-cache entry's `checkout` (assigned, set in a literal, captured by pointer,
  written through `.?`) outside `prefix_cache.zig`'s `beginCheckout`, `donateCheckout` and `endCheckout`, tests exempt.
  `error-body-literal` flags a string literal that spells an error envelope (`{"error":`, `"type":"error"`) outside
  `src/http.zig`, whose `sendError` picks the envelope by the request's API; tests and `*_test.zig` files exempt.
  `request-bool-read` flags `x == .bool` and `x != .bool` in the files that read a request body (`server`, `anthropic`,
  `completions`, `responses_api`, `http_conn`, `request_policy`, `generation_stream`): a request's bool is read by
  `request_parse.boolField`, so another type is refused by name and never read as the default.
  `in-place-buffer` flags a fn outside `mlx_pages.zig` that makes a buffer with `mlx_zeros` and writes with
  `mlx_slice_update`: a buffer written in place is made by `mlx_pages.inPlaceBuffer`, which refuses one off MLX's page grid.
  `in-place-buffer` flags a fn outside `mlx_pages.zig` that makes a buffer (`mlx_zeros`, `mlx_full`, `mlx_concatenate_axis`,
  a scope's `zeros` or `join`) and writes with `mlx_slice_update`: a buffer written in place is made by
  `mlx_pages.inPlaceBuffer` or `grownBuffer`, which refuse one off MLX's page grid. A buffer made in one fn and written in
  another is out of its reach (`Rows.put`, `KVCache.writeAtOffset`, DeepSeek's `Ring.write`).
  `cache-room` flags `roomFor` outside `mlx_pages.zig` and `src/tensor/`: the constructor that makes a cache buffer takes
  the engine's room for it once, as admission bills it.
- `tests/guards/lint.allow` holds `<rule> <file>:<fn> <reason>`, one per line, for a hit that is deliberate. An entry
  that matches nothing is stale and fails, so the list only shrinks.
- A new rule ships its red probe in `tests/test_lint.sh`, a green case for the near miss it must leave alone, and its
  allowlist entries.

## Characterization tapes

`tests/tape.sh record|check <pack> <dir>` ([matrix row](../tests/CLAUDE.md)) is the byte-identity proof for an engine/arch decoupling stage: a logit tape (`sushi kld capture` against itself, `cmp`) and a serving tape (HTTP cases per pack, each kept only if repeated recordings agree). Tapes live under `~/.sushi/runs/tapes/<base-commit>/<pack>/`, never in the repo; it is not a release gate.

## Microbenches

`sushi-ubench` is not a tier: `zig build test` never builds it, `tests/gate.sh` only compiles it, and its files hold no `test`
block (the build refuses one). Its output is ms across rungs, never pass/fail; see [process-ubench](process-ubench.md).
Timing code still inside a `SUSHI_*_UBENCH` test block moves there.

## Where a test lives

- Beside the code it tests, at the bottom of that file. Tests move with their code.
- A test's model is its file's folder (`src/qwen4/`, `src/mimo2/`, `src/glm5/`, `src/dsv41/`); the `src/` root is shared.
  `-Dmodel=<arch>` runs one folder's tests plus the shared ones. A new test file of an arch is listed in that
  arch's `arch_tests.zig` (`pub const tests`, or `host_tests` for an MLX-free file), never in `src/tests.zig`.
  A source file with tests at its bottom is listed too: the domains select tests by listed file, so an unlisted
  file's tests never run (lint `arch-test-unlisted`). A shared file with tests is imported from `src/tests.zig`, or
  `src/tests_host.zig` when MLX-free (lint `test-unrooted`), and an MLX one is listed in its domain (`test_domains`).
- A shared file (`trunk.zig`, `server.zig`, `generate.zig`, `scheduler.zig`, `prefix_cache.zig`, the EXL3 kernels)
  holds only `shared` tests. Code for one arch lives in that arch's files, and so do its tests.
- A scenario across several modules, or a table of real captures, is its own `<topic>_test.zig`.
- A helper, constant or fixture builder only tests use is not declared in a shared file outside its `test` blocks: it
  goes to the arch's folder when one arch owns it, else to a `*_test.zig` beside the shared file.
- A server scenario under concurrency (requests racing a delete, a shutdown or an unload) runs in process through
  `src/server_test.zig`: `serve` with an embedded client, each arch's servable tiny (`cooler_contract.serverDir`) on
  Sushi Cooler, socketpair connections, an allocator that poisons what it frees.
  A scenario that fails while a connection thread still loads leaves `serve` with `ConnectionsOutliveShutdown` and no
  teardown: the harness prints the scenario's error beside it and ends the process, as `main` does, since freeing under
  the live scheduler and inference thread is undefined (a segfault that hid the cause).
  A cell under a declared wired limit sets it with `setWiredFloor`, whose limit is the box's RAM (a floor is at most
  the RAM less the margin) and which refuses a floor the bill would not read: a 120,000 MB limit declared none below
  about 118 GB of RAM, so its cells tested nothing there.
  A tiny that streams its experts names the SSD budget its load takes (`cooler_contract.server_ssd_budget_gb`, written
  to the test's model-settings.json): a box under 96 GB has no RAM-tier default, and its cells would test a refusal.
  A cell that needs a request to stand still (queued, or between two tokens) holds the inference thread in a
  blocker's prefill (`PrefillHold`), never a wall-clock race against the tiny's decode.
- Integration scripts in `tests/` cover only what HTTP shows and a unit test cannot reach.

## How a test is built

- Smallest input that reaches the arm under test. A sweep samples: shipped rates plus the table's ends, not every value.
- A test that counts a speculative request's calls or pool asks pins the MTP depth (`mtp_depth_bounds.active`, as
  `--mtp-min-depth N --mtp-max-depth N`): unpinned, each round's plan follows the trunk's measured cost table and live
  sync times, so the round count, and every call per round, changes from request to request with the same tokens.
- One function, many inputs: one table-driven test, one row per case.
- One fixture or pack load per process, shared by the tests that read it.
- No sleeps to wait for a thread. Use a hook or a counter. Any wait is bounded and under 100 ms.
- A served test that waits on another request bounds what it claims (the request is not held: its model is claimed
  soon after the step it waited on), never a wall clock that holds a reload: every load runs its served warm.
- A test client's read limit only guards against a hang: it budgets a cold first load (its kernel compiles too), and
  a read past it fails by name, never as a short answer.
- Deterministic: no dependence on scheduler timing, machine memory or test order. Global overrides restore in `defer`.
- A number a load or the GPU's class decides is pinned by the test that reads it: DeepSeek's wide-forward shape sum
  reads the arms of the config it prices (`Config.wide_arms`), the served kernels' from `parseText`, never a global.
- No dependence on where the test binary sits: `tests/gate.sh` builds it in a cache outside any checkout.
- "No worse than the reference" states its margin in ulps, or compares bytes within one arm: a bare `<=` between two
  rounded errors flips on other hardware.
- A GPU f32 result held to a reference takes its bar from MLX's TF32 reading (`matmul.gpu().tf32`): MLX keeps the
  first one for the process, so a bar that holds only on f32 tiles passes or fails by test order.
- A passing test prints nothing, on either stream. Diagnostics go behind an env switch or into `errdefer`.
- A regression pin is written red first. Its name says what failed, never an issue number or date.
- A test name is one line without escapes (`test "name" {`): the build scans names to place tests in binaries.

## What a change runs

| change | runs before commit |
|---|---|
| host files only (chat, parsers, grammar, tokenizer, pure planners) | `zig build test-host` |
| one arch's files only | `zig build test -Dmodel=<qwen4\|mimo2\|glm5\|dsv41>`, plus `tests/run.sh --tier live --model <arch>` for the scripts it touches (`--only <glob>` narrows) |
| `src/exl3` or `src/encoder` only | `zig build test -Dmodel=exl3` or `-Dmodel=enc` |
| a broad area | `-Dmodel=` (above); a narrow check names its tests with `-Dtest-filter=`, which stays parallel |
| docs only, or test scripts only | nothing in Zig; `tests/run.sh --check` and the scripts it touched |
| a landing on main | the row above that fits its change, with the tiny-fixture envs set, plus `python3 tests/suite_guards.py` and the touched area's scripts |
| the end of a series of landings (shared files, `build.zig`, `mlx.zig`, several areas stacked) | `zig build test` with the tiny-fixture envs set |
| a version bump (release step 1) | `tests/gate.sh` on the pushed bump commit (below) |
| an architecture joins the family, or one is dropped for real | `tests/test_arch_drop.sh` (the only time it runs) |
| a bisect of an unexplained failure | the subset that fails, widened only if it does not reproduce there |

The full suite is the only run that sees every test in its real binary and order: a test that leaks or fails alone can
pass after its neighbours warm a cache. That is why a wide change and a version bump pay for it; a landing confined to
one area does not.
A full-suite run (series end, version bump) also sets `QWEN4_TEST_MODEL` to a Qwen cut fixture
(`QWEN4_STQ_CUT`, then `QWEN4_STQ4_CUT`): the real-pack Qwen tests run nowhere else, and left unset they rot unseen.
Only a box of 64 GB or more sets it: the contract holds an MLX reference load and a Cooler load of the cut at once,
beside the served warm's widest chunk (about 6 GB of transient).
It sets `QWEN4_BF16_TINY` to `tiny-models/qwen4-bf16-tiny-hd256` (the HF BF16 source of the STQ hd256 tiny), so the
Cooler contract holds the BF16 checkpoint's format at served head geometry.
A full-suite run compares its test total (`Build Summary` plus `test-host`) with the previous full run's: an
unexplained drop means tests were not built, and blocks the push.

**The local gate.** Every check a local run can prove runs locally, once, on the exact commit a release cuts:
`tests/gate.sh [--test-budget-gib N] [<ref>]` (default `origin/main`; fetch first) refuses while the GPU lock is held,
makes a clean worktree of the commit (no untracked file can mask a missing one) and runs, under `taskpolicy -a`,
`zig build lint`, `zig build ubench` and `zig build test -Doptimize=ReleaseFast --summary all` with the tiny-fixture
envs of `tests/lib/fixture_envs.sh`, `-Dtest-log`, `--maxrss` of the test budget and, when the envs name the Qwen cut,
`-Dfixture-cut=true`, with `tests/hermetic_guards.sh` running beside the test step. The budget is 72 GiB on a box of
96 GiB or more (M5 Max: at most four 16 GiB cut steps at once), 20 GiB and `-j8` on 32 GiB or more (M1 Max); a
smaller box refuses until `--test-budget-gib` names one. Each test's log line shows as it lands; at the end
`tests/lib/test_log.py` prints the per-step table (tests, pass, skip, fail, wall), the 10 slowest tests and, last,
every failed test with the error block the build runner printed for it.
`--continue <prior-ref>` reruns only what changed: every run step is keyed (`tests/lib/test_keys.py`) by its test
binary's code and data (not its UUID, signature or symbol tables, which every build changes), its env, the box,
`lib/mlx`, the tracked files that are not Zig or docs, the SUSHI_/MLX_/fixture env and every fixture folder's
contents; a step whose key is the prior gate's and whose run there ended whole is left out (`-Dtest-skip-step`) and
its prior lines go into the log, and the record names the reuse (`reused.from`, `reused.steps`). Lint, ubench and the
guards always run; in doubt, a step reruns. With the domains, an edit to one domain's tests reruns that domain. The
worktree is `~/.sushi/runs/gate/worktree` for every gate, since a binary holds its sources' paths: one a running gate
holds (`worktree.pid`) refuses the gate, one a stopped gate left is removed with a line saying so.
Its zig builds share a persistent local cache, `~/.sushi/runs/gate/zig-cache` (`--cache-dir`), so a fresh worktree is
not a cold build; remove it after a toolchain or SDK change, as a checkout's `.zig-cache`.
Green writes `~/.sushi/runs/gate/<sha>.json` (sha, time, Zig version, the Build Summary counts, each step's minutes,
every test's ms, step and status from the log, `<sha>.logs/tests.tsv`); a failed step names itself and writes none.
`./release.sh` dispatches only with a record for origin's main head and passes its sha; `release.yml` starts only
from a dispatch and refuses a release run that builds any other commit (a `dry_run` packaging check needs none). The runner keeps only what a clean checkout alone proves: the build from the pins,
the shipped binary's smoke test and staged-mlx check, `test_no_local_paths.sh`, signing and packaging.
The hermetic guards are one list, `tests/hermetic_guards.sh`, which the gate and every landing run; it fails on a skip,
so a box without PyYAML (`python3 -m pip install pyyaml`; the gate takes `SUSHI_GATE_PYTHON`) cannot pass it unseen.
The arch drop test runs only when an architecture joins the family or one is dropped for
real: never per landing, at a series' end or at a version bump. `--since <ref>` takes only the architectures whose folder `src/<arch>/` changed after that commit: an
architecture the series did not touch is not dropped, and a series that touched no arch folder drops nothing. Its jobs
(a drop and a single-arch build for each architecture taken) run at once: all eight took 11.5 minutes on the M5 Max at
a954450b (the slowest drop's 687 s), against 42 minutes one after another.
A landing runs what its diff can break and nothing for ceremony: its new tests, the filters of the code it changed, one
arch domain when it reaches that arch widely. A rebase that touched none of its files repeats no run.

## Integration scripts

A live script defaults to a pack under `${SUSHI_MODELS_DIR:-$HOME/.sushi/models}` that the release gate uses, and skips
loudly when the pack is missing. Every script is in the [tests/CLAUDE.md](../tests/CLAUDE.md) matrix, or it is deleted.

**Header.** The line after the shebang (`#` for sh/py, `//` for cjs/mjs) is `sushi-test: key=value ...`; a second line
may add only `needs=<commands>`. Scope is the folder (`tests/qwen4|mimo2|glm5|dsv41/`; the root is shared), never a tag.

| key | values |
|---|---|
| `tier` | `hermetic` (no binary), `host` (the binary, no weights, no lock), `pack` (real weights, run.sh takes the lock), `live` (loads a pack over HTTP); `bench`, `tool` and `lib` are never run |
| `boot` | live only: `api`, `exact`, `ram`, `knobs` attach to a shared boot; `own` boots itself |
| `pack` | `own` and `pack`: `default` (the folder's model), `each` (once per model pass), a model key or an alias. An attach script (`api`, `exact`, `ram`) runs in every model pass unless it names one, `pack=qwen4` (`qwen4`, `glm5`, `dsv41`, `mimo2`): for a property of the server, not of the model (`test_metrics.sh`, `test_timings.sh`) |
| `boots` | `own` only: the scripted number of boots, for the ledger |

Exit codes: 0 pass, 1 fail, 3 SKIP (one line naming what is missing).

**Boot helper.** `tests/lib/boot.sh` (flags in `lib/profiles.sh`, Python twin `lib/sushi_test.py`) is the only way a
script starts a server. `sushi_boot <label> <pack> [flags]` takes the GPU lock, boots on a free port with a private
`HOME` and `SUSHI_PREFIX_CACHE_DIR` under `SUSHI_RUN_DIR`, and waits for `/health` plus `Model ready`; `sushi_stop`
(also the EXIT trap) stops by PID and writes `<label>.END`. An attach script calls `sushi_attach` (a nonce for every
prompt), reads `sushi_log_since` (a line the engine logs once per process, such as an `engaged` line, is read over the whole
`$SUSHI_LOG`, since an earlier script on the boot may have logged it), runs the binary as a client through `sushi_cli`
(`launch`), and asks `sushi_require <mtp|vision|video|batching|stq|thinking|spec>`. Profiles `api`, `exact`, `ram` and
`knobs` are one boot each per model pass (`knobs` keeps every prefix cache off and boots with a model-settings file
naming `lib/knobs-bias.json`, written by `sushi_profile_settings`); `SUSHI_TEST_PACK_<MODEL>` overrides a pack,
`SUSHI_BOOT_TIMEOUT_S` and `SUSHI_STOP_WAIT_S` bound the waits.

**A script that runs `--serve` without the helper names its `--port`** (a flag or refusal test whose model never
loads): with none the process is the default port's server, and its lines land in `~/.sushi/logs/sushi-12345.log`,
the log of the server a user runs there. Such a script passes `--log-file off` too. `tests/run.sh --check` refuses a
launch with no `--port`.
A script that stops its server on purpose (a SIGTERM test) sets `SUSHI_EXPECT_GONE` before `sushi_stop`, so the exit is
not reported as a crash. A Python `own` script uses `lib/sushi_boot.py` (`Server`, `gpu_lock`, `private_env`, `pack`).

**Runner.** `tests/run.sh [--tier hermetic,host,pack,live|all] [--model qwen4|mimo2|glm5|dsv41|shared|all] [--only glob]
[--list] [--check] [--keep-going] [--no-build] [--out dir]`. The default is `hermetic,host`, every pass. A selection that holds no script exits 2: an `--only` that
matches nothing, or names a script outside the selected `--tier` or `--model` pass (it names the tier or pass to give;
`boot=own` scripts on the default pack run in the `shared` pass), is refused, never a green empty run.
`--list` prints the plan and boots nothing; `--check` refuses a missing or invalid header, a missing or stale matrix
row and an attach script that launches `sushi` (`--no-matrix` waives the matrix rules for a tree under construction). A pass builds ReleaseFast, boots each profile once with the lock held
for that boot only, reboots once after a crash, and stops at the end of the boot that failed unless `--keep-going`.
The summary and an `END` marker land in `--out` (default `~/.sushi/runs/tests-<ts>/`). `tests/test_run.sh` proves it
on a fixture tree and a stub server, without a model or a GPU.

## Cutting a test

- Dead code: the test goes with the code, in the same commit.
- A regression pin goes only when the code it guards is gone, or a named stronger test catches the same failure. The
  commit names that test.
- A merge keeps every input row of the tests it replaces.
