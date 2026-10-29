# Test suite

A test earns its place by catching a failure nothing else catches, on code a served model reaches. Anything else is
cut: there is no "might be useful" tier. The served models are Flash-Next (`qwen4`), MiMo (`mimo2`), GLM (`glm5`) and
DeepSeek (`dsv41`); MiniCPM5 and Gemma tool-call parsing stay because quant work and agents use them. The TDD order
and the change classes are in [CLAUDE.md](../CLAUDE.md#testing--tdd-is-mandatory); the script matrix is in
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
- A struct literal read back, a compile-shape check, a "does not crash" or "is finite" smoke with no oracle.
- A wall-clock assertion. Timing is a microbench in `sushi-ubench` ([process-ubench](process-ubench.md)); a test that
  timed something keeps its exactness half.
- A test of code no served path reaches. Dead code goes with its tests in the same commit.
- An assertion that a model must think, answer or call a tool. Assert the invariant, branch on the model's choice.
- A second copy of a check another test already makes on the same input with as strong an assertion.

## Tiers

| tier | means | where it runs |
|---|---|---|
| host | touches no MLX array; its file reaches no `mlx.zig` | `zig build test-host`: anywhere, beside a live server, no GPU lock, every PR in CI |
| gpu | Metal kernels on small synthetic tensors; the first line is `try mlx.requireGpu();` | `zig build test` |
| pack | needs a fixture or pack named by an env var; skips when it is unset | `zig build test` with the env set |

- `zig build test-host` builds `src/tests_host.zig`, a root that links no MLX: a file in it that reaches MLX fails the
  link. The build reads that root's imports as the host list, and no MLX bucket may also select a host file's tests.
- `zig build test` runs the host tier, the MLX buckets, the exl3 binaries and the encoder binary.
- `SUSHI_TEST_NO_GPU=1` makes a test build see no Metal backend, so `requireGpu` skips (diagnostic, test builds only).
- The pack tier's envs: `DSV41_TINY`, `DSV41_STQ_TINY` (tiny DeepSeek fixtures), `QWEN4_TEST_MODEL`,
  `MIMO_V2_MODEL`, `SUSHI_MODELS_DIR`. A landing on main sets the tiny-fixture envs; the real-pack envs run with the release gate. Without them a green
  suite proves nothing there.
- Model scope is `qwen4`, `mimo2`, `glm5`, `dsv41`, `enc` (the encoder) or `shared` (every model). A test's file says
  its model; a test of one model's code lives in that model's files.

## Microbenches

`sushi-ubench` is not a tier: `zig build test` never builds it, CI only compiles it, and its files hold no `test`
block (the build refuses one). Its output is ms across rungs, never pass/fail; see [process-ubench](process-ubench.md).
Timing code still inside a `SUSHI_*_UBENCH` test block moves there.

## Where a test lives

- Beside the code it tests, at the bottom of that file. Tests move with their code.
- A test's model is its file's folder (`src/qwen4/`, `src/mimo2/`, `src/glm5/`, `src/dsv41/`); the `src/` root is shared.
  `-Dmodel=<arch>` runs one folder's tests plus the shared ones.
- A shared file (`transformer.zig`, `server.zig`, `generate.zig`, `scheduler.zig`, `prefix_cache.zig`, the EXL3 kernels)
  holds only `shared` tests. Code for one arch lives in that arch's files, and so do its tests.
- A scenario across several modules, or a table of real captures, is its own `<topic>_test.zig`.
- Integration scripts in `tests/` cover only what HTTP shows and a unit test cannot reach.

## How a test is built

- Smallest input that reaches the arm under test. A sweep samples: shipped rates plus the table's ends, not every value.
- One function, many inputs: one table-driven test, one row per case.
- One fixture or pack load per process, shared by the tests that read it.
- No sleeps to wait for a thread. Use a hook or a counter. Any wait is bounded and under 100 ms.
- Deterministic: no dependence on scheduler timing, machine memory or test order. Global overrides restore in `defer`.
- "No worse than the reference" states its margin in ulps, or compares bytes within one arm: a bare `<=` between two
  rounded errors flips on other hardware.
- A passing test prints nothing, on either stream. Diagnostics go behind an env switch or into `errdefer`.
- A regression pin is written red first. Its name says what failed, never an issue number or date.
- A test name is one line without escapes (`test "name" {`): the build scans names to place tests in binaries.

## What a change runs

| change | runs before commit |
|---|---|
| host files only (chat, parsers, grammar, tokenizer, pure planners) | `zig build test-host` |
| one arch's files only | `zig build test -Dmodel=<qwen4\|mimo2\|glm5\|dsv41>`, plus that arch's scripts |
| `src/exl3` or `src/encoder` only | `zig build test -Dmodel=exl3` or `-Dmodel=enc` |
| a shared file, `build.zig` or `mlx.zig` | `zig build test` |
| a landing on main | `zig build test` with the tiny-fixture envs set, plus the touched area's scripts |

CI runs `test-build` (every test compiles), `test-host` and the hermetic guards on every PR; the GPU tier is a
`workflow_dispatch` choice (`tier: all`), because a hosted runner's GPU flips rounding ties the M5 holds.

## Integration scripts

A live script defaults to a pack under `${SUSHI_MODELS_DIR:-$HOME/.sushi/models}` that the release gate uses, and skips
loudly when the pack is missing. Every script is in the [tests/CLAUDE.md](../tests/CLAUDE.md) matrix, or it is deleted.

**Header.** The line after the shebang (`#` for sh/py, `//` for cjs/mjs) is `sushi-test: key=value ...`; a second line
may add only `needs=<commands>`. Scope is the folder (`tests/qwen4|mimo2|glm5|dsv41/`; the root is shared), never a tag.

| key | values |
|---|---|
| `tier` | `hermetic` (no binary), `host` (the binary, no weights, no lock), `pack` (real weights, run.sh takes the lock), `live` (loads a pack over HTTP); `bench`, `tool` and `lib` are never run |
| `boot` | live only: `api`, `exact`, `ram`, `knobs` attach to a shared boot; `own` boots itself |
| `pack` | `own` and `pack` only: `default` (the folder's model), `each` (once per model pass), a model key or an alias |
| `boots` | `own` only: the scripted number of boots, for the ledger |

Exit codes: 0 pass, 1 fail, 3 SKIP (one line naming what is missing).

**Boot helper.** `tests/lib/boot.sh` (flags in `lib/profiles.sh`, Python twin `lib/sushi_test.py`) is the only way a
script starts a server. `sushi_boot <label> <pack> [flags]` takes the GPU lock, boots on a free port with a private
`HOME` and `SUSHI_PREFIX_CACHE_DIR` under `SUSHI_RUN_DIR`, and waits for `/health` plus `Model ready`; `sushi_stop`
(also the EXIT trap) stops by PID and writes `<label>.END`. An attach script calls `sushi_attach` (a nonce for every
prompt), reads `sushi_log_since`, runs the binary as a client through `sushi_cli` (`launch`), and asks `sushi_require <mtp|vision|video|batching|stq|thinking|spec>`. Profiles
`api`, `exact`, `ram` and `knobs` are one boot each per model pass; `SUSHI_TEST_PACK_<MODEL>` overrides a pack,
`SUSHI_BOOT_TIMEOUT_S` and `SUSHI_STOP_WAIT_S` bound the waits.

**Runner.** `tests/run.sh [--tier hermetic,host,pack,live|all] [--model qwen4|mimo2|glm5|dsv41|shared|all] [--only glob]
[--list] [--check] [--keep-going] [--no-build] [--out dir]`. The default (`hermetic,host`, every pass) is what CI runs.
`--list` prints the plan and boots nothing; `--check` refuses a missing or invalid header, a missing or stale matrix
row and an attach script that launches `sushi` (`--no-matrix` waives the matrix rules until the matrix is complete). A pass builds ReleaseFast, boots each profile once with the lock held
for that boot only, reboots once after a crash, and stops at the end of the boot that failed unless `--keep-going`.
The summary and an `END` marker land in `--out` (default `~/.sushi/runs/tests-<ts>/`). `tests/test_run.sh` proves it
on a fixture tree and a stub server, without a model or a GPU.

## Cutting a test

- Dead code: the test goes with the code, in the same commit.
- A regression pin goes only when the code it guards is gone, or a named stronger test catches the same failure. The
  commit names that test.
- A merge keeps every input row of the tests it replaces.
