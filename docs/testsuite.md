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
- A wall-clock assertion. Timing is a ubench behind its env switch, never in the suite.
- A test of code no served path reaches. Dead code goes with its tests in the same commit.
- An assertion that a model must think, answer or call a tool. Assert the invariant, branch on the model's choice.
- A second copy of a check another test already makes on the same input with as strong an assertion.

## Tiers

| tier | means | where it runs |
|---|---|---|
| host | touches no MLX array; its file reaches no `mlx.zig` | `zig build test-host`: anywhere, beside a live server, no GPU lock, every PR in CI |
| gpu | Metal kernels on small synthetic tensors; the first line is `try mlx.requireGpu();` | `zig build test` |
| pack | needs a fixture or pack named by an env var; skips when it is unset | `zig build test` with the env set |
| bench | timing behind its env switch; asserts nothing about time | on request only |

- `zig build test-host` builds `src/tests_host.zig`, a root that links no MLX: a file in it that reaches MLX fails the
  link. The build reads that root's imports as the host list, and no MLX bucket may also select a host file's tests.
- `zig build test` runs the host tier, the MLX buckets, the exl3 binaries and the encoder binary.
- `SUSHI_TEST_NO_GPU=1` makes a test build see no Metal backend, so `requireGpu` skips (diagnostic, test builds only).
- The pack tier's envs: `DSV41_TINY`, `DSV41_STQ_TINY` (tiny DeepSeek fixtures), `QWEN4_TEST_MODEL`,
  `MIMO_V2_MODEL`, `SUSHI_MODELS_DIR`. A landing on main sets the tiny-fixture envs; the real-pack envs run with the release gate. Without them a green
  suite proves nothing there.
- Model scope is `qwen4`, `mimo2`, `glm5`, `dsv41`, `enc` (the encoder) or `shared` (every model). A test's file says
  its model; a test of one model's code lives in that model's files.

## Where a test lives

- Beside the code it tests, at the bottom of that file. Tests move with their code.
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
| one arch's files only | `zig build test -Dtest-filter=<stem>.test.` for each touched file, plus that arch's scripts |
| a shared file, `build.zig` or `mlx.zig` | `zig build test` |
| a landing on main | `zig build test` with the tiny-fixture envs set, plus the touched area's scripts |

CI runs `test-build` (every test compiles), `test-host` and the hermetic guards on every PR; the GPU tier is a
`workflow_dispatch` choice (`tier: all`), because a hosted runner's GPU flips rounding ties the M5 holds.

## Integration scripts

A live script defaults to a pack under `${SUSHI_MODELS_DIR:-$HOME/.sushi/models}` that the release gate uses, and skips
loudly when the pack is missing. Every script is in the [tests/CLAUDE.md](../tests/CLAUDE.md) matrix, or it is deleted.

## Cutting a test

- Dead code: the test goes with the code, in the same commit.
- A regression pin goes only when the code it guards is gone, or a named stronger test catches the same failure. The
  commit names that test.
- A merge keeps every input row of the tests it replaces.
