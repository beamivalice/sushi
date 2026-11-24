# Brief: stage 3 of the static bill, Hy4's narrow phases on the arena

You are taking over the static Cooler bill at stage 3. Stages 0 to 2 are built and tested: the recorder runs on the
engine, dry and real, and every Hy4 forward's survivors have owners. Nothing is placed in an arena yet. Your job is to
build the arena and the binder's hooks, then move Hy4's narrow phases onto them, in the order below, without
redesigning what exists.

Read first, in this order:
1. `CLAUDE.md`. Its rules bind you: test-first, ReleaseFast, commit per step, no push.
2. [bill_redesign.md](bill_redesign.md), all of it, and closely: part 5 (the plan, packing), part 6 (runtime: the
   arena and the binder), part 8 (persistent state), part 11 (what is written), part 12 (the build, with stage 1's and
   stage 2's as-built notes) and part 14 (decisions D1 to D17).
3. [bill_redesign_prompt.md](bill_redesign_prompt.md), the brief stages 0 to 2 worked from. Its traps still hold.

This file only tells you where stage 2 left things, what is decided, and where the traps are.

## What stage 3 is

From part 12 of the design:
- **3a.** Build:
  - the arena (one buffer, `[H x D | G]`, made once at the load's end);
  - the binder's hooks in the engine (`Binder.ask`, `release`, `exit`, `begin`);
  - the marks and floors in `core.dispatchLevel`;
  - the host ring with its replicas;
  - the bounds on passes (how many, how long);
  - poison and canaries in a test build;
  - what follows a miss (D5): the one shape recorded after the refused request, the tables swapped between
    executions when they fit the arena held, and the shape cached for the next load when they do not.

  The floors are part of the packer (D13). Until they land, a plan packs by lifetimes alone
  (`plan_pack.pack(.., .{ .level_rule = false })`), and the level rule is used only with floors.
- **3b.** Put `decode` on the arena. Then, one at a time:
  - `verify` (see the open question on its widths below);
  - `draft`;
  - `sample`.

  Each phase must give equal bytes on the tinies and on the release pack. The release pack run is the coordinator's.
  In the commit that moves a phase's storage, that phase's term leaves `hy4_bill`.
- **Gate (the fast bench, decode).** Compare against the pool, arms interleaved in one session: tok/s, barriers per
  forward, GPU busy time and host ms per token must be equal or better. The coordinator runs it; you hand over the
  command.

Stop and report after 3a, and after each phase of 3b. Do not start stage 4 without the owner's word.

## Where stage 2 left things

Branch `agent/opus-bill`, which the coordinator pushes to `exp`. Work in your own clone of it, on a branch of it.

Stage 2's commits, in order:
- `37665894` **2a.** Appends land at once on the engine: `mlx_pages.InPlace.landed`, `route.settleNow`. A layer that
  closes a group hands up no pending write.
- `ad5b10c5` **2b.** `Model.reserveFor(reach)`, called by the served forward (`hy4flash/arch.zig`):
  - each sparse layer's rows and pooled keys are reserved at `capBound(reach)`;
  - each sliding ring is made, and each sparse layer's open-group block;
  - with a drafter, the handover block and the picks blocks are made too;
  - `writeFixed` writes the open group, the handover and the picks in place;
  - a point copies the small state (`copyOf`, the engine's `copyOwned`), and a rollback writes the copy back;
  - `Model.LayerKey.closes` counts the groups a forward closes.
- `d8c5160e` **2c.** `hy4_prefix.Held` keys a store by `core.storeKey`, not by its bytes: every dry store shares one
  address.
- `98dfb2e2` **2d.** `fwd.placedZeros` is a GPU pass (`ops.zerosOnGpu`), off the host ring.
- `b3a9bf3b` **2e.** `hy4_hc` holds one wide-mix grant per pipeline. `Model.probeWide` runs a dry census of the wide
  pipelines, then launches each one once.
- `c32f2ab2` `Model.dryState(n, reach)`: the state of `n` rows, made without forwards. The audit reaches each context
  with it.
- `8a73af8a` The gate also records a prefill chunk.
- The review's fixes:
  - `2961582a` A failed landing is released once.
  - `4f3746e0` The reservation follows the running request's reach, and a rollback keeps it (`Model.reach`).
  - `c1bd2eaa` The bill and the held bytes count the open, handover and picks blocks.
  - `031404e8` A stored point shares the live point's small state.
  - `143792ad` Doc comments.
  - `16a4de7a` The gate's draft phase really runs, and drafter layers are keyed frames.
  - `2f18735d` The census bounds its inputs.

Where the code stands:
- **The gate holds.** `hy4_plan_test`'s cell "hy4 plan: dry decode, verify at every block width, draft and a prefill
  chunk, from a served request's state, ..." records from a served request's state:
  - decode, verify at 2 to 8 rows and a 12-row prefill chunk, on `HY4_BF16_TINY` and `HY4_STQ_TINY`;
  - the draft, on `HY4_STQ_TINY` with DFly2 loaded.

  None ends with an escape. Each run gives one trace under both fills. Decode, each verify width and the draft each
  merge into one shape. The restore and rollback cells give the serial run's bytes.
- **The dry state.** `Model.dryState` holds only inside a dry recording. A decode or verify from it records the same
  trace as one from a prefill to `n`, with the handover kept and without.
- **The audit.** `hy4_plan_audit_test` builds a dry model of the release config. Set `SUSHI_PLAN_AUDIT=<file>` to
  write its table. It takes about 4 s. Its numbers are shapes, not timings; columns: packed by lifetimes, host
  ring, today's `hy4_bill` term.

  | phase | rows | keys | packed | host ring | bill |
  |---|---|---|---|---|---|
  | prefill chunk | 4,096 | 4,096 | 2.73 GB | 143 MB | 4.23 GB |
  | prefill chunk | 4,096 | 1,048,512 | 2.73 GB | 144 MB | 5.25 GB |
  | decode | 1 | 4,097 | 67.7 MB | 171 KB | 288 MB |
  | decode | 1 | 1,048,513 | 67.7 MB | 177 KB | 1.55 GB |
  | verify | 2 / 4 / 8 | 4k to 1M | 68.2 / 69.4 / 71.6 MB | 208 to 433 KB | 291 MB to 2.11 GB |
  | rows (not a verify block) | 16 | 4k / 1M | 130 / 281 MB | 663 KB | 557 MB / 2.75 GB |
  | draft | 6 | any | 70.6 MB | 12 KB | 719 MB |

  At stage 1 the prefill chunk asked 16.7 GB of the host ring and 2.79 GB of the arena. The decode, verify and draft
  arena figures did not move in stage 2.
- **Not done in stage 2, by ruling or by scope:**
  - Media (the vision rows).
  - The readout logits a served forward returns to the sampler (`generate.Pending.logits`, shared code: D17).
  - `Fixed` and `Points` are blocks and copies on the pool, not arena slots.
  - Growth is still a copy between forwards: spike S4 (in-place growth of a wrapped reservation) is unanswered.

## What is decided, and what is not yours to decide

- **[What may not change](bill_redesign.md#fixed)** is eight rules. If you need one changed, stop and say which and
  why, with the case that breaks it. Never add a fallback "for now".
- **[D1 to D23](bill_redesign.md#decisions)** are the owner's rulings (D23 the coordinator's, under the owner's word to decide). Those that bind stage 3 most:
  - D5: a miss heals in process only within the arena held; the arena never grows while serving.
  - D9: the recorder's levels are the execution's.
  - D12: a phase plans as one shape only with eager appends. That is done; keep it so.
  - D13: the floors are part of the packer.
  - D16: `hy4_bill` is not tightened. Its terms leave as their storage moves, and stage 5 deletes it.
  - D17: 1.8.0 ends at stage 5. Nothing in stages 1 to 5 may change DeepSeek's or Qwen's bytes or bills. The pool and
    its budget machinery stay for them. Any shared-code change must be a no-op for them, and their fixtures in the
    Cooler contract are the proof.
- **Not yours:**
  - any flag, subcommand or env twin;
  - changing a fixed rule;
  - loading a real pack, a timed run, and anything that holds tens of GB (ask the coordinator, with the command);
  - designing for stages 6 to 8 beyond what the shared seams already need.
- **The foundations are a contract.** You may fix a bug in them (red test first) and extend them. You may not replace
  a mechanism with one of your own.
- **Answered by the owner** before stage 3 began ([D18 to D22](bill_redesign.md#decisions)): verify is 2 to 8 rows;
  the logits are Hy4's state (a `Fixed` pair); `Model.probeWide` runs at load and the streamed BF16 route is classified
  so every Hy4 load plans; a smaller reach shrinks the state by a copy; the double free class gets a lint rule.

## Traps stage 2 hit

- **A dry recording of a streamed BF16 route is refused.** `moeLayerRest` reads its picks with a plain readback
  (`hy4_forward.hostPicks(.., null)`), which a dry recording refuses by design. A load whose experts are streamed
  BF16 cannot be planned until that route is classified or moved. Plan resident and STQ loads.
- **Process-wide caches keyed by row count** (`exl3/linear.zig`'s GEMM input ring, `exl3/side_cache.zig`) refuse a
  fill inside a dry recording (`CacheFillDry`). A real forward of that width must run first, or the cache moves to the
  load. The gate's prefill probe runs a real chunk of its width first for this reason.
- **`Model.probeWide` resets the state.** It must run before a dry recording, never in the middle of a request. It is
  not a load step: on a load that cannot record dry it would fail. Its census runs at position 0; a pipeline that only
  a later position reaches is refused by name (`Hy4HcWideUnprobed`).
- **The engine's `mlx_copy` is a share** (copy-on-write). A copy with a buffer of its own is
  `mlx.engine.copyOwned`.
- **An in-place write copies when another handle holds the buffer** (`core.sole`). So:
  - a point must copy the small state, not share it;
  - a view over a block (`st.open` over `open_buf`, `last_hidden` over `hidden_fixed`, `last_picks` over
    `picks_fixed`) must be freed before `writeFixed` writes the block.
- **An `errdefer` that frees a handle must end where the handle passes to its owner.** After the hand-off, a failure
  frees the owner's live handle and the owner frees it again (`core.double_releases` counts this). The fix pattern is
  in `row_store.put`. No lint rule covers this class yet.
- **The served forward reserves `@max(end_tokens, pos + rows)`.** A cache's end binding can be short of a forward's
  rows; a load's warm is one case.
- **DSpark's verify does not pass through `arch.forward`.** `hy4_dspark.verify` calls `forwardReadout` directly, so
  nothing reserves there. The reservation from the request's earlier forwards covers it (`Model.reach`).
- **A frame's key must carry every fact that decides its structure:**
  - the groups a layer closes, because the pool takes their rows one at a time from two (`fwd.mmByRow`);
  - a drafter layer's variant, which is whether its target's selection stands.

  A merge refusal names the frame and node. To find which asks the probes part at, record each ask's call stack in a
  diagnostic build and diff the instances' event lists.
- **The tinies' drafters.** `HY4_BF16_TINY` ships no DFly2 tensors; loading them fails. `HY4_STQ_TINY` ships them
  but loads them only with `stream_mtp_head` on (`hy4_plan_test`'s `Drafting` prepare). A draft check must assert
  that the draft ran: a skipped phase passes silently.
- **Reading a test failure:**
  - A test that fails through `testing.expect` prints nothing: the build only says "failed without output". Assert
    with `expectEqual`, or run the test binary directly (`SUSHI_TEST_STEP`, `SUSHI_TEST_SELECT` and the fixture
    variables in the environment), which prints `FAIL (ErrorName)`.
  - `core.refused_readback` holds the stack of a refused readback; symbolize it with `atos` against the test binary.
- **`-Dtest-filter` given twice is an OR across every step.** A broad word ("held") pulls in other steps' cells.
- **The `cooler-requests-fault` step reports memory over its declared 4 GiB.** It already did at the base, before
  stage 2. It is not a failure of yours, but report its peak.
- **Do not trust the tinies for sizes.** Their `d` is 96, so the wide mix's parts and rows forms never run there.
  Shapes come from the dry release model in the audit.

## Tests and fixtures

Fixture variables:
- `source tests/lib/fixture_envs.sh`, then `unset QWEN4_TEST_MODEL`. With it set, `test-exl3-trunk` fails.
- With `M=${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/tiny-models`, set `HY4_BF16_TINY=$M/hy4-bf16-tiny`,
  `HY4_BF16_TINY_STQ=$M/hy4-bf16-tiny-stq` and `HY4_STQ_TINY=$M/hy4-stq-tiny`.
- A skip is not a pass: name the cells that ran.

What to run:
- **Each landing:** the cells its diff can break, by name. For this work that is the `hy4_plan_test` cells:
  - the gate cell;
  - "a dry state of n rows";
  - "a request's state is reserved at its own reach";
  - "a GPU that refuses one form's wide mix";
  - "a dry decode asks what the real decode asks".

  Add the audit, and for any change to the merge, packer or binder the soundness test (`-Dtest-filter="a recorded
  plan replayed at any smaller shape"`), with a mutation check for a new rule.
- **After each sub-stage, both of these:**
  - every Hy4 cell: `zig build test -Doptimize=ReleaseFast -Dtest-filter="hy4"`. At stage 2's head: 253 pass, 3 skip,
    about 6 min.
  - the Cooler contract, with
    `SUSHI_CONTRACT_FIXTURE=HY4_BF16_TINY,HY4_STQ_TINY,DSV41_STQ_TINY,DSV41_STQ_TINY_PM,DSV41_TINY,QWEN4_STQ_TINY,QWEN4_BF16_TINY`
    and `-Dtest-filter="cooler contract"`. At every stage 2 step: 135 pass, 7 skip, 0 fail, about 13 min.
- **Host tier:** `zig build test-host -Doptimize=ReleaseFast`.
- **The release pack's bytes and the gate's timings** are the coordinator's runs.

## Box rules

- You never take the GPU lock (`scripts/gpu-lock.sh`). It is for a quiet run, a recorded timing and a job that could
  run the box out of memory, and the coordinator runs those.
- Check `~/.sushi/runs/QUIET` before every build or test, and wait while it exists.
- Never start `sushi` as a server. Stop only processes you started, and stop them by PID.
- `zig build -Doptimize=ReleaseFast` always, never bare `zig build`. Run builds and tests under `taskpolicy -a`.
- Test first, red for the right reason, and state the red run. A mutant red is only for a test truly written with its
  code; say so when you use one.
- One commit per step, by pathspec, under the owner's identity (the coordinator gives it): a one-line behaviour
  subject and a body with the proof. No `Co-Authored-By` or session trailers. Never push, merge or rebase another
  branch. Nothing goes to agent memory.
- A tracked file names no directory of a box: packs are `${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/<pack>`, run
  outputs `~/.sushi/runs/`.
- Every change lands with its doc: the as-built note in `bill_redesign.md`, and `docs/envvar.md` for any new env var.
- A long job runs detached with an END marker. Wait on the marker or the PID, never `pgrep -f`.

## What to report

After 3a, and after each phase of 3b, under these headings:
- **Built**
- **Tests**: counts, and the red run before each green one.
- **Blueprint contradictions**: quote the design's line and show the code.
- **Not verified**
- **Commits**
- **Open questions**

For 3b, give each phase's arena block (`G`) and host ring (`H`) from the audit beside its `hy4_bill` term, and the
command the coordinator should run for the release pack's bytes and for the fast-bench gate. A finding that the
design is wrong is a result, not a failure: bring it with the smallest case that shows it.
