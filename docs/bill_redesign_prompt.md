# Brief: continue the static bill

You are taking over the static Cooler bill. The design is settled and its pure core is written and tested; the
engine is not wired to it yet. Your job is to wire it, in the order below, without redesigning what exists.

Read first, in this order: `CLAUDE.md` (the rules of this repo bind you: test-first, ReleaseFast, the GPU lock,
commit per step, no push), then [bill_redesign.md](bill_redesign.md) whole. That file is the blueprint; this one
only tells you where to start, what is already decided and where the traps are.

## What this is for

Sushi Cooler's memory bill is a hand-written prediction of what the forward will ask a dynamic pool for. It drifts
every time a kernel changes (78 bill commits in four days), and a drift is a Metal OOM on a user's box. The
redesign makes the bill the allocation itself: a forward's buffers are recorded from the forward's own code by a
shape-only dry run, packed offline, and replayed by offset in one arena fixed at load; state is owned by its
storage and reserved at admission; a ledger sums the owners. Nobody writes a size by hand again, and a plan that
missed something refuses by name instead of corrupting memory.

## Where things stand

- Branch `plan/static-bill-next` on `exp`: the Hy4 line (`hy4-next`, on main) with the bill's commits carried onto it.
  Work in your own clone of it, on a branch of it. The older `plan/static-bill` sits on the Hy4 line from before that
  line was rebased onto main: never build on it. The Hy4 coordinator lands on the Hy4 line itself: never commit in the
  coordinator's clone, never push, and report to the coordinator as `CLAUDE.md` says.
- **Written, host tier, green** (`zig build test-host -Doptimize=ReleaseFast`): `src/tensor/plan/` (frame ids,
  recorder, merge, packer and tables, replay binder, the calls a forward makes, the probe driver) and
  `src/ledger.zig`. [Part 11](bill_redesign.md#built) lists each file, and the table there gives every call the
  engine makes into them.
- **Not written**: anything in `src/tensor/core.zig`, any forward's frames, dry mode, the arena, the state owners,
  admission. No served path calls the new code. Nothing has run on a GPU.
- **Not proven**: that a dry trace equals a real one, and that a real trace is deterministic. Everything rests on
  these two; they are spikes S1 and S2 and they come first.

## Your first assignment

Do these in order. Stop and report after each numbered item; do not start the next stage without the owner's word.

1. **S1 and S2** ([spikes](bill_redesign.md#spikes)), on a throwaway branch. Attach `plan_trace.Recorder` to the
   pool's choke points in real mode (the "recording" column of part 11's table) and get the trace of one Hy4 decode
   forward on the tiny fixture. Then:
   - S2: two runs at one shape give one trace, with pass hand-overs at different points. Check the Hy4, DeepSeek and
     Qwen tinies.
   - S1: the smallest dry mode that runs that decode, and its trace equals the real one: ask for ask (count, bytes,
     order, host flag) and death for death.

   Write what you found into the spikes table of `bill_redesign.md`. **If either fails in a way a marker or a
   classified readback cannot fix, stop: do not work around it.** That result goes to the owner, because it
   overturns the recorded plan.
2. **Stage 1** ([the build](bill_redesign.md#build)), 1a to 1e, one landing each. 1e, the audit, is the first thing
   of value to the owner: the packed peak of each Hy4 phase at the release config beside today's `hy4_bill` term,
   from a dry model in the host tier. Expect it to disagree with today's bill in places; report each disagreement
   with its buffers, do not "fix" the old bill to match.

Stages 2 to 5 follow the doc; 1.8.0 ends at stage 5 (D17): stages 6 to 8 (DeepSeek, Qwen, the bill before load) are
after the release, and nothing before them changes DeepSeek's or Qwen's bytes or bills. Stage 2 is state, before the
arena, for the reason the doc gives.

## What is decided, and what is not yours to decide

- [What may not change](bill_redesign.md#fixed) is eight rules. If you find you need one changed, stop and say
  which and why, with the case that breaks it. Do not add a fallback path "for now".
- [Decisions D1 to D17](bill_redesign.md#decisions) are the owner's, and all are ruled: the plan is recorded (S1
  and S2 passed); one buffer at offsets (S3 could reopen it); a load trusts its own cached
  recording, never a pin; `--prefill-chunk-max` and `--prefill-chunk-min` bound the load's choice and force the
  chunk when equal, for a load on the static bill only (Hy4 for now; there `--prefill-chunk` is deprecated, and on
  every other load it is unchanged and the new flags are refused); a served miss heals in process only within the
  arena held, never by growing it; the stage 8 report is `sushi memory-bill <pack>`; the margin becomes a measured
  constant at each arch's cutover; after S1 and S2 (D8 to D11), a dry host ask holds a host scratch and its filler
  runs, the recorder's levels are the execution's, a readback is a read of bytes the execution's GPU work wrote, and
  dry and real recording are two named modes; after stage 1 (D12 to D16), a phase plans as one only with eager
  appends, the floors are part of the packer, `placedZeros` is a GPU fill, the wide grant is a pipeline's, and the old
  Hy4 bill is left as it is until stage 5 deletes it. Any other flag or subcommand is never yours to add.
- The foundations are a contract, not a draft. You may fix a bug in them (red test first) and you will extend
  them (see below). You may not replace a mechanism with one of your own.

## What the foundations still lack (build these; they are designed)

- Serialization of `plan_pack.Tables` beside the pipeline manifest, under the manifest's key plus arms and settings.
- The engine side of the surface: `into`, `hostResult`, `core.freshOut` (an op's output against its scratch).
- The two-fill check on readbacks, the load probes for arms a first launch latches today, and the registry of
  context boundaries that engine ops with arms declare.
- Levels: the recorder takes them (`Recorder.touch`) and the packer's level rule reads them, but nothing feeds
  them yet. The rule is a heuristic; the binder's marks are the guarantee.
- The miss policy after a served request, the `[plan]` log line, the `/props` ledger block.

## Traps I hit or the review found

- **Any change to the merge, the packer or the binder runs the soundness test**
  (`-Dtest-filter="a recorded plan replayed at any smaller shape"`), and a new rule gets a mutation check: break the
  rule, see the test go red, restore it. That test found three holes in a design that looked right on paper. If it
  passes too easily, the generator is not producing your case: extend the generator first.
- **A host file's stem must not end another file's stem.** The host binary keeps tests by `<stem>.test.`; a file
  named `plan.zig` pulled `prefill_plan`'s MLX tests into the MLX-free link. Keep the `plan_` prefix.
- **Zig is 0.17.1.** `@typeInfo` of a struct has `field_names` and `field_types`, not `fields`; a function that
  must yield a comptime value returns `comptime blk: { ... }`. Read `.zig-toolchain/lib/std` before trusting memory.
- **A git worktree has no toolchain.** Symlink `.zig-toolchain` and `lib/mlx` from the main checkout.
- **A passing test prints nothing**, and a test name has no apostrophe (the build cuts a filter at one).
- **Reuse is invisible to the engine.** `core.dispatchLevel` orders by store, and two tensors on the same bytes are
  two stores. The binder's `after` mark is the only thing that orders them; a pass can be cut anywhere, so never
  treat the mark as an assertion that should not fire. Seed every level map with it, not only the write map.
- **State writes are lazy today.** `mlx_pages.InPlace.write` leaves a pending update that holds a forward's scratch
  and may copy the whole store later. Under the plan that is an escape and a history-dependent ask. It is why
  state comes before the arena.
- **Caches fill while serving** (`exl3/side_cache.zig`, `exl3/linear.zig`'s input ring, about ten users). Each will
  show as an escape in your first recording. They move to the load; they are not bugs in the recorder.
- **`mlx_array_eval` is used as a plain wait** inside forwards. In dry mode it is a no-op; only a read of bytes is
  a readback to classify.
- **`tracePoolAsk` is not a trace**: it is the set of buffers alive at the pool's fullest, unordered. Use the
  recorder.
- **The EXL3 module is vendored by another project** and names no engine. Reach it through `host_side`, as today.
- **Do not trust the tiny fixtures for sizes.** Their 128-wide linears never split and their partials fit one page;
  the classes of error this work removes are invisible there. Shapes come from a dry model at the release config.

## How to work

- Test first, for the right reason; state the red run in your report. You never take the GPU lock
  (`scripts/gpu-lock.sh rule`): it is for a quiet run, a recorded timing and a big job that could run the box out of
  memory, and the coordinator runs those. Host-tier work and the tiny fixtures' GPU tests run beside other work. Check
  `~/.sushi/runs/QUIET` before every build or test and wait while it exists. Run CPU-heavy jobs under `taskpolicy -a`.
- A load of a real pack, a timed run (a speed gate) and anything that holds tens of GB are the coordinator's: ask, with
  the command, and the coordinator runs it or says when the box is free. The coordinator tells you the box's state
  when you start.
- Every landing leaves every arch's bytes unchanged until its cutover, and drops the old bill's term in the commit
  that moves the storage. Until an arch's cutover the pool and the arena coexist.
- A speed gate is arms interleaved in one session on one box, with the commit and settings beside the number
  (`/bench`). The gates are in the doc per stage; "equal or better" is the bar, never a trade.
- Put what you learn in `bill_redesign.md` in the same landing: a spike's answer in its row, a new rule where it
  belongs.
- Fixtures: `source tests/lib/fixture_envs.sh` sets the Qwen and DeepSeek tinies. The Hy4 tests skip unless their
  variables are set: with `M=${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/tiny-models`, `HY4_BF16_TINY=$M/hy4-bf16-tiny`,
  `HY4_BF16_TINY_STQ=$M/hy4-bf16-tiny-stq`, `HY4_STQ_TINY=$M/hy4-stq-tiny` (all pair-major). A skip is not a pass:
  name the cells that ran.

## What to report

After each numbered item: what you built, the tests and their counts, the red run that preceded the green one,
anything in the blueprint the code contradicted (quote the line and show the code), and what you did not verify.
A finding that the design is wrong is a result, not a failure: bring it with the smallest case that shows it.
