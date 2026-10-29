# Process: microbenches (`sushi-ubench`)

A microbench times one kernel or one forward across its rungs. It is a measurement, never a test: it lives in its own
binary, `zig build test` never builds or runs it, and its exit code says whether it measured, never how fast. A test
never asserts on time ([testsuite](testsuite.md)). The release bench gate ([process-measurement](process-measurement.md))
covers the shipped models end to end; a microbench answers the per-kernel question on any model, as ms across rungs.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [process-measurement](process-measurement.md),
[perf-baselines](perf-baselines.md).

## Build and run

```sh
./.zig-toolchain/zig build ubench          # zig-out/bin/sushi-ubench, always ReleaseFast
taskpolicy -a zig-out/bin/sushi-ubench list
taskpolicy -a zig-out/bin/sushi-ubench run dsv41.experts --rungs rows=1,2,4 --rounds 9
zig-out/bin/sushi-ubench compare a.json b.json
```

| flag | meaning |
|---|---|
| `--rungs <axis>=<v,..>` | rungs to time on one axis (repeat per axis); default every rung the bench declares |
| `--rounds N` | same-process rounds per rung (default 7) |
| `--steps N` | dependent calls per round |
| `--arms a,b`, `--ref arm` | arms to time, and the arm ratios are read against |
| `--pack <dir>` | the pack a pack bench loads |
| `--opt <k>=<v>` | a knob the bench declares |
| `--out <dir>` | where the JSON goes (default `~/.sushi/runs/ubench`) |
| `--contended` | skip the quiet-box steps except the lock; the result is stamped CONTENDED |

Exit codes: 0 measured; 1 bench error, invalid run, or `compare` refused; 2 bad flag or value (refused before anything
runs); 3 the box refused the run.

## The quiet box, taken by the binary

1. Refuse a binary older than HEAD or built from a dirty tree (untracked files do not count).
2. Refuse below `taskpolicy -a` priority, or while a `sushi` server or another GPU tool runs.
3. Take the GPU lock as `ubench:<bench>`, then touch `~/.sushi/runs/QUIET` and wait until no `zig`, `clang` or `ld` runs.
4. Refuse on battery.
5. Fans to max (renewed every 5 min), idle until the hottest die sensor is under 50 °C (at most 180 s), run, fans back
   to auto, release the lock, remove QUIET.

A `--contended` number is a smoke, never a recorded baseline.

## Reading a result

- One table per bench: rows are rungs, columns are arms. A cell is the median (min-max) of N rounds in one process,
  the arm order rotated each round, plus the ratio to the reference arm and the bench's metric columns (MB read, GB/s,
  share of roofline).
- The header stamps commit, binary sha and mtime, QoS, power source, die temperature and the lock holder.
- `compare` intersects rungs and arms of two results of one bench and marks overlapping ranges `~` (noise, not a
  verdict). It warns when chip, QoS, dirty state or rounds x steps differ, and refuses a contended-vs-quiet pair.
- Record a number the [process-measurement](process-measurement.md) way: commit and settings beside it, the JSON path
  in the private raw-file index.

## Adding a bench

- A bench is a `report.Bench` entry in the registry (`src/ubench_main.zig`): name `<scope>.<what>` (scope `qwen4`,
  `mimo2`, `glm5`, `dsv41`, `enc` or `shared`), kind (kernel on synthetic tensors, or pack), unit, steps, axes, arms,
  reference arm, opts, and a body.
- The body calls `run.interleave(label, ctx, f)` per rung, then `run.metric` / `run.invalid`. Chains of dependent calls
  come from `src/exl3/bench_util.zig`, so launch latency counts as it does in a forward.
- Bench files (`src/ubench_*.zig`, `src/exl3/ubench.zig`, `src/encoder/ubench.zig`) hold no `test` block: the build
  refuses one. A bench's correctness half (bytes against an oracle) stays a test beside the code.
- Knobs are flags (`--opt`), never env vars.

| bench | scope | what it times |
|---|---|---|
| `dsv41.experts` | dsv41 | the 384-expert PAIR1 grouped chain at the plan's layers 2, 5, 23 and the DSpark stage bank, by rows and routing mode |
| `dsv41.index_decode` | dsv41 | a decode block's indexer stages over packed keys (scores, mask applied, top-512, candidate-source blocks, scores + top-k), by keys and rows |
| `dsv41.index_peak` | dsv41 | one wide span's indexer, whole scores against the streamed top-k, by context; MLX peak beyond the inputs in MB, reported as the lowest and the highest over the rounds (when freed buffers are released varies from run to run, so the floor and the worst case differ; a bill covers the worst) |
| `dsv41.index_block_peak` | dsv41 | a 16-row decode block's indexer with and without the candidate mask, by context; peak MB (min and max) |
| `dsv41.rows_growth` | dsv41 | a full row store growing by one chunk: time, held MB and the peak beyond them (min and max) |
| `dsv41.attn_span` | dsv41 | one attention span at the release geometry: composed core, f16 wide arm, fused kernel; peak MB (min and max) |
| `dsv41.topk` | dsv41 | the decode top-k over index scores, one threadgroup a row against the served split, per row of work in one eval of one or eight |
