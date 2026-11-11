# The engine of our own (`src/tensor/`)

A decode forward that runs on our own Metal queue instead of MLX's. The goal is an engine with no MLX in it, decode
first; DeepSeek-V4.1 STQ is the pilot. It sits behind `src/mlx.zig`, so a model's code is the same on either side:
the stream a call carries says which side runs it.

## Where it serves

One decision at the load, `dsv41_borrow.ownDecode`, which the bills ask too; the load line names the side and, on
MLX, the reason. There is no flag and no fallback: a forward of the engine that fails is the request's error.

| pack and GPU | decode forwards (one row, DSpark draft, verify rows) |
|---|---|
| DeepSeek-V4.1 STQ (experts and trunk), a GPU with matrix units (M5) | the engine |
| DeepSeek-V4.1 STQ, a GPU without matrix units | MLX: the engine's other matmul tiles have not run on one |
| DeepSeek-V4.1 with an FP8 trunk or MXFP4 experts | MLX: the engine has no `quantized_matmul` |
| prefill of any width, every other architecture | MLX |

A DSpark round's verdict, greedy or sampled, is decided on the engine wherever its verify ran there.

## Speed against MLX

Release bench (`tests/bench.sh --tokens 256`, `--ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000`), M5 Max,
AC, DeepSeek-V4.1 STQ 1.75bpw, decode tok/s, median of 3. MLX: the same tree with decode on MLX (the host overhead
patch in), two sessions on 90c83d71 and 537e8e91. Engine: two runs of this change on 84f901ad, one session.

| rung | code MLX | code engine | prose MLX | prose engine |
|---|---:|---:|---:|---:|
| 4k | 99.5 / 101.2 | 97.5 / 97.5 | 55.8 / 55.9 | 55.2 / 55.1 |
| 8k | 101.8 / 102.9 | 97.9 / 102.5 | 60.4 / 59.8 | 59.1 / 58.7 |
| 16k | 79.9 / 80.5 | 77.2 / 76.1 | 51.3 / 51.5 | 51.6 / 51.5 |
| 32k | 89.9 / 90.3 | 86.7 / 86.3 | 52.4 / 52.1 | 51.5 / 51.4 |
| 64k | 85.1 / 83.2 | 83.5 / 82.4 | 52.4 / 49.9 | 49.8 / 49.2 |
| 128k | 74.9 / 75.0 | 77.5 / 75.1 | 54.8 / 54.8 | 50.6 / 49.9 |

- The arms are from different sessions, so a gap under about 4% is not decided: the same MLX arm read 87.2 to
  101.2 at code 4k across sessions. Prose at 128k is behind by 8% in both runs.
- A greedy request gives the same tokens on both sides, and prefill is MLX's on both.
- While decoding, the server takes 19% of one core against 32% (prose) and 10% against 15% (code).
- The round-cost policy picks row counts from measured times, so two runs of one request can take different
  rounds; compare `[spec-stats]` round mixes before reading a tok/s gap as a kernel's.

## The seam

- **The stream decides the side.** An op on `engine.stream()` runs on the engine; an op the engine lacks is refused
  by name (`[engine] <op>: ...`), never handed to MLX. An op on an MLX stream runs on MLX.
- **`engine.active`** is on while a forward of ours runs. An MLX stream inside it is refused by name
  (`an MLX stream inside a forward of the engine`): a stray stream would silently move work back to MLX.
- **`engine.enabled`** turns on at the first load whose decode the engine serves, before that load's first kernel
  is made, and stays on for the process. Off, every call is mlx-c's and no kernel is recorded, so an arch that does
  not use the engine runs the code it ran before; after such a load a later model's calls pass one test of their
  arguments more on the way to mlx-c.
- A handle of ours is a tagged pointer (low bit set); the makers that take no stream (a host array, a vector) make
  engine objects while `active` is on.
- A kernel config is always MLX's, mirrored for the engine as it is filled: a module keeps the configs it builds, and
  one made inside a forward of ours is launched on MLX's side later. A mirror that cannot hold a config refuses the
  launch by name.

## Crossing between the two

- **MLX reads ours (lend).** A tensor of ours goes to an MLX op as an MLX array over the same bytes; our queue is
  waited on first, for the pass that wrote those bytes only. The store stays alive until MLX's last reference goes
  (MLX may drop it on a Metal completion thread: the stores come back through a locked list).
- **We read MLX's (import).** A pass of ours binds MLX's own `MTLBuffer` at the array's offset, with the array's
  strides; the store keeps a handle to the array. A tensor whose bytes MLX made goes back to MLX as MLX's own array.
- **An append is written in place when no one else reads the bytes.** `slice_update` returns a share of its base with
  the update pending; the write happens at first use, into the base's bytes when the store has one holder
  (`core.sole`), else into a copy. For bytes MLX made the test is MLX's own rule for writing in place
  (`_mlx_array_is_donatable`): the handle is the only holder of the array and of its buffer, a command buffer of
  MLX's still in flight counted. A write never goes through a broadcast axis.
- **MLX's queue is kept in use while decode runs on ours** (`dsv41_serving.keepMlxQueue`). The driver drops the
  resident set of a queue left unused for about a second, and the next MLX submission then waits before any work
  runs: a request's end (the commit's first MLX eval) took 0.56 to 0.77 s, a 2 s gap between MLX ops 0.55 s, a
  0.5 s gap 0.2 ms (M5 Max, STQ 1.75bpw, 53+256 tokens, this change on 90c83d71). One synced MLX op per
  `GPU_WARM_TICK_NS` keeps it, the tick the parked inference thread runs between requests.
- The two accessors (`patches/mlxc-metal-buffer.patch`) are temporary aids of the transition; they go when decode no
  longer reads memory that MLX holds.

## Metal rules the engine follows

Each one cost a wrong result or a large slowdown before it was a rule.

- **Bind MLX's buffer, never a second buffer over MLX's memory.** A second no-copy buffer over pages MLX wires counts
  again toward the wired limit; past it every command buffer wires its buffers again (1.4 s per forward at the full
  expert budget).
- **An untracked buffer needs a fence between passes.** Our buffers are untracked, as MLX's are. Each pass waits the
  fence of the pass before it and of any unfinished pass that wrote a store it binds; without that a pass reads
  stale bytes, or never completes.
- **Inside a pass, a barrier where MLX puts one**: before a dispatch that reads what an earlier dispatch of the pass
  wrote, or writes what one read.
- **A host read waits for the pass that wrote the bytes, not for the queue.** Waiting for everything in flight loses
  the layer the forward queued ahead (25.5 against 20.4 ms per forward).
- The pool never hands out a store the open pass binds, and its buffers sit in a residency set of our own.
- **A thread never ends with a pass open** (`core.close`): Metal asserts when the thread's autorelease pool drains
  an encoder still open. A request's end, an unload and the inference loop's exit close it.
- **A pass handed to an idle queue starts 60 to 70 us later, whatever the idle; one handed over behind a pass that
  has not finished starts as that one ends.** So after a host wait the first four dispatches go at once and the
  rest follow in steps that double, without waiting for the GPU to run dry (`core.pass`): waiting for it cost each
  DSpark draft the start latency a second time (0.2 ms a round). Past ~5 ms of idle the latency is ~350 us.
- **A residency set on a queue is paid at every commit, by what it holds.** MLX's sets attached to our queue took a
  forward's CPU from 5.6 to 6.8 ms (its 20 sets) and 7.9 ms (one set). Left off, a bind of an MLX buffer costs
  0.2 us of system CPU per command buffer. Our set holds the pool only.
- **A command buffer costs CPU on two threads**: about 9 us where it is encoded and 15 us on Metal's submit thread,
  more when the host waits for it. Close to half of a decode forward's CPU is those and the binds of MLX's
  buffers, not our code.
- **The host waits on the command buffer, not on a shared event** (`waitUntilCompleted`, oldest pass first so one
  that failed is found before the passes behind it): an event signaled by every pass cost a forward 0.6 of its
  5.6 ms of CPU, on Metal's threads.

## Kernels: what runs, and whether it gives MLX's bytes

Every op is tested against the MLX op it replaces (`src/tensor/route.zig`, `expectSame`: same dtype, shape and bytes).

| op | kernel | against MLX |
|---|---|---|
| reshape, transpose, slice, squeeze, expand, broadcast, view | none (a view) | same bytes |
| astype, contiguous copy | ours | same bytes |
| add, subtract, multiply, divide, comparisons, logical and, where | ours | same bytes |
| maximum, minimum | ours, MLX's NaN rule | same bytes |
| exp, sigmoid, sqrt, rsqrt, square, floor, negative | ours, MLX's precise math and sigmoid form | same bytes |
| concatenate, take, take along an axis, repeat, stack, arange, zeros, slice update | ours | same bytes |
| sum, max over an axis of floats; mean (f32); argmax | MLX's, from `mlx.metallib`, under MLX's reduction plan | same bytes |
| sum, max over an axis of integers | ours | same bytes |
| log | ours, MLX's precise math | same bytes |
| softmax over an inner axis | ours, MLX's op chain | same bytes |
| softmax over the last axis, RMS norm, dequantize | MLX's, from `mlx.metallib` | same bytes |
| matmul: vector, rows against a weight, two matrices, batches, dot product | MLX's, from `mlx.metallib` | same bytes |
| sort, argsort over the last axis | MLX's, from `mlx.metallib`, one block or the multi-block merge under MLX's plan | same bytes, ties in MLX's order |
| cumsum over the last axis (f32, f16, bf16) | MLX's scan kernel on MLX's grid | same bytes |
| put along an axis, constant pad | ours (a copy, then the scatter or the border fill) | same bytes |
| random bits | MLX's `rbitsc`, from `mlx.metallib` | same bytes |
| uniform, categorical | MLX's bits, then `random.cpp`'s op chain op for op: inverse CDF (with MLX's searchsorted) for one row, Gumbel for a block | same bytes |
| a model's custom kernel (`mlx_fast_metal_kernel`) | the model's source, MLX's signature | same bytes |

- **A stock kernel** is fetched from `mlx.metallib` by the name MLX gives it and launched with MLX's buffers,
  constants and grid (`stock.zig`, `matmul.zig`: ports of MLX's `eval_gpu`, MIT, see `NOTICE`). `matmul.zig` follows
  `Matmul::eval_gpu` choice for choice: the wide gemv, the fused and the split-K tiles, NAX and SIMD, picked by the
  GPU's architecture as MLX reads it.
- **A float sum is MLX's kernel under MLX's plan** (`reduce.zig`): MLX sums a long row or column in parallel parts,
  so a sum taken in order is other bytes (a 4096-element row differed in the last bits).
- **A kernel of ours that walks an axis on one thread per output is a regression at vocabulary size**: the first
  argmax here took 7.7 ms a row over the vocabulary, 37 of a DSpark draft's 42 ms, against 1.6 ms for the whole
  Markov chain under MLX. An op over a long axis takes MLX's kernel.
- **Not ported, refused by name**: `quantized_matmul` (a non-STQ DeepSeek pack's FP8 trunk), a matmul or a reduction
  with an empty operand, a reduction of more than 2^31 elements, a sort or a scan over an axis but the last,
  argpartition, top-k, logsumexp, a non-constant pad.
- **Random: no key sequence of our own.** A key made by `mlx_random_key` reaches the engine by import; an unseeded
  draw takes the bits of MLX's next key from MLX (a tiny op on MLX's queue, read back), so a seeded
  `mlx_random_seed` run draws the same numbers on both sides.
- **Only one hardware path is proven on a given box.** On an M5 the tests take the NAX tiles; `MLX_ENABLE_TF32=0`
  sends f32 through the SIMD tiles there. The half-precision SIMD tiles need a run on a pre-M5 machine.

## Memory

- **The pool is the engine's allocator**: a tensor's bytes are a Metal buffer of whole pages, in sizes sixteen to an
  octave, so a buffer is under 1/16 larger than what was asked. A dropped buffer waits in the pool for the next tensor
  of its size.
- **A forward runs under a budget** (`core.budget`): Metal holds for the pool at most the budget beyond what the
  tensors in use ask for, free buffers and size rounding included. A new buffer that would pass it first takes the
  room of free ones, the largest first (waiting for a pass that still binds them); with none left the op is refused
  (`PoolOverBudget`).
- **DeepSeek installs and bills it** (`dsv41_borrow`, `dsv41_bill`): the budget is a decode forward's cap
  (`Config.forwardCap`), set as every forward begins, and the request bill carries that cap a second time, beside
  MLX's. A request's end leaves the pool what the shortest request bills; an unload leaves it nothing free.

## Load

- **A pipeline is compiled per side.** MLX's warm at load compiles MLX's; the engine's are its own, so the load runs
  the engine's decode widths too (`dsv41_model.warmOwn`). Without it the first verify of each width paid its compile
  inside a request (up to 2 s), and the round-cost policy stopped drafting for the rest of that request.
- **A kernel has one signature.** MLX binds an input under 8 elements as a `constant` reference and a longer one
  as a `device` pointer, so one source is two pipelines; ours binds every array as a `device` pointer.
- **The warm runs every shape that names a pipeline** (`warmOwn`, `warmOwnFills`, `warmOwnBlocks`, `warmOwnLong`):
  a verify of every width to the ring's slack (a copy round is wider than the block), each with the routed chains
  queued and waiting, the block indexer at a context it can afford (fewer candidate blocks kept), a draft at every
  fill of the stages' windows (MLX picks a matmul tile and a row reduction by the row count, so a prompt shorter
  than the window is other kernels), and the kernels a step takes from 131072 keys (the one-row index scan on the
  matrix units by lowering `Model.index_nax_keys` for one step, the split top-k on scores of its own).
- **The sampled verdict is warmed** (`dsv41_serving.warmSampledVerdict`, after `warmKernels`): its op chain
  (`sampledDraws`, the function a round calls) runs on the engine's stream over zero logits of `[1, w, vocab]` f32 for
  every verify width, seeded, at the four filter settings (neither, top-k 40, top-p 0.95, both) with a suppress mask. An
  unseeded draw reaches no other pipeline and is not run, since it would advance MLX's key sequence at the load. A
  request's own top-k changes only the sorted widths: the single-block sort variant (`bn`, 32 to 512) of the shortlist,
  the multi-block sort when the chunk maxima or candidates pass 2048, whether the pad runs, and `block_softmax` against
  `looped_softmax` at a shortlist over 4096; a k other than 40 can compile one of those inside a request.
- **A kernel reached only past a threshold of context is found by a bench at that context**, not by the tiny
  fixture: the 128k rung showed three compiles (0.26 s) inside its first decode step.
- **A pipeline made after the warm is logged** (`engine.seal`): `[engine] compiled inside a request (N ms): <name>`
  at warning level, and the request-end debug line counts them. The count is 0 on the release pack.

## What a forward costs

- `[engine] since load:` (debug, at a request's end) and the decode meter's `[fwd-ubench] engine:` line carry
  dispatches, command buffers, waits, imports, the pool, and the GPU's own times: busy, idle between passes, and
  that idle by pause length and by what the host was building (forward, draft, verify, round end).
- **`SUSHI_ENGINE_KERNEL_PROF=1` prices each kernel**: every op's pass is committed and waited for before the next op
  begins, so its command buffer's GPU times are that kernel's (an op of several dispatches is one row, its kernel names
  joined by ` + `). The `[engine-prof] <name>: <n> dispatches, <ms> ms total, <us> us each` lines (debug; the top 40
  by total GPU time and one remainder line) follow `[engine] since load:` and the decode meter's `[fwd-ubench] engine:`
  line, the latter per forward. The forward runs slower under it; read the table's shares, not the forward's wall time.
- **GPU busy and idle are read from the command buffers** (`GPUStartTime`, `GPUEndTime`), which Metal fills after
  the completion signal: a finished buffer is read at a later prune, in order.
- **The round's verdict stays on the engine** (`dsv41_serving.verdictStream`): the masked argmax over the verify's
  logits is our op, so the host waits once for the picks instead of lending the logits to MLX first. A sampled round
  (temperature, top-k, top-p, seeded or not) takes the same stream: `sampler.probsAllPositions` and the draws run on
  the engine, the sampler's code unchanged, and a seeded request draws MLX's tokens.

## Proving a change

- A new op: a case in `route.zig` beside its family, red before the op exists.
- DeepSeek: `dsv41 engine:` tests on the tiny STQ fixture hold one-row decode logits, DSpark drafts, confidences and
  verify logits to MLX's bytes at every verify width, and the tokens at every round-width mix.
- `src/tensor/` is in `build.zig`'s test scan; a new folder under `src/` has to be added there or its tests never
  compile.
- **In a test the engine takes TF32 from what MLX does, not from the environment** (`matmul.mlxTf32`): MLX reads
  `MLX_ENABLE_TF32` once per process, and in a full `zig build test` it ends up off while the variable is unset, so
  the two matmul and DSpark byte tests failed by rounding only there. A server reads the variable as MLX does.
- **A request's wall time is compared on both sides, not only its tok/s**: the decode clock stops at the last token,
  so a wait at the request's end (0.6 s here) showed in no bench column, only in the time to the last stream chunk.
- **A cleanup is checked by the full suite**: a filter compiles only the tests it names, and a type removed as
  unused was still a test's.

## Ruled out

- **Starting the next step before a one-row round's verdict is read.** The next step's embedding and layer 0 were
  queued on the verdict tensor, as MLX's lazy start does for a serial decode. It engaged on about 40 of 100 prose
  rounds (only a round the policy already runs plain predicts a plain successor), verify time per round stayed at
  21.1 to 21.2 ms against 21.0 to 21.1, and prose tok/s stayed inside noise at every rung (M5 Max, AC, STQ 1.75bpw,
  in-process `sushi bench --prose-only --tokens 128`, the change on 537e8e91). The pause between one-row rounds is
  not where prose is won: a one-row verify is 20 to 21 ms on both sides, so it needs less GPU work in the forward.
