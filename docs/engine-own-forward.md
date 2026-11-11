# The engine of our own (`src/tensor/`)

The engine of our own ships as **Sushi Cooler**: users read that name in the settings page, `--help`, the load line and
the release notes; the code and these internal docs keep their names.

A decode forward that runs on our own Metal queue instead of MLX's. The goal is an engine with no MLX in it, decode
first; DeepSeek-V4.1 STQ is the pilot. It sits behind `src/mlx.zig`, so a model's code is the same on either side:
the stream a call carries says which side runs it.

## Where it serves

One decision at the load, `dsv41_borrow.ownDecode`, which the bills ask too; the load line names the side and, on
MLX, the reason. A user opts out of the engine up front, never per forward: `--ds41-mlx-engine` or the model's
`ds41_cooler: false` (the settings page's "Use Sushi's Cooler Engine instead of MLX" checkbox, ticked by default) moves
every forward of the table below to MLX at the next load (`DecodeSide.mlx_by_flag` / `.mlx_by_setting`); the flag
outranks the file. There is no fallback after a start: a forward of the engine that fails is the request's error.

Prefill follows decode (`Model.engine_prefill`, set at the load): a pack that decodes on the engine prefills there, a
chunk of any width included, and a pack that decodes on MLX prefills on MLX.

| pack and GPU | forwards on the engine |
|---|---|
| DeepSeek-V4.1 STQ (experts and trunk), a GPU with matrix units (M5) | every forward: decode, DSpark draft and verify, every prefill chunk, wider than 16 rows or not, with image rows or without |
| DeepSeek-V4.1 STQ, a GPU without matrix units | none: the engine's other matmul tiles have not run on one |
| DeepSeek-V4.1 with an FP8 trunk or MXFP4 experts | none: the engine runs neither |
| every other architecture | none: MLX |

The vision tower of an image request (once per image, on a load that decodes on the engine) runs on the engine whole:
the table's `quantized_matmul`, `fast::scaled_dot_product_attention` and `erf` rows. Its load-time requantization
(bf16 to affine 8-bit in groups of 128) is the engine's too (`stock.quantize`: MLX's own `affine_quantize` kernel,
so MLX's codes, scales and biases), and the weights it makes are the engine's tensors, not imports.

A forward that splices image rows runs on the engine like any other, and so does one issued while a DSpark verify is
open (only after a round failed): the MLX route's logits, byte for byte. An image row takes the non-queued route (a
routing bias read row by row, the Engram write masked by row), whose pipelines a load with the image bias warms at
every decode width and at a prefill width (`dsv41_model.warmOwnImages`).

What a wide chunk still asks of MLX: the keep-alive op of MLX's queue. The slabs, their views and the union workspace
of a spilled route are the engine's own on a Cooler load.

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
- A greedy request gives the same tokens on both sides, and prefill was MLX's on both in this bench.
- While decoding, the server takes 19% of one core against 32% (prose) and 10% against 15% (code).
- The round-cost policy picks row counts from measured times, so two runs of one request can take different
  rounds; compare `[spec-stats]` round mixes before reading a tok/s gap as a kernel's.

## A prefill chunk on the engine

A chunk of any width goes through the engine (`forwardDsv41With`), a state-only chunk (`extendState`) included. On the tiny STQ fixture a chunk of 17 to 513
rows, a prompt cut in chunks with restore points between and state-only chunks, and the decode after it give MLX's
restore-point digests, last-row logits and tokens. What the wide path reached that decode does not: `abs`, `logaddexp`
(the router's softplus) and an unsigned `floor_divide` (the routed slots' rows). The pack's indexer `weights_proj` is
dense, so `quantized_matmul` is not reached on an STQ pack; an FP8-trunk fixture stops there.

A wide chunk whose experts stream (cache below the pack) cannot hand the GPU the engine's whole open pass at a rate
group's start (`EarlyPlanes.start`, the MLX arm's way): the engine encodes each op into the pass as it is called, so the
groups not yet filled would go with it, where MLX evaluates only the planes asked for. The test with a stalled fill on a
three-expert cache (`DSV41_STQ_TINY`) is the guard; a cache holding every expert never reaches the route.

The engine enforces the rule: a slab being filled is marked (`fill_marks`, set at `PageSlab.tryBeginFill`, cleared at
`publish`), and `commitPass` drops a pass that binds any marked bytes and fails with `ReadsFillingSlab`. With no fill in
flight the check is one atomic load per commit; with one (the prefetcher's) it is one lock and a compare per bound
range. The bytes compared are the ones a bind reads: a slab the engine wraps is one store that all its views share, so
a pass records the range of each view it binds (`pass_mapped`, `Tensor.reach`), and a fill of one part of a slab does
not refuse a pass that reads another. A refused pass never runs, so the buffers it bound stop waiting for it
(`refuseFillingReads` sets their last pass back by one): left waiting, a freed buffer could not leave the pool until a
later pass ran, and a request whose first buffer needed that room was refused for good.

The engine's wide chunk reads its experts group by group (`rate_groups.moePrefillSharedFusedStaged` and
`moePrefillStaged`, beside the MLX arm's `...Early` functions, which are untouched; `StqStream.applyScored`):
`Engine.stageScales` first, then `stageGroup(g)` just before the chain encodes group g's gate and up planes, which it
hands to the GPU at once, so the GPU runs group g while the host reads group g+1; `stageEnd` publishes and leases. The
route marks the ranges each stage writes (`Staged.marks`, one covering range per stage and slab, in `fill_marks`)
instead of the slabs whole, so the guard lets a pass bind a group that has landed and still refuses one that binds a
group still being read. `StqStream.stage_on_engine` (test only) restores the old order, which the guard refuses.

The load also runs the wide forwards a prompt reaches (`warmWide`: a full-width state-only and head chunk, head chunks
whole, part and not at all on the GEMM tiles, the held-back tail, one chunk per pick-sort size and one per
compressed-row count under 17), so a request compiles no kernel; MLX's own prefill warm stays beside it, for the image
forwards.

A wide chunk on the engine bills the one `Config.forwardCap` it always did, split between the two sides
(`forwardCapMlx`: the state rows MLX appends and 64 MB; the engine's pool takes the rest), so MLX's peak above its
start plus the pool's held peak stays under the bill. `serve` installs the caps after the load, so the load's warm
gives every forward of the engine, decode-width or wide, the engine's share a request's forward has (`warm_budget`,
`enterWarm`, the same `splitOf` as serve): a forward the bill does not cover fails the load with `PoolOverBudget`, not a
request with a Metal OOM, and the budget is lifted when the warm ends.

A ranged route builds its bank on the engine once stage 0 has landed (`Engine.stageBank`, called through
`FillStages.scales`): the union's slab views (`rate_groups.rowsView` slice and reshape) are engine ops over the slabs,
and the scale concatenation (`concatAxis0`) is one dispatch that reads the landed scales. The old
route and the MLX arm still build the bank at prepare time.

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
- A kernel, its config and its lists of names made while the engine is on are objects of the engine's (tagged
  handles), so making and filling them calls nothing in mlx-c. A module keeps the kernels and configs it builds and
  may launch one on MLX's side later: MLX's kernel and config (the twins) are made at that first launch, the config
  filled in the order its maker filled it, and a config's twin is dropped when its outputs or template values change
  (a new grid or threadgroup is passed on to it). A config that cannot hold an output or a template argument refuses
  it by name. A kernel or config MLX made (the engine was off; a test turned it on later) is still mirrored as filled.
- **A launch finds its pipeline without building the instance key** (`[engine] since load:` counts the keys built;
  a decode forward, a draft and a verify round build none once they have run). Two layers: a mirrored config that
  outlives its launch remembers its last pipeline (`Config.last`; cleared by `addOutput` and `addTemplate`, checked
  against the kernel's never-reused id and the inputs' dtypes and passing), and each kernel keeps its last 16
  instantiations, compared field by field (template values, output dtypes, input signature), for the configs the
  DeepSeek kernels make anew on every launch. A launch with more than 16 template arguments takes the key every
  time. A new way to change a mirrored config must clear `last` too.
- **`mlx.calls` counts every entry into mlx-c from Zig**, the measure of the move off MLX: `mlx.engine.mlxCalls()`
  is the `; N MLX calls` that ends the `[engine] since load:` line. Each mlx-c symbol is declared once in `src/mlx.zig`
  through `counted` (or `routed`, on its MLX branch), which is one relaxed atomic add and the call; atomic because
  the gauge sampler and request threads call MLX beside the inference thread. A call the seam answers on the engine
  is not counted; a call the engine itself makes into MLX (an import, a retired import's free) is. An empty
  handle (`mlx_array_new`, `mlx_stream_new`, `mlx_device_new`, `mlx_string_new`, `mlx_device_info_new`) is a zero
  `ctx` made in Zig, no call. It cannot see MLX's own threads, work before the load, or the Metal objects MLX already
  made. Another file declaring an `mlx_*` extern fails the `mlx-extern` lint; an address of the library for `dladdr`
  is `mlx.libraryAnchor()`, never a wrapper's address. A one-row STQ decode step on the tiny fixture makes no call
  into mlx-c (pinned in `dsv41_generate_test.zig`).
- **What a Cooler request still asks of MLX is pinned phase by phase** (the census test in `dsv41_generate_test.zig`:
  the load, a greedy and a seeded request in rounds, serial steps with logprobs, a restore point captured and
  restored, a step after it, an image prompt, the unload). A change that adds a call fails it; a stage of the move off
  MLX lowers a row; the target of every row is zero. `SUSHI_DSV41_CENSUS_TRACE=1` prints a phase's calls by entry
  point. Every phase of a request makes no call at all: the rounds and the serial steps through the generator, a
  restore point captured and restored, in RAM and through the SSD tier (its own test, with the restored state
  compared byte for byte), a step after it and an image prompt. What a caller makes for a forward is of the side the
  forward's output is on (`Trunk.outStream`): the generator's token arrays (`generate.idsArray`), the slice of the
  prefill's last logits, the reserved-token mask the sampler applies every round (`sampler.installSuppressMask`; made
  on MLX it was imported once a round, a vocabulary of bools each time). A model that forwards on the engine asks MLX
  for no pool clear either (`generate.clearPool`). The free of an empty handle is no call (`mlx_array_free` in the
  seam): mlx-c holds nothing for one.
  The load itself (the weights and the model over them, not the fixture's own reads) makes 25 calls: the shared trunk
  shell's six scalar constants, cast on MLX's stream, and one touch of MLX's queue.
  A phase's ops are exact. Its calls are held under a ceiling and within a twentieth of it, so a real cut moves the
  ceiling. The census counts a load that is not the process's first (it opens and closes the pack once before): what
  a process asks once (the device's description, a kernel probe) is its first load's, and with that kept out a row
  reads the same alone and after other tests. Dispatches still follow what the process ran before (a 12-token request
  read 7,336 as a process's only load and 7,017 after other loads).
- **A Cooler load runs no forward on MLX.** The load's warm used to run three prefill chunks through MLX whatever
  the decode side; they only built the state the engine's own warms start from. Where prefill is the engine's they
  run on the engine, and with the pipelines read from a manifest they do not run at all.

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
- **A request's rings are made on the engine's stream on a Cooler load** (`dsv41_model.newState`). A ring of MLX's
  zeros met the engine's first step while MLX's own command buffer could still be in flight, so that step copied the
  ring into the pool or wrote it in place by timing, and an in-place one asked MLX who holds it at every later step.
  No forward of a Cooler load runs on MLX.
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
- The pool never hands out a store the open pass binds.
- **Residency is kept the way MLX keeps it** (`metal.zig`, after mlx/backend/metal/resident.cpp): every buffer of
  ours sits in a residency set attached to our queue with a standing request, and the sets are capped at 5% of the
  working set (floor 64 MiB, at most 32, first fit; a buffer over the cap has a set of its own). The cap bounds what
  one lost set costs to wire again: macOS decides residency a set at a time. A set attached to no queue loses its
  residency within half a minute even with a standing request, and so does a queue that runs nothing: while the
  engine wraps mappings, the warm tick runs a four-byte pass on our queue too (`core.touch`, from `mlx.gpuTouch`).
- **A slab is unmapped only after the engine has waited.** A slab's buffer outlives the tensor that held it until the
  last pass that binds it has run, and `SlabOperand.destroy` closes the engine first (`mlx.engine.close()`, a no-op
  while `enabled` is off, so no other architecture reaches it). A slab MLX wraps gets 32 tries of MLX's wait, then
  stays mapped for good; a slab the engine wraps is given back by the store's own release and asks MLX nothing.
- **A Cooler load's expert slabs, slot maps and rate tables are the engine's** (`expert_io.importHostBytes` with
  `ImportOptions.engine`, `SlabOperand.create*`, `LayerState.refreshSpec`, `rate_groups.metaTable`). The load gives the
  expert stream the engine's stream, and what the stream makes follows the stream it holds: a slab is wrapped like a
  weight (`core.wrapMapped`, counted in `mapped_bytes`, in our residency sets), its views are engine ops, and a map or
  a table is a host array made under `engine.enter(s)`. On MLX's stream every one of them is MLX's, as before. On the
  tiny STQ fixture a 12-token request fell from 19,532 calls into mlx-c to 146 and the load from 107,984 to 11,059,
  with the same dispatches. Live (this change on 66f98656, STQ 1.75bpw, `--ssd-budget-gb 108`, M5 Max, fans at max,
  GPU lock, one load a side against b5a0ee9b, same replies): a 20k-token prefill 697,799 calls to 42,877 at 126,611
  dispatches against 125,610, and 1,335 tok/s against 650 to 855 on the earlier trees, because the host no longer
  waits on an MLX evaluation of every slab view; decode tok/s unchanged; a new binary's load 30.9 s against 44.4 s.
- **An array cached for the process belongs to one side** (`rate_groups.unitScore`): a constant made once and kept in
  a global is MLX's or the engine's by whoever made it first, and the other side then imports or lends it at every
  launch. A cached array is kept per side, chosen by the stream of the op that takes it.
- **A route is a load's for its life**: no load mixes an MLX forward with the engine's weights or slabs. A test that
  wants MLX's bytes loads the pack on the MLX route (`EngineFixture.openMlx`), takes them and closes it before the
  Cooler load opens. One forward of the load's warm had stayed on MLX (the context chunk of `warmOwnBlocks`, 536 rows
  on the 1.75bpw pack; the tiny fixture's geometry skips that step): with the slabs the engine's it built its bank on
  the engine while its fill was in flight, the guard refused it and the whole warm was dropped, so every kernel
  compiled inside the first requests. It runs on the engine now, and the order a wide route reads its fill in follows
  the stream its chain is encoded on (`StqStream.applyScored`), not a scope flag a caller may not have set;
  `StqStream.mlx_order_routes` counts the routes read in MLX's order, none on a Cooler load.
- **A fill refusal names its bytes** (`core.refused_fill`): the message gives the pass's dispatches, the bytes one of
  them binds and the range the fill is writing, which is what told a scale slab's first rows from a whole-slab fill.
- **A thread never ends with a pass open** (`core.close`): Metal asserts when the thread's autorelease pool drains
  an encoder still open. A request's end, an unload and the inference loop's exit close it.
- **A pass handed to an idle queue starts 60 to 70 us later, whatever the idle; one handed over behind a pass that
  has not finished starts as that one ends.** So after a host wait the first four dispatches go at once and the
  rest follow in steps that double, without waiting for the GPU to run dry (`core.pass`): waiting for it cost each
  DSpark draft the start latency a second time (0.2 ms a round). Past ~5 ms of idle the latency is ~350 us.
- **MLX's residency sets attached to a second queue cost every commit on it**: with them on our queue a forward's
  CPU went from 5.6 to 6.8 ms (its 20 sets) and 7.9 ms (one set). Left off, a bind of an MLX buffer costs 0.2 us of
  system CPU per command buffer. A set of our own on our own queue costs nothing per commit, whatever it holds: a
  decode forward's process CPU read 5.181 ms with the weights and slabs in MLX's sets and 5.170 ms with all 111.6 GB
  of them in one set on our queue (same binary, same reply).
- **What a lost set costs** (exp branch `exp-e1-residency` 4aa24b0c on main aad856bc, M5 Max 128 GB, macOS 27.0, GPU
  lock; DeepSeek-V4.1-Flash STQ 1.75bpw, `--ssd-budget-gb 108`, and a probe of 9.49 GB in 608 no-copy regions): a
  set on no queue, or on a queue with no 500 ms tick, lost its residency within 30 s; wiring again takes 6 to 7 ms
  a GB on the committing thread; after 120 s idle the next short request paid 0.75 s more with MLX's twenty-odd
  sets of 5% and 1.61 s more with one set of everything, 2.1 times slower. Hence the cap.
- **A Cooler load's weights are the engine's own buffers** (`core.wrapMapped`, `metal.Device.wrap`; decided with
  the decode side, `dsv41_borrow.ownDecode`). Every large DeepSeek weight is a page-aligned mapping sushi fills and
  hands over without a copy at one place, `model.wrapRegion`; there the engine wraps it (a no-copy Metal buffer of
  the tensor's bytes, in our residency sets) instead of MLX, and the load binds, slices and casts the weights on the
  engine's stream. A tensor under the mapping size (`model.import_min_bytes`, 1 MiB) is a copy either way; on a
  Cooler load the copy is a pool buffer of the engine's (`readTensorArray`), where an MLX array was imported at every
  load-time use and freed through MLX at the unload (tiny STQ fixture: the load 11,059 calls into mlx-c to 2,517, the
  unload 485 to 103). The store gives the mapping back once the last pass that binds it has run, so an unload needs no
  MLX wait; it is never written in place (`sole` is false). On the tiny STQ fixture that took a one-row step from
  5497 calls into mlx-c to 1366 and a 12-token request from 94,508 to 19,602; live, STQ 1.75bpw, a request made 3
  to 4 times fewer, with the same dispatches, waits and imports of slabs.
- **A lend of a wrapped weight lasts one call** (`core.lendsOnce`). MLX counts and wires every buffer it wraps for as
  long as it holds it, and a tensor keeps the array a lend made (its twin): for a weight that was the whole load,
  so one load-time op (the tower's requantization, a widening) left 4.6 GB billed and wired twice, and the first
  request released expert cache to fit. A wrapped mapping keeps no twin; the seam frees the lend when the MLX call
  that took it returns. `Stats.mapped_lends` counts them, a served load names each lending op once
  (`mlx.noteWeightLend`), and no request makes one.
- **Pool buffers and wrapped mappings never share a residency set** (`metal.Kind`): the pool's buffers come and go
  with every forward's budget, and a set that also listed gigabytes of weights was dirtied by each of them.
- **A load step moved to the engine's stream is checked by its decisions and its counts, never only by its bytes.**
  The fused hyper-connection kernel's load probe built inputs with MLX's random; on the engine's stream that was
  refused, the probe swallowed the refusal and read it as "this GPU's matmul differs": same text, 12 to 25% more
  dispatches a request. The probe's inputs are host-made, and a probe that cannot run says why. Compare
  `[dsv41] fused kernels ..., hc expand kernel ...` and a request's dispatch count with the MLX route's.
- **A release is told to its set at once when no pass follows** (`metal.commitResident`, from the engine's sync, a
  trim, `close` and the warm tick): a set that still lists a released buffer keeps its pages wired, and after an
  unload no pass of ours may come. **MLX's wired limit gives up what the engine wires** (`mlx.applyWiredPolicy`
  subtracts `mappedTotal`), so the two sets of residency together stay under the working set.
- **The pool is the engine's allocator**: a tensor's bytes are a Metal buffer of whole pages, in sizes sixteen to an
  octave, so a buffer is under 1/16 larger than what was asked. A dropped buffer waits in the pool for the next tensor
  of its size.
- **The pool's bytes are published for other threads**: `core.poolTotal` and `core.poolFree` mirror `Stats.pool_bytes` and
  `pool_free` as atomics, stored where the pool changes; `gpu_mem` adds them to MLX's counters
  ([engine-memory-admission](engine-memory-admission.md#gpu-memory-readers)).
- **A forward runs under a budget** (`core.budget`): Metal holds for the pool at most the budget beyond what the
  tensors in use ask for, free buffers and size rounding included. A new buffer that would pass it first takes the
  room of free ones, the largest first (waiting for a pass that still binds them); with none left the op is refused
  (`PoolOverBudget`).
- **What is made between forwards and outlives them asks for room** (`core.room`): a restore point's copies are taken
  under the cap the last forward left, which covers a forward's scratch and not state that stays. The capture names
  the sizes it is about to copy and the cap rises by their buffers; the next forward's cap starts from what is in use
  then. A restore point's absent slots share one placeholder. The vision tower's requantized codes, scales and biases
  are made the same way, after the model's warm left its cap (`dsv41_vision.lin`). State read back from the SSD tier
  has no sizes to name ahead: the cap is lifted for the read and the install and comes back raised by the buffers
  that came into use (`core.uncapped`, `core.capped`).
- **A scope of the engine's is given back as it was found.** `engine.enter(s)` and `Scope.end` nest; a hand-set
  `active = false` at the end of a callee closed its caller's scope, and every host array the caller made after it
  was MLX's (the wide warm's token arrays, a prefill chunk's scalars). A forward's scope counts its depth on the model
  (`ownBegin`, `ownEnd`), and DeepSeek's `Scope` makes a scalar or a host array on the side of its own stream,
  whether or not a forward is open.
- **Maps are the engine's inside its scope** (`route.zig`: names to arrays, names to strings, the iterator): a map
  made under `engine.enter` is tagged like a vector, every call that takes one follows it, and a name it lacks
  answers 2 with no error latched, as mlx-c does. The SSD tier's readers hand tensors around in them.
- **DeepSeek installs and bills it** (`dsv41_borrow`, `dsv41_bill`): the budget is a decode forward's cap
  (`Config.forwardCap`), set as every forward begins, and the request bill carries that cap a second time, beside
  MLX's. A request's end leaves the pool what the shortest request bills; an unload leaves it nothing free.

## Load

- **A pipeline is compiled per side.** MLX's warm at load compiles MLX's; the engine's are its own, so the load runs
  the engine's decode widths too (`dsv41_model.warmOwn`). Without it the first verify of each width paid its compile
  inside a request (up to 2 s), and the round-cost policy stopped drafting for the rest of that request.
- **With decode on the engine the load does not warm MLX's widths 1 to 16.** No forward of 16 rows or fewer is MLX's
  any more (see the table), so `warmKernels` skips MLX's decode widths and DSpark rounds and keeps MLX's prefill
  (256, 42 and 31 rows), which the image forwards still run. An MLX-decode load warms as before.
- **The vision tower warms itself on a first load only** (`Dsv41Vision.warmEngine`, from `VisionEncoder.init`, after the
  model's seal): one block over five patch counts, the seal lifted for its span. Its recipes ride the manifest with a
  `marker` recipe (`vision tower`, keyed by the tower's widths) that says the warm is whole; a same-key load makes the
  recipes with the rest, trusts the marker, and runs no tower warm. A load that ran the warm writes the manifest again
  (`dsv41_pipelines.rewriteLate`). A manifest whose recipes did not all make drops its markers. The tower's load is
  inside the lifted seal too (`Dsv41Vision.init`): its requantization makes one stock kernel, whose recipe rides the
  manifest with the warm's.
- **Every warm forward of the engine runs under a request's budget** (`dsv41_model.warmBudget`, before each decode-width
  forward; `warmWide` through `forwardDsv41With`). The decode cap follows the position the forward is at, so the
  long-context steps of `warmOwnBlocks` and `warmOwnLong` are billed as a request there is.
- **The NAX GEMM kernel is proven before the engine warm** (`kernels.prepareGemmNax`, called by `initDsv41` for the
  experts' and the trunk's decode). Its proof is MLX work of its own, which a forward of the engine refuses; first
  reached inside one, it was declined for the process (`[exl3-gemm] NAX arm declined`) and every GEMM, prefill
  included, ran the SIMD arm at a third of the speed.
- **A failed launch declines a kernel arm only before that arm has ever worked** (`kernels.gemm_nax_worked`). After
  one launch has run, a failure is the launch's own (a pass the engine refuses by name, memory) and is returned: taken
  as "the arm is unavailable" it switched the whole process to another arm, with other bytes, without a word.
- **Engine pipeline creation is 0.1 s with the system's shader cache warm and 9 to 16 s cold**, all of it
  `newLibraryWithSource` (435 pipelines, M5 Max); the pipeline state is 0.03 to 0.06 s. A binary archive replaces only
  that state step, so it buys nothing here. A binary the system has not run is cold again, and the
  manifest's recipes then compile on 8 threads in about 2.5 s ([the manifest](#the-pipeline-manifest)).
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
  every verify width, unseeded, at the four filter settings (neither, top-k 40, top-p 0.95, both) with a suppress mask;
  a seeded request runs the same chain and differs only in its root. A
  request's own top-k changes only the sorted widths: the single-block sort variant (`bn`, 32 to 512) of the shortlist,
  the multi-block sort when the chunk maxima or candidates pass 2048, whether the pad runs, and `block_softmax` against
  `looped_softmax` at a shortlist over 4096; a k other than 40 can compile one of those inside a request.
- **The serial sampler is warmed** (`warmSerialSampler`, with the sampled verdict): a first-token or serial draw at the
  four filter settings, greedy, with and without the suppress mask, the three penalty mixes with logprobs, a grammar mask,
  and a verify block of every width under the penalties, the thinking penalty with and without a bias, and a row mask.
- **A kernel reached only past a threshold of context is found by a bench at that context**, not by the tiny
  fixture: the 128k rung showed three compiles (0.26 s) inside its first decode step.
- **A pipeline made after the warm is logged** (`engine.seal`): `[engine] compiled inside a request (N ms): <name>`
  at warning level, and the request-end debug line counts them. The count is 0 on the release pack.

### The pipeline manifest

A complete warm leaves `~/.sushi/cache/pipelines/<scope>-<build>/manifest.bin` (`src/pipeline_manifest.zig`): the recipe
of each engine pipeline the process made (`tensor/pipeline_registry.zig`, recorded at the three places a pipeline is
created: `stock.kernel`, `stock.kernelFlags`, `stock.kernelConstants`, `core.pipelineFromSource`). A recipe is the
metallib function and its boolean and int constants, or the exact source text and entry point; a source recipe is the
identity of its pipeline, so a stale recipe cannot answer for another source. A `marker` recipe makes no pipeline: it
says that a warm run after the load's own (the vision tower's) is whole in the file. The file ends in a SHA-256 of
itself; a short, changed or foreign file is refused by name and never read partly. The format number is in the file
and in the build key, so a manifest of an older format is never read: its directory is another key's, and the load
rebuilds one.

- **Key.** `scope` (16 hex) is the pack (its config.json, its index, the name and size of each shard: not their bytes) and
  what shapes the warm (context, window, YaRN factor, candidate blocks, DSpark, tile order, expert-cache and SSD bytes,
  resident or streamed, dropped experts, and the `recipe_env` variables of `dsv41_pipelines.zig`). `build` (16 hex) is the
  SHA-256 of the running binary, the size and mtime of `mlx.metallib`, the GPU architecture, the OS build and the manifest
  version. A diagnostic that changes no recipe (the kernel profile, the report, probe and profile switches) is not in it.
- **Same key.** The recipes are made on up to 8 worker threads (our own Metal binding only, never MLX), after the model is built;
  the inference thread then installs them. A launch finds its pipeline
  by source or key, counts it as preloaded (`core.pipelines_preloaded`), not compiled and not late. The engine's warm
  forwards, the wide chunks included, are skipped (`WarmScope.mlx`: MLX's prefill warm and its spilled-route chain stay); the NAX proof and the NAX probes already run at init.
  A recipe the manifest lacks compiles in the request and shows in `kernels compiled after the load`.
- **Another build of the same scope.** The newest such manifest is made the same way, then the full warm runs as the
  complete enumeration and the manifest is rewritten; the older builds of the scope are removed and the log says so.
- **No manifest, or one that cannot be read** (the reason is named): today's load, then a manifest is written. A recipe that
  fails to make from a same-key manifest makes it untrusted: the full warm runs.
- Written only after a warm that returned without error, through a temp file and a rename. One `[pipelines]` line says
  which case ran.

## What a forward costs

- `[engine] since load:` (debug, at a request's end) and the decode meter's `[fwd-ubench] engine:` line carry
  dispatches, command buffers, waits, imports, the pool, and the GPU's own times: busy, idle between passes, and
  that idle by pause length and by what the host was building (forward, draft, verify, round end).
- **`SUSHI_ENGINE_KERNEL_PROF=1` prices each kernel**: every op's pass is committed and waited for before the next op
  begins, so its command buffer's GPU times are that kernel's (an op of several dispatches is one row, its kernel names
  joined by ` + `). The `[engine-prof] <name>: <n> dispatches, <ms> ms total, <us> us each` lines (debug; the top 40
  by total GPU time and one remainder line) follow `[engine] since load:` and the decode meter's `[fwd-ubench] engine:`
  line, the latter per forward. The forward runs slower under it; read the table's shares, not the forward's wall time.
- **Dispatches a forward, one row at 4096 keys, M5 Max, STQ 1.75bpw** (`SUSHI_DECODE_FWD_UBENCH` S=1 and S=3,
  `SUSHI_ENGINE_KERNEL_PROF=1`, `--ssd-budget-gb 108`, 100 forwards, one load each): 1140 on 97995014, 1043 with three glue ops folded into the kernel before them
  (slots of the picks in the router's launch, the window row's ring write in `dsv41_rope_kv_ring`, the index picks' mask in the
  top-k); three rows 1292 and 1154. The profile's GPU ms differ 3.6 times between loads by the clock state the idle gaps leave the GPU in
  (the untouched `fz_down` kernel read 207, 144 and 57 us in three loads), so compare kernels by their time scaled to an untouched
  kernel's, not across loads: the glue removed at one row is `sushi_take` 0.39 ms, `sushi_copy_into` 0.32, `sushi_bin_less` +
  `sushi_where` 0.16 in the base load's clock, and the three fused kernels cost what their first halves did (route 1.080 against 1.079 ms,
  rope 0.632 against 0.669, top-k 0.253 against 0.256).
- **At 131,072 keys the forward grows by the index scan alone** (the same tree with the three folds, same settings, `SUSHI_DECODE_FWD_UBENCH_KV=131072`,
  S=1 and 3 in one load, 60 forwards): 1071 and 1172 dispatches against 1043 and 1154. The attention kernels (`attn_split`, `attn_merge`)
  hold their share (7.7% to 6.2% at one row, 8.8% to 9.0% at three: window plus top-k rows, not keys); the index family
  (`index_scores_*`, `topk*`, `cand_keys`, `pack_fp4`) goes from 2.1 to 5.2% at one row and 2.5 to 4.9% at three, a candidate source
  scanning in full and four later layers scoring only its blocks through `index_scores_gather`. Shares are scaled to `fz_down` within a
  load; the one-row 4096 and 131k loads sat at different clocks (144 and 55 us), so only the three-row ratios are steady.
- **Fusing a dependent pair needs the consumer to read only what its own threadgroup wrote.** `gate` into `route_split` (48 threadgroups
  into one), `attn_split` into `attn_merge` (the S partials), `rows` into `rows_finish` (the k-split partials) and each stage of the
  routed chain fail that test; a last-arrival counter would pass it with bytes equal, but needs a zeroed, mutable, hazard-tracked
  scratch the launch seam does not offer (outputs are fresh and unzeroed, inputs are const).
- **GPU busy and idle are read from the command buffers** (`GPUStartTime`, `GPUEndTime`), which Metal fills after
  the completion signal: a finished buffer is read at a later prune, in order.
- **The round's verdict stays on the engine** (`dsv41_serving.verdictStream`): the masked argmax over the verify's
  logits is our op, so the host waits once for the picks instead of lending the logits to MLX first. A sampled round
  (temperature, top-k, top-p, seeded or not) takes the same stream: `sampler.probsAllPositions` and the draws run on
  the engine, the filters unchanged and the draws under keys of the engine's own (see Random above); a round puts no
  op on MLX and lends nothing for its draws.

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
