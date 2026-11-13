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

A wide chunk asks nothing of MLX: the slabs, their views and the union workspace of a spilled route are the engine's
own on a Cooler load.

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
The route hands the open pass to the GPU before each of its reads (`mlx.engine.submit`): the prep and the shared
expert's GEMMs encoded ahead of the first group would otherwise wait in the pass for the SSD
(`StqStream.held_at_read` counts dispatches a read found unsubmitted; a served route leaves it 0).

The load also runs the wide forwards a prompt reaches (`warmWide`: a full-width state-only and head chunk, head chunks
whole, part and not at all on the GEMM tiles, the held-back tail, one chunk per pick-sort size and one per
compressed-row count under 17), so a request compiles no kernel; MLX's own prefill warm stays beside it, for the image
forwards.

A wide chunk on the engine bills the one `Config.forwardCap` it always did and the engine's pool takes it whole: the
state rows the chunk appends grow in that pool (a ring's new buffer beside its old one), and MLX allocates nothing
under a Cooler load. The split that left the state's growth to MLX (`forwardCapMlx`) dates from the state on MLX; it
under-capped the pool by 0.18 GB at 225k tokens and 0.58 GB at 1M. `serve` installs the caps after the load, so the load's warm
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
- **A Cooler load, its requests and its unload ask nothing of MLX, pinned phase by phase** (the census test in
  `dsv41_generate_test.zig`: the load, a greedy and a seeded request in rounds, serial steps with logprobs, a restore
  point captured and restored, a step after it, an image prompt, the unload; the SSD tier has its own test, with the
  restored state compared byte for byte). Every row is zero calls, and a change that adds one fails it.
  `SUSHI_DSV41_CENSUS_TRACE=1` prints a phase's calls by entry point. The free of an empty handle is no call
  (`mlx_array_free` in the seam): mlx-c holds nothing for one. The census counts a load that is not the process's
  first (it opens and closes the pack once before): what a process asks once (the device's description) is its first
  load's, and with that kept out a row reads the same alone and after other tests. The census reads MLX's allocator
  before the load it counts, as a served process does at the end of its boot. Dispatches
  still follow what the process ran before (a 12-token request read 7,336 as a process's only load and 7,017 after
  other loads).
- **A Cooler load holds the engine's scope from its first tensor to its unload** (`engine.hold`, `engine.unhold`,
  counted, so a second load's unload leaves the first one's). Its trunk's stream is the engine's (`arch_life.ownStream`,
  the arch hook `Trunk.init` asks before it would take the thread's GPU stream), so `xfm.s`, `m.s` and
  `Trunk.outStream` are one stream, every host array, scalar and vector made for the load is the engine's wherever it
  is made, and an op that reaches an MLX stream is refused by name instead of lending a tensor. What a held scope
  turned up, each of which had been a call or an import per round or per request: the `-inf` scalar of the greedy
  verify's reserved-token mask (`spec_types.verifyArgmax`), the warm-up's token arrays, the think-marker mask, an
  image's patch array, the trunk shell's scalar constants. The `[engine] since load` line counts the refusals; a
  served request counts none.
- **MLX's allocator is not asked under a hold** (`mlx.zig`, "MLX's allocator"; `engine.MlxAtRest`). A hold begins
  with MLX's pool emptied and its allocator read once (active, cache, peak, external: five calls), unless nothing has
  changed in MLX since the last such reading (`mlx.changes` counts every call but a reading of the allocator or the
  device): then the reading still holds and no call is made. A served process reads at the end of its boot
  (`mlx.noteAllocator`), so its Cooler loads ask nothing; a load after an MLX-route model reads once more. Nothing of
  MLX's is made or freed under a hold, so the readings stay true to its end and the seam answers with them on any
  thread: the gauges, the bills, a request's headroom. A pool clear, a peak reset and a limit
  (`mlx_set_cache_limit`, the transient cap, the command-buffer budget of `DecodeBuffers`) are no call and change
  nothing, a limit handing back the value given; a hold spans whole jobs, so no set and restore straddle its ends.
  These were the last calls of a request: 300 to 700 each on the real pack.
- **The hold begins with the load job** (`model_load.doLoadOnInferenceThread`, for an arch that brings its own
  stream; the model keeps a hold of its own from `initDsv41` to its unload). What the job did on MLX before the
  model was made is under it: the readings around the weights, the wired limit. The ledger's cap of MLX's pool is
  an MLX load's alone (`capMlxPool`): a Cooler load swallowed it, and an MLX model loaded after it ran uncapped.
- **Another model takes a turn** (`engine.turnTo`, `Turn.end`; `arch_handle.turnFor`, `model_load.JobTurn`). While a
  Cooler load holds, work for a model that is not the engine's steps out of the scope and of the allocator at rest
  for its span and steps back at its end: the scheduler's prefill, decode ticks, slot cleanup and live-state flush,
  an image encode, a cache clear, and the load, evict and unload jobs. Inside a turn MLX is asked as it always was;
  after it the hold reads MLX's allocator again (MLX changed). A Cooler model's forwards after another model's turn
  ask nothing of MLX (pinned: `beside a resident Cooler model, a model on MLX loads, runs and unloads`).
- **What the two sides share is kept per side.** A config MLX made, or the MLX twin of a config of ours, takes a new
  grid or threadgroup when it next launches on MLX (`Config.behind`), never from inside the engine's scope. The EXL3
  module asks its host which side a stream is (`src/exl3/host_side.zig`, which names no engine: mlx-serve vendors
  the module) and keeps its GEMM input ring and zero-slot rows one per side.
- **A kernel is noted in a served process whatever the engine's state** (`route.noting`: a GPU with matrix units),
  so a kernel or config a module made before the first Cooler load launches on the engine like one made after it.
- **A Cooler pack loads again after its unload.** `dsv41_borrow.unloadModel` leaves the pool empty and uncapped, and
  `initDsv41` sets its own budget and unseals the pipelines before it proves a kernel. The NAX GEMM arm proven on the
  engine's side stays proven: a later build that fails there is an error, never a switch to another arm.
- **The device is asked once a process** (`mlx.deviceArchitecture`, `maxRecommendedWorkingSet`; `main` asks what a
  load would, the NAX question of the EXL3 kernels included, above every GPU command: `sushi kld` loads too). A live Cooler process (STQ 1.75bpw, 348b99fb plus this
  change) makes 27 calls into mlx-c, all in its boot: the device, the default device and seed, the pool cap, one
  allocator reading. Its load, its requests and its unload make none. `SUSHI_MLX_CALLS_TRACE=1` prints the tally by
  entry point at both ends of a load and after a request ([envvar](envvar.md)).
- **A trial that picks a kernel arm runs on the load's stream.** The attention and index-scan matrix-unit trials
  (`kernels.attnNaxTrial`, `indexNaxTrials`) ran on MLX's stream with MLX's normal draw; inside a held scope both
  were refused, the refusal read as "the trial differs", the SIMD and scalar kernels served and four of five live
  replies changed bytes. On the engine's stream the inputs are made on the host (`trialValues`) and a trial that
  cannot run fails the load (`setAttnNax`, `setIndexNax`); on MLX's stream a failed launch still declines the arm,
  as a GPU without matrix units needs.
- **A tower the engine cannot run refuses a Cooler load** (`error.VisionTowerNotPorted`, with the shape that
  `engineGap` found); it no longer stays on MLX beside the engine. The MLX route serves such a pack. The image
  encoder's own stream is the tower's (`Tower.ownStream`): the patch arrays and the join of several images' spans are
  the engine's there.
- **A Cooler load runs no forward on MLX.** The load's warm used to run three prefill chunks through MLX whatever
  the decode side; they only built the state the engine's own warms start from. Where prefill is the engine's they
  run on the engine, and with the pipelines read from a manifest they do not run at all.

- **At rest MLX is asked nothing and told nothing, and both are counted.** A call that reaches mlx-c while MLX's
  allocator is at rest is a break (`mlx.rest_breaks`); a setting asked there (a pool clear, a limit, a peak reset)
  reaches nothing and is counted (`mlx.rest_swallows`). Both are zero in the census, and a Cooler unload warns when
  they are not. An array of MLX's that reaches an op of ours at rest is refused by name (`MlxAtRest`), as is a lend.
- **Work for MLX asked at rest is owed, not dropped** (`mlx.payOwed`): the free of a config's MLX twin and the wired
  limit a load asks for are paid at MLX's next turn or at the last unhold. Before MLX goes to rest the engine frees
  the MLX arrays it retired (`core.settleMlxAliases`).
- **A turn that changes nothing asks nothing.** A turn of the engine's inside another reads nothing; MLX's pool is
  emptied by the first hold only, and its allocator is read again only by the turn that woke it.
- **The scope's state has one owner** (`src/tensor/route.zig`): `enable`, `makers(own)`, `unsealed()`, `enter`,
  `turnTo`, `hold` and `unhold` are guards that give back what they found. No other file writes `engine.active` or
  `engine.enabled` (lint rule `scope-state`).
- **The last Cooler unload leaves nothing of the engine's in use**: the pool is empty and the EXL3 module's arrays of
  the engine's side (its GEMM input ring, a unit scalar) go with it (`sushi_exl3.dropHostArrays`). Pinned from zero:
  `requests, repeated, leave the engine's pool as they found it, and an unload gives back everything`.
  The last unload of a model on MLX gives back MLX's side the same way (`arch_handle.adopt` counts them, `destroy`
  drops at zero), pinned from zero by `the last model on MLX's side gives back the arrays the EXL3 module cached`.

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
- **Nothing keeps MLX's queue in use under a Cooler load.** While a request's end still ran on MLX, a queue left
  unused for about a second made that end wait 0.56 to 0.77 s, and one MLX op per warm tick kept it
  (`keepMlxQueue`, now deleted). No part of a request runs on MLX, so nothing waits on its queue: the idle tick
  (`mlx.gpuTouch`) passes the engine's queue alone under a held scope.
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
  whether or not a forward is open. Under a Cooler load what they find, and give back, is the load's held scope.
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
- **The NAX GEMM kernel is proven on the stream that runs it** (`kernels.prepareGemmNax(dec, order, s)`, called by
  `initDsv41` for the expert and trunk decodes under the load's hold). The proof is a one-tile launch; it used to run
  on MLX's stream before the hold, the last two kernels a Cooler load made and ran on MLX. A forward that meets a
  kernel not yet built proves it on its own stream the same way.
- **A failed launch declines a kernel arm only before that arm has ever worked** (`kernels.gemm_nax_worked`). After
  one launch has run, a failure is the launch's own (a pass the engine refuses by name, memory) and is returned: taken
  as "the arm is unavailable" it switched the whole process to another arm, with other bytes, without a word.
- **Engine pipeline creation is 0.1 s with the system's shader cache warm and 9 to 16 s cold**, all of it
  `newLibraryWithSource` (435 pipelines, M5 Max); the pipeline state is 0.03 to 0.06 s. A binary archive replaces only
  that state step, so it buys nothing here. A binary the system has not run is cold again, and the
  manifest's recipes then compile on 8 threads in about 2.5 s ([the manifest](#the-pipeline-manifest)).
- **A kernel has one signature.** MLX binds an input under 8 elements as a `constant` reference and a longer one
  as a `device` pointer, so one source is two pipelines; ours binds every array as a `device` pointer.
- **The warm runs every shape that names a pipeline** (`warmOwn`, `warmOwnFills`, `warmOwnBlocks`, `warmOwnLong`,
  `warmOwnStream`):
  a verify of every width to the ring's slack (a copy round is wider than the block), each with the routed chains
  queued and waiting, the block indexer at a context it can afford (fewer candidate blocks kept), a draft at every
  fill of the stages' windows (MLX picks a matmul tile and a row reduction by the row count, so a prompt shorter
  than the window is other kernels), and the kernels a step takes from 131072 keys (the one-row index scan on the
  matrix units by lowering `Model.index_nax_keys` for one step, the split top-k on scores of its own), and a chunk
  whose index top-k streams over key tiles, by lowering `Model.index_stream_tile` to its smallest (a prompt streams
  only past 32768 keys; the fold of a second tile takes ids along an axis, a pipeline no shorter prompt makes).
- **A path a prompt reaches only past a threshold is warmed by lowering the threshold**, never left to the first long
  prompt: the count must be no kernel's template value, and a test seals the engine and runs the path at that setting.
- **A round whose rows take two attention split counts is warmed by telling the rounds to split** (`Model.attn_apart`,
  `DecodeArgs.apart`): the split count steps with the context (768, 1408 and 2048 tokens on the release pack), a round
  across a step launches its rows in parts on pipelines keyed by each part's row count, and no context of the warm
  reaches a step. The warm runs every width once more with row 0 launched apart from the rows after it.
- **The sampled verdict is warmed** (`dsv41_serving.warmSampledVerdict`, after `warmKernels`): `sampledDraws`, the
  function a round calls, runs on the engine's stream over zero logits of `[1, w, vocab]` f32 for every verify width
  with a suppress mask. The draw is one kernel (`sampler_draw`) whose temperature, top-k and top-p are inputs and whose
  only template values are the vocabulary and the logits' dtype, so one setting warms every request's filters; a seeded
  request differs only in its root.
- **The serial sampler is warmed** (`warmSerialSampler`, with the sampled verdict): a first-token or serial draw,
  sampled and greedy, with and without the suppress mask, the three penalty mixes with logprobs, a grammar mask,
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
  which case ran. A later warm (the tower's) writes it again only where a complete one stands, found or stored by this
  load: a short manifest would be trusted by every later load, which skips the warm that could fill it.
- **A Cooler load whose warm fails does not serve**: a forward of the engine's that fails in the warm fails in a
  request. On MLX a failed warm is a warning and costs requests their compiles.
- **A launch the warm must reach on the release shape is tested at that shape.** The tiny fixtures hold no 5120 x 129280
  head, so the head's 9- to 16-row launch failed only on the release pack and the load warned and served.

## What a forward costs

- `[engine] since load:` (debug, at a request's end) and the decode meter's `[fwd-ubench] engine:` line carry
  dispatches, command buffers, waits, imports, the pool, and the GPU's own times: busy, idle between passes, and
  that idle by pause length and by what the host was building (forward, draft, verify, round end).
- **Every request states its own GPU time at info**: `[engine] GPU busy N ms, idle between passes M ms (P% busy) in
  this request`, the union of its command buffers' spans as Metal timed them (`engine.gpuTimeTaken`, reset when the
  load seals). It is the engine's own reading: the dashboard's "busy" is the whole Mac's GPU as IOKit reports it.
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

- **Error paths are swept, not reasoned about**: `mlx.engine.fault.arm(k)` refuses the pool's k-th buffer, and a test
  walks k over a whole request (`a request refused at any one of its pool buffers leaves the next request its clean
  bytes`). A tensor released twice is latched and counted (`core.double_releases`), never returned to the pool twice.
- **A read of bytes nothing wrote is caught by poison** (`mlx.engine.poison`): every buffer the pool hands out and
  every free one written over, the forwards and requests must give the clean bytes.
- **The census counts the engine's work too**: every phase of a Cooler load's life dispatches the same at a later
  load of the process, so a one-time cost that slipped into a request shows as a difference.
- **Kernels are held to MLX over the edges, in tables**: the special floats and every exponent through each
  elementwise op, cast, reduction, sort and scan, and each op over a turned view, a strided slice, a slice of a
  slice, a broadcast and an empty view (`engine elementwise ops, casts, ...`, `engine ops give MLX's bytes over ...`).
  What MLX itself gets wrong or refuses is named in the test ([engine-mlx-gotchas](engine-mlx-gotchas.md#what-mlxs-bytes-depend-on)).
- An op whose result has no elements returns it without a dispatch; a reduction over an axis of no elements into a
  result that has some is refused by name. An MLX array of no elements imports as an empty tensor of ours.

## Ruled out

- **Starting the next step before a one-row round's verdict is read.** The next step's embedding and layer 0 were
  queued on the verdict tensor, as MLX's lazy start does for a serial decode. It engaged on about 40 of 100 prose
  rounds (only a round the policy already runs plain predicts a plain successor), verify time per round stayed at
  21.1 to 21.2 ms against 21.0 to 21.1, and prose tok/s stayed inside noise at every rung (M5 Max, AC, STQ 1.75bpw,
  in-process `sushi bench --prose-only --tokens 128`, the change on 537e8e91). The pause between one-row rounds is
  not where prose is won: a one-row verify is 20 to 21 ms on both sides, so it needs less GPU work in the forward.
