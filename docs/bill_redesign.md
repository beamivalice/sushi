# Static bill: the masterplan

Status: the blueprint, with [the owner's seven decisions](#decisions) ruled on 2026-11-22 and written through it.
The direction is the owner's (a static, exact Cooler bill; pilot on Hy4-Flash for 1.8.0, then DeepSeek-V4.1 and
Qwen3.8; MLX-side loads keep today's bills). This file works that direction out into mechanisms, an order of work
and gates. It changes four things in the first outline, each with its reason, in [what changed](#changed). Two
spikes can still overturn the recorded plan. No implementation worker starts before the owner orders it.

How to read it: [the contract](#contract), [what changed](#changed) and [the decisions](#decisions) are what is
settled. Parts 3 to 9 are the mechanism. [The build](#build) is what an agent takes a work package from.
[Spikes](#spikes) come first.

**The foundations are written** ([part 11](#built)): the plan's pure core and the ledger, host tier, with a test
that records, packs and replays generated forwards. An implementor wires the engine to them and does not redesign
them; [what may not change](#fixed) is the list.

<a id="contract"></a>
## 1. The contract

Five statements. Every mechanism below exists to make one of them true, and every test names the one it holds.

- **C1. The bill is the allocation.** Every byte of device memory a Cooler load holds belongs to one owner in one
  ledger, and the bill is the sum of the owners' capacities, read from the same numbers they allocate with. There is
  no second description of memory to drift.
- **C2. Serving allocates at three moments only.** A load (weights, slots, the arena), an admission (a request's
  state) and a release. Inside a forward, between forwards and in the sampler the engine never asks Metal for memory.
- **C3. A forward gets memory by offset.** Its buffers are places in one arena fixed at load. Nothing depends on
  request order or on when the GPU finishes a pass.
- **C4. A wrong plan fails closed, by name.** A buffer the plan did not foresee refuses that request with the site,
  the size and the cap. It is never an OOM, never bytes over another buffer, never a silent slow path.
- **C5. No size is written by hand.** A forward's buffer sizes come from running the forward's own code. A kernel
  that grows a scratch moves the bill in the same commit without anyone editing a bill.

What "cannot be wrong" means here, exactly: C1 to C3 hold by construction, always. What can still be wrong is a
plan's coverage (a shape whose structure no probe reached). C4 turns that into a named refusal, and
[coverage](#coverage) is how a release proves there is none.

## 2. The facts that decide the mechanism

Read on the Hy4 line at 534527ca. Each one rules a design in or out.

- **A Cooler forward never names a buffer.** All three archs are written against the MLX-shaped API; on the engine's
  stream `src/tensor/route.zig` answers each call and every op output and kernel output is an anonymous ask to the
  pool. Hy4's forward is composed of ops on purpose (`hy4_forward.zig`: "the same code runs on Sushi Cooler's stream
  when served and on the CPU stream against the oracle").
- **Every ask goes through one function.** `core.freshFor` to `core.poolStore` (`src/tensor/core.zig`); a kernel's
  outputs through one line of `src/tensor/kernel.zig`; a host array through `core.fromHost`. One choke point can serve
  every arch, every shared module and the vendored EXL3 module without touching them.
- **Lifetimes are refcounts, not scopes.** Hy4 has about 99 `Ops` scopes and 56 `o.result` hand-offs to an outer
  scope; DeepSeek and Qwen free arrays by hand in hundreds of places (`trunk.zig`, `dsv41_model.zig`). No stack or
  scope discipline fits all three without rewriting them.
- **Today's Hy4 bill is the failure it is meant to prevent.** `hy4_bill.zig` restates the op chain with hand-counted
  multiplicities (`s.add(24, t*n*f32b)`, `s.add(9, r*h*(keys+1)*f32b)`). A declared plan with a size function per
  buffer would be the same second copy, larger.
- **The GPU already runs passes in the order they were made.** `core.commitPass` makes each pass wait the fence of the
  pass before it, and inside a pass `metal.Recorder` orders dispatches by the stores they bind. So bytes may be reused
  by later work without the host waiting, as long as the engine can see the reuse. Only the host's own writes and
  reads care about GPU timing.
- **There is no bound on passes in flight** (`State.flying` grows), no fixed Cooler budget (every limit is re-read
  live through `bills.currentGpuMemoryCeiling`), and Hy4's prefill chunk is re-chosen per request and per chunk from
  the live headroom (`bills.chooseRequestPrefillChunk`, `adaptivePrefillWidthNow`).
- **State is ordinary pool buffers that regrow by copy.** Hy4's row stores grow a sixteenth at a time
  (`row_store.Store.growTo`), Qwen's KV by a quarter (`kv_cache.nextCapacityPolicy`); old and new are both live.
  DeepSeek alone reserves at the request's end (`dsv41_model.Rows.bind`) and trims by another copy after.
- **A borrow frees and remaps.** `expert_stream.borrowLayer` unmaps the lent slabs, `dsv41_borrow` waits up to a
  second for macOS to unwire them, and the prefill grows the pool into what came free. Nothing is aliased.
- **Metal, on the M5 Ultra (macOS 27.2, a probe to be rerun and recorded in [S3](#spikes)):** `maxBufferLength` is
  193,273,528,320; a shared, untracked placement heap is granted, its buffers align to 256 bytes and may overlap, and
  its allocated size equals its declared size; a plain buffer is rounded up to 16 KB pages.

**Where the recent fixes went** (78 bill commits in four days, about 53 of them fixes; a fix may sit in two classes),
and what removes each class:

| class | fixes | removed by |
|---|---|---|
| a lend ledger against live readings | 11 | the borrow is an alias fixed at load ([part 7](#borrow)); nothing live is read |
| the allocator's growth, size class or page rounding not mirrored | 9 | no size classes, no growth: one arena, state at its reserved capacity |
| the wrong width or row count priced | 6 | sizes are recorded from the forward at every served width |
| a regrow's old and new buffer | 6 | state is reserved once and written in place |
| buffers a pass holds, parked free buffers, cap bookkeeping | 6 | no free lists, no caps, no trims |
| a kernel's scratch the bill did not name | 5 | C5: the recorder sees every ask |
| the bill priced another arm or config | 5 | the plan is recorded under the load's own arms and settings |
| bytes left in the pool by an error path | 3 | the pool is closed after the load; an arena has nothing to leak |
| terms that never coexist summed | 2 | the arena is the packed peak, not a sum of terms |

<a id="changed"></a>
## 3. What changed from the outline, and why

| outline | masterplan | why |
|---|---|---|
| a forward is a hand-declared comptime list of steps with a size function per buffer | **the plan is recorded from the forward** by a shape-only dry run, then packed and replayed | the forwards are op chains through one choke point; a declared plan restates every op's output (C5 fails) or needs every served forward rewritten and forked from its oracle path. A recorded plan has the same tables, offsets and checks; only who writes them differs, so a hand-written plan can still replace a recorded one for a hot phase later, behind the same binder |
| one `MTLHeap` of the placement type, a buffer made at each offset | **one buffer, bound at offsets** (`Store.at`, `Pass.buffer(b, offset, index)`, both already there) | a replayed plan has tens of thousands of placements, too many buffer objects to make; the borrow needs the arena's tail to be the lent slabs' own pages, and a heap cannot be laid over existing memory. Same bytes, same exactness. A placement heap stays a valid backing behind the same `Region` if [S3](#spikes) finds a reason |
| the transient arena replicated `D` times | **only what the host touches is replicated** (`G + D·H`, not `D·arena`) | the fence chain orders GPU reuse; replicas are needed only where the host writes before the GPU has read |
| comptime arena sizes for release configs | **every load records once and caches beside the pipeline manifest; pinned tables for release packs are proven equal to the code by a host-tier test** | the forward cannot run at comptime. The pins give the numbers ahead of serving and make a change visible in its diff; they are a test's proof and no load reads one ([D3](#decisions)) |

Kept as written: storage-owned state reserved at admission, the bill's shape, the chunk chosen at load, slots from
the budget, the borrow as a declared alias, the enforcement list, Hy4 first.

**Ruled out while working it out**, each by a fact above:
- **A stack or scope arena.** Lifetimes are refcounts: a layer's new streams are made before the old ones are freed,
  and a span's result outlives its scope by a hand-off made after the op ran. No frame can be popped in order.
- **Size-class slots frozen at a recorded count.** They keep the rounding, and a buffer changes class as a shape
  grows, so a smaller shape can need a slot a larger one did not.
- **Any allocator that decides online inside a fixed arena.** A smaller early buffer leaves a hole a later buffer no
  longer fits, so a shape below a probe can need more than the probe did. Replay has no such case: each ask lands at
  its recorded place and is no larger than its cap, so what was disjoint stays disjoint.
- **A dry run before every served execution.** Exact with no keys at all, but it puts host work ahead of the first
  dispatch of every decode token. It is kept for tools ([record-then-run](#tools)).

## 4. Memory map: owners and the ledger

Everything the engine holds for a Cooler load is one of these. `src/ledger.zig` (host tier) holds one line per owner.

| owner | bytes | known | allocated |
|---|---|---|---|
| `weights` | everything the loader maps: the pack index's tensors, each rounded to pages (`model.ownedRegion`), and the weights a load builds and moves out of the pool (`model.ownInPlace`: decoded banks, a gather's copy, a widened head) | the index before the load; the built ones at the seal, and from a dry load at [stage 8](#build) | load |
| `slots` | the expert stream's slabs, bounce, union workspace and kept scales, `expert_stream.budgetLedger` | before the load | load |
| `static` | what the load makes on the engine and keeps: tables, maps, kernel configs' arrays, the counter page, the touch scratch, and every cache of engine arrays, filled at load ([below](#caches)) | measured at the seal, cached for the next preflight; from a dry load at stage 8 | load |
| `arena` | `G + D·H` ([part 6](#arena)) | from the plan | load |
| `pipelines` | the compiled pipelines, measured once per GPU class and build | at the load's end | load |
| `state`, `parked` | each conversation's stores at their reserved capacity plus its fixed block ([part 8](#state)); `parked` once its request ended | closed form in the reach | admission |
| `host`, `media` | host memory that follows a load (the Engram row caches, a file-mapped table's residency cap) on `host`, or a request (the Engram history, the SSD writer's permit, an image request's decoded pixels) on `media`, which its end gives back | closed form | load, admission |
| `margin` | Metal's own per-command-buffer and per-buffer overhead: one named constant per GPU class, measured ([S6](#spikes)), taken by each arch at its cutover; until then that arch keeps `BILL_MARGIN_BYTES` ([D7](#decisions)) | constant | — |

```
model bill   = weights + slots + static + pipelines + arena + margin            (known when the load ends)
request bill = state(reach) + fixed + points + host(reach)                      (closed form, reach = prompt + max_tokens + spec rows)
admission    = model bill + sum(admitted and parked request bills) <= budget    (a lookup and an addition)
```

- **The pool closes at the seal.** `arch_cooler.warmEnded` already seals pipelines; from there `core.poolStore`
  refuses (`PoolClosed`). What the pool holds then is `static`, counted from its own stats. Offline tools
  (`sushi kld`, `sushi convert`, `sushi-ubench`) run [record-then-run](#tools) and never open the pool either.
- **The ledger replaces the readings.** `bills.currentGpuMemoryCeiling`, `admissionAvailableNow`,
  `prefillHeadroomNow`, the lent and parked readings and the pool's atomics are no input of a Cooler bill. Free RAM
  is read as one extra gate that can refuse and never admit ([part 9](#admission)). An owner reserves on the ledger
  before it allocates and allocates exactly what it reserved (`ledger.reserve`), so the two cannot differ.
<a id="caches"></a>
- **A cache of engine arrays is filled at load or has fixed slots.** About ten sites keep arrays in a cache that
  fills on a miss while serving (`exl3/side_cache.zig`'s LRU, `exl3/linear.zig`'s input ring, and their users in
  `rate_groups`, `moe_router`, `verify_qmm`, `mtp_qmv`, `hc_prefill`, `trunk`, `qwen4_hc`). Each asks only on a miss,
  outlives its execution and frees inside an unrelated one: it would break both the closed pool and a frame's
  structure. Each becomes entries made at load (`static`), or a ring of fixed slots written in place. The first
  recording names every one of them as an escape.
- **Arrays made outside a forward belong to a phase.** The generator's ids (`generate.zig`), the sampler's masks
  and draws (`sampler.zig`), a restore's installs (`prefix_vfs_tensors.zig`) are pool asks today; each is made inside
  the execution that reads it. After the seal `PoolClosed` names any that was missed, in the census.

## 5. The plan

### Vocabulary

- **Phase**: a kind of work the engine runs: `decode`, `verify`, `draft`, `sample`, `prefill`, `encode` (a vision
  tower), `capture` and `restore` (a restore point), `tool`.
- **Execution**: one run of a phase, from `Binder.begin` to `Binder.end`. The arena is empty at both ends.
- **Ask**: one call of `core.freshFor`. It has an ordinal in its frame.
- **Frame**: a replay unit, a code site plus a key. Within a frame the sequence of asks and child frames is a pure
  function of `(site, key)`; only sizes vary, and they have a recorded maximum. An execution's root is a frame. A
  child frame is **exact** (part of its parent's structure at its place, `plan.frame`) or **one iteration of a loop**
  whose count varies (`plan.each`): consecutive iterations of one site are one group, of any count up to the longest
  recorded, and their keys may differ.
- **Lift**: how many frame levels an ask outlives. A span's result that the layer keeps has lift 1.
- **Key**: the small tuple of ints, enums and bools that decides a frame's structure (a span's row class, a layer's
  index, a route's kind). The author names it; the recorder proves it sufficient.

### What a forward's code gains

Almost nothing. The forward keeps its ops, its scopes and its frees. It gains markers where its structure varies:

```zig
// src/tensor/plan/plan_api.zig: what a forward's code calls. Outside an execution (an MLX stream, the CPU oracle
// stream, a load) each is a branch on one flag and changes nothing.
pub inline fn frame(comptime src: std.builtin.SourceLocation, key: anytype) Error!Frame; // exact; `defer f.end()`
pub inline fn each(comptime src: std.builtin.SourceLocation, key: anytype) Error!Frame; // one iteration of a loop
pub fn choose(comptime src: std.builtin.SourceLocation, comptime T: type, real: T) T; // data decides a structure
pub fn bound(real: usize, max: usize) Error!usize; // data decides a size: the ask is planned at `max`
// On the engine's side of the seam (stage 1, `tensor/core.zig`), since they name tensors:
pub fn into(dst: *core.Tensor) void; // the next op's output is this state-owned tensor
pub fn hostResult(t: *const core.Tensor) Error![]u8; // a readback that steers no shape
```

- **`each` goes where a trip count varies, `frame` where a structure is fixed but worth a unit of its own**: a span
  loop, a routed-group loop, the embedding gather's groups (`each`); the layer loop, whose count is the config's
  (`frame`, keyed by the layer index: the tables stay small and each layer's quant geometry is its own); a tower's
  blocks. Hy4's forward needs about ten by this reading. An engine op with arms of different asks (`matmul`'s
  split-K against fused, `attention.planOf`'s passes, the top-k's splits) frames itself, keyed by the arm it took.
- **What a loop hands its parent is held to the loop's end.** Every iteration of a loop hands up the same number of
  results, whatever its key; the plan gives each iteration's results their own bytes until the loop is over, and
  nothing of the parent's is taken to end inside a loop. That is what makes one recording at the longest run hold for
  every shorter one. Its cost: a value a variable-count loop carries from one iteration to the next is not reused
  across iterations. A loop of fixed count (the layers) is exact frames and has no such cost: its carried streams
  take two places.
- **`choose` goes where data picks a structure**: a streamed route's kind (cache bank, union, spill split), a rollback
  that replays. Its value is the key of the frame it guards, so each variant is its own unit and they never multiply.
- **`bound` goes where data picks a size**: the picks of an expert group, the distinct id blocks of an embedding
  gather. The bound is the op's own math (`rows · top_k`, `min(t, vocab/128)`), written at the site that owns it.
- **`into` goes where a forward's output outlives it**: the logits a step keeps, a drafter's hidden row, a GDN state.
  The destination belongs to a state owner; the forward writes it and holds nothing after. It names an op's
  **output**, not its next ask: an op asks for its scratch first (`kernel.zig` copies inputs before it makes
  outputs, `matmul` makes its partials before its sum), so the engine distinguishes the two at its own seam
  (`core.freshOut` beside `core.fresh`). The plan records that ordinal as state-owned, and a replay that finds
  scratch there, or the reverse, is a miss.
- Nothing else in a forward changes. `mlx_array_free`, `Ops.deinit` and `o.result` keep their meaning; on an arena
  store a free returns no memory and only ends a lifetime.

### Structure, and how a wrong key is found

The recorder merges every instance of `(site, key)` it sees across all probes. Two instances that differ in ask
count, in an ask's lift, in a host flag or in their child sites are an error at plan build
(`PlanAmbiguous`, printing both sequences side by side from their first difference): the author adds what differs
to the key or frames the loop that varied. Authors do not reason about keys; they run the planner until it is quiet.

Invariants the merge checks, each with a host-tier test on synthetic traces:
- the keys of one loop hand up the same number of results; only sizes differ;
- consecutive iterations of one site are one group, any count up to the recorded maximum, none included;
- an ask alive at the execution's end that did not come from `into` is an escape, an error naming its site;
- a frame site never opens inside itself.

<a id="recorder"></a>
### Dry mode and the recorder

A recording runs real code in **dry mode**: every host decision runs, no GPU work does. It lives at the
engine's choke points, not in any op:

| call | in dry mode |
|---|---|
| `core.freshFor` | logs the ask (frame, ordinal, bytes, host flag), returns a tensor with a shape and no bytes |
| `core.dropStore` at zero refs | logs the death of that ask |
| `core.bind*`, `core.dispatch` | computes the dispatch's level by `dispatchLevel`'s rule over the whole execution, on maps no pass cut clears and keyed by ask (the packer reads it), encodes nothing |
| pipeline lookup and creation (`stock.kernel*`, `core.pipelineFromSource`) | skipped |
| `core.fromHost`, a host scalar, `route.hostFilled` | logged as a host ask over a host scratch of its bytes, which its filler writes (`hostFilled` and `core.hostBytes` hand their caller the bytes); no buffer |
| an eval used as a wait (`mlx_array_eval`, `mlx_eval`, `mlx_async_eval`), `sync`, `submit`, `waitFor` | no-ops |
| a read of bytes a dispatch of the execution wrote (`mlx_array_data_*`, `item_*`) | an error (`DryReadback`) naming the caller, unless the site went through `plan.hostResult` or `plan.choose`/`bound`; a read of bytes the host filled is a plain read, and so is one of the load's own state from before the execution (a dry model's state has none) |
| `core.hostBytes` (a host write: `ops.zeros`, `arange`) | a host ask's scratch; any other store is refused (`DryHostWrite`) |
| `core.lend` (a tensor handed to MLX) | refused: MLX's work is never dry, and a lent store dies on MLX's thread at a time no recording orders |
| the expert stream's fills (`expert_io`) | skipped; the route's kind comes from `choose` |

The recorder itself is mode-blind (`plan_trace.Recorder`): attached while the pool still serves, it logs a real
forward's asks, deaths and levels, the levels over the execution as in dry mode: a pass's own restart at every cut, and
where the GPU's timing cuts a pass moves them (S2). That is the real trace a dry trace is held to; the pool's peak trace
(`tracePoolAsk`) is a set of live buffers with no order and cannot serve.

- **The model is the load's own**, weights and all, or a **dry model**: the same structs over shape-only tensors
  built from the config and the pack index. The dry model is what lets a host-tier test plan a release config with no
  pack and no GPU, and what [stage 8](#build) uses to bill a load before reading a weight.
- **State is dry too**: a state owner answers `used = N` for a probe's context without holding N rows, so a probe at
  1,048,576 keys costs milliseconds.
- **Machine arms are inputs, latched before recording.** What a trial decides at load (the matrix-unit GEMM arm,
  `hc_tg`, an attention plan's pass count) is part of the plan's cache key. An arm that today latches at a first
  real launch from a pipeline's own limits (`hy4_hc`'s wide grant, `sampler_draw`, `hc_kernels`, `dsv41_kernels`: a
  dry run makes no pipeline) gets a load-time probe, as `probeIdentity` is one, and a recording with an arm
  unlatched is refused by name. Tests set the arms by their seams.
- **A readback is classified by its author or the recording fails.** That is the audit the outline's open question
  asks for: every site where the host reads a GPU value is found by the first dry run, by name.
- **A classification is checked, not trusted.** `hostResult` answers a recording with made-up bytes, and a site
  wrongly called a result would record whatever those bytes steer it to (zeros give `hy4_expert_mv`'s group list
  its minimum). So every probe is recorded under two fills (zeros, and a ramp of distinct values); two traces that
  differ prove a readback steers a shape, and the planner names it. S1 met that case: on the BF16 tiny's decode, zeros
  make the group list one group (12 B) where the picks and the ramp make two (24 B).

### Probes

A probe is one execution in dry mode at a chosen shape. The probe script and the load's warm are **one list**:
`src/cooler_warm.zig` drives the requests that reach every shape naming a pipeline (every verify width, each sampler
mix, joined rows, an image request) through the generator, and the same requests run dry to plan, so the generator,
the sampler and the round logic are recorded with the forwards. A load whose manifest lets it skip the warm reads
its plan from the cache under the same key. To the list the planner adds:

- every decode and verify width the arch's forward can make (Hy4: 1, and 2 to 8, D18), each at every context boundary;
- the chosen chunk width, the tail it may absorb where the arch merges one, and one span of every row class;
- **context boundaries**: the keys counts where a structure or an arm changes. An arch lists its own
  (Hy4: the window, the selection threshold `group_topk · ratio`, the top-k split counts); every engine op with arms
  registers its thresholds (`plan.boundaries`), so the caller cannot forget one. Each boundary is probed on both
  sides, and the model's maximum context always;
- every value of every `choose` site, by the driver (`plan_api.Driver`): a pass holds each site to one value, and to
  explore a site's next value it replays the choices that first led to that site. One site at a time is not enough:
  a choice that exists only under another's value would never be reached again.

<a id="coverage"></a>
### Coverage

Sizes are safe when each frame's sizes peak at a probe; structure is safe when every `(site, key)` a request can
produce was recorded. Neither is assumed:

- **A property test replays random shapes in dry mode against the built plan** (host tier, release config, thousands
  of `(phase, rows, keys, position)` draws, milliseconds each): no draw may miss. Today's tests sample tiny fixtures
  on a GPU; this one samples the release's whole shape domain without one.
- **The warm is the same script on the GPU**: a load whose warm misses does not serve (`model_load.warmLoad`).
- **The release bench's long-context rungs run with the miss counter pinned at zero.**
- **A miss in a served request** fails that request by name (`PlanMiss`, with `Binder.miss`: kind, frame,
  ordinal, asked, cap), logs `[plan] miss` at warning level and appends the shape to the plan cache; after the
  request the inference thread records that one shape and re-packs. Tables that fit the arena as allocated replace
  the old ones, between executions, when the arena is empty; tables that need more wait for the next load, and the
  log line says which ([D5](#decisions)). This is the pipeline manifest's rule for a kernel compiled late, with the
  same target: zero on a release pack.

### Packing

Pure functions in `src/tensor/plan/plan_pack.zig`, host tier, deterministic. For each `(site, key)`, on its own
timeline (its asks and children in order, each loop unrolled to its longest run):

1. **Items**: its own asks with lift 0 (from birth to the latest death any instance showed); what its children hand
   it, one item for each iteration and result (a loop's held to the loop's end, all iterations at one size); and one
   block per child: an exact child's own, or a loop's, sized to its largest key and shared by its iterations.
2. **Order**: by size descending, then first birth, then ordinal.
3. **Place** each at the lowest aligned offset where it overlaps in address no placed item it must stay apart from:
   one that is alive when it is, or one the level rule names.
4. **The level rule** (with the floors only, D13): two items that never live together still do not share bytes when the earlier one's last
   dispatch level is not below the later one's first in the recording. It keeps most reuse off work the GPU could
   have run side by side. It is a heuristic and not the guarantee: a pass may be cut anywhere, levels restart with
   each pass, and then a reuse does order two dispatches that would have overlapped. [Order](#order) is what makes
   every reuse correct; the cost of the ones the rule misses is measured by the stage gates.
5. **Predecessors**: for each item and each child block, the latest earlier tenant of each of its bytes. A tenant a
   shorter run never makes (a loop's block, a loop's result) hides nothing: what lay under it is named too.
6. The frame's block is the highest end. `G` is the largest root block. **`H` is structural, not the largest probe**:
   a frame's own host asks, plus each exact child's, plus each loop's longest run of its heaviest key, since a served
   run may combine what no one probe ran together.

This is the offset form of the greedy-by-size planner that static inference runtimes use, and it packs tighter than
any allocator that must decide online. Each frame's plan carries its block and its lower bound (`FramePlan.peak`,
the most bytes alive at once); a test pins the ratio per release plan so a regression in the packer is a visible
diff.

**Tables** (`plan_pack.Tables`). Per `(site, key)`: its nodes in order; for each ask its kind (here, lifted, host,
state-owned), cap, offset and predecessors; for each child its block's offset, size and predecessors; for each
result a child hands up, whether it is placed here or handed on. Dense arrays: a replay hashes nothing but the frame
id at a frame's entry. A plan serializes beside the pipeline manifest under the same key (pack, build, GPU class)
plus the arms and the settings that shape it (KV width, drafter, vision, chunk, context, slots). A repeat load reads
it and records nothing.

<a id="arena"></a>
## 6. Runtime: the arena and the binder

**One buffer.** `arena = [ H × D | G ]`, made once at the load's end, in a residency set of its own kind. Its length
is the ledger's `arena` line; a test reads Metal's `allocatedSize` for it and compares. The arena is the engine's,
not a model's: executions never overlap, so two resident Cooler models share one arena sized to the larger need,
remade at the second one's load while nothing is in flight. Each model keeps its own tables.

**The binder** (`src/tensor/plan/plan_binder.zig`, inference thread only) is a stack of open frames. An ask in an
execution (`Binder.ask`):

```
the frame's next node must be an ask                     (a child was due, or none is left: miss)
e = frame.plan.asks[ordinal]
its kind (scratch, host, state-owned) is not what was asked, or bytes > e.cap      -> miss
host         -> bump in this execution's replica of H; past H: miss
state-owned  -> the tensor `into` named
scratch      -> its place: in this frame, or up through the frames that hand it on, in the one that holds it
                the place, or a predecessor of it, still referenced                -> miss
                offset = that frame's base + the place's offset
                after  = the latest mark of the frame's floor and of every predecessor
```

A few compares and table reads: less host work than a free-list walk, and no bookkeeping on transients beyond the
flag that says a buffer is dead. A frame's entry checks the same for its block, and its exit checks that every
buffer of its block is dead and that nothing its structure still had due was skipped.

<a id="order"></a>
- **Order: reuse is made visible to the engine.** The engine orders dispatches by the stores they bind, and two
  tensors over the same bytes are two stores to it. So the binder carries the order. A **mark** is the engine's
  measure of how far its encoding has got (a pass and a level in it). The engine gives one when a buffer's last
  reference goes, when a frame closes and when an execution begins; every buffer comes back with `after`, the latest
  mark of anything that held any of its bytes: a dead buffer, a closed child frame, an earlier iteration of a loop,
  the execution before. The engine encodes the buffer's first dispatch after that mark (`core.dispatchLevel` takes
  the level as a floor when the mark is of the open pass; an older pass is already ordered by the fence chain), and
  a host write into it (`core.hostBytes`) waits for the mark's pass. This is the correctness mechanism, and it does
  move dispatches when a pass is cut between two users of some bytes: the stage gates count barriers and GPU busy
  time, not floor moves.
- **The host never writes arena scratch.** A host-filled ask comes from `H`: execution `k` uses replica `k mod D`,
  bumps through it, and begins only once execution `k - D` has run. `D` is a constant of the engine (2: one
  execution of encode-ahead).
- **A host read needs no memory**: `core.hostRead` waits for the writer's pass, and the store is alive while read.
- **Passes stay bounded in number and in length.** The host waits when `F` passes are committed and unfinished, so
  Metal's per-command-buffer memory is bounded and the margin can be small and measured; and a pass is still handed
  over by a dispatch count once the pool's `pass_freed_cap` is gone, or one pass would grow with the doubling kick
  and every reuse would fall inside it.
- **An execution that fails** pops every frame; the stores are the plan's own and stay valid for whatever still
  unwinds. The arena is empty again by definition.
- **In a test build** the engine poisons a place's bytes when it is handed out and keeps a canary after it, checked
  when its bytes are next handed out and at the execution's end: a kernel that writes past its output or reads bytes
  nothing wrote fails the test that ran it. A place whose last tenant ran in the open pass goes unpoisoned, since an op
  may be encoding that pass. `mlx.engine.poison` does the same for the pool.

Entry points: `arch/forward.zig` `forward` (prefill, decode, verify) replaces `cooler.enterForward`'s caps with
the execution's begin and end; `arch/drafter.zig` (draft), `sampler.zig` (sample and the round's verdict),
`arch/media.zig` (encode), `arch/prefix.zig` (capture, restore), `arch/sched.zig` `rowsForward` (joined rows: the
same phases at a wider row key). An execution's root frame is its phase and whatever of its shape decides its
structure.

<a id="tools"></a>
**Record-then-run.** A caller with a one-off shape (`sushi kld`'s 418-row chunk, a ubench arm, a layer-major
capture) records that exact execution, packs it, sizes an arena for it and runs it: exact by construction, at the
cost of the host work twice, which no tool minds.

<a id="borrow"></a>
## 7. Expert slots and the borrow

- **Slots are a pure function of the budget, at load**, as today (`expert_stream.budgetLedger`). What the stream
  makes while serving today becomes fixed at load: the spec maps (rewritten in place), the kept scales (reserved
  whole), and the union workspace, made at load when the plan lends.
- **A borrow is an alias, decided at load.** Let `G_keep` be the largest root block of the phases that run while
  pools are full (decode, verify, draft, sample) and `G_wide` the prefill's. The arena's backing is then one mapped
  region `[ H × D | G_keep | L ]` whose tail `L` **is** the slab memory of the layers the arch lends, latest first,
  enough that `G_keep + L >= G_wide`. The lent layers' slabs are views of that region; every other slab stays its
  own mapping.
- **A lease is a statement, not a remap.** An execution whose root block passes `G_keep` needs the lease
  (`PlanNeedsLease` otherwise): the stream marks those layers empty (their picks read through the union workspace,
  as today), the prefill writes over their pages, and the lessee's first narrow forward refills them from SSD
  (`restoreLayer`'s read, without its `mmap`). No unmap, no wait for macOS to unwire, no live headroom.
- **The lease's two edges are waits, because slab stores and arena stores over `L` are different stores.** Taking
  it waits for every pass that binds a lent slab (`borrowLayer` drains and syncs today) and begins the execution
  after that mark. Giving it back starts the refill only once the last pass that bound `L` as scratch has run: a
  fill is a `pread` on worker threads, and `core.refuseFillingReads` learns the arena's range over `L` as it knows
  mapped ranges. `expert_io.PageSlab` is given the arena's tail to host instead of its own mapping.
- **Memory is `max(slots, slots − lent + prefill arena)`, known at load**, and the chunk chosen at load counts it: a
  wider chunk is one that lends more. Hy4 and Qwen lend nothing; the mechanism costs them nothing.

<a id="state"></a>
## 8. Persistent state

Everything that outlives an execution is owned by a type that allocates it and states its bytes in closed form.
Shared owners live in `src/tensor/state/`; an arch's state module composes them.

| owner | holds | bytes |
|---|---|---|
| `Rows` | append-only rows in parts (Hy4's latent codes, scales, biases and rope rows; DeepSeek's compressed KV and index keys; Qwen's KV triples and QSA banks) | `sum(part row bytes) × capacity rows`, rounded to pages |
| `Fixed` | one block per conversation: rings, open groups, a drafter's hidden row and picks, the logits a step keeps, GDN and conv state as a written-in-place pair | from the config and the launch settings |
| `Points` | `n` equal slots for the small state a live point or a mark keeps | `n × point bytes` |
| `Media` | a request's image rows after the encode | from the request's grids, known at admission |

- **`Rows` is reserved once, at the reach.** Its backing is a virtual reservation at the model's maximum context,
  wrapped as a buffer up to the admitted capacity (the loader's own `core.wrapMapped` pattern). A later turn with a
  longer reach wraps a longer prefix of the same pages: no copy, no second buffer, no transient. Parking a
  conversation trims the wrap to what it used. If [S4](#spikes) rules the mapping out, the fallback is a plain
  buffer whose regrow is one copy between forwards, billed in closed form at admission (old capacity beside new).
- **Rows are written in place, at once, and never copied.** Rows below `used` do not change; an append writes past
  them. Today an append is a lazy `slice_update` (`mlx_pages.InPlace.write`, under `row_store`, DeepSeek's `Rows`
  and Hy4's rings): the update waits as a pending write that holds a forward's scratch until the bytes are next
  used, and `ops.settle` copies the whole base when the store is not sole (so does `kernel.launchInPlace`). Both
  break the plan: the scratch would outlive its execution, and the copy would land in whichever execution touches
  the state next. An owner's append is an eager write into its own bytes (as `hy4_latent.append` already is),
  settled before the execution ends, and there is no copy to fall back on.
- **A view names the rows it reads, and a rollback respects it.** A view of old rows stays valid under an append. A
  rollback below the end of a live view would let the next append overwrite what the view still reads; with no
  refcount to force a copy, `Rows` keeps the highest row any live view covers and refuses such a rollback by name.
- **A point is a slot, not a share.** Taking a live point copies the rings into a free `Points` slot; the caps that
  exist today (the model's four, the record's eight, one mark) are the slot count. Exhausting them is a named
  error, as a cap is today.
- **What a step keeps and the next step reads while it writes is a pair.** The generator may build the next forward
  before it has read the last one's logits (`generate.Pending`), and GDN state is read and replaced in one forward:
  each is two slots of `Fixed` written alternately.
- **Nothing made in a forward is kept by handle.** The survivors of today become `into` destinations in `Fixed`:

| arch | survivor today | owner |
|---|---|---|
| Hy4 | `last_hidden`, `last_picks`, the open group remade each forward, `generate.Pending.logits`, ring shares held by points, `Slot.vision_embeddings` | `Fixed`, `Points`, `Media` |
| DeepSeek | `pend_kv` / `pend_gate`, a `Verify` record's pending arrays, DSpark's stage rings, restored arrays copied on the next bind | `Fixed`; a restore reads straight into `Rows` |
| Qwen | GDN `ssm_state` and `conv_state` replaced every forward, the verify captures (`spec_state_seq`), the PLE window, the MTP head's cache and entry, `PersistentSsmGroup`'s merged copy | `Fixed` (a pair written alternately), `Rows` for the head |

- **Parked state is in the ledger.** A finished conversation's stores stay as `state` until they are flushed to SSD
  and released. Admission may release a parked conversation to fit a request; that is a policy over ledger lines, with
  no bytes to estimate.
- **Host memory that follows the context is reserved with the request**: the Engram history
  (`ensureTotalCapacity(reach)`), billed on the `host` line.

<a id="admission"></a>
## 9. Budget, context, chunk and admission

- **The budget is fixed when the load ends.** `B` is the planning ceiling read once then (`bills.Ceiling.planning`,
  as today: the declared wired limit less what stays unplanned, else the working set), less what MLX holds at rest
  and what other resident models bill. From there Cooler owns `B`.
- **The context is closed form**: `ctx = (B − model bill − fixed − points) / state bytes per token`, capped by the
  model's window. An explicit `--ctx-size` that does not fit refuses the load by name, at the load.
- **The chunk is chosen once, at load** (the owner's ruling): the context first (at the narrowest candidate), then
  the widest candidate whose arena still fits `B` beside that context, the borrow counted. The candidates are the
  ladder's rungs within the two flags below. The plan is recorded per candidate and the chosen one kept.
  `bills.chooseRequestPrefillChunk`, `adaptivePrefillWidthNow` and the re-ask after eviction go; a chunk boundary
  reads nothing.
- **`--prefill-chunk-max` and `--prefill-chunk-min` bound the load's choice** ([D4](#decisions)). The load takes
  the widest chunk that fits among `max`, the ladder's rungs between the two, and `min`. Equal values leave one
  candidate: the chunk is forced. Omitted, `max` is the model's widest rung and `min` its narrowest.
  - A load that cannot hold `min` (beside an explicit `--ctx-size`, when both are given) refuses with the numbers.
  - `min` above `max` exits with an error before anything runs.
  - **Only a load on the static bill has the two flags**: Hy4 at stage 4b, then DeepSeek and Qwen each at its own
    cutover. They are the arch's flags (`<arch>/<arch>_flags.zig`); on any other load (GLM, MiMo, an arch not yet
    cut over, an MLX opt-out) they are refused by name and `--prefill-chunk` keeps today's meaning, undeprecated.
  - On a static-bill load `--prefill-chunk N` is deprecated. It is still parsed and listed in `--help` as such,
    means `--prefill-chunk-max N` (a ceiling is what it is today), and says so once at launch; given beside
    `--prefill-chunk-max` it is an error. Its removal is a later release's call.
- **Admission**: `ledger.committed + request bill <= B`, then the allocation itself. Free RAM is read as a second
  gate that can only refuse (another program took the memory): the engine states it in the refusal and never names
  the program. A request that does not fit gets today's outcomes (wait behind running requests, release parked state,
  refuse 400), decided on ledger lines.
- **Per-forward caps go.** `core.budget`, `room`, `uncapped`/`capped`, `trimTo`, `pass_freed_cap`, the engine's
  share of `free_cap`, `arch/cooler.zig`'s `enterWide`/`liftWide`/`restCap`, DeepSeek's `cooler_own_caps`: none has
  a job when nothing grows.

**Observability.** `/props` `memory` gains `ledger` (each line, `budget`, `committed`) and keeps `active_bytes`
(now the ledger's sum). `[admission]` prints `needed= committed= budget= verdict=`. New tag `[plan]`: one line at
load (arena, `G`, `H`, the chunk and the bounds it was chosen within, the largest frames, packed against lower
bound, recorded or read from cache), a warning per miss that says whether the shape was learned in place or waits
for the next load. `[engine] since load:` gains plan misses and floor moves beside its late pipelines. From stage 8
`sushi memory-bill <pack>` prints the bill by owner with no load ([D6](#decisions)). `docs/envvar.md` gains
`SUSHI_PLAN_TRACE` (a diagnostic: the report in full); `SUSHI_ENGINE_POOL_PEAK_TRACE` and `SUSHI_TRANSIENT_PEAK` go
with the pool.

## 10. Enforcement

| holds | test | tier |
|---|---|---|
| C3 | **written**: generated forwards (nested frames, loops of any count, results handed up, carried values, loops of two kinds) recorded at their largest, then replayed at other shapes against a shadow arena: no two live buffers share a byte, none leaves the arena, each follows all that held its bytes (`plan_soundness_test.zig`) | host |
| C5 | **written**: the merge and the packer on recorded traces: the level rule, lifts, loops, last tenants, every ambiguity named (`plan_shape.zig`, `plan_pack.zig`) | host |
| C4 | **written**: each miss kind refuses by name with what was asked against what was recorded (`plan_binder.zig`) | host |
| C1 | **written**: the bill is the sum of its lines; a reserve past the budget is refused and changes nothing, under racing threads too (`ledger.zig`) | host |
| C5 | per arch and phase on the tiny fixtures: the dry trace equals the real trace, ask for ask | gpu |
| C3 | the planned forward's logits are the pool forward's bytes, with poison and canaries on (the pool forward goes once the plan is proven) | gpu |
| C1 | Metal's allocated size for the arena equals the plan's; the ledger's sum equals the engine's own count of what it holds | gpu |
| C2 | the census gains a column: a Cooler request asks the pool nothing and Metal nothing | gpu |
| C5 | release pins: each release pack's `G`, `H`, state rates and fixed bytes per settings and arm set, from a dry model; a pin is this test's proof and never a load's input (D3) | host |
| C2 | after a miss, tables that fit the arena held replace the old ones and the next such request is served; tables that need more are refused, allocate nothing and reach the plan cache (D5) | host |
| D4 | the two chunk flags: `min` above `max` is a launch error, equal values force the chunk, a load that cannot hold `min` refuses with the numbers, `--prefill-chunk` means `max` on a static-bill load, and a load off the static bill refuses the two flags by name | host |
| coverage | the random-shape replay at the release config | host |
| C4 | the fault sweep arms every miss site of a request, first and last (`core.fault.armAt`) | gpu |
| coverage | the release warm and the long-context rungs end with zero misses | pack, bench |

Guards that are not tests:
- **Comptime**: an arch that declares `cooler` with no opt-out may not declare a transient bill hook
  (`transientReserve`, `chunkTransientAt`, `chunkGrowthBytes`, `coolerPrefill*`): a compile error in
  `arch/cooler.zig`, as `cooler_own_caps` is checked today.
- **Lint** (`scripts/lint/`, on `std.zig.Ast`): `metal-alloc` (no `Device.alloc` or `wrap` outside `tensor/core.zig`,
  the plan module, the state owners and the loader); `frame-unpaired` (a `plan.frame` with no `defer` of its `end`).
- **Dry mode itself**: an unclassified readback and an escape cannot pass the first recording.

<a id="built"></a>
## 11. What is written, and the contract for the rest

The plan's core and the ledger are in the tree, pure and in the host tier (`zig build test-host`): nothing of them
reaches MLX or Metal, and no served path calls them yet. They are the part where a private design would cost the
most, so they are fixed first; the engine is wired to them, not the other way round.

| file | holds |
|---|---|
| `src/tensor/plan/plan_frame.zig` | `Site`, `Key`, `FrameId`, `Kind`; a call site's id at comptime, a key's hash |
| `src/tensor/plan/plan_trace.zig` | `Recorder`: what an execution asked for, frame by frame, with each ask's death, owner and dispatch levels; `PlanEscape` |
| `src/tensor/plan/plan_shape.zig` | `merge`: every instance of a `(site, key)` into one shape; `PlanAmbiguous` with the frame and what differs |
| `src/tensor/plan/plan_pack.zig` | `pack`: offsets, caps and predecessors; `Tables`, `G`, `H` |
| `src/tensor/plan/plan_binder.zig` | `Binder`: the replay, every miss by name, the host ring's replicas, the lease check, the order marks |
| `src/tensor/plan/plan_api.zig` | `frame`, `each`, `choose`, `bound`; the session a recording or an execution sets; `Driver` |
| `src/tensor/plan/plan_soundness_test.zig` | the end-to-end property test |
| `src/ledger.zig` | `Ledger`: lines, the budget, `reserve`, `release`, `move`, `snapshot` |

The stems carry `plan_` because the host tier selects tests by file stem and a stem that ends another file's
(`plan`, `pack`) would pull that file's MLX tests into the MLX-free binary.

**What the property test found, each now a rule above**: a predecessor list that stopped at a tenant a shorter run
never makes; a host ring sized by the largest probe, which a run combining two loops' heavier kinds passed; a probe
driver that could not reach a choice nested under another; and (stage 3's review, seed 467 of 6,000) a result an exact
child hands up from a loop inside it, which a short run never makes either, so it hides no earlier tenant. Each showed as a named miss or a shadow violation, never
as a wrong byte. Mutating a rule (the loop results' common death, a frame's inherited floor) turns the test red.

**The engine's side, for stage 1 and stage 3.** These are the only calls; each is one line at a choke point.

| engine site | recording | replay |
|---|---|---|
| an execution's entry (`arch/forward.zig` and the other entry points) | `Recorder.begin(root)`; `session = .{ .mode = .record, ... }` | `Binder.begin(tables, root, leased, mark)`; `session = .{ .mode = .replay, ... }` |
| its exit | `Recorder.end()`: the trace, or the escape by name | `Binder.end()`; on an error `Binder.abort()` |
| `core.freshFor` | `Recorder.ask(bytes, for_host, external)`; the store keeps the `Ref` | `Binder.ask(bytes, want)`; the store is `{ arena, got.offset, bytes }` and keeps `got.slot` and `got.after` |
| `core.dropStore` at zero refs | `Recorder.free(ref)` | `Binder.release(slot, mark of the store's last bind)` |
| `core.dispatchLevel`, per store bound | `Recorder.touch(ref, level)`, the level over the execution, never the pass's | a store's first bind in the open pass takes `after`'s level as its floor |
| `session.mark` | — | the engine's `(pass << 32) \| highest level in the open pass` |
| `plan.frame` / `each` in a forward | through `session` | through `session` |

<a id="fixed"></a>
**What may not change without the owner** (an implementor who needs one of these changed stops and says why):
1. The recorder is the only source of a size, a lifetime or a structure. No table is written by hand, and no
   function prices a Cooler transient.
2. A replayed ask lands at its recorded offset or is a miss. The arena is allocated at a load and is never
   re-allocated while serving (D5). There is no fallback allocation, no "grow the block", no second arena, and no
   path from an arena miss to the pool.
3. The binder decides only from its tables and from what the engine tells it (`release`, `exit`, `begin`). It reads
   no clock, no memory figure and no GPU state.
4. A buffer is placed only after every predecessor is dead, and it carries the latest mark of all that held its
   bytes. A change to the packer's predecessors or to the binder's marks runs the property test, and so does any
   change to the merge.
5. What a loop hands its parent is held to the loop's end, and nothing of the parent ends inside a loop.
6. A host-filled ask never shares bytes with anything in its execution, and never comes from scratch.
7. An owner reserves on the ledger before it allocates. The ledger never reads the machine.
8. The six miss kinds and their fields are the refusal's vocabulary: a new failure gets a kind and a test, never
   a reuse of `unplanned`.

<a id="build"></a>
## 12. The build

Every package is test-first, lands alone with every arch's bytes unchanged, and names the lock for anything on the
GPU. "A bill follows the storage in the same commit" holds throughout: a phase that moves to the arena drops its
term from the old bill in that landing. Until an arch's cutover its pool and the arena coexist; the engine's
recorder orders work on both.

**1.8.0 ships stages 0 to 5 (D17):** Hy4 on the static bill. Stages 6 to 8 are after 1.8.0; DeepSeek and Qwen serve
on today's code and bills, and the pool with its budget machinery stays for them.

**Stage 0. Spikes** ([below](#spikes)). Throwaway branches; each answer lands in this file. S1 and S2 gate the
rest: if a dry trace cannot be made equal to a real one, stop and go back to the owner.

**Stage 1. The recorder on the engine, with no behaviour change.**
- 1a. The recorder's hooks in real mode (the table above), behind the session: a served Hy4 forward on the pool
  yields its real trace. Red first: a tiny Hy4 decode whose trace has every ask the pool's stats count.
- 1b. Dry mode at the choke points of `core.zig`, `kernel.zig`, `stock.zig`, `route.zig`; dry state for
  `row_store`; `plan_api.Mode` names dry and real recording apart (D11). Red first: the dry trace of that decode
  equals its real trace, ask for ask, death for death and level for level (D9).
- 1c. `into`, `hostResult` and `core.freshOut`; the two-fill check; the load probes for the arms a first launch
  latches today.
- 1d. Hy4's frames, its `choose` and `bound` sites and its readback classification, until a recording is quiet
  for decode, verify and draft; the lazy caches made at load ([part 4](#caches)).
  - As built: the layer loop is `frame` by index, and inside it one `each` keyed by `Model.layerKey` (a sparse
    layer's open group, pooled keys, closing group and selection arm; a sliding layer's window filling, ring wrap,
    ring arm and edge). The resident route's picks are a `hostResult` whose group list is `bound` at one group a
    pick; a streamed route's stay a plain readback, refused dry, until its `choose`. The drafter's block is a result.
  - A probe whose layer closes a group plans alone: its pooled key goes in through a pending slice update, which
    holds the key past the layer, so that layer hands up one result more (merge: `exports`). With every
    `row_store` append settled at once, decode, verify at 2, 5, 8 and 16 and draft each merge into one shape on
    both tinies: eager appends (2a) are what make a phase one plan, and the decode probes that close no group do
    so today.
  - A process-wide cache (`side_cache.SideCache`, `exl3.linear`'s GEMM input ring) refuses a fill while a
    recording runs dry (`CacheFillDry`): its entry would be a store with no bytes that a later real forward binds.
    After a real prefill the tinies' decode, verify and draft fill none.
- 1e. **The audit**: a host-tier test plans the Hy4 release config from a dry model and prints, per phase, the
  packed peak beside today's `hy4_bill` term. This is worth landing on its own: it is ground truth for the old bill
  at release shapes and at the full context, which no test has today.
  - As built: `hy4_plan_audit_test.zig`, in `test-hy4` (the forward reaches `mlx.zig`, so not the host tier), no
    pack. The dry model is the release's config and its 1,964 tensors as the loader keeps them, shapes and no bytes,
    experts resident, each Engram table resident (a served load reads it by row) and each token map a ramp; the
    probes latch for real first. It prefills dry in 4,096-row chunks to 4k, 32k, 128k and 1M - 64 keys and records
    a decode, verify blocks of 4, 8 and 16 and a DFly2 draft at each, kv8, fill zeros, survivors state-owned; each
    recording merges and packs on its own. 40 s on the M5 Ultra; `SUSHI_PLAN_AUDIT=<abs file>` writes the table.
  - At the tree of 1d (9c6aa11c), the arena packed by lifetimes alone beside the old bill:

    | phase | rows | keys | packed | `hy4_bill` | of it, state grow | packed with the level rule | host ring |
    |---|---|---|---|---|---|---|---|
    | prefill | 4,096 | 4,096 | 2.79 GB | 4.23 GB | 3.5 MB | 242.76 GB | 16.70 GB |
    | decode | 1 | 4,097 | 67.7 MB | 287.9 MB | 3.7 MB | 1.63 GB | 2.2 MB |
    | verify | 16 | 4,125 | 130.1 MB | 557.4 MB | 3.7 MB | 5.75 GB | 0.7 MB |
    | draft | 6 | 4,125 | 70.6 MB | 719.3 MB | - | 7.46 GB | 0 |
    | prefill | 4,003 | 1,048,512 | 2.73 GB | 5.25 GB | 891.3 MB | 382.24 GB | 16.27 GB |
    | decode | 1 | 1,048,513 | 67.7 MB | 1.55 GB | 891.3 MB | 1.64 GB | 0.2 MB |
    | verify | 16 | 1,048,541 | 280.5 MB | 2.75 GB | 891.3 MB | 7.53 GB | 0.7 MB |
    | draft | 6 | 1,048,541 | 70.6 MB | 719.3 MB | - | 7.46 GB | 0 |

    A forward of 16 rows, measured here as a verify, is a prefill block under D18 (verify is 2 to 8 rows).

  - The old bill covers every phase at every context: 1.5× a prefill chunk, 4 to 6× a decode or verify block below
    128k, 10× a draft; at 1M its key-scaled terms and the state grow (891 MB, which the arena holds none of: those
    stores are the state's) take it to 10 to 17×, while a decode's arena stays 67.7 MB at every context.
  - The level rule over the execution's levels packs nothing usable: a dispatch that reads only the load's state (a
    weight widened by block, a scratch's first write) sits at level 1 for the whole execution, so no byte is ever
    reused under it (87× the lifetimes at a prefill chunk, 24× at a decode). 3a's floors are what make the arena.
  - The host ring as the packer sums it is 16.7 GB a 4k chunk: `fwd.placedZeros`, the [T·k, latent] f32 sum each MoE
    layer's groups land in, is 403 MB of host-written zeros a layer; a GPU fill would take it off the ring.
- Stage 1's rules, as built:
  - A dry recording on the load's own model leaves its state unusable: a store it made has no buffer, and one from
    before it that a dry dispatch wrote (`Store.dry_written`) no longer holds what its tensors say. Real work refuses
    both by name (`DryStore` at a host read, a panic at a bind) until they are freed: reset the state after a probe.
  - `core.into` is taken by a launch's output alone (`kernel.launchWith`), never by an elementwise op (as often an
    input's copy or a cast inside another), never by a launch that reads it; 2b passes a target to a composed op.
  - One wide-mix grant answered every form, probed at load on the forms the geometry takes; 2e gives each pipeline
    its own.

**Stage 2. Hy4's state, still on the pool.** State comes before the arena: a forward's survivors need an owner to
be written into before the binder may call one an escape, and a store that regrows makes a frame's asks depend on
history.
- 2a. `Rows` under `row_store` (reserve at the reach, eager in-place appends, wrap growth, the view check).
  - As built: every in-place append of Hy4's state lands at once on the engine (`mlx_pages.InPlace.landed`, after
    `row_store`'s put and Hy4's ring write; `row_store` is Hy4's alone, so no other arch moves). A group-closing layer
    hands up nothing more, and the merge's `exports` refusal is gone; what still parts the probes is the survivors'
    flags (2b). Eager appends do not depend on S4: growth is today's copy until the reserve below.
- 2b. `Fixed`, `Points`, `Media`; every survivor of a Hy4 forward goes `into` its owner.
  - As built for Hy4: `Model.reserveFor(reach)`, called by the served forward at the request's end (`KVCache.end_tokens`)
    and between forwards only, reserves each sparse layer's rows and pooled keys at `capBound` of the reach (what
    `hy4_bill.requestBill` already holds; growth is the design's fallback, a copy between forwards, so it does not wait on
    S4), and makes each sliding layer's ring and each sparse layer's open block. The open group, the drafter's
    handover and the picks are written into fixed blocks in place (`writeFixed`); a point or mark copies the small
    state (`copyOf`, the engine's `copyOwned`) instead of sharing it, and a rollback or restore writes it back into the
    state's own blocks. The layer key counts the groups a forward closes (the pool takes their rows one at a time
    from two). `Fixed` and `Points` are these blocks and copies on the pool, not yet slots in an arena; `Media` and
    the generator's kept logits are not moved (they belong to the encode and sample phases).
  - The reservation is the running request's: `reserveFor` sizes each store to the bill of its reach, grown or cut
    back (a parked state from a longer request holds no more than the next one's bill), never short of the rows the
    forward writes; a rollback inside the request keeps it (`Model.reach`), so the verify after a rejected block
    grows nothing either.
- 2c. A host map keyed by a store's pointer takes another key before its phase runs dry: a dry store has no address
  of its own (`hy4_prefix.zig`'s held points).
  - As built: `Held` keys a store of ours by the store (`core.storeKey`), not by its bytes (`storePointer`), which
    every dry store shares.
- 2d. `fwd.placedZeros` filled on the GPU, off the host ring (D14).
  - As built: on the engine `placedZeros` takes `ops.zerosOnGpu`, a pass that writes the zeros (arena bytes); every
    other `mlx_zeros` still fills on the host, so no other arch moves (D17).
- 2e. Hy4's wide-mix grant latched a pipeline (D15).
  - As built: `hy4_hc` keys a grant by the launch's pipeline (form, template values, input dtypes); a served launch
    latches its own. Before a model's forward is recorded dry, `Model.probeWide` records dry forwards of 1 to 8 rows
    with `hy4_hc.census` set, which notes each wide pipeline with no grant where a recording would refuse it, and
    `probeNoted` launches each once on zeros, for real. It is not a load step: a load whose forward cannot record dry
    (a streamed BF16 route reads its picks plainly) would fail it, so whoever records calls it (stage 3: the plan's
    build). The census runs at position 0: a pipeline only a later position reaches is refused by name.
- Gate: restore and rollback bytes equal; a recording of each Hy4 phase ends with no escape; with appends eager,
  decode, verify at every block width and draft each merge into one shape on the tinies (D12).
  - As checked: `hy4_plan_test`'s gate cell records decode, verify at 2 to 8 rows and a 12-row prefill chunk dry
    from a served request's state on both tinies, and the draft on `HY4_STQ_TINY` (the one that ships DFly2, loaded
    with it); none ends with an escape, and each narrow phase merges into one shape (the prefill's probes are planned
    apart: span and group frames are stage 4's). A drafter layer is a frame keyed by whether it reads its target's
    selection. The restore and rollback cells (`hy4_oracle_test`, `hy4_dfly_test`) give the serial run's bytes.
- Dry state, as built for Hy4 (stage 2): `Model.dryState(n, reach)`, inside a dry recording, reserves the state for
  `reach` and sets what a prefill to `n` leaves (each store's `used`, the open groups, the rings' positions, the
  history), with no forward run. A decode or verify block from it records the trace one from a prefill to `n` records,
  with the drafter's handover kept and without (`hy4_plan_test`); the picks' blocks are reserved with the handover's,
  so the first selection past the threshold writes them in place. The audit reaches each context this way: 3.8 s for
  its 44 recordings to 1,048,512 keys, where walking the prefill dry took 39 s for 24.

**Stage 3. Hy4's narrow phases on the arena.**
- 3a. The arena, the binder's hooks, the marks and floors in `dispatchLevel`, the host ring, the bounds on passes,
  poison and canaries; what follows a miss (D5): the one shape recorded after the request, the tables swapped
  between executions when they fit the arena held, the shape cached for the next load when they do not.
  - As built, the arena and the hooks (`core.zig`, "The arena"): `arenaFor(layout)` makes one buffer `[H x 2 | G]` in
    residency sets of its own kind (`metal.Device.allocArena`), keeps it while it holds each part, and remakes it
    larger only between executions. `beginExecution(tables, root)`, `endExecution` and `abortExecution` set and clear
    the replay session; every ask between them is a place (`arenaStore`), a store over the arena at the binder's
    offset, its record recycled. A host-filled ask takes this execution's replica of the ring, whose first write
    waits once for the execution two before it.
  - Marks and floors: the engine's mark is the open pass and the highest level of its dispatches (`markNow`); an
    arena store's is its last dispatches' pass and level (`markOf`), given to the binder when its last reference goes.
    A place whose `after` names the open pass starts all three level maps at that level, so every dispatch that binds
    it, a disjoint write included, goes above the last reader of its bytes; a place whose `after` names an earlier
    pass is ordered by the fence chain. `Stats.arena_floors` counts the places that took a floor.
  - Fail closed: a store alive when its execution ends is a `lifetime` miss (the engine judges a host ask, which
    holds no slot of the binder's) and is escaped: binding it panics, a host read or write, a lend to MLX or a slice update owed from it is
    `PlanEscape`, and the arena does not go while one is alive. An execution ended with a frame open is a `structure`
    miss. Inside an execution an MLX import is refused, and so is a host write into arena scratch. A place that fails
    after the binder gave it goes back to the binder dead. `Metal.allocatedSize` of the arena is the engine's count
    (`arenaTotal`, the layout rounded to pages). Each planned load holds the arena (`arenaFor`, `arenaRelease`); a
    larger one is made before the old goes, so a refusal leaves the held one.
  - Bounds on passes (`execution_pass_bounds`, an execution's alone, so no other arch's passes move): a pass is handed
    over at 512 dispatches, and before one is committed the host waits until fewer than 16 are unfinished. Both are first values for S6 and
    the stage gate to confirm; `takePassHighs` reads the longest pass and the most in flight.
  - Poison and canaries (`core.arena_check`, a test build's): packed with `Options.guard` (rounded up to the
    alignment), each place keeps that many bytes after its own; with `byte` set, a place is handed out over it once the
    passes already handed over have run, and its guard holds a canary, checked when its bytes are next handed out and at
    the execution's end. `hits` counts a changed canary. A place whose last tenant ran in the open pass is handed out as
    it is (`skipped`): an op may ask for scratch with its pass open (`reduce.all`), and no wait may commit that pass.
  - What follows a miss (`plan_book.zig`): a model's `Book` holds, per root, the traces of its probes and the tables
    packed from them (`layout` is the arena they need). `learn(root, trace, held)` packs the root again with the trace
    the missed shape recorded: tables that fit the arena held replace the root's (`swapped`, between executions); tables
    that need more leave it as it was (`next_load`), and the arch keeps the shape (`keepShape`: its phase and four
    numbers, a line of `plan-shapes` in the load's cache directory, written through a temp file) for the next load to
    record (`readShapes`). Hy4 records the shape and calls `learn` from 3b.
  - D22: `scripts/lint/errdefer_handover.zig`. Its first run found three more of the class, each closed by ending the
    errdefer's block before the store: `dsv41_prefix.markRows` (an eval failing after the slot adopted its zeros),
    `qwen4_gdn`'s spec capture (a replay hold failing after the state took the conv input), and a `kv_disk_cache` test
    helper. Error paths only: no bytes move (D17).
- 3b. `decode` on the arena; then `verify` 2..8 (D18), `draft`, `sample`. Each: bytes equal on the tinies and the release
  pack; the term leaves `hy4_bill`.
  - As built, decode (`hy4flash/hy4_plan.zig`): a root is a forward's rows, readout and drafter handover
    (`root(rows, readout, keep_hidden)`). A resident load plans its one-row forwards at its warm's end, before the seal
    (`arch/cooler.zig` `planLoad`, `arch.planBuild`): the wide pipelines latched first (`probeWide`, D20), then each
    root recorded dry under both fills at every decode position and packed, the kept shapes learned, the arena made.
    A served forward of a planned root is an execution (`run`); any other runs on the pool. A load whose plans fail to
    build does not serve.
  - The decode positions (`decodePositions`): every residue of a group just below and just past each edge of the
    layer key (the first group, the window, the selection's threshold, the top-k's split), and the end of the
    context: the largest position of each key holds its largest sizes. A row at p scores the groups closed by p + 1, so
    the split's edge is the position before `topk_split_min_n_one_row` groups' rows.
  - Two facts joined `Model.layerKey`, each found as an ambiguity between two probes of one key: `split`, the top-k
    over 131,072 groups or more taking passes and scratch of its own (`sparse_select.topkSplits`; at the release
    config from position 524,287); `lone`, a forward whose only key is its own row, whose one-row operands an op lays
    out as their transpose and never copies.
  - Logits (D19): `Model.forwardServed` writes a planned forward's logits into one of two fixed blocks of a verify
    block's rows (`writeFixed`), in turn, and hands the caller a view. A dry recording writes blocks of its own, made
    for it, so the caller's view and the turn are left as they were.
  - A miss refuses the forward by name (`PlanMiss`), the state reset as a failed forward leaves it; the shape is
    recorded and learned (`Book.learn`), and kept either way in the plan cache, the directory of the load's pipeline
    manifest (`plan_book.Cache`), so a restart plans it too. A load learns each kept shape with no bound on the arena;
    one it cannot run, record or merge is passed over by name, and the load serves without it.
  - Every load plans (D20). A stream that holds every STQ expert resident maps each rate-grouped layer's picks to their
    slots on the GPU (`Engine.residentRoute`); that is how a served load takes a pack its budget covers, so a streamed
    config is the usual one. A streamed route's picks are a readback a dry recording makes up (`hostPicks` with the
    expert count): its groups are `bound` at one per cache's worth of the picks' or the router's experts (`mostGroups`), a group's picks at every
    pick, and the BF16 route's one-group and many-group arms are a `choose`, which keys the loop and its iterations;
    each group is an iteration of a loop frame, the first apart. `Engine.prepareDry` gives a dry recording the slots
    an all-hit route takes, with no fill and the cache untouched, or, where a rate group's pool can hold fewer slots
    than a route's distinct picks (`Engine.routeMayOverflow`), the union workspace's as a `choose` holds it; the
    routed product's frame is keyed by the bank it read.
  - A probe records a pass for each value of each `choose` it meets (`hy4_plan.record`, the driver's passes), and the
    passes are learned together (`Book.learnAll`). Both arms of a choice hand up the same arrays: the many-group sum's
    workspace is freed inside its frame, and the STQ route's first group carries a copy of its product, so every
    group's product dies inside its iteration.
  - The merge held a loop's result where the key that lifts it furthest says, but recorded its bytes where the
    ask's own key said: a first iteration whose carried sum dies in the next left the loop's owner a cap of 0, and
    every such forward missed, the relearned trace the same (D20's review, F2). The walk now goes up while a frame
    passes the result on. A load that streams with `--expert-cache-bias` plans too (D102). A load's request bill
    holds no one-row forward, and the arena counts in the engine's live and peak bytes from the load's end
    (`gpu_mem.activeBytes`), as its weights do.
  - An engine op whose arm follows its sizes frames itself, keyed by the arm, so each arm is a variant of its own:
    the first is `matmul.steel` (split-K or fused, `steelPath`), found as the release's ambiguity at 512 keys, where
    the keys pass the latent's width (`kv_lora_rank`, now an edge). Under a plan, split-K's partials take their most
    partitions: the count steps with the tiles, so a product's partials can peak inside the arm (196,608 B at 96 and
    192 keys for the score product). Every split-K call shares the arm's variant frame, whose largest instance holds
    them today; the hold keeps that true whatever the call sites, and a cell pins it. NAX split-K's count is 3 at an
    odd inner axis to 1,024 and 2 past it to 8,192; under a plan it takes at least 3.
    Section 5's `plan.boundaries` (every op registering its thresholds) is not built: an arm the key and the edges miss
    shows as an ambiguity when the load builds (it does not serve), or as a miss the sweep below finds.
  - A root carries the KV width it was recorded at, the load's (`bills.configuredKvQuantFor`); a request that names
    another runs in the region (D97), as does a forward that splices an image's rows; an image request's decode past
    its image is the text forward and is planned
    (`Model.splicesImage`). A forward that splices moves the request's splice offset past the rows it consumed, so an
    image token a decode samples is a text token there.
  - The arena is measured, never priced in closed form (D98): the plans recorded before the warm and the region at the
    load's chunk, the larger of each part; the chunk's choice prices it and the seal's model bill holds what Metal made.
    A served miss's recording makes its logits blocks dry, beside the state.
  - Checked: on both tinies, a one-row forward at each edge runs as one execution with the pool's bytes and no ask of
    the pool, canaries clean; a shape no probe recorded misses, is learned and kept, and is then served. A dry model
    of the release builds every root's plan to 1,048,575 with no ambiguity, and a sweep of 1,460 positions (every one
    to 600, then denser to sparser to 8,200, either side of each power of two to 1M, 100 seeded) replays against it
    with no miss (`SUSHI_PLAN_SWEEP`). On the release pack, the bytes to 8,192 positions (`HY4_PLAN_PACK`).
- As built, verify (stage 3b): a load that drafts plans a DSpark round's block of 2 to `verify_rows_max` (8) rows,
  every row read out and the handover kept (`hy4_plan.verifyRoot`); a wider round is a prefill block, on the pool, and
  its term stays in the request bill until stage 4. A block's probes are candidates either side of every edge its
  layers' keys turn at (a block's own split count, the window's second wrap, the end's last window), and of those that
  take one set of keys the largest (`blockPositions`): 301 probes for the release's 11 roots. Two facts joined
  `Model.layerKey`, each found as a refused merge on the tinies: `first` (a block at the start reads no state) and
  `cut` (a block across the selection's threshold runs the rows before it dense, the rest selected: either part one
  row is a one-row operand). The STQ embedding's distinct id blocks are a `bound` (`min(t, vocab / 128)`). On the
  tinies, every width at every position the prompt holds gives the pool's bytes, logits and handover.
  - The arena's bound holds a verify block of 8 rows at the load's reach where the load drafts; the request bill
    drops the 8-row form where the load plans.
- As built, draft (stage 3b): DFly2's draft (`Drafter.draft`, the block `[t1, mask x draft]` from the target's handover
  and its shared layers' rows) is a root of its own, served through `hy4_plan.runDraft`. A probe makes the state
  `n - 1` rows deep, runs one row dry with the handover kept (the hidden row and each sparse layer's selection, as a
  verify's last row leaves them), then records the draft (`draftPositions`: either side of the window, the latent's
  width and the selection's threshold, the first rows, the end). The gather's bound keeps the ids' own blocks (a
  draft's mask id names another than its anchor's). The draft's term leaves the request bill where the load plans at
  its KV width; the arena's bound holds a draft at the load's reach. On the tiny that ships DFly2, a draft at every
  depth the prompt holds gives the pool's drafts and confidence; the release's 12 roots take 346 probes, 2.3 s, and
  the arena stays 74 MB.
  - A dry recording's host asks of 64 KB or more take zeroed pages, backed only as touched, and a dry `zeros` writes
    nothing: every probe had zeroed the state reserved for the whole reach twice. The release's build went from
    50.5 s to 2.0 s.
- As built, the rest of a DSpark round (D40 to D43): its mark and its rollback's replay. Measured on the STQ tiny past
  the seal (`core.closePool(.census)`), a round asked the pool 16 times for its mark and 348 for its rollback.
  - The mark is storage-owned (D40): `Model.mark` copies each sliding layer's ring, each sparse layer's open block
    (whole, its rows a count) and the handover into the model's mark blocks (`mark_fixed`, `mark_hidden`), in place;
    no part of it follows the position. `reserveFor` makes them with a drafting request's state, the request bill
    holds them (`hy4_bill.markBytes`, D56) and `ramTierBytes` no longer does; one mark stands at a time (`Hy4MarkHeld`). A
    live point keeps its own copies (`point`), the code it shares with the mark (`rollbackFrom`) unchanged for it.
  - The replay is planned (D41): `rollbackTo` runs its kept rows through `hy4_plan.run`, a root of 1 to
    `verify_rows_max - 1` rows read out by none with the handover kept, probed where the one-row forward (one row) or a
    verify block of its rows is (`decodePositions`, `blockPositions`). After the rollback the state is a plain forward's
    at the mark: the rows cut back, the rings and open blocks written back into the blocks they were made in, so a dry
    state at the mark's depth records it. A wider round's replay stays on the pool (D42).
  - Checked on the tiny that ships DFly2: a round at every depth to 24 and from 116 to 136 keeping each of 1 to 7 of
    8 rows gives the pool's logits and state in two executions, its mark and rollback asking nothing of a closed pool;
    served rounds (`Generator`, rows forced) give the pool's tokens and state, each round its draft, verify and replay
    an execution, the one ask left the verdict's (the sample root, D24). The release's dry build takes 19 roots from
    512 probes (12 from 346 before) in 3.9 s (2.5 s), M5 Max under background QoS beside other work; its arena stays
    74,491 KB, the verify blocks' still the largest.
- As built, capture and restore (D50 to D54). Measured on the BF16 tiny at 512 rows with `core.closePool(.census)`
  (stacks printed at each closed ask, aggregated by caller): a live point asked 12 times (`captureFixed`'s copies), a
  stored point where the live one is 13 (with the checkpoint's placeholder), a restore from a RAM entry's point 24 (the
  rings' copies, then the restored point's own), from SSD 87 (with the tier's read: 33 arrays from the point's files,
  30 from joining its row segments); a write-through, a publish, a restore in place and one from a live point asked
  nothing.
  - Points are slots (D50): `Model.point_fixed` and `point_hidden`, 12 (`point_slots`: the request's 4 and the
    record's 8), made with the request's state where the mark's are; a point copies its small state into a free slot
    in place, in the mark's form (`copySmall`, shared by both), and a rollback writes the slot back (`heldBlocks`). The
    record's points are slots of the model: a point names its slot's take, `Live.clear(m)` gives them back, and
    `Live.deinit` touches none. The bill's term moves from the copies beside the state (`ramTierBytes`) to the state
    (`pointsBytes`, the same closed form); `heldBytes` counts the slots once.
  - A restore of a stored point installs into the request's state (D51): `restoreArrays` empties the state in place
    (`empty`), reserves it for the point's rows or its reach, the larger, appends the rows into its stores and writes
    the rings and open blocks in place. A stored point's absent arrays share the model's placeholder (D53).
  - Checked on the BF16 tiny (`hy4 points:`): each capture (a live point, a stored point, the SSD write-through, a
    publish behind the reply) and each restore (in place, from a live point, from a RAM entry's point, a point the SSD
    tier read back) asks nothing of a closed pool, and each restore gives a cold prefill's state and the logits of the
    chunk and the decode row past it. The tier's own read of a point is not yet on the bill (D54).
- Gate (the fast bench, decode): tok/s, barriers a forward, GPU busy and host ms a token equal or better than the
  pool, arms interleaved. On the M5 Max (a throttling laptop: the Ultra holds the timed gate), the release pack
  resident, the decode meter at 4k: by lifetimes alone 1,048 barriers against 911 and 1.2% slower; with the level
  rule (D23) 910 barriers, 1.5% faster, 19% less host CPU, decode 48.2 tok/s against 47.7. A cell pins the tinies'
  barriers to the pool's but a few (the logits' write, D19, is one dispatch the pool does not make).

**Stage 4. Hy4's wide phases.**
- 4a. Span, group and tower frames; `prefill` at the chunk ladder's rungs.
- 4b. The chunk at load; `--prefill-chunk-max` and `--prefill-chunk-min` for Hy4 alone, `--prefill-chunk`
  deprecated on a Hy4 load and untouched elsewhere (D4: the Settings twin, `--help`, `docs/flags.md`,
  `docs/usage.md` in the same landing); the adaptive width and the per-request chooser leave Hy4.
  - Built: the two flags are Hy4's own (`hy4_flags.zig`), refused by name on every other arch's load and in its
    settings file and API (`arch/flags.zig`, `model_settings.unavailable`); `--prefill-chunk N` on a Hy4 load is the
    max, said once at the load, and refused beside `--prefill-chunk-max`. `bills.fixStaticPrefillChunk` runs before the
    warm and the plans, on `pinPrefillChunk`'s reading; it writes `ModelConfig.static_prefill_chunk` (and the pin) on
    the registry's config and the trunk's copy, which 4a's `hy4_plan.build` reads. With it set the load is off the
    per-request chooser and the adaptive width, the chunk loop merges no tail (`prefill_plan.chunkEndFor`), and the
    served warm runs the fixed chunk or refuses. The prefill runs in the arena's region (D97); a load on the ledger
    fixes its chunk on the arena it measures (D98).
- 4c. `encode`, `capture`, `restore`; record-then-run for `sushi kld` and the layer-major capture; the pool closes
  at the seal for Hy4.
  - As built, sample (D24, `hy4flash/hy4_sample_plan.zig`): every draw of a served request, from the logits to its token,
    is one execution of a root keyed by its mix (`sampler.Mix`, D34). The seam is `arch/cooler.zig` `Sample`: the
    generator (`Generator.sampleMasked`, `sampleSync`, the first token, a forced token's logprobs) and DSpark's verdict
    (`dspark_round.sampleVerdict`) begin a sample with its mix, run the shared sampler, and end it; an arch that declares
    `sampleBegin` answers, every other load samples as before. A step's token outlives its execution in one of two
    token blocks written in turn (D36). The load records each mix it can take dry (`mixes`: 1,233 at the release, two
    value sets a mix under both fills), packs them into the forwards' book after `hy4_plan.build` and makes the arena
    again where it needs more (D39): on the release's dry model 718 ms, the scratch unchanged and the host ring from
    444 KB to 1,892 KB a replica (the penalties' counts and ids and the verify rows' grammar masks). A mix no probe
    recorded is refused by name (`PlanMiss`), recorded and learned, and kept as phase 3. A draw that fails inside its
    execution lets go the token block it was writing (it owes a write from the arena's stores, which escape), as a
    forward's miss resets the model; `remakeTokens` makes it again outside an execution before the next draw.
  - On the tinies, every mix runs as one execution with the pool's token and verdict and asks the pool
    nothing; a served request's pipelined decode asks it nothing at all. The serial draw's (logprobs, penalties) next
    forward still takes its id from a host array (`generate.idsArray`), one pool ask a token: the forward's input.
  - As built, encode (`hy4flash/hy4_encode_plan.zig`): an image's tower and merger are one execution of an `encode`
    root keyed by `Class` (one attention chunk or several; patches and merge cells to the vector kernels' 15 rows, each
    count its own; a merged grid one row or one column across, where the merger's reshapes are views). Blocks are one
    frame, attention chunks a loop; the patches' upload and the position tables are the execution's host asks. A load
    whose loader map keeps the tower records it dry beside its forwards (`hy4_plan.buildWith` from `arch.planBuild`):
    for each count of cells probed (each to 16, each class's most, and every count at which a tiled product's path may
    turn, `matmul.steelRowEdges`), the grids of its classes with the most cells and the most merged rows, to the most
    patches the load accepts (`maxImagePatches`); a host cell checks that every grid the load accepts has a probe of its
    class with as many cells and as many rows at most the next count probed. An image the processor resizes past that
    most (an extreme aspect: a side's floor of one cell outweighs the pixel bound) is refused by name at admission.
  - `Media`: a request's image rows, [1, rows, out_hidden] f32 over pages mapped for it (`core.wrapOwned`: a state
    owner's mapping, written in place while sole), sized from its grids; each image's encode writes its rows into it,
    `vision_queue` hands it to the slot as the splice's rows, and it is unmapped with the request. Its closed form is
    the request's encode bill (`hy4_media.mediaBill`); the planned encode's scratch is the arena's, measured with it.
    Checked on the BF16 tiny: at one chunk, two, a short last one and the most patches (square
    and one cell wide), one execution with the pool's bytes and no ask of it, canaries clean; a request of two images
    encodes into one Media asking the closed pool nothing (`closePool(.census)`), and its prompt answers as the tower's
    rows joined do. A dry model of the release (BF16 tower, 9,216 patches at most; commit `3c24db102`, M5 Max): 113
    encode probes, 43 roots, the build 2.36 s to 2.57 s; the arena 72.7 MB to 2.44 GiB (scratch 2.39 GiB, host ring
    27.3 MB x 2).
  - As built, the close (D57 to D63; `hy4flash/hy4_close_test.zig`): past the seal, under `core.closePool(.census)`,
    served text requests on the STQ tiny (greedy, sampled, logprobs and penalties, DSpark greedy and sampled), an image
    request on the BF16 tiny (its encode into Media and its prompt) and a scheduler restore from the live record each
    give the plans-off tokens and ask the pool nothing outside the named sites (D63). Not yet closed: a RAM entry's
    restore in the served path, and Metal's bytes: a restore request's prefill tail on the pool moved the engine's
    bytes by 304 KB on the BF16 tiny, which stage 4a's planned prefill removes.
  - As built, the switch (D64): the pool's gate is a model's, kept in its turn slot (`BudgetSlot.gate`, which a
    forward's cap never overwrites): `warmEnded` sets it to `.refuse` for a load that serves on its plans (an arch's
    `arch/cooler.zig` `plansForwards`: an arch on the ledger), a turn of that model (`turnToModel`) puts it in force, and every turn's end
    gives back the gate it found; a turn of MLX's side, of another model or of no model runs open, so other archs in the
    process keep their pool (D17). The open sites are D63's, and a DSpark verdict wider than a verify block
    (`hy4_sample_plan.begin`, a forced width) draws in a wide phase of its own. A warm opens its load's gate until its
    seal (`arch_cooler.warmBegins`), so a test that warms a sealed load again runs its warm as the first. A test's
    plans-off arm opens the pool by `widePhase` for that arm; a test that drives a load's internals past its seal
    (forwards off the served path, raw ops) takes `hy4_contract.openTurn`, whose pool stays open and counts what a
    planned forward asks; the gate cell (`hy4_close_test.zig`) runs every served case under `.refuse`.
  - As built, the SSD tier's read into the state (D65): `DiskTier.restorePointInto` reads a point's tensors straight
    into bytes its caller names (`PointSink`: a snapshot layer's array and the row it starts at), the row segments and
    the point's own file each at their rows, on both readers (`prefix_vfs_tensors.readInto` on the layer,
    `readBytesInto` for a held point); Hy4's sink (`hy4_prefix.restoreDiskInto`) names its `Rows` parts, pooled keys,
    rings and open blocks (`Model.restoreBegin`, `restoreDest`, `restoreEnd`). Checked on the BF16 tiny at 512 rows on a
    256-token grid and at 602 past it: the restore asked the pool nothing, wide phases included (78 and 93 asks of the
    closed pool before), and gives a cold prefill's state and next logits. On the bill, `hy4_bill.restoreBytes` is 0 at
    any depth: the release config's request restoring 900K tokens is billed no more than its cold twin.
- Gate: prefill tok/s at 4k and 8k equal or better; the audit's arena is what Metal reports; the census shows a
  Hy4 request asking the pool and Metal nothing.

**Stage 5. Hy4's cutover.** The budget at load, the closed-form context, admission on the ledger, `/props` and the
log lines; Hy4's margin becomes the measured constant (D7; kept at `BILL_MARGIN_BYTES` until S6 measures the
in-flight peak, D109). `hy4_bill.zig` and Hy4's hooks in `bills.zig` are
deleted. The hunt closes the phase.
- Gate (2026-11-24, A = `c5fdbebab` the stage 4 head, B = `a34d07d2e`; Hy4 STQ 4bpw, `tests/bench.sh --rungs 4k,8k
  -- --no-mtp`, A B B A, no compile inside a request in any arm): on the M5 Ultra prefill B/A is +0.4% / −0.2% (code
  4k / 8k) and +1.5% / +1.6% (prose), decode level (66.4 to 66.8 tok/s); on the M5 Max +4.5% / −0.2% and −0.3% / −0.1%,
  decode +0.4 to +1.2%. A streamed load (`--ssd-budget-gb 60`) prefills at 380 to 432 tok/s against the stage 4 head's
  86 to 125 (a route's groups by expert). A build's first load makes every layout's matrix tiles (about 4.5 s more
  warm on the Ultra); each load measures its region (about 3.8 s); a same-build load makes all from its manifest.
- As built (the cutover's ledger, D80 to D89; `hy4_bill.zig` deleted, D104 and D105): `static_bill.zig` opens a load's `Ledger` with `B`, commits the model bill at the
  seal from the engine's own counts, pins the closed-form context and refuses a `--ctx-size` that does not fit; a
  request is admitted on the lines (`ledger.decide`), reserves its terms at its reach (`static_bill.Reservation`, the one
  price every reader of a request's bill takes: admission, the connection thread's check, the scheduler's hold, a
  refusal's figures) and parks what the model holds where the inference thread finishes it; `/props` shows `memory.ledger`. Hy4's terms in closed form
  are `hy4_ledger.ledgerTerms`. Release config, resident, dry: 1,048,576 tokens at chunk 4,096 on 128 GB, 8,192 on
  256 GB ([engine-memory-admission](engine-memory-admission.md#the-ledger-a-load-on-the-static-bill)).

**Stage 6. DeepSeek-V4.1** (after 1.8.0, D17). `choose` on the streamed route and its frames; `Rows` under `dsv41_model.Rows` (its
`bind` is already the reserve); `Fixed` for the pending groups and DSpark; the borrow as an alias; joined rows as
row keys; the stream's maps and scales fixed at load; the two chunk flags join DeepSeek's (D4). Deleted:
`dsv41_bill`'s Cooler terms, `dsv41_wide_bytes.zig`, `cooler_own_caps`, `dsv41_borrow`'s caps and its wait for
unwiring.

**Stage 7. Qwen3.8** (after 1.8.0, D17). First, the batched decode stops branching on whether an array is computed yet
(`_mlx_array_is_available`, where the deferred PLE flushes): under the scheduler its ids come from a lazy sample, so
the trace would follow the GPU's timing. Frames in the shared `trunk.zig` (no-ops on MLX, so GLM and MiMo do not move); `Rows` for KV
and the QSA banks; GDN state as a written-in-place pair; the MTP head's state; `PersistentSsmGroup`; the two chunk
flags join Qwen's (D4). Deleted:
`qwen4_bill`'s Cooler hooks, `arch/cooler.zig`'s caps, the pool's budget machinery in `core.zig`, every
`cooler_side.onEngine` branch of `bills.zig`, `transient_peak.zig`, `BILL_MARGIN_BYTES`.

**Stage 8. Before the load** (after 1.8.0, D17). The dry load: `static` and the whole model bill from the config and the pack index,
so the preflight is the bill itself and `LOAD_SAFETY_NET_BYTES` goes for Cooler. `sushi memory-bill <pack>` prints
it (D6). Release pins feed `Alias.tier_gb` and `recommend_gb` in each `arch.zig` and the tier table in
`docs/usage.md`.

**What stays untouched**: GLM, MiMo, a Qwen or DeepSeek load on its MLX opt-out, and every MLX branch of
`bills.zig`. After stage 7 that file holds MLX bills only.

**Joining or dropping an arch.** An arch on Cooler owes, in its own folder: its frames and keys, its `choose` and
`bound` sites, its context boundaries, its state module over the shared owners, and its lend order if it lends. It
owes shared code nothing: no hook in `bills.zig`, no term, no cap. Dropping one deletes its folder and its registry
line; the ledger, the plan module and the owners do not name it. `docs/engine-arch-seam.md` gains this paragraph at
stage 5.

**Docs each stage updates**: `engine-own-forward.md` (the allocator, the Metal rules that go),
`engine-memory-admission.md` (the ledger, admission), the arch's own doc (its bill table shrinks to its state rates), `envvar.md`, `flags.md`,
`testsuite.md`, `source-map.md`; this file leaves the tree when stage 8 lands, its content by then in those.

<a id="spikes"></a>
## 13. Spikes

Each is a day or two on a throwaway branch, a micro indicator, no quiet window unless it times something.

| | question | passes when | answer |
|---|---|---|---|
| S1 | can dry mode run a Hy4 decode forward on the tiny fixture, and is its trace the real one's? | the ask list (count, bytes, order, host flags) and the death order are equal | **passes** ([below](#spike-s1-s2)) |
| S2 | is a real trace deterministic? | two runs at one shape give one trace on the Hy4, DeepSeek and Qwen tinies, with kicks and hand-overs at different points | **passes**; a pass's own levels do not ([below](#spike-s1-s2)) |
| S3 | one buffer bound at offsets: what alignment does a compute bind need, what does a bind cost against a pool buffer's, does a 20 GB buffer and a mapped one of that size behave the same, what does residency cost | bytes equal, host time a token equal or lower; the alignment and the limits recorded here; the packer's alignment at least 16 B, since `attn_kernels.attn256Base16Aligned` picks its arm by a store's offset | not run |
| S4 | `Rows` as a wrapped reservation: does wrapping a longer prefix keep the bytes with no copy, what does it wire and how long does it take, does releasing before wrapping avoid a double count | a 10 GB store grows in place; the cost a turn is recorded | not run |
| S5 | what does offset reuse inside a pass cost? | Hy4 decode with a packed layout: barriers a forward and GPU busy against the pool's (part of stage 3's gate) | not run |
| S6 | Metal's overhead per command buffer and per buffer, how the compiled pipelines' bytes are read, and the `F` that keeps the GPU fed | a margin and a pipelines figure stated per GPU class; decode and prefill tok/s unchanged at that `F` | not run |

<a id="spike-s1-s2"></a>
**S1 and S2, as run** (branch `spike/s1-s2` at d7fe1594, M5 Ultra, `cooler-plan-spike`; the recorder fed at
`freshFor`, `dropStore` and each dispatch, the forward's survivors left as they are):
- **S1.** A dry decode of `HY4_BF16_TINY` and `HY4_STQ_TINY`, experts resident, at positions 7, 20 and 39, after a
  real prefill and a warm decode: ask for ask, death for death and level for level the real one's (504, 507 and 543
  asks on the BF16 tiny; 370, 367 and 403 on the STQ tiny). It encoded no dispatch, committed no pass, made no pool
  byte and compiled no pipeline. Its readbacks were answered from the real run's, in order: 5 on the BF16 tiny (the
  routed picks, 40 B), none on the STQ tiny; the two fills found the picks steer a size (above). At position 20 three
  128 B asks outlive the decode on both (none at 7 and 39): survivors stage 2 gives an owner.
- **S2.** Three greedy requests a fixture, the second with every dispatch in a pass of its own (the kernel profile):
  every forward's asks, deaths, readbacks' bytes and execution-wide levels are one on `HY4_BF16_TINY` and
  `HY4_STQ_TINY` (resident, and streamed as the contract loads it), `DSV41_STQ_TINY`, `QWEN4_STQ_TINY` and Qwen's
  batched joined rows, and the tokens are equal. The levels in a pass are not: up to 832 of 996 dispatches move under
  the per-dispatch cuts, and 11 of 302 between two plain runs of DeepSeek's decode (27 in another session).
- **What they add**: the recorder's levels are the execution's (the dry table); a readback is a host read of bytes a
  dispatch of the execution wrote, answered (the spike's tape, later `hostResult`, `choose`, `bound`) by the ask it
  reads; a dry host ask holds a host scratch, since Hy4's forward reads its ids back from the generator's host array
  through a same-bits `astype`, which is a share; a dry store has no address of its own, so a host map keyed by a
  store's pointer (`hy4_prefix.zig`'s held points) takes another key before it runs dry.
- **Not yet run**: dry mode past a Hy4 decode (prefill, verify, draft, sample; DeepSeek, Qwen; streamed fills), with
  pipeline lookups skipped, on a dry model, and the cold first recording; the tape that answered the readbacks is the
  spike's stand-in for `hostResult`, `choose` and `bound`.

<a id="decisions"></a>
## 14. The owner's decisions

Ruled by the owner on 2026-11-22:

- **D1. The plan is recorded from the forward**, not declared beside it. Everything else follows from it. S1 and S2
  are what could still overturn it. (Not taken: each served forward rewritten as a launch list, the op-composed one
  kept for the oracle.)
- **D2. One buffer at offsets**, not a placement heap. Same memory; the heap is verified to work on the M5 Ultra
  and stays available behind `Region` should S3 find a fault.
- **D3. A load trusts its own recording, cached like pipelines**: it cannot be stale. The embedded pins and the
  test that holds them equal to a recording still exist, and are what fixes a release pack's numbers at build time;
  no load skips its recording for a pin.
- **D4. `--prefill-chunk-max` and `--prefill-chunk-min` bound the load's choice; equal, they force the chunk.**
  A load that cannot hold `min` refuses with the numbers. The change is for loads on the static bill only: Hy4
  for now, each other Cooler arch at its cutover. There `--prefill-chunk` is deprecated; everywhere else it is
  unchanged and the new flags are refused ([part 9](#admission)). Nothing re-sizes a chunk after the load.
- **D5. A miss heals in process only within the arena held.** After the refused request the inference thread
  records that one shape and re-packs; if the new tables fit `G` and `H` as allocated, the next such request is
  served. If they need more, the shape goes to the plan cache and the next load sizes for it. The arena never grows
  while serving (C2).
- **D6. `sushi memory-bill <pack>`**, at stage 8: a pack's bill by owner from its config and index, with no GPU and
  no load, and the context that fits a budget.
- **D7. The margin becomes a measured constant per GPU class (S6), arch by arch at its cutover**: Hy4 at stage 5,
  DeepSeek at 6, Qwen at 7. An arch not yet cut over keeps `BILL_MARGIN_BYTES`.

Ruled by the owner on 2026-11-23, after S1 and S2 (the design follows the code where they disagreed):

- **D8. A dry host ask holds a host scratch of its bytes, and its filler runs.** Host bytes steer sizes (zeros make
  `hy4_expert_mv`'s group list 12 B where the picks make 24 B), and Hy4's forward reads its ids back through a
  same-bits share; a recording that skips the filler records the wrong plan. The scratch lives only while recording
  and is never arena or pool memory. (Not taken: "no copy, the filler not called", which would forbid host bytes
  from steering a shape.)
- **D9. The recorder's levels are the execution's, never a pass's.** A pass's levels restart at every cut and move
  with the GPU's timing (S2: up to 832 of 996 dispatches); the packer's lifetimes (`plan_shape`) read levels that
  rise over the execution. The engine's own hazard tracking per pass (`clearHazards`) is unchanged.
- **D10. A readback is a host read of bytes a dispatch of the execution wrote.** `core.hostBytes` is a host write
  (`ops.zeros`, `arange`): in dry mode it hands back a host ask's scratch and refuses any other store.
- **D11. Dry and real recording are two named modes.** `plan_api.Mode` names them apart in stage 1; neither works by
  a driver happening to be unset.

Ruled by the owner on 2026-11-23, after stage 1 (its audit and open questions):

- **D12. A phase plans as one shape only once appends are eager (2a).** The gate of "quiet across probes" moves from
  1d to stage 2: a lazy pending append hands a group-closing layer one result more, and the merge refuses it. Until
  2a lands each probe plans alone.
- **D13. The floors are part of the packer, not an optimisation of it.** Over the execution's levels (D9) the level
  rule alone reuses no byte under a dispatch that reads only load-time state (24x to 87x the lifetimes). Until 3a's
  floors land, a plan packs by lifetimes alone; the level rule packs only with floors.
- **D14. `placedZeros` becomes a GPU fill** (stage 2d): 403 MB of host-written zeros a MoE layer would make the host
  ring 16.7 GB a 4k chunk. Zeros either way, so no byte changes.
- **D15. One wide-mix grant a pipeline**, not one for every form and split (stage 2e): a GPU that grants one form's
  pipeline and not another's then plans and serves right.
- **D16. The old Hy4 bill is not tightened.** The audit shows it covers every phase (1.5x to 29x over); over-billing
  is safe and stage 5 deletes it. The end state is the static bill: each Cooler arch serves on it from its cutover.
- **D17. 1.8.0 ends at stage 5.** Hy4 serves on the static bill; DeepSeek-V4.1 and Qwen3.8 ship on today's code and
  bill calculations, and stages 6 to 8 (with `sushi memory-bill`, D6) come after the release. Nothing in stages 1 to 5
  changes their bytes or their bills.

Ruled by the owner on 2026-11-23, before stage 3 (the open questions of stage 2):

- **D18. Hy4's `verify` is 2 to 8 rows**, its kernels' block (`hy4_forward.verify_rows_max`). A forward of more rows is
  a prefill (stage 4); the probes take each arch's own verify widths, not `sampler_draw.max_rows`. (Not taken: widening
  Hy4's verify kernels to 16 first.)
- **D19. The logits a Hy4 forward hands the generator are Hy4's state**: a `Fixed` pair written alternately, the
  head's output going `into` it; `generate` keeps a share and its code does not change, so no other arch moves (D17).
  (Not taken: a pair the shared generator owns for every arch.)
- **D20. `Model.probeWide` runs at load, as the plan build's first step, and every Hy4 load plans**: the streamed BF16
  route's picks are classified (`choose`, `bound`), so no Hy4 load is left on the pool. (Not taken: streamed BF16 loads
  left unplanned on the pool.)
- **D21. A request whose reach is smaller than its state's reservation shrinks it by a copy**, as stage 2 built it:
  the reservation is always the reach's bill, and admission bills the old capacity beside the new for that moment.
  (Not taken: keeping the larger reservation and billing it.)
- **D22. The double free stage 2 met (an `errdefer` that frees a handle after the handle passed to its owner) is a
  lint rule**, `errdefer-handover` in `scripts/lint/`, landed in stage 3.
- **D23. A plan packs with the level rule, the floors under it** (stage 3b, decided by the coordinator under the
  owner's standing word to decide; it supersedes D13's "by lifetimes alone"). Packed by lifetimes alone, the release's
  decode took 1,048 barriers a forward against the pool's 911 and ran 1.2% slower (M5 Max, fwd meter at 4k, arms
  interleaved A B B A); with the rule, 910 barriers, 20.73 ms a forward against the pool's 21.04, host CPU 2.90 ms
  against 3.57, and an arena of 126 MB at the full context against 11 MB. The floors still order every reuse; the
  rule keeps reuse off work the GPU could run side by side. D13's audit (24x the lifetimes at a decode) priced the
  rule's bytes, which a load holds once, against no measure of its time. The rule keeps apart buffers of at most
  4 MB (`hy4_plan.level_rule_cap`): uncapped, the verify blocks' key-sized buffers took the release's arena at the
  full context to 3.99 GB; capped, 74 MB, against 70 MB by lifetimes. A large buffer's dispatches are long beside the
  barrier its reuse takes. Measured capped (M5 Max, the release resident, decode meter at 4k): 926 barriers against
  the pool's 911 and the uncapped rule's 910, 20.72 ms a forward against the pool's 21.14 and the uncapped rule's
  20.73, host CPU 3.03 ms against 3.72, decode 48.3 tok/s against 47.5; the decode arena 18 MB, built in 115 ms.
- **D24. The sample is a root of its own, keyed by its mix** (stage 4c, decided by the coordinator under the owner's
  standing word to decide). Measured on the STQ tiny, a one-row draw asks the pool once (greedy, 4 B), five times
  (temperature and top-p, 36 B), or sixteen times (top-k, min-p and a repeat penalty, 2.6 MB). Folded into the
  forward's execution, every decode root would multiply by the mixes; a ring of its own would hold bytes no plan
  names. So the sampler's call is an execution of a `sample` root keyed by what decides its structure (rows, the
  suppress mask, greedy or not, each of top-k, top-p, min-p, the penalties, logprobs, a grammar mask), each mix the
  sampler can take recorded dry at load, the arrays sized by the reply (the penalties' ids) bound at the load's
  reach. The pool then closes at the seal for the narrow phases (`PoolClosed` names an ask past it).
- **D26 to D33** (stage 4b, decided under the owner's standing word; the commit `918f4f0f2` carries each reason):
  an omitted min never passes the max (D26); a chunk key in another arch's settings refuses its load (D27); the chunk
  is fixed in the server's load path, so a harness that warms directly keeps the per-request chunk (D28); a fixed
  chunk skips the score narrowing and the tail merge (D29); any width from a prompt's final forward (31 rows: the snapshot backoff and the logits row, so no forward is wider than the chunk) to 8,192 (D30); the context priced first is
  `--ctx-size`, else the pinned one, else the auto context beside min's arena, and a wider chunk may take the auto
  context's headroom (D31, interim: stage 5's closed-form context and D7's margin replace it); the warm runs the chunk
  or fails (D32); an unload clears the chunk (D33).
- **D34 to D39, D44** (D24's sample, decided under the owner's standing word):
  - D34: a sample's key is what changes its structure on the engine: its entry (a lazy step, a serial draw, a
    verdict, a forced token's logprobs), one row or a block, the reserved-token mask, greedy or not, the think shift's
    kind and each row's gate (open, opening, closed, no opener), the repeat penalty, the presence and frequency
    penalties as one, a grammar mask, logprobs. The temperature's value, top-k, top-p, min-p, the seed, the logprobs'
    count and the reply's length are arguments (the engine's draw takes them; the penalties count into a vocabulary of
    ints), held so by probing two value sets of each mix into its root.
  - D35: a verdict's rows are a size: one row and a block of 2 to `verify_rows_max` are two roots, probed at every
    width.
  - D36: a step's token is written into one of two `[1]` u32 blocks in turn and its caller handed a view: the
    generator holds a token in flight across the next draw, as D19's logits. A token handed to a draw has this one form
    (a view of a block, `uint32`): a forced call's is written into the block in turn too (`hy4_sample_plan.forced`), on
    a load that plans its samples, where the host array of its id would miss every think kind that reads ids.
  - D37: a verdict wider than `verify_rows_max` draws on the pool beside its forward, a prefill block (D18).
  - D38: a draw's readbacks (a serial draw's token, a verdict's ids, the logprobs' row) are results (`hostResult`):
    they steer no shape.
  - D39: the samples are planned after `hy4_plan.build` into the same book, the arena made again only where they need
    more, so the forwards' build is untouched.
  - D44: `hy4_bill.arenaBound` adds the widest draw's every array, scratch and host ring, beside the forwards'
    (`sampleBytes`): the arena's parts are each root's largest, so a sum is the bound. The request bill held no term
    for the sampler's asks, so none leaves it.
- **D40. A round's mark is fixed blocks reserved with a drafting request's state** (stage 4c, decided by the
  coordinator under the owner's standing word to decide, as are D41 to D43). Each block is its state block's shape and
  is copied whole, so the mark's bytes do not follow the position: the open group's copy takes its block of `ratio - 1` rows and a row count, where the point's
  copy took its rows. The blocks are state (`markBytes` in the request's held bytes, `hy4_prefix.heldBytes` counts
  them), not a RAM tier point. A model with no drafting reservation (an oracle cell) makes them at its first mark, as
  the state makes any block it lacks. (Not taken: slots made at load, which a request at another KV width would remake
  inside it.)
- **D41. The replay is planned, not removed.** Keeping the verify's own state would need each verify row's handover
  (its hidden row, each sparse layer's picks) and the index keys of a group the block closed, and would keep a
  block's rows where today a serial run of the kept rows writes them: on a load whose verify rows are not its one-row
  steps' bytes (BF16, `widthExact` false) the state would change. The replay costs a forward of 1 to 7 rows a round
  whose draft is not all accepted; removing it is a speed lever for the stage 3 gate to size, not this landing.
- **D42. A round wider than `verify_rows_max`** (only a forced width, `SUSHI_HY4_DSPARK_ROWS`) is refused by name on a
  load that plans (`Hy4ReadoutPastBlock`): its logits have no block (D19) and a wide execution leaves none behind. Its
  replay of the kept rows reads out none and runs wide as D97 says; the request bill had no replay term of its own, so
  nothing leaves it. The arena's bound holds a planned replay within the verify of its rows (`forwardBytes` is
  monotone, the replay reads nothing out).
- **D43. A mark is held by the model, one at a time**: a second is `Hy4MarkHeld`, a reset lets it go, and a mark from
  before a reset rolls back to nothing (`Hy4PointMismatch`).
- **D50. Points are slots reserved with the request's state** (stage 4c, decided under the owner's standing word to
  decide, as are D51 to D54), as the mark's blocks are (D40), not made at load: a slot is its state blocks' shape, so a
  request at another KV width would remake load-time slots inside it. Their count is today's caps, the request's 4
  and the record's 8; a take with every slot held is `Hy4PointsFull`, never a copy on the pool. A reset lets every slot
  go; a point whose slot a reset or a later take holds stands no more (`holdsPoint`).
- **D51. A restore installs into the state's own blocks, reserved where the state is.** The state is emptied in place
  and reserved at the larger of the point's rows and its reach: a parked state keeps its last request's reservation,
  so a restore within it asks nothing; one past it (a fresh load, a deeper point) grows the stores as the first
  forward would, the state's reservation, not a staging copy. A point at another KV width resets the state, whose
  every block takes the new width.
- **D52. A live point that cannot be kept is the prefill's error**, named: `captureKept` no longer falls back to a
  stored point of the pool's copies (`capture`), which stays for offline tools and tests.
- **D53. The placeholder a stored point's absent arrays share is the model's, made at load** (`static`), shared by
  handle: a capture asked one byte of the pool for it.
- **D54. The SSD tier's read of a point stays on its own arrays this round.** Reading straight into `Rows` needs the
  tier to read a point's tensors and row segments into bytes its caller names, a seam in `kv_disk_cache` on both file
  backends (`SUSHI_PREFIX_FS`); until then a restore from SSD asks for what the read makes (63 arrays on the tiny at
  512 rows), and the install that follows asks nothing.
- **D56. The bill follows the allocation's predicate: a load that drafts bills the drafting blocks for every request.**
  `reserveFor` makes the mark, the handover, the selections and each point's hidden state whenever `keep_hidden` is set,
  which a load that ships DFly2 does once at load (`arch.dsparkOn`); a request that does not run a round holds them
  all the same. `hy4_bill.reservedBytes` bills them on that load whatever `warm.mtp_on` says; only the round's own
  terms (the reply's rows past its end, the verify block, the draft) follow the request.
- **D57. A point takes a free slot no stored point still reads** (stage 4c, decided under the owner's standing word to
  decide, as are D58 to D63). A stored point taken over a live one (`captureKept`) shares its slot's blocks, and they
  stay shared after the live point lets the slot go: a take of that slot copied its 12 blocks on the pool. `point`
  passes such a slot over (`Model.freeSlot`, whose blocks are each sole); with none left the take is `Hy4PointsFull`,
  never a copy. (Not taken: holding the slot until the stored point goes, which would need the slot to know its
  stored point's handles.)
- **D58. An encode the load did not plan is a miss by name**, as a forward's and a sample's are: a class with no root,
  or a miss inside the execution, refuses the image (`PlanMiss`), records its grid dry and learns it in place
  (`hy4_encode_plan.missed`), and keeps it beside the manifest as phase 4 (`shape_phase`, the grid's rows and columns),
  which the next load's encode probes record (`addKept`). It no longer encodes on the pool, unbilled and silent.
  (Superseded by D97: an encode runs in the region, measured at each class's largest grid; one past the region is
  `WideFull` by name, and nothing is learned or kept.)
- **D59. The arena holds the encode where the load keeps the tower** (`ModelConfig.keeps_tower`, set where a load decides:
  the server's load from its planner and `--no-vision`, the CLI, the contract's loads), not where the checkpoint has one.
  `planBuild` refuses a load whose kept tower and recorded fact differ by name (`Hy4TowerUnbilled`).
- **D60. A planned encode's request bill keeps its images' host pixels** (`mediaBill`'s `input_bytes`, the f32 patches
  the decode window holds, up to `decode_workers` images of ~28 MB at 9,216 patches): only the encode's work moved to
  the arena.
- **D61. A request's state is reserved before its lookup**, at the end its first forward reserves it at
  (`KVCache.requestEnd` of its prompt, `max_tokens` and speculative rows; `hy4_prefix.reserveRequest`, from
  `hy4_sched.restoreTarget`), so a restore into it, after a fresh load too, asks the pool for nothing. A state deeper
  than that end is left to the lookup, which rolls it back or starts it over; a reservation that fails is logged by name
  and the restore reserves as before.
- **D62. A planned load's forward input is one fixed block**: `Model.ids_fixed`, [1, 8,192] u32 (the widest chunk of
  the ladder), made with the plans and billed beside the logits blocks (`logitsBlocksBytes`). The generator's ids
  (`generate.idsArray`: a prefill chunk's, the last prompt row's, a serial draw's next token, a forced or spec-exit
  token's) are written into it in place by the host (`arch_cooler.forwardIds`, `Model.idsInput`) and handed to the
  forward as a view; a wider input is refused by name (`Hy4IdsPastBlock`). Every other load makes its own array as
  before. A DSpark round's blocks are Hy4's own host ids and ask nothing.
- **D63. The pool's open sites past Hy4's seal are named**: a forward no root holds (`hy4_plan.unplanned`: a prefill
  chunk, an image's splice, a round wider than a verify block, a request at another KV width) and the request's state
  reservation (`Model.reserveFor`, the state owner's own, billed with the request) each open it by `core.widePhase()`.
  The first goes when stage 4a plans the prefill; the second when stage 5's ledger allocates state apart from the pool.
- **D64. The pool closes per model's turn** (stage 4c, the coordinator's): a load that plans gets `.refuse` when its
  turn begins and the previous gate back at `Turn.end`; the gate lives in the model's turn slot, set at its seal
  (`arch_cooler.warmEnded`). The open sites stay named: D63's (`hy4_plan.unplanned`, `Model.reserveFor`, the state
  allocated at admission and billed on the ledger's state line) and a DSpark verdict wider than a verify block.
- **D65. A restore from SSD reads into the request's state, no array made** (stage 4c, closing D54): the tier hands each
  tensor of a point's file and of its row segments to its caller with the row it starts at, and Hy4 writes it into the
  block that holds it, reserved at the request's end before the lookup (D61). No staging copy and no segment join at any
  depth (on the release config at kv8 a join of 900K rows staged 8.6 GB beside the reserved state). A block another
  handle still reads (a stored point's view of the rows) is made anew by the reservation first; a point whose arrays do
  not fit the state's blocks is `Hy4PointMismatch`, and the restore falls back to the RAM or cold path as before.

- **D80 to D89** (stage 5's ledger, decided by its implementor under the coordinator's brief; each open to the owner):
  - D80: `B` is read once when the weights are resident, before the chunk: the chunk and the context are chosen in it,
    and the seal commits on it. Read at the load's very end it would come after the choices it decides.
  - D81: a request's rate a token is the state's as its stores hold it: `capRows`' grow slack (1/16) and the pool's size
    class (under 1/16) on the exact rate, the stores' 64-row floors and a page apiece in `fixed`. Both are structure,
    not a calibrated uplift; without them `requestBill` passes the terms at 837,325 tokens (the test is red). `Rows`
    reserving exact pages removes both.
  - D82: while the prefill runs on the pool its wide cap stays (`enterWide`, `liftWide`): nothing else bounds the free
    buffers a chunk leaves mid-request. A request's end leaves the pool no free buffer (`restCap` 0), which C1 holds.
  - D83: until the prefill is planned, the request bill keeps its prefill forward (`forward`, at the load's chunk and
    window) and the arena's term in the chunk choice is the planned roots' bound (`arenaBound(0)`): `arenaBound(chunk)`
    beside it would bill the prefill twice.
  - D84: `pipelines` is what Metal allocated for the load (`currentAllocatedSize`) past every buffer the engine and MLX
    count, read at the seal, after the warm compiled every pipeline: 1.27 MB on the BF16 tiny, 4.19 MB on the STQ tiny.
    It is a remainder (pipelines and what the driver keeps with them), so S6 still names the margin per GPU class.
    Hy4's `host` line is 0: no reading counts it. The pool's free buffers from before the load that the warm's trim
    released leave Metal's count, not the load's, and are credited back (D66).
- **D66. The pipelines line credits the free buffers the load released** (stage 5, C1's order dependence): an earlier
  load or test that leaves free pool buffers (16 MB after an expert kernel test on the BF16 tiny) had the warm's trim
  take them out of Metal's count, so the remainder read 0 and the line under-billed. `pipelinesSince` now counts the free
  bytes the load found and no longer holds. C1 leaves such buffers before its load, and holds the line above 0 only
  when the load made pipelines: one an earlier load or test made is not this load's.
  - D85: C1's engine count is what Metal holds: pool buffers by class, free ones too, the arena, and the loader's
    regions and slabs in whole pages (the engine's mapped counter is the arrays' bytes, 2.25 MB under on the BF16 tiny);
    every wrapped byte lies in a billed region.
  - D86: releasing the parked state writes the live record's points to SSD, resets the model's state and trims the pool;
    the next request restores from SSD or starts over.
  - D87: a request the lines would make wait is held pending by the scheduler (`memoryAdmitCount` reads the same bill);
    met at its prefill (a load on the ledger runs one request at a time), it is refused by name.
  - D88: the auto context is the closed form capped by the model's default context where it has one, else its window;
    D31's 93% headroom goes for loads on the ledger.
  - D89: a context a session pinned and a reload can no longer hold refuses the load, as `--ctx-size` does.
- **D90. A load on the ledger holds its budget whole** (the coordinator's, from the stage 5 review). Every memory
  reading outside its ledger counts `B` as taken, not what the load has allocated: `bills.promisedActive` adds each
  load's `B` less what the active count already holds of it (`Ledger.seen`, its model bill as counted at the seal, and
  its parked state: `ledger.unseenBeside`), for another model's preflight (its free RAM too), an off-ledger model's
  admission and headrooms, and the max-context search; the load's own readings read its lines. A new model loads only
  where it fits beside `B` (else its preflight refuses it by name); unloading another model frees no ledger room, so a
  request on the ledger unloads none. `B` itself (`static_bill.budgetFor`) counts MLX's cache and the engine's free
  buffers held before the load as the other models'.
- **D91. While a request runs, a reading outside the ledger counts its allocations beside `B`**: the engine's counts
  cannot tell them from another load's, so the reading errs high by at most the request's bill; idle or parked, it
  counts `B` exactly (`hy4_ledger_test.zig`).
- **D92. An image request's media is on the ledger with its request**: the rows its encode writes (`Media`, from the
  grids) on `state`, its decoded f32 pixels on `media` (`static_bill.reservation`); the encode reads no live
  headroom. The connection thread's check and the scheduler's hold price the same media, so a request that does not fit
  is refused with its figures before its encode, or held to wait.

- **D97. The wide phases run in a measured region, not on exact plans** (the owner's ruling, 2026-11-24; it supersedes
  the exact prefill and encode roots of stage 4a and D45 to D49, and D70's single arena for them). A prefill chunk, a
  round wider than a verify block or a request at another KV width than the load planned runs as a wide execution
  (`core.beginWide`): each ask takes the lowest gap of the arena's scratch that holds it (address-ordered first fit, a
  dead place joining its neighbours), its host asks one after another in the host ring's replica, and bytes reused
  inside the open pass take the arena's barrier floors. The region is what `plan_region.need` gives over dry probes of
  the load's chunk, a row short of it (its short last attention span launched apart and joined, `linearSpans`, asks more
  than the chunk on an STQ load) and the final span, each at the state's start and deepest, read out by none and by the
  last row, the arena each part the larger of the plans' and the region's; an ask past it is refused by name
  (`WideFull`), never served on the pool. The host ring is not monotone in depth (a span the selection threshold cuts
  asks more for each row of a dense part that is a verify block), so each width is also probed at the one cut that
  leaves a verify block's most dense rows (`hy4_plan.straddleDepth`): the audit sweeps every depth a chunk of 512
  rows touches the threshold at, and none asks more. A probe is kept only where a sweep shows it can set the maximum.
  Nothing matches a recorded structure, so no prefill shape can miss (D110 pads a deep-zone chunk to its class's
  recording). The narrow phases keep their exact plans. A stack
  (bytes back only once every ask above is dead) was measured first and dropped:
  a 2,048-row chunk of the BF16 tiny topped at 511 MB against 65 MB packed by lifetimes, a layer's result pinning all
  below it; first fit took 67 MB, best fit 83 MB. The forward's own waste went first: a chunk's spans write the layer's
  rows in place (D76), and the MoE pick blocks scatter into the sum's bytes (`ops.putAlongInPlace`) instead of a copy
  each, the copies that made a chunk's memory grow with its square. On the BF16 tiny the region then grows linearly,
  61.2 MB at 64 rows to 73.0 MB at 4,096 against 69.1 MB packed by lifetimes (commit 9a93cc7e3 and its successor).
- **D98. A load on the ledger records its plans before its warm and chooses its chunk on the arena they measure.**
  After its weights and expert cache are resident and its budget read, the load records every narrow root and the
  samples (`hy4_plan.record`, no arena yet), then prices each chunk candidate widest first as its measured arena
  (`arch.planArena`: the plans' layout and that chunk's region, each part the larger) beside the request terms at the
  context; the warm runs, and its end makes the arena for the fixed chunk (`hy4_plan.makeArena`). The closed-form arena
  bound (`hy4_bill.arenaBound`) and the ledger's pool-forward term are gone: a request on the ledger holds its state and
  its copies alone (`RequestTerms` has no `forward`), and the preflight before the weights prices no arena, which the
  chunk's choice refuses by name when even the min does not fit; the seal refuses a fixed context its lines no longer hold
  (the warm's static and pipelines are unknown to the choice), naming the widest narrower rung that holds it. A load that plans with no chunk fixed (a test's)
  measures its region at the widest chunk its prefill runs (`prefill_plan.widestPrefillWidth`).
- **D99. A size that follows a forward's data is declared, or the load refuses it.** As a `?Sized` value travels
  only behind a length, a data-sized ask is legal only through `plan.bound(real, max)` (a dry recording takes `max`, a
  served one `real`) and a data-steered branch only through `plan.choose`. The rule is enforced by the recording, not
  by review: every probe is recorded twice (`hy4_plan.recordChecked`), over zero ids under the zeros fill and over ids
  spread over the vocabulary under the ramp, and the two must be one execution, ask for ask. An undeclared size parts
  them and the probe is refused by name at load (`Hy4PlanDataSteers`, naming the frame and both sizes), where before a
  narrow plan met it as a served miss and the region as a dry probe that under-measured (`WideFull`). Its first find:
  the STQ embedding gather uploaded each group's pairs and launched its rows at how many ids fell in its 128-id blocks,
  and skipped the permutation when they fell in order; now every group's pairs go up in one array, each group's
  launch is bounded at the chunk's rows and lands in one block of them, and the permutation is always taken. Whether
  the groups land and the width each launches at followed how many blocks the ids touch: a prompt in a few blocks
  took the one-group arm, whose first fit is not the recording's (the release 4bpw pack's warm asked 4,182,272 KB of
  the 3,833,388 KB measured and failed `WideFull`); both now follow the most blocks the rows can name. A
  process-wide cache a wide forward fills is the same class in time: its GEMM inputs are the forward's own
  (`exl3.linear.GemmScope`), made inside its execution or probe and freed at its end.
- **D100. A wide execution's host ring comes back down at each eval.** Summing a wide forward's host asks bills
  every upload of every layer at once, and a dry probe's bounds multiply: a streamed layer's expert groups each upload
  their picks' indices, and the probe gives each of its most groups every pick, so a tiny's 8,192-row chunk measured
  8.5 GB of host ring (68 GB on a cache of one group). An eval (`route`'s `mlx_array_eval` and `mlx_eval`,
  `core.evalPoint`) is a point in the forward where, inside a wide execution, every pass submitted has run: the ring's
  top comes down to the end of the highest host ask still alive, and the next ask lands there. A recording marks each
  eval with how many asks had died by then (`plan_trace.Wait`, `Ask.died`), so `plan_region.need` gives back exactly
  the bytes the execution will; a bump that only comes down when sizes shrink stays at or below the probe's. The two
  recordings of a probe must wait at the same places (`plan_trace.sameWaits`). The real waits at these points already
  stood (each group and each layer evaluates); a single-array eval inside a wide execution now waits for every pass,
  not only the one that wrote its array.
- **D101. A kernel proven at its first launch is proven at load, never in a recording.** A dry recording runs no
  launch, so a proof inside one proves nothing and keeps its kernel for the process. Hy4 proves its experts' and its
  trunk's NAX GEMMs where they run at load (`expert_exl3_kernels.prepareGemmNax`, as DeepSeek's load does), before a
  ledger load records its plans (D98); a recording that reaches a kernel no load proved is refused by name
  (`Exl3KernelProofDry`). Found by D99: the first STQ probe made the proof's six asks and the second, finding the
  kernel kept, did not.
- **D102. A streamed load with `--expert-cache-bias` plans** (the owner's, 2026-11-24, over refusing the flag on Hy4
  or keeping its old bill). Its route reads the cache's penalty as a host array of the forward's own, scored from the
  layer's host map (`expert_stream.Engine.cachePenaltyHost`): the same bytes whatever the cache holds, so a planned
  forward asks it of its host ring. The stream's GPU maps (`specRoute`, `specGrouped`, `cachePenalty`, DeepSeek's and
  Qwen's) are unchanged. Every Hy4 load plans, so every one that warms is on the ledger.
- **D103. A load on the ledger serves its own KV width**, and a request naming another is refused by name
  (`static_bill.kvRefusal`, beside the arch's own KV refusal on every request surface). The load sized its closed-form
  context, measured its region and planned its forwards at its width: a request at BF16 on a kv8 load would run in a
  region measured at kv8, draft on a closed pool, and hold twice the state the context priced. A request reserves its
  terms at its reach (`bills.ledgerRequestBill`), never an arch's request bill.
- **D104. The layer-major capture measures its window on the load, dry**: `sushi kld capture --layer-major` loads at its
  floor budget, then records each piece of its longest window the capture runs (a prompt chunk's rows after the pieces
  before it, at BF16 KV: `hy4_plan.recordChecked`, phase `.wide`) and takes the most its scratch and host asks hold at
  once (`plan.region.need`) as the window's forward
  (`hy4_layer_major.windowBytes`); the batch is what the budget holds beside it. Not taken: keeping
  `hy4_bill.forwardBytes` for the capture alone, the last reader of the forward estimators.
- **D105. Before the load, an arch on the ledger prices a request by its closed-form terms alone**: the preflight's warm
  need (`bills.warmNeedBytes`), the streamed fit's planned state and serving request
  (`expertStreamingPlannedKvBytes`, `expertStreamingServingBytes`) and the RAM-tier default's request beside the ssd
  budget (`defaultRequestBill`) all return from one function (`bills.ledgerRequestBeforeLoad`: `ledgerTerms` at the
  widest chunk the load may fix, at a reach), the warm's at its longest request at that chunk; none prices a forward.
  Its forwards run in the arena the ledger sizes with the chunk, which narrows the chunk, or refuses the load by name,
  where the arena does not fit. Not taken: a pre-load estimate of the arena, which is the deleted estimators again
  (the dry load of stage 8 gives the measured one).
- **D106. A static-bill load's chunk is one of 512, 1024, 2048, 4096 and 8192 rows** (the owner, 2026-11-24): every
  bound (`--prefill-chunk-max`, `--prefill-chunk-min`, their settings keys, `--prefill-chunk` where it means the max) is
  one of them or is refused by name, at the flag's parse, at the load (`bills.staticChunkBounds`) and at a settings save
  (`model_settings.refuseEdits`); the load tries the rungs from the max down to the min
  (`settings_resolve.static_chunk_rungs`). A chunk off the span grid is never fixed, so the region measures the
  rungs alone (with the row short of each, D97).
- **D107. A load on the ledger makes its arena when its chunk is fixed, before its warm**, so the warm runs what a
  request runs: its prefill in the region, its narrow phases on the plans, its state alone on the pool. With the arena
  made at the warm's end, the warm's prefill ran on the pool, which no request does: the real 4bpw pack at 8192-row
  chunks refused itself at its warm (3,456 MB asked, 10,841 MB held, against a 1,406 MB bill), and a cap and a region
  term for that one phase followed it. A shape the warm misses is learned and kept as a served request's is.
- **D108. A request's copy of its state is a capped term of the request terms** (`RequestTerms.copy_cap`): the restore
  point a writer stages is a copy of the request's state held beside it, at most the writer's permit, so
  `at(reach) = s + points + min(s, copy_cap)` with `s = per_token * reach + fixed`. Priced as the permit whatever the
  length (`points`, at the window), every Hy4 request SSD-first reserved about 1.2 GB: a 4k request now reserves
  251 MB (was 1,265 MB) and a 32k one 872 MB (was 1,576 MB) on the release config; at 128k both are 2,641 MB. `contextFor`
  solves it in closed form: with `free = room - points`, `s` reaches `free - copy_cap` once `free` holds two caps, else
  `free / 2`; with no cap it is the old formula exactly. Hy4's cap is `hy4_ledger.checkpointCap` (the permit where a
  checkpoint is written, else 0); `points` is the `Points` slots alone.
- **D109. Hy4's margin stays `BILL_MARGIN_BYTES` (256 MB) at its cutover; D7's measured constant waits for S6's
  in-flight peak** (2026-11-24). The real-pack matrix on the M5 Max (applegpu_g17s; both Hy4 packs, 11 settings of
  chunk, KV width, context, SSD budget and MTP, 101 request ends) finds Metal holding 1 to 3 MB more than the ledger's
  lines without the margin at every request end: the lines are exact at rest. A request end is not the peak the margin
  is for (Metal's per-command-buffer and per-buffer overhead while a forward is in flight), and no other GPU class is
  measured, so no class takes a measured margin yet. Kept, the margin costs a load about 24k tokens of context at KV 8
  (10,838 B a token), 2.3% of the release config's 1M. S6 measures the in-flight peak per GPU class with DeepSeek's
  cutover (stage 6); Hy4 takes the class's constant then.
- **Stage 5 as built: Hy4 owes shared code no term.** `hy4_bill.zig` is deleted; Hy4's `bill` slot is
  `hy4_ledger.zig`, its state rates and `ledgerTerms`. Every shared path a Hy4 load reaches takes a ledger arm by name:
  its forwards install no pool cap (`arch/cooler.zig` `plansForwards`) and its pool keeps no free buffer past a
  request; its arena is made before its warm (D107), and the warm holds its longest request to the terms at its reach
  (`bills.warmRequestBill`, `cooler_warm`). A
  request's refusal names the reply and context the terms fit (`requestContextRefusal`); the RAM tier's ask stands, its
  bytes what the model parks (`prefixCacheMemForLoad`). A static-bill load that does not warm is refused by name
  (`StaticBillLoadDoesNotWarm`), so the off-ledger chunk chooser is gone.
- **D110. A load-width chunk in the deep zone runs padded to its class's recording** (2026-11-25; amends D97). First fit
  is not monotone: a smaller ask leaves a gap a later one no longer fits, so it lands on top (pinned by the core cell
  "padded to its recording"). D97 measured each width at the state's start and its deepest, taking the deepest for the
  most, and the owner's live test on the release 2.25bpw pack failed `WideFull` with an 8,192-row chunk at 40,960
  asking 4,296,592 KB against 4,293,236 KB measured. Past the selection threshold and the window (`deepZone`, 2,048
  tokens on the release config) a chunk's structure turns on a few facts (`Model.wideClass`: each kind of layer's
  edge key, the spans' width, a ring read in one run), and within a class every ask only grows with depth. The load
  records each class at its deepest member, read out by its last row as a served chunk is (`measureClasses`: 12
  classes a width on the tiny), the
  region takes their most, and a chunk of the class runs with each scratch ask placed at its recorded size
  (`core.beginWidePadded`, `plan.region.padOf`): first fit then places it as the recording did, so it tops out exactly
  there. At each `plan.choose` the execution follows a pass that took the same value, sound where every pass leaves
  first fit in one state at each choice (`plan.region.choicesAgree`, checked at load). A chunk of no recorded class, an
  ask past its recorded size or past the sequence is refused by name (`Hy4WidePlanMiss`, `PlanMiss`), never placed.
  Chunks below the deep zone, other widths, an image's splice and a readout other than the last row (a capture's
  hidden row) keep D97's measure.

## 15. Risks

- **Dry mode drifts from real mode.** One flag at the engine's choke points is the whole surface; the gpu-tier trace
  equality test runs on every fixture and phase, and a load that warms replays its plan on the GPU. S1 is the first
  thing built because this is the assumption everything rests on.
- **A key nobody thought of.** Named at plan build when two probes disagree; otherwise the coverage test, the warm,
  the bench rungs, then a named miss. It is never memory corruption: the binder checks size, structure and the
  liveness of what it overwrites on every ask.
- **Reuse costs GPU overlap.** The level rule keeps most reuse off work that could have overlapped, but a pass cut
  between two users of some bytes still orders them. The stage 3 gate measures barriers and GPU busy time against
  the pool before anything else builds on the arena; if it loses, the packer takes a stricter rule (reuse only
  across a recorded dependency) at the price of a larger arena, and nothing else changes.
- **A forward's code holds state by a lazy write or a cache.** Both are named by the first recording (an escape),
  and stage 2 removes them before the binder exists.
- **Recording time at a first load.** Host work only, cached per pack, build and GPU class; measured in stage 1e and
  stated in the `[plan]` line. A repeat load reads tables. By D3 every first load pays it, a release pack's too, and
  once per chunk candidate (D4); if 1e finds it long, that number goes back to the owner before stage 3.
- **A miss whose shape does not fit the arena held** keeps failing until a restart (D5). The refusal says so, with
  the bytes it lacks; the target stays zero misses on a release pack.
- **The series is long.** Stage 1 lands value alone (the audit), every stage leaves the tree serving, and Hy4 is
  whole at stage 5 before another arch starts.
