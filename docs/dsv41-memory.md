# DeepSeek-V4.1 serving memory

How `deepseek_v41` spends memory under CED prefill: the ledger, the per-layer cache split, the 1M-token state, the
prefill that borrows decoder pools and the fully resident load. Index: [CLAUDE.md](../CLAUDE.md#docs-index); related:
[dsv41-arch](dsv41-arch.md), [engine-expert-streaming](engine-expert-streaming.md).

## The ledger (an STQ pack, `--ssd-budget-gb 96` = 103.08 GB, no vision tower; the [tower](#vision-requant) adds 0.50 GB)

| term | GB |
|---|---|
| trunk (resident STQ banks, head, the 384-expert routers as stored in bf16, the 0.37 GB embedding) | 4.62 |
| DSpark stages (resident) | 3.92 |
| union workspace (one layer's 384 experts at the widest layer) | 3.70 |
| bounce + selected | 0.60 |
| MLX buffer pool, decode cap (`dsv41_pool_bytes`, 1 GiB) | 1.07 |
| expert cache (what is left; whole slots a layer) | 89.2 |

Routed experts by layer, rate-sized (shard headers): the encoder (layers 0-19) holds 57.13 GB (6.8-9.6 MB an
expert), the decoder (20-39) holds 62.46 GB (7.1-9.2 MB), 119.6 GB in all. The cache holds 75% of it.

- Without a profile the ledger allots each layer the same bytes: 234 slots of 9.62 MB, the largest layer average, which
  makes 2.25 GB a layer. Inside a layer the rate-group pools split that allotment by bytes, so a small-expert layer
  caches more experts (layer 0: 331 of 384) and layer 15 caches 234.
- A pack's `expert_cache_profile.json` ([pack-format](pack-format.md#expert-cache-profile)) moves slots between layers by
  decode hits per byte at the same total bytes: the layers that miss most (0, 1, 19, 23) take 340-374 slots, the
  steadiest (16, 25) about 215. On a 4k and a 16k prompt with a 1024-token decode at budget 96 the decode speed matched
  the uniform split (68.9 and 63.8 tok/s against 68.1 and 65.0), hits being ~99% in both; the profile matters for long
  decodes over changing documents, and the larger the cache the more of the remaining misses it removes.
- **Making the encoder fully resident does not speed up prefill.** CED runs only layers 0-19 on three prefill chunks
  of four, and the profile is tuned for decode, so those layers miss most in a prefill. With layers 0-19 at their full
  bytes and the profile spending the rest on 20-39 (a throwaway knob on `42c364c6`, `--ssd-budget-gb 108`, same cache
  bytes, replies byte-equal), a 25.8k-token prefill reads 16.2 GB from the SSD in place of 42.9, and its tok/s does
  not move: cooled pair, one binary, `tests/bench.sh --rungs 32k`, code windows 4k to 32k 1,350 / 1,552 / 1,649 /
  1,554 / 1,539 against 1,351 / 1,615 / 1,651 / 1,571 / 1,556 with the profile split, prose within 4% both ways,
  decode at 32k equal. On the 25.8k request the GPU's idle time fell from 2,515 to 1,522 ms and its busy time rose
  from 15,392 to 16,426 ms (`[engine]` line; arms back to back), and in the cooled pair the resident arm's GPU clock
  reads lower from 16k on (1,463 and 1,513 MHz against 1,495 and 1,540): a sustained prefill is bound by the heat of
  its GPU work, so idle time won back is paid in clock ([perf-gpu-power](perf-gpu-power.md)). What speeds a long
  prefill is less GPU work per token, not fewer waits.
- **The MLX buffer pool is a ledger term, in two phases.** `server.applyMlxCacheLimit` caps freed buffers at RAM/16
  (8 GiB here), and that pool sat outside the plan: a bench went 118 to 126 GB against a 125 GB wired limit. Measured
  (`--ssd-budget-gb 96`, 4k to 128k code and prose, 150 ms samples of `/props`): every 8k prefill window and the cold
  4k prefill fill the pool to 7.2-8.0 GB at every depth, so the peak is the cap, not a demand below it; decode and a
  DSpark round keep it under 1.1 GB, and 0.3 MB between turns. Active bytes run 94.0 GB at rest, 97.3-98.0 GB in a
  prefill window. So the cap follows the forward's rows (`dsv41_borrow.enterForward`, on the inference thread, from
  `beforeForward`; `poolCap` is the pure rule):
  - **Decode** (a forward of up to 17 rows, DSpark's widest verify is 16): `dsv41_pool_bytes`, 1 GiB, billed in the
    ledger as the `mlx pool` term (`capMlxPool` at load, `[mem] MLX buffer-pool cap ... from the ssd-budget ledger`), so
    the cache gives up ~3 slots a layer instead of ~24.
  - **Kept scales** (an STQ pack on Sushi Cooler): host memory, not Metal's, 0.68 GB on the 1.75bpw pack, the ledger's
    `kept scales` term ([engine-expert-streaming](engine-expert-streaming.md#budget)): two slots a layer at 108 GiB.
  - **Prefill** (a wider forward): the RAM-wide cap, with no cap of its own. The transient cap
    ([below](#transient-caps)) bounds the pool together with the forward's allocations, so MLX trims the pool on demand
    at it and the parked pool never passes what was billed. The decode cap is a standing term of the live headroom
    (`server.dsv41Headroom` takes it off ceiling less active, since the parked pool sits in MLX's cache, not its active
    bytes; on a Cooler load it is the engine's free buffers under the cap, taken off once, at the larger of the two);
    a lower cap clears the pool. On MLX admission (`prefillNeededAtChunk`) bills the decode cap beside every request;
    on Cooler the bill holds no pool term, so admission adds none back (`billedParkedBytes`), and the borrow counts the
    engine's free buffers under a forward's cap as the forward's (`borrowHeadroomFrom`), since the forward reuses them;
    at load `expertStreamingServingBytes` leaves it out, the ledger's staging term already holds it.
  - Unloading a DeepSeek model gives the next model the process-wide cap `applyMlxCacheLimit` installed, MLX's and the
    engine's free cap both (`gpu_mem.restoreMlxPool`), forgets the raised cap and lifts the transient cap
    (`dsv41_borrow.unloadModel`, in the scheduler's unload and eviction paths). A load that fails after its plan capped
    the pool runs the same hook from its own errdefer: a failed load runs no unload (`failLoadRequest`).
  - A request that ends, by finish, cancel or error, drops the pool cap to the decode cap and lifts the transient cap
    (`endRequest`, in the scheduler's slot cleanup), so a prefill with no decode forward after it does not leave the
    raised pool parked, and the next request's restore allocates its state uncapped.
  - Measured (STQ-1.75bpw, M5 Max, one cold 14,766-token prefill, three live-state repeats with a 31-row tail, one turn
    adding 1,000 rows; DSpark default, fans max, die < 50 °C):

    | build, `--ssd-budget-gb` | slots/layer | expert hits | 31-row repeat | 1,000-row turn | peak wired | decode wired |
    |---|---|---|---|---|---|---|
    | before, 108 | 263 | 98.62% | 141 tok/s | — | 126.3 GB | 118.5 GB |
    | this, 108 | 260 | 98.50% | 138 tok/s | — | 124.8 GB | 117.0 GB |
    | this, 114 | 277 | 99.24% | 146-152 tok/s, no release | 541 tok/s, no release | 127.7 GB | 123.6 GB |

    At 114 a cold prefill releases layers 37..39 (8.15 GB) once and reads them back before decode (~1.4 s a request).
    The 1,000-row turn parks ~4 GB of pool on top of decode, which is the peak. A 2 GiB prefill pool cost 12% of cold
    prefill (666 against 760-781 tok/s); those rows are from before the transient cap, which replaced the pool's own
    prefill cap.
- The per-request state (890 B a token, 0.93 GB at 1M) and the prefill transient are not in the SSD budget. They come
  from the serving budget (the GPU ceiling less everything resident), billed per request. `/props`
  `memory.kv_cache_bytes` and a live `/metrics` row's `state_bytes` count it (`dsv41_model.stateBytes` plus the live
  points) while the request runs; between requests the model parks it and `kv_cache_bytes` still counts it
  (`sched.parkedStateBytes`).

<a id="transient-caps"></a>
## Transient caps

MLX allocates through a patched allocator that can refuse a buffer ([engine-mlx-gotchas](engine-mlx-gotchas.md#transient-cap)).
Every DeepSeek phase runs under a cap on the bytes MLX allocates itself beyond what it holds when the phase starts: the
cap is `limit = active - imported + (the phase's bytes + the decode pool)`, set on the inference thread by
`dsv41_borrow.enterForward` before each forward (prefill chunk, decode, DSpark verify; the draft runs under the cap of
the forward before it) and by `enterVision` around an image encode. The pool counts in it, so the allocator trims the
pool on demand and the phase's bytes plus the pool never pass the limit. A request that reaches it fails as
`TransientCapExceeded` (503) and the server keeps serving. A request's end lifts the cap. There is no flag.

The phase's bytes are the bill: `Config.forwardCap(rows, seq)` for a forward, `request_media.visionEncodeBill` for an image.

    prefill-width forward (> 17 rows):  shape sum(rows, seq) + seq x (state/16 + worst layer) + 256 MiB
    decode or verify forward (<= 17):   seq x 16 x 4 x 3 + seq x (state/16 + worst layer) + 123 MiB + 256 MiB
    joined decode forward (a row each): the sum of each request's decode term at its own seq, + 379 MiB once
    prefill-width forward on MLX:       rows x dim x 480 + seq x (state/16 + worst layer) + 123 MiB + 256 MiB

The per-context term is a decode block's scores (16 rows, 3 copies, f32) and the row stores' growth (a sixteenth of the
state, plus one layer's old rows beside the new buffer: 55 + 356 B a token); the 123 MiB is the 0.12 GiB of small
tensors a decode forward holds (`forward_held_bytes`) and the 256 MiB the bill's one margin (`BILL_MARGIN_BYTES`); no
term is multiplied. A last chunk absorbs a tail, so a chunk bills `chunk + 511` rows (`chunkCap`).

<a id="wide-shape-sum"></a>
**The shape sum** (`dsv41_wide_bytes.bytes`, Sushi Cooler) is what the engine's pool holds in use at the fullest point
of a wide forward, each tensor at its pool size class, from the shapes the forward makes. A layer's intermediates live
in its scope to its end, its attention's and its routed chain's in child scopes, so the fullest point is the end of one
layer's attention or of its routed chain, at the layer holding the most: an encoder layer that is an Engram layer, a KV
source and an index owner (the tiny's 2, the release's 14). L is the chunk's rows, hc 4, n = L x top_k:

| owner | live tensors (bf16 = act unless named) |
|---|---|
| pass | `pipe_depth` (2) streams `[L,hc,D]`; the collapse weights `[L,hc]` f32; each Engram layer's rows read so far `[L x cols, ehd]` f32; each ratio-2 source's open group, copied out, `[1,hd]` f32 twice; positions `[L]`, `[L/r]` and two `[span]` a span to eight arrays; each span's index top-k `[span, topk]` i32; the state rows appended so far (FP4 codes and scales, at the worst compression phase); STQ trunk: `linear.gemmInputs` row tables for `[L]` and `[o_groups x L]` |
| Engram write | the rows `[L, cols x ehd]`, `wkv` `[L,(hc+1)D]`, eight f32 `[L,hc,D]`, value `[L,D]` f32, sixteen `[L,hc]` f32, the sign mask, the new stream `[L,hc,D]` |
| hc, twice | mixes `[L,(2+hc)hc]` f32, pre and post `[L,hc]` f32, comb `[L,hc,hc]` f32, the normed input `[L,D]`; the FFN stage's stream expanded `[L,hc,D]`, and where the fused expand does not serve (`dsv41_wide_bytes.expand_fused`, which a load records, else the GPU class: a non-NAX GPU, whose f32 matmul `hcExpandFused` does not reproduce, or the fused kernels off) `hcExpandOps`'s chain beside it: four f32 `[L,hc,D]` (the stream's copy, the product, the matmul, their sum) and the output's f32 `[L,D]` |
| attention (child) | `wq_a`, `q_norm` `[L,q_lora]`, `wkv` and the window rows `[L,hd]`, `wq_b` and the roped queries `[L,H,hd]`; the compressor (f32 input `[L,D]`, its projections and their join to the pending rows `[L+r-1,hd]` f32, the new pending rows copied out, softmax and pool, latents, index keys, FP4 codes); every span's heads; the last span's band, indexer (queries f32, scores `[span, min(keys, 32768)]` f32, the keys it unpacks) and the head kernel's or `attnCore`'s scores `[span,H,window+span+topk]` f32 |
| after attention | its output `[L,D]`; the window rows `[L,hd]`, which the ring write's source keeps in use until the layer's pass runs |
| last expand (after the routed chain's scope) | the routed sum `[L,D]`, the new stream `[L,hc,D]`, and on the chain arm `hcExpandOps`'s four f32 `[L,hc,D]` and f32 `[L,D]`; a wide forward never folds it into the routed chain (`foldsExpand` is decode-wide only), so a layer peaks at its attention, its routed chain or this |
| router | the input `[L,D]` f32, the weight `[E,D]` f32, a `[2048,E]` f32 tile at a time and their concat, the picks and weights `[L,top_k]` |
| routed chain (child), STQ fused (`moePrefillSharedFusedFrom`, up to 6 pools) | slots, sort and order `[n]`, the shared and routed prep `[L,D]`, `[n,D]` twice each, shared gate and up `[L,I]`, the window table, a gate and an up plane `[n,I]` per pool, the SwiGLU `[n,I]`, shared `[L,I]` and `[L,D]`, a down plane `[n,D]` per pool, the f32 sum `[L,D]` |
| routed chain, STQ combined (`moePrefillFrom`, more than 6 pools) | the shared expert as ops (gate, up, eight f32 `[L,I]`, down), sort, prep `[n,D]` twice, gate and up combined and the SwiGLU `[n,I]`, a down plane per pool and their combine `[n,D]` |
| routed chain, FP4 (`moeWide`) | the shared expert as ops, six `[n]` index arrays, the gathered rows `[n,D]`, gate, up, SwiGLU `[n,I]`, down `[n,D]`, the sum `[L,D]` |

The data-dependent maxima are the bill's: a streamed route reads each rate group from its cache pool and its spill
pool, so it bills `2 x rate_groups` pools (`ModelConfig.expert_rate_groups`, the pack's widest layer; above 6 pools the
combined chain), and a ratio-r compressor bills `r - 1` pending rows. A resident load's route reads one pool a group:
its bill carries the spill pools it never makes. Decode forwards keep their scores term and `forward_held_bytes`.

Measured (`dsv41 engine: a wide forward's in-use peak is its cap's shape sum less the spill pools its route did not
take`, DSV41_STQ_TINY and DSV41_TINY at 512 to 8192 rows, one row in, all experts cached; the pair-major tiny is left out: tile order is fixed at the slab fill, never in the forward pool): the shape sum at the pools taken sits over the
in-use peak by 10 pages on the STQ tinies (9 at 512 rows) and 1 page on DSV41_TINY (3 at 512): page-sized tables and
constants a route reuses. On a non-NAX GPU (M1 Max) the expand's chain, at the attention's expand and at the layer's last
expand, keeps every tiny inside its bill at every width: the STQ tiny was 46.2 MB in use against 40.5 MB billed at 512
rows, DSV41_TINY about 600 B a row over (its routed chain is smaller than the last expand's chain); DSV41_TINY now sits
108 KB under at 512 rows and 536 KB under at 8,192. The STQ tiny's sum at 8192 rows is 637.5 MB at six pools, 75.5 MB of them the three spill
pools its cached route never makes.

| release (STQ-1.75bpw, 10 pools) rows | shape sum, GB | new cap at seq = rows, GB | old cap, GB |
|---|---|---|---|
| 363 | 0.751 | 1.020 | 1.290 |
| 2,048 | 4.081 | 4.350 (4.443 at 225k) | 5.431 (5.523 at 225k) |
| 4,096 | 8.161 | 8.431 | 10.465 |
| 8,192 | 16.314 | 16.586 | 20.533 |

The release checkpoint itself (FP4 experts, FP8 trunk) on Sushi Cooler: 2.718 GB at 2,048 rows, cap 2.987 GB.

**MLX keeps the measured row term** (`Config.mlxForwardCap`, the 480 B a channel below): MLX's graph holds what its
evaluation order keeps, and the recorded MLX peaks sit above the engine's shape sum (2,048 rows: 4.57 GiB against 4.11
GB).

Measured peaks of MLX's own bytes, imported weights left out and the state's rows in (the reading then counted what
was resident at the phase's start; it now counts only what the phase adds past it) (`SUSHI_TRANSIENT_PEAK=1`,
binary 26a28040, STQ-1.75bpw, `--ssd-budget-gb 108`, DSpark, one 225k-token cold prefill, raw
`p2b-1028`), and the cap beside them (the pool allowance, 1 GiB, is added on top):

| phase | rows | context | measured peak, GiB | cap, GB |
|---|---|---|---|---|
| decode | 1 | 9k / 225k | 0.116 / 0.336 | 0.40 / 0.53 |
| DSpark draft, verify | 5, <= 17 | 225k | 0.304, 0.327 | 0.53 |
| prefill chunk | 2048 | 2k / 225k | 4.57 / 4.77 | 5.4 / 5.5 |
| prefill chunk | 256 / 1000 / 1183 | short | 0.57 / 2.25 / 2.48 | 1.0 / 2.9 / 3.3 |
| vision encode | 8463 patches (1031 tokens) | | 1.11 | 1.47 (`visionEncodeBill`) |

A cap is over the level its phase starts from, which holds the state: each MLX cap covers the measured peak less the
state of the context (`dsv41 forward cap: on MLX` test); the 363-row chunk is the one whose rows and fixed bytes alone
fall short (by 74 MB, inside the margin). Extrapolated to 1M tokens: a decode or verify forward 1.03 GB, a 2048-row chunk 5.9 GB and a
tail-merged one (2559 rows) 7.1 GB. The decode slope the measurements give beyond the state's 890 B a token is 204 B; the
cap carries 603 B.

**What the bill is now** (`prefillNeededAtChunk`, the borrow's `prefillNeed`; the old bill was
`transientBytes x 5/4 + (8 GiB - 1 GiB)` pool growth for a prefill and `decodeTransientBytes(17) x 5/4` for a decode):

| term | context | old, GB | new, GB |
|---|---|---|---|
| borrow need, 2048 rows | 14.8k / 225k / 1M | 15.1 / 15.3 / 15.8 | 5.4 / 5.5 / 5.9 |
| chunk bill, 2048 + 511 rows | 14.8k / 225k / 1M | 16.4 / 16.6 / 17.1 | 6.7 / 6.8 / 7.1 |
| decode or verify block | 9k / 225k / 1M | 0.55 / 0.71 / 1.33 | 0.40 / 0.53 / 1.03 |
| image encode | any | unchanged: the bill already covered the measured peak by a quarter | |

Those are MLX's; on a Cooler load (the shape sum) a 2048-row chunk needs 4.35 / 4.44 / 4.78 GB at 2k / 225k / 1M.

On a Cooler load (an STQ pack, [engine-own-forward](engine-own-forward.md#memory)) the request
bill is the state and one forward's cap: the engine's pool is that cap (its free buffers are inside it), a prefill chunk
gives its cap whole to the pool (the state rows it appends grow there too), and MLX allocates nothing, so MLX's parked
decode pool (1 GiB on the MLX route) is not billed. The ledger's pool line of a Cooler load is what the engine parks
between requests, about 0.26 GB (`arch_config.expertPoolBytes`, the arch's own answer for the config).
- **The caps belong to a side** (`dsv41_borrow`): a request's end or a forward of a model on MLX moves MLX's caps
  only, a Cooler model's the engine's budget only, so one model's request end never takes the other's room
  (`after a model on MLX beside it served and ended a request, the Cooler's next request of any kind has the engine's room`).
- **A restore reserves the rows' slack** (`Rows.reserve`): the first step after a restore from a point someone keeps
  copied every layer's rows (1,441,792 bytes on the tiny fixture against 163,840 cold); it now takes 196,608 and does
  not grow with the context.
- **The engine's cap is the forward's own** (`dsv41_borrow.enterForward`, `enterRows`), whether or not `serve`
  installed MLX's: off serve (`sushi kld`, tests) a request's end left its rest cap (0.26 GB) in force, which refused a
  4096-row chunk of the tiny from the load's second request on, and the first ran under no cap at all.
- **A decode forward's fixed bytes** (`cap_fixed_bytes`, 379 MB) hold its small tensors; a wide forward's are in its
  shape sum, page by page, and on MLX's stream the router's zero pad of a short tile (42 MB at the release's width,
  [dsv41-arch](dsv41-arch.md#prefill-width)) sits in the MLX cap's fixed bytes.
- **A wide forward copies a compressor's open group out** (`compress`): the pending rows are the tail of the kv and gate
  joined to the rows before them, and a slice of it would keep the whole `[rows + r - 1, head_dim]` f32 pair in use
  until the next chunk (1 KB a row on the STQ tiny; 25 MB after a 2,048-row chunk of the release, three ratio-2 sources),
  past the state term (`stateFixedBytes`, `r - 1` rows). A decode or verify forward keeps the slice of its few rows: a
  copy there is an op a token (`between two 2,048-row chunks the pool holds no more than the request's billed state`).

The live headroom the borrow reads (`dsv41Headroom`) is the ceiling less `gpu_mem.activeBytes`, which holds the
engine's pool buffers bound to live tensors (the state rows, window rings and rollback record) beside MLX's active
bytes, and the weights and expert slabs, which a Cooler load's engine wraps itself; the ceiling's footprint adds
`gpu_mem.cacheBytes`, which holds the engine's free buffers. The headroom also
takes off the engine's free buffers (`engine.poolFree`): MLX makes what a borrow frees room for, and nothing on MLX's
side can reclaim a buffer the engine holds. The cap bills the pool's growth over its tensors in use; the live read
counts what it holds now. A tensor lent to MLX is in both counters, so the headroom errs low
([engine-memory-admission](engine-memory-admission.md#gpu-memory-readers)).

The pool growth (7.5 GB) was most of the old bill and the cap replaces it: a 2048-row chunk's need falls 8.3-8.7 GB,
the rest being the measured per-row rate against the old formula. At wired 124000 with DSpark and the SSD tier on (the
`with its decoder pools lendable` test), budget 117 and no decoder pool to lend, the advertised context is 581,632
tokens where it was under 262,144.

**Gates** (09977e23 vs main b66f36a8, budget 108, DSpark): greedy code, prose and image answers and their SSD-restore
follow-ups are byte-identical; a 225,674-token cold prefill runs 764.7 tok/s against 753.1 (AC, fans max, both started
under 50 °C, one run each); no `TransientCapExceeded` over that prefill, an image, or 445 chunks at `--prefill-chunk
512`. The peak diagnostic slows a cold prefill by itself (a diag run read 414 tok/s on the same prompt): never time
with it on.

**A larger budget slows prefill past 108.** The same 225,674-token cold prefill (09977e23, AC, fans max, each run
started under 50 °C, wired 125000): budget 108 764.7 tok/s, 111 709.1, 114 647.0. The bill admits all three with 1M
context, but at 111 and 114 a 2048-row chunk's compute (`other_ms`) ran 2.9x and 3.3x the 108 chunk's while its SSD
fill fell. The cause is macOS's `vm.global_user_wire_limit` (116.82 GB by default), which the startup `[mem] warning:`
checks ([engine-memory-admission](engine-memory-admission.md#metal-oom)). With both user-wire sysctls at 125 GB, 111
prefilled the 225k prompt at 759.0 tok/s; 114 still ran a cold 32,888-token prompt at 622.6 tok/s against 108's 792.7
(one run each), so 111 is the raised-limit launch.

<a id="vision-requant"></a>
## The vision tower, requantized

The pack keeps the tower bf16 (266 tensors, 0.971 GB); the load requantizes the 322 block and aligner matrices to MLX
affine 8-bit, group 128, and frees each bf16 copy as its codes exist ([dsv41-arch](dsv41-arch.md#image-input)); `--vision-bf16` skips it and the ledger bills the tower as stored (0.971 GB). The
ledger's `vision` term is what the load holds: 501,303,296 B (codes at a byte a weight, a bf16 scale and bias per group,
the 588-wide patch embedding, norms, biases and the 40 image biases bf16), from `dsv41_source.visionBytes`, which applies
the load's own rule (`dsv41_vision.quantizesAtLoad`) to the shard headers. The codes are allocations, not
sushi-owned mappings: the engine's pool buffers on a Sushi Cooler load, MLX's on the MLX route. `[dsv41-mem]` does not
count them and `active_bytes` does.

| STQ-1.75bpw, `--ssd-budget-gb` | vision term | slots/layer without tower | with tower | stored bf16 tower would give |
|---|---|---|---|---|
| 96 | 0.50 GB | 239 | 238 (-1) | 237 (-2) |
| 108 | 0.50 GB | 273 | 272 (-1) | 270 (-3) |

A slot a layer is 0.385 GB (the ledger's per-slot bytes). Measured on the real tower (`tower.safetensors` of the
vision fixture) with house.jpeg letterboxed to a 58 x 93 grid (642 span tokens), requantized against bf16, both in one
process (`SUSHI_VISION_Q8_REPORT=1`, the `requantized tower stays within bf16's drift` test; base 2db0e8d5 plus this
change, M5 Max, GPU lock held, QUIET, `taskpolicy -a`):

| stage | cosine | max abs | relative RMS |
|---|---|---|---|
| after 16 blocks | 0.99967 | 88 (a massive-activation channel) | 0.0257 |
| after 31 blocks | 0.99966 | 88 | 0.0260 |
| final norm | 0.99906 | 0.052 | 0.0434 |
| span (aligner out) | 0.99957 | 0.045 | 0.0295 |

- Load: 322 linears requantized in 49 ms (`[vision] tower requantized at load: bf16 0.97 GB -> affine8 g128 0.50 GB`).
  A linear's codes weigh exactly its bill (the aligner's `w1`: 48,660,480 B active for 48,660,480 B billed).
- Time, 642 tokens, median of 9 interleaved warm runs: bf16 tower 291.9 ms, q8 300.9 ms (+3%); with the aligner 294.0 and
  303.0 ms. A quantized matmul over 5,394 rows is compute-bound and the dequantizing kernel does not beat bf16's, so the
  gain is memory, not speed.
- The encode's transient peak is the same on both arms (1.198 GiB in that process, which also holds the codes); the quantized
  matmul reads the codes in its kernel and makes no dequantized copy, so `visionEncodeBill` and the phase cap stand.
- On a Sushi Cooler load the encode's scratch is the engine pool's, not MLX's: `enterVision` sets the pool budget to
  the encode's bill for the whole encode (each image's patches, its pass and the join of the spans), and each pass
  trims the pool when it ends, so the bill and the phase cap stand; the weights are MLX's buffers, imported, and the
  span the pass returns is the pool's until the trunk is done with it.

<a id="owned-weights"></a>
## Weight memory sushi owns

The weight load builds every long-lived weight of 1 MiB or more in a mapping of its own that MLX imports without a copy
(`model.ownedRegion` and `wrapOwned`); a Cooler load maps every weight, whatever its size (`model.importFloor`), since a
weight in the engine's pool sits under every forward's cap, which counts what tensors ask, not the size classes they
take. `/props` `memory.owned_bytes` and the `[dsv41-mem]` load line report it, the pool's share apart.
- `model.wrapOwned` takes the caller's region handle and nulls it before anything can fail, so a builder's
  `errdefer` over the handle never frees a region the wrap has released (a failed wrap of a stack crashed on it).
- A shard tensor is read into its mapping (`readTensorArray`); the weight load refuses one MLX copied instead
  (`OwnedImportCopied`), since a copy would sit outside the ledger.
- The DSpark stage's trellis experts are nine regions (trellis, `suh`, `svh` per projection), the groups' bytes back to
  back, copied on the host; each group's source is dropped after its copy, so a stage's load peak is its storage plus
  one group (`owned_weights.raggedOwned`, into `Ragged.initPools`).
- The DSpark stage's mxfp4 experts (the release's layout) are six regions (codes and scales per projection), every
  expert's bytes back to back, copied on the host (`owned_weights.stack`, `.experts`). An op would make the bank in the
  engine's pool on a Cooler load, where a forward's cap would count it (7.5 GB on the release pack).
- f32 widenings (bf16 to f32 is the bits shifted up, as MLX's cast), the fp8 block scales repeated down their 32 rows
  (`owned_weights.repeatRows`) and the fused-stack bank of `fuseShared` are built in regions; the slices stay views.
- Left outside the regions (MLX's, or the engine's pool on a Cooler load): tensors under the floor, the meta tables, the
  rope tables, Engram's q.k product (its q and k are dropped once it is made) and the vision tower's requantized codes
  ([above](#vision-requant)), billed at the pool's size classes on a Cooler load.
- The bill (`dsv41_source.sumBytes`, per side) rounds each tensor from the load's floor, and each DSpark (stage,
  projection, part) as one region (a bank under the floor is a stack's, at its stored bytes), up to the page. It holds
  a pooling compressor's weights and Engram's product in f32. A test holds the bill to what each side's load of the
  release layout holds (`the trunk bill is what the release layout's load holds`). MLX refuses no page-aligned mapping; it also accepts a start inside a page, so the
  `OwnedImportCopied` check is the data-pointer test, not an alignment rule.

## Plans that lost

Replayed on recorded routes, two plans were dropped. Pinning the encoder's experts resident (layers 0-19) saves prefill
fill but costs decode misses on the decoder layers, and decode reads all 40 layers: it loses once a turn decodes more
than about a fifth of its new prompt tokens. Swapping the encoder and decoder halves between phases moves ~29.5 GB a
flip (about 3.3 s at 9 GB/s) and loses at every decode length, with the swap in the foreground or the background.

## The 1M KV reserve

An auto boot serves 1,000,000 tokens (`ModelConfig.defaultContext`) whenever the margined memory context
(`safeAutoContext`, 93% of what the bill admits, so the memory search runs past the checkpoint's window to about
1.08M tokens) reaches it, and the checkpoint's window caps it. A box whose margined context falls short keeps that
margined value and logs one warning naming both numbers. A request that omits
`max_tokens` gets 256,000 (`defaultMaxTokens`); admission reserves 8192 of it, so the budget bills nothing more up front.
`--ctx-size` and `--max-tokens` outrank both, and `model-settings.json` has no `max_tokens` key.

With no `--ssd-budget-gb` and no `ssd_budget_gb` setting the load picks one from physical RAM (`ramDefaultBudget`): 80 GiB on 96 GB, 96 on 128 GB (108 with `iogpu.wired_limit_mb` at 120000 or more), resident on 192 and 256 GB. The step-down bound is the lower of the residency limit (the declared wired limit, else Metal's working set) and `vm.global_user_wire_limit` / `vm.user_wire_limit`, each less its margin, beside the 1M request bill (`dsv41DefaultRequestBill`) and the image encode. The `/v1/load-model` registry gate bills the same default (`ramDefaultFor`), not a budget of 0.

The state (0.93 GB at 1M) is billed per request in admission. The SSD budget is fixed at load, so the expert cache
never squeezes the state.

What squeezed a 1M request was the prefill transient's context term: the indexer's scores, span x context x f32,
billed three copies (3.9 KB a context token with the keys unpacked and the growth). The indexer now streams key tiles
into a running top-k (`indexStream`), so one span's peak is flat in the context (44-81 MB at the release shapes, 948 MB
whole at 1M). The context-scaled bill is 0.6 KB a token (the decode block's scores 192 B, growth slack 56 B, one
layer's old rows during a growth 356 B) on top of the state's 0.89 KB. The prefix cache's restore points no longer
hold RAM ([below](#borrow)), so a 1M request bills 3.4 GB.

<a id="borrow"></a>
## Admission bills the state: the prefill borrows the decoder pools, restore points go to SSD

Rules: a prompt's prefill working set must not limit the admitted context, and the prefix cache is SSD with RAM 0.

**What admission bills** (`server.prefillNeededAtChunk`, x 5/4): the request's state (`(seq + max_tokens) x 890 B` and
the rings), the restore points in flight to SSD (below), and a decode or verify forward's working set
(`Config.forwardCap(17, seq)`, the transient cap of a verify block; 1.06 GB at 1M). The prefill working set is its cap
(`Config.chunkCap`, [below](#transient-caps)), billed only where the decoder pools a prefill may still release
(`ModelConfig.expert_borrow_bytes`, published live by the engine) do not cover it. At 1M: 3.4 GB in all.

**The borrow** (`src/dsv41/dsv41_borrow.zig`, `Engine.borrowLayer`/`restoreLayer`):
- Before every forward wider than 17 rows, its need, `Config.forwardCap(rows, seq)` (4.35 GB at 2048 rows on the STQ pack),
  against the live headroom (ceiling less MLX's active bytes). Short of it, whole decoder pools go, latest layer first
  (39 down to the CED split, 20): the plan is made from the pools' bytes before anything is released, a forward the
  pools cannot cover is refused (`PrefillDoesNotFit`) with nothing released. The planned pools go, the forward waits for
  the kernel to unwire them, and only then is the live headroom re-read: freed pages count in it once unwired, so a
  reading inside the release loop stays flat and released every pool (issue 105). Still short, one more pool goes and
  is waited for (`releaseUntil`).
- A release drops the layer's bank, waits for the slabs' holder to let go of each (MLX, or on a Cooler load the
  engine, whose own wait is enough) and unmaps it: the active count falls by exactly the pool's bytes (tested), and the forward waits up to 1 s for 90% of them to leave the wired set.
- Under CED a decoder layer runs only the chunk's last 128 rows, so a released layer reads that chunk's picks (~130
  experts of ~8.1 MB, ~1.06 GB) into the union workspace, which stays resident, and the union bank serves them.
  Same bytes out as with a pool (tested on the tiny STQ pack, prefill and decode). Against a pool's ~30% misses that
  is about +0.075 s a borrowed layer a chunk at ~10 GB/s.
- The first decode or verify forward reads each released pool back with the experts it held, while each fits beside
  that forward's working set; one that does not stays released (union path) and is retried at the next forward.
  Eager, not on demand: the bytes are the same order (a cold layer refills its ~200-300 decode experts by misses
  either way) but as one ~9 GB/s read before the first decode step (~0.3 s a layer) instead of per-token stalls,
  which would also feed DSpark's measured round cost (a cold cache once drove it off wide verifies). Against a 2k-20k
  token decode at ~65 ms a token that is 0.02-0.2% a layer.
- A release or a restore that fails leaves the layer as it was, and the borrow stops at the first failure: a released
  layer always has the union workspace, so a release that cannot make it takes its pool back. One whose take-back
  failed too is healed before the next forward (workspace, else pool), or that forward is `ReleasedLayerHasNoUnion`.
- Log lines: `[dsv41-borrow] released layers a..39 (X GB) for a N-row forward at P: need, headroom before -> after,
  unwired in T ms` and `[dsv41-borrow] restored n layers (X GB) in T ms before decode`.

Why it never over-commits: admission proved `state + max(decode, prefill - lendable) <= available` against live
memory, and the request runs alone (`exclusive`). During the prefill each forward runs only once `headroom >= need`,
counting the bytes just released; its transient is freed before the next, so the state it promised later still fits.
A restore gives back exactly the bytes released, so the decode finds what admission promised it.

**Restore points** (`dsv41_prefix.captureShared`, `HotPrefixCache.writeThroughDsv41`): SSD-first, a point at or past the
tier's 512-token floor shares the request's compressed rows and index keys (views, no copy), and the prefill
write-through (armed for DeepSeek whatever the new span) stages it into the writer's bounded host buffers and releases
it before the next forward appends, the newest included; the prompt-end snapshot is staged before the final span. The
admission bill is the writer's permit (or one point, when smaller) and a point's fixed rings, where it was up to five
full copies (4.7 GB at 1M, 5.9 with the quarter). A RAM tier keeps the copies and the old bill. A write the SSD cannot
keep up with holds the prefill on the permit: `[dsv41-prefix] restore points to SSD at P: X GB staged, prefill held T ms`.
Restores are bit-identical (tested: a point staged mid-prefill restores the same bytes from a fresh tier and continues
like cold).

**What it unlocks** (wired 124000, DSpark, SSD tier on; the budget-100 boot's 122,170 MiB ceiling and 100,328 MiB active,
1 GiB more active a GiB of budget; lendable = half the cache; bill arithmetic, no boot):

| `--ssd-budget-gb` | pools held (profile split) | of 119.6 GB | memory context, borrow | without the borrow |
|---|---|---|---|---|
| 113 | 106.7 GB | 89.2% | 1M | 1M |
| 114 | 107.9 GB | 90.2% | 1M | 997,638 |
| 116 | 110.2 GB | 92.1% | 1M | 276,703 |
| 117 | 111.4 GB | 93.1% | 1M (advertised 974,848) | 924 |
| 118 | 112.1 GB | 93.8% | 1,011,346 | 505 |

Pools held are the profile split after each pool's floor rounding (0.65-0.75 GB under the billed cache), mirrored from
`profileAllot` and `sizePools` on the pack's headers; they match the boots' slot counts and active bytes.

- With the restore points on SSD alone, budget 113 already serves 1M; the borrow takes the working set from there.
- At 113 a 2048-row chunk borrows nothing until ~1M (one layer there). At 117 the headroom is ~4.6 GB, so every chunk
  of a prompt past a few thousand tokens borrows two layers: ~+0.15 s a chunk and ~0.6 s of refill a turn. Decode gains
  little for it: the simulated misses at 108-112 GB of cache are ~0.4-0.6 a token (~0.3% of a token's time).

- **A 32 GB Mac** (no RAM tier: `--ssd-budget-gb` is required) serves the STQ 1.75bpw pack at 18: trunk 5.20, MTP 3.85,
  vision 0.50, workspace 3.70, bounce 0.54, pool 0.27, kept scales 0.68 GB, an expert cache of 4.23 GB (11 slots a layer).

<a id="resident"></a>
## Fully resident

A budget that holds every routed expert beside the trunk and DSpark loads the pack resident
([engine-expert-streaming](engine-expert-streaming.md#budget)): no union workspace (3.70 GB), selected slab or bounce,
every expert read at load, and each layer's picks mapped to slots on the GPU, so a forward never waits on a host read
of the picks. The borrow stays: a prefill past the headroom releases decoder pools, its first release paying for a
union workspace (3.54 GB, the widest decoder layer) that the last restore drops.

The 1M bill with the prefill borrowed is 3.4 GB (state 0.93, writer in flight 0.93, rings, x5/4, and the decode cap 1.06).
`--drop-expert` ([dsv41-arch](dsv41-arch.md#drop-expert)) frees bytes from the experts to make this fit: on an STQ pack
whose experts total 119.6 GB, dropping 8 per layer frees 2.1-2.5 GB and dropping 16 frees 4.3-5.0 GB.

- The near-resident arm with DSpark is the streamed path at the largest budget whose 1M bill fits (94% of the expert
  bytes cached). It keeps the 3.70 GB union and reads its misses from SSD; the lossy pick (`--expert-pick-tolerance`)
  swaps those misses on grouped layers.
- Every arm wires more than `vm.global_user_wire_limit` (116.82 GB); a decode measured at half speed past it once.
